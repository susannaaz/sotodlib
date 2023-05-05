import os
import sys
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import yaml
import sotodlib.io.g3tsmurf_utils as g3u
import sotodlib.io.load_smurf as ls
from functools import partial
from sotodlib.core import AxisManager, metadata
from sotodlib.io.metadata import read_dataset, write_dataset
from sotodlib.site_pipeline import util
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment
from pycpd import AffineRegistration
from detmap.makemap import MapMaker

logger = util.init_logger(__name__, "make_position_match: ")


def create_db(filename):
    """
    Create db for storing results if it doesn't already exist

    Arguments:

        filename: Path where database should be made.
    """
    if os.path.isfile(filename):
        return
    if not os.path.isdir(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    scheme = metadata.ManifestScheme()
    scheme.add_exact_match("obs:obs_id")
    scheme.add_data_field("dataset")
    scheme.add_data_field("input_paths")

    metadata.ManifestDb(scheme=scheme).to_file(filename)


def LAT_coord_transform(xy, rot_fp, rot_ufm, r=72.645):
    """
    Transform from instrument model coords to LAT Zemax coords

    Arguments:

        xy: XY coords from instrument model.
            Should be a (2, n) array.

        rot_fp: Angle of array location on focal plane in deg.

        rot_ufm: Rotatation of UFM about its center.

    Returns:

        xy_trans: Transformed coords.
    """
    xy_trans = np.zeros((xy.shape[1], 3))
    xy_trans[:, :2] = xy.T

    r1 = R.from_euler("z", rot_fp, degrees=True)
    shift = r1.apply(np.array([r, 0, 0]))

    r2 = R.from_euler("z", rot_ufm, degrees=True)
    xy_trans = r2.apply(xy_trans) + shift

    return xy_trans.T[:2]


def rescale(xy):
    """
    Rescale pointing or template to [0, 1]

    Arguments:

        xy: Pointing or template, should have two columns.

    Returns:

        xy_rs: Rescaled array.
    """
    xy_rs = xy.copy()
    xy_rs[:, 0] /= xy[:, 0].max() - xy[:, 0].min()
    xy_rs[:, 0] -= xy_rs[:, 0].min()
    xy_rs[:, 1] /= xy[:, 1].max() - xy[:, 1].min()
    xy_rs[:, 1] -= xy_rs[:, 1].min()
    return xy_rs


def priors_from_result(
    fp_readout_ids,
    template_det_ids,
    final_fp_readout_ids,
    final_template_det_ids,
    likelihoods,
    normalization=0.2,
):
    """
    Generate priors from a previous run of the template matching.

    Arguments:

        fp_readout_ids: Array of readout_ids in the basis of the focal plane that was already matched.

        template_det_ids: Array of det_ids in the basis of the template that was already matched.

        final_fp_readout_ids: Array of readout_ids in the basis of the focal plane that will be matched.

        final_template_det_ids: Array of det_ids in the basis of the template that will be matched.

        likelihoods: Liklihood array from template matching.

        normalization: Value to normalize likelihoods to. The maximum prior will be 1+normalization.

    Returns:

        priors: The 2d array of priors in the basis of the focal plane and template that are to be matched.
    """
    likelihoods *= normalization
    priors = 1 + likelihoods

    missing = np.setdiff1d(final_template_det_ids, template_det_ids)
    template_det_ids = np.concatenate(missing)
    priors = np.concatenate((priors, np.ones((len(missing), len(fp_readout_ids)))))
    asort = np.argsort(template_det_ids)
    template_map = np.argsort(np.argsort(final_template_det_ids))
    priors = priors[asort][template_map]

    missing = np.setdiff1d(final_fp_readout_ids, fp_readout_ids)
    fp_readout_ids = np.concatenate(missing)
    priors = np.concatenate((priors.T, np.ones((len(missing), len(template_det_ids)))))
    asort = np.argsort(fp_readout_ids)
    fp_map = np.argsort(np.argsort(final_fp_readout_ids))
    priors = priors[asort][fp_map].T

    return priors


def gen_priors(aman, template_det_ids, prior, method="flat", width=1, basis=None):
    """
    Generate priors from detmap.

    Arguments:

        aman: AxisManager assumed to contain aman.det_info.det_id and aman.det_info.wafer.

        template_det_ids: Array of det_ids in the same order as the template.

        prior: Prior value at locations from the detmap.
               Should be greater than 1.

        method: What sort of priors to implement.
                Currently only 'flat' and 'gaussian' are accepted.

        width: Width of priors. For gaussian priors this is sigma.

        basis: Basis to calculate width in.
               Nominally will load values from aman.det_info.wafer.
               Pass in None to use the indices as the basis.

    Returns:

        priors: The 2d array of priors.
    """

    def _flat(x_axis, idx):
        arr = np.ones_like(x_axis)
        lower_bound = x_axis[idx] - width // 2
        upper_bound = x_axis[idx] + width // 2 + width % 2
        prior_range = np.where((x_axis >= lower_bound) & (x_axis < upper_bound))[0]
        arr[prior_range] = prior
        return arr

    def _gaussian(x_axis, idx):
        arr = 1 + (prior - 1) * np.exp(
            -0.5 * ((x_axis - x_axis[idx]) ** 2) / (width ** 2)
        )
        return arr

    if method == "flat":
        prior_method = _flat
    elif method == "gaussian":
        prior_method = _gaussian
    else:
        raise ValueError("Method " + method + " not implemented")

    if basis is None:
        x_axis = np.arange(aman.dets.count)
    else:
        x_axis = aman.det_info.wafer[basis]

    priors = np.ones((aman.dets.count, len(template_det_ids)))
    # TODO: Could probably vectorize this
    for i in range(aman.dets.count):
        _prior = prior_method(x_axis, i)
        _, msk, template_msk = np.intersect1d(
            aman.det_info.det_id, template_det_ids, return_indices=True
        )
        priors[i, template_msk] = _prior[msk]

    return priors.T


def transform_from_detmap(aman):
    """
    Do an approximate transformation of the pointing back to
    the focal plane using the mapping from the loaded detmap.

    Arguments:

        aman: AxisManager containing both pointing and datmap results.
    """
    xi_0 = np.nanmedian(aman.xi)
    eta_0 = np.nanmedian(aman.eta)
    msk_1 = (aman.xi > xi_0) & (aman.eta > eta_0)
    p1_fp = np.array(
        (
            np.nanmedian(aman.xi[msk_1]),
            np.nanmedian(aman.eta[msk_1]),
            1,
        )
    )
    msk_2 = (aman.xi < xi_0) & (aman.eta > eta_0)
    p2_fp = np.array(
        (
            np.nanmedian(aman.xi[msk_2]),
            np.nanmedian(aman.eta[msk_2]),
            1,
        )
    )
    msk_3 = aman.eta < eta_0
    p3_fp = np.array(
        (
            np.nanmedian(aman.xi[msk_3]),
            np.nanmedian(aman.eta[msk_3]),
            1,
        )
    )
    X = np.transpose(np.matrix([p1_fp, p2_fp, p3_fp]))

    p1_dm = np.array(
        (
            np.nanmedian(aman.det_info.wafer.det_x[msk_1]),
            np.nanmedian(aman.det_info.wafer.det_y[msk_1]),
            1,
        )
    )
    p2_dm = np.array(
        (
            np.nanmedian(aman.det_info.wafer.det_x[msk_2]),
            np.nanmedian(aman.det_info.wafer.det_y[msk_2]),
            1,
        )
    )
    p3_dm = np.array(
        (
            np.nanmedian(aman.det_info.wafer.det_x[msk_3]),
            np.nanmedian(aman.det_info.wafer.det_y[msk_3]),
            1,
        )
    )
    Y = np.transpose(np.matrix([p1_dm, p2_dm, p3_dm]))

    A2 = Y * X.I

    coords = np.vstack((aman.xi, aman.eta)).T
    transformed = [
        (A2 * np.vstack((np.matrix(pt).reshape(2, 1), 1)))[0:2, :] for pt in coords
    ]
    transformed = np.reshape(transformed, coords.shape)

    aman.xi = transformed[:, 0]
    aman.eta = transformed[:, 1]


def visualize(frame, frames, ax, bias_lines):
    """
    Visualize CPD matching process.
    Modified from the pycpd example scripts.

    Arguments:

        frame: The frame to display.

        frames: List of frames, each frame should be [iteration, error, X, Y]

        ax: Axis to use for plots.

        bias_lines: True if bias lines are included in points.
    """
    iteration, error, X, Y = frames[frame]
    cmap = "Set3"
    if bias_lines:
        x = 1
        y = 2
        c_t = np.around(np.abs(X[:, 0])) / 11.0
        c_s = np.around(np.abs(Y[:, 0])) / 11.0
        srt = np.lexsort(X.T[1:3])
    else:
        x = 0
        y = 1
        c_t = np.zeros(len(X))
        c_s = np.ones(len(Y))
        srt = np.lexsort(X.T[0:2])
    plt.cla()
    ax.scatter(
        X[:, x][srt[0::4]],
        X[:, y][srt[0::4]],
        c=c_t[srt[0::4]],
        cmap=cmap,
        alpha=0.5,
        marker=4,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        X[:, x][srt[1::4]],
        X[:, y][srt[1::4]],
        c=c_t[srt[1::4]],
        cmap=cmap,
        alpha=0.5,
        marker=5,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        X[:, x][srt[2::4]],
        X[:, y][srt[2::4]],
        c=c_t[srt[2::4]],
        cmap=cmap,
        alpha=0.5,
        marker=6,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        X[:, x][srt[3::4]],
        X[:, y][srt[3::4]],
        c=c_t[srt[3::4]],
        cmap=cmap,
        alpha=0.5,
        marker=7,
        vmin=0,
        vmax=1,
    )
    ax.scatter(
        Y[:, x], Y[:, y], c=c_s, cmap=cmap, alpha=0.5, marker="X", vmin=0, vmax=1
    )
    plt.text(
        0.87,
        0.92,
        "Iteration: {:d}\nQ: {:06.4f}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="x-large",
    )


def match_template(
    focal_plane,
    template,
    priors=None,
    out_thresh=0,
    bias_lines=True,
    reverse=False,
    vis=False,
    cpd_args={},
):
    """
    Match fit focal plane againts a template.

    Arguments:

        focal_plane: Measured pointing and optionally polarization angle.
                     Should be a (2, n) or (3, n) array with columns: xi, eta, pol.
                     Optionally an optics model can be preapplied to this to map the pointing
                     onto the physical focal plane in which case the columns are: x, y, pol.

        template: Designed x, y, and polarization angle of each detector.
                  Should be a (2, n) or (3, n) array with columns: x, y, pol.

        priors: Priors to apply when matching.
                Should be be a n by n array where n is the number of points.
                The priors[i, j] is the prior on the i'th point in template matching the j'th point in focal_plane.

        out_thresh: Threshold at which points will be considered outliers.
                    Should be in range [0, 1) and is checked against the
                    probability that a point matches its mapped point in the template.

        bias_lines: Include bias lines in matching.

        reverse: Reverse direction of match.

        vis: If true generate plots to watch the matching process.
             Should only be used for debugging with human interaction.

        cpd_args: Dictionairy containing kwargs to be passed into AffineRegistration.
                  See the pycpd docs for what these can be.
    Returns:

        mapping: Mapping between elements in template and focal_plane.
                 focal_plane[i] = template[mapping[i]]

        outliers: Indices of points that are outliers.
                  Note that this is in the basis of mapping and focal_plane, not template.

        P: The likelihood array without priors applied.

        TY: The transformed points.
    """
    if not bias_lines:
        focal_plane = focal_plane[:, 1:]
        template = template[:, 1:]
    if reverse:
        cpd_args.update({"X": focal_plane, "Y": template})
    else:
        cpd_args.update({"Y": focal_plane, "X": template})
    reg = AffineRegistration(**cpd_args)

    if vis:
        frames = []

        def store_frames(frames, iteration, error, X, Y):
            frames += [[iteration, error, X, Y]]

        fig = plt.figure()
        fig.add_axes([0, 0, 1, 1])
        callback = partial(store_frames, frames=frames)
        reg.register(callback)
        anim = ani.FuncAnimation(
            fig=fig,
            func=partial(
                visualize, frames=frames, ax=fig.axes[0], bias_lines=bias_lines
            ),
            frames=len(frames),
            interval=200,
        )
        plt.show()
    else:
        reg.register()

    P = reg.P.T
    if reverse:
        P = reg.P
    TY = reg.TY
    if not bias_lines:
        TY = np.column_stack((np.zeros(len(TY)), TY))

    if priors is None:
        priors = 1

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(P * priors, True)
    if len(row_ind) < len(focal_plane):
        mapping = np.argmax(P * priors, axis=0)
        mapping[col_ind] = row_ind
    else:
        mapping = row_ind[np.argsort(col_ind)]

    outliers = np.array([])
    if out_thresh > 0:
        outliers = np.where(P[mapping, range(P.shape[1])] < out_thresh)[0]

    return mapping, outliers, P, TY


def main():
    # Read in input pars
    parser = ap.ArgumentParser()

    # NOTE: Eventually all of this should just be metadata I can load from a single context?

    # Making some assumtions about pointing data that aren't currently true:
    # 1. I am assuming that the HDF5 file is a ResultSet
    # 2. I am assuming it contains aman.det_info
    parser.add_argument("config_path", help="Location of the config file")
    args = parser.parse_args()

    # Open config file
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    pointing_paths = np.atleast_1d(config["pointing_data"])
    if "polangs" in config:
        polangs_paths = np.atleast_1d(config["polangs"])
        polangs = []
    else:
        polangs_paths = np.zeros(len(pointing_paths), dtype=bool)

    # Figure out the tuneset
    SMURF = ls.G3tSmurf(**config["g3tsmurf"]["paths"])
    ses = SMURF.Session()
    obs = (
        ses.query(ls.Observations)
        .filter(ls.Observations.obs_id == config["g3tsmurf"]["obs_id"])
        .one()
    )
    tunefile = obs.tunesets[0].path

    # Load data
    pointings = []
    polangs = []
    for point_path, pol_path in zip(pointing_paths, polangs_paths):
        logger.info("Loading pointing from: " + point_path)
        aman = AxisManager.load(point_path)
        g3u.add_detmap_info(aman, config["detmap"], columns="all")
        pointings.append(aman)

        if not pol_path:
            logger.warning("No polang associated with this pointing")
            polangs.append(False)
            continue
        # NOTE: Assuming some standin structure for the pol data
        # This may change in the future
        logger.info("Loading polangs from: " + pol_path)
        rset = read_dataset(pol_path, "polarization_angle")
        pol_rid = rset["dets:readout_id"]
        pols = rset["polarization_angle"]
        rid_map = np.argsort(np.argsort(aman.det_info.readout_id))
        rid_sort = np.argsort(pol_rid)
        pols = pols[rid_sort][rid_map]
        polangs.append(pols)
    bg_map = np.load(config["bias_map"], allow_pickle=True).item()

    # Build output path
    ufm = config["ufm"]
    append = ""
    if "append" in config:
        append = "_" + config["append"]
    if len(pointings) == 1:
        create_db(config["manifest_db"])
        db = metadata.ManifestDb(config["manifest_db"])
        outpath = os.path.join(
            config["outdir"], f"{ufm}_{obs.obs_id}{append}.h5"
        )
    else:
        outpath = os.path.join(
            config["outdir"], f"{ufm}_{obs.tunesets[0].id}{append}.h5"
        )
    dataset = "focal_plane"
    input_paths = "input_data_paths"
    outpath = os.path.abspath(outpath)

    # Make ResultSet of input paths for later reference
    types = ["config", "tunefile", "bgmap", "results"]
    paths = [args.config_path, tunefile, config["bias_map"], outpath]
    types += ["pointing"] * len(pointing_paths)
    paths += list(pointing_paths)
    if "polangs" in config:
        types += ["polang"] * len(polangs)
        paths += list(pointing_paths)
    paths = [os.path.abspath(p) for p in paths]
    rset_paths = metadata.ResultSet(
        keys=["type", "path"],
        src=np.vstack((types, paths)).T,
    )

    valid_bg = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

    # If requested generate a template for the UFM with the instrument model
    gen_template = config["gen_template"]
    if gen_template:
        logger.info("Generating template for " + ufm)
        wafer = MapMaker(north_is_highband=False, array_name=ufm)
        det_x = []
        det_y = []
        polang = []
        det_ids = []
        template_bg = []
        is_north = []
        for det in wafer.grab_metadata():
            if not det.is_optical:
                continue
            det_x.append(det.det_x)
            det_y.append(det.det_y)
            polang.append(det.angle_actual_deg)
            det_ids.append(det.detector_id)
            template_bg.append(det.bias_line)
            is_north.append(det.is_north)
        det_ids = np.array(det_ids)
        template_bg = np.array(template_bg)
        template_msk = np.isin(template_bg, valid_bg)
        template_n = np.array(is_north) & template_msk
        template_s = ~np.array(is_north) & template_msk
        template_bg[template_bg % 2 == 1] *= -1
        template = np.column_stack(
            (template_bg, np.array(det_x), np.array(det_y), np.array(polang))
        )

    reverse = config["matching"].get("reverse", False)
    if reverse:
        logger.warning(
            "Matching running in reverse mode. meas_x and meas_y will actually be fit pointing."
        )
    make_priors = "priors" in config
    priors_n = None
    priors_s = None
    avg_fp = {}
    master_template = []
    results = [[], [], []]
    for i, (aman, pol) in enumerate(zip(pointings, polangs)):
        logger.info("Starting match number " + str(i))
        # Do a radial cut
        r = np.sqrt(
            (aman.xi - np.median(aman.xi)) ** 2 + (aman.eta - np.median(aman.eta)) ** 2
        )
        r_msk = r < config["radial_thresh"] * np.median(r)
        aman = aman.restrict("dets", aman.dets.vals[r_msk])
        logger.info("\tCut " + str(np.sum(~r_msk)) + " detectors with bad pointing")

        if config["dm_transform"]:
            logger.info("\tApplying transformation from detmap")
            original = aman.copy()
            transform_from_detmap(aman)

        bias_group = np.zeros(aman.dets.count) - 1
        for i in range(aman.dets.count):
            msk = np.all(
                [
                    aman.det_info.smurf.band[i] == bg_map["bands"],
                    aman.det_info.smurf.channel[i] == bg_map["channels"],
                ],
                axis=0,
            )
            bias_group[i] = bg_map["bgmap"][msk][0]

        msk_bp = np.isin(bias_group, valid_bg)

        bl_diff = np.sum(~(bias_group == aman.det_info.wafer.bias_line)) - np.sum(
            ~(msk_bp)
        )
        logger.info(
            "\t"
            + str(bl_diff)
            + " detectors have bias lines that don't match the detmap"
        )

        north = np.isin(aman.det_info.smurf.band.astype(int), (0, 1, 2, 3))
        msk_n = north & msk_bp
        msk_s = (~north) & msk_bp
        bias_group[bias_group % 2 == 1] *= -1

        # Prep inputs
        dm_msk = slice(None)
        if not gen_template:
            dm_msk = (
                np.isfinite(aman.det_info.wafer.det_x)
                | np.isfinite(aman.det_info.wafer.det_y)
                | np.isfinite(aman.det_info.wafer.angle)
            )
            dm_aman = aman.restrict("dets", aman.dets.vals[dm_msk], False)

            template_msk = np.isin(dm_aman.det_info.wafer.bias_line, valid_bg)
            bias_line = dm_aman.det_info.wafer.bias_line
            bias_line[bias_line % 2 == 1] *= -1

            template = np.column_stack(
                (
                    bias_line,
                    dm_aman.det_info.wafer.det_x,
                    dm_aman.det_info.wafer.det_y,
                    dm_aman.det_info.wafer.angle,
                )
            )
            master_template.append(np.column_stack((template, dm_aman.det_info.det_id)))
            det_ids = dm_aman.det_info.det_id
            is_north = dm_aman.det_info.wafer.is_north == "True"
            template_n = is_north & template_msk
            template_s = ~(is_north) & template_msk

            if np.sum(template_n | template_n) < np.sum(msk_n | msk_s):
                logger.warning(
                    "\tTemplate is smaller than input pointing, uniqueness of mapping is no longer gauranteed"
                )

        if make_priors:
            priors = gen_priors(
                aman,
                det_ids,
                config["priors"]["val"],
                config["priors"]["method"],
                config["priors"]["width"],
                config["priors"]["basis"],
            )
            priors_n = priors[np.ix_(template_n, msk_n)]
            priors_s = priors[np.ix_(template_s, msk_s)]

        if pol:
            focal_plane = np.column_stack((bias_group, aman.xi, aman.eta, pol[r_msk]))
            original_focal_plane = np.column_stack(
                (original.xi, original.eta, pol[r_msk])
            )
            _focal_plane = focal_plane
            ndim = 3
        else:
            focal_plane = np.column_stack(
                (bias_group, aman.xi, aman.eta, np.zeros_like(aman.eta) + np.nan)
            )
            original_focal_plane = np.column_stack(
                (original.xi, original.eta, np.zeros_like(original.eta) + np.nan)
            )
            _focal_plane = focal_plane[:, :-1]
            template = template[:, :-1]
            ndim = 2

        # Do actual matching
        map_n, out_n, P_n, TY_n = match_template(
            _focal_plane[msk_n],
            template[template_n],
            priors=priors_n,
            **config["matching"],
        )
        map_s, out_s, P_s, TY_s = match_template(
            _focal_plane[msk_s],
            template[template_s],
            priors=priors_s,
            **config["matching"],
        )
        P_avg = (
            np.median(P_n[map_n, range(P_n.shape[1])])
            + np.median(P_s[map_s, range(P_s.shape[1])])
        ) / 2
        logger.info("\tAverage matched likelihood = " + str(P_avg))

        # Store outputs for now
        results[0].append(aman.det_info.readout_id)
        results[1].append(det_ids)
        P = np.zeros((len(template), len(focal_plane)), dtype=bool)
        P[np.ix_(template_n, msk_n)] = P_n
        P[np.ix_(template_s, msk_s)] = P_s
        results[2].append(P)

        out_msk = np.zeros(aman.dets.count, dtype=bool)
        out_msk[np.flatnonzero(msk_n)[out_n]] = True
        out_msk[np.flatnonzero(msk_s)[out_s]] = True
        out_msk[~(msk_n | msk_s)] = True

        ns_msk = np.zeros(aman.dets.count)
        ns_msk[msk_n] = 1
        focal_plane = np.column_stack(
            (
                aman.det_info.smurf.band,
                aman.det_info.smurf.channel,
                original_focal_plane,
                focal_plane,
                ns_msk,
            )
        )
        focal_plane[out_msk, 2:] = np.nan
        for ri, fp in zip(aman.det_info.readout_id, focal_plane):
            try:
                avg_fp[ri].append(fp)
            except KeyError:
                avg_fp[ri] = [fp]

    out_dt = np.dtype(
        [
            ("dets:readout_id", aman.det_info.readout_id.dtype),
            ("matched_det_id", det_ids.dtype),
            ("band", int),
            ("channel", int),
            ("xi", np.float32),
            ("eta", np.float32),
            ("polang", np.float32),
            ("meas_x", np.float32),
            ("meas_y", np.float32),
            ("meas_pol", np.float32),
            ("likelihood", np.float16),
            ("outliers", bool),
        ]
    )

    # It we only have a single dataset
    if len(pointing_paths) == 1:
        det_id = np.zeros(aman.dets.count, dtype=det_ids.dtype)
        det_id[msk_n] = det_ids[template_n][map_n]
        det_id[msk_s] = det_ids[template_s][map_s]

        logger.info(str(np.sum(msk_n | msk_s)) + " detectors matched")
        logger.info(str(np.unique(det_id).shape[0]) + " unique matches")
        logger.info(str(np.sum(det_id == aman.det_info.det_id)) + " match with detmap")

        transformed = np.nan + np.zeros((aman.dets.count, 3))
        transformed[msk_n, :ndim] = TY_n[:, 1:]
        transformed[msk_s, :ndim] = TY_s[:, 1:]

        P_mapped = np.zeros(aman.dets.count)
        P_mapped[msk_n] = P_n[map_n, range(P_n.shape[1])]
        P_mapped[msk_s] = P_s[map_s, range(P_s.shape[1])]

        data_out = np.fromiter(
            zip(
                aman.det_info.readout_id,
                det_id,
                *focal_plane.T[:5],
                *transformed.T,
                P_mapped,
                out_msk,
            ),
            out_dt,
            count=len(det_id),
        )
        rset_data = metadata.ResultSet.from_friend(data_out)
        write_dataset(rset_data, outpath, dataset, overwrite=True)
        write_dataset(rset_paths, outpath, input_paths, overwrite=True)
        db.add_entry(
            {"obs:obs_id": obs.obs_id, "dataset": dataset, "input_paths": input_paths},
            outpath,
            replace=True,
        )
        sys.exit()

    if not gen_template:
        template = np.unique(np.vstack(master_template), axis=0)
        det_ids = template[:, -1]
        template = template[:, :-1].astype(float)

    # Compute the average focal plane while ignoring outliers
    focal_plane = []
    readout_ids = np.array(list(avg_fp.keys()))
    for rid in readout_ids:
        avg_pointing = np.nanmedian(np.vstack(avg_fp[rid]), axis=0)
        focal_plane.append(avg_pointing)
    focal_plane = np.column_stack(focal_plane)
    msk_n = focal_plane[-1].astype(bool)
    msk_s = ~msk_n
    bc_avg_pointing = focal_plane[:5]
    focal_plane = focal_plane[5:9].T
    ndim = 3
    if np.isnan(focal_plane[:, -1]).all():
        focal_plane = focal_plane[:, :-1]
        template = template[:, :-1]
        ndim = 2

    # Build priors from previous results
    priors = 1
    for fp_readout_id, template_det_id, P in zip(*results):
        priors *= priors_from_result(
            fp_readout_id,
            template_det_id,
            readout_ids,
            det_ids,
            P,
            config["prior_normalization"],
        )

    # Do final matching
    map_n, out_n, P_n, TY_n = match_template(
        focal_plane[msk_n],
        template[template_n],
        priors=priors[np.ix_(template_n, msk_n)],
        **config["matching"],
    )
    map_s, out_s, P_n, TY_s = match_template(
        focal_plane[msk_s],
        template[template_s],
        priors=priors[np.ix_(template_s, msk_s)],
        **config["matching"],
    )

    # Make final outputs and save
    transformed = np.nan + np.zeros((len(readout_ids), 3))
    transformed[msk_n, :ndim] = TY_n[:, 1:]
    transformed[msk_s, :ndim] = TY_s[:, 1:]

    det_id = np.zeros(len(readout_ids), dtype=np.dtype(("U", len(det_ids[0]))))
    det_id[msk_n] = det_ids[template_n][map_n]
    det_id[msk_s] = det_ids[template_s][map_s]

    out_msk = np.zeros(len(readout_ids), dtype=bool)
    out_msk[np.flatnonzero(msk_n)[out_n]] = True
    out_msk[np.flatnonzero(msk_s)[out_s]] = True
    out_msk[~(msk_n | msk_s)] = True

    P_mapped = np.zeros(len(readout_ids))
    P_mapped[msk_n] = P_n[map_n, range(P_n.shape[1])]
    P_mapped[msk_s] = P_s[map_s, range(P_s.shape[1])]

    logger.info(str(np.sum(msk_n | msk_s)) + " detectors matched")
    logger.info(str(np.unique(det_id).shape[0]) + " unique matches")

    data_out = np.fromiter(
        zip(readout_ids, det_id, *bc_avg_pointing, *transformed.T, P_mapped, out_msk),
        out_dt,
        count=len(det_id),
    )
    rset_data = metadata.ResultSet.from_friend(data_out)
    write_dataset(rset_data, outpath, dataset, overwrite=True)
    write_dataset(rset_paths, outpath, input_paths, overwrite=True)


if __name__ == "__main__":
    main()
