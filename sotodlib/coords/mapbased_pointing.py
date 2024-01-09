import os
from tqdm import tqdm
import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

from sotodlib import coords

from sotodlib.core import metadata
from sotodlib.io.metadata import write_dataset

from so3g.proj import quat
from pixell import enmap
import h5py
from scipy.ndimage.filters import maximum_filter

def get_planet_trajectry(tod, planet, _split=20, return_model=False):
    timestamps_sparse = np.linspace(tod.timestamps[0], tod.timestamps[-1], _split)
    
    planet_az_sparse = np.zeros_like(timestamps_sparse)
    planet_el_sparse = np.zeros_like(timestamps_sparse)
    for i, timestamp in enumerate(timestamps_sparse):
        az, el, _ = coords.planets.get_source_azel(planet, timestamp)
        planet_az_sparse[i] = az
        planet_el_sparse[i] = el
    planet_az_func = interpolate.interp1d(timestamps_sparse, planet_az_sparse, kind="quadratic", fill_value='extrapolate')
    planet_el_func = interpolate.interp1d(timestamps_sparse, planet_el_sparse, kind="quadratic", fill_value='extrapolate')
    if return_model:
        return planet_az_func, planet_el_func
    else:
        planet_az = planet_az_func(tod.timestamps)
        planet_el = planet_el_func(tod.timestamps)
        q_planet = quat.rotation_lonlat(planet_az, planet_el)
        return q_planet

def get_det_centered_sight(tod, planet, q_planet=None, q_bs=None, q_det=None, 
                           det=None):
    if q_planet is None:
        q_planet = get_planet_trajectry(tod, planet)
    if q_bs is None:
        q_bs = quat.rotation_lonlat(tod.boresight.az, tod.boresight.el)
    if q_det is None:
        di = np.where(tod.dets.vals == det)[0][0]
        xi = tod.focal_plane.xi[di]
        eta = tod.focal_plane.eta[di]
        q_det = quat.rotation_xieta(-xi, eta)
    
    z_to_x = quat.rotation_lonlat(0, 0)
    sight = z_to_x * ~(q_bs * q_det) * q_planet
    return sight

def make_det_centered_maps(tod, planet, hdf_path, have_same_xieta=True, cuts=None,
                           wcs_kernel=None, res=0.1*coords.DEG, signal='signal'):
    if wcs_kernel is None:
        wcs_kernel = coords.get_wcs_kernel('car', 0, 0, res)
    
    q_planet = get_planet_trajectry(tod, planet)
    q_bs = quat.rotation_lonlat(tod.boresight.az, tod.boresight.el)
    if have_same_xieta:
        xi_det, eta_det = tod.focal_plane.xi[0], tod.focal_plane.eta[0]
        q_det = quat.rotation_xieta(-xi_det, eta_det)
        tod_single = tod.restrict('dets', tod.dets.vals[:1], in_place=False)
        sight = get_det_centered_sight(tod_single, planet, q_planet=q_planet, q_bs=q_bs, q_det=q_det)
        tod_single.focal_plane.xi *= 0.
        tod_single.focal_plane.eta *= 0.
        tod_single.boresight.roll *= 0.
        if cuts is None:
            cuts_single = None
        else:
            cuts_single = cuts[0]
        P = coords.P.for_tod(tod=tod_single, wcs_kernel=wcs_kernel, comps='T', cuts=cuts_single, sight=sight)
            
    for di, det in enumerate(tqdm(tod.dets.vals)):
        tod_single = tod.restrict('dets', tod.dets.vals[di:di+1], in_place=False)
        
        if not have_same_xieta:
            xi_det, eta_det = tod_single.focal_plane.xi[0], tod_single.focal_plane.eta[0]
            q_det = quat.rotation_xieta(-xi_det, eta_det)
            sight = get_det_centered_sight(tod_single, planet, q_planet=q_planet, q_bs=q_bs, q_det=q_det)
            tod_single.focal_plane.xi *= 0.
            tod_single.focal_plane.eta *= 0.
            if cuts is None:
                cuts_single = None
            else:
                cuts_single = cuts[di]
            P = coords.P.for_tod(tod=tod_single, wcs_kernel=wcs_kernel, comps='T', cuts=cuts_single, sight=sight)
        
        mT_weighted = P.to_map(tod=tod_single, signal=signal, comps='T', det_weights=None)
        wT = P.to_weights(tod_single, signal=signal, comps='T', det_weights=None)
        mT = P.remove_weights(signal_map=mT_weighted, weights_map=wT, comps='T')[0]
        enmap.write_hdf(hdf_path, mT, address=det, extra={'xi0': xi_det, 'eta0': eta_det})
    return

def detect_peak_xieta(mT, filter_size=None):
    if filter_size is None:
        filter_size = int(np.min(mT.shape)//10)
    local_max = maximum_filter(mT, footprint=np.ones((filter_size, filter_size)), mode='constant', cval=np.nan)
    peak_i, peak_j = np.where(mT == np.nanmax(local_max))
    peak_i = int(np.median(peak_i))
    peak_j = int(np.median(peak_j))
    dec_grid, ra_grid = mT.posmap()
    
    ra_peak = ra_grid[peak_i][peak_j]
    dec_peak = dec_grid[peak_i][peak_j]
    xi_peak, eta_peak = radec2xieta(ra_peak, dec_peak)
    return xi_peak, eta_peak, ra_peak, dec_peak, peak_i, peak_j

def get_center_of_mass(x, y, z, 
                       circle_mask={'x0':0, 'y0':0, 'r_circle':3.0*coords.DEG},
                       percentile_mask = {'q': 50}):
    mask = ~np.isnan(z)
    if circle_mask is not None:
        x0, y0 = circle_mask['x0'], circle_mask['y0']
        r_circle = circle_mask['r_circle']
        r = np.sqrt((x-x0)**2 + (y-y0)**2)
        mask = np.logical_and(mask, r<r_circle)
        
    if percentile_mask is not None:
        q = percentile_mask['q']
        mask = np.logical_and(mask, z>np.nanpercentile(z[mask], q))
    
    _x = x[mask]
    _y = y[mask]
    _z = z[mask]
        
    total_mass = np.nansum(_z)
    x_center = np.nansum(_x * _z) / total_mass
    y_center = np.nansum(_y * _z) / total_mass
    return x_center, y_center

def get_edgemap(mT, edge_avoidance=1*coords.DEG, edge_check='nan'):
    if edge_check not in ('nan', 'zero'):
        raise ValueError('only `nan` or `zero` is supported')
    
    edge_map = enmap.zeros(mT.shape, mT.wcs)
    edge_margin_size = int(edge_avoidance/np.mean(mT.pixshape()))
    
    for i, row in enumerate(mT):
        if edge_check == 'nan':
            nonzero_idxes = np.where(~np.isnan(row))[0]
        elif edge_check == 'zero':
            nonzero_idxes = np.where(row != 0)[0]
        if len(nonzero_idxes>0):
            edge_map[i, :nonzero_idxes[0] + edge_margin_size] = True
            edge_map[i, nonzero_idxes[-1] - edge_margin_size:] = True
        else:
            edge_map[i, :] = True
            
    for j, col in enumerate(mT.T):
        if edge_check == 'nan':
            nonzero_idxes = np.where(~np.isnan(col))[0]
        elif edge_check == 'zero':
            nonzero_idxes = np.where(col != 0)[0]
        if len(nonzero_idxes>0):
            edge_map[:nonzero_idxes[0] + edge_margin_size, j] = True
            edge_map[nonzero_idxes[-1] - edge_margin_size:, j] = True
        else:
            edge_map[:, j] = True
    return edge_map

def radec2xieta(ra, dec, ra0=0, dec0=0):
    q0 = quat.rotation_lonlat(lon=ra0, lat=dec0)
    q = quat.rotation_lonlat(lon=ra, lat=dec)
    xi, eta, _ = quat.decompose_xieta(~q0 * q)
    return xi, eta

def _gauss1d(x, peak, sigma, base):
    return peak * np.exp( - x**2 / (2*sigma**2)) + base
    
def map_to_xieta(mT, edge_avoidance=1.0*coords.DEG, edge_check='nan',
                 r_tune_circle=1.0*coords.DEG, q_tune=50, 
                 R2_threshold=0.5, r_fit_circle=3.0*coords.DEG, beam_sigma_init=0.5*coords.DEG, ):
    xi_peak, eta_peak, ra_peak, dec_peak, peak_i, peak_j = detect_peak_xieta(mT)
    edge_map = get_edgemap(mT, edge_avoidance=edge_avoidance, edge_check=edge_check)
    edge_valid = not edge_map[peak_i, peak_j]
    
    if edge_valid:
        dec_flat, ra_flat = mT.posmap()
        dec_flat = dec_flat.flatten()
        ra_flat = ra_flat.flatten()
        xi_flat, eta_flat = radec2xieta(ra_flat, dec_flat)

        circle_mask = {'x0':xi_peak, 'y0':eta_peak, 'r_circle':r_tune_circle}
        percentile_mask = {'q': q_tune}
        xi_peak, eta_peak = get_center_of_mass(xi_flat, eta_flat, mT.flatten(), 
                                              circle_mask=circle_mask, percentile_mask=percentile_mask)
        
        # check R2(=coefficient of determination)
        r = np.sqrt((xi_flat - xi_peak)**2 + (eta_flat - eta_peak)**2)
        z = mT.flatten()
        mask_fit = np.logical_and(~np.isnan(z), r<r_fit_circle)
        _r = r[mask_fit]
        _z = z[mask_fit]
        
        if _r.shape[0] == 0:
            xi_det = np.nan
            eta_det = np.nan
        else:
            popt, pcov = curve_fit(_gauss1d, _r, _z,
                                   p0=[np.ptp(_z), beam_sigma_init, np.percentile(_z, 1)],
                                   bounds = ((-np.inf, beam_sigma_init/5, -np.inf),
                                              (np.inf, beam_sigma_init*5, np.inf),),
                                   max_nfev = 1000000
                                  )
            R2 = 1 - np.sum((_z - _gauss1d(_r, *popt))**2)/np.sum((_z - np.mean(_z))**2) # R2(=coefficient of determination)
            if R2 >= R2_threshold:
                xi_det = -xi_peak
                eta_det = eta_peak
            else:
                xi_det = np.nan
                eta_det = np.nan
    else:
        xi_det = np.nan
        eta_det = np.nan
    return xi_det, eta_det    

def get_xieta_from_maps(map_hdf_file, 
                        edge_avoidance=1.0*coords.DEG, edge_check='nan',
                        r_tune_circle=1.0*coords.DEG, q_tune=50,
                        save=False, output_dir=None, filename=None):
    _input_file = h5py.File(map_hdf_file, 'r')
    
    with _input_file as ifile:
        keys = list(ifile.keys())
        dets = [key for key in keys if isinstance(ifile[key], h5py.Group)]
        xieta_dict = {}
        for di, det in enumerate(tqdm(dets)):
            mT = enmap.read_hdf(ifile[det])
            mT[mT==0] = np.nan
            xi, eta = map_to_xieta(mT, edge_avoidance=edge_avoidance, edge_check=edge_check,
                                    r_tune_circle=r_tune_circle, q_tune=q_tune)
            xi0 = float(ifile[det]['xi0'][...])
            eta0 = float(ifile[det]['eta0'][...])
            xi, eta = _add_xieta((xi0, eta0), (xi, eta))
            xieta_dict[det] = {'xi':xi, 'eta':eta}
    if save:
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'pointing_results')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if filename is None:
            filename = 'focalplane_' + os.path.splitext(os.path.basename(map_hdf_file))[0] + '.hdf'
        output_file = os.path.join(output_dir, filename)
        
        focalplane = metadata.ResultSet(keys=['dets:readout_id', 'band', 'channel', 'xi0', 'eta0', 'gamma'])
        for det in dets:
            band = int(det.split('_')[-2])
            channel = int(det.split('_')[-1])
            focalplane.rows.append((det, band, channel, xieta_dict[det]['xi'], xieta_dict[det]['eta'], 0.))
        write_dataset(focalplane, output_file, 'focalplane', overwrite=True)
    return xieta_dict

def _add_xieta(xieta1, xieta2):
    xi1, eta1 = xieta1
    xi2, eta2 = xieta2
    
    q1 = quat.rotation_xieta(xi1, eta1)
    q2 = quat.rotation_xieta(xi2, eta2)
    
    q_add = q2*q1
    xi_add, eta_add, _ = quat.decompose_xieta(q_add)
    return xi_add, eta_add
