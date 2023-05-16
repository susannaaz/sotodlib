import numpy as np
from scipy.optimize import curve_fit
from sotodlib import core, tod_ops
import logging

logger = logging.getLogger(__name__)


def extract_hwpss(aman, signal=None, hwp_angle=None,
                  bin_signal=True, bins=3600,
                  lin_reg=True, modes=[1, 2, 3, 4, 6, 8],
                  apply_prefilt=True, prefilt_cutoff=1.0,
                  mask_flags=True,
                  merge_stats=True, hwpss_stats_name='hwpss_stats',
                  merge_extract=True, hwpss_extract_name='hwpss_extract'):
    """
    Extracts HWP synchronous signal (HWPSS) from a time-ordered data (TOD) using linear regression or curve-fitting.

    Parameters
    ----------
    aman : AxisManager object
        The TOD to extract HWPSS from.
    signal : array-like, optional
        The TOD signal to use. If not provided, `aman.signal` will be used.
    hwp_angle : array-like, optional
        The HWP angle for each sample in `aman`. If not provided, `aman.hwp_angle` will be used.
    bin_signal : bool, optional
        Whether to bin the TOD signal into HWP angle bins before extracting HWPSS. Default is `True`.
    bins : int, optional
        The number of HWP angle bins to use if `bin_signal` is `True`. Default is 3600.
    lin_reg : bool, optional
        Whether to use linear regression to extract HWPSS from the binned signal. If `False`, curve-fitting will be used instead.
        Default is `True`.
    modes : list of int, optional
        The HWPSS harmonic modes to extract. Default is [1, 2, 3, 4, 6, 8].
    apply_prefilt : bool, optional
        Whether to apply a high-pass filter to signal before extracting HWPSS. Default is `True`.
    prefilt_cutoff : float, optional
        The cutoff frequency of the high-pass filter, in Hz. Only used if `apply_prefilt` is `True`. Default is 1.0.
    mask_flags : bool, optional
        Whether to mask out flagged samples before extracting HWPSS. Default is `True`.
    merge_stats : bool, optional
        Whether to add the extracted HWPSS statistics to `aman` as new axes. Default is `True`.
    hwpss_stats_name : str, optional
        The name to use for the new field containing the HWPSS statistics if `merge_stats` is `True`. Default is 'hwpss_stats'.
    merge_extract : bool, optional
        Whether to add the extracted HWPSS to `aman` as a new signal field. Default is `True`.
    hwpss_extract_name : str, optional
        The name to use for the new signal field containing the extracted HWPSS if `merge_extract` is `True`. Default is 'hwpss_extract'.

    Returns
    -------
    hwpss_stats : AxisManager object
        The extracted HWPSS and its statistics.
    """

    if signal is None:
        if apply_prefilt:
            filt = tod_ops.filters.high_pass_sine2(cutoff=prefilt_cutoff)
            signal = np.array(tod_ops.fourier_filter(
                aman, filt, detrend='linear', signal_name='signal'))
        else:
            signal = aman.signal

    if hwp_angle is None:
        hwp_angle = aman.hwp_angle

    # define hwpss_stats
    mode_names = []
    for mode in modes:
        mode_names.append(f'S{mode}')
        mode_names.append(f'C{mode}')

    hwpss_stats = core.AxisManager(aman.dets, aman.samps, core.LabelAxis(
        name='modes', vals=np.array(mode_names, dtype='<U3')))
    if bin_signal:
        hwp_angle_bin_centers, binned_hwpss, hwpss_sigma_bin = binning_signal(
            aman, signal, hwp_angle=None, bins=bins, mask_flags=mask_flags)
        hwpss_stats.wrap('hwp_angle_bin_centers', hwp_angle_bin_centers, [
                       (0, core.IndexAxis('bin_samps', count=bins))])
        hwpss_stats.wrap('binned_hwpss', binned_hwpss, [
                       (0, 'dets'), (1, 'bin_samps')])
        hwpss_stats.wrap('hwpss_sigma_bin', hwpss_sigma_bin, [(0, 'dets')])

        if lin_reg:
            fitsig_binned, coeffs, covars, redchi2s = hwpss_linreg(
                x=hwp_angle_bin_centers, ys=binned_hwpss, yerrs=hwpss_sigma_bin, modes=modes)
        else:
            Params_init = guess_hwpss_params(
                x=hwp_angle_bin_centers, ys=binned_hwpss, modes=modes)
            fitsig_binned, coeffs, covars, redchi2s = hwpss_curvefit(x=hwp_angle_bin_centers, ys=binned_hwpss, yerrs=hwpss_sigma_bin,
                                                                     modes=modes, Params_init=Params_init)
        # tod template
        fitsig_tod = harms_func(hwp_angle, modes, coeffs)

        # wrap the optimal values and stats
        hwpss_stats.wrap('fitsig_binned', fitsig_binned,
                       [(0, 'dets'), (1, 'bin_samps')])
        hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes')])
        hwpss_stats.wrap('covars', covars, [
                       (0, 'dets'), (1, 'modes'), (2, 'modes')])
        hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])

    else:
        if mask_flags:
            m = ~aman.flags.glitches.mask()
        else:
            m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)

        hwpss_sigma_tod = estimate_sigma_tod(signal, hwp_angle)
        hwpss_stats.wrap('hwpss_sigma_tod', hwpss_sigma_tod, [(0, 'dets')])

        if lin_reg:
            fitsig_tod, coeffs, covars, redchi2s = hwpss_linreg(
                x=hwp_angle, ys=signal, yerrs=hwpss_sigma_tod, modes=modes)

        else:
            raise ValueError('Curve-fitting for TOD are specified.' +
                             'It will take too long time and return meaningless result.' +
                             'Specify (bin_signal, lin_reg) = (True, True) or (True, False) or (False, True)')

        hwpss_stats.wrap('coeffs', coeffs, [(0, 'dets'), (1, 'modes')])
        hwpss_stats.wrap('covars', covars, [
                       (0, 'dets'), (1, 'modes'), (2, 'modes')])
        hwpss_stats.wrap('redchi2s', redchi2s, [(0, 'dets')])
    
    if merge_stats:
        aman.wrap(hwpss_stats_name, hwpss_stats)
    if merge_extract:
        aman.wrap(hwpss_extract_name, fitsig_tod, [(0, 'dets'), (1, 'samps')])
    return hwpss_stats


def binning_signal(aman, signal=None, hwp_angle=None,
                   bins=360, mask_flags=False):
    """
    Bin time-ordered data by the HWP angle and return the binned signal and its standard deviation.

    Parameters
    ----------
    aman : TOD
        The Axismanager object to be binned.
    signal : str, optional
        The name of the signal to be binned. Defaults to aman.signal if not specified.
    hwp_angle : str, optional
        The name of the timestream of hwp_angle. Defaults to aman.hwp_angle if not specified.
    bins : int, optional
        The number of HWP angle bins to use. Default is 360.
    mask_flags : bool, optional
        Flag indicating whether to exclude flagged samples when binning the signal. Default is False.

    Returns
    -------
    aman_proc:
        The AxisManager object which contains
        * center of each bin of hwp_angle
        * binned hwp synchrounous signal
        * estimated sigma of binned signal
    """
    if signal is None:
        signal = aman.signal
    if hwp_angle is None:
        hwp_angle = aman.hwp_angle

    # binning hwp_angle tod
    hwpss_denom, hwp_angle_bins = np.histogram(
        hwp_angle, bins=bins, range=[0, 2 * np.pi])

    # convert bin edges into bin centers
    hwp_angle_bin_centers = (
        hwp_angle_bins[1]-hwp_angle_bins[0])/2 + hwp_angle_bins[:-1]

    # prepare binned signals
    binned_hwpss = np.zeros((aman.dets.count, bins), dtype='float32')
    binned_hwpss_squared_mean = np.zeros(
        (aman.dets.count, bins), dtype='float32')
    binned_hwpss_sigma = np.zeros((aman.dets.count, bins), dtype='float32')

    # get mask from aman
    if mask_flags:
        m = ~aman.flags.glitches.mask()
    else:
        m = np.ones([aman.dets.count, aman.samps.count], dtype=bool)

    # binning tod
    for i in range(aman.dets.count):
        binned_hwpss[i][:] = np.histogram(hwp_angle[m[i]], bins=bins, range=[0, 2*np.pi],
                                          weights=signal[i][m[i]])[0] / np.where(hwpss_denom == 0, 1, hwpss_denom)

        binned_hwpss_squared_mean[i][:] = np.histogram(hwp_angle[m[i]], bins=bins, range=[0, 2*np.pi],
                                                       weights=signal[i][m[i]]**2)[0] / np.where(hwpss_denom == 0, 1, hwpss_denom)

    # get sigma of each bin
    binned_hwpss_sigma = np.sqrt(np.abs(binned_hwpss_squared_mean - binned_hwpss**2)
                                 ) / np.sqrt(np.where(hwpss_denom == 0, 1, hwpss_denom))
    # use median of sigma of each bin as uniform sigma for a detector
    hwpss_sigma = np.nanmedian(binned_hwpss_sigma, axis=-1)

    return hwp_angle_bin_centers, binned_hwpss, hwpss_sigma


def hwpss_linreg(x, ys, yerrs, modes):
    """
    Performs a linear regression of the input data ys as a function of x, using a set of sine and cosine
    basis functions defined by the input modes. Returns the fitted signal, the coefficients of the
    basis functions, their covariance matrix, and the reduced chi-square.

    Parameters
    -----------
    x : numpy.ndarray
        The independent variable values of the data points to fit.
    ys : numpy.ndarray
        The dependent variable values of the data points to fit.
    yerrs : numpy.ndarray
        The error estimates of the dependent variable values.
    modes : list of int
        The frequencies of the sine and cosine basis functions to use.

    Returns
    -------
    fitsig : numpy.ndarray
        The fitted signal, obtained by evaluating the model with the optimal coefficients.
    coeffs : numpy.ndarray
        The coefficients of the sine and cosine basis functions that best fit the data.
    covars : numpy.ndarray
        The covariance matrix of the coefficients, estimated from the data errors.
    redchi2s : numpy.ndarray
        The reduced chi-square statistic of the fit, computed for each data point.
    """
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    I = np.linalg.inv(np.tensordot(vects, vects, (1, 1)))
    coeffs = np.matmul(ys, vects.T)
    coeffs = np.dot(I, coeffs.T).T
    fitsig = np.matmul(vects.T, coeffs.T).T

    # covariance of coefficients
    covars = np.zeros((ys.shape[0], 2*len(modes), 2*len(modes)))
    for det_idx in range(ys.shape[0]):
        covars[det_idx, :, :] = I * yerrs[det_idx]**2

    # reduced chi-square
    redchi2s = np.sum(
        ((ys - fitsig)/yerrs[:, np.newaxis])**2, axis=-1) / (x.shape[0] - 2*len(modes))

    return fitsig, coeffs, covars, redchi2s


def wrapper_harms_func(x, modes, *args):
    """
    A wrapper function for the harmonics function to be used for fitting data using Scipy's curve-fitting algorithm.
    Parameters
    ----------
    x : array-like
        The x-values of the data points to be fitted.

    modes : array-like
        An array of integers representing the modes of the harmonics function.

    *args : tuple
        A tuple of arguments. The first argument should be an array of coefficients used to calculate the harmonics function.

    Returns
    -------
    y : array-like
        An array of the same length as x representing the values of the harmonics function evaluated at x using the given 
        modes and coefficients.
    """
    coeffs = np.array(args[0])
    return harms_func(x, modes, coeffs)


def harms_func(x, modes, coeffs):
    """
    calculates the harmonics function given the input values, modes and coefficients.

    Parameters
    ----------
    x (numpy.ndarray): Input values
    modes (list): List of modes to be used in the harmonics function
    coeffs (numpy.ndarray): Coefficients of the harmonics function

    Returns
    -------
    numpy.ndarray: The calculated harmonics function.
    """
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)

    harmonics = np.matmul(vects.T, coeffs.T).T
    return harmonics


def guess_hwpss_params(x, ys, modes):
    """
    Compute initial guess for the coefficients of a harmonics-based fit to data.

    Parameters
    ----------
    x : array-like of shape (nsamps,)
    ys : array-like of shape (ndets, nsamps)
    modes : array-like of shape (nmodes,)
        List of modes to use in the fit.

    Returns
    -------
    Params_init : ndarray of shape (m, 2*p)
        Initial guess for the coefficients of a harmonics-based fit to the data.
    """
    vects = np.zeros([2*len(modes), x.shape[0]], dtype='float32')
    for i, mode in enumerate(modes):
        vects[2*i, :] = np.sin(mode*x)
        vects[2*i+1, :] = np.cos(mode*x)
    Params_init = 2 * np.matmul(ys, vects.T) / x.shape[0]
    return Params_init


def hwpss_curvefit(x, ys, yerrs, modes, Params_init=None):
    """
    Fit harmonics to input data using scipy's curve_fit method.

    Parameters
    ----------
    x : array_like
        1-D array of x values.
    ys : array_like
        2-D array of y values for each detector.
    yerrs : array_like
        1-D array of the standard deviation of the y values for each detector.
    modes : array_like
        1-D array of mode numbers to be fitted.
    Params_init : array_like, optional
        2-D array of initial parameter values for each detector. Default is None.

    Returns
    -------
    fitsig : ndarray
        2-D array of the fitted values for each detector.
    coeffs : ndarray
        2-D array of the fitted coefficients for each detector.
    covars : ndarray
        3-D array of the covariance matrix of the fitted coefficients for each detector.
    redchi2s : ndarray
        1-D array of the reduced chi-square values for each detector.

    Notes
    -----
    This function fits a set of harmonic functions to the input data using scipy's curve_fit method.
    The `modes` parameter specifies the mode numbers to be fitted.
    The `Params_init` parameter can be used to provide initial guesses for the fit parameters.
    """
    N_dets = ys.shape[0]
    N_samps = ys.shape[-1]
    N_modes = len(modes)

    if Params_init is None:
        Params_init = np.zeros((N_dets, 2*N_modes))

    coeffs = np.zeros((N_dets, 2*len(modes)))
    covars = np.zeros((N_dets, 2*len(modes), 2*len(modes)))
    redchi2s = np.zeros(N_dets)
    fitsig = np.zeros((N_dets, N_samps))

    for det_idx in range(N_dets):
        params_init = Params_init[det_idx]
        coeff, covar = curve_fit(lambda x, *params_init: wrapper_harms_func(x, modes, params_init),
                                 x, ys[det_idx], p0=params_init, sigma=yerrs[det_idx] *
                                 np.ones_like(ys[det_idx]),
                                 absolute_sigma=True)

        coeffs[det_idx, :] = coeff
        covars[det_idx, :] = covar

        yfit = harms_func(x, modes, coeff)
        fitsig[det_idx, :] = yfit
        redchi2s[det_idx] = np.sum(
            ((ys[det_idx] - yfit) / yerrs[det_idx])**2) / (x.shape[0] - 2*len(modes))

    return fitsig, coeffs, covars, redchi2s


def estimate_sigma_tod(signal, hwp_angle):
    """
    Estimate the noise level of a signal in a time-ordered data (TOD) using a half-wave plate (HWP) modulation.

    Parameters
    ----------
    signal : ndarray
        A 2D numpy array of shape (n_dets, n_samps) containing the TOD of each detector.
    hwp_angle : ndarray
        A 1D numpy array containing the HWP angles in degrees.

    Returns
    -------
    hwpss_sigma_tod : ndarray
        A 1D numpy array containing the estimated noise level for each detector.

    Notes
    -----
    This function computes the mean of the signal in each period of HWP rotation and multiplies it
    by the square root of the number of samples in that period. The standard deviation of the
    resulting values for all periods is then computed and returned as the estimated sigma of each data point.
    """
    hwp_zeros_idxes = np.where(np.abs(np.diff(hwp_angle)) > 5)[0][:] + 1
    hwpss_sigma_tod = np.zeros((signal.shape[0], hwp_zeros_idxes.shape[0] - 1))

    for i, (init_idx, end_idx) in enumerate(zip(hwp_zeros_idxes[:-1], hwp_zeros_idxes[1:])):
        hwpss_sigma_tod[:, i] = np.mean(
            signal[:, init_idx:end_idx], axis=-1) * np.sqrt(end_idx - init_idx)
    hwpss_sigma_tod = np.std(hwpss_sigma_tod, axis=-1)
    return hwpss_sigma_tod


def subtract_hwpss(aman, signal=None, hwpss_template=None,
                   subtract_name='hwpss_remove'):
    """
    Subtract the half-wave plate synchronous signal (HWPSS) template from the
    signal in the given axis manager.

    Parameters
    ----------
    aman : AxisManager
        The axis manager containing the signal to which the HWPSS template will
        be applied.
    signal : ndarray, optional
        The signal to which the HWPSS template will be applied. If `signal` is
        None (default), the signal contained in the axis manager will be used.
    hwpss_template : ndarray, optional
        The HWPSS template to be subtracted from the signal. If `hwpss_template`
        is None (default), the HWPSS template stored in the axis manager under
        the key 'hwpss_extract' will be used.
    subtract_name : str, optional
        The name of the output axis manager that will contain the HWPSS-
        subtracted signal. Defaults to 'hwpss_remove'.

    Returns
    -------
    None
    """
    if signal is None:
        signal = aman.signal
    if hwpss_template is None:
        hwpss_template = aman['hwpss_extract']

    aman.wrap(subtract_name, np.subtract(
        signal, hwpss_template), [(0, 'dets'), (1, 'samps')])


def demod_tod(aman, signal_name='signal', demod_mode=4,
              bpf_cfg=None, lpf_cfg=None):
    """
    Demodulate TOD based on HWP angle

    Parameters
    ----------
    aman : AxisManager
        The AxisManager object
    signal_name : str, optional
        Axis name of the demodulated signal in aman. Default is 'signal'.
    demod_mode : int, optional
        Demodulation mode. Default is 4.
    bpf_cfg : dict
        Configuration for Band-pass filter applied to the TOD data before demodulation.
        If not specified, a 4th-order Butterworth filter of 
        (demod_mode * HWP speed) +/- 0.95*(HWP speed) is used.
        Example) bpf_cfg = {'type': 'butter4', 'center': 8.0, 'width': 3.8}
    lpf_cfg : dict
        Configuration for Low-pass filter applied to the demodulated TOD data. If not specified,
        a 4th-order Butterworth filter with a cutoff frequency of 0.95*(HWP speed)
        is used.
        Example) lpf_cfg = {'type': 'butter4', 'cutoff': 1.9}

    Returns
    -------
    None
        The demodulated TOD data is added to the input `aman` container as new signals:
        'dsT' for the original signal filtered with `lpf`, 'demodQ' for the demodulated
        signal real component filtered with `lpf` and multiplied by 2, and 'demodU' for
        the demodulated signal imaginary component filtered with `lpf` and multiplied by 2.

    """
    # HWP speed in Hz
    speed = (np.sum(np.abs(np.diff(np.unwrap(aman.hwp_angle)))) /
            (aman.timestamps[-1] - aman.timestamps[0])) / (2 * np.pi)
    
    if bpf_cfg is None:
        bpf_center = demod_mode * speed
        bpf_width = speed * 2. * 0.95
        bpf_cfg = {'type': 'butter4',
                   'center': bpf_center,
                   'width': bpf_width}
    bpf = get_bpf(bpf_cfg)
    
    if lpf_cfg is None:
        lpf_cutoff = speed * 0.95
        lpf_cfg = {'type': 'butter4',
                  'cutoff': lpf_cutoff}
    lpf = get_lpf(lpf_cfg)
        
    phasor = np.exp(demod_mode * 1.j * aman.hwp_angle)
    demod = tod_ops.fourier_filter(aman, bpf, detrend=None,
                                   signal_name=signal_name) * phasor
    
    # dsT
    aman.wrap_new('dsT', dtype='float32', shape=('dets', 'samps'))
    aman.dsT = aman[signal_name]
    aman['dsT'] = tod_ops.fourier_filter(
        aman, lpf, signal_name='dsT', detrend=None)
    # demodQ
    aman.wrap_new('demodQ', dtype='float32', shape=('dets', 'samps'))
    aman['demodQ'] = demod.real
    aman['demodQ'] = tod_ops.fourier_filter(
        aman, lpf, signal_name='demodQ', detrend=None) * 2.
    # demodU
    aman.wrap_new('demodU', dtype='float32', shape=('dets', 'samps'))
    aman['demodU'] = demod.imag
    aman['demodU'] = tod_ops.fourier_filter(
        aman, lpf, signal_name='demodU', detrend=None) * 2.

def get_lpf(lpf_cfg):
    """
    Returns a low-pass filter based on the configuration.

    Args:
        lpf_cfg (dict): A dictionary containing the low-pass filter configuration.
            It must have the following keys:
            - "type": A string specifying the type of low-pass filter. Supported values are "butter4" and "sine2".
            - "cutoff": A float specifying the cutoff frequency of the low-pass filter.
            - "trans_width": A float specifying the transition width of the low-pass filter (only for "sine2" type).

    Returns:
        numpy.ndarray: A 1D array representing the filter coefficients of the low-pass filter.
    """
    if lpf_cfg['type'] == 'butter4':
        cutoff = lpf_cfg['cutoff']
        return tod_ops.filters.low_pass_butter4(fc=cutoff)
    elif lpf_cfg['type'] == 'sine2':
        cutoff = lpf_cfg['cutoff']
        trans_width = lpf_cfg['trans_width']
        return tod_ops.filters.low_pass_sine2(cutoff=cutoff, width=trans_width)
    else:
        raise ValueError('Unsupported filter type. Supported filters are `butter4` and `sine2`')


def get_bpf(bpf_cfg):
    """
    Returns a band-pass filter based on the configuration.

    Args:
        bpf_cfg (dict): A dictionary containing the band-pass filter configuration.
            It must have the following keys:
            - "type": A string specifying the type of band-pass filter. Supported values are "butter4" and "sine2".
            - "center": A float specifying the center frequency of the band-pass filter.
            - "width": A float specifying the width of the band-pass filter.
            - "trans_width": A float specifying the transition width of the band-pass filter (only for "sine2" type).

    Returns:
        numpy.ndarray: A 1D array representing the filter coefficients of the band-pass filter.
    """
    if bpf_cfg['type'] == 'butter4':
        center = bpf_cfg['center']
        width = bpf_cfg['width']
        return tod_ops.filters.low_pass_butter4(fc=center + width/2.) *\
                tod_ops.filters.high_pass_butter4(fc=center - width/2.)
    elif bpf_cfg['type'] == 'sine2':
        center = bpf_cfg['center']
        width = bpf_cfg['width']
        trans_width = bpf_cfg['trans_width']
        return tod_ops.filters.low_pass_sine2(cutoff=center + width/2., width=trans_width)*\
                tod_ops.filters.high_pass_sine2(cutoff=center - width/2., width=trans_width)
    else:
        raise ValueError('Unsupported filter type. Supported filters are `butter4` and `sine2`')


        
        
        