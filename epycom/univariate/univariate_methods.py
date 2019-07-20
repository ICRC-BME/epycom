# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
import pickle

# Third pary imports
import math
import numpy as np
from scipy.signal import butter, hilbert, filtfilt
from scipy.spatial.distance import pdist, squareform

# Local imports


def compute_signal_stats(sig, **kwargs):
    """
    Function to analyze basic stats of signal

    Parameters:
    ----------
    sig: np.array
        signal to analyze, time series (array, int, float)

    Returns
    -------
    results: list
        - power_std: standard deviation of power in band
        - power_mean: mean of power in band
        - power_median: median of power in band
        - power_max: max value of power in band
        - power_min: min value of power in band
        - power_perc25: 25 percentile of power in band
        - power_perc75: 75 percentile of power in band

    Example
    -------
    sig_stats = compute_signal_stats(sig)
    """

    # signal power
    sig = sig.astype(np.float)**2

    # compute signal power statistics
    sig_f_pw_std = sig.std()
    sig_f_pw_mean = sig.mean()
    sig_f_pw_median = np.median(sig)
    sig_f_pw_max = sig.max()
    sig_f_pw_min = sig.min()
    sig_f_pw_perc25 = np.percentile(sig, 25)
    sig_f_pw_perc75 = np.percentile(sig, 75)

    sig_stats = [sig_f_pw_std, sig_f_pw_mean, sig_f_pw_median, sig_f_pw_max,
                 sig_f_pw_min, sig_f_pw_perc25, sig_f_pw_perc75]

    return sig_stats


def compute_hjorth_mobility(signal, fs, **kwargs):
    """
    Function to compute Hjorth mobility of time series

    Parameters
    ----------
    signal: np.array
        Signal to analyze, time series (array, int, float)
    fs: float
        Sampling frequency of the time series

    Returns
    -------
    hjorth_mobility: float

    Example
    -------
    hjorth_mobility = compute_hjorth_mobility(data, 5000)

    Note
    ----
    result is frequency dependent
    """

    variancex = signal.var(ddof=1)
    # diff signal is one sample shorter
    variancedx = np.var(np.diff(signal) * fs, ddof=1)
    # compute variance with degree of freedom=1 => The mean is normally
    # calculated as x.sum() / N, where N = len(x). If, however, ddof is
    # specified, the divisor N - ddof is used instead.

    hjorth_mobility = np.sqrt(variancedx / variancex)
    return hjorth_mobility


def compute_hjorth_complexity(signal, fs, **kwargs):
    """
    Function to compute Hjorth complexity of time series

    Parameters
    ----------
    signal: np.array
        Signal to analyze, time series (array, int, float)
    fs: float
        Sampling frequency of the time series

    Returns
    -------
    hjorth_complexity: float

    Example
    -------
    hjorth_complexity = compute_hjorth_complexity(data, 5000)

    Note
    ----
    result is NOT frequency dependent
    """

    mob = compute_hjorth_mobility(signal, fs)
    # diff signal is one sample shorter
    mobd = compute_hjorth_mobility(np.diff(signal) * fs, fs)
    hjorth_complexity = mobd / mob
    return hjorth_complexity

# ______Help functions for large Lyapunov exponent computation


def _compute_phase_space(data, dimensions, sample_lag):
    """
    Function to create phase space of time series data. This function
    takes value lagged by sample_lag as coordination in new dimension.

    Parameters
    ----------
    data: np.array
        Signal to analyze, time series (array, int, float)
    dimensions: int
        Number of dimensions to create (int)
    sample_lag: int
        Delay in samples used for coordination extraction (int)

    Returns
    -------
    space: numpy.ndarray
        "Dimensions" times dinmensional space

    Example
    -------
    space = _compute_phase_space(data,8,5000)
    """

    length = np.size(data)
    space = np.zeros([dimensions, length - ((dimensions - 1) * sample_lag)])
    for i in range(dimensions):
        start = i * sample_lag
        end = length - ((dimensions - i - 1) * sample_lag)
        space[i, :] = data[start:end]
    return space


def _compute_acorr_exp(data, fs):
    """
    Function to find point, where autocorrelation drops to 1-1/np.e of it's
    maximum

    Paremeters
    ----------
    data: np.array
        Signal to analyze, time series (array, int, float)
    fs: float
        Sampling frequency

    Returns
    -------
    point: sample where autocorrelation drops to  1-1/np.e of it's
            maximum
    """

    # It is supposed that autocorrelation function of EEG drops to 1-1/e of
    # it's value within one second. This assumption masively reduces
    # computation time
    data = data[0:fs]

    # normalize data
    data = data - np.mean(data)

    acorr = np.correlate(data, data, mode='full')
    acorr = acorr / max(acorr)
    acorr = acorr[acorr > (1 - 1 / np.e)]
    point = len(acorr) // 2

    return point


def compute_lyapunov_exp(data, fs=5000, dimension=5, sample_lag=None,
                         trajectory_len=20, min_tsep=500, **kwargs):
    """
    Lyapnov largest exponent estimation according to Rosenstein algorythm

    With use of some parts from nolds library:
    https://pypi.org/project/nolds
    https://github.com/CSchoel

    Parameters
    ----------
    data: np.array
        Signal to analyze, time series (array, int, float).
    fs: float
        Sampling frequency
    dimensions: int
        Number of dimensions to compute lyapunov exponent.
    sample_lag: int
        Delay in samples used for coordination extraction.
    trajectory_len: int
        Number of points on divergence trajectory.
    min_tstep: int
        Nearest neighbors have temporal separation greater then min_tstep.

    Returns
    -------
    le: float
        Estimation of largest Lyapunov coeficient acording to Rosenstein
        algorithm.

    Example
    -------
    le = compute_lyapunov_exp(data, fs=5000, dimension=5, sample_lag=None,
                         trajectory_len=20, min_tsep=500)
    """

    # If sample lag for creating orbit is not set, it will be counted as
    # a point, where autocorrelation function is 1-1/e.
    if sample_lag is None:
        sample_lag = _compute_acorr_exp(data, fs)

    # creating m-dimensional orbit by delaying 1D signal
    orbit = _compute_phase_space(data, int(dimension), int(sample_lag))

    # calculate euclidian distances amog all points in orbit
    distances = squareform(pdist(orbit.T, 'euclidean'))

    m = len(distances)

    # we do not want to consider vectors as neighbor that are less than
    # min_tsep time steps together => mask the distances min_tsep to the right
    # and left of each index by setting them to infinity (will never be
    # considered as nearest neighbors)

    for i in range(m):
        distances[i, max(0, i - min_tsep):i + min_tsep + 1] = float("inf")

    # check that we have enough data points to continue
    ntraj = m - trajectory_len + 1
    min_traj = min_tsep * 2 + 2  # in each row min_tsep + 1 disances are inf
    if ntraj <= 0:
        msg = "Not enough data points. Need {} additional data points to " \
            + "follow a complete trajectory."
        raise ValueError(msg.format(-ntraj + 1))
    if ntraj < min_traj:
        # not enough data points => there are rows where all values are inf
        assert np.any(np.all(np.isinf(distances[:ntraj, :ntraj]), axis=1))
        msg = "Not enough data points. At least {} trajectories are " \
            + "required to find a valid neighbor for each orbit vector with " \
            + "min_tsep={} but only {} could be created."
        raise ValueError(msg.format(min_traj, min_tsep, ntraj))
    assert np.all(np.any(np.isfinite(distances[:ntraj, :ntraj]), axis=1))

    # find nearest neighbors (exclude last columns, because these vectors
    # cannot be followed in time for trajectory_len steps)
    nb_idx = np.argmin(distances[:ntraj, :ntraj], axis=1)

    # build divergence trajectory by averaging distances along the trajectory
    # over all neighbor pairs
    div_traj = np.zeros(trajectory_len, dtype=float)
    for k in range(trajectory_len):
        # calculate mean trajectory distance at step k
        indices = (np.arange(ntraj) + k, nb_idx + k)
        div_traj_k = distances[indices]
        # filter entries where distance is zero (would lead to -inf after log)
        nonzero = np.where(div_traj_k != 0)
        if len(nonzero[0]) == 0:
            # if all entries where zero, we have to use -inf
            div_traj[k] = -np.inf
        else:
            div_traj[k] = np.mean(np.log(div_traj_k[nonzero]))

    # filter -inf entries from mean trajectory
    ks = np.arange(trajectory_len)
    finite = np.where(np.isfinite(div_traj))
    ks = ks[finite]
    div_traj = div_traj[finite]

    if len(ks) < 1:
        # if all points or all but one point in the trajectory is -inf, we
        # cannot fit a line through the remaining points => return -inf as
        # exponent
        poly = [-np.inf, 0]
    else:
        # normal line fitting
        poly = np.polyfit(np.arange(len(div_traj)), div_traj, 1)

    le = poly[0] / (sample_lag / fs)

    return le


def compute_fac(sig, fs, lfc1=1, hfc1=30, lfc2=65, hfc2=180, **kwargs):
    """
    Frequency-amplitude coupling

    Parameters
    ----------
    sig: np.array
        time series (float)

    Returns
    -------
    fac: float
        correlation of low freq. and high freq. envelope <-1,1>

    Example
    -------
    fac = compute_fac(sig, 5000)

    """
    nsamp = len(sig)

    zpad = 2**(math.ceil(math.log(nsamp, 2)))
    sig_zeros = np.zeros(zpad)

    b, a = butter(2, [lfc1 / (fs / 2), hfc1 / (fs / 2)], 'bandpass')
    sig_f = filtfilt(b, a, sig)

    b, a = butter(2, [lfc2 / (fs / 2), hfc2 / (fs / 2)], 'bandpass')
    sig_fh = filtfilt(b, a, sig)
    sig_zeros[0:nsamp] = sig_fh

    sig_a = abs(hilbert(sig_zeros))**2
    sig_a = sig_a[0:nsamp]

    fac = np.corrcoef(sig_f, sig_a)[0][1]

    return fac


def compute_pac(sig, fs, lfc1=1, hfc1=30, lfc2=65, hfc2=180, **kwargs):
    """
    Phase-amplitude coupling

    Parameters
    ----------
    sig: np.array
        time series (float)

    Returns
    -------
    pac: float
        correlation of low freq. phase and high freq. envelope <-1,1>

    Example
    -------
    pac = compute_pac(sig, 5000)
    """
    b, a = butter(2, [lfc1 / (fs / 2), hfc1 / (fs / 2)], 'bandpass')
    sig_f = filtfilt(b, a, sig)
    sig_f_ph = np.angle(hilbert(sig_f))

    b, a = butter(2, [lfc2 / (fs / 2), hfc2 / (fs / 2)], 'bandpass')
    sig_fh = filtfilt(b, a, sig)
    sig_a = abs(hilbert(sig_fh))**2

    pac = np.corrcoef(sig_f_ph, sig_a)[0][1]

    return pac


def compute_pse(sig, **kwargs):
    """
    Power spectral entropy

    Parameters
    ----------
    sig: np.array
        time series (float)

    Returns
    -------
    pse - power spectral entropy of analyzed signal, a non-negative value

    Example
    -------
    pac = comute_pse(sig)
    """

    ps = np.abs(np.fft.fft(sig))  # power spectrum
    ps = ps**2
    ps = ps / len(ps)  # power spectral density
    ps = ps / sum(ps)  # normalized to probability density function

    pse = -sum(ps * np.log2(ps))  # power spectral entropy

    return pse
