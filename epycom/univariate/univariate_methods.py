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

# Local imports


def compute_signal_stats(sig,**kwargs):
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
    sig = sig**2

    # compute signal power statistics
    sig_f_pw_std = sig.std()
    sig_f_pw_mean = sig.mean()
    sig_f_pw_median = np.median(sig)
    sig_f_pw_max = sig.max()
    sig_f_pw_min = sig.min()
    sig_f_pw_perc25 = np.percentile(sig, 25)
    sig_f_pw_perc75 = np.percentile(sig, 75)

    # TODO: why this? isn't everyhing a float before?
    sig_stats = [float(x) for x in [sig_f_pw_std, sig_f_pw_mean,
                                    sig_f_pw_median, sig_f_pw_max, sig_f_pw_min,
                                    sig_f_pw_perc25, sig_f_pw_perc75]]

    return sig_stats


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
