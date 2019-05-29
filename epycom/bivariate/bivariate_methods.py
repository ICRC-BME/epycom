# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
from numpy import angle, mean, sqrt
from scipy.signal import hilbert
from scipy.stats import entropy

# Local imports


def compute_linear_correlation(sig1, sig2, win=0, win_step=0):
    """
    Linear correlation (Pearson's coefficient), zero lag.

    Calculates correlation between two time series
    when win and win_step is assigned, calculates evolution of correlation
    when win>len(sig) or win<=0, calculates only one corr coef

    Parameters
    ----------
    sig1: np.array
        first time series (int, float)
    sig2: np.array
        second time series (int, float)
    win: int
        width of correlation win in samples (default=0)
    win_step: int
        step of win in samples (default=0

    Returns
    -------
    lincorr: list
        calculated correlation coeficients

    Example
    -------
    to calculate evolution of correlation (multiple steps of corr window):
    lincorr = compute_linear_correlation(sig1, sig2, 2500, 250)

    to calculate overall correlation (one win step):
    lincorr = compute_linear_correlation(sig1, sig2)
    """

    # TODO: this function should return numpy array or only one value

    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    if win > len(sig1) or win <= 0:
        win = len(sig1)

    if win_step <= 0:
        win_step = 1

    nstep = int((len(sig1) - win) / win_step)

    if nstep <= 0:
        nstep = 1

    lincorr = []
    for i in range(0, nstep):
        ind1 = i * win_step
        ind2 = ind1 + win
        if ind2 <= len(sig1):
            sig1_w = sig1[ind1:ind2]
            sig2_w = sig2[ind1:ind2]
            corr_val = np.corrcoef(sig1_w, sig2_w)
            lincorr.append(corr_val[0][1])

    #lincorr = np.median(lincorr)

    return lincorr


def compute_xcorr(sig1, sig2, lag, lag_step, win=0, win_step=0):
    """
    linear cross-correlation (max Pearson's coefficient with lag)
    between two time series

    shifts the sig2 from negative to positive lag:
    tau<0: sig2 -> sig1
    tau>0: sig1 -> sig2

    Parameters
    ----------
    sig1: np.array
        first time series (int, float)
    sig2: np.array
        second time series (int, float)
    winL int
        width of correlation win in samples
    win_step: int
        step of win in samples
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift (int)

    Returns
    -------
    xcorr: list
        maximum linear correlation in shift
    tau: float
        shift of maximum correlation in samples,
         a value in range <-lag,+lag> (float)

    Example
    -------
    xcorr,tau = compute_xcorr(sig1,sig2,2500,250,200,20)
    """

    # TODO: do not use lists - use numpy instead
    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    if win > len(sig1) or win <= 0:
        win = len(sig1)

    if win_step <= 0:
        win_step = 1

    nstep = int((len(sig1) - win) / win_step)
    nstep_lag = int(lag * 2 / lag_step)

    if nstep <= 0:
        nstep = 1

    xcorr = []
    tau = []
    for i in range(0, nstep):
        ind1 = i * win_step
        ind2 = ind1 + win

        if ind2 <= len(sig1):
            sig1_w = sig1[ind1:ind2]
            sig2_w = sig2[ind1:ind2]

            sig1_wl = sig1_w[lag:len(sig1_w) - lag]

            lincorr = []
            for i in range(0, nstep_lag + 1):
                ind1 = i * lag_step
                ind2 = ind1 + len(sig1_wl)

                sig2_wl = sig2_w[ind1:ind2]

                corr_val = np.corrcoef(sig1_wl, sig2_wl)
                lincorr.append(corr_val[0][1])

            tau_ind = lincorr.index(max(lincorr))
            tau.append(tau_ind * lag_step - lag)
            xcorr.append(np.max(lincorr))

    return xcorr, tau


def compute_spect_multp(sig1, sig2):
    """
    multiply spectra of two time series and transforms it back to time domain, 
    where the mean and std is calculated

    Parameters
    ----------
    sig1: np.array
        first time series (float)
    sig2: np.array
        second time series (float)

    Returns
    -------
    sig_sm_mean: float
        aritmetic mean value of multiplied signals
    sig_sm_std: float
        standard deviation of multiplied signals

    Example
    -------
    signal = spect_multp(sig1, sig2)
    """

    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    fft_1 = np.fft.rfft(sig1)
    fft_2 = np.fft.rfft(sig2)
    fft_p = np.multiply(fft_1, fft_2)

    sig_sm = np.abs(hilbert(np.fft.irfft(fft_p)))

    sig_sm_mean = np.mean(sig_sm)
    sig_sm_std = np.std(sig_sm)

    return sig_sm_mean, sig_sm_std


def compute_relative_entropy(sig1, sig2):
    """
    Calculation of Kullback-Leibler divergence:
    relative entropy of sig1 with respect to sig2
    and relative entropy of sig2 with respect to sig1

    Parameters
    ----------
    sig1: np.array
        first time series (float)
    sig2: np.array
        second time series (float)

    Returns
    -------
    ren: float
        max value of relative entropy between sig1 and sig2

    Example:
    -------
    ren12, ren21 = compute_relative_entropy(sig1, sig2)
    """

    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    h1 = np.histogram(sig1, 10)
    h2 = np.histogram(sig2, 10)

    ren = entropy(h1[0], h2[0])
    ren21 = entropy(h2[0], h1[0])

    if ren21 > ren:
        ren = ren21

    if ren == float('Inf'):
        ren = np.nan

    return ren


def compute_phase_sync(sig1, sig2):
    """
    Calculation of phase synchronization using Hilbert transf. 
    {Quiroga et al. 2008} sensitive to phases, irrespective of the amplitude
     and phase shift pre-filtering of the signals is necessary

    Parameters
    ----------
    sig1: np.array
        first time series (float)
    sig2: np.array
        second time series (float)

    Returns
    -------
    phase_sync: float
        ranges between 0 and 1 (1 for the perfect synchronization)

    Example
    -------
    phs = compute_phase_sync(sig1, sig2)
    """

    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    sig1_ph = angle(hilbert(sig1))
    sig2_ph = angle(hilbert(sig2))

    ph_12 = sig1_ph - sig2_ph
    phase_sync = sqrt(mean(np.cos(ph_12))**2 + mean(np.sin(ph_12))**2)
    # {Quiroga et al. 2008, equation 17 and 18}

    return phase_sync


def compute_phase_const(sig1, sig2, lag, lag_step):
    """

    **under development**

    calculation of phase consistency between two signals
    irrespective of the amplitude
    pre-filtering of the signals is necessary
    use appropriate lag and step (it calculates phase_const between single
    lag steps in whole length of given time signals)

    Parameters
    ----------
    sig1: np.array
        first time series (float)
    sig2: np.array
        second time series (float)

    Returns
    -------
    phase_const: float
        ranges between 0 and 1 
        (1 for the phase lock which does not shift during the whole time period)

    Example
    -------
    phsc = compute_phase_const(sig1, sig2)
    """

    # TODO: example is not correct - needs lag and lag_step

    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    nstep = int((len(sig1) - lag) / lag_step)

    phs_sync_temp = []
    for i in range(0, nstep):
        ind1 = i * lag_step
        ind2 = ind1 + lag

        if ind2 >= len(sig1):
            continue

        sig1_w = sig1[ind1:ind2]
        sig2_w = sig2[ind1:ind2]

        sig1_ph = np.unwrap(angle(hilbert(sig1_w)))
        sig2_ph = np.unwrap(angle(hilbert(sig2_w)))
        ph_12 = sig1_ph - sig2_ph
        phs_sync_temp.append(
            sqrt(mean(np.cos(ph_12))**2 + mean(np.sin(ph_12))**2))

    phase_const = (1 - np.std(phs_sync_temp) / 0.5) * np.mean(phs_sync_temp)

    return phase_const


def compute_pli(sig1, sig2, lag, lag_step, win=0, win_step=0):
    """
    phase-lag index {Stam et al. 2007}

    - filter signal before pli calculation (if one is filtered and the other 
      is not (or in different f-band), it can return fake high pli, which is 
      caused by substraction (difference) of inst. phases at different scales)
    - use appropriate win and lag (max lag ~= fs/2*fmax, else it runs over one 
      period and finds pli=1)

    Parameters
    ----------
    sig1: np.array
        first time series (float)
    sig2: np.array
        second time series (float)

    Returns
    -------
    pli: float
        ranges between 0 and 1 (1 for the best phase match between signals)
    tau: int
        phase lag for max pli value (in samples, 0 means no lag)

    Example
    -------
    pli, tau = compute_pli(sig1,sig2)
    """

    # TODO: example is not correct
    # TODO: print out warnings if conditions are met for warnings in doc string

    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    if win > len(sig1) or win <= 0:
        win = len(sig1)

    if win_step <= 0:
        win_step = 1

    nstep = int((len(sig1) - win) / win_step)
    nstep_lag = int(lag * 2 / lag_step)

    if nstep <= 0:
        nstep = 1

    pli = []
    tau = []
    for i in range(0, nstep):
        ind1 = i * win_step
        ind2 = ind1 + win

        if ind2 > len(sig1):
            continue

        sig1_w = sig1[ind1:ind2]
        sig2_w = sig2[ind1:ind2]

        sig1_wl = sig1_w[lag:len(sig1_w) - lag]

        pli_temp = []
        for i in range(0, nstep_lag + 1):
            ind1 = i * lag_step
            ind2 = ind1 + len(sig1_wl)

            sig2_wl = sig2_w[ind1:ind2]

            sig1_ph = np.unwrap(angle(hilbert(sig1_wl)))
            sig2_ph = np.unwrap(angle(hilbert(sig2_wl)))
            dph = sig1_ph - sig2_ph

            if np.max(dph) == np.min(dph):
                pli_temp.append(1)
                continue

            pli_temp.append(np.abs(np.mean(np.sign(dph))))

        tau_ind = pli_temp.index(max(pli_temp))
        tau.append(tau_ind * lag_step - lag)
        pli.append(np.max(pli_temp))

    return pli, tau
