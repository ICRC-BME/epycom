# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
from numpy import angle, mean, sqrt
from scipy.signal import hilbert, coherence
from scipy.stats import entropy

# Local imports


def compute_lincorr(sig1, sig2, lag=0, lag_step=0, win=0, win_step=0):
    """
    Linear correlation (Pearson's coefficient) between two time series
    
    When win and win_step is not 0, calculates evolution of correlation
    
    When win>len(sig) or win<=0, calculates only one corr coef

    When lag and lag_step is not 0, shifts the sig2 from negative
    to positive lag and takes the max correlation (best fit)

    Parameters
    ----------
    sig1: np.array
        first time series (int, float)
    sig2: np.array
        second time series (int, float)
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift
    win: int
        width of correlation win in samples
    win_step: int
        step of win in samples

    Returns
    -------
    lincorr: list
        maximum linear correlation in shift
    tau: float
        shift of maximum correlation in samples, 
        value in range <-lag,+lag> (float)
        tau<0: sig2 -> sig1
        tau>0: sig1 -> sig2

    Example
    -------
    lincorr,tau = compute_lincorr(sig1,sig2,200,20,2500,250)
    """

    # TODO: do not use lists - use numpy instead
    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    if win > len(sig1) or win <= 0:
        win = len(sig1)
        win_step = 1

    if win_step <= 0:
        win_step = 1

    nstep = int((len(sig1) - win) / win_step)
    if nstep <= 0:
        nstep = 1
    
    if lag == 0:
        lag_step = 1
    nstep_lag = int(lag * 2 / lag_step)

    max_corr = []
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
            max_corr.append(np.max(lincorr))

    return max_corr, tau


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
    mspect = spect_multp(sig1, sig2)
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
    and phase shift, pre-filtering of the signals is necessary

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
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift in samples

    Returns
    -------
    phase_const: float
        ranges between 0 and 1 
        (1 for the phase lock which does not shift during the whole time period)

    Example
    -------
    phsc = compute_phase_const(sig1, sig2, 500, 100)
    """

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
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift in samples
    win: int
        width of correlation win in samples
    win_step: int
        step of win in samples

    Returns
    -------
    pli: float
        ranges between 0 and 1 (1 for the best phase match between signals)
    tau: int
        phase lag for max pli value (in samples, 0 means no lag)

    Example
    -------
    pli, tau = compute_pli(sig1,sig2,lag=500,lag_step=50)
    """

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


def compute_coherence(sig1, sig2, fs, fband, lag=0, lag_step=0, win=0, win_step=0, fft_win=1):
    """
    Magnitude squared coherence between two time series (raw, not filtered signals)
    
    When win and win_step is not 0, calculates evolution of coherence
    
    When win>len(sig) or win<=0, calculates only one coherence value

    When lag and lag_step is not 0, shifts the sig2 from negative
    to positive lag and takes the max coherence (best fit)

    Parameters
    ----------
    sig1: np.array
        first time series (int, float)
    sig2: np.array
        second time series (int, float)
    fs: int, float
        sampling frequency in Hz
    fband: list
        frequency range in Hz (float)
    lag: int
        negative and positive shift of time series in samples
    lag_step: int
        step of shift in samples
    win: int
        width of correlation win in samples
    win_step: int
        step of win in samples
    fft_win: int
        length of fft window in sec

    Returns
    -------
    max_coh: list
        maximum coherence in shift
    tau: float
        shift of maximum coherence in samples, 
        value in range <-lag,+lag> (float)
        tau<0: sig2 -> sig1
        tau>0: sig1 -> sig2

    Example
    -------
    max_coh,tau = compute_coherence(sig1, sig2, fs=5000, fband=[1.0,4.0], lag=0, lag_step=0, win=0, win_step=0, fft_win=1)
    """

    # TODO: do not use lists - use numpy instead
    if len(sig1) != len(sig2):
        print('different length of signals!')
        return

    if win > len(sig1) or win <= 0:
        win = len(sig1)
        win_step = 1

    if win_step <= 0:
        win_step = 1

    nstep = int((len(sig1) - win) / win_step)
    if nstep <= 0:
        nstep = 1
    
    if lag == 0:
        lag_step = 1
    nstep_lag = int(lag * 2 / lag_step)
    
    fft_win = int(fft_win*fs)
    hz_bins = (fft_win/2)/(fs/2)
    fc1 = int(fband[0]*hz_bins)
    fc2 = int(fband[1]*hz_bins)

    max_coh = []
    tau = []
    for i in range(0, nstep):
        ind1 = i * win_step
        ind2 = ind1 + win

        if ind2 <= len(sig1):
            sig1_w = sig1[ind1:ind2]
            sig2_w = sig2[ind1:ind2]

            sig1_wl = sig1_w[lag:len(sig1_w) - lag]

            coh = []
            for i in range(0, nstep_lag + 1):
                ind1 = i * lag_step
                ind2 = ind1 + len(sig1_wl)

                sig2_wl = sig2_w[ind1:ind2]

                f, coh = coherence(sig1_wl, sig2_wl, fs, nperseg=fft_win)
                coh.append(np.mean(coh[fc1:fc2]))

            tau_ind = coh.index(max(coh))
            tau.append(tau_ind * lag_step - lag)
            max_coh.append(np.max(coh))

    return max_coh, tau
