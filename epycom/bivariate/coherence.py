# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
from scipy.signal import coherence

# Local imports
from ..utils.method import Method


def compute_coherence(sig, fs=5000, fband=[1.0, 4.0], lag=0, lag_step=0, win=0,
                      win_step=0, fft_win=1):
    """
    Magnitude squared coherence between two time series (raw,
    not filtered signals)
    
    When win and win_step is not 0, calculates evolution of coherence
    
    When win>len(sig) or win<=0, calculates only one coherence value

    When lag and lag_step is not 0, shifts the sig[1] from negative
    to positive lag and takes the max coherence (best fit)

    Parameters
    ----------
    sig: np.array
        2D numpy array of shape (signals, samples), time series (int, float)
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
        tau<0: sig[1] -> sig[0]
        tau>0: sig[0] -> sig[1]

    Example
    -------
    max_coh,tau = compute_coherence(sig, fs=5000, fband=[1.0,4.0], lag=0,
                                    lag_step=0, win=0, win_step=0, fft_win=1)
    """

    if type(sig) != np.ndarray:
        raise TypeError(f"Signals have to be in numpy arrays!")

    if win > len(sig) or win <= 0:
        win = len(sig)
        win_step = 1

    if win_step <= 0:
        win_step = 1

    nstep = int((len(sig) - win) / win_step)
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

        if ind2 <= len(sig):

            # OPTIMIZE: check if we can do this in one array

            sig1_w = sig[0][ind1:ind2]
            sig2_w = sig[1][ind1:ind2]

            sig1_wl = sig1_w[lag:len(sig1_w) - lag]

            coh_win = []
            for i in range(0, nstep_lag + 1):
                ind1 = i * lag_step
                ind2 = ind1 + len(sig1_wl)

                sig2_wl = sig2_w[ind1:ind2]

                f, coh = coherence(sig1_wl, sig2_wl, fs, nperseg=fft_win)
                coh_win.append(np.mean(coh[fc1:fc2]))

            tau_ind = coh_win.index(max(coh_win))
            tau.append(tau_ind * lag_step - lag)
            max_coh.append(np.max(coh_win))

    return max_coh[0], tau[0]


class Coherence(Method):

    def __init__(self, **kwargs):
        """
        Magnitude squared coherence between two time series (raw,
        not filtered signals)
        
        When win and win_step is not 0, calculates evolution of coherence
        
        When win>len(sig) or win<=0, calculates only one coherence value

        When lag and lag_step is not 0, shifts the sig[1] from negative
        to positive lag and takes the max coherence (best fit)

        Parameters
        ----------
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
        """

        super().__init__(compute_coherence, **kwargs)

        self.algorithm = 'COHERENCE'
        self.version = '1.0.0'
        self.dtype = [('max_coh', 'float32'),
                      ('tau', 'float32')]