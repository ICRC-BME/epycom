# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
from scipy.signal import hilbert

# Local imports
from ..utils.method import Method


def compute_pli(sig, lag=500, lag_step=50, win=0, win_step=0):
    """
    Phase-lag index.

    - filter signal before pli calculation (if one is filtered and the other 
      is not (or in different f-band), it can return fake high pli, which is 
      caused by substraction (difference) of inst. phases at different scales)
    - use appropriate win and lag (max lag ~= fs/2*fmax, else it runs over one 
      period and finds pli=1)

    Parameters
    ----------
    sig: np.array
        2D numpy array of shape (signals, samples), time series (float)
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
    pli, tau = compute_pli(sig, lag=500, lag_step=50)

    References
    ----------
    [1] C. J. Stam and J. C. Reijneveld, “Graph theoretical analysis of 
    complex networks in the brain,” Nonlinear Biomed. Phys., vol. 1, no. 1, 
    p. 3, 2007.
    """

    # TODO: print out warnings if conditions are met for warnings in doc string

    if type(sig) != np.ndarray:
        raise TypeError(f"Signals have to be in numpy arrays!")

    if win > sig.shape[1] or win <= 0:
        win = sig.shape[1]

    if win_step <= 0:
        win_step = 1

    nstep = int((sig.shape[1] - win) / win_step)
    nstep_lag = int(lag * 2 / lag_step)

    if nstep <= 0:
        nstep = 1

    pli = []
    tau = []
    for i in range(0, nstep):
        ind1 = i * win_step
        ind2 = ind1 + win

        if ind2 > sig.shape[1]:
            continue

        # OPTIMIZE: check if we can do this in one array
        sig1_w = sig[0][ind1:ind2]
        sig2_w = sig[1][ind1:ind2]

        sig1_wl = sig1_w[lag:len(sig1_w) - lag]

        pli_temp = []
        for i in range(0, nstep_lag + 1):
            ind1 = i * lag_step
            ind2 = ind1 + len(sig1_wl)

            sig2_wl = sig2_w[ind1:ind2]

            sig1_ph = np.unwrap(np.angle(hilbert(sig1_wl)))
            sig2_ph = np.unwrap(np.angle(hilbert(sig2_wl)))
            dph = sig1_ph - sig2_ph

            if np.max(dph) == np.min(dph):
                pli_temp.append(1)
                continue

            pli_temp.append(np.abs(np.mean(np.sign(dph))))

        tau_ind = pli_temp.index(max(pli_temp))
        tau.append(tau_ind * lag_step - lag)
        pli.append(np.max(pli_temp))

    return pli[0], tau[0]


class PhaseLagIndex(Method):
    
    def __init__(self, **kwargs):
        """
        Phase-lag index.

        - filter signal before pli calculation (if one is filtered and the
          other is not (or in different f-band), it can return fake high pli,
          which is caused by substraction (difference) of inst. phases at
           different scales)
        - use appropriate win and lag (max lag ~= fs/2*fmax, else it runs over
          one period and finds pli=1)

        Parameters
        ----------
        lag: int
            negative and positive shift of time series in samples
        lag_step: int
            step of shift in samples
        win: int
            width of correlation win in samples
        win_step: int
            step of win in samples
        
        References
        ----------
        [1] C. J. Stam and J. C. Reijneveld, “Graph theoretical analysis of 
        complex networks in the brain,” Nonlinear Biomed. Phys., vol. 1, no. 1, 
        p. 3, 2007.
        """

        # TODO: print out warnings if conditions for warnings in doc string

        super().__init__(compute_pli, **kwargs)

        self.algorithm = 'PHASE_LAG_INDEX'
        self.version = '1.0.0'
        self.dtype = [('pli', 'float32'),
                      ('tau', 'float32')]