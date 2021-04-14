# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Third pary imports
import numpy as np

# Local imports
from ..utils.method import Method

def _phi(data, m, N, r):
    z = N - m + 1
    x = np.array([data[i: i + m] for i in range(N - m + 1)])
    C = np.array([np.sum(np.abs(xi - x).max(axis=1) <= r, axis=0)
                  / z for xi in x])
    
    return np.log(C).sum() / z

def _approximate_entropy(sig, r, m=2):
    """
    Function computes approximate entropy of given signal

    Parameters
    ----------
    sig: np.ndarray
        1D signal
    r: np.float64
        filtering treshold, recommended values: (0.1-0.25)*np.nanstd(sig)
    m: int
        window length of compared run of data, recommended (2-8)

    Returns
    -------
    entro: numpy.float64
        approximate entropy

    Example
    -------
    signal_entropy = approximate_entropy(data, 0.1*np.nanstd(data))
    """
    
    sig = np.array(sig)
    N = sig.shape[0]

    entro = abs(_phi(sig, m + 1, N, r) - _phi(sig, m, N, r))
    return entro

def compute_approximate_entropy(sig, r, m=2, window_size=100,
                                window_overlap=1):
    """
    Function computes approximate entropy of given signal

    Parameters
    ----------
    sig: np.ndarray
        1D signal
    r: np.float64
        filtering treshold, recommended values: (0.1-0.25)*np.nanstd(sig)
    m: int
        window length of compared run of data, recommended (2-8)
    window_size: int
        Sliding window size in samples
    window_overlap: float
        Fraction of the window overlap (0 to 1)

    Returns
    -------
    entro: numpy.float64
        maximum approximate entropy in the given statistical window

    Example
    -------
    signal_entropy = compute_aproximate_entropy(data, 0.1*np.nanstd(data))
    
    Note
    ----
    For appropriate choice of parameters see:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6549512/
    """

    # Calculate window values for easier operation
    window_increment = int(np.ceil(window_size * window_overlap))
    
    # Overlapping window

    win_start = 0
    win_stop = window_size
    n_windows = int(np.ceil((len(sig) - window_size) / window_increment)) + 1
    ae = np.empty(n_windows)
    ae_i = 0
    while win_start < len(sig):
        if win_stop > len(sig):
            win_stop = len(sig)

        ae[ae_i] = _approximate_entropy(sig[int(win_start):int(win_stop)],
                                        r, m)

        if win_stop == len(sig):
            break

        win_start += window_increment
        win_stop += window_increment

        ae_i += 1
        
    return np.max(ae)


class ApproximateEntropy(Method):

    algorithm = 'APPROXIMATE_ENTROPY'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('apen', 'float32')]

    def __init__(self, **kwargs):
        """
        Approximate entropy

        Parameters
        ----------
        sig: np.ndarray
            1D signal
        m: int
            window length of compared run of data, recommended (2-8)
        r: float64
            filtering treshold, recommended values: (0.1-0.25)*std
        window_size: int
            Sliding window size in samples
        window_overlap: float
            Fraction of the window overlap (0 to 1)
            
        Note
        ----
        For appropriate choice of parameters see:
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6549512/
       """

        super().__init__(compute_approximate_entropy, **kwargs)
        self._event_flag = False

