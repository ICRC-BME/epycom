# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Third pary imports
import numpy as np

# Local imports
from ..utils.method import Method
from ..utils.tools import try_jit_decorate


def _sample_entropy(sig, r, m=2):
    """
    Function to compute sample entropy

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
    entropy: numpy.float64 (computed as -np.log(A / B))
        sample entropy
    """

    # sig = np.array(sig)
    N = sig.shape[0]

    # Split time series and save all templates of length m
    x = np.array([sig[i: i + m] for i in range(N - m)])
    x_B = np.array([sig[i: i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xi - x_B).max(axis=1) <= r) - 1 for xi in x])

    # Similar for computing A
    m += 1
    x_A = np.array([sig[i: i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xj - x_A).max(axis=1) <= r) - 1 for xj in x_A])

    return -np.log(A / B)


def compute_sample_entropy(sig, r, m=2, window_size=100,
                           window_overlap=1):
    """
    Function to compute sample entropy

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
        maximum sample entropy in the given statistical window

    Example
    -------
    sample_entropy = compute_sample_entropy(data, 0.1*np.nanstd(data))
    
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
    se = np.empty(n_windows)
    se_i = 0
    while win_start < len(sig):
        if win_stop > len(sig):
            win_stop = len(sig)

        se[se_i] = _sample_entropy(sig[int(win_start):int(win_stop)],
                                   r, m)

        if win_stop == len(sig):
            break

        win_start += window_increment
        win_stop += window_increment

        se_i += 1
        
    return np.max(se)


class SampleEntropy(Method):

    algorithm = 'SAMPLE_ENTROPY'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('sampen', 'float32')]

    def __init__(self, **kwargs):
        """
        Sample entropy

        Parameters
        ----------
        sig: np.ndarray
            1D signal
        m: int
            window length of compared run of data, recommended (2-8)
        r: float64
            filtering treshold, recommended values: (0.1-0.25)*std
        """

        super().__init__(compute_sample_entropy, **kwargs)
        self._event_flag = False

