# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Third pary imports
import numpy as np

# Local imports
from ..utils.method import Method


def sample_entropy(sig, r, m=2):
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
        approximate entropy

    Example
    -------
    sample_entropy = approximate_entropy(data, 0.1*np.nanstd(data))
    """

    sig = np.array(sig)
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

        super().__init__(sample_entropy, **kwargs)
        self._event_flag = False

