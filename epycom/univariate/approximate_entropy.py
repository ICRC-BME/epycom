# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Third pary imports
import numpy as np

# Local imports
from ..utils.method import Method


def approximate_entropy(sig, r, m=2):
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

    def _phi(m):
        z = N - m + 1
        x = np.array([sig[i: i + m] for i in range(N - m + 1)])
        C = np.array([np.sum(np.abs(xi - x).max(axis=1) <= r, axis=0) / z for xi in x])
        return np.log(C).sum() / z

    entro = abs(_phi(m + 1) - _phi(m))
    return entro


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
       """

        super().__init__(approximate_entropy, **kwargs)
        self._event_flag = False

