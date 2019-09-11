# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
import pickle

# Third pary imports
import numpy as np

# Local imports
from ..utils.method import Method

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
    ps = ps**2 # power spectral density
    ps = ps / sum(ps)  # normalized to probability density function

    pse = -sum(ps * np.log2(ps))  # power spectral entropy

    return pse


class PowerSpectralEntropy(Method):
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

    def __init__(self):
        super().__init__(compute_pse)

        self.algorithm = 'POWER_SPECTRAL_ENTROPY'
        self.version = '1.0.0'
        self.dtype = [('pse', 'float32')]