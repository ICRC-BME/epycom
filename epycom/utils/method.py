# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports
import multiprocessing as mp
import warnings
import inspect
from itertools import chain

# Third pary imports
import numpy as np
import numpy.lib.recfunctions as rfn

# Local imports

"""
Basic template for method classes
"""


class Method:
    """
    Class for computations using underlying compute function

    Attributes
    ----------
    algorithm: str
        Algorithm name
    version: str
        Current version of the algorithm
    dtype: np.dtype
        Numpy data type of the results returned by run_windowed method

    Methods
    -------
    compute(data)

    run_windowed(data, window_size=5000, overlap=None, n_cores=None)
    """
    def __init__(self, compute_function, **kwargs):
        self.algorithm = ''
        self.version = ''
        self.dtype = []

        self._params = kwargs
        self._compute_function = compute_function
        self._check_params()

        self._window_indices = True

        return

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params
        self._check_params()

    def compute(self, data):
        """
        Function to run underlying compute function.

        Parameters
        ----------
        data: numpy.ndarray
            Data to analyze. Either 1D or 2D where shape = (signals, samples)

        Returns
        -------
        result: float | tuple
            Result of the compute function
        """
        if np.any(data == np.nan):
            warnings.warn(RuntimeWarning,
                          "Detected NaN in time series returning Nan")
            return np.nan

        return self._compute_function(data, **self._params)

    def _check_params(self):
        func_sig = inspect.signature(self._compute_function)
        keys_to_pop = []
        if self._compute_function is not None:
            for key in self._params.keys():
                if key not in func_sig.parameters.keys():
                    warnings.warn(f"Unrecognized keyword argument {key}.\
                                    It will be ignored",
                                  RuntimeWarning)
                    keys_to_pop.append(key)
        for key in keys_to_pop:
            self._params.pop(key)

    def run_windowed(self, data, window_size=5000, overlap=None, n_cores=None):
        """
        Function using sliding window for computing features.

        Parameters
        ----------
        data: numpy.ndarray
            Data to analyze. Either 1D or 2D where shape = (signals, samples)
        window_size: int
            Sliding window size in samples. Default=5000
        overlap: float
            Fraction of the window overlap <0, 1>. Default=0
        n_cores: None | int
            Number of cores to use in multiprocessing. When None,
            multiprocessing is not used. Default=None

        Returns
        -------
        results array: numpy.ndarray
            Array with the results of processing with dtype = self.dtype
        """

        # Take care of parameters
        data = np.squeeze(data)

        if overlap is None:
            overlap = 0

        overlap_samp = window_size * overlap

        # Calculate window indices
        n_windows = int(np.floor((np.max(data.shape) - window_size)
                        / (window_size - overlap_samp)) + 1)
        window_idxs = np.empty(n_windows, [('event_start', 'int32'),
                                           ('event_stop', 'int32')])
        window_idxs['event_start'] = (np.arange(n_windows) * window_size
                                      - np.arange(n_windows) * overlap_samp)
        window_idxs['event_stop'] = window_idxs['event_start'] + window_size
        if n_cores is None or mp.cpu_count() < 2:
            results = []

            for ci, idx in enumerate(window_idxs):
                if not self._window_indices:
                    self._params['sample_offset'] = idx[0]
                if data.ndim > 1:
                    results.append(self.compute(data[:, idx[0]: idx[1]]))
                else:
                    results.append(self.compute(data[idx[0]: idx[1]]))
        else:
            if n_cores > mp.cpu_count():
                n_cores = mp.cpu_count()
                warnings.warn(f"Maximum number of cores is {mp.cpu_count()}",
                              RuntimeWarning)
            pool = mp.Pool(n_cores)

            chunks = []
            for idx in window_idxs:
                if not self._window_indices:
                    self._params['sample_offset'] = idx[0]
                if data.ndim > 1:
                    chunks.append((data[:, idx[0]: idx[1]]))
                else:
                    chunks.append((data[idx[0]: idx[1]]))

            # TODO: pass parameters into comput function!!!!
            results = pool.map(self.compute, chunks)

            pool.close()

        # Output dtype
        if self._window_indices:
            output = rfn.merge_arrays([window_idxs,
                                       np.array(results, self.dtype)],
                                      flatten=True,
                                      usemask=False)
        else:
            # Make sure the results list is flat
            results = list(chain.from_iterable(results))
            output = np.array(results, self.dtype)

        return output
