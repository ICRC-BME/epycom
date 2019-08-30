# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports
import multiprocessing as mp
import warnings

# Third pary imports
import numpy as np

# Local imports


def _get_results(inputs):
    """Function for multiprocessing"""

    method, chunk, args, kwargs = inputs
    if chunk.ndim > 1:
        return np.array(method(chunk[0],
                               chunk[1],
                               *args, **kwargs))
    else:
        return np.array(method(chunk, *args, **kwargs))


def window(data, fs, method, method_args=None, wsize=None, overlap=0,
           n_cores=None):
    """
    Function using sliding window for computing features.

    Parameters
    ----------
    data: numpy.ndarray
        Data to analyze.
    fs: float
        Sampling frequency.
    method: object | list
        Method or list of methods for data processing. Optimized for epycom
        methods.
    method_args: dict | list
        Dictionary or list of dictionaries with arguments and keyword arguments
         for selected methods.
    wsize: float
        Sliding window size in samples. Default=None -> 1*fs
    overlap: float
        Fraction of the window overlap <0, 1>. Default=0
    n_cores: None | int
        Number of cores to use in multiprocessing. When None, multiprocessing
        is not used. Default=None

    Returns
    -------
    feature_dataframe: pandas.DataFrame
        Dataframe with the results of precessing.

    Example
    -------
    Example of providing multiple methods with specified arguments.

    >>> method = [method_a, method_b]
    >>> method_args = [{'args': [m1arg_1, m1arg_2],
                        'kwargs': {'m1kw_1': val_1, 'm1kw_2': val_2}},
                       {'args': [m2arg_1, m2arg2],
                        'kwargs': {'m2kw_1': val_1, 'm2kw_2': val_2}}]
    >>> results = window(data, fs, method, method_args)

    """

    data = np.squeeze(data)

    if isinstance(method, object) and not isinstance(method, list):
        method = [method]
        was_list = False
    else:
        was_list = True

    if wsize is None:
        wsize = fs
    overlapsamp = wsize * overlap

    k = int(np.floor((np.max(data.shape) - wsize) / (wsize - overlapsamp)) + 1)
    indexes = np.zeros([k, 2])
    indexes[:, 0] = np.arange(k) * wsize - np.arange(k) * overlapsamp
    indexes[:, 1] = indexes[:, 0] + wsize
    indexes = indexes.astype(np.int32)

    if n_cores is None or mp.cpu_count() < 2:
        method_results = []
        for mi, m in enumerate(method):
            # Allocate array for results
            results_arr = np.zeros(indexes.shape[0], object)
            
            # Construct method arguments
            if method_args is not None:
                if method_args[mi] is not None:
                    args = method_args[mi]['args']
                    kwargs = method_args[mi]['kwargs']
                else:
                    args = []
                    kwargs = {}
            else:
                args = []
                kwargs = {}
            
            for ci, idx in enumerate(indexes):
                # At the first run of the method inspect the results and
                # constructcolumns
                if data.ndim > 1:
                    vals = m(data[0, idx[0]: idx[1]],
                             data[1, idx[0]: idx[1]],
                             *args, **kwargs)
                else:
                    vals = m(data[idx[0]: idx[1]], *args, **kwargs)
                    
                results_arr[ci] = vals
                
            method_results.append(results_arr)
            
    else:
        if n_cores > mp.cpu_count():
            n_cores = mp.cpu_count()
            warnings.warn(f"Maximum number of cores is {mp.cpu_count()}",
                          RuntimeWarning)
        pool = mp.Pool(n_cores)
        
        method_results = []
        for mi, m in enumerate(method):
            
            # Construct method arguments
            if method_args is not None:
                if method_args[mi] is not None:
                    args = method_args[mi]['args']
                    kwargs = method_args[mi]['kwargs']
                else:
                    args = []
                    kwargs = {}
            else:
                args = []
                kwargs = {}
            
            chunks = []
            for idx in indexes:
                if data.ndim > 1:
                    chunks.append((m, data[:, idx[0]: idx[1]], args, kwargs))
                else:
                    chunks.append((m, data[idx[0]: idx[1]], args, kwargs))
            
            results = np.array(pool.map(_get_results, chunks))
                    
            method_results.append(results)
            
        pool.close()

    if was_list:
        return indexes, method_results
    else:
        return indexes, method_results[0]
