# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import pandas as pd
import numpy as np
import multiprocessing as mp


def get_results(inputs):
    '''
    Function for multiprocessing
    '''
    method, chunk, event_start, event_stop, fvz = inputs
    cc = method(chunk, fvz)
    apendix = pd.DataFrame([[cc, event_start, event_stop]],
                           columns=['value', 'event_start', 'event_stop'])

    return apendix


def window(data, fvz, wsize, overlap, method):

    """   function using sliding window for computing univariate features

    inputs:
        data - float64
        wsize - window size in samples
        shift in samples
        method - method to be computed in the window

    output
        feature signal


    TODO
        option to choose multiprocessing
        implement Pool.array



        go through PEP8 standard
        go through numpy docstrings

    """

    arguments = locals()

    k = int(np.floor((np.size(data)-wsize)/(wsize-overlap))+1)
    results = pd.DataFrame(columns=['value', 'event_start', 'event_stop'])
    indexes = [np.arange(k)*wsize-np.arange(k)*overlap,
               (np.arange(k)*wsize+wsize-1)-np.arange(k)*overlap]
    results['event_start'] = indexes[0]
    results['event_stop'] = indexes[1]

    chunks = list()
    for i in np.arange(len(indexes[0])):
        chunks.append((method, data[int(indexes[0][i]):int(indexes[1][i])],
                      indexes[0][i], indexes[1][i], fvz))
    
    if mp.cpu_count() < 2:
        processes = 1
    else:
        processes = mp.cpu_count() - 1
            
    
    pool = mp.Pool(processes)

    results = pool.map(get_results, chunks)

    results_df = pd.concat(results, ignore_index=True)

    return results_df


# =============================================================================
#     for i in list(range(0,k)):
#         cc=method(data[(wsize-overlap)*i:(wsize-overlap)*i+wsize],fvz)
#         apendix=pd.DataFrame([[cc, (wsize-overlap)*i,
#                (wsize-overlap)*i+wsize]],
#                 columns=['value', 'event_start', 'event_stop'])
#         results=results.append(apendix, ignore_index=True, sort=False)
# =============================================================================
