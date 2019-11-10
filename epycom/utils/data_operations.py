# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
import pandas as pd

# Local imports


def calculate_absolute_samples(computation_result, window_size, overlap):
    """
    Convenience function to calculate absolute samples for the output of
    run_windowed class method.

    Parameters
    ----------
    computation_result: numpy.ndarray
        Output of run_windowed method
    window_size: int
        Size of the window in samples
    overlap: float
        Value between 0 and 1 specifying the fraction of the window to overlap

    Returns
    -------
    absolute_samples: numpy.ndarray
        Structured array with window starts
    """

    return np.floor(computation_result['win_idx']
                    * window_size * (1 - overlap)).astype(np.int32)


def create_output_df(fields={}):
    """
    Function to create a custom pandas dataframe depending on the algorithm
    needs. Fields: event_start,event_stop are preset.

    Parameters
    ----------
    fields: dict
        Dictionary with keys as field names and values as data types

    Returns
    -------
    output_df: pandas dataframe
        Pandas dataframe with specified fields and field dtypes
    """

    dtype_dict = {'event_start': np.int32,
                  'event_stop': np.int32}

    dtype_dict.update(fields)

    out_df = pd.DataFrame(columns=dtype_dict.keys())
    out_df = out_df.astype(dtype=dtype_dict)

    return out_df


def add_metadata(df, metadata):
    """
    Convenience function to add metadata to the output dataframe.

    Parameters
    ----------
    df: pandas dataframe
        Dataframe with original data
    metadata: dict
        Dictionary with column_name:value

    Returns
    -------
    new_df: pandas dataframe
        Updated dataframe
    """

    for key in metadata.keys():
        df[key] = metadata[key]

    return df
