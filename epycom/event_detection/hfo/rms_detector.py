# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np
from scipy.signal import butter, filtfilt

# Local imports
from epycom.utils.signal_transforms import compute_rms
from epycom.utils.thresholds import th_std
from epycom.utils.data_operations import create_output_df


def detect_hfo_rms(data, fs, low_fc, high_fc,
                   threshold, window_size, window_overlap):
    """
    Root mean square detection algorithm {Staba et al. 2002, Blanco et al 2010}.

    Parameters
    ----------
    data: numpy array
        1D array with raw data
    fs: int
        Sampling frequency
    low_fc: float
        Low cut-off frequency
    high_fc: float
        High cut-off frequency
    window_size: float
        Sliding window size in seconds
    window_overlap: float
        Fraction of the window overlap (0 to 1)

    Returns:
    --------
    df_out: pandas dataframe
        Output dataframe with detections
    """

    # Calculate window values for easier operation
    samp_win_size = int(window_size * fs)
    samp_win_inc = int(samp_win_size * window_overlap)

    # Create output dataframe
    df_out = create_output_df()

    # Filter the signal
    b, a = butter(3, [low_fc / (fs / 2), high_fc / (fs / 2)], 'bandpass')
    filt_data = filtfilt(b, a, data)

    # Transform the signal - one sample window shift

    #LL = compute_line_lenght(filt_data, window_size*fs)

    # Alternative approach - overlapping window

    win_start = 0
    win_stop = window_size * fs
    n_windows = int(np.ceil((len(data) - samp_win_size) / samp_win_inc)) + 1
    RMS = np.zeros(n_windows)
    RMS_i = 0
    while win_start < len(filt_data):
        if win_stop > len(filt_data):
            win_stop = len(filt_data)

        RMS[RMS_i] = compute_rms(filt_data[int(win_start):int(win_stop)],
                                 samp_win_size)[0]

        if win_stop == len(filt_data):
            break

        win_start += samp_win_inc
        win_stop += samp_win_inc

        RMS_i += 1

    # Create threshold
    det_th = th_std(RMS, threshold)

    # Detect
    RMS_idx = 0
    df_idx = 0
    while RMS_idx < len(RMS):
        if RMS[RMS_idx] >= det_th:
            event_start = RMS_idx * samp_win_inc
            while RMS_idx < len(RMS) and RMS[RMS_idx] >= det_th:
                RMS_idx += 1
            event_stop = (RMS_idx * samp_win_inc) + samp_win_size

            if event_stop > len(data):
                event_stop = len(data)

            # Optional feature calculations can go here

            # Write into dataframe
            df_out.loc[df_idx] = [event_start, event_stop]
            df_idx += 1

            RMS_idx += 1
        else:
            RMS_idx += 1

    return df_out
