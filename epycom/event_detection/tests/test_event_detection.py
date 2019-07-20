# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
import numpy as np

# Local imports
from epycom.event_detection.spike.barkmeier_detector import detect_spikes

from epycom.event_detection.hfo.ll_detector import detect_hfo_ll
from epycom.event_detection.hfo.rms_detector import detect_hfo_rms
from epycom.event_detection.hfo.hilbert_detector import detect_hfo_hilbert
from epycom.event_detection.hfo.cs_detector import detect_hfo_cs_beta


# ----- Spikes -----
def test_detect_spikes(create_testing_eeg_data):
    data = create_testing_eeg_data
    fs = 5000
    spike_df = detect_spikes(data, fs, 10)

    assert int(spike_df.loc[0, 'event_peak']) == 20242
    assert round(spike_df.loc[0, 'event_amp'], 5) == 1368.23346
    assert round(spike_df.loc[0, 'left_amp'], 5) == 1517.99373
    assert spike_df.loc[0, 'left_dur'] == 0.05
    assert round(spike_df.loc[0, 'right_amp'], 5) == 1486.87514
    assert spike_df.loc[0, 'right_dur'] == 0.0376


# ----- HFO -----
def test_detect_hfo_ll(create_testing_eeg_data):
    data = create_testing_eeg_data
    fs = 5000
    low_fc = 80
    high_fc = 600
    threshold = 3
    window_size = 1 / 80
    window_overlap = 0.25

    hfo_df = detect_hfo_ll(data, fs, low_fc, high_fc, threshold,
                           window_size, window_overlap)

    assert hfo_df.loc[0, 'event_start'] == 5040
    assert hfo_df.loc[0, 'event_stop'] == 5199

    assert hfo_df.loc[1, 'event_start'] == 34992
    assert hfo_df.loc[1, 'event_stop'] == 35135


def test_detect_hfo_rms(create_testing_eeg_data):
    data = create_testing_eeg_data
    fs = 5000
    low_fc = 80
    high_fc = 600
    threshold = 3
    window_size = 1 / 80
    window_overlap = 0.25

    hfo_df = detect_hfo_rms(data, fs, low_fc, high_fc, threshold,
                            window_size, window_overlap)

    assert hfo_df.loc[0, 'event_start'] == 5040
    assert hfo_df.loc[0, 'event_stop'] == 5207

    assert hfo_df.loc[1, 'event_start'] == 35010
    assert hfo_df.loc[1, 'event_stop'] == 35132


def test_detect_hfo_hilbert(create_testing_eeg_data):
    data = create_testing_eeg_data
    fs = 5000
    low_fc = 80
    high_fc = 600
    threshold = 7

    hfo_df = detect_hfo_hilbert(data, fs, low_fc, high_fc, threshold)

    assert int(hfo_df.loc[0, 'event_start']) == 5056
    assert int(hfo_df.loc[0, 'event_stop']) == 5123

    assert int(hfo_df.loc[1, 'event_start']) == 35028
    assert int(hfo_df.loc[1, 'event_stop']) == 35063


def test_detect_hfo_cs_beta(create_testing_eeg_data):
    data = create_testing_eeg_data
    fs = 5000
    low_fc = 40
    high_fc = 1000
    threshold = 0.1
    band_detections = False

    hfo_df = detect_hfo_cs_beta(data, fs, low_fc, high_fc, threshold,
                                band_detections)

    # Only the second HFO is caught by CS (due to signal artificiality)
    assert int(hfo_df.loc[0, 'event_start']) == 34992
    assert int(hfo_df.loc[0, 'event_stop']) == 35090

