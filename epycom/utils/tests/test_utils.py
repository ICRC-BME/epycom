# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import numpy as np

# Local imports
from epycom.utils.data_operations import create_output_df, add_metadata
from epycom.univariate.univariate_methods import compute_hjorth_complexity
from epycom.utils.window_features import window
from epycom.utils.signal_transforms import (compute_hilbert_envelope,
                                            compute_hilbert_power,
                                            compute_teager_energy,
                                            compute_rms,
                                            compute_stenergy,
                                            compute_line_lenght,
                                            compute_stockwell_transform)
from epycom.utils.thresholds import th_std, th_tukey, th_percentile, th_quian


# ----- Data operations -----
def test_create_output_df():
    res_df = create_output_df(fields={'field_1': np.int32,
                                      'field_2': np.float})
    expected_columns = ['event_start', 'event_stop', 'field_1', 'field_2']
    assert expected_columns == list(res_df.columns)


def test_add_metadata():
    res_df = create_output_df(fields={'field_1': np.int32,
                                      'field_2': np.float})
    metadata = {'field_3': np.int32,
                'field_4': np.float}
    add_metadata(res_df, metadata)
    expected_columns = ['event_start', 'event_stop',
                        'field_1', 'field_2', 'field_3', 'field_4']
    assert expected_columns == list(res_df.columns)


# ----- Signal transforms -----
def test_compute_hilbert_envelope(create_testing_data):
    assert (round(np.sum(compute_hilbert_envelope(create_testing_data)), 5)
            == round(141021.90763537044, 5))


def test_compute_hilbert_power(create_testing_data):
    assert (round(np.sum(compute_hilbert_power(create_testing_data)), 5)
            == round(499812.84844509006, 5))


def test_compute_teager_energy(create_testing_data):
    assert (round(np.sum(compute_teager_energy(create_testing_data)), 5)
            == round(96410.92390890958, 5))


def test_compute_rms(create_testing_data):
    assert (round(np.sum(compute_rms(create_testing_data)), 5)
            == round(101737.24636480425, 5))


def test_compute_stenergy(create_testing_data):
    assert (round(np.sum(compute_stenergy(create_testing_data)), 5)
            == round(249993.5292416787, 5))


def test_compute_line_lenght(create_testing_data):
    assert (round(np.sum(compute_line_lenght(create_testing_data)), 5)
            == round(58085.86029191726, 5))


def test_compute_stockwell_transform(create_testing_data):
    s = compute_stockwell_transform(create_testing_data, 5000, 80, 600)[0]
    assert round(np.abs(np.sum(np.sum(s))), 5) == round(75000.00000000402, 5)


# ----- Thresholds -----
def test_th_std(create_testing_data):
    assert (round(th_std(create_testing_data, 3), 5)
            == round(6.708203932499344, 5))


def test_th_tukey(create_testing_data):
    assert (round(th_tukey(create_testing_data, 3), 5)
            == round(10.659619047273361, 5))


def test_th_percentile(create_testing_data):
    assert (round(th_percentile(create_testing_data, 75), 5)
            == round(1.5228027210391037, 5))


def test_th_quian(create_testing_data):
    assert (round(th_quian(create_testing_data, 3), 5)
            == round(6.777704219110832, 5))


# -------Window function ------
def test_window(create_testing_data):

    method_args = [{'args': [5000], 'kwargs': {}}]
    result_dataframe = window(create_testing_data,
                              5000, compute_hjorth_complexity,
                              method_args=method_args,
                              wsize=1, overlap=0.2)
    start_check = int(result_dataframe.event_start[4])
    end_check = int(result_dataframe.event_stop[4])
    assert (start_check == 16000 and end_check == 21000)
