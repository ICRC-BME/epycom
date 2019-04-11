# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

import pytest


# Std imports
import pickle

# Third pary imports
import pytest

# Local imports
from epycom.bivariate.bivariate_methods import (compute_linear_correlation,
                                                compute_xcorr,
                                                compute_spect_multp,
                                                compute_relative_entropy,
                                                compute_phase_sync,
                                                compute_phase_const,
                                                compute_pli)


def test_linear_correlation(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    assert compute_linear_correlation(ch_1, ch_2)[0] == 0.8148835654651208


def test_xcorr(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    assert compute_xcorr(ch_1, ch_2, lag, lag_step)[0][0] == 0.8173253454623266


def test_spect_multp(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    assert compute_spect_multp(ch_1, ch_2) == (74635071954.20273,
                                               12969599334.390364)


def test_relative_entropy(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    assert compute_relative_entropy(ch_1, ch_2) == 1.3346064285349755


def test_phase_sync(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    assert compute_phase_sync(ch_1, ch_2) == 0.8328288906252623


def test_phase_const(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    assert compute_phase_const(ch_1, ch_2, lag, lag_step) == 0.6394870223157648


def test_pli(get_ieeg_data):
    ch_1 = get_ieeg_data['data'][-2]
    ch_2 = get_ieeg_data['data'][-1]
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    assert compute_pli(ch_1, ch_2, lag, lag_step)[0][0] == 0.9316516516516516
