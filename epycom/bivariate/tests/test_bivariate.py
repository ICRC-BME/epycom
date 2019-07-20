# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports

# Local imports
from epycom.bivariate.bivariate_methods import (compute_lincorr,
                                                compute_spect_multp,
                                                compute_relative_entropy,
                                                compute_phase_sync,
                                                compute_phase_const,
                                                compute_pli)


def test_lincorr(create_testing_data):
    ch_1 = create_testing_data[0]
    ch_2 = create_testing_data[1]
    assert (round(compute_lincorr(ch_1, ch_2)[0][0], 5)
            == 0)


def test_spect_multp(create_testing_data):
    ch_1 = create_testing_data[0]
    ch_2 = create_testing_data[1]

    sm_mean, sm_std = compute_spect_multp(ch_1, ch_2)

    assert ((round(sm_mean, 5), round(sm_std, 5))
            == (70522.64105, 35728.93925))


def test_relative_entropy(create_testing_data):
    ch_1 = create_testing_data[0]
    ch_2 = create_testing_data[1]
    assert (round(compute_relative_entropy(ch_1, ch_2), 5)
            == 0.17262)


def test_phase_sync(create_testing_data):
    ch_1 = create_testing_data[0]
    ch_2 = create_testing_data[1]
    assert (round(compute_phase_sync(ch_1, ch_2), 5)
            == 1.0)


def test_phase_const(create_testing_data):
    ch_1 = create_testing_data[0]
    ch_2 = create_testing_data[1]
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    assert (round(compute_phase_const(ch_1, ch_2, lag, lag_step), 5)
            == 0.41204)


def test_pli(create_testing_data):
    ch_1 = create_testing_data[0]
    ch_2 = create_testing_data[1]
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    assert (round(compute_pli(ch_1, ch_2, lag, lag_step)[0][0], 5)
            == 1.0)
