# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import pytest

# Local imports
from epycom.univariate.univariate_methods import (compute_signal_stats,
                                                  compute_fac,
                                                  compute_pac,
                                                  comute_pse)


def test_signal_stats(create_testing_data):
    ch = create_testing_data
    stats = compute_signal_stats(ch)

    expected_vals = [6.68954,
                     5.0,
                     2.32213,
                     67.65263,
                     0.0,
                     0.49719,
                     7.05092]

    for exp_stat, stat in zip(expected_vals, stats):
        assert round(stat, 5) == exp_stat


def test_fac(create_testing_data):
    ch = create_testing_data
    assert round(compute_fac(ch, 5000), 5) == -0.00019


def test_pac(create_testing_data):
    ch = create_testing_data
    assert round(compute_pac(ch, 5000), 5) == 0.01189


def test_pse(create_testing_data):
    ch = create_testing_data
    assert round(comute_pse(ch), 5) == 4.32193
