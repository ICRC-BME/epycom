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
from epycom.univariate.univariate_methods import (compute_signal_stats,
                                                  compute_fac,
                                                  compute_pac,
                                                  comute_pse)


def test_signal_stats(get_ieeg_data):
    ch = get_ieeg_data['data'][-1]
    stats = compute_signal_stats(ch)

    expected_vals = [6863666.519704277,
                     4022165.30204,
                     2119936.0,
                     187717401.0,
                     0.0,
                     625681.0,
                     5475600.0]

    for exp_stat, stat in zip(expected_vals, stats):
        assert round(exp_stat, 5) == round(stat, 5)


def test_fac(get_ieeg_data):
    ch = get_ieeg_data['data'][-1]
    assert round(compute_fac(ch, 5000), 5) == round(-0.24878536953248276, 5)


def test_pac(get_ieeg_data):
    ch = get_ieeg_data['data'][-1]
    assert round(compute_pac(ch, 5000), 5) == round(0.30791423722563455, 5)


def test_pse(get_ieeg_data):
    ch = get_ieeg_data['data'][-1]
    assert round(comute_pse(ch), 5) == round(7.639563163873471, 5)
