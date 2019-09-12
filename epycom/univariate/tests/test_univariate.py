# -*def- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

# Third pary imports
import pytest

# Local imports
from epycom.univariate import (SignalStats,
                               compute_fac,
                               compute_pac,
                               PowerSpectralEntropy,
                               LyapunovExponent,
                               HjorthMobility,
                               HjorthComplexity)


def test_signal_stats(create_testing_data, benchmark):
    compute_instance = SignalStats()
    stats = benchmark(compute_instance.compute, create_testing_data)

    expected_vals = (6.68954,
                     5.0,
                     2.32213,
                     67.65263,
                     0.0,
                     0.49719,
                     7.05092)

    for exp_stat, stat in zip(expected_vals, stats):
        assert round(stat, 5) == exp_stat


def test_fac(create_testing_data, benchmark):
    res = round(benchmark(compute_fac, create_testing_data, 5000), 5)
    assert res == -0.00019


def test_pac(create_testing_data, benchmark):
    res = round(benchmark(compute_pac, create_testing_data, 5000), 5)
    assert res == 0.01189


def test_pse(create_testing_data, benchmark):
    compute_instance = PowerSpectralEntropy()
    res = round(benchmark(compute_instance.compute, create_testing_data), 5)
    assert res == 4.32193


def test_lyap_large(create_testing_data, benchmark):
    compute_instance = LyapunovExponent(sample_lag=25)
    res = round(benchmark(compute_instance.compute,
                          create_testing_data[0:5000]), 5)
    assert res == 5.79481


def test_hjorth_mobility(create_testing_data, benchmark):
    compute_instance = HjorthMobility(fs=5000)
    res = round(benchmark(compute_instance.compute, create_testing_data), 5)
    assert res == 3113.28291


def test_hjorth_complexity(create_testing_data, benchmark):
    compute_instance = HjorthComplexity(fs=5000)
    res = round(benchmark(compute_instance.compute, create_testing_data), 5)
    assert res == 2.27728
