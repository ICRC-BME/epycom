# -*def- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports
from math import isclose

# Third pary imports

# Local imports
from epycom.univariate import (SignalStats,
                               PowerSpectralEntropy,
                               LyapunovExponent,
                               HjorthMobility,
                               HjorthComplexity)


def test_signal_stats(create_testing_data, benchmark):
    compute_instance = SignalStats()
    stats = benchmark(compute_instance.run_windowed,
                      create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    expected_vals = (6.68954,
                     5.0,
                     2.32213,
                     67.65263,
                     0.0,
                     0.49719,
                     7.05092)

    for exp_stat, stat in zip(expected_vals, list(stats[0])[:-1]):
        assert isclose(stat, exp_stat, abs_tol=10e-6)


def test_pse(create_testing_data, benchmark):
    compute_instance = PowerSpectralEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 4.32193, abs_tol=10e-6)


def test_lyapunov_exponent(create_testing_data, benchmark):
    compute_instance = LyapunovExponent(sample_lag=25)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data[:5000], 5000)
    compute_instance.run_windowed(create_testing_data[:5000],
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 5.794813, abs_tol=10e-6)


def test_hjorth_mobility(create_testing_data, benchmark):
    compute_instance = HjorthMobility(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 3113.283, abs_tol=10e-3)


def test_hjorth_complexity(create_testing_data, benchmark):
    compute_instance = HjorthComplexity(fs=5000)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    compute_instance.run_windowed(create_testing_data,
                                  5000,
                                  n_cores=2)
    assert isclose(res[0][0], 2.27728, abs_tol=10e-6)
