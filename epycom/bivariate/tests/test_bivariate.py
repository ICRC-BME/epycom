# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports
from math import isclose

# Third pary imports

# Local imports
from epycom.bivariate import (LinearCorrelation,
                              SpectraMultiplication,
                              RelativeEntropy,
                              PhaseSynchrony,
                              PhaseConsistency,
                              PhaseLagIndex)


def test_lincorr(create_testing_data, benchmark):
    compute_instance = LinearCorrelation()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)

    assert isclose(res[0][2], 0, abs_tol=10-6)


def test_spect_multp(create_testing_data, benchmark):
    compute_instance = SpectraMultiplication()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)

    assert isclose(res[0][2], 70522.64105, abs_tol=10-6)
    assert isclose(res[0][3], 35728.93925, abs_tol=10-6)


def test_relative_entropy(create_testing_data, benchmark):
    compute_instance = RelativeEntropy()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)

    assert isclose(res[0][2], 0.17262, abs_tol=10-6)


def test_phase_sync(create_testing_data, benchmark):
    compute_instance = PhaseSynchrony()
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    assert isclose(res[0][2], 1.0, abs_tol=10-6)


def test_phase_const(create_testing_data, benchmark):
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    compute_instance = PhaseConsistency(lag=lag, lag_step=lag_step)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    assert isclose(res[0][2], 0.41203687, abs_tol=10-6)


def test_pli(create_testing_data, benchmark):
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
<<<<<<< HEAD
    compute_instance = PhaseLagIndex(lag=lag, lag_step=lag_step)
    res = benchmark(compute_instance.run_windowed,
                    create_testing_data, 50000)
    assert isclose(res[0][2], 1.0, abs_tol=10-6)
=======
    assert (round(compute_pli(ch_1, ch_2, lag, lag_step)[0][0], 5)
            == 1.0)


def test_nonlinear_correlation(create_testing_data):
    noncorr, tau2 = compute_nonlinear_corr(create_testing_data[0], create_testing_data[1], 200, 20, 2500, 250,  100)
    assert (round(noncorr[0], 5) == 0.23222)
    noncorr, tau2 = compute_nonlinear_corr(create_testing_data[0], create_testing_data[1], )
    assert (round(noncorr[0], 5) == 0.00242)
    noncorr, tau2 = compute_nonlinear_corr(create_testing_data[0], abs(create_testing_data[0]), )
    assert (round(noncorr[0], 5) == 0.9952)
    noncorr, tau2 = compute_nonlinear_corr(abs(create_testing_data[0]), create_testing_data[0])
    assert (round(noncorr[0], 5) == 0.0)
>>>>>>> origin/master
