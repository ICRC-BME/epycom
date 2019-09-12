# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.


# Std imports

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
    res = round(benchmark(compute_instance.compute,
                          create_testing_data)[0][0], 5)
    assert (res == 0)


def test_spect_multp(create_testing_data, benchmark):
    compute_instance = SpectraMultiplication()
    sm_mean, sm_std = benchmark(compute_instance.compute,
                                create_testing_data)
    sm_mean = round(sm_mean, 5)
    sm_std = round(sm_std, 5)

    assert ((sm_mean, sm_std) == (70522.64105, 35728.93925))


def test_relative_entropy(create_testing_data, benchmark):
    compute_instance = RelativeEntropy()
    res = round(benchmark(compute_instance.compute,
                          create_testing_data), 5)
    assert (res == 0.17262)


def test_phase_sync(create_testing_data, benchmark):
    compute_instance = PhaseSynchrony()
    res = round(benchmark(compute_instance.compute,
                          create_testing_data), 5)
    assert (res == 1.0)


def test_phase_const(create_testing_data, benchmark):
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    compute_instance = PhaseConsistency(lag=lag, lag_step=lag_step)
    res = round(benchmark(compute_instance.compute,
                          create_testing_data), 5)
    assert (res == 0.41204)


def test_pli(create_testing_data, benchmark):
    lag = int((5000 / 100) / 2)
    lag_step = int(lag / 10)
    compute_instance = PhaseLagIndex(lag=lag, lag_step=lag_step)
    res = round(benchmark(compute_instance.compute,
                          create_testing_data)[0][0], 5)
    assert (res == 1.0)
