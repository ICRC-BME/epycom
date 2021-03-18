# -*- coding: utf-8 -*-
# Copyright (c) St. Anne's University Hospital in Brno. International Clinical
# Research Center, Biomedical Engineering. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

# Std imports

# Third pary imports
import numpy as np
import scipy
from math import nan

# Local imports
from ..utils.method import Method

import numpy as np
import warnings


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Parameters
    ----------
    target: int
        Target number for finding a regular number (must be a positive integer).

    Returns
    --------
    match/target: int
            The next regular number greater than or equal to target.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float("inf")  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            p2 = 2 ** ((quotient - 1).bit_length())

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def _acovf(x, adjusted=False, demean=True, fft=None, missing="none", nlag=None):
    """
    Estimate autocovariances. Recoded from statsmodels package.

    Parameters
    ----------
    x : array_like
        Time series data. Must be 1d.
    adjusted : bool, default False
        If True, then denominators is n-k, otherwise n.
    demean : bool, default True
        If True, then subtract the mean x from each element of x.
    fft : bool, default None
        If True, use FFT convolution.  This method should be preferred
        for long time series.
    missing : str, default "none"
        A string in ["none", "raise", "conservative", "drop"] specifying how
        the NaNs are to be treated. "none" performs no checks. "raise" raises
        an exception if NaN values are found. "drop" removes the missing
        observations and then estimates the autocovariances treating the
        non-missing as contiguous. "conservative" computes the autocovariance
        using nan-ops so that nans are removed when computing the mean
        and cross-products that are used to estimate the autocovariance.
        When using "conservative", n is set to the number of non-missing
        observations.
    nlag : {int, None}, default None
        Limit the number of autocovariances returned.  Size of returned
        array is nlag + 1.  Setting nlag when fft is False uses a simple,
        direct estimator of the autocovariances that only computes the first
        nlag + 1 values. This can be much faster when the time series is long
        and only a small number of autocovariances are needed.

    Returns
    -------
    acov: ndarray
        The estimated autocovariances.

    References
    -----------
    .. [1] Parzen, E., 1963. On spectral analysis with missing observations
           and amplitude modulation. Sankhya: The Indian Journal of
           Statistics, Series A, pp.383-392.
    """
    if not isinstance(adjusted, bool):
        raise ValueError("adjusted must be of type bool")
    if not isinstance(demean, bool):
        raise ValueError("demean must be of type bool")
    if fft is not None and not isinstance(adjusted, bool):
        raise ValueError("fft must be of type bool")
    if not isinstance(demean, str) or demean not in ["none", "raise",
                                                     "conservative", "drop"]:
        raise ValueError('demean must be on of the following strings ["none", \
                         "raise","conservative", "drop"]')
    if nlag is not None and not isinstance(nlag, bool):
        raise ValueError("nlag must be of type int")

    if fft is None:
        msg = (
            "fft=True will become the default after the release of the 0.12 "
            "release of statsmodels. To suppress this warning, explicitly "
            "set fft=False."
        )
        warnings.warn(msg, FutureWarning)
        fft = False

    if not isinstance(x, np.array) and x.ndim != 1:
        raise ValueError("x must be a onedimensional array")

    missing = missing.lower()
    if missing == "none":
        deal_with_masked = False
    else:
        deal_with_masked = np.any(np.isnan(x))
    if deal_with_masked:
        if missing == "raise":
            raise RuntimeError("NaNs were encountered in the data")
        notmask_bool = ~np.isnan(x)  # bool
        if missing == "conservative":
            # Must copy for thread safety
            x = x.copy()
            x[~notmask_bool] = 0
        else:  # "drop"
            x = x[notmask_bool]  # copies non-missing
        notmask_int = notmask_bool.astype(int)  # int

    if demean and deal_with_masked:
        # whether "drop" or "conservative":
        xo = x - x.sum() / notmask_int.sum()
        if missing == "conservative":
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    lag_len = nlag
    if nlag is None:
        lag_len = n - 1
    elif nlag > n - 1:
        raise ValueError("nlag must be smaller than nobs - 1")

    if not fft and nlag is not None:
        acov = np.empty(lag_len + 1)
        acov[0] = xo.dot(xo)
        for i in range(lag_len):
            acov[i + 1] = xo[i + 1 :].dot(xo[: -(i + 1)])
        if not deal_with_masked or missing == "drop":
            if adjusted:
                acov /= n - np.arange(lag_len + 1)
            else:
                acov /= n
        else:
            if adjusted:
                divisor = np.empty(lag_len + 1, dtype=np.int64)
                divisor[0] = notmask_int.sum()
                for i in range(lag_len):
                    divisor[i + 1] = notmask_int[i + 1 :].dot(
                        notmask_int[: -(i + 1)]
                    )
                divisor[divisor == 0] = 1
                acov /= divisor
            else:  # biased, missing data but npt "drop"
                acov /= notmask_int.sum()
        return acov

    if adjusted and deal_with_masked and missing == "conservative":
        d = np.correlate(notmask_int, notmask_int, "full")
        d[d == 0] = 1
    elif adjusted:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked:
        # biased and NaNs given and ("drop" or "conservative")
        d = notmask_int.sum() * np.ones(2 * n - 1)
    else:  # biased and no NaNs or missing=="none"
        d = n * np.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(xo, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1 :]
        acov = acov.real
    else:
        acov = np.correlate(xo, xo, "full")[n - 1 :] / d[n - 1 :]

    if nlag is not None:
        # Copy to allow gc of full array rather than view
        return acov[: lag_len + 1].copy()
    return acov


def _corrmtx(x_input, m, method='autocorrelation'):
    """
    Correlation matrix

    This function is used by PSD estimator functions. It generates
    the correlation matrix from a correlation data set and a maximum lag.
    Recorded from spectrum package.

    Parameters:
    -----------
    x_input: array
        autocorrelation samples (1D)
    m: int
        the maximum lag (Depending on the choice of the method, the 
        correlation matrix has different sizes, but the number of rows is
        always m+1)
    method: string
        'autocorrelation'- (default) X is the (n+m)-by-(m+1) rectangular
            Toeplitz matrix derived using prewindowed and postwindowed data.
        'prewindowed'- X is the n-by-(m+1) rectangular Toeplitz matrix derived
            using prewindowed data only.
        'postwindowed'- X is the n-by-(m+1) rectangular Toeplitz matrix that
            derived using postwindowed data only.
        'covariance'- X is the (n-m)-by-(m+1) rectangular Toeplitz matrix
            derived using nonwindowed data.
        'modified'- X is the 2(n-m)-by-(m+1) modified rectangular Toeplitz
            matrix that generates an autocorrelation estimate for the length n
            data vector x, derived using forward and backward prediction error
             estimates.

    Returns
    -------
    c:
        the autocorrelation matrix

    Note
    ----
    Function implemented from original library spectrum
        https://pyspectrum.readthedocs.io/en/latest/#
    """

    valid_methods = ['autocorrelation', 'prewindowed', 'postwindowed',
                     'covariance', 'modified']
    if method not in valid_methods:
        raise ValueError("Invalid method. Try one of %s" % valid_methods)

    from scipy.linalg import toeplitz

    # create the relevant matrices that will be useful to create
    # the correlation matrices
    N = len(x_input)

    # FIXME:do we need a copy ?
    if isinstance(x_input, list):
        x = np.array(x_input)
    else:
        x = x_input.copy()

    if x.dtype == complex:
        complex_type = True
    else:
        complex_type = False

    # Compute the Lp, Up and Tp matrices according to the requested method
    if method in ['autocorrelation', 'prewindowed']:
        Lp = toeplitz(x[0:m], [0] * (m + 1))
    Tp = toeplitz(x[m:N], x[m::-1])
    if method in ['autocorrelation', 'postwindowed']:
        Up = toeplitz([0] * (m + 1), np.insert(x[N:N - m - 1:-1], 0, 0))

    # Create the output matrix
    if method == 'autocorrelation':
        if complex_type:
            C = np.zeros((N + m, m + 1), dtype=complex)
        else:
            C = np.zeros((N + m, m + 1))
        for i in range(0, m):
            C[i] = Lp[i]
        for i in range(m, N):
            C[i] = Tp[i - m]
        for i in range(N, N + m):
            C[i] = Up[i - N]
    elif method == 'prewindowed':
        if complex_type:
            C = np.zeros((N, m + 1), dtype=complex)
        else:
            C = np.zeros((N, m + 1))

        for i in range(0, m):
            C[i] = Lp[i]
        for i in range(m, N):
            C[i] = Tp[i - m]
    elif method == 'postwindowed':
        if complex_type:
            C = np.zeros((N, m + 1), dtype=complex)
        else:
            C = np.zeros((N, m + 1))
        for i in range(0, N - m):
            C[i] = Tp[i]
        for i in range(N - m, N):
            C[i] = Up[i - N + m]
    elif method == 'covariance':
        return Tp
    elif method == 'modified':
        if complex_type:
            C = np.zeros((2 * (N - m), m + 1), dtype=complex)
        else:
            C = np.zeros((2 * (N - m), m + 1))
        for i in range(0, N - m):
            C[i] = Tp[i]
        Tp = np.fliplr(Tp.conj())
        for i in range(N - m, 2 * (N - m)):
            C[i] = Tp[i - N + m]

    return C


def _arcovar(x, order):
    """
    Simple and fast implementation of the covariance AR estimate.
    Recorded from spectrum package.

    Parameters
    ----------
    x: array
      Array of complex data samples
    order: int
        Order of linear prediction model

    Returns
    -------
    a: array
        Array of complex forward linear prediction coefficients
        (The output vector contains the normalized estimate of the AR system parameters)
    e: float64
        error

    Note
    ----
    Function implemented from original library spectrum
        https://pyspectrum.readthedocs.io/en/latest/#
    """

    X = _corrmtx(x, order, 'covariance')
    Xc = np.array(X[:, 1:])
    X1 = np.array(X[:, 0])

    # Coefficients estimated via the covariance method
    # Here we use lstsq rathre than solve function because Xc is not square
    # matrix

    a, _residues, _rank, _singular_values = scipy.linalg.lstsq(-Xc, X1)

    # Estimate the input white noise variance
    Cz = np.dot(X1.conj().transpose(), Xc)
    e = np.dot(X1.conj().transpose(), X1) + np.dot(Cz, a)
    assert e.imag < 1e-4, 'wierd behaviour'
    e = float(e.real)  # ignore imag part that should be small

    return a, e


def _arburg(X, order, criteria=None):
    """
    Estimate the complex autoregressive parameters by the Burg algorithm.

    .. math:: x(n) = qrt{(v}) e(n) + sum_{k=1}^{P+1} a(k) x(n-k)

    Parameters
    ----------
    X: numpy.ndarray
      array of complex data samples (length N)
    order: int
        order of autoregressive process (0<order<N)
    criteria:
            select a criteria to automatically select the order

    Returns
    -------
    a: numpy.ndarray
        Array of complex autoregressive parameters A(1) to A(order).
        First value (unity) is not included !!
    rho: numpy.float64
        Real variable representing driving noise variance (mean square
        of residual noise) from the whitening operation of the Burg filter.
    ref: numpy.ndarray
        reflection coefficients defining the filter of the model

    Example
    -------
    _, v, _ = arburg(sig, n)
    """

    if order <= 0.:
        raise ValueError("order must be > 0")

    if order > len(X):
        raise ValueError("order must be less than length input - 2")

    x = np.array(X)
    N = len(x)

    # Initialisation
    # ------ rho, den
    rho = sum(abs(x) ** 2.) / float(N)  # Eq 8.21 [Marple]_
    den = rho * 2. * N

    # p = 0
    a = np.zeros(0, dtype=complex)
    ref = np.zeros(0, dtype=complex)
    ef = x.astype(complex)
    eb = x.astype(complex)
    temp = 1.
    #   Main recursion

    for k in range(0, order):

        # calculate the next order reflection coefficient Eq 8.14 Marple
        num = sum([ef[j] * eb[j - 1].conjugate() for j in range(k + 1, N)])
        den = temp * den - abs(ef[k]) ** 2 - abs(eb[N - 1]) ** 2
        kp = -2. * num / den  # eq 8.14

        temp = 1. - abs(kp) ** 2.
        new_rho = temp * rho

        # this should be after the criteria
        rho = new_rho
        if rho <= 0:
            raise ValueError("Found a negative value (expected positive strictly) %s. Decrease the order" % rho)

        a.resize(a.size + 1, refcheck=False)
        a[k] = kp
        if k == 0:
            for j in range(N - 1, k, -1):
                save2 = ef[j]
                ef[j] = save2 + kp * eb[j - 1]  # Eq. (8.7)
                eb[j] = eb[j - 1] + kp.conjugate() * save2

        else:
            # update the AR coeff
            khalf = int((k + 1) // 2)
            for j in range(0, khalf):
                ap = a[j]  # previous value
                a[j] = ap + kp * a[k - j - 1].conjugate()  # Eq. (8.2)
                if j != k - j - 1:
                    a[k - j - 1] = a[k - j - 1] + kp * ap.conjugate()  # Eq. (8.2)

            # update the prediction error
            for j in range(N - 1, k, -1):
                save2 = ef[j]
                ef[j] = save2 + kp * eb[j - 1]  # Eq. (8.7)
                eb[j] = eb[j - 1] + kp.conjugate() * save2

        # save the reflection coefficient
        ref.resize(ref.size + 1, refcheck=False)
        ref[k] = kp

    return a, rho, ref


def _extract_poles(sig, n):
    """
    gives the complex poles from auto-regressive analysis

    Parameters
    ----------
    sig[channel, samples]: numpy.ndarray
    n: int
        requested order

    Returns
    -------
    roots: numpy.ndarray
        complex poles of the transfer function Z=exp(-u+2*Pii*w)
    v: numpy.float64
        residual variance(noise)
    A: numpy.ndarray
        complex amplitudes

    Example
    -------
     _, res, _ = _extract_poles(E[:, j], 3)
    """

    coeffs, _ = _arcovar(sig, n)  # complex forward linear prediction coefficients
    coeffs = np.append([1], coeffs)
    _, v, _ = _arburg(sig, n)  # noise variance (mean square of residual noise)
    roots = np.roots(coeffs)
    R = np.transpose(roots * np.ones([roots.shape[0], roots.shape[0]]))
    R = R - np.transpose(R) + np.identity(roots.shape[0])
    A = 1 / np.prod(R, axis=0)

    return roots, v, A


def _embed(sig, tau, nED, step=1):
    """
    creates a time-delay embeded vector-signal from a 1D signal

    Parameters
    ----------
    sig: numpy.ndarray
        1D signal
    tau: int
        <-60> embedding delay, if tau<0 minimal correlation method is used in the range [1,-tau]
    nED: float
        number od embedding dimensions
    step: int
        default 1

    Returns
    -------
    V: list
        [embedding,time] vector signal
    tau: int
        the embedding delay (if automatically computed ), if <0 no zero-cross is detected

    Example
    -------
    E, _ = embed(sig, 1, 100.0, 50)
    """

    length = len(sig)
    tau1 = tau
    if tau < 0:
        tau = np.abs(tau)
        XC = _acovf(sig, fft=False)  # estimated autocovariances
        XC = XC[1:]
        L = np.where((np.sign(XC[0:-1]) - np.sign(XC[1:])) != 0)  # indices of zero crossing
        if (len(L) > 0) and L[0] < tau:
            tau = L[0]
            tau1 = tau
        else:
            tau = np.argmin(np.abs(XC[0:tau])) + 1
            tau1 = -tau

    nv = np.floor((length - 1 - (nED - 1) * tau) / step) + 1
    V = [sig[0 + (j - 1) * tau::step][0:int(nv)] for j in range(1, int(nED + 1))]

    return V, tau1


def _get_residual(sig, n, winL):
    """
    Function calculates residuals

    Parameters
    ----------
    sig[channel,samples]: numpy.ndarray
    n: int
        requested model order
    winL: float
        window length with 50% overlap

    Returns
    -------
    res_var: list
        residual value
    """

    res_var = []

    E, _ = _embed(sig, 1, winL, round(winL / 2))
    E = np.array(E)

    for j in range(E.shape[1]):
        _, res, _ = _extract_poles(E[:, j], n)
        res_var.append(res)
        # roots.append(root)
        # ampt.append(A)

    # return np.array(roots).squeeze(), np.array(res_var), np.array(ampt).squeeze()
    return res_var


def compute_arr(sig, fs):
    """
    Function computes ARR parameters

    Parameters
    ----------
    sig: numpy.ndarray
    fs: float64
        sample frequency

    Returns
    -------
    ARRm: numpy.float64
        ARR parameters
    r1, r2, r3: list
                residuals for model order 1-3

    Example
    -------
    arrm = compute_arr(data, 5000)
    """

    # sig = stats.zscore(sig)
    winL = 0.02 * fs
    r1 = _get_residual(sig, 1, winL)
    r2 = _get_residual(sig, 2, winL)
    r3 = _get_residual(sig, 3, winL)

    # residual decline over residual from order 1 and 2 models
    rn = np.array([r1, r2])
    D = -2 * np.diff(rn, axis=0) / np.sum(rn, axis=0)
    r_clean = r3.copy()
    p95 = np.nanpercentile(r_clean, 95)
    w = D.shape[1] - 2

    while w > 0:
        if r_clean[w] > p95 and D[0][w] < 0.9:
            # remove possible artefacts at w and two neighbors windows
            # checking for border conditions
            r_clean[w] = nan  # is artefact

            if w != D.shape[1]:
                r_clean[w + 1] = nan

            if w != 0:
                r_clean[w - 1] = nan

            w = w - 1

        w = w - 1

    if any(sig[:]):
        # ARR = np.std(r3) / np.mean(r3)
        ARRm = np.nanstd(r_clean) / np.nanmean(r_clean)
    else:
        # ARR = nan
        ARRm = nan

    # return r1,r2,r3,r_clean,ARR,ARRm
    return ARRm


class AutoregressiveResidualModulation(Method):

    algorithm = 'AUTOROGRESSIVE_RESIDUAL_MODULATION'
    algorithm_type = 'univariate'
    version = '1.0.0'
    dtype = [('arr', 'float32')]

    def __init__(self, **kwargs):
        """
        Autoregressive residual modulation

        Parameters
        ----------
        sig:
            numpy.ndarray
        fsamp: float64
            sample frequency
        """

        super().__init__(compute_arr, **kwargs)
        self._event_flag = False
