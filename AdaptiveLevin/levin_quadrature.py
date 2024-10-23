import time
from typing import Tuple

import numpy as np
from math import floor, ceil
from scipy.linalg import toeplitz

from utilities import format_time

# default interval at which to log progress of the integration
DEFAULT_LEVIN_NOTIFY_INTERVAL = 5 * 60


def chebyshev_matrices(x_span: Tuple[float, float], N: int):
    """
    Compute the Chebyshev spectral collocation points (corresponding to the extremal points)
    and the first order spectral differentiation matrix.
    Based on the implementation by Wiedeman & Reddy (http://dx.doi.org/10.1145/365723.365727)
    and pyddx (https://github.com/ronojoy/pyddx)
    :param x_span:
    :param N:
    :return:
    """
    n1 = floor(N / 2)
    n2 = ceil(N / 2)

    k = np.arange(N)
    theta = k * np.pi / (N - 1)

    # Compute the Chebyshev collocation points
    # x = np.cos(np.pi*np.linspace(N-1,0,N)/(N-1))                # obvious way
    x = np.sin(
        np.pi * ((N - 1.0) - 2.0 * np.linspace(N - 1.0, 0, N)) / (2.0 * (N - 1.0))
    )  # W&R way
    x = x[::-1]

    # Assemble the differentiation matrices
    T = np.tile(theta / 2, (N, 1))
    DX = 2 * np.sin(T.T + T) * np.sin(T.T - T)  # trigonometric identity
    DX[n1:, :] = -np.flipud(np.fliplr(DX[0:n2, :]))  # flipping trick
    DX[range(N), range(N)] = 1.0  # diagonals of D
    DX = DX.T

    C = toeplitz((-1.0) ** k)  # matrix with entries c(k)/c(j)
    C[0, :] *= 2
    C[-1, :] *= 2
    C[:, 0] *= 0.5
    C[:, -1] *= 0.5

    Z = 1.0 / DX  # Z contains entries 1/(x(k)-x(j))
    Z[range(N), range(N)] = 0.0  # with zeros on the diagonal.

    # set up differentiation matrix on [-1, 1]
    D = np.eye(N)

    D = Z * (C * np.tile(np.diag(D), (N, 1)).T - D)  # off-diagonals
    D[range(N), range(N)] = -np.sum(D, axis=1)  # negative sum trick

    # rescale for the arbitrary interval
    a, b = x_span
    x = (a + b) / 2.0 + (b - a) / 2.0 * x
    D = 2.0 * D / (b - a)

    return x, D


class _Weight_SinCos:
    def __init__(self, theta):
        """
        :param theta:
        """
        self._theta = theta

    def __call__(self, grid, Dmat):
        # sample the phase function theta on the Chebyshev grid
        theta_Cheb = np.array([self._theta(x) for x in grid])

        # multiply theta by the spectral differentiation matrix Dmat in order to produce an estimate of theta'(x)
        # evaluated at the collocation points
        theta_prime_Cheb = np.matmul(Dmat, theta_Cheb)

        # if w = (sin, cos) (regarded as a column vector), then w' = A w where A is the matrix
        #   Amat = ( 0,       theta' )
        #          ( -theta', 0      )
        # and therefore its transpose is
        #   Amat^T = ( 0,      -theta' )
        #            ( theta', 0       )
        theta_prime_I = np.diag(theta_prime_Cheb)

        zero_block = np.zeros_like(theta_prime_I)
        AmatT = np.block([[zero_block, -theta_prime_I], [theta_prime_I, zero_block]])

        # note grid is in reverse order, with largest value in position 1 and smallest value in last position -1
        theta0 = theta_Cheb[-1]
        thetak = theta_Cheb[0]

        w0 = [np.sin(theta0), np.cos(theta0)]
        wk = [np.sin(thetak), np.cos(thetak)]

        return AmatT, w0, wk


def _adaptive_levin_subregion(
    x_span: Tuple[float, float],
    f,
    Weights,
    chebyshev_order: int = 12,
    build_p_sample: bool = False,
):
    """
    f should be an m-vector of non-rapidly oscillating functions (Levin 96 eq. 2.1)
    theta should be an (m x m)-matrix representing the phase matrix of the system (Levin 96's A or A^t matrix)
    :param x_span:
    :param f: iterable of callables representing the integrand f-functions
    :param Weights: callable
    :param tol:
    :param chebyshev_order:
    :return:
    """
    grid, Dmat = chebyshev_matrices(x_span, chebyshev_order)

    # sample each component of f on the Chebyshev grid,
    # then assemble the result into a flattened vector in an m x k representation
    # Chen et al. around (166), (167)
    m = len(f)
    f_Cheb = np.hstack([[func(x) for x in grid] for func in f])

    # build the Levin A^T matrix, and also the vector of weights w evaluated at theta0, thetak
    # (these are needed in the final stap)
    AmatT, w0, wk = Weights(grid, Dmat)

    # build the Levin superoperator corresponding to this system
    # Chen et al. (168)
    zero_block = np.zeros((chebyshev_order, chebyshev_order))

    row_list = []
    for i in range(m):
        row = [zero_block for _ in range(m)]
        row[i] = Dmat
        row_list.append(row)

    LevinL = np.block(row_list) + AmatT

    # now try to invert the Levin superoperator, to find the Levin antiderivatives p(x)
    # Chen et al.
    p, residuals, rank, s = np.linalg.lstsq(LevinL, f_Cheb)

    if build_p_sample:
        p_sample = [
            (x, [p[j * chebyshev_order + i] for j in range(m)])
            for i, x in enumerate(grid)
        ]
    else:
        p_sample = []

    # note grid is in reverse order, with largest value in position 1 and smallest value in last position -1
    lower_limit = sum(p[(i + 1) * chebyshev_order - 1] * w0[i] for i in range(m))
    upper_limit = sum(p[i * chebyshev_order] * wk[i] for i in range(m))

    return upper_limit - lower_limit, p_sample


def _adaptive_levin(
    x_span: Tuple[float, float],
    f,
    Weights,
    tol: float = 1e-7,
    chebyshev_order: int = 12,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
):
    regions = [x_span]

    val = 0.0
    used_regions = []
    p_points = []
    num_evaluations = 0

    driver_start: float = time.perf_counter()
    start_time: float = time.time()
    last_notify: float = start_time
    updates_issued: int = 0

    while len(regions) > 0:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            _notify_progress(
                now,
                last_notify,
                start_time,
                len(used_regions),
                num_evaluations,
                len(regions),
                val,
                updates_issued,
                notify_label,
            )
            last_notify = time.time()

        a, b = regions.pop()

        # Chen et al. (172)
        estimate, p_sample = _adaptive_levin_subregion(
            (a, b),
            f,
            Weights,
            chebyshev_order=chebyshev_order,
            build_p_sample=build_p_sample,
        )

        # Chen et al. (173)
        c = (a + b) / 2.0
        estimate_L, pL_sample = _adaptive_levin_subregion(
            (a, c),
            f,
            Weights,
            chebyshev_order=chebyshev_order,
            build_p_sample=build_p_sample,
        )
        estimate_R, pR_sample = _adaptive_levin_subregion(
            (c, b),
            f,
            Weights,
            chebyshev_order=chebyshev_order,
            build_p_sample=build_p_sample,
        )

        num_evaluations += 3

        # Chen et al. step (4), below (173)
        if np.fabs(estimate - estimate_L - estimate_R) < tol:
            val = val + estimate
            used_regions.append((a, b))
            if build_p_sample:
                p_points.extend(p_sample)

        else:
            regions.extend([(a, c), (c, b)])

    used_regions.sort(key=lambda x: x[0])
    if build_p_sample:
        p_points.sort(key=lambda x: x[0])

    driver_stop = time.perf_counter()
    elapsed = driver_stop - driver_start

    return {
        "value": val,
        "p_points": p_points,
        "regions": used_regions,
        "evaluations": num_evaluations,
        "elapsed": elapsed,
    }


def _notify_progress(
    now: float,
    last_notify: float,
    start_time: float,
    used_regions: int,
    num_evaluations: int,
    remain_regions: int,
    current_val: float,
    update_number: int,
    notify_label: str = None,
):
    since_last_notify = now - last_notify
    since_start = now - start_time

    if notify_label is not None:
        print(
            f'** STATUS UDPATE #{update_number}: Levin quadrature "{notify_label}" has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)'
        )
    else:
        print(
            f"** STATUS UDPATE #{update_number}: Levin quadrature  has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )

    print(
        f"|  -- current value = {current_val:.7g}, {used_regions} used subintervals, {remain_regions} subintervals remain | {num_evaluations} integrand evaluations"
    )


def adaptive_levin_sincos(
    x_span: Tuple[float, float],
    f,
    theta,
    tol: float = 1e-7,
    chebyshev_order: int = 12,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
):
    A = _Weight_SinCos(theta)

    return _adaptive_levin(
        x_span,
        f,
        A,
        tol=tol,
        chebyshev_order=chebyshev_order,
        build_p_sample=build_p_sample,
        notify_interval=notify_interval,
        notify_label=notify_label,
    )
