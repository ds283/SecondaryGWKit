import json
import time
import uuid
from datetime import datetime
from math import floor, ceil
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError
from scipy.linalg import toeplitz

from Quadrature.simple_quadrature import simple_quadrature
from utilities import format_time

# default interval at which to log progress of the integration
DEFAULT_LEVIN_NOTIFY_INTERVAL = 5 * 60

# default Chebyshev spectral order
DEFAULT_LEVIN_CHEBSHEV_ORDER = 12
_LEVIN_MINIMUM_ALLOWED_ORDER = 8

# default maximum bisection depth. 1/2^20 is roughly 1E-6
DEFAULT_LEVIN_MAX_DEPTH = 20

# default abs tolerance
DEFAULT_LEVIN_ABSTOL = 1e-15

# default rel tolerance
DEFAULT_LEVIN_RELTOL = 1e-7

# approximate machine epsilon for 64-bit floats
MACHINE_EPSILON = 1e-16

SIX_PI = 6.0 * np.pi

INTERVAL_TYPE_LEVIN = 0
INTERVAL_TYPE_DIRECT = 1
types = {0: "Levin", 1: "direct"}


class used_interval:

    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        type: int,
        abserr: Optional[float] = None,
        relerr: Optional[float] = None,
    ):
        self._start = start
        self._end = end
        self._depth = depth

        self._type = type

        self._abserr = abserr
        self._relerr = relerr

    def __str__(self):
        if self._abserr is not None:
            abserr_label = f"abserr={self._abserr:.8g}"
        else:
            abserr_label = "(not recorded)"

        if self._relerr is not None:
            relerr_label = f"relerr={self._relerr:.8g}"
        else:
            relerr_label = "(not recorded)"

        return f"({self._start:.8g}, {self._end:.8g}), depth={self._depth} | {types[self._type]}, abserr={abserr_label}, relerr={relerr_label}"

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def width(self) -> float:
        return np.fabs(self._end - self._start)

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def type(self) -> str:
        return types[self._type]

    @property
    def abserr(self) -> float:
        return self._abserr

    @property
    def relerr(self) -> float:
        return self._relerr


ErrorHistoryType = dict[int, float]


class _levin_interval:
    def __init__(
        self,
        start: float,
        end: float,
        depth: int,
        abserr_history: Optional[ErrorHistoryType] = None,
        relerr_history: Optional[ErrorHistoryType] = None,
    ):
        self._start = start
        self._end = end
        self._depth = depth

        self._abserr_history = abserr_history if abserr_history is not None else {}
        self._relerr_history = relerr_history if relerr_history is not None else {}

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def width(self) -> float:
        return np.fabs(self._end - self._start)

    @property
    def break_point(self) -> float:
        return (self._end + self._start) / 2.0

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def abserr_history(self) -> ErrorHistoryType:
        return self._abserr_history

    @property
    def relerr_history(self) -> ErrorHistoryType:
        return self._relerr_history


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


class _Basis_SinCos:
    def __init__(self, theta):
        """
        :param theta:
        """
        if "theta" not in theta:
            raise RuntimeError(
                "levin_quadrature: Levin phase function theta must be provided"
            )

        self._theta = theta["theta"]

        if "theta_mod_2pi" in theta:
            self._theta_mod_2pi = theta["theta_mod_2pi"]

        if "theta_deriv" in theta:
            self._theta_deriv = theta["theta_deriv"]

    def raw_theta(self, x):
        return self._theta(x)

    def build_Levin_data(self, grid, Dmat):
        if hasattr(self, "_theta_deriv"):
            theta_prime_Cheb = np.array([self._theta_deriv(x) for x in grid])
        else:
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

        if hasattr(self, "_theta_mod_2pi"):
            theta0_mod_2pi = self._theta_mod_2pi(grid[-1])
            thetak_mod_2pi = self._theta_mod_2pi(grid[0])

            w0 = [np.sin(theta0_mod_2pi), np.cos(theta0_mod_2pi)]
            wk = [np.sin(thetak_mod_2pi), np.cos(thetak_mod_2pi)]
        else:
            # note grid is in reverse order, with largest value in position 1 and smallest value in last position -1
            theta0 = theta_Cheb[-1]
            thetak = theta_Cheb[0]

            w0 = [np.sin(theta0), np.cos(theta0)]
            wk = [np.sin(thetak), np.cos(thetak)]

        return AmatT, w0, wk

    def eval_basis(self, x):
        if hasattr(self, "_theta_mod_2pi"):
            return [np.sin(self._theta_mod_2pi(x)), np.cos(self._theta_mod_2pi(x))]

        return [np.sin(self._theta(x)), np.cos(self._theta(x))]


def _adaptive_levin_subregion(
    x_span: Tuple[float, float],
    f,
    BasisData,
    id_label: uuid,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    build_p_sample: bool = False,
    notify_label: Optional[str] = None,
):
    working_order = max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER)
    num_order_changes = 0
    metadata = {}

    # to handle possible SVD failures, allow the working Chebyshev order to be stepped down.
    # this changes the matrices that we need to invert, so gives another change for the required SVD to converge
    finished = False
    while not finished and working_order >= _LEVIN_MINIMUM_ALLOWED_ORDER:
        value, p_sample, _metadata = _adaptive_levin_subregion_impl(
            x_span,
            f,
            BasisData,
            id_label,
            working_order,
            build_p_sample,
            notify_label,
        )
        if _metadata.get("SVD_failure", False):
            working_order = working_order - 2
            num_order_changes = num_order_changes + 1

            if id_label is not None:
                label = f"{notify_label} id={id_label}"
            else:
                label = f"{id_label}"
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): SVD failure - stepping down Chebyshev order to {working_order}"
            )
            continue

        finished = True

    if value is None:
        raise LinAlgError("SVD failure")

    metadata["SVD_failure"] = _metadata.get("SVD_failure", False)
    metadata["num_order_changes"] = num_order_changes
    metadata["chebyshev_order"] = working_order

    return value, p_sample, metadata


def _adaptive_levin_subregion_impl(
    x_span: Tuple[float, float],
    f,
    BasisData,
    id_label: uuid,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    build_p_sample: bool = False,
    notify_label: Optional[str] = None,
):
    """
    f should be an m-vector of non-rapidly oscillating functions (Levin 96 eq. 2.1)
    theta should be an (m x m)-matrix representing the phase matrix of the system (Levin 96's A or A^t matrix)
    :param x_span:
    :param f: iterable of callables representing the integrand f-functions
    :param BasisData: callable
    :param chebyshev_order:
    :return:
    """
    metadata = {}
    if id_label is not None:
        label = f"{notify_label} id={id_label}"
    else:
        label = f"{id_label}"

    grid, Dmat = chebyshev_matrices(x_span, chebyshev_order)

    # sample each component of f on the Chebyshev grid,
    # then assemble the result into a flattened vector in an m x k representation
    # Chen et al. around (166), (167)
    m = len(f)
    f_Cheb = np.hstack([[func(x) for x in grid] for func in f])

    # build the Levin A^T matrix, and also the vector of weights w evaluated at theta0, thetak
    # (these are needed in the final stap)
    AmatT, w0, wk = BasisData.build_Levin_data(grid, Dmat)

    # build the Levin superoperator corresponding to this system
    # Chen et al. (168)
    zero_block = np.zeros((chebyshev_order, chebyshev_order))

    row_list = []
    for i in range(m):
        row = [zero_block for _ in range(m)]
        row[i] = Dmat
        row_list.append(row)

    LevinL = np.block(row_list) + AmatT

    if not np.isfinite(LevinL).all():
        print(
            f"!! WARNING (adaptive_levin_subregion, {label}): Levin super-operator contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

        raise ValueError(
            "Levin super-operator contains non-numeric values (np.nan, np.inf, or np.-inf)"
        )

    # now try to invert the Levin superoperator, to find the Levin antiderivatives p(x)
    # Chen et al.
    success = False
    try:
        p, residuals, rank, s = np.linalg.lstsq(LevinL, f_Cheb)
    except LinAlgError as e:
        print(
            f"!! WARNING (adaptive_levin_subregion, {label}): could not solve Levin collocation system using numpy.linalg.lstsq (chebyshev_order={chebyshev_order}; will now attempt to use pseudo-inverse)"
        )
        now = datetime.now().replace(microsecond=0)
        LevinL_filename = f"LevinL_{now.isoformat()}.txt"
        f_Cheb_filename = f"f_Cheb_{now.isoformat()}.txt"
        print(
            (
                f'   -- Levin L super-operator written to file "{LevinL_filename}", f_Cheb written to file "{f_Cheb_filename}"'
            )
        )
        np.savetxt(LevinL_filename, LevinL)
        np.savetxt(f_Cheb_filename, f_Cheb)
        metadata["SVD_errors"] = 1
    else:
        success = True

    if not success:
        try:
            LevinL_inv = np.linalg.pinv(LevinL)
            p = np.matmul(LevinL_inv, f_Cheb)
        except LinAlgError as e:
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): could not solve Levin collocation system using numpy.linalg.pinv (chebyshev_order={chebyshev_order}; final failure at this order)"
            )
            metadata["SVD_failure"] = True
            return None, None, metadata

    if build_p_sample:
        p_sample = [
            (x, [p[j * chebyshev_order + i] for j in range(m)])
            for i, x in enumerate(grid)
        ]
    else:
        p_sample = []

    # note grid is in reverse order, with largest value in position o and smallest value in last position -1
    lower_limit = sum(p[(i + 1) * chebyshev_order - 1] * w0[i] for i in range(m))
    upper_limit = sum(p[i * chebyshev_order] * wk[i] for i in range(m))

    return upper_limit - lower_limit, p_sample, metadata


def _adaptive_levin(
    x_span: Tuple[float, float],
    f,
    BasisData,
    atol: float = DEFAULT_LEVIN_ABSTOL,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    depth_max: int = DEFAULT_LEVIN_MAX_DEPTH,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics: bool = False,
):
    driver_start: float = time.perf_counter()
    start_time: float = time.time()
    last_notify: float = start_time
    updates_issued: int = 0

    # generate unique id to identify this calculation
    id_label = uuid.uuid4()
    if notify_label is not None:
        label = f"{notify_label}, id={id_label}"
    else:
        label = f"{id_label}"

    m = len(f)

    regions = [_levin_interval(start=x_span[0], end=x_span[1], depth=0)]

    val = 0.0
    used_regions = []
    p_points = []
    num_used_regions = 0
    num_simple_regions = 0
    num_evaluations = 0

    num_SVD_errors = 0
    num_order_changes = 0
    chebyshev_min_order = None
    max_depth = 0

    num_errhistory_messages = 0

    while len(regions) > 0:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            _notify_progress(
                now,
                last_notify,
                start_time,
                val,
                num_used_regions,
                len(regions),
                num_simple_regions,
                max_depth,
                num_evaluations,
                num_SVD_errors,
                num_order_changes,
                chebyshev_min_order,
                updates_issued,
                id_label,
                notify_label,
            )

            # dump data every 3 notifications
            if emit_diagnostics and updates_issued % 3 == 1:
                _write_progress_data(
                    f,
                    BasisData,
                    regions,
                    chebyshev_order,
                    val,
                    id_label,
                    atol,
                    rtol,
                    notify_label,
                )

            last_notify = time.time()

        current_region = regions.pop()
        a = current_region.start
        b = current_region.end

        if current_region.depth >= 18 and num_errhistory_messages < 20:
            num_errhistory_messages += 1

            print(
                f"@@ adaptive_levin ({label}): encountered subinterval of depth {current_region.depth} (notification {num_errhistory_messages}/20 for this quadrature)"
            )

            abs_history = current_region.abserr_history
            rel_history = current_region.relerr_history

            prev_abs = None
            prev_rel = None

            for i in range(0, current_region.depth):
                this_abs = abs_history[i]
                this_rel = rel_history[i]

                if i == 0:
                    print(
                        f"   -- {i+1}. abserr={this_abs :.5g}, relerr={this_rel :.5g}"
                    )

                else:
                    abs_improvement = prev_abs / this_abs
                    rel_improvement = prev_rel / this_rel
                    print(
                        f"   -- {i+1}. abserr={this_abs :.5g} (improvement={abs_improvement:.3g}), relerr={this_rel :.5g} (improvement={rel_improvement:.3g})"
                    )

                    prev_abs = this_abs
                    prev_rel = this_rel

        # if phase difference across this region is small enough that we do not have many oscillations,
        # there is likely no advantage in using the Levin rule to do the computation.
        # We can terminate the adaptive process by doing ordinary numerical quadrature
        phase_diff = np.fabs(BasisData.raw_theta(b) - BasisData.raw_theta(a))
        if np.fabs(phase_diff) < SIX_PI:

            def integrand(x):
                basis = BasisData.eval_basis(x)
                return sum(f[i](x) * basis[i] for i in range(m))

            data = simple_quadrature(
                integrand,
                a=a,
                b=b,
                atol=atol,
                rtol=rtol,
                method="quad",
            )

            val = val + data["value"]
            used_regions.append(
                used_interval(
                    start=a,
                    end=b,
                    depth=current_region.depth,
                    abserr=data["abserr"],
                    relerr=None,
                    type=INTERVAL_TYPE_DIRECT,
                )
            )
            num_used_regions = num_used_regions + 1
            num_simple_regions = num_simple_regions + 1
            continue

        # Chen et al. (172)
        try:
            estimate, p_sample, metadata = _adaptive_levin_subregion(
                (a, b),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                build_p_sample=build_p_sample,
                notify_label=notify_label,
            )
            order = metadata.get("chebyshev_order", None)
            if order is not None:
                if chebyshev_min_order is None or order < chebyshev_min_order:
                    chebyshev_min_order = order

            num_SVD_errors = num_SVD_errors + metadata.get("SVD_errors", 0)
            num_order_changes = num_order_changes + metadata.get("num_order_changes", 0)
        except LinAlgError as e:
            print(
                f"!! adaptive_levin ({label}): linear algebra error when estimating Levin subregion ({a}, {b}), width={current_region.width :.8g}"
            )
            raise e

        c = current_region.break_point
        # Chen et al. (173)
        try:
            estimate_L, pL_sample, metadataL = _adaptive_levin_subregion(
                (a, c),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                build_p_sample=build_p_sample,
                notify_label=notify_label,
            )
        except LinAlgError as e:
            print(
                f"!! adaptive_levin ({label}): linear algebra error when estimating Levin left-comparison region ({a}, {c}), parent region = ({a}, {b})"
            )
            raise e

        try:
            estimate_R, pR_sample, metadataR = _adaptive_levin_subregion(
                (c, b),
                f,
                BasisData,
                id_label=id_label,
                chebyshev_order=chebyshev_order,
                build_p_sample=build_p_sample,
                notify_label=notify_label,
            )
        except LinAlgError as e:
            print(
                f"!! adaptive_levin ({label}): linear algebra error when estimating Levin right-comparison region ({c}, {b}), parent region = ({a}, {b})"
            )
            raise e

        num_evaluations += 3
        refined_estimate = estimate_L + estimate_R

        relerr = np.fabs((estimate - refined_estimate)) / min(
            np.fabs(estimate), np.fabs(refined_estimate)
        )
        abserr = np.fabs(estimate - refined_estimate)

        # print(
        #     f"region: ({a}, {b}) | depth {current_region.depth+1} | abserr={abserr:.8g}, relerr={relerr:.8g}"
        # )
        # if current_region.abserr_history is not None:
        #     print(
        #         f"prev abserr={current_region.abserr_history:.8g}, improvement abserr={current_region.abserr_history/abserr:.4g}"
        #     )
        # if current_region.relerr_history is not None:
        #     print(
        #         f"prev relerr={current_region.relerr_history:.8g}, improvement relerr={current_region.relerr_history/relerr:.4g}"
        #     )

        # Chen et al. step (4), below (173) [adapted to also include a relative tolerance check]
        # but terminate the process if we exceed a specified number of bisections
        if (abserr < atol or relerr < rtol) or current_region.depth >= depth_max:
            val = val + estimate
            new_depth = current_region.depth + 1

            used_regions.append(
                used_interval(
                    start=a,
                    end=b,
                    depth=new_depth,
                    type=INTERVAL_TYPE_LEVIN,
                    abserr=abserr,
                    relerr=relerr,
                )
            )
            num_used_regions = num_used_regions + 1

            if new_depth > max_depth:
                max_depth = new_depth

            if build_p_sample:
                p_points.extend(p_sample)

        else:
            new_abs_history = current_region.abserr_history | {
                current_region.depth: abserr
            }
            new_rel_history = current_region.relerr_history | {
                current_region.depth: relerr
            }
            new_depth = current_region.depth + 1

            regions.extend(
                [
                    _levin_interval(
                        start=a,
                        end=c,
                        depth=new_depth,
                        abserr_history=new_abs_history,
                        relerr_history=new_rel_history,
                    ),
                    _levin_interval(
                        start=c,
                        end=b,
                        depth=new_depth,
                        abserr_history=new_abs_history,
                        relerr_history=new_rel_history,
                    ),
                ]
            )

    used_regions.sort(key=lambda x: x.start)
    if build_p_sample:
        p_points.sort(key=lambda x: x[0])

    driver_stop = time.perf_counter()
    elapsed = driver_stop - driver_start

    return {
        "value": float(val),
        "p_points": p_points,
        "num_regions": num_used_regions,
        "regions": used_regions,
        "num_simple_regions": num_simple_regions,
        "evaluations": int(num_evaluations),
        "elapsed": float(elapsed),
        "num_SVD_errors": num_SVD_errors,
        "num_order_changes": num_order_changes,
        "chebyshev_min_order": chebyshev_min_order,
        "max_depth": max_depth,
    }


def _notify_progress(
    now: float,
    last_notify: float,
    start_time: float,
    current_val: float,
    used_regions: int,
    remain_regions: int,
    simple_regions: int,
    max_depth: int,
    num_evaluations: int,
    num_SVD_errors,
    num_order_changes,
    chebyshev_min_order,
    update_number: int,
    id_label,
    notify_label: str = None,
):
    since_last_notify = now - last_notify
    since_start = now - start_time

    if notify_label is not None:
        print(
            f'** STATUS UPDATE #{update_number}: Levin quadrature "{notify_label}" ({id_label}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)'
        )
    else:
        print(
            f"** STATUS UPDATE #{update_number}: Levin quadrature {id_label} has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )

    print(
        f"|  -- current value = {current_val:.7g}, {used_regions} used subintervals (of which {simple_regions} direct quadrature), {remain_regions} subintervals remain | {num_evaluations} integrand evaluations | max depth {max_depth}"
    )
    if num_SVD_errors > 0 or num_order_changes > 0:
        print(
            f"|  -- {num_SVD_errors} SVD errors, {num_order_changes} Chebyshev order changes | minimum used Chebyshev order = {chebyshev_min_order}"
        )


def _safe_fabs(x):
    if x is None:
        return None

    return np.fabs(x)


def _write_progress_data(
    f,
    BasisData,
    regions: List[_levin_interval],
    chebyshev_order: int,
    current_val: float,
    id_label: uuid,
    atol: float,
    rtol: float,
    notify_label: Optional[str] = None,
):
    path = Path(
        f"SlowLevinData/{id_label}/{datetime.now().replace(microsecond=0).isoformat()}"
    ).resolve()
    path.mkdir(parents=True, exist_ok=True)

    payload = {
        "current_val": current_val,
        "id_label": str(id_label),
        "user_label": notify_label,
        "atol": atol,
        "rtol": rtol,
    }

    sns.set_theme()

    m = len(f)

    region_list = []
    for reg_num, region in enumerate(regions):
        start = region.start
        end = region.end
        region_data = {
            "id": reg_num,
            "start": start,
            "end": end,
            "depth": region.depth,
            "abserr_history": region.abserr_history,
            "relerr_history": region.relerr_history,
        }

        x_grid = np.linspace(start, end, 500)
        y_grid = []
        f_grid = [[] for _ in range(m)]
        for x in x_grid:
            basis = BasisData.eval_basis(x)
            y = sum(f[i](x) * basis[i] for i in range(m))
            y_grid.append(_safe_fabs(y))
            for i in range(m):
                f_grid[i].append(_safe_fabs(f[i](x)))

        estimate, p_sample, metadata = _adaptive_levin_subregion(
            (start, end),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            build_p_sample=True,
            notify_label=notify_label,
        )
        region_data["estimate"] = estimate

        mid = region.break_point

        estimate_L, pL_sample, metadataL = _adaptive_levin_subregion(
            (start, mid),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            build_p_sample=True,
            notify_label=notify_label,
        )
        estimate_R, pR_sample, metadataR = _adaptive_levin_subregion(
            (mid, end),
            f,
            BasisData,
            id_label=id_label,
            chebyshev_order=chebyshev_order,
            build_p_sample=True,
            notify_label=notify_label,
        )
        refined_estimate = estimate_L + estimate_R
        region_data["estimate_L"] = estimate_L
        region_data["estimate_R"] = estimate_R
        region_data["refined_estimate"] = refined_estimate

        relerr = np.fabs((estimate - refined_estimate)) / min(
            np.fabs(estimate), np.fabs(refined_estimate)
        )
        abserr = np.fabs(estimate - refined_estimate)

        region_data["relerr"] = relerr
        region_data["abserr"] = abserr
        region_data["abserr_status"] = "fail" if abserr >= atol else "pass"
        region_data["relerr_status"] = "fail" if relerr >= rtol else "pass"

        # only report regions that will not be accepted in the next step
        if abserr >= atol and relerr >= rtol:
            region_list.append(region_data)

            p_x_grid = []
            p_y_grid = [[] for _ in range(m)]
            for x, p_data in p_sample:
                p_x_grid.append(x)
                for i in range(m):
                    p_y_grid[i].append(_safe_fabs(p_data[i]))

            pL_x_grid = []
            pL_y_grid = [[] for _ in range(m)]
            for x, p_data in pL_sample:
                pL_x_grid.append(x)
                for i in range(m):
                    pL_y_grid[i].append(_safe_fabs(p_data[i]))

            pR_x_grid = []
            pR_y_grid = [[] for _ in range(m)]
            for x, p_data in pR_sample:
                pR_x_grid.append(x)
                for i in range(m):
                    pR_y_grid[i].append(_safe_fabs(p_data[i]))

            fig = plt.figure()
            ax = plt.gca()

            ax.plot(x_grid, y_grid, label="integrand", color="r")

            for i in range(m):
                ax.plot(x_grid, f_grid[i], linestyle="dashdot", label=f"Levin f{i+1}")
                ax.plot(
                    p_x_grid, p_y_grid[i], linestyle="dashed", label=f"Levin p{i+1}"
                )
                ax.plot(
                    pL_x_grid, pL_y_grid[i], linestyle="dotted", label=f"Levin p{i+1} L"
                )
                ax.plot(
                    pR_x_grid, pR_y_grid[i], linestyle="dotted", label=f"Levin p{i+1} R"
                )

            ax.set_xscale("linear")
            ax.set_yscale("log")
            ax.legend(loc="best")
            ax.grid(True)

            fig_path = path / f"region{reg_num}_start{start:.5g}_end{end:.5g}.pdf"
            fig.savefig(fig_path)
            fig.savefig(fig_path.with_suffix(".png"))

            plt.close()

    payload["regions"] = region_list
    payload_path = path / "payload.json"
    with open(payload_path, "w", newline="") as handle:
        json.dump(payload, handle, indent=4, sort_keys=True)


def adaptive_levin_sincos(
    x_span: Tuple[float, float],
    f,
    theta: dict,
    atol: float = DEFAULT_LEVIN_ABSTOL,
    rtol: float = DEFAULT_LEVIN_RELTOL,
    chebyshev_order: int = DEFAULT_LEVIN_CHEBSHEV_ORDER,
    depth_max: int = DEFAULT_LEVIN_MAX_DEPTH,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics=False,
):
    A = _Basis_SinCos(theta)

    return _adaptive_levin(
        x_span,
        f,
        A,
        atol=atol,
        rtol=rtol,
        chebyshev_order=chebyshev_order,
        depth_max=depth_max,
        build_p_sample=build_p_sample,
        notify_interval=notify_interval,
        notify_label=notify_label,
        emit_diagnostics=emit_diagnostics,
    )
