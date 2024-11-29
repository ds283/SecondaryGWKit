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

# approximate machine epsilon for 64-bit floats
MACHINE_EPSILON = 1e-16

SIX_PI = 6.0 * np.pi


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
    chebyshev_order: int = 12,
    build_p_sample: bool = False,
    notify_label: Optional[str] = None,
):
    working_order = max(chebyshev_order, 12)
    num_order_changes = 0
    metadata = {}

    # to handle possible SVD failures, allow the working Chebyshev order to be stepped down.
    # this changes the matrices that we need to invert, so gives another change for the required SVD to converge
    while working_order >= 12:
        value, p_sample, _metadata = _adaptive_levin_subregion(
            x_span,
            f,
            BasisData,
            id_label,
            chebyshev_order,
            build_p_sample,
            notify_label,
        )
        if metadata.get("SVD_failure", False):
            working_order -= 2
            num_order_changes += 1

            if id_label is not None:
                label = f"{notify_label} id={id_label}"
            else:
                label = f"{id_label}"
            print(
                f"!! WARNING (adaptive_levin_subregion, {label}): SVD failure - stepping down Chebyshev spectral order to {working_order}"
            )
            continue

        break

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
    chebyshev_order: int = 12,
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
            f"!! WARNING (adaptive_levin_subregion, {label}): could not solve Levin collocation system using numpy.linalg.lstsq; attemping to use pseudo-inverse"
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
                f"!! WARNING (adaptive_levin_subregion, {label}): could not solve Levin collocation system using numpy.linalg.pinv; failing"
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
    atol: float = 1e-15,
    rtol: float = 1e-7,
    chebyshev_order: int = 12,
    build_p_sample: bool = False,
    notify_interval: int = DEFAULT_LEVIN_NOTIFY_INTERVAL,
    notify_label: str = None,
    emit_diagnostics=False,
):
    # generate unique id to identify this calculation
    id_label = uuid.uuid4()
    if notify_label is not None:
        label = f"{notify_label}, id={id_label}"
    else:
        label = f"{id_label}"

    m = len(f)

    regions = [x_span]

    val = 0.0
    used_regions = []
    p_points = []
    num_used_regions = 0
    num_simple_regions = 0
    num_evaluations = 0

    driver_start: float = time.perf_counter()
    start_time: float = time.time()
    last_notify: float = start_time
    updates_issued: int = 0

    num_quad_warnings = 0

    num_SVD_errors = 0
    num_order_changes = 0

    start, end = x_span
    full_width = np.fabs(end - start)

    while len(regions) > 0:
        now = time.time()
        if now - last_notify > notify_interval:
            updates_issued = updates_issued + 1
            _notify_progress(
                now,
                last_notify,
                start_time,
                num_used_regions,
                num_evaluations,
                len(regions),
                val,
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

        a, b = regions.pop()

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

            if num_quad_warnings < 5:
                print(
                    f'## adaptive_levin ({label}): phase difference in region ({a:.8g}, {b:.8g}) is {phase_diff:.5g} = {phase_diff/np.pi:.3g} pi; using simple quadrature: value={data["value"]:.8g}'
                )
                num_quad_warnings += 1
                if num_quad_warnings == 5:
                    print(
                        f"## adaptive_levin ({label}): further warnings for simple quadrature will be suppressed for this computation"
                    )

            val = val + data["value"]
            used_regions.append((a, b))
            num_used_regions = num_used_regions + 1
            num_simple_regions = num_simple_regions + 1
            continue

        width = np.fabs(b - a)
        # print(f">> region: (a,b) = ({a}, {b}), width = {width:.8g}")
        # if width < 1e-5:
        #     print(f"** small region (a,b) = ({a}, {b}), width={width:.8g}")
        #
        #     x_sample = np.linspace(a, b, 25)
        #     sin_sample = [f[0](x) for x in x_sample]
        #     cos_sample = [f[1](x) for x in x_sample]
        #     print(f"   -- x_sample = {x_sample}")
        #     print(f"   -- sin_sample = {sin_sample}")
        #     print(f"   -- cos_sample = {cos_sample}")

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
            num_SVD_errors = num_SVD_errors + metadata.get("SVD_errors", 0)
            num_order_changes = num_order_changes + metadata.get("num_order_changes", 0)
        except LinAlgError as e:
            print(
                f"!! adaptive_levin ({label}): linear algebra error when estimating Levin subregion ({a}, {b}), width={width:.8g}"
            )
            raise e

        c = (a + b) / 2.0
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

        # Chen et al. step (4), below (173) [adapted to also include a relative tolerance check]
        # but terminate the process if the width of the interval has become extremely small
        if (abserr < atol or relerr < rtol) or (
            num_used_regions > 1e3 and width / full_width < 1e-2
        ):
            # print(
            #     f"** accepting interval [{a:.12g},{b:.12g}] -> {estimate:.12g} against [{a:.12g},{c:.12g}] -> {estimate_L:.12g} + [{c:.12g},{b:.12g}] -> {estimate_R:.12g} | abserr={abserr:.12g}, relerr={relerr:.12g}, atol={atol:.8g}, rtol={rtol:.8g}"
            # )

            # if np.fabs(estimate - refined_estimate) < tol:
            val = val + estimate
            used_regions.append((a, b))
            num_used_regions = num_used_regions + 1
            if build_p_sample:
                p_points.extend(p_sample)

        else:
            # print(
            #     f"** subdivided interval [{a:.12g},{b:.12g}] -> {estimate:.12g} against [{a:.12g},{c:.12g}] -> {estimate_L:.12g} + [{c:.12g},{b:.12g}] -> {estimate_R:.12g} | abserr={abserr:.12g}, relerr={relerr:.12g}, atol={atol:.8g}, rtol={rtol:.8g}"
            # )
            regions.extend([(a, c), (c, b)])

    used_regions.sort(key=lambda x: x[0])
    if build_p_sample:
        p_points.sort(key=lambda x: x[0])

    driver_stop = time.perf_counter()
    elapsed = driver_stop - driver_start

    return {
        "value": float(val),
        "p_points": p_points,
        "num_regions": num_used_regions,
        "regions": used_regions,
        "num_simple_region": num_simple_regions,
        "evaluations": int(num_evaluations),
        "elapsed": float(elapsed),
        "num_SVD_errors": num_SVD_errors,
        "num_order_changes": num_order_changes,
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
    id_label,
    notify_label: str = None,
):
    since_last_notify = now - last_notify
    since_start = now - start_time

    if notify_label is not None:
        print(
            f'** STATUS UDPATE #{update_number}: Levin quadrature "{notify_label}" ({id_label}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)'
        )
    else:
        print(
            f"** STATUS UDPATE #{update_number}: Levin quadrature {id_label} has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )

    print(
        f"|  -- current value = {current_val:.7g}, {used_regions} used subintervals, {remain_regions} subintervals remain | {num_evaluations} integrand evaluations"
    )


def _safe_fabs(x):
    if x is None:
        return None

    return np.fabs(x)


def _write_progress_data(
    f,
    BasisData,
    regions: List[Tuple[float, float]],
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
        start, end = region
        region_data = {"id": reg_num, "start": start, "end": end}

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

        mid = (start + end) / 2

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
    atol: float = 1e-15,
    rtol: float = 1e-7,
    chebyshev_order: int = 12,
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
        build_p_sample=build_p_sample,
        notify_interval=notify_interval,
        notify_label=notify_label,
        emit_diagnostics=emit_diagnostics,
    )
