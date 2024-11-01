import time
from typing import Optional, List, Union

import ray
from math import log, exp, pow, sqrt, pi, gamma
from scipy.integrate import solve_ivp
from scipy.special import yv, jv

from AdaptiveLevin import adaptive_levin_sincos
from AdaptiveLevin.bessel_phase import bessel_phase
from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy
from ComputeTargets.GkSource import GkSource
from ComputeTargets.GkSourcePolicyData import GkSourcePolicyData
from ComputeTargets.QuadSource import QuadSource
from ComputeTargets.integration_metadata import IntegrationData, LevinData
from ComputeTargets.integration_supervisor import (
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
    RHS_timer,
)
from CosmologyConcepts import wavenumber, wavenumber_exit_time, redshift
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance, GkSourcePolicy
from defaults import (
    DEFAULT_QUADRATURE_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)
from utilities import WallclockTimer, format_time


class QuadSourceSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        q: wavenumber,
        r: wavenumber,
        label: str,
        z_response: redshift,
        log_z_init: float,
        log_z_final: float,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k = k
        self._q = q
        self._r = r

        self._label = label

        self._z_response: float = z_response.z
        self._log_z_init: float = log_z_init
        self._log_z_final: float = log_z_final

        self._log_z_range: float = self._log_z_final - self._log_z_init

        self._last_log_z: float = self._log_z_init

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def message(self, current_log_z, msg):
        current_time = time.time()
        since_last_notify = current_time - self._last_notify
        since_start = current_time - self._start_time

        update_number = self.report_notify()

        z_complete = current_log_z - self._log_z_init
        z_remain = self._log_z_range - z_complete
        percent_remain = z_remain / self._log_z_range
        print(
            f"** STATUS UPDATE #{update_number}: QuadSourceIntegral quadrature (type={self._label}) for k={self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}), q={self._q.k_inv_Mpc:.5g}/Mpc (store_id={self._q.store_id}), r={self._r.k_inv_Mpc:.5g}/Mpc (store_id={self._r.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current log(1+z)={current_log_z:.5g} (init log(1+z)={self._log_z_init:.5g}, target log(1+z)={self._log_z_final:.5g}, log(1+z) complete={z_complete:.5g}, log(1+z) remain={z_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_log_z is not None:
            log_z_delta = current_log_z - self._last_log_z
            print(
                f"|    redshift advance since last update: Delta log(1+z) = {log_z_delta:.5g}"
            )
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_log_z = current_log_z


def get_z(z):
    if isinstance(z, redshift):
        return z.z

    elif isinstance(z, float):
        return z

    return float(z)


@ray.remote
def compute_QuadSource_integral(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    q: wavenumber_exit_time,
    r: wavenumber_exit_time,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    z_source_max: redshift,
    tol: float = DEFAULT_QUADRATURE_TOLERANCE,
) -> dict:
    # we are entitled to assume that the GkSource embedded in GkPolicy and source are evaluated at the same z_response
    # also that source and Gk have z_samples at least as far back as z_source_max

    # print(
    #     f"** QUADRATIC SOURCE INTEGRAL: k={k.k.k_inv_Mpc:.3g}/Mpc, q={q.k.k_inv_Mpc:.3g}/Mpc, r={r.k.k_inv_Mpc:.3g}/Mpc (source store_id={source.store_id}, k store_id={k.store_id}) starting calculation for z_response={z_response.z:.5g}"
    # )
    start_time = time.time()

    model: BackgroundModel = model_proxy.get()

    with WallclockTimer() as timer:
        numeric_quad: float = 0.0
        WKB_quad: float = 0.0
        WKB_Levin: float = 0.0

        numeric_quad_data = None
        WKB_quad_data = None
        WKB_Levin_data = None

        if GkPolicy.type == "numeric":
            regions = [
                (z_source_max, z_response),
                (None, None),
                (None, None),
            ]

        elif GkPolicy.type == "WKB":
            Levin_z: Optional[float] = GkPolicy.Levin_z

            regions = [
                (None, None),
                (z_source_max, Levin_z if Levin_z is not None else z_response),
                (Levin_z, z_response),
            ]

        elif GkPolicy.type == "mixed":
            crossover_z: Optional[float] = GkPolicy.crossover_z
            Levin_z: Optional[float] = GkPolicy.Levin_z

            # if Levin threshold occurs before the crossover point, move it up to the crossover point
            if (
                Levin_z is not None
                and crossover_z is not None
                and Levin_z > crossover_z
            ):
                Levin_z = crossover_z

            regions = [
                (z_source_max, crossover_z),
                (
                    crossover_z if crossover_z is not None else z_source_max,
                    Levin_z if Levin_z is not None else z_response,
                ),
                (Levin_z, z_response),
            ]

        else:
            raise NotImplementedError(f"Gk {GkPolicy.type} not implemented")

        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) - get_z(min_z) > DEFAULT_QUADRATURE_TOLERANCE
        ):
            now = time.time()
            # print(
            #     f"|  --  (source store_id={source.store_id}, k store_id={k.store_id}) running time={format_time(now - start_time)}, starting numerical quadrature part"
            # )
            payload = numeric_quad_integral(
                model,
                k.k,
                q.k,
                r.k,
                source,
                GkPolicy,
                z_response,
                max_z=get_z(max_z),
                min_z=get_z(min_z),
                tol=tol,
            )
            numeric_quad = payload["value"]
            numeric_quad_data = payload["data"]

        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) - get_z(min_z) > DEFAULT_QUADRATURE_TOLERANCE
        ):
            now = time.time()
            # print(
            #     f"|  --  (source store_id={source.store_id}, k store_id={k.store_id}) running time={format_time(now - start_time)}, starting WKB quadrature part"
            # )
            payload = WKB_quad_integral(
                model,
                k.k,
                q.k,
                r.k,
                source,
                GkPolicy,
                z_response,
                max_z=get_z(max_z),
                min_z=get_z(min_z),
                tol=tol,
            )
            WKB_quad = payload["value"]
            WKB_quad_data = payload["data"]

        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) - get_z(min_z) > DEFAULT_QUADRATURE_TOLERANCE
        ):
            now = time.time()
            # print(
            #     f"|  --  (source store_id={source.store_id}, k store_id={k.store_id}) running time={format_time(now - start_time)}, starting WKB Levin part"
            # )
            payload = WKB_Levin_integral(
                model,
                k.k,
                q.k,
                r.k,
                source,
                GkPolicy,
                z_response,
                max_z=get_z(max_z),
                min_z=get_z(min_z),
                tol=tol,
            )
            WKB_Levin = payload["value"]
            WKB_Levin_data = payload["data"]

        # calculate analytic approximation for radiation
        analytic_rad = analytic_integral(
            model,
            k.k,
            q.k,
            r.k,
            z_response,
            max_z=z_source_max,
            min_z=z_response,
            b=0.0,
            tol=tol,
        )

    return {
        "total": numeric_quad + WKB_quad + WKB_Levin,
        "numeric_quad": numeric_quad,
        "WKB_quad": WKB_quad,
        "WKB_Levin": WKB_Levin,
        "GkPolicy_serial": GkPolicy.store_id,
        "source_serial": source.store_id,
        "numeric_quad_data": numeric_quad_data,
        "WKB_quad_data": WKB_quad_data,
        "WKB_Levin_data": WKB_Levin_data,
        "analytic_rad": analytic_rad,
        "compute_time": timer.elapsed,
        "metadata": {},
    }


XCUT_SEARCH_TOLERANCE = 1e-2
XCUT_SEARCH_MAX_ITERATIONS = 200
XCUT_SEARCH_LEVIN_THRESHOLD = 0.1


def find_x_cut(x_min, x_max, dphase_A, dphase_B):
    iter = 0

    while (x_max - x_min) > XCUT_SEARCH_TOLERANCE and iter < XCUT_SEARCH_MAX_ITERATIONS:
        assert x_max > x_min

        dp_A_min = dphase_A(x_min)
        dp_B_min = dphase_B(x_min)

        if (
            dp_A_min > XCUT_SEARCH_LEVIN_THRESHOLD
            and dp_B_min > XCUT_SEARCH_LEVIN_THRESHOLD
        ):
            return x_min

        dp_A_max = dphase_A(x_max)
        dp_B_max = dphase_B(x_max)

        if (
            dp_A_max < XCUT_SEARCH_LEVIN_THRESHOLD
            and dp_B_max < XCUT_SEARCH_LEVIN_THRESHOLD
        ):
            raise RuntimeError(
                "rate of change of phase is too small even at right-hand endpoint"
            )

        x_mid = (x_min + x_max) / 2.0

        dp_A_mid = dphase_A(x_mid)
        dp_B_mid = dphase_B(x_mid)

        if (
            dp_A_mid > XCUT_SEARCH_LEVIN_THRESHOLD
            and dp_B_mid > XCUT_SEARCH_LEVIN_THRESHOLD
        ):
            x_max = x_mid
        else:
            x_min = x_mid

        iter = iter + 1

    if iter >= XCUT_SEARCH_MAX_ITERATIONS:
        print(
            f"** find_x_cut: recursive bisection to find x_cut did not converge: x_min={x_min:.5g}, x_max={x_max:.5g}"
        )

    x_cut_candidate = (x_min + x_max) / 2.0
    dp_A = dphase_A(x_cut_candidate)
    dp_B = dphase_B(x_cut_candidate)

    if dp_A < XCUT_SEARCH_LEVIN_THRESHOLD or dp_B < XCUT_SEARCH_LEVIN_THRESHOLD:
        # x_max should be guaranteed to satisfy dphase_A,B > XCUT_SEARCH_LEVIN_THRESHOLD
        return x_max

    return x_cut_candidate


def _three_bessel_integrals(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    phase_data: dict,
    nu_type: str,
    tol: float,
):
    if nu_type not in ["0pt5", "2pt5"]:
        raise RuntimeError('phase_data should be "0pt5" or "2pt5"')

    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)

    phase_data_A = phase_data["0pt5"]
    phase_data_B = phase_data[nu_type]

    dphase_A = phase_data_A["dphase"]
    dphase_B = phase_data_B["dphase"]

    min_k_mode = min(k.k, q.k, r.k)
    max_k_mode = max(k.k, q.k, r.k)

    min_x = min_k_mode * min_eta * cs
    max_x = max_k_mode * max_eta

    x_cut = find_x_cut(min_x, max_x, dphase_A, dphase_B)
    eta_cut = x_cut / min_k_mode / cs

    if eta_cut <= min_eta:
        Levin = _three_bessel_Levin(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            phase_data_A=phase_data_A,
            phase_data_B=phase_data_B,
            x_cut=x_cut,
            tol=tol,
        )
        return Levin

    if eta_cut < max_eta:
        quad = _three_bessel_quad(
            k, q, r, min_eta=min_eta, max_eta=eta_cut, b=b, nu_type=nu_type, tol=tol
        )
        Levin = _three_bessel_Levin(
            k,
            q,
            r,
            min_eta=eta_cut,
            max_eta=max_eta,
            b=b,
            phase_data_A=phase_data_A,
            phase_data_B=phase_data_B,
            x_cut=x_cut,
            tol=tol,
        )
        return (quad[0] + Levin[0], quad[1] + Levin[1])

    assert eta_cut >= max_eta

    quad = _three_bessel_quad(
        k, q, r, min_eta=min_eta, max_eta=max_eta, b=b, nu_type=nu_type, tol=tol
    )
    return quad


def _three_bessel_Levin(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    phase_data_A,
    phase_data_B,
    x_cut: float,
    tol: float,
):
    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    x_span = (log(min_eta), log(max_eta))

    min_k_mode = min(k.k, q.k, r.k)
    if min_k_mode * min_eta * cs > (1.0 + DEFAULT_FLOAT_PRECISION) * x_cut:
        raise RuntimeError(
            f"!! three_bessel_Levin: ERROR: smallest required x is smaller than x_cut (smallest x={min_k_mode * min_eta * cs:.5g}, x_cut={x_cut:.5g}, min_eta={min_eta:.5g}, min_k_mode={min_k_mode:.5g}, cs={cs:.5g}, coeff={min_k_mode*cs:.5g}, x_cut/coeff={x_cut/(min_k_mode*cs):.5g})"
        )

    phase_A = phase_data_A["phase"]
    dphase_A = phase_data_A["dphase"]

    phase_B = phase_data_B["phase"]
    dphase_B = phase_data_B["dphase"]

    def Levin_f(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        beta1 = dphase_A(x1)
        beta2 = dphase_B(x2)
        beta3 = dphase_B(x3)

        if beta1 < 0.0:
            raise ValueError(
                f"beta_1(x_1) < 0.0 (beta1 value={beta1:.5g} @ x1={x1:.5g})"
            )

        if beta2 < 0.0:
            raise ValueError(
                f"beta_2(x_2) < 0.0 (beta2 value={beta2:.5g} @ x2={x2:.5g})"
            )

        if beta3 < 0.0:
            raise ValueError(
                f"beta_3(x_3) < 0.0 (beta3 value={beta3:.5g} @ x3={x3:.5g})"
            )

        A = pow(eta, -b)
        B = 1.0 / sqrt(beta1 * beta2 * beta3)

        return A * B

    def phase1(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) + phase_B(x2) + phase_B(x3)

    J1_data = adaptive_levin_sincos(
        x_span, f=[Levin_f, lambda x: 0.0], theta=phase1, tol=tol
    )
    Y1_data = adaptive_levin_sincos(
        x_span, f=[lambda x: 0.0, Levin_f], theta=phase1, tol=tol
    )

    def phase2(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) + phase_B(x2) - phase_B(x3)

    J2_data = adaptive_levin_sincos(
        x_span, f=[Levin_f, lambda x: 0.0], theta=phase2, tol=tol
    )
    Y2_data = adaptive_levin_sincos(
        x_span, f=[lambda x: 0.0, Levin_f], theta=phase2, tol=tol
    )

    def phase3(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) - phase_B(x2) + phase_B(x3)

    J3_data = adaptive_levin_sincos(
        x_span, f=[Levin_f, lambda x: 0.0], theta=phase3, tol=tol
    )
    Y3_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, Levin_f],
        theta=phase3,
        tol=tol,
    )

    def phase4(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_A(x1) - phase_B(x2) - phase_B(x3)

    J4_data = adaptive_levin_sincos(
        x_span, f=[Levin_f, lambda x: 0.0], theta=phase4, tol=tol
    )
    Y4_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, Levin_f],
        theta=phase4,
        tol=tol,
    )

    J1_value = J1_data["value"]
    J2_value = J2_data["value"]
    J3_value = J3_data["value"]
    J4_value = J4_data["value"]

    Y1_value = Y1_data["value"]
    Y2_value = Y2_data["value"]
    Y3_value = Y3_data["value"]
    Y4_value = Y4_data["value"]

    norm_factor = pow(2.0 / pi, 3.0 / 2.0) / sqrt(k.k * q.k * r.k * cs * cs) / 4.0
    J = norm_factor * (-J1_value + J2_value + J3_value - J4_value)
    Y = norm_factor * (-Y1_value + Y2_value + Y3_value - Y4_value)

    return J, Y


def _three_bessel_quad(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    nu_type: str,
    tol: float,
):
    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    log_min_eta = log(min_eta)
    log_max_eta = log(max_eta)
    x_span = (log_min_eta, log_max_eta)

    nu_types = {"0pt5": 0.5, "2pt5": 2.5}
    nu = nu_types[nu_type]

    def RHS(log_eta, state):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        A = pow(eta, 1.5 - b)

        dJ = jv(0.5 + b, x1) * jv(nu + b, x2) * jv(nu + b, x3)
        dY = yv(0.5 + b, x1) * jv(nu + b, x2) * jv(nu + b, x3)

        return [A * dJ, A * dY]

    sol = solve_ivp(
        RHS,
        method="RK45",
        t_span=x_span,
        t_eval=[log_max_eta],
        y0=[0.0, 0.0],
        dense_output=True,
        atol=tol,
        rtol=tol,
    )

    if not sol.success:
        raise RuntimeError(
            f'_three_bessel_quad: quadrature did not terminate successfully | error at log(eta)={sol.t[-1]:.5g}, "{sol.message}"'
        )

    if sol.t[0] < log_max_eta:
        raise RuntimeError(
            f"_three_bessel_quad: quadrature did not terminate at expected conformal time log(eta)={log_max_eta:.5g} (final log(eta)={sol.t[0]:.5g})"
        )

    if len(sol.sol(log_max_eta)) != 2:
        raise RuntimeError(
            f"_three_bessel_quad: solution does not have expected number of members (expected 2, found {len(sol.sol(log_max_eta))})"
        )

    return sol.sol(log_max_eta)


def analytic_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    z_response: redshift,
    max_z: redshift,
    min_z: redshift,
    b: float,
    tol: float,
):
    functions = model.functions
    min_eta: float = functions.tau(max_z.z)
    max_eta: float = functions.tau(min_z.z)

    eta_response = functions.tau(z_response.z)

    cs_sq = (1.0 - b) / (1.0 + b) / 3.0

    max_x = max(k.k, q.k, r.k) * max_eta
    if max_x > 1e5:
        print(
            f"!! analytic: WARNING: max_x very large, {max_x:.5g}. This may lead to issues with calculation of the Liouville-Green phase function for the Bessel functions."
        )

    phase_data_0pt5 = bessel_phase(0.5 + b, max_x + 10.0)
    phase_data_2pt5 = bessel_phase(2.5 + b, max_x + 10.0)

    J0pt5, Y0pt5 = _three_bessel_integrals(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        phase_data={"0pt5": phase_data_0pt5, "2pt5": phase_data_2pt5},
        nu_type="0pt5",
        tol=tol,
    )
    J2pt5, Y2pt5 = _three_bessel_integrals(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        phase_data={"0pt5": phase_data_0pt5, "2pt5": phase_data_2pt5},
        nu_type="2pt5",
        tol=tol,
    )

    A = (2.0 + b) / (1.0 + b)

    Y_factor = J0pt5 + A * J2pt5
    J_factor = Y0pt5 + A * Y2pt5

    B = pi / 2.0
    C = pow(2.0, 3.0 + 2.0 * b) / (3.0 + 2.0 * b) / (2.0 + b)
    D = gamma(2.5 + b) * gamma(2.5 + b)
    E = pow(q.k * r.k * cs_sq * eta_response, -0.5 - b)

    F = -B * C * D * E
    x = k.k * eta_response

    value = F * (yv(0.5 + b, x) * Y_factor - jv(0.5 + b, x) * J_factor)
    return value


def _extract_z(z: Union[type(None), redshift, float]) -> str:
    if z is None:
        return "(not set)"

    if isinstance(z, redshift):
        return f"{z.z:.5g}"

    if isinstance(z, float):
        return f"{z:.5g}"

    return f"{float(z):..5g}"


def numeric_quad_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    max_z: float,
    min_z: float,
    tol: float,
) -> dict:
    source_f = source.functions
    Gk_f = GkPolicy.functions
    model_f = model.functions

    if GkPolicy.type not in ["numeric", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk object is not of "numeric" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.numerical_Gk is None:
        Gk: GkSource = GkPolicy._source_proxy.get()
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk_f.numerical_Gk is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.numerical_region is None:
        Gk: GkSource = GkPolicy._source_proxy.get()
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk_f.numerical_region is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    region_max_z, region_min_z = Gk_f.numerical_region
    if max_z > region_max_z + DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a numerical solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if min_z < region_min_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a numerical solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    def RHS(log_z_source, state, supervisor: QuadSourceSupervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            current_value = state[0]

            if supervisor.notify_available:
                supervisor.message(
                    log_z_source, f"current state: current value = {current_value:.5g}"
                )

            Green = Gk_f.numerical_Gk(log_z_source, z_is_log=True)
            H = model_f.Hubble(exp(log_z_source) - 1.0)
            H_sq = H * H
            f = source_f.source(log_z_source, z_is_log=True)

        return [Green * f / H_sq]

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    with QuadSourceSupervisor(
        k, q, r, "numeric quad", z_response, min_z, max_z
    ) as supervisor:
        state = [0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(log_min_z, log_max_z),
            t_eval=[log_max_z],
            y0=state,
            dense_output=True,
            atol=tol,
            rtol=tol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_QuadSource_integral: quadrature did not terminate successfully | error at log(1+z)={sol.t[0]:.5g}, "{sol.message}"'
        )

    if sol.t[0] < log_max_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: quadrature did not terminate at expected redshift log(1+z)={log_max_z:.5g} (final log(1+z)={sol.t[0]:.5g})"
        )

    if len(sol.sol(log_max_z)) != 1:
        raise RuntimeError(
            f"compute_QuadSource_integral: solution does not have expected number of members (expected 1, found {len(sol.sol(log_max_z))})"
        )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "value": (1.0 + z_response.z) * sol.sol(log_max_z)[0],
    }


def WKB_quad_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    max_z: float,
    min_z: float,
    tol: float,
) -> dict:
    source_f = source.functions
    Gk_f = GkPolicy.functions
    model_f = model.functions

    if GkPolicy.type not in ["WKB", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate WKB quadrature, but Gk object is not of "WKB" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.WKB_Gk is None:
        Gk: GkSource = GkPolicy._source_proxy.get()
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but Gk_f.WKB_Gk is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(GkPolicy.crossover_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.WKB_region is None:
        Gk: GkSource = GkPolicy._source_proxy.get()
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but Gk_f.WKB_region is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}), z_crossover={_extract_z(GkPolicy.crossover_z)} [domain={max_z:.5g}, {min_z:.5g}]"
        )

    region_max_z, region_min_z = Gk_f.WKB_region
    if max_z > region_max_z + DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if min_z < region_min_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    def RHS(log_z_source, state, supervisor: QuadSourceSupervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            current_value = state[0]

            if supervisor.notify_available:
                supervisor.message(
                    log_z_source, f"current state: current value = {current_value:.5g}"
                )

            Green = Gk_f.WKB_Gk(log_z_source, z_is_log=True)
            H = model_f.Hubble(exp(log_z_source) - 1.0)
            H_sq = H * H
            f = source_f.source(log_z_source, z_is_log=True)

        return [Green * f / H_sq]

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    with QuadSourceSupervisor(
        k, q, r, "WKB quad", z_response, log_min_z, log_max_z
    ) as supervisor:
        state = [0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(log_min_z, log_max_z),
            t_eval=[log_max_z],
            y0=state,
            dense_output=True,
            atol=tol,
            rtol=tol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_QuadSource_integral: quadrature did not terminate successfully | error at log(1+z)={sol.t[0]}, "{sol.message}"'
        )

    if sol.t[0] < log_max_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: quadrature did not terminate at expected redshift log(1+z)={log_max_z:.5g} (final log(1+z)={sol.t[0]:.3g})"
        )

    if len(sol.sol(log_max_z)) != 1:
        raise RuntimeError(
            f"compute_QuadSource_integral: solution does not have expected number of members (expected 1, found {len(sol.sol(log_max_z))})"
        )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "value": (1.0 + z_response.z) * sol.sol(log_max_z)[0],
    }


def WKB_Levin_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    GkPolicy: GkSourcePolicyData,
    z_response: redshift,
    max_z: float,
    min_z: float,
    tol: float,
) -> dict:
    source_f = source.functions
    Gk_f = GkPolicy.functions
    model_f = model.functions

    if GkPolicy.type not in ["WKB", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but Gk object is not of "WKB" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.theta is None:
        Gk: GkSource = GkPolicy._source_proxy.get()
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but Gk_f.theta is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(GkPolicy.crossover_z)}, z_Levin={_extract_z(GkPolicy.Levin_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.WKB_region is None:
        Gk: GkSource = GkPolicy._source_proxy.get()
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but Gk_f.WKB_region is absent (type={GkPolicy.type}, quality={GkPolicy.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(GkPolicy.crossover_z)}, z_Levin={_extract_z(GkPolicy.Levin_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    region_max_z, region_min_z = Gk_f.WKB_region
    if max_z > region_max_z + DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if min_z < region_min_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    x_span = (log_min_z, log_max_z)

    def Levin_f(log_z_source: float) -> float:
        H = model_f.Hubble(exp(log_z_source) - 1.0)
        H_sq = H * H
        f = source_f.source(log_z_source, z_is_log=True)

        return f / H_sq

    def Levin_theta(log_z_source: float) -> float:
        return Gk_f.theta(log_z_source, z_is_log=True)

    data = adaptive_levin_sincos(
        x_span,
        [Levin_f, lambda x: 0.0],
        Levin_theta,
        tol=tol,
        chebyshev_order=12,
        notify_label=f"k={k.k_inv_Mpc:.3g}/Mpc, q={q.k_inv_Mpc:.3g}/Mpc, r={r.k_inv_Mpc:.3g}/Mpc @ z_response={z_response.z:.5g}",
    )

    return {
        "data": LevinData(
            num_regions=len(data["regions"]),
            evaluations=data["evaluations"],
            elapsed=data["elapsed"],
        ),
        "value": (1.0 + z_response.z) * data["value"],
    }


class QuadSourceIntegral(DatastoreObject):
    def __init__(
        self,
        payload,
        model: ModelProxy,
        policy: GkSourcePolicy,
        z_response: redshift,
        z_source_max: redshift,
        k: wavenumber_exit_time,
        q: wavenumber_exit_time,
        r: wavenumber_exit_time,
        tol: tolerance,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._model_proxy = model
        self._policy = policy

        self._k_exit = k
        self._q_exit = q
        self._r_exit = r

        self._z_response = z_response
        self._z_source_max = z_source_max

        if payload is None:
            DatastoreObject.__init__(self, None)

            self._total = None
            self._numeric_quad = None
            self._WKB_quad = None
            self._WKB_Levin = None

            self._numeric_quad_data = None
            self._WKB_quad_data = None
            self._WKB_Levin_data = None

            self._analytic_rad = None

            self._compute_time = None
            self._data_serial = None
            self._source_serial = None

            self._metadata = {}

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._total = payload["total"]
            self._numeric_quad = payload["numeric_quad"]
            self._WKB_quad = payload["WKB_quad"]
            self._WKB_Levin = payload["WKB_Levin"]

            self._analytic_rad = payload["analytic_rad"]

            self._source_serial = payload["source_serial"]
            self._data_serial = payload["data_serial"]

            self._numeric_quad_data = payload["numeric_quad_data"]
            self._WKB_quad_data = payload["WKB_quad_data"]
            self._WKB_Levin_data = payload["WKB_Levin_data"]

            self._compute_time = payload["compute_time"]

            self._metadata = payload["metadata"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._tol = tol

        self._compute_ref = None

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def q(self) -> wavenumber:
        return self._q_exit.k

    @property
    def r(self) -> wavenumber:
        return self._r_exit.k

    @property
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def total(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._total

    @property
    def numeric_quad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._numeric_quad

    @property
    def WKB_quad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_quad

    @property
    def WKB_Levin(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_Levin

    @property
    def analytic_rad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._analytic_rad

    @property
    def data_serial(self) -> Optional[int]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._data_serial

    @property
    def source_serial(self) -> Optional[int]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._source_serial

    @property
    def numeric_quad_data(self) -> Optional[IntegrationData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._numeric_quad_data

    @property
    def WKB_quad_data(self) -> Optional[IntegrationData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_quad_data

    @property
    def WKB_Levin_data(self) -> Optional[LevinData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_Levin_data

    @property
    def compute_time(self) -> Optional[float]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._compute_time

    @property
    def metadata(self) -> dict:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._metadata

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    def compute(self, payload, label: Optional[str] = None):
        if self._total is not None:
            raise RuntimeError(
                "QuadSourceIntegral: compute() called, but value has already been computed"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        source: QuadSource = payload["source"]
        GkPolicy: GkSourcePolicyData = payload["GkPolicy"]

        if self._policy.store_id != GkPolicy.policy.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied GkSourcePolicyData object does not match specified policy (required policy store_id={self._policy.store_id}, supplied GkSourcePolicyData object has policy store_id={GkPolicy.policy.store_id})"
            )

        Gk = GkPolicy._source_proxy.get()

        # TODO: improve compatibility check between source and Gk
        if self._z_response.store_id != Gk.z_response.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied GkSource object does not match specified z_response (z_response={self._z_response.z:.5g}, GkSource object evaluated at z_response={Gk.z_response.z:.5g})"
            )

        if source.z_sample.max.z < self._z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied quadratic source term has maximum z_source={source.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.z_sample.max.z < self._z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied Gk has maximum z_source={Gk.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.k.store_id != self._k_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied Gk is evaluated for a k-mode that does not match the required value (supplied Gk is for k={Gk.k.k_inv_Mpc:.3g}/Mpc [store_id={Gk.k.store_id}], required value is k={self._k_exit.k.k_inv_Mpc:.3g}/Mpc [store_id]{self._k_exit.k.store_id})"
            )

        if source.q.store_id != self._q_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for a q-mode that does not match the required value (supplied source is for q={source.q.k_inv_Mpc:.3g}/Mpc [store_id={source.q.store_id}], required value is k={self._q_exit.k.k_inv_Mpc:.3g}/Mpc [store_id]{self._q_exit.k.store_id})"
            )

        if source.r.store_id != self._r_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for an r-mode that does not match the required value (supplied source is for r={source.r.k_inv_Mpc:.3g}/Mpc [store_id={source.r.store_id}], required value is k={self._r_exit.k.k_inv_Mpc:.3g}/Mpc [store_id]{self._r_exit.k.store_id})"
            )

        self._compute_ref = compute_QuadSource_integral.remote(
            self._model_proxy,
            self._k_exit,
            self._q_exit,
            self._r_exit,
            source,
            GkPolicy,
            self._z_response,
            self._z_source_max,
        )

        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "QuadSourceIntegral: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None

        payload = ray.get(self._compute_ref)
        self._compute_ref = None

        self._total = payload["total"]
        self._numeric_quad = payload["numeric_quad"]
        self._WKB_quad = payload["WKB_quad"]
        self._WKB_Levin = payload["WKB_Levin"]

        self._analytic_rad = payload["analytic_rad"]

        self._data_serial = payload["GkPolicy_serial"]
        self._source_serial = payload["source_serial"]

        self._numeric_quad_data = payload["numeric_quad_data"]
        self._WKB_quad_data = payload["WKB_quad_data"]
        self._WKB_Levin_data = payload["WKB_Levin_data"]

        self._compute_time = payload["compute_time"]
        self._metadata = payload["metadata"]
