from datetime import datetime
from typing import Optional, List, Union

import ray
from math import log, exp, pow, sqrt, pi, gamma
from numpy.linalg import LinAlgError
from scipy.special import yv, jv

from AdaptiveLevin import adaptive_levin_sincos
from AdaptiveLevin.bessel_phase import bessel_phase
from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy, ModelFunctions
from ComputeTargets.GkSource import GkSource
from ComputeTargets.GkSourcePolicyData import GkSourcePolicyData, GkSourceFunctions
from ComputeTargets.QuadSource import QuadSource, QuadSourceFunctions
from ComputeTargets.QuadSourceIntegral_debug import (
    _bessel_function_plot,
    _three_bessel_plot,
)
from CosmologyConcepts import wavenumber, wavenumber_exit_time, redshift
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance, GkSourcePolicy
from Quadrature.integration_metadata import IntegrationData, LevinData
from Quadrature.simple_quadrature import simple_quadrature
from defaults import (
    DEFAULT_QUADRATURE_RTOL,
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_QUADRATURE_ATOL,
)
from utilities import WallclockTimer

LEVIN_MIN_2PI_CYCLES = 10
LEVIN_MIN_PHASE_DIFF = LEVIN_MIN_2PI_CYCLES * 2.0 * pi

CHEBYSHEV_ORDER = 64
LEVIN_RELERR = 1e-8
LEVIN_ABSERR = 1e-23


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
    atol: float = DEFAULT_QUADRATURE_ATOL,
    rtol: float = DEFAULT_QUADRATURE_RTOL,
) -> dict:
    # we are entitled to assume that the GkSource embedded in GkPolicy and source are evaluated at the same z_response
    # also that source and Gk have z_samples at least as far back as z_source_max
    model: BackgroundModel = model_proxy.get()

    model_f: ModelFunctions = model.functions
    Gk_f: GkSourceFunctions = GkPolicy.functions

    with WallclockTimer() as timer:
        numeric_quad: float = 0.0
        WKB_quad: float = 0.0
        WKB_Levin: float = 0.0

        numeric_quad_data = None
        WKB_quad_data = None
        WKB_Levin_data = None

        if GkPolicy.type == "numeric":
            regions = [(z_source_max, z_response), (None, None), (None, None)]

        elif GkPolicy.type == "WKB":
            Levin_z: Optional[float] = GkPolicy.Levin_z

            if Levin_z is not None:
                # unlikely to be worth doing Levin method (or that we will get an especially accurate result)
                # unless the phase goes through enough cycles
                phase_diff = Gk_f.theta(z_response.z) - Gk_f.theta(Levin_z)
                if phase_diff > LEVIN_MIN_PHASE_DIFF:
                    regions = [
                        (None, None),
                        (z_source_max, Levin_z),
                        (Levin_z, z_response),
                    ]
                else:
                    regions = [(None, None), (z_source_max, z_response), (None, None)]
            else:
                regions = [(None, None), (z_source_max, z_response), (None, None)]

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

            if Levin_z is not None:
                phase_diff = Gk_f.theta(z_response.z) - Gk_f.theta(Levin_z)
                if phase_diff > LEVIN_MIN_PHASE_DIFF:
                    regions = [
                        (z_source_max, crossover_z),
                        (
                            crossover_z if crossover_z is not None else z_source_max,
                            Levin_z,
                        ),
                        (Levin_z, z_response),
                    ]
                else:
                    regions = [
                        (z_source_max, crossover_z),
                        (
                            crossover_z if crossover_z is not None else z_source_max,
                            z_response,
                        ),
                        (None, None),
                    ]
            else:
                regions = [
                    (z_source_max, crossover_z),
                    (
                        crossover_z if crossover_z is not None else z_source_max,
                        z_response,
                    ),
                    (None, None),
                ]

        else:
            raise NotImplementedError(f"Gk {GkPolicy.type} not implemented")

        # REGION 1: ORDINARY QUADRATURE OF NUMERICAL RESULT
        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) / get_z(min_z) > 1.0 + DEFAULT_QUADRATURE_RTOL
        ):
            # now = time.time()
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
                atol=atol,
                rtol=rtol,
            )
            numeric_quad = payload["value"]
            numeric_quad_data = payload["data"]

        # REGION 2: ORDINARY QUADRATURE OF WKB RESULTS
        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) / get_z(min_z) > 1.0 + DEFAULT_QUADRATURE_RTOL
        ):
            # now = time.time()
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
                atol=atol,
                rtol=rtol,
            )
            WKB_quad = payload["value"]
            WKB_quad_data = payload["data"]

        # REGION 3: LEVIN QUADRATURE
        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) / get_z(min_z) > 1.0 + DEFAULT_QUADRATURE_RTOL
        ):
            # now = time.time()
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
                atol=atol,
                rtol=rtol,
            )
            WKB_Levin = payload["value"]
            WKB_Levin_data = payload["data"]

        # calculate analytic approximation for radiation
        analytic_data = analytic_integral(
            model,
            k.k,
            q.k,
            r.k,
            z_response,
            max_z=z_source_max,
            min_z=z_response,
            b=0.0,
            rtol=rtol,
            atol=atol,
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
        "eta_source_max": model_f.tau(z_source_max.z),
        "eta_response": model_f.tau(z_response.z),
        "analytic_rad": analytic_data["value"],
        "compute_time": timer.elapsed,
        "metadata": {"analytic": analytic_data["metadata"]},
    }


XCUT_SEARCH_TOLERANCE = 1e-2
XCUT_SEARCH_MAX_ITERATIONS = 200
XCUT_SEARCH_LEVIN_THRESHOLD = 0.01


def find_x_cut(x_min, x_max, dphase_Gk, dphase_Tk):
    iter = 0

    while (x_max - x_min) > XCUT_SEARCH_TOLERANCE and iter < XCUT_SEARCH_MAX_ITERATIONS:
        assert x_max > x_min

        dp_Gk_min = dphase_Gk(x_min)
        dp_Tk_min = dphase_Tk(x_min)

        if (
            dp_Gk_min > XCUT_SEARCH_LEVIN_THRESHOLD
            and dp_Tk_min > XCUT_SEARCH_LEVIN_THRESHOLD
        ):
            return x_min

        dp_Gk_max = dphase_Gk(x_max)
        dp_Tk_max = dphase_Tk(x_max)

        if (
            dp_Gk_max < XCUT_SEARCH_LEVIN_THRESHOLD
            and dp_Tk_max < XCUT_SEARCH_LEVIN_THRESHOLD
        ):
            raise RuntimeError(
                "rate of change of phase is too small even at right-hand endpoint"
            )

        x_mid = (x_min + x_max) / 2.0

        dp_Gk_mid = dphase_Gk(x_mid)
        dp_Tk_mid = dphase_Tk(x_mid)

        if (
            dp_Gk_mid > XCUT_SEARCH_LEVIN_THRESHOLD
            and dp_Tk_mid > XCUT_SEARCH_LEVIN_THRESHOLD
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
    dp_Gk = dphase_Gk(x_cut_candidate)
    dp_Tk = dphase_Tk(x_cut_candidate)

    if dp_Gk < XCUT_SEARCH_LEVIN_THRESHOLD or dp_Tk < XCUT_SEARCH_LEVIN_THRESHOLD:
        # x_max should be guaranteed to satisfy dphase_Gk,Tk > XCUT_SEARCH_LEVIN_THRESHOLD
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
    timestamp: datetime,
    atol: float,
    rtol: float,
    mode: str = "default",
    debug_plots: bool = False,
):
    if nu_type not in ["0pt5", "2pt5"]:
        raise RuntimeError('phase_data should be "0pt5" or "2pt5"')

    if mode not in ["default", "quad", "Levin"]:
        mode = "default"

    if debug_plots:
        _bessel_function_plot(phase_data, b, timestamp)
        _three_bessel_plot(k, q, r, min_eta, max_eta, b, phase_data, nu_type, timestamp)

    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    metadata = {}

    phase_data_Gk = phase_data["0pt5"]
    phase_data_Tk = phase_data[nu_type]

    phase_Gk = phase_data_Gk["phase"]
    phase_Tk = phase_data_Tk["phase"]

    # unlikely to be worth doing Levin (or to yield an especially accurate result)
    # unless the phase goes through at least quite a few 2pi cycles
    phase_diff_Gk = phase_Gk(k.k * max_eta) - phase_Gk(k.k * min_eta)
    phase_diff_Tk_q = phase_Tk(q.k * cs * max_eta) - phase_Tk(q.k * cs * min_eta)
    phase_diff_Tk_r = phase_Tk(r.k * cs * max_eta) - phase_Tk(r.k * cs * min_eta)

    if phase_diff_Gk < -1.0:
        raise RuntimeError(
            f"_three_bessel_integrals: phase_diff_Gk < 0.0 (value={phase_diff_Gk:.5g}"
        )
    if phase_diff_Tk_q < -1.0:
        raise RuntimeError(
            f"_three_bessel_integrals: phase_diff_Tk_q < 0.0 (value={phase_diff_Tk_q:.5g}"
        )
    if phase_diff_Tk_r < -1.0:
        raise RuntimeError(
            f"_three_bessel_integrals: phase_diff_Tk_q < 0.0 (value={phase_diff_Tk_r:.5g}"
        )

    dphase_Gk = phase_data_Gk["dphase"]
    dphase_Tk = phase_data_Tk["dphase"]

    min_k_mode = min(k.k, q.k, r.k)
    max_k_mode = max(k.k, q.k, r.k)

    min_x = min_k_mode * min_eta * cs
    max_x = max_k_mode * max_eta

    x_cut = find_x_cut(min_x, max_x, dphase_Gk, dphase_Tk)
    eta_cut = x_cut / min_k_mode / cs

    # metadata["x_cut"] = x_cut
    # metadata["eta_cut"] = eta_cut
    # metadata["min_eta"] = min_eta
    # metadata["max_eta"] = max_eta
    # metadata["min_x"] = min_x
    # metadata["max_x"] = max_x

    if mode == "Levin" or (
        mode == "default" and eta_cut / min_eta <= 1.0 + DEFAULT_FLOAT_PRECISION
    ):
        # quad_comparison = _three_bessel_quad(
        #     k, q, r, min_eta=min_eta, max_eta=max_eta, b=b, nu_type=nu_type, tol=tol
        # )

        # if it is worth doing a Levin integration (or we are explicitly instructed to use a Levin method), do so
        if mode == "Levin" or (
            phase_diff_Gk > LEVIN_MIN_PHASE_DIFF
            or phase_diff_Tk_q > LEVIN_MIN_PHASE_DIFF
            or phase_diff_Tk_r > LEVIN_MIN_PHASE_DIFF
        ):
            try:
                Levin = _three_bessel_Levin(
                    k,
                    q,
                    r,
                    min_eta=min_eta,
                    max_eta=max_eta,
                    b=b,
                    phase_data_Gk=phase_data_Gk,
                    phase_data_Tk=phase_data_Tk,
                    x_cut=x_cut,
                    atol=atol,
                    rtol=rtol,
                )
                metadata["Levin_J"] = Levin[0]
                metadata["Levin_Y"] = Levin[1]
                # metadata["quad_comparison_J"] = quad_comparison[0]
                # metadata["quad_comparison_Y"] = quad_comparison[1]
                return {"J": Levin[0], "Y": Levin[1], "metadata": metadata}
            except LinAlgError as e:
                print(
                    f"!! three_bessel_integrals: linear algebra error k={k.k_inv_Mpc:.5g}/Mpc, q={q.k_inv_Mpc:.5g}/Mpc, r={r.k_inv_Mpc:.5g}, min_eta={min_eta:.5g}, max_eta={max_eta:.5g}, x_cut={x_cut:.5g}, nu_type={nu_type}, mode={mode}"
                )
                raise e

        quad = _three_bessel_quad(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            nu_type=nu_type,
            atol=atol,
            rtol=rtol,
        )
        metadata["quad_J"] = quad[0]
        metadata["quad_Y"] = quad[1]
        return {"J": quad[0], "Y": quad[1], "metadata": metadata}

    if mode == "quad" or (
        mode == "default" and eta_cut / max_eta >= 1.0 - DEFAULT_FLOAT_PRECISION
    ):
        quad = _three_bessel_quad(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            nu_type=nu_type,
            atol=atol,
            rtol=rtol,
        )
        metadata["quad_J"] = quad[0]
        metadata["quad_Y"] = quad[1]
        return {"J": quad[0], "Y": quad[1], "metadata": metadata}

    assert mode == "default"
    # can assume eta_cut falls between min_eta and max_eta

    # quad_comparison = _three_bessel_quad(
    #     k, q, r, min_eta=eta_cut, max_eta=max_eta, b=b, nu_type=nu_type, tol=tol
    # )

    # if it is worth doing a Levin integration between eta_cut and max_eta, do so
    # otherwise, do a numerical quadrature everywhere
    if (
        phase_diff_Gk > LEVIN_MIN_PHASE_DIFF
        or phase_diff_Tk_q > LEVIN_MIN_PHASE_DIFF
        or phase_diff_Tk_r > LEVIN_MIN_PHASE_DIFF
    ):
        quad = _three_bessel_quad(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=eta_cut,
            b=b,
            nu_type=nu_type,
            atol=atol,
            rtol=rtol,
        )
        Levin = _three_bessel_Levin(
            k,
            q,
            r,
            min_eta=eta_cut,
            max_eta=max_eta,
            b=b,
            phase_data_Gk=phase_data_Gk,
            phase_data_Tk=phase_data_Tk,
            x_cut=x_cut,
            atol=atol,
            rtol=rtol,
        )
        metadata["Levin_J"] = Levin[0]
        metadata["Levin_Y"] = Levin[1]
        metadata["quad_J"] = quad[0]
        metadata["quad_Y"] = quad[1]
        # metadata["quad_comparison_J"] = quad_comparison[0]
        # metadata["quad_comparison_Y"] = quad_comparison[1]
        return {
            "J": quad[0] + Levin[0],
            "Y": quad[1] + Levin[1],
            "metadata": metadata,
        }

    quad = _three_bessel_quad(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        nu_type=nu_type,
        atol=atol,
        rtol=rtol,
    )
    metadata["quad_J"] = quad[0]
    metadata["quad_Y"] = quad[1]
    return {"J": quad[0], "Y": quad[1], "metadata": metadata}


def _three_bessel_Levin(
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    min_eta: float,
    max_eta: float,
    b: float,
    phase_data_Gk,
    phase_data_Tk,
    x_cut: float,
    atol: float,
    rtol: float,
):
    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    x_span = (log(min_eta), log(max_eta))

    min_k_mode = min(k.k, q.k, r.k)
    if min_k_mode * min_eta * cs > (1.0 + DEFAULT_FLOAT_PRECISION) * x_cut:
        raise RuntimeError(
            f"!! three_bessel_Levin: ERROR: smallest required x is smaller than x_cut (smallest x={min_k_mode * min_eta * cs:.5g}, x_cut={x_cut:.5g}, min_eta={min_eta:.5g}, min_k_mode={min_k_mode:.5g}, cs={cs:.5g}, coeff={min_k_mode*cs:.5g}, x_cut/coeff={x_cut/(min_k_mode*cs):.5g})"
        )

    phase_Gk = phase_data_Gk["phase"]
    dphase_Gk = phase_data_Gk["dphase"]

    phase_Tk = phase_data_Tk["phase"]
    dphase_Tk = phase_data_Tk["dphase"]

    def Levin_f(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        beta1 = dphase_Gk(x1)
        beta2 = dphase_Tk(x2)
        beta3 = dphase_Tk(x3)

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

        return phase_Gk(x1) + phase_Tk(x2) + phase_Tk(x3)

    J1_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta=phase1,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )
    Y1_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta=phase1,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )

    def phase2(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk(x1) + phase_Tk(x2) - phase_Tk(x3)

    J2_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta=phase2,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )
    Y2_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta=phase2,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )

    def phase3(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk(x1) - phase_Tk(x2) + phase_Tk(x3)

    J3_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta=phase3,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )
    Y3_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta=phase3,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )

    def phase4(log_eta: float):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        return phase_Gk(x1) - phase_Tk(x2) - phase_Tk(x3)

    J4_data = adaptive_levin_sincos(
        x_span,
        f=[Levin_f, lambda x: 0.0],
        theta=phase4,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )
    Y4_data = adaptive_levin_sincos(
        x_span,
        f=[lambda x: 0.0, lambda x: -Levin_f(x)],
        theta=phase4,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
    )

    J1_value = J1_data["value"]
    J2_value = J2_data["value"]
    J3_value = J3_data["value"]
    J4_value = J4_data["value"]

    Y1_value = Y1_data["value"]
    Y2_value = Y2_data["value"]
    Y3_value = Y3_data["value"]
    Y4_value = Y4_data["value"]

    norm_factor = pow(2.0 / pi, 3.0 / 2.0) / sqrt(k.k * q.k * r.k) / cs / 4.0
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
    atol: float,
    rtol: float,
):
    assert min_eta <= max_eta

    cs = sqrt((1.0 - b) / (1.0 + b) / 3.0)
    log_min_eta = log(min_eta)
    log_max_eta = log(max_eta)

    nu_types = {"0pt5": 0.5, "2pt5": 2.5}
    nu = nu_types[nu_type]

    def J_integrand(log_eta):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        A = pow(eta, 1.5 - b)
        dJ = jv(0.5 + b, x1) * jv(nu + b, x2) * jv(nu + b, x3)

        return A * dJ

    def Y_integrand(log_eta):
        eta = exp(log_eta)

        x1 = k.k * eta
        x2 = q.k * cs * eta
        x3 = r.k * cs * eta

        A = pow(eta, 1.5 - b)
        dY = yv(0.5 + b, x1) * jv(nu + b, x2) * jv(nu + b, x3)

        return A * dY

    data = simple_quadrature(
        [J_integrand, Y_integrand],
        a=log_min_eta,
        b=log_max_eta,
        atol=atol,
        rtol=rtol,
        label=f"three_bessel_quad for  k={k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}), q={q.k_inv_Mpc:.5g}/Mpc (store_id={q.store_id}), r={r.k_inv_Mpc:.5g}/Mpc (store_id={r.store_id})",
        method="quad",
    )

    return data["value"]


def analytic_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    z_response: redshift,
    max_z: redshift,
    min_z: redshift,
    b: float,
    rtol: float,
    atol: float,
):
    functions: ModelFunctions = model.functions
    min_eta: float = functions.tau(max_z.z)
    max_eta: float = functions.tau(min_z.z)

    eta_response: float = functions.tau(z_response.z)

    cs_sq = (1.0 - b) / (1.0 + b) / 3.0

    max_x = max(k.k, q.k, r.k) * max_eta
    if max_x > 1e5:
        print(
            f"!! analytic: WARNING: max_x very large, {max_x:.5g}. This may lead to issues with calculation of the Liouville-Green phase function for the Bessel functions."
        )

    phase_data_0pt5 = bessel_phase(0.5 + b, max_x + 10.0)
    phase_data_2pt5 = bessel_phase(2.5 + b, max_x + 10.0)

    timestamp = datetime.now().replace(microsecond=0)

    data0pt5 = _three_bessel_integrals(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        phase_data={"0pt5": phase_data_0pt5, "2pt5": phase_data_2pt5},
        nu_type="0pt5",
        timestamp=timestamp,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        mode="default",
    )
    data2pt5 = _three_bessel_integrals(
        k,
        q,
        r,
        min_eta=min_eta,
        max_eta=max_eta,
        b=b,
        phase_data={"0pt5": phase_data_0pt5, "2pt5": phase_data_2pt5},
        nu_type="2pt5",
        timestamp=timestamp,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        mode="default",
    )

    metadata = {
        # "k": k.k,
        # "q": q.k,
        # "r": r.k,
        # "max_z": max_z.z,
        # "min_z": min_z.z,
        "max_eta": max_eta,
        "min_eta": min_eta,
        # "eta_response": eta_response,
        "0pt5": data0pt5,
        "2pt5": data2pt5,
    }

    A = (2.0 + b) / (1.0 + b)

    Y_factor = data0pt5["J"] + A * data2pt5["J"]
    J_factor = data0pt5["Y"] + A * data2pt5["Y"]

    B = pi / 2.0
    C = pow(2.0, 3.0 + 2.0 * b) / (3.0 + 2.0 * b) / (2.0 + b)
    D = gamma(2.5 + b) * gamma(2.5 + b)
    E = pow(q.k * r.k * cs_sq * eta_response, -0.5 - b)

    F = -B * C * D * E
    x = k.k * eta_response

    value = F * (yv(0.5 + b, x) * Y_factor - jv(0.5 + b, x) * J_factor)
    return {"value": value, "metadata": metadata}


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
    atol: float,
    rtol: float,
) -> dict:
    source_f: QuadSourceFunctions = source.functions
    Gk_f: GkSourceFunctions = GkPolicy.functions
    model_f: ModelFunctions = model.functions

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

    def integrand(log_z_source) -> float:
        Green = Gk_f.numerical_Gk(log_z_source, z_is_log=True)
        H = model_f.Hubble(exp(log_z_source) - 1.0)
        H_sq = H * H
        f = source_f.source(log_z_source, z_is_log=True)

        return Green * f / H_sq

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    data = simple_quadrature(
        integrand,
        a=log_min_z,
        b=log_max_z,
        atol=atol,
        rtol=rtol,
        label=f"numeric_quad_integral for k={k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}), q={q.k_inv_Mpc:.5g}/Mpc (store_id={q.store_id}), r={r.k_inv_Mpc:.5g}/Mpc (store_id={r.store_id})",
        method="quad",
    )

    data["value"] = (1.0 + z_response.z) * data["value"]

    return data


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
    atol: float,
    rtol: float,
) -> dict:
    source_f: QuadSourceFunctions = source.functions
    Gk_f: GkSourceFunctions = GkPolicy.functions
    model_f: ModelFunctions = model.functions

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

    def integrand(log_z_source) -> float:
        Green = Gk_f.WKB_Gk(log_z_source, z_is_log=True)
        H = model_f.Hubble(exp(log_z_source) - 1.0)
        H_sq = H * H
        f = source_f.source(log_z_source, z_is_log=True)

        return Green * f / H_sq

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    data = simple_quadrature(
        integrand,
        a=log_min_z,
        b=log_max_z,
        atol=atol,
        rtol=rtol,
        label=f"WKB_quad_integral for k={k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}), q={q.k_inv_Mpc:.5g}/Mpc (store_id={q.store_id}), r={r.k_inv_Mpc:.5g}/Mpc (store_id={r.store_id})",
        method="quad",
    )

    data["value"] = (1.0 + z_response.z) * data["value"]

    return data


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
    atol: float,
    rtol: float,
) -> dict:
    source_f: QuadSourceFunctions = source.functions
    Gk_f: GkSourceFunctions = GkPolicy.functions
    model_f: ModelFunctions = model.functions

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
        sin_ampl = Gk_f.sin_amplitude(log_z_source, z_is_log=True)

        return sin_ampl * f / H_sq

    def Levin_theta(log_z_source: float) -> float:
        return Gk_f.theta(log_z_source, z_is_log=True)

    data = adaptive_levin_sincos(
        x_span,
        [Levin_f, lambda x: 0.0],
        Levin_theta,
        atol=LEVIN_ABSERR,
        rtol=LEVIN_RELERR,
        chebyshev_order=CHEBYSHEV_ORDER,
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
        atol: tolerance,
        rtol: tolerance,
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

            self._eta_source_max = None
            self._eta_response = None
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

            self._eta_source_max = payload["eta_source_max"]
            self._eta_response = payload["eta_response"]
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

        self._atol = atol
        self._rtol = rtol

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
    def z_source_max(self) -> redshift:
        return self._z_source_max

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
    def eta_source_max(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._eta_source_max

    @property
    def eta_response(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._eta_response

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
                f"QuadSourceIntegral: supplied Gk is evaluated for a k-mode that does not match the required value (supplied Gk is for k={Gk.k.k_inv_Mpc:.3g}/Mpc [store_id={Gk.k.store_id}], required value is k={self._k_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={self._k_exit.k.store_id}])"
            )

        if source.q.store_id != self._q_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for a q-mode that does not match the required value (supplied source is for q={source.q.k_inv_Mpc:.3g}/Mpc [store_id={source.q.store_id}], required value is k={self._q_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={self._q_exit.k.store_id}])"
            )

        if source.r.store_id != self._r_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for an r-mode that does not match the required value (supplied source is for r={source.r.k_inv_Mpc:.3g}/Mpc [store_id={source.r.store_id}], required value is k={self._r_exit.k.k_inv_Mpc:.3g}/Mpc [store_id={self._r_exit.k.store_id}])"
            )

        self._compute_ref = compute_QuadSource_integral.remote(
            self._model_proxy,
            self._k_exit,
            self._q_exit,
            self._r_exit,
            source=source,
            GkPolicy=GkPolicy,
            z_response=self._z_response,
            z_source_max=self._z_source_max,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
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

        self._eta_source_max = payload["eta_source_max"]
        self._eta_response = payload["eta_response"]
        self._analytic_rad = payload["analytic_rad"]

        self._data_serial = payload["GkPolicy_serial"]
        self._source_serial = payload["source_serial"]

        self._numeric_quad_data = payload["numeric_quad_data"]
        self._WKB_quad_data = payload["WKB_quad_data"]
        self._WKB_Levin_data = payload["WKB_Levin_data"]

        self._compute_time = payload["compute_time"]
        self._metadata = payload["metadata"]
