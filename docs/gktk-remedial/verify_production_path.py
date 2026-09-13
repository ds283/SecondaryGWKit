"""
Layer 1 of the ``prompts/GkTk-remedial`` close-out (prompt 13 §1): the production-path
measurements, offline, on both production background models.

Everything below runs through the **production** functions -- the undecorated
``Quadrature.integrators.WKB_phase_function``, ``ComputeTargets.BackgroundModel``'s
``compute_background`` and its ``TablePrimitive`` accessors, the ``store()`` algebra of
``TkWKBIntegration``, ``ComputeTargets.primitive_phase.PrimitivePhase`` and
``ComputeTargets.TkSourceFunctions`` -- with the stand-in *models* of
``ComputeTargets/tests/wkb_reference.py`` and the stand-in *fixtures* of
``ComputeTargets/tests/test_gk_wkb_phase.py`` and ``test_tk_wkb_phase.py``. No Ray, no
datastore: the live pipeline is Layer 2 (``docs/source-remediation-verification/scoped_pipeline_run.py``).

Sections, selectable with ``--section`` (default: all, in this order):

``primitives``  the tables themselves at the production nodes, against prompt 01's references:
                tau, one-interval Delta-tau, cs_tau, F and rho, plus the interval accessor's
                cost per call (README §6 rows 1-5).

``producers``   review §4 and §12.3 re-measured. Phase error of the stored ``theta_G`` and
                ``theta_T`` against prompt 01's references at the JSON checkpoints, on
                ``LambdaCDMModel`` and ``QCDModel`` at k = 1e5, 1e7, 3e8/Mpc, with the "before"
                column taken from the review's own tables; plus the cost per object (best-of-N
                wall time and integrand evaluations, both with and without the per-k residual
                table build -- ``[07-tk-per-object-cost-is-all-setup]``).

``stored``      the friction integral F and the residual rho_T **as stored**, against the
                references (prompt 13 §1 item 3).

``consumers``   the consumers at production x (prompt 13 §1 item 2): ``PrimitivePhase`` for the
                Green's function at fixed z_response = 0.1 over the source grid, and
                ``TkSourceFunctions.phase`` over its WKB region, each scored at 10 points per
                production interval against the producer evaluated at those same points. Also
                re-measures ``omega`` against ``theta_deriv`` on the real background, which the
                board assigns here (``[10-residual-spline-end-condition]``).

``cycles``      the self-consistency of the stored (div, mod) pair at the production |theta|
                (found here: a one-cycle inconsistency in ``WKB_mod_2pi``).

``margins``     the residual table's cut-to-anchor margin across the production k range, both
                models, both sectors (``[14-residual-range-top-margin]``).

``throughput``  the consumer's per-call cost in bulk on ``QCD_Cosmology``, the Levin-side
                measurement ``[01-offgrid-accessor-cost-on-qcd]`` names as its closing condition.

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py
    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py --section consumers

Timings on this machine are unreliable as elapsed durations (campaign board §5 note 14), so every
cost figure below is a **best-of-N minimum** and is labelled as such; the integrand-evaluation
counts are the reproducible measure.
"""

import argparse
import json
import sys
import time
from math import fabs, log, log1p, pi, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ComputeTargets.WKB_Gk import Gk_omegaEff_sq  # noqa: E402
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq  # noqa: E402
from ComputeTargets.phase_residual import (  # noqa: E402
    RESIDUAL_WKB_REGION_MARGIN,
    cached_phase_residual,
    clear_phase_residual_cache,
    residual_node_range,
)
from ComputeTargets.primitive_phase import (
    PrimitivePhase,
    build_phi_samples,
)  # noqa: E402
from ComputeTargets.TkSourceFunctions import TkSourceFunctions  # noqa: E402
from ComputeTargets.tests.test_gk_wkb_phase import (  # noqa: E402
    _run_Gk,
    _run_Tk,
    _unwrapped,
    lambdacdm_model_with_tables,
    qcd_model_with_tables,
)
from ComputeTargets.tests.test_tk_wkb_phase import build_and_store  # noqa: E402
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    PRODUCTION_Z_END,
    horizon_exit_z,
    load_references,
    phase_error,
    production_source_grid,
    to_redshift_array,
)
from LiouvilleGreen.constants import TWO_PI  # noqa: E402

# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------

K_VALUES = (1.0e5, 1.0e7, 3.0e8)
K_KEYS = tuple(f"{k:.6e}" for k in K_VALUES)
MODEL_KEYS = ("LambdaCDMModel", "QCDModel")

# the production k-grid of main.py, for the section that sweeps it
PRODUCTION_K_COUNT = 50

# prompt 13 §1 item 2: the consumer is scored at this many points per production interval
CONSUMER_POINTS_PER_INTERVAL = 10

# best-of-N for every wall-clock figure
TIMING_REPEATS = 5

# review §4 (theta_G) and §12.3 (theta_T): the "before" column. LambdaCDM, production
# tolerances, the two-stage phase ODE this campaign retired. Keyed (k_key, z) -> error in rad.
REVIEW_BEFORE_G = {
    "1.000000e+05": [
        (1.75e9, 6.8e-9),
        (1.24e7, 7.9e-7),
        (1.68e4, 1.6e-4),
        (3.213e3, -2.5e-3),
        (2.17e1, -0.17),
        (0.1, 13.9),
    ],
    "3.000000e+08": [
        (7.8e8, 1.5e-4),
        (1.05e6, 1.8e-4),
        (1.16e5, -0.35),
        (1.416e3, -70.0),
        (1.63e1, 292.0),
        (0.1, 7366.0),
    ],
}
REVIEW_BEFORE_T = {
    "1.000000e+05": [
        (3.8e8, 2.3e-8),
        (5.5e4, 1.9e-4),
        (2.68e2, 1.1e-2),
        (6.8, -4.4e-2),
        (0.32, 2.17),
        (0.1, 2.01),
    ],
    "3.000000e+08": [
        (3.0e6, 1.7e-4),
        (2.6e5, 1.5e-5),
        (2.2e4, 0.42),
        (1.69e2, -105.0),
        (1.38e1, -193.0),
        (0.1, 5.1e3),
    ],
}

# --------------------------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------------------------


class Fixtures:
    """The two production background models with their three tables, built once."""

    references = None
    grid = None
    z_nodes = None
    _models: Dict[str, object] = {}

    @classmethod
    def build(cls):
        if cls.references is None:
            cls.references = load_references()
            cls.grid = production_source_grid(
                cls.references["models"]["LambdaCDMModel"]["grid"]["z_init"]
            )
            cls.z_nodes = np.array(cls.grid.as_float_list(), dtype=float)

    @classmethod
    def model(cls, model_key: str):
        cls.build()
        if model_key not in cls._models:
            t0 = time.perf_counter()
            if model_key == "LambdaCDMModel":
                cls._models[model_key] = lambdacdm_model_with_tables(cls.z_nodes)
            elif model_key == "QCDModel":
                cls._models[model_key] = qcd_model_with_tables(cls.grid)
            else:
                raise ValueError(f"unknown model key {model_key!r}")
            print(
                f"   [built {model_key} with its tau / cs_tau / friction_F tables in "
                f"{time.perf_counter() - t0:.2f} s]"
            )
        return cls._models[model_key]

    @classmethod
    def block(cls, model_key: str) -> dict:
        cls.build()
        return cls.references["models"][model_key]


def reference_theta_G(model_key: str, k_key: str) -> List[Tuple[float, float]]:
    """``[(z, theta_G_ref)]`` at the JSON checkpoints below that k's residual anchor."""
    block = Fixtures.block(model_key)
    k = float(k_key)
    anchor_tau = block["primitives_at_rho_anchor"][k_key]["tau_minus_top"]
    position = {cp["index"]: i for i, cp in enumerate(block["checkpoints"])}
    return [
        (
            float(e["z"]),
            -k * (block["tau_minus_top"][position[e["index"]]] - anchor_tau)
            - float(e["value"]),
        )
        for e in block["rho_G"][k_key]
    ]


def reference_theta_T(
    model_key: str, k_key: str
) -> List[Tuple[float, float, float, float]]:
    """``[(z, theta_T_ref, delta_F_ref, rho_T_ref)]`` at the same checkpoints."""
    block = Fixtures.block(model_key)
    k = float(k_key)
    anchor = block["primitives_at_rho_anchor"][k_key]
    position = {cp["index"]: i for i, cp in enumerate(block["checkpoints"])}
    out = []
    for e in block["rho_T"][k_key]:
        i = position[e["index"]]
        rho = float(e["value"])
        theta = -k * (block["cs_tau_minus_top"][i] - anchor["cs_tau_minus_top"]) - rho
        dF = block["friction_F_minus_top"][i] - anchor["friction_F_minus_top"]
        out.append((float(e["z"]), theta, dF, rho))
    return out


def rho_anchor_z(model_key: str, k_key: str) -> float:
    return float(Fixtures.block(model_key)["rho_anchor_z"][k_key])


def best_of(fn, repeats: int = TIMING_REPEATS):
    """``(result, best wall time)`` over ``repeats`` calls -- a sleep can inflate a sample but
    never shrink it, so the minimum is the defensible figure (board §5 note 14)."""
    best = None
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return out, best


# --------------------------------------------------------------------------------------------
# section: producers (prompt 13 §1 item 1)
# --------------------------------------------------------------------------------------------


def section_producers() -> dict:
    print("\n" + "=" * 94)
    print("1. PRODUCERS -- review §4 (theta_G) and §12.3 (theta_T) re-measured")
    print("=" * 94)

    results: dict = {"Gk": {}, "Tk": {}, "cost": {}}

    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        units = model.cosmology.units
        for k, k_key in zip(K_VALUES, K_KEYS):
            z_anchor = rho_anchor_z(model_key, k_key)

            # ---- Gk -------------------------------------------------------------------
            ref = reference_theta_G(model_key, k_key)
            samples = to_redshift_array([z for z, _ in ref])
            payload = _run_Gk(model, units, k, z_anchor, samples)
            theta = dict(zip(samples.as_float_list(), _unwrapped(payload)))
            rows = []
            worst, worst_z, span = 0.0, None, 0.0
            for z, theta_ref in ref:
                err = phase_error(theta[z], theta_ref)
                span = max(span, fabs(theta_ref))
                rows.append((z, theta_ref, err))
                if err > worst:
                    worst, worst_z = err, z
            results["Gk"][(model_key, k_key)] = {
                "rows": rows,
                "worst": worst,
                "worst_z": worst_z,
                "span": span,
                "eps_k_tau": float(np.finfo(float).eps * span),
                "anchor": z_anchor,
                "metadata": payload["metadata"],
            }

            # ---- Tk -------------------------------------------------------------------
            refT = reference_theta_T(model_key, k_key)
            samplesT = to_redshift_array([z for z, _, _, _ in refT])
            payloadT = _run_Tk(model, units, k, z_anchor, samplesT)
            thetaT = dict(zip(samplesT.as_float_list(), _unwrapped(payloadT)))
            frictionT = dict(zip(samplesT.as_float_list(), payloadT["friction_sample"]))
            rowsT = []
            worstT, worstT_z, spanT = 0.0, None, 0.0
            worstF, worstF_z = 0.0, None
            for z, theta_ref, dF_ref, _ in refT:
                err = phase_error(thetaT[z], theta_ref)
                spanT = max(spanT, fabs(theta_ref))
                dF = frictionT[z]
                ferr = fabs(dF - dF_ref) / max(1.0e-300, fabs(dF_ref))
                rowsT.append((z, theta_ref, err, ferr))
                if err > worstT:
                    worstT, worstT_z = err, z
                if ferr > worstF:
                    worstF, worstF_z = ferr, z
            results["Tk"][(model_key, k_key)] = {
                "rows": rowsT,
                "worst": worstT,
                "worst_z": worstT_z,
                "span": spanT,
                "eps_k_tau": float(np.finfo(float).eps * spanT),
                "worst_friction_rel": worstF,
                "worst_friction_z": worstF_z,
                "anchor": z_anchor,
                "metadata": payloadT["metadata"],
            }

    # ---- the tables -----------------------------------------------------------------------
    for sector, before in (("Gk", REVIEW_BEFORE_G), ("Tk", REVIEW_BEFORE_T)):
        label = "theta_G (review §4)" if sector == "Gk" else "theta_T (review §12.3)"
        for model_key in MODEL_KEYS:
            for k_key in K_KEYS:
                r = results[sector][(model_key, k_key)]
                print(
                    f"\n-- {label} | {model_key} | k = {float(k_key):.3e}/Mpc | "
                    f"anchor z = {r['anchor']:.6g} | span {r['span']:.4e} rad"
                )
                print(
                    f"   {'z':>12} {'theta_ref [rad]':>17} {'after [rad]':>13} "
                    f"{'before (review) [rad]':>22}"
                )
                before_rows = (
                    before.get(k_key, []) if model_key == "LambdaCDMModel" else []
                )
                for i, row in enumerate(r["rows"]):
                    z, theta_ref, err = row[0], row[1], row[2]
                    b = ""
                    # the review's z rows are not the JSON checkpoints except at z = 0.1;
                    # quote its own row only where the redshift matches to 1 %
                    for bz, berr in before_rows:
                        if fabs(bz - z) <= 0.01 * max(bz, z):
                            b = f"{berr:+.3g} (at z={bz:.3g})"
                            break
                    print(f"   {z:12.5g} {theta_ref:17.6e} {err:13.4e} {b:>22}")
                print(
                    f"   max {r['worst']:.4e} rad at z = {r['worst_z']:.6g}; "
                    f"eps*|theta|_max floor {r['eps_k_tau']:.3e} rad"
                    + (
                        f"; max |dF|/|dF_ref| {r['worst_friction_rel']:.3e} at "
                        f"z = {r['worst_friction_z']:.6g}"
                        if sector == "Tk"
                        else ""
                    )
                )

    # ---- cost per object ------------------------------------------------------------------
    print(
        "\n-- cost per object (best of %d; integrand evaluations are the reproducible measure)"
        % TIMING_REPEATS
    )
    print(
        f"   {'model':>14} {'sector':>6} {'k [1/Mpc]':>11} {'samples':>8} "
        f"{'build s':>9} {'cached s':>9} {'build evals':>12} {'cached evals':>13}"
    )
    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        units = model.cosmology.units
        k = 3.0e8
        k_key = f"{k:.6e}"
        z_anchor = rho_anchor_z(model_key, k_key)

        # the Gk production geometry: the response grid below the anchor, 12x sparser
        nodes = Fixtures.z_nodes
        below = nodes[nodes <= z_anchor]
        response = below[::-1][::12][::-1]
        gk_samples = to_redshift_array([float(z) for z in response])
        # the Tk production geometry: the whole source grid below the anchor
        tk_samples = to_redshift_array([float(z) for z in below])

        for sector, samples, runner in (
            ("Gk", gk_samples, _run_Gk),
            ("Tk", tk_samples, _run_Tk),
        ):
            # cold: the residual table is built by this call
            clear_phase_residual_cache()
            t0 = time.perf_counter()
            cold_payload = runner(model, units, k, z_anchor, samples)
            cold_time = time.perf_counter() - t0
            cold_evals = cold_payload["stage_1_data"].RHS_evaluations

            # warm: every later object of the same (model, k, sector) reuses the table
            warm_payload, warm_time = best_of(
                lambda: runner(model, units, k, z_anchor, samples)
            )
            warm_evals = warm_payload["stage_1_data"].RHS_evaluations

            results["cost"][(model_key, sector)] = {
                "k": k,
                "samples": len(samples),
                "cold_time": cold_time,
                "warm_time": warm_time,
                "cold_evals": cold_evals,
                "warm_evals": warm_evals,
                "metadata": warm_payload["metadata"],
            }
            print(
                f"   {model_key:>14} {sector:>6} {k:11.3e} {len(samples):8d} "
                f"{cold_time:9.4f} {warm_time:9.4f} {cold_evals:12d} {warm_evals:13d}"
            )

    return results


# --------------------------------------------------------------------------------------------
# section: stored F and rho_T (prompt 13 §1 item 3)
# --------------------------------------------------------------------------------------------


def section_stored() -> dict:
    print("\n" + "=" * 94)
    print("2. F AND rho_T AS STORED, against prompt 01's references")
    print("=" * 94)

    out = {}
    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        units = model.cosmology.units
        for k, k_key in zip(K_VALUES, K_KEYS):
            z_anchor = rho_anchor_z(model_key, k_key)
            ref = reference_theta_T(model_key, k_key)
            z_list = [z for z, _, _, _ in ref]
            samples = to_redshift_array(z_list)

            obj = build_and_store(
                model,
                k,
                z_anchor,
                samples,
                T_init=1.0e-3,
                Tprime_init=0.0,
                units=units,
            )
            stored = {v.z.z: v for v in obj.values}

            # the constant initial-data offset delta that store() adds to every sample of one
            # object, recovered at the highest sample; it is a property of (B, delta), not of
            # the phase, and must be the same at every sample -- which is checked below
            z_top = max(z_list)
            delta = (
                stored[z_top].theta_div_2pi * TWO_PI + stored[z_top].theta_mod_2pi
            ) - tk_phase_from_tables(model, k, z_anchor, z_top)

            worst_F, worst_F_z = 0.0, None
            worst_rho, worst_rho_z = 0.0, None
            rho_at_end = None
            for z, theta_ref, dF_ref, rho_ref in ref:
                v = stored[z]
                ferr = fabs(v.friction - dF_ref) / max(1.0e-300, fabs(dF_ref))
                if ferr > worst_F:
                    worst_F, worst_F_z = ferr, z
                # rho_T as stored: back it out of the stored phase by removing delta and the
                # leading sound-horizon term, and score it against the JSON reference
                theta_stored = v.theta_div_2pi * TWO_PI + v.theta_mod_2pi - delta
                rho_stored = -theta_stored - k * model.functions.cs_tau.delta(
                    z_anchor, z
                )
                err = fabs(rho_stored - rho_ref)
                if err > worst_rho:
                    worst_rho, worst_rho_z = err, z
                rho_at_end = rho_ref

            out[(model_key, k_key)] = {
                "worst_F_rel": worst_F,
                "worst_F_z": worst_F_z,
                "worst_rho_abs": worst_rho,
                "worst_rho_z": worst_rho_z,
                "rho_at_z_end": rho_at_end,
                "delta": float(delta),
                "sin_coeff": obj.sin_coeff,
                "cos_coeff": obj.cos_coeff,
            }
            print(
                f"   {model_key:>14} k = {k:9.3e}: max |F_stored/F_ref - 1| {worst_F:.3e} at "
                f"z = {worst_F_z:.5g}; max |rho_T - ref| {worst_rho:.3e} rad at "
                f"z = {worst_rho_z:.5g}; rho_T(z=0.1) = {rho_at_end:+.6f} rad; "
                f"sin_coeff = {obj.sin_coeff:.6e} > 0, cos_coeff = {obj.cos_coeff:.1e}"
            )

    return out


# --------------------------------------------------------------------------------------------
# section: consumers (prompt 13 §1 item 2)
# --------------------------------------------------------------------------------------------


def _midpoint_grid(z_nodes: np.ndarray, points_per_interval: int) -> np.ndarray:
    """``points_per_interval`` interior points in ``u = log(1+z)`` per production interval,
    descending in z, excluding the nodes themselves."""
    u = np.log1p(np.asarray(z_nodes, dtype=float))  # descending in z => descending in u
    out = []
    for i in range(len(u) - 1):
        a, b = u[i], u[i + 1]
        for j in range(1, points_per_interval + 1):
            out.append(a + (b - a) * j / (points_per_interval + 1.0))
    return np.expm1(np.array(out, dtype=float))


def section_consumers() -> dict:
    print("\n" + "=" * 94)
    print("3. CONSUMERS AT PRODUCTION x -- PrimitivePhase and TkSourceFunctions.phase")
    print("=" * 94)

    out = {"Gk": {}, "Tk": {}, "deriv": {}}

    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        units = model.cosmology.units
        cosmology = model.cosmology
        nodes = Fixtures.z_nodes

        for k in K_VALUES:
            k_key = f"{k:.6e}"

            # =========================== the Green's-function consumer =====================
            # production geometry (main.py:1558, :1588): sources below
            # sqrt(z_e3 z_e4) are pure-WKB objects (G_init = 0, Gprime_init = 1, z_init =
            # z_source, so delta = atan2(0, +) = 0 exactly), and the response redshift is
            # held fixed at the production --zend.
            z_e3 = horizon_exit_z(cosmology, k, 3.0)
            z_e4 = horizon_exit_z(cosmology, k, 4.0)
            z_source_limit = sqrt(z_e3 * z_e4)
            z_r = PRODUCTION_Z_END

            band = nodes[(nodes < z_source_limit) & (nodes > z_r)]
            if len(band) < 8:
                continue
            # the stored phase of a pure-WKB object at the fixed response redshift, from the
            # production producer: theta(z_r; z_source) for every source of the band
            theta_stored = []
            for z_s in band:
                p = _run_Gk(model, units, k, float(z_s), to_redshift_array([z_r]))
                theta_stored.append(
                    p["theta_div_2pi_sample"][0] * TWO_PI + p["theta_mod_2pi_sample"][0]
                )
            z_points = [float(z) for z in band]
            phi_points = build_phi_samples(
                k, model.functions.tau, z_r, z_points, theta_stored, sign=-1
            )
            phase = PrimitivePhase(
                k,
                model.functions.tau,
                z_r,
                z_points,
                phi_points,
                sign=-1,
                model_functions=model.functions,
                label="G_k WKB phase (verification)",
            )

            # the reference between the samples: the same producer, evaluated there. Its
            # leading term is the double-double table's exact interval and its residual the
            # order-4 table's; what the consumer approximates, and this does not, is phi.
            mid = _midpoint_grid(np.array(z_points), CONSUMER_POINTS_PER_INTERVAL)
            mid = mid[(mid > min(z_points)) & (mid < max(z_points))]
            ref_payload = _run_Gk(
                model, units, k, z_r, to_redshift_array(sorted(mid, reverse=True))
            )
            # the producer anchored at z_r returns theta(z; z_r) = -theta(z_r; z)
            ref_theta = {
                z: -t
                for z, t in zip(sorted(mid, reverse=True), _unwrapped(ref_payload))
            }

            # a sample whose phi is a whole cycle off its neighbours is not an interpolation
            # error but a defect of the stored (div, mod) pair -- see the "cycles" section --
            # so it is counted, reported and then excluded from the interpolation figure
            phi_arr = np.asarray(phi_points, dtype=float)
            outliers = np.where(np.abs(phi_arr - np.median(phi_arr)) > pi)[0]
            outlier_z = [float(z_points[i]) for i in outliers]
            # a cubic spline's response to one bad ordinate decays as (sqrt(3)-2)^n per knot,
            # so 15 production intervals (a factor 10^0.15) either side puts it below 1e-5
            # of the 2pi it carries
            excluded_lo = min(outlier_z) / 1.4126 if outlier_z else None
            excluded_hi = max(outlier_z) * 1.4126 if outlier_z else None

            worst, worst_z, span = -1.0, None, 0.0
            clean, clean_z = -1.0, None
            for z, t_ref in ref_theta.items():
                err = phase_error(phase.raw_theta(z), t_ref)
                span = max(span, fabs(t_ref))
                if err > worst:
                    worst, worst_z = err, z
                if outlier_z and excluded_lo <= z <= excluded_hi:
                    continue
                if err > clean:
                    clean, clean_z = err, z
            at_samples = max(
                phase_error(phase.raw_theta(z), t)
                for z, t in zip(z_points, theta_stored)
            )
            # x = k(1+z)/H grows as z falls, so the largest x of this band is at its foot
            z_deepest = min(z_points)
            x_max = k * (1.0 + z_deepest) / cosmology.Hubble(z_deepest)
            out["Gk"][(model_key, k_key)] = {
                "worst": worst,
                "worst_z": worst_z,
                "span": span,
                "at_samples": at_samples,
                "n_samples": len(z_points),
                "n_scored": len(ref_theta),
                "z_lo": min(z_points),
                "z_hi": max(z_points),
                "x_source_max": x_max,
                "eps_k_tau": float(np.finfo(float).eps * span),
                "ulp": float(np.spacing(span)),
                "phi_range": (float(min(phi_points)), float(max(phi_points))),
                "cycle_outliers": len(outliers),
                "cycle_outlier_z": outlier_z,
                "clean": clean,
                "clean_z": clean_z,
            }
            print(
                f"\n   [Gk consumer] {model_key} k = {k:.3e}: {len(z_points)} sources in "
                f"[{min(z_points):.4g}, {max(z_points):.4g}], z_r = {z_r}, "
                f"x_source up to {x_max:.3e}"
            )
            print(
                f"      max |theta - ref| over {len(ref_theta)} interior points: "
                f"{worst:.4e} rad at z = {worst_z:.6g}  (at the samples {at_samples:.3e}; "
                f"span {span:.4e} rad = {worst/max(np.spacing(span),1e-300):.2f} ulp; "
                f"eps*k*tau {np.finfo(float).eps*span:.3e}); "
                f"phi in [{min(phi_points):.4g}, {max(phi_points):.4g}] rad"
            )
            if outliers.size:
                print(
                    f"      !! {len(outliers)} of {len(z_points)} stored samples carry a "
                    f"whole-cycle phi outlier (z = {['%.6g' % z for z in outlier_z]}); "
                    f"excluding 15 grid intervals either side the maximum is {clean:.4e} "
                    f"rad at z = {clean_z:.6g}"
                )

            # =========================== the transfer-function consumer ====================
            # TkSourceFunctions builds its PrimitivePhase from a TkWKBIntegration anchored at
            # the hand-over; the production hand-over is the numeric stop point, which we take
            # at the 3-e-fold anchor of prompt 01's references (the same anchor the producer
            # section uses).
            z_init = rho_anchor_z(model_key, k_key)
            wkb_nodes = nodes[nodes <= z_init]
            if len(wkb_nodes) < 8:
                continue
            wkb_samples = to_redshift_array([float(z) for z in wkb_nodes])
            tk_obj = build_and_store(
                model,
                k,
                z_init,
                wkb_samples,
                T_init=1.0e-3,
                Tprime_init=0.0,
                units=units,
            )
            functions = TkSourceFunctions(
                model,
                _Wavenumber(k),
                _FakeTkNumeric(z_init, model),
                tk_obj,
            )
            tk_phase = functions.phase

            mid_t = _midpoint_grid(wkb_nodes, CONSUMER_POINTS_PER_INTERVAL)
            mid_t = mid_t[
                (mid_t > float(wkb_nodes[-1])) & (mid_t < float(wkb_nodes[0]))
            ]
            ref_t_payload = _run_Tk(
                model, units, k, z_init, to_redshift_array(sorted(mid_t, reverse=True))
            )
            ref_t = dict(zip(sorted(mid_t, reverse=True), _unwrapped(ref_t_payload)))
            # the consumer's phase carries the same constant delta as the stored samples;
            # recover it at the sample nearest the anchor, where the phase is smallest
            stored_first = max(tk_obj.values, key=lambda v: v.z.z)
            delta = (
                stored_first.theta_div_2pi * TWO_PI + stored_first.theta_mod_2pi
            ) - tk_phase_from_tables(model, k, z_init, stored_first.z.z)

            worst_t, worst_t_z, span_t = -1.0, None, 0.0
            for z, t_ref in ref_t.items():
                err = phase_error(tk_phase.raw_theta(z), t_ref + delta)
                span_t = max(span_t, fabs(t_ref))
                if err > worst_t:
                    worst_t, worst_t_z = err, z
            x_T = (
                k
                * sqrt(cosmology.wPerturbations(PRODUCTION_Z_END))
                * (1.0 + PRODUCTION_Z_END)
                / cosmology.Hubble(PRODUCTION_Z_END)
            )
            out["Tk"][(model_key, k_key)] = {
                "worst": worst_t,
                "worst_z": worst_t_z,
                "span": span_t,
                "n_samples": len(wkb_nodes),
                "n_scored": len(ref_t),
                "x_T_at_zend": x_T,
                "eps_k_tau": float(np.finfo(float).eps * span_t),
                "ulp": float(np.spacing(span_t)),
            }
            print(
                f"   [Tk consumer] {model_key} k = {k:.3e}: {len(wkb_nodes)} samples from "
                f"z_init = {z_init:.5g} to {PRODUCTION_Z_END}, x_T(z=0.1) = {x_T:.3e}"
            )
            print(
                f"      max |theta_T - ref| over {len(ref_t)} interior points: "
                f"{worst_t:.4e} rad at z = {worst_t_z:.6g}  (span {span_t:.4e} rad = "
                f"{worst_t/max(np.spacing(span_t),1e-300):.2f} ulp; "
                f"eps*k*cs_tau {np.finfo(float).eps*span_t:.3e})"
            )

            # ------- [10-residual-spline-end-condition] on the real background -------------
            out["deriv"][(model_key, k_key)] = _measure_theta_deriv(
                model, k, phase, z_points, tk_phase, wkb_nodes
            )

    return out


def tk_phase_from_tables(model, k: float, z_init: float, z: float) -> float:
    """``theta_T(z; z_init)`` straight from the tables, for recovering the stored ``delta``."""
    rho, _ = cached_phase_residual(
        model, float(k), model.functions.cs_tau.table.z_nodes, "Tk", store_id=None
    )
    return -(float(k) * model.functions.cs_tau.delta(z_init, z) + rho.delta(z_init, z))


def _measure_theta_deriv(model, k, gk_phase, gk_z, tk_phase, tk_nodes) -> dict:
    """``omega(z)`` against ``phase.theta_deriv(z)`` on the real background, for both sectors:
    the identity ``[10-residual-spline-end-condition]`` is open on, which the board assigns to
    prompt 13. Reports the window maxima *and* the error at the second and third samples in
    from each end, which is what the issue asks for."""
    out = {}

    def score(phase, z_nodes, omega_sq, sign):
        z_sorted = np.sort(np.asarray(z_nodes, dtype=float))  # ascending
        errs = []
        for z in z_sorted:
            omega = sqrt(fabs(omega_sq(model, float(k), float(z))))
            deriv = fabs(phase.theta_deriv(float(z)))
            errs.append(fabs(deriv - omega) / omega)
        errs = np.array(errs)
        n = len(errs)
        return {
            "max_all": float(errs.max()),
            "max_at_z": float(z_sorted[int(errs.argmax())]),
            "max_3_3": float(errs[3 : n - 3].max()) if n > 6 else float("nan"),
            "max_5_3": float(errs[5 : n - 3].max()) if n > 8 else float("nan"),
            "interior": float(errs[10 : n - 10].max()) if n > 20 else float("nan"),
            "ends_low": [float(e) for e in errs[:5]],
            "ends_high": [float(e) for e in errs[-5:]],
            # how far in from the top sample the end condition has decayed away
            "trim_top": {
                m: float(errs[3 : n - m].max()) for m in (3, 5, 8, 12, 20) if n > m + 4
            },
        }

    out["Gk"] = score(gk_phase, gk_z, Gk_omegaEff_sq, -1)
    out["Tk"] = score(tk_phase, tk_nodes, Tk_omegaEff_sq, +1)
    for sector in ("Gk", "Tk"):
        r = out[sector]
        print(
            f"      [theta_deriv vs omega, {sector}] max over all samples "
            f"{r['max_all']:.4e} at z = {r['max_at_z']:.5g}; [3:-3] {r['max_3_3']:.4e}; "
            f"[5:-3] {r['max_5_3']:.4e}; deep interior {r['interior']:.4e}"
        )
        print(
            f"         first five (low z) {['%.2e' % e for e in r['ends_low']]}; "
            f"last five (high z) {['%.2e' % e for e in r['ends_high']]}"
        )
        print(
            "         max over [3:-m]: "
            + ", ".join(f"m={m}: {v:.3e}" for m, v in sorted(r["trim_top"].items()))
        )
    return out


class _Wavenumber:
    """Enough of a ``wavenumber`` for ``TkSourceFunctions``: ``float(k)`` and no ``z_exit``
    (so the crossover cross-check falls back to the numeric stand-in's)."""

    def __init__(self, k: float):
        self._k = float(k)

    def __float__(self):
        return self._k


class _FakeNumericSample:
    def __init__(self, z: float, T: float, Tprime: float):
        self.z = type("Z", (), {"z": float(z)})()
        self.T = float(T)
        self.Tprime = float(Tprime)


class _FakeTkNumeric:
    """A ``TkNumericIntegration`` stand-in carrying the five samples at or above the hand-over
    that ``TkSourceFunctions._build_numeric`` requires. Only the WKB half is measured here.
    """

    def __init__(self, z_init: float, model):
        zs = [z_init * (1.0 + 0.02 * j) for j in range(8)][::-1]
        self.values = [_FakeNumericSample(z, 1.0e-3, 0.0) for z in zs]
        self.stop_deltaz_subh = None
        self.z_exit = None


# --------------------------------------------------------------------------------------------
# section: residual-table margins ([14-residual-range-top-margin])
# --------------------------------------------------------------------------------------------


def section_margins() -> dict:
    print("\n" + "=" * 94)
    print("4. RESIDUAL-TABLE CUT-TO-ANCHOR MARGIN across the production k range")
    print("   ([14-residual-range-top-margin]; the board assigns this to prompt 13)")
    print("=" * 94)

    k_grid = np.geomspace(1.0e5, 3.0e8, PRODUCTION_K_COUNT)
    out = {}
    print(
        f"   {'model':>14} {'sector':>6} {'worst cut/anchor':>17} {'e-folds':>9} "
        f"{'at k [1/Mpc]':>13} {'cut z':>12} {'anchor z':>12} {'min nodes':>10}"
    )
    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        cosmology = model.cosmology
        for sector in ("Gk", "Tk"):
            leading = model.functions.tau if sector == "Gk" else model.functions.cs_tau
            worst = None
            min_nodes = None
            for k in k_grid:
                nodes = residual_node_range(
                    model, float(k), leading.table.z_nodes, sector
                )
                cut = float(nodes[0])
                anchor = horizon_exit_z(cosmology, float(k), 3.0)
                ratio = cut / anchor
                efolds = log(ratio)
                if worst is None or ratio < worst[0]:
                    worst = (ratio, efolds, float(k), cut, anchor)
                min_nodes = (
                    len(nodes) if min_nodes is None else min(min_nodes, len(nodes))
                )
            out[(model_key, sector)] = {
                "worst_ratio": worst[0],
                "worst_efolds": worst[1],
                "at_k": worst[2],
                "cut_z": worst[3],
                "anchor_z": worst[4],
                "min_nodes": min_nodes,
            }
            print(
                f"   {model_key:>14} {sector:>6} {worst[0]:17.4g} {worst[1]:9.3f} "
                f"{worst[2]:13.4e} {worst[3]:12.4e} {worst[4]:12.4e} {min_nodes:10d}"
            )
    print(
        f"   (RESIDUAL_WKB_REGION_MARGIN = {RESIDUAL_WKB_REGION_MARGIN}; a margin below ~1 "
        f"e-fold is what would make the constant worth revisiting)"
    )
    return out


# --------------------------------------------------------------------------------------------
# section: the primitives themselves (README §6 rows 1-5)
# --------------------------------------------------------------------------------------------


def section_primitives() -> dict:
    """The four table rows of README §6 -- tau and Delta-tau at the production nodes, tau_s and
    F at the nodes, and rho against the references -- re-measured on the production path, plus
    the interval accessor's cost per call."""
    print("\n" + "=" * 94)
    print(
        "0. THE PRIMITIVES -- tau, Delta-tau, cs_tau, F and rho at the production nodes"
    )
    print("=" * 94)

    out = {}
    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        block = Fixtures.block(model_key)
        z_top = float(block["z_top"])
        checkpoints = block["checkpoints"]
        tau = model.functions.tau
        cs_tau = model.functions.cs_tau
        friction_F = model.functions.friction_F

        worst = {"tau": 0.0, "cs_tau": 0.0, "friction_F": 0.0}
        worst_z = {"tau": None, "cs_tau": None, "friction_F": None}
        for i, cp in enumerate(checkpoints):
            z = float(cp["z"])
            if z == z_top:
                continue
            for name, accessor, key in (
                ("tau", tau, "tau_minus_top"),
                ("cs_tau", cs_tau, "cs_tau_minus_top"),
                ("friction_F", friction_F, "friction_F_minus_top"),
            ):
                ref = float(block[key][i])
                got = accessor.delta(z_top, z)
                rel = fabs(got - ref) / max(1.0e-300, fabs(ref))
                if rel > worst[name]:
                    worst[name], worst_z[name] = rel, z

        # Delta-tau over one production interval, relative to the interval itself
        nodes = Fixtures.z_nodes
        worst_interval, worst_interval_z = 0.0, None
        for rec in block["short_baseline"]:
            hi, lo = float(rec["z_node_hi"]), float(rec["z_node_lo"])
            got = tau.delta(hi, lo)
            ref = float(rec["delta_tau_full"])
            rel = fabs(got - ref) / fabs(ref)
            if rel > worst_interval:
                worst_interval, worst_interval_z = rel, hi
        worst_frac, worst_frac_z = 0.0, None
        for rec in block["short_baseline"]:
            hi = float(rec["z_node_hi"])
            zf = float(rec["z_fraction"])
            got = tau.delta(hi, zf)
            ref = float(rec["delta_tau_fraction"])
            rel = fabs(got - ref) / fabs(ref)
            if rel > worst_frac:
                worst_frac, worst_frac_z = rel, hi

        # rho against the references, absolute, over all three k and both sectors
        worst_rho, worst_rho_where = 0.0, None
        for k, k_key in zip(K_VALUES, K_KEYS):
            anchor = rho_anchor_z(model_key, k_key)
            for sector, key in (("Gk", "rho_G"), ("Tk", "rho_T")):
                leading = tau if sector == "Gk" else cs_tau
                # the production construction: one table per (model, k, sector) on the
                # background grid, reached at the anchor through delta's off-grid partial
                rho, _ = cached_phase_residual(
                    model, k, leading.table.z_nodes, sector, store_id=None
                )
                for e in block[key][k_key]:
                    got = rho.delta(anchor, float(e["z"]))
                    err = fabs(got - float(e["value"]))
                    if err > worst_rho:
                        worst_rho, worst_rho_where = err, (sector, k, float(e["z"]))

        # accessor cost
        u = np.log1p(nodes)
        on_grid_pairs = [
            (float(nodes[i]), float(nodes[i + 40])) for i in range(0, 800, 4)
        ]
        one_off = [
            (float(nodes[i]), float(np.expm1(0.5 * (u[i + 40] + u[i + 41]))))
            for i in range(0, 800, 4)
        ]
        both_off = [
            (
                float(np.expm1(0.5 * (u[i] + u[i + 1]))),
                float(np.expm1(0.37 * u[i + 40] + 0.63 * u[i + 41])),
            )
            for i in range(0, 800, 4)
        ]
        costs = {}
        for label, pairs in (
            ("on-grid", on_grid_pairs),
            ("one off-grid", one_off),
            ("both off-grid", both_off),
        ):
            _, dt = best_of(lambda: [tau.delta(a, b) for a, b in pairs], repeats=5)
            costs[label] = 1.0e6 * dt / len(pairs)

        out[model_key] = {
            "tau_rel": worst["tau"],
            "tau_at_z": worst_z["tau"],
            "cs_tau_rel": worst["cs_tau"],
            "cs_tau_at_z": worst_z["cs_tau"],
            "friction_F_rel": worst["friction_F"],
            "friction_F_at_z": worst_z["friction_F"],
            "interval_rel": worst_interval,
            "interval_at_z": worst_interval_z,
            "fraction_rel": worst_frac,
            "fraction_at_z": worst_frac_z,
            "rho_abs": worst_rho,
            "rho_where": worst_rho_where,
            "accessor_us": costs,
        }
        print(
            f"   {model_key}: tau {worst['tau']:.3e} rel at z = {worst_z['tau']:.5g}; "
            f"cs_tau {worst['cs_tau']:.3e} at z = {worst_z['cs_tau']:.5g}; "
            f"F {worst['friction_F']:.3e} at z = {worst_z['friction_F']:.5g}"
        )
        print(
            f"      one-interval Delta-tau {worst_interval:.3e} rel (z = "
            f"{worst_interval_z:.5g}); 37 % fraction {worst_frac:.3e} rel "
            f"(z = {worst_frac_z:.5g})"
        )
        print(f"      rho worst {worst_rho:.3e} rad absolute at {worst_rho_where}")
        print(
            "      tau.delta cost per call (best of 5): "
            + ", ".join(f"{k} {v:.2f} us" for k, v in costs.items())
        )
    return out


# --------------------------------------------------------------------------------------------
# section: the (div, mod) pair's one-cycle inconsistency at large |theta|
# --------------------------------------------------------------------------------------------


def section_cycles() -> dict:
    """
    ``LiouvilleGreen.WKBtools.WKB_mod_2pi`` forms the cycle count as
    ``int(floor(fabs(theta) / TWO_PI))`` -- a **rounded** division -- while the remainder is
    ``fmod(theta, TWO_PI)``, which is exact. When the true quotient sits within half an ulp
    below an integer the division rounds up across it, ``floor`` returns one cycle too many,
    and the pair no longer reconstructs its own phase: ``div * 2pi + mod == theta - 2pi``.

    The stored ``theta_mod_2pi`` -- what ``G_WKB`` and ``T_WKB`` are built from -- is unaffected,
    because ``fmod`` is exact. What is affected is every consumer that reconstructs the
    *unwrapped* phase from the pair, which since prompt 09/10 is both of them
    (``build_phi_samples``). This section measures the rate on the production geometry.
    """
    print("\n" + "=" * 94)
    print(
        "6. THE STORED (div, mod) PAIR AT LARGE |theta| -- WKB_mod_2pi self-consistency"
    )
    print("=" * 94)

    from LiouvilleGreen.WKBtools import WKB_mod_2pi

    Fixtures.build()
    out = {}
    k_grid = np.geomspace(1.0e5, 3.0e8, PRODUCTION_K_COUNT)
    nodes = Fixtures.z_nodes

    print(
        "   Production exposure. The Green's-function stage is one object per (k, z_source)"
        "\n   and every object holds the response grid below its source, so the sample the"
        "\n   consumer reconstructs is the whole (source x response) rectangle; the"
        "\n   transfer-function stage is one object per k over the source grid."
    )
    print(
        f"\n   {'model':>14} {'sector':>6} {'k [1/Mpc]':>11} {'samples':>10} "
        f"{'inconsistent':>13} {'rate':>10} {'max |theta|':>13}"
    )
    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        cosmology = model.cosmology
        for sector in ("Gk", "Tk"):
            leading = model.functions.tau if sector == "Gk" else model.functions.cs_tau
            for k in K_VALUES:
                rho, _ = cached_phase_residual(
                    model, float(k), leading.table.z_nodes, sector, store_id=None
                )
                rho_top = float(rho.z_nodes[0])
                z_e3 = horizon_exit_z(cosmology, float(k), 3.0)
                z_e4 = horizon_exit_z(cosmology, float(k), 4.0)
                total = bad = 0
                worst_theta = 0.0

                if sector == "Gk":
                    # the pure-WKB band of main.py:1588: z_source < sqrt(z_e3 z_e4), anchored
                    # at the source itself, sampled on the 12x-sparser response grid below it
                    response = nodes[::-1][::12][::-1]
                    sources = nodes[
                        (nodes < min(sqrt(z_e3 * z_e4), rho_top))
                        & (nodes > PRODUCTION_Z_END)
                    ]
                    for z_s in sources:
                        for z_r in response[response <= z_s]:
                            theta = -(
                                float(k) * leading.delta(float(z_s), float(z_r))
                                + rho.delta(float(z_s), float(z_r))
                            )
                            div, mod = WKB_mod_2pi(theta)
                            total += 1
                            worst_theta = max(worst_theta, fabs(theta))
                            if fabs(div * TWO_PI + mod - theta) > 1.0:
                                bad += 1
                else:
                    anchor = min(z_e3, rho_top)
                    for z in nodes[nodes <= anchor]:
                        theta = -(
                            float(k) * leading.delta(anchor, float(z))
                            + rho.delta(anchor, float(z))
                        )
                        div, mod = WKB_mod_2pi(theta)
                        total += 1
                        worst_theta = max(worst_theta, fabs(theta))
                        if fabs(div * TWO_PI + mod - theta) > 1.0:
                            bad += 1

                out[(model_key, sector, f"{k:.6e}")] = {
                    "samples": total,
                    "inconsistent": bad,
                    "rate": bad / max(1, total),
                    "max_theta": worst_theta,
                }
                print(
                    f"   {model_key:>14} {sector:>6} {k:11.3e} {total:10d} {bad:13d} "
                    f"{bad/max(1,total):10.3e} {worst_theta:13.4e}"
                )

    # the closed-form rate, so the production figures can be read as a sample of it
    rng = np.random.default_rng(20260913)
    for scale in (1.0e9, 1.0e11, 4.0e12):
        draws = -rng.uniform(0.5 * scale, scale, 400000)
        bad = sum(
            1
            for t in draws
            if fabs(
                (lambda dm: dm[0] * TWO_PI + dm[1])(WKB_mod_2pi(float(t))) - float(t)
            )
            > 1.0
        )
        print(
            f"   [uniform control] |theta| ~ {scale:.1e}: {bad} of {len(draws)} "
            f"({bad/len(draws):.3e}); half an ulp of |theta|/2pi is "
            f"{0.5*np.spacing(scale/TWO_PI):.3e} cycles"
        )
        out[("control", scale)] = {"bad": bad, "n": len(draws)}

    return out


# --------------------------------------------------------------------------------------------
# section: consumer throughput on QCD ([01-offgrid-accessor-cost-on-qcd])
# --------------------------------------------------------------------------------------------


def section_throughput() -> dict:
    print("\n" + "=" * 94)
    print("5. CONSUMER THROUGHPUT IN BULK -- the Levin-side evaluation pattern")
    print(
        "   ([01-offgrid-accessor-cost-on-qcd]; closes if the per-call cost is acceptable)"
    )
    print("=" * 94)

    out = {}
    k = 3.0e8
    for model_key in MODEL_KEYS:
        model = Fixtures.model(model_key)
        cosmology = model.cosmology
        nodes = Fixtures.z_nodes
        z_e3 = horizon_exit_z(cosmology, k, 3.0)
        z_e4 = horizon_exit_z(cosmology, k, 4.0)
        band = nodes[(nodes < sqrt(z_e3 * z_e4)) & (nodes > PRODUCTION_Z_END)]
        z_points = [float(z) for z in band]
        theta_stored = []
        for z_s in band:
            p = _run_Gk(
                model,
                cosmology.units,
                k,
                float(z_s),
                to_redshift_array([PRODUCTION_Z_END]),
            )
            theta_stored.append(
                p["theta_div_2pi_sample"][0] * TWO_PI + p["theta_mod_2pi_sample"][0]
            )
        phi = build_phi_samples(
            k, model.functions.tau, PRODUCTION_Z_END, z_points, theta_stored, sign=-1
        )
        phase = PrimitivePhase(
            k,
            model.functions.tau,
            PRODUCTION_Z_END,
            z_points,
            phi,
            sign=-1,
            model_functions=model.functions,
            label="throughput",
        )

        # a Levin region's abscissae are off-grid by construction; the anchor (z_response) is
        # a node, so exactly one endpoint is off-grid, which is log 03's 26.1 us column on QCD
        rng = np.random.default_rng(20260913)
        u_lo, u_hi = log1p(min(z_points)), log1p(max(z_points))
        probes = np.expm1(rng.uniform(u_lo, u_hi, 4000))

        table = model.functions.tau.table
        before = table.total_evaluations
        _, t_offgrid = best_of(
            lambda: [phase.raw_theta(float(z)) for z in probes], repeats=3
        )
        evals_offgrid = (table.total_evaluations - before) / (3 * len(probes))

        on_grid = np.array(z_points[:: max(1, len(z_points) // 4000 or 1)])
        before = table.total_evaluations
        _, t_ongrid = best_of(
            lambda: [phase.raw_theta(float(z)) for z in on_grid], repeats=3
        )
        evals_ongrid = (table.total_evaluations - before) / (3 * len(on_grid))

        out[model_key] = {
            "us_per_call_offgrid": 1.0e6 * t_offgrid / len(probes),
            "us_per_call_ongrid": 1.0e6 * t_ongrid / len(on_grid),
            "evals_offgrid": evals_offgrid,
            "evals_ongrid": evals_ongrid,
            "n_offgrid": len(probes),
            "n_ongrid": len(on_grid),
        }
        print(
            f"   {model_key:>14}: raw_theta off-grid {out[model_key]['us_per_call_offgrid']:.2f} "
            f"us/call ({evals_offgrid:.2f} integrand evaluations), on-grid "
            f"{out[model_key]['us_per_call_ongrid']:.2f} us/call ({evals_ongrid:.2f}); "
            f"best of 3 over {len(probes)} / {len(on_grid)} calls"
        )
    return out


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------

SECTIONS = {
    "primitives": section_primitives,
    "producers": section_producers,
    "stored": section_stored,
    "consumers": section_consumers,
    "margins": section_margins,
    "cycles": section_cycles,
    "throughput": section_throughput,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(SECTIONS),
        help="run only this section (repeatable); default: all",
    )
    parser.add_argument("--json", type=str, default=None, help="write raw results here")
    args = parser.parse_args()

    wanted = args.section or list(SECTIONS)
    t0 = time.perf_counter()
    results = {}
    for name in SECTIONS:
        if name in wanted:
            results[name] = SECTIONS[name]()
    print(f"\nTotal wall time {time.perf_counter() - t0:.1f} s")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {k: _stringify(v) for k, v in results.items()}, f, indent=1, default=str
            )
        print(f"Raw results written to {args.json}")


def _stringify(obj):
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


if __name__ == "__main__":
    main()
