"""
Re-measure, on this tree, the two cheap baselines of review §4 and §12.3 -- the Green's-function
and transfer-function WKB phase errors at ``k = 1e5/Mpc``, ``z = 0.1``, on the real LambdaCDM
background -- through the *production* phase solver, scored against
``ComputeTargets/tests/wkb_reference_data.json``.

Run from the repository root, **after** ``generate_references.py``:

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/baseline_k1e5.py

The review measured 13.9 rad (G_k, 1.3 s) and 2.01 rad (T_k, 1.1 s) on ``f06f587``; agreement to
5 % is the acceptance. ``k = 3e8`` is deliberately *not* run here -- 64 s and 58 s per object --
and the review's numbers stand as the baseline for that row of README §6.

The script appends a ``"baselines"`` block to the JSON. ``generate_references.py`` carries an
existing block forward, so the two can be re-run in either order, but a *changed* reference
invalidates a stale baseline block: re-run this script after the generator.
"""

import json
import os
import sys
import time
from math import fabs
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ComputeTargets must be imported before Quadrature.integrators.WKB_phase_function:
# ComputeTargets/__init__.py pulls in GkWKBIntegration, which imports WKB_phase_function, so
# importing the integrator first hits a partially-initialised module
from ComputeTargets.TkWKBIntegration import friction_RHS  # noqa: E402
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq, Gk_d_ln_omegaEff_dz  # noqa: E402
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq, Tk_d_ln_omegaEff_dz  # noqa: E402

import Quadrature.integrators.WKB_phase_function as W  # noqa: E402
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    LambdaCDMModel,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    REFERENCE_DATA_PATH,
    horizon_exit_z,
    load_references,
    phase_error,
    production_source_grid,
)
from LiouvilleGreen.constants import TWO_PI  # noqa: E402

K = 1.0e5
ATOL = 1.0e-10
RTOL = 1.0e-8

# review §4 and §12.3, measured on f06f587
REVIEW_GK_ERROR_RAD = 13.9
REVIEW_TK_ERROR_RAD = 2.01


def _k_key(k: float) -> str:
    return f"{k:.6e}"


def _wavenumber_stub(k: float):
    """The duck-typed ``wavenumber_exit_time`` the phase solver needs (only for messages)."""
    return SimpleNamespace(k=SimpleNamespace(k=k, k_inv_Mpc=k, store_id=0))


def unwrapped(payload, index: int) -> float:
    return float(int(payload["theta_div_2pi_sample"][index])) * TWO_PI + float(
        payload["theta_mod_2pi_sample"][index]
    )


def run(label, model, z_anchor, z_sample, omega_sq, d_ln_omega_dz, friction):
    metadata = {}
    if friction is None:
        t0 = time.perf_counter()
        payload = W.integrate_phase_function(
            model,
            _wavenumber_stub(K),
            z_anchor,
            z_sample,
            omega_sq,
            d_ln_omega_dz,
            omega_sq(model, K, z_anchor),
            ATOL,
            RTOL,
            metadata,
            label,
            label,
        )
        elapsed = time.perf_counter() - t0
    else:
        # the Tk path needs the friction integral too, which only the outer entry point runs;
        # call the undecorated body, as docs/gk-wkb-review-fable-2026-09-09/tk2_production_real.py
        # does, with the units check disabled (the stand-in k is not a `wavenumber`)
        saved = W.check_units
        W.check_units = lambda *a, **kw: None
        try:
            t0 = time.perf_counter()
            payload = W.WKB_phase_function._function(
                SimpleNamespace(get=lambda: model),
                _wavenumber_stub(K),
                z_anchor,
                z_sample,
                omega_sq=omega_sq,
                d_ln_omega_dz=d_ln_omega_dz,
                friction=friction,
                atol=ATOL,
                rtol=RTOL,
                task_label=label,
                object_label=label,
            )
            elapsed = time.perf_counter() - t0
        finally:
            W.check_units = saved
        metadata = payload["metadata"]

    n1 = payload["stage_1_data"].RHS_evaluations if payload["stage_1_data"] else 0
    n2 = payload["stage_2_data"].RHS_evaluations if payload["stage_2_data"] else 0
    return payload, elapsed, n1, n2, metadata


def main():
    data = load_references()
    block = data["models"]["LambdaCDMModel"]
    key = _k_key(K)

    lam = LambdaCDMModel()
    z_init = horizon_exit_z(
        lam.cosmology, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid = production_source_grid(z_init)

    z_anchor = block["rho_anchor_z"][key]
    z_sample = grid.truncate(z_anchor, keep="lower")
    print(
        f"k = {K:.3g}/Mpc, LambdaCDM: 3-e-fold sub-horizon start z_init = {z_anchor:.6g}, "
        f"{len(z_sample)} production source samples down to z = {z_sample.min.z:.4g}"
    )

    at_anchor = block["primitives_at_rho_anchor"][key]
    index_of = {z.z: i for i, z in enumerate(z_sample)}

    results = {}

    # ------------------------------------------------------------------------------------
    # G_k
    # ------------------------------------------------------------------------------------
    payload, elapsed, n1, n2, metadata = run(
        "baseline-Gk",
        lam,
        z_anchor,
        z_sample,
        Gk_omegaEff_sq,
        Gk_d_ln_omegaEff_dz,
        None,
    )
    print(
        f"\nG_k: stage 1 {n1} RHS evaluations, {metadata.get('phase_cycle_events')} resets; "
        f"stage 2 {n2}; {elapsed:.2f} s"
    )
    print(f"{'z':>12} {'theta_ref [rad]':>18} {'error [rad]':>13} {'relative':>10}")
    worst_G = 0.0
    error_at_zend_G = None
    for entry in block["rho_G"][key]:
        z = entry["z"]
        if z not in index_of:
            continue
        tau_delta = (
            block["tau_minus_top"][
                [c["index"] for c in block["checkpoints"]].index(entry["index"])
            ]
            - at_anchor["tau_minus_top"]
        )
        reference = -(K * tau_delta + entry["value"])
        got = unwrapped(payload, index_of[z])
        err = phase_error(got, reference)
        worst_G = max(worst_G, err)
        if z == z_sample.min.z:
            error_at_zend_G = err
        print(
            f"{z:>12.5g} {reference:>18.10g} {err:>13.3e} "
            f"{err / fabs(reference):>10.2e}"
        )
    results["Gk"] = {
        "k_inv_Mpc": K,
        "z_init": z_anchor,
        "error_at_z_end_rad": error_at_zend_G,
        "max_error_rad": worst_G,
        "seconds": elapsed,
        "stage_1_RHS_evaluations": n1,
        "stage_2_RHS_evaluations": n2,
        "phase_cycle_events": metadata.get("phase_cycle_events"),
        "review_value_rad": REVIEW_GK_ERROR_RAD,
    }

    # ------------------------------------------------------------------------------------
    # T_k
    # ------------------------------------------------------------------------------------
    payload, elapsed, n1, n2, metadata = run(
        "baseline-Tk",
        lam,
        z_anchor,
        z_sample,
        Tk_omegaEff_sq,
        Tk_d_ln_omegaEff_dz,
        friction_RHS,
    )
    nf = payload["friction_data"].RHS_evaluations
    print(
        f"\nT_k: stage 1 {n1} RHS evaluations, {metadata.get('phase_cycle_events')} resets; "
        f"stage 2 {n2}; friction {nf}; {elapsed:.2f} s"
    )
    print(
        f"{'z':>12} {'theta_ref [rad]':>18} {'error [rad]':>13} {'relative':>10} "
        f"{'F_ref':>10} {'dF':>10}"
    )
    worst_T = 0.0
    worst_F = 0.0
    error_at_zend_T = None
    for entry in block["rho_T"][key]:
        z = entry["z"]
        if z not in index_of:
            continue
        position = [c["index"] for c in block["checkpoints"]].index(entry["index"])
        cs_tau_delta = (
            block["cs_tau_minus_top"][position] - at_anchor["cs_tau_minus_top"]
        )
        reference = -(K * cs_tau_delta + entry["value"])
        got = unwrapped(payload, index_of[z])
        err = phase_error(got, reference)
        worst_T = max(worst_T, err)
        if z == z_sample.min.z:
            error_at_zend_T = err

        F_reference = (
            block["friction_F_minus_top"][position] - at_anchor["friction_F_minus_top"]
        )
        dF = float(payload["friction_sample"][index_of[z]]) - F_reference
        worst_F = max(worst_F, fabs(dF))
        print(
            f"{z:>12.5g} {reference:>18.10g} {err:>13.3e} "
            f"{err / fabs(reference):>10.2e} {F_reference:>10.5f} {dF:>10.2e}"
        )
    results["Tk"] = {
        "k_inv_Mpc": K,
        "z_init": z_anchor,
        "error_at_z_end_rad": error_at_zend_T,
        "max_error_rad": worst_T,
        "max_friction_error": worst_F,
        "seconds": elapsed,
        "stage_1_RHS_evaluations": n1,
        "stage_2_RHS_evaluations": n2,
        "friction_RHS_evaluations": nf,
        "phase_cycle_events": metadata.get("phase_cycle_events"),
        "review_value_rad": REVIEW_TK_ERROR_RAD,
    }

    # ------------------------------------------------------------------------------------
    print(f"\n{'=' * 88}\nAGREEMENT WITH THE REVIEW (acceptance: 5 %)\n{'=' * 88}")
    for sector, review in (("Gk", REVIEW_GK_ERROR_RAD), ("Tk", REVIEW_TK_ERROR_RAD)):
        measured = results[sector]["error_at_z_end_rad"]
        ratio = fabs(measured - review) / review
        results[sector]["relative_difference_from_review"] = ratio
        print(
            f"{sector} phase error at z = 0.1: measured {measured:.4g} rad, "
            f"review {review:.4g} rad -> {100.0 * ratio:.1f} % "
            f"({'PASS' if ratio <= 0.05 else 'FAIL'})"
        )

    data["baselines"] = {
        "model": "LambdaCDMModel",
        "solver": "Quadrature/integrators/WKB_phase_function.integrate_phase_function",
        "atol": ATOL,
        "rtol": RTOL,
        "note": "phase error in radians against the JSON references at the JSON checkpoints; "
        "the sample grid is the production source grid truncated to z <= the 3-e-fold "
        "sub-horizon start point",
        "results": results,
    }
    with open(REFERENCE_DATA_PATH, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
    print(f"\nwrote the baselines block to {REFERENCE_DATA_PATH}")


if __name__ == "__main__":
    main()
