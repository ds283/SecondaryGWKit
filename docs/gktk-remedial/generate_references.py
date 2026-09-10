"""
Generate ``ComputeTargets/tests/wkb_reference_data.json`` for the Gk/Tk WKB phase remedial
campaign (prompt 01).

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/generate_references.py

Long runtime is expected and permitted; the test module that reads the JSON must not repeat any
of this work. Every reference is evaluated **at the supplied double** (``mpf(float(z))``), never
at a re-derived argument.
"""

import json
import os
import platform
import sys
import time
from datetime import date
from math import log1p, fsum

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mpmath as mp  # noqa: E402
import reference_lib as R  # noqa: E402

from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    LambdaCDMModel,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_END,
    QCDModel,
    RadiationModel,
    REFERENCE_DATA_PATH,
    REFERENCE_K_VALUES,
    horizon_exit_z,
    production_source_grid,
)

# the number of e-folds *inside* the horizon at which the production WKB region starts, and
# therefore the anchor of every residual reference (review §4, §12.2 tabulate rho from here)
RHO_ANCHOR_EFOLDS_SUBH = 3

QUAD_PRIMARY_EPSREL = 1.5e-14
QUAD_CROSSCHECK_EPSREL = 1e-12
GAUSS_CROSSCHECK_RTOL = 1e-14
GAUSS_CROSSCHECK_MAX_LEVEL = 8


def k_key(k: float) -> str:
    return f"{k:.6e}"


# =============================================================================================
# mpmath models (RadiationModel, LambdaCDMModel)
# =============================================================================================


def build_mpmath_model(model, bg, z_nodes, checkpoints, k_anchor):
    """
    Assemble the reference block for a model whose background has a closed form, using mpmath at
    40 digits.
    """
    z_top = float(z_nodes[0])
    block = {}

    # --- primitives, cumulated from the top node down through the checkpoints ---------------
    for quantity, json_key, sign in (
        ("tau", "tau_minus_top", +1),
        ("cs_tau", "cs_tau_minus_top", +1),
        ("friction", "friction_F_minus_top", -1),
    ):
        running = mp.mpf(0)
        values = []
        previous_z = z_top
        for j in checkpoints:
            z_j = float(z_nodes[j])
            if z_j < previous_z:
                running += R.mp_increment(bg, quantity, z_j, previous_z)
                previous_z = z_j
            values.append(float(sign * running))
        block[json_key] = values

    # --- residuals, anchored at each k's own 3-e-fold sub-horizon point ----------------------
    rho_anchor = {}
    at_anchor = {}
    for sector in ("rho_G", "rho_T"):
        block[sector] = {}
    for k in REFERENCE_K_VALUES:
        z_anchor = k_anchor[k]
        rho_anchor[k_key(k)] = z_anchor
        at_anchor[k_key(k)] = {
            "tau_minus_top": float(R.mp_increment(bg, "tau", z_anchor, z_top)),
            "cs_tau_minus_top": float(R.mp_increment(bg, "cs_tau", z_anchor, z_top)),
            "friction_F_minus_top": float(
                -R.mp_increment(bg, "friction", z_anchor, z_top)
            ),
        }
        below = [j for j in checkpoints if float(z_nodes[j]) < z_anchor]
        for sector in ("rho_G", "rho_T"):
            running = mp.mpf(0)
            previous_z = z_anchor
            entries = []
            for j in below:
                z_j = float(z_nodes[j])
                running += R.mp_increment(bg, sector, z_j, previous_z, k=k)
                previous_z = z_j
                entries.append({"index": int(j), "z": z_j, "value": float(running)})
            block[sector][k_key(k)] = entries
    block["rho_anchor_z"] = rho_anchor
    block["primitives_at_rho_anchor"] = at_anchor

    # --- short baselines --------------------------------------------------------------------
    block["short_baseline"] = []
    for j in R.short_baseline_locations(z_nodes):
        z_hi = float(z_nodes[j])
        z_lo = float(z_nodes[j + 1])
        z_frac = R.fractional_point(z_hi, z_lo)
        block["short_baseline"].append(
            {
                "index": int(j),
                "z_node_hi": z_hi,
                "z_node_lo": z_lo,
                "z_fraction_offset": R.FRACTIONAL_OFFSET,
                "z_fraction": z_frac,
                "delta_tau_full": float(R.mp_increment(bg, "tau", z_lo, z_hi)),
                "delta_tau_fraction": float(R.mp_increment(bg, "tau", z_frac, z_hi)),
            }
        )

    return block


# =============================================================================================
# converged adaptive quadrature of the double-precision integrand (QCDModel)
# =============================================================================================


def _adaptive_reference(f, u_edges):
    """
    Primary value plus the two cross-checks, over the supplied production interval edges.

    :return: ``(primary, {"quad_1e-12": rel, "gauss40_bisect": rel}, diagnostics)``
    """
    primary, parts, quad_abserr, quad_warned = R.quad_sum_over_intervals(
        f, u_edges, QUAD_PRIMARY_EPSREL
    )
    loose, _, _, _ = R.quad_sum_over_intervals(f, u_edges, QUAD_CROSSCHECK_EPSREL)
    gl, _, gl_change, gl_level = R.gauss_sum_over_intervals(
        f, u_edges, GAUSS_CROSSCHECK_RTOL, GAUSS_CROSSCHECK_MAX_LEVEL
    )

    denom = abs(primary) if primary != 0.0 else 1.0
    agreement = {
        "quad_epsrel_1e-12": abs(primary - loose) / denom,
        "gauss40_bisect": abs(primary - gl) / denom,
    }
    diagnostics = {
        "quad_max_reported_abserr": quad_abserr,
        "quad_intervals_with_warnings": quad_warned,
        "quad_intervals": len(u_edges) - 1,
        "gauss40_worst_interval_relative_change": gl_change,
        "gauss40_worst_level": gl_level,
    }
    return primary, parts, agreement, diagnostics


def build_adaptive_model(model, z_nodes, checkpoints, k_anchor):
    """
    Assemble the reference block for ``QCDModel``: SciPy ``quad`` per production interval in
    ``u = log(1+z)``, summed with ``math.fsum``, cross-checked two ways.
    """
    functions = model.functions
    u_ascending = np.log1p(np.asarray(z_nodes, dtype=float))[::-1]
    n = len(z_nodes)
    # index into the ascending u array of the descending node index j
    asc = lambda j: n - 1 - j  # noqa: E731

    block = {}
    floors = {}

    for quantity, json_key, sign, integrand in (
        ("tau", "tau_minus_top", +1, R.integrand_tau_double(functions)),
        ("cs_tau", "cs_tau_minus_top", +1, R.integrand_cs_tau_double(functions)),
        (
            "friction",
            "friction_F_minus_top",
            -1,
            R.integrand_friction_double(functions),
        ),
    ):
        primary, parts, agreement, diagnostics = _adaptive_reference(
            integrand, u_ascending
        )
        # parts[i] is the integral over [u_ascending[i], u_ascending[i+1]]; the cumulative
        # int_{z_j}^{z_top} is the sum of the parts above node j
        values = []
        for j in checkpoints:
            i = asc(j)
            values.append(sign * fsum(parts[i:]))
        block[json_key] = values
        floors[json_key] = {"agreement": agreement, "diagnostics": diagnostics}

    rho_anchor = {}
    at_anchor = {}
    for sector in ("rho_G", "rho_T"):
        block[sector] = {}
    for k in REFERENCE_K_VALUES:
        z_anchor = k_anchor[k]
        rho_anchor[k_key(k)] = z_anchor
        u_anchor = log1p(z_anchor)

        # the primitives at the (off-grid) anchor: the production intervals above it plus one
        # partial interval from the anchor up to the next node
        i_first = int(np.searchsorted(u_ascending, u_anchor))
        anchor_edges = np.concatenate([[u_anchor], u_ascending[i_first:]])
        at_anchor[k_key(k)] = {}
        for quantity, json_key, sign, integrand in (
            ("tau", "tau_minus_top", +1, R.integrand_tau_double(functions)),
            ("cs_tau", "cs_tau_minus_top", +1, R.integrand_cs_tau_double(functions)),
            (
                "friction",
                "friction_F_minus_top",
                -1,
                R.integrand_friction_double(functions),
            ),
        ):
            total, _, _, _ = R.quad_sum_over_intervals(
                integrand, anchor_edges, QUAD_PRIMARY_EPSREL
            )
            at_anchor[k_key(k)][json_key] = sign * total

        below = [j for j in checkpoints if float(z_nodes[j]) < z_anchor]
        if not below:
            for sector in ("rho_G", "rho_T"):
                block[sector][k_key(k)] = []
            continue

        # integrate on the production intervals strictly below the anchor, plus one partial
        # interval from the anchor (off-grid, a root_scalar root) down to the next node
        i_lo = asc(max(below))  # ascending index of the lowest-z checkpoint
        i_anchor = int(
            np.searchsorted(u_ascending, u_anchor)
        )  # first node above u_anchor
        edges = np.concatenate([u_ascending[i_lo:i_anchor], [u_anchor]])

        for sector, maker in (
            ("rho_G", R.integrand_rho_G_double),
            ("rho_T", R.integrand_rho_T_double),
        ):
            integrand = maker(functions, k)
            primary, parts, agreement, diagnostics = _adaptive_reference(
                integrand, edges
            )
            entries = []
            for j in below:
                i = asc(j)
                entries.append(
                    {
                        "index": int(j),
                        "z": float(z_nodes[j]),
                        "value": fsum(parts[i - i_lo :]),
                    }
                )
            block[sector][k_key(k)] = entries
            floors[f"{sector}[{k_key(k)}]"] = {
                "agreement": agreement,
                "diagnostics": diagnostics,
            }
    block["rho_anchor_z"] = rho_anchor
    block["primitives_at_rho_anchor"] = at_anchor

    block["short_baseline"] = []
    integrand = R.integrand_tau_double(functions)
    for j in R.short_baseline_locations(z_nodes):
        z_hi = float(z_nodes[j])
        z_lo = float(z_nodes[j + 1])
        z_frac = R.fractional_point(z_hi, z_lo)
        u_hi, u_lo, u_frac = log1p(z_hi), log1p(z_lo), log1p(z_frac)
        full, _, agree_full, _ = _adaptive_reference(integrand, np.array([u_lo, u_hi]))
        frac, _, agree_frac, _ = _adaptive_reference(
            integrand, np.array([u_frac, u_hi])
        )
        block["short_baseline"].append(
            {
                "index": int(j),
                "z_node_hi": z_hi,
                "z_node_lo": z_lo,
                "z_fraction_offset": R.FRACTIONAL_OFFSET,
                "z_fraction": z_frac,
                "delta_tau_full": full,
                "delta_tau_fraction": frac,
                "agreement_full": agree_full,
                "agreement_fraction": agree_frac,
            }
        )

    return block, floors


# =============================================================================================
# driver
# =============================================================================================

SCHEMA_NOTES = {
    "units": "Mpc_units (Mpc = 1.0): tau, cs_tau in Mpc; friction_F, rho_G, rho_T dimensionless "
    "(radians for the residuals).",
    "grid": "The production source grid: wavenumber_exit_time.populate_z_sample reproduced by "
    "ComputeTargets.tests.wkb_reference.production_source_z_values -- descending, log-spaced "
    "in z (NOT in 1+z), 100 points per decade of z, from the 5-e-fold super-horizon point of "
    "k = 3e8/Mpc down to z = 0.1.",
    "checkpoints": "A list of {index, z}: index is into the descending node array, z is the "
    "exact double of that node. tau_minus_top / cs_tau_minus_top / friction_F_minus_top are "
    "parallel lists in the same order.",
    "tau_minus_top": "tau(z_j) - tau(z_top) = int_{z_j}^{z_top} dz/H > 0. This is "
    "tau.delta(z_top, z_j) in the README §2 (c) convention.",
    "cs_tau_minus_top": "cs_tau(z_j) - cs_tau(z_top) = int_{z_j}^{z_top} c_s dz/H > 0, with "
    "c_s^2 = wPerturbations.",
    "friction_F_minus_top": "F(z_j) - F(z_top) = -int_{z_j}^{z_top} (3/2)(1+c_s^2) dz/(1+z) < 0, "
    "where F is the primitive of TkWKBIntegration.friction_RHS (dF/dz = +(3/2)(1+c_s^2)/(1+z)). "
    "Note the sign is opposite to tau and cs_tau, which increase towards lower z.",
    "primitives_at_rho_anchor": "Per k: tau_minus_top, cs_tau_minus_top and "
    "friction_F_minus_top evaluated at that k's rho anchor, which is off the node grid. Subtract "
    "these from a checkpoint's entry to obtain the primitive increment from the anchor, which is "
    "what the WKB phase from the production start point needs.",
    "rho_anchor_z": "Per k: the 3-e-folds-sub-horizon redshift for that k, where the production "
    "WKB region begins. It is a root of k(1+z)/H(z) = e^3 and is deliberately OFF the node grid "
    "(RECONCILIATION.md §2 item 5).",
    "rho_G": "Per k, a list of {index, z, value} for the checkpoints strictly below the anchor. "
    "value = int_z^{z_anchor} C/(omega + k/H) dz with omega^2 = (k/H)^2 + C and C the "
    "non-leading (B + C) terms of Gk_omegaEff_sq. The Green's-function phase is then "
    "theta_G(z; z_anchor) = -k*tau.delta(z_anchor, z) - value.",
    "rho_T": "As rho_G, with C_T the non-leading terms of Tk_omegaEff_sq, omega_T^2 = "
    "c_s^2 (k/H)^2 + C_T and the leading primitive cs_tau. NOTE the sign convention: this is the "
    "rho of README §2 (a)/(c), theta_T = -(x - x_i) - rho_T; review §12.4's quoted 'rho_T = "
    "1/x_i - 1/x' is its negative.",
    "short_baseline": "Three one-grid-interval Delta-tau references per model, near z = 1e6, "
    "1e2 and 1, plus the same interval truncated at an off-grid endpoint 37% of the way through "
    "it in u = log(1+z). delta_tau_full = int_{z_node_lo}^{z_node_hi} dz/H, "
    "delta_tau_fraction = int_{z_fraction}^{z_node_hi} dz/H. Both positive.",
    "method": "Per model: the reference method and its own floor.",
}


def main():
    t_start = time.perf_counter()

    print("** building stand-in models")
    lam = LambdaCDMModel()
    z_init = horizon_exit_z(
        lam.cosmology, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid = production_source_grid(z_init)
    z_nodes_lam = np.array([z.z for z in grid], dtype=float)
    print(
        f"   LambdaCDM: z_init={z_init:.6g}, {len(z_nodes_lam)} nodes, "
        f"z_end={z_nodes_lam[-1]:.6g}"
    )

    t0 = time.perf_counter()
    qcd = QCDModel(grid)
    qcd_build_time = time.perf_counter() - t0
    print(f"   QCDModel: compute_background + splines took {qcd_build_time:.3f} s")

    rad = RadiationModel()
    z_init_rad = horizon_exit_z(
        rad, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid_rad = production_source_grid(z_init_rad)
    z_nodes_rad = np.array([z.z for z in grid_rad], dtype=float)
    print(
        f"   RadiationModel: z_init={z_init_rad:.6g}, {len(z_nodes_rad)} nodes, "
        f"z_end={z_nodes_rad[-1]:.6g}"
    )

    payload = {
        "schema_version": 1,
        "generated": date.today().isoformat(),
        "generator": "docs/gktk-remedial/generate_references.py",
        "campaign": "prompts/GkTk-remedial (prompt 01)",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mpmath": mp.__version__,
        },
        "schema": SCHEMA_NOTES,
        "k_values": list(REFERENCE_K_VALUES),
        "k_keys": [k_key(k) for k in REFERENCE_K_VALUES],
        "rho_anchor_efolds_subh": RHO_ANCHOR_EFOLDS_SUBH,
        "models": {},
    }

    for model, z_nodes, hubble_source, grid_obj in (
        (rad, z_nodes_rad, rad, grid_rad),
        (lam, z_nodes_lam, lam.cosmology, grid),
        (qcd, z_nodes_lam, qcd.cosmology, grid),
    ):
        print(f"\n** references for {model.name}")
        checkpoints = R.select_checkpoints(z_nodes)
        k_anchor = {
            k: horizon_exit_z(hubble_source, k, RHO_ANCHOR_EFOLDS_SUBH)
            for k in REFERENCE_K_VALUES
        }
        print(
            "   checkpoints: "
            + ", ".join(f"{z_nodes[j]:.4g}" for j in checkpoints)
            + f"  ({len(checkpoints)} nodes)"
        )
        print(
            "   rho anchors: "
            + ", ".join(f"k={k:.3g}: z={k_anchor[k]:.5g}" for k in REFERENCE_K_VALUES)
        )

        t0 = time.perf_counter()
        if model.name == "QCDModel":
            block, floors = build_adaptive_model(model, z_nodes, checkpoints, k_anchor)
            block["method"] = (
                "converged adaptive quadrature of the double-precision integrand: SciPy quad in "
                f"u = log(1+z) per production interval at epsabs=0, epsrel={QUAD_PRIMARY_EPSREL:g}, "
                "summed with math.fsum; cross-checked against the same at "
                f"epsrel={QUAD_CROSSCHECK_EPSREL:g} and against composite Gauss-Legendre order 40 "
                f"with uniform bisection to rtol={GAUSS_CROSSCHECK_RTOL:g} "
                f"(max {GAUSS_CROSSCHECK_MAX_LEVEL} levels). H(z) here is itself a spline "
                "evaluation of T(z), so no higher-precision reference exists."
            )
            block["reference_floor"] = floors
            block["build_time_seconds"] = qcd_build_time
        else:
            bg = (
                R.MpRadiation(model.H0)
                if model.name == "RadiationModel"
                else R.MpLambdaCDM(model.cosmology)
            )
            block = build_mpmath_model(model, bg, z_nodes, checkpoints, k_anchor)
            block["method"] = (
                "mpmath at mp.dps = 40: mp.quad in u = log(1+z) with breakpoints at every "
                "integer power of ten of 1+z; H, epsilon, epsilon', w and w' re-derived at 40 "
                "digits from the closed forms of the cosmology model, so the reference never "
                "passes through the double-precision ModelFunctions."
            )
            block["reference_floor"] = {
                "note": "mp.dps = 40 against a closed-form background; the reference's own "
                "floor is ~1e-38 relative, far below every acceptance threshold in README §6. "
                "Values are stored as doubles, so the JSON itself is the 1-ulp limit."
            }
        elapsed = time.perf_counter() - t0
        print(f"   done in {elapsed:.1f} s")

        block["grid"] = {
            "z_init": float(z_nodes[0]),
            "z_end": PRODUCTION_Z_END,
            "samples_per_log10z": PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
            "num_nodes": int(len(z_nodes)),
            "superhorizon_efolds": PRODUCTION_SUPERHORIZON_EFOLDS,
            "largest_k_inv_Mpc": PRODUCTION_LARGEST_K_INV_MPC,
        }
        block["z_top"] = float(z_nodes[0])
        block["checkpoints"] = [
            {"index": int(j), "z": float(z_nodes[j])} for j in checkpoints
        ]
        block["generation_time_seconds"] = elapsed
        if model.name == "RadiationModel":
            block["H0"] = model.H0

        payload["models"][model.name] = block

    # carry an existing "baselines" block forward (written by baseline_k1e5.py)
    if os.path.exists(REFERENCE_DATA_PATH):
        with open(REFERENCE_DATA_PATH, "r") as f:
            previous = json.load(f)
        if "baselines" in previous:
            payload["baselines"] = previous["baselines"]

    with open(REFERENCE_DATA_PATH, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=False)

    print(
        f"\n** wrote {REFERENCE_DATA_PATH} "
        f"({os.path.getsize(REFERENCE_DATA_PATH)} bytes) in "
        f"{time.perf_counter() - t_start:.1f} s total"
    )


if __name__ == "__main__":
    main()
