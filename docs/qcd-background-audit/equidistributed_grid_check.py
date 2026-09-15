"""
Score the source grid prompt 15 builds against the one it replaces, on prompt 12's own oracle.

Written for prompt 15 of ``prompts/qcd-background-audit/``, which implements the *decision* half
of audit §8 recommendation 5: the base density of the source grid is now set by the measured
fourth-derivative equidistribution criterion of ``docs/qcd-background-verification.md`` §10, capped
so that no interval is ever wider than the uniform lattice would have put there.

**This script rewrites nothing.** The oracle, the scoring set, the criterion and the
derivative-pad transcription are ``grid_density_criterion.py``'s, imported from it; what is added
here is the comparison of two *grids* -- the one prompt 14 left and the one prompt 15 builds --
each scored in the configuration production actually uses, which §10.4 calls the four-way table
and is the trap prompt 11 fell into:

    background built on grid X  x  consumer samples taken from grid Y

Production is X = Y = the shipped grid, because ``main.py`` passes ``z_sample=z_source_sample`` to
``BackgroundModel``. Sections:

* **A.** The grids themselves: sizes, the cap ("no interval coarser than the base lattice puts
  there, anywhere"), the protected set, the response grid and
  ``[03-derivative-pad-clamp-on-coarse-grids]``.
* **B.** Every (model, sector, k) row in its own production configuration, before and after,
  against the storage floor.
* **C.** The four-way table for the two rows §10.2 records as a miss.

No Ray, no datastore. ~4 minutes.

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/equidistributed_grid_check.py
    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/equidistributed_grid_check.py \
        --models QCDModel --k 1e5 --sections A B
"""

import argparse
import json
import sys
import time
from math import expm1
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from grid_density_criterion import (  # noqa: E402
    Case,
    K_VALUES,
    MODEL_KEYS,
    SECTORS,
    derivative_pad,
)

from ComputeTargets.BackgroundModel import _cosmology_break_points  # noqa: E402
from ComputeTargets.phase_residual import (  # noqa: E402
    phase_residual_integrand,
    residual_node_range,
)
from ComputeTargets.tests.test_gk_wkb_phase import (  # noqa: E402
    lambdacdm_model_with_tables,
    qcd_model_with_tables,
)
from ComputeTargets.tests.test_main_plumbing import (  # noqa: E402
    load_main_py_functions,
)
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    PRODUCTION_RESPONSE_SPARSENESS,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_Z_END,
    load_references,
    to_redshift_array,
)
from CosmologyConcepts import (  # noqa: E402
    build_z_sample,
    redshift_grid_digest,
    SOURCE_GRID_CONSUMER_TARGET_RAD,
    SOURCE_GRID_CROSSING_MASK_U,
    SOURCE_GRID_CUBIC_ERROR_CONST,
    SOURCE_GRID_CURVATURE_FD_STEP_U,
    SOURCE_GRID_CURVATURE_STEP_U,
    SOURCE_GRID_SPLINE_EDGE_FACTOR,
    SOURCE_GRID_SPLINE_EDGE_INTERVALS,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology  # noqa: E402
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018  # noqa: E402
from Units import Mpc_units  # noqa: E402

# main.py:3292 -- the production source and response wavenumber arrays are the same fifty
# logspaced values, and the universal grid has to serve every one of them
PRODUCTION_K_INV_MPC = np.logspace(np.log10(1.0e5), np.log10(3.0e8), 50)

CROSSING_HALO_U = 0.15

_main = load_main_py_functions(
    [
        "cosmology_feature_redshifts",
        "pre_grid_background_proxy",
        "source_grid_spacing_profile",
    ],
    extra_globals={
        "np": np,
        "_cosmology_break_points": _cosmology_break_points,
        "phase_residual_integrand": phase_residual_integrand,
        "residual_node_range": residual_node_range,
        "SOURCE_GRID_CONSUMER_TARGET_RAD": SOURCE_GRID_CONSUMER_TARGET_RAD,
        "SOURCE_GRID_CROSSING_MASK_U": SOURCE_GRID_CROSSING_MASK_U,
        "SOURCE_GRID_CUBIC_ERROR_CONST": SOURCE_GRID_CUBIC_ERROR_CONST,
        "SOURCE_GRID_CURVATURE_FD_STEP_U": SOURCE_GRID_CURVATURE_FD_STEP_U,
        "SOURCE_GRID_CURVATURE_STEP_U": SOURCE_GRID_CURVATURE_STEP_U,
        "SOURCE_GRID_SPLINE_EDGE_FACTOR": SOURCE_GRID_SPLINE_EDGE_FACTOR,
        "SOURCE_GRID_SPLINE_EDGE_INTERVALS": SOURCE_GRID_SPLINE_EDGE_INTERVALS,
    },
)


def build_cosmology(model_key):
    if model_key == "LambdaCDMModel":
        return LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
    return QCD_Cosmology(store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20)


def build_model(model_key, z_values):
    if model_key == "LambdaCDMModel":
        return lambdacdm_model_with_tables(np.asarray(z_values, dtype=float))
    return qcd_model_with_tables(to_redshift_array(np.asarray(z_values, dtype=float)))


def three_grids(cosmology, z_init, units):
    """``(base, shipped, equidistributed)``: the uniform lattice, prompt 11/14's grid, and the
    grid the criterion builds on top of it."""
    base = build_z_sample(
        z_init, PRODUCTION_Z_END, PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z
    ).z_values
    break_z, feature_z = _main["cosmology_feature_redshifts"](
        cosmology, PRODUCTION_Z_END, z_init
    )
    shipped = build_z_sample(
        z_init,
        PRODUCTION_Z_END,
        PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
        break_z=break_z,
        feature_z=feature_z,
    )
    t0 = time.perf_counter()
    spacing = _main["source_grid_spacing_profile"](
        cosmology, base, [k / units.Mpc for k in PRODUCTION_K_INV_MPC]
    )
    seconds = time.perf_counter() - t0
    equi = build_z_sample(
        z_init,
        PRODUCTION_Z_END,
        PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
        break_z=break_z,
        feature_z=feature_z,
        spacing=spacing,
    )
    return base, shipped, equi, np.asarray(break_z, dtype=float), seconds


def widest_interval_against(candidate, reference):
    """The largest ratio of a candidate interval to the reference interval that contains it, in
    ``u = log(1+z)``. At most 1 means "never coarser than the reference, anywhere"."""
    a = np.log1p(np.sort(np.asarray(candidate, dtype=float)))
    b = np.log1p(np.sort(np.asarray(reference, dtype=float)))
    worst, where = 0.0, None
    for i in range(len(a) - 1):
        lo, hi = a[i], a[i + 1]
        j = max(int(np.searchsorted(b, 0.5 * (lo + hi))) - 1, 0)
        j = min(j, len(b) - 2)
        ratio = (hi - lo) / (b[j + 1] - b[j])
        if ratio > worst:
            worst, where = ratio, 0.5 * (lo + hi)
    return worst, (expm1(where) if where is not None else float("nan"))


def score(case, u_fit, breaks_u):
    """Max interpolation error over the scoring set, split near/away from a declared crossing."""
    result = case.score(u_fit, exact=True)
    near = np.zeros_like(result["u"], dtype=bool)
    for b in breaks_u:
        near |= np.abs(result["u"] - b) < CROSSING_HALO_U
    return {
        "n": result["n"],
        "max": float(result["max"]),
        "max_near": float(result["err"][near].max()) if near.any() else 0.0,
        "max_away": float(result["err"][~near].max()) if (~near).any() else 0.0,
        "max_z": expm1(result["max_u"]),
    }


def band_u(case, z_values):
    z = np.asarray(z_values, dtype=float)
    inside = (z >= case.nodes[-1]) & (z <= case.nodes[0])
    return np.sort(np.log1p(z[inside]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_KEYS))
    parser.add_argument("--k", nargs="+", type=float, default=list(K_VALUES))
    parser.add_argument("--sectors", nargs="+", default=list(SECTORS))
    parser.add_argument("--sections", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    want = {s.upper() for s in args.sections}

    t_start = time.perf_counter()
    units = Mpc_units()
    z_init = float(load_references()["models"]["LambdaCDMModel"]["grid"]["z_init"])

    print("=" * 102)
    print("THE EQUIDISTRIBUTED SOURCE GRID -- prompt 15, prompts/qcd-background-audit")
    print("=" * 102)

    grids, out = {}, {}
    for model_key in args.models:
        cosmology = build_cosmology(model_key)
        base, shipped, equi, break_z, seconds = three_grids(cosmology, z_init, units)
        grids[model_key] = (base, shipped, equi, break_z, cosmology)
        print(
            f"   [{model_key}] base {len(base)}, shipped {len(shipped.z_values)}, "
            f"equidistributed {len(equi.z_values)}; the criterion took {seconds:.2f} s over "
            f"{len(PRODUCTION_K_INV_MPC)} wavenumbers in 2 sectors"
        )

    # =========================================================================================
    if "A" in want:
        print("\n" + "=" * 102)
        print("A. THE GRIDS, THE CAP, THE PROTECTED SET AND THE RESPONSE GRID")
        print("=" * 102)
        a_out = {}
        for model_key in args.models:
            base, shipped, equi, break_z, _ = grids[model_key]
            z = equi.z_values
            worst_base, z_base = widest_interval_against(z, base)
            worst_shipped, z_shipped = widest_interval_against(z, shipped.z_values)
            superset = set(shipped.z_values.tolist()) <= set(z.tolist())

            src = to_redshift_array(z)
            protected_values = set(equi.protected_z.tolist())
            protected = [r for r in src if float(r) in protected_values]
            resp = src.winnow(
                sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=protected
            )
            resp_values = np.array(resp.as_float_list(), dtype=float)

            src_shipped = to_redshift_array(shipped.z_values)
            protected_shipped = [
                r for r in src_shipped if float(r) in set(shipped.protected_z.tolist())
            ]
            resp_shipped = src_shipped.winnow(
                sparseness=PRODUCTION_RESPONSE_SPARSENESS, protect=protected_shipped
            )

            u = np.log1p(z[::-1])
            pad = derivative_pad(u)
            separation = np.abs(np.diff(z)) / (1.0 + z[1:])

            print(f"\n   {model_key}")
            print(
                f"      samples {len(shipped.z_values)} -> {len(z)}; digest "
                f"{redshift_grid_digest(shipped.z_values)} -> {redshift_grid_digest(z)}"
            )
            print(
                f"      widest interval against the base lattice {worst_base:.4f}x "
                f"(at z = {z_base:.4e}); against the shipped grid {worst_shipped:.4f}x -- "
                f"'never coarser than today, anywhere' is {worst_base <= 1.0 + 1e-12}"
            )
            print(
                f"      the shipped grid is a subset of it: {superset}; protected points "
                f"{len(equi.protected_z)} (was {len(shipped.protected_z)}), all present: "
                f"{set(equi.protected_z.tolist()) <= set(z.tolist())}"
            )
            print(
                f"      strictly descending {bool(np.all(np.diff(z) < 0.0))}, distinct "
                f"{len(set(z.tolist())) == len(z)}, closest approach {separation.min():.4e} "
                "relative in (1+z)"
            )
            print(
                f"      response grid by the production stride {PRODUCTION_RESPONSE_SPARSENESS}: "
                f"{len(resp)} samples (was {len(resp_shipped)}), a subset of the source grid: "
                f"{set(resp_values.tolist()) <= set(z.tolist())}"
            )
            print(
                f"      derivative pad h_lo {pad['h_lo']:.4e} (first {pad['first']:.4e}, floor "
                f"cap {pad['floor_cap']:.4e}) -- [03-derivative-pad-clamp-on-coarse-grids] "
                f"{'BINDS' if pad['clamped'] else 'does not bind'}"
            )
            a_out[model_key] = {
                "n_base": int(len(base)),
                "n_shipped": int(len(shipped.z_values)),
                "n_equi": int(len(z)),
                "digest_shipped": redshift_grid_digest(shipped.z_values),
                "digest_equi": redshift_grid_digest(z),
                "widest_vs_base": worst_base,
                "widest_vs_shipped": worst_shipped,
                "shipped_is_subset": bool(superset),
                "n_protected": int(len(equi.protected_z)),
                "n_response": int(len(resp)),
                "n_response_shipped": int(len(resp_shipped)),
                "min_separation": float(separation.min()),
                "pad": pad,
            }
        out["A"] = a_out

    # =========================================================================================
    if "B" in want or "C" in want:
        built = {}
        for model_key in args.models:
            base, shipped, equi, break_z, _ = grids[model_key]
            built[(model_key, "shipped")] = build_model(model_key, shipped.z_values)
            built[(model_key, "equi")] = build_model(model_key, equi.z_values)

    if "B" in want:
        print("\n" + "=" * 102)
        print(
            "B. EVERY ROW IN ITS OWN PRODUCTION CONFIGURATION -- background built on the grid the\n"
            "   run samples, samples taken from it. 'floor' is 1 ulp of the k*tau span; 'away'\n"
            "   excludes 0.15 in u either side of a declared crossing."
        )
        print("=" * 102)
        b_out = {}
        for model_key in args.models:
            base, shipped, equi, break_z, _ = grids[model_key]
            breaks_u = np.log1p(break_z) if len(break_z) else np.empty(0)
            print(
                f"\n   {model_key}"
                f"\n      {'sector':<7}{'k':<11}{'n before':>9}{'n after':>9}"
                f"{'away before':>14}{'away after':>14}{'floor':>13}"
                f"{'before/floor':>14}{'after/floor':>13}"
            )
            for sector in args.sectors:
                for k in args.k:
                    rows = {}
                    for tag, grid_z in (
                        ("shipped", shipped.z_values),
                        ("equi", equi.z_values),
                    ):
                        model = built[(model_key, tag)]
                        case = Case(
                            model, model_key, sector, k, grid_z, breaks_u=breaks_u
                        )
                        rows[tag] = score(case, case.u_node, breaks_u)
                        rows[tag]["floor"] = case.ulp
                    before, after = rows["shipped"], rows["equi"]
                    floor = before["floor"]
                    print(
                        f"      {sector:<7}{k:<11.2e}{before['n']:>9}{after['n']:>9}"
                        f"{before['max_away']:>14.4e}{after['max_away']:>14.4e}"
                        f"{floor:>13.4e}{before['max_away'] / floor:>14.2f}"
                        f"{after['max_away'] / floor:>13.2f}"
                    )
                    b_out[f"{model_key} {sector} {k:.3e}"] = rows
        out["B"] = b_out

    # =========================================================================================
    if "C" in want:
        print("\n" + "=" * 102)
        print(
            "C. THE FOUR-WAY TABLE (verification document §10.4) -- the grid is not only a sample\n"
            "   set; on a cosmology that declares break points it is also the lattice the\n"
            "   background's own derivative fields are splined on."
        )
        print("=" * 102)
        c_out = {}
        for model_key in args.models:
            base, shipped, equi, break_z, _ = grids[model_key]
            breaks_u = np.log1p(break_z) if len(break_z) else np.empty(0)
            for sector in args.sectors:
                for k in args.k:
                    print(
                        f"\n   {model_key} {sector} k = {k:.3e}"
                        f"\n      {'background on':<16}{'samples from':<16}{'n':>7}"
                        f"{'max/rad':>13}{'near a crossing':>18}{'away':>13}"
                    )
                    for bg_tag in ("shipped", "equi"):
                        model = built[(model_key, bg_tag)]
                        grid_for_case = (
                            equi.z_values if bg_tag == "equi" else shipped.z_values
                        )
                        case = Case(
                            model,
                            model_key,
                            sector,
                            k,
                            grid_for_case,
                            breaks_u=breaks_u,
                        )
                        for s_tag, s_z in (
                            ("shipped", shipped.z_values),
                            ("equi", equi.z_values),
                        ):
                            row = score(case, band_u(case, s_z), breaks_u)
                            print(
                                f"      {bg_tag:<16}{s_tag:<16}{row['n']:>7}"
                                f"{row['max']:>13.4e}{row['max_near']:>18.4e}"
                                f"{row['max_away']:>13.4e}"
                            )
                            c_out[
                                f"{model_key} {sector} {k:.3e} bg={bg_tag} s={s_tag}"
                            ] = row
        out["C"] = c_out

    print(f"\ntotal {time.perf_counter() - t_start:.1f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"raw results written to {args.json}")


if __name__ == "__main__":
    main()
