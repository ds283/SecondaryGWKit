"""
What does the cosmology-aware source grid buy the consumer, and what would the prompt's letter
have bought on its own?

Written for prompt 11 of ``prompts/qcd-background-audit/``, which builds the source grid around
the features the cosmology declares. Prompt 10 established the target and refuted the
alternative: no knot scheme recovers the two QCD ``k = 1e5`` consumer rows, while **samples** do
-- refining the +-5 grid intervals around ``QCD_EOS``'s ``T_LO`` crossing by 2x takes them from
34.11 to 1.64 ulp (``G_k``) and from 1876.61 to 74.60 ulp (``T_k``)
(``docs/qcd-background-verification.md`` section 8).

**This script does not build a second scorer.** The expensive half -- the producer runs that
supply the stored samples and the dense reference -- is
``consumer_knot_scheme_scan.build_cases``, imported unchanged, and the ladder rows it prints for
comparison are ``consumer_knot_scheme_scan.resolution_ladder``, also unchanged. The one thing it
adds is the ability to score an *arbitrary* extra sample set rather than the fixed ladder rungs,
which is what is needed to score the grid ``CosmologyConcepts.wavenumber.build_z_sample``
actually produces. Its scoring closure is a transcription of ``resolution_ladder``'s, and
:func:`check_against_ladder` asserts, every run, that the two agree on the shipped consumer to
the last bit.

Three questions, in three sections:

* **A.** The shipped grid, prompt 10's ladder, and the grid this prompt ships -- one table, same
  scorer.
* **B.** The standoff. What does the straddling pair's separation cost or buy, over the window
  the datastore and the redshift arithmetic leave open?
* **C.** The decomposition the prompt's section 2 item 2 needs: straddling pair alone, refinement
  alone, both.

No Ray, no datastore. ~130 s for the default (QCD, both sectors, k = 1e5, plus the LambdaCDM
control).

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/source_grid_consumer_check.py
"""

import argparse
import json
import sys
import time
from math import expm1, fabs, log10
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).parent))

from scipy.interpolate import make_interp_spline  # noqa: E402

from ComputeTargets.BackgroundModel import _cosmology_break_points  # noqa: E402
from ComputeTargets.tests.test_main_plumbing import (  # noqa: E402
    load_main_py_functions,
)
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_Z_END,
    load_references,
)
from CosmologyConcepts import (  # noqa: E402
    SOURCE_GRID_BREAK_HALF_WIDTH,
    SOURCE_GRID_BREAK_REFINEMENT,
    SOURCE_GRID_BREAK_STANDOFF,
    build_z_sample,
)

from consumer_knot_scheme_scan import (  # noqa: E402
    SPLINE_ORDER,
    build_cases,
    resolution_ladder,
)

# the three declared crossings of QCD_Cosmology's equation of state, in u = log(1+z); prompt 07's
# bisected values, transcribed from logs/10-primitive-phase-break-points.md
U_CROSSINGS = (17.565806941870026, 23.197460552819653, 27.485391822044257)

cosmology_feature_redshifts = load_main_py_functions(
    ["cosmology_feature_redshifts"],
    extra_globals={"np": np, "_cosmology_break_points": _cosmology_break_points},
)["cosmology_feature_redshifts"]


# ---------------------------------------------------------------------------------------------
# the scorer -- a transcription of consumer_knot_scheme_scan.resolution_ladder's closure
# ---------------------------------------------------------------------------------------------


def scorer(case, u_break: float):
    """
    Build the pieces ``resolution_ladder`` builds, and return ``(score, with_extra, context)``
    so that an arbitrary extra sample set can be scored with exactly its arithmetic.
    """
    phase = case["make_phase"]()
    sign, lead, z_anchor, k = phase.sign, phase._leading, phase.z_anchor, case["k"]

    z_ref = np.array(sorted(case["ref_theta"].keys()))
    u_ref = np.log1p(z_ref)
    theta_ref = np.array([case["ref_theta"][z] + case["delta"] for z in z_ref])
    phi_ref = theta_ref - sign * k * np.array(
        [lead.delta(float(z), z_anchor) for z in z_ref]
    )

    u_node = np.log1p(np.asarray(phase.z_samples))
    phi_node = np.asarray(phase.phi_samples)
    h = float(np.median(np.diff(u_node)))
    span = max(fabs(v) for v in case["ref_theta"].values())
    ulp = float(np.spacing(span))

    def score(u_fit, phi_fit, tag):
        spline = make_interp_spline(u_fit, phi_fit, k=SPLINE_ORDER)
        fitted = set(np.round(u_fit, 15))
        idx = [i for i in range(len(u_ref)) if round(u_ref[i], 15) not in fitted]
        err = np.array([abs(float(spline(u_ref[i])) - phi_ref[i]) for i in idx])
        near = np.array(
            [
                abs(float(spline(u_ref[i])) - phi_ref[i])
                for i in idx
                if abs(u_ref[i] - u_break) < 3.0 * h
            ]
        )
        return {
            "tag": tag,
            "n_samples": len(u_fit),
            "worst": float(err.max()),
            "worst_ulp": float(err.max() / ulp),
            "near_ulp": float(near.max() / ulp) if near.size else float("nan"),
        }

    def with_extra(u_extra):
        u_fit = np.sort(np.concatenate([u_node, np.asarray(u_extra, dtype=float)]))
        phi_fit = np.interp(u_fit, u_ref, phi_ref)
        phi_fit[np.searchsorted(u_fit, u_node)] = phi_node
        return u_fit, phi_fit

    return (
        score,
        with_extra,
        {"u_node": u_node, "phi_node": phi_node, "h": h, "ulp": ulp},
    )


def check_against_ladder(case, u_break: float, ladder) -> None:
    """
    The self-check that licenses every row this script prints: the transcribed scorer must
    reproduce ``resolution_ladder``'s own first row -- the shipped consumer -- exactly.
    """
    score, _, ctx = scorer(case, u_break)
    mine = score(ctx["u_node"], ctx["phi_node"], "shipped")
    theirs = ladder["rows"][0]
    if mine["worst"] != theirs["worst"] or mine["n_samples"] != theirs["n_samples"]:
        raise RuntimeError(
            f"source_grid_consumer_check: the transcribed scorer does not reproduce "
            f"resolution_ladder's shipped row ({mine['worst']!r} against {theirs['worst']!r})"
        )


# ---------------------------------------------------------------------------------------------
# the grids under test
# ---------------------------------------------------------------------------------------------


def production_grids(cosmology, *, standoff, half_width, refinement, parts="both"):
    """
    The base grid and the cosmology-aware grid at production parameters, as descending arrays of
    z. ``parts`` is ``"both"``, ``"straddle"`` (the prompt's section 2 item 2 alone: the pair, no
    neighbourhood refinement) or ``"refine"`` (the neighbourhood alone, no pair).
    """
    references = load_references()
    z_init = float(references["models"]["LambdaCDMModel"]["grid"]["z_init"])

    base = build_z_sample(
        z_init, PRODUCTION_Z_END, PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z
    ).z_values

    # main.py's own policy function, read out of main.py with ast rather than transcribed, so
    # that the LambdaCDM control below is the control it claims to be
    break_z, feature_z = cosmology_feature_redshifts(
        cosmology, PRODUCTION_Z_END, z_init
    )

    grid = build_z_sample(
        z_init,
        PRODUCTION_Z_END,
        PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
        break_z=break_z,
        feature_z=feature_z,
        standoff=standoff,
        half_width=(0 if parts == "straddle" else half_width),
        refinement=(1 if parts == "straddle" else refinement),
    )
    if parts == "refine":
        # drop the straddling pair, keep the refinement: the other half of the decomposition
        keep = np.array(
            [z not in set(grid.protected_z.tolist()) for z in grid.z_values]
        )
        grid = grid._replace(
            z_values=grid.z_values[keep], protected_z=np.empty(0, dtype=float)
        )
    return base, grid, break_z, feature_z


def extras_in_band(base, grid, band_lo: float, band_hi: float) -> np.ndarray:
    """The samples ``grid`` has that ``base`` does not, restricted to a consumer's own band."""
    extra = np.array(sorted(set(grid.z_values.tolist()) - set(base.tolist())))
    return extra[(extra > band_lo) & (extra < band_hi)]


def score_grid(case, u_break, base, grid, tag):
    score, with_extra, ctx = scorer(case, u_break)
    z_nodes = np.asarray(case["z_points"], dtype=float)
    extra_z = extras_in_band(base, grid, float(z_nodes.min()), float(z_nodes.max()))
    if len(extra_z) == 0:
        return score(
            ctx["u_node"], ctx["phi_node"], tag + " (no extra samples in band)"
        )
    return score(*with_extra(np.log1p(extra_z)), tag)


# ---------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["QCDModel", "LambdaCDMModel"])
    parser.add_argument("--k", nargs="+", type=float, default=[1.0e5])
    parser.add_argument("--json", default=None)
    parser.add_argument("--no-standoff-scan", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("=" * 100)
    print("SOURCE-GRID CONSUMER CHECK -- prompt 11, prompts/qcd-background-audit")
    print("=" * 100)
    cases = build_cases(args.models, args.k)

    out = {}
    for case in cases:
        key = f"{case['model_key']} {case['sector']} k={case['k']:.3e}"
        cosmology = case["cosmology"]
        base, grid, break_z, feature_z = production_grids(
            cosmology,
            standoff=SOURCE_GRID_BREAK_STANDOFF,
            half_width=SOURCE_GRID_BREAK_HALF_WIDTH,
            refinement=SOURCE_GRID_BREAK_REFINEMENT,
        )
        u_break = U_CROSSINGS[0] if len(break_z) > 0 else None

        print("\n" + "-" * 100)
        print(
            f"{key}: base grid {len(base)}, cosmology-aware grid {len(grid.z_values)}"
        )
        print("-" * 100)

        if u_break is None:
            # the control: a cosmology that declares nothing must get the grid it gets today
            identical = np.array_equal(base, grid.z_values)
            print(
                f"   declares nothing: grid bit-identical to today's = {identical}, "
                f"protected set empty = {len(grid.protected_z) == 0}"
            )
            out[key] = {"declares_nothing": bool(identical)}
            continue

        ladder = resolution_ladder(case, u_break)
        check_against_ladder(case, u_break, ladder)
        shipped = ladder["rows"][0]
        print(
            f"   scorer self-check against resolution_ladder: OK "
            f"(shipped {shipped['worst']:.4e} rad, {shipped['worst_ulp']:.2f} ulp, "
            f"{shipped['n_samples']} samples; h = {ladder['h']:.4e})"
        )

        rows = [shipped]
        for row in ladder["rows"]:
            if "refine +-5 interval(s) by 2x" in row["tag"]:
                rows.append(row)

        # --- section A: the grid this prompt ships ---
        rows.append(
            score_grid(case, u_break, base, grid, "SHIPPED HERE: straddle + +-5 x2")
        )

        # --- section C: the decomposition ---
        for parts, tag in (
            ("straddle", "straddling pair alone (prompt section 2 item 2's letter)"),
            ("refine", "+-5 x2 refinement alone, no pair"),
        ):
            _, part_grid, _, _ = production_grids(
                cosmology,
                standoff=SOURCE_GRID_BREAK_STANDOFF,
                half_width=SOURCE_GRID_BREAK_HALF_WIDTH,
                refinement=SOURCE_GRID_BREAK_REFINEMENT,
                parts=parts,
            )
            rows.append(score_grid(case, u_break, base, part_grid, tag))

        for row in rows:
            print(
                f"   {row['n_samples']:5d} samples  "
                f"worst {row['worst']:.4e} rad  {row['worst_ulp']:9.2f} ulp  "
                f"near break {row['near_ulp']:9.2f} ulp   {row['tag']}"
            )

        # --- section B: the standoff ---
        scan = []
        if not args.no_standoff_scan:
            print(
                f"\n   standoff scan (fraction of a grid interval; shipped value "
                f"{SOURCE_GRID_BREAK_STANDOFF:.4g}):"
            )
            for standoff in (
                0.5,
                0.25,
                0.125,
                0.0625,
                1.0 / 32,
                1.0 / 64,
                1.0e-3,
                1.0e-4,
            ):
                _, g, _, _ = production_grids(
                    cosmology,
                    standoff=standoff,
                    half_width=SOURCE_GRID_BREAK_HALF_WIDTH,
                    refinement=SOURCE_GRID_BREAK_REFINEMENT,
                )
                r = score_grid(case, u_break, base, g, f"standoff {standoff:.1e}")
                scan.append({"standoff": standoff, **r})
                print(
                    f"      {standoff:10.5g} of an interval   worst {r['worst']:.4e} rad  "
                    f"{r['worst_ulp']:9.2f} ulp   near break {r['near_ulp']:9.2f} ulp"
                )

        out[key] = {"rows": rows, "standoff_scan": scan}

    print(f"\ntotal {time.perf_counter() - t0:.1f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
