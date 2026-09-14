"""
Regenerate the ``QCDModel`` block of ``ComputeTargets/tests/wkb_reference_data.json``
(``prompts/qcd-background-audit``, prompt 02).

This is the qcd-background-audit campaign's own tool, reduced from
``docs/gktk-remedial/generate_references.py`` (the ``GkTk-remedial`` campaign's generator, which
this prompt must not touch) to the one thing this campaign needs: rebuilding **only**
``payload["models"]["QCDModel"]``, in the JSON's existing schema, leaving
``payload["models"]["RadiationModel"]``, ``payload["models"]["LambdaCDMModel"]`` and the
top-level ``convergence`` block (written by the separate ``docs/gktk-remedial/residual_convergence.py``,
see ``REFERENCE-FIXTURE.md`` §4) untouched byte for byte.

Every QCD reference in the shipped JSON is built from the shipped ``T(z)`` spline (its own
``method`` string says so: *"H(z) here is itself a spline evaluation of T(z), so no
higher-precision reference exists"*), so **on this tree the reference is circular** -- it is
scored against the very representation this campaign is about to replace. Prompt 06 makes it
non-circular, once the accurate root solve (``CosmologyModels/tests/T_z_reference.py``) is what
the representation is built from rather than merely a check on it.

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run
    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py

``--dry-run`` recomputes the QCD block and reports whether its *science content* (the arrays in
:data:`SCIENCE_KEYS`) would change, without writing anything. Timing fields
(``build_time_seconds``, ``generation_time_seconds``) and the top-level ``generated`` date are
deliberately excluded from that comparison -- they differ on every run regardless of whether the
representation moved, so they would never let a dry run report "no change". On this tree (before
any of prompts 04-07 has landed) the science content is bit-identical to the shipped file, and
this script deliberately makes **no write at all** in that case -- not even to the provenance
fields -- so that ``git status --porcelain ComputeTargets/tests/wkb_reference_data.json`` is
empty whether or not ``--dry-run`` was passed. This is prompt 02's acceptance test: it changes no
number.

When a later prompt (04, 05, 06 or 07) *does* move the representation, a real run (without
``--dry-run``) rewrites the QCD block's science content, refreshes its timing fields, sets the
top-level ``generated`` date, and **appends** (never replaces) a provenance line to the block's
own ``method`` string recording which script and, once ``T_Z_REPRESENTATION_VERSION`` exists
(prompt 03), which version produced it.

No Ray and no datastore. Building ``QCDModel`` (an undecorated ``compute_background`` call over
the ~1,700-point production grid) is the dominant cost; see the log for the measured wall time.
"""

import argparse
import json
import os
import sys
import time
from datetime import date
from math import log1p

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GKTK_REMEDIAL_DIR = os.path.join(os.path.dirname(_THIS_DIR), "gktk-remedial")
sys.path.insert(0, _GKTK_REMEDIAL_DIR)

import reference_lib as R  # noqa: E402

from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    LambdaCDMModel,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_END,
    QCDModel,
    REFERENCE_DATA_PATH,
    REFERENCE_K_VALUES,
    horizon_exit_z,
    production_source_grid,
)

# the number of e-folds *inside* the horizon at which the production WKB region starts -- the
# anchor of every residual reference (docs/gktk-remedial/generate_references.py's own constant,
# reproduced here so this script needs no import from that file)
RHO_ANCHOR_EFOLDS_SUBH = 3

QUAD_PRIMARY_EPSREL = 1.5e-14
QUAD_CROSSCHECK_EPSREL = 1e-12
GAUSS_CROSSCHECK_RTOL = 1e-14
GAUSS_CROSSCHECK_MAX_LEVEL = 8

# The keys in the QCDModel block that are the *science content*. An unchanged value here, on a
# dry run, is this prompt's acceptance test. Excluded deliberately: "method" (free text, only
# ever appended to when the science below moves), "build_time_seconds" and
# "generation_time_seconds" (wall-clock measurements that differ on every run whether or not the
# representation moved).
SCIENCE_KEYS = (
    "tau_minus_top",
    "cs_tau_minus_top",
    "friction_F_minus_top",
    "rho_G",
    "rho_T",
    "rho_anchor_z",
    "primitives_at_rho_anchor",
    "short_baseline",
    "reference_floor",
    "grid",
    "z_top",
    "checkpoints",
)


def k_key(k: float) -> str:
    return f"{k:.6e}"


# =============================================================================================
# converged adaptive quadrature of the double-precision integrand (QCDModel only)
#
# Reduced from docs/gktk-remedial/generate_references.py's build_adaptive_model: same method,
# same tolerances, QCD-only.
# =============================================================================================


def _adaptive_reference(f, u_edges):
    """
    Primary value plus the two cross-checks, over the supplied production interval edges.

    :return: ``(primary, {"quad_epsrel_1e-12": rel, "gauss40_bisect": rel}, diagnostics)``
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


def build_qcd_block(model, z_nodes, checkpoints, k_anchor):
    """
    Assemble the science-content dict for ``QCDModel``, exactly as
    ``generate_references.py.build_adaptive_model`` does. Returns ``(block, floors)`` where
    ``block`` holds ``SCIENCE_KEYS`` minus ``reference_floor``/``grid``/``z_top``/``checkpoints``
    (the caller adds those) and ``floors`` is this run's ``reference_floor`` value.
    """
    from math import fsum

    functions = model.functions
    u_ascending = np.log1p(np.asarray(z_nodes, dtype=float))[::-1]
    n = len(z_nodes)
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

        i_lo = asc(max(below))
        i_anchor = int(np.searchsorted(u_ascending, u_anchor))
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
# comparison: does the science content differ from the shipped file?
# =============================================================================================


def _normalise(value):
    """A JSON round trip, so that float/np.float64 and tuple/list differences do not register as
    changes that are only representational."""
    return json.loads(json.dumps(value))


def science_diff(existing_block: dict, new_block: dict):
    """
    Compare ``SCIENCE_KEYS`` between the shipped block and a freshly computed one.

    :return: ``(changed: bool, report: list[str])``
    """
    changed = False
    report = []
    for key in SCIENCE_KEYS:
        old = _normalise(existing_block.get(key))
        new = _normalise(new_block.get(key))
        if old != new:
            changed = True
            report.append(f"  {key}: DIFFERS")
        else:
            report.append(f"  {key}: unchanged")
    return changed, report


def _t_z_representation_version():
    """
    ``T_Z_REPRESENTATION_VERSION`` does not exist until prompt 03. Returns
    ``(version_or_None, note_if_none)``.
    """
    try:
        from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
    except Exception as exc:  # pragma: no cover - defensive only
        return None, f"import of QCD_Cosmology failed: {exc!r}"
    version = getattr(QCD_Cosmology, "T_Z_REPRESENTATION_VERSION", None)
    if version is None:
        return None, "not yet introduced (prompts/qcd-background-audit prompt 03)"
    return version, None


# =============================================================================================
# driver
# =============================================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute the QCD block and report whether it would change, without writing.",
    )
    args = parser.parse_args()

    t_start = time.perf_counter()

    with open(REFERENCE_DATA_PATH, "r") as f:
        payload = json.load(f)
    existing_block = payload["models"]["QCDModel"]

    print("** building the QCD stand-in on the production grid")
    # QCDModel is built on the SAME grid as LambdaCDMModel in generate_references.py's main():
    # both are keyed off LambdaCDM's horizon-exit z_init, not a QCD-specific one. Reproduced here
    # so this script's grid is bit-identical to the shipped file's.
    lam = LambdaCDMModel()
    z_init = horizon_exit_z(
        lam.cosmology, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid = production_source_grid(z_init)
    z_nodes = np.array([z.z for z in grid], dtype=float)

    t0 = time.perf_counter()
    qcd = QCDModel(grid)
    build_time = time.perf_counter() - t0
    print(f"   QCDModel: compute_background + splines took {build_time:.3f} s")

    checkpoints = R.select_checkpoints(z_nodes)
    k_anchor = {
        k: horizon_exit_z(qcd.cosmology, k, RHO_ANCHOR_EFOLDS_SUBH)
        for k in REFERENCE_K_VALUES
    }

    t0 = time.perf_counter()
    new_block, floors = build_qcd_block(qcd, z_nodes, checkpoints, k_anchor)
    generation_time = time.perf_counter() - t0
    print(f"   reference quadrature took {generation_time:.3f} s")

    new_block["reference_floor"] = floors
    new_block["grid"] = {
        "z_init": float(z_nodes[0]),
        "z_end": PRODUCTION_Z_END,
        "samples_per_log10z": PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
        "num_nodes": int(len(z_nodes)),
        "superhorizon_efolds": PRODUCTION_SUPERHORIZON_EFOLDS,
        "largest_k_inv_Mpc": PRODUCTION_LARGEST_K_INV_MPC,
    }
    new_block["z_top"] = float(z_nodes[0])
    new_block["checkpoints"] = [
        {"index": int(j), "z": float(z_nodes[j])} for j in checkpoints
    ]
    # carried through unchanged unless the science moved (see below)
    new_block["method"] = existing_block.get("method", "")
    new_block["build_time_seconds"] = build_time
    new_block["generation_time_seconds"] = generation_time

    changed, report = science_diff(existing_block, new_block)

    print("\n** QCD block science-content comparison against the shipped file:")
    for line in report:
        print(line)

    total_elapsed = time.perf_counter() - t_start

    if not changed:
        print(
            f"\nNo change: the QCD block's science content ({len(SCIENCE_KEYS)} keys) is "
            f"bit-identical to the shipped file. Nothing written (not even the provenance "
            f"fields). Total wall time {total_elapsed:.3f} s "
            f"(build {build_time:.3f} s, reference quadrature {generation_time:.3f} s)."
        )
        return 0

    # the science moved: this is what prompts 04-07 will see
    version, version_note = _t_z_representation_version()
    version_text = (
        f"T_Z_REPRESENTATION_VERSION={version}" if version is not None else version_note
    )
    provenance = (
        f" | regenerated by docs/qcd-background-audit/generate_qcd_references.py on "
        f"{date.today().isoformat()}, {version_text}"
    )
    new_block["method"] = existing_block.get("method", "") + provenance

    if args.dry_run:
        print(
            f"\nCHANGE DETECTED in the QCD block's science content -- see the differing keys "
            f"above. Re-run without --dry-run to write it. (would append to method: "
            f"{provenance.strip(' |')!r})"
        )
        return 1

    payload["models"]["QCDModel"] = new_block
    payload["generated"] = date.today().isoformat()
    with open(REFERENCE_DATA_PATH, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=False)
    print(
        f"\nWrote {REFERENCE_DATA_PATH}: QCD block regenerated in {total_elapsed:.3f} s total."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
