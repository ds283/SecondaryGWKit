"""
Does one absolute tolerance cover the production k-grid for the transfer function's numeric run?
(``prompts/GkTk-remedial/`` prompt 17.)

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py

Prompt 12 gave ``TkNumericIntegration`` its own absolute tolerance,
``config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13``, on a measurement at **one**
wavenumber (k = 1e6/Mpc) on the exact-radiation control: ``max dT/env`` falls 9.928e-6 -> 2.534e-6
for +14.3 % right-hand-side evaluations, meeting README section 6's <= 3e-6 row. A sweep of the
same control then found an isolated excursion of 2.56e-4 of the envelope surviving near x ~ 10.8 at
k = 3e8/Mpc, which ``atol = 1e-16`` removes at *fewer* evaluations
(``[12-tk-numeric-atol-largest-k-excursion]``). This script measures the whole production k-grid,
on both production backgrounds as well as the control, and says whether one constant covers it.

**It changes nothing.** No production module is touched and the constant stays at 1e-13 whatever
comes out; the recommendation in the generated document is for the user to accept or reject.

What is measured, for ``RadiationModel``, ``LambdaCDMModel`` and ``QCDModel`` (prompt 01's
stand-ins) at the 50 production source wavenumbers ``np.logspace(log10(1e5), log10(3e8), 50)``:

1. a **reference** run of the same integrator per (model, k) at ``(atol, rtol) = (1e-18, 1e-12)``,
   with a convergence check against ``(1e-19, 1e-13)`` -- the check is the part that can invalidate
   everything else, so it is reported per k, not once;
2. the three candidates ``atol`` in {1e-10, 1e-13, 1e-16} at fixed ``rtol = 1e-8``, scored against
   that reference by README section 6's **envelope-relative** error, dividing by the local
   Liouville-Green envelope ``hypot(T, T'/omega)`` with ``omega = sqrt(Tk_omegaEff_sq)`` of the
   *reference* run, skipping samples where ``omega^2 <= 0`` (the super-horizon part of the grid);
3. per (model, k, atol): the maximum and where it falls, the **second-largest** per-sample value,
   the **median**, and the right-hand-side evaluation count. The median and second-largest are what
   separate an isolated spike from a genuinely raised error level, which is the open question;
4. the initial-condition floor: at reference tolerance, the production ``T = 1, T' = 0`` against
   the super-horizon series ``T = 1 - x^2/10`` (README section 2 (d) puts the floor at 2.5e-6 of
   the envelope). An error at that floor is not fixable by any ``atol``;
5. on ``RadiationModel`` only, everything again against the exact
   ``T = 3(sin x - x cos x)/x^3`` -- a second column that checks the reference construction itself
   against a true oracle, and the column prompt 12's numbers are in.

**Why the comparison is against a converged run of the same integrator, and what that costs.**
There is no closed-form T on ``LambdaCDMModel`` or ``QCDModel``, so no oracle exists. Measuring a
candidate against a reference that shares its initial data removes the initial-condition error as
common mode: the sweep's ``dT/env`` is therefore **solver error alone**, which is the quantity
``atol`` controls, and item 4 supplies the floor it has to be read against. On the radiation
control the same run is also scored against the exact T, which is how prompt 12's two figures are
reproduced.

The script needs no Ray and no datastore: ``numeric_with_phase_cut`` is called through its
undecorated ``_function`` with the stand-ins of ``ComputeTargets/tests/wkb_reference.py``, the
pattern of ``docs/gktk-remedial/residual_convergence.py`` (prompt 02).

Output is a markdown fragment on stdout (the tables of
``docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md``, which is assembled from it by hand because that
document is additive: a later re-run adds a section rather than rewriting this one) plus a progress
log on stderr.

**Second entry point, added by prompt 18** (this file is additive too; nothing above the
``prompt 18`` banner near the bottom was changed):

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --break-points

emits §9 of the same document -- the same reference-convergence test re-run after
``numeric_with_phase_cut`` learned to split its integration at the cosmology's declared
discontinuities, extended to the Green's-function sector, with the cost and the movement of the
production answer on ``QCDModel``. See :func:`main_break_points`.

**Third entry point, added by prompt 19** (additive again; nothing above the ``prompt 19`` banner
at the bottom was changed except that ``run``, ``run_gk`` and ``reproduce_control`` gained a
``break_point_kind`` argument defaulting to what they did before):

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --per-sector

emits §10 -- the same test again once the break-point policy became a *per-caller* choice, with
``TkNumericIntegration`` asking for every declared break point and ``GkNumericIntegration`` for the
jumps alone. See :func:`main_per_sector`.
"""

import platform
import sys
import time
from datetime import date
from math import cos, fabs, log10, sin, sqrt
from statistics import median

import numpy as np
import scipy

from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.tests.convergence_reference import (
    UNITS,
    V0_PER_K_GRID,
    sector_errors,
    summarise,
    tk_geometry,
    tk_run as run,
    x_local,
)
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SMALLEST_K_INV_MPC,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    QCDModel,
    RadiationModel,
    envelope_relative_error,
    horizon_exit_z,
    production_source_grid,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from ComputeTargets.BackgroundModel import (
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
)

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

# main.py:3090-3099 -- NUMBER_SOURCE_K_VALUES = 50, and TkNumericIntegration is one object per k
NUMBER_SOURCE_K_VALUES = 50
PRODUCTION_K_GRID = np.logspace(
    log10(PRODUCTION_SMALLEST_K_INV_MPC),
    log10(PRODUCTION_LARGEST_K_INV_MPC),
    NUMBER_SOURCE_K_VALUES,
)

# prompt 17 section 2.3: the three candidates, at the production relative tolerance
CANDIDATE_ATOL = (1e-10, 1e-13, 1e-16)
PRODUCTION_RTOL = 1e-8

# prompt 17 section 2.1: the reference, and the tightening used to demonstrate it is converged.
# NOTE scipy clamps rtol at 2.22e-14 (`rk.py`), so 1e-13 is the last tightening that is really
# applied; a further decade of rtol would be silently ignored, which is why the check is one
# decade and not two.
REFERENCE_ATOL = 1e-18
REFERENCE_RTOL = 1e-12
TIGHTENED_ATOL = 1e-19
TIGHTENED_RTOL = 1e-13

# README section 6's row, and README section 2 (d)'s floor
TARGET_ERROR = 3.0e-6
INITIAL_CONDITION_FLOOR = 2.5e-6

# prompt 17 section 2.4: the two figures that must reproduce before the sweep means anything
CONTROL_K_MPC = 1.0e6
CONTROL_K_EXCURSION_MPC = 3.0e8
CONTROL_EXPECTED = 2.534e-6
CONTROL_EXCURSION_EXPECTED = 2.56e-4
CONTROL_EXCURSION_X = 10.8
CONTROL_TOLERANCE = 0.03  # relative, "a couple of significant figures"

# a maximum is called a *spike* rather than a raised level when it stands this far above the
# median of the same run
SPIKE_RATIO = 10.0


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# stand-ins, geometry, solves and error sampling
#
# All of it moved to ComputeTargets/tests/convergence_reference.py at prompt 01 of
# prompts/tolerance-convergence, so that the second sector, the second campaign and prompt 04's
# integer orders share one implementation instead of copying this one. Nothing about the method
# changed: the names below are that module's, and this script's published figures are unmoved,
# which is that prompt's own acceptance test.
#
# The `_Wavenumber` / `_KExit` / `_Proxy` stand-ins are the pattern of
# ComputeTargets/tests/test_tk_numeric_atol.py (prompt 12) and live there now; `run`, `run_gk`,
# `x_local`, `summarise` and `sector_errors` are imported above and `geometry` / `gk_geometry`
# below name the source-grid generation they are built on.
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# the production geometry of a TkNumericIntegration work item, and the runs on it
# ---------------------------------------------------------------------------------------------


def geometry(cosmology, k_inv_Mpc: float) -> dict:
    """
    ``main.py``'s ``build_Tk_numeric_work`` geometry for one wavenumber, **on the version-0 source
    grid**: one bare ``logspace`` per wavenumber from five e-folds outside the horizon, truncated
    below at ``0.85 z_e6``, with the ``(z_e3, z_e6)`` stop window.

    Every figure this script publishes is scored on that grid, and it is not the one ``main.py``
    builds -- production has been at ``SOURCE_GRID_CONSTRUCTION_VERSION = 2`` since prompt 15 of
    ``prompts/qcd-background-audit``. The generation is named here rather than assumed
    (``prompts/tolerance-convergence`` README §5 rule 6, ``[00-three-production-grid-reproductions]``):
    the figures stay comparable with the ones already published, and a reader can see which grid
    they belong to.

    ``cosmology`` is anything with ``Hubble(z)`` and ``H0`` -- the ``RadiationModel`` stand-in
    itself, or the real cosmology behind ``LambdaCDMModel`` / ``QCDModel``.
    """
    return tk_geometry(cosmology, k_inv_Mpc, V0_PER_K_GRID.build(cosmology))


def sample_errors(model, k_inv_Mpc: float, geo: dict, candidate: dict, reference: dict):
    """
    Envelope-relative error of ``candidate`` against ``reference``, sample by sample: the
    transfer-function sector of ``convergence_reference.sector_errors``.
    """
    return sector_errors("Tk", model, k_inv_Mpc, geo, candidate, reference)


# ---------------------------------------------------------------------------------------------
# the phase variable, the envelope, and the errors
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# the exact radiation transfer function (the oracle, on the control only)
# ---------------------------------------------------------------------------------------------


def T_exact(x: float) -> float:
    """``T = 3(sin x - x cos x)/x^3`` (review section 12.4)."""
    return 3.0 * (sin(x) - x * cos(x)) / (x * x * x)


def T_envelope_exact(x: float) -> float:
    """Its Liouville-Green envelope, ``3 sqrt(1+x^2)/x^3``."""
    return 3.0 * sqrt(1.0 + x * x) / (x * x * x)


def dT_dx(x: float) -> float:
    """``d/dx`` of ``T_exact``."""
    return 3.0 * (x * x * sin(x) - 3.0 * sin(x) + 3.0 * x * cos(x)) / (x**4)


def exact_errors(model, k_inv_Mpc: float, geo: dict, payload: dict):
    """
    Envelope-relative error of a ``RadiationModel`` run against the exact T, sample by sample.

    This is prompt 12's measure, and it includes the initial-condition error, which
    :func:`sample_errors` removes as common mode. Both are reported on the control so that the two
    can be compared.
    """
    out = []
    for z, value in zip(geo["grid"], payload["value_sample"]):
        x = x_local(model, k_inv_Mpc, z.z)
        out.append(
            (envelope_relative_error(value, T_exact(x), T_envelope_exact(x)), z.z, x)
        )
    return out


def exact_initial_data(model, k_inv_Mpc: float, z: float):
    """Exact ``(T, dT/dz)`` at ``z`` in exact radiation; ``dx/dz = -k c_s/H``."""
    x = x_local(model, k_inv_Mpc, z)
    dx_dz = (
        -k_inv_Mpc * sqrt(model.functions.wPerturbations(z)) / model.functions.Hubble(z)
    )
    return T_exact(x), dT_dx(x) * dx_dz


def series_initial_data(model, k_inv_Mpc: float, z: float):
    """
    Super-horizon series initial data, ``T = 1 - x^2/10`` with
    ``dT/dz = -(x/5) dx/dz = (x/5) k c_s/H`` (review section 12.5, README section 2 (d)).

    It is the leading correction to the production ``T = 1, T' = 0`` in *any* radiation-dominated
    background, which is where every k on the production grid starts its grid; using it to
    *measure* the initial-condition floor on a real background does not commit anything to
    changing the initial data, which is ``[00-tk-superhorizon-ic-series]`` and out of scope.
    """
    x = x_local(model, k_inv_Mpc, z)
    dx_dz = (
        -k_inv_Mpc * sqrt(model.functions.wPerturbations(z)) / model.functions.Hubble(z)
    )
    return 1.0 - x * x / 10.0, -(x / 5.0) * dx_dz


# ---------------------------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------------------------


def emit(line: str = "") -> None:
    print(line, flush=True)


def table(header, rows) -> None:
    emit("| " + " | ".join(header) + " |")
    emit("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        emit("| " + " | ".join(row) + " |")
    emit()


def g(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}g}"


# ---------------------------------------------------------------------------------------------
# the control (prompt 17 section 2.4)
# ---------------------------------------------------------------------------------------------


def reproduce_control(
    radiation, break_point_kind: str = BREAK_POINT_DISCONTINUITY
) -> dict:
    """
    Prompt 12's two figures on ``RadiationModel``, against the exact T: 2.534e-6 at k = 1e6 with
    ``atol = 1e-13``, and the 2.56e-4 excursion near x ~ 10.8 at k = 3e8 with the same tolerance.

    If either misses by more than :data:`CONTROL_TOLERANCE` the harness differs from prompt 12's
    and nothing downstream is comparable, so the script stops.

    ``break_point_kind`` lets §10 run the control under the transfer function's *production*
    policy; ``RadiationModel`` declares nothing, so the two policies must give the same two
    figures and the same two evaluation counts, and §10 checks that they do.
    """
    results = {}
    for label, k, expected, expected_x in (
        ("k=1e6", CONTROL_K_MPC, CONTROL_EXPECTED, None),
        (
            "k=3e8",
            CONTROL_K_EXCURSION_MPC,
            CONTROL_EXCURSION_EXPECTED,
            CONTROL_EXCURSION_X,
        ),
    ):
        geo = geometry(radiation, k)
        payload = run(
            radiation, k, geo, 1e-13, PRODUCTION_RTOL, break_point_kind=break_point_kind
        )
        summary = summarise(exact_errors(radiation, k, geo, payload))
        summary["evaluations"] = payload["data"].RHS_evaluations
        summary["expected"] = expected
        summary["expected_x"] = expected_x
        summary["relative_miss"] = fabs(summary["max"] / expected - 1.0)
        results[label] = summary

        if summary["relative_miss"] > CONTROL_TOLERANCE:
            raise RuntimeError(
                f"control {label} did not reproduce prompt 12: measured {summary['max']:.4g}, "
                f"expected {expected:.4g} ({summary['relative_miss']:.1%} off). The harness "
                "differs from prompt 12's and nothing downstream is comparable."
            )
        if expected_x is not None and fabs(summary["max_x"] / expected_x - 1.0) > 0.05:
            raise RuntimeError(
                f"control {label} reproduced the size of the excursion but not its position: "
                f"x={summary['max_x']:.4g}, expected ~{expected_x:.4g}"
            )

    return results


# ---------------------------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------------------------


def sweep_model(name: str, model, cosmology, is_radiation: bool) -> dict:
    """Every k of the production grid, for one model."""
    rows = []
    t_model = time.perf_counter()

    for index, k in enumerate(PRODUCTION_K_GRID):
        k = float(k)
        geo = geometry(cosmology, k)

        reference = run(model, k, geo, REFERENCE_ATOL, REFERENCE_RTOL)
        tightened = run(model, k, geo, TIGHTENED_ATOL, TIGHTENED_RTOL)
        reference_drift = summarise(sample_errors(model, k, geo, tightened, reference))

        series = run(
            model,
            k,
            geo,
            REFERENCE_ATOL,
            REFERENCE_RTOL,
            ic=series_initial_data(model, k, geo["grid"].max.z),
        )
        ic_floor = summarise(sample_errors(model, k, geo, series, reference))

        entry = {
            "k": k,
            "x_init": x_local(model, k, geo["grid"].max.z),
            "z_init": geo["grid"].max.z,
            "samples_requested": len(geo["grid"]),
            "samples_returned": len(reference["value_sample"]),
            "reference_evaluations": reference["data"].RHS_evaluations,
            "reference_drift": reference_drift,
            "ic_floor": ic_floor,
            "candidates": {},
        }

        if is_radiation:
            entry["reference_vs_exact"] = summarise(
                exact_errors(model, k, geo, reference)
            )
            exact_run = run(
                model,
                k,
                geo,
                REFERENCE_ATOL,
                REFERENCE_RTOL,
                ic=exact_initial_data(model, k, geo["grid"].max.z),
            )
            entry["reference_exact_ic_vs_exact"] = summarise(
                exact_errors(model, k, geo, exact_run)
            )

        for atol in CANDIDATE_ATOL:
            payload = run(model, k, geo, atol, PRODUCTION_RTOL)
            summary = summarise(sample_errors(model, k, geo, payload, reference))
            summary["evaluations"] = payload["data"].RHS_evaluations
            if is_radiation:
                summary["vs_exact"] = summarise(exact_errors(model, k, geo, payload))[
                    "max"
                ]
            entry["candidates"][atol] = summary

        rows.append(entry)
        log(
            f"   {name} [{index + 1:2d}/{len(PRODUCTION_K_GRID)}] k={k:.4g}: "
            f"ref drift {reference_drift['max']:.2e}, "
            + ", ".join(
                f"{atol:.0e}->{entry['candidates'][atol]['max']:.2e}"
                for atol in CANDIDATE_ATOL
            )
        )

    log(f"   {name} done in {time.perf_counter() - t_model:.1f} s")
    return {"name": name, "rows": rows}


# ---------------------------------------------------------------------------------------------
# two follow-up diagnostics, at the wavenumber where the shipped tolerance is worst
#
# Neither is in the prompt's method. Both are here because the sweep's answer to "does 1e-16 fix
# it" is *no*, and a recommendation to leave the constant alone has to say what the excursions
# are, if they are not an absolute-tolerance phenomenon. They change nothing and cost ~15 solves.
# ---------------------------------------------------------------------------------------------

RTOL_LADDER = (1e-8, 1e-9, 1e-10)
K_PERTURBATIONS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3)


def worst_k_for(result, atol: float = 1e-13) -> float:
    return max(result["rows"], key=lambda row: row["candidates"][atol]["max"])["k"]


def rtol_ladder(name, model, cosmology, k: float) -> dict:
    """
    Hold ``atol = 1e-13`` and tighten the *relative* tolerance, at the wavenumber where the
    shipped tolerance is worst. If the excursion is an absolute-tolerance artefact this does
    nothing; if it is step selection, it goes.
    """
    geo = geometry(cosmology, k)
    reference = run(model, k, geo, REFERENCE_ATOL, REFERENCE_RTOL)
    out = []
    for rtol in RTOL_LADDER:
        payload = run(model, k, geo, 1e-13, rtol)
        summary = summarise(sample_errors(model, k, geo, payload, reference))
        summary["rtol"] = rtol
        summary["evaluations"] = payload["data"].RHS_evaluations
        out.append(summary)
    log(
        f"   {name} rtol ladder at k={k:.5g}: "
        + ", ".join(f"{s['max']:.2e}" for s in out)
    )
    return {"name": name, "k": k, "ladder": out}


def k_sensitivity(name, model, cosmology, k: float) -> dict:
    """
    The same wavenumber perturbed by 1e-6 to 1e-3 relative, at the shipped tolerance. A physical
    feature of the solution stays put under a 1e-6 change in k; a step-sequence accident does not.
    """
    out = []
    for delta in K_PERTURBATIONS:
        k_perturbed = k * (1.0 + delta)
        geo = geometry(cosmology, k_perturbed)
        reference = run(model, k_perturbed, geo, REFERENCE_ATOL, REFERENCE_RTOL)
        payload = run(model, k_perturbed, geo, 1e-13, PRODUCTION_RTOL)
        summary = summarise(sample_errors(model, k_perturbed, geo, payload, reference))
        summary["delta"] = delta
        summary["k"] = k_perturbed
        summary["evaluations"] = payload["data"].RHS_evaluations
        out.append(summary)
    log(
        f"   {name} k-sensitivity at k={k:.5g}: "
        + ", ".join(f"{s['max']:.2e}" for s in out)
    )
    return {"name": name, "k": k, "perturbations": out}


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def report_reference_convergence(results) -> None:
    emit("### Reference convergence")
    emit()
    rows = []
    for result in results:
        drifts = [row["reference_drift"] for row in result["rows"]]
        worst_index = max(range(len(drifts)), key=lambda i: drifts[i]["max"])
        worst = result["rows"][worst_index]
        candidate_floor = min(
            row["candidates"][atol]["max"]
            for row in result["rows"]
            for atol in CANDIDATE_ATOL
        )
        rows.append(
            [
                result["name"],
                g(max(d["max"] for d in drifts)),
                f"{worst['k']:.4g}",
                g(worst["reference_drift"]["median"]),
                g(median(d["max"] for d in drifts)),
                g(candidate_floor),
                (
                    "yes"
                    if max(d["max"] for d in drifts) * 10.0 <= candidate_floor
                    else "**no**"
                ),
            ]
        )
    table(
        [
            "model",
            "worst reference drift",
            "at k [1/Mpc]",
            "its median",
            "median drift over the grid",
            "smallest candidate error reported",
            "drift <= 1/10 of it?",
        ],
        rows,
    )


def report_control(control) -> None:
    emit("### The prompt 12 control")
    emit()
    rows = []
    for label, summary in control.items():
        rows.append(
            [
                label,
                g(summary["expected"]),
                g(summary["max"], 4),
                f"{summary['relative_miss']:.2%}",
                g(summary["max_x"], 4),
                "--" if summary["expected_x"] is None else g(summary["expected_x"], 3),
                str(summary["evaluations"]),
            ]
        )
    table(
        [
            "control",
            "prompt 12",
            "measured max dT/env",
            "miss",
            "at x",
            "prompt 12 x",
            "RHS evals",
        ],
        rows,
    )


def report_model(result) -> None:
    name = result["name"]
    rows = result["rows"]
    emit(f"### {name}")
    emit()

    offenders = {
        atol: [row for row in rows if row["candidates"][atol]["max"] > TARGET_ERROR]
        for atol in CANDIDATE_ATOL
    }
    summary_rows = []
    for atol in CANDIDATE_ATOL:
        worst = max(rows, key=lambda row: row["candidates"][atol]["max"])
        summary_rows.append(
            [
                f"{atol:.0e}",
                g(worst["candidates"][atol]["max"]),
                f"{worst['k']:.4g}",
                g(worst["candidates"][atol]["max_x"], 4),
                g(median(row["candidates"][atol]["max"] for row in rows)),
                str(len(offenders[atol])),
                str(sum(row["candidates"][atol]["evaluations"] for row in rows)),
            ]
        )
    table(
        [
            "atol",
            "worst max dT/env",
            "at k [1/Mpc]",
            "at x",
            "median over the 50 k of the per-k max",
            "k with max > 3e-6",
            "total RHS evals over the grid",
        ],
        summary_rows,
    )

    shapes = {
        atol: [
            ("level" if row["candidates"][atol]["median"] > TARGET_ERROR else "spike")
            for row in offenders[atol]
        ]
        for atol in CANDIDATE_ATOL
    }
    emit(
        "Shape of the offending runs — *level* when the median is itself above "
        "$3\\times10^{-6}$, *spike* when only the maximum is:"
    )
    emit()
    table(
        ["atol", "k above 3e-6", "of those, level", "of those, spike"],
        [
            [
                f"{atol:.0e}",
                str(len(offenders[atol])),
                str(shapes[atol].count("level")),
                str(shapes[atol].count("spike")),
            ]
            for atol in CANDIDATE_ATOL
        ],
    )

    emit(
        "Every wavenumber of the production grid, as "
        "**max / 2nd-largest / median / last sample** of $\\delta T/\\mathrm{env}$. "
        "`x@max` is where the maximum fell; the grid runs from $x_i\\approx0.0039$ to "
        "$x\\approx275$."
    )
    emit()
    detail_rows = []
    for row in rows:
        entry = [f"{row['k']:.4g}"]
        for atol in CANDIDATE_ATOL:
            summary = row["candidates"][atol]
            entry.append(
                f"{g(summary['max'], 2)} / {g(summary['second'], 2)} / "
                f"{g(summary['median'], 2)} / {g(summary['terminal'], 2)}"
                f" @{g(summary['max_x'], 3)}"
            )
        entry.append(g(row["reference_drift"]["max"], 2))
        entry.append(g(row["ic_floor"]["max"], 3))
        detail_rows.append(entry)
    table(
        ["k [1/Mpc]"]
        + [f"atol {atol:.0e} (x@max)" for atol in CANDIDATE_ATOL]
        + ["ref drift", "IC floor"],
        detail_rows,
    )

    ic = [row["ic_floor"]["max"] for row in rows]
    emit(
        f"Initial-condition floor (production $T=1,T'=0$ against the series $1-x^2/10$, both at "
        f"reference tolerance): min {g(min(ic))}, median {g(median(ic))}, max {g(max(ic))} of the "
        f"envelope, at $x_i$ = {g(rows[0]['x_init'], 4)}--{g(rows[-1]['x_init'], 4)}."
    )
    emit()

    if "reference_vs_exact" in rows[0]:
        emit("The oracle column (exact radiation only):")
        emit()
        oracle_rows = []
        for row in rows[:: max(1, len(rows) // 8)]:
            oracle_rows.append(
                [
                    f"{row['k']:.4g}",
                    g(row["reference_vs_exact"]["max"]),
                    g(row["reference_exact_ic_vs_exact"]["max"]),
                    g(row["ic_floor"]["max"]),
                    g(row["candidates"][1e-13]["vs_exact"]),
                    g(row["candidates"][1e-13]["max"]),
                ]
            )
        table(
            [
                "k [1/Mpc]",
                "reference vs exact T",
                "reference, exact IC, vs exact T",
                "IC floor (series measure)",
                "atol 1e-13 vs exact T",
                "atol 1e-13 vs reference",
            ],
            oracle_rows,
        )


def report_cost(results) -> None:
    emit("### Cost")
    emit()
    rows = []
    for result in results:
        entry = [result["name"]]
        for atol in CANDIDATE_ATOL:
            entry.append(
                str(
                    sum(
                        row["candidates"][atol]["evaluations"] for row in result["rows"]
                    )
                )
            )
        baseline = sum(
            row["candidates"][1e-10]["evaluations"] for row in result["rows"]
        )
        for atol in (1e-13, 1e-16):
            total = sum(
                row["candidates"][atol]["evaluations"] for row in result["rows"]
            )
            entry.append(f"{total / baseline - 1.0:+.1%}")
        cheaper = sum(
            1
            for row in result["rows"]
            if row["candidates"][1e-16]["evaluations"]
            < row["candidates"][1e-13]["evaluations"]
        )
        entry.append(f"{cheaper}/{len(result['rows'])}")
        rows.append(entry)
    table(
        [
            "model",
            "RHS evals, atol 1e-10",
            "1e-13",
            "1e-16",
            "1e-13 vs 1e-10",
            "1e-16 vs 1e-10",
            "k where 1e-16 is cheaper than 1e-13",
        ],
        rows,
    )


def report_diagnostics(ladders, sensitivities) -> None:
    emit("### Is it the absolute tolerance at all?")
    emit()
    emit(
        "**(a) Hold `atol = 1e-13` and tighten `rtol`**, at the wavenumber where the shipped "
        "tolerance is worst on each model:"
    )
    emit()
    table(
        ["model", "k [1/Mpc]"]
        + [f"rtol {rtol:.0e}: max (evals)" for rtol in RTOL_LADDER],
        [
            [entry["name"], f"{entry['k']:.5g}"]
            + [
                f"{g(step['max'])} @x={g(step['max_x'], 4)} ({step['evaluations']})"
                for step in entry["ladder"]
            ]
            for entry in ladders
        ],
    )
    emit(
        "**(b) Perturb $k$ at the shipped tolerance** (`atol = 1e-13`, `rtol = 1e-8`), same "
        "wavenumber. A feature of the solution survives a $10^{-6}$ change in $k$; an accident "
        "of step selection does not:"
    )
    emit()
    table(
        ["model", "k [1/Mpc]"]
        + [
            ("baseline" if delta == 0.0 else f"k(1+{delta:.0e})")
            for delta in K_PERTURBATIONS
        ],
        [
            [entry["name"], f"{entry['k']:.5g}"]
            + [
                f"{g(step['max'])} @x={g(step['max_x'], 4)}"
                for step in entry["perturbations"]
            ]
            for entry in sensitivities
        ],
    )


def main() -> None:
    t_start = time.perf_counter()

    log("** building stand-in models")
    radiation = RadiationModel()
    lambda_cdm = LambdaCDMModel()

    # the QCD stand-in is built on the full production source grid of *its own* cosmology: the
    # largest k exits the horizon earlier on QCD_Cosmology than on LambdaCDM, and a grid cut from
    # the LambdaCDM crossing leaves the k = 3e8 run's first samples outside the splines' range
    qcd_cosmology = QCD_Cosmology(
        store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
    )
    t0 = time.perf_counter()
    qcd = QCDModel(
        production_source_grid(
            horizon_exit_z(
                qcd_cosmology,
                PRODUCTION_LARGEST_K_INV_MPC,
                -float(PRODUCTION_SUPERHORIZON_EFOLDS),
            )
        ),
        cosmology=qcd_cosmology,
    )
    qcd_build_seconds = time.perf_counter() - t0
    log(f"   QCDModel built in {qcd_build_seconds:.2f} s")

    log("** reproducing prompt 12's control")
    control = reproduce_control(radiation)
    for label, summary in control.items():
        log(
            f"   {label}: max dT/env = {summary['max']:.4g} at x = {summary['max_x']:.4g} "
            f"({summary['relative_miss']:.2%} from prompt 12's {summary['expected']:.4g})"
        )

    log("** sweeping")
    models = (
        ("RadiationModel", radiation, radiation, True),
        ("LambdaCDMModel", lambda_cdm, lambda_cdm.cosmology, False),
        ("QCDModel", qcd, qcd_cosmology, False),
    )
    results = [sweep_model(*entry) for entry in models]

    log("** follow-up diagnostics at the worst k of each model")
    ladders = []
    sensitivities = []
    for (name, model, cosmology, _), result in zip(models, results):
        k = worst_k_for(result)
        ladders.append(rtol_ladder(name, model, cosmology, k))
        sensitivities.append(k_sensitivity(name, model, cosmology, k))

    elapsed = time.perf_counter() - t_start

    emit(f"<!-- generated {date.today().isoformat()} by")
    emit(
        "     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py"
    )
    emit(
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_control(control)
    report_reference_convergence(results)
    report_cost(results)
    report_diagnostics(ladders, sensitivities)
    for result in results:
        report_model(result)

    emit(
        f"*Runtime {elapsed:.0f} s "
        f"(QCD stand-in build {qcd_build_seconds:.1f} s of it); "
        f"{len(PRODUCTION_K_GRID)} wavenumbers x 3 models x 6-7 solves each, "
        f"plus {len(RTOL_LADDER) + 2 * len(K_PERTURBATIONS) + 1} per model "
        f"for the diagnostics.*"
    )

    log(f"** total runtime {elapsed:.1f} s")


# =============================================================================================
# prompt 18: does splitting the ODE at the cosmology's declared discontinuities fix the
# reference convergence that §4 above could not demonstrate on QCDModel?
#
# Run with:
#
#     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --break-points
#
# This is an *additive* second entry point. It reuses everything above -- the stand-ins, the
# geometry, the envelope-relative error, the control -- and emits the markdown of §9 of
# TK-NUMERIC-ATOL-SWEEP.md. Nothing above is modified: §§1-8 were correct for the tree they were
# taken on and are not rewritten (prompt 17 §3).
#
# What it measures, per prompt 18 §3:
#
#   1. the §2.1 convergence test on QCDModel at all 50 wavenumbers, before and after the split;
#   2. the same test for G_k, on all three models, which has never been done;
#   3. a regression on the two smooth models, which declare no discontinuities and must therefore
#      reproduce §§3-4 exactly rather than nearly;
#   4. the cost, in right-hand-side evaluations, on QCD;
#   5. how far the split moves the *production* answer on QCD, in both sectors -- which is what
#      tells prompt 13 whether a QCD datastore built before the split is still usable.
# =============================================================================================

from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.tests.convergence_reference import gk_geometry as _gk_geometry
from ComputeTargets.tests.convergence_reference import gk_run as run_gk
from Quadrature.integrators.numeric_with_phase_cut import declared_discontinuities_in_z

# prompt 17 §2.1's criterion, quantified for QCD by the smallest candidate difference the sweep
# reports there (3.45e-7): the drift must be at most a tenth of it
ACCEPTANCE_DRIFT = 3.4e-8

# the four wavenumbers §4 above names as failing, as indices into PRODUCTION_K_GRID
NAMED_FAILURES = (31, 38, 39, 48)

# production tolerances for the "does anything else move" comparison (config/defaults.py, and
# main.py's tolerance objects): G_k keeps atol, T_k has its own since prompt 12
GK_PRODUCTION_ATOL = 1e-10
TK_PRODUCTION_ATOL = 1e-13


class _SmoothCosmology:
    """
    A cosmology that declares nothing, wrapping one that does.

    This is how the "before" column is measured *after* the change: the physics is untouched --
    every accessor is the real cosmology's, and the model's ``functions`` are the real ones -- but
    ``_cosmology_break_points`` finds no ``integration_break_points`` to call, so
    ``numeric_with_phase_cut`` takes its historic single-``solve_ivp`` path. Nothing is
    monkeypatched and no production module is touched.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name == "integration_break_points":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)


class _UnsplitModel:
    """A model view carrying the real ``functions`` but a cosmology that declares nothing."""

    def __init__(self, model, cosmology):
        self.name = f"{model.name} (unsplit)"
        self.functions = model.functions
        self.cosmology = _SmoothCosmology(cosmology)


def gk_geometry(cosmology, k_inv_Mpc: float) -> dict:
    """
    ``main.py``'s ``build_Gk_numeric_work`` geometry for one object, **on the version-0 source
    grid** (see :func:`geometry`): the source redshift five e-folds outside the horizon, the
    *response* grid -- ``winnow(12)`` of the source grid, which is what ``GkNumericIntegration``
    is sampled on -- cut to the source redshift above and to ``0.85 z_e6`` below, and the
    ``(z_e3, z_e6)`` stop window.

    ``GkNumericIntegration`` is one object per ``(k, z_source)``; one source redshift per k is
    taken here, the outermost, which is the longest and therefore the least favourable run.

    The construction is ``convergence_reference.gk_geometry``; ``run_gk`` is that module's
    ``gk_run``, imported above under its historic name.
    """
    return _gk_geometry(cosmology, k_inv_Mpc, V0_PER_K_GRID.build(cosmology))


SECTORS = {
    "Tk": {"geometry": geometry, "run": run, "omega_sq": Tk_omegaEff_sq},
    "Gk": {"geometry": gk_geometry, "run": run_gk, "omega_sq": Gk_omegaEff_sq},
}


def break_point_sweep(name: str, model, cosmology, sector: str, declares: bool) -> dict:
    """
    Per wavenumber: the reference-convergence drift with the split in force and, where the
    cosmology declares anything, with it suppressed; the segment count; the reference cost; and
    how far the split moves the answer at the production tolerance.

    ``declares`` says whether this model has anything to declare. When it does not, the split and
    unsplit paths are the *same code path*, so the unsplit columns are omitted rather than
    measured twice.
    """
    unsplit_model = _UnsplitModel(model, cosmology) if declares else None
    geometry_fn = SECTORS[sector]["geometry"]
    run_fn = SECTORS[sector]["run"]
    production_atol = TK_PRODUCTION_ATOL if sector == "Tk" else GK_PRODUCTION_ATOL

    rows = []
    t_model = time.perf_counter()
    for index, k in enumerate(PRODUCTION_K_GRID):
        k = float(k)
        geo = geometry_fn(cosmology, k)

        reference = run_fn(model, k, geo, REFERENCE_ATOL, REFERENCE_RTOL)
        tightened = run_fn(model, k, geo, TIGHTENED_ATOL, TIGHTENED_RTOL)
        drift = summarise(sector_errors(sector, model, k, geo, tightened, reference))

        production = run_fn(model, k, geo, production_atol, PRODUCTION_RTOL)

        entry = {
            "k": k,
            "segments": len(
                declared_discontinuities_in_z(
                    model, float(geo["grid"].min), geo["grid"].max.z
                )
            )
            + 1,
            "drift": drift,
            "reference_evaluations": reference["data"].RHS_evaluations,
            "production_evaluations": production["data"].RHS_evaluations,
            "production_vs_reference": summarise(
                sector_errors(sector, model, k, geo, production, reference)
            )["max"],
        }

        if declares:
            unsplit_reference = run_fn(
                unsplit_model, k, geo, REFERENCE_ATOL, REFERENCE_RTOL
            )
            unsplit_tightened = run_fn(
                unsplit_model, k, geo, TIGHTENED_ATOL, TIGHTENED_RTOL
            )
            entry["unsplit_drift"] = summarise(
                sector_errors(
                    sector, unsplit_model, k, geo, unsplit_tightened, unsplit_reference
                )
            )
            entry["unsplit_reference_evaluations"] = unsplit_reference[
                "data"
            ].RHS_evaluations

            unsplit_production = run_fn(
                unsplit_model, k, geo, production_atol, PRODUCTION_RTOL
            )
            entry["unsplit_production_evaluations"] = unsplit_production[
                "data"
            ].RHS_evaluations
            # how far the split moves the shipped answer, scored against the split reference
            entry["production_shift"] = summarise(
                sector_errors(sector, model, k, geo, unsplit_production, production)
            )

        rows.append(entry)
        log(
            f"   {sector} {name} [{index + 1:2d}/{len(PRODUCTION_K_GRID)}] k={k:.4g}: "
            f"{entry['segments']} segment(s), drift {drift['max']:.2e}"
            + (
                f" (unsplit {entry['unsplit_drift']['max']:.2e}, shift "
                f"{entry['production_shift']['max']:.2e})"
                if declares
                else ""
            )
        )

    log(f"   {sector} {name} done in {time.perf_counter() - t_model:.1f} s")
    return {"name": name, "sector": sector, "declares": declares, "rows": rows}


def report_break_point_convergence(results) -> None:
    emit("### 9.1 Is the reference converged now?")
    emit()
    rows = []
    for result in results:
        drifts = [row["drift"]["max"] for row in result["rows"]]
        worst_index = max(range(len(drifts)), key=lambda i: drifts[i])
        worst = result["rows"][worst_index]
        offenders = sum(1 for d in drifts if d > ACCEPTANCE_DRIFT)
        entry = [
            result["sector"],
            result["name"],
            (
                str(max(row["segments"] for row in result["rows"]))
                if result["declares"]
                else "1"
            ),
            g(max(drifts)),
            f"{worst['k']:.4g}",
            g(median(drifts)),
            str(offenders),
            "**yes**" if offenders == 0 else "**no**",
        ]
        if result["declares"]:
            unsplit = [row["unsplit_drift"]["max"] for row in result["rows"]]
            entry.insert(4, g(max(unsplit)))
        else:
            entry.insert(4, "--")
        rows.append(entry)
    table(
        [
            "sector",
            "model",
            "max segments",
            "worst drift, split",
            "worst drift, unsplit",
            "at k [1/Mpc]",
            "median drift",
            f"k above {ACCEPTANCE_DRIFT:.1g}",
            "acceptance met?",
        ],
        rows,
    )


def report_named_failures(results) -> None:
    emit("### 9.2 The four wavenumbers §4 names")
    emit()
    rows = []
    for result in results:
        if not result["declares"]:
            continue
        for index in NAMED_FAILURES:
            row = result["rows"][index]
            rows.append(
                [
                    result["sector"],
                    f"{row['k']:.4g}",
                    g(row["unsplit_drift"]["max"]),
                    g(row["drift"]["max"]),
                    f"{row['unsplit_drift']['max'] / max(row['drift']['max'], 1e-30):.0f}x",
                    "yes" if row["drift"]["max"] <= ACCEPTANCE_DRIFT else "**no**",
                ]
            )
    table(
        [
            "sector",
            "k [1/Mpc]",
            "drift before",
            "drift after",
            "improvement",
            f"<= {ACCEPTANCE_DRIFT:.1g}?",
        ],
        rows,
    )


def report_break_point_cost(results) -> None:
    emit("### 9.3 What the split costs, in right-hand-side evaluations")
    emit()
    rows = []
    for result in results:
        if not result["declares"]:
            continue
        before = sum(row["unsplit_production_evaluations"] for row in result["rows"])
        after = sum(row["production_evaluations"] for row in result["rows"])
        per_object_before = before / len(result["rows"])
        per_object_after = after / len(result["rows"])
        rows.append(
            [
                result["sector"],
                result["name"],
                f"{per_object_before:.0f}",
                f"{per_object_after:.0f}",
                str(before),
                str(after),
                f"{after / before - 1.0:+.2%}",
            ]
        )
    table(
        [
            "sector",
            "model",
            "per object, unsplit",
            "per object, split",
            "grid total, unsplit",
            "grid total, split",
            "change",
        ],
        rows,
    )


def report_production_shift(results) -> None:
    emit("### 9.4 How far the split moves the production answer on QCD")
    emit()
    rows = []
    for result in results:
        if not result["declares"]:
            continue
        shifts = [row["production_shift"]["max"] for row in result["rows"]]
        worst_index = max(range(len(shifts)), key=lambda i: shifts[i])
        worst = result["rows"][worst_index]
        rows.append(
            [
                result["sector"],
                result["name"],
                g(max(shifts)),
                f"{worst['k']:.4g}",
                g(median(shifts)),
                g(min(shifts)),
                g(median(row["production_vs_reference"] for row in result["rows"])),
            ]
        )
    table(
        [
            "sector",
            "model",
            "worst shift",
            "at k [1/Mpc]",
            "median shift",
            "smallest shift",
            "median solver error at production tolerance",
        ],
        rows,
    )


def report_break_point_detail(result) -> None:
    emit(f"#### {result['sector']}, {result['name']}")
    emit()
    header = ["k [1/Mpc]", "segments", "drift after"]
    if result["declares"]:
        header += ["drift before", "production shift", "RHS evals before -> after"]
    rows = []
    for row in result["rows"]:
        entry = [f"{row['k']:.4g}", str(row["segments"]), g(row["drift"]["max"], 2)]
        if result["declares"]:
            entry += [
                g(row["unsplit_drift"]["max"], 2),
                g(row["production_shift"]["max"], 2),
                f"{row['unsplit_production_evaluations']} -> {row['production_evaluations']}",
            ]
        rows.append(entry)
    table(header, rows)


# ---------------------------------------------------------------------------------------------
# two follow-up diagnostics, neither in prompt 18's method, both load-bearing for a decision it
# had to take. They cost ~60 solves.
#
#  (a) the standoff. numeric_with_phase_cut places its segment boundary BREAK_POINT_STANDOFF on
#      the near side of the declared discontinuity rather than exactly on it. Moving it to the
#      far side instead, or leaving it exactly on the crossing, shows why.
#  (b) the knots. Prompt 18 §4 permits splitting at the C2 spline knots as well only if the cost
#      is stated and the user is asked. This is the cost, and the accuracy it would buy.
# ---------------------------------------------------------------------------------------------

# the boundary placements compared in (a), as a relative displacement in z of the declared
# crossing: 0 is "exactly on it", positive is the near (higher-z) side, negative the far side
STANDOFF_PLACEMENTS = (0.0, 1e-12, -1e-12, 1e-9, -1e-9)

# every fifth wavenumber, for (b)'s production-tolerance cost
KNOT_COST_STRIDE = 5


class _PlacedBreakCosmology:
    """
    Reports the declared discontinuities displaced by a fixed relative amount in z, so that the
    effect of where the segment boundary is placed can be measured against placing it exactly on
    the crossing. ``kind="all"`` is passed through untouched, so the quadrature path is unaffected.
    """

    def __init__(self, inner, relative: float):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_relative", float(relative))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def integration_break_points(self, z_lo, z_hi, kind="all"):
        points = object.__getattribute__(self, "_inner").integration_break_points(
            z_lo, z_hi, kind=kind
        )
        relative = object.__getattribute__(self, "_relative")
        if kind == "all" or relative == 0.0 or len(points) == 0:
            return points
        return np.asarray(
            [np.log1p(np.expm1(float(u)) * (1.0 + relative)) for u in points]
        )


class _AllBreaksCosmology:
    """Reports *every* break point -- spline knots included -- to the ODE as well."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def integration_break_points(self, z_lo, z_hi, kind="all"):
        return object.__getattribute__(self, "_inner").integration_break_points(
            z_lo, z_hi, kind="all"
        )


class _ViewWith:
    """The model's real ``functions``, with a substituted cosmology."""

    def __init__(self, model, cosmology, suffix: str):
        self.name = f"{model.name} ({suffix})"
        self.functions = model.functions
        self.cosmology = cosmology


def _drift_at(model, cosmology, sector: str, k: float) -> dict:
    geometry_fn = SECTORS[sector]["geometry"]
    run_fn = SECTORS[sector]["run"]
    geo = geometry_fn(cosmology, k)
    reference = run_fn(model, k, geo, REFERENCE_ATOL, REFERENCE_RTOL)
    tightened = run_fn(model, k, geo, TIGHTENED_ATOL, TIGHTENED_RTOL)
    summary = summarise(sector_errors(sector, model, k, geo, tightened, reference))
    summary["evaluations"] = reference["data"].RHS_evaluations
    return summary


def standoff_experiment(model, cosmology, indices) -> list:
    """(a): the same Tk run with the segment boundary placed five different ways."""
    out = []
    for index in indices:
        k = float(PRODUCTION_K_GRID[index])
        row = {"k": k, "placements": []}
        for relative in STANDOFF_PLACEMENTS:
            view = _ViewWith(
                model, _PlacedBreakCosmology(cosmology, relative), f"{relative:+.0e}"
            )
            row["placements"].append(_drift_at(view, cosmology, "Tk", k)["max"])
        out.append(row)
        log(
            f"   standoff k={k:.5g}: "
            + ", ".join(f"{d:.2e}" for d in row["placements"])
        )
    return out


def knot_split_experiment(model, cosmology, offenders) -> dict:
    """(b): what splitting at the 404 C2 spline knots as well would buy, and what it would cost."""
    all_breaks = _ViewWith(model, _AllBreaksCosmology(cosmology), "all breaks")

    accuracy = []
    for index in offenders:
        k = float(PRODUCTION_K_GRID[index])
        jumps = _drift_at(model, cosmology, "Tk", k)
        every = _drift_at(all_breaks, cosmology, "Tk", k)
        accuracy.append(
            {
                "k": k,
                "jumps": jumps["max"],
                "all": every["max"],
                "jumps_evaluations": jumps["evaluations"],
                "all_evaluations": every["evaluations"],
            }
        )
        log(
            f"   knots k={k:.5g}: jumps {jumps['max']:.2e} ({jumps['evaluations']}), "
            f"all {every['max']:.2e} ({every['evaluations']})"
        )

    cost = []
    indices = range(0, len(PRODUCTION_K_GRID), KNOT_COST_STRIDE)
    for sector in ("Tk", "Gk"):
        geometry_fn = SECTORS[sector]["geometry"]
        run_fn = SECTORS[sector]["run"]
        production_atol = TK_PRODUCTION_ATOL if sector == "Tk" else GK_PRODUCTION_ATOL
        jumps_total = 0
        all_total = 0
        for index in indices:
            k = float(PRODUCTION_K_GRID[index])
            geo = geometry_fn(cosmology, k)
            jumps_total += run_fn(model, k, geo, production_atol, PRODUCTION_RTOL)[
                "data"
            ].RHS_evaluations
            all_total += run_fn(all_breaks, k, geo, production_atol, PRODUCTION_RTOL)[
                "data"
            ].RHS_evaluations
        cost.append(
            {
                "sector": sector,
                "k_count": len(list(indices)),
                "jumps": jumps_total,
                "all": all_total,
            }
        )
        log(
            f"   knots cost {sector}: jumps {jumps_total}, all {all_total} "
            f"({all_total / jumps_total - 1.0:+.1%})"
        )

    return {"accuracy": accuracy, "cost": cost}


def report_standoff(rows) -> None:
    emit("### 9.6 Where the segment boundary is placed")
    emit()
    emit(
        "The same `QCDModel` $T_k$ reference-convergence drift, differing only in where the "
        "*declared crossing* is reported, as a relative displacement in $z$. The shipped "
        "`BREAK_POINT_STANDOFF` of $+10^{-12}$ is applied on top of whatever is declared, so the "
        "columns read: **as shipped** = one standoff on the near (higher-$z$) side, the side the "
        "departing segment lives on; $-10^{-12}$ = the two cancel and the boundary sits *on* the "
        "crossing; $-10^{-9}$ = the boundary is on the *far* side; the two positive columns move "
        "further onto the near side."
    )
    emit()
    table(
        ["k [1/Mpc]"]
        + [
            ("as shipped" if r == 0.0 else f"declared crossing {r:+.0e}")
            for r in STANDOFF_PLACEMENTS
        ],
        [[f"{row['k']:.5g}"] + [g(d, 3) for d in row["placements"]] for row in rows],
    )


def report_knot_split(result) -> None:
    emit("### 9.7 Would splitting at the C2 spline knots as well close the gap?")
    emit()
    emit(
        "Accuracy, at the wavenumbers §9.1 leaves above the criterion -- the reference-convergence "
        "drift with the ODE split at the 3 declared jumps, and with it split at all 407 declared "
        "break points:"
    )
    emit()
    table(
        [
            "k [1/Mpc]",
            "drift, jumps only",
            "drift, jumps + knots",
            "reference evals, jumps only",
            "reference evals, jumps + knots",
        ],
        [
            [
                f"{row['k']:.5g}",
                g(row["jumps"]),
                g(row["all"]),
                str(row["jumps_evaluations"]),
                str(row["all_evaluations"]),
            ]
            for row in result["accuracy"]
        ],
    )
    emit(
        "Cost, at the **production** tolerances, over every fifth wavenumber of the grid — which "
        "is the figure that matters, because `GkNumericIntegration` is one object per "
        "$(k, z_{\\rm source})$ and there are ~65,000 of them per model:"
    )
    emit()
    table(
        ["sector", "k sampled", "RHS evals, jumps only", "jumps + knots", "change"],
        [
            [
                row["sector"],
                str(row["k_count"]),
                str(row["jumps"]),
                str(row["all"]),
                f"{row['all'] / row['jumps'] - 1.0:+.1%}",
            ]
            for row in result["cost"]
        ],
    )


def main_break_points() -> None:
    t_start = time.perf_counter()

    log("** building stand-in models")
    radiation = RadiationModel()
    lambda_cdm = LambdaCDMModel()
    qcd_cosmology = QCD_Cosmology(
        store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
    )
    t0 = time.perf_counter()
    qcd = QCDModel(
        production_source_grid(
            horizon_exit_z(
                qcd_cosmology,
                PRODUCTION_LARGEST_K_INV_MPC,
                -float(PRODUCTION_SUPERHORIZON_EFOLDS),
            )
        ),
        cosmology=qcd_cosmology,
    )
    qcd_build_seconds = time.perf_counter() - t0
    log(f"   QCDModel built in {qcd_build_seconds:.2f} s")

    log(
        "** reproducing prompt 12's control (prompt 18 §3 item 3: it must be unchanged)"
    )
    control = reproduce_control(radiation)
    for label, summary in control.items():
        log(
            f"   {label}: max dT/env = {summary['max']:.4g} at x = {summary['max_x']:.4g}, "
            f"{summary['evaluations']} RHS evaluations"
        )

    models = (
        ("RadiationModel", radiation, radiation, False),
        ("LambdaCDMModel", lambda_cdm, lambda_cdm.cosmology, False),
        ("QCDModel", qcd, qcd_cosmology, True),
    )

    log("** sweeping")
    results = []
    for sector in ("Tk", "Gk"):
        for name, model, cosmology, declares in models:
            results.append(break_point_sweep(name, model, cosmology, sector, declares))

    log("** where the segment boundary is placed")
    standoff = standoff_experiment(qcd, qcd_cosmology, NAMED_FAILURES[:3])

    tk_qcd = next(r for r in results if r["sector"] == "Tk" and r["declares"])
    offenders = [
        index
        for index, row in enumerate(tk_qcd["rows"])
        if row["drift"]["max"] > ACCEPTANCE_DRIFT
    ]
    log(
        f"** the C2 knots, at the {len(offenders)} wavenumber(s) still above the criterion"
    )
    knots = knot_split_experiment(qcd, qcd_cosmology, offenders)

    elapsed = time.perf_counter() - t_start

    emit("## 9. After the split: prompt 18's measurement")
    emit()
    emit(f"<!-- generated {date.today().isoformat()} by")
    emit(
        "     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py "
        "--break-points"
    )
    emit(
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_control(control)
    report_break_point_convergence(results)
    report_named_failures(results)
    report_break_point_cost(results)
    report_production_shift(results)
    emit("### 9.5 Every wavenumber")
    emit()
    for result in results:
        report_break_point_detail(result)

    report_standoff(standoff)
    report_knot_split(knots)

    emit(
        f"*Runtime {elapsed:.0f} s (QCD stand-in build {qcd_build_seconds:.1f} s of it); "
        f"{len(PRODUCTION_K_GRID)} wavenumbers x 3 models x 2 sectors, 3 solves each on a model "
        f"that declares nothing and 6 on QCDModel.*"
    )
    log(f"** total runtime {elapsed:.1f} s")


# =============================================================================================
# prompt 19 -- the per-sector break-point policy (section 10 of the generated document)
#
# This file is additive: nothing above this banner was changed except to give `run`, `run_gk` and
# `reproduce_control` a `break_point_kind` argument that defaults to what they did before, so that
# section 9's entry point still emits section 9.
#
#     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --per-sector
#
# Prompt 18 left the ODE splitting at the declared *jumps* in both sectors, and measured (section
# 9.7) that splitting at the C2 spline knots as well would close the residual three-wavenumber
# failure on QCDModel's transfer function at +219 % / +155 % of the production evaluations. The
# user's decision of 2026-09-13 is that the cosmology declares everything and each consumer
# chooses: TkNumericIntegration asks for BREAK_POINT_ALL, GkNumericIntegration for
# BREAK_POINT_DISCONTINUITY. What has to be established here:
#
#   1. the acceptance test -- QCDModel Tk at all 50 wavenumbers under the new policy, section 9.7
#      having measured only three;
#   2. that G_k, whose policy did not change, reproduces section 9 *bit for bit*;
#   3. that RadiationModel and LambdaCDMModel, which declare nothing, reproduce bit for bit under
#      *either* policy, and take the single-call path under both;
#   4. the cost, in right-hand-side evaluations and (secondarily) in seconds per object;
#   5. how far the transfer function's answer moves on QCDModel a second time.
# =============================================================================================

# the shipped policy of each sector, matching ComputeTargets/{Tk,Gk}NumericIntegration.py
SECTOR_POLICY = {"Tk": BREAK_POINT_ALL, "Gk": BREAK_POINT_DISCONTINUITY}

# section 9's headline G_k figures, which item 2 requires to reproduce exactly. Quoted to the
# three significant figures section 9.1 printed; the comparison below is against the *printed*
# precision, and the per-sample bit-for-bit statement is made separately by re-running the same
# policy with the argument omitted.
SECTION_9_GK_DRIFT = {
    "RadiationModel": 1.94e-11,
    "LambdaCDMModel": 2.1e-11,
    "QCDModel": 8.41e-09,
}
SECTION_9_GK_QCD_PER_OBJECT = 13320
SECTION_9_PRINTED_DIGITS = 3

# item 4's secondary measure: best-of-N single-core wall time per object, at one representative
# wavenumber of the production grid (index 38 = 4.97e7/Mpc, the wavenumber section 9.6 turns on)
TIMING_REPEATS = 5
TIMING_K_INDEX = 38


def _bitwise_equal(a: dict, b: dict) -> bool:
    """
    Two payloads of :func:`numeric_with_phase_cut` agree in every returned floating-point number
    and in the evaluation count -- ``==`` on doubles, not ``isclose``. This is the check item 2
    and item 3 are stated in: a shared driver that changed behaviour for a sector that did not ask
    it to would show up here and nowhere else.
    """
    for field in ("value_sample", "deriv_sample"):
        if len(a[field]) != len(b[field]):
            return False
        if any(float(x) != float(y) for x, y in zip(a[field], b[field])):
            return False
    for field in ("stop_value", "stop_deriv", "stop_deltaz_subh"):
        if (a[field] is None) != (b[field] is None):
            return False
        if a[field] is not None and float(a[field]) != float(b[field]):
            return False
    return a["data"].RHS_evaluations == b["data"].RHS_evaluations


def _matches_to_printed_precision(measured: float, printed: float) -> bool:
    """``measured`` rounds to ``printed`` at the precision section 9 printed it to."""
    return float(f"{measured:.{SECTION_9_PRINTED_DIGITS}g}") == float(
        f"{printed:.{SECTION_9_PRINTED_DIGITS}g}"
    )


def per_sector_sweep(name: str, model, cosmology, sector: str, declares: bool) -> dict:
    """
    Every wavenumber, for one (model, sector), under that sector's shipped policy -- and, where
    the two policies are not the same request, under the jumps-only policy as well, so that the
    "before" column is measured on this tree rather than quoted from section 9.

    For a model that declares nothing the two policies are the same *code path*, and what is
    measured is that they are also the same *numbers*: the production run is issued twice and
    compared bit for bit.

    For a sector whose policy this prompt did not change, the production run is issued a second
    time with ``break_point_kind`` omitted, which is what establishes that the module default
    really is what the call site now names.
    """
    policy = SECTOR_POLICY[sector]
    geometry_fn = SECTORS[sector]["geometry"]
    run_fn = SECTORS[sector]["run"]
    production_atol = TK_PRODUCTION_ATOL if sector == "Tk" else GK_PRODUCTION_ATOL

    rows = []
    t_model = time.perf_counter()
    for index, k in enumerate(PRODUCTION_K_GRID):
        k = float(k)
        geo = geometry_fn(cosmology, k)
        z_lo, z_hi = float(geo["grid"].min), geo["grid"].max.z

        reference = run_fn(
            model, k, geo, REFERENCE_ATOL, REFERENCE_RTOL, break_point_kind=policy
        )
        tightened = run_fn(
            model, k, geo, TIGHTENED_ATOL, TIGHTENED_RTOL, break_point_kind=policy
        )
        production = run_fn(
            model, k, geo, production_atol, PRODUCTION_RTOL, break_point_kind=policy
        )
        drift = summarise(sector_errors(sector, model, k, geo, tightened, reference))

        entry = {
            "k": k,
            "segments": 1
            + len(declared_discontinuities_in_z(model, z_lo, z_hi, kind=policy)),
            "jumps_segments": 1
            + len(
                declared_discontinuities_in_z(
                    model, z_lo, z_hi, kind=BREAK_POINT_DISCONTINUITY
                )
            ),
            "drift": drift,
            "reference_evaluations": reference["data"].RHS_evaluations,
            "production_evaluations": production["data"].RHS_evaluations,
        }

        # the jumps-only comparison. For G_k that is the same request as the shipped policy, so
        # the second run is issued with the argument *omitted* -- the default check -- and the
        # before/after columns are the same numbers by construction.
        if policy == BREAK_POINT_DISCONTINUITY:
            default = run_fn(model, k, geo, production_atol, PRODUCTION_RTOL)
            entry["default_is_the_policy"] = _bitwise_equal(default, production)
            entry["jumps_drift"] = drift
            entry["jumps_production_evaluations"] = entry["production_evaluations"]
            entry["shift"] = 0.0
        else:
            jumps_production = run_fn(
                model,
                k,
                geo,
                production_atol,
                PRODUCTION_RTOL,
                break_point_kind=BREAK_POINT_DISCONTINUITY,
            )
            entry["jumps_production_evaluations"] = jumps_production[
                "data"
            ].RHS_evaluations

            if declares:
                jumps_reference = run_fn(
                    model,
                    k,
                    geo,
                    REFERENCE_ATOL,
                    REFERENCE_RTOL,
                    break_point_kind=BREAK_POINT_DISCONTINUITY,
                )
                jumps_tightened = run_fn(
                    model,
                    k,
                    geo,
                    TIGHTENED_ATOL,
                    TIGHTENED_RTOL,
                    break_point_kind=BREAK_POINT_DISCONTINUITY,
                )
                entry["jumps_drift"] = summarise(
                    sector_errors(
                        sector, model, k, geo, jumps_tightened, jumps_reference
                    )
                )
                entry["shift"] = summarise(
                    sector_errors(sector, model, k, geo, jumps_production, production)
                )["max"]
            else:
                # declares nothing: the two policies must be the same numbers, not merely the
                # same code path, and that is the statement -- not a drift comparison
                entry["policy_identical"] = _bitwise_equal(jumps_production, production)
                entry["jumps_drift"] = drift
                entry["shift"] = 0.0

        rows.append(entry)
        log(
            f"   {sector} {name} [{index + 1:2d}/{len(PRODUCTION_K_GRID)}] k={k:.4g}: "
            f"{entry['segments']} segment(s) (jumps only: {entry['jumps_segments']}), "
            f"drift {drift['max']:.2e}, {entry['production_evaluations']} evals"
        )

    log(f"   {sector} {name} done in {time.perf_counter() - t_model:.1f} s")
    return {"name": name, "sector": sector, "declares": declares, "rows": rows}


def timing_experiment(models) -> list:
    """
    Item 4's secondary measure. Best of :data:`TIMING_REPEATS` single-core wall-clock times for
    one object at the production tolerances, per sector under its own shipped policy, with the
    jumps-only time alongside. Best-of rather than mean because the quantity wanted is the cost of
    the work, not of the machine's other tenants.
    """
    k = float(PRODUCTION_K_GRID[TIMING_K_INDEX])
    out = []
    for name, model, cosmology, declares in models:
        for sector in ("Tk", "Gk"):
            geometry_fn = SECTORS[sector]["geometry"]
            run_fn = SECTORS[sector]["run"]
            atol = TK_PRODUCTION_ATOL if sector == "Tk" else GK_PRODUCTION_ATOL
            geo = geometry_fn(cosmology, k)

            timings = {}
            for label, kind in (
                ("shipped", SECTOR_POLICY[sector]),
                ("jumps", BREAK_POINT_DISCONTINUITY),
            ):
                best = None
                evaluations = None
                for _ in range(TIMING_REPEATS):
                    t0 = time.perf_counter()
                    payload = run_fn(
                        model, k, geo, atol, PRODUCTION_RTOL, break_point_kind=kind
                    )
                    elapsed = time.perf_counter() - t0
                    best = elapsed if best is None else min(best, elapsed)
                    evaluations = payload["data"].RHS_evaluations
                timings[label] = (best, evaluations)

            out.append(
                {
                    "model": name,
                    "sector": sector,
                    "k": k,
                    "policy": SECTOR_POLICY[sector],
                    "shipped": timings["shipped"],
                    "jumps": timings["jumps"],
                }
            )
            log(
                f"   timing {sector} {name}: shipped {timings['shipped'][0]:.4f} s "
                f"({timings['shipped'][1]} evals), jumps-only {timings['jumps'][0]:.4f} s "
                f"({timings['jumps'][1]} evals)"
            )
    return out


def report_policy(results) -> None:
    emit("### 10.1 The policy, and the acceptance test")
    emit()
    emit(
        "`numeric_with_phase_cut` now takes a `break_point_kind`, defaulting to "
        "`BREAK_POINT_DISCONTINUITY` -- what it asked for unconditionally when §9 was measured. "
        "Both production integrators pass it explicitly: `TkNumericIntegration` asks for "
        "`BREAK_POINT_ALL` and `GkNumericIntegration` for `BREAK_POINT_DISCONTINUITY`. "
        '"Segments" below is the number of integrations one object is cut into under the '
        "sector's own policy, against the number the jumps alone would give."
    )
    emit()
    rows = []
    for result in results:
        drifts = [row["drift"]["max"] for row in result["rows"]]
        before = [row["jumps_drift"]["max"] for row in result["rows"]]
        worst = result["rows"][max(range(len(drifts)), key=lambda i: drifts[i])]
        offenders = sum(1 for d in drifts if d > ACCEPTANCE_DRIFT)
        rows.append(
            [
                result["sector"],
                result["name"],
                SECTOR_POLICY[result["sector"]],
                f"{max(row['segments'] for row in result['rows'])} / "
                f"{max(row['jumps_segments'] for row in result['rows'])}",
                g(max(before)),
                g(max(drifts)),
                f"{worst['k']:.4g}",
                g(median(drifts)),
                str(offenders),
                "**yes**" if offenders == 0 else "**no**",
            ]
        )
    table(
        [
            "sector",
            "model",
            "policy",
            "segments, shipped / jumps only",
            "worst drift, jumps only",
            "worst drift, shipped",
            "at k [1/Mpc]",
            "median drift, shipped",
            f"k above {ACCEPTANCE_DRIFT:.1g}",
            "acceptance met?",
        ],
        rows,
    )


def report_gk_regression(results) -> None:
    emit("### 10.2 The $G_k$ regression: §9 reproduced, bit for bit")
    emit()
    emit(
        "$G_k$'s policy is unchanged by this prompt, so every §9 figure for that sector must come "
        "back unchanged -- and the driver itself was touched (a guard on the separation of "
        "segment boundaries), so this is the check that the sector which did not ask for a change "
        'did not get one. "default matches" is the same production run issued with '
        "`break_point_kind` omitted, compared sample by sample with `==`."
    )
    emit()
    rows = []
    for result in results:
        if result["sector"] != "Gk":
            continue
        drifts = [row["drift"]["max"] for row in result["rows"]]
        per_object = sum(row["production_evaluations"] for row in result["rows"]) / len(
            result["rows"]
        )
        expected = SECTION_9_GK_DRIFT[result["name"]]
        rows.append(
            [
                result["name"],
                g(expected),
                g(max(drifts)),
                (
                    "**yes**"
                    if _matches_to_printed_precision(max(drifts), expected)
                    else "**NO**"
                ),
                str(sum(1 for d in drifts if d > ACCEPTANCE_DRIFT)),
                f"{per_object:.0f}",
                (
                    "all 50"
                    if all(row["default_is_the_policy"] for row in result["rows"])
                    else "**NOT ALL**"
                ),
            ]
        )
    table(
        [
            "model",
            "§9.1 worst drift",
            "measured now",
            "reproduces?",
            f"k above {ACCEPTANCE_DRIFT:.1g}",
            "RHS evals per object",
            "default matches explicit policy",
        ],
        rows,
    )
    emit(
        f"§9.3 gives {SECTION_9_GK_QCD_PER_OBJECT} right-hand-side evaluations per QCD $G_k$ "
        "object at the production tolerances; the table's QCD row is the same quantity."
    )
    emit()


def report_smooth_regression(results, control_jumps, control_shipped) -> None:
    emit("### 10.3 The smooth-model regression")
    emit()
    emit(
        "`RadiationModel` and `LambdaCDMModel` declare nothing, so both policies reach the same "
        "single-`solve_ivp` call. That is asserted twice over: the segment count is 1 under "
        "either policy at every wavenumber, and the production run issued under one policy is "
        "compared with the run issued under the other sample by sample with `==`."
    )
    emit()
    rows = []
    for result in results:
        if result["declares"]:
            continue
        identical = [row.get("policy_identical") for row in result["rows"]]
        checked = [value for value in identical if value is not None]
        rows.append(
            [
                result["sector"],
                result["name"],
                str(max(row["segments"] for row in result["rows"])),
                str(max(row["jumps_segments"] for row in result["rows"])),
                g(max(row["drift"]["max"] for row in result["rows"])),
                (
                    "n/a (same request)"
                    if len(checked) == 0
                    else (f"all {len(checked)}" if all(checked) else "**NOT ALL**")
                ),
            ]
        )
    table(
        [
            "sector",
            "model",
            "max segments, shipped policy",
            "max segments, jumps only",
            "worst drift",
            "policies bit-identical",
        ],
        rows,
    )
    emit(
        "And prompt 17's two control figures, against the exact $T$, under the jumps-only policy "
        "(§9's column) and under the transfer function's shipped policy:"
    )
    emit()
    table(
        [
            "control",
            "prompt 12",
            "jumps only",
            "RHS evals",
            "shipped policy",
            "RHS evals",
            "identical?",
        ],
        [
            [
                label,
                g(control_jumps[label]["expected"]),
                g(control_jumps[label]["max"]),
                str(control_jumps[label]["evaluations"]),
                g(control_shipped[label]["max"]),
                str(control_shipped[label]["evaluations"]),
                (
                    "yes"
                    if control_jumps[label]["max"] == control_shipped[label]["max"]
                    and control_jumps[label]["evaluations"]
                    == control_shipped[label]["evaluations"]
                    else "**no**"
                ),
            ]
            for label in control_jumps
        ],
    )


def report_per_sector_cost(results, timings) -> None:
    emit("### 10.4 What the policy costs, in evaluations and in seconds")
    emit()
    emit(
        "Right-hand-side evaluation counts are the reproducible measure (§5 note 14): they do not "
        "depend on the machine. The seconds below them are for scoping only -- they are what make "
        "the decision legible, because it turned on the sector's *object count* rather than its "
        "per-object cost."
    )
    emit()
    rows = []
    for result in results:
        if not result["declares"]:
            continue
        before = sum(row["jumps_production_evaluations"] for row in result["rows"])
        after = sum(row["production_evaluations"] for row in result["rows"])
        rows.append(
            [
                result["sector"],
                result["name"],
                SECTOR_POLICY[result["sector"]],
                f"{before / len(result['rows']):.0f}",
                f"{after / len(result['rows']):.0f}",
                str(before),
                str(after),
                f"{after / before - 1.0:+.2%}",
            ]
        )
    table(
        [
            "sector",
            "model",
            "policy",
            "per object, jumps only",
            "per object, shipped",
            "grid total, jumps only",
            "grid total, shipped",
            "change",
        ],
        rows,
    )
    emit(
        f"Wall time per object, best of {TIMING_REPEATS} single-core runs at "
        f"k = {float(PRODUCTION_K_GRID[TIMING_K_INDEX]):.4g}/Mpc and the production tolerances. "
        "**The counts above are the measure; these seconds are for scoping.**"
    )
    emit()
    table(
        [
            "sector",
            "model",
            "policy",
            "s per object, shipped",
            "s per object, jumps only",
            "objects per model",
            "sector total, shipped",
        ],
        [
            [
                row["sector"],
                row["model"],
                row["policy"],
                f"{row['shipped'][0]:.4f}",
                f"{row['jumps'][0]:.4f}",
                "50" if row["sector"] == "Tk" else "~65,000",
                (
                    f"{50 * row['shipped'][0]:.1f} s"
                    if row["sector"] == "Tk"
                    else f"{65000 * row['shipped'][0] / 3600.0:.1f} core-hours"
                ),
            ]
            for row in timings
        ],
    )


def report_per_sector_shift(results) -> None:
    emit("### 10.5 How far the transfer function's answer moves on QCD, again")
    emit()
    emit(
        "The production-tolerance run under the jumps-only policy, scored against the converged "
        "reference under the shipped policy -- the same envelope-relative measure as §9.4. Prompt "
        "18 already moved this sector's QCD values by up to 2.82e-04; this is the second move, on "
        "top of it."
    )
    emit()
    rows = []
    for result in results:
        if not result["declares"] or result["sector"] != "Tk":
            continue
        shifts = [row["shift"] for row in result["rows"]]
        worst = result["rows"][max(range(len(shifts)), key=lambda i: shifts[i])]
        rows.append(
            [
                result["sector"],
                result["name"],
                g(max(shifts)),
                f"{worst['k']:.4g}",
                g(median(shifts)),
                g(min(shifts)),
            ]
        )
    table(
        [
            "sector",
            "model",
            "worst shift",
            "at k [1/Mpc]",
            "median shift",
            "smallest shift",
        ],
        rows,
    )


def report_per_sector_detail(result) -> None:
    emit(f"#### {result['sector']}, {result['name']}")
    emit()
    header = [
        "k [1/Mpc]",
        "segments",
        "drift, jumps only",
        "drift, shipped",
        f"<= {ACCEPTANCE_DRIFT:.1g}?",
        "production shift",
        "RHS evals, jumps only -> shipped",
    ]
    rows = []
    for row in result["rows"]:
        rows.append(
            [
                f"{row['k']:.4g}",
                str(row["segments"]),
                g(row["jumps_drift"]["max"], 2),
                g(row["drift"]["max"], 2),
                "yes" if row["drift"]["max"] <= ACCEPTANCE_DRIFT else "**no**",
                g(row["shift"], 2),
                f"{row['jumps_production_evaluations']} -> {row['production_evaluations']}",
            ]
        )
    table(header, rows)


def main_per_sector() -> None:
    t_start = time.perf_counter()

    log("** building stand-in models")
    radiation = RadiationModel()
    lambda_cdm = LambdaCDMModel()
    qcd_cosmology = QCD_Cosmology(
        store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
    )
    t0 = time.perf_counter()
    qcd = QCDModel(
        production_source_grid(
            horizon_exit_z(
                qcd_cosmology,
                PRODUCTION_LARGEST_K_INV_MPC,
                -float(PRODUCTION_SUPERHORIZON_EFOLDS),
            )
        ),
        cosmology=qcd_cosmology,
    )
    qcd_build_seconds = time.perf_counter() - t0
    log(f"   QCDModel built in {qcd_build_seconds:.2f} s")

    log("** the control, under both policies")
    control_jumps = reproduce_control(radiation, BREAK_POINT_DISCONTINUITY)
    control_shipped = reproduce_control(radiation, SECTOR_POLICY["Tk"])
    for label in control_jumps:
        log(
            f"   {label}: jumps only {control_jumps[label]['max']:.4g} "
            f"({control_jumps[label]['evaluations']} evals), shipped "
            f"{control_shipped[label]['max']:.4g} "
            f"({control_shipped[label]['evaluations']} evals)"
        )

    models = (
        ("RadiationModel", radiation, radiation, False),
        ("LambdaCDMModel", lambda_cdm, lambda_cdm.cosmology, False),
        ("QCDModel", qcd, qcd_cosmology, True),
    )

    log("** sweeping")
    results = []
    for sector in ("Tk", "Gk"):
        for name, model, cosmology, declares in models:
            results.append(per_sector_sweep(name, model, cosmology, sector, declares))

    log("** wall time per object")
    timings = timing_experiment(models)

    elapsed = time.perf_counter() - t_start

    emit("## 10. The per-sector policy: prompt 19's measurement")
    emit()
    emit(f"<!-- generated {date.today().isoformat()} by")
    emit(
        "     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py "
        "--per-sector"
    )
    emit(
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_policy(results)
    report_gk_regression(results)
    report_smooth_regression(results, control_jumps, control_shipped)
    report_per_sector_cost(results, timings)
    report_per_sector_shift(results)
    emit("### 10.6 Every wavenumber")
    emit()
    for result in results:
        report_per_sector_detail(result)

    emit(
        f"*Runtime {elapsed:.0f} s (QCD stand-in build {qcd_build_seconds:.1f} s of it); "
        f"{len(PRODUCTION_K_GRID)} wavenumbers x 3 models x 2 sectors.*"
    )
    log(f"** total runtime {elapsed:.1f} s")


if __name__ == "__main__":
    if "--break-points" in sys.argv[1:]:
        main_break_points()
    elif "--per-sector" in sys.argv[1:]:
        main_per_sector()
    else:
        main()
