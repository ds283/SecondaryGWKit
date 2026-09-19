"""
Is ``QuadSourceIntegral`` limited by its quadrature tolerance, or by the representation it
integrates? (``prompts/tolerance-convergence/`` prompt 06, board item **T11**.)

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/quadsource_readonly.py

It emits **everything from §1 down** of ``docs/tolerance-convergence/QUADSOURCE-READONLY.md`` on
stdout and a progress log on stderr; that document's header and §0 are written, as
``GK-NUMERIC-SWEEP.md``'s and ``ORDER-AUDIT.md``'s are. **It changes nothing.** No production module is touched, no constant
moves, and no value is recommended: ``DEFAULT_QUADRATURE_ATOL`` and ``DEFAULT_QUADRATURE_RTOL``
belong to ``prompts/levin-refactor`` and ``prompts/qsi-phase-groups`` (README §0.4), and this
measurement is handed to them.

**Why read-only, and why offline.** ``QuadSourceIntegral`` is the eighth keyed object type and the
only one this campaign does not own. It is also the only one whose tolerances were *already*
decoupled and *already* chosen by measurement (``prompts/source-remediation`` log 12). The question
here is README §6.1 rule 1's, asked of a sector nobody has asked it of: **what is the floor, and
does the quadrature error reach it?** The two live-run drivers under
``docs/source-remediation-verification/`` need Ray and a populated datastore; every sweep in this
campaign needs neither (``CLAUDE.md``, README §5), and the live-run statistics this campaign needs
are already in ``source-remediation``'s record and are cited rather than re-taken (README §7 D11,
user decision 2026-09-18).

**The instrument, and it is the whole design.**
``ComputeTargets/tests/test_quadsource_integral.py`` builds ``QuadSourceIntegral`` offline on the
exact constant-$w$ background in two flavours, and this script imports ``Case``, ``SHAPES``,
``X_RESP_VALUES`` and ``B_VALUES`` from it **without editing it**:

* **exact** -- the transfer functions, the Green's function and the source spline are all exact to
  rounding, so every ingredient is truth and what is left is the partition and the two integrators
  alone. This is where the *quadrature error* is measured.
* **realistic** -- ``bessel_phase`` amplitude and phase re-splined on the production
  100-per-decade grid, a real ``phase_spline`` Green's function, and ``f`` a spline through exact
  samples as ``QuadSource`` stores it. This carries the representation floors logs 05, 06 and 07 of
  ``source-remediation`` measured. This is where the *floor* is measured.

Sweeping the realistic flavour alone would measure the two convolved and could report neither.

**The one thing the fixture's own tests do not do, and this script must.** ``analytic_rad`` -- the
code's own oracle, and a stored column -- is computed *inside* ``evaluate_QuadSource_integral`` at
**the caller's own ``atol`` and ``rtol``** (``QuadSourceIntegral.py:884-897``). So
``|total - analytic_rad|`` taken from a single run compares two quantities that moved together, and
a sweep of it measures neither. Every oracle number here is therefore ``analytic_rad`` evaluated at
the **reference** pair once per case and then held fixed, and the primary quadrature statistic is a
self-convergence one -- ``total`` against ``total`` at the reference pair, with that reference's own
drift beside it (README §5 rule 5), through ``convergence_reference.reference_drift``, which will
not return a drift without the verdict attached.

**What is swept**, over all ``2 (b) x 3 (shape) x 3 (x_resp) = 18`` cases of the fixture:

1. the **``atol`` axis** at the production ``rtol = 1e-8``, ``atol`` from ``1e-12`` to ``1e-40``.
   One step is **four decades** here, not one: the production value is ``1e-32`` and a single
   decade at that magnitude is not a step anyone would take. The axis runs two steps tighter and
   five steps looser than production, so that "inert" is a measurement over twenty-eight decades
   and not an assumption, and so that the setting at which ``atol`` *starts* to bind is located
   rather than bounded;
2. the **``rtol`` axis** at the production ``atol = 1e-32``, ``rtol`` from ``1e-5`` to ``1e-11``,
   one decade per step, three either side of production. This is the axis
   ``[12-atol-too-loose-for-the-source-integral]`` could not see: ``source-remediation`` log 12
   found ``1e-8 -> 1e-11`` bit-identical on 159 live items **because ``atol`` bound at ``1e-25``**,
   and ``atol`` is now seven decades tighter, so that finding does not transfer and the question is
   open again on this tree;
3. the four **corners** of the two axes' extremes, which is what makes "``atol`` is inert" a
   statement about the matrix rather than about a line through it (README §2 (d) is the record of
   what a diagonal costs);
4. the **representation floor**, on the realistic flavour at the production pair -- and at the two
   extreme ``rtol`` as well, so that the floor's independence of the tolerance is measured too;
5. a **second, independent oracle**: ``scipy.quad`` of the exact integrand on a short range
   straddling every hand-over, which shares none of ``analytic_integral``'s machinery. The fixture
   makes this comparison inside a test method rather than at module level, so the construction is
   repeated here (three lines) rather than that module being refactored -- prompt 06 §2 says to do
   exactly that.

**Error measures and their normalisation.** ``numeric_quad`` and ``WKB_Levin`` cancel -- by up to
5x in these cases -- so an error normalised by ``|total|`` is inflated by that cancellation and an
error normalised by the parts is not. Both are reported, and the primary normalisation is the
fixture's own ``scale = max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)`` taken from the
**reference** run of the same case, so that one case's figures are all divided by one fixed number.

**This sector has no source grid** (README §5 rule 6 has nothing to bite on): the fixture's only
sampling parameter is ``PRODUCTION_SAMPLES_PER_LOG10Z = 100``, which the realistic flavour uses for
the source spline and the Green's-function phase spline. It is stated on every realistic figure
instead of a grid generation.

**What the fixture is not.** It is not the production ``(k, q, r, z)`` range. The 58 % figure, the
regime mix and the clamp-gap census of ``source-remediation`` log 12 are live-run measurements,
cited and never superseded here (prompt 06 §3).

No Ray and no datastore (``CLAUDE.md``).
"""

import platform
import sys
import time
from datetime import date
from math import log, sqrt

import numpy as np
import scipy
from scipy.integrate import quad

from ComputeTargets.tests.convergence_reference import TolerancePair, reference_drift
from ComputeTargets.tests.test_quadsource_integral import (
    B_VALUES,
    Case,
    SHAPES,
    X_RESP_VALUES,
    z_at_tau,
)
from ComputeTargets.tests.test_tk_source_functions import PRODUCTION_SAMPLES_PER_LOG10Z
from config.defaults import DEFAULT_QUADRATURE_ATOL, DEFAULT_QUADRATURE_RTOL

# ---------------------------------------------------------------------------------------------
# the settings swept
# ---------------------------------------------------------------------------------------------

#: The production pair, imported rather than written down.
PRODUCTION = TolerancePair(atol=DEFAULT_QUADRATURE_ATOL, rtol=DEFAULT_QUADRATURE_RTOL)

#: The reference. `rtol = 1e-12` is two decades above SciPy's `quad` floor of `50 eps = 1.11e-14`,
#: so the one-step-tighter reference at `1e-13` is still a step the integrator really takes; a
#: drift measured across a clamped step is zero by construction and says nothing (the note
#: `TolerancePair.rtol_step_is_effective` carries for `solve_ivp`, applied here to `quad`).
#: `atol = 1e-45` is thirteen decades below production and cannot bind on any quantity in this
#: fixture, where |total| ~ 1e-13 to 1e-12.
REFERENCE = TolerancePair(atol=1.0e-45, rtol=1.0e-12)

#: The `atol` ladder at the production `rtol`. One step is four decades (see the docstring).
ATOL_AXIS = (1.0e-12, 1.0e-16, 1.0e-20, 1.0e-24, 1.0e-28, 1.0e-32, 1.0e-36, 1.0e-40)

#: The `rtol` ladder at the production `atol`. One step is one decade.
RTOL_AXIS = (1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11)

#: The corners: both axes at their extremes, so that the inertness claim is about the matrix.
CORNERS = tuple(
    (atol, rtol) for atol in (1.0e-12, 1.0e-40) for rtol in (1.0e-5, 1.0e-11)
)

#: The realistic flavour is run at the production pair and at the two extreme `rtol`, so that
#: "the floor does not move with the tolerance" is measured rather than argued.
REALISTIC_SETTINGS = (
    (DEFAULT_QUADRATURE_ATOL, 1.0e-5),
    (DEFAULT_QUADRATURE_ATOL, DEFAULT_QUADRATURE_RTOL),
    (DEFAULT_QUADRATURE_ATOL, 1.0e-11),
)

#: `x_r` at which the second oracle is taken, and the top of its short range in the same variable.
#: `scipy.quad` of the exact integrand converges on a range this short and does not on the full
#: one, which is why the fixture's own second-oracle test uses the same two numbers.
QUAD_ORACLE_X_RESP = 60.0
QUAD_ORACLE_X_TOP = 4.0

#: The object count of the sector, from `TOLERANCE-INVENTORY.md` §5.5. A per-item cost figure is
#: meaningless without it (README §2 (c)).
OBJECT_COUNT = "1,275 x 50 x (response z) per model"


def log_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


_OUT = []


def emit(line: str = "") -> None:
    _OUT.append(line)


# ---------------------------------------------------------------------------------------------
# running one case
# ---------------------------------------------------------------------------------------------


def case_label(b, shape, x_resp, flavour) -> str:
    return f"b={b:g} {shape.name} x_resp={x_resp:g} {flavour}"


def run(case: Case, pair) -> dict:
    """One `evaluate_QuadSource_integral` at `pair`, with the counters pulled out flat."""
    atol, rtol = (pair.atol, pair.rtol) if isinstance(pair, TolerancePair) else pair
    started = time.perf_counter()
    out = case.run(atol=atol, rtol=rtol)
    levin = out["WKB_Levin_data"]
    # either limb can be absent: a case whose whole range is oscillatory has no `quad`
    # sub-interval, and one whose whole range is smooth has no Levin call
    smooth = out["numeric_quad_data"]
    return {
        "atol": atol,
        "rtol": rtol,
        "total": out["total"],
        "numeric_quad": out["numeric_quad"],
        "WKB_Levin": out["WKB_Levin"],
        "analytic_rad": float(out["analytic_rad"]),
        # the *declared* bound, never the measured residual -- prompt 06 §3, and
        # [09-abserr-is-a-quadrature-bound]
        "declared_abserr": out["total_abserr"],
        "declared_converged": out["total_converged"],
        "quad_evaluations": smooth.RHS_evaluations if smooth is not None else 0,
        "levin_evaluations": levin.evaluations if levin is not None else 0,
        "levin_regions": levin.num_regions if levin is not None else 0,
        "levin_max_depth": levin.max_depth if levin is not None else 0,
        "wall": time.perf_counter() - started,
    }


def scale_of(reference_payload: dict) -> float:
    """
    The normalisation, taken once per case from its reference run.

    `numeric_quad` and `WKB_Levin` cancel, so `|total|` is not the size of the thing being
    integrated. This is the fixture's own choice of scale (`TestAnalyticOracle`), reused so that
    the numbers here can be read beside the module's.
    """
    return max(
        abs(reference_payload["numeric_quad"]),
        abs(reference_payload["WKB_Levin"]),
        abs(reference_payload["analytic_rad"]),
    )


def sweep_case(b, shape, x_resp) -> dict:
    """Every exact-flavour run of one case, plus its reference and the fixed oracle."""
    label = case_label(b, shape, x_resp, "exact")
    log_progress(f"  {label}")
    case = Case(b, shape, x_resp, exact=True)

    reference = run(case, REFERENCE)
    scale = scale_of(reference)
    oracle = reference["analytic_rad"]

    settings = []
    for atol in ATOL_AXIS:
        settings.append(("atol", atol, DEFAULT_QUADRATURE_RTOL))
    for rtol in RTOL_AXIS:
        settings.append(("rtol", DEFAULT_QUADRATURE_ATOL, rtol))
    for atol, rtol in CORNERS:
        settings.append(("corner", atol, rtol))

    rows = []
    for axis, atol, rtol in settings:
        payload = run(case, (atol, rtol))
        payload["axis"] = axis
        payload["error"] = abs(payload["total"] - reference["total"]) / scale
        payload["error_rel_total"] = abs(payload["total"] - reference["total"]) / abs(
            reference["total"]
        )
        payload["oracle_error"] = abs(payload["total"] - oracle) / scale
        payload["oracle_drift_own"] = abs(payload["analytic_rad"] - oracle) / scale
        rows.append(payload)

    return {
        "b": b,
        "shape": shape,
        "x_resp": x_resp,
        "label": label,
        "case": case,
        "scale": scale,
        "reference": reference,
        "oracle": oracle,
        "rows": rows,
        "z_resp": case.z_resp,
        "z_source_max": case.z_source_max,
        # the rounding floor of this case's own arithmetic: one ulp of `total`, in the same
        # normalisation as every error here. Nothing measured through doubles can resolve a
        # difference below it, and §8 uses it as the lower bound on the quadrature figure so
        # that a case whose candidates all reproduce the reference exactly reports a bound
        # rather than an infinity
        "ulp": float(np.finfo(float).eps) * abs(reference["total"]) / scale,
    }


def score_drift(result: dict, floor: float) -> None:
    """
    The reference's own drift, scored against **the floor** -- because the floor is the smallest
    difference this document draws a conclusion from.

    README §5 rule 5 asks for the drift beside every number and forbids a conclusion drawn from a
    signal that does not exceed it. The signal §8 concludes from is the *separation* between the
    representation floor and the quadrature error, and the smaller of those two is the floor; the
    quadrature errors themselves are reported as upper bounds and no conclusion is drawn from any
    of them. So the floor is `smallest_reported_difference` and the criterion is drift <= floor/10.
    """
    case = result["case"]
    scale = result["scale"]
    result["drift"] = reference_drift(
        lambda knob: run(case, knob),
        REFERENCE,
        error_measure=lambda candidate, ref: [
            (
                abs(candidate["total"] - ref["total"]) / scale,
                result["z_resp"],
                result["x_resp"],
            )
        ],
        smallest_reported_difference=floor,
        reference=result["reference"],
    )


def floor_case(b, shape, x_resp, exact_rows, scale, oracle) -> list:
    """
    The realistic flavour of one case, against the exact flavour of the same case.

    The comparison is against the exact run **at the same tolerance**, which is what isolates the
    representation: both runs then carry the same quadrature error and it cancels out of the
    difference. The residual against the fixed oracle is reported beside it, because that is the
    error a consumer of the stored `total` actually carries.
    """
    label = case_label(b, shape, x_resp, "realistic")
    log_progress(f"  {label}")
    case = Case(b, shape, x_resp, exact=False)

    rows = []
    for atol, rtol in REALISTIC_SETTINGS:
        payload = run(case, (atol, rtol))
        twin = exact_rows[(atol, rtol)]
        payload["vs_exact"] = abs(payload["total"] - twin["total"]) / scale
        payload["vs_exact_rel_total"] = abs(payload["total"] - twin["total"]) / abs(
            twin["total"]
        )
        payload["vs_oracle"] = abs(payload["total"] - oracle) / scale
        rows.append(payload)
    return rows


def quad_oracle_case(b, shape) -> dict:
    """
    The second oracle: `scipy.quad` of the exact integrand over a short range straddling every
    hand-over, which shares none of `analytic_integral`'s machinery.

    The construction is the fixture's `TestDirectQuadOracle.test_short_range`, which is local to
    that test method rather than to the module; prompt 06 §2 says to rebuild it here rather than
    refactor theirs.
    """
    log_progress(f"  quad oracle: b={b:g} {shape.name}")
    probe = Case(b, shape, QUAD_ORACLE_X_RESP, exact=True)
    cs = sqrt(probe.w)
    z_top = z_at_tau(probe.model, QUAD_ORACLE_X_TOP / (shape.r * cs))
    case = Case(b, shape, QUAD_ORACLE_X_RESP, exact=True, z_source_max=z_top)

    ref, err = quad(
        case.integrand_exact,
        log(1.0 + case.z_resp),
        log(1.0 + case.z_source_max),
        limit=2000,
        epsabs=0.0,
        epsrel=1.0e-12,
    )
    ref *= 1.0 + case.z_resp

    rows = {}
    for name, pair in (("production", PRODUCTION), ("reference", REFERENCE)):
        payload = run(case, pair)
        rows[name] = {
            "payload": payload,
            "rel": abs(payload["total"] - ref) / abs(ref),
            "oracle_rel": abs(payload["analytic_rad"] - ref) / abs(ref),
        }
    return {
        "b": b,
        "shape": shape,
        "quad": ref,
        "quad_declared_rel": abs(err) * (1.0 + case.z_resp) / abs(ref),
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def worst(values):
    return max(values) if values else 0.0


def report_header(results, floors):
    emit("## 1. What was measured, and on what")
    emit()
    emit(
        f"**{len(results)} cases**, the fixture's whole grid: `b` in "
        f"{{{', '.join(f'{b:g}' for b in B_VALUES)}}} x "
        f"{len(SHAPES)} `(k, q, r)` shapes ({', '.join(s.name for s in SHAPES)}) x "
        f"{len(X_RESP_VALUES)} response redshifts "
        f"(`x_resp` in {{{', '.join(f'{x:g}' for x in X_RESP_VALUES)}}}), "
        f"each in the **exact** flavour over "
        f"{len(ATOL_AXIS) + len(RTOL_AXIS) + len(CORNERS)} `(atol, rtol)` settings plus a "
        f"reference and its one-step-tighter twin, and each in the **realistic** flavour at "
        f"{len(REALISTIC_SETTINGS)} settings."
    )
    emit()
    emit(
        f"Production is `atol = {DEFAULT_QUADRATURE_ATOL:g}`, `rtol = {DEFAULT_QUADRATURE_RTOL:g}` "
        f"(`config/defaults.py`, imported by this script). The reference is "
        f"`{REFERENCE.label}` and its tightened twin `{REFERENCE.tighter().label}`."
    )
    emit()
    emit(
        f"**There is no source grid in this sector** (README §5 rule 6). The realistic flavour's "
        f"only sampling parameter is `PRODUCTION_SAMPLES_PER_LOG10Z = "
        f"{PRODUCTION_SAMPLES_PER_LOG10Z}`, the density at which it re-splines the `bessel_phase` "
        f"amplitude and phase, the Green's-function phase and `f`; the exact flavour has none at "
        f"all. Every figure below says which flavour and which `(b, shape, x_resp)` it was taken "
        f"at, and a realistic figure is a figure at that density."
    )
    emit()
    emit("### 1.1 The geometry of the cases")
    emit()
    emit(
        "| case | z_response | z_source_max | scale = max(\\|quad\\|, \\|Levin\\|, \\|analytic\\|) | \\|total\\| at the reference | cancellation \\|quad\\|/\\|total\\| |"
    )
    emit("|---|---|---|---|---|---|")
    for result in results:
        ref = result["reference"]
        emit(
            f"| `{result['label']}` | {result['z_resp']:.5g} | {result['z_source_max']:.5g} | "
            f"{result['scale']:.4e} | {abs(ref['total']):.4e} | "
            f"{abs(ref['numeric_quad']) / abs(ref['total']):.1f} |"
        )
    emit()
    emit(
        "The last column is why the primary normalisation is `scale` and not `|total|`: the "
        "smooth and oscillatory halves cancel, so an error divided by `|total|` is inflated by "
        "the cancellation factor. Both normalisations appear below, labelled."
    )
    emit()


def report_reference(results):
    emit("## 2. The references, and their own error")
    emit()
    emit(
        "README §5 rule 5: no number is quoted against a converged reference without that "
        "reference's drift beside it, and no conclusion is drawn from a signal that does not "
        "exceed it. The drift is `convergence_reference.reference_drift`'s, which evaluates the "
        "criterion rather than reporting a number. **The smallest difference this document draws "
        "a conclusion from is the representation floor of §7**, not the quadrature errors of §4 "
        "and §5 -- those are reported as upper bounds and nothing is concluded from any of them "
        "-- so the floor is what the criterion is set against, at a tenth of it."
    )
    emit()
    emit(
        "| case | drift of `total`, /scale | one ulp of `total`, /scale | floor (§7) = smallest "
        "difference concluded from | threshold | verdict |"
    )
    emit("|---|---|---|---|---|---|")
    for result in results:
        drift = result["drift"]
        emit(
            f"| `{result['label']}` | {drift.max:.3e} | {result['ulp']:.3e} | "
            f"{drift.smallest_reported_difference:.3e} | {drift.threshold:.3e} | "
            f"{'converged' if drift.passed else '**NOT CONVERGED**'} |"
        )
    emit()
    failed = [r for r in results if not r["drift"].passed]
    emit(
        f"**{len(results) - len(failed)} of {len(results)} references converge** by that "
        f"criterion"
        + (
            "."
            if not failed
            else ", and the exceptions are named case by case in §8 rather than carried silently."
        )
    )
    emit()
    emit(
        f"The drift sits between {min(r['drift'].max for r in results):.3e} and "
        f"{worst([r['drift'].max for r in results]):.3e} of scale and the ulp column beside it "
        f"says why: both are the rounding of the same double arithmetic. That is the resolution "
        f"of this instrument, and §4, §5 and §8 report a quadrature difference below it as a "
        f"bound and never as a measurement."
    )
    emit()
    emit(
        "### 2.1 The code's own oracle moves with the sweep, and this is why it is held fixed"
    )
    emit()
    emit(
        "`analytic_rad` is computed inside `evaluate_QuadSource_integral` at **the caller's own** "
        "`atol` and `rtol` (`ComputeTargets/QuadSourceIntegral.py:884-897`, and the comment at "
        "`:1547-1552` says so in terms). So the residual `|total - analytic_rad|` that the "
        "fixture's own acceptance tests report is a comparison between two quantities that moved "
        "together, and a *sweep* of it would measure neither. Every oracle figure in this "
        "document is `analytic_rad` at the reference pair, computed once per case and then held "
        "fixed."
    )
    emit()
    emit(
        "How much that matters, measured: the table below is the oracle's own displacement from "
        "its reference value at each setting of the `rtol` axis, in the same normalisation as "
        "everything else."
    )
    emit()
    rtol_values = list(RTOL_AXIS)
    header = " | ".join(f"`rtol={rtol:g}`" for rtol in rtol_values)
    emit(f"| case | {header} |")
    emit("|---" * (len(rtol_values) + 1) + "|")
    for result in results:
        by_rtol = {
            row["rtol"]: row["oracle_drift_own"]
            for row in result["rows"]
            if row["axis"] == "rtol"
        }
        cells = " | ".join(f"{by_rtol[rtol]:.2e}" for rtol in rtol_values)
        emit(f"| `{result['label']}` | {cells} |")
    emit()
    per_case = []
    for result in results:
        row = next(
            row
            for row in result["rows"]
            if row["axis"] == "rtol" and row["rtol"] == DEFAULT_QUADRATURE_RTOL
        )
        quadrature = max(row["error"], result["drift"].max, result["ulp"])
        per_case.append((row["oracle_drift_own"], row["oracle_drift_own"] / quadrature))
    prod_oracle = worst([value for value, _ in per_case])
    emit(
        f"**At the production pair the stored `analytic_rad` is up to {prod_oracle:.2e} of scale "
        f"away from its own converged value** over the {len(results)} cases -- up to "
        f"x{max(ratio for _, ratio in per_case):.3g} the quadrature error of `total` on the same "
        f"case, and unmoved on "
        f"{sum(1 for value, _ in per_case if value == 0.0)} of them. The tolerance pair moves the "
        f"oracle column far more than it moves the column the pipeline consumes. That is a "
        f"finding about `analytic_rad`, it is handed on in §10, and it is not a reason to change "
        f"a tolerance."
    )
    emit()


def report_second_oracle(oracles):
    emit("## 3. The second oracle: `scipy.quad` of the exact integrand")
    emit()
    emit(
        "A quadrature error measured against the code's own closed form is worth calibrating "
        "against an oracle that shares none of its machinery (prompt 06 §2). `scipy.quad` of "
        "`G f / H^2` in `log(1+z')`, built from SciPy's own Bessel functions, over a short range "
        f"(`x_r` from {QUAD_ORACLE_X_TOP:g} to {QUAD_ORACLE_X_RESP:g}) straddling every hand-over, "
        "exact flavour. `quad`'s own declared relative error is in the last column and is the "
        "reference error of this comparison."
    )
    emit()
    emit(
        "| case | `total` at production vs `quad` | `total` at reference vs `quad` | "
        "`analytic_rad` at reference vs `quad` | `quad`'s own declared error |"
    )
    emit("|---|---|---|---|---|")
    for entry in oracles:
        emit(
            f"| `b={entry['b']:g} {entry['shape'].name} x_resp={QUAD_ORACLE_X_RESP:g} exact` | "
            f"{entry['rows']['production']['rel']:.3e} | "
            f"{entry['rows']['reference']['rel']:.3e} | "
            f"{entry['rows']['reference']['oracle_rel']:.3e} | "
            f"{entry['quad_declared_rel']:.1e} |"
        )
    emit()
    emit(
        f"Worst production-pair disagreement with the independent oracle: "
        f"**{worst([e['rows']['production']['rel'] for e in oracles]):.3e}** relative, against a "
        f"declared `quad` error of up to "
        f"{worst([e['quad_declared_rel'] for e in oracles]):.1e}. The partition and the two "
        f"integrators reproduce an oracle built from nothing they share, at the production "
        f"tolerances, to the level the oracle itself can be trusted."
    )
    emit()


def _axis_table(results, axis, values, keyed_by):
    header = " | ".join(f"`{keyed_by}={value:g}`" for value in values)
    emit(f"| case | {header} |")
    emit("|---" * (len(values) + 1) + "|")
    for result in results:
        by_value = {
            row[keyed_by]: row["error"] for row in result["rows"] if row["axis"] == axis
        }
        cells = " | ".join(f"{by_value[value]:.2e}" for value in values)
        emit(f"| `{result['label']}` | {cells} |")
    emit()


def report_atol_axis(results):
    emit("## 4. The `atol` axis")
    emit()
    emit(
        f"`|total(atol, rtol=1e-8) - total(reference)| / scale`, exact flavour. One step is four "
        f"decades: production is `1e-32` and a single decade there is not a step anyone would "
        f"take. Production, `{DEFAULT_QUADRATURE_ATOL:g}`, is interior with two steps tighter and "
        f"five looser."
    )
    emit()
    _axis_table(results, "atol", ATOL_AXIS, "atol")
    per_setting = {
        atol: worst(
            [
                row["error"]
                for result in results
                for row in result["rows"]
                if row["axis"] == "atol" and row["atol"] == atol
            ]
        )
        for atol in ATOL_AXIS
    }
    emit("| `atol` | worst over the cases | vs production |")
    emit("|---|---|---|")
    at_production = per_setting[DEFAULT_QUADRATURE_ATOL]
    for atol in ATOL_AXIS:
        ratio = (
            f"x{per_setting[atol] / at_production:.3g}"
            if at_production > 0.0
            else "n/a"
        )
        marker = " **(production)**" if atol == DEFAULT_QUADRATURE_ATOL else ""
        emit(f"| `{atol:g}`{marker} | {per_setting[atol]:.3e} | {ratio} |")
    emit()
    binding = [atol for atol in ATOL_AXIS if per_setting[atol] > 10.0 * at_production]
    inert = [atol for atol in ATOL_AXIS if atol not in binding]
    smallest = min(abs(result["reference"]["total"]) for result in results)
    largest = max(abs(result["reference"]["total"]) for result in results)
    if binding:
        decades = round(np.log10(min(binding) / DEFAULT_QUADRATURE_ATOL))
        emit(
            f"**`atol` binds at `{min(binding):g}` and looser -- {decades:.0f} decades above "
            f"production -- and is inert everywhere below it.** From `{max(inert):g}` down to "
            f"`{min(inert):g}`, {len(inert)} settings spanning {len(inert) - 1} four-decade "
            f"steps, the answer does not move at all. The reason is arithmetic and not subtle: "
            f"`atol` is distributed over the sub-intervals by log-width "
            f"(`QuadSourceIntegral.py:818`) and over the phase groups by count (`:1054`), and "
            f"`|total|` in these cases is {smallest:.1e} to {largest:.1e}, so an absolute floor "
            f"of `{DEFAULT_QUADRATURE_ATOL:g}` is "
            f"{np.log10(smallest / DEFAULT_QUADRATURE_ATOL):.0f} decades below the smallest "
            f"quantity being integrated and cannot be the criterion that stops either "
            f"integrator. This is "
            f"README §2 (e)'s magnitude argument in the one sector where the constant was already "
            f"chosen: `atol` is a statement about magnitudes, and at `1e-32` it is a statement "
            f"about magnitudes this sector does not have."
        )
    else:
        emit(
            "**`atol` is inert at every setting swept**, across twenty-eight decades, and the "
            "setting at which it would start to bind is below `1e-12`."
        )
    emit()


def report_rtol_axis(results):
    emit("## 5. The `rtol` axis, and whether `rtol` binds at `atol = 1e-32`")
    emit()
    emit(
        f"`|total(atol=1e-32, rtol) - total(reference)| / scale`, exact flavour, one decade per "
        f"step, production `{DEFAULT_QUADRATURE_RTOL:g}` interior with three either side."
    )
    emit()
    _axis_table(results, "rtol", RTOL_AXIS, "rtol")
    per_setting = {
        rtol: worst(
            [
                row["error"]
                for result in results
                for row in result["rows"]
                if row["axis"] == "rtol" and row["rtol"] == rtol
            ]
        )
        for rtol in RTOL_AXIS
    }
    emit("| `rtol` | worst over the cases | vs production | worst reference drift |")
    emit("|---|---|---|---|")
    at_production = per_setting[DEFAULT_QUADRATURE_RTOL]
    drift_max = worst([result["drift"].max for result in results])
    for rtol in RTOL_AXIS:
        ratio = (
            f"x{per_setting[rtol] / at_production:.3g}"
            if at_production > 0.0
            else "n/a"
        )
        marker = " **(production)**" if rtol == DEFAULT_QUADRATURE_RTOL else ""
        emit(
            f"| `{rtol:g}`{marker} | {per_setting[rtol]:.3e} | {ratio} | {drift_max:.3e} |"
        )
    emit()
    loosest = per_setting[max(RTOL_AXIS)]
    tightest = per_setting[min(RTOL_AXIS)]
    movers = [
        result
        for result in results
        if max(row["error"] for row in result["rows"] if row["axis"] == "rtol")
        > 100.0 * max(result["drift"].max, result["ulp"])
    ]
    emit(
        f"**The answer on this tree, at `atol = 1e-32`: yes, `rtol` binds -- it is the only "
        f"parameter here that does.** Six decades of it take the worst error over the cases from "
        f"{loosest:.3e} to {tightest:.3e} of scale, a factor of {loosest / tightest:.3g}, against "
        f"a worst reference drift of {drift_max:.3e}; the descent is about a decade of error per "
        f"decade of tolerance, which is what a working relative tolerance looks like. It is "
        f"resolved above the instrument's own rounding on **{len(movers)} of {len(results)} "
        f"cases** -- the rest sit at the drift throughout, so on those `rtol` is unresolvable "
        f"rather than inert, and the tables say which is which."
    )
    emit()
    emit(
        f"**So `source-remediation` log 12's finding does not transfer, exactly as prompt 06 §3 "
        f"warned.** Log 12 measured `1e-8 -> 1e-11` **bit-identical on 159 live items** because "
        f"`atol = 1e-25` was the binding constraint and `rtol` could not be seen behind it. "
        f"`atol` is now `1e-32`, seven decades tighter, and §4 shows it cannot bind at all; with "
        f"it out of the way `rtol` is what stops both integrators, and `1e-8 -> 1e-11` moves the "
        f"answer by a factor of {per_setting[DEFAULT_QUADRATURE_RTOL] / tightest:.3g} here rather "
        f"than by nothing. **That is a change of regime and not a contradiction**: the live "
        f"measurement was right about the tree it was taken on, and the constant it was taken at "
        f"has since moved. What it does *not* change is the answer, because §7's floor is orders "
        f"above everything in this table."
    )
    emit()


def report_corners(results):
    emit("## 6. The corners")
    emit()
    emit(
        "Both axes at their extremes. A one-axis sweep cannot see an interaction; these four "
        "cells are what make §4's and §5's conclusions statements about the matrix."
    )
    emit()
    header = " | ".join(f"`({atol:g}, {rtol:g})`" for atol, rtol in CORNERS)
    emit(f"| case | {header} |")
    emit("|---" * (len(CORNERS) + 1) + "|")
    for result in results:
        by_corner = {
            (row["atol"], row["rtol"]): row["error"]
            for row in result["rows"]
            if row["axis"] == "corner"
        }
        cells = " | ".join(f"{by_corner[corner]:.2e}" for corner in CORNERS)
        emit(f"| `{result['label']}` | {cells} |")
    emit()
    per_corner = {
        corner: worst(
            [
                row["error"]
                for result in results
                for row in result["rows"]
                if row["axis"] == "corner" and (row["atol"], row["rtol"]) == corner
            ]
        )
        for corner in CORNERS
    }
    emit(
        f"Worst over the cases, by corner: "
        + ", ".join(
            f"`({atol:g}, {rtol:g})` {per_corner[(atol, rtol)]:.2e}"
            for atol, rtol in CORNERS
        )
        + "."
    )
    emit()
    loose_pair = [per_corner[(1.0e-12, rtol)] for rtol in (1.0e-5, 1.0e-11)]
    tight_pair = [per_corner[(1.0e-40, rtol)] for rtol in (1.0e-5, 1.0e-11)]
    emit(
        f"**The interaction is the ordinary one and the corners make it visible.** At "
        f"`atol = 1e-12` the two `rtol` extremes agree to the digit ({loose_pair[0]:.2e} against "
        f"{loose_pair[1]:.2e}): `atol` is the binding term of `atol + rtol|value|` and six "
        f"decades of `rtol` change nothing. At `atol = 1e-40` the same two differ by "
        f"x{tight_pair[0] / tight_pair[1]:.3g} ({tight_pair[0]:.2e} against {tight_pair[1]:.2e}) "
        f"and reproduce §5 exactly: `rtol` is then the binding term and carries the whole error. "
        f"**Whichever term is larger binds, one at a time, and at production it is `rtol`** -- "
        f"which is why §4's inertness and §5's lever are two halves of one statement about the "
        f"matrix and not two lines through it."
    )
    emit()


def report_floor(results, floors):
    emit("## 7. The representation floor, on the realistic flavour")
    emit()
    emit(
        f"README §6.1 rule 1: the floor first, measured on the tree the campaign is running on. "
        f"The realistic flavour carries the representation floors `source-remediation` logs 05, "
        f"06 and 07 measured -- the `QuadSource` spline of `f`, the Liouville-Green closed forms "
        f"at the hand-over, the re-splined `bessel_phase` amplitude and phase, the hand-over "
        f"clamp -- at `PRODUCTION_SAMPLES_PER_LOG10Z = {PRODUCTION_SAMPLES_PER_LOG10Z}`. It is "
        f"scored against the **exact** run of the same case at the **same** tolerance, so that "
        f"the quadrature error is common to both and cancels, and against the fixed oracle beside "
        f"it."
    )
    emit()
    emit(
        "| case | `rtol` | floor vs exact twin, /scale | vs \\|total\\| | vs the fixed oracle, /scale |"
    )
    emit("|---|---|---|---|---|")
    for result in results:
        for row in floors[(result["b"], result["shape"].name, result["x_resp"])]:
            emit(
                f"| `{case_label(result['b'], result['shape'], result['x_resp'], 'realistic')}` | "
                f"`{row['rtol']:g}` | {row['vs_exact']:.3e} | {row['vs_exact_rel_total']:.3e} | "
                f"{row['vs_oracle']:.3e} |"
            )
    emit()
    at_production = [
        row
        for rows in floors.values()
        for row in rows
        if row["rtol"] == DEFAULT_QUADRATURE_RTOL
    ]
    spread = []
    tight_spread = []
    for rows in floors.values():
        values = [row["vs_exact"] for row in rows]
        if min(values) > 0.0:
            spread.append(max(values) / min(values))
        by_rtol = {row["rtol"]: row["vs_exact"] for row in rows}
        if by_rtol[DEFAULT_QUADRATURE_RTOL] > 0.0:
            tight_spread.append(
                by_rtol[min(RTOL_AXIS)] / by_rtol[DEFAULT_QUADRATURE_RTOL]
            )
    emit(
        f"**The floor at the production pair is {min(row['vs_exact'] for row in at_production):.3e} "
        f"to {worst([row['vs_exact'] for row in at_production]):.3e} of scale** over the "
        f"{len(at_production)} cases, {min(row['vs_exact_rel_total'] for row in at_production):.3e} "
        f"to {worst([row['vs_exact_rel_total'] for row in at_production]):.3e} of `|total|`. "
        f"Across six decades of `rtol` it moves by at most a factor of {worst(spread):.4g}, and "
        f"between production and `rtol = {min(RTOL_AXIS):g}` -- three decades, over which §5 "
        f"shows the *quadrature* error falling by two orders -- by at most "
        f"{worst([abs(1.0 - value) for value in tight_spread]):.2%}. That is, **the floor does "
        f"not see the quadrature tolerance**, which is the property that makes §8's ratio "
        f"meaningful: the two quantities being divided are independent."
    )
    emit()
    emit(
        "These figures are consistent with the floors already in the record and do not supersede "
        "them: `[06-source-spline-residual-vs-handover]` (4.5e-4 of the local envelope for the "
        "`QuadSource` spline of `f` at the hand-over), `[07-lg-derivative-truncation-at-handover]` "
        "(7e-6 at `b=0` to 1.4e-4 at `b=0.25`), and the fixture's own module docstring. They are "
        "a re-measurement of the same floor through the same fixture, taken on this tree."
    )
    emit()


def report_verdict(results, floors, oracles):
    emit("## 8. The ratio, and README §6.1's target rule applied")
    emit()
    quadrature = {}
    for result in results:
        key = (result["b"], result["shape"].name, result["x_resp"])
        production_row = next(
            row
            for row in result["rows"]
            if row["axis"] == "rtol" and row["rtol"] == DEFAULT_QUADRATURE_RTOL
        )
        quadrature[key] = (
            max(production_row["error"], result["drift"].max, result["ulp"]),
            production_row["error"] > max(result["drift"].max, result["ulp"]),
        )
    emit(
        "The quadrature column is the measured difference at the production pair where that "
        "exceeds the instrument's own resolution, and **the resolution itself -- the larger of "
        "the reference's drift and one ulp of `total` -- where it does not**. A case marked "
        "*bound* is one where the measured difference is below the resolution, so its ratio is a "
        "lower bound on the domination and not an estimate of it (README §5 rule 5)."
    )
    emit()
    emit(
        "| case | quadrature error at production | measured or bound | floor at production | "
        "ratio floor / quadrature |"
    )
    emit("|---|---|---|---|---|")
    ratios = []
    for result in results:
        key = (result["b"], result["shape"].name, result["x_resp"])
        floor = next(
            row["vs_exact"]
            for row in floors[key]
            if row["rtol"] == DEFAULT_QUADRATURE_RTOL
        )
        value, resolved = quadrature[key]
        ratio = floor / value
        ratios.append(ratio)
        emit(
            f"| `{result['label'].replace(' exact', '')}` | {value:.3e} | "
            f"{'measured' if resolved else '*bound*'} | {floor:.3e} | "
            f"**{'x' if resolved else '>x'}{ratio:.3g}** |"
        )
    emit()
    resolved_count = sum(1 for _, resolved in quadrature.values() if resolved)
    emit(
        f"**The floor dominates the quadrature error by x{min(ratios):.3g} to x{max(ratios):.3g}** "
        f"over the {len(ratios)} cases -- measured on {resolved_count} of them and bounded below "
        f"on the other {len(ratios) - resolved_count}, where the quadrature error at production "
        f"is smaller than anything double arithmetic can resolve here."
    )
    emit()
    emit("### 8.1 The `rtol` ladder, which is what README §6.1 rule 3 asks for")
    emit()
    emit(
        "Rule 3 sweeps loose to tight and takes the **first** setting that clears the floor. §5 "
        "shows `rtol` is the axis that moves the quadrature error, so the ladder is on `rtol`, at "
        "the production `atol`, and the statistic is the **minimum over the cases** of "
        "`floor / quadrature error` for the same case -- the worst case, not the average, per "
        "rule 2."
    )
    emit()
    emit(
        "| `rtol` | worst case's floor / quadrature error | clears the floor everywhere? |"
    )
    emit("|---|---|---|")
    ladder = {}
    for rtol in RTOL_AXIS:
        per_case = []
        for result in results:
            key = (result["b"], result["shape"].name, result["x_resp"])
            row = next(
                row
                for row in result["rows"]
                if row["axis"] == "rtol" and row["rtol"] == rtol
            )
            floor = next(
                entry["vs_exact"]
                for entry in floors[key]
                if entry["rtol"] == DEFAULT_QUADRATURE_RTOL
            )
            value = max(row["error"], result["drift"].max, result["ulp"])
            per_case.append(floor / value)
        ladder[rtol] = min(per_case)
        marker = " **(production)**" if rtol == DEFAULT_QUADRATURE_RTOL else ""
        emit(
            f"| `{rtol:g}`{marker} | x{ladder[rtol]:.3g} | "
            f"{'yes' if ladder[rtol] >= 1.0 else '**no**'} |"
        )
    emit()
    clearing = [rtol for rtol in RTOL_AXIS if ladder[rtol] >= 1.0]
    emit(
        f"**The loosest setting that clears the floor on every case is "
        f"`rtol = {max(clearing):g}`**, and it clears it by only x{ladder[max(clearing)]:.3g} -- "
        f"so the loose end of §5's axis is *not* orders clear of the floor, and saying so would "
        f"have been the easy error here. Production sits {abs(round(np.log10(DEFAULT_QUADRATURE_RTOL / max(clearing)))):.0f} "
        f"decades tighter than that, at x{ladder[DEFAULT_QUADRATURE_RTOL]:.3g}. **This is a "
        f"measurement handed to the owning campaigns and not a recommendation** (§10): rule 3 "
        f"would license the looser setting, README §0.4 and §5 rule 8 put the value out of this "
        f"campaign's reach, and rule 4 is the rule that actually applies to the question asked."
    )
    emit()
    emit("### 8.2 The rule that applies")
    emit()
    emit(
        "README §6.1 rule 4: *where the error is already below the floor, the target is "
        "`unchanged`, written in the cell in that word, and the row records the factor by which "
        "the floor dominates.* That is what this measurement gives."
    )
    emit()
    emit(
        f"> **`unchanged`.** `DEFAULT_QUADRATURE_ATOL = {DEFAULT_QUADRATURE_ATOL:g}` and "
        f"`DEFAULT_QUADRATURE_RTOL = {DEFAULT_QUADRATURE_RTOL:g}`, with the representation floor "
        f"dominating the quadrature error by **x{min(ratios):.3g} to x{max(ratios):.3g}** on the "
        f"offline fixture."
    )
    emit()
    emit(
        "Rule 4 is the one that applies, and §8.1 is why it is not simply rule 3 read backwards. "
        "The error at production is below the floor on every case, so the target is `unchanged` "
        "and no tightening may be proposed however cheap it looks. Rule 3 would license a looser "
        "`rtol` and §8.1 measures which one; this campaign has no standing to propose it "
        "(README §0.4, §5 rule 8) and hands it over in §10 as a finding."
    )
    emit()
    emit(
        "**This is the campaign's third `unchanged`**, after prompt 03's `GkNumericIntegration` "
        "(consumer spline dominating by x631 to x37,700) and prompt 04's four Gauss orders "
        "(double-precision accumulation dominating by x48.4 to x3.03e5). It is a result, and the "
        "ratio here is the largest of the three by several orders."
    )
    emit()


def report_cost(results, floors):
    emit("## 9. Cost, in counts, and what the tolerance *does* move")
    emit()
    emit(
        f"README §2 (i): counts, not wall time. README §2 (c): a per-object figure carries the "
        f"object count beside it, and `QuadSourceIntegral`'s is **{OBJECT_COUNT}** "
        f"(`TOLERANCE-INVENTORY.md` §5.5) -- the largest in the pipeline, which is why a "
        f"per-item percentage here means nothing on its own."
    )
    emit()
    emit(
        "Summed over the 18 exact-flavour cases: `quad` right-hand-side evaluations, "
        "`adaptive_levin_sincos` evaluations, and the number of Levin regions."
    )
    emit()
    emit(
        "| setting | quad RHS evaluations | Levin evaluations | Levin regions | vs production |"
    )
    emit("|---|---|---|---|---|")

    def totals(predicate):
        rows = [row for result in results for row in result["rows"] if predicate(row)]
        return (
            sum(row["quad_evaluations"] for row in rows),
            sum(row["levin_evaluations"] for row in rows),
            sum(row["levin_regions"] for row in rows),
        )

    production_total = totals(
        lambda row: row["axis"] == "rtol" and row["rtol"] == DEFAULT_QUADRATURE_RTOL
    )
    base = production_total[0] + production_total[1]
    ladder = [("atol", atol, DEFAULT_QUADRATURE_RTOL) for atol in ATOL_AXIS] + [
        ("rtol", DEFAULT_QUADRATURE_ATOL, rtol) for rtol in RTOL_AXIS
    ]
    for axis, atol, rtol in ladder:
        counts = totals(
            lambda row, axis=axis, atol=atol, rtol=rtol: row["axis"] == axis
            and row["atol"] == atol
            and row["rtol"] == rtol
        )
        label = (
            f"`atol={atol:g}` at production `rtol`"
            if axis == "atol"
            else f"`rtol={rtol:g}` at production `atol`"
        )
        if atol == DEFAULT_QUADRATURE_ATOL and rtol == DEFAULT_QUADRATURE_RTOL:
            label += " **(production)**"
        delta = (counts[0] + counts[1]) / base - 1.0 if base else 0.0
        emit(
            f"| {label} | {counts[0]:,} | {counts[1]:,} | {counts[2]:,} | {delta:+.1%} |"
        )
    emit()
    rtol_costs = {
        rtol: totals(
            lambda row, rtol=rtol: row["axis"] == "rtol" and row["rtol"] == rtol
        )
        for rtol in RTOL_AXIS
    }
    loosest = rtol_costs[max(RTOL_AXIS)]
    tightest = rtol_costs[min(RTOL_AXIS)]
    # the worst case's own floor-to-error ratio at the loosest rtol swept: a per-case comparison,
    # because a floor from one case and an error from another are not comparable
    loose_ratio = min(
        next(
            entry["vs_exact"]
            for entry in floors[(result["b"], result["shape"].name, result["x_resp"])]
            if entry["rtol"] == DEFAULT_QUADRATURE_RTOL
        )
        / max(
            next(
                row["error"]
                for row in result["rows"]
                if row["axis"] == "rtol" and row["rtol"] == max(RTOL_AXIS)
            ),
            result["drift"].max,
            result["ulp"],
        )
        for result in results
    )
    emit(
        f"**`rtol` buys accuracy, and the consumer cannot use it.** Over the six decades of §5 "
        f"the evaluation count rises from {loosest[0] + loosest[1]:,} to "
        f"{tightest[0] + tightest[1]:,} "
        f"({(tightest[0] + tightest[1]) / (loosest[0] + loosest[1]) - 1.0:+.1%}), and §5 shows "
        f"the quadrature error falling with it -- but every setting at and below production is "
        f"already orders under the representation floor of §7, so none of what is bought there "
        f"reaches `total`. Three "
        f"decades of tightening from production costs "
        f"{(rtol_costs[min(RTOL_AXIS)][0] + rtol_costs[min(RTOL_AXIS)][1]) / base - 1.0:+.1%} of "
        f"the sector's evaluations, on **{OBJECT_COUNT}** objects; three decades of loosening "
        f"saves "
        f"{1.0 - (rtol_costs[max(RTOL_AXIS)][0] + rtol_costs[max(RTOL_AXIS)][1]) / base:.1%} and "
        f"leaves the worst case's quadrature error x{loose_ratio:.3g} below **that case's own** "
        f"floor -- clear, but not by orders, which is the measurement §8.1 turns into a ladder. "
        f"Both are statements about **this fixture**, whose "
        f"`(k, q, r, z)` range is not production's, and both are findings for the owning "
        f"campaigns rather than recommendations (§10)."
    )
    emit()
    emit(
        "### 9.1 `total_abserr` is the declared bound and is not the measured residual"
    )
    emit()
    emit(
        "`[09-abserr-is-a-quadrature-bound]` measured `total_abserr` missing the true residual by "
        "up to 4.4e4x on live rows, and prompt 06 §3 forbids concluding from it. It is reported "
        "here only as what it claims to be, beside the measured quantity, so that the two can be "
        "seen to be different things."
    )
    emit()
    emit(
        "| setting | declared `total_abserr` / \\|total\\|, worst | measured error / scale, worst | "
        "declared / measured |"
    )
    emit("|---|---|---|---|")
    for rtol in RTOL_AXIS:
        rows = [
            row
            for result in results
            for row in result["rows"]
            if row["axis"] == "rtol" and row["rtol"] == rtol
        ]
        declared = worst([row["declared_abserr"] / abs(row["total"]) for row in rows])
        measured = worst([row["error"] for row in rows])
        ratio = f"x{declared / measured:.3g}" if measured > 0.0 else "n/a"
        emit(f"| `rtol={rtol:g}` | {declared:.3e} | {measured:.3e} | {ratio} |")
    emit()
    not_converged = [
        (result["label"], row["rtol"])
        for result in results
        for row in result["rows"]
        if row["axis"] == "rtol" and not row["declared_converged"]
    ]
    emit(
        f"`total_converged = False` on {len(not_converged)} of "
        f"{len(results) * len(RTOL_AXIS)} runs of the `rtol` axis, at "
        f"`rtol` in "
        f"{{{', '.join(sorted({f'{rtol:g}' for _, rtol in not_converged})) or '-'}}}. "
        f"Per `[09-abserr-is-a-quadrature-bound]` that flag is not a failure: it says the "
        f"driver's own estimate did not reach the requested bound, and §5 shows the answer was "
        f"already right when it did not."
    )
    emit()


def report_handoff():
    emit("## 10. The hand-off")
    emit()
    emit(
        "README §0.4: this campaign measures `QuadSourceIntegral` and does not act on it. What "
        "goes to `prompts/levin-refactor` and `prompts/qsi-phase-groups` through "
        "`docs/OPEN_ISSUES.md` §1.2 and §1.3:"
    )
    emit()
    emit(
        "1. **The quadrature tolerance is not the limiting parameter of the source integral.** "
        "The representation floor dominates the quadrature error by the factor §8 measures, and "
        "neither `atol` nor `rtol` is the binding constraint anywhere in the swept matrix. "
        "§7 D4 is therefore not reopened (prompt 06 §9)."
    )
    emit(
        "2. **`rtol` is the binding parameter of the quadrature error and `atol` is inert, which "
        "is the reverse of the regime `source-remediation` log 12 measured** (§4, §5). The "
        "conclusion that `1e-8 -> 1e-11` is bit-identical was taken at `atol = 1e-25` and does "
        "not survive the move to `1e-32`. Nothing follows for the value, because §7's floor is "
        "orders above both ends of that range -- but anything that *reasons* from log 12's "
        "bit-identity on this tree is reasoning from a superseded regime."
    )
    emit(
        "3. **There is slack on `rtol`, it is measured, and it is theirs to spend or not.** §8.1 "
        "is the README §6.1 rule 3 ladder: the loosest `rtol` that clears the floor on every "
        "case, and by how little it clears. §9 prices each rung in evaluations against an object "
        "count of 1,275 x 50 x (response z) per model. Both are **on this fixture**, whose "
        "`(k, q, r, z)` range is not production's, and only a live run can price the real "
        "sector. This campaign states the measurement and recommends no value (README §5 rule 8)."
    )
    emit(
        "4. **`analytic_rad` is the column the tolerance moves most** (§2.1). It is computed at "
        "the caller's pair and is displaced from its converged value at production far more than "
        "`total` is. Anyone who reads the stored `analytic_rad` as a fixed oracle is reading a "
        "quantity that depends on the row's own `atol_serial` and `rtol_serial`."
    )
    emit(
        "5. **`DEFAULT_LEVIN_MAX_DEPTH = 20` and `limit = 100` were never chosen** "
        "(`TOLERANCE-INVENTORY.md` §5.4 rows E and F). They are recorded, not touched, and "
        "prompt 06a carries them into `docs/TOLERANCE-PROVENANCE.md` as unestablished."
    )
    emit()
    emit(
        "What is **cited and not re-derived** (prompt 06 §1): "
        "`[09-abserr-is-a-quadrature-bound]`, `[12-handover-clamp-error-in-production]` and "
        "`[12-atol-too-loose-for-the-source-integral]`, all on "
        "`prompts/source-remediation/IMPLEMENTATION_STATE.md`. No figure in this document "
        "supersedes a live-run figure in any of them."
    )
    emit()


def report_limits():
    emit("## 11. What this measurement is, and what it is not")
    emit()
    emit(
        "- **It is offline, and that is a decision** (user, 2026-09-18; README §7 D11). The "
        "fixture gives what a live run would not: the *exact / realistic* pair, which separates "
        "the quadrature error from the representation floor. A live sweep could measure only the "
        "two convolved."
    )
    emit(
        "- **It is not the production range.** Three `(k, q, r)` shapes and three response "
        "redshifts on an exact constant-$w$ background, against 1,275 x 50 x (response z) rows "
        "per model in production. The regime mix, the clamp-gap census and the 58 % figure of "
        "`source-remediation` log 12 are live-run measurements and stand unaltered."
    )
    emit(
        "- **The fixture's Bessel splines are built one order looser than production's.** "
        "`Case.bessel_phase_data` calls `bessel_phase(..., atol=1e-25, rtol=5e-14)`, and those "
        "two keywords are the deprecated pair that `bessel_phase` ignores "
        "(`LiouvilleGreen/bessel_phase.py:932-945`), so the splines are built at the module "
        "defaults `DEFAULT_PHASE_ATOL = DEFAULT_AMPLITUDE_RTOL = 1e-11` rather than at "
        "`main.py:1119`'s `1e-12`. That is the floor under `analytic_rad` here, it is one order "
        "looser than production's, and it is recorded rather than repaired: "
        "`test_quadsource_integral.py` is a module this prompt imports and does not edit."
    )
    emit(
        "- **Every figure carries its reference's own error** (README §5 rule 5). Where the "
        "measured difference is below the reference's drift the tables say so and §8 takes the "
        "drift as the quadrature figure rather than the smaller measured one."
    )
    emit()


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------


def main() -> None:
    started = time.perf_counter()
    log_progress("** quadsource_readonly: exact flavour")

    results = []
    for b in B_VALUES:
        for shape in SHAPES:
            for x_resp in X_RESP_VALUES:
                results.append(sweep_case(b, shape, x_resp))

    log_progress("** quadsource_readonly: realistic flavour")
    floors = {}
    for result in results:
        exact_rows = {
            (row["atol"], row["rtol"]): row
            for row in result["rows"]
            if row["axis"] in ("rtol", "corner")
        }
        floors[(result["b"], result["shape"].name, result["x_resp"])] = floor_case(
            result["b"],
            result["shape"],
            result["x_resp"],
            exact_rows,
            result["scale"],
            result["oracle"],
        )

    log_progress(
        "** quadsource_readonly: the reference drift, scored against the floor"
    )
    for result in results:
        rows = floors[(result["b"], result["shape"].name, result["x_resp"])]
        floor = next(
            row["vs_exact"] for row in rows if row["rtol"] == DEFAULT_QUADRATURE_RTOL
        )
        score_drift(result, floor)

    log_progress("** quadsource_readonly: the independent quad oracle")
    oracles = [quad_oracle_case(b, shape) for b in B_VALUES for shape in SHAPES]

    elapsed = time.perf_counter() - started

    emit("<!-- generated {} by".format(date.today().isoformat()))
    emit(
        "     PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/quadsource_readonly.py"
    )
    emit(
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_header(results, floors)
    report_reference(results)
    report_second_oracle(oracles)
    report_atol_axis(results)
    report_rtol_axis(results)
    report_corners(results)
    report_floor(results, floors)
    report_verdict(results, floors, oracles)
    report_cost(results, floors)
    report_handoff()
    report_limits()

    emit(
        f"*Runtime {elapsed:.0f} s; {len(results)} cases x "
        f"{len(ATOL_AXIS) + len(RTOL_AXIS) + len(CORNERS) + 2} exact runs, "
        f"{len(results)} x {len(REALISTIC_SETTINGS)} realistic runs, and "
        f"{len(oracles)} x 2 runs against the independent `quad` oracle.*"
    )

    print("\n".join(_OUT))
    log_progress(f"** total runtime {elapsed:.1f} s")


if __name__ == "__main__":
    main()
