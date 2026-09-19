"""
Gauss-order convergence of the WKB primitives, on both production background models
(``prompts/GkTk-remedial/`` prompt 02).

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/residual_convergence.py

The review's design (§7) builds every primitive by a **fixed-order Gauss-Legendre rule per
production interval** in ``u = log(1+z)``.  That is measured at the double-precision floor for
``tau`` on LambdaCDM (review §7) and unmeasured everywhere else.  Review §11 calls the test this
script performs "the first test of any implementation":

    "The QCD model's H(z) comes from a spline of T(z) whose knots need not align with the
    production grid, so a fixed Gauss rule for rho could converge slowly across them."

What is measured, for ``LambdaCDMModel`` and ``QCDModel`` on the production source grid:

1. per-interval relative error of the fixed-order increment, at orders 2, 4, 6, 8, 12, 16,
   against a converged adaptive reference, for the five integrands

       1/H,  c_s/H,  (3/2)(1 + c_s^2),  C/(omega + k/H),  C_T/(omega_T + k c_s/H)

   the last two for k = 1e5, 1e7, 3e8 /Mpc (all in ``u = log(1+z)``, so the ``1/(1+z)`` of the
   friction integrand and the ``dz = (1+z) du`` of the others are already folded in);
2. the cumulative error at the ``wkb_reference_data.json`` checkpoints, per order -- the quantity
   the tables actually deliver;
3. where the QCD model's features sit on the production grid: the ``QCD_EOS`` branch boundaries,
   the ``T(z)``-spline knots, the ``epsilon`` departure at the QCD transition and the ``c_s^2``
   transition;
4. the smoothness of ``rho_G``, ``rho_T`` -- ``max|d2 rho/du2|``, ``max|d4 rho/du4|`` and the
   cubic-spline interpolation error ``h^4 max|rho''''|/384`` those imply for prompts 09 and 10;
5. build cost -- integrand evaluations and wall time -- at the recommended orders.

and, as controls, the exact-radiation identities on ``RadiationModel``: ``rho_G`` bit-exactly zero
at every order, and ``rho_T`` against its closed form.

Three build schemes are measured side by side on QCD:

* **plain** -- one order-N Gauss panel per production interval, the review's design as written;
* **branch** -- the same rule, but a production interval containing one of the three ``QCD_EOS``
  temperature break points is split there and each sub-panel integrated separately;
* **branch+knots** -- also split at the interior knots of the model's own ``T(z)`` spline.

Every break point is a property of the cosmology, known in closed form before any integration;
nothing is detected at run time.  **The converged reference is itself computed break-aware**, at
the finest break set the model offers, so that the per-interval error of the ``plain`` scheme
across a discontinuity is measured against a value that is actually right.

Outputs a ``"convergence"`` block appended to
``ComputeTargets/tests/wkb_reference_data.json``, and -- **only under ``--legacy-markdown``** --
``docs/gktk-remedial/RESIDUAL-CONVERGENCE.md``.

**Re-run 2026-09-17 by ``prompts/tolerance-convergence`` prompt 04** (README §7 **D5**), which is
the first prompt anywhere allowed to write the fixture this script generates. Four things changed
and every one of them is a change to *what is recorded*, not to what is measured:

1. **``branch+knots`` is a control, not a candidate.** ``integration_break_points`` has not
   returned the ``T(z)`` spline's knots since ``qcd-background-audit`` prompt 07, so production
   cannot execute that scheme and it may not be recommended. It stays in the sweep -- the script
   takes the knots off the model's own spline rather than off the cosmology's contract, and two
   test modules index the block by that key -- and the question it now answers is *what does
   splitting at the knots still buy?*
2. **Every figure carries the reference's own drift** (README §5 rule 5), measured through
   ``ComputeTargets/tests/convergence_reference.reference_drift`` -- the campaign's one
   implementation of the convergence test -- as the movement of the adaptive reference under one
   decade of ``epsrel``, with the criterion evaluated against the smallest difference the block
   goes on to report.
3. **Every figure carries its source-grid generation** (README §5 rule 6). This script measures
   on the **version-0** grid and must continue to: the quantity it scores, the JSON's own
   ``checkpoints`` / ``rho_*`` reference values, lives on that grid, and the two test modules that
   read the block compare against it there. The generation is now recorded rather than implied.
4. **The markdown output is opt-in.** ``docs/gktk-remedial/RESIDUAL-CONVERGENCE.md`` is
   ``GkTk-remedial`` prompt 02's published document and verification documents are additive
   (README §5 rule 7), so it is written only when ``--legacy-markdown`` is passed. The re-run's
   own tables are in ``docs/tolerance-convergence/ORDER-AUDIT.md``.

Options:

* ``--json-out PATH`` -- write the ``convergence`` block into ``PATH`` instead of the fixture
  (used for the dry run prompt 04 §2.4 requires: *look, then regenerate*);
* ``--no-json`` -- measure and print, write nothing;
* ``--legacy-markdown`` -- also rewrite ``RESIDUAL-CONVERGENCE.md``.
"""

import argparse
import json
import os
import platform
import sys
import time
import warnings
from datetime import date
from math import fsum, log, log1p, expm1, fabs

import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference_lib as R  # noqa: E402

from ComputeTargets.tests.convergence_reference import (  # noqa: E402
    CRITERION_RATIO,
    TolerancePair,
    reference_drift,
)
from ComputeTargets.tests.wkb_reference import (  # noqa: E402
    SOURCE_GRID_V0,
    LambdaCDMModel,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    QCDModel,
    RadiationModel,
    REFERENCE_DATA_PATH,
    REFERENCE_K_VALUES,
    horizon_exit_z,
    load_references,
    production_source_grid,
)
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS  # noqa: E402
from Units import Mpc_units  # noqa: E402

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

ORDERS = (2, 4, 6, 8, 12, 16)

# the converged adaptive reference, per (sub-)panel
QUAD_EPSREL = 1.5e-14
QUAD_CROSSCHECK_EPSREL = 1.0e-12
# One decade *looser*, for the reference's own drift (README §5 rule 5). It is on this side
# because `quad` refuses an `epsrel` below 50*eps = 1.11e-14 when `epsabs = 0`: QUAD_EPSREL is
# already the tightest attainable setting and there is no decade below it to step to. What the
# drift then measures is the movement across the last decade the method admits -- see
# `reference_case_drift`.
QUAD_DRIFT_EPSREL = 1.5e-13
# the independent per-panel cross-check on that reference
GAUSS_CROSSCHECK_RTOL = 1.0e-14
GAUSS_CROSSCHECK_MAX_LEVEL = 8

# the anchor of every residual reference in the JSON: 3 e-folds inside the horizon
RHO_ANCHOR_EFOLDS_SUBH = 3

# prompt 02 §3: N_rho is the smallest order reaching this cumulative absolute error, a decade of
# margin below README §6's 1e-6 rad target for rho
RHO_TARGET_RAD = 1.0e-7

# prompt 02 §4: ...and simultaneously reproducing the exact-radiation control for rho_T to this
# relative accuracy. The two criteria are not redundant. rho is 1e-3 rad on QCD and 1e-10 rad on
# LambdaCDM, so an *absolute* 1e-7 rad target is met by orders that have not converged at all --
# order 2 passes it on every scheme while carrying 7e-11 relative error on a control whose answer
# is known in closed form. Requiring both is what the prompt asks for and what separates a
# converged rule from a lucky one.
RHO_RADIATION_CONTROL_RTOL = 1.0e-14

# prompt 02 §3: N_tau / N_cs_tau / N_F are the smallest orders within this factor of the floor
FLOOR_FACTOR = 3.0

# an interval whose fixed-order increment is worse than this is counted as "offending"
# (the threshold prompt 01's log used, so the two are comparable)
INTERVAL_REPORT_THRESHOLD = 1.0e-12

# k at which a relative error of an increment is also quoted as a phase error
PHASE_K = 3.0e8

# schemes in increasing order of cost; the recommendation is the cheapest that works
SCHEME_ORDER = ("plain", "branch", "branch+knots")

# The schemes production can actually execute. `integration_break_points` returns the three
# equation-of-state temperature crossings and nothing else (qcd-background-audit prompt 07), so a
# panel edge at a T(z) spline knot is a thing this script can build and the pipeline cannot. A
# scheme production cannot execute may not be recommended (tolerance-convergence prompt 04 §3.2).
CANDIDATE_SCHEMES = ("plain", "branch")

# ...and the one that is retained as a control. It is kept in `schemes` and in the block because
# ComputeTargets/tests/test_background_tau.py and test_background_cs_tau_friction.py index
# `models.QCDModel["branch+knots"]` by name for the reference floor they assert against; dropping
# the key is a two-module edit outside prompt 04's carve-out.
CONTROL_SCHEME = "branch+knots"

# The source-grid generation everything here is measured on (README §5 rule 6). It is version 0
# deliberately and must stay there: the quantity scored is the agreement of a fixed-order rule
# with the JSON's own `checkpoints` / `rho_*` reference values, and those live on the version-0
# grid, as do the two test modules that compare against them.
GRID_GENERATION = SOURCE_GRID_V0

MARKDOWN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "RESIDUAL-CONVERGENCE.md"
)

_GAUSS_CACHE = {}


def gauss_rule(order: int):
    if order not in _GAUSS_CACHE:
        _GAUSS_CACHE[order] = np.polynomial.legendre.leggauss(order)
    return _GAUSS_CACHE[order]


def gauss_panel(f, a: float, b: float, order: int) -> float:
    x, w = gauss_rule(order)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return half * fsum(wi * f(mid + half * xi) for xi, wi in zip(x, w))


def gauss_interval(f, a: float, b: float, order: int, breaks=()) -> float:
    """Order-``order`` Gauss over ``[a, b]``, split at any supplied interior break point."""
    if len(breaks) == 0:
        return gauss_panel(f, a, b, order)
    edges = [a, *breaks, b]
    return fsum(
        gauss_panel(f, edges[i], edges[i + 1], order) for i in range(len(edges) - 1)
    )


def k_key(k: float) -> str:
    return f"{k:.6e}"


class CountingIntegrand:
    """Wrap an integrand in ``u`` and count the calls (one background evaluation each)."""

    def __init__(self, f):
        self._f = f
        self.calls = 0

    def __call__(self, u):
        self.calls += 1
        return self._f(u)

    def reset(self):
        self.calls = 0


# ---------------------------------------------------------------------------------------------
# the integrands under test
# ---------------------------------------------------------------------------------------------

# (key, human label, maker, JSON cumulative field, sign relating that field to +int)
PRIMITIVE_SPECS = (
    ("tau", "1/H", R.integrand_tau_double, "tau_minus_top", +1),
    ("cs_tau", "c_s/H", R.integrand_cs_tau_double, "cs_tau_minus_top", +1),
    (
        "friction",
        "(3/2)(1 + c_s^2)/(1+z)",
        R.integrand_friction_double,
        "friction_F_minus_top",
        -1,
    ),
)

RESIDUAL_SPECS = (
    ("rho_G", "C/(omega + k/H)", R.integrand_rho_G_double),
    ("rho_T", "C_T/(omega_T + k c_s/H)", R.integrand_rho_T_double),
)

PRIMITIVE_KEYS = tuple(spec[0] for spec in PRIMITIVE_SPECS)


def residual_keys():
    return tuple(
        f"{key}@{k_key(k)}" for key, _, _ in RESIDUAL_SPECS for k in REFERENCE_K_VALUES
    )


def integrand_keys():
    return (*PRIMITIVE_KEYS, *residual_keys())


# ---------------------------------------------------------------------------------------------
# break points of the QCD background
# ---------------------------------------------------------------------------------------------

# Every temperature at which a QCD_EOS branch changes, and the quantity it breaks.
#
#   QCD_EOS.G / Gs switch between the Saikawa-Shirai fit and an asymptotic constant at T_LO,
#   T_120_MEV and T_HI, and the pieces do not match, so rho_r -- and therefore H(z) -- is
#   discontinuous there;
#   QCD_EOS.w clamps its argument to EOS_T_LO below EOS_T_LO, so wPerturbations (the campaign's
#   c_s^2) has a break there, which reaches c_s/H, the friction integrand and rho_T.
#
# EOS_T_LO is *not* one of the two boundaries prompt 01 found: it is a third break point in range,
# above T_LO, that only the c_s^2-dependent integrands see, and it is a kink rather than a jump
# (measured: the value of c_s^2 is continuous to 7e-15, its u-derivative is not).
QCD_BREAK_TEMPERATURES_GEV = (
    ("T_LO", QCD_EOS.T_LO, "G(T), Gs(T) -> H(z)"),
    ("EOS_T_LO", QCD_EOS.EOS_T_LO, "w(T) -> c_s^2"),
    ("T_120_MEV", QCD_EOS.T_120_MEV, "G(T), Gs(T) -> H(z)"),
    ("T_HI", QCD_EOS.T_HI, "G(T), Gs(T) -> H(z)"),
)


def qcd_branch_boundary_u(cosmology, T_in_GeV: float, u_lo: float, u_hi: float):
    """
    The point in ``u = log(1+z)`` at which the model's ``T_photon(z)`` crosses ``T_in_GeV``, or
    ``None`` when the crossing lies outside ``[u_lo, u_hi]``.
    """
    target = T_in_GeV * Mpc_units().GeV

    def q(u: float) -> float:
        return log(cosmology.T_photon(expm1(u))) - log(target)

    if q(u_lo) * q(u_hi) >= 0.0:
        return None
    root = root_scalar(q, bracket=(u_lo, u_hi), xtol=1e-15, rtol=1e-15)
    if not root.converged:
        raise RuntimeError(f"qcd_branch_boundary_u: no root for T = {T_in_GeV:g} GeV")
    return float(root.root)


def qcd_break_points(cosmology, u_lo: float, u_hi: float):
    """
    Every point in ``[u_lo, u_hi]`` at which the ``QCD_Cosmology`` integrands lose smoothness.

    :return: ``(branch, knots)`` -- a list of ``(name, T_in_GeV, u, what_it_breaks)`` and an
        array of the interior knots of the model's own ``T(z)`` spline, both in ``u = log(1+z)``.
    """
    branch = []
    for name, T_in_GeV, breaks_what in QCD_BREAK_TEMPERATURES_GEV:
        u = qcd_branch_boundary_u(cosmology, T_in_GeV, u_lo, u_hi)
        if u is not None:
            branch.append((name, T_in_GeV, u, breaks_what))

    # The T(z) spline's own knots. ZSplineWrapper keeps the SciPy BSpline privately; this is a
    # docs-directory measurement script, so reaching for it is acceptable here. A production
    # implementation would ask the cosmology for its break points.
    spline = cosmology._T_z_spline._spline
    knots = np.unique(np.asarray(spline.t, dtype=float))
    knots = knots[(knots > u_lo) & (knots < u_hi)]

    return branch, knots


def assign_breaks(edges, break_u):
    """
    For interval ``i = [edges[i], edges[i+1]]``, the sorted break points strictly inside it.

    :return: a list, one entry per interval, of tuples of break points
    """
    break_u = np.sort(np.asarray(break_u, dtype=float))
    out = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        lo = int(np.searchsorted(break_u, a, side="right"))
        hi = int(np.searchsorted(break_u, b, side="left"))
        out.append(tuple(float(x) for x in break_u[lo:hi]))
    return out


# ---------------------------------------------------------------------------------------------
# the converged reference
# ---------------------------------------------------------------------------------------------


def _quad_panel(f, a, b, epsrel):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        val, err = quad(f, a, b, limit=400, epsabs=0.0, epsrel=epsrel)
    return val, err, len(caught) > 0


def quad_over_interval(f, a, b, breaks, epsrel):
    """SciPy ``quad`` on each sub-panel of ``[a, b]``, summed with ``fsum``."""
    edges = [a, *breaks, b]
    parts = []
    worst_err = 0.0
    warned = False
    for i in range(len(edges) - 1):
        val, err, w = _quad_panel(f, edges[i], edges[i + 1], epsrel)
        parts.append(val)
        worst_err = max(worst_err, fabs(err))
        warned = warned or w
    return fsum(parts), worst_err, warned


def gauss40_over_interval(f, a, b, breaks):
    """``reference_lib.gauss_bisect`` on each sub-panel of ``[a, b]``, summed with ``fsum``."""
    edges = [a, *breaks, b]
    parts = []
    worst_change = 0.0
    worst_level = 0
    for i in range(len(edges) - 1):
        val, change, level = R.gauss_bisect(
            f,
            edges[i],
            edges[i + 1],
            rtol=GAUSS_CROSSCHECK_RTOL,
            max_level=GAUSS_CROSSCHECK_MAX_LEVEL,
        )
        parts.append(val)
        worst_change = max(worst_change, change)
        worst_level = max(worst_level, level)
    return fsum(parts), worst_change, worst_level


def build_reference(f, edges, break_lists):
    """
    The converged reference increment for every interval of ``edges``, computed **break-aware**:
    each production interval is split at the break points it contains before the adaptive rule
    runs, so that the reference is trustworthy even where the integrand is discontinuous.

    Three independent values are formed per interval -- ``quad`` at ``QUAD_EPSREL``, ``quad`` at
    the looser ``QUAD_CROSSCHECK_EPSREL``, and composite Gauss-Legendre order 40 with bisection --
    and their worst disagreement is reported.

    :return: ``(ref_parts, diagnostics)``
    """
    n_int = len(edges) - 1
    ref_parts = []
    worst_quad_err = 0.0
    warned = 0
    worst_loose = 0.0
    worst_loose_at = 0
    worst_gl = 0.0
    worst_gl_at = 0
    worst_gl_change = 0.0
    worst_gl_level = 0

    for i in range(n_int):
        breaks = () if break_lists is None else break_lists[i]
        primary, err, w = quad_over_interval(
            f, edges[i], edges[i + 1], breaks, QUAD_EPSREL
        )
        loose, _, _ = quad_over_interval(
            f, edges[i], edges[i + 1], breaks, QUAD_CROSSCHECK_EPSREL
        )
        gl, change, level = gauss40_over_interval(f, edges[i], edges[i + 1], breaks)

        ref_parts.append(primary)
        worst_quad_err = max(worst_quad_err, err)
        warned += 1 if w else 0
        worst_gl_change = max(worst_gl_change, change)
        worst_gl_level = max(worst_gl_level, level)

        denom = fabs(primary)
        if denom == 0.0:
            continue
        rel_loose = fabs(primary - loose) / denom
        if rel_loose > worst_loose:
            worst_loose, worst_loose_at = rel_loose, i
        rel_gl = fabs(primary - gl) / denom
        if rel_gl > worst_gl:
            worst_gl, worst_gl_at = rel_gl, i

    diagnostics = {
        "num_intervals": n_int,
        "reference_total": fsum(ref_parts),
        "grid_generation": GRID_GENERATION,
        "crosscheck_quad_1e-12_max_rel": worst_loose,
        "crosscheck_quad_1e-12_at_z": float(expm1(edges[worst_loose_at])),
        "crosscheck_gauss40_max_rel": worst_gl,
        "crosscheck_gauss40_at_z": float(expm1(edges[worst_gl_at])),
        "quad_max_reported_abserr": worst_quad_err,
        "quad_intervals_with_warnings": warned,
        "gauss40_worst_relative_change": worst_gl_change,
        "gauss40_worst_level": worst_gl_level,
    }
    return ref_parts, diagnostics


def reference_case_drift(
    f,
    edges,
    break_lists,
    ref_parts,
    checkpoint_index,
    smallest_reported: float,
    crosscheck_gauss40_max_rel: float,
    residual: bool,
) -> dict:
    """
    How far the adaptive reference moves under **one step of tightening**, and whether that is
    small enough for the numbers this case goes on to report (README §5 rule 5).

    Taken through ``ComputeTargets/tests/convergence_reference.reference_drift``, which is the
    campaign's one implementation of the convergence test (board standing note 14) and which will
    not hand back a drift without the verdict attached.

    **The drift is measured in the kind the case decides in.** For the three primitives the
    decision reads ``max_cumulative_rel_error``, so the drift is relative, at every checkpoint
    cumulative and every production interval. For the two residuals it reads
    ``max_cumulative_abs_error`` against a target in radians, so the drift is **absolute**, at the
    checkpoints only: a rho increment is ~1e-13 rad and an absolutely negligible movement is
    enormous against that denominator, so a per-interval relative drift there would report the
    conditioning of the denominator and not the convergence of the reference.

    **The step is ``QUAD_DRIFT_EPSREL -> QUAD_EPSREL``, and it is taken from the loose side
    because the reference is already at QUADPACK's floor.** ``scipy.integrate.quad`` refuses an
    ``epsrel`` below ``50 * eps = 1.11e-14`` outright when ``epsabs = 0``, so the reference at
    ``QUAD_EPSREL = 1.5e-14`` cannot be tightened by a decade at all -- there is no such setting.
    What is measured instead is the decade *into* it: the reference at ``1.5e-13`` moved by this
    much when the tolerance was tightened to the value the block actually uses. Under the usual
    reading of a convergence test that bounds what a further decade could move it, and it is
    reported as a bound and not as a two-sided error.

    The second leg is the one this script already had: the independent composite order-40
    Gauss-Legendre rule with bisection, which is a different method rather than the same one with
    a smaller number. ``reference_error_bound_max_rel`` is the larger of the two, and the verdict
    is re-scored against it.

    **One note the verdict carries is about the wrong solver.** ``TolerancePair`` reports
    ``scipy.integrate``'s ``100 eps`` clamp, which applies to ``solve_ivp`` and not to
    ``quad`` -- the same mismatch prompt 03a recorded for ``brentq`` in
    ``[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve]``. It is recorded here rather
    than worked around: ``convergence_reference.py`` is outside prompt 04's file list, and the
    measured movement below is the evidence that the step really was applied.
    """
    n_int = len(edges) - 1
    cache = {QUAD_EPSREL: list(ref_parts)}

    def build(knob):
        if knob.rtol not in cache:
            cache[knob.rtol] = [
                quad_over_interval(
                    f,
                    edges[i],
                    edges[i + 1],
                    () if break_lists is None else break_lists[i],
                    knob.rtol,
                )[0]
                for i in range(n_int)
            ]
        return cache[knob.rtol]

    def error_measure(candidate, reference):
        out = []
        for idx in checkpoint_index:
            got, want = fsum(candidate[idx:]), fsum(reference[idx:])
            if residual:
                out.append((fabs(got - want), float(expm1(edges[idx])), idx))
            elif fabs(want) != 0.0:
                out.append(
                    (fabs(got - want) / fabs(want), float(expm1(edges[idx])), idx)
                )
        if residual:
            return out
        for i in range(n_int):
            if fabs(reference[i]) == 0.0:
                continue
            out.append(
                (
                    fabs(candidate[i] - reference[i]) / fabs(reference[i]),
                    float(expm1(edges[i])),
                    i,
                )
            )
        return out

    # a case whose every order is exact has no "smallest difference it intends to report"; the
    # criterion is then applied against one ulp of the quantity, which is the smallest thing it
    # could distinguish
    smallest = (
        float(smallest_reported) if smallest_reported > 0.0 else 2.220446049250313e-16
    )

    verdict = reference_drift(
        build,
        TolerancePair(atol=0.0, rtol=QUAD_DRIFT_EPSREL),
        error_measure=error_measure,
        smallest_reported_difference=smallest,
        tightened_knob=TolerancePair(atol=0.0, rtol=QUAD_EPSREL),
    )
    # the independent leg is a *relative* disagreement, so it only joins the bound when the drift
    # is measured in the same kind
    bound = (
        verdict.max if residual else max(verdict.max, float(crosscheck_gauss40_max_rel))
    )
    return {
        "reference_drift_epsrel_step": [QUAD_DRIFT_EPSREL, QUAD_EPSREL],
        "reference_drift_step_is_from_the_loose_side": True,
        "reference_drift_kind": "absolute" if residual else "relative",
        "reference_drift_max": verdict.max,
        "reference_drift_at_z": verdict.max_z,
        "reference_drift_median": verdict.median,
        "reference_drift_threshold": verdict.threshold,
        "reference_drift_criterion_ratio": CRITERION_RATIO,
        "reference_drift_smallest_reported": smallest,
        "reference_drift_passed": bool(verdict.passed),
        "reference_error_bound": bound,
        "reference_error_bound_passed": bool(bound <= verdict.threshold),
        "reference_drift_notes": list(verdict.notes),
    }


# ---------------------------------------------------------------------------------------------
# per-order measurement
# ---------------------------------------------------------------------------------------------


def measure_orders(f, edges, ref_parts, break_lists, checkpoint_index, json_cumulative):
    """
    Fixed-order Gauss on every interval of ``edges``, at each order in ``ORDERS``, under one
    build scheme.

    :param break_lists: per-interval interior break points, or ``None`` for the plain scheme
    :param checkpoint_index: parts index of each checkpoint (cumulative = sum of the parts above)
    :param json_cumulative: the JSON reference cumulative at each checkpoint, same order
    """
    n_int = len(edges) - 1
    out = {}

    for order in ORDERS:
        worst = 0.0
        worst_i = 0
        offenders = []
        parts = []
        for i in range(n_int):
            breaks = () if break_lists is None else break_lists[i]
            got = gauss_interval(f, edges[i], edges[i + 1], order, breaks)
            parts.append(got)
            denom = fabs(ref_parts[i])
            if denom == 0.0:
                continue
            rel = fabs(got - ref_parts[i]) / denom
            if rel > INTERVAL_REPORT_THRESHOLD:
                offenders.append(i)
            if rel > worst:
                worst, worst_i = rel, i

        cum_abs = 0.0
        cum_abs_z = None
        cum_rel = 0.0
        cum_rel_z = None
        for idx, reference_value in zip(checkpoint_index, json_cumulative):
            got = fsum(parts[idx:])
            err = fabs(got - reference_value)
            z_here = float(expm1(edges[idx])) if idx < len(edges) else None
            if err > cum_abs:
                cum_abs, cum_abs_z = err, z_here
            denom = fabs(reference_value)
            if denom == 0.0:
                # the top checkpoint, where the cumulative is identically zero
                continue
            rel = err / denom
            if rel > cum_rel:
                cum_rel, cum_rel_z = rel, z_here

        out[str(order)] = {
            "max_increment_rel_error": worst,
            "max_increment_at_z": float(expm1(edges[worst_i])),
            "max_increment_abs_error": worst * fabs(ref_parts[worst_i]),
            "max_increment_as_rad_at_3e8": worst * fabs(ref_parts[worst_i]) * PHASE_K,
            "intervals_above_1e-12": len(offenders),
            "offending_interval_z": [float(expm1(edges[i])) for i in offenders[:8]],
            "max_cumulative_abs_error": cum_abs,
            "max_cumulative_abs_at_z": cum_abs_z,
            "max_cumulative_rel_error": cum_rel,
            "max_cumulative_rel_at_z": cum_rel_z,
        }

    return out


def residual_edges(u_ascending, u_anchor):
    """
    The integration edges for a residual anchored at the off-grid ``u_anchor``: every production
    interval below it, plus one partial interval up to the anchor.  This is the same decomposition
    ``generate_references.py`` used to build the JSON (its lowest checkpoint is the bottom node),
    so the cumulative comparison is like for like.
    """
    i_anchor = int(np.searchsorted(u_ascending, u_anchor))
    return np.concatenate([u_ascending[:i_anchor], [u_anchor]]), i_anchor


def build_cases(model, z_nodes, u_ascending, references):
    """
    Every (integrand, edges, checkpoint) case for one model.

    :return: a list of ``(key, label, f, edges, checkpoint_index, json_cumulative, extra)``
    """
    functions = model.functions
    n_nodes = len(z_nodes)
    checkpoints = references["checkpoints"]
    cp_index = [n_nodes - 1 - c["index"] for c in checkpoints]
    anchors = {kk: float(z) for kk, z in references["rho_anchor_z"].items()}

    cases = []
    for key, label, maker, json_field, sign in PRIMITIVE_SPECS:
        cases.append(
            (
                key,
                label,
                maker(functions),
                u_ascending,
                cp_index,
                [sign * v for v in references[json_field]],
                {},
            )
        )
    for key, label, maker in RESIDUAL_SPECS:
        for k in REFERENCE_K_VALUES:
            kk = k_key(k)
            entries = references[key][kk]
            if not entries:
                continue
            edges, _ = residual_edges(u_ascending, log1p(anchors[kk]))
            cases.append(
                (
                    f"{key}@{kk}",
                    f"{label} (k = {k:.3g}/Mpc)",
                    maker(functions, k),
                    edges,
                    [n_nodes - 1 - e["index"] for e in entries],
                    [e["value"] for e in entries],
                    {"k": k, "rho_anchor_z": anchors[kk]},
                )
            )
    return cases


def measure_model(
    name, model, z_nodes, u_ascending, references, schemes, reference_breaks
):
    """
    Every per-order measurement for one model, under each supplied build scheme.

    :param schemes: dict scheme name -> array of break points (empty array for "plain")
    :param reference_breaks: the break points used to build the *reference* -- the finest set the
        model offers, so the reference is right even where the plain scheme is not
    :return: scheme -> integrand key -> block
    """
    print(f"\n{'=' * 96}\n{name}\n{'=' * 96}", flush=True)
    results = {scheme: {} for scheme in schemes}

    for key, label, f, edges, cp_index, json_cum, extra in build_cases(
        model, z_nodes, u_ascending, references
    ):
        ref_break_lists = (
            None
            if len(reference_breaks) == 0
            else assign_breaks(edges, reference_breaks)
        )
        t0 = time.perf_counter()
        ref_parts, diagnostics = build_reference(f, edges, ref_break_lists)
        reference_seconds = time.perf_counter() - t0

        # how the break-aware reference compares with the JSON value the orders are scored against
        json_vs_ref = 0.0
        for idx, reference_value in zip(cp_index, json_cum):
            denom = fabs(reference_value)
            if denom == 0.0:
                continue
            json_vs_ref = max(
                json_vs_ref, fabs(fsum(ref_parts[idx:]) - reference_value) / denom
            )
        diagnostics["json_vs_reference_max_rel"] = json_vs_ref

        print(
            f"  {key:<24} reference: {diagnostics['num_intervals']} intervals, "
            f"quad@1e-12 {diagnostics['crosscheck_quad_1e-12_max_rel']:.2e}, "
            f"GL40 {diagnostics['crosscheck_gauss40_max_rel']:.2e}, "
            f"vs JSON {json_vs_ref:.2e}  ({reference_seconds:.1f} s)",
            flush=True,
        )

        per_scheme = {}
        for scheme, break_u in schemes.items():
            break_lists = None if len(break_u) == 0 else assign_breaks(edges, break_u)
            t0 = time.perf_counter()
            orders = measure_orders(
                f, edges, ref_parts, break_lists, cp_index, json_cum
            )
            per_scheme[scheme] = (orders, time.perf_counter() - t0)
            print(
                f"    [{scheme:>12}] increment "
                + "  ".join(
                    f"N={o}:{orders[str(o)]['max_increment_rel_error']:.2e}"
                    for o in ORDERS
                ),
                flush=True,
            )

        # README §5 rule 5, and the reason this is computed *after* the sweep rather than before:
        # the criterion is a tenth of the smallest difference the case intends to report, and
        # that is not known until every order has been scored. The field read is the one the
        # decision reads for this kind of case -- relative for the primitives, absolute radians
        # for the residuals.
        residual = key not in PRIMITIVE_KEYS
        field = "max_cumulative_abs_error" if residual else "max_cumulative_rel_error"
        smallest_reported = min(
            (
                entry[field]
                for orders, _ in per_scheme.values()
                for entry in orders.values()
                if entry[field] > 0.0
            ),
            default=0.0,
        )
        t0 = time.perf_counter()
        diagnostics.update(
            reference_case_drift(
                f,
                edges,
                ref_break_lists,
                ref_parts,
                cp_index,
                smallest_reported,
                diagnostics["crosscheck_gauss40_max_rel"],
                residual,
            )
        )
        print(
            f"    [       drift] reference moves {diagnostics['reference_drift_max']:.2e} "
            f"({diagnostics['reference_drift_kind']}) under epsrel {QUAD_DRIFT_EPSREL:g} -> "
            f"{QUAD_EPSREL:g}, against a threshold of "
            f"{diagnostics['reference_drift_threshold']:.2e}: "
            f"{'converged' if diagnostics['reference_drift_passed'] else 'NOT CONVERGED'} "
            f"({time.perf_counter() - t0:.1f} s)",
            flush=True,
        )

        for scheme, (orders, seconds) in per_scheme.items():
            block = dict(diagnostics)
            block.update(extra)
            block["label"] = label
            block["scheme"] = scheme
            block["reference_seconds"] = reference_seconds
            block["seconds"] = seconds
            block["orders"] = orders
            results[scheme][key] = block

    return results


# ---------------------------------------------------------------------------------------------
# item 4: smoothness of what prompts 09 and 10 will spline
# ---------------------------------------------------------------------------------------------


def smoothness(functions, u_ascending, u_anchor, maker, k):
    """
    ``rho(u) = int_u^{u_anchor} g``, so ``d2 rho/du2 = -g'`` and ``d4 rho/du4 = -g'''``.  Both are
    estimated by central finite differences of the integrand ``g`` on the production grid's own
    local spacing ``h``, at every interior node strictly below the anchor.

    :return: a dict of the two maxima, the implied cubic-spline error, and where each occurs
    """
    g = maker(functions, k)
    worst2 = worst4 = worst_spline = 0.0
    z2 = z4 = zs = None
    i_anchor = int(np.searchsorted(u_ascending, u_anchor))
    for i in range(2, i_anchor - 2):
        u = float(u_ascending[i])
        h = float(u_ascending[i + 1] - u)
        if u + 2.0 * h >= u_anchor:
            break
        g_m2, g_m1, g_p1, g_p2 = (g(u - 2 * h), g(u - h), g(u + h), g(u + 2 * h))
        d2 = fabs((g_p1 - g_m1) / (2.0 * h))
        d4 = fabs((-g_m2 + 2.0 * g_m1 - 2.0 * g_p1 + g_p2) / (2.0 * h * h * h))
        spline_err = h**4 * d4 / 384.0
        z = float(expm1(u))
        if d2 > worst2:
            worst2, z2 = d2, z
        if d4 > worst4:
            worst4, z4 = d4, z
        if spline_err > worst_spline:
            worst_spline, zs = spline_err, z
    return {
        "max_d2_rho_du2": worst2,
        "max_d2_at_z": z2,
        "max_d4_rho_du4": worst4,
        "max_d4_at_z": z4,
        "max_cubic_spline_error_rad": worst_spline,
        "max_cubic_spline_error_at_z": zs,
    }


# ---------------------------------------------------------------------------------------------
# the decision rules (prompt 02 §3)
# ---------------------------------------------------------------------------------------------


def smallest_within_factor(errors_by_order, factor=FLOOR_FACTOR):
    """
    The smallest order whose error is within ``factor`` of the best any order in ``ORDERS``
    achieves.  That best value is the operational reading of prompt 02 §3's "reference floor":
    the level at which raising the order stops buying anything, whether it is set by the
    double-precision evaluation of the background or by the reference itself.

    :param errors_by_order: {order: error}
    :return: ``(order, floor)``
    """
    floor = min(errors_by_order.values())
    if floor == 0.0:
        return min(o for o, e in errors_by_order.items() if e == 0.0), floor
    for order in sorted(errors_by_order):
        if errors_by_order[order] <= factor * floor:
            return order, floor
    return max(errors_by_order), floor


def smallest_meeting(errors_by_order, target):
    for order in sorted(errors_by_order):
        if errors_by_order[order] <= target:
            return order
    return None


def _primitive_errors(results, scheme, key, field):
    return {o: results[scheme][key]["orders"][str(o)][field] for o in ORDERS}


def _residual_errors(results, scheme):
    """max over sector, k and checkpoint of the absolute cumulative error, per order."""
    out = {}
    for order in ORDERS:
        worst = 0.0
        for key in residual_keys():
            block = results[scheme].get(key)
            if block is None:
                continue
            worst = max(worst, block["orders"][str(order)]["max_cumulative_abs_error"])
        out[order] = worst
    return out


def _radiation_control_errors(controls):
    """max over k and checkpoint of the rho_T relative error against the closed form, per order."""
    return {
        o: max(
            v["orders"][str(o)]["max_rel_error_vs_closed_form"]
            for v in controls["rho_T"].values()
        )
        for o in ORDERS
    }


def decide(lam_results, qcd_results, controls):
    """
    Apply prompt 02 §3's rules, and §4's exact-radiation control, to the measurements.

    The QCD build scheme is chosen first, from the measurements rather than in advance: the
    cheapest of ``plain``, ``branch``, ``branch+knots`` that (i) admits a fixed order <= 16
    meeting the rho target and (ii) does not lose accuracy on any primitive relative to the best
    scheme measured.  The four orders are then the larger of the per-model choices, LambdaCDM
    always on ``plain`` (it has no break points).
    """
    out = {}

    # ---- scheme selection on QCD ------------------------------------------------------------
    scheme_summary = {}
    best_primitive_floor = {key: float("inf") for key in PRIMITIVE_KEYS}
    for scheme in SCHEME_ORDER:
        entry = {
            "primitive_floor_rel": {},
            "primitive_floor_abs": {},
            "primitive_choice": {},
        }
        for key in PRIMITIVE_KEYS:
            errs = _primitive_errors(
                qcd_results, scheme, key, "max_cumulative_rel_error"
            )
            errs_abs = _primitive_errors(
                qcd_results, scheme, key, "max_cumulative_abs_error"
            )
            order, floor = smallest_within_factor(errs)
            entry["primitive_floor_rel"][key] = floor
            entry["primitive_floor_abs"][key] = min(errs_abs.values())
            entry["primitive_choice"][key] = order
            entry[f"{key}_errors_rel"] = {str(o): errs[o] for o in ORDERS}
            entry[f"{key}_errors_abs"] = {str(o): errs_abs[o] for o in ORDERS}
            best_primitive_floor[key] = min(best_primitive_floor[key], floor)
        rho_errs = _residual_errors(qcd_results, scheme)
        entry["rho_errors_abs"] = {str(o): rho_errs[o] for o in ORDERS}
        entry["rho_min_order"] = smallest_meeting(rho_errs, RHO_TARGET_RAD)
        scheme_summary[scheme] = entry

    # Only a scheme production can execute may be recommended. `integration_break_points` has
    # returned the three temperature crossings and nothing else since qcd-background-audit
    # prompt 07, so `branch+knots` is a control from here on and the choice is plain or branch
    # (tolerance-convergence prompt 04 §3.2). The comparison it must survive is still against the
    # best floor *any* measured scheme reaches, the control included: that is the question
    # "does refusing to split at a knot cost anything?", and it is the one worth asking.
    recommended = None
    for scheme in CANDIDATE_SCHEMES:
        entry = scheme_summary[scheme]
        if entry["rho_min_order"] is None:
            continue
        if any(
            # A best floor of exactly zero carries no information about how much better one
            # scheme is than another -- it says the reference and the JSON agree to the bit, and
            # nothing can be three times worse than zero. Comparing against it disqualifies every
            # scheme including the one that achieved it, which is not what this filter is for.
            # Measured on the 2026-09-17 re-run: `branch+knots` reaches 0.0 on `friction`, which
            # under a bare `> 3 * best` rejected `branch` as well (tolerance-convergence log 04,
            # deviation 3).
            best_primitive_floor[key] > 0.0
            and entry["primitive_floor_rel"][key]
            > FLOOR_FACTOR * best_primitive_floor[key]
            for key in PRIMITIVE_KEYS
        ):
            continue
        recommended = scheme
        break
    no_candidate_qualified = recommended is None
    if no_candidate_qualified:
        # no executable scheme works: report the finest executable one and say so. This would be
        # a finding about the representation, not a recommendation to split at the knots, which
        # production cannot do at all.
        recommended = CANDIDATE_SCHEMES[-1]

    out["recommended_scheme"] = recommended
    out["candidate_schemes"] = list(CANDIDATE_SCHEMES)
    out["control_scheme"] = CONTROL_SCHEME
    out["no_candidate_scheme_qualified"] = no_candidate_qualified
    out["zero_best_floor_primitives"] = [
        key for key in PRIMITIVE_KEYS if best_primitive_floor[key] == 0.0
    ]
    out["scheme_selection_rule"] = (
        "the cheapest scheme production can execute -- `integration_break_points` returns the "
        "equation-of-state temperature crossings and nothing else, so `plain` and `branch` are "
        "the candidates -- that admits a fixed order <= 16 meeting the rho target and stays "
        f"within {FLOOR_FACTOR:g}x the best primitive floor any measured scheme reaches, "
        f"`{CONTROL_SCHEME}` included -- skipping any primitive whose best floor is exactly "
        "zero, which is listed in `zero_best_floor_primitives` and against which no multiple is "
        f"meaningful. `{CONTROL_SCHEME}` is retained as a control and is scored in "
        "`knots_control`; it is not a candidate."
    )
    out["scheme_summary"] = scheme_summary
    out["best_primitive_floor_rel"] = best_primitive_floor

    # what splitting at the T(z) spline's knots still buys, over the recommended executable
    # scheme. qcd-background-audit prompt 07 measured that there is nothing left at a knot for a
    # panel edge to protect against; this is the same question at the quadrature level.
    out["knots_control"] = {
        "scheme": CONTROL_SCHEME,
        "criterion": (
            "ratio of the recommended executable scheme's error to the control's, per quantity "
            "and order; > 1 means splitting at the knots is still buying something"
        ),
        "primitive_floor_rel": {
            key: {
                "recommended": scheme_summary[recommended]["primitive_floor_rel"][key],
                "control": scheme_summary[CONTROL_SCHEME]["primitive_floor_rel"][key],
                "ratio": (
                    scheme_summary[recommended]["primitive_floor_rel"][key]
                    / scheme_summary[CONTROL_SCHEME]["primitive_floor_rel"][key]
                    if scheme_summary[CONTROL_SCHEME]["primitive_floor_rel"][key] > 0.0
                    else None
                ),
            }
            for key in PRIMITIVE_KEYS
        },
        "primitive_errors_rel": {
            key: {
                str(o): {
                    "recommended": scheme_summary[recommended][f"{key}_errors_rel"][
                        str(o)
                    ],
                    "control": scheme_summary[CONTROL_SCHEME][f"{key}_errors_rel"][
                        str(o)
                    ],
                }
                for o in ORDERS
            }
            for key in PRIMITIVE_KEYS
        },
        "rho_errors_abs": {
            str(o): {
                "recommended": scheme_summary[recommended]["rho_errors_abs"][str(o)],
                "control": scheme_summary[CONTROL_SCHEME]["rho_errors_abs"][str(o)],
            }
            for o in ORDERS
        },
        "rho_min_order": {
            "recommended": scheme_summary[recommended]["rho_min_order"],
            "control": scheme_summary[CONTROL_SCHEME]["rho_min_order"],
        },
    }

    # ---- the three primitive orders ---------------------------------------------------------
    for key, field in (("tau", "N_tau"), ("cs_tau", "N_cs_tau"), ("friction", "N_F")):
        lam_errs = _primitive_errors(
            lam_results, "plain", key, "max_cumulative_rel_error"
        )
        lam_order, lam_floor = smallest_within_factor(lam_errs)
        qcd_errs = _primitive_errors(
            qcd_results, recommended, key, "max_cumulative_rel_error"
        )
        qcd_order, qcd_floor = smallest_within_factor(qcd_errs)
        out[field] = max(lam_order, qcd_order)
        out[f"{field}_detail"] = {
            "LambdaCDMModel": {
                "scheme": "plain",
                "floor_rel": lam_floor,
                "order": lam_order,
                "errors_rel": {str(o): lam_errs[o] for o in ORDERS},
                "errors_abs": {
                    str(o): v
                    for o, v in _primitive_errors(
                        lam_results, "plain", key, "max_cumulative_abs_error"
                    ).items()
                },
            },
            "QCDModel": {
                "scheme": recommended,
                "floor_rel": qcd_floor,
                "order": qcd_order,
                "errors_rel": {str(o): qcd_errs[o] for o in ORDERS},
                "errors_abs": {
                    str(o): v
                    for o, v in _primitive_errors(
                        qcd_results, recommended, key, "max_cumulative_abs_error"
                    ).items()
                },
            },
        }

    # ---- N_rho ------------------------------------------------------------------------------
    lam_rho = _residual_errors(lam_results, "plain")
    qcd_rho = _residual_errors(qcd_results, recommended)
    rad_rho = _radiation_control_errors(controls)
    lam_n = smallest_meeting(lam_rho, RHO_TARGET_RAD)
    qcd_n = smallest_meeting(qcd_rho, RHO_TARGET_RAD)
    rad_n = smallest_meeting(rad_rho, RHO_RADIATION_CONTROL_RTOL)

    fallback = qcd_n is None
    if lam_n is None or qcd_n is None or rad_n is None:
        out["N_rho"] = max(ORDERS)
    else:
        out["N_rho"] = max(lam_n, qcd_n, rad_n)
    out["N_rho_detail"] = {
        "LambdaCDMModel": {
            "scheme": "plain",
            "criterion": f"cumulative |d rho| <= {RHO_TARGET_RAD:g} rad",
            "order": lam_n,
            "errors_abs": {str(o): lam_rho[o] for o in ORDERS},
        },
        "QCDModel": {
            "scheme": recommended,
            "criterion": f"cumulative |d rho| <= {RHO_TARGET_RAD:g} rad",
            "order": qcd_n,
            "errors_abs": {str(o): qcd_rho[o] for o in ORDERS},
        },
        "RadiationModel": {
            "scheme": "plain",
            "criterion": f"rho_T vs the closed form <= {RHO_RADIATION_CONTROL_RTOL:g} relative",
            "order": rad_n,
            "errors_rel": {str(o): rad_rho[o] for o in ORDERS},
        },
    }
    out["rho_target_rad"] = RHO_TARGET_RAD
    out["rho_radiation_control_rtol"] = RHO_RADIATION_CONTROL_RTOL
    out["rho_adaptive_fallback_required"] = fallback
    out["rho_fixed_order_works_without_subdivision"] = (
        scheme_summary["plain"]["rho_min_order"] is not None
    )
    out["floor_factor"] = FLOOR_FACTOR
    out["orders_considered"] = list(ORDERS)

    # ---- what carrying rho buys on LambdaCDM (prompt 02 §3, final bullet) --------------------
    out["lambdacdm_rho_magnitude_rad"] = {
        key: fabs(lam_results["plain"][key]["reference_total"])
        for key in residual_keys()
        if key in lam_results["plain"]
    }
    out["qcd_rho_magnitude_rad"] = {
        key: fabs(qcd_results[recommended][key]["reference_total"])
        for key in residual_keys()
        if key in qcd_results[recommended]
    }
    return out


# ---------------------------------------------------------------------------------------------
# item 5: build cost
# ---------------------------------------------------------------------------------------------


def build_cost(models, u_ascending, references, decision, scheme_breaks):
    """Integrand evaluations and wall time to build each full table at the recommended order."""
    orders = {
        "tau": decision["N_tau"],
        "cs_tau": decision["N_cs_tau"],
        "friction": decision["N_F"],
    }
    out = {}
    for label, model, break_u in models:
        refs = references["models"][label]
        entry = {
            "scheme": "plain" if len(break_u) == 0 else decision["recommended_scheme"]
        }
        break_lists = None if len(break_u) == 0 else assign_breaks(u_ascending, break_u)

        table_evals = 0
        table_seconds = 0.0
        for key, _, maker, _, _ in PRIMITIVE_SPECS:
            order = orders[key]
            counter = CountingIntegrand(maker(model.functions))
            t0 = time.perf_counter()
            total = fsum(
                gauss_interval(
                    counter,
                    u_ascending[i],
                    u_ascending[i + 1],
                    order,
                    () if break_lists is None else break_lists[i],
                )
                for i in range(len(u_ascending) - 1)
            )
            seconds = time.perf_counter() - t0
            entry[key] = {
                "order": order,
                "integrand_evaluations": counter.calls,
                "seconds": seconds,
                "value": total,
            }
            table_evals += counter.calls
            table_seconds += seconds

        for key, _, maker in RESIDUAL_SPECS:
            order = decision["N_rho"]
            k = max(REFERENCE_K_VALUES)
            kk = k_key(k)
            edges, _ = residual_edges(
                u_ascending, log1p(float(refs["rho_anchor_z"][kk]))
            )
            bl = None if len(break_u) == 0 else assign_breaks(edges, break_u)
            counter = CountingIntegrand(maker(model.functions, k))
            t0 = time.perf_counter()
            total = fsum(
                gauss_interval(
                    counter, edges[i], edges[i + 1], order, () if bl is None else bl[i]
                )
                for i in range(len(edges) - 1)
            )
            entry[f"{key}@{kk}"] = {
                "order": order,
                "integrand_evaluations": counter.calls,
                "seconds": time.perf_counter() - t0,
                "value": total,
            }

        entry["three_tables_total_evaluations"] = table_evals
        entry["three_tables_total_seconds"] = table_seconds
        out[label] = entry
    return out


# ---------------------------------------------------------------------------------------------
# exact-radiation controls (prompt 02 §4)
# ---------------------------------------------------------------------------------------------


def radiation_controls(rad, z_nodes_rad, refs):
    """
    * ``C == 0`` identically for the Green's function in exact radiation, so every Gauss increment
      of ``rho_G`` must be **bit-exactly** zero at every order -- asserted, not toleranced.
    * ``rho_T`` is scored against ``RadiationModel``'s closed-form primitive.  Review §12.4's
      ``1/x_i - 1/x`` is that primitive's large-x asymptote (and carries the opposite sign in this
      campaign's convention, ``wkb_reference.RadiationModel`` docstring), so its departure from
      the closed form is reported separately rather than used as the reference.
    """
    u_ascending = np.log1p(np.asarray(z_nodes_rad, dtype=float))[::-1].copy()
    out = {"rho_G_exactly_zero": {}, "rho_G_max_abs_increment": {}, "rho_T": {}}

    for k in REFERENCE_K_VALUES:
        kk = k_key(k)
        z_anchor = float(refs["rho_anchor_z"][kk])
        u_anchor = log1p(z_anchor)
        edges, _ = residual_edges(u_ascending, u_anchor)

        g = R.integrand_rho_G_double(rad.functions, k)
        exact_zero = True
        worst_abs = 0.0
        for order in ORDERS:
            for i in range(len(edges) - 1):
                panel = gauss_panel(g, edges[i], edges[i + 1], order)
                worst_abs = max(worst_abs, fabs(panel))
                if panel != 0.0:
                    exact_zero = False
        out["rho_G_exactly_zero"][kk] = exact_zero
        out["rho_G_max_abs_increment"][kk] = worst_abs

        gt = R.integrand_rho_T_double(rad.functions, k)
        entries = refs["rho_T"][kk]
        per_order = {}
        for order in ORDERS:
            parts = [
                gauss_panel(gt, edges[i], edges[i + 1], order)
                for i in range(len(edges) - 1)
            ]
            worst = 0.0
            worst_z = None
            worst_json = 0.0
            for e in entries:
                i = len(z_nodes_rad) - 1 - e["index"]
                got = fsum(parts[i:])
                exact = rad.rho_T(k, e["z"], z_anchor)
                rel = fabs(got - exact) / fabs(exact) if exact != 0.0 else fabs(got)
                if rel > worst:
                    worst, worst_z = rel, e["z"]
                if e["value"] != 0.0:
                    worst_json = max(
                        worst_json, fabs(got - e["value"]) / fabs(e["value"])
                    )
            per_order[str(order)] = {
                "max_rel_error_vs_closed_form": worst,
                "at_z": worst_z,
                "max_rel_error_vs_json": worst_json,
            }

        # the large-x asymptote, at the bottom checkpoint, for the record
        e = entries[-1]
        x_i = k * rad.cs_tau(z_anchor)
        x = k * rad.cs_tau(e["z"])
        asymptote = 1.0 / x - 1.0 / x_i
        exact = rad.rho_T(k, e["z"], z_anchor)
        out["rho_T"][kk] = {
            "orders": per_order,
            "x_init": x_i,
            "x_final": x,
            "closed_form": exact,
            "asymptote_1_over_x_minus_1_over_x_i": asymptote,
            "asymptote_relative_departure": fabs(asymptote - exact) / fabs(exact),
        }
    return out


# ---------------------------------------------------------------------------------------------
# markdown emission
# ---------------------------------------------------------------------------------------------


def fmt(x, spec=".2e"):
    if x is None:
        return "--"
    if isinstance(x, float) and x != x:
        return "n/a"
    return format(x, spec)


def increment_table(results, keys, scheme):
    lines = [
        "| integrand | "
        + " | ".join(f"N={o}" for o in ORDERS)
        + " | worst interval at $z$ (N=4) |",
        "|---|" + "---|" * (len(ORDERS) + 1),
    ]
    for key in keys:
        block = results[scheme].get(key)
        if block is None:
            continue
        cells = [
            fmt(block["orders"][str(o)]["max_increment_rel_error"]) for o in ORDERS
        ]
        worst_z = block["orders"]["4"]["max_increment_at_z"]
        lines.append(f"| `{key}` | " + " | ".join(cells) + f" | {worst_z:.4g} |")
    return "\n".join(lines)


def cumulative_table(results, keys, scheme, field):
    lines = [
        "| integrand | " + " | ".join(f"N={o}" for o in ORDERS) + " |",
        "|---|" + "---|" * len(ORDERS),
    ]
    for key in keys:
        block = results[scheme].get(key)
        if block is None:
            continue
        cells = [fmt(block["orders"][str(o)][field]) for o in ORDERS]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def offender_table(results, keys, scheme):
    lines = [
        "| integrand | " + " | ".join(f"N={o}" for o in ORDERS) + " |",
        "|---|" + "---|" * len(ORDERS),
    ]
    for key in keys:
        block = results[scheme].get(key)
        if block is None:
            continue
        cells = [
            f"{block['orders'][str(o)]['intervals_above_1e-12']} / {block['num_intervals']}"
            for o in ORDERS
        ]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


CONVERGENCE_SCHEMA_NOTES = {
    "models": "model -> scheme -> integrand key -> block. Scheme is 'plain' (one Gauss panel per "
    "production interval), 'branch' (panels split at the QCD_EOS temperature break points) or "
    "'branch+knots' (also split at the T(z) spline's knots). LambdaCDMModel has only 'plain'. "
    "Within a model, every scheme is scored against the SAME break-aware converged reference.",
    "integrand keys": "'tau' = 1/H, 'cs_tau' = c_s/H, 'friction' = (3/2)(1+c_s^2)/(1+z); "
    "'rho_G@<k>' and 'rho_T@<k>' with <k> the JSON's f'{k:.6e}' key. All are integrands in "
    "u = log(1+z).",
    "orders": "Per order N: max_increment_rel_error is the largest relative error of a single "
    "production interval's Gauss increment against the converged reference; "
    "max_cumulative_abs_error and max_cumulative_rel_error are the largest absolute and relative "
    "errors of the cumulative value at a wkb_reference_data.json checkpoint (Mpc for tau/cs_tau, "
    "radians for friction and the residuals), the top checkpoint -- where the cumulative is "
    "identically zero -- being excluded from the relative measure; intervals_above_1e-12 counts "
    "intervals whose increment is worse than 1e-12 relative.",
    "decision": "N_tau, N_cs_tau, N_F, N_rho are the Gauss orders prompts 03-07 must use; "
    "recommended_scheme is the QCD break-point scheme they must build the tables under, and is "
    "one of candidate_schemes -- the schemes production can actually execute, since "
    "integration_break_points returns the equation-of-state temperature crossings and nothing "
    "else; control_scheme ('branch+knots') is measured and reported in knots_control but is not "
    "a candidate; rho_adaptive_fallback_required says whether NO fixed order <= 16 met the 1e-7 "
    "rad target for rho on QCD_Cosmology under the recommended scheme.",
    "grid": "The source-grid generation every figure in this block was measured on "
    "(tolerance-convergence README §5 rule 6). It is version 0, deliberately: this block scores "
    "the agreement of a fixed-order rule with the JSON's own reference values, which live on "
    "that grid.",
    "reference drift": "Per case, how far the adaptive reference moves across one decade of "
    "quad's epsrel (reference_drift_epsrel_step, loose -> the value used), the largest such "
    "movement (reference_drift_max) and where, the criterion it was scored against "
    "(reference_drift_threshold = the smallest difference this case reports, divided by "
    "reference_drift_criterion_ratio) and whether it passed. reference_drift_kind is 'relative' "
    "for the three primitives, whose decision reads max_cumulative_rel_error, and 'absolute' "
    "(radians) for the two residuals, whose decision reads max_cumulative_abs_error. The step is "
    "taken from the loose side because quad refuses an epsrel below 50*eps when epsabs = 0, so "
    "the reference is already at the method's floor. reference_error_bound is the larger of that "
    "movement and, for the primitives, the disagreement with the independent order-40 Gauss "
    "rule. Taken through ComputeTargets/tests/convergence_reference.reference_drift. "
    "reference_drift_notes may carry a warning about scipy.integrate's 100*eps rtol clamp, which "
    "is solve_ivp's floor and not quad's -- see "
    "[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve].",
    "knots_control": "What splitting every Gauss panel at the T(z) spline's interior knots still "
    "buys, against the recommended executable scheme. Production cannot perform that split, so "
    "this is a measurement and not an option.",
    "smoothness": "Finite-difference bounds on rho as a function of u = log(1+z), and the cubic "
    "spline interpolation error h^4 max|rho''''|/384 they imply -- the bound prompts 09 and 10 "
    "inherit for the residual spline.",
    "radiation_controls": "rho_G must be bit-exactly zero at every order in exact radiation "
    "(C == 0 identically); rho_T is scored against RadiationModel's closed-form primitive, of "
    "which review §12.4's 1/x_i - 1/x is the large-x asymptote (and carries the opposite sign in "
    "this campaign's convention).",
}


def write_json(payload, path=None):
    """
    Replace the ``convergence`` key of the fixture -- **and only that key**.

    ``tolerance-convergence`` prompt 04 §2.3: ``schema_version``, ``generated``, ``models``,
    ``baselines``, ``k_values``, ``k_keys`` and ``rho_anchor_efolds_subh`` at the top level are
    other prompts' fixtures and other campaigns' evidence. The read-modify-write below is what
    keeps them untouched, and ``git diff`` on the file is the check.

    ``path`` defaults to the fixture; a different path is the dry run §2.4 requires, in which the
    block is measured and inspected before anything is written into the tree.
    """
    path = REFERENCE_DATA_PATH if path is None else path
    with open(path, "r") as f:
        data = json.load(f)
    data["convergence"] = payload
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=False)
    print(
        f"\n** wrote the 'convergence' block to {path} "
        f"({os.path.getsize(path)} bytes)",
        flush=True,
    )


def write_markdown(p, path=None):
    keys = integrand_keys()
    d = p["decision"]
    geo = p["geometry"]["QCDModel"]
    lam = p["models"]["LambdaCDMModel"]
    qcd = p["models"]["QCDModel"]
    rec = d["recommended_scheme"]

    fallback = "required" if d["rho_adaptive_fallback_required"] else "not required"

    branch_rows = "\n".join(
        f"| `{b['name']}` | {b['T_GeV']:g} GeV | {b['z']:.6g} | {b['interval_index']} | "
        f"{b['breaks']} | {b['Hubble_relative_jump']:.2e} | {b['cs2_relative_jump']:.2e} | "
        f"{b['cs2_slope_relative_jump']:.2e} |"
        for b in geo["branch_boundaries"]
    )

    best_floor = d["best_primitive_floor_rel"]
    scheme_rows = "\n".join(
        f"| `{s}` | "
        + " | ".join(
            fmt(d["scheme_summary"][s]["primitive_floor_rel"][key])
            for key in PRIMITIVE_KEYS
        )
        + " | "
        + (
            str(d["scheme_summary"][s]["rho_min_order"])
            if d["scheme_summary"][s]["rho_min_order"] is not None
            else "**none $\\le16$**"
        )
        + " | "
        + (
            "**selected**"
            if s == rec
            else (
                "rejected: $\\rho$ target unreachable"
                if d["scheme_summary"][s]["rho_min_order"] is None
                else "rejected: primitive floor "
                + fmt(
                    max(
                        d["scheme_summary"][s]["primitive_floor_rel"][key]
                        / best_floor[key]
                        for key in PRIMITIVE_KEYS
                    ),
                    ".0f",
                )
                + "× the best"
            )
        )
        + " |"
        for s in SCHEME_ORDER
    )

    smooth_rows = "\n".join(
        f"| {model} | `{key}` | {v['max_d2_rho_du2']:.2e} | {v['max_d4_rho_du4']:.2e} | "
        f"{v['max_cubic_spline_error_rad']:.2e} | {fmt(v['max_cubic_spline_error_at_z'], '.4g')} |"
        for model, block in p["smoothness"].items()
        for key, v in block.items()
    )

    cost_rows = "\n".join(
        f"| {model} | `{key}` | {v['order']} | {v['integrand_evaluations']} | {v['seconds']:.3f} |"
        for model, block in p["build_cost"].items()
        for key, v in block.items()
        if isinstance(v, dict)
    )

    control_rows = "\n".join(
        f"| {kk} | {v['x_init']:.4g} | {v['x_final']:.4g} | "
        f"{v['at_N_rho']['max_rel_error_vs_closed_form']:.2e} | "
        f"{v['asymptote_relative_departure']:.2e} |"
        for kk, v in p["radiation_controls"]["rho_T"].items()
    )

    rho_rows = "\n".join(
        f"| `{key}` | {d['lambdacdm_rho_magnitude_rad'].get(key, float('nan')):.3e} | "
        f"{d['qcd_rho_magnitude_rad'].get(key, float('nan')):.3e} |"
        for key in residual_keys()
    )

    text = f"""# Gauss-order convergence of the WKB primitives on both production models

**Campaign:** [`prompts/GkTk-remedial/README.md`](../../prompts/GkTk-remedial/README.md) ·
**Prompt:** [`02-qcd-residual-convergence.md`](../../prompts/GkTk-remedial/02-qcd-residual-convergence.md)
**Generated:** {p["generated"]} by `{p["generator"]}` in {p["runtime_seconds"]:.0f} s
(Python {p["environment"]["python"]}, NumPy {p["environment"]["numpy"]}, SciPy {p["environment"]["scipy"]}).

> This file is **regenerated in full** by the script; do not hand-edit it. Every number below is a
> measurement, and the decision in §1 applies the rules of the prompt's §3 and §4 to those numbers.

Review §11 names this measurement "the first test of any implementation": whether the fixed-order
Gauss–Legendre-per-production-interval rule of review §7 converges across the QCD model's spline
knots. It does not, and the reason is stronger than the review anticipated — **`QCD_EOS.G(T)` and
`Gs(T)` jump discontinuously at their branch boundaries, so `QCD_Cosmology`'s `H(z)` is itself
discontinuous** — and a third break point, the clamp inside `QCD_EOS.w(T)` at `EOS_T_LO`, kinks
`c_s^2` (§2). The remedy is deterministic and costs {100.0 * (p["build_cost"]["QCDModel"]["three_tables_total_evaluations"] / p["build_cost"]["LambdaCDMModel"]["three_tables_total_evaluations"] - 1.0):.0f} % more integrand evaluations:
split the containing production interval at the break point, which is known in closed form from
the cosmology before any integration happens.

---

## 1. Decision

| quantity | order | how it was chosen |
|---|---|---|
| $N_\\tau$ | **{d["N_tau"]}** | smallest order within {d["floor_factor"]:.0f}× the floor of the *relative* cumulative checkpoint error, on both models |
| $N_{{\\tau_s}}$ | **{d["N_cs_tau"]}** | same rule, integrand $c_s/H$ |
| $N_F$ | **{d["N_F"]}** | same rule, integrand $\\tfrac32(1+c_s^2)/(1+z)$ |
| $N_\\rho$ | **{d["N_rho"]}** | smallest order with cumulative $|\\delta\\rho|\\le10^{{-7}}$ rad on both models for all three $k$ **and** reproducing the exact-radiation $\\rho_T$ to $10^{{-14}}$ relative (§7) |

**Adaptive fallback for $\\rho$: {fallback}** — a fixed order $\\le16$ reaches the target on both
models under the recommended scheme, so prompt 05 needs no adaptive path.

The two $\\rho$ criteria are not redundant, and the second is what fixes the answer. $\\rho$ is
$10^{{-3}}$ rad on QCD and $10^{{-10}}$ rad on LambdaCDM, so an *absolute* $10^{{-7}}$ rad target is
met by rules that have not converged at all: order 2 passes it under every scheme
({fmt(d["N_rho_detail"]["QCDModel"]["errors_abs"]["2"])} rad on QCD) while carrying
{fmt(d["N_rho_detail"]["RadiationModel"]["errors_rel"]["2"])} relative error on a control whose
answer is known in closed form. Order {d["N_rho"]} is at the floor on both
({fmt(d["N_rho_detail"]["QCDModel"]["errors_abs"][str(d["N_rho"])])} rad and
{fmt(d["N_rho_detail"]["RadiationModel"]["errors_rel"][str(d["N_rho"])])} relative), and no higher
order improves either.

**Recommended build scheme: `{rec}`** — one order-$N$ Gauss panel per production interval on
LambdaCDM, and on `QCD_Cosmology` the same panel split at every break point of the cosmology that
falls inside it. The scheme was chosen from the measurement, not in advance: the cheapest of the
three that admits a fixed order $\\le16$ for $\\rho$ *and* loses nothing on the primitives relative
to the best scheme measured.

| QCD scheme | floor, `tau` | floor, `cs_tau` | floor, `friction` | smallest $N$ meeting §3's $\\rho$ target | verdict |
|---|---|---|---|---|---|
{scheme_rows}

The three floor columns are the smallest relative cumulative checkpoint error any order in
{list(ORDERS)} achieves under that scheme. **Note what disqualifies `plain`: not $\\rho$, but
$\\tau$.** The residual is so small that even an unconverged rule delivers it to $10^{{-9}}$ rad
absolute, so review §11's worry — "a fixed Gauss rule for $\\rho$ could converge slowly across
[the knots]" — is real but harmless. What is *not* harmless is the same roughness in the leading
term: `plain` leaves the cumulative $\\tau$ at best
{fmt(d["scheme_summary"]["plain"]["primitive_floor_abs"]["tau"])} Mpc from the reference however
high the order — {fmt(d["scheme_summary"]["plain"]["primitive_floor_abs"]["tau"] * PHASE_K, ".2f")}
rad of phase at $k=3\\times10^8$, and
{fmt(d["scheme_summary"]["plain"]["tau_errors_abs"]["4"] * PHASE_K, ".2f")} rad at order 4 —
against {fmt(d["scheme_summary"][rec]["tau_errors_abs"][str(d["N_tau"])] * PHASE_K, ".1e")} rad
under `{rec}`. The test the review asked for gives the right answer for a different reason than
it expected.

Magnitude of $\\rho$ over the whole WKB range, so that a later reader knows what carrying it buys
(review §6: on LambdaCDM $\\rho_G$ is below the $\\varepsilon k\\tau$ floor, but the machinery is
required for $T_k$ and for QCD, so one code path is carried — README §7 D3):

| integrand | $|\\rho|$, LambdaCDM [rad] | $|\\rho|$, QCD [rad] |
|---|---|---|
{rho_rows}

---

## 2. Where the QCD model's features sit on the production grid

The production source grid has {geo["num_nodes"]} nodes and {geo["num_intervals"]} intervals, with
$\\Delta u$ from {geo["du_min"]:.4g} to {geo["du_max"]:.4g} in $u=\\log(1+z)$.

**`QCD_EOS` break points.** `G(T)`, `Gs(T)` switch between the Saikawa–Shirai fit and an asymptotic
constant at fixed temperatures and the two pieces do not match, so `H(z)` *jumps*. `w(T)` clamps
its argument to `EOS_T_LO` below `EOS_T_LO`, so $c_s^2$ is continuous there but its slope is not:
that one is a kink, not a jump, which the last two columns separate. `T_HI = 1e16` GeV is far above
the production range and does not appear.

| constant | $T$ | $z$ | production interval | breaks | rel. jump in $H$ | rel. jump in $c_s^2$ | rel. jump in $dc_s^2/du$ |
|---|---|---|---|---|---|---|---|
{branch_rows}

Prompt 01 reported the two `G(T)` boundaries as $z=4.191\\times10^7$ and $8.579\\times10^{{11}}$: those
are the *lower edges of the containing production intervals*, which §3.2 below reproduces exactly
as the worst-converging intervals. The $z$ column here is the break point itself, from a
`root_scalar` solve of $T_{{\\rm photon}}(z)=T$ on the model's own spline to $10^{{-15}}$, and is what
the subdivision splits at.

**`T(z)` spline knots.** The model builds `T(z)` as a {geo["T_spline_samples"]}-point
`make_interp_spline` in $\\log(1+z)$ (`LambdaCDM_GenericEOS._build_T_z_spline`);
{geo["T_spline_knots_in_range"]} of its knots lie inside the production range, a median $\\Delta u$
of {fmt(geo["T_spline_knot_du"], ".4g")} apart. That is
{geo["knots_per_production_interval"]["mean"]:.3f} knots per production interval — **one knot every
{1.0 / max(geo["knots_per_production_interval"]["mean"], 1e-30):.1f} intervals** — and
{geo["knots_per_production_interval"]["intervals_containing_a_knot"]} of {geo["num_intervals"]}
intervals contain one. (The prompt's estimate of ~20 knots per interval came from review §6's
*quintic $\\epsilon$ spline at 2000 points per decade*, which is the review's own reference
construction, not the model's `T(z)` spline. The estimate is wrong by two orders of magnitude in
the reassuring direction.)

**The QCD transition.** $\\epsilon$ departs from 2 by its largest amount,
$\\epsilon={geo["epsilon_departure"]["epsilon"]:.6f}$, at $z={geo["epsilon_departure"]["z"]:.4g}$
(production interval {geo["epsilon_departure"]["interval_index"]}), reproducing review §6's
"$\\epsilon$ departs from 2 by up to 0.150 at $z\\approx1.2\\times10^{{12}}$".

**The $c_s^2$ transition.** The node-to-node $|dw/du|$ is largest at
$z={geo["cs2_transition"]["z"]:.4g}$ (production interval
{geo["cs2_transition"]["interval_index"]}), i.e. at the QCD transition and within a few production
intervals of the `T_120_MEV` boundary, not at matter–radiation equality — the fall of $c_s^2$
towards equality is spread over many decades and is gentle per interval. Once `T_120_MEV` is a
split point the remaining variation is smooth and needs no further treatment; the `friction` rows
of §3.4 and §4 confirm it.

---

## 3. Per-interval convergence of the increments

The largest relative error of a *single production interval's* Gauss increment, against the
converged reference. **The reference is break-aware**: within each production interval it is
`quad` (`epsabs=0`, `epsrel={QUAD_EPSREL:g}`, `limit=400`) on each sub-panel between break points,
cross-checked per interval against `quad` at `epsrel={QUAD_CROSSCHECK_EPSREL:g}` and against
composite Gauss–Legendre order 40 with bisection. Without that, the reference would be as wrong as
the rule it is meant to score across a discontinuity.

### 3.1 LambdaCDM (`plain`; the model has no break points)

{increment_table(lam, keys, "plain")}

### 3.2 QCD, `plain` — one panel per production interval

{increment_table(qcd, keys, "plain")}

### 3.3 QCD, `branch` — split at the `QCD_EOS` break points only

{increment_table(qcd, keys, "branch")}

### 3.4 QCD, `branch+knots` — also split at the `T(z)` spline's knots

{increment_table(qcd, keys, "branch+knots")}

### 3.5 How many intervals are worse than $10^{{-12}}$ relative

QCD, `plain`:

{offender_table(qcd, keys, "plain")}

QCD, `branch`:

{offender_table(qcd, keys, "branch")}

QCD, `branch+knots`:

{offender_table(qcd, keys, "branch+knots")}

The `rho_G` rows do not reach zero under `branch+knots` and this is not a convergence failure. On
QCD a single production interval's $\\rho_G$ increment is $10^{{-6}}$–$10^{{-15}}$ rad, formed from a
$C$ that is itself a difference of $O(1)$ spline values, so a *relative* $10^{{-12}}$ on the increment
is being asked of a quantity already at its own rounding floor. The absolute cumulative error is
what matters and §4.2 gives it as {fmt(qcd["branch+knots"]["rho_G@" + k_key(3.0e8)]["orders"][str(d["N_rho"])]["max_cumulative_abs_error"])} rad.

---

## 4. Cumulative error at the checkpoints

The quantity the tables actually deliver: the largest error of the cumulative value at a
`wkb_reference_data.json` checkpoint, scored against the JSON references — mpmath at 40 digits for
LambdaCDM, converged adaptive quadrature of the double-precision integrand for QCD. The LambdaCDM
rows therefore include the double-precision evaluation floor of `LambdaCDM.Hubble`
(`[01-lambdacdm-hubble-rounding-floor]`, 2–9e-15 relative) and the QCD rows do not.

Two floors are visible in these tables and neither is a property of the Gauss rule. On LambdaCDM
the $\\tau$ and $\\tau_s$ rows bottom out at $\\approx2\\times10^{{-15}}$, the `Hubble` rounding floor.
On QCD they bottom out at $\\approx1.9\\times10^{{-14}}$, which is exactly the disagreement between
this script's break-aware reference and the JSON reference it is scored against
(`json_vs_reference_max_rel` in the JSON block); the QCD $\\tau$ column therefore *cannot* resolve
below $2\\times10^{{-14}}$, and 1.9e-14 should be read as "at the reference's own floor", not as a
quadrature error. Both are below README §6's $2\\times10^{{-14}}$ target for $\\tau$ at the nodes.

### 4.1 Relative

LambdaCDM (`plain`):

{cumulative_table(lam, keys, "plain", "max_cumulative_rel_error")}

QCD, `plain`:

{cumulative_table(qcd, keys, "plain", "max_cumulative_rel_error")}

QCD, `{rec}` (the recommended scheme):

{cumulative_table(qcd, keys, rec, "max_cumulative_rel_error")}

### 4.2 Absolute (Mpc for $\\tau$, $\\tau_s$; radians for $F$ and the residuals)

LambdaCDM (`plain`):

{cumulative_table(lam, keys, "plain", "max_cumulative_abs_error")}

QCD, `plain`:

{cumulative_table(qcd, keys, "plain", "max_cumulative_abs_error")}

QCD, `{rec}`:

{cumulative_table(qcd, keys, rec, "max_cumulative_abs_error")}

---

## 5. Smoothness of the residual — the bound prompts 09 and 10 inherit

$\\rho(u)=\\int_u^{{u_{{\\rm anchor}}}}g$, so $\\rho''=-g'$ and $\\rho''''=-g'''$; both are central
finite differences of the integrand on the production grid's own local spacing $h$. The last column
is $h^4\\max|\\rho''''|/384$, the cubic-spline interpolation error of the *residual* spline that
`PrimitivePhase` (prompt 09) and `TkSourceFunctions` (prompt 10) will build. On QCD the stencil
straddles the break points of §2 at a handful of nodes, so the QCD maxima are upper bounds
contaminated by those few points rather than a statement about the smooth part. Even so, the worst
predicted $\\varphi$-spline error is $3\\times10^{{-7}}$ rad — well inside README §6's $10^{{-6}}$ rad
consumer target, and set by the QCD transition, not by the residual's generic smoothness.

| model | integrand | $\\max|\\rho''|$ | $\\max|\\rho''''|$ | $h^4\\max|\\rho''''|/384$ [rad] | at $z$ |
|---|---|---|---|---|---|
{smooth_rows}

---

## 6. Build cost at the recommended orders

Integrand evaluations (one background evaluation each) and wall time for one full table on the
{geo["num_intervals"]}-interval production grid. `QCDModel` construction itself
(`compute_background` plus the derivative splines) costs {p["qcd_model_build_seconds"]:.3f} s and is
not counted here. The `rho` rows are per $k$ and per anchor, at $k=3\\times10^8$ (the longest
range); the three tables are built once per model.

| model | table | order | integrand evaluations | seconds |
|---|---|---|---|---|
{cost_rows}

Three tables together: {p["build_cost"]["LambdaCDMModel"]["three_tables_total_evaluations"]}
evaluations in {p["build_cost"]["LambdaCDMModel"]["three_tables_total_seconds"]:.3f} s (LambdaCDM),
{p["build_cost"]["QCDModel"]["three_tables_total_evaluations"]} in
{p["build_cost"]["QCDModel"]["three_tables_total_seconds"]:.3f} s (QCD, `{rec}`).

---

## 7. Exact-radiation controls

`RadiationModel` has $C\\equiv0$ for the Green's function, so **every** Gauss increment of $\\rho_G$
must be bit-exactly zero at every order. Measured, at all {len(ORDERS)} orders and all three $k$:
{"all bit-exactly zero" if all(p["radiation_controls"]["rho_G_exactly_zero"].values()) else "**NOT all zero**"}.

$\\rho_T$ is scored against `RadiationModel`'s closed-form primitive
$g(s)=-2s/(\\sqrt{{a^2-2s^2}}+a)+\\sqrt2\\arcsin(\\sqrt2 s/a)$, $a=k/(\\sqrt3 H_0)$. Review §12.4's
$1/x_i-1/x$ is that primitive's *large-$x$ asymptote*, and carries the opposite sign in this
campaign's convention (`wkb_reference.RadiationModel` docstring), so its departure from the closed
form is reported rather than used as the reference — at $x_{{T,i}}\\approx12$ that departure is a
few parts in $10^3$, far above any quadrature error.

| $k$ | $x_{{T,\\rm init}}$ | $x_{{T,\\rm final}}$ | rel. error at $N_\\rho$ | departure of the asymptote from the closed form |
|---|---|---|---|---|
{control_rows}

---

## 8. What prompts 03–07 must do

1. Build every table with the orders in §1: $N_\\tau={d["N_tau"]}$, $N_{{\\tau_s}}={d["N_cs_tau"]}$,
   $N_F={d["N_F"]}$, $N_\\rho={d["N_rho"]}$.
2. On a cosmology that has break points — `QCD_Cosmology` does, through `QCD_EOS.T_LO`,
   `QCD_EOS.EOS_T_LO`, `QCD_EOS.T_120_MEV` and the `T(z)` spline's knots — **split the production
   interval at each break point it contains** and integrate each sub-panel separately. On
   LambdaCDM the break-point list is empty and the scheme reduces to review §7's as written.
   Note that `EOS_T_LO` breaks `c_s^2` only, so it matters for $\\tau_s$, $F$ and $\\rho_T$ and not
   for $\\tau$ or $\\rho_G$; splitting on the union costs nothing measurable and is simpler.
3. The cumulative errors of §4 under `{rec}` are the accuracies prompt 13 should expect to
   reproduce; the `plain` rows are what the review's design as written would have given.
"""

    path = MARKDOWN_PATH if path is None else path
    with open(path, "w") as f:
        f.write(text)
    print(f"** wrote {path}", flush=True)


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Gauss-order convergence of the WKB primitives; regenerates the 'convergence' block "
            "of ComputeTargets/tests/wkb_reference_data.json"
        )
    )
    parser.add_argument(
        "--json-out",
        default=None,
        metavar="PATH",
        help=(
            "write the regenerated block into PATH instead of the fixture. PATH must already be "
            "a wkb_reference_data.json-shaped file; only its 'convergence' key is replaced. This "
            "is the dry run of tolerance-convergence prompt 04 §2.4 -- look, then regenerate."
        ),
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="measure and print, write no JSON at all",
    )
    parser.add_argument(
        "--legacy-markdown",
        action="store_true",
        help=(
            "also rewrite docs/gktk-remedial/RESIDUAL-CONVERGENCE.md. Off by default: that is "
            "GkTk-remedial prompt 02's published document and verification documents are "
            "additive (tolerance-convergence README §5 rule 7)."
        ),
    )
    return parser.parse_args(argv)


def main(args):
    t_start = time.perf_counter()
    references = load_references()

    print("** building stand-in models", flush=True)
    lam = LambdaCDMModel()
    z_init = horizon_exit_z(
        lam.cosmology, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid = production_source_grid(z_init)
    z_nodes = np.array([z.z for z in grid], dtype=float)
    t0 = time.perf_counter()
    qcd = QCDModel(grid)
    qcd_build_seconds = time.perf_counter() - t0

    rad = RadiationModel()
    z_init_rad = horizon_exit_z(
        rad, PRODUCTION_LARGEST_K_INV_MPC, -PRODUCTION_SUPERHORIZON_EFOLDS
    )
    grid_rad = production_source_grid(z_init_rad)
    z_nodes_rad = np.array([z.z for z in grid_rad], dtype=float)

    u_ascending = np.log1p(z_nodes)[::-1].copy()
    u_lo, u_hi = float(u_ascending[0]), float(u_ascending[-1])

    # -----------------------------------------------------------------------------------------
    # item 3: the geometry of the QCD features on the production grid
    # -----------------------------------------------------------------------------------------
    print("\n** QCD features on the production grid", flush=True)
    branch, knots = qcd_break_points(qcd.cosmology, u_lo, u_hi)
    geometry = {
        "num_nodes": int(len(z_nodes)),
        "num_intervals": int(len(z_nodes) - 1),
        "du_min": float(np.diff(u_ascending).min()),
        "du_max": float(np.diff(u_ascending).max()),
        "branch_boundaries": [
            {
                "name": name,
                "T_GeV": T,
                "breaks": what,
                "z": float(expm1(u)),
                "u": float(u),
                "interval_index": int(np.searchsorted(u_ascending, u)) - 1,
            }
            for name, T, u, what in branch
        ],
        "T_spline_samples": 500,
        "T_spline_knots_in_range": int(len(knots)),
        "T_spline_knot_du": (
            float(np.median(np.diff(knots))) if len(knots) > 1 else None
        ),
    }
    counts = np.histogram(knots, bins=u_ascending)[0]
    geometry["knots_per_production_interval"] = {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "intervals_containing_a_knot": int((counts > 0).sum()),
    }
    # The size of the jump at each break point, in the *value* of H and c_s^2 and in the
    # one-sided u-derivative of c_s^2. Everything is evaluated a hair either side of the crossing,
    # far closer than the T(z) spline's own knot spacing, so what is measured is the EOS branch
    # change and not the smooth variation with z. The derivative column is what separates a jump
    # (G, Gs at T_LO and T_120_MEV) from a kink (w clamped at EOS_T_LO, where the value is
    # continuous to rounding but the slope is not).
    for entry in geometry["branch_boundaries"]:
        u = entry["u"]
        z_below, z_above = expm1(u - 1.0e-10), expm1(u + 1.0e-10)
        H_lo, H_hi = qcd.functions.Hubble(z_below), qcd.functions.Hubble(z_above)
        w_lo = qcd.functions.wPerturbations(z_below)
        w_hi = qcd.functions.wPerturbations(z_above)
        entry["Hubble_relative_jump"] = fabs(H_hi - H_lo) / fabs(H_hi)
        entry["cs2_relative_jump"] = fabs(w_hi - w_lo) / fabs(w_hi)

        du = 1.0e-6
        w = qcd.functions.wPerturbations
        slope_lo = (w(expm1(u - du)) - w(expm1(u - 2.0 * du))) / du
        slope_hi = (w(expm1(u + 2.0 * du)) - w(expm1(u + du))) / du
        denom = max(fabs(slope_lo), fabs(slope_hi))
        entry["cs2_slope_relative_jump"] = (
            fabs(slope_hi - slope_lo) / denom if denom > 0.0 else 0.0
        )
        print(
            f"   {entry['name']:<10} T={entry['T_GeV']:g} GeV  z={entry['z']:.6g}  "
            f"dH/H={entry['Hubble_relative_jump']:.3e}  "
            f"dcs2/cs2={entry['cs2_relative_jump']:.3e}  "
            f"d(dcs2/du)={entry['cs2_slope_relative_jump']:.3e}",
            flush=True,
        )
    print(
        f"   T(z) spline: {geometry['T_spline_knots_in_range']} interior knots in range, "
        f"{geometry['knots_per_production_interval']['mean']:.4f} per production interval",
        flush=True,
    )

    # the epsilon departure and the c_s^2 transition
    eps = np.array([qcd.functions.epsilon(z) for z in z_nodes])
    w_qcd = np.array([qcd.functions.wPerturbations(z) for z in z_nodes])
    mask = z_nodes > 1e10
    j_eps = int(np.argmin(eps[mask]))
    z_eps = float(z_nodes[mask][j_eps])
    dw_du = np.abs(np.diff(w_qcd) / np.diff(np.log1p(z_nodes)))
    j_cs = int(np.argmax(dw_du))
    geometry["epsilon_departure"] = {
        "z": z_eps,
        "epsilon": float(eps[mask][j_eps]),
        "interval_index": int(np.searchsorted(u_ascending, log1p(z_eps))) - 1,
    }
    geometry["cs2_transition"] = {
        "z": float(z_nodes[j_cs]),
        "max_abs_dw_du": float(dw_du[j_cs]),
        "interval_index": int(np.searchsorted(u_ascending, log1p(float(z_nodes[j_cs]))))
        - 1,
    }
    print(
        f"   epsilon minimum above z=1e10: eps={geometry['epsilon_departure']['epsilon']:.6f} "
        f"at z={z_eps:.4g}",
        flush=True,
    )
    print(
        f"   steepest c_s^2 transition at z={geometry['cs2_transition']['z']:.4g}",
        flush=True,
    )

    # -----------------------------------------------------------------------------------------
    # items 1 and 2
    # -----------------------------------------------------------------------------------------
    branch_u = np.array([u for _, _, u, _ in branch], dtype=float)
    all_u = np.sort(np.concatenate([branch_u, knots])) if len(branch_u) else knots

    lam_results = measure_model(
        "LambdaCDMModel",
        lam,
        z_nodes,
        u_ascending,
        references["models"]["LambdaCDMModel"],
        {"plain": np.array([])},
        np.array([]),
    )
    qcd_results = measure_model(
        "QCDModel",
        qcd,
        z_nodes,
        u_ascending,
        references["models"]["QCDModel"],
        {"plain": np.array([]), "branch": branch_u, "branch+knots": all_u},
        all_u,
    )

    # -----------------------------------------------------------------------------------------
    # item 4: smoothness of rho
    # -----------------------------------------------------------------------------------------
    print("\n** smoothness of the residuals", flush=True)
    smooth = {}
    for label, model in (("LambdaCDMModel", lam), ("QCDModel", qcd)):
        refs = references["models"][label]
        smooth[label] = {}
        for key, _, maker in RESIDUAL_SPECS:
            for k in REFERENCE_K_VALUES:
                kk = k_key(k)
                u_anchor = log1p(float(refs["rho_anchor_z"][kk]))
                entry = smoothness(model.functions, u_ascending, u_anchor, maker, k)
                smooth[label][f"{key}@{kk}"] = entry
                print(
                    f"   {label:<15} {key}@{kk}: |rho''|<={entry['max_d2_rho_du2']:.2e}, "
                    f"|rho''''|<={entry['max_d4_rho_du4']:.2e}, h^4|rho''''|/384 <= "
                    f"{entry['max_cubic_spline_error_rad']:.2e} rad "
                    f"(at z={fmt(entry['max_cubic_spline_error_at_z'], '.4g')})",
                    flush=True,
                )

    # -----------------------------------------------------------------------------------------
    # the exact-radiation controls, which N_rho's second criterion reads, then the decision
    # -----------------------------------------------------------------------------------------
    controls = radiation_controls(
        rad, z_nodes_rad, references["models"]["RadiationModel"]
    )
    print("\n** exact-radiation controls", flush=True)
    print(
        f"   rho_G bit-exactly zero at every order and k: "
        f"{all(controls['rho_G_exactly_zero'].values())}",
        flush=True,
    )
    for kk, v in controls["rho_T"].items():
        print(
            f"   rho_T k={kk}: rel error vs the closed form "
            + " ".join(
                f"N={o}:{v['orders'][str(o)]['max_rel_error_vs_closed_form']:.2e}"
                for o in ORDERS
            )
            + f"; asymptote departs by {v['asymptote_relative_departure']:.3e}",
            flush=True,
        )

    decision = decide(lam_results, qcd_results, controls)
    for v in controls["rho_T"].values():
        v["at_N_rho"] = v["orders"][str(decision["N_rho"])]
    print("\n** decision", flush=True)
    print(
        json.dumps(
            {
                key: decision[key]
                for key in (
                    "N_tau",
                    "N_cs_tau",
                    "N_F",
                    "N_rho",
                    "recommended_scheme",
                    "rho_adaptive_fallback_required",
                    "rho_fixed_order_works_without_subdivision",
                )
            },
            indent=1,
        ),
        flush=True,
    )

    scheme_breaks = {
        "plain": np.array([]),
        "branch": branch_u,
        "branch+knots": all_u,
    }
    cost = build_cost(
        (
            ("LambdaCDMModel", lam, np.array([])),
            ("QCDModel", qcd, scheme_breaks[decision["recommended_scheme"]]),
        ),
        u_ascending,
        references,
        decision,
        scheme_breaks,
    )
    print("\n** build cost", flush=True)
    print(json.dumps(cost, indent=1), flush=True)

    elapsed = time.perf_counter() - t_start

    import scipy

    payload = {
        "schema_version": 2,
        "generated": date.today().isoformat(),
        "generator": "docs/gktk-remedial/residual_convergence.py",
        "campaign": "prompts/tolerance-convergence (prompt 04, board item T7)",
        "superseded": {
            "generated": "2026-09-10",
            "campaign": "prompts/GkTk-remedial (prompt 02)",
            "why": (
                "the T(z) representation qcd-background-audit prompts 04-06 replaced, an "
                "integration_break_points that returned the interpolant's ~404 knots which "
                "prompt 07 removed, and the background derivative splines prompt 13 segmented "
                "(tolerance-convergence RECONCILIATION.md §5, "
                "[01-convergence-block-has-a-separate-generator])"
            ),
        },
        "grid": {
            "generation": GRID_GENERATION,
            "samples": int(len(z_nodes)),
            "z_init": float(z_init),
            "z_end": float(z_nodes[-1]),
            "anchor": (
                "horizon_exit_z(LambdaCDM, 3e8/Mpc, -5) -- the version-0 grid consults no "
                "cosmology, so the same lattice serves every model here"
            ),
            "why_version_0": (
                "what this block scores is the agreement of a fixed-order Gauss rule with the "
                "JSON's own `checkpoints` and `rho_*` reference values, and those live on the "
                "version-0 grid, as do the test modules that read this block. Production has "
                "built version 2 since qcd-background-audit prompt 15; the version-2 sweep at "
                "every production wavenumber is docs/tolerance-convergence/ORDER-AUDIT.md's "
                "(tolerance-convergence README §5 rule 6)"
            ),
            "radiation_control_samples": int(len(z_nodes_rad)),
            "radiation_control_z_init": float(z_init_rad),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "orders": list(ORDERS),
        "schemes": list(SCHEME_ORDER),
        "schema": CONVERGENCE_SCHEMA_NOTES,
        "geometry": {"QCDModel": geometry},
        "decision": decision,
        "smoothness": smooth,
        "build_cost": cost,
        "radiation_controls": controls,
        "models": {"LambdaCDMModel": lam_results, "QCDModel": qcd_results},
        "runtime_seconds": elapsed,
        "qcd_model_build_seconds": qcd_build_seconds,
    }

    if args.no_json:
        print(
            "\n** --no-json: the 'convergence' block was measured and not written",
            flush=True,
        )
    else:
        write_json(payload, args.json_out)
    if args.legacy_markdown:
        write_markdown(payload)
    else:
        print(
            f"\n** --legacy-markdown not given: {MARKDOWN_PATH} is left as GkTk-remedial "
            "prompt 02 published it",
            flush=True,
        )
    print(f"\n** total runtime {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
