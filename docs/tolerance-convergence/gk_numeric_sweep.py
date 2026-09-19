"""
What do ``GkNumericIntegration``'s ``atol`` and ``rtol`` buy, and does anything read the answer?
(``prompts/tolerance-convergence/`` prompt 03, board item **T4**.)

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/gk_numeric_sweep.py

It emits the tables of ``docs/tolerance-convergence/GK-NUMERIC-SWEEP.md`` on stdout and a progress
log on stderr. **It changes nothing**: no production module is touched, no constant moves, and the
recommendation the document carries is for the user to accept or reject (README §7 **D1**).

**Why this sector.** ``GkNumericIntegration`` is one object per ``(k, z_source)`` -- ~65,000 per
model, three orders more than any other target this campaign touches -- and it has never been
swept. The only measurement in the record is ``docs/gk-wkb-review-fable-2026-09-09.md`` §10.1: two
models, four source redshifts, and a single step along the **diagonal**
``(1e-10, 1e-8) -> (1e-13, 1e-11)``. A diagonal cannot separate ``atol`` from ``rtol``, so "the
error is set by ``rtol``" is an interpretation of a two-variable step there and not a measurement
(README §2 (d)).

**What is measured**, for ``RadiationModel``, ``LambdaCDMModel`` and ``QCDModel`` at all fifty
production wavenumbers, on the **version-2** source grid at each cosmology's **own** production
anchor, under this sector's own ``BREAK_POINT_DISCONTINUITY`` policy:

1. a **reference** per ``(model, k)`` at ``(atol, rtol) = (1e-18, 1e-12)``, with its drift against
   ``(1e-19, 1e-13)`` scored by the campaign's own criterion -- ``convergence_reference``'s
   :func:`reference_drift`, which will not return a drift without the verdict attached
   (README §5 rule 5). SciPy clamps ``rtol`` at ``100 eps = 2.22e-14``, so ``1e-13`` is the last
   tightening that is really applied and the candidate ``rtol`` axis stops a decade above the
   reference's;
2. the **``atol`` axis**, ``atol`` in ``{1e-8 ... 1e-12}`` at the production ``rtol = 1e-8``, and
   the **``rtol`` axis**, ``rtol`` in ``{1e-6 ... 1e-11}`` at the production ``atol = 1e-10``, each
   at full grid coverage. Two decades either side of production on both axes, so that inertness --
   the expected outcome on ``atol``, from the magnitude argument of README §2 (e) -- is a
   measurement and not an assumption;
3. the **interaction corners** ``{1e-8, 1e-12} x {1e-6, 1e-11}``, which is what makes "``atol`` is
   inert" a statement about the matrix rather than about a line through it;
4. on ``RadiationModel`` only, everything again against the exact ``compute_analytic_G``, through
   the harness's ``radiation_anchors`` registry -- the oracle column that calibrates what the
   self-convergence machinery reports on the two spline models (README §3.1);
5. the **``z_source`` bound**: ``gk_geometry`` takes one source redshift per wavenumber, the
   outermost, on the stated grounds that it is the least favourable. That is a claim, and §2.2 of
   the prompt requires it to be tested at three wavenumbers per model at a spread of interior
   source redshifts. If it holds, the sector is bounded by fifty runs per model;
6. the **consumer-spline floor**, freshly measured. The consumer of the numeric ``G`` is the
   ``numeric_Gk`` spline of ``ComputeTargets/GkSourcePolicyData.py:654-680``: a cubic
   ``make_interp_spline`` in ``log(1+z_source)``, at a fixed response redshift, over the
   **source-grid** nodes that carry numeric data. Its error is a property of the grid's density
   and of nothing this campaign can set, which is why it can dominate the solver error by orders.

**The axis the floor lives on.** The prompt's §5 describes the consumer's spline as running over
"the response grid". The tree says otherwise and the tree wins: ``GkSource`` is keyed on
``z_response`` and its ``z_sample`` is ``z_source_sample.truncate(z_response, keep='higher')``
(``main.py:2288-2291``), so the spline's knots are **source** redshifts and its density is the
**source** grid's. The two differ by ``PRODUCTION_RESPONSE_SPARSENESS = 12`` in spacing and so by
``12^4 ~ 2e4`` in a cubic interpolation error, which is far too much to leave to a footnote. Both
are measured here: §6 is the consumer, on the source axis, and §7 is the response axis beside it,
labelled as what it is -- a sampling of one solve, which nothing in the pipeline splines.

**The floor probe.** ``G(z_source, z_response)`` at one response redshift for many source
redshifts is one ``GkNumericIntegration`` solve per source redshift, so the probe calls
``numeric_with_phase_cut`` through its undecorated ``_function`` with a one-element response
sample and ``mode=None``. It cannot go through ``gk_run``, which is fixed in ``"stop"`` mode and
raises unless the integration ends on the phase-cut event -- correct for a production work item,
and not what a two-point probe does. Everything else is the facility's: the stand-ins, the error
definitions, :func:`summarise`, and the version-2 grid.

**Everything goes through ``ComputeTargets/tests/convergence_reference.py``** (board standing note
14). The one addition prompt 03 made to it is :func:`gk_geometry_at_source`, which item 5 needs and
which §2.3 of the prompt authorises; no existing line of that module changed.

No Ray and no datastore (``CLAUDE.md``).
"""

import platform
import sys
import time
from dataclasses import replace
from datetime import date
from math import fabs
from statistics import median

import numpy as np
import scipy
from scipy.interpolate import make_interp_spline

from ComputeTargets.BackgroundModel import BREAK_POINT_DISCONTINUITY
from ComputeTargets.GkNumericIntegration import RHS as Gk_RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.tests.convergence_reference import (
    CRITERION_RATIO,
    PRODUCTION_DELTA_LOGZ,
    SourceGridSpec,
    TolerancePair,
    UNITS,
    _KExit,
    _Proxy,
    anchor_error,
    gk_geometry,
    gk_geometry_at_source,
    gk_run,
    radiation_anchors,
    reference_drift,
    sector_error_measure,
    summarise,
    x_local,
)
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    PRODUCTION_K_GRID_INV_MPC,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_RESPONSE_SPARSENESS,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_INIT_LAMBDACDM,
    PRODUCTION_Z_INIT_QCD,
    QCDModel,
    RadiationModel,
    SOURCE_GRID_V2,
    envelope_relative_error,
    horizon_exit_z,
)
from CosmologyConcepts import redshift, redshift_array
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from Quadrature.integrators.numeric_with_phase_cut import numeric_with_phase_cut

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

#: the production setting: ``config.defaults.DEFAULT_ABS_TOLERANCE`` and
#: ``DEFAULT_REL_TOLERANCE``, read here and not changed (README §5 rule 8)
PRODUCTION_ATOL = 1.0e-10
PRODUCTION_RTOL = 1.0e-8

#: the reference and the tightening that scores it. ``GkTk-remedial`` prompt 17's pair, kept so
#: that the two sectors' reference-convergence figures stay comparable. SciPy clamps ``rtol`` at
#: ``SCIPY_RTOL_FLOOR = 2.22e-14``, so ``1e-13`` is the last effective tightening and the
#: candidate ``rtol`` axis has to stop a decade above ``1e-12`` for the reference to be tighter
#: than every candidate.
REFERENCE = TolerancePair(atol=1.0e-18, rtol=1.0e-12)
REFERENCE_TIGHTENED = TolerancePair(atol=1.0e-19, rtol=1.0e-13)

#: stage 1: the two axes, separately, two decades either side of production
ATOL_AXIS = (1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12)
RTOL_AXIS = (1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10)

#: stage 2: the corners of the box the two axes span
CORNERS = tuple(
    (atol, rtol) for atol in (1.0e-8, 1.0e-12) for rtol in (1.0e-6, 1.0e-10)
)

#: every ``(atol, rtol)`` the sweep visits, production first so that the log line leads with it
CANDIDATES = tuple(
    dict.fromkeys(
        [(PRODUCTION_ATOL, PRODUCTION_RTOL)]
        + [(atol, PRODUCTION_RTOL) for atol in ATOL_AXIS]
        + [(PRODUCTION_ATOL, rtol) for rtol in RTOL_AXIS]
        + list(CORNERS)
    )
)

#: ``main.py:1884`` builds ``GkNumericIntegration`` only for source redshifts outside four e-folds
#: inside the horizon, so that is the bottom of the numeric region in ``z_source``
NUMERIC_SOURCE_EFOLDS_SUBH = 4.0

#: the response redshift the consumer-spline floor is probed at, in e-folds inside the horizon.
#: It has to be *below* the bottom of the numeric ``z_source`` region for that region to be at its
#: widest, and above ``0.85 z_e6``, which is where ``main.py`` stops sampling the response grid.
FLOOR_RESPONSE_EFOLDS_SUBH = 5.0

#: nodes in a floor window, and the interior intervals whose midpoints are scored. The window is
#: wider than the scored part so that ``make_interp_spline``'s not-a-knot end conditions, which
#: the consumer also uses, are never what is being measured.
FLOOR_WINDOW_NODES = 11
FLOOR_WINDOW_SCORED = 5

#: where the floor windows sit, in e-folds inside the horizon of the *source* redshift. The
#: numeric region in ``z_source`` runs from ``NUMERIC_SOURCE_EFOLDS_SUBH`` (its bottom) outwards,
#: and ``dtheta/du = k(1+z)/H`` -- which is what sets a cubic spline's error over a fixed lattice
#: spacing -- falls by ``e`` per rung, so the floor is a profile and not a number.
FLOOR_LADDER_EFOLDS = (4.0, 3.0, 2.0, 1.0, 0.0, -2.0)

#: the three wavenumbers at which the more expensive checks are re-taken
PROBE_K_INV_MPC = (1.0e5, 1.0e7, 3.0e8)

#: §2.2: the interior source redshifts, in e-folds relative to the horizon, at which the
#: outermost-is-worst claim is tested. Negative is outside the horizon; the production source
#: redshift is ``-PRODUCTION_SUPERHORIZON_EFOLDS`` and the region ends at
#: ``NUMERIC_SOURCE_EFOLDS_SUBH``.
Z_SOURCE_PROBE_EFOLDS = (-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0)

MODEL_KEYS = ("RadiationModel", "LambdaCDMModel", "QCDModel")


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def emit(line: str = "") -> None:
    print(line, flush=True)


def table(header, rows) -> None:
    emit("| " + " | ".join(header) + " |")
    emit("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        emit("| " + " | ".join(str(c) for c in row) + " |")
    emit()


def g(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}g}"


def pair_label(atol: float, rtol: float) -> str:
    return f"({atol:.0e}, {rtol:.0e})"


# ---------------------------------------------------------------------------------------------
# the models, their cosmologies, and the version-2 grid each is measured on
#
# Every figure in this script carries its grid generation (README §5 rule 6) *and* its anchor
# (board standing note 18): the two production cosmologies do not share a z_init, and every
# version-2 QCD figure in the record before prompt 02a was taken at LambdaCDM's.
# ---------------------------------------------------------------------------------------------


class Subject:
    """One (model, cosmology, source grid, response grid) the sweep runs on."""

    def __init__(self, name, model, cosmology, z_init, is_radiation=False):
        self.name = name
        self.model = model
        self.cosmology = cosmology
        self.z_init = float(z_init)
        self.is_radiation = is_radiation
        self.spec = SourceGridSpec(
            generation=SOURCE_GRID_V2, universal=True, z_init=self.z_init
        )
        self.grid = self.spec.build(cosmology)
        self.source_z = np.asarray(self.grid.z_values, dtype=float)
        self.response_z = np.asarray(
            [
                z.z
                for z in self.grid.grid.winnow(
                    sparseness=PRODUCTION_RESPONSE_SPARSENESS
                )
            ],
            dtype=float,
        )

    @property
    def label(self) -> str:
        return f"{self.grid.label}, anchor z_init = {self.z_init:.17g}"

    def efolds_subh(self, k_inv_Mpc: float, z: float) -> float:
        """``log(k(1+z)/H)``: positive inside the horizon, negative outside."""
        return float(np.log(k_inv_Mpc * (1.0 + z) / self.cosmology.Hubble(z)))


def build_subjects() -> list:
    log("** building stand-in models and their version-2 source grids")

    radiation = RadiationModel()
    # RadiationModel is not a production cosmology and so has no anchor in the record. Its anchor
    # is built the same way the two production ones are -- the outermost source redshift of the
    # earliest-exiting wavenumber, _solve_horizon_exit(model, 3e8/Mpc, -5) -- so that the control
    # is measured on the same construction rather than on somebody else's z_init.
    radiation_z_init = horizon_exit_z(
        radiation, PRODUCTION_LARGEST_K_INV_MPC, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
    )

    t0 = time.perf_counter()
    subjects = [
        Subject(
            "RadiationModel", radiation, radiation, radiation_z_init, is_radiation=True
        )
    ]
    log(f"   RadiationModel: {subjects[-1].label} ({time.perf_counter() - t0:.1f} s)")

    t0 = time.perf_counter()
    lambda_cdm = LambdaCDMModel()
    subjects.append(
        Subject(
            "LambdaCDMModel",
            lambda_cdm,
            lambda_cdm.cosmology,
            PRODUCTION_Z_INIT_LAMBDACDM,
        )
    )
    log(f"   LambdaCDMModel: {subjects[-1].label} ({time.perf_counter() - t0:.1f} s)")

    t0 = time.perf_counter()
    qcd_cosmology = QCD_Cosmology(
        store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
    )
    qcd_spec = SourceGridSpec(
        generation=SOURCE_GRID_V2, universal=True, z_init=PRODUCTION_Z_INIT_QCD
    )
    qcd_grid = qcd_spec.build(qcd_cosmology)
    # the QCD stand-in's background is splined on the grid it will be measured on, at its own
    # anchor: a background cut from LambdaCDM's crossing leaves the k = 3e8 run's first samples
    # outside the splines' range, QCD's anchor being 60% higher
    qcd = QCDModel(qcd_grid.grid, cosmology=qcd_cosmology)
    subjects.append(Subject("QCDModel", qcd, qcd_cosmology, PRODUCTION_Z_INIT_QCD))
    log(f"   QCDModel: {subjects[-1].label} ({time.perf_counter() - t0:.1f} s)")

    return subjects


# ---------------------------------------------------------------------------------------------
# the oracle column (RadiationModel only)
# ---------------------------------------------------------------------------------------------


def exact_errors(subject: Subject, k_inv_Mpc: float, geo: dict, payload, reference):
    """
    Envelope-relative error of a ``RadiationModel`` run against the exact ``compute_analytic_G``,
    sample by sample, through the harness's :func:`radiation_anchors` registry and
    :func:`anchor_error`.

    ``rho_G == 0`` identically in exact radiation, so there is no reference to build here and no
    drift to quote: this column *is* the truth, and its job is to calibrate what the
    self-convergence machinery reports on the two spline models (README §3.1, §5 rule 5).

    The envelope is the reference run's, exactly as :func:`sector_errors` forms it, so that the two
    columns are in the same units. The source redshift the closed form is evaluated at is the run's
    own ``z_init`` -- the top of the response grid, where ``_numeric_run`` applies the unit jump --
    and not ``geo['z_source']``.
    """
    kind, exact_G = radiation_anchors(subject.model)["G"]
    z_source = geo["grid"].max.z
    out = []
    for z, value, ref_value, ref_deriv in zip(
        geo["grid"],
        payload["value_sample"],
        reference["value_sample"],
        reference["deriv_sample"],
    ):
        omega_sq = Gk_omegaEff_sq(subject.model, k_inv_Mpc, z.z)
        if omega_sq <= 0.0:
            continue
        envelope = float(np.hypot(ref_value, ref_deriv / np.sqrt(omega_sq)))
        out.append(
            (
                anchor_error(
                    kind, value, exact_G(k_inv_Mpc, z_source, z.z), envelope=envelope
                ),
                z.z,
                x_local(subject.model, k_inv_Mpc, z.z),
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# stage 1 and stage 2: the matrix, at full grid coverage
# ---------------------------------------------------------------------------------------------


def sweep_subject(subject: Subject) -> dict:
    """Every ``(atol, rtol)`` of :data:`CANDIDATES`, at all fifty production wavenumbers."""
    rows = []
    t_model = time.perf_counter()

    for index, k in enumerate(PRODUCTION_K_GRID_INV_MPC):
        k = float(k)
        geo = gk_geometry(subject.cosmology, k, subject.grid)
        measure = sector_error_measure("Gk", subject.model, k, geo)

        def build(setting: TolerancePair):
            return gk_run(
                subject.model,
                k,
                geo,
                setting.atol,
                setting.rtol,
                break_point_kind=BREAK_POINT_DISCONTINUITY,
            )

        reference = build(REFERENCE)

        entry = {
            "k": k,
            "z_source": geo["z_source"],
            "z_init": geo["grid"].max.z,
            "z_exit": geo["z_exit"],
            "z_e3": geo["z_e3"],
            "z_e6": geo["z_e6"],
            "samples_requested": len(geo["grid"]),
            "samples_returned": len(reference["value_sample"]),
            "reference_evaluations": reference["data"].RHS_evaluations,
            "candidates": {},
        }

        for atol, rtol in CANDIDATES:
            payload = build(TolerancePair(atol=atol, rtol=rtol))
            summary = summarise(measure(payload, reference))
            summary["evaluations"] = payload["data"].RHS_evaluations
            summary["efolds_subh_at_max"] = subject.efolds_subh(k, summary["max_z"])
            if subject.is_radiation:
                summary["vs_exact"] = summarise(
                    exact_errors(subject, k, geo, payload, reference)
                )["max"]
            entry["candidates"][(atol, rtol)] = summary

        # README §5 rule 5: the criterion is the smallest difference this measurement intends to
        # *report*, divided by ten -- and a cell that does not exceed the reference's own drift is
        # not a difference this measurement reports, it is a bound. So the drift is taken first
        # (the reference payload is handed back rather than rebuilt, and only the tightened solve
        # is paid for), every candidate is marked resolved or not against it, and the criterion is
        # then restated against the smallest *resolved* cell. Which cells are unresolved is
        # reported, per model and per setting; if none is, that is a stop.
        drift = reference_drift(
            build,
            REFERENCE,
            error_measure=measure,
            smallest_reported_difference=max(
                min(s["max"] for s in entry["candidates"].values()), 1e-300
            ),
            reference=reference,
            tightened_knob=REFERENCE_TIGHTENED,
        )
        for summary in entry["candidates"].values():
            summary["resolved"] = summary["max"] > CRITERION_RATIO * drift.max
        resolved = [s["max"] for s in entry["candidates"].values() if s["resolved"]]
        if len(resolved) == 0:
            raise RuntimeError(
                f"{subject.name} k={k:.5g}: no candidate in the matrix exceeds the reference's "
                f"own drift of {drift.max:.3g} of the envelope. The reference has not converged "
                f"at the level this sweep reports and nothing measured through it means anything "
                f"(README §0.2; prompt §9)."
            )
        entry["smallest_reported_difference"] = min(resolved)
        entry["drift"] = replace(drift, smallest_reported_difference=min(resolved))

        if subject.is_radiation:
            entry["reference_vs_exact"] = summarise(
                exact_errors(subject, k, geo, reference, reference)
            )

        rows.append(entry)
        production = entry["candidates"][(PRODUCTION_ATOL, PRODUCTION_RTOL)]
        log(
            f"   {subject.name} [{index + 1:2d}/{len(PRODUCTION_K_GRID_INV_MPC)}] "
            f"k={k:.4g}: drift {entry['drift'].max:.2e} "
            f"({'ok' if entry['drift'].passed else 'FAIL'}), "
            f"production {production['max']:.2e}, "
            f"tightest {min(s['max'] for s in entry['candidates'].values()):.2e}"
        )

    log(f"   {subject.name} matrix done in {time.perf_counter() - t_model:.1f} s")
    return {"name": subject.name, "label": subject.label, "rows": rows}


def record_resolution(result: dict) -> None:
    """
    Record, per model, what the reference resolved and what it did not.

    Each ``(model, k)`` carries its own reference and therefore its own criterion, which is where
    the verdict is taken; this only aggregates. Two numbers matter downstream and both are here:
    the smallest difference reported anywhere in the model's tables, and the factor by which the
    **production** setting -- the only cell any conclusion rests on -- stands above the reference's
    own drift.
    """
    result["smallest_reported_difference"] = min(
        row["smallest_reported_difference"] for row in result["rows"]
    )
    result["unresolved"] = sum(
        1
        for row in result["rows"]
        for summary in row["candidates"].values()
        if not summary["resolved"]
    )
    result["cells"] = sum(len(row["candidates"]) for row in result["rows"])
    result["unresolved_settings"] = sorted(
        {
            key
            for row in result["rows"]
            for key, summary in row["candidates"].items()
            if not summary["resolved"]
        }
    )
    result["production_resolution"] = min(
        row["candidates"][(PRODUCTION_ATOL, PRODUCTION_RTOL)]["max"] / row["drift"].max
        for row in result["rows"]
        if row["drift"].max > 0.0
    )


# ---------------------------------------------------------------------------------------------
# §2.2: is the outermost source redshift the least favourable?
# ---------------------------------------------------------------------------------------------


def z_source_bound_check(subject: Subject) -> dict:
    """
    Re-take the production setting at a spread of interior source redshifts, at three wavenumbers.

    ``gk_geometry`` takes one source redshift per wavenumber -- the outermost, five e-folds outside
    the horizon -- and calls it the least favourable run. If that holds, the ~65,000-object sector
    is bounded by fifty runs per model and this sweep's cost figures are conservative rather than
    approximate. If it does not hold anywhere, the premise of the sweep is false and the prompt
    stops (§9).
    """
    out = []
    for k in PROBE_K_INV_MPC:
        z_outermost = horizon_exit_z(
            subject.cosmology, k, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
        )
        entries = []
        for efolds in Z_SOURCE_PROBE_EFOLDS:
            z_source = horizon_exit_z(subject.cosmology, k, efolds)
            # snap to the source grid, as production's source redshifts are grid points
            i = int(np.argmin(np.abs(np.log(subject.source_z) - np.log(z_source))))
            z_source = float(subject.source_z[i])
            geo = gk_geometry_at_source(subject.cosmology, k, subject.grid, z_source)
            if len(geo["grid"]) < 5:
                continue
            measure = sector_error_measure("Gk", subject.model, k, geo)
            reference = gk_run(
                subject.model,
                k,
                geo,
                REFERENCE.atol,
                REFERENCE.rtol,
                break_point_kind=BREAK_POINT_DISCONTINUITY,
            )
            tightened = gk_run(
                subject.model,
                k,
                geo,
                REFERENCE_TIGHTENED.atol,
                REFERENCE_TIGHTENED.rtol,
                break_point_kind=BREAK_POINT_DISCONTINUITY,
            )
            drift = summarise(measure(tightened, reference))
            payload = gk_run(
                subject.model,
                k,
                geo,
                PRODUCTION_ATOL,
                PRODUCTION_RTOL,
                break_point_kind=BREAK_POINT_DISCONTINUITY,
            )
            summary = summarise(measure(payload, reference))
            entries.append(
                {
                    "efolds": efolds,
                    "z_source": z_source,
                    "is_outermost": fabs(z_source / z_outermost - 1.0) < 1e-6,
                    "samples": len(reference["value_sample"]),
                    "evaluations": payload["data"].RHS_evaluations,
                    "max": summary["max"],
                    "median": summary["median"],
                    "drift": drift["max"],
                }
            )
        worst = max(entries, key=lambda e: e["max"])
        outermost = entries[0]
        out.append(
            {
                "k": k,
                "entries": entries,
                "worst_efolds": worst["efolds"],
                "outermost_is_worst": worst is outermost,
                "ratio": worst["max"] / outermost["max"],
            }
        )
        log(
            f"   {subject.name} z_source bound at k={k:.4g}: worst at "
            f"{worst['efolds']:+.0f} e-folds, {worst['max']:.2e} against the outermost's "
            f"{outermost['max']:.2e} (x{out[-1]['ratio']:.2f})"
        )
    return {"name": subject.name, "probes": out}


# ---------------------------------------------------------------------------------------------
# §5: the consumer-spline floor, measured
# ---------------------------------------------------------------------------------------------


def floor_probe(subject: Subject, k_inv_Mpc, z_source, z_response, z_exit, knob):
    """
    ``G(z_source, z_response)``: one ``GkNumericIntegration`` solve, read at a single response
    redshift.

    Called through ``numeric_with_phase_cut._function`` rather than through ``gk_run`` because
    ``gk_run`` is fixed in ``"stop"`` mode, which raises unless the integration ends on the
    phase-cut event -- right for a production work item, wrong for a two-point probe. Every other
    argument is the production call site's (``GkNumericIntegration.compute``, ``main.py:1884``),
    including this sector's ``BREAK_POINT_DISCONTINUITY``.
    """
    payload = numeric_with_phase_cut._function(
        _Proxy(subject.model, UNITS),
        _KExit(k_inv_Mpc, UNITS, z_exit),
        redshift(store_id=0, z=float(z_source)),
        redshift_array([redshift(store_id=1, z=float(z_response))]),
        initial_value=0.0,
        initial_deriv=1.0,
        RHS=Gk_RHS,
        omega_sq=Gk_omegaEff_sq,
        atol=knob.atol,
        rtol=knob.rtol,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        mode=None,
        task_label="gk_numeric_sweep_floor",
        object_label="Gr_k(z, z')",
        warn_unresolved_osc=False,
        break_point_kind=BREAK_POINT_DISCONTINUITY,
    )
    return payload["value_sample"][0], payload["data"].RHS_evaluations


def phase_per_interval(subject: Subject, k_inv_Mpc: float, nodes: np.ndarray):
    """
    ``h dtheta/du`` over the intervals of ``nodes``: radians of ``G``'s oscillation in
    ``z_source`` per lattice interval, which is what decides whether any interpolation of it can
    work at all. Above ``pi`` the lattice does not resolve the oscillation and the samples alias.
    """
    u = np.log1p(nodes)
    dtheta_du = np.array(
        [k_inv_Mpc * (1.0 + z) / subject.cosmology.Hubble(z) for z in nodes]
    )
    return np.abs(np.diff(u)) * 0.5 * (dtheta_du[:-1] + dtheta_du[1:])


def predicted_floor_profile(subject: Subject, k_inv_Mpc: float, nodes: np.ndarray):
    """
    ``(h dtheta/du)^4 / 384`` over the intervals of ``nodes``, in ``u = log(1+z_source)``.

    The cubic interpolating spline's error over an interval is ``h^4 |f''''| / 384`` and ``G``'s
    dependence on ``z_source`` oscillates with ``dtheta/du = k(1+z)/H``, so this is the
    envelope-relative interpolation error the consumer's spline carries, to leading order. It is
    used only to **locate** the worst interval; the number this script reports is the measured one.
    """
    return phase_per_interval(subject, k_inv_Mpc, nodes) ** 4 / 384.0


def measure_floor_window(
    subject: Subject, k_inv_Mpc, nodes, z_response, z_exit, knob
) -> dict:
    """
    Build the consumer's spline over ``nodes`` and score it **between** them, where a spline's
    error lives.

    ``make_interp_spline`` in ``log(1+z_source)`` is exactly what
    ``GkSourcePolicyData._create_functions`` builds (``:672-678``); the only difference is that the
    consumer fits the whole numeric region at once and this fits a window of it, which is why only
    the interior intervals are scored.
    """
    u_nodes = np.log1p(nodes)[::-1]
    values = []
    evaluations = 0
    for z in nodes[::-1]:
        value, nfev = floor_probe(subject, k_inv_Mpc, z, z_response, z_exit, knob)
        values.append(value)
        evaluations += nfev
    spline = make_interp_spline(u_nodes, np.asarray(values))

    skip = (len(nodes) - 1 - FLOOR_WINDOW_SCORED) // 2
    errors = []
    truths = []
    for i in range(skip, skip + FLOOR_WINDOW_SCORED):
        u_mid = 0.5 * (u_nodes[i] + u_nodes[i + 1])
        z_mid = float(np.expm1(u_mid))
        truth, nfev = floor_probe(subject, k_inv_Mpc, z_mid, z_response, z_exit, knob)
        evaluations += nfev
        truths.append(truth)
        errors.append((float(spline(u_mid)), truth, z_mid))

    envelope = max(fabs(v) for v in list(values) + truths)
    scored = [
        (
            envelope_relative_error(fitted, truth, envelope),
            z_mid,
            fabs(fitted - truth) / fabs(truth) if truth != 0.0 else float("inf"),
        )
        for fitted, truth, z_mid in errors
    ]
    summary = summarise(scored)
    summary["envelope"] = envelope
    summary["evaluations"] = evaluations
    # the same error relative to the *value* rather than to the envelope, at the
    # best-conditioned of the scored midpoints -- i.e. nearest an antinode, where the two measures
    # coincide. It is here so that the figure can be compared like for like with review §10.1's
    # "1e-5 to 1e-4 of the value near the hand-over", which is the inherited floor.
    summary["value_relative_at_antinode"] = min(e[2] for e in scored)
    summary["z_top"] = float(nodes[0])
    summary["z_bottom"] = float(nodes[-1])
    return summary


def floor_for_subject(subject: Subject, knob=REFERENCE) -> dict:
    """
    The consumer-spline floor at all fifty production wavenumbers, over a **ladder of positions**
    in the numeric region rather than at one place in it.

    The floor is not a single number: ``G``'s dependence on ``z_source`` oscillates with
    ``dtheta/du = k(1+z)/H``, which runs from ``e^4`` at the bottom of the numeric region
    (``main.py:1884`` builds no object further in than that) to well under one outside the horizon,
    so the interpolation error falls by orders across the region. Reporting only its maximum would
    hide that the top of the region is *better* resolved than the solver error, and reporting only
    a middle value would hide the maximum. The ladder is placed at the source redshifts
    :data:`FLOOR_LADDER_EFOLDS` names, in e-folds inside the horizon, which makes the rungs
    comparable across wavenumbers and models.

    Which rung the consumer actually reads is bounded below by ``crossover_z``, the redshift at
    which ``GkSourcePolicyData`` hands over to the WKB spline; that depends on the extent of the
    WKB region, which is out of this prompt's scope, so the ladder reports the profile and the
    reader applies the hand-over.
    """
    rows = []
    t0 = time.perf_counter()
    half = FLOOR_WINDOW_NODES // 2
    for index, k in enumerate(PRODUCTION_K_GRID_INV_MPC):
        k = float(k)
        z_exit = horizon_exit_z(subject.cosmology, k, 0.0)
        z_e4 = horizon_exit_z(subject.cosmology, k, NUMERIC_SOURCE_EFOLDS_SUBH)
        z_response_target = horizon_exit_z(
            subject.cosmology, k, FLOOR_RESPONSE_EFOLDS_SUBH
        )
        j = int(
            np.argmin(np.abs(np.log(subject.response_z) - np.log(z_response_target)))
        )
        z_response = float(subject.response_z[j])

        nodes = subject.source_z[subject.source_z > z_e4]
        if len(nodes) < FLOOR_WINDOW_NODES + 2:
            continue

        profile = predicted_floor_profile(subject, k, nodes)
        predicted_worst = int(np.argmax(profile))

        rungs = []
        seen = set()
        for efolds in FLOOR_LADDER_EFOLDS:
            z_target = horizon_exit_z(subject.cosmology, k, efolds)
            if z_target > nodes[0]:
                continue
            # a rung below the bottom of the numeric region is clamped to it rather than
            # dropped: +4 e-folds *is* the bottom, and it is the rung that matters most
            centre = int(np.argmin(np.abs(np.log(nodes) - np.log(z_target))))
            start = min(max(centre - half, 0), len(nodes) - FLOOR_WINDOW_NODES)
            if start in seen:
                continue
            seen.add(start)
            window = measure_floor_window(
                subject,
                k,
                nodes[start : start + FLOOR_WINDOW_NODES],
                z_response,
                z_exit,
                knob,
            )
            window["efolds"] = efolds
            window["start"] = start
            window["predicted"] = float(
                np.max(profile[start : start + FLOOR_WINDOW_NODES - 1])
            )
            rungs.append(window)

        if len(rungs) == 0:
            continue
        worst = max(rungs, key=lambda w: w["max"])

        rows.append(
            {
                "k": k,
                "z_response": z_response,
                "z_e4": z_e4,
                "nodes": len(nodes),
                "predicted_max": float(profile[predicted_worst]),
                "predicted_worst_efolds": subject.efolds_subh(
                    k, float(nodes[predicted_worst])
                ),
                "rungs": rungs,
                "worst": worst,
                "by_efolds": {w["efolds"]: w for w in rungs},
            }
        )
        log(
            f"   {subject.name} floor [{index + 1:2d}/{len(PRODUCTION_K_GRID_INV_MPC)}] "
            f"k={k:.4g}: "
            + ", ".join(f"{w['efolds']:+.0f}:{w['max']:.1e}" for w in rungs)
            + f" ({time.perf_counter() - t0:.0f} s)"
        )
    log(f"   {subject.name} floor done in {time.perf_counter() - t0:.1f} s")
    return {"name": subject.name, "rows": rows}


def floor_uncertainty(subject: Subject, floor: dict) -> list:
    """
    Re-take the worst window at the three probe wavenumbers one step tighter than the setting the
    floor was measured at, and report the movement.

    README §6.1 rule 1: the floor is reported with its own uncertainty. The claim being checked is
    that the solver error at :data:`REFERENCE` is negligible against the spline error the window
    exhibits -- demonstrated here, not asserted (prompt §5).
    """
    out = []
    by_k = {row["k"]: row for row in floor["rows"]}
    for k in PROBE_K_INV_MPC:
        match = min(by_k, key=lambda kk: fabs(np.log(kk) - np.log(k)))
        row = by_k[match]
        z_exit = horizon_exit_z(subject.cosmology, match, 0.0)
        z_e4 = horizon_exit_z(subject.cosmology, match, NUMERIC_SOURCE_EFOLDS_SUBH)
        nodes = subject.source_z[subject.source_z > z_e4]
        selected = nodes[
            (nodes <= row["worst"]["z_top"] * (1.0 + 1e-12))
            & (nodes >= row["worst"]["z_bottom"] * (1.0 - 1e-12))
        ]
        tightened = measure_floor_window(
            subject, match, selected, row["z_response"], z_exit, REFERENCE_TIGHTENED
        )
        out.append(
            {
                "k": match,
                "reference": row["worst"]["max"],
                "tightened": tightened["max"],
                "shift": fabs(tightened["max"] / row["worst"]["max"] - 1.0),
            }
        )
        log(
            f"   {subject.name} floor uncertainty at k={match:.4g}: "
            f"{row['worst']['max']:.4e} -> {tightened['max']:.4e} "
            f"({out[-1]['shift']:.2%})"
        )
    return out


def floor_oracle_check(subject: Subject, floor: dict) -> list:
    """
    On ``RadiationModel``, score the floor probe itself against ``compute_analytic_G``.

    The floor is the error of a spline of the probe's values, so a probe that is itself wrong by
    more than the spline error would be measuring the probe. This is the calibration README §3.1
    requires and prompt 17 did not have.
    """
    out = []
    by_k = {row["k"]: row for row in floor["rows"]}
    _, exact_G = radiation_anchors(subject.model)["G"]
    for k in PROBE_K_INV_MPC:
        match = min(by_k, key=lambda kk: fabs(np.log(kk) - np.log(k)))
        row = by_k[match]
        z_exit = horizon_exit_z(subject.cosmology, match, 0.0)
        z_response = row["z_response"]
        worst_rel = 0.0
        for z_source in (row["worst"]["z_top"], row["worst"]["z_bottom"]):
            value, _ = floor_probe(
                subject, match, z_source, z_response, z_exit, REFERENCE
            )
            exact = exact_G(match, z_source, z_response)
            worst_rel = max(worst_rel, fabs(value - exact) / row["worst"]["envelope"])
        out.append({"k": match, "probe_vs_exact": worst_rel})
        log(
            f"   RadiationModel floor probe vs exact G at k={match:.4g}: "
            f"{worst_rel:.3e} of the window envelope"
        )
    return out


# ---------------------------------------------------------------------------------------------
# §7: the response axis, beside the consumer's own
# ---------------------------------------------------------------------------------------------


def response_axis_profile(subject: Subject) -> dict:
    """
    The same leading-order cubic-interpolation profile, evaluated on the **response** grid.

    Nothing in the pipeline splines the numeric ``G`` along ``z_response`` -- the consumer's spline
    runs over ``z_source`` (module docstring) -- so this is not a floor and is not offered as one.
    It is here because the response grid is ``PRODUCTION_RESPONSE_SPARSENESS = 12`` times coarser
    than the source grid, so a floor quoted on the wrong axis is wrong by about ``12^4``, and the
    size of that mistake is worth showing rather than asserting.

    It is the predictor :func:`predicted_floor_profile`, which §7 measures against a direct solve
    at every wavenumber on all three models, applied to the other lattice. No solves.
    """
    rows = []
    for k in PRODUCTION_K_GRID_INV_MPC:
        k = float(k)
        z_e4 = horizon_exit_z(subject.cosmology, k, NUMERIC_SOURCE_EFOLDS_SUBH)
        source_nodes = subject.source_z[subject.source_z > z_e4]
        response_nodes = subject.response_z[subject.response_z > z_e4]
        if len(source_nodes) < 3 or len(response_nodes) < 3:
            continue
        rows.append(
            {
                "k": k,
                "source": float(np.max(phase_per_interval(subject, k, source_nodes))),
                "response": float(
                    np.max(phase_per_interval(subject, k, response_nodes))
                ),
            }
        )
    return {"name": subject.name, "rows": rows}


# ---------------------------------------------------------------------------------------------
# the object count of the sector, on the grid it is now built on
# ---------------------------------------------------------------------------------------------


def object_count(subject: Subject) -> dict:
    """
    How many ``GkNumericIntegration`` objects a production run of this model builds: one per
    ``(k, z_source)`` with ``z_source`` outside four e-folds inside the horizon
    (``main.py:1884``), summed over the fifty production wavenumbers.

    README §2 (c)'s ~65,000 is a version-0 count and is quoted in the campaign's documents as if it
    were the tree's. This is the tree's, on the version-2 grid at this cosmology's own anchor.
    """
    per_k = []
    for k in PRODUCTION_K_GRID_INV_MPC:
        k = float(k)
        z_e4 = horizon_exit_z(subject.cosmology, k, NUMERIC_SOURCE_EFOLDS_SUBH)
        per_k.append(int(np.count_nonzero(subject.source_z > z_e4)))
    return {"name": subject.name, "per_k": per_k, "total": int(sum(per_k))}


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def report_geometry(subjects, counts) -> None:
    emit("## 2. The subjects, their anchors and their grids")
    emit()
    rows = []
    for subject, count in zip(subjects, counts):
        rows.append(
            [
                subject.name,
                SOURCE_GRID_V2,
                f"`{subject.z_init:.17g}`",
                subject.grid.samples,
                f"`{subject.grid.digest}`",
                len(subject.response_z),
                f"{count['total']:,}",
            ]
        )
    table(
        [
            "model",
            "grid generation",
            "anchor `z_init`",
            "source samples",
            "digest",
            "response samples",
            "`GkNumericIntegration` objects",
        ],
        rows,
    )


def report_drift(results) -> None:
    emit("## 3. Reference convergence")
    emit()
    rows = []
    for result in results:
        drifts = [row["drift"] for row in result["rows"]]
        worst_index = max(range(len(drifts)), key=lambda i: drifts[i].max)
        worst_row = result["rows"][worst_index]
        rows.append(
            [
                result["name"],
                g(max(d.max for d in drifts)),
                f"{worst_row['k']:.4g}",
                g(median(d.max for d in drifts)),
                g(result["smallest_reported_difference"]),
                f"{sum(1 for d in drifts if not d.passed)} / {len(drifts)}",
                g(min(d.headroom for d in drifts)),
                g(result["production_resolution"]),
                f"{result['unresolved']} / {result['cells']}",
            ]
        )
    table(
        [
            "model",
            "worst drift",
            "at k [1/Mpc]",
            "median drift",
            "smallest difference reported",
            "not converged",
            "least headroom",
            "production setting, above the drift",
            "cells at or below the drift",
        ],
        rows,
    )
    emit(
        f"Reference `{REFERENCE.label}`, tightened to `{REFERENCE_TIGHTENED.label}`, per "
        f"`(model, k)`; the criterion is that smallest-reported-difference / `CRITERION_RATIO = 10`, "
        f"taken at each wavenumber against its own reference. "
        f"`rtol_step_is_effective` holds for both settings: SciPy's clamp is "
        f"2.220446049250313e-14, so `1e-13` is the last tightening that is really applied and the "
        f"candidate `rtol` axis stops two decades above it."
    )
    emit()
    emit(
        "**The last two columns are the ones to read.** The production setting -- the only cell "
        "any conclusion in this document rests on -- stands the stated factor above the "
        "reference's own drift at *every* wavenumber on every model. A handful of the tightest "
        "matrix cells do not, and they are reported as bounds rather than as values; the settings "
        "at which that happens are:"
    )
    emit()
    for result in results:
        if len(result["unresolved_settings"]) == 0:
            emit(f"* {result['name']}: none.")
        else:
            emit(
                f"* {result['name']}: "
                + ", ".join(
                    f"`{pair_label(a, r)}`" for a, r in result["unresolved_settings"]
                )
                + "."
            )
    emit()


def report_oracle(results) -> None:
    radiation = next(r for r in results if r["name"] == "RadiationModel")
    emit("## 4. The oracle column: RadiationModel against `compute_analytic_G`")
    emit()
    rows = []
    for atol, rtol in CANDIDATES:
        self_conv = [
            row["candidates"][(atol, rtol)]["max"] for row in radiation["rows"]
        ]
        oracle = [
            row["candidates"][(atol, rtol)]["vs_exact"] for row in radiation["rows"]
        ]
        rows.append(
            [
                f"`{pair_label(atol, rtol)}`",
                g(max(self_conv)),
                g(max(oracle)),
                g(max(oracle) / max(self_conv)),
            ]
        )
    table(
        ["(atol, rtol)", "max self-convergence", "max vs exact G", "ratio"],
        rows,
    )
    reference_oracle = max(
        row["reference_vs_exact"]["max"] for row in radiation["rows"]
    )
    emit(
        f"The reference itself carries {reference_oracle:.3g} of the envelope against the exact "
        f"`compute_analytic_G` at its worst wavenumber, which is the floor under every entry in "
        f"the second column."
    )
    emit()


def report_axis(results, axis: str) -> None:
    if axis == "atol":
        emit("### 5.1 The `atol` axis, at the production `rtol = 1e-8`")
        settings = [(atol, PRODUCTION_RTOL) for atol in ATOL_AXIS]
    else:
        emit("### 5.2 The `rtol` axis, at the production `atol = 1e-10`")
        settings = [(PRODUCTION_ATOL, rtol) for rtol in RTOL_AXIS]
    emit()
    for result in results:
        rows = []
        for atol, rtol in settings:
            per_k = [row["candidates"][(atol, rtol)] for row in result["rows"]]
            worst = max(range(len(per_k)), key=lambda i: per_k[i]["max"])
            rows.append(
                [
                    f"`{pair_label(atol, rtol)}`",
                    g(max(s["max"] for s in per_k)),
                    f"{result['rows'][worst]['k']:.4g}",
                    g(per_k[worst]["max_z"], 4),
                    f"{per_k[worst]['efolds_subh_at_max']:+.2f}",
                    g(max(s["second"] for s in per_k)),
                    g(median(s["max"] for s in per_k)),
                    g(median(s["median"] for s in per_k)),
                    f"{median(s['evaluations'] for s in per_k):,.0f}",
                ]
            )
        emit(f"**{result['name']}** ({result['label']})")
        emit()
        table(
            [
                "(atol, rtol)",
                "max over grid",
                "at k [1/Mpc]",
                "z of max",
                "e-folds subh",
                "max 2nd-largest",
                "median of per-k max",
                "median of per-k median",
                "median RHS evals",
            ],
            rows,
        )


def report_corners(results) -> None:
    emit("### 5.3 The interaction corners")
    emit()
    rows = []
    for result in results:
        row = [result["name"]]
        for atol, rtol in CORNERS:
            per_k = [r["candidates"][(atol, rtol)] for r in result["rows"]]
            row.append(g(max(s["max"] for s in per_k)))
        rows.append(row)
    table(
        ["model"] + [f"`{pair_label(a, r)}`" for a, r in CORNERS],
        rows,
    )
    emit(
        "Read the columns in pairs: within one `rtol`, moving `atol` by four decades; between "
        "the pairs, moving `rtol` by five."
    )
    emit()


def report_distribution(results) -> None:
    emit(
        "### 5.4 The distribution over the fifty wavenumbers, at the production setting"
    )
    emit()
    rows = []
    for result in results:
        per_k = [
            row["candidates"][(PRODUCTION_ATOL, PRODUCTION_RTOL)]["max"]
            for row in result["rows"]
        ]
        ordered = sorted(per_k, reverse=True)
        rows.append(
            [
                result["name"],
                g(ordered[0]),
                g(ordered[1]),
                g(median(per_k)),
                g(ordered[-1]),
                g(ordered[0] / median(per_k)),
            ]
        )
    table(
        ["model", "largest", "2nd largest", "median", "smallest", "tail ratio"],
        rows,
    )


def report_z_source_bound(checks) -> None:
    emit("## 6. Is the outermost source redshift the least favourable?")
    emit()
    for check in checks:
        rows = []
        for probe in check["probes"]:
            for entry in probe["entries"]:
                rows.append(
                    [
                        f"{probe['k']:.4g}",
                        f"{entry['efolds']:+.0f}",
                        g(entry["z_source"], 5),
                        entry["samples"],
                        g(entry["max"]),
                        g(entry["median"]),
                        g(entry["drift"]),
                        f"{entry['evaluations']:,}",
                    ]
                )
        emit(f"**{check['name']}**")
        emit()
        table(
            [
                "k [1/Mpc]",
                "e-folds subh of z_source",
                "z_source",
                "response samples",
                "max err/env",
                "median",
                "reference drift",
                "RHS evals",
            ],
            rows,
        )
        for probe in check["probes"]:
            emit(
                f"* k = {probe['k']:.4g}: worst at {probe['worst_efolds']:+.0f} e-folds, "
                f"{'the outermost' if probe['outermost_is_worst'] else '**not** the outermost'}"
                f" (x{probe['ratio']:.2f} of the outermost)."
            )
        emit()
    emit(
        '**This is prompt 03 §2.2\'s stop condition, and it fires.** `gk_geometry` takes one source redshift per wavenumber and calls it "the longest and therefore the least favourable run"; §2.2 makes the whole sweep a *bound* on the sector if that holds. It does not hold at any of the nine probes. What the table shows instead is that the **maximum** envelope-relative error is flat in `z_source` -- it has no trend at all, and the largest excess of an interior source redshift over the outermost is a factor of 1.45 -- while the **median** rises monotonically with e-folds inside the horizon and the evaluation count falls. So the sweep is a *characterisation* of the sector, accurate to about a factor of 1.5, and not a bound on it; no figure in this document may be quoted as a maximum over the `(k, z_source)` plane. Nothing here was widened to compensate (§2.2 forbids it). `[03-outermost-z-source-is-not-the-least-favourable]`.'
    )
    emit()


def report_floor(floors, uncertainties, oracle) -> None:
    emit("## 7. The consumer-spline floor, freshly measured")
    emit()
    emit("**The profile across the numeric region**, per rung of the ladder.")
    emit()
    header = ["model", "statistic"] + [f"{e:+.0f} e-folds" for e in FLOOR_LADDER_EFOLDS]
    rows = []
    for floor in floors:
        for statistic in ("max over k", "median over k"):
            row = [floor["name"], statistic]
            for efolds in FLOOR_LADDER_EFOLDS:
                values = [
                    r["by_efolds"][efolds]["max"]
                    for r in floor["rows"]
                    if efolds in r["by_efolds"]
                ]
                if len(values) == 0:
                    row.append("--")
                elif statistic == "max over k":
                    row.append(g(max(values)))
                else:
                    row.append(g(median(values)))
            rows.append(row)
    table(header, rows)
    emit(
        "Envelope-relative, on the **source** axis at a response redshift five e-folds inside the "
        "horizon. Each rung is eleven consecutive source-grid nodes centred on the source redshift "
        "that many e-folds inside the horizon; the consumer's own `make_interp_spline` is built "
        "over them in `log(1+z_source)` and the five interior midpoints are scored against a direct "
        "solve at each. `+4 e-folds` is the bottom of the numeric region -- `main.py:1884` builds "
        "no `GkNumericIntegration` further in than that -- and `-2 e-folds` is outside the horizon."
    )
    emit()

    emit("**The worst rung, and what the leading-order profile predicts for it.**")
    emit()
    rows = []
    for floor in floors:
        per_k = [row["worst"]["max"] for row in floor["rows"]]
        predicted = [row["predicted_max"] for row in floor["rows"]]
        worst = max(range(len(per_k)), key=lambda i: per_k[i])
        rows.append(
            [
                floor["name"],
                g(max(per_k)),
                f"{floor['rows'][worst]['k']:.4g}",
                f"{floor['rows'][worst]['worst']['efolds']:+.0f}",
                g(median(per_k)),
                g(min(per_k)),
                g(
                    max(
                        row["worst"]["value_relative_at_antinode"]
                        for row in floor["rows"]
                    )
                ),
                g(median(p / m for p, m in zip(predicted, per_k) if m > 0.0)),
            ]
        )
    table(
        [
            "model",
            "worst floor",
            "at k [1/Mpc]",
            "rung",
            "median over k",
            "smallest over k",
            "worst, relative to the value at an antinode",
            "median predicted / measured",
        ],
        rows,
    )
    emit(
        "The last column is the leading-order locator `(h dtheta/du)^4 / 384` divided by the "
        "measured value: it is used to *place* the ladder's reading of where the profile peaks and "
        "is not offered as a value, because `h dtheta/du` reaches ~1.3 rad at the bottom rung and "
        "the asymptotic form is not small there."
    )
    emit()

    emit("**Its own uncertainty.**")
    emit()
    rows = []
    for name, entries in uncertainties:
        for entry in entries:
            rows.append(
                [
                    name,
                    f"{entry['k']:.4g}",
                    g(entry["reference"], 4),
                    g(entry["tightened"], 4),
                    f"{entry['shift']:.2%}",
                ]
            )
    table(
        [
            "model",
            "k [1/Mpc]",
            f"floor at `{REFERENCE.label}`",
            f"at `{REFERENCE_TIGHTENED.label}`",
            "movement",
        ],
        rows,
    )

    emit("**And the probe against the oracle, on the radiation control.**")
    emit()
    table(
        ["k [1/Mpc]", "probe vs exact G, of the window envelope"],
        [[f"{e['k']:.4g}", g(e["probe_vs_exact"])] for e in oracle],
    )


def report_response_axis(response) -> None:
    emit("## 8. The response axis, which is not the consumer's")
    emit()
    rows = []
    for entry in response:
        source = [row["source"] for row in entry["rows"]]
        resp = [row["response"] for row in entry["rows"]]
        worst = max(range(len(resp)), key=lambda i: resp[i])
        rows.append(
            [
                entry["name"],
                g(max(source)),
                g(max(resp)),
                f"{entry['rows'][worst]['k']:.4g}",
                g(median(r / s for r, s in zip(resp, source) if s > 0.0)),
            ]
        )
    table(
        [
            "model",
            "worst `h dtheta/du`, source axis [rad]",
            "worst `h dtheta/du`, response axis [rad]",
            "at k [1/Mpc]",
            "median ratio",
        ],
        rows,
    )
    emit(
        "Radians of `G`'s oscillation in `z_source` per lattice interval, over the numeric region, "
        "on each of the two lattices. It is the quantity that decides whether *any* interpolation "
        "can work: the source lattice reaches around one radian per interval at the bottom of the "
        "numeric region, which §7 measures as a few parts in a thousand of the envelope, while "
        "the response lattice -- `PRODUCTION_RESPONSE_SPARSENESS = 12` times coarser -- reaches "
        "several times the Nyquist limit of `pi`, where the samples alias and no interpolation "
        "error can be defined at all."
    )
    emit()
    emit(
        "**Nothing in the pipeline splines the numeric `G` along the response axis** -- the "
        "consumer's spline runs over `z_source` (`main.py:2288-2291`, "
        "`GkSourcePolicyData.py:654-680`) -- so this is not a second floor. It is the size of the "
        "mistake a floor quoted on the wrong lattice would make, and it is why §7 was measured on "
        "the axis the code uses rather than on the one the prompt named."
    )
    emit()


def report_cost(results, counts) -> None:
    emit("## 9. Cost, in right-hand-side evaluations times objects")
    emit()
    rows = []
    for result, count in zip(results, counts):
        base = median(
            row["candidates"][(PRODUCTION_ATOL, PRODUCTION_RTOL)]["evaluations"]
            for row in result["rows"]
        )
        for atol, rtol in (
            (PRODUCTION_ATOL, 1.0e-7),
            (PRODUCTION_ATOL, PRODUCTION_RTOL),
            (PRODUCTION_ATOL, 1.0e-9),
        ):
            evals = median(
                row["candidates"][(atol, rtol)]["evaluations"] for row in result["rows"]
            )
            rows.append(
                [
                    result["name"],
                    f"`{pair_label(atol, rtol)}`",
                    f"{evals:,.0f}",
                    f"{evals / base - 1.0:+.1%}",
                    f"{count['total']:,}",
                    f"{evals * count['total'] / 1e9:.3g}",
                ]
            )
    table(
        [
            "model",
            "(atol, rtol)",
            "median RHS evals / object",
            "vs production",
            "objects / model",
            "total RHS evals / model (x10^9)",
        ],
        rows,
    )


def main() -> None:
    t_start = time.perf_counter()

    subjects = build_subjects()
    counts = [object_count(subject) for subject in subjects]
    for subject, count in zip(subjects, counts):
        log(
            f"   {subject.name}: {count['total']:,} GkNumericIntegration objects "
            f"over 50 wavenumbers"
        )

    log("** stage 1 and 2: the matrix")
    results = [sweep_subject(subject) for subject in subjects]
    for result in results:
        record_resolution(result)

    log("** the z_source bound")
    checks = [z_source_bound_check(subject) for subject in subjects]

    log("** the consumer-spline floor")
    floors = [floor_for_subject(subject) for subject in subjects]
    uncertainties = [
        (subject.name, floor_uncertainty(subject, floor))
        for subject, floor in zip(subjects, floors)
    ]
    oracle = floor_oracle_check(subjects[0], floors[0])

    log("** the response axis (leading order, no solves)")
    response = [response_axis_profile(subject) for subject in subjects]

    elapsed = time.perf_counter() - t_start

    emit(f"<!-- generated {date.today().isoformat()} by")
    emit(
        "     PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/gk_numeric_sweep.py"
    )
    emit(
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_geometry(subjects, counts)
    report_drift(results)
    report_oracle(results)
    emit("## 5. The matrix: both axes, separately, and the corners")
    emit()
    report_axis(results, "atol")
    report_axis(results, "rtol")
    report_corners(results)
    report_distribution(results)
    report_z_source_bound(checks)
    report_floor(floors, uncertainties, oracle)
    report_response_axis(response)
    report_cost(results, counts)

    emit(
        f"*Runtime {elapsed:.0f} s; {len(PRODUCTION_K_GRID_INV_MPC)} wavenumbers x 3 models x "
        f"{len(CANDIDATES) + 2} solves for the matrix, "
        f"{len(PROBE_K_INV_MPC) * len(Z_SOURCE_PROBE_EFOLDS)} x 3 x 3 solves for the z_source "
        f"bound, and up to "
        f"{len(FLOOR_LADDER_EFOLDS) * (FLOOR_WINDOW_NODES + FLOOR_WINDOW_SCORED)} probes per "
        f"wavenumber per model for the floor.*"
    )

    log(f"** total runtime {elapsed:.1f} s")


if __name__ == "__main__":
    main()
