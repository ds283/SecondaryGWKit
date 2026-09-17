"""
What do ``TkNumericIntegration``'s and ``wavenumber_exit_time``'s tolerances buy?
(``prompts/tolerance-convergence/`` prompt 03a, board items **T5** and **T6**.)

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/tk_numeric_exit_sweep.py

It emits every table of ``docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md`` on stdout and a
progress log on stderr. **It changes nothing**: no production module is touched, no constant
moves, and the recommendations the document carries are for the user to accept or reject
(README §7 **D1**).

**Two targets, and they fail in different ways.**

``TkNumericIntegration`` is 50 objects per model -- one per wavenumber -- and it *has* been swept,
by ``GkTk-remedial`` prompt 17. That sweep ran on the **version-0** source grid and at
``numeric_with_phase_cut``'s module default, ``BREAK_POINT_DISCONTINUITY``. The production call
site has passed ``BREAK_POINT_ALL`` since ``GkTk-remedial`` prompt 19
(``TkNumericIntegration.BREAK_POINT_KIND``), and production has built the **version-2** grid since
``qcd-background-audit`` prompt 15. So the configuration in the record is not the configuration
production runs, and §3 here separates the two changes by running the 2x2:
``{v0 per-k, v2 universal} x {discontinuity, all}`` at the production setting.

``wavenumber_exit_time`` has never been measured at all. It is not a value but a **location**: its
root fixes where the source grid begins and where every horizon-relative cut sits. Its error
measure is therefore chosen here rather than inherited, and §7 reports all three of the things
prompt 03a §2.4 asks for -- the residual the code itself checks, the displacement from a converged
re-solve, and what that displacement does to a grid.

**What is measured**, for ``RadiationModel``, ``LambdaCDMModel`` and ``QCDModel`` at all fifty
production wavenumbers, on the **version-2** source grid at each cosmology's **own** production
anchor:

1. a ``Tk`` **reference** per ``(model, k)`` at ``(atol, rtol) = (1e-18, 1e-12)`` -- prompt 17's
   pair, kept so the two campaigns' figures stay comparable -- with its drift against
   ``(1e-19, 1e-13)`` scored by ``convergence_reference.reference_drift``, which will not return a
   drift without the verdict attached (README §5 rule 5);
2. the **``rtol`` axis** at ``atol = DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13``, six decades
   through the production ``rtol = 1e-8``, at full grid coverage, under **``BREAK_POINT_ALL``**;
3. **off-axis points** -- one decade of ``atol`` either side of ``1e-13`` at two ``rtol``
   settings, also at full grid coverage -- which say whether the axes interact in this sector.
   The ``atol`` **value** is settled (§7 D1, the user 2026-09-12) and is not reopened; whether it
   *binds* is a different question and the ``rtol`` recommendation is not separable without it;
4. the **initial-condition floor**, re-measured on this tree both ways prompt 17 measured it: the
   production ``T = 1, T' = 0`` against the super-horizon series ``T = 1 - x^2/10`` on all three
   models, and against the exact ``T`` on the radiation control;
5. the **2x2** of grid generation against break-point policy at the production setting, which is
   what turns "prompt 17's figures were taken under other conditions" into a number;
6. ``_solve_horizon_exit`` over a matrix in ``xtol`` and ``rtol``, at all fifty wavenumbers, three
   models and the three offsets a grid is built from -- crossing itself, ``z_exit_suph_e5`` and
   ``z_exit_subh_e4`` -- **never through the datastore**
   (``[02-wavenumber-exit-time-tolerance-is-an-inequality-key]``);
7. what an anchor displacement does to the version-2 grid it builds: sample count, content digest,
   and the largest relative row displacement, against
   ``DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7``, which is the tolerance at which the datastore
   matches an existing redshift row.

**The radiation column is an oracle, not a self-convergence.** README §3.1's anchor table gives a
closed form for both of this prompt's quantities on ``RadiationModel``: ``compute_analytic_T`` for
the transfer function and ``exact_z_exit`` -- ``1 + z = k/(H0 e^N)`` -- for the horizon crossing.
Both are taken through the facility's :func:`radiation_anchors` registry and :func:`anchor_error`,
and they calibrate what the self-convergence machinery reports on the two spline models.

**Everything goes through ``ComputeTargets/tests/convergence_reference.py``** (board standing note
14), including the exit-time sweep, whose drift is :func:`reference_drift` with a root solve in
place of an ODE solve. **No line of that module is changed by this prompt**: ``tk_run`` already
takes ``break_point_kind`` and ``exact_z_exit`` already exists, which is what prompt 03a §2.5
predicted.

No Ray and no datastore (``CLAUDE.md``).
"""

import platform
import sys
import time
from dataclasses import replace
from datetime import date
from math import fabs, log, log1p, sqrt
from statistics import median

import numpy as np
import scipy

from ComputeTargets.BackgroundModel import BREAK_POINT_ALL, BREAK_POINT_DISCONTINUITY
from ComputeTargets.tests.convergence_reference import (
    CRITERION_RATIO,
    SourceGridSpec,
    TolerancePair,
    UNITS,
    V0_PER_K_GRID,
    anchor_error,
    grid_cosmology,
    radiation_anchors,
    reference_drift,
    sector_error_measure,
    summarise,
    tk_geometry,
    tk_run,
    x_local,
)
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    PRODUCTION_K_GRID_INV_MPC,
    PRODUCTION_LARGEST_K_INV_MPC,
    PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
    PRODUCTION_SUPERHORIZON_EFOLDS,
    PRODUCTION_Z_END,
    PRODUCTION_Z_INIT_LAMBDACDM,
    PRODUCTION_Z_INIT_QCD,
    QCDModel,
    RadiationModel,
    SOURCE_GRID_V2,
    horizon_exit_z,
    source_grid,
)
from CosmologyConcepts import redshift_grid_digest, wavenumber
from CosmologyConcepts.wavenumber import (
    DEFAULT_HEXIT_TOLERANCE,
    _solve_horizon_exit,
)
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import Planck2018
from config.defaults import DEFAULT_REDSHIFT_RELATIVE_PRECISION

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

#: the production setting for this sector: ``config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE``
#: and ``DEFAULT_REL_TOLERANCE``, read here and not changed (README §5 rule 8)
PRODUCTION_ATOL = 1.0e-13
PRODUCTION_RTOL = 1.0e-8

#: the shared pair, for the exit-time target: ``DEFAULT_ABS_TOLERANCE`` reaches
#: ``root_scalar`` as ``xtol`` and ``DEFAULT_REL_TOLERANCE`` as ``rtol``
PRODUCTION_XTOL_HEXIT = 1.0e-10
PRODUCTION_RTOL_HEXIT = 1.0e-8

#: the ``Tk`` reference and the tightening that scores it: ``GkTk-remedial`` prompt 17's pair,
#: unchanged so that the two campaigns' reference-convergence figures stay comparable. SciPy
#: clamps ``solve_ivp``'s ``rtol`` at ``SCIPY_RTOL_FLOOR = 2.22e-14``, so ``1e-13`` is the last
#: tightening that is really applied.
REFERENCE = TolerancePair(atol=1.0e-18, rtol=1.0e-12)
REFERENCE_TIGHTENED = TolerancePair(atol=1.0e-19, rtol=1.0e-13)

#: the measurement this prompt exists to take: the ``rtol`` axis at the settled ``atol``, three
#: decades below the production setting and two above it, at full grid coverage
RTOL_AXIS = (
    1.0e-6,
    1.0e-7,
    1.0e-8,
    3.0e-9,
    1.0e-9,
    3.0e-10,
    1.0e-10,
    3.0e-11,
    1.0e-11,
)

#: the off-axis points of §4.1: one decade of ``atol`` either side of the settled ``1e-13``, at
#: the production ``rtol`` and at one decade tighter. Not a second axis, and no ``atol`` is
#: recommended from them -- they answer only whether the two axes interact here.
OFF_AXIS = tuple(
    (atol, rtol) for atol in (1.0e-12, 1.0e-14) for rtol in (1.0e-8, 1.0e-10)
)

#: every ``(atol, rtol)`` the sweep visits, production first so the progress log leads with it
CANDIDATES = tuple(
    dict.fromkeys(
        [(PRODUCTION_ATOL, PRODUCTION_RTOL)]
        + [(PRODUCTION_ATOL, rtol) for rtol in RTOL_AXIS]
        + list(OFF_AXIS)
    )
)

#: ``GkTk-remedial`` README §6's row for this sector, and the statistic
#: ``[12-tk-numeric-atol-largest-k-excursion]`` is written in: the count of wavenumbers whose
#: maximum envelope-relative error exceeds it
TARGET_ERROR = 3.0e-6

#: README §2 (f)'s inherited floor for this sector, from ``GkTk-remedial`` prompt 17 §8, taken on
#: the **version-0** grid under ``BREAK_POINT_DISCONTINUITY``. §5 re-measures it here.
INHERITED_IC_FLOOR = 2.52e-6

#: prompt 17's published headline, for §3's comparison: the count of the fifty wavenumbers above
#: :data:`TARGET_ERROR` on Radiation / LambdaCDM / QCD, and the worst of them, both at the
#: production setting on the version-0 grid under ``BREAK_POINT_DISCONTINUITY``
PROMPT_17_ABOVE_TARGET = {
    "RadiationModel": 3,
    "LambdaCDMModel": 13,
    "QCDModel": 8,
}
PROMPT_17_WORST = 8.64e-4

#: the three anchors, and what the version-2 grid is at each of them. Prompt 03 recorded the
#: radiation control's (its log, deviation 6); prompt 02a recorded the two production ones (board
#: standing note 18). A mismatch means the grid construction has moved under this campaign's
#: figures and is a stop, not a thing to re-baseline.
EXPECTED_GRIDS = {
    "RadiationModel": (2306, "3bef2c06"),
    "LambdaCDMModel": (1778, "60a3205a"),
    "QCDModel": (2034, "21ffc126"),
}
EXPECTED_RADIATION_ANCHOR = 44523947729.772957

# ---------------------------------------------------------------------------------------------
# the exit-time matrix
# ---------------------------------------------------------------------------------------------

#: Brent's own relative floor, ``4 * eps``. ``scipy.optimize.brentq`` raises for anything smaller,
#: so this -- and not ``solve_ivp``'s ``100 eps`` clamp -- is where the ``rtol`` axis of a root
#: solve stops. ``TolerancePair.rtol_step_is_effective`` reports the ``solve_ivp`` clamp, which is
#: the wrong floor for this target; the mismatch is recorded in the log rather than worked around.
BRENT_RTOL_FLOOR = 4.0 * np.finfo(float).eps

#: the ``xtol`` axis, in ``u = log(1+z)``. It spans the production ``1e-10`` by four decades
#: either side so that the settings where the ``xtol`` term of Brent's ``xtol + rtol*|u|``
#: actually binds are *in* the matrix -- establishing that it never binds over the production
#: range is one of this prompt's answers, and an inference is not a measurement.
HEXIT_XTOL_AXIS = (1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12, 1.0e-14, 1.0e-16, 1.0e-300)

#: the ``rtol`` axis, stopping one decade above Brent's floor
HEXIT_RTOL_AXIS = (
    1.0e-6,
    1.0e-7,
    1.0e-8,
    1.0e-9,
    1.0e-10,
    1.0e-11,
    1.0e-12,
    1.0e-13,
    1.0e-14,
)

#: the reference re-solve and its tightening. ``xtol = 1e-300, rtol = 1e-14`` is the pair
#: ``_solve_T_z`` already uses and the one ``[02a-grid-digest-not-reproducible]`` measured the
#: shipped anchors against; one step tighter is ``1e-15``, which is still above Brent's floor.
HEXIT_REFERENCE = TolerancePair(atol=1.0e-300, rtol=1.0e-14)
HEXIT_REFERENCE_TIGHTENED = TolerancePair(atol=1.0e-300, rtol=1.0e-15)

#: the offsets a grid is built from (prompt 03a §2.3): horizon crossing itself, the outermost
#: super-horizon offset the source grid anchors on (``z_exit_suph_e5``), and the sub-horizon
#: offset at which the ``Gk`` numeric region stops (``z_exit_subh_e4``)
HEXIT_OFFSETS = (0, -5, 4)

#: the relative perturbations of the anchor at which §8 rebuilds the version-2 grid. They bracket
#: the production solve's own convergence bound, ``xtol + rtol*|u| ~ 3.8e-7`` relative, and
#: ``DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7``, the tolerance at which the datastore matches an
#: existing redshift row.
ANCHOR_PERTURBATIONS = (
    1.0e-16,
    1.0e-14,
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0e-7,
    3.8e-7,
    1.0e-6,
)

MODEL_KEYS = ("RadiationModel", "LambdaCDMModel", "QCDModel")


def log_line(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def emit(line: str = "") -> None:
    print(line, flush=True)


def table(header, rows) -> None:
    emit("| " + " | ".join(str(c) for c in header) + " |")
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


def xtol_label(xtol: float) -> str:
    return "0" if xtol <= 1e-300 else f"{xtol:.0e}"


# ---------------------------------------------------------------------------------------------
# the exact radiation transfer function, for the oracle column
#
# ``compute_analytic_T`` is reached through the facility's radiation_anchors registry; the two
# helpers below are the *denominator* and the phase variable of prompt 17's oracle measure, kept
# identical so that §5's re-measured floor is comparable with the 2.52e-6 in the record.
# ---------------------------------------------------------------------------------------------


def T_envelope_exact(x: float) -> float:
    """The Liouville-Green envelope of ``T = 3(sin x - x cos x)/x^3``: ``3 sqrt(1+x^2)/x^3``."""
    return 3.0 * sqrt(1.0 + x * x) / (x * x * x)


def series_initial_data(model, k_inv_Mpc: float, z: float):
    """
    Super-horizon series initial data, ``T = 1 - x^2/10``, ``dT/dz = (x/5) k c_s/H``.

    It is the leading correction to the production ``T = 1, T' = 0`` in any radiation-dominated
    background, which is where every wavenumber's grid starts. Using it to *measure* the
    initial-condition floor commits nothing to changing the initial data, which is
    ``[00-tk-superhorizon-ic-series]`` and out of scope (README §0.5).
    """
    x = x_local(model, k_inv_Mpc, z)
    dx_dz = (
        -k_inv_Mpc * sqrt(model.functions.wPerturbations(z)) / model.functions.Hubble(z)
    )
    return 1.0 - x * x / 10.0, -(x / 5.0) * dx_dz


# ---------------------------------------------------------------------------------------------
# the subjects: model, cosmology, anchor, version-2 grid
#
# Every figure carries its grid generation (README §5 rule 6) *and* its anchor (board standing
# note 18): the two production cosmologies do not share a z_init.
# ---------------------------------------------------------------------------------------------


class Subject:
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
        self.v0_grid = V0_PER_K_GRID.build(cosmology)

    @property
    def label(self) -> str:
        return f"{self.grid.label}, anchor z_init = {self.z_init!r}"

    def efolds_subh(self, k_inv_Mpc: float, z: float) -> float:
        return float(np.log(k_inv_Mpc * (1.0 + z) / self.cosmology.Hubble(z)))

    def grid_for(self, generation: str):
        return self.grid if generation == "v2" else self.v0_grid


def build_subjects() -> list:
    log_line("** building stand-in models and their version-2 source grids")

    radiation = RadiationModel()
    # RadiationModel is not a production cosmology and has no anchor in the record. Prompt 03
    # built one the same way the two production anchors are built -- the outermost source
    # redshift of the earliest-exiting wavenumber, _solve_horizon_exit(model, 3e8/Mpc, -5) -- and
    # published its grid; this prompt uses that one so the two documents are comparable.
    radiation_z_init = horizon_exit_z(
        radiation, PRODUCTION_LARGEST_K_INV_MPC, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
    )
    if fabs(radiation_z_init / EXPECTED_RADIATION_ANCHOR - 1.0) > 1e-13:
        raise RuntimeError(
            f"the radiation control's anchor is {radiation_z_init:.17g}, not prompt 03's "
            f"{EXPECTED_RADIATION_ANCHOR:.17g}. Something under the anchor solve has moved and "
            "this prompt's figures would not be comparable with prompt 03's (prompt 03a §2.1)."
        )

    t0 = time.perf_counter()
    subjects = [
        Subject(
            "RadiationModel", radiation, radiation, radiation_z_init, is_radiation=True
        )
    ]
    log_line(
        f"   RadiationModel: {subjects[-1].label} ({time.perf_counter() - t0:.1f} s)"
    )

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
    log_line(
        f"   LambdaCDMModel: {subjects[-1].label} ({time.perf_counter() - t0:.1f} s)"
    )

    t0 = time.perf_counter()
    qcd_cosmology = QCD_Cosmology(
        store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
    )
    qcd_spec = SourceGridSpec(
        generation=SOURCE_GRID_V2, universal=True, z_init=PRODUCTION_Z_INIT_QCD
    )
    qcd_grid = qcd_spec.build(qcd_cosmology)
    # the QCD stand-in's background is splined on the grid it will be measured on, at its own
    # anchor: a background cut from LambdaCDM's anchor leaves the k = 3e8 run's first samples
    # outside the splines' range, QCD's anchor being 60% higher (prompt 03)
    qcd = QCDModel(qcd_grid.grid, cosmology=qcd_cosmology)
    subjects.append(Subject("QCDModel", qcd, qcd_cosmology, PRODUCTION_Z_INIT_QCD))
    log_line(f"   QCDModel: {subjects[-1].label} ({time.perf_counter() - t0:.1f} s)")

    for subject in subjects:
        samples, digest = EXPECTED_GRIDS[subject.name]
        if subject.grid.samples != samples or subject.grid.digest != digest:
            raise RuntimeError(
                f"{subject.name}: the version-2 grid at this anchor is "
                f"{subject.grid.samples} samples / {subject.grid.digest}, not the published "
                f"{samples} / {digest}. A moved grid digest is a stop (prompt 03a §9)."
            )

    return subjects


# ---------------------------------------------------------------------------------------------
# the oracle column: RadiationModel against the exact T
# ---------------------------------------------------------------------------------------------


def exact_errors(subject: Subject, k_inv_Mpc: float, geo: dict, payload):
    """
    Envelope-relative error of a ``RadiationModel`` run against the exact ``compute_analytic_T``,
    sample by sample, through :func:`radiation_anchors` and :func:`anchor_error`.

    Unlike :func:`sector_errors` this **includes** the initial-condition error, which the
    self-convergence measure removes as common mode. That is what makes it the oracle measurement
    of the floor (§5), and it is prompt 17 §8's first measure.
    """
    kind, exact_T = radiation_anchors(subject.model)["T"]
    out = []
    for z, value in zip(geo["grid"], payload["value_sample"]):
        x = x_local(subject.model, k_inv_Mpc, z.z)
        out.append(
            (
                anchor_error(
                    kind, value, exact_T(k_inv_Mpc, z.z), envelope=T_envelope_exact(x)
                ),
                z.z,
                x,
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# stage 1: the matrix, at full grid coverage, on the version-2 grid under BREAK_POINT_ALL
# ---------------------------------------------------------------------------------------------


def sweep_subject(subject: Subject) -> dict:
    rows = []
    t_model = time.perf_counter()

    for index, k in enumerate(PRODUCTION_K_GRID_INV_MPC):
        k = float(k)
        geo = tk_geometry(subject.cosmology, k, subject.grid)
        measure = sector_error_measure("Tk", subject.model, k, geo)

        def build(setting: TolerancePair):
            return tk_run(
                subject.model,
                k,
                geo,
                setting.atol,
                setting.rtol,
                break_point_kind=BREAK_POINT_ALL,
                task_label="tk_numeric_exit_sweep",
            )

        reference = build(REFERENCE)

        entry = {
            "k": k,
            "x_init": x_local(subject.model, k, geo["grid"].max.z),
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
                summary["vs_exact"] = summarise(exact_errors(subject, k, geo, payload))[
                    "max"
                ]
            entry["candidates"][(atol, rtol)] = summary

        # the initial-condition floor, both ways prompt 17 §8 measured it. The series run is at
        # the *reference* tolerance so that what is measured is the initial datum and not the
        # solver.
        series = build_with_ic(subject, k, geo, REFERENCE, series_initial_data)
        entry["ic_floor"] = summarise(measure(series, reference))
        if subject.is_radiation:
            entry["reference_vs_exact"] = summarise(
                exact_errors(subject, k, geo, reference)
            )

        # README §5 rule 5, in prompt 03's shape: the criterion is the smallest difference this
        # measurement intends to *report* over ten, and a cell that does not stand clear of the
        # reference's own drift is not a difference this measurement reports. So the drift is
        # taken first, every cell is marked resolved or not against it, and the criterion is then
        # restated against the smallest resolved cell.
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
                f"{subject.name} k={k:.5g}: no cell of the matrix exceeds the reference's own "
                f"drift of {drift.max:.3g} of the envelope. The reference has not converged at "
                f"the level this sweep reports and nothing measured through it means anything "
                f"(README §0.2; prompt 03a §9)."
            )
        entry["smallest_reported_difference"] = min(resolved)
        entry["drift"] = replace(drift, smallest_reported_difference=min(resolved))

        rows.append(entry)
        production = entry["candidates"][(PRODUCTION_ATOL, PRODUCTION_RTOL)]
        log_line(
            f"   {subject.name} [{index + 1:2d}/{len(PRODUCTION_K_GRID_INV_MPC)}] "
            f"k={k:.4g}: drift {entry['drift'].max:.2e} "
            f"({'ok' if entry['drift'].passed else 'FAIL'}), "
            f"production {production['max']:.2e}, "
            f"IC floor {entry['ic_floor']['max']:.2e}"
        )

    log_line(f"   {subject.name} matrix done in {time.perf_counter() - t_model:.1f} s")
    return {"name": subject.name, "label": subject.label, "rows": rows}


def build_with_ic(subject: Subject, k: float, geo: dict, setting: TolerancePair, ic_fn):
    return tk_run(
        subject.model,
        k,
        geo,
        setting.atol,
        setting.rtol,
        ic=ic_fn(subject.model, k, geo["grid"].max.z),
        break_point_kind=BREAK_POINT_ALL,
        task_label="tk_numeric_exit_sweep",
    )


# ---------------------------------------------------------------------------------------------
# stage 2: the 2x2 of grid generation against break-point policy, at the production setting
# ---------------------------------------------------------------------------------------------

CONFIGURATIONS = (
    ("v0", BREAK_POINT_DISCONTINUITY),
    ("v0", BREAK_POINT_ALL),
    ("v2", BREAK_POINT_DISCONTINUITY),
)


def sweep_configuration(subject: Subject, generation: str, policy: str) -> dict:
    """
    The production setting alone, over all fifty wavenumbers, on one (grid generation, policy).

    ``("v0", BREAK_POINT_DISCONTINUITY)`` is exactly ``GkTk-remedial`` prompt 17's configuration;
    ``("v2", BREAK_POINT_ALL)`` is production today and comes from :func:`sweep_subject`. The two
    intermediate cells are what separates the grid change from the policy change (§3).
    """
    grid = subject.grid_for(generation)
    rows = []
    t0 = time.perf_counter()

    for k in PRODUCTION_K_GRID_INV_MPC:
        k = float(k)
        geo = tk_geometry(subject.cosmology, k, grid)
        measure = sector_error_measure("Tk", subject.model, k, geo)

        def build(setting: TolerancePair):
            return tk_run(
                subject.model,
                k,
                geo,
                setting.atol,
                setting.rtol,
                break_point_kind=policy,
                task_label="tk_numeric_exit_sweep",
            )

        reference = build(REFERENCE)
        payload = build(TolerancePair(atol=PRODUCTION_ATOL, rtol=PRODUCTION_RTOL))
        summary = summarise(measure(payload, reference))
        summary["evaluations"] = payload["data"].RHS_evaluations
        drift = reference_drift(
            build,
            REFERENCE,
            error_measure=measure,
            smallest_reported_difference=max(summary["max"], 1e-300),
            reference=reference,
            tightened_knob=REFERENCE_TIGHTENED,
        )
        summary["drift"] = drift.max
        summary["drift_passed"] = drift.passed
        summary["k"] = k
        summary["samples"] = len(geo["grid"])
        rows.append(summary)

    log_line(
        f"   {subject.name} {generation}/{policy}: "
        f"worst {max(r['max'] for r in rows):.2e}, "
        f"{sum(1 for r in rows if r['max'] > TARGET_ERROR)}/50 above target "
        f"({time.perf_counter() - t0:.1f} s)"
    )
    return {"generation": generation, "policy": policy, "rows": rows}


# ---------------------------------------------------------------------------------------------
# stage 3: wavenumber_exit_time, through _solve_horizon_exit and never through the store
# ---------------------------------------------------------------------------------------------


class CountingCosmology:
    """
    A cosmology view that counts ``Hubble`` calls.

    ``_solve_horizon_exit`` reads exactly two things off the cosmology it is given, ``Hubble(z)``
    and ``H0``, and the Hubble count is this target's cost in the units README §2 (i) requires:
    counts, never wall time. Nothing else is forwarded differently, and no value is supplied that
    the wrapped object does not already have.
    """

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "calls", 0)

    def Hubble(self, z):
        object.__setattr__(self, "calls", object.__getattribute__(self, "calls") + 1)
        return object.__getattribute__(self, "_inner").Hubble(z)

    def reset(self):
        object.__setattr__(self, "calls", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def hexit_solve(counting, k_obj, offset, setting: TolerancePair):
    """One ``_solve_horizon_exit`` at ``(xtol, rtol) = (setting.atol, setting.rtol)``."""
    counting.reset()
    z = _solve_horizon_exit(
        counting, k_obj, offset, atol=setting.atol, rtol=setting.rtol
    )
    return {"z": float(z), "u": log1p(float(z)), "calls": int(counting.calls)}


def hexit_residual(cosmology, k_inv_Mpc: float, z: float, offset) -> float:
    """``|q|`` at the returned root -- the quantity ``_solve_horizon_exit`` itself checks."""
    return fabs(log(k_inv_Mpc * (1.0 + z) / cosmology.Hubble(z)) - float(offset))


def sweep_exit_time(subject: Subject) -> dict:
    """
    The ``(xtol, rtol)`` matrix at all fifty wavenumbers and the three offsets a grid is built
    from, per model.

    The error measure is the **displacement in ``u = log(1+z)``** from a converged re-solve, which
    is also the *relative* displacement in ``1+z``: ``d(1+z)/(1+z) = du``. That is the measure
    §2.4 asks for, because what a misplaced root does is move a grid by a relative amount, and it
    is taken through :func:`reference_drift` exactly as the ODE sectors' error is -- the facility
    does not care that the "solve" here is a root solve (board standing note 14).
    """
    counting = CountingCosmology(subject.cosmology)
    anchors = radiation_anchors(subject.model) if subject.is_radiation else None
    rows = []
    t0 = time.perf_counter()

    for k in PRODUCTION_K_GRID_INV_MPC:
        k = float(k)
        k_obj = wavenumber(store_id=0, k_inv_Mpc=k, units=UNITS)
        for offset in HEXIT_OFFSETS:

            def build(setting: TolerancePair):
                return hexit_solve(counting, k_obj, offset, setting)

            def measure(candidate, reference):
                return [
                    (
                        fabs(candidate["u"] - reference["u"]),
                        candidate["z"],
                        candidate["u"],
                    )
                ]

            reference = build(HEXIT_REFERENCE)
            cells = {}
            for xtol in HEXIT_XTOL_AXIS:
                for rtol in HEXIT_RTOL_AXIS:
                    setting = TolerancePair(atol=xtol, rtol=rtol)
                    payload = build(setting)
                    cells[(xtol, rtol)] = {
                        "displacement_u": fabs(payload["u"] - reference["u"]),
                        "residual": hexit_residual(
                            subject.cosmology, k, payload["z"], offset
                        ),
                        "calls": payload["calls"],
                        "z": payload["z"],
                        # which term of Brent's xtol + rtol*|u| is the larger at this cell
                        "criterion": xtol + rtol * fabs(reference["u"]),
                        "rtol_term": rtol * fabs(reference["u"]),
                        "xtol_binds": xtol > rtol * fabs(reference["u"]),
                    }

            drift = reference_drift(
                build,
                HEXIT_REFERENCE,
                error_measure=measure,
                smallest_reported_difference=max(
                    min(c["displacement_u"] for c in cells.values()), 1e-300
                ),
                reference=reference,
                tightened_knob=HEXIT_REFERENCE_TIGHTENED,
            )
            for cell in cells.values():
                cell["resolved"] = cell["displacement_u"] > CRITERION_RATIO * drift.max

            entry = {
                "k": k,
                "offset": offset,
                "u_reference": reference["u"],
                "z_reference": reference["z"],
                "reference_calls": reference["calls"],
                "reference_residual": hexit_residual(
                    subject.cosmology, k, reference["z"], offset
                ),
                "cells": cells,
                "drift": drift,
                # the u -> z recovery: one ulp of a double u at this magnitude is this much
                # relative displacement in 1+z, which is the granularity at which z can carry the
                # root at all (CLAUDE.md's redshift-arithmetic note)
                "u_ulp": float(np.spacing(reference["u"])),
                "z_ulp_relative": float(
                    np.spacing(reference["z"]) / (1.0 + reference["z"])
                ),
            }
            if subject.is_radiation:
                kind, exact = anchors["z_exit"]
                z_exact = exact(k, float(offset))
                entry["z_exact"] = z_exact
                entry["u_exact"] = log1p(z_exact)
                entry["reference_vs_exact"] = fabs(log1p(z_exact) - reference["u"])
                for (xtol, rtol), cell in cells.items():
                    cell["vs_exact"] = fabs(log1p(z_exact) - log1p(cell["z"]))
            rows.append(entry)

    log_line(
        f"   {subject.name} exit-time matrix: "
        f"{len(rows)} (k, offset) pairs, worst drift "
        f"{max(r['drift'].max for r in rows):.2e} "
        f"({time.perf_counter() - t0:.1f} s)"
    )
    return {"name": subject.name, "rows": rows}


# ---------------------------------------------------------------------------------------------
# stage 4: what an anchor displacement does to the grid it builds
# ---------------------------------------------------------------------------------------------


def anchor_sensitivity(subject: Subject) -> dict:
    """
    Rebuild the version-2 grid at a perturbed anchor and report what moved.

    This is the only thing that turns a displacement in ``u`` into an *accuracy* rather than a
    curiosity (§2.4). Three things can tell: the sample count, the content digest that tags the
    grid in the datastore, and whether any sample moves further than
    ``DEFAULT_REDSHIFT_RELATIVE_PRECISION``, the tolerance at which
    ``Datastore/SQL/ObjectFactories/redshift.py`` matches an existing redshift row.
    """
    base_values = np.asarray(subject.grid.z_values, dtype=float)
    out = []
    for delta in ANCHOR_PERTURBATIONS:
        z_init = subject.z_init * (1.0 + delta)
        grid = source_grid(
            SOURCE_GRID_V2,
            z_init,
            PRODUCTION_Z_END,
            PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z,
            cosmology=grid_cosmology(subject.cosmology),
            k_inv_Mpc=PRODUCTION_K_GRID_INV_MPC,
        )
        values = np.asarray(grid.z_values, dtype=float)
        digest = redshift_grid_digest(grid.z_values)
        row = {
            "delta": delta,
            "samples": len(values),
            "digest": digest,
            "digest_moved": digest != subject.grid.digest,
            "count_moved": len(values) != len(base_values),
        }
        if len(values) == len(base_values):
            rel = np.abs(values - base_values) / np.maximum(base_values, 1e-300)
            row["max_relative_shift"] = float(rel.max())
            row["bitwise_identical"] = int(np.count_nonzero(values == base_values))
            row["above_row_match"] = int(
                np.count_nonzero(rel > DEFAULT_REDSHIFT_RELATIVE_PRECISION)
            )
        out.append(row)
        log_line(
            f"   {subject.name} anchor +{delta:.1g}: {row['samples']} samples, "
            f"{row['digest']}, max shift {g(row.get('max_relative_shift'))}"
        )
    return {"name": subject.name, "rows": out}


# ---------------------------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------------------------


def per_setting_summary(result: dict) -> dict:
    """Aggregate one model's matrix over the fifty wavenumbers, per ``(atol, rtol)``."""
    out = {}
    for setting in CANDIDATES:
        cells = [row["candidates"][setting] for row in result["rows"]]
        out[setting] = {
            "max": max(c["max"] for c in cells),
            "max_k": result["rows"][
                max(range(len(cells)), key=lambda i: cells[i]["max"])
            ]["k"],
            "median_of_maxima": median(c["max"] for c in cells),
            "above_target": sum(1 for c in cells if c["max"] > TARGET_ERROR),
            "evaluations": sum(c["evaluations"] for c in cells),
            "unresolved": sum(1 for c in cells if not c["resolved"]),
        }
        if "vs_exact" in cells[0]:
            out[setting]["vs_exact"] = max(c["vs_exact"] for c in cells)
    return out


def report_method(subjects, tk_results, started: float) -> None:
    emit("## 2. Method, and what every figure carries")
    emit()
    emit(
        "Every figure below is taken on the **version-2** source grid at each cosmology's **own** "
        "production anchor, and every table says so (README §5 rule 6; board standing note 18). "
        "The three grids:"
    )
    emit()
    table(
        ["model", "anchor $z_{\\rm init}$", "grid", "samples", "digest"],
        [
            [
                s.name,
                f"{s.z_init!r}",
                "version 2",
                s.grid.samples,
                f"`{s.grid.digest}`",
            ]
            for s in subjects
        ],
    )
    emit(
        "The two production anchors are `wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM` and "
        "`PRODUCTION_Z_INIT_QCD` (prompt 02a). The radiation control has no anchor in the record; "
        "prompt 03 built one the same way -- `_solve_horizon_exit(model, 3e8/Mpc, -5)` -- and this "
        "script reproduces its anchor, sample count and digest or refuses to run."
    )
    emit()
    emit(
        f"**The break-point policy is `{BREAK_POINT_ALL}`**, which is what the production call "
        "site passes (`TkNumericIntegration.BREAK_POINT_KIND`, `GkTk-remedial` prompt 19). "
        f"`convergence_reference.tk_run` defaults to `{BREAK_POINT_DISCONTINUITY}` so that "
        "`TK-NUMERIC-ATOL-SWEEP.md` §9's entry point still reproduces §9, so the policy is passed "
        "explicitly at every call in this script and is stated in every table heading. §3 measures "
        "what the difference is worth."
    )
    emit()
    emit(
        "**The reference** is `(atol, rtol) = "
        f"{pair_label(REFERENCE.atol, REFERENCE.rtol)}` per `(model, k)`, scored against "
        f"{pair_label(REFERENCE_TIGHTENED.atol, REFERENCE_TIGHTENED.rtol)} through "
        "`convergence_reference.reference_drift`, which will not hand back a drift without the "
        "verdict attached. It is `GkTk-remedial` prompt 17's pair, kept unchanged so that the two "
        "campaigns' reference-convergence figures are comparable. A cell that does not stand ten "
        "times clear of its own reference's drift is marked **unresolved** and no conclusion is "
        "drawn from it."
    )
    emit()
    rows = []
    for result in tk_results:
        drifts = [row["drift"] for row in result["rows"]]
        rows.append(
            [
                result["name"],
                g(max(d.max for d in drifts)),
                g(median(d.max for d in drifts)),
                f"{sum(1 for d in drifts if d.passed)}/{len(drifts)}",
                g(min(row["smallest_reported_difference"] for row in result["rows"])),
                sum(
                    sum(1 for c in row["candidates"].values() if not c["resolved"])
                    for row in result["rows"]
                ),
            ]
        )
    table(
        [
            "model",
            "worst reference drift",
            "median drift",
            "converged",
            "smallest difference reported",
            f"unresolved cells (of {len(CANDIDATES) * len(tk_results[0]['rows'])})",
        ],
        rows,
    )
    emit(
        "**The radiation column is an oracle, not a self-convergence.** On `RadiationModel` the "
        "exact $T = 3(\\sin x - x\\cos x)/x^3$ is reached through the facility's "
        "`radiation_anchors` registry, so every self-convergence figure in that column has the "
        "distance to truth beside it (README §3.1, §5 rule 5)."
    )
    emit()


def report_policy_and_grid(tk_results, config_results) -> None:
    emit("## 3. What `BREAK_POINT_ALL` and the version-2 grid changed (§6 question 1)")
    emit()
    emit(
        "`GkTk-remedial` prompt 17's figures for this sector -- "
        f"**{PROMPT_17_ABOVE_TARGET['RadiationModel']} / "
        f"{PROMPT_17_ABOVE_TARGET['LambdaCDMModel']} / "
        f"{PROMPT_17_ABOVE_TARGET['QCDModel']}** of 50 wavenumbers above "
        f"{g(TARGET_ERROR)} of the envelope, worst **{g(PROMPT_17_WORST)}** -- were taken on the "
        "**version-0** per-$k$ grid under **`discontinuity`**. Production runs the **version-2** "
        "grid under **`all`**. Both changed, so the 2x2 below is taken at the production setting "
        f"{pair_label(PRODUCTION_ATOL, PRODUCTION_RTOL)} and the two changes are separated rather "
        "than confounded. Each cell carries its own reference and its own drift."
    )
    emit()
    for result in tk_results:
        name = result["name"]
        rows = []
        for config in config_results[name]:
            cells = config["rows"]
            rows.append(
                [
                    f"{'version 0, per-$k$' if config['generation'] == 'v0' else 'version 2'}",
                    f"`{config['policy']}`",
                    g(max(c["max"] for c in cells)),
                    g(median(c["max"] for c in cells)),
                    sum(1 for c in cells if c["max"] > TARGET_ERROR),
                    g(max(c["drift"] for c in cells)),
                    sum(c["evaluations"] for c in cells),
                ]
            )
        production_cells = [
            row["candidates"][(PRODUCTION_ATOL, PRODUCTION_RTOL)]
            for row in result["rows"]
        ]
        rows.append(
            [
                "version 2",
                f"`{BREAK_POINT_ALL}` **(production)**",
                g(max(c["max"] for c in production_cells)),
                g(median(c["max"] for c in production_cells)),
                sum(1 for c in production_cells if c["max"] > TARGET_ERROR),
                g(max(row["drift"].max for row in result["rows"])),
                sum(c["evaluations"] for c in production_cells),
            ]
        )
        emit(f"**{name}**")
        emit()
        table(
            [
                "grid",
                "policy",
                "worst max",
                "median of maxima",
                f"$k$ above {g(TARGET_ERROR)}",
                "worst drift",
                "RHS evals, 50 objects",
            ],
            rows,
        )


def report_axes(tk_results, summaries) -> None:
    emit("## 4. The two axes (§6 question 2)")
    emit()
    emit(
        "`atol` is held at the settled `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` on the `rtol` "
        "axis, and moved one decade either side of it at two `rtol` settings off it. **No `atol` "
        "is recommended here** -- the user settled its value on 2026-09-12 and §7 D1 says no "
        "prompt in this campaign reopens it. What is measured is whether it *binds*, because the "
        "`rtol` recommendation is not separable without the answer."
    )
    emit()
    for result in tk_results:
        name = result["name"]
        summary = summaries[name]
        emit(f"**{name}** -- version-2 grid, `{BREAK_POINT_ALL}`, fifty wavenumbers")
        emit()
        rows = []
        for setting in CANDIDATES:
            cell = summary[setting]
            marker = (
                " **(production)**"
                if setting == (PRODUCTION_ATOL, PRODUCTION_RTOL)
                else ""
            )
            rows.append(
                [
                    pair_label(*setting) + marker,
                    g(cell["max"]),
                    g(cell["max_k"], 4),
                    g(cell["median_of_maxima"]),
                    cell["above_target"],
                    cell["evaluations"],
                    cell["unresolved"],
                    g(cell.get("vs_exact")),
                ]
            )
        table(
            [
                "(atol, rtol)",
                "worst max",
                "at $k$ [1/Mpc]",
                "median of maxima",
                f"$k$ above {g(TARGET_ERROR)}",
                "RHS evals, 50 objects",
                "unresolved",
                "worst vs exact $T$",
            ],
            rows,
        )


def report_target_rule(tk_results, summaries, floor: float) -> None:
    """
    README §6.1's rule, executed on the `rtol` axis rather than argued about.

    Rule 2: the target is scored at the **maximum** over the whole production grid on all three
    models. Rule 3: swept loose to tight, it is the **first** setting that clears the floor.
    """
    emit("### 4.1 README §6.1's rule, applied (§6 question 3)")
    emit()
    emit(
        "The rule is mechanical and is applied here rather than argued about: the target is the "
        "**loosest** `rtol` whose **maximum** envelope-relative error, over all fifty wavenumbers "
        "on **all three** models, is at or below the floor §5 measures -- swept loose to tight, "
        f"first one that clears. The floor is **{g(floor)}** of the envelope."
    )
    emit()
    rows = []
    for rtol in RTOL_AXIS:
        worst = max(
            summaries[r["name"]][(PRODUCTION_ATOL, rtol)]["max"] for r in tk_results
        )
        worst_model = max(
            tk_results,
            key=lambda r: summaries[r["name"]][(PRODUCTION_ATOL, rtol)]["max"],
        )["name"]
        above = sum(
            summaries[r["name"]][(PRODUCTION_ATOL, rtol)]["above_target"]
            for r in tk_results
        )
        evals = sum(
            summaries[r["name"]][(PRODUCTION_ATOL, rtol)]["evaluations"]
            for r in tk_results
        )
        base = sum(
            summaries[r["name"]][(PRODUCTION_ATOL, PRODUCTION_RTOL)]["evaluations"]
            for r in tk_results
        )
        rows.append(
            [
                f"{rtol:.0e}"
                + (" **(production)**" if rtol == PRODUCTION_RTOL else ""),
                g(worst),
                worst_model,
                "**yes**" if worst <= floor else "no",
                f"{above}/150",
                f"{evals} ({(evals / base - 1.0) * 100:+.1f}%)",
            ]
        )
    table(
        [
            "rtol",
            "worst max, three models",
            "on",
            f"clears {g(floor)}?",
            f"$k$ above {g(TARGET_ERROR)}, of 150",
            "RHS evals, 150 objects",
        ],
        rows,
    )


def report_per_k(tk_results) -> None:
    emit("## 8. Appendix: every wavenumber, on the `rtol` axis")
    emit()
    emit(
        "Version-2 grid at each cosmology's own anchor, `"
        + BREAK_POINT_ALL
        + "`, `atol = 1e-13`. Each cell is the **maximum** envelope-relative error over that "
        "wavenumber's grid, with the location of the maximum in $x = k c_s (1+z)/H$ and the "
        "right-hand-side evaluation count. `drift` is that wavenumber's own reference drift, and "
        "an italicised cell did not stand ten times clear of it."
    )
    emit()
    for result in tk_results:
        emit(f"**{result['name']}** -- {result['label']}")
        emit()
        header = ["$k$ [1/Mpc]", "samples", "drift", "IC floor"] + [
            f"rtol {r:.0e}" for r in RTOL_AXIS
        ]
        rows = []
        for row in result["rows"]:
            cells = []
            for rtol in RTOL_AXIS:
                cell = row["candidates"][(PRODUCTION_ATOL, rtol)]
                text = (
                    f"{cell['max']:.3g} @x={cell['max_x']:.4g} ({cell['evaluations']})"
                )
                cells.append(text if cell["resolved"] else f"*{text}*")
            rows.append(
                [
                    g(row["k"], 5),
                    row["samples_returned"],
                    g(row["drift"].max),
                    g(row["ic_floor"]["max"]),
                ]
                + cells
            )
        table(header, rows)


def report_second_and_median(tk_results) -> None:
    emit("## 9. Appendix: maximum, second-largest and median, at three settings")
    emit()
    emit(
        "§4.1 of the prompt asks for all three per `(model, k, atol, rtol)`; the appendix above "
        "carries the maximum at every `rtol`, and this one carries the shape of the distribution "
        "at the production setting, the recommended setting and one step either side of it."
    )
    emit()
    settings = [
        (PRODUCTION_ATOL, PRODUCTION_RTOL),
        (PRODUCTION_ATOL, 1.0e-9),
        (PRODUCTION_ATOL, 1.0e-10),
    ]
    for result in tk_results:
        emit(f"**{result['name']}**")
        emit()
        header = ["$k$ [1/Mpc]"]
        for setting in settings:
            header += [
                f"{pair_label(*setting)} max",
                "2nd",
                "median",
            ]
        rows = []
        for row in result["rows"]:
            entry = [g(row["k"], 5)]
            for setting in settings:
                cell = row["candidates"][setting]
                entry += [g(cell["max"]), g(cell["second"]), g(cell["median"])]
            rows.append(entry)
        table(header, rows)


def report_floor(tk_results) -> None:
    emit(
        "## 5. The initial-condition floor, re-measured on this tree (§6 question 3's input)"
    )
    emit()
    emit(
        "README §2 (f) inherits **"
        + g(INHERITED_IC_FLOOR)
        + " of the envelope**, $k$-independent, from `GkTk-remedial` prompt 17 §8 -- taken on the "
        "**version-0** grid under `discontinuity`. It is an initial-condition truncation rather "
        "than a property of the grid, so it ought to survive the move to the version-2 grid and "
        "`all`; here is whether it does. Both of prompt 17's measures are re-taken: the production "
        "$T=1,T'=0$ against the super-horizon series $T = 1 - x^2/10$ at the reference tolerance "
        "on all three models, and, on the radiation control, the converged run with production "
        "initial data against the exact $T$."
    )
    emit()
    rows = []
    for result in tk_results:
        floors = [row["ic_floor"]["max"] for row in result["rows"]]
        entry = [
            result["name"],
            g(min(floors)),
            g(median(floors)),
            g(max(floors)),
            g(result["rows"][0]["x_init"], 4)
            + "--"
            + g(result["rows"][-1]["x_init"], 4),
        ]
        if "reference_vs_exact" in result["rows"][0]:
            oracle = [row["reference_vs_exact"]["max"] for row in result["rows"]]
            entry.append(f"{g(min(oracle))}--{g(max(oracle))}")
        else:
            entry.append("--")
        rows.append(entry)
    table(
        [
            "model",
            "IC floor, min over $k$",
            "median",
            "max",
            "$x_i$ range",
            "converged run vs exact $T$",
        ],
        rows,
    )


def report_cost(tk_results, summaries) -> None:
    emit("## 6. Cost (§6 question 4)")
    emit()
    emit(
        "Counts, never wall time (README §2 (i)). **`TkNumericIntegration` is one object per "
        "wavenumber -- 50 per model** (`main.py:1215`, inside the $k$ loop alone), so the totals "
        "below *are* the sector's whole production cost for one model. That is three orders of "
        "magnitude smaller than the sector prompt 03 measured, which is why a decade here is not "
        "the decision a decade there would have been."
    )
    emit()
    rows = []
    for result in tk_results:
        summary = summaries[result["name"]]
        base = summary[(PRODUCTION_ATOL, PRODUCTION_RTOL)]["evaluations"]
        entry = [result["name"]]
        for rtol in RTOL_AXIS:
            cell = summary[(PRODUCTION_ATOL, rtol)]
            entry.append(
                f"{cell['evaluations']} ({(cell['evaluations'] / base - 1.0) * 100:+.1f}%)"
            )
        rows.append(entry)
    table(
        ["model"] + [f"rtol {r:.0e}" for r in RTOL_AXIS],
        rows,
    )
    emit(
        "Percentages are against the production `rtol = 1e-8` in the same row. The whole sector, "
        "all three models, is the sum of one column."
    )
    emit()


def report_exit_time(exit_results, subjects) -> None:
    emit("## 7. `wavenumber_exit_time` (§6 question 5)")
    emit()
    emit(
        "Taken through `CosmologyConcepts.wavenumber._solve_horizon_exit` directly and **never "
        "through the datastore**: that target's lookup accepts any stored row at least as tight "
        "as the request and returns the loosest such, so a loosened sweep point would be served a "
        "tighter stored row and the object would report the stored pair "
        "(`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`). All fifty production "
        "wavenumbers, three models, and the three offsets a grid is built from: horizon crossing "
        "itself, $z_{\\rm exit,suph,e5}$ (the source grid's anchor) and $z_{\\rm exit,subh,e4}$ "
        "(where the $G_k$ numeric region stops)."
    )
    emit()
    emit(
        "**The error measure is the displacement in $u = \\log(1+z)$**, which is where the root "
        "lives and is also the *relative* displacement in $1+z$, since $d(1+z)/(1+z) = du$. That "
        "is what makes it an accuracy for a location rather than for a value: a misplaced root "
        "moves a grid by a relative amount. The reference is a converged re-solve at "
        f"`{pair_label(HEXIT_REFERENCE.atol, HEXIT_REFERENCE.rtol)}` -- the pair `_solve_T_z` "
        "already uses -- scored against "
        f"`{pair_label(HEXIT_REFERENCE_TIGHTENED.atol, HEXIT_REFERENCE_TIGHTENED.rtol)}` through "
        "the same `reference_drift` the ODE sectors use."
    )
    emit()

    emit("### 7.1 Is the reference converged, and what does truth say?")
    emit()
    rows = []
    for result in exit_results:
        drifts = [row["drift"].max for row in result["rows"]]
        entry = [
            result["name"],
            len(result["rows"]),
            g(max(drifts)),
            g(median(drifts)),
            g(max(row["reference_residual"] for row in result["rows"])),
        ]
        if "reference_vs_exact" in result["rows"][0]:
            entry.append(g(max(row["reference_vs_exact"] for row in result["rows"])))
        else:
            entry.append("-- (no closed form)")
        rows.append(entry)
    table(
        [
            "model",
            "(k, offset) pairs",
            "worst reference drift [$\\Delta u$]",
            "median",
            "worst $|q|$ at the reference root",
            "reference vs exact $z_{\\rm exit}$",
        ],
        rows,
    )
    emit(
        "The radiation column is the oracle README §3.1 promises: $1 + z = k/(H_0 e^N)$ is "
        "elementary, so the reference's distance to truth is measured rather than inferred, and it "
        "calibrates the self-convergence figure on the two spline models."
    )
    emit()

    emit(
        "### 7.2 The matrix: displacement in $u$, worst over fifty wavenumbers and three offsets"
    )
    emit()
    for result in exit_results:
        emit(f"**{result['name']}**")
        emit()
        header = ["xtol"] + [f"rtol {r:.0e}" for r in HEXIT_RTOL_AXIS]
        rows = []
        for xtol in HEXIT_XTOL_AXIS:
            entry = [xtol_label(xtol)]
            for rtol in HEXIT_RTOL_AXIS:
                cells = [row["cells"][(xtol, rtol)] for row in result["rows"]]
                worst = max(c["displacement_u"] for c in cells)
                binds = sum(1 for c in cells if c["xtol_binds"])
                text = f"{worst:.3g}"
                if binds > 0:
                    text += f" [xtol binds {binds}/{len(cells)}]"
                if (xtol, rtol) == (PRODUCTION_XTOL_HEXIT, PRODUCTION_RTOL_HEXIT):
                    text = "**" + text + "**"
                entry.append(text)
            rows.append(entry)
        table(header, rows)
    emit(
        "The production cell is in bold. `xtol = 0` is `1e-300`, i.e. the `xtol` term switched "
        "off; the rows above it are how the matrix establishes by measurement, rather than by "
        "inference, where the absolute term can bind at all."
    )
    emit()

    emit("### 7.3 Which term of Brent's $x_{\\rm tol} + r_{\\rm tol}|u|$ is active")
    emit()
    rows = []
    for result in exit_results:
        us = [row["u_reference"] for row in result["rows"]]
        production = [
            row["cells"][(PRODUCTION_XTOL_HEXIT, PRODUCTION_RTOL_HEXIT)]
            for row in result["rows"]
        ]
        rows.append(
            [
                result["name"],
                f"{min(us):.4g}--{max(us):.4g}",
                g(min(c["rtol_term"] for c in production)),
                g(max(c["rtol_term"] for c in production)),
                f"{sum(1 for c in production if c['xtol_binds'])}/{len(production)}",
                g(max(c["displacement_u"] for c in production)),
                g(max(c["residual"] for c in production)),
            ]
        )
    table(
        [
            "model",
            "$|u|$ over the production range",
            "rtol term, min",
            "max",
            "$x_{\\rm tol}$ binds at",
            "worst displacement",
            "worst $|q|$",
        ],
        rows,
    )
    emit(
        f"`DEFAULT_HEXIT_TOLERANCE = {g(DEFAULT_HEXIT_TOLERANCE)}` is the post-hoc guard "
        "`_solve_horizon_exit` applies to $|q|$ after the solve; it is not a tolerance being swept "
        "here, and the worst $|q|$ column says how much room every setting has against it."
    )
    emit()

    emit("### 7.4 How much of the root survives the $u \\to z$ recovery")
    emit()
    emit(
        "`_solve_horizon_exit` solves in $u$ and returns `exp(u) - 1`. One ulp of a double $u$ at "
        "these magnitudes is a *relative* step in $1+z$ of $\\mathrm{ulp}(u)$, so the returned $z$ "
        "can carry the root only to that granularity -- which is coarser than $z$'s own ulp by "
        "the factor $u$ (`CLAUDE.md`'s redshift-arithmetic note). The question is whether that "
        "granularity ever competes with the tolerance:"
    )
    emit()
    rows = []
    for result in exit_results:
        u_ulp = [row["u_ulp"] for row in result["rows"]]
        z_ulp = [row["z_ulp_relative"] for row in result["rows"]]
        production = [
            row["cells"][(PRODUCTION_XTOL_HEXIT, PRODUCTION_RTOL_HEXIT)]
            for row in result["rows"]
        ]
        tightest = [row["cells"][(1.0e-300, 1.0e-14)] for row in result["rows"]]
        rows.append(
            [
                result["name"],
                g(max(u_ulp)),
                g(max(z_ulp)),
                g(max(c["criterion"] for c in production)),
                g(max(c["criterion"] for c in production) / max(u_ulp), 4),
                g(max(c["criterion"] for c in tightest)),
                g(max(c["criterion"] for c in tightest) / max(u_ulp), 4),
            ]
        )
    table(
        [
            "model",
            "worst ulp($u$) [= relative in $1+z$]",
            "worst ulp($z$)/$(1+z)$",
            "production criterion",
            "x above ulp($u$)",
            "tightest criterion in the matrix",
            "x above ulp($u$)",
        ],
        rows,
    )


def report_exit_target(exit_results) -> None:
    """
    The `rtol` axis at the production `xtol`, in the two readings §6 question 6 has to separate:
    what Brent *guarantees* at the widest production `|u|`, and what the solve actually achieved
    at the worst of the 150 (k, offset) pairs per model.
    """
    emit("### 7.6 The axis, the guarantee, the achieved displacement, and the cost")
    emit()
    emit(
        "At the production `xtol = 1e-10`. The **guarantee** is Brent's own stopping bound "
        "$x_{\\rm tol} + r_{\\rm tol}|u|$ at the largest $|u|$ any production solve reaches; the "
        "**achieved** column is the measured displacement from the converged re-solve, worst over "
        "all fifty wavenumbers and all three offsets. The two are not the same number and the "
        "difference is the whole of §6 question 6: a bound holds on any machine and any libm, an "
        "achieved value holds on this one."
    )
    emit()
    u_max = max(row["u_reference"] for result in exit_results for row in result["rows"])
    rows = []
    for rtol in HEXIT_RTOL_AXIS:
        guarantee = PRODUCTION_XTOL_HEXIT + rtol * u_max
        entry = [
            f"{rtol:.0e}"
            + (" **(production)**" if rtol == PRODUCTION_RTOL_HEXIT else ""),
            g(guarantee),
            "yes" if guarantee <= DEFAULT_REDSHIFT_RELATIVE_PRECISION else "**no**",
        ]
        calls = 0
        for result in exit_results:
            cells = [
                row["cells"][(PRODUCTION_XTOL_HEXIT, rtol)] for row in result["rows"]
            ]
            entry.append(g(max(c["displacement_u"] for c in cells)))
            calls += sum(c["calls"] for c in cells)
        entry.append(calls)
        rows.append(entry)
    table(
        [
            "rtol",
            f"guarantee at $|u| = {u_max:.4g}$",
            f"clears {g(DEFAULT_REDSHIFT_RELATIVE_PRECISION)}?",
        ]
        + [f"achieved, {r['name']}" for r in exit_results]
        + ["Hubble calls, all three models"],
        rows,
    )
    emit(
        "**Cost is the last column and it is negligible**: the whole exit-time stage of a "
        "production run is 50 wavenumbers times the offsets `find_horizon_exit_time` asks for, "
        "and one solve is tens of Hubble evaluations. Counts, never wall time (README §2 (i))."
    )
    emit()


def report_exit_per_k(exit_results) -> None:
    emit("## 10. Appendix: every wavenumber, exit time at the production setting")
    emit()
    emit(
        f"`(xtol, rtol) = {pair_label(PRODUCTION_XTOL_HEXIT, PRODUCTION_RTOL_HEXIT)}`, the "
        "displacement in $u$ from the converged re-solve at each of the three offsets, with that "
        "$(k, \\rm{offset})$'s own reference drift and the Hubble-call count. `offset 0` is "
        "horizon crossing, `-5` is the source grid's anchor, `+4` is where the $G_k$ numeric "
        "region stops."
    )
    emit()
    for result in exit_results:
        emit(f"**{result['name']}**")
        emit()
        by_k = {}
        for row in result["rows"]:
            by_k.setdefault(row["k"], {})[row["offset"]] = row
        rows = []
        for k in sorted(by_k):
            entry = [g(k, 5)]
            for offset in HEXIT_OFFSETS:
                row = by_k[k][offset]
                cell = row["cells"][(PRODUCTION_XTOL_HEXIT, PRODUCTION_RTOL_HEXIT)]
                entry += [
                    g(cell["displacement_u"]),
                    g(row["drift"].max),
                    cell["calls"],
                ]
            rows.append(entry)
        header = ["$k$ [1/Mpc]"]
        for offset in HEXIT_OFFSETS:
            header += [f"offset {offset:+d}: $\\Delta u$", "drift", "calls"]
        table(header, rows)


def report_anchor_sensitivity(sensitivity) -> None:
    emit("### 7.5 What a displacement does to the grid it anchors (§2.4)")
    emit()
    emit(
        "The source grid's top is $z_{\\rm exit,suph,e5}$ of the earliest-exiting wavenumber, so a "
        "displacement of the root is a relative displacement of the whole lattice. Two things in "
        "the tree can tell: the **content digest** that tags the grid "
        "(`CosmologyConcepts.redshift.redshift_grid_digest`, `main.py:854`), which digests the "
        "exact bits, and the **row match**, "
        f"`DEFAULT_REDSHIFT_RELATIVE_PRECISION = {g(DEFAULT_REDSHIFT_RELATIVE_PRECISION)}`, at "
        "which `Datastore/SQL/ObjectFactories/redshift.py:40` reuses an existing redshift row. The "
        "version-2 grid is rebuilt at perturbed anchors below."
    )
    emit()
    for result in sensitivity:
        emit(f"**{result['name']}**")
        emit()
        table(
            [
                "relative anchor shift",
                "samples",
                "digest",
                "digest moved?",
                "max relative row shift",
                "rows bitwise identical",
                f"rows above {g(DEFAULT_REDSHIFT_RELATIVE_PRECISION)}",
            ],
            [
                [
                    g(row["delta"]),
                    row["samples"],
                    f"`{row['digest']}`",
                    "**yes**" if row["digest_moved"] else "no",
                    g(row.get("max_relative_shift")),
                    row.get("bitwise_identical", "-- (count moved)"),
                    row.get("above_row_match", "--"),
                ]
                for row in result["rows"]
            ],
        )


def main() -> None:
    started = time.perf_counter()
    subjects = build_subjects()

    log_line(
        "** stage 1: the (atol, rtol) matrix on the version-2 grid under BREAK_POINT_ALL"
    )
    tk_results = [sweep_subject(subject) for subject in subjects]
    summaries = {r["name"]: per_setting_summary(r) for r in tk_results}

    log_line("** stage 2: the grid x policy 2x2 at the production setting")
    config_results = {
        subject.name: [
            sweep_configuration(subject, generation, policy)
            for generation, policy in CONFIGURATIONS
        ]
        for subject in subjects
    }

    log_line("** stage 3: the wavenumber_exit_time matrix")
    exit_results = [sweep_exit_time(subject) for subject in subjects]

    log_line("** stage 4: anchor sensitivity of the version-2 grid")
    sensitivity = [anchor_sensitivity(subject) for subject in subjects]

    elapsed = time.perf_counter() - started

    emit(
        f"<!-- generated {date.today().isoformat()} by\n"
        f"     PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/tk_numeric_exit_sweep.py\n"
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__} -->"
    )
    emit()
    report_method(subjects, tk_results, started)
    report_policy_and_grid(tk_results, config_results)
    report_axes(tk_results, summaries)
    # the smallest floor measured anywhere: "clears the floor" then means clears it at every
    # wavenumber on every model, which is the conservative reading of §6.1 rule 2
    floor = min(
        row["ic_floor"]["max"] for result in tk_results for row in result["rows"]
    )
    report_target_rule(tk_results, summaries, floor)
    report_floor(tk_results)
    report_cost(tk_results, summaries)
    report_exit_time(exit_results, subjects)
    report_anchor_sensitivity(sensitivity)
    report_exit_target(exit_results)
    report_per_k(tk_results)
    report_second_and_median(tk_results)
    report_exit_per_k(exit_results)

    log_line(f"** done in {elapsed:.0f} s")


if __name__ == "__main__":
    main()
