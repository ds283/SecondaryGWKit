"""
Does the transfer function's numeric sector still need ``BREAK_POINT_ALL``, now that the
cosmology declares three break points rather than four hundred and seven?
(``prompts/qcd-background-audit/`` prompt 08.)

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/per_sector_policy_remeasure.py

It needs no Ray and no datastore: ``numeric_with_phase_cut`` is called through its undecorated
``_function`` with the stand-ins of ``ComputeTargets/tests/wkb_reference.py``.

**It changes nothing.** No production module is touched and neither ``BREAK_POINT_KIND`` moves
whatever comes out: that constant is in a datastore lookup key (``prompts/GkTk-remedial`` prompt
20) and moving it is ``prompts/qcd-background-audit/README.md`` §7 **D5**, the user's decision.
This script measures; the prompt reports.

**Provenance.** This is a *reduction* of ``docs/gktk-remedial/tk_numeric_atol_sweep.py`` as that
file stood at ``c2bf596``, copied rather than extended because it belongs to another campaign
(the precedent is ``docs/qcd-background-audit/generate_qcd_references.py``, prompt 02, and
``[13-scoped-run-driver-k-grid-literal]``). Everything between the two "copied verbatim" banners
below is that file's text, unmodified, so that the measurement is the same measurement: the
stand-ins, the two sector geometries, the two ``run`` wrappers, the envelope-relative error, the
control, and prompt 19's ``SECTOR_POLICY``/``_bitwise_equal`` helpers. The reduction drops
§§1--8's ``atol`` sweep, prompt 18's §9 break-point entry point and prompt 19's §10 reporting,
none of which this prompt re-takes. What is new is below the second banner.

**Why it exists.** ``GkTk-remedial`` prompt 19 chose
``TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL`` *on measurement*: with the
equation-of-state jumps alone, 3 of 50 QCD wavenumbers missed the 3.4e-08 reference-convergence
criterion (worst 1.97e-07); with the 404 interior knots of the ``T(z)`` spline added, 4.65e-09 or
better. Prompt 07 of this campaign removed those knots -- ``integration_break_points`` now
declares the equation of state's own temperature crossings and nothing else, **3** points under
``BREAK_POINT_ALL`` and **2** under ``BREAK_POINT_DISCONTINUITY`` -- so prompt 19's justification
has to be re-taken against a set in which the two policies differ by a single kink
(``EOS_T_LO``, where ``w`` changes analytic form but ``g_s`` does not step).

What it measures, per prompt 08 §2, for ``RadiationModel``, ``LambdaCDMModel`` and ``QCDModel``
at the 50 production source wavenumbers:

1. QCD $T_k$, all 50, reference-convergence drift under **both** policies against the 3.4e-08
   criterion: the worst of each and the count above the criterion;
2. QCD $G_k$, all 50, the same -- prompt 19 measured that sector under its own policy only, so
   the ``BREAK_POINT_ALL`` column for $G_k$ is new;
3. the control: ``RadiationModel`` and ``LambdaCDMModel`` declare nothing, so both policies must
   be **bit-identical** in both sectors and must reproduce prompt 19's §10 figures;
4. the cost, in right-hand-side evaluations per object and over the grid, per sector per policy,
   with wall time alongside and the load average beside it -- the counts are the reproducible
   measure (``prompts/GkTk-remedial/README.md`` §5 note 14) and the seconds are one machine's;

and, beyond prompt 08 §2's list, a third column that is neither policy: the same drift with the
cosmology's declaration suppressed altogether, because the board entry this prompt inherits,
``[04-unsplit-tk-run-now-meets-the-criterion]``, measured exactly that at one wavenumber and asks
whether a fifty-wavenumber sweep agrees.

Output is a markdown fragment on stdout plus a progress log on stderr.
"""

# =============================================================================================
# copied verbatim from docs/gktk-remedial/tk_numeric_atol_sweep.py at c2bf596 -- BEGIN
# =============================================================================================

import platform
import sys
import time
from datetime import date
from math import cos, fabs, hypot, log10, sin, sqrt
from statistics import median

import numpy as np
import scipy

from ComputeTargets.TkNumericIntegration import RHS as Tk_RHS
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
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
from Quadrature.integrators.numeric_with_phase_cut import numeric_with_phase_cut
from Units import Mpc_units

# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------

UNITS = Mpc_units()

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

# main.py:630, :1199 -- still part of the call, no longer part of the diagnostic (prompt 11)
PRODUCTION_DELTA_LOGZ = 1.0 / 100.0

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
# stand-ins (the pattern of ComputeTargets/tests/test_tk_numeric_atol.py, prompt 12)
# ---------------------------------------------------------------------------------------------


class _Wavenumber:
    def __init__(self, k: float, store_id: int, units):
        self.k = float(k)
        self.k_inv_Mpc = float(k)
        self.store_id = store_id
        self.units = units


class _KExit:
    """A ``wavenumber_exit_time`` stand-in: ``.k`` and ``.z_exit``."""

    def __init__(self, k: float, units, z_exit: float, store_id: int = 1):
        self.k = _Wavenumber(k, store_id, units)
        self.z_exit = z_exit


class _Proxy:
    """A ``ModelProxy`` stand-in: ``.get()`` and ``.units`` (for ``check_units``)."""

    def __init__(self, model, units):
        self._model = model
        self.units = units

    def get(self):
        return self._model


# ---------------------------------------------------------------------------------------------
# the production geometry of a TkNumericIntegration work item, and the runs on it
# ---------------------------------------------------------------------------------------------


def geometry(cosmology, k_inv_Mpc: float) -> dict:
    """
    ``main.py``'s ``build_Tk_numeric_work`` geometry for one wavenumber: the production source
    grid (100 samples per decade of z) from five e-folds outside the horizon, truncated below at
    ``0.85 z_e6``, with the ``(z_e3, z_e6)`` stop window.

    ``cosmology`` is anything with ``Hubble(z)`` and ``H0`` -- the ``RadiationModel`` stand-in
    itself, or the real cosmology behind ``LambdaCDMModel`` / ``QCDModel``.
    """
    z_exit = horizon_exit_z(cosmology, k_inv_Mpc, 0.0)
    z_e3 = horizon_exit_z(cosmology, k_inv_Mpc, 3.0)
    z_e6 = horizon_exit_z(cosmology, k_inv_Mpc, 6.0)
    z_source = horizon_exit_z(
        cosmology, k_inv_Mpc, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
    )

    grid = production_source_grid(z_source).truncate(0.85 * z_e6, keep="higher-include")
    return {"z_exit": z_exit, "z_e3": z_e3, "z_e6": z_e6, "grid": grid}


def run(
    model,
    k_inv_Mpc: float,
    geo: dict,
    atol: float,
    rtol: float,
    ic=None,
    break_point_kind: str = BREAK_POINT_DISCONTINUITY,
) -> dict:
    """
    One ``TkNumericIntegration`` solve through the undecorated ``numeric_with_phase_cut``.

    ``ic`` is ``(T, dT/dz)`` at the top of the grid; ``None`` means the production
    ``T = 1, T' = 0``. ``warn_unresolved_osc=False`` is what both production integrators now pass
    (prompt 16); it gates the printed warning and nothing else.

    ``break_point_kind`` defaults to ``BREAK_POINT_DISCONTINUITY`` -- what ``numeric_with_phase_cut``
    asked for unconditionally when §9 was measured, so that §9's entry point reproduces §9. The
    production ``TkNumericIntegration`` call site passes ``BREAK_POINT_ALL`` since prompt 19, and
    §10's entry point passes it here.
    """
    grid = geo["grid"]
    z_init = grid.max
    value, deriv = (1.0, 0.0) if ic is None else ic

    return numeric_with_phase_cut._function(
        _Proxy(model, UNITS),
        _KExit(k_inv_Mpc, UNITS, geo["z_exit"]),
        z_init,
        grid,
        initial_value=value,
        initial_deriv=deriv,
        RHS=Tk_RHS,
        omega_sq=Tk_omegaEff_sq,
        atol=atol,
        rtol=rtol,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        mode="stop",
        stop_search_window_z_begin=min(geo["z_e3"], z_init.z),
        stop_search_window_z_end=geo["z_e6"],
        task_label="tk_numeric_atol_sweep",
        object_label="Tk(z)",
        warn_unresolved_osc=False,
        break_point_kind=break_point_kind,
    )


# ---------------------------------------------------------------------------------------------
# the phase variable, the envelope, and the errors
# ---------------------------------------------------------------------------------------------


def x_local(model, k_inv_Mpc: float, z: float) -> float:
    """
    The transfer function's dimensionless phase variable, ``x = k c_s (1+z)/H``.

    In exact radiation this is ``k c_s tau`` identically (``tau = 1/(H0(1+z))``), which is the
    ``x`` of review section 12.4-12.5 and of prompt 12's measurements; on a real background it is
    the same quantity evaluated locally, and the two agree to the matter fraction deep in the
    radiation era. It is used only to *label* where an error falls.
    """
    functions = model.functions
    c_s = sqrt(functions.wPerturbations(z))
    return k_inv_Mpc * c_s * (1.0 + z) / functions.Hubble(z)


def sample_errors(model, k_inv_Mpc: float, geo: dict, candidate: dict, reference: dict):
    """
    Envelope-relative error of ``candidate`` against ``reference``, sample by sample.

    README section 6: the denominator is the local Liouville-Green envelope
    ``hypot(T, T'/omega)`` of the reference run, with ``omega = sqrt(Tk_omegaEff_sq)``; samples
    where ``omega^2 <= 0`` are skipped, the mode not being oscillatory there.

    :return: list of ``(error, z, x)``, in the returned (descending z) order
    """
    out = []
    for z, value, ref_value, ref_deriv in zip(
        geo["grid"],
        candidate["value_sample"],
        reference["value_sample"],
        reference["deriv_sample"],
    ):
        omega_sq = Tk_omegaEff_sq(model, k_inv_Mpc, z.z)
        if omega_sq <= 0.0:
            continue
        envelope = hypot(ref_value, ref_deriv / sqrt(omega_sq))
        out.append(
            (
                envelope_relative_error(value, ref_value, envelope),
                z.z,
                x_local(model, k_inv_Mpc, z.z),
            )
        )
    return out


def summarise(errors) -> dict:
    """
    Maximum (and where it fell), second-largest, median, and the value at the **last** returned
    sample, of a list of ``(error, z, x)``.

    The terminal value is not in the prompt's list. It is here because it is what separates "one
    bad sample" from "one bad step whose consequence is carried to the end of the integration":
    the samples are in descending z, so the last entry is the deepest point reached, and it is
    the part of the run ``TkWKBIntegration`` reads as its initial condition.
    """
    ordered = sorted(errors, key=lambda e: e[0], reverse=True)
    worst = ordered[0]
    return {
        "max": worst[0],
        "max_z": worst[1],
        "max_x": worst[2],
        "second": ordered[1][0],
        "median": median(e[0] for e in errors),
        "terminal": errors[-1][0],
        "terminal_x": errors[-1][2],
        "samples": len(errors),
    }


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


from ComputeTargets.GkNumericIntegration import RHS as Gk_RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.tests.wkb_reference import production_response_grid
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
    ``main.py``'s ``build_Gk_numeric_work`` geometry for one object: the source redshift five
    e-folds outside the horizon (the top of the universal source grid), the *response* grid --
    ``winnow(12)`` of the source grid, which is what ``GkNumericIntegration`` is sampled on -- cut
    to the source redshift above and to ``0.85 z_e6`` below, and the ``(z_e3, z_e6)`` stop window.

    ``GkNumericIntegration`` is one object per ``(k, z_source)``; one source redshift per $k$ is
    taken here, the outermost, which is the longest and therefore the least favourable run.
    """
    z_exit = horizon_exit_z(cosmology, k_inv_Mpc, 0.0)
    z_e3 = horizon_exit_z(cosmology, k_inv_Mpc, 3.0)
    z_e6 = horizon_exit_z(cosmology, k_inv_Mpc, 6.0)
    z_source = horizon_exit_z(
        cosmology, k_inv_Mpc, -float(PRODUCTION_SUPERHORIZON_EFOLDS)
    )

    source_grid = production_source_grid(z_source)
    grid = (
        production_response_grid(source_grid)
        .truncate(source_grid.max, keep="lower")
        .truncate(0.85 * z_e6, keep="higher-include")
    )
    return {"z_exit": z_exit, "z_e3": z_e3, "z_e6": z_e6, "grid": grid}


def run_gk(
    model,
    k_inv_Mpc: float,
    geo: dict,
    atol: float,
    rtol: float,
    break_point_kind: str = BREAK_POINT_DISCONTINUITY,
) -> dict:
    """
    One ``GkNumericIntegration`` solve through the undecorated ``numeric_with_phase_cut``.

    ``break_point_kind`` is the sector's production policy *and* the module default; it is named
    here for the same reason the production call site names it (prompt 19).
    """
    grid = geo["grid"]
    z_init = grid.max
    return numeric_with_phase_cut._function(
        _Proxy(model, UNITS),
        _KExit(k_inv_Mpc, UNITS, geo["z_exit"]),
        z_init,
        grid,
        initial_value=0.0,
        initial_deriv=1.0,
        RHS=Gk_RHS,
        omega_sq=Gk_omegaEff_sq,
        atol=atol,
        rtol=rtol,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        mode="stop",
        stop_search_window_z_begin=min(geo["z_e3"], z_init.z),
        stop_search_window_z_end=geo["z_e6"],
        task_label="gk_break_point_sweep",
        object_label="Gr_k(z, z')",
        warn_unresolved_osc=False,
        break_point_kind=break_point_kind,
    )


SECTORS = {
    "Tk": {"geometry": geometry, "run": run, "omega_sq": Tk_omegaEff_sq},
    "Gk": {"geometry": gk_geometry, "run": run_gk, "omega_sq": Gk_omegaEff_sq},
}


def sector_errors(sector: str, model, k_inv_Mpc: float, geo, candidate, reference):
    """
    :func:`sample_errors` for either sector: envelope-relative against the reference run, with
    the sector's own effective frequency supplying the Liouville-Green envelope.
    """
    omega_sq_fn = SECTORS[sector]["omega_sq"]
    out = []
    for z, value, ref_value, ref_deriv in zip(
        geo["grid"],
        candidate["value_sample"],
        reference["value_sample"],
        reference["deriv_sample"],
    ):
        omega_sq = omega_sq_fn(model, k_inv_Mpc, z.z)
        if omega_sq <= 0.0:
            continue
        envelope = hypot(ref_value, ref_deriv / sqrt(omega_sq))
        out.append(
            (
                envelope_relative_error(value, ref_value, envelope),
                z.z,
                x_local(model, k_inv_Mpc, z.z),
            )
        )
    return out


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


# =============================================================================================
# copied verbatim from docs/gktk-remedial/tk_numeric_atol_sweep.py at c2bf596 -- END
# =============================================================================================


# =============================================================================================
# prompt 08 of prompts/qcd-background-audit -- new below this line
#
# GkTk-remedial prompt 19 asked "does the transfer function's numeric sector need the knots?" and
# answered yes, on a background whose knots carried a 1e-04-level defect of their own. Prompt 07
# of this campaign has removed them. The same question, re-taken, is now a question about a single
# kink: BREAK_POINT_ALL and BREAK_POINT_DISCONTINUITY differ only by EOS_T_LO = 0.002 GeV, the one
# equation-of-state branch join at which w changes analytic form while g_s does not step.
#
# Prompt 08 §1 names three possible states of the world, and this script is written to tell them
# apart rather than to confirm one:
#
#   (a) both policies converge at all 50 QCD wavenumbers in the Tk sector -- the knots were a
#       proxy for the representation's defect, the per-sector distinction becomes vestigial, and
#       nothing changes but the comment blocks;
#   (b) the Tk sector still needs more split points than the cosmology declares -- stop and ask;
#   (c) the Gk sector has moved -- something in prompts 04-07 did more than it was supposed to,
#       and that is a stop as well.
#
# Neither BREAK_POINT_KIND is read by anything here: both policies are measured on both sectors,
# so the answer does not depend on which one production happens to ask for today.
# =============================================================================================

import os

# prompt 19's §10 figures, which are what "unchanged" is measured against. Grid totals are sums of
# the right-hand-side evaluation counts in §10.6's per-wavenumber tables, which are integers and
# do not depend on the machine; the worst drifts are §10.1's, to the precision it printed them.
PROMPT_19_SECTION_10 = {
    ("Tk", "RadiationModel"): {"worst": 4.21e-11, "grid_evaluations": 401677},
    ("Tk", "LambdaCDMModel"): {"worst": 5.70e-11, "grid_evaluations": 429178},
    ("Tk", "QCDModel"): {
        "worst": 8.72e-09,
        "worst_jumps": 1.97e-07,
        "grid_evaluations": 1576030,
        "grid_evaluations_jumps": 492158,
    },
    ("Gk", "RadiationModel"): {"worst": 1.94e-11, "grid_evaluations": 637138},
    ("Gk", "LambdaCDMModel"): {"worst": 2.10e-11, "grid_evaluations": 640213},
    ("Gk", "QCDModel"): {"worst": 8.41e-09, "grid_evaluations": 666011},
}

# the three QCD Tk wavenumbers that were above the criterion under BREAK_POINT_DISCONTINUITY when
# prompt 19 measured it (§10.6, indices 13, 23 and 37), with the drift it recorded for each under
# each policy. These are the wavenumbers prompt 19's decision actually turned on.
PROMPT_19_TK_OFFENDERS = (
    (13, 8.366e05, 6.1e-08, 6.9e-10),
    (23, 4.287e06, 2.0e-07, 4.7e-09),
    (37, 4.223e07, 3.5e-08, 2.3e-09),
)

POLICIES = (BREAK_POINT_ALL, BREAK_POINT_DISCONTINUITY)


def load_average() -> str:
    """
    The one-, five- and fifteen-minute load averages, printed beside every wall-clock figure.

    This machine was under heavy and erratic external load while prompt 08 ran, so a second is not
    a reproducible unit here and the document says so wherever it quotes one. The evaluation
    counts are unaffected and are the measure.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except (AttributeError, OSError):
        return "unavailable"
    return f"{one:.1f} / {five:.1f} / {fifteen:.1f}"


def both_policies_sweep(
    name: str, model, cosmology, sector: str, declares: bool
) -> dict:
    """
    Every wavenumber, for one (model, sector), under **both** policies -- which is the difference
    from prompt 19's :func:`per_sector_sweep`, where each sector was measured under its own.

    For a model that declares nothing the two policies are the same code path, so the drift is
    measured once (under ``BREAK_POINT_ALL``) and what is measured twice is the *production* run,
    compared with ``==`` sample by sample: the statement wanted there is bit-identity, not a
    second drift figure. For every (model, sector) the production run is also issued a third time
    with ``break_point_kind`` omitted, which is what establishes that the module default is still
    ``BREAK_POINT_DISCONTINUITY``.
    """
    geometry_fn = SECTORS[sector]["geometry"]
    run_fn = SECTORS[sector]["run"]
    production_atol = TK_PRODUCTION_ATOL if sector == "Tk" else GK_PRODUCTION_ATOL

    rows = []
    t_model = time.perf_counter()
    for index, k in enumerate(PRODUCTION_K_GRID):
        k = float(k)
        geo = geometry_fn(cosmology, k)
        z_lo, z_hi = float(geo["grid"].min), geo["grid"].max.z

        entry = {"k": k, "index": index, "drift": {}, "evaluations": {}, "segments": {}}
        production = {}
        for policy in POLICIES:
            entry["segments"][policy] = 1 + len(
                declared_discontinuities_in_z(model, z_lo, z_hi, kind=policy)
            )
            production[policy] = run_fn(
                model, k, geo, production_atol, PRODUCTION_RTOL, break_point_kind=policy
            )
            entry["evaluations"][policy] = production[policy]["data"].RHS_evaluations

            if declares or policy == BREAK_POINT_ALL:
                reference = run_fn(
                    model,
                    k,
                    geo,
                    REFERENCE_ATOL,
                    REFERENCE_RTOL,
                    break_point_kind=policy,
                )
                tightened = run_fn(
                    model,
                    k,
                    geo,
                    TIGHTENED_ATOL,
                    TIGHTENED_RTOL,
                    break_point_kind=policy,
                )
                entry["drift"][policy] = summarise(
                    sector_errors(sector, model, k, geo, tightened, reference)
                )

        if not declares:
            # the two policies are the same request here, and the statement is that they are also
            # the same numbers -- not a second drift figure
            entry["drift"][BREAK_POINT_DISCONTINUITY] = entry["drift"][BREAK_POINT_ALL]
            entry["policy_identical"] = _bitwise_equal(
                production[BREAK_POINT_ALL], production[BREAK_POINT_DISCONTINUITY]
            )

        default = run_fn(model, k, geo, production_atol, PRODUCTION_RTOL)
        entry["default_is_discontinuity"] = _bitwise_equal(
            default, production[BREAK_POINT_DISCONTINUITY]
        )

        # how far the *answer* moves between the two policies, in the envelope-relative measure
        entry["shift"] = summarise(
            sector_errors(
                sector,
                model,
                k,
                geo,
                production[BREAK_POINT_DISCONTINUITY],
                production[BREAK_POINT_ALL],
            )
        )["max"]

        rows.append(entry)
        log(
            f"   {sector} {name} [{index + 1:2d}/{len(PRODUCTION_K_GRID)}] k={k:.4g}: "
            f"segments {entry['segments'][BREAK_POINT_ALL]}/"
            f"{entry['segments'][BREAK_POINT_DISCONTINUITY]}, drift "
            f"{entry['drift'][BREAK_POINT_ALL]['max']:.2e}/"
            f"{entry['drift'][BREAK_POINT_DISCONTINUITY]['max']:.2e}, evals "
            f"{entry['evaluations'][BREAK_POINT_ALL]}/"
            f"{entry['evaluations'][BREAK_POINT_DISCONTINUITY]}"
        )

    log(f"   {sector} {name} done in {time.perf_counter() - t_model:.1f} s")
    return {"name": name, "sector": sector, "declares": declares, "rows": rows}


def unsplit_sweep(name: str, model, cosmology, sector: str) -> dict:
    """
    The same drift measurement again with the cosmology's declaration *suppressed altogether* --
    neither policy, but the historic single-``solve_ivp`` path.

    This is not one of the two policies prompt 08 §2 asks about, and nothing in production can
    request it. It is here because the board entry this prompt inherits,
    ``[04-unsplit-tk-run-now-meets-the-criterion]``, is a measurement of exactly this quantity at
    **one** wavenumber (1.0213e-06 on prompt 04's tree, 2.2767e-08 on prompt 05's, against the
    3.4e-08 criterion, at k = 4.972e7/Mpc), and the natural way to say whether a fifty-wavenumber
    sweep agrees with it is to take the same column across the grid.

    ``_UnsplitModel`` carries the real ``functions`` behind a cosmology that raises
    ``AttributeError`` for ``integration_break_points``, so the physics is untouched and nothing
    is monkeypatched; it is prompt 18's own device, copied with the rest of the harness.
    """
    geometry_fn = SECTORS[sector]["geometry"]
    run_fn = SECTORS[sector]["run"]
    production_atol = TK_PRODUCTION_ATOL if sector == "Tk" else GK_PRODUCTION_ATOL
    unsplit = _UnsplitModel(model, cosmology)

    rows = []
    t_model = time.perf_counter()
    for index, k in enumerate(PRODUCTION_K_GRID):
        k = float(k)
        geo = geometry_fn(cosmology, k)

        reference = run_fn(unsplit, k, geo, REFERENCE_ATOL, REFERENCE_RTOL)
        tightened = run_fn(unsplit, k, geo, TIGHTENED_ATOL, TIGHTENED_RTOL)
        production = run_fn(unsplit, k, geo, production_atol, PRODUCTION_RTOL)
        drift = summarise(sector_errors(sector, model, k, geo, tightened, reference))

        rows.append(
            {
                "k": k,
                "index": index,
                "drift": drift,
                "evaluations": production["data"].RHS_evaluations,
            }
        )
        log(
            f"   unsplit {sector} {name} [{index + 1:2d}/{len(PRODUCTION_K_GRID)}] "
            f"k={k:.4g}: drift {drift['max']:.2e}, {rows[-1]['evaluations']} evals"
        )

    log(f"   unsplit {sector} {name} done in {time.perf_counter() - t_model:.1f} s")
    return {"name": name, "sector": sector, "rows": rows}


def report_unsplit(unsplit_results, results) -> None:
    emit("### 2b. And if the cosmology declared nothing at all")
    emit()
    emit(
        "Neither policy, but the third column the board entry "
        "`[04-unsplit-tk-run-now-meets-the-criterion]` is about: the same runs with "
        "`integration_break_points` suppressed, so the integrator takes its historic "
        "single-`solve_ivp` path and never restarts at the jump in $H(z)$. That entry measured "
        "**1.0213e-06** on prompt 04's tree and **2.2767e-08** on prompt 05's at "
        "k = 4.972e7/Mpc; this is the same quantity across the whole grid, on prompt 07's."
    )
    emit()
    rows = []
    for unsplit in unsplit_results:
        declared = next(
            r
            for r in results
            if r["sector"] == unsplit["sector"] and r["name"] == unsplit["name"]
        )
        drifts = [row["drift"]["max"] for row in unsplit["rows"]]
        worst_index = max(range(len(drifts)), key=lambda i: drifts[i])
        at_38 = unsplit["rows"][TIMING_K_INDEX]["drift"]["max"]
        rows.append(
            [
                unsplit["sector"],
                unsplit["name"],
                g(max(drifts)),
                f"{unsplit['rows'][worst_index]['k']:.4g}",
                g(median(drifts)),
                str(sum(1 for d in drifts if d > ACCEPTANCE_DRIFT)),
                g(at_38),
                g(
                    declared["rows"][TIMING_K_INDEX]["drift"][
                        BREAK_POINT_DISCONTINUITY
                    ]["max"]
                ),
                f"{sum(row['evaluations'] for row in unsplit['rows']) / len(unsplit['rows']):.0f}",
            ]
        )
    table(
        [
            "sector",
            "model",
            "worst drift, unsplit",
            "at k [1/Mpc]",
            "median drift",
            f"k above {ACCEPTANCE_DRIFT:.1g}",
            "at k = 4.972e+07, unsplit",
            "at k = 4.972e+07, `discontinuity`",
            "evals per object",
        ],
        rows,
    )


def policy_timings(models) -> list:
    """
    Wall-clock cost per object, best of :data:`TIMING_REPEATS`, at the same wavenumber prompt 19
    timed (index 38 = 4.97e7/Mpc).

    **These seconds are not comparable with prompt 19's**: this machine was loaded when they were
    taken and prompt 19's were not. What makes them usable at all is that the two policies are
    timed back to back in the same process, so the *ratio* between them is a same-process
    comparison; and the two smooth models, where the two policies are the same computation, give
    the null control for that ratio.
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
            for policy in POLICIES:
                best = None
                evaluations = None
                for _ in range(TIMING_REPEATS):
                    t0 = time.perf_counter()
                    payload = run_fn(
                        model, k, geo, atol, PRODUCTION_RTOL, break_point_kind=policy
                    )
                    elapsed = time.perf_counter() - t0
                    best = elapsed if best is None else min(best, elapsed)
                    evaluations = payload["data"].RHS_evaluations
                timings[policy] = (best, evaluations)

            out.append(
                {
                    "model": name,
                    "sector": sector,
                    "declares": declares,
                    "k": k,
                    "all": timings[BREAK_POINT_ALL],
                    "jumps": timings[BREAK_POINT_DISCONTINUITY],
                    "load": load_average(),
                }
            )
            log(
                f"   timing {sector} {name}: all {timings[BREAK_POINT_ALL][0]:.4f} s "
                f"({timings[BREAK_POINT_ALL][1]} evals), jumps "
                f"{timings[BREAK_POINT_DISCONTINUITY][0]:.4f} s "
                f"({timings[BREAK_POINT_DISCONTINUITY][1]} evals), load {load_average()}"
            )
    return out


def _worst(result, policy) -> tuple:
    drifts = [row["drift"][policy]["max"] for row in result["rows"]]
    worst_index = max(range(len(drifts)), key=lambda i: drifts[i])
    return (
        max(drifts),
        result["rows"][worst_index]["k"],
        median(drifts),
        sum(1 for d in drifts if d > ACCEPTANCE_DRIFT),
    )


def report_declared_set(qcd_cosmology, qcd) -> None:
    emit("### 1. What the two policies now ask for")
    emit()
    emit(
        "Prompt 07 took `integration_break_points` off the `T(z)` interpolant's knot lattice, so "
        "the two policies no longer differ by four hundred points. On the widest production "
        "geometry of either sector they differ by **one**: `EOS_T_LO = 0.002` GeV, the branch "
        "join at which `w` changes analytic form while `g_s` does not step."
    )
    emit()
    rows = []
    for sector in ("Tk", "Gk"):
        geometry_fn = SECTORS[sector]["geometry"]
        for label, index in (
            ("smallest k", 0),
            ("largest k", len(PRODUCTION_K_GRID) - 1),
        ):
            k = float(PRODUCTION_K_GRID[index])
            geo = geometry_fn(qcd_cosmology, k)
            z_lo, z_hi = float(geo["grid"].min), geo["grid"].max.z
            points = {
                policy: declared_discontinuities_in_z(qcd, z_lo, z_hi, kind=policy)
                for policy in POLICIES
            }
            extra = [
                z
                for z in points[BREAK_POINT_ALL]
                if z not in points[BREAK_POINT_DISCONTINUITY]
            ]
            rows.append(
                [
                    sector,
                    f"{label} = {k:.4g}",
                    f"{z_lo:.4g} -- {z_hi:.4g}",
                    str(len(points[BREAK_POINT_ALL])),
                    str(len(points[BREAK_POINT_DISCONTINUITY])),
                    ", ".join(f"{z:.6g}" for z in extra) if extra else "--",
                ]
            )
    table(
        [
            "sector",
            "wavenumber [1/Mpc]",
            "z range of the run",
            "points, `all`",
            "points, `discontinuity`",
            "in `all` and not in `discontinuity`",
        ],
        rows,
    )


def report_acceptance(results) -> None:
    emit("### 2. The acceptance test, both sectors under both policies")
    emit()
    emit(
        "Reference-convergence drift in the envelope-relative measure of "
        "`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9.4, against the "
        f"{ACCEPTANCE_DRIFT:.1g} criterion of `prompts/GkTk-remedial` prompt 17 §2.1. Prompt 19's "
        "column is §10.1's, measured on the pre-campaign background."
    )
    emit()
    rows = []
    for result in results:
        expected = PROMPT_19_SECTION_10[(result["sector"], result["name"])]
        for policy in POLICIES:
            worst, worst_k, med, offenders = _worst(result, policy)
            if policy == BREAK_POINT_ALL:
                prompt_19 = expected["worst"] if result["sector"] == "Tk" else None
            else:
                prompt_19 = expected.get(
                    "worst_jumps",
                    expected["worst"] if result["sector"] == "Gk" else None,
                )
            if not result["declares"]:
                prompt_19 = expected["worst"]
            rows.append(
                [
                    result["sector"],
                    result["name"],
                    policy,
                    g(prompt_19),
                    g(worst),
                    f"{worst_k:.4g}",
                    g(med),
                    str(offenders),
                    "**yes**" if offenders == 0 else "**NO**",
                ]
            )
    table(
        [
            "sector",
            "model",
            "policy",
            "prompt 19 (§10.1)",
            "worst drift now",
            "at k [1/Mpc]",
            "median drift",
            f"k above {ACCEPTANCE_DRIFT:.1g}",
            "criterion met?",
        ],
        rows,
    )


def report_offenders(results) -> None:
    emit("### 3. The three wavenumbers prompt 19's decision turned on")
    emit()
    emit(
        "Under `BREAK_POINT_DISCONTINUITY`, three of the fifty QCD $T_k$ wavenumbers were above "
        "the criterion when prompt 19 measured them (§10.6, indices 13, 23 and 37). This is what "
        "they read now, on the corrected background."
    )
    emit()
    result = next(r for r in results if r["sector"] == "Tk" and r["name"] == "QCDModel")
    rows = []
    for index, k_19, jumps_19, all_19 in PROMPT_19_TK_OFFENDERS:
        row = result["rows"][index]
        rows.append(
            [
                f"{row['k']:.4g}",
                g(jumps_19, 2),
                g(row["drift"][BREAK_POINT_DISCONTINUITY]["max"], 3),
                g(all_19, 2),
                g(row["drift"][BREAK_POINT_ALL]["max"], 3),
                (
                    "yes"
                    if row["drift"][BREAK_POINT_DISCONTINUITY]["max"]
                    <= ACCEPTANCE_DRIFT
                    else "**no**"
                ),
            ]
        )
    table(
        [
            "k [1/Mpc]",
            "prompt 19, `discontinuity`",
            "now, `discontinuity`",
            "prompt 19, `all`",
            "now, `all`",
            f"now <= {ACCEPTANCE_DRIFT:.1g} under `discontinuity`?",
        ],
        rows,
    )


def report_control(results) -> None:
    emit("### 4. The control: the models that declare nothing")
    emit()
    emit(
        "`RadiationModel` and `LambdaCDMModel` have no equation of state and declare no break "
        "points, so both policies must reach the same single-`solve_ivp` call and must give the "
        "same numbers -- and, since nothing in this campaign may move a LambdaCDM value "
        "(`README` §2 (g)), the same numbers prompt 19 measured. Grid totals are sums of the "
        "per-wavenumber right-hand-side evaluation counts: integers, machine-independent, and the "
        "sharpest available statement that nothing moved."
    )
    emit()
    rows = []
    for result in results:
        if result["declares"]:
            continue
        expected = PROMPT_19_SECTION_10[(result["sector"], result["name"])]
        worst, _, _, _ = _worst(result, BREAK_POINT_ALL)
        totals = {
            policy: sum(row["evaluations"][policy] for row in result["rows"])
            for policy in POLICIES
        }
        identical = all(row["policy_identical"] for row in result["rows"])
        rows.append(
            [
                result["sector"],
                result["name"],
                str(max(row["segments"][BREAK_POINT_ALL] for row in result["rows"])),
                g(expected["worst"]),
                g(worst),
                str(expected["grid_evaluations"]),
                str(totals[BREAK_POINT_ALL]),
                str(totals[BREAK_POINT_DISCONTINUITY]),
                (
                    "**yes**"
                    if identical
                    and totals[BREAK_POINT_ALL] == expected["grid_evaluations"]
                    and totals[BREAK_POINT_DISCONTINUITY]
                    == expected["grid_evaluations"]
                    else "**NO**"
                ),
            ]
        )
    table(
        [
            "sector",
            "model",
            "max segments",
            "prompt 19 worst drift",
            "worst drift now",
            "prompt 19 grid evals",
            "grid evals, `all`",
            "grid evals, `discontinuity`",
            "unchanged and bit-identical?",
        ],
        rows,
    )
    emit(
        "And the module default is still `BREAK_POINT_DISCONTINUITY`: at every wavenumber of "
        "every (model, sector) the production run issued with `break_point_kind` omitted is "
        "bit-identical to the run issued with it named."
    )
    emit()
    table(
        ["sector", "model", "default matches `discontinuity`"],
        [
            [
                result["sector"],
                result["name"],
                (
                    f"all {len(result['rows'])}"
                    if all(row["default_is_discontinuity"] for row in result["rows"])
                    else "**NOT ALL**"
                ),
            ]
            for result in results
        ],
    )


def report_cost(results, timings) -> None:
    emit("### 5. What each policy costs")
    emit()
    emit(
        "Right-hand-side evaluations per object, averaged over the 50-wavenumber grid. **These "
        "counts are the measure** (`prompts/GkTk-remedial/README.md` §5 note 14): they are "
        "integers and do not depend on the machine, which is why this prompt quotes them first "
        "and wall time second."
    )
    emit()
    rows = []
    for result in results:
        expected = PROMPT_19_SECTION_10[(result["sector"], result["name"])]
        totals = {
            policy: sum(row["evaluations"][policy] for row in result["rows"])
            for policy in POLICIES
        }
        n = len(result["rows"])
        rows.append(
            [
                result["sector"],
                result["name"],
                f"{expected.get('grid_evaluations_jumps', expected['grid_evaluations']) / n:.0f}",
                f"{expected['grid_evaluations'] / n:.0f}",
                f"{totals[BREAK_POINT_DISCONTINUITY] / n:.0f}",
                f"{totals[BREAK_POINT_ALL] / n:.0f}",
                f"{totals[BREAK_POINT_ALL] / totals[BREAK_POINT_DISCONTINUITY] - 1.0:+.2%}",
            ]
        )
    table(
        [
            "sector",
            "model",
            "prompt 19, `discontinuity`",
            "prompt 19, `all`",
            "now, `discontinuity`",
            "now, `all`",
            "`all` over `discontinuity`",
        ],
        rows,
    )
    emit(
        f"Wall time per object, best of {TIMING_REPEATS} single-core runs at "
        f"k = {float(PRODUCTION_K_GRID[TIMING_K_INDEX]):.4g}/Mpc and the production tolerances, "
        "with the one-, five- and fifteen-minute load averages at which each pair was taken. "
        "Absolute seconds are a property of the machine and its load, so they are **not** "
        "comparable with prompt 19's, which were taken in a different session. What is comparable "
        "is the *ratio*: the two policies are timed back to back in the same process, and the "
        "four smooth-model rows -- where the two policies are literally the same computation -- "
        "are the null control that says how far from 1.000 a ratio has to be before it means "
        "anything."
    )
    emit()
    table(
        [
            "sector",
            "model",
            "s per object, `all`",
            "s per object, `discontinuity`",
            "ratio",
            "evals, `all` / `discontinuity`",
            "load average",
        ],
        [
            [
                row["sector"],
                row["model"],
                f"{row['all'][0]:.4f}",
                f"{row['jumps'][0]:.4f}",
                f"{row['all'][0] / row['jumps'][0]:.3f}",
                f"{row['all'][1]} / {row['jumps'][1]}",
                row["load"],
            ]
            for row in timings
        ],
    )


def report_shift(results) -> None:
    emit("### 6. How far the answer itself moves between the policies")
    emit()
    emit(
        "The production-tolerance run under `BREAK_POINT_DISCONTINUITY` scored against the "
        "production-tolerance run under `BREAK_POINT_ALL`, in the same envelope-relative measure "
        "-- i.e. what a stored object would move by if the policy were changed. On the models "
        "that declare nothing it is identically zero, which is the same statement as §4's "
        "bit-identity."
    )
    emit()
    rows = []
    for result in results:
        shifts = [row["shift"] for row in result["rows"]]
        worst_index = max(range(len(shifts)), key=lambda i: shifts[i])
        rows.append(
            [
                result["sector"],
                result["name"],
                g(max(shifts)),
                f"{result['rows'][worst_index]['k']:.4g}",
                g(median(shifts)),
            ]
        )
    table(
        ["sector", "model", "worst shift", "at k [1/Mpc]", "median shift"],
        rows,
    )


def report_detail(result) -> None:
    emit(f"#### {result['sector']}, {result['name']}")
    emit()
    rows = []
    for row in result["rows"]:
        rows.append(
            [
                f"{row['k']:.4g}",
                f"{row['segments'][BREAK_POINT_ALL]} / "
                f"{row['segments'][BREAK_POINT_DISCONTINUITY]}",
                g(row["drift"][BREAK_POINT_ALL]["max"], 2),
                g(row["drift"][BREAK_POINT_DISCONTINUITY]["max"], 2),
                (
                    "yes"
                    if row["drift"][BREAK_POINT_DISCONTINUITY]["max"]
                    <= ACCEPTANCE_DRIFT
                    else "**no**"
                ),
                g(row["shift"], 2),
                f"{row['evaluations'][BREAK_POINT_ALL]} / "
                f"{row['evaluations'][BREAK_POINT_DISCONTINUITY]}",
            ]
        )
    table(
        [
            "k [1/Mpc]",
            "segments, `all` / `disc`",
            "drift, `all`",
            "drift, `disc`",
            f"`disc` <= {ACCEPTANCE_DRIFT:.1g}?",
            "shift between policies",
            "evals, `all` / `disc`",
        ],
        rows,
    )


def main_remeasure() -> None:
    t_start = time.perf_counter()
    load_at_start = load_average()

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
    control = {policy: reproduce_control(radiation, policy) for policy in POLICIES}
    for label in control[BREAK_POINT_ALL]:
        log(
            f"   {label}: all {control[BREAK_POINT_ALL][label]['max']:.4g} "
            f"({control[BREAK_POINT_ALL][label]['evaluations']} evals), discontinuity "
            f"{control[BREAK_POINT_DISCONTINUITY][label]['max']:.4g} "
            f"({control[BREAK_POINT_DISCONTINUITY][label]['evaluations']} evals)"
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
            results.append(
                both_policies_sweep(name, model, cosmology, sector, declares)
            )

    log("** the third column: the declaration suppressed altogether")
    unsplit_results = [
        unsplit_sweep("QCDModel", qcd, qcd_cosmology, sector) for sector in ("Tk", "Gk")
    ]

    log("** wall time per object")
    timings = policy_timings(models)

    elapsed = time.perf_counter() - t_start

    emit("# The per-sector break-point policy, re-measured on the corrected background")
    emit()
    emit(f"<!-- generated {date.today().isoformat()} by")
    emit(
        "     PYTHONPATH=. ./venv/bin/python "
        "docs/qcd-background-audit/per_sector_policy_remeasure.py"
    )
    emit(
        f"     in {elapsed:.0f} s; Python {platform.python_version()}, "
        f"NumPy {np.__version__}, SciPy {scipy.__version__};"
    )
    emit(
        f"     load average {load_at_start} at the start, {load_average()} at the end -->"
    )
    emit()
    report_declared_set(qcd_cosmology, qcd)
    report_acceptance(results)
    report_unsplit(unsplit_results, results)
    report_offenders(results)
    report_control(results)
    report_cost(results, timings)
    report_shift(results)
    emit("### 7. Prompt 17's two control figures, under both policies")
    emit()
    table(
        [
            "control",
            "prompt 12",
            "`all`",
            "evals",
            "`discontinuity`",
            "evals",
            "identical?",
        ],
        [
            [
                label,
                g(control[BREAK_POINT_ALL][label]["expected"]),
                g(control[BREAK_POINT_ALL][label]["max"]),
                str(control[BREAK_POINT_ALL][label]["evaluations"]),
                g(control[BREAK_POINT_DISCONTINUITY][label]["max"]),
                str(control[BREAK_POINT_DISCONTINUITY][label]["evaluations"]),
                (
                    "yes"
                    if control[BREAK_POINT_ALL][label]["max"]
                    == control[BREAK_POINT_DISCONTINUITY][label]["max"]
                    and control[BREAK_POINT_ALL][label]["evaluations"]
                    == control[BREAK_POINT_DISCONTINUITY][label]["evaluations"]
                    else "**no**"
                ),
            ]
            for label in control[BREAK_POINT_ALL]
        ],
    )
    emit("### 8. Every wavenumber")
    emit()
    for result in results:
        report_detail(result)

    emit(
        f"*Runtime {elapsed:.0f} s (QCD stand-in build {qcd_build_seconds:.1f} s of it); "
        f"{len(PRODUCTION_K_GRID)} wavenumbers x 3 models x 2 sectors x 2 policies.*"
    )
    log(f"** total runtime {elapsed:.1f} s")


if __name__ == "__main__":
    main_remeasure()
