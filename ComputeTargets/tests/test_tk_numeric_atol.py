"""
Tests for prompt 12 of ``prompts/GkTk-remedial``: the absolute tolerance of the *numeric* part of
the matter transfer function.

Review §12.5 of ``docs/gk-wkb-review-fable-2026-09-09.md`` measured the transfer function's
numeric region at 1.1e-5 of the Liouville-Green envelope -- fifty times worse than the Green's
function's 2e-7 (§10.1) -- and identified the cause as the *absolute* tolerance rather than the
solver: ``T`` decays as ``3/x^2``, so ``|T| ~ 1e-5`` deep inside the horizon and the shared
``atol = 1e-10`` is a 1e-5 relative tolerance there. ``config.defaults`` therefore gains
``DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`` for this run alone, and ``main.py`` builds a separate
``tolerance`` object for it (the plumbing is guarded structurally in
``ComputeTargets/tests/test_main_plumbing.py``, because the tolerance is part of the datastore
lookup key).

The geometry here is review §12.5's: exact radiation, ``T = 1, T' = 0`` five e-folds outside the
horizon, the production source grid (100 samples per decade of z) truncated at ``0.85 z_e6``, the
``(z_e3, z_e6)`` stop window, and ``rtol = 1e-8``. Nothing needs Ray or a datastore:
``numeric_with_phase_cut`` is called through its undecorated ``_function`` with the stand-ins of
``ComputeTargets/tests/wkb_reference.py`` (prompt 01).

**The wavenumber is k = 1e6/Mpc because that is the review's own geometry.** The review does not
say which k it used, but at k = 1e6 all four right-hand-side evaluation counts of its §12.5 table
are reproduced exactly -- 6476 / 6401 / 7379 / 7829 -- so the numbers asserted below are
comparable with the review's line by line:

===========================  ================  ==========  ===========  ===========
initial data                 (atol, rtol)      RHS evals   review dT    measured dT
===========================  ================  ==========  ===========  ===========
T=1, T'=0 (production)       (1e-10, 1e-8)     6476        1.1e-5       9.93e-6
exact                        (1e-10, 1e-8)     6401        1.1e-5       1.16e-5
exact                        (1e-13, 1e-8)     7379        3.6e-7       3.28e-7
exact                        (1e-16, 1e-8)     7829        1.5e-7       1.34e-7
T=1, T'=0 (production)       (1e-13, 1e-8)     7403        --           2.53e-6
===========================  ================  ==========  ===========  ===========

The last row is the shipped configuration, and its 2.5e-6 is the super-horizon initial condition,
not the solver: with exact initial data the same run gives 3.3e-7. Removing that floor means
replacing ``T = 1, T' = 0`` with the series ``T ~ 1 - x^2/10``, which is a specification decision
and is out of scope here (``[00-tk-superhorizon-ic-series]``, README §0.4).
"""

import unittest
from math import cos, fabs, hypot, sin, sqrt

from ComputeTargets.GkNumericIntegration import RHS as Gk_RHS
from ComputeTargets.TkNumericIntegration import RHS as Tk_RHS
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.tests.wkb_reference import (
    RadiationModel,
    envelope_relative_error,
    horizon_exit_z,
    production_source_grid,
)
from Quadrature.integrators.numeric_with_phase_cut import numeric_with_phase_cut
from Units import Mpc_units
from config.defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_TK_NUMERIC_ABS_TOLERANCE

UNITS = Mpc_units()

# review §12.5's geometry (see the module docstring for why this k)
K_INV_MPC = 1.0e6
EFOLDS_SUPH = 5.0
PRODUCTION_ATOL = 1e-10
PRODUCTION_RTOL = 1e-8

# main.py:630, :1199 -- delta_logz is a spacing in log10(1+z); it no longer enters the
# oscillation-resolution test (prompt 11) but is still part of the call
PRODUCTION_DELTA_LOGZ = 1.0 / 100.0

# prompt 12 §2's acceptance thresholds
BASELINE_ERROR_RANGE = (9.0e-6, 1.3e-5)
TIGHT_ERROR_TARGET = 3.0e-6
TIGHT_ERROR_EXACT_IC_TARGET = 5.0e-7
MAX_EVALUATION_INCREASE = 0.25
GK_INDIFFERENCE_TARGET = 1.0e-9


# ---------------------------------------------------------------------------------------------
# stand-ins (the pattern of ComputeTargets/tests/test_numeric_phase_cut.py)
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
# the exact radiation transfer function, and the geometry the runs are made on
# ---------------------------------------------------------------------------------------------

MODEL = RadiationModel()

# x = k c_s tau = A/(1+z) with H0 = 1 (wkb_reference.RadiationModel)
RADIATION_A = K_INV_MPC / sqrt(3.0)


def x_of_z(z: float) -> float:
    return RADIATION_A / (1.0 + z)


def T_exact(x: float) -> float:
    """The exact radiation transfer function ``T = 3(sin x - x cos x)/x^3`` (review §12.4)."""
    return 3.0 * (sin(x) - x * cos(x)) / (x * x * x)


def dT_dx(x: float) -> float:
    """``d/dx`` of ``T_exact``; ``d[sin x - x cos x]/dx = x sin x``."""
    return 3.0 * (x * x * sin(x) - 3.0 * sin(x) + 3.0 * x * cos(x)) / (x**4)


def T_envelope(x: float) -> float:
    """The Liouville-Green envelope of the exact ``T``, ``3 sqrt(1+x^2)/x^3`` (review §12.4)."""
    return 3.0 * sqrt(1.0 + x * x) / (x * x * x)


def exact_initial_data(z: float):
    """Exact ``(T, dT/dz)`` at ``z``; ``dx/dz = -A/(1+z)^2 = -x^2/A``."""
    x = x_of_z(z)
    return T_exact(x), dT_dx(x) * (-(x * x) / RADIATION_A)


def geometry():
    """
    The production geometry of the ``TkNumericIntegration`` work item (main.py's
    ``build_Tk_numeric_work``): the source grid from five e-folds outside the horizon, truncated
    below at ``0.85 z_e6``, with the ``(z_e3, z_e6)`` stop window.
    """
    z_exit = horizon_exit_z(MODEL, K_INV_MPC, 0.0)
    z_e3 = horizon_exit_z(MODEL, K_INV_MPC, 3.0)
    z_e6 = horizon_exit_z(MODEL, K_INV_MPC, 6.0)
    z_source = horizon_exit_z(MODEL, K_INV_MPC, -EFOLDS_SUPH)

    grid = production_source_grid(z_source).truncate(0.85 * z_e6, keep="higher-include")
    return {"z_exit": z_exit, "z_e3": z_e3, "z_e6": z_e6, "grid": grid}


GEOMETRY = geometry()


def run(sector: str, atol: float, initial_data=None) -> dict:
    """Call the undecorated ``numeric_with_phase_cut`` for one sector at one absolute tolerance.

    ``warn_unresolved_osc=False`` matches what the two production integrators now pass (prompt
    16); it gates only the printed warning, never the returned values.
    """
    is_Gk = sector == "Gk"
    grid = GEOMETRY["grid"]
    z_init = grid.max

    if initial_data is None:
        value, deriv = (0.0, 1.0) if is_Gk else (1.0, 0.0)
    else:
        value, deriv = initial_data

    return numeric_with_phase_cut._function(
        _Proxy(MODEL, UNITS),
        _KExit(K_INV_MPC, UNITS, GEOMETRY["z_exit"]),
        z_init,
        grid,
        initial_value=value,
        initial_deriv=deriv,
        RHS=Gk_RHS if is_Gk else Tk_RHS,
        omega_sq=Gk_omegaEff_sq if is_Gk else Tk_omegaEff_sq,
        atol=atol,
        rtol=PRODUCTION_RTOL,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        mode="stop",
        stop_search_window_z_begin=min(GEOMETRY["z_e3"], z_init.z),
        stop_search_window_z_end=GEOMETRY["z_e6"],
        task_label=f"test_tk_numeric_atol_{sector}",
        object_label="Tk(z)" if not is_Gk else "Gr_k(z, z')",
        warn_unresolved_osc=False,
    )


_CACHE = {}


def cached_run(sector: str, atol: float, exact_ic: bool = False) -> dict:
    """Runs are shared between tests: each (sector, atol, initial data) is solved once."""
    key = (sector, atol, exact_ic)
    if key not in _CACHE:
        initial_data = exact_initial_data(GEOMETRY["grid"].max.z) if exact_ic else None
        _CACHE[key] = run(sector, atol, initial_data=initial_data)
    return _CACHE[key]


def max_T_error(payload: dict):
    """``max |T - T_exact| / envelope`` over the returned samples, with the ``x`` where it fell.

    In ``"stop"`` mode the ODE terminates on the ``z_e6`` event and the trailing requested
    samples are never produced, so the returned list is shorter than the grid; the zip is over
    what came back.
    """
    worst = 0.0
    worst_x = None
    for z, value in zip(GEOMETRY["grid"], payload["value_sample"]):
        x = x_of_z(z.z)
        error = envelope_relative_error(value, T_exact(x), T_envelope(x))
        if error > worst:
            worst, worst_x = error, x
    return worst, worst_x


def max_G_difference(payload_a: dict, payload_b: dict):
    """``max |G_a - G_b| / envelope``, the envelope being the local Liouville-Green
    ``hypot(G, G'/omega)`` of the second run (the same measure review §10.1 uses)."""
    worst = 0.0
    worst_z = None
    for z, value_a, value_b, deriv_b in zip(
        GEOMETRY["grid"],
        payload_a["value_sample"],
        payload_b["value_sample"],
        payload_b["deriv_sample"],
    ):
        omega = sqrt(Gk_omegaEff_sq(MODEL, K_INV_MPC, z.z))
        envelope = hypot(value_b, deriv_b / omega)
        error = fabs(value_a - value_b) / envelope
        if error > worst:
            worst, worst_z = error, z.z
    return worst, worst_z


class TkNumericToleranceConstantTestCase(unittest.TestCase):
    def test_the_transfer_function_tolerance_is_tighter_than_the_shared_one(self):
        self.assertEqual(DEFAULT_TK_NUMERIC_ABS_TOLERANCE, 1e-13)
        self.assertLess(DEFAULT_TK_NUMERIC_ABS_TOLERANCE, DEFAULT_ABS_TOLERANCE)


class TkNumericToleranceAccuracyTestCase(unittest.TestCase):
    """Review §12.5 measured, reproduced, and improved."""

    def test_the_shared_tolerance_reproduces_the_review_baseline(self):
        """At ``atol = 1e-10`` the numeric transfer function is ~1e-5 of the envelope."""
        error, x = max_T_error(cached_run("Tk", PRODUCTION_ATOL))

        self.assertGreaterEqual(
            error,
            BASELINE_ERROR_RANGE[0],
            msg=f"baseline error {error:.4g} at x={x:.6g} is better than review §12.5's 1.1e-5",
        )
        self.assertLessEqual(
            error,
            BASELINE_ERROR_RANGE[1],
            msg=f"baseline error {error:.4g} at x={x:.6g} is worse than review §12.5's 1.1e-5",
        )

    def test_the_new_tolerance_meets_the_campaign_target(self):
        """README §6's row: 1.1e-5 now, target <= 3e-6, floor 2.5e-6 (the initial condition)."""
        error, x = max_T_error(cached_run("Tk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE))

        self.assertLessEqual(
            error,
            TIGHT_ERROR_TARGET,
            msg=f"error {error:.4g} at x={x:.6g} misses the <= 3e-6 target",
        )

    def test_with_exact_initial_data_the_solver_error_alone_is_far_smaller(self):
        """The 2.5e-6 that remains is the ``T = 1, T' = 0`` initial condition, not the solver:
        exact initial data at the same tolerance gives ~3e-7 (README §2 (d))."""
        error, x = max_T_error(
            cached_run("Tk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE, exact_ic=True)
        )

        self.assertLessEqual(
            error,
            TIGHT_ERROR_EXACT_IC_TARGET,
            msg=f"error {error:.4g} at x={x:.6g} misses the <= 5e-7 target",
        )

        # and the improvement over the production initial condition is real, not noise
        production, _ = max_T_error(cached_run("Tk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE))
        self.assertLess(error, production / 4.0)

    def test_the_initial_condition_floor_is_where_the_review_says(self):
        """``T = 1`` at ``x_i`` is wrong by ``x_i^2/10`` of the envelope; review §12.5 calls the
        resulting floor 2.5e-6 and prompt 12 leaves it in place."""
        error, _ = max_T_error(cached_run("Tk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE))

        self.assertGreater(error, 1.0e-6)
        self.assertLess(error, TIGHT_ERROR_TARGET)


class TkNumericToleranceCostTestCase(unittest.TestCase):
    def test_the_tighter_tolerance_costs_at_most_a_quarter_more_evaluations(self):
        """Review §12.5: "a factor 30 for 15 % more evaluations"."""
        baseline = cached_run("Tk", PRODUCTION_ATOL)["data"].RHS_evaluations
        tightened = cached_run("Tk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE)[
            "data"
        ].RHS_evaluations

        increase = float(tightened) / float(baseline) - 1.0
        self.assertLessEqual(
            increase,
            MAX_EVALUATION_INCREASE,
            msg=f"RHS evaluations {baseline} -> {tightened} ({increase:+.1%})",
        )


class GkNumericToleranceIsIndifferentTestCase(unittest.TestCase):
    """The other half of the decision: nothing is gained by tightening ``atol`` for ``G_k``, so
    ``GkNumericIntegration`` keeps the shared tolerance and no datastore key moves for it.
    """

    def test_the_green_function_is_unchanged_by_the_tighter_tolerance(self):
        loose = cached_run("Gk", PRODUCTION_ATOL)
        tight = cached_run("Gk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE)

        error, z = max_G_difference(loose, tight)
        self.assertLessEqual(
            error,
            GK_INDIFFERENCE_TARGET,
            msg=f"G moved by {error:.4g} of the envelope at z={z:.6g} when atol was tightened",
        )

    def test_the_green_function_does_not_pay_for_it_either(self):
        """``atol`` never binds for ``G``, so the cost of tightening it is nil -- which is why the
        review's recommendation is for the transfer function alone."""
        loose = cached_run("Gk", PRODUCTION_ATOL)["data"].RHS_evaluations
        tight = cached_run("Gk", DEFAULT_TK_NUMERIC_ABS_TOLERANCE)[
            "data"
        ].RHS_evaluations

        self.assertLess(fabs(float(tight) / float(loose) - 1.0), 0.05)


if __name__ == "__main__":
    unittest.main()
