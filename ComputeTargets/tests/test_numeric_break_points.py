"""
Tests for prompt 18 of ``prompts/GkTk-remedial``: the numeric ODE is split at the cosmology's
*declared discontinuities*, so that no DOP853 step straddles a jump in the right-hand side.

Three things are tested, in the order the prompt states them:

1. **The declaration.** ``GenericEOSBase.discontinuity_temperatures_GeV`` defaults to "smooth";
   ``QCD_EOS`` overrides it with the subset of its break temperatures at which G, Gs or w really
   do jump, and that subset is *measured* here rather than taken from the docstring;
   ``integration_break_points(..., kind=)`` returns the spline knots for the quadrature path and
   only the temperature crossings for the ODE path.
2. **The unsplit path is untouched.** A cosmology declaring no discontinuities produces exactly
   the samples a single ``solve_ivp`` call with the same arguments produces -- bit for bit, not
   nearly.
3. **The split path is correct.** A synthetic stand-in whose oscillation frequency jumps at a
   declared redshift has a closed-form solution; the split run reproduces it, converges monotonely
   under refinement, returns exactly the sample grid requested, aggregates the supervisor's
   accounting over segments, and still finds its ``mode="stop"`` extremum when the stop event lies
   in an interior segment and the search window straddles a segment boundary.
4. **The split is more accurate, on the cosmology that motivated it.**
   :class:`TestQCDReferenceConvergence` runs the campaign's reference-convergence test on
   ``QCD_Cosmology`` at k = 4.97e7/Mpc, split and unsplit, and requires the split to converge
   where the unsplit does not.

Prompt 19 then made *which kind* of declared break point is split at a per-caller choice, and adds
a fifth group:

5. **The policy is the caller's.** ``numeric_with_phase_cut`` takes a ``break_point_kind``;
   :class:`TestPerSectorPolicy` checks that it selects the segmentation on a stand-in declaring
   both a jump and a kink, that its default is bit-for-bit today's jumps-only behaviour, that a
   cosmology declaring nothing takes the single-call path under either policy, and -- read with
   ``ast``, as prompt 16's call-site test is -- that each production integrator passes the kind its
   sector decided on. :class:`TestManyBreakPoints` then exercises what ~400 boundaries expose that
   one did not: segments holding no requested sample, boundaries closer to one another than
   ``BREAK_POINT_STANDOFF``, and ``mode="stop"`` with the event far down a long chain of segments.

**Why item 4 is not done on the synthetic fixture.** It was tried. Across frequency ratios from
1.0001 to 10, grids from 20 to 800 points and tolerances from (1e-10, 1e-8) to (1e-16, 1e-13), the
split and unsplit runs of the synthetic oscillator agree to within a factor of two and sometimes
the unsplit one is better: DOP853's step controller *detects* a jump of that size, rejects the
straddling step and grinds the step down until it is resolved, which costs steps rather than
accuracy. The production failure needs a jump small enough to slip past the controller -- 1.04e-4
relative in H, on QCD_Cosmology -- and large enough to matter once it has, which is a combination
the synthetic could not be tuned into. Claiming the improvement on a fixture that does not show it
would be worse than measuring it where it is real.

Nothing here needs Ray or a datastore: ``numeric_with_phase_cut`` is exercised through its
undecorated ``_function`` with the stand-in pattern of ``ComputeTargets/tests/wkb_reference.py``
(prompt 01) and of ``ComputeTargets/tests/test_numeric_phase_cut.py`` (prompt 11).
"""

import ast
import unittest
from math import cos, expm1, fabs, hypot, log1p, log10, sin, sqrt
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from ComputeTargets.BackgroundModel import (
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
    ModelFunctions,
    _cosmology_break_points,
)
from ComputeTargets.GkNumericIntegration import RHS as Gk_RHS, GkNumericIntegration
from ComputeTargets.TkNumericIntegration import RHS as Tk_RHS, TkNumericIntegration
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.tests.wkb_reference import (
    LambdaCDMModel,
    QCDModel,
    RadiationModel,
    envelope_relative_error,
    horizon_exit_z,
    production_response_grid,
    production_source_grid,
    to_redshift_array,
)
from CosmologyModels.GenericEOS.GenericEOS import GenericEOSBase
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS
from CosmologyModels.LambdaCDM import Planck2018
from Quadrature.integrators.numeric_with_phase_cut import (
    BREAK_POINT_STANDOFF,
    DERIV_INDEX,
    VALUE_INDEX,
    _solve_segmented,
    declared_discontinuities_in_z,
    numeric_with_phase_cut,
)
from Quadrature.supervisors.base import RHS_timer
from Quadrature.supervisors.numeric import NumericIntegrationSupervisor
from Units import Mpc_units

UNITS = Mpc_units()

PRODUCTION_ATOL = 1e-10
PRODUCTION_RTOL = 1e-8
PRODUCTION_DELTA_LOGZ = 1.0 / 100.0

# separation, relative in T, at which the jump across each declared break temperature is measured.
# Small enough that a continuous function moves by ~1e-14 and a genuine step does not move at all.
JUMP_PROBE = 1e-12

# what "a jump" means for the purposes of the QCD_EOS declaration test: a relative change in G, Gs
# or w that does not shrink when the probe separation shrinks. 1e-6 separates the measured jumps
# (2e-4 and larger) from the measured continuities (2e-14 and smaller) by eight orders.
JUMP_THRESHOLD = 1e-6


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
# a synthetic cosmology whose right-hand side jumps, with a closed-form solution
#
# The system is y'' = -omega(z)^2 y with omega piecewise constant, integrated downwards in z. The
# solution is a sinusoid of one frequency above the declared redshift and of another below it,
# matched by continuity of (y, y') -- so an exact answer is available on both sides of a genuine
# discontinuity in the right-hand side, which is what the split has to be scored against.
# ---------------------------------------------------------------------------------------------


class _JumpCosmology:
    """
    Declares one *jump* at ``z_jump`` and, when ``z_kink`` is given, one further break that is
    **not** a jump. The ODE path must split at the first and ignore the second; the quadrature
    path (which does not run here) would split at both.

    ``declare`` is the switch the accuracy comparison turns off: with ``declare=False`` the
    cosmology reports nothing at all, which is the pre-prompt-18 behaviour and the "before"
    column of the comparison.
    """

    def __init__(self, z_jump: float, z_kink=None, declare: bool = True):
        self.z_jump = float(z_jump)
        self.z_kink = None if z_kink is None else float(z_kink)
        self.declare = bool(declare)

    def integration_break_points(
        self, z_lo: float, z_hi: float, kind: str = BREAK_POINT_ALL
    ) -> np.ndarray:
        if not self.declare:
            return np.empty(0, dtype=float)

        points = [self.z_jump]
        if kind == BREAK_POINT_ALL and self.z_kink is not None:
            points.append(self.z_kink)

        inside = [log1p(z) for z in points if z_lo < z < z_hi]
        if len(inside) == 0:
            return np.empty(0, dtype=float)
        return np.unique(np.asarray(inside, dtype=float))


class _TwoJumpCosmology:
    """Declares two jumps, so that the range is cut into three segments."""

    def __init__(self, z_upper: float, z_lower: float):
        self.z_upper = float(z_upper)
        self.z_lower = float(z_lower)

    def integration_break_points(
        self, z_lo: float, z_hi: float, kind: str = BREAK_POINT_ALL
    ) -> np.ndarray:
        inside = [log1p(z) for z in (self.z_upper, self.z_lower) if z_lo < z < z_hi]
        if len(inside) == 0:
            return np.empty(0, dtype=float)
        return np.unique(np.asarray(inside, dtype=float))


class _ManyBreakCosmology:
    """
    Declares one jump and ``count`` further break points that are **not** jumps, spread uniformly
    in ``u = log(1+z)`` across the range, plus -- when ``coincident`` is set -- a pair of extra
    kinks placed closer to an existing one than ``BREAK_POINT_STANDOFF``.

    This is what a caller asking for ``BREAK_POINT_ALL`` on ``QCD_Cosmology`` gets: ~125 declared
    points inside one production numeric range against ~100 requested samples, so most segments
    carry no output point at all. The underlying right-hand side is smooth at every one of the
    kinks, which is the point -- splitting there must change the answer by no more than the
    integration tolerance, and must not disturb the returned grid.
    """

    def __init__(
        self, z_jump: float, z_lo: float, z_hi: float, count: int, coincident=()
    ):
        self.z_jump = float(z_jump)
        u = np.linspace(log1p(float(z_lo)), log1p(float(z_hi)), count + 2)[1:-1]
        self.z_kinks = [float(expm1(value)) for value in u]
        self.z_coincident = [float(z) for z in coincident]

    def integration_break_points(
        self, z_lo: float, z_hi: float, kind: str = BREAK_POINT_ALL
    ) -> np.ndarray:
        points = [self.z_jump]
        if kind == BREAK_POINT_ALL:
            points.extend(self.z_kinks)
            points.extend(self.z_coincident)

        inside = [log1p(z) for z in points if z_lo < z < z_hi]
        if len(inside) == 0:
            return np.empty(0, dtype=float)
        return np.unique(np.asarray(inside, dtype=float))


class _JumpModel:
    """
    A ``BackgroundModel`` stand-in for the piecewise-frequency oscillator: ``.cosmology`` carries
    the declaration, ``.functions`` supplies the one accessor the post-solve diagnostic needs, and
    ``.omega(z)`` is the discontinuous coefficient of the right-hand side.
    """

    name = "JumpModel"

    def __init__(
        self,
        z_jump: float,
        omega_hi: float,
        omega_lo: float,
        z_kink=None,
        declare: bool = True,
    ):
        self.z_jump = float(z_jump)
        self.omega_hi = float(omega_hi)
        self.omega_lo = float(omega_lo)
        self.cosmology = _JumpCosmology(z_jump, z_kink=z_kink, declare=declare)
        self.functions = ModelFunctions(
            Hubble=lambda z: 1.0,
            epsilon=lambda z: 0.0,
            d_epsilon_dz=lambda z: 0.0,
            d2_epsilon_dz2=lambda z: 0.0,
            wBackground=lambda z: 1.0 / 3.0,
            wPerturbations=lambda z: 1.0 / 3.0,
            tau=None,
            T_photon=lambda z: 0.0,
            d_lnH_dz=lambda z: 0.0,
            d2_lnH_dz2=lambda z: 0.0,
            d3_lnH_dz3=lambda z: 0.0,
            d_wPerturbations_dz=lambda z: 0.0,
            d2_wPerturbations_dz2=lambda z: 0.0,
        )

    def omega(self, z: float) -> float:
        return self.omega_hi if z > self.z_jump else self.omega_lo

    def exact(self, z: float, z_init: float, y0: float, y0p: float):
        """
        ``(y, y')`` at ``z``, integrating downwards from ``(y0, y0p)`` at ``z_init``.

        Above the jump the frequency is ``omega_hi``; the state is propagated analytically to
        ``z_jump`` and continued below it with ``omega_lo``. Both the value and the derivative are
        continuous there -- it is the *second* derivative that jumps, which is precisely what
        breaks the embedded error estimator of a step that straddles the point.
        """

        def propagate(state, z_from, z_to, omega):
            phase = omega * (z_to - z_from)
            y, yp = state
            return (
                y * cos(phase) + (yp / omega) * sin(phase),
                -y * omega * sin(phase) + yp * cos(phase),
            )

        state = (float(y0), float(y0p))
        if z >= self.z_jump:
            return propagate(state, z_init, z, self.omega_hi)

        state = propagate(state, z_init, self.z_jump, self.omega_hi)
        return propagate(state, self.z_jump, z, self.omega_lo)


class _SmoothCosmology:
    """A cosmology that declares nothing, wrapping one that does."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        if name == "integration_break_points":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_inner"), name)


class _UnsplitView:
    """
    The same model with the same ``functions`` -- identical physics -- but a cosmology that
    declares no discontinuities, so that ``numeric_with_phase_cut`` takes its pre-prompt-18
    single-``solve_ivp`` path. This is how the "before" column is measured after the change,
    without monkeypatching anything.
    """

    def __init__(self, model, cosmology):
        self.name = f"{model.name} (unsplit)"
        self.functions = model.functions
        self.cosmology = _SmoothCosmology(cosmology)


def _jump_RHS(z, state, model, k_float, supervisor):
    """``y'' = -omega(z)^2 y``, with the supervisor timing every evaluation as the real ones do."""
    with RHS_timer(supervisor):
        omega = model.omega(z)
        return [state[DERIV_INDEX], -omega * omega * state[VALUE_INDEX]]


def _jump_omega_sq(model, k_float, z):
    omega = model.omega(z)
    return omega * omega


# ---------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------

# the synthetic problem: a grid of 200 points between z = 2 and z = 1, a frequency of 400 above
# z = 1.5 and 260 below it. That is ~32 cycles above the jump and ~13 below, so a step sequence
# chosen for the upper piece is badly wrong for the lower one and the jump is not a small
# perturbation of it.
JUMP_Z_INIT = 2.0
JUMP_Z_END = 1.0
JUMP_Z = 1.5
JUMP_Z_KINK = 1.25
JUMP_OMEGA_HI = 400.0
JUMP_OMEGA_LO = 260.0
JUMP_SAMPLES = 200
JUMP_INITIAL_VALUE = 0.0
JUMP_INITIAL_DERIV = 1.0


def _jump_grid(num: int = JUMP_SAMPLES):
    return to_redshift_array(np.linspace(JUMP_Z_INIT, JUMP_Z_END, num))


def _run_jump(model, grid, atol: float, rtol: float, **kwargs):
    z_init = grid.max
    return numeric_with_phase_cut._function(
        _Proxy(model, UNITS),
        _KExit(1.0, UNITS, 1.0),
        z_init,
        grid,
        initial_value=JUMP_INITIAL_VALUE,
        initial_deriv=JUMP_INITIAL_DERIV,
        RHS=_jump_RHS,
        omega_sq=_jump_omega_sq,
        atol=atol,
        rtol=rtol,
        delta_logz=PRODUCTION_DELTA_LOGZ,
        task_label="test_numeric_break_points",
        object_label="y(z)",
        warn_unresolved_osc=False,
        **kwargs,
    )


def _jump_error(model, grid, payload) -> float:
    """Worst envelope-relative departure of a run from the closed-form solution."""
    worst = 0.0
    for z, value in zip(grid, payload["value_sample"]):
        y_exact, yp_exact = model.exact(
            z.z, JUMP_Z_INIT, JUMP_INITIAL_VALUE, JUMP_INITIAL_DERIV
        )
        envelope = hypot(y_exact, yp_exact / model.omega(z.z))
        worst = max(worst, envelope_relative_error(value, y_exact, envelope))
    return worst


# ---------------------------------------------------------------------------------------------
# 1. the declaration
# ---------------------------------------------------------------------------------------------


class TestDeclaration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units = UNITS
        cls.eos = QCD_EOS(cls.units)
        cls.cosmology = QCD_Cosmology(
            store_id=0, units=cls.units, params=Planck2018(), max_z=1e20
        )

    def test_generic_eos_default_is_smooth(self):
        """An equation of state that has never heard of this campaign declares nothing."""

        class _Smooth(GenericEOSBase):
            @property
            def name(self):
                return "smooth"

            @property
            def type_id(self) -> int:
                return -1

            def G(self, T: float) -> float:
                return 1.0

            def Gs(self, T: float) -> float:
                return 1.0

        eos = _Smooth(self.units)
        self.assertEqual(eos.break_temperatures_GeV, ())
        self.assertEqual(eos.discontinuity_temperatures_GeV, ())

    def test_qcd_discontinuities_are_a_subset_of_the_breaks(self):
        breaks = set(self.eos.break_temperatures_GeV)
        jumps = set(self.eos.discontinuity_temperatures_GeV)
        self.assertTrue(jumps.issubset(breaks))
        self.assertLess(len(jumps), len(breaks))

    def test_qcd_declares_exactly_the_temperatures_at_which_it_jumps(self):
        """
        Measured, not transcribed: evaluate G, Gs and w either side of each declared break
        temperature and check that the ones declared as discontinuities are the ones that step.
        """
        GeV = self.units.GeV
        measured_jumps = set()
        for T_in_GeV in self.eos.break_temperatures_GeV:
            T_minus = T_in_GeV * (1.0 - JUMP_PROBE) * GeV
            T_plus = T_in_GeV * (1.0 + JUMP_PROBE) * GeV

            steps = [
                fabs(f(T_plus) - f(T_minus)) / fabs(f(T_minus))
                for f in (self.eos.G, self.eos.Gs)
            ]
            w_minus, w_plus = self.eos.w(T_minus), self.eos.w(T_plus)
            steps.append(fabs(w_plus - w_minus) / fabs(w_minus))

            if max(steps) > JUMP_THRESHOLD:
                measured_jumps.add(T_in_GeV)

        self.assertEqual(measured_jumps, set(self.eos.discontinuity_temperatures_GeV))
        self.assertEqual(
            measured_jumps, {QCD_EOS.T_LO, QCD_EOS.T_120_MEV, QCD_EOS.T_HI}
        )

    def test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink(self):
        """
        The same statement one level up, in the quantity the ODE right-hand side actually reads:
        H(z) steps across T_LO and T_120_MEV and is continuous across EOS_T_LO.
        """
        u_lo, u_hi = log1p(0.1), log1p(1e14)
        expected = {
            QCD_EOS.T_LO: 4.4e-4,
            QCD_EOS.EOS_T_LO: None,
            QCD_EOS.T_120_MEV: 1.0e-4,
        }
        for T_in_GeV, jump in expected.items():
            u = self.cosmology._temperature_crossing_log1pz(
                T_in_GeV * self.units.GeV, u_lo, u_hi
            )
            self.assertIsNotNone(u, msg=f"T = {T_in_GeV} GeV does not cross in range")
            H_minus = self.cosmology.Hubble(expm1(u * (1.0 - 1e-12)))
            H_plus = self.cosmology.Hubble(expm1(u * (1.0 + 1e-12)))
            step = fabs(H_plus - H_minus) / fabs(H_minus)
            if jump is None:
                self.assertLess(step, 1e-8, msg=f"T = {T_in_GeV} GeV")
            else:
                self.assertAlmostEqual(
                    step / jump, 1.0, delta=0.05, msg=f"T = {T_in_GeV} GeV"
                )

    def test_kind_selects_knots_or_jumps(self):
        z_lo, z_hi = 0.1, 1e14
        every = _cosmology_break_points(self.cosmology, z_lo, z_hi)
        jumps = _cosmology_break_points(
            self.cosmology, z_lo, z_hi, kind=BREAK_POINT_DISCONTINUITY
        )

        # two of the four declared temperatures cross inside this range, and both are jumps;
        # everything else `every` carries is a T(z) spline knot
        self.assertEqual(len(jumps), 2)
        self.assertGreater(len(every), 100)
        self.assertTrue(np.all(np.isin(jumps, every)))

        # and the quadrature path is unchanged: kind defaults to "all"
        self.assertTrue(
            np.array_equal(
                every,
                self.cosmology.integration_break_points(
                    z_lo, z_hi, kind=BREAK_POINT_ALL
                ),
            )
        )

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            self.cosmology.integration_break_points(0.1, 1e14, kind="jumps-please")


# ---------------------------------------------------------------------------------------------
# 2. a cosmology declaring nothing takes the unsplit path
# ---------------------------------------------------------------------------------------------


class TestSmoothCosmologiesAreUnchanged(unittest.TestCase):
    def test_no_declaration_means_no_break_points(self):
        for model in (RadiationModel(), LambdaCDMModel()):
            self.assertEqual(
                declared_discontinuities_in_z(model, 0.1, 1e14),
                [],
                msg=model.name,
            )

    def test_lambdacdm_reproduces_a_direct_solve_ivp_call_exactly(self):
        """
        The prompt's "reproduces a known result exactly": the known result is what a single
        ``solve_ivp`` call with the same arguments produces, computed here independently of the
        integrator under test. Bit for bit, not nearly.
        """
        model = LambdaCDMModel()
        k = 1.0e7
        z_source = horizon_exit_z(model.cosmology, k, -5.0)
        z_e6 = horizon_exit_z(model.cosmology, k, 6.0)
        source_grid = production_source_grid(z_source)
        grid = production_response_grid(source_grid).truncate(
            0.85 * z_e6, keep="higher-include"
        )
        z_init = grid.max

        payload = numeric_with_phase_cut._function(
            _Proxy(model, UNITS),
            _KExit(k, UNITS, horizon_exit_z(model.cosmology, k, 0.0)),
            z_init,
            grid,
            initial_value=0.0,
            initial_deriv=1.0,
            RHS=Gk_RHS,
            omega_sq=Gk_omegaEff_sq,
            atol=PRODUCTION_ATOL,
            rtol=PRODUCTION_RTOL,
            delta_logz=PRODUCTION_DELTA_LOGZ,
            task_label="test_lambdacdm_unsplit",
            object_label="Gr_k(z, z')",
            warn_unresolved_osc=False,
        )

        with NumericIntegrationSupervisor(
            _Wavenumber(k, 1, UNITS), z_init, grid.min, "reference"
        ) as supervisor:
            reference = solve_ivp(
                Gk_RHS,
                method="DOP853",
                t_span=(z_init.z, float(grid.min)),
                y0=[0.0, 1.0],
                t_eval=grid.as_float_list(),
                events=None,
                dense_output=False,
                atol=PRODUCTION_ATOL,
                rtol=PRODUCTION_RTOL,
                args=(model, k, supervisor),
            )

        self.assertTrue(reference.success)
        self.assertEqual(len(payload["value_sample"]), len(reference.t))
        for i, (value, deriv) in enumerate(
            zip(payload["value_sample"], payload["deriv_sample"])
        ):
            self.assertEqual(value, reference.y[VALUE_INDEX][i], msg=f"sample {i}")
            self.assertEqual(deriv, reference.y[DERIV_INDEX][i], msg=f"sample {i}")


# ---------------------------------------------------------------------------------------------
# 3. the split path
# ---------------------------------------------------------------------------------------------


class TestSplitAtDeclaredJump(unittest.TestCase):
    def setUp(self):
        self.grid = _jump_grid()
        self.split = _JumpModel(
            JUMP_Z,
            JUMP_OMEGA_HI,
            JUMP_OMEGA_LO,
            z_kink=JUMP_Z_KINK,
            declare=True,
        )
        self.unsplit = _JumpModel(
            JUMP_Z,
            JUMP_OMEGA_HI,
            JUMP_OMEGA_LO,
            z_kink=JUMP_Z_KINK,
            declare=False,
        )

    def test_only_the_jump_is_returned_to_the_ode(self):
        """The kink is declared but is not a discontinuity, so the ODE does not split at it."""
        points = declared_discontinuities_in_z(self.split, JUMP_Z_END, JUMP_Z_INIT)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0], JUMP_Z, places=12)

        every = _cosmology_break_points(
            self.split.cosmology, JUMP_Z_END, JUMP_Z_INIT, kind=BREAK_POINT_ALL
        )
        self.assertEqual(len(every), 2)

        self.assertEqual(
            declared_discontinuities_in_z(self.unsplit, JUMP_Z_END, JUMP_Z_INIT), []
        )

    def test_the_split_run_matches_the_closed_form(self):
        """
        The split run reproduces the piecewise-analytic solution to the tolerance it was asked
        for, on both sides of the discontinuity. This is the correctness half; that the split is
        also *more accurate* than not splitting is not demonstrable on this fixture and is tested
        on the production background in :class:`TestQCDReferenceConvergence` -- see the module
        docstring.
        """
        for atol, rtol, bound in (
            (PRODUCTION_ATOL, PRODUCTION_RTOL, 1e-6),
            (1e-13, 1e-11, 1e-8),
        ):
            split = _run_jump(self.split, self.grid, atol, rtol)
            error = _jump_error(self.split, self.grid, split)
            self.assertLess(error, bound, msg=f"atol={atol:.0e}, rtol={rtol:.0e}")

    def test_the_split_run_converges_as_the_tolerance_is_tightened(self):
        """
        Monotone refinement is the property a straddling step destroys, and it is what the
        campaign's reference-convergence test relies on. Here it is checked directly: three
        decades of tolerance buy at least two decades of accuracy.
        """
        loose = _jump_error(
            self.split, self.grid, _run_jump(self.split, self.grid, 1e-10, 1e-8)
        )
        tight = _jump_error(
            self.split, self.grid, _run_jump(self.split, self.grid, 1e-13, 1e-11)
        )
        self.assertLess(tight, loose / 100.0, msg=f"{loose:.4g} -> {tight:.4g}")

    def test_sample_grid_is_exactly_the_grid_requested(self):
        payload = _run_jump(self.split, self.grid, PRODUCTION_ATOL, PRODUCTION_RTOL)
        self.assertEqual(len(payload["value_sample"]), len(self.grid))
        self.assertEqual(len(payload["deriv_sample"]), len(self.grid))

        # the break redshift is not one of the requested samples and must not have become one
        self.assertNotIn(JUMP_Z, self.grid.as_float_list())

    def test_a_sample_sitting_on_the_break_is_returned_once(self):
        """
        The degenerate case the segment partition has to get right: a requested sample lying
        exactly on the declared discontinuity.
        """
        z_values = sorted(
            set(np.linspace(JUMP_Z_INIT, JUMP_Z_END, 51)) | {JUMP_Z}, reverse=True
        )
        grid = to_redshift_array(z_values)
        payload = _run_jump(self.split, grid, PRODUCTION_ATOL, PRODUCTION_RTOL)

        self.assertEqual(len(payload["value_sample"]), len(grid))
        self.assertEqual([z.z for z in grid], [float(v) for v in grid.as_float_list()])
        error = _jump_error(self.split, grid, payload)
        self.assertLess(error, 1e-6)

    def test_supervisor_accounting_aggregates_over_segments(self):
        """
        The accounting must cover the whole run, not the last segment. The lower segment alone --
        the same integration restricted to the grid below the jump -- is the number a
        last-segment-only report would give, so the full run has to cost substantially more than
        that.
        """
        split = _run_jump(self.split, self.grid, PRODUCTION_ATOL, PRODUCTION_RTOL)

        below = to_redshift_array(
            [z.z for z in self.grid if z.z <= JUMP_Z],
        )
        lower_only = _run_jump(self.split, below, PRODUCTION_ATOL, PRODUCTION_RTOL)

        self.assertGreater(
            split["data"].RHS_evaluations,
            1.5 * lower_only["data"].RHS_evaluations,
            msg=(
                f"whole run {split['data'].RHS_evaluations} against lower segment alone "
                f"{lower_only['data'].RHS_evaluations}"
            ),
        )
        self.assertGreater(
            split["data"].compute_steps, 1.5 * lower_only["data"].compute_steps
        )
        self.assertGreater(split["data"].max_RHS_time, 0.0)
        self.assertGreater(split["data"].mean_RHS_time, 0.0)

        # and the oscillation diagnostic still runs, over the whole returned grid
        self.assertIn("has_unresolved_osc", split)
        self.assertIsNotNone(split["has_unresolved_osc"])

    def test_unresolved_oscillation_flag_survives_the_split(self):
        """
        A grid too coarse to resolve the oscillation trips the flag in the split run exactly as it
        does in the unsplit one, and at the same redshift: the scan runs once, after the solve, on
        the concatenated sample grid.
        """
        coarse = _jump_grid(num=20)
        split = _run_jump(self.split, coarse, PRODUCTION_ATOL, PRODUCTION_RTOL)
        unsplit = _run_jump(self.unsplit, coarse, PRODUCTION_ATOL, PRODUCTION_RTOL)

        self.assertTrue(split["has_unresolved_osc"])
        self.assertEqual(split["has_unresolved_osc"], unsplit["has_unresolved_osc"])
        self.assertEqual(split["unresolved_z"], unsplit["unresolved_z"])


class TestStopModeAcrossSegments(unittest.TestCase):
    """
    ``mode="stop"`` with the stop event in an interior segment: the whole integration must
    terminate there (not merely that segment), and the dense-output root-find must work when the
    search window straddles a segment boundary.
    """

    def setUp(self):
        # two declared jumps, so that there are three segments and the stop event falls in the
        # second of them
        self.model = _JumpModel(1.5, JUMP_OMEGA_HI, JUMP_OMEGA_LO)
        self.model.cosmology = _JumpCosmology(1.5)
        self.grid = _jump_grid()

    def _run_stop(self, window_begin: float, window_end: float):
        return _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            mode="stop",
            stop_search_window_z_begin=window_begin,
            stop_search_window_z_end=window_end,
        )

    def test_event_in_the_interior_segment_terminates_the_whole_integration(self):
        # the window starts above the jump and ends below it, so both the event and the search
        # straddle the segment boundary at z = 1.5
        payload = self._run_stop(1.8, 1.3)

        returned = len(payload["value_sample"])
        self.assertGreater(returned, 0)
        self.assertLess(returned, len(self.grid))

        # nothing below the event was produced
        self.assertGreaterEqual(min(z.z for z in self.grid[:returned]), 1.3 - 1e-6)

    def test_extremum_matches_the_closed_form(self):
        """
        The state returned at the stop point is the closed-form state there, which is a genuine
        test of the composite dense output: the root-find walks down through the segment boundary
        and reads the solution from whichever segment contains the abscissa.
        """
        payload = self._run_stop(1.8, 1.3)

        # _KExit is constructed with z_exit = 1.0, and stop_deltaz_subh = z_exit - z_stop
        z_stop = 1.0 - payload["stop_deltaz_subh"]
        self.assertGreater(z_stop, 1.3)
        self.assertLess(z_stop, 1.8)

        y_exact, yp_exact = self.model.exact(
            z_stop, JUMP_Z_INIT, JUMP_INITIAL_VALUE, JUMP_INITIAL_DERIV
        )
        envelope = hypot(y_exact, yp_exact / self.model.omega(z_stop))

        self.assertLess(
            fabs(payload["stop_value"] - y_exact) / envelope,
            1e-6,
            msg=f"z_stop={z_stop:.8g}",
        )
        self.assertLess(
            fabs(payload["stop_deriv"] - yp_exact)
            / (envelope * self.model.omega(z_stop)),
            1e-4,
            msg=f"z_stop={z_stop:.8g}",
        )

    def test_event_fires_in_a_segment_that_returns_no_output_points(self):
        """
        The degenerate segment: one that holds no requested sample and whose lower boundary is
        never reached, because the stop event fires above it. SciPy leaves ``sol.y`` as an empty
        *list* rather than an empty array in that case, which the driver has to normalise before
        slicing.
        """
        model = _JumpModel(1.9, JUMP_OMEGA_HI, JUMP_OMEGA_LO)
        model.cosmology = _TwoJumpCosmology(1.9, 1.85)
        grid = to_redshift_array([2.0, 1.95, 1.5, 1.0])

        payload = _run_jump(
            model,
            grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            mode="stop",
            stop_search_window_z_begin=1.99,
            stop_search_window_z_end=1.87,
        )

        self.assertIsNotNone(payload["stop_value"])
        # only the samples above the event survive
        self.assertEqual(len(payload["value_sample"]), 2)

    def test_extremum_is_the_same_when_the_window_lies_inside_one_segment(self):
        """
        A control: with the search window entirely above the declared jump the root-find sees a
        single segment's dense output, exactly as it did before this prompt. The extremum found
        must be the same one the straddling search finds when the window is widened downwards.
        """
        inside = self._run_stop(1.9, 1.6)
        straddling = self._run_stop(1.9, 1.3)

        self.assertAlmostEqual(
            inside["stop_deltaz_subh"],
            straddling["stop_deltaz_subh"],
            places=6,
        )


class TestQCDReferenceConvergence(unittest.TestCase):
    """
    The claim the prompt exists for, on the cosmology that motivated it.

    ``QCD_Cosmology``'s H(z) jumps by 1.04e-4 where T(z) crosses the equation of state's
    ``T_120_MEV``, at z = 8.64e11, which is inside the transfer function's numeric range at every
    production wavenumber. Before the split, two runs of the same integrator a decade apart in
    tolerance differ by microns of the envelope where they should differ by picometres, so the
    campaign's reference-convergence estimate is meaningless there
    (``docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`` §4). After it, they converge.

    **That description was measured on a background whose ``T(z)`` carried a 1e-07-level
    interpolation error throughout**, and most of the unsplit run's drift turned out to be that
    rather than the jump: since ``prompts/qcd-background-audit/`` prompt 05 the unsplit run also
    converges at this wavenumber. See ``test_split_converges_where_unsplit_does_not`` for the
    numbers, and README §7 D5 for why that does not by itself retire the split.

    k = 4.97e7/Mpc is chosen because it is the wavenumber at which the *standoff* of the segment
    boundary matters: with the boundary placed exactly on the crossing rather than a hair above
    it, this is the one of §4's four named failures that does not recover
    (``BREAK_POINT_STANDOFF``).
    """

    # index 38 of np.logspace(log10(1e5), log10(3e8), 50), the production source k-grid
    K_INV_MPC = float(np.logspace(log10(1.0e5), log10(3.0e8), 50)[38])

    # prompt 17 §2.1's criterion for QCD: a tenth of the smallest candidate difference the sweep
    # reports there, 3.45e-7
    ACCEPTANCE_DRIFT = 3.4e-8

    # How much better the split run has to be than the unsplit one. See
    # test_split_converges_where_unsplit_does_not, which used to assert `unsplit > 1e-6` -- that
    # the unsplit run *fails* ACCEPTANCE_DRIFT outright. That premise was falsified by
    # prompts/qcd-background-audit/ prompt 05 and the test now asserts the weaker, true statement
    # that splitting still buys the better part of an order of magnitude. Measured 7.65x; 5.0
    # leaves room for the arithmetic to move without letting a genuine loss of the split's value
    # through.
    UNSPLIT_PENALTY_FACTOR = 5.0

    REFERENCE = (1e-18, 1e-12)
    TIGHTENED = (1e-19, 1e-13)

    @classmethod
    def setUpClass(cls):
        cls.cosmology = QCD_Cosmology(
            store_id=0, units=UNITS, params=Planck2018(), max_z=1e20
        )
        # the stand-in is built on the source grid of the largest production wavenumber, which
        # covers every other wavenumber's range (the pattern of the sweep script)
        cls.model = QCDModel(
            production_source_grid(
                horizon_exit_z(cls.cosmology, 3.0e8, -5.0),
            ),
            cosmology=cls.cosmology,
        )
        cls.unsplit = _UnsplitView(cls.model, cls.cosmology)

        z_e3 = horizon_exit_z(cls.cosmology, cls.K_INV_MPC, 3.0)
        z_e6 = horizon_exit_z(cls.cosmology, cls.K_INV_MPC, 6.0)
        z_source = horizon_exit_z(cls.cosmology, cls.K_INV_MPC, -5.0)
        cls.grid = production_source_grid(z_source).truncate(
            0.85 * z_e6, keep="higher-include"
        )
        cls.z_e3 = z_e3
        cls.z_e6 = z_e6
        cls.z_exit = horizon_exit_z(cls.cosmology, cls.K_INV_MPC, 0.0)

    def _run(self, model, atol, rtol):
        return numeric_with_phase_cut._function(
            _Proxy(model, UNITS),
            _KExit(self.K_INV_MPC, UNITS, self.z_exit),
            self.grid.max,
            self.grid,
            initial_value=1.0,
            initial_deriv=0.0,
            RHS=Tk_RHS,
            omega_sq=Tk_omegaEff_sq,
            atol=atol,
            rtol=rtol,
            delta_logz=PRODUCTION_DELTA_LOGZ,
            mode="stop",
            stop_search_window_z_begin=min(self.z_e3, self.grid.max.z),
            stop_search_window_z_end=self.z_e6,
            task_label="test_qcd_reference_convergence",
            object_label="Tk(z)",
            warn_unresolved_osc=False,
        )

    def _drift(self, model) -> float:
        reference = self._run(model, *self.REFERENCE)
        tightened = self._run(model, *self.TIGHTENED)
        worst = 0.0
        for z, value, ref_value, ref_deriv in zip(
            self.grid,
            tightened["value_sample"],
            reference["value_sample"],
            reference["deriv_sample"],
        ):
            omega_sq = Tk_omegaEff_sq(model, self.K_INV_MPC, z.z)
            if omega_sq <= 0.0:
                continue
            envelope = hypot(ref_value, ref_deriv / sqrt(omega_sq))
            worst = max(worst, envelope_relative_error(value, ref_value, envelope))
        return worst

    def test_the_branch_crossing_is_inside_the_range(self):
        # prompts/qcd-background-audit/ prompt 04 tightened _solve_T_z's node solve, which moves
        # the T(z) spline this crossing is solved against (README §2 (d)): 8.64355463e11 ->
        # 8.64366999e11, a 1.335e-05 relative shift -- not a new physical crossing, the same
        # T_120_MEV branch located slightly more accurately. Prompt 05 moves it once more, for
        # the same reason -- the splined quantity became the entropy factor, so the spline moved
        # again -- by a further 1.712e-05 relative: 8.64366999e11 -> 8.6438180e11. Still the same
        # branch, still the same physics; `places` is unchanged.
        points = declared_discontinuities_in_z(
            self.model, float(self.grid.min), self.grid.max.z
        )
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0] / 8.6438180e11, 1.0, places=6)

    def test_split_converges_where_unsplit_does_not(self):
        """
        **The method name records the measurement as it stood when the test was written, and that
        measurement no longer holds.** It asserted ``unsplit > 1e-6`` -- that a run which does not
        split at the declared discontinuity *fails* ``ACCEPTANCE_DRIFT`` outright. Measured on
        this wavenumber, at the two commits either side of
        ``prompts/qcd-background-audit/`` prompt 05:

            prompt 04 (tightened nodes, T splined against u):  split 2.2136e-09
                                                               unsplit 1.0213e-06  (fails 3.4e-08,
                                                                                    and cleared
                                                                                    the 1e-6 bound
                                                                                    by only 2 %)
            prompt 05 (entropy factor splined):                split 2.9753e-09
                                                               unsplit 2.2767e-08  (passes 3.4e-08)

        The unsplit run improved by a factor of 45 while the split run barely moved, so most of
        what the split was rescuing was never the jump in ``H(z)`` at the ``T_120_MEV`` crossing:
        it was the interpolation noise the old representation carried across the whole range,
        which the entropy factor removes (audit §3, T3).

        **Whether that means ``BREAK_POINT_ALL`` has stopped being load-bearing is README §7 D5's
        question, and it is not decided here.** This is one wavenumber of fifty, in the ``T_k``
        sector alone; ``TkNumericIntegration.BREAK_POINT_KIND`` is in a datastore lookup key; and
        the campaign's README §2 (f) makes an unmeasured collapse of the break-point set a stop
        condition. Prompt 08 re-takes ``GkTk-remedial`` prompt 19's measurement across all fifty
        wavenumbers and reports; the board issue is
        ``[04-unsplit-tk-run-now-meets-the-criterion]``.

        What is asserted instead is what is still true and still worth guarding: the split run
        meets the criterion, and splitting still buys the better part of an order of magnitude
        (measured 7.65x).
        """
        split = self._drift(self.model)
        unsplit = self._drift(self.unsplit)

        print(
            f"[break points] k = {self.K_INV_MPC:.4g}/Mpc: split drift {split:.4e}, "
            f"unsplit {unsplit:.4e} ({unsplit / split:.2f}x), criterion "
            f"{self.ACCEPTANCE_DRIFT:.3e}"
        )

        self.assertLess(
            split,
            self.ACCEPTANCE_DRIFT,
            msg=f"split {split:.4g} against unsplit {unsplit:.4g}",
        )
        self.assertGreater(
            unsplit,
            self.UNSPLIT_PENALTY_FACTOR * split,
            msg=(
                f"splitting at the discontinuity no longer buys a factor of "
                f"{self.UNSPLIT_PENALTY_FACTOR:g}: split {split:.4g}, unsplit {unsplit:.4g}. "
                "See README §7 D5 and [04-unsplit-tk-run-now-meets-the-criterion]"
            ),
        )


# ---------------------------------------------------------------------------------------------
# 5. the policy is the caller's (prompt 19)
# ---------------------------------------------------------------------------------------------

# how many non-jump break points the "many boundaries" fixture declares, against JUMP_SAMPLES=200
# requested samples over the same range: twice as many boundaries as samples, so that a majority
# of segments hold none. On QCD_Cosmology the production ratio is ~125 boundaries to ~100 samples
# in the transfer function's numeric range, so this is the same regime, exaggerated.
MANY_BREAK_COUNT = 400


def _segmented_solve(model, grid, kind: str, events=None, dense_output: bool = False):
    """
    Drive :func:`_solve_segmented` directly, which is the only way to see the *number of segments*
    a policy produces -- ``numeric_with_phase_cut``'s payload deliberately does not carry it, and
    this prompt does not add a field to a dict two datastore factories consume.
    """
    break_z = declared_discontinuities_in_z(
        model, float(grid.min), grid.max.z, kind=kind
    )
    with NumericIntegrationSupervisor(
        _Wavenumber(1.0, 1, UNITS), grid.max, grid.min, "y(z)"
    ) as supervisor:
        return _solve_segmented(
            _jump_RHS,
            grid.max.z,
            float(grid.min),
            grid.as_float_list(),
            [JUMP_INITIAL_VALUE, JUMP_INITIAL_DERIV],
            break_z,
            events,
            dense_output,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            (model, 1.0, supervisor),
            "test_segmented_solve",
            1.0,
        )


def _payloads_are_bit_identical(a: dict, b: dict) -> bool:
    """Every returned floating-point number equal under ``==``, and the same evaluation count."""
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


class TestPerSectorPolicy(unittest.TestCase):
    """
    ``break_point_kind`` is a choice the caller makes, not a change of behaviour for everyone.
    """

    def setUp(self):
        self.grid = _jump_grid()
        self.model = _JumpModel(
            JUMP_Z, JUMP_OMEGA_HI, JUMP_OMEGA_LO, z_kink=JUMP_Z_KINK, declare=True
        )
        self.smooth = _JumpModel(
            JUMP_Z, JUMP_OMEGA_HI, JUMP_OMEGA_LO, z_kink=JUMP_Z_KINK, declare=False
        )

    def test_the_parameter_selects_the_number_of_segments(self):
        """
        The stand-in declares one jump and one kink. Asking for the jumps alone cuts the range in
        two; asking for every declared break point cuts it in three.
        """
        self.assertEqual(
            len(
                declared_discontinuities_in_z(
                    self.model, JUMP_Z_END, JUMP_Z_INIT, kind=BREAK_POINT_DISCONTINUITY
                )
            ),
            1,
        )
        self.assertEqual(
            len(
                declared_discontinuities_in_z(
                    self.model, JUMP_Z_END, JUMP_Z_INIT, kind=BREAK_POINT_ALL
                )
            ),
            2,
        )

        self.assertEqual(
            _segmented_solve(
                self.model, self.grid, BREAK_POINT_DISCONTINUITY
            ).num_segments,
            2,
        )
        self.assertEqual(
            _segmented_solve(self.model, self.grid, BREAK_POINT_ALL).num_segments, 3
        )

    def test_the_default_reproduces_the_jumps_only_result_exactly(self):
        """
        Every caller written before this prompt -- the tests, the reproduction scripts under
        ``docs/``, a future integrator -- keeps the numbers it had, bit for bit.
        """
        omitted = _run_jump(self.model, self.grid, PRODUCTION_ATOL, PRODUCTION_RTOL)
        explicit = _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            break_point_kind=BREAK_POINT_DISCONTINUITY,
        )
        self.assertTrue(_payloads_are_bit_identical(omitted, explicit))

        # and the other policy really is a different integration, so the check above is not
        # comparing two identical code paths
        every = _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            break_point_kind=BREAK_POINT_ALL,
        )
        self.assertFalse(_payloads_are_bit_identical(omitted, every))

    def test_a_cosmology_declaring_nothing_takes_the_single_call_path_either_way(self):
        """
        Reachable two ways since this prompt, so asserted two ways: the declaration is empty under
        both policies, and the two runs agree bit for bit.
        """
        for model in (RadiationModel(), LambdaCDMModel(), self.smooth):
            for kind in (BREAK_POINT_DISCONTINUITY, BREAK_POINT_ALL):
                self.assertEqual(
                    declared_discontinuities_in_z(model, 0.1, 1e14, kind=kind),
                    [],
                    msg=f"{model.name}, kind={kind}",
                )

        self.assertEqual(
            _segmented_solve(self.smooth, self.grid, BREAK_POINT_ALL).num_segments, 1
        )
        self.assertTrue(
            _payloads_are_bit_identical(
                _run_jump(
                    self.smooth,
                    self.grid,
                    PRODUCTION_ATOL,
                    PRODUCTION_RTOL,
                    break_point_kind=BREAK_POINT_DISCONTINUITY,
                ),
                _run_jump(
                    self.smooth,
                    self.grid,
                    PRODUCTION_ATOL,
                    PRODUCTION_RTOL,
                    break_point_kind=BREAK_POINT_ALL,
                ),
            )
        )

    def test_each_production_call_site_passes_the_kind_its_sector_decided_on(self):
        """
        Read with ``ast``, following
        ``test_numeric_phase_cut.test_both_integrators_pass_warn_unresolved_osc_False``: neither
        integration module can be imported without Ray, and the point is the argument, not the
        call.

        The asymmetry is the substance of this prompt -- Tk splits at every declared break point
        because measurement says it must, Gk at the jumps alone because measurement says the rest
        would buy nothing at ~65,000 objects per model -- so both sites name their kind, the Gk
        one included even though it is the module default.

        **Prompt 20 moved where the kind is named, and this test follows it.** When prompt 19
        wrote it the call site carried the imported literal; since prompt 20 the same value is
        also stored in and filtered on by the datastore lookup key, so each compute target holds
        one declaration, ``BREAK_POINT_KIND``, and the call site passes that. What is checked is
        unchanged in substance and slightly stronger: the site names its kind explicitly, and the
        constant it names is the vocabulary value this sector's measurement chose.
        ``test_numeric_break_point_key`` then checks that the stored value and the queried value
        read the same declaration.
        """
        expected = {
            "ComputeTargets/TkNumericIntegration.py": (
                TkNumericIntegration,
                "BREAK_POINT_ALL",
                BREAK_POINT_ALL,
            ),
            "ComputeTargets/GkNumericIntegration.py": (
                GkNumericIntegration,
                "BREAK_POINT_DISCONTINUITY",
                BREAK_POINT_DISCONTINUITY,
            ),
        }

        for module, (cls, name, value) in expected.items():
            path = Path(__file__).parents[2] / module
            tree = ast.parse(path.read_text(), filename=str(path))

            calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "remote"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "numeric_with_phase_cut"
            ]
            self.assertEqual(len(calls), 1, msg=module)

            keywords = {kw.arg: kw.value for kw in calls[0].keywords}
            self.assertIn("break_point_kind", keywords, msg=module)

            # the site passes the compute target's single declaration, self.BREAK_POINT_KIND
            argument = keywords["break_point_kind"]
            self.assertIsInstance(argument, ast.Attribute, msg=module)
            self.assertEqual(argument.attr, "BREAK_POINT_KIND", msg=module)
            self.assertIsInstance(argument.value, ast.Name, msg=module)
            self.assertEqual(argument.value.id, "self", msg=module)

            # and that declaration is the constant this test thinks it is: the class attribute
            # carries the value, and the name it is written from is imported from
            # ComputeTargets.BackgroundModel
            self.assertEqual(cls.BREAK_POINT_KIND, value, msg=module)
            self.assertIsInstance(value, str)

            declarations = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "BREAK_POINT_KIND"
            ]
            self.assertEqual(len(declarations), 1, msg=module)
            self.assertIsInstance(declarations[0].value, ast.Name, msg=module)
            self.assertEqual(declarations[0].value.id, name, msg=module)

            imported = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
                and node.module == "ComputeTargets.BackgroundModel"
                and any(alias.name == name for alias in node.names)
            ]
            self.assertEqual(len(imported), 1, msg=module)


class TestManyBreakPoints(unittest.TestCase):
    """
    What ~400 boundaries expose that prompt 18's single production boundary could not.

    The fixture declares one genuine jump and :data:`MANY_BREAK_COUNT` further break points at
    which the right-hand side is in fact smooth, so the closed-form solution is unchanged by
    splitting at them and any damage the segmentation does shows up directly.
    """

    def setUp(self):
        self.grid = _jump_grid()
        self.model = _JumpModel(JUMP_Z, JUMP_OMEGA_HI, JUMP_OMEGA_LO)
        self.model.cosmology = _ManyBreakCosmology(
            JUMP_Z, JUMP_Z_END, JUMP_Z_INIT, MANY_BREAK_COUNT
        )

    def test_most_segments_hold_no_requested_sample(self):
        """
        §2.3 item 1. With more boundaries than samples the ``num_requested == 0`` branch is the
        common case, and the assembled grid must still be exactly the grid requested, in order.
        """
        solution = _segmented_solve(self.model, self.grid, BREAK_POINT_ALL)

        self.assertEqual(solution.num_segments, MANY_BREAK_COUNT + 2)
        self.assertGreater(solution.num_segments, 2 * len(self.grid) * 0.9)

        requested = self.grid.as_float_list()
        self.assertEqual(len(solution.t), len(requested))
        for i, (returned, asked) in enumerate(zip(solution.t, requested)):
            self.assertEqual(float(returned), float(asked), msg=f"sample {i}")
        self.assertEqual(solution.y.shape, (2, len(requested)))

    def test_splitting_where_the_right_hand_side_is_smooth_changes_nothing_material(
        self,
    ):
        """
        The kinks are declared but the coefficient is continuous there, so the extra 400 restarts
        must cost evaluations and not accuracy.
        """
        jumps = _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            break_point_kind=BREAK_POINT_DISCONTINUITY,
        )
        every = _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            break_point_kind=BREAK_POINT_ALL,
        )

        self.assertLess(_jump_error(self.model, self.grid, every), 1e-6)
        self.assertGreater(every["data"].RHS_evaluations, jumps["data"].RHS_evaluations)

        # the diagnostic and the accounting aggregate over all of them, not over the last
        self.assertEqual(every["has_unresolved_osc"], jumps["has_unresolved_osc"])
        self.assertGreater(every["data"].max_RHS_time, 0.0)
        self.assertGreater(every["data"].compute_steps, jumps["data"].compute_steps)

    def test_boundaries_closer_than_the_standoff_collapse(self):
        """
        §2.3 item 2. Two declared points closer to each other than ``BREAK_POINT_STANDOFF`` would,
        after the standoff is applied, leave a segment of zero or negative length. The driver drops
        the offending boundary instead. This cannot arise on any production geometry -- the T(z)
        spline's knots are 2.85e-02 apart in log(1+z), and the closest a declared temperature
        crossing comes to a knot is 3.4e-03, nine orders above the 1e-12 standoff -- so it is
        constructed here.
        """
        base = _ManyBreakCosmology(JUMP_Z, JUMP_Z_END, JUMP_Z_INIT, MANY_BREAK_COUNT)
        crowded = _ManyBreakCosmology(
            JUMP_Z,
            JUMP_Z_END,
            JUMP_Z_INIT,
            MANY_BREAK_COUNT,
            coincident=[
                base.z_kinks[7]
                + offset * BREAK_POINT_STANDOFF * (1.0 + base.z_kinks[7])
                for offset in (0.1, 0.2)
            ],
        )

        model = _JumpModel(JUMP_Z, JUMP_OMEGA_HI, JUMP_OMEGA_LO)
        model.cosmology = crowded

        # the cosmology declares two more points than the base fixture ...
        self.assertEqual(
            len(
                declared_discontinuities_in_z(
                    model, JUMP_Z_END, JUMP_Z_INIT, kind=BREAK_POINT_ALL
                )
            ),
            MANY_BREAK_COUNT + 3,
        )
        # ... and the driver emits the same number of segments as if it had not, because the
        # three mutually unresolvable boundaries collapse to one
        self.assertEqual(
            _segmented_solve(model, self.grid, BREAK_POINT_ALL).num_segments,
            MANY_BREAK_COUNT + 2,
        )

        payload = _run_jump(
            model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            break_point_kind=BREAK_POINT_ALL,
        )
        self.assertEqual(len(payload["value_sample"]), len(self.grid))
        self.assertLess(_jump_error(model, self.grid, payload), 1e-6)

    def test_stop_mode_survives_a_long_chain_of_segments(self):
        """
        §2.3 item 3. The terminal event must stop the *whole* integration, and
        ``find_phase_extremum`` must still find its extremum through a composite dense output of
        hundreds of segments rather than two.
        """
        payload = _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            mode="stop",
            stop_search_window_z_begin=1.8,
            stop_search_window_z_end=1.3,
            break_point_kind=BREAK_POINT_ALL,
        )

        returned = len(payload["value_sample"])
        self.assertGreater(returned, 0)
        self.assertLess(returned, len(self.grid))
        self.assertGreaterEqual(min(z.z for z in self.grid[:returned]), 1.3 - 1e-6)

        # _KExit is constructed with z_exit = 1.0, and stop_deltaz_subh = z_exit - z_stop
        z_stop = 1.0 - payload["stop_deltaz_subh"]
        self.assertGreater(z_stop, 1.3)
        self.assertLess(z_stop, 1.8)

        y_exact, yp_exact = self.model.exact(
            z_stop, JUMP_Z_INIT, JUMP_INITIAL_VALUE, JUMP_INITIAL_DERIV
        )
        envelope = hypot(y_exact, yp_exact / self.model.omega(z_stop))
        self.assertLess(
            fabs(payload["stop_value"] - y_exact) / envelope,
            1e-6,
            msg=f"z_stop={z_stop:.8g}",
        )

        # and it is the same extremum the jumps-only policy finds
        jumps = _run_jump(
            self.model,
            self.grid,
            PRODUCTION_ATOL,
            PRODUCTION_RTOL,
            mode="stop",
            stop_search_window_z_begin=1.8,
            stop_search_window_z_end=1.3,
            break_point_kind=BREAK_POINT_DISCONTINUITY,
        )
        self.assertAlmostEqual(
            payload["stop_deltaz_subh"], jumps["stop_deltaz_subh"], places=6
        )


if __name__ == "__main__":
    unittest.main()
