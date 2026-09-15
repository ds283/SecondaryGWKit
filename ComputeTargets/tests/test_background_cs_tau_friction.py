"""
Tests for the two transfer-function primitives ``BackgroundModel`` tabulates alongside the
conformal time (prompts/GkTk-remedial, prompt 04 §3): the sound horizon
``cs_tau = int c_s dz/H`` with ``c_s^2 = wPerturbations`` (review §12.2), and the Liouville-Green
friction integral ``friction_F``, the primitive of ``dF/dz = (3/2)(1 + c_s^2)/(1+z)``
(review §12.7). Prompt 07 deleted that right-hand side from ``ComputeTargets/TkWKBIntegration.py``
when the producer switched to the table, so the ODE now lives only here, as ``_friction_RHS``
below, for the sole use of ``TestFrictionODEComparison`` -- the test that retires it.

As in ``test_background_tau.py``, the undecorated ``compute_background`` is driven directly and
the ``BackgroundModel`` is assembled offline through ``values_from_payload``, so what is exercised
is the production path: the table is rebuilt inside ``_create_functions`` from the persisted
limbs. No Ray and no datastore is needed.

Sign conventions being tested (README §2 (c); ``BackgroundModel._friction_integrand``):

* ``cs_tau.delta(z_a, z_b) = cs_tau(z_b) - cs_tau(z_a) = int_{z_b}^{z_a} c_s dz/H``, positive for
  ``z_b < z_a`` -- ``cs_tau`` *increases* towards lower redshift, like ``tau``.
* ``friction_F.delta(z_a, z_b) = F(z_b) - F(z_a)``, **negative** for ``z_b < z_a``: ``F`` decreases
  towards lower redshift, because ``dF/dz > 0``. ``friction_F.delta(z_init, z)`` is therefore
  exactly what ``integrate_friction_function`` accumulates today from ``F(z_init) = 0``, which is
  what prompt 07 will read.

Floors, so that nothing here is asserted below a reference's own accuracy:

* LambdaCDM: mpmath at 40 digits (``[01-lambdacdm-hubble-rounding-floor]`` for the Hubble
  evaluation itself).
* QCD ``cs_tau_minus_top``: the JSON reference is break-unaware ``quad`` and disagrees with the
  break-aware one by 1.89e-14 relative (``[02-qcd-reference-floor]``); the threshold is three
  times that, read from the JSON's own ``convergence`` block.
* ``friction_F`` is persisted as a **single** double (prompt 04 §1; README §7 D1), so every
  ``delta`` inherits up to one ulp of ``max |F|`` -- 7.1e-15 on the radiation grid used here,
  1.42e-14 on the production grid, where ``max |F| = 70.6``. That is a floor on the *absolute*
  error of a friction increment and it does not shrink with the baseline, so short-baseline
  friction increments are scored absolutely, never relatively (a 0.046-wide production increment
  at the QCD transition shows 1.3e-13 *relative* and 6.1e-15 absolute). It is deliberate: ``F``
  enters only as ``exp(F(z) - F(z_i))``, so 1e-14 absolute is 1e-14 relative in the amplitude,
  four orders below the 2.3-4.1e-7 of the friction ODE this replaces (review §12.3, and
  ``test_friction_ode_is_the_inaccurate_one`` below).
"""

import time
import unittest
from math import sqrt, log1p, expm1, fabs
from typing import List

import numpy as np
from scipy.integrate import quad, solve_ivp

from ComputeTargets.BackgroundModel import (
    BackgroundModel,
    BackgroundModelValue,
    ModelFunctions,
    TablePrimitive,
    CS_TAU_GAUSS_ORDER,
    FRICTION_F_GAUSS_ORDER,
    compute_background,
)
from ComputeTargets.tests.wkb_reference import (
    difference_error,
    horizon_exit_z,
    load_references,
    production_source_grid,
    to_redshift_array,
)
from CosmologyConcepts import wavenumber
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Quadrature.supervisors.base import RHS_timer
from Quadrature.supervisors.numeric import NumericIntegrationSupervisor
from Units import Mpc_units

# The retired friction ODE, relocated here verbatim from ``ComputeTargets/TkWKBIntegration.py``
# by prompt 07 of ``prompts/GkTk-remedial`` (its §2 item 1 deletes it from the producer, which
# now reads ``friction_F.delta`` from the background table instead). It is dead everywhere in
# the production path; it survives only in ``TestFrictionODEComparison`` below, the test that
# measures what it cost, so that prompt 04's measurement keeps measuring exactly the same ODE.
_FRICTION_INDEX = 0


def _friction_RHS(
    z: float,
    state: List[float],
    model: BackgroundModel,
    k_float: float,
    supervisor: NumericIntegrationSupervisor,
) -> List[float]:
    """
    k *must* be measured using the same units used for H(z) in the cosmology, otherwise we will not get
    correct dimensionless ratios
    """
    with RHS_timer(supervisor) as timer:
        if supervisor.notify_available:
            f = state[_FRICTION_INDEX]

            supervisor.message(
                z,
                f"current state: friction_func = {f:.5g}",
            )
            supervisor.reset_notify_time()

        one_plus_z = 1.0 + z
        cs2 = model.functions.wPerturbations(z)

        return [(3.0 / 2.0) * (1.0 + cs2) / one_plus_z]


# prompt 04 §3 test 1: the exact-radiation control
RADIATION_REL_TOL = 2.0e-15
RADIATION_FRICTION_ABS_TOL = 1.0e-14

# prompt 04 §3 test 2 / README §6
LAMBDACDM_CS_TAU_REL_TOL = 2.0e-14
FRICTION_REL_TOL = 1.0e-13
CS_TAU_SHORT_BASELINE_REL_TOL = 1.0e-13

# the single-limb friction floor: ~1.5 ulp of max |F| = 70.6 on the production grid
FRICTION_SHORT_BASELINE_ABS_TOL = 2.0e-14

# prompt 04 §3 test 3: the QCD references carry their own floor -- here
# references["convergence"]["models"]["QCDModel"]["branch+knots"]["cs_tau"]["json_vs_reference_max_rel"],
# "how well the JSON's cs_tau agrees with a converged adaptive reference". What is scored against
# it in test_qcd_checkpoints is the same kind of quantity one level down: how well the model's
# fixed-order Gauss table agrees with the JSON.
#
# Loosened 3.0 -> 8.3 by prompts/qcd-background-audit/ prompt 05, for
# [01-convergence-block-has-a-separate-generator] (docs/OPEN_ISSUES.md) and nothing else: that
# commit regenerated the JSON's QCD block with
# docs/qcd-background-audit/generate_qcd_references.py, but the floor sits in the top-level
# convergence block, which only docs/gktk-remedial/residual_convergence.py writes and which has
# not been re-run since the background moved. Numerator and denominator are therefore measured on
# two different backgrounds. Measured: 2.186e-14 (prompt 04 tree) -> 1.5501e-13 here, against a
# floor still recorded as 1.887e-14; 1.5501e-13/1.887e-14 = 8.213. Both are at the 1e-13 level, a
# few hundred ulp of a cumulative quadrature over twenty decades, and the worst point moves from
# z = 1.005e7 to z = 1.007e11. The companion friction_F assertion below is scored against
# FRICTION_REL_TOL rather than this floor and still passes untouched (6.525e-14 against 1e-13).
# **Taken back to 3.0 by prompt 06, one prompt earlier than expected**, and not because the block
# was regenerated -- it still has not been -- but because segmenting the representation at the
# jumps moved the numerator by two orders: 1.5501e-13 -> 2.212e-15, measured at z = 1.005e+07
# against the same recorded floor of 1.887e-14. The model's fixed-order table now agrees with the
# JSON an order *below* the floor recorded for the JSON itself, where prompt 05 sat eight times
# above it. Prompt 08 still re-runs residual_convergence.py and re-measures both sides;
# QCD_BREAK_POINT_ALIGNMENT_TOL in test_background_tau.py is the one figure of the three that
# prompt 05 loosened that this prompt could not take back.
QCD_FLOOR_FACTOR = 3.0

# prompt 04 §3 test 6: review §12.3 measures the friction ODE's amplitude error as 2.3e-7
# (k = 1e5) to 4.1e-7 (k = 3e8); the window brackets it by an order either way
FRICTION_ODE_ERROR_MIN = 5.0e-8
FRICTION_ODE_ERROR_MAX = 5.0e-6
FRICTION_ODE_K_INV_MPC = 1.0e5
FRICTION_ODE_ATOL = 1.0e-10
FRICTION_ODE_RTOL = 1.0e-8

# the radiation control grid: 100 samples per decade, and a top redshift low enough that
# max |F| = 2 log(1 + z_top) = 55.1 stays inside the binade whose ulp is 7.1e-15
RADIATION_Z_TOP = 1.0e12
RADIATION_Z_END = 0.1
RADIATION_NODES = 1301

_compute_background = compute_background._function


class _RadiationCosmology:
    """
    Exact radiation: ``H = H0 (1+z)^2``, ``w = c_s^2 = 1/3``, ``rho = 3 M_P^2 H^2`` (so that
    ``compute_background``'s ``tau_init`` asymptote returns exactly ``1/(H0 (1+z_init))``, the
    closed form). The closed forms this control is scored against:

    * ``tau(z)    = 1/(H0 (1+z))``
    * ``cs_tau(z) = tau(z)/sqrt(3)``
    * ``F(z_b) - F(z_a) = 2 log((1+z_b)/(1+z_a))``  (review §12.4: ``F = 2 log(s/s_i)``)

    It supplies no analytic derivatives, so ``compute_background`` takes the spline path for
    those; nothing here looks at them.
    """

    def __init__(self, H0: float = 1.0):
        self.store_id = 0
        self.H0 = float(H0)
        self._units = Mpc_units()
        self._Mpsq = self._units.PlanckMass * self._units.PlanckMass

    @property
    def units(self):
        return self._units

    def Hubble(self, z: float) -> float:
        return self.H0 * (1.0 + z) * (1.0 + z)

    def rho(self, z: float) -> float:
        return 3.0 * self._Mpsq * self.Hubble(z) * self.Hubble(z)

    def T_photon(self, z: float) -> float:
        return 0.0

    def wBackground(self, z: float) -> float:
        return 1.0 / 3.0

    def wPerturbations(self, z: float) -> float:
        return 1.0 / 3.0

    # closed forms
    def tau(self, z: float) -> float:
        return 1.0 / (self.H0 * (1.0 + z))

    def cs_tau_exact(self, z: float) -> float:
        return self.tau(z) / sqrt(3.0)

    def tau_delta_exact(self, z_a: float, z_b: float) -> float:
        """Factored, so the reference carries a few ulps of itself and not of ``tau``."""
        return (z_a - z_b) / (self.H0 * (1.0 + z_a) * (1.0 + z_b))

    def cs_tau_delta_exact(self, z_a: float, z_b: float) -> float:
        return self.tau_delta_exact(z_a, z_b) / sqrt(3.0)

    def friction_delta_exact(self, z_a: float, z_b: float) -> float:
        """``F(z_b) - F(z_a) = 2 log((1+z_b)/(1+z_a))``, negative for ``z_b < z_a``."""
        return 2.0 * (log1p(z_b) - log1p(z_a))


class _NegativeSoundSpeedCosmology(_RadiationCosmology):
    """A cosmology whose ``wPerturbations`` turns negative: it has no sound horizon."""

    def wPerturbations(self, z: float) -> float:
        return 1.0 / 3.0 if z > 1.0e3 else -0.25


def _offline_model(cosmology, z_sample, payload) -> BackgroundModel:
    """A BackgroundModel populated the way store() populates it, without Ray."""
    values = BackgroundModel.values_from_payload(z_sample, payload)
    return BackgroundModel(
        payload={
            "store_id": None,
            "data": payload["data"],
            "solver": None,
            "values": values,
        },
        solver_labels={},
        cosmology=cosmology,
        atol=None,
        rtol=None,
        z_sample=z_sample,
    )


def _fraction_point(z_hi: float, z_lo: float, fraction: float = 0.37) -> float:
    """The point ``fraction`` of the way from ``z_hi`` to ``z_lo`` in ``u = log(1+z)``."""
    return float(expm1(log1p(z_hi) + fraction * (log1p(z_lo) - log1p(z_hi))))


def _quad_reference(cosmology, z_lo: float, z_hi: float, integrand, break_points):
    """
    ``int_{z_lo}^{z_hi} f dz`` by adaptive quadrature, in the **exact-width** parametrisation
    ``1 + z = (1 + z_lo) e^t``, ``t in [0, W]`` with ``W = log1p((z_hi - z_lo)/(1 + z_lo))``.

    The reference is built this way, rather than between two rounded ``u = log(1+z)`` endpoints,
    because a short baseline inherits ``ulp(u)/W`` from the rounding -- ~1e-13 relative on a
    production interval near ``z = 1e6``, which is exactly the size being measured
    (``[03-qcd-short-baseline-reference-endpoint-rounding]``). The declared break points are
    passed to ``quad`` as subdivision points, so the QCD equation-of-state kinks are not
    integrated across.
    """
    width = log1p((z_hi - z_lo) / (1.0 + z_lo))
    one_plus_z_lo = 1.0 + z_lo
    u_lo = log1p(z_lo)

    def g(t: float) -> float:
        z = z_lo + one_plus_z_lo * expm1(t)
        return integrand(z) * (1.0 + z)

    kwargs = {"epsabs": 1.0e-16, "epsrel": 1.0e-13, "limit": 400}
    if break_points is not None:
        interior = [
            float(u) - u_lo for u in break_points if 0.0 < float(u) - u_lo < width
        ]
        if len(interior) > 0:
            kwargs["points"] = interior

    return quad(g, 0.0, width, **kwargs)[0]


def _cs_integrand(cosmology):
    return lambda z: sqrt(cosmology.wPerturbations(z)) / cosmology.Hubble(z)


def _friction_integrand(cosmology):
    """Minus ``dF/dz``: the integrand whose cumulative table *is* ``F`` (README §2 (c))."""
    return lambda z: -1.5 * (1.0 + cosmology.wPerturbations(z)) / (1.0 + z)


class _Shared:
    """Built once: the radiation control and both production models on the production grid."""

    references = None

    radiation = None
    radiation_grid = None
    radiation_z = None
    radiation_model = None

    grid = None
    z_nodes = None

    lambdacdm = None
    lambdacdm_payload = None
    lambdacdm_model = None

    qcd = None
    qcd_payload = None
    qcd_seconds = None
    qcd_model = None

    @classmethod
    def build(cls):
        if cls.references is not None:
            return
        cls.references = load_references()

        cls.radiation = _RadiationCosmology()
        cls.radiation_z = np.logspace(
            np.log10(RADIATION_Z_TOP), np.log10(RADIATION_Z_END), num=RADIATION_NODES
        )
        cls.radiation_grid = to_redshift_array(cls.radiation_z)
        radiation_payload = _compute_background(cls.radiation, cls.radiation_grid)
        cls.radiation_model = _offline_model(
            cls.radiation, cls.radiation_grid, radiation_payload
        )

        cls.grid = production_source_grid(
            cls.references["models"]["LambdaCDMModel"]["grid"]["z_init"]
        )
        cls.z_nodes = np.array(cls.grid.as_float_list(), dtype=float)

        cls.lambdacdm = LambdaCDM(store_id=0, units=Mpc_units(), params=Planck2018())
        t0 = time.perf_counter()
        cls.lambdacdm_payload = _compute_background(cls.lambdacdm, cls.grid)
        cls.lambdacdm_seconds = time.perf_counter() - t0
        cls.lambdacdm_model = _offline_model(
            cls.lambdacdm, cls.grid, cls.lambdacdm_payload
        )

        cls.qcd = QCD_Cosmology(
            store_id=0, units=Mpc_units(), params=Planck2018(), max_z=1e20
        )
        t0 = time.perf_counter()
        cls.qcd_payload = _compute_background(cls.qcd, cls.grid)
        cls.qcd_seconds = time.perf_counter() - t0
        cls.qcd_model = _offline_model(cls.qcd, cls.grid, cls.qcd_payload)


class TestRadiationControl(unittest.TestCase):
    """Prompt 04 §3 test 1: both primitives against their exact-radiation closed forms."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_cs_tau_is_tau_over_root_three(self):
        model = self.s.radiation_model
        cosmology = self.s.radiation
        cs_tau = model.functions.cs_tau
        tau = model.functions.tau

        worst_closed_form, worst_ratio, worst_z = 0.0, 0.0, None
        for z in self.s.radiation_z[::17]:
            z = float(z)
            exact = cosmology.cs_tau_exact(z)
            err = difference_error(cs_tau(z), exact)
            if err > worst_closed_form:
                worst_closed_form, worst_z = err, z
            worst_ratio = max(
                worst_ratio, difference_error(cs_tau(z), tau(z) / sqrt(3.0))
            )
        print(
            f"\n[cs_tau] radiation nodes: vs closed form {worst_closed_form:.3e} at "
            f"z = {worst_z:.4g}; vs tau/sqrt(3) {worst_ratio:.3e}"
        )
        self.assertLessEqual(worst_closed_form, RADIATION_REL_TOL)
        self.assertLessEqual(worst_ratio, RADIATION_REL_TOL)

    def test_cs_tau_delta_in_radiation(self):
        model = self.s.radiation_model
        cosmology = self.s.radiation
        cs_tau = model.functions.cs_tau
        z = self.s.radiation_z

        worst_adjacent, worst_z = 0.0, None
        for j in range(0, len(z) - 1, 7):
            z_a, z_b = float(z[j]), float(z[j + 1])
            err = difference_error(
                cs_tau.delta(z_a, z_b), cosmology.cs_tau_delta_exact(z_a, z_b)
            )
            if err > worst_adjacent:
                worst_adjacent, worst_z = err, z_a

        # one off-grid endpoint (the per-object anchor case), and the whole range
        j = len(z) // 2
        z_hi, z_off = float(z[j]), _fraction_point(float(z[j]), float(z[j + 1]))
        err_fraction = difference_error(
            cs_tau.delta(z_hi, z_off), cosmology.cs_tau_delta_exact(z_hi, z_off)
        )
        err_full = difference_error(
            cs_tau.delta(float(z[0]), float(z[-1])),
            cosmology.cs_tau_delta_exact(float(z[0]), float(z[-1])),
        )
        print(
            f"[cs_tau] radiation delta: adjacent {worst_adjacent:.3e} at z = {worst_z:.4g}; "
            f"37% fraction (off-grid) {err_fraction:.3e}; whole range {err_full:.3e}"
        )
        for err in (worst_adjacent, err_fraction, err_full):
            self.assertLessEqual(err, RADIATION_REL_TOL)

    def test_friction_delta_is_two_log_one_plus_z(self):
        """``friction_F.delta(z_a, z_b) = F(z_b) - F(z_a) = 2 log((1+z_b)/(1+z_a))``."""
        model = self.s.radiation_model
        cosmology = self.s.radiation
        friction_F = model.functions.friction_F
        z = self.s.radiation_z

        worst_adjacent, worst_z = 0.0, None
        for j in range(0, len(z) - 1, 7):
            z_a, z_b = float(z[j]), float(z[j + 1])
            err = fabs(
                friction_F.delta(z_a, z_b) - cosmology.friction_delta_exact(z_a, z_b)
            )
            if err > worst_adjacent:
                worst_adjacent, worst_z = err, z_a

        j = len(z) // 2
        z_hi, z_off = float(z[j]), _fraction_point(float(z[j]), float(z[j + 1]))
        err_fraction = fabs(
            friction_F.delta(z_hi, z_off) - cosmology.friction_delta_exact(z_hi, z_off)
        )
        z_a, z_b = float(z[0]), float(z[-1])
        exact_full = cosmology.friction_delta_exact(z_a, z_b)
        err_full = fabs(friction_F.delta(z_a, z_b) - exact_full)
        print(
            f"[friction_F] radiation delta (absolute): adjacent {worst_adjacent:.3e} at "
            f"z = {worst_z:.4g}; 37% fraction (off-grid) {err_fraction:.3e}; whole range "
            f"{err_full:.3e} (F = {exact_full:.6g}, single-limb floor "
            f"{np.spacing(fabs(exact_full)):.3e})"
        )
        for err in (worst_adjacent, err_fraction, err_full):
            self.assertLessEqual(err, RADIATION_FRICTION_ABS_TOL)

    def test_signs_and_anchors(self):
        """``cs_tau`` rises and ``F`` falls towards lower z; the anchors are the conventions."""
        model = self.s.radiation_model
        cosmology = self.s.radiation
        cs_tau = model.functions.cs_tau
        friction_F = model.functions.friction_F
        z_top = float(self.s.radiation_z[0])

        self.assertIsInstance(cs_tau, TablePrimitive)
        self.assertIsInstance(friction_F, TablePrimitive)
        self.assertEqual(cs_tau.label, "cs_tau")
        self.assertEqual(friction_F.label, "friction_F")

        self.assertGreater(cs_tau.delta(1.0e3, 1.0e2), 0.0)
        self.assertLess(cs_tau.delta(1.0e2, 1.0e3), 0.0)
        self.assertLess(friction_F.delta(1.0e3, 1.0e2), 0.0)
        self.assertGreater(friction_F.delta(1.0e2, 1.0e3), 0.0)

        # F is anchored at zero at the top of the grid; cs_tau at the radiation asymptote
        self.assertEqual(friction_F(z_top), 0.0)
        tau_init = model.functions.tau(z_top)
        self.assertEqual(
            cs_tau(z_top), sqrt(cosmology.wPerturbations(z_top)) * tau_init
        )

    def test_negative_sound_speed_is_refused(self):
        """A negative ``c_s^2`` has no sound horizon (prompt 04 §2 item 1)."""
        cosmology = _NegativeSoundSpeedCosmology()
        grid = to_redshift_array(np.logspace(6.0, 1.0, 101))
        with self.assertRaises(ValueError) as caught:
            _compute_background(cosmology, grid)
        message = str(caught.exception)
        self.assertIn("wPerturbations", message)
        self.assertIn("_NegativeSoundSpeedCosmology", message)
        self.assertIn("sound horizon", message)


class TestProductionModels(unittest.TestCase):
    """Prompt 04 §3 tests 2 and 3: both primitives against the prompt 01/02 references."""

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def _checkpoint_errors(self, model, block):
        cs_tau = model.functions.cs_tau
        friction_F = model.functions.friction_F
        z_top = block["z_top"]

        worst_cs, worst_cs_z = 0.0, None
        worst_F, worst_F_z, worst_F_abs = 0.0, None, 0.0
        for checkpoint, cs_reference, F_reference in zip(
            block["checkpoints"],
            block["cs_tau_minus_top"],
            block["friction_F_minus_top"],
        ):
            z = checkpoint["z"]
            self.assertEqual(cs_tau.table.node_index(z), checkpoint["index"])
            if cs_reference == 0.0:
                self.assertEqual(cs_tau.delta(z_top, z), 0.0)
            else:
                err = difference_error(cs_tau.delta(z_top, z), cs_reference)
                if err > worst_cs:
                    worst_cs, worst_cs_z = err, z
            if F_reference == 0.0:
                self.assertEqual(friction_F.delta(z_top, z), 0.0)
            else:
                got = friction_F.delta(z_top, z)
                # the reference is negative: F falls towards lower z
                self.assertLess(got, 0.0)
                err = difference_error(got, F_reference)
                if err > worst_F:
                    worst_F, worst_F_z = err, z
                worst_F_abs = max(worst_F_abs, fabs(got - F_reference))
        return worst_cs, worst_cs_z, worst_F, worst_F_z, worst_F_abs

    def _short_baseline_errors(self, model, cosmology, intervals):
        """``(z_hi, cs_tau rel full, cs_tau rel fraction, F abs full, F abs fraction)`` rows."""
        cs_tau = model.functions.cs_tau
        friction_F = model.functions.friction_F
        break_points = (
            cosmology.integration_break_points(self.s.grid.min.z, self.s.grid.max.z)
            if hasattr(cosmology, "integration_break_points")
            else None
        )
        cs_integrand = _cs_integrand(cosmology)
        friction_integrand = _friction_integrand(cosmology)

        rows = []
        for index in intervals:
            z_hi = float(self.s.z_nodes[index])
            z_lo = float(self.s.z_nodes[index + 1])
            z_fraction = _fraction_point(z_hi, z_lo)
            self.assertIsNone(cs_tau.table.node_index(z_fraction))

            row = [z_hi]
            for z_end in (z_lo, z_fraction):
                row.append(
                    difference_error(
                        cs_tau.delta(z_hi, z_end),
                        _quad_reference(
                            cosmology, z_end, z_hi, cs_integrand, break_points
                        ),
                    )
                )
            for z_end in (z_lo, z_fraction):
                row.append(
                    fabs(
                        friction_F.delta(z_hi, z_end)
                        - _quad_reference(
                            cosmology, z_end, z_hi, friction_integrand, break_points
                        )
                    )
                )
            rows.append(tuple(row))
        return rows

    def _json_short_baseline_intervals(self, block):
        return [record["index"] for record in block["short_baseline"]]

    def test_lambdacdm_checkpoints(self):
        """Prompt 04 §3 test 2 / README §6: cs_tau <= 2e-14, F <= 1e-13 relative."""
        block = self.s.references["models"]["LambdaCDMModel"]
        worst_cs, cs_z, worst_F, F_z, worst_F_abs = self._checkpoint_errors(
            self.s.lambdacdm_model, block
        )
        print(
            f"\n[cs_tau] LambdaCDM checkpoints: max rel err {worst_cs:.3e} at z = {cs_z:.4g}"
        )
        print(
            f"[friction_F] LambdaCDM checkpoints: max rel err {worst_F:.3e} at z = {F_z:.4g} "
            f"(absolute {worst_F_abs:.3e})"
        )
        self.assertLessEqual(worst_cs, LAMBDACDM_CS_TAU_REL_TOL)
        self.assertLessEqual(worst_F, FRICTION_REL_TOL)

    def test_lambdacdm_short_baselines(self):
        block = self.s.references["models"]["LambdaCDMModel"]
        rows = self._short_baseline_errors(
            self.s.lambdacdm_model,
            self.s.lambdacdm,
            self._json_short_baseline_intervals(block),
        )
        for z_hi, cs_full, cs_fraction, F_full, F_fraction in rows:
            print(
                f"[cs_tau] LambdaCDM short baseline at z = {z_hi:.4g}: one interval "
                f"{cs_full:.3e}, 37% fraction {cs_fraction:.3e} | friction absolute "
                f"{F_full:.3e} / {F_fraction:.3e}"
            )
            self.assertLessEqual(cs_full, CS_TAU_SHORT_BASELINE_REL_TOL)
            self.assertLessEqual(cs_fraction, CS_TAU_SHORT_BASELINE_REL_TOL)
            self.assertLessEqual(F_full, FRICTION_SHORT_BASELINE_ABS_TOL)
            self.assertLessEqual(F_fraction, FRICTION_SHORT_BASELINE_ABS_TOL)

    def test_qcd_checkpoints(self):
        """Prompt 04 §3 test 3: within 3x the reference's own recorded floor."""
        block = self.s.references["models"]["QCDModel"]
        floor = self.s.references["convergence"]["models"]["QCDModel"]["branch+knots"][
            "cs_tau"
        ]["json_vs_reference_max_rel"]
        worst_cs, cs_z, worst_F, F_z, worst_F_abs = self._checkpoint_errors(
            self.s.qcd_model, block
        )
        print(
            f"[cs_tau] QCD checkpoints: max rel err {worst_cs:.3e} at z = {cs_z:.4g}; "
            f"reference floor {floor:.3e} (threshold {QCD_FLOOR_FACTOR * floor:.3e})"
        )
        print(
            f"[friction_F] QCD checkpoints: max rel err {worst_F:.3e} at z = {F_z:.4g} "
            f"(absolute {worst_F_abs:.3e})"
        )
        self.assertLessEqual(worst_cs, QCD_FLOOR_FACTOR * floor)
        self.assertLessEqual(worst_F, FRICTION_REL_TOL)

    def test_qcd_short_baselines_including_the_transitions(self):
        """
        Prompt 04 §3 test 3: the JSON's three short baselines *and* the production intervals
        that carry the equation-of-state break points -- the two Hubble branch boundaries, the
        c_s^2 kink at EOS_T_LO, and the steepest c_s^2 transition -- read from prompt 02's
        geometry block.
        """
        block = self.s.references["models"]["QCDModel"]
        geometry = self.s.references["convergence"]["geometry"]["QCDModel"]

        labelled = {
            record["index"]: f"JSON baseline z = {record['z_node_hi']:.4g}"
            for record in block["short_baseline"]
        }
        for boundary in geometry["branch_boundaries"]:
            labelled.setdefault(boundary["interval_index"], "")
            labelled[boundary["interval_index"]] += f" [{boundary['name']}]"
        transition_index = geometry["cs2_transition"]["interval_index"]
        labelled.setdefault(transition_index, "")
        labelled[transition_index] += " [steepest c_s^2 transition]"

        indices = sorted(labelled)
        # the transition intervals really are among those checked
        for boundary in geometry["branch_boundaries"]:
            self.assertIn(boundary["interval_index"], indices)
        self.assertIn(transition_index, indices)

        rows = self._short_baseline_errors(self.s.qcd_model, self.s.qcd, indices)
        for index, (z_hi, cs_full, cs_fraction, F_full, F_fraction) in zip(
            indices, rows
        ):
            print(
                f"[cs_tau] QCD interval {index} at z = {z_hi:.5g}{labelled[index]}: one interval "
                f"{cs_full:.3e}, 37% fraction {cs_fraction:.3e} | friction absolute "
                f"{F_full:.3e} / {F_fraction:.3e}"
            )
            self.assertLessEqual(cs_full, CS_TAU_SHORT_BASELINE_REL_TOL)
            self.assertLessEqual(cs_fraction, CS_TAU_SHORT_BASELINE_REL_TOL)
            self.assertLessEqual(F_full, FRICTION_SHORT_BASELINE_ABS_TOL)
            self.assertLessEqual(F_fraction, FRICTION_SHORT_BASELINE_ABS_TOL)

    def test_build_cost(self):
        """Recorded, not thresholded: the two new tables against the tau table of prompt 03."""
        for name, payload, seconds in (
            ("LambdaCDM", self.s.lambdacdm_payload, self.s.lambdacdm_seconds),
            ("QCD", self.s.qcd_payload, self.s.qcd_seconds),
        ):
            print(
                f"[tables] {name} build: tau {payload['data'].RHS_evaluations}, cs_tau "
                f"{payload['cs_tau_evaluations']}, friction_F "
                f"{payload['friction_F_evaluations']} integrand evaluations; all three tables "
                f"{payload['data'].compute_time:.3f} s, whole compute_background {seconds:.3f} s"
            )


class TestPayloadSchemaAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_payload_shape(self):
        for payload in (self.s.lambdacdm_payload, self.s.qcd_payload):
            for key in ("cs_tau_hi_sample", "cs_tau_lo_sample", "friction_F_sample"):
                self.assertEqual(len(payload[key]), len(self.s.grid))
            self.assertEqual(payload["cs_tau_order"], CS_TAU_GAUSS_ORDER)
            self.assertEqual(payload["friction_F_order"], FRICTION_F_GAUSS_ORDER)
            for key in ("cs_tau_evaluations", "friction_F_evaluations"):
                self.assertGreaterEqual(
                    payload[key], CS_TAU_GAUSS_ORDER * (len(self.s.grid) - 1)
                )
        # the tau table's evaluation count is untouched by the two new tables: IntegrationData
        # has one counter and prompt 03's test pins it ([04-background-rhs-evaluations-count])
        self.assertEqual(
            self.s.lambdacdm_payload["data"].RHS_evaluations,
            CS_TAU_GAUSS_ORDER * (len(self.s.grid) - 1),
        )

    def test_model_functions_still_constructs_with_thirteen_positional_arguments(self):
        """
        Prompt 04 §3 test 4 / README §5 rule 7: every existing stand-in builds a ModelFunctions
        with the thirteen original fields, so the two new ones must have defaults.
        """
        thirteen = tuple(lambda z: 0.0 for _ in range(13))
        functions = ModelFunctions(*thirteen)
        self.assertIsNone(functions.cs_tau)
        self.assertIsNone(functions.friction_F)
        self.assertEqual(len(ModelFunctions._fields), 15)
        self.assertEqual(ModelFunctions._fields[-2:], ("cs_tau", "friction_F"))

    def test_background_model_value_keeps_constructing(self):
        value = BackgroundModelValue(
            None,
            self.s.grid[0],
            Hubble=1.0,
            wBackground=0.0,
            wPerturbations=0.0,
            rho=1.0,
            tau=2.0,
            T_photon=1.0,
            d_lnH_dz=0.0,
        )
        self.assertIsNone(value.cs_tau)
        self.assertIsNone(value.cs_tau_lo)
        self.assertIsNone(value.friction_F)

    def test_persisted_values_round_trip_exactly_in_Mpc_units(self):
        """
        Prompt 04 §3 test 5 (RECONCILIATION.md §2 item 12): Mpc = 1.0, so /Mpc then *Mpc is
        exact for the sound horizon; friction_F is dimensionless and is stored unscaled.
        """
        for cosmology, model in (
            (self.s.lambdacdm, self.s.lambdacdm_model),
            (self.s.qcd, self.s.qcd_model),
        ):
            Mpc = cosmology.units.Mpc
            self.assertEqual(Mpc, 1.0)
            for value in model.values:
                value: BackgroundModelValue
                self.assertEqual((value.cs_tau / Mpc) * Mpc, value.cs_tau)
                self.assertEqual((value.cs_tau_lo / Mpc) * Mpc, value.cs_tau_lo)
                # dimensionless: whatever the unit system, friction_F is written as it stands
                self.assertIs(type(value.friction_F), float)
                self.assertLessEqual(
                    fabs(value.cs_tau_lo), np.spacing(fabs(value.cs_tau))
                )

    def test_reconstruction_from_values(self):
        """
        ``_create_functions`` rebuilds both tables from the persisted limbs with no quadrature.
        ``cs_tau`` is reconstructed exactly (both limbs are stored); ``friction_F`` keeps only
        its high limb by design, so the rebuilt table agrees with the built one to the dropped
        limb, which is below an ulp of F.
        """
        for model, payload in (
            (self.s.lambdacdm_model, self.s.lambdacdm_payload),
            (self.s.qcd_model, self.s.qcd_payload),
        ):
            cs_table = model.functions.cs_tau.table
            self.assertEqual(cs_table.evaluations, 0)
            self.assertTrue(
                np.array_equal(cs_table.hi, np.asarray(payload["cs_tau_hi_sample"]))
            )
            self.assertTrue(
                np.array_equal(cs_table.lo, np.asarray(payload["cs_tau_lo_sample"]))
            )

            friction_table = model.functions.friction_F.table
            self.assertEqual(friction_table.evaluations, 0)
            self.assertTrue(
                np.array_equal(
                    friction_table.hi, np.asarray(payload["friction_F_sample"])
                )
            )
            self.assertTrue(
                np.array_equal(friction_table.lo, np.zeros(len(model.values)))
            )
            self.assertEqual(cs_table.order, CS_TAU_GAUSS_ORDER)
            self.assertEqual(friction_table.order, FRICTION_F_GAUSS_ORDER)
            # the break-point set is the cosmology's, exactly as for tau
            self.assertEqual(
                cs_table.break_points.size, model.functions.tau.table.break_points.size
            )
            self.assertEqual(
                friction_table.break_points.size,
                model.functions.tau.table.break_points.size,
            )

    def test_a_value_without_the_new_limbs_is_refused(self):
        """A pre-prompt-04 value cannot take part in a rebuilt table, and says so."""
        values = BackgroundModel.values_from_payload(
            self.s.grid, self.s.lambdacdm_payload
        )
        values[17]._cs_tau = None
        model = BackgroundModel(
            payload={
                "store_id": None,
                "data": self.s.lambdacdm_payload["data"],
                "solver": None,
                "values": values,
            },
            solver_labels={},
            cosmology=self.s.lambdacdm,
            atol=None,
            rtol=None,
            z_sample=self.s.grid,
        )
        with self.assertRaises(RuntimeError) as caught:
            model.functions
        self.assertIn("cs_tau", str(caught.exception))
        self.assertIn("regenerated", str(caught.exception))


class TestFrictionODEComparison(unittest.TestCase):
    """
    Prompt 04 §3 test 6: the friction ODE this table replaces is the inaccurate one.

    ``TkWKBIntegration`` used to integrate ``_friction_RHS`` (relocated to the top of this
    module by prompt 07) with DOP853 at the production tolerances from the numeric hand-over
    down the source grid, starting from ``F(z_init) = 0``. Review
    §12.3 measures its error as 2.3e-7 (k = 1e5) to 4.1e-7 (k = 3e8) *of the amplitude*, i.e.
    absolute in F, set by ``rtol``. The table is at 1e-14 absolute (the tests above), so the
    difference measured here is the ODE's error.
    """

    @classmethod
    def setUpClass(cls):
        _Shared.build()
        cls.s = _Shared

    def test_friction_ode_is_the_inaccurate_one(self):
        cosmology = self.s.lambdacdm
        model = self.s.lambdacdm_model
        friction_F = model.functions.friction_F

        z_init = horizon_exit_z(cosmology, FRICTION_ODE_K_INV_MPC, efolds_subh=3.0)
        z_samples = [float(z) for z in self.s.z_nodes if z < z_init]
        self.assertGreater(len(z_samples), 100)

        k = wavenumber(store_id=0, k_inv_Mpc=FRICTION_ODE_K_INV_MPC, units=Mpc_units())
        with NumericIntegrationSupervisor(
            k, z_init, z_samples[-1], "test_friction_ode", delta_logz=None
        ) as supervisor:
            sol = solve_ivp(
                _friction_RHS,
                method="DOP853",
                t_span=(z_init, z_samples[-1]),
                y0=[0.0],
                t_eval=z_samples,
                atol=FRICTION_ODE_ATOL,
                rtol=FRICTION_ODE_RTOL,
                args=(model, float(k.k), supervisor),
            )
        self.assertTrue(sol.success)

        worst_abs, worst_z, worst_rel = 0.0, None, 0.0
        for z, F_ode in zip(sol.t, sol.y[0]):
            F_table = friction_F.delta(z_init, float(z))
            err = fabs(float(F_ode) - F_table)
            if err > worst_abs:
                worst_abs, worst_z = err, float(z)
                worst_rel = err / fabs(F_table) if F_table != 0.0 else 0.0
        print(
            f"\n[friction_F] LambdaCDM, k = {FRICTION_ODE_K_INV_MPC:.0e}/Mpc, z_init = z_e3 = "
            f"{z_init:.5g} down to {z_samples[-1]}: DOP853 _friction_RHS at "
            f"(atol, rtol) = ({FRICTION_ODE_ATOL:.0e}, {FRICTION_ODE_RTOL:.0e}) differs from "
            f"friction_F.delta by {worst_abs:.3e} absolute in F -- i.e. relative in the "
            f"amplitude exp(F) -- at z = {worst_z:.4g} ({worst_rel:.3e} relative in F), "
            f"{supervisor.RHS_evaluations} RHS evaluations. Review §12.3 measures 2.3e-7."
        )
        self.assertGreaterEqual(worst_abs, FRICTION_ODE_ERROR_MIN)
        self.assertLessEqual(worst_abs, FRICTION_ODE_ERROR_MAX)

        # and the ODE's own samples are the ones that move: the table's F at those nodes is the
        # break-aware Gauss value, which the checkpoint tests scored against the references
        self.assertLess(friction_F.delta(z_init, z_samples[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
