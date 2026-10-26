"""
Tests for ``prompts/levin-phase-floor`` prompt 06: the producers' stored phase.

**What changed.** Each ``GkWKBIntegration`` and ``TkWKBIntegration`` value used to store its phase
``theta = -(k L.delta(z_init, z) + rho.delta(z_init, z)) + deltaTheta`` only as a cycle count and a
remainder, formed from one double of size ``k tau`` reduced against the double ``TWO_PI``, with
``deltaTheta`` added afterwards in double. The small part of the phase -- ``rho``, at most ~1.5e-3
rad, and ``deltaTheta`` -- survived only to ``ulp(k tau)``, and each consumer rebuilt ``phi`` as
``div * TWO_PI + mod - sign k L.delta``, a difference of two numbers of size ``k tau``
(``[02-consumer-phi-below-the-storage-granularity]``). Now:

- ``WKB_phase_function`` forms each sample's phase exactly (the leading table's limbs and partials,
  ``k`` exact, ``rho.delta`` added exactly) and reduces it exactly against 2pi
  (``primitive_phase.exact_leading_phase``, ``reduce_exactly_mod_2pi``);
- the producers add ``deltaTheta`` to that exact phase, reduce again, and store
  ``theta_residual = deltaTheta - rho.delta`` from its two small terms;
- ``TkSourceFunctions`` takes ``phi_T = r``; ``GkSourcePolicyData._build_phase`` forms
  ``phi_G = r + k tau.delta(z_s, z_init)``, unwrapped by whole cycles and checked against the
  rectifier's shifts.

**The tests**, numbered as prompt 06 §3:

1. the stored remainder is exact against 2pi, and the cycle count exact, at four ``k`` with
   ``z_init`` on a node and off one, in each sector; plus the helper's own normalisation of a
   remainder that rounds to ``-TWO_PI`` (§9 point 6) and its agreement with ``mpmath``;
2. ``theta_residual`` is the small part, and with the exact large term added back it reproduces
   the stored remainder;
3. ``phi_G`` and ``phi_T``, as the consumers now form them, are noise-free at ``k = 3e8`` and
   ``|theta| >= 1e11`` (04's ``window_rms``); the old construction's numbers are printed, unscored;
4. on a ``GkSource`` assembled by ``assemble_GkSource_values`` from producer values whose
   ``deltaTheta`` wraps (twice), the unwrap of ``phi_G`` and the rectifier agree at every
   sample, and ``phi_G`` is continuous across the wraps;
5. both consumers' ``raw_theta`` reproduces the exact phase at the samples;
6. nothing else moves: (a) ``PrimitivePhase`` is bit-identical to its own formulas, and its
   reducer and the producers' agree to the bit; (b) the background tables, ``rho`` and the numeric
   producers; (c) ``QuadSource``;
7. end to end, ``test_phase_groups``' realistic fixture in regime ``G q r`` with both factors on
   stand-ins carrying the residual.

**Stand-ins.** The background is ``test_gk_wkb_phase``'s LambdaCDM model with its three
``TablePrimitive`` tables on the production source grid. The producers run as production runs
them: ``WKB_phase_function`` through its undecorated ``_function``, then the real ``store()`` with
``ray`` stood in (``test_tk_wkb_phase``'s ``_FakeRay``). ``assemble_GkSource_values`` runs through
its own ``_function``. No Ray, no datastore.
"""

import ast
import contextlib
import io
import sys
import unittest
from fractions import Fraction
from math import atan2, cos, expm1, fabs, log1p, sin, sqrt
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mpmath
import numpy as np

import ComputeTargets.GkWKBIntegration  # noqa: F401  (for sys.modules, below)
import ComputeTargets.TkWKBIntegration  # noqa: F401
from ComputeTargets.GkSource import GkSourceValue, assemble_GkSource_values
from ComputeTargets.GkSourcePolicyData import GK_PHASE_SIGN, _build_phase
from ComputeTargets.GkWKBIntegration import GkWKBIntegration
from ComputeTargets.TkSourceFunctions import TK_PHASE_SIGN, TkSourceFunctions
from ComputeTargets.TkWKBIntegration import TkWKBIntegration
from ComputeTargets.WKB_Gk import Gk_d_ln_omegaEff_dz, Gk_omegaEff_sq
from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz, Tk_omegaEff_sq
from ComputeTargets.handover import HANDOVER_OVERLAP_NODES
from ComputeTargets.phase_residual import cached_phase_residual
from ComputeTargets.primitive_phase import (
    PrimitivePhase,
    _remainder_against_two_pi,
    build_phi_samples,
    exact_leading_phase,
    reduce_exactly_mod_2pi,
)
from ComputeTargets.tests import test_phase_groups
from ComputeTargets.tests.test_gk_wkb_phase import (
    _KExit,
    _Proxy,
    _Shared,
    _run_Gk,
    _run_Tk,
)
from ComputeTargets.tests.test_phase_groups import Case, max_relative_residual
from ComputeTargets.tests.test_tk_source_functions import (
    FakeNumericValue,
    FakeTkNumeric,
    PRODUCTION_SAMPLES_PER_LOG10Z,
    log_grid,
)
from ComputeTargets.tests.test_tk_wkb_phase import _FakeRay, _Solver
from CosmologyConcepts import redshift, redshift_array
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI
from Quadrature.integrators.WKB_phase_function import (
    PHASE_SOLVER_LABEL,
    nearest_table_node,
)

REPO_ROOT = Path(__file__).parents[2]

# the modules, not the classes ComputeTargets re-exports under the same names: store() is run with
# their ``ray`` stood in
_GKWKB_MODULE = sys.modules["ComputeTargets.GkWKBIntegration"]
_TKWKB_MODULE = sys.modules["ComputeTargets.TkWKBIntegration"]

EPS = 2.0**-52
ULP_TWO_PI = float(np.spacing(TWO_PI))

# test 1: the four wavenumbers of 05's and 05a's exactness tests, and the off-node anchor's place
EXACTNESS_K = (1.0e5, 3.0e6, 1.0e8, 3.0e8)
OFF_NODE_FRACTION = 0.37
# the anchor is this many residual-table nodes below the top of the WKB region, taken as the
# largest node where k L(z) >= ANCHOR_K_L (L = tau or cs_tau): 20 e-folds' worth of phase
ANCHOR_NODE = 4
ANCHOR_K_L = 100.0
# the object's offset deltaTheta, chosen through its initial data (B = 1)
DELTA_THETA = 2.0

# test 3: 04's phi metric (docs/levin-phase-floor/04/scripts/check_phases.py, window_rms)
PHI_WINDOW = 9
PHI_DEG = 4
NOISE_K = 3.0e8
NOISE_MIN_THETA = 1.0e11
NOISE_BOUND_ULP = 1.0e-3
# the GkSource of test 3: response redshift, and how many source redshifts below the WKB top
NOISE_Z_RESPONSE = 0.1
NOISE_SOURCES = 160

# test 4: a GkSource whose deltaTheta wraps; k and the band of k tau(z_s) it is drawn from. At
# k = 3e8 and z_R = 0.1, |theta| ~ 4e12 rad, so test 3's bound (|theta| >= 1e11) applies; the
# band keeps k Delta tau between neighbouring sources below pi, as the rectifier needs
WRAP_K = 3.0e8
WRAP_KTAU = (25.0, 80.0)
WRAP_DELTA_START = 3.0
WRAP_DELTA_STEP = 0.35
# each object's z_init: this many grid nodes below its source redshift
WRAP_INIT_NODES_BELOW = 2
WRAP_Z_RESPONSE = 0.1

# the mpmath reference context of tests 1, 2 and 5: 60 digits, private
_MP = mpmath.MPContext()
_MP.dps = 60
_MP_TWO_PI = 2 * _MP.pi


# ------------------------------------------------------------------------------------------------
# references, independent of the helpers under test
# ------------------------------------------------------------------------------------------------


def _mp_leading_delta(leading, z_a: float, z_b: float):
    """``leading.delta(z_a, z_b)`` exactly, in mpmath, from the table's own limbs and partials."""
    j_a, j_b, p_a, p_b = leading.delta_parts(z_a, z_b)
    hi_a, lo_a = leading.node_limbs(j_a)
    hi_b, lo_b = leading.node_limbs(j_b)
    return (
        (_MP.mpf(hi_b) - _MP.mpf(hi_a))
        + (_MP.mpf(lo_b) - _MP.mpf(lo_a))
        + (_MP.mpf(p_b) - _MP.mpf(p_a))
    )


def _mp_reduce(theta):
    """``(div, rem)`` of an mpmath phase against 2pi itself: ``div = ceil(theta/2pi)``,
    ``rem = theta - div 2pi`` in ``(-2pi, 0]``."""
    div = int(_MP.ceil(theta / _MP_TWO_PI))
    rem = theta - div * _MP_TWO_PI
    if rem > 0:
        rem -= _MP_TWO_PI
        div += 1
    if rem <= -_MP_TWO_PI:
        rem += _MP_TWO_PI
        div -= 1
    return div, rem


def _cyclic(a, b) -> float:
    """The distance between two phases on the circle, in rad (mpmath or float)."""
    d = _MP.fmod(_MP.mpf(a) - _MP.mpf(b), _MP_TWO_PI)
    d = abs(d)
    return float(min(d, _MP_TWO_PI - d))


def window_rms(u, y):
    """04's ``window_rms``: the RMS residual about a degree-4 fit over each window of 9
    consecutive samples, in u. Returns ``(rms per window, the index of each window's start)``.
    """
    out = []
    starts = []
    for i in range(0, len(u) - PHI_WINDOW + 1):
        uu = np.asarray(u[i : i + PHI_WINDOW], dtype=float)
        yy = np.asarray(y[i : i + PHI_WINDOW], dtype=float)
        t = (2.0 * uu - (uu[0] + uu[-1])) / (uu[-1] - uu[0])
        coef = np.polynomial.polynomial.polyfit(t, yy, PHI_DEG)
        res = yy - np.polynomial.polynomial.polyval(t, coef)
        out.append(float(np.sqrt(np.mean(res * res))))
        starts.append(i)
    return out, starts


# ------------------------------------------------------------------------------------------------
# the producers, run as production runs them
# ------------------------------------------------------------------------------------------------


class _Model:
    """The LambdaCDM background with tables on the production source grid, built once."""

    model = None
    units = None
    z_nodes = None

    @classmethod
    def build(cls):
        if cls.model is None:
            _Shared.build()
            cls.model = _Shared.lambdacdm
            cls.units = cls.model.cosmology.units
            cls.z_nodes = np.asarray(_Shared.z_nodes, dtype=float)
        return cls


def _leading(model, sector: str):
    return model.functions.tau if sector == "Gk" else model.functions.cs_tau


def _wkb_nodes(model, k: float, sector: str):
    """The nodes of the residual table (descending) at which the object is well inside the WKB
    region, ``k L(z) >= ANCHOR_K_L``, ``L`` the leading primitive (``tau`` or ``cs_tau``).
    """
    leading = _leading(model, sector)
    rho, _ = cached_phase_residual(
        model, k, leading.table.z_nodes, sector, store_id=None
    )
    return [float(z) for z in rho.z_nodes if k * leading(float(z)) >= ANCHOR_K_L]


def _anchor(model, k: float, sector: str, on_node: bool) -> float:
    """``z_init``: a node of the residual table a few nodes below the top of the WKB region, or
    37% of the way in ``log(1+z)`` to the next node down."""
    nodes = _wkb_nodes(model, k, sector)
    z_hi = nodes[ANCHOR_NODE]
    if on_node:
        return z_hi
    z_lo = nodes[ANCHOR_NODE + 1]
    width = log1p((z_hi - z_lo) / (1.0 + z_lo))
    return z_lo + (1.0 + z_lo) * expm1((1.0 - OFF_NODE_FRACTION) * width)


def _initial_data(model, k: float, sector: str, z_init: float, delta: float):
    """Initial data for which the store's algebra gives ``deltaTheta = delta`` and ``B = 1``."""
    if sector == "Gk":
        omega_sq = Gk_omegaEff_sq(model, k, z_init)
        dln = Gk_d_ln_omegaEff_dz(model, k, z_init)
        extra = model.functions.epsilon(z_init) / (1.0 + z_init)
    else:
        omega_sq = Tk_omegaEff_sq(model, k, z_init)
        dln = Tk_d_ln_omegaEff_dz(model, k, z_init)
        cs2 = model.functions.wPerturbations(z_init)
        extra = (model.functions.epsilon(z_init) - 3.0 * (1.0 + cs2)) / (1.0 + z_init)
    sqrt_omega = sqrt(sqrt(omega_sq))
    value = sin(delta) / sqrt_omega
    derivative = cos(delta) * sqrt_omega - (value / 2.0) * (dln + extra)
    return value, derivative


def _delta_theta(model, k: float, sector: str, z_init: float, value, derivative):
    """The store's ``deltaTheta``, by its algebra (``test_gk_wkb_phase.store_algebra`` and
    ``test_tk_wkb_phase.tk_store_algebra`` are the verbatim replicas)."""
    if sector == "Gk":
        from ComputeTargets.tests.test_gk_wkb_phase import store_algebra

        return store_algebra(
            Gk_omegaEff_sq(model, k, z_init),
            Gk_d_ln_omegaEff_dz(model, k, z_init),
            model.functions.epsilon(z_init),
            z_init,
            value,
            derivative,
        )[1]
    from ComputeTargets.tests.test_tk_wkb_phase import tk_store_algebra

    return tk_store_algebra(
        Tk_omegaEff_sq(model, k, z_init),
        Tk_d_ln_omegaEff_dz(model, k, z_init),
        model.functions.epsilon(z_init),
        model.functions.wPerturbations(z_init),
        z_init,
        value,
        derivative,
    )[1]


_SERIALS = iter(range(1000, 10**9))


def _redshifts(z_values, first_id: int):
    """``redshift`` objects with distinct store ids, in the order given."""
    return [redshift(store_id=first_id + i, z=float(z)) for i, z in enumerate(z_values)]


def build_Gk(model, units, k, z_source, z_init, samples, value=0.0, derivative=1.0):
    """
    A stored ``GkWKBIntegration``: ``WKB_phase_function`` then the real ``store()``. Pure WKB
    when ``z_init`` is ``None`` (anchored at ``z_source``), numerically seeded otherwise. Returns
    ``(object, payload)``.
    """
    anchor = z_source.z if z_init is None else z_init
    payload = _run_Gk(model, units, k, anchor, samples)
    obj = GkWKBIntegration(
        payload=None,
        solver_labels={PHASE_SOLVER_LABEL: _Solver(PHASE_SOLVER_LABEL)},
        model=_Proxy(model, units),
        k=_KExit(k, units, z_exit_subh_e3=1.0e300),
        z_source=z_source,
        numeric_serial=None if z_init is None else next(_SERIALS),
        z_init=z_init,
        G_init=value,
        Gprime_init=derivative,
        z_sample=samples,
    )
    obj._compute_ref = payload
    with mock.patch.object(_GKWKB_MODULE, "ray", _FakeRay):
        if obj.store() is not True:
            raise RuntimeError("build_Gk: store() did not report success")
    return obj, payload


def build_Tk(model, units, k, z_init, samples, value, derivative):
    """A stored ``TkWKBIntegration``. Returns ``(object, payload)``."""
    payload = _run_Tk(model, units, k, z_init, samples)
    obj = TkWKBIntegration(
        payload=None,
        solver_labels={PHASE_SOLVER_LABEL: _Solver(PHASE_SOLVER_LABEL)},
        model=_Proxy(model, units),
        k=_KExit(k, units, z_exit_subh_e3=1.0e300),
        numeric_serial=next(_SERIALS),
        z_init=z_init,
        T_init=value,
        Tprime_init=derivative,
        z_sample=samples,
    )
    obj._compute_ref = payload
    with mock.patch.object(_TKWKB_MODULE, "ray", _FakeRay):
        if obj.store() is not True:
            raise RuntimeError("build_Tk: store() did not report success")
    return obj, payload


class _Objects:
    """
    One stored object per (sector, k, anchor) for tests 1, 2, 3 and 5, sampled at every node of
    the production grid below its ``z_init``. Built once, lazily.
    """

    cache = {}

    @classmethod
    def get(cls, sector: str, k: float, on_node: bool):
        key = (sector, k, on_node)
        if key not in cls.cache:
            m = _Model.build()
            z_init = _anchor(m.model, k, sector, on_node)
            z_samples = [float(z) for z in m.z_nodes if z < z_init]
            samples = redshift_array(_redshifts(z_samples, 1))
            value, derivative = _initial_data(m.model, k, sector, z_init, DELTA_THETA)
            if sector == "Gk":
                # a numerically seeded object: its source one node above the anchor
                above = [float(z) for z in m.z_nodes if z > z_init][-1]
                z_source = redshift(store_id=0, z=above)
                obj, payload = build_Gk(
                    m.model, m.units, k, z_source, z_init, samples, value, derivative
                )
            else:
                obj, payload = build_Tk(
                    m.model, m.units, k, z_init, samples, value, derivative
                )
            delta = _delta_theta(m.model, k, sector, z_init, value, derivative)
            cls.cache[key] = SimpleNamespace(
                sector=sector,
                k=k,
                on_node=on_node,
                z_init=z_init,
                obj=obj,
                payload=payload,
                delta=delta,
                leading=_leading(m.model, sector),
                z=z_samples,
            )
        return cls.cache[key]


def _anchor_name(on_node: bool) -> str:
    return "on a node" if on_node else "off a node"


def _exact_theta(o, i: int):
    """The true phase of sample ``i`` of object ``o``, in mpmath: ``-k L.delta(z_init, z) -
    rho.delta(z_init, z) + deltaTheta``, from the same limbs, partials, rho and deltaTheta.
    """
    z = o.z[i]
    large = -_MP.mpf(o.k) * _mp_leading_delta(o.leading, o.z_init, z)
    return large - _MP.mpf(o.payload["rho_delta_sample"][i]) + _MP.mpf(o.delta)


# ------------------------------------------------------------------------------------------------
# 1. the stored remainder is exact
# ------------------------------------------------------------------------------------------------


class TestStoredRemainderIsExact(unittest.TestCase):
    def test_remainder_and_cycle_count_against_2pi(self):
        lines = []
        for sector in ("Gk", "Tk"):
            for on_node in (True, False):
                for k in EXACTNESS_K:
                    with self.subTest(sector=sector, anchor=_anchor_name(on_node), k=k):
                        o = _Objects.get(sector, k, on_node)
                        self.assertNotEqual(o.delta, 0.0)
                        worst = 0.0
                        largest = 0.0
                        for i, value in enumerate(o.obj.values):
                            theta = _exact_theta(o, i)
                            div, rem = _mp_reduce(theta)
                            largest = max(largest, abs(float(theta)))
                            if value.theta_div_2pi == div:
                                err = float(abs(_MP.mpf(value.theta_mod_2pi) - rem))
                            else:
                                # only the normalisation of a remainder that rounds to -TWO_PI
                                self.assertEqual(value.theta_div_2pi, div - 1)
                                self.assertEqual(value.theta_mod_2pi, 0.0)
                                err = float(abs(rem + _MP_TWO_PI))
                            self.assertGreater(value.theta_mod_2pi, -TWO_PI)
                            self.assertLessEqual(value.theta_mod_2pi, 0.0)
                            worst = max(worst, err)
                        lines.append(
                            f"   {sector} z_init {_anchor_name(on_node)}, k={k:.0e}: "
                            f"{len(o.obj.values)} samples, max |theta| {largest:.3e} rad, "
                            f"worst |mod - exact| {worst:.3e} rad = "
                            f"{worst / ULP_TWO_PI:.3f} ulp(2pi)"
                        )
                        self.assertLessEqual(worst, 2.0 * ULP_TWO_PI)
        print("\n[06 test 1] stored remainder against the exact one, against 2pi:")
        print("\n".join(lines))

    def test_the_helper_normalises_a_remainder_that_rounds_to_minus_two_pi(self):
        """§9 point 6: a remainder in ``(-2pi, -TWO_PI]`` would round to ``-TWO_PI``. It is
        stored as 0.0 with the cycle count one further from zero, ``div - 1``, the same point of
        the circle to within ``2pi - TWO_PI``."""
        cases = [
            Fraction(1, 2**70),
            Fraction(3, 2**60),
            # a negative phase: (div - 1) 2pi + d with d tiny, as an exact dyadic rational
            None,
        ]
        # -2pi * 5 rounded up to the dyadic just above it (2^-120 resolution, inside the 60-digit
        # reference's precision): ceil puts it one cycle up, with a remainder within 2^-120 of
        # -2pi
        theta = Fraction(int(_MP.ceil(-10 * _MP.pi * 2**120)), 2**120)
        cases[2] = theta
        for theta in cases:
            div, mod = reduce_exactly_mod_2pi(theta)
            exact_div, rem = _mp_reduce(_MP.mpf(theta.numerator) / theta.denominator)
            self.assertLessEqual(rem, -TWO_PI)
            self.assertEqual(mod, 0.0)
            self.assertEqual(div, exact_div - 1)
            # the stored pair is the phase to within 2pi - TWO_PI
            stored = div * _MP_TWO_PI + mod
            self.assertLess(
                float(abs(stored - _MP.mpf(theta.numerator) / theta.denominator)),
                2.5e-16,
            )
        print(
            f"\n[06 test 1] normalisation: {len(cases)} phases whose remainder rounds to "
            f"-TWO_PI are stored as (div - 1, 0.0)"
        )

    def test_the_helper_against_mpmath(self):
        rng = np.random.default_rng(20261026)
        n = 0
        for magnitude in (1.0, 1.0e3, 1.0e8, 1.0e12, 5.0e12):
            for _ in range(400):
                theta = Fraction(float(rng.uniform(-magnitude, magnitude))) + Fraction(
                    float(rng.uniform(-1.0, 1.0))
                ) / (2 ** int(rng.integers(0, 90)))
                div, mod = reduce_exactly_mod_2pi(theta)
                exact_div, rem = _mp_reduce(
                    _MP.mpf(theta.numerator) / theta.denominator
                )
                if rem <= -TWO_PI:
                    self.assertEqual((div, mod), (exact_div - 1, 0.0))
                else:
                    self.assertEqual(div, exact_div)
                    self.assertEqual(mod, float(rem))
                n += 1
        print(
            f"\n[06 test 1] reduce_exactly_mod_2pi: {n} rationals, exact against mpmath"
        )


# ------------------------------------------------------------------------------------------------
# 2. the residual is the small part
# ------------------------------------------------------------------------------------------------


class TestResidualIsTheSmallPart(unittest.TestCase):
    def test_residual_and_its_reconstruction(self):
        lines = []
        for sector in ("Gk", "Tk"):
            for on_node in (True, False):
                for k in EXACTNESS_K:
                    with self.subTest(sector=sector, anchor=_anchor_name(on_node), k=k):
                        o = _Objects.get(sector, k, on_node)
                        worst_r, worst_mod = 0.0, 0.0
                        for i, value in enumerate(o.obj.values):
                            rho = o.payload["rho_delta_sample"][i]
                            small = abs(o.delta) + abs(rho)
                            exact_r = _MP.mpf(o.delta) - _MP.mpf(rho)
                            err_r = float(abs(_MP.mpf(value.theta_residual) - exact_r))
                            worst_r = max(worst_r, err_r / (4.0 * EPS * small))

                            # the exact large term added back, reduced in mpmath
                            large = -_MP.mpf(k) * _mp_leading_delta(
                                o.leading, o.z_init, o.z[i]
                            )
                            _, rem = _mp_reduce(large + _MP.mpf(value.theta_residual))
                            d = _cyclic(value.theta_mod_2pi, rem)
                            worst_mod = max(
                                worst_mod, d / (2.0 * ULP_TWO_PI + 4.0 * EPS * small)
                            )
                        lines.append(
                            f"   {sector} z_init {_anchor_name(on_node)}, k={k:.0e}: "
                            f"theta_residual against deltaTheta - rho.delta, worst / bound "
                            f"{worst_r:.3f}; reconstructed remainder, worst / bound "
                            f"{worst_mod:.3f}"
                        )
                        self.assertLessEqual(worst_r, 1.0)
                        self.assertLessEqual(worst_mod, 1.0)
        print("\n[06 test 2] the stored residual:")
        print("\n".join(lines))


# ------------------------------------------------------------------------------------------------
# the consumers, on the producers' stored values
# ------------------------------------------------------------------------------------------------


def transfer_functions(o) -> TkSourceFunctions:
    """``TkSourceFunctions`` on a stored ``TkWKBIntegration``, with a numeric stand-in that holds
    the switch node (its first sample) and the overlap, as production's rows do."""
    m = _Model.build()
    above = [float(z) for z in m.z_nodes if z > o.z_init][-3:]
    below = o.z[: 1 + HANDOVER_OVERLAP_NODES + 3]
    numeric = FakeTkNumeric(
        [
            FakeNumericValue(z, 1.0 / (1.0 + z), -1.0 / (1.0 + z) ** 2)
            for z in above + below
        ],
        z_exit=None,
        stop_deltaz_subh=None,
    )
    return TkSourceFunctions(m.model, o.k, numeric, o.obj)


class _NoiseSource:
    """
    Test 3's ``GkSource``: at ``z_R = 0.1`` and ``k = 3e8``, assembled by
    ``assemble_GkSource_values`` from one pure-WKB ``GkWKBIntegration`` per source redshift, the
    ``NOISE_SOURCES`` grid nodes below the top of the WKB region; its phase from ``_build_phase``.
    """

    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            m = _Model.build()
            k = NOISE_K
            top = _wkb_nodes(m.model, k, "Gk")[ANCHOR_NODE]
            sources = [float(z) for z in m.z_nodes if z <= top][:NOISE_SOURCES]
            cls.built = _assembled_source(
                m, k, NOISE_Z_RESPONSE, sources, lambda i, z: (None, 0.0, 1.0)
            )
        return cls.built


def _assembled_source(m, k, z_response_value, sources, initial):
    """
    A ``GkSource`` stand-in assembled by ``assemble_GkSource_values``' own body from one stored
    ``GkWKBIntegration`` per source redshift (``sources``, descending), each sampled at the
    response redshift alone. ``initial(i, z_s)`` gives ``(z_init or None, G_init, G'_init)``.
    Returns the stand-in source, its values with WKB data (ascending in ``z_s``) and the
    ``GkWKBValue`` of each source (ascending).
    """
    z_response = redshift(store_id=1, z=float(z_response_value))
    z_sources = _redshifts(sources, 10)
    WKB_data = {}
    wkb_values = []
    for i, z_s in enumerate(z_sources):
        z_init, value, derivative = initial(i, z_s.z)
        obj, _ = build_Gk(
            m.model,
            m.units,
            k,
            z_s,
            z_init,
            redshift_array([z_response]),
            value,
            derivative,
        )
        (wkb,) = obj.values
        wkb._parent_serial = next(_SERIALS)
        WKB_data[z_s.store_id] = wkb
        wkb_values.append(wkb)

    k_exit = SimpleNamespace(
        k=SimpleNamespace(k=k, k_inv_Mpc=k),
        store_id=1,
        z_exit_subh_e3=1.0e300,
        z_exit_subh_e4=1.0e300,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        out = assemble_GkSource_values._function(
            k_exit, z_response, redshift_array(z_sources), {}, WKB_data
        )
    values = sorted((v for v in out["values"] if v.has_WKB), key=lambda v: v.z_source.z)
    functions = m.model.functions
    source = SimpleNamespace(
        model_proxy=SimpleNamespace(get=lambda: SimpleNamespace(functions=functions)),
        k=SimpleNamespace(k=k, k_inv_Mpc=k),
        z_response=z_response,
    )
    wkb_values.reverse()
    return SimpleNamespace(
        source=source, values=values, wkb=wkb_values, k=k, z_response=z_response
    )


def _build_G(src):
    """``_build_phase`` on the assembled source, and anything it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        phase = _build_phase(src.source, src.values, "test 06 G")
    return phase, out.getvalue()


def _theta_magnitude(value) -> float:
    """``|theta|`` of a stored sample, from its own cycle count (for a ``GkSource`` value, the
    producer's, ``raw_theta_div_2pi``, not the rectified one, whose first block is zero).
    """
    div = getattr(value, "raw_theta_div_2pi", value.theta_div_2pi)
    return fabs(div * TWO_PI + value.theta_mod_2pi)


# ------------------------------------------------------------------------------------------------
# 3. phi is noise-free
# ------------------------------------------------------------------------------------------------


class TestPhiIsNoiseFree(unittest.TestCase):
    def _score(self, label, u, phi, theta, old_phi):
        """The windows with every |theta| >= 1e11: their RMS against 1e-3 ulp(theta), and the
        old construction's, printed unscored."""
        rms, starts = window_rms(u, phi)
        old_rms, _ = window_rms(u, old_phi)
        scored = []
        for r, r_old, i in zip(rms, old_rms, starts):
            window = theta[i : i + PHI_WINDOW]
            if min(window) < NOISE_MIN_THETA:
                continue
            ulp = float(np.spacing(window[PHI_WINDOW // 2]))
            scored.append((r / ulp, r_old / ulp))
        self.assertGreater(len(scored), 20, label)
        new = np.array([s[0] for s in scored])
        old = np.array([s[1] for s in scored])
        print(
            f"   {label}: {len(scored)} windows with |theta| >= 1e11; window RMS / ulp(theta) "
            f"now median {np.median(new):.3e}, max {np.max(new):.3e} (bound "
            f"{NOISE_BOUND_ULP:g}); the old construction (unscored) median "
            f"{np.median(old):.3e}, max {np.max(old):.3e}, zero in "
            f"{int(np.sum(old == 0.0))} windows"
        )
        self.assertLessEqual(float(np.max(new)), NOISE_BOUND_ULP, label)

    def test_phi_T_and_phi_G(self):
        print("\n[06 test 3] 04's window RMS of phi, k = 3e8:")

        # T: the consumer's phi, and the old phi built from the same stored values
        for on_node in (True, False):
            o = _Objects.get("Tk", NOISE_K, on_node)
            functions = transfer_functions(o)
            phase = functions.phase
            z = list(phase.z_samples)
            u = list(np.log1p(phase.z_samples))
            by_z = {v.z.z: v for v in o.obj.values}
            stored = [by_z[zz] for zz in z]
            theta = [_theta_magnitude(v) for v in stored]
            old_phi = build_phi_samples(
                o.k,
                o.leading,
                o.z_init,
                z,
                [v.theta_div_2pi * TWO_PI + v.theta_mod_2pi for v in stored],
                sign=TK_PHASE_SIGN,
            )
            self._score(
                f"phi_T, z_init {_anchor_name(on_node)}",
                u,
                list(phase.phi_samples),
                theta,
                old_phi,
            )

        # G: a GkSource at z_R = 0.1 over pure-WKB objects
        src = _NoiseSource.get()
        phase, printed = _build_G(src)
        self.assertNotIn("!! WARNING", printed)
        z = list(phase.z_samples)
        u = list(np.log1p(phase.z_samples))
        theta = [_theta_magnitude(v.WKB) for v in src.values]
        old_phi = build_phi_samples(
            src.k,
            src.source.model_proxy.get().functions.tau,
            src.z_response.z,
            z,
            [v.WKB.theta_div_2pi * TWO_PI + v.WKB.theta_mod_2pi for v in src.values],
            sign=GK_PHASE_SIGN,
        )
        self._score("phi_G, z_R = 0.1", u, list(phase.phi_samples), theta, old_phi)


# ------------------------------------------------------------------------------------------------
# 4. the two repairs agree
# ------------------------------------------------------------------------------------------------


class _WrapSource:
    """
    Test 4's ``GkSource``: numerically seeded objects at ``k = 3e8`` over the source nodes where
    ``k tau(z_s)`` is 25-80, each anchored two grid nodes below its source, so that the numeric
    span term ``k tau.delta(z_s, z_init)`` of ``phi_G`` is present and ~3 rad, with initial data whose ``deltaTheta`` falls by 0.35 rad from one source to the next,
    from 3 rad: it wraps through ``-pi`` to ``+pi`` twice. Assembled by
    ``assemble_GkSource_values``, so the rectifier repairs the wraps on the cycle counts.
    """

    built = None

    @classmethod
    def get(cls):
        if cls.built is None:
            m = _Model.build()
            k = WRAP_K
            tau = m.model.functions.tau
            rho, _ = cached_phase_residual(
                m.model, k, tau.table.z_nodes, "Gk", store_id=None
            )
            top = float(rho.z_nodes[0])
            nodes = [float(z) for z in m.z_nodes]
            below = {
                nodes[j]: nodes[j + WRAP_INIT_NODES_BELOW]
                for j in range(len(nodes) - WRAP_INIT_NODES_BELOW)
            }
            sources = [
                z
                for z in nodes
                if WRAP_KTAU[0] <= k * tau(z) <= WRAP_KTAU[1]
                and z in below
                and below[z] <= top
            ]
            n = len(sources)
            cls.targets = {}

            def initial(i, z_s):
                # sources are descending in z; deltaTheta falls with rising z_s
                target = WRAP_DELTA_START - WRAP_DELTA_STEP * (n - 1 - i)
                cls.targets[z_s] = target
                z_init = below[z_s]
                value, derivative = _initial_data(m.model, k, "Gk", z_init, target)
                return z_init, value, derivative

            cls.built = _assembled_source(m, k, WRAP_Z_RESPONSE, sources, initial)
        return cls.built


class TestTheTwoRepairsAgree(unittest.TestCase):
    def test_unwrap_matches_the_rectifier_and_phi_is_continuous(self):
        src = _WrapSource.get()
        phase, printed = _build_G(src)
        self.assertNotIn("!! WARNING", printed)

        tau = src.source.model_proxy.get().functions.tau
        phi = list(phase.phi_samples)
        first_shift = None
        cycles_seen = []
        for v, phi_i in zip(src.values, phi):
            raw = v.WKB.theta_residual + src.k * tau.delta(v.z_source.z, v.WKB.z_init)
            cycles = (raw - phi_i) / TWO_PI
            m = int(round(cycles))
            self.assertLess(abs(cycles - m), 1.0e-9)
            shift = v.WKB.raw_theta_div_2pi - v.WKB.theta_div_2pi
            if first_shift is None:
                first_shift = shift
            # the unwrap's whole cycles equal the rectifier's block shift, at every sample
            self.assertEqual(m, shift - first_shift, msg=f"z_s={v.z_source.z:.8g}")
            cycles_seen.append(m)
        wraps = int(np.sum(np.diff(cycles_seen) != 0))
        self.assertGreaterEqual(wraps, 2)

        # phi_G is continuous across them: 04's window RMS, within test 3's bound
        u = list(np.log1p(phase.z_samples))
        theta = [_theta_magnitude(v.WKB) for v in src.values]
        rms, starts = window_rms(u, phi)
        worst = max(
            r / float(np.spacing(theta[i + PHI_WINDOW // 2]))
            for r, i in zip(rms, starts)
        )
        print(
            f"\n[06 test 4] {len(src.values)} sources at k = {src.k:.0e}, z_R = "
            f"{src.z_response.z:g}: {wraps} wraps of deltaTheta, the unwrap equal to the "
            f"rectifier's shifts at every sample ({sorted(set(cycles_seen))}); window RMS of "
            f"phi_G / ulp(theta) at most {worst:.3e} (bound {NOISE_BOUND_ULP:g})"
        )
        self.assertLessEqual(worst, NOISE_BOUND_ULP)

    def test_a_disagreement_warns_and_the_build_goes_on(self):
        """In a build, a disagreement prints one ``!! WARNING`` line naming ``k``, ``z_R`` and the
        samples, and ``phi_G`` is built from its own unwrap (prompt 06 §0.2)."""
        src = _WrapSource.get()
        values = []
        for v in src.values:
            # a rectifier record with no shifts: the unwrap's cycles cannot agree with it
            wkb = v.WKB._replace(raw_theta_div_2pi=v.WKB.theta_div_2pi)
            clone = SimpleNamespace(z_source=v.z_source, WKB=wkb)
            values.append(clone)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            phase = _build_phase(src.source, values, "test 06 G disagree")
        printed = [line for line in out.getvalue().splitlines() if "!! WARNING" in line]
        self.assertEqual(len(printed), 1)
        self.assertIn(f"k={src.k:.8g}/Mpc", printed[0])
        self.assertIn(f"z_response={src.z_response.z:.8g}", printed[0])
        self.assertIn("z_s=", printed[0])
        reference, _ = _build_G(src)
        self.assertEqual(
            [float.hex(x) for x in phase.phi_samples],
            [float.hex(x) for x in reference.phi_samples],
        )


# ------------------------------------------------------------------------------------------------
# 5. the consumers
# ------------------------------------------------------------------------------------------------


class TestTheConsumers(unittest.TestCase):
    def test_raw_theta_at_the_samples(self):
        lines = []
        # T, both anchors, at every k
        for on_node in (True, False):
            for k in EXACTNESS_K:
                with self.subTest(sector="Tk", anchor=_anchor_name(on_node), k=k):
                    o = _Objects.get("Tk", k, on_node)
                    functions = transfer_functions(o)
                    phase = functions.phase
                    self.assertTrue(phase.exact_remainder)
                    # phi_T is the stored residual, sample for sample
                    by_z = {v.z.z: v for v in o.obj.values}
                    self.assertEqual(
                        [float.hex(x) for x in phase.phi_samples],
                        [float.hex(by_z[z].theta_residual) for z in phase.z_samples],
                    )
                    worst = 0.0
                    for i, z in enumerate(o.z):
                        exact = _exact_theta(o, i)
                        got = phase.raw_theta(z)
                        large = abs(
                            float(
                                _MP.mpf(k) * _mp_leading_delta(o.leading, z, o.z_init)
                            )
                        )
                        bound = 4.0 * EPS * (large + abs(phase.phi(z)))
                        worst = max(worst, float(abs(_MP.mpf(got) - exact)) / bound)
                    lines.append(
                        f"   T z_init {_anchor_name(on_node)}, k={k:.0e}: raw_theta against the "
                        f"exact phase, worst / bound {worst:.3f}"
                    )
                    self.assertLessEqual(worst, 1.0)

        # G: the GkSource of test 3, pure WKB, so the exact phase is -k tau.delta(z_s, z_R) - rho
        src = _NoiseSource.get()
        phase, _ = _build_G(src)
        self.assertTrue(phase.exact_remainder)
        tau = src.source.model_proxy.get().functions.tau
        worst = 0.0
        for v in src.values:
            z_s = v.z_source.z
            # the producer's phase at z_R: -(k tau.delta(z_s, z_R) + rho.delta(z_s, z_R)),
            # deltaTheta = 0 for a pure-WKB object; theta_residual = -rho.delta exactly
            exact = -_MP.mpf(src.k) * _mp_leading_delta(
                tau, z_s, src.z_response.z
            ) + _MP.mpf(v.WKB.theta_residual)
            got = phase.raw_theta(z_s)
            large = abs(
                float(_MP.mpf(src.k) * _mp_leading_delta(tau, z_s, src.z_response.z))
            )
            bound = 4.0 * EPS * (large + abs(phase.phi(z_s)))
            worst = max(worst, float(abs(_MP.mpf(got) - exact)) / bound)
        lines.append(
            f"   G at z_R = {src.z_response.z:g}, k={src.k:.0e}, {len(src.values)} sources: "
            f"raw_theta against the exact phase, worst / bound {worst:.3f}"
        )
        self.assertLessEqual(worst, 1.0)
        print("\n[06 test 5] the consumers' raw_theta at the samples:")
        print("\n".join(lines))

    def test_a_value_without_the_residual_is_refused(self):
        """No fallback: a value from a store written before prompt 06 carries no residual."""
        src = _NoiseSource.get()
        values = [
            SimpleNamespace(
                z_source=v.z_source, WKB=v.WKB._replace(theta_residual=None)
            )
            for v in src.values
        ]
        with self.assertRaisesRegex(RuntimeError, "theta_residual"):
            _build_phase(src.source, values, "test 06 G no residual")

        o = _Objects.get("Tk", 3.0e6, True)
        saved = [v._theta_residual for v in o.obj.values]
        try:
            for v in o.obj.values:
                v._theta_residual = None
            with self.assertRaisesRegex(RuntimeError, "theta_residual"):
                transfer_functions(o)
        finally:
            for v, r in zip(o.obj.values, saved):
                v._theta_residual = r


# ------------------------------------------------------------------------------------------------
# 6. nothing else moves
# ------------------------------------------------------------------------------------------------


class TestNothingElseMoves(unittest.TestCase):
    def test_a_primitive_phase_is_its_own_formulas_and_reduces_as_the_producers_do(
        self,
    ):
        """
        (a) ``PrimitivePhase`` is not touched by prompt 06, and no helper is factored out of it.
        With the same phi samples its ``raw_theta``, ``theta_deriv`` and ``theta_mod_2pi`` (both
        modes) are bit-identical to its own formulas, written out here; and its node constants,
        reduced by ``_remainder_against_two_pi`` in mpmath, are bit-identical to the producers'
        integer reduction of the same rationals.
        """
        from scipy.interpolate import make_interp_spline

        o = _Objects.get("Tk", 3.0e8, False)
        functions = transfer_functions(o)
        on = functions.phase
        model = _Model.build().model
        off = PrimitivePhase(
            on.k,
            o.leading,
            on.z_anchor,
            on.z_samples,
            on.phi_samples,
            sign=on.sign,
            model_functions=model.functions,
            rate=on._rate,
            exact_remainder=False,
        )
        u_points = np.log1p(on.z_samples)
        spline = make_interp_spline(u_points, on.phi_samples, k=3)
        spline_d = spline.derivative()
        anchor_j, anchor_p = o.leading.end_partial(on.z_anchor)
        hi, lo = o.leading.node_limbs(anchor_j)
        anchor = Fraction(hi) + Fraction(lo) + Fraction(anchor_p)
        n = 0
        constants = 0
        for z in list(on.z_samples[::7]) + [
            expm1(0.5 * (a + b)) for a, b in zip(u_points[::11], u_points[1::11])
        ]:
            u = log1p(z)
            raw = on.sign * on.k * o.leading.delta(z, on.z_anchor) + float(spline(u))
            self.assertEqual(float.hex(on.raw_theta(z)), float.hex(raw))
            self.assertEqual(float.hex(off.raw_theta(z)), float.hex(raw))
            deriv = on.sign * on.k * on._rate(z) + float(spline_d(u)) / (1.0 + z)
            self.assertEqual(float.hex(on.theta_deriv(z)), float.hex(deriv))
            self.assertEqual(
                float.hex(off.theta_mod_2pi(z)), float.hex(WKB_mod_2pi(raw)[1])
            )
            j, p = o.leading.end_partial(z)
            h, l = o.leading.node_limbs(j)
            exact = on.sign * Fraction(on.k) * (anchor - (Fraction(h) + Fraction(l)))
            c = _remainder_against_two_pi(exact)
            v = -on.sign * on.k * p + float(spline(u))
            r = c + WKB_mod_2pi(v)[1]
            if r <= -TWO_PI:
                r += TWO_PI
            self.assertEqual(float.hex(on.theta_mod_2pi(z)), float.hex(r))
            # the producers' reducer gives the same constant, to the bit
            div, mod = reduce_exactly_mod_2pi(exact)
            if c > -TWO_PI:
                self.assertEqual(float.hex(mod), float.hex(c))
                constants += 1
            n += 1
        print(
            f"\n[06 test 6(a)] PrimitivePhase at {n} abscissae bit-identical to its formulas; "
            f"{constants} node constants identical under both reducers"
        )

    def test_rho_and_the_background_tables(self):
        """(b) The producer's ``rho.delta`` is the residual table's own, bit for bit, and the
        table's ``delta`` is still ``node_term + (p_b - p_a)`` (05's formula)."""
        m = _Model.build()
        for sector in ("Gk", "Tk"):
            o = _Objects.get(sector, 1.0e8, False)
            rho, _ = cached_phase_residual(
                m.model, o.k, o.leading.table.z_nodes, sector, store_id=None
            )
            node = nearest_table_node(rho, o.z_init)
            anchor = rho.delta(o.z_init, node)
            for z, got in zip(o.z, o.payload["rho_delta_sample"]):
                self.assertEqual(float.hex(got), float.hex(anchor + rho.delta(node, z)))
            table = o.leading.table
            for z in o.z[::13]:
                j_a, j_b, p_a, p_b = table.delta_parts(o.z_init, z)
                self.assertEqual(
                    float.hex(table.delta(o.z_init, z)),
                    float.hex(table.node_term(j_a, j_b) + (p_b - p_a)),
                )

    def test_the_numeric_producers_and_QuadSource_import_nothing_this_prompt_changed(
        self,
    ):
        """
        (b), (c) The numeric producers and ``QuadSource`` run no code this prompt changed: none of
        the modules they import, followed through this repository, is one it changed. (The
        changed ``LiouvilleGreen/WKBtools.py`` holds docstrings and comments only.)
        """
        changed = {
            "ComputeTargets.primitive_phase",
            "ComputeTargets.GkWKBIntegration",
            "ComputeTargets.TkWKBIntegration",
            "ComputeTargets.GkSource",
            "ComputeTargets.GkSourcePolicyData",
            "ComputeTargets.TkSourceFunctions",
            "Quadrature.integrators.WKB_phase_function",
        }

        def imports(module: str):
            path = REPO_ROOT / (module.replace(".", "/") + ".py")
            if not path.exists():
                return set()
            tree = ast.parse(path.read_text())
            out = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    out.add(node.module)
                    for alias in node.names:
                        out.add(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.Import):
                    out.update(alias.name for alias in node.names)
            return {
                m for m in out if (REPO_ROOT / (m.replace(".", "/") + ".py")).exists()
            }

        for root in (
            "ComputeTargets.GkNumericIntegration",
            "ComputeTargets.TkNumericIntegration",
            "ComputeTargets.QuadSource",
            "ComputeTargets.BackgroundModel",
            "ComputeTargets.phase_residual",
            "ComputeTargets.cumulative_table",
        ):
            seen = set()
            todo = [root]
            while todo:
                module = todo.pop()
                if module in seen:
                    continue
                seen.add(module)
                todo.extend(imports(module))
            self.assertEqual(seen & changed, set(), root)


# ------------------------------------------------------------------------------------------------
# 7. end to end
# ------------------------------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    """
    ``test_phase_groups``' realistic fixture in regime ``G q r`` (w = 1/3), with ``G`` built by
    ``_build_phase`` on stand-in values carrying the residual (05's ``_ProductionPhaseGk``) and
    both ``T`` factors by ``TkSourceFunctions`` on stand-ins carrying it (05a's table ``cs_tau``
    fixtures); ``evaluate_sum`` against the direct product.
    """

    def test_evaluate_sum_against_the_direct_product(self):
        import copy

        from ComputeTargets.tests.test_g_phase_reanchor import (
            _ProductionPhaseGk,
            _TableCsTauFixture,
        )
        from ComputeTargets.tests.test_phase_groups import (
            K_WAVENUMBER,
            Q_WAVENUMBER,
            R_WAVENUMBER,
        )
        from ComputeTargets.tests.test_t_phase_reanchor import _table_fixture_functions

        w = 1.0 / 3.0
        regime = (True, True, True)
        base = Case.get(w)
        lowest = min(base.Tq_real.WKB_region[1], base.Tr_real.WKB_region[1])
        z_grid = log_grid(
            1.0 + base.both_numeric[0], 1.0 + lowest, PRODUCTION_SAMPLES_PER_LOG10Z
        )
        G = _ProductionPhaseGk(K_WAVENUMBER, w, base.z_resp, base.model, z_grid)
        case = copy.copy(base)
        case.Gk_real = G
        for name, k in (("q", Q_WAVENUMBER), ("r", R_WAVENUMBER)):
            fixture = _TableCsTauFixture(
                w,
                k=k,
                WKB_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
                numeric_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
            )
            setattr(case, f"T{name}_real", _table_fixture_functions(fixture))
        case.rng = np.random.default_rng(20261026 + 7)
        for factor in (case.Gk_real, case.Tq_real, case.Tr_real):
            self.assertTrue(factor.phase.exact_remainder)

        groups = case.groups(regime, exact=False)
        log_z = case.random_log_z(case.range_for(regime))
        worst = max_relative_residual(
            groups,
            lambda u: case.direct_product(regime, u, exact=False),
            log_z,
            extra_scale=lambda u: case.term_envelope(regime, u, exact=False),
        )
        bound = test_phase_groups.TestOracle1DirectProduct.TOLERANCE
        print(
            f"\n[06 test 7] Oracle 1, G q r at w = 1/3, every factor's phi from a stored "
            f"residual: max relative residual {worst:.3e} (bound {bound:g})"
        )
        self.assertLess(worst, bound)


if __name__ == "__main__":
    unittest.main()
