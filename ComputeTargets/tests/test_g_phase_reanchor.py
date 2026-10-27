"""
Tests for the node-anchored remainder of ``PrimitivePhase`` and the ``CumulativeTable`` node
accessors it reads (``prompts/levin-phase-floor`` prompt 05 §3, layer 1(a)).

**What changed.** ``PrimitivePhase.theta_mod_2pi`` used to reduce ``raw_theta``, a double of size
``k tau_R`` (up to ~1e12 rad), so every remainder carried ``eps k tau_R`` of rounding, and the
rounding is not smooth in ``z``: 03 found it at 0.22-0.34 F in every expensive phase group and 04
showed it sends the Levin bisection to its cap. With ``exact_remainder=True`` the remainder is
``c_{n_z} + rem(v(z))``: a per-node constant reduced exactly against 2pi itself, plus a small
variable part ``v = -sign k p_z + phi(u)`` in double (``primitive_phase.py``'s module docstring,
"Anchoring"). ``GkSourcePolicyData._build_phase`` turns it on for G; since prompt 05a,
``TkSourceFunctions`` turns it on for T too (``test_t_phase_reanchor``).

The six tests of prompt 05 §3, in order:

1. **The accessor** (``TestNodeAccessor``): ``delta`` is bit-identical to today's formula, and the
   exact node term rounds to ``delta``'s table term within one ulp.
2. **Exactness** (``TestExactness``): the remainder is within ``4 eps (k |p_z| + 2pi)`` of the exact
   remainder of theta against 2pi, at four k, two z_R and both signs.
3. **No jitter** (``TestNoJitter``): 04's sigma_theta at z' ~ 1e10, k = 3e6, off and on.
4. **Nothing else moves** (``TestNothingElseMoves``): ``raw_theta``/``theta_deriv`` bit-identical
   with the keyword on and off; a ``TkSourceFunctions`` phase with the keyword off keeps the
   pre-05 remainders bit for bit (prompt 05a §3 test 4(b), which replaced 05's test that T's
   remainders were unchanged).
5. **The default is today's** (``TestDefault``): keyword off is ``WKB_mod_2pi(raw_theta)[1]`` bit
   for bit; keyword on without the accessor raises.
6. **End to end** (``TestEndToEnd``): ``test_phase_groups``' realistic fixture at w = 1/3 with G's
   phase built by ``_build_phase``.

**The background.** "A table built as ``functions.tau`` is built" (``compute_background``): a
``CumulativeTable`` of ``1/H`` at ``TAU_GAUSS_ORDER`` on a descending grid, shifted by the
radiation-era ``tau_init`` at the top node. ``H`` is a closed-form flat Lambda-CDM with radiation,
in 1/Mpc, so that ``tau`` (~1.37e4 Mpc at z = 0.1) and the phases (``k tau_R`` up to 4e12 rad at
``k = 3e8/Mpc``) have production's magnitudes. The grid is 100 nodes per decade of ``z`` from 1e12
to 0.1, with z = 0.1 a node (the production case: a response redshift is a grid node). z = 30 is
not a node, so the second anchor exercises the end partial ``p_R`` too.

No Ray, no datastore. mpmath precision is set in private contexts only.
"""

import copy
import unittest
from fractions import Fraction
from math import expm1, log1p, sqrt, ulp

import mpmath
import numpy as np

from ComputeTargets.BackgroundModel import TAU_GAUSS_ORDER, TablePrimitive
from ComputeTargets.GkSourcePolicyData import GK_PHASE_SIGN
from ComputeTargets.TkSourceFunctions import TkSourceFunctions
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.phase_groups import build_phase_groups
from ComputeTargets.primitive_phase import PrimitivePhase
from ComputeTargets.tests import test_phase_groups
from ComputeTargets.tests.test_phase_groups import (
    BesselPhaseGk,
    _StoredGkValue,
    Case,
    K_WAVENUMBER,
    circular_distance,
    max_relative_residual,
)
from ComputeTargets.tests.test_tk_source_functions import (
    Fixture,
    PRODUCTION_SAMPLES_PER_LOG10Z,
    log_grid,
)
from ComputeTargets.tests.wkb_reference import ClosedFormPrimitive
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI

EPS = float(np.finfo(np.float64).eps)

# flat Lambda-CDM with radiation, H in 1/Mpc (h = 0.674; Omega_r includes neutrinos)
H0_INV_MPC = 0.674 * 100.0 / 299792.458
OMEGA_R = 9.2e-5
OMEGA_M = 0.315
OMEGA_L = 1.0 - OMEGA_M - OMEGA_R

GRID_Z_TOP = 1.0e12
GRID_Z_BOTTOM = 0.1
GRID_PER_DECADE = 100

# the top of the phi samples, and of the abscissae of tests 2 and 4
Z_SAMPLES_TOP = 2.0e11
Z_ABSCISSAE_TOP = 1.0e11

# prompt 05 §3 test 1
ACCESSOR_PAIRS = 1000
# prompt 05 §3 test 2
EXACTNESS_K = (1.0e5, 3.0e6, 1.0e8, 3.0e8)
EXACTNESS_Z_RESPONSE = (0.1, 30.0)
EXACTNESS_RANDOM_POINTS = 200
EXACTNESS_NODE_STRIDE = 25
EXACTNESS_BOUND_EPS = 4.0
# prompt 05 §3 test 3: 04's sigma_theta
JITTER_K = 3.0e6
JITTER_Z = 1.0e10
JITTER_Z_RESPONSE = 0.1
JITTER_WIDTH_U = 1.0e-6
SIGMA_POINTS = 65
SIGMA_DEGREE = 4
JITTER_OFF_AT_LEAST = 0.1  # in units of eps k eta_R
JITTER_ON_FACTOR = 1.0e-3  # of JITTER_OFF_AT_LEAST eps k eta_R
# prompt 05 §3 test 6
END_TO_END_W = 1.0 / 3.0
END_TO_END_REGIME = (True, True, True)
END_TO_END_FLOOR_EPS = 4.0
END_TO_END_FLOOR_ABS = 1.0e-14

SEED = 20261026

# the reference remainders are formed in this private context, never in mpmath's global one
_MP = mpmath.MPContext()
_MP.prec = 320


def _Hubble(z: float) -> float:
    s = 1.0 + z
    return H0_INV_MPC * sqrt(OMEGA_R * s**4 + OMEGA_M * s**3 + OMEGA_L)


class _Functions:
    """The one field ``PrimitivePhase`` reads off a ``ModelFunctions``."""

    Hubble = staticmethod(_Hubble)


def _grid() -> np.ndarray:
    n = int(round(GRID_PER_DECADE * (np.log10(GRID_Z_TOP) - np.log10(GRID_Z_BOTTOM))))
    z = np.logspace(np.log10(GRID_Z_TOP), np.log10(GRID_Z_BOTTOM), n + 1)
    z[0] = GRID_Z_TOP
    z[-1] = GRID_Z_BOTTOM
    return z


class _Background:
    """Built once: the grid and ``functions.tau`` built as ``compute_background`` builds it."""

    z_nodes = None
    tau = None

    @classmethod
    def build(cls):
        if cls.tau is not None:
            return
        cls.z_nodes = _grid()
        table = CumulativeTable(
            cls.z_nodes, lambda z: 1.0 / _Hubble(z), TAU_GAUSS_ORDER, label="tau"
        )
        # the radiation-era asymptote at the top of the grid, as compute_background attaches it
        tau_init = 1.0 / (H0_INV_MPC * sqrt(OMEGA_R) * (1.0 + cls.z_nodes[0]))
        cls.tau = TablePrimitive(table.shifted(tau_init), "tau")


def _phi_of_u(u):
    """A smooth O(1) residual."""
    return 0.3 * np.sin(u) + 0.05 * u


def _phase(k, z_response, sign, exact, label="test") -> PrimitivePhase:
    """A phase on the realistic background, with phi sampled at the table's nodes."""
    _Background.build()
    z = _Background.z_nodes[::-1]
    z_samples = z[(z >= z_response) & (z <= Z_SAMPLES_TOP)]
    return PrimitivePhase(
        k,
        _Background.tau,
        z_response,
        z_samples,
        _phi_of_u(np.log1p(z_samples)),
        sign=sign,
        model_functions=_Functions,
        label=label,
        exact_remainder=exact,
    )


def _abscissae(z_response, rng, n_random, node_stride):
    """``u = log(1+z)``: ``n_random`` log-uniform in (z_response, Z_ABSCISSAE_TOP), and every
    ``node_stride``-th table node in that range exactly (on the grid)."""
    u_lo = log1p(z_response) + 0.01
    u_hi = log1p(Z_ABSCISSAE_TOP)
    out = [float(u) for u in rng.uniform(u_lo, u_hi, n_random)]
    nodes = _Background.z_nodes[
        (_Background.z_nodes > expm1(u_lo)) & (_Background.z_nodes < Z_ABSCISSAE_TOP)
    ]
    # on-grid points are passed as z, so that the exact node lookup finds them
    out_z = [float(z) for z in nodes[::node_stride]]
    return out, out_z


def _mp(x: Fraction):
    return _MP.mpf(x.numerator) / _MP.mpf(x.denominator)


def _wrap_mp(d):
    """``d`` onto (-pi, pi], in the private context."""
    two_pi = 2 * _MP.pi
    return d - _MP.nint(d / two_pi) * two_pi


def _todays_delta(table: CumulativeTable, z_a: float, z_b: float) -> float:
    """
    ``CumulativeTable.delta`` as it was written before prompt 05, copied here (``cumulative_table``
    at ``e648951``, lines 360-401), so that the refactored method is compared against it.
    """
    z_a = float(z_a)
    z_b = float(z_b)
    j_a, on_a = table._locate(z_a)
    j_b, on_b = table._locate(z_b)

    tab = (table._hi_list[j_b] - table._hi_list[j_a]) + (
        table._lo_list[j_b] - table._lo_list[j_a]
    )

    p_b = 0.0 if on_b else table._integral(z_b, table._z_list[j_b])
    p_a = 0.0 if on_a else table._integral(z_a, table._z_list[j_a])
    ends = p_b - p_a

    return tab + ends


def _sigma_theta(phase, a: float, b: float) -> float:
    """04's sigma_theta (03's ``analyse.sigma_theta``): ``theta_mod_2pi`` at 65 equispaced points
    of ``[a, b]`` in ``log(1+z)``, ends included, ``numpy.unwrap``, a degree-4 least-squares
    polynomial in ``t`` in [-1, 1], and the RMS residual."""
    x = np.linspace(a, b, SIGMA_POINTS)
    th = np.unwrap(
        np.array([phase.theta_mod_2pi(float(xi), x_is_log=True) for xi in x])
    )
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, th, SIGMA_DEGREE)
    res = th - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res)))


# ------------------------------------------------------------------------------------------------
# 1. the accessor
# ------------------------------------------------------------------------------------------------


class TestNodeAccessor(unittest.TestCase):
    """Prompt 05 §3 test 1."""

    @classmethod
    def setUpClass(cls):
        _Background.build()
        cls.tau = _Background.tau
        cls.table = _Background.tau.table
        rng = np.random.default_rng(SEED)
        z_nodes = _Background.z_nodes
        u_lo = log1p(z_nodes[-1])
        u_hi = log1p(z_nodes[0])

        def draw():
            # half on a node, half off (log-uniform over the whole grid)
            if rng.random() < 0.5:
                return float(z_nodes[int(rng.integers(0, len(z_nodes)))])
            return float(expm1(rng.uniform(u_lo, u_hi)))

        cls.pairs = [(draw(), draw()) for _ in range(ACCESSOR_PAIRS)]

    def test_delta_is_bit_identical_to_todays_formula(self):
        on_grid = 0
        for z_a, z_b in self.pairs:
            got = self.table.delta(z_a, z_b)
            want = _todays_delta(self.table, z_a, z_b)
            self.assertEqual(float.hex(got), float.hex(want), (z_a, z_b))
            # TablePrimitive passes the table through unchanged
            self.assertEqual(float.hex(self.tau.delta(z_a, z_b)), float.hex(got))
            # and delta is node_term + ends from the parts, bit for bit
            j_a, j_b, p_a, p_b = self.tau.delta_parts(z_a, z_b)
            self.assertEqual(
                float.hex(self.tau.node_term(j_a, j_b) + (p_b - p_a)), float.hex(got)
            )
            on_grid += (self.table.node_index(z_a) is not None) + (
                self.table.node_index(z_b) is not None
            )
        print(
            f"\n[05 test 1] delta bit-identical to today's formula at {len(self.pairs)} pairs "
            f"({on_grid} of {2 * len(self.pairs)} endpoints on a node)"
        )
        # both kinds of endpoint were exercised
        self.assertGreater(on_grid, 0.3 * 2 * len(self.pairs))
        self.assertLess(on_grid, 0.7 * 2 * len(self.pairs))

    def test_exact_node_term_rounds_to_the_table_term(self):
        worst = 0.0
        for z_a, z_b in self.pairs:
            j_a, j_b, _p_a, _p_b = self.tau.delta_parts(z_a, z_b)
            hi_a, lo_a = self.tau.node_limbs(j_a)
            hi_b, lo_b = self.tau.node_limbs(j_b)
            exact = (Fraction(hi_b) - Fraction(hi_a)) + (
                Fraction(lo_b) - Fraction(lo_a)
            )
            table_term = self.tau.node_term(j_a, j_b)
            err = abs(exact - Fraction(table_term))
            if table_term == 0.0:
                self.assertEqual(exact, 0)
                continue
            in_ulp = float(err / Fraction(ulp(table_term)))
            worst = max(worst, in_ulp)
            self.assertLessEqual(in_ulp, 1.0, (z_a, z_b, j_a, j_b))
        print(
            f"[05 test 1] exact node term vs delta's table term: worst {worst:.3f} ulp "
            f"over {len(self.pairs)} pairs"
        )

    def test_end_partial_is_delta_s_own_end(self):
        """``end_partial(z)`` is the node and partial ``delta`` uses for that endpoint."""
        for z_a, z_b in self.pairs[:200]:
            j_a, j_b, p_a, p_b = self.tau.delta_parts(z_a, z_b)
            self.assertEqual((j_a, float.hex(p_a)), self._end(z_a))
            self.assertEqual((j_b, float.hex(p_b)), self._end(z_b))
        with self.assertRaises(IndexError):
            self.tau.node_limbs(len(self.table))
        with self.assertRaises(IndexError):
            self.tau.node_limbs(-1)

    def _end(self, z):
        j, p = self.tau.end_partial(z)
        return j, float.hex(p)


# ------------------------------------------------------------------------------------------------
# 2. exactness against 2pi
# ------------------------------------------------------------------------------------------------


def _reference_remainder(phase: PrimitivePhase, x, x_is_log: bool):
    """
    ``(theta mod 2pi, p_z)``: theta formed exactly from the same limbs, the same end partials and
    the same phi double that ``PrimitivePhase`` uses, and reduced against 2pi in ``_MP``.
    """
    tau = phase._leading
    raw_x, log_x = phase._get_x(x, x_is_log)
    u = phase._clamped_u(raw_x, log_x)
    phi = float(phase._spline(u))
    j, p_z = tau.end_partial(raw_x)
    j_R, p_R = tau.end_partial(phase.z_anchor)
    hi_R, lo_R = tau.node_limbs(j_R)
    hi_n, lo_n = tau.node_limbs(j)
    node_term = (Fraction(hi_R) + Fraction(lo_R)) - (Fraction(hi_n) + Fraction(lo_n))
    theta = phase.sign * Fraction(phase.k) * (
        node_term + Fraction(p_R) - Fraction(p_z)
    ) + Fraction(phi)
    th = _mp(theta)
    two_pi = 2 * _MP.pi
    return th - _MP.floor(th / two_pi) * two_pi, p_z


class TestExactness(unittest.TestCase):
    """Prompt 05 §3 test 2: within ``4 eps (k |p_z| + 2pi)`` of the exact remainder against 2pi."""

    def test_remainder_against_two_pi(self):
        rng = np.random.default_rng(SEED)
        lines = []
        for z_response in EXACTNESS_Z_RESPONSE:
            for k in EXACTNESS_K:
                for sign in (-1, +1):
                    with self.subTest(z_response=z_response, k=k, sign=sign):
                        phase = _phase(k, z_response, sign, exact=True)
                        u_points, z_points = _abscissae(
                            z_response,
                            rng,
                            EXACTNESS_RANDOM_POINTS,
                            EXACTNESS_NODE_STRIDE,
                        )
                        worst = 0.0
                        worst_at = None
                        for x, x_is_log in [(u, True) for u in u_points] + [
                            (z, False) for z in z_points
                        ]:
                            ref, p_z = _reference_remainder(phase, x, x_is_log)
                            got = phase.theta_mod_2pi(x, x_is_log=x_is_log)
                            self.assertTrue(-TWO_PI <= got <= 0.0, got)
                            err = float(abs(_wrap_mp(_MP.mpf(got) - ref)))
                            bound = (
                                EXACTNESS_BOUND_EPS * EPS * (k * abs(p_z) + 2.0 * np.pi)
                            )
                            if err / bound > worst:
                                worst, worst_at = err / bound, (x, x_is_log, err)
                        lines.append(
                            f"   z_R={z_response:g} k={k:.0e} sign={sign:+d}: "
                            f"worst err/bound {worst:.3f} ({worst_at[2]:.3e} rad), "
                            f"{len(u_points) + len(z_points)} points, "
                            f"{len(phase._node_constants)} node constants"
                        )
                        self.assertLessEqual(worst, 1.0, worst_at)
        print("\n[05 test 2] node-anchored remainder against the exact one (vs 2pi):")
        print("\n".join(lines))


# ------------------------------------------------------------------------------------------------
# 3. no jitter
# ------------------------------------------------------------------------------------------------


class TestNoJitter(unittest.TestCase):
    """
    Prompt 05 §3 test 3: 04's sigma_theta on a region of width 1e-6 in u at z' ~ 1e10,
    k = 3e6/Mpc. The region is centred on the midpoint, in u, between two table nodes, where
    ``_nearest_node`` switches from one to the other, so that the switch between nodes (prompt
    05 §0.2) is inside it: the two node constants must agree there to rounding.
    """

    def test_sigma_theta_off_and_on(self):
        _Background.build()
        u_asc = np.log1p(_Background.z_nodes)[::-1]
        i = int(np.searchsorted(u_asc, log1p(JITTER_Z)))
        u_mid = 0.5 * (u_asc[i - 1] + u_asc[i])
        a = u_mid - 0.5 * JITTER_WIDTH_U
        b = u_mid + 0.5 * JITTER_WIDTH_U

        eta_R = _Background.tau(JITTER_Z_RESPONSE)
        F = EPS * JITTER_K * eta_R
        for sign in (-1, +1):
            with self.subTest(sign=sign):
                off = _phase(JITTER_K, JITTER_Z_RESPONSE, sign, exact=False)
                on = _phase(JITTER_K, JITTER_Z_RESPONSE, sign, exact=True)
                sigma_off = _sigma_theta(off, a, b)
                sigma_on = _sigma_theta(on, a, b)
                # the window does straddle the switch between nodes
                nodes = {
                    _Background.tau.end_partial(expm1(float(x)))[0]
                    for x in np.linspace(a, b, SIGMA_POINTS)
                }
                print(
                    f"\n[05 test 3] sign={sign:+d}, z'={expm1(u_mid):.4g}, k={JITTER_K:.0e}, "
                    f"F = eps k eta_R = {F:.4e} rad: sigma_theta off {sigma_off:.4e} rad "
                    f"({sigma_off / F:.4f} F), on {sigma_on:.4e} rad ({sigma_on / F:.3e} F); "
                    f"{len(nodes)} nodes in the window"
                )
                self.assertEqual(len(nodes), 2)
                self.assertGreaterEqual(sigma_off, JITTER_OFF_AT_LEAST * F)
                self.assertLessEqual(
                    sigma_on, JITTER_ON_FACTOR * JITTER_OFF_AT_LEAST * F
                )


# ------------------------------------------------------------------------------------------------
# 4. nothing else moves
# ------------------------------------------------------------------------------------------------


class _TableCsTauFixture(Fixture):
    """
    ``test_tk_source_functions``' exact fixture with ``cs_tau`` a real ``TablePrimitive``, as
    production's ``functions.cs_tau`` is, instead of a ``ClosedFormPrimitive``. So a
    ``TkSourceFunctions`` built on it with no keyword builds its phase as production does, on a
    leading primitive that *has* the node accessors, node-anchored (prompt 05a). The inherited
    ``exact_functions`` passes ``exact_remainder=False``, as every closed-form fixture site does;
    a test that wants production's construction captures the fixture's samples and builds the
    object itself (``test_t_phase_reanchor._table_fixture_functions``,
    ``test_quadsource_integral.Case``).

    **The table** is a ``CumulativeTable`` of ``c_s/H`` at ``TAU_GAUSS_ORDER`` on the union of the
    fixture's numeric and Liouville-Green nodes (``z_numeric``, ``z_numeric_below``, ``z_WKB``),
    **without the one numeric node equal to** ``z_init`` (``z_numeric[-1] == crossover_z`` bit for
    bit), descending, shifted to the closed form at its top node. So the table covers ``z_init``,
    where ``T``'s phase is anchored, and ``z_init`` is off a node, as on production's background
    grid, where ``z_init`` is a root; the interval around it is two grid steps wide. This is A4's
    ``_UnionTableCsTauFixture`` (``docs/handover/a4/scripts/a4_cells.py``, the user's answers at
    A4's stops), adopted here by prompts/handover prompt 13. Until then the table was built on
    ``z_WKB`` alone, which with ``drop_first_WKB_sample=True`` starts one grid step below
    ``z_init``, and whether ``z_init`` was refused turned on one ulp of ``log1p``
    (``[04-table-cs-tau-fixture-does-not-cover-z-init-when-the-first-sample-is-dropped]``).
    """

    def cs_tau_nodes(self):
        """The table's nodes, descending; ``z_init`` lies strictly inside them and is not one."""
        nodes = {float(z) for z in self.z_numeric}
        nodes |= {float(z) for z in self.z_numeric_below}
        nodes |= {float(z) for z in self.z_WKB}
        nodes.discard(float(self.crossover_z))
        nodes = sorted(nodes, reverse=True)
        if not nodes[0] > self.crossover_z > nodes[-1]:
            raise RuntimeError(
                f"_TableCsTauFixture: the cs_tau table's nodes {nodes[0]!r}..{nodes[-1]!r} do "
                f"not cover z_init={self.crossover_z!r}"
            )
        return nodes

    def exact_envelope_model(self):
        model = super().exact_envelope_model()
        nodes = self.cs_tau_nodes()
        w = self.w
        table = CumulativeTable(
            nodes, lambda z: sqrt(w) / model.Hubble(z), TAU_GAUSS_ORDER, label="cs_tau"
        ).shifted(model.cs_tau(nodes[0]))
        model.functions = model.functions._replace(
            cs_tau=TablePrimitive(table, "cs_tau")
        )
        return model


class TestNothingElseMoves(unittest.TestCase):
    """Prompt 05 §3 test 4."""

    def test_raw_theta_and_theta_deriv_are_unchanged(self):
        rng = np.random.default_rng(SEED + 4)
        n = 0
        for sign in (-1, +1):
            for k in (3.0e6, 3.0e8):
                with self.subTest(sign=sign, k=k):
                    off = _phase(k, 0.1, sign, exact=False)
                    on = _phase(k, 0.1, sign, exact=True)
                    u_points, z_points = _abscissae(0.1, rng, 200, 25)
                    for x, x_is_log in [(u, True) for u in u_points] + [
                        (z, False) for z in z_points
                    ]:
                        self.assertEqual(
                            float.hex(on.raw_theta(x, x_is_log=x_is_log)),
                            float.hex(off.raw_theta(x, x_is_log=x_is_log)),
                        )
                        for log_derivative in (False, True):
                            self.assertEqual(
                                float.hex(
                                    on.theta_deriv(
                                        x,
                                        x_is_log=x_is_log,
                                        log_derivative=log_derivative,
                                    )
                                ),
                                float.hex(
                                    off.theta_deriv(
                                        x,
                                        x_is_log=x_is_log,
                                        log_derivative=log_derivative,
                                    )
                                ),
                            )
                        n += 1
        print(
            f"\n[05 test 4] raw_theta and theta_deriv bit-identical, keyword on and off: "
            f"{n} abscissae, both signs"
        )

    def test_transfer_function_keyword_off_is_WKB_mod_2pi_of_raw_theta(self):
        """
        **Replaced by prompt 05a §3 test 4(b)** (prompt 05a §2). This was
        ``test_transfer_function_remainders_are_unchanged``, which asserted G1's decision that
        ``T``'s remainders stay as they were with ``exact_remainder`` left at its default. The
        user reversed that on 2026-10-26: ``TkSourceFunctions`` now defaults the keyword on
        (``test_t_phase_reanchor`` tests that). What remains true, and is asserted here, is that
        with the keyword *off* a ``TkSourceFunctions`` phase on a table ``cs_tau`` keeps the old
        remainder bit for bit: ``WKB_mod_2pi(raw_theta)[1]``, which is what ``theta_mod_2pi``
        returned before prompt 05.
        """
        from unittest.mock import patch

        import ComputeTargets.tests.test_tk_source_functions as tk_fixtures

        fixture = _TableCsTauFixture(END_TO_END_W)
        # the fixture's own stored samples, captured from exact_functions() as
        # test_quadsource_integral.exact_Tk_inputs captures them, and read against its table
        # cs_tau with the keyword passed explicitly off
        captured = {}

        def capture(model, k, Tk_numeric, Tk_WKB, exact_remainder=True):
            captured["inputs"] = (Tk_numeric, Tk_WKB)

        with patch.object(tk_fixtures, "TkSourceFunctions", capture):
            fixture.exact_functions()
        T = TkSourceFunctions(
            fixture.exact_envelope_model(),
            fixture.k_object,
            *captured["inputs"],
            exact_remainder=False,
        )
        phase = T.phase
        self.assertIsInstance(phase._leading, TablePrimitive)
        self.assertFalse(phase.exact_remainder)
        self.assertEqual(phase.sign, +1)
        # every WKB sample, and the midpoints between them
        z = sorted(fixture.z_WKB)
        points = z + [0.5 * (z[i] + z[i + 1]) for i in range(len(z) - 1)]
        differ = []
        for zz in points:
            got = phase.theta_mod_2pi(zz)
            want = WKB_mod_2pi(phase.raw_theta(zz))[1]
            if float.hex(got) != float.hex(want):
                differ.append((zz, got, want))
        print(
            f"\n[05a test 4(b)] TkSourceFunctions remainders, keyword off: "
            f"{len(points) - len(differ)} of {len(points)} bit-identical to "
            f"WKB_mod_2pi(raw_theta)[1]"
            + (
                f"; first difference at z={differ[0][0]!r}: {differ[0][1]!r} against "
                f"{differ[0][2]!r}"
                if differ
                else ""
            )
        )
        self.assertEqual(differ, [])


# ------------------------------------------------------------------------------------------------
# 5. the default is today's
# ------------------------------------------------------------------------------------------------


class TestDefault(unittest.TestCase):
    """Prompt 05 §3 test 5, and §2's "without the accessor it raises"."""

    def test_keyword_off_is_WKB_mod_2pi_of_raw_theta(self):
        rng = np.random.default_rng(SEED + 5)
        n = 0
        for sign in (-1, +1):
            for k in EXACTNESS_K:
                phase = _phase(k, 30.0, sign, exact=False)
                self.assertFalse(phase.exact_remainder)
                u_points, z_points = _abscissae(30.0, rng, 100, 50)
                for x, x_is_log in [(u, True) for u in u_points] + [
                    (z, False) for z in z_points
                ]:
                    self.assertEqual(
                        float.hex(phase.theta_mod_2pi(x, x_is_log=x_is_log)),
                        float.hex(
                            WKB_mod_2pi(phase.raw_theta(x, x_is_log=x_is_log))[1]
                        ),
                    )
                    n += 1
        # the keyword's default is off
        _Background.build()
        z = _Background.z_nodes[::-1]
        z_samples = z[(z >= 30.0) & (z <= Z_SAMPLES_TOP)]
        default = PrimitivePhase(
            1.0e5,
            _Background.tau,
            30.0,
            z_samples,
            _phi_of_u(np.log1p(z_samples)),
            sign=-1,
            model_functions=_Functions,
        )
        self.assertFalse(default.exact_remainder)
        print(f"\n[05 test 5] keyword off: {n} remainders bit-identical to today's")

    def test_without_the_accessor_it_raises(self):
        _Background.build()
        closed_form = ClosedFormPrimitive(lambda z: _Background.tau(z), label="tau")
        z = _Background.z_nodes[::-1]
        z_samples = z[(z >= 0.1) & (z <= 1.0e3)]
        phi = _phi_of_u(np.log1p(z_samples))
        with self.assertRaisesRegex(ValueError, "end_partial"):
            PrimitivePhase(
                1.0e5,
                closed_form,
                0.1,
                z_samples,
                phi,
                sign=-1,
                model_functions=_Functions,
                exact_remainder=True,
            )
        # off, the same primitive is accepted, as before
        PrimitivePhase(
            1.0e5, closed_form, 0.1, z_samples, phi, sign=-1, model_functions=_Functions
        )


# ------------------------------------------------------------------------------------------------
# 6. end to end
# ------------------------------------------------------------------------------------------------


# The fields ``_build_phase`` reads off a ``GkSourceValue``: since prompts/handover prompt 13 the
# stand-in lives beside ``BesselPhaseGk``, which builds its phase with ``_build_phase`` too.
_Value = _StoredGkValue


class _ProductionPhaseGk(BesselPhaseGk):
    """
    ``test_phase_groups``' realistic G fixture, ``BesselPhaseGk`` (bessel_phase amplitude, stored
    (div 2pi, mod 2pi) samples of the exact radiation phase carrying ``theta_residual``), whose
    phase is built by ``GkSourcePolicyData._build_phase`` itself -- a ``PrimitivePhase`` on a real
    ``CumulativeTable`` of ``1/H``, anchored at the response redshift. Since prompts/handover
    prompt 13 that is ``BesselPhaseGk``'s own construction, so this class adds only ``old_phase``:
    the same object with the keyword off, today's remainder before prompt 05.
    """

    def __init__(self, k, w, z_resp, model, z_grid):
        super().__init__(k, w, z_resp, model, z_grid)
        self.old_phase = PrimitivePhase(
            self.phase.k,
            self.tau,
            self.phase.z_anchor,
            self.phase.z_samples,
            self.phase.phi_samples,
            sign=self.phase.sign,
            model_functions=self.model_functions,
            label="test G, keyword off",
        )


class TestEndToEnd(unittest.TestCase):
    """Prompt 05 §3 test 6, at one cheap configuration: w = 1/3, regime G q r."""

    @classmethod
    def setUpClass(cls):
        base = Case.get(END_TO_END_W)
        cls.base = base
        # the realistic G fixture's own grid, as Case builds it
        lowest = min(base.Tq_real.WKB_region[1], base.Tr_real.WKB_region[1])
        z_grid = log_grid(
            1.0 + base.both_numeric[0], 1.0 + lowest, PRODUCTION_SAMPLES_PER_LOG10Z
        )
        cls.G = _ProductionPhaseGk(
            K_WAVENUMBER, END_TO_END_W, base.z_resp, base.model, z_grid
        )
        # the same Case with G's phase built as production builds it
        cls.case = copy.copy(base)
        cls.case.Gk_real = cls.G
        cls.case.rng = np.random.default_rng(SEED + 6)

    def test_build_phase_turns_the_node_anchored_remainder_on(self):
        self.assertTrue(self.G.phase.exact_remainder)
        self.assertEqual(self.G.phase.sign, GK_PHASE_SIGN)
        self.assertFalse(self.G.old_phase.exact_remainder)

    def test_evaluate_sum_against_the_direct_product(self):
        case = self.case
        regime = END_TO_END_REGIME
        groups = case.groups(regime, exact=False)
        log_z = case.random_log_z(case.range_for(regime))
        worst = max_relative_residual(
            groups,
            lambda u: case.direct_product(regime, u, exact=False),
            log_z,
            extra_scale=lambda u: case.term_envelope(regime, u, exact=False),
        )
        print(
            f"\n[05 test 6] Oracle 1 with G's phase from _build_phase (w=1/3, G q r): "
            f"max relative residual {worst:.3e} (bound {test_phase_groups.TestOracle1DirectProduct.TOLERANCE:g})"
        )
        self.assertLess(worst, test_phase_groups.TestOracle1DirectProduct.TOLERANCE)

    def test_composed_remainder_moves_by_at_most_the_old_floor(self):
        case = self.case
        regime = END_TO_END_REGIME
        G = self.G
        eta_R = G.tau(case.z_resp)
        bound = END_TO_END_FLOOR_EPS * EPS * K_WAVENUMBER * eta_R + END_TO_END_FLOOR_ABS

        def groups_with(phase):
            stand_in = copy.copy(G)
            stand_in.phase = phase
            return build_phase_groups(
                regime,
                Gk=stand_in,
                Tq=case.Tq_real,
                Tr=case.Tr_real,
                model_functions=case.model.functions,
                w_background=case.model.functions.wBackground,
            )

        new_groups = groups_with(G.phase)
        old_groups = groups_with(G.old_phase)
        self.assertEqual([g.signs for g in new_groups], [g.signs for g in old_groups])
        log_z = case.random_log_z(case.range_for(regime))
        worst = 0.0
        moved = 0
        for g_new, g_old in zip(new_groups, old_groups):
            self.assertNotEqual(g_new.signs[0], 0)
            for u in log_z:
                d = circular_distance(g_new.theta_mod_2pi(u), g_old.theta_mod_2pi(u))
                worst = max(worst, d)
                moved += d > 0.0
        print(
            f"[05 test 6] composed remainder, new against old: worst {worst:.3e} rad, "
            f"{moved} of {len(new_groups) * len(log_z)} moved; bound 4 eps k eta_R + 1e-14 = "
            f"{bound:.3e} rad (k eta_R = {K_WAVENUMBER * eta_R:.4g} rad)"
        )
        self.assertLessEqual(worst, bound)


if __name__ == "__main__":
    unittest.main()
