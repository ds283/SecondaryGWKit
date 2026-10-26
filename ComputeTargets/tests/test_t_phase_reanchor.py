"""
Tests for the node-anchored remainder of the transfer function's phase
(``prompts/levin-phase-floor`` prompt 05a §3: rows 1 and 2 for ``T``).

**What changed.** ``TkSourceFunctions`` passes a new constructor keyword, ``exact_remainder``
(default ``True``, which production never overrides), to the ``PrimitivePhase`` of its WKB region.
Before 05a, ``T``'s ``theta_mod_2pi`` reduced ``raw_theta``, one double of size ``k tau_s(z)``,
against the double ``TWO_PI``: every remainder carried ``eps k tau_s`` of rounding that is not
smooth in ``z`` (row 1), and a smooth ``0.18 eps |theta_T|`` of modulus error (row 2). With the
keyword on it is ``c_{n_z} + rem(v(z))``, exactly as 05 made ``G``'s: a per-node constant
``c_n = (k [T(n_R) - T(n) + p_R]) mod 2pi``, reduced against 2pi itself at 256 bits, plus
``v = -k p_z + phi(u)`` in double (``primitive_phase.py``'s module docstring, "Anchoring").

**What is new for T: ``p_R != 0``.** ``G``'s anchor is a table node in production; ``T``'s,
``z_init``, is a root, so its end partial ``p_R`` is a non-zero double that enters every ``c_n``.
Tests 2-5 run with ``z_init`` on a node and off one.

The six tests of prompt 05a §3, in order:

1. **Production's configuration** (``TestProductionConfiguration``): on a table ``cs_tau`` with no
   keyword the phase is node-anchored; on a ``ClosedFormPrimitive`` with no keyword it raises the
   ``ValueError`` naming the missing accessors; with ``exact_remainder=False`` it builds.
2. **Exactness against 2pi** (``TestExactness``): within ``4 eps (k |p_z| + 2pi)`` of the exact
   remainder, at four ``k``, with ``z_init`` on a node and off one.
3. **No jitter** (``TestNoJitter``): 04's sigma_theta on a window of relative width 1e-6 *in z*,
   sampled on 65 doubles in ``z`` so that no ``expm1`` rounding enters; the ``u``-sampled
   sigma_theta and ``eps z/(1+z) |d theta/du|`` are printed beside it, unscored.
4. **Nothing else moves** (``TestNothingElseMoves``): (a) ``raw_theta`` and ``theta_deriv`` are
   bit-identical with the keyword on and off; (c) ``G``'s phase from
   ``GkSourcePolicyData._build_phase`` is still node-anchored. (b), keyword off is
   ``WKB_mod_2pi(raw_theta)[1]`` bit for bit, is in ``test_g_phase_reanchor``, where it replaces
   05's ``test_transfer_function_remainders_are_unchanged`` (prompt 05a §2).
5. **T itself** (``TestTWKB``): ``T_WKB`` moves by no more than its amplitude times the old error.
6. **End to end** (``TestEndToEnd``): ``test_phase_groups``' realistic fixture at w = 1/3, regime
   ``G q r``, with ``G`` as production builds it (05's ``_ProductionPhaseGk``) and both ``T``
   factors on a table ``cs_tau`` with the keyword on.

**The backgrounds.** Tests 2-5 need production's magnitudes (``k |Delta_{tau_s}|`` up to ~2e12 rad),
so they use ``_LCDMTk``: 05's flat Lambda-CDM with radiation in 1/Mpc and its 100-per-decade grid
from 1e12 to 0.1, with ``cs_tau`` a real ``CumulativeTable`` of ``c_s/H`` at
``CS_TAU_GAUSS_ORDER``, shifted by ``c_s tau_init`` as ``compute_background`` shifts it, and
``c_s^2 = 1/3``. ``z_init`` is the grid node at 1e9, or a point three tenths of the way from it to
the next node up; the stored phase samples are ``WKB_mod_2pi(k cs_tau.delta(z, z_init) + phi(u))``
at the grid nodes at and below ``z_init``, down to 0.1. Tests 1 and 6 use 05's
``_TableCsTauFixture``: ``test_tk_source_functions``' exact Bessel fixture with a table ``cs_tau``.

No Ray, no datastore. mpmath precision is set in private contexts only.
"""

import copy
import unittest
from math import expm1, log1p, pi, sqrt
from types import SimpleNamespace

import numpy as np

from ComputeTargets.BackgroundModel import (
    CS_TAU_GAUSS_ORDER,
    ModelFunctions,
    TablePrimitive,
)
from ComputeTargets.GkSourcePolicyData import GK_PHASE_SIGN, _build_phase
from ComputeTargets.TkSourceFunctions import TK_PHASE_SIGN, TkSourceFunctions
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.cumulative_table import CumulativeTable
from ComputeTargets.handover import HANDOVER_OVERLAP_NODES
from ComputeTargets.phase_groups import build_phase_groups
from ComputeTargets.tests import test_phase_groups
from ComputeTargets.tests.test_g_phase_reanchor import (
    EPS,
    H0_INV_MPC,
    OMEGA_M,
    OMEGA_R,
    _Background,
    _Hubble,
    _MP,
    _ProductionPhaseGk,
    _TableCsTauFixture,
    _Value,
    _grid,
    _reference_remainder,
    _wrap_mp,
)
from ComputeTargets.tests.test_phase_groups import (
    Case,
    K_WAVENUMBER,
    Q_WAVENUMBER,
    R_WAVENUMBER,
    circular_distance,
    max_relative_residual,
)
from ComputeTargets.tests.test_quadsource_integral import exact_Tk_inputs
from ComputeTargets.tests.test_tk_source_functions import (
    FakeNumericValue,
    FakeTkNumeric,
    FakeTkWKB,
    FakeWKBValue,
    Fixture,
    PRODUCTION_SAMPLES_PER_LOG10Z,
    log_grid,
)
from ComputeTargets.tests.wkb_reference import ClosedFormPrimitive
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI

# c_s^2 = wPerturbations, constant (the radiation value)
CS2 = 1.0 / 3.0

# z_init: the grid node at 1e9, or OFF_NODE_FRACTION of the way in log(1+z) from it to the next
# node up (so the node delta assigns to it is still the one at 1e9, and p_R < 0)
Z_INIT_NODE = 300
OFF_NODE_FRACTION = 0.3
Z_WKB_BOTTOM = 0.1
NUMERIC_NODES_ABOVE_INIT = 20

# prompt 05a §3 test 2
EXACTNESS_K = (1.0e5, 3.0e6, 1.0e8, 3.0e8)
EXACTNESS_RANDOM_POINTS = 200
EXACTNESS_NODE_STRIDE = 25
EXACTNESS_BOUND_EPS = 4.0

# prompt 05a §3 test 3. The window must be *resolved* by its 65 samples: numpy.unwrap can only
# recover steps below pi, and where the phase moves by more than ~64 pi across the window the
# fold it adds is a multiple of 2pi per *index*, which the rounding of the sampled doubles turns
# into jitter of the same order as the floor being measured (log, deviation 4). A window of
# relative width 1e-6 in z moves the phase by ~1e-6 z/(1+z) |d theta/du|, so with k |Delta| >= 1e9
# it is resolved only at small z and modest k: here k = 2e5/Mpc at z ~ 0.12, where it moves ~59 rad.
JITTER_K = 2.0e5
JITTER_Z = 0.12
JITTER_WIDTH_REL_Z = 1.0e-6
JITTER_MIN_PHASE = 1.0e9
JITTER_MAX_STEP = 0.5 * pi
SIGMA_POINTS = 65
SIGMA_DEGREE = 4
JITTER_OFF_AT_LEAST = 0.1  # in units of eps |theta_T|
JITTER_ON_FACTOR = 1.0e-3  # of JITTER_OFF_AT_LEAST eps |theta_T|

# prompt 05a §3 tests 5 and 6: the old error, 4 eps k |Delta| + 0.18 eps |theta_T|, plus 1e-14
OLD_ERROR_ANCHOR_EPS = 4.0
OLD_ERROR_MODULUS_EPS = 0.18
OLD_ERROR_ABS = 1.0e-14
T_WKB_POINTS = 200

# prompt 05a §3 test 6
END_TO_END_W = 1.0 / 3.0
END_TO_END_REGIME = (True, True, True)

SEED = 20261026 + 5


# ------------------------------------------------------------------------------------------------
# the Lambda-CDM transfer-function fixture
# ------------------------------------------------------------------------------------------------


def _epsilon(z: float) -> float:
    """``d ln H / d ln(1+z)`` for ``_Hubble``."""
    s = 1.0 + z
    a, b, c = OMEGA_R, OMEGA_M, 1.0 - OMEGA_M - OMEGA_R
    return (4.0 * a * s**4 + 3.0 * b * s**3) / (2.0 * (a * s**4 + b * s**3 + c))


def _d_epsilon_dz(z: float) -> float:
    s = 1.0 + z
    a, b, c = OMEGA_R, OMEGA_M, 1.0 - OMEGA_M - OMEGA_R
    N = 4.0 * a * s**4 + 3.0 * b * s**3
    D = 2.0 * (a * s**4 + b * s**3 + c)
    dN = 16.0 * a * s**3 + 9.0 * b * s**2
    dD = 2.0 * (4.0 * a * s**3 + 3.0 * b * s**2)
    return (dN * D - N * dD) / (D * D)


def _friction_F(z: float) -> float:
    """``dF/dz = (3/2)(1 + c_s^2)/(1+z)``, with an arbitrary constant."""
    return 1.5 * (1.0 + CS2) * log1p(z)


def _phi_true(u):
    """A smooth residual of the size T's carries (rho_T ~ -0.09 rad)."""
    return -0.09 + 0.02 * np.sin(u)


class _LCDMTk:
    """
    A ``BackgroundModel`` stand-in on 05's Lambda-CDM background, with ``cs_tau`` a real
    ``TablePrimitive``, and the ``(Tk_numeric, Tk_WKB)`` stand-ins of a transfer function whose
    stored phase is ``k cs_tau.delta(z, z_init) + phi(u)``. Built once.
    """

    z_nodes = None
    model = None
    cs_tau = None

    @classmethod
    def build(cls):
        if cls.model is not None:
            return
        cls.z_nodes = _grid()
        table = CumulativeTable(
            cls.z_nodes,
            lambda z: sqrt(CS2) / _Hubble(z),
            CS_TAU_GAUSS_ORDER,
            label="cs_tau",
        )
        # compute_background's shift: tau_s(z_top) = c_s(z_top) tau_init, with the radiation-era
        # asymptote for tau_init
        tau_init = 1.0 / (H0_INV_MPC * sqrt(OMEGA_R) * (1.0 + cls.z_nodes[0]))
        cls.cs_tau = TablePrimitive(table.shifted(sqrt(CS2) * tau_init), "cs_tau")
        functions = ModelFunctions(
            Hubble=_Hubble,
            epsilon=_epsilon,
            d_epsilon_dz=_d_epsilon_dz,
            d2_epsilon_dz2=lambda z: 0.0,
            wBackground=lambda z: CS2,
            wPerturbations=lambda z: CS2,
            tau=None,
            T_photon=lambda z: 0.0,
            d_lnH_dz=lambda z: _epsilon(z) / (1.0 + z),
            d2_lnH_dz2=lambda z: 0.0,
            d3_lnH_dz3=lambda z: 0.0,
            d_wPerturbations_dz=lambda z: 0.0,
            d2_wPerturbations_dz2=lambda z: 0.0,
            cs_tau=cls.cs_tau,
            friction_F=ClosedFormPrimitive(_friction_F, label="friction_F"),
        )
        cls.model = SimpleNamespace(functions=functions)

    @classmethod
    def z_init(cls, on_node: bool) -> float:
        cls.build()
        z_n = float(cls.z_nodes[Z_INIT_NODE])
        if on_node:
            return z_n
        u_n = log1p(z_n)
        u_up = log1p(float(cls.z_nodes[Z_INIT_NODE - 1]))
        return expm1(u_n + OFF_NODE_FRACTION * (u_up - u_n))

    @classmethod
    def inputs(cls, k: float, z_init: float):
        """``(Tk_numeric, Tk_WKB)``, sampled on the grid nodes, as production's rows are."""
        cls.build()
        functions = cls.model.functions
        wkb_z = [
            float(z) for z in cls.z_nodes if Z_WKB_BOTTOM <= z <= z_init
        ]  # descending
        above = [float(z) for z in cls.z_nodes if z > z_init][
            -NUMERIC_NODES_ABOVE_INIT:
        ]
        below = wkb_z[: 1 + HANDOVER_OVERLAP_NODES + 3]
        numeric = FakeTkNumeric(
            [
                FakeNumericValue(z, 1.0 / (1.0 + z), -1.0 / (1.0 + z) ** 2)
                for z in above + below
            ],
            z_exit=None,
            stop_deltaz_subh=None,
        )
        H_init = _Hubble(z_init)
        values = []
        for z in wkb_z:
            theta = k * cls.cs_tau.delta(z, z_init) + float(_phi_true(log1p(z)))
            div, mod = WKB_mod_2pi(theta)
            values.append(
                FakeWKBValue(
                    z,
                    div,
                    mod,
                    functions.friction_F.delta(z_init, z),
                    H_init / _Hubble(z),
                    Tk_omegaEff_sq(cls.model, k, z),
                )
            )
        return numeric, FakeTkWKB(values, 0.37, z_init)

    @classmethod
    def functions(cls, k: float, on_node: bool, **kwargs) -> TkSourceFunctions:
        z_init = cls.z_init(on_node)
        return TkSourceFunctions(cls.model, k, *cls.inputs(k, z_init), **kwargs)


def _anchor_name(on_node: bool) -> str:
    return "on a node" if on_node else "off a node"


def _table_fixture_functions(fixture: Fixture, **kwargs) -> TkSourceFunctions:
    """
    ``TkSourceFunctions`` on ``fixture``'s own stored samples (captured from its
    ``exact_functions``, as ``test_quadsource_integral.exact_Tk_inputs`` captures them), read
    against ``fixture.exact_envelope_model()``: a table ``cs_tau`` for a ``_TableCsTauFixture``, a
    ``ClosedFormPrimitive`` for a plain ``Fixture``. ``kwargs`` go to ``TkSourceFunctions`` as given,
    so that with none it is built as production builds it.
    """
    Tk_numeric, Tk_WKB = exact_Tk_inputs(fixture)
    return TkSourceFunctions(
        fixture.exact_envelope_model(), fixture.k_object, Tk_numeric, Tk_WKB, **kwargs
    )


def _old_error(T: TkSourceFunctions, x: float, x_is_log: bool) -> float:
    """
    The keyword-off remainder's known error at ``x``: ``4 eps k |Delta_{tau_s}|`` of anchoring
    rounding plus ``0.18 eps |theta_T|`` of modulus error (prompt 05a §3 test 5).
    """
    phase = T.phase
    z = expm1(x) if x_is_log else float(x)
    delta = phase._leading.delta(z, phase.z_anchor)
    theta = phase.raw_theta(x, x_is_log=x_is_log)
    return OLD_ERROR_ANCHOR_EPS * EPS * phase.k * abs(
        delta
    ) + OLD_ERROR_MODULUS_EPS * EPS * abs(theta)


def _sigma_theta(fn, a: float, b: float):
    """
    04's sigma_theta (03's ``analyse.sigma_theta``) of ``fn`` at 65 equispaced doubles of
    ``[a, b]``, ends included: ``numpy.unwrap``, a degree-4 least-squares polynomial in ``t`` in
    [-1, 1], and the RMS residual. Also returns the largest step of the unwrapped samples, which
    must be well below pi for the window to be resolved.
    """
    x = np.linspace(a, b, SIGMA_POINTS)
    th = np.unwrap(np.array([fn(float(xi)) for xi in x]))
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, th, SIGMA_DEGREE)
    res = th - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res))), float(np.max(np.abs(np.diff(th))))


# ------------------------------------------------------------------------------------------------
# 1. production's configuration
# ------------------------------------------------------------------------------------------------


class TestProductionConfiguration(unittest.TestCase):
    """Prompt 05a §3 test 1."""

    def test_table_cs_tau_with_no_keyword_is_node_anchored(self):
        T = _table_fixture_functions(_TableCsTauFixture(END_TO_END_W))
        self.assertIsInstance(T.phase._leading, TablePrimitive)
        self.assertTrue(T.phase.exact_remainder)
        self.assertEqual(T.phase.sign, TK_PHASE_SIGN)
        for on_node in (True, False):
            with self.subTest(anchor=_anchor_name(on_node)):
                T = _LCDMTk.functions(3.0e6, on_node)
                self.assertTrue(T.phase.exact_remainder)

    def test_closed_form_cs_tau_with_no_keyword_raises(self):
        fixture = Fixture(END_TO_END_W)
        self.assertIsInstance(
            fixture.exact_envelope_model().functions.cs_tau, ClosedFormPrimitive
        )
        with self.assertRaisesRegex(
            ValueError, "ClosedFormPrimitive lacks end_partial, node_limbs"
        ):
            _table_fixture_functions(fixture)

    def test_closed_form_cs_tau_with_keyword_off_builds(self):
        T = _table_fixture_functions(Fixture(END_TO_END_W), exact_remainder=False)
        self.assertFalse(T.phase.exact_remainder)
        self.assertIsInstance(T.phase._leading, ClosedFormPrimitive)


# ------------------------------------------------------------------------------------------------
# 2. exactness against 2pi
# ------------------------------------------------------------------------------------------------


class TestExactness(unittest.TestCase):
    """
    Prompt 05a §3 test 2: within ``4 eps (k |p_z| + 2pi)`` of the exact remainder of theta_T
    against 2pi, with the reference formed in mpmath from the same limbs, the same ``p_R`` and
    ``p_z`` and the same ``phi`` double (05's ``_reference_remainder``, which reads ``p_R`` from
    the table itself, not from the phase).
    """

    def test_remainder_against_two_pi(self):
        rng = np.random.default_rng(SEED)
        lines = []
        for on_node in (True, False):
            z_init = _LCDMTk.z_init(on_node)
            _j_R, p_R = _LCDMTk.cs_tau.end_partial(z_init)
            if on_node:
                self.assertEqual(p_R, 0.0)
            else:
                self.assertNotEqual(p_R, 0.0)
            for k in EXACTNESS_K:
                with self.subTest(anchor=_anchor_name(on_node), k=k):
                    T = _LCDMTk.functions(k, on_node)
                    phase = T.phase
                    WKB_top, WKB_bottom = T.WKB_region
                    # u spanning the WKB range from z_init to its bottom, and every 25th stored
                    # sample, z_init and the WKB region's two ends exactly, as z
                    u_points = [
                        float(u)
                        for u in rng.uniform(
                            log1p(WKB_bottom), log1p(z_init), EXACTNESS_RANDOM_POINTS
                        )
                    ]
                    z_points = sorted(
                        {float(z) for z in phase.z_samples[::EXACTNESS_NODE_STRIDE]}
                        | {z_init, WKB_top, WKB_bottom}
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
                        bound = EXACTNESS_BOUND_EPS * EPS * (k * abs(p_z) + 2.0 * pi)
                        if err / bound > worst:
                            worst, worst_at = err / bound, (x, x_is_log, err)
                    lines.append(
                        f"   z_init {_anchor_name(on_node)} ({z_init:.6g}, k p_R = "
                        f"{k * p_R:.4g} rad) k={k:.0e}: worst err/bound {worst:.3f} "
                        f"({worst_at[2]:.3e} rad), {len(u_points) + len(z_points)} points, "
                        f"{len(phase._node_constants)} node constants, max |theta_T| "
                        f"{abs(phase.raw_theta(WKB_bottom)):.4g} rad"
                    )
                    self.assertLessEqual(worst, 1.0, worst_at)
        print(
            "\n[05a test 2] T's node-anchored remainder against the exact one (vs 2pi):"
        )
        print("\n".join(lines))


# ------------------------------------------------------------------------------------------------
# 3. no jitter
# ------------------------------------------------------------------------------------------------


class TestNoJitter(unittest.TestCase):
    """
    Prompt 05a §3 test 3. The window is centred on the table node nearest ``JITTER_Z``, with
    ``z_init`` off a node (production's shape), and is sampled on 65 doubles in ``z``, so that the
    rounding of ``expm1`` (prompt 05a §0.1) does not enter.

    **Why a node.** The node-anchored remainder's own residual is the rounding of
    ``v = -k p_z + phi``, about ``eps k |p_z|``. For ``G`` that was negligible beside the floor
    ``eps k tau_R``. For ``T`` it is not: half a table interval of ``tau_s`` is about 1% of
    ``|d theta_T/du|/k``, so midway between nodes ``eps k |p_z|`` is ~1e-3-1e-2 of ``eps |theta_T|``,
    above the 1e-4 this test asserts; at a node it vanishes. The midway window is measured too,
    and printed beside the scored one, unscored (log, deviation 4).
    """

    def test_sigma_theta_off_and_on(self):
        on = _LCDMTk.functions(JITTER_K, on_node=False)
        off = _LCDMTk.functions(JITTER_K, on_node=False, exact_remainder=False)
        z_nodes = _LCDMTk.z_nodes
        j = int(np.argmin(np.abs(z_nodes - JITTER_Z)))
        windows = {
            "at a node": float(z_nodes[j]),
            "midway between nodes": expm1(
                0.5 * (log1p(z_nodes[j]) + log1p(z_nodes[j - 1]))
            ),
        }
        out = {}
        for name, z_c in windows.items():
            a = z_c * (1.0 - 0.5 * JITTER_WIDTH_REL_Z)
            b = z_c * (1.0 + 0.5 * JITTER_WIDTH_REL_Z)
            theta = abs(off.phase.raw_theta(z_c))
            k_delta = JITTER_K * abs(_LCDMTk.cs_tau.delta(z_c, off.phase.z_anchor))
            scale = EPS * theta
            rec = {"z": z_c, "theta": theta, "k_delta": k_delta, "scale": scale}
            for mode, T in (("off", off), ("on", on)):
                rec[f"z_{mode}"], rec[f"z_{mode}_step"] = _sigma_theta(
                    lambda z: T.phase.theta_mod_2pi(z, x_is_log=False), a, b
                )
                rec[f"u_{mode}"], _ = _sigma_theta(
                    lambda u: T.phase.theta_mod_2pi(u, x_is_log=True),
                    log1p(a),
                    log1p(b),
                )
            rec["abscissa"] = (
                EPS
                * z_c
                / (1.0 + z_c)
                * abs(on.phase.theta_deriv(z_c, log_derivative=True))
            )
            rec["nodes"] = {
                _LCDMTk.cs_tau.end_partial(float(z))[0]
                for z in np.linspace(a, b, SIGMA_POINTS)
            }
            out[name] = rec
            print(
                f"\n[05a test 3] {name}: z = {z_c!r}, k = {JITTER_K:.0e}/Mpc, "
                f"theta_T = {off.phase.raw_theta(z_c)!r} rad (k |Delta| = {k_delta:.6g} rad), "
                f"eps |theta_T| = {scale:.4e} rad; largest unwrapped step "
                f"{rec['z_off_step']:.3f} rad\n"
                f"   sampled in z: sigma_theta off {rec['z_off']:.4e} rad "
                f"({rec['z_off'] / scale:.4f} eps|theta|), on {rec['z_on']:.4e} rad "
                f"({rec['z_on'] / scale:.3e} eps|theta|)\n"
                f"   sampled in u (unscored): off {rec['u_off']:.4e} rad "
                f"({rec['u_off'] / scale:.4f} eps|theta|), on {rec['u_on']:.4e} rad "
                f"({rec['u_on'] / scale:.4f} eps|theta|); eps z/(1+z) |d theta/du| = "
                f"{rec['abscissa']:.4e} rad ({rec['abscissa'] / scale:.4f} eps|theta|)"
            )

        # the scored window
        rec = out["at a node"]
        self.assertGreaterEqual(rec["k_delta"], JITTER_MIN_PHASE)
        self.assertEqual(len(rec["nodes"]), 1)
        # resolved: numpy.unwrap recovers every step
        self.assertLess(rec["z_off_step"], JITTER_MAX_STEP)
        self.assertLess(rec["z_on_step"], JITTER_MAX_STEP)
        self.assertGreaterEqual(rec["z_off"], JITTER_OFF_AT_LEAST * rec["scale"])
        self.assertLessEqual(
            rec["z_on"], JITTER_ON_FACTOR * JITTER_OFF_AT_LEAST * rec["scale"]
        )


# ------------------------------------------------------------------------------------------------
# 4. nothing else moves
# ------------------------------------------------------------------------------------------------


class TestNothingElseMoves(unittest.TestCase):
    """Prompt 05a §3 test 4 (a) and (c); (b) is in ``test_g_phase_reanchor``."""

    def test_raw_theta_and_theta_deriv_are_unchanged(self):
        rng = np.random.default_rng(SEED + 4)
        n = 0
        for on_node in (True, False):
            for k in (3.0e6, 3.0e8):
                with self.subTest(anchor=_anchor_name(on_node), k=k):
                    on = _LCDMTk.functions(k, on_node).phase
                    off = _LCDMTk.functions(k, on_node, exact_remainder=False).phase
                    self.assertTrue(on.exact_remainder)
                    self.assertFalse(off.exact_remainder)
                    u_points = [
                        float(u)
                        for u in rng.uniform(
                            log1p(on.min_x), log1p(on.z_anchor), EXACTNESS_RANDOM_POINTS
                        )
                    ]
                    z_points = [float(z) for z in on.z_samples[::EXACTNESS_NODE_STRIDE]]
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
            f"\n[05a test 4(a)] T's raw_theta and theta_deriv bit-identical, keyword on and "
            f"off: {n} abscissae, z_init on and off a node"
        )

    def test_G_phase_from_build_phase_is_still_node_anchored(self):
        """(c): ``_build_phase`` on a stand-in ``GkSource`` over 05's realistic ``tau`` table."""
        _Background.build()
        k = 3.0e6
        z_response = 0.1
        tau = _Background.tau
        z = _Background.z_nodes[::-1]
        z_samples = z[(z >= z_response) & (z <= 1.0e4)]
        values = []
        for zz in z_samples:
            theta = GK_PHASE_SIGN * k * tau.delta(float(zz), z_response) + 0.01
            div, mod = WKB_mod_2pi(theta)
            values.append(_Value(zz, div, mod))
        functions = SimpleNamespace(tau=tau, Hubble=_Hubble)
        source = SimpleNamespace(
            model_proxy=SimpleNamespace(
                get=lambda: SimpleNamespace(functions=functions)
            ),
            k=SimpleNamespace(k=k),
            z_response=SimpleNamespace(z=z_response),
        )
        phase = _build_phase(source, values, "test G")
        self.assertTrue(phase.exact_remainder)
        self.assertEqual(phase.sign, GK_PHASE_SIGN)


# ------------------------------------------------------------------------------------------------
# 5. T itself
# ------------------------------------------------------------------------------------------------


class TestTWKB(unittest.TestCase):
    """
    Prompt 05a §3 test 5: ``T_WKB`` moves between the keyword off and on by no more than
    ``|M| (4 eps k |Delta_{tau_s}| + 0.18 eps |theta_T|)`` plus 1e-14 of ``|M|``.
    """

    def test_T_WKB_moves_by_at_most_the_old_error(self):
        rng = np.random.default_rng(SEED + 5)
        lines = []
        for on_node in (True, False):
            for k in EXACTNESS_K:
                with self.subTest(anchor=_anchor_name(on_node), k=k):
                    on = _LCDMTk.functions(k, on_node)
                    off = _LCDMTk.functions(k, on_node, exact_remainder=False)
                    WKB_top, WKB_bottom = on.WKB_region
                    z_points = [
                        expm1(float(u))
                        for u in rng.uniform(
                            log1p(WKB_bottom), log1p(WKB_top), T_WKB_POINTS
                        )
                    ] + [WKB_top, WKB_bottom]
                    worst = 0.0
                    moved = 0
                    for z in z_points:
                        M = abs(on.M(z))
                        self.assertEqual(float.hex(M), float.hex(abs(off.M(z))))
                        d = abs(on.T_WKB(z) - off.T_WKB(z))
                        # T_WKB evaluates its phase at log(1+z), x_is_log=True
                        log_z = log1p(z)
                        bound = M * (_old_error(off, log_z, True) + OLD_ERROR_ABS)
                        worst = max(worst, d / bound)
                        moved += d > 0.0
                    lines.append(
                        f"   z_init {_anchor_name(on_node)}, k={k:.0e}: worst |dT|/bound "
                        f"{worst:.3e}, {moved} of {len(z_points)} moved"
                    )
                    self.assertLessEqual(worst, 1.0)
        print("\n[05a test 5] T_WKB, keyword on against off:")
        print("\n".join(lines))


# ------------------------------------------------------------------------------------------------
# 6. end to end
# ------------------------------------------------------------------------------------------------


class TestEndToEnd(unittest.TestCase):
    """
    Prompt 05a §3 test 6, at one cheap configuration: w = 1/3, regime G q r. ``G`` is 05's
    ``_ProductionPhaseGk`` (its phase from ``_build_phase``); ``T_q`` and ``T_r`` are
    ``_TableCsTauFixture``s at ``test_phase_groups``' wavenumbers and sampling, built as
    production builds them (keyword on) and with the keyword off.
    """

    @classmethod
    def setUpClass(cls):
        base = Case.get(END_TO_END_W)
        lowest = min(base.Tq_real.WKB_region[1], base.Tr_real.WKB_region[1])
        z_grid = log_grid(
            1.0 + base.both_numeric[0], 1.0 + lowest, PRODUCTION_SAMPLES_PER_LOG10Z
        )
        cls.G = _ProductionPhaseGk(
            K_WAVENUMBER, END_TO_END_W, base.z_resp, base.model, z_grid
        )
        T = {}
        for name, k in (("q", Q_WAVENUMBER), ("r", R_WAVENUMBER)):
            fixture = _TableCsTauFixture(
                END_TO_END_W,
                k=k,
                WKB_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
                numeric_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
            )
            T[name, True] = _table_fixture_functions(fixture)
            T[name, False] = _table_fixture_functions(fixture, exact_remainder=False)
        cls.T = T
        # the same Case with G as production builds it and both T factors keyword-on
        cls.case = copy.copy(base)
        cls.case.Gk_real = cls.G
        cls.case.Tq_real = T["q", True]
        cls.case.Tr_real = T["r", True]
        cls.case.rng = np.random.default_rng(SEED + 6)

    def test_evaluate_sum_against_the_direct_product(self):
        case = self.case
        regime = END_TO_END_REGIME
        self.assertTrue(case.Tq_real.phase.exact_remainder)
        self.assertTrue(case.Tr_real.phase.exact_remainder)
        self.assertTrue(case.Gk_real.phase.exact_remainder)
        groups = case.groups(regime, exact=False)
        log_z = case.random_log_z(case.range_for(regime))
        worst = max_relative_residual(
            groups,
            lambda u: case.direct_product(regime, u, exact=False),
            log_z,
            extra_scale=lambda u: case.term_envelope(regime, u, exact=False),
        )
        print(
            f"\n[05a test 6] Oracle 1 with G from _build_phase and both T node-anchored "
            f"(w=1/3, G q r): max relative residual {worst:.3e} (bound "
            f"{test_phase_groups.TestOracle1DirectProduct.TOLERANCE:g})"
        )
        self.assertLess(worst, test_phase_groups.TestOracle1DirectProduct.TOLERANCE)

    def test_composed_remainder_moves_by_at_most_the_T_factors_old_errors(self):
        case = self.case
        regime = END_TO_END_REGIME

        def groups_with(on):
            return build_phase_groups(
                regime,
                Gk=self.G,
                Tq=self.T["q", on],
                Tr=self.T["r", on],
                model_functions=case.model.functions,
                w_background=case.model.functions.wBackground,
            )

        new_groups = groups_with(True)
        old_groups = groups_with(False)
        self.assertEqual([g.signs for g in new_groups], [g.signs for g in old_groups])
        log_z = case.random_log_z(case.range_for(regime))
        worst = 0.0
        worst_d = 0.0
        moved = 0
        for g_new, g_old in zip(new_groups, old_groups):
            for u in log_z:
                d = circular_distance(g_new.theta_mod_2pi(u), g_old.theta_mod_2pi(u))
                bound = (
                    _old_error(self.T["q", False], u, True)
                    + _old_error(self.T["r", False], u, True)
                    + OLD_ERROR_ABS
                )
                worst = max(worst, d / bound)
                worst_d = max(worst_d, d)
                moved += d > 0.0
        print(
            f"[05a test 6] composed remainder, T keyword on against off: worst {worst_d:.3e} "
            f"rad, worst / bound {worst:.3e}, {moved} of "
            f"{len(new_groups) * len(log_z)} moved"
        )
        self.assertLessEqual(worst, 1.0)


if __name__ == "__main__":
    unittest.main()
