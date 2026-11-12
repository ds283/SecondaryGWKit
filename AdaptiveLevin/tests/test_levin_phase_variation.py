"""
The phase variation the C2 gate routes on (prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md,
section 1.1 and section 3 tests 1-4, 7 and 8; tests 5 and 6 are test_levin_stationary.py's test 13).

A basis's build_Levin_data() reports V, the phase's total variation across a region, computed from the raw
phase: |theta(b) - theta(a)| where theta' has no two node samples of opposite strict sign, the variation on the
node partition, sum_j |theta(x_{j+1}) - theta(x_j)|, where it has. The C2 gate (V < SIX_PI: the nested
Clenshaw-Curtis rule) and the choice of LU over least squares (V > 20 pi) route on it. Until 07a both routed on
the estimate mean|theta'| * width from the region's Chebyshev nodes, which this module calls "the parent's
estimate" and computes in the tests (_parent_estimate()) to show where the two disagree. The location no longer
depends on any estimate of the variation: a phase is left unlocated only when its rate is zero at every finite
scan sample.

The parent's behaviour quoted in the docstrings was measured by running the dispatch commit's module (a9f6abc3,
07's code) on the same problems; the log (prompts/levin-phase-floor/logs/07a-window-stop-and-phase-variation.md)
records the runs. This module does not run the parent.

References are closed forms: test 2's by docs/levin-phase-floor/07a/code/test_references_07a.py (mpmath at 40
and 60 digits, agreeing to 4.2e-42 relative; out/test_references_07a.json); test 4's is test_levin_stationary.py's
P1_REF. Every honesty assertion checks both that the value is within its abserr of the reference and that abserr
is at least the true error. No Ray and no datastore.
"""

import math
import unittest
from unittest import mock

import numpy as np
from scipy.special import erf

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.levin_quadrature import (
    _Basis_SinCos,
    _adaptive_levin_subregion,
    _adaptive_levin_subregion_impl,
    _changes_sign,
    adaptive_levin_sincos,
    chebyshev_matrices,
)
from AdaptiveLevin.tests.test_levin_quadrature import _TwoIndependentSinCosBasis
from AdaptiveLevin.tests.test_levin_stationary import (
    F_23,
    P1_REF,
    P1_XS,
    P_A,
    P_ATOL,
    P_L,
    P_RTOL,
    T3_CASES,
    X0,
    _Honest,
    _TwoPhaseQuadraticBasis,
    p_problem,
)

N = lq.DEFAULT_LEVIN_CHEBSHEV_ORDER


def _parent_estimate(rate, span, order=N):
    """The parent's estimate of a region's variation: mean|theta'| on its Chebyshev nodes times its width."""
    grid, _ = chebyshev_matrices(span, order)
    return float(np.mean(np.fabs(rate(grid))) * np.fabs(grid[0] - grid[-1]))


def _variation(basis, span, order=N):
    """V as the basis reports it from build_Levin_data() on span's nodes, with theta' on them."""
    grid, Dmat = chebyshev_matrices(span, order)
    _, theta_prime, _, _, V = basis.build_Levin_data(grid, Dmat, need_AmatT=False)
    return V, theta_prime, grid


# The bump of section 0.2: |theta'| a Gaussian of width BUMP_W centred on x = 1/2, theta monotone, scaled so
# that V over [0, 1] is v_total. On 16 nodes the parent's estimate reads 0.473 of V over [0, 1] and 2.32 times
# V over [0, 1/2].
BUMP_W = 0.05


def bump_theta(v_total):
    amplitude = v_total / (BUMP_W * math.sqrt(math.pi) * erf(0.5 / BUMP_W))
    half = 0.5 * amplitude * BUMP_W * math.sqrt(math.pi)
    return {
        "theta": lambda x: half * erf((x - 0.5) / BUMP_W),
        "theta_deriv": lambda x: amplitude * np.exp(-(((x - 0.5) / BUMP_W) ** 2)),
    }


def exp_theta(alpha):
    return {
        "theta": lambda x: np.exp(alpha * x),
        "theta_deriv": lambda x: alpha * np.exp(alpha * x),
    }


# Test 2: theta = e^{alpha x} on [0, 1], V = e^alpha - 1 = 0.98 * 6 pi; f = (1, 0).
T2_ALPHA = math.log1p(0.98 * 6.0 * math.pi)
T2_REF = float.fromhex("0x1.91418ab5f8be8p-3")  # 0.195925792387327425791065062874
F_SIN = [lambda x: 1.0 + 0.0 * x, lambda x: 0.0 * x]


class TestVariationExact(unittest.TestCase, _Honest):
    def test_1_V_exact_on_monotone_regions(self):
        """
        Test 1. On regions where theta is monotone, V is |theta(b) - theta(a)| computed here, bit for bit: for
        _Basis_SinCos with theta_deriv (the raw phase sampled on the nodes, as it is without theta_mod_2pi),
        with theta_deriv and theta_mod_2pi (the production path: the raw phase sampled at the two ends only), and
        without theta_deriv (theta' by spectral differentiation); and for both test bases. theta = 3x + x^3 has
        theta' >= 3; theta = x^3 on [0, 1] has theta' zero at the node x = 0 and no strict sign change, and is
        still exact. The phases are products and sums, so their scalar and array evaluations round alike.
        """
        cubic = lambda x: 3.0 * x + x * x * x
        cubic_rate = lambda x: 3.0 + 3.0 * x * x
        regions = [(0.0, 1.0), (0.1, 0.7), (-2.0, 0.3), (1.0, 0.25), (5.0, 5.0625)]
        variants = {
            "theta_deriv": {"theta": cubic, "theta_deriv": cubic_rate},
            "theta_deriv and theta_mod_2pi": {
                "theta": cubic,
                "theta_deriv": cubic_rate,
                "theta_mod_2pi": lambda x: np.fmod(cubic(x), 2.0 * np.pi),
            },
            "spectral": {"theta": cubic},
        }
        for name, theta in variants.items():
            for a, b in regions:
                with self.subTest(basis=name, region=(a, b)):
                    V, theta_prime, grid = _variation(_Basis_SinCos(theta), (a, b))
                    self.assertFalse(_changes_sign(theta_prime))
                    self.assertEqual(grid[0], b)
                    self.assertEqual(grid[-1], a)
                    want = abs(cubic(b) - cubic(a))
                    self.assertEqual(float(V).hex(), float(want).hex())

        cube = {"theta": lambda x: x * x * x, "theta_deriv": lambda x: 3.0 * x * x}
        V, theta_prime, _ = _variation(_Basis_SinCos(cube), (0.0, 1.0))
        self.assertEqual(theta_prime[-1], 0.0)
        self.assertFalse(_changes_sign(theta_prime))
        self.assertEqual(float(V).hex(), float(1.0).hex())

        two_linear = _TwoIndependentSinCosBasis(50.0, -30.0)
        for a, b in regions:
            with self.subTest(basis="_TwoIndependentSinCosBasis", region=(a, b)):
                V, _, _ = _variation(two_linear, (a, b))
                want = abs(50.0 * b - 50.0 * a) + abs(-30.0 * b - -30.0 * a)
                self.assertEqual(float(V).hex(), float(want).hex())

        two_phase = _TwoPhaseQuadraticBasis(200.0, 1.0e5, X0)
        phases = two_phase.independent_phases()
        for a, b in [(0.5, 0.9), (0.0, 0.3), (0.75, 0.5)]:
            with self.subTest(basis="_TwoPhaseQuadraticBasis", region=(a, b)):
                V, _, _ = _variation(two_phase, (a, b))
                ends = np.array([b, a])
                want = 0.0
                for phase in phases:
                    theta_b, theta_a = phase(ends)
                    want = want + abs(theta_b - theta_a)
                self.assertEqual(float(V).hex(), float(want).hex())

    def test_2_end_weighted_shape(self):
        """
        Test 2. theta = e^{alpha x} on [0, 1], alpha = log1p(0.98 * 6 pi), so V = e^alpha - 1 = 18.4726, just
        below 6 pi = 18.8496, while the parent's estimate, which leans to |theta'| at the right end where the
        nodes crowd, reads 21.9403 = 6.98 pi. At the parent the region went to the Levin solve (is_direct False,
        phase_span 21.940294566299677); under the commit it goes to the Clenshaw-Curtis rule. The call
        (adaptive_levin_sincos, f = (1, 0), rtol 1e-10) is honest against (Si(e^alpha) - Si(1)) / alpha; at the
        parent it returned the same value and abserr, its root leaf bisected all the same.
        """
        theta = exp_theta(T2_ALPHA)
        V, _, _ = _variation(_Basis_SinCos(theta), (0.0, 1.0))
        self.assertEqual(float(V).hex(), float(abs(math.exp(T2_ALPHA) - 1.0)).hex())
        self.assertLess(V, lq.SIX_PI)
        estimate = _parent_estimate(theta["theta_deriv"], (0.0, 1.0))
        self.assertGreater(estimate, lq.SIX_PI)
        self.assertAlmostEqual(estimate, 21.940294566299677, delta=1e-9)

        solve = _adaptive_levin_subregion(
            (0.0, 1.0), F_SIN, _Basis_SinCos(theta), id_label=None, rtol=1e-10
        )
        self.assertTrue(solve["is_direct"])
        self.assertEqual(solve["phase_span"], V)

        data = adaptive_levin_sincos((0.0, 1.0), F_SIN, theta, atol=1e-16, rtol=1e-10)
        self.assertEqual(data["stationary_points"], [])
        self.assertTrue(data["converged"])
        self.assertHonest(data, T2_REF)


class TestMonotoneUnderBisection(unittest.TestCase):
    def test_3_no_half_of_a_cc_region_goes_to_levin(self):
        """
        Test 3. On the bump of section 0.2 (V = 10 pi over [0, 1]) and two other shapes -- theta = e^{4x}, and a
        Lorentzian rate weighted to the left end, theta' = A / (1 + (x / 0.05)^2) with V = 12 pi -- every
        region the driver solves is recorded (adaptive_levin_sincos at rtol 1e-12, F_23): no half of a region
        the gate sent to the Clenshaw-Curtis rule is sent to the Levin solve. On a monotone phase V is additive,
        so a half never reads more than its parent.

        At the parent the bump violates it, as section 0.2 says: [0, 1] read 4.7308 pi and went to the
        Clenshaw-Curtis rule, and each of its halves read 11.5877 pi and went to the Levin solve (2 such
        pairs among 159 regions visited). The parent's estimates are recomputed here; the commit routes [0, 1]
        to the Levin solve (V = 10 pi) and its halves (V = 5 pi) to the Clenshaw-Curtis rule. The other two
        shapes show no violation at either code.
        """
        bump = bump_theta(10.0 * math.pi)
        self.assertLess(_parent_estimate(bump["theta_deriv"], (0.0, 1.0)), lq.SIX_PI)
        self.assertGreater(_parent_estimate(bump["theta_deriv"], (0.0, 0.5)), lq.SIX_PI)
        self.assertAlmostEqual(
            _parent_estimate(bump["theta_deriv"], (0.0, 1.0)) / math.pi,
            4.730842969258386,
            delta=1e-9,
        )

        lorentz_width = 0.05
        lorentz_amplitude = 12.0 * math.pi / (lorentz_width * math.atan(1.0 / 0.05))
        shapes = {
            "bump": bump,
            "exp": exp_theta(4.0),
            "lorentzian": {
                "theta": lambda x: lorentz_amplitude
                * lorentz_width
                * np.arctan(x / lorentz_width),
                "theta_deriv": lambda x: lorentz_amplitude
                / (1.0 + (x / lorentz_width) ** 2),
            },
        }
        for name, theta in shapes.items():
            with self.subTest(shape=name):
                visited = {}
                solve = lq._adaptive_levin_subregion

                def recording(span, *args, **kwargs):
                    data = solve(span, *args, **kwargs)
                    key = (min(span), max(span))
                    visited.setdefault(key, set()).add(
                        (bool(data["is_direct"]), bool(kwargs.get("force_direct")))
                    )
                    return data

                with mock.patch.object(lq, "_adaptive_levin_subregion", recording):
                    data = adaptive_levin_sincos(
                        (0.0, 1.0), F_23, theta, atol=1e-16, rtol=1e-12
                    )
                self.assertEqual(data["stationary_points"], [])
                cc_with_halves = 0
                for (a, b), routes in visited.items():
                    if not any(
                        is_direct and not forced for is_direct, forced in routes
                    ):
                        continue
                    c = (a + b) / 2.0
                    for half in ((a, c), (c, b)):
                        if half in visited:
                            cc_with_halves += 1
                            self.assertTrue(
                                all(is_direct for is_direct, _ in visited[half]),
                                f"{name}: half {half} of a Clenshaw-Curtis region went to the Levin solve",
                            )
                self.assertGreater(cc_with_halves, 0)
                if name == "bump":
                    self.assertEqual(visited[(0.0, 1.0)], {(False, False)})
                    self.assertEqual(visited[(0.0, 0.5)], {(True, False)})
                    self.assertEqual(visited[(0.5, 1.0)], {(True, False)})


class TestNonMonotoneRegion(unittest.TestCase, _Honest):
    def test_4_node_partition_sum(self):
        """
        Test 4. Test 4's problem of test_levin_stationary.py (P1 with theta alone, no rates): on the root region
        [0, 16] theta' by spectral differentiation changes sign on the nodes (its root is x_s = 9.38), so V is
        the node-partition sum, bit for bit, a lower bound on the true variation |theta(x_s) - theta(0)| +
        |theta(16) - theta(x_s)|, and above the net change |theta(16) - theta(0)|; and the call is honest.
        """
        f, theta = p_problem(P1_XS, with_deriv=False)
        basis = _Basis_SinCos(theta)
        V, theta_prime, grid = _variation(basis, (0.0, P_L))
        self.assertTrue(_changes_sign(theta_prime))
        samples = theta["theta"](grid)
        want = math.fsum(
            abs(float(samples[j + 1]) - float(samples[j]))
            for j in range(len(samples) - 1)
        )
        self.assertEqual(float(V).hex(), float(want).hex())
        a_coeff = P_A * math.exp(P1_XS)
        th = lambda x: P_A * math.exp(x) - a_coeff * x
        true_variation = abs(th(P1_XS) - th(0.0)) + abs(th(P_L) - th(P1_XS))
        self.assertLessEqual(V, true_variation * (1.0 + 1e-12))
        self.assertGreater(V, abs(th(P_L) - th(0.0)))

        data = adaptive_levin_sincos((0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL)
        self.assertFalse(any(r.type == "Levin" and r.guarded for r in data["regions"]))
        self.assertHonest(data, P1_REF)


class TestMinimaStayLocated(unittest.TestCase, _Honest):
    def test_7_nearly_coalescing_pair_located(self):
        """
        Test 7. Test 3's nearly coalescing pair (test_levin_stationary.py; family (C) at lambda = 1, b = 1e4 and
        1e8): the rate a + b (x - x0)^2 is of one strict sign on every scan sample, and nothing is skipped; the
        minimum of |theta'| at x0 is located, kind "minimum", and windowed, and the value is honest -- as at the
        parent, where test 3 passes with the same points.
        """
        scan = np.linspace(0.0, 1.0, lq._LEVIN_STATIONARY_SCAN_POINTS)
        for b, a, ref in T3_CASES:
            with self.subTest(b=b):
                theta = {
                    "theta": lambda x, a=a, b=b: a * (x - X0) + b / 3.0 * (x - X0) ** 3,
                    "theta_deriv": lambda x, a=a, b=b: a + b * (x - X0) ** 2,
                }
                self.assertTrue(np.all(theta["theta_deriv"](scan) > 0.0))
                data = adaptive_levin_sincos(
                    (0.0, 1.0), F_23, theta, atol=1e-14 * abs(ref), rtol=1e-9
                )
                self.assertEqual(data["location_skipped_phases"], [])
                (point,) = data["stationary_points"]
                self.assertEqual(point["kind"], "minimum")
                self.assertLess(point["lambda"], lq._LEVIN_COALESCENCE_LAMBDA)
                self.assertEqual(len(data["stationary_windows"]), 1)
                self.assertGreater(data["num_window_pieces"], 0)
                self.assertHonest(data, ref)


class TestLUChoice(unittest.TestCase):
    def test_8_lu_choice_follows_V(self):
        """
        Test 8. The choice of LU over least squares (V > 20 pi) follows V, on the root region [0, 1], F_23:
          * the bump at V = 21 pi: the parent's estimate reads 9.9348 pi, and at the parent the region took
            least squares (metadata lstsq_solve 1); under the commit it takes LU (direct_solve 1);
          * the reverse, theta = e^{alpha x} with V = 0.98 * 20 pi = 19.6 pi: the parent's estimate reads
            25.9678 pi, and at the parent the region took LU; under the commit it takes least squares.
        Both regions go to the Levin solve under either code (V and the estimate above 6 pi).
        """
        bump = bump_theta(21.0 * math.pi)
        reverse = exp_theta(math.log1p(0.98 * 20.0 * math.pi))
        cases = (
            ("bump", bump, True, 9.934770235442615),
            ("exp", reverse, False, 25.967762989417846),
        )
        for name, theta, lu, parent_over_pi in cases:
            with self.subTest(shape=name):
                estimate = _parent_estimate(theta["theta_deriv"], (0.0, 1.0))
                self.assertAlmostEqual(estimate / math.pi, parent_over_pi, delta=1e-9)
                data = _adaptive_levin_subregion_impl(
                    (0.0, 1.0), F_23, _Basis_SinCos(theta), id_label=None, rtol=1e-10
                )
                self.assertFalse(data["is_direct"])
                V = data["phase_span"]
                if lu:
                    self.assertLess(estimate, lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN)
                    self.assertGreater(V, lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN)
                    self.assertEqual(data["metadata"].get("direct_solve"), 1)
                    self.assertNotIn("lstsq_solve", data["metadata"])
                else:
                    self.assertGreater(estimate, lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN)
                    self.assertLess(V, lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN)
                    self.assertEqual(data["metadata"].get("lstsq_solve"), 1)
                    self.assertNotIn("direct_solve", data["metadata"])


if __name__ == "__main__":
    unittest.main()
