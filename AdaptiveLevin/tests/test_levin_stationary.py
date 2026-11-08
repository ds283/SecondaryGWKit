"""
The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md, section 3).

_adaptive_levin() locates every sign change of each closed-form phase rate a basis supplies, and every
interior minimum of |theta'| whose Airy parameter is below _LEVIN_COALESCENCE_LAMBDA, before its loop; it
integrates a window of phase half-width _LEVIN_WINDOW_HALF_PHASE round each by the nested Clenshaw-Curtis
pair, in pieces of at most 2 pi; it stops a window whose bisections no longer lower its error by QUADPACK's
iroff1 counted per window (section 1.3); and it never returns a Levin leaf whose phase rate changes sign on its
own nodes (the guard). A phase whose total variation over the span is below SIX_PI is not located (Rule A, section
10; test 13), and the guard tests such a phase of a basis of several phases for a sign change.

References are closed forms (the incomplete gamma function and the error function), contour integrals, or
mpmath.quad on pieces of at most pi/2 of phase split at every stationary point. They are computed, each at 40
and at 60 (or 30 and 45) digits, by docs/levin-phase-floor/07/calibration/scripts/test_references.py, whose
output docs/levin-phase-floor/07/calibration/out/run2/test_references.json holds the values used here (as
float hex) and the two readings' agreement (at most 1.5e-30 relative). The parent's behaviour quoted for tests 1
and 5 was measured with the dispatch commit's module (bd21bf23, 06m's code, loaded from git show) by
scripts/parent_cases.py; its output is out/run2/parent_cases.json. This module does not run the parent.

Every accuracy test asserts both that the value lies within its declared abserr of the reference and that
abserr is at least the true error. No Ray and no datastore.
"""

import math
import unittest

import numpy as np

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.levin_quadrature import (
    _Basis_SinCos,
    _adaptive_levin,
    _adaptive_levin_subregion,
    adaptive_levin_sincos,
    chebyshev_matrices,
)
from AdaptiveLevin.tests.test_levin_quadrature import _TwoIndependentSinCosBasis

EPS = float(np.finfo(float).eps)

# --- the problems ------------------------------------------------------------------------------
#
# P1 (tests 1, 4, 6, 7, 10 (c)) and P5 (test 5): on [0, L], theta = A e^x - a x with a = A e^{x_s}, so that
# theta' = A e^x - a has one root, at x_s (non-dyadic), and theta varies by about 1.5e9 rad over the span;
# f_sin = e^{g (x - x_s)}, f_cos = cos(h x).
P_L, P_A, P_G, P_H = 16.0, 169.0, 0.33, 0.78
P1_XS = 9.381966011250105
P5_XS = 10.0918
P1_REF = float.fromhex("0x1.71eac90375ce6p-10")  # 1.41112186326158290958497602806e-3
P5_REF = float.fromhex("-0x1.27089781798d1p-10")  # -1.12546372144488687714088877903e-3
P_ATOL = 1e-20
P_RTOL = 1e-7


def p_problem(x_s, with_deriv=True):
    a = P_A * math.exp(x_s)
    theta = {"theta": lambda x: P_A * np.exp(x) - a * x}
    if with_deriv:
        theta["theta_deriv"] = lambda x: P_A * np.exp(x) - a
    f = [lambda x: np.exp(P_G * (x - x_s)), lambda x: np.cos(P_H * x)]
    return f, theta


# The amplitudes of the cubic and quadratic families of section 2 (tests 2, 3, 10).
F_23 = [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]
X0 = 0.3819660112501051
XS_A = 0.7236067977499790

# Test 2: theta = a (x - x0) + (b/3)(x - x0)^3, a = -b (d/2)^2: roots at x0 -+ d/2, d = 0.01.
T2_B, T2_A, T2_D = 1.0e4, -0.25, 0.01
T2_REF = float.fromhex("0x1.43aaf33358c16p-5")  # 3.95102262507622088794189764873e-2

# Test 3: family (C) at lambda = 1 = Lambda_c / 4, a the double nearest (lambda^2 b)^(1/3).
T3_CASES = (
    (1.0e4, 21.544346900318835, float.fromhex("0x1.e514bad60edf4p-7")),
    (1.0e8, 464.1588833612779, float.fromhex("0x1.8a29b9f63becdp-11")),
)

# Test 9: theta_1 = 200 x, theta_2 = (c/2)(x - x_s)^2, c = 1e5, f = (1, 0, 0, 1).
T9_L1, T9_C, T9_XS = 200.0, 1.0e5, X0
T9_REF = float.fromhex("0x1.0b82123e1329fp-7")  # 8.16369906183328340852021528759e-3

# Test 10: family (R), theta = (c/2)(x - x_s)^2, c = 1e6.
T10_C = 1.0e6
T10A_REF = float.fromhex("-0x1.2c63942e63e04p-13")  # x_s = 0.7236...
T10B_REF = float.fromhex("0x1.fd3bb3697bb72p-10")  # x_s = 0.3819...


def quadratic_theta(c, x_s):
    return {
        "theta": lambda x: 0.5 * c * (x - x_s) ** 2,
        "theta_deriv": lambda x: c * (x - x_s),
    }


class _Honest:
    def assertHonest(self, data, ref, margin=0.0):
        """Within abserr of the reference, abserr at least the true error (with margin: the reference's own
        uncertainty, added to the true error)."""
        err = abs(data["value"] - ref) + margin
        self.assertLessEqual(
            err, data["abserr"], f"error {err:.3e} > abserr {data['abserr']:.3e}"
        )
        self.assertGreaterEqual(data["abserr"], err)


class TestOneStationaryPoint(unittest.TestCase, _Honest):
    def test_1_located_windowed_and_honest(self):
        """
        Test 1. P1: one interior root of theta' at the non-dyadic x_s = 9.381966011250105, theta varying by
        about 1.5e9 rad. The parent (bd21bf23) returns the whole span as one Levin leaf, found floor-limited
        at depth 0 across the point: value -3.1261800297467865e-07 (hex -0x1.4fabc846d3046p-22), abserr
        2.542e-11, 1 region, converged False, true error 1.411e-3 = 5.55e7 times its abserr
        (out/run2/parent_cases.json). Under the guard the point is located within 4 eps |x_s|, it has one
        window, and the value is honest.
        """
        f, theta = p_problem(P1_XS)
        data = adaptive_levin_sincos((0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL)
        points = data["stationary_points"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["kind"], "root")
        self.assertLessEqual(abs(points[0]["x"] - P1_XS), 4.0 * EPS * abs(P1_XS))
        self.assertEqual(len(data["stationary_windows"]), 1)
        self.assertGreater(data["num_window_pieces"], 0)
        self.assertEqual(data["num_guarded_regions"], 0)
        self.assertHonest(data, P1_REF)

    def test_7_window_edges_and_pieces(self):
        """
        Test 7. On test 1's problem, |theta - theta(x_s)| at each edge of the window equals Phi_w to within the
        precision the location states (_locate_stationary_points(): about 5 eps |x| |theta'| + ulp(|theta(x)|)
        + ulp(|theta(x_s)|)), and every piece of the window spans at most 2 pi of phase, to the same precision.
        """
        f, theta = p_problem(P1_XS)
        data = adaptive_levin_sincos((0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL)
        th, rate = theta["theta"], theta["theta_deriv"]
        x_p = data["stationary_points"][0]["x"]
        t_p = float(th(x_p))
        window = data["stationary_windows"][0]
        for edge in (window["lo"], window["hi"]):
            tol = (
                5.0 * EPS * abs(edge) * abs(float(rate(edge)))
                + np.spacing(abs(float(th(edge))))
                + np.spacing(abs(t_p))
            )
            self.assertLessEqual(
                abs(abs(float(th(edge)) - t_p) - lq._LEVIN_WINDOW_HALF_PHASE), tol
            )
        cuts = window["cuts"]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            tol = 2.0 * (
                5.0 * EPS * abs(hi) * max(abs(float(rate(lo))), abs(float(rate(hi))))
            ) + 2.0 * np.spacing(max(abs(float(th(lo))), abs(float(th(hi)))))
            self.assertLessEqual(
                abs(float(th(hi)) - float(th(lo))), lq._LEVIN_WINDOW_PIECE_PHASE + tol
            )


class TestTwoClosePoints(unittest.TestCase, _Honest):
    def test_2_two_points_between_adjacent_nodes(self):
        """
        Test 2. theta' = a + b (x - x0)^2 has two roots, x0 -+ 0.005, both between the same two adjacent
        Chebyshev nodes of the root leaf (order 16 on [0, 1]), so theta' has no sign change on the root's
        nodes and the guard alone cannot see them there. The scan locates both; the value is honest.
        """
        theta = {
            "theta": lambda x: T2_A * (x - X0) + T2_B / 3.0 * (x - X0) ** 3,
            "theta_deriv": lambda x: T2_A + T2_B * (x - X0) ** 2,
        }
        grid, _ = chebyshev_matrices((0.0, 1.0), lq.DEFAULT_LEVIN_CHEBSHEV_ORDER)
        rate_on_nodes = theta["theta_deriv"](grid)
        self.assertTrue(np.all(rate_on_nodes > 0.0) or np.all(rate_on_nodes < 0.0))
        data = adaptive_levin_sincos(
            (0.0, 1.0), F_23, theta, atol=1e-14 * abs(T2_REF), rtol=1e-9
        )
        roots = sorted(p["x"] for p in data["stationary_points"] if p["kind"] == "root")
        self.assertEqual(len(roots), 2)
        for got, want in zip(roots, (X0 - T2_D / 2.0, X0 + T2_D / 2.0)):
            self.assertLessEqual(abs(got - want), 4.0 * EPS * abs(want) + 1e-15)
        self.assertHonest(data, T2_REF)


class TestNearlyCoalescingPair(unittest.TestCase, _Honest):
    def test_3_minimum_located_and_windowed(self):
        """
        Test 3. Section 2's family (C), theta = a (x - x0) + (b/3)(x - x0)^3, at lambda = a^(3/2) / b^(1/2) =
        Lambda_c / 4 = 1, at b = 1e4 and 1e8: the minimum of |theta'| at x0 is located, as kind "minimum" with
        lambda below Lambda_c, and windowed; the value is honest.
        """
        self.assertEqual(lq._LEVIN_COALESCENCE_LAMBDA, 4.0)
        for b, a, ref in T3_CASES:
            with self.subTest(b=b):
                theta = {
                    "theta": lambda x, a=a, b=b: a * (x - X0) + b / 3.0 * (x - X0) ** 3,
                    "theta_deriv": lambda x, a=a, b=b: a + b * (x - X0) ** 2,
                }
                data = adaptive_levin_sincos(
                    (0.0, 1.0), F_23, theta, atol=1e-14 * abs(ref), rtol=1e-9
                )
                minima = [
                    p for p in data["stationary_points"] if p["kind"] == "minimum"
                ]
                self.assertEqual(len(minima), 1)
                self.assertLess(minima[0]["lambda"], lq._LEVIN_COALESCENCE_LAMBDA)
                self.assertLessEqual(abs(minima[0]["x"] - X0), 1e-6)
                self.assertEqual(len(data["stationary_windows"]), 1)
                self.assertGreater(len(data["stationary_windows"][0]["cuts"]), 1)
                self.assertHonest(data, ref)


class TestTheGuard(unittest.TestCase, _Honest):
    def test_4_no_theta_deriv(self):
        """
        Test 4. Test 1's problem with theta alone: the basis supplies no rates, so nothing is located; the
        guard fires on the Levin leaves whose theta' changes sign on their nodes, and the value is honest.
        """
        f, theta = p_problem(P1_XS, with_deriv=False)
        data = adaptive_levin_sincos((0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL)
        self.assertEqual(data["stationary_points"], [])
        self.assertEqual(data["stationary_windows"], [])
        self.assertGreater(data["num_guarded_regions"], 0)
        self.assertFalse(any(r.type == "Levin" and r.guarded for r in data["regions"]))
        self.assertHonest(data, P1_REF)

    def test_6_the_cap(self):
        """
        Test 6. Test 4 with depth_max 4, so that a guarded leaf reaches it: it is re-solved by the nested
        Clenshaw-Curtis pair and flagged guard_at_depth_max; the call is not converged; the leaf is a
        Clenshaw-Curtis leaf whose abserr is that solve's own nested-pair estimate; and the value is honest.
        """
        f, theta = p_problem(P1_XS, with_deriv=False)
        data = adaptive_levin_sincos(
            (0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL, depth_max=4
        )
        capped = [r for r in data["regions"] if r.guard_at_depth_max]
        self.assertGreater(len(capped), 0)
        self.assertEqual(data["num_guard_at_depth_max"], len(capped))
        self.assertFalse(data["converged"])
        for region in capped:
            self.assertEqual(region.type, lq.types[lq.INTERVAL_TYPE_DIRECT])
            self.assertTrue(region.guarded)
            self.assertEqual(region.depth, 4)
            solve = _adaptive_levin_subregion(
                (region.start, region.end),
                f,
                _Basis_SinCos(theta),
                id_label=None,
                rtol=P_RTOL,
                force_direct=True,
            )
            self.assertEqual(region.abserr, solve["abserr_direct"])
            self.assertEqual(region.value_own, solve["value"])
        self.assertHonest(data, P1_REF)


class TestFloorLimitedExitBarred(unittest.TestCase, _Honest):
    def test_5_no_window_leaf_floor_limited_by_the_exit(self):
        """
        Test 5. P5: the parent (bd21bf23) marks the whole span, one Levin leaf across the stationary point at
        x_s = 10.0918, floor-limited at depth 0: value -1.5860631569162417e-07, abserr 9.012e-12, 1 region,
        converged False, true error 1.125e-3 = 1.25e8 times its abserr (out/run2/parent_cases.json). Under the
        guard no window leaf and no guarded leaf is floor-limited by the round-off exit -- here no window is
        stopped by section 1.3's rule either, so no window leaf is floor-limited at all -- and the value is
        honest. With the exit allowed on window leaves (breakage (iii)) two window leaves are floor-limited.
        """
        f, theta = p_problem(P5_XS)
        data = adaptive_levin_sincos((0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL)
        self.assertEqual(len(data["stationary_points"]), 1)
        self.assertFalse(any(w["stopped"] for w in data["stationary_windows"]))
        window_leaves = [r for r in data["regions"] if r.window]
        self.assertGreater(len(window_leaves), 0)
        self.assertFalse(any(r.phase_limited for r in window_leaves))
        self.assertFalse(any(r.phase_limited for r in data["regions"] if r.guarded))
        self.assertHonest(data, P5_REF)


class TestNothingElseMoves(unittest.TestCase):
    def test_8_bit_identical_without_a_stationary_point(self):
        """
        Test 8. Every problem of test_levin_quadrature.py that has no stationary point, with its closed-form
        theta_deriv: nothing is located and no leaf is guarded, and value, abserr and every returned leaf are
        bit-identical to the same call with the location bypassed (_test_only_bypass_location).
        """
        problems = [
            (
                "sin",
                (1.0, 50000.0),
                [lambda x: 1.0, lambda x: 0.0],
                1.0,
                0.0,
                dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            ),
            (
                "cos",
                (1.0, 500000.0),
                [lambda x: 0.0, lambda x: 1.0],
                1.0,
                0.0,
                dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            ),
            (
                "sinc",
                (1.0, 100.0),
                [lambda x: 1.0 / x, lambda x: 0.0],
                100.0,
                0.0,
                dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            ),
            (
                "lstsq",
                (0.0, 1.0),
                [lambda x: math.exp(-x), lambda x: 0.3],
                10.0 * math.pi,
                0.0,
                dict(atol=1e-15, rtol=1e-12),
            ),
            (
                "direct",
                (0.0, 1.0),
                [lambda x: math.exp(-x), lambda x: 0.3],
                30.0 * math.pi,
                0.0,
                dict(atol=1e-15, rtol=1e-12),
            ),
            (
                "bump",
                (0.0, 3.0),
                [lambda x: math.exp(-400.0 * (x - 1.1) ** 2), lambda x: 0.0],
                3.0e4,
                0.0,
                dict(atol=1e-10),
            ),
            (
                "omega",
                (1.0 / 3.0, 7.0 / 3.0),
                [lambda x: math.exp(-x), lambda x: 0.0],
                1.0e6,
                0.0,
                dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
            ),
        ]
        cases = [
            (
                name,
                span,
                f,
                {
                    "theta": lambda x, w=w: w * x,
                    "theta_deriv": lambda x, w=w: w + 0.0 * x,
                },
                kw,
            )
            for name, span, f, w, _, kw in problems
        ]
        for lbda in (10.0, 100.0, 1000.0):
            cases.append(
                (
                    f"grz{lbda:g}",
                    (-1.0, 1.0),
                    [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)],
                    {
                        "theta": lambda x, l=lbda: l * math.atan(x),
                        "theta_deriv": lambda x, l=lbda: l / (1.0 + x * x),
                    },
                    dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
                )
            )
        fields = (
            "start",
            "end",
            "depth",
            "type",
            "abserr",
            "relerr",
            "phase_err",
            "phase_limited",
            "value",
            "value_kind",
            "value_own",
            "trusted",
            "delta1",
            "delta2",
            "convergence_ratio",
            "declared_resolution",
            "total_err",
            "budget_err",
            "node_error_signed",
            "phase_err_less_node",
            "window",
            "guarded",
        )

        def record(region):
            out = []
            for name in fields:
                v = getattr(region, name)
                out.append(float(v).hex() if isinstance(v, float) else v)
            return out

        for name, span, f, theta, kw in cases:
            with self.subTest(problem=name):
                located = _adaptive_levin(span, f, _Basis_SinCos(theta), **kw)
                bypassed = _adaptive_levin(
                    span, f, _Basis_SinCos(theta), _test_only_bypass_location=True, **kw
                )
                self.assertEqual(located["stationary_points"], [])
                self.assertEqual(located["num_guarded_regions"], 0)
                self.assertEqual(bypassed["num_guarded_regions"], 0)
                self.assertEqual(
                    float(located["value"]).hex(), float(bypassed["value"]).hex()
                )
                self.assertEqual(
                    float(located["abserr"]).hex(), float(bypassed["abserr"]).hex()
                )
                self.assertEqual(
                    [record(r) for r in located["regions"]],
                    [record(r) for r in bypassed["regions"]],
                )


class _TwoPhaseQuadraticBasis(_TwoIndependentSinCosBasis):
    """
    Test 9's basis: test_levin_quadrature.py's _TwoIndependentSinCosBasis with its second phase quadratic,
    theta_2 = (c/2)(x - x_s)^2, the Clenshaw-Curtis methods the driver calls (eval_basis_and_arguments,
    theta_prime_on_fine_grid, basis_derivative) and section 1.0's two (independent_phases,
    independent_phase_rates). Components: (sin theta_1, cos theta_1, sin theta_2, cos theta_2).

    The fallback's node term takes one theta' per node; this basis hands it the larger |theta_i'| at each node,
    an approximation for a basis with two phases (the node term is of order u |x| theta' and decides
    nothing here).
    """

    def __init__(self, lambda1, c, x_s):
        super().__init__(lambda1, 0.0)
        self._c = c
        self._xs = x_s

    def _phases(self, x):
        return self._lambda1 * x, 0.5 * self._c * (x - self._xs) ** 2

    def _rates(self, x):
        return self._lambda1 + 0.0 * x, self._c * (x - self._xs)

    def independent_phases(self):
        return [
            lambda x: self._lambda1 * x,
            lambda x: 0.5 * self._c * (x - self._xs) ** 2,
        ]

    def independent_phase_rates(self):
        return [lambda x: self._lambda1 + 0.0 * x, lambda x: self._c * (x - self._xs)]

    def build_Levin_data(
        self,
        grid,
        Dmat,
        notify_label=None,
        id_label=None,
        need_AmatT=True,
        vectorize_cache=None,
    ):
        theta1, theta2 = self._phases(grid)
        t1, t2 = self._rates(grid)
        AmatT = None
        if need_AmatT:
            N = len(grid)
            zero = np.zeros((N, N))
            d1, d2 = np.diag(t1), np.diag(t2)
            AmatT = np.block(
                [
                    [zero, -d1, zero, zero],
                    [d1, zero, zero, zero],
                    [zero, zero, zero, -d2],
                    [zero, zero, d2, zero],
                ]
            )
        phase_span = float(
            (np.mean(np.fabs(t1)) + np.mean(np.fabs(t2))) * np.fabs(grid[0] - grid[-1])
        )
        w0 = [
            np.sin(theta1[-1]),
            np.cos(theta1[-1]),
            np.sin(theta2[-1]),
            np.cos(theta2[-1]),
        ]
        wk = [
            np.sin(theta1[0]),
            np.cos(theta1[0]),
            np.sin(theta2[0]),
            np.cos(theta2[0]),
        ]
        return AmatT, t1, w0, wk, phase_span

    def eval_basis(self, x):
        theta1, theta2 = self._phases(x)
        return [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)]

    def eval_basis_and_arguments(self, x):
        theta1, theta2 = self._phases(x)
        return (
            [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)],
            [theta1, theta1, theta2, theta2],
        )

    @staticmethod
    def basis_derivative(basis_values):
        return np.vstack(
            [basis_values[1], -basis_values[0], basis_values[3], -basis_values[2]]
        )

    def theta_prime_on_fine_grid(
        self, fine_grid, D_fine, theta_prime_coarse, arguments_fine, vectorize_cache
    ):
        t1, t2 = self._rates(fine_grid)
        return (
            np.maximum(np.fabs(t1), np.fabs(t2)),
            lq._CC_THETA_PRIME_REL_ERROR_SUPPLIED,
        )


class TestGenericPath(unittest.TestCase, _Honest):
    def test_9_two_phases(self):
        """
        Test 9. A basis of two independent phases, theta_1 = 200 x (linear) and theta_2 = (c/2)(x - x_s)^2 at the
        non-dyadic x_s = 0.3819660112501051, c = 1e5, f = (1, 0, 0, 1): I = int sin(theta_1) + cos(theta_2), an
        elementary integral and a Fresnel one. Driven through _adaptive_levin directly. The point is located on
        phase 2 (index 1) and windowed; phase 1 locates nothing; the value is honest.
        """
        basis = _TwoPhaseQuadraticBasis(T9_L1, T9_C, T9_XS)
        f = [
            lambda x: 1.0 + 0.0 * x,
            lambda x: 0.0 * x,
            lambda x: 0.0 * x,
            lambda x: 1.0 + 0.0 * x,
        ]
        data = _adaptive_levin(
            (0.0, 1.0), f, basis, atol=1e-14 * abs(T9_REF), rtol=1e-9
        )
        points = data["stationary_points"]
        self.assertEqual([p["phase"] for p in points], [1])
        self.assertEqual(points[0]["kind"], "root")
        self.assertLessEqual(abs(points[0]["x"] - T9_XS), 4.0 * EPS * T9_XS)
        self.assertEqual(len(data["stationary_windows"]), 1)
        self.assertGreater(data["num_window_pieces"], 0)
        self.assertFalse(
            any(p["phase"] == 0 for p in points + data["unlocated_minima"])
        )
        self.assertHonest(data, T9_REF)


class TestWindowStoppingRule(unittest.TestCase, _Honest):
    def test_10a_flat_window_stops(self):
        """
        Test 10 (a). Family (R), c = 1e6, x_s = 0.7236..., omega = 0, at rtol 1e-12 (a request of 1.4e-16
        against |I| = 1.4e-4, below the window leaves' Clenshaw-Curtis floor): the window's bisections stop
        lowering its error, six count, and the window stops far short of depth_max (10 here); its leaves are
        returned floor-limited; the call is unconverged and honest.
        """
        data = adaptive_levin_sincos(
            (0.0, 1.0),
            F_23,
            quadratic_theta(T10_C, XS_A),
            atol=1e-14 * abs(T10A_REF),
            rtol=1e-12,
            depth_max=10,
        )
        (window,) = data["stationary_windows"]
        self.assertTrue(window["stopped"])
        self.assertEqual(window["iroff1_count"], lq._LEVIN_WINDOW_IROFF1_LIMIT)
        window_leaves = [r for r in data["regions"] if r.window]
        self.assertGreater(len(window_leaves), 0)
        self.assertTrue(all(r.phase_limited for r in window_leaves))
        self.assertLess(max(r.depth for r in window_leaves), 10 - 4)
        self.assertFalse(data["converged"])
        self.assertHonest(data, T10A_REF)

    def test_10b_untrusted_leaves_at_noise_stop(self):
        """
        Test 10 (b). Family (R), c = 1e6, x_s = 0.3819..., omega = 0, rtol 1e-9, with f_cos quantised to
        multiples of 2^-40, a deterministic noise of relative size about 1e-12 in the integrand (z12's shape:
        its window leaves' nested-pair differences sit at the integrand's own noise). Without the rule 88
        window leaves are returned untrusted at depth_max 14, their summed error 1.7e-14 against a request of
        1.9e-12, and the call is not converged (the log). With it the window stops, the call is converged, and
        the value is honest against the closed form for the unquantised integrand with the quantisation's
        bound, |int (q(f_cos) - f_cos) cos theta| <= 2^-40, added to the true error.
        """
        q = 2.0**40
        f = [lambda x: np.exp(-x), lambda x: np.floor(np.cos(3.0 * x) * q) / q]
        data = adaptive_levin_sincos(
            (0.0, 1.0),
            f,
            quadratic_theta(T10_C, X0),
            atol=1e-14 * abs(T10B_REF),
            rtol=1e-9,
        )
        (window,) = data["stationary_windows"]
        self.assertTrue(window["stopped"])
        self.assertTrue(data["converged"])
        self.assertEqual(data["num_untrusted_regions"], 0)
        self.assertHonest(data, T10B_REF, margin=1.0 / q)

    def test_10c_does_not_fire_where_leaves_converge(self):
        """Test 10 (c). On test 1's window leaves no bisection counts: its window's iroff1 count is zero."""
        f, theta = p_problem(P1_XS)
        data = adaptive_levin_sincos((0.0, P_L), f, theta, atol=P_ATOL, rtol=P_RTOL)
        (window,) = data["stationary_windows"]
        self.assertEqual(window["iroff1_count"], 0)
        self.assertFalse(window["stopped"])


# Test 13 (section 10, Rule A): references by rule-a/scripts/test_references_rule_a.py, at 40 and 60 digits
# (agreeing to at most 1.1e-41 relative) and checked against mpmath.quad; its output is
# docs/levin-phase-floor/07/calibration/rule-a/out/test_references_rule_a.json.
# (a) theta = 0: I = int_0^1 cos 3x dx = sin(3) / 3.
T13A_REF = float.fromhex("0x1.815a092491c7ap-5")  # 4.70400026866224073669149342694e-2
# (b), (c) theta = (c/2)(x - x0)^2 on [0, 1] with section 2's amplitudes, V = (c/2)(x0^2 + (1 - x0)^2):
# c = 70, V = 18.475 = 0.980 * 6 pi (b); c = 72, V = 19.003 = 1.008 * 6 pi (c).
T13B_C, T13C_C = 70.0, 72.0
T13B_REF = float.fromhex("0x1.58f28f18e1695p-3")  # 1.68431394526284234165192922057e-1
T13C_REF = float.fromhex("0x1.4b17706bee4a3p-3")  # 1.61665800374781824059347863032e-1
T13B_V = 18.47524157501472  # 18.4752415750147221490017579642
# (d) test 9's basis with theta_1 = 200 x and theta_2 = 0, f = (e^{-x}, 0, 0, cos 3x):
# I = (l - e^{-1}(sin l + l cos l)) / (1 + l^2) + sin(3) / 3, l = 200.
T13D_L1 = 200.0
T13D_REF = float.fromhex("0x1.a309197cde8f6p-5")  # 5.11517999465968404313882463355e-2

_REGION_FIELDS_13 = (
    "start",
    "end",
    "depth",
    "type",
    "abserr",
    "relerr",
    "phase_err",
    "phase_limited",
    "value",
    "value_kind",
    "value_own",
    "trusted",
    "delta1",
    "delta2",
    "convergence_ratio",
    "declared_resolution",
    "total_err",
    "budget_err",
    "node_error_signed",
    "phase_err_less_node",
    "window",
    "guarded",
)


def _region_record_13(region):
    out = []
    for name in _REGION_FIELDS_13:
        v = getattr(region, name)
        out.append(float(v).hex() if isinstance(v, float) else v)
    return out


class TestRuleA(unittest.TestCase, _Honest):
    """
    Test 13 (prompt 07 section 10, Rule A): a phase whose total variation over the span, estimated on the
    location's scan by the trapezoidal rule, is below SIX_PI is not located; it is reported under
    "location_skipped_phases"; for a basis of several phases the guard tests it for a sign change.
    """

    KW = dict(rtol=1e-9)

    def assertBitIdenticalToBypassed(self, located, bypassed):
        self.assertEqual(float(located["value"]).hex(), float(bypassed["value"]).hex())
        self.assertEqual(
            float(located["abserr"]).hex(), float(bypassed["abserr"]).hex()
        )
        self.assertEqual(located["converged"], bypassed["converged"])
        self.assertEqual(
            [_region_record_13(r) for r in located["regions"]],
            [_region_record_13(r) for r in bypassed["regions"]],
        )

    def assertNothingLocated(self, data):
        self.assertEqual(data["stationary_points"], [])
        self.assertEqual(data["stationary_windows"], [])
        self.assertEqual(data["unlocated_minima"], [])
        self.assertEqual(data["num_window_pieces"], 0)
        self.assertFalse(any(r.window for r in data["regions"]))

    def test_13a_zero_rate(self):
        """
        Test 13 (a). _Basis_SinCos with theta = 0 and theta_deriv = 0 identically, and a smooth, varying f_cos =
        cos 3x (with f_sin = e^{-x}, which sin 0 annuls), the shape of R28's q - r calls with q = r. Nothing is
        located, no window piece is made, the phase is reported skipped with variation 0, nothing is guarded;
        value, abserr and every returned leaf are bit-identical to the call with the location bypassed (test 8's
        switch); and the value is honest against sin(3) / 3. Before Rule A every one of the 4,097 scan samples
        was a root (breakage (x)).
        """
        theta = {"theta": lambda x: 0.0 * x, "theta_deriv": lambda x: 0.0 * x}
        kw = dict(self.KW, atol=1e-14 * abs(T13A_REF))
        located = _adaptive_levin((0.0, 1.0), F_23, _Basis_SinCos(theta), **kw)
        bypassed = _adaptive_levin(
            (0.0, 1.0),
            F_23,
            _Basis_SinCos(theta),
            _test_only_bypass_location=True,
            **kw,
        )
        self.assertNothingLocated(located)
        self.assertEqual(
            located["location_skipped_phases"], [{"phase": 0, "variation": 0.0}]
        )
        self.assertEqual(bypassed["location_skipped_phases"], [])
        self.assertEqual(located["num_guarded_regions"], 0)
        self.assertBitIdenticalToBypassed(located, bypassed)
        self.assertTrue(located["converged"])
        self.assertHonest(located, T13A_REF)

    def test_13b_small_phase_with_a_root(self):
        """
        Test 13 (b). theta = (c/2)(x - x0)^2 on [0, 1] at the non-dyadic x0 = 0.3819660112501051, c = 70, so that
        its total variation V = (c/2)(x0^2 + (1 - x0)^2) = 18.475 is below 6 pi = 18.850. Its root is not
        located; the phase is reported skipped, its variation within the estimate's stated error of V (the
        trapezoidal rule on the scan: at most h^2 |theta''| / 4 at the root, where |theta'| has a kink, theta' being
        linear elsewhere, plus the sum's rounding; h = 1/4096). The root leaf, whose node estimate of its span
        exceeds 6 pi, is caught by the nodal guard. The result is bit-identical to the call with the location
        bypassed, and the value is honest against the closed form in erf.
        """
        theta = quadratic_theta(T13B_C, X0)
        kw = dict(self.KW, atol=1e-14 * abs(T13B_REF))
        located = _adaptive_levin((0.0, 1.0), F_23, _Basis_SinCos(theta), **kw)
        bypassed = _adaptive_levin(
            (0.0, 1.0),
            F_23,
            _Basis_SinCos(theta),
            _test_only_bypass_location=True,
            **kw,
        )
        self.assertNothingLocated(located)
        (skipped,) = located["location_skipped_phases"]
        self.assertEqual(skipped["phase"], 0)
        self.assertLess(skipped["variation"], lq.SIX_PI)
        h = 1.0 / (lq._LEVIN_STATIONARY_SCAN_POINTS - 1)
        bound = T13B_C * h * h / 4.0 + lq._LEVIN_STATIONARY_SCAN_POINTS * EPS * T13B_V
        self.assertLessEqual(abs(skipped["variation"] - T13B_V), bound)
        # The root leaf's own span estimate, from its 16 nodes, is 23.69 > 6 pi, so the C2 gate sends it to the
        # Levin solve, where the nodal guard catches the root Rule A left unlocated (section 10, "Why the gate's
        # threshold"): it is guarded and bisected, and its halves go to the Clenshaw-Curtis rule. No Levin leaf
        # is returned guarded, and the guard acts as it does with the location bypassed.
        self.assertEqual(located["num_guarded_regions"], 1)
        self.assertEqual(bypassed["num_guarded_regions"], 1)
        self.assertFalse(any(r.guarded for r in located["regions"]))
        self.assertBitIdenticalToBypassed(located, bypassed)
        self.assertHonest(located, T13B_REF)

    def test_13c_same_phase_just_above_six_pi(self):
        """
        Test 13 (c). Test 13 (b)'s phase at c = 72, V = 19.003, just above 6 pi: no phase is skipped, and the root
        is located at x0 (within 4 eps x0) and windowed, as before Rule A; the value is honest.
        """
        theta = quadratic_theta(T13C_C, X0)
        data = adaptive_levin_sincos(
            (0.0, 1.0), F_23, theta, atol=1e-14 * abs(T13C_REF), **self.KW
        )
        self.assertEqual(data["location_skipped_phases"], [])
        (point,) = data["stationary_points"]
        self.assertEqual(point["kind"], "root")
        self.assertLessEqual(abs(point["x"] - X0), 4.0 * EPS * X0)
        self.assertEqual(len(data["stationary_windows"]), 1)
        self.assertGreater(data["num_window_pieces"], 0)
        self.assertEqual(data["num_guarded_regions"], 0)
        self.assertHonest(data, T13C_REF)

    def test_13d_several_phases(self):
        """
        Test 13 (d). Test 9's two-phase basis with phase 2 identically zero (c = 0) and phase 1 linear, theta_1 =
        200 x, V_1 = 200 >= 6 pi; f = (e^{-x}, 0, 0, cos 3x). Phase 2 (index 1) is reported skipped with variation 0,
        phase 1 locates nothing; no leaf is guarded -- the guard tests the skipped phase for a sign change, and
        its rate, 0 or -0 on every node, has none (tested for one strict sign, breakage (xi), every Levin leaf is
        guarded); the call converges and is honest against the closed form.
        """
        basis = _TwoPhaseQuadraticBasis(T13D_L1, 0.0, X0)
        f = [
            lambda x: np.exp(-x),
            lambda x: 0.0 * x,
            lambda x: 0.0 * x,
            lambda x: np.cos(3.0 * x),
        ]
        data = _adaptive_levin(
            (0.0, 1.0), f, basis, atol=1e-14 * abs(T13D_REF), **self.KW
        )
        self.assertEqual(
            data["location_skipped_phases"], [{"phase": 1, "variation": 0.0}]
        )
        self.assertNothingLocated(data)
        self.assertEqual(data["num_guarded_regions"], 0)
        self.assertFalse(any(r.guarded for r in data["regions"]))
        self.assertTrue(data["converged"])
        self.assertHonest(data, T13D_REF)


if __name__ == "__main__":
    unittest.main()
