"""
The stationary-point guard (prompts/levin-phase-floor/07-stationary-point-guard.md, section 3).

_adaptive_levin() locates every sign change of each closed-form phase rate a basis supplies, and every
interior minimum of |theta'| whose Airy parameter is below _LEVIN_COALESCENCE_LAMBDA, before its loop; it
integrates a window of phase half-width _LEVIN_WINDOW_HALF_PHASE round each by the nested Clenshaw-Curtis
pair, in pieces of at most 2 pi; it stops a window whose bisections no longer lower its error by QUADPACK's
iroff1 counted per window (section 1.3); and it never returns a Levin leaf whose phase rate changes sign on its
own nodes (the guard). A phase whose rate is exactly zero at every finite scan sample has nothing to locate and is not
located, and the guard tests such a phase of a basis of several phases for a sign change (prompts/levin-phase-floor/
07a-window-stop-and-phase-variation.md section 1.1, in place of 07's section 10 Rule A; test 13). The window's count
is QUADPACK dqage's, with a value test no looser than the request (07a section 1.2; test 14).

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
from unittest import mock

import numpy as np

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.levin_quadrature import (
    _Basis_SinCos,
    _adaptive_levin,
    _adaptive_levin_subregion,
    adaptive_levin_sincos,
    chebyshev_matrices,
)
from AdaptiveLevin.tests.test_levin_quadrature import (
    _TwoIndependentSinCosBasis,
    _phase_variation,
)

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
        # V summed over the two phases, each by _Basis_SinCos's rule (prompt 07a section 1.1 item
        # 3; until 07a, (mean|theta1'| + mean|theta2'|) * width)
        phase_span = _phase_variation(theta1, t1) + _phase_variation(theta2, t2)
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
        # 07a section 9.4 test 14 (c): the window still stops with its count at the limit under dqage's count
        # (10 (a) asserts the same, 10 (c) a count of zero)
        self.assertEqual(window["iroff1_count"], lq._LEVIN_WINDOW_IROFF1_LIMIT)
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


# Test 13 (07 section 10's Rule A, rewritten for 07a section 1.1, 07a section 3 test 6): references by 07's
# rule-a/scripts/test_references_rule_a.py, at 40 and 60 digits (agreeing to at most 1.1e-41 relative) and checked
# against mpmath.quad; its output is docs/levin-phase-floor/07/calibration/rule-a/out/test_references_rule_a.json.
# (a) theta = 0: I = int_0^1 cos 3x dx = sin(3) / 3.
T13A_REF = float.fromhex("0x1.815a092491c7ap-5")  # 4.70400026866224073669149342694e-2
# (b), (c) theta = (c/2)(x - x0)^2 on [0, 1] with section 2's amplitudes, V = (c/2)(x0^2 + (1 - x0)^2):
# c = 70, V = 18.475 = 0.980 * 6 pi (b); c = 72, V = 19.003 = 1.008 * 6 pi (c).
T13B_C, T13C_C = 70.0, 72.0
T13B_REF = float.fromhex("0x1.58f28f18e1695p-3")  # 1.68431394526284234165192922057e-1
T13C_REF = float.fromhex("0x1.4b17706bee4a3p-3")  # 1.61665800374781824059347863032e-1
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
    Test 13, rewritten by prompt 07a (section 3 test 6) for the location's first step that replaces 07's Rule A
    (07a section 1.1 item 1): a phase whose rate is exactly zero at every finite scan sample is not located; it
    is reported under "location_skipped_phases" with the reason, "zero_rate"; for a basis of several phases the
    guard tests it for a sign change. Every other phase is located whatever its variation: (b) is 07a's test 5,
    O9's case, whose phase Rule A skipped. The class keeps its name, as the tests' records cite it.
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
        located, no window piece is made, the phase is reported skipped with the reason "zero_rate" (07 reported
        it with variation 0, 07a's rule in place of Rule A), nothing is guarded; value, abserr and every returned
        leaf are bit-identical to the call with the location bypassed (test 8's switch), as at the parent; and the
        value is honest against sin(3) / 3. Before Rule A every one of the 4,097 scan samples was a root (07's
        breakage (x)).
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
            located["location_skipped_phases"], [{"phase": 0, "reason": "zero_rate"}]
        )
        self.assertEqual(bypassed["location_skipped_phases"], [])
        self.assertEqual(located["num_guarded_regions"], 0)
        self.assertBitIdenticalToBypassed(located, bypassed)
        self.assertTrue(located["converged"])
        self.assertHonest(located, T13A_REF)

    def test_13b_small_phase_with_a_root(self):
        """
        Test 13 (b), 07a's section 3 test 5, O9's case. theta = (c/2)(x - x0)^2 = 35 (x - x0)^2 on [0, 1] at the
        non-dyadic x0 = 0.3819660112501051, c = 70: its total variation V = (c/2)(x0^2 + (1 - x0)^2) = 18.475 is
        below 6 pi = 18.850, while the root leaf's estimate at the parent, mean|theta'| on its 16 nodes times its
        width, read 23.69. At the parent Rule A skipped the phase (its scan estimate 18.475242613612156), the gate
        sent the root leaf to the Levin solve on its estimate, and the nodal guard caught the unlocated root: one
        guarded leaf, value 0x1.58f28f18e1696p-3, abserr 1.962e-13, 5 regions, honest. Under 07a the root is
        located at x0 (within 4 eps x0) and windowed, nothing is skipped, no leaf is guarded, and the value is
        honest against the closed form in erf.
        """
        theta = quadratic_theta(T13B_C, X0)
        kw = dict(self.KW, atol=1e-14 * abs(T13B_REF))
        data = _adaptive_levin((0.0, 1.0), F_23, _Basis_SinCos(theta), **kw)
        self.assertEqual(data["location_skipped_phases"], [])
        (point,) = data["stationary_points"]
        self.assertEqual(point["kind"], "root")
        self.assertLessEqual(abs(point["x"] - X0), 4.0 * EPS * X0)
        self.assertEqual(len(data["stationary_windows"]), 1)
        self.assertGreater(data["num_window_pieces"], 0)
        self.assertEqual(data["num_guarded_regions"], 0)
        self.assertFalse(any(r.guarded for r in data["regions"]))
        self.assertTrue(data["converged"])
        self.assertHonest(data, T13B_REF)

    def test_13c_same_phase_just_above_six_pi(self):
        """
        Test 13 (c). Test 13 (b)'s phase at c = 72, V = 19.003, just above 6 pi: no phase is skipped, and the root
        is located at x0 (within 4 eps x0) and windowed, as before Rule A; the value is honest. Unchanged by 07a.
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
        200 x, V_1 = 200 >= 6 pi; f = (e^{-x}, 0, 0, cos 3x). Phase 2 (index 1), its rate 0 or -0 at every scan
        sample, is reported skipped with the reason "zero_rate" (07: with variation 0); phase 1 locates nothing; no
        leaf is guarded -- the guard tests the skipped phase for a sign change, and its rate, 0 or -0 on every node,
        has none (tested for one strict sign, 07's breakage (xi), every Levin leaf is guarded); the call converges
        and is honest against the closed form.
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
            data["location_skipped_phases"], [{"phase": 1, "reason": "zero_rate"}]
        )
        self.assertNothingLocated(data)
        self.assertEqual(data["num_guarded_regions"], 0)
        self.assertFalse(any(r.guarded for r in data["regions"]))
        self.assertTrue(data["converged"])
        self.assertHonest(data, T13D_REF)


# Test 14 (prompt 07a section 9.4, the window's count after section 1.2). (d)'s reference is the closed form in
# Fresnel integrals, by docs/levin-phase-floor/07a/code/test_references_07a.py at 40 and 60 digits (agreeing to
# 7.2e-39 relative; out/test_references_07a.json).
T14D_THETA0, T14D_C = 1.0e8, 1.0e6
T14D_REF = float.fromhex("0x1.08956905b4489p-10")  # 1.00930646424427149117727776641e-3


def _interval(abserr, floor, declared=None):
    """A Clenshaw-Curtis region with the given nested-pair estimate and floor (less its node term), and, if
    declared is given, the trust check's widened resolution error, so that its e' (budget_err) is the larger of
    declared and floor while its own error e (own_err) is the larger of abserr and floor.
    """
    region = lq.used_interval(
        start=0.0,
        end=1.0,
        depth=0,
        type=lq.INTERVAL_TYPE_DIRECT,
        abserr=abserr,
        phase_err=floor,
        phase_err_less_node=floor,
    )
    region._declared_resolution = declared
    return region


class TestWindowCount(unittest.TestCase, _Honest):
    """
    Test 14 (prompt 07a section 9.4): the window's iroff1 count is dqage's, with a value test no looser than the
    request. lq._window_bisection_counts() decides one bisection from V_P, V_L + V_R, the leaves' own errors e
    (lq._own_error(), used_interval.own_err) and the halves' unresolved flags (lq._cc_leaf_unresolved()).
    """

    def test_14a_the_count_directly(self):
        """
        Test 14 (a). On constructed inputs:
          * at rtol 1e-8 a value change of 1e-6 relative, between rtol |V| and 1e-5 |V|, is not counted, where
            07's fixed 1e-5 counts it (the same bisection at an rtol above 1e-5, where the cap 1e-5 applies);
          * either half unresolved: not counted;
          * raw estimates flat while e' falls (a parent whose e' the trust check widened): counted;
          * raw estimates falling while e' is flat (halves whose e' the trust check widened): not counted;
          * each inequality's boundary, inclusive: the value test at rtol = 2^-20 and the error test at 0.99.
        """
        count = lq._window_bisection_counts
        flat = dict(error_parent=1.0, error_left=0.5, error_right=0.5)
        resolved = dict(unresolved_left=False, unresolved_right=False)

        v_parent = 1.0 - 1.0e-6
        self.assertLess(1.0e-8 * 1.0, abs(1.0 - v_parent))
        self.assertLessEqual(abs(1.0 - v_parent), 1.0e-5 * 1.0)
        self.assertFalse(count(v_parent, 1.0, **flat, **resolved, rtol=1.0e-8))
        self.assertTrue(count(v_parent, 1.0, **flat, **resolved, rtol=1.0e-3))
        self.assertFalse(count(1.0 - 2.0e-5, 1.0, **flat, **resolved, rtol=1.0e-3))

        for left, right in ((True, False), (False, True), (True, True)):
            with self.subTest(unresolved=(left, right)):
                self.assertFalse(
                    count(
                        1.0,
                        1.0,
                        **flat,
                        unresolved_left=left,
                        unresolved_right=right,
                        rtol=1.0e-8,
                    )
                )
        self.assertTrue(count(1.0, 1.0, **flat, **resolved, rtol=1.0e-8))

        # raw flat, e' falling: the parent's e' widened to 1e-8 by the trust check, its own e 1e-10
        parent = _interval(1.0e-10, 1.0e-12, declared=1.0e-8)
        left, right = _interval(0.5e-10, 1.0e-12), _interval(0.5e-10, 1.0e-12)
        self.assertLess(
            left.budget_err + right.budget_err,
            lq._LEVIN_WINDOW_IROFF1_ERROR_RATIO * parent.budget_err,
        )
        self.assertTrue(
            count(
                1.0,
                1.0,
                lq._own_error(parent),
                lq._own_error(left),
                lq._own_error(right),
                **resolved,
                rtol=1.0e-8,
            )
        )

        # raw falling, e' flat: the halves' e' widened to 0.5e-8 each, their own e 1e-11
        parent = _interval(1.0e-8, 1.0e-12)
        left = _interval(1.0e-11, 1.0e-12, declared=0.5e-8)
        right = _interval(1.0e-11, 1.0e-12, declared=0.5e-8)
        self.assertGreaterEqual(
            left.budget_err + right.budget_err,
            lq._LEVIN_WINDOW_IROFF1_ERROR_RATIO * parent.budget_err,
        )
        self.assertFalse(
            count(
                1.0,
                1.0,
                lq._own_error(parent),
                lq._own_error(left),
                lq._own_error(right),
                **resolved,
                rtol=1.0e-8,
            )
        )

        # the boundaries: |dV| = 2^-20 |V_L + V_R| at rtol 2^-20, and e_L + e_R = 0.99 e_P, both counted
        tol = 2.0**-20
        self.assertTrue(count(1.0 - tol, 1.0, **flat, **resolved, rtol=tol))
        self.assertFalse(
            count(np.nextafter(1.0 - tol, 0.0), 1.0, **flat, **resolved, rtol=tol)
        )
        ratio = lq._LEVIN_WINDOW_IROFF1_ERROR_RATIO
        self.assertTrue(count(1.0, 1.0, 1.0, ratio * 1.0, 0.0, **resolved, rtol=1e-8))
        self.assertFalse(
            count(
                1.0,
                1.0,
                1.0,
                np.nextafter(ratio * 1.0, 0.0),
                0.0,
                **resolved,
                rtol=1e-8,
            )
        )

    def test_14b_unresolved(self):
        """
        Test 14 (b). A Clenshaw-Curtis leaf of a smooth integrand in its asymptotic regime (F_23, theta = 10x, on
        [0, 1]: |CC_{2N-1} - CC_N| = 4.1e-9, positive, and 200 times it 1.3e-6 of resasc) is resolved; a leaf whose integrand is a peak narrower than the node spacing, centred on an
        odd node of the fine rule (so the coarse rule misses it), is unresolved -- its 200 |CC_{2N-1} - CC_N| is
        105 times its resasc, while |CC_{2N-1} - CC_N| alone is 0.53 of it. Each leaf's resasc
        equals, to rounding, sum_j |w_j| |g_j - gbar| evaluated here. And the test itself is inclusive at the
        factor 200, never true with either number zero or missing.
        """
        order = lq.DEFAULT_LEVIN_CHEBSHEV_ORDER
        fine_order = 2 * order - 1
        peak_span = (0.0, 1.0)
        fine_grid, _ = chebyshev_matrices(peak_span, fine_order)
        x_peak = float(fine_grid[13])
        width = 1.0e-3
        cases = (
            (
                "smooth",
                (0.0, 1.0),
                F_23,
                {"theta": lambda x: 10.0 * x, "theta_deriv": lambda x: 10.0 + 0.0 * x},
                False,
            ),
            (
                "peak",
                peak_span,
                [
                    lambda x: 0.0 * x,
                    lambda x: np.exp(-(((x - x_peak) / width) ** 2)),
                ],
                {"theta": lambda x: 0.0 * x, "theta_deriv": lambda x: 0.0 * x},
                True,
            ),
        )
        for name, span, f, theta, unresolved in cases:
            with self.subTest(leaf=name):
                data = _adaptive_levin_subregion(
                    span, f, _Basis_SinCos(theta), id_label=None, force_direct=True
                )
                self.assertTrue(data["is_direct"])
                self.assertEqual(
                    lq._cc_leaf_unresolved(data["abserr_direct"], data["resasc"]),
                    unresolved,
                )
                grid, _ = chebyshev_matrices(span, fine_order)
                weights = lq._cc_weights(span, fine_order)
                th = theta["theta"](grid)
                g = f[0](grid) * np.sin(th) + f[1](grid) * np.cos(th)
                value = math.fsum(weights * g)
                gbar = value / math.fsum(weights)
                resasc = math.fsum(np.fabs(weights) * np.fabs(g - gbar))
                self.assertAlmostEqual(
                    data["resasc"], resasc, delta=1e-13 * abs(resasc)
                )
                if unresolved:
                    self.assertLess(data["abserr_direct"], data["resasc"])
                else:
                    self.assertGreater(data["abserr_direct"], 0.0)
                    self.assertLess(
                        lq._DQK21_SATURATION_FACTOR * data["abserr_direct"],
                        1e-3 * data["resasc"],
                    )

        self.assertEqual(lq._DQK21_SATURATION_FACTOR, 200.0)
        self.assertTrue(lq._cc_leaf_unresolved(1.0, 200.0))
        self.assertFalse(lq._cc_leaf_unresolved(1.0, np.nextafter(200.0, np.inf)))
        self.assertFalse(lq._cc_leaf_unresolved(0.0, 1.0))
        self.assertFalse(lq._cc_leaf_unresolved(1.0, 0.0))
        self.assertFalse(lq._cc_leaf_unresolved(1.0, None))
        self.assertFalse(lq._cc_leaf_unresolved(None, 1.0))

    def test_14d_value_test_against_the_request(self):
        """
        Test 14 (d). A synthetic window at rtol 1e-9: theta = theta_0 + (c/2)(x - x_s)^2 on [0, 1], theta_0 = 1e8,
        c = 1e6, x_s = 0.3819660112501051, f = (1, 0), depth_max 12. The raw phase near 1e8 puts the window
        leaves' Clenshaw-Curtis floor above the request, so their own errors stop falling (e_L + e_R = e_P), and
        their first bisections change the value by 2.8e-9 and 3.8e-9 relative: between rtol and 1e-5.
          * 07's code (the parent, a9f6abc3) counts those bisections and stops the window after the call's 18th
            bisection: 28 regions, value 0x1.0895691a63805p-10, abserr 1.977e-10, unconverged, honest at 0.0238
            of its abserr.
          * The commit does not count them: the window stops after the 22nd bisection, 32 regions, unconverged,
            honest at 0.0126 of its abserr. Asserted here: the stop, its split and its count; at least one
            bisection with a value change in (rtol, 1e-5], both halves resolved and the raw error flat, not
            counted; and the honesty.
        Section 3 asks for a case that leaves the call converged and dishonest at the parent. None was built:
        on every case tried (this family at theta_0 1e6 and 1e8, rtol 1e-8 and 1e-9, f = (1, 0) and (0, 1)) both
        codes are honest, as Part A found on production calls -- the stopped windows' declarations at or above
        their true error, the five dishonest calls' excess outside the window (section 9.2). Test 14 (a) asserts
        the rule directly.
        """
        theta = {
            "theta": lambda x: T14D_THETA0 + 0.5 * T14D_C * (x - X0) * (x - X0),
            "theta_deriv": lambda x: T14D_C * (x - X0),
        }
        f = [lambda x: 1.0 + 0.0 * x, lambda x: 0.0 * x]
        rtol = 1.0e-9
        decisions = []
        count = lq._window_bisection_counts

        def recording(
            v_parent, v_halves, e_parent, e_left, e_right, u_left, u_right, r
        ):
            counted = count(
                v_parent, v_halves, e_parent, e_left, e_right, u_left, u_right, r
            )
            decisions.append(
                (
                    abs(v_halves - v_parent) / abs(v_halves),
                    e_left + e_right >= lq._LEVIN_WINDOW_IROFF1_ERROR_RATIO * e_parent,
                    u_left or u_right,
                    counted,
                )
            )
            return counted

        with mock.patch.object(lq, "_window_bisection_counts", recording):
            data = adaptive_levin_sincos(
                (0.0, 1.0),
                f,
                theta,
                atol=1e-14 * abs(T14D_REF),
                rtol=rtol,
                depth_max=12,
            )
        (window,) = data["stationary_windows"]
        self.assertTrue(window["stopped"])
        self.assertEqual(window["iroff1_count"], lq._LEVIN_WINDOW_IROFF1_LIMIT)
        self.assertEqual(window["stopped_after_split"], 22)
        self.assertEqual(data["num_regions"], 32)
        self.assertFalse(data["converged"])
        between = [
            d
            for d in decisions
            if rtol < d[0] <= lq._LEVIN_WINDOW_IROFF1_VALUE_RTOL and d[1] and not d[2]
        ]
        self.assertGreater(len(between), 0)
        self.assertFalse(any(counted for *_, counted in between))
        self.assertHonest(data, T14D_REF)


if __name__ == "__main__":
    unittest.main()
