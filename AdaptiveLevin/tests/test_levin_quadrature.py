import contextlib
import io
import unittest
from math import fabs, atan, sin, exp, pi, nan

import numpy as np

from AdaptiveLevin.levin_quadrature import (
    adaptive_levin_sincos,
    _Basis_SinCos,
    chebyshev_matrices,
    _cc_weights,
)
from utilities import format_time


class TestAdaptiveLevinSinCos(unittest.TestCase):

    def test_SinIntegral(self):
        x_span = (1.0, 50000.0)

        f = [lambda x: 1.0, lambda x: 0.0]

        theta = lambda x: x

        data = adaptive_levin_sincos(
            x_span,
            f,
            theta={"theta": theta},
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=12,
        )
        value = data["value"]
        regions = data["regions"]
        evaluations = data["evaluations"]
        elapsed = data["elapsed"]
        print(
            f"integral_(1)^(50,000) sin(x) = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: {region}")
        self.assertTrue(
            fabs(value - 0.5581795618) < 1e-10,
            f"Sin integral test failed: expected {0.5581795618}, obtained {value}",
        )

    def test_CosIntegral(self):
        x_span = (1.0, 500000.0)

        f = [lambda x: 0.0, lambda x: 1.0]

        theta = lambda x: x

        data = adaptive_levin_sincos(
            x_span,
            f,
            theta={"theta": theta},
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=12,
        )
        value = data["value"]
        regions = data["regions"]
        evaluations = data["evaluations"]
        elapsed = data["elapsed"]
        print(
            f"integral_(1)^(500,000) cos(x) = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: {region}")
        self.assertTrue(
            fabs(value - (-0.6636397833)) < 1e-10,
            f"Cos integral test failed: expected {-0.6636397833}, obtained {value}",
        )

    def test_SincIntegral(self):
        x_span = (1.0, 100.0)

        # sinc integral is sin(x)/x, so sin weight is 1/x and cos weight is 0
        f = [lambda x: 1.0 / x, lambda x: 0.0]

        # take theta to be 100x
        theta = lambda x: 100.0 * x

        data = adaptive_levin_sincos(
            x_span,
            f,
            theta={"theta": theta},
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=12,
        )
        value = data["value"]
        regions = data["regions"]
        evaluations = data["evaluations"]
        elapsed = data["elapsed"]
        print(
            f"integral_1^100 sin(100x)/x = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: {region}")
        self.assertTrue(
            fabs(value - 0.00866607847) < 1e-10,
            f"Sinc integral test failed: expected {0.00866607847}, obtained {value}",
        )

    def _GRZIntegral(self, lbda):

        x_span = (-1.0, 1.0)

        # integral is cos(lambda arctan(x)) / (1 + x^2)
        f = [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)]
        theta = lambda x: lbda * atan(x)

        data = adaptive_levin_sincos(
            x_span,
            f,
            theta={"theta": theta},
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=12,
        )
        value = data["value"]
        regions = data["regions"]
        evaluations = data["evaluations"]
        elapsed = data["elapsed"]
        print(
            f"integral_(-1)^(+1) cos({lbda} arctan(x)) / (1 + x^2) = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: {region}")

        analytic_sol = (2.0 / lbda) * sin(pi * lbda / 4.0)
        self.assertTrue(
            fabs(value - analytic_sol) < 1e-10,
            f"Gradshtein & Ryzhik integral test: expected {analytic_sol}, obtained {value}",
        )

    def test_GRZIntegral(self):
        self._GRZIntegral(10.0)
        self._GRZIntegral(100.0)
        self._GRZIntegral(1000.0)

    def test_nan_amplitude_raises(self):
        # amplitude goes non-finite partway through the region; C1: previously this made the
        # region contribute exactly 0.0 with a reported error of exactly 0.0 instead of raising.
        def f0(x):
            return nan if x > 1.9 else 1.0

        f = [f0, lambda x: 0.0]
        theta = lambda x: 1.0e5 * x

        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), f, theta={"theta": theta})
        self.assertIn("amplitude", str(ctx.exception))

    def test_all_nan_amplitude_raises(self):
        # C1: an entirely non-finite integrand previously returned value=0.0, abserr=0.0 -- the
        # most dangerous possible response to a completely broken integrand.
        f = [lambda x: nan, lambda x: nan]
        theta = lambda x: 1.0e5 * x

        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), f, theta={"theta": theta})
        self.assertIn("amplitude", str(ctx.exception))

    def test_input_validation(self):
        theta = lambda x: x
        good_f = [lambda x: 1.0, lambda x: 0.0]

        # len(f) != 2
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), [lambda x: 1.0], theta={"theta": theta})
        self.assertIn("f=", str(ctx.exception))

        # f = []
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), [], theta={"theta": theta})
        self.assertIn("f=", str(ctx.exception))

        # len(x_span) != 2
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0, 3.0), good_f, theta={"theta": theta})
        self.assertIn("x_span", str(ctx.exception))

        # atol <= 0
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), good_f, theta={"theta": theta}, atol=0.0)
        self.assertIn("atol", str(ctx.exception))

        # rtol < 0
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), good_f, theta={"theta": theta}, rtol=-1.0)
        self.assertIn("rtol", str(ctx.exception))

        # depth_max < 0
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos(
                (1.0, 2.0), good_f, theta={"theta": theta}, depth_max=-1
            )
        self.assertIn("depth_max", str(ctx.exception))

        # theta as a bare callable, rather than {"theta": callable} -- an easy mistake, since the
        # parameter is *named* theta
        with self.assertRaises(TypeError) as ctx:
            adaptive_levin_sincos((1.0, 2.0), good_f, theta=theta)
        self.assertIn("theta", str(ctx.exception))

        # NaN endpoint
        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((nan, 2.0), good_f, theta={"theta": theta})
        self.assertIn("x_span", str(ctx.exception))

        # chebyshev_order < 8: warned about and clamped, not rejected
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            data = adaptive_levin_sincos(
                (1.0, 50.0), good_f, theta={"theta": theta}, chebyshev_order=4
            )
        self.assertIn("clamp", buf.getvalue().lower())
        self.assertIsNotNone(data["value"])

    def test_atol_zero_rejected(self):
        theta = lambda x: x
        f = [lambda x: 1.0, lambda x: 0.0]

        with self.assertRaises(ValueError) as ctx:
            adaptive_levin_sincos((1.0, 50.0), f, theta={"theta": theta}, atol=0.0)
        self.assertIn("atol", str(ctx.exception))

    def test_sincos_basis_reports_complex_support(self):
        # prompt 02: the (sin, cos) basis is solved via the complexified N x N system rather than
        # the realified 2N x 2N one, gated on this capability rather than on isinstance(). Lock in
        # the contract so a future refactor that silently drops it is caught here rather than only
        # by a performance regression nobody notices.
        basis = _Basis_SinCos({"theta": lambda x: x})
        self.assertTrue(basis.supports_complexified_solve)

    def test_lstsq_and_direct_solve_paths_converge(self):
        # prompt 02: the complexified system keeps the same phase-span gate between the direct LU
        # solve and the lstsq fallback (_LEVIN_DIRECT_SOLVE_PHASE_SPAN = 20*pi). Exercise both
        # branches of the complexified solve explicitly, rather than relying on incidental coverage
        # from the other tests. See prompt 02's log for a standalone before/after comparison of the
        # lstsq branch against the pre-complexification code on this same problem.
        x_span = (0.0, 1.0)
        f = [lambda x: exp(-x), lambda x: 0.3]
        theta_ill = (
            lambda x: 10.0 * pi * x
        )  # phase span 10*pi < 20*pi gate: forces lstsq
        theta_direct = (
            lambda x: 30.0 * pi * x
        )  # phase span 30*pi > 20*pi gate: forces direct LU

        data_ill = adaptive_levin_sincos(
            x_span, f, theta={"theta": theta_ill}, atol=1e-15, rtol=1e-12
        )
        data_direct = adaptive_levin_sincos(
            x_span, f, theta={"theta": theta_direct}, atol=1e-15, rtol=1e-12
        )
        self.assertTrue(data_ill["converged"])
        self.assertTrue(data_direct["converged"])

    def test_converged_flag(self):
        theta = lambda x: x
        f = [lambda x: 1.0, lambda x: 0.0]

        data = adaptive_levin_sincos(
            (1.0, 50000.0),
            f,
            theta={"theta": theta},
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=12,
        )
        self.assertTrue(data["converged"])

        # C3: reproduced at HEAD as reporting abserr=1.26e-10 at atol=1e-10, exceeding what was
        # requested, with nothing previously indicating it.
        def gaussian_bump(x):
            return exp(-400.0 * (x - 1.1) ** 2)

        f_bump = [gaussian_bump, lambda x: 0.0]
        theta_bump = lambda x: 3.0e4 * x

        data_bump = adaptive_levin_sincos(
            (0.0, 3.0), f_bump, theta={"theta": theta_bump}, atol=1e-10
        )
        self.assertFalse(data_bump["converged"])

    def test_chebyshev_nesting(self):
        # prompt 03: the N-point extremal Chebyshev grid must be exactly every other node of the
        # (2*N - 1)-point grid, for every N -- this is what lets the nested Clenshaw-Curtis fallback
        # reuse one set of integrand samples for both rules. The audit's claim that nesting needs
        # N - 1 even is wrong; verify across both even and odd N, and non-trivial spans.
        for N in (8, 12, 13, 16, 17, 25, 33):
            for span in [(0.0, 1.0), (-2.0, 5.0), (1.0, 1.0e5)]:
                xN, _ = chebyshev_matrices(span, N)
                xfine, _ = chebyshev_matrices(span, 2 * N - 1)
                self.assertEqual(
                    0.0,
                    float(np.max(np.fabs(xfine[::2] - xN))),
                    f"nesting failed at N={N}, span={span}",
                )

    def test_cc_weights_exact_for_polynomials(self):
        # prompt 03: Clenshaw-Curtis weights of order N must integrate every polynomial of degree
        # < N exactly. This is the classic place to get the construction subtly wrong (endpoint
        # weights), and it would not be caught by any existing accuracy test.
        a, b = -0.3, 1.7
        for N in (8, 12, 13, 16, 25):
            x, _ = chebyshev_matrices((a, b), N)
            w = _cc_weights((a, b), N)
            self.assertAlmostEqual(float(np.sum(w)), b - a, places=12)
            for deg in range(0, N):
                exact = (b ** (deg + 1) - a ** (deg + 1)) / (deg + 1)
                approx = float(np.dot(w, x**deg))
                self.assertAlmostEqual(
                    approx / max(1.0, fabs(exact)),
                    exact / max(1.0, fabs(exact)),
                    places=10,
                    msg=f"CC weights not exact for degree {deg} at N={N}",
                )

    def test_cc_weights_transcendental(self):
        # cross-check against a known transcendental integral, not just polynomials.
        a, b = 0.0, 1.0
        x, _ = chebyshev_matrices((a, b), 25)
        w = _cc_weights((a, b), 25)
        approx = float(np.dot(w, np.exp(x)))
        exact = exp(b) - exp(a)
        self.assertTrue(
            fabs(approx - exact) < 1e-13,
            f"CC quadrature of exp(x) on [0,1]: expected {exact}, got {approx}",
        )

    def test_stationary_phase_gate_uses_total_variation(self):
        # C2: a phase whose *net* change across the region is zero (it returns to zero at x=1)
        # but whose total variation is ~5e5 radians must not be handed to the fallback as a single
        # region -- the old net-phase gate did exactly that and returned an answer wrong by 1590%.
        theta = lambda x: 1.0e6 * (x - x * x)
        f = [lambda x: exp(-x), lambda x: 0.0]
        oracle = -6.879079716900e-04

        data = adaptive_levin_sincos(
            (0.0, 1.0), f, theta={"theta": theta}, atol=1e-10, rtol=1e-10
        )
        value = data["value"]
        true_err = fabs(value - oracle)
        relerr = true_err / fabs(oracle)

        self.assertTrue(
            relerr < 1e-8,
            f"stationary-phase oracle: expected relative error < 1e-8, got {relerr:.3e} "
            f"(value={value}, oracle={oracle})",
        )
        # the reported abserr must not under-report the true error -- this is the property the
        # whole campaign is defending.
        self.assertGreater(data["abserr"], true_err)
        # more than one region, and more than one fallback region, confirms the gate did not just
        # hand the whole span to a single fallback panel the way the net-phase gate did.
        self.assertGreater(data["num_regions"], 1)
        self.assertGreater(data["num_simple_regions"], 1)

    def test_fallback_region_bisects_on_missed_tolerance(self):
        # C7: a fallback region whose nested-pair error estimate misses the requested tolerance
        # must be bisected, not accepted unconditionally the way the old scipy.quad branch was
        # (its abserr was recorded but never tested). Reuse the stationary-phase problem above,
        # whose fallback regions straddle the stationary point at x=0.5 and would otherwise need
        # depth 0 to satisfy a loose tolerance -- confirm they are bisected to depth > 0.
        theta = lambda x: 1.0e6 * (x - x * x)
        f = [lambda x: exp(-x), lambda x: 0.0]

        data = adaptive_levin_sincos(
            (0.0, 1.0), f, theta={"theta": theta}, atol=1e-10, rtol=1e-10
        )
        direct_regions = [r for r in data["regions"] if r.type == "direct"]
        self.assertGreater(len(direct_regions), 1)
        self.assertTrue(
            any(r.depth > 0 for r in direct_regions),
            "expected at least one fallback region to have been bisected (depth > 0)",
        )


if __name__ == "__main__":
    unittest.main()
