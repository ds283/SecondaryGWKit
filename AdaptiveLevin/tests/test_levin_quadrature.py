import contextlib
import io
import unittest
from math import fabs, atan, sin, exp, pi, nan

import numpy as np

from AdaptiveLevin.levin_quadrature import (
    adaptive_levin_sincos,
    _Basis_SinCos,
    _adaptive_levin_subregion_impl,
    chebyshev_matrices,
    _cc_weights,
    _roundoff_floor,
    _levin_G0_G1,
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

        # C3: before prompt 05, this reproduced at HEAD as reporting abserr=1.26e-10 at
        # atol=1e-10, exceeding what was requested, with nothing previously indicating it (the
        # per-region acceptance test compared each region's residual against atol directly, so
        # the summed abserr scaled with the number of accepted regions). As of prompt 05, atol is
        # distributed across regions by length share, so the summed abserr is bounded by atol by
        # construction and this now converges.
        def gaussian_bump(x):
            return exp(-400.0 * (x - 1.1) ** 2)

        f_bump = [gaussian_bump, lambda x: 0.0]
        theta_bump = lambda x: 3.0e4 * x

        data_bump = adaptive_levin_sincos(
            (0.0, 3.0), f_bump, theta={"theta": theta_bump}, atol=1e-10
        )
        self.assertTrue(data_bump["converged"])
        self.assertLess(data_bump["abserr"], 1e-10)

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

    def test_roundoff_floor_independent_of_theta_presentation(self):
        # C4 (prompt 04): the round-off floor (Chen et al. eq. 151) must not depend on whether
        # the phase is presented raw or range-reduced -- unlike the endpoint model it replaces,
        # which was optimistic by up to 8.1e9 on the reduced-phase path (see
        # prompts/levin-refactor/logs/04-roundoff-floor.md). Reproduce the audit's own problem,
        # int_{1/3}^{7/3} exp(-x) sin(w x) dx, at two representative frequencies.
        a, b = 1.0 / 3.0, 7.0 / 3.0

        def closed_form(omega):
            def F(x):
                return -exp(-x) * (sin(omega * x) + omega * np.cos(omega * x)) / (1.0 + omega * omega)

            return F(b) - F(a)

        for omega in (1.0e6, 1.0e12):
            f = [lambda x: exp(-x), lambda x: 0.0]
            theta_raw = {"theta": lambda x, w=omega: w * x}

            def theta_mod_2pi(x, w=omega):
                val = np.fmod(w * x, 2.0 * pi)
                if val < 0.0:
                    val += 2.0 * pi
                return val

            theta_reduced = {
                "theta": lambda x, w=omega: w * x,
                "theta_mod_2pi": theta_mod_2pi,
                "theta_deriv": lambda x, w=omega: w,
            }

            data_raw = adaptive_levin_sincos(
                (a, b), f, theta=theta_raw, atol=1e-18, rtol=1e-13, chebyshev_order=12
            )
            data_reduced = adaptive_levin_sincos(
                (a, b), f, theta=theta_reduced, atol=1e-18, rtol=1e-13, chebyshev_order=12
            )

            self.assertAlmostEqual(
                data_raw["abserr"],
                data_reduced["abserr"],
                delta=1e-3 * data_raw["abserr"],
                msg=f"reported abserr should be (near-)identical for raw vs reduced phase at omega={omega}",
            )

            true_err_raw = fabs(data_raw["value"] - closed_form(omega))
            true_err_reduced = fabs(data_reduced["value"] - closed_form(omega))
            self.assertLessEqual(true_err_raw, data_raw["abserr"])
            self.assertLessEqual(true_err_reduced, data_reduced["abserr"])

    def test_theta_abserr_declared_endpoint_term(self):
        # Recommendation 5.2 (prompt 04): a declared theta_abserr must raise the reported abserr
        # without changing the computed value.
        a, b = 1.0 / 3.0, 7.0 / 3.0
        omega = 1.0e6
        f = [lambda x: exp(-x), lambda x: 0.0]
        theta = lambda x: omega * x

        data_plain = adaptive_levin_sincos(
            (a, b), f, theta={"theta": theta}, atol=1e-18, rtol=1e-13, chebyshev_order=12
        )
        data_declared = adaptive_levin_sincos(
            (a, b),
            f,
            theta={"theta": theta, "theta_abserr": 1e-8},
            atol=1e-18,
            rtol=1e-13,
            chebyshev_order=12,
        )

        self.assertEqual(data_plain["value"], data_declared["value"])
        self.assertGreater(data_declared["abserr"], 10.0 * data_plain["abserr"])

        # a callable theta_abserr must behave identically to an equal-valued scalar
        data_callable = adaptive_levin_sincos(
            (a, b),
            f,
            theta={"theta": theta, "theta_abserr": lambda x: 1e-8},
            atol=1e-18,
            rtol=1e-13,
            chebyshev_order=12,
        )
        self.assertEqual(data_declared["value"], data_callable["value"])
        self.assertAlmostEqual(
            data_declared["abserr"], data_callable["abserr"], delta=1e-20
        )

    def test_roundoff_floor_G0_zero_is_unbounded(self):
        # prompt 04, note (b): eq. (151) is undefined at an interior stationary point (G0 = 0),
        # and this must be reported as +inf rather than as a division-by-zero crash or a
        # fabricated finite number.
        self.assertEqual(_roundoff_floor(1.0, 1.0, 0.0, 1.0, 12), float("inf"))

    def test_roundoff_floor_G0_noise_is_unbounded(self):
        # prompt 04, note (b): a G0 that is nonzero but at the floating-point noise floor of
        # theta' (relative to G1) must ALSO be treated as unbounded -- a bare G0 > 0 test is not
        # enough. This is a regression test for a real failure: on the C2 stationary-phase
        # oracle, a Levin region with one endpoint exactly at the true stationary point produced
        # G0 ~ 1.9e-9 against G1 ~ 2.5e5 from spectral-differentiation noise alone, and treating
        # it as a genuine value let a wrong value (490% off) be accepted as phase_limited. See
        # prompts/levin-refactor/logs/04-roundoff-floor.md.
        G0_noise = 1.9e-9
        G1 = 2.5e5
        self.assertEqual(_roundoff_floor(1.0, 1.0, G0_noise, G1, 12), float("inf"))
        # a G0 that is a genuine (non-noise) small fraction of G1 must NOT be treated as unbounded
        self.assertTrue(np.isfinite(_roundoff_floor(1.0, 1.0, 1.0e-3 * G1, G1, 12)))

    def test_roundoff_floor_subdivision_invariant(self):
        # A further argument for eq. (151) the audit itself does not make (README Sec 2.3g,
        # prompt 04): the summed floor over a cell decomposition is roughly invariant under how
        # finely the interval is subdivided, unlike the endpoint model it replaces (which grows
        # linearly with the region count). Reproduce it directly against _roundoff_floor(),
        # covering the constant-theta' regime the closed-form argument in README Sec 2.3g assumes
        # (G1 = omega*(h_cell/2) kept comfortably above k^2 = 144 at every N tested here).
        a, b = 1.0 / 3.0, 7.0 / 3.0
        omega = 1.0e6
        k = 12

        def cell_sum(n):
            edges = np.linspace(a, b, n + 1)
            total = 0.0
            for i in range(n):
                width = edges[i + 1] - edges[i]
                grid, _ = chebyshev_matrices((edges[i], edges[i + 1]), k)
                f_scale = float(np.max(np.exp(-grid)))
                theta_prime = omega * np.ones_like(grid)
                G0, G1 = _levin_G0_G1(theta_prime, width)
                total += _roundoff_floor(f_scale, width, G0, G1, k)
            return total

        sums = [cell_sum(n) for n in (1, 10, 100, 1000)]
        # flat to within a factor of a few, not the ~500x growth (matching the cell count) that
        # the endpoint model this replaces exhibited over a comparable range (README Sec 2.3g).
        self.assertLess(max(sums) / min(sums), 5.0)

    def test_mode_filter_gates_on_endpoint_not_mean(self):
        # Prompt 06 (problem (a)) regression: found while verifying this prompt, not
        # anticipated by the audit's own 400-problem randomised sweep (which reported no case
        # where the old mean-based gate changed a result). On this region the second
        # component's collocation-point mean ratio (~2.2e-11) sits *below* rtol=1e-10, while
        # its endpoint ratio (~1.4e-10) -- the quantity the estimate actually consumes -- sits
        # *above* it: the old gate would have wrongly discarded a mode the new one correctly
        # keeps. See prompts/levin-refactor/logs/06-mode-filter.md for the full measurement
        # (this is the GRZ integral at lambda=1000, restricted to its left half).
        f = [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)]
        BasisData = _Basis_SinCos({"theta": lambda x: 1000.0 * atan(x)})
        data = _adaptive_levin_subregion_impl(
            (-1.0, 0.0), f, BasisData, id_label=None, chebyshev_order=16, rtol=1e-10
        )
        self.assertFalse(data["is_direct"])
        self.assertGreater(data["p_ratios"][1], 1e-10)

    def test_mode_filter_truncation_visible_in_abserr(self):
        # Prompt 06 (problem (b)): a p-mode dropped by the rtol gate used to vanish from the
        # region's value with no matching change in abserr -- parent and children apply
        # identical gating, so the drop is common-mode in the step-(4) residual and subtracts
        # out exactly. Reproduces the audit's Sec1.5b measurement: a single region (order 16,
        # phase span >> SIX_PI) whose second (coupled, f2 = 0) component has an endpoint ratio
        # near 1e-6 -- discarded at rtol = 1e-2, kept at rtol <= 1e-6. See
        # prompts/levin-refactor/logs/06-mode-filter.md for the full before/after table.
        a, b = 1.0, 1.0003
        omega = 1.0e6
        f = [lambda x: exp(-x), lambda x: 0.0]
        theta = {"theta": lambda x: omega * x}

        loose = adaptive_levin_sincos(
            (a, b), f, theta, atol=1e-20, rtol=1e-2, chebyshev_order=16, depth_max=20
        )
        tight = adaptive_levin_sincos(
            (a, b), f, theta, atol=1e-20, rtol=1e-8, chebyshev_order=16, depth_max=20
        )

        self.assertEqual(loose["num_regions"], 1)
        self.assertEqual(tight["num_regions"], 1)

        jump = fabs(loose["value"] - tight["value"])
        self.assertGreater(jump, 0.0)

        # the mode was actually discarded at the loose tolerance ...
        self.assertGreater(loose["abserr_truncation"], 0.0)
        # ... and the reported abserr now bounds the value it perturbed -- before this prompt,
        # abserr stayed near 1e-16 (round-off only) regardless of rtol, while the value itself
        # jumped by an amount bounded only by rtol; see the log for the actual numbers.
        self.assertGreaterEqual(loose["abserr"], jump)

        # at a tight tolerance the mode is retained and there is nothing to account for
        self.assertEqual(tight["abserr_truncation"], 0.0)


if __name__ == "__main__":
    unittest.main()
