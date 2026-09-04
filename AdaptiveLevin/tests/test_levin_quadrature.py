import unittest
from math import fabs, atan, sin, cos, exp, pi, nan

import numpy as np
from scipy.special import sici

from AdaptiveLevin.levin_quadrature import (
    adaptive_levin_sincos,
    _Basis_SinCos,
    _adaptive_levin,
    _adaptive_levin_subregion_impl,
    chebyshev_matrices,
    _cc_weights,
    _roundoff_floor,
    _levin_G0_G1,
    _detect_vectorized,
    _sample_vectorized,
)
from utilities import format_time


class _TwoIndependentSinCosBasis:
    """
    Prompt 10 (C12 gap: m != 2). A minimal, self-contained basis exercising the *generic*
    m-component code path in _adaptive_levin_subregion_impl() -- the np.block-assembled real
    system, the row_list construction, the P.reshape(m, chebyshev_order) layout -- which
    _Basis_SinCos's m == 2 complexified fast path (prompt 02) bypasses entirely. There is no
    public entry point for m != 2 (adaptive_levin_sincos hard-validates len(f) == 2), so this
    basis is driven directly through _adaptive_levin(), which is written for general m (standing
    note 10 of IMPLEMENTATION_STATE.md).

    The basis is w = (sin theta1, cos theta1, sin theta2, cos theta2): two independent (sin, cos)
    pairs at independent linear phases theta1(x) = lambda1*x, theta2(x) = lambda2*x, stacked block-
    diagonally. Each pair obeys exactly the same w' = A w relation _Basis_SinCos implements (Amat =
    [[0, theta'], [-theta', 0]]) with the two pairs decoupled, so

        w1' = theta1' w2,   w2' = -theta1' w1,   w3' = theta2' w4,   w4' = -theta2' w3

    i.e. A^T = block_diag([[0, -theta1'], [theta1', 0]], [[0, -theta2'], [theta2', 0]]). This is
    not a contrived degenerate case: it is exactly the shape a future basis combining two
    independent oscillatory phases (e.g. two Liouville-Green modes) would take -- the audit itself
    (docs/adaptive-levin-audit-2026-09.md Sec 4.2) says the generic m-component path "should be
    retained for the future bases sketched in the [prior] review's Sec 8.3", and
    IMPLEMENTATION_STATE.md standing note 10 records the same requirement.

    With f = [1, 0, 0, 1] this evaluates
        integral f1(x) sin(theta1) + f2(x) cos(theta1) + f3(x) sin(theta2) + f4(x) cos(theta2) dx
      = integral sin(lambda1 x) dx + integral cos(lambda2 x) dx
    which has an elementary closed form -- see test_generic_m_component_basis_two_decoupled_phases.
    """

    def __init__(self, lambda1: float, lambda2: float):
        self._lambda1 = lambda1
        self._lambda2 = lambda2

    def theta_abserr_at(self, x):
        # No declared phase error for this synthetic basis -- same "no theta_abserr supplied"
        # contract _Basis_SinCos.theta_abserr_at() implements when its own "theta_abserr" key is
        # absent (see _declared_endpoint_phase_err()).
        return None

    def build_Levin_data(
        self,
        grid,
        Dmat,
        notify_label=None,
        id_label=None,
        need_AmatT: bool = True,
        vectorize_cache=None,
    ):
        theta1 = self._lambda1 * grid
        theta2 = self._lambda2 * grid
        theta1_prime = self._lambda1 * np.ones_like(grid)
        theta2_prime = self._lambda2 * np.ones_like(grid)

        AmatT = None
        if need_AmatT:
            N = len(grid)
            zero = np.zeros((N, N))
            t1 = np.diag(theta1_prime)
            t2 = np.diag(theta2_prime)
            AmatT = np.block(
                [
                    [zero, -t1, zero, zero],
                    [t1, zero, zero, zero],
                    [zero, zero, zero, -t2],
                    [zero, zero, t2, zero],
                ]
            )

        # Total variation across both phases -- generalises _Basis_SinCos's own
        # mean|theta'| * width construction to a two-phase basis. This basis is only ever driven
        # with a phase_span comfortably above SIX_PI in this test file's own use of it, so the
        # Clenshaw-Curtis fallback path (which also calls eval_basis() below) is not separately
        # exercised by this basis.
        phase_span = float(
            (np.mean(np.fabs(theta1_prime)) + np.mean(np.fabs(theta2_prime)))
            * np.fabs(grid[0] - grid[-1])
        )

        # grid is descending: grid[-1] is the lower endpoint a, grid[0] is the upper endpoint b.
        theta1_a, theta1_b = theta1[-1], theta1[0]
        theta2_a, theta2_b = theta2[-1], theta2[0]
        w0 = [np.sin(theta1_a), np.cos(theta1_a), np.sin(theta2_a), np.cos(theta2_a)]
        wk = [np.sin(theta1_b), np.cos(theta1_b), np.sin(theta2_b), np.cos(theta2_b)]

        return AmatT, theta1_prime, w0, wk, phase_span

    def eval_basis(self, x):
        theta1 = self._lambda1 * x
        theta2 = self._lambda2 * x
        return [np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2)]


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

        # Prompt 10: tightened from the campaign-original 1e-10 (eight orders of magnitude
        # looser than what this module actually delivers -- README standing note 4) to 1e-12,
        # using the full-precision closed form (the old literal, 0.5581795618, is itself only
        # accurate to ~3.5e-11, which is *tighter* than the old threshold allowed for). Measured
        # true error here is ~7.8e-16; 1e-12 is a ~1000x margin above that, not a value set at
        # the measured error itself (which would flake).
        true_value = -cos(50000.0) + cos(1.0)
        true_err = fabs(value - true_value)
        self.assertTrue(
            true_err < 1e-12,
            f"Sin integral test failed: expected {true_value}, obtained {value} (true_err={true_err:.3e})",
        )
        # abserr must bound the true error against the closed form (README Sec 5.2; audit rec 16).
        self.assertGreaterEqual(data["abserr"], true_err)

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

        # Prompt 10: tightened from 1e-10 to 1e-12 using the full-precision closed form (the old
        # literal, -0.6636397833, is itself only accurate to ~3.6e-11). Measured true error here
        # is ~1.2e-15; 1e-12 is a ~1000x margin above that.
        true_value = sin(500000.0) - sin(1.0)
        true_err = fabs(value - true_value)
        self.assertTrue(
            true_err < 1e-12,
            f"Cos integral test failed: expected {true_value}, obtained {value} (true_err={true_err:.3e})",
        )
        self.assertGreaterEqual(data["abserr"], true_err)

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

        # Prompt 10: tightened from 1e-10 to 1e-11 using the exact closed form Si(10000) - Si(100)
        # (substitute u = 100x in integral_1^100 sin(100x)/x dx) instead of the truncated literal
        # 0.00866607847. Measured true error here is ~9.8e-14 (this problem needs 10 regions to
        # resolve, so the round-off floor is not the whole story) -- 1e-11 is a ~100x margin above
        # that.
        si_hi, _ = sici(100.0 * 100.0)
        si_lo, _ = sici(100.0 * 1.0)
        true_value = si_hi - si_lo
        true_err = fabs(value - true_value)
        self.assertTrue(
            true_err < 1e-11,
            f"Sinc integral test failed: expected {true_value}, obtained {value} (true_err={true_err:.3e})",
        )
        self.assertGreaterEqual(data["abserr"], true_err)

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
        true_err = fabs(value - analytic_sol)
        # Prompt 10: tightened from 1e-10 to 1e-11. Measured worst-case true error across
        # lambda = 10, 100, 1000 is ~2.8e-13 (at lambda=100); 1e-11 is a ~35x margin above that,
        # not a value set at the measured error itself.
        self.assertTrue(
            true_err < 1e-11,
            f"Gradshtein & Ryzhik integral test: expected {analytic_sol}, obtained {value} "
            f"(true_err={true_err:.3e})",
        )
        self.assertGreaterEqual(data["abserr"], true_err)
        return data

    def test_GRZIntegral(self):
        # C12: the audit's own text says "_GRZIntegral(10.0) takes the fallback by accident...
        # that test never exercises Levin at all". README Sec 2.4 note (d) corrects this: the
        # test also runs at lambda=100 and lambda=1000, which do reach the Levin rule -- but which
        # path each lambda takes was never asserted, only exercised incidentally. Make it
        # explicit, using num_simple_regions (rec 16). Re-derived against the post-prompt-03
        # total-variation gate, not assumed unchanged from the audit's pre-campaign analysis:
        # theta'(x) = lambda/(1+x^2) never changes sign on [-1, 1], so total variation equals net
        # phase change here regardless of which gate is used.
        data10 = self._GRZIntegral(10.0)
        self.assertEqual(
            data10["num_simple_regions"],
            data10["num_regions"],
            "GRZIntegral(10.0): expected every region to take the Clenshaw-Curtis fallback "
            "(net phase change 10*(atan(1)-atan(-1)) ~= 15.7 rad is below SIX_PI ~= 18.8 rad, "
            "so the Levin rule should never be invoked)",
        )

        data100 = self._GRZIntegral(100.0)
        self.assertLess(
            data100["num_simple_regions"],
            data100["num_regions"],
            "GRZIntegral(100.0): expected at least one region to take the Levin rule",
        )

        data1000 = self._GRZIntegral(1000.0)
        self.assertLess(
            data1000["num_simple_regions"],
            data1000["num_regions"],
            "GRZIntegral(1000.0): expected at least one region to take the Levin rule",
        )

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

        # chebyshev_order < 8: warned about and clamped, not rejected. As of prompt 07
        # (prompts/levin-refactor/logs/07-diagnostics-hygiene.md) this module logs through the
        # standard "AdaptiveLevin.levin_quadrature" logger rather than printing to stdout, so the
        # warning is captured with assertLogs rather than stdout redirection.
        with self.assertLogs(
            "AdaptiveLevin.levin_quadrature", level="WARNING"
        ) as log_ctx:
            data = adaptive_levin_sincos(
                (1.0, 50.0), good_f, theta={"theta": theta}, chebyshev_order=4
            )
        self.assertTrue(any("clamp" in message.lower() for message in log_ctx.output))
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
                return (
                    -exp(-x)
                    * (sin(omega * x) + omega * np.cos(omega * x))
                    / (1.0 + omega * omega)
                )

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
                (a, b),
                f,
                theta=theta_reduced,
                atol=1e-18,
                rtol=1e-13,
                chebyshev_order=12,
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
            (a, b),
            f,
            theta={"theta": theta},
            atol=1e-18,
            rtol=1e-13,
            chebyshev_order=12,
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

    def test_theta_mod_2pi_path_only(self):
        # C12 gap: "the theta_mod_2pi path -- the branch production actually uses". Every other
        # test in this file supplies either the raw "theta" alone or a full {theta, theta_mod_2pi,
        # theta_deriv} triple; ComputeTargets/QuadSourceIntegral.py (the production caller,
        # standing note 7) supplies theta + theta_mod_2pi but deliberately NOT theta_deriv, so
        # theta' is obtained by spectral differentiation of the raw phase (need_theta_Cheb is
        # True) while sin/cos are evaluated from the range-reduced value (the
        # hasattr(self, "_theta_mod_2pi") branch in build_Levin_data/eval_basis). This is the one
        # combination no existing test isolated.
        a, b = 1.0 / 3.0, 7.0 / 3.0
        omega = 1.0e6
        f = [lambda x: exp(-x), lambda x: 0.0]

        def theta_mod_2pi(x, w=omega):
            val = np.fmod(w * x, 2.0 * pi)
            if val < 0.0:
                val += 2.0 * pi
            return val

        theta = {"theta": lambda x, w=omega: w * x, "theta_mod_2pi": theta_mod_2pi}
        data = adaptive_levin_sincos(
            (a, b), f, theta=theta, atol=1e-18, rtol=1e-13, chebyshev_order=12
        )

        def closed_form(w):
            def F(x):
                return -exp(-x) * (sin(w * x) + w * cos(w * x)) / (1.0 + w * w)

            return F(b) - F(a)

        true_err = fabs(data["value"] - closed_form(omega))
        # Measured true error here is ~1.1e-16; 1e-13 is a >500x margin above that.
        self.assertLess(true_err, 1e-13)
        self.assertGreaterEqual(data["abserr"], true_err)

    def test_theta_deriv_path_only(self):
        # C12 gap: "the theta_deriv path (all four [original] tests use spectral
        # differentiation)". Supply theta_deriv alone (no theta_mod_2pi), so sin/cos are still
        # evaluated from the raw phase but theta' comes directly from the caller rather than from
        # differentiating a sampled theta -- the branch three_bessel_integrals.py's four phase
        # groups use (standing note in README Sec 2.5).
        a, b = 1.0 / 3.0, 7.0 / 3.0
        omega = 1.0e6
        f = [lambda x: exp(-x), lambda x: 0.0]
        theta = {"theta": lambda x, w=omega: w * x, "theta_deriv": lambda x, w=omega: w}

        data = adaptive_levin_sincos(
            (a, b), f, theta=theta, atol=1e-18, rtol=1e-13, chebyshev_order=12
        )

        def closed_form(w):
            def F(x):
                return -exp(-x) * (sin(w * x) + w * cos(w * x)) / (1.0 + w * w)

            return F(b) - F(a)

        true_err = fabs(data["value"] - closed_form(omega))
        # Measured true error here is ~1.0e-16; 1e-13 is a >500x margin above that.
        self.assertLess(true_err, 1e-13)
        self.assertGreaterEqual(data["abserr"], true_err)

    def test_reversed_span(self):
        # C12 gap: "reversed span". Not covered by any existing test -- module docstring/audit
        # Sec 1.1 claims a reversed span (b < a) "works and returns the correctly negated value.
        # Untested, but correct." Verify directly, and check both directions converge.
        f = [lambda x: 1.0, lambda x: 0.0]
        theta = {"theta": lambda x: 100.0 * x}

        forward = adaptive_levin_sincos(
            (1.0, 5.0), f, theta=theta, atol=1e-13, rtol=1e-11
        )
        reversed_ = adaptive_levin_sincos(
            (5.0, 1.0), f, theta=theta, atol=1e-13, rtol=1e-11
        )

        self.assertTrue(forward["converged"])
        self.assertTrue(reversed_["converged"])
        # exactly negated, up to round-off -- not merely "close"
        self.assertAlmostEqual(forward["value"], -reversed_["value"], delta=1e-13)

        true_value = (cos(1.0 * 100.0) - cos(5.0 * 100.0)) / 100.0
        self.assertLess(fabs(forward["value"] - true_value), 1e-11)
        self.assertLess(fabs(reversed_["value"] + true_value), 1e-11)

    def test_generic_m_component_basis_two_decoupled_phases(self):
        # C12 gap: "m != 2". There is no public entry point for a basis with other than two
        # components (adaptive_levin_sincos hard-validates len(f) == 2, per prompt 01's
        # _Basis_SinCos-specific check), so this drives _adaptive_levin() directly with a minimal
        # m=4 basis (_TwoIndependentSinCosBasis, defined at module scope above) -- two independent
        # (sin, cos) pairs at independent linear phases, exercising the generic real-valued
        # np.block assembly / row_list construction that _Basis_SinCos's m==2 complexified fast
        # path (prompt 02) bypasses. This is the only test of the generic m-component path in the
        # repository; the generic path exists precisely so a future basis (prior review Sec 8.3)
        # has somewhere to land -- see the class docstring for the derivation and the closed form.
        lambda1, lambda2 = 1.0, 2.0
        x_span = (1.0, 50.0)
        f = [lambda x: 1.0, lambda x: 0.0, lambda x: 0.0, lambda x: 1.0]
        basis = _TwoIndependentSinCosBasis(lambda1, lambda2)

        data = _adaptive_levin(
            x_span, f, basis, atol=1e-13, rtol=1e-11, chebyshev_order=16
        )

        true_value = (cos(1.0) - cos(50.0)) + 0.5 * (sin(100.0) - sin(2.0))
        true_err = fabs(data["value"] - true_value)

        self.assertTrue(data["converged"])
        # Measured true error here is ~2.0e-15; 1e-11 is a >1000x margin above that.
        self.assertLess(true_err, 1e-11)
        self.assertGreaterEqual(data["abserr"], true_err)


class TestVectorizedSampling(unittest.TestCase):
    """
    Prompt 08 (recommendation 14): _sample_vectorized() samples a callable at every point of a
    grid with one array call in place of a Python loop, when detection (_detect_vectorized())
    confirms the callable supports it. See prompts/levin-refactor/logs/08-order-and-sampling.md.
    """

    def test_detects_and_uses_genuinely_vectorized_callable(self):
        grid = np.linspace(1.0, 5.0, 12)

        def f(x):
            return np.sin(x) * x

        self.assertTrue(_detect_vectorized(f, grid))

        cache = {}
        result = _sample_vectorized(f, grid, cache, ("f", id(f)))
        expected = np.array([f(x) for x in grid])
        # bit-equal: sampling the same callable at the same points as a single array call or as
        # an elementwise loop must agree exactly for a callable that does not branch on input type
        # (verification item 4 of prompt 08).
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(cache[("f", id(f))])

    def test_constant_broadcast_is_not_mistaken_for_vectorized(self):
        # lambda x: 1.0 "vectorises" trivially (it ignores its argument) but returns a bare
        # float, not an array of the grid's shape, for an array input -- exactly the "silently
        # broadcasts" failure mode the prior review's Sec 7.6 warns about. Detection must reject
        # it on the shape check, not accept it and corrupt every non-constant caller that shares
        # its cache key by coincidence (it does not, since the key includes id(func), but the
        # rejection itself is the property under test).
        grid = np.linspace(1.0, 5.0, 12)

        def const_f(x):
            return 1.0

        self.assertFalse(_detect_vectorized(const_f, grid))

        cache = {}
        result = _sample_vectorized(const_f, grid, cache, ("f", id(const_f)))
        np.testing.assert_array_equal(result, np.ones_like(grid))
        self.assertFalse(cache[("f", id(const_f))])

    def test_scalar_only_callable_falls_back_without_raising(self):
        # math.atan raises on an array argument (TypeError: only length-1 arrays can be
        # converted to Python scalars) -- exactly the shape of theta = lambda x: lbda*atan(x)
        # used by this module's own _GRZIntegral test helper. Detection must catch the failure
        # and fall back to the loop, not propagate the exception.
        grid = np.linspace(0.1, 0.9, 12)

        def scalar_only(x):
            return atan(x)

        self.assertFalse(_detect_vectorized(scalar_only, grid))

        cache = {}
        result = _sample_vectorized(scalar_only, grid, cache, ("f", id(scalar_only)))
        expected = np.array([scalar_only(x) for x in grid])
        np.testing.assert_array_equal(result, expected)
        self.assertFalse(cache[("f", id(scalar_only))])

    def test_detection_runs_once_per_callable_not_once_per_call(self):
        grid = np.linspace(1.0, 5.0, 12)
        calls = {"count": 0}

        def f(x):
            calls["count"] += 1
            return np.sin(x)

        cache = {}
        for _ in range(5):
            _sample_vectorized(f, grid, cache, ("f", id(f)))

        # one array call to sample the grid, per invocation (5), plus the one-time detection
        # probe's one array call and two scalar calls -- detection itself must not repeat.
        self.assertEqual(calls["count"], 5 + 1 + 2)

    def test_end_to_end_value_unchanged_by_vectorized_sampling(self):
        # A full adaptive_levin_sincos() run with a genuinely vectorizing integrand must return
        # the same value as the same integral evaluated through the non-vectorizing SincIntegral-
        # style callables elsewhere in this file -- vectorized sampling is a performance path,
        # not a numerical one.
        x_span = (1.0, 100.0)
        f_vectorized = [lambda x: 1.0 / x, lambda x: 0.0 * x]
        theta = {"theta": lambda x: 100.0 * x}

        data = adaptive_levin_sincos(
            x_span,
            f_vectorized,
            theta=theta,
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=12,
        )
        self.assertTrue(
            fabs(data["value"] - 0.00866607847) < 1e-10,
            f"expected {0.00866607847}, obtained {data['value']}",
        )


if __name__ == "__main__":
    unittest.main()
