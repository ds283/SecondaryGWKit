"""
Tests for ``LiouvilleGreen.bessel_near_region``, the branch-tracked near-region sampler.

Everything numerical here is scored against ``LiouvilleGreen.tests.bessel_reference`` (campaign
prompt 01), never against a reference this module invents for itself. The three error metrics are
the campaign's, defined once in ``DRAFT-PLAN.md`` §6.1 and implemented in ``bessel_reference``:

    E_theta = max_x max(|sin theta_ours - J/A|, |-cos theta_ours - Y/A|),
    E_A     = max_x |A_ours/A - 1|,        A = hypot(J, Y) from the reference functions,

plus the relative error of ``theta'`` against ``(2/pi)/(x (J^2 + Y^2))``.

Two reference notes that shape which tier is used where. SciPy ``jv``/``yv`` are an adequate
reference below ``nu = 20.5`` at the 1e-11 level, but above it they carry their own floor -- 1.2e-12
in ``r`` at ``nu = 100.5`` and 8.4e-12 at ``nu = 1000.5``, measured in campaign log 03 -- which is
above what this construction achieves there. So the high-order sweeps assert at the campaign's
1e-6 high-order target, where the SciPy floor is irrelevant, and the *sharp* high-order statements
are made at the cached 40-digit ``mpmath`` corners instead. Second, everything tested here lies
below ``x = 6e4``, hence far below ``scipy_reference_max_x(nu)`` in every case, so the silent Amos
boundary of ``bessel_reference`` is never approached.
"""

import ast
import inspect
import math
import time
import unittest
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import make_interp_spline

from LiouvilleGreen import bessel_near_region as nr
from LiouvilleGreen import bessel_tail as bt
from LiouvilleGreen.tests import bessel_reference as br

LOW_ORDERS = (0.5, 1.5, 1.75, 2.5)
HIGH_ORDERS = (20.5, 100.5, 1000.5)

#: Budgets the low-order builds are asked for: the campaign's ``1e-11`` acceptance target.
TIGHT_BUDGET = 1e-11

#: The campaign's high-order acceptance target. "Accuracy is not the objective at these orders;
#: correctness is" (``README.md`` §6).
LOOSE_BUDGET = 1e-6

#: Derivative targets from the acceptance table: ``1e-9`` at low order, ``1e-6`` at high order.
LOW_ORDER_DERIV_TARGET = 1e-9
HIGH_ORDER_DERIV_TARGET = 1e-6

_BUILD_CACHE: Dict[Tuple[float, float], nr.NearRegionData] = {}


def build(nu: float, budget: float = TIGHT_BUDGET) -> nr.NearRegionData:
    """Build (and cache) the near region for one order at one budget, with its own crossover."""
    key = (nu, budget)
    cached = _BUILD_CACHE.get(key)
    if cached is None:
        x_lo = bt.construction_min_x(nu)
        crossover = bt.tail_crossover(nu, budget, budget)
        cached = nr.build_near_region(nu, x_lo, crossover.x_star, budget, budget)
        _BUILD_CACHE[key] = cached
    return cached


def reconstruct(nu: float, x, data: nr.NearRegionData):
    """
    Reconstruct ``(sin theta, -cos theta, A, theta'_from_ell, theta'_from_r)`` at ``x``.

    The sine and cosine are formed by **angle addition** on ``d = c_nu + r``, never as
    ``sin(x + d)``: at ``x = 1e15`` the naive form loses 4.7e-2 (``DRAFT-PLAN.md`` §6.3). Nothing
    tested here reaches that argument, but the evaluation path this module's data is destined for
    does, and testing it any other way would measure a different construction.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    u = np.log(x)
    r = np.atleast_1d(data.r_interp(u))
    ell = np.atleast_1d(data.log_a_interp(u))

    d = br.c_nu(nu) + r
    sin_theta = np.sin(x) * np.cos(d) + np.cos(x) * np.sin(d)
    cos_theta = np.cos(x) * np.cos(d) - np.sin(x) * np.sin(d)

    amplitude = np.sqrt(2.0 / (math.pi * x)) * np.exp(ell)
    theta_prime = np.exp(-2.0 * ell)
    theta_prime_alt = 1.0 + np.atleast_1d(data.r_interp.derivative()(u)) / x

    return sin_theta, -cos_theta, amplitude, theta_prime, theta_prime_alt


def score(nu: float, x, data: nr.NearRegionData, tier: str = br.TIER_SCIPY) -> dict:
    """Every metric at once, each with the ``x`` at which its maximum falls."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    J, Y = br.reference_JY(nu, x, tier=tier)
    amplitude = br.reference_amplitude(J, Y)
    theta_prime_ref = br.reference_theta_deriv(x, J, Y)

    sin_theta, minus_cos, A_ours, theta_prime, theta_prime_alt = reconstruct(
        nu, x, data
    )

    e_theta, i_theta = br.phase_pair_error(sin_theta, minus_cos, J, Y, amplitude)
    e_A, i_A = br.amplitude_error(A_ours, amplitude)
    e_d, i_d = br.derivative_error(theta_prime, theta_prime_ref)
    e_alt, i_alt = br.derivative_error(theta_prime_alt, theta_prime_ref)
    e_routes, i_routes = br.derivative_error(theta_prime, theta_prime_alt)

    return {
        "E_theta": (e_theta, float(x[i_theta])),
        "E_A": (e_A, float(x[i_A])),
        "deriv": (e_d, float(x[i_d])),
        "deriv_alt": (e_alt, float(x[i_alt])),
        "routes": (e_routes, float(x[i_routes])),
    }


def point_sets(data: nr.NearRegionData) -> Dict[str, np.ndarray]:
    """
    The three point sets ``DRAFT-PLAN.md`` §9 Stage 1 requires be kept separate: the sample nodes,
    the log-interval midpoints, and the endpoint intervals sampled densely.

    Endpoint intervals get their own set because they are where a piecewise fit is least
    constrained and where the existing ``test_phase_derivative`` deliberately does *not* look --
    it starts at ``2 x_0 + 1`` (``RECONCILIATION.md`` C4).
    """
    u = data.log_x_nodes
    return {
        "nodes": np.exp(u),
        "midpoints": np.exp(0.5 * (u[:-1] + u[1:])),
        "endpoint intervals": np.exp(
            np.concatenate([np.linspace(u[0], u[1], 33), np.linspace(u[-2], u[-1], 33)])
        ),
    }


class TestScaledHankelSampling(unittest.TestCase):
    """The sampled quantity, and the two deliberate refinements in how it is formed."""

    def test_the_identity_is_exact_not_asymptotic(self):
        """
        ``S_nu = sqrt(pi x/2) exp(i(pi nu/2 + pi/4)) hankel1e(nu, x) = a_nu exp(i r_nu)`` exactly,
        so ``A sin theta`` and ``-A cos theta`` reproduce ``J`` and ``Y`` from the *samples alone*,
        with no interpolation and no branch tracking anywhere (campaign ``README.md`` §2 (a)).

        The threshold is loosened above ``nu = 20.5`` because at those orders it is the SciPy
        reference that is limiting, not the samples: campaign log 03 measures a ``jv``/``yv``
        floor of 1.2e-12 in ``r`` at ``nu = 100.5`` and 8.4e-12 at 1000.5. What the samples
        actually achieve is settled against the 40-digit corners in
        ``TestHighOrderAccuracy.test_against_the_cached_mpmath_corners``.
        """
        report = []
        for nu in (0.5, 1.5, 1.75, 2.5, 20.5, 100.5, 1000.5):
            with self.subTest(nu=nu):
                x_lo = bt.construction_min_x(nu)
                x = np.exp(
                    np.linspace(math.log(x_lo), math.log(50.0 * max(nu, 1.0)), 40)
                )
                a, raw_angle, _ = nr.sample_scaled_hankel(nu, x, band=(0.99, 8.0))

                d = br.c_nu(nu) + raw_angle
                A = np.sqrt(2.0 / (math.pi * x)) * a
                sin_theta = np.sin(x) * np.cos(d) + np.cos(x) * np.sin(d)
                cos_theta = np.cos(x) * np.cos(d) - np.sin(x) * np.sin(d)
                J, Y = br.reference_JY(nu, x, tier=br.TIER_SCIPY)
                amplitude = br.reference_amplitude(J, Y)

                e_theta, i_theta = br.phase_pair_error(
                    sin_theta, -cos_theta, J, Y, amplitude
                )
                e_A, i_A = br.amplitude_error(A, amplitude)
                target = 1e-13 if nu <= 20.5 else 1e-10
                report.append(
                    f"nu={nu:8} samples only: E_theta={e_theta:.3e}@{x[i_theta]:.6g} "
                    f"E_A={e_A:.3e}@{x[i_A]:.6g} (target {target:.0e})"
                )
                self.assertLess(e_theta, target, report[-1])
                self.assertLess(e_A, target, report[-1])
        print("\n" + "\n".join(report))

    def test_the_rotation_constant_is_reduced_before_multiplication(self):
        """
        ``scaled_hankel_phase_constant`` reduces ``(2 nu + 1)/4`` mod 2 before multiplying by
        ``pi``, so it costs one rounding instead of an error growing with ``nu``. At
        ``nu = 1000.5`` the exact answer is ``pi/2``.

        Two costs of not doing it, both asserted: reducing the naive
        ``pi nu/2 + pi/4 = 1572.367...`` mod ``2 pi`` afterwards leaves an error of 6.6e-14, and
        *not* reducing it -- which is what a residual accumulated on top of an unreduced constant
        would do -- carries an ulp of 2.3e-13. Both are at or above the sampling floor
        ``DRAFT-PLAN.md`` §4.4 records for this order, 2.96e-13.
        """
        self.assertEqual(nr.scaled_hankel_phase_constant(1000.5), 0.5 * math.pi)

        naive = math.pi * 1000.5 / 2.0 + math.pi / 4.0
        discrepancy = abs(math.fmod(naive, 2.0 * math.pi) - 0.5 * math.pi)
        self.assertGreater(
            discrepancy,
            1e-14,
            "the naive rotation constant is expected to be wrong by ~7e-14 at nu=1000.5; "
            f"measured {discrepancy:.3e}",
        )
        self.assertGreater(math.ulp(naive), 2e-13)

        # ... and it agrees with the naive form wherever the naive form is still accurate.
        for nu in (0.5, 1.5, 1.75, 2.5):
            with self.subTest(nu=nu):
                naive = math.fmod(math.pi * nu / 2.0 + math.pi / 4.0, 2.0 * math.pi)
                self.assertAlmostEqual(
                    nr.scaled_hankel_phase_constant(nu), naive, delta=1e-15
                )


class TestAmplitudePlausibilityBand(unittest.TestCase):
    """
    The guard is a two-sided band on ``a_nu`` and not an ``isfinite`` check. Both sides are
    tested, because a one-sided ``a >~ 1`` test would pass a spuriously large value.
    """

    def test_it_rejects_the_minus_zero_j_failure_that_isfinite_passes(self):
        """
        ``hankel1e(100.5, 1e9)`` is exactly ``-0j`` (``README.md`` §2 (e), prompt 02 Fact 1).
        This asserts both halves of the design decision: ``isfinite`` accepts it, and the band
        does not.
        """
        from scipy.special import hankel1e

        nu, x = 100.5, 1e9
        value = hankel1e(nu, x)
        self.assertEqual(value, -0j)
        self.assertTrue(
            np.isfinite(value),
            "the -0j return is finite, which is exactly why an isfinite guard does not catch it",
        )
        self.assertEqual(np.log(np.abs(value)), -np.inf)

        a = math.sqrt(0.5 * math.pi * x) * abs(value)
        self.assertEqual(a, 0.0)

        with self.assertRaises(nr.AmplitudeBandError) as caught:
            nr.check_amplitude_band(nu, x, a, hankel=value)
        message = str(caught.exception)
        self.assertIn("100.5", message)
        self.assertIn(f"{x:.17g}", message)
        self.assertIn("isfinite", message)

    def test_it_rejects_a_spuriously_large_amplitude(self):
        """The upper side. A one-sided ``a >~ 1`` test would accept this without complaint."""
        with self.assertRaises(nr.AmplitudeBandError) as caught:
            nr.check_amplitude_band(1000.5, 1234.5, 42.0)
        message = str(caught.exception)
        self.assertIn("1000.5", message)
        self.assertIn("1234.5", message)

    def test_it_accepts_the_measured_range(self):
        """
        The measured range over the whole near region is ``[1.0000000000, 3.54597]`` for every
        order up to 1000.5 (``RECONCILIATION.md`` §3.1); the band must contain all of it with
        room to spare, and must contain nothing silly.
        """
        for a in (1.0, 1.0000250016, 1.32288, 1.85684, 2.41795, 3.54597):
            nr.check_amplitude_band(1000.5, 1000.0, a)
        self.assertLess(nr.AMPLITUDE_BAND_LO, 1.0)
        self.assertGreater(nr.AMPLITUDE_BAND_HI, 3.54597)

    def test_every_built_sample_lies_inside_the_band(self):
        """A build is a standing assertion that the band holds; make it an explicit one too."""
        for nu in LOW_ORDERS + HIGH_ORDERS:
            with self.subTest(nu=nu):
                data = build(nu)
                self.assertGreaterEqual(
                    float(np.min(data.a_nodes)), nr.AMPLITUDE_BAND_LO
                )
                self.assertLessEqual(float(np.max(data.a_nodes)), nr.AMPLITUDE_BAND_HI)


class TestLowOrderAccuracy(unittest.TestCase):
    """``E_theta, E_A <= 1e-11`` at the orders production builds, on three separate point sets."""

    def test_phase_and_amplitude(self):
        report = []
        for nu in (1.5, 1.75, 2.5):
            data = build(nu)
            for label, x in point_sets(data).items():
                with self.subTest(nu=nu, points=label):
                    s = score(nu, x, data)
                    report.append(
                        f"nu={nu:6} {label:19s} E_theta={s['E_theta'][0]:.3e} at "
                        f"x={s['E_theta'][1]:.6g}, E_A={s['E_A'][0]:.3e} at x={s['E_A'][1]:.6g}"
                    )
                    self.assertLess(s["E_theta"][0], TIGHT_BUDGET, report[-1])
                    self.assertLess(s["E_A"][0], TIGHT_BUDGET, report[-1])
        print("\n" + "\n".join(report))

    def test_phase_derivative(self):
        """
        ``theta' = exp(-2 ell)`` to 1e-9 relative at the midpoints. ``DRAFT-PLAN.md`` §4.7 predicts
        the maximum in the interval adjacent to the turning point; the location is printed so a
        maximum somewhere else is visible rather than silently absorbed.
        """
        report = []
        for nu in (1.5, 1.75, 2.5):
            data = build(nu)
            with self.subTest(nu=nu):
                x = point_sets(data)["midpoints"]
                s = score(nu, x, data)
                report.append(
                    f"nu={nu:6} theta' relerr={s['deriv'][0]:.3e} at x={s['deriv'][1]:.6g} "
                    f"(x_0={bt.construction_min_x(nu):.6g})"
                )
                self.assertLess(s["deriv"][0], LOW_ORDER_DERIV_TARGET, report[-1])
        print("\n" + "\n".join(report))


class TestHighOrderAccuracy(unittest.TestCase):
    """
    ``E_theta, E_A <= 1e-6`` and ``theta'`` to 1e-6 relative at ``nu >= 20.5``. Asserted over the
    *whole* constructed region, which contains the acceptance table's
    ``[x_lo, min(x_star, max(1000, 10 nu))]``.
    """

    def test_phase_amplitude_and_derivative(self):
        report = []
        for nu in HIGH_ORDERS:
            data = build(nu)
            for label, x in point_sets(data).items():
                with self.subTest(nu=nu, points=label):
                    s = score(nu, x, data)
                    report.append(
                        f"nu={nu:8} {label:19s} E_theta={s['E_theta'][0]:.3e}@{s['E_theta'][1]:.6g} "
                        f"E_A={s['E_A'][0]:.3e}@{s['E_A'][1]:.6g} "
                        f"theta'={s['deriv'][0]:.3e}@{s['deriv'][1]:.6g}"
                    )
                    self.assertLess(s["E_theta"][0], LOOSE_BUDGET, report[-1])
                    self.assertLess(s["E_A"][0], LOOSE_BUDGET, report[-1])
                    self.assertLess(s["deriv"][0], HIGH_ORDER_DERIV_TARGET, report[-1])
        print("\n" + "\n".join(report))

    def test_against_the_cached_mpmath_corners(self):
        """
        The sharp high-order statement. Scoring against SciPy at ``nu >= 100.5`` measures the
        SciPy floor as much as this construction (campaign log 03), so the corners -- 40-digit
        ``mpmath``, committed in ``bessel_reference_data.json`` -- are what says how good the
        construction really is.
        """
        report = []
        for nu in (20.5, 100.5, 1000.5):
            data = build(nu)
            for corner in br.cached_corners(nu):
                if corner.x > data.x_star:
                    continue
                with self.subTest(nu=nu, x=corner.x):
                    sin_theta, minus_cos, A_ours, theta_prime, _ = reconstruct(
                        nu, corner.x, data
                    )
                    e_theta = max(
                        abs(float(sin_theta[0]) - corner.J / corner.amplitude),
                        abs(float(minus_cos[0]) - corner.Y / corner.amplitude),
                    )
                    e_A = abs(float(A_ours[0]) / corner.amplitude - 1.0)
                    e_d = abs(float(theta_prime[0]) / corner.theta_deriv - 1.0)
                    r_ours = float(data.r_interp(math.log(corner.x)))
                    report.append(
                        f"nu={nu:8} x={corner.x:12.7g} E_theta={e_theta:.3e} E_A={e_A:.3e} "
                        f"theta'={e_d:.3e} dr={r_ours - corner.r:+.3e}"
                    )
                    self.assertLess(e_theta, 1e-11, report[-1])
                    self.assertLess(e_A, 1e-11, report[-1])
                    self.assertLess(e_d, 1e-11, report[-1])
                    # The branch, not merely the value: r is on the tail-continuous branch, so it
                    # must match the cached r outright and not modulo 2 pi.
                    self.assertLess(abs(r_ours - corner.r), 1e-9, report[-1])
        print("\n" + "\n".join(report))


class TestBranchTracking(unittest.TestCase):
    """
    The acceptance criterion of ``README.md`` §6's high-order structural row: branch tracking is
    *verified*, not assumed.
    """

    def test_fixed_density_unwrap_fails_at_nu_1000(self):
        """
        The negative control. At a uniform 250 samples per e-fold the residual advances 3.68 rad
        per interval at ``nu = 1000.5``, above ``pi``, so ``np.unwrap`` cannot recover
        (``DRAFT-PLAN.md`` §4.5, ``RECONCILIATION.md`` C2).

        The second assertion is the one that makes the failure legible: at the *nodes* the
        mis-unwrapped representation looks perfect, because an integer ``2 pi`` error in the phase
        leaves ``J`` and ``Y`` unchanged. It is only between the nodes, where the interpolant
        crosses from a correctly unwrapped sample to an incorrectly unwrapped one, that the
        representation is order-unity wrong. No sample density repairs that, and no interpolation
        degree repairs it either.
        """
        nu = 1000.5
        x_lo = bt.construction_min_x(nu)
        x_star = bt.tail_crossover(nu, TIGHT_BUDGET, TIGHT_BUDGET).x_star
        u_lo, u_star = math.log(x_lo), math.log(x_star)

        n = int(math.ceil(250.0 * (u_star - u_lo))) + 1
        u = np.linspace(u_lo, u_star, n)
        x = np.exp(u)
        a, raw_angle, _ = nr.sample_scaled_hankel(nu, x, band=(0.99, 8.0))

        advance = float(
            np.max(np.abs(nr.residual_log_derivative(x, a))) * (u[1] - u[0])
        )
        self.assertGreater(
            advance,
            math.pi,
            f"the premise of this test is that the per-interval advance exceeds pi; "
            f"measured {advance:.4f} rad at {(n - 1) / (u_star - u_lo):.1f} samples per e-fold",
        )

        r_unwrapped = np.unwrap(raw_angle)
        spline = make_interp_spline(u, r_unwrapped, k=5)

        u_mid = 0.5 * (u[:-1] + u[1:])
        x_mid = np.exp(u_mid)
        d = br.c_nu(nu) + spline(u_mid)
        sin_theta = np.sin(x_mid) * np.cos(d) + np.cos(x_mid) * np.sin(d)
        minus_cos = -(np.cos(x_mid) * np.cos(d) - np.sin(x_mid) * np.sin(d))
        J, Y = br.reference_JY(nu, x_mid, tier=br.TIER_SCIPY)
        e_mid, i_mid = br.phase_pair_error(
            sin_theta, minus_cos, J, Y, br.reference_amplitude(J, Y)
        )

        d_nodes = br.c_nu(nu) + r_unwrapped
        sin_nodes = np.sin(x) * np.cos(d_nodes) + np.cos(x) * np.sin(d_nodes)
        minus_cos_nodes = -(np.cos(x) * np.cos(d_nodes) - np.sin(x) * np.sin(d_nodes))
        Jn, Yn = br.reference_JY(nu, x, tier=br.TIER_SCIPY)
        e_nodes, _ = br.phase_pair_error(
            sin_nodes, minus_cos_nodes, Jn, Yn, br.reference_amplitude(Jn, Yn)
        )

        print(
            f"\nnegative control: advance {advance:.4f} rad/interval, "
            f"E_theta at midpoints {e_mid:.4e} (x={x_mid[i_mid]:.6g}), "
            f"E_theta at nodes {e_nodes:.4e}"
        )
        self.assertGreater(
            e_mid,
            0.1,
            f"a fixed-density np.unwrap is expected to fail with an order-unity phase-pair "
            f"error; measured {e_mid:.4e}",
        )
        self.assertLess(
            e_nodes,
            1e-6,
            f"the failure is invisible at the nodes themselves; measured {e_nodes:.4e}",
        )

    def test_the_shipped_tracker_succeeds_at_nu_1000(self):
        """The positive result: ``E_theta <= 1e-6`` and 90 +/- 1 branch crossings resolved."""
        data = build(1000.5)
        x = np.concatenate([point_sets(data)["nodes"], point_sets(data)["midpoints"]])
        s = score(1000.5, x, data)
        print(
            f"\nshipped tracker at nu=1000.5: E_theta={s['E_theta'][0]:.4e} at "
            f"x={s['E_theta'][1]:.6g}, wraps_tracked={data.wraps_tracked}, "
            f"max branch advance={data.max_branch_advance:.4f} rad "
            f"(limit {nr.BRANCH_SAFETY_FRACTION * math.pi:.4f}), "
            f"max snap={data.max_branch_snap:.3e} rad"
        )
        self.assertLess(s["E_theta"][0], LOOSE_BUDGET)
        self.assertGreaterEqual(data.wraps_tracked, 89)
        self.assertLessEqual(data.wraps_tracked, 91)

    def test_the_residual_is_on_the_tail_continuous_branch(self):
        """
        ``r(x_0) = 570.820039`` at ``nu = 1000.5`` (``RECONCILIATION.md`` C2), not that value
        folded into ``(-pi, pi]``. A fold is the easiest way to poison every downstream
        comparison, so it gets its own assertion rather than being left implicit in the corner
        checks.
        """
        expected = {
            1.5: 0.615480,
            1.75: 0.757566,
            2.5: 1.183200,
            20.5: 11.443651,
            100.5: 57.104238,
            1000.5: 570.820039,
        }
        for nu, r0 in expected.items():
            with self.subTest(nu=nu):
                data = build(nu)
                self.assertAlmostEqual(float(data.r_nodes[0]), r0, places=5)

    def test_every_node_gap_meets_the_branch_safety_criterion(self):
        """``max|dr/d log x| * h <= BRANCH_SAFETY_FRACTION * pi`` across the whole region."""
        for nu in LOW_ORDERS + HIGH_ORDERS:
            with self.subTest(nu=nu):
                data = build(nu)
                self.assertLessEqual(
                    data.max_branch_advance,
                    nr.BRANCH_SAFETY_FRACTION * math.pi,
                    f"nu={nu}: max advance {data.max_branch_advance:.4f} rad",
                )
                self.assertLessEqual(
                    data.max_branch_snap, nr.BRANCH_CONSISTENCY_FRACTION * math.pi
                )


class TestTwoSidedAdaptivity(unittest.TestCase):
    """
    Adaptivity has to work in both directions (``DRAFT-PLAN.md`` §4.5). A construction that only
    ever refines passes every accuracy test above and still fails the design, so this is the test
    that catches it.
    """

    def test_the_tail_is_coarser_than_the_turning_point(self):
        report = []
        for nu in (100.5, 1000.5):
            with self.subTest(nu=nu):
                data = build(nu)
                u = data.log_x_nodes
                gaps = np.diff(u)
                mid = 0.5 * (u[:-1] + u[1:])

                top = gaps[mid > u[-1] - math.log(10.0)]
                bottom = gaps[mid < u[0] + math.log(10.0)]
                factor = float(np.median(top) / np.median(bottom))
                report.append(
                    f"nu={nu:8}: top decade {1.0 / float(np.median(top)):.1f} nodes/e-fold, "
                    f"turning-point decade {1.0 / float(np.median(bottom)):.1f} nodes/e-fold, "
                    f"coarsening factor {factor:.2f}; the top decade is "
                    f"{250.0 * float(np.median(top)):.2f}x coarser than the prototype's "
                    f"250 per e-fold"
                )
                self.assertGreater(factor, 2.0, report[-1])
                self.assertLess(
                    1.0 / float(np.median(top)),
                    250.0,
                    "the tail must be coarser than the prototype's uniform 250 per e-fold, "
                    "not merely coarser than the turning point: " + report[-1],
                )
        print("\n" + "\n".join(report))


class TestDerivativeRoutes(unittest.TestCase):
    """
    With ``theta'`` derived from ``ell``, the Wronskian check ``a^2 theta' = 1`` is satisfied by
    construction and checks nothing. So the required independent checks are ``exp(-2 ell)``
    against ``1 + r_u/x`` -- two different interpolants -- and both against the reference
    (``README.md`` §2 (f), ``DRAFT-PLAN.md`` §7.3). Required, not optional.
    """

    def test_the_two_routes_agree_and_both_match_the_reference(self):
        report = []
        for nu, budget, target in (
            (1.5, TIGHT_BUDGET, LOW_ORDER_DERIV_TARGET),
            (1.75, TIGHT_BUDGET, LOW_ORDER_DERIV_TARGET),
            (2.5, TIGHT_BUDGET, LOW_ORDER_DERIV_TARGET),
            (20.5, TIGHT_BUDGET, HIGH_ORDER_DERIV_TARGET),
            (100.5, TIGHT_BUDGET, HIGH_ORDER_DERIV_TARGET),
            (1000.5, TIGHT_BUDGET, HIGH_ORDER_DERIV_TARGET),
        ):
            with self.subTest(nu=nu):
                data = build(nu, budget)
                x = point_sets(data)["midpoints"]
                s = score(nu, x, data)
                report.append(
                    f"nu={nu:8} exp(-2 ell)={s['deriv'][0]:.3e}@{s['deriv'][1]:.6g}  "
                    f"1+r_u/x={s['deriv_alt'][0]:.3e}@{s['deriv_alt'][1]:.6g}  "
                    f"routes agree to {s['routes'][0]:.3e}@{s['routes'][1]:.6g}"
                )
                self.assertLess(s["deriv"][0], target, report[-1])
                self.assertLess(s["deriv_alt"][0], target, report[-1])
                self.assertLess(s["routes"][0], target, report[-1])
        print("\n" + "\n".join(report))

    def test_the_wronskian_check_really_is_a_tautology(self):
        """
        Documenting *why* the test above is required: ``a^2 theta' - 1`` is zero to rounding by
        construction, so it detects nothing and cannot substitute for the two-interpolant check.
        """
        data = build(100.5)
        u = 0.5 * (data.log_x_nodes[:-1] + data.log_x_nodes[1:])
        ell = np.atleast_1d(data.log_a_interp(u))
        residual = float(np.max(np.abs(np.exp(2.0 * ell) * np.exp(-2.0 * ell) - 1.0)))
        self.assertLess(residual, 1e-15)


class TestDegenerateHalfOrder(unittest.TestCase):
    """
    ``nu = 1/2`` is exactly degenerate: ``mu - 1 = 0`` gives ``r == 0`` and ``a == 1`` identically
    at every ``x``, ``tail_crossover`` returns ``x_star = x_0``, and the sampler must not run.
    """

    def test_the_near_region_is_empty_and_exact(self):
        nu = 0.5
        x_lo = bt.construction_min_x(nu)
        crossover = bt.tail_crossover(nu, TIGHT_BUDGET, TIGHT_BUDGET)
        self.assertEqual(crossover.x_star, x_lo)

        data = build(nu)
        self.assertTrue(data.degenerate)
        self.assertEqual(data.n_panels, 0)
        self.assertEqual(data.n_nodes, 1)
        self.assertEqual(data.wraps_tracked, 0)
        self.assertTrue(data.converged)

        self.assertAlmostEqual(float(data.r_nodes[0]), 0.0, delta=1e-15)
        self.assertAlmostEqual(float(data.a_nodes[0]), 1.0, delta=1e-15)

        u = math.log(x_lo)
        self.assertAlmostEqual(float(data.r_interp(u)), 0.0, delta=1e-15)
        self.assertAlmostEqual(float(data.log_a_interp(u)), 0.0, delta=1e-15)
        self.assertAlmostEqual(float(data.r_interp.derivative()(u)), 0.0, delta=1e-15)


class TestInterpolantStructure(unittest.TestCase):
    """
    ``DRAFT-PLAN.md`` §7.3 item 4: validate continuity and derivative behaviour across refinement
    and interpolation boundaries.
    """

    def test_continuity_at_panel_boundaries(self):
        report = []
        for nu in (2.5, 100.5, 1000.5):
            with self.subTest(nu=nu):
                data = build(nu)
                edges = data.panel_edges[1:-1]
                if edges.size == 0:
                    continue

                dr = data.r_interp.derivative()
                r_jump = l_jump = dr_jump = 0.0
                for k, edge in enumerate(edges):
                    r_jump = max(
                        r_jump,
                        abs(
                            data.r_interp.on_panel(k, edge)
                            - data.r_interp.on_panel(k + 1, edge)
                        ),
                    )
                    l_jump = max(
                        l_jump,
                        abs(
                            data.log_a_interp.on_panel(k, edge)
                            - data.log_a_interp.on_panel(k + 1, edge)
                        ),
                    )
                    dr_jump = max(
                        dr_jump, abs(dr.on_panel(k, edge) - dr.on_panel(k + 1, edge))
                    )
                report.append(
                    f"nu={nu:8}: {edges.size} interior panel edges, r jump {r_jump:.3e}, "
                    f"ell jump {l_jump:.3e}, dr/du jump {dr_jump:.3e}"
                )
                # C0 is exact by construction, up to the conditioning of the coefficient solve:
                # both panels carry a Lobatto node at the shared edge and interpolate the same
                # sampled value there. C1 is not imposed, and the derivative jump is reported
                # rather than asserted -- it is bounded by the fit accuracy, not by design.
                self.assertLess(r_jump, data.phase_atol, report[-1])
                self.assertLess(l_jump, data.amplitude_rtol, report[-1])
        print("\n" + "\n".join(report))

    def test_it_refuses_to_evaluate_above_the_crossover(self):
        """
        The near region ends at ``x_star``; above it the closed-form tail is the representation.
        Silently extrapolating a Chebyshev panel there would be a wrong answer with no symptom.
        """
        data = build(2.5)
        with self.assertRaises(ValueError):
            data.r_interp(math.log(data.x_star) + 1.0)
        with self.assertRaises(ValueError):
            data.log_a_interp(math.log(data.x_lo) - 1.0)

    def test_the_panel_degree_floor_is_enforced(self):
        """
        The coefficient-tail estimate sums the last three Chebyshev coefficients, which below
        degree 8 is a large fraction of the function rather than of its truncation error, so the
        criterion never clears however small the panel becomes. Left unguarded, degree 4 at
        ``nu = 100.5`` runs to the 8192-panel cap with `converged = False` while its interior
        residual is already 1.3e-13. Odd degrees are rejected too: an even Lobatto grid carries a
        node at the panel midpoint, which is what keeps the interior test points off the nodes.
        """
        x_lo = bt.construction_min_x(2.5)
        x_star = bt.tail_crossover(2.5, TIGHT_BUDGET, TIGHT_BUDGET).x_star
        for degree in (6, 7, 9, 2.0):
            with self.subTest(degree=degree):
                with self.assertRaises(nr.NearRegionError):
                    nr.build_near_region(
                        2.5, x_lo, x_star, TIGHT_BUDGET, TIGHT_BUDGET, degree=degree
                    )

        accepted = nr.build_near_region(
            2.5, x_lo, x_star, TIGHT_BUDGET, TIGHT_BUDGET, degree=nr.MIN_PANEL_DEGREE
        )
        self.assertTrue(accepted.converged)
        self.assertEqual(accepted.interp_degree, nr.MIN_PANEL_DEGREE)


class TestRefinementCap(unittest.TestCase):
    """
    A refinement cap is mandatory, and on hitting it the construction must report unmet accuracy
    rather than silently accept it (``DRAFT-PLAN.md`` §9 Stage 2). Whether to raise is prompt 05's
    decision, so this module returns the measured errors either way.
    """

    def test_it_reports_rather_than_pretends(self):
        nu = 1000.5
        x_lo = bt.construction_min_x(nu)
        x_star = bt.tail_crossover(nu, TIGHT_BUDGET, TIGHT_BUDGET).x_star
        capped = nr.build_near_region(
            nu, x_lo, x_star, TIGHT_BUDGET, TIGHT_BUDGET, max_refinement_passes=1
        )
        self.assertFalse(capped.converged)
        self.assertGreater(capped.achieved_phase_abserr, TIGHT_BUDGET)
        print(
            f"\nrefinement cap at 1 pass, nu=1000.5: converged={capped.converged}, "
            f"achieved phase abserr={capped.achieved_phase_abserr:.3e} against a budget of "
            f"{TIGHT_BUDGET:.0e}, {capped.n_panels} panels"
        )

        full = build(nu)
        self.assertTrue(full.converged)
        self.assertLess(full.achieved_phase_abserr, TIGHT_BUDGET)

    def test_the_achieved_estimates_bracket_the_measured_errors(self):
        """
        The ``achieved_*`` numbers are practical estimators, not supremum bounds. What matters for
        prompt 05, which propagates them into ``theta_abserr``, is that they do not *under*-report
        the construction's own interpolation error. They are compared here against the error
        measured at the cached 40-digit corners, which is the only comparison in which the
        reference is better than the construction at every order.
        """
        report = []
        for nu in (2.5, 20.5, 100.5, 1000.5):
            with self.subTest(nu=nu):
                data = build(nu)
                worst = 0.0
                for corner in br.cached_corners(nu):
                    if corner.x > data.x_star:
                        continue
                    r_ours = float(data.r_interp(math.log(corner.x)))
                    worst = max(worst, abs(r_ours - corner.r))
                report.append(
                    f"nu={nu:8}: achieved_phase_abserr={data.achieved_phase_abserr:.3e}, "
                    f"worst |dr| at the cached corners={worst:.3e}, "
                    f"ratio={data.achieved_phase_abserr / max(worst, 1e-300):.2f}"
                )
                self.assertGreater(
                    data.achieved_phase_abserr,
                    0.1 * worst,
                    "the estimator must not under-report the measured error by an order: "
                    + report[-1],
                )
        print("\n" + "\n".join(report))


class TestCost(unittest.TestCase):
    """
    Node count and build time per order, recorded rather than bounded. ``nu = 1000.5`` is the
    expensive case; if it ever exceeded ~30 s it would make prompt 08's high-order tests painful,
    so that one gets an assertion.
    """

    def test_record_cost_per_order(self):
        report = []
        for nu in LOW_ORDERS + HIGH_ORDERS:
            x_lo = bt.construction_min_x(nu)
            crossover = bt.tail_crossover(nu, TIGHT_BUDGET, TIGHT_BUDGET)
            start = time.perf_counter()
            data = nr.build_near_region(
                nu, x_lo, crossover.x_star, TIGHT_BUDGET, TIGHT_BUDGET
            )
            elapsed = time.perf_counter() - start
            report.append(
                f"nu={nu:8} x_star={crossover.x_star:11.6g} nodes={data.n_nodes:5d} "
                f"panels={data.n_panels:4d} passes={data.refinement_passes:2d} "
                f"wraps={data.wraps_tracked:4d} build={elapsed:.3f} s"
            )
            if nu == 1000.5:
                self.assertLess(elapsed, 30.0, report[-1])
        print("\n" + "\n".join(report))


class TestModuleHygiene(unittest.TestCase):
    """Constraints on the shipped module that are cheaper to assert than to re-read."""

    #: Names the shipped module must not *use*. Checked through the AST rather than by a text
    #: search, because several of them are named in the docstrings -- which is the point: the
    #: module explains why it does not use them.
    FORBIDDEN = (
        "unwrap",
        "simple_mod_2pi",
        "root_scalar",
        "phase_spline",
        "bessel_phase",
    )

    def _identifiers(self):
        tree = ast.parse(inspect.getsource(nr))
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    used.update(alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                used.update((node.module or "").split("."))
                for alias in node.names:
                    used.add(alias.name)
        return used

    def test_no_forbidden_machinery_in_the_tracker(self):
        """
        No ``np.unwrap``, no ``simple_mod_2pi``, no ``phase_spline`` and no root solve. A
        ``np.unwrap`` in *this* module's negative control is expected and correct; the shipped
        tracker has none.
        """
        used = self._identifiers()
        for forbidden in self.FORBIDDEN:
            self.assertNotIn(
                forbidden, used, f"bessel_near_region must not use {forbidden}"
            )

    def test_it_does_not_import_bessel_phase(self):
        """The module boundary of ``DRAFT-PLAN.md`` §7.1: the sampler knows nothing downstream."""
        tree = ast.parse(inspect.getsource(nr))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        self.assertNotIn("LiouvilleGreen.bessel_phase", imported)
        self.assertFalse(
            [name for name in imported if name.endswith("bessel_phase")], imported
        )

    def test_the_docstring_does_not_repeat_the_corrected_claim(self):
        """
        "The residual never exceeds a cycle" is false above ``nu ~ 630``
        (``RECONCILIATION.md`` C2) and must not enter the codebase; the correct argument, that
        ``eps |r|_max ~ 1.3e-13``, must be the one that is there.
        """
        doc = nr.__doc__
        self.assertNotIn("never exceeds a cycle", doc)
        self.assertIn("1.3e-13", doc)
        self.assertIn("570.82", doc)


if __name__ == "__main__":
    unittest.main()
