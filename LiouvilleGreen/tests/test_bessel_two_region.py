"""
Acceptance tests for the two-region Bessel amplitude and phase construction.

Everything here is scored against ``LiouvilleGreen.tests.bessel_reference``, never against the
object under test and never against a reference derived from it. The error metrics are the
phase-pair definitions of ``DRAFT-PLAN.md`` §6.1, fixed by prompt 01 and used unchanged:

    E_theta = max_x max(|sin theta_ours - J/A|, |-cos theta_ours - Y/A|),
    E_A     = max_x |A_ours/A - 1|,        A = hypot(J, Y) from the reference.

A phase-pair error rather than an unwrapped phase difference: it measures the local effect on the
Bessel values, normalized by their envelope, without dividing by a function that passes through
zero.

Reference tiers. SciPy ``jv``/``yv`` are used where they are a reference and nowhere else:
``bessel_reference.scipy_reference`` refuses above ``scipy_reference_max_x(nu)`` -- 7.13e8 for
``nu > 85.5`` and 2e15 otherwise -- and the cached 40-digit ``mpmath`` corners or a live ``mpmath``
call are used past that, and at the large arguments of :meth:`TestSplitEvaluation`. Note also the
*precision* floor recorded as standing note 15 of the campaign board: above ``nu = 20.5``, SciPy is
not adequate below about 1e-12 even well inside its validity boundary, so the high-order thresholds
here are set at the campaign's 1e-6 target and the corner-scored values are reported separately.
"""

import math
import time
import unittest
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, yv

from LiouvilleGreen import bessel_tail
from LiouvilleGreen.bessel_phase import (
    BesselPhaseAccuracyError,
    BesselPhaseError,
    DEFAULT_AMPLITUDE_RTOL,
    DEFAULT_PHASE_ATOL,
    MAX_SUPPORTED_X,
    MIN_SUPPORTED_NU,
    XSplineWrapper,
    bessel_phase,
    c_nu,
    c_nu_reduced,
)
from LiouvilleGreen.tests import bessel_reference as br

LOW_ORDERS = (0.5, 1.5, 1.75, 2.5)
HIGH_ORDERS = (20.5, 100.5, 1000.5)

#: Acceptance targets, campaign ``README.md`` §6.
LOW_ORDER_TARGET = 1.0e-11
LOW_ORDER_DERIV_TARGET = 1.0e-9
HIGH_ORDER_TARGET = 1.0e-6

_BUILD_CACHE = {}


def build(nu: float, max_x: float, **kwargs):
    """Build once per (nu, max_x) and share; the objects are immutable in practice."""
    key = (nu, max_x, tuple(sorted(kwargs.items())))
    if key not in _BUILD_CACHE:
        _BUILD_CACHE[key] = bessel_phase(nu, max_x, **kwargs)
    return _BUILD_CACHE[key]


def point_sets(data):
    """
    The three point sets ``DRAFT-PLAN.md`` §9 Stage 1 requires kept separate, plus the tail.

    * ``"nodes"`` -- the near-region sample nodes themselves, where an interpolant is exact by
      construction and only the sampling floor shows;
    * ``"midpoints"`` -- geometric midpoints of consecutive nodes, where interpolation error peaks;
    * ``"endpoints"`` -- dense points inside the first and last node gaps, the intervals a interior
      sweep silently omits and where ``DRAFT-PLAN.md`` §4.7 locates every derivative maximum;
    * ``"tail"`` -- a logarithmic sweep of the closed-form region, plus its two endpoints.
    """
    min_x = data["min_x"]
    max_x = data["max_x"]
    x_star = data["x_star"]
    near = data["near_region"]

    sets = {}

    if near is not None and near.n_nodes > 1:
        u = np.asarray(near.log_x_nodes, dtype=float)
        nodes = np.exp(u)
        nodes = nodes[(nodes >= min_x) & (nodes <= max_x)]
        sets["nodes"] = nodes

        mid = np.exp(0.5 * (u[:-1] + u[1:]))
        mid = mid[(mid >= min_x) & (mid <= max_x)]
        sets["midpoints"] = mid

        first = np.exp(np.linspace(u[0], u[1], 41)[1:-1])
        last = np.exp(np.linspace(u[-2], u[-1], 41)[1:-1])
        ends = np.concatenate([first, last])
        ends = ends[(ends >= min_x) & (ends <= max_x)]
        sets["endpoints"] = ends

    if max_x > x_star:
        lo = max(x_star, min_x)
        tail = np.geomspace(lo, max_x, 400)
        sets["tail"] = tail
    elif near is None:
        sets["tail"] = np.geomspace(min_x, max_x, 400)

    return sets


def score(nu: float, xs, data, tier: str = br.TIER_SCIPY):
    """
    Return ``{"E_theta": (value, x), "E_A": (value, x), "E_deriv": (value, x)}`` on ``xs``.

    ``E_deriv`` is the relative error of ``theta'`` against the reference identity
    ``theta' = (2/pi)/(x (J^2 + Y^2))``.
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    J, Y = br.reference_JY(nu, xs, tier=tier)
    J = np.atleast_1d(np.asarray(J, dtype=float))
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    amplitude = br.reference_amplitude(J, Y)

    phase = data["phase"]
    mod = data["mod"]

    sin_theta, cos_theta = phase.sin_cos_theta(xs)
    e_theta = np.maximum(
        np.fabs(sin_theta - J / amplitude), np.fabs(-cos_theta - Y / amplitude)
    )
    e_amplitude = np.fabs(mod(xs) / amplitude - 1.0)

    reference_deriv = br.reference_theta_deriv(xs, J, Y)
    e_deriv = np.fabs(phase.theta_deriv(xs) / reference_deriv - 1.0)

    out = {}
    for label, values in (
        ("E_theta", e_theta),
        ("E_A", e_amplitude),
        ("E_deriv", e_deriv),
    ):
        index = int(np.argmax(values))
        out[label] = (float(values[index]), float(xs[index]))
    return out


class TestAcceptanceTable(unittest.TestCase):
    """The acceptance table of campaign ``README.md`` §6, in full."""

    def _check(self, nu, max_x, target, deriv_target, tier=br.TIER_SCIPY):
        data = build(nu, max_x)
        worst = {}
        for label, xs in point_sets(data).items():
            if xs.size == 0:
                continue
            result = score(nu, xs, data, tier=tier)
            print(
                f"nu={nu:>8} {label:>10}: E_theta={result['E_theta'][0]:.3e}"
                f"@{result['E_theta'][1]:.6g} E_A={result['E_A'][0]:.3e}"
                f"@{result['E_A'][1]:.6g} E_deriv={result['E_deriv'][0]:.3e}"
                f"@{result['E_deriv'][1]:.6g}"
            )
            for key, (value, x) in result.items():
                limit = deriv_target if key == "E_deriv" else target
                self.assertLessEqual(
                    value,
                    limit,
                    f"{key} = {value:.4g} at nu={nu}, x={x:.8g} ({label}) exceeds {limit:.1g}",
                )
                if value > worst.get(key, (0.0, 0.0))[0]:
                    worst[key] = (value, x)
        return worst

    def test_low_orders_through_1e7(self):
        """
        Orders 1/2, 3/2, 7/4, 5/2 -- the orders production builds -- through ``x_max = 1e7``, at
        ``E_theta, E_A <= 1e-11`` and ``theta'`` relative error ``<= 1e-9``.
        """
        for nu in LOW_ORDERS:
            self._check(nu, 1.0e7, LOW_ORDER_TARGET, LOW_ORDER_DERIV_TARGET)

    def test_high_orders(self):
        """
        Orders 20.5, 100.5, 1000.5 from the domain lower bound through ``max(1000, 10 nu)``, at
        1e-6 in all three quantities. Accuracy is not the objective at these orders; correctness is
        -- the structural requirements are checked in :class:`TestStructure`.

        ``nu = 1000.5`` is the slowest case in this module; prompt 08 inherits it.
        """
        for nu in HIGH_ORDERS:
            self._check(
                nu, max(1000.0, 10.0 * nu), HIGH_ORDER_TARGET, HIGH_ORDER_TARGET
            )

    def test_declared_accuracy_bounds_the_corner_measured_error(self):
        """
        The object's own ``theta_abserr`` and ``amplitude_relerr`` must not under-report.

        This is what prompt 06 will hand to ``AdaptiveLevin`` as a declared phase error, whose
        entire purpose is that the caller sees an honest number; a declared value below the true
        error would be worse than declaring nothing. Scored against the committed 40-digit
        ``mpmath`` corners, which are independent of SciPy at every order.
        """
        for nu in LOW_ORDERS + HIGH_ORDERS:
            max_x = 1.0e7 if nu in LOW_ORDERS else max(1000.0, 10.0 * nu)
            data = build(nu, max_x)
            phase = data["phase"]
            mod = data["mod"]

            worst_theta = 0.0
            worst_amplitude = 0.0
            for corner in br.cached_corners(nu):
                if not (data["min_x"] <= corner.x <= data["max_x"]):
                    continue
                sin_theta, cos_theta = phase.sin_cos_theta(corner.x)
                worst_theta = max(
                    worst_theta,
                    abs(sin_theta - corner.J / corner.amplitude),
                    abs(-cos_theta - corner.Y / corner.amplitude),
                )
                worst_amplitude = max(
                    worst_amplitude, abs(mod(corner.x) / corner.amplitude - 1.0)
                )

            print(
                f"nu={nu:>8}: corner E_theta={worst_theta:.3e} declared "
                f"theta_abserr={data['theta_abserr']:.3e}; corner E_A={worst_amplitude:.3e} "
                f"declared amplitude_relerr={data['amplitude_relerr']:.3e}"
            )
            self.assertLessEqual(worst_theta, data["theta_abserr"])
            self.assertLessEqual(worst_amplitude, data["amplitude_relerr"])


class TestDegenerateOrder(unittest.TestCase):
    def test_nu_one_half_is_exact_and_all_tail(self):
        """
        ``nu = 1/2`` is a first-class case, not a degenerate accident.

        ``mu - 1 = 4 nu^2 - 1 = 0`` makes every coefficient of the series vanish, so ``r == 0`` and
        ``a == 1`` identically at *every* ``x``: the "asymptotic" tail is exact, the crossover is
        the bottom of the domain, and the near-region sampler must never run.
        """
        data = build(0.5, 1.0e7)

        self.assertIsNone(
            data["near_region"], "the near-region sampler ran at nu = 1/2"
        )
        self.assertEqual(data["x_star"], data["min_x"])
        self.assertEqual(data["accuracy"]["sampled_phase_floor"], 0.0)

        phase = data["phase"]
        xs = np.geomspace(data["min_x"], data["max_x"], 500)

        self.assertTrue(np.all(phase.residual(xs) == 0.0))
        self.assertTrue(np.all(np.asarray(data["mod"].a(xs)) == 1.0))
        self.assertEqual(c_nu(0.5), 0.0)

        J, Y = br.reference_JY(0.5, xs, tier=br.TIER_EXACT)
        amplitude = br.reference_amplitude(J, Y)
        sin_theta, cos_theta = phase.sin_cos_theta(xs)
        e_theta = float(
            np.max(
                np.maximum(
                    np.fabs(sin_theta - J / amplitude),
                    np.fabs(-cos_theta - Y / amplitude),
                )
            )
        )
        e_amplitude = float(np.max(np.fabs(data["mod"](xs) / amplitude - 1.0)))
        print(
            f"nu=0.5 against the exact tier: E_theta={e_theta:.3e} E_A={e_amplitude:.3e}"
        )
        self.assertLessEqual(e_theta, 1.0e-14)
        self.assertLessEqual(e_amplitude, 1.0e-14)


class TestCrossover(unittest.TestCase):
    """
    The second half of the remainder test of ``DRAFT-PLAN.md`` §7.2.

    Prompt 03 could only test the series against its own first omitted term. This is the test of
    the *sampled* region against the series at the same point, and of the derivative's continuity
    across a seam that is deliberately not blended.
    """

    def test_regions_agree_at_the_crossover(self):
        for nu in (1.5, 1.75, 2.5, 20.5, 100.5, 1000.5):
            crossover = bessel_tail.tail_crossover(
                nu, DEFAULT_PHASE_ATOL, DEFAULT_AMPLITUDE_RTOL
            )
            data = build(nu, 10.0 * crossover.x_star)
            near = data["near_region"]
            self.assertIsNotNone(near)

            x_star = data["x_star"]
            u_star = math.log(x_star)
            near_r = float(near.r_interp(u_star))
            near_a = math.exp(float(near.log_a_interp(u_star)))
            tail_r = float(bessel_tail.tail_residual(nu, x_star))
            tail_a = float(bessel_tail.tail_amplitude(nu, x_star))

            measured_phase = abs(near_r - tail_r)
            measured_amplitude = abs(near_a / tail_a - 1.0)
            print(
                f"nu={nu:>8} x_star={x_star:.6g}: phase agreement {measured_phase:.3e} "
                f"(budget {DEFAULT_PHASE_ATOL:.1g}), amplitude agreement "
                f"{measured_amplitude:.3e} (budget {DEFAULT_AMPLITUDE_RTOL:.1g})"
            )

            self.assertLessEqual(measured_phase, DEFAULT_PHASE_ATOL)
            self.assertLessEqual(measured_amplitude, DEFAULT_AMPLITUDE_RTOL)

            # the stored agreement is what the test measures, not an independent claim
            self.assertAlmostEqual(
                data["accuracy"]["crossover_phase_agreement"], measured_phase, delta=0.0
            )
            self.assertAlmostEqual(
                data["accuracy"]["crossover_amplitude_agreement"],
                measured_amplitude,
                delta=0.0,
            )

    def test_the_seam_is_continuous(self):
        """
        ``r``, ``a`` and ``theta'`` are continuous across ``x_star`` to the budget. Nothing is
        blended: a seam that had to be hidden by tapering would be a construction that failed the
        test above, and blending would also destroy the ``C^1`` behaviour that the Levin
        quadrature's subdivision reads through ``theta_deriv``.
        """
        for nu in (1.5, 2.5, 20.5, 100.5, 1000.5):
            crossover = bessel_tail.tail_crossover(
                nu, DEFAULT_PHASE_ATOL, DEFAULT_AMPLITUDE_RTOL
            )
            data = build(nu, 10.0 * crossover.x_star)
            phase = data["phase"]
            mod = data["mod"]
            x_star = data["x_star"]

            below = x_star * (1.0 - 1.0e-9)
            above = x_star * (1.0 + 1.0e-9)
            delta_u = math.log(above) - math.log(below)

            # r and a genuinely *vary* between the two test points, at a rate the representation
            # supplies for itself: dr/du = x(a^-2 - 1) and d ell/du. What a seam defect would look
            # like is a change in excess of that variation, so the variation is subtracted. Not
            # doing so would measure 5.6e-11 at nu = 3/2 -- the true change in r across 2e-9 in
            # log x -- and call a correct construction discontinuous.
            r_u = phase.residual_log_deriv(x_star)
            ell_u = x_star * mod.log_deriv(x_star) + 0.5

            jump_r = abs(
                (phase.residual(above) - phase.residual(below)) - r_u * delta_u
            )
            jump_a = abs((mod.a(above) / mod.a(below) - 1.0) - ell_u * delta_u)
            jump_deriv = abs(
                (phase.theta_deriv(above) / phase.theta_deriv(below) - 1.0)
                + 2.0 * ell_u * delta_u
            )
            print(
                f"nu={nu:>8} seam jumps: r {jump_r:.3e}, a {jump_a:.3e}, theta' "
                f"{jump_deriv:.3e}"
            )
            self.assertLessEqual(jump_r, DEFAULT_PHASE_ATOL)
            self.assertLessEqual(jump_a, DEFAULT_AMPLITUDE_RTOL)
            self.assertLessEqual(jump_deriv, 2.0 * DEFAULT_AMPLITUDE_RTOL)


class TestSplitEvaluation(unittest.TestCase):
    """
    ``sin theta`` by angle addition on ``theta = x + d``, against 70-digit ``mpmath`` at the
    *supplied* floating-point argument (``DRAFT-PLAN.md`` §7.5: accuracy is defined at the supplied
    ``x``, so the reference is built from ``mpf(float(x))`` and never from a re-derived argument).
    """

    X_VALUES = (1.0e3, 1.0e7, 1.0e12, 1.0e15)

    def test_split_matches_mpmath(self):
        for nu in (1.5, 1.75):
            data = build(nu, 2.0e15)
            phase = data["phase"]
            mod = data["mod"]
            for x in self.X_VALUES:
                J, Y = br.mpmath_reference(nu, x, dps=70)
                amplitude = math.hypot(J, Y)
                sin_theta, cos_theta = phase.sin_cos_theta(x)
                e_theta = max(
                    abs(sin_theta - J / amplitude), abs(-cos_theta - Y / amplitude)
                )
                e_amplitude = abs(mod(x) / amplitude - 1.0)
                print(
                    f"nu={nu} x={x:.3g}: split E_theta={e_theta:.3e} E_A={e_amplitude:.3e}"
                )
                self.assertLessEqual(e_theta, 1.0e-14)
                self.assertLessEqual(e_amplitude, 1.0e-14)

    def test_split_beats_the_naive_sum(self):
        """
        The test that keeps the design decision from being quietly undone.

        Forming ``x + d`` first rounds the correction against the leading term: at ``x = 1e15`` the
        ulp of the sum is 0.125 rad. ``RECONCILIATION.md`` §1 measures the naive route at 3.5e-14,
        1.2e-10, 2.7e-6 and 4.7e-2 at the four arguments below, against 1.1e-16 or better for angle
        addition.
        """
        minimum_naive = {1.0e3: 1.0e-15, 1.0e7: 1.0e-11, 1.0e12: 1.0e-7, 1.0e15: 1.0e-3}
        for nu in (1.5, 1.75):
            data = build(nu, 2.0e15)
            phase = data["phase"]
            for x in self.X_VALUES:
                J, Y = br.mpmath_reference(nu, x, dps=70)
                amplitude = math.hypot(J, Y)

                sin_theta, _ = phase.sin_cos_theta(x)
                split_error = abs(sin_theta - J / amplitude)

                naive_theta = x + phase.c_nu + phase.residual(x)
                naive_error = abs(math.sin(naive_theta) - J / amplitude)

                print(
                    f"nu={nu} x={x:.3g}: naive {naive_error:.3e} vs split "
                    f"{split_error:.3e}"
                )
                self.assertLessEqual(split_error, 1.0e-14)
                self.assertGreaterEqual(
                    naive_error,
                    minimum_naive[x],
                    f"the naive route is unexpectedly accurate at nu={nu}, x={x:.3g}; "
                    f"has the split been undone, or the reference changed?",
                )

    def test_bounded_angle_uses_atan2_of_the_split_pair(self):
        """
        ``theta_mod_2pi`` must carry the split's accuracy, not that of a double-precision
        reduction of the raw phase against a 53-bit ``2 pi``.
        """
        for nu in (1.5, 1.75):
            data = build(nu, 2.0e15)
            phase = data["phase"]
            for x in self.X_VALUES:
                bounded = phase.theta_mod_2pi(x)
                self.assertLessEqual(abs(bounded), math.pi + 1.0e-12)
                sin_theta, cos_theta = phase.sin_cos_theta(x)
                self.assertLessEqual(abs(math.sin(bounded) - sin_theta), 1.0e-15)
                self.assertLessEqual(abs(math.cos(bounded) - cos_theta), 1.0e-15)


class TestDerivativeRoutes(unittest.TestCase):
    def test_two_routes_agree_with_each_other_and_the_reference(self):
        """
        ``README.md`` §2 (f), mandatory rather than optional.

        The shipped derivative is ``theta' = a^-2 = exp(-2 ell)``, read off the amplitude
        interpolant as a value. That makes the Wronskian identity ``a^2 theta' = 1`` **a tautology**
        here -- it is satisfied by construction and checks nothing -- so the two independent checks
        are (i) ``exp(-2 ell)`` against ``1 + r_u/x``, which uses the *other* interpolant and its
        derivative, and (ii) both against ``(2/pi)/(x (J^2 + Y^2))`` from the reference functions.
        """
        for nu in LOW_ORDERS + HIGH_ORDERS:
            max_x = 1.0e7 if nu in LOW_ORDERS else max(1000.0, 10.0 * nu)
            target = LOW_ORDER_DERIV_TARGET if nu in LOW_ORDERS else HIGH_ORDER_TARGET
            data = build(nu, max_x)
            phase = data["phase"]

            sets = point_sets(data)
            xs = np.concatenate([v for v in sets.values() if v.size])

            shipped = np.asarray(phase.theta_deriv(xs), dtype=float)
            alternative = np.asarray(phase.theta_deriv_from_residual(xs), dtype=float)
            J, Y = br.reference_JY(nu, xs, tier=br.TIER_SCIPY)
            reference = br.reference_theta_deriv(xs, J, Y)

            between = np.max(np.fabs(shipped / alternative - 1.0))
            shipped_vs_reference = np.max(np.fabs(shipped / reference - 1.0))
            alt_vs_reference = np.max(np.fabs(alternative / reference - 1.0))
            print(
                f"nu={nu:>8}: exp(-2 ell) vs 1+r_u/x {between:.3e}; vs reference "
                f"{shipped_vs_reference:.3e}; alternative vs reference "
                f"{alt_vs_reference:.3e}"
            )
            self.assertLessEqual(between, target)
            self.assertLessEqual(shipped_vs_reference, target)
            self.assertLessEqual(alt_vs_reference, target)


class TestStructure(unittest.TestCase):
    def test_phi_is_identically_zero(self):
        """
        ``DRAFT-PLAN.md`` §4.3: for every ``nu > 1/2`` the old match point *was* the initial node,
        where the phase had already been fixed exactly, so the matching function vanished at
        ``phi = 0`` and the returned offset was an artefact of the solve's own loose tolerances.
        There is no root solve here, and ``phi`` is reported as zero rather than quietly dropped.
        """
        for nu in LOW_ORDERS + HIGH_ORDERS:
            max_x = 1.0e7 if nu in LOW_ORDERS else max(1000.0, 10.0 * nu)
            data = build(nu, max_x)
            self.assertEqual(data["phi"], 0.0)

    def test_public_surface_is_preserved(self):
        data = build(2.5, 1.0e7)
        for key in (
            "phase",
            "mod",
            "Q",
            "phi",
            "bessel_j",
            "bessel_y",
            "min_x",
            "max_x",
        ):
            self.assertIn(key, data)

        self.assertIsInstance(data["mod"], XSplineWrapper)

        phase = data["phase"]
        mod = data["mod"]
        x = 137.0
        # raw and logarithmic input modes agree at explicitly matched arguments
        u = math.log(x)
        self.assertAlmostEqual(
            phase.raw_theta(x), phase.raw_theta(u, x_is_log=True), delta=1.0e-12
        )
        self.assertAlmostEqual(mod(x), mod(u, is_log=True), delta=1.0e-15 * mod(x))
        self.assertAlmostEqual(
            phase.theta_deriv(x),
            phase.theta_deriv(u, x_is_log=True),
            delta=1.0e-14,
        )
        self.assertAlmostEqual(
            phase.theta_deriv(x, log_derivative=True),
            x * phase.theta_deriv(x),
            delta=1.0e-12,
        )
        # Q is theta/x, the quantity the old diagnostic plotted
        self.assertAlmostEqual(data["Q"](x), phase.raw_theta(x) / x, delta=1.0e-15)

    def test_c_nu_and_its_reduction(self):
        for nu in LOW_ORDERS + HIGH_ORDERS:
            self.assertAlmostEqual(c_nu(nu), 0.25 * math.pi - 0.5 * math.pi * nu)
            difference = c_nu(nu) - c_nu_reduced(nu)
            cycles = difference / (2.0 * math.pi)
            self.assertLess(abs(cycles - round(cycles)), 1.0e-12)

    def test_sampling_stays_far_below_the_silent_hankel_boundary(self):
        """
        Nothing is sampled above ``x_star``, and ``x_star`` is orders below the argument at which
        ``hankel1e`` returns exactly ``-0j`` -- which is *finite*, so an ``isfinite`` guard would
        pass it and ``log(abs(.))`` would put ``-inf`` into an interpolant.
        """
        for nu in LOW_ORDERS + HIGH_ORDERS:
            data = build(nu, 1.0e7 if nu in LOW_ORDERS else max(1000.0, 10.0 * nu))
            near = data["near_region"]
            if near is None:
                continue
            top = float(np.max(np.exp(near.log_x_nodes)))
            self.assertLessEqual(top, data["x_star"] * (1.0 + 1.0e-12))
            self.assertLess(top, br.scipy_reference_max_x(nu))
            self.assertTrue(np.all(np.isfinite(near.a_nodes)))
            self.assertTrue(np.all(near.a_nodes >= near.amplitude_band[0]))
            self.assertTrue(np.all(near.a_nodes <= near.amplitude_band[1]))

    def test_no_infinity_reaches_an_interpolant(self):
        for nu in (2.5, 100.5, 1000.5):
            data = build(nu, 1.0e7 if nu < 50 else max(1000.0, 10.0 * nu))
            xs = np.geomspace(data["min_x"], data["max_x"], 2000)
            for values in (
                data["mod"](xs),
                data["mod"].a(xs),
                data["phase"].residual(xs),
                data["phase"].theta_deriv(xs),
                data["bessel_j"](xs),
                data["bessel_y"](xs),
            ):
                self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all(np.asarray(data["mod"](xs)) > 0.0))

    def test_requests_outside_the_declared_domain_fail_loudly(self):
        with self.assertRaises(BesselPhaseError):
            bessel_phase(MIN_SUPPORTED_NU - 0.25, 1000.0)
        with self.assertRaises(BesselPhaseError):
            bessel_phase(2.5, 10.0 * MAX_SUPPORTED_X)
        with self.assertRaises(BesselPhaseError):
            # max_x below sqrt(nu^2 - 1/4): no oscillatory region at all
            bessel_phase(100.5, 50.0)

        data = build(2.5, 1000.0)
        with self.assertRaises(ValueError):
            data["mod"](0.5 * data["min_x"])
        with self.assertRaises(ValueError):
            data["phase"].raw_theta(2.0 * data["max_x"])

    def test_unmeetable_accuracy_fails_loudly(self):
        """
        A construction that cannot meet its requested accuracy must say so rather than return
        silently. Either the crossover cannot be placed at all, or the near region hits its
        refinement cap; both raise, and both are :class:`BesselPhaseError`.
        """
        with self.assertRaises(BesselPhaseError):
            bessel_phase(2.5, 1000.0, phase_atol=1.0e-20, amplitude_rtol=1.0e-20)

    def test_deprecated_tolerance_arguments(self):
        """
        ``atol``/``rtol`` named tolerances of an ODE solve that no longer exists, so they map to
        nothing rather than to an invented translation; the new arguments win when both are given.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated = bessel_phase(2.5, 1000.0, atol=1e-25, rtol=5e-14)
        self.assertTrue(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "supplying atol/rtol did not emit a DeprecationWarning",
        )
        self.assertEqual(deprecated["accuracy"]["phase_atol"], DEFAULT_PHASE_ATOL)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            both = bessel_phase(2.5, 1000.0, atol=1e-25, phase_atol=1.0e-9)
        self.assertEqual(both["accuracy"]["phase_atol"], 1.0e-9)
        self.assertTrue(
            any("take precedence" in str(w.message) for w in caught),
            "the warning did not say that the new arguments win",
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bessel_phase(2.5, 1001.0)
        self.assertFalse(
            any(issubclass(w.category, DeprecationWarning) for w in caught),
            "a call with no deprecated argument warned anyway",
        )

    def test_sample_points_is_a_floor_on_the_initial_density(self):
        """
        ``sample_points`` can no longer mean "use exactly this many" under adaptive sampling. It is
        reinterpreted as a floor on the initial node density, so it may make the grid finer and
        never coarser. ``docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:82`` passes it.
        """
        plain = bessel_phase(2.5, 1.0e7)
        dense = bessel_phase(2.5, 1.0e7, sample_points=20000)
        self.assertGreater(dense["near_region"].n_nodes, plain["near_region"].n_nodes)

        sparse = bessel_phase(2.5, 1.0e7, sample_points=1)
        self.assertGreaterEqual(
            sparse["near_region"].n_nodes, plain["near_region"].n_nodes
        )


class TestCostAndTheCliff(unittest.TestCase):
    """
    ``RECONCILIATION.md`` C1. The point is **not** a speed-up: the old build was ~0.1 s over the
    whole production range, so a ratio claim would be noise. The point is that the old construction
    stalls above ``x ~ 2.5e15``, where Amos ``jv``/``yv`` become O(1)-relatively noisy and DOP853 at
    ``rtol=5e-14`` drives its step size to zero -- and that the replacement never evaluates a Bessel
    routine there at all, because everything above ``x_star ~ 10-60 nu`` is closed form.
    """

    def test_build_cost_is_independent_of_x_max(self):
        times = {}
        for max_x in (1.0e3, 1.0e7, 1.0e11, 1.0e13, 1.0e15):
            start = time.perf_counter()
            bessel_phase(2.5, max_x)
            times[max_x] = time.perf_counter() - start
        print(
            "build times, nu=5/2: "
            + ", ".join(f"{k:.0e}:{v:.4f}s" for k, v in times.items())
        )
        self.assertLess(max(times.values()), 2.0)
        self.assertLess(max(times.values()), 20.0 * min(times.values()) + 0.05)

    def test_construction_clears_the_amos_cliff(self):
        """
        The two arguments at which the old construction did not return: ``3e15`` (measured
        abandoned after 60 s, log 01) and ``8.6e15`` (the benchmark's ``kappa = 1000`` tier).
        """
        for max_x in (3.0e15, 8.6e15):
            start = time.perf_counter()
            data = bessel_phase(2.5, max_x)
            elapsed = time.perf_counter() - start
            print(f"max_x={max_x:.3g}: built in {elapsed:.4f}s")
            self.assertLess(elapsed, 5.0)
            self.assertEqual(data["max_x"], max_x)

        # and the values up there are still right, scored against a 40-digit cached corner --
        # SciPy jv/yv are not a reference at 2.5e15 (RECONCILIATION.md C1.3)
        data = build(2.5, 3.0e15)
        corner = br.cached_corner(2.5, 2.5e15)
        sin_theta, cos_theta = data["phase"].sin_cos_theta(corner.x)
        e_theta = max(
            abs(sin_theta - corner.J / corner.amplitude),
            abs(-cos_theta - corner.Y / corner.amplitude),
        )
        e_amplitude = abs(data["mod"](corner.x) / corner.amplitude - 1.0)
        print(f"x=2.5e15 corner: E_theta={e_theta:.3e} E_A={e_amplitude:.3e}")
        self.assertLessEqual(e_theta, 1.0e-14)
        self.assertLessEqual(e_amplitude, 1.0e-14)


class TestZerosAndExtrema(unittest.TestCase):
    """
    ``README.md`` §6 requires coverage of the Bessel zeros and extrema, where a pointwise relative
    error is misleading -- ``|ours/theirs - 1|`` diverges at a zero of the reference however small
    the absolute error is. Only the envelope-normalized phase-pair measure is meaningful there.
    """

    @staticmethod
    def _roots(f, lo, hi, count):
        grid = np.linspace(lo, hi, 4000)
        values = f(grid)
        roots = []
        for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if fl == 0.0:
                roots.append(float(left))
            elif fl * fr < 0.0:
                roots.append(float(brentq(f, left, right, xtol=1e-14, rtol=1e-15)))
            if len(roots) >= count:
                break
        return np.array(roots)

    def test_near_zeros_and_extrema(self):
        for nu in (1.5, 2.5, 20.5):
            data = build(nu, 1.0e4)
            lo = 1.05 * data["min_x"]
            hi = 500.0

            j_zeros = self._roots(lambda x: jv(nu, x), lo, hi, 12)
            y_zeros = self._roots(lambda x: yv(nu, x), lo, hi, 12)
            # extrema of J_nu are the zeros of its derivative, J_{nu-1} - J_{nu+1} = 2 J'_nu
            j_extrema = self._roots(
                lambda x: jv(nu - 1.0, x) - jv(nu + 1.0, x), lo, hi, 12
            )
            xs = np.concatenate([j_zeros, y_zeros, j_extrema])
            self.assertGreater(xs.size, 20)

            result = score(nu, xs, data)
            print(
                f"nu={nu:>8} zeros/extrema: E_theta={result['E_theta'][0]:.3e}"
                f"@{result['E_theta'][1]:.6g} E_A={result['E_A'][0]:.3e}"
            )
            self.assertLessEqual(result["E_theta"][0], LOW_ORDER_TARGET)
            self.assertLessEqual(result["E_A"][0], LOW_ORDER_TARGET)

            # at a zero of J the absolute error must be small against the envelope; the relative
            # error against jv() itself is meaningless there and is deliberately not asserted
            envelope = data["mod"](j_zeros)
            self.assertTrue(
                np.all(
                    np.fabs(data["bessel_j"](j_zeros)) <= LOW_ORDER_TARGET * envelope
                )
            )


class TestNoLegacyMachinery(unittest.TestCase):
    def test_the_odE_the_root_solve_and_phase_spline_are_gone(self):
        import inspect

        from LiouvilleGreen import bessel_phase as module

        source = inspect.getsource(module)
        for token in ("solve_ivp", "root_scalar", "phase_spline", "simple_mod_2pi"):
            self.assertNotIn(token, source, f"{token} survives in bessel_phase.py")

    def test_the_false_justification_is_absent(self):
        """
        ``DRAFT-PLAN.md`` §4.6 justifies dropping the ``(div_2pi, mod_2pi)`` representation by
        saying the residual "never exceeds a cycle". That is **false** above ``nu ~ 630``: measured
        ``r(x_0) = 570.82`` rad at ``nu = 1000.5``, some 90 cycles (``RECONCILIATION.md`` C2). The
        conclusion stands on ``eps |r|_max ~ 1.3e-13`` instead, and the false justification must not
        appear anywhere in the module.
        """
        import inspect

        from LiouvilleGreen import bessel_phase as module

        source = inspect.getsource(module).lower()
        for phrase in ("never exceeds a cycle", "never exceeds one cycle", "sub-cycle"):
            self.assertNotIn(phrase, source)

        # and the fact itself, so that the claim cannot quietly become true again
        data = build(1000.5, 1.0e5)
        self.assertGreater(abs(data["accuracy"]["max_abs_residual"]), 500.0)


if __name__ == "__main__":
    unittest.main()
