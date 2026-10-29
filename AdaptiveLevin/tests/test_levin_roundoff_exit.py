"""
The round-off-limited exit checks its premise (prompts/levin-phase-floor/06b-roundoff-exit-premise.md).

_adaptive_levin() accepts a region without bisecting it as round-off-limited (phase_limited) when
its eq. (151) floor is finite and above tolerance and its residual is already down at that floor.
The exit rests on one premise: that bisecting the region cannot lower the floor. Before 06b it was
assumed; it fails where G1/G0 = max|theta'| / min|theta'| is large, because there the floor comes
from fitting one region over many decades of frequency. It is now checked: the exit fires only if
the region's two halves' floors sum to at least _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 1/2 of its
own, on the Levin branch from the halves the step-(4) residual already solved, and on the nested
Clenshaw-Curtis fallback from halves solved for the purpose.

The parent behaviour is run through _adaptive_levin()'s test-only keyword
_test_only_bypass_roundoff_premise, which restores the old test. That the bypass reproduces the
parent commit's module bit for bit is shown by
docs/levin-phase-floor/06b/calibration/scripts/parent_demo.py, against the parent's own file.

References are closed forms evaluated with mpmath at 30 digits; each reference's own error is
estimated as its difference from the same expression at 40 digits, and asserted negligible. Every
accuracy test asserts both that the value lies within its declared abserr of the reference and
that abserr is at least the true error.

No Ray and no datastore.
"""

import inspect
import unittest
from math import exp, atan, nan

import mpmath
import numpy as np

from AdaptiveLevin.levin_quadrature import (
    DEFAULT_LEVIN_MAX_DEPTH,
    _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO,
    _Basis_SinCos,
    _adaptive_levin,
    _adaptive_levin_subregion,
    _levin_G0_G1,
    _roundoff_exit_premise_holds,
    adaptive_levin_sincos,
    chebyshev_matrices,
)
from AdaptiveLevin.tests.test_levin_quadrature import _TwoIndependentSinCosBasis


def _run(span, f, theta, bypass=False, basis=None, **kwargs):
    """
    _adaptive_levin() on a (sin, cos) phase dict -- exactly what adaptive_levin_sincos() does
    after validating len(f) -- with the parent behaviour selected by bypass=True. basis, when
    given, replaces _Basis_SinCos(theta) (the m = 4 problem of test 4).
    """
    if basis is None:
        basis = _Basis_SinCos(theta)
    return _adaptive_levin(
        span, f, basis, _test_only_bypass_roundoff_premise=bypass, **kwargs
    )


def _reference(expression):
    """
    expression() evaluated with mpmath at 30 digits, and the reference's own error estimated as
    its difference from the same expression at 40 digits.
    """
    with mpmath.workdps(30):
        value30 = expression()
    with mpmath.workdps(40):
        value40 = expression()
    with mpmath.workdps(40):
        ref_err = float(abs(value30 - value40))
    return float(value40), ref_err


def _region_signature(region):
    """Every stored field of an accepted region, for bit-for-bit comparison."""
    return (
        region.start,
        region.end,
        region.depth,
        region.type,
        region.abserr,
        region.relerr,
        region.phase_err,
        region.phase_limited,
        tuple(region.p_ratios) if region.p_ratios is not None else None,
        region.abserr_truncation,
    )


def _call_signature(data):
    """value, abserr, evaluation count and region list of a call, for bit-for-bit comparison."""
    return (
        data["value"],
        data["abserr"],
        data["evaluations"],
        data["num_regions"],
        data["num_phase_limited_regions"],
        data["max_depth"],
        tuple(_region_signature(r) for r in data["regions"]),
    )


# ---------------------------------------------------------------------------------------------
# Test 1: theta = A e^x over [0, 20], so theta' spans e^20 = 4.85e8 (8.7 decades), with no
# stationary point. f = (1, 0), so the integral is Si(A e^20) - Si(A).
TEST1_A = 100.0
TEST1_SPAN = (0.0, 20.0)
TEST1_RTOL = 1e-7
TEST1_ATOL = 1e-15


def _test1_problem():
    f = [lambda x: 1.0, lambda x: 0.0]
    theta = {
        "theta": lambda x: TEST1_A * np.exp(x),
        "theta_deriv": lambda x: TEST1_A * np.exp(x),
    }
    return f, theta


def _test1_reference():
    a, b = TEST1_SPAN
    return _reference(
        lambda: mpmath.si(mpmath.mpf(TEST1_A) * mpmath.exp(mpmath.mpf(b)))
        - mpmath.si(mpmath.mpf(TEST1_A) * mpmath.exp(mpmath.mpf(a)))
    )


# Test 3: theta = A e^x over [0, 6] with A = 1e-6, so the total phase is ~4e-4 rad and every
# region takes the Clenshaw-Curtis fallback; f = (e^x, 0). theta' > 0 everywhere. The floor is
# dominated by its k^2/G0 term, ~eps ||f|| k^2 / min|theta'|, which the half holding min|theta'|
# shares with the parent -- but there ||f|| is e^-3 of the parent's, so the halves' floors sum to
# ~0.1 of the parent's. The integral is (cos(A) - cos(A e^6)) / A.
TEST3_A = 1e-6
TEST3_SPAN = (0.0, 6.0)
TEST3_RTOL = 1e-10
TEST3_ATOL = 1e-20


def _test3_problem():
    f = [lambda x: np.exp(x), lambda x: 0.0]
    theta = {
        "theta": lambda x: TEST3_A * np.exp(x),
        "theta_deriv": lambda x: TEST3_A * np.exp(x),
    }
    return f, theta


def _test3_reference():
    a, b = TEST3_SPAN
    return _reference(
        lambda: (
            mpmath.cos(mpmath.mpf(TEST3_A) * mpmath.exp(mpmath.mpf(a)))
            - mpmath.cos(mpmath.mpf(TEST3_A) * mpmath.exp(mpmath.mpf(b)))
        )
        / mpmath.mpf(TEST3_A)
    )


# Test 3 (b), since prompts/levin-phase-floor 06d: a fallback problem whose floor is 06d's
# node-rounding term. theta = A e^(x - X0) on [X0, X0 + 6], A = 0.025 (total phase ~10 rad), f =
# (e^-2(x - X0), 0), X0 = 1e6: the node rounding delta ~ 5u (X0 + h) is large, max|theta'| sits at
# the right end and ||f|| at the left, so the halves' floors sum to ~0.2 of the root's. The integral
# is int_0^6 e^-2t sin(A e^t) dt.
TEST3B_X0 = 1.0e6
TEST3B_A = 0.025
TEST3B_GAMMA = 2.0
TEST3B_SPAN = (TEST3B_X0, TEST3B_X0 + 6.0)
TEST3B_RTOL = 1e-16
TEST3B_ATOL = 1e-30


def _test3b_problem():
    f = [lambda x: exp(-TEST3B_GAMMA * (x - TEST3B_X0)), lambda x: 0.0]
    theta = {
        "theta": lambda x: TEST3B_A * exp(x - TEST3B_X0),
        "theta_deriv": lambda x: TEST3B_A * exp(x - TEST3B_X0),
    }
    return f, theta


def _test3b_reference():
    def expression():
        A = mpmath.mpf(TEST3B_A)
        g = mpmath.mpf(TEST3B_GAMMA)
        return mpmath.quad(
            lambda t: mpmath.exp(-g * t) * mpmath.sin(A * mpmath.exp(t)),
            mpmath.linspace(0, 6, 13),
        )

    return _reference(expression)


# Test 2's two problems, each with G1/G0 below 3 on every region.
#  (a) int_{1/3}^{7/3} e^-x sin(omega x) dx at omega = 1e6 -- the problem of
#      test_levin_quadrature.py's test_roundoff_floor_independent_of_theta_presentation, with
#      theta' supplied. theta' is constant, so G1/G0 = 1.
#  (b) int_0^2 sin(omega e^{x/2}) dx = 2 (Si(omega e) - Si(omega)) at omega = 1e6. G1/G0 = e on
#      the whole span, so the halves' floors sum to ~(1 + 2 sqrt(e)) / (1 + 2 e) = 0.67 of the
#      parent's: the premise holds and the exit fires. atol and rtol are set below the step-(4)
#      residual's own noise (8.5e-22, relerr 7.4e-16) so that the floor (6.4e-16) binds.
TEST2A_OMEGA = 1e6
TEST2A_SPAN = (1.0 / 3.0, 7.0 / 3.0)
TEST2B_OMEGA = 1e6
TEST2B_SPAN = (0.0, 2.0)


def _test2a_problem():
    f = [lambda x: exp(-x), lambda x: 0.0]
    theta = {
        "theta": lambda x: TEST2A_OMEGA * x,
        "theta_deriv": lambda x: TEST2A_OMEGA,
    }
    kwargs = dict(atol=1e-18, rtol=1e-13, chebyshev_order=12)
    a, b = TEST2A_SPAN

    def expression():
        w = mpmath.mpf(TEST2A_OMEGA)

        def F(x):
            x = mpmath.mpf(x)
            return (
                -mpmath.exp(-x)
                * (mpmath.sin(w * x) + w * mpmath.cos(w * x))
                / (1 + w * w)
            )

        return F(b) - F(a)

    return f, theta, kwargs, (lambda x: TEST2A_OMEGA + 0.0 * x), expression


def _test2b_problem():
    f = [lambda x: 1.0, lambda x: 0.0]
    theta = {
        "theta": lambda x: TEST2B_OMEGA * np.exp(0.5 * x),
        "theta_deriv": lambda x: 0.5 * TEST2B_OMEGA * np.exp(0.5 * x),
    }
    kwargs = dict(atol=1e-24, rtol=1e-16, chebyshev_order=16)
    a, b = TEST2B_SPAN

    def expression():
        w = mpmath.mpf(TEST2B_OMEGA)
        half = mpmath.mpf(1) / 2
        return 2 * (
            mpmath.si(w * mpmath.exp(half * mpmath.mpf(b)))
            - mpmath.si(w * mpmath.exp(half * mpmath.mpf(a)))
        )

    return f, theta, kwargs, theta["theta_deriv"], expression


# ---------------------------------------------------------------------------------------------
# Test 4: every problem test_levin_quadrature.py drives _adaptive_levin() with, restated here with
# the arguments that file passes (the file itself is not edited). Each entry is
# (name, span, f, theta, kwargs, basis, raises): basis is None except for the m = 4 problem;
# raises names the exception a problem is expected to raise, or None.
def _theta_mod_2pi(omega):
    def theta_mod_2pi(x, w=omega):
        val = np.fmod(w * x, 2.0 * np.pi)
        if val < 0.0:
            val += 2.0 * np.pi
        return val

    return theta_mod_2pi


def _gaussian_bump(x):
    return exp(-400.0 * (x - 1.1) ** 2)


def levin_quadrature_problems():
    """test_levin_quadrature.py's driver problems; see the comment above."""
    problems = [
        (
            "SinIntegral",
            (1.0, 50000.0),
            [lambda x: 1.0, lambda x: 0.0],
            {"theta": lambda x: x},
            dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            None,
            None,
        ),
        (
            "CosIntegral",
            (1.0, 500000.0),
            [lambda x: 0.0, lambda x: 1.0],
            {"theta": lambda x: x},
            dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            None,
            None,
        ),
        (
            "SincIntegral",
            (1.0, 100.0),
            [lambda x: 1.0 / x, lambda x: 0.0],
            {"theta": lambda x: 100.0 * x},
            dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            None,
            None,
        ),
    ]
    for lbda in (10.0, 100.0, 1000.0):
        problems.append(
            (
                f"GRZIntegral({lbda:g})",
                (-1.0, 1.0),
                [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)],
                {"theta": lambda x, l=lbda: l * atan(x)},
                dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
                None,
                None,
            )
        )
    problems += [
        (
            "nan_amplitude",
            (1.0, 2.0),
            [lambda x: nan if x > 1.9 else 1.0, lambda x: 0.0],
            {"theta": lambda x: 1.0e5 * x},
            dict(),
            None,
            ValueError,
        ),
        (
            "all_nan_amplitude",
            (1.0, 2.0),
            [lambda x: nan, lambda x: nan],
            {"theta": lambda x: 1.0e5 * x},
            dict(),
            None,
            ValueError,
        ),
        (
            "input_validation_order_clamp",
            (1.0, 50.0),
            [lambda x: 1.0, lambda x: 0.0],
            {"theta": lambda x: x},
            dict(chebyshev_order=4),
            None,
            None,
        ),
        (
            "lstsq_path",
            (0.0, 1.0),
            [lambda x: exp(-x), lambda x: 0.3],
            {"theta": lambda x: 10.0 * np.pi * x},
            dict(atol=1e-15, rtol=1e-12),
            None,
            None,
        ),
        (
            "direct_solve_path",
            (0.0, 1.0),
            [lambda x: exp(-x), lambda x: 0.3],
            {"theta": lambda x: 30.0 * np.pi * x},
            dict(atol=1e-15, rtol=1e-12),
            None,
            None,
        ),
        (
            "converged_flag_bump",
            (0.0, 3.0),
            [_gaussian_bump, lambda x: 0.0],
            {"theta": lambda x: 3.0e4 * x},
            dict(atol=1e-10),
            None,
            None,
        ),
        (
            "stationary_phase",
            (0.0, 1.0),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * (x - x * x)},
            dict(atol=1e-10, rtol=1e-10),
            None,
            None,
        ),
    ]
    for omega in (1.0e6, 1.0e12):
        problems += [
            (
                f"roundoff_floor_raw({omega:g})",
                (1.0 / 3.0, 7.0 / 3.0),
                [lambda x: exp(-x), lambda x: 0.0],
                {"theta": lambda x, w=omega: w * x},
                dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
                None,
                None,
            ),
            (
                f"roundoff_floor_reduced({omega:g})",
                (1.0 / 3.0, 7.0 / 3.0),
                [lambda x: exp(-x), lambda x: 0.0],
                {
                    "theta": lambda x, w=omega: w * x,
                    "theta_mod_2pi": _theta_mod_2pi(omega),
                    "theta_deriv": lambda x, w=omega: w,
                },
                dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
                None,
                None,
            ),
        ]
    problems += [
        (
            "theta_abserr_scalar",
            (1.0 / 3.0, 7.0 / 3.0),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * x, "theta_abserr": 1e-8},
            dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
            None,
            None,
        ),
        (
            "theta_abserr_callable",
            (1.0 / 3.0, 7.0 / 3.0),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * x, "theta_abserr": lambda x: 1e-8},
            dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
            None,
            None,
        ),
        (
            "mode_filter_loose",
            (1.0, 1.0003),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * x},
            dict(atol=1e-20, rtol=1e-2, chebyshev_order=16, depth_max=20),
            None,
            None,
        ),
        (
            "mode_filter_tight",
            (1.0, 1.0003),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * x},
            dict(atol=1e-20, rtol=1e-8, chebyshev_order=16, depth_max=20),
            None,
            None,
        ),
        (
            "theta_mod_2pi_path_only",
            (1.0 / 3.0, 7.0 / 3.0),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * x, "theta_mod_2pi": _theta_mod_2pi(1.0e6)},
            dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
            None,
            None,
        ),
        (
            "theta_deriv_path_only",
            (1.0 / 3.0, 7.0 / 3.0),
            [lambda x: exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1.0e6 * x, "theta_deriv": lambda x: 1.0e6},
            dict(atol=1e-18, rtol=1e-13, chebyshev_order=12),
            None,
            None,
        ),
        (
            "reversed_span_forward",
            (1.0, 5.0),
            [lambda x: 1.0, lambda x: 0.0],
            {"theta": lambda x: 100.0 * x},
            dict(atol=1e-13, rtol=1e-11),
            None,
            None,
        ),
        (
            "reversed_span_reversed",
            (5.0, 1.0),
            [lambda x: 1.0, lambda x: 0.0],
            {"theta": lambda x: 100.0 * x},
            dict(atol=1e-13, rtol=1e-11),
            None,
            None,
        ),
        (
            "generic_m4_basis",
            (1.0, 50.0),
            [lambda x: 1.0, lambda x: 0.0, lambda x: 0.0, lambda x: 1.0],
            None,
            dict(atol=1e-13, rtol=1e-11, chebyshev_order=16),
            _TwoIndependentSinCosBasis(1.0, 2.0),
            None,
        ),
        (
            "vectorized_sinc",
            (1.0, 100.0),
            [lambda x: 1.0 / x, lambda x: 0.0 * x],
            {"theta": lambda x: 100.0 * x},
            dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
            None,
            None,
        ),
    ]
    return problems


class TestRoundoffExitPremise(unittest.TestCase):

    def _assert_honest(self, data, ref, ref_err, label):
        true_err = abs(data["value"] - ref)
        # the reference's own error must be negligible against what is asserted on it
        self.assertLess(ref_err, 1e-6 * max(true_err, data["abserr"], 1e-300), label)
        # the value lies within its declared abserr of the reference ...
        self.assertLessEqual(
            abs(data["value"] - ref),
            data["abserr"],
            f"{label}: value {data['value']!r} is not within abserr {data['abserr']:.3e} of "
            f"the reference {ref!r}",
        )
        # ... and abserr is at least the true error (the same inequality, stated as the prompt
        # states it)
        self.assertGreaterEqual(data["abserr"], true_err, label)
        return true_err

    def test_premise_holds_on_infinite_half_floor(self):
        # Prompt 06b section 1 item 6: a half's floor of +inf makes the sum +inf, so the premise
        # holds and the exit may fire -- today's behaviour, since the exit still needs the
        # parent's own floor to be finite.
        self.assertTrue(_roundoff_exit_premise_holds(1.0, float("inf"), 0.0))
        self.assertTrue(_roundoff_exit_premise_holds(1.0, 0.0, float("inf")))
        # the ratio is rho = 1/2, inclusive
        self.assertEqual(_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO, 0.5)
        self.assertTrue(_roundoff_exit_premise_holds(1.0, 0.25, 0.25))
        self.assertFalse(_roundoff_exit_premise_holds(1.0, 0.25, 0.2499))

    def test_bypass_is_not_exposed(self):
        # Prompt 06b section 1 item 4: the bypass is a private, test-only keyword of
        # _adaptive_levin(), off by default; adaptive_levin_sincos() does not expose it.
        self.assertNotIn(
            "_test_only_bypass_roundoff_premise",
            inspect.signature(adaptive_levin_sincos).parameters,
        )
        self.assertIs(
            inspect.signature(_adaptive_levin)
            .parameters["_test_only_bypass_roundoff_premise"]
            .default,
            False,
        )

    def test_exit_against_its_premise_levin_branch(self):
        """
        Test 1. theta = 100 e^x on [0, 20]: theta' spans 8.7 decades, with no stationary point.

        At the parent (the bypass; and the parent commit's own module, in
        docs/levin-phase-floor/06b/calibration/out/parent_demo.txt) the whole span is accepted at
        depth 0 through the exit, on a floor of 9.7e-7 with a step-(4) residual of 6.4e-9:
            value  0.008570866361715342
            abserr 9.703303918195807e-07
            1 region, 1 phase-limited, converged False
        against the reference 0.008570859924939634 (Si(100 e^20) - Si(100)): a true error of
        6.4e-9, relative 7.5e-7 against the requested rtol of 1e-7. Honest, but outside the
        tolerance -- the pattern of [07-roundoff-exit-assumes-bisection-cannot-help].

        The halves' floors sum to ~1e-4 of the parent's, so under the new code the region is
        bisected, and the call converges within its tolerance with no phase-limited region.
        """
        f, theta = _test1_problem()
        ref, ref_err = _test1_reference()
        kwargs = dict(atol=TEST1_ATOL, rtol=TEST1_RTOL)

        parent = _run(TEST1_SPAN, f, theta, bypass=True, **kwargs)
        self.assertEqual(parent["num_regions"], 1)
        self.assertEqual(parent["num_phase_limited_regions"], 1)
        self.assertEqual(parent["regions"][0].depth, 0)
        self.assertEqual(parent["regions"][0].type, "Levin")
        self.assertTrue(parent["regions"][0].phase_limited)
        parent_true_err = self._assert_honest(parent, ref, ref_err, "test 1 parent")
        self.assertGreater(parent_true_err, TEST1_RTOL * abs(ref))

        # the premise fails on that region, by orders of magnitude
        basis = _Basis_SinCos(theta)
        a, b = TEST1_SPAN
        c = 0.5 * (a + b)
        floor_L = _adaptive_levin_subregion((a, c), f, basis, None, rtol=TEST1_RTOL)[
            "phase_err"
        ]
        floor_R = _adaptive_levin_subregion((c, b), f, basis, None, rtol=TEST1_RTOL)[
            "phase_err"
        ]
        self.assertLess(floor_L + floor_R, 1e-3 * parent["regions"][0].phase_err)

        new = _run(TEST1_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        self.assertEqual(new["num_phase_limited_regions"], 0)
        self.assertFalse(new["phase_limited"])
        new_true_err = self._assert_honest(new, ref, ref_err, "test 1 new")
        self.assertLessEqual(new_true_err, TEST1_RTOL * abs(ref))

    def _check_premise_holds_problem(self, problem, label):
        f, theta, kwargs, theta_deriv, expression = problem
        span = TEST2A_SPAN if label == "test 2a" else TEST2B_SPAN
        ref, ref_err = _reference(expression)

        new = _run(span, f, theta, **kwargs)
        parent = _run(span, f, theta, bypass=True, **kwargs)

        # the exit fires under the new code ...
        self.assertGreaterEqual(new["num_phase_limited_regions"], 1, label)
        # ... no region reaches depth_max ...
        depth_max = kwargs.get("depth_max", DEFAULT_LEVIN_MAX_DEPTH)
        self.assertLess(new["max_depth"], depth_max, label)
        # ... the result is bit-identical to the bypassed (parent) call ...
        self.assertEqual(_call_signature(new), _call_signature(parent), label)
        # ... G1/G0 is below 3 on every region ...
        order = kwargs.get("chebyshev_order")
        for region in new["regions"]:
            grid, _ = chebyshev_matrices((region.start, region.end), order)
            G0, G1 = _levin_G0_G1(
                np.array([theta_deriv(x) for x in grid]),
                abs(region.end - region.start),
            )
            self.assertLess(G1 / G0, 3.0, f"{label}: region {region}")
        # ... and the value is honest
        self._assert_honest(new, ref, ref_err, label)

    def test_exit_fires_where_premise_holds(self):
        """
        Test 2. Two problems with G1/G0 below 3 on every region, at tolerances the floor binds:
        (a) constant theta' (G1/G0 = 1), the problem of test_levin_quadrature.py's
        test_roundoff_floor_independent_of_theta_presentation; (b) theta' = 5e5 e^{x/2} on
        [0, 2] (G1/G0 = e). Each is accepted at depth 0 through the exit, at the parent and under
        the new code alike, bit for bit.
        """
        self._check_premise_holds_problem(_test2a_problem(), "test 2a")
        self._check_premise_holds_problem(_test2b_problem(), "test 2b")

    def test_exit_against_its_premise_fallback_branch(self):
        """
        Test 3. The round-off-limited exit on the nested Clenshaw-Curtis fallback, against its
        premise.

        (a) theta = 1e-6 e^x on [0, 6], f = (e^x, 0): every region takes the fallback (total phase
        ~4e-4 rad), and theta' > 0 throughout. Until prompts/levin-phase-floor 06d the fallback
        carried the Levin rule's eq. (151) floor, whose 1/min|theta'| made the root's floor 1.03e-5
        against halves' floors of 5.1e-7 each: the bypass (06b's parent) accepted the root at depth
        0 through the exit, the premise check bisected it. Before (06c, ad9a5f4): the bypass 1
        direct region at depth 0, phase-limited, abserr 1.03e-5; the new code 4 direct regions at
        depth 2, all phase-limited, abserr 4.59e-7. Under 06d the fallback's floor is the rule's
        own (_cc_roundoff_floor()), which has no 1/theta': the call is bit-identical across the
        bypass -- 2 direct regions at depth 1, the left one phase-limited, abserr 1.98e-12,
        converged -- and honest. The assertions on (a) were changed to that outcome (06d section 5
        point 3: region count, depth, which rule, the phase-limited count, bit-identity across the
        bypass); its honesty assertions are unchanged.

        (b) The assertion that the premise check at least halves the declared error moves to a
        problem where 06d's floor binds and the premise fails at the root: theta = A e^(x - X0),
        A = 0.025, f = (e^-2(x - X0), 0) on [X0, X0 + 6] with X0 = 1e6, total phase 10 rad, a
        request below round-off. The floor is the node-rounding term, delta ~ 5u (X0 + h) times
        max|theta'| sum|w||f| + sum|w||f'|: max|theta'| sits at the right end, ||f|| at the left,
        so the halves' floors sum to ~0.2 of the root's. The bypass accepts the root at depth 0
        through the exit (abserr 3.35e-9); the new code bisects, and the exit fires at depth 2 on
        leaves whose premise holds (abserr 6.96e-10, 0.21 of the bypass's). Both honest.
        """
        # (a)
        f, theta = _test3_problem()
        ref, ref_err = _test3_reference()
        kwargs = dict(atol=TEST3_ATOL, rtol=TEST3_RTOL)

        parent = _run(TEST3_SPAN, f, theta, bypass=True, **kwargs)
        self.assertEqual(parent["num_regions"], 2)
        self.assertEqual(parent["num_simple_regions"], 2)
        self.assertTrue(all(r.type == "direct" for r in parent["regions"]))
        self.assertEqual(parent["max_depth"], 1)
        self.assertEqual(parent["num_phase_limited_regions"], 1)
        self._assert_honest(parent, ref, ref_err, "test 3a parent")

        new = _run(TEST3_SPAN, f, theta, **kwargs)
        self.assertEqual(_call_signature(new), _call_signature(parent))
        self._assert_honest(new, ref, ref_err, "test 3a new")

        # (b)
        f, theta = _test3b_problem()
        ref, ref_err = _test3b_reference()
        kwargs = dict(atol=TEST3B_ATOL, rtol=TEST3B_RTOL)

        parent = _run(TEST3B_SPAN, f, theta, bypass=True, **kwargs)
        self.assertEqual(parent["num_regions"], 1)
        self.assertEqual(parent["regions"][0].type, "direct")
        self.assertEqual(parent["regions"][0].depth, 0)
        self.assertTrue(parent["regions"][0].phase_limited)
        self._assert_honest(parent, ref, ref_err, "test 3b parent")

        new = _run(TEST3B_SPAN, f, theta, **kwargs)
        # it bisects: no region is accepted at depth 0, and every region is still a fallback
        # region; the exit fires below the root
        self.assertGreaterEqual(new["max_depth"], 1)
        self.assertTrue(all(r.depth >= 1 for r in new["regions"]))
        self.assertEqual(new["num_simple_regions"], new["num_regions"])
        self.assertGreater(new["num_phase_limited_regions"], 0)
        # every region flagged phase-limited satisfies the premise, from its own halves
        basis = _Basis_SinCos(theta)
        for region in new["regions"]:
            if region.phase_limited:
                c = 0.5 * (region.start + region.end)
                floor_L = _adaptive_levin_subregion(
                    (region.start, c), f, basis, None, rtol=TEST3B_RTOL
                )["phase_err"]
                floor_R = _adaptive_levin_subregion(
                    (c, region.end), f, basis, None, rtol=TEST3B_RTOL
                )["phase_err"]
                self.assertTrue(
                    _roundoff_exit_premise_holds(region.phase_err, floor_L, floor_R),
                    f"test 3b new: {region}",
                )
        self._assert_honest(new, ref, ref_err, "test 3b new")
        # the declared error falls with the floor
        self.assertLess(new["abserr"], 0.5 * parent["abserr"])

    def test_nothing_else_moves(self):
        """
        Test 4. Every problem test_levin_quadrature.py drives _adaptive_levin() with, with and
        without the bypass. Where the bypassed call has no phase-limited region, value, abserr,
        region list and evaluation count are bit-identical: the new condition is strictly
        stronger than the old, so only a region the parent accepted through the exit can move.
        Where it has one, both results are tabulated in the log
        (docs/levin-phase-floor/06b/calibration/out/parent_demo.txt); at this commit every such
        problem is bit-identical too, and is asserted so.

        Under prompts/levin-phase-floor 06c's global budget the evaluation count is no longer
        compared: a fallback leaf the loop chooses solves its halves for the premise, which the
        bypass skips, so on GRZIntegral(100) (203 bypassed, 407 not) and the stationary-phase
        oracle (193,893 and 218,735) the counts differ while value, abserr and every region are
        bit-identical. Both are of the kinds 06c's section 2 test 9 allows to change (an
        evaluation count; bit-identity across the bypass). The count is asserted no smaller
        without the bypass.
        """
        for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
            with self.subTest(problem=name):
                if raises is not None:
                    with self.assertRaises(raises):
                        _run(span, f, theta, bypass=True, basis=basis, **kwargs)
                    with self.assertRaises(raises):
                        _run(span, f, theta, basis=basis, **kwargs)
                    continue

                parent = _run(span, f, theta, bypass=True, basis=basis, **kwargs)
                new = _run(span, f, theta, basis=basis, **kwargs)
                new_signature = _call_signature(new)
                parent_signature = _call_signature(parent)
                # element 2 is the evaluation count (see the docstring)
                self.assertEqual(
                    new_signature[:2] + new_signature[3:],
                    parent_signature[:2] + parent_signature[3:],
                    name,
                )
                self.assertGreaterEqual(new["evaluations"], parent["evaluations"], name)


if __name__ == "__main__":
    unittest.main()
