"""
Global error control in the Levin driver (prompts/levin-phase-floor/06c-global-error-control.md).

_adaptive_levin() keeps every leaf of its partition, with its value v_i and its error e_i
(used_interval.total_err), and stops when sum e_i <= max(atol, rtol |sum v_i|); until then it
bisects the refinable leaf with the largest error. A leaf is not refinable at depth_max, or when it
is floor-limited (round-off floor finite, residual at most the floor, and 06b's premise holding).
Before 06c each region was accepted against its own value, or against atol scaled by its share of
the span's length, and the call's total was compared with the request only afterwards.

The parent behaviour quoted in the docstrings below was measured with the parent commit's own module
(3f2a36d, extracted with `git show` and imported under another name) by
docs/levin-phase-floor/06c/calibration/scripts/parent_demo.py; its output is
docs/levin-phase-floor/06c/calibration/out/parent_demo.txt. This module does not run the parent.

References are closed forms evaluated with mpmath; each reference's own error is estimated as its
30-digit value's difference from its 40-digit value, and asserted negligible. Every accuracy test
asserts both that the value lies within its declared abserr of the reference and that abserr is at
least the true error.

No Ray and no datastore.
"""

import math
import unittest

import mpmath
import numpy as np

from AdaptiveLevin.levin_quadrature import (
    DEFAULT_LEVIN_RELTOL as NEW_DEFAULT_RTOL,
    _Basis_SinCos,
    _ExactSum,
    _adaptive_levin,
    _adaptive_levin_subregion,
    _error_for_ordering,
)
from AdaptiveLevin.tests.test_levin_roundoff_exit import (
    TEST2A_SPAN,
    TEST2B_SPAN,
    _test2a_problem,
    _test2b_problem,
    levin_quadrature_problems,
)

PARENT_SHA = "3f2a36d"


def _run(span, f, theta, **kwargs):
    """_adaptive_levin() on a (sin, cos) phase dict, as adaptive_levin_sincos() calls it."""
    return _adaptive_levin(span, f, _Basis_SinCos(theta), **kwargs)


def _reference(expression):
    """expression() at 30 and at 40 digits: the 40-digit value, and their difference as its error."""
    with mpmath.workdps(30):
        value30 = expression()
    with mpmath.workdps(40):
        value40 = expression()
    with mpmath.workdps(40):
        ref_err = float(abs(value30 - value40))
    return float(value40), ref_err


def _gauss_integral(gamma, beta, a, b):
    """int_a^b exp(gamma u^2 + beta u) du for complex gamma != 0, in closed form (mpmath)."""
    s = mpmath.sqrt(-gamma)
    c = beta / (2 * gamma)
    return (
        mpmath.exp(-(beta**2) / (4 * gamma))
        * mpmath.sqrt(mpmath.pi)
        / (2 * s)
        * (mpmath.erf(s * (b + c)) - mpmath.erf(s * (a + c)))
    )


def _region_signature(region):
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
    return (
        data["value"],
        data["abserr"],
        data["evaluations"],
        data["num_regions"],
        data["num_phase_limited_regions"],
        data["max_depth"],
        data["converged"],
        tuple(_region_signature(r) for r in data["regions"]),
    )


def _sum_abs_leaf_values(data, f, theta, rtol, chebyshev_order=16):
    """sum_i |v_i| over a call's leaves, each leaf re-solved on its own interval."""
    basis = _Basis_SinCos(theta)
    return sum(
        abs(
            _adaptive_levin_subregion(
                (r.start, r.end),
                f,
                basis,
                None,
                chebyshev_order=chebyshev_order,
                rtol=rtol,
            )["value"]
        )
        for r in data["regions"]
    )


# ---------------------------------------------------------------------------------------------
# Test 1: cancellation on the fallback branch. 07 section 2's family (R), theta = (c/2)(x - x_s)^2
# on [0, 1], with f_sin = -f_cos = -cos 3x, so the integrand is sqrt(2) cos 3x cos(theta + pi/4)
# and the stationary-phase term vanishes. The leaves round x_s are Clenshaw-Curtis fallback
# leaves. With u = x - x_s and cos 3x = (e^{3ix} + e^{-3ix})/2,
#   I = Re[ (1 + i)/2 sum_{beta = +-3} e^{i beta x_s} int_{-x_s}^{1 - x_s} e^{i (c/2) u^2 + i beta u} du ].
TEST1_C = 1.0e4
TEST1_XS = 0.7236067977499790
TEST1_SPAN = (0.0, 1.0)
TEST1_ATOL = 1e-18
TEST1_RTOL = 1e-7


def test1_problem():
    c, xs = TEST1_C, TEST1_XS
    f = [lambda x: -np.cos(3.0 * x), lambda x: np.cos(3.0 * x)]
    theta = {
        "theta": lambda x: 0.5 * c * (x - xs) ** 2,
        "theta_deriv": lambda x: c * (x - xs),
    }

    def expression():
        cm = mpmath.mpf(TEST1_C)
        xm = mpmath.mpf(TEST1_XS)
        total = sum(
            mpmath.exp(1j * beta * xm)
            * _gauss_integral(1j * cm / 2, 1j * beta, -xm, 1 - xm)
            for beta in (3, -3)
        )
        return mpmath.re((1 + 1j) / 2 * total)

    return f, theta, dict(atol=TEST1_ATOL, rtol=TEST1_RTOL), expression


# Test 2: cancellation on the Levin branch. f = (1/x, 0), theta = omega x on [1, L], with
# omega = 100.5 pi and omega L = 10050.5 pi, so cos(omega) = cos(omega L) = 0 and the total
# Si(omega L) - Si(omega) is O(1/omega^2), while each Levin leaf's value is O(1/omega): the leaves'
# boundary terms telescope.
TEST2_OMEGA = 100.5 * math.pi
TEST2_SPAN = (1.0, 10050.5 * math.pi / TEST2_OMEGA)
TEST2_ATOL = 1e-20
TEST2_RTOL = 1e-8
TEST2_ORDER = 16


def test2_problem():
    w = TEST2_OMEGA
    f = [lambda x: 1.0 / x, lambda x: 0.0]
    theta = {"theta": lambda x: w * x, "theta_deriv": lambda x: w + 0.0 * x}

    def expression():
        wm = (100 + mpmath.mpf(1) / 2) * mpmath.pi
        return mpmath.si(wm * mpmath.mpf(TEST2_SPAN[1])) - mpmath.si(
            wm * mpmath.mpf(TEST2_SPAN[0])
        )

    kwargs = dict(atol=TEST2_ATOL, rtol=TEST2_RTOL, chebyshev_order=TEST2_ORDER)
    return f, theta, kwargs, expression


# Test 3: the inflated floor ([06b-resolved-region-keeps-a-floor-bisection-would-cut]):
# theta = 1000 e^x on [0, 20], f = (1, 0), rtol 1e-7. I = Si(1000 e^20) - Si(1000).
TEST3_A = 1000.0
TEST3_SPAN = (0.0, 20.0)
TEST3_ATOL = 1e-15
TEST3_RTOL = 1e-7


def test3_problem():
    f = [lambda x: 1.0, lambda x: 0.0]
    theta = {
        "theta": lambda x: TEST3_A * np.exp(x),
        "theta_deriv": lambda x: TEST3_A * np.exp(x),
    }

    def expression():
        a, b = TEST3_SPAN
        return mpmath.si(mpmath.mpf(TEST3_A) * mpmath.exp(mpmath.mpf(b))) - mpmath.si(
            mpmath.mpf(TEST3_A) * mpmath.exp(mpmath.mpf(a))
        )

    return f, theta, dict(atol=TEST3_ATOL, rtol=TEST3_RTOL), expression


# Test 5: an unattainable request. f = (|x - 1/3|, 0), theta = 100 x on [0, 1]: the kink at the
# non-dyadic x = 1/3 is never on a node, so the leaf holding it converges only algebraically, and
# rtol 1e-12 cannot be met by depth_max = 6. The antiderivative of (x - a) sin(w x) is
# -(x - a) cos(w x)/w + sin(w x)/w^2.
TEST5_OMEGA = 100.0
TEST5_SPAN = (0.0, 1.0)
TEST5_ATOL = 1e-20
TEST5_RTOL = 1e-12
TEST5_DEPTH_MAX = 6


def test5_problem():
    w = TEST5_OMEGA
    f = [lambda x: abs(x - 1.0 / 3.0), lambda x: 0.0]
    theta = {"theta": lambda x: w * x, "theta_deriv": lambda x: w + 0.0 * x}

    def expression():
        wm = mpmath.mpf(TEST5_OMEGA)
        k = mpmath.mpf(1) / 3

        def F(x):
            x = mpmath.mpf(x)
            return -(x - k) * mpmath.cos(wm * x) / wm + mpmath.sin(wm * x) / wm**2

        # |x - k| = -(x - k) on [0, k], +(x - k) on [k, 1]
        return -(F(k) - F(0)) + (F(1) - F(k))

    kwargs = dict(atol=TEST5_ATOL, rtol=TEST5_RTOL, depth_max=TEST5_DEPTH_MAX)
    return f, theta, kwargs, expression


# Test 6: an infinite floor. theta = c (x - x^2) on [0, 1], c = 1e4, f = (e^-x, 0), theta' by
# spectral differentiation (no theta_deriv), as test_levin_quadrature.py's stationary-phase oracle
# but at c = 1e4: the stationary point x = 1/2 is the first break point, so it is a node of every
# leaf touching it, and eq. (151) is +inf there (G0 at the noise floor of G1). With u = x - 1/2,
# e^-x e^{i theta} = e^{-1/2 + i c/4} e^{-i c u^2 - u}, so I = Im[e^{-1/2 + i c/4} G], G the
# Gaussian integral over u in [-1/2, 1/2].
TEST6_C = 1.0e4
TEST6_SPAN = (0.0, 1.0)
TEST6_ATOL = 1e-10
TEST6_RTOL = 1e-10


def test6_problem():
    c = TEST6_C
    f = [lambda x: np.exp(-x), lambda x: 0.0]
    theta = {"theta": lambda x: c * (x - x * x)}

    def expression():
        cm = mpmath.mpf(TEST6_C)
        half = mpmath.mpf(1) / 2
        return mpmath.im(
            mpmath.exp(-half + 1j * cm / 4)
            * _gauss_integral(-1j * cm, mpmath.mpf(-1), -half, half)
        )

    return f, theta, dict(atol=TEST6_ATOL, rtol=TEST6_RTOL), expression


# Test 7: a total near zero. f = (0, tanh(30 x)), theta = 20 x on [-1, 1]: tanh(30 x) cos(20 x) is
# odd, so the integral is zero, and the budget is atol.
TEST7_SPAN = (-1.0, 1.0)
TEST7_ATOL = 1e-12
TEST7_RTOL = 1e-10


def test7_problem():
    f = [lambda x: 0.0, lambda x: np.tanh(30.0 * x)]
    theta = {"theta": lambda x: 20.0 * x, "theta_deriv": lambda x: 20.0 + 0.0 * x}
    return f, theta, dict(atol=TEST7_ATOL, rtol=TEST7_RTOL)


# ---------------------------------------------------------------------------------------------
# Test 12: the trust check one level down (06c section 8, the third amendment).
#
# 12a: test_converged_flag's Gaussian bump, e^{-400 (x - 1.1)^2} sin(3e4 x) on [0, 3], atol 1e-10.
# Its integral is Im[e^{i 3e4 1.1} int_{-1.1}^{1.9} e^{-400 u^2 + i 3e4 u} du] ~ 2.1e-215.
TEST12A_SPAN = (0.0, 3.0)
TEST12A_ATOL = 1e-10


def test12a_problem():
    f = [lambda x: math.exp(-400.0 * (x - 1.1) ** 2), lambda x: 0.0]
    theta = {"theta": lambda x: 3.0e4 * x}

    def expression():
        k = mpmath.mpf(30000)
        x0 = mpmath.mpf(1.1)
        return mpmath.im(
            mpmath.exp(1j * k * x0)
            * _gauss_integral(mpmath.mpf(-400), 1j * k, -x0, 3 - x0)
        )

    return f, theta, dict(atol=TEST12A_ATOL), expression


# 12b: SincIntegral, int_1^100 sin(100 x)/x, rtol 1e-10, order 12 = Si(1e4) - Si(100).
TEST12B_SPAN = (1.0, 100.0)
TEST12B_RTOL = 1e-10


def test12b_problem():
    f = [lambda x: 1.0 / x, lambda x: 0.0]
    theta = {"theta": lambda x: 100.0 * x}

    def expression():
        return mpmath.si(mpmath.mpf(10000)) - mpmath.si(mpmath.mpf(100))

    return f, theta, dict(atol=1e-15, rtol=TEST12B_RTOL, chebyshev_order=12), expression


# 12c: family (R) with cancelling amplitudes at x_s = 0.382, c = 1e5, rtol 1e-7, theta' supplied
# -- the case the withdrawn stall rule broke. The same closed form as test 1.
TEST12C_C = 1.0e5
TEST12C_XS = 0.3819660112501051
TEST12C_SPAN = (0.0, 1.0)
TEST12C_RTOL = 1e-7


def test12c_problem():
    c, xs = TEST12C_C, TEST12C_XS
    f = [lambda x: -np.cos(3.0 * x), lambda x: np.cos(3.0 * x)]
    theta = {
        "theta": lambda x: 0.5 * c * (x - xs) ** 2,
        "theta_deriv": lambda x: c * (x - xs),
    }

    def expression():
        cm = mpmath.mpf(TEST12C_C)
        xm = mpmath.mpf(TEST12C_XS)
        total = sum(
            mpmath.exp(1j * beta * xm)
            * _gauss_integral(1j * cm / 2, 1j * beta, -xm, 1 - xm)
            for beta in (3, -3)
        )
        return mpmath.re((1 + 1j) / 2 * total)

    # atol is 1e-14 of the integral (5.1e-5), as 07 section 2's runs set it
    return f, theta, dict(atol=5.1e-19, rtol=TEST12C_RTOL), expression


# 12d: an untrusted leaf at the cap. A narrow spike, f = (0, e^{-((x - 0.4)/1e-3)^2}), on a weakly
# oscillatory phase theta = 0.1 x over [0, 1], at depth_max 0: the root is a fallback leaf whose
# nested pair misses the spike (V_P and V_L + V_R agree to ~1e-8), while its grandchildren's nodes
# catch it, so r >> 1. I = Re[e^{0.1 i x0} int_{-x0}^{1-x0} e^{-(u/w)^2 + 0.1 i u} du].
TEST12D_X0 = 0.4
TEST12D_WIDTH = 1e-3
TEST12D_SPAN = (0.0, 1.0)


def test12d_problem():
    x0, w = TEST12D_X0, TEST12D_WIDTH
    f = [lambda x: 0.0, lambda x: math.exp(-(((x - x0) / w) ** 2))]
    theta = {"theta": lambda x: 0.1 * x, "theta_deriv": lambda x: 0.1 + 0.0 * x}

    def expression():
        xm = mpmath.mpf(TEST12D_X0)
        wm = mpmath.mpf(TEST12D_WIDTH)
        k = mpmath.mpf("0.1")
        return mpmath.re(
            mpmath.exp(1j * k * xm) * _gauss_integral(-1 / wm**2, 1j * k, -xm, 1 - xm)
        )

    return f, theta, dict(atol=1e-6, rtol=1e-6, depth_max=0), expression


# 12e: the references for the problems test_levin_quadrature.py drives
# (test_levin_roundoff_exit.levin_quadrature_problems()), by name. Every one has a closed form; the
# two that raise have none.
def _exp_sin_closed_form(omega, a, b):
    """int_a^b e^{-x} sin(omega x) dx."""
    w = mpmath.mpf(omega)

    def F(x):
        x = mpmath.mpf(x)
        return (
            -mpmath.exp(-x) * (mpmath.sin(w * x) + w * mpmath.cos(w * x)) / (1 + w * w)
        )

    return F(b) - F(a)


def _exp_sin_plus_const_cos(omega):
    """int_0^1 e^{-x} sin(omega x) + 0.3 cos(omega x) dx."""
    return _exp_sin_closed_form(omega, 0, 1) + mpmath.mpf("0.3") * mpmath.sin(
        mpmath.mpf(omega)
    ) / mpmath.mpf(omega)


def levin_quadrature_references():
    # the spans are the doubles the problems pass: 1/3, 7/3 and 1.0003 as floats, exactly
    third = mpmath.mpf(1.0 / 3.0)
    seven_thirds = mpmath.mpf(7.0 / 3.0)
    refs = {
        "SinIntegral": lambda: -mpmath.cos(mpmath.mpf(50000)) + mpmath.cos(1),
        "CosIntegral": lambda: mpmath.sin(mpmath.mpf(500000)) - mpmath.sin(1),
        "SincIntegral": lambda: mpmath.si(mpmath.mpf(10000)) - mpmath.si(100),
        "vectorized_sinc": lambda: mpmath.si(mpmath.mpf(10000)) - mpmath.si(100),
        "input_validation_order_clamp": lambda: mpmath.cos(1) - mpmath.cos(50),
        "lstsq_path": lambda: _exp_sin_plus_const_cos(10 * mpmath.pi),
        "direct_solve_path": lambda: _exp_sin_plus_const_cos(30 * mpmath.pi),
        "converged_flag_bump": lambda: test12a_problem()[3](),
        "stationary_phase": lambda: mpmath.im(
            mpmath.exp(-mpmath.mpf(1) / 2 + 1j * mpmath.mpf(10) ** 6 / 4)
            * _gauss_integral(
                -1j * mpmath.mpf(10) ** 6,
                mpmath.mpf(-1),
                -mpmath.mpf(1) / 2,
                mpmath.mpf(1) / 2,
            )
        ),
        "mode_filter_loose": lambda: _exp_sin_closed_form(10**6, 1, mpmath.mpf(1.0003)),
        "mode_filter_tight": lambda: _exp_sin_closed_form(10**6, 1, mpmath.mpf(1.0003)),
        "reversed_span_forward": lambda: (mpmath.cos(100) - mpmath.cos(500)) / 100,
        "reversed_span_reversed": lambda: -(mpmath.cos(100) - mpmath.cos(500)) / 100,
        "generic_m4_basis": lambda: (mpmath.cos(1) - mpmath.cos(50))
        + (mpmath.sin(100) - mpmath.sin(2)) / 2,
    }
    for lbda in (10, 100, 1000):
        refs[f"GRZIntegral({lbda:g})"] = lambda l=lbda: (
            2 / mpmath.mpf(l)
        ) * mpmath.sin(mpmath.pi * mpmath.mpf(l) / 4)
    for omega, label in ((10**6, "1e+06"), (10**12, "1e+12")):
        for kind in ("raw", "reduced"):
            refs[f"roundoff_floor_{kind}({label})"] = (
                lambda w=omega: _exp_sin_closed_form(w, third, seven_thirds)
            )
    for name in (
        "theta_abserr_scalar",
        "theta_abserr_callable",
        "theta_mod_2pi_path_only",
        "theta_deriv_path_only",
    ):
        refs[name] = lambda: _exp_sin_closed_form(10**6, third, seven_thirds)
    return refs


class TestLevinGlobalBudget(unittest.TestCase):

    def _assert_honest(self, data, ref, ref_err, label):
        true_err = abs(data["value"] - ref)
        # the reference's own error is negligible against what is asserted on it
        self.assertLess(ref_err, 1e-6 * max(true_err, data["abserr"], 1e-300), label)
        # the value lies within its declared abserr of the reference ...
        self.assertLessEqual(
            abs(data["value"] - ref),
            data["abserr"],
            f"{label}: value {data['value']!r} is not within abserr {data['abserr']:.3e} of "
            f"the reference {ref!r}",
        )
        # ... and abserr is at least the true error
        self.assertGreaterEqual(data["abserr"], true_err, label)
        return true_err

    def test_cancellation_fallback_branch(self):
        """
        Test 1. Family (R) with cancelling amplitudes, x_s = 0.7236, c = 1e4, rtol 1e-7: I =
        5.785e-5, and rtol |I| = 5.79e-12.

        At the parent (3f2a36d; parent_demo.txt) each region is accepted against its own value:
        8 regions (2 fallback), depth 5, 27 solves, sum_i |v_i| = 137 |I|:
            value  5.7851520051960546e-05
            abserr 1.6051161810283487e-11  (2.77 rtol |I|)
            converged False
        against a true error of 4.15e-12 = 0.72 rtol |I|: the call missed -- it met the request
        and said it had not.

        Under the global budget, with the trust check: 17 leaves (11 fallback), depth 7, 135
        solves, abserr 1.77e-12, converged, true error 1.16e-12 = 0.20 rtol |I|, honest.
        """
        f, theta, kwargs, expression = test1_problem()
        ref, ref_err = _reference(expression)

        new = _run(TEST1_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        true_err = self._assert_honest(new, ref, ref_err, "test 1")
        self.assertLessEqual(true_err, max(TEST1_ATOL, TEST1_RTOL * abs(ref)))
        # the leaves round x_s are fallback leaves, and the leaves' values cancel in the total
        self.assertGreater(new["num_simple_regions"], 0)
        self.assertGreaterEqual(
            _sum_abs_leaf_values(new, f, theta, TEST1_RTOL), 100.0 * abs(ref)
        )

    def test_cancellation_levin_branch(self):
        """
        Test 2. f = (1/x, 0), theta = 100.5 pi x on [1, 100.004975...], order 16, rtol 1e-8: I =
        1.003e-5, rtol |I| = 1.0e-13.

        At the parent (3f2a36d): 7 regions, every one Levin, depth 6, 27 solves; sum_i |v_i| =
        310 |I|:
            value  1.0029945228158706e-05
            abserr 1.0088575122850702e-13  (1.006 rtol |I|)
            converged False
        against a true error of 1.0017e-13 = 0.9987 rtol |I|: a miss, by a narrow margin (06c's
        log records the search for this problem).

        Under the global budget, with the trust check: 8 Levin leaves, depth 7, 63 solves, abserr
        1.13e-15, converged, true error 1.2e-16, honest.
        """
        f, theta, kwargs, expression = test2_problem()
        ref, ref_err = _reference(expression)

        new = _run(TEST2_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        true_err = self._assert_honest(new, ref, ref_err, "test 2")
        self.assertLessEqual(true_err, max(TEST2_ATOL, TEST2_RTOL * abs(ref)))
        self.assertEqual(new["num_simple_regions"], 0)
        self.assertGreaterEqual(
            _sum_abs_leaf_values(new, f, theta, TEST2_RTOL, TEST2_ORDER),
            100.0 * abs(ref),
        )

    def test_inflated_floor_is_chased(self):
        """
        Test 3 ([06b-resolved-region-keeps-a-floor-bisection-would-cut]). theta = 1000 e^x on
        [0, 20], f = (1, 0), rtol 1e-7: I = 5.632e-4, rtol |I| = 5.63e-11.

        At the parent (3f2a36d): the whole span is accepted resolved at depth 0 (residual
        2.09e-11, relerr 3.7e-8), and its eq. (151) floor is the call's abserr:
            value  0.00056320480676449829
            abserr 9.703303918195807e-07
            1 region, converged False
        against a true error of 2.09e-11 = 0.37 rtol |I|.

        Under the global budget that floor is the largest error, and its halves' floors sum to
        far less than it, so it is bisected; with the trust check, 3 leaves, depth 2, 23 solves,
        abserr 2.22e-11, converged, true error 1.1e-14.
        """
        f, theta, kwargs, expression = test3_problem()
        ref, ref_err = _reference(expression)

        new = _run(TEST3_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        self.assertLessEqual(new["abserr"], TEST3_RTOL * abs(ref))
        self._assert_honest(new, ref, ref_err, "test 3")
        self.assertGreaterEqual(new["max_depth"], 1)

    def test_floor_still_stops_refinement(self):
        """
        Test 4. 06b's test 2 problems, where bisection cannot lower the floor: (a) constant
        theta', (b) theta' = 5e5 e^{x/2}, at tolerances the floor binds. Each leaf the budget
        needs is floor-limited at depth 0, and flagged phase-limited; nothing reaches depth_max.
        """
        for problem, span, label in (
            (_test2a_problem(), TEST2A_SPAN, "test 4 (06b 2a)"),
            (_test2b_problem(), TEST2B_SPAN, "test 4 (06b 2b)"),
        ):
            f, theta, kwargs, theta_deriv, expression = problem
            ref, ref_err = _reference(expression)
            new = _run(span, f, theta, **kwargs)
            self.assertGreaterEqual(new["num_phase_limited_regions"], 1, label)
            self.assertTrue(new["phase_limited"], label)
            depth_max = kwargs.get("depth_max", 20)
            self.assertLess(new["max_depth"], depth_max, label)
            self.assertTrue(
                all(r.depth < depth_max for r in new["regions"]),
                label,
            )
            self._assert_honest(new, ref, ref_err, label)

    def test_unattainable_request_terminates(self):
        """
        Test 5. f = (|x - 1/3|, 0), theta = 100 x on [0, 1], rtol 1e-12, depth_max 6. The kink
        at 1/3 holds the total above the request at every depth the call may reach. The call
        returns unconverged; every leaf is at depth_max or floor-limited (the loop chose every
        refinable leaf before it stopped), and the result is honest.
        """
        f, theta, kwargs, expression = test5_problem()
        ref, ref_err = _reference(expression)

        new = _run(TEST5_SPAN, f, theta, **kwargs)
        self.assertFalse(new["converged"])
        self.assertEqual(new["max_depth"], TEST5_DEPTH_MAX)
        for region in new["regions"]:
            self.assertTrue(
                region.depth >= TEST5_DEPTH_MAX or region.phase_limited,
                f"test 5: {region}",
            )
        # the leaf holding the kink is one at depth_max
        kink = [r for r in new["regions"] if r.start < 1.0 / 3.0 < r.end]
        self.assertEqual(len(kink), 1)
        self.assertEqual(kink[0].depth, TEST5_DEPTH_MAX)
        self._assert_honest(new, ref, ref_err, "test 5")

    def test_infinite_floor_is_chased(self):
        """
        Test 6. theta = 1e4 (x - x^2) on [0, 1], f = (e^-x, 0), theta' by spectral
        differentiation: x = 1/2 is a node of every leaf touching it.

        At the parent (3f2a36d): 12 regions, depth 6; the two regions with an endpoint at 1/2
        are accepted on their residuals with an eq. (151) floor of +inf, so the call's abserr is
        +inf (a true error of 7.9e-15).

        Under the global budget a leaf with an infinite error is chosen before every finite one
        and bisected until its floor is finite (once the leaf is small enough that theta''s
        differentiation noise at 1/2 is no longer below the noise floor's ratio): abserr is
        finite, and honest. The test allows the other outcome the prompt names -- abserr infinite
        only with a leaf at depth_max -- and shows which happened. Measured (parent_demo.txt): 811
        leaves (806 fallback), depth 15, 3,243 solves, abserr 3.91e-7 against a true error of
        1.8e-16; unconverged, every leaf floor-limited, the floors near 1/2 (where G1/G0 is large)
        summing above the request of 1.06e-12.
        """
        f, theta, kwargs, expression = test6_problem()
        ref, ref_err = _reference(expression)

        new = _run(TEST6_SPAN, f, theta, **kwargs)
        depth_max = kwargs.get("depth_max", 20)
        if not math.isfinite(new["abserr"]):
            self.assertTrue(
                any(
                    r.depth >= depth_max and not math.isfinite(r.total_err)
                    for r in new["regions"]
                )
            )
        # at this problem the floor becomes finite well before depth_max
        self.assertTrue(math.isfinite(new["abserr"]))
        self.assertTrue(all(math.isfinite(r.total_err) for r in new["regions"]))
        self._assert_honest(new, ref, ref_err, "test 6")

    def test_total_near_zero(self):
        """
        Test 7. tanh(30 x) cos(20 x) on [-1, 1] is odd, so its integral is zero and the budget is
        atol = 1e-12: abserr at most atol, converged, and the value within atol of 0.
        """
        f, theta, kwargs = test7_problem()
        new = _run(TEST7_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        self.assertLessEqual(new["abserr"], TEST7_ATOL)
        self.assertLessEqual(abs(new["value"]), TEST7_ATOL)
        self.assertLessEqual(abs(new["value"]), new["abserr"])

    def test_determinism(self):
        """Test 8. Two runs of tests 1 and 3: value, abserr and every region bit-identical."""
        for problem, span in (
            (test1_problem(), TEST1_SPAN),
            (test3_problem(), TEST3_SPAN),
        ):
            f, theta, kwargs, _ = problem
            first = _run(span, f, theta, **kwargs)
            second = _run(span, f, theta, **kwargs)
            self.assertEqual(_call_signature(first), _call_signature(second))

    def test_trust_check_bump(self):
        """
        Test 12a. test_converged_flag's Gaussian bump at atol 1e-10 (integral ~2.1e-215).

        At the parent (3f2a36d): 4 regions, depth 3, value -3.437e-12, abserr 1.888e-11,
        converged, honest. Under the global budget without the trust check (79ae8c7, never
        pushed): 3 leaves, value -8.563e-11, abserr 8.219e-11, converged -- dishonest by 4 %, the
        leaf [0.75, 1.5] carrying a residual of 8.2e-11 while its halves carry a further 3.4e-12 in
        the same direction ([06c-budget-can-stop-on-a-pre-asymptotic-residual]).

        With the check: that leaf's r = delta2 / delta1 = 0.040, so it declares
        8.219e-11 / (1 - r) = 8.564e-11 against a true error of 8.563e-11 -- honest, by a margin
        of about 1e-4 relative; the value is unchanged. Converged, within the request.
        """
        f, theta, kwargs, expression = test12a_problem()
        ref, ref_err = _reference(expression)
        new = _run(TEST12A_SPAN, f, theta, **kwargs)
        true_err = self._assert_honest(new, ref, ref_err, "test 12a")
        if new["converged"]:
            self.assertLessEqual(
                true_err, max(TEST12A_ATOL, NEW_DEFAULT_RTOL * abs(ref)), "test 12a"
            )
        self.assertEqual(new["num_untrusted_regions"], 0)
        self.assertTrue(all(r.trusted is not None for r in new["regions"]))

    def test_trust_check_before_resolution(self):
        """
        Test 12b. SincIntegral, int_1^100 sin(100 x)/x, rtol 1e-10, order 12: the problem on
        which the withdrawn stall rule stopped early. Its leaves' r are recorded in 06c's log; along
        the leftmost line the check passes once the leaves are resolved. Converged and honest.
        """
        f, theta, kwargs, expression = test12b_problem()
        ref, ref_err = _reference(expression)
        new = _run(TEST12B_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        self._assert_honest(new, ref, ref_err, "test 12b")
        self.assertTrue(all(r.trusted for r in new["regions"]))
        self.assertTrue(all(r.convergence_ratio < 1.0 for r in new["regions"]))

    def test_trust_check_stationary_point_line(self):
        """
        Test 12c. Family (R) with cancelling amplitudes, x_s = 0.382, c = 1e5, rtol 1e-7, theta'
        supplied: Levin leaves straddling the stationary point have estimates that rise under
        bisection until they are small enough for the fallback. The parent (3f2a36d) converges
        here (11 regions, true error 1.07e-12, abserr 1.39e-12); the withdrawn stall rule (b2)
        stalled it at depth 3, dishonest. The trust check must not: converged and honest.
        """
        f, theta, kwargs, expression = test12c_problem()
        ref, ref_err = _reference(expression)
        new = _run(TEST12C_SPAN, f, theta, **kwargs)
        self.assertTrue(new["converged"])
        true_err = self._assert_honest(new, ref, ref_err, "test 12c")
        self.assertLessEqual(true_err, TEST12C_RTOL * abs(ref))

    def test_trust_check_untrusted_at_cap(self):
        """
        Test 12d. A narrow spike on a weakly oscillatory phase, depth_max 0: the root's nested
        pair misses the spike, so its own estimate (8e-9) is far below the request and the budget
        passes; the check then finds delta2 >> delta1 (r ~ 2.6e3). The root cannot be refined, so
        it declares max(residual, delta1 + delta2), is flagged, and the call is not converged.
        Honest.
        """
        f, theta, kwargs, expression = test12d_problem()
        ref, ref_err = _reference(expression)
        new = _run(TEST12D_SPAN, f, theta, **kwargs)
        self.assertFalse(new["converged"])
        self.assertEqual(new["num_untrusted_regions"], 1)
        (region,) = new["regions"]
        self.assertIs(region.trusted, False)
        self.assertEqual(region.depth, 0)
        self.assertGreaterEqual(region.convergence_ratio, 1.0)
        self.assertEqual(
            region.declared_resolution,
            max(region.abserr, region.delta1 + region.delta2),
        )
        self._assert_honest(new, ref, ref_err, "test 12d")

    def test_trust_check_nothing_honest_becomes_dishonest(self):
        """
        Test 12e. Every problem test_levin_quadrature.py drives that has a reference (25: the two
        that raise have none). At the parent (3f2a36d; parent_demo.txt) every one is honest, so
        every one must be honest under the new code.
        """
        references = levin_quadrature_references()
        for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
            if raises is not None:
                continue
            with self.subTest(problem=name):
                ref, ref_err = _reference(references[name])
                if basis is None:
                    basis_object = _Basis_SinCos(theta)
                else:
                    basis_object = basis
                new = _adaptive_levin(span, f, basis_object, **kwargs)
                self._assert_honest(new, ref, ref_err, f"test 12e {name}")

    def test_nan_error_is_ordered_as_infinite(self):
        """
        Section 1 item 2: a NaN leaf error is read as +inf -- chosen first, and the budget cannot
        be met while it is present. No path is known to produce one; the rule is tested directly.
        """
        self.assertEqual(_error_for_ordering(float("nan")), float("inf"))
        self.assertEqual(_error_for_ordering(float("inf")), float("inf"))
        self.assertEqual(_error_for_ordering(1.5e-300), 1.5e-300)
        self.assertEqual(_error_for_ordering(0.0), 0.0)

    def test_running_sum_is_exact(self):
        """
        Section 1 item 3: the loop's running sums are exact. Adding and removing terms many
        orders above what remains leaves the remainder correctly rounded.
        """
        acc = _ExactSum()
        big = [1.0e-3 * (1.0 + k / 7.0) for k in range(1000)]
        for x in big:
            acc.add(x)
        acc.add(3.0e-19)
        for x in reversed(big):
            acc.add(-x)
        self.assertEqual(acc.total(), 3.0e-19)
        # a plain float accumulator does not hold it
        plain = 0.0
        for x in big:
            plain += x
        plain += 3.0e-19
        for x in reversed(big):
            plain -= x
        self.assertNotEqual(plain, 3.0e-19)


if __name__ == "__main__":
    unittest.main()
