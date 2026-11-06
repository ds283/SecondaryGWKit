"""
The trusted leaf's value, and slow leaves (prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md,
section 2 tests 1-3).

06c's trust check solves a leaf's halves and quarters, forms delta1 = |V_P - H|, delta2 = |H - Q| and
r = delta2 / delta1, and declares max(residual, delta1) / (1 - r). Since 06g:
  (a) a trusted leaf returns Q = (V_LL + V_LR) + (V_RL + V_RR), the finest value computed, in place
      of V_P, with the same declared error; the running value sum follows exactly;
  (b) a leaf is trusted only if r < 1/4, or r < 1 with delta1 at most its own round-off floor
      (since prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md, at most the noise
      yardstick nu_B, which adds the halves' floors; trust_test() restates the current rule).

The parent is the dispatch commit's module (PARENT_SHA, read with `git show` and executed under
another name). References are closed forms evaluated with mpmath; each reference's own error is its
30-digit value's difference from its 40-digit value, and is asserted negligible. The numbers quoted
below are docs/levin-phase-floor/06g/calibration/out/cells.txt's.

No Ray and no datastore.
"""

import math
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import mpmath
import numpy as np

import AdaptiveLevin.levin_quadrature as lq

PARENT_SHA = "4f4f9ed2a31fb7437fcfca01263d260eebaff842"
_PARENT_NAME = "levin_quadrature_parent_4f4f9ed"
_REPO = Path(__file__).resolve().parents[2]


def load_parent():
    """
    The dispatch commit's module: AdaptiveLevin/levin_quadrature.py at PARENT_SHA, read with `git
    show` and executed as a module of another name, once per process.
    """
    module = sys.modules.get(_PARENT_NAME)
    if module is None:
        source = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO),
                "show",
                f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        module = types.ModuleType(_PARENT_NAME)
        module.__file__ = (
            f"<git show {PARENT_SHA[:7]}:AdaptiveLevin/levin_quadrature.py>"
        )
        sys.modules[_PARENT_NAME] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def run(module, span, f, theta, **kwargs):
    """module._adaptive_levin() on a (sin, cos) phase dict, as adaptive_levin_sincos() calls it."""
    return module._adaptive_levin(span, f, module._Basis_SinCos(theta), **kwargs)


def reference(expression):
    """expression() at 30 and at 40 digits: the 40-digit value, and their difference as its error."""
    with mpmath.workdps(30):
        value30 = expression()
    with mpmath.workdps(40):
        value40 = expression()
    with mpmath.workdps(40):
        ref_err = float(abs(value30 - value40))
    return float(value40), ref_err


def gauss_integral(gamma, beta, a, b):
    """int_a^b exp(gamma u^2 + beta u) du for complex gamma != 0, in closed form (mpmath)."""
    s = mpmath.sqrt(-gamma)
    c = beta / (2 * gamma)
    return (
        mpmath.exp(-(beta**2) / (4 * gamma))
        * mpmath.sqrt(mpmath.pi)
        / (2 * s)
        * (mpmath.erf(s * (b + c)) - mpmath.erf(s * (a + c)))
    )


def exp_sin_closed_form(omega, a, b):
    """int_a^b e^{-x} sin(omega x) dx, a and b mpf."""
    w = mpmath.mpf(omega)

    def F(x):
        return (
            -mpmath.exp(-x) * (mpmath.sin(w * x) + w * mpmath.cos(w * x)) / (1 + w * w)
        )

    return F(b) - F(a)


@contextmanager
def recording_regions():
    """
    Within the block, every used_interval the driver makes is recorded, in the order made: every
    leaf of the call, those later bisected included, so that every checked leaf is seen.
    """
    made = []
    original = lq.used_interval

    class Recorded(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    with mock.patch.object(lq, "used_interval", Recorded):
        yield made


@contextmanager
def recording_exact_sums():
    """
    Within the block, every _ExactSum the driver makes is recorded, in the order made. _adaptive_levin()
    makes three, in this order: the errors, the values and the node errors.
    """
    made = []
    original = lq._ExactSum

    class Recorded(original):
        __slots__ = ()

        def __init__(self):
            super().__init__()
            made.append(self)

    with mock.patch.object(lq, "_ExactSum", Recorded):
        yield made


def quarters_sum(region, f, theta, kwargs):
    """
    Q for a returned region, from its four quarters solved afresh as the driver solves them (the
    leaf's break point (end + start) / 2, and each half's), summed (LL + LR) + (RL + RR).
    """
    basis = lq._Basis_SinCos(theta)
    order = kwargs.get("chebyshev_order", lq.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kwargs.get("rtol", lq.DEFAULT_LEVIN_RELTOL)
    a, b = region.start, region.end
    c = (b + a) / 2.0
    cL = (c + a) / 2.0
    cR = (b + c) / 2.0

    def V(span):
        return lq._adaptive_levin_subregion(
            span, f, basis, None, chebyshev_order=order, rtol=rtol
        )["value"]

    return (V((a, cL)) + V((cL, c))) + (V((c, cR)) + V((cR, b)))


def own_value(region, f, theta, kwargs):
    basis = lq._Basis_SinCos(theta)
    return lq._adaptive_levin_subregion(
        (region.start, region.end),
        f,
        basis,
        None,
        chebyshev_order=kwargs.get("chebyshev_order", lq.DEFAULT_LEVIN_CHEBSHEV_ORDER),
        rtol=kwargs.get("rtol", lq.DEFAULT_LEVIN_RELTOL),
    )["value"]


def ascending_value(data):
    """The call's value as the driver sums it: the regions' values, in ascending x."""
    total = 0.0
    for region in data["regions"]:
        total = total + region.value
    return total


def trust_test(region, f, theta, kwargs):
    """
    The driver's trust test, restated: r < 1/4, or r < 1 with delta1 at most the noise yardstick
    nu_B of _trust_noise_yardstick() (since prompts/levin-phase-floor/06m-signed-yardstick-in-the-
    driver.md; from 06g to 06m it was the leaf's own floor). nu_B is formed from the leaf's halves,
    solved afresh as the driver solves them, at its break point (end + start) / 2.
    """
    r = region.convergence_ratio
    if r < 0.25:
        return True
    if not r < 1.0:
        return False
    basis = lq._Basis_SinCos(theta)
    order = kwargs.get("chebyshev_order", lq.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kwargs.get("rtol", lq.DEFAULT_LEVIN_RELTOL)
    a, b = region.start, region.end
    c = (b + a) / 2.0

    def solve(span):
        return lq._adaptive_levin_subregion(
            span, f, basis, None, chebyshev_order=order, rtol=rtol
        )

    nu_B = lq._trust_noise_yardstick(
        region.phase_err, region.type == "direct", solve((a, c)), solve((c, b))
    )
    return region.delta1 <= nu_B


def r_class(region):
    """Test 3's descriptive class of a checked leaf: its r band and delta1 against its floor."""
    r = region.convergence_ratio
    if r < 1.0 / 16.0:
        band = "r<1/16"
    elif r < 0.25:
        band = "1/16<=r<1/4"
    elif r < 1.0:
        band = "1/4<=r<1"
    else:
        band = "r>=1"
    return band, "d1<=phi" if region.delta1 <= region.phase_err else "d1>phi"


# ---------------------------------------------------------------------------------------------
# Test 1's cells. Each: (name, span, f, theta, kwargs, reference expression, identical), where
# identical is True on a cell where (b), (c) and (d) do not apply -- every leaf Levin, no checked
# leaf at the parent with 1/4 <= r < 1 above its floor, and any exit firing the parent's own -- so
# that abserr, converged and every region's span, depth and error are the parent's bit for bit.
# The request on each is chosen with the error away from rtol |V|, which the value's move could
# cross.
def _sqrt_problem(rtol):
    return (
        f"sqrt_fallback({rtol:g})",
        (0.0, 1.0),
        [lambda x: math.sqrt(x + 1e-3), lambda x: 0.0],
        {"theta": lambda x: 2.0 * x, "theta_deriv": lambda x: 2.0 + 0.0 * x},
        dict(atol=1e-20, rtol=rtol),
        lambda: mpmath.quad(
            lambda x: mpmath.sqrt(x + mpmath.mpf("1e-3")) * mpmath.sin(2 * x),
            [0, mpmath.mpf("0.01"), mpmath.mpf("0.1"), 1],
        ),
        False,
    )


def test1_cells():
    third = 1.0 / 3.0
    seven_thirds = 7.0 / 3.0
    return [
        # Levin, one leaf: e^-x sin(1e3 x) on [1/3, 7/3]
        (
            "exp_sin_1e3",
            (third, seven_thirds),
            [lambda x: math.exp(-x), lambda x: 0.0],
            {"theta": lambda x: 1e3 * x, "theta_deriv": lambda x: 1e3 + 0.0 * x},
            dict(atol=1e-20, rtol=1e-12),
            lambda: exp_sin_closed_form(
                1000, mpmath.mpf(third), mpmath.mpf(seven_thirds)
            ),
            True,
        ),
        # Levin, leaves at depth 2: 06c's test 3, sin(1000 e^x) on [0, 20]
        (
            "si_1000_exp",
            (0.0, 20.0),
            [lambda x: 1.0, lambda x: 0.0],
            {
                "theta": lambda x: 1000.0 * np.exp(x),
                "theta_deriv": lambda x: 1000.0 * np.exp(x),
            },
            dict(atol=1e-15, rtol=1e-7),
            lambda: mpmath.si(1000 * mpmath.exp(20)) - mpmath.si(1000),
            True,
        ),
        # Levin, leaves to depth 4, two of them floor-limited at the parent and here alike
        (
            "sinc_1e3",
            (1.0, 10.0),
            [lambda x: 1.0 / x, lambda x: 0.0],
            {"theta": lambda x: 1e3 * x, "theta_deriv": lambda x: 1e3 + 0.0 * x},
            dict(atol=1e-20, rtol=1e-12),
            lambda: mpmath.si(10000) - mpmath.si(1000),
            True,
        ),
        # Levin, oscillatory: 06c's test_converged_flag Gaussian bump, where (b) applies
        (
            "gaussian_bump",
            (0.0, 3.0),
            [lambda x: math.exp(-400.0 * (x - 1.1) ** 2), lambda x: 0.0],
            {"theta": lambda x: 3.0e4 * x},
            dict(atol=1e-10),
            lambda: mpmath.im(
                mpmath.exp(1j * 30000 * mpmath.mpf(1.1))
                * gauss_integral(
                    mpmath.mpf(-400),
                    1j * 30000,
                    -mpmath.mpf(1.1),
                    3 - mpmath.mpf(1.1),
                )
            ),
            False,
        ),
        # fallback, smooth amplitude with a near-singular derivative, to depth 2 and to depth 4
        _sqrt_problem(1e-8),
        _sqrt_problem(1e-10),
        # fallback, smooth: a Runge amplitude on a weak phase
        (
            "runge_fallback",
            (-1.0, 1.0),
            [lambda x: 1.0 / (1.0 + 400.0 * x * x), lambda x: 0.0],
            {"theta": lambda x: x + 1.0, "theta_deriv": lambda x: 1.0 + 0.0 * x},
            dict(atol=1e-20, rtol=1e-8),
            lambda: mpmath.quad(
                lambda x: mpmath.sin(x + 1) / (1 + 400 * x * x), [-1, 0, 1]
            ),
            False,
        ),
    ]


# Measured true errors, parent -> 06g, and their ratio (cells.txt): the smaller-by factor test 1
# requires is asserted on the cells where it is large.
TEST1_IMPROVED = {"gaussian_bump": 100.0, "sqrt_fallback(1e-08)": 100.0}


def test3_sinc_problem():
    """06c's test 12b: SincIntegral, int_1^100 sin(100 x)/x, rtol 1e-10, order 12."""
    return (
        (1.0, 100.0),
        [lambda x: 1.0 / x, lambda x: 0.0],
        {"theta": lambda x: 100.0 * x},
        dict(atol=1e-15, rtol=1e-10, chebyshev_order=12),
        lambda: mpmath.si(mpmath.mpf(10000)) - mpmath.si(mpmath.mpf(100)),
    )


# The parent's slow leaf on test 3's cell (cells.txt): [1.38671875, 1.7734375] at depth 8, Levin,
# r = 0.9107, delta1 = 1.155e-15 against its floor 1.317e-16 -- trusted at the parent, declaring
# 1.155e-15 / (1 - 0.9107) = 1.29e-14.
TEST3_SLOW_LEAF = (1.38671875, 1.7734375, 8)


class _HonestyMixin:

    def assert_honest(self, data, ref, ref_err, label):
        true_err = abs(data["value"] - ref)
        self.assertLess(ref_err, 1e-6 * max(true_err, data["abserr"], 1e-300), label)
        self.assertLessEqual(true_err, data["abserr"], label)
        return true_err


class TestTrustedLeafReturnsQuarters(_HonestyMixin, unittest.TestCase):

    def _check_values(self, data, f, theta, kwargs, label):
        """Every returned region's value is Q if trusted, V_P otherwise, and V_P is recorded."""
        for region in data["regions"]:
            own = own_value(region, f, theta, kwargs)
            self.assertEqual(region.value_own, own, f"{label}: {region}")
            if region.trusted:
                self.assertEqual(region.value_kind, "quarters", f"{label}: {region}")
                self.assertEqual(
                    region.value, quarters_sum(region, f, theta, kwargs), label
                )
            else:
                self.assertEqual(region.value_kind, "own", f"{label}: {region}")
                self.assertEqual(region.value, own, label)
        self.assertEqual(data["value"], ascending_value(data), label)

    def test_trusted_leaf_returns_quarters(self):
        """
        Test 1. Seven cells, Levin and fallback, smooth and oscillatory, leaves to depth 4. On each:
        every trusted leaf's value is its quarters sum bit for bit, and records V_P; every other
        leaf returns V_P; the call's value is the ascending sum of the returned values. Where
        (b), (c) and (d) do not apply, abserr, converged and every region's span, depth and error
        are the parent's bit for bit. The true error is at most the parent's on every cell, and
        honest; it is smaller by the factors cells.txt records:

            exp_sin_1e3            5.42e-19 -> 3.25e-19   (1.67x)
            si_1000_exp            1.095e-14 -> 1.096e-15  (9.99x)
            sinc_1e3               1.08e-18 -> 5.42e-19   (2x)
            gaussian_bump          8.563e-11 -> 1.24e-13  (691x; (b) refines it too, 5 -> 10 leaves)
            sqrt_fallback(1e-08)   3.06e-11 -> 2.91e-14   (1,051x)
            sqrt_fallback(1e-10)   2.91e-14 -> 0          (the value is the reference's double)
            runge_fallback         0 -> 0                 (Q equals V_P to the last bit)

        (true errors against the reference rounded to a double)
        """
        parent = load_parent()
        for name, span, f, theta, kwargs, expression, identical in test1_cells():
            with self.subTest(cell=name):
                ref, ref_err = reference(expression)
                new = run(lq, span, f, theta, **kwargs)
                old = run(parent, span, f, theta, **kwargs)
                self.assertFalse(new["node_limited"], name)
                self._check_values(new, f, theta, kwargs, name)
                self.assertTrue(
                    any(r.value_kind == "quarters" for r in new["regions"]), name
                )
                if identical:
                    self.assertEqual(new["abserr"], old["abserr"], name)
                    self.assertEqual(new["converged"], old["converged"], name)
                    self.assertEqual(
                        [
                            (r.start, r.end, r.depth, r.total_err)
                            for r in new["regions"]
                        ],
                        [
                            (r.start, r.end, r.depth, r.total_err)
                            for r in old["regions"]
                        ],
                        name,
                    )
                old_true = self.assert_honest(old, ref, ref_err, f"{name} parent")
                new_true = self.assert_honest(new, ref, ref_err, f"{name} 06g")
                self.assertLessEqual(new_true, old_true, name)
                if name in TEST1_IMPROVED:
                    self.assertGreaterEqual(old_true, TEST1_IMPROVED[name] * new_true)

    def test_noise_level_cell(self):
        """
        Not one of test 1's cells, and recorded because test 1's "at most the parent's" does not
        hold on it: sin(x + 1) / (1 + 25 x^2) on [-1, 1], rtol 1e-12, six fallback leaves whose
        delta1 are rounding noise (0 to 2.8e-17, r = 0 on four). There Q and V_P differ in their
        last bits, the true error (against the reference rounded to a double) moves from 0 to
        5.55e-17, one ulp of the value, and abserr, 1.07e-13, covers both 1,900 times over. Asserted: the value moves by at most the trusted leaves' summed
        delta1 (section 0.3), and both codes are honest.
        """
        span = (-1.0, 1.0)
        f = [lambda x: 1.0 / (1.0 + 25.0 * x * x), lambda x: 0.0]
        theta = {"theta": lambda x: x + 1.0, "theta_deriv": lambda x: 1.0 + 0.0 * x}
        kwargs = dict(atol=1e-20, rtol=1e-12)
        ref, ref_err = reference(
            lambda: mpmath.quad(
                lambda x: mpmath.sin(x + 1) / (1 + 25 * x * x), [-1, 0, 1]
            )
        )
        new = run(lq, span, f, theta, **kwargs)
        old = run(load_parent(), span, f, theta, **kwargs)
        self._check_values(new, f, theta, kwargs, "noise cell")
        moved = abs(new["value"] - old["value"])
        allowance = math.fsum(r.delta1 for r in new["regions"] if r.trusted)
        self.assertLessEqual(moved, allowance + 4 * np.spacing(abs(new["value"])))
        self.assert_honest(old, ref, ref_err, "noise cell parent")
        self.assert_honest(new, ref, ref_err, "noise cell 06g")


# ---------------------------------------------------------------------------------------------
# Test 2's stand-in: _adaptive_levin_subregion wrapped to return fixed values, because a trusted
# leaf later bisected or found floor-limited needs its budget to fall between its residual and its
# widened declaration, which no real cell found in cells.txt's search did. Each solve of [a, b] is
# a Levin solve with value h + K h^4, h = b - a: so a leaf's halves and quarters give
# delta1 = (7/8) K h^4, delta2 = (7/64) K h^4, r = 1/8 (trusted), its residual is delta1, and its
# declared error is delta1 / (1 - r) = K h^4. The floor is the caller's phi(h).
STANDIN_K = 1.0e-3


def _standin_solve(phi):
    def solve(x_span, f, BasisData, id_label, **kwargs):
        h = x_span[1] - x_span[0]
        floor = phi(h)
        return {
            "value": h + STANDIN_K * h**4,
            "p_sample": None,
            "p_ratios": [1.0, 1.0],
            "phase_span": 100.0,
            "phase_err": floor,
            "phase_err_roundoff": floor,
            "phase_err_declared": 0.0,
            "abserr_truncation": 0.0,
            "metadata": {"chebyshev_order": 16},
            "is_direct": False,
        }

    return solve


class TestValueSumFollowsLeaves(unittest.TestCase):

    def _run_recorded(self, span, f, theta, kwargs, solve=None):
        with recording_exact_sums() as sums, recording_regions() as made:
            if solve is None:
                data = run(lq, span, f, theta, **kwargs)
            else:
                with mock.patch.object(lq, "_adaptive_levin_subregion", solve):
                    data = run(lq, span, f, theta, **kwargs)
        self.assertEqual(len(sums), 3)
        value_sum = sums[1]
        # the running value sum is the exact sum of the returned values
        self.assertEqual(value_sum.total(), math.fsum(r.value for r in data["regions"]))
        self.assertEqual(data["value"], ascending_value(data))
        return data, made

    def test_checked_leaf_bisected(self):
        """
        Test 2 (a). 06c's Gaussian bump: (b) finds leaves untrusted after the check and bisects
        them (cells.txt: 17 leaves checked, 10 returned), and the value sum follows.
        """
        _, span, f, theta, kwargs, _, _ = test1_cells()[3]
        data, made = self._run_recorded(span, f, theta, kwargs)
        returned = {id(r) for r in data["regions"]}
        bisected_checked = [
            r for r in made if r.delta1 is not None and id(r) not in returned
        ]
        self.assertGreater(len(bisected_checked), 0)
        self.assertFalse(data["node_limited"])

    def test_trusted_leaf_bisected(self):
        """
        Test 2 (b), the stand-in, with a floor far below everything (1e-30): the root's residual
        (7/8) K meets the request 0.95 K, so it is checked and trusted, and returns Q; its declared
        error K then does not, so it is bisected, its Q leaving the value sum; its children meet
        it and return their own Q.
        """
        solve = _standin_solve(lambda h: 1e-30)
        kwargs = dict(atol=0.95 * STANDIN_K, rtol=0.0)
        data, made = self._run_recorded(
            (0.0, 1.0),
            [lambda x: 1.0, lambda x: 0.0],
            {"theta": lambda x: x},
            kwargs,
            solve,
        )
        root = [r for r in made if r.depth == 0]
        self.assertEqual(len(root), 1)
        self.assertIs(root[0].trusted, True)
        self.assertEqual(root[0].value_kind, "quarters")
        self.assertNotIn(id(root[0]), {id(r) for r in data["regions"]})
        self.assertEqual(data["num_regions"], 2)
        self.assertTrue(all(r.value_kind == "quarters" for r in data["regions"]))
        self.assertTrue(data["converged"])

    def test_trusted_leaf_found_floor_limited(self):
        """
        Test 2 (c), the stand-in, with the root's floor 0.9 K and its halves' 0.5 K each: the root
        is checked and trusted (its error max(7K/8, 0.9 K) meets the request 0.95 K) and returns
        Q; its declared error K then does not, it is chosen, its residual is within its floor, its
        halves' floors sum to 1.0 K (the halving test holds) and would leave the call above its
        request (the budget clause holds), so it is floor-limited: its value reverts to V_P and
        its declared error to its own, and the call stops converged on max(7K/8, 0.9 K).
        """
        solve = _standin_solve(lambda h: 0.9 * STANDIN_K if h == 1.0 else STANDIN_K * h)
        kwargs = dict(atol=0.95 * STANDIN_K, rtol=0.0)
        data, _ = self._run_recorded(
            (0.0, 1.0),
            [lambda x: 1.0, lambda x: 0.0],
            {"theta": lambda x: x},
            kwargs,
            solve,
        )
        (region,) = data["regions"]
        self.assertTrue(region.phase_limited)
        self.assertIsNone(region.trusted)
        self.assertAlmostEqual(region.convergence_ratio, 0.125, places=9)
        self.assertEqual(region.value_kind, "own")
        self.assertEqual(region.value, region.value_own)
        self.assertEqual(region.value, 1.0 + STANDIN_K)
        self.assertTrue(data["converged"])
        self.assertFalse(data["node_limited"])


class TestSlowLeaves(_HonestyMixin, unittest.TestCase):

    def test_slow_leaf_untrusted_and_refined(self):
        """
        Test 3, the first case. SincIntegral at rtol 1e-10, order 12 (06c's test 12b). At the parent
        (cells.txt) 9 Levin leaves, depth 8, 71 solves, converged, abserr 2.159e-13, true error
        5.2e-14; the leaf [1.38671875, 1.7734375] at depth 8 has r = 0.9107 and delta1 = 1.155e-15
        above its floor 1.317e-16, and is trusted. Under 06g it is untrusted, and refined before
        every trusted leaf: every other leaf of the parent's partition is returned as it was (no
        trusted leaf is bisected), and the 9 leaves that replace it lie inside it (17 leaves, 8 of
        them fallback, depth 12, 135 solves). Converged, abserr 2.074e-13, true error 2.1e-15:
        the call's error covers its true error.
        """
        span, f, theta, kwargs, expression = test3_sinc_problem()
        ref, ref_err = reference(expression)
        old = run(load_parent(), span, f, theta, **kwargs)
        (slow,) = [
            r for r in old["regions"] if (r.start, r.end, r.depth) == TEST3_SLOW_LEAF
        ]
        self.assertIs(slow.trusted, True)
        self.assertTrue(0.25 <= slow.convergence_ratio < 1.0)
        self.assertGreater(slow.delta1, slow.phase_err)
        self.assertAlmostEqual(slow.convergence_ratio, 0.9107, places=4)

        with recording_regions() as made:
            new = run(lq, span, f, theta, **kwargs)
        self.assertFalse(new["node_limited"])
        (checked,) = [r for r in made if (r.start, r.end, r.depth) == TEST3_SLOW_LEAF]
        self.assertIs(checked.trusted, False)
        self.assertEqual(checked.convergence_ratio, slow.convergence_ratio)
        new_keys = {(r.start, r.end, r.depth) for r in new["regions"]}
        self.assertNotIn(TEST3_SLOW_LEAF, new_keys)
        for r in old["regions"]:
            if (r.start, r.end, r.depth) != TEST3_SLOW_LEAF:
                self.assertIn((r.start, r.end, r.depth), new_keys)
        for r in new["regions"]:
            if (r.start, r.end, r.depth) not in {
                (o.start, o.end, o.depth) for o in old["regions"]
            }:
                self.assertTrue(
                    TEST3_SLOW_LEAF[0] <= r.start and r.end <= TEST3_SLOW_LEAF[1], r
                )
        self.assertTrue(new["converged"])
        self.assert_honest(new, ref, ref_err, "test 3 slow leaf")
        # every checked leaf, returned or not, is trusted exactly when the trust test says so
        for r in made:
            if r.delta1 is not None and not r.phase_limited:
                self.assertEqual(r.trusted, trust_test(r, f, theta, kwargs), r)

    def test_noise_leaf_trusted(self):
        """
        Test 3, the second case: on the same cell under 06g, the fallback leaf
        [1.604248046875, 1.62841796875] at depth 12 has r = 1/3 and delta1 = 1.04e-17 within its
        floor 4.04e-16 -- rounding noise -- and is trusted (cells.txt).
        """
        span, f, theta, kwargs, _ = test3_sinc_problem()
        new = run(lq, span, f, theta, **kwargs)
        noise = [
            r
            for r in new["regions"]
            if r.convergence_ratio is not None
            and 0.25 <= r.convergence_ratio < 1.0
            and r.delta1 <= r.phase_err
        ]
        self.assertGreater(len(noise), 0)
        for r in noise:
            self.assertIs(r.trusted, True, r)
            self.assertEqual(r.value_kind, "quarters", r)

    def test_gaussian_bump_honest(self):
        """
        Test 3: 06c's test_converged_flag Gaussian bump is honest. At the parent (cells.txt) 5
        leaves, abserr 8.564e-11 against a true error 8.563e-11, a margin of 1.6e-4; under 06g 10
        leaves, abserr 8.564e-11 against 1.24e-13.
        """
        _, span, f, theta, kwargs, expression, _ = test1_cells()[3]
        ref, ref_err = reference(expression)
        new = run(lq, span, f, theta, **kwargs)
        self.assert_honest(new, ref, ref_err, "bump")
        self.assertEqual(new["num_untrusted_regions"], 0)
        self.assertFalse(new["node_limited"])


if __name__ == "__main__":
    unittest.main()
