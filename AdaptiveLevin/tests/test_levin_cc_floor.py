"""
A round-off floor for the Clenshaw-Curtis rule (prompts/levin-phase-floor/06d-cc-roundoff-floor.md,
with its section 5 amendment).

From levin-refactor prompt 04 until 06d, _adaptive_levin_subregion_cc() formed its round-off floor
from Chen et al. eq. (151), the Levin collocation solve's conditioning bound, with its
1/min|theta'|. A fallback solve is a weighted sum of samples and solves no such system; on a region
holding a stationary point that floor grew as the region shrank, and was +inf with the point at a
node. 06d replaces it with the rule's own floor (_cc_roundoff_floor()):

    phi_CC = (n + 4) u S + u sum_k |w_k| |a_k| sum_i |f_ik|
             + sum_k |w_k| delta_k sum_i (|theta'| |f_ik| + |f'_ik|),
    S = sum_k |w_k| sum_i |f_ik b_ik|,

the rounding of the sum, of the argument handed to sin and cos, and of the nodes.

The parent (06c, code ad9a5f4, at the dispatch SHA 4bbeade) is reconstructed in-process by
_parent_cc(): the working tree's fallback solve with its "phase_err" replaced by what the parent
computed, eq. (151) through the same helpers (_levin_G0_G1(), _roundoff_floor(), unchanged on the
Levin branch) plus the unchanged declared-phase term. That this reproduces the parent's own module
bit for bit, on every problem test 6 drives, is shown by
docs/levin-phase-floor/06d/calibration/scripts/parent_demo.py against the parent's file (`git show`),
in docs/levin-phase-floor/06d/calibration/out/parent_demo.txt, which also holds every number the
docstrings below quote.

References are closed forms or mpmath at 30 digits; each reference's own error is estimated as its
difference from the same expression at 40 digits, and asserted negligible. Every accuracy test
asserts that the value lies within its declared error of the reference and that the declared error
is at least the true error.

No Ray and no datastore.
"""

import contextlib
import logging
import math
import unittest
from unittest import mock

import mpmath
import numpy as np

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.levin_quadrature import (
    DEFAULT_LEVIN_CHEBSHEV_ORDER,
    _UNIT_ROUNDOFF,
    _Basis_SinCos,
    _adaptive_levin,
    _adaptive_levin_subregion_cc,
    _cc_node_error_bound,
    _cc_weights,
    _declared_endpoint_phase_err,
    _history_improvement,
    _levin_G0_G1,
    _log_depth_history,
    _roundoff_floor,
    _sample_vectorized,
    adaptive_levin_sincos,
    chebyshev_matrices,
)
from AdaptiveLevin.tests.test_levin_global_budget import levin_quadrature_references
from AdaptiveLevin.tests.test_levin_roundoff_exit import (
    _call_signature,
    levin_quadrature_problems,
)

N = DEFAULT_LEVIN_CHEBSHEV_ORDER
N_FINE = 2 * N - 1


def _reference(expression):
    """expression() at 30 and at 40 digits: the 40-digit value, and their difference as its error."""
    with mpmath.workdps(30):
        value30 = expression()
    with mpmath.workdps(40):
        value40 = expression()
    with mpmath.workdps(40):
        ref_err = float(abs(value30 - value40))
    return float(value40), ref_err


def _parent_cc(
    x_span,
    f,
    BasisData,
    chebyshev_order,
    phase_span,
    theta_prime_Cheb,
    notify_label,
    id_label,
    vectorize_cache=None,
):
    """
    The parent's fallback solve: the working tree's, whose value and nested-pair estimate are the
    parent's bit for bit, with "phase_err" replaced by the parent's eq. (151) floor plus the
    declared-phase term, formed from the same samples and helpers as the parent formed them.
    """
    data = _adaptive_levin_subregion_cc(
        x_span,
        f,
        BasisData,
        chebyshev_order,
        phase_span,
        theta_prime_Cheb,
        notify_label,
        id_label,
        vectorize_cache=vectorize_cache,
    )
    if vectorize_cache is None:
        vectorize_cache = {}
    fine_grid, _ = chebyshev_matrices(x_span, 2 * chebyshev_order - 1)
    f_fine = np.vstack(
        [
            _sample_vectorized(func, fine_grid, vectorize_cache, ("f", id(func)))
            for func in f
        ]
    )
    width = np.fabs(x_span[1] - x_span[0])
    G0, G1 = _levin_G0_G1(theta_prime_Cheb, width)
    f_scale = float(np.max(np.sqrt(np.sum(f_fine * f_fine, axis=0))))
    round_off_err = _roundoff_floor(f_scale, width, G0, G1, chebyshev_order)
    endpoint_proxy = 0.5 * f_scale * width
    declared_err = _declared_endpoint_phase_err(
        BasisData, x_span[0], x_span[1], endpoint_proxy, endpoint_proxy
    )
    return dict(data, phase_err=round_off_err + declared_err)


@contextlib.contextmanager
def parent_floor():
    """Within the block, every fallback solve carries the parent's floor (_parent_cc())."""
    with mock.patch.object(lq, "_adaptive_levin_subregion_cc", _parent_cc):
        yield


def _cc_solve(span, f, theta):
    """
    One fallback solve on span, as _adaptive_levin_subregion_impl() makes it: theta' and the phase
    span from build_Levin_data() on the coarse grid, then _adaptive_levin_subregion_cc(). Returns
    (the solve's dict, the parent's floor for the same solve).
    """
    basis = _Basis_SinCos(theta)
    grid, Dmat = chebyshev_matrices(span, N)
    _, theta_prime, _, _, phase_span = basis.build_Levin_data(
        grid, Dmat, need_AmatT=False
    )
    data = _adaptive_levin_subregion_cc(
        span, f, basis, N, phase_span, theta_prime, None, None
    )
    parent = _parent_cc(span, f, basis, N, phase_span, theta_prime, None, None)
    return data, parent["phase_err"]


def _sums(span, f, theta, f_prime):
    """
    On the fine grid: S = sum_k |w_k| sum_i |f_ik b_ik|, F = sum_k |w_k| sum_i |f_ik| and
    F' = sum_k |w_k| sum_i |f'_ik|, with f' the exact derivatives supplied.
    """
    basis = _Basis_SinCos(theta)
    fine_grid, _ = chebyshev_matrices(span, N_FINE)
    w = np.fabs(_cc_weights(span, N_FINE))
    f_fine = np.vstack([[func(x) for x in fine_grid] for func in f])
    fp_fine = np.vstack([[func(x) for x in fine_grid] for func in f_prime])
    b_fine = np.array([basis.eval_basis(x) for x in fine_grid]).T
    S = float(np.dot(w, np.sum(np.fabs(f_fine * b_fine), axis=0)))
    F = float(np.dot(w, np.sum(np.fabs(f_fine), axis=0)))
    Fp = float(np.dot(w, np.sum(np.fabs(fp_fine), axis=0)))
    return S, F, Fp


class _HonestyMixin:

    def _assert_honest(self, value, declared, ref, ref_err, label):
        true_err = abs(value - ref)
        # the reference's own error is negligible against what is asserted on it
        self.assertLess(ref_err, 1e-6 * max(true_err, declared, 1e-300), label)
        # the value lies within its declared error of the reference ...
        self.assertLessEqual(
            abs(value - ref),
            declared,
            f"{label}: value {value!r} is not within {declared:.3e} of the reference {ref!r}",
        )
        # ... and the declared error is at least the true error
        self.assertGreaterEqual(declared, true_err, label)
        return true_err

    def _assert_solve_honest(self, data, ref, ref_err, label):
        """A single fallback solve's error as the driver takes it: max(pair, floor)."""
        declared = max(data["abserr_direct"], data["phase_err"])
        return self._assert_honest(data["value"], declared, ref, ref_err, label)


# ---------------------------------------------------------------------------------------------
# Tests 1 and 3: theta = -C (x - XS)^2, a stationary point at XS. theta' is supplied exactly, so
# the parent's G0 is theta' at the coarse node nearest XS, exactly. theta_mod_2pi is supplied.
# ---------------------------------------------------------------------------------------------

STATIONARY_C = 1.0e6


def _stationary_problem(xs):
    theta = {
        "theta": lambda x: -STATIONARY_C * (x - xs) ** 2,
        "theta_mod_2pi": lambda x: math.fmod(
            -STATIONARY_C * (x - xs) ** 2, 2.0 * math.pi
        ),
        "theta_deriv": lambda x: -2.0 * STATIONARY_C * (x - xs),
    }
    f = [lambda x: math.exp(-x), lambda x: 0.0]
    f_prime = [lambda x: -math.exp(-x), lambda x: 0.0]
    return f, theta, f_prime


def _stationary_reference(span, xs):
    """int_span e^{-x} sin(-C (x - xs)^2) dx, by mpmath quadrature split at xs."""
    a, b = span

    def expression():
        C = mpmath.mpf(STATIONARY_C)
        XS = mpmath.mpf(xs)
        integrand = lambda x: mpmath.exp(-x) * mpmath.sin(-C * (x - XS) ** 2)
        points = [mpmath.mpf(a)]
        if a < xs < b:
            points.append(XS)
        points.append(mpmath.mpf(b))
        return mpmath.quad(integrand, points, maxdegree=10)

    return _reference(expression)


class TestNoInverseThetaPrime(_HonestyMixin, unittest.TestCase):

    def test_floor_finite_at_a_stationary_node(self):
        """
        Test 1. theta = -1e6 (x - 1/2)^2 on [1/2, 1/2 + 2^-8], whose lower endpoint -- the
        stationary point -- is a node of every Chebyshev grid on it; theta' is supplied exactly, so
        the parent's G0 = |theta'(1/2)| is exactly 0 and its eq. (151) floor is +inf. The phase's
        total variation is 1e6 * 2^-16 = 15.3 rad < 6 pi, so the region takes the fallback.

        06d's floor is finite, at least (n + 4) u S, and at most the three terms with their
        arguments bounded from outside: |a| < 2 pi, |theta'| <= 2C h (its maximum on the region),
        f' exact (the floor's own f' is the Chebyshev derivative of the samples, which agrees with
        it to well inside the 1 % allowed). The node term dominates; the parent_demo output
        records the three terms (section 5 point 1 replaced "of order u S", which held before
        the node term was added).
        """
        xs = 0.5
        span = (0.5, 0.5 + 2.0**-8)
        f, theta, f_prime = _stationary_problem(xs)
        data, parent = _cc_solve(span, f, theta)
        self.assertTrue(data["is_direct"])

        self.assertEqual(parent, float("inf"))

        phi = data["phase_err"]
        S, F, Fp = _sums(span, f, theta, f_prime)
        h = span[1] - span[0]
        delta = _cc_node_error_bound(span)
        upper = (
            (N_FINE + 4) * _UNIT_ROUNDOFF * S
            + 2.0 * math.pi * _UNIT_ROUNDOFF * F
            + delta * (2.0 * STATIONARY_C * h * F + Fp)
        )
        self.assertTrue(math.isfinite(phi), f"06d floor {phi}")
        self.assertGreaterEqual(phi, (N_FINE + 4) * _UNIT_ROUNDOFF * S)
        self.assertLessEqual(phi, 1.01 * upper)

        ref, ref_err = _stationary_reference(span, xs)
        self._assert_solve_honest(data, ref, ref_err, "test 1")


class TestFloorDoesNotGrow(_HonestyMixin, unittest.TestCase):

    def test_nested_leaves_round_a_stationary_point(self):
        """
        Test 3. theta = -1e6 (x - xs)^2 with xs = 0.7236 (07's stationary point), on a fallback
        leaf of width 2^-8 holding xs, bisected three levels by hand: at each level the half
        holding xs. 06d's halves' floors sum to at most twice the leaf's, and every solve is honest.

        The parent's eq. (151) floors over the four levels were 1.98e-16, 9.04e-17, 6.63e-16 and
        5.31e-16 (parent_demo.txt): not monotone over these levels, so the assertion that it rises
        level on level was withdrawn (section 5 point 4); the growth section 0.1 cites is from depth
        2 to 8 on 07's window leaf.
        """
        xs = 0.7236
        a = 0.72265625  # 185 / 256
        span = (a, a + 2.0**-8)
        f, theta, _ = _stationary_problem(xs)

        for level in range(4):
            data, _ = _cc_solve(span, f, theta)
            self.assertTrue(data["is_direct"])

            c = 0.5 * (span[0] + span[1])
            left, _ = _cc_solve((span[0], c), f, theta)
            right, _ = _cc_solve((c, span[1]), f, theta)
            self.assertTrue(math.isfinite(data["phase_err"]))
            self.assertLessEqual(
                left["phase_err"] + right["phase_err"],
                2.0 * data["phase_err"],
                f"level {level}: {left['phase_err']} + {right['phase_err']} against "
                f"{data['phase_err']}",
            )

            ref, ref_err = _stationary_reference(span, xs)
            self._assert_solve_honest(data, ref, ref_err, f"test 3 level {level}")

            span = (span[0], c) if xs < c else (c, span[1])


# ---------------------------------------------------------------------------------------------
# Test 2: the floor bounds round-off, raw and reduced. Two families, f = (e^{-x}, 0):
#   (L) theta = c x on [1, 1 + 10/c]: |theta| ~ c and theta' x ~ c, for c = 1 ... 1e12;
#   (O) theta = Theta + 10 x on [0, 1]: |theta| ~ Theta, theta' x <= 10, for Theta = 1 ... 1e12.
# Each phase changes by 10 rad (< 6 pi). The reduced phase is computed exactly (mpmath), so it
# carries no error of its own. A request below what round-off allows: atol 1e-40, rtol 1e-17.
# ---------------------------------------------------------------------------------------------

TEST2_SCALES = (1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12)
TEST2_KWARGS = dict(atol=1e-40, rtol=1e-17)


def _test2_problem(family, scale, reduced):
    if family == "L":
        c, offset = scale, 0.0
        span = (1.0, 1.0 + 10.0 / scale)
    else:
        c, offset = 10.0, scale
        span = (0.0, 1.0)

    def theta_mod_2pi(x):
        with mpmath.workdps(40):
            return float(
                mpmath.fmod(
                    mpmath.mpf(offset) + mpmath.mpf(c) * mpmath.mpf(x), 2 * mpmath.pi
                )
            )

    theta = {"theta": lambda x: offset + c * x}
    if reduced:
        theta["theta_mod_2pi"] = theta_mod_2pi
    f = [lambda x: math.exp(-x), lambda x: 0.0]

    def expression():
        C, O = mpmath.mpf(c), mpmath.mpf(offset)
        F = (
            lambda x: -mpmath.exp(-x)
            * (mpmath.sin(O + C * x) + C * mpmath.cos(O + C * x))
            / (1 + C * C)
        )
        return F(mpmath.mpf(span[1])) - F(mpmath.mpf(span[0]))

    return span, f, theta, expression


def test2_cells():
    for family in ("L", "O"):
        for reduced in (False, True):
            for scale in TEST2_SCALES:
                yield family, reduced, scale


class TestFloorBoundsRoundoff(_HonestyMixin, unittest.TestCase):

    def test_floor_bounds_a_single_solve(self):
        """
        Test 2, one fallback solve per cell. The true error is at most the round-off floor alone
        in every cell: the floor, not the nested-pair estimate, is what has to carry it.

        At the parent (parent_demo.txt, section B) eq. (151) understated family (L) in every
        cell with c >= 1e3, raw and reduced, by 7.3 to 7.6e9, and family (O)'s raw cells from
        Theta = 1e6 on (1.8e3 to 5.3e8). 06d's first two terms alone understated family (L)'s
        reduced cells by the same order (06d section 5); with the node term every cell is bounded,
        true/floor at most 0.066.
        """
        for family, reduced, scale in test2_cells():
            with self.subTest(family=family, reduced=reduced, scale=scale):
                span, f, theta, expression = _test2_problem(family, scale, reduced)
                data, _ = _cc_solve(span, f, theta)
                self.assertTrue(data["is_direct"])
                ref, ref_err = _reference(expression)
                true_err = abs(data["value"] - ref)
                self.assertLess(ref_err, 1e-6 * max(true_err, 1e-300))
                self.assertLessEqual(
                    true_err,
                    data["phase_err"],
                    f"{family} reduced={reduced} scale={scale:.0e}: true {true_err:.3e}, "
                    f"floor {data['phase_err']:.3e}, pair {data['abserr_direct']:.3e}",
                )

    def test_calls_below_roundoff_are_honest(self):
        """
        Test 2, the call (atol 1e-40, rtol 1e-17): every region is a fallback region, and the
        result is honest in every cell.
        """
        for family, reduced, scale in test2_cells():
            with self.subTest(family=family, reduced=reduced, scale=scale):
                span, f, theta, expression = _test2_problem(family, scale, reduced)
                data = _adaptive_levin(span, f, _Basis_SinCos(theta), **TEST2_KWARGS)
                self.assertEqual(data["num_simple_regions"], data["num_regions"])
                ref, ref_err = _reference(expression)
                self._assert_honest(
                    data["value"],
                    data["abserr"],
                    ref,
                    ref_err,
                    f"test 2 call {family} reduced={reduced} scale={scale:.0e}",
                )


# ---------------------------------------------------------------------------------------------
# Test 4: the stationary-phase oracle.
# ---------------------------------------------------------------------------------------------


class TestStationaryPhaseOracle(_HonestyMixin, unittest.TestCase):

    def test_stationary_phase_oracle(self):
        """
        Test 4. test_stationary_phase_gate_uses_total_variation's problem: theta = 1e6 (x - x^2),
        f = (e^{-x}, 0) on [0, 1], atol = rtol = 1e-10, with the closed-form reference of
        test_levin_global_budget's 12e. At 06c (parent_demo.txt) it refines to 96,941 leaves, its
        leaves at x = 1/2 carrying an infinite floor; under 06d it converges in 76 leaves (12
        fallback), depth 14, 607 solves. Asserted: honest, abserr finite.
        """
        theta = lambda x: 1.0e6 * (x - x * x)
        f = [lambda x: math.exp(-x), lambda x: 0.0]
        data = adaptive_levin_sincos(
            (0.0, 1.0), f, theta={"theta": theta}, atol=1e-10, rtol=1e-10
        )
        ref, ref_err = _reference(levin_quadrature_references()["stationary_phase"])
        self.assertTrue(math.isfinite(data["abserr"]))
        self._assert_honest(data["value"], data["abserr"], ref, ref_err, "test 4")


# ---------------------------------------------------------------------------------------------
# Test 5: the guard in the depth-18 history log.
# ---------------------------------------------------------------------------------------------


class TestDepthHistoryGuard(unittest.TestCase):

    def test_zero_abserr_in_history(self):
        """
        Test 5. No natural problem was searched for: the depth-18 history log is factored into
        _log_depth_history(), and a history whose ancestor's abserr and relerr are exactly 0 (a
        Python float, as a fallback leaf's are) completes, with the improvement on a zero logged as
        not defined.
        """
        depth = 18
        abserr_history = {i: 2.0 ** (-i) for i in range(depth)}
        relerr_history = {i: 2.0 ** (-i - 1) for i in range(depth)}
        abserr_history[5] = 0.0
        relerr_history[5] = 0.0
        abserr_history[9] = float("inf")
        p_ratios_history = {i: [] for i in range(depth)}

        logger = logging.getLogger("AdaptiveLevin.levin_quadrature")
        with self.assertLogs(logger, level="INFO") as captured:
            _log_depth_history(depth, abserr_history, relerr_history, p_ratios_history)
        self.assertEqual(len(captured.records), depth)
        self.assertIn("improvement=not defined", captured.output[5])
        self.assertIn("improvement=not defined", captured.output[9])

        self.assertIsNone(_history_improvement(1.0, 0.0))
        self.assertIsNone(_history_improvement(1.0, float("inf")))
        self.assertIsNone(_history_improvement(1.0, float("nan")))
        self.assertEqual(_history_improvement(1.0, 0.5), 2.0)


# ---------------------------------------------------------------------------------------------
# Test 6: nothing else moves.
# ---------------------------------------------------------------------------------------------

# The driven problems whose call at the parent has a fallback leaf (parent_demo.txt, against the
# parent's own module). The rest are compared bit for bit against the in-process parent.
PARENT_HAS_FALLBACK_LEAF = frozenset(
    {
        "GRZIntegral(10)",
        "GRZIntegral(100)",
        "GRZIntegral(1000)",
        "stationary_phase",
        "mode_filter_tight",
    }
)


class TestNothingElseMoves(_HonestyMixin, unittest.TestCase):

    def test_nothing_else_moves(self):
        """
        Test 6. Every problem test_levin_quadrature.py drives. Where the parent's call has no
        fallback leaf, value, abserr and every region are bit-identical to the parent's (in-process,
        _parent_cc()); the rest are tabulated against the parent in parent_demo.txt, and asserted
        honest here, as every one is at the parent.
        """
        references = levin_quadrature_references()
        for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
            with self.subTest(problem=name):
                basis_object = _Basis_SinCos(theta) if basis is None else basis
                if raises is not None:
                    with self.assertRaises(raises):
                        _adaptive_levin(span, f, basis_object, **kwargs)
                    continue
                new = _adaptive_levin(span, f, basis_object, **kwargs)
                if name in PARENT_HAS_FALLBACK_LEAF:
                    self.assertGreater(new["num_simple_regions"], 0)
                    ref, ref_err = _reference(references[name])
                    self._assert_honest(
                        new["value"], new["abserr"], ref, ref_err, f"test 6 {name}"
                    )
                else:
                    with parent_floor():
                        basis_parent = _Basis_SinCos(theta) if basis is None else basis
                        parent = _adaptive_levin(span, f, basis_parent, **kwargs)
                    self.assertEqual(parent["num_simple_regions"], 0)
                    self.assertEqual(_call_signature(new), _call_signature(parent))


if __name__ == "__main__":
    unittest.main()
