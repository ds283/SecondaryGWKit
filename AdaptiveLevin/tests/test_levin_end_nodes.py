"""
Exact end nodes, and a node term that measures the displacement
(prompts/levin-phase-floor/06e-exact-end-nodes.md).

Two defects, one prompt:

(A) chebyshev_matrices() formed x = (a + b)/2 + (b - a)/2 x_base, whose end nodes miss b and a by an
    ulp in about a third of spans although x_base[0] = 1 and x_base[-1] = -1 exactly. A Levin leaf
    then integrated a shifted interval, a parent's ends and its halves' failed to tile, and the
    step-(4) residual saw the offset at every depth ([06c-chebyshev-end-nodes-miss-the-span]).
    06e sets x[0] := b and x[-1] := a after the map.
(B) 06d's node term in the Clenshaw-Curtis floor was a worst case, 5u (max(|a|, |b|) + h) max|theta'|
    at every node with absolute values throughout ([06d-cc-node-term-is-pessimistic]). 06e replaces
    it by phi_node = |E| + R_g + R_2 + R_c, E = sum_k w_k g'(x^_k) Delta_k, from the signed
    displacement Delta_k of every node and theta' at every node (_cc_node_term_parts()).

The parent is 06d's code, the module AdaptiveLevin/levin_quadrature.py at the dispatch SHA 50277f3,
read with `git show` and imported under another name (_parent()). The numbers quoted below are
from docs/levin-phase-floor/06e/calibration/out/parent_demo.txt, which runs the same problems.

References are closed forms or mpmath at 30 digits; each reference's own error is estimated as its
difference from the same expression at 40 digits, and asserted negligible. Every test on an integral
asserts honesty: the value within its declared error of the reference.

No Ray and no datastore.
"""

import math
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import mpmath
import numpy as np

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.tests.test_levin_global_budget import levin_quadrature_references
from AdaptiveLevin.tests.test_levin_roundoff_exit import levin_quadrature_problems

PARENT_SHA = "50277f3e019bb37df6d91a5c2340877755498508"
_PARENT_NAME = "levin_quadrature_06d_parent_50277f3"
_REPO = Path(__file__).resolve().parents[2]

N = lq.DEFAULT_LEVIN_CHEBSHEV_ORDER
N_FINE = 2 * N - 1
U = 2.0**-53


def _parent():
    """
    06d's module: AdaptiveLevin/levin_quadrature.py at PARENT_SHA, read with `git show` and
    executed as a module of another name, once per process.
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


def _reference(expression):
    """expression() at 30 and at 40 digits: the 40-digit value, and their difference as its error."""
    with mpmath.workdps(30):
        value30 = expression()
    with mpmath.workdps(40):
        value40 = expression()
    with mpmath.workdps(40):
        ref_err = float(abs(value30 - value40))
    return float(value40), ref_err


class _HonestyMixin:

    def _assert_honest(self, value, declared, ref, ref_err, label):
        true_err = abs(value - ref)
        # the reference's own error is negligible against what is asserted on it
        self.assertLess(ref_err, 1e-6 * max(true_err, declared, 1e-300), label)
        self.assertLessEqual(
            true_err,
            declared,
            f"{label}: value {value!r} is not within {declared:.3e} of the reference {ref!r}",
        )
        return true_err


# ---------------------------------------------------------------------------------------------
# Spans. Seeded, with |a| and |b| from 1e-3 to about 1e15 and relative widths from 1e-12 to 1, of
# either sign.
# ---------------------------------------------------------------------------------------------


def seeded_spans(count, seed):
    rng = np.random.default_rng(seed)
    spans = []
    while len(spans) < count:
        magnitude = 10.0 ** rng.uniform(-3.0, 15.0)
        lower = magnitude
        upper = magnitude * (1.0 + 10.0 ** rng.uniform(-12.0, 0.0))
        if upper == lower:
            continue
        if rng.uniform() < 0.5:
            lower, upper = -upper, -lower
        spans.append((float(lower), float(upper)))
    return spans


TEST1_ORDERS = (2, 16, 24, 31, 64)
TEST1_SEED = 20261030
TEST1_COUNT = 4000


class TestEndNodesExact(unittest.TestCase):

    def test_end_nodes_exact(self):
        """
        Test 1. 4,000 seeded spans at orders 2, 16, 24, 31 and 64: x[0] == b and x[-1] == a, and the
        interior nodes and D are bit-identical to the parent's. At the parent the end nodes miss on
        2,046 of the 4,000 spans, the same spans at every order, since x[0] and x[-1] are the
        midpoint plus and minus the half-width whatever the order (parent_demo.txt, section A).
        """
        parent = _parent()
        parent_misses = 0
        for a, b in seeded_spans(TEST1_COUNT, TEST1_SEED):
            missed = False
            for order in TEST1_ORDERS:
                x, D = lq.chebyshev_matrices((a, b), order)
                x_parent, D_parent = parent.chebyshev_matrices((a, b), order)
                self.assertEqual(x[0], b, f"[{a!r}, {b!r}] order {order}")
                self.assertEqual(x[-1], a, f"[{a!r}, {b!r}] order {order}")
                self.assertTrue(np.array_equal(x[1:-1], x_parent[1:-1]))
                self.assertTrue(np.array_equal(D, D_parent))
                missed = missed or x_parent[0] != b or x_parent[-1] != a
            parent_misses += missed
        # the defect is real at the parent on these spans
        self.assertGreater(parent_misses, TEST1_COUNT // 10)


TEST2_SEED = 20261031
TEST2_COUNT = 500
TEST2_LEVELS = 3


def _bisect(a, b):
    """
    The midpoint as _adaptive_levin() takes it: the leaf's break_point, at a bisection and for a
    fallback leaf's halves; (b + a) / 2.0 for a Levin leaf's step-(4) halves. Returns it, after
    checking the two agree.
    """
    leaf = lq._levin_leaf(
        start=a,
        end=b,
        depth=0,
        seq=0,
        value=0.0,
        region=None,
        is_direct=False,
        data=None,
        dataL=None,
        dataR=None,
        abserr_history={},
        relerr_history={},
        p_ratios_history={},
    )
    c = leaf.break_point
    assert c == (b + a) / 2.0
    return c


class TestLeavesTile(unittest.TestCase):

    def test_halves_tile(self):
        """
        Test 2. 500 seeded spans, each bisected three levels as _adaptive_levin() bisects (the trust
        check's grandchildren are its children's halves, so three levels cover them): at the
        Levin order and at the fallback's fine order, the parent's ends and its halves' outer ends
        are equal, and the halves share their inner end, bit for bit. At the parent this fails on
        1,841 of the 3,500 bisections (parent_demo.txt, section A).
        """
        for a, b in seeded_spans(TEST2_COUNT, TEST2_SEED):
            level = [(a, b)]
            for _ in range(TEST2_LEVELS):
                children = []
                for lo, hi in level:
                    c = _bisect(lo, hi)
                    for order in (N, N_FINE):
                        x_parent, _ = lq.chebyshev_matrices((lo, hi), order)
                        x_left, _ = lq.chebyshev_matrices((lo, c), order)
                        x_right, _ = lq.chebyshev_matrices((c, hi), order)
                        label = f"[{lo!r}, {hi!r}] at {c!r}, order {order}"
                        self.assertEqual(x_parent[0], x_right[0], label)
                        self.assertEqual(x_parent[-1], x_left[-1], label)
                        self.assertEqual(x_left[0], x_right[-1], label)
                    children += [(lo, c), (c, hi)]
                level = children


# ---------------------------------------------------------------------------------------------
# Test 3: theta = C (x - X0), C = 1e6, on [X0 + 0.1, X0 + 0.8] with X0 = 100, f = (e^-(x - X0), 0):
# no stationary point, theta' = 1e6, |x| ~ 100. theta_mod_2pi is computed exactly (mpmath) and
# theta' supplied, so the Levin values carry no phase rounding: what the parent chases is the
# end-node offset, about |f| ulp(100) = 1.4e-14 per leaf, against a request of 1.3e-15.
# ---------------------------------------------------------------------------------------------

TEST3_C = 1.0e6
TEST3_X0 = 100.0
TEST3_SPAN = (TEST3_X0 + 0.1, TEST3_X0 + 0.8)
TEST3_KWARGS = dict(atol=1e-30, rtol=1e-9, depth_max=20)


def _test3_problem():
    def theta_mod_2pi(x):
        with mpmath.workdps(40):
            return float(
                mpmath.fmod(
                    mpmath.mpf(TEST3_C) * (mpmath.mpf(x) - mpmath.mpf(TEST3_X0)),
                    2 * mpmath.pi,
                )
            )

    theta = {
        "theta": lambda x: TEST3_C * (x - TEST3_X0),
        "theta_mod_2pi": theta_mod_2pi,
        "theta_deriv": lambda x: TEST3_C,
    }
    f = [lambda x: math.exp(-(x - TEST3_X0)), lambda x: 0.0]

    def expression():
        C, X0 = mpmath.mpf(TEST3_C), mpmath.mpf(TEST3_X0)

        def F(x):
            t = mpmath.mpf(x) - X0
            return (
                -mpmath.exp(-t)
                * (mpmath.sin(C * t) + C * mpmath.cos(C * t))
                / (1 + C * C)
            )

        return F(TEST3_SPAN[1]) - F(TEST3_SPAN[0])

    return f, theta, expression


class TestOffsetChaseGone(_HonestyMixin, unittest.TestCase):

    def test_offset_chase_gone(self):
        """
        Test 3. At the parent the call reaches depth 16 (639 leaves, 512 of them fallback leaves
        below the deepest Levin level, 13, where the phase span falls under 6 pi), unconverged,
        abserr 2.3e-10 against a true error of 1.8e-13 (2,555 solves). Along its deepest line the
        Levin residual stays at 3.6e-15 - 2.5e-14 from depth 0 to 15 while the eq. (151) floor falls
        from 9.5e-17 to 2.5e-20, residual over floor 38 rising to 9.2e5, and no level's halves tile
        (parent_demo.txt, section B). Under 06e the root is floor-limited and the call converges at
        depth 0 (7 solves), abserr 9.5e-17. Both honest.
        """
        f, theta, expression = _test3_problem()
        ref, ref_err = _reference(expression)

        parent = _parent()
        old = parent._adaptive_levin(
            TEST3_SPAN, f, parent._Basis_SinCos(theta), **TEST3_KWARGS
        )
        self.assertGreaterEqual(old["max_depth"], 12)
        self._assert_honest(old["value"], old["abserr"], ref, ref_err, "test 3 parent")

        new = lq._adaptive_levin(TEST3_SPAN, f, lq._Basis_SinCos(theta), **TEST3_KWARGS)
        self.assertTrue(new["converged"])
        self.assertLess(new["max_depth"], 8)
        self._assert_honest(new["value"], new["abserr"], ref, ref_err, "test 3 06e")


# ---------------------------------------------------------------------------------------------
# Test 4: the node term bounds the node error, and is tighter than 06d's. The cells are single
# fallback solves:
#   T2: 06d's test 2, theta = c x on [1, 1 + 10/c] (L) and theta = Theta + 10 x on [0, 1] (O),
#       f = (e^-x, 0), raw and reduced, no theta' supplied (the spectral path, eta_theta = 1);
#   T3: 06d's test 3, the nested leaves round theta = -1e6 (x - 0.7236)^2, f = (e^-x, 0), theta'
#       supplied and theta_mod_2pi (eta_theta = 2^-10);
#   W:  window-like cells, 07's problem theta = omega x + (c/2)(x - x_s)^2, theta' supplied,
#       f = (e^-x, cos 3x), x_s = 0.7236..., at c = 1e6 (omega = 0) and 1e8 (omega = 0, 1): the
#       dyadic leaves of [0, 1] at the leaf study's depths (8-12 at 1e6, 11-14 at 1e8) nearest the
#       root of theta', up to four each side, that take the fallback.
# The true node error is the rule evaluated in mpmath at the float nodes against the exact nodes,
# with the rule's own float weights, as 06d section 5 measured it.
# ---------------------------------------------------------------------------------------------

T2_SCALES = (1.0, 1.0e3, 1.0e6, 1.0e9, 1.0e12)
STATIONARY_C = 1.0e6
STATIONARY_XS = 0.7236
WINDOW_XS = 0.7236067977499790
WINDOW_CASES = (
    (1.0e6, 0.0, (8, 9, 10, 11, 12)),
    (1.0e8, 0.0, (11, 12, 13, 14)),
    (1.0e8, 1.0, (11, 12, 13, 14)),
)
WINDOW_PER_SIDE = 4


class _Cell:
    """
    One fallback solve: span, the amplitudes f and the phase dict theta for the module, and
    their mpmath forms g(x) (the integrand) and, where theta' is supplied, theta'(x).
    """

    def __init__(self, label, span, f, theta, g_mp, theta_prime_mp=None):
        self.label = label
        self.span = span
        self.f = f
        self.theta = theta
        self.g_mp = g_mp
        self.theta_prime_mp = theta_prime_mp

    def reference(self):
        a, b = self.span
        return _reference(
            lambda: mpmath.quad(self.g_mp, [mpmath.mpf(a), mpmath.mpf(b)], maxdegree=10)
        )


def _t2_cells():
    for family in ("L", "O"):
        for reduced in (False, True):
            for scale in T2_SCALES:
                if family == "L":
                    c, offset, span = scale, 0.0, (1.0, 1.0 + 10.0 / scale)
                else:
                    c, offset, span = 10.0, scale, (0.0, 1.0)

                def theta_mod_2pi(x, c=c, offset=offset):
                    with mpmath.workdps(40):
                        return float(
                            mpmath.fmod(
                                mpmath.mpf(offset) + mpmath.mpf(c) * mpmath.mpf(x),
                                2 * mpmath.pi,
                            )
                        )

                theta = {"theta": lambda x, c=c, offset=offset: offset + c * x}
                if reduced:
                    theta["theta_mod_2pi"] = theta_mod_2pi
                f = [lambda x: math.exp(-x), lambda x: 0.0]
                g_mp = lambda x, c=c, offset=offset: mpmath.exp(-x) * mpmath.sin(
                    mpmath.mpf(offset) + mpmath.mpf(c) * x
                )
                yield _Cell(
                    f"T2 {family} {'reduced' if reduced else 'raw'} {scale:.0e}",
                    span,
                    f,
                    theta,
                    g_mp,
                )


def _t3_cells():
    xs = STATIONARY_XS
    theta = {
        "theta": lambda x: -STATIONARY_C * (x - xs) ** 2,
        "theta_mod_2pi": lambda x: math.fmod(
            -STATIONARY_C * (x - xs) ** 2, 2.0 * math.pi
        ),
        "theta_deriv": lambda x: -2.0 * STATIONARY_C * (x - xs),
    }
    f = [lambda x: math.exp(-x), lambda x: 0.0]
    g_mp = lambda x: mpmath.exp(-x) * mpmath.sin(
        -mpmath.mpf(STATIONARY_C) * (x - mpmath.mpf(xs)) ** 2
    )
    tp_mp = lambda x: -2 * mpmath.mpf(STATIONARY_C) * (x - mpmath.mpf(xs))
    a = 0.72265625
    span = (a, a + 2.0**-8)
    for level in range(4):
        c = 0.5 * (span[0] + span[1])
        for part, s in (
            ("leaf", span),
            ("left", (span[0], c)),
            ("right", (c, span[1])),
        ):
            yield _Cell(f"T3 level {level} {part}", s, f, theta, g_mp, tp_mp)
        span = (span[0], c) if xs < c else (c, span[1])


def _window_total_variation(c, xr, lo, hi):
    """int_lo^hi |theta'| for theta' = c (x - xr), c > 0."""
    if lo < xr < hi:
        return 0.5 * c * ((hi - xr) ** 2 + (xr - lo) ** 2)
    return 0.5 * c * abs((hi - xr) ** 2 - (lo - xr) ** 2)


def _window_cells():
    for c, omega, depths in WINDOW_CASES:
        xr = WINDOW_XS - omega / c
        theta = {
            "theta": lambda x, c=c, omega=omega: omega * x
            + 0.5 * c * (x - WINDOW_XS) ** 2,
            "theta_deriv": lambda x, c=c, omega=omega: omega + c * (x - WINDOW_XS),
        }
        f = [lambda x: np.exp(-x), lambda x: np.cos(3.0 * x)]

        def g_mp(x, c=c, omega=omega):
            t = (
                mpmath.mpf(omega) * x
                + mpmath.mpf(c) / 2 * (x - mpmath.mpf(WINDOW_XS)) ** 2
            )
            return mpmath.exp(-x) * mpmath.sin(t) + mpmath.cos(3 * x) * mpmath.cos(t)

        tp_mp = lambda x, c=c, omega=omega: mpmath.mpf(omega) + mpmath.mpf(c) * (
            x - mpmath.mpf(WINDOW_XS)
        )
        for depth in depths:
            h = 2.0**-depth
            k0 = int(math.floor(xr / h))
            nearest = []
            for k in range(k0 - 4 * WINDOW_PER_SIDE, k0 + 4 * WINDOW_PER_SIDE + 1):
                lo, hi = k * h, (k + 1) * h
                # well inside the 6 pi gate, so the cell is a fallback solve on either code
                if _window_total_variation(c, xr, lo, hi) < 5.5 * math.pi:
                    nearest.append((abs(k - k0), k - k0, (lo, hi)))
            nearest.sort()
            for _, offset, span in nearest[: 2 * WINDOW_PER_SIDE + 1]:
                yield _Cell(
                    f"W c={c:.0e} omega={omega:g} depth {depth} leaf {offset:+d}",
                    span,
                    f,
                    theta,
                    g_mp,
                    tp_mp,
                )


def test4_cells():
    yield from _t2_cells()
    yield from _t3_cells()
    yield from _window_cells()


def node_term_solve(cell):
    """
    One fallback solve under 06e, as _adaptive_levin_subregion_impl() makes it (theta' and the
    phase span from build_Levin_data() on the coarse grid, then _adaptive_levin_subregion_cc()),
    with the node term's parts and inputs captured. Returns (solve dict, (E, R_g, R_2, R_c),
    captured positional arguments of _cc_node_term_parts()).
    """
    captured = {}
    original = lq._cc_node_term_parts

    def spy(*args):
        parts = original(*args)
        captured["args"] = args
        captured["parts"] = parts
        return parts

    basis = lq._Basis_SinCos(cell.theta)
    grid, Dmat = lq.chebyshev_matrices(cell.span, N)
    _, theta_prime, _, _, phase_span = basis.build_Levin_data(
        grid, Dmat, need_AmatT=False
    )
    with mock.patch.object(lq, "_cc_node_term_parts", spy):
        data = lq._adaptive_levin_subregion_cc(
            cell.span, cell.f, basis, N, phase_span, theta_prime, None, None
        )
    return data, captured["parts"], captured["args"]


def node_term_06d(cell):
    """
    06d's third term on the same cell, as 06d's own fallback solve forms it: node_error times
    sum_k |w_k| sum_i (theta_prime_bound |f_ik| + |f'_ik|), from the arguments its
    _cc_roundoff_floor() is handed.
    """
    parent = _parent()
    captured = {}
    original = parent._cc_roundoff_floor

    def spy(*args):
        captured["args"] = args
        return original(*args)

    basis = parent._Basis_SinCos(cell.theta)
    grid, Dmat = parent.chebyshev_matrices(cell.span, N)
    _, theta_prime, _, _, phase_span = basis.build_Levin_data(
        grid, Dmat, need_AmatT=False
    )
    with mock.patch.object(parent, "_cc_roundoff_floor", spy):
        parent._adaptive_levin_subregion_cc(
            cell.span, cell.f, basis, N, phase_span, theta_prime, None, None
        )
    weights, f, _, _, node_error, theta_prime_bound, f_prime = captured["args"]
    return node_error * float(
        np.dot(
            np.fabs(weights),
            np.sum(theta_prime_bound * np.fabs(f) + np.fabs(f_prime), axis=0),
        )
    )


def true_node_error(cell):
    """
    sum_k w_k (g(x^_k) - g(x_k)) in mpmath at 40 digits: the fine rule, with its own float
    weights, at 06e's float nodes against the exact nodes.
    """
    a, b = cell.span
    nodes, _ = lq.chebyshev_matrices(cell.span, N_FINE)
    weights = lq._cc_weights(cell.span, N_FINE)
    with mpmath.workdps(40):
        A, B = mpmath.mpf(a), mpmath.mpf(b)
        total = mpmath.mpf(0)
        for k in range(N_FINE):
            exact = (A + B) / 2 + (B - A) / 2 * mpmath.cos(mpmath.pi * k / (N_FINE - 1))
            total += mpmath.mpf(weights[k]) * (
                cell.g_mp(mpmath.mpf(nodes[k])) - cell.g_mp(exact)
            )
        return float(total)


class TestNodeTerm(_HonestyMixin, unittest.TestCase):

    def test_node_term_bounds_node_error(self):
        """
        Test 4. In every cell the solve is honest (its value within max(nested pair, floor) of the
        reference), phi_node is at least the true node error, and phi_node is below 06d's third
        term. Over the 122 cells, true/phi_node is at most 0.997; 06d's term over phi_node is
        4.3 - 58 on T2's 20 cells (eta_theta = 1 there), 174 - 2,974 on T3's 12 and 76 - 20,660 on
        the 90 window cells, median 848 (parent_demo.txt, section C, which tables every cell).
        """
        for cell in test4_cells():
            with self.subTest(cell=cell.label):
                data, (E, R_g, R_2, R_c), _ = node_term_solve(cell)
                self.assertTrue(data["is_direct"])
                phi_node = abs(E) + R_g + R_2 + R_c
                node_error = true_node_error(cell)
                self.assertLessEqual(
                    abs(node_error),
                    phi_node,
                    f"{cell.label}: true node error {node_error:.3e}, phi_node {phi_node:.3e} "
                    f"(|E| {abs(E):.3e}, R_g {R_g:.3e}, R_2 {R_2:.3e}, R_c {R_c:.3e})",
                )
                self.assertLess(phi_node, node_term_06d(cell), cell.label)
                ref, ref_err = cell.reference()
                self._assert_honest(
                    data["value"],
                    max(data["abserr_direct"], data["phase_err"]),
                    ref,
                    ref_err,
                    cell.label,
                )

    def test_supplied_theta_prime_within_eta(self):
        """
        The measurement behind _CC_THETA_PRIME_REL_ERROR_SUPPLIED: on every cell of test 4 whose
        theta' is supplied (T3 and W), the theta' the node term is handed at each fine node is
        within a few u of theta' evaluated in mpmath at the same float node (at most 1.0 u;
        parent_demo.txt, section C), far inside eta_theta = 2^-10.
        """
        worst = 0.0
        for cell in test4_cells():
            if cell.theta_prime_mp is None:
                continue
            _, _, args = node_term_solve(cell)
            theta_prime = args[9]
            self.assertEqual(args[11], lq._CC_THETA_PRIME_REL_ERROR_SUPPLIED)
            nodes, _ = lq.chebyshev_matrices(cell.span, N_FINE)
            with mpmath.workdps(40):
                for value, x in zip(theta_prime, nodes):
                    if value != 0.0:
                        exact = cell.theta_prime_mp(mpmath.mpf(x))
                        worst = max(
                            worst, float(abs(mpmath.mpf(value) - exact) / abs(value))
                        )
        self.assertLess(worst, 4.0 * U)

    def test_displacements_exact(self):
        """
        _chebyshev_node_displacements() against mpmath at 80 digits on 300 seeded spans at the
        five orders of test 1: every Delta_k within its stated bound, 2u |Delta_k| + err_abs, and
        the interior nodes it reconstructs are chebyshev_matrices()'s.
        """
        rng = np.random.default_rng(20261101)
        spans = seeded_spans(300, 20261101)
        for a, b in spans:
            order = int(rng.choice(TEST1_ORDERS))
            nodes, _ = lq.chebyshev_matrices((a, b), order)
            delta, err_abs = lq._chebyshev_node_displacements((a, b), order)
            self.assertEqual(delta[0], 0.0)
            self.assertEqual(delta[-1], 0.0)
            with mpmath.workdps(80):
                A, B = mpmath.mpf(a), mpmath.mpf(b)
                for k in range(order):
                    exact = (A + B) / 2 + (B - A) / 2 * mpmath.cos(
                        mpmath.pi * k / (order - 1)
                    )
                    true_delta = mpmath.mpf(nodes[k]) - exact
                    self.assertLessEqual(
                        float(abs(mpmath.mpf(delta[k]) - true_delta)),
                        2.0 * U * abs(delta[k]) + err_abs,
                        f"[{a!r}, {b!r}] order {order} node {k}",
                    )


# ---------------------------------------------------------------------------------------------
# Test 5: the stationary-phase oracle (06d's test 4).
# ---------------------------------------------------------------------------------------------


class TestStationaryPhaseOracle(_HonestyMixin, unittest.TestCase):

    def test_stationary_phase_oracle(self):
        """
        Test 5. test_stationary_phase_gate_uses_total_variation's problem: theta = 1e6 (x - x^2),
        f = (e^-x, 0) on [0, 1], atol = rtol = 1e-10. Under 06d and under 06e alike: 76 leaves (12
        fallback), depth 14, 607 solves, converged, abserr 2.52e-11 against a true error of 1.9e-14
        (parent_demo.txt, section D). Asserted: honest and converged.
        """
        f = [lambda x: math.exp(-x), lambda x: 0.0]
        data = lq.adaptive_levin_sincos(
            (0.0, 1.0),
            f,
            theta={"theta": lambda x: 1.0e6 * (x - x * x)},
            atol=1e-10,
            rtol=1e-10,
        )
        ref, ref_err = _reference(levin_quadrature_references()["stationary_phase"])
        self.assertTrue(data["converged"])
        self._assert_honest(data["value"], data["abserr"], ref, ref_err, "test 5")


# ---------------------------------------------------------------------------------------------
# Test 6: every problem test_levin_quadrature.py drives.
# ---------------------------------------------------------------------------------------------


class TestDrivenProblems(_HonestyMixin, unittest.TestCase):

    def test_driven_problems(self):
        """
        Test 6. Every problem test_levin_quadrature.py drives _adaptive_levin() with: honest under
        06d and under 06e, and |v_06e - v_06d| <= abserr_06d + abserr_06e. The two that raise
        raise under both. parent_demo.txt, section E, tables each.
        """
        parent = _parent()
        references = levin_quadrature_references()
        for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
            with self.subTest(problem=name):
                old_basis = parent._Basis_SinCos(theta) if basis is None else basis
                new_basis = lq._Basis_SinCos(theta) if basis is None else basis
                if raises is not None:
                    with self.assertRaises(raises):
                        parent._adaptive_levin(span, f, old_basis, **kwargs)
                    with self.assertRaises(raises):
                        lq._adaptive_levin(span, f, new_basis, **kwargs)
                    continue
                old = parent._adaptive_levin(span, f, old_basis, **kwargs)
                new = lq._adaptive_levin(span, f, new_basis, **kwargs)
                ref, ref_err = _reference(references[name])
                self._assert_honest(
                    old["value"], old["abserr"], ref, ref_err, f"{name} 06d"
                )
                self._assert_honest(
                    new["value"], new["abserr"], ref, ref_err, f"{name} 06e"
                )
                self.assertLessEqual(
                    abs(new["value"] - old["value"]),
                    old["abserr"] + new["abserr"],
                    name,
                )


if __name__ == "__main__":
    unittest.main()
