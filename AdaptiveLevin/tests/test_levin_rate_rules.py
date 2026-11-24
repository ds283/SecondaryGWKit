"""
The rate rules on a trusted Levin leaf (prompts/levin-phase-floor/07b-levin-leaf-beside-a-window.md, section 1 and
section 3 as amended by section 9; the fix is Part A's variant (h)).

The trust check one level down sees only what a leaf, its halves and its quarters disagree on. 07b's Part A measured
trusted Levin leaves carrying an error all three share: beside a stationary point's window, where the leaf's first
grid spacing is many times the local scale l = |theta' / theta''|, and away from windows on leaves whose |theta'|
ranges widely over their nodes. After the tree's trust test passes a Levin leaf, _leaf_rate_rule() runs: the edge
rule on a leaf with an end at a window's edge (|theta'| at the edge node and at its grid neighbour within
_LEVIN_EDGE_RATE_RATIO_MAX = 2), the range rule on any other (max|theta'| / min|theta'| over its nodes within
_LEVIN_LEAF_RATE_RANGE_MAX = 10). A refused leaf is untrusted exactly as one the tree's test refuses. The result
dict counts the refusals ("num_edge_rule_refusals", "num_range_rule_refusals").

  (a) the rule directly, on _leaf_rate_rule() with constructed rates and spans;
  (b) the edge mechanism: a call converged and dishonest at the parent, honest at the commit, the edge rule acting;
  (c) the range mechanism: a call with no window, the same;
  (d) a leaf the rule refuses at a small depth_max: the call unconverged, the leaf declaring as the tree declares an
      untrusted leaf there;
  (e) calls the rule must not touch: bit-identical to the parent, both counts 0.

The parent is the dispatch commit c768c4ad's module, 07a's code. Its behaviour quoted here was measured by running
that module on the same problems, from a detached worktree at the dispatch SHA (the log,
prompts/levin-phase-floor/logs/07b-levin-leaf-beside-a-window.md, records the runs); its value and abserr are held
here as float hex. This module does not run the parent. Where a test needs the parent's call beside the commit's,
it runs the commit with _leaf_rate_rule patched to refuse nothing, which is the parent's code path, and asserts
that this reproduces the parent's value and abserr bit for bit.

References are closed forms (the error function), by docs/levin-phase-floor/07b/code/test_references_07b.py at 40
and 60 digits (out/test_references_07b.json: agreeing to 8.9e-39 and 3.8e-37 relative). Every honesty assertion
checks both that the value is within its abserr of the reference and that abserr is at least the true error. No Ray
and no datastore.
"""

import contextlib
import math
import unittest
from unittest import mock

import numpy as np

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.levin_quadrature import (
    _leaf_rate_rule,
    adaptive_levin_sincos,
    chebyshev_matrices,
)
from AdaptiveLevin.tests.test_levin_cc_estimate import cc_rule_07c
from AdaptiveLevin.tests.test_levin_stationary import F_23, _Honest

EDGE_MAX = lq._LEVIN_EDGE_RATE_RATIO_MAX
RANGE_MAX = lq._LEVIN_LEAF_RATE_RANGE_MAX

# The rule refusing nothing: the parent's code path.
_REFUSE_NOTHING = (False, None, 1.0)

# --- (a): constructed rates on a dyadic span --------------------------------------------------------------------
A_LO, A_HI, A_ORDER = 0.25, 0.75, 16


def _grid(lo=A_LO, hi=A_HI, order=A_ORDER):
    grid, _ = chebyshev_matrices((lo, hi), max(order, lq._LEVIN_MINIMUM_ALLOWED_ORDER))
    return grid


def _rate(values, default=1.0):
    """A rate taking the given values at the given float nodes, and default elsewhere."""
    return lambda x: values.get(float(x), default)


# --- (b) and (c): theta = (c/2)(x - x_s)^2, f = (e^{g x}, cos(h x)) -----------------------------------------------
# A root of theta' at x_s, near the right end of [0, 1]. The amplitude's cos(h x) makes the halves and quarters of the
# Levin leaf beside the window disagree by more than its unresolved edge does, so the tree's test trusts it (r < 1/4).
Q_C, Q_XS, Q_G, Q_H = 1.19e9, 0.8186383592305085, 0.744, 109.28
Q_ORDER = 24
Q_F = [lambda x: np.exp(Q_G * x), lambda x: np.cos(Q_H * x)]
Q_THETA = {
    "theta": lambda x: 0.5 * Q_C * (x - Q_XS) ** 2,
    "theta_deriv": lambda x: Q_C * (x - Q_XS),
}

# (b): the call on [0, 1], atol 1e-7, rtol 1e-7.
B_ATOL, B_RTOL = 1e-7, 1e-7
B_REF = float.fromhex(
    "0x1.9c4e2dd17f34bp-14"
)  # 9.83012644368400574577854202488928940679011427e-5
# The parent (c768c4ad): 10 regions, max depth 5, converged, 7.62 times its abserr from B_REF.
B_PARENT_VALUE = float.fromhex("0x1.9d27c7fc9702bp-14")
B_PARENT_ABSERR = float.fromhex("0x1.c8e0696db4c5fp-26")
# The parent's carrying leaf: the Levin leaf at depth 1 from the middle of [0, 1] to the window's lower edge, trusted at
# r = 0.125 with e' = 1.97e-9; its true error is 2.03e-7, 103 e' (out/test_references_07b.json, "edge_leaves"[1]).
B_CARRIER = (
    float.fromhex("0x1.a317190ba9ba3p-2"),
    float.fromhex("0x1.a317190ba9ba3p-1"),
)

# (c): the same integrand on the carrier's span alone, atol 1e-8, rtol 1e-7. theta' has no zero on it, so nothing is
# located; |theta'| ranges by 3,984 over its nodes.
C_SPAN = B_CARRIER
C_ATOL, C_RTOL = 1e-8, 1e-7
C_REF = float.fromhex(
    "0x1.f1edc864828bbp-17"
)  # 1.48394357859692572515522832264301511862023987e-5
# The parent: 1 region (the span, trusted at depth 0), converged, 102.8 times its abserr from C_REF.
C_PARENT_VALUE = float.fromhex("0x1.f8bb5b7d9f97ep-17")
C_PARENT_ABSERR = float.fromhex("0x1.0f1fa0a7f80bcp-29")

# --- (d): theta = 2 e^{4x} on [0, 1.5], f = (e^{-x}, cos x), depth_max 1 -------------------------------------------
D_SPAN, D_ATOL, D_RTOL, D_DEPTH_MAX = (0.0, 1.5), 1e-14, 1e-7, 1
D_F = [lambda x: np.exp(-x), lambda x: np.cos(x)]
D_THETA = {
    "theta": lambda x: 2.0 * np.exp(4.0 * x),
    "theta_deriv": lambda x: 8.0 * np.exp(4.0 * x),
}
# The parent: two leaves at depth 1, both trusted, converged.
D_PARENT_VALUE = float.fromhex("-0x1.b5db4757865afp-4")
D_PARENT_ABSERR = float.fromhex("0x1.8075aba2113acp-33")

# --- (e) --------------------------------------------------------------------------------------------------------
# (e1) constant |theta'|: theta = 1e3 x on [0, 1], f = (e^{-x}, cos 30x): four Levin leaves at depth 2, all trusted.
E1_F = [lambda x: np.exp(-x), lambda x: np.cos(30.0 * x)]
E1_THETA = {"theta": lambda x: 1.0e3 * x, "theta_deriv": lambda x: 1.0e3 + 0.0 * x}
E1_PARENT_VALUE = float.fromhex("0x1.eb56afc607e00p-11")
E1_PARENT_ABSERR = float.fromhex("0x1.72de5e19b6d87p-39")
# (e2) theta = 0.04 e^x on [0, 6], f = F_23: every leaf Clenshaw-Curtis (the phase varies by 16.1 rad < 6 pi), the
# leaf [0, 3] trusted with |theta'| ranging by e^3 = 20.1 over its nodes.
E2_THETA = {
    "theta": lambda x: 0.04 * np.exp(x),
    "theta_deriv": lambda x: 0.04 * np.exp(x),
}
E2_PARENT_VALUE = float.fromhex("-0x1.24a89a8b8f0cap-1")
E2_PARENT_ABSERR = float.fromhex("0x1.44e3e19d1ad33p-34")
E_ATOL, E_RTOL = 1e-14, 1e-7


def _node_range(rate, lo, hi, order):
    values = np.abs(np.array([float(rate(x)) for x in _grid(lo, hi, order)]))
    return float(np.max(values) / np.min(values))


class TestTheRuleDirectly(unittest.TestCase):
    """(a) _leaf_rate_rule() with constructed rates and spans."""

    def test_a1_edge_threshold(self):
        """An edge q of exactly 2 is not refused, the next double above is; at either end of the leaf."""
        grid = _grid()
        just_above = math.nextafter(EDGE_MAX, math.inf)
        for edge, neighbour in ((A_LO, grid[-2]), (A_HI, grid[1])):
            with self.subTest(edge=edge):
                at = _leaf_rate_rule(
                    (A_LO, A_HI), A_ORDER, [_rate({edge: 1.0, neighbour: 2.0})], (edge,)
                )
                self.assertEqual(at, (False, "edge", 2.0))
                above = _leaf_rate_rule(
                    (A_LO, A_HI),
                    A_ORDER,
                    [_rate({edge: 1.0, neighbour: just_above})],
                    (edge,),
                )
                self.assertEqual(above, (True, "edge", just_above))
                # the ratio is symmetric: the edge's rate the larger
                self.assertEqual(
                    _leaf_rate_rule(
                        (A_LO, A_HI),
                        A_ORDER,
                        [_rate({edge: -2.0, neighbour: -1.0})],
                        (edge,),
                    ),
                    (False, "edge", 2.0),
                )

    def test_a2_range_threshold(self):
        """A range q of exactly 10 over the nodes is not refused, the next double above is."""
        grid = _grid()
        just_above = math.nextafter(RANGE_MAX, math.inf)
        interior = float(grid[5])
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, [_rate({interior: 10.0})], ()),
            (False, "range", 10.0),
        )
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, [_rate({interior: just_above})], ()),
            (True, "range", just_above),
        )

    def test_a3_zero_rate(self):
        """A rate exactly 0 at a node tested gives q = inf and a refusal, under either rule."""
        grid = _grid()
        inf = float("inf")
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, [_rate({A_LO: 0.0})], (A_LO,)),
            (True, "edge", inf),
        )
        self.assertEqual(
            _leaf_rate_rule(
                (A_LO, A_HI), A_ORDER, [_rate({float(grid[-2]): 0.0})], (A_LO,)
            ),
            (True, "edge", inf),
        )
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, [_rate({float(grid[7]): 0.0})], ()),
            (True, "range", inf),
        )

    def test_a4_only_window_edge_ends(self):
        """The edge rule tests only the ends that are window edges: a jump of 100 at the upper end's neighbour is not
        seen when only the lower end is an edge, and is when the upper end is."""
        grid = _grid()
        rate = [_rate({A_HI: 1.0, float(grid[1]): 100.0})]
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, rate, (A_LO,)),
            (False, "edge", 1.0),
        )
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, rate, (A_HI,)),
            (True, "edge", 100.0),
        )
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, rate, (A_LO, A_HI)),
            (True, "edge", 100.0),
        )

    def test_a5_the_rules_do_not_cross(self):
        """A touching leaf whose rate ranges by 1000 over its nodes, with edge q at most 2, is not refused: the range
        rule does not apply to it. A leaf touching none with an edge-like jump of 5 at an end is not refused: its
        range is 5, below 10, and the edge rule does not apply to it."""
        grid = _grid()
        steep = {float(x): 1000.0 for x in grid}
        steep.update({A_LO: 1.0, float(grid[-2]): 1.5})
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, [_rate(steep)], (A_LO,)),
            (False, "edge", 1.5),
        )
        jump = [_rate({A_HI: 1.0, float(grid[1]): 5.0})]
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, jump, ()),
            (False, "range", 5.0),
        )

    def test_a6_the_largest_q_decides(self):
        """With several phases the largest q decides, under either rule, in either order of the phases."""
        grid = _grid()
        mild = _rate({A_LO: 1.0, float(grid[-2]): 1.5})
        sharp = _rate({A_LO: 1.0, float(grid[-2]): 3.0})
        for rates in ([mild, sharp], [sharp, mild]):
            self.assertEqual(
                _leaf_rate_rule((A_LO, A_HI), A_ORDER, rates, (A_LO,)),
                (True, "edge", 3.0),
            )
        four = _rate({float(grid[3]): 4.0})
        twelve = _rate({float(grid[9]): 12.0})
        nine = _rate({float(grid[9]): 9.0})
        for rates in ([four, twelve], [twelve, four]):
            self.assertEqual(
                _leaf_rate_rule((A_LO, A_HI), A_ORDER, rates, ()),
                (True, "range", 12.0),
            )
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, [four, nine], ()),
            (False, "range", 9.0),
        )

    def test_a7_no_rates(self):
        """With no rates nothing is tested: not refused, no rule, q = 1."""
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), A_ORDER, None, ()), (False, None, 1.0)
        )

    def test_a8_order_clamped(self):
        """The nodes are the solve's: chebyshev_order is clamped up to _LEVIN_MINIMUM_ALLOWED_ORDER, as the solve's."""
        grid = _grid(order=lq._LEVIN_MINIMUM_ALLOWED_ORDER)
        rate = [_rate({A_LO: 1.0, float(grid[-2]): 3.0})]
        self.assertEqual(
            _leaf_rate_rule((A_LO, A_HI), 2, rate, (A_LO,)), (True, "edge", 3.0)
        )


class TestEdgeMechanism(unittest.TestCase, _Honest):
    def test_b_leaf_beside_a_window(self):
        """
        (b) theta = (c/2)(x - x_s)^2 on [0, 1], c = 1.19e9, x_s = 0.8186383592305085, f = (e^{0.744 x},
        cos(109.28 x)), order 24, atol 1e-7, rtol 1e-7. One root at x_s, one window of half-phase 2 pi round it. The
        parent returns 10 regions, converged, 7.62 times its abserr from the reference. The excess is carried by one
        Levin leaf, from 0.409 to the window's lower edge, at depth 1: trusted at r = 0.125, e' = 1.97e-9, true error
        2.03e-7 = 103 e' (every other leaf within 0.75 of its e'). Its first grid spacing at the edge is 18.5 times l
        there, and |theta'| differs by a factor 19.5 across it: the mechanism 07b's Part A measured beside V4's windows.

        At the commit the edge rule refuses that leaf, it is bisected, and the call is honest (0.0044 of its abserr),
        converged, with 16 regions. The parent's call is reproduced bit for bit by the commit with the rule refusing
        nothing.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): the
        call makes 14 Clenshaw-Curtis solves (the window's pieces and their checks), so 07d acts in it. At 07d's code
        the call is still converged and honest, with 16 regions and one edge refusal (0.0044 of its abserr; value
        0x1.9c4de6ecf4452p-14 against 07c's 0x1.9c4de6ecf3c4fp-14), and is asserted as before. The parent's path
        (the rule refusing nothing) is now run on 07c's Clenshaw-Curtis rule as well (cc_rule_07c()), since the
        hex it reproduces is 07a's, whose window pieces are CC_{2N-1}'s: there it reproduces B_PARENT_VALUE and
        B_PARENT_ABSERR bit for bit, as before. (At 07d's code with the rule refusing nothing the carrier is still
        returned trusted and the call is 7.62 times its abserr from B_REF.)
        """
        data = adaptive_levin_sincos(
            (0.0, 1.0), Q_F, Q_THETA, atol=B_ATOL, rtol=B_RTOL, chebyshev_order=Q_ORDER
        )
        self.assertEqual(len(data["stationary_points"]), 1)
        self.assertEqual(data["stationary_points"][0]["kind"], "root")
        self.assertEqual(len(data["stationary_windows"]), 1)
        window = data["stationary_windows"][0]
        self.assertEqual(window["lo"], B_CARRIER[1])
        self.assertGreater(data["num_edge_rule_refusals"], 0)
        self.assertEqual(data["num_range_rule_refusals"], 0)
        self.assertTrue(data["converged"])
        self.assertHonest(data, B_REF)

        # the parent's carrying leaf touches the window, and the edge rule refuses it
        refused, rule, q = _leaf_rate_rule(
            B_CARRIER, Q_ORDER, [Q_THETA["theta_deriv"]], (B_CARRIER[1],)
        )
        self.assertTrue(refused)
        self.assertEqual(rule, "edge")
        self.assertGreater(q, 5.0 * EDGE_MAX)

        # the parent: the rule refusing nothing, on 07c's Clenshaw-Curtis rule (07d)
        with mock.patch.object(
            lq, "_leaf_rate_rule", return_value=_REFUSE_NOTHING
        ), cc_rule_07c():
            parent = adaptive_levin_sincos(
                (0.0, 1.0),
                Q_F,
                Q_THETA,
                atol=B_ATOL,
                rtol=B_RTOL,
                chebyshev_order=Q_ORDER,
            )
        self.assertEqual(parent["value"].hex(), B_PARENT_VALUE.hex())
        self.assertEqual(parent["abserr"].hex(), B_PARENT_ABSERR.hex())
        self.assertTrue(parent["converged"])
        self.assertGreater(abs(parent["value"] - B_REF), 5.0 * parent["abserr"])
        self.assertIn(B_CARRIER, [(r.start, r.end) for r in parent["regions"]])


class TestRangeMechanism(unittest.TestCase, _Honest):
    def test_c_leaf_with_a_wide_rate_range(self):
        """
        (c) (b)'s integrand on the span of (b)'s carrying leaf, [0x1.a317190ba9ba3p-2, 0x1.a317190ba9ba3p-1], as a
        call of its own: order 24, atol 1e-8, rtol 1e-7. theta' = c (x - x_s) has no zero on the span (its upper end
        lies 1.03e-4 below x_s), so nothing is located and there is no window; |theta'| ranges by 3,984 over the
        span's nodes. The parent returns the span as one Levin leaf, trusted at depth 0, converged, 102.8 times its
        abserr from the reference. At the commit the range rule refuses it, and the call is honest (6.0e-6 of its
        abserr), converged, with 10 regions.
        """
        data = adaptive_levin_sincos(
            C_SPAN, Q_F, Q_THETA, atol=C_ATOL, rtol=C_RTOL, chebyshev_order=Q_ORDER
        )
        self.assertEqual(data["stationary_points"], [])
        self.assertEqual(data["stationary_windows"], [])
        self.assertGreater(
            _node_range(Q_THETA["theta_deriv"], *C_SPAN, Q_ORDER), 10.0 * RANGE_MAX
        )
        self.assertGreater(data["num_range_rule_refusals"], 0)
        self.assertEqual(data["num_edge_rule_refusals"], 0)
        self.assertTrue(data["converged"])
        self.assertHonest(data, C_REF)

        with mock.patch.object(lq, "_leaf_rate_rule", return_value=_REFUSE_NOTHING):
            parent = adaptive_levin_sincos(
                C_SPAN, Q_F, Q_THETA, atol=C_ATOL, rtol=C_RTOL, chebyshev_order=Q_ORDER
            )
        self.assertEqual(parent["value"].hex(), C_PARENT_VALUE.hex())
        self.assertEqual(parent["abserr"].hex(), C_PARENT_ABSERR.hex())
        self.assertEqual(parent["num_regions"], 1)
        self.assertTrue(parent["converged"])
        self.assertGreater(abs(parent["value"] - C_REF), 50.0 * parent["abserr"])


class TestAtTheDepthCap(unittest.TestCase):
    def test_d_refused_at_depth_max(self):
        """
        (d) theta = 2 e^{4x} on [0, 1.5], f = (e^{-x}, cos x), atol 1e-14, rtol 1e-7, depth_max 1: no window; the
        root is bisected for its error, and its two halves, at the cap, are checked. The parent trusts both (r =
        1.6e-5 and 8.5e-4) and is converged. At the commit the range rule refuses both (|theta'| ranges by e^3 = 20
        over each): each is returned untrusted at depth_max, keeps its own value V_P, and declares as the tree
        declares an untrusted leaf there with r < 1, max(residual, delta1) / (1 - r); the call is not converged.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): the
        halves' checks make Clenshaw-Curtis solves (6, two of them sent there by the gate at 10 pi), so 07d acts in
        the call. At 07d's code the call is asserted as before: unconverged, both leaves refused and untrusted at
        the cap, each declaring as the tree declares; its abserr is now 0x1.8075b9a053779p-33 (07c's
        0x1.8075aba2113acp-33). The parent's hex is 07b's parent's, so the parent's path is run on 07c's
        Clenshaw-Curtis rule (cc_rule_07c()), where it reproduces it bit for bit. That the declarations are the same
        with and without the rule is asserted at 07d's code, against the commit with the rule refusing nothing
        (converged there, as at the parent).
        """
        data = adaptive_levin_sincos(
            D_SPAN, D_F, D_THETA, atol=D_ATOL, rtol=D_RTOL, depth_max=D_DEPTH_MAX
        )
        self.assertEqual(data["stationary_windows"], [])
        self.assertFalse(data["converged"])
        self.assertEqual(data["num_range_rule_refusals"], 2)
        self.assertEqual(data["num_edge_rule_refusals"], 0)
        self.assertEqual(data["num_untrusted_regions"], 2)
        self.assertEqual(data["num_regions"], 2)
        for region in data["regions"]:
            with self.subTest(start=region.start):
                self.assertEqual(region.depth, D_DEPTH_MAX)
                self.assertIs(region.trusted, False)
                self.assertLess(region.convergence_ratio, lq._LEVIN_TRUST_MAX_RATIO)
                self.assertEqual(region.value_kind, "own")
                self.assertEqual(region.value, region.value_own)
                self.assertEqual(
                    region.declared_resolution,
                    max(region.abserr, region.delta1)
                    / (1.0 - region.convergence_ratio),
                )
                self.assertGreater(
                    _node_range(
                        D_THETA["theta_deriv"],
                        region.start,
                        region.end,
                        lq.DEFAULT_LEVIN_CHEBSHEV_ORDER,
                    ),
                    RANGE_MAX,
                )

        with mock.patch.object(
            lq, "_leaf_rate_rule", return_value=_REFUSE_NOTHING
        ), cc_rule_07c():
            parent = adaptive_levin_sincos(
                D_SPAN, D_F, D_THETA, atol=D_ATOL, rtol=D_RTOL, depth_max=D_DEPTH_MAX
            )
        self.assertEqual(parent["value"].hex(), D_PARENT_VALUE.hex())
        self.assertEqual(parent["abserr"].hex(), D_PARENT_ABSERR.hex())
        self.assertTrue(parent["converged"])
        # the declarations are the same; only the value returned (V_P, not the quarters sum) and the flag differ
        # (07d: against the commit with the rule refusing nothing, both on 07d's Clenshaw-Curtis rule)
        with mock.patch.object(lq, "_leaf_rate_rule", return_value=_REFUSE_NOTHING):
            unrefused = adaptive_levin_sincos(
                D_SPAN, D_F, D_THETA, atol=D_ATOL, rtol=D_RTOL, depth_max=D_DEPTH_MAX
            )
        self.assertTrue(unrefused["converged"])
        self.assertEqual(data["abserr"].hex(), unrefused["abserr"].hex())


class TestNotTouched(unittest.TestCase):
    def _spy(self):
        calls = []
        original = lq._leaf_rate_rule

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            calls.append(out)
            return out

        return calls, spy

    def test_e1_constant_rate(self):
        """
        (e) theta = 1e3 x on [0, 1], f = (e^{-x}, cos 30x), atol 1e-14, rtol 1e-7: no window, |theta'| constant,
        four Levin leaves at depth 2, each trusted by the tree's test and then passed by the range rule at q = 1. The
        call is bit-identical to the parent's, and both counts are 0.
        """
        calls, spy = self._spy()
        with mock.patch.object(lq, "_leaf_rate_rule", side_effect=spy):
            data = adaptive_levin_sincos(
                (0.0, 1.0), E1_F, E1_THETA, atol=E_ATOL, rtol=E_RTOL
            )
        self.assertEqual(data["value"].hex(), E1_PARENT_VALUE.hex())
        self.assertEqual(data["abserr"].hex(), E1_PARENT_ABSERR.hex())
        self.assertTrue(data["converged"])
        self.assertEqual(data["num_edge_rule_refusals"], 0)
        self.assertEqual(data["num_range_rule_refusals"], 0)
        self.assertEqual(data["stationary_windows"], [])
        levin = [r for r in data["regions"] if r.type == "Levin"]
        self.assertEqual(len(levin), data["num_regions"])
        self.assertTrue(all(r.trusted is True for r in levin))
        self.assertGreaterEqual(len(calls), len(levin))
        self.assertTrue(all(call == (False, "range", 1.0) for call in calls))

    def test_e2_clenshaw_curtis_leaves_untested(self):
        """
        (e) theta = 0.04 e^x on [0, 6], f = (e^{-x}, cos 3x), atol 1e-14, rtol 1e-7: the phase varies by 16.1 rad
        over the span, below 6 pi, so every leaf is a Clenshaw-Curtis leaf; the leaf [0, 3] is trusted with |theta'|
        ranging by e^3 = 20.1 over its nodes. Clenshaw-Curtis leaves are not tested: the rule is never called, the
        call is bit-identical to the parent's, and both counts are 0.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): every
        leaf is a Clenshaw-Curtis leaf, so 07d acts in the call (2 regions at 07d's code, 6 at 07b's parent; value
        -0x1.24a89a8b8f0cbp-1, abserr 0x1.568170599c493p-45). What the test is for -- the rule never called on a
        Clenshaw-Curtis leaf, both counts 0, a trusted leaf [0, 3] of range above RANGE_MAX -- is asserted at 07d's
        code; the bit-identity with the parent is asserted on 07c's Clenshaw-Curtis rule (cc_rule_07c()), with the
        same assertions besides.
        """
        for path in ("07d", "07c"):
            with self.subTest(path=path):
                calls, spy = self._spy()
                with mock.patch.object(lq, "_leaf_rate_rule", side_effect=spy), (
                    cc_rule_07c() if path == "07c" else contextlib.nullcontext()
                ):
                    data = adaptive_levin_sincos(
                        (0.0, 6.0), F_23, E2_THETA, atol=E_ATOL, rtol=E_RTOL
                    )
                if path == "07c":
                    self.assertEqual(data["value"].hex(), E2_PARENT_VALUE.hex())
                    self.assertEqual(data["abserr"].hex(), E2_PARENT_ABSERR.hex())
                self.assertTrue(data["converged"])
                self.assertEqual(data["num_edge_rule_refusals"], 0)
                self.assertEqual(data["num_range_rule_refusals"], 0)
                self.assertEqual(calls, [])
                self.assertTrue(all(r.type == "direct" for r in data["regions"]))
                wide = [
                    r
                    for r in data["regions"]
                    if r.trusted is True
                    and _node_range(
                        E2_THETA["theta_deriv"],
                        r.start,
                        r.end,
                        lq.DEFAULT_LEVIN_CHEBSHEV_ORDER,
                    )
                    > RANGE_MAX
                ]
                self.assertEqual([(r.start, r.end) for r in wide], [(0.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
