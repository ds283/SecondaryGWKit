"""
The stop below a refused leaf (prompts/levin-phase-floor/07c-chase-below-a-refused-leaf.md, section 1 and section 3 as
amended by section 9; the fix is Part A's variant (S)).

Below a leaf 07b's rate rules refuse, the leaves are refined whatever the budget says until the trust check trusts
them. Where they reach the noise level the check's r is random, about half of them fail it, and before 07c a call
could be chased to depth_max and returned unconverged ([07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]).
Now the outermost refused leaf above a leaf roots its subtree; each bisection of a leaf of the subtree at the noise
level (delta1 and delta2 both within _LEVIN_CHASE_NOISE_FACTOR times the noise yardstick nu_B of its check,
_chase_at_noise_level()) counts against the root; at _LEVIN_CHASE_STOP_LIMIT the root stops, and every leaf of the
subtree at the noise level, then and later, is returned floor-limited, declaring what the tree would declare for it
at depth_max. The result dict counts it ("num_chase_stops", "num_chase_stopped_leaves",
"num_chase_noise_bisections").

  (a) the noise level, directly, on _chase_at_noise_level();
  (b) the chase: a call unconverged at the depth cap at the parent, every depth-20 leaf below a refused leaf;
      converged and honest at the commit, the stop acting;
  (c) the stop's reach: leaves made after the stop and at the noise level are stopped; leaves of the stopped subtree
      not at the noise level are not, and are returned as at the parent; leaves at the noise level below a leaf the
      trust test refused, with no rate-rule refusal above them, are neither counted nor stopped;
  (d) the outermost root: a refused leaf below a refused leaf counts against the outer root, whose stop it is;
  (e) the declarations of stopped leaves, untrusted (r < 1 and r >= 1) and trusted, against the parent's for the
      same leaf at depth_max;
  (f) calls the stop must not touch: bit-identical to the parent, no root stopped.

The chase needs integrand noise that the round-off floor does not model (07c Part A's issues.md, items 1 and 12: on
the census's chase leaves a per-sample scatter of 3-6e-12 of the integrand). The problems here are
theta = A e^{kappa x} on [0, 1.5], f = (e^{-x}, cos x), each sample multiplied by 1 + eta n(x), with n(x) in [-1, 1)
a deterministic hash of x's bits (noise() below), eta = 1e-12 where noise is used. The noise moves the integral by at
most eta int_0^1.5 (e^{-x} + |cos x|) dx = 1.774e-12 (NOISE_BOUND), which the honesty assertions add to the true
error as the reference's own uncertainty.

The parent is the dispatch commit cff4237f's module, 07b's code. Its values quoted here were measured by running that
module (docs/levin-phase-floor/07c/code/compare_s_07c.py, out/compare_s_07c.txt). This module does not run the parent.
Where a test needs the parent's call it runs the commit with _LEVIN_CHASE_STOP_LIMIT patched to infinity, so that no
root ever stops: the count alone changes nothing, so that is the parent's code path, and the test asserts it
reproduces the parent's value and abserr bit for bit (and compare_s_07c.py checks it field by field). Which leaves a
stop reached at the stop and which after it, and which leaves were at the noise level, were read from an instrument
on the commit's module (docs/levin-phase-floor/07c/code/trace_07c.py, out/trace_07c.txt); the tests assert them
through the result's regions and counts, and the noise level by solving a leaf's halves afresh, as the driver
solves them, for its yardstick nu_B.

References: docs/levin-phase-floor/07c/code/test_references_07c.py (out/test_references_07c.json), mpmath's quad on
the pieces between consecutive zeros of sin(theta), at 30 and 45 digits, agreeing to 4.5e-33 relative or better.
No Ray and no datastore.

Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): every call
here that makes a Clenshaw-Curtis solve ((b)-(e), (c) item 3 and (f) item 2) is run on 07c's code path,
cc_rule_07c() (AdaptiveLevin/tests/test_levin_cc_estimate.py: the parent's Clenshaw-Curtis rule, CC_{2N-1} with
|CC_{2N-1} - CC_N|, and the gate at 6 pi), and so are the leaf solves _nu_B() makes; every assertion is unchanged.
These tests certify the stop below a refused leaf, a mechanism 07d does not change, on constructions made of 07c's
rule: the chase in (b) is driven by Clenshaw-Curtis leaves whose residual |CC_47 - CC_24| carries the integrand's
noise above their floor. 07d's noise term declares that noise, and at the new code the chase does not arise: (b)'s
call converges in 26 regions at depth at most 7, with three bisections at the noise level counted and no root
stopped (158 Clenshaw-Curtis solves, 18 route changes, the noise term setting all 158), true error 9.1e-15 against
abserr 6.8e-13. That is within its abserr against the reference, but below NOISE_BOUND (1.774e-12), the margin these
assertions add, so the honesty assertions with that margin could not be made there. (f) item 1 makes no
Clenshaw-Curtis solve and runs at the new code as before.
"""

import functools
import math
import unittest
from unittest import mock

import numpy as np

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.levin_quadrature import (
    _chase_at_noise_level,
    _leaf_rate_rule,
    _trust_noise_yardstick,
    adaptive_levin_sincos,
)
from AdaptiveLevin.tests.test_levin_cc_estimate import cc_rule_07c

CHASE_KEYS = (
    "num_chase_stops",
    "num_chase_stopped_leaves",
    "num_chase_noise_bisections",
)


# --- the problems ---------------------------------------------------------------------------------------------------
_MIX_1 = np.uint64(0x9E3779B97F4A7C15)
_MIX_2 = np.uint64(0xBF58476D1CE4E5B9)


def noise(x):
    """
    A deterministic noise n(x) in [-1, 1), a hash of the bits of the double x (a splitmix-style mixer), the same for
    a scalar and for an array element by element, so that the driver's vectorised sampling sees the function its
    scalar sampling sees.
    """
    a = np.asarray(x, dtype=np.float64)
    bits = np.atleast_1d(a).view(np.uint64)
    with np.errstate(over="ignore"):
        h = bits * _MIX_1
        h = h ^ (h >> np.uint64(31))
        h = h * _MIX_2
        h = h ^ (h >> np.uint64(29))
    n = (h >> np.uint64(11)).astype(np.float64) * (2.0**-52) - 1.0
    return n if a.ndim else float(n[0])


SPAN = (0.0, 1.5)
ETA = 1e-12
NOISE_BOUND = ETA * (1.0 - math.exp(-1.5) + math.sin(1.5))


def problem(A, kappa, eta):
    """theta = A e^{kappa x}, f = (e^{-x}, cos x), each sample times 1 + eta n(x)."""
    f = [
        lambda x: np.exp(-x) * (1.0 + eta * noise(x)),
        lambda x: np.cos(x) * (1.0 + eta * noise(x)),
    ]
    theta = {
        "theta": lambda x: A * np.exp(kappa * x),
        "theta_deriv": lambda x: A * kappa * np.exp(kappa * x),
    }
    return f, theta


# (b), (c) items 1 and 2, (d), (e): A = 20, kappa = 4, eta = 1e-12, atol 1e-9, rtol 1e-7, the default order and
# depth_max (20). theta' ranges by e^6 = 403 over the span, which the range rule refuses at depth 0; the phase varies
# by 8,049 rad.
B_A, B_KAPPA = 20.0, 4.0
B_ATOL, B_RTOL = 1e-9, 1e-7
B_REF = float.fromhex(
    "-0x1.5e0cdda29812cp-8"
)  # -5.34134303106296672246984486946646527987784678e-3
# The parent (cff4237f): 100 regions, converged False, 6 depth-20 leaves, each a Clenshaw-Curtis leaf returned
# untrusted.
B_PARENT_VALUE = float.fromhex("-0x1.5e0cdda293813p-8")
B_PARENT_ABSERR = float.fromhex("0x1.e76e24a93ede7p-31")
B_INNER_REFUSED = (0.75, 1.5)
# (c) item 1: leaves made at the stop (the children of the sixth counted bisection) and after it (from bisections of
# Levin leaves not at the noise level, made after the stop), each found at the noise level by its own check and so
# stopped (out/trace_07c.txt).
B_MADE_AT_THE_STOP = [(0.515625, 0.521484375), (0.521484375, 0.52734375)]
B_MADE_AFTER_THE_STOP = [
    (0.5625, 0.57421875),
    (0.57421875, 0.5859375),
    (0.65625, 0.66796875),
    (0.66796875, 0.6796875),
    (0.6796875, 0.69140625),
    (0.69140625, 0.697265625),
    (0.697265625, 0.703125),
]
# (c) item 2: Levin leaves of the stopped subtree not at the noise level (delta1 / nu_B from 12.9 to 183).
B_NOT_AT_THE_NOISE_LEVEL = [
    (0.0, 0.1875),
    (0.1875, 0.375),
    (0.375, 0.46875),
    (0.5859375, 0.609375),
    (0.609375, 0.65625),
    (0.703125, 0.75),
]
# (e): stopped leaves at depth 8, compared with the parent's call at depth_max 8.
E_DEPTH_MAX = 8
E_UNTRUSTED_R_AT_LEAST_1 = (0.533203125, 0.5390625)  # r = 1.30
E_UNTRUSTED_R_BELOW_1 = (0.521484375, 0.52734375)  # r = 0.35, delta1 above nu_B
E_TRUSTED = (0.5390625, 0.544921875)  # r = 0.04
E_PARENT_VALUE = float.fromhex("-0x1.5e0cdda293ceap-8")
E_PARENT_ABSERR = float.fromhex("0x1.f4dbf7103c802p-31")

# (c) item 3: A = 200, kappa = 1, eta = 1e-12, atol 1e-9, rtol 1e-7. theta' ranges by e^1.5 = 4.5 over the span: no
# leaf is refused by a rate rule.
C3_A, C3_KAPPA = 200.0, 1.0
C3_REF = float.fromhex("0x1.c08442ac320c2p-8")  # 6.8438208365437257760978671053481e-3
C3_PARENT_VALUE = float.fromhex("0x1.c08442ac306bbp-8")
C3_PARENT_ABSERR = float.fromhex("0x1.9d299a6b906c1p-33")

# (f) item 1: 07b's test (e1), theta = 1e3 x on [0, 1], f = (e^{-x}, cos 30x), atol 1e-14, rtol 1e-7: no refusal.
F1_F = [lambda x: np.exp(-x), lambda x: np.cos(30.0 * x)]
F1_THETA = {"theta": lambda x: 1.0e3 * x, "theta_deriv": lambda x: 1.0e3 + 0.0 * x}
F1_ATOL, F1_RTOL = 1e-14, 1e-7
F1_PARENT_VALUE = float.fromhex("0x1.eb56afc607e00p-11")
F1_PARENT_ABSERR = float.fromhex("0x1.72de5e19b6d87p-39")
# (f) item 2: A = 2, kappa = 4, no noise, atol 1e-14, rtol 1e-7: two range-rule refusals, one bisection at the noise
# level counted, the limit never reached.
F2_A, F2_KAPPA = 2.0, 4.0
F2_ATOL, F2_RTOL = 1e-14, 1e-7
F2_REF = float.fromhex("-0x1.b5db4757865acp-4")  # -0.106898573569557954652739104528678
F2_PARENT_VALUE = float.fromhex("-0x1.b5db4757865afp-4")
F2_PARENT_ABSERR = float.fromhex("0x1.d9395a192f7cbp-43")


# --- helpers --------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _b_call(parent=False, depth_max=lq.DEFAULT_LEVIN_MAX_DEPTH):
    """
    (b)'s call at the commit, or on the parent's code path (no root ever stops); since 07d both on 07c's
    Clenshaw-Curtis rule (cc_rule_07c(); see the module docstring).
    """
    f, theta = problem(B_A, B_KAPPA, ETA)
    kwargs = dict(atol=B_ATOL, rtol=B_RTOL, depth_max=depth_max)
    with cc_rule_07c():
        if parent:
            with mock.patch.object(lq, "_LEVIN_CHASE_STOP_LIMIT", math.inf):
                return adaptive_levin_sincos(SPAN, f, theta, **kwargs)
        return adaptive_levin_sincos(SPAN, f, theta, **kwargs)


def _region(data, span):
    found = [r for r in data["regions"] if (r.start, r.end) == span]
    if len(found) != 1:
        raise AssertionError(f"no single region {span} in the result")
    return found[0]


def _nu_B(f, theta, span, rtol):
    """
    The noise yardstick of a leaf's trust check, from its own solve and its halves' solved afresh as the driver
    solves them (break point (end + start) / 2), with delta1 recomputed from the same solves. Since 07d on 07c's
    Clenshaw-Curtis rule (cc_rule_07c()), as the calls it is compared with are.
    """
    basis = lq._Basis_SinCos(theta)
    a, b = span
    c = (b + a) / 2.0

    def solve(s):
        return lq._adaptive_levin_subregion(
            s,
            f,
            basis,
            None,
            chebyshev_order=lq.DEFAULT_LEVIN_CHEBSHEV_ORDER,
            rtol=rtol,
        )

    with cc_rule_07c():
        P, L, R = solve((a, b)), solve((a, c)), solve((c, b))
    nu = _trust_noise_yardstick(P["phase_err"], bool(P.get("is_direct")), L, R)
    return nu, abs(P["value"] - (L["value"] + R["value"]))


def _floor_limited_own(test, region):
    test.assertTrue(region.phase_limited)
    test.assertIsNone(region.trusted)
    test.assertEqual(region.value_kind, "own")
    test.assertEqual(region.value, region.value_own)


def _declared_at_depth_max(region):
    """What the tree declares for a checked leaf at depth_max, from the differences the region records."""
    r = region.convergence_ratio
    if r < 1.0:
        return max(region.abserr, region.delta1) / (1.0 - r)
    return max(region.abserr, region.delta1 + region.delta2)


class _Honest:
    def assertHonest(self, data, ref, margin=0.0):
        """Within abserr of the reference, abserr at least the true error, the margin added to the true error."""
        err = abs(data["value"] - ref) + margin
        self.assertLessEqual(
            err, data["abserr"], f"error {err:.3e} > abserr {data['abserr']:.3e}"
        )
        self.assertGreaterEqual(data["abserr"], err)


# --- (a) ------------------------------------------------------------------------------------------------------------
class TestNoiseLevel(unittest.TestCase):
    """(a) _chase_at_noise_level() directly."""

    NU = 3.0e-16

    def test_a1_at_the_threshold(self):
        """delta1 = delta2 = C nu_B is at the noise level; C is 4 (section 1.2), and so is anything below it."""
        self.assertEqual(lq._LEVIN_CHASE_NOISE_FACTOR, 4.0)
        at = lq._LEVIN_CHASE_NOISE_FACTOR * self.NU
        self.assertTrue(_chase_at_noise_level(at, at, self.NU))
        self.assertTrue(_chase_at_noise_level(0.0, at, self.NU))
        self.assertTrue(_chase_at_noise_level(at, 0.5 * at, self.NU))

    def test_a2_just_above(self):
        """Either difference the next double above C nu_B is not at the noise level."""
        at = lq._LEVIN_CHASE_NOISE_FACTOR * self.NU
        above = math.nextafter(at, math.inf)
        self.assertFalse(_chase_at_noise_level(above, at, self.NU))
        self.assertFalse(_chase_at_noise_level(at, above, self.NU))
        self.assertFalse(_chase_at_noise_level(above, above, self.NU))

    def test_a3_zero_yardstick(self):
        """nu_B = 0: both differences 0 is at the noise level; either positive is not."""
        tiny = math.nextafter(0.0, 1.0)
        self.assertTrue(_chase_at_noise_level(0.0, 0.0, 0.0))
        self.assertFalse(_chase_at_noise_level(tiny, 0.0, 0.0))
        self.assertFalse(_chase_at_noise_level(0.0, tiny, 0.0))

    def test_a4_not_finite(self):
        """A difference that is NaN is never at the noise level."""
        self.assertFalse(_chase_at_noise_level(float("nan"), 0.0, self.NU))
        self.assertFalse(_chase_at_noise_level(0.0, float("nan"), self.NU))


# --- (b) ------------------------------------------------------------------------------------------------------------
class TestTheChase(unittest.TestCase, _Honest):
    def test_b_chase_below_a_refused_leaf(self):
        """
        (b) theta = 20 e^{4x} on [0, 1.5], f = (e^{-x}, cos x) with noise eta = 1e-12, atol 1e-9, rtol 1e-7, the
        default order and depth_max 20. The range rule refuses the span at depth 0 (theta' ranges by e^6 = 403) and
        [0.75, 1.5] at depth 1 (e^3 = 20). Below them the Levin leaves are refined to where the phase varies by less
        than 6 pi, and the Clenshaw-Curtis leaves there reach the noise level: r is random, and their residual
        |CC_47 - CC_24| carries the integrand's noise above their floor, so the round-off exit is not considered.

        At the parent: 100 regions, unconverged, with 6 depth-20 leaves, every one a Clenshaw-Curtis leaf returned
        untrusted, every one below the refused leaf [0, 1.5]. At the commit: the root stops at the sixth bisection
        at the noise level and 20 leaves are stopped; 26 regions, depth at most 8, converged, honest (true error
        1.4e-14 against abserr 6.2e-10), no leaf returned untrusted.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged (see the module docstring for why, and for this call at
        the new code).
        """
        data = _b_call()
        self.assertTrue(data["converged"])
        self.assertHonest(data, B_REF, margin=NOISE_BOUND)
        self.assertEqual(data["num_range_rule_refusals"], 2)
        self.assertEqual(data["num_edge_rule_refusals"], 0)
        self.assertEqual(data["num_chase_stops"], 1)
        self.assertGreater(data["num_chase_stopped_leaves"], 0)
        self.assertEqual(data["num_chase_noise_bisections"], lq._LEVIN_CHASE_STOP_LIMIT)
        self.assertEqual(data["num_untrusted_regions"], 0)
        self.assertLess(data["max_depth"], lq.DEFAULT_LEVIN_MAX_DEPTH)

        # the span is refused by the range rule
        f, theta = problem(B_A, B_KAPPA, ETA)
        refused, rule, q = _leaf_rate_rule(
            SPAN,
            lq.DEFAULT_LEVIN_CHEBSHEV_ORDER,
            [theta["theta_deriv"]],
            (),
        )
        self.assertEqual((refused, rule), (True, "range"))
        self.assertGreater(q, 100.0)

        # the parent: no root ever stops
        parent = _b_call(parent=True)
        self.assertEqual(parent["value"].hex(), B_PARENT_VALUE.hex())
        self.assertEqual(parent["abserr"].hex(), B_PARENT_ABSERR.hex())
        self.assertFalse(parent["converged"])
        self.assertEqual(parent["num_chase_stops"], 0)
        self.assertEqual(parent["num_range_rule_refusals"], 2)
        at_cap = [r for r in parent["regions"] if r.depth == lq.DEFAULT_LEVIN_MAX_DEPTH]
        self.assertGreater(len(at_cap), 0)
        for region in at_cap:
            self.assertEqual(region.type, "direct")
            self.assertIs(region.trusted, False)
            self.assertTrue(SPAN[0] <= region.start < region.end <= SPAN[1])
        self.assertEqual(parent["num_untrusted_regions"], len(at_cap))


# --- (c) ------------------------------------------------------------------------------------------------------------
class TestTheStopsReach(unittest.TestCase, _Honest):
    def test_c1_leaves_made_after_the_stop(self):
        """
        (c) item 1. In (b)'s call the root stops at the 17th bisection, the sixth at the noise level below it. The two
        children of that bisection, and seven leaves made later from Levin leaves of the subtree not at the noise
        level, are each found at the noise level by their own check: each is returned floor-limited, with its own
        value. After the stop no leaf at the noise level below the root is bisected, so the bisections counted are
        exactly the limit's.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged (see the module docstring for why, and for this call at
        the new code).
        """
        data = _b_call()
        f, theta = problem(B_A, B_KAPPA, ETA)
        self.assertEqual(
            data["num_chase_noise_bisections"],
            lq._LEVIN_CHASE_STOP_LIMIT * data["num_chase_stops"],
        )
        self.assertGreaterEqual(
            data["num_chase_stopped_leaves"],
            len(B_MADE_AT_THE_STOP) + len(B_MADE_AFTER_THE_STOP),
        )
        for span in B_MADE_AT_THE_STOP + B_MADE_AFTER_THE_STOP:
            with self.subTest(span=span):
                region = _region(data, span)
                _floor_limited_own(self, region)
                self.assertLess(region.depth, lq.DEFAULT_LEVIN_MAX_DEPTH)
                nu, delta1 = _nu_B(f, theta, span, B_RTOL)
                self.assertEqual(delta1, region.delta1)
                self.assertTrue(_chase_at_noise_level(region.delta1, region.delta2, nu))

    def test_c2_not_at_the_noise_level(self):
        """
        (c) item 2. Six Levin leaves of the stopped subtree are not at the noise level (delta1 / nu_B from 12.9 to
        183): none is stopped. Each is returned trusted, with its quarters' value, exactly as the parent returns the
        same leaf: span, depth, value, errors, differences and declaration bit for bit.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged (see the module docstring for why, and for this call at
        the new code).
        """
        data = _b_call()
        parent = _b_call(parent=True)
        f, theta = problem(B_A, B_KAPPA, ETA)
        fields = (
            "depth",
            "type",
            "trusted",
            "phase_limited",
            "value",
            "value_own",
            "value_kind",
            "abserr",
            "phase_err",
            "delta1",
            "delta2",
            "convergence_ratio",
            "declared_resolution",
            "budget_err",
        )
        for span in B_NOT_AT_THE_NOISE_LEVEL:
            with self.subTest(span=span):
                region = _region(data, span)
                self.assertEqual(region.type, "Levin")
                self.assertFalse(region.phase_limited)
                self.assertIs(region.trusted, True)
                self.assertEqual(region.value_kind, "quarters")
                nu, delta1 = _nu_B(f, theta, span, B_RTOL)
                self.assertEqual(delta1, region.delta1)
                self.assertFalse(
                    _chase_at_noise_level(region.delta1, region.delta2, nu)
                )
                at_parent = _region(parent, span)
                for name in fields:
                    self.assertEqual(
                        getattr(region, name), getattr(at_parent, name), name
                    )

    def test_c3_below_a_trust_test_refusal(self):
        """
        (c) item 3. theta = 200 e^x on [0, 1.5], f = (e^{-x}, cos x) with noise eta = 1e-12, atol 1e-9, rtol 1e-7:
        theta' ranges by e^1.5 = 4.5, so no leaf is refused by a rate rule. Leaves the trust test refuses are
        bisected at the noise level beneath one another (18 bisections, out/trace_07c.txt): none is counted and none
        stopped, and the call is bit-identical to the parent's: 32 regions, converged, honest.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged. At the new code this call converges in 11 regions
        (62 Clenshaw-Curtis solves, 7 route changes, the noise term setting all 62), nothing counted or stopped, true
        error 8.3e-15 against abserr 5.3e-13: within its abserr, but below NOISE_BOUND, this module's margin.
        """
        f, theta = problem(C3_A, C3_KAPPA, ETA)
        with cc_rule_07c():
            data = adaptive_levin_sincos(SPAN, f, theta, atol=B_ATOL, rtol=B_RTOL)
        self.assertEqual(data["num_range_rule_refusals"], 0)
        self.assertEqual(data["num_edge_rule_refusals"], 0)
        self.assertEqual(tuple(data[key] for key in CHASE_KEYS), (0, 0, 0))
        self.assertEqual(data["value"].hex(), C3_PARENT_VALUE.hex())
        self.assertEqual(data["abserr"].hex(), C3_PARENT_ABSERR.hex())
        self.assertTrue(data["converged"])
        self.assertHonest(data, C3_REF, margin=NOISE_BOUND)


# --- (d) ------------------------------------------------------------------------------------------------------------
class TestOutermostRoot(unittest.TestCase):
    def test_d_the_outer_root_counts_and_stops(self):
        """
        (d) In (b)'s call the range rule refuses the span [0, 1.5] and, below it, [0.75, 1.5] (e^3 = 20 over its
        nodes). The inner refused leaf belongs to the outer root's subtree, so its bisection, at the noise level, is
        counted against the outer root (the first of its six), with five more in [0, 0.75]. One root stops, and its
        stop reaches leaves on both sides of 0.75: the two leaves of [0.75, 1.5] at depth 2, below the inner refused
        leaf, and leaves in [0, 0.75], which no stop of the inner leaf's could reach. So the one stop is the outer
        root's.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged (see the module docstring for why, and for this call at
        the new code).
        """
        data = _b_call()
        f, theta = problem(B_A, B_KAPPA, ETA)
        self.assertEqual(data["num_range_rule_refusals"], 2)
        inner = _leaf_rate_rule(
            B_INNER_REFUSED,
            lq.DEFAULT_LEVIN_CHEBSHEV_ORDER,
            [theta["theta_deriv"]],
            (),
        )
        self.assertEqual(inner[:2], (True, "range"))
        self.assertEqual(data["num_chase_stops"], 1)
        self.assertEqual(data["num_chase_noise_bisections"], lq._LEVIN_CHASE_STOP_LIMIT)
        for span in ((0.75, 1.125), (1.125, 1.5)):
            with self.subTest(span=span):
                region = _region(data, span)
                self.assertEqual(region.depth, 2)
                _floor_limited_own(self, region)
        outside = [
            r
            for r in data["regions"]
            if r.end <= B_INNER_REFUSED[0] and r.phase_limited and r.trusted is None
        ]
        self.assertGreater(len(outside), 0)


# --- (e) ------------------------------------------------------------------------------------------------------------
class TestDeclarations(unittest.TestCase):
    def test_e_stopped_leaves_declare_as_at_depth_max(self):
        """
        (e) Three leaves of (b)'s call stopped at depth 8: [0.533203125, 0.5390625], untrusted with r = 1.30;
        [0.521484375, 0.52734375], untrusted with r = 0.35 (delta1 above its nu_B); [0.5390625, 0.544921875], trusted
        with r = 0.04. Each returns its own value V_P. The untrusted ones declare max(residual, delta1 + delta2) and
        max(residual, delta1) / (1 - r), the trusted one its widened declaration max(residual, delta1) / (1 - r).
        The parent's call at depth_max 8 returns the same three leaves at the cap: each declares there exactly what
        the commit's stopped leaf declares, the untrusted ones with the same value V_P, the trusted one with its
        quarters' value.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged (see the module docstring for why, and for this call at
        the new code).
        """
        data = _b_call()
        parent = _b_call(parent=True, depth_max=E_DEPTH_MAX)
        self.assertEqual(parent["value"].hex(), E_PARENT_VALUE.hex())
        self.assertEqual(parent["abserr"].hex(), E_PARENT_ABSERR.hex())
        for span, trusted in (
            (E_UNTRUSTED_R_AT_LEAST_1, False),
            (E_UNTRUSTED_R_BELOW_1, False),
            (E_TRUSTED, True),
        ):
            with self.subTest(span=span):
                region = _region(data, span)
                at_parent = _region(parent, span)
                self.assertEqual(region.depth, E_DEPTH_MAX)
                _floor_limited_own(self, region)
                self.assertEqual(
                    region.declared_resolution, _declared_at_depth_max(region)
                )
                self.assertEqual(at_parent.depth, E_DEPTH_MAX)
                self.assertIs(at_parent.trusted, trusted)
                self.assertFalse(at_parent.phase_limited)
                self.assertEqual(
                    region.declared_resolution, at_parent.declared_resolution
                )
                self.assertEqual(region.value_own, at_parent.value_own)
                if trusted:
                    self.assertEqual(at_parent.value_kind, "quarters")
                else:
                    self.assertEqual(at_parent.value_kind, "own")
                    self.assertEqual(region.value, at_parent.value)
        self.assertGreaterEqual(
            _region(data, E_UNTRUSTED_R_AT_LEAST_1).convergence_ratio, 1.0
        )
        self.assertLess(_region(data, E_UNTRUSTED_R_BELOW_1).convergence_ratio, 1.0)
        self.assertLess(
            _region(data, E_TRUSTED).convergence_ratio, lq._LEVIN_TRUST_MAX_RATIO
        )


# --- (f) ------------------------------------------------------------------------------------------------------------
class TestNotTouched(unittest.TestCase, _Honest):
    def test_f1_no_refusal(self):
        """
        (f) item 1. 07b's (e1): theta = 1e3 x on [0, 1], f = (e^{-x}, cos 30x), atol 1e-14, rtol 1e-7. No refusal, so
        nothing is rooted: the three counts are 0 and the call is bit-identical to the parent's.
        """
        data = adaptive_levin_sincos(
            (0.0, 1.0), F1_F, F1_THETA, atol=F1_ATOL, rtol=F1_RTOL
        )
        self.assertEqual(data["num_range_rule_refusals"], 0)
        self.assertEqual(tuple(data[key] for key in CHASE_KEYS), (0, 0, 0))
        self.assertEqual(data["value"].hex(), F1_PARENT_VALUE.hex())
        self.assertEqual(data["abserr"].hex(), F1_PARENT_ABSERR.hex())
        self.assertTrue(data["converged"])

    def test_f2_refusals_below_the_limit(self):
        """
        (f) item 2. theta = 2 e^{4x} on [0, 1.5], f = (e^{-x}, cos x) without noise, atol 1e-14, rtol 1e-7. The range
        rule refuses two leaves; one bisection at the noise level is counted against their root, and the limit is
        never reached: no root stops, no leaf is stopped, and the call is bit-identical to the parent's, converged
        and honest.

        Since prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md (section 3, "Existing assertions"): run on
        07c's code path, cc_rule_07c(), assertions unchanged. At the new code no bisection at the noise level is
        counted (the count this test needs positive): 4 regions, 16 Clenshaw-Curtis solves, 3 route changes,
        converged and honest (true error 8.3e-17 against abserr 1.8e-14).
        """
        f, theta = problem(F2_A, F2_KAPPA, 0.0)
        with cc_rule_07c():
            data = adaptive_levin_sincos(SPAN, f, theta, atol=F2_ATOL, rtol=F2_RTOL)
        self.assertEqual(data["num_range_rule_refusals"], 2)
        self.assertEqual(data["num_chase_stops"], 0)
        self.assertEqual(data["num_chase_stopped_leaves"], 0)
        self.assertGreater(data["num_chase_noise_bisections"], 0)
        self.assertLess(data["num_chase_noise_bisections"], lq._LEVIN_CHASE_STOP_LIMIT)
        self.assertEqual(data["value"].hex(), F2_PARENT_VALUE.hex())
        self.assertEqual(data["abserr"].hex(), F2_PARENT_ABSERR.hex())
        self.assertTrue(data["converged"])
        self.assertHonest(data, F2_REF)


if __name__ == "__main__":
    unittest.main()
