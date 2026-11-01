"""
The Clenshaw-Curtis leaves' node errors, carried signed (prompts/levin-phase-floor/
06g-leaf-values-and-node-sum.md, section 2 tests 4-6).

Each fallback leaf's floor carries |E|, the signed first-order error of its computed nodes, and
before 06g the call summed the floors, so sum |E|. Since 06g a leaf enters the call's error with
|E| taken out of its floor, e'_i = max(resolution, phi_i - |E_i|) + truncation, and the call carries
|sum E| once: abserr = sum e'_i + |sum E|. A call that the leaves' errors e'_i alone would bring
within its request, but |sum E| holds above it, stops unconverged, flagged node_limited. The leaf
keeps |E| in the floor its floor-limited test compares its residual against.

The parent is the dispatch commit's module (test_levin_leaf_values.load_parent()). The numbers
quoted are docs/levin-phase-floor/06g/calibration/out/cells.txt's.

No Ray and no datastore.
"""

import math
import unittest

import mpmath

import AdaptiveLevin.levin_quadrature as lq
from AdaptiveLevin.tests.test_levin_leaf_values import load_parent, reference, run
from AdaptiveLevin.tests.test_levin_roundoff_exit import levin_quadrature_problems


def _grz1000():
    """test_levin_quadrature.py's GRZIntegral(1000): cos(1000 arctan x) / (1 + x^2) on [-1, 1]."""
    for name, span, f, theta, kwargs, basis, raises in levin_quadrature_problems():
        if name == "GRZIntegral(1000)":
            return span, f, theta, kwargs
    raise KeyError("GRZIntegral(1000)")


def _far_sine(x0, length, c):
    """
    sin(c (x - x0)) on [x0, x0 + length], f = (1, 0), theta' supplied: at large x0 every node's
    rounding is large, the node term dominates each fallback leaf's floor, and the integral is
    (1 - cos(c (b - x0))) / c with b the double x0 + length.
    """
    span = (x0, x0 + length)
    f = [lambda x: 1.0, lambda x: 0.0]
    theta = {
        "theta": lambda x: c * (x - x0),
        "theta_deriv": lambda x: c + 0.0 * x,
    }

    def expression():
        width = mpmath.mpf(span[1]) - mpmath.mpf(span[0])
        return (1 - mpmath.cos(mpmath.mpf(c) * width)) / mpmath.mpf(c)

    return span, f, theta, expression


# Test 5's cell: x0 = 4e7, length 3.6, c = 0.9, rtol 1e-9.
TEST5 = (4.0e7, 3.6, 0.9)
TEST5_KWARGS = dict(atol=1e-40, rtol=1e-9)
# Test 6's cell: x0 = 3e8, length 1.25, c = 1.25, rtol 1e-10.
TEST6 = (3.0e8, 1.25, 1.25)
TEST6_KWARGS = dict(atol=1e-40, rtol=1e-10)


def _budget_err(region):
    """e'_i restated from the region's fields: max(resolution, floor less |E|) + truncation."""
    resolution = (
        region.declared_resolution
        if region.declared_resolution is not None
        else region.abserr
    )
    floor = (
        region.phase_err_less_node
        if region.node_error_signed is not None
        else region.phase_err
    )
    base = max(resolution, floor)
    if region.abserr_truncation is not None:
        base = base + region.abserr_truncation
    return base


class TestSignedNodeSum(unittest.TestCase):

    def test_signed_node_sum(self):
        """
        Test 4. GRZIntegral(1000) at rtol 1e-10, order 12 (cells.txt): 238 leaves, 232 of them
        fallback, whose E are of both signs (113 positive, 119 negative). node_error_signed_sum
        is the exact sum of the returned regions' E, node_error_abs_sum the exact sum of |E|, and
        abserr is sum e'_i (in ascending x) + |sum E|, each e'_i restated from the region's
        fields. sum |E| = 1.297e-15 against |sum E| = 5.20e-18, a ratio of 250; the parent's
        combination on the same partition (sum of total_err) is 4.3015e-14 against abserr
        4.1724e-14 (1.031), the node term being a small part of these leaves' floors. The parent
        itself returns the same 238 leaves with abserr 4.3015e-14.
        """
        span, f, theta, kwargs = _grz1000()
        new = run(lq, span, f, theta, **kwargs)
        node_errors = [
            r.node_error_signed
            for r in new["regions"]
            if r.node_error_signed is not None
        ]
        self.assertGreaterEqual(len(node_errors), 100)
        self.assertGreater(sum(1 for e in node_errors if e > 0), 0)
        self.assertGreater(sum(1 for e in node_errors if e < 0), 0)
        self.assertEqual(new["node_error_signed_sum"], math.fsum(node_errors))
        self.assertEqual(
            new["node_error_abs_sum"], math.fsum(abs(e) for e in node_errors)
        )
        self.assertEqual(new["abserr_node_signed"], abs(math.fsum(node_errors)))
        eprime = 0.0
        for region in new["regions"]:
            self.assertEqual(region.budget_err, _budget_err(region), region)
            eprime = eprime + _budget_err(region)
        self.assertEqual(new["abserr"], eprime + abs(math.fsum(node_errors)))
        self.assertFalse(new["node_limited"])
        # the cancellation, and the parent's combination on the same partition
        self.assertGreater(
            new["node_error_abs_sum"], 100.0 * abs(new["node_error_signed_sum"])
        )
        parent_combination = 0.0
        for region in new["regions"]:
            parent_combination = parent_combination + region.total_err
        self.assertGreater(parent_combination, new["abserr"])
        self.assertAlmostEqual(parent_combination / new["abserr"], 1.031, places=3)
        old = run(load_parent(), span, f, theta, **kwargs)
        self.assertEqual(old["num_regions"], new["num_regions"])
        self.assertEqual(old["abserr"], parent_combination)

    def test_node_limited_stop(self):
        """
        Test 5. sin(0.9 (x - 4e7)) on [4e7, 4e7 + 3.6], rtol 1e-9 (cells.txt). Under 06g the call
        stops node-limited at depth 4 on 5 fallback leaves (39 solves): sum e'_i = 1.278e-9 meets the
        request 2.217e-9, and |sum E| = 5.366e-9 (sum |E| = 5.383e-9) holds it above; abserr
        6.645e-9, unconverged, against a true error of 4.655e-9. The parent, on the same input,
        chases on to depth 11 on 12 leaves, every one floor-limited (47 solves), and stops
        unconverged at abserr 4.825e-9 (true error 4.799e-9). The correction by -sum E, measured
        and not applied, would leave 7.1e-10.
        """
        span, f, theta, expression = _far_sine(*TEST5)
        ref, _ = reference(expression)
        new = run(lq, span, f, theta, **TEST5_KWARGS)
        self.assertTrue(new["node_limited"])
        self.assertFalse(new["converged"])
        self.assertLess(new["max_depth"], lq.DEFAULT_LEVIN_MAX_DEPTH)
        request = max(TEST5_KWARGS["atol"], TEST5_KWARGS["rtol"] * abs(new["value"]))
        eprime = new["abserr"] - new["abserr_node_signed"]
        self.assertLessEqual(eprime, request)
        self.assertGreater(new["abserr"], request)
        self.assertLessEqual(abs(new["value"] - ref), new["abserr"])
        self.assertTrue(all(r.trusted is not False for r in new["regions"]))

        old = run(load_parent(), span, f, theta, **TEST5_KWARGS)
        self.assertGreater(old["max_depth"], new["max_depth"])
        self.assertEqual(old["max_depth"], 11)
        self.assertEqual(old["num_regions"], 12)
        self.assertFalse(old["converged"])

    def test_floor_limited_keeps_own_node_error(self):
        """
        Test 6. sin(1.25 (x - 3e8)) on [3e8, 3e8 + 1.25], rtol 1e-10 (cells.txt): the root is a
        fallback leaf whose nested-pair residual 7.997e-10 is within its whole floor 1.181e-9 but
        above its floor less |E|, 1.446e-11 (|E| = 1.167e-9). It is found floor-limited at depth 0:
        its residual is compared with the floor that holds its own node error. Unconverged, not
        node-limited, abserr 1.966e-9 (the parent's 1.181e-9, max(residual, floor), now
        residual + |E|), true error 1.167e-9; the correction by -sum E would leave 1.2e-16.
        """
        span, f, theta, expression = _far_sine(*TEST6)
        ref, _ = reference(expression)
        new = run(lq, span, f, theta, **TEST6_KWARGS)
        (region,) = new["regions"]
        self.assertEqual(region.type, "direct")
        self.assertEqual(region.depth, 0)
        self.assertTrue(region.phase_limited)
        self.assertGreater(region.abserr, region.phase_err_less_node)
        self.assertLessEqual(region.abserr, region.phase_err)
        self.assertFalse(new["node_limited"])
        self.assertFalse(new["converged"])
        self.assertLessEqual(abs(new["value"] - ref), new["abserr"])


if __name__ == "__main__":
    unittest.main()
