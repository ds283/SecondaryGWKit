"""
Tests for ComputeTargets/spline_wrappers.py.

Both wrappers allow a fraction SPLINE_BOUND_SLACK of slack outside their declared range, where an
evaluation is softly clamped to the bound rather than rejected. The slack was written as
`1.01 * max_log_z` / `0.99 * min_log_z`, which is only outward-going while the bound is positive:
log(1+z) is negative for z < 0, and there the lower test moved *inward* and rejected a band lying
inside the declared range. `_outward` follows the sign of the bound instead, and reduces to
exactly the retired expressions whenever the bound is positive.

Offline: no Ray, no datastore.
"""

import unittest
from math import log, exp

import numpy as np
from scipy.interpolate import make_interp_spline

from ComputeTargets.spline_wrappers import (
    ZSplineWrapper,
    SPLINE_BOUND_SLACK,
    _outward,
)


def _wrapper(min_z: float, max_z: float, n: int = 200) -> ZSplineWrapper:
    """
    A wrapper around y = log(1+z), splined in log(1+z) over exactly [min_z, max_z]. The ordinate
    is linear in the abscissa, so the cubic interpolating spline reproduces it to rounding and the
    tests below can assert on bounds handling without an interpolation error to allow for.
    """
    log_x = np.linspace(log(1.0 + min_z), log(1.0 + max_z), n)
    return ZSplineWrapper(
        make_interp_spline(log_x, log_x),
        label="test",
        max_z=max_z,
        min_z=min_z,
        log_z=True,
    )


class TestOutward(unittest.TestCase):
    def test_positive_bounds_reproduce_the_retired_expressions_exactly(self):
        """
        Every other spline in this repository declares positive bounds, so this is the guarantee
        that the fix changed nothing for them: bit-identical, not merely close.
        """
        for bound in (1.0e-3, 0.5, 1.0, 11.4, 46.05, 32.3):
            with self.subTest(bound=bound):
                self.assertEqual(_outward(bound, +1), 1.01 * bound)
                self.assertEqual(_outward(bound, -1), 0.99 * bound)

    def test_negative_bounds_move_outward_not_inward(self):
        for bound in (-0.2107, -0.2744, -1.0e-4):
            with self.subTest(bound=bound):
                # downward means more negative, upward means less negative
                self.assertLess(_outward(bound, -1), bound)
                self.assertGreater(_outward(bound, +1), bound)
                # the retired expression went the wrong way
                self.assertGreater(0.99 * bound, bound)

    def test_zero_is_a_fixed_point(self):
        self.assertEqual(_outward(0.0, +1), 0.0)
        self.assertEqual(_outward(0.0, -1), 0.0)

    def test_the_slack_is_the_documented_fraction(self):
        self.assertAlmostEqual(
            _outward(10.0, +1) / 10.0 - 1.0, SPLINE_BOUND_SLACK, delta=1.0e-15
        )


class TestZSplineWrapperBounds(unittest.TestCase):
    def test_a_negative_min_z_is_evaluable_at_its_own_declared_floor(self):
        """
        The regression. With min_z = -0.24 the retired test rejected everything below
        log(1+z) = 0.99 * log(0.76) = -0.27166, i.e. z < -0.23782 -- so the declared floor itself,
        and a band just above it, raised instead of evaluating. z = -0.24 is the discriminating
        probe here; the others passed before the fix too.
        """
        w = _wrapper(-0.24, 1.0e4)
        for z in (-0.24, -0.2377, -0.2, -0.1, 0.0):
            with self.subTest(z=z):
                self.assertAlmostEqual(w(z), log(1.0 + z), delta=1.0e-12)

    def test_a_positive_min_z_is_evaluable_at_its_own_declared_floor(self):
        w = _wrapper(0.1, 1.0e4)
        for z in (0.1, 0.2, 1.0):
            with self.subTest(z=z):
                self.assertAlmostEqual(w(z), log(1.0 + z), delta=1.0e-12)

    def test_inside_the_slack_is_clamped_not_rejected(self):
        """Just outside a bound, the wrapper returns the bound's value rather than raising."""
        w = _wrapper(-0.24, 1.0e4)
        just_below = (
            exp(log(1.0 - 0.24) - 0.5 * SPLINE_BOUND_SLACK * abs(log(0.76))) - 1.0
        )
        self.assertLess(just_below, -0.24)
        self.assertAlmostEqual(w(just_below), log(0.76), delta=1.0e-12)

    def test_well_outside_still_raises_at_both_ends(self):
        """The guard must still be a guard, for negative and positive bounds alike."""
        for min_z, max_z in ((-0.24, 1.0e4), (0.1, 1.0e4)):
            w = _wrapper(min_z, max_z)
            with self.subTest(min_z=min_z, end="low"):
                with self.assertRaises(RuntimeError):
                    w(-0.5)
            with self.subTest(min_z=min_z, end="high"):
                with self.assertRaises(RuntimeError):
                    w(1.0e6)

    def test_the_error_message_recommends_a_limit_inside_the_range(self):
        """
        The message's "recommended limit" must lie inside the declared range, or it advises the
        reader to do the thing that just failed. With a negative bound the retired
        `1.05 * min_z` was *below* min_z.
        """
        w = _wrapper(-0.24, 1.0e4)
        with self.assertRaises(RuntimeError) as ctx:
            w(-0.5)
        message = str(ctx.exception)
        print(f"\n[spline bounds] {message}")
        recommended = float(message.split("z >= ")[1].rstrip(")"))
        self.assertGreater(recommended, -0.24)


if __name__ == "__main__":
    unittest.main()
