import unittest
from math import sin, fmod, fabs

import mpmath as mp

from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.range_reduce_mod_2pi import simple_mod_2pi


class TestRangeReduce(unittest.TestCase):

    def test_representation_identity(self):
        # simple_mod_2pi must be an exact decomposition: num == div*2pi + mod, to rounding
        for num in [6.2948, 0.0, 1e-8, 3.0, 1e6, 1.0e12, -6.2948, -1e6, -1.0e12]:
            div_2pi, mod_2pi = simple_mod_2pi(num)
            self.assertAlmostEqual(
                div_2pi * TWO_PI + mod_2pi,
                num,
                delta=1e-9 * max(1.0, fabs(num)),
                msg=f"decomposition failed for num={num}",
            )

    def test_remainder_bounds_and_sign(self):
        for num in [6.2948, 3.0, 1e6, 1.0e12]:
            div_2pi, mod_2pi = simple_mod_2pi(num)
            self.assertGreaterEqual(mod_2pi, 0.0)
            self.assertLess(mod_2pi, TWO_PI)
            self.assertGreaterEqual(div_2pi, 0)

            # negating the argument must negate both components
            neg_div, neg_mod = simple_mod_2pi(-num)
            self.assertEqual(neg_div, -div_2pi)
            self.assertAlmostEqual(neg_mod, -mod_2pi, delta=1e-12)

    def test_known_value(self):
        div_2pi, mod_2pi = simple_mod_2pi(6.2948)
        self.assertEqual(div_2pi, 1)
        self.assertAlmostEqual(mod_2pi, 6.2948 - TWO_PI, delta=1e-15)

    def test_do_not_pre_reduce_before_sin(self):
        """
        Regression test for the rule documented in the range_reduce_mod_2pi module docstring:
        to evaluate sin(theta), hand theta to libm unreduced. Pre-reducing with fmod against a
        53-bit TWO_PI computes the remainder with respect to the wrong modulus, and the error
        grows with the number of cycles removed.

        This test pins the *ordering* of the two approaches, not particular error values, so it
        should be stable across platforms with a correctly-rounded libm.
        """
        mp.mp.dps = 60
        for exponent in [4, 7, 10, 13]:
            theta = 1.234567 * 10.0**exponent
            exact = float(mp.sin(mp.mpf(theta)))

            err_direct = fabs(sin(theta) - exact)
            err_prereduced = fabs(sin(fmod(theta, TWO_PI)) - exact)

            self.assertLessEqual(
                err_direct,
                max(err_prereduced, 1e-16),
                msg=(
                    f"pre-reduction beat libm at theta={theta:.6g} "
                    f"(direct={err_direct:.3g}, pre-reduced={err_prereduced:.3g}); "
                    "if this fires, re-check the guidance in range_reduce_mod_2pi.py"
                ),
            )
