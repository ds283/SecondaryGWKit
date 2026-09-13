import unittest
from math import fmod, fabs, floor, nextafter, ulp

from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.range_reduce_mod_2pi import simple_mod_2pi
from LiouvilleGreen.WKBtools import WKB_mod_2pi


# The pre-fix implementations, kept here as the reference the "remainder does not move"
# test scores against. They are *not* the production code and must not be used as one: they
# carry [13-wkb-mod-2pi-cycle-count-inconsistent] in their cycle count, which is exactly what
# prompts/phase-representation prompt 01 removed. Only their *remainder* is quoted.
def _legacy_WKB_mod_2pi(theta: float):
    theta_mod_2pi = fmod(theta, TWO_PI)
    theta_div_2pi = int(floor(fabs(theta) / TWO_PI))

    if theta < 0.0:
        theta_div_2pi = -theta_div_2pi

    if theta_mod_2pi > 0:
        theta_div_2pi = theta_div_2pi + 1
        theta_mod_2pi = theta_mod_2pi - TWO_PI

    return theta_div_2pi, theta_mod_2pi


def _legacy_simple_mod_2pi(num: float):
    mod_2pi = fmod(fabs(num), TWO_PI)
    div_2pi = int(floor(fabs(num) / TWO_PI))

    if num < 0.0:
        div_2pi = -div_2pi
        mod_2pi = -mod_2pi

    return div_2pi, mod_2pi


def _adversarial_thetas():
    """
    theta values deliberately placed just *below* an integer number of cycles, which is where
    the rounded division of the old cycle count crossed the integer and floor() returned one
    cycle too many. Random draws find these at the half-ulp rate of |theta|/2pi -- 25 in
    400,000 at |theta| ~ 4e12 (docs/gktk-remedial-verification.md §3.7) -- so the test walks
    downwards from N * TWO_PI with nextafter instead of sampling.
    """
    out = []
    for exponent in range(0, 13):
        # a spread of large cycle counts N at each decade of |theta|
        base = 10.0**exponent
        for multiplier in (1.0, 1.7, 2.3, 3.9, 6.1, 9.3):
            N = floor(multiplier * base / TWO_PI)
            if N < 1:
                continue
            theta = N * TWO_PI
            for _ in range(24):
                out.append(theta)
                out.append(-theta)
                theta = nextafter(theta, 0.0)

    return out


def _sweep_thetas():
    out = [0.0, -0.0]
    for exponent in range(0, 13):
        base = 10.0**exponent
        for multiplier in (
            1.0,
            1.2345,
            2.7182818,
            3.1415926,
            4.6692,
            6.2831853,
            7.389,
            9.8696,
        ):
            theta = multiplier * base
            out.append(theta)
            out.append(-theta)

    # the §3.7 worked example, by value: the one production sample at LambdaCDM k = 3e8/Mpc
    # whose stored pair reconstructed theta - 2*pi
    out.append(-3832989103139.361)
    out.append(3832989103139.361)

    out.extend(_adversarial_thetas())
    return out


class TestWKBModTwoPi(unittest.TestCase):
    """
    The (cycle count, remainder) representation stored by the WKB producers must reconstruct
    its own phase. [13-wkb-mod-2pi-cycle-count-inconsistent].
    """

    def test_WKB_mod_2pi_reconstructs_theta(self):
        for theta in _sweep_thetas():
            div_2pi, mod_2pi = WKB_mod_2pi(theta)
            recon = div_2pi * TWO_PI + mod_2pi
            err = fabs(recon - theta)

            # never off by a whole cycle -- the defect's signature
            self.assertLess(
                err,
                TWO_PI / 2.0,
                msg=(
                    f"whole-cycle inconsistency at theta={theta!r}: "
                    f"div={div_2pi}, mod={mod_2pi!r}, recon - theta = {recon - theta!r} "
                    f"({(recon - theta) / TWO_PI:.6f} cycles)"
                ),
            )

            # and correct to one ulp of theta: div * TWO_PI is itself a rounded product, so
            # one ulp is the floor, not a slack tolerance
            self.assertLessEqual(
                err,
                ulp(max(fabs(theta), 1.0)),
                msg=f"reconstruction off by more than 1 ulp at theta={theta!r}",
            )

    def test_simple_mod_2pi_reconstructs_num(self):
        for num in _sweep_thetas():
            div_2pi, mod_2pi = simple_mod_2pi(num)
            recon = div_2pi * TWO_PI + mod_2pi
            err = fabs(recon - num)

            self.assertLess(
                err,
                TWO_PI / 2.0,
                msg=(
                    f"whole-cycle inconsistency at num={num!r}: "
                    f"div={div_2pi}, mod={mod_2pi!r}, recon - num = {recon - num!r}"
                ),
            )
            self.assertLessEqual(
                err,
                ulp(max(fabs(num), 1.0)),
                msg=f"reconstruction off by more than 1 ulp at num={num!r}",
            )

    def test_production_regression_sample(self):
        """
        The one LambdaCDM k = 3e8/Mpc Green's-function sample of
        docs/gktk-remedial-verification.md §3.7, by value. The old cycle count was
        610039162582; the exact quotient is 610039162581.99994.
        """
        theta = -3832989103139.361
        div_2pi, mod_2pi = WKB_mod_2pi(theta)

        self.assertEqual(div_2pi, -610039162581)
        self.assertEqual(mod_2pi, fmod(theta, TWO_PI))
        self.assertEqual(div_2pi * TWO_PI + mod_2pi - theta, 0.0)

    def test_adversarial_abscissae_are_actually_adversarial(self):
        """
        Confirm the construction in _adversarial_thetas() really does exercise the defect: the
        pre-fix implementation must fail on it, and by a whole cycle. If this test ever stops
        finding failures the sweep has stopped probing the half-ulp band and
        test_WKB_mod_2pi_reconstructs_theta has quietly become vacuous.
        """
        failures = 0
        for theta in _adversarial_thetas():
            div_2pi, mod_2pi = _legacy_WKB_mod_2pi(theta)
            if fabs(div_2pi * TWO_PI + mod_2pi - theta) > TWO_PI / 2.0:
                failures += 1

        self.assertGreater(
            failures,
            0,
            msg="the adversarial sweep no longer reproduces the pre-fix defect",
        )

    def test_remainder_is_bit_identical_to_the_old_implementation(self):
        """
        prompts/phase-representation README §2 (b): the remainder is exact and must not move.
        This is what guarantees that no stored G or T changes -- G_WKB and T_WKB are built from
        theta_mod_2pi, never from theta_div_2pi.
        """
        for theta in _sweep_thetas():
            _, mod_new = WKB_mod_2pi(theta)
            _, mod_old = _legacy_WKB_mod_2pi(theta)
            self.assertEqual(
                mod_new.hex(),
                mod_old.hex(),
                msg=f"WKB_mod_2pi remainder moved at theta={theta!r}",
            )

            _, smod_new = simple_mod_2pi(theta)
            _, smod_old = _legacy_simple_mod_2pi(theta)
            self.assertEqual(
                smod_new.hex(),
                smod_old.hex(),
                msg=f"simple_mod_2pi remainder moved at num={theta!r}",
            )

    def test_negative_remainder_convention_is_preserved(self):
        """
        WKB_mod_2pi's convention is theta_mod_2pi in (-2pi, 0]; simple_mod_2pi's remainder
        carries the sign of its argument. They deliberately differ (WKBtools.py:9-14).
        """
        for theta in _sweep_thetas():
            _, mod_2pi = WKB_mod_2pi(theta)
            self.assertLessEqual(mod_2pi, 0.0, msg=f"at theta={theta!r}")
            self.assertGreater(mod_2pi, -TWO_PI, msg=f"at theta={theta!r}")

            div_s, mod_s = simple_mod_2pi(theta)
            if theta > 0.0:
                self.assertGreaterEqual(mod_s, 0.0, msg=f"at num={theta!r}")
                self.assertGreaterEqual(div_s, 0, msg=f"at num={theta!r}")
            elif theta < 0.0:
                self.assertLessEqual(mod_s, 0.0, msg=f"at num={theta!r}")
                self.assertLessEqual(div_s, 0, msg=f"at num={theta!r}")
            self.assertLess(fabs(mod_s), TWO_PI, msg=f"at num={theta!r}")

    def test_cycle_count_matches_the_exact_floor(self):
        """
        The recovered count is the mathematically exact floor(|theta| / 2pi), scored against an
        integer arithmetic reference built from the exact fmod remainder.
        """
        from fractions import Fraction

        exact_two_pi = Fraction(TWO_PI)
        for theta in _sweep_thetas():
            if theta == 0.0:
                continue
            exact_n = int(Fraction(fabs(theta)) / exact_two_pi)

            div_2pi, mod_2pi = WKB_mod_2pi(theta)
            # undo the convention shift to recover the raw count
            raw = abs(div_2pi - 1) if (theta > 0.0 and mod_2pi < 0.0) else abs(div_2pi)
            self.assertEqual(
                raw,
                exact_n,
                msg=f"WKB_mod_2pi cycle count wrong at theta={theta!r}",
            )

            div_s, _ = simple_mod_2pi(theta)
            self.assertEqual(
                abs(div_s),
                exact_n,
                msg=f"simple_mod_2pi cycle count wrong at num={theta!r}",
            )


if __name__ == "__main__":
    unittest.main()
