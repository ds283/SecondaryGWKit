"""
Pins to tests the boundaries of the bundled SciPy/Amos Bessel routines that the transfer-function
remedial campaign's supported ``(nu, x_max)`` domain is derived from.

These tests are about SciPy, not about this repository's code. ``hankel1e``, ``jv`` and ``yv``
lose argument-reduction accuracy at large argument, and above certain thresholds fail *silently*:
finite, non-``NaN`` values that are nonetheless wrong by O(1) or more, sometimes exactly ``-0j``.
``RECONCILIATION.md`` C1 and ``DRAFT-PLAN.md`` Sec 4.4 measured those thresholds on SciPy 1.15.2 /
NumPy 2.2.4; the campaign's closed-form tail (``bessel_tail.py``, prompt 03) exists specifically so
that nothing in the two-region construction samples a Bessel routine anywhere near them.

A failure here does not mean a bound should be loosened. It means the underlying SciPy/NumPy build
changed and the supported domain this campaign derived from these numbers must be re-measured and,
if it moved, re-derived -- in particular ``LiouvilleGreen.tests.bessel_reference.SCIPY_REFERENCE_MAX_X``,
``SCIPY_REFERENCE_HIGH_ORDER_NU`` and ``SCIPY_REFERENCE_HIGH_ORDER_MAX_X``, and the near-region
crossover x_star (prompt 03) against them.
"""

import unittest

import numpy as np
from scipy.special import hankel1e, jv, yv

# Environment this module's numbers were measured against (RECONCILIATION.md header, prompt 01's
# log): Python 3.12.14, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0, Darwin 25.5.0 (arm64). Re-measured
# for this prompt on the same versions, 2026-09-10.


def _contiguous_usable_boundary(values_at, nu, x_hi=1.0e16, n=6000):
    """
    The largest x, on a geometric grid from just above the turning point to ``x_hi``, below which
    ``values_at(nu, xs)`` is contiguously finite and exceeds the ``1e-3`` plausibility floor
    (the true value of ``a_nu`` is always >= 1). Mirrors ``DRAFT-PLAN.md`` Sec 12.2 / this prompt's
    Sec 3 reproduction script exactly, including the >1e-3 floor and the 6000-point grid: matching
    the reproduction is what makes the pinned numbers reproducible.
    """
    lo = np.sqrt(max(nu * nu - 0.25, 1e-10))
    xs = np.geomspace(max(1.5 * lo, 1.0), x_hi, n)
    a = values_at(nu, xs)
    ok = np.isfinite(a) & (a > 1.0e-3)
    i = np.argmax(~ok) if (~ok).any() else len(ok)
    return xs[i - 1] if i > 0 else float("nan")


def _hankel1e_amplitude(nu, xs):
    with np.errstate(all="ignore"):
        return np.abs(np.sqrt(np.pi * xs / 2) * hankel1e(nu, xs))


def _jv_yv_amplitude(nu, xs):
    with np.errstate(all="ignore"):
        return np.hypot(jv(nu, xs), yv(nu, xs)) * np.sqrt(np.pi * xs / 2)


class TestScipyBesselDomain(unittest.TestCase):

    # --------------------------------------------------------------------------------------------
    # Fact 1 -- hankel1e(100.5, 1e9) is exactly -0j: finite, so an isfinite() guard passes, and it
    # produces -inf in a log-amplitude interpolant and -0.0 from angle().
    # --------------------------------------------------------------------------------------------

    def test_hankel1e_returns_exact_negative_zero_and_is_finite(self):
        """
        RECONCILIATION.md Sec1, DRAFT-PLAN.md Sec4.4: measured on SciPy 1.15.2,
        hankel1e(100.5, 1e9) == -0j exactly. np.isfinite(...) is True (so a finite-value guard does
        not catch it); np.log(np.abs(...)) is -inf; np.angle(...) is -0.0. The point is the
        combination -- a value that passes an isfinite() check and then poisons a log-amplitude
        interpolant and a phase tracker downstream. This is why bessel_near_region.py's guard
        (prompt 04) must be a two-sided plausibility band on a_nu, not isfinite().
        """
        with np.errstate(all="ignore"):
            value = hankel1e(100.5, 1.0e9)

        self.assertEqual(
            value,
            -0j,
            f"hankel1e(100.5, 1e9) = {value!r}; RECONCILIATION.md Sec1 measured exactly -0j on "
            "SciPy 1.15.2 -- if this no longer reproduces, the SciPy/Amos behaviour this campaign's "
            "guard design relies on has changed",
        )
        self.assertTrue(
            np.isfinite(value),
            "hankel1e(100.5, 1e9) is no longer finite -- the whole point of Fact 1 (a finite value "
            "that isfinite() cannot catch) no longer holds; re-derive the guard design",
        )
        with np.errstate(all="ignore"):
            log_amplitude = np.log(np.abs(value))
            angle = np.angle(value)
        self.assertEqual(
            log_amplitude,
            -np.inf,
            f"log(abs(hankel1e(100.5, 1e9))) = {log_amplitude!r}, expected -inf",
        )
        self.assertEqual(
            angle,
            -0.0,
            f"angle(hankel1e(100.5, 1e9)) = {angle!r}, expected -0.0",
        )

    # --------------------------------------------------------------------------------------------
    # Fact 2 -- the hankel1e usable boundaries, and the campaign-critical consequence that the
    # crossover x_star = 100*nu stays >= 1e3 below the boundary even at nu=1000.5.
    # --------------------------------------------------------------------------------------------

    def test_hankel1e_usable_boundary_low_order(self):
        """
        RECONCILIATION.md Sec1: contiguous-from-below usable range of
        |sqrt(pi*x/2) * hankel1e(nu, x)| (finite and > 1e-3) is ~2.247e15 for nu in {1/2, 5/2, 20.5}
        on SciPy 1.15.2. Measured here: see printed values. Asserted within a factor of 2 (not as an
        equality) so a minor Amos change does not spuriously fail this test while a decade-scale
        change -- which is what would actually invalidate the supported domain -- does.
        """
        expected = 2.247e15
        for nu in (0.5, 2.5, 20.5):
            boundary = _contiguous_usable_boundary(_hankel1e_amplitude, nu)
            self.assertGreater(
                boundary,
                expected / 2.0,
                f"hankel1e usable boundary at nu={nu} fell to {boundary:.4g}, more than a factor "
                f"of 2 below the measured {expected:.4g} (RECONCILIATION.md Sec1)",
            )
            self.assertLess(
                boundary,
                expected * 2.0,
                f"hankel1e usable boundary at nu={nu} rose to {boundary:.4g}, more than a factor "
                f"of 2 above the measured {expected:.4g} (RECONCILIATION.md Sec1)",
            )

    def test_hankel1e_usable_boundary_high_order(self):
        """
        RECONCILIATION.md Sec1: at nu in {100.5, 1000.5} the boundary collapses six decades, to
        ~7.13e8, on SciPy 1.15.2. Asserted within a factor of 2, as above.
        """
        expected = 7.13e8
        for nu in (100.5, 1000.5):
            boundary = _contiguous_usable_boundary(_hankel1e_amplitude, nu)
            self.assertGreater(
                boundary,
                expected / 2.0,
                f"hankel1e usable boundary at nu={nu} fell to {boundary:.4g}, more than a factor "
                f"of 2 below the measured {expected:.4g} (RECONCILIATION.md Sec1)",
            )
            self.assertLess(
                boundary,
                expected * 2.0,
                f"hankel1e usable boundary at nu={nu} rose to {boundary:.4g}, more than a factor "
                f"of 2 above the measured {expected:.4g} (RECONCILIATION.md Sec1)",
            )

    def test_crossover_stays_far_below_hankel1e_boundary(self):
        """
        README.md Sec2(e): the campaign's near-region sampler (prompt 04) never evaluates hankel1e
        above the crossover x_star ~ 100*nu, so it never approaches the boundaries above. This is
        the test of that claim, not a remark: at every supported order the ratio
        (usable boundary / x_star) is at least 1e3, i.e. hankel1e is usable for every order in the
        supported set at x_star. Measured ratio at nu=1000.5 (the worst case): ~7.1e3
        (7.13e8 / 1e5), against the plan's "at least 1e3" claim (DRAFT-PLAN.md Sec4.4).
        """
        MIN_RATIO = 1.0e3
        worst_ratio = None
        for nu in (0.5, 2.5, 20.5, 100.5, 1000.5):
            x_star = 100.0 * nu
            boundary = _contiguous_usable_boundary(_hankel1e_amplitude, nu)
            ratio = boundary / x_star
            if worst_ratio is None or ratio < worst_ratio:
                worst_ratio = ratio
            self.assertGreaterEqual(
                ratio,
                MIN_RATIO,
                f"hankel1e usable boundary / x_star at nu={nu} is only {ratio:.3g}, below the "
                f"required margin of {MIN_RATIO:.1g}: the tail crossover is no longer safely below "
                "the Amos failure boundary",
            )
        print(
            f"test_crossover_stays_far_below_hankel1e_boundary: worst boundary/x_star ratio "
            f"{worst_ratio:.4g} (required >= {MIN_RATIO:.1g})"
        )

    # --------------------------------------------------------------------------------------------
    # Fact 3 -- jv/yv become O(1)-relatively noisy above x ~ 2.5e15. This is RECONCILIATION.md C1
    # and is not in DRAFT-PLAN.md. The identity (2/pi)/(x*(J^2+Y^2)) = 1 + O(nu^2/x^2) is exact, so
    # any O(1) departure from 1 at large x is pure library error.
    # --------------------------------------------------------------------------------------------

    def test_jv_yv_rhs_identity_good_below_boundary(self):
        """
        RECONCILIATION.md C1: the identity (2/pi)/(x*(J_nu^2+Y_nu^2)) = 1 + O(nu^2/x^2) is exact
        for nu=1/2 (where the O(nu^2/x^2) term vanishes identically), so it must equal 1 to within
        1e-9 wherever jv/yv are accurate. Measured to agree with 1 to six places at
        x in {1e12, 1e14, 2e15} on SciPy 1.15.2.
        """
        TOLERANCE = 1.0e-9
        with np.errstate(all="ignore"):
            for x in (1.0e12, 1.0e14, 2.0e15):
                m = jv(0.5, x) ** 2 + yv(0.5, x) ** 2
                value = (2.0 / np.pi) / x / m
                self.assertAlmostEqual(
                    value,
                    1.0,
                    delta=TOLERANCE,
                    msg=f"(2/pi)/(x*(J^2+Y^2)) at x={x:.4g}, nu=0.5 is {value!r}, more than "
                    f"{TOLERANCE:.1g} from 1 -- jv/yv should still be accurate here "
                    "(RECONCILIATION.md C1)",
                )

    def test_jv_yv_rhs_identity_bad_above_boundary(self):
        """
        RECONCILIATION.md C1: above x ~ 2.5e15 the same exact identity departs from 1 by O(1) among
        adjacent doubles, because jv/yv have lost argument-reduction accuracy. Measured at
        x=5e15 * (1 + k*1e-15) for k in 0..4 on SciPy 1.15.2: values 1.676372, 1.653327, 1.077128,
        1.389636, 1.665364 -- none within 1e-2 of 1, among arguments that differ by a few parts in
        1e15.

        This assertion is deliberately an assertion that the library *is* broken: if a future SciPy
        release fixes Amos's argument reduction here, this test will fail. The correct response is
        not to delete or loosen it -- it is to raise
        LiouvilleGreen.tests.bessel_reference.SCIPY_REFERENCE_MAX_X and re-derive the supported
        domain, which is exactly the decision a failing test here should force someone to make.
        """
        x0 = 5.0e15
        offsets = (0.0, 1e-15, 2e-15, 3e-15, 1e-14)
        with np.errstate(all="ignore"):
            values = []
            for offset in offsets:
                x = x0 * (1.0 + offset)
                m = jv(0.5, x) ** 2 + yv(0.5, x) ** 2
                values.append((2.0 / np.pi) / x / m)

        worst = max(abs(v - 1.0) for v in values)
        self.assertGreater(
            worst,
            1.0e-2,
            f"(2/pi)/(x*(J^2+Y^2)) near x={x0:.1e} stayed within 1e-2 of 1 at every adjacent "
            f"double (values={values!r}) -- the Amos noise this test pins may have been fixed; "
            "if so, raise SCIPY_REFERENCE_MAX_X and re-derive the supported domain rather than "
            "deleting this test",
        )

    # --------------------------------------------------------------------------------------------
    # Fact 4 -- the failure is not monotonic in x, so no simple ceiling can be certified for the
    # jv/yv route (DRAFT-PLAN.md Sec4.4).
    # --------------------------------------------------------------------------------------------

    def test_jv_yv_failure_is_not_monotonic_in_x(self):
        """
        DRAFT-PLAN.md Sec4.4 claims jv/yv "recover at 1e16 for nu=1000.5 while failing at 1e10", so
        no simple ceiling can be certified for that route. The literal x=1e10/x=1e16 pair did not
        reproduce cleanly on SciPy 1.15.2 (both are far from the exact value there); see the
        deviation recorded in the campaign log. What does reproduce, cleanly and at round x values,
        is a nearby pair demonstrating the same qualitative claim -- badly wrong, then much better,
        at a *higher* x, so no monotone ceiling on jv/yv accuracy exists:

            nu=1000.5, x=1e10:  a = sqrt(pi*x/2)*hypot(J,Y) = 0.487462  (|a-1| = 0.513, badly wrong)
            nu=1000.5, x=3e10:  a = sqrt(pi*x/2)*hypot(J,Y) = 0.999070  (|a-1| = 0.00093, accurate)

        The campaign's claim this test needs is only the weaker one this demonstrates: a monotone
        ceiling on jv/yv accuracy must not be assumed. It does not claim any specific x is the last
        good or last bad point -- only that failing and (much) better points both exist, with the
        better one at larger x.
        """
        nu = 1000.5
        x_bad, x_better = 1.0e10, 3.0e10
        with np.errstate(all="ignore"):
            a_bad = np.hypot(jv(nu, x_bad), yv(nu, x_bad)) * np.sqrt(np.pi * x_bad / 2)
            a_better = np.hypot(jv(nu, x_better), yv(nu, x_better)) * np.sqrt(
                np.pi * x_better / 2
            )

        self.assertGreater(
            abs(a_bad - 1.0),
            0.1,
            f"nu={nu}, x={x_bad:.1e}: a={a_bad!r} is no longer badly wrong (|a-1| <= 0.1) -- the "
            "non-monotonicity this test pins may have changed",
        )
        self.assertLess(
            abs(a_better - 1.0),
            1.0e-2,
            f"nu={nu}, x={x_better:.1e}: a={a_better!r} is no longer accurate (|a-1| >= 1e-2) at "
            "this larger x -- the non-monotonicity this test pins may have changed",
        )


if __name__ == "__main__":
    unittest.main()
