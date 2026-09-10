"""
Tests of the measurement harness itself.

A harness nobody checks is worse than none: every acceptance threshold in the transfer-function
remedial campaign's later prompts is expressed against the metrics and references in
``bessel_reference.py``, so an error there would be inherited silently by all of them and would
look like a property of the construction under test.

These tests must stay cheap -- they run on every commit -- so they read the cached corner table
rather than calling ``mpmath``. :meth:`TestBesselReference.test_no_test_imports_mpmath` asserts
that, by re-running every other test in this module in a fresh interpreter and checking
``sys.modules`` afterwards.
"""

import decimal
import math
import subprocess
import sys
import unittest

import numpy as np

from LiouvilleGreen.tests import bessel_reference as br


class TestBesselReference(unittest.TestCase):

    # ------------------------------------------------------------------------------------------
    # 1. the three tiers agree where all three are valid
    # ------------------------------------------------------------------------------------------

    def test_tiers_agree(self):
        """
        ``exact_half_integer``, ``scipy_reference`` and the cached ``mpmath`` corners must agree to
        1e-14 relative on J, Y and A wherever all three are valid.

        This is the test that makes the tier hierarchy usable: it says that below the Amos
        boundaries the cheap tiers may be substituted for the expensive one, so a later prompt that
        sweeps thousands of points does not have to pay 70-digit costs to be believed. It is also
        the only place the *upward order recurrence* of ``exact_half_integer`` is checked against an
        independent implementation.
        """
        TOLERANCE = 1.0e-14

        worst = 0.0
        worst_where = None
        for nu in (0.5, 1.5, 2.5):
            for x in (10.0, 1.0e3, 1.0e7):
                corner = br.cached_corner(nu, x)

                J_exact, Y_exact = br.exact_half_integer(nu, x)
                J_scipy, Y_scipy = br.scipy_reference(nu, x)
                A_exact = br.reference_amplitude(J_exact, Y_exact)
                A_scipy = br.reference_amplitude(J_scipy, Y_scipy)

                for label, ours, cached in (
                    ("J exact", J_exact, corner.J),
                    ("Y exact", Y_exact, corner.Y),
                    ("A exact", A_exact, corner.amplitude),
                    ("J scipy", J_scipy, corner.J),
                    ("Y scipy", Y_scipy, corner.Y),
                    ("A scipy", A_scipy, corner.amplitude),
                ):
                    relerr = abs(ours / cached - 1.0)
                    if relerr > worst:
                        worst, worst_where = relerr, (label, nu, x)
                    self.assertLess(
                        relerr,
                        TOLERANCE,
                        f"{label} disagrees with the cached mpmath corner at nu={nu}, x={x:.6g}: "
                        f"ours={ours!r}, cached={cached!r}, relerr={relerr:.3g}",
                    )

        print(
            f"test_tiers_agree: worst relative disagreement {worst:.3g} at {worst_where} "
            f"(tolerance {TOLERANCE:.1g})"
        )

    # ------------------------------------------------------------------------------------------
    # 2. scipy_reference refuses where Amos is not a reference
    # ------------------------------------------------------------------------------------------

    def test_scipy_reference_refuses_large_x(self):
        """
        ``scipy_reference`` must raise above 2e15 rather than answer.

        ``RECONCILIATION.md`` C1: above ``x ~ 2.5e15`` Amos loses argument reduction and ``jv``/
        ``yv`` become O(1)-relatively noisy while remaining finite. A reference that answered there
        would reproduce the silent-failure mode the campaign exists to remove, and would do so
        inside the instrument used to measure the fix.
        """
        for x in (2.0e15 * (1.0 + 1e-12), 2.5e15, 3.0e15, 8.6e15, 1.0e16):
            with self.assertRaises(ValueError) as caught:
                br.scipy_reference(1.5, x)
            message = str(caught.exception)
            self.assertIn("RECONCILIATION.md", message)
            self.assertIn("mpmath", message)

        # arrays must be rejected on their largest element, not silently truncated
        with self.assertRaises(ValueError):
            br.scipy_reference(1.5, np.array([1.0e3, 1.0e16]))

        # and it must still answer at and below the boundary
        J, Y = br.scipy_reference(1.5, br.SCIPY_REFERENCE_MAX_X)
        self.assertTrue(math.isfinite(J) and math.isfinite(Y))

    def test_scipy_reference_refuses_high_order_above_collapsed_boundary(self):
        """
        Above ``nu = 85.5`` the Amos boundary collapses to 7.13e8, and ``scipy_reference`` must
        refuse there too.

        This is the harness's own version of the ``hankel1e`` failure of ``RECONCILIATION.md`` 1 and
        ``DRAFT-PLAN.md`` 4.4, measured for ``jv``/``yv``, which that work did not cover. The
        failure is not a non-finite value: at ``nu = 100.5, x = 1e12`` the normalized amplitude
        ``a = sqrt(pi x/2) hypot(J, Y)`` -- which is ``1 + O(nu^2/x^2)`` and must be 1 to twelve
        places -- comes back as 0.611. ``isfinite`` would pass it.
        """
        self.assertEqual(br.scipy_reference_max_x(1.5), br.SCIPY_REFERENCE_MAX_X)
        self.assertEqual(br.scipy_reference_max_x(20.5), br.SCIPY_REFERENCE_MAX_X)
        self.assertEqual(br.scipy_reference_max_x(85.5), br.SCIPY_REFERENCE_MAX_X)
        self.assertEqual(
            br.scipy_reference_max_x(100.5), br.SCIPY_REFERENCE_HIGH_ORDER_MAX_X
        )

        from scipy.special import jv, yv

        for nu in (100.5, 1000.5):
            with self.assertRaises(ValueError) as caught:
                br.scipy_reference(nu, 1.0e12)
            self.assertIn("RECONCILIATION.md", str(caught.exception))

            # The point of the guard: what it is refusing to return is finite and wrong. Compared
            # against the cached mpmath corner rather than a live mpmath call, so that this module
            # never imports mpmath -- test_no_test_imports_mpmath asserts that.
            corner = br.cached_corner(nu, 1.0e12)
            self.assertLess(abs(corner.a - 1.0), 1.0e-12)

            raw = br.reference_a(1.0e12, jv(nu, 1.0e12), yv(nu, 1.0e12))
            self.assertTrue(
                math.isfinite(raw), "the failure mode is finite, not inf or nan"
            )
            self.assertGreater(
                abs(raw - 1.0),
                0.1,
                f"jv/yv at nu={nu}, x=1e12 gave a={raw!r}; the guard would be pointless if this "
                f"were close to 1",
            )

        # below the collapsed boundary the tier is still good -- degraded, but by eight orders of
        # magnitude less than above it, which is what makes the boundary a boundary rather than a
        # taste
        for nu in (100.5, 1000.5):
            x = 0.9 * br.SCIPY_REFERENCE_HIGH_ORDER_MAX_X
            J, Y = br.scipy_reference(nu, x)
            self.assertLess(abs(br.reference_a(x, J, Y) - 1.0), 1.0e-6)

    # ------------------------------------------------------------------------------------------
    # 3. the derivative oracle satisfies the Wronskian at every cached corner
    # ------------------------------------------------------------------------------------------

    def test_wronskian_at_every_corner(self):
        """
        ``a^2 theta' = 1`` at every cached corner, to 1e-13.

        This is the Wronskian ``A^2 theta' = 2/(pi x)`` written in normalized variables, and it is
        an identity: it holds for the *exact* functions at every ``x``, so any departure is a defect
        in the cached table or in the derived-quantity definitions, never a property of the Bessel
        functions. Note that ``README`` 2 (f) warns this check becomes a tautology once the
        construction reads ``theta'`` off its own amplitude interpolant -- here it is not a
        tautology, because ``a`` and ``theta'`` were computed from ``J`` and ``Y`` by different
        routes (``sqrt(pi x/2) hypot`` and ``(2/pi)/(x(J^2+Y^2))``) and written to the file
        independently.
        """
        TOLERANCE = 1.0e-13

        corners = br.cached_corners()
        self.assertGreater(len(corners), 60)

        worst = 0.0
        worst_where = None
        for corner in corners:
            residual = abs(corner.a * corner.a * corner.theta_deriv - 1.0)
            if residual > worst:
                worst, worst_where = residual, (corner.nu, corner.x)
            self.assertLess(
                residual,
                TOLERANCE,
                f"Wronskian a^2 theta' = 1 fails at nu={corner.nu}, x={corner.x:.6g}: "
                f"a={corner.a!r}, theta_deriv={corner.theta_deriv!r}, "
                f"|a^2 theta' - 1|={residual:.3g}",
            )

        print(
            f"test_wronskian_at_every_corner: worst |a^2 theta' - 1| = {worst:.3g} at "
            f"(nu, x)={worst_where} over {len(corners)} corners (tolerance {TOLERANCE:.1g})"
        )

    def test_corner_self_consistency(self):
        """
        Each corner's stored quantities agree with each other: ``amplitude = hypot(J, Y)``,
        ``a = sqrt(pi x/2) A``, ``theta_principal = atan2(J, -Y)``, and the stored branch closes,
        ``theta_principal + 2 pi theta_div_2pi = x + c_nu + r``.

        The branch closure is the one that matters. ``theta_div_2pi`` reaches 1.59e15, so the
        closure can only be checked to the ``eps x`` resolution of the reconstructed phase -- about
        2 rad at ``x = 1e16`` -- which is itself the reason the campaign keeps the split
        representation and never forms ``x + d``. So the test scales its tolerance with ``eps x``
        and reports the achieved figure.
        """
        for corner in br.cached_corners():
            self.assertAlmostEqual(
                corner.amplitude,
                math.hypot(corner.J, corner.Y),
                delta=4.0 * np.finfo(float).eps * corner.amplitude,
                msg=f"amplitude != hypot(J, Y) at nu={corner.nu}, x={corner.x:.6g}",
            )
            self.assertAlmostEqual(
                corner.a,
                math.sqrt(0.5 * math.pi * corner.x) * corner.amplitude,
                delta=4.0 * np.finfo(float).eps * corner.a,
                msg=f"a != sqrt(pi x/2) A at nu={corner.nu}, x={corner.x:.6g}",
            )
            self.assertAlmostEqual(
                corner.theta_principal,
                math.atan2(corner.J, -corner.Y),
                delta=1.0e-13,
                msg=f"theta_principal != atan2(J, -Y) at nu={corner.nu}, x={corner.x:.6g}",
            )

            closure = abs(
                corner.theta_continuous - (corner.x + br.c_nu(corner.nu) + corner.r)
            )
            budget = 8.0 * np.finfo(float).eps * max(corner.x, 1.0) + 1.0e-12
            self.assertLess(
                closure,
                budget,
                f"stored branch does not close at nu={corner.nu}, x={corner.x:.6g}: "
                f"theta_principal + 2 pi n differs from x + c_nu + r by {closure:.3g} "
                f"(budget {budget:.3g})",
            )

    # ------------------------------------------------------------------------------------------
    # 4. the cached residuals reproduce the tail series -- the branch check, asserted
    # ------------------------------------------------------------------------------------------

    def test_cached_residual_matches_tail_series(self):
        """
        At the largest ``x`` of each order the cached residual must reproduce the two-term series
        ``r ~ (mu-1)/(8x) + (mu-1)(mu-25)/(384 x^3)`` to 1e-8.

        This asserts, rather than assumes, the branch check of the generator. ``r`` is **not**
        ``theta - x - c_nu`` folded into ``(-pi, pi]``: at ``nu = 1000.5`` it is 570.82 rad at the
        turning point, about 91 cycles from zero (``RECONCILIATION.md`` C2), so a naive fold gives a
        residual wrong by an exact multiple of ``2 pi``. Such an error is invisible in ``sin theta``
        and ``cos theta`` and would therefore survive every value-level check in the campaign while
        corrupting every residual-level one.

        As a second, independent guard the test also checks the residual against the series at
        *every* corner with ``x >= 1000 nu``, where the first omitted term is far below 1e-8.
        """
        TOLERANCE = 1.0e-8

        orders = br.cached_orders()
        self.assertEqual(len(orders), len(br.CACHED_ORDERS))

        for nu in orders:
            corners = br.cached_corners(nu)
            largest = corners[-1]
            series = br.tail_residual_series(nu, largest.x)

            # compare the stored 40-digit string, not its rounding to double: at nu=1000.5,
            # r(1e16) = 5.005e-11, so a double-level comparison could not resolve a discrepancy
            # below ~1e-27 and would report a spurious exact zero
            with decimal.localcontext() as context:
                context.prec = 50
                mismatch = abs(
                    decimal.Decimal(largest.strings["r"])
                    - decimal.Decimal(repr(series))
                )
            self.assertLess(
                mismatch,
                decimal.Decimal(TOLERANCE),
                f"cached residual does not match the tail series at nu={nu}, x={largest.x:.6g}: "
                f"r={largest.strings['r']}, series={series!r}, mismatch={mismatch:.3g}. The "
                f"residual branch is wrong; regenerate the table.",
            )
            print(
                f"test_cached_residual_matches_tail_series: nu={nu} at x={largest.x:.6g}: "
                f"|r - series| = {float(mismatch):.3g}"
            )

            for corner in corners:
                if corner.x < 1000.0 * nu:
                    continue
                series = br.tail_residual_series(nu, corner.x)
                self.assertLess(
                    abs(corner.r - series),
                    TOLERANCE,
                    f"cached residual does not match the tail series at nu={nu}, "
                    f"x={corner.x:.6g}: r={corner.r!r}, series={series!r}",
                )

    def test_cached_residual_reproduces_reconciliation_c2(self):
        """
        The cached residual at the turning point reproduces ``RECONCILIATION.md`` C2's independently
        measured ``r(x_0)``, and the cached amplitude reproduces its 3.1 plausibility band.

        C2 and 3.1 were measured from continuously tracked ``hankel1e`` with no series anchor and no
        ``mpmath``; this table was built from ``mpmath`` with a series anchor. Agreement is
        therefore a genuine cross-check of the branch, not a restatement -- and the ``a`` figures
        are the constants prompt 04's two-sided plausibility band is calibrated on.
        """
        expected_r0 = {
            1.5: 0.6155,
            2.5: 1.1832,
            20.5: 11.4437,
            100.5: 57.1042,
            1000.5: 570.8200,
        }
        expected_a0 = {2.5: 1.32288, 20.5: 1.85684, 100.5: 2.41795, 1000.5: 3.54597}

        for nu, expected in expected_r0.items():
            corner = br.cached_corners(nu)[0]
            self.assertAlmostEqual(
                corner.x,
                br.construction_min_x(nu),
                delta=1e-12,
                msg=f"first cached corner for nu={nu} is not the turning point",
            )
            self.assertAlmostEqual(
                corner.r,
                expected,
                delta=5e-4,
                msg=f"r(x_0) at nu={nu} is {corner.r!r}, RECONCILIATION.md C2 measured {expected}",
            )

        for nu, expected in expected_a0.items():
            corner = br.cached_corners(nu)[0]
            self.assertAlmostEqual(
                corner.a,
                expected,
                delta=5e-5,
                msg=f"a(x_0) at nu={nu} is {corner.a!r}, RECONCILIATION.md 3.1 measured {expected}",
            )

    # ------------------------------------------------------------------------------------------
    # 5. the metrics themselves
    # ------------------------------------------------------------------------------------------

    def test_phase_pair_error_on_perfect_and_perturbed_input(self):
        """
        ``phase_pair_error`` returns exactly 0 for a perfect reconstruction, and returns both the
        size and the *index* of the maximum for a deliberately perturbed one.

        The index is load-bearing: ``DRAFT-PLAN.md`` 4.7 finds every derivative maximum in the
        interval adjacent to the turning point, and prompts 04 and 05 have to be able to confirm
        that on their own output rather than take it on trust.
        """
        corners = br.cached_corners(2.5)
        x = np.array([corner.x for corner in corners])
        J = np.array([corner.J for corner in corners])
        Y = np.array([corner.Y for corner in corners])
        A = np.array([corner.amplitude for corner in corners])

        error, index = br.phase_pair_error(J / A, Y / A, J, Y, A)
        self.assertEqual(error, 0.0)
        self.assertEqual(index, 0)

        # A phase perturbation of delta at one point shows up at that index, at size
        # delta * max(|cos theta|, |sin theta|) -- the two branches differentiate to
        # delta*cos(theta) and delta*sin(theta), and the metric keeps the larger. So the metric
        # reads a phase error faithfully up to a factor between 1/sqrt(2) and 1, never larger and
        # never vanishing: that bounded, x-independent relationship is the reason it is used in
        # place of a pointwise relative error in J or Y, which is unbounded at every Bessel zero.
        DELTA = 3.0e-7
        TARGET = 4
        theta = np.arctan2(J, -Y)
        shifted = theta.copy()
        shifted[TARGET] += DELTA
        error, index = br.phase_pair_error(np.sin(shifted), -np.cos(shifted), J, Y, A)
        self.assertEqual(index, TARGET)
        expected = DELTA * max(
            abs(math.cos(theta[TARGET])), abs(math.sin(theta[TARGET]))
        )
        self.assertAlmostEqual(error / expected, 1.0, delta=1e-5)
        self.assertGreater(error, DELTA / math.sqrt(2.0) * (1.0 - 1e-5))
        self.assertLessEqual(error, DELTA * (1.0 + 1e-5))

        # scalars are accepted and report index 0
        error, index = br.phase_pair_error(J[0] / A[0], Y[0] / A[0], J[0], Y[0], A[0])
        self.assertEqual((error, index), (0.0, 0))

        # a pure sign error in the Y branch must be caught -- the convention is Y = -A cos(theta),
        # and the metric compares both branches precisely so a convention slip cannot hide
        error, _ = br.phase_pair_error(np.sin(theta), np.cos(theta), J, Y, A)
        self.assertGreater(error, 0.1)

    def test_amplitude_and_derivative_errors(self):
        """``amplitude_error`` and ``derivative_error`` are relative, and report their argmax."""
        corners = br.cached_corners(20.5)
        A = np.array([corner.amplitude for corner in corners])
        theta_prime = np.array([corner.theta_deriv for corner in corners])

        error, index = br.amplitude_error(A, A)
        self.assertEqual((error, index), (0.0, 0))
        error, index = br.derivative_error(theta_prime, theta_prime)
        self.assertEqual((error, index), (0.0, 0))

        TARGET = 3
        RELATIVE = 2.5e-9
        perturbed = A.copy()
        perturbed[TARGET] *= 1.0 + RELATIVE
        error, index = br.amplitude_error(perturbed, A)
        self.assertEqual(index, TARGET)
        self.assertAlmostEqual(error / RELATIVE, 1.0, delta=1e-6)

        perturbed = theta_prime.copy()
        perturbed[TARGET] *= 1.0 - RELATIVE
        error, index = br.derivative_error(perturbed, theta_prime)
        self.assertEqual(index, TARGET)
        self.assertAlmostEqual(error / RELATIVE, 1.0, delta=1e-6)

    def test_derived_quantities_against_the_exact_tier(self):
        """
        The derived-quantity helpers agree with closed forms at ``nu = 1/2``, where the whole
        construction is exact: ``a == 1``, ``r == 0``, ``theta == x`` and ``theta' == 1`` at every
        ``x``, because ``mu - 1 = 0`` (``README`` 2 (b)).
        """
        x = np.array([1.0e-5, 0.5, 7.0, 1.0e3, 1.0e7, 1.0e12])
        J, Y = br.exact_half_integer(0.5, x)

        self.assertEqual(br.c_nu(0.5), 0.0)
        np.testing.assert_allclose(br.reference_a(x, J, Y), 1.0, rtol=4e-16, atol=0.0)
        np.testing.assert_allclose(
            br.reference_theta_deriv(x, J, Y), 1.0, rtol=4e-16, atol=0.0
        )
        np.testing.assert_allclose(br.tail_residual_series(0.5, x), 0.0, atol=0.0)

        # theta = x for nu = 1/2, so atan2(J, -Y) is x folded into (-pi, pi]
        principal = br.reference_theta_principal(J, Y)
        folded = np.arctan2(np.sin(x), np.cos(x))
        np.testing.assert_allclose(principal, folded, rtol=0.0, atol=1e-15)

    def test_reference_tier_selector(self):
        """
        The selector is explicit: unknown tiers raise, and no tier is chosen by default.

        The ``mpmath`` tier is deliberately **not** called here, so that the module stays free of
        ``mpmath`` (:meth:`test_no_test_imports_mpmath`). It is not untested for that: it is the
        tier that produced ``bessel_reference_data.json``, which eight of these twelve tests score
        against, and ``regenerate_reference_table`` asserts its own output against the tail series
        before writing.
        """
        for tier in (br.TIER_EXACT, br.TIER_SCIPY):
            J, Y = br.reference_JY(1.5, 100.0, tier)
            self.assertTrue(math.isfinite(J) and math.isfinite(Y))
        self.assertEqual(
            br.REFERENCE_TIERS, (br.TIER_EXACT, br.TIER_SCIPY, br.TIER_MPMATH)
        )

        with self.assertRaises(ValueError):
            br.reference_JY(1.5, 100.0, "bessel_phase")

        with self.assertRaises(ValueError):
            # 7/4 has no closed form; the exact tier must say so rather than approximate
            br.exact_half_integer(1.75, 100.0)

        self.assertEqual(br.best_available_tier(1.5, 1.0e7), br.TIER_EXACT)
        self.assertEqual(br.best_available_tier(1.75, 1.0e7), br.TIER_SCIPY)
        self.assertEqual(br.best_available_tier(1.75, 1.0e16), br.TIER_MPMATH)
        self.assertEqual(br.best_available_tier(100.5, 1.0e10), br.TIER_MPMATH)

        bundle = br.reference_bundle(1.5, np.array([10.0, 1.0e3]), br.TIER_EXACT)
        self.assertEqual(bundle.tier, br.TIER_EXACT)
        np.testing.assert_allclose(
            bundle.amplitude, np.hypot(bundle.J, bundle.Y), rtol=0.0, atol=0.0
        )

    # ------------------------------------------------------------------------------------------
    # the cost contract
    # ------------------------------------------------------------------------------------------

    def test_no_test_imports_mpmath(self):
        """
        **No test in this module may import ``mpmath``**, and the cached table is how that is
        achieved.

        Prompt 01 §5 makes this an acceptance criterion, and §2.3 explains why: ``mpmath`` at 70
        digits is far too slow to run on every commit, so the corners are cached and regenerating
        them is a separately invoked function. The campaign's rules (``README`` 5 item 7) do permit
        ``mpmath`` at test time, but a stray import here would cost every later prompt's iteration
        loop for no benefit, and it would fail nothing -- which is exactly why it needs an
        assertion rather than a convention.

        Checked by re-running this module's *other* tests in a fresh interpreter and looking at
        ``sys.modules`` afterwards. That is stronger than checking the import alone, which would
        say nothing about what the test bodies do; and it has to be a subprocess, because by the
        time this test runs the parent may well have imported ``mpmath`` for unrelated reasons.
        """
        script = (
            "import sys, unittest\n"
            "\n"
            "def flatten(suite):\n"
            "    for item in suite:\n"
            "        if isinstance(item, unittest.TestSuite):\n"
            "            yield from flatten(item)\n"
            "        else:\n"
            "            yield item\n"
            "\n"
            "loaded = unittest.TestLoader().loadTestsFromName(\n"
            "    'LiouvilleGreen.tests.test_bessel_reference'\n"
            ")\n"
            "suite = unittest.TestSuite(\n"
            "    test for test in flatten(loaded)\n"
            "    if not test.id().endswith('test_no_test_imports_mpmath')\n"
            ")\n"
            "result = unittest.TextTestRunner(stream=sys.stderr, verbosity=0).run(suite)\n"
            "print('GUARD', result.wasSuccessful(), 'mpmath' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
        )

        # the child's tests print their own measured figures to stdout, so pick the sentinel line
        verdicts = [
            line for line in result.stdout.splitlines() if line.startswith("GUARD ")
        ]
        self.assertEqual(len(verdicts), 1, f"child produced {verdicts!r}")
        self.assertEqual(
            verdicts[0],
            "GUARD True False",
            f"expected 'GUARD True False' (all other tests passed, mpmath never imported); got "
            f"{verdicts[0]!r}. stderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
