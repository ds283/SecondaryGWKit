"""
Tests for ``LiouvilleGreen.bessel_tail``, the closed-form tail of the two-region Bessel
amplitude/phase construction (transfer-function remedial campaign, prompt 03).

Scoring
-------

Every number here is scored against ``LiouvilleGreen.tests.bessel_reference`` (campaign prompt
01), never against ``bessel_phase`` and never against the series itself. The reference tier is
chosen by :func:`_reference_tier` per order, from the measured accuracy of each tier rather than
by preference:

* ``exact`` where a closed form exists (``nu`` in ``EXACT_HALF_INTEGER_ORDERS``);
* ``scipy`` for the remaining orders up to 20.5, where ``jv``/``yv`` agree with 50-digit
  ``mpmath`` to ``|delta r| <= 7.6e-15`` and ``|delta a/a| <= 9.0e-15`` over the range used here;
* ``mpmath`` for ``nu >= 100.5``, where they do **not**: measured, ``jv``/``yv`` give
  ``|delta r|`` up to 1.2e-12 at ``nu = 100.5`` and 8.4e-12 at ``nu = 1000.5``, and
  ``|delta a/a|`` up to 1.7e-11 -- above this module's three-term series error, so scoring the
  series against them would be measuring Amos.

``mpmath`` is used at 50 digits, and only at points far from the turning point (``x >= 4.4 nu``),
where it is fast: the whole module runs in well under a second.

The residual branch
-------------------

``r`` is recovered from a reference ``(J, Y)`` pair as

    r = atan2(J, -Y) + 2 pi n - (x + c_nu),

and the subtraction is done with ``(x + c_nu)`` reduced mod ``2 pi`` **exactly**, at 50 digits,
rather than in double precision. That matters: a double-precision reduction of ``x + c_nu`` errs
by roughly ``4e-17 x``, which is 8e-12 at ``x = 2e5`` and would swamp the three-term series error
this module has to resolve. The integer ``n`` is then fixed by lifting to the branch nearest the
series prediction, which is unambiguous because the series is accurate to far better than ``pi``
everywhere it is used here -- and is *never* a fold into ``(-pi, pi]``, which at ``nu = 1000.5``
would be wrong by an exact multiple of ``2 pi`` (``RECONCILIATION.md`` C2, board standing note 9).
:meth:`TestBesselTail.test_reference_helper_agrees_with_cached_corners` checks the helper against
the cached ``mpmath`` corners of prompt 01, which carry an independently resolved branch.
"""

import math
import os
import subprocess
import sys
import unittest

import numpy as np

from LiouvilleGreen import bessel_tail as bt
from LiouvilleGreen.tests import bessel_reference as br

#: Orders covered. ``1/2`` is the exactly degenerate case; ``3/2``, ``7/4`` and ``5/2`` are the
#: orders production builds; ``20.5``, ``100.5`` and ``1000.5`` are the campaign's high orders.
ORDERS = (0.5, 1.5, 1.75, 2.5, 20.5, 100.5, 1000.5)

#: The two accuracy budgets of ``README.md`` §6: the low-order row and the high-order row.
BUDGETS = (1e-11, 1e-6)

#: Absolute floor below which a measured ``|delta r|`` is the double-precision reference, not the
#: series. ``atan2(J, -Y)`` is O(1), so its own error is a few ulp of 1.
REFERENCE_R_FLOOR = 1e-15

#: Bounds on ``|delta r|``, keyed ``(nu, x/nu, n_terms)``. Measured values are in campaign log 03;
#: these are set about 3x above them, which is loose enough to survive an ulp-level change of
#: reference tier and tight enough that a wrong coefficient could not pass.
RESIDUAL_BOUNDS = {
    (2.5, 50.0, 2): 6.0e-10,
    (2.5, 50.0, 3): 8.0e-14,
    (2.5, 100.0, 2): 2.0e-11,
    (2.5, 100.0, 3): 1.0e-15,
    (2.5, 200.0, 2): 6.0e-13,
    (2.5, 200.0, 3): 1.0e-15,
    (20.5, 50.0, 2): 2.4e-09,
    (20.5, 50.0, 3): 4.0e-13,
    (20.5, 100.0, 2): 8.0e-11,
    (20.5, 100.0, 3): 3.0e-15,
    (20.5, 200.0, 2): 2.4e-12,
    (20.5, 200.0, 3): 1.0e-15,
    (100.5, 50.0, 2): 1.3e-08,
    (100.5, 50.0, 3): 2.2e-12,
    (100.5, 100.0, 2): 4.0e-10,
    (100.5, 100.0, 3): 2.0e-14,
    (100.5, 200.0, 2): 1.3e-11,
    (100.5, 200.0, 3): 1.0e-15,
    (1000.5, 50.0, 2): 1.3e-07,
    (1000.5, 50.0, 3): 2.2e-11,
    (1000.5, 100.0, 2): 4.0e-09,
    (1000.5, 100.0, 3): 2.0e-13,
    (1000.5, 200.0, 2): 1.3e-10,
    (1000.5, 200.0, 3): 1.0e-15,
}

#: Bounds on ``|delta a / a|`` at the same points. The amplitude condition is weaker than the
#: phase one by a factor ``~2x``, which is what these show: at ``50 nu`` with two terms the
#: measured values are 3.5e-12 (nu=5/2), 1.9e-12 (20.5), 2.0e-12 (100.5), 2.0e-12 (1000.5),
#: two to four orders below the corresponding ``|delta r|``.
AMPLITUDE_BOUNDS = {
    (2.5, 50.0, 2): 1.1e-11,
    (2.5, 100.0, 2): 1.7e-13,
    (20.5, 50.0, 2): 6.0e-12,
    (20.5, 100.0, 2): 9.0e-14,
    (100.5, 50.0, 2): 6.0e-12,
    (100.5, 100.0, 2): 1.0e-13,
    (1000.5, 50.0, 2): 6.0e-12,
    (1000.5, 100.0, 2): 1.0e-13,
}

#: Bound on ``|delta a / a|`` with three terms anywhere in the table above: at that point the
#: amplitude is at the double-precision floor for every order tested.
AMPLITUDE_THREE_TERM_BOUND = 5.0e-15


def _reference_tier(nu: float) -> str:
    """Which prompt-01 reference tier to score ``nu`` against; see the module docstring."""
    if any(abs(nu - order) < 1e-13 for order in br.EXACT_HALF_INTEGER_ORDERS):
        return br.TIER_EXACT
    if nu <= 20.5:
        return br.TIER_SCIPY
    return br.TIER_MPMATH


def _reduce_mod_two_pi(value: float, nu: float) -> "object":
    """``(x + c_nu) mod 2 pi``, computed at 50 decimal digits. Returns an ``mpf``."""
    from mpmath import mp, mpf, floor as mfloor, pi as mpi

    mp.dps = 50
    total = mpf(float(value)) + (mpi / 4 - mpi * mpf(float(nu)) / 2)
    return total - 2 * mpi * mfloor(total / (2 * mpi))


def reference_residual(nu: float, x: float) -> float:
    """
    The reference phase residual ``r_nu(x)`` on its **correct branch**.

    Built from a prompt-01 reference ``(J, Y)`` pair and an exact mod-``2 pi`` reduction of
    ``x + c_nu``; the branch integer is fixed by proximity to the series prediction. See the
    module docstring for why neither step may be done in double precision or by folding.
    """
    from mpmath import mpf, floor as mfloor, pi as mpi

    J, Y = br.reference_JY(nu, x, _reference_tier(nu))
    principal = mpf(float(br.reference_theta_principal(J, Y)))

    residual_principal = principal - _reduce_mod_two_pi(x, nu)
    predicted = mpf(float(bt.tail_residual(nu, x)))
    branch = mfloor((predicted - residual_principal) / (2 * mpi) + mpf("0.5"))

    return float(residual_principal + 2 * mpi * branch)


def reference_a_and_theta_deriv(nu: float, x: float):
    """``(a_nu, theta')`` from the same reference pair, as doubles."""
    J, Y = br.reference_JY(nu, x, _reference_tier(nu))
    return br.reference_a(x, J, Y), br.reference_theta_deriv(x, J, Y)


class TestBesselTail(unittest.TestCase):

    # ------------------------------------------------------------------------------------------
    # 1. nu = 1/2 is exact, not merely accurate
    # ------------------------------------------------------------------------------------------

    def test_half_order_is_identically_exact(self):
        """
        ``mu - 1 = 0`` at ``nu = 1/2`` (and ``4 * 0.5 * 0.5 == 1.0`` exactly in binary), so every
        coefficient vanishes and ``r == 0``, ``r' == 0``, ``a == 1`` at *every* ``x``, not
        asymptotically.

        Asserted as exact equality rather than to a tolerance. This is the one place in the
        campaign where that is the right assertion: any nonzero value here means an algebra slip,
        not a rounding effect.
        """
        for x in (1e-3, 1.0, 10.0, 1e3, 1e7, 1e12, 1e15):
            for n_terms in (1, 2, 3):
                self.assertEqual(
                    bt.tail_residual(0.5, x, n_terms=n_terms),
                    0.0,
                    f"r(1/2, {x:.6g}) with {n_terms} terms is not exactly zero",
                )
                self.assertEqual(
                    abs(bt.tail_residual_deriv(0.5, x, n_terms=n_terms)),
                    0.0,
                    f"r'(1/2, {x:.6g}) with {n_terms} terms is not exactly zero",
                )
                self.assertEqual(
                    abs(bt.tail_residual_log_deriv(0.5, x, n_terms=n_terms)),
                    0.0,
                    f"dr/dlog x (1/2, {x:.6g}) with {n_terms} terms is not exactly zero",
                )
                self.assertEqual(
                    bt.tail_amplitude(0.5, x, n_terms=n_terms),
                    1.0,
                    f"a(1/2, {x:.6g}) with {n_terms} terms is not exactly one",
                )
                self.assertEqual(
                    bt.tail_first_omitted_term(0.5, x, n_terms=n_terms),
                    0.0,
                    f"first omitted term at nu=1/2, x={x:.6g} is not exactly zero",
                )

        self.assertEqual(bt.tail_series_coefficients(0.5), (0.0, 0.0, 0.0, 0.0))

    def test_half_order_crossover_is_the_domain_lower_bound(self):
        """
        The series is exact for ``nu = 1/2`` at every ``x``, so the whole domain is closed form
        and the near-region sampler must never run. ``tail_crossover`` therefore has to return the
        bottom of the domain, not a large "safe" value.
        """
        for budget in BUDGETS:
            crossover = bt.tail_crossover(0.5, budget, budget)
            self.assertEqual(crossover.x_star, bt.construction_min_x(0.5))
            self.assertEqual(crossover.first_omitted_at_x_star, 0.0)
            self.assertEqual(crossover.binding, "domain")

    # ------------------------------------------------------------------------------------------
    # 2. Series accuracy against the independent references
    # ------------------------------------------------------------------------------------------

    def test_reference_helper_agrees_with_cached_corners(self):
        """
        Validate this module's own reference machinery before trusting anything it scores.

        ``100 nu`` is a cached corner for every order in :data:`ORDERS`, and the cached rows carry
        a residual whose branch was resolved independently by prompt 01's generator (a walked
        ``mpmath`` phase, asserted against the two-term series). If the reduction or the branch
        lift here were wrong, this is where it shows up -- an error of an exact multiple of
        ``2 pi``, not a small one.
        """
        for nu in ORDERS:
            x = 100.0 * nu
            row = br.cached_corner(nu, x)

            r = reference_residual(nu, x)
            a, theta_deriv = reference_a_and_theta_deriv(nu, x)

            self.assertLess(
                abs(r - row.r),
                1e-11,
                f"reference_residual disagrees with the cached corner at nu={nu}, x={x:.6g}: "
                f"{r!r} vs {row.r!r}",
            )
            self.assertLess(abs(a / row.a - 1.0), 1e-13)
            self.assertLess(abs(theta_deriv / row.theta_deriv - 1.0), 1e-13)

    def test_series_accuracy_against_references(self):
        """
        ``|delta r|`` and ``|delta a/a|`` at ``50 nu``, ``100 nu`` and ``200 nu``, with two and
        three terms, for the four orders ``RECONCILIATION.md`` §1 measured.

        The two-term column reproduces that document: 1.77e-10 (nu=5/2 at 50 nu), 7.64e-10 (20.5),
        4.01e-9 (100.5), and ``|delta a/a|`` 3.5e-12, 1.9e-12, 2.0e-12 there falling below
        5.5e-14 at ``100 nu``.
        """
        for nu in (2.5, 20.5, 100.5, 1000.5):
            for multiple in (50.0, 100.0, 200.0):
                x = multiple * nu
                r_ref = reference_residual(nu, x)
                a_ref, _ = reference_a_and_theta_deriv(nu, x)

                for n_terms in (2, 3):
                    residual_error = abs(
                        bt.tail_residual(nu, x, n_terms=n_terms) - r_ref
                    )
                    self.assertLess(
                        residual_error,
                        RESIDUAL_BOUNDS[(nu, multiple, n_terms)],
                        f"|delta r| at nu={nu}, x={x:.6g} with {n_terms} terms is "
                        f"{residual_error:.3e}",
                    )

                    amplitude_error = abs(
                        bt.tail_amplitude(nu, x, n_terms=n_terms) / a_ref - 1.0
                    )
                    bound = (
                        AMPLITUDE_BOUNDS.get((nu, multiple, n_terms))
                        or AMPLITUDE_THREE_TERM_BOUND
                    )
                    self.assertLess(
                        amplitude_error,
                        bound,
                        f"|delta a/a| at nu={nu}, x={x:.6g} with {n_terms} terms is "
                        f"{amplitude_error:.3e}",
                    )

    def test_first_omitted_term_estimates_the_true_error(self):
        """
        The remainder test of :func:`bessel_tail.tail_crossover` is only meaningful if the first
        omitted term actually tracks the truth. Measured, the ratio is 1.000 wherever the truth is
        above the reference floor, so require a factor of two plus that floor.

        This is the assertion that would fail first if a series coefficient were wrong: a wrong
        ``c2`` leaves a three-term error the ``c3`` estimator does not predict.
        """
        for nu in (2.5, 20.5, 100.5, 1000.5):
            for multiple in (50.0, 100.0, 200.0):
                x = multiple * nu
                r_ref = reference_residual(nu, x)

                for n_terms in (1, 2, 3):
                    error = abs(bt.tail_residual(nu, x, n_terms=n_terms) - r_ref)
                    estimate = bt.tail_first_omitted_term(nu, x, n_terms=n_terms)
                    if estimate == 0.0:
                        # nu = 5/2 with one term: mu - 25 = 0 exactly, so the *second* term
                        # vanishes identically while the series does not.
                        # test_estimator_vanishes_at_an_isolated_order covers that case; here it
                        # would just assert a false claim about the estimator.
                        self.assertEqual((nu, n_terms), (2.5, 1))
                        continue
                    self.assertLess(
                        error,
                        2.0 * estimate + REFERENCE_R_FLOOR,
                        f"the first omitted term underestimates the error at nu={nu}, "
                        f"x={x:.6g} with {n_terms} terms: error {error:.3e} vs estimate "
                        f"{estimate:.3e}",
                    )

    # ------------------------------------------------------------------------------------------
    # 3. The Wronskian, by construction and against the reference
    # ------------------------------------------------------------------------------------------

    def test_wronskian_holds_by_construction(self):
        """
        ``a = (1 + r')^(-1/2)`` makes ``a^2 (1 + r') = 1`` a tautology. Asserted anyway, as a
        guard against an algebra slip in the term-by-term differentiation: if
        :func:`bessel_tail.tail_amplitude` were ever rebuilt from DLMF 10.18.17 -- which this
        design deliberately does not use -- this would stop being exact.
        """
        for nu in ORDERS:
            for multiple in (50.0, 100.0, 1000.0):
                x = multiple * max(nu, 1.0)
                for n_terms in (1, 2, 3):
                    a = bt.tail_amplitude(nu, x, n_terms=n_terms)
                    deriv = bt.tail_residual_deriv(nu, x, n_terms=n_terms)
                    self.assertLess(
                        abs(a * a * (1.0 + deriv) - 1.0),
                        4e-16,
                        f"a^2 (1 + r') != 1 at nu={nu}, x={x:.6g} with {n_terms} terms",
                    )

    def test_theta_deriv_against_the_independent_oracle(self):
        """
        ``theta' = a^-2`` against ``(2/pi)/(x (J^2 + Y^2))`` from the reference functions, which
        is an identity rather than an approximation, at ``x >= 100 nu``. Required to ``1e-11``;
        measured worst case 1.11e-15.

        The Wronskian check above is a tautology, so this is the *independent* half of it. It is
        also the quantity the campaign's standing regression gate
        (``test_bessel_phase.test_phase_derivative``, 1e-6 relative) contracts on.
        """
        worst = 0.0
        worst_at = None
        for nu in ORDERS:
            for multiple in (100.0, 200.0, 1000.0):
                x = multiple * nu
                _, theta_deriv_ref = reference_a_and_theta_deriv(nu, x)
                ours = bt.tail_amplitude(nu, x) ** -2
                error = abs(ours / theta_deriv_ref - 1.0)
                if error > worst:
                    worst, worst_at = error, (nu, x)

        self.assertLess(
            worst, 1e-11, f"theta' relative error {worst:.3e} at (nu, x)={worst_at}"
        )

    def test_log_derivative_is_x_times_the_derivative(self):
        """``dr/d(log x) = x r'`` -- the near region interpolates in ``u = log x``."""
        for nu in ORDERS:
            for x in (3.0 * max(nu, 1.0), 100.0 * nu, 1e6):
                for n_terms in (1, 2, 3):
                    log_deriv = bt.tail_residual_log_deriv(nu, x, n_terms=n_terms)
                    deriv = bt.tail_residual_deriv(nu, x, n_terms=n_terms)
                    if deriv == 0.0:
                        self.assertEqual(log_deriv, 0.0)
                    else:
                        self.assertLess(abs(log_deriv / (x * deriv) - 1.0), 4e-16)

    # ------------------------------------------------------------------------------------------
    # 4. The crossover is a real remainder test
    # ------------------------------------------------------------------------------------------

    def test_crossover_is_a_real_remainder_test(self):
        """
        For every order and both budgets: ``x_star`` lies inside the domain and at or below the
        plan's nominal ``100 nu`` crossover (and well inside ``DRAFT-PLAN.md`` §5.3's safe
        matching window, whose top is ``4800 nu``); the first omitted term there is below the
        budget; and the **measured** ``|delta r|`` against the reference is below the budget too.

        The last of those is the point of the test: the first omitted term is an estimator of an
        asymptotic remainder, so a crossover that passed only the estimator would not have been
        tested at all.
        """
        for nu in ORDERS:
            for budget in BUDGETS:
                crossover = bt.tail_crossover(nu, budget, budget)
                x_star = crossover.x_star

                self.assertGreaterEqual(x_star, bt.construction_min_x(nu))
                self.assertLessEqual(
                    x_star,
                    100.0 * nu,
                    f"x_star={x_star:.6g} at nu={nu}, budget={budget:.0e} is wider than the "
                    f"plan's nominal 100 nu crossover",
                )
                self.assertLessEqual(x_star, bt.tail_crossover_max_x(nu))

                self.assertLessEqual(
                    crossover.first_omitted_at_x_star,
                    budget,
                    f"first omitted term {crossover.first_omitted_at_x_star:.3e} at nu={nu}, "
                    f"x_star={x_star:.6g} exceeds the budget {budget:.0e}",
                )

                if nu == 0.5:
                    continue

                measured = abs(
                    bt.tail_residual(nu, x_star) - reference_residual(nu, x_star)
                )
                self.assertLess(
                    measured,
                    budget,
                    f"measured |delta r| {measured:.3e} at nu={nu}, x_star={x_star:.6g} "
                    f"exceeds the budget {budget:.0e}",
                )

                a_ref, _ = reference_a_and_theta_deriv(nu, x_star)
                measured_amplitude = abs(bt.tail_amplitude(nu, x_star) / a_ref - 1.0)
                self.assertLess(
                    measured_amplitude,
                    budget,
                    f"measured |delta a/a| {measured_amplitude:.3e} at nu={nu}, "
                    f"x_star={x_star:.6g} exceeds the budget {budget:.0e}",
                )

    def test_crossover_phase_condition_binds(self):
        """
        ``|delta a/a| ~ |delta r'|/2`` and ``r' ~ r/x``, so the amplitude condition is weaker than
        the phase one by a factor ``~2x``. Both are applied; assert the phase one is what actually
        sets ``x_star`` at every order and budget the campaign uses.

        If this ever fails it means the amplitude budget has been tightened relative to the phase
        budget by more than ``2 x_star``, and prompt 05's claim that the phase crossover test
        automatically covers the amplitude no longer holds.
        """
        for nu in ORDERS:
            for budget in BUDGETS:
                crossover = bt.tail_crossover(nu, budget, budget)
                if nu == 0.5:
                    self.assertEqual(crossover.binding, "domain")
                    continue
                self.assertEqual(crossover.binding, "phase")
                self.assertGreater(crossover.x_phase, crossover.x_amplitude)

    def test_crossover_is_monotone_in_the_budget(self):
        """A looser budget can only move the crossover inwards."""
        budgets = (1e-13, 1e-11, 1e-9, 1e-6, 1e-4)
        for nu in ORDERS:
            previous = None
            for budget in budgets:
                x_star = bt.tail_crossover(nu, budget, budget).x_star
                if previous is not None:
                    self.assertLessEqual(
                        x_star,
                        previous,
                        f"x_star increased as the budget loosened at nu={nu}, budget="
                        f"{budget:.0e}",
                    )
                previous = x_star

    def test_crossover_raises_when_the_budget_is_unreachable(self):
        """
        Clamp and report, do not silently extend: prompt 05 must be able to fail construction
        loudly rather than accept a tail it cannot certify. The message has to name the order and
        both budgets so the failure is actionable without a debugger.
        """
        with self.assertRaises(ValueError) as caught:
            bt.tail_crossover(1000.5, 1e-30, 1e-30, x_max=1.0e6)
        message = str(caught.exception)
        self.assertIn("1000.5", message)
        self.assertIn("1e-30", message)
        self.assertIn("x_max", message)

        # the default ceiling is 4800 max(nu, 1); a budget that needs more than that also raises
        with self.assertRaises(ValueError):
            bt.tail_crossover(1000.5, 1e-40, 1e-40)

        # and a budget just inside the default ceiling does not
        crossover = bt.tail_crossover(1000.5, 1e-30, 1e-30, x_max=1.0e9)
        self.assertGreater(crossover.x_star, 1.0e6)

    def test_estimator_vanishes_at_an_isolated_order(self):
        """
        ``mu - 25`` is exactly zero at ``nu = 5/2``, so the *second* series term vanishes
        identically there while the series itself does not. A one-term crossover at that order
        would therefore be sized by an estimator of zero, which is why
        :func:`bessel_tail.tail_crossover` refuses instead of returning the domain lower bound --
        the ``mu = 1`` degeneracy is special-cased separately, and only that one is real.
        """
        self.assertEqual(bt.tail_series_coefficients(2.5)[1], 0.0)
        self.assertEqual(bt.tail_first_omitted_term(2.5, 125.0, n_terms=1), 0.0)

        with self.assertRaises(ValueError) as caught:
            bt.tail_crossover(2.5, 1e-11, 1e-11, n_terms=1)
        self.assertIn("vanishes identically", str(caught.exception))

        # the nu = 1/2 degeneracy is *not* caught by that guard
        self.assertEqual(
            bt.tail_crossover(0.5, 1e-11, 1e-11, n_terms=1).binding, "domain"
        )

    def test_amplitude_fails_loudly_below_the_series_region(self):
        """
        ``a = (1 + r')^(-1/2)`` becomes a negative base once the asymptotic series diverges, and
        the guard raises rather than returning ``nan``: a silent ``nan`` reaching an interpolant
        is the failure mode ``README.md`` §2 (e) is about.

        Worth recording where the guard actually bites. Down the whole *supported* domain it does
        not: ``r' -> -(mu - 1)/(8 x^2)`` at leading order, so at the turning point ``x = nu`` the
        leading contribution is only ``-1/2`` and the three-term ``1 + r'`` is still 0.31 at
        ``nu = 1000.5``. It takes an argument well below the domain lower bound
        ``x_0 = sqrt(nu^2 - 1/4)`` -- here ``0.3 nu`` -- to drive it negative. So this is a guard
        against a caller handing the tail a near-region argument, not against anything the
        two-region construction does on its own.
        """
        below_domain = 0.3 * 1000.5
        self.assertLess(below_domain, bt.construction_min_x(1000.5))

        with self.assertRaises(ValueError):
            bt.tail_amplitude(1000.5, below_domain)
        with self.assertRaises(ValueError):
            bt.tail_amplitude(1000.5, np.array([below_domain, 100.0 * 1000.5]))

        # at the turning point itself the three-term series is still usable, if inaccurate
        self.assertGreater(bt.tail_amplitude(1000.5, 1000.5), 1.0)

    # ------------------------------------------------------------------------------------------
    # 5. Branch sanity at high order
    # ------------------------------------------------------------------------------------------

    def test_residual_is_not_reduced_mod_two_pi(self):
        """
        ``RECONCILIATION.md`` C2: the residual is **not** sub-cycle at high order. At
        ``nu = 1000.5`` it is 5.002540 rad at ``100 nu`` and 570.82 rad at the bottom of the
        domain, so anything that wrapped it into ``(-pi, pi]`` would be wrong by an exact multiple
        of ``2 pi``.

        Assert the value at ``100 nu`` against that measurement, and assert ``r > pi`` at the
        crossover for both budgets, which is what fails if anyone ever folds this quantity.
        """
        r_at_100nu = bt.tail_residual(1000.5, 100.0 * 1000.5)
        self.assertAlmostEqual(r_at_100nu, 5.002540, places=6)
        self.assertGreater(r_at_100nu, math.pi)

        for budget in BUDGETS:
            crossover = bt.tail_crossover(1000.5, budget, budget)
            r_at_x_star = bt.tail_residual(1000.5, crossover.x_star)
            self.assertGreater(
                r_at_x_star,
                math.pi,
                f"r(x_star) = {r_at_x_star:.6g} at nu=1000.5, budget={budget:.0e} looks "
                f"wrapped",
            )

        # and the residual grows monotonically as x decreases towards the turning point
        r_low = bt.tail_residual(1000.5, 5.0 * 1000.5)
        self.assertGreater(r_low, r_at_100nu)

    # ------------------------------------------------------------------------------------------
    # 6. Interface and hygiene
    # ------------------------------------------------------------------------------------------

    def test_module_does_not_import_a_scipy_bessel_routine(self):
        """
        The tail must evaluate no Bessel routine at all -- that is what carries the supported
        ``x_max`` past the ``x ~ 2.5e15`` Amos cliff (``RECONCILIATION.md`` C1) and what keeps the
        sampler three decades below the silent ``hankel1e`` failure boundary. References belong in
        the tests, not in the module.

        Checked in a subprocess so that a ``scipy.special`` already imported by another test
        module cannot mask the result.
        """
        script = (
            "import sys\n"
            "import LiouvilleGreen.bessel_tail\n"
            "print('SPECIAL', 'scipy.special' in sys.modules)\n"
        )
        repository_root = os.path.dirname(os.path.dirname(os.path.abspath(bt.__file__)))
        environment = dict(os.environ)
        environment["PYTHONPATH"] = repository_root
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=repository_root,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"importing bessel_tail failed: {completed.stderr}",
        )
        self.assertIn("SPECIAL False", completed.stdout, completed.stdout)

    def test_vectorized_and_scalar_agree(self):
        """Every entry point accepts a scalar or an array and returns the matching shape."""
        nu = 20.5
        grid = np.array([100.0, 250.0, 1e3, 1e5, 1e9])

        for function in (
            bt.tail_residual,
            bt.tail_residual_deriv,
            bt.tail_residual_log_deriv,
            bt.tail_amplitude,
            bt.tail_first_omitted_term,
        ):
            vector = function(nu, grid)
            self.assertIsInstance(vector, np.ndarray)
            self.assertEqual(vector.shape, grid.shape)
            for index, x in enumerate(grid):
                scalar = function(nu, float(x))
                self.assertIsInstance(scalar, float)
                self.assertEqual(scalar, vector[index])

    def test_invalid_arguments_raise(self):
        for n_terms in (0, 4, 2.5):
            with self.assertRaises(ValueError):
                bt.tail_residual(2.5, 100.0, n_terms=n_terms)
        for x in (0.0, -1.0):
            with self.assertRaises(ValueError):
                bt.tail_residual(2.5, x)
        with self.assertRaises(ValueError):
            bt.tail_residual(2.5, np.array([1.0, -1.0]))
        with self.assertRaises(ValueError):
            bt.tail_crossover(2.5, 0.0, 1e-11)
        with self.assertRaises(ValueError):
            bt.tail_crossover(2.5, 1e-11, -1.0)
        with self.assertRaises(ValueError):
            bt.tail_crossover(2.5, 1e-11, 1e-11, safety=0.0)

    def test_coefficients_match_the_published_series(self):
        """
        Pin the four DLMF 10.18.18 / A&S 9.2.29 coefficients as literals, in the ``(8x)^k``
        grouping they were checked in.

        The third and fourth denominators here are 5120 and 229376. ``DRAFT-PLAN.md`` §7.2 and
        campaign prompt 03 §2 print 15360 for the third, which is wrong by a factor of three;
        campaign log 03 records the numerical determination that settled it. This test exists so
        that nobody "corrects" the module back towards the plan text.
        """
        for nu in ORDERS:
            mu = 4.0 * nu * nu
            expected = (
                (mu - 1.0) / 8.0,
                4.0 * (mu - 1.0) * (mu - 25.0) / (3.0 * 8.0**3),
                32.0 * (mu - 1.0) * (mu * mu - 114.0 * mu + 1073.0) / (5.0 * 8.0**5),
                64.0
                * (mu - 1.0)
                * (5.0 * mu**3 - 1535.0 * mu**2 + 54703.0 * mu - 375733.0)
                / (7.0 * 8.0**7),
            )
            for shipped, reference in zip(bt.tail_series_coefficients(nu), expected):
                if reference == 0.0:
                    self.assertEqual(shipped, 0.0)
                else:
                    self.assertLess(abs(shipped / reference - 1.0), 1e-14)


if __name__ == "__main__":
    unittest.main()
