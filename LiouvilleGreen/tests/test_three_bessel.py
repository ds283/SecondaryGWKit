import json
import math
import unittest
from datetime import datetime
from math import sqrt, pi
from unittest.mock import patch

import numpy as np

from AdaptiveLevin import adaptive_levin_sincos
from ComputeTargets.QuadSourceIntegral import _three_bessel_integrals
from ComputeTargets.QuadSourceIntegral_debug import bessel_function_plot
from CosmologyConcepts import wavenumber
from LiouvilleGreen import three_bessel_integrals as three_bessel_module
from LiouvilleGreen.bessel_phase import bessel_phase, XSplineWrapper
from LiouvilleGreen.phase_spline import phase_spline
from LiouvilleGreen.tests.bessel_reference import (
    TIER_EXACT,
    reference_bundle,
)
from LiouvilleGreen.three_bessel_integrals import (
    _PHASE_GROUP_SIGNS,
    _PhaseGroup,
    quad_JJJ,
    quad_YJJ,
)
from Units import Mpc_units
from utilities import format_time

EPS = float(np.finfo(float).eps)


class TestBessel(unittest.TestCase):

    atol = 1e-25
    rtol = 1e-8

    sqrt_2_over_pi = sqrt(2.0 / pi)

    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)

        self.units = Mpc_units

        # self.r = wavenumber(store_id=3, k_inv_Mpc=79.37005259841, units=self.units)
        # self.q = wavenumber(store_id=2, k_inv_Mpc=79.37005259841, units=self.units)
        # self.k = wavenumber(store_id=1, k_inv_Mpc=49999999.99999999, units=self.units)

        self.r = wavenumber(
            store_id=3, k_inv_Mpc=sqrt(100.0 * 100.0 + 200.0 * 200.0), units=self.units
        )
        self.q = wavenumber(store_id=2, k_inv_Mpc=200.0, units=self.units)
        self.k = wavenumber(store_id=1, k_inv_Mpc=100.0, units=self.units)

        self.b = 0
        self.cs_sq = 1.0

        self.min_eta = 0.0000000001696765007801783
        self.max_eta = 10524
        # self.max_eta = 7.196808204366744e-06
        # self.max_eta = 5.05340787180641e-07
        # self.max_eta = 1e-5
        # self.max_eta = 1e-3

        self.max_x = max(self.k.k, self.q.k, self.r.k) * self.max_eta
        self.min_x = min(self.k.k, self.q.k, self.r.k) * self.min_eta * sqrt(self.cs_sq)

        self.phase_data_0pt5 = bessel_phase(
            0.5 + self.b, self.max_x + 10.0, atol=1e-25, rtol=1e-14
        )
        self.phase_data_2pt5 = bessel_phase(
            2.5 + self.b, self.max_x + 10.0, atol=1e-25, rtol=1e-14
        )

    def test_integrals(self):
        self._test_integrals(100.0, self.phase_data_0pt5, label="0.5")
        self._test_integrals(3.0, self.phase_data_0pt5, label="0.5")

        self._test_integrals(100.0, self.phase_data_2pt5, label="2.5")
        self._test_integrals(3.0, self.phase_data_2pt5, label="2.5")

    def _test_integrals(self, min_x, phase_data, label):
        phase: phase_spline = phase_data["phase"]
        m: XSplineWrapper = phase_data["mod"]

        def Levin_f(x):
            return m(x)

        # J integral order 0.5
        J_data = adaptive_levin_sincos(
            x_span=(min_x, self.max_x),
            f=[Levin_f, lambda x: 0.0],
            theta={"theta": lambda x: phase.raw_theta(x)},
            atol=self.atol,
            rtol=self.rtol,
            chebyshev_order=12,
        )
        print(f"\n** J integral data")
        value = J_data["value"]
        regions = J_data["regions"]
        evaluations = J_data["evaluations"]
        elapsed = J_data["elapsed"]
        print(
            f"integral_({min_x})^({self.max_x}) J_({label})(x) = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        # for region in regions:
        #     print(f"  -- region: ({region[0]}, {region[1]})")

        # Y integral order 0.5
        Y_data = adaptive_levin_sincos(
            x_span=(min_x, self.max_x),
            f=[lambda x: 0.0, lambda x: -Levin_f(x)],
            theta={"theta": lambda x: phase.raw_theta(x)},
            atol=self.atol,
            rtol=self.rtol,
            chebyshev_order=12,
        )
        print(f"\n** Y integral data")
        value = Y_data["value"]
        regions = Y_data["regions"]
        evaluations = Y_data["evaluations"]
        elapsed = Y_data["elapsed"]
        print(
            f"integral_({min_x})^({self.max_x}) Y_({label})(x) = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        # for region in regions:
        #     print(f"  -- region: ({region[0]}, {region[1]})")

    def test_three_bessel(self):
        timestamp = datetime.now().replace(microsecond=0)
        phase_data = {"0pt5": self.phase_data_0pt5, "2pt5": self.phase_data_2pt5}
        bessel_function_plot(phase_data, 0.0, timestamp)

        data0pt5_Levin = _three_bessel_integrals(
            self.k,
            self.q,
            self.r,
            min_eta=self.min_eta,
            max_eta=self.max_eta,
            b=self.b,
            phase_data=phase_data,
            nu_type="0pt5",
            atol=self.atol,
            rtol=self.rtol,
        )

        data2pt5_Levin = _three_bessel_integrals(
            self.k,
            self.q,
            self.r,
            min_eta=self.min_eta,
            max_eta=self.max_eta,
            b=self.b,
            phase_data=phase_data,
            nu_type="2pt5",
            atol=self.atol,
            rtol=self.rtol,
        )

        print(f"\n** 0pt5 integral data")
        print(json.dumps(data0pt5_Levin, indent=4, sort_keys=True))

        print(f"\n** 2pt5 integral data")
        print(json.dumps(data2pt5_Levin, indent=4, sort_keys=True))


# ----------------------------------------------------------------------------------------------
# Phase groups (prompt 07)
#
# The reference here is built with mpmath at 60 digits, and it is built at the *exact* products
# k x, q x, s x rather than at the doubles fl(k x), fl(q x), fl(s x). That is deliberate and it is
# what makes the measurement mean anything: the true group phase of the true integrand is
#
#     Theta(x) = sum_i e_i theta_i(m_i . x)          (exact real products)
#              = K x + C + sum_i e_i r_i(m_i . x),
#
# and a route that forms fl(m_i x) for each leading term commits an error ~eps m_i x with unit
# sensitivity, because theta' -> 1. Scoring against a reference taken at fl(m_i x) would hide
# exactly the error the restructure removes. bessel_reference's helpers all evaluate at a supplied
# double (correctly -- DRAFT-PLAN.md 7.5 defines single-factor accuracy at the supplied x), so they
# cannot be used for this and the two mpmath helpers below are local.
#
# Both hold |r| <= pi, which is true for every order used here (r(x_0) = 1.18 rad at nu = 5/2 and
# grows like pi nu/2 - sqrt(nu^2 - 1/4)), so the residual branch needs no walk: the principal
# argument *is* the tail-continuous residual. Do not raise the orders here without re-checking that
# -- see IMPLEMENTATION_STATE.md standing note 9.
# ----------------------------------------------------------------------------------------------

REFERENCE_DPS = 60

#: Orders of the three factors in the phase-group tests, distinct so that a sign slip in
#: (e_nu, e_sigma) cannot cancel out, and all three exact half-integers so that TIER_EXACT is
#: available for the convention check.
GROUP_ORDERS = (0.5, 1.5, 2.5)

#: Signs (1, e_nu, e_sigma) of the cancellative group used by the accuracy tests.
GROUP_SIGNS = (1.0, -1.0, -1.0)

#: Coefficient triples, with those signs, spanning exact cancellation to none. The first three are
#: built so that K = k - q - s is *exactly* representable (2 + 2^-n, 1.5, 0.5 are all exact and
#: each subtraction is exact by Sterbenz), which isolates the arithmetic of the group phase from
#: the input-coefficient uncertainty of DRAFT-PLAN.md 8.2: an inexact K contributes x delta K,
#: which is a property of the inputs and not of the assembly.
GROUP_CASES = {
    "K = 0 exactly": (2.0, 1.5, 0.5),
    "K/max ~ 1e-6": (2.0 + 2.0**-20, 1.5, 0.5),
    "K/max ~ 1e-10": (2.0 + 2.0**-33, 1.5, 0.5),
    "generic": (2.1, 1.5, 0.5),
}

GROUP_X_MAX = 1.0e12


def _mpmath_theta_and_deriv(nu: float, argument, mpmath):
    """
    ``(theta_nu, theta_nu')`` at an ``mpmath`` argument, inside an active ``workdps`` block.

    ``theta = y + c_nu + r`` with ``r`` the principal value of ``atan2(J, -Y) - (y + c_nu)``, which
    is the tail-continuous branch while ``|r| <= pi``; ``theta' = (2/pi)/(y (J^2 + Y^2))`` is the
    exact Wronskian oracle (README 2 (f)).
    """
    J = mpmath.besselj(nu, argument, maxterms=10**7, maxprec=10**6)
    Y = mpmath.bessely(nu, argument, maxterms=10**7, maxprec=10**6)

    c_nu = mpmath.pi / 4 - mpmath.pi * mpmath.mpf(nu) / 2
    r = mpmath.atan2(J, -Y) - (argument + c_nu)
    r = r - 2 * mpmath.pi * mpmath.floor(r / (2 * mpmath.pi))
    if r > mpmath.pi:
        r = r - 2 * mpmath.pi

    return argument + c_nu + r, (2 / mpmath.pi) / (argument * (J * J + Y * Y))


def _group_reference(orders, coefficients, signs, x):
    """
    ``(Theta(x), dTheta/dlog x)`` as doubles, from 60-digit ``mpmath`` at the exact products.

    ``dTheta/dlog x = sum_i e_i (m_i x) theta_i'(m_i x)``, formed without restructuring: at 60
    digits the cancellation of the leading terms costs a dozen digits out of sixty, so the
    reference does not need to be clever in the way the object under test does.
    """
    import mpmath

    with mpmath.workdps(REFERENCE_DPS):
        x_mp = mpmath.mpf(float(x))
        theta = mpmath.mpf(0)
        theta_log_deriv = mpmath.mpf(0)
        for order, coefficient, sign in zip(orders, coefficients, signs):
            argument = mpmath.mpf(float(coefficient)) * x_mp
            value, deriv = _mpmath_theta_and_deriv(order, argument, mpmath)
            theta = theta + sign * value
            theta_log_deriv = theta_log_deriv + sign * argument * deriv

        return float(theta), float(theta_log_deriv)


def _legacy_group_theta(phases, coefficients, signs):
    """
    The route prompt 07 replaced: the group phase as a sum of three independently reconstructed
    raw phases, and its bounded angle as a sum of three bounded angles
    (``three_bessel_integrals._phase_group`` before this commit).

    Kept here, rather than measured once and quoted in a log, because the improvement is the
    point of the commit and a claim about it should be re-checkable on any tree.
    """
    (phase_mu, phase_nu, phase_sigma) = phases
    (k, q, s) = coefficients
    (_, e_nu, e_sigma) = signs

    def theta(log_x):
        x = np.exp(log_x)
        return (
            phase_mu.raw_theta(k * x)
            + e_nu * phase_nu.raw_theta(q * x)
            + e_sigma * phase_sigma.raw_theta(s * x)
        )

    def theta_mod_2pi(log_x):
        x = np.exp(log_x)
        return (
            phase_mu.theta_mod_2pi(k * x)
            + e_nu * phase_nu.theta_mod_2pi(q * x)
            + e_sigma * phase_sigma.theta_mod_2pi(s * x)
        )

    def theta_deriv(log_x):
        x = np.exp(log_x)
        return (
            phase_mu.theta_deriv(k * x, log_derivative=True)
            + e_nu * phase_nu.theta_deriv(q * x, log_derivative=True)
            + e_sigma * phase_sigma.theta_deriv(s * x, log_derivative=True)
        )

    return theta, theta_mod_2pi, theta_deriv


class TestPhaseGroups(unittest.TestCase):
    """
    The ``K x + C + R(x)`` assembly of ``three_bessel_integrals._PhaseGroup`` (prompt 07).

    Every accuracy claim here is scored against 60-digit ``mpmath`` built independently of the
    objects under test, at the exact products; see the module comment above for why that
    distinction is load-bearing.
    """

    def _build(self, coefficients, orders=GROUP_ORDERS, x_max=GROUP_X_MAX):
        """The three phase objects and the sampling grid for one coefficient triple."""
        phases = tuple(
            bessel_phase(order, 1.1 * coefficient * x_max)["phase"]
            for order, coefficient in zip(orders, coefficients)
        )
        # every factor must be inside its own domain, i.e. x >= min_x_i / m_i
        x_lo = max(
            phase.min_x / coefficient
            for phase, coefficient in zip(phases, coefficients)
        )
        grid = np.logspace(np.log10(1.0001 * x_lo), np.log10(x_max), 41)
        return phases, grid

    # -- the restructure itself ----------------------------------------------------------------

    def test_the_group_phase_survives_cancellation_of_its_leading_term(self):
        """
        |delta Theta| for exact, near and absent cancellation, against the route this replaced.

        The assertion that matters is the *declaration*: the group's own theta_abserr, plus the
        one rounding of the product K x that no double-valued phase of that size can avoid, must
        bound the measured error at every point. The before/after ratio is measured and printed
        rather than being the only claim -- DRAFT-PLAN.md 8.2's restructure is justified by an
        argument, and this is the number that makes it a measurement.
        """
        for label, coefficients in GROUP_CASES.items():
            with self.subTest(case=label):
                phases, grid = self._build(coefficients)
                group = _PhaseGroup(phases, coefficients, GROUP_SIGNS)
                legacy_theta, _, _ = _legacy_group_theta(
                    phases, coefficients, GROUP_SIGNS
                )

                worst_new = worst_old = worst_ratio = 0.0
                at_new = at_old = 0.0
                for x in grid:
                    log_x = math.log(x)
                    x_used = math.exp(log_x)
                    reference, _ = _group_reference(
                        GROUP_ORDERS, coefficients, GROUP_SIGNS, x_used
                    )

                    new_error = abs(group.theta(log_x) - reference)
                    old_error = abs(legacy_theta(log_x) - reference)

                    # the declared error of the group, plus one rounding of K x
                    bound = (
                        group.theta_abserr(log_x) + 2.0 * EPS * abs(group.K) * x_used
                    )
                    self.assertLessEqual(
                        new_error,
                        bound,
                        msg=(
                            f"{label}: |delta Theta| = {new_error:.6e} at x = {x_used:.6e} "
                            f"exceeds the declared {group.theta_abserr(log_x):.6e} plus the "
                            f"eps |K| x product floor (bound {bound:.6e})"
                        ),
                    )

                    worst_ratio = max(worst_ratio, new_error / bound)
                    if new_error > worst_new:
                        worst_new, at_new = new_error, x_used
                    if old_error > worst_old:
                        worst_old, at_old = old_error, x_used

                print(
                    f"\n@@ group phase, {label}: K={group.K!r}\n"
                    f"   |delta Theta| new {worst_new:.4e} (x={at_new:.5g}), "
                    f"old {worst_old:.4e} (x={at_old:.5g}), "
                    f"ratio old/new {worst_old / worst_new:.4g}; "
                    f"worst error/declared-bound {worst_ratio:.4g}"
                )

                if label == "generic":
                    # no cancellation to preserve: the group phase is genuinely ~1e11 rad, and
                    # both routes sit at one ulp of it. Pin the floor model rather than claim an
                    # improvement -- see [07-generic-K-product-rounding].
                    self.assertLessEqual(
                        worst_new, 4.0 * EPS * abs(group.K) * GROUP_X_MAX
                    )
                else:
                    self.assertLess(worst_new, 1.0e-9)
                    self.assertGreater(worst_old, 1.0e-6)
                    self.assertLess(worst_new, worst_old / 1.0e5)

    def test_the_group_derivative_is_scored_against_the_constituent_frequencies(self):
        """
        |delta dTheta/dlog x| against ``max(k, q, s) x``, never against the group's own derivative.

        At exact resonance dTheta/dlog x = sum_i e_i (dr_i/dlog x) passes through zero, so a
        relative error is undefined there and would be meaningless just short of it (README 6).
        The test asserts that explicitly: the group log-derivative is measured to be O(1) or
        smaller at the top of the domain in the exactly cancelling case, while the constituent
        scale is ~1e12.

        Two metrics are reported, because they say different things and only one of them is the
        prompt's acceptance criterion:

        * scaled by ``max(k, q, s) x`` per point, the two routes are **equal** to within 1 %, and
          both attain their maximum at the *bottom* of the near region. There the interpolants
          bind, x is small, and there is no large leading term to cancel, so nothing distinguishes
          them. This is the metric README 6 asks for, and 1.5e-14 is what it measures.
        * in absolute radians the routes differ by six to seven orders, because the route this
          replaced has an error growing like eps max(k, q, s) x while the restructured one does
          not. That is the improvement, and it is invisible in the scaled metric precisely because
          the scaled metric divides by x.
        """
        for label, coefficients in GROUP_CASES.items():
            with self.subTest(case=label):
                phases, grid = self._build(coefficients)
                group = _PhaseGroup(phases, coefficients, GROUP_SIGNS)
                _, _, legacy_deriv = _legacy_group_theta(
                    phases, coefficients, GROUP_SIGNS
                )
                scale_coefficient = max(coefficients)

                worst_new = worst_old = 0.0
                worst_new_abs = worst_old_abs = 0.0
                at_new = at_old = at_new_abs = at_old_abs = 0.0
                for x in grid:
                    log_x = math.log(x)
                    x_used = math.exp(log_x)
                    _, reference = _group_reference(
                        GROUP_ORDERS, coefficients, GROUP_SIGNS, x_used
                    )
                    scale = scale_coefficient * x_used

                    new_abs = abs(group.theta_deriv(log_x) - reference)
                    old_abs = abs(legacy_deriv(log_x) - reference)
                    if new_abs / scale > worst_new:
                        worst_new, at_new = new_abs / scale, x_used
                    if old_abs / scale > worst_old:
                        worst_old, at_old = old_abs / scale, x_used
                    if new_abs > worst_new_abs:
                        worst_new_abs, at_new_abs = new_abs, x_used
                    if old_abs > worst_old_abs:
                        worst_old_abs, at_old_abs = old_abs, x_used

                print(
                    f"\n@@ group derivative, {label}:\n"
                    f"   |delta dTheta/dlog x| / (max(k,q,s) x): new {worst_new:.4e} "
                    f"(x={at_new:.5g}), old {worst_old:.4e} (x={at_old:.5g})\n"
                    f"   |delta dTheta/dlog x| absolute: new {worst_new_abs:.4e} "
                    f"(x={at_new_abs:.5g}), old {worst_old_abs:.4e} (x={at_old_abs:.5g}), "
                    f"ratio old/new {worst_old_abs / worst_new_abs:.4g}"
                )
                self.assertLess(worst_new, 1.0e-11)
                if label == "generic":
                    self.assertLess(worst_new_abs, worst_old_abs / 10.0)
                else:
                    self.assertLess(worst_new_abs, worst_old_abs / 1.0e5)

        # the trap the scaling exists to avoid, stated as a measurement: at exact resonance the
        # group derivative is not bounded away from zero, so |delta| / |dTheta| is not a metric
        phases, grid = self._build(GROUP_CASES["K = 0 exactly"])
        resonant = _PhaseGroup(phases, GROUP_CASES["K = 0 exactly"], GROUP_SIGNS)
        top = math.log(grid[-1])
        self.assertEqual(resonant.K, 0.0)
        self.assertLess(abs(resonant.theta_deriv(top)), 1.0)
        _, resonant_reference = _group_reference(
            GROUP_ORDERS, GROUP_CASES["K = 0 exactly"], GROUP_SIGNS, math.exp(top)
        )
        _, _, resonant_legacy = _legacy_group_theta(
            phases, GROUP_CASES["K = 0 exactly"], GROUP_SIGNS
        )
        print(
            f"\n@@ at exact resonance dTheta/dlog x = {resonant.theta_deriv(top):.6e} at "
            f"x = {math.exp(top):.5g}, against a constituent scale of "
            f"{max(GROUP_CASES['K = 0 exactly']) * math.exp(top):.5g}\n"
            f"   reference {resonant_reference:.6e}; relative error (the metric this test "
            f"refuses to use) new "
            f"{abs(resonant.theta_deriv(top) / resonant_reference - 1.0):.3e}, old "
            f"{abs(resonant_legacy(top) / resonant_reference - 1.0):.3e}"
        )

    def test_the_bounded_angle_is_the_angle_of_the_group_not_a_sum_of_angles(self):
        """
        The (sin, cos) pair of the group, against 60-digit mpmath, versus the summed-bounded-angle
        route.

        This is the metric a Levin consumer actually sees: ``theta_mod_2pi`` reaches it only
        through sin and cos (levin_quadrature.py:1103-1105, :1120-1121).
        """
        for label, coefficients in GROUP_CASES.items():
            with self.subTest(case=label):
                import mpmath

                phases, grid = self._build(coefficients)
                group = _PhaseGroup(phases, coefficients, GROUP_SIGNS)
                _, legacy_mod, _ = _legacy_group_theta(
                    phases, coefficients, GROUP_SIGNS
                )

                worst_new = worst_old = 0.0
                for x in grid:
                    log_x = math.log(x)
                    x_used = math.exp(log_x)
                    reference, _ = _group_reference(
                        GROUP_ORDERS, coefficients, GROUP_SIGNS, x_used
                    )
                    with mpmath.workdps(REFERENCE_DPS):
                        reference_mp = mpmath.mpf(reference)
                        sin_reference = float(mpmath.sin(reference_mp))
                        cos_reference = float(mpmath.cos(reference_mp))

                    sin_new, cos_new = group.sin_cos(log_x)
                    legacy_angle = legacy_mod(log_x)
                    worst_new = max(
                        worst_new,
                        abs(sin_new - sin_reference),
                        abs(cos_new - cos_reference),
                    )
                    worst_old = max(
                        worst_old,
                        abs(math.sin(legacy_angle) - sin_reference),
                        abs(math.cos(legacy_angle) - cos_reference),
                    )

                    # the bounded angle must reproduce the pair it came from
                    bounded = group.theta_mod_2pi(log_x)
                    self.assertLess(abs(math.sin(bounded) - sin_new), 1.0e-15)
                    self.assertLess(abs(math.cos(bounded) - cos_new), 1.0e-15)
                    self.assertLessEqual(abs(bounded), math.pi)

                print(
                    f"\n@@ group (sin, cos), {label}: new {worst_new:.4e}, old {worst_old:.4e}, "
                    f"ratio old/new {worst_old / worst_new:.4g}"
                )
                if label == "generic":
                    self.assertLessEqual(worst_new, worst_old)
                else:
                    self.assertLess(worst_new, 1.0e-9)
                    self.assertLess(worst_new, worst_old / 1.0e5)

    # -- conventions ---------------------------------------------------------------------------

    def test_the_four_groups_reconstruct_the_triple_product(self):
        """
        Sign and convention check (DRAFT-PLAN.md 9 Stage 4).

        The decomposition is the identity

            sin A sin B sin C = (-sin G1 + sin G2 + sin G3 - sin G4)/4,
            (-cos A) sin B sin C = (cos G1 - cos G2 - cos G3 + cos G4)/4,

        with G_i running over ``_PHASE_GROUP_SIGNS`` and the combinations the two ``_Levin_*``
        drivers pass. Because the left-hand sides are built from each factor's own
        ``sin_cos_theta`` and then anchored to ``J/A`` and ``-Y/A`` from an independent reference,
        a slip in ``e_nu``/``e_sigma``, in a combination sign, or in the ``J = A sin theta`` /
        ``Y = -A cos theta`` convention cannot hide.

        The identity is checked against ``4 eps max_i |K_i| x`` rather than against a constant:
        the ``(+, +)`` group has ``K = k + q + s``, so its phase is ~4e12 rad at the top of the
        grid and one ulp of it is 1e-3. That floor is the subject of
        [07-generic-K-product-rounding] and is not what this test is about, so the bound carries it
        explicitly and the strict part of the test is at the bottom of the grid, where a sign slip
        would have nowhere to hide.
        """
        coefficients = (2.1, 1.5, 0.5)
        phases, grid = self._build(coefficients)
        groups = [
            _PhaseGroup(phases, coefficients, (1.0, e_nu, e_sigma))
            for (e_nu, e_sigma) in _PHASE_GROUP_SIGNS
        ]
        JJJ_combination = (-1.0, 1.0, 1.0, -1.0)
        YJJ_combination = (1.0, -1.0, -1.0, 1.0)
        max_abs_K = max(abs(group.K) for group in groups)

        worst_identity = 0.0
        worst_ratio = 0.0
        worst_convention = 0.0
        for x in grid:
            log_x = math.log(x)
            x_used = math.exp(log_x)

            pairs = []
            for order, phase, coefficient in zip(GROUP_ORDERS, phases, coefficients):
                argument = coefficient * x_used
                sin_theta, cos_theta = phase.sin_cos_theta(argument)
                pairs.append((sin_theta, cos_theta))

                # J = A sin theta, Y = -A cos theta, against an independent reference
                bundle = reference_bundle(order, argument, TIER_EXACT)
                worst_convention = max(
                    worst_convention,
                    abs(sin_theta - bundle.J / bundle.amplitude),
                    abs(-cos_theta - bundle.Y / bundle.amplitude),
                )

            sin_A, cos_A = pairs[0]
            sin_B, cos_B = pairs[1]
            sin_C, cos_C = pairs[2]

            group_sin = [group.sin_cos(log_x)[0] for group in groups]
            group_cos = [group.sin_cos(log_x)[1] for group in groups]

            JJJ_direct = sin_A * sin_B * sin_C
            JJJ_groups = (
                sum(sign * value for sign, value in zip(JJJ_combination, group_sin))
                / 4.0
            )
            YJJ_direct = -cos_A * sin_B * sin_C
            YJJ_groups = (
                sum(sign * value for sign, value in zip(YJJ_combination, group_cos))
                / 4.0
            )

            error = max(abs(JJJ_direct - JJJ_groups), abs(YJJ_direct - YJJ_groups))
            bound = 1.0e-14 + 4.0 * EPS * max_abs_K * x_used
            self.assertLessEqual(
                error,
                bound,
                msg=(
                    f"the four groups do not reconstruct the triple product at x = "
                    f"{x_used:.6e}: |direct - groups| = {error:.6e} against a bound of "
                    f"{bound:.6e}"
                ),
            )
            worst_identity = max(worst_identity, error)
            worst_ratio = max(worst_ratio, error / bound)

        print(
            f"\n@@ four-group identity: worst |direct - groups| = {worst_identity:.4e} "
            f"(worst error/bound {worst_ratio:.4g}); "
            f"worst |constituent pair - reference| = {worst_convention:.4e}"
        )
        self.assertLess(worst_convention, 1.0e-9)

    # -- what reaches the quadrature -----------------------------------------------------------

    def test_every_levin_call_receives_a_declared_phase_error(self):
        """
        All four ``adaptive_levin_sincos`` calls of a ``quad_JJJ``/``quad_YJJ`` evaluation are
        handed a "theta_abserr" (DRAFT-PLAN.md 8.1: "its consumers should pass it through"), and
        the group's declared error is the *linear* sum of its constituents' at their own
        arguments.
        """
        coefficients = (1.3, 1.7, 2.1)
        max_x = 1.0e6
        phase_data = [
            bessel_phase(order, 1.1 * coefficient * max_x)
            for order, coefficient in zip(GROUP_ORDERS, coefficients)
        ]
        phases = tuple(data["phase"] for data in phase_data)

        group = _PhaseGroup(phases, coefficients, GROUP_SIGNS)
        log_x = math.log(0.5 * max_x)
        constituents = [
            phase.theta_abserr_at(coefficient * math.exp(log_x))
            for phase, coefficient in zip(phases, coefficients)
        ]
        self.assertAlmostEqual(
            group.theta_abserr(log_x), sum(constituents), delta=1.0e-18
        )
        # linear, not in quadrature: the two differ, and the linear one is larger
        quadrature = math.sqrt(sum(value * value for value in constituents))
        self.assertGreater(group.theta_abserr(log_x), quadrature)

        keys_seen = []
        real_levin = three_bessel_module.adaptive_levin_sincos

        def spy(x_span, f, theta, *args, **kwargs):
            keys_seen.append(sorted(theta.keys()))
            return real_levin(x_span, f, theta, *args, **kwargs)

        with patch.object(three_bessel_module, "adaptive_levin_sincos", spy):
            for evaluator in (quad_JJJ, quad_YJJ):
                result = evaluator(
                    *phase_data,
                    0.0,
                    1.0,
                    2.0,
                    *coefficients,
                    max_x,
                    atol=1.0e-14,
                    rtol=1.0e-10,
                )
                self.assertTrue(np.isfinite(result.value))

        self.assertEqual(len(keys_seen), 8)
        for keys in keys_seen:
            self.assertEqual(
                keys, ["theta", "theta_abserr", "theta_deriv", "theta_mod_2pi"]
            )
        print(f"\n@@ {len(keys_seen)} Levin calls, all with keys {keys_seen[0]}")

    def test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy(self):
        """
        Refining the quadrature tolerance stops improving the answer (DRAFT-PLAN.md 9 Stage 4,
        README 6).

        The positive statement is the one asserted: the value converges to a floor, the driver
        reports ``phase_limited`` there -- meaning the region's achievable accuracy was set by the
        round-off-plus-declared-phase floor rather than by the Levin rule -- and the residual
        spread of the last three tolerances is consistent with that floor, not with the requested
        tolerance. Reading "the integral matches to 1e-12" as evidence that the phases are good is
        the error this test exists to prevent.

        J000's closed form is re-derived here rather than imported from test_3bessel_analytic.py
        (prompt 08's file): int_0^inf j_0(kx) j_0(qx) j_0(sx) x^2 dx = (pi/4)/(kqs) for a triangle
        (k, q, s), in this module's normalisation.
        """
        k, q, s = 1.3, 1.7, 2.1
        self.assertLess(abs(k - q), s)
        self.assertLess(s, k + q)
        analytic = (math.pi / 4.0) / (k * q * s)

        max_x = 1.0e12
        phase_data = [
            bessel_phase(0.5, 1.075 * coefficient * max_x) for coefficient in (k, q, s)
        ]

        results = {}
        for rtol in (1.0e-8, 1.0e-10, 1.0e-12, 1.0e-13, 1.0e-14):
            results[rtol] = quad_JJJ(
                *phase_data,
                0.0,
                0.0,
                0.0,
                k,
                q,
                s,
                max_x,
                atol=1.0e-16,
                rtol=rtol,
            )
            print(
                f"\n@@ quad_JJJ rtol={rtol:.1e}: value={results[rtol].value:.16e}, "
                f"|value - analytic|={abs(results[rtol].value - analytic):.4e}, "
                f"reported abserr={results[rtol].abserr:.4e}, "
                f"phase_limited={results[rtol].phase_limited}"
            )

        tightest = results[1.0e-14].value
        coarse_gap = abs(results[1.0e-8].value - tightest)
        floor_gap = max(
            abs(results[rtol].value - tightest) for rtol in (1.0e-12, 1.0e-13)
        )

        # it improved, and then it stopped improving
        self.assertGreater(coarse_gap, 10.0 * floor_gap)
        self.assertLess(floor_gap, 1.0e-14 * abs(analytic) + 1.0e-20)

        # and the driver says why it stopped: the floor, not the rule
        self.assertTrue(results[1.0e-14].phase_limited)

        # the residual disagreement with the oracle is far above the requested rtol, so a passing
        # tolerance would have been no evidence about the phases either way
        true_error = abs(tightest - analytic)
        self.assertGreater(true_error, 1.0e-14 * abs(analytic))
        print(
            f"\n@@ floor: |value(1e-12..1e-14) spread| = {floor_gap:.4e}, "
            f"|value - analytic| = {true_error:.4e} "
            f"({true_error / abs(analytic):.4e} relative), reported abserr "
            f"{results[1.0e-14].abserr:.4e}"
        )

    def test_the_three_bessel_integrals_still_match_their_closed_forms(self):
        """
        A smoke check on the values, not a re-tightening: ``test_3bessel_analytic.py`` owns the
        tolerances and is prompt 08's file. Two closed forms are re-derived here, J000 and Y000,
        at one fixed triangle triple.
        """
        k, q, s = 1.3, 1.7, 2.1
        max_x = 1.0e12

        JJJ_analytic = (math.pi / 4.0) / (k * q * s)
        YJJ_analytic = (
            (1.0 / 4.0)
            / (k * q * s)
            * math.log(abs(((k - q + s) * (k + q - s)) / ((k + q + s) * (k - q - s))))
        )

        for label, evaluator, analytic in (
            ("J000", quad_JJJ, JJJ_analytic),
            ("Y000", quad_YJJ, YJJ_analytic),
        ):
            with self.subTest(case=label):
                phase_data = [
                    bessel_phase(0.5, 1.075 * coefficient * max_x)
                    for coefficient in (k, q, s)
                ]
                result = evaluator(
                    *phase_data,
                    0.0,
                    0.0,
                    0.0,
                    k,
                    q,
                    s,
                    max_x,
                    atol=1.0e-14,
                    rtol=1.0e-10,
                )
                relerr = abs(result.value - analytic) / abs(analytic)
                print(
                    f"\n@@ {label}: value={result.value:.16e}, analytic={analytic:.16e}, "
                    f"relerr={relerr:.4e}, reported abserr={result.abserr:.4e}, "
                    f"converged={result.converged}, phase_limited={result.phase_limited}"
                )
                self.assertLess(relerr, 1.0e-6)


if __name__ == "__main__":
    unittest.main()
