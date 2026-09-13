"""
The long-standing tests of ``LiouvilleGreen.bessel_phase``, re-scored against the acceptance table
of ``prompts/transfer-remedial/README.md`` §6.

Three of the four tests here were written for the phase-ODE construction and were set where they
could not see a change in it: 50 % on the reconstructed Bessel values, 1e-3 at high order "to catch
garbage rather than to measure precision", and 1e-3 on the Levin integral. The two-region
amplitude-residual construction that replaced it (campaign prompts 03-05) is six to eight orders
more accurate, so the thresholds here are now the campaign's targets and the measure is the
campaign's, not a pointwise relative error against ``jv``.

**The measure.** ``bessel_reference.phase_pair_error`` and ``amplitude_error``
(``DRAFT-PLAN.md`` §6.1) replace ``|ours/theirs - 1|`` throughout. The old measure divided by
``jv(nu, x)``, which passes through zero twelve times inside the interval this file sweeps, so a
threshold on it was either meaningless near a zero or had to be set loose enough to survive one.
The phase-pair error normalizes by the reference envelope ``hypot(J, Y)`` instead, which never
vanishes, and is the phase error in radians to leading order.

**Division of labour with the newer modules.** ``test_bessel_two_region.py`` owns the acceptance
table proper -- low orders through ``x_max = 1e7``, the crossover and seam, the split evaluation
against live ``mpmath``, the zeros and extrema, the declared-accuracy check -- and
``test_bessel_compatibility.py`` owns the interface, the raw/log accessor agreement and
serialization. Nothing here duplicates them: this file keeps the *historical* four tests on their
own grids and orders, and adds only the one item README §6 requires that neither covers, namely
large arguments through 1e15 **at the two orders production actually builds** (:class:`TestProductionOrdersAtLargeArgument`).
"""

import math
import unittest
from math import fabs

import numpy as np

from AdaptiveLevin import adaptive_levin_sincos
from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.tests import bessel_reference as br
from utilities import format_time

#: Campaign ``README.md`` §6, the orders production builds and the ordinary phase derivative.
LOW_ORDER_TARGET = 1.0e-11
LOW_ORDER_DERIV_TARGET = 1.0e-9

#: Campaign ``README.md`` §6, orders >= 20.5: "accuracy is not the objective at these orders;
#: correctness is". The structural requirements those orders really carry -- the ``a_nu``
#: plausibility band, loud failure, verified branch tracking -- are asserted in
#: ``test_bessel_near_region.py`` and ``test_bessel_two_region.py``, not here.
HIGH_ORDER_TARGET = 1.0e-6

#: :class:`TestProductionOrdersAtLargeArgument`. The cached corners are 40-digit ``mpmath`` values
#: (``bessel_reference_data.json``); the measured maximum over every corner of ``nu`` = 1/2 and 5/2
#: is 5.3e-16, so this is a floor set by double-precision evaluation arithmetic and not by the
#: representation.
CORNER_TARGET = 1.0e-14


def _check_points(data, max_x):
    """
    The historical ``linspace`` grid, plus the three point sets README §6 additionally requires
    that a uniform sweep does not reliably hit: the crossover ``x_star``, the near-region
    interpolation nodes, and the midpoints between them where interpolation error peaks.

    Adding the nodes and midpoints is what covers "changes of interpolation interval"; adding
    ``x_star`` covers the crossover, and matters because every low-order maximum measured in this
    file lands exactly there (it is the tail series remainder, not interpolation). A 100-point
    ``linspace`` misses all three except by accident.
    """
    min_x = data["min_x"]
    points = [np.linspace(min_x, max_x, 100)]

    x_star = data["x_star"]
    if min_x <= x_star <= max_x:
        points.append(np.array([x_star]))

    near = data["near_region"]
    if near is not None and near.n_nodes > 1:
        u = np.asarray(near.log_x_nodes, dtype=float)
        nodes = np.exp(u)
        midpoints = np.exp(0.5 * (u[:-1] + u[1:]))
        for candidate in (nodes, midpoints):
            candidate = candidate[(candidate >= min_x) & (candidate <= max_x)]
            if candidate.size:
                points.append(candidate)

    return np.unique(np.concatenate(points))


def _score(nu, xs, data, tier):
    """
    ``(E_theta, E_A, E_deriv)`` on ``xs``, each as ``(value, x_at_maximum)``.

    ``sin theta`` and ``-cos theta`` come from ``sin_cos_theta``, i.e. from angle addition on the
    split ``theta = x + c_nu + r``, which is the quantity a consumer evaluates; taking
    ``sin(raw_theta(x))`` instead would measure the ``eps * theta`` rounding of the leading term
    rather than the representation.
    """
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    reference = br.reference_bundle(nu, xs, tier)

    phase = data["phase"]
    sin_theta, cos_theta = phase.sin_cos_theta(xs)

    e_theta, i_theta = br.phase_pair_error(
        sin_theta, -cos_theta, reference.J, reference.Y, reference.amplitude
    )
    e_amplitude, i_amplitude = br.amplitude_error(data["mod"](xs), reference.amplitude)
    e_deriv, i_deriv = br.derivative_error(phase.theta_deriv(xs), reference.theta_deriv)
    return (
        (e_theta, float(xs[i_theta])),
        (e_amplitude, float(xs[i_amplitude])),
        (e_deriv, float(xs[i_deriv])),
    )


class TestBesselPhase(unittest.TestCase):

    def _test_bessel_value(self, nu: float, max_x: float):
        if max_x <= 10.0:
            rhs_x = 10.0
        else:
            rhs_x = max_x + 5.0
        data = bessel_phase(nu, rhs_x)

        test_grid = _check_points(data, max_x)
        # nu = 3/2 and 5/2 are half-integer, so the exact closed-form tier is available and shares
        # no library at all with the object under test -- not even Amos, whose own error is 1.1e-14
        # at nu = 1/2, x = 16.26 and would be the limit of a 1e-11 claim scored against jv/yv
        # (IMPLEMENTATION_STATE.md standing notes 15 and 24)
        tier = br.best_available_tier(nu, float(test_grid[-1]))
        (e_theta, x_theta), (e_amplitude, x_amplitude), (e_deriv, x_deriv) = _score(
            nu, test_grid, data, tier
        )
        print(
            f"\n[nu={nu}, x <= {max_x:.6g}, {test_grid.size} points, tier={tier}] "
            f"E_theta={e_theta:.3e}@{x_theta:.6g} E_A={e_amplitude:.3e}@{x_amplitude:.6g} "
            f"theta' relerr={e_deriv:.3e}@{x_deriv:.6g} (x_star={data['x_star']:.6g}, "
            f"declared theta_abserr={data['theta_abserr']:.3e})"
        )

        self.assertLessEqual(
            e_theta,
            LOW_ORDER_TARGET,
            f"E_theta = {e_theta:.4g} at nu={nu}, x={x_theta:.8g} exceeds "
            f"{LOW_ORDER_TARGET:.1g}",
        )
        self.assertLessEqual(
            e_amplitude,
            LOW_ORDER_TARGET,
            f"E_A = {e_amplitude:.4g} at nu={nu}, x={x_amplitude:.8g} exceeds "
            f"{LOW_ORDER_TARGET:.1g}",
        )
        self.assertLessEqual(
            e_deriv,
            LOW_ORDER_DERIV_TARGET,
            f"theta' relative error = {e_deriv:.4g} at nu={nu}, x={x_deriv:.8g} exceeds "
            f"{LOW_ORDER_DERIV_TARGET:.1g}",
        )

    def _integrate_bessel_J(self, nu: float, min_x: float, max_x: float):
        if max_x <= 10.0:
            rhs_x = 10.0
        else:
            rhs_x = max_x + 5.0

        phase_data = bessel_phase(nu, rhs_x)

        phase = phase_data["phase"]
        mod = phase_data["mod"]

        # J_nu = m sin(theta), so the modulus is the slowly varying amplitude in the sine slot
        f = [lambda x: mod(x), lambda x: 0.0]

        Levin_data = adaptive_levin_sincos(
            x_span=(min_x, max_x),
            f=f,
            theta={
                "theta": lambda x: phase.raw_theta(x),
                "theta_mod_2pi": lambda x: phase.theta_mod_2pi(x),
                "theta_deriv": lambda x: phase.theta_deriv(x),
                # the fourth key, which nothing supplied before campaign prompt 06: the phase's
                # own declared absolute error. It changes no value here, only the reported
                # abserr, which it *raises* -- that is the accuracy claim reaching the consumer
                # (IMPLEMENTATION_STATE.md standing note 22)
                "theta_abserr": lambda x: phase.theta_abserr_at(x),
            },
            atol=1e-15,
            rtol=1e-10,
            chebyshev_order=36,
        )
        value = Levin_data["value"]
        regions = Levin_data["regions"]
        evaluations = Levin_data["evaluations"]
        elapsed = Levin_data["elapsed"]
        print(
            f"integral = {value} (reported abserr = {Levin_data['abserr']:.5g}, "
            f"{len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: {region}")

        return value

    def test_Bessel(self):
        self._test_bessel_value(3.0 / 2.0, 1000.0)
        self._test_bessel_value(5.0 / 2.0, 1000.0)

    def test_high_order(self):
        """
        High orders construct at all and give correct values.

        The orders are ``ell + 1/2`` for ``ell`` up to 1000, the range needed for non-Limber
        angular power spectra; that use case is why they are tested here rather than only at the
        orders production builds, and tightening beyond 1e-6 for it is deliberately deferred
        (campaign README §7).

        The 1e-3 this test used to assert was chosen "to catch garbage rather than to measure
        precision" for a construction whose high-order accuracy was set by the sample density of
        a phase ODE. That construction is gone: the threshold is now README §6's 1e-6 for orders
        >= 20.5, which the two-region construction clears by five to six orders. The reference is
        ``jv``/``yv``, adequate at 1e-6 but *not* at these orders below about 1e-11
        (IMPLEMENTATION_STATE.md standing note 15), which is why this test is not tightened
        further even though the object is more accurate than 1e-6.
        """
        for ell in [2, 20, 100, 400, 1000]:
            nu = ell + 0.5
            max_x = max(10.0 * nu, 1000.0)

            data = bessel_phase(nu, max_x)

            # the full domain, turning-point interval included. The old sweep started at
            # 1.5 min_x + 1 to "stay away from the turning point"; the two-region construction is
            # sampled adaptively there and needs no such exclusion, and DRAFT-PLAN.md §4.7 locates
            # every derivative maximum in exactly the interval that exclusion removed
            test_grid = _check_points(data, max_x)
            tier = br.best_available_tier(nu, float(test_grid[-1]))
            (e_theta, x_theta), (e_amplitude, x_amplitude), (e_deriv, x_deriv) = _score(
                nu, test_grid, data, tier
            )
            print(
                f"\n[nu={nu}, x <= {max_x:.6g}, {test_grid.size} points, tier={tier}] "
                f"E_theta={e_theta:.3e}@{x_theta:.6g} E_A={e_amplitude:.3e}@{x_amplitude:.6g} "
                f"theta' relerr={e_deriv:.3e}@{x_deriv:.6g} (x_star={data['x_star']:.6g})"
            )

            for label, value, x in (
                ("E_theta", e_theta, x_theta),
                ("E_A", e_amplitude, x_amplitude),
                ("theta' relerr", e_deriv, x_deriv),
            ):
                self.assertLessEqual(
                    value,
                    HIGH_ORDER_TARGET,
                    f"{label} = {value:.4g} at nu={nu}, x={x:.8g} exceeds "
                    f"{HIGH_ORDER_TARGET:.1g}",
                )

    def test_phase_derivative(self):
        """
        The phase function satisfies ``dtheta/dx = (2/pi)/(x m(x))`` exactly -- it is the
        Wronskian identity ``A^2 theta' = 2/(pi x)`` rearranged, and it holds throughout the
        oscillatory region rather than asymptotically -- which gives a closed-form oracle for
        ``theta_deriv`` built only from the reference ``J`` and ``Y``.

        This is the campaign's standing regression gate (``RECONCILIATION.md`` C4): it is the one
        pre-existing test that contracted the quantity ``DRAFT-PLAN.md`` §4.7 identifies as
        binding, and it must pass from prompt 05 onward. **It is tightened here and must never be
        loosened.** ``nu = 2.5`` moves to README §6's low-order 1e-9; 20.5 and 100.5 stay at 1e-6,
        which is README §6's target for orders >= 20.5 and also all ``jv``/``yv`` can score at
        those orders (standing note 15).

        The lower bound also moves. The test used to sweep from ``2 x_0 + 1``, excluding the
        interval next to the turning point -- where §4.7 locates every derivative maximum -- which
        made it easier than the quantity it claimed to measure. It now starts at ``x_0`` itself.
        """
        for nu, target in (
            (2.5, LOW_ORDER_DERIV_TARGET),
            (20.5, HIGH_ORDER_TARGET),
            (100.5, HIGH_ORDER_TARGET),
        ):
            max_x = max(10.0 * nu, 1000.0)
            data = bessel_phase(nu, max_x)

            phase = data["phase"]
            min_x = data["min_x"]

            grid = np.concatenate(
                [
                    np.linspace(min_x, 0.95 * max_x, 25),
                    # the turning-point interval itself, densely: the interval the old lower
                    # bound of 2 x_0 + 1 removed
                    np.linspace(min_x, min(2.0 * min_x + 1.0, 0.95 * max_x), 40),
                ]
            )
            grid = np.unique(grid)

            tier = br.best_available_tier(nu, float(grid[-1]))
            reference = br.reference_bundle(nu, grid, tier)
            relerr, index = br.derivative_error(
                phase.theta_deriv(grid), reference.theta_deriv
            )
            print(
                f"\n[nu={nu}, theta' from x_0={min_x:.6g}, tier={tier}] "
                f"max relative error {relerr:.3e} at x={grid[index]:.6g} "
                f"(target {target:.1g})"
            )
            self.assertLess(
                relerr,
                target,
                f"theta_deriv nu={nu} at x={grid[index]:.5g}: relerr={relerr:.3g} "
                f"exceeds {target:.1g}",
            )

    def test_bessel_J_integral(self):
        """
        ``int J_nu dx`` by Levin quadrature on the amplitude-phase representation.

        **What limits this test is the closed-form reference values, and then the quadrature --
        not the Bessel phase.** The four expectations were previously quoted to ten significant
        digits, which is a relative floor of 1.9e-10 on the ``nu = 3/2`` pair; they are replaced
        below by 20-digit values, so the floor is now the quadrature itself, measured at 1.2e-12
        (``nu = 3/2``) and 8.9e-12 (``nu = 5/2``) on the ``[5, 100]`` spans against those values.
        The threshold is set to 1e-10, an order above that, rather than to anything the phase
        construction could support: the object declares 5.0e-12 rad of phase error over these
        spans and the reported Levin ``abserr`` is a few times 1e-12, so tightening further would
        be measuring ``adaptive_levin_sincos``, not ``bessel_phase``.
        """
        MAX_INTEGRAL_RELERR = 1e-10

        def do_test(nu, min_x, max_x, expected):
            value = self._integrate_bessel_J(nu, min_x, max_x)
            relerr = fabs((value - expected) / expected)
            self.assertLess(
                relerr,
                MAX_INTEGRAL_RELERR,
                f"BesselJ integral nu={nu} min_x={min_x} max_x={max_x} "
                f"(expected={expected!r}, value={value!r}, relerr={relerr:.5g})",
            )

        # 20-digit reference values, from mpmath.quad at 50 and 80 decimal digits over two
        # different panelizations of the interval (one panel, and unit-width panels so that the
        # oscillation is resolved); all four agree to every digit shown. The 10-digit values these
        # replace agreed with them, so this changes no claim about the integrand -- it removes the
        # reference as the limiting error. Reproduce with:
        #     mpmath.quad(lambda t: mpmath.besselj(nu, t), [lo, hi], maxdegree=14)
        do_test(3.0 / 2.0, 5.0, 10.0, -0.19279384236256379845)
        do_test(3.0 / 2.0, 5.0, 100.0, -0.30118794575409196193)

        do_test(5.0 / 2.0, 5.0, 10.0, -0.45027807320356300301)
        do_test(5.0 / 2.0, 5.0, 100.0, -0.20136738094269598562)


class TestProductionOrdersAtLargeArgument(unittest.TestCase):
    """
    README §6's "selected large arguments through 1e15", at the two orders ``main.py`` builds.

    ``test_bessel_two_region.TestSplitEvaluation`` already checks the split evaluation against
    live 70-digit ``mpmath`` at 1e3, 1e7, 1e12 and 1e15 -- but only for ``nu`` = 3/2 and 7/4, and
    ``test_bessel_two_region.TestAcceptanceTable`` stops at ``x_max = 1e7``. The orders production
    actually builds are 1/2 and 5/2 (``main.py:516-528``, ``b_value = 0.0``), so those are the two
    that carry a production consequence and neither module scores them above 1e7. This closes that
    gap using the committed 40-digit corner table, so it costs no ``mpmath`` at test time.
    """

    ORDERS = (0.5, 2.5)

    def test_every_cached_corner_through_1e15(self):
        for nu in self.ORDERS:
            data = bessel_phase(nu, 2.0e15)
            phase = data["phase"]
            mod = data["mod"]

            checked = 0
            worst_theta = (-1.0, float("nan"))
            worst_amplitude = (-1.0, float("nan"))
            worst_deriv = (-1.0, float("nan"))
            for corner in br.cached_corners(nu):
                if not (data["min_x"] <= corner.x <= data["max_x"]):
                    continue
                checked += 1

                sin_theta, cos_theta = phase.sin_cos_theta(corner.x)
                e_theta = max(
                    abs(sin_theta - corner.J / corner.amplitude),
                    abs(-cos_theta - corner.Y / corner.amplitude),
                )
                e_amplitude = abs(mod(corner.x) / corner.amplitude - 1.0)
                e_deriv = abs(phase.theta_deriv(corner.x) / corner.theta_deriv - 1.0)

                if e_theta > worst_theta[0]:
                    worst_theta = (e_theta, corner.x)
                if e_amplitude > worst_amplitude[0]:
                    worst_amplitude = (e_amplitude, corner.x)
                if e_deriv > worst_deriv[0]:
                    worst_deriv = (e_deriv, corner.x)

                self.assertLessEqual(
                    e_theta,
                    CORNER_TARGET,
                    f"E_theta = {e_theta:.4g} at nu={nu}, x={corner.x:.8g}",
                )
                self.assertLessEqual(
                    e_amplitude,
                    CORNER_TARGET,
                    f"E_A = {e_amplitude:.4g} at nu={nu}, x={corner.x:.8g}",
                )
                self.assertLessEqual(
                    e_deriv,
                    CORNER_TARGET,
                    f"theta' relerr = {e_deriv:.4g} at nu={nu}, x={corner.x:.8g}",
                )

            # the corner set runs {x_0, 1.5 x_0, 10 nu, 100 nu, 10, 1e3, 1e7, 1e12, 1e15} for
            # these orders, so anything much smaller means the table or the domain moved
            self.assertGreaterEqual(checked, 8)
            print(
                f"\n[nu={nu}, {checked} cached 40-digit corners up to 1e15] "
                f"E_theta={worst_theta[0]:.3e}@{worst_theta[1]:.4g} "
                f"E_A={worst_amplitude[0]:.3e}@{worst_amplitude[1]:.4g} "
                f"theta' relerr={worst_deriv[0]:.3e}@{worst_deriv[1]:.4g}"
            )

    def test_the_log_input_mode_is_scored_at_its_own_argument(self):
        """
        README §6 requires "raw and logarithmic input modes, with explicitly matched reference
        arguments", and the matching is the whole content of the requirement.

        ``test_bessel_compatibility.test_raw_and_log_input_modes_agree_at_matched_arguments``
        compares the two *accessors* to each other. This compares the logarithmic mode to an
        independent ``mpmath`` reference, and it must be built at ``exp(u)`` -- the double the
        accessor is actually handed -- not at the ``x`` that ``u`` came from. The distinction is
        not pedantic: ``exp(log(1e15))`` differs from ``1e15`` by about 0.03 in ``x``, hence by
        0.03 rad in the phase, so a reference taken at ``1e15`` would report a 2.5e-2 "error" in a
        representation that is correct to 1.1e-16 (``DRAFT-PLAN.md`` §7.5). A handful of points
        only, because each ``mpmath`` evaluation costs a few milliseconds.
        """
        for nu in self.ORDERS:
            data = bessel_phase(nu, 1.0e7)
            phase = data["phase"]
            for x in (25.0, 1.0e3, 1.0e6):
                u = math.log(x)
                x_round = math.exp(u)
                self.assertGreaterEqual(x_round, data["min_x"])

                J, Y = br.mpmath_reference(nu, x_round, dps=50)
                amplitude = math.hypot(J, Y)

                sin_theta, cos_theta = phase.sin_cos_theta(u, x_is_log=True)
                e_theta = max(
                    abs(sin_theta - J / amplitude), abs(-cos_theta - Y / amplitude)
                )
                print(
                    f"\n[nu={nu}, log mode at u=log({x:.4g}), scored at exp(u)="
                    f"{x_round!r}] E_theta={e_theta:.3e}"
                )
                self.assertLessEqual(e_theta, CORNER_TARGET)


if __name__ == "__main__":
    unittest.main()
