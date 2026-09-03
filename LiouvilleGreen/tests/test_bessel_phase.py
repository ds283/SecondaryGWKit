import unittest
from math import fabs

import numpy as np
from scipy.special import jv, yv

from AdaptiveLevin import adaptive_levin_sincos
from LiouvilleGreen.bessel_phase import bessel_phase
from utilities import format_time


class TestBesselPhase(unittest.TestCase):

    def _test_bessel_value(self, nu: float, max_x: float):
        if max_x <= 10.0:
            rhs_x = 10.0
        else:
            rhs_x = max_x + 5.0
        data = bessel_phase(nu, rhs_x)

        min_x = data["min_x"]
        phase = data["phase"]
        bessel_j = data["bessel_j"]
        bessel_y = data["bessel_y"]

        test_grid = np.linspace(min_x, max_x, 100)
        for x in test_grid:
            our_j = bessel_j(x)
            our_y = bessel_y(x)

            their_j = jv(nu, x)
            their_y = yv(nu, x)

            REL_DIFF = 0.5
            self.assertTrue(
                fabs((our_j - their_j) / their_j) < REL_DIFF,
                f"BesselJ nu=3/2 at x={x:.5g} (phase={phase.raw_theta(x)}, our value={our_j:.5g}, their value={their_j:.5g}, relative error = {fabs((our_j - their_j) / their_j):.2%})",
            )
            self.assertTrue(
                fabs((our_y - their_y) / their_y) < REL_DIFF,
                f"BesselY nu=3/2 at x={x:.5g} (phase={phase.raw_theta(x)}, our value={our_y:.5g}, their value={their_y:.5g}, relative error = {fabs((our_y - their_y) / their_y):.2%})",
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
            f"integral = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: {region}")

        return value

    def test_Bessel(self):
        self._test_bessel_value(3.0 / 2.0, 1000.0)
        self._test_bessel_value(5.0 / 2.0, 1000.0)

    def test_high_order(self):
        """
        The Liouville-Green initial condition is sin(theta) = J/sqrt(m) with m = J^2 + Y^2. Taking J/m
        instead is within the domain of asin() only for nu below about 5.5, because asymptotically
        m ~ 2/(pi x) and so J/m grows like sqrt(x); above that the construction failed outright with a
        non-finite initial state. These orders (ell + 1/2 for ell up to 1000) all sit above that
        threshold and are the range needed for non-Limber angular power spectra.
        """
        for ell in [2, 20, 100, 400, 1000]:
            nu = ell + 0.5
            max_x = max(10.0 * nu, 1000.0)

            data = bessel_phase(nu, max_x)

            min_x = data["min_x"]
            bessel_j = data["bessel_j"]
            bessel_y = data["bessel_y"]

            # stay away from the turning point at min_x, where the Liouville-Green representation is
            # marginal and both J and Y are changing character
            for x in np.linspace(1.5 * min_x + 1.0, 0.95 * max_x, 25):
                for label, ours, theirs in (
                    ("J", bessel_j(x), jv(nu, x)),
                    ("Y", bessel_y(x), yv(nu, x)),
                ):
                    # the point of this test is that high orders construct at all and give correct
                    # values; at the default phase sample density the reconstruction is good to about
                    # 1e-4, so the threshold is set to catch garbage rather than to measure precision
                    relerr = fabs((ours - theirs) / theirs)
                    self.assertTrue(
                        relerr < 1e-3,
                        f"Bessel{label} nu={nu} at x={x:.5g}: ours={ours:.8g}, "
                        f"scipy={theirs:.8g}, relerr={relerr:.3g}",
                    )

    def test_phase_derivative(self):
        """
        The phase function satisfies dtheta/dx = (2/pi)/(x m(x)) exactly -- this is the ODE that
        bessel_phase() integrates -- which gives a closed-form oracle for theta_deriv.
        """
        for nu in [2.5, 20.5, 100.5]:
            max_x = max(10.0 * nu, 1000.0)
            data = bessel_phase(nu, max_x)

            phase = data["phase"]
            min_x = data["min_x"]

            for x in np.linspace(2.0 * min_x + 1.0, 0.95 * max_x, 25):
                exact = (2.0 / np.pi) / (x * (jv(nu, x) ** 2 + yv(nu, x) ** 2))
                ours = phase.theta_deriv(x)

                relerr = fabs((ours - exact) / exact)
                self.assertTrue(
                    relerr < 1e-6,
                    f"theta_deriv nu={nu} at x={x:.5g}: ours={ours:.8g}, "
                    f"exact={exact:.8g}, relerr={relerr:.3g}",
                )

    def test_bessel_J_integral(self):
        MAX_INTEGRAL_RELERR = 1e-3

        def do_test(nu, min_x, max_x, expected):
            value = self._integrate_bessel_J(nu, min_x, max_x)
            self.assertTrue(
                fabs((value - expected) / expected) < MAX_INTEGRAL_RELERR,
                f"BesselJ integral nu={nu} min_x={min_x} max_x={max_x} (expected={expected}, value={value}, relerr={100.0 * (value - expected) / expected:.5g}%)",
            )

        do_test(3.0 / 2.0, 5.0, 10.0, -0.1927938424)
        do_test(3.0 / 2.0, 5.0, 100.0, -0.3011879458)

        do_test(5.0 / 2.0, 5.0, 10.0, -0.4502780732035633)
        do_test(5.0 / 2.0, 5.0, 100.0, -0.201367380942696)


if __name__ == "__main__":
    unittest.main()
