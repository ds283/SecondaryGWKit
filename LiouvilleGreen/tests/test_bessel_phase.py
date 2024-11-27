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
                f"BesselJ nu=3/2 at x={x:.5g} (phase={phase(x)}, our value={our_j:.5g}, their value={their_j:.5g}, relative error = {fabs((our_j - their_j) / their_j):.2%})",
            )
            self.assertTrue(
                fabs((our_y - their_y) / their_y) < REL_DIFF,
                f"BesselY nu=3/2 at x={x:.5g} (phase={phase(x)}, our value={our_y:.5g}, their value={their_y:.5g}, relative error = {fabs((our_y - their_y) / their_y):.2%})",
            )

    def _integrate_bessel_J(self, nu: float, min_x: float, max_x: float):
        if max_x <= 10.0:
            rhs_x = 10.0
        else:
            rhs_x = max_x + 5.0

        phase_data = bessel_phase(nu, rhs_x)

        phase = phase_data["phase"]
        mod = phase_data["mod"]

        f = [lambda x: mod(x), lambda x: 0.0]

        Levin_data = adaptive_levin_sincos(
            x_span=(min_x, max_x),
            f=f,
            theta=phase,
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
            print(f"  -- region: ({region[0]}, {region[1]})")

        return value

    def test_Bessel(self):
        self._test_bessel_value(3.0 / 2.0, 1000.0)
        self._test_bessel_value(5.0 / 2.0, 1000.0)

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
