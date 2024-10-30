import unittest

import numpy as np
from math import fabs, sqrt, pi
from scipy.special import jv, yv

from AdaptiveLevin import adaptive_levin_sincos
from AdaptiveLevin.bessel_phase import bessel_phase
from utilities import format_time


class TestBesselLiouvilleGreen(unittest.TestCase):

    def _test_Bessel_JandY(self, nu: float, max_x: float):
        data = bessel_phase(nu, max_x)

        bessel_j = data["bessel_j"]
        bessel_y = data["bessel_y"]

        test_grid = np.linspace(1e-2, max_x, 100)
        for x in test_grid:
            our_j = bessel_j(x)
            our_y = bessel_y(x)

            their_j = jv(nu, x)
            their_y = yv(nu, x)

            REL_DIFF = 1e-10
            self.assertTrue(
                fabs((our_j - their_j) / their_j) > REL_DIFF,
                f"BesselJ nu=3/2 at x={x:.5g} (our value={our_j:.5g}, their value={their_j:.5g}, relative error = {fabs((our_j - their_j) / their_j):.2%})",
            )
            self.assertTrue(
                fabs((our_y - their_y) / their_y) > REL_DIFF,
                f"BesselY nu=3/2 at x={x:.5g} (our value={our_y:.5g}, their value={their_y:.5g}, relative error = {fabs((our_y - their_y) / their_y):.2%})",
            )

    def _integrate_BesselJ(self, nu: float, max_x: float):
        phase_data = bessel_phase(nu, max_x)

        phase = phase_data["phase"]
        dphase = phase_data["dphase"]

        f = [lambda x: sqrt(2.0 / (pi * x * dphase(x))), lambda x: 0.0]
        theta = phase

        Levin_data = adaptive_levin_sincos(
            (1e-4, max_x),
            f,
            theta,
            tol=1e-10,
            chebyshev_order=12,
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
        self._test_Bessel_JandY(3.0 / 2.0, 10.0)
        self._test_Bessel_JandY(3.0 / 2.0, 100.0)
        self._test_Bessel_JandY(3.0 / 2.0, 100.0)

        self._test_Bessel_JandY(5.0 / 2.0, 10.0)
        self._test_Bessel_JandY(5.0 / 2.0, 100.0)
        self._test_Bessel_JandY(5.0 / 2.0, 1000.0)

    def test_BesselJ_integral(self):
        value = self._integrate_BesselJ(3.0 / 2.0, 10.0)
        MAX_INTEGRAL_RELERR = 1e-4
        self.assertTrue(
            fabs(value - 1.148455377) / 1.148455377 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=3/2 max_x=10 (expected={1.148455377}, value={value}, relerr={100.0*(value-1.148455377)/1.148455377:.5g}%)",
        )

        value = self._integrate_BesselJ(3.0 / 2.0, 100.0)
        self.assertTrue(
            fabs(value - 1.040061274) / 1.040061274 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=3/2 max_x=100 (expected={1.040061274}, value={value}, relerr={100.0*(value-1.040061274)/1.040061274:.5g}%)",
        )

        value = self._integrate_BesselJ(5.0 / 2.0, 10.0)
        self.assertTrue(
            fabs(value - 0.8209075325) / 0.8209075325 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=5/2 max_x=10 (expected={0.8209075325}, value={value}, relerr={100.0*(value-0.8209075325)/0.8209075325:.5g}%)",
        )

        value = self._integrate_BesselJ(5.0 / 2.0, 100.0)
        self.assertTrue(
            fabs(value - 1.069818225) / 1.069818225 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=5/2 max_x=100 (expected={1.069818225}, value={value}, relerr={100.0*(value-1.069818225)/1.069818225:.5g}%)",
        )


if __name__ == "__main__":
    unittest.main()
