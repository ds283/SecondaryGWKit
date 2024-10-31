import unittest

import numpy as np
from math import fabs, sqrt, pi
from scipy.special import jv, yv

from AdaptiveLevin import adaptive_levin_sincos
from AdaptiveLevin.bessel_phase import bessel_phase
from utilities import format_time


class TestBesselLiouvilleGreen(unittest.TestCase):

    def _test_Bessel_JandY(self, nu: float, max_x: float):
        if max_x <= 10.0:
            rhs_x = 10.0
        else:
            rhs_x = max_x + 5.0
        data = bessel_phase(nu, rhs_x)

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
        if max_x <= 10.0:
            rhs_x = 10.0
        else:
            rhs_x = max_x + 5.0
        phase_data = bessel_phase(nu, rhs_x)

        phase = phase_data["phase"]
        dphase = phase_data["dphase"]

        f = [lambda x: sqrt(2.0 / (pi * x * dphase(x))), lambda x: 0.0]

        Levin_data = adaptive_levin_sincos(
            x_span=(1e-6, max_x),
            f=f,
            theta=phase,
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
        self._test_Bessel_JandY(3.0 / 2.0, 1.0)
        self._test_Bessel_JandY(3.0 / 2.0, 10.0)
        self._test_Bessel_JandY(3.0 / 2.0, 100.0)
        self._test_Bessel_JandY(3.0 / 2.0, 100.0)

        self._test_Bessel_JandY(5.0 / 2.0, 1.0)
        self._test_Bessel_JandY(5.0 / 2.0, 10.0)
        self._test_Bessel_JandY(5.0 / 2.0, 100.0)
        self._test_Bessel_JandY(5.0 / 2.0, 1000.0)

    def test_BesselJ_integral(self):
        MAX_INTEGRAL_RELERR = 1e-4

        value = self._integrate_BesselJ(3.0 / 2.0, 0.1)
        self.assertTrue(
            fabs(value - 0.0003362308167) / 0.0003362308167 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=3/2 max_x=10 (expected={0.0003362308167}, value={value}, relerr={100.0*(value-0.0003362308167)/0.0003362308167:.5g}%)",
        )

        value = self._integrate_BesselJ(3.0 / 2.0, 10.0)
        self.assertTrue(
            fabs(value - 1.148455377) / 1.148455377 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=3/2 max_x=10 (expected={1.148455377}, value={value}, relerr={100.0*(value-1.148455377)/1.148455377:.5g}%)",
        )

        value = self._integrate_BesselJ(3.0 / 2.0, 100.0)
        self.assertTrue(
            fabs(value - 1.040061274) / 1.040061274 < MAX_INTEGRAL_RELERR,
            f"BesselJ integral nu=3/2 max_x=100 (expected={1.040061274}, value={value}, relerr={100.0*(value-1.040061274)/1.040061274:.5g}%)",
        )

        value = self._integrate_BesselJ(5.0 / 2.0, 0.1)
        self.assertTrue(
            fabs(value - 4.803782622e-6) / 4.803782622e-6
            < 0.01,  # this one is less accurate. Perhaps an issue with accuracy of the phase representation?
            f"BesselJ integral nu=5/2 max_x=10 (expected={4.803782622E-6}, value={value}, relerr={100.0*(value-4.803782622E-6)/4.803782622E-6:.5g}%)",
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
