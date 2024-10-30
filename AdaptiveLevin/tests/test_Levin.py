import unittest

from math import fabs, atan, sin, pi

from AdaptiveLevin.levin_quadrature import adaptive_levin_sincos
from utilities import format_time


class TestAdaptiveLevinSinCos(unittest.TestCase):

    def test_SincIntegral(self):
        x_span = (1.0, 100.0)

        # sinc integral is sin(x)/x, so sin weight is 1/x and cos weight is 0
        f = [lambda x: 1.0 / x, lambda x: 0.0]

        # take theta to be 100x
        theta = lambda x: 100.0 * x

        data = adaptive_levin_sincos(
            x_span,
            f,
            theta,
            tol=1e-10,
            chebyshev_order=12,
        )
        value = data["value"]
        regions = data["regions"]
        evaluations = data["evaluations"]
        elapsed = data["elapsed"]
        print(
            f"integral_1^100 sin(100x)/x = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: ({region[0]}, {region[1]})")
        self.assertTrue(
            fabs(value - 0.00866607847) < 1e-10,
            f"Sinc integral test failed: expected {0.00866607847}, obtained {value}",
        )

    def _GRZIntegral(self, lbda):

        x_span = (-1.0, 1.0)

        # integral is cos(lambda arctan(x)) / (1 + x^2)
        f = [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)]
        theta = lambda x: lbda * atan(x)

        data = adaptive_levin_sincos(
            x_span,
            f,
            theta,
            tol=1e-10,
            chebyshev_order=12,
        )
        value = data["value"]
        regions = data["regions"]
        evaluations = data["evaluations"]
        elapsed = data["elapsed"]
        print(
            f"integral_(-1)^(+1) cos({lbda} arctan(x)) / (1 + x^2) = {value} ({len(regions)} regions, {evaluations} evaluations in time {format_time(elapsed)})"
        )
        for region in regions:
            print(f"  -- region: ({region[0]}, {region[1]})")

        analytic_sol = (2.0 / lbda) * sin(pi * lbda / 4.0)
        self.assertTrue(
            fabs(value - analytic_sol) < 1e-10,
            f"Gradshtein & Ryzhik integral test: expected {analytic_sol}, obtained {value}",
        )

    def test_GRZIntegral(self):
        self._GRZIntegral(10.0)
        self._GRZIntegral(100.0)
        self._GRZIntegral(1000.0)


if __name__ == "__main__":
    unittest.main()
