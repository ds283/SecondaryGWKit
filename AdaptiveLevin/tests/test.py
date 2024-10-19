import unittest

from math import fabs

from AdaptiveLevin.levin_quadrature import adaptive_levin_sincos


class TestAdaptiveLevinSinCos(unittest.TestCase):

    def test_SincIntegral(self):
        x_span = (1.0, 100.0)

        # sinc integral is sin(x)/x, so sin weight is 1/x and cos weight is 0
        f = [lambda x: 1.0 / x, lambda x: 0.0]

        # take theta to be 100x
        theta = lambda x: 100.0 * x

        value, regions, evaluations = adaptive_levin_sincos(
            x_span, f, theta, tol=1e-10, chebyshev_order=12
        )
        print(
            f"integral_1^100 sin(100x)/x = {value} ({len(regions)} regions, {evaluations} evaluations)"
        )
        for region in regions:
            print(f"  -- region: ({region[0]}, {region[1]})")
        self.assertTrue(
            fabs(value - 0.00866607847) < 1e-10, "Sinc integral test failed"
        )


if __name__ == "__main__":
    unittest.main()
