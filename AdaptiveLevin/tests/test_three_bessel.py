import json
import unittest

from math import sqrt, pi

from AdaptiveLevin import adaptive_levin_sincos
from AdaptiveLevin.bessel_phase import bessel_phase
from ComputeTargets.QuadSourceIntegral import _three_bessel_integrals
from CosmologyConcepts import wavenumber
from Units import Mpc_units
from utilities import format_time


class TestBessel(unittest.TestCase):

    atol = 1e-23
    rtol = 1e-8

    sqrt_2_over_pi = sqrt(2.0 / pi)

    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)

        self.units = Mpc_units

        # self.r = wavenumber(store_id=3, k_inv_Mpc=79.37005259841, units=self.units)
        # self.q = wavenumber(store_id=2, k_inv_Mpc=79.37005259841, units=self.units)
        # self.k = wavenumber(store_id=1, k_inv_Mpc=49999999.99999999, units=self.units)

        self.r = wavenumber(store_id=3, k_inv_Mpc=49999999.99999999, units=self.units)
        self.q = wavenumber(store_id=2, k_inv_Mpc=49999999.99999999, units=self.units)
        self.k = wavenumber(store_id=1, k_inv_Mpc=11071.731788899118, units=self.units)

        self.b = 0
        self.cs_sq = (1.0 - self.b) / (1.0 + self.b) / 3.0

        self.min_eta = 1.5713009646109173e-10
        self.max_eta = 2.2066e-06
        # self.max_eta = 7.196808204366744e-06
        # self.max_eta = 5.05340787180641e-07
        # self.max_eta = 1e-5
        # self.max_eta = 1e-3

        self.max_x = max(self.k.k, self.q.k, self.r.k) * self.max_eta
        self.min_x = min(self.k.k, self.q.k, self.r.k) * self.min_eta * sqrt(self.cs_sq)

        self.phase_data_0pt5 = bessel_phase(0.5 + self.b, self.max_x + 10.0)
        self.phase_data_2pt5 = bessel_phase(2.5 + self.b, self.max_x + 10.0)

    def test_integrals(self):
        phase = self.phase_data_0pt5["phase"]
        dphase = self.phase_data_0pt5["dphase"]

        self._test_integrals(100.0, phase, dphase, label="0.5")
        self._test_integrals(1.0, phase, dphase, label="0.5")

        phase = self.phase_data_2pt5["phase"]
        dphase = self.phase_data_2pt5["dphase"]

        self._test_integrals(100.0, phase, dphase, label="2.5")
        self._test_integrals(1.0, phase, dphase, label="2.5")

    def _test_integrals(self, min_x, phase, dphase, label):
        def Levin_f(x):
            return self.sqrt_2_over_pi / sqrt(x * dphase(x))

        # J integral order 0.5
        J_data = adaptive_levin_sincos(
            x_span=(min_x, self.max_x),
            f=[Levin_f, lambda x: 0.0],
            theta=phase,
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
            theta=phase,
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
        data0pt5_Levin = _three_bessel_integrals(
            self.k,
            self.q,
            self.r,
            min_eta=self.min_eta,
            max_eta=self.max_eta,
            b=self.b,
            phase_data={"0pt5": self.phase_data_0pt5, "2pt5": self.phase_data_2pt5},
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
            phase_data={"0pt5": self.phase_data_0pt5, "2pt5": self.phase_data_2pt5},
            nu_type="2pt5",
            atol=self.atol,
            rtol=self.rtol,
        )

        print(f"\n** 0pt5 integral data")
        print(json.dumps(data0pt5_Levin, indent=4, sort_keys=True))

        print(f"\n** 2pt5 integral data")
        print(json.dumps(data2pt5_Levin, indent=4, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
