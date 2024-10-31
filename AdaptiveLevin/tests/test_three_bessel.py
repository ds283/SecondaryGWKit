import unittest

from math import sqrt

from AdaptiveLevin.bessel_phase import bessel_phase
from ComputeTargets.QuadSourceIntegral import _three_bessel_integrals
from CosmologyConcepts import wavenumber
from Units import Mpc_units


class TestBessel(unittest.TestCase):

    def test_three_bessel(self):
        # k=79.4/Mpc, q=8.57/Mpc, r=5.83e+05/Mpc
        units = Mpc_units

        k = wavenumber(store_id=1, k_inv_Mpc=79.4, units=units)
        q = wavenumber(store_id=2, k_inv_Mpc=8.57, units=units)
        r = wavenumber(store_id=3, k_inv_Mpc=5.83e05, units=units)

        min_eta = 1e-5
        max_eta = 1e-3
        b = 0

        cs_sq = (1.0 - b) / (1.0 + b) / 3.0

        min_x = min(k.k, q.k, r.k) * min_eta * sqrt(cs_sq)
        max_x = max(k.k, q.k, r.k) * max_eta

        print(f"min_x={min_x:.5g}, max_x={max_x:.5g}")

        phase_data_0pt5 = bessel_phase(0.5 + b, max_x + 10.0)
        phase_data_2pt5 = bessel_phase(2.5 + b, max_x + 10.0)

        J0pt5, Y0pt5 = _three_bessel_integrals(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            phase_data={"0pt5": phase_data_0pt5, "2pt5": phase_data_2pt5},
            nu_type="0pt5",
            tol=1e-7,
        )

        print(f"** J_0pt5 = {J0pt5:.7g}, Y_0pt5 = {Y0pt5:.7g}")

        J2pt5, Y2pt5 = _three_bessel_integrals(
            k,
            q,
            r,
            min_eta=min_eta,
            max_eta=max_eta,
            b=b,
            phase_data={"0pt5": phase_data_0pt5, "2pt5": phase_data_2pt5},
            nu_type="2pt5",
            tol=1e-7,
        )

        print(f"** J_2pt5 = {J2pt5:.7g}, Y_2pt5 = {Y2pt5:.7g}")


if __name__ == "__main__":
    unittest.main()
