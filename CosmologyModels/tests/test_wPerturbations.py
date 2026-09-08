"""
Regression tests for the perturbation equation-of-state parameter wPerturbations(z).

The author's convention (spec 01 Tier 3 block, spec 03 §0.5) is that wPerturbations(z)
is c_s^2 = delta p / delta rho of the *perturbed* fluid, with the cosmological constant
taken to be unperturbed. The denominator is therefore rho_matter + rho_radiation, with
rho_lambda excluded. `LambdaCDM.wPerturbations` has always done this;
`LambdaCDM_GenericEOS.wPerturbations` used to divide by the total density including
rho_lambda (audit `docs/spec-code-audit-2026-09.md` finding A1 / TK-1), which made c_s^2
too small by a factor 3.21 at z=0.

These tests pin the fixed behaviour:

  * a `LambdaCDM_GenericEOS` built on a pure-radiation equation of state, with the same
    Omegas and units, reproduces `LambdaCDM.wPerturbations`;
  * `wPerturbations` does not depend on Omega_lambda at fixed Omega_m, Omega_r (this is
    exactly the property the defect violated), while `wBackground` does;
  * the expected limits, and agreement with wBackground deep in radiation domination.

No Ray and no datastore is needed.
"""

import unittest
from math import pow

from CosmologyModels.GenericEOS.GenericEOS import GenericEOSBase
from CosmologyModels.GenericEOS.LambdaCDM_GenericEOS import LambdaCDM_GenericEOS
from CosmologyModels.LambdaCDM import LambdaCDM, Planck2018
from Units import Mpc_units

# An arbitrary type_id for the stand-in equation of state used here. It is never persisted
# (these tests do not touch the datastore); it only has to be a stable integer.
TEST_EOS_TYPE_ID = 900001

# The T(z) inversion inside LambdaCDM_GenericEOS is tabulated on a 500-point spline in
# log(1+z) spanning [DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT, max_z]. That spline, not the
# physics, sets the floor on how exactly the GenericEOS model can reproduce the closed-form
# LambdaCDM expressions: measured ~1.3e-9 relative with max_z = 1e4 and ~4e-7 with the
# default max_z = 1e20. 1e-8 is therefore the tightest robust threshold at max_z = 1e4, and
# is still seven orders of magnitude below the 0.69 relative discrepancy the A1 defect
# produced at z = 0.
AGREEMENT_RTOL = 1.0e-8

# redshift at which the T(z) spline is built for the "agreement" model
AGREEMENT_MAX_Z = 1.0e4

# a model reaching deep into radiation domination, for the limit tests
HIGH_Z_MAX_Z = 1.0e12


class PureRadiationEOS(GenericEOSBase):
    """
    Stand-in equation of state with constant, equal g_*(T) and g_{S,*}(T). GenericEOSBase.w()
    then returns 4 g_S / (3 g) - 1 = 1/3 identically, so the model is a pure-radiation fluid
    with no entropy production, and T(z) = T_CMB (1+z) exactly.

    The value of g is chosen to match LambdaCDM's photon + neutrino budget so that the two
    classes have identical rho_r0, hence identical Omega_r.
    """

    def __init__(self, units, g: float):
        super().__init__(units)
        self._g = g

    @property
    def name(self) -> str:
        return "pure radiation (test stub)"

    @property
    def type_id(self) -> int:
        return TEST_EOS_TYPE_ID

    def G(self, T: float) -> float:
        return self._g

    def Gs(self, T: float) -> float:
        return self._g


def lambdaCDM_gstar(Neff: float) -> float:
    """
    The effective number of bosonic degrees of freedom implied by LambdaCDM.__init__'s
    rho_r0: two photon polarisations plus Neff neutrino species, with the (4/11)^(4/3)
    reheating factor.
    """
    return 2.0 + 2.0 * (7.0 / 8.0) * Neff * pow(4.0 / 11.0, 4.0 / 3.0)


class _Params:
    """
    Minimal parameter block of the shape LambdaCDM / LambdaCDM_GenericEOS expect, so that
    Omega_lambda can be varied independently of Omega_m.
    """

    def __init__(self, source, omega_cc: float):
        self.name = f"{source.name} [omega_cc={omega_cc:.4g}]"
        self.omega_cc = omega_cc
        self.omega_m = source.omega_m
        self.f_baryon = source.f_baryon
        self.h = source.h
        self.T_CMB_Kelvin = source.T_CMB_Kelvin
        self.Neff = source.Neff


class TestWPerturbations(unittest.TestCase):
    """
    Construction of a LambdaCDM_GenericEOS is expensive (500 root solves for the T(z)
    spline), so the models are built once for the whole class.
    """

    @classmethod
    def setUpClass(cls):
        cls.units = Mpc_units()
        cls.params = Planck2018()
        cls.g = lambdaCDM_gstar(cls.params.Neff)

        cls.lcdm = LambdaCDM(store_id=1, units=cls.units, params=cls.params)
        cls.generic = LambdaCDM_GenericEOS(
            store_id=2,
            eos=PureRadiationEOS(cls.units, cls.g),
            units=cls.units,
            params=cls.params,
            max_z=AGREEMENT_MAX_Z,
        )

        # same Omega_m, same Omega_r, different Omega_lambda
        cls.generic_low_cc = LambdaCDM_GenericEOS(
            store_id=3,
            eos=PureRadiationEOS(cls.units, cls.g),
            units=cls.units,
            params=_Params(cls.params, omega_cc=0.4),
            max_z=AGREEMENT_MAX_Z,
        )

        cls.generic_high_z = LambdaCDM_GenericEOS(
            store_id=4,
            eos=PureRadiationEOS(cls.units, cls.g),
            units=cls.units,
            params=cls.params,
            max_z=HIGH_Z_MAX_Z,
        )

    def test_omega_r_matches(self):
        """
        The two classes must agree on Omega_r, otherwise the agreement test below would be
        comparing two different cosmologies.
        """
        self.assertAlmostEqual(
            self.generic.omega_r / self.lcdm.omega_r,
            1.0,
            delta=1.0e-14,
            msg="pure-radiation GenericEOS and LambdaCDM disagree on Omega_r",
        )

    def test_agrees_with_LambdaCDM(self):
        """
        A1 regression. With a pure-radiation equation of state the two classes describe the
        same fluid, so wPerturbations must agree. Before the fix the GenericEOS value was
        too small by a factor 3.21 at z=0, 1.28 at z=1.
        """
        for z in [0.0, 0.5, 1.0, 2.0, 10.0, 1.0e3]:
            with self.subTest(z=z):
                expected = self.lcdm.wPerturbations(z)
                actual = self.generic.wPerturbations(z)
                self.assertAlmostEqual(
                    actual / expected,
                    1.0,
                    delta=AGREEMENT_RTOL,
                    msg=f"wPerturbations mismatch at z={z:g}: "
                    f"LambdaCDM={expected:.12e}, GenericEOS={actual:.12e}",
                )

    def test_independent_of_omega_lambda(self):
        """
        With Lambda unperturbed, wPerturbations cannot depend on Omega_lambda. This is the
        property the defect violated: dividing by the total density made wPerturbations a
        function of rho_lambda.
        """
        for z in [0.0, 0.5, 1.0, 2.0, 10.0, 1.0e3]:
            with self.subTest(z=z):
                a = self.generic.wPerturbations(z)
                b = self.generic_low_cc.wPerturbations(z)
                self.assertAlmostEqual(
                    b / a,
                    1.0,
                    delta=1.0e-12,
                    msg=f"wPerturbations depends on Omega_lambda at z={z:g}: "
                    f"omega_cc=0.6889 gives {a:.12e}, omega_cc=0.4 gives {b:.12e}",
                )

    def test_wBackground_does_depend_on_omega_lambda(self):
        """
        The companion assertion: wBackground *does* include Lambda, so it must move when
        Omega_lambda moves. Without this a wPerturbations that ignored the density
        composition entirely would pass the test above.
        """
        for z in [0.0, 0.5, 1.0]:
            with self.subTest(z=z):
                a = self.generic.wBackground(z)
                b = self.generic_low_cc.wBackground(z)
                self.assertGreater(
                    abs(b / a - 1.0),
                    1.0e-3,
                    msg=f"wBackground unexpectedly insensitive to Omega_lambda at z={z:g}: "
                    f"{a:.12e} vs {b:.12e}",
                )

    def test_limits(self):
        """
        wPerturbations -> 1/3 deep in radiation domination, -> 0 (i.e. O(Omega_r)) today,
        and -> wBackground once Lambda is negligible.
        """
        deep = self.generic_high_z.wPerturbations(1.0e8)
        self.assertAlmostEqual(
            deep,
            1.0 / 3.0,
            delta=1.0e-4,
            msg=f"wPerturbations(z=1e8) = {deep:.12e}, expected ~1/3",
        )

        today = self.generic.wPerturbations(0.0)
        self.assertGreater(today, 0.0)
        self.assertLess(
            today,
            1.0e-4,
            msg=f"wPerturbations(z=0) = {today:.12e}, expected O(Omega_r) and positive",
        )

        for z in [1.0e3, 1.0e4]:
            with self.subTest(z=z):
                ratio = self.generic_high_z.wPerturbations(
                    z
                ) / self.generic_high_z.wBackground(z)
                self.assertAlmostEqual(
                    ratio,
                    1.0,
                    delta=1.0e-3,
                    msg=f"wPerturbations/wBackground = {ratio:.12e} at z={z:g}, "
                    "expected 1 to within 1e-3 once Lambda is negligible",
                )


if __name__ == "__main__":
    unittest.main()
