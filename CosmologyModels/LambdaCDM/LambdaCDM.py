from collections import namedtuple
from math import sqrt, pow

from CosmologyModels import BaseCosmology
from CosmologyModels.model_ids import LAMBDACDM_IDENTIFIER
from Units.base import UnitsLike
from constants import RadiationConstant

LambdaCDM_params = namedtuple(
    "LambdaCDM_params",
    ["name", "omega_cc", "omega_m", "f_baryon", "h", "T_CMB_Kelvin", "Neff"],
)


class LambdaCDM(BaseCosmology):
    def __init__(self, store_id: int, units: UnitsLike, params):
        """
        Construct a datastore-backed object representing a simple LambdaCDM cosmology
        :param store_id: unique Datastore id. Should not be None
        :param params: parameter block for the LambdaCDM model (e.g. Planck2018)
        :param units: units block (e.g. Mpc-based units)
        """
        BaseCosmology.__init__(self, store_id)

        self._params = params
        self._units = units

        # unpack details of the parameter block so we can access them without extensive nesting
        self._name = params.name

        # Omega factors are all measured today
        self.omega_cc = params.omega_cc
        self.omega_m = params.omega_m
        self.f_baryon = params.f_baryon
        self.h = params.h
        self.T_CMB_Kelvin = params.T_CMB_Kelvin
        self.Neff = params.Neff

        # derived dimensionful quantities, expressed in whatever system of units we require
        self._H0 = 100.0 * params.h * units.Kilometre / (units.Second * units.Mpc)
        self._T_CMB = params.T_CMB_Kelvin * units.Kelvin

        self.H0sq = self._H0 * self._H0
        self.Mpsq = units.PlanckMass * units.PlanckMass

        T_CMB_2 = self._T_CMB * self._T_CMB
        T_CMB_4 = T_CMB_2 * T_CMB_2

        Omega_factor = 3.0 * self.H0sq * self.Mpsq

        self.rho_m0 = Omega_factor * self.omega_m
        # dof calculation = 1 x RadiationConstant to get energy density for a single bosonic polarization
        #   2 polarizations of the photon
        #   N_eff effective neutrino species (the Standard Model value is 3.046, accounting for small QED effects,
        #   neutrinos not being completely decoupled during electron-positron annihilation (so they carry away some entropy)
        #   and a few other misc small effects)
        #   2 polarizations of each fermion, plus fermions contribution like 7/8 of a boson
        #   famous factor of (11/4)^(1/3) to account for reheating of the photons during electron-positron
        #   annihilation
        self.rho_r0 = (
            RadiationConstant
            * (2.0 + 2.0 * (7.0 / 8.0) * self.Neff * pow(4.0 / 11.0, 4.0 / 3.0))
            * T_CMB_4
        )
        self.rho_cc = Omega_factor * self.omega_cc

        self.omega_r = self.rho_r0 / Omega_factor

        rho_today = self.rho(0)
        gram_per_m3 = units.Gram / (units.Metre * units.Metre * units.Metre)
        rho_today_gram_m3 = rho_today / gram_per_m3

        matter_radiation_equality = self.omega_m / self.omega_r - 1.0
        matter_cc_equality = pow(self.omega_cc / self.omega_m, 1.0 / 3.0) - 1.0

        print(f'@@ LambdaCDM model "{self._name}"')
        print(f"|  Omega_m = {self.omega_m:.4g}")
        print(f"|  Omega_cc = {self.omega_cc:.4g}")
        print(f"|  Omega_r = {self.omega_r:.4g}")
        print(f"|  present-day energy density = {rho_today_gram_m3:.4g} g/m^3")
        print(f"|  matter-radiation equality at z = {matter_radiation_equality:.4g}")
        print(f"|  matter-Lambda equality at z = {matter_cc_equality:.4g}")

    @property
    def type_id(self) -> int:
        # return the unique ID for the LambdaCDM cosmology type
        return LAMBDACDM_IDENTIFIER

    @property
    def name(self) -> str:
        return self._name

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def H0(self) -> float:
        return self._H0

    def rho(self, z: float) -> float:
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        rho_m = self.rho_m0 * one_plus_z_3
        rho_r = self.rho_r0 * one_plus_z_4

        return rho_m + rho_r + self.rho_cc

    def Hubble(self, z: float) -> float:
        """
        Evaluate the Hubble rate H(z) at the specified redshift z
        :param z: required redshift
        :return: value of H(z)
        """
        rho_total = self.rho(z)
        Hsq = rho_total / (3.0 * self.Mpsq)
        return sqrt(Hsq)

    def d_lnH_dz(self, z: float) -> float:
        """
        Evaluate the logarithmic derivative d(ln H)/dz at the specified redshift z
        :param z: required redshift
        :return: value of d(ln H)/dz
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        numerator = (
            3.0 * self.omega_m * one_plus_z_2 + 4.0 * self.omega_r * one_plus_z_3
        )
        denominator = 2.0 * (
            self.omega_m * one_plus_z_3 + self.omega_r * one_plus_z_4 + self.omega_cc
        )

        return numerator / denominator

    def d2_lnH_dz2(self, z: float) -> float:
        """
        Evaluate the logarithmic derivative d^2(ln H)/dz^2 at the specified redshift z
        :param z: required redshift
        :return: value of d^2(ln H)/dz^2
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        numerator = 6.0 * self.omega_m * one_plus_z + 12.0 * self.omega_r * one_plus_z_2
        denominator = 2.0 * (
            self.omega_m * one_plus_z_3 + self.omega_r * one_plus_z_4 + self.omega_cc
        )

        d_lnH_dz = self.d_lnH_dz(z)

        return numerator / denominator - 2.0 * d_lnH_dz * d_lnH_dz

    def d3_lnH_dz3(self, z: float) -> float:
        """
        Evaluate the logarithmic derivative d^3(ln H)/dz^3 at the specified redshift z
        :param z: required redshift
        :return: value of d^3(ln H)/dz^3
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        numerator = 6.0 * self.omega_m + 24.0 * self.omega_r * one_plus_z
        denominator = 2.0 * (
            self.omega_m * one_plus_z_3 + self.omega_r * one_plus_z_4 + self.omega_cc
        )

        d_lnH_dz = self.d_lnH_dz(z)
        d2_lnH_dz2 = self.d2_lnH_dz2(z)

        return (
            numerator / denominator
            - 6.0 * d_lnH_dz * d2_lnH_dz2
            - 4.0 * d_lnH_dz * d_lnH_dz * d_lnH_dz
        )

    def wBackground(self, z: float) -> float:
        w_rad = 1.0 / 3.0

        one_plus_z = 1.0 + z
        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        # discard w_matter contribution to the numerator, which is proportional to zero
        numerator = w_rad * self.omega_r * one_plus_z_4 - self.omega_cc
        denominator = (
            self.omega_m * one_plus_z_3 + self.omega_r * one_plus_z_4 + self.omega_cc
        )

        return numerator / denominator

    def wPerturbations(self, z: float) -> float:
        w_rad = 1.0 / 3.0

        one_plus_z = 1.0 + z

        # discard w_matter contribution to the numerator, which is proportional to zero
        numerator = w_rad * self.omega_r * one_plus_z
        denominator = self.omega_m + self.omega_r * one_plus_z

        return numerator / denominator

    def d_wPerturbations_dz(self, z: float) -> float:
        """
        Evaluate the derivative dw/dz at the specified redshift z0
        :param z0: returned redshift
        :return: value of dw/dz
        """
        w_rad = 1.0 / 3.0

        one_plus_z = 1.0 + z

        numerator = w_rad * self.omega_r * self.omega_m
        denominator = self.omega_m + self.omega_r * one_plus_z

        return numerator / (denominator * denominator)

    def d2_wPerturbations_dz2(self, z: float) -> float:
        """
        Evaluate the derivative d^2w/dz^2 at the specified redshift z0
        :param z0: returned redshift
        :return: value of d^2w/dz^2
        """
        w_rad = 1.0 / 3.0

        one_plus_z = 1.0 + z

        numerator = -2.0 * w_rad * self.omega_r * self.omega_r * self.omega_m
        denominator = self.omega_m + self.omega_r * one_plus_z

        return numerator / (denominator * denominator * denominator)
