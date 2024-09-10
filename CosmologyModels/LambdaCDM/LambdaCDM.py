from collections import namedtuple
from math import sqrt

import ray

from CosmologyModels.base import CosmologyBase
from Units.base import UnitsLike
from constants import RadiationConstant

LambdaCDM_params = namedtuple(
    "LambdaCDM_params",
    ["name", "omega_cc", "omega_m", "f_baryon", "h", "T_CMB_Kelvin", "Neff"],
)


class LambdaCDM(CosmologyBase):
    def __init__(self, store_id: int, units: UnitsLike, params):
        """
        Construct a datastore-backed object representing a simple LambdaCDM cosmology
        :param store_id: unique Datastore id. Should not be None
        :param params: parameter block for the LambdaCDM model (e.g. Planck2018)
        :param units: units block (e.g. Mpc-based units)
        """
        CosmologyBase.__init__(self, store_id)

        self._params = params
        self._units = units

        # unpack details of the parameter block so we can access them without extensive nesting
        self._name = params.name

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

        self.rhom0 = Omega_factor * self.omega_m
        self.rhoCMB0 = RadiationConstant * T_CMB_4
        self.rhocc = Omega_factor * self.omega_cc

        self.omega_CMB = self.rhoCMB0 / Omega_factor

    @property
    def type_id(self) -> int:
        # 0 is the unique ID for the LambdaCDM cosmology type
        return 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def H0(self) -> float:
        return self._H0

    def build_storage_payload(self):
        return {
            "name": self.name,
            "omega_m": self.omega_m,
            "omega_cc": self.omega_cc,
            "h": self.h,
            "f_baryon": self.f_baryon,
            "T_CMB_Kelvin": self.T_CMB_Kelvin,
            "Neff": self.Neff,
        }

    def Hubble(self, z: float) -> float:
        """
        Evaluate the Hubble rate H(z) at the specified redshift z
        :param z: required redshift
        :return: value of H(z)
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        rhom = self.rhom0 * one_plus_z_3
        rhoCMB = self.rhoCMB0 * one_plus_z_4

        H0sq = (rhom + rhoCMB + self.rhocc) / (3.0 * self.Mpsq)
        return sqrt(H0sq)

    def d_lnH_dz(self, z: float) -> float:
        """
        Evaluate the logarithmic derivative d(ln H)/dz at the specified redshift z
        :param z: returned redshift
        :return: value of d(ln H)/dz
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z
        one_plus_z_4 = one_plus_z_2 * one_plus_z_2

        numerator = (
            9.0 * self.omega_m * one_plus_z_2 + 12.0 * self.omega_CMB * one_plus_z_3
        )
        denominator = (
            6.0 * self.omega_m * one_plus_z_3
            + 6.0 * self.omega_CMB * one_plus_z_4
            + 6.0 * self.omega_cc
        )

        return numerator / denominator
