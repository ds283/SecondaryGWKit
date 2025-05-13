from math import exp, sqrt, log
from typing import Mapping

import numpy as np
from numpy import linspace
from scipy.differentiate import derivative
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import root_scalar

from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyModels import BaseCosmology
from CosmologyModels.GenericEOS.GenericEOS import GenericEOSBase, HIGH_T_GSTAR
from Units.base import UnitsLike
from constants import RadiationConstant

DEFAULT_MAX_TEMPERATURE_Z_REDSHIFT = 1e20
DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2
DEFAULT_Z_DERIVATIVE_STEPSIZE = 0.05


class LambdaCDM_GenericEOS(BaseCosmology):
    """
    Construct a datastore
    """

    def __init__(
        self,
        store_id: int,
        eos: GenericEOSBase,
        units: UnitsLike,
        params,
        max_z: float = DEFAULT_MAX_TEMPERATURE_Z_REDSHIFT,
    ):
        BaseCosmology.__init__(self, store_id)

        self._params = params
        self._units = units
        self._eos = eos

        self._max_z = max_z

        # unpack details of the parameter block so we can access them without extensive nesting
        self._name = f"{eos.name} | {params.name}"

        # Omega factors are all measured today
        self.omega_cc = params.omega_cc
        self.omega_m = params.omega_m
        self.f_baryon = params.f_baryon
        self.h = params.h
        self.T_CMB_Kelvin = params.T_CMB_Kelvin

        # Neff not used here because it is baked into the G(T) and G_S(T) parametersa computed by the EOS object
        # self.Neff = params.Neff

        # derived dimensionful quantities, expressed in whatever system of units we require
        self._H0 = 100.0 * params.h * units.Kilometre / (units.Second * units.Mpc)
        self._T_CMB = params.T_CMB_Kelvin * units.Kelvin

        self.H0sq = self._H0 * self._H0
        self.Mpsq = units.PlanckMass * units.PlanckMass

        # POPULATE KEY DATA NOT PROVIDED AS PART OF THE PARAMS BLOCK

        T_CMB_2 = self._T_CMB * self._T_CMB
        T_CMB_4 = T_CMB_2 * T_CMB_2

        Omega_factor = 3.0 * self.H0sq * self.Mpsq

        self.rho_m0 = Omega_factor * self.omega_m
        # note the effective G* reported by the EOS object should have reheating
        # of the thermal bath relative to the neutrinos already included.
        # Therefore, we don't need the extra famous factor (4/11)^(4/3)
        self.rho_r0 = RadiationConstant * self._eos.G(self._T_CMB) * T_CMB_4
        self.rho_cc = Omega_factor * self.omega_cc

        self.omega_r = self.rho_r0 / Omega_factor

        # cache values of G(T_CMB), G_S(T_CMB), [G(T_CMB)]^(4/3) and [G_S(T_CMB)]^(4/3) which we need to use later
        self._G_CMB = eos.G(self._T_CMB)
        self._G_S_CMB = eos.Gs(self._T_CMB)
        self._G_CMB_pow13 = pow(self._G_CMB, 1.0 / 3.0)
        self._G_S_CMB_pow13 = pow(self._G_S_CMB, 1.0 / 3.0)

        # Build the spline used to map z to a temperature, given the specified equation of state.
        # We need to go all the way to z=0 so that we can compute the radiation temperature today
        # (needed to match to the CMB), but we need to compute a bit into the future (negative z)
        # so that we can accurately compute numerical derivatives at z=0
        self._T_z_spline = self._build_T_z_spline(
            min_z=DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT, max_z=max_z
        )

        # COMPUTE BASIC DATA ABOUT THIS COSMOLOGICAL MODEL

        rho_today = self.rho(0)
        gram_per_m3 = units.Gram / (units.Metre * units.Metre * units.Metre)
        rho_today_gram_m3 = rho_today / gram_per_m3

        # solve for epochs of matter/radiation and matter/Lambda equality
        matter_radiation_equality = self._find_rho_equality(
            "matter", "radiation", init_z=self.omega_m / self.omega_r - 1.0
        )
        matter_cc_equality = self._find_rho_equality(
            "matter",
            "lambda",
            init_z=pow(self.omega_cc / self.omega_m, 1.0 / 3.0) - 1.0,
        )

        print(f'@@ Parametrized equation-of-state LambdaCDM-like model "{self._name}"')
        print(f'|  equation of state = "{self._eos.name}", max_z = {self._max_z:.5g}')
        print(f"|  Omega_m = {self.omega_m:.4g}")
        print(f"|  Omega_cc = {self.omega_cc:.4g}")
        print(f"|  Omega_r = {self.omega_r:.4g}")
        print(f"|  present-day energy density = {rho_today_gram_m3:.4g} g/m^3")
        print(f"|  matter-radiation equality at z = {matter_radiation_equality:.4g}")
        print(f"|  matter-Lambda equality at z = {matter_cc_equality:.4g}")

    @property
    def type_id(self) -> int:
        # inherit our unique ID from the underlying choice of equation of state
        return self._eos.type_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def units(self) -> UnitsLike:
        return self._units

    @property
    def H0(self) -> float:
        return self._H0

    def T_z(self, z: float) -> float:
        return self._T_z_spline(z)

    def _solve_T_z(self, z: float) -> float:
        """
        Solve for T(z), the temperature as a function of redshift z
        :param z: redshift
        :return: temperature at this redshift, as a dimensionful quantity
        """

        # in the absence of entropy effects, T(z) a(z) = T_CMB a0, where a0 is the value of
        # the scale factor today, and T_CMB is the radiation temperature today. Then
        #   T(z) = T_CMB (a0/a) = T_CMB (1 + z)
        # With entropy effects included, this scaling is no longer exact. Instead, T(z)
        # should solve the implicit equation
        #   T(z) [G_S(T(z))]^(1/3) = T_CMB [G_S(T_CMB)]^(1/3) (1 + z)

        # a good initial guess for T(z) is given by simple redshfting, without including factors
        # of G_S(T). We always work with dimensionful values of T.
        # This initial guess will be an overestimate, because G_S(T) >= G_S(T_CMB)
        bracket_hi = 1.05 * self._T_CMB * (1.0 + z)

        # meanwhile, the largest G_S(T) can be is given by its asymptotic value
        bracket_lo = 0.95 * self._T_CMB * (1.0 + z) / pow(HIGH_T_GSTAR, 1.0 / 3.0)

        target = self._T_CMB * self._G_S_CMB_pow13 * (1.0 + z)

        def T_equation(T: float) -> float:
            G_S = self._eos.Gs(T)
            G_S_pow13 = pow(G_S, 1.0 / 3.0)
            return T * G_S_pow13 - target

        if T_equation(bracket_lo) * T_equation(bracket_hi) >= 0.0:
            raise RuntimeError(
                f"Could not bracket target temperature T(z) at z={z:.4g}, bracket_lo={bracket_lo:.5g}, bracket_hi={bracket_hi:.5g}"
            )

        root = root_scalar(
            T_equation, bracket=(bracket_lo, bracket_hi), xtol=1e-6, rtol=1e-4
        )

        if not root.converged:
            raise RuntimeError(
                f'root_scalar() did not converge to a solution: x_bracket=({bracket_lo:.5g}, {bracket_hi:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
            )

        return root.root

    def _build_T_z_spline(
        self, min_z: float, max_z: float, samples: int = 500
    ) -> ZSplineWrapper:
        # add a 5% buffer to the min/max z range
        min_z = 0.95 * min_z
        max_z = 1.05 * max_z

        log_z_values = linspace(log(1.0 + min_z), log(1.0 + max_z), samples)
        T_values = [self._solve_T_z(exp(logz) - 1.0) for logz in log_z_values]

        spline = InterpolatedUnivariateSpline(log_z_values, T_values, ext="raise")
        return ZSplineWrapper(
            spline,
            label="T(z)",
            min_z=min_z,
            max_z=max_z,
            log_z=True,
        )

    def _rho_fluid(self, z: float) -> Mapping[str, float]:
        """
        Determine the densities of the matter, radiation (etc.) fluids at redshift z
        :param z:
        :return:
        """
        one_plus_z = 1.0 + z

        one_plus_z_2 = one_plus_z * one_plus_z
        one_plus_z_3 = one_plus_z_2 * one_plus_z

        T: float = self._T_z_spline(z)
        T_2 = T * T
        T_4 = T_2 * T_2

        rho_m = self.rho_m0 * one_plus_z_3

        # reheating of the thermal bath due to annihilations, and splitting of the photon and
        # neutrino temperatures at low redshift, should be included already in the EOS object
        rho_r = RadiationConstant * self._eos.G(T) * T_4

        return {
            "T": T,
            "matter": rho_m,
            "radiation": rho_r,
            "lambda": self.rho_cc,
        }

    def _find_rho_equality(
        self, species_A: str, species_B: str, init_z: float
    ) -> float:
        """
        Determine the redshift at which the energy density in species A equals the energy density in species B
        :param species_A:
        :param species_B:
        :return:
        """

        def match_rho(z: float) -> float:
            rho = self._rho_fluid(z)
            return rho[species_A] - rho[species_B]

        root = root_scalar(match_rho, x0=init_z, xtol=1e-6, rtol=1e-4)

        if not root.converged:
            raise RuntimeError(
                f'root_scalar() did not converge to a solution: iterations={root.iterations}, method={root.method}: "{root.flag}"'
            )

        return root.root

    def rho(self, z: float) -> float:
        """
        Determine the total matter density at redshift z
        :param z:
        :return:
        """
        rho = self._rho_fluid(z)
        return rho["matter"] + rho["radiation"] + rho["lambda"]

    def Hubble(self, z: float) -> float:
        """
        Evaluate the Hubble rate H(z) at the specified redshift z
        :param z: required redshift
        :return: value of H(z)
        """
        rho_total = self.rho(z)
        H0sq = rho_total / (3.0 * self.Mpsq)
        return sqrt(H0sq)

    def d_lnH_dz(self, z0: float) -> float:
        """
        Evaluate the logarithmic derivative d(ln H)/dz at the specified redshift z0
        :param z0: required redshift
        :return: value of d(ln H)/dz
        """

        f = np.vectorize(pyfunc=lambda z: log(self.Hubble(z)))

        def ln_H(z_array: np.ndarray) -> np.ndarray:
            return f(z_array)

        data = derivative(
            f=ln_H,
            x=[z0],
            tolerances={"atol": 1e-6, "rtol": 1e-4},
            initial_step=DEFAULT_Z_DERIVATIVE_STEPSIZE,
        )

        if not data.success:
            raise RuntimeError(
                f"derivative() failed: {data.message} (status={data.status}, iterations={data.nit}, nfev={data.nfev}, error={data.error})"
            )

        return data.df.item()

    def d2_lnH_dz2(self, z0: float) -> float:
        """
        Evaluate the logarithmic derivative d^2(ln H)/dz^2 at the specified redshift z0
        :param z0: returned redshift
        :return: value of d^2(ln H)/dz^2
        """

        f = np.vectorize(pyfunc=lambda z: self.d_lnH_dz(z))

        def d_ln_H(z_array: np.ndarray) -> np.ndarray:
            return f(z_array)

        data = derivative(
            f=d_ln_H,
            x=[z0],
            tolerances={"atol": 1e-6, "rtol": 1e-4},
            initial_step=DEFAULT_Z_DERIVATIVE_STEPSIZE,
        )

        if not data.success:
            raise RuntimeError(
                f"derivative() failed: {data.message} (status={data.status}, iterations={data.nit}, nfev={data.nfev}, error={data.error})"
            )

        return data.df.item()

    def d3_lnH_dz3(self, z0: float) -> float:
        """
        Evaluate the logarithmic derivative d^3(ln H)/dz^3 at the specified redshift z0
        :param z0: returned redshift
        :return: value of d^3(ln H)/dz^3
        """

        f = np.vectorize(pyfunc=lambda z: self.d2_lnH_dz2(z))

        def d2_ln_H(z_array: np.ndarray) -> np.ndarray:
            return f(z_array)

        data = derivative(
            f=d2_ln_H,
            x=[z0],
            tolerances={"atol": 1e-6, "rtol": 1e-4},
            initial_step=DEFAULT_Z_DERIVATIVE_STEPSIZE,
        )

        if not data.success:
            raise RuntimeError(
                f"derivative() failed: {data.message} (status={data.status}, iterations={data.nit}, nfev={data.nfev}, error={data.error})"
            )

        return data.df.item()

    def wBackground(self, z: float) -> float:
        rho = self._rho_fluid(z)

        T = rho["T"]

        # background w(z) includes contributions from radiation, cosmological constant, and matter
        # (but matter has w=0 and drops out)
        numerator = self._eos.w(T) * rho["radiation"] + (-1.0) * rho["lambda"]
        denominator = self.rho(z)

        return numerator / denominator

    def wPerturbations(self, z: float) -> float:
        rho = self._rho_fluid(z)
        T = rho["T"]

        # perturbations w(z) includes contributions from radiation and matter, but not the cosmological constant,
        # which we take not to have perturbations. (Possibly we shouldn't do that, but instead allow the cosmological
        # constant to cluster with c_s=1?)
        # As for the background, matter has w=0 and drops out.
        numerator = self._eos.w(T) * rho["radiation"]
        denominator = self.rho(z)

        return numerator / denominator

    def d_wPerturbations_dz(self, z0: float) -> float:
        """
        Evaluate the derivative dw/dz at the specified redshift z0
        :param z0: returned redshift
        :return: value of dw/dz
        """

        f = np.vectorize(pyfunc=lambda z: self.wPerturbations(z))

        def w(z_array: np.ndarray) -> np.ndarray:
            return f(z_array)

        data = derivative(
            f=w,
            x=[z0],
            tolerances={"atol": 1e-6, "rtol": 1e-4},
            initial_step=DEFAULT_Z_DERIVATIVE_STEPSIZE,
        )

        if not data.success:
            raise RuntimeError(
                f"derivative() failed: {data.message} (status={data.status}, iterations={data.nit}, nfev={data.nfev}, error={data.error})"
            )

        return data.df.item()

    def d2_wPerturbations_dz2(self, z0: float) -> float:
        """
        Evaluate the derivative d^2w/dz^2 at the specified redshift z0
        :param z0: returned redshift
        :return: value of d^2w/dz^2
        """

        f = np.vectorize(pyfunc=lambda z: self.d_wPerturbations_dz(z))

        def d_w(z_array: np.ndarray) -> np.ndarray:
            return f(z_array)

        data = derivative(
            f=d_w,
            x=[z0],
            tolerances={"atol": 1e-6, "rtol": 1e-4},
            initial_step=DEFAULT_Z_DERIVATIVE_STEPSIZE,
        )

        if not data.success:
            raise RuntimeError(
                f"derivative() failed: {data.message} (status={data.status}, iterations={data.nit}, nfev={data.nfev}, error={data.error})"
            )

        return data.df.item()
