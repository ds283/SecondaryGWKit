from math import exp, log, pow
from typing import Mapping

from CosmologyModels.GenericEOS.GenericEOS import (
    GenericEOSBase,
    HIGH_T_GSTAR,
    LOW_T_G_S_STAR,
    LOW_T_GSTAR,
)
from CosmologyModels.model_ids import QCD_EOS_IDENTIFIER
from Units.base import UnitsLike

# Fitting functions for g and g_s from Appendix C of Saikawa & Shirai https://arxiv.org/pdf/1803.01038

# Fitting functions needed for g(T), taken from Table 1 (p.48) of 1803.01038 v2
a_coeffs = {
    0: 1.0,
    1: 1.11724,
    2: 3.12672e-1,
    3: -4.68049e-2,
    4: -2.65004e-2,
    5: -1.19760e-3,
    6: 1.82812e-4,
    7: 1.36436e-4,
    8: 8.55051e-5,
    9: 1.2284e-5,
    10: 3.82259e-7,
    11: -6.87035e-9,
}

b_coeffs = {
    0: 1.43382e-2,
    1: 1.37559e-2,
    2: 2.92108e-3,
    3: -5.38533e-4,
    4: -1.62496e-4,
    5: -2.87906e-5,
    6: -3.84278e-6,
    7: 2.78776e-6,
    8: 7.40342e-7,
    9: 1.1721e-7,
    10: 3.72499e-9,
    11: -6.74107e-11,
}

# Fitting functions needed for g_S(T), also taken from Table 1 (p.48) of 1803.01038 v2
c_coeffs = {
    0: 1.0,
    1: 6.07869e-1,
    2: -1.54485e-1,
    3: -2.24034e-1,
    4: -2.82147e-2,
    5: 2.9062e-2,
    6: 6.86778e-3,
    7: -1.00005e-3,
    8: -1.69104e-4,
    9: 1.06301e-5,
    10: 1.69528e-6,
    11: -9.33311e-8,
}

d_coeffs = {
    0: 7.07388e1,
    1: 9.18011e1,
    2: 3.31892e1,
    3: -1.39779,
    4: -1.52558,
    5: -1.97857e-2,
    6: -1.60146e-1,
    7: 8.22615e-5,
    8: 2.02651e-2,
    9: -1.82134e-5,
    10: 7.83943e-5,
    11: 7.13518e-5,
}

# Particle masses in GeV, needed fpr fitting function below 120 MeV
M_e = 511e-6  # 511 keV
M_mu = 0.1056
M_pi0 = 0.135
M_piplus = M_piminus = 0.140
M_1 = 0.5
M_2 = 0.77
M_3 = 1.2
M_4 = 2.0


def polynomial_sum(coeffs: Mapping[int, float], x: float) -> float:
    """Compute sum of polynomial terms"""
    return sum(c * pow(x, i) for i, c in coeffs.items())


# Fitting functions for energy density
def f_rho(x: float) -> float:
    """Low-temperature fitting function for fermion energy density"""
    return exp(-1.04855 * x) * (
        1.0 + 1.03757 * x + 0.508630 * (x * x) + 0.0893988 * (x * x * x)
    )


def b_rho(x: float) -> float:
    """Low-temperature fitting function for boson energy density"""
    return exp(-1.03149 * x) * (
        1.0 + 1.03317 * x + 0.398264 * (x * x) + 0.0648056 * (x * x * x)
    )


# Fitting functions for entropy density
def f_s(x: float) -> float:
    """Low-temperature fitting function for fermion entropy"""
    return exp(-1.04190 * x) * (
        1.0 + 1.03400 * x + 0.456426 * (x * x) + 0.0595249 * (x * x * x)
    )


def b_s(x: float) -> float:
    """Low-temperature fitting function for boson entropy"""
    return exp(-1.03365 * x) * (
        1.0 + 1.03397 * x + 0.342548 * (x * x) + 0.0506182 * (x * x * x)
    )


def S_fit(x: float) -> float:
    """Combined entropy fitting function"""
    return 1.0 + (7.0 / 4.0) * f_s(x)


class QCD_EOS(GenericEOSBase):

    # above T_HI (measured in GeV) we assume the asymptotic high temperature degrees of freedom
    T_HI = 1e16

    # boundary value (measured in GeV) from Saikawa & Shirai
    T_120_MEV = 0.12

    # below T_LOW (measured in GeV) we assume the asymptotic low temperature degrees of freedom
    T_LO = 1e-5

    def __init__(self, units: UnitsLike):
        GenericEOSBase.__init__(self, units)

    @property
    def name(self):
        return "QCD equation of state in Saikawa & Shirai parametrization (arXiv:1803.01038)"

    @property
    def type_id(self) -> int:
        # 0 is the unique ID for the LambdaCDM cosmology type
        return QCD_EOS_IDENTIFIER

    # Complete effective degrees of freedom functions
    def G(self, T: float) -> float:
        """
        Compute effective number of bosonic degrees of freedom g(T) for the energy, at temperature T.
        T should be regarded as a dimensionful quantity, measured in the given UnitsLike system
        :param T: dimensionful temperature T
        :return: dimensionless number representing g(T)
        """

        T_in_GeV = T / self._units.GeV

        if T_in_GeV > QCD_EOS.T_HI:
            return HIGH_T_GSTAR  # Asymptotic high temperature limit
        elif QCD_EOS.T_120_MEV <= T_in_GeV <= QCD_EOS.T_HI:
            log_T_in_GeV = log(T_in_GeV)
            return polynomial_sum(a_coeffs, log_T_in_GeV) / polynomial_sum(
                b_coeffs, log_T_in_GeV
            )
        elif QCD_EOS.T_LO <= T_in_GeV < QCD_EOS.T_120_MEV:
            # Eq. (C.3) of 1803.01038 v2
            return (
                2.030
                + 1.353 * pow(S_fit(M_e / T_in_GeV), 4.0 / 3.0)
                + 3.495 * f_rho(M_e / T_in_GeV)
                + 3.446 * f_rho(M_mu / T_in_GeV)
                + 1.05 * b_rho(M_pi0 / T_in_GeV)
                + 2.08 * b_rho(M_piplus / T_in_GeV)
                + 4.165 * b_rho(M_1 / T_in_GeV)
                + 30.55 * b_rho(M_2 / T_in_GeV)
                + 89.4 * b_rho(M_3 / T_in_GeV)
                + 8209.0 * b_rho(M_4 / T_in_GeV)
            )
        else:
            return LOW_T_GSTAR  # Low temperature limit

    def Gs(self, T: float) -> float:
        """
        Compute effective number of bosonic degrees of freedom g_S(T) for the entropy, at temperature T
        T should be regarded as a dimensionful quantity, measured in the given UnitsLike system
        :param T: dimensionful temperature T
        :return: dimensionless number representing g_S(T)
        """

        T_in_GeV = T / self._units.GeV

        if T_in_GeV > QCD_EOS.T_HI:
            return HIGH_T_GSTAR  # Asymptotic high temperature limit
        elif QCD_EOS.T_120_MEV <= T_in_GeV <= QCD_EOS.T_HI:
            log_T_in_GeV = log(T)
            return self.G(T) / (
                1.0
                + polynomial_sum(c_coeffs, log_T_in_GeV)
                / polynomial_sum(d_coeffs, log_T_in_GeV)
            )
        elif QCD_EOS.T_LO <= T_in_GeV < QCD_EOS.T_120_MEV:
            # Eq. (C.4) of 1803.01038 v2
            return (
                2.008
                + 1.923 * S_fit(M_e / T_in_GeV)
                + 3.442 * f_s(M_e / T_in_GeV)
                + 3.468 * f_s(M_mu / T_in_GeV)
                + 1.034 * b_s(M_pi0 / T_in_GeV)
                + 2.068 * b_s(M_piplus / T_in_GeV)
                + 4.16 * b_s(M_1 / T_in_GeV)
                + 30.55 * b_s(M_2 / T_in_GeV)
                + 90.0 * b_s(M_3 / T_in_GeV)
                + 6209.0 * b_s(M_4 / T_in_GeV)
            )
        else:
            return LOW_T_G_S_STAR  # Low temperature limit

    # override equation of state implementation
    def w(self, T: float) -> float:
        """
        Compute equation of state parameter w(T) as a function of temperature T.
        :return:
        """

        # Note: This formula is valid only in thermal equilibrium, where all species have the same temperature T.
        # It strictly IS NOT VALID after e+e- annihilation, when the neutrino and photon temperatures separate.
        T_in_GeV = T / self._units.GeV

        if T_in_GeV > QCD_EOS.T_LO:
            G = self.G(T)
            Gs = self.Gs(T)
            return (4.0 * Gs) / (3.0 * G) - 1.0

        # below T_LO we have photons and neutrinos, each with
        #   P(T) = s(T) T - rho(T)
        # so we cannot write a single formula for w(T) = P(T)/rho(T) that is valid both above and below T_LO.
        # However, with our choices w(z) will just evaluate to 1/3 for all temperatures in this range.
        return 1.0 / 3.0
