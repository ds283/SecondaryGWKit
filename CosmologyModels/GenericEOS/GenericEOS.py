from abc import ABC, abstractmethod

from Units.base import UnitsLike

# at high temperature, G and G_S usually have the same value
HIGH_T_GSTAR = 106.75

# G and G_S usually only differ at low temperatures after neutrino decoupling, once e+/e- annihilation
# reheats the photons (but *not* the neutrinos)
# LOW_T_GSTAR = 3.36
# LOW_T_G_S_STAR = 3.91

# TODO: check, https://www.astronomy.ohio-state.edu/weinberg.21/A8873/notes7a.pdf quotes instead
LOW_T_GSTAR = 3.38
LOW_T_G_S_STAR = 3.94
# these values look correct to me because e.g.
#   2 + 2 * 3.042 * (7/8) * (4/11)^(4/3) = 3.38172
# so this value of G* includes N_eff from Planck, plus reheating of the photons but not the neutrinos


class GenericEOSBase(ABC):

    def __init__(self, units: UnitsLike):
        self._units = units

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def type_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def G(self, T: float) -> float:
        """
        Compute effective number of bosonic degrees of freedom g(T) for the energy, at temperature T.
        T should be regarded as a dimensionful quantity, measured in the given UnitsLike system
        :param T: dimensionful temperature T
        :return: dimensionless number representing g(T)
        """
        raise NotImplementedError

    @abstractmethod
    def Gs(self, T: float) -> float:
        """
        Compute effective number of bosonic degrees of freedom g_S(T) for the entropy, at temperature T
        T should be regarded as a dimensionful quantity, measured in the given UnitsLike system
        :param T: dimensionful temperature T
        :return: dimensionless number representing g_S(T)
        """
        raise NotImplementedError

    def w(self, T: float) -> float:
        """
        Generic formula for equation of state parameter w(T) as a function of temperature T.
        T should be regarded as a dimensionful quantity, measured in the given UnitsLike system
        :return:
        """
        G = self.G(T)
        Gs = self.Gs(T)
        w = (4.0 * Gs) / (3.0 * G) - 1.0

        # print(
        #     f">> evaluate w(T) at T = {T/self._units.GeV:.5g} GeV = {T/self._units.Kelvin:.5g} K | g* = {G:.5g}, g_S* = {Gs:.5g}, w = {w:.5g}"
        # )

        return w
