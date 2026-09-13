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


# The two kinds of non-smoothness a consumer can ask a cosmology for; see
# GenericEOSBase.break_temperatures_GeV and .discontinuity_temperatures_GeV, and
# LambdaCDM_GenericEOS.integration_break_points, which takes one of these as its `kind` argument.
#
# BREAK_POINT_ALL is *every* point at which some background quantity loses smoothness, jump and
# kink alike. It is what a fixed-order quadrature panel needs: a Gauss-Legendre rule converges only
# as N^-2 across a point where the integrand is merely C2, so the cumulative tables of
# ComputeTargets/BackgroundModel.py split their panels at all of them.
#
# BREAK_POINT_DISCONTINUITY is the strict subset at which a quantity *jumps*. It is what an
# adaptive ODE solver needs, and only that: a C2 point does not invalidate an embedded
# Runge-Kutta error estimator -- the step controller absorbs it, at worst paying a few extra steps
# -- whereas a jump in the right-hand side makes the estimate meaningless. The two sets differ by
# three orders of magnitude on the production range of QCD_Cosmology (404 spline knots against 3
# temperature crossings), which is why they are asked for separately (prompts/GkTk-remedial,
# prompt 18).
BREAK_POINT_ALL = "all"
BREAK_POINT_DISCONTINUITY = "discontinuity"

BREAK_POINT_KINDS = (BREAK_POINT_ALL, BREAK_POINT_DISCONTINUITY)


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

    @property
    def break_temperatures_GeV(self) -> tuple:
        """
        Temperatures, in GeV, at which G(T), Gs(T) or w(T) switch between analytic pieces and so
        lose smoothness (a jump, or a kink). Quadratures of anything built from H(z) or c_s^2(z)
        must not straddle a crossing of one of these temperatures: LambdaCDM_GenericEOS reports
        the corresponding redshifts through integration_break_points(), and the cumulative tables
        of ComputeTargets/BackgroundModel.py split their Gauss panels there
        (prompts/GkTk-remedial, log 02). A smooth equation of state has none.
        """
        return ()

    @property
    def discontinuity_temperatures_GeV(self) -> tuple:
        """
        The subset of break_temperatures_GeV at which a quantity actually *jumps*, rather than
        merely losing a derivative: the pieces of G(T), Gs(T) or w(T) that meet here do not join.

        This is a strictly weaker declaration than break_temperatures_GeV and it exists for a
        different consumer. A fixed-order quadrature panel has to be split at every break, jump or
        kink alike, so it asks for break_temperatures_GeV; an *adaptive* ODE solver only has to be
        split at a jump, because a C2 point does not invalidate an embedded Runge-Kutta error
        estimator while a discontinuous right-hand side does. See
        Quadrature/integrators/numeric_with_phase_cut.py for why that matters, and
        LambdaCDM_GenericEOS.integration_break_points(..., kind=) for how the two are requested.

        A smooth equation of state has none, which is also the default: an equation of state
        written without any knowledge of this distinction declares nothing, and every consumer
        then treats it as smooth and behaves exactly as it did before this method existed.

        Must be a subset of break_temperatures_GeV; integration_break_points() checks.
        """
        return ()

    def w(self, T: float) -> float:
        """
        Generic formula for equation of state parameter w(T) as a function of temperature T.
        T should be regarded as a dimensionful quantity, measured in the given UnitsLike system
        :return:
        """

        # TODO: This formula is valid only in thermal equilibrium, where all species have the same temperature T.
        #  It strictly IS NOT VALID after e+e- annihilation, when the neutrino and photon temperatures separate.
        G = self.G(T)
        Gs = self.Gs(T)
        w = (4.0 * Gs) / (3.0 * G) - 1.0

        # print(
        #     f">> evaluate w(T) at T = {T/self._units.GeV:.5g} GeV = {T/self._units.Kelvin:.5g} K | g* = {G:.5g}, g_S* = {Gs:.5g}, w = {w:.5g}"
        # )

        return w
