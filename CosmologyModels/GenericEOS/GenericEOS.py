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
# BREAK_POINT_DISCONTINUITY is the subset at which a quantity *jumps*. It is what an adaptive ODE
# solver needs, and only that: a C2 point does not invalidate an embedded Runge-Kutta error
# estimator -- the step controller absorbs it, at worst paying a few extra steps -- whereas a jump
# in the right-hand side makes the estimate meaningless. The two are asked for separately
# (prompts/GkTk-remedial, prompt 18) because the right answer differs between the two sectors that
# share the numeric driver, and it differs on measurement rather than on principle.
#
# What the two sets are, measured on the production range of QCD_Cosmology
# (prompts/qcd-background-audit/, prompt 07): BREAK_POINT_ALL is the **3** redshifts at which T(z)
# reaches one of break_temperatures_GeV, and BREAK_POINT_DISCONTINUITY the **2** of those at which
# g_s actually steps. Both are crossings of an equation-of-state temperature, and the difference
# between them is one kink, not three orders of magnitude. It used to be three orders of
# magnitude: LambdaCDM_GenericEOS also declared every interior knot of its own T(z) tabulation as
# a break point, 404 of them at 500 nodes and 2,414 at 3,000, which is finding G1 of
# docs/qcd-background-audit-2026-09.md. A knot lattice is a property of an approximation and not
# of a cosmology, and at the shipped order-5 tabulation the first discontinuous derivative of the
# interpolant is the fifth -- three levels below the deepest derivative anything in the tree
# builds. See LambdaCDM_GenericEOS.integration_break_points for the measurement.
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

        **This declaration is the whole of what the cosmology reports.** Since prompt 07 of
        prompts/qcd-background-audit/, integration_break_points(kind=BREAK_POINT_ALL) returns the
        crossings of these temperatures and nothing else -- 3 of QCD_EOS's 4 fall inside the
        production range. It used to return the interior knots of the T(z) tabulation as well,
        which is why a "break point" and "a crossing of one of these temperatures" were once
        different things and are now the same thing.
        """
        return ()

    @property
    def discontinuity_temperatures_GeV(self) -> tuple:
        """
        The subset of break_temperatures_GeV at which a quantity actually *jumps*, rather than
        merely losing a derivative: the pieces of G(T), Gs(T) or w(T) that meet here do not join.

        This is a weaker declaration than break_temperatures_GeV and it exists for a different
        consumer. A fixed-order quadrature panel has to be split at every break, jump or kink
        alike, so it asks for break_temperatures_GeV; an *adaptive* ODE solver is in principle
        troubled only by a jump, because a C2 point does not invalidate an embedded Runge-Kutta
        error estimator while a discontinuous right-hand side does.

        That principle is not, by itself, what the numeric sectors go by, and this docstring used
        to say it was. GkTk-remedial prompt 19 *measured* the question and found that the Tk
        numeric sector did better with more than the jumps, so the choice of kind is a per-caller
        decision taken on measurement: Quadrature/integrators/numeric_with_phase_cut.py's module
        docstring records which sector asks for which and why, and
        LambdaCDM_GenericEOS.integration_break_points(..., kind=) is how the two are requested.
        On QCD_Cosmology's production range the two sets are 3 points and 2 -- the same
        temperature crossings, less the one join at which g_s does not actually step
        (prompts/qcd-background-audit/, prompt 07).

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
