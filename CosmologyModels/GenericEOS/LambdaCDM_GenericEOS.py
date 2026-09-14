from math import exp, sqrt, log, log1p, expm1
from typing import Mapping, Optional

import numpy as np
from numpy import linspace
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyModels import BaseCosmology
from CosmologyModels.GenericEOS.GenericEOS import (
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
    BREAK_POINT_KINDS,
    GenericEOSBase,
    HIGH_T_GSTAR,
)
from Units.base import UnitsLike
from constants import RadiationConstant

DEFAULT_MAX_TEMPERATURE_Z_REDSHIFT = 1e20
# The T(z) tabulation extends a little into the future so that numerical derivatives at z = 0 are
# accurate; see the comment where the spline is built. Four steps of 0.05 is where this value came
# from, but nothing reads a step size any more.
DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2


class LambdaCDM_GenericEOS(BaseCosmology):
    """
    Construct a datastore
    """

    # The identity of the T(z) representation this class builds its background from. It is part of
    # the QCD cosmology's datastore lookup key
    # (Datastore/SQL/ObjectFactories/QCD_Cosmology.py), and it exists because nothing else in that
    # key can see the representation: build() matches on the seven parameter values and
    # log10_max_z, every one of which is an *input* to the model rather than a property of how
    # T(z) is approximated from those inputs.
    #
    # What it identifies -- everything about the background that the parameter key does not
    # otherwise capture:
    #
    #   * the tolerance to which _solve_T_z root-solves each node of the tabulation;
    #   * the quantity that is tabulated and splined -- T itself, or the entropy factor
    #     F(u) = log(T / [T_CMB (1+z)]) whose ramp is known in closed form;
    #   * the number of nodes and the order of the spline through them;
    #   * whether the representation is segmented at the redshifts where T(z) genuinely jumps, and
    #     where those segment edges are placed;
    #   * the set of points integration_break_points declares, because every quadrature and every
    #     ODE in the tree splits its panels there, so a BackgroundModel built against a different
    #     set is a different background.
    #
    # Every prompt in prompts/qcd-background-audit/ that changes any of those bumps this constant,
    # and a bump is the only signal a datastore ever gets:
    #
    #   version | prompt | what changed
    #   --------+--------+-----------------------------------------------------------------------
    #      1    |   03   | nothing numerically; the key exists
    #      2    |   04   | _solve_T_z tightened from xtol=1e-6, rtol=1e-4 to xtol=1e-300, rtol=1e-14
    #
    # (prompts 05, 06 and 07 each append a row here as they land.)
    #
    # Why this is not optional. Without it the same cosmology row is returned under the same
    # serial when the representation changes; every BackgroundModel keyed on that serial is found
    # and deserialised; its tau, cs_tau and friction_F limbs are the *old* background's; and every
    # Gk/Tk numeric and WKB row built on it is served against a background that no longer exists
    # in the code. There is no exception, no warning and no column that differs -- a stale row is
    # otherwise undetectable, which is the whole reason this constant exists. The discrepancy the
    # campaign removes is 3.461e-08 relative in conformal time, which is of order 1.4e5 radians of
    # oscillation phase at k = 3e8/Mpc, so a stale row is not a small inaccuracy.
    #
    # This follows TkNumericIntegration.BREAK_POINT_KIND: a single declaration, readable from the
    # class without an instance, so that the factory can filter on it before any model is built.
    T_Z_REPRESENTATION_VERSION: int = 2

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

    def T_photon(self, z: float) -> float:
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

        # This solve fixes one node of the T(z) spline built in _build_T_z_spline, so its cost is
        # paid once per node at build time (~500 nodes, 8.3 us each -- a few ms total), never at
        # evaluation time. Each node converges independently, so a loose tolerance here does not
        # give a uniformly-scaled error: it gives an *uncorrelated scatter* between neighbouring
        # nodes, and a cubic spline through scattered nodes has a scattered derivative too
        # (prompts/qcd-background-audit/, prompt 04; audit §3, T2). rtol=1e-14 sits just above
        # Brent's own floor of ~4*eps=8.9e-16, so it is the tightest tolerance root_scalar can
        # actually resolve; xtol=1e-300 disables the absolute component entirely -- T spans twenty
        # decades over the tabulated range, so any finite absolute tolerance binds at the cold end
        # long before the hot end is resolved, and the previous xtol=1e-6 did exactly that.
        root = root_scalar(
            T_equation, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=1e-14
        )

        if not root.converged:
            raise RuntimeError(
                f'root_scalar() did not converge to a solution: x_bracket=({bracket_lo:.5g}, {bracket_hi:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
            )

        return root.root

    def _build_T_z_spline(
        self, min_z: float, max_z: float, samples: int = 500
    ) -> ZSplineWrapper:
        # Add a 5% buffer to the min/max z range, so that a caller asking for exactly the
        # requested bounds is inside the tabulated range rather than on its edge.
        #
        # The buffer has to be applied to 1+z, not to z. Scaling z itself works only while z is
        # positive: with min_z = DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2, `0.95 * min_z` is
        # -0.19, which moves the lower bound *inward* and shrinks the range by 5% instead of
        # widening it -- so the model could not be evaluated at its own declared floor. 1+z is
        # positive throughout (z > -1), so scaling that is correct at both ends.
        min_z = 0.95 * (1.0 + min_z) - 1.0
        max_z = 1.05 * (1.0 + max_z) - 1.0

        log_z_values = linspace(log(1.0 + min_z), log(1.0 + max_z), samples)
        T_values = [self._solve_T_z(exp(logz) - 1.0) for logz in log_z_values]

        spline = make_interp_spline(log_z_values, T_values)

        # The knot vector, in log(1+z), is kept because the spline is only C2 at its knots:
        # every quadrature of a quantity built from T(z) has to split its panels there
        # (integration_break_points below). Recorded here, where the spline is built, so that
        # nothing outside this class has to reach into the wrapper for it.
        self._T_z_spline_knots_log1pz = np.unique(np.asarray(spline.t, dtype=float))

        return ZSplineWrapper(
            spline,
            label="T(z)",
            min_z=min_z,
            max_z=max_z,
            log_z=True,
        )

    def _temperature_crossing_log1pz(
        self, T: float, u_lo: float, u_hi: float
    ) -> Optional[float]:
        """
        The point u = log(1+z), strictly inside (u_lo, u_hi), at which T_photon(z) crosses the
        dimensionful temperature T; None if it does not cross inside the range.

        T_photon(z) is monotone in z, so the crossing is unique. It is solved for in u, which is
        the campaign's integration variable, to xtol = rtol = 1e-15; the root is only ever used
        as a Gauss panel edge. The expm1(u) inside q() is the lossy log(1+z) -> z direction
        (CLAUDE.md), but T_photon takes log(1+z) again internally, so it costs ~1 ulp of u.
        """
        log_T = log(T)

        def q(u: float) -> float:
            return log(self.T_photon(expm1(u))) - log_T

        q_lo = q(u_lo)
        q_hi = q(u_hi)
        if q_lo == 0.0 or q_hi == 0.0 or (q_lo > 0.0) == (q_hi > 0.0):
            return None

        root = root_scalar(q, bracket=(u_lo, u_hi), xtol=1e-15, rtol=1e-15)
        if not root.converged:
            raise RuntimeError(
                f"LambdaCDM_GenericEOS.integration_break_points: root_scalar() did not converge "
                f"for T = {T / self._units.GeV:.5g} GeV between u = {u_lo:.6g} and {u_hi:.6g}: "
                f'"{root.flag}"'
            )
        u = float(root.root)
        if not u_lo < u < u_hi:
            return None
        return u

    def integration_break_points(
        self, z_lo: float, z_hi: float, kind: str = BREAK_POINT_ALL
    ) -> np.ndarray:
        """
        Every point in u = log(1+z), strictly inside (log(1+z_lo), log(1+z_hi)), at which
        Hubble(z), rho(z), T_photon(z), wBackground(z) or wPerturbations(z) loses smoothness of
        the requested ``kind``.

        With ``kind = BREAK_POINT_ALL`` (the default, and the historic behaviour) that is:

        * the interior knots of the T(z) spline, where everything built from T(z) is only C2;
        * the redshifts at which T(z) crosses one of the equation of state's
          break_temperatures_GeV, where G, Gs or w change analytic form (a jump in H(z) at a
          G/Gs boundary, a kink in c_s^2 at a w clamp).

        A fixed-order Gauss-Legendre panel that straddles one of these converges only as N^-2
        (docs/gktk-remedial/RESIDUAL-CONVERGENCE.md, §3), so the cumulative tables of
        ComputeTargets/BackgroundModel.py split every production interval at the points returned
        here and integrate the pieces separately. Splitting at the temperatures alone is not
        enough; the knots are the load-bearing half.

        With ``kind = BREAK_POINT_DISCONTINUITY`` only the crossings of the equation of state's
        discontinuity_temperatures_GeV are returned -- the points at which a quantity *jumps*.
        The spline knots are deliberately not included: they are C2 points, which an adaptive ODE
        stepper absorbs, and there are two orders of magnitude more of them (404 against 3 on the
        production range of QCD_Cosmology). This is what
        Quadrature/integrators/numeric_with_phase_cut.py asks for, and its module docstring says
        why the distinction matters there and not in a quadrature.

        :param z_lo: lower redshift of the range (inclusive; a break exactly here is not returned)
        :param z_hi: upper redshift of the range
        :param kind: BREAK_POINT_ALL for every non-smooth point, BREAK_POINT_DISCONTINUITY for
            the subset at which a quantity jumps
        :return: an ascending numpy array of u values, empty if none fall inside the range
        """
        if kind not in BREAK_POINT_KINDS:
            raise ValueError(
                f"LambdaCDM_GenericEOS.integration_break_points: unknown break-point kind "
                f'"{kind}" (expected one of {", ".join(BREAK_POINT_KINDS)})'
            )

        u_lo = log1p(z_lo)
        u_hi = log1p(z_hi)
        if not u_lo < u_hi:
            raise ValueError(
                f"LambdaCDM_GenericEOS.integration_break_points: need z_lo < z_hi "
                f"(got z_lo={z_lo:.6g}, z_hi={z_hi:.6g})"
            )

        breaks = tuple(self._eos.break_temperatures_GeV)
        jumps = tuple(self._eos.discontinuity_temperatures_GeV)
        if not set(jumps).issubset(set(breaks)):
            raise RuntimeError(
                f"LambdaCDM_GenericEOS.integration_break_points: the equation of state "
                f'"{self._eos.name}" declares discontinuity temperatures that are not among its '
                f"break temperatures ({sorted(set(jumps) - set(breaks))} GeV). Every "
                f"discontinuity is a break, so the quadrature path would not be split where the "
                f"ODE path is."
            )

        if kind == BREAK_POINT_ALL:
            knots = self._T_z_spline_knots_log1pz
            points = list(knots[(knots > u_lo) & (knots < u_hi)])
            temperatures = breaks
        else:
            points = []
            temperatures = jumps

        GeV = self._units.GeV
        for T_in_GeV in temperatures:
            u = self._temperature_crossing_log1pz(T_in_GeV * GeV, u_lo, u_hi)
            if u is not None:
                points.append(u)

        if len(points) == 0:
            return np.empty(0, dtype=float)
        return np.unique(np.asarray(points, dtype=float))

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
        denominator = rho["matter"] + rho["radiation"]

        return numerator / denominator
