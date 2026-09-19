"""
Tests for ComputeTargets/TkSourceFunctions.py, the two-region (numeric + Liouville-Green)
representation of the transfer function T_k(z).

Everything is built from the *exact* constant-w solution, so no Ray cluster and no datastore is
needed: `TkSourceFunctions` consumes its two inputs through a duck-typed protocol (see its
module docstring), and the stand-ins below supply exactly that protocol. The background is the
constant-w `FakeModel` used by docs/spec-code-audit/scripts/TK_03_numeric_vs_analytic.py,
reproduced here rather than imported (nothing in docs/ is importable machinery).

Two WKB fixtures are used, because a single one cannot test both of the things that need
testing:

  * fixture "exact": theta and the amplitude are the *exact* amplitude-phase decomposition of
    the analytic transfer function, T = M sin(theta) with
    M = 2^(3/2+b) Gamma(5/2+b) x^(-3/2-b) m(x) and theta = pi - vartheta(x), where (m, vartheta)
    are the Bessel modulus and phase from LiouvilleGreen.bessel_phase (J_nu = m sin vartheta)
    and x = k c_s a0 tau. The stored `friction` samples are backed out from that exact envelope,
    so `TkSourceFunctions.M` must reproduce it to the arithmetic of the reconstruction. This
    fixture tests the assembly of the amplitude and the reconstruction of the phase.

    Since prompt 10 of prompts/GkTk-remedial the amplitude's friction factor is read from
    `model.functions.friction_F`, not from the stored samples, and the two are cross-checked at
    construction. This fixture therefore hands `TkSourceFunctions` a *model variant* whose
    `friction_F` is the primitive of the backed-out samples (`Fixture.exact_envelope_model`) --
    a stand-in background whose Liouville-Green friction integral is the exact envelope's rather
    than the constant-w closed form. Everything else about the model is unchanged, and only the
    amplitude path reads `friction_F`.

  * fixture "LG": `friction` is the exact Liouville-Green friction integral
    F = (3/2)(1+w) log((1+z)/(1+z_init)) (the integral of `TkWKBIntegration.friction_RHS`) and
    theta is the exact integral of omega_eff. This fixture is the one on which the closed-form
    identities can be checked, because it is the code's own LG representation: d ln M/dz must
    equal a finite difference of log M, and omega() must equal phase.theta_deriv().

They differ because the exact Bessel envelope is *not* the Liouville-Green amplitude: they
differ by the LG truncation error, ~1e-5 in d ln M/dz at 3.5 e-folds sub-horizon (measured
below, and grid-independent), which is far above the 1e-6 the identity check needs.

Three error sources therefore sit under the numbers this module prints, and they must not be
conflated (prompts/transfer-remedial/DRAFT-PLAN.md 8.3):

  * the **Bessel oracle** -- how well `bessel_phase` reproduces J_nu and Y_nu. Until the
    transfer-remedial campaign's prompt 05 this was ~x * 1e-8 in phase, and it was the largest
    of the three below the hand-over; it is now 5e-12 rad declared (2.5e-12 measured at these
    orders), so it no longer contributes to anything measured here;
  * the **consumer re-spline** -- the h^4 cubic fit `TkSourceFunctions` puts through the sampled
    (div 2pi, mod 2pi) phase and the friction samples on the production redshift grid. This
    campaign does **not** improve it, and it is now the binding term in `err_T` below;
  * the **LG truncation** -- physical, ~1e-5 in d ln M/dz, unchanged by any of this.
"""

import unittest
from math import log, exp, sqrt, sin, pi, gamma, atan, fabs

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv

from ComputeTargets.BackgroundModel import ModelFunctions
from ComputeTargets.TkSourceFunctions import TkSourceFunctions
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from ComputeTargets.tests.wkb_reference import ClosedFormPrimitive
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.phase_spline import phase_spline

# the sampling density main.py uses for the source redshift grid
PRODUCTION_SAMPLES_PER_LOG10Z = 100

# a refined WKB grid, used where the assertion is about the object rather than about the
# production grid's resolution
REFINED_SAMPLES_PER_LOG10Z = 300

# the two equations of state exercised: b = 0 and b = 0.25
W_VALUES = (1.0 / 3.0, 0.2)


class FakeModel:
    """
    Constant-w background: H = H0 (1+z)^p with p = 3(1+w)/2, epsilon = p, and
    a0*tau = 1/(H0 (p-1) (1+z)^(p-1)).

    The two primitives prompt 04 of prompts/GkTk-remedial added to `ModelFunctions` are supplied
    here in closed form, as `ClosedFormPrimitive` stand-ins for its `TablePrimitive`:

        cs_tau(z) = sqrt(w) * tau(z)                    (c_s^2 = wPerturbations = w)
        friction_F(z) = (3/2)(1 + w) log(1 + z)         (dF/dz = (3/2)(1 + c_s^2)/(1+z))

    `friction_F` carries an arbitrary additive constant, as the production table does: only
    `delta` is used downstream.
    """

    def __init__(self, w, H0=1.0):
        self.w = w
        self.H0 = H0
        self.p = 1.5 * (1.0 + w)
        self.functions = ModelFunctions(
            Hubble=self.Hubble,
            epsilon=lambda z: self.p,
            d_epsilon_dz=lambda z: 0.0,
            d2_epsilon_dz2=lambda z: 0.0,
            wBackground=lambda z: self.w,
            wPerturbations=lambda z: self.w,
            tau=self.tau,
            T_photon=lambda z: 0.0,
            d_lnH_dz=lambda z: self.p / (1.0 + z),
            d2_lnH_dz2=lambda z: -self.p / (1.0 + z) ** 2,
            d3_lnH_dz3=lambda z: 2.0 * self.p / (1.0 + z) ** 3,
            d_wPerturbations_dz=lambda z: 0.0,
            d2_wPerturbations_dz2=lambda z: 0.0,
            cs_tau=ClosedFormPrimitive(self.cs_tau, label="cs_tau"),
            friction_F=ClosedFormPrimitive(self.friction_F, label="friction_F"),
        )

    def Hubble(self, z):
        return self.H0 * (1.0 + z) ** self.p

    def tau(self, z):
        return 1.0 / (self.H0 * (self.p - 1.0) * (1.0 + z) ** (self.p - 1.0))

    def cs_tau(self, z):
        return sqrt(self.w) * self.tau(z)

    def friction_F(self, z):
        return 1.5 * (1.0 + self.w) * log(1.0 + z)


class FakeRedshift:
    def __init__(self, z):
        self.z = z


class FakeNumericValue:
    def __init__(self, z, T, Tprime):
        self.z = FakeRedshift(z)
        self.T = T
        self.Tprime = Tprime


class FakeWKBValue:
    def __init__(
        self, z, theta_div_2pi, theta_mod_2pi, friction, H_ratio, omega_WKB_sq
    ):
        self.z = FakeRedshift(z)
        self.theta_div_2pi = theta_div_2pi
        self.theta_mod_2pi = theta_mod_2pi
        self.friction = friction
        self.H_ratio = H_ratio
        self.omega_WKB_sq = omega_WKB_sq


class FakeTkNumeric:
    def __init__(self, values, z_exit, stop_deltaz_subh):
        self.values = values
        self.z_exit = z_exit
        self.stop_deltaz_subh = stop_deltaz_subh


class FakeTkWKB:
    def __init__(self, values, sin_coeff, z_init, cos_coeff=0.0):
        self.values = values
        self.sin_coeff = sin_coeff
        self.cos_coeff = cos_coeff
        self.z_init = z_init


class FakeWavenumber:
    """
    Stands in for a `wavenumber_exit_time`: `float(k)` is k/a0, and `z_exit` is horizon crossing.
    """

    def __init__(self, k, z_exit):
        self.k = k
        self.z_exit = z_exit

    def __float__(self):
        return self.k


def log_grid(one_plus_z_high, one_plus_z_low, samples_per_log10z):
    n = int(round(samples_per_log10z * np.log10(one_plus_z_high / one_plus_z_low))) + 1
    return list(
        np.exp(np.linspace(np.log(one_plus_z_high), np.log(one_plus_z_low), n)) - 1.0
    )


def log_midpoints(z_list):
    return [
        exp(0.5 * (log(1.0 + z_list[i]) + log(1.0 + z_list[i + 1]))) - 1.0
        for i in range(len(z_list) - 1)
    ]


class Fixture:
    """
    Build the exact constant-w inputs for `TkSourceFunctions`.

    The numeric region runs from `efolds_suph` e-folds outside the horizon down to a hand-over
    `efolds_subh` e-folds inside it; the WKB region runs from the hand-over down to the redshift
    at which x = k c_s a0 tau reaches `x_max`.
    """

    def __init__(
        self,
        w,
        k=1.0e4,
        efolds_suph=5.0,
        efolds_subh=3.5,
        x_max=1.0e3,
        numeric_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
        WKB_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
        drop_first_WKB_sample=False,
    ):
        self.w = w
        self.k = k
        self.model = FakeModel(w)
        self.b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
        self.cs = sqrt(w)
        self.nu = 1.5 + self.b

        p = self.model.p

        def one_plus_z_at(k_over_aH):
            # k/(aH) = k (1+z)^(1-p) / H0
            return (k / self.model.H0 / k_over_aH) ** (1.0 / (p - 1.0))

        self.z_exit = one_plus_z_at(1.0) - 1.0

        z_numeric = log_grid(
            one_plus_z_at(exp(-efolds_suph)),
            one_plus_z_at(exp(efolds_subh)),
            numeric_samples,
        )
        # x = c_s/(p-1) * k/(aH)
        z_WKB = log_grid(
            one_plus_z_at(exp(efolds_subh)),
            one_plus_z_at(x_max * (p - 1.0) / self.cs),
            WKB_samples,
        )

        self.crossover_z = z_WKB[0]
        if drop_first_WKB_sample:
            # production shape: main.py truncates the WKB grid to the largest grid point at or
            # below z_init, so the first WKB sample can sit one grid step below the hand-over
            z_WKB = z_WKB[1:]

        self.z_numeric = z_numeric
        self.z_WKB = z_WKB

        self.k_object = FakeWavenumber(k, self.z_exit)
        self.Tk_numeric = FakeTkNumeric(
            [
                FakeNumericValue(
                    z,
                    self.T_exact(z),
                    self.Tprime_exact(z),
                )
                for z in z_numeric
            ],
            z_exit=self.z_exit,
            stop_deltaz_subh=self.z_exit - self.crossover_z,
        )

        self._bessel = bessel_phase(self.nu, 1.02 * self.x(z_WKB[-1]))

    # --- exact solution ----------------------------------------------------------------

    def x(self, z):
        return self.k * self.cs * self.model.tau(z)

    def T_exact(self, z):
        return compute_analytic_T(self.k, self.w, self.model.tau(z))

    def Tprime_exact(self, z):
        return compute_analytic_Tprime(
            self.k, self.w, self.model.tau(z), self.model.Hubble(z)
        )

    def omega(self, z):
        return sqrt(Tk_omegaEff_sq(self.model, self.k, z))

    def M_exact(self, z):
        """
        Exact envelope of the analytic transfer function: 2^nu Gamma(5/2+b) x^-nu m(x), with
        m = sqrt(J^2 + Y^2) supplied by bessel_phase (its module docstring, bessel_phase.py:13:
        J_nu = A_nu sin theta_nu, Y_nu = -A_nu cos theta_nu). The line reference used to be
        bessel_phase.py:270-282, which prompt 05 of prompts/transfer-remedial rewrote; the
        convention it named is unchanged.
        """
        x = self.x(z)
        return (
            2.0**self.nu
            * gamma(2.5 + self.b)
            * x ** (-self.nu)
            * self._bessel["mod"](x)
        )

    def theta_exact(self, z):
        """
        The exact phase, rotated into the code's convention. bessel_phase's vartheta *increases*
        with x, hence decreases with z; theta = pi - vartheta therefore increases with z, as the
        code's d(theta)/dz = +omega_eff convention requires, and sin(theta) = sin(vartheta) keeps
        the amplitude positive.

        Re-checked clause by clause against the two-region construction and its zero-point
        c_nu = pi/4 - pi nu/2 (prompts/transfer-remedial prompt 08, log 08), because the whole
        phase construction under `vartheta` was replaced:

          * vartheta increases with x -- theta_deriv = a_nu^-2 >= 0.9959 over this fixture's
            x in [18.5, 1000] at both w, and the sampled vartheta is strictly increasing;
          * x = k c_s a0 tau(z) is strictly decreasing in z, so vartheta decreases with z;
          * hence theta = pi - vartheta increases with z, confirmed on the stored z_WKB grid;
          * d theta/dz is positive and agrees with +sqrt(Tk_omegaEff_sq) to 1.1e-5 (w = 1/3) and
            1.9e-4 (w = 0.2) relative. That residual is the LG truncation of the closed-form
            omega_eff, not a phase error: it is the same O(x^-4) quantity
            `test_omega_matches_phase_derivative` measures at 4.2e-8 on the *LG* fixture, seen
            here on the *exact* one where the two representations genuinely differ;
          * sin(pi - vartheta) equals sin(vartheta) to 1.1e-14, which is the rounding of the
            subtraction `pi - vartheta` itself (pi is a 53-bit approximation and vartheta reaches
            1000), and is a floor introduced by this fixture's rotation rather than by
            bessel_phase. It sits three orders below the re-spline term that dominates `err_T`.

        The zero-point is exact: `phase.c_nu` equals pi/4 - pi nu/2 bit for bit at both orders,
        and raw_theta(x) - (x + c_nu + residual(x)) is exactly 0.
        """
        return pi - self._bessel["phase"].raw_theta(self.x(z))

    # --- the two WKB fixtures ----------------------------------------------------------

    def exact_envelope_F(self, z):
        """
        The friction primitive implied by the *exact* envelope, i.e. the F for which

            M(z) = sin_coeff sqrt(H_init/H) omega^(-1/2) exp(F(z) - F(z_init)) = M_exact(z)

        identically. Rearranging the amplitude formula,

            F(z) = log(M_exact(z) sqrt(omega(z))) + (1/2) log H(z)   (+ any constant).

        This is not the constant-w Liouville-Green friction integral: the two differ by the LG
        truncation error of the amplitude (~1e-5 at this fixture's hand-over, measured by
        `test_LG_truncation_error_of_the_exact_envelope`). The "exact" fixture is the one that
        tests the *assembly* of the amplitude, so its model must be one whose friction table
        agrees with its stored samples -- see `exact_envelope_model`.
        """
        return log(self.M_exact(z) * sqrt(self.omega(z))) + 0.5 * log(
            self.model.Hubble(z)
        )

    def exact_envelope_model(self):
        """
        A copy of this fixture's `FakeModel` whose `friction_F` is `exact_envelope_F` instead of
        the constant-w closed form. Everything else -- Hubble, epsilon, wPerturbations, cs_tau --
        is unchanged, and only `TkSourceFunctions`' amplitude path reads `friction_F`.
        """
        model = FakeModel(self.w, self.model.H0)
        model.functions = model.functions._replace(
            friction_F=ClosedFormPrimitive(
                self.exact_envelope_F, label="friction_F (exact envelope)"
            )
        )
        return model

    def exact_functions(self):
        """
        `TkSourceFunctions` whose amplitude is the exact Bessel envelope and whose phase is the
        exact Bessel phase. `sin_coeff` is backed out from the exact amplitude at the hand-over
        (where H_ratio = 1 and F = 0, exactly as TkWKBIntegration.store() arranges), and the
        stored friction samples are then whatever makes the code's amplitude formula exact --
        which is `exact_envelope_F` measured from the hand-over, so `exact_envelope_model()` is
        the background this fixture must be read against.

        The phase is reduced by `WKB_mod_2pi`, not `wrap_theta`. Both return the same
        (div 2pi, mod 2pi) pair with mod in (-2pi, 0] -- the cycle counts agree on every sample
        of every fixture the suite builds -- but `wrap_theta` subtracts TWO_PI once per cycle,
        costing O(theta / 2pi) and accumulating 2.3e-12 rad of rounding at this fixture's
        |theta| ~ 1e3 (1.5e-4 rad at 1e7), whereas `WKB_mod_2pi`'s remainder is an `fmod` and
        exact (docs/radiation-oracle/KOHRI-TERADA-ORACLE.md section 8, Table 8.3).
        """
        z_init = self.crossover_z
        sin_coeff = self.M_exact(z_init) * sqrt(self.omega(z_init))

        values = []
        for z in self.z_WKB:
            div_2pi, mod_2pi = WKB_mod_2pi(self.theta_exact(z))
            omega = self.omega(z)
            H_ratio = self.model.Hubble(self.crossover_z) / self.model.Hubble(z)
            friction = log(self.M_exact(z) * sqrt(omega) / (sin_coeff * sqrt(H_ratio)))
            values.append(
                FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
            )

        return TkSourceFunctions(
            self.exact_envelope_model(),
            self.k_object,
            self.Tk_numeric,
            FakeTkWKB(values, sin_coeff, self.crossover_z),
        )

    def LG_functions(self, sin_coeff=0.37):
        """
        `TkSourceFunctions` built from the code's own Liouville-Green ingredients: the friction
        integral in closed form and the phase as the exact integral of omega_eff from the
        hand-over.
        """
        sol = solve_ivp(
            lambda z, y: [self.omega(z)],
            t_span=(self.crossover_z, self.z_WKB[-1]),
            y0=[0.0],
            t_eval=self.z_WKB,
            method="DOP853",
            atol=1.0e-14,
            rtol=1.0e-13,
        )
        assert sol.success, sol.message

        values = []
        for i, z in enumerate(self.z_WKB):
            div_2pi, mod_2pi = WKB_mod_2pi(sol.y[0][i])
            omega = self.omega(z)
            H_ratio = self.model.Hubble(self.crossover_z) / self.model.Hubble(z)
            friction = 1.5 * (1.0 + self.w) * log((1.0 + z) / (1.0 + self.crossover_z))
            values.append(
                FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
            )

        return TkSourceFunctions(
            self.model,
            self.k_object,
            self.Tk_numeric,
            FakeTkWKB(values, sin_coeff, self.crossover_z),
        )


class AnalyticRadiationFixture:
    """
    A w = 1/3 fixture that reaches x_T = 1e6 without touching `bessel_phase`.

    At w = 1/3 the transfer function and its exact amplitude-phase decomposition are elementary
    (b = 0, nu = 3/2, x = k c_s a0 tau = k/(sqrt(3)(1+z)) since p = 2):

        T(x) = 3 (sin x - x cos x)/x^3 = M(x) sin(psi(x)),
        M(x) = 3 sqrt(1 + x^2)/x^3,        psi(x) = x - arctan(x),

    because sin x - x cos x = sqrt(1 + x^2) sin(x - arctan x). The phase is rotated into the
    code's convention as theta = pi - psi, which increases with z and leaves sin(theta) =
    sin(psi), exactly as `Fixture.theta_exact` does with the Bessel phase.

    `Fixture` cannot be used for this: it builds a `bessel_phase` table up to 1.02 x_max in its
    constructor, and the campaign's own note on that oracle
    (docs/lg-phase-and-handover-followup-2026-09.md 2.4) is that its phase error on `main` grew
    like x * 1e-8, i.e. ~1e-2 rad at x = 1e6 -- five orders above what this test measures. The
    closed forms above are exact.

    The stored friction is backed out from the exact envelope in the same way as
    `Fixture.exact_functions`, and the model handed to `TkSourceFunctions` carries the matching
    `friction_F` primitive.
    """

    def __init__(
        self,
        k=1.0e8,
        efolds_suph=5.0,
        efolds_subh=3.5,
        x_max=1.0e6,
        WKB_samples=PRODUCTION_SAMPLES_PER_LOG10Z,
    ):
        self.w = 1.0 / 3.0
        self.k = k
        self.cs = sqrt(self.w)
        self.model = FakeModel(self.w)

        p = self.model.p  # = 2

        def one_plus_z_at(k_over_aH):
            return (k / self.model.H0 / k_over_aH) ** (1.0 / (p - 1.0))

        self.z_exit = one_plus_z_at(1.0) - 1.0

        self.z_numeric = log_grid(
            one_plus_z_at(exp(-efolds_suph)),
            one_plus_z_at(exp(efolds_subh)),
            PRODUCTION_SAMPLES_PER_LOG10Z,
        )
        self.z_WKB = log_grid(
            one_plus_z_at(exp(efolds_subh)),
            one_plus_z_at(x_max * (p - 1.0) / self.cs),
            WKB_samples,
        )
        self.crossover_z = self.z_WKB[0]

        self.k_object = FakeWavenumber(k, self.z_exit)
        self.Tk_numeric = FakeTkNumeric(
            [
                FakeNumericValue(
                    z,
                    compute_analytic_T(self.k, self.w, self.model.tau(z)),
                    compute_analytic_Tprime(
                        self.k, self.w, self.model.tau(z), self.model.Hubble(z)
                    ),
                )
                for z in self.z_numeric
            ],
            z_exit=self.z_exit,
            stop_deltaz_subh=self.z_exit - self.crossover_z,
        )

    def x(self, z):
        return self.k * self.cs * self.model.tau(z)

    def omega(self, z):
        return sqrt(Tk_omegaEff_sq(self.model, self.k, z))

    def M_exact(self, z):
        x = self.x(z)
        return 3.0 * sqrt(1.0 + x * x) / (x * x * x)

    def theta_exact(self, z):
        """theta = pi - (x - arctan x): decreasing in x, hence increasing in z."""
        x = self.x(z)
        return pi - (x - atan(x))

    def exact_envelope_F(self, z):
        return log(self.M_exact(z) * sqrt(self.omega(z))) + 0.5 * log(
            self.model.Hubble(z)
        )

    def exact_envelope_model(self):
        model = FakeModel(self.w, self.model.H0)
        model.functions = model.functions._replace(
            friction_F=ClosedFormPrimitive(
                self.exact_envelope_F, label="friction_F (exact envelope)"
            )
        )
        return model

    def stored_values(self):
        """
        The stored (div 2pi, mod 2pi, friction) samples.

        The reduction is `WKB_mod_2pi`, the producers' own (README section 2 (e) of
        prompts/GkTk-remedial), and **not** `wrap_theta` as the smaller fixtures once did:
        `wrap_theta` reduces by adding TWO_PI in a loop, so at theta ~ -1e6 rad it takes ~1.6e5
        additions and accumulates ~1.4e-06 rad of rounding -- fourteen times the bound this
        fixture's test asserts, and injected by the fixture rather than by anything under test.
        `WKB_mod_2pi`'s *remainder* is an `fmod` and is exact -- its cycle count is not an
        `fmod`, and used to be a separately rounded division that could be one cycle out
        (`[13-wkb-mod-2pi-cycle-count-inconsistent]`, fixed by prompt 01 of
        prompts/phase-representation, which derives the count from the remainder). That leaves
        only the ~1 ulp of `div * TWO_PI` in the reconstruction (1.2e-10 rad at 1e6 rad).
        """
        z_init = self.crossover_z
        sin_coeff = self.M_exact(z_init) * sqrt(self.omega(z_init))

        values = []
        for z in self.z_WKB:
            div_2pi, mod_2pi = WKB_mod_2pi(self.theta_exact(z))
            omega = self.omega(z)
            H_ratio = self.model.Hubble(z_init) / self.model.Hubble(z)
            friction = log(self.M_exact(z) * sqrt(omega) / (sin_coeff * sqrt(H_ratio)))
            values.append(
                FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
            )

        return values, sin_coeff

    def functions(self):
        values, sin_coeff = self.stored_values()
        return TkSourceFunctions(
            self.exact_envelope_model(),
            self.k_object,
            self.Tk_numeric,
            FakeTkWKB(values, sin_coeff, self.crossover_z),
        )

    def stored_phase_spline(self):
        """
        The consumer this prompt replaces: a cubic `phase_spline` through the same stored
        (div 2pi, mod 2pi) samples, with the arguments `TkSourceFunctions._build_WKB` used to
        pass. Built here so that the two representations can be scored against the same exact
        phase on the same grid.
        """
        values, _ = self.stored_values()
        return phase_spline(
            [log(1.0 + v.z.z) for v in values],
            [v.theta_div_2pi for v in values],
            [v.theta_mod_2pi for v in values],
            x_is_log=True,
            x_is_redshift=True,
            chunk_step=None,
            chunk_logstep=125,
            increasing=True,
        )


class TestNumericRegion(unittest.TestCase):
    def test_reproduces_exact_solution(self):
        for w in W_VALUES:
            with self.subTest(w=w):
                f = Fixture(w)
                functions = f.exact_functions()

                self.assertAlmostEqual(
                    functions.numeric_region[0], f.z_numeric[0], delta=1.0e-9
                )
                self.assertAlmostEqual(
                    functions.numeric_region[1], f.crossover_z, delta=1.0e-9
                )
                self.assertAlmostEqual(
                    functions.crossover_z, f.crossover_z, delta=1.0e-9
                )

                T_envelope = max(abs(f.T_exact(z)) for z in f.z_numeric)
                Tprime_envelope = max(abs(f.Tprime_exact(z)) for z in f.z_numeric)

                # at the sample points the spline is the data, so this checks the wiring:
                # the right values against the right abscissae
                node_T = max(
                    abs(functions.T(z) - f.T_exact(z)) / T_envelope for z in f.z_numeric
                )
                node_Tprime = max(
                    abs(functions.dT_dz(z) - f.Tprime_exact(z)) / Tprime_envelope
                    for z in f.z_numeric
                )
                self.assertLess(node_T, 1.0e-12)
                self.assertLess(node_Tprime, 1.0e-12)

                # between the sample points the residual is the cubic-spline fit error on the
                # production 100-per-log10(1+z) grid, at 3.5 e-folds sub-horizon
                mid = log_midpoints(f.z_numeric)
                mid_T = max(
                    abs(functions.T(z) - f.T_exact(z)) / T_envelope for z in mid
                )
                mid_Tprime = max(
                    abs(functions.dT_dz(z) - f.Tprime_exact(z)) / Tprime_envelope
                    for z in mid
                )
                print(
                    f"\n[numeric region, w={w:.5g}] node error: T {node_T:.3e}, dT/dz {node_Tprime:.3e}; "
                    f"midpoint error: T {mid_T:.3e}, dT/dz {mid_Tprime:.3e}"
                )
                self.assertLess(mid_T, 1.0e-4)
                self.assertLess(mid_Tprime, 1.0e-3)

    def test_unity_above_the_numeric_region(self):
        f = Fixture(1.0 / 3.0)
        functions = f.exact_functions()

        z_above = 10.0 * (1.0 + f.z_numeric[0]) - 1.0
        self.assertEqual(functions.T(z_above), 1.0)
        self.assertEqual(functions.dT_dz(z_above), 0.0)

        # and T is still the spline at the top of the region itself
        self.assertNotEqual(functions.T(f.z_numeric[0]), 1.0)

    def test_raises_below_the_numeric_region(self):
        f = Fixture(1.0 / 3.0)
        functions = f.exact_functions()

        with self.assertRaises(RuntimeError):
            functions.T(0.5 * f.crossover_z)
        with self.assertRaises(RuntimeError):
            functions.dT_dz(0.5 * f.crossover_z)


class TestExactLGFixture(unittest.TestCase):
    """
    The object must reproduce an exact amplitude-phase decomposition of T_k, up to spline error.
    """

    def test_amplitude_and_reconstruction(self):
        for w in W_VALUES:
            with self.subTest(w=w):
                f = Fixture(w, WKB_samples=REFINED_SAMPLES_PER_LOG10Z)
                functions = f.exact_functions()

                self.assertAlmostEqual(
                    functions.WKB_region[0], f.crossover_z, delta=1.0e-9
                )
                self.assertAlmostEqual(
                    functions.WKB_region[1], f.z_WKB[-1], delta=1.0e-9
                )

                mid = log_midpoints(f.z_WKB)

                err_M = max(
                    abs(functions.M(z) - f.M_exact(z)) / abs(f.M_exact(z)) for z in mid
                )
                # normalised by the *local* envelope M(z), i.e. by the amplitude of the
                # oscillation being reconstructed
                err_T = max(
                    abs(functions.T_WKB(z) - f.M_exact(z) * sin(f.theta_exact(z)))
                    / abs(f.M_exact(z))
                    for z in mid
                )
                # The same comparison against scipy's J_nu, i.e. against a Bessel function this
                # fixture never touched. It used to carry bessel_phase's own phase-function error
                # on top of err_T and so saturated near 2e-6 -- measured 1.985e-06 (w = 1/3) and
                # 1.550e-06 (w = 0.2) on the tree before prompts/transfer-remedial's prompt 05.
                # It is now 3.021e-08 and 2.234e-08, i.e. equal to err_T to every printed digit:
                # the oracle's contribution has dropped below the fixture's own phase re-spline
                # error and this comparison no longer measures bessel_phase at all. Printed and
                # not asserted, because adding an assertion here is outside prompt 08's remit
                # (campaign README 4.2 allows comments and tolerance constants only in this file);
                # board issue [08-tk-fixture-scipy-comparison-unasserted].
                err_scipy = max(
                    abs(
                        functions.T_WKB(z)
                        - 2.0**f.nu
                        * gamma(2.5 + f.b)
                        * f.x(z) ** (-f.nu)
                        * jv(f.nu, f.x(z))
                    )
                    / abs(f.M_exact(z))
                    for z in mid
                )
                print(
                    f"\n[exact LG fixture, w={w:.5g}] max |dM|/M = {err_M:.3e}; "
                    f"|T_WKB - M sin(theta)|/envelope = {err_T:.3e}; vs scipy J_nu = {err_scipy:.3e}"
                )
                # err_M is the amplitude path only: the friction samples are backed out from
                # M_exact and re-splined, so the residual is the amplitude re-spline plus the
                # arithmetic of the reconstruction. Measured 1.272e-13 (w = 1/3) and 7.843e-14
                # (w = 0.2), essentially unchanged by the campaign (1.268e-13 and 7.708e-14
                # before it) -- the old oracle's amplitude was already good to ~5e-14. Tightened
                # from 1e-8 to 1e-12, about eight times the measured maximum.
                self.assertLess(err_M, 1.0e-12)
                # err_T is **not** tightened. What limits it is the consumer re-spline -- the
                # h^4 cubic fit TkSourceFunctions puts through the sampled phase on this
                # fixture's grid -- which this campaign does not touch: measured 3.021e-08
                # (w = 1/3) and 2.234e-08 (w = 0.2), against 3.021e-08 and 2.215e-08 before it,
                # and shown to be the re-spline rather than the oracle by
                # test_spline_error_dominates_on_the_production_grid and by the [grid refinement]
                # line, which moves by two orders (6.090e-06 -> 3.021e-08) when the grid is
                # tripled. 1e-7 leaves a factor 3.3, which is the right margin for a fit error
                # whose size depends on where the grid lands.
                self.assertLess(err_T, 1.0e-7)

    def test_phase_convention(self):
        f = Fixture(1.0 / 3.0)
        functions = f.exact_functions()

        mid = log_midpoints(f.z_WKB)
        # phase_spline rebases each chunk by an integer number of cycles, so the remainder it
        # returns is congruent to theta mod 2pi but does not carry the stored samples' negative
        # sign convention. What must hold is that it is a remainder, and that its sine is the
        # sine of the exact phase.
        #
        # `functions.phase` is TkSourceFunctions' own phase_spline over the stored samples, not
        # bessel_phase, so the 1e-6 below is limited by that re-spline and not by the Bessel
        # oracle. Measured 7.342e-08 over mid[:20] (x ~ 19, just sub-hand-over, where the fit
        # error is smallest) and 6.09e-06 over the whole grid, so the factor 13.6 of headroom
        # here is the right margin for a fit error and 1e-6 is not tightened. Note also that
        # bessel_phase's own theta_mod_2pi now returns (-pi, pi] rather than fmod(theta, 2 pi)
        # (prompt 05); the bound asserted below is on the *consumer's* remainder, which is
        # unaffected.
        for z in mid[:20]:
            remainder = functions.phase.theta_mod_2pi(z)
            self.assertLessEqual(abs(remainder), TWO_PI)
            self.assertAlmostEqual(sin(remainder), sin(f.theta_exact(z)), delta=1.0e-6)

        thetas = [functions.phase.raw_theta(z) for z in mid]
        # mid is in descending z order, so raw_theta must be ascending as we walk backwards
        self.assertTrue(all(thetas[i] > thetas[i + 1] for i in range(len(thetas) - 1)))

    def test_spline_error_dominates_on_the_production_grid(self):
        """
        On the production grid the reconstruction error is the phase spline's cubic fit error,
        so refining the grid by 3x must improve it by roughly 3^4.
        """
        errors = {}
        for samples in (PRODUCTION_SAMPLES_PER_LOG10Z, REFINED_SAMPLES_PER_LOG10Z):
            f = Fixture(1.0 / 3.0, WKB_samples=samples)
            functions = f.exact_functions()
            mid = log_midpoints(f.z_WKB)
            errors[samples] = max(
                abs(functions.T_WKB(z) - f.M_exact(z) * sin(f.theta_exact(z)))
                / abs(f.M_exact(z))
                for z in mid
            )

        print(
            f"\n[grid refinement] |T_WKB - exact|/envelope: "
            f"{PRODUCTION_SAMPLES_PER_LOG10Z}/decade {errors[PRODUCTION_SAMPLES_PER_LOG10Z]:.3e}, "
            f"{REFINED_SAMPLES_PER_LOG10Z}/decade {errors[REFINED_SAMPLES_PER_LOG10Z]:.3e}"
        )
        # not tightened: this *is* the consumer re-spline floor, which the transfer-remedial
        # campaign does not improve. Measured 6.090e-06 on both the pre-campaign tree and the
        # current one, against 1e-5 -- a factor 1.6, which is as tight as a grid-dependent fit
        # error should be asserted
        self.assertLess(errors[PRODUCTION_SAMPLES_PER_LOG10Z], 1.0e-5)
        self.assertLess(
            errors[REFINED_SAMPLES_PER_LOG10Z],
            errors[PRODUCTION_SAMPLES_PER_LOG10Z] / 10.0,
        )


class TestClosedFormIdentities(unittest.TestCase):
    """
    d ln M/dz and omega() are closed forms; they must agree with what the represented amplitude
    and phase actually do.
    """

    def test_dlnM_dz_matches_finite_difference(self):
        for w in W_VALUES:
            with self.subTest(w=w):
                f = Fixture(w)
                functions = f.LG_functions()

                h = 1.0e-5
                worst = 0.0
                for z in f.z_WKB[3:-3]:
                    log_z = log(1.0 + z)
                    fd = (
                        (
                            log(abs(functions.M(log_z + h, z_is_log=True)))
                            - log(abs(functions.M(log_z - h, z_is_log=True)))
                        )
                        / (2.0 * h)
                        / (1.0 + z)
                    )
                    closed = functions.dlnM_dz(z)
                    worst = max(worst, abs(fd - closed) / abs(closed))

                print(
                    f"\n[LG fixture, w={w:.5g}] max d ln M/dz vs finite difference = {worst:.3e}"
                )
                # not tightened, and nothing here involves bessel_phase: the LG fixture's
                # friction is the exact closed-form integral, so what limits this is the O(h^2)
                # truncation of the central difference at h = 1e-5. Measured 6.355e-11
                # (w = 1/3) and 8.323e-11 (w = 0.2), identical before and after the
                # transfer-remedial campaign
                self.assertLess(worst, 1.0e-6)

    def test_omega_matches_phase_derivative(self):
        for w in W_VALUES:
            with self.subTest(w=w):
                f = Fixture(w)
                functions = f.LG_functions()

                worst = 0.0
                for z in f.z_WKB[3:-3]:
                    worst = max(
                        worst,
                        abs(functions.phase.theta_deriv(z) - functions.omega(z))
                        / abs(functions.omega(z)),
                    )

                print(
                    f"[LG fixture, w={w:.5g}] max omega vs phase.theta_deriv = {worst:.3e}"
                )
                # not tightened: measured 4.249e-08 (w = 1/3) and 2.163e-08 (w = 0.2), the same
                # to four figures before and after the campaign, because `functions.phase` here
                # is a phase_spline through the exact integral of omega_eff and the residual is
                # that spline's derivative error -- a consumer floor, not the Bessel oracle. The
                # factor 24 of headroom at w = 1/3 is the smallest in this class and is grid
                # dependent, so it stays
                self.assertLess(worst, 1.0e-6)

    def test_LG_truncation_error_of_the_exact_envelope(self):
        """
        Documentation, not a physics assertion: on the *exact* fixture the closed-form
        d ln M/dz differs from the exact envelope's logarithmic derivative by the Liouville-Green
        truncation error, which is grid-independent. This is why the identity above is checked on
        the LG fixture instead.
        """
        h = 1.0e-5
        for samples in (PRODUCTION_SAMPLES_PER_LOG10Z, REFINED_SAMPLES_PER_LOG10Z):
            f = Fixture(1.0 / 3.0, WKB_samples=samples)
            functions = f.exact_functions()

            worst = 0.0
            for z in f.z_WKB[3:-3]:
                log_z = log(1.0 + z)
                fd = (
                    (
                        log(abs(functions.M(log_z + h, z_is_log=True)))
                        - log(abs(functions.M(log_z - h, z_is_log=True)))
                    )
                    / (2.0 * h)
                    / (1.0 + z)
                )
                worst = max(
                    worst, abs(fd - functions.dlnM_dz(z)) / abs(functions.dlnM_dz(z))
                )

            print(
                f"[exact envelope, {samples}/decade] max d ln M/dz vs finite difference = {worst:.3e}"
            )
            self.assertLess(worst, 1.0e-3)


class TestConsistencyChecks(unittest.TestCase):
    def test_crossover_inconsistency_raises(self):
        f = Fixture(1.0 / 3.0)
        functions = f.exact_functions()  # the consistent case must build

        self.assertAlmostEqual(
            f.z_exit - f.Tk_numeric.stop_deltaz_subh,
            functions.crossover_z,
            delta=1.0e-9,
        )

        f.Tk_numeric.stop_deltaz_subh = 1.001 * f.Tk_numeric.stop_deltaz_subh
        with self.assertRaises(RuntimeError):
            f.exact_functions()

    def test_nonzero_cos_coeff_raises(self):
        f = Fixture(1.0 / 3.0)
        functions = f.exact_functions()
        values = [
            FakeWKBValue(
                z,
                *WKB_mod_2pi(f.theta_exact(z)),
                friction=0.0,
                H_ratio=1.0,
                omega_WKB_sq=f.omega(z) ** 2,
            )
            for z in f.z_WKB
        ]

        with self.assertRaises(RuntimeError):
            TkSourceFunctions(
                f.model,
                f.k_object,
                f.Tk_numeric,
                FakeTkWKB(values, functions.sin_coeff, f.crossover_z, cos_coeff=1.0e-3),
            )

    def test_WKB_grid_starting_below_the_hand_over(self):
        """
        In production the WKB grid is the source grid truncated at z_init, so its largest sample
        can sit up to one grid step below the hand-over. The object must build, must report the
        sampled range (not the nominal hand-over) as its WKB region, and must refuse to be
        evaluated above it -- `phase_spline` cannot be extrapolated.
        """
        f = Fixture(1.0 / 3.0, drop_first_WKB_sample=True)
        functions = f.exact_functions()

        self.assertLess(functions.WKB_region[0], functions.crossover_z)
        self.assertAlmostEqual(functions.WKB_region[0], f.z_WKB[0], delta=1.0e-9)

        # evaluation at the top of the sampled range is fine
        self.assertTrue(np.isfinite(functions.T_WKB(functions.WKB_region[0])))
        self.assertTrue(np.isfinite(functions.M(functions.WKB_region[0])))

        with self.assertRaises(RuntimeError):
            functions.T_WKB(functions.crossover_z)

        # the numeric region still reaches down to the hand-over
        self.assertAlmostEqual(
            functions.numeric_region[1], functions.crossover_z, delta=1.0e-9
        )


class TestPrimitivePhaseConsumer(unittest.TestCase):
    """
    Prompt 10 of prompts/GkTk-remedial: the phase is no longer a cubic spline of the stored
    theta samples but `-k cs_tau.delta(z_init, z)` from the background model's sound-horizon
    table plus a spline of the small residual, and the friction factor of the amplitude is read
    from the friction table rather than splined.
    """

    def test_growing_phase_is_no_longer_interpolated(self):
        """
        Review section 5 / section 12.6: a cubic spline of theta carries h^4 x_T/384, which is
        the term this prompt removes. Both representations are built here from the *same* stored
        samples and scored against the same exact phase, on the production 100-per-decade grid,
        at x_T = 1e6.
        """
        f = AnalyticRadiationFixture(x_max=1.0e6)
        functions = f.functions()
        stored_spline = f.stored_phase_spline()

        mid = log_midpoints(f.z_WKB)

        primitive_err = max(
            fabs(functions.phase.raw_theta(z) - f.theta_exact(z)) for z in mid
        )
        node_err = max(
            fabs(functions.phase.raw_theta(z) - f.theta_exact(z)) for z in f.z_WKB
        )
        spline_err = max(
            fabs(stored_spline.raw_theta(z) - f.theta_exact(z)) for z in mid
        )

        print(
            f"\n[primitive phase, x_T = {f.x(f.z_WKB[-1]):.3g}] max |theta - exact|: "
            f"PrimitivePhase {primitive_err:.4e} rad (at the stored samples {node_err:.4e} rad); "
            f"cubic spline of the same samples {spline_err:.4e} rad; ratio {spline_err/primitive_err:.4e}"
        )

        # 1e-7 rad is prompt 10 section 3 item 1. It is 859 ulp of the 1e6 rad phase asserted
        # here, so it is comfortably above the representation floor (contrast
        # [09-consumer-threshold-below-representation-floor] on the board). Measured
        # 3.4482e-10 rad, which is 2.96 ulp of 1e6 rad -- i.e. the floor itself, coming from
        # the rounding of `div * TWO_PI` in the stored (cycle, remainder) pair and of the
        # order-1 arithmetic of the leading term.
        self.assertLess(primitive_err, 1.0e-7)

        # the h^4 x_T/384 law: h = ln(10)/100 = 0.02302, so h^4 x_T/384 = 7.3e-4 rad at
        # x_T = 1e6 in the interior, and 7.09e-3 rad is what a not-a-knot cubic actually
        # reaches in the last interval. (Prompt 10 section 3 item 1 quotes ~0.8 rad for this
        # number; that is the value at x_T ~ 1e9, not 1e6 -- see the log.)
        self.assertGreater(spline_err, 1.0e-3)
        self.assertGreater(spline_err / primitive_err, 1.0e5)

    def test_friction_comes_from_the_table(self):
        """
        `friction(z)` must be `friction_F.delta(crossover_z, z)` -- for the constant-w stand-in,
        the closed form (3/2)(1+w) log((1+z)/(1+z_init)) -- with no spline anywhere.
        """
        for w in W_VALUES:
            with self.subTest(w=w):
                f = Fixture(w)
                functions = f.LG_functions()

                one_plus_z_init = 1.0 + f.crossover_z

                def closed_form(z):
                    return 1.5 * (1.0 + w) * log((1.0 + z) / one_plus_z_init)

                probes = list(f.z_WKB) + log_midpoints(f.z_WKB)
                worst = max(
                    fabs(functions.friction(z) - closed_form(z)) for z in probes
                )
                span = max(fabs(closed_form(z)) for z in probes)

                print(
                    f"\n[friction from the table, w={w:.5g}] max |F - closed form| = "
                    f"{worst:.4e} over |F| <= {span:.4g}"
                )
                # prompt 10 section 3 item 2. |F| reaches ~8 here, one ulp of which is 1.8e-15,
                # so 1e-13 is ~56 ulp of the quantity asserted -- above its floor, not below.
                # Measured 1.78e-15 (w = 1/3) and 1.78e-15 (w = 0.2): the closed form is
                # evaluated twice by two different expressions and they agree to rounding.
                self.assertLess(worst, 1.0e-13)

    def test_omega_matches_phase_derivative_from_the_primitive(self):
        """
        `omega()` and `phase.theta_deriv()` are now two closed forms plus, in the second case,
        the derivative of the *residual* spline: the identity the module docstring promises no
        longer has a spline of theta between them.
        """
        for w in W_VALUES:
            with self.subTest(w=w):
                f = Fixture(w)
                functions = f.LG_functions()

                def worst_over(z_values):
                    return max(
                        fabs(functions.phase.theta_deriv(z) - functions.omega(z))
                        / fabs(functions.omega(z))
                        for z in z_values
                    )

                worst = worst_over(f.z_WKB[3:-3])
                interior = worst_over(f.z_WKB[5:-3])

                print(
                    f"\n[primitive phase, w={w:.5g}] max |omega - theta_deriv|/omega = "
                    f"{worst:.4e} over z_WKB[3:-3], {interior:.4e} over z_WKB[5:-3]"
                )
                # Prompt 10 section 3 item 3 asks for 1e-10 relative. That is missed at exactly
                # one abscissa per equation of state -- the third stored sample from the top of
                # the WKB region, where the not-a-knot end condition of the residual spline
                # lives: 1.049e-10 (w = 1/3) and 7.950e-11 (w = 0.2), against 4.249e-08 and
                # 2.163e-08 for the representation this replaces. From the fifth sample inwards
                # it is 5.559e-12 and 3.757e-12. 1e-9 is an order of magnitude of headroom on a
                # grid-dependent end effect; the deeper bound below is the claim that matters.
                # See the log of prompt 10 for the measurement and the alternatives.
                self.assertLess(worst, 1.0e-9)
                self.assertLess(interior, 1.0e-11)

    def test_inconsistent_friction_sample_raises(self):
        """
        The construction-time cross-check is what detects a datastore written by the retired
        friction ODE (~2e-7 absolute in F) being read against a table-built background model.
        """
        f = Fixture(1.0 / 3.0)

        sol = solve_ivp(
            lambda z, y: [f.omega(z)],
            t_span=(f.crossover_z, f.z_WKB[-1]),
            y0=[0.0],
            t_eval=f.z_WKB,
            method="DOP853",
            atol=1.0e-14,
            rtol=1.0e-13,
        )
        assert sol.success, sol.message

        def build(perturbation, index):
            values = []
            for i, z in enumerate(f.z_WKB):
                div_2pi, mod_2pi = WKB_mod_2pi(sol.y[0][i])
                omega = f.omega(z)
                H_ratio = f.model.Hubble(f.crossover_z) / f.model.Hubble(z)
                friction = 1.5 * (1.0 + f.w) * log((1.0 + z) / (1.0 + f.crossover_z))
                if i == index:
                    friction = friction + perturbation
                values.append(
                    FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
                )

            return TkSourceFunctions(
                f.model,
                f.k_object,
                f.Tk_numeric,
                FakeTkWKB(values, 0.37, f.crossover_z),
            )

        # the consistent case must build
        build(0.0, 0)

        # a single sample displaced by the retired ODE's own error must not
        with self.assertRaises(RuntimeError):
            build(2.0e-7, len(f.z_WKB) // 2)

    def test_sign_convention_is_checked(self):
        """
        `TkSourceFunctions` fixes the sign of the leading term explicitly rather than through a
        spline's `increasing` flag, and confirms it against the stored samples: phases carrying
        the opposite convention must be refused rather than splined through a residual twice the
        size of the phase.
        """
        f = Fixture(1.0 / 3.0)

        sol = solve_ivp(
            lambda z, y: [f.omega(z)],
            t_span=(f.crossover_z, f.z_WKB[-1]),
            y0=[0.0],
            t_eval=f.z_WKB,
            method="DOP853",
            atol=1.0e-14,
            rtol=1.0e-13,
        )
        assert sol.success, sol.message

        values = []
        for i, z in enumerate(f.z_WKB):
            div_2pi, mod_2pi = WKB_mod_2pi(-sol.y[0][i])
            omega = f.omega(z)
            H_ratio = f.model.Hubble(f.crossover_z) / f.model.Hubble(z)
            friction = 1.5 * (1.0 + f.w) * log((1.0 + z) / (1.0 + f.crossover_z))
            values.append(
                FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
            )

        with self.assertRaises(RuntimeError):
            TkSourceFunctions(
                f.model,
                f.k_object,
                f.Tk_numeric,
                FakeTkWKB(values, 0.37, f.crossover_z),
            )

    def test_missing_background_tables_are_refused_by_name(self):
        """
        A `ModelFunctions` built before prompt 04 leaves `cs_tau` and `friction_F` at their None
        default. The consumer must say which one is missing, not fail on `None.delta`.
        """
        f = Fixture(1.0 / 3.0)

        for name in ("cs_tau", "friction_F"):
            with self.subTest(missing=name):
                model = FakeModel(f.w)
                model.functions = model.functions._replace(**{name: None})

                values, sin_coeff = [], 0.37
                sol = solve_ivp(
                    lambda z, y: [f.omega(z)],
                    t_span=(f.crossover_z, f.z_WKB[-1]),
                    y0=[0.0],
                    t_eval=f.z_WKB,
                    method="DOP853",
                    atol=1.0e-14,
                    rtol=1.0e-13,
                )
                assert sol.success, sol.message
                for i, z in enumerate(f.z_WKB):
                    div_2pi, mod_2pi = WKB_mod_2pi(sol.y[0][i])
                    omega = f.omega(z)
                    H_ratio = f.model.Hubble(f.crossover_z) / f.model.Hubble(z)
                    friction = (
                        1.5 * (1.0 + f.w) * log((1.0 + z) / (1.0 + f.crossover_z))
                    )
                    values.append(
                        FakeWKBValue(
                            z, div_2pi, mod_2pi, friction, H_ratio, omega * omega
                        )
                    )

                with self.assertRaises(RuntimeError) as caught:
                    TkSourceFunctions(
                        model,
                        f.k_object,
                        f.Tk_numeric,
                        FakeTkWKB(values, sin_coeff, f.crossover_z),
                    )
                self.assertIn(name, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
