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
    so `TkSourceFunctions.M` must reproduce it to spline error only. This fixture tests the
    assembly of the amplitude and the re-splining of the phase.

  * fixture "LG": `friction` is the exact Liouville-Green friction integral
    F = (3/2)(1+w) log((1+z)/(1+z_init)) (the integral of `TkWKBIntegration.friction_RHS`) and
    theta is the exact integral of omega_eff. This fixture is the one on which the closed-form
    identities can be checked, because it is the code's own LG representation: d ln M/dz must
    equal a finite difference of log M, and omega() must equal phase.theta_deriv().

They differ because the exact Bessel envelope is *not* the Liouville-Green amplitude: they
differ by the LG truncation error, ~1e-5 in d ln M/dz at 3.5 e-folds sub-horizon (measured
below, and grid-independent), which is far above the 1e-6 the identity check needs.
"""

import unittest
from math import log, exp, sqrt, sin, pi, gamma

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv

from ComputeTargets.BackgroundModel import ModelFunctions
from ComputeTargets.TkSourceFunctions import TkSourceFunctions
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from LiouvilleGreen.WKBtools import wrap_theta
from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.constants import TWO_PI

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
        )

    def Hubble(self, z):
        return self.H0 * (1.0 + z) ** self.p

    def tau(self, z):
        return 1.0 / (self.H0 * (self.p - 1.0) * (1.0 + z) ** (self.p - 1.0))


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
        m = sqrt(J^2 + Y^2) supplied by bessel_phase (bessel_phase.py:270-282: J = m sin theta).
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
        """
        return pi - self._bessel["phase"].raw_theta(self.x(z))

    # --- the two WKB fixtures ----------------------------------------------------------

    def exact_functions(self):
        """
        `TkSourceFunctions` whose amplitude is the exact Bessel envelope and whose phase is the
        exact Bessel phase. `sin_coeff` is backed out from the exact amplitude at the hand-over
        (where H_ratio = 1 and F = 0, exactly as TkWKBIntegration.store() arranges), and the
        stored friction samples are then whatever makes the code's amplitude formula exact.
        """
        z_init = self.z_WKB[0]
        sin_coeff = self.M_exact(z_init) * sqrt(self.omega(z_init))

        values = []
        for z in self.z_WKB:
            div_2pi, mod_2pi = wrap_theta(self.theta_exact(z))
            omega = self.omega(z)
            H_ratio = self.model.Hubble(self.crossover_z) / self.model.Hubble(z)
            friction = log(self.M_exact(z) * sqrt(omega) / (sin_coeff * sqrt(H_ratio)))
            values.append(
                FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
            )

        return TkSourceFunctions(
            self.model,
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
            div_2pi, mod_2pi = wrap_theta(sol.y[0][i])
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
                # for the record: the same comparison against scipy's J_nu, which additionally
                # carries bessel_phase's own phase-function error and so saturates near 2e-6
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
                self.assertLess(err_M, 1.0e-8)
                self.assertLess(err_T, 1.0e-7)

    def test_phase_convention(self):
        f = Fixture(1.0 / 3.0)
        functions = f.exact_functions()

        mid = log_midpoints(f.z_WKB)
        # phase_spline rebases each chunk by an integer number of cycles, so the remainder it
        # returns is congruent to theta mod 2pi but does not carry the stored samples' negative
        # sign convention. What must hold is that it is a remainder, and that its sine is the
        # sine of the exact phase.
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
                *wrap_theta(f.theta_exact(z)),
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


if __name__ == "__main__":
    unittest.main()
