"""
Tests for ComputeTargets/phase_groups.py, the phase-group decomposition of the source integrand
G_k(z, z') f(z' | q, r) / H(z')^2.

Two independent oracles, as prompts/source-remediation/07-phase-group-algebra.md section 6
requires, plus the phase-composition and region-boundary checks:

  * Oracle 1 -- pointwise identity against the direct product. `evaluate_sum` is compared with
    G * source_function(T_q, T_r, T_q', T_r') / H^2, the T_i and T_i' being reconstructed from
    the very same (M_i, d ln M_i/dz, theta_i, omega_i) accessors the module reads. This checks
    the algebra and the sign bookkeeping only, and is independent of any Bessel function: it
    holds for whatever callables the factor objects supply.

  * Oracle 2 -- the exact Bessel fixture. For constant w the transfer function is
    T = 2^nu Gamma(5/2+b) x^-nu J_nu(x), nu = 3/2 + b, x = q c_s a0 eta, and the Green's function
    is G_code(z, z') = H(z') (pi/2) sqrt(eta eta') m(k eta) m(k eta') sin(vartheta(k eta') -
    vartheta(k eta)) with J_(1/2+b) = m sin vartheta, Y_(1/2+b) = -m cos vartheta. The module,
    fed exact amplitude-phase decompositions, is compared with compute_analytic_G *
    source_function(analytic T) / H^2, which never sees an amplitude or a phase. Two flavours:

      - "exact" stand-ins built directly on scipy.special (m = sqrt(J^2+Y^2), vartheta =
        atan2(J, -Y), derivatives from the Bessel recurrences and the Wronskian). These are exact
        to rounding, so the comparison tests the module and the conventions at the 1e-8 the
        prompt asks for (and reaches ~1e-13).
      - "realistic" fixtures: real TkSourceFunctions objects from prompt 05's exact fixture
        (bessel_phase amplitude and phase, re-splined on the production grid) and a
        Green's-function stand-in whose phase is a real phase_spline through bessel_phase
        samples. This carries bessel_phase's own accuracy (~x * 1e-8 in phase,
        docs/lg-phase-and-handover-followup-2026-09.md section 2.4) and the phase re-spline
        error, so it cannot reach 1e-8; its measured floor is what prompt 08 needs as an
        acceptance threshold and is printed.

Stand-in background and transfer-function fixtures are imported from test_tk_source_functions.
"""

import unittest
from math import log, exp, sqrt, sin, cos, pi, gamma, atan2, fmod

import mpmath
import numpy as np
from scipy.special import jv, yv

from ComputeTargets.QuadSource import source_function
from ComputeTargets.analytic_Gk import compute_analytic_G
from ComputeTargets.analytic_Tk import compute_analytic_T, compute_analytic_Tprime
from ComputeTargets.phase_groups import (
    PhaseGroup,
    build_phase_groups,
    evaluate_sum,
    evaluate_envelope,
    group_signs,
    signs_label,
)
from ComputeTargets.TkSourceFunctions import TkSourceFunctions
from ComputeTargets.WKB_Tk import Tk_d_ln_omegaEff_dz
from ComputeTargets.tests.test_tk_source_functions import (
    Fixture,
    FakeModel,
    FakeWKBValue,
    FakeTkWKB,
    log_grid,
    W_VALUES,
    PRODUCTION_SAMPLES_PER_LOG10Z,
    REFINED_SAMPLES_PER_LOG10Z,
)
from scipy.integrate import solve_ivp
from LiouvilleGreen.WKBtools import wrap_theta
from LiouvilleGreen.bessel_phase import bessel_phase
from LiouvilleGreen.constants import TWO_PI
from LiouvilleGreen.phase_spline import phase_spline

# wavenumbers (k/a0, H0 = 1 units) used throughout: q < r so that T_r hands over to its LG
# representation at a higher redshift than T_q, giving a populated "T_r oscillatory, T_q numeric"
# region of log(2) = 0.69 e-folds; k_G satisfies the triangle condition |q - r| <= k <= q + r
Q_WAVENUMBER = 1.0e4
R_WAVENUMBER = 2.0e4
K_WAVENUMBER = 1.2e4

ALL_REGIMES = [
    (True, True, True),
    (True, True, False),
    (True, False, True),
    (True, False, False),
    (False, True, True),
    (False, True, False),
    (False, False, True),
]

N_RANDOM = 200
SEED = 20260908


def regime_name(regime):
    return "G" * regime[0] + "q" * regime[1] + "r" * regime[2] or "-"


# =============================================================================================
# exact (scipy-based) stand-ins
# =============================================================================================


def bessel_modulus_phase(nu: float, x: float):
    """
    (m, vartheta, dvartheta/dx, d ln m/dx) with J_nu = m sin vartheta, Y_nu = -m cos vartheta,
    vartheta unwrapped so that it is continuous and increasing in x. The cycle count is fixed
    from the large-x asymptotic vartheta ~ x - (nu/2 - 1/4) pi + (4 nu^2 - 1)/(8x), whose
    remainder is O(x^-3) and far below pi for every x used here; dvartheta/dx = 2/(pi x m^2) is
    the Wronskian J Y' - J' Y = 2/(pi x).
    """
    J = jv(nu, x)
    Y = yv(nu, x)
    m_sq = J * J + Y * Y
    principal = atan2(J, -Y)
    asymptotic = x - (0.5 * nu - 0.25) * pi + (4.0 * nu * nu - 1.0) / (8.0 * x)
    cycles = round((asymptotic - principal) / TWO_PI)
    vartheta = principal + cycles * TWO_PI
    assert abs(vartheta - asymptotic) < 1.0, (nu, x, vartheta, asymptotic)

    Jp = jv(nu - 1.0, x) - (nu / x) * J
    Yp = yv(nu - 1.0, x) - (nu / x) * Y
    dln_m_dx = (J * Jp + Y * Yp) / m_sq
    dvartheta_dx = 2.0 / (pi * x * m_sq)
    return sqrt(m_sq), vartheta, dvartheta_dx, dln_m_dx


class _ExactPhase:
    """
    A phase_spline look-alike whose raw_theta/theta_mod_2pi/theta_deriv are exact callables of
    log(1+z). `theta(u)` returns (raw, remainder, d theta/d log(1+z)).
    """

    def __init__(self, theta):
        self._theta = theta

    def raw_theta(self, x, x_is_log=False):
        assert x_is_log
        return self._theta(x)[0]

    def theta_mod_2pi(self, x, x_is_log=False):
        assert x_is_log
        return self._theta(x)[1]

    def theta_deriv(self, x, x_is_log=False, log_derivative=False):
        assert x_is_log
        deriv = self._theta(x)[2]
        if log_derivative:
            return deriv
        return deriv / exp(x)


class ExactTk:
    """
    Exact stand-in for TkSourceFunctions on a constant-w background: the numeric accessors are
    the analytic T, dT/dz everywhere; the LG accessors are the exact amplitude-phase
    decomposition T = M sin theta with M = 2^nu Gamma(5/2+b) x^-nu m(x), theta = pi - vartheta(x)
    (so that theta increases with z, the code's convention, and sin theta = sin vartheta).
    No range restrictions: every accessor is valid at every z.
    """

    def __init__(self, k, w, model: FakeModel):
        self.k = k
        self.w = w
        self.model = model
        self.b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
        self.cs = sqrt(w)
        self.nu = 1.5 + self.b
        self._norm = 2.0**self.nu * gamma(2.5 + self.b)
        self.phase = _ExactPhase(self._theta)

    def x(self, z):
        return self.k * self.cs * self.model.tau(z)

    def dx_dz(self, z):
        # tau ~ (1+z)^-(p-1)
        return -(self.model.p - 1.0) * self.x(z) / (1.0 + z)

    @staticmethod
    def _z(x, z_is_log):
        return exp(x) - 1.0 if z_is_log else x

    # numeric-region protocol
    def T(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        return compute_analytic_T(self.k, self.w, self.model.tau(z))

    def dT_dz(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        return compute_analytic_Tprime(
            self.k, self.w, self.model.tau(z), self.model.Hubble(z)
        )

    # LG protocol
    def M(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        xx = self.x(z)
        m, _, _, _ = bessel_modulus_phase(self.nu, xx)
        return self._norm * xx ** (-self.nu) * m

    def dlnM_dz(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        xx = self.x(z)
        _, _, _, dln_m_dx = bessel_modulus_phase(self.nu, xx)
        return (-self.nu / xx + dln_m_dx) * self.dx_dz(z)

    def omega(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        xx = self.x(z)
        _, _, dvartheta_dx, _ = bessel_modulus_phase(self.nu, xx)
        # theta = pi - vartheta
        return -dvartheta_dx * self.dx_dz(z)

    def _theta(self, log_z):
        z = exp(log_z) - 1.0
        xx = self.x(z)
        _, vartheta, dvartheta_dx, _ = bessel_modulus_phase(self.nu, xx)
        raw = pi - vartheta
        remainder = pi - atan2(jv(self.nu, xx), -yv(self.nu, xx))
        deriv_log = -dvartheta_dx * self.dx_dz(z) * (1.0 + z)
        return raw, remainder, deriv_log


class ExactGk:
    """
    Exact stand-in for GkSourceFunctions at fixed response redshift z_resp:
    numeric_Gk is compute_analytic_G; the LG form is
        A_G(z') = H(z') (pi/2) sqrt(eta eta') m(k eta) m(k eta'),
        theta_G(z') = vartheta(k eta') - vartheta(k eta),
    with (m, vartheta) of order 1/2 + b (README section 2(c)); asserted against compute_analytic_G
    by TestBesselOracle.test_LG_form_of_the_Greens_function.
    """

    def __init__(self, k, w, z_resp, model: FakeModel):
        self.k = k
        self.w = w
        self.model = model
        self.b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
        self.nu = 0.5 + self.b
        self.z_resp = z_resp
        self.tau_resp = model.tau(z_resp)
        m_resp, vartheta_resp, _, _ = bessel_modulus_phase(self.nu, k * self.tau_resp)
        self._m_resp = m_resp
        self._vartheta_resp = vartheta_resp
        self._principal_resp = atan2(
            jv(self.nu, k * self.tau_resp), -yv(self.nu, k * self.tau_resp)
        )
        self.phase = _ExactPhase(self._theta)

    @staticmethod
    def _z(x, z_is_log):
        return exp(x) - 1.0 if z_is_log else x

    def numeric_Gk(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        return compute_analytic_G(
            self.k, self.w, self.model.tau(z), self.tau_resp, self.model.Hubble(z)
        )

    def sin_amplitude(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        tau = self.model.tau(z)
        m, _, _, _ = bessel_modulus_phase(self.nu, self.k * tau)
        return (
            self.model.Hubble(z)
            * (pi / 2.0)
            * sqrt(tau * self.tau_resp)
            * m
            * self._m_resp
        )

    def _theta(self, log_z):
        z = exp(log_z) - 1.0
        tau = self.model.tau(z)
        xx = self.k * tau
        _, vartheta, dvartheta_dx, _ = bessel_modulus_phase(self.nu, xx)
        raw = vartheta - self._vartheta_resp
        remainder = atan2(jv(self.nu, xx), -yv(self.nu, xx)) - self._principal_resp
        # d vartheta / d log(1+z) = dvartheta/dx * dx/dz * (1+z), dx/dz = -(p-1) x/(1+z)
        deriv_log = -dvartheta_dx * (self.model.p - 1.0) * xx
        return raw, remainder, deriv_log


# =============================================================================================
# realistic Green's-function fixture: bessel_phase + a real phase_spline
# =============================================================================================


class BesselPhaseGk:
    """
    GkSourceFunctions stand-in built the way GkSourcePolicyData._create_functions builds the
    real one: the phase is a phase_spline over log(1+z') through (div 2pi, mod 2pi) samples,
    chunk_logstep=125, increasing=False (theta_G decreases with the source redshift); the
    amplitude is bessel_phase's modulus. numeric_Gk is compute_analytic_G.
    """

    def __init__(self, k, w, z_resp, model: FakeModel, z_grid):
        self.k = k
        self.w = w
        self.model = model
        self.b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)
        self.nu = 0.5 + self.b
        self.z_resp = z_resp
        self.tau_resp = model.tau(z_resp)

        self._bessel = bessel_phase(self.nu, 1.02 * k * self.tau_resp)
        self._m_resp = self._bessel["mod"](k * self.tau_resp)
        self._vartheta_resp = self._bessel["phase"].raw_theta(k * self.tau_resp)

        samples = []
        for z in z_grid:
            theta = (
                self._bessel["phase"].raw_theta(k * model.tau(z)) - self._vartheta_resp
            )
            div, mod = wrap_theta(theta)
            samples.append((log(1.0 + z), div, mod))
        samples.sort(key=lambda s: s[0])
        log_x, div_2pi, mod_2pi = zip(*samples)
        self.phase = phase_spline(
            list(log_x),
            list(div_2pi),
            list(mod_2pi),
            x_is_log=True,
            x_is_redshift=True,
            chunk_step=None,
            chunk_logstep=125,
            increasing=False,
        )
        self.WKB_region = (max(z_grid), min(z_grid))

    @staticmethod
    def _z(x, z_is_log):
        return exp(x) - 1.0 if z_is_log else x

    def numeric_Gk(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        return compute_analytic_G(
            self.k, self.w, self.model.tau(z), self.tau_resp, self.model.Hubble(z)
        )

    def sin_amplitude(self, x, z_is_log=False):
        z = self._z(x, z_is_log)
        tau = self.model.tau(z)
        return (
            self.model.Hubble(z)
            * (pi / 2.0)
            * sqrt(tau * self.tau_resp)
            * self._bessel["mod"](self.k * tau)
            * self._m_resp
        )


# =============================================================================================
# a complete test configuration for one equation of state
# =============================================================================================


class Case:
    """
    Everything the tests need for one w: the background, exact stand-ins for G, T_q, T_r, the
    realistic TkSourceFunctions fixtures from prompt 05, a realistic G fixture, and the
    redshift ranges of each regime of the phase-group table.
    """

    _cache = {}

    @classmethod
    def get(cls, w, samples=PRODUCTION_SAMPLES_PER_LOG10Z):
        key = (w, samples)
        if key not in cls._cache:
            cls._cache[key] = cls(w, samples)
        return cls._cache[key]

    def __init__(self, w, samples):
        self.w = w
        self.model = FakeModel(w)
        self.b = (1.0 - 3.0 * w) / (1.0 + 3.0 * w)

        # realistic fixtures (prompt 05): numeric 5 e-folds super-horizon to 3.5 e-folds
        # sub-horizon, LG from there down to x = 1e3
        self.Fq = Fixture(
            w, k=Q_WAVENUMBER, WKB_samples=samples, numeric_samples=samples
        )
        self.Fr = Fixture(
            w, k=R_WAVENUMBER, WKB_samples=samples, numeric_samples=samples
        )
        self.Tq_real = self.Fq.exact_functions()
        self.Tr_real = self.Fr.exact_functions()

        # response redshift below every region
        lowest = min(self.Tq_real.WKB_region[1], self.Tr_real.WKB_region[1])
        self.z_resp = 0.95 * (1.0 + lowest) - 1.0

        # exact stand-ins
        self.Tq = ExactTk(Q_WAVENUMBER, w, self.model)
        self.Tr = ExactTk(R_WAVENUMBER, w, self.model)
        self.Gk = ExactGk(K_WAVENUMBER, w, self.z_resp, self.model)

        # regime ranges (z_max, z_min) on the realistic fixtures
        self.both_WKB = (
            min(self.Tq_real.WKB_region[0], self.Tr_real.WKB_region[0]),
            max(self.Tq_real.WKB_region[1], self.Tr_real.WKB_region[1]),
        )
        self.r_only = (self.Tr_real.WKB_region[0], self.Tq_real.numeric_region[1])
        # both numeric, with G in its LG form: start where x_G = 3
        x_G_of = lambda z: K_WAVENUMBER * self.model.tau(z)
        one_plus_z_xG3 = (
            K_WAVENUMBER / (self.model.H0 * (self.model.p - 1.0)) / 3.0
        ) ** (1.0 / (self.model.p - 1.0))
        self.both_numeric = (
            min(one_plus_z_xG3 - 1.0, self.Tq_real.numeric_region[0]),
            self.Tr_real.numeric_region[1],
        )
        assert x_G_of(self.both_numeric[0]) >= 3.0 * (1.0 - 1e-9)
        assert self.both_numeric[0] > self.both_numeric[1]
        assert self.r_only[0] > self.r_only[1]
        assert self.both_WKB[0] > self.both_WKB[1]

        # realistic G fixture: phase spline over the whole span the tests use
        z_grid = log_grid(1.0 + self.both_numeric[0], 1.0 + lowest, samples)
        self.Gk_real = BesselPhaseGk(K_WAVENUMBER, w, self.z_resp, self.model, z_grid)

        self.rng = np.random.default_rng(SEED)

    # --- helpers -------------------------------------------------------------------------

    def random_log_z(self, z_range, n=N_RANDOM, margin=1e-3):
        """n log(1+z) values log-uniform inside (z_max, z_min), keeping a small margin."""
        hi = log(1.0 + z_range[0])
        lo = log(1.0 + z_range[1])
        width = hi - lo
        return self.rng.uniform(lo + margin * width, hi - margin * width, n)

    def range_for(self, regime):
        G_osc, q_osc, r_osc = regime
        if q_osc and r_osc:
            return self.both_WKB
        if r_osc:
            return self.r_only
        if q_osc:
            raise ValueError(
                "T_q oscillatory with T_r numeric is unpopulated for q < r"
            )
        return self.both_numeric

    def groups(self, regime, exact=True):
        if exact:
            Gk, Tq, Tr = self.Gk, self.Tq, self.Tr
        else:
            Gk, Tq, Tr = self.Gk_real, self.Tq_real, self.Tr_real
        if not regime[0]:
            Gk = Gk.numeric_Gk
        return build_phase_groups(
            regime,
            Gk=Gk,
            Tq=Tq,
            Tr=Tr,
            model_functions=self.model.functions,
            w_background=self.model.functions.wBackground,
        )

    def direct_product(self, regime, log_z, exact=True):
        """
        G f / H^2 with each factor reconstructed from the SAME accessors the module reads:
        T_i = M_i sin theta_i, dT_i/dz = M_i (d ln M_i/dz) sin theta_i + M_i omega_i cos theta_i
        for an oscillatory factor, the numeric accessors for a smooth one, G = A_G sin theta_G or
        numeric_Gk. This is Oracle 1's reference.
        """
        G_osc, q_osc, r_osc = regime
        if exact:
            Gk, Tq, Tr = self.Gk, self.Tq, self.Tr
        else:
            Gk, Tq, Tr = self.Gk_real, self.Tq_real, self.Tr_real
        z = exp(log_z) - 1.0

        def T_pair(Tk, osc):
            if not osc:
                return Tk.T(log_z, z_is_log=True), Tk.dT_dz(log_z, z_is_log=True)
            M = Tk.M(log_z, z_is_log=True)
            th = Tk.phase.theta_mod_2pi(log_z, x_is_log=True)
            return (
                M * sin(th),
                M * Tk.dlnM_dz(log_z, z_is_log=True) * sin(th)
                + M * Tk.omega(log_z, z_is_log=True) * cos(th),
            )

        Tq_val, Tq_prime = T_pair(Tq, q_osc)
        Tr_val, Tr_prime = T_pair(Tr, r_osc)
        if G_osc:
            G = Gk.sin_amplitude(log_z, z_is_log=True) * sin(
                Gk.phase.theta_mod_2pi(log_z, x_is_log=True)
            )
        else:
            G = Gk.numeric_Gk(log_z, z_is_log=True)

        H = self.model.Hubble(z)
        f = source_function(Tq_val, Tr_val, Tq_prime, Tr_prime, z, self.w)["source"]
        return G * f / (H * H)

    def term_envelope(self, regime, log_z, exact=True):
        """
        |G| (alpha |T_q||T_r| + beta (|DT_q||T_r| + |DT_r||T_q| + |DT_q||DT_r|)) / H^2, with
        |T_i| <= |M_i| and |DT_i| <= |a_i| + |b_i| for an oscillatory factor and |G| <= |A_G|: the
        sum of the magnitudes of the kernel's terms. A residual normalised by this cannot be
        inflated by a zero of f, which the group envelope alone does not protect against in the
        regimes where f itself is the smooth amplitude.
        """
        G_osc, q_osc, r_osc = regime
        if exact:
            Gk, Tq, Tr = self.Gk, self.Tq, self.Tr
        else:
            Gk, Tq, Tr = self.Gk_real, self.Tq_real, self.Tr_real
        z = exp(log_z) - 1.0
        one_plus_z = 1.0 + z

        def bounds(Tk, osc):
            if not osc:
                return abs(Tk.T(log_z, z_is_log=True)), one_plus_z * abs(
                    Tk.dT_dz(log_z, z_is_log=True)
                )
            M = abs(Tk.M(log_z, z_is_log=True))
            return M, one_plus_z * M * (
                abs(Tk.dlnM_dz(log_z, z_is_log=True))
                + abs(Tk.omega(log_z, z_is_log=True))
            )

        Tq_b, DTq_b = bounds(Tq, q_osc)
        Tr_b, DTr_b = bounds(Tr, r_osc)
        G_b = abs(
            Gk.sin_amplitude(log_z, z_is_log=True)
            if G_osc
            else Gk.numeric_Gk(log_z, z_is_log=True)
        )
        alpha = (5.0 + 3.0 * self.w) / (3.0 * (1.0 + self.w))
        beta = 2.0 / (3.0 * (1.0 + self.w))
        H = self.model.Hubble(z)
        return (
            G_b
            * (
                alpha * Tq_b * Tr_b
                + beta * (DTq_b * Tr_b + DTr_b * Tq_b + DTq_b * DTr_b)
            )
            / (H * H)
        )

    def scipy_oracle(self, log_z):
        """
        compute_analytic_G * source_function(analytic T) / H^2, using nothing but scipy's
        J_nu, Y_nu: Oracle 2's reference. It knows nothing about amplitudes or phases.
        """
        z = exp(log_z) - 1.0
        tau = self.model.tau(z)
        H = self.model.Hubble(z)
        G = compute_analytic_G(
            K_WAVENUMBER, self.w, tau, self.model.tau(self.z_resp), H
        )
        f = source_function(
            compute_analytic_T(Q_WAVENUMBER, self.w, tau),
            compute_analytic_T(R_WAVENUMBER, self.w, tau),
            compute_analytic_Tprime(Q_WAVENUMBER, self.w, tau, H),
            compute_analytic_Tprime(R_WAVENUMBER, self.w, tau, H),
            z,
            self.w,
        )["source"]
        return G * f / (H * H)


def max_relative_residual(groups, reference, log_z_values, extra_scale=None):
    """
    max over points of |evaluate_sum - reference| / max(|reference|, envelope, extra_scale),
    where envelope is the sum over groups of sqrt(f_sin^2 + f_cos^2) -- the scale of the
    oscillation being decomposed, so that a residual near a zero of the integrand is not
    amplified -- and extra_scale (optional callable of log_z) is a further scale such as
    Case.term_envelope.
    """
    worst = 0.0
    for u in log_z_values:
        value = evaluate_sum(groups, u)
        ref = reference(u)
        scale = max(abs(ref), evaluate_envelope(groups, u))
        if extra_scale is not None:
            scale = max(scale, extra_scale(u))
        worst = max(worst, abs(value - ref) / scale)
    return worst


def matched_LG_functions(fixture: Fixture) -> TkSourceFunctions:
    """
    A TkSourceFunctions built from the code's own Liouville-Green ingredients AND matched to the
    exact T, dT/dz at the hand-over exactly as TkWKBIntegration.store() does
    (TkWKBIntegration.py:436-462): raw sin/cos coefficients from (T_init, T'_init), the phase
    shifted by deltaTheta = atan2(raw_cos, raw_sin) so that only a sine survives, and
    sin_coeff = +-B. Unlike Fixture.LG_functions(), whose phase starts at zero with an arbitrary
    sin_coeff and which exists only for the closed-form identity checks, this is the LG
    representation production would carry for this transfer function: it differs from the exact
    solution by the LG truncation error and by nothing else.
    """
    model = fixture.model
    k = fixture.k
    z_init = fixture.crossover_z
    one_plus_z_init = 1.0 + z_init

    T_init = fixture.T_exact(z_init)
    Tprime_init = fixture.Tprime_exact(z_init)
    omega_init = fixture.omega(z_init)
    sqrt_omega_init = sqrt(omega_init)
    d_ln_omega_init = Tk_d_ln_omegaEff_dz(model, k, z_init)
    eps_init = model.functions.epsilon(z_init)
    cs2_init = model.functions.wPerturbations(z_init)

    raw_cos_coeff = sqrt_omega_init * T_init
    raw_sin_coeff = (
        Tprime_init
        + (T_init / 2.0)
        * (d_ln_omega_init + (eps_init - 3.0 * (1.0 + cs2_init)) / one_plus_z_init)
    ) / sqrt_omega_init
    deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
    B = sqrt(raw_cos_coeff * raw_cos_coeff + raw_sin_coeff * raw_sin_coeff)
    sgn = (+1 if sin(deltaTheta) >= 0.0 else -1) * (+1 if T_init >= 0.0 else -1)
    sin_coeff = sgn * B

    sol = solve_ivp(
        lambda z, y: [fixture.omega(z)],
        t_span=(z_init, fixture.z_WKB[-1]),
        y0=[deltaTheta],
        t_eval=fixture.z_WKB,
        method="DOP853",
        atol=1.0e-14,
        rtol=1.0e-13,
    )
    assert sol.success, sol.message

    values = []
    for i, z in enumerate(fixture.z_WKB):
        div_2pi, mod_2pi = wrap_theta(sol.y[0][i])
        omega = fixture.omega(z)
        H_ratio = model.Hubble(z_init) / model.Hubble(z)
        friction = 1.5 * (1.0 + fixture.w) * log((1.0 + z) / one_plus_z_init)
        values.append(
            FakeWKBValue(z, div_2pi, mod_2pi, friction, H_ratio, omega * omega)
        )

    return TkSourceFunctions(
        model,
        fixture.k_object,
        fixture.Tk_numeric,
        FakeTkWKB(values, sin_coeff, z_init),
    )


# =============================================================================================
# API and structure
# =============================================================================================


class TestStructure(unittest.TestCase):
    def test_all_smooth_regime_raises(self):
        case = Case.get(1.0 / 3.0)
        with self.assertRaises(ValueError):
            case.groups((False, False, False))

    def test_group_counts_labels_and_signs(self):
        expected = {
            (True, True, True): ["G+q+r", "G+q-r", "G-q+r", "G-q-r"],
            (True, True, False): ["G+q", "G-q"],
            (True, False, True): ["G+r", "G-r"],
            (True, False, False): ["G"],
            (False, True, True): ["q+r", "q-r"],
            (False, True, False): ["q"],
            (False, False, True): ["r"],
        }
        case = Case.get(1.0 / 3.0)
        for regime, labels in expected.items():
            with self.subTest(regime=regime_name(regime)):
                groups = case.groups(regime)
                self.assertEqual([g.label for g in groups], labels)
                self.assertEqual([signs_label(s) for s in group_signs(regime)], labels)
                for g in groups:
                    self.assertIsInstance(g, PhaseGroup)
                    # a smooth factor has sign 0, an oscillatory one +-1
                    for flag, s in zip(regime, g.signs):
                        self.assertEqual(flag, s != 0)
                    self.assertEqual(
                        set(g.levin_theta().keys()),
                        {"theta", "theta_mod_2pi", "theta_deriv"},
                    )
                    self.assertEqual(
                        set(g.levin_theta(include_deriv=False).keys()),
                        {"theta", "theta_mod_2pi"},
                    )
        # 2^(n-1) groups for n oscillatory factors
        for regime in ALL_REGIMES:
            n = sum(regime)
            self.assertEqual(len(group_signs(regime)), 2 ** (n - 1))

    def test_G_only_regime_is_the_current_Levin_integrand(self):
        """
        With both T smooth the single group must be f_sin = A_G f / H^2, f_cos = 0, the
        integrand WKB_Levin_integral hands to the driver today (QuadSourceIntegral.py:1099-1105).
        """
        for w in W_VALUES:
            case = Case.get(w)
            (group,) = case.groups((True, False, False))
            worst = 0.0
            for u in case.random_log_z(case.both_numeric, n=50):
                z = exp(u) - 1.0
                H = case.model.Hubble(z)
                f = source_function(
                    case.Tq.T(u, z_is_log=True),
                    case.Tr.T(u, z_is_log=True),
                    case.Tq.dT_dz(u, z_is_log=True),
                    case.Tr.dT_dz(u, z_is_log=True),
                    z,
                    w,
                )["source"]
                expected = case.Gk.sin_amplitude(u, z_is_log=True) * f / (H * H)
                self.assertEqual(group.f_cos(u), 0.0)
                # normalised by the sum of the kernel's term magnitudes, not by f itself, so a
                # zero of f does not inflate rounding into a false failure
                scale = max(abs(expected), case.term_envelope((True, False, False), u))
                worst = max(worst, abs(group.f_sin(u) - expected) / scale)
                # and the phase is theta_G alone
                self.assertEqual(
                    group.theta(u), case.Gk.phase.raw_theta(u, x_is_log=True)
                )
            print(f"\n[G-only regime, w={w:.5g}] f_sin vs A_G f/H^2: {worst:.3e}")
            self.assertLess(worst, 1.0e-14)

    def test_smooth_G_accepts_callable_or_object(self):
        case = Case.get(1.0 / 3.0)
        u = case.random_log_z(case.both_WKB, n=1)[0]
        by_callable = build_phase_groups(
            (False, True, True),
            Gk=case.Gk.numeric_Gk,
            Tq=case.Tq,
            Tr=case.Tr,
            model_functions=case.model.functions,
            w_background=case.model.functions.wBackground,
        )
        by_object = build_phase_groups(
            (False, True, True),
            Gk=case.Gk,
            Tq=case.Tq,
            Tr=case.Tr,
            model_functions=case.model.functions,
            w_background=case.model.functions.wBackground,
        )
        self.assertEqual(evaluate_sum(by_callable, u), evaluate_sum(by_object, u))

    def test_protocol_violations_raise(self):
        case = Case.get(1.0 / 3.0)
        kwargs = dict(
            model_functions=case.model.functions,
            w_background=case.model.functions.wBackground,
        )
        # an oscillatory G needs sin_amplitude and phase
        with self.assertRaises(TypeError):
            build_phase_groups(
                (True, True, True),
                Gk=case.Gk.numeric_Gk,
                Tq=case.Tq,
                Tr=case.Tr,
                **kwargs,
            )
        # an oscillatory T needs the LG accessors
        with self.assertRaises(TypeError):
            build_phase_groups(
                (True, True, True), Gk=case.Gk, Tq=object(), Tr=case.Tr, **kwargs
            )
        # a smooth G must be callable or expose numeric_Gk
        with self.assertRaises(TypeError):
            build_phase_groups(
                (False, True, True), Gk=object(), Tq=case.Tq, Tr=case.Tr, **kwargs
            )
        with self.assertRaises(ValueError):
            build_phase_groups(
                (True, True), Gk=case.Gk, Tq=case.Tq, Tr=case.Tr, **kwargs
            )

    def test_numpy_scalar_abscissae(self):
        """The Levin driver hands numpy floats; the cached evaluation must accept them."""
        case = Case.get(1.0 / 3.0)
        groups = case.groups((True, True, True))
        u = case.random_log_z(case.both_WKB, n=1)[0]
        self.assertEqual(
            evaluate_sum(groups, np.float64(u)), evaluate_sum(groups, float(u))
        )


# =============================================================================================
# Oracle 1: pointwise identity against the direct product
# =============================================================================================


class TestOracle1DirectProduct(unittest.TestCase):
    TOLERANCE = 1.0e-12

    def test_exact_stand_ins_every_regime(self):
        """
        All seven oscillatory regimes, on exact stand-ins that are valid at every z, so that even
        the regimes unpopulated by the realistic fixtures (T_q oscillatory with T_r numeric) are
        checked. Points are log-uniform over the both-LG range of the realistic fixtures.
        """
        table = {}
        for w in W_VALUES:
            case = Case.get(w)
            for regime in ALL_REGIMES:
                with self.subTest(w=w, regime=regime_name(regime)):
                    groups = case.groups(regime, exact=True)
                    log_z = case.random_log_z(case.both_WKB)
                    worst = max_relative_residual(
                        groups,
                        lambda u: case.direct_product(regime, u, exact=True),
                        log_z,
                        extra_scale=lambda u: case.term_envelope(regime, u, exact=True),
                    )
                    table[(w, regime)] = worst
                    self.assertLess(worst, self.TOLERANCE)
        print(
            "\n[Oracle 1, exact stand-ins] max |sum - G f/H^2| / max(|G f/H^2|, envelope):"
        )
        for regime in ALL_REGIMES:
            print(
                f"   {regime_name(regime):>4s}: "
                + "  ".join(f"w={w:.4g}: {table[(w, regime)]:.3e}" for w in W_VALUES)
            )

    def test_realistic_fixtures_populated_regimes(self):
        """
        The same identity on real TkSourceFunctions objects (bessel_phase amplitude, re-splined
        phase) and the phase_spline-backed G fixture, in the regions where each representation
        is actually valid.
        """
        populated = [
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (False, False, True),
            (True, False, False),
        ]
        table = {}
        for w in W_VALUES:
            case = Case.get(w)
            for regime in populated:
                with self.subTest(w=w, regime=regime_name(regime)):
                    groups = case.groups(regime, exact=False)
                    log_z = case.random_log_z(case.range_for(regime))
                    worst = max_relative_residual(
                        groups,
                        lambda u: case.direct_product(regime, u, exact=False),
                        log_z,
                        extra_scale=lambda u: case.term_envelope(
                            regime, u, exact=False
                        ),
                    )
                    table[(w, regime)] = worst
                    self.assertLess(worst, self.TOLERANCE)
        print(
            "\n[Oracle 1, realistic fixtures] max |sum - G f/H^2| / max(|G f/H^2|, envelope):"
        )
        for regime in populated:
            print(
                f"   {regime_name(regime):>4s}: "
                + "  ".join(f"w={w:.4g}: {table[(w, regime)]:.3e}" for w in W_VALUES)
            )

    def test_degenerate_equal_wavenumbers(self):
        """
        q = r: the same object passed as T_q and T_r. The q-r group then has identically zero
        phase and zero derivative, and the identity must still hold.
        """
        for w in W_VALUES:
            case = Case.get(w)
            for regime in ((True, True, True), (False, True, True)):
                with self.subTest(w=w, regime=regime_name(regime)):
                    Gk = case.Gk if regime[0] else case.Gk.numeric_Gk
                    groups = build_phase_groups(
                        regime,
                        Gk=Gk,
                        Tq=case.Tq,
                        Tr=case.Tq,
                        model_functions=case.model.functions,
                        w_background=case.model.functions.wBackground,
                    )
                    log_z = case.random_log_z(case.both_WKB, n=50)

                    def reference(u):
                        z = exp(u) - 1.0
                        M = case.Tq.M(u, z_is_log=True)
                        th = case.Tq.phase.theta_mod_2pi(u, x_is_log=True)
                        T = M * sin(th)
                        Tp = M * case.Tq.dlnM_dz(u, z_is_log=True) * sin(
                            th
                        ) + M * case.Tq.omega(u, z_is_log=True) * cos(th)
                        if regime[0]:
                            G = case.Gk.sin_amplitude(u, z_is_log=True) * sin(
                                case.Gk.phase.theta_mod_2pi(u, x_is_log=True)
                            )
                        else:
                            G = case.Gk.numeric_Gk(u, z_is_log=True)
                        H = case.model.Hubble(z)
                        return (
                            G * source_function(T, T, Tp, Tp, z, w)["source"] / (H * H)
                        )

                    worst = max_relative_residual(groups, reference, log_z)
                    self.assertLess(worst, self.TOLERANCE)
                    degenerate = [g for g in groups if g.signs[1] == -g.signs[2]]
                    for g in degenerate:
                        # theta_q - theta_r contributes nothing
                        self.assertEqual(
                            g.theta_mod_2pi(log_z[0])
                            - (
                                g.signs[0]
                                * case.Gk.phase.theta_mod_2pi(log_z[0], x_is_log=True)
                            ),
                            0.0,
                        )


# =============================================================================================
# Oracle 2: the exact Bessel fixture
# =============================================================================================


class TestOracle2Bessel(unittest.TestCase):
    def test_LG_form_of_the_Greens_function(self):
        """
        G_code(z, z') = H(z') (pi/2) sqrt(eta eta') m(k eta) m(k eta') sin(vartheta(k eta') -
        vartheta(k eta)) against compute_analytic_G, with exact (scipy) m and vartheta.
        """
        for w in W_VALUES:
            case = Case.get(w)
            worst = 0.0
            for u in case.random_log_z((case.both_numeric[0], case.z_resp), n=N_RANDOM):
                LG = case.Gk.sin_amplitude(u, z_is_log=True) * sin(
                    case.Gk.phase.theta_mod_2pi(u, x_is_log=True)
                )
                LG_raw = case.Gk.sin_amplitude(u, z_is_log=True) * sin(
                    case.Gk.phase.raw_theta(u, x_is_log=True)
                )
                exact = case.Gk.numeric_Gk(u, z_is_log=True)
                scale = abs(case.Gk.sin_amplitude(u, z_is_log=True))
                worst = max(worst, abs(LG - exact) / scale, abs(LG_raw - exact) / scale)
            print(
                f"\n[LG form of G, w={w:.5g}] max |A_G sin theta_G - G_analytic| / A_G = {worst:.3e}"
            )
            self.assertLess(worst, 1.0e-10)

    def test_exact_stand_ins_vs_scipy_every_regime(self):
        """
        evaluate_sum on the exact amplitude-phase stand-ins against compute_analytic_G *
        source_function(analytic T) / H^2, which never sees an amplitude or a phase. Checks the
        fixtures, the phase conventions (sign of theta, the constant offset in theta_G) and the
        module end to end; 1e-8 is the prompt's threshold.
        """
        table = {}
        for w in W_VALUES:
            case = Case.get(w)
            for regime in ALL_REGIMES:
                with self.subTest(w=w, regime=regime_name(regime)):
                    groups = case.groups(regime, exact=True)
                    log_z = case.random_log_z(case.both_WKB)
                    worst = max_relative_residual(groups, case.scipy_oracle, log_z)
                    table[(w, regime)] = worst
                    self.assertLess(worst, 1.0e-8)
        print(
            "\n[Oracle 2, exact stand-ins vs scipy] max |sum - G_an f_an/H^2| / max(|.|, envelope):"
        )
        for regime in ALL_REGIMES:
            print(
                f"   {regime_name(regime):>4s}: "
                + "  ".join(f"w={w:.4g}: {table[(w, regime)]:.3e}" for w in W_VALUES)
            )

    # The realistic fixtures cannot reach the 1e-8 of the exact stand-ins, and the reason is
    # neither bessel_phase (~x * 1e-8 in phase, docs/lg-phase-and-handover-followup-2026-09.md
    # section 2.4) nor the phase re-spline (~h^4 x/384): the residual is grid-independent. It is
    # the Liouville-Green truncation of the *derivative* pieces that TkSourceFunctions supplies
    # in closed form -- omega = sqrt(Tk_omegaEff_sq) and d ln M/dz -- which on the exact fixture
    # differ from the exact d theta/dz and d ln M/dz by O(x^-4) relative near the hand-over
    # (logs/05 deviation 5 measured 8.5e-6 at w = 1/3 and 7.0e-5 at w = 0.2 in d ln M/dz). Those
    # pieces dominate f through DT_q DT_r. The threshold is the measured floor with headroom.
    REALISTIC_THRESHOLD = 5.0e-4

    def test_realistic_fixtures_vs_scipy(self):
        """
        The all-oscillatory regime on real TkSourceFunctions objects and the phase_spline-backed
        G fixture against the scipy oracle. The residual is the fixtures' floor, reported for
        prompt 08 as an acceptance threshold; the same points restricted to x_q > 100 show how
        fast it falls away from the hand-over, and the 300/decade repeat shows it is not a grid
        effect.
        """
        for w in W_VALUES:
            with self.subTest(w=w):
                results = {}
                for samples in (
                    PRODUCTION_SAMPLES_PER_LOG10Z,
                    REFINED_SAMPLES_PER_LOG10Z,
                ):
                    case = Case.get(w, samples)
                    groups = case.groups((True, True, True), exact=False)
                    log_z = case.random_log_z(case.both_WKB)
                    results[samples] = max_relative_residual(
                        groups, case.scipy_oracle, log_z
                    )
                    deep = [u for u in log_z if case.Fq.x(exp(u) - 1.0) > 100.0]
                    results[("deep", samples)] = max_relative_residual(
                        groups, case.scipy_oracle, deep
                    )
                    # the same points with G exact, to separate the T fixtures' floor from the
                    # G fixture's
                    groups_T_only = build_phase_groups(
                        (True, True, True),
                        Gk=case.Gk,
                        Tq=case.Tq_real,
                        Tr=case.Tr_real,
                        model_functions=case.model.functions,
                        w_background=case.model.functions.wBackground,
                    )
                    results[("T only", samples)] = max_relative_residual(
                        groups_T_only, case.scipy_oracle, log_z
                    )
                print(
                    f"\n[Oracle 2, realistic fixtures vs scipy, w={w:.5g}] all-oscillatory, x_q in [19, 1000]: "
                    f"{PRODUCTION_SAMPLES_PER_LOG10Z}/decade {results[PRODUCTION_SAMPLES_PER_LOG10Z]:.3e} "
                    f"(T fixtures only {results[('T only', PRODUCTION_SAMPLES_PER_LOG10Z)]:.3e}; "
                    f"x_q > 100 only {results[('deep', PRODUCTION_SAMPLES_PER_LOG10Z)]:.3e}); "
                    f"{REFINED_SAMPLES_PER_LOG10Z}/decade {results[REFINED_SAMPLES_PER_LOG10Z]:.3e} "
                    f"(T fixtures only {results[('T only', REFINED_SAMPLES_PER_LOG10Z)]:.3e}; "
                    f"x_q > 100 only {results[('deep', REFINED_SAMPLES_PER_LOG10Z)]:.3e})"
                )
                self.assertLess(
                    results[PRODUCTION_SAMPLES_PER_LOG10Z], self.REALISTIC_THRESHOLD
                )
                self.assertLess(
                    results[REFINED_SAMPLES_PER_LOG10Z], self.REALISTIC_THRESHOLD
                )

    def test_LG_truncation_is_the_realistic_floor(self):
        """
        Pin the attribution: replacing the fixture's LG closed forms omega and d ln M/dz by the
        exact d theta/dz and d ln M/dz (while keeping its bessel_phase amplitude and re-splined
        phase) must remove most of the residual of the test above.
        """
        for w in W_VALUES:
            with self.subTest(w=w):
                case = Case.get(w)
                log_z = case.random_log_z(case.both_WKB)
                before = max_relative_residual(
                    case.groups((True, True, True), exact=False),
                    case.scipy_oracle,
                    log_z,
                )

                class Hybrid:
                    """TkSourceFunctions fixture with exact derivative pieces."""

                    def __init__(self, real, exact):
                        self.M = real.M
                        self.phase = real.phase
                        self.dlnM_dz = exact.dlnM_dz
                        self.omega = exact.omega

                groups = build_phase_groups(
                    (True, True, True),
                    Gk=case.Gk_real,
                    Tq=Hybrid(case.Tq_real, case.Tq),
                    Tr=Hybrid(case.Tr_real, case.Tr),
                    model_functions=case.model.functions,
                    w_background=case.model.functions.wBackground,
                )
                after = max_relative_residual(groups, case.scipy_oracle, log_z)
                print(
                    f"\n[LG truncation attribution, w={w:.5g}] realistic floor {before:.3e} -> "
                    f"{after:.3e} with exact omega, d ln M/dz"
                )
                self.assertLess(after, before / 2.0)
                self.assertLess(after, 1.0e-5)


# =============================================================================================
# phase composition
# =============================================================================================


class _PolynomialPhaseFactor:
    """
    A factor whose phase is an exactly-known polynomial in v = u - u0, u = log(1+z):
    theta(v) = c0 + c1 v + c2 v^2, stored in a real phase_spline through (div 2pi, mod 2pi)
    samples computed in extended precision. A cubic spline reproduces a quadratic exactly, so
    any residual is rounding, not fit error. The phase must keep one sign over the grid
    (phase_spline's logarithmic chunking requires it), as every production phase does. Exposes
    the TkSourceFunctions LG-phase protocol (`phase`, `omega`) and, for a Green's-function role,
    `sin_amplitude`.
    """

    def __init__(self, coefficients, u0, u_grid, increasing):
        self.c = coefficients
        self.u0 = u0
        mpmath.mp.dps = 40
        samples = []
        for u in u_grid:
            theta = self.exact(u)
            div = int(mpmath.floor(theta / (2 * mpmath.pi)))
            mod = float(theta - div * 2 * mpmath.pi)  # in [0, 2pi)
            # the stored convention is a negative remainder in (-2pi, 0]
            div, mod = div + 1, mod - float(TWO_PI)
            samples.append((float(u), div, mod))
        log_x, div_2pi, mod_2pi = zip(*samples)
        self.phase = phase_spline(
            list(log_x),
            list(div_2pi),
            list(mod_2pi),
            x_is_log=True,
            x_is_redshift=True,
            chunk_step=None,
            chunk_logstep=125,
            increasing=increasing,
        )

    def exact(self, u):
        v = mpmath.mpf(u) - mpmath.mpf(self.u0)
        c0, c1, c2 = (mpmath.mpf(c) for c in self.c)
        return c0 + c1 * v + c2 * v * v

    def exact_mod_2pi(self, u):
        theta = self.exact(u)
        return theta - 2 * mpmath.pi * mpmath.floor(theta / (2 * mpmath.pi))

    def exact_deriv_log(self, u):
        v = mpmath.mpf(u) - mpmath.mpf(self.u0)
        c0, c1, c2 = (mpmath.mpf(c) for c in self.c)
        return c1 + 2 * c2 * v

    def with_exact_phase(self):
        """
        The same factor with `phase` replaced by exact callables (raw phase, remainder and
        derivative each correctly rounded from extended precision), so that the composition can
        be tested independently of phase_spline's own rounding.
        """
        clone = _PolynomialPhaseFactor.__new__(_PolynomialPhaseFactor)
        clone.c = self.c
        clone.u0 = self.u0
        clone.phase = _ExactPhase(
            lambda u: (
                float(self.exact(u)),
                float(self.exact_mod_2pi(u)),
                float(self.exact_deriv_log(u)),
            )
        )
        return clone

    # TkSourceFunctions protocol: omega = d theta/dz = (d theta/du) / (1+z)
    def omega(self, x, z_is_log=False):
        assert z_is_log
        return float(self.exact_deriv_log(x)) / exp(x)

    def sin_amplitude(self, x, z_is_log=False):
        return 1.0

    def M(self, x, z_is_log=False):
        return 1.0

    def dlnM_dz(self, x, z_is_log=False):
        return 0.0


def circular_distance(a, b):
    """|a - b| mod 2pi, folded into [0, pi]."""
    d = fmod(a - b, TWO_PI)
    if d < 0.0:
        d += TWO_PI
    return min(d, TWO_PI - d)


class TestPhaseComposition(unittest.TestCase):
    def _factors(self, scale):
        """
        G, q, r factors whose phases reach |theta| ~ scale rad over z in [1e3, 1e5].
        theta_G decreases with u (increasing=False), theta_q, theta_r increase (increasing=True),
        matching the conventions of GkSourceFunctions and TkSourceFunctions respectively.
        """
        u_grid = list(np.linspace(log(1.0 + 1.0e3), log(1.0 + 1.0e5), 3000))
        u0 = u_grid[0]
        span = u_grid[-1] - u_grid[0]
        rate = 0.9 * scale / span
        # all three negative over the grid, as every production phase is
        G = _PolynomialPhaseFactor(
            (-0.2 * scale, -0.7 * rate, -0.02 * rate / span), u0, u_grid, False
        )
        q = _PolynomialPhaseFactor(
            (-1.1 * scale, rate, 0.05 * rate / span), u0, u_grid, True
        )
        r = _PolynomialPhaseFactor(
            (-0.9 * scale, 0.6 * rate, 0.03 * rate / span), u0, u_grid, True
        )
        return G, q, r, u_grid

    def _measure(self, G, q, r, u_grid, rng):
        """
        Compose the three phases with the module and compare with the exact composed phase:
        the module's remainder route, the reduce-the-raw-sum route it deliberately does not
        take, the raw composed phase, and theta_deriv against a centred finite difference.
        """
        model = FakeModel(1.0 / 3.0)
        groups = build_phase_groups(
            (True, True, True),
            Gk=G,
            Tq=q,
            Tr=r,
            model_functions=model.functions,
            w_background=model.functions.wBackground,
        )
        points = rng.uniform(u_grid[0] + 0.05, u_grid[-1] - 0.05, N_RANDOM)
        out = {"mod": 0.0, "raw_route": 0.0, "raw": 0.0, "deriv": 0.0, "largest": 0.0}
        for g in groups:
            sG, sq, sr = g.signs
            # the composed remainder is a signed sum of three remainders
            self.assertLess(abs(g.theta_mod_2pi(points[0])), 3.0 * TWO_PI)
            for u in points:
                exact = sG * G.exact(u) + sq * q.exact(u) + sr * r.exact(u)
                exact_mod = float(
                    exact - 2 * mpmath.pi * mpmath.floor(exact / (2 * mpmath.pi))
                )
                out["largest"] = max(out["largest"], abs(float(exact)))
                out["mod"] = max(
                    out["mod"], circular_distance(g.theta_mod_2pi(u), exact_mod)
                )
                out["raw_route"] = max(
                    out["raw_route"],
                    circular_distance(fmod(g.theta(u), TWO_PI), exact_mod),
                )
                out["raw"] = max(out["raw"], abs(g.theta(u) - float(exact)))

                h = 1.0e-4
                fd = float(
                    (
                        sG * G.exact(u + h)
                        + sq * q.exact(u + h)
                        + sr * r.exact(u + h)
                        - (
                            sG * G.exact(u - h)
                            + sq * q.exact(u - h)
                            + sr * r.exact(u - h)
                        )
                    )
                    / (2 * h)
                )
                out["deriv"] = max(out["deriv"], abs(g.theta_deriv(u) - fd) / abs(fd))
        return out

    def test_composed_phase_with_exact_constituents(self):
        """
        The module's composition itself: with constituents whose (raw, remainder, derivative)
        are each correctly rounded, the composed remainder must agree with (Psi mod 2pi) to
        1e-10 at |Psi| ~ 1e6 rad, where reducing the raw sum instead loses ~eps |Psi| ~ 1e-10.
        """
        rng = np.random.default_rng(SEED)
        for scale in (1.0e5, 1.0e6):
            with self.subTest(scale=scale):
                G, q, r, u_grid = self._factors(scale)
                out = self._measure(
                    G.with_exact_phase(),
                    q.with_exact_phase(),
                    r.with_exact_phase(),
                    u_grid,
                    rng,
                )
                print(
                    f"\n[phase composition, exact constituents, |Psi| up to {out['largest']:.3g} rad] "
                    f"remainder route {out['mod']:.3e} rad; reduce-raw route {out['raw_route']:.3e} rad; "
                    f"raw phase {out['raw']:.3e} rad; theta_deriv vs FD {out['deriv']:.3e}"
                )
                self.assertLess(out["mod"], 1.0e-10)
                self.assertLess(out["deriv"], 1.0e-8)

    def test_composed_phase_through_phase_spline(self):
        """
        The same composition through real phase_spline objects (chunk_logstep=125) holding exact
        (div 2pi, mod 2pi) samples of the quadratic phases. Measurement, for the record: with
        geometric chunking the top chunk spans most of the range, so its rebased phase is as
        large as the raw one and the spline rounds at ~20 eps |theta| -- 4e-8 rad at 2e6 rad --
        for either route. That is a property of phase_spline (LiouvilleGreen/, out of scope
        here), not of the composition, which the exact-constituent test above isolates.
        """
        rng = np.random.default_rng(SEED)
        for scale in (1.0e5, 1.0e6):
            with self.subTest(scale=scale):
                G, q, r, u_grid = self._factors(scale)
                out = self._measure(G, q, r, u_grid, rng)
                print(
                    f"\n[phase composition, phase_spline constituents, |Psi| up to {out['largest']:.3g} rad] "
                    f"remainder route {out['mod']:.3e} rad; reduce-raw route {out['raw_route']:.3e} rad; "
                    f"raw phase {out['raw']:.3e} rad; theta_deriv vs FD {out['deriv']:.3e}"
                )
                # the composition adds nothing beyond the constituents' own spline rounding
                self.assertLess(out["mod"], 2.0 * out["raw"] + 1.0e-12)
                self.assertLess(out["mod"], 1.0e-6)
                self.assertLess(out["deriv"], 1.0e-8)

    def test_theta_deriv_uses_closed_form_omega_for_T(self):
        """
        For a transfer-function factor theta_deriv must be omega (1+z) from the object's closed
        form, not the spline derivative; for G it must be the spline's log-derivative.
        """
        model = FakeModel(1.0 / 3.0)
        G, q, r, u_grid = self._factors(1.0e5)
        # make the spline derivative distinguishable from omega
        q_broken = _PolynomialPhaseFactor(q.c, q.u0, u_grid, True)
        q_broken.omega = lambda x, z_is_log=False: 12345.0 / exp(x)
        groups = build_phase_groups(
            (True, True, True),
            Gk=G,
            Tq=q_broken,
            Tr=r,
            model_functions=model.functions,
            w_background=model.functions.wBackground,
        )
        u = 0.5 * (u_grid[0] + u_grid[-1])
        for g in groups:
            sG, sq, sr = g.signs
            expected = (
                sG * G.phase.theta_deriv(u, x_is_log=True, log_derivative=True)
                + sq * 12345.0
                + sr * r.omega(u, z_is_log=True) * exp(u)
            )
            self.assertAlmostEqual(
                g.theta_deriv(u), expected, delta=1.0e-9 * abs(expected)
            )


# =============================================================================================
# region-boundary consistency
# =============================================================================================


class TestBoundaryConsistency(unittest.TestCase):
    """
    Just below the hand-over of T_r (T_q still numeric), the "T_r oscillatory" decomposition must
    agree with source_function * G / H^2 in which T_r is taken from a numeric spline that extends
    below the hand-over -- the seam prompt 08 stitches across. The reference numeric spline comes
    from a second fixture for the same r whose numeric region reaches 0.4 e-folds further in.
    """

    EXTRA_EFOLDS = 0.4

    def _seam(self, w, Tr_functions, label):
        case = Case.get(w)
        extended = Fixture(w, k=R_WAVENUMBER, efolds_subh=3.5 + self.EXTRA_EFOLDS)
        Tr_extended = extended.exact_functions()
        hand_over = Tr_functions.WKB_region[0]
        # nodes of the extended numeric grid below the hand-over, and their log-midpoints
        nodes = [
            z
            for z in extended.z_numeric
            if z < hand_over
            and z > Tr_extended.numeric_region[1]
            and z > case.Tq_real.numeric_region[1]
        ]
        self.assertGreater(len(nodes), 10)
        mids = [
            exp(0.5 * (log(1 + a) + log(1 + b))) - 1.0
            for a, b in zip(nodes[:-1], nodes[1:])
        ]

        out = {}
        for G_osc in (False, True):
            Gk = case.Gk if G_osc else case.Gk.numeric_Gk
            groups = build_phase_groups(
                (G_osc, False, True),
                Gk=Gk,
                Tq=case.Tq_real,
                Tr=Tr_functions,
                model_functions=case.model.functions,
                w_background=case.model.functions.wBackground,
            )

            def reference(u):
                z = exp(u) - 1.0
                H = case.model.Hubble(z)
                if G_osc:
                    G = case.Gk.sin_amplitude(u, z_is_log=True) * sin(
                        case.Gk.phase.theta_mod_2pi(u, x_is_log=True)
                    )
                else:
                    G = case.Gk.numeric_Gk(u, z_is_log=True)
                f = source_function(
                    case.Tq_real.T(u, z_is_log=True),
                    Tr_extended.T(u, z_is_log=True),
                    case.Tq_real.dT_dz(u, z_is_log=True),
                    Tr_extended.dT_dz(u, z_is_log=True),
                    z,
                    w,
                )["source"]
                return G * f / (H * H)

            out[(G_osc, "nodes")] = max_relative_residual(
                groups, reference, [log(1.0 + z) for z in nodes]
            )
            out[(G_osc, "mids")] = max_relative_residual(
                groups, reference, [log(1.0 + z) for z in mids]
            )
        x_at_hand_over = extended.x(hand_over)
        print(
            f"\n[boundary, {label}, w={w:.5g}, x_r = {x_at_hand_over:.1f} -> "
            f"{extended.x(nodes[-1]):.1f}] LG side vs extended numeric spline, /envelope: "
            f"G smooth: nodes {out[(False, 'nodes')]:.3e}, midpoints {out[(False, 'mids')]:.3e}; "
            f"G oscillatory: nodes {out[(True, 'nodes')]:.3e}, midpoints {out[(True, 'mids')]:.3e}"
        )
        return out

    def test_exact_fixture_seam(self):
        """
        Exact-envelope fixture (exact M and theta; LG closed-form omega and d ln M/dz): at the
        extended spline's own nodes the residual is the LG truncation of the derivative pieces
        (see TestOracle2Bessel.REALISTIC_THRESHOLD), plus the phase re-spline and bessel_phase
        floors; between nodes the numeric dT/dz spline's fit error is added (board issue
        [05-numeric-region-is-now-the-accuracy-floor]: 2.7e-4 of envelope between grid points at
        the hand-over).
        """
        for w in W_VALUES:
            with self.subTest(w=w):
                case = Case.get(w)
                out = self._seam(w, case.Tr_real, "exact fixture")
                for G_osc in (False, True):
                    self.assertLess(out[(G_osc, "nodes")], 5.0e-4)
                    self.assertLess(out[(G_osc, "mids")], 2.0e-3)

    def test_matched_LG_fixture_seam(self):
        """
        Documentation for prompt 08, not a threshold on this module: with the code's own
        Liouville-Green representation, matched to T and dT/dz at the hand-over exactly as
        TkWKBIntegration.store() matches it, the seam carries the LG truncation error of the
        representation itself -- audit TK-8(e) gives 3.7e-4 to 5.6e-3 of envelope in T at
        3 e-folds sub-horizon. A residual of this size at the hand-over is the representation,
        not a bug.
        """
        for w in W_VALUES:
            with self.subTest(w=w):
                case = Case.get(w)
                matched = matched_LG_functions(case.Fr)
                # the matching must reproduce T at the hand-over
                z_init = case.Fr.crossover_z
                self.assertAlmostEqual(
                    matched.T_WKB(z_init) / case.Fr.T_exact(z_init), 1.0, delta=1.0e-10
                )
                out = self._seam(w, matched, "matched LG fixture")
                for G_osc in (False, True):
                    self.assertLess(out[(G_osc, "nodes")], 5.0e-2)


if __name__ == "__main__":
    unittest.main()
