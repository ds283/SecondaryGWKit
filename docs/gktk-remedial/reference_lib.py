"""
Independent reference quadratures for the Gk/Tk WKB phase remedial campaign (prompt 01).

Not importable machinery: this is a script-support module for the other scripts in this
directory, in the pattern of ``docs/gk-wkb-review-fable-2026-09-09/common.py``. Sibling scripts
pick it up with

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Two independent reference methods live here.

* **mpmath** (``mp.dps = 40``) for ``RadiationModel`` and ``LambdaCDMModel``, whose backgrounds
  are closed-form: ``H``, ``epsilon``, ``epsilon'``, ``w`` and ``w'`` are re-derived here in
  ``mpmath`` from the same algebra ``CosmologyModels/LambdaCDM/LambdaCDM.py`` uses, so nothing
  in the reference passes through the double-precision ``ModelFunctions`` the campaign is about
  to replace.
* **converged adaptive quadrature of the double-precision integrand** for ``QCDModel``, whose
  ``H(z)`` is itself a spline evaluation of ``T(z)`` and cannot be lifted to 40 digits: SciPy
  ``quad`` in ``u = log(1+z)`` per production interval, summed with ``math.fsum``, cross-checked
  against the same at a looser tolerance and against composite Gauss-Legendre order 40 with
  interval bisection.

**Sign conventions** (``prompts/GkTk-remedial/README.md`` §2 (a), (c)); every routine here
returns an integral over an *increasing* range of ``u``, i.e. from the lower redshift to the
higher one:

    tau_delta(z_lo, z_hi)      = int_{z_lo}^{z_hi} dz/H            = tau(z_lo) - tau(z_hi) > 0
    cs_tau_delta(z_lo, z_hi)   = int_{z_lo}^{z_hi} c_s dz/H        = cs_tau(z_lo) - cs_tau(z_hi) > 0
    friction_delta(z_lo, z_hi) = int_{z_lo}^{z_hi} (3/2)(1+c_s^2) dz/(1+z) > 0
    rho_G(z_lo, z_hi; k)       = int_{z_lo}^{z_hi} C/(omega + k/H) dz
    rho_T(z_lo, z_hi; k)       = int_{z_lo}^{z_hi} C_T/(omega_T + k c_s/H) dz

The JSON generator records ``F(z_j) - F(z_top) = -friction_delta(z_j, z_top)`` (the primitive of
``TkWKBIntegration.friction_RHS`` *decreases* towards lower z), but ``tau`` and ``cs_tau``
*increase*, so their ``X(z_j) - X(z_top)`` entries are ``+..._delta(z_j, z_top)``.

``C`` and ``C_T`` are the **non-leading terms of** ``Gk_omegaEff_sq`` / ``Tk_omegaEff_sq``,
re-implemented here rather than imported and never formed as ``omega^2 - omega_0^2``
(``RECONCILIATION.md`` §2 item 3).
"""

import math
import warnings
from math import log, log1p, expm1, sqrt, fsum

import mpmath as mp
import numpy as np
from scipy.integrate import quad

mp.mp.dps = 40


# =============================================================================================
# the non-leading parts of the two frequency functions, in double precision
# =============================================================================================


def C_G_double(functions, z: float) -> float:
    """
    ``C(z) = -eps'/(2 s) + (3 eps/2 - eps^2/4 - 2)/s^2``: the ``B + C`` terms of
    ``ComputeTargets/WKB_Gk.py:Gk_omegaEff_sq``, re-implemented.
    """
    s = 1.0 + z
    eps = functions.epsilon(z)
    epsPrime = functions.d_epsilon_dz(z)
    return -epsPrime / 2.0 / s + (3.0 * eps / 2.0 - eps * eps / 4.0 - 2.0) / (s * s)


def C_T_double(functions, z: float) -> float:
    """
    The ``B + C`` terms of ``ComputeTargets/WKB_Tk.py:Tk_omegaEff_sq``, re-implemented.
    """
    s = 1.0 + z
    w = functions.wPerturbations(z)
    wPrime = functions.d_wPerturbations_dz(z)
    eps = functions.epsilon(z)
    epsPrime = functions.d_epsilon_dz(z)

    B = (3.0 / 2.0 * wPrime - epsPrime / 2.0) / s
    C = (
        3.0 / 2.0 * (1.0 + eps) * (1.0 + w)
        - eps * (3.0 + eps / 2.0) / 2.0
        - 9.0 / 4.0 * (1.0 + w) * (1.0 + w)
    ) / (s * s)
    return B + C


# =============================================================================================
# double-precision integrands in u = log(1+z)
# =============================================================================================


def integrand_tau_double(functions):
    def f(u: float) -> float:
        z = expm1(u)
        return (1.0 + z) / functions.Hubble(z)

    return f


def integrand_cs_tau_double(functions):
    def f(u: float) -> float:
        z = expm1(u)
        return (1.0 + z) * sqrt(functions.wPerturbations(z)) / functions.Hubble(z)

    return f


def integrand_friction_double(functions):
    def f(u: float) -> float:
        z = expm1(u)
        return 1.5 * (1.0 + functions.wPerturbations(z))

    return f


def integrand_rho_G_double(functions, k: float):
    def f(u: float) -> float:
        z = expm1(u)
        H = functions.Hubble(z)
        k_over_H = k / H
        C = C_G_double(functions, z)
        omega = sqrt(k_over_H * k_over_H + C)
        return C / (omega + k_over_H) * (1.0 + z)

    return f


def integrand_rho_T_double(functions, k: float):
    def f(u: float) -> float:
        z = expm1(u)
        H = functions.Hubble(z)
        cs = sqrt(functions.wPerturbations(z))
        k_cs_over_H = k * cs / H
        C = C_T_double(functions, z)
        omega = sqrt(k_cs_over_H * k_cs_over_H + C)
        return C / (omega + k_cs_over_H) * (1.0 + z)

    return f


# =============================================================================================
# mpmath backgrounds
# =============================================================================================


class MpLambdaCDM:
    """
    ``LambdaCDM`` re-derived at 40 digits from the same algebra as
    ``CosmologyModels/LambdaCDM/LambdaCDM.py``.
    """

    def __init__(self, cosmology):
        self.rho_m0 = mp.mpf(cosmology.rho_m0)
        self.rho_r0 = mp.mpf(cosmology.rho_r0)
        self.rho_cc = mp.mpf(cosmology.rho_cc)
        self.Mpsq = mp.mpf(cosmology.Mpsq)
        self.omega_m = mp.mpf(cosmology.omega_m)
        self.omega_r = mp.mpf(cosmology.omega_r)
        self.omega_cc = mp.mpf(cosmology.omega_cc)
        self.w_rad = mp.mpf(1) / 3

    def Hubble(self, s):
        rho = self.rho_m0 * s**3 + self.rho_r0 * s**4 + self.rho_cc
        return mp.sqrt(rho / (3 * self.Mpsq))

    def _D(self, s):
        return self.omega_m * s**3 + self.omega_r * s**4 + self.omega_cc

    def d_lnH_dz(self, s):
        return (3 * self.omega_m * s**2 + 4 * self.omega_r * s**3) / (2 * self._D(s))

    def d2_lnH_dz2(self, s):
        first = (6 * self.omega_m * s + 12 * self.omega_r * s**2) / (2 * self._D(s))
        d1 = self.d_lnH_dz(s)
        return first - 2 * d1 * d1

    def epsilon(self, s):
        return s * self.d_lnH_dz(s)

    def d_epsilon_dz(self, s):
        return self.d_lnH_dz(s) + s * self.d2_lnH_dz2(s)

    def wPerturbations(self, s):
        return (self.w_rad * self.omega_r * s) / (self.omega_m + self.omega_r * s)

    def d_wPerturbations_dz(self, s):
        d = self.omega_m + self.omega_r * s
        return (self.w_rad * self.omega_r * self.omega_m) / (d * d)


class MpRadiation:
    """Exact radiation, ``H = H0 s^2``, ``epsilon = 2``, ``w = 1/3``, at 40 digits."""

    def __init__(self, H0=1.0):
        self.H0 = mp.mpf(H0)
        self.w_rad = mp.mpf(1) / 3

    def Hubble(self, s):
        return self.H0 * s * s

    def epsilon(self, s):
        return mp.mpf(2)

    def d_epsilon_dz(self, s):
        return mp.mpf(0)

    def wPerturbations(self, s):
        return self.w_rad

    def d_wPerturbations_dz(self, s):
        return mp.mpf(0)


def mp_C_G(bg, s):
    eps = bg.epsilon(s)
    epsPrime = bg.d_epsilon_dz(s)
    return -epsPrime / (2 * s) + (3 * eps / 2 - eps * eps / 4 - 2) / (s * s)


def mp_C_T(bg, s):
    eps = bg.epsilon(s)
    epsPrime = bg.d_epsilon_dz(s)
    w = bg.wPerturbations(s)
    wPrime = bg.d_wPerturbations_dz(s)
    B = (mp.mpf(3) / 2 * wPrime - epsPrime / 2) / s
    C = (
        mp.mpf(3) / 2 * (1 + eps) * (1 + w)
        - eps * (3 + eps / 2) / 2
        - mp.mpf(9) / 4 * (1 + w) * (1 + w)
    ) / (s * s)
    return B + C


def mp_integrand(bg, quantity: str, k=None):
    """Return the mpmath integrand in ``u = log(1+z)`` for the named quantity."""
    if quantity == "tau":
        return lambda u: mp.exp(u) / bg.Hubble(mp.exp(u))
    if quantity == "cs_tau":
        return (
            lambda u: mp.exp(u)
            * mp.sqrt(bg.wPerturbations(mp.exp(u)))
            / bg.Hubble(mp.exp(u))
        )
    if quantity == "friction":
        return lambda u: mp.mpf(3) / 2 * (1 + bg.wPerturbations(mp.exp(u)))
    if quantity == "rho_G":
        kk = mp.mpf(k)

        def f(u):
            s = mp.exp(u)
            k_over_H = kk / bg.Hubble(s)
            C = mp_C_G(bg, s)
            omega = mp.sqrt(k_over_H * k_over_H + C)
            return C / (omega + k_over_H) * s

        return f
    if quantity == "rho_T":
        kk = mp.mpf(k)

        def f(u):
            s = mp.exp(u)
            cs = mp.sqrt(bg.wPerturbations(s))
            k_cs_over_H = kk * cs / bg.Hubble(s)
            C = mp_C_T(bg, s)
            omega = mp.sqrt(k_cs_over_H * k_cs_over_H + C)
            return C / (omega + k_cs_over_H) * s

        return f
    raise ValueError(f"mp_integrand: unknown quantity {quantity!r}")


def mp_increment(bg, quantity: str, z_lo: float, z_hi: float, k=None):
    """
    ``int_{z_lo}^{z_hi} <quantity integrand> dz`` at 40 digits, with the interval split at each
    integer power of ten of ``1+z`` so that ``mp.quad`` never sees more than one decade of
    dynamic range at a time. Returns an ``mpf``.
    """
    if z_hi <= z_lo:
        return mp.mpf(0)

    u_lo = mp.log(1 + mp.mpf(z_lo))
    u_hi = mp.log(1 + mp.mpf(z_hi))

    ln10 = mp.log(10)
    n_lo = int(mp.floor(u_lo / ln10)) + 1
    n_hi = int(mp.ceil(u_hi / ln10)) - 1
    points = [u_lo]
    for n in range(n_lo, n_hi + 1):
        p = n * ln10
        if u_lo < p < u_hi:
            points.append(p)
    points.append(u_hi)

    f = mp_integrand(bg, quantity, k)
    return mp.quad(f, points)


# =============================================================================================
# converged adaptive quadrature of the double-precision integrand (QCD)
# =============================================================================================

_GL40_X, _GL40_W = np.polynomial.legendre.leggauss(40)


def _gauss_panel(f, a: float, b: float) -> float:
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return half * fsum(w * f(mid + half * x) for x, w in zip(_GL40_X, _GL40_W))


def gauss_bisect(f, a: float, b: float, rtol: float = 1e-14, max_level: int = 12):
    """
    Composite Gauss-Legendre order 40 with uniform bisection, refined until two successive
    refinements agree to ``rtol`` relative (or ``max_level`` is reached).

    :return: ``(value, achieved_relative_change, levels_used)``
    """
    prev = _gauss_panel(f, a, b)
    n = 1
    for level in range(1, max_level + 1):
        n *= 2
        edges = np.linspace(a, b, n + 1)
        cur = fsum(_gauss_panel(f, edges[i], edges[i + 1]) for i in range(n))
        denom = abs(cur) if cur != 0.0 else 1.0
        change = abs(cur - prev) / denom
        if change <= rtol:
            return cur, change, level
        prev = cur
    return prev, change, max_level


def quad_sum_over_intervals(f, u_edges, epsrel: float):
    """
    SciPy ``quad`` on each ``[u_edges[i], u_edges[i+1]]``, summed with ``math.fsum``.

    QUADPACK refuses ``epsabs = 0`` with ``epsrel < 50*eps = 1.11e-14``, so the tightest
    pure-relative tolerance available is a shade above that.

    :return: ``(total, per_interval_list, max_reported_abserr, n_intervals_warned)``
    """
    parts = []
    worst = 0.0
    warned = 0
    for i in range(len(u_edges) - 1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val, err = quad(
                f,
                u_edges[i],
                u_edges[i + 1],
                limit=400,
                epsabs=0.0,
                epsrel=epsrel,
            )
        if caught:
            warned += 1
        parts.append(val)
        worst = max(worst, abs(err))
    return fsum(parts), parts, worst, warned


def gauss_sum_over_intervals(f, u_edges, rtol: float = 1e-14, max_level: int = 4):
    """``gauss_bisect`` on each production interval, summed with ``math.fsum``."""
    parts = []
    worst_change = 0.0
    worst_level = 0
    for i in range(len(u_edges) - 1):
        val, change, level = gauss_bisect(
            f, u_edges[i], u_edges[i + 1], rtol=rtol, max_level=max_level
        )
        parts.append(val)
        worst_change = max(worst_change, change)
        worst_level = max(worst_level, level)
    return fsum(parts), parts, worst_change, worst_level


def prefix_sums(parts):
    """``[fsum(parts[:j]) for j in range(len(parts)+1)]`` -- cumulative, correctly rounded."""
    return [fsum(parts[:j]) for j in range(len(parts) + 1)]


# =============================================================================================
# checkpoint selection
# =============================================================================================

CHECKPOINT_TARGET_Z = (1e15, 1e13, 1e11, 1e9, 1e7, 1e6, 1e4, 1e3, 1e2, 1e1, 1.0)


def select_checkpoints(z_nodes) -> list:
    """
    Pick ~12 checkpoint node indices spanning a descending grid: the top node, the bottom node,
    and the node nearest each of ``CHECKPOINT_TARGET_Z`` that lies strictly inside the grid.

    :param z_nodes: descending array of node redshifts
    :return: a sorted list of indices into ``z_nodes``
    """
    z = np.asarray(z_nodes, dtype=float)
    n = len(z)
    idx = {0, n - 1}
    logz = np.log10(z)
    for target in CHECKPOINT_TARGET_Z:
        if not (z[-1] < target < z[0]):
            continue
        j = int(np.argmin(np.abs(logz - math.log10(target))))
        if 0 < j < n - 1:
            idx.add(j)
    return sorted(idx)


def short_baseline_locations(z_nodes, targets=(1e6, 1e2, 1.0)) -> list:
    """
    For each target redshift, the index ``j`` of the nearest interior node. The short-baseline
    reference is then taken over the interval ``[z_nodes[j+1], z_nodes[j]]`` (descending grid, so
    ``z_nodes[j+1] < z_nodes[j]``).
    """
    z = np.asarray(z_nodes, dtype=float)
    logz = np.log10(z)
    out = []
    for target in targets:
        j = int(np.argmin(np.abs(logz - math.log10(target))))
        j = min(max(j, 0), len(z) - 2)
        out.append(j)
    return out


FRACTIONAL_OFFSET = 0.37


def fractional_point(
    z_hi: float, z_lo: float, fraction: float = FRACTIONAL_OFFSET
) -> float:
    """
    The point ``fraction`` of the way from ``z_hi`` to ``z_lo`` **in** ``u = log(1+z)`` -- the
    campaign's integration variable (README §5 rule 9).
    """
    u_hi = log1p(z_hi)
    u_lo = log1p(z_lo)
    return expm1(u_hi + fraction * (u_lo - u_hi))
