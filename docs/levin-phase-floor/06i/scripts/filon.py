"""levin-phase-floor prompt 06i: the Filon-type quadrature both references are built from. New in 06i.

Two implementations of  I = int_L^R g(y) exp(i omega y) dy  for a smooth g:

* FCC (Filon-Clenshaw-Curtis) in mpmath: on each panel [c - h, c + h] the amplitude g is sampled at
  the N + 1 Clenshaw-Curtis points (evaluated in double by the caller's own callables), its
  Chebyshev coefficients are formed in mpmath, and the moments
  M_k(lam) = int_{-1}^{1} T_k(s) exp(i lam s) ds, lam = omega h, are computed in mpmath at DPS
  digits (plus guard digits); the panel is h exp(i omega c) sum'' c_k M_k(lam), and panels are
  summed in mpmath.
* QAWO (scipy.integrate.quad, weight='cos'/'sin') in double precision, panel by panel, with the
  variable shifted to s = y - c on a symmetric panel [-h, h] and exp(i omega c) formed exactly in
  mpmath. To make every product QAWO forms internally exact, omega is replaced by omega_r, omega
  rounded to OMEGA_BITS significant bits (the remainder (omega - omega_r) s is a phase of
  ~2^-40 of the panel's, folded into the amplitude), and h is a power of two: QUADPACK's bisection
  of [-h, h] then produces only dyadic centres and half-widths, and omega_r times any of them
  (down to 2^-13 h) is exact in double. A segment whose length is not a sum of such panels ends
  with one small panel whose phase span is at most 2^-SLIVER_BITS of a full panel's.

Nothing here imports the repository: the callers supply g.
"""

import cmath
import math
import warnings
from fractions import Fraction

import mpmath as mp
from mpmath.calculus.quadrature import GaussLegendre
from scipy import integrate

DPS = 40  # the prompt's 40 digits
GUARD = 25  # extra digits inside the moment computation
OMEGA_BITS = 40
SLIVER_BITS = 12


def set_precision():
    mp.mp.dps = DPS


# ------------------------------------------------------------------------------------------
# moments


_GL_CACHE = {}


def _gl_nodes():
    """384-point Gauss-Legendre nodes and weights on [-1, 1] at the working precision + guard."""
    prec = mp.mp.prec
    nodes = _GL_CACHE.get(prec)
    if nodes is None:
        gl = GaussLegendre(mp.mp)
        nodes = gl.calc_nodes(8, prec)
        _GL_CACHE[prec] = nodes
    return nodes


def cheb_moments(lam, N):
    """
    [M_0, ..., M_N], M_k = int_{-1}^{1} T_k(s) exp(i lam s) ds, as mpc at DPS digits.

    |lam| >= 2N + 10: the forward three-term recurrence from integration by parts (stable while
    k < |lam|), at DPS + GUARD digits:
        M_0 = 2 sin(lam)/lam,  M_1 = (2 cos(lam) - M_0)/(i lam),  M_2 = (B_2 - 4 M_1)/(i lam),
        M_{k+1} = [B_{k+1} - (k+1)(2 M_k + (B_{k-1} - i lam M_{k-1})/(k-1))]/(i lam),
        B_n = exp(i lam) - (-1)^n exp(-i lam).
    Otherwise: 384-point Gauss-Legendre in mpmath, exact to far below DPS digits for
    |lam| < 2N + 10 <= 120 (polynomial degree 767 against T_k exp(i lam s)).
    """
    with mp.workdps(DPS + GUARD):
        lam = mp.mpf(lam)
        if abs(lam) >= 2 * N + 10:
            il = mp.mpc(0, lam)
            e_p = mp.expj(lam)
            e_m = mp.expj(-lam)
            B = [e_p - (1 if n % 2 == 0 else -1) * e_m for n in range(N + 2)]
            M = [mp.mpc(2 * mp.sin(lam) / lam)]
            M.append((2 * mp.cos(lam) - M[0]) / il)
            if N >= 2:
                M.append((B[2] - 4 * M[1]) / il)
            for k in range(2, N):
                M.append(
                    (
                        B[k + 1]
                        - (k + 1) * (2 * M[k] + (B[k - 1] - il * M[k - 1]) / (k - 1))
                    )
                    / il
                )
            M = M[: N + 1]
        else:
            nodes = _gl_nodes()
            M = [mp.mpc(0)] * (N + 1)
            for x, w in nodes:
                e = w * mp.expj(lam * x)
                # T_k(x) by recurrence
                t0, t1 = mp.mpf(1), x
                M[0] += e
                if N >= 1:
                    M[1] += e * t1
                for k in range(2, N + 1):
                    t0, t1 = t1, 2 * x * t1 - t0
                    M[k] += e * t1
    return [+m for m in M]


# ------------------------------------------------------------------------------------------
# Clenshaw-Curtis points and the cosine table


_CC_CACHE = {}


def cc_table(N):
    """(s_j, cos(j k pi / N)) for j, k = 0..N, at the working precision."""
    key = (N, mp.mp.prec)
    tab = _CC_CACHE.get(key)
    if tab is None:
        s = [mp.cos(j * mp.pi / N) for j in range(N + 1)]
        C = [[mp.cos(j * k * mp.pi / N) for k in range(N + 1)] for j in range(N + 1)]
        tab = (s, C)
        _CC_CACHE[key] = tab
    return tab


def cheb_coeffs(gvals, N):
    """c_k = (2/N) sum''_j g_j cos(j k pi/N): the interpolant is sum''_k c_k T_k."""
    _, C = cc_table(N)
    out = []
    for k in range(N + 1):
        acc = mp.mpc(0)
        for j in range(N + 1):
            term = gvals[j] * C[j][k]
            if j == 0 or j == N:
                term = term / 2
            acc += term
        out.append(2 * acc / N)
    return out


def cheb_deriv_at_nodes(c, N):
    """d/ds of sum''_k c_k T_k(s) at the N + 1 Clenshaw-Curtis points."""
    a = list(c)
    a[0] = a[0] / 2
    a[N] = a[N] / 2
    d = [mp.mpc(0)] * (N + 2)
    for k in range(N, 0, -1):
        d[k - 1] = d[k + 1] + 2 * k * a[k]
    d[0] = d[0] / 2
    _, C = cc_table(N)
    return [mp.fsum(d[k] * C[j][k] for k in range(N)) for j in range(N + 1)]


def fcc_sum(c, M, N):
    """sum''_k c_k M_k."""
    acc = mp.mpc(0)
    for k in range(N + 1):
        t = c[k] * M[k]
        if k == 0 or k == N:
            t = t / 2
        acc += t
    return acc


def fcc_panel(gfun_mp, cen, h, omega, N):
    """
    int_{cen-h}^{cen+h} g(y) exp(i omega y) dy by FCC with N + 1 points. cen, h, omega are mpf
    (h may be negative); gfun_mp(y_mp) returns g as mpc (the caller decides how it evaluates).
    """
    s, _ = cc_table(N)
    gv = [gfun_mp(cen + h * sj) for sj in s]
    c = cheb_coeffs(gv, N)
    M = cheb_moments(omega * h, N)
    return h * mp.expj(omega * cen) * fcc_sum(c, M, N)


def fcc_integrate(gfun_mp, edges, omega, N):
    """sum of fcc_panel over consecutive edges (mp)."""
    total = mp.mpc(0)
    for l, r in zip(edges[:-1], edges[1:]):
        total += fcc_panel(gfun_mp, (l + r) / 2, (r - l) / 2, omega, N)
    return total


# ------------------------------------------------------------------------------------------
# QAWO, panel by panel


def round_bits(x: Fraction, bits: int) -> float:
    """x rounded to `bits` significant bits, as a double (exact: bits <= 53)."""
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    ax = abs(x)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    # scale so that the integer part has `bits` bits
    shift = bits - 1 - e
    scaled = ax * (Fraction(2) ** shift)
    n = round(scaled)
    return sign * math.ldexp(float(n), -shift)


def dyadic_tiling(L, R, h_max):
    """
    Tile [L, R] (mpf, L < R) into panels (centre mpf, half-width double).

    Full panels have half-width h = 2^e <= h_max. The remainder is peeled into dyadic panels of
    half-width h/2, h/4, ... while it is at least 2^-SLIVER_BITS of 2h, and what is left after that
    becomes one last panel of arbitrary half-width (its phase span is at most 2^-SLIVER_BITS of a
    full panel's, so the rounding of omega times its half-width is negligible).
    """
    e = math.floor(math.log2(h_max))
    h = math.ldexp(1.0, e)
    out = []
    length = R - L
    J = int(mp.floor(length / (2 * h)))
    pos = L
    for _ in range(J):
        out.append((pos + h, h))
        pos = pos + 2 * h
    rem = R - pos
    hh = h / 2
    min_width = 2 * h * 2.0 ** (-SLIVER_BITS)
    while rem >= min_width and hh > 0:
        if rem >= 2 * hh:
            out.append((pos + hh, hh))
            pos = pos + 2 * hh
            rem = R - pos
        hh = hh / 2
    if rem > 0:
        hr = rem / 2
        out.append((pos + hr, float(hr)))
    return out


def qawo_panel(gfun_d, cen, h, omega_r, d_omega, epsabs, epsrel, limit=200, maxp1=100):
    """
    int_{-h}^{h} G(s) exp(i omega_r s) ds with G(s) = g(fl(cen) + s) exp(i d_omega s), in double, by
    four QAWO calls (real/imaginary parts against cos/sin) sharing one memo of G. Returns
    (complex result, info dict).
    """
    cf = float(cen)
    memo = {}

    def G(s):
        v = memo.get(s)
        if v is None:
            v = gfun_d(cf + s)
            if d_omega != 0.0:
                v = v * cmath.exp(1j * d_omega * s)
            memo[s] = v
        return v

    w = abs(omega_r)
    sg = 1.0 if omega_r >= 0 else -1.0
    res = {}
    info = {"ier": [], "last": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for part, fn in (("r", lambda s: G(s).real), ("i", lambda s: G(s).imag)):
            for wt in ("cos", "sin"):
                out = integrate.quad(
                    fn,
                    -h,
                    h,
                    weight=wt,
                    wvar=w,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                    maxp1=maxp1,
                    full_output=1,
                )
                res[part + wt] = out[0]
                d = out[2]
                # scipy appends a message only when QUADPACK's ier != 0
                info["ier"].append(1 if len(out) > 3 else 0)
                info["abserr"] = info.get("abserr", 0.0) + abs(out[1])
                info["last"] = max(info["last"], d.get("last", 0))
    info["neval"] = len(memo)
    # exp(i omega_r s) = cos(w s) + i sg sin(w s)
    re = res["rcos"] - sg * res["isin"]
    im = sg * res["rsin"] + res["icos"]
    return complex(re, im), info


def qawo_integrate(gfun_d, panels, omega_exact: Fraction, epsabs_fn, epsrel):
    """
    sum over panels (cen mpf, h double) of exp(i omega cen) * qawo_panel(...), in mpmath.
    epsabs_fn(cen, h) gives the absolute tolerance per panel.
    """
    omega_r = round_bits(omega_exact, OMEGA_BITS)
    d_omega = float(omega_exact - Fraction(omega_r))
    om_mp = mp.mpf(omega_exact.numerator) / omega_exact.denominator
    total = mp.mpc(0)
    stats = {
        "panels": 0,
        "neval": 0,
        "ier_nonzero": 0,
        "max_last": 0,
        "sum_quad_abserr": 0.0,
    }
    for cen, h in panels:
        v, info = qawo_panel(
            gfun_d, cen, h, omega_r, d_omega, epsabs_fn(cen, h), epsrel
        )
        total += mp.expj(om_mp * cen) * mp.mpc(v.real, v.imag)
        stats["panels"] += 1
        stats["neval"] += info["neval"]
        stats["ier_nonzero"] += sum(1 for x in info["ier"] if x)
        stats["max_last"] = max(stats["max_last"], info["last"])
        stats["sum_quad_abserr"] += info["abserr"]
    return total, stats
