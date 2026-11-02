"""
Filon-type references for a captured ``adaptive_levin_sincos`` call
(prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md, section 0.2 (e)).

A captured call is the dict 06c's ``x1_capture.py`` pickles: ``x_span``, ``f`` (the pair
``[f_sin, f_cos]``), ``theta`` (the phase dict) and ``kw``. Its integral is

    I = int_{x_a}^{x_b} [f_sin(x) sin theta(x) + f_cos(x) cos theta(x)] dx
      = Re int F(x) exp(i theta(x)) dx,      F = f_cos - i f_sin.

This module returns a reference for I, independent of the Levin driver, by making the phase linear
and integrating against it with a Filon-type rule, in two implementations at two resolutions each.
It is 06i's method (docs/levin-phase-floor/06i/scripts/: ``filon.py``, ``route_a.py``,
``route_b.py``, ``samples.py``), turned into a module; 06i's design choices (its deviations 2-6
and 10) are kept.

**The method.** For ``int g(y) exp(i omega y) dy`` with ``g`` smooth, a Filon-type rule interpolates
``g`` alone and integrates the interpolant against ``exp(i omega y)`` exactly, through the moments
``M_k(lam) = int_{-1}^{1} T_k(s) exp(i lam s) ds``. The cancellation between oscillations happens
inside the exact moments, so the rounding is about ``eps |g| / omega``, the size of the answer, not
of ``int |g|``, and the cost depends on ``g``'s smoothness, not on the number of oscillations. The
two implementations, each checked against the other:

* **FCC** (Filon-Clenshaw-Curtis) in ``mpmath``: on each panel ``g`` is sampled at the ``N + 1``
  Clenshaw-Curtis points (evaluated in double, by the call's own callables), its Chebyshev
  coefficients are formed in ``mpmath``, and the moments are computed at ``DPS`` digits plus
  ``GUARD`` guard digits (a forward three-term recurrence for ``|lam| >= 2N + 10``, 384-point
  Gauss-Legendre below); panels are summed in ``mpmath``.
* **QAWO** (QUADPACK's adaptive rule for a cos or sin weight, ``scipy.integrate.quad(weight=...,
  wvar=...)``) in double precision, panel by panel, with every product it forms internally exact:
  ``omega`` is replaced by ``omega_r``, ``omega`` rounded to ``OMEGA_BITS`` significant bits, with
  the remainder ``(omega - omega_r) s`` folded into the amplitude; panels are symmetric, ``[-h, h]``
  with ``h`` a power of two, so QUADPACK's bisection yields only dyadic centres, and ``omega_r``
  times any of them (down to ``h / 2^13``) is exact; ``exp(i omega c)`` at each panel's centre is
  formed in ``mpmath``. A segment's remainder is peeled into dyadic panels down to
  ``2^-SLIVER_BITS`` of a full panel, plus one last small panel. (A panel shift alone would leave
  ``omega s`` rounded inside QAWO, about ``eps int |g|`` per panel: 06i's deviation 2.)

**Route A, an analytic Bessel block** (the phase is a ``BesselPhaseGroup``'s ``theta``, found as
``theta["theta"].__self__``). The variable is ``u = log eta`` and the phase is exactly
``theta = K eta + C + R(eta)``; in ``eta``,

    I = Re int_{e^{u_a}}^{e^{u_b}} g(eta) exp(i K eta) d eta,   g = F(log eta) exp(i (C + R(eta))) / eta,

a linear phase with ``omega = K``. ``K`` is taken **exactly**, as the sum of the group's signed
coefficients in ``fractions.Fraction`` (each term exact), so the reference is for the intended
integrand, independent of how any version of the group rounds ``K eta``; ``C`` is the sum of
``e_i (pi/4 - pi nu_i / 2)`` in ``mpmath``; the residual ``R`` and the amplitude are the group's
own, evaluated in double at the double ``eta``. Panels are uniform in ``u`` between each
constituent's representation switches (its near-region interpolants' breakpoints and its
``x_star``, mapped by ``u = log x - log m_i``), so that the amplitude and residual are analytic
inside each panel (06i's deviation 3). It needs no inverse map and no monotone phase.

**Route B, a cosmological group** (any other phase dict, which must supply ``theta``,
``theta_mod_2pi`` and ``theta_deriv``). The variable is the phase. On each ``x`` segment the map
``t = p(x)`` is a least-squares smooth phase: a polynomial of degree ``M`` in the segment's
variable, fitted to the unwrapped captured phase ``raw + rem(theta_mod_2pi - raw)`` at ``K``
Chebyshev-Lobatto points, then refitted with the edge values shared with its neighbours, so that
``t`` is continuous (06i's deviation 4). Then exactly

    int_seg F(x) exp(i p(x)) dx = int_{p(X_l)}^{p(X_r)} [F(x(t)) / p'(x(t))] exp(i t) dt,

linear with ``omega = 1``. ``x(t)`` is solved by Newton's method (in double, then one step in
``mpmath``), the callables are evaluated at ``x^ = fl(x(t))``, and the sample, taken at
``t^ = p(x^)``, is corrected to first order in ``t^ - t`` (06i's deviation 6). The reference is the
integral of the captured amplitude against the **smooth** phase: ``theta_mod_2pi`` carries a white
jitter (about ``0.6 eps |theta'|``, the rounding of the abscissa map), which a rule with Filon's few
nodes would otherwise integrate; every dense quadrature of the captured integrand approaches the
smooth-phase integral in the limit (06i, the user's ruling on R0). Its error terms are reported:
the fit's residual rms per segment (``sigma``), the fit-noise bound (the 1-sigma error of the
reference due to ``t(x)``'s noise, propagated through the fit's covariance; FCC only), the jitter's
estimated contribution at the FCC nodes, the abscissa correction, monotonicity and the
``2 pi`` branch margin. A segment that holds a stationary point of the phase (``p'`` of both signs
at 161 points, or ``theta_deriv`` disagreeing with it; the segment's ``monotone`` False) has no
inverse ``x(t)``, and is integrated **directly** in its variable ``y``,
``half int_{-1}^{1} F(x(y)) exp(i p(y)) dy``, against the same smooth ``p``, on uniform ``y``-panels
of phase span at most ``dphi``: by Gauss-Legendre in ``mpmath`` in the FCC readings and by
QUADPACK's QAG in double in the QAWO readings. The phase is taken at the node itself, the amplitude
at ``fl(x(y))``. The FCC readings also integrate directly a monotone segment beside a stationary
point, where ``1/p'`` is nearly singular in ``t``: one whose ``max |p'| / min |p'|`` (its
``slope_ratio``) exceeds ``ROUTE_B_FCC_DIRECT_SLOPE_RATIO``; QAWO's adaptive rule keeps such a
segment's ``t``-panels. Such a reading reports ``all_monotone`` False and ``direct_segments``, the
number of segments so integrated; the inverse map refuses a segment with a stationary point
(``ValueError``), and so does either Newton iteration where ``dp/dy`` is exactly zero. (Before
2026-11-02 the tool solved for ``x(t)`` on such a segment regardless, and until the slope-ratio rule
FCC kept ``t``-panels beside one: those readings are unsound.) The segment edges are the
caller's (``edges=``), or else found in the captured integrand's closures as 06i found them (the
phase splines' knots, the Green's function amplitude spline's knots, and every cumulative table's
nodes, switch midpoints and breaks: 06i's deviation 5), which ties that default to the integrand's
layout at the census tree.

**The readings and the count rule.** :func:`reference` takes four readings, both implementations at
both resolutions, and reports each as a hex double, with 30 digits, its imaginary part and the
route's own error terms; the readings' scatter (largest minus smallest); and the verdict, which is
the rule 06i used and 06g's census uses: **a reference counts when its four readings agree to
within the caller's tolerance** (scatter at most ``tolerance``). The reference value is the FCC
reading at resolution 2 (06i's choice). The resolutions:

* Route A: FCC ``du = 0.25, N = 32`` and ``du = 0.17, N = 48``; QAWO ``du = 0.25``, panels of phase
  span at most ``2^12`` rad, ``epsrel = 1e-13``, and ``du = 0.125``, ``2^9`` rad, ``1.2e-14``.
* Route B: fit ``(M, K) = (20, 241)`` and ``(24, 301)``; FCC one ``t``-panel per segment with
  ``N = 32``, three with ``N = 48``; QAWO dyadic ``t``-panels of span at most ``2^12`` rad with
  ``epsrel = 1e-13``, and ``2^10`` rad with ``1.2e-14``. A segment with a stationary point (or, for
  FCC, a slope ratio above 2): FCC readings 24-point Gauss-Legendre on panels of at most 2 rad, and
  48-point on 1 rad; QAWO readings
  QAG on 4 rad with ``epsrel = 1e-13``, and on 2 rad with ``1.2e-14``.
* :func:`linear_phase_integral`, for a ``g`` and ``omega`` given directly (06i's Gate R0): FCC
  ``N = 32`` on 4 panels and ``N = 48`` on 7; QAWO as Route A's, with panels at most a sixteenth of
  the span.

**What it does not do.** It does not test the phase's representation: every reference is built from
the call's own amplitude and phase (Route A from the group's own ``K``, ``C`` and residual; Route B
from its own captured phase, smoothed). It does not load a captured call: a pickle written through
Ray's cloudpickle is loaded by
the caller (the command line below takes the unpickling module by name). It is a tool, not
production code: nothing in ``ComputeTargets/``, ``LiouvilleGreen/`` or ``AdaptiveLevin/`` imports
it, and it imports neither Ray nor a datastore.

**Work and determinism.** A reading is split into chunks of a fixed number of panels or segments,
independent of the number of worker processes, and the chunks' sums are added in order, so a
reading does not depend on ``workers``. With ``workers > 1`` the chunks run in spawned processes,
each of which rebuilds the call from ``call_source = (module, function, args)``.

Command line: ``python -m tools.filon_reference PICKLE --tolerance T [--unpickler ray.cloudpickle]
[--workers N] [--route A|B] [--out JSON]``.
"""

import argparse
import cmath
import importlib
import json
import math
import multiprocessing
import time
import warnings
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from mpmath.calculus.quadrature import GaussLegendre
from scipy import integrate

#: decimal digits of the mpmath arithmetic (06i's 40)
DPS = 40
#: extra digits inside the moment computation
GUARD = 25
#: significant bits omega is rounded to before QAWO sees it, so every product it forms is exact
OMEGA_BITS = 40
#: the smallest dyadic panel peeled off a segment's remainder, as a power of two of a full panel
SLIVER_BITS = 12

IMPLEMENTATIONS = ("qawo", "fcc")
RESOLUTIONS = (1, 2)

ROUTE_A_FCC = {1: dict(du=0.25, N=32), 2: dict(du=0.17, N=48)}
ROUTE_A_QAWO = {
    1: dict(du=0.25, Pmax=2.0**12, epsrel=1e-13),
    2: dict(du=0.125, Pmax=2.0**9, epsrel=1.2e-14),
}
ROUTE_B_FIT = {1: dict(M=20, K=241), 2: dict(M=24, K=301)}
ROUTE_B_FCC = {1: dict(pieces=1, N=32), 2: dict(pieces=3, N=48)}
ROUTE_B_QAWO = {
    1: dict(Pmax=2.0**12, epsrel=1e-13),
    2: dict(Pmax=2.0**10, epsrel=1.2e-14),
}
#: a Route B segment holding a stationary point, integrated directly in its variable y on uniform
#: panels of phase span at most dphi: the FCC readings by Gauss-Legendre in mpmath (24 and 48
#: points), the QAWO readings by QUADPACK's QAG in double
ROUTE_B_DIRECT_FCC = {1: dict(dphi=2.0, degree=4), 2: dict(dphi=1.0, degree=5)}
ROUTE_B_DIRECT_QAWO = {
    1: dict(dphi=4.0, epsrel=1e-13),
    2: dict(dphi=2.0, epsrel=1.2e-14),
}
#: FCC also integrates directly a monotone segment whose max |p'| / min |p'| exceeds this. Beside a
#: stationary point 1/p' is nearly singular in t, which a fixed-order rule cannot resolve: on V1's
#: w07 the segment next to it, at ratio 24.9, was 0.1 and 1.8e-3 out at FCC resolutions 1 and 2,
#: and on the Fresnel test case one at ratio 3.78 was about 2e-13 out at resolution 1; segments at
#: 1.93 and 2.07 were within 5e-15. Every segment of the 41 monotone Route B samples of 06g's
#: references is at most 1.08 (theta_deriv at 161 points), so none of them is affected. QAWO's
#: adaptive rule resolves such a segment, and keeps its t-panels.
ROUTE_B_FCC_DIRECT_SLOPE_RATIO = 2.0
LINEAR_FCC = {1: dict(N=32, panels=4), 2: dict(N=48, panels=7)}
LINEAR_QAWO = {
    1: dict(Pmax=2.0**12, epsrel=1e-13),
    2: dict(Pmax=2.0**9, epsrel=1.2e-14),
}

#: panels (Route A) or segments (Route B) per chunk of work: fixed, so a reading is the same
#: whatever the number of workers
ROUTE_A_FCC_CHUNK = 8
ROUTE_A_QAWO_CHUNK = 256
ROUTE_B_CHUNK = 16


# ------------------------------------------------------------------------------------------------
# The moments, the Clenshaw-Curtis table, and FCC


_GL_CACHE = {}


def _gl_nodes(degree=8):
    """``3 2^(degree-1)``-point Gauss-Legendre nodes and weights on [-1, 1] at the working
    precision (384 points at the default ``degree = 8``)."""
    key = (degree, mp.mp.prec)
    nodes = _GL_CACHE.get(key)
    if nodes is None:
        nodes = GaussLegendre(mp.mp).calc_nodes(degree, mp.mp.prec)
        _GL_CACHE[key] = nodes
    return nodes


def cheb_moments(lam, N):
    """
    ``[M_0, ..., M_N]``, ``M_k = int_{-1}^{1} T_k(s) exp(i lam s) ds``, as ``mpc`` at the working
    precision. For ``|lam| >= 2N + 10``, the forward recurrence from integration by parts (stable
    while ``k < |lam|``), at ``GUARD`` extra digits:

        M_0 = 2 sin(lam)/lam,  M_1 = (2 cos(lam) - M_0)/(i lam),  M_2 = (B_2 - 4 M_1)/(i lam),
        M_{k+1} = [B_{k+1} - (k+1)(2 M_k + (B_{k-1} - i lam M_{k-1})/(k-1))]/(i lam),
        B_n = exp(i lam) - (-1)^n exp(-i lam).

    Otherwise 384-point Gauss-Legendre in ``mpmath``, exact far below the working precision for
    ``|lam| < 2N + 10 <= 120`` (polynomial degree 767 against ``T_k exp(i lam s)``).
    """
    with mp.workdps(mp.mp.dps + GUARD):
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
            M = [mp.mpc(0)] * (N + 1)
            for x, w in _gl_nodes():
                e = w * mp.expj(lam * x)
                t0, t1 = mp.mpf(1), x
                M[0] += e
                if N >= 1:
                    M[1] += e * t1
                for k in range(2, N + 1):
                    t0, t1 = t1, 2 * x * t1 - t0
                    M[k] += e * t1
    return [+m for m in M]


_CC_CACHE = {}


def cc_table(N):
    """``(s_j, cos(j k pi / N))`` for ``j, k = 0..N``, at the working precision."""
    key = (N, mp.mp.prec)
    table = _CC_CACHE.get(key)
    if table is None:
        s = [mp.cos(j * mp.pi / N) for j in range(N + 1)]
        C = [[mp.cos(j * k * mp.pi / N) for k in range(N + 1)] for j in range(N + 1)]
        table = (s, C)
        _CC_CACHE[key] = table
    return table


def cheb_coeffs(gvals, N):
    """``c_k = (2/N) sum''_j g_j cos(j k pi / N)``: the interpolant is ``sum''_k c_k T_k``."""
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
    """``d/ds`` of ``sum''_k c_k T_k(s)`` at the ``N + 1`` Clenshaw-Curtis points."""
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
    """``sum''_k c_k M_k``."""
    acc = mp.mpc(0)
    for k in range(N + 1):
        t = c[k] * M[k]
        if k == 0 or k == N:
            t = t / 2
        acc += t
    return acc


def fcc_panel(gfun_mp, cen, h, omega, N):
    """
    ``int_{cen-h}^{cen+h} g(y) exp(i omega y) dy`` by FCC with ``N + 1`` points; ``cen``, ``h``,
    ``omega`` are ``mpf`` (``h`` may be negative), and ``gfun_mp(y)`` returns ``g`` as ``mpc``.
    """
    s, _ = cc_table(N)
    gv = [gfun_mp(cen + h * sj) for sj in s]
    c = cheb_coeffs(gv, N)
    M = cheb_moments(omega * h, N)
    return h * mp.expj(omega * cen) * fcc_sum(c, M, N)


# ------------------------------------------------------------------------------------------------
# QAWO, panel by panel


def round_bits(x: Fraction, bits: int) -> float:
    """``x`` rounded to ``bits`` significant bits, as a double (exact for ``bits <= 53``)."""
    if x == 0:
        return 0.0
    sign = -1 if x < 0 else 1
    ax = abs(x)
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    shift = bits - 1 - e
    n = round(ax * (Fraction(2) ** shift))
    return sign * math.ldexp(float(n), -shift)


def dyadic_tiling(L, R, h_max):
    """
    Tile ``[L, R]`` (``mpf``, ``L < R``) into panels ``(centre mpf, half-width double)``. Full panels
    have half-width ``h = 2^e <= h_max``; the remainder is peeled into dyadic panels ``h/2, h/4,
    ...`` while it is at least ``2^-SLIVER_BITS`` of ``2h``, and what is left becomes one last panel
    of arbitrary half-width, whose phase span is at most ``2^-SLIVER_BITS`` of a full panel's.
    """
    h = math.ldexp(1.0, math.floor(math.log2(h_max)))
    out = []
    count = int(mp.floor((R - L) / (2 * h)))
    pos = L
    for _ in range(count):
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
    ``int_{-h}^{h} G(s) exp(i omega_r s) ds`` with ``G(s) = g(fl(cen) + s) exp(i d_omega s)``, in
    double, by four QAWO calls (real and imaginary parts against cos and sin) sharing one memo of
    ``G``. Returns (the complex result, an info dict).
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
    info = {"ier": [], "messages": [], "last": 0, "abserr": 0.0}
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
                _note_quad(info, out)
    info["neval"] = len(memo)
    # exp(i omega_r s) = cos(w s) + i sg sin(w s)
    re = res["rcos"] - sg * res["isin"]
    im = sg * res["rsin"] + res["icos"]
    return complex(re, im), info


def _note_quad(info, out):
    """Record one ``integrate.quad(..., full_output=1)`` call's error, subdivisions and flag."""
    # scipy appends a message only when QUADPACK's ier != 0; its first line names the ier
    flagged = len(out) > 3
    info["ier"].append(1 if flagged else 0)
    if flagged:
        info["messages"].append(str(out[3]).strip().splitlines()[0][:80])
    info["abserr"] += abs(out[1])
    info["last"] = max(info["last"], out[2].get("last", 0))


def _new_qawo_stats():
    return {
        "panels": 0,
        "direct_panels": 0,
        "neval": 0,
        "ier_nonzero": 0,
        "ier_messages": {},
        "max_last": 0,
        "sum_quad_abserr": 0.0,
    }


def _add_qawo_stats(stats, info, direct=False):
    stats["panels"] += 1
    if direct:
        stats["direct_panels"] += 1
    stats["neval"] += info["neval"]
    stats["ier_nonzero"] += sum(1 for x in info["ier"] if x)
    for message in info["messages"]:
        stats["ier_messages"][message] = stats["ier_messages"].get(message, 0) + 1
    stats["max_last"] = max(stats["max_last"], info["last"])
    stats["sum_quad_abserr"] += info["abserr"]


def _merge_qawo_stats(parts):
    out = _new_qawo_stats()
    for st in parts:
        for key in (
            "panels",
            "direct_panels",
            "neval",
            "ier_nonzero",
            "sum_quad_abserr",
        ):
            out[key] += st[key]
        for message, n in st["ier_messages"].items():
            out["ier_messages"][message] = out["ier_messages"].get(message, 0) + n
        out["max_last"] = max(out["max_last"], st["max_last"])
    return out


def qawo_integrate(gfun_d, panels, omega_exact: Fraction, epsabs_fn, epsrel):
    """
    The sum over ``panels`` (``(cen mpf, h double)``) of ``exp(i omega cen) qawo_panel(...)``, in
    ``mpmath``; ``epsabs_fn(cen, h)`` gives each panel's absolute tolerance. Returns (``mpc``,
    stats).
    """
    omega_r = round_bits(omega_exact, OMEGA_BITS)
    d_omega = float(omega_exact - Fraction(omega_r))
    om_mp = mp.mpf(omega_exact.numerator) / omega_exact.denominator
    total = mp.mpc(0)
    stats = _new_qawo_stats()
    for cen, h in panels:
        v, info = qawo_panel(
            gfun_d, cen, h, omega_r, d_omega, epsabs_fn(cen, h), epsrel
        )
        total += mp.expj(om_mp * cen) * mp.mpc(v.real, v.imag)
        _add_qawo_stats(stats, info)
    return total, stats


# ------------------------------------------------------------------------------------------------
# A linear phase given directly (06i's Gate R0)


def linear_phase_integral(gfun_d, L, R, omega, impl, res):
    """
    ``int_L^R g(y) exp(i omega y) dy`` for a ``g`` evaluated in double (``gfun_d(float) ->
    complex``) and an ``omega`` given exactly (a float, or a ``Fraction``), by ``impl`` ("fcc" or
    "qawo") at resolution ``res``; ``L < R`` as ``mpf`` or floats. Returns a reading dict (as
    :func:`reference`'s) with its value as ``mpc``, under the key "value_mp". The QAWO panel's
    absolute tolerance is ``1e-17 |g(c)| min(2h, 2/omega)``.
    """
    start = time.perf_counter()
    with mp.workdps(DPS):
        L = mp.mpf(L)
        R = mp.mpf(R)
        omega_exact = Fraction(omega)
        om = mp.mpf(omega_exact.numerator) / omega_exact.denominator
        if impl == "fcc":
            cfg = LINEAR_FCC[res]
            n = cfg["panels"]
            edges = [L + (R - L) * j / n for j in range(n + 1)]

            def gfun_mp(y):
                v = gfun_d(float(y))
                return mp.mpc(v.real, v.imag)

            value = mp.mpc(0)
            for l, r in zip(edges[:-1], edges[1:]):
                value += fcc_panel(gfun_mp, (l + r) / 2, (r - l) / 2, om, cfg["N"])
            stats = {"panels": n}
        elif impl == "qawo":
            cfg = LINEAR_QAWO[res]
            w = abs(float(omega_exact))
            h_max = min(cfg["Pmax"] / (2 * w), float(R - L) / 16)
            panels = dyadic_tiling(L, R, h_max)
            value, stats = qawo_integrate(
                gfun_d,
                panels,
                omega_exact,
                lambda c, h: 1e-17 * abs(gfun_d(float(c))) * min(2 * h, 2 / w),
                cfg["epsrel"],
            )
        else:
            raise ValueError(f"unknown implementation {impl!r}")
        return _reading("linear", impl, res, cfg, value, start, stats=stats)


# ------------------------------------------------------------------------------------------------
# Readings


def _reading(route, impl, res, cfg, value, start, **extra):
    out = {
        "route": route,
        "impl": impl,
        "res": res,
        "config": dict(cfg),
        "value_hex": float(value.real).hex(),
        "value_30": mp.nstr(value.real, 30),
        "imag_30": mp.nstr(value.imag, 30),
        "value_mp": value,
        "wall_s": time.perf_counter() - start,
    }
    out.update(extra)
    return out


def scatter(readings):
    """The largest minus the smallest of the readings' real parts, from their 30-digit values."""
    with mp.workdps(DPS):
        values = [mp.mpf(r["value_30"]) for r in readings]
        return float(max(values) - min(values))


def counts(readings, tolerance):
    """06i's and 06g's count rule: the readings agree to within ``tolerance`` (their scatter)."""
    return scatter(readings) <= tolerance


# ------------------------------------------------------------------------------------------------
# Route A


def _rem2pi(v):
    """``v`` reduced into ``(-pi, pi]``, in ``mpmath``."""
    two_pi = 2 * mp.pi
    r = v - two_pi * mp.floor(v / two_pi + mp.mpf(1) / 2)
    if r <= -mp.pi:
        r += two_pi
    if r > mp.pi:
        r -= two_pi
    return r


def route_of(call):
    """ "A" if the call's phase is a Bessel group's ``theta`` (an object with ``signs``,
    ``coefficients``, ``phases`` and ``residual``), else "B"."""
    owner = getattr(call["theta"]["theta"], "__self__", None)
    if owner is not None and all(
        hasattr(owner, name) for name in ("signs", "coefficients", "phases", "residual")
    ):
        return "A"
    return "B"


class RouteA:
    """A captured analytic-block call, in ``eta``: see the module docstring."""

    def __init__(self, call):
        self.call = call
        self.group = call["theta"]["theta"].__self__
        g = self.group
        self.f_sin, self.f_cos = call["f"]
        self.u_a, self.u_b = (float(v) for v in call["x_span"])
        # K exactly, from the float coefficients and signs (each term exact)
        self.K_exact = sum(
            (Fraction(s) * Fraction(c) for s, c in zip(g.signs, g.coefficients)),
            Fraction(0),
        )
        self.K_hat = float(g.K) if hasattr(g, "K") else float(self.K_exact)
        self.K_lo = float(self.K_exact - Fraction(self.K_hat))

    def K_mp(self):
        return mp.mpf(self.K_exact.numerator) / self.K_exact.denominator

    def C_mp(self):
        g = self.group
        return mp.fsum(
            s * (mp.pi / 4 - mp.pi * mp.mpf(p.nu) / 2)
            for s, p in zip(g.signs, g.phases)
        )

    def F(self, u):
        return complex(self.f_cos(u), -self.f_sin(u))

    def g_double(self, C_d):
        group = self.group
        F = self.F

        def gd(eta):
            R = group.residual(eta)
            return (
                F(math.log(eta)) * complex(math.cos(C_d + R), math.sin(C_d + R)) / eta
            )

        return gd

    def g_mp(self):
        """``g`` at an ``mpf`` ``eta``: amplitude and residual in double at ``fl(eta)``, the rest
        in ``mpmath``."""
        group = self.group
        C = self.C_mp()
        F = self.F

        def gm(eta):
            Fv = F(float(mp.log(eta)))
            return (
                mp.mpc(Fv.real, Fv.imag) * mp.expj(C + group.residual(float(eta))) / eta
            )

        return gm

    def u_breaks(self):
        """``u`` at which a constituent's representation switches: its near-region panel edges
        and ``x_star`` (none for a constituent that does not expose them)."""
        out = set()
        for p, m in zip(self.group.phases, self.group.coefficients):
            c = getattr(p, "_corrections", None)
            if c is None:
                continue
            lm = math.log(m)
            cand = [math.log(c.x_star) - lm]
            near = getattr(c, "near", None)
            if near is not None:
                cand += [float(b) - lm for b in near.r_interp.breakpoints]
                cand += [float(b) - lm for b in near.log_a_interp.breakpoints]
            for v in cand:
                if self.u_a < v < self.u_b:
                    out.add(v)
        return sorted(out)

    def u_segments(self, du_max):
        """``u`` edges (``mpf``): the span, the breaks, and uniform subdivision to ``du_max``."""
        pts = [self.u_a] + self.u_breaks() + [self.u_b]
        edges = [mp.mpf(self.u_a)]
        for l, r in zip(pts[:-1], pts[1:]):
            n = max(1, math.ceil((r - l) / du_max))
            L, R = mp.mpf(l), mp.mpf(r)
            for j in range(1, n + 1):
                edges.append(L + (R - L) * j / n)
        return edges

    def panels(self, impl, res):
        """The reading's panels: ``(eta_l, eta_r)`` (FCC) or ``(centre, half-width)`` (QAWO)."""
        if impl == "fcc":
            ue = self.u_segments(ROUTE_A_FCC[res]["du"])
            ee = [mp.exp(u) for u in ue]
            return list(zip(ee[:-1], ee[1:]))
        cfg = ROUTE_A_QAWO[res]
        ue = self.u_segments(cfg["du"])
        ee = [mp.exp(u) for u in ue]
        Kabs = abs(float(self.K_exact))
        out = []
        for l, r in zip(ee[:-1], ee[1:]):
            h_max = min(cfg["Pmax"] / (2 * Kabs), float(r - l) / 2)
            out += dyadic_tiling(l, r, h_max)
        return out


def _route_a_fcc_chunk(state, panels, N):
    gm = state["gmp"]
    K = state["K"]
    total = mp.mpc(0)
    for l, r in panels:
        total += fcc_panel(gm, (l + r) / 2, (r - l) / 2, K, N)
    return total, None


def _route_a_qawo_chunk(state, panels, epsrel):
    A = state["route"]
    gd = state["gd"]
    Kabs = abs(float(A.K_exact))

    def epsabs(c, h):
        return 1e-17 * abs(gd(float(c))) * min(2 * h, 2 / Kabs)

    value, stats = qawo_integrate(gd, panels, A.K_exact, epsabs, epsrel)
    return value, {"stats": stats}


def _route_a_state(call):
    A = RouteA(call)
    C = A.C_mp()
    return {
        "route": A,
        "gmp": A.g_mp(),
        "gd": A.g_double(float(_rem2pi(C))),
        "K": A.K_mp(),
    }


# ------------------------------------------------------------------------------------------------
# Route B


def cheb_eval(c, y):
    """``sum_k c_k T_k(y)`` (a plain sum, not ``sum''``), by Clenshaw; for floats or ``mpf``."""
    b1 = b2 = 0 * y
    for ck in reversed(c[1:]):
        b1, b2 = 2 * y * b1 - b2 + ck, b1
    return y * b1 - b2 + c[0]


def cheb_der(c):
    """The coefficients of ``d/dy sum_k c_k T_k``."""
    n = len(c) - 1
    if n == 0:
        return [0 * c[0]]
    d = [0 * c[0]] * (n + 1)
    for k in range(n, 0, -1):
        d[k - 1] = (d[k + 1] if k + 1 <= n else 0 * c[0]) + 2 * k * c[k]
    d[0] = d[0] / 2
    return d[:n]


def production_edges(call):
    """
    The ``x`` edges inside a cosmological group call's span at which some ingredient of its
    integrand changes representation (06i's ``samples.route_b_edges``): each constituent's phase
    spline knots, the Green's function amplitude spline's knots, and every cumulative table's nodes,
    nearest-node switch midpoints and declared breaks. Found by walking the captured integrand's
    closures, so it holds only for the layout of ``ComputeTargets.QuadSourceIntegral``'s integrand
    that 06i read; a call it does not fit raises, and its caller passes ``edges`` instead. Returns
    (edges, the number of tables).
    """
    a, b = (float(v) for v in call["x_span"])
    ta = call["f"][0].__closure__[2].cell_contents
    cl = dict(
        zip(
            ta.__wrapped__.__code__.co_freevars,
            [c.cell_contents for c in ta.__wrapped__.__closure__],
        )
    )
    active = call["theta"]["theta"].__closure__[0].cell_contents
    xs = set()
    tables = []
    for _, fac in active:
        ph = fac.phase
        xs.update(float(t) for t in ph._spline.t)
        tables.append(ph._leading._table)
    for name in ("q_factor", "r_factor"):
        f = cl[name]
        if hasattr(f, "_f") and hasattr(f._f, "_friction_F"):
            tables.append(f._f._friction_F._table)
    G = cl["G_factor"]
    if hasattr(G, "_Gk"):
        xs.update(float(t) for t in G._Gk.sin_amplitude._spline.t)
    seen = set()
    for t in tables:
        if id(t) in seen:
            continue
        seen.add(id(t))
        asc = np.asarray(t._u_asc, dtype=float)
        xs.update(asc.tolist())
        xs.update((asc[:-1] + (asc[1:] - asc[:-1]) / 2).tolist())
        if t._breaks is not None:
            xs.update(float(v) for v in np.asarray(t._breaks).ravel())
    inner = sorted(v for v in xs if a < v < b)
    edges = [a]
    for v in inner:
        if v - edges[-1] > 1e-9:
            edges.append(v)
    if b - edges[-1] <= 1e-9:
        edges.pop()
    edges.append(b)
    return edges, len(seen)


class Segment:
    """
    One ``x`` segment ``[X_l, X_r]`` of a Route B call, with the map ``t = p(x)``: a polynomial of
    degree ``M`` in ``y = (x - mid)/half``, fitted by least squares to the unwrapped captured phase
    at ``K`` points (doubles ``x_k`` at Chebyshev-Lobatto positions):

        theta~_k = raw_k + rem(theta_mod_2pi_k - raw_k)      (exact, in mpmath)

    ``raw`` only chooses the ``2 pi`` branch; ``theta_mod_2pi`` supplies the value. ``p`` is the
    least-squares smooth phase, and its residual rms, ``sigma``, measures the jitter. ``p = b(y) +
    q(y)``: ``b`` is the ``mpmath`` line through ``theta~`` at the two ends, ``q`` is in double.
    With ``ends = (E_l, E_r)`` (``mpf``) given, ``p`` is refitted with ``p(-1) = E_l`` and ``p(+1) =
    E_r`` exactly, so that adjacent segments share their edge value and ``t(x)`` is continuous.
    """

    def __init__(self, call, X_l, X_r, M=20, K=241, ends=None, samples=None):
        th = call["theta"]
        self.X_l, self.X_r, self.M, self.K = X_l, X_r, M, K
        self.mid = (mp.mpf(X_l) + mp.mpf(X_r)) / 2
        self.half = (mp.mpf(X_r) - mp.mpf(X_l)) / 2
        if samples is None:
            xs = [
                float(self.mid + self.half * mp.cos(mp.pi * i / (K - 1)))
                for i in range(K)
            ]
            xs[0], xs[-1] = X_r, X_l
            vals = []
            branch = 0.0
            for x in xs:
                raw = th["theta"](x)
                mod = th["theta_mod_2pi"](x)
                d = _rem2pi(mp.mpf(mod) - mp.mpf(raw))
                branch = max(branch, float(abs(d)))
                vals.append(mp.mpf(raw) + d)
            samples = (xs, vals, branch)
        self.samples = samples
        xs, vals, self.branch_margin = samples
        ys = np.array([float((mp.mpf(x) - self.mid) / self.half) for x in xs])
        self.b0 = (vals[0] + vals[-1]) / 2
        self.b1 = (vals[0] - vals[-1]) / 2
        resid = np.array(
            [float(v - (self.b0 + self.b1 * mp.mpf(y))) for v, y in zip(vals, ys)]
        )
        V = np.polynomial.chebyshev.chebvander(ys, M)
        if ends is None:
            q, *_ = np.linalg.lstsq(V, resid, rcond=None)
        else:
            E_l, E_r = ends
            qr = float(E_r - (self.b0 + self.b1))
            ql = float(E_l - (self.b0 - self.b1))
            A0, B0 = (qr + ql) / 2, (qr - ql) / 2
            cols = [
                V[:, k] - (V[:, 0] if k % 2 == 0 else V[:, 1]) for k in range(2, M + 1)
            ]
            Vc = np.stack(cols, axis=1)
            rc, *_ = np.linalg.lstsq(Vc, resid - (A0 + B0 * ys), rcond=None)
            q = np.zeros(M + 1)
            q[2:] = rc
            q[0] = A0 - sum(rc[k - 2] for k in range(2, M + 1) if k % 2 == 0)
            q[1] = B0 - sum(rc[k - 2] for k in range(3, M + 1) if k % 2 == 1)
        r = resid - V @ q
        self.sigma = float(np.sqrt(np.sum(r * r) / max(1, K - M - 1)))
        self.max_resid = float(np.max(np.abs(r)))
        self.design = V
        self.q = [float(v) for v in q]
        self.c = [mp.mpf(v) for v in self.q]
        self.c[0] += self.b0
        self.c[1] += self.b1
        self.dc = cheb_der(self.c)
        self.ddc = cheb_der(self.dc)
        self.T_l = cheb_eval(self.c, mp.mpf(-1))
        self.T_r = cheb_eval(self.c, mp.mpf(1))
        self.c_d = [float(v) for v in self.c]
        self.dc_d = [float(v) for v in self.dc]
        self.ddc_d = [float(v) for v in self.ddc]
        self.mid_d = float(self.mid)
        self.half_d = float(self.half)
        # monotone: p' at 161 points has one sign, and it is theta_deriv's sign
        signs = set()
        slopes = []
        for i in range(161):
            slopes.append(cheb_eval(self.dc_d, -1 + 2 * i / 160))
            signs.add(slopes[-1] > 0)
        td = [th["theta_deriv"](x) for x in (X_l, float(self.mid), X_r)]
        signs.update(v > 0 for v in td)
        self.monotone = len(signs) == 1 and all(v != 0 for v in td)
        # max |p'| / min |p'| at the same points: large beside a stationary point
        low = min(abs(s) for s in slopes)
        self.slope_ratio = max(abs(s) for s in slopes) / low if low > 0 else math.inf

    def require_monotone(self):
        """Raise unless ``t = p(x)`` is invertible on this segment."""
        if not self.monotone:
            raise ValueError(
                f"Route B segment [{self.X_l!r}, {self.X_r!r}] holds a stationary point of the "
                "phase: t does not determine x there, so it is integrated directly in y"
            )

    def nonzero_slope(self, y):
        """``dp/dy`` at the double ``y``, refused where it is zero."""
        slope = cheb_eval(self.dc_d, y)
        if slope == 0.0:
            raise ValueError(
                f"Route B segment [{self.X_l!r}, {self.X_r!r}]: dp/dy = 0 at y = {y!r}, a "
                "stationary point the monotonicity check did not find"
            )
        return slope

    def max_abs_slope(self):
        """``max |dp/dy|`` over 161 points of ``[-1, 1]``."""
        return max(abs(cheb_eval(self.dc_d, -1 + 2 * i / 160)) for i in range(161))

    def q_shift(self, shift):
        """``p - shift`` as double Chebyshev coefficients (``shift`` an ``mpf`` near ``p``)."""
        c = list(self.c_d)
        c[0] = float(self.c[0] - shift)
        return c

    def y_of(self, x_mp):
        return (x_mp - self.mid) / self.half

    def p(self, y):
        return cheb_eval(self.c, y)

    def pprime_x(self, y):
        """``dp/dx`` at ``y`` (``mpf``)."""
        return cheb_eval(self.dc, y) / self.half

    def solve_y(self, t, shift=None):
        """``y`` in ``[-1, 1]`` with ``p(y) = t``: Newton in double on ``p(y) - t`` with the constant
        shifted, from a linear guess, then one Newton step in ``mpmath``. Refused on a segment that
        holds a stationary point, where ``t`` does not determine ``y``."""
        self.require_monotone()
        if shift is None:
            shift = t
        c = list(self.c_d)
        c[0] = float(self.c[0] - shift)
        target = float(t - shift)
        span = float(self.T_r - self.T_l)
        y = -1 + 2 * float(t - self.T_l) / span if span != 0 else 0.0
        y = min(1.0, max(-1.0, y))
        for _ in range(30):
            step = (cheb_eval(c, y) - target) / self.nonzero_slope(y)
            yn = min(1.0, max(-1.0, y - step))
            if abs(yn - y) <= 1e-16 * max(1.0, abs(y)):
                y = yn
                break
            y = yn
        ym = mp.mpf(y)
        return ym - (self.p(ym) - t) / cheb_eval(self.dc, ym)


def fit_segments(call, edges, i0, i1, M, K):
    """Segments ``i0 .. i1 - 1``, each refitted to the edge values it shares with its neighbours
    (the average of the two unconstrained fits there)."""
    n = len(edges) - 1
    lo, hi = max(0, i0 - 1), min(n, i1 + 1)
    un = {j: Segment(call, edges[j], edges[j + 1], M=M, K=K) for j in range(lo, hi)}

    def E(j):
        if j == 0:
            return un[0].T_l
        if j == n:
            return un[n - 1].T_r
        return (un[j - 1].T_r + un[j].T_l) / 2

    out = []
    for j in range(i0, i1):
        sg = Segment(
            call,
            edges[j],
            edges[j + 1],
            M=M,
            K=K,
            ends=(E(j), E(j + 1)),
            samples=un[j].samples,
        )
        sg.sigma_unconstrained = un[j].sigma
        sg.edge_jump = (
            float(abs(un[j - 1].T_r - un[j].T_l)) if j > 0 else 0.0,
            float(abs(un[j].T_r - un[j + 1].T_l)) if j + 1 < n else 0.0,
        )
        out.append(sg)
    return out


def _F(call, x):
    f_sin, f_cos = call["f"]
    return complex(f_cos(x), -f_sin(x))


def _segment_diag(sg):
    return dict(
        X_l=sg.X_l,
        X_r=sg.X_r,
        sigma=sg.sigma,
        sigma_unconstrained=sg.sigma_unconstrained,
        max_resid=sg.max_resid,
        branch_margin=sg.branch_margin,
        monotone=sg.monotone,
        direct=not sg.monotone,
        slope_ratio=sg.slope_ratio,
        edge_jump=max(sg.edge_jump),
        t_span=float(sg.T_r - sg.T_l),
    )


def _direct_panels(sg, dphi):
    """The number of uniform ``y``-panels that keeps each one's phase span at most ``dphi``, from a
    quarter above the sampled ``max |dp/dy|``."""
    return max(1, math.ceil(2 * 1.25 * sg.max_abs_slope() / dphi))


def _direct_segment_fcc(call, sg, cfg):
    """
    A segment holding a stationary point, ``half int_{-1}^{1} F(x(y)) exp(i p(y)) dy``, by
    Gauss-Legendre in ``mpmath`` on uniform ``y``-panels. The phase is the smooth ``p`` at the exact
    node and the amplitude is taken at ``fl(x(y))``, so no abscissa correction arises. Returns the
    smooth-phase value, the value with the captured phase (``exp(i delta)``, ``delta`` as in
    :func:`_route_b_fcc_chunk`), the variance white jitter of the segment's sigma puts on ``Re(I)``,
    and ``J_k = i half int F exp(i p) T_k dy`` for the fit-noise bound.
    """
    theta_mod_2pi = call["theta"]["theta_mod_2pi"]
    nodes = _gl_nodes(cfg["degree"])
    n_pan = _direct_panels(sg, cfg["dphi"])
    total = mp.mpc(0)
    total_jit = mp.mpc(0)
    var_jit = 0.0
    Jk = [mp.mpc(0)] * (sg.M + 1)
    for j in range(n_pan):
        c = -1 + (2 * j + 1) * mp.mpf(1) / n_pan
        h = mp.mpf(1) / n_pan
        for s, w in nodes:
            y = c + h * s
            xh = float(sg.mid + sg.half * y)
            Fv = _F(call, xh)
            term = h * w * sg.half * mp.mpc(Fv.real, Fv.imag) * mp.expj(sg.p(y))
            total += term
            delta = _rem2pi(mp.mpf(theta_mod_2pi(xh)) - sg.p(sg.y_of(mp.mpf(xh))))
            total_jit += term * mp.expj(delta)
            var_jit += float(term.imag) ** 2 * sg.sigma**2
            t0, t1 = mp.mpf(1), y
            Jk[0] += mp.mpc(0, 1) * term
            Jk[1] += mp.mpc(0, 1) * term * t1
            for k in range(2, sg.M + 1):
                t0, t1 = t1, 2 * y * t1 - t0
                Jk[k] += mp.mpc(0, 1) * term * t1
    return total, total_jit, var_jit, Jk


def qag_panel(gfun_d, a, b, epsabs, epsrel, limit=200):
    """``int_a^b g(y) dy`` in double by QUADPACK's QAG (``scipy.integrate.quad``, no weight), real
    and imaginary parts sharing one memo of ``g``. Returns (the complex result, an info dict as
    :func:`qawo_panel`'s)."""
    memo = {}

    def G(y):
        v = memo.get(y)
        if v is None:
            v = gfun_d(y)
            memo[y] = v
        return v

    res = {}
    info = {"ier": [], "messages": [], "last": 0, "abserr": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for part, fn in (("r", lambda y: G(y).real), ("i", lambda y: G(y).imag)):
            out = integrate.quad(
                fn, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit, full_output=1
            )
            res[part] = out[0]
            _note_quad(info, out)
    info["neval"] = len(memo)
    return complex(res["r"], res["i"]), info


def _direct_segment_qawo(call, sg, cfg, stats):
    """
    A segment holding a stationary point, ``half int_{-1}^{1} F(x(y)) exp(i p(y)) dy``, by QAG in
    double on uniform ``y``-panels: on each, ``p - p(y_c)`` in double from the shifted coefficients
    (:meth:`Segment.q_shift`), and ``exp(i p(y_c))`` in ``mpmath``. Adds to ``stats``; returns the
    value.
    """
    n_pan = _direct_panels(sg, cfg["dphi"])
    half_d, mid_d = sg.half_d, sg.mid_d
    total = mp.mpc(0)
    for j in range(n_pan):
        ya = -1.0 + 2.0 * j / n_pan
        yb = -1.0 + 2.0 * (j + 1) / n_pan
        yc = (ya + yb) / 2
        pc = sg.p(mp.mpf(yc))
        qc = sg.q_shift(pc)

        def gfun(y, qc=qc):
            return (
                half_d * _F(call, mid_d + half_d * y) * cmath.exp(1j * cheb_eval(qc, y))
            )

        g0 = abs(gfun(yc))
        v, info = qag_panel(gfun, ya, yb, 1e-17 * g0 * (yb - ya), cfg["epsrel"])
        total += mp.expj(pc) * mp.mpc(v.real, v.imag)
        _add_qawo_stats(stats, info, direct=True)
    return total


def _route_b_fcc_chunk(state, i0, i1, res):
    """
    FCC over segments ``i0 .. i1 - 1``: the smooth-phase value, the uncorrected value, the
    fit-noise variance, the abscissa diagnostics, and (06i's addition 1) the same nodes and weights
    with the captured phase, ``g_j exp(i delta_j)``, ``delta_j = rem(theta_mod_2pi(x^_j) - p(x^_j))``,
    with the variance white jitter of each segment's sigma puts on ``Re(I)`` at these nodes.
    """
    call, edges = state["call"], state["edges"]
    fit, cfg = ROUTE_B_FIT[res], ROUTE_B_FCC[res]
    N = cfg["N"]
    _, C = cc_table(N)
    segs = fit_segments(call, edges, i0, i1, fit["M"], fit["K"])
    total = mp.mpc(0)
    total_unc = mp.mpc(0)
    total_jit = mp.mpc(0)
    var = 0.0
    var_jit = 0.0
    max_eps = 0.0
    max_corr = 0.0
    diags = []
    s_nodes, _ = cc_table(N)
    for sg in segs:
        d = _segment_diag(sg)
        Jk = [mp.mpc(0)] * (sg.M + 1)
        P = cfg["pieces"]
        if not sg.monotone or sg.slope_ratio > ROUTE_B_FCC_DIRECT_SLOPE_RATIO:
            # a stationary point, or a segment beside one: no t-panels; the segment is
            # integrated directly in y
            P = 0
            d["direct"] = True
            v_seg, v_jit, var_seg, Jk = _direct_segment_fcc(
                call, sg, ROUTE_B_DIRECT_FCC[res]
            )
            total += v_seg
            total_unc += v_seg
            total_jit += v_jit
            var_jit += var_seg
        for piece in range(P):
            ta = sg.T_l + (sg.T_r - sg.T_l) * piece / P
            tb = sg.T_l + (sg.T_r - sg.T_l) * (piece + 1) / P
            c, h = (ta + tb) / 2, (tb - ta) / 2
            g, eps, ys, dl = [], [], [], []
            for sj in s_nodes:
                t = c + h * sj
                y = sg.solve_y(t, shift=c)
                xh = float(sg.mid + sg.half * y)
                yh = sg.y_of(mp.mpf(xh))
                th_ = sg.p(yh)
                Fv = _F(call, xh)
                g.append(mp.mpc(Fv.real, Fv.imag) / sg.pprime_x(yh))
                eps.append(th_ - t)
                ys.append(yh)
                dl.append(_rem2pi(mp.mpf(call["theta"]["theta_mod_2pi"](xh)) - th_))
            cg = cheb_coeffs(g, N)
            dg = cheb_deriv_at_nodes(cg, N)
            gc = [g[j] - eps[j] * dg[j] / h for j in range(N + 1)]
            M = cheb_moments(h, N)
            W = []
            for j in range(N + 1):
                acc = mp.mpc(0)
                for k in range(N + 1):
                    term = M[k] * C[j][k]
                    if k == 0 or k == N:
                        term = term / 2
                    acc += term
                if j == 0 or j == N:
                    acc = acc / 2
                W.append(2 * acc / N)
            pref = h * mp.expj(c)
            total += pref * mp.fsum(W[j] * gc[j] for j in range(N + 1))
            total_unc += pref * mp.fsum(W[j] * g[j] for j in range(N + 1))
            terms = [pref * W[j] * gc[j] for j in range(N + 1)]
            total_jit += mp.fsum(terms[j] * mp.expj(dl[j]) for j in range(N + 1))
            var_jit += sum(float(tm.imag) ** 2 for tm in terms) * sg.sigma**2
            me = max(float(abs(e)) for e in eps)
            max_eps = max(max_eps, me)
            max_corr = max(
                max_corr,
                max(float(abs(eps[j] * dg[j] / h / g[j])) for j in range(N + 1)),
            )
            for k in range(sg.M + 1):
                Jk[k] += (
                    mp.mpc(0, 1)
                    * pref
                    * mp.fsum(W[j] * gc[j] * mp.chebyt(k, ys[j]) for j in range(N + 1))
                )
        V = sg.design
        cov = sg.sigma_unconstrained**2 * np.linalg.inv(V.T @ V)
        rj = np.array([float(J.real) for J in Jk])
        v = float(rj @ cov @ rj)
        var += v
        d["bound_var"] = v
        diags.append(d)
    return total, {
        "uncorrected": total_unc,
        "jittered": total_jit,
        "var": var,
        "var_jit": var_jit,
        "max_eps_t": max_eps,
        "max_rel_correction": max_corr,
        "segments": diags,
    }


def _route_b_qawo_chunk(state, i0, i1, res):
    """QAWO over segments ``i0 .. i1 - 1``, in ``sigma = t - centre`` on dyadic ``t``-panels, the
    abscissa corrected to first order from a degree-24 interpolant of ``F`` on the segment.
    """
    call, edges = state["call"], state["edges"]
    fit, cfg = ROUTE_B_FIT[res], ROUTE_B_QAWO[res]
    segs = fit_segments(call, edges, i0, i1, fit["M"], fit["K"])
    total = mp.mpc(0)
    stats = _new_qawo_stats()
    stats["max_eps_t"] = 0.0
    diags = []
    for sg in segs:
        diags.append(_segment_diag(sg))
        if not sg.monotone:
            # a stationary point: no t-panels; the segment is integrated directly in y
            total += _direct_segment_qawo(call, sg, ROUTE_B_DIRECT_QAWO[res], stats)
            continue
        nF = 24
        yF = np.cos(np.pi * np.arange(nF + 1) / nF)
        xF = [float(sg.mid + sg.half * mp.mpf(float(y))) for y in yF]
        yF = np.array([float(sg.y_of(mp.mpf(x))) for x in xF])
        FF = np.array([_F(call, x) for x in xF])
        cF = np.polynomial.chebyshev.chebfit(
            yF, FF.real, nF
        ) + 1j * np.polynomial.chebyshev.chebfit(yF, FF.imag, nF)
        cFd = np.polynomial.chebyshev.chebder(cF)
        mid_d = sg.mid_d
        mid_lo = float(sg.mid - mid_d)
        half_d = sg.half_d
        tlo, thi = (sg.T_l, sg.T_r) if sg.T_r > sg.T_l else (sg.T_r, sg.T_l)
        sgn = 1 if sg.T_r > sg.T_l else -1
        span_d = float(sg.T_r - sg.T_l)
        T_l = sg.T_l
        for cen, h in dyadic_tiling(tlo, thi, cfg["Pmax"] / 2):
            qc = sg.q_shift(cen)
            y_c = -1 + 2 * float(cen - T_l) / span_d
            box = {"eps": 0.0}

            def gfun(sig, qc=qc, y_c=y_c, box=box, sg=sg, cFd=cFd):
                y = min(1.0, max(-1.0, y_c + 2 * sig / span_d))
                for _ in range(40):
                    fval = cheb_eval(qc, y) - sig
                    dpy = sg.nonzero_slope(y)
                    yn = min(1.0, max(-1.0, y - fval / dpy))
                    if abs(yn - y) <= 2e-16:
                        y = yn
                        break
                    y = yn
                xh = mid_d + half_d * y
                yh = ((xh - mid_d) - mid_lo) / half_d
                eps = cheb_eval(qc, yh) - sig
                box["eps"] = max(box["eps"], abs(eps))
                pp = cheb_eval(sg.dc_d, yh) / half_d
                ppp = cheb_eval(sg.ddc_d, yh) / half_d**2
                Fv = _F(call, xh)
                Fd = complex(np.polynomial.chebyshev.chebval(yh, cFd)) / half_d
                return Fv / pp - eps * (Fd / pp**2 - Fv * ppp / pp**3)

            g0 = abs(gfun(0.0))
            v, info = qawo_panel(
                gfun,
                mp.mpf(0),
                h,
                1.0,
                0.0,
                1e-17 * g0 * min(2 * h, 2.0),
                cfg["epsrel"],
            )
            total += sgn * mp.expj(cen) * mp.mpc(v.real, v.imag)
            _add_qawo_stats(stats, info)
            stats["max_eps_t"] = max(stats["max_eps_t"], box["eps"])
    return total, {"stats": stats, "segments": diags}


def _route_b_state(call, edges):
    if edges is None:
        edges, n_tables = production_edges(call)
    else:
        edges, n_tables = [float(v) for v in edges], None
    return {"call": call, "edges": edges, "n_tables": n_tables}


# ------------------------------------------------------------------------------------------------
# Running the chunks, serially or in spawned workers


_WORKER = {}


def _worker_init(call_source, route, edges):
    module, function, args = call_source
    call = getattr(importlib.import_module(module), function)(*args)
    _WORKER["state"] = _make_state(call, route, edges)


def _make_state(call, route, edges):
    if route == "A":
        return _route_a_state(call)
    return _route_b_state(call, edges)


def _run_chunk(job):
    kind, args = job
    with mp.workdps(DPS):
        state = _WORKER["state"]
        value, extra = _CHUNK_FUNCTIONS[kind](state, *args)
        return mp.nstr(value.real, 50), mp.nstr(value.imag, 50), _serialise(extra)


_CHUNK_FUNCTIONS = {
    "a-fcc": _route_a_fcc_chunk,
    "a-qawo": _route_a_qawo_chunk,
    "b-fcc": _route_b_fcc_chunk,
    "b-qawo": _route_b_qawo_chunk,
}


def _serialise(extra):
    if extra is None:
        return None
    out = {}
    for key, value in extra.items():
        if isinstance(value, (mp.mpc,)):
            out[key] = (mp.nstr(value.real, 50), mp.nstr(value.imag, 50))
        else:
            out[key] = value
    return out


def _run_jobs(jobs, state, workers, call_source, route, edges):
    if workers <= 1:
        _WORKER["state"] = state
        return [_run_chunk(job) for job in jobs]
    if call_source is None:
        raise ValueError("workers > 1 needs call_source = (module, function, args)")
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        workers, initializer=_worker_init, initargs=(call_source, route, edges)
    ) as pool:
        return pool.map(_run_chunk, jobs, chunksize=1)


def _ordered_total(results):
    total = mp.mpc(0)
    for re_, im_, _ in results:
        total += mp.mpc(mp.mpf(re_), mp.mpf(im_))
    return total


def reading(call, impl, res, route=None, edges=None, workers=1, call_source=None):
    """
    One reading of a captured call's reference: implementation ``impl`` ("fcc" or "qawo") at
    resolution ``res`` (1 or 2), by ``route`` ("A" or "B"; :func:`route_of` when None). Returns a
    dict with the value (``value_hex``, ``value_30``, ``imag_30``, and ``value_mp``), the
    configuration, and the route's own error terms.
    """
    if route is None:
        route = route_of(call)
    start = time.perf_counter()
    with mp.workdps(DPS):
        state = _make_state(call, route, edges)
        if route == "A":
            A = state["route"]
            panels = A.panels(impl, res)
            if impl == "fcc":
                size, kind, extra_args = (
                    ROUTE_A_FCC_CHUNK,
                    "a-fcc",
                    (ROUTE_A_FCC[res]["N"],),
                )
                cfg = ROUTE_A_FCC[res]
            else:
                size, kind, extra_args = (
                    ROUTE_A_QAWO_CHUNK,
                    "a-qawo",
                    (ROUTE_A_QAWO[res]["epsrel"],),
                )
                cfg = ROUTE_A_QAWO[res]
            jobs = [
                (kind, (panels[i : i + size],) + extra_args)
                for i in range(0, len(panels), size)
            ]
            results = _run_jobs(jobs, state, workers, call_source, route, edges)
            value = _ordered_total(results)
            extra = {
                "n_panels": len(panels),
                "u_breaks": A.u_breaks(),
                "K_exact": str(A.K_exact),
                "K_hat": A.K_hat,
                "K_lo": A.K_lo,
                "C_exact": mp.nstr(A.C_mp(), 30),
            }
            if impl == "qawo":
                extra["stats"] = _merge_qawo_stats([r[2]["stats"] for r in results])
            return _reading("A", impl, res, cfg, value, start, **extra)

        edges_used = state["edges"]
        n_segments = len(edges_used) - 1
        bounds = list(range(0, n_segments, ROUTE_B_CHUNK)) + [n_segments]
        kind = "b-fcc" if impl == "fcc" else "b-qawo"
        jobs = [(kind, (bounds[i], bounds[i + 1], res)) for i in range(len(bounds) - 1)]
        results = _run_jobs(jobs, state, workers, call_source, route, edges)
        value = _ordered_total(results)
        segments = [d for r in results for d in r[2]["segments"]]
        tmap = {
            "max_sigma": max(d["sigma"] for d in segments),
            "max_branch_margin": max(d["branch_margin"] for d in segments),
            "all_monotone": all(d["monotone"] for d in segments),
            "direct_segments": sum(1 for d in segments if d["direct"]),
            "max_edge_jump_unconstrained": max(d["edge_jump"] for d in segments),
        }
        extra = {
            "fit": dict(ROUTE_B_FIT[res]),
            "n_segments": n_segments,
            "n_tables": state["n_tables"],
            "tmap": tmap,
            "segments": segments,
        }
        if impl == "fcc":
            cfg = ROUTE_B_FCC[res]
            unc = mp.mpf(0)
            jit = mp.mpf(0)
            for r in results:
                unc += mp.mpf(r[2]["uncorrected"][0])
                jit += mp.mpf(r[2]["jittered"][0])
            tmap["max_eps_t"] = max(r[2]["max_eps_t"] for r in results)
            tmap["max_rel_correction"] = max(
                r[2]["max_rel_correction"] for r in results
            )
            tmap["fit_noise_1sigma"] = math.sqrt(sum(r[2]["var"] for r in results))
            extra["value_uncorrected_30"] = mp.nstr(unc, 30)
            extra["abscissa_correction"] = float(value.real - unc)
            extra["jitter"] = {
                "value_jittered_30": mp.nstr(jit, 30),
                "jittered_minus_smooth": float(jit - value.real),
                "floor_1sigma_at_these_nodes": math.sqrt(
                    sum(r[2]["var_jit"] for r in results)
                ),
            }
        else:
            cfg = ROUTE_B_QAWO[res]
            stats = _merge_qawo_stats([r[2]["stats"] for r in results])
            stats["max_eps_t"] = max(r[2]["stats"]["max_eps_t"] for r in results)
            extra["stats"] = stats
        return _reading("B", impl, res, cfg, value, start, **extra)


def reference(call, tolerance, route=None, edges=None, workers=1, call_source=None):
    """
    The reference for a captured call: four readings (QAWO and FCC, each at resolutions 1 and 2),
    their scatter, and the verdict -- the reference **counts** when the scatter is at most
    ``tolerance``. The value is the FCC reading at resolution 2. Returns a dict with ``route``,
    ``readings`` (each as :func:`reading` returns it, without ``value_mp``), ``scatter``,
    ``tolerance``, ``counts``, ``value_hex`` and ``value_30``.
    """
    if route is None:
        route = route_of(call)
    readings = []
    for impl in IMPLEMENTATIONS:
        for res in RESOLUTIONS:
            r = reading(
                call,
                impl,
                res,
                route=route,
                edges=edges,
                workers=workers,
                call_source=call_source,
            )
            readings.append(r)
    chosen = next(r for r in readings if r["impl"] == "fcc" and r["res"] == 2)
    spread = scatter(readings)
    return {
        "route": route,
        "readings": [{k: v for k, v in r.items() if k != "value_mp"} for r in readings],
        "scatter": spread,
        "tolerance": float(tolerance),
        "counts": spread <= tolerance,
        "value_hex": chosen["value_hex"],
        "value_30": chosen["value_30"],
    }


# ------------------------------------------------------------------------------------------------
# Command line


def _load_pickle(path, unpickler):
    module = importlib.import_module(unpickler)
    return module.loads(Path(path).read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="A Filon reference for a captured adaptive_levin_sincos call."
    )
    parser.add_argument("pickle", help="the captured call (x1_capture.py's format)")
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument(
        "--unpickler",
        default="pickle",
        help="the module whose loads() reads the pickle (e.g. ray.cloudpickle)",
    )
    parser.add_argument("--route", choices=("A", "B"), default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    call = _load_pickle(args.pickle, args.unpickler)
    result = reference(
        call,
        args.tolerance,
        route=args.route,
        workers=args.workers,
        call_source=(
            "tools.filon_reference",
            "_load_pickle",
            (args.pickle, args.unpickler),
        ),
    )
    text = json.dumps(result, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
