"""levin-phase-floor 07, an orchestrator investigation at the section 2 stop: where the window code's
error floor on (R) c = 1e8, x_s = 0.7236..., omega = 0 comes from.

Usage: leaf_errors.py sweep            -- the call at rtol 1e-7 ... 1e-10, parent and Phi_w 2, 8, 32 pi
       leaf_errors.py leaves MODE RTOL [PHI_PI]
                                       -- every leaf's true error, by class and by midpoint rounding

MODE is "parent" (bd21bf23's AdaptiveLevin/levin_quadrature.py, read with git show and loaded as a
module of its own) or "new" (the working tree's module, with _LEVIN_WINDOW_HALF_PHASE = PHI_PI * pi
and _LEVIN_COALESCENCE_LAMBDA = 4, which this family never reaches: (R) has no minimum).

The integrand is calibrate.py's: f_sin = exp(-x), f_cos = cos 3x on [0, 1], theta = (c/2)(x - x_s)^2,
theta' supplied, atol 1e-14 |ref|. A leaf's exact integral is the closed form in erf
(reference.closed_R's, over [a, b] in place of [0, 1]) at 60 digits; the leaves' errors sum to the
call's.

The midpoint test. chebyshev_matrices maps the nodes as fl((a + b)/2) + fl((b - a)/2 * x_base) and
pins the two end nodes (06e). When fl(a + b) rounds, every interior node moves by the same
delta = fl((a + b)/2) - (a + b)/2, and the rule integrates a shifted copy of the leaf: an error of
about delta * (g(b) - g(a)), g the integrand. "coh" is that prediction, summed over the leaf's own
solve (value_kind "own") or over its four quarters (value_kind "quarters").
"""

import hashlib
import math
import os
import platform
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from fractions import Fraction as F

import mpmath as mp
import numpy as np

REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"
PARENT_SHA = "bd21bf23da006aa21c5aa670cf32ea2ed726cb6b"
MODULE = os.path.join(REPO, "AdaptiveLevin/levin_quadrature.py")
C_VAL, XS = 1e8, 0.7236067977499790
ATOL_FACTOR = 1.0e-14


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def header(argv):
    head = (
        subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"])
        .decode()
        .strip()
    )
    status = subprocess.check_output(["git", "-C", REPO, "status", "--short"]).decode()
    uptime = subprocess.check_output(["uptime"]).decode().strip()
    print("# levin-phase-floor 07, orchestrator investigation: the CC midpoint shift")
    print(f"# written {datetime.now(timezone.utc).isoformat()} on {platform.node()}")
    print(f"# python {platform.python_version()} ({sys.executable})")
    print(f"# argv {' '.join(argv[1:])}")
    print(f"# script sha256 {sha256(os.path.abspath(__file__))}")
    print(f"# AdaptiveLevin/levin_quadrature.py (working tree) sha256 {sha256(MODULE)}")
    print(f"# parent {PARENT_SHA}")
    print(f"# HEAD {head}")
    print(
        "# tree dirty: " + ("yes" if status.strip() else "no") + "; git status --short:"
    )
    for line in status.splitlines():
        print(f"#     {line}")
    print(f"# uptime at start: {uptime}")
    print(f"# mpmath {mp.__version__}, numpy {np.__version__}")
    sys.stdout.flush()


def load(mode, phi_pi):
    sys.path.insert(0, REPO)
    if mode == "parent":
        src = subprocess.check_output(
            [
                "git",
                "-C",
                REPO,
                "show",
                f"{PARENT_SHA}:AdaptiveLevin/levin_quadrature.py",
            ]
        ).decode()
        m = types.ModuleType("lq_parent")
        m.__file__ = "lq_parent.py"
        sys.modules["lq_parent"] = m
        exec(compile(src, "lq_parent.py", "exec"), m.__dict__)
        return m
    import AdaptiveLevin.levin_quadrature as lq

    lq._LEVIN_WINDOW_HALF_PHASE = phi_pi * np.pi
    lq._LEVIN_COALESCENCE_LAMBDA = 4.0
    return lq


mp.mp.dps = 60
_C = mp.mpf(C_VAL)
_XS = mp.mpf(XS)
_SB = mp.sqrt(mp.mpc(0, -1) * _C / 2)


def exact(a, b):
    u0, u1 = mp.mpf(a) - _XS, mp.mpf(b) - _XS

    def J(g):
        shift = mp.mpc(0, 1) * g / _C
        pref = (
            mp.exp(g * _XS)
            * mp.exp(mp.mpc(0, 1) * g**2 / (2 * _C))
            * mp.sqrt(mp.pi)
            / (2 * _SB)
        )
        return pref * (mp.erf(_SB * (u1 - shift)) - mp.erf(_SB * (u0 - shift)))

    return mp.im(J(mp.mpf(-1))) + mp.re(J(mp.mpc(0, 3)) + J(mp.mpc(0, -3))) / 2


REF = exact(0.0, 1.0)


def g(x):
    x = mp.mpf(x)
    th = _C / 2 * (x - _XS) ** 2
    return mp.exp(-x) * mp.sin(th) + mp.cos(3 * x) * mp.cos(th)


def run(lq, rtol):
    atol = ATOL_FACTOR * abs(float(REF))
    t0 = time.perf_counter()
    res = lq.adaptive_levin_sincos(
        (0.0, 1.0),
        [lambda x: np.exp(-x), lambda x: np.cos(3 * x)],
        {
            "theta": lambda x: 0.5 * C_VAL * (x - XS) ** 2,
            "theta_deriv": lambda x: C_VAL * (x - XS),
        },
        atol=atol,
        rtol=rtol,
        depth_max=20,
    )
    return res, atol, time.perf_counter() - t0


def one_sweep_line(mode, rtol, phi_pi):
    lq = load(mode, phi_pi)
    res, atol, wall = run(lq, rtol)
    v, a = res["value"], res["abserr"]
    req = max(atol, rtol * abs(v))
    t = float(abs(mp.mpf(v) - REF))
    md = max(r._depth for r in res["regions"])
    print(
        f"{mode:6s} {str(phi_pi):5s} rtol={rtol:.0e} req={req:.3g} abserr={a:.3g} true={t:.3g} "
        f"honest={t <= a} conv={res.get('converged')} nl={res.get('node_limited')} "
        f"leaves={len(res['regions'])} maxdepth={md} wall={wall:.1f}s",
        flush=True,
    )


def classify(lq, r, w):
    if r._type == lq.INTERVAL_TYPE_LEVIN:
        kind = "LEV"
    elif getattr(r, "_window", False):
        kind = "WIN"
    else:
        kind = "CC "
    d = (
        0.0
        if r._start <= XS <= r._end
        else min(abs(r._start - XS), abs(r._end - XS)) / w
    )
    zone = "<=1w" if d < 1 else ("1-2w" if d < 2 else ("2-6w" if d < 6 else ">6w"))
    return kind, zone


def pieces_of(r):
    a, b = r._start, r._end
    if r._value_kind == "own":
        return [(a, b)]
    m = (a + b) / 2.0
    q1, q3 = (a + m) / 2.0, (m + b) / 2.0
    return [(a, q1), (q1, m), (m, q3), (q3, b)]


def leaves(mode, rtol, phi_pi):
    lq = load(mode, phi_pi)
    res, atol, wall = run(lq, rtol)
    regs = sorted(res["regions"], key=lambda r: r._start)
    vals = [r._value for r in regs]
    true = float(mp.mpf(res["value"]) - REF)
    sumE = sum((r._node_error_signed or 0.0) for r in regs)
    print(
        f"== {mode} {phi_pi} rtol={rtol:.0e}: value={res['value']!r} true err={true:+.3e} "
        f"sum E={sumE:+.3e} leaves={len(regs)} converged={res.get('converged')} "
        f"node_limited={res.get('node_limited')} wall={wall:.1f}s"
    )
    print(f"   fsum(leaf values) - value = {math.fsum(vals) - res['value']:+.1e}")
    w = (4 * np.pi / C_VAL) ** 0.5
    by_zone, by_mid = {}, {}
    errs = []
    for r in regs:
        e = float(mp.mpf(r._value) - exact(r._start, r._end))
        errs.append(e)
        kind, zone = classify(lq, r, w)
        E = r._node_error_signed
        s = by_zone.setdefault((kind, zone), [0, 0.0, 0.0, 0.0])
        s[0] += 1
        s[1] += e
        s[2] += E or 0.0
        s[3] += abs(e)
        if r._type == lq.INTERVAL_TYPE_LEVIN:
            continue
        ps = pieces_of(r)
        coh = 0.0
        dyadic = True
        for p, q in ps:
            dm = F((p + q) / 2.0) - (F(p) + F(q)) / 2
            dyadic = dyadic and dm == 0
            coh += float(dm) * float(g(q) - g(p))
        k = (r._value_kind, "dyadic-mid" if dyadic else "rounded-mid")
        s = by_mid.setdefault(k, [0, 0.0, 0.0, 0.0, 0.0])
        s[0] += 1
        s[1] += e
        s[2] += coh
        s[3] += abs(e)
        s[4] += abs(e - coh)
    print(f"   sum of leaf errors = {sum(errs):+.3e}")
    print(
        "   by class and distance from the root (w = sqrt(4 pi / c), the 2 pi window's half-width):"
    )
    print("   kind zone      n   sum err     sum E       sum|err|")
    for (k, z), (n, se, sE, sa) in sorted(by_zone.items()):
        print(f"   {k}  {z:5s} {n:5d}  {se:+.3e}  {sE:+.3e}  {sa:.3e}")
    print(
        "   Clenshaw-Curtis leaves by midpoint rounding (coh = delta_mid * (g(b) - g(a)), summed):"
    )
    print(
        "   value    mids          n   sum err     sum coh     sum|err|   sum|err-coh|"
    )
    for (vk, mk), (n, se, sc, sa, sr) in sorted(by_mid.items()):
        print(f"   {vk:8s} {mk:11s} {n:4d}  {se:+.3e}  {sc:+.3e}  {sa:.3e}  {sr:.3e}")
    sys.stdout.flush()


def main(argv):
    header(argv)
    if argv[1] == "sweep":
        for rtol in (1e-7, 1e-8, 1e-9, 1e-10):
            one_sweep_line("parent", rtol, None)
            for phi in (2.0, 8.0, 32.0):
                one_sweep_line("new", rtol, phi)
    elif argv[1] == "leaves":
        leaves(argv[2], float(argv[3]), float(argv[4]) if len(argv) > 4 else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
