"""levin-phase-floor 06, finding P1, step 3(d): an independent reference for the G+q+r group on the
all-oscillatory sub-interval, under one tree, after the validated rebuild (p1lib).

The integrand is the group's own  g(x) = f_sin(x) sin psi(x) + f_cos(x) cos psi(x),  x = log(1+z),
psi = group.theta_mod_2pi (the composed node-anchored remainder), f_sin/f_cos the group's
amplitudes: exactly the integrand adaptive_levin_sincos is handed, evaluated pointwise.

Split of the sub-interval [x0, x1]:
  * the **window** W(Phi) = {x : |theta(x) - theta(x_s)| <= Phi} around the stationary point x_s
    (the one sign change of theta', found by brentq on group.theta_deriv), clipped to [x0, x1].
    Its edges are found by brentq on theta(x) - theta(x_s) -+ Phi (theta is monotone on each
    side of x_s). **Gauss-Legendre, no Levin**: the window is cut into pieces each spanning at
    most about DPHI rad of phase (edges marched outwards from x_s with width
    DPHI / max(|theta'|, sqrt(theta'' DPHI)), every piece's actual span |theta(b) - theta(a)|
    measured and reported), and each piece is integrated by an n-point Gauss-Legendre rule;
    pieces summed with math.fsum. Two resolutions (DPHI, n) = (1.0, 12) and (0.5, 20); their
    difference is the GL error estimate.
  * the two **outer** ranges [x0, W_lo] and [W_hi, x1], where |theta'| >= sqrt(2 theta'' Phi)
    and the phase is fast: adaptive_levin_sincos exactly as QSI calls it (the captured f, theta
    dict, chebyshev_order; atol scaled by width share), forced onto 2^m equal pieces, m = 6 and 9
    at the stored rtol, and m = 9 at rtol 1e-10; their spread is the Levin error estimate.
  * the reference is GL(window) + Levin(outer), for Phi in PHIS; its dependence on Phi measures
    the hand-off between the two methods directly (a wider window replaces Levin by GL on the
    annulus between).
Also reported, for comparison only: the same Levin forced subdivision (m = 9) applied *over the
window*, against GL there.

Values are unscaled (no (1 + z_response)).

Usage: PYTHONPATH=<tree> OMP_NUM_THREADS=1 python -B reference.py r13|r14 [row1 row2]
Writes out/reference_<tree>.json.
"""

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from prov import OUT, header, sha256  # noqa: E402
import p1lib  # noqa: E402
from p1lib import fx  # noqa: E402

PHIS = (1.0e3, 1.0e4, 3.0e4)
GL_RES = ((1.0, 12), (0.5, 20))
GROUP = "G+q+r"


def stationary_point(group, x0, x1, n=40001):
    thd = group.theta_deriv
    xs = np.linspace(x0, x1, n)
    d = np.array([thd(float(x)) for x in xs])
    idx = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    if len(idx) != 1:
        raise RuntimeError(f"p1 reference: expected one sign change of theta', found {len(idx)}")
    i = int(idx[0])
    r = brentq(thd, float(xs[i]), float(xs[i + 1]), xtol=1e-15, rtol=1e-15, maxiter=200)
    h = 1e-5
    return r, (thd(r + h) - thd(r - h)) / (2 * h)


def window(group, xs, x0, x1, Phi):
    th = group.theta
    t_s = th(xs)

    a = x0 if abs(th(x0) - t_s) <= Phi else brentq(lambda x: abs(th(x) - t_s) - Phi, x0, xs, xtol=1e-15, rtol=1e-15)
    b = x1 if abs(th(x1) - t_s) <= Phi else brentq(lambda x: abs(th(x) - t_s) - Phi, xs, x1, xtol=1e-15, rtol=1e-15)
    return a, b


def gl_edges(group, xs, d2, a, b, dphi):
    """Piece edges marched outward from xs, each piece about dphi rad of phase or less."""
    thd = group.theta_deriv
    floor_rate = math.sqrt(abs(d2) * dphi)
    right = [xs]
    x = xs
    while x < b:
        x = x + dphi / max(abs(thd(x)), floor_rate)
        right.append(min(x, b))
    left = [xs]
    x = xs
    while x > a:
        x = x - dphi / max(abs(thd(x)), floor_rate)
        left.append(max(x, a))
    edges = left[::-1] + right[1:]
    return np.array(edges)


def gl_integrate(group, edges, n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    f_s, f_c, psi = group.f_sin, group.f_cos, group.theta_mod_2pi
    parts = []
    for i in range(len(edges) - 1):
        a, b = float(edges[i]), float(edges[i + 1])
        hw = 0.5 * (b - a)
        mid = 0.5 * (a + b)
        acc = []
        for t, w in zip(nodes, weights):
            x = mid + hw * float(t)
            p = psi(x)
            acc.append(float(w) * (f_s(x) * math.sin(p) + f_c(x) * math.cos(p)))
        parts.append(hw * math.fsum(acc))
    return math.fsum(parts), len(parts)


def levin_forced(levin, call, a, b, m, rtol):
    if b <= a:
        return 0.0, 0.0, 0, True
    f = call["args"][1]
    theta = call["kwargs"]["theta"]
    atol = call["kwargs"]["atol"]
    order = call["kwargs"]["chebyshev_order"]
    x_span = call["args"][0]
    W = x_span[1] - x_span[0]
    n = 2**m
    edges = np.linspace(a, b, n + 1)
    edges[0], edges[-1] = a, b
    vals, errs, nreg, conv = [], [], 0, True
    for i in range(n):
        lo, hi = float(edges[i]), float(edges[i + 1])
        res = levin((lo, hi), f, theta=theta, atol=atol * (hi - lo) / W, rtol=rtol, chebyshev_order=order)
        vals.append(res["value"])
        errs.append(res["abserr"])
        nreg += int(res["num_regions"])
        conv = conv and bool(res["converged"])
    return math.fsum(vals), math.fsum(errs), nreg, conv


def main():
    tree = sys.argv[1]
    row_names = tuple(sys.argv[2:]) or ("row1", "row2")
    log = lambda s: print(s, flush=True)  # noqa: E731
    out = {"header": header(Path(__file__), f"p1 reference, tree {tree}"), "PHIS": PHIS, "GL_RES": GL_RES}
    with p1lib.rebuilt_rows(tree, row_names=row_names, log=log) as (ev, ctx, rows):
        from harness import QSI_MOD

        levin = QSI_MOD.adaptive_levin_sincos
        out["evidence"] = ev
        out["rows"] = {}
        for name, row in rows.items():
            if not row["validation"]["all_identical"]:
                raise SystemExit(f"p1: {tree} {name} does not reproduce its store; stopping")
            gc, groups, calls = p1lib.low_subinterval_groups(row)
            group, call = groups[GROUP], calls[GROUP]
            x0, x1 = call["args"][0]
            rtol0 = call["kwargs"]["rtol"]
            xs, d2 = stationary_point(group, x0, x1)
            log(f"== {tree} {name} {GROUP}: x_s {xs!r} (z {math.expm1(xs):.10g}), theta'' {d2:.8g}")
            # timing probe
            t0 = time.perf_counter()
            for j in range(200):
                x = xs + (j - 100) * 1e-4
                group.f_sin(x), group.f_cos(x), group.theta_mod_2pi(x)
            per_eval = (time.perf_counter() - t0) / 200
            log(f"   {per_eval * 1e6:.1f} us per integrand evaluation")
            rec = {"x_s": fx(xs), "z_s": fx(math.expm1(xs)), "theta_pp": fx(d2), "per_eval_s": per_eval,
                   "captured_value": fx(call["result"]["value"]), "windows": {}}
            for Phi in PHIS:
                a, b = window(group, xs, x0, x1, Phi)
                w = {"Phi": Phi, "window": [fx(a), fx(b)], "z_window": [fx(math.expm1(a)), fx(math.expm1(b))],
                     "rate_at_window_edges": [fx(group.theta_deriv(a)), fx(group.theta_deriv(b))],
                     "gl": {}, "levin_outer": {}}
                for dphi, n in GL_RES:
                    t0 = time.perf_counter()
                    edges = gl_edges(group, xs, d2, a, b, dphi)
                    spans = np.abs(np.diff([group.theta(float(e)) for e in edges]))
                    val, npieces = gl_integrate(group, edges, n)
                    w["gl"][f"dphi{dphi:g}_n{n}"] = {
                        "value": fx(val),
                        "pieces": npieces,
                        "max_piece_phase_span": float(spans.max()),
                        "wall_s": time.perf_counter() - t0,
                    }
                    log(f"   Phi {Phi:g} [{a:.9f}, {b:.9f}] GL dphi {dphi:g} n {n}: {val!r} ({npieces} pieces, "
                        f"max span {spans.max():.3g} rad, {time.perf_counter() - t0:.1f} s)")
                for m, rt in ((6, rtol0), (9, rtol0), (9, 1e-10)):
                    t0 = time.perf_counter()
                    lv, le, ln, lc = levin_forced(levin, call, x0, a, m, rt)
                    rv, re_, rn, rc = levin_forced(levin, call, b, x1, m, rt)
                    w["levin_outer"][f"m{m}_rtol{rt:g}"] = {
                        "left": fx(lv), "right": fx(rv), "sum": fx(math.fsum([lv, rv])),
                        "abserr": fx(le + re_), "regions": ln + rn, "converged": lc and rc,
                        "wall_s": time.perf_counter() - t0,
                    }
                    log(f"   Phi {Phi:g} Levin outer m {m} rtol {rt:g}: left {lv!r} right {rv!r} "
                        f"(abserr {le + re_:.3g}, {time.perf_counter() - t0:.1f} s)")
                wv, we, wn, wc = levin_forced(levin, call, a, b, 9, rtol0)
                w["levin_over_window_m9"] = {"value": fx(wv), "abserr": fx(we), "regions": wn, "converged": wc}
                gl_f = float.fromhex(w["gl"]["dphi0.5_n20"]["value"]["hex"])
                gl_c = float.fromhex(w["gl"]["dphi1_n12"]["value"]["hex"])
                lo_ref = float.fromhex(w["levin_outer"][f"m9_rtol{rtol0:g}"]["sum"]["hex"])
                lo_6 = float.fromhex(w["levin_outer"][f"m6_rtol{rtol0:g}"]["sum"]["hex"])
                lo_t = float.fromhex(w["levin_outer"]["m9_rtol1e-10"]["sum"]["hex"])
                ref = math.fsum([gl_f, lo_ref])
                w["reference"] = fx(ref)
                w["gl_error_estimate"] = fx(abs(gl_f - gl_c))
                w["levin_outer_error_estimate"] = fx(max(abs(lo_ref - lo_6), abs(lo_ref - lo_t)))
                log(f"   Phi {Phi:g}: REFERENCE {ref!r} (GL err {abs(gl_f - gl_c):.3g}, Levin outer err "
                    f"{max(abs(lo_ref - lo_6), abs(lo_ref - lo_t)):.3g}); Levin m9 over window {wv!r} vs GL {gl_f!r}")
                rec["windows"][f"{Phi:g}"] = w
            out["rows"][name] = rec
    path = OUT / f"reference_{tree}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {path} sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
