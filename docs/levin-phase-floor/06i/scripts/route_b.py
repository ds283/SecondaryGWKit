"""levin-phase-floor prompt 06i Sec 3: Route B, one reading of one cosmological-group sample. New in 06i.

The variable is the phase. On each x segment (between the knots and table switches of
samples.route_b_edges) the map t = p(x) is samples.Segment's least-squares smooth phase,
continuous across segments (each segment refitted with the shared edge values, the average of
the two neighbours' unconstrained fits). Then exactly

    int_seg F(x) exp(i p(x)) dx = int_{p(X_l)}^{p(X_r)} g(t) exp(i t) dt,   g = F(x(t)) / p'(x(t)),

and the reference is Re sum_seg of it: the integral of the captured amplitude against the smooth
phase, i.e. of the captured integrand with theta_mod_2pi's white jitter averaged out (as every
dense quadrature of the captured integrand does in the limit; the jitter's measure is each
segment's residual rms, `sigma`).

x(t) is solved (Newton in double, one step in mpmath) and the callables are evaluated at the double
x^ = fl(x(t)); the sample is g at t^ = p(x^), not at t, and is corrected to first order,
g(t) = g(t^) - (t^ - t) g'(t^): FCC from its own interpolant's derivative, QAWO from a degree-24
interpolant of F in x on the segment.

--impl fcc: res 1: fit (M, K) = (20, 241), one t-panel per segment, N = 32;
            res 2: fit (24, 301), three t-panels per segment, N = 48.
--impl qawo: res 1: fit (20, 241), dyadic t-panels of span <= 2^12 rad, epsrel 1e-13;
             res 2: fit (24, 301), span <= 2^10 rad, epsrel 1.2e-14.
Addition 1 (the user's ruling, 2026-11-01), every fcc reading: the same nodes and weights with
    the captured phase, g_j exp(i delta_j), delta_j = rem(theta_mod_2pi(x^_j) - p(x^_j)): the
    jittered value, its difference from the smooth reference, and the 1-sigma floor
    sqrt(sum_j Im(h e^{ic} W_j g_j)^2 sigma_seg^2) that white jitter of the segment's measured sigma
    puts on Re(I) at these nodes.
--bound (with fcc): also propagate the fit's covariance, sigma^2 (V^T V)^-1 per segment
    (unconstrained design), through J_k = i int_seg F T_k(y) e^{ip} dx: the 1-sigma error of the
    reference due to t(x)'s noise, sqrt(sum_seg Re(J)^T Cov Re(J)).
Usage (census tree as cwd): python -B route_b.py N --impl fcc|qawo --res 1|2 [--bound] [--limit-segments K]
"""

import argparse
import json
import math
import multiprocessing as mpp
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filon  # noqa: E402
import prov  # noqa: E402
import samples as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out" / "readings"
FIT = {1: dict(M=20, K=241), 2: dict(M=24, K=301)}
FCC = {1: dict(pieces=1, N=32), 2: dict(pieces=3, N=48)}
QAWO = {1: dict(Pmax=2.0**12, epsrel=1e-13), 2: dict(Pmax=2.0**10, epsrel=1.2e-14)}

_W = {}


def _init(n):
    filon.set_precision()
    _W["rec"] = S.load(n)
    _W["edges"], _ = S.route_b_edges(_W["rec"])


def fit_segments(rec, edges, i0, i1, M, K):
    n = len(edges) - 1
    lo, hi = max(0, i0 - 1), min(n, i1 + 1)
    un = {j: S.Segment(rec, edges[j], edges[j + 1], M=M, K=K) for j in range(lo, hi)}

    def E(j):
        if j == 0:
            return un[0].T_l
        if j == n:
            return un[n - 1].T_r
        return (un[j - 1].T_r + un[j].T_l) / 2

    out = []
    for j in range(i0, i1):
        sg = S.Segment(
            rec,
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


def _F(rec, x):
    f_sin, f_cos = rec["f"]
    return complex(f_cos(x), -f_sin(x))


def _seg_diag(sg):
    return dict(
        X_l=sg.X_l,
        X_r=sg.X_r,
        sigma=sg.sigma,
        sigma_unconstrained=sg.sigma_unconstrained,
        max_resid=sg.max_resid,
        branch_margin=sg.branch_margin,
        monotone=sg.monotone,
        edge_jump=max(sg.edge_jump),
        t_span=float(sg.T_r - sg.T_l),
    )


def _fcc_job(args):
    i0, i1, res, bound = args
    rec, edges = _W["rec"], _W["edges"]
    fit, cfg = FIT[res], FCC[res]
    N = cfg["N"]
    s_nodes, C = filon.cc_table(N)
    segs = fit_segments(rec, edges, i0, i1, fit["M"], fit["K"])
    total = mp.mpc(0)
    total_unc = mp.mpc(0)
    total_jit = mp.mpc(0)
    var = 0.0
    var_jit = 0.0
    max_eps = 0.0
    max_corr = 0.0
    diags = []
    for sg in segs:
        d = _seg_diag(sg)
        d["max_eps_t"] = 0.0
        if not sg.monotone:
            d["NOT_MONOTONE"] = True
        Jk = [mp.mpc(0)] * (sg.M + 1) if bound else None
        P = cfg["pieces"]
        for q in range(P):
            ta = sg.T_l + (sg.T_r - sg.T_l) * q / P
            tb = sg.T_l + (sg.T_r - sg.T_l) * (q + 1) / P
            c, h = (ta + tb) / 2, (tb - ta) / 2
            g, eps, ys, dl = [], [], [], []
            for sj in s_nodes:
                t = c + h * sj
                y = sg.solve_y(t, shift=c)
                xh = float(sg.mid + sg.half * y)
                yh = sg.y_of(mp.mpf(xh))
                th_ = sg.p(yh)
                Fv = _F(rec, xh)
                g.append(mp.mpc(Fv.real, Fv.imag) / sg.pprime_x(yh))
                eps.append(th_ - t)
                ys.append(yh)
                # addition 1: the captured phase's departure from the smooth one at this node
                dl.append(S.rem2pi(mp.mpf(rec["theta"]["theta_mod_2pi"](xh)) - th_))
            cg = filon.cheb_coeffs(g, N)
            dg = filon.cheb_deriv_at_nodes(cg, N)  # d/ds; d/dt = (d/ds)/h
            gc = [g[j] - eps[j] * dg[j] / h for j in range(N + 1)]
            M = filon.cheb_moments(h, N)
            # FCC weights: I_panel = h e^{ic} sum_j W_j g_j
            W = []
            for j in range(N + 1):
                acc = mp.mpc(0)
                for k in range(N + 1):
                    t_ = M[k] * C[j][k]
                    if k == 0 or k == N:
                        t_ = t_ / 2
                    acc += t_
                if j == 0 or j == N:
                    acc = acc / 2
                W.append(2 * acc / N)
            pref = h * mp.expj(c)
            val = pref * mp.fsum(W[j] * gc[j] for j in range(N + 1))
            val_unc = pref * mp.fsum(W[j] * g[j] for j in range(N + 1))
            total += val
            total_unc += val_unc
            # addition 1: the same nodes and weights with the captured (jittered) phase, and the
            # variance that white jitter of the segment's sigma gives Re(I) at these nodes
            terms = [pref * W[j] * gc[j] for j in range(N + 1)]
            total_jit += mp.fsum(terms[j] * mp.expj(dl[j]) for j in range(N + 1))
            var_jit += sum(float(tm.imag) ** 2 for tm in terms) * sg.sigma**2
            me = max(float(abs(e)) for e in eps)
            d["max_eps_t"] = max(d["max_eps_t"], me)
            max_eps = max(max_eps, me)
            max_corr = max(
                max_corr,
                max(float(abs(eps[j] * dg[j] / h / g[j])) for j in range(N + 1)),
            )
            if bound:
                for k in range(sg.M + 1):
                    Jk[k] += (
                        mp.mpc(0, 1)
                        * pref
                        * mp.fsum(
                            W[j] * gc[j] * mp.chebyt(k, ys[j]) for j in range(N + 1)
                        )
                    )
        if bound:
            V = sg.design
            cov = sg.sigma_unconstrained**2 * np.linalg.inv(V.T @ V)
            rj = np.array([float(J.real) for J in Jk])
            v = float(rj @ cov @ rj)
            var += v
            d["bound_var"] = v
        diags.append(d)
    return (
        mp.nstr(total.real, 45),
        mp.nstr(total.imag, 45),
        mp.nstr(total_unc.real, 45),
        var,
        max_eps,
        max_corr,
        mp.nstr(total_jit.real, 45),
        var_jit,
        diags,
    )


def _qawo_job(args):
    i0, i1, res = args
    rec, edges = _W["rec"], _W["edges"]
    fit, cfg = FIT[res], QAWO[res]
    segs = fit_segments(rec, edges, i0, i1, fit["M"], fit["K"])
    total = mp.mpc(0)
    stats = {
        "panels": 0,
        "neval": 0,
        "ier_nonzero": 0,
        "max_last": 0,
        "sum_quad_abserr": 0.0,
        "max_eps_t": 0.0,
    }
    diags = []
    for sg in segs:
        diags.append(_seg_diag(sg))
        # F's interpolant in x on the segment (degree 24, Lobatto points), for the correction only
        nF = 24
        yF = np.cos(np.pi * np.arange(nF + 1) / nF)
        xF = [float(sg.mid + sg.half * mp.mpf(float(y))) for y in yF]
        yF = np.array([float(sg.y_of(mp.mpf(x))) for x in xF])
        FF = np.array([_F(rec, x) for x in xF])
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
        for cen, h in filon.dyadic_tiling(tlo, thi, cfg["Pmax"] / 2):
            qc = sg.q_shift(cen)
            y_c = -1 + 2 * float(cen - T_l) / span_d
            box = {"eps": 0.0}

            def gfun(sig, qc=qc, y_c=y_c, box=box):
                # Newton on p(y) - cen = sig, in double
                y = min(1.0, max(-1.0, y_c + 2 * sig / span_d))
                for _ in range(40):
                    f = S.cheb_eval(qc, y) - sig
                    dpy = S.cheb_eval(sg.dc_d, y)
                    yn = min(1.0, max(-1.0, y - f / dpy))
                    if abs(yn - y) <= 2e-16:
                        y = yn
                        break
                    y = yn
                xh = mid_d + half_d * y
                yh = ((xh - mid_d) - mid_lo) / half_d
                eps = S.cheb_eval(qc, yh) - sig
                box["eps"] = max(box["eps"], abs(eps))
                pp = S.cheb_eval(sg.dc_d, yh) / half_d
                ppp = S.cheb_eval(sg.ddc_d, yh) / half_d**2
                Fv = _F(rec, xh)
                Fd = complex(np.polynomial.chebyshev.chebval(yh, cFd)) / half_d
                return Fv / pp - eps * (Fd / pp**2 - Fv * ppp / pp**3)

            g0 = abs(gfun(0.0))
            v, info = filon.qawo_panel(
                gfun,
                mp.mpf(0),
                h,
                1.0,
                0.0,
                1e-17 * g0 * min(2 * h, 2.0),
                cfg["epsrel"],
            )
            # the panel's variable is sigma = t - cen, sampled at fl(0) + s = s exactly
            total += sgn * mp.expj(cen) * mp.mpc(v.real, v.imag)
            stats["panels"] += 1
            stats["neval"] += info["neval"]
            stats["ier_nonzero"] += sum(1 for x in info["ier"] if x)
            stats["max_last"] = max(stats["max_last"], info["last"])
            stats["sum_quad_abserr"] += info["abserr"]
            stats["max_eps_t"] = max(stats["max_eps_t"], box["eps"])
    return mp.nstr(total.real, 45), mp.nstr(total.imag, 45), stats, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--impl", choices=("fcc", "qawo"), required=True)
    ap.add_argument("--res", type=int, choices=(1, 2), required=True)
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--limit-segments", type=int, default=0, help="testing only: first K segments"
    )
    a = ap.parse_args()
    assert a.workers <= 6 and a.n in S.ROUTE_B
    assert not (a.bound and a.impl != "fcc")
    filon.set_precision()
    t0 = time.time()
    rec = S.load(a.n)
    edges, ntab = S.route_b_edges(rec)
    nseg = (
        len(edges) - 1
        if not a.limit_segments
        else min(a.limit_segments, len(edges) - 1)
    )
    nch = min(nseg, a.workers * 6)
    bounds = [round(nseg * i / nch) for i in range(nch + 1)]
    if a.impl == "fcc":
        jobs = [
            (bounds[i], bounds[i + 1], a.res, a.bound)
            for i in range(nch)
            if bounds[i + 1] > bounds[i]
        ]
        fn = _fcc_job
    else:
        jobs = [
            (bounds[i], bounds[i + 1], a.res)
            for i in range(nch)
            if bounds[i + 1] > bounds[i]
        ]
        fn = _qawo_job
    with mpp.get_context("spawn").Pool(
        a.workers, initializer=_init, initargs=(a.n,)
    ) as pool:
        res = pool.map(fn, jobs, chunksize=1)
    re_ = mp.fsum(mp.mpf(r[0]) for r in res)
    im_ = mp.fsum(mp.mpf(r[1]) for r in res)
    diags = [d for r in res for d in r[-1]]
    out = {
        "header": prov.header([S.pickle_path(a.n), S.Z3]),
        "sample": a.n,
        "route": "B",
        "impl": a.impl,
        "res": a.res,
        "fit": FIT[a.res],
        "config": FCC[a.res] if a.impl == "fcc" else QAWO[a.res],
        "workers": a.workers,
        "limit_segments": a.limit_segments,
        "n_segments": nseg,
        "n_tables": ntab,
        "value_hex": float(re_).hex(),
        "value_30": mp.nstr(re_, 30),
        "imag_30": mp.nstr(im_, 30),
        "tmap": {
            "max_sigma": max(d["sigma"] for d in diags),
            "max_branch_margin": max(d["branch_margin"] for d in diags),
            "all_monotone": all(d["monotone"] for d in diags),
            "max_edge_jump_unconstrained": max(d["edge_jump"] for d in diags),
        },
    }
    if a.impl == "fcc":
        out["value_uncorrected_30"] = mp.nstr(mp.fsum(mp.mpf(r[2]) for r in res), 30)
        out["abscissa_correction"] = float(re_ - mp.fsum(mp.mpf(r[2]) for r in res))
        out["tmap"]["max_eps_t"] = max(r[4] for r in res)
        out["tmap"]["max_rel_correction"] = max(r[5] for r in res)
        if a.bound:
            out["tmap"]["fit_noise_1sigma"] = math.sqrt(sum(r[3] for r in res))
        jit = mp.fsum(mp.mpf(r[6]) for r in res)
        out["jitter"] = {
            "value_jittered_30": mp.nstr(jit, 30),
            "jittered_minus_smooth": float(jit - re_),
            "floor_1sigma_at_these_nodes": math.sqrt(sum(r[7] for r in res)),
        }
    else:
        st = {}
        for key in ("panels", "neval", "ier_nonzero", "sum_quad_abserr"):
            st[key] = sum(r[2][key] for r in res)
        st["max_last"] = max(r[2]["max_last"] for r in res)
        st["max_eps_t"] = max(r[2]["max_eps_t"] for r in res)
        out["stats"] = st
    out["wall_s"] = time.time() - t0
    z = S.z3_row(a.n)
    out["z3_unit"], out["z3_index"] = z["unit"], z["index"]
    out["segments"] = diags
    tag = (
        f"s{a.n:02d}-{a.impl}-r{a.res}"
        + ("-bound" if a.bound else "")
        + (f"-TEST{a.limit_segments}" if a.limit_segments else "")
    )
    if (
        not a.limit_segments
    ):  # a test of the mechanics is not a reading, and is not kept
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1))
    print(
        tag,
        out["value_30"],
        out["value_hex"],
        f"wall {out['wall_s']:.1f} s",
        nseg,
        "segments",
        out["tmap"],
        out.get("stats"),
        out.get("abscissa_correction"),
        out.get("jitter"),
    )


if __name__ == "__main__":
    main()
