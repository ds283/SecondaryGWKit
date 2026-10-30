"""Question 5, the LiouvilleGreen call: seed 0's test_JJJ, Levin call 15 (as 07's stop4_lg_seed0.py
captures it: random.seed(0), that test alone, the module's _adaptive_levin wrapped and counted).

At call 15, 07's driver is run with lc's recording hooks at forced depth_max 6, 8, 10 and 12 with the
call's own arguments otherwise. For each run: the call's value, abserr, request; the located points;
the window's and the outside's summed declared error by component; every leaf outside the window
(declared quantities: type, depth, abserr, floor, truncation, trust check, component, and its
position from the root in phase, |theta(x) - theta(x_r)| from the basis's raw theta, and theta').

For the window leaves at depth_max 6 and 8: the floor recomputed term by term -- C1, C2, C3 of
_cc_roundoff_floor and the declared-theta_abserr endpoint term D -- checked against phase_err.

A reference for sampled leaves, both window and outside: Gauss-Legendre on the leaf at n = 32, 64,
128 and 256 points, the integrand evaluated by the basis's own eval_basis (theta_mod_2pi) and the
amplitudes, summed by math.fsum; the leaf's value against each, so the reference's convergence is
shown beside the difference.

Writes out/lg_seed0_call15.json.
"""

import json
import math
import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import lc  # noqa: E402

lq = lc.lq
STUCK_CALL = 15


class _Abandon(Exception):
    pass


def lg_terms(basis, cache, f, span, order):
    N = order
    a, b = span
    grid, Dmat = lq.chebyshev_matrices(span, N)
    _, thp, _, _, _ = basis.build_Levin_data(grid, Dmat, need_AmatT=False, vectorize_cache=cache)
    fine = 2 * N - 1
    fine_grid, D_fine = lq.chebyshev_matrices(span, fine)
    f_fine = np.vstack(
        [lq._sample_vectorized(fn, fine_grid, cache, ("f", id(fn))) for fn in f]
    )
    ba = [basis.eval_basis_and_arguments(x) for x in fine_grid]
    bas = np.array([v for v, _ in ba]).T
    args = np.array([g for _, g in ba], dtype=float).T
    w = lq._cc_weights(span, fine)
    aw = np.fabs(w)
    n = aw.shape[0]
    S = float(np.dot(aw, np.sum(np.fabs(f_fine * bas), axis=0)))
    C1 = (n + lq._CC_ROUNDOFF_PER_TERM_COUNT) * lq._UNIT_ROUNDOFF * S
    C2 = lq._UNIT_ROUNDOFF * float(np.dot(aw, np.sum(np.fabs(f_fine) * np.fabs(args), axis=0)))
    fp = np.vstack([np.matmul(D_fine, row) for row in f_fine])
    delta = lq._cc_node_error_bound(span)
    tpb = float(np.max(np.fabs(thp)))
    C3 = delta * float(np.dot(aw, np.sum(tpb * np.fabs(f_fine) + np.fabs(fp), axis=0)))
    width = abs(b - a)
    f_scale = float(np.max(np.sqrt(np.sum(f_fine * f_fine, axis=0))))
    proxy = 0.5 * f_scale * width
    D = lq._declared_endpoint_phase_err(basis, a, b, proxy, proxy)
    return {"C1": C1, "C2": C2, "C3": C3, "D": D, "theta_prime_bound": tpb}


def gl_reference(basis, f, span, n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    a, b = span
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    terms = []
    for t, w in zip(xg, wg):
        x = mid + half * t
        sc = basis.eval_basis(x)
        val = sum(float(fn(x)) * float(s) for fn, s in zip(f, sc))
        terms.append(w * half * val)
    return math.fsum(terms)


def analyse(x_span, f, basis, args, kwargs, out):
    rec = {"x_span": [float(x_span[0]), float(x_span[1])]}
    rec["kwargs"] = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str))}
    rec["runs"] = {}
    lc.install_hooks()
    raw_theta = basis.independent_phases()[0]
    rate = basis.independent_phase_rates()[0]
    for dm in (6, 8, 10, 12):
        lc.reset_rec()
        kw = dict(kwargs)
        kw["depth_max"] = dm
        t0 = time.perf_counter()
        res = lq._adaptive_levin(x_span, f, basis, *args, **kw)
        wall = time.perf_counter() - t0
        cache = lc.REC["cache"]
        regions = res["regions"]
        win = [g for g in regions if g.type == "window"]
        other = [g for g in regions if g.type != "window"]
        pts = res["stationary_points"]
        x_r = pts[0]["x"] if pts else None
        th_r = float(raw_theta(x_r)) if pts else None
        request = max(kw.get("atol", 0.0), kw.get("rtol", 0.0) * abs(res["value"]))
        outside = []
        for g in other:
            near = g.end if (x_r is not None and g.end <= x_r) else g.start
            comp, base, trunc = lc.component(g)
            outside.append(
                {
                    "a": g.start,
                    "b": g.end,
                    "depth": g.depth,
                    "type": g.type,
                    "value": g._rec_value,
                    "abserr": g.abserr,
                    "floor": g.phase_err,
                    "truncation": g.abserr_truncation,
                    "delta1": g.delta1,
                    "delta2": g.delta2,
                    "r": g.convergence_ratio,
                    "trusted": g.trusted,
                    "declared_resolution": g.declared_resolution,
                    "phase_limited": g.phase_limited,
                    "total_err": g.total_err,
                    "component": comp,
                    "dtheta_near": abs(float(raw_theta(near)) - th_r) if pts else None,
                    "theta_prime_near": float(rate(near)),
                }
            )
        comp_out = {}
        for o in outside:
            comp_out[o["component"]] = comp_out.get(o["component"], 0.0) + o["total_err"]
        comp_win = {}
        for g in win:
            c, _, _ = lc.component(g)
            comp_win[c] = comp_win.get(c, 0.0) + g.total_err
        run = {
            "wall_s": wall,
            "value": res["value"],
            "abserr": res["abserr"],
            "request": request,
            "converged": res["converged"],
            "max_depth": res["max_depth"],
            "points": res["stationary_points"],
            "window_leaves": len(win),
            "window_sum_total_err": sum(g.total_err for g in win),
            "window_sum_nested": sum(g.abserr for g in win),
            "window_sum_floor": sum(g.phase_err for g in win),
            "window_by_component": comp_win,
            "window_sum_value": math.fsum(g._rec_value for g in win),
            "outside_leaves": len(other),
            "outside_sum_total_err": sum(o["total_err"] for o in outside),
            "outside_by_component": comp_out,
            "outside": outside,
        }
        if dm in (6, 8):
            order = max(kw.get("chebyshev_order", lc.ORDER), lq._LEVIN_MINIMUM_ALLOWED_ORDER)
            agg = {"C1": 0.0, "C2": 0.0, "C3": 0.0, "D": 0.0, "recon_max_rel_diff": 0.0}
            for g in win:
                tm = lg_terms(basis, cache, f, (g.start, g.end), order)
                for key in ("C1", "C2", "C3", "D"):
                    agg[key] += tm[key]
                fl = tm["C1"] + tm["C2"] + tm["C3"] + tm["D"]
                agg["recon_max_rel_diff"] = max(
                    agg["recon_max_rel_diff"], abs(fl - g.phase_err) / g.phase_err
                )
            run["window_floor_terms"] = agg
            # GL reference on sampled leaves: the window leaf holding x_r, the outermost window
            # leaf on the right, one halfway; and the three outside leaves of largest declared error
            picks = []
            if pts:
                picks.append(min(win, key=lambda g: 0 if g.start <= x_r <= g.end else min(abs(g.start - x_r), abs(g.end - x_r))))
            picks.append(max(win, key=lambda g: g.end))
            picks.append(win[len(win) // 2])
            picks += sorted(other, key=lambda g: -g.total_err)[:3]
            refs = []
            for g in picks:
                gl = {n: gl_reference(basis, f, (g.start, g.end), n) for n in (32, 64, 128, 256)}
                refs.append(
                    {
                        "span": [g.start, g.end],
                        "type": g.type,
                        "depth": g.depth,
                        "value": g._rec_value,
                        "declared": g.total_err,
                        "nested_or_residual": g.abserr,
                        "floor": g.phase_err,
                        "gl": {str(n): v for n, v in gl.items()},
                        "gl_diffs": {
                            "64-32": gl[64] - gl[32],
                            "128-64": gl[128] - gl[64],
                            "256-128": gl[256] - gl[128],
                        },
                        "value_minus_gl256": g._rec_value - gl[256],
                        "value_minus_gl128": g._rec_value - gl[128],
                    }
                )
            run["gl_refs"] = refs
        rec["runs"][f"d{dm}"] = run
        print(
            f"LG call15 d{dm}: {wall:.1f}s value {res['value']:.6g} abserr {res['abserr']:.3g} "
            f"request {request:.3g} window {run['window_sum_total_err']:.3g} "
            f"{ {k: f'{v:.2g}' for k, v in comp_win.items()} } outside "
            f"{run['outside_sum_total_err']:.3g} { {k: f'{v:.2g}' for k, v in comp_out.items()} }",
            flush=True,
        )
        out["stuck_call"] = rec
        (lc.OUT / "lg_seed0_call15.json").write_text(json.dumps(out, indent=1, default=str))


def main():
    out = {"header": lc.header(__file__)}
    new = lq._adaptive_levin
    count = [0]

    def wrapped(x_span, f, basis, *args, **kwargs):
        count[0] += 1
        if count[0] != STUCK_CALL:
            return new(x_span, f, basis, *args, **kwargs)
        lq._adaptive_levin = new
        analyse(x_span, f, basis, args, kwargs, out)
        raise _Abandon()

    lq._adaptive_levin = wrapped
    random.seed(0)
    try:
        unittest.TextTestRunner(stream=open("/dev/null", "w"), verbosity=0).run(
            unittest.TestLoader().loadTestsFromName(
                "LiouvilleGreen.tests.test_3bessel_analytic.Test3BesselAnalytic.test_JJJ"
            )
        )
    finally:
        lq._adaptive_levin = new
    print("wrote", lc.OUT / "lg_seed0_call15.json")


if __name__ == "__main__":
    main()
