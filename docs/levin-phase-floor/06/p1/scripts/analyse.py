"""levin-phase-floor 06, finding P1, steps 3(a)-(c) and 5: under one tree, after the validated
rebuild (p1lib), for every phase group of the all-oscillatory sub-interval z in [250, 5.42e9]:

(a) the composed phase theta(log(1+z)) = group.theta and theta' = group.theta_deriv (the very
    callables handed to adaptive_levin_sincos): total variation, every sign change of theta' on a
    uniform grid of NGRID points refined by brentq, theta'' there (central difference of
    theta_deriv), the factors' rates there, the smallest local minima of |theta'| that do not
    cross zero, and the leading stationary-phase estimate of each stationary point's
    contribution, Re[(f_cos - i f_sin) sqrt(2 pi/|theta''|) exp(i(theta mod 2pi + sgn pi/4))];
(b) adaptive_levin_sincos exactly as QSI calls it (the captured x_span, f, theta dict, atol and
    chebyshev_order), at the stored rtol (must reproduce the captured value bit for bit) and at
    rtol 1e-9 and 1e-11;
(c) the same call on 2^m equal pieces in log(1+z), each with atol scaled by its share of the
    width, summed, for m in MS.

Values are *unscaled* (no (1 + z_response) factor): the stored group values are these times
(1 + z_response) = 251 (to rounding).

Usage: PYTHONPATH=<tree> OMP_NUM_THREADS=1 python -B analyse.py r13|r14
Writes out/analyse_<tree>.json and out/analyse_<tree>_pieces.npz.
"""

import cmath
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

NGRID = 40001
RTOLS = (1e-7, 1e-9, 1e-11)
MS = (0, 3, 6, 9, 12)


def levin_summary(res):
    regs = res["regions"]
    return {
        "value": fx(res["value"]),
        "abserr": fx(res["abserr"]),
        "abserr_resolution": fx(res["abserr_resolution"]),
        "abserr_roundoff": fx(res["abserr_roundoff"]),
        "converged": bool(res["converged"]),
        "num_regions": int(res["num_regions"]),
        "num_simple_regions": int(res["num_simple_regions"]),
        "evaluations": int(res["evaluations"]),
        "max_depth": int(max(r.depth for r in regs)) if regs else None,
        "elapsed": float(res["elapsed"]),
    }


def phase_analysis(group, factors, x0, x1, log):
    th = group.theta
    thd = group.theta_deriv
    xs = np.linspace(x0, x1, NGRID)
    t0 = time.perf_counter()
    d = np.array([thd(float(x)) for x in xs])
    t = np.array([th(float(x)) for x in xs])
    log(f"   phase grid {NGRID} pts in {time.perf_counter() - t0:.1f} s")
    sgn = np.sign(d)
    idx = np.nonzero(sgn[:-1] * sgn[1:] < 0)[0]
    roots = []
    for i in idx:
        r = brentq(lambda x: thd(x), float(xs[i]), float(xs[i + 1]), xtol=1e-15, rtol=1e-15, maxiter=200)
        h = 1e-5
        d2 = (thd(r + h) - thd(r - h)) / (2 * h)
        f_s = group.f_sin(r)
        f_c = group.f_cos(r)
        psi = group.theta_mod_2pi(r)
        amp = math.sqrt(2 * math.pi / abs(d2))
        est = ((f_c - 1j * f_s) * amp * cmath.exp(1j * (psi + math.copysign(math.pi / 4, d2)))).real
        rates = {name: fn(r) for name, fn in factors.items()}
        roots.append(
            {
                "x": fx(r),
                "z": fx(math.expm1(r)),
                "theta_raw": fx(th(r)),
                "theta_mod_2pi": fx(psi),
                "theta_pp": fx(d2),
                "f_sin": fx(f_s),
                "f_cos": fx(f_c),
                "envelope": fx(math.hypot(f_s, f_c)),
                "stationary_width_x": fx(amp),
                "stationary_phase_estimate": fx(est),
                "stationary_phase_magnitude": fx(math.hypot(f_s, f_c) * amp),
                "factor_rates_dtheta_dlog1pz": {k: fx(v) for k, v in rates.items()},
            }
        )
    # total variation: |theta| differences between consecutive turning points
    turning = [x0] + [float.fromhex(r["x"]["hex"]) for r in roots] + [x1]
    tv = sum(abs(th(turning[i + 1]) - th(turning[i])) for i in range(len(turning) - 1))
    # local minima of |theta'| that do not cross zero
    a = np.abs(d)
    mins = [
        i for i in range(1, NGRID - 1) if a[i] <= a[i - 1] and a[i] <= a[i + 1] and i not in idx and i - 1 not in idx
    ]
    mins = sorted(mins, key=lambda i: a[i])[:5]
    return {
        "x_span": [fx(x0), fx(x1)],
        "theta_at_ends": [fx(th(x0)), fx(th(x1))],
        "theta_deriv_at_ends": [fx(thd(x0)), fx(thd(x1))],
        "theta_deriv_range_on_grid": [fx(d.min()), fx(d.max())],
        "total_variation": fx(tv),
        "n_sign_changes": len(roots),
        "stationary_points": roots,
        "smallest_local_minima_abs_theta_deriv": [
            {"x": fx(xs[i]), "z": fx(math.expm1(xs[i])), "abs_theta_deriv": fx(a[i])} for i in mins
        ],
        # coarse profile, every 400th grid point
        "profile": [[float(xs[i]), float(t[i]), float(d[i])] for i in range(0, NGRID, 400)],
    }


def main():
    tree = sys.argv[1]
    log = lambda s: print(s, flush=True)  # noqa: E731
    out = {"header": header(Path(__file__), f"p1 analyse, tree {tree}")}
    pieces_npz = {}
    with p1lib.rebuilt_rows(tree, log=log) as (ev, ctx, rows):
        from harness import QSI_MOD

        levin = QSI_MOD.adaptive_levin_sincos
        out["evidence"] = ev
        out["rows"] = {}
        for name, row in rows.items():
            if not row["validation"]["all_identical"]:
                raise SystemExit(f"p1: {tree} {name} does not reproduce its store; stopping")
            gc, groups, calls = p1lib.low_subinterval_groups(row)
            kw = gc["kwargs"]
            Gk, Tq, Tr = kw["Gk"], kw["Tq"], kw["Tr"]
            G_rate = lambda x: Gk.phase.theta_deriv(x, x_is_log=True, log_derivative=True)  # noqa: E731
            q_rate = lambda x: Tq.omega(x, z_is_log=True) * math.exp(x)  # noqa: E731
            r_rate = lambda x: Tr.omega(x, z_is_log=True) * math.exp(x)  # noqa: E731
            rec = {}
            for label, group in groups.items():
                c = calls[label]
                x_span = tuple(c["args"][0])
                f = c["args"][1]
                theta = c["kwargs"]["theta"]
                atol = c["kwargs"]["atol"]
                order = c["kwargs"]["chebyshev_order"]
                rtol0 = c["kwargs"]["rtol"]
                log(f"== {tree} {name} {label}: x_span {x_span}, atol {atol!r}, rtol {rtol0}, order {order}")
                s = {"label": label, "signs": list(group.signs)}
                # (a)
                s["phase"] = phase_analysis(
                    group,
                    {"G": G_rate, "q": q_rate, "r": r_rate, "group": group.theta_deriv},
                    x_span[0],
                    x_span[1],
                    log,
                )
                for sp in s["phase"]["stationary_points"]:
                    log(f"   stationary point z={sp['z']['repr']} theta''={sp['theta_pp']['repr']} "
                        f"estimate {sp['stationary_phase_estimate']['repr']} |.| {sp['stationary_phase_magnitude']['repr']}")
                log(f"   total variation {s['phase']['total_variation']['repr']} rad")
                # (b)
                s["rtol_sweep"] = {}
                for rt in RTOLS:
                    t0 = time.perf_counter()
                    res = levin(x_span, f, theta=theta, atol=atol, rtol=rt, chebyshev_order=order,
                                notify_label=f"p1 {tree} {name} {label} rtol {rt:g}")
                    summ = levin_summary(res)
                    summ["wall_s"] = time.perf_counter() - t0
                    if rt == rtol0:
                        summ["reproduces_captured_bit_for_bit"] = (
                            float(res["value"]).hex() == float(c["result"]["value"]).hex()
                            and float(res["abserr"]).hex() == float(c["result"]["abserr"]).hex()
                        )
                    s["rtol_sweep"][f"{rt:g}"] = summ
                    log(f"   (b) rtol {rt:g}: {summ['value']['repr']} abserr {summ['abserr']['repr']} "
                        f"regions {summ['num_regions']} evals {summ['evaluations']} conv {summ['converged']} "
                        f"[{summ['wall_s']:.1f} s]")
                # (c)
                s["forced"] = {}
                W = x_span[1] - x_span[0]
                for m in MS:
                    n = 2**m
                    edges = np.linspace(x_span[0], x_span[1], n + 1)
                    edges[0], edges[-1] = x_span[0], x_span[1]
                    vals = np.empty(n)
                    errs = np.empty(n)
                    nreg = 0
                    allconv = True
                    t0 = time.perf_counter()
                    for i in range(n):
                        a, b = float(edges[i]), float(edges[i + 1])
                        res = levin((a, b), f, theta=theta, atol=atol * (b - a) / W, rtol=rtol0,
                                    chebyshev_order=order)
                        vals[i] = res["value"]
                        errs[i] = res["abserr"]
                        nreg += int(res["num_regions"])
                        allconv = allconv and bool(res["converged"])
                    total = float(math.fsum(vals))
                    s["forced"][str(m)] = {
                        "pieces": n,
                        "sum": fx(total),
                        "sum_abserr": fx(float(math.fsum(errs))),
                        "num_regions_total": nreg,
                        "all_converged": allconv,
                        "wall_s": time.perf_counter() - t0,
                    }
                    pieces_npz[f"{name}|{label}|m{m}|edges"] = edges
                    pieces_npz[f"{name}|{label}|m{m}|values"] = vals
                    pieces_npz[f"{name}|{label}|m{m}|abserr"] = errs
                    log(f"   (c) m={m}: sum {total!r} sum abserr {math.fsum(errs):.3g} regions {nreg} "
                        f"conv {allconv} [{time.perf_counter() - t0:.1f} s]")
                rec[label] = s
            out["rows"][name] = {
                "groups": rec,
                "stored_scale_1_plus_z_response": fx(1.0 + float(row["rb"]["z_response"].z)),
            }
    path = OUT / f"analyse_{tree}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    npz = OUT / f"analyse_{tree}_pieces.npz"
    np.savez(npz, **pieces_npz)
    log(f"wrote {path} sha256 {sha256(path)}")
    log(f"wrote {npz} sha256 {sha256(npz)}")


if __name__ == "__main__":
    main()
