"""levin-phase-floor prompt 07b Part A, section 2.2 (M3): an input recomputed more precisely. New in 07b.

The Levin solve takes the rate from ``theta_deriv`` (its ODE) and the phase at the ends from ``theta_mod_2pi``; the
references (tools/filon_reference.py Route B) integrate against a smooth phase fitted to the unwrapped captured phase
``raw + rem(theta_mod_2pi - raw)``, never against ``theta_deriv``. A mismatch between ``theta_deriv`` and the derivative
of the captured phase would therefore show as an error of the leaf against its reference, common to every level: (M3).
Here the derivative is recomputed from the phase itself and compared with ``theta_deriv``.

**Method.** On the carrying leaf's span, segments run from the chosen end (the window edge; with no window, the end
where the solve's grid resolves the rate worst, mech_07b.py's rule) with first width l/8 (l = |theta'/theta''| there,
from out/mech/mech_all.json) doubling to the far end; any segment whose |theta'| varies by more than 1.5 over 17
Chebyshev-Lobatto samples is bisected (at most 400 segments). On each segment the unwrapped phase is sampled in mpmath
(dps 40) at K = 25 Chebyshev-Lobatto points and interpolated (degree 24); its derivative p' is evaluated at the 24
interior points of the second kind (not nodes) and compared with ``theta_deriv`` there: delta = |p' - theta'|/|theta'|.
The interpolant's own accuracy is checked by its value at those points against the sampled phase (fit residual in rad).
The capture's jitter (about 0.6 eps |theta'| x per sample) sets a floor on delta of order K^2 0.6 eps x / h.

**Scale.** M3's perturbation (out/mech/mech_all.json ``M3``) gives the change of Q under theta' x (1 + 2^-10); the
uniform relative mismatch that would account for the leaf's true error is delta* = 2^-10 |err(Q)| / |dQ|. M3 by
theta' is excluded on a leaf when max delta is far below delta*.

**The integrated test** (added after the pointwise test's first run, whose floor on v09's leaf was near delta*): on each
segment, int theta' dx by 24-point Gauss-Legendre against the unwrapped phase's difference; delta_int its relative
mismatch. A phase noise sigma gives a floor sigma/|dphase|, far lower than the pointwise test's on wide segments.

Writes OUT_JSON. Usage: PYTHONPATH=<tree>:<scripts> python -B m3_theta_07b.py OUT_JSON SID[:VARIANT] ...
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from mpmath import mp, mpf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from leafref_07b import load_call  # noqa: E402

MECH = ROOT / "out" / "mech" / "mech_all.json"
mp.dps = 40
TWO_PI = 2 * mp.pi


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rem(v):
    """v reduced to (-pi, pi]."""
    r = v - TWO_PI * mp.floor(v / TWO_PI)
    return r - TWO_PI if r > mp.pi else r


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    mech = json.loads(MECH.read_text())["rows"]
    targets = []
    for t in sys.argv[2:]:
        sid, _, var = t.partition(":")
        var = var or "tree"
        rows = [r for r in mech if r["sid"] == sid and r["variant"] == var]
        targets.append(max(rows, key=lambda r: abs(r["err_over_eprime"].get("Q") or 0.0)))
    K = 25
    yk = [mp.cos(mp.pi * j / (K - 1)) for j in range(K)]  # nodes, 1 .. -1
    yt = [mp.cos(mp.pi * (j + 0.5) / (K - 1)) for j in range(K - 1)]  # test points, interior, not nodes
    y17 = np.cos(np.pi * np.arange(17) / 16)
    GL = np.polynomial.legendre.leggauss(24)
    results = []
    for r in targets:
        t0 = time.perf_counter()
        hdr = json.loads(Path(r["probe"]).read_text())["header"]
        c = hdr["capture"]
        ov = c.get("override") or {}
        call = load_call(c["path"], c["offset"], c["length"], c["sha256"], ov.get("rtol"), ov.get("atol"))
        th = call["theta"]
        lo, hi = r["lo"], r["hi"]
        if r["edge_lo"]:
            side = "lo"
        elif r["edge_hi"]:
            side = "hi"
        else:
            side = max(("lo", "hi"), key=lambda k: r["edge"][k]["spacing_over_l"] or 0.0)
        ell = r["edge"][side]["l"]
        # geometric segments from the chosen end
        h = ell / 8.0
        cuts = [0.0]
        while cuts[-1] + h < (hi - lo):
            cuts.append(cuts[-1] + h)
            h *= 2.0
        cuts.append(hi - lo)
        segs = [(lo + a, lo + b) for a, b in zip(cuts[:-1], cuts[1:])] if side == "lo" else \
            [(hi - b, hi - a) for a, b in zip(cuts[:-1], cuts[1:])]

        def steep(a, b):
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            d = np.abs([float(th["theta_deriv"](mid + half * v)) for v in y17])
            return float(d.max() / d.min()) > 1.5

        work, final = list(segs), []
        while work:
            a, b = work.pop()
            if len(final) + len(work) < 400 and steep(a, b) and a < 0.5 * (a + b) < b:
                m = 0.5 * (a + b)
                work += [(a, m), (m, b)]
            else:
                final.append((a, b))
        final.sort()

        def unwrapped(x):
            raw = mpf(th["theta"](x))
            return raw + rem(mpf(th["theta_mod_2pi"](x)) - raw)

        rows = []
        for a, b in final:
            mid, half = (mpf(a) + mpf(b)) / 2, (mpf(b) - mpf(a)) / 2
            xs = [float(mid + half * v) for v in yk]
            fv = [unwrapped(x) for x in xs]
            yk_act = [(mpf(x) - mid) / half for x in xs]  # the nodes actually sampled (rounded abscissae)
            w = [(-1) ** j * (mpf(1) / 2 if j in (0, K - 1) else 1) for j in range(K)]  # barycentric weights, CL points
            dmax, fres = 0.0, 0.0
            for y in yt:
                x = float(mid + half * y)
                ya = (mpf(x) - mid) / half
                num = den = dnum1 = dden1 = mpf(0)
                for j in range(K):
                    q = w[j] / (ya - yk_act[j])
                    num += q * fv[j]
                    den += q
                    q2 = w[j] / (ya - yk_act[j]) ** 2
                    dnum1 += q2 * fv[j]
                    dden1 += q2
                p = num / den
                dp_dy = (-dnum1 * den + num * dden1) / den ** 2  # d/dy of num/den, with d(q)/dy = -q2
                dp = dp_dy / half
                td = mpf(th["theta_deriv"](x))
                dmax = max(dmax, float(abs(dp - td) / abs(td)))
                fres = max(fres, float(abs(p - unwrapped(x))))
            rate = abs(float(th["theta_deriv"](float(mid))))
            # the integrated test: int_a^b theta' dx by Gauss-Legendre (24 points) against the phase's own difference
            integral = half * mp.fsum(mpf(wg) * mpf(th["theta_deriv"](float(mid + half * mpf(yg)))) for yg, wg in zip(*GL))
            dphase = unwrapped(b) - unwrapped(a)
            rows.append({"a": a, "b": b, "rate_mid": rate, "delta_max": dmax, "fit_residual_rad": fres,
                         "jitter_floor": K * K * 0.6 * 2.2e-16 * abs(float(mid)) / float(2 * half),
                         "dphase": float(dphase), "int_minus_dphase": float(integral - dphase),
                         "delta_int": float(abs(integral - dphase) / abs(dphase))})
        m3 = r["M3"]["theta_prime_+"]
        errQ = r["err"]["Q"]
        dstar = 2.0 ** -10 * abs(errQ) / abs(m3["Q"]) if m3["Q"] else None
        dmax = max(x["delta_max"] for x in rows)
        res = {"sid": r["sid"], "variant": r["variant"], "leaf": r["i"], "depth": r["depth"], "lo": lo, "hi": hi,
               "end": side, "window_end": bool(r["edge_lo"] or r["edge_hi"]), "ell": ell, "n_segments": len(rows),
               "err_Q_over_eprime": r["err_over_eprime"]["Q"], "dQ_theta_prime_plus": m3["Q"], "err_Q": errQ,
               "delta_star": dstar, "delta_max": dmax,
               "delta_max_first_eighth": max((x["delta_max"] for x in rows
                                             if (x["b"] <= lo + (hi - lo) / 8 if side == "lo" else x["a"] >= hi - (hi - lo) / 8)),
                                             default=rows[0 if side == "lo" else -1]["delta_max"]),  # a segment wider than the eighth: the end one
               "fit_residual_max_rad": max(x["fit_residual_rad"] for x in rows),
               "delta_int_max": max(x["delta_int"] for x in rows),
               "delta_int_leaf": abs(sum(x["int_minus_dphase"] for x in rows)) / abs(sum(x["dphase"] for x in rows)),
               "delta_int_weighted": sum(abs(x["int_minus_dphase"]) for x in rows) / sum(abs(x["dphase"]) for x in rows),
               "segments": rows,
               "wall_s": time.perf_counter() - t0}
        results.append(res)
        print(f"{r['sid']:16s} {r['variant']:4s} leaf {r['i']:3d} end {side}{'(w)' if res['window_end'] else ''} "
              f"{len(rows):3d} segs  max delta {dmax:.2e} (first eighth {res['delta_max_first_eighth']:.2e})  "
              f"int max {res['delta_int_max']:.2e} weighted {res['delta_int_weighted']:.2e} leaf {res['delta_int_leaf']:.2e}  "
              + (f"delta* {dstar:.2e}  ratio {dmax / dstar:.2e}  " if dstar else "delta* -  ")
              + f"fit res {res['fit_residual_max_rad']:.1e} rad  "
              f"{res['wall_s']:.0f} s", flush=True)
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "mech": str(MECH), "mech_sha256": sha256(MECH),
                      "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "results": results}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    main()
