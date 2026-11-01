"""levin-phase-floor prompt 06i: where Route B's dependence on the smooth-phase fit comes from. A
diagnostic, not a reading. New in 06i.

For one Route B sample, every segment's FCC integral (N = 48, one t-panel, first-order abscissa
correction as in route_b.py) under four fit designs (M, K) in {20, 24} x {241, 301}, and per
segment: the difference of each design from (20, 241), the two edge values p(X_l), p(X_r) under
each design, and the segment's sigma. Summed in bins of x, so that the source of a design
dependence can be located.
Usage (census tree as cwd): python -B diag_fit.py N
"""

import json
import multiprocessing as mpp
import sys
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filon  # noqa: E402
import prov  # noqa: E402
import route_b as RB  # noqa: E402
import samples as S  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "out"
DESIGNS = ((20, 241), (24, 301), (24, 241), (20, 301))
N = 48
_W = {}


def _init(n):
    filon.set_precision()
    _W["rec"] = S.load(n)
    _W["edges"], _ = S.route_b_edges(_W["rec"])


def seg_integral(rec, sg):
    s_nodes, _ = filon.cc_table(N)
    c, h = (sg.T_l + sg.T_r) / 2, (sg.T_r - sg.T_l) / 2
    g, eps = [], []
    for sj in s_nodes:
        t = c + h * sj
        y = sg.solve_y(t, shift=c)
        xh = float(sg.mid + sg.half * y)
        yh = sg.y_of(mp.mpf(xh))
        Fv = RB._F(rec, xh)
        g.append(mp.mpc(Fv.real, Fv.imag) / sg.pprime_x(yh))
        eps.append(sg.p(yh) - t)
    cg = filon.cheb_coeffs(g, N)
    dg = filon.cheb_deriv_at_nodes(cg, N)
    gc = [g[j] - eps[j] * dg[j] / h for j in range(N + 1)]
    c2 = filon.cheb_coeffs(gc, N)
    return h * mp.expj(c) * filon.fcc_sum(c2, filon.cheb_moments(h, N), N)


def job(args):
    i0, i1 = args
    rec, edges = _W["rec"], _W["edges"]
    res = {}
    for M, K in DESIGNS:
        segs = RB.fit_segments(rec, edges, i0, i1, M, K)
        res[(M, K)] = [
            (
                mp.nstr(seg_integral(rec, sg).real, 40),
                mp.nstr(sg.T_l, 40),
                mp.nstr(sg.T_r, 40),
                sg.sigma,
            )
            for sg in segs
        ]
    out = []
    for k, i in enumerate(range(i0, i1)):
        out.append(
            dict(
                i=i,
                X_l=edges[i],
                X_r=edges[i + 1],
                sigma=res[DESIGNS[0]][k][3],
                I={f"{M},{K}": res[(M, K)][k][0] for M, K in DESIGNS},
                T_l={f"{M},{K}": res[(M, K)][k][1] for M, K in DESIGNS},
                T_r={f"{M},{K}": res[(M, K)][k][2] for M, K in DESIGNS},
            )
        )
    return out


def main():
    n = int(sys.argv[1])
    filon.set_precision()
    rec = S.load(n)
    edges, _ = S.route_b_edges(rec)
    nseg = len(edges) - 1
    nch = 36
    b = [round(nseg * i / nch) for i in range(nch + 1)]
    with mpp.get_context("spawn").Pool(6, initializer=_init, initargs=(n,)) as pool:
        rows = [
            r
            for part in pool.map(
                job, [(b[i], b[i + 1]) for i in range(nch)], chunksize=1
            )
            for r in part
        ]
    mp.mp.dps = 50
    keys = [f"{M},{K}" for M, K in DESIGNS]
    tot = {k: mp.fsum(mp.mpf(r["I"][k]) for r in rows) for k in keys}
    L = [
        f"sample {n}: {nseg} segments; totals "
        + ", ".join(f"({k}) {mp.nstr(v, 25)}" for k, v in tot.items())
    ]
    for k in keys[1:]:
        L.append(f"  ({k}) - (20,241) = {float(tot[k] - tot[keys[0]]):+.3e}")
    # where: the 10 segments with the largest |difference| for (24,301) - (20,241), and the
    # difference binned in x
    d = [(abs(mp.mpf(r["I"][keys[1]]) - mp.mpf(r["I"][keys[0]])), r) for r in rows]
    d.sort(key=lambda z: -z[0])
    L.append("largest per-segment |I(24,301) - I(20,241)|:")
    for v, r in d[:10]:
        dT_l = float(mp.mpf(r["T_l"][keys[1]]) - mp.mpf(r["T_l"][keys[0]]))
        dT_r = float(mp.mpf(r["T_r"][keys[1]]) - mp.mpf(r["T_r"][keys[0]]))
        L.append(
            f"  seg {r['i']} x [{r['X_l']:.6f}, {r['X_r']:.6f}] |dI| {float(v):.3e} sigma {r['sigma']:.2e} "
            f"dp(X_l) {dT_l:+.2e} dp(X_r) {dT_r:+.2e} I {float(mp.mpf(r['I'][keys[0]])):.3e}"
        )
    bins = {}
    for r in rows:
        bk = round(r["X_l"], 0)
        bins.setdefault(bk, mp.mpf(0))
        bins[bk] += mp.mpf(r["I"][keys[1]]) - mp.mpf(r["I"][keys[0]])
    L.append(
        "binned in x (integer bins), sum of I(24,301) - I(20,241): "
        + ", ".join(f"{k:.0f}: {float(v):+.2e}" for k, v in sorted(bins.items()))
    )
    out = {
        "header": prov.header([S.pickle_path(n)]),
        "sample": n,
        "summary": L,
        "rows": rows,
    }
    (OUT / f"diag_fit_s{n:02d}.json").write_text(json.dumps(out, indent=1))
    (OUT / f"diag_fit_s{n:02d}.txt").write_text(
        json.dumps(out["header"], indent=1) + "\n\n" + "\n".join(L) + "\n"
    )
    print("\n".join(L))


if __name__ == "__main__":
    main()
