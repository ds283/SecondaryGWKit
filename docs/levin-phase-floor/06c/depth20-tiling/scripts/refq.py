"""depth20-tiling step 4: an independent Gauss-Legendre reference for a captured call.

Adapted from var/levin-phase-floor-06c/y1-references/scripts/refquad.py (a record, read only; its
copy, unchanged, is scripts/refquad_y1.py). The method is that script's, unchanged: composite
Gauss-Legendre of f_sin sin(theta_mod_2pi) + f_cos cos(theta_mod_2pi) on the captured callables
themselves (not the Levin driver), panels of equal steps of the work coordinate
W = Phi/D + (x - a)/hmax (Phi the trapezoid of |theta'| on cells of width <= cell), two resolutions
(nlow, nhigh nodes per panel) on the same panels, each node's term w_i (h/2)[...] in double, summed
exactly (Shewchuk partials, merged, one final rounding). What changed: the capture format (this
task's, written by worker.py through x1_capture.py; no store scale); the replay check is
dropped here (each capture's replay is runcall.py's, state/runs/replay/); no cross-tree sample or
compare sub-command; ``size`` estimates the cost before anything is integrated; one command does
panels and integration.

  size PICKLE-ENTRY: Phi, the panel count and the per-node cost of the callables (timed on 2000
     nodes), hence an estimate of the integration's CPU and wall time at --workers.
  run: panels, then both resolutions; writes state/ref/NN.ref.json.

Usage: PYTHONPATH=<tree>:<scripts> python -B refq.py {size|run} CAPTURED_JSON SAMPLE_NO [--D 512] [--hmax 0.005]
       [--cell 1e-3] [--nlow 192] [--nhigh 384] [--workers 6] [--chunks 48]
"""

import argparse
import concurrent.futures as cf
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import STATE, header, hx  # noqa: E402


def msum_add(partials, x):
    i = 0
    for y in partials:
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo:
            partials[i] = lo
            i += 1
        x = hi
    partials[i:] = [x]


def entry_and_rec(cap_p, no):
    import dcommon as DC

    e = [x for x in json.loads(Path(cap_p).read_text()) if x["sample_no"] == no][0]
    return e, DC.load_capture(e)


def make_panels(rec, D, hmax, cell):
    import numpy as np

    x0, x1 = tuple(rec["x_span"])
    thd = rec["theta"]["theta_deriv"]
    ncell = max(1, int(math.ceil((x1 - x0) / cell)))
    edges = np.linspace(x0, x1, ncell + 1)
    edges[0], edges[-1] = x0, x1
    tp = np.array([thd(float(x)) for x in edges])
    atp = np.abs(tp)
    phi = 0.5 * (atp[1:] + atp[:-1]) * np.diff(edges)
    W = np.concatenate([[0.0], np.cumsum(phi / D + np.diff(edges) / hmax)])
    P = int(math.ceil(W[-1]))
    pe = np.interp(np.linspace(0.0, W[-1], P + 1), W, edges)
    pe[0], pe[-1] = x0, x1
    assert pe[0] == x0 and pe[-1] == x1 and np.all(np.diff(pe) > 0)
    return pe, float(phi.sum()), float(atp.max())


_W = {}


def _winit(cap_p, no, nlow, nhigh):
    from scipy.special import roots_legendre

    _e, rec = entry_and_rec(cap_p, no)
    _W["rec"] = rec
    _W["gl"] = {n: roots_legendre(n) for n in (nlow, nhigh)}


def _work(args):
    edges, i0, i1, nlow, nhigh = args
    rec = _W["rec"]
    fs, fc = rec["f"]
    thm = rec["theta"]["theta_mod_2pi"]
    out = {}
    t0 = time.perf_counter()
    for n in (nlow, nhigh):
        t, w = _W["gl"][n]
        parts = []
        absum = 0.0
        for p in range(i0, i1):
            a_, b_ = float(edges[p]), float(edges[p + 1])
            c = 0.5 * (a_ + b_)
            hh = 0.5 * (b_ - a_)
            for k in range(n):
                x = c + hh * float(t[k])
                m_ = thm(x)
                term = (float(w[k]) * hh) * (float(fs(x)) * math.sin(m_) + float(fc(x)) * math.cos(m_))
                msum_add(parts, term)
                absum += abs(term)
        out[n] = {"partials": parts, "absum": absum}
    out["seconds"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["size", "run"])
    ap.add_argument("captured")
    ap.add_argument("no", type=int)
    ap.add_argument("--D", type=float, default=512.0)
    ap.add_argument("--hmax", type=float, default=0.005)
    ap.add_argument("--cell", type=float, default=1e-3)
    ap.add_argument("--nlow", type=int, default=192)
    ap.add_argument("--nhigh", type=int, default=384)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunks", type=int, default=48)
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("at most six workers")
    import numpy as np

    cap_p = str(Path(a.captured).resolve())
    hdr = header(Path(__file__).resolve(), f"depth20-tiling: GL reference, {a.cmd}, sample {a.no}", [Path(cap_p), HERE / "dcommon.py"])
    for l in hdr:
        print(l, flush=True)
    e, rec = entry_and_rec(cap_p, a.no)
    t0 = time.perf_counter()
    pe, Phi, maxtp = make_panels(rec, a.D, a.hmax, a.cell)
    t_pan = time.perf_counter() - t0
    npan = len(pe) - 1
    fs, fc = rec["f"]
    thm = rec["theta"]["theta_mod_2pi"]
    xs = np.linspace(float(pe[0]), float(pe[-1]), 2000)
    t1 = time.perf_counter()
    for x in xs:
        m_ = thm(float(x))
        _ = float(fs(float(x))) * math.sin(m_) + float(fc(float(x))) * math.cos(m_)
    per_node = (time.perf_counter() - t1) / len(xs)
    nodes = npan * (a.nlow + a.nhigh)
    est_cpu = nodes * per_node * 1.3
    info = {"sample_no": a.no, "unit": e["unit"], "index": e["index"], "notify_label": rec["notify_label"], "x_span": [hx(pe[0]), hx(pe[-1])],
            "Phi_rad": Phi, "max_abs_theta_prime": maxtp, "npanels": npan, "nodes_both": nodes, "per_node_s": per_node,
            "est_cpu_s": est_cpu, "est_wall_s_at_workers": est_cpu / a.workers, "panels_s": t_pan, "D": a.D, "hmax": a.hmax, "cell": a.cell,
            "nlow": a.nlow, "nhigh": a.nhigh}
    print(json.dumps(info, indent=1), flush=True)
    if a.cmd == "size":
        return
    bounds = np.linspace(0, npan, min(a.chunks, npan) + 1).astype(int)
    jobs = [(pe, int(bounds[i]), int(bounds[i + 1]), a.nlow, a.nhigh) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    t2 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_winit, initargs=(cap_p, a.no, a.nlow, a.nhigh)) as ex:
        res = list(ex.map(_work, jobs))
    wall = time.perf_counter() - t2
    S, absum = {}, {}
    for n in (a.nlow, a.nhigh):
        allp = []
        for r in res:
            allp.extend(r[n]["partials"])
        S[n] = math.fsum(allp)
        absum[n] = math.fsum(r[n]["absum"] for r in res)
    nl, nh = a.nlow, a.nhigh
    doc = dict(info, header=hdr, S_low=hx(S[nl]), S_high=hx(S[nh]), ref=hx(S[nh]), ref_repr=repr(S[nh]), conv_diff=hx(abs(S[nh] - S[nl])),
               absum_high=hx(absum[nh]), rounding_bound_linear=hx(4 * 2.0**-52 * absum[nh]),
               cancellation=absum[nh] / abs(S[nh]) if S[nh] != 0 else None, wall_s=wall, cpu_s=sum(r["seconds"] for r in res),
               census_value=e["census_value"], census_abserr=e["census_abserr"])
    od = STATE / "ref"
    od.mkdir(exist_ok=True)
    tag = "" if (nl, nh) == (192, 384) else f".n{nl}-{nh}"
    (od / f"{a.no:02d}{tag}.ref.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "header"}, indent=1))


if __name__ == "__main__":
    main()
