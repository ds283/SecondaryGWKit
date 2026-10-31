"""Orchestrator's own re-check of one Y1 reference (06c Part B follow-up). Read-only.

Usage: PYTHONPATH=<tree> python -B orch_ref.py PICKLE [--P 128] [--cell 1e-3] [--n1 64] [--n2 96] [--workers 6]

Independent of the agent's refquad.py: panels from the raw phase theta (not theta_deriv), tabulated
on uniform cells, each cell cut into ceil(|dtheta|/P) equal sub-panels; Gauss-Legendre from numpy
(leggauss) at n1 and n2 nodes per sub-panel; per-chunk math.fsum, chunks summed by math.fsum.
Integrand as adaptive_levin_sincos takes it: f[0] sin(theta_mod_2pi) + f[1] cos(theta_mod_2pi).
Also replays the driver on the pickle, and reports sign changes of theta between cells.
"""

import argparse
import hashlib
import math
import multiprocessing as mp
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

_R = {}


def load(path):
    import ray.cloudpickle as cp

    blob = open(path, "rb").read()
    return cp.loads(blob), hashlib.sha256(blob).hexdigest()


def _init(path, n1, n2):
    _R["rec"], _ = load(path)
    _R["gl"] = {n: np.polynomial.legendre.leggauss(n) for n in (n1, n2)}


def _chunk(args):
    edges, n = args
    rec = _R["rec"]
    fs, fc = rec["f"]
    thm = rec["theta"]["theta_mod_2pi"]
    t, w = _R["gl"][n]
    terms = []
    for a, b in zip(edges[:-1], edges[1:]):
        c, h = 0.5 * (a + b), 0.5 * (b - a)
        for tk, wk in zip(t, w):
            x = c + h * float(tk)
            m = thm(x)
            terms.append(float(wk) * h * (fs(x) * math.sin(m) + fc(x) * math.cos(m)))
    return math.fsum(terms), math.fsum(abs(v) for v in terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--P", type=float, default=128.0)
    ap.add_argument("--cell", type=float, default=1e-3)
    ap.add_argument("--n1", type=int, default=64)
    ap.add_argument("--n2", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert a.workers <= 6
    import AdaptiveLevin.levin_quadrature as LQ

    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    print("host", socket.gethostname(), "utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print("tree", tree, subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
          "status lines", len(subprocess.run(["git", "-C", tree, "status", "--short"], capture_output=True, text=True).stdout.splitlines()))
    print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
    rec, sha = load(a.pickle)
    print("pickle", a.pickle, "sha256", sha)
    span = tuple(rec["x_span"])
    rr = LQ.adaptive_levin_sincos(span, rec["f"], theta=rec["theta"], **rec["kw"])
    print("span", span[0].hex(), span[1].hex(), "kw", {k: v for k, v in rec["kw"].items() if not callable(v)})
    print("driver replay value", rr["value"].hex(), "abserr", rr["abserr"].hex(), "regions", rr["num_regions"], "depth", rr["max_depth"],
          "converged", rr.get("converged"))
    print("  equals captured", (rr["value"].hex(), rr["abserr"].hex()) == (rec["value"], rec["abserr"]))
    print("  scaled by", rec.get("scale"), "->", (float.fromhex(rec["scale"]) * rr["value"]).hex() if rec.get("scale") else None,
          repr(float.fromhex(rec["scale"]) * rr["value"]) if rec.get("scale") else None)

    th = rec["theta"]["theta"]
    thd = rec["theta"]["theta_deriv"]
    ncell = math.ceil((span[1] - span[0]) / a.cell)
    xs = np.linspace(span[0], span[1], ncell + 1)
    xs[0], xs[-1] = span
    T = np.array([th(float(x)) for x in xs])
    d = np.diff(T)
    sc = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    print(f"cells {ncell}; total |dtheta| {np.abs(d).sum():.6g} rad; theta extrema between cells at x =",
          [float(xs[i + 1]) for i in sc])
    for i in sc:
        x0 = float(xs[i + 1])
        print(f"  near x={x0!r}: theta_deriv at x0-cell, x0, x0+cell = {thd(x0 - a.cell):.4g}, {thd(x0):.4g}, {thd(x0 + a.cell):.4g};"
              f" max|theta_deriv| on cell grid sample {max(abs(thd(float(x))) for x in xs[:: max(1, ncell // 2000)]):.4g}")
    edges = []
    for i in range(ncell):
        m = max(1, math.ceil(abs(d[i]) / a.P))
        e = np.linspace(xs[i], xs[i + 1], m + 1)
        edges.extend(map(float, e[:-1]))
    edges.append(float(xs[-1]))
    npan = len(edges) - 1
    print("sub-panels", npan, "nodes", npan * (a.n1 + a.n2))
    nch = 6 * a.workers * 4
    step = math.ceil(npan / nch)
    jobs = []
    for n in (a.n1, a.n2):
        for i in range(0, npan, step):
            jobs.append((edges[i: i + step + 1], n))
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.pickle, a.n1, a.n2)) as pool:
        res = pool.map(_chunk, jobs, chunksize=1)
    half = len(jobs) // 2
    S1 = math.fsum(r[0] for r in res[:half])
    S2 = math.fsum(r[0] for r in res[half:])
    A2 = math.fsum(r[1] for r in res[half:])
    print(f"wall {time.time() - t0:.1f} s")
    print(f"S{a.n1} = {S1.hex()} ({S1!r})")
    print(f"S{a.n2} = {S2.hex()} ({S2!r})   |S2-S1| = {abs(S2 - S1):.4g}   sum|terms| {A2:.4g}")
    print(f"driver (this tree) |value - S{a.n2}| / abserr = {abs(rr['value'] - S2) / rr['abserr']:.4g}")


if __name__ == "__main__":
    main()
