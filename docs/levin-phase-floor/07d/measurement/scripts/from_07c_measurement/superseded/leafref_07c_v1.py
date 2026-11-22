"""levin-phase-floor prompt 07c Part A, section 2.5: 07b Part A's scripts/leafref_07b.py (a record) copied, changed only
where marked "prompt 07c": TREE is this worktree (2fd979fd), whose tools/filon_reference.py is the same file
(d37e633b..., checked). 07b Part A's text follows.

levin-phase-floor prompt 07b Part A, section 2.2: references on a leaf's span and on its pieces (halves, quarters,
eighths, and the leaves of the window variants), by ``tools.filon_reference`` Route B FCC. New in 07b, after 07a Part A's
attr_07a.py (a record), whose method it keeps and restricts to the spans asked for.

**The method and its basis** (07a Part A's, attr_07a.py's docstring): the segment edges are the integrand's production
edges (``production_edges``) **and every point of the spec** (leaf ends, the halves', quarters' and eighths' break points,
the variants' leaf ends); a production edge within 1e-9 of a point is dropped (the tool's merging distance), so that every
segment lies between two consecutive points or production edges. The state is the tool's ``_make_state(call, "B",
edges)``; the jobs are the tool's ``("b-fcc", (i0, i1, res))`` over the segments inside the spec's spans only, each job at
most ``ROUTE_B_CHUNK`` segments and never crossing a point, run by the tool's spawn pool (each worker loading the call by
``load_call`` below). A piece's reference is the sum, in order, of its jobs. Every segment's fit is the tool's
(``fit_segments`` shares each end with its neighbour's unconstrained fit), so the pieces are parts of one reading on
these edges. A segment holding the stationary point, or beside it (slope ratio above 2), is integrated directly by
Gauss-Legendre in ``mpmath`` (the tool's direct path). Independent of the code under test: the tool imports nothing of
``AdaptiveLevin``'s driver; it evaluates the captured callables.

Both resolutions are read by default; a piece's stated accuracy is |ref_2 - ref_1| together with resolution 2's fit-noise
bound, and its reference is resolution 2 (07's rule).

The spec (JSON): {"sid", "capture": {path, offset, length, sha256, override}, "spans": [[A, B], ...] (hex; each A and B
must be points), "points": [hex, ...]}. Writes OUT_JSON: per piece between consecutive points inside a span, at each
resolution, the value (30 digits, real and imaginary), the fit-noise variance, the jittered-minus-smooth value, the
segment count, the direct segments and the summed job wall; per job its wall; headers with this script's, the tool's and
the capture's SHA-256. Refuses to overwrite.

``--split-ratio R`` (prompt 07b, after the pilot): every monotone segment inside a span whose rate varies by more than R
is bisected until none does (``split_steep``), so that the tool integrates it in t rather than directly (its own rule
sends a segment of slope ratio above 2 to the direct path); the pieces, and so the references, are unchanged in what
they integrate. Validated against the pilot's production-edge reading of v08 (out/leafrefs/pilot_v08.json).

Usage: PYTHONPATH=<tree>:<scripts> python -B leafref_07b.py SPEC_JSON OUT_JSON [--res 1 2] [--workers 6] [--split-ratio R]
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f")  # prompt 07c
TOOL = TREE / "tools" / "filon_reference.py"
TOOL_SHA = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"  # 07's readings' tool, unchanged


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_call(path, offset, length, sha, rtol=None, atol=None):
    """The captured call, its blob SHA-256 checked; for the tool's workers too (attr_07a.py's)."""
    import ray.cloudpickle as cp

    with open(path, "rb") as f:
        f.seek(offset)
        b = f.read(length)
    if hashlib.sha256(b).hexdigest() != sha:
        raise RuntimeError(f"{path}@{offset}: sha256 differs")
    rec = cp.loads(b)
    kw = dict(rec["kw"])
    if rtol is not None:
        kw["rtol"], kw["atol"] = rtol, atol
    return {"x_span": rec["x_span"], "f": rec["f"], "theta": rec["theta"], "kw": kw, "notify_label": rec["notify_label"],
            "unit": rec["unit"], "index": rec["index"]}


def timed_chunk(job):
    import tools.filon_reference as F

    t0 = time.perf_counter()
    re_, im_, ex = F._run_chunk(job)
    return re_, im_, ex, time.perf_counter() - t0


def merged_edges(prod, points):
    pset = sorted(set(points))
    keep = set(pset)
    for v in prod:
        if min(abs(v - b) for b in pset) > 1e-9:
            keep.add(v)
    edges = sorted(keep)
    for a, b in zip(edges[:-1], edges[1:]):
        if not b > a:
            raise RuntimeError("edges not strictly increasing")
    return edges


def split_steep(call, edges, lo, hi, ratio, max_depth=80):
    """prompt 07b: the monotone segments inside [lo, hi] whose rate |theta'| varies by more than ``ratio`` over 33
    Chebyshev-Lobatto samples (07's ``features`` rule), bisected in x until no part does; segments where theta' takes both
    signs (a stationary point) are left whole. Returns the added edges."""
    import numpy as np

    thd = call["theta"]["theta_deriv"]
    y = np.cos(np.pi * np.arange(33) / 32)

    def steep(a, b):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        d = np.array([float(thd(mid + half * v)) for v in y])
        if not (np.all(d > 0) or np.all(d < 0)):
            return False
        d = np.abs(d)
        return float(np.max(d) / np.min(d)) > ratio

    added = []
    stack = [(a, b, 0) for a, b in zip(edges[:-1], edges[1:]) if a >= lo and b <= hi]
    while stack:
        a, b, dep = stack.pop()
        if dep < max_depth and steep(a, b):
            m = (a + b) / 2.0
            if a < m < b:
                added.append(m)
                stack += [(a, m, dep + 1), (m, b, dep + 1)]
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out_json")
    ap.add_argument("--res", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--split-ratio", type=float, default=None)  # prompt 07b: split_steep's ratio; None leaves the edges
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("at most six workers")
    out_p = Path(a.out_json)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    if sha256(TOOL) != TOOL_SHA:
        raise SystemExit("the tool has another SHA-256")
    import mpmath as mp

    import tools.filon_reference as F

    spec = json.loads(Path(a.spec).read_text())
    c = spec["capture"]
    ov = c.get("override") or {}
    src_args = (c["path"], c["offset"], c["length"], c["sha256"], ov.get("rtol"), ov.get("atol"))
    call = load_call(*src_args)
    if F.route_of(call) != "B":
        raise SystemExit("not route B")
    fl = float.fromhex
    points = sorted({fl(p) for p in spec["points"]})
    spans = [(fl(s0), fl(s1)) for s0, s1 in spec["spans"]]
    for s0, s1 in spans:
        if s0 not in points or s1 not in points or not s1 > s0:
            raise SystemExit("a span's ends must be points, increasing")
    prod, n_tables = F.production_edges(call)
    edges = merged_edges(prod, points)
    n_split = 0
    if a.split_ratio is not None:
        add = []
        for s0, s1 in spans:
            add += split_steep(call, edges, s0, s1, a.split_ratio)
        n_split = len(set(add))
        edges = sorted(set(edges) | set(add))
    pos = {v: i for i, v in enumerate(edges)}
    pidx = sorted(pos[p] for p in points)
    pieces = []  # (point i, point j) consecutive points inside a span
    for s0, s1 in spans:
        i0, i1 = pos[s0], pos[s1]
        inner = [p for p in pidx if i0 <= p <= i1]
        for p, q in zip(inner[:-1], inner[1:]):
            if (p, q) not in pieces:
                pieces.append((p, q))
    pieces.sort()
    hdr = {"title": f"levin-phase-floor 07b Part A: piece references of {spec['sid']}",
           "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "tool": str(TOOL), "tool_sha256": TOOL_SHA, "spec": str(Path(a.spec).resolve()), "spec_sha256": sha256(a.spec),
           "capture": c, "workers": a.workers}
    doc = {"header": hdr, "sid": spec["sid"], "n_points": len(points), "n_production_edges": len(prod), "n_edges": len(edges),
           "n_tables": n_tables, "split_ratio": a.split_ratio, "n_split_edges": n_split, "pieces": [{"lo": edges[p].hex(), "hi": edges[q].hex(), "n_segments": q - p} for p, q in pieces],
           "readings": {}}
    call_source = ("leafref_07b", "load_call", src_args)
    for res in a.res:
        t0 = time.perf_counter()
        jobs, owner = [], []
        for k, (p, q) in enumerate(pieces):
            for j in range(p, q, F.ROUTE_B_CHUNK):
                jobs.append(("b-fcc", (j, min(j + F.ROUTE_B_CHUNK, q), res)))
                owner.append(k)
        with mp.workdps(F.DPS):
            if a.workers <= 1:
                F._WORKER["state"] = F._make_state(call, "B", edges)
                results = [timed_chunk(j) for j in jobs]
            else:
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(a.workers, initializer=F._worker_init, initargs=(call_source, "B", edges)) as pool:
                    results = pool.map(timed_chunk, jobs, chunksize=1)
            per = [mp.mpc(0) for _ in pieces]
            var = [0.0 for _ in pieces]
            jit = [mp.mpf(0) for _ in pieces]
            ndir = [0 for _ in pieces]
            wsum = [0.0 for _ in pieces]
            for k, (re_, im_, ex, w) in zip(owner, results):
                per[k] += mp.mpc(mp.mpf(re_), mp.mpf(im_))
                var[k] += ex["var"]
                jit[k] += mp.mpf(ex["jittered"][0]) - mp.mpf(re_)
                ndir[k] += sum(1 for d in ex["segments"] if d["direct"])
                wsum[k] += w
            wall = time.perf_counter() - t0
            doc["readings"][str(res)] = {
                "res": res, "wall_s": wall, "n_jobs": len(jobs), "job_wall_s": [r[3] for r in results],
                "pieces": [{"k": k, "value_30": mp.nstr(per[k].real, 30), "imag_30": mp.nstr(per[k].imag, 30),
                            "fit_noise_var": var[k], "jittered_minus_smooth": float(jit[k]), "direct_segments": ndir[k],
                            "job_wall_s": wsum[k]} for k in range(len(pieces))],
            }
        print(f"{spec['sid']} res {res}: {len(pieces)} pieces, {sum(q - p for p, q in pieces)} segments, {len(jobs)} jobs, "
              f"{sum(ndir)} direct, in {wall:.1f} s (job CPU {sum(r[3] for r in results):.0f} s)", flush=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    sys.exit(main())
