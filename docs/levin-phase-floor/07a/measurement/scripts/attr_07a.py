"""levin-phase-floor prompt 07a Part A, section 2.2, the attribution: a reference for every returned leaf of a call at
07's code, by ``tools.filon_reference`` (Route B), the method of the call's own readings. New in 07a.

**The method and its basis.** The call's reading (07's ``one_reading.py``: ``tools.filon_reference.reading``) splits
the span at the integrand's own representation changes (``production_edges``), fits a smooth phase on each segment
(its ends shared with its neighbours), and sums the segments in chunks of ``ROUTE_B_CHUNK`` in order; a segment that
holds a stationary point, or lies beside one, is integrated directly by Gauss-Legendre in ``mpmath`` on panels of at
most 2 rad (resolution 1) or 1 rad (2) of phase. Here the edges are the production edges **and every returned leaf's
two ends** (a production edge within 1e-9 of a leaf end, the tool's own merging distance, is dropped, so that every
segment lies inside one leaf), the state is the tool's ``_make_state(call, "B", edges)``, and the jobs are the tool's
``("b-fcc", (i0, i1, res))``, cut so that no job crosses a leaf end, run by the tool's ``_run_jobs`` (its spawn pool,
each worker loading the call by ``load_call`` below). Each leaf's reference is the sum, in order, of its jobs; their
sum over the leaves is a whole-call reading with the leaf ends as extra edges. Every segment's fit is the tool's
(``fit_segments`` shares each end with the neighbour's unconstrained fit, across job boundaries too), so the leaf
references are pieces of one reading, not separate readings with their own end fits. The window leaves span at
most 2 pi of phase and hold the stationary point: there the tool's direct path is ``mpmath`` Gauss-Legendre at
``DPS`` digits, the "mpmath at extended precision" of section 2.2. Independent of the code under test: the tool
imports nothing of ``AdaptiveLevin``'s driver; it evaluates the captured callables.

Both resolutions are read (``--res 1 2``); a leaf's stated accuracy is ``|ref_2 - ref_1|`` together with resolution
2's fit-noise bound (the tool's ``fit_noise_1sigma`` per leaf), and the reference is resolution 2 (07's rule).

Writes OUT_JSON: per leaf its span, the code's value and e' (from the replay at variant a, instrumented), and per
resolution the reference (``value_30``, ``imag_30``), the segment count, the direct segments, the fit-noise variance,
the jittered-minus-smooth value and the wall time; the whole sum; headers with this script's, the tool's and the
capture's SHA-256. Refuses to overwrite.

Usage: PYTHONPATH=<tree>:<this dir> python -B attr_07a.py SID REPLAY_JSON OUT_JSON [--res 1 2] [--workers 6]
"""

import argparse
import datetime
import hashlib
import json
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
TOOL = TREE / "tools" / "filon_reference.py"
TOOL_SHA = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"  # 07's readings' tool


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_call(path, offset, length, sha, rtol=None, atol=None):
    """The captured call (x1_capture.py's record), its blob SHA-256 checked; for the tool's workers too."""
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


def leaf_edges(prod, bounds):
    """The production edges with every leaf end added; a production edge within 1e-9 of a leaf end is dropped."""
    bset = sorted(set(bounds))
    keep = set(bset)
    for v in prod:
        if min(abs(v - b) for b in bset) > 1e-9:
            keep.add(v)
    edges = sorted(keep)
    for a, b in zip(edges[:-1], edges[1:]):
        if not b > a:
            raise RuntimeError("edges not strictly increasing")
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sid")
    ap.add_argument("replay_json")
    ap.add_argument("out_json")
    ap.add_argument("--res", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out_p = Path(a.out_json)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    if sha256(TOOL) != TOOL_SHA:
        raise SystemExit("the tool has another SHA-256")
    import mpmath as mp

    import tools.filon_reference as F

    rep = json.loads(Path(a.replay_json).read_text())
    if rep["variant"] != "a" or not rep["reproduces_census"] or rep["sid"] != a.sid:
        raise SystemExit("the replay is not variant a reproducing the census for this call")
    c = rep["header"]["capture"]
    ov = c.get("override") or {}
    src_args = (c["path"], c["offset"], c["length"], c["sha256"], ov.get("rtol"), ov.get("atol"))
    call = load_call(*src_args)
    route = F.route_of(call)
    if route != "B":
        raise SystemExit(f"route {route}, not B")
    x0, x1 = (float(v) for v in call["x_span"])
    if not x1 > x0:
        raise SystemExit("a reversed span; not handled")
    fl = float.fromhex
    leaves = rep["leaves"]
    bounds = [fl(leaves[0]["start"])] + [fl(l["end"]) for l in leaves]
    for l0, l1 in zip(leaves[:-1], leaves[1:]):
        if l0["end"] != l1["start"]:
            raise SystemExit("the leaves do not tile the span")
    if bounds[0] != x0 or bounds[-1] != x1:
        raise SystemExit("the leaves do not cover the span")
    prod, n_tables = F.production_edges(call)
    edges = leaf_edges(prod, bounds)
    pos = {v: i for i, v in enumerate(edges)}
    seg_ranges = [(pos[fl(l["start"])], pos[fl(l["end"])]) for l in leaves]
    hdr = {"title": f"levin-phase-floor 07a Part A: leaf references of {a.sid}",
           "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
           "tool": str(TOOL), "tool_sha256": TOOL_SHA, "replay_json": str(Path(a.replay_json).resolve()),
           "replay_json_sha256": sha256(a.replay_json), "capture": c, "workers": a.workers}
    doc = {"header": hdr, "sid": a.sid, "n_leaves": len(leaves), "n_production_edges": len(prod),
           "n_edges": len(edges), "n_segments": len(edges) - 1, "n_tables": n_tables, "readings": {}}
    call_source = ("attr_07a", "load_call", src_args)
    for res in a.res:
        t0 = time.perf_counter()
        jobs, owner = [], []
        for li, (i0, i1) in enumerate(seg_ranges):
            for j in range(i0, i1, F.ROUTE_B_CHUNK):
                jobs.append(("b-fcc", (j, min(j + F.ROUTE_B_CHUNK, i1), res)))
                owner.append(li)
        with mp.workdps(F.DPS):
            state = F._make_state(call, "B", edges) if a.workers <= 1 else None
            results = F._run_jobs(jobs, state, a.workers, call_source, "B", edges)
            per = [mp.mpc(0) for _ in leaves]
            var = [0.0 for _ in leaves]
            jit = [mp.mpf(0) for _ in leaves]
            nseg = [0 for _ in leaves]
            ndir = [0 for _ in leaves]
            for li, (re_, im_, ex) in zip(owner, results):
                per[li] += mp.mpc(mp.mpf(re_), mp.mpf(im_))
                var[li] += ex["var"]
                jit[li] += mp.mpf(ex["jittered"][0]) - mp.mpf(re_)
                nseg[li] += len(ex["segments"])
                ndir[li] += sum(1 for d in ex["segments"] if d["direct"])
            total = mp.mpc(0)
            for v in per:
                total += v
            wall = time.perf_counter() - t0
            doc["readings"][str(res)] = {
                "res": res, "wall_s": wall, "n_jobs": len(jobs), "total_30": mp.nstr(total.real, 30),
                "total_imag_30": mp.nstr(total.imag, 30),
                "leaves": [{"i": li, "value_30": mp.nstr(per[li].real, 30), "imag_30": mp.nstr(per[li].imag, 30),
                            "fit_noise_var": var[li], "jittered_minus_smooth": float(jit[li]), "n_segments": nseg[li],
                            "direct_segments": ndir[li]} for li in range(len(leaves))],
            }
        print(f"{a.sid} res {res}: {len(leaves)} leaves, {len(edges) - 1} segments, {len(jobs)} jobs, total "
              f"{mp.nstr(total.real, 20)} in {wall:.1f} s", flush=True)
    doc["leaves"] = [{"i": li, "start": l["start"], "end": l["end"], "depth": l["depth"], "type": l["type"],
                      "window": l["window"], "window_index": l["window_index"], "fell": l["fell"],
                      "phase_limited": l["phase_limited"], "value": l["value"], "budget_err": l["budget_err"],
                      "node_error_signed": l["node_error_signed"], "raw": l["raw"]} for li, l in enumerate(leaves)]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    sys.exit(main())
