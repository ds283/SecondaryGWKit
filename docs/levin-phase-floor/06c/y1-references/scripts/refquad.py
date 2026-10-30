"""Y1 references, step 5: an independent quadrature of one captured Levin call.

Not the Levin driver: composite Gauss-Legendre on panels, of the integrand the driver is handed,

    I = int_a^b [ f_sin(x) sin(theta_mod_2pi(x)) + f_cos(x) cos(theta_mod_2pi(x)) ] dx,

with f = [f_sin, f_cos] and theta = {"theta", "theta_mod_2pi", "theta_deriv"} the captured call's
own callables (adaptive_levin_sincos integrates f[0] sin(theta) + f[1] cos(theta) over x_span =
(a, b), a < b, evaluating sin and cos from theta_mod_2pi when the dict supplies it:
_Basis_SinCos.eval_basis / build_Levin_data). Each node's contribution is w_i (h/2) [ ... ] in
double; the contributions are summed exactly (Shewchuk partials, merged across workers, rounded
once by math.fsum).

Sub-commands (PYTHONPATH=<tree>:<scripts>):

  panels PICKLE OUT_PREFIX [--D 512] [--hmax 0.005] [--cell 1e-3]
      Replays the call (bit for bit against the capture and the store), samples theta' (the
      dict's theta_deriv) at the edges of uniform cells of width <= --cell in x, and forms the
      work coordinate W(x) = Phi(x) / D + (x - a) / hmax, Phi the trapezoid of |theta'|, piecewise
      linear between cell edges. The panels are ceil(W(b)) equal steps of W. So each panel spans
      about D radians of phase or less, and at most hmax in x. Locates every zero of theta' (sign change between cell edges,
      refined by bisection to adjacent doubles). Writes OUT_PREFIX.edges.npy and .json.

  integrate PICKLE OUT_PREFIX [--nlow 192] [--nhigh 384] [--workers 6] [--chunks 48]
            [--sample-every 16]
      Both resolutions on the same panels (n and 2n Gauss-Legendre nodes per panel), in worker
      processes. For every --sample-every-th panel it also keeps, at both node sets, x, f_sin,
      f_cos, theta_mod_2pi, theta (raw) and theta_deriv (OUT_PREFIX.sample.npz), for the other
      tree's comparison, and theta_mod_2pi at the panel's two ends, for the consistency check of
      theta_deriv against theta. Writes OUT_PREFIX.ref.json.

  compare PICKLE SAMPLE_NPZ OUT_JSON [--workers 6]
      In the other tree: evaluates its own captured callables at every sampled node and compares
      with the stored values bit for bit (float64 bit patterns).

Accuracy: S_low and S_high are the two sums; |S_high - S_low| is the reference's convergence
estimate, reported against the smaller of the two codes' declared abserr. Also reported: the
linear worst-case bound on the arithmetic rounding of the products, eps * sum|terms| (pessimistic).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from yprov import header, which_tree  # noqa: E402

EPS = 2.0**-52
MAX_SAMPLED_PANELS = 3000


def hx(x):
    return None if x is None else float(x).hex()


# ---------------------------------------------------------------- exact summation (Shewchuk)
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


def load(pickle_path):
    import ray.cloudpickle as cloudpickle

    blob = Path(pickle_path).read_bytes()
    return cloudpickle.loads(blob), hashlib.sha256(blob).hexdigest()


def replay(rec):
    import AdaptiveLevin.levin_quadrature as LQ

    rr = LQ.adaptive_levin_sincos(rec["x_span"], rec["f"], theta=rec["theta"], **rec["kw"])
    scale = float.fromhex(rec["scale"])
    return {
        "value": hx(rr["value"]),
        "abserr": hx(rr["abserr"]),
        "equals_capture": (hx(rr["value"]), hx(rr["abserr"])) == (rec["value"], rec["abserr"]),
        "equals_store": (hx(scale * rr["value"]), hx(scale * rr["abserr"]))
        == (rec["store_scaled_value"], rec["store_scaled_abserr"]),
        "regions_depth_equal": [int(rr["num_regions"]), int(rr["max_depth"])] == [rec["num_regions"], rec["max_depth"]],
    }


# ---------------------------------------------------------------- panels
def cmd_panels(a):
    import numpy as np

    rec, sha = load(a.pickle)
    rp = replay(rec)
    if not (rp["equals_capture"] and rp["equals_store"]):
        raise SystemExit(f"replay does not reproduce: {rp}")
    x0, x1 = rec["x_span"]
    thd = rec["theta"]["theta_deriv"]
    ncell = max(1, int(math.ceil((x1 - x0) / a.cell)))
    edges = np.linspace(x0, x1, ncell + 1)
    edges[0], edges[-1] = x0, x1
    tp = np.array([thd(float(x)) for x in edges])
    # zeros of theta': sign changes between cell edges, bisected to adjacent doubles
    zeros = []
    for i in np.nonzero(np.signbit(tp[1:]) != np.signbit(tp[:-1]))[0]:
        lo, hi = float(edges[i]), float(edges[i + 1])
        slo = math.copysign(1.0, thd(lo))
        while True:
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if math.copysign(1.0, thd(mid)) == slo:
                lo = mid
            else:
                hi = mid
        h = 1e-6
        xm = 0.5 * (lo + hi)
        tpp = (thd(xm + h) - thd(xm - h)) / (2 * h)
        zeros.append({"x": xm, "x_hex": hx(xm), "z": math.expm1(xm), "theta_deriv_lo": thd(lo), "theta_deriv_hi": thd(hi),
                      "theta_second_deriv": tpp,
                      "stationary_width_x": 1.0 / math.sqrt(abs(tpp)) if tpp != 0 else None})
    atp = np.abs(tp)
    phi = 0.5 * (atp[1:] + atp[:-1]) * np.diff(edges)
    width = np.diff(edges)
    # the work coordinate W = Phi / D + x / hmax, piecewise linear between cell edges; panels are
    # equal steps of W, at most one unit each, so each spans at most about D radians and hmax in x
    W = np.concatenate([[0.0], np.cumsum(phi / a.D + width / a.hmax)])
    P = int(math.ceil(W[-1]))
    pe = np.interp(np.linspace(0.0, W[-1], P + 1), W, edges)
    pe[0], pe[-1] = x0, x1
    assert pe[0] == x0 and pe[-1] == x1 and np.all(np.diff(pe) > 0)
    np.save(a.out_prefix + ".edges.npy", pe)
    info = {
        "pickle": a.pickle, "pickle_sha256": sha, "row": rec["row"], "index": rec["index"], "label": rec["label"],
        "tree": which_tree(), "replay": rp, "x_span": [hx(x0), hx(x1)], "D": a.D, "hmax": a.hmax, "cell": a.cell,
        "ncell": ncell, "npanels": int(len(pe) - 1), "phase_variation_rad": float(phi.sum()),
        "oscillations_total_variation": float(phi.sum() / (2 * math.pi)),
        "theta_deriv_max_abs": float(atp.max()), "theta_deriv_at_ends": [float(tp[0]), float(tp[-1])],
        "zeros_of_theta_deriv": zeros, "edges_sha256": hashlib.sha256(pe.tobytes()).hexdigest(),
    }
    Path(a.out_prefix + ".panels.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("replay",)}, indent=1))
    print("replay", rp)


# ---------------------------------------------------------------- integrate
_W = {}


def _winit(pickle_path, nlow, nhigh):
    from scipy.special import roots_legendre

    rec, _ = load(pickle_path)
    _W["rec"] = rec
    _W["gl"] = {n: roots_legendre(n) for n in (nlow, nhigh)}


def _work(args):
    """Panels [i0, i1) of edges: returns the exact partials and sums of |terms| at each n, and
    the sampled panels' node values."""
    import numpy as np

    edges, i0, i1, nlow, nhigh, every = args
    rec = _W["rec"]
    fs, fc = rec["f"]
    th = rec["theta"]
    thm, thr, thd = th["theta_mod_2pi"], th["theta"], th["theta_deriv"]
    out = {}
    samples = []
    t0 = time.perf_counter()
    for n in (nlow, nhigh):
        t, w = _W["gl"][n]
        parts = []
        absum = 0.0
        nodes = 0
        for p in range(i0, i1):
            a_, b_ = float(edges[p]), float(edges[p + 1])
            c = 0.5 * (a_ + b_)
            hh = 0.5 * (b_ - a_)
            keep = every > 0 and p % every == 0
            if keep:
                rows = []
            for k in range(n):
                x = c + hh * float(t[k])
                s_, c_, m_ = fs(x), fc(x), thm(x)
                term = (float(w[k]) * hh) * (s_ * math.sin(m_) + c_ * math.cos(m_))
                msum_add(parts, term)
                absum += abs(term)
                if keep:
                    rows.append((x, s_, c_, m_, thr(x), thd(x)))
            if keep:
                rows = np.array(rows)
            nodes += n
            if keep:
                samples.append({"panel": p, "n": n, "a": a_, "b": b_, "thm_a": thm(a_), "thm_b": thm(b_),
                                "w": w, "rows": rows})
        out[n] = {"partials": parts, "absum": absum, "nodes": nodes}
    out["seconds"] = time.perf_counter() - t0
    out["samples"] = samples
    out["range"] = (i0, i1)
    return out


def cmd_integrate(a):
    import concurrent.futures as cf

    import numpy as np

    rec, sha = load(a.pickle)
    t0 = time.perf_counter()
    rp = replay(rec)
    t_replay = time.perf_counter() - t0
    if not (rp["equals_capture"] and rp["equals_store"]):
        raise SystemExit(f"replay does not reproduce: {rp}")
    edges = np.load(a.out_prefix + ".edges.npy")
    pinfo = json.loads(Path(a.out_prefix + ".panels.json").read_text())
    if pinfo["edges_sha256"] != hashlib.sha256(edges.tobytes()).hexdigest():
        raise SystemExit("edges file differs from its record")
    x0, x1 = rec["x_span"]
    if (edges[0], edges[-1]) != (x0, x1):
        raise SystemExit("edges do not span the call's x_span")
    npan = len(edges) - 1
    bounds = np.linspace(0, npan, min(a.chunks, npan) + 1).astype(int)
    # the sample: every --sample-every-th panel, thinned to at most MAX_SAMPLED_PANELS panels
    every = max(a.sample_every, int(math.ceil(npan / MAX_SAMPLED_PANELS)))
    jobs = [(edges, int(bounds[i]), int(bounds[i + 1]), a.nlow, a.nhigh, every) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]
    t1 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_winit, initargs=(a.pickle, a.nlow, a.nhigh)) as ex:
        res = list(ex.map(_work, jobs))
    wall = time.perf_counter() - t1
    S, absum, nodes = {}, {}, {}
    for n in (a.nlow, a.nhigh):
        allp = []
        for r in res:
            allp.extend(r[n]["partials"])
        S[n] = math.fsum(allp)
        absum[n] = math.fsum(r[n]["absum"] for r in res)
        nodes[n] = sum(r[n]["nodes"] for r in res)
    samples = [s for r in res for s in r["samples"]]
    # the sample, for the other tree's comparison
    allrows = np.concatenate([s["rows"] for s in samples]) if samples else np.zeros((0, 6))
    xs = allrows[:, 0]
    cols = allrows[:, 1:]
    np.savez(a.out_prefix + ".sample.npz", x=xs, f_sin=cols[:, 0], f_cos=cols[:, 1], theta_mod_2pi=cols[:, 2],
             theta=cols[:, 3], theta_deriv=cols[:, 4],
             ends=np.array([[s["a"], s["b"], s["thm_a"], s["thm_b"]] for s in samples if s["n"] == a.nlow]))
    # theta_deriv against theta, on the sampled panels (low-n rule): the phase change across the
    # panel from theta_mod_2pi at its ends, wrapped, against the Gauss-Legendre integral of theta_deriv
    mism = []
    for s in samples:
        if s["n"] != a.nlow:
            continue
        hh = 0.5 * (s["b"] - s["a"])
        integ = math.fsum(float(wk) * hh * float(row[5]) for wk, row in zip(s["w"], s["rows"]))
        d = (s["thm_b"] - s["thm_a"]) - integ
        d = math.remainder(d, 2 * math.pi)
        mism.append((d, integ))
    v18 = float.fromhex(rec["value"])
    e18 = float.fromhex(rec["abserr"])
    nl, nh = a.nlow, a.nhigh
    doc = {
        "pickle": a.pickle, "pickle_sha256": sha, "row": rec["row"], "index": rec["index"], "label": rec["label"],
        "tree": which_tree(), "replay": rp, "replay_seconds": t_replay, "x_span": [hx(x0), hx(x1)],
        "npanels": npan, "nlow": nl, "nhigh": nh, "nodes": {str(k): v for k, v in nodes.items()},
        "S_low": hx(S[nl]), "S_high": hx(S[nh]), "ref": hx(S[nh]), "ref_repr": repr(S[nh]),
        "conv_diff": hx(abs(S[nh] - S[nl])), "absum_low": hx(absum[nl]), "absum_high": hx(absum[nh]),
        "rounding_bound_linear": hx(4 * EPS * absum[nh]),
        "cancellation": absum[nh] / abs(S[nh]) if S[nh] != 0 else None,
        "wall_seconds": wall, "cpu_seconds": sum(r["seconds"] for r in res), "workers": a.workers, "chunks": len(jobs),
        "sample_every": every, "sample_nodes": int(len(xs)),
        "sample_sha256": hashlib.sha256(Path(a.out_prefix + ".sample.npz").read_bytes()).hexdigest(),
        "theta_deriv_consistency": {
            "panels": len(mism),
            "max_abs_mismatch_rad": max((abs(d) for d, _ in mism), default=None),
            "mean_signed_mismatch_rad": (math.fsum(d for d, _ in mism) / len(mism)) if mism else None,
            "max_abs_mismatch_per_rad_of_phase": max((abs(d) / abs(i) for d, i in mism if i != 0), default=None),
        },
        "this_tree_value": rec["value"], "this_tree_abserr": rec["abserr"],
        "this_tree_err_over_abserr": abs(v18 - S[nh]) / e18 if e18 > 0 else None,
    }
    Path(a.out_prefix + ".ref.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


# ---------------------------------------------------------------- compare
def _cwork(args):
    import numpy as np

    xs = args
    rec = _W["rec"]
    fs, fc = rec["f"]
    th = rec["theta"]
    return np.array([(fs(x), fc(x), th["theta_mod_2pi"](x), th["theta"](x), th["theta_deriv"](x)) for x in map(float, xs)])


def _cinit(pickle_path):
    rec, _ = load(pickle_path)
    _W["rec"] = rec


def cmd_compare(a):
    import concurrent.futures as cf

    import numpy as np

    rec, sha = load(a.pickle)
    rp = replay(rec)
    if not (rp["equals_capture"] and rp["equals_store"]):
        raise SystemExit(f"replay does not reproduce: {rp}")
    smp = np.load(a.sample_npz)
    xs = smp["x"]
    parts = np.array_split(xs, max(1, min(len(xs) // 2000, 600)))
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=a.workers, initializer=_cinit, initargs=(a.pickle,)) as ex:
        got = np.concatenate(list(ex.map(_cwork, parts)))
    names = ["f_sin", "f_cos", "theta_mod_2pi", "theta", "theta_deriv"]
    res = {}
    for j, nm in enumerate(names):
        want = np.ascontiguousarray(smp[nm])
        have = np.ascontiguousarray(got[:, j])
        diff = int(np.count_nonzero(want.view(np.uint64) != have.view(np.uint64)))
        res[nm] = {"nodes": int(len(want)), "bit_differences": diff}
    ends = smp["ends"]
    thm = rec["theta"]["theta_mod_2pi"]
    end_diff = sum(int(float(thm(float(r[0]))).hex() != float(r[2]).hex()) + int(float(thm(float(r[1]))).hex() != float(r[3]).hex())
                   for r in ends)
    doc = {"pickle": a.pickle, "pickle_sha256": sha, "tree": which_tree(), "replay": rp, "sample_npz": a.sample_npz,
           "sample_sha256": hashlib.sha256(Path(a.sample_npz).read_bytes()).hexdigest(), "columns": res,
           "panel_end_theta_mod_2pi": {"values": int(2 * len(ends)), "bit_differences": end_diff},
           "all_bit_identical": all(v["bit_differences"] == 0 for v in res.values()) and end_diff == 0,
           "seconds": time.perf_counter() - t0}
    Path(a.out_json).write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("panels")
    p.add_argument("pickle")
    p.add_argument("out_prefix")
    p.add_argument("--D", type=float, default=512.0)
    p.add_argument("--hmax", type=float, default=0.005)
    p.add_argument("--cell", type=float, default=1e-3)
    p = sub.add_parser("integrate")
    p.add_argument("pickle")
    p.add_argument("out_prefix")
    p.add_argument("--nlow", type=int, default=192)
    p.add_argument("--nhigh", type=int, default=384)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--chunks", type=int, default=48)
    p.add_argument("--sample-every", type=int, default=16)
    p = sub.add_parser("compare")
    p.add_argument("pickle")
    p.add_argument("sample_npz")
    p.add_argument("out_json")
    p.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd in ("integrate", "compare") and a.workers > 6:
        raise SystemExit("at most six workers")
    tree = which_tree()
    if tree is None:
        raise SystemExit("PYTHONPATH must name one of the two trees")
    for l in header(Path(__file__).resolve(), f"Y1 references: refquad {a.cmd} in tree {tree}", [Path(a.pickle)]):
        print(l, flush=True)
    {"panels": cmd_panels, "integrate": cmd_integrate, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
