"""levin-phase-floor prompt 06b section 5, W2: the reference for a captured call. New in 06b (prompt
06b section 5: "Change only: ...; the reference for W2").

For each entry of SELECTION_JSON (a list of {"unit", "index", "dir", "pickle", "offset", "length",
"sha256", ...}; any other keys are carried into the output), in its own worker process, at most
six at once (``--workers``, default 6), with ``PYTHONPATH`` the tree b:

1. **Reload** the call's record that worker.py pickled after the census's timed section (its span,
   its integrand ``f = [f_sin, f_cos]``, its phase dict ``theta`` and its keyword arguments, as the
   call saw them), checking the blob's SHA-256.
2. **Re-run the call** with ``adaptive_levin_sincos(x_span, f, theta=theta, **kw)`` and compare its
   value and abserr, by bit pattern, with the census's: the check that the reloaded inputs are the
   call's.
3. **The reference** (prompt 06b section 5, W2): "Levin forced to 2^9 and to 2^10 equal pieces, with
   each piece's own floor reported." The span is cut into 2^D equal pieces by D rounds of the
   driver's own bisection (each piece (s, e) halved at ``(e + s) / 2.0``, ``_levin_interval
   .break_point``), so that every piece is a region the adaptive driver could itself reach at depth
   D. Each piece is evaluated once by the module's ``_adaptive_levin_subregion``, as the driver
   evaluates a region (with the call's ``chebyshev_order`` and ``rtol``; a piece whose phase span is
   below the Levin gate takes the nested Clenshaw-Curtis branch there, as in the driver, and the
   count is reported). The piece values are summed with ``math.fsum``; each piece's ``phase_err``
   (its own floor) is kept, and their sum and maximum reported.
4. ``agree``: |S_9 - S_10| <= abserr_b / 100; ``ref`` = S_10; ``err_b`` = |value_b - ref|;
   ``holds`` = err_b <= abserr_b. If the entry carries 06a's ``value_06a`` and ``abserr_06a`` (hex),
   ``err_06a`` = |value_06a - ref| is reported against them (descriptive).

Writes OUT_JSON (every number also as hex) and prints a table. Read-only outside OUT_JSON.

Usage: PYTHONPATH=<tree b> python -B w2_reference.py SELECTION_JSON OUT_JSON [--workers N] [--depths 9,10]
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import pickle
import socket
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import TREES, header, sha256, tree_state  # noqa: E402


def _hx(x):
    return None if x is None else float(x).hex()


def pieces(span, depth):
    out = [tuple(span)]
    for _ in range(depth):
        nxt = []
        for s, e in out:
            c = (e + s) / 2.0
            nxt.extend([(s, c), (c, e)])
        out = nxt
    return out


def one(entry, depths):
    import AdaptiveLevin.levin_quadrature as LQ

    t0 = time.perf_counter()
    res = {k: v for k, v in entry.items()}
    res["levin_quadrature_file"] = LQ.__file__
    res["pid"] = os.getpid()
    try:
        with open(Path(entry["dir"]) / entry["pickle"], "rb") as f:
            f.seek(entry["offset"])
            blob = f.read(entry["length"])
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise RuntimeError("the input record's sha256 differs")
        rec = pickle.loads(blob)
        if rec["unit"] != entry["unit"] or rec["index"] != entry["index"]:
            raise RuntimeError(f"record is {rec['unit']} {rec['index']}")
        x_span = tuple(rec["x_span"])
        f_ = rec["f"]
        theta = rec["theta"]
        kw = dict(rec["kw"])
        res["notify_label"] = rec["notify_label"]
        res["x_span"] = [_hx(x_span[0]), _hx(x_span[1])]
        res["kw"] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else repr(v)) for k, v in kw.items()}
        res["value_b"] = rec["value"]
        res["abserr_b"] = rec["abserr"]
        # 2. the call re-run from the reloaded inputs
        rr = LQ.adaptive_levin_sincos(x_span, f_, theta=theta, **kw)
        res["rerun_value"] = _hx(rr["value"])
        res["rerun_abserr"] = _hx(rr["abserr"])
        res["rerun_equals_census"] = (_hx(rr["value"]), _hx(rr["abserr"])) == (rec["value"], rec["abserr"])
        # 3. the reference
        basis = LQ._Basis_SinCos(theta)
        order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
        rtol = kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL)
        dpath = kw.get("diagnostics_path")
        dpath = Path(dpath) if dpath is not None else LQ.DEFAULT_LEVIN_DIAGNOSTICS_PATH
        vcache = {}
        sums = {}
        for D in depths:
            vals, floors, n_direct = [], [], 0
            for p in pieces(x_span, D):
                d = LQ._adaptive_levin_subregion(
                    p,
                    f_,
                    basis,
                    id_label=LQ._LazyUUID(),
                    chebyshev_order=order,
                    rtol=rtol,
                    notify_label=None,
                    build_p_sample=False,
                    diagnostics_path=dpath,
                    vectorize_cache=vcache,
                )
                vals.append(float(d["value"]))
                floors.append(float(d.get("phase_err", 0.0) or 0.0))
                n_direct += bool(d.get("is_direct", False))
            S = math.fsum(vals)
            sums[D] = S
            res[f"S{D}"] = _hx(S)
            res[f"S{D}_repr"] = repr(S)
            res[f"floor_sum_{D}"] = _hx(math.fsum(floors))
            res[f"floor_max_{D}"] = _hx(max(floors))
            res[f"n_pieces_{D}"] = len(vals)
            res[f"n_direct_{D}"] = n_direct
        Dlo, Dhi = depths[0], depths[-1]
        vb = float.fromhex(rec["value"])
        eb = float.fromhex(rec["abserr"])
        ref = sums[Dhi]
        res["ref"] = _hx(ref)
        res["ref_repr"] = repr(ref)
        res["S_diff"] = _hx(abs(sums[Dlo] - sums[Dhi]))
        res["agree"] = bool(abs(sums[Dlo] - sums[Dhi]) <= eb / 100.0)
        res["err_b"] = _hx(abs(vb - ref))
        res["err_b_over_abserr_b"] = abs(vb - ref) / eb if eb > 0 else None
        res["holds"] = bool(abs(vb - ref) <= eb)
        if entry.get("value_06a") is not None:
            va = float.fromhex(entry["value_06a"])
            ea = float.fromhex(entry["abserr_06a"])
            res["err_06a"] = _hx(abs(va - ref))
            res["err_06a_over_abserr_06a"] = abs(va - ref) / ea if ea > 0 else None
            res["within_06a"] = bool(abs(va - ref) <= ea)
        res["error"] = None
    except Exception:
        res["error"] = traceback.format_exc()
    res["seconds"] = time.perf_counter() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--depths", default="9,10")
    args = ap.parse_args()
    if args.workers > 6:
        raise SystemExit("at most six workers")
    depths = [int(x) for x in args.depths.split(",")]
    sel = json.loads(Path(args.selection).read_text())
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06b: W2's reference", [Path(args.selection)])
    import AdaptiveLevin.levin_quadrature as LQ

    tree = next((n for n, (t, _s) in TREES.items() if LQ.__file__.startswith(str(t) + os.sep)), None)
    for l in hdr:
        print(l)
    print(f"# AdaptiveLevin.levin_quadrature from {LQ.__file__} (tree {tree}); host {socket.gethostname()}")
    print(f"# depths {depths}; {len(sel)} calls; {args.workers} worker processes")
    if tree != "b":
        raise SystemExit("PYTHONPATH must be the tree b")
    out = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, e, depths) for e in sel]
        for fu in futs:
            out.append(fu.result())
    doc = {"header": hdr, "depths": depths, "tree": tree, "tree_state": tree_state(TREES[tree][0]), "results": out}
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(f"\n{'unit':<18} {'idx':>3} {'rerun=':>6} {'agree':>5} {'holds':>5} {'err_b/abserr_b':>14} {'|S9-S10|/abserr_b':>18} "
          f"{'floor_sum_10':>12} {'direct10':>8} {'err06a/abserr06a':>16} {'s':>6}")
    for r in out:
        if r.get("error"):
            print(f"{r['unit']:<18} {r['index']:>3} ERROR {r['error'].splitlines()[-1]}")
            continue
        eb = float.fromhex(r["abserr_b"])
        print(
            f"{r['unit']:<18} {r['index']:>3} {str(r['rerun_equals_census']):>6} {str(r['agree']):>5} {str(r['holds']):>5} "
            f"{r['err_b_over_abserr_b']:>14.4g} {float.fromhex(r['S_diff']) / eb:>18.4g} "
            f"{float.fromhex(r['floor_sum_10']):>12.3g} {r['n_direct_10']:>8} "
            f"{(r.get('err_06a_over_abserr_06a') if r.get('err_06a_over_abserr_06a') is not None else float('nan')):>16.4g} {r['seconds']:>6.1f}"
        )
    print(f"\nwrote {args.out} sha256 {sha256(args.out)}")


if __name__ == "__main__":
    main()
