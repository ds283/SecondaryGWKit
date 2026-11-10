"""levin-phase-floor prompt 06j, Part W: everything for the W calls that needs no reference reading. New in 06j;
definitions are state/predictions_and_rules_20261103T0905Z.md sections 4 and 5 (written before this ran).

For each W call (W-5 f00..f09, W-12 c00..c11), from its capture (state/inputs/<sid>.pkl, ray.cloudpickle, this
directory on PYTHONPATH after the tree) and both censuses (read only):
- **route and segments:** ``tools.filon_reference.route_of`` and ``production_edges``; the seams with their kinds
  (seams_06j.py);
- **phase span:** per segment theta (raw) at its two edges, and theta_deriv at 17 points across it (a sign change
  flags a possible stationary point, which the tool would integrate directly); the net phase change
  |theta(b) - theta(a)| and the sum over segments of |theta(X_r) - theta(X_l)| (the total variation where every
  segment is monotone). This is for the plan's cost projection only; the tool's own monotonicity test is made
  when it reads;
- **W-5's rtol equality:** for each W-5 row, compare13.describe (06g's criterion, unchanged) on the 1e-7 and
  1e-8 captures; every differing field is listed;
- **06g's returned leaves** (its census region pickle: start, end, depth, value, value_own, value_kind,
  convergence_ratio, trusted) with the seam test; the exact sum of the returned values against the census value;
- **06h's returned leaves** (its census region pickle: start, end, depth, convergence_ratio, trusted) with the
  seam test; their values recomputed by solving each 06h leaf's span with the tree's ``_adaptive_levin_subregion``
  (06h's value path, unchanged by 06g: git diff 4f4f9ed 1c5336e) with the call's arguments, accepted only if
  their exact sum (math.fsum, as _ExactSum.total) equals 06h's recorded value bit for bit;
- **the constant move:** the common refinement of the two partitions; for each cell of the coarser partition
  (each leaf of either code that contains leaves of the other), each code's partial sum on it, and the share
  of |v_g - v_06h| each such cell carries.

Writes out/w_prep/<sid>.json and out/w_prep/summary.json. Usage:
PYTHONPATH=<tree>:<this directory> python -B w_prep.py [OUT_SUBDIR]

Changed after run 1 (marked "06j v2"; run 1's text is state/w_prep_as_run1.py): the codes return the plain
floating-point sum of their returned leaves' values in ascending x (_sum_leaves_ascending, at 06h's tree and at
06g's), not their exact sum; each code's sums are now checked both ways, and 06h's recomputed values are accepted
when the code's own arithmetic (the ascending plain sum) reproduces 06h's value bit for bit. OUT_SUBDIR (default
w_prep) names the output directory under out/.
"""

import hashlib
import json
import math
import pickle
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
import seams_06j as SM  # noqa: E402
from prov import header  # noqa: E402

SCAN_POINTS = 17


def pickle_record(run, ent):
    name, off, ln, sha = ent
    with open(run / name, "rb") as f:
        f.seek(off)
        blob = f.read(ln)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit(f"{name}@{off}: sha256 differs")
    return pickle.loads(blob)


def regions_of(prec, index):
    hits = [x for x in prec["regions"] if x["index"] == index]
    if len(hits) != 1:
        raise SystemExit(f"{len(hits)} region entries for call {index}")
    return hits[0]


def fx(v):
    return None if v is None else float(v).hex()


def leaves_table(reg, keys):
    n = len(reg["start"])
    out = []
    for i in range(n):
        row = {}
        for k in keys:
            if k not in reg:
                continue
            v = reg[k][i]
            row[k] = float(v) if np.asarray(reg[k]).dtype.kind == "f" else int(v)
        out.append(row)
    return out


def main():
    import ray
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ
    import tools.filon_reference as F
    import compare13 as C13

    t0 = time.perf_counter()
    sel = {e["sid"]: e for e in json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())}
    wsam = json.loads((RC.ROOT / "out" / "sample" / "w_sample.json").read_text())
    sids = [x["sid"] for x in wsam["w5"]] + [x["sid"] for x in wsam["w12"]]
    outd = RC.ROOT / "out" / (sys.argv[1] if len(sys.argv) > 1 else "w_prep")  # 06j v2
    outd.mkdir(parents=True, exist_ok=False)  # 06j v2: never overwrite
    run06g = RC.census_run_06g()
    if RC.sha256(RC.RUN06H / "checkpoint.jsonl") != RC.CHECKPOINT06H_SHA256:
        raise SystemExit("06h's checkpoint is not 3104c18e...")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: Part W's preparation (no reading)",
                 [RC.ROOT / "out" / "sample" / "selection.json", RC.ROOT / "out" / "sample" / "w_sample.json",
                  Path(LQ.__file__), Path(F.__file__), HERE / "seams_06j.py", HERE / "compare13.py"])
    for l in hdr:
        print(l)
    calls = {}
    summary = {"header": hdr, "calls": {}, "w5_equality": {}}
    for sid in sids:
        e = sel[sid]
        pkl = RC.ROOT / "state" / "inputs" / f"{sid}.pkl"
        if not pkl.exists():
            print(f"{sid}: no capture file; skipped")
            summary["calls"][sid] = {"skipped": "no capture file"}
            continue
        t1 = time.perf_counter()
        rec = cp.loads(pkl.read_bytes())
        calls[sid] = rec
        if rec["unit"] != e["unit"] or rec["index"] != e["index"]:
            raise SystemExit(f"{sid}: capture is not the selection's call")
        a0, b0 = (float(v) for v in rec["x_span"])
        route = F.route_of(rec)
        doc = {"sid": sid, "unit": e["unit"], "index": e["index"], "route": route, "x_span": [a0, b0],
               "pickle": str(pkl), "pickle_sha256": RC.sha256(pkl)}
        th = rec["theta"]
        sm_x = np.array([])
        if route == "B":
            edges, n_tables = F.production_edges(rec)
            sm, nt2 = SM.seams(rec)
            sm_x = np.array(sorted({s["x"] for s in sm}))
            by_kind = {}
            for s in sm:
                by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
            thv = [float(th["theta"](x)) for x in edges]
            seg_var = [abs(thv[i + 1] - thv[i]) for i in range(len(edges) - 1)]
            n_sign = 0
            flagged = []
            for i in range(len(edges) - 1):
                xl, xr = edges[i], edges[i + 1]
                ds = [float(th["theta_deriv"](xl + (xr - xl) * j / (SCAN_POINTS - 1))) for j in range(SCAN_POINTS)]
                sg = {d > 0 for d in ds} | ({0} if any(d == 0 for d in ds) else set())
                if len(sg) > 1:
                    n_sign += 1
                    flagged.append([xl, xr])
            doc.update(n_segments=len(edges) - 1, n_tables=n_tables, seams_by_kind=by_kind,
                       seams_merge_check=SM.check_against_tool(rec, sm, nt2),
                       phase_net=abs(thv[-1] - thv[0]), phase_sum_segments=float(sum(seg_var)),
                       phase_max_segment=float(max(seg_var)), n_segments_theta_deriv_sign_change=n_sign,
                       segments_flagged=flagged[:50])
        # 06g's returned leaves
        g = e["g_record"]
        regg = regions_of(pickle_record(run06g, e["g_pickle"]), e["index"])
        gl = leaves_table(regg, ("start", "end", "depth", "value", "value_own", "value_kind", "convergence_ratio",
                                  "trusted", "delta1", "phase_err", "abserr", "type"))
        for L in gl:
            hs = SM.held(sm_x, L["start"], L["end"]) if len(sm_x) else []
            L["holds_seam"] = bool(hs)
            L["n_seams"] = len(hs)
        sum_g = math.fsum(L["value"] for L in gl)
        asc_g = 0.0  # 06j v2: the code's own arithmetic
        for L in gl:
            asc_g = asc_g + L["value"]
        doc["g"] = {"value_hex": g["value"]["hex"], "abserr_hex": g["abserr"]["hex"], "num_regions": g["num_regions"],
                    "max_depth": g["max_depth"], "converged": g["converged"], "leaves": gl,
                    "fsum_of_returned_values_hex": fx(sum_g), "fsum_equals_value": fx(sum_g) == g["value"]["hex"],
                    "ascending_sum_hex": fx(asc_g), "ascending_sum_equals_value": fx(asc_g) == g["value"]["hex"]}
        # 06h's returned leaves, values recomputed
        h = e["06h_record"]
        regh = regions_of(pickle_record(RC.RUN06H, e["06h_pickle"]), e["index"])
        hl = leaves_table(regh, ("start", "end", "depth", "convergence_ratio", "trusted", "delta1", "phase_err",
                                  "abserr", "type"))
        A = LQ._Basis_SinCos(th)
        cache = {}
        kw = rec["kw"]
        for L in hl:
            d = LQ._adaptive_levin_subregion((L["start"], L["end"]), rec["f"], A, id_label="06j-w-prep",
                                             chebyshev_order=kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER),
                                             rtol=kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL),
                                             notify_label=kw.get("notify_label"), build_p_sample=False,
                                             diagnostics_path=RC.ROOT / "state" / "levin_diagnostics",
                                             vectorize_cache=cache)
            L["value_recomputed"] = float(d["value"])
            L["phase_err_recomputed_equal"] = fx(d.get("phase_err")) == fx(L["phase_err"])
            L["type_recomputed"] = 1 if d.get("is_direct") else 0
            hs = SM.held(sm_x, L["start"], L["end"]) if len(sm_x) else []
            L["holds_seam"] = bool(hs)
            L["n_seams"] = len(hs)
        sum_h = math.fsum(L["value_recomputed"] for L in hl)
        asc_h = 0.0  # 06j v2: the code's own arithmetic
        for L in hl:
            asc_h = asc_h + L["value_recomputed"]
        h_ok = fx(asc_h) == h["value"]["hex"]
        doc["h06"] = {"value_hex": h["value"]["hex"], "abserr_hex": h["abserr"]["hex"], "num_regions": h["num_regions"],
                      "max_depth": h["max_depth"], "converged": h["converged"], "leaves": hl,
                      "fsum_of_recomputed_values_hex": fx(sum_h), "fsum_equals_value": fx(sum_h) == h["value"]["hex"],
                      "ascending_sum_hex": fx(asc_h), "recomputed_sum_equals_value": h_ok,
                      "phase_err_all_equal": all(L["phase_err_recomputed_equal"] for L in hl)}
        # the constant move: common refinement
        vg, vh = float.fromhex(g["value"]["hex"]), float.fromhex(h["value"]["hex"])
        dv = Fraction(vg) - Fraction(vh)
        cells = []
        for L in hl:
            inside = [G_ for G_ in gl if G_["start"] >= L["start"] and G_["end"] <= L["end"]]
            if inside and abs(sum(G_["end"] - G_["start"] for G_ in inside) - (L["end"] - L["start"])) <= 1e-12 * (L["end"] - L["start"]):
                sg = math.fsum(G_["value"] for G_ in inside)
                cells.append({"coarse": "06h", "start": L["start"], "end": L["end"], "v_06h": L["value_recomputed"],
                              "v_g_sum": sg, "n_g_leaves": len(inside), "diff": sg - L["value_recomputed"]})
        for G_ in gl:
            inside = [L for L in hl if L["start"] >= G_["start"] and L["end"] <= G_["end"]]
            if len(inside) > 1 and abs(sum(L["end"] - L["start"] for L in inside) - (G_["end"] - G_["start"])) <= 1e-12 * (G_["end"] - G_["start"]):
                sh = math.fsum(L["value_recomputed"] for L in inside)
                cells.append({"coarse": "06g", "start": G_["start"], "end": G_["end"], "v_g": G_["value"],
                              "v_06h_sum": sh, "n_06h_leaves": len(inside), "diff": G_["value"] - sh})
        tot = float(dv)
        for c in cells:
            c["share_of_move"] = c["diff"] / tot if tot != 0 else None
        covered = sum(c["end"] - c["start"] for c in cells)
        doc["constant_move"] = {"v_g_minus_v_06h": tot, "cells": cells, "covered_width": covered,
                                "span_width": b0 - a0, "sum_of_cell_diffs": math.fsum(c["diff"] for c in cells),
                                "localised_90pct": any(c["share_of_move"] is not None and abs(c["share_of_move"]) >= 0.9 for c in cells)
                                if h_ok else None}
        doc["seconds"] = time.perf_counter() - t1
        (outd / f"{sid}.json").write_text(json.dumps(doc, indent=1))
        summary["calls"][sid] = {k: doc.get(k) for k in ("unit", "index", "route", "n_segments", "phase_net", "phase_sum_segments",
                                                          "phase_max_segment", "n_segments_theta_deriv_sign_change", "seconds")}
        summary["calls"][sid].update(g_fsum_ok=doc["g"]["fsum_equals_value"], g_ascending_ok=doc["g"]["ascending_sum_equals_value"], h06_recomputed_ok=h_ok,
                                     localised=doc["constant_move"]["localised_90pct"])
        print(f"{sid} {e['unit']} #{e['index']} route {route}: segments {doc.get('n_segments')} phase net {doc.get('phase_net', float('nan')):.4g} "
              f"sum {doc.get('phase_sum_segments', float('nan')):.4g} sign-change segments {doc.get('n_segments_theta_deriv_sign_change')}; "
              f"g fsum ok {doc['g']['fsum_equals_value']} ascending ok {doc['g']['ascending_sum_equals_value']}; 06h recomputed ok {h_ok}; move {tot:.4e} localised "
              f"{doc['constant_move']['localised_90pct']}; {doc['seconds']:.1f} s", flush=True)
    # W-5 rtol equality
    for x in wsam["w5"]:
        if x["rtol"] != "1e-7":
            continue
        twin = [y for y in wsam["w5"] if y["rtol"] == "1e-8" and y["row"] == x["row"]]
        if not twin or x["sid"] not in calls or twin[0]["sid"] not in calls:
            summary["w5_equality"][x["row"]] = {"compared": False}
            continue
        y = twin[0]
        D1 = C13.describe(calls[x["sid"]], F)
        D2 = C13.describe(calls[y["sid"]], F)
        res = {}
        for i, nm in enumerate(("criterion", "integrand", "descriptive")):
            diffs = []

            def walk(p, q, path=""):
                if isinstance(p, dict) and isinstance(q, dict):
                    for k in sorted(set(p) | set(q)):
                        if k not in p or k not in q:
                            diffs.append(f"{path}/{k}: in one only")
                        else:
                            walk(p[k], q[k], f"{path}/{k}")
                elif isinstance(p, list) and isinstance(q, list):
                    if len(p) != len(q):
                        diffs.append(f"{path}: lengths {len(p)} vs {len(q)}")
                    else:
                        for j, (u, v) in enumerate(zip(p, q)):
                            walk(u, v, f"{path}[{j}]")
                elif p != q:
                    diffs.append(f"{path}: {str(p)[:60]} vs {str(q)[:60]}")

            walk(D1[i], D2[i])
            res[nm] = {"equal": not diffs, "differences": diffs[:40]}
        res["pair"] = [x["sid"], y["sid"]]
        res["one_reference_serves_both"] = res["criterion"]["equal"] and res["integrand"]["equal"]
        summary["w5_equality"][x["row"]] = res
        print(f"W-5 row {x['row']}: {x['sid']} vs {y['sid']}: criterion {res['criterion']['equal']}, integrand "
              f"{res['integrand']['equal']}, descriptive {res['descriptive']['equal']} {res['descriptive']['differences'][:3]}")
    summary["seconds"] = time.perf_counter() - t0
    summary["ray_initialized"] = ray.is_initialized()
    (outd / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {outd / 'summary.json'} sha256 {RC.sha256(outd / 'summary.json')}; {summary['seconds']:.1f} s; ray initialised {ray.is_initialized()}")
    if ray.is_initialized():
        raise SystemExit("Ray was initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
