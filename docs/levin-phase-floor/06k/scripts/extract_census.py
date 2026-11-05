"""levin-phase-floor prompt 06k, Part C: a compact extract of one census checkpoint, for the comparisons. New in
06k (its per-call fields follow 06g's v8_06g.py ``slim_call``). Read only.

One pass over RUN_DIR/checkpoint.jsonl. Per unit: uid, rtol, row, status, wall_s, rebuild_s, the record's
"recorded" time, the row's total, total_abserr and total_converged, the region-pickle pointer, and per call (group
calls, kind "group", then the analytic blocks, kind "analytic"): index, label, group, value and abserr (hex),
converged, num_regions, max_depth, n_at_depth_max, depth_max, num_untrusted_regions, node_limited, the checked-leaf
counts n_b_untrusted and n_checked, and the route record compactly: n_firings, and per firing the tuple (kind,
depth, at_depth_max, r, delta1, phi_P, D, cap, is_direct, trusted_g) (None where the field is absent), with route
3's left-leaf numbers. Writes OUT_PICKLE (plain pickle) and prints the counts.
Usage: python -B extract_census.py RUN_DIR OUT_PICKLE
"""

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path


def h(x):
    return None if x is None else (x["hex"] if isinstance(x, dict) else x)


def num(x):
    if x is None:
        return None
    return float(x)


def firing(f):
    return (f.get("kind"), f.get("depth"), f.get("at_depth_max"), num(f.get("r")), num(f.get("delta1")),
            num(f.get("phi_P")), num(f.get("D")), num(f.get("cap")), f.get("is_direct"), f.get("trusted_g"))


def slim(kind, c):
    ck = c.get("checked_06g") if isinstance(c.get("checked_06g"), dict) else {}
    r = c.get("route_06k")
    route = None
    if r is not None:
        route = {"n_firings": r.get("n_firings"), "firings": [firing(f) for f in r.get("firings", [])]}
        for k in ("n_left_returned", "eta_min", "eta_min_with_depth_max", "n_untrusted_at_depth_max_not_left",
                  "n_untrusted_not_left", "left_D_sum", "request", "n_unleft", "budget_met"):
            if k in r:
                route[k] = r[k]
    return {"kind": kind, "index": c["index"], "label": c.get("notify_label"), "group": c.get("group"),
            "value": h(c["value"]), "abserr": h(c["abserr"]), "converged": c.get("converged"),
            "num_regions": c.get("num_regions"), "max_depth": c.get("max_depth"), "n_at_depth_max": c.get("n_at_depth_max"),
            "depth_max": c.get("depth_max"), "num_untrusted_regions": c.get("num_untrusted_regions"),
            "node_limited": c.get("node_limited"), "n_b_untrusted": ck.get("n_b_untrusted"), "n_checked": ck.get("n_checked"),
            "x_span": [h(c["x_span"][0]), h(c["x_span"][1])], "atol": c.get("atol"), "rtol": c.get("rtol"), "route": route}


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.time()
    ck = run / "checkpoint.jsonl"
    sha = hashlib.sha256()
    units = {}
    with open(ck, "rb") as f:
        for line in f:
            sha.update(line)
            r = json.loads(line)
            d = r["data"]
            res = d.get("result") or {}
            units[r["unit"]] = {
                "uid": r["unit"], "rtol": d["rtol"], "row": d["row"], "status": d["status"], "wall_s": d.get("wall_s"),
                "rebuild_s": d.get("rebuild_s"), "recorded": r.get("recorded"), "worker": d.get("worker"),
                "total": h(res.get("total")), "total_abserr": h(res.get("total_abserr")),
                "total_converged": res.get("total_converged"),
                "pickle": [d.get("pickle"), d.get("pickle_offset"), d.get("pickle_length"), d.get("pickle_record_sha256")],
                "calls": [slim("group", c) for c in d.get("calls") or []] + [slim("analytic", c) for c in d.get("other_calls") or []],
            }
    doc = {"run": str(run), "checkpoint_sha256": sha.hexdigest(), "manifest": json.loads((run / "manifest.json").read_text()),
           "status": json.loads((run / "status.json").read_text()), "units": units}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fo:
        pickle.dump(doc, fo, protocol=5)
    n_calls = sum(len(u["calls"]) for u in units.values())
    print(f"{run.name}: {len(units)} units, {n_calls} calls; checkpoint sha256 {doc['checkpoint_sha256']}; "
          f"{time.time() - t0:.0f} s; wrote {out} sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
