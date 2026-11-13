"""levin-phase-floor prompt 07a section 6: one pass over the census h (and 07's census beside it, by state/index_07.json,
each line's SHA-256 checked) gathering what W2, W3, W5 and the descriptive items need. New in 07a, after 07's
score_07.py (a record, not edited), whose V2 and V3 extraction (scan_07 against the returned leaves and the located
roots) it keeps. Read only. W1, W7 and the classing are compare_07a.py's (out/stop3/), W4 score_v4_07a.py's, W6
xcheck_07a.py's; this script decides no verdict (summarise_07a.py does, by state/predictions.md).

Per unit: rtol, row, band, k, in the 720 (07's census scan at 1e-7, as 07 defined the 720), status, unit times, totals,
row depth and group regions at h and at 07. Per call, kept when it carries anything W2, W3, W5 or a descriptive item
reads: its route_07a summary (solves changed by class, the partition counts, forced and gated, the Clenshaw-Curtis
regions with a Levin half, the margins, the largest |theta| at a gated end), every changed solve in full from the
pickle (``changed_all``: V, 07's estimate, direction, role, class), guard_07 at h and 07 (points, windows with their stop
records, unlocated minima, guarded counts), converged, value, abserr, depth and regions at both, the scan (V2's listed
leaves with their window and guard flags, V3's sign changes and h), and an unconverged call's flag measures.

Usage: PYTHONPATH=<tree h>:<scripts> python -B score_07a.py RUN_DIR OUT_JSON [--processes N]
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07_SHA256, RUN07, WIDER, header, sha256  # noqa: E402

INDEX07 = WIDER / "state" / "index_07.json"
ROWS = WIDER / "state" / "rows.json"
SIX = 6.0 * math.pi
RUN = None


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def load_pickle(path, off, length, sha):
    import ray.cloudpickle as cp

    with open(path, "rb") as f:
        f.seek(off)
        blob = f.read(length)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise RuntimeError(f"{path}@{off}: sha256 differs")
    return cp.loads(blob)


def unit_time(d):
    t = d.get("wall_s") or 0.0
    for k in ("scan_s", "scan_07_s", "cc_floor_s", "declared_06h_s", "checked_06g_s"):
        t += d.get(k) or 0.0
    for k in ("w3", "deep_lines", "analytic_06h"):
        t += (d.get(k) or {}).get("seconds") or 0.0
    return t


def call_flag(c, total_abserr):
    v, e = H(c["value"]), H(c["abserr"])
    req = max(c["atol"], c["rtol"] * abs(v))
    return {"within_request": e <= req, "over_request_ratio": e / req if req > 0 else math.inf,
            "node_limited": bool(c.get("node_limited")), "untrusted": int((c.get("leaves") or {}).get("untrusted") or 0),
            "n_at_depth_max": c.get("n_at_depth_max"), "share": e / total_abserr if total_abserr else None}


def one(args):
    uid, off, length, ent, rowinfo, in720, run = args
    with open(Path(run) / "checkpoint.jsonl", "rb") as f:
        f.seek(off)
        line = f.read(length)
    r = json.loads(line)
    assert r["unit"] == uid
    d = r["data"]
    with open(RUN07 / "checkpoint.jsonl", "rb") as f:
        f.seek(ent["offset"])
        line7 = f.read(ent["length"])
    if hashlib.sha256(line7).hexdigest() != ent["line_sha256"]:
        raise RuntimeError(f"{uid}: 07's line sha256 differs")
    m = json.loads(line7)["data"]
    out = {"unit": uid, "rtol": d["rtol"], "row": d["row"], "band": rowinfo["band"], "k": rowinfo["k"], "in720": in720,
           "status": d["status"], "status_07": m["status"], "wall_s": [d.get("wall_s"), m.get("wall_s")],
           "time_s": [unit_time(d), unit_time(m)], "scan_07_error": d.get("scan_07_error"),
           "post_unit": d.get("route_07a_post_unit") and {k: v for k, v in d["route_07a_post_unit"].items() if k != "changed"}}
    if d["status"] != "done":
        return out
    calls = [("group", c) for c in d["calls"]] + [("analytic", c) for c in d["other_calls"]]
    calls7 = {(k, c["index"]): c for k, c in [("group", c) for c in m["calls"]] + [("analytic", c) for c in m["other_calls"]]}
    res, res7 = d.get("result") or {}, m.get("result") or {}
    out["total"] = [(res.get("total") or {}).get("hex"), (res7.get("total") or {}).get("hex")]
    out["total_abserr"] = [(res.get("total_abserr") or {}).get("hex"), (res7.get("total_abserr") or {}).get("hex")]
    out["total_converged"] = [res.get("total_converged"), res7.get("total_converged")]
    out["row_depth"] = [max((c["max_depth"] for _, c in calls), default=None),
                        max((c["max_depth"] for c in calls7.values()), default=None)]
    out["group_regions"] = [sum(c["num_regions"] for c in d["calls"]), sum(c["num_regions"] for c in m["calls"])]
    out["all_regions"] = [sum(c["num_regions"] for _, c in calls), sum(c["num_regions"] for c in calls7.values())]
    out["n_solves"] = sum(((c.get("route_07a") or {}).get("n_solves") or 0) for _, c in calls)
    tae, tae7 = H(res.get("total_abserr")), H(res7.get("total_abserr"))
    need = any((c.get("guard_07") or {}).get("locates") or (c.get("guard_07") or {}).get("guards")
               or ((c.get("route_07a") or {}).get("n_route_differs") or 0) + ((c.get("route_07a") or {}).get("n_lu_differs") or 0)
               > (c.get("route_07a") or {}).get("n_changed_kept", 0) for _, c in calls)
    need |= any((s.get("n_regions_listed") or 0) > 0 or (s.get("n_sign_changes") or 0) > 0 for s in d.get("scan_07") or [])
    rec = load_pickle(Path(run) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]) if need else None
    g7full = {e["index"]: e for e in (rec or {}).get("guard_07") or []}
    s7full = {e["index"]: e for e in (rec or {}).get("scan_07") or []}
    r7afull = {e["index"]: e for e in (rec or {}).get("route_07a") or []}
    regarr = {e["index"]: e for e in ((rec or {}).get("regions") or []) + ((rec or {}).get("other_regions") or [])}
    out["calls"] = []
    agg = {"partition_forced": 0, "partition_nonforced": 0, "cc_lh": 0, "cc_lh_07est": 0, "guarded": 0,
           "guard_at_depth_max": 0, "no_build": 0, "route_errors": 0, "theta_max": {}, "m6": None, "m20": None,
           "n_route": 0, "n_lu": 0}
    for kind, c in calls:
        g7 = c.get("guard_07") or {}
        c7 = calls7.get((kind, c["index"]))
        g77 = (c7 or {}).get("guard_07") or {}
        ra = c.get("route_07a")
        if not isinstance(ra, dict) or "error" in ra:
            agg["route_errors"] += 1
            ra = {}
        part = ra.get("partition") or {}
        agg["partition_forced"] += part.get("forced", 0)
        agg["partition_nonforced"] += ra.get("n_partition_nonforced", 0)
        agg["cc_lh"] += ra.get("n_cc_with_levin_half", 0)
        agg["cc_lh_07est"] += ra.get("n_cc_with_levin_half_07est", 0)
        agg["guarded"] += int(g7.get("num_guarded_regions") or 0)
        agg["guard_at_depth_max"] += int(g7.get("num_guard_at_depth_max") or 0)
        agg["no_build"] += ra.get("n_no_build", 0)
        agg["n_route"] += ra.get("n_route_differs", 0)
        agg["n_lu"] += ra.get("n_lu_differs", 0)
        th = H(ra.get("theta_end_max_nonforced"))
        if th is not None and th == th and (agg["theta_max"].get(kind) is None or th > agg["theta_max"][kind][0]):
            agg["theta_max"][kind] = (th, c["index"], c["notify_label"])
        for key_, nm in (("min_margin_6pi_same", "m6"), ("min_margin_20pi_same_levin", "m20")):
            mm = ra.get(key_)
            if mm and (agg[nm] is None or H(mm["margin"]) < agg[nm][0]):
                agg[nm] = (H(mm["margin"]), kind, c["index"], {k_: (H(v_) if isinstance(v_, dict) else v_) for k_, v_ in mm.items()})
        row = {"kind": kind, "index": c["index"], "label": c["notify_label"], "converged": [c["converged"], (c7 or {}).get("converged")],
               "value": [c["value"]["hex"], (c7 or {}).get("value", {}).get("hex")],
               "abserr": [c["abserr"]["hex"], (c7 or {}).get("abserr", {}).get("hex")],
               "max_depth": [c["max_depth"], (c7 or {}).get("max_depth")], "regions": [c["num_regions"], (c7 or {}).get("num_regions")],
               "n_at_depth_max": c.get("n_at_depth_max"), "locates": g7.get("locates"), "guards": g7.get("guards"),
               "n_windows": [g7.get("n_windows"), g77.get("n_windows")], "n_points": [g7.get("n_points"), g77.get("n_points")],
               "skipped": [g7.get("location_skipped_phases") or [], g77.get("location_skipped_phases") or []],
               "route": {k_: ra.get(k_) for k_ in ("n_solves", "n_route_differs", "n_lu_differs", "differs_by_class", "partition",
                                                    "n_cc_with_levin_half", "n_cc_with_levin_half_07est", "n_forced")}}
        keep = bool(g7.get("locates") or g7.get("guards") or g77.get("locates") or row["skipped"][0] or row["skipped"][1]
                    or not c["converged"] or not (c7 or {}).get("converged") or ra.get("n_route_differs") or ra.get("n_lu_differs")
                    or ra.get("n_partition_nonforced") or part.get("forced") or ra.get("n_cc_with_levin_half"))
        if ra.get("n_route_differs") or ra.get("n_lu_differs"):
            full_r = (r7afull.get(c["index"]) or {}).get("record") or {}
            row["changed"] = [{k_: x.get(k_) for k_ in ("a", "b", "desc", "role", "class", "made", "returned", "depth",
                                                           "V", "est_07", "est_over_V", "decision", "is_direct", "lu_used")}
                              for x in (full_r.get("changed_all") if full_r else ra.get("changed") or [])]
        if ra.get("n_cc_with_levin_half"):
            row["cc_with_levin_half"] = ra.get("cc_with_levin_half")
        if not c["converged"]:
            row["flag"] = call_flag(c, tae)
        if c7 is not None and not c7["converged"]:
            row["flag_07"] = call_flag(c7, tae7)
        if g7.get("locates") or g7.get("guards") or g77.get("locates"):
            full = g7full.get(c["index"]) or {}
            row["node_limited"] = c.get("node_limited")
            row["points"] = [{"kind": p["kind"], "found_as": p.get("found_as"), "x": p["x"], "lambda": p.get("lambda"),
                              "window_pieces": p.get("window_pieces")} for p in full.get("stationary_points") or []]
            row["windows"] = [{k: w.get(k) for k in ("lo", "hi", "iroff1_count", "stopped", "stopped_after_split",
                                                    "leaves_at_stop", "leaves_falling_at_stop", "max_depth_at_stop")}
                              | {"n_pieces": max(len(w.get("cuts") or []) - 1, 0)} for w in full.get("stationary_windows") or []]
            row["windows_07"] = [{k: w.get(k) for k in ("iroff1_count", "stopped", "leaves_falling_at_stop", "n_pieces")}
                                 for w in g77.get("windows") or []]
            row["n_windows_stopped"] = [g7.get("n_windows_stopped"), g77.get("n_windows_stopped")]
            row["num_window_pieces"] = [full.get("num_window_pieces"), g77.get("num_window_pieces")]
            row["num_guarded_regions"] = full.get("num_guarded_regions")
            row["returned"] = g7.get("returned")
        row["unlocated_minima_lambda"] = [m_.get("lambda") for m_ in (g7full.get(c["index"]) or {}).get("unlocated_minima") or []]
        if not row["unlocated_minima_lambda"] and g7.get("n_unlocated_minima"):
            row["unlocated_minima_lambda"] = [m_.get("lambda") for m_ in g7.get("unlocated_minima") or []]
        keep |= bool(row["unlocated_minima_lambda"])
        s7c = [s for s in d.get("scan_07") or [] if s["index"] == c["index"]]
        if s7c:
            sc = s7c[0]
            row["scanned"] = True
            if "error" in sc:
                row["scan_error"] = sc["error"][-300:]
                keep = True
            else:
                full_s = (s7full.get(c["index"]) or {}).get("scan") if (sc["n_regions_listed"] or sc["n_sign_changes"]) else None
                if sc["n_regions_listed"]:
                    ra_ = regarr[c["index"]]
                    row["listed"] = [{"region_index": x["region_index"], "how": x["how"], "branch": x["branch"],
                                      "depth": x["depth"], "start": x["start"], "end": x["end"],
                                      "window": int(ra_["window_07"][x["region_index"]]),
                                      "guarded": int(ra_["guarded_07"][x["region_index"]]),
                                      "guard_at_cap": int(ra_["guard_at_depth_max_07"][x["region_index"]])}
                                     for x in full_s["listed"]]
                    keep = True
                row["scan"] = {"h": sc["h"]["hex"], "n_sign_changes": sc["n_sign_changes"],
                               "min_over_max_abs_theta_deriv": sc.get("min_over_max_abs_theta_deriv"),
                               "n_nonfinite": sc.get("n_nonfinite"),
                               "sign_change_x": [] if not sc["n_sign_changes"] else list(full_s["sign_change_x"])}
                keep |= bool(sc["n_sign_changes"])
        else:
            row["scanned"] = False
        if keep:
            out["calls"].append(row)
    out["agg"] = agg
    out["n_calls"] = len(calls)
    out["n_scanned"] = len(d.get("scan_07") or [])
    out["n_scanned_by_kind"] = {"group": sum(1 for s in d.get("scan_07") or [] if s["kind"] == "group"),
                                "analytic": sum(1 for s in d.get("scan_07") or [] if s["kind"] == "analytic")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    if a.processes > 6:
        raise SystemExit("at most six processes")
    run = Path(a.run_dir).resolve()
    ck = run / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: W2, W3, W5 and the descriptive items, gathered from the census h",
                 [ck, INDEX07, ROWS])
    idx = json.loads(INDEX07.read_text())
    if idx["checkpoint_sha256"] != CHECKPOINT07_SHA256:
        raise SystemExit("the index is not of 07's checkpoint")
    rows = {r["name"]: r for r in json.loads(ROWS.read_text())["rows"]}
    jobs, off = [], 0
    with open(ck, "rb") as f:
        for line in f:
            uid = line[:120].decode()[len('{"unit": "'):].split('"', 1)[0]
            jobs.append([uid, off, len(line)])
            off += len(line)
    sp_rows = set()  # the 720: rows whose 1e-7 G+q+r call shows a sign change in 07's census's 06a scan
    with open(RUN07 / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["data"]["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in r["data"]["scan"] or []):
                sp_rows.add(r["data"]["row"])
    if len(sp_rows) != 720:
        raise SystemExit(f"the 720: {len(sp_rows)} rows")
    full = []
    for uid, o, n in jobs:
        p = uid.split("-")
        if len(p) != 5 or p[0] != "h":
            raise SystemExit(f"unexpected unit {uid}")
        row = f"{p[3]}-{p[4]}"
        full.append((uid, o, n, idx["units"]["g-" + uid.split("-", 1)[1]], rows[row], row in sp_rows, str(run)))
    with mp.get_context("spawn").Pool(a.processes) as pool:
        units = list(pool.imap(one, full, chunksize=16))
    doc = {"header": hdr, "run": str(run), "checkpoint_sha256": sha256(ck), "n_units": len(units), "n_720": len(sp_rows), "units": units}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(doc, default=str))
    print(f"units {len(units)}; statuses {sorted({u['status'] for u in units})}; wrote {a.out_json} sha256 {sha256(a.out_json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
