"""levin-phase-floor prompt 07 section 6: score V1, V2, V3 (with section 10's set-aside) and V5 on 07's census g, and
gather the descriptive items, against 06m's census. New in 07. Read only: 07's census run directory (checkpoint SHA-256
checked), 06m's census (by state/index_06m.json, each line's SHA-256 checked), state/rows.json, and the stop-3
comparison out/stop3/compare_census_calls.json (V1's per-call verdicts, which this script reads, not recomputes).

The rules are state/predictions.md's (part 1, verbatim; part 2, section 10's set-aside; part 4, how they are read):
- **V1** from compare_07.py ``calls`` (stop 3): every call that locates nothing and guards nothing equal, and every
  unit all of whose calls are such equal whole, with ``total`` and ``total_abserr`` equal (hex). Split here by rtol.
- **V2**: every scanned call (group at both rtols, analytic at 1e-7; ``scan_07`` in full, from the pickle): every
  returned leaf the scan lists (``grid`` or ``ends``) whose branch is Levin counts against V2. Each listed leaf's
  window, guarded and guard-at-cap flags are read from the region arrays (B.5: a listed leaf that is a window or a
  guarded leaf is not a Levin leaf; it is reported).
- **V3**: every scanned call with no phase Rule A skipped: each scan sign change (abscissa ``sign_change_x``) must have
  a located point of kind ``root`` within one scan spacing h, and each located ``root`` must lie within h of a scan
  sign change. Points of kind ``end`` are reported apart, with the verdict they would give counted as roots. A call
  with a phase Rule A skipped is set aside, counted and listed with its scan sign changes (section 10).
- **V5**: every call, group or analytic, at either rtol, that locates a point or guards a leaf, with a leaf at
  ``depth_max`` (``n_at_depth_max`` > 0), counts against it; an over-budget unit counts as false.
Descriptive (prompt section 6 and the note's B.5): row depth against 06m's; windows (pieces per call, widths in x,
theta'^2/|theta''| at their own edges); the stopped windows with ``leaves_falling_at_stop`` by rtol and the call's
convergence; totals in the 720 against 06m's, relative and against total_abserr_06m, by band and k; minima located
and their lambda, points at an end, analytic calls that located, ``num_guarded_regions``; unconverged calls in and
outside the 720 with 06m's flag-cost measures beside them; section 1.3's firings; cost (group-call regions, unit
walls and unit times against 06m's).

Usage: PYTHONPATH=<tree g>:<scripts> python -B score_07.py OUT_JSON [--processes N]
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
import refs_common as RC  # noqa: E402
from prov import CHECKPOINT06M_SHA256, RUN06M, WIDER, header, sha256  # noqa: E402

RUN = RC.CENSUS07
INDEX06M = WIDER / "state" / "index_06m.json"
ROWS = WIDER / "state" / "rows.json"
STOP3 = WIDER / "out" / "stop3" / "compare_census_calls.json"


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
    """06m's flag-cost measures of one unconverged call."""
    v, e = H(c["value"]), H(c["abserr"])
    req = max(c["atol"], c["rtol"] * abs(v))
    return {"within_request": e <= req, "over_request_ratio": e / req if req > 0 else math.inf,
            "node_limited": bool(c.get("node_limited")), "untrusted": int((c.get("leaves") or {}).get("untrusted") or 0),
            "n_at_depth_max": c.get("n_at_depth_max"), "share": e / total_abserr if total_abserr else None}


def one(args):
    uid, off, length, ent, rowinfo, in720 = args
    with open(RUN / "checkpoint.jsonl", "rb") as f:
        f.seek(off)
        line = f.read(length)
    r = json.loads(line)
    assert r["unit"] == uid
    d = r["data"]
    with open(RUN06M / "checkpoint.jsonl", "rb") as f:
        f.seek(ent["offset"])
        line6 = f.read(ent["length"])
    if hashlib.sha256(line6).hexdigest() != ent["line_sha256"]:
        raise RuntimeError(f"{uid}: 06m's line sha256 differs")
    m = json.loads(line6)["data"]
    out = {"unit": uid, "rtol": d["rtol"], "row": d["row"], "band": rowinfo["band"], "k": rowinfo["k"], "in720": in720,
           "status": d["status"], "status_06m": m["status"], "wall_s": [d.get("wall_s"), m.get("wall_s")],
           "time_s": [unit_time(d), unit_time(m)], "scan_07_error": d.get("scan_07_error")}
    if d["status"] != "done":
        return out
    calls = [("group", c) for c in d["calls"]] + [("analytic", c) for c in d["other_calls"]]
    calls6 = {(k, c["index"]): c for k, c in [("group", c) for c in m["calls"]] + [("analytic", c) for c in m["other_calls"]]}
    res, res6 = d.get("result") or {}, m.get("result") or {}
    out["total"] = [(res.get("total") or {}).get("hex"), (res6.get("total") or {}).get("hex")]
    out["total_abserr"] = [(res.get("total_abserr") or {}).get("hex"), (res6.get("total_abserr") or {}).get("hex")]
    out["total_converged"] = [res.get("total_converged"), res6.get("total_converged")]
    out["row_depth"] = [max((c["max_depth"] for _, c in calls), default=None),
                        max((c["max_depth"] for c in calls6.values()), default=None)]
    out["group_regions"] = [sum(c["num_regions"] for c in d["calls"]), sum(c["num_regions"] for c in m["calls"])]
    tae, tae6 = H(res.get("total_abserr")), H(res6.get("total_abserr"))
    need_pickle = any((c.get("guard_07") or {}).get("locates") or (c.get("guard_07") or {}).get("guards") for _, c in calls)
    need_pickle |= any((s.get("n_regions_listed") or 0) > 0 or (s.get("n_sign_changes") or 0) > 0 for s in d.get("scan_07") or [])
    rec = load_pickle(RUN / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]) if need_pickle else None
    g7full = {e["index"]: e for e in (rec or {}).get("guard_07") or []}
    s7full = {e["index"]: e for e in (rec or {}).get("scan_07") or []}
    regarr = {e["index"]: e for e in ((rec or {}).get("regions") or []) + ((rec or {}).get("other_regions") or [])}
    out["calls"] = []
    for kind, c in calls:
        g7 = c.get("guard_07") or {}
        c6 = calls6.get((kind, c["index"]))
        row = {"kind": kind, "index": c["index"], "label": c["notify_label"], "converged": [c["converged"], c6["converged"]],
               "locates": g7.get("locates"), "guards": g7.get("guards"), "n_at_depth_max": c.get("n_at_depth_max"),
               "max_depth": [c["max_depth"], c6["max_depth"]], "regions": [c["num_regions"], c6["num_regions"]],
               "skipped": g7.get("location_skipped_phases") or []}
        keep = bool(g7.get("locates") or g7.get("guards") or row["skipped"] or not c["converged"] or not c6["converged"])
        if not c["converged"]:
            row["flag"] = call_flag(c, tae)
        if not c6["converged"]:
            row["flag_06m"] = call_flag(c6, tae6)
        if g7.get("locates") or g7.get("guards"):
            full = g7full.get(c["index"]) or {}
            row["value"] = [c["value"]["hex"], c6["value"]["hex"]]
            row["abserr"] = [c["abserr"]["hex"], c6["abserr"]["hex"]]
            row["node_limited"] = c.get("node_limited")
            row["points"] = [{"kind": p["kind"], "found_as": p.get("found_as"), "x": p["x"], "lambda": p.get("lambda"),
                              "theta_deriv": p.get("theta_deriv"), "own_window": p.get("own_window"),
                              "edge_ratio": [e.get("theta_deriv_sq_over_second") for e in p.get("own_window_edges") or []],
                              "window_pieces": p.get("window_pieces")}
                             for p in full.get("stationary_points") or []]
            row["windows"] = [{k: w.get(k) for k in ("lo", "hi", "iroff1_count", "stopped", "stopped_after_split",
                                                    "leaves_at_stop", "leaves_falling_at_stop", "max_depth_at_stop")}
                              | {"n_pieces": max(len(w.get("cuts") or []) - 1, 0)} for w in full.get("stationary_windows") or []]
            row["num_window_pieces"] = full.get("num_window_pieces")
            row["num_guarded_regions"] = full.get("num_guarded_regions")
            row["num_guard_at_depth_max"] = full.get("num_guard_at_depth_max")
            row["returned"] = g7.get("returned")
        row["unlocated_minima_lambda"] = [m_.get("lambda") for m_ in (g7full.get(c["index"]) or {}).get("unlocated_minima") or []]
        keep |= bool(row["unlocated_minima_lambda"])
        # V2 and V3 on scanned calls
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
                    ra = regarr[c["index"]]
                    lst = []
                    for x in full_s["listed"]:
                        i = x["region_index"]
                        lst.append({"region_index": i, "how": x["how"], "branch": x["branch"], "depth": x["depth"],
                                    "start": x["start"], "end": x["end"], "window": int(ra["window_07"][i]),
                                    "guarded": int(ra["guarded_07"][i]), "guard_at_cap": int(ra["guard_at_depth_max_07"][i]),
                                    "phase_limited": x["phase_limited"]})
                    row["listed"] = lst
                    keep = True
                h = sc["h"]["hex"]
                row["scan"] = {"h": h, "n_sign_changes": sc["n_sign_changes"],
                               "sign_change_x": [] if not sc["n_sign_changes"] else list(full_s["sign_change_x"])}
                if sc["n_sign_changes"]:
                    keep = True
        else:
            row["scanned"] = False
        if keep:
            if (g7.get("locates") or g7.get("guards")) and "points" not in row:
                row["points"] = []
            out["calls"].append(row)
    out["n_calls"] = len(calls)
    out["n_scanned"] = len(d.get("scan_07") or [])
    out["n_scanned_by_kind"] = {"group": sum(1 for s in d.get("scan_07") or [] if s["kind"] == "group"),
                                "analytic": sum(1 for s in d.get("scan_07") or [] if s["kind"] == "analytic")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_json")
    ap.add_argument("--processes", type=int, default=6)
    ap.add_argument("--units", default=None, help="development only: a comma-separated subset of units")
    a = ap.parse_args()
    if a.processes > 6:
        raise SystemExit("at most six processes")
    ck = RUN / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: V1, V2, V3, V5 and the descriptive items on the census g",
                 [ck, INDEX06M, ROWS, STOP3])
    if sha256(ck) != RC.CENSUS07_CHECKPOINT_SHA256:
        raise SystemExit("07's census checkpoint has another SHA-256")
    idx = json.loads(INDEX06M.read_text())
    if idx["checkpoint_sha256"] != CHECKPOINT06M_SHA256:
        raise SystemExit("the index is not of 06m's checkpoint")
    rows = {r["name"]: r for r in json.loads(ROWS.read_text())["rows"]}
    jobs, off = [], 0
    with open(ck, "rb") as f:
        for line in f:
            head = line[:120].decode()
            if not head.startswith('{"unit": "'):
                raise SystemExit(f"unexpected record start at offset {off}")
            uid = head[len('{"unit": "'):].split('"', 1)[0]
            jobs.append([uid, off, len(line)])
            off += len(line)
    # the 720: rows whose 1e-7 G+q+r call shows a sign change in 06m's scan (state/v4_selection.json's population)
    sel = json.loads((WIDER / "state" / "v4_selection.json").read_text())
    n720 = sel["counts"]["population"]
    sp_rows = set()
    with open(RUN06M / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["data"]["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in r["data"]["scan"] or []):
                sp_rows.add(r["data"]["row"])
    if len(sp_rows) != 720 or n720 != 1438:
        raise SystemExit(f"the 720: {len(sp_rows)} rows; population {n720}")
    full = []
    for uid, o, n in jobs:
        parts = uid.split("-")  # g-1e-7-<shard>-<serial>
        if len(parts) != 5 or parts[0] != "g":
            raise SystemExit(f"unexpected unit {uid}")
        row = f"{parts[3]}-{parts[4]}"
        full.append((uid, o, n, idx["units"][uid], rows[row], row in sp_rows))
    if a.units:  # development only
        want = set(a.units.split(","))
        full = [j for j in full if j[0] in want]
    with mp.get_context("spawn").Pool(a.processes) as pool:
        units = list(pool.imap(one, full, chunksize=16))
    stop3 = json.loads(STOP3.read_text())
    doc = {"header": hdr, "n_units": len(units), "units": units,
           "stop3": {k: stop3[k] for k in ("n_units", "n_units_equal", "n_units_differing", "statuses", "totals", "calls_mode")},
           "stop3_by_rtol": {}}
    for rt in ("1e-7", "1e-8"):
        us = [u for u in stop3["units"] if u.get("rtol") == rt]
        doc["stop3_by_rtol"][rt] = {"units": len(us), "units_ok": sum(1 for u in us if u["ok"]),
                                    "clean_calls": sum(u.get("n_clean", 0) for u in us),
                                    "units_all_clean": sum(1 for u in us if u.get("all_clean")),
                                    "units_all_clean_total_equal": sum(1 for u in us if u.get("all_clean") and u.get("total_equal"))}
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(doc, default=str))
    print(f"units {len(units)}; statuses {sorted({u['status'] for u in units})}; wrote {a.out_json} sha256 {sha256(a.out_json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
