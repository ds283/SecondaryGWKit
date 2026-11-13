"""W1 by rtol, for the review (§C.2): stop3_compare_07a.py with its class counts and its untouched-unit totals split by rtol;
nothing else changed. Orchestrator's own comparison at planned stop 3 (levin-phase-floor 07a Part C). Read only but for OUT_JSON.
Independent of the census agent's compare scripts; modelled on 07's review/scripts/stop3_compare.py.

usage: python -I -B stop3_compare_07a.py RUN07A RUN07 INDEX07A INDEX07 OUT_JSON [--processes N]

For every unit, both checkpoint lines are read at the offsets of my own indices (index07.py), accepted only if the
bytes' SHA-256 equals the index's; each region record is read from its pickle at the line's pointer, accepted only if
it matches the line's pickle_record_sha256. Both records are re-keyed by call index (calls, other_calls, every pickle
list carrying `index`, the pickle's positional `scan` by group-call order) and every leaf compared (floats by
float.hex, arrays by dtype, shape and bytes). Each differing path is classified:
  timing, provenance, cache_built                         as 07's stop2_compare.classify;
  only_h_07a  a path absent at 07 whose name parts include a part containing "07a" or one of NEW07A_EXTRA;
  location    under guard_07.location_skipped_phases;
  otherwise   attributed to its call (CALL<i>) as `call_other`, or to the unit as `unit_other`.
Classes, per call: LOCATED if guard_07.n_points or n_windows > 0 at 07 or at h (window_both / window_h_only /
window_07_only); else OWN if h's route_07a.own_differs; else AUX if route_07a.aux_differs; else UNTOUCHED.
Guard counts (guarded regions, guard at depth max, returned flags) are tabulated per class at both trees.
Strict test reported: UNTOUCHED calls with any call_other path (the stop-3 condition is narrower: the call's outputs).
"""
import argparse, hashlib, json, multiprocessing as mp, pickle, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

PROV = ("unit", "tree", "tree_head", "tree_status_short", "worker", "worker_pid", "pickle", "pickle_offset",
        "pickle_length", "pickle_record_sha256", "scripts_sha256", "modules", "py", "worker_tree")
NEW07A_EXTRA = ("x1_inputs",)


def leaves(x, p=""):
    if isinstance(x, dict):
        if not x:
            yield p, ("emptydict",)
        for k, v in x.items():
            yield from leaves(v, f"{p}.{k}" if p else str(k))
    elif isinstance(x, (list, tuple)):
        if not x:
            yield p, ("emptyseq", type(x).__name__)
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    elif isinstance(x, np.ndarray):
        yield p, ("nd", str(x.dtype), x.shape, x.tobytes())
    elif isinstance(x, float):
        yield p, ("f", x.hex())
    elif isinstance(x, np.floating):
        yield p, ("f", float(x).hex(), str(type(x)))
    else:
        yield p, (type(x).__name__, repr(x))


def rekey_list(lst):
    out = {}
    for e in lst:
        out.setdefault(f"CALL<{e['index']}>", []).append(e)
    return out


def rekey_data(d):
    d = dict(d)
    for name in ("calls", "other_calls", "scan", "deep_lines"):
        v = d.get(name)
        if isinstance(v, list) and all(isinstance(e, dict) and "index" in e for e in v):
            d[name] = rekey_list(v)
    dl = d.get("deep_lines_06e")
    if isinstance(dl, dict) and isinstance(dl.get("lines"), list) and all(
            isinstance(e, dict) and "index" in e for e in dl["lines"]):
        dl = dict(dl); dl["lines"] = rekey_list(dl["lines"]); d["deep_lines_06e"] = dl
    return d


def rekey_pickle(p, gi):
    p = dict(p)
    for name, v in list(p.items()):
        if name == "scan" and isinstance(v, list):
            p[name] = {f"CALL<{i}>": [e] for i, e in zip(gi, v)} if len(v) == len(gi) else {"SCAN_LEN_MISMATCH": v}
        elif isinstance(v, list) and v and all(isinstance(e, dict) and "index" in e for e in v):
            p[name] = rekey_list(v)
    return p


def classify(p, absent07):
    parts = [x for x in re.split(r"[.\[\]]+", p) if x]
    if absent07 and any(("07a" in x) or x in NEW07A_EXTRA for x in parts):
        return "only_h_07a"
    if any(x.endswith("_s") or x in ("seconds", "elapsed", "compute_time") for x in parts):
        return "timing"
    if parts[0] in PROV or any(x in ("scripts_sha256", "modules") for x in parts):
        return "provenance"
    if parts[0] == "cache_built":
        return "cache_built"
    if "location_skipped_phases" in parts:
        return "location"
    m = re.search(r"CALL<(-?\d+)>", p)
    return ("call_other", int(m.group(1))) if m else "unit_other"


def readat(path, off, length, sha):
    with open(path, "rb") as f:
        f.seek(off); raw = f.read(length)
    if hashlib.sha256(raw).hexdigest() != sha:
        raise RuntimeError(f"sha mismatch {path}@{off}")
    return raw


def calls(d):
    return (d.get("calls") or []) + (d.get("other_calls") or [])


def gcount(c):
    g = c.get("guard_07") or {}
    r = g.get("returned") or {}
    return (g.get("num_guarded_regions", 0), g.get("num_guard_at_depth_max", 0), r.get("n_guarded", 0),
            r.get("n_guard_at_depth_max", 0))


def one(job):
    key, e7a, e07, run7a, run07 = job
    d = json.loads(readat(Path(run7a) / "checkpoint.jsonl", e7a[0], e7a[1], e7a[2]))["data"]
    g = json.loads(readat(Path(run07) / "checkpoint.jsonl", e07[0], e07[1], e07[2]))["data"]
    assert d["row"] + "@" + d["rtol"] == key == g["row"] + "@" + g["rtol"]
    ph = pickle.loads(readat(Path(run7a) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]))
    p7 = pickle.loads(readat(Path(run07) / g["pickle"], g["pickle_offset"], g["pickle_length"], g["pickle_record_sha256"]))
    sig = lambda dd: sorted((c["index"], c["notify_label"], json.dumps(c["x_span"], sort_keys=True)) for c in calls(dd))
    row = {"key": key, "unit": d["unit"], "status": [d["status"], g["status"]], "calls_match": sig(d) == sig(g)}
    c7 = {c["index"]: c for c in calls(g)}
    cls = {}
    for c in calls(d):
        i = c["index"]; o = c7[i]
        gh, g7 = c.get("guard_07") or {}, o.get("guard_07") or {}
        lh = gh.get("n_points", 0) > 0 or gh.get("n_windows", 0) > 0
        l7 = g7.get("n_points", 0) > 0 or g7.get("n_windows", 0) > 0
        ra = c.get("route_07a") or {}
        if lh and l7: k = "window_both"
        elif lh: k = "window_h_only"
        elif l7: k = "window_07_only"
        elif ra.get("own_differs"): k = "own"
        elif ra.get("aux_differs"): k = "aux_only"
        else: k = "untouched"
        cls[i] = k
    gi_h = [c["index"] for c in d.get("calls") or []]
    gi_7 = [c["index"] for c in g.get("calls") or []]
    cnt, per_call, unit_other, only_h_names, loc_pat = Counter(), {}, Counter(), Counter(), Counter()
    for name, A, B in (("data", rekey_data(g), rekey_data(d)), ("pickle", rekey_pickle(p7, gi_7), rekey_pickle(ph, gi_h))):
        la, lb = dict(leaves(A)), dict(leaves(B))
        for p in set(la) | set(lb):
            if la.get(p, "<absent>") == lb.get(p, "<absent>"):
                continue
            c = classify(p, p not in la)
            if isinstance(c, tuple):
                per_call.setdefault(c[1], Counter())[re.sub(r"\[\d+\]", "[i]", re.sub(r"CALL<-?\d+>", "CALL", f"{name}:{p}"))] += 1
                c = "call_other"
            elif c == "location":
                m = re.search(r"CALL<(-?\d+)>", p)
                per_call.setdefault(int(m.group(1)) if m else None, Counter())["__location__"] += 1
            elif c == "unit_other":
                unit_other[re.sub(r"\[\d+\]", "[i]", f"{name}:{p}")] += 1
            elif c == "only_h_07a":
                only_h_names[re.sub(r"\[\d+\]", "[i]", re.sub(r"CALL<-?\d+>", "CALL", p)).split(".")[0 if name == "data" else 0]] += 1
            cnt[c] += 1
    # paths absent at 07 that were not classed only_h_07a would have fallen to other classes; record them
    calls_out = {}
    for i, k in cls.items():
        pc = per_call.get(i, Counter())
        other = {p: n for p, n in pc.items() if p != "__location__"}
        ch = next(c for c in calls(d) if c["index"] == i)
        co = c7[i]
        lp = None
        if pc.get("__location__"):
            lp = (json.dumps(sorted(sorted(e.keys()) for e in (co["guard_07"].get("location_skipped_phases") or []))),
                  json.dumps(sorted(sorted((kk, e[kk] if kk == "reason" else "*") for kk in e)
                                    for e in (ch["guard_07"].get("location_skipped_phases") or []))))
        calls_out[i] = {"class": k, "location": bool(pc.get("__location__")), "other": other, "loc_pattern": lp,
                        "guards_h": gcount(ch), "guards_07": gcount(co),
                        "out_equal": all(json.dumps(ch.get(f)) == json.dumps(co.get(f))
                                         for f in ("value", "abserr", "converged", "num_regions"))}
    row.update(calls=calls_out, unit_other=dict(unit_other), only_h_names=dict(only_h_names), counts=dict(cnt),
               all_untouched=all(v == "untouched" for v in cls.values()),
               totals_hex_equal=(json.dumps(d["result"]["total"], sort_keys=True) == json.dumps(g["result"]["total"], sort_keys=True)
                                 and json.dumps(d["result"]["total_abserr"], sort_keys=True) == json.dumps(g["result"]["total_abserr"], sort_keys=True))
               if d.get("result") and g.get("result") else None)
    return row


def main():
    ap = argparse.ArgumentParser()
    for a in ("run7a", "run07", "index7a", "index07", "out_json"):
        ap.add_argument(a)
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    i7a, i07 = json.load(open(a.index7a)), json.load(open(a.index07))
    assert set(i7a["units"]) == set(i07["units"])
    jobs = [(k, i7a["units"][k], i07["units"][k], a.run7a, a.run07) for k in sorted(i7a["units"])]
    with mp.Pool(a.processes) as pool:
        rows = pool.map(one, jobs, chunksize=20)
    T = Counter(); byclass = Counter(); strict_bad = []; loc_only = Counter(); loc_pat = Counter()
    guards = Counter(); outeq = Counter(); unit_other_all = Counter(); only_h = Counter(); unit_other_untouched = []
    for r in rows:
        T["units"] += 1
        T["calls_mismatch_units"] += not r["calls_match"]
        T["units_not_done"] += r["status"] != ["done", "done"]
        for k, v in r["only_h_names"].items():
            only_h[k] += v
        for p, n in r["unit_other"].items():
            unit_other_all[p] += 1
        if r["all_untouched"]:
            T["units_all_untouched " + r["key"].split("@")[1]] += 1
            T["units_all_untouched_totals_hex_equal " + r["key"].split("@")[1]] += bool(r["totals_hex_equal"])
            if r["unit_other"]:
                unit_other_untouched.append((r["key"], r["unit_other"]))
        for i, c in r["calls"].items():
            k = c["class"] + " " + r["key"].split("@")[1]; byclass[k] += 1; T["calls"] += 1
            outeq[(k, c["out_equal"])] += 1
            if c["guards_h"] != (0, 0, 0, 0) or c["guards_07"] != (0, 0, 0, 0):
                guards[k] += 1
            if c["other"]:
                byclass[(k, "other")] += 1
                if k.startswith("untouched"):
                    strict_bad.append((r["key"], i, c["other"]))
            elif c["location"]:
                loc_only[k] += 1
                if k.startswith("untouched"):
                    loc_pat[c["loc_pattern"]] += 1
            else:
                byclass[(k, "equal")] += 1
    summary = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "index7a_sha256": i7a["sha256"], "index07_sha256": i07["sha256"],
        "totals": dict(T),
        "class_counts": {str(k): v for k, v in sorted(byclass.items(), key=str)},
        "location_only_by_class": dict(loc_only),
        "untouched_location_patterns_07_vs_h": {str(k): v for k, v in loc_pat.most_common()},
        "calls_with_any_guard_count_by_class": dict(guards),
        "outputs_equal(value,abserr,converged,num_regions)_by_class": {str(k): v for k, v in sorted(outeq.items(), key=str)},
        "untouched_with_other_differences": len(strict_bad), "untouched_other_examples": strict_bad[:50],
        "only_h_07a_names": dict(only_h),
        "unit_other_paths_units": dict(unit_other_all.most_common(60)),
        "all_untouched_units_with_unit_other": len(unit_other_untouched), "examples": unit_other_untouched[:20],
    }
    Path(a.out_json).write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("unit_other_paths_units",)}, indent=1, default=str)[:12000])
    print("unit_other paths (units):")
    for p, n in unit_other_all.most_common(60):
        print(f"  {n:6d} {p}")


if __name__ == "__main__":
    main()
