"""Orchestrator's own comparison at planned stop 3 (levin-phase-floor 07b Part C). Read only but for OUT_JSON.
Copied from 07a's review/stop3/scripts/stop3_compare_07a.py; the before is 07a's census, the classes are 07b's.
Independent of the census agent's compare scripts.

usage: python -I -B stop3_compare_07b.py RUN07B RUN07A INDEX07B INDEX07A OUT_JSON [--processes N]

For every unit, both checkpoint lines are read at the offsets of my own indices (index_ck.py), accepted only if the
bytes' SHA-256 equals the index's; each region record is read from its pickle at the line's pointer, accepted only if
it matches the line's pickle_record_sha256. Both records are re-keyed by call index (calls, other_calls, every pickle
list carrying `index`, the pickle's positional `scan` by group-call order) and every leaf compared (floats by
float.hex, arrays by dtype, shape and bytes). Each differing path is classified:
  timing, provenance, cache_built                         as 07a's reader;
  only_h_07b  a path absent at 07a whose name parts include a part containing "07b" or one of NEW07B_EXTRA;
  location    under guard_07.location_skipped_phases;
  otherwise   attributed to its call (CALL<i>) as `call_other`, or to the unit as `unit_other`.
Class, per call (07b's U1): UNTOUCHED iff num_edge_rule_refusals + num_range_rule_refusals == 0 (both must be ints);
else EDGE (an edge refusal) or RANGE (range refusals only). The five outputs compared per call: value, abserr,
converged, num_regions, max_depth (json of the recorded objects, which carry float hex).
"""
import argparse, hashlib, json, multiprocessing as mp, pickle, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

PROV = ("unit", "tree", "tree_head", "tree_status_short", "worker", "worker_pid", "pickle", "pickle_offset",
        "pickle_length", "pickle_record_sha256", "scripts_sha256", "modules", "py", "worker_tree")
NEW07B_EXTRA = ("num_edge_rule_refusals", "num_range_rule_refusals")
FIVE = ("value", "abserr", "converged", "num_regions", "max_depth")


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


def classify(p, absent07a):
    parts = [x for x in re.split(r"[.\[\]]+", p) if x]
    if absent07a and any(("07b" in x) or x in NEW07B_EXTRA for x in parts):
        return "only_h_07b"
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
    return [("group", c) for c in (d.get("calls") or [])] + [("analytic", c) for c in (d.get("other_calls") or [])]


def one(job):
    key, eb, ea, runb, runa = job
    d = json.loads(readat(Path(runb) / "checkpoint.jsonl", eb[0], eb[1], eb[2]))["data"]
    g = json.loads(readat(Path(runa) / "checkpoint.jsonl", ea[0], ea[1], ea[2]))["data"]
    assert d["row"] + "@" + d["rtol"] == key == g["row"] + "@" + g["rtol"]
    ph = pickle.loads(readat(Path(runb) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"]))
    pa = pickle.loads(readat(Path(runa) / g["pickle"], g["pickle_offset"], g["pickle_length"], g["pickle_record_sha256"]))
    sig = lambda dd: sorted((k, c["index"], c["notify_label"], json.dumps(c["x_span"], sort_keys=True)) for k, c in calls(dd))
    row = {"key": key, "unit": d["unit"], "unit_07a": g["unit"], "status": [d["status"], g["status"]],
           "calls_match": sig(d) == sig(g)}
    idx_b = [c["index"] for _, c in calls(d)]
    assert len(idx_b) == len(set(idx_b)), ("index not unique across calls and other_calls", key)
    ca = {c["index"]: (k, c) for k, c in calls(g)}
    cls, refs = {}, {}
    for k, c in calls(d):
        e, r = c["num_edge_rule_refusals"], c["num_range_rule_refusals"]
        assert isinstance(e, int) and isinstance(r, int), (key, c["index"], e, r)
        assert (c.get("rate_rule_07b") or {}).get("counts_equal_result") is True, (key, c["index"])
        cls[c["index"]] = "untouched" if e + r == 0 else ("edge" if e > 0 else "range")
        refs[c["index"]] = (k, e, r)
    gi_b = [c["index"] for c in d.get("calls") or []]
    gi_a = [c["index"] for c in g.get("calls") or []]
    cnt, per_call, unit_other, only_h_names = Counter(), {}, Counter(), Counter()
    for name, A, B in (("data", rekey_data(g), rekey_data(d)), ("pickle", rekey_pickle(pa, gi_a), rekey_pickle(ph, gi_b))):
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
            elif c == "only_h_07b":
                only_h_names[re.sub(r"\[\d+\]", "[i]", re.sub(r"CALL<-?\d+>", "CALL", p)).split(".")[0]] += 1
            cnt[c] += 1
    calls_out = {}
    cb = {c["index"]: (k, c) for k, c in calls(d)}
    for i, k in cls.items():
        pc = per_call.get(i, Counter())
        other = {p: n for p, n in pc.items() if p != "__location__"}
        kb, ch = cb[i]
        ka, co = ca[i]
        assert kb == ka, (key, i)
        for f in FIVE:
            assert f in ch and f in co, (key, i, f)
        five_eq = {f: json.dumps(ch[f], sort_keys=True) == json.dumps(co[f], sort_keys=True) for f in FIVE}
        calls_out[i] = {"class": k, "kind": kb, "refusals": refs[i][1:], "location": bool(pc.get("__location__")),
                        "other": other, "five_equal": all(five_eq.values()),
                        "five_differ": [f for f, v in five_eq.items() if not v],
                        "converged": [ch["converged"], co["converged"]],
                        "regions": [ch["num_regions"], co["num_regions"]], "max_depth": [ch["max_depth"], co["max_depth"]]}
    tot = lambda x: json.dumps((x.get("result") or {}).get("total"), sort_keys=True) + json.dumps(
        (x.get("result") or {}).get("total_abserr"), sort_keys=True)
    row.update(calls=calls_out, unit_other=dict(unit_other), only_h_names=dict(only_h_names), counts=dict(cnt),
               all_untouched=all(v == "untouched" for v in cls.values()),
               totals_hex_equal=(tot(d) == tot(g)) if d.get("result") and g.get("result") else None,
               wall=[d.get("wall_s"), g.get("wall_s")])
    return row


def main():
    ap = argparse.ArgumentParser()
    for a in ("runb", "runa", "indexb", "indexa", "out_json"):
        ap.add_argument(a)
    ap.add_argument("--processes", type=int, default=6)
    a = ap.parse_args()
    ib, ia = json.load(open(a.indexb)), json.load(open(a.indexa))
    assert set(ib["units"]) == set(ia["units"])
    jobs = [(k, ib["units"][k], ia["units"][k], a.runb, a.runa) for k in sorted(ib["units"])]
    with mp.Pool(a.processes) as pool:
        rows = pool.map(one, jobs, chunksize=20)
    T = Counter(); byclass = Counter(); five = Counter(); bad = []; strict_other = []; loc_only = Counter()
    unit_other_all = Counter(); only_h = Counter(); unit_other_untouched = []; conv = Counter(); regions = Counter()
    refusals = Counter(); depth = Counter(); not_done = []; wall = Counter()
    for r in rows:
        T["units"] += 1
        T["calls_mismatch_units"] += not r["calls_match"]
        if r["status"] != ["done", "done"]:
            not_done.append((r["key"], r["status"]))
        for k, v in r["only_h_names"].items():
            only_h[k] += v
        for p in r["unit_other"]:
            unit_other_all[p] += 1
        if r["all_untouched"]:
            T["units_all_untouched"] += 1
            T["units_all_untouched_totals_hex_equal"] += bool(r["totals_hex_equal"])
            if r["unit_other"]:
                unit_other_untouched.append((r["key"], r["unit_other"]))
        rt = r["key"].split("@")[1]
        wall[(rt, "07b")] += r["wall"][0] or 0; wall[(rt, "07a")] += r["wall"][1] or 0
        for i, c in r["calls"].items():
            k = c["class"]; byclass[(rt, c["kind"], k)] += 1; T["calls"] += 1
            five[(k, c["five_equal"])] += 1
            if k == "untouched" and not c["five_equal"]:
                bad.append((r["key"], i, c["five_differ"]))
            if k == "untouched" and c["other"]:
                strict_other.append((r["key"], i, c["other"]))
            if c["location"] and not c["other"]:
                loc_only[k] += 1
            conv[(k, tuple(c["converged"]))] += 1
            regions[(k, "07b")] += c["regions"][0]; regions[(k, "07a")] += c["regions"][1]
            refusals[(rt, c["kind"], "edge")] += c["refusals"][0]; refusals[(rt, c["kind"], "range")] += c["refusals"][1]
            refusals[(rt, c["kind"], "calls_edge")] += c["refusals"][0] > 0
            refusals[(rt, c["kind"], "calls_range")] += c["refusals"][1] > 0
            depth[(k, "deeper" if c["max_depth"][0] > c["max_depth"][1] else "shallower" if c["max_depth"][0] < c["max_depth"][1] else "same")] += 1
    summary = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "indexb_sha256": ib["sha256"], "indexa_sha256": ia["sha256"],
        "totals": dict(T), "units_not_done": not_done,
        "unit_wall_s_sum": {str(k): v for k, v in sorted(wall.items(), key=str)},
        "class_counts_by_rtol_kind": {str(k): v for k, v in sorted(byclass.items(), key=str)},
        "five_equal_by_class": {str(k): v for k, v in sorted(five.items(), key=str)},
        "untouched_five_differ": len(bad), "untouched_five_differ_list": bad[:200],
        "untouched_with_other_record_differences": len(strict_other), "untouched_other_examples": strict_other[:50],
        "location_only_by_class": dict(loc_only),
        "converged_07b_07a_by_class": {str(k): v for k, v in sorted(conv.items(), key=str)},
        "regions_by_class": {str(k): v for k, v in sorted(regions.items(), key=str)},
        "refusals": {str(k): v for k, v in sorted(refusals.items(), key=str)},
        "max_depth_vs_07a_by_class": {str(k): v for k, v in sorted(depth.items(), key=str)},
        "only_h_07b_names": dict(only_h),
        "unit_other_paths_units": dict(unit_other_all.most_common(60)),
        "all_untouched_units_with_unit_other": len(unit_other_untouched), "examples": unit_other_untouched[:20],
    }
    Path(a.out_json).write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("unit_other_paths_units",)}, indent=1, default=str)[:15000])
    print("unit_other paths (units):")
    for p, n in unit_other_all.most_common(60):
        print(f"  {n:6d} {p}")


if __name__ == "__main__":
    main()
