"""Orchestrator's own stop-3 reader (07c Part C), independent of the census agent's scripts.

usage: stop3_y1_y8.py <07b checkpoint.jsonl> <07c checkpoint.jsonl> <(S) replay dir> <out dir>

1. Streams both checkpoints. For every Levin call (group `calls`, analytic `other_calls`) keeps
   (value hex, abserr hex, converged, num_regions, max_depth) and, for 07c, the three chase counts.
   Writes compact per-call files calls_07b.jsonl / calls_07c.jsonl for the later review.
2. Y1: every 07c call with num_chase_stops == 0 equals 07b's in the five outputs (hex); counts by rtol
   and class; units untouched in every call compared in hex total and total_abserr.
3. converged against 07b: newly unconverged / newly converged, by rtol and class.
4. Y8: each replay record whose census is '07b census' is located in 07b's unit by its recorded
   07b (value hex, abserr hex) (must be unique); 07c's call at that key must equal the replay in the
   five outputs.
Floats compared by hex string only.
"""
import collections, hashlib, json, os, sys

print("script", hashlib.sha256(open(__file__, "rb").read()).hexdigest(), flush=True)
P07B, P07C, SDIR, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)


def hx(x):
    if isinstance(x, dict):
        return x.get("hex")
    if isinstance(x, str):
        return x
    return None if x is None else float(x).hex()


def five(c):
    return (hx(c["value"]), hx(c["abserr"]), c["converged"], c["num_regions"], c["max_depth"])


CLASS = {"calls": "group", "other_calls": "analytic"}


def stream(path, keep_chase, outpath):
    units, dup, label_mismatch = {}, [], collections.Counter()
    with open(outpath, "w") as fo:
        for n, line in enumerate(open(path)):
            rec = json.loads(line)
            d = rec["data"]
            u = rec["unit"]
            if u in units:
                dup.append(u)
            cs = {}
            for kind in ("calls", "other_calls"):
                for c in d.get(kind) or []:
                    lab = c.get("notify_label") or ""
                    if (kind == "other_calls") != lab.startswith("analytic"):
                        label_mismatch[kind] += 1
                    t = five(c)
                    extra = None
                    if keep_chase:
                        extra = (c.get("num_chase_stops"), c.get("num_chase_stopped_leaves"),
                                 c.get("num_chase_noise_bisections"))
                    cs[(kind, c["index"])] = (t, extra)
                    row = {"unit": u, "rtol": d["rtol"], "kind": kind, "index": c["index"],
                           "value": t[0], "abserr": t[1], "converged": t[2], "num_regions": t[3],
                           "max_depth": t[4], "phase_limited": c.get("phase_limited"),
                           "num_range_rule_refusals": c.get("num_range_rule_refusals"),
                           "num_edge_rule_refusals": c.get("num_edge_rule_refusals")}
                    if keep_chase:
                        row.update(num_chase_stops=extra[0], num_chase_stopped_leaves=extra[1],
                                   num_chase_noise_bisections=extra[2])
                        if extra[0]:
                            row["chase_07c"] = c.get("chase_07c")
                    fo.write(json.dumps(row) + "\n")
            res = d.get("result") or {}
            units[u] = {"rtol": d["rtol"], "status": d.get("status"), "calls": cs,
                        "tot": (hx(res.get("total")), hx(res.get("total_abserr")))}
            if (n + 1) % 2000 == 0:
                print(" ", os.path.basename(os.path.dirname(path)), n + 1, flush=True)
    return units, dup, label_mismatch


A, dupA, lmA = stream(P07B, False, os.path.join(OUT, "calls_07b.jsonl"))
B, dupB, lmB = stream(P07C, True, os.path.join(OUT, "calls_07c.jsonl"))
rep = {"units_07b": len(A), "units_07c": len(B), "dup_07b": dupA, "dup_07c": dupB,
       "unit_sets_equal": set(A) == set(B),
       "status_not_done_07c": sorted(u for u in B if B[u]["status"] != "done"),
       "label_class_mismatch_07b": dict(lmA), "label_class_mismatch_07c": dict(lmB)}

y1 = collections.defaultdict(collections.Counter)
conv = collections.defaultdict(collections.Counter)
untouched_diff, key_mismatch_units, missing_counts = [], [], 0
newly_unconv, newly_conv = [], []
units_all_untouched = units_all_untouched_tot_equal = 0
units_all_untouched_tot_diff = []
for u, b in B.items():
    a = A[u]
    if set(a["calls"]) != set(b["calls"]):
        key_mismatch_units.append(u)
    all_untouched = True
    for k, (tb, ex) in b["calls"].items():
        cls = (b["rtol"], CLASS[k[0]])
        if ex is None or any(e is None for e in ex):
            missing_counts += 1
        ta = a["calls"].get(k, (None,))[0]
        ncs = ex[0] if ex else None
        if ncs == 0:
            y1[cls]["untouched"] += 1
            if ta == tb:
                y1[cls]["untouched_equal"] += 1
            else:
                untouched_diff.append((u, k, ta, tb))
        else:
            all_untouched = False
            y1[cls]["touched"] += 1
            if ta == tb:
                y1[cls]["touched_equal_five"] += 1
        if ta is not None:
            ca, cb = ta[2], tb[2]
            conv[cls][f"{ca}->{cb}"] += 1
            if ca and not cb:
                newly_unconv.append((u, k))
            if cb and not ca:
                newly_conv.append((u, k))
    if all_untouched:
        units_all_untouched += 1
        if a["tot"] == b["tot"]:
            units_all_untouched_tot_equal += 1
        else:
            units_all_untouched_tot_diff.append(u)

rep["y1"] = {f"{r} {c}": dict(v) for (r, c), v in sorted(y1.items())}
rep["y1_untouched_differing"] = [[u, list(k), ta, tb] for u, k, ta, tb in untouched_diff]
rep["y1_all_untouched_equal"] = not untouched_diff
rep["calls_missing_chase_counts"] = missing_counts
rep["units_with_call_key_mismatch"] = key_mismatch_units
rep["units_untouched_in_every_call"] = units_all_untouched
rep["of_which_totals_equal_hex"] = units_all_untouched_tot_equal
rep["of_which_totals_differ"] = units_all_untouched_tot_diff
rep["converged_transitions"] = {f"{r} {c}": dict(v) for (r, c), v in sorted(conv.items())}
rep["newly_unconverged"] = [[u, list(k)] for u, k in newly_unconv]
rep["newly_converged_count_by_class"] = dict(collections.Counter(
    f"{B[u]['rtol']} {CLASS[k[0]]}" for u, k in newly_conv))

# Y8
y8 = collections.Counter()
y8_fail, located = [], set()
for f in sorted(os.listdir(SDIR)):
    r = json.load(open(os.path.join(SDIR, f)))
    if r.get("census_which") != "07b census":
        y8["skipped_" + str(r.get("census_which"))] += 1
        continue
    u = r["census_unit"]
    cen = r["census"]
    cand = [k for k, (t, _) in A[u]["calls"].items() if t[0] == hx(cen["value"]) and t[1] == hx(cen["abserr"])]
    if len(cand) != 1:
        y8_fail.append((f, "07b match count", len(cand)))
        y8["unlocated"] += 1
        continue
    k = cand[0]
    ta = A[u]["calls"][k][0]
    cen5 = (hx(cen["value"]), hx(cen["abserr"]), cen["converged"], cen["num_regions"], cen["max_depth"])
    if ta != cen5:
        y8_fail.append((f, "07b record differs from replay's census field", ta, cen5))
    rp = r["replay"]
    rp5 = (hx(rp["value"]), hx(rp["abserr"]), rp["converged"], rp["num_regions"], rp["max_depth"])
    tb, ex = B[u]["calls"][k]
    located.add((u, k))
    y8["ids"] += 1
    if tb == rp5:
        y8["equal"] += 1
        if ex[0]:
            y8["equal_and_stop_acted_in_census"] += 1
    else:
        y8_fail.append((f, u, list(k), "07c", tb, "replay", rp5))
rep["y8"] = dict(y8)
rep["y8_distinct_census_calls"] = len(located)
rep["y8_failures"] = y8_fail

json.dump(rep, open(os.path.join(OUT, "stop3_y1_y8.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k not in ("newly_unconverged",)}, indent=1)[:6000])
print("newly unconverged:", len(newly_unconv))
