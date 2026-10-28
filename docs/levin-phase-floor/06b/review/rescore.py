"""Orchestrator's independent re-scoring of 06b's W1, W3, W4 and the depth distribution, from the
two checkpoints (06a and b) and the census's out/w3_regions.csv. Read-only."""

import collections
import csv
import json

A = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
B = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06b-census-20261028T161212/checkpoint.jsonl"
W3 = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06b/census/out/w3_regions.csv"
DEPTH_MAX = 20


def load(p, prefix):
    out = {}
    for line in open(p):
        d = json.loads(line)["data"]
        assert d["unit"].startswith(prefix + "-")
        key = (d["rtol"], d["row"])
        assert key not in out
        out[key] = d
    return out


def h(x):
    return x["hex"] if isinstance(x, dict) else x


a = load(A, "h")
b = load(B, "b")
print("units 06a", len(a), "b", len(b), "same keys", set(a) == set(b))
print("b status", collections.Counter(d["status"] for d in b.values()))
print("b tree_head", collections.Counter(d["tree_head"] for d in b.values()),
      "status_short", collections.Counter(d["tree_status_short"] for d in b.values()))
over = [k for k, d in b.items() if d["status"] != "done"]
print("not done / over budget:", over[:10])
walls = sorted(d["wall_s"] for d in b.values())
print("max unit wall_s b %.1f ; 06a %.1f" % (walls[-1], max(d["wall_s"] for d in a.values())))

for rtol in ("1e-7", "1e-8"):
    keys = [k for k in a if k[0] == rtol]
    calls_np = calls_np_ok = 0
    rows_rule = rows_ok = 0
    w1_bad = []
    xcheck_bad = 0
    pl_calls_a = pl_rows_a = pl_calls_b = 0
    pl_an_a = pl_an_b = 0
    moved = []
    depth_a = collections.Counter()
    depth_b = collections.Counter()
    depth_b_all = collections.Counter()
    for k in keys:
        da, db = a[k], b[k]
        ca, cb = da["calls"], db["calls"]
        assert len(ca) == len(cb), k
        row_pl = False
        for x, y in zip(ca, cb):
            assert x["index"] == y["index"] and x["notify_label"] == y["notify_label"], k
            pla = bool(x["phase_limited"])
            if pla:
                pl_calls_a += 1
                row_pl = True
            if y["phase_limited"]:
                pl_calls_b += 1
            # cross-check: premise asked in b <=> phase-limited in 06a
            if (y["premise_checks"] > 0) != pla:
                xcheck_bad += 1
            same = (h(x["value"]), h(x["abserr"]), x["num_regions"], x["max_depth"]) == (
                h(y["value"]), h(y["abserr"]), y["num_regions"], y["max_depth"])
            if not pla:
                calls_np += 1
                if same:
                    calls_np_ok += 1
                else:
                    w1_bad.append((k, x["index"]))
            mv = (h(x["value"]), h(x["abserr"])) != (h(y["value"]), h(y["abserr"]))
            if mv:
                moved.append((k, "group", y["index"], y["max_depth"]))
        for y in db["other_calls"]:
            if y["premise_checks"] > 0:
                pl_an_a += 1
                row_pl = True
            if y["phase_limited"]:
                pl_an_b += 1
            if y["premise_failed"] > 0:
                moved.append((k, "analytic", y["index"], y["max_depth"], y["notify_label"]))
        if row_pl:
            pl_rows_a += 1
        else:
            rows_rule += 1
            ra, rb = da["result"], db["result"]
            if (h(ra["total"]), h(ra["total_abserr"])) == (h(rb["total"]), h(rb["total_abserr"])):
                rows_ok += 1
            else:
                w1_bad.append((k, "row"))
        depth_a[max([c["max_depth"] for c in ca], default=-1)] += 1
        depth_b[max([c["max_depth"] for c in cb], default=-1)] += 1
        depth_b_all[max([c["max_depth"] for c in cb + db["other_calls"]], default=-1)] += 1
    print(f"\n=== rtol {rtol}: units {len(keys)}")
    print(f"W1 calls not PL in 06a: {calls_np_ok}/{calls_np} equal; rows with no PL call: {rows_ok}/{rows_rule} equal; failures {len(w1_bad)} {w1_bad[:5]}")
    print(f"cross-check (premise asked in b) vs (06a phase_limited) on group calls: disagreements {xcheck_bad}")
    print(f"PL group calls 06a {pl_calls_a} on rows(any PL incl analytic) {pl_rows_a}; PL group calls b {pl_calls_b}; analytic calls reaching old exit {pl_an_a}, PL in b {pl_an_b}")
    w4 = [m for m in moved if m[3] >= DEPTH_MAX]
    print(f"moved calls {len(moved)} (group {sum(m[1]=='group' for m in moved)}, analytic {sum(m[1]=='analytic' for m in moved)}); reaching depth {DEPTH_MAX}: {len(w4)}")
    for m in w4:
        print("   W4 violator:", m, "unit wall_s %.1f" % b[m[0]]["wall_s"])
    print("row depth (group calls) 06a:", dict(sorted(depth_a.items())))
    print("row depth (group calls) b  :", dict(sorted(depth_b.items())))
    print("row depth (group+analytic) b:", dict(sorted(depth_b_all.items())))

# W3, from the hex floors
n = viol = neq = 0
ratios = []
for r in csv.DictReader(open(W3)):
    n += 1
    own = float.fromhex(r["phase_err_own"])
    L = float.fromhex(r["phase_err_L"])
    R = float.fromhex(r["phase_err_R"])
    if r["phase_err_own"] != r["phase_err_stored"]:
        neq += 1
    if not (L + R >= 0.5 * own):
        viol += 1
    ratios.append((L + R) / own)
print(f"\nW3 (csv): regions {n}, violations {viol}, own!=stored {neq}, ratio min {min(ratios):.4f} max {max(ratios):.4f}")
# W3 count of b's PL regions at 1e-7 from the checkpoint's own counters
tot = sum(c["n_phase_limited_regions"] for (rt, _), d in b.items() if rt == "1e-7" for c in d["calls"] + d["other_calls"])
w3n = sum(d["w3"]["n_regions"] for (rt, _), d in b.items() if rt == "1e-7")
w3f = sum(d["w3"]["n_premise_fails"] for (rt, _), d in b.items() if rt == "1e-7")
w3e = sum(d["w3"]["n_errors"] for (rt, _), d in b.items() if rt == "1e-7")
print(f"W3 checkpoint: PL regions counted in calls {tot}; w3.n_regions {w3n}; premise fails {w3f}; errors {w3e}")
