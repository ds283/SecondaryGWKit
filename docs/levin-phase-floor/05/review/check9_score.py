"""Review check 9 (scratch): re-score L1, L1q, L2, L3 from the wider run's checkpoint and pickles,
against prompt 05 section 6's rules as written."""
import json, pickle, sys
from collections import Counter
RUN = "var/runs/levin-phase-floor-05-wider-20261026T020005"
S = json.load(open("var/levin-phase-floor-05/wider/state/sample.json"))
rows = {r["name"]: r for r in S["rows"]}
units = {}
for l in open(f"{RUN}/checkpoint.jsonl"):
    d = json.loads(l)["data"]
    units[(d["tree"], d["rtol"], d["row"])] = d
def regions(uid):
    rec = pickle.load(open(f"{RUN}/unit_{uid}.pkl", "rb"))
    out = {}
    for g in rec["regions"]:
        key = (g["regime"], g["group"], tuple(x["hex"] if isinstance(x, dict) else float(x).hex() for x in g["x_span"]))
        assert key not in out, key
        out[key] = [int(x) for x in g["depth"]]
    return out
def deep(ds): return sum(1 for x in ds if x >= 18)
def total(d): return float.fromhex(d["result"]["total"]["hex"])
def abserr(d): return float.fromhex(d["result"]["total_abserr"]["hex"])
over = [u for u, d in units.items() if d["status"] != "done"]
print("units", len(units), Counter(d["status"] for d in units.values()), "not done:", over)
L1, L1q, capped = [], [], []
for name, r in rows.items():
    if r.get("role") != "d20": continue
    p = units[("p", "1e-7", name)]; c = units[("c", "1e-7", name)]
    gp = regions(p["unit"]); gc = regions(c["unit"])
    scored = [k for k, ds in gp.items() if deep(ds) >= 10]
    if c["status"] != "done":
        ok1 = ok2 = False
    else:
        # match groups by (regime, label); x_span is the sub-interval, which p and c share
        missing = [k for k in scored if k not in gc]
        ok1 = (not missing) and all(deep(gc[k]) < 10 for k in scored)
        rp = p["result"]["WKB_Levin_data"]["num_regions"]; rc = c["result"]["WKB_Levin_data"]["num_regions"]
        ok2 = rc <= rp / 10
    dp = p["result"]["WKB_Levin_data"]["max_depth"]; dc = c["result"]["WKB_Levin_data"]["max_depth"]
    rp = p["result"]["WKB_Levin_data"]["num_regions"]; rc = c["result"]["WKB_Levin_data"]["num_regions"]
    sp_regions = sum(len(v) for v in gp.values())
    line = (name, r["cls"], dp, rp, dc, rc, round(rc / rp, 4) if rp else None, len(scored), ok1, ok2, ok1 and ok2)
    (L1q if r["cls"] == "Gqr" else L1).append(line)
    if dp == 20: capped.append(line)
hdr = "row cls p_depth p_regions c_depth c_regions c/p n_scored clause1 clause2 rule"
for title, lst in (("L1", L1), ("L1q", L1q), ("capped under p (p max_depth 20), unscored", capped)):
    print(f"\n== {title}: {sum(x[-1] for x in lst)}/{len(lst)} true; verdict {'holds' if all(x[-1] for x in lst) else 'fails'}")
    print("  " + hdr)
    for x in lst: print("  ", *x)
print("\n== L2 (controls): |total_c - total_p| <= total_abserr_p")
ok = []
for name, r in rows.items():
    if r.get("role") != "control": continue
    p = units[("p", "1e-7", name)]; c = units[("c", "1e-7", name)]
    if c["status"] != "done": ok.append(False); print("  ", name, "c not done"); continue
    ratio = abs(total(c) - total(p)) / abserr(p) if abserr(p) else (0.0 if total(c) == total(p) else float("inf"))
    same = p["result"]["total"]["hex"] == c["result"]["total"]["hex"]
    ok.append(ratio <= 1); print("  ", name, "ratio", f"{ratio:.3g}", "bit-identical" if same else "", ratio <= 1)
print(f"  L2: {sum(ok)}/{len(ok)}; verdict {'holds' if all(ok) else 'fails'}")
print("\n== L3: |total_c(1e-7) - total_c(1e-8)| <= total_abserr_c(1e-7), d20 rows with both c units in budget")
ok = []; undecided = []
for name, r in rows.items():
    if r.get("role") != "d20": continue
    a = units[("c", "1e-7", name)]; b = units[("c", "1e-8", name)]
    if a["status"] != "done" or b["status"] != "done": undecided.append(name); continue
    ratio = abs(total(a) - total(b)) / abserr(a)
    ok.append((name, ratio <= 1, ratio))
fails = [(n, round(x, 3)) for n, t, x in ok if not t]
print(f"  {sum(t for _, t, _ in ok)}/{len(ok)} true; excluded (not in budget) {undecided}; failures {fails}; verdict {'holds' if not fails else 'fails'}")
capnames = {x[0] for x in capped}
print("  on capped rows: max ratio", max(x for n, t, x in ok if n in capnames), "all true:", all(t for n, t, x in ok if n in capnames))
