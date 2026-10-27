import json, sys
R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/"
def load(run, unit):
    for l in open(R + run + "/checkpoint.jsonl"):
        d = json.loads(l)
        if d["unit"] == unit:
            return d["data"]
    raise SystemExit(f"no {unit} in {run}")
row = sys.argv[1]
a = load("levin-phase-floor-06-wider-20261026T223634", f"t5-1e-7-{row}")
b = load("levin-phase-floor-05a-wider-20261026T123047", f"t-1e-7-{row}")
print("06 t5 tree", a["tree_head"][:7], "status", a["status"], "| 05a t tree", b["tree_head"][:7], "status", b["status"])
def hx(x):
    if isinstance(x, dict):
        return x["hex"]
    return x if x is None or isinstance(x, str) else float.hex(float(x))
ok = True
res_a, res_b = a["result"], b["result"]
for key in ("total", "total_abserr"):
    va, vb = res_a.get(key), res_b.get(key)
    same = hx(va) == hx(vb)
    ok &= same
    print(key, hx(va), hx(vb), same)
ca, cb = a["calls"], b["calls"]
print("calls", len(ca), len(cb))
ok &= len(ca) == len(cb)
for i, (x, y) in enumerate(zip(ca, cb)):
    for key in ("value", "abserr", "num_regions", "max_depth", "converged", "x_span", "group"):
        same = hx(x.get(key)) == hx(y.get(key)) if key in ("value", "abserr") else x.get(key) == y.get(key)
        n_compared = globals().get("n_compared", 0) + 1; globals()["n_compared"] = n_compared
        ok &= same
        if not same:
            print("  DIFF call", i, key, x.get(key), y.get(key))
print("call fields compared", globals().get("n_compared", 0)); print("ALL EQUAL" if ok else "DIFFERENCES")
