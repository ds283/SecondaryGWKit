# Laptop re-checks of levin-phase-floor 06a (orchestrator note §5), read-only.
import json, gzip, csv, sys
RUNS = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/"
C = RUNS + "levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl"
S = RUNS + "levin-phase-floor-06-wider-20261026T223634/checkpoint.jsonl"
def load(p):
    return {json.loads(l)["data"]["unit"]: json.loads(l)["data"] for l in open(p)}
c, s = load(C), load(S)
hx = lambda x: (x["hex"] if isinstance(x, dict) else float(x).hex()) if x is not None else None
print("census keys:", sorted(next(iter(c.values())).keys()))
print("result keys:", sorted(next(iter(c.values()))["result"].keys())[:40])
def calls(d):
    return [(hx(k.get("value")), hx(k.get("abserr")), k.get("num_regions"), k.get("max_depth")) for k in d["calls"]]
for name, rt in [("mid-G-3", "1e-7"), ("R4", "1e-8")]:
    cu = [d for d in c.values() if name in (d.get("names05") or []) and d["rtol"] == rt]
    assert len(cu) == 1, (name, rt, len(cu))
    cu = cu[0]; su = s[f"s-{rt}-{name}"]
    tot = lambda d: (hx(d["result"]["total"]), hx(d["result"]["total_abserr"]))
    print(name, rt, cu["unit"], "total/abserr census", tot(cu), "06", tot(su), "equal", tot(cu) == tot(su))
    print("  calls", len(cu["calls"]), len(su["calls"]), "every call equal (value, abserr, regions, depth):", calls(cu) == calls(su))
# CSV line against checkpoint: the deepest row at 1e-8
rows = [l for l in gzip.open(sys.argv[1], "rt") if not l.startswith("#")]
print("csv data lines (with header)", len(rows))
r = [x for x in csv.DictReader(l for l in gzip.open(sys.argv[1], "rt") if not l.startswith("#")) if x["rtol"] == "1e-8" and x["name"] == "0-1971"]; r = [{k: r[0][k] for k in ("h_depth", "regions", "total_hex", "total_abserr_hex")}]
print("csv 0-1971 @1e-8:", r)
d = c["h-1e-8-0-1971"]
print("checkpoint: total", hx(d["result"]["total"]), "abserr", hx(d["result"]["total_abserr"]),
      "max depth", max(k["max_depth"] for k in d["calls"]), "regions", sum(k["num_regions"] for k in d["calls"]))
