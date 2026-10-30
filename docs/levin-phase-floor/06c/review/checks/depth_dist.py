"""Orchestrator's descriptive reader (06c Part B review): row depth (max over a row's Levin calls'
max_depth) and region totals, 06b vs g, at each rtol. Read-only."""
import collections, json
R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/"
def rd(p):
    out = {}
    for l in open(p):
        d = json.loads(l).get("data") or {}
        if "calls" not in d: continue
        cs = d["calls"] + d["other_calls"]
        out[(d["rtol"], d["row"])] = (max([c["max_depth"] for c in cs], default=None), sum(c["num_regions"] for c in cs),
                                     sum(c.get("evaluations") or 0 for c in cs), d["wall_s"])
    return out
b = rd(R + "levin-phase-floor-06b-census-20261028T161212/checkpoint.jsonl")
g = rd(R + "levin-phase-floor-06c-census-20261029T130214/checkpoint.jsonl")
bins = [("none", None, None), ("0", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9), ("10-14", 10, 14), ("15-19", 15, 19), ("20", 20, 99)]
for rtol in ("1e-7", "1e-8"):
    print(f"== {rtol}: row depth (rows)          06b      g")
    for name, lo, hi in bins:
        f = (lambda v: v is None) if lo is None else (lambda v, lo=lo, hi=hi: v is not None and lo <= v <= hi)
        print(f"   {name:>6}                      {sum(1 for k,v in b.items() if k[0]==rtol and f(v[0])):>6} {sum(1 for k,v in g.items() if k[0]==rtol and f(v[0])):>6}")
    rb = sum(v[1] for k, v in b.items() if k[0] == rtol); rg = sum(v[1] for k, v in g.items() if k[0] == rtol)
    eb = sum(v[2] for k, v in b.items() if k[0] == rtol); eg = sum(v[2] for k, v in g.items() if k[0] == rtol)
    wb = sum(v[3] for k, v in b.items() if k[0] == rtol); wg = sum(v[3] for k, v in g.items() if k[0] == rtol)
    print(f"   regions 06b {rb} g {rg} (x{rg/rb:.2f}); evaluations 06b {eb} g {eg} (x{eg/eb:.2f}); unit wall sum 06b {wb/3600:.2f} h g {wg/3600:.2f} h")
