# Orchestrator's own reading of out/partL_leaves.csv at stop 1: nu_B recomputed from the row's
# floors, E and kinds per prompt 06l section 0.2; the decision; L1, L2 and Gate L3.
import csv, math, collections
F = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06l/out/partL_leaves.csv"
rows = list(csv.DictReader(l for l in open(F) if not l.startswith("#")))
print("rows", len(rows), collections.Counter(r["population"] for r in rows))
print("kinds", collections.Counter((r["P_type"], r["L_type"], r["R_type"]) for r in rows))
f = lambda x: float(x) if x not in ("", "None", "nan") else None
def is_cc(t): return t.strip().lower() in ("cc", "clenshaw-curtis", "direct", "true", "fallback")
mis = collections.Counter(); l3 = collections.Counter(); dec = collections.Counter()
maxratio = 0.0
for r in rows:
    pc, lc, rc = is_cc(r["P_type"]), is_cc(r["L_type"]), is_cc(r["R_type"])
    terms = [f(r["phi_P"])]
    for c, phi in ((lc, "phi_L"), (rc, "phi_R")):
        if not c: terms.append(f(r[phi]))
    if pc:
        Es = []
        for c, pm, E in ((lc, "phim_L", "E_L"), (rc, "phim_R", "E_R")):
            if c: terms.append(f(r[pm])); Es.append(f(r[E]))
        if Es: terms.append(abs(math.fsum(Es)))
    nuB = math.fsum(terms)
    nu1 = math.fsum([f(r["phi_P"]), f(r["phi_L"]), f(r["phi_R"])])
    rr, d1 = f(r["r"]), f(r["delta1"])
    tB = rr < 0.25 or (rr < 1 and d1 <= nuB)
    t1 = rr < 0.25 or (rr < 1 and d1 <= nu1)
    rel = abs(nuB - f(r["nu_B"])) / nuB if nuB else 0.0
    mis["nuB rel diff > 1e-15"] += rel > 1e-15
    mis["dec_routeB disagrees"] += (str(tB) != r["dec_routeB"]) and (("trusted" if tB else "untrusted") != r["dec_routeB"])
    mis["dec_route1 disagrees"] += (str(t1) != r["dec_route1"]) and (("trusted" if t1 else "untrusted") != r["dec_route1"])
    maxratio = max(maxratio, nuB / nu1)
    l3["nuB > nu1(1+1e-12)"] += nuB > nu1 * (1 + 1e-12)
    l3["B trusts, route 1 not"] += tB and not t1
    dec[(r["population"], r.get("sid","")[:0] , tB)] += 1
print("mismatches", dict(mis)); print("L3", dict(l3), "max nuB/nu1", maxratio)
print("decisions", dict(dec))
la = [r for r in rows if r["population"].startswith("L-a") or r["population"] == "La"]
print("example dec values", collections.Counter(r["dec_routeB"] for r in rows))
print("--- L2 (L-a with delta1/phi_P < 2)")
first = [r for r in rows if r["population"] == "L-a" and f(r["delta1"]) / f(r["phi_P"]) < 2]
second = [r for r in rows if r["population"] == "L-a" and f(r["delta1"]) / f(r["phi_P"]) >= 2]
tr = lambda r: r["dec_routeB"] == "trusted"
print("first", len(first), "trusted", sum(map(tr, first)), "| second", len(second), "trusted", sum(map(tr, second)))
un = [r for r in first if not tr(r)]
for r in un:
    EP, EL, ER = f(r["E_P"]), f(r["E_L"]), f(r["E_R"])
    print(f"  {r['sid']:>4} {r['index']:>4} r={f(r['r']):.4f} d1/phiP={f(r['delta1'])/f(r['phi_P']):.4f} d1/nuB={f(r['delta1'])/f(r['nu_B']):.5f}"
          f" opp_sign={EP*(EL+ER) < 0} d1/|EP-(EL+ER)|={f(r['delta1'])/abs(EP-(EL+ER)):.5f} kinds={r['P_type']},{r['L_type']},{r['R_type']}")
print("--- phim vs phi - |E| on CC halves (is phim the half's own phase_err_less_node, not phi-|E|?)")
d = [abs(f(r["phim_L"]) - (f(r["phi_L"]) - abs(f(r["E_L"])))) / f(r["phi_L"]) for r in rows if r["L_type"] == "direct"]
print("  n", len(d), "max rel diff", max(d), "n exactly equal", sum(x == 0 for x in d))
