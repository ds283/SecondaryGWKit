# Orchestrator's review §B.3 check 4 (06l): Part L re-checked by re-solving. For six leaves (three
# first-population slow leaves, two second-population, one 06h carrying leaf), the captured call
# (the agent's state/inputs/<sid>.pkl, checked against 06g's census by the agent's run partL) is
# loaded, and the seven solves P, L, R, LL, LR, RL, RR are re-made here with the tree's
# _adaptive_levin_subregion (midpoints as _check_leaf forms them). From the halves' own solve dicts
# (is_direct, phase_err, phase_err_less_node, node_error_signed) nu_B is formed per prompt 06l
# section 0.2, in exact arithmetic (math.fsum), and B's decision taken; both are compared with the
# agent's table (out/partL_leaves.csv) and, where B's census recorded a trust record at the same span
# in the same call, with that record.
import csv, json, math, sys
from pathlib import Path
V = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06l")
CK = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-06l-census-routeB-20261105T100126/checkpoint.jsonl")
DIAG = V / "review" / "checks" / "c4_diag"
import ray.cloudpickle as cp
import AdaptiveLevin.levin_quadrature as LQ
assert LQ.__file__.startswith("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06l-af4c4dc/"), LQ.__file__
rows = list(csv.DictReader(l for l in open(V / "out/partL_leaves.csv") if not l.startswith("#")))
PICK = [("L-a", "s02", "0.200247"), ("L-a", "s04", "0.999669"), ("L-a", "s09", "1.004082"),
        ("L-a", "s01", "0.689692"), ("L-a", "s01", "1.011298"), ("L-b", "f02", None)]
sel = []
for pop, sid, pre in PICK:
    m = [r for r in rows if r["population"] == pop and r["sid"] == sid and (pre is None or r["d1_over_nuB"].startswith(pre))]
    assert len(m) == 1, (pop, sid, pre, len(m)); sel.append(m[0])
units = {r["unit"] for r in sel}
census = {}
for line in open(CK):
    r = json.loads(line)
    if r["unit"] in units:
        census[r["unit"]] = r["data"]
def ratio(d1, d2):
    if not (math.isfinite(d1) and math.isfinite(d2)): return math.inf
    if d1 == 0.0: return 0.0 if d2 == 0.0 else math.inf
    return d2 / d1
ok = True
for row in sel:
    rec = cp.loads((V / "state/inputs" / f"{row['sid']}.pkl").read_bytes())
    kw = rec["kw"]; A = LQ._Basis_SinCos(rec["theta"]); cache = {}
    def solve(a, b):
        return LQ._adaptive_levin_subregion((a, b), rec["f"], A, id_label="06l-review", chebyshev_order=kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER),
                                            rtol=kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL), notify_label=kw.get("notify_label"), build_p_sample=False,
                                            diagnostics_path=DIAG, vectorize_cache=cache)
    fh = lambda s: float.fromhex(s) if s.startswith(("0x", "-0x")) else float(s)
    a, b = fh(row["start"]), fh(row["end"])
    c = (b + a) / 2.0; cL = (c + a) / 2.0; cR = (b + c) / 2.0
    P, L, R = solve(a, b), solve(a, c), solve(c, b)
    LL, LR, RL, RR = solve(a, cL), solve(cL, c), solve(c, cR), solve(cR, b)
    d1 = abs(P["value"] - (L["value"] + R["value"]))
    d2 = abs((L["value"] + R["value"]) - ((LL["value"] + LR["value"]) + (RL["value"] + RR["value"])))
    r = ratio(d1, d2)
    dP, dL, dR = (bool(x.get("is_direct", False)) for x in (P, L, R))
    terms = [P["phase_err"]]
    for h, d in ((L, dL), (R, dR)):
        if not d: terms.append(h["phase_err"])
    Ecc = []
    if dP:
        for h, d in ((L, dL), (R, dR)):
            if d: terms.append(h["phase_err_less_node"]); Ecc.append(h["node_error_signed"])
        if Ecc: terms.append(abs(math.fsum(Ecc)))
    nuB = math.fsum(terms)
    nu1 = math.fsum([P["phase_err"], L["phase_err"], R["phase_err"]])
    tB = r < 0.25 or (r < 1 and d1 <= nuB); t1 = r < 0.25 or (r < 1 and d1 <= nu1); tg = r < 0.25 or (r < 1 and d1 <= P["phase_err"])
    dec = lambda t: "trusted" if t else "untrusted"
    eq_tab = {"delta1": d1 == float(row["delta1"]), "r": r == float(row["r"]), "phi_P": P["phase_err"] == float(row["phi_P"]),
              "nu_B_rel": abs(nuB - float(row["nu_B"])) / nuB, "dec_routeB": dec(tB) == row["dec_routeB"], "dec_route1": dec(t1) == row["dec_route1"],
              "dec_g": dec(tg) == row["dec_g"], "kinds": [("direct" if x else "Levin") for x in (dP, dL, dR)] == [row["P_type"], row["L_type"], row["R_type"]]}
    ok &= all(v for k, v in eq_tab.items() if k != "nu_B_rel") and eq_tab["nu_B_rel"] < 1e-15
    print(f"== {row['population']} {row['sid']} ({row['unit']} #{row['index']}) depth {row['depth']} [{a!r}, {b!r}] kinds {[('CC' if x else 'Levin') for x in (dP, dL, dR)]}")
    print(f"   r {r:.6g}; delta1/phi_P {d1/P['phase_err']:.6g}; nu_B {nuB:.6e} = phi_P {P['phase_err']:.4e} + " +
          (f"phim_L {L['phase_err_less_node']:.4e} + phim_R {R['phase_err_less_node']:.4e} + |E_L+E_R| {abs(math.fsum(Ecc)):.4e}" if dP and dL and dR else f"terms {terms}")
          + f"; delta1/nu_B {d1/nuB:.6g}; delta1/nu1 {d1/nu1:.6g}")
    print(f"   decisions recomputed: g {dec(tg)}, route1 {dec(t1)}, routeB {dec(tB)}; vs table: {eq_tab}")
    hits = []
    for kind in ("calls", "other_calls"):
        for cc in census[row["unit"]][kind]:
            if cc["index"] == row["index"] and cc.get("route_06l"):
                hits += [x for x in cc["route_06l"]["records"] if x["start"] == a and x["end"] == b]
    for x in hits:
        same = {k: x[k] == v for k, v in (("delta1", d1), ("r", r), ("phi_P", P["phase_err"]))}
        print(f"   B's census record at this span: kind {x['kind']}, nuB rel diff {abs(x['nuB']-nuB)/nuB:.2e}, equal {same}")
    if not hits:
        print("   B's census: no trust record at this span (B's census records only firings, withdrawals and untrusted-vs-route-1/depth_max leaves; its refinement may not reach this span)")
print("ALL AGREE:", ok)
