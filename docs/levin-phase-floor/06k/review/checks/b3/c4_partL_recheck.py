# Orchestrator's review §B.3 check 4: Part L re-checked on three of 06j's slow leaves (L-a) and two
# 06h carrying leaves (L-b). For each leaf, from the probe's per-solve numbers (out/partL/<sid>.json):
#   (i)  route 1: nu1 = (phi_P + phi_L) + phi_R, and rule (b) with nu1;
#   (ii) route 2: delta1c = |(V_P - E_P) - (halves - E_L - E_R)|, delta2c likewise with the quarters
#        and E_LL..E_RR, r_c, and rule (b) on them (route 2 = g's trust, else this; literal = this);
#   (iii) route 3: D = max(res, d1)/(1-r) if r < 1 else max(res, d1 + d2), against eta * request,
#        with the request recomputed as max(atol, rtol |v_g|) from 06g's census value of the call;
# the decisions recomputed and compared with the probe's. Independently of the probe:
#   (a) for L-a leaves, delta1, delta2, r, phi_P, V_P and E_P against 06g's census region pickle
#       (checked_leaves_06g, matched by start/end hex);
#   (b) where a route's census recorded a firing at the same span in the same call, its numbers
#       (delta1c, r_c / D) against the recomputation.
# Halves and quarters are recorded as sums, so delta1c and delta2c here differ from the variant's
# (which corrects each solve before summing) by rounding; the relative difference is printed.
import json, pickle, math
V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06k/"
R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/"
G = R + "levin-phase-floor-06g-census-20261101T114315/"
RUNS = {"route1": "levin-phase-floor-06k-census-route1-20261104T103009",
        "route2": "levin-phase-floor-06k-census-route2-20261104T151120",
        "route2lit": "levin-phase-floor-06k-census-route2lit-20261104T185927",
        "route3": "levin-phase-floor-06k-census-route3-20261104T225339"}
PICK = [("s02", 3), ("s01", 1), ("s05", 1), ("c01", 0), ("c07", 0)]
probes = {sid: json.load(open(V + f"out/partL/{sid}.json")) for sid, _ in PICK}
units = {probes[s]["unit"] for s, _ in PICK}
def records(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["unit"] in units:
            out[r["unit"]] = r["data"]
    return out
g = records(G + "checkpoint.jsonl")
routes = {k: records(R + v + "/checkpoint.jsonl") for k, v in RUNS.items()}
def ruleb(r, d1, phi):
    return r < 0.25 or (r < 1.0 and d1 <= phi)
def ratio(d1, d2):
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return float("inf")
    if d1 == 0.0:
        return 0.0 if d2 == 0.0 else float("inf")
    return d2 / d1
def call_of(data, index):
    for kind in ("calls", "other_calls"):
        for c in data[kind]:
            if c["index"] == index:
                return c
def hx(x):
    return float(x).hex()
allok = True
for sid, i in PICK:
    p = probes[sid]
    L = p["leaves"][i]
    E = {k: (0.0 if v is None else v) for k, v in L["E"].items()}  # a solve without E: 0, as the variant
    print(f"== {sid} leaf {i}: unit {p['unit']} call {p['index']} population {L['population']} depth {L['depth']} "
          f"span [{L['start_hex']}, {L['end_hex']}]")
    # request from 06g's census value
    cg = call_of(g[p["unit"]], p["index"])
    vg = None
    for key in ("value", "val", "total"):
        if key in cg:
            vg = cg[key]; break
    vg = float(vg["repr"]) if isinstance(vg, dict) else float(vg)
    req = max(p["atol"], p["rtol"] * abs(vg))
    print(f"   request: recomputed {req:.6e} (06g census value {vg:.6e}); probe {p['request']:.6e}; equal {req == p['request']}")
    nu1 = (L["phi_P"] + L["phi_L"]) + L["phi_R"]
    t_g = ruleb(L["r"], L["delta1"], L["phi_P"])
    t_1 = ruleb(L["r"], L["delta1"], nu1)
    d1c = abs((L["V_P"] - E["P"]) - (L["halves"] - E["L"] - E["R"]))
    d2c = abs((L["halves"] - E["L"] - E["R"]) - (L["quarters"] - E["LL"] - E["LR"] - E["RL"] - E["RR"]))
    rc = ratio(d1c, d2c)
    t_2l = ruleb(rc, d1c, L["phi_P"])
    t_2 = t_g or t_2l
    D = max(L["residual"], L["delta1"]) / (1 - L["r"]) if L["r"] < 1 else max(L["residual"], L["delta1"] + L["delta2"])
    left = (not t_g) and D <= p["eta"] * req
    mine = {"g": "trusted" if t_g else "untrusted", "route1": "trusted" if t_1 else "untrusted",
            "route2": "trusted" if t_2 else "untrusted", "route2_literal": "trusted" if t_2l else "untrusted",
            "route3": "trusted" if t_g else ("left" if left else "untrusted")}
    same = mine == L["decision"]
    allok &= same
    print(f"   r {L['r']:.4g}, delta1/phi_P {L['delta1']/L['phi_P']:.4g}; nu1 {nu1:.6e} (probe {L['nu1']:.6e}, equal {nu1 == L['nu1']}), delta1/nu1 {L['delta1']/nu1:.4g}")
    print(f"   delta1c {d1c:.6e} (probe {L['delta1c']:.6e}, rel diff {abs(d1c-L['delta1c'])/max(L['delta1c'],1e-300):.2e}); "
          f"delta2c {d2c:.6e} (probe {L['delta2c']:.6e}); r_c {rc:.6g} (probe {L['r_c']:.6g}); delta1c/phi_P {d1c/L['phi_P']:.4g}")
    print(f"   D {D:.6e} (probe {L['D']:.6e}, equal {D == L['D']}); D/(eta*request) {D/(p['eta']*req):.4g}")
    print(f"   decisions recomputed {mine}")
    print(f"   decisions of probe   {L['decision']}  -> equal {same}")
    # (a) against 06g's census region pickle
    with open(G + g[p["unit"]]["pickle"], "rb") as f:
        f.seek(g[p["unit"]]["pickle_offset"]); pk = pickle.loads(f.read(g[p["unit"]]["pickle_length"]))
    hit = None
    for cl in pk["checked_leaves_06g"]:
        if cl["index"] != p["index"]:
            continue
        for j in range(len(cl["start"])):
            if hx(cl["start"][j]) == L["start_hex"] and hx(cl["end"][j]) == L["end_hex"]:
                hit = (cl, j)
    if hit is None:
        print("   (a) 06g census: leaf not among the call's checked leaves (expected for an 06h leaf evaluated in 06g's code)")
    else:
        cl, j = hit
        cmp = {"delta1": (cl["delta1"][j], L["delta1"]), "delta2": (cl["delta2"][j], L["delta2"]),
               "r": (cl["convergence_ratio"][j], L["r"]), "phi_P": (cl["phase_err"][j], L["phi_P"]),
               "V_P": (cl["value_own"][j], L["V_P"]), "E_P": (cl["node_error_signed"][j], E["P"])}
        cmp["E_P"] = (cl["node_error_signed"][j], L["E"]["P"])
        # a Levin solve has no node error: NaN in the census region pickle, None in the probe
        eq = {k: (hx(a) == hx(b)) if b is not None else bool(a != a) for k, (a, b) in cmp.items()}
        allok &= all(eq.values())
        print(f"   (a) 06g census checked leaf: equal by hex {eq}; trusted_final {int(cl['trusted_final'][j])}")
    # (b) the route censuses' firings at this span
    for rname, recs in routes.items():
        c = call_of(recs[p["unit"]], p["index"])
        fs = [f for f in (c.get("route_06k") or {}).get("firings", [])
              if hx(f["start"]) == L["start_hex"] and hx(f["end"]) == L["end_hex"]]
        if not fs:
            print(f"   (b) {rname} census: no firing at this span (n_firings in call {(c.get('route_06k') or {}).get('n_firings')})")
            continue
        f = fs[0]
        s = {k: f[k] for k in ("kind", "delta1", "r", "phi_P", "delta1c", "r_c", "D", "request") if k in f}
        print(f"   (b) {rname} census firing: {s}")
        if "delta1c" in f:
            print(f"       delta1c rel diff vs recomputed {abs(f['delta1c']-d1c)/max(d1c,1e-300):.2e}; r_c {f['r_c']:.6g} vs {rc:.6g}")
        if "D" in f:
            print(f"       D vs recomputed: equal {f['D'] == D}; D/(eta*request) {f['D']/(1e-3*f['request']):.4g}")
print("ALL DECISIONS AND CENSUS NUMBERS AGREE:", allok)
