"""levin-phase-floor prompt 06l, Part C: route B's census against 06g's and against route 1's, and C1, C2, S1 and K1
scored by the rules as written (prompt 06l section 4). New in 06l, after 06k's analyse_c.py (whose comparisons with
06g are kept here rule for rule, for one route). Read only. All value arithmetic is exact (fractions of the recorded
doubles).

Inputs: the compact extracts of extract_census.py: route B's (state/extract/routeB.pkl, this prompt's), and 06k's
of 06g's census and route 1's (var/levin-phase-floor-06k/state/extract/g.pkl and route1.pkl, read only, each
recording its checkpoint's SHA-256); 06g's v8_calls.json; 06a's 720 (refs_common.scan06a_sp); and 06h's cost as
06k's committed out/partC/summary.json records it (06h's extract was deleted on 2026-11-05).

Against 06g, at each rtol, outside the 720 and apart within it:
- **C1:** V8's lost calls converging under B; what holds back the rest (an untrusted leaf, node_limited, other).
- **C2:** every call 06g converges that B does not, with its cause and its abserr against 06g's; C2 holds if every
  one outside the 720 ends node_limited.
- **Depth:** group calls with a leaf at depth_max.
- **S1 (moves):** every call whose value moves by more than e_B + e_g; rows (shard-serial) and units (rtol, row)
  counted apart; and row totals moving past both total abserrs.
- **Firings** (route B's kind "fired"): by depth, by the parent's and halves' kinds, by delta1/phi_P, by
  delta1/nu_B, by r, and for Clenshaw-Curtis parents with two Clenshaw-Curtis halves by
  delta1/(|E_P| + |E_L + E_R|). **K1:** every firing on a Levin parent outside the 720 has delta1/phi_P < 5.8.
- **Cost:** census wall (manifest created to last heartbeat), unit wall per rtol, group-call regions per rtol.
Against route 1:
- the calls route 1 converges and B does not, with the leaves that hold each under B (B's untrusted leaves at
  depth_max) and B's records of the leaves route 1's yardstick would trust and B's does not (its kind "untrusted"
  with trusted_r1), classed by the parent's kind: a call whose withheld leaves are all on Levin parents is one route
  1 recovered only through Levin parents with a Clenshaw-Curtis half (where B and A agree), so A would not recover it
  either, to the extent the refinements coincide up to the first differing decision;
- the calls B converges and route 1 does not, with what holds route 1 back;
- every call whose value moves by more than e_B + e_route1.
Writes out/partC/routeB.json (with a provenance header of its own), out/partC/summary.json, and prints the verdicts.
Usage: PYTHONPATH=<tree>:<scripts> python -B analyse_c_06l.py
"""

import collections
import datetime
import json
import math
import pickle
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

RTOLS = ("1e-7", "1e-8")
EX_B = RC.ROOT / "state" / "extract" / "routeB.pkl"
EX06K = RC.MAIN / "var" / "levin-phase-floor-06k" / "state" / "extract"
EX_G, EX_R1 = EX06K / "g.pkl", EX06K / "route1.pkl"
SUM06K = RC.G_TREE / "docs" / "levin-phase-floor" / "06k" / "out" / "partC" / "summary.json"
V8 = RC.MAIN / "var" / "levin-phase-floor-06g" / "census" / "out" / "v8_calls.json"
K1_LIMIT = 5.8
CK = {"g": "03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa",
      "route1": "36e3dd4c9125f6ea620b80e1bc15d656827b5dd2e449c983d479d1577c6bed38"}
RB_FIELDS = ("kind", "depth", "at_depth_max", "start", "end", "is_direct", "is_direct_L", "is_direct_R", "r", "delta1",
             "delta2", "phi_P", "phi_L", "phi_R", "phim_L", "phim_R", "E_P", "E_L", "E_R", "nu1", "nuB", "trusted_g",
             "trusted_r1", "residual")


def F(h):
    return Fraction(float.fromhex(h))


def finite(h):
    return h is not None and math.isfinite(float.fromhex(h))


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def wall_of(doc):
    created = doc["manifest"].get("created") or doc["manifest"].get("started")
    end = doc["status"].get("heartbeat")
    return (datetime.datetime.fromisoformat(end) - datetime.datetime.fromisoformat(created)).total_seconds(), created, end


def band(x, edges):
    if x is None or x != x:
        return "nan"
    for e in edges:
        if x < e:
            return f"<{e:g}"
    return f">={edges[-1]:g}"


def calls_of(u):
    return {(c["kind"], c["index"]): c for c in u["calls"]}


def holdback(c):
    if (c.get("num_untrusted_regions") or 0) > 0:
        return "untrusted leaf"
    if c.get("node_limited"):
        return "node_limited"
    return "other"


def kname(d):
    return "CC" if d else "Levin"


def rec_dict(t):
    return dict(zip(RB_FIELDS, t))


def leafinfo(x):
    sab = (abs(x["E_L"]) + abs(x["E_R"])) if (x["E_L"] is not None and x["E_R"] is not None) else None
    return {"depth": x["depth"], "start": x["start"], "end": x["end"], "r": x["r"],
            "d1_over_phiP": x["delta1"] / x["phi_P"] if x["phi_P"] else math.inf,
            "d1_over_nuB": x["delta1"] / x["nuB"] if x["nuB"] else math.inf,
            "d1_over_nu1": x["delta1"] / x["nu1"] if x["nu1"] else math.inf,
            "kinds": f"{kname(x['is_direct'])} / {kname(x['is_direct_L'])}+{kname(x['is_direct_R'])}",
            "E_LR_cancellation": abs(x["E_L"] + x["E_R"]) / sab if sab else None,
            "trusted_r1": x["trusted_r1"], "at_depth_max": x["at_depth_max"]}


def main():
    files = [EX_B, EX_G, EX_R1, V8, SUM06K]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: Part C, route B's census against 06g's and route 1's", files)
    sp = RC.scan06a_sp()
    assert len(sp) == 720
    v8 = json.loads(V8.read_text())
    lost = collections.defaultdict(list)
    for c in v8["lost"]:
        lost[c["rtol"]].append(c)
    B, G, R1 = load(EX_B), load(EX_G), load(EX_R1)
    for n, d in (("g", G), ("route1", R1)):
        if d["checkpoint_sha256"] != CK[n]:
            raise SystemExit(f"{n}'s extract records checkpoint {d['checkpoint_sha256']}, not {CK[n]}")
    s06k = json.loads(SUM06K.read_text())
    out_dir = RC.ROOT / "out" / "partC"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"header": hdr, "checkpoint_sha256": {"routeB": B["checkpoint_sha256"], "g": G["checkpoint_sha256"], "route1": R1["checkpoint_sha256"]},
           "status": B["status"], "units": len(B["units"]),
           "units_not_done": [u for u, x in B["units"].items() if x["status"] != "done"]}
    # ------------------------------------------------------------------ cost
    cost = {}
    for name, doc in (("06g", G), ("route1", R1), ("routeB", B)):
        w, a, b = wall_of(doc)
        per = {}
        for rt in RTOLS:
            us = [u for u in doc["units"].values() if u["rtol"] == rt]
            per[rt] = {"unit_wall_s": sum(u["wall_s"] or 0 for u in us),
                       "group_call_regions": sum(c["num_regions"] or 0 for u in us for c in u["calls"] if c["kind"] == "group")}
        cost[name] = {"census_wall_s": w, "created": a, "last_heartbeat": b, "per_rtol": per,
                      "unit_wall_s": sum(p["unit_wall_s"] for p in per.values()),
                      "group_call_regions": sum(p["group_call_regions"] for p in per.values())}
    hb = s06k["baselines"]["h"]
    cost["06h"] = {"census_wall_s": hb["census_wall_s"], "per_rtol": hb["per_rtol"],
                   "unit_wall_s": sum(hb["per_rtol"][rt]["unit_wall_s"] for rt in RTOLS),
                   "group_call_regions": sum(hb["per_rtol"][rt]["group_call_regions"] for rt in RTOLS),
                   "source": "06k's committed out/partC/summary.json baselines.h"}
    for name in cost:
        cost[name]["unit_wall_ratio_to_06g"] = cost[name]["unit_wall_s"] / cost["06g"]["unit_wall_s"]
        cost[name]["regions_ratio_to_06g"] = cost[name]["group_call_regions"] / cost["06g"]["group_call_regions"]
    res["cost"] = cost
    # ------------------------------------------------------------------ against 06g, and against route 1
    unmatched = []
    depth = {rt: {"outside_720": 0, "in_720": 0} for rt in RTOLS}
    depth_g = {rt: {"outside_720": 0, "in_720": 0} for rt in RTOLS}
    depth_r1 = {rt: {"outside_720": 0, "in_720": 0} for rt in RTOLS}
    moves_g, moves_r1, row_moves, lost_calls, gained_g = [], [], [], [], []
    r1_not_B, B_not_r1 = [], []
    fire = {z: {k: collections.Counter() for k in ("by_depth", "by_kinds", "by_d1_over_phiP", "by_d1_over_nuB", "by_r",
                                                    "cc_cc_by_d1_over_EP_plus_ELR", "levin_parent_by_d1_over_phiP")}
            for z in ("outside_720", "in_720")}
    k1 = {"outside_720": [0, 0], "in_720": [0, 0], "max_levin": None, "max_cc": None, "examples": []}
    n_withdrawn = 0
    n_fire_not_r1 = 0
    for uid, ug in G["units"].items():
        ub, u1 = B["units"].get(uid), R1["units"].get(uid)
        rt = ug["rtol"]
        in720 = rowkey(uid) in sp
        zone = "in_720" if in720 else "outside_720"
        if ub is None or u1 is None:
            unmatched.append([uid, "unit missing"])
            continue
        cg, cb, c1 = calls_of(ug), calls_of(ub), calls_of(u1)
        if set(cg) != set(cb) or set(cg) != set(c1):
            unmatched.append([uid, "call sets differ"])
        for key, c in cb.items():
            g, r1 = cg.get(key), c1.get(key)
            if g is None or g["label"] != c["label"] or g["x_span"] != c["x_span"] or r1 is None or r1["x_span"] != c["x_span"]:
                unmatched.append([uid, key, "no matching 06g or route 1 call"])
                continue
            if c["kind"] == "group":
                depth[rt][zone] += (c["n_at_depth_max"] or 0) >= 1
                depth_g[rt][zone] += (g["n_at_depth_max"] or 0) >= 1
                depth_r1[rt][zone] += (r1["n_at_depth_max"] or 0) >= 1
            rb = c.get("routeB")
            if rb is None:
                raise SystemExit(f"{uid} {key}: no route B record")
            recs = [rec_dict(t) for t in rb["records"]]
            n_withdrawn += rb["n_withdrawn"] or 0
            base = {"unit": uid, "rtol": rt, "row": rowkey(uid), "in_720": in720, "kind": c["kind"], "index": c["index"],
                    "group": c["group"] if c["kind"] == "group" else c["label"]}
            if g["converged"] and not c["converged"]:
                lost_calls.append(base | {"holdback": holdback(c), "node_limited": c.get("node_limited"),
                                          "abserr": [g["abserr"], c["abserr"]],
                                          "abserr_ratio_B_over_g": float(F(c["abserr"]) / F(g["abserr"])) if finite(g["abserr"]) and finite(c["abserr"]) else None,
                                          "regions": [g["num_regions"], c["num_regions"]], "depth": [g["max_depth"], c["max_depth"]],
                                          "n_firings": rb["n_firings"]})
            if c["converged"] and not g["converged"]:
                gained_g.append(base)
            for other, sink in ((g, moves_g), (r1, moves_r1)):
                if all(finite(x) for x in (other["value"], other["abserr"], c["value"], c["abserr"])):
                    d = abs(F(c["value"]) - F(other["value"]))
                    s = F(c["abserr"]) + F(other["abserr"])
                    if d > s:
                        sink.append(base | {"ratio": float(d / s), "dv": float(d), "abserr": [other["abserr"], c["abserr"]],
                                            "regions": [other["num_regions"], c["num_regions"]],
                                            "depth": [other["max_depth"], c["max_depth"]],
                                            "converged": [other["converged"], c["converged"]], "n_firings": rb["n_firings"],
                                            "x_span": c["x_span"]})
            if r1["converged"] and not c["converged"]:
                withheld = [x for x in recs if x["kind"] == "untrusted" and x["trusted_r1"]]
                holding = [x for x in recs if x["kind"] == "untrusted" and x["at_depth_max"]]
                wk = collections.Counter("Levin" if not x["is_direct"] else "CC" for x in withheld)
                cls = ("withheld only on Levin parents" if withheld and set(wk) == {"Levin"} else
                       "withheld only on CC parents" if withheld and set(wk) == {"CC"} else
                       "withheld on both kinds" if withheld else "no withheld leaf recorded")
                r1_not_B.append(base | {"holdback_B": holdback(c), "node_limited_B": c.get("node_limited"),
                                        "abserr": {"route1": r1["abserr"], "B": c["abserr"]},
                                        "regions": [r1["num_regions"], c["num_regions"]], "depth": [r1["max_depth"], c["max_depth"]],
                                        "withheld_by_parent_kind": dict(wk), "class": cls,
                                        "withheld_leaves": [leafinfo(x) for x in withheld][:40],
                                        "holding_leaves": [leafinfo(x) for x in holding][:40],
                                        "n_holding_leaves": len(holding), "n_withheld": len(withheld)})
            if c["converged"] and not r1["converged"]:
                B_not_r1.append(base | {"holdback_route1": holdback(r1), "node_limited_route1": r1.get("node_limited"),
                                        "abserr": {"route1": r1["abserr"], "B": c["abserr"]}})
            for x in recs:
                if x["kind"] != "fired":
                    continue
                n_fire_not_r1 += not x["trusted_r1"]
                ft = fire[zone]
                ratio = x["delta1"] / x["phi_P"] if x["phi_P"] and x["phi_P"] > 0 else math.inf
                ft["by_depth"][x["depth"]] += 1
                ft["by_kinds"][f"{kname(x['is_direct'])} parent / {kname(x['is_direct_L'])}+{kname(x['is_direct_R'])}"] += 1
                ft["by_d1_over_phiP"][band(ratio, (1, 2, 5.8, 25, 1e2, 1e3))] += 1
                ft["by_d1_over_nuB"][band(x["delta1"] / x["nuB"] if x["nuB"] else math.inf, (0.01, 0.1, 0.5, 0.9, 1.0 + 1e-12))] += 1
                ft["by_r"][band(x["r"], (0.25, 0.5, 0.75, 1.0))] += 1
                if x["is_direct"] and x["is_direct_L"] and x["is_direct_R"] and x["E_P"] is not None:
                    den = abs(x["E_P"]) + abs(x["E_L"] + x["E_R"])
                    ft["cc_cc_by_d1_over_EP_plus_ELR"][band(x["delta1"] / den if den else math.inf, (0.5, 1, 1.01, 2, 5.8, 25, 1e2))] += 1
                if not x["is_direct"]:
                    ft["levin_parent_by_d1_over_phiP"][band(ratio, (1, 2, 5.8))] += 1
                    k1[zone][0] += 1
                    if ratio >= K1_LIMIT:
                        k1[zone][1] += 1
                        if len(k1["examples"]) < 20:
                            k1["examples"].append(base | leafinfo(x))
                    k1["max_levin"] = ratio if k1["max_levin"] is None else max(k1["max_levin"], ratio)
                else:
                    k1["max_cc"] = ratio if k1["max_cc"] is None else max(k1["max_cc"], ratio)
        if all(finite(x) for x in (ug["total"], ug["total_abserr"], ub["total"], ub["total_abserr"])):
            d = abs(F(ub["total"]) - F(ug["total"]))
            s = F(ub["total_abserr"]) + F(ug["total_abserr"])
            if d > s:
                row_moves.append({"unit": uid, "rtol": rt, "row": rowkey(uid), "in_720": in720, "ratio": float(d / s)})
    # C1
    c1 = {}
    for rt in RTOLS:
        for zone in ("outside_720", "in_720"):
            L = [c for c in lost[rt] if c["stationary_point_row"] == (zone == "in_720")]
            conv, hbk, conv_r1 = 0, collections.Counter(), 0
            for c in L:
                cbb = calls_of(B["units"][c["unit"]])
                cr = cbb.get(("group", c["g"]["index"])) or cbb.get(("analytic", c["g"]["index"]))
                c11 = calls_of(R1["units"][c["unit"]])
                cr1 = c11.get(("group", c["g"]["index"])) or c11.get(("analytic", c["g"]["index"]))
                if cr is None or cr["label"] != c["g"]["notify_label"]:
                    hbk["UNMATCHED"] += 1
                    continue
                conv_r1 += bool(cr1 and cr1["converged"])
                if cr["converged"]:
                    conv += 1
                else:
                    hbk[holdback(cr)] += 1
            c1[f"{rt}/{zone}"] = {"lost_under_g": len(L), "converge_under_B": conv, "fraction": conv / len(L) if L else None,
                                  "held_back": dict(hbk), "converge_under_route1": conv_r1}
    res["C1_detail"] = c1
    res["C1"] = "holds" if all((c1[f"{rt}/outside_720"]["fraction"] or 0) >= 0.9 for rt in RTOLS) else "fails"
    lost_out = [c for c in lost_calls if not c["in_720"]]
    res["C2"] = "holds" if all(c["holdback"] == "node_limited" for c in lost_out) else "fails"
    res["C2_detail"] = {"lost_outside_720": len(lost_out), "lost_in_720": len(lost_calls) - len(lost_out),
                        "by_rtol_kind": dict(collections.Counter(f"{c['rtol']}/{c['kind']}" for c in lost_out)),
                        "holdback": dict(collections.Counter(c["holdback"] for c in lost_out)),
                        "holdback_in_720": dict(collections.Counter(c["holdback"] for c in lost_calls if c["in_720"])),
                        "calls_outside_720": lost_out, "calls_in_720": [c for c in lost_calls if c["in_720"]][:100]}
    res["gained_against_06g"] = {"outside_720": sum(1 for c in gained_g if not c["in_720"]), "in_720": sum(1 for c in gained_g if c["in_720"])}
    mv_out = sorted([m for m in moves_g if not m["in_720"]], key=lambda m: -m["ratio"])
    res["S1"] = "holds" if not mv_out else "fails"
    res["S1_detail"] = {"calls_moving_outside_720": len(mv_out), "calls_moving_in_720": len(moves_g) - len(mv_out),
                        "units_outside": len({(m["rtol"], m["row"]) for m in mv_out}),
                        "rows_outside_shard_serial": len({m["row"] for m in mv_out}),
                        "units_in_720": len({(m["rtol"], m["row"]) for m in moves_g if m["in_720"]}),
                        "rows_in_720_shard_serial": len({m["row"] for m in moves_g if m["in_720"]}),
                        "max_ratio_outside": mv_out[0]["ratio"] if mv_out else None, "calls_outside_720": mv_out,
                        "calls_in_720": sorted([m for m in moves_g if m["in_720"]], key=lambda m: -m["ratio"]),
                        "row_totals_moving_outside_720": sorted([r for r in row_moves if not r["in_720"]], key=lambda r: -r["ratio"]),
                        "row_totals_moving_in_720": sorted([r for r in row_moves if r["in_720"]], key=lambda r: -r["ratio"])}
    res["depth"] = {"routeB": depth, "06g": depth_g, "route1": depth_r1}
    res["firings"] = {z: {k: dict(sorted(v.items(), key=lambda kv: str(kv[0]))) for k, v in ft.items()} for z, ft in fire.items()}
    res["n_withdrawn"] = n_withdrawn
    res["n_fired_not_trusted_by_route1_yardstick"] = n_fire_not_r1
    res["K1"] = "holds" if k1["outside_720"][1] == 0 else "fails"
    res["K1_detail"] = {"levin_parent_firings_outside_720": k1["outside_720"][0], "of_them_ge_5.8_outside": k1["outside_720"][1],
                        "levin_parent_firings_in_720": k1["in_720"][0], "of_them_ge_5.8_in_720": k1["in_720"][1],
                        "max_d1_over_phiP_levin_parent": k1["max_levin"], "max_d1_over_phiP_cc_parent": k1["max_cc"],
                        "examples": k1["examples"]}
    # against route 1
    def split(lst, rt=None):
        return [c for c in lst if not c["in_720"] and (rt is None or c["rtol"] == rt)]
    res["against_route1"] = {
        "route1_converges_B_not": {
            "outside_720_by_rtol": {rt: len(split(r1_not_B, rt)) for rt in RTOLS},
            "in_720": sum(1 for c in r1_not_B if c["in_720"]),
            "outside_by_class": dict(collections.Counter(c["class"] for c in split(r1_not_B))),
            "outside_by_holdback_B": dict(collections.Counter(c["holdback_B"] for c in split(r1_not_B))),
            "in_720_by_class": dict(collections.Counter(c["class"] for c in r1_not_B if c["in_720"])),
            "calls_outside_720": split(r1_not_B),
        },
        "B_converges_route1_not": {
            "outside_720_by_rtol": {rt: len(split(B_not_r1, rt)) for rt in RTOLS},
            "in_720": sum(1 for c in B_not_r1 if c["in_720"]),
            "outside_by_holdback_route1": dict(collections.Counter(c["holdback_route1"] for c in split(B_not_r1))),
            "calls_outside_720": split(B_not_r1),
        },
        "moves_vs_route1": {
            "calls_outside_720": len(split(moves_r1)), "calls_in_720": sum(1 for m in moves_r1 if m["in_720"]),
            "units_outside": len({(m["rtol"], m["row"]) for m in split(moves_r1)}),
            "rows_outside_shard_serial": len({m["row"] for m in split(moves_r1)}),
            "max_ratio_outside": max((m["ratio"] for m in split(moves_r1)), default=None),
            "list_outside": sorted(split(moves_r1), key=lambda m: -m["ratio"]),
            "list_in_720": sorted([m for m in moves_r1 if m["in_720"]], key=lambda m: -m["ratio"]),
        },
    }
    res["n_unmatched"] = len(unmatched)
    res["unmatched"] = unmatched[:50]
    p = out_dir / "routeB.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    summ = {k: v for k, v in res.items() if k not in ("C2_detail", "S1_detail", "against_route1", "K1_detail")}
    summ["C2_detail"] = {k: v for k, v in res["C2_detail"].items() if not k.startswith("calls")}
    summ["S1_detail"] = {k: v for k, v in res["S1_detail"].items() if not k.startswith("calls")}
    summ["K1_detail"] = {k: v for k, v in res["K1_detail"].items() if k != "examples"}
    summ["against_route1"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith(("calls", "list"))} for k, v in res["against_route1"].items()}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: v for k, v in summ.items() if k != "header"}, indent=1, default=str))
    print(f"wrote {p} sha256 {RC.sha256(p)}; {out_dir / 'summary.json'} sha256 {RC.sha256(out_dir / 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
