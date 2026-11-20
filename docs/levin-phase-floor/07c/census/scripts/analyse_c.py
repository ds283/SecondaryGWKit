"""levin-phase-floor prompt 06k, Part C: each route's census against 06g's (06g's own analyses, v8_06g.py and
values_beyond06g.py, rerun with the route in the place of 06g's code, on the compact extracts of
extract_census.py), and C1, C2, S1 and K1 scored by the rules as written (state/predictions_and_rules_*.md
section 1). New in 06k. Read only. All value arithmetic is exact (fractions of the recorded doubles).

Per route (route1, route2, route2lit, route3), against 06g (g), at each rtol, outside the 720 and apart within it:
- **Convergence (C1):** V8's lost calls (06g's v8_calls.json, ``lost``): how many converge under the route; what
  holds back the others: an untrusted leaf (for route 3, one not left), ``node_limited``, or something else.
- **Lost (C2):** every call 06g converges and the route does not (group and analytic calls).
- **Depth:** group calls with a leaf at depth_max (V5's count), and those calls' convergence.
- **Moves (S1):** every call whose value moves by more than e_route + e_g (V2's measure), and every row whose
  ``total`` moves past the sum of the two total abserrs.
- **Firings:** by depth, delta1/phi_P and r (and by kind for route 2 literal; D against the request for route 3);
  K1 (routes 1, 2, 2 literal): every firing that trusts a leaf 06g's rule finds untrusted has delta1/phi_P < 5.8.
- **Cost:** census wall from each run's manifest and status, unit wall per rtol, group-call regions per rtol;
  06g's and 06h's beside.
Writes out/partC/<route>.json, out/partC/summary.json and out/partC/tables.md.
Usage: python -B analyse_c.py
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

ROUTES = ("route1", "route2", "route2lit", "route3")
RTOLS = ("1e-7", "1e-8")
EX = RC.ROOT / "state" / "extract"
V8 = RC.MAIN / "var" / "levin-phase-floor-06g" / "census" / "out" / "v8_calls.json"
K1_LIMIT = 5.8


def F(h):
    return Fraction(float.fromhex(h))


def finite(h):
    if h is None:
        return False
    x = float.fromhex(h)
    return math.isfinite(x)


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def load(name):
    with open(EX / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def wall_of(doc):
    created = doc["manifest"].get("created") or doc["manifest"].get("started")
    end = doc["status"].get("heartbeat")
    try:
        t0 = datetime.datetime.fromisoformat(created)
        t1 = datetime.datetime.fromisoformat(end)
        return (t1 - t0).total_seconds(), created, end
    except Exception:
        return None, created, end


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
    """What holds a route's unconverged call back."""
    r = c.get("route") or {}
    n_untr = c.get("num_untrusted_regions") or 0
    if "n_untrusted_not_left" in r:
        n_untr = r["n_untrusted_not_left"]
    if n_untr > 0:
        return "untrusted leaf"
    if c.get("node_limited"):
        return "node_limited"
    return "other"


def main():
    files = [EX / f"{n}.pkl" for n in ("g", "h") + ROUTES] + [V8]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06k: Part C, each route's census against 06g's", files)
    sp = RC.scan06a_sp()
    assert len(sp) == 720
    v8 = json.loads(V8.read_text())
    lost = collections.defaultdict(list)
    for c in v8["lost"]:
        lost[c["rtol"]].append(c)
    G = load("g")
    Hh = load("h")
    out_dir = RC.ROOT / "out" / "partC"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"header": hdr, "routes": {}, "baselines": {}}
    for name, doc in (("g", G), ("h", Hh)):
        w, a, b = wall_of(doc)
        per = {}
        for rt in RTOLS:
            us = [u for u in doc["units"].values() if u["rtol"] == rt]
            per[rt] = {"unit_wall_s": sum(u["wall_s"] or 0 for u in us),
                       "group_call_regions": sum(c["num_regions"] or 0 for u in us for c in u["calls"] if c["kind"] == "group"),
                       "group_calls_at_depth_max_outside_720": sum(1 for u in us if rowkey(u["uid"]) not in sp for c in u["calls"]
                                                                   if c["kind"] == "group" and (c["n_at_depth_max"] or 0) >= 1),
                       "group_calls_at_depth_max_in_720": sum(1 for u in us if rowkey(u["uid"]) in sp for c in u["calls"]
                                                              if c["kind"] == "group" and (c["n_at_depth_max"] or 0) >= 1)}
        summary["baselines"][name] = {"census_wall_s": w, "created": a, "last_heartbeat": b, "checkpoint_sha256": doc["checkpoint_sha256"], "per_rtol": per}
    for route in ROUTES:
        R = load(route)
        res = {"checkpoint_sha256": R["checkpoint_sha256"], "status": R["status"]}
        w, a, b = wall_of(R)
        res["census_wall_s"], res["created"], res["last_heartbeat"] = w, a, b
        res["units_not_done"] = [u for u, x in R["units"].items() if x["status"] != "done"]
        unmatched = []
        per = {}
        moves, row_moves, lost_calls, gained_calls = [], [], [], []
        fire_tab = {s: collections.Counter() for s in ("by_depth", "by_d1_over_phiP", "by_r", "by_kind", "by_D_over_cap")}
        k1 = {"outside_720": [0, 0], "in_720": [0, 0], "max_ratio": None, "examples": []}
        for rt in RTOLS:
            per[rt] = {"unit_wall_s": 0.0, "group_call_regions": 0, "depth_max": {"outside_720": 0, "in_720": 0},
                       "depth_max_converged": {"outside_720": 0, "in_720": 0}}
        for uid, ug in G["units"].items():
            ur = R["units"].get(uid)
            rt = ug["rtol"]
            in720 = rowkey(uid) in sp
            zone = "in_720" if in720 else "outside_720"
            if ur is None:
                unmatched.append([uid, "unit missing"])
                continue
            per[rt]["unit_wall_s"] += ur["wall_s"] or 0
            cg, cr = calls_of(ug), calls_of(ur)
            if set(cg) != set(cr):
                unmatched.append([uid, "call sets differ"])
            for key, c in cr.items():
                g = cg.get(key)
                if g is None or g["label"] != c["label"] or g["x_span"] != c["x_span"]:
                    unmatched.append([uid, key, "no matching 06g call"])
                    continue
                if c["kind"] == "group":
                    per[rt]["group_call_regions"] += c["num_regions"] or 0
                    if (c["n_at_depth_max"] or 0) >= 1:
                        per[rt]["depth_max"][zone] += 1
                        per[rt]["depth_max_converged"][zone] += bool(c["converged"])
                rec = {"unit": uid, "rtol": rt, "row": rowkey(uid), "in_720": in720, "kind": c["kind"], "index": c["index"],
                       "group": c["group"] if c["kind"] == "group" else c["label"]}
                if g["converged"] and not c["converged"]:
                    lost_calls.append(rec | {"holdback": holdback(c), "regions": [g["num_regions"], c["num_regions"]],
                                              "depth": [g["max_depth"], c["max_depth"]], "abserr": [g["abserr"], c["abserr"]],
                                              "n_firings": (c.get("route") or {}).get("n_firings")})
                if c["converged"] and not g["converged"]:
                    gained_calls.append(rec)
                if all(finite(x) for x in (g["value"], g["abserr"], c["value"], c["abserr"])):
                    d = abs(F(c["value"]) - F(g["value"]))
                    s = F(c["abserr"]) + F(g["abserr"])
                    if d > s:
                        moves.append(rec | {"ratio": float(d / s), "dv": float(d), "abserr": [g["abserr"], c["abserr"]],
                                            "regions": [g["num_regions"], c["num_regions"]], "depth": [g["max_depth"], c["max_depth"]],
                                            "converged": [g["converged"], c["converged"]], "n_firings": (c.get("route") or {}).get("n_firings"),
                                            "x_span": c["x_span"]})
                for f in ((c.get("route") or {}).get("firings") or []):
                    kind, depth, at_cap, r, d1, phi, D, cap, is_direct, trusted_g = f
                    ratio = d1 / phi if phi and phi > 0 else math.inf
                    fire_tab["by_depth"][depth] += 1
                    fire_tab["by_d1_over_phiP"][band(ratio, (1, 2, 5.8, 25, 1e3, 1e7))] += 1
                    fire_tab["by_r"][band(r, (0.25, 0.5, 0.75, 1.0))] += 1
                    fire_tab["by_kind"][kind or ("left" if route == "route3" else "trusted")] += 1
                    if D is not None and cap:
                        fire_tab["by_D_over_cap"][band(D / cap, (1e-6, 1e-3, 1e-1, 1.0000001))] += 1
                    adds_trust = route in ("route1", "route2") or kind == "trust_added"
                    if route != "route3" and adds_trust:
                        k1[zone][0] += 1
                        if ratio >= K1_LIMIT:
                            k1[zone][1] += 1
                            if len(k1["examples"]) < 20:
                                k1["examples"].append(rec | {"depth": depth, "r": r, "d1_over_phiP": ratio, "is_direct": is_direct})
                        k1["max_ratio"] = ratio if k1["max_ratio"] is None else max(k1["max_ratio"], ratio)
            if all(finite(x) for x in (ug["total"], ug["total_abserr"], ur["total"], ur["total_abserr"])):
                d = abs(F(ur["total"]) - F(ug["total"]))
                s = F(ur["total_abserr"]) + F(ug["total_abserr"])
                if d > s:
                    row_moves.append({"unit": uid, "rtol": rt, "row": rowkey(uid), "in_720": in720, "ratio": float(d / s)})
        # C1: V8's lost calls
        c1 = {}
        for rt in RTOLS:
            for zone in ("outside_720", "in_720"):
                L = [c for c in lost[rt] if c["stationary_point_row"] == (zone == "in_720")]
                conv, hb = 0, collections.Counter()
                for c in L:
                    ur = R["units"][c["unit"]]
                    cr = calls_of(ur).get(("group", c["g"]["index"])) or calls_of(ur).get(("analytic", c["g"]["index"]))
                    if cr is None or cr["label"] != c["g"]["notify_label"]:
                        hb["UNMATCHED"] += 1
                        continue
                    if cr["converged"]:
                        conv += 1
                    else:
                        hb[holdback(cr)] += 1
                c1[f"{rt}/{zone}"] = {"lost_under_g": len(L), "converge_under_route": conv,
                                      "fraction": conv / len(L) if L else None, "held_back": dict(hb)}
        res["C1_detail"] = c1
        frac = [c1[f"{rt}/outside_720"]["fraction"] for rt in RTOLS]
        res["C1"] = "holds" if all(f is not None and f >= 0.9 for f in frac) else "fails"
        lost_out = [c for c in lost_calls if not c["in_720"]]
        res["C2"] = "holds" if not lost_out else "fails"
        res["C2_detail"] = {"lost_outside_720": len(lost_out), "lost_in_720": len(lost_calls) - len(lost_out),
                            "by_rtol_kind": dict(collections.Counter(f"{c['rtol']}/{c['kind']}" for c in lost_out)),
                            "holdback": dict(collections.Counter(c["holdback"] for c in lost_out)), "calls": lost_calls[:400]}
        res["gained"] = {"outside_720": sum(1 for c in gained_calls if not c["in_720"]), "in_720": sum(1 for c in gained_calls if c["in_720"]),
                         "by_rtol_kind_outside": dict(collections.Counter(f"{c['rtol']}/{c['kind']}" for c in gained_calls if not c["in_720"]))}
        mv_out = [m for m in moves if not m["in_720"]]
        mv_out.sort(key=lambda m: -m["ratio"])
        res["S1"] = "holds" if not mv_out else "fails"
        res["S1_detail"] = {"calls_moving_outside_720": len(mv_out), "calls_moving_in_720": len(moves) - len(mv_out),
                            "rows_outside": len({(m["rtol"], m["row"]) for m in mv_out}),
                            "by_rtol_kind": dict(collections.Counter(f"{m['rtol']}/{m['kind']}" for m in mv_out)),
                            "by_group": dict(collections.Counter(m["group"] for m in mv_out).most_common(20)),
                            "max_ratio": mv_out[0]["ratio"] if mv_out else None,
                            "calls": mv_out[:400],
                            "row_totals_moving_outside_720": sorted([r for r in row_moves if not r["in_720"]], key=lambda r: -r["ratio"]),
                            "row_totals_moving_in_720": len([r for r in row_moves if r["in_720"]])}
        res["all_moves"] = moves
        res["firings"] = {k: dict(sorted(v.items(), key=lambda kv: str(kv[0]))) for k, v in fire_tab.items()}
        if route != "route3":
            res["K1"] = "holds" if k1["outside_720"][1] + k1["in_720"][1] == 0 else "fails"
            res["K1_detail"] = {"trust_firings_outside_720": k1["outside_720"][0], "of_them_d1_over_phiP_ge_5.8_outside": k1["outside_720"][1],
                                "trust_firings_in_720": k1["in_720"][0], "of_them_ge_5.8_in_720": k1["in_720"][1],
                                "max_d1_over_phiP": k1["max_ratio"], "examples": k1["examples"]}
        else:
            res["K1"] = "not scored (route 3)"
        res["per_rtol"] = per
        res["unmatched"] = unmatched[:50]
        res["n_unmatched"] = len(unmatched)
        (out_dir / f"{route}.json").write_text(json.dumps(res, indent=1, default=str))
        summary["routes"][route] = {k: v for k, v in res.items() if k not in ("all_moves",)}
        summary["routes"][route]["S1_detail"] = {k: v for k, v in res["S1_detail"].items() if k != "calls"} | {"top_calls": res["S1_detail"]["calls"][:15]}
        summary["routes"][route]["C2_detail"] = {k: v for k, v in res["C2_detail"].items() if k != "calls"} | {"calls": res["C2_detail"]["calls"][:40]}
        print(f"{route}: C1 {res['C1']} {json.dumps({k: [v['converge_under_route'], v['lost_under_g'], v['held_back']] for k, v in c1.items()})}")
        print(f"   C2 {res['C2']} {json.dumps({k: v for k, v in res['C2_detail'].items() if k != 'calls'})}")
        print(f"   S1 {res['S1']} calls {len(mv_out)} rows {res['S1_detail']['rows_outside']} max {res['S1_detail']['max_ratio']}; row totals {len(res['S1_detail']['row_totals_moving_outside_720'])}")
        print(f"   K1 {res['K1']} {json.dumps(res.get('K1_detail', {}).get('trust_firings_outside_720'))} / ge5.8 {json.dumps(res.get('K1_detail', {}).get('of_them_d1_over_phiP_ge_5.8_outside'))}")
        print(f"   wall {w}; per rtol {json.dumps(per)}; unmatched {len(unmatched)}")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"wrote {out_dir / 'summary.json'} sha256 {RC.sha256(out_dir / 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
