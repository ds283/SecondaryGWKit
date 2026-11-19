"""levin-phase-floor prompt 07c Part A, section 2.2 (D1): the chase, from the instrument's records. New in 07c.

Reads out/replays/instr/<sid>.instrument.json (make_instr_07c.py's records: seed, check, pop, split, stop_window, final)
for every sid given, and 07b's census agent's descent analysis (docs/levin-phase-floor/07b/census/out/scores/summary.json
``newly_unconverged``) for the chase calls, read only. Per call:
- **the descent analysis:** each returned leaf at depth_max (20), its route (``levin``, ``cc_gate`` (the C2 gate's
  fallback), ``window`` (a window piece), ``guarded`` (a guarded leaf re-solved at the cap)), its ancestors (by the split
  records), whether it descends from a leaf the rate rules refused (a ``check`` record with ``rate_rule.refused``, on an
  ancestor; the leaf itself refused is reported apart), the nearest and the shallowest such ancestor's depth;
- **every bisection below a refused leaf** (a split whose parent is a strict descendant of a refused leaf): the leaf's route
  and depth; its last trust check before it was chosen (delta1, delta2, r, nu_B, the clause that failed, the rate rules'
  outcome); why it was chosen (``untrusted`` or ``budget``); its residual against its floor; whether the round-off exit was
  considered and, if so, which test refused it (the halving test, the budget clause); and delta1 over the call's request
  then (the leaf's value move against the request). The refused leaves' own bisections are counted apart.
- **calls with no depth-20 leaf:** every returned leaf with its e', floor, residual, trusted, phase_limited, against the
  call's request.

Writes OUT_JSON (per call, and totals). Usage: python -B d1_descent_07c.py OUT_JSON SID [SID ...] | --chase
"""

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f")
SUMMARY = TREE / "docs/levin-phase-floor/07b/census/out/scores/summary.json"
DEPTH_MAX = 20


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def analyse(sid):
    inst = json.loads((ROOT / "out" / "replays" / "instr" / f"{sid}.instrument.json").read_text())
    rep = json.loads((ROOT / "out" / "replays" / "instr" / f"{sid}.json").read_text())
    recs = inst["records"]
    parent, depth_of, route_of = {}, {}, {}
    refused = {}
    last_check = {}
    checks_by_seq = {}
    pops = []
    splits = {}
    finals = []
    for i, r in enumerate(recs):
        k = r["kind"]
        if k == "seed":
            route_of[r["seq"]] = r["route"]
            depth_of[r["seq"]] = 0
        elif k == "check":
            last_check[r["seq"]] = r
            checks_by_seq.setdefault(r["seq"], []).append(r)
            depth_of[r["seq"]] = r["depth"]
            route_of[r["seq"]] = r["route"]
            rr = r.get("rate_rule")
            if rr and rr["refused"]:
                refused.setdefault(r["seq"], {"depth": r["depth"], "rule": rr["rule"], "q": fx(rr["q"])})
        elif k == "pop":
            pops.append((i, r, dict(last_check.get(r["seq"]) or {}) or None))
        elif k == "split":
            splits[r["seq"]] = r
            for ch, rt in ((r["left"], r["left_route"]), (r["right"], r["right_route"])):
                parent[ch] = r["seq"]
                depth_of[ch] = r["depth"] + 1
                route_of[ch] = rt
        elif k == "final":
            finals.append(r)

    def ancestors(s):
        out = []
        while s in parent:
            s = parent[s]
            out.append(s)
        return out

    # descent of depth-20 leaves
    d20 = []
    for r in finals:
        if r["depth"] < DEPTH_MAX:
            continue
        anc = ancestors(r["seq"])
        ref_anc = [a for a in anc if a in refused]
        d20.append({"seq": r["seq"], "route": r["route"], "trusted": r["trusted"],
                    "self_refused": r["seq"] in refused,
                    "descends_from_refused": bool(ref_anc),
                    "nearest_refused_depth": refused[ref_anc[0]]["depth"] if ref_anc else None,
                    "shallowest_refused_depth": min(refused[a]["depth"] for a in ref_anc) if ref_anc else None})
    # bisections below a refused leaf
    below, own = [], []
    for i, r, chk in pops:
        if r["seq"] not in splits:
            # chosen but not bisected (floor-limited), or bisection made after? the split record follows the pop
            outcome = "floor_limited" if (r.get("exit") or {}).get("floor_limited") else "not_split"
        else:
            outcome = "split"
        anc = ancestors(r["seq"])
        strict = any(a in refused for a in anc)
        if not strict and r["seq"] not in refused:
            continue
        ex = r.get("exit")
        if not r["exit_considered"]:
            cause = "residual_above_floor"
        elif ex is None:
            cause = "exit_considered_no_record"
        elif ex["floor_limited"]:
            cause = "floor_limited"
        elif not ex["halving_holds"]:
            cause = "halving_test"
        elif ex["split_could_meet_budget"]:
            cause = "budget_clause"
        else:
            cause = "other"
        req = fx(r["request"])
        row = {"seq": r["seq"], "depth": r["depth"], "route": r["route"], "why": r["why"], "outcome": outcome, "cause": cause,
               "residual": fx(r["residual"]), "floor": fx(r["floor"]), "residual_over_floor": fx(r["residual"]) / fx(r["floor"]) if fx(r["floor"]) else None,
               "request": req, "start": r["start"], "end": r["end"],
               "halving_holds": None if not ex else ex["halving_holds"], "split_could_meet_budget": None if not ex else ex["split_could_meet_budget"]}
        if chk:
            row.update({"delta1": fx(chk["delta1"]), "delta2": fx(chk["delta2"]), "r": fx(chk["r"]), "nu_B": fx(chk["nu_B"]),
                        "clause_failed": chk["clause_failed"], "rate_rule": chk["rate_rule"], "trusted_by_test": chk["trusted_by_test"],
                        "delta1_over_request": fx(chk["delta1"]) / req if req else None,
                        "delta1_over_nuB": fx(chk["delta1"]) / fx(chk["nu_B"]) if fx(chk["nu_B"]) else None,
                        "own_value": chk["own_value"], "resasc": chk.get("resasc"), "floor_less_node": chk.get("floor_less_node")})
            if chk["rate_rule"] and chk["rate_rule"]["refused"]:
                row["clause_failed"] = "rate_rule_" + chk["rate_rule"]["rule"]
        (below if strict else own).append(row)
    # calls with no depth-20 leaf
    no_d20 = None
    if not d20:
        val = fx(rep["replay"]["value"])
        no_d20 = {"abserr": fx(rep["replay"]["abserr"]), "converged": rep["replay"]["converged"],
                  "leaves": [{"seq": r["seq"], "depth": r["depth"], "route": r["route"], "trusted": r["trusted"],
                              "phase_limited": r["phase_limited"], "eprime": fx(r["eprime"]), "floor": fx(r["floor"]),
                              "residual": fx(r["residual"])} for r in finals]}
    nd = {"n_depth20": len(d20), "routes": dict(Counter(x["route"] for x in d20)),
          "descends_from_refused": sum(1 for x in d20 if x["descends_from_refused"]),
          "self_refused": sum(1 for x in d20 if x["self_refused"]),
          "routes_descending": dict(Counter(x["route"] for x in d20 if x["descends_from_refused"])),
          "nearest_refused_depths": dict(Counter(str(x["nearest_refused_depth"]) for x in d20 if x["descends_from_refused"])),
          "shallowest_refused_depths": dict(Counter(str(x["shallowest_refused_depth"]) for x in d20 if x["descends_from_refused"])),
          "min_refused_ancestor_depth_of_depth20": min((x["shallowest_refused_depth"] for x in d20 if x["descends_from_refused"]), default=None),
          "all_depth20_below_refused": bool(d20) and all(x["descends_from_refused"] for x in d20),
          "n_refused": len(refused), "refused_depths": dict(Counter(str(v["depth"]) for v in refused.values())),
          "refused_rules": dict(Counter(v["rule"] for v in refused.values())),
          "converged": rep["replay"]["converged"], "num_regions": rep["replay"]["num_regions"], "wall_s": rep["wall_s"],
          "reproduces_census": rep["reproduces_census"]}
    # every returned Clenshaw-Curtis leaf (any depth) that descends from a refused leaf: D2's chase stratum
    cc_below = [[r["start"], r["end"]] for r in finals if r["route"] != "levin" and any(a in refused for a in ancestors(r["seq"]))]
    nd["n_cc_leaves_below_refused"] = len(cc_below)
    return nd, below, own, no_d20, d20, cc_below


def main():
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists")
    if sys.argv[2] == "--chase":
        man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]
        sids = sorted(s for s, c in man.items() if c["set"] == "chase") + ["a:R2-28", "a:R2-28~o", "r31-3-20363-1"]
    else:
        sids = sys.argv[2:]
    census = {}
    for c in json.loads(SUMMARY.read_text())["newly_unconverged"]["list"]:
        p = c["unit"].split("-")
        census[f"c-{'-'.join(p[1:-2])}-{p[-2]}-{p[-1]}-{c['index']}"] = c
    calls, all_below, all_own, totals = {}, [], [], Counter()
    chase_cc = {}
    for sid in sids:
        nd, below, own, no_d20, d20, cc_below = analyse(sid)
        chase_cc[sid] = cc_below
        cz = census.get(sid)
        if cz is not None:
            nd["census_agent"] = {k: cz[k] for k in ("n_depth20", "routes", "descends_from_refused", "routes_descending",
                                                      "refused_ancestor_depths", "num_range_rule_refusals", "num_edge_rule_refusals")}
            nd["agrees_with_census_agent"] = {
                "n_depth20": nd["n_depth20"] == cz["n_depth20"],
                "descends_from_refused": nd["descends_from_refused"] == cz["descends_from_refused"],
                "routes": {("Clenshaw-Curtis" if k == "cc_gate" else k): v for k, v in nd["routes"].items()} == cz["routes"],
                "refused_depths_nearest": nd["nearest_refused_depths"] == cz["refused_ancestor_depths"],
                "refused_depths_shallowest": nd["shallowest_refused_depths"] == cz["refused_ancestor_depths"]}
        nd["bisections_below_refused"] = len(below)
        nd["bisections_of_refused"] = len(own)
        nd["below_by_cause"] = dict(Counter(f"{x['route']}|{x['cause']}" for x in below))
        nd["no_depth20"] = no_d20
        calls[sid] = nd
        for x in below:
            x["sid"] = sid
        for x in own:
            x["sid"] = sid
        all_below += below
        all_own += own
        totals["calls"] += 1
        totals["all_depth20_below_refused"] += int(nd["all_depth20_below_refused"])
        totals["no_depth20"] += int(nd["n_depth20"] == 0)
        totals["depth20_leaves"] += nd["n_depth20"]
    doc = {"header": {"script": str(Path(__file__).resolve()),
                      "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
                      "summary_sha256": __import__("hashlib").sha256(SUMMARY.read_bytes()).hexdigest()},
           "totals": dict(totals), "calls": calls, "below": all_below, "own": all_own, "chase_cc_leaves": chase_cc}
    out.write_text(json.dumps(doc))
    print(json.dumps(dict(totals)))
    print("below by route|cause:", dict(Counter(f"{x['route']}|{x['cause']}" for x in all_below)))
    print("wrote", out)


if __name__ == "__main__":
    main()
