"""levin-phase-floor prompt 07d Part A, section 2.4 (D-T): both routes against the phase variation V, and each threshold's
verdict. New in 07d.

**The bands:** V/6pi in quarters below 1 (the gate strata ``gate_q1``..``gate_q4``, Clenshaw-Curtis leaves of the tree)
and the Levin bands ``band_6_8`` .. ``band_16_20`` (Levin leaves of the tree), each split by kind and rtol, each its fill
prefix. Every leaf was solved both ways on its own span (leafcc_07d.py): Clenshaw-Curtis at the call's order, under the
tree's declaration and each candidate of state/candidates_07d.json; the Levin solve with its own declaration
max(residual, floor) + truncation. Each route is scored against the leaf's reference by state/refs_rule_leaf_07d.txt,
on its own: a route's verdict counts if it is not undecided.
**The table**, per band (pooled over kind and rtol, and by kind) and route: bounded, understates, undecided; D/true error
min, median, max; the true error against the request a leaf of that width would be given in its call (request x width /
call span); evaluations (Clenshaw-Curtis 2N - 1 (+ the candidate's extra); Levin N for its own solve, 2N more for the
halves of its residual). Beside the Levin route, not in its verdict: ``levin_tc``, the solve as the driver's trust
check would declare it (the quarters' sum Q with max(residual, delta1)/(1 - r) if r < 1; V_P with max(residual,
delta1 + delta2) if not).
**Each threshold T** of THRESHOLDS: the route below T is Clenshaw-Curtis and above it Levin; T is "bounded on both
sides" under a declaration if Clenshaw-Curtis understates on no leaf with V < T and the Levin solve understates on no
leaf with V >= T, over every band's leaves (each leaf was solved both ways), among the decided verdicts. Rule A reads no threshold
at 07c's code (``_locate_stationary_points`` reads ``SIX_PI`` only in its docstring; out/cmd/01_gate.txt): it changes
nothing at any T, which the table states.
Writes out/tables/d_t.json and out/tables/d_t.md. Usage: PYTHONPATH=<tree>:<scripts> python -B d_t_table_07d.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402
from d_n_table_07d import prefix_leaves  # noqa: E402
from d_e_table_07d import declaration, verdict  # noqa: E402

fx = float.fromhex
BAND_LIMITS = {"gate_q1": (0, 1.5), "gate_q2": (1.5, 3), "gate_q3": (3, 4.5), "gate_q4": (4.5, 6), "band_6_8": (6, 8),
               "band_8_10": (8, 10), "band_10_12": (10, 12), "band_12_16": (12, 16), "band_16_20": (16, 20)}
THRESHOLDS = [3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]  # in units of pi


def main():
    cdoc = json.loads((ROOT / "state" / "candidates_07d.json").read_text())
    cands, consts = cdoc["candidates"], cdoc["constants"]
    sel = json.loads((ROOT / "state" / "selection_leaves_07d.json").read_text())["selection"]
    state = json.loads((ROOT / "out" / "leaves" / "fill_state.json").read_text())
    N = 24
    rows = []
    for st, rk, l, _ in prefix_leaves(sel, state, {"gate_q", "band_6_8", "band_8_10", "band_10_12", "band_12_16", "band_16_20"}):
        if l is None or "error" in l:
            continue
        band = st.split("|")[0]
        ref, eps = fx(l["ref"]), fx(l["eps_ref"])
        te_cc = abs(fx(l["cc47"]) - ref)
        te_l = abs(fx(l["levin"]["vP"]) - ref)
        r = {"stratum": st, "band": band, "kind": st.split("|")[1], "rtol": st.split("|")[2], "rank": rk,
             "V_over_pi": 6 * l["V_over_6pi"], "te_cc": te_cc, "te_levin": te_l, "eps": eps, "request_share": l["request_share"],
             "cc": {}, "levin": None}
        for name, cand in cands.items():
            D, _ = declaration(l, cand, consts)
            if D is None:
                r["cc"][name] = ("n/a", None, cand["extra_evals"])
                continue
            r["cc"][name] = (verdict(D, te_cc, eps), D / te_cc if te_cc > 0 else None, cand["extra_evals"])
        Dl = fx(l["levin"]["own_decl"])
        r["levin"] = (verdict(Dl, te_l, eps), Dl / te_l if te_l > 0 else None)
        # beside it (not the prompt's route verdict): the Levin solve as the driver's trust check would declare it, the
        # quarters' sum Q with max(residual, delta1)/(1 - r) if r < 1, else V_P with max(residual, delta1 + delta2)
        lv = l["levin"]
        rr_, d1_, d2_, res_ = lv["r"], fx(lv["delta1"]), fx(lv["delta2"]), fx(lv["residual"])
        if rr_ < 1:
            vtc, Dtc = fx(lv["s2"]), max(res_, d1_) / (1.0 - rr_) + fx(lv["trunc"])
        else:
            vtc, Dtc = fx(lv["vP"]), max(res_, d1_ + d2_) + fx(lv["trunc"])
        te_tc = abs(vtc - ref)
        r["te_levin_tc"] = te_tc
        r["levin_tc"] = (verdict(Dtc, te_tc, eps), Dtc / te_tc if te_tc > 0 else None, "trusted" if rr_ < 0.25 else ("r<1" if rr_ < 1 else "r>=1"))
        rows.append(r)

    def summ(sub, route, name=None):
        vs = [(x["cc"][name] if route == "cc" else x[route]) for x in sub]
        vs = [v for v in vs if v[0] != "n/a"]
        rat = [v[1] for v in vs if v[0] != "undecided" and v[1] is not None]
        tr = [(x["te_cc"] if route == "cc" else (x["te_levin"] if route == "levin" else x["te_levin_tc"])) / x["request_share"]
              for x in sub if x["request_share"] > 0]
        return {"n": len(vs), "bounded": sum(v[0] == "bounded" for v in vs), "understates": sum(v[0] == "understates" for v in vs),
                "undecided": sum(v[0] == "undecided" for v in vs),
                "ratio_min": min(rat) if rat else None, "ratio_median": float(np.median(rat)) if rat else None,
                "ratio_max": max(rat) if rat else None,
                "true_over_request_median": float(np.median(tr)) if tr else None, "true_over_request_max": max(tr) if tr else None,
                "evals": (2 * N - 1 + (cands[name]["extra_evals"] if name else 0)) if route == "cc" else 3 * N}
    table = {}
    for band in BAND_LIMITS:
        for kind in ("all", "group", "analytic"):
            sub = [x for x in rows if x["band"] == band and (kind == "all" or x["kind"] == kind)]
            if not sub:
                continue
            table[f"{band}|{kind}"] = {"levin": summ(sub, "levin"), "levin_tc": summ(sub, "levin_tc"),
                                       **{f"cc:{n}": summ(sub, "cc", n) for n in cands}}
    verdicts = {}
    for T in THRESHOLDS:
        below = [x for x in rows if x["V_over_pi"] < T]
        above = [x for x in rows if x["V_over_pi"] >= T]
        lev_under = [x for x in above if x["levin"][0] == "understates"]
        v = {"levin_above": {"n": len(above), "understates": len(lev_under),
                             "understating": [(x["stratum"], x["rank"]) for x in lev_under]},
             "levin_tc_above": {"n": len(above), "understates": sum(1 for x in above if x["levin_tc"][0] == "understates")},
             "cc_below": {}, "rule_a_changes": 0}
        for name in cands:
            cu = [x for x in below if x["cc"][name][0] == "understates"]
            v["cc_below"][name] = {"n": len(below), "understates": len(cu),
                                   "bounded_both_sides": not cu and not lev_under}
        verdicts[f"{T}pi"] = v
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: D-T, both routes against V, and the thresholds",
                 [ROOT / "state" / "candidates_07d.json", ROOT / "out" / "leaves" / "fill_state.json"])
    od = ROOT / "out" / "tables"
    (od / "d_t.json").write_text(json.dumps({"header": hdr, "table": table, "thresholds": verdicts, "rows": rows}, indent=1))
    g = lambda x: "-" if x is None else f"{x:.3g}"  # noqa: E731
    L = [l.replace("# ", "", 1) for l in hdr] + ["", "| band|kind | route | n | bounded | understates | undecided | D/true min | median | max | true/request median | max | evals |",
                                                 "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, t in table.items():
        for route, s in t.items():
            L.append(f"| {k} | {route} | {s['n']} | {s['bounded']} | {s['understates']} | {s['undecided']} | {g(s['ratio_min'])} | "
                     f"{g(s['ratio_median'])} | {g(s['ratio_max'])} | {g(s['true_over_request_median'])} | {g(s['true_over_request_max'])} | {s['evals']} |")
    L += ["", "## Thresholds (Rule A reads no threshold at 07c's code: no change at any T)", "",
          "| T | Levin above: n / understates (with the trust check, beside) | " + " | ".join(f"CC below ({n}): n / understates / both sides bounded" for n in cands) + " |",
          "|---|---|" + "---|" * len(cands)]
    for T, v in verdicts.items():
        L.append(f"| {T} | {v['levin_above']['n']} / {v['levin_above']['understates']} ({v['levin_tc_above']['understates']}) | " + " | ".join(
            f"{c['n']} / {c['understates']} / {'yes' if c['bounded_both_sides'] else 'no'}" for c in v["cc_below"].values()) + " |")
    (od / "d_t.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-16:]))
    print("wrote", od / "d_t.json", sha256(od / "d_t.json"))


if __name__ == "__main__":
    main()
