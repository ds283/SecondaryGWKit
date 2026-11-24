"""levin-phase-floor prompt 07d Part A' (the causes of understatement): every understatement of tables 2 and 4 re-scored
against the second reference (out/repro/second_ref_a2.json, second_ref_a2.py) beside v4. New in Part A'.

For each: the value's error against ref2 (the second reference's recorded error of that value), the same declaration D,
and the verdict with eps2 in place of eps_v4 (bounded: D >= te2 + eps2; understates: D < te2 - eps2; else undecided).
Table 2: the tree's declaration on CC_47 (t2_de_v4.json tree_understates). Table 4: the Levin side's trusted Q and
total_err_trusted (t4_dt.json per_leaf, keys "<T>|levin|<fallback>"). A cause "reference" is given where v4 and ref2
disagree (|ref - ref2| > eps + eps2) and the verdict on ref2 is not "understates".
Usage: python3 -B rescore_ref2_a2.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import ROOT, header  # noqa: E402


def verdict(D, te, eps):
    if D >= te + eps:
        return "bounded"
    if D < te - eps:
        return "understates"
    return "undecided"


def main():
    T = ROOT / "out" / "tables"
    s2p = ROOT / "out" / "repro" / "second_ref_a2.json"
    s2 = {r["id"]: r for r in json.loads(s2p.read_text())["rows"]}
    rows = []
    for u in json.loads((T / "t2_de_v4.json").read_text())["tree_understates"]:
        r = s2[u["id"]]
        te2 = r["errors"]["CC47"]["ref2"]
        rows.append({"table": 2, "id": u["id"], "what": "tree CC_47 max(pair, floor)", "D": u["D_tree"], "te_v4": u["te"],
                     "eps_v4": u["eps"], "te_ref2": te2, "eps2": r["eps2"], "refs_agree": r["refs_agree"],
                     "verdict_ref2": verdict(u["D_tree"], te2, r["eps2"]), "cause_v4_table2": u["cause"]})
    for L in json.loads((T / "t4_dt.json").read_text())["per_leaf"]:
        for k, v in L["by_T"].items():
            if not (isinstance(v, dict) and v.get("v") == "understates" and "|levin|" in k):
                continue
            Tp, _, fb = k.split("|")
            r = s2[L["id"]]
            te2 = r["errors"][f"Q T{Tp} {fb}"]["ref2"]
            rows.append({"table": 4, "id": L["id"], "what": f"Levin Q, T = {Tp} pi, fallback {fb}", "D": v["D"],
                         "te_v4": v["te"], "eps_v4": v["eps"], "te_ref2": te2, "eps2": r["eps2"],
                         "refs_agree": r["refs_agree"], "verdict_ref2": verdict(v["D"], te2, r["eps2"])})
    for x in rows:
        x["cause_ref"] = "reference" if (not x["refs_agree"] and x["verdict_ref2"] != "understates") else None
    summ = {f"table {t}": {"n": sum(1 for x in rows if x["table"] == t),
                           "verdict_ref2": dict(Counter(x["verdict_ref2"] for x in rows if x["table"] == t)),
                           "refs_disagree": sum(1 for x in rows if x["table"] == t and not x["refs_agree"]),
                           "cause_reference": sum(1 for x in rows if x["table"] == t and x["cause_ref"])}
            for t in (2, 4)}
    summ["table 2 by v4 cause"] = {c: dict(Counter(x["verdict_ref2"] for x in rows if x["table"] == 2
                                                   and x["cause_v4_table2"] == c))
                                   for c in sorted({x.get("cause_v4_table2") for x in rows if x["table"] == 2})}
    hdr = header("levin-phase-floor 07d Part A': the understatements re-scored on the second reference", __file__,
                 [s2p, T / "t2_de_v4.json", T / "t4_dt.json"])
    (T / "understate_ref2.json").write_text(json.dumps({"header": hdr, "summary": summ, "rows": rows}, indent=1))
    Lm = hdr + ["", "```", json.dumps(summ, indent=1), "```", "", "| table | leaf | what | D | te v4 | eps v4 | te ref2 | eps2 | refs agree | verdict on ref2 |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    Lm += [f"| {x['table']} | {x['id']} | {x['what']} | {x['D']:.3g} | {x['te_v4']:.3g} | {x['eps_v4']:.3g} | {x['te_ref2']:.3g} | "
           f"{x['eps2']:.3g} | {x['refs_agree']} | {x['verdict_ref2']} |" for x in rows if x["table"] == 4 or not x["refs_agree"]]
    (T / "understate_ref2.md").write_text("\n".join(Lm) + "\n")
    print("\n".join(Lm[len(hdr):]))


if __name__ == "__main__":
    main()
