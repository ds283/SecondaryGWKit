"""levin-phase-floor 07d Part A' addendum (observation): on every check still undecided after the supplementary pass
(out/tables_add_supp.json), the check's V_P, halves' sum and Q (the leaf record's item 1 entry for that T and fallback),
each against the smooth-phase reference, and D. Asks whether D ~ te because V_P is accurate and Q carries an error the
halves share (D ~ delta1 = |V_P - halves|). Writes out/edge_vp_add.json and prints a summary. Usage: python3 -B edge_vp_add.py"""
import json, statistics as S
from pathlib import Path
ADD = Path(__file__).resolve().parent.parent
fx = float.fromhex
sel = json.loads((ADD / "state" / "selection_add.json").read_text())
sup = json.loads((ADD / "out" / "tables_add_supp.json").read_text())["rows"]
sm = {}
for sub in ("smooth", "smooth_supp"):
    for p in sorted((ADD / "out" / sub).glob("*.json")):
        s = json.loads(p.read_text()); sm[s["id"]] = s
recs = {}
for i, lv in sel["leaves"].items():
    for L in json.loads(Path(lv["file"]).read_text())["leaves"]:
        if L["id"] == i:
            recs[i] = L
rows = []
for r in sup:
    if "Q, fallback" not in r["what"]:
        continue
    fb = r["what"].split()[-1]
    c = next(c for c in recs[r["id"]]["item1"] if c["T_pi"] == r["T_pi"] and c["fallback"] == fb and c.get("checked"))
    ref = fx(sm[r["id"]]["value_hex"])
    e = {k: abs(fx(c[k]) - ref) for k in ("V_P", "halves_sum", "Q")}
    rows.append({"id": r["id"], "T_pi": r["T_pi"], "fallback": fb, "D": r["D"], "delta1": fx(c["delta1"]), "r": c["r"],
                 "err_V_P": e["V_P"], "err_halves": e["halves_sum"], "err_Q": e["Q"], "eps_s": r["eps_s"],
                 "D_over_te": r["D_over_te"]})
edge = [x for x in rows if x["D_over_te"] and 0.95 <= x["D_over_te"] <= 1.05]
doc = {"rows": rows, "edge_n": len(edge),
       "edge_VP_better_than_Q": sum(1 for x in edge if x["err_V_P"] < x["err_Q"]),
       "edge_err_VP_over_err_Q_median": S.median(x["err_V_P"] / x["err_Q"] for x in edge) if edge else None,
       "edge_D_over_delta1_median": S.median(x["D"] / x["delta1"] for x in edge) if edge else None,
       "edge_errQ_over_errHalves_median": S.median(x["err_Q"] / x["err_halves"] for x in edge) if edge else None}
(ADD / "out" / "edge_vp_add.json").write_text(json.dumps(doc, indent=1))
print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
for x in edge[:60]:
    print(x["id"], x["T_pi"], x["fallback"], "D/te %.3f errVP %.3g errH %.3g errQ %.3g D %.3g d1 %.3g r %.3g" % (
        x["D_over_te"], x["err_V_P"], x["err_halves"], x["err_Q"], x["D"], x["delta1"], x["r"]))
