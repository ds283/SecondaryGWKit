"""levin-phase-floor 07d Part A' addendum: the supplementary pass (state/supplementary_pass_add.json; written after the
A.4 tables were read) scored. Every check whose fixed-pass final verdict (out/tables_add.json) is undecided, on the
smooth-phase reference of out/smooth_supp/ (or out/smooth/ where the fixed pass took one): te_s, eps_s, the verdict
(v4's rule), D/te_s, and |D - te_s| against eps_s. Writes out/tables_add_supp.json and .md.
Usage: python3 -B supp_score_add.py"""
import json
from collections import Counter
from pathlib import Path
ADD = Path(__file__).resolve().parent.parent
fx = float.fromhex
d = json.loads((ADD / "out" / "tables_add.json").read_text())
sm = {}
for sub in ("smooth", "smooth_supp"):
    for p in sorted((ADD / "out" / sub).glob("*.json")):
        s = json.loads(p.read_text())
        sm[s["id"]] = dict(s, source=sub)
def verdict(D, te, eps):
    return "bounded" if D >= te + eps else "understates" if D < te - eps else "undecided"
rows = []
for r in d["rows"]:
    if r["final_verdict"] != "undecided":
        continue
    S = sm[r["id"]]
    rs, es = fx(S["value_hex"]), S["eps_s"]
    te = abs(fx(r["value"]) - rs)
    rows.append({"id": r["id"], "set": r["set"], "T_pi": r["T_pi"], "what": r["what"], "D": r["D"], "te_s": te, "eps_s": es,
                 "source": S["source"], "verdict_s": verdict(r["D"], te, es), "D_over_te": r["D"] / te if te else None,
                 "gap": abs(r["D"] - te), "te48": r["te48"], "eps48": r["eps48"]})
cnt = Counter(f"{x['set']}|T{x['T_pi']}|{x['what']}|{x['verdict_s']}" for x in rows)
doc = {"rows": rows, "counts": dict(sorted(cnt.items())), "summary": dict(Counter(x["verdict_s"] for x in rows))}
(ADD / "out" / "tables_add_supp.json").write_text(json.dumps(doc, indent=1))
L = ["# Addendum: the fixed passes' undecided checks on the smooth-phase reference (supplementary pass)", "",
     "summary: " + json.dumps(doc["summary"]), "", "| key | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in doc["counts"].items()]
L += ["", "| leaf | set | T/pi | what | D | te_s | eps_s | D/te_s | |D - te_s| | verdict | ref from |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for x in sorted(rows, key=lambda x: (x["verdict_s"], x["D_over_te"] or 0)):
    L.append(f"| {x['id']} | {x['set']} | {x['T_pi']} | {x['what']} | {x['D']:.3g} | {x['te_s']:.3g} | {x['eps_s']:.3g} | "
             f"{(x['D_over_te'] or float('nan')):.3g} | {x['gap']:.3g} | {x['verdict_s']} | {x['source']} |")
(ADD / "out" / "tables_add_supp.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
