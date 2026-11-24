"""levin-phase-floor 07d Part A' addendum, items A.3's selection and A.4: every selected check scored on v4, on ref48 and,
where taken, on the smooth-phase reference, by state/rule_ref48_add.md.

Mode ``pass2``: writes state/second_pass_add.json, the leaves whose ref48 verdict the noise term decides (undecided on
eps48, decided on the change alone, |D - te48| > change), before any smooth-phase reference is taken.
Mode ``final``: writes out/tables_add.json and .md: per check and T, D, the error and stated error on each reference,
the verdict on each, the final verdict (the last reference taken on its leaf), "at the noise floor" (final undecided and
2 s_47 >= |D - te|, s_47 = sigma_hat sqrt(sum w^2 of the leaf's CC_47 rule)), and the counts by set, T, route and what.
Usage: python3 -B score_add.py pass2|final
"""

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ADD = Path(__file__).resolve().parent.parent
A2 = ADD.parent
fx = float.fromhex


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verdict(D, te, eps):
    if D >= te + eps:
        return "bounded"
    if D < te - eps:
        return "understates"
    return "undecided"


def fn(i):
    return i.replace("|", "_").replace("#", "_") + ".json"


def main():
    mode = sys.argv[1]
    sel = json.loads((ADD / "state" / "selection_add.json").read_text())
    r48 = {i: json.loads((ADD / "out" / "ref48" / fn(i)).read_text()) for i in sel["leaves"]}
    sm = {}
    if mode == "final":
        for i in sel["leaves"]:
            p = ADD / "out" / "smooth" / fn(i)
            if p.exists():
                sm[i] = json.loads(p.read_text())
    recs = {}
    for i, lv in sel["leaves"].items():
        for L in json.loads(Path(lv["file"]).read_text())["leaves"]:
            if L["id"] == i:
                recs[i] = L
    rows = []
    for c in sel["checks"]:
        i = c["id"]
        R = r48[i]
        val = fx(c["value"])
        ref, eps, ch = fx(R["ref48"]), R["eps48"], R["change"]
        te48 = abs(val - ref)
        v48 = verdict(c["D"], te48, eps)
        noise_decides = v48 == "undecided" and abs(c["D"] - te48) > ch
        row = dict(c, te48=te48, eps48=eps, change48=ch, noise48=R["noise"], dominant48=R["dominant"],
                   finest48=R["finest"], moving_at_2048=R["moving_at_2048"], v_v4="undecided", v48=v48,
                   noise_decides=noise_decides, refs_v4_48_agree=abs(fx(recs[i]["v4"]["ref"]) - ref) <= c["eps_v4"] + eps)
        if c["D_variant"] is not None:
            row["v48_variant_decl"] = verdict(c["D_variant"], te48, eps)
        L = recs[i]
        s47 = L["v3_sigma"]["sigma_hat"] * math.sqrt(L["cc"]["c47"]["sum_w2"])
        row["s47"] = s47
        final_ref, final_eps, final_v, final_src = ref, eps, v48, "ref48"
        if i in sm and sm[i].get("value_hex"):
            S = sm[i]
            rs, es = fx(S["value_hex"]), S["eps_s"]
            tes = abs(val - rs)
            row.update(te_s=tes, eps_s=es, v_s=verdict(c["D"], tes, es), smooth_counts=S.get("counts"))
            if c["D_variant"] is not None:
                row["v_s_variant_decl"] = verdict(c["D_variant"], tes, es)
            final_ref, final_eps, final_v, final_src = rs, es, row["v_s"], "smooth"
        te_f = abs(val - final_ref)
        row.update(final_source=final_src, final_verdict=final_v, te_final=te_f, gap=abs(c["D"] - te_f),
                   at_noise_floor=final_v == "undecided" and 2 * s47 >= abs(c["D"] - te_f))
        rows.append(row)
    if mode == "pass2":
        out = ADD / "state" / "second_pass_add.json"
        if out.exists():
            raise SystemExit(f"{out} exists")
        leaves = sorted({r["id"] for r in rows if r["noise_decides"]})
        doc = {"rule": "state/rule_ref48_add.md, A.3", "ref48_sha256": {i: sha(ADD / "out" / "ref48" / fn(i)) for i in leaves},
               "n_checks_noise_decides": sum(r["noise_decides"] for r in rows), "leaves": leaves,
               "v48_counts": dict(Counter(f"{r['set']}|{r['route']}|{r['v48']}" for r in rows)),
               "dominant48_by_leaf": dict(Counter(R["dominant"] for R in r48.values())),
               "finest48_by_leaf": dict(Counter(R["finest"] for R in r48.values())),
               "moving_at_2048_leaves": sorted(i for i, R in r48.items() if R["moving_at_2048"])}
        out.write_text(json.dumps(doc, indent=1))
        print(json.dumps({k: v for k, v in doc.items() if k != "ref48_sha256"}, indent=1))
        return
    cnt = defaultdict(Counter)
    for r in rows:
        k = f"{r['set']}|T{r['T_pi']}|{r['route']}|{r['what']}"
        cnt[k][r["final_verdict"] + (" (noise floor)" if r["at_noise_floor"] else "")] += 1
        cnt[k]["ref48: " + r["v48"]] += 1
    doc = {"rows": rows, "counts": {k: dict(v) for k, v in sorted(cnt.items())},
           "summary": {"final": dict(Counter(f"{r['set']}|{r['route']}|{r['final_verdict']}" for r in rows)),
                       "at_noise_floor": sum(r["at_noise_floor"] for r in rows),
                       "ref48": dict(Counter(f"{r['set']}|{r['route']}|{r['v48']}" for r in rows)),
                       "v4_ref48_disagree_checks": sum(not r["refs_v4_48_agree"] for r in rows),
                       "smooth_taken_leaves": len(sm)}}
    (ADD / "out" / "tables_add.json").write_text(json.dumps(doc, indent=1))
    L = ["# Addendum A.4: the near-edge checks on ref48 and the smooth-phase reference", "",
         "```", json.dumps(doc["summary"], indent=1), "```", "", "## Counts by set, T, route and what (final verdict; and ref48's)", "",
         "| key | counts |", "|---|---|"]
    L += [f"| {k} | {json.dumps(v)} |" for k, v in doc["counts"].items()]
    L += ["", "## Every check", "", "| leaf | set | T/pi | what | D | te v4 / eps v4 | te48 / eps48 (dominant) | v48 | te smooth / eps_s | "
          "final | |D - te| | 2 s_47 | noise floor |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["set"], r["T_pi"], r["route"], r["id"])):
        sm_s = f"{r['te_s']:.3g} / {r['eps_s']:.3g}" if "te_s" in r else "—"
        L.append(f"| {r['id']} | {r['set']} | {r['T_pi']} | {r['what']} | {r['D']:.3g} | {r['te_v4']:.3g} / {r['eps_v4']:.3g} | "
                 f"{r['te48']:.3g} / {r['eps48']:.3g} ({r['dominant48']}) | {r['v48']} | {sm_s} | {r['final_verdict']} | "
                 f"{r['gap']:.3g} | {2 * r['s47']:.3g} | {'yes' if r['at_noise_floor'] else ''} |")
    (ADD / "out" / "tables_add.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:8 + len(doc["counts"]) + 4]))


if __name__ == "__main__":
    main()
