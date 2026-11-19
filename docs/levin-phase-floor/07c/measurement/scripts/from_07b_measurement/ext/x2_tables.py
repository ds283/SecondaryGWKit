"""levin-phase-floor prompt 07b extension, section 2.7.1 (X2): the candidates' tables. New in the extension.

From x2_candidates.py's outputs (out/ext/x2/x2_tree.json, x2_07.json). The leaves are Part A's scored trusted Levin
leaves (reference accuracy at most e'/100; Part A's rate-range set). The groups:
- **the six** dishonest leaves away from windows, named by the prompt: v09 leaf 38, item 3's leaf 5 (at 07's code),
  W2-0-12977-5 leaf 2, W2-1-2174-2 leaf 2, W2-2-14345-3 leaf 5, W2-2-34473-3 leaf 102;
- **the window carriers:** the other scored dishonest leaves (|err(Q)| > e');
- **honest, rate range above 30** (end to end, Part A's measure): the controls the prompt names;
- **honest, rate range at most 30**.
For each candidate, its value on every leaf beside err(Q)/e'; and its separation of the six from the honest leaves above
30 (and from every honest leaf): the smallest value among the six over the largest among the honest. Above 1 the
candidate separates them, by that margin; at most 1 it does not.
Writes OUT_JSON and OUT_MD. Usage: python -B x2_tables.py OUT_JSON OUT_MD
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SIX = {("v09", 38), ("i3-34599-07", 5), ("W2-0-12977-5", 2), ("W2-1-2174-2", 2), ("W2-2-14345-3", 5), ("W2-2-34473-3", 102)}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out_j, out_m = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (out_j, out_m):
        if p.exists():
            raise SystemExit(f"{p} exists")
    ins = [ROOT / "out" / "ext" / "x2" / "x2_tree.json", ROOT / "out" / "ext" / "x2" / "x2_07.json"]
    leaves = [l for p in ins for l in json.loads(p.read_text())["leaves"]]
    if not all(l["V_P_reproduces"] and l["Q_reproduces"] for l in leaves):
        raise SystemExit("a leaf's V_P or Q does not reproduce its probe")
    scored = [l for l in leaves if l["ref_acc_over_eprime"] <= 0.01]
    for l in scored:
        bad = abs(l["err_Q_over_eprime"]) > 1
        if (l["sid"], l["i"]) in SIX:
            l["group"] = "six"
        elif bad:
            l["group"] = "window carrier" if l["touches_window"] else "other dishonest"
        else:
            l["group"] = "honest >30" if l["rate_end_to_end"] > 30 else "honest <=30"
    missing = SIX - {(l["sid"], l["i"]) for l in scored if l["group"] == "six"}
    if missing:
        raise SystemExit(f"the six are not all scored: {missing}")
    N = scored[0]["order"]
    ords = list(scored[0]["orders"])
    cands = {}
    for o in ords:
        cands[f"D_Q({o})"] = lambda l, o=o: l["orders"][o]["D_Q"]
        cands[f"D_P({o})"] = lambda l, o=o: l["orders"][o]["D_P"]
    cands.update({
        "r3": lambda l: l["r3"], "|S3-Q|/e'": lambda l: l["S3_minus_Q_over_eprime"],
        "tail_P/e'": lambda l: l["tail_P_over_eprime"], "tail_Q/e'": lambda l: l["tail_Q_over_eprime"],
        "tail_P_rel": lambda l: l["tail_P_rel"], "tail_Q_rel": lambda l: l["tail_Q_rel_max"],
        "cond_P": lambda l: l["cond_P"], "rate range (nodes)": lambda l: l["rate_range_nodes"],
        "rate range (end to end)": lambda l: l["rate_end_to_end"], "r": lambda l: l["r"],
    })
    sep = {}
    for name, fn in cands.items():
        six = [fn(l) for l in scored if l["group"] == "six"]
        h30 = [fn(l) for l in scored if l["group"] == "honest >30"]
        hall = [fn(l) for l in scored if l["group"].startswith("honest")]
        win = [fn(l) for l in scored if l["group"] == "window carrier"]
        sep[name] = {"six_min": min(six), "six_max": max(six), "honest30_max": max(h30), "honest_all_max": max(hall),
                     "window_carriers_min": min(win) if win else None,
                     "margin_vs_honest30": min(six) / max(h30) if max(h30) > 0 else float("inf"),
                     "margin_vs_all_honest": min(six) / max(hall) if max(hall) > 0 else float("inf"),
                     "honest30_above_six_min": sum(1 for v in h30 if v >= min(six)),
                     "honest_all_above_six_min": sum(1 for v in hall if v >= min(six))}
    # does a higher order remove the error?
    fix = [{"sid": l["sid"], "i": l["i"], "group": l["group"], "err_Q": l["err_Q_over_eprime"],
            **{f"err_Q({o})": l["orders"][o]["err_Q_over_eprime"] for o in ords}} for l in scored
           if l["group"] in ("six", "window carrier", "other dishonest")]
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "inputs": {str(p): sha256(p) for p in ins}, "order": N},
           "groups": {g: sum(1 for l in scored if l["group"] == g) for g in sorted({l["group"] for l in scored})},
           "separation": sep, "higher_order_on_the_dishonest": fix, "leaves": scored}
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(doc, indent=1))
    md = [f"## (X2) What separates the six dishonest leaves away from windows from the honest trusted leaves", "",
          f"Scored trusted Levin leaves (reference accuracy at most e'/100): {len(scored)}; groups: "
          + ", ".join(f"{g} {n}" for g, n in doc["groups"].items()) + f". The calls' collocation order N = {N}.", "",
          "### Separation (smallest among the six over largest among the honest; above 1 separates)", "",
          "| candidate | six: min - max | honest >30: max | margin vs honest >30 | honest >30 at or above the six's min | all honest: max | margin vs all honest | honest at or above | window carriers: min |",
          "|---|---|---|---|---|---|---|---|---|"]
    g = lambda v: "-" if v is None else f"{v:.3g}"
    for name, s in sep.items():
        md.append(f"| {name} | {g(s['six_min'])} - {g(s['six_max'])} | {g(s['honest30_max'])} | {g(s['margin_vs_honest30'])} | "
                  f"{s['honest30_above_six_min']} | {g(s['honest_all_max'])} | {g(s['margin_vs_all_honest'])} | "
                  f"{s['honest_all_above_six_min']} | {g(s['window_carriers_min'])} |")
    md += ["", "### Does a higher collocation order remove the error? err(Q at order N')/e' on the dishonest leaves", "",
           "| call | leaf | group | err(Q)/e' (N) | " + " | ".join(f"N'={o}" for o in ords) + " |", "|---|---|---|---|" + "---|" * len(ords)]
    for x in fix:
        md.append(f"| {x['sid']} | {x['i']} | {x['group']} | {x['err_Q']:+.3g} | " + " | ".join(f"{x[f'err_Q({o})']:+.3g}" for o in ords) + " |")
    md += ["", "### Every leaf", "",
           "| call | leaf | group | depth | err(Q)/e' | rate range | " + " | ".join(cands) + " |", "|---|---|---|---|---|---|" + "---|" * len(cands)]
    for l in sorted(scored, key=lambda l: (l["group"], l["sid"], l["i"])):
        md.append(f"| {l['sid']} | {l['i']} | {l['group']} | {l['depth']} | {l['err_Q_over_eprime']:+.3g} | {l['rate_end_to_end']:.3g} | "
                  + " | ".join(g(fn(l)) for fn in cands.values()) + " |")
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md[:8 + len(sep) + 4 + len(fix)]))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
