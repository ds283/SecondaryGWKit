"""levin-phase-floor prompt 07c Part A extension (R2-28), questions 1, 4 and 5: the carrying leaf's share in each code's
error, and whether R2-28 is one call or a class, from what this Part already holds. New in the extension.

- **The split** (out/r2-28/leaves.json): in each partition, the carrying leaf (the tree's leaf 295, [0x1.65b4fd067cd45p+4,
  0x1.78a11e65a7f80p+4], found in question 1), its value, error and e', and the rest of the call: its error (the call's
  error against 07b's reference, less the leaf's) against the call's abserr and against the abserr less the leaf's e'.
- **The class, among the scored calls** (out/tables/scores_07c.json, the 305 referenced calls, R31's record and the chase
  calls among them): every call on which every 07b-lineage code (tree, S, S2, N, Eg, Ea, SEg, SEa) errs by more than its
  abserr; and the calls on which the eight codes' errors agree with one another to within a tenth of their smallest
  error (a shared error), with the smallest ratio |v - ref|/abserr over the eight.
- **The class, among the leaves with references**: this Part has leaf references on trusted Levin leaves only for R2-28
  (out/r2-28/leaves.json); its trusted Levin leaves whose error exceeds their e' and their reference's stated error, in
  every partition. (D1's and D2's leaf references are of Clenshaw-Curtis leaves.)
Writes out/r2-28/summary.json and out/r2-28/summary.md (refuses to overwrite). Usage: python -B r2_28_summary_07c.py
"""

import hashlib
import json
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
ROOT = Path(__file__).resolve().parent.parent
LEAF = ("0x1.65b4fd067cd45p+4", "0x1.78a11e65a7f80p+4")
CODES = ["tree", "S", "S2", "N", "Eg", "Ea", "SEg", "SEa"]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def main():
    oj, om = ROOT / "out" / "r2-28" / "summary.json", ROOT / "out" / "r2-28" / "summary.md"
    for p in (oj, om):
        if p.exists():
            raise SystemExit(f"{p} exists")
    lv_p, sc_p = ROOT / "out" / "r2-28" / "leaves.json", ROOT / "out" / "tables" / "scores_07c.json"
    lv, sc = json.loads(lv_p.read_text()), json.loads(sc_p.read_text())
    doc = {"inputs": {str(lv_p): sha256(lv_p), str(sc_p): sha256(sc_p)}, "split": {}, "class_scored": {}, "class_leaves": {}}
    md = ["# R2-28: the carrying leaf's share, and one call or a class (extension, questions 1, 4 and 5)", "",
          "## The carrying leaf in each partition", "",
          "| partition | regions | leaf value (hex) | leaf error | leaf e' | e' / call abserr | rest of the call: error | rest / abserr | rest / (abserr - e') |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, P in lv["partitions"].items():
        ab = fx(P["call_abserr"])
        L = [r for r in P["leaves"] if (r["lo"], r["hi"]) == LEAF]
        if not L:
            doc["split"][name] = None
            md.append(f"| {name} | {P['regions']} | (no such leaf) | | | | {P['call_err_vs_07b_ref']:.4g} | {abs(P['call_err_vs_07b_ref']) / ab:.3g} | |")
            continue
        l = L[0]
        rest = P["call_err_vs_07b_ref"] - l["err"]
        doc["split"][name] = {"regions": P["regions"], "leaf_value": l["value"], "leaf_err": l["err"], "leaf_eprime": l["eprime"],
                              "call_abserr": ab, "rest_err": rest, "rest_over_abserr": abs(rest) / ab,
                              "rest_over_abserr_less_eprime": abs(rest) / (ab - l["eprime"])}
        md.append(f"| {name} | {P['regions']} | {l['value']} | {l['err']:.4g} | {l['eprime']:.4g} | {l['eprime'] / ab:.3f} | {rest:.3g} | "
                  f"{abs(rest) / ab:.4f} | {abs(rest) / (ab - l['eprime']):.4f} |")
    s, t = sc["scored"], sc["table"]
    rows = []
    for k, v in s.items():
        ref = mpf(v["ref"])
        errs = {c: float(mpf(fx(t[k][c]["value"])) - ref) for c in CODES}
        ab = {c: fx(t[k][c]["abserr"]) for c in CODES}
        ratios = {c: abs(errs[c]) / ab[c] for c in CODES}
        shared = (max(errs.values()) - min(errs.values())) < 0.1 * min(abs(e) for e in errs.values()) and len({e > 0 for e in errs.values()}) == 1
        rows.append({"call": k, "set": v["set"], "scored": v["scored"], "min_ratio": min(ratios.values()),
                     "max_ratio": max(ratios.values()), "shared": shared, "all_above": all(r > 1 for r in ratios.values())})
    rows.sort(key=lambda r: -r["min_ratio"])
    above = [r for r in rows if r["all_above"]]
    shared = [r for r in rows if r["shared"] and r["min_ratio"] > 0.5]
    doc["class_scored"] = {"n_referenced": len(rows), "all_codes_above_abserr": above, "shared_min_ratio_above_half": shared}
    md += ["", f"## Among the {len(rows)} referenced calls (305 scored and not)", "",
           f"- every 07b-lineage code's error above its abserr: **{len(above)}** rows: " + ", ".join(r["call"] for r in above),
           f"- the eight codes share an error (agree to a tenth of it) with every ratio above 0.5: {len(shared)} rows: "
           + ", ".join(f"{r['call']} ({r['min_ratio']:.3f}-{r['max_ratio']:.3f})" for r in shared), ""]
    md += ["## Trusted Levin leaves above their e' (R2-28's partitions: the only trusted Levin leaves with references here)", "",
           "| partition | trusted Levin leaves | above e' and resolved (i, depth, error/e', error, ref err) |", "|---|---|---|"]
    for name, P in lv["partitions"].items():
        tl = [r for r in P["leaves"] if r["kind"] == "levin_trusted"]
        big = [r for r in tl if r["err_over_eprime"] is not None and abs(r["err_over_eprime"]) > 1 and r["resolved"]]
        doc["class_leaves"][name] = {"n_trusted_levin": len(tl), "above": [{k: r[k] for k in ("i", "depth", "lo", "hi", "err", "ref_acc", "eprime", "err_over_eprime")} for r in big]}
        md.append(f"| {name} | {len(tl)} | " + "; ".join(f"{r['i']} d{r['depth']} {r['err_over_eprime']:.3g} ({r['err']:.3g}, {r['ref_acc']:.2g})" for r in big) + " |")
    oj.write_text(json.dumps(doc, indent=1))
    om.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
