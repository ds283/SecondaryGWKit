"""levin-phase-floor prompt 06l, Part C ("Against the references") and H1: copied from 06k's score_h1.py (a record;
in state/scripts_06k_reference/), changed only where marked "prompt 06l": the routes are route 1 (06k's extract, read
only) and route B (this prompt's extract), 06g's extract is 06k's (read only), H1 is scored for route B against 06g's
ratio with route 1's ratio beside it, the 115 references outside the 720 are counted, and the title names 06l. V1's 12
are listed unscored as in 06k (06k's deviation 24, on the user's ruling). 06k's text follows.

levin-phase-floor prompt 06k, Part C ("Against the references") and H1, scored by the rule as written. New in
06k. Read only. Exact arithmetic: census doubles from hex, references from their 30-digit strings.

The references, each used as its source scored it (its readings, its class, its reference value ``ref_30``):
- 06g's 102 (docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json, ``rows``: V3's 90 and V8's 12),
  each matched to its census call by unit and index (kind group or analytic), its label checked;
- 06j's 22 (docs/levin-phase-floor/06j/out/w_calls.csv), matched by unit and index;
- V1's 12 (the same file, ``V1``): these are calls of 06g's rehearsal stores R23/R24, not of A3 v2's census, and
  carry no census unit; they cannot be matched to a census call and are reported unscored (prompt section 7,
  "a reference cannot be matched to its call").
Per route and referenced call: |v_route - ref| / e_route beside |v_g - ref| / e_g (recomputed from 06g's census
record, and checked against the source's own ratio). **H1:** on every referenced call outside the 720,
|v_route - ref| / e_route <= max(1, |v_g - ref| / e_g). The 720's calls are scored apart, not for H1.
Writes out/partC/h1.json and out/partC/h1_calls.csv.
Usage: python -B score_h1.py
"""

import csv
import json
import pickle
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

ROUTES = ("route1", "routeB")  # prompt 06l
EX = RC.ROOT / "state" / "extract"
EX06K = RC.MAIN / "var" / "levin-phase-floor-06k" / "state" / "extract"  # prompt 06l: 06g's and route 1's extracts
EXP = {"g": EX06K / "g.pkl", "route1": EX06K / "route1.pkl", "routeB": EX / "routeB.pkl"}  # prompt 06l
SCORES = RC.G_TREE / "docs" / "levin-phase-floor" / "06g" / "references" / "out" / "scores_stage3" / "scores.json"
WCALLS = RC.G_TREE / "docs" / "levin-phase-floor" / "06j" / "out" / "w_calls.csv"


def F(h):
    return Fraction(float.fromhex(h))


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def main():
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06l: route B and route 1 against the references, H1",
                 [SCORES, WCALLS] + [EXP[n] for n in ("g",) + ROUTES])  # prompt 06l
    sp = RC.scan06a_sp()
    sc = json.loads(SCORES.read_text())
    refs = []
    for r in sc["rows"]:
        refs.append({"rid": r["sid"], "source": f"06g {r['source']}", "class": r["class"], "unit": r["unit"],
                     "index": r["index"], "kind": r["kind"], "label": r["label"], "ref_30": r["ref_30"],
                     "source_ratio_g": r.get("r_g")})
    with open(WCALLS) as f:
        rows = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(rows):
        refs.append({"rid": r["sid"], "source": f"06j {r['part']}", "class": "four readings agree" if r["four_agree"] == "True" else "readings disagree",
                     "unit": r["unit"], "index": int(r["index"]), "kind": "group", "label": None, "ref_30": r["ref_30"],
                     "source_ratio_g": float(r["g_ratio"])})
    docs = {}
    for n in ("g",) + ROUTES:
        with open(EXP[n], "rb") as f:  # prompt 06l
            docs[n] = pickle.load(f)

    def call(n, unit, kind, index):
        u = docs[n]["units"][unit]
        hits = [c for c in u["calls"] if c["kind"] == kind and c["index"] == index]
        return hits[0] if len(hits) == 1 else None

    out = []
    for ref in refs:
        g = call("g", ref["unit"], ref["kind"], ref["index"])
        if g is None or (ref["label"] is not None and g["label"] != ref["label"]):
            ref["error"] = "no matching census call"
            out.append(ref)
            continue
        rv = Fraction(Decimal(ref["ref_30"]))
        rg = float(abs(F(g["value"]) - rv) / F(g["abserr"]))
        ref.update({"in_720": rowkey(ref["unit"]) in sp, "group": g["group"] or g["label"], "rtol": ref["unit"].split("-")[1] + "-" + ref["unit"].split("-")[2],
                    "ratio_g": rg, "ratio_g_matches_source": None if ref["source_ratio_g"] is None else abs(rg - ref["source_ratio_g"]) <= 1e-9 * max(1.0, rg),
                    "g": [g["value"], g["abserr"], g["converged"], g["num_regions"], g["max_depth"]]})
        for n in ROUTES:
            c = call(n, ref["unit"], ref["kind"], ref["index"])
            ratio = float(abs(F(c["value"]) - rv) / F(c["abserr"]))
            ref[n] = {"ratio": ratio, "bound": max(1.0, rg), "ok": ratio <= max(1.0, rg), "converged": c["converged"],
                      "regions": c["num_regions"], "depth": c["max_depth"], "n_firings": (c.get("route") or {}).get("n_firings")}
        out.append(ref)
    v1 = [{"wid": w["wid"], "k": w["k"], "z_response": w["z_response"], "unscored": "a call of 06g's rehearsal store R24, not of A3 v2's census: no census unit to match"} for w in sc["V1"]]
    verdict = {}
    for n in ROUTES:
        sc_out = [r for r in out if "error" not in r and not r["in_720"]]
        bad = [r for r in sc_out if not r[n]["ok"]]
        verdict[n] = {"scored_outside_720": len(sc_out), "violations": [(r["rid"], r["unit"], r["index"], r["group"], round(r[n]["ratio"], 4), round(r["ratio_g"], 4)) for r in bad],
                      "worst_ratio_outside": max((r[n]["ratio"] for r in sc_out), default=None),
                      "within_own_error_outside": sum(1 for r in sc_out if r[n]["ratio"] <= 1),
                      "H1_on_matched_references": "holds" if not bad else "fails",
                      "in_720": [(r["rid"], round(r[n]["ratio"], 4), round(r["ratio_g"], 4)) for r in out if "error" not in r and r["in_720"]]}
    doc = {"header": hdr, "n_refs": len(out), "n_unmatched": sum(1 for r in out if "error" in r), "V1_unscored": v1,
           "g_ratio_check": {"matches": sum(1 for r in out if r.get("ratio_g_matches_source")), "of": sum(1 for r in out if r.get("ratio_g_matches_source") is not None)},
           "verdicts": verdict, "calls": out}
    p = RC.ROOT / "out" / "partC" / "h1.json"
    p.write_text(json.dumps(doc, indent=1, default=str))
    pc = RC.ROOT / "out" / "partC" / "h1_calls.csv"
    with open(pc, "w", newline="") as fo:
        for l in hdr:
            fo.write(l + "\n")
        w = csv.writer(fo)
        w.writerow(["rid", "source", "class", "unit", "index", "kind", "group", "in_720", "ratio_g"] + [f"{n}_{k}" for n in ROUTES for k in ("ratio", "ok", "converged", "regions")])
        for r in out:
            if "error" in r:
                continue
            w.writerow([r["rid"], r["source"], r["class"], r["unit"], r["index"], r["kind"], r["group"], r["in_720"], r["ratio_g"]]
                       + [r[n][k] for n in ROUTES for k in ("ratio", "ok", "converged", "regions")])
    for l in hdr:
        print(l)
    print(f"references {len(out)}; unmatched {doc['n_unmatched']}; 06g ratio reproduces the source's on {doc['g_ratio_check']}; V1's 12 unscored (no census call)")
    for n, v in verdict.items():
        print(f"{n}: H1 on the matched references outside the 720 {v['H1_on_matched_references']}: {v['scored_outside_720']} scored, "
              f"within own error {v['within_own_error_outside']}, worst {v['worst_ratio_outside']:.4g}; violations {v['violations']}")
        print(f"   in the 720: {v['in_720']}")
    print(f"wrote {p} sha256 {RC.sha256(p)}; {pc} sha256 {RC.sha256(pc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
