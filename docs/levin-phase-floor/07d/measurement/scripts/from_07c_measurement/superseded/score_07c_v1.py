"""levin-phase-floor prompt 07c Part A, section 2.6 tables 3, 4 and 5, and the qualification. New in 07c, after 07b Part A's
score_07b.py (its ``refs()`` and ``ref_key()`` imported from scripts/from_07b_measurement/, their paths 07b's records, read
only).

**References** (each a reading of the integrand alone, so it scores every code and variant):
- 07b Part A's: score_07b's ``refs()`` with its paths pointed at 07b's records (V4's 25 from 07's readings in this tree's
  docs; 06b's W2 by S_10; item 3; 07b Part A's readings, out/refs/readings/ of var/levin-phase-floor-07b/measurement);
  the key of a Part A call ``a:<sid>`` is score_07b's ``ref_key(sid)``;
- this Part's (out/refs/readings/): R4 ``ref-R4-<shard>-<serial>-g<i>``, R5 ``ref-R5-...``, R31's record
  ``ref-R31-3-20363-g1``, read by score_07b's rule: the reference is split FCC r2 (or FCC r2), its stated error the spread of
  every reading taken (largest minus smallest).
A chase call (``c-``, ``x-``) is referenced if its integrand is R4's, or R2-28's (``ref-R2-28``); an R5 call (``m-``, its
``x-``) by R5's.

**Scored** (07's rule, 07b's W4): a call is scored if its reference's stated error is at most min(abserr)/100, the minimum
over the codes and variants scored on it (07a's census value too for R5's calls); otherwise reported, not scored.
|v - ref|/abserr > 1 is dishonest.

**Table 3 (honesty):** for the tree and each variant, every referenced call: converged and honest, converged and dishonest,
unconverged (with its ratio). **Table 4 (the chase):** on the 198 chase calls and item 2's (R2-28 at both rtols, R31's
record): unconverged; of those, how many end with every depth-20 leaf below a refused leaf (from the instrumented replays
``<variant>_i``, else the plain variant's leaf records cannot say and the column is left blank); regions and wall.
**Table 5 (cost and reach):** regions and wall over every call of section 2.1 against the tree's, and the calls a variant
touches (value, abserr, converged, num_regions or max_depth differing from the tree's replay).

**Qualification:** a variant qualifies if no referenced call is converged and dishonest under it that is not so at the
tree, and it leaves fewer chase calls unconverged than the tree (section 2.6). R5's verdicts at 07a's code (07a's census
record, from 07b's comparison) and at 07b's are reported beside, not folded in.

Usage: PYTHONPATH=<tree>:<scripts> python -B score_07c.py OUT_JSON OUT_MD VARIANT [VARIANT ...]   (``tree`` first)
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from mpmath import mp, mpf

mp.dps = 40
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f")
R07B = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement")
READ = ROOT / "out" / "refs" / "readings"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return float.fromhex(h) if isinstance(h, str) else h


def load_07b_scorer():
    spec = importlib.util.spec_from_file_location("score_07b", str(HERE / "from_07b_measurement" / "score_07b.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    # its paths, 07b's records (read only): V4's scores are in this tree's docs too, unchanged since
    m.V4S = TREE / "docs/levin-phase-floor/07/census/out/v4/v4_scores.json"
    m.W2R = TREE / "docs/levin-phase-floor/06b/census/out/w2_reference.json"
    m.SEL = R07B / "state" / "selection_07b.json"
    m.MAN = R07B / "state" / "calls_07b.json"
    m.READ = R07B / "out" / "refs" / "readings"
    return m


def my_refs():
    out = {}
    keys = {p.name.split("-fccsplit")[0] for p in READ.glob("ref-*-fccsplit.json")} | {
        p.name[: -len("-fcc-r2.json")] for p in READ.glob("ref-*-fcc-r2.json")}
    for k in sorted(keys):
        rd = {}
        fs = READ / f"{k}-fccsplit.json"
        if fs.exists():
            d = json.loads(fs.read_text())
            for res, r in d["readings"].items():
                rd[f"fccsplit-r{res}"] = mp.fsum(mpf(x["value_30"]) for x in r["pieces"])
        for impl in ("fcc", "qawo"):
            for res in (1, 2):
                q = READ / f"{k}-{impl}-r{res}.json"
                if q.exists():
                    d = json.loads(q.read_text())
                    if d.get("value_30") is not None:
                        rd[f"{impl}-r{res}"] = mpf(d["value_30"])
        key = "fccsplit-r2" if "fccsplit-r2" in rd else ("fcc-r2" if "fcc-r2" in rd else None)
        if key is None:
            continue
        vals = list(rd.values())
        out[k] = {"ref": rd[key], "spread": float(max(vals) - min(vals)), "basis": "+".join(sorted(rd)), "n": len(vals)}
    return out


def ref_key_of(sid, c, S7B, r4_ints, r5_ints):
    if sid.startswith("a:"):
        return S7B.ref_key(sid[2:], None, None)
    if sid.startswith("r31"):
        return "ref-R31-3-20363-g1"
    p = sid.split("-")
    # c-<rt>-<sh>-<se>-<i>, x-<rt>-<sh>-<se>-<i>, m-<rt>-<sh>-<se>-g<i>
    sh, se, last = p[-3], p[-2], p[-1]
    i = last[1:] if last[0] in "ga" else last
    ik = f"{sh}-{se}-g{i}"
    if ik == "0-13757-g5":
        return "ref-R2-28"
    if c["set"] in ("chase", "chase~o") and ik in r4_ints:
        return f"ref-R4-{ik}"
    if c["set"] in ("r5", "r5~o") and ik in r5_ints:
        return f"ref-R5-{ik}"
    return None


def main():
    out_j, out_m, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    if variants[0] != "tree":
        raise SystemExit("tree first")
    S7B = load_07b_scorer()
    R = dict(S7B.refs())
    R.update(my_refs())
    man = json.loads((ROOT / "state" / "calls_07c.json").read_text())["calls"]
    man.update(json.loads((ROOT / "state" / "calls_07c_r5.json").read_text())["calls"])
    r4 = set(json.loads((ROOT / "state" / "selection_r4.json").read_text())["taken"])
    r5 = set(json.loads((ROOT / "state" / "selection_r5.json").read_text())["taken"])
    pops = json.loads((ROOT / "state" / "populations_07c.json").read_text())
    a07 = {}
    for c in pops["moved_beyond_sum"]["calls"]:
        p = c["unit"].split("-")
        a07[f"m-{c['rtol']}-{p[-2]}-{p[-1]}-{c['kind'][0]}{c['index']}"] = {"value": c["value"][1], "abserr": c["abserr"][1],
                                                                            "converged": c["converged"][1]}
    # every replay
    table = {}
    for v in variants:
        for sid, c in man.items():
            p = ROOT / "out" / "replays" / v / f"{sid}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            table.setdefault(sid, {})[v] = {"value": d["replay"]["value"], "abserr": d["replay"]["abserr"],
                                            "converged": d["replay"]["converged"], "regions": d["replay"]["num_regions"],
                                            "max_depth": d["replay"]["max_depth"], "wall_s": d["wall_s"],
                                            "reproduces_census": d["reproduces_census"], "regions_sha256": d.get("regions_sha256")}
    # scoring
    scored = {}
    for sid, rows in table.items():
        rk = ref_key_of(sid, man[sid], S7B, r4, r5)
        if rk is None or rk not in R:
            continue
        rr = R[rk]
        es = [fx(r["abserr"]) for r in rows.values()]
        if sid in a07:
            es.append(fx(a07[sid]["abserr"]))
        is_scored = rr["spread"] <= min(es) / 100
        s = {"ref_key": rk, "ref": mp.nstr(rr["ref"], 25), "spread": rr["spread"], "basis": rr["basis"], "scored": is_scored,
             "set": man[sid]["set"], "by": {}}
        for v, r in rows.items():
            ratio = float(abs(mpf(fx(r["value"])) - rr["ref"]) / mpf(fx(r["abserr"])))
            s["by"][v] = {"ratio": ratio, "converged": r["converged"],
                          "verdict": ("not scored" if not is_scored else
                                      ("unconverged" if not r["converged"] else ("dishonest" if ratio > 1 else "honest")))}
        if sid in a07:
            r = a07[sid]
            ratio = float(abs(mpf(fx(r["value"])) - rr["ref"]) / mpf(fx(r["abserr"])))
            s["by"]["07a"] = {"ratio": ratio, "converged": r["converged"],
                              "verdict": ("not scored" if not is_scored else
                                          ("unconverged" if not r["converged"] else ("dishonest" if ratio > 1 else "honest")))}
        scored[sid] = s
    # descent (table 4) from instrumented replays
    descent = {}
    d1p = ROOT / "out" / "d1" / "variants_descent.json"
    if d1p.exists():
        descent = json.loads(d1p.read_text())
    descent["tree"] = json.loads((ROOT / "out" / "d1" / "d1_descent.json").read_text())["calls"]  # the instrument, equal to the tree
    chase_sids = sorted(s for s, c in man.items() if c["set"] == "chase") + ["a:R2-28", "a:R2-28~o", "r31-3-20363-1"]
    t3, t4, t5, qual = {}, {}, {}, {}
    for v in variants:
        sc = [s for s in scored.values() if v in s["by"] and s["scored"]]
        t3[v] = {"converged_honest": sum(1 for s in sc if s["by"][v]["verdict"] == "honest"),
                 "converged_dishonest": sorted(k for k, s in scored.items() if v in s["by"] and s["by"][v]["verdict"] == "dishonest"),
                 "unconverged": {k: s["by"][v]["ratio"] for k, s in scored.items() if v in s["by"] and s["by"][v]["verdict"] == "unconverged"},
                 "not_scored": sorted(k for k, s in scored.items() if v in s["by"] and not s["scored"]),
                 "n_referenced": sum(1 for s in scored.values() if v in s["by"])}
        rows = [(s, table[s][v]) for s in chase_sids if v in table.get(s, {})]
        unc = [s for s, r in rows if not r["converged"]]
        dv = descent.get(v, {})
        t4[v] = {"n": len(rows), "unconverged": len(unc),
                 "unconverged_all_depth20_below_refused": (sum(1 for s in unc if dv.get(s, {}).get("all_depth20_below_refused")) if dv else None),
                 "regions": sum(r["regions"] for _, r in rows), "wall_s": sum(r["wall_s"] for _, r in rows),
                 "unconverged_sids": unc}
        allrows = [(s, table[s][v], table[s].get("tree")) for s in table if v in table[s]]
        touched = [s for s, r, t in allrows if t is not None and any(r[k] != t[k] for k in ("value", "abserr", "converged", "regions", "max_depth"))]
        t5[v] = {"n": len(allrows), "regions": sum(r["regions"] for _, r, _ in allrows), "wall_s": sum(r["wall_s"] for _, r, _ in allrows),
                 "regions_tree": sum(t["regions"] for _, _, t in allrows if t), "wall_s_tree": sum(t["wall_s"] for _, _, t in allrows if t),
                 "touched": len(touched), "touched_sids": touched}
    for v in variants[1:]:
        new_bad = sorted(set(t3[v]["converged_dishonest"]) - set(t3["tree"]["converged_dishonest"]))
        qual[v] = {"new_converged_dishonest": new_bad, "fewer_chase_unconverged": t4[v]["unconverged"] < t4["tree"]["unconverged"],
                   "qualifies": not new_bad and t4[v]["unconverged"] < t4["tree"]["unconverged"]}
    r5v = {k: {c: s["by"].get(c, {}).get("verdict") for c in ("07a", "tree")} | {"ratio_07a": s["by"].get("07a", {}).get("ratio"),
                                                                                 "ratio_07b": s["by"].get("tree", {}).get("ratio")}
           for k, s in scored.items() if s["set"] in ("r5", "r5~o")}
    hdr = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "variants": variants}
    doc = {"header": hdr, "refs": {k: {"ref": mp.nstr(v["ref"], 25), "spread": v["spread"], "basis": v["basis"]} for k, v in R.items()},
           "table": table, "scored": scored, "table3": t3, "table4": t4, "table5": t5, "qualification": qual, "r5_verdicts": r5v}
    out_j.write_text(json.dumps(doc, indent=1))
    md = ["## Table 3: honesty on every referenced call (07's rule; scored if the reference's spread <= min(abserr)/100)", "",
          "| variant | referenced | not scored | converged & honest | converged & dishonest | unconverged (ratio) |", "|---|---|---|---|---|---|"]
    for v in variants:
        t = t3[v]
        md.append(f"| {v} | {t['n_referenced']} | {len(t['not_scored'])} | {t['converged_honest']} | {len(t['converged_dishonest'])}"
                  f"{' (' + ', '.join(t['converged_dishonest']) + ')' if t['converged_dishonest'] else ''} | {len(t['unconverged'])}"
                  f"{' (' + ', '.join(f'{k} {x:.3g}' for k, x in sorted(t['unconverged'].items())) + ')' if t['unconverged'] else ''} |")
    md += ["", "## Table 4: the chase (198 calls and item 2's three)", "",
           "| variant | calls | unconverged | of which every depth-20 leaf below a refused leaf | regions | wall s |", "|---|---|---|---|---|---|"]
    for v in variants:
        t = t4[v]
        md.append(f"| {v} | {t['n']} | {t['unconverged']} | {'-' if t['unconverged_all_depth20_below_refused'] is None else t['unconverged_all_depth20_below_refused']} | {t['regions']} | {t['wall_s']:.0f} |")
    md += ["", "## Table 5: cost and reach on every call of section 2.1", "",
           "| variant | calls | regions | regions / tree | wall s | wall / tree | calls touched |", "|---|---|---|---|---|---|---|"]
    for v in variants:
        t = t5[v]
        md.append(f"| {v} | {t['n']} | {t['regions']} | {t['regions'] / t['regions_tree']:.3f} | {t['wall_s']:.0f} | {t['wall_s'] / t['wall_s_tree']:.3f} | {t['touched']} |")
    md += ["", "## Qualification", "", "| variant | new converged & dishonest | fewer chase calls unconverged | qualifies |", "|---|---|---|---|"]
    for v, q in qual.items():
        md.append(f"| {v} | {', '.join(q['new_converged_dishonest']) or '-'} | {'yes' if q['fewer_chase_unconverged'] else 'no'} | {'yes' if q['qualifies'] else 'no'} |")
    md += ["", "## R5's verdicts at 07a's and 07b's code (beside the table)", "", "| call | 07a | ratio | 07b | ratio |", "|---|---|---|---|---|"]
    for k, x in sorted(r5v.items()):
        md.append(f"| {k} | {x['07a']} | {x['ratio_07a']:.3g} | {x['tree']} | {x['ratio_07b']:.3g} |")
    md += ["", f"script sha256 {sha256(__file__)}"]
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md[:40]))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
