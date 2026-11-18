"""levin-phase-floor prompt 07a section 6, W4: V4's 33 calls re-scored from 07a's census h against the readings 07
scored them on, by 07's rule. Copied from 07's census score_v4_07.py (a record, not edited), changed only where marked
"prompt 07a": "the new" is h's census (the run given), "the before" is 07's census (its checkpoint read by
state/index_07.json, each line's SHA-256 checked); 07's V4 records are read from 07's working records
(var/levin-phase-floor-07/census/: state/plan_references.json, state/v4_reading_list_20261109.json, state/v4_selection.json,
out/v4/inputs.json, out/v4/readings/; read only); 06g's scores from the tree h. The tolerance is therefore
min(abserr_07, abserr_h)/100 (state/predictions.md part 5), and each call also carries 07's own recorded verdict,
tolerance and ratio (07's v4_scores.json) beside it. Keys keep 07's names: ``_g`` is h, ``_06m`` is 07.
Usage: python -B score_v4_07a.py RUN_DIR OUT_JSON OUT_MD

07's text follows.
levin-phase-floor prompt 07 section 6, V4: score the 24 calls against their references, and z00, z12 and Z3's seven
against 06g's, and give "the stationary-point calls before". New in 07. Read only.

**The rule** (state/predictions.md, V4, as the user ruled at stop 4; the log's deviation 27):
- the tolerance of a call is min(abserr_06m, abserr_g)/100, both the call's own census abserrs at its rtol;
- a call with QAWO r1, r2 and FCC r1, r2 is scored as written: its readings agree when their spread (largest minus
  smallest real part, ``value_30``) is at most the tolerance; the reference is FCC r2;
- a call without QAWO is scored on the FCC readings it has: with both, the reference is FCC r2 and the two must agree
  within the same tolerance; with only one, against that reading, with no agreement test;
- a call whose readings do not agree is reported, not scored; a scored call is honest iff |v_g - ref| <= abserr_g.
- **A reading is usable** if it is present, made by the tool at g (SHA-256 d37e633b...), on the capture listed in
  out/v4/inputs.json (SHA-256 checked), Route B, with Ray not initialised, and every non-monotone segment of its own
  table integrated directly (``direct``; the tool's own condition for a sound reading of a segment with a stationary
  point). An unusable or missing reading is recorded, and the call is scored on the rest by the rule above.
- **z00, z12 and Z3's seven** are scored on 06g's readings as scores_stage3/scores.json records them (``rows``: each
  reading's ``value_30``; the readings 06g marked ``unusable`` left out; the reference its ``ref_30``, FCC r2), with this
  rule's tolerance and agreement test, against 07's census value and abserr; 06m's from 06m's census.
- **"The stationary-point calls before":** |v_06m - ref| / abserr_06m for every scored call.
Every comparison is exact: doubles as Fractions from their hex, readings as Fractions of their 30-digit decimal strings.

Usage: python -B score_v4_07.py OUT_JSON OUT_MD
"""

import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import RUN06M, WIDER, header, sha256  # noqa: E402
from prov import CHECKPOINT07_SHA256, H_TREE, REC07, RUN07  # noqa: E402  prompt 07a

V4 = REC07 / "out" / "v4"  # prompt 07a: 07's records
PLAN = REC07 / "state" / "plan_references.json"  # prompt 07a
LIST = REC07 / "state" / "v4_reading_list_20261109.json"  # prompt 07a
INPUTS = V4 / "inputs.json"
SEL = REC07 / "state" / "v4_selection.json"  # prompt 07a
SCORES06G = H_TREE / "docs" / "levin-phase-floor" / "06g" / "references" / "out" / "scores_stage3" / "scores.json"
INDEX06M = WIDER / "state" / "index_07.json"  # prompt 07a: the before is 07's census
V4_07 = REC07 / "out" / "v4" / "v4_scores.json"  # prompt 07a: 07's own scores, beside
TOOL_SHA256 = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
Z = ("z00", "z12", "z07", "z16", "z18", "z21", "z24", "z28", "z29")


def FH(h):
    return Fraction(float.fromhex(h))


def FD(s):
    return Fraction(s)


def census_call(ck, off, kind, index):
    with open(ck, "rb") as f:
        f.seek(off[0])
        line = f.read(off[1])
    d = json.loads(line)["data"]
    key = "calls" if kind == "group" else "other_calls"
    c = [c for c in d[key] if c["index"] == index]
    if len(c) != 1:
        raise SystemExit(f"{d['unit']} #{index}: not one call")
    return c[0]


def offsets07(units, ck=None):  # prompt 07a: the run's checkpoint (h's), its units h-... keyed as 07's g-...
    out, off = {}, 0
    with open(ck, "rb") as f:
        for line in f:
            uid = line[10:60].decode().split('"', 1)[0]
            gu = "g-" + uid.split("-", 1)[1]
            if gu in units:
                out[gu] = (off, len(line))
            off += len(line)
    return out


def score(readings, kinds_ref, vg, eg, vm, em):
    """readings: {kind: Fraction}; kinds_ref: the reference kind. Returns the verdict dict."""
    tol = min(eg, em) / 100
    vals = list(readings.values())
    spread = max(vals) - min(vals) if len(vals) > 1 else None
    if len(vals) == 0:
        return {"verdict": "no usable reading", "tol": float(tol)}
    if len(vals) > 1 and spread > tol:
        return {"verdict": "readings disagree (reported, not scored)", "tol": float(tol), "spread": float(spread),
                "spread_over_tol": float(spread / tol)}
    ref = readings[kinds_ref] if kinds_ref in readings else vals[0]
    dg, dm = abs(vg - ref), abs(vm - ref)
    return {"verdict": "honest" if dg <= eg else "NOT honest", "tol": float(tol),
            "spread": None if spread is None else float(spread), "spread_over_tol": None if spread is None else float(spread / tol),
            "ref_kind": kinds_ref if kinds_ref in readings else list(readings)[0], "ref": float(ref),
            "ratio_g": float(dg / eg), "ratio_06m": float(dm / em), "honest_06m": dm <= em}


def main():
    run_h, out_json, out_md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])  # prompt 07a
    CKH = run_h / "checkpoint.jsonl"  # prompt 07a
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: W4 scored (V4's 33 calls at h, on 07's readings)",
                 [PLAN, LIST, INPUTS, SEL, SCORES06G, CKH, INDEX06M, V4_07])  # prompt 07a
    if json.loads(INDEX06M.read_text())["checkpoint_sha256"] != CHECKPOINT07_SHA256 or sha256(RUN07 / "checkpoint.jsonl") != CHECKPOINT07_SHA256:
        raise SystemExit("07's census checkpoint has another SHA-256")  # prompt 07a
    v4_07 = {x["sid"]: x for x in (lambda d: d["calls"] + d["z"])(json.loads(V4_07.read_text()))}  # prompt 07a
    plan_jobs = {j["name"] for j in json.loads(PLAN.read_text())["stages"][0]["jobs"]}
    inputs = json.loads(INPUTS.read_text())["inputs"]
    sel = {c["sid"]: c for c in json.loads(SEL.read_text())["calls"]}
    rows06g = {r["sid"]: r for r in json.loads(SCORES06G.read_text())["rows"]}
    idx06 = json.loads(INDEX06M.read_text())["units"]
    units = {c["unit"] for c in sel.values()} | {rows06g[z]["unit"] for z in Z}
    off07 = offsets07(units, CKH)  # prompt 07a: h's offsets, by 07's unit names
    res = {"header": hdr, "rule": __doc__, "calls": [], "z": []}
    for sid in sorted(sel):
        c = sel[sid]
        g = census_call(CKH, off07[c["unit"]], "group", c["index"])  # prompt 07a: h
        e6 = idx06[c["unit"]]
        m = census_call(RUN07 / "checkpoint.jsonl", (e6["offset"], e6["length"]), "group", c["index"])  # prompt 07a: 07
        vg, eg, vm, em = FH(g["value"]["hex"]), FH(g["abserr"]["hex"]), FH(m["value"]["hex"]), FH(m["abserr"]["hex"])
        readings, notes = {}, []
        for impl, r in (("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)):
            name = f"{sid}-{impl}-r{r}"
            if name not in plan_jobs:
                continue
            p = V4 / "readings" / f"{name}.json"
            if not p.exists():
                notes.append(f"{name}: missing")
                continue
            d = json.loads(p.read_text())
            why = []
            if d.get("tool_sha256") != TOOL_SHA256:
                why.append("tool sha256")
            if d.get("pickle_sha256") != inputs[sid]["sha256"] or sha256(d["pickle"]) != inputs[sid]["sha256"]:
                why.append("pickle sha256")
            if d.get("route") != "B":
                why.append("route")
            if any(bool(x) for x in (d.get("ray_initialized") or [])) if isinstance(d.get("ray_initialized"), list) else d.get("ray_initialized"):
                why.append("ray initialised")
            if any((not s.get("monotone")) and not s.get("direct") for s in d.get("segments") or []):
                why.append("a non-monotone segment not integrated directly")
            if d.get("value_30") is None:
                why.append("no value")
            if why:
                notes.append(f"{name}: unusable ({', '.join(why)})")
                continue
            readings[f"{impl}_r{r}"] = FD(d["value_30"])
        basis = ("four readings" if len(readings) == 4 else "FCC r1 and r2" if set(readings) == {"fcc_r1", "fcc_r2"}
                 else "one FCC" if len(readings) == 1 and list(readings)[0].startswith("fcc") else f"other: {sorted(readings)}")
        if basis.startswith("other"):
            raise SystemExit(f"{sid}: a basis the rule does not decide: {sorted(readings)} ({notes})")
        s = score(readings, "fcc_r2", vg, eg, vm, em)
        res["calls"].append({"sid": sid, "unit": c["unit"], "rtol": c["rtol"], "index": c["index"], "band": c["band"],
                             "k": c["k"], "basis": basis, "notes": notes, "value_g": g["value"]["hex"],
                             "abserr_g": g["abserr"]["hex"], "value_06m": m["value"]["hex"], "abserr_06m": m["abserr"]["hex"],
                             "converged_g": g["converged"], "converged_06m": m["converged"],
                             "readings_30": {k: str(float(v)) for k, v in readings.items()},
                             "wall_s": {f"{i}_r{r}": json.loads((V4 / "readings" / f"{sid}-{i}-r{r}.json").read_text())["wall_s"]
                                        for i, r in (("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2))
                                        if (V4 / "readings" / f"{sid}-{i}-r{r}.json").exists()}, **s})
    for z in Z:
        r = rows06g[z]
        kind = r["kind"]
        g = census_call(CKH, off07[r["unit"]], kind, r["index"])  # prompt 07a: h
        e6 = idx06[r["unit"]]
        m = census_call(RUN07 / "checkpoint.jsonl", (e6["offset"], e6["length"]), kind, r["index"])  # prompt 07a: 07
        if g["notify_label"] != r["label"] or m["notify_label"] != r["label"]:
            raise SystemExit(f"{z}: label differs")
        vg, eg, vm, em = FH(g["value"]["hex"]), FH(g["abserr"]["hex"]), FH(m["value"]["hex"]), FH(m["abserr"]["hex"])
        unusable = set(r.get("unusable") or [])
        readings = {k: FD(v["value_30"]) for k, v in r["readings"].items() if k not in unusable and v.get("value_30")}
        if r.get("ref_30") is None or FD(r["ref_30"]) != readings.get("fcc_r2"):
            raise SystemExit(f"{z}: 06g's ref_30 is not its FCC r2 reading")
        s = score(readings, "fcc_r2", vg, eg, vm, em)
        res["z"].append({"sid": z, "unit": r["unit"], "rtol": r["rtol"], "kind": kind, "index": r["index"], "label": r["label"],
                         "basis": f"06g's readings: {sorted(readings)}", "unusable_06g": sorted(unusable),
                         "value_g": g["value"]["hex"], "abserr_g": g["abserr"]["hex"], "value_06m": m["value"]["hex"],
                         "abserr_06m": m["abserr"]["hex"], "converged_g": g["converged"], "converged_06m": m["converged"],
                         "ratio_06g_record": r.get("r_g"), **s})
    for x in res["calls"] + res["z"]:  # prompt 07a: 07's own score of the call, beside
        o = v4_07[x["sid"]]
        x["score_07"] = {k: o.get(k) for k in ("verdict", "tol", "ratio_g", "spread_over_tol", "value_g", "abserr_g", "converged_g")}
        x["value_equal_07"] = x["value_g"] == o["value_g"]
        x["abserr_equal_07"] = x["abserr_g"] == o["abserr_g"]
    allc = res["calls"] + res["z"]
    tot = {}
    for x in allc:
        tot[x["verdict"]] = tot.get(x["verdict"], 0) + 1
    res["totals"] = {"24": {v: sum(1 for x in res["calls"] if x["verdict"] == v) for v in sorted({x["verdict"] for x in res["calls"]})},
                     "z": {v: sum(1 for x in res["z"] if x["verdict"] == v) for v in sorted({x["verdict"] for x in res["z"]})},
                     "all": tot,
                     "basis_24": {b: sum(1 for x in res["calls"] if x["basis"] == b) for b in sorted({x["basis"] for x in res["calls"]})}}
    scored = [x for x in allc if "ratio_g" in x]
    res["V4"] = {"scored": len(scored), "honest": sum(1 for x in scored if x["verdict"] == "honest"),
                 "not_honest": [x["sid"] for x in scored if x["verdict"] != "honest"],
                 "reported_not_scored": [x["sid"] for x in allc if "ratio_g" not in x],
                 "holds": all(x["verdict"] == "honest" for x in scored) and bool(scored)}
    res["before"] = {"honest_06m": sum(1 for x in scored if x["honest_06m"]), "of": len(scored),
                     "ratio_06m": {x["sid"]: x["ratio_06m"] for x in scored}}
    out_json.write_text(json.dumps(res, indent=1, default=str))
    L = [l for l in hdr] + ["", "# V4 scored", "",
                            f"V4: {'HOLDS' if res['V4']['holds'] else 'FAILS'}: {res['V4']['honest']} of {res['V4']['scored']} scored "
                            f"calls honest; not honest {res['V4']['not_honest']}; reported, not scored {res['V4']['reported_not_scored']}.",
                            f"Totals: {json.dumps(res['totals'])}", "",
                            "| call | unit (07's name) | rtol | basis | |v_h - ref|/abserr_h | |v_07 - ref|/abserr_07 | spread/tol | verdict | notes |",
                            "|---|---|---|---|---|---|---|---|---|"]
    for x in allc:
        L.append(f"| {x['sid']} | {x['unit']} | {x['rtol']} | {x['basis']} | {x.get('ratio_g', '-'):.4g} | {x.get('ratio_06m', '-'):.4g} | "
                 f"{x.get('spread_over_tol') if x.get('spread_over_tol') is None else format(x['spread_over_tol'], '.3g')} | {x['verdict']} | "
                 f"{'; '.join(x.get('notes') or [])} |" if "ratio_g" in x else
                 f"| {x['sid']} | {x['unit']} | {x['rtol']} | {x['basis']} | - | - | {x.get('spread_over_tol')} | {x['verdict']} | "
                 f"{'; '.join(x.get('notes') or [])} |")
    L += ["", f"The before (07's census): honest on {res['before']['honest_06m']} of {res['before']['of']} scored calls.",  # prompt 07a
          "", "Beside 07's own scores (07's v4_scores.json): value and abserr at h equal 07's (hex), 07's verdict and ratio:", ""]
    L += [f"- {x['sid']}: value equal {x['value_equal_07']}, abserr equal {x['abserr_equal_07']}; 07: {x['score_07']['verdict']}, "
          f"{x['score_07']['ratio_g']:.4g}" for x in allc]
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L[len(hdr):]))
    print(f"wrote {out_json} sha256 {sha256(out_json)}; {out_md} sha256 {sha256(out_md)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
