"""levin-phase-floor prompt 07a section 6, the user's references (state/ruling_w5.md item 2): score the two rows' calls.
New in 07a, after 07's score_v4_07.py (a record, not edited), whose reading-usability check and exact arithmetic it keeps.
Read only.

- **Readings:** out/refs/readings/<sid>-<impl>-r<res>.json for the four captured calls (the 1e-7 capture of each row and
  index; the 1e-8 call's integrand is the same bit for bit, out/refs/integrands.json). A reading is usable if made by the
  tool d37e633b..., on the capture out/refs/inputs.json lists (SHA-256 checked), Route B, Ray not initialised, every
  non-monotone segment integrated directly, with a value (07's check, unchanged).
- **The reference:** FCC r2's ``value_30``. **Its stated error:** the spread of the call's usable readings (largest minus
  smallest ``value_30``), 07's agreement measure.
- **Each selected call (row, index, rtol):** |v - ref| / abserr at h (07a's census) and at 07 (07's census), each with its
  abserr; and whether the stated error is at least 10x below both abserrs (spread <= min(abserr_h, abserr_07)/10). Where
  it is not, the call is reported and not scored. Also 07's own agreement test (spread <= min/100) beside it.
Exact: doubles as Fractions from their hex, readings as Fractions of their 30-digit strings.
Usage: python -B score_refs_07a.py OUT_JSON OUT_MD
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

SEL = WIDER / "state" / "refs" / "selection.json"
INP = WIDER / "out" / "refs" / "inputs.json"
INTEG = WIDER / "out" / "refs" / "integrands.json"
PLAN = WIDER / "state" / "plan_references.json"
RD = WIDER / "out" / "refs" / "readings"
TOOL_SHA256 = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"


def FH(h):
    return Fraction(float.fromhex(h))


def main():
    out_json, out_md = Path(sys.argv[1]), Path(sys.argv[2])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: the user's references scored", [SEL, INP, INTEG, PLAN])
    inputs = json.loads(INP.read_text())["inputs"]
    if not all(v["all_equal"] for v in json.loads(INTEG.read_text())["across_rtol"].values()):
        raise SystemExit("the integrands differ across the rtols")
    plan = {j["name"] for j in json.loads(PLAN.read_text())["stages"][0]["jobs"]}
    refs = {}
    for sid, e in sorted(inputs.items()):
        readings, notes, walls = {}, [], {}
        for impl, r in (("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)):
            name = f"{sid}-{impl}-r{r}"
            if name not in plan:
                continue
            p = RD / f"{name}.json"
            if not p.exists():
                notes.append(f"{name}: missing")
                continue
            d = json.loads(p.read_text())
            why = []
            if d.get("tool_sha256") != TOOL_SHA256:
                why.append("tool sha256")
            if d.get("pickle_sha256") != e["sha256"] or sha256(d["pickle"]) != e["sha256"]:
                why.append("pickle sha256")
            if d.get("route") != "B":
                why.append("route")
            ri = d.get("ray_initialized")
            if (any(bool(x) for x in ri) if isinstance(ri, list) else ri):
                why.append("ray initialised")
            if any((not s.get("monotone")) and not s.get("direct") for s in d.get("segments") or []):
                why.append("a non-monotone segment not integrated directly")
            if d.get("value_30") is None:
                why.append("no value")
            if why:
                notes.append(f"{name}: unusable ({', '.join(why)})")
                continue
            readings[f"{impl}_r{r}"] = Fraction(d["value_30"])
            walls[f"{impl}_r{r}"] = d["wall_s"]
        if "fcc_r2" not in readings:
            raise SystemExit(f"{sid}: no usable FCC r2 reading ({notes})")
        vals = list(readings.values())
        refs[sid[: -len("-1e-7-2")] + sid[-2:]] = {"capture": sid, "ref": readings["fcc_r2"], "spread": max(vals) - min(vals),
                                                   "readings": readings, "notes": notes, "walls": walls}
    calls = []
    for c in json.loads(SEL.read_text()):
        key = c["sid"][: -len("-1e-7-2")] + c["sid"][-2:]
        R = refs[key]
        vh, eh, v7, e7 = FH(c["census_value"]), FH(c["census_abserr"]), FH(c["census07_value"]), FH(c["census07_abserr"])
        ok10 = R["spread"] <= min(eh, e7) / 10
        row = {"sid": c["sid"], "unit": c["unit"], "index": c["index"], "class": c["class"], "label": c["notify_label"],
               "reference_capture": R["capture"], "ref_30": str(float(R["ref"])), "ref_exact_30": None,
               "stated_error_spread": float(R["spread"]), "readings": {k: str(float(v)) for k, v in R["readings"].items()},
               "reading_notes": R["notes"], "reading_walls_s": R["walls"],
               "value_h": c["census_value"], "abserr_h": c["census_abserr"], "abserr_h_float": float(eh), "converged_h": c["census_converged"],
               "value_07": c["census07_value"], "abserr_07": c["census07_abserr"], "abserr_07_float": float(e7), "converged_07": c["census07_converged"],
               "regions_h_07": [c["census_num_regions"], c["census07_num_regions"]],
               "stated_error_over_min_abserr": float(R["spread"] / min(eh, e7)),
               "stated_error_10x_below_both": ok10, "readings_agree_07_rule": R["spread"] <= min(eh, e7) / 100}
        if ok10:
            row["ratio_h"] = float(abs(vh - R["ref"]) / eh)
            row["ratio_07"] = float(abs(v7 - R["ref"]) / e7)
            row["honest_h"] = abs(vh - R["ref"]) <= eh
            row["honest_07"] = abs(v7 - R["ref"]) <= e7
        else:
            row["verdict"] = "the stated error is not 10x below both abserrs: reported, not scored"
        calls.append(row)
    out_json.write_text(json.dumps({"header": hdr, "rule": __doc__, "calls": calls}, indent=1, default=str))
    L = hdr + ["", "# The user's references (rows 3-678 and 1-34599)", "",
               "| call | class | ref (FCC r2) | stated error (spread) | spread / min(abserr) | abserr_h | abserr_07 | "
               "|v_h - ref|/abserr_h | |v_07 - ref|/abserr_07 | regions h / 07 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
    for r in calls:
        L.append(f"| {r['sid']} | {r['class']} | {r['ref_30']} | {r['stated_error_spread']:.3g} | {r['stated_error_over_min_abserr']:.3g} | "
                 f"{r['abserr_h_float']:.4g} | {r['abserr_07_float']:.4g} | "
                 + (f"{r['ratio_h']:.4g} | {r['ratio_07']:.4g}" if "ratio_h" in r else "not scored | not scored")
                 + f" | {r['regions_h_07'][0]} / {r['regions_h_07'][1]} |")
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L[len(hdr):]))
    print(f"wrote {out_json} sha256 {sha256(out_json)}; {out_md} sha256 {sha256(out_md)}")


if __name__ == "__main__":
    main()
