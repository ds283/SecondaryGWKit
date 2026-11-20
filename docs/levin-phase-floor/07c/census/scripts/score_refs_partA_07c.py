"""levin-phase-floor prompt 07c section 6, Y4 (b): 07c Part A's referenced census calls, R4 (60), R5 (60) and R2-28's census
call at both rtols, scored from 07c's census against Part A's references (state/predictions.md part 5). New in 07c. Read
only.

- **The calls:** every entry of Part A's out/tables/scores_07c.json ``scored`` whose ``ref_key`` starts ``ref-R4-`` or
  ``ref-R5-``, and ``c-1e-7-0-13757-5`` and ``c-1e-8-0-13757-5`` (``ref-R2-28``). Each id's census call (unit, kind,
  index) is read from Part A's mappings (out/census_refs_07c.json ``calls``; state/census_wanted_07c*.json ``wanted``).
- **The reference and its stated error** are Part A's (``ref``, a decimal string; ``spread``).
- **The rule:** scored iff spread <= min(abserr)/100 over this census's abserr and 07b's census's for the call (and, for
  R5, 07a's census's, read from 07b's census's stop-3 call list, out/stop3/compare_census_calls.calls.jsonl, its second
  column); otherwise "reported, not scored". A scored call unconverged here is "unconverged"; otherwise honest iff
  |v - ref| <= abserr, else "converged and NOT honest". Ratios are computed exactly (Fractions of the hex values and of the
  reference's decimal string) and printed as floats.

Writes OUT_JSON and prints a summary. Usage: python -B score_refs_partA_07c.py RUN_DIR OUT_JSON
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, REC07B, RUN07B, header, sha256  # noqa: E402

PART_A = MAIN / "var" / "levin-phase-floor-07c" / "measurement"
SCORES = PART_A / "out" / "tables" / "scores_07c.json"
MAPS = [(PART_A / "out" / "census_refs_07c.json", "calls"), (PART_A / "state" / "census_wanted_07c.json", "wanted"),
        (PART_A / "state" / "census_wanted_07c_chase.json", "wanted"), (PART_A / "state" / "census_wanted_07c_r5.json", "wanted")]
CALLS07B = REC07B / "out" / "stop3" / "compare_census_calls.calls.jsonl"
R2_28 = ("c-1e-7-0-13757-5", "c-1e-8-0-13757-5")


def FX(h):
    return Fraction(float.fromhex(h))


def reader(ck, units):
    out = {}
    with open(ck) as f:
        for line in f:
            u = line[10:60].split('"', 1)[0]
            if u in units:
                out[u] = json.loads(line)["data"]
    return out


def call_of(d, kind, index):
    cs = [c for c in d.get("calls" if kind == "group" else "other_calls") or [] if c["index"] == index]
    if len(cs) != 1:
        raise SystemExit(f"{d['unit']} {kind} #{index}: {len(cs)} calls")
    return cs[0]


def main():
    run, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    ck = run / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: Y4 (b), Part A's R4, R5 and R2-28, scored from 07c's census",
                 [SCORES] + [p for p, _ in MAPS] + [CALLS07B])
    m = {}
    for p, key in MAPS:
        for k, v in json.loads(p.read_text())[key].items():
            t = (v["unit"], v["kind"], v["index"])
            if k in m and m[k] != t:
                raise SystemExit(f"{k} mapped two ways")
            m[k] = t
    sc = json.loads(SCORES.read_text())["scored"]
    want = {k: v for k, v in sc.items()
            if (v.get("ref_key") or "").startswith(("ref-R4-", "ref-R5-")) or k in R2_28}
    sets = {k: ("R2-28" if k in R2_28 else "R4" if v["ref_key"].startswith("ref-R4-") else "R5") for k, v in want.items()}
    keys = {k: m[k] for k in want}
    units = {t[0] for t in keys.values()}
    mine = reader(ck, units)
    theirs = reader(RUN07B / "checkpoint.jsonl", units)
    a07 = {}
    for line in open(CALLS07B):
        c = json.loads(line)
        t = (c["unit"], c["kind"], c["index"])
        if t in set(keys.values()):
            a07[t] = c["abserr"][1]
    rows = {}
    for sid in sorted(want):
        t = keys[sid]
        c, c7 = call_of(mine[t[0]], t[1], t[2]), call_of(theirs[t[0]], t[1], t[2])
        ref = Fraction(want[sid]["ref"])
        spread = Fraction(want[sid]["spread"])
        v, e, v7, e7 = FX(c["value"]["hex"]), FX(c["abserr"]["hex"]), FX(c7["value"]["hex"]), FX(c7["abserr"]["hex"])
        es = [e, e7] + ([FX(a07[t])] if sets[sid] == "R5" else [])
        scored = spread <= min(es) / 100
        ratio = abs(v - ref) / e
        verdict = ("reported, not scored" if not scored else "unconverged" if not c["converged"]
                   else "honest" if ratio <= 1 else "converged and NOT honest")
        rows[sid] = {"set": sets[sid], "unit": t[0], "kind": t[1], "index": t[2], "ref_key": want[sid]["ref_key"],
                     "ref": want[sid]["ref"], "spread": want[sid]["spread"], "value": c["value"]["hex"],
                     "abserr": c["abserr"]["hex"], "converged": c["converged"], "regions": c["num_regions"],
                     "max_depth": c["max_depth"], "chase_counts": [c.get("num_chase_stops"), c.get("num_chase_stopped_leaves"),
                                                                    c.get("num_chase_noise_bisections")],
                     "ratio": float(ratio), "ratio_07b": float(abs(v7 - ref) / e7), "converged_07b": c7["converged"],
                     "abserr_07a": a07.get(t) if sets[sid] == "R5" else None, "scored": scored, "verdict": verdict,
                     "partA_S_ratio": (want[sid]["by"].get("S") or {}).get("ratio"),
                     "partA_S_verdict": (want[sid]["by"].get("S") or {}).get("verdict")}
    tally = {}
    for x in rows.values():
        tally.setdefault(x["set"], {}).setdefault(x["verdict"], 0)
        tally[x["set"]][x["verdict"]] += 1
    bad = sorted(k for k, x in rows.items() if x["verdict"] == "converged and NOT honest")
    doc = {"header": hdr, "rule": __doc__, "n": len(rows), "by_set": tally, "converged_dishonest": bad,
           "converged_dishonest_ratios": {k: rows[k]["ratio"] for k in bad},
           "converged_dishonest_other_than_r2_28": [k for k in bad if k not in R2_28],
           "unconverged": {k: x["ratio"] for k, x in rows.items() if x["verdict"] == "unconverged"},
           "not_scored": sorted(k for k, x in rows.items() if not x["scored"]),
           "max_ratio_honest": max((x["ratio"] for x in rows.values() if x["verdict"] == "honest"), default=None),
           "n_distinct_census_calls": len({(x["unit"], x["kind"], x["index"]) for x in rows.values()}),
           "verdict_equals_partA_S": sum(1 for x in rows.values() if x["verdict"] == x["partA_S_verdict"]
                                         or (x["verdict"] == "converged and NOT honest" and x["partA_S_verdict"] == "dishonest")),
           "calls": rows}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("header", "rule", "calls")}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
