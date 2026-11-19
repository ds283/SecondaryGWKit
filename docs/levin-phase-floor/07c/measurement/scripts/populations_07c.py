"""levin-phase-floor prompt 07c Part A, section 2.1: the populations, recomputed from 07b's census comparison, read only.
New in 07c. (v2: the pairs read as [07b, 07a]; v1, which read them as [07a, 07b], is in scripts/superseded/.)

Reads var/levin-phase-floor-07b/census/out/stop3/compare_census_calls.calls.jsonl (one line per Levin call of 07b's census,
07b and 07a side by side: value, abserr, converged, num_regions, max_depth as [07b, 07a], the order of 07b's
compare_07b.py, ``[s, s7]`` with s the census's own record and s7 the "before"; its SHA-256 is checked against
07b's census KEPT_ON_MAC_STUDIO.txt line before and after.

- **The chase (item 1):** group calls converged at 07a and unconverged at 07b. Compared with 07b's census
  ``summary.json`` ``newly_unconverged`` (the census agent's list): the two must be equal as sets of (unit, index).
- **Item 6, the 324:** calls with |v_07b - v_07a| > abserr_07a + abserr_07b (the sum; exact, by Fractions of the hex
  values) and converged under both, the rule of ``docs/levin-phase-floor/07b/review/final/scripts/moves.py``.
- **The referenced among the 324:** a call is referenced if its (rtol, shard, serial, kind, index) is one of 07b's
  census U4 (u4_partA, u4_07a_rows, u4_v4_scores ``calls`` and ``z``) or U6 calls (``u4_u6_scores.json``,
  ``u4_v4_scores.json``). VERIFICATION.md section 7b.5 counts 18; the rest are R5's population (306).

Writes state/populations_07c.json. Usage: python -B populations_07c.py
"""

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import MAIN, WIDER, header, sha256  # noqa: E402

TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f")
CMP = MAIN / "var/levin-phase-floor-07b/census/out/stop3/compare_census_calls.calls.jsonl"
CMP_SHA = "a327106d0b0890fd9dd4b8a85dc7e64aec8340579711ff006fe86230bf25ad8d"
SUMMARY = TREE / "docs/levin-phase-floor/07b/census/out/scores/summary.json"
U4U6 = TREE / "docs/levin-phase-floor/07b/census/out/scores/u4_u6_scores.json"
U4V4 = TREE / "docs/levin-phase-floor/07b/census/out/scores/u4_v4_scores.json"


def key_of(unit, kind, index):
    p = unit.split("-")
    return ("-".join(p[1:-2]), p[-2], p[-1], kind, int(index))


def fx(h):
    return F(float.fromhex(h))


def main():
    out = WIDER / "state" / "populations_07c.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    if sha256(CMP) != CMP_SHA:
        raise SystemExit("compare_census_calls.calls.jsonl has another SHA-256")
    chase, moved, n = [], [], 0
    with open(CMP) as f:
        for line in f:
            c = json.loads(line)
            n += 1
            conv = c["converged"]
            rec = {"unit": c["unit"], "rtol": c["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"],
                   "value": c["value"], "abserr": c["abserr"], "converged": conv, "num_regions": c["num_regions"],
                   "max_depth": c["max_depth"], "num_edge_rule_refusals": c.get("num_edge_rule_refusals"),
                   "num_range_rule_refusals": c.get("num_range_rule_refusals")}
            if c["kind"] == "group" and conv[1] and not conv[0]:  # [07b, 07a]: converged at 07a, not at 07b
                chase.append(rec)
            dv = abs(fx(c["value"][0]) - fx(c["value"][1]))
            eb, ea = fx(c["abserr"][0]), fx(c["abserr"][1])  # [07b, 07a]
            if dv > ea + eb:
                rec = dict(rec, dv_over_abserr_07a=float(dv / ea), dv_over_abserr_07b=float(dv / eb),
                           dv_over_sum=float(dv / (ea + eb)), smaller_abserr="07a" if ea < eb else ("07b" if eb < ea else "equal"))
                moved.append(rec)
    if sha256(CMP) != CMP_SHA:
        raise SystemExit("compare_census_calls.calls.jsonl changed")
    moved_both = [m for m in moved if m["converged"][0] and m["converged"][1]]
    # the census agent's list
    nu = json.loads(SUMMARY.read_text())["newly_unconverged"]["list"]
    mine = {(c["unit"], c["index"]) for c in chase}
    theirs = {(c["unit"], c["index"]) for c in nu}
    # referenced
    ref = set()
    u = json.loads(U4U6.read_text())
    for k in ("u4_partA", "u4_07a_rows", "u6"):
        for v in u[k].values():
            ref.add(key_of(v["unit"], v.get("kind", "group"), v["index"]))
    v4 = json.loads(U4V4.read_text())
    for k in ("calls", "z"):
        for v in v4[k]:
            ref.add(key_of(v["unit"], v.get("kind", "group"), v["index"]))
    for m in moved_both:
        m["referenced"] = key_of(m["unit"], m["kind"], m["index"]) in ref
    unref = [m for m in moved_both if not m["referenced"]]
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c Part A: the populations of section 2.1", [CMP, SUMMARY, U4U6, U4V4])
    doc = {"header": hdr, "n_calls_read": n,
           "chase": {"n": len(chase), "equals_census_agents_list": mine == theirs,
                     "only_mine": sorted(mine - theirs), "only_theirs": sorted(theirs - mine), "calls": chase},
           "moved_beyond_sum": {"n": len(moved), "n_converged_both": len(moved_both),
                                "n_referenced": sum(m["referenced"] for m in moved_both), "n_unreferenced": len(unref),
                                "n_referenced_keys": len(ref), "calls": moved_both,
                                "not_converged_both": [m for m in moved if not (m["converged"][0] and m["converged"][1])]}}
    out.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print(f"calls read {n}; chase {len(chase)} (equals the census agent's list: {mine == theirs}); moved beyond the sum "
          f"{len(moved)}, converged both {len(moved_both)}, referenced {doc['moved_beyond_sum']['n_referenced']}, "
          f"unreferenced {len(unref)}")
    print(f"wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
