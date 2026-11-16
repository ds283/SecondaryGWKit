"""levin-phase-floor prompt 06g section 5, the references: the rescoring per addendum 6 (the user's
rulings of 2026-11-02/03), an **addition beside** the scores made after stage 2 of the rerun batch
(out/scores_after_stage2/, not rewritten). New in 06g; it reuses score_refs.py's rules and tables,
changed only where addendum 6 changes them:

- **The usable rule gains clause (iii):** a Route B FCC reading made by the tool at ``b5626ecb``
  (``tools/filon_reference.py`` SHA-256 d37e633b...) is usable iff every segment with ``monotone``
  False has ``direct`` True. Clauses (i) and (ii) are unchanged.
- **The 14 stationary-point samples** (w00..w11, z00, z12): FCC r1 and r2 from the laptop's batch
  (out/readings_fcc_b5626ec_laptop/), QAWO r1 and r2 from the rerun batch's stage 2
  (out/readings/<tag>-rerun.json, tool ff3e8394). Their stage-2 FCC readings are superseded and
  unscored ("superseded by the laptop's FCC at b5626ecb, addendum 6").
- **Every other sample** uses exactly the readings it had in the stage-2 scoring: 06i's for its 13,
  part 2's for the rest. The stage-3 readings that have landed since (r24 qawo r1, r53 qawo r1) are
  not used here; they are rechecked separately when stage 3 ends.
- **Added:** V3's verdict with clauses (ii) and (iii) accepted in place of all_monotone in the count
  rule as written (four usable readings within tol); and, for each of the 14, the count rule's
  verdict (all four readings within 1/100 of the minimum abserr) with its spread.

Writes out/scores_addendum6/ (scores.json, report.md). One process; reads only.
Usage: PYTHONPATH=<tree g>:<this directory> python -B score_refs_a6.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
import score_refs as S  # noqa: E402

B5626_SHA = "d37e633bf54a179114b2462529eaaa3836cccff8fe22e0250e6d36d61992c7be"
LAPTOP = RC.ROOT / "out" / "readings_fcc_b5626ec_laptop"
FOURTEEN = S.RERUN_ONLY
_usable0 = S.usable


def usable(d):
    if (
        d.get("route") == "B"
        and d.get("impl") == "fcc"
        and d.get("tool_sha256") == B5626_SHA
    ):
        segs = d.get("segments") or []
        if all(s.get("direct") for s in segs if not s.get("monotone")):
            n = sum(1 for s in segs if not s.get("monotone"))
            return True, f"(iii) b5626ecb FCC, {n} non-monotone segment(s) direct"
        return False, "not usable"
    return _usable0(d)


def readings_of(e):
    sid = e["sid"]
    if e.get("in_06i"):
        return S_readings_of0(e)
    out = {}
    if sid in FOURTEEN:
        for impl, res in S.KEYS:
            p = (
                (LAPTOP / f"{sid}-{impl}-r{res}.json")
                if impl == "fcc"
                else (RC.ROOT / "out" / "readings" / f"{sid}-{impl}-r{res}-rerun.json")
            )
            d = S.rd(p)
            if d:
                out[(impl, res)] = d
        return out, "laptop FCC at b5626ecb + rerun-batch QAWO at ff3e8394"
    for impl, res in S.KEYS:
        d = S.rd(RC.ROOT / "out" / "readings" / f"{sid}-{impl}-r{res}.json")
        if d:
            out[(impl, res)] = d
    return out, "part 2 (as in the stage-2 scoring)"


S_readings_of0 = S.readings_of
S.usable = usable
S.readings_of = readings_of
S.OUT = RC.ROOT / "out" / "scores_addendum6"


def main():
    S.main()
    rep = (S.OUT / "report.md").read_text()
    rep = rep.replace(
        "# The references, scored after stage 2 of the rerun batch",
        "# The references, rescored per addendum 6 (an addition beside the stage-2 scores)\n\n"
        f"# rescoring script score_refs_a6.py sha256 {RC.sha256(Path(__file__).resolve())} (it runs score_refs.py's rules with addendum 6's changes)",
        1,
    )
    (S.OUT / "report.md").write_text(rep)
    doc = json.loads((S.OUT / "scores.json").read_text())
    rows = {r["sid"]: r for r in doc["rows"]}
    v3 = [r for r in doc["rows"] if r["source"] in ("z3", "draw")]
    alt = [
        r
        for r in v3
        if r["scored"] and r["n_readings"] == 4 and r["spread_over_tol"] <= 1.0
    ]
    alt_viol = [r["sid"] for r in alt if r["violates"]]
    alt_verdict = "fails" if alt_viol else ("undecided" if len(alt) < 45 else "holds")
    per14 = []
    for r in doc["V1"]:
        per14.append(
            {
                "sid": r["wid"],
                "n_readings": r["n_readings"],
                "spread_over_tol": r.get("spread_over_tol"),
                "tol": r["tol"],
                "agree_within_tol": bool(
                    r["n_readings"] == 4 and r.get("spread_over_tol", 9e99) <= 1.0
                ),
                "clauses": sorted({x["clause"] for x in r["readings"].values()}),
            }
        )
    for sid in ("z00", "z12"):
        r = rows[sid]
        per14.append(
            {
                "sid": sid,
                "n_readings": r["n_readings"],
                "spread_over_tol": r.get("spread_over_tol"),
                "tol": r["tol"],
                "agree_within_tol": bool(
                    r["n_readings"] == 4 and r.get("spread_over_tol", 9e99) <= 1.0
                ),
                "clauses": sorted({x["clause"] for x in r["readings"].values()}),
            }
        )
    doc["addendum6"] = {
        "V3_with_clauses_ii_iii_accepted": {
            "verdict": alt_verdict,
            "n_counted": len(alt),
            "violations": alt_viol,
            "added_by_ii_iii": [r["sid"] for r in alt if not r["counts_as_written"]],
        },
        "count_rule_per_14": per14,
        "superseded": "the 14 stationary-point samples' stage-2 FCC readings (out/readings/*-fcc-r*-rerun.json): "
        "superseded by the laptop's FCC at b5626ecb (addendum 6); on disk, unscored",
    }
    (S.OUT / "scores.json").write_text(json.dumps(doc, indent=1, default=str))
    L = ["\n## Addendum 6's additions\n"]
    L.append(
        f"- **V3 with clauses (ii) and (iii) accepted in place of all_monotone** in the count rule as written (four usable "
        f"readings within tol): **{alt_verdict}**; counted {len(alt)} of 90; violations {alt_viol}; counted only because of "
        f"(ii)/(iii): {doc['addendum6']['V3_with_clauses_ii_iii_accepted']['added_by_ii_iii']}."
    )
    L.append(
        "- As written (all_monotone required), z00 and z12 cannot count: their readings are usable only by (ii) or (iii).\n"
    )
    L.append(
        "**The count rule for each of the 14** (four readings within 1/100 of the minimum abserr):\n"
    )
    L.append(
        "| sample | readings | spread / tol | agree within tol | clauses |\n|---|---|---|---|---|"
    )
    for x in per14:
        L.append(
            f"| {x['sid']} | {x['n_readings']} | {x['spread_over_tol']:.3g} | {'yes' if x['agree_within_tol'] else 'no'} | {'; '.join(x['clauses'])} |"
        )
    L.append(f"\nSuperseded and unscored: {doc['addendum6']['superseded']}.")
    with open(S.OUT / "report.md", "a") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
