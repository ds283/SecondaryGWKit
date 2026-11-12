"""levin-phase-floor 06g, the references: the stage-3 recheck (the user's ruling: "check whether
anything in stage 3 would change those scorings, after it lands"), an **addition beside** the
stage-2 and addendum-6 scores (neither rewritten). It is score_refs_a6.py (addendum 6's readings and
usable rule), changed only in r24 and r53, which gain their stage-3 QAWO r1 readings
(out/readings/<sid>-qawo-r1-rerun.json, tool ff3e8394, all monotone: clause (i)). What did not land:
r47 qawo r2, r53 qawo r2 and r26 qawo r1 were stopped by the batch's backstop; r47 stays at three
readings, r26 FCC-only, r53 without r2. Then every verdict and every row is compared with the
addendum-6 scores (out/scores_addendum6/scores_20261103T020815Z.json), and each difference listed.
Writes out/scores_stage3/ (scores.json, report.md, changes.txt). One process; reads only.
Usage: PYTHONPATH=<tree g>:<this directory> python -B score_refs_s3.py"""

import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
import score_refs_a6 as A6  # noqa: E402  (patches score_refs as addendum 6 has it)
import score_refs as S  # noqa: E402

_ro_a6 = S.readings_of
STAGE3 = ("r24", "r53")


def readings_of(e):
    out, src = _ro_a6(e)
    if e.get("sid") in STAGE3:
        d = S.rd(RC.ROOT / "out" / "readings" / f"{e['sid']}-qawo-r1-rerun.json")
        if d:
            out[("qawo", 1)] = d
            src += " + stage-3 QAWO r1 (ff3e8394)"
    return out, src


S.readings_of = readings_of
S.OUT = RC.ROOT / "out" / "scores_stage3"


def main():
    A6.main()
    rep = (
        (S.OUT / "report.md")
        .read_text()
        .replace(
            "# The references, rescored per addendum 6 (an addition beside the stage-2 scores)",
            "# The references, the stage-3 recheck (an addition beside the stage-2 and addendum-6 scores)\n\n"
            f"# recheck script score_refs_s3.py sha256 {RC.sha256(Path(__file__).resolve())}",
            1,
        )
    )
    (S.OUT / "report.md").write_text(rep)
    new = json.loads((S.OUT / "scores.json").read_text())
    old = json.loads(
        (
            RC.ROOT / "out" / "scores_addendum6" / "scores_20261103T020815Z.json"
        ).read_text()
    )
    ch = []

    def cmp(path, a, b):
        if a != b:
            ch.append(f"{path}: {a} -> {b}")

    for k in ("as_written", "as_ruled"):
        for f in ("verdict", "n_counted", "n_scored", "violations"):
            if f in old["V3"][k]:
                cmp(f"V3 {k} {f}", old["V3"][k][f], new["V3"][k][f])
    cmp(
        "V3 as_ruled two-implementations",
        old["V3"]["as_ruled"]["two_implementations"],
        new["V3"]["as_ruled"]["two_implementations"],
    )
    for f in (
        "n_fcc_only",
        "fcc_only_violations",
        "fcc_r1_r2_over_tol",
        "spread_over_tol",
    ):
        cmp(f"V3 as_ruled {f}", old["V3"]["as_ruled"][f], new["V3"]["as_ruled"][f])
    cmp(
        "V3 with (ii)/(iii)",
        old["addendum6"]["V3_with_clauses_ii_iii_accepted"],
        new["addendum6"]["V3_with_clauses_ii_iii_accepted"],
    )
    cmp("V4 as_written", old["V4"]["as_written"], new["V4"]["as_written"])
    cmp("V4 as_ruled", old["V4"]["as_ruled"], new["V4"]["as_ruled"])
    orow = {r["sid"]: r for r in old["rows"]}
    for r in new["rows"]:
        o = orow[r["sid"]]
        for f in (
            "class",
            "methods",
            "n_readings",
            "spread_over_tol",
            "counts_as_written",
            "r_g",
            "r_06h",
            "violates",
            "ref_hex",
        ):
            cmp(f"row {r['sid']} {f}", o.get(f), r.get(f))
    for o, n in zip(old["V1"], new["V1"]):
        for f in ("r23", "r24", "nearer", "nearer_covered", "spread_over_tol"):
            cmp(f"V1 {n['wid']} {f}", o.get(f), n.get(f))
    txt = "Changes against the addendum-6 scores:\n" + ("\n".join(ch) if ch else "none")
    (S.OUT / "changes.txt").write_text(txt + "\n")
    with open(S.OUT / "report.md", "a") as f:
        f.write(
            "\n## Changes against the addendum-6 scores\n\n"
            + "\n".join(f"- {c}" for c in ch)
            + "\n"
        )
    print(txt)


if __name__ == "__main__":
    main()
