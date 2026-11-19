"""levin-phase-floor prompt 07b section 6: write state/predictions.md, after Gate U0 and before the census is launched.
New in 07b, after 07a's write_predictions_07a.py (a record, not edited), which copied by line range and once copied a
block truncated. Here each block of the prompt is copied **by text**: it starts at a fixed opening string and ends at a
fixed closing string of the prompt file in the tree h, each of which must occur exactly once; the block is the text
between them, inclusive, unchanged. After writing, every block is checked to be a substring of the prompt and of the
written file, and its first and last lines are printed. The rest is this agent's, marked as such. Refuses to overwrite;
the file is made read-only.

Usage: python -B write_predictions_07b.py OUT_MD
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_TREE, header, sha256  # noqa: E402

PROMPT = H_TREE / "prompts" / "levin-phase-floor" / "07b-levin-leaf-beside-a-window.md"
BLOCKS = [  # (name, title, opening string, closing string)
    ("s6", "Prompt §6, the predictions and the descriptive items",
     "**Predictions, fixed before the run** (README §5 rule 3)",
     "changes in each direction; the abserr ratio against 07a's; regions and cost against 07a's."),
    ("s95", "Prompt §9.5, the additions to §5's and §6's predictions (U8 among them)",
     "### 9.5 Additions to §5's and §6's predictions",
     "replay record in `float.hex`, and R2-22 equal to 07a's census too."),
    ("s97", "Prompt §9.7, the stops added (for reference; not predictions)",
     "### 9.7 Stops added (into §8)",
     "- In Part C: the census's projection beyond 48 hours; U8 failing, before anything else is scored."),
]

OWN = r"""
## 4. How each rule is read (this agent's, fixed here before the census)

- **U1.** A call is untouched iff `num_edge_rule_refusals + num_range_rule_refusals` is 0 (§9.5), the two counts as the
  call's result returns them (the census records them on every call; None would be a fault). Each untouched call is
  compared with 07a's census call (the same unit `h-<rtol>-<shard>-<serial>`, the same kind and index) in value,
  abserr, `converged`, `num_regions` and `max_depth`, floats by `float.hex`. **Holds** if every untouched call is equal
  in all five at both rtols, and every unit all of whose Levin calls (group and analytic) are untouched equals 07a's in
  `result.total` and `result.total_abserr` by hex. Beside it, descriptive: each untouched call's whole record diffed
  against 07a's by category (`call`, `location`, `harness`, `oracle`), compare_07b.py mode `calls`. A unit not `done`
  in either census is listed and not compared. **Stop** (§8): any untouched call differing in the five, before anything
  else is scored.
- **U8.** The calls are the 187 Part A replayed at both rtols, `var/levin-phase-floor-07b/measurement/state/
  census_wanted_07b.json` less `i3-34599-07` (replayed at 07's module): each id's census call is that file's (unit,
  kind, index). (h)'s replay record is `out/ext/replays/h/<id>.json`'s `replay` (and `h_summary_all.json` /
  `h_summary_i3.json`, checked to agree with it). **Holds** if every one equals in value and abserr (hex), `converged`,
  `num_regions` and `max_depth`. **Stop** (§9.7): any failure, before anything else is scored.
- **U2, U3, U5** as 07a's W2, W3, W5 were scored (07a's census `log.md`), U5 on gated solves only (07a's W5 ruling,
  `var/levin-phase-floor-07a/census/state/ruling_w5.md`), the forced solves' count descriptive.
- **U4.** The referenced calls and their references are Part A's (`var/levin-phase-floor-07b/measurement/out/tables/
  scores_final.json` and `tables_07b.json`: V4's 25, R1, R2, R3, item 3), 07's V4 readings for 07a's census's other
  eight V4 calls (`docs/levin-phase-floor/07/census/out/v4/v4_scores.json`), and 07a's census references for `1-34599`
  and `3-678` (`var/levin-phase-floor-07a/census/out/refs/scores.json`), each scored from this census against its
  reference by 07's rule as those records apply it. **Holds** if every one is honest, or unconverged. §9.5's prediction
  (exactly v04 and R2-28 at both rtols unconverged, R2-28 at 1.107 of its abserr) is scored beside it.
- **U6.** 06b's 24 calls (`docs/levin-phase-floor/06b/census/out/w2_reference.json`; unit `b-1e-7-X-NNNNN` is this
  census's `h-1e-7-X-NNNNN`, the same call index) against $S_{10}$ by 06b's rule, $|v - S_{10}|/{\rm abserr} \le 1$, or
  unconverged.
- **U7.** $r_g$ over all 2,484 pairs, by 07's cross-check as 07a's `xcheck_07a.py` computed it; every pair $r_g \le 1$
  or a member unconverged.
- **No new references** are taken; U4 and U6 are scored as written and attributed to nothing (the dispatch).

## 5. Gate U0 and the projection, as they stood when this file was written

- U0 (1): `out/u0/u0a_ast.txt` and `out/u0/u0a_imports.txt` (both HOLD).
- U0 (2): `out/k0/compare_u0b.json` (100 of 100 K0 units equal 07a's census at the parent).
- U0 (3): `out/u0c3/compare_instr.json` (`h-1e-8-3-14501`, where the fix acts, and `h-1e-7-1-20582`, where it does not:
  IDENTICAL on and off).
- U0 (4): `out/u0c4/compare_det.json` (`h-1e-8-3-14501` twice: IDENTICAL).
- The projection: `out/census_projection.json`.
"""


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    text = PROMPT.read_text()
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b section 6: the census's predictions, fixed before the census",
                 [PROMPT])
    blocks = []
    for name, title, a, b in BLOCKS:
        if text.count(a) != 1 or text.count(b) != 1:
            raise SystemExit(f"{name}: the opening or closing string does not occur exactly once")
        i = text.index(a)
        j = text.index(b) + len(b)
        if j <= i:
            raise SystemExit(f"{name}: the closing string comes before the opening one")
        blk = text[i:j]
        l0 = text.count("\n", 0, i) + 1
        l1 = text.count("\n", 0, j - 1) + 1
        blocks.append((name, title, blk, l0, l1))
    parts = [
        "# levin-phase-floor 07b §6 — the census's predictions, fixed before the census",
        "",
        "Provenance (the writer's header):",
        "",
        "```",
        *[l[2:] for l in hdr],
        "```",
        "",
        "Written after Gate U0 (all four clauses) and before the census is launched. Not to be rewritten: a correction is a "
        "new file and a deviation.",
        "",
        f"Source: `{PROMPT}` (file SHA-256 {sha256(PROMPT)}). Parts 1-3 are copied **by text** (each from a fixed opening "
        "string to a fixed closing string, both unique in the file; the line numbers are given for reference only) and "
        "checked to be substrings of the prompt. Parts 4 and 5 are this agent's.",
        "",
    ]
    for k, (name, title, blk, l0, l1) in enumerate(blocks, 1):
        parts += [f"## {k}. {title} (verbatim; prompt lines {l0}-{l1})", "", blk, ""]
    parts.append(OWN.strip("\n"))
    doc = "\n".join(parts) + "\n"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(doc)
    out_p.chmod(0o444)
    back = out_p.read_text()
    ok = True
    for name, title, blk, l0, l1 in blocks:
        in_prompt = blk in text
        in_file = blk in back
        ok &= in_prompt and in_file
        lines = blk.splitlines()
        print(f"{name}: lines {l0}-{l1}, {len(lines)} lines, {len(blk)} chars; substring of the prompt {in_prompt}, of the "
              f"file {in_file}")
        print(f"  first: {lines[0]}")
        print(f"  last:  {lines[-1]}")
    for must in ("U1 — nothing else moves", "U2 — no Levin leaf", "U3 — every phase", "U4 — honest where it is referenced",
                 "U5 — the gate is consistent", "U6 — 06b's 24 are honest", "U7 — the pairs", "U8 — the code is (h)",
                 "**U1 counts**", "**U4 is predicted to hold,**", "**U6 is predicted to hold.**", "**U7 is predicted to hold:**",
                 "**Descriptive, not scored:**", "**Descriptive, added:**"):
        f = must in back
        ok &= f
        print(f"  contains {must!r}: {f}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}; all checks {'HOLD' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
