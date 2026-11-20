"""levin-phase-floor prompt 07c section 6: write state/predictions.md, after Gate Y0 and before the census is launched.
New in 07c, after 07b's write_predictions_07b.py (a record, not edited), whose method it keeps: each block of the prompt
is copied **by text**, from a fixed opening string to a fixed closing string of the prompt file in the tree h, each of
which must occur exactly once; the block is the text between them, inclusive, unchanged. After writing, every block is
checked to be a substring of the prompt and of the written file, and its first and last lines are printed. Part 5 is
this agent's reading of each rule; part 6 states Gate Y0 and the projection as they stood, read from their output files.
Refuses to overwrite; the file is made read-only.

Usage: python -B write_predictions_07c.py OUT_MD
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_TREE, WIDER, header, sha256  # noqa: E402

PROMPT = H_TREE / "prompts" / "levin-phase-floor" / "07c-chase-below-a-refused-leaf.md"
BLOCKS = [  # (name, title, opening string, closing string)
    ("s6", "Prompt §6, the predictions and the descriptive items",
     "**Predictions, fixed before the run** (README §5 rule 3)",
     "(**the sum**, not each separately); regions and cost against 07b's."),
    ("s95", "Prompt §9.5, the additions to §5's and §6's predictions (Y9 among them)",
     "### 9.5 Additions to §5's and §6's predictions",
     "- `c-1e-8-1-2229-4` (untouched), equal to (S)'s record and to 07b's census."),
    ("s97", "Prompt §9.7, the stops added (for reference; not predictions)",
     "### 9.7 Stops added (into §8)",
     "- **In Part C:** Y8 failing, before anything else is scored."),
    ("s8", "Prompt §8, Part C's stops (for reference; not predictions)",
     "- In Part C: S0′ failing;",
     "before anything else is scored; a lock held."),
]

OWN = r"""
## 5. How each rule is read (this agent's, fixed here before the census)

Throughout: a census call is named by its unit `h-<rtol>-<shard>-<serial>`, its kind (group, in the unit's `calls`; or
analytic, in `other_calls`) and its index; 07b's census call is the one of the same name in
`levin-phase-floor-07b-census-20261117T010200` (checkpoint `396784bc…`). Floats are compared by `float.hex`; ratios are
computed in exact `Fraction`s from the hex values and the references' decimal strings. "Unconverged" is the call's
`converged` false. A unit not `done` in either census is listed and not compared.

- **Y1.** A call is untouched iff `num_chase_stops` is 0 (§9.5), as the call's result returns it (the census records the
  three counts on every call; a missing count is a fault, reported). Each untouched call is compared with 07b's call in
  value, abserr, `converged`, `num_regions` and `max_depth`. A "row all of whose calls are such" is read as a unit (one
  row at one rtol) every one of whose Levin calls, group and analytic, is untouched; it is compared with 07b's unit in
  `result.total` and `result.total_abserr`. **Holds** if every untouched call is equal in all five at both rtols and
  every such unit is equal in both totals. Beside it, descriptive: each untouched call's whole record diffed against 07b's
  by category (compare_07c.py mode `calls`). **Stop** (§8): any untouched call differing in the five, before anything
  else is scored.
- **Y8.** The calls are every record Part A wrote of (S)'s replays, `var/levin-phase-floor-07c/measurement/out/replays/S/
  <id>.json` (454), less `r31-3-20363-1` (R31's record, a rehearsal row and not a census call): 453 ids, 448 distinct
  census calls (five census calls are named by two ids each; each id is compared). Each id's census call is read from
  Part A's mappings (`out/census_refs_07c.json` `calls`; `state/census_wanted_07c.json`, `census_wanted_07c_chase.json`,
  `census_wanted_07c_r5.json` `wanted`). (S)'s replay record is the file's `replay`, checked to equal
  `S_summary_vplain.json`'s row. **Holds** if all 453 equal the census call in value and abserr (hex), `converged`,
  `num_regions` and `max_depth`. **Stop** (§9.7): any failure, before anything else is scored (scripts/score_y8_07c.py).
- **Y2, Y3, Y5** as 07b's census scored U2, U3 and U5 (07a's W2, W3 and W5; score_07b.py), Y5 on gated solves only
  (07a's W5 ruling), the forced solves' count descriptive.
- **Y4.** Two sets, each scored from this census against its reference by 07's rule: a call is scored if its reference's
  stated error is at most min(abserr)/100 over this census's and 07b's census's abserr for the call (07c's form of 07b's
  U4 rule, min(abserr_07b, abserr_07a)/100; for R5, 07a's census abserr as well, as Part A's R5 rule takes it), otherwise
  reported and not scored; a scored call is honest iff |v − ref| ≤ abserr; a call unconverged here counts as
  "unconverged under §9's rule".
  - (a) **07b's U4's 190 scorings**: the same calls, on the same references, by 07b's census scripts (score_v4_07b.py:
    V4's 25 and z00 and Z3's seven, with z12, on 07's and 06g's readings; score_refs_07b.py: 07b Part A's R1 60, R2 60,
    R3 (W7 6, aux 12, own 6, pairs 4) and item 3, on 07b Part A's references, and 07a's rows 1-34599 and 3-678, 8 calls,
    on 07a's census references), copied with this census in place of 07b's and 07b's census in place of 07a's as the
    other code.
  - (b) **07c Part A's R4, R5 and R2-28**: every entry of Part A's `out/tables/scores_07c.json` `scored` whose
    `ref_key` is `ref-R4-*` (sets `chase` and `chase~o`, 60) or `ref-R5-*` (sets `r5` and `r5~o`, 60), and R2-28's census
    call at both rtols (`c-1e-7-0-13757-5`, `c-1e-8-0-13757-5`, `ref-R2-28`), each id's census call by Part A's mappings
    (as Y8), on Part A's reference and stated error (`ref`, `spread` there).
  - **Holds** if every scored call in (a) and (b) is honest or unconverged, except R2-28: its census call at both rtols in
    (b), and its two scorings in (a) (07b Part A's `R2-28` and `R2-28~o`, the same integrand), which §9.5 predicts
    converged and dishonest at 1.143 of the abserr; each is reported with its ratio against
    `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]` and does not fail Y4. Any other scored call converged and
    dishonest fails Y4. A call named in both sets is scored once under each. Nothing is attributed (the dispatch).
- **Y6.** 06b's 24 calls (`docs/levin-phase-floor/06b/census/out/w2_reference.json`; unit `b-1e-7-X-N` is this census's
  `h-1e-7-X-N`, the group call of the same index) against $S_{10}$ by 06b's rule, as 07b's score_refs_07b.py scored U6:
  scored if $|S_9 - S_{10}| \le$ abserr/100; honest iff $|v - S_{10}| \le$ abserr; **holds** if every one is honest or
  unconverged. Nothing is attributed.
- **Y7.** $r_g$ over 07's 2,484 pairs (07's census marks), by 07b's xcheck_07b.py with this census and 07b's census as
  the before; **holds** if every pair has $r_g \le 1$ or a member unconverged.
- **Y9.** Two clauses, both over this census's Levin calls at both rtols.
  - (i) **A chase call** is a call that is unconverged, returns at least one leaf at depth 20, and every one of whose
    depth-20 leaves lies inside the span of a leaf a rate rule refused in the same call at a smaller depth (07b's census
    rule, score_07b.py's `r07b_unit`: the rate-rule instrument's refused evaluations, `lo ≤ start`, `end ≤ hi`, refused
    depth below the leaf's). A call unconverged with no leaf at depth 20 (a budget miss) is not a chase call: "every
    depth-20 leaf" is not read as true of a call with none. Clause (i) is true iff there is no chase call.
  - (ii) **07b's 198 newly unconverged calls** are the list of 07b's census `out/scores/summary.json`
    `newly_unconverged`. Clause (ii) is true iff, of the 198, exactly the five named in §9.5 are unconverged here, and
    each of the five has `num_chase_stops` 0 and equals 07b's call in Y1's five outputs.
  - **Holds** iff (i) and (ii) are both true. A call newly unconverged against 07b is not a stop and changes no verdict
    (Y4, Y6 and Y7 count an unconverged call as passing; Y9 is about depth-20 leaves).
- **The projection** (§9.4): within 48 hours at its highest margin; stated also times 07b's shortfall (1.197).
- **Descriptive** (§6 and §9.5), not scored:
  - where the fix acts (`num_chase_stops` > 0) by kind, 07b's class of the call (the rate rules' refusals), window,
    the 720, band, $k$ and rtol; the stops, stopped leaves and counted bisections by the same;
  - by the stop instrument (`chase_07c`, marked "prompt 07c"): each stopped leaf's route (Levin or the Clenshaw–Curtis
    pair, `type` `direct`), its trust before the stop, depth, declaration (declared resolution and e'), $r$, and its share
    of its call's abserr (its e' as returned over the call's abserr); the instrument's counts against the three result
    counts on every call;
  - `converged` changes against 07b's and against 07a's (07b's census `out/stop3/compare_census_calls.calls.jsonl`
    carries 07a's), in each direction; **every call newly unconverged against 07b listed with the stopped leaf carrying
    the most of its abserr** (route, depth, $r$, share), or "no stopped leaf" if the stop did not act in it;
  - the calls still unconverged with every depth-20 leaf below a refused leaf (Y9 (i)'s list, expected empty);
  - rows (units) with a leaf at depth 20, each rtol, against 07b's;
  - the abserr ratio against 07b's, untouched and acted calls apart;
  - values moved past **the sum**: $|v_{07c} - v_{07b}| > {\rm abserr}_{07c} + {\rm abserr}_{07b}$, in exact
    arithmetic; never each separately;
  - regions and cost against 07b's (census wall, unit walls, unit time with post sections, regions by rtol and kind).
"""


def y0_status():
    def load(p):
        p = WIDER / p
        return json.loads(p.read_text()) if p.exists() else None
    lines = ["## 6. Gate Y0 and the projection, as they stood when this file was written", ""]
    a = (WIDER / "out" / "y0" / "y0a_ast.txt")
    b = (WIDER / "out" / "y0" / "y0a_imports.txt")
    for p in (a, b):
        v = [l for l in p.read_text().splitlines() if l.startswith("# verdict")] if p.exists() else ["(missing)"]
        lines.append(f"- Y0 (1): `{p.relative_to(WIDER)}`: {v[0][2:] if v else '(no verdict line)'}")
    w = load("out/k0/compare_y0b.json")
    lines.append(f"- Y0 (2): `out/k0/compare_y0b.json`: {w and w['n_units_equal']} of {w and w['n_units']} K0 units equal "
                 f"07b's census at the parent; totals {w and w['totals']}; worker evidence all hold "
                 f"{w and w['worker_evidence_all_hold']}; new keys in 07b's records {w and w['new_keys_in_07_records']}")
    i = load("out/y0c3/compare_instr.json")
    lines.append(f"- Y0 (3): `out/y0c3/compare_instr.json`: "
                 + ("; ".join(f"{u['unit']} {'IDENTICAL' if u['ok'] else 'DIFFERS'} (stops {u.get('stops')})" for u in i["units"])
                    if i else "(missing)"))
    d = load("out/y0c4/compare_det.json")
    lines.append(f"- Y0 (4): `out/y0c4/compare_det.json`: "
                 + ("; ".join(f"{u['unit']} {'IDENTICAL' if u['ok'] else 'DIFFERS'}" for u in d["units"]) if d else "(missing)"))
    pj = load("out/census_projection.json")
    if pj:
        lines.append(f"- The projection (`out/census_projection.json`): central {pj['projection_central_s']:.0f} s "
                     f"({pj['projection_central_s'] / 3600:.2f} h); times 07b's shortfall {pj['shortfall_07b']:.3f}: "
                     f"{pj['projection_central_x_07b_shortfall_s']:.0f} s; highest margin "
                     f"{pj['projection_highest_with_overbudget_s']:.0f} s, times the shortfall "
                     f"{pj['projection_highest_x_07b_shortfall_s']:.0f} s; within 48 h: {pj['within_48h']} "
                     f"(times the shortfall: {pj['within_48h_x_07b_shortfall']})")
    return "\n".join(lines)


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    text = PROMPT.read_text()
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c section 6: the census's predictions, fixed before the census",
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
        "# levin-phase-floor 07c §6 — the census's predictions, fixed before the census",
        "",
        "Provenance (the writer's header):",
        "",
        "```",
        *[l[2:] for l in hdr],
        "```",
        "",
        "Written after Gate Y0 (all its clauses) and before the census is launched. Not to be rewritten: a correction is a "
        "new file and a deviation.",
        "",
        f"Source: `{PROMPT}` (file SHA-256 {sha256(PROMPT)}). Parts 1-4 are copied **by text** (each from a fixed opening "
        "string to a fixed closing string, both unique in the file; the line numbers are given for reference only) and "
        "checked to be substrings of the prompt. Parts 5 and 6 are this agent's.",
        "",
    ]
    for k, (name, title, blk, l0, l1) in enumerate(blocks, 1):
        parts += [f"## {k}. {title} (verbatim; prompt lines {l0}-{l1})", "", blk, ""]
    parts.append(OWN.strip("\n"))
    parts += ["", y0_status()]
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
    for must in ("Y1 — nothing else moves", "Y2 — no Levin leaf", "Y3 — every phase", "Y4 — honest where it is referenced",
                 "Y5 — the gate is consistent", "Y6 — 06b's 24 are honest", "Y7 — the pairs",
                 "Y8 — the code is the chosen variant", "**Y1 counts**", "**Y4, on the user's ruling.**",
                 "**Y6 and Y7 are predicted to hold.**", "**Y8 — the code is (S).**", "**Y9 — the chase ends.**",
                 "**Descriptive, not scored:**", "**Descriptive, added:**", "In Part C:** Y8 failing"):
        f = must in back
        ok &= f
        print(f"  contains {must!r}: {f}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}; all checks {'HOLD' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
