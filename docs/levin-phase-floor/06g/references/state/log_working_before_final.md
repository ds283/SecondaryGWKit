# levin-phase-floor 06g — §5, the references (V3, V4, the correction −ΣE): the references agent's record

**Subject:** prompt 06g §5's references, with addenda 1 (V1's 12 group calls) and 2 (V8's 12 calls),
on the Mac Studio (mps026298), 2026-11-01. **Commit:** none; records committed by the orchestrator
from the laptop. **Result:** *in progress* (stage 2).

## What shipped

*In progress.*

## Predictions and verdicts

*Copied in at 2026-11-01T17:51:20Z, before any number of the references was computed; the record is
`state/predictions_20261101T175120Z.md` (read-only, SHA-256 `825b6f04…`), with the seed, the count
rule as applied, the agreement rule for 06i's 13 and how each number is formed. Addenda 1 and 2 have
their own rule files, `state/v1_rule_20261101T175144Z.md` (`d03fde45…`) and
`state/v8_rule_20261101T175144Z.md` (`04bacdc5…`).*

## The user's ruling on part 2 (addendum 3, 2026-11-01)

Recorded before any part-2 reading in `state/ruling_addendum3_20261101T201347Z.md` (`c036412a…`,
read-only), which also fixes the backstop and the second scoring. The user's words, in part:
- "Quibbling over ~8 hours of compute that only has to be run once strikes me as a false economy."
- "A runtime that exceeds 30 minutes means that we should stop and check, not guillotine it
  immediately. If it is projected to finish in the next 5 minutes, we should finish it. If it's not
  projected to finish for a day, then we would kill it. In between these extremes is the judgement
  call."
- On scoring: "We are iteratively improving on our algorithm, and we don't know what to expect."

V3 and V4 are scored both by their rules as written and as the ruling scores them, each labelled.

## Prompt 06g §5, verbatim


- **A reference counts** when its four readings agree to within 1/100 of
  $\min({\rm abserr}_{06h}, {\rm abserr}_g)$. Those that do not are reported with their scatter and
  dropped. A reading over 30 minutes is stopped, reported, and the sample's reference dropped.

- **V3 — honest where referenced.** **Rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every
  counted reference, Z3's and the random sample's. $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ is
  reported beside it. **Holds** if no counted reference violates it; fewer than 45 of the 90 counted is
  **undecided**.
- **V4 — the trusted leaf's value.** **Rule:** on Z3's samples 2, 8, 9 and 14,
  $|v_g - {\rm ref}|/{\rm abserr}_g \le 0.01$ (06e's breakdown, with the quarters: 0.0012, 0.00046,
  0.0015). **Holds** if true on all four.

**Descriptive, not scored:**
- **The correction $-\sum E$:** on every counted reference with a Clenshaw–Curtis leaf,
  $|v_g - \sum E - {\rm ref}|$ against $|v_g - {\rm ref}|$ and against ${\rm abserr}_g$.

Prompt 06g §7, the references' stop: fewer than half of the 77 new samples count, or a reading
disagrees with 06i's on one of its 13.

Verdicts: *not yet scored.*

## Scored after stage 2 of the rerun batch (2026-11-02T20:44:20Z)

*The user's ruling (relayed 2026-11-02): score now, while stage 3 runs; "we can use the result to
inform our view on whether we should let Stage 3 run to completion. We should then check whether
anything in stage 3 would change those scorings, after it lands." Files:
`out/scores_after_stage2/report_20261102T204420Z.md` and `scores_20261102T204420Z.json`, by
`scripts/score_refs.py` (`out/cmd/71`). At this time r24, r26 and r53 are FCC-only and r47 has three
readings; stage 3 will be rechecked beside these scores, not in their place. The 14
stationary-point samples are scored on their rerun-batch readings only; their old readings are
unscored: "made by the tool at `1c5336e`, which inverted through a non-monotone segment; unsound by
the tool's own rule". Every scored Route B reading's usable clause is in the report.*

- **V3 as written: holds** (66 of 90 counted, no violation). **V3 as the ruling scores it: fails**
  (90 of 90 scored, 80 two implementations and 10 FCC only; one violation, z12, a two-implementation
  reference, |v_g − ref|/e_g = 1,681; 06h 1,678). On two-implementation references alone: fails (z12).
- **V4: holds** both ways (z02 0.00121, z08 2.1e-7, z09 1.4e-4, z14 0.00121).
- V1's 12 and the V8 sample, the correction and z00 across hosts: in the report.

## Rescored per addendum 6 (2026-11-03T02:08:15Z), an addition beside the stage-2 scores

*Addendum 6 (`fb1c273c…`): clause (iii) of the usable rule (FCC at `b5626ecb`, every non-monotone
segment integrated directly); the 14 stationary-point samples' stage-2 FCC readings superseded by the
laptop's 28 (`out/readings_fcc_b5626ec_laptop/`), their stage-2 QAWO readings standing; every other
sample on exactly its stage-2 readings (the stage-3 readings landed since, r24 and r53 qawo r1, are
not used here). Files: `out/scores_addendum6/report_20261103T020815Z.md`, `scores_20261103T020815Z.json`,
by `scripts/score_refs_a6.py` (`out/cmd/80`); the open check `out/compare_z00_z12.json` (`out/cmd/81`).*

- **V1's 12:** each reference's four readings agree within 3.1e-6 to 3.5e-4 of the tolerance; R24 is
  nearer on all 12, and R24's declared error covers its distance on all 12 (worst w07 0.942, w01
  0.817); R23 is 4.5e3 to 6.0e5 of its own abserr from every reading.
- **V3 as written: holds** (66 of 90; z00 and z12 cannot count, being usable only by (ii)/(iii)).
  **With (ii) and (iii) accepted in place of all_monotone: fails** (68 counted; z00 and z12 added;
  z12 violates, 1,681). **As ruled: fails** (z12), as at stage 2.
- **The count rule for each of the 14:** all agree within the tolerance (spread 3.1e-6 to 3.5e-4).
- **z00 and z12's captured integrands are equal** by the 06i-13 criterion and the integrand at 65
  points and the default edges; the calls differ only in their rtol (and so their results).

## The stage-3 recheck (2026-11-03T06:51:58Z), an addition beside the stage-2 and addendum-6 scores

*The rerun batch `levin-phase-floor-06g-references-rerun-20261102T185324` ended done, exit 0, 67 of 67
(64 done, 3 stopped). Files: `out/scores_stage3/report_20261103T065158Z.md`, `scores_…json`,
`changes.txt` (`scripts/score_refs_s3.py`, `out/cmd/91`); the projections `out/stage3_projection.json`
(`scripts/stage3_projection.py`, `out/cmd/90`).*

- **Landed:** r24 qawo r1 (6,098 s, `0x1.3a013cf6f38aep-92`) and r53 qawo r1 (7,772 s,
  `0x1.b2dd0734ed889p-93`), clause (i). r24 and r53 now have three readings and two implementations;
  spread/tol 5.9e-5 and 1.8e-3.
- **Stopped by the batch's backstop** (each at its first chunk back, "projected total ≥ 86,400 s"):
  r47 qawo r2 at 3,905 s (projected 453,016 s), r53 qawo r2 at 2,103 s (290,165 s), r26 qawo r1 at
  16,522 s (2,296,531 s). r47 stays at three readings, r26 FCC-only, r53 without r2.
- **Changes against the addendum-6 scores:** r24 and r53 move from FCC-only to two implementations
  (82 two-implementation references, 8 FCC-only); r53's spread/tol 1.68e-3 → 1.80e-3. Their
  references (FCC r2) and ratios are unchanged (r24 0.307, 06h 0.221; r53 0.0219, 06h 0.0878).
  **No verdict changes:** V3 as written holds (66), with (ii)/(iii) accepted fails (68; z12), as
  ruled fails (z12; two-implementation only also fails, z12); V4 holds both ways. Neither r24 nor r53
  is in the V8 table or V4; V1's table is unchanged.

## Deviations from the prompt

*In progress; the list so far is in `state/plan_stage2.md`.*

## Verification performed

*Part 1 (captures, checks, 06i's 13): `out/part1_results.md`. Part 2 not launched (the cost projection, `out/cost_projection_part2.txt`).*

## Observations not acted on

1. **The FCC readings on four of 06i's Route A samples (9, 22, 25, 27) are offset from 06i's by a constant at both resolutions** (2.8e-6 to 4.0e-5 of m; QAWO bit-identical; the inputs' integrand at 65 points bit-identical). Within the agreement rule; the cause was not investigated.
2. **Route A's QAWO cost scales with the phase span** (panels at most 2^12 / 2^9 rad): 17 of part 2's 45 Route A calls are projected to exceed 30 minutes on QAWO res 2 (`out/cost_projection_part2.txt`).
3. **The rerun batch's day-or-more projection was wrong, and stopped three readings that would have
   finished in under a day** (`out/stage3_projection.json`). `run_rerun.py` projected
   `elapsed × total / done`; at done = 1 that ignores the five other chunks in flight on the six
   workers, includes setup, and assumes uniform chunks, while QAWO's chunk cost follows the chunk's
   phase span, which is very uneven. A six-worker schedule with chunk cost ∝ phase span, calibrated on
   the first chunk back, reproduces the finished r24 and r53 qawo r1 monitor counts at every 5-minute
   check (to within 2 chunks) and gives: r47 qawo r2 10,826 s (3.0 h), r53 qawo r2 31,889 s (8.9 h),
   r26 qawo r1 49,162 s (13.7 h), each under a day; the uniform six-in-flight correction gives 70,389,
   44,943 and 378,324 s. Its cost per radian per worker, 0.342 µs/rad for r26 r1, equals r24's and
   r53's finished r1 (0.343, 0.344).

*More in progress.*

## State handed on

*In progress.*
