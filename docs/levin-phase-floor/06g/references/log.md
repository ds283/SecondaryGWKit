# levin-phase-floor 06g — Part B §5, the references (V3, V4, the correction −ΣE; addenda 1–6): the references agent's record

**Subject:** prompt 06g §5's references, with V3 and V4 scored from them and the descriptive
correction −ΣE, and the user's addenda 1–6 (V1's 12 group calls; V8's 12 calls; the cost gate and
the 30-minute check; the non-monotone hold and the rerun batch; the new tool at `ff3e8394`; FCC at
`b5626ecb`). On the Mac Studio (mps026298), 2026-11-01 to 2026-11-03.
- **Trees:** the code worktree `SecondaryGWKit-lpf06g-1c5336e` (Part B's dispatch SHA, code `3ae6f3a`):
  every capture of V3's and V8's calls and V1's at g, and parts 1 and 2's readings; the parent worktree
  `SecondaryGWKit-lpf06g-4f4f9ed`: V1's captures against R23; the tool worktree
  `SecondaryGWKit-lpf06g-ff3e839` (`ff3e8394`): the rerun batch's readings. The laptop's FCC batch ran
  at `b5626ecb` on the laptop. The main checkout stayed at `28ebaf0`, clean.
- **Commit:** none; records committed by the orchestrator from the laptop.

**Result.**
- **Every capture reproduces its record bit for bit:** 102 of 102 census calls (value, abserr, label,
  converged, num_regions, max_depth, region arrays, harness hashes) and 24 of 24 V1 captures (the
  stored group, in its (1 + z_response) form, and the row total).
- **06i's 13:** groups equal, the tool agrees with 06i's readings on all 13.
- **V3 — as written: holds** (66 of 90 count, none violates). **With clauses (ii) and (iii) accepted in
  place of all_monotone: fails** (68 count; z12). **As the ruling scores it: fails** (90 of 90 scored, 82
  with two implementations; one violation, z12 at 1,681× its abserr, 06h 1,678×).
- **V4: holds** both ways (z02 0.00121, z08 2.1e-7, z09 1.4e-4, z14 0.00121).
- **V1's 12:** R24 nearer the reference on all 12 and within its abserr on all 12 (worst 0.942); R23 4.5e3
  to 6.0e5 of its abserr away.
- **V8's 12:** g within its abserr on 11 (v03 1.20×); 06h on 9 (v03 2.31×, v08 4.29×, v11 1.07×).
- **The correction −ΣE:** nearer the reference on 12 of 25 references with a CC leaf; mostly no change.
- Three QAWO readings were stopped by a faulty projection (r47 r2, r53 r2, r26 r1); whether to rerun
  them is the user's question.

## What shipped

Nothing in code. Records in `var/levin-phase-floor-06g/references/`:
- `log.md` (this), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
- `scripts/`: the census agent's harness (copied unchanged, its nine `SCRIPTS` equal to the census
  records' hashes) with `x1_capture.py` changed (4 marked changes), and the agent's own scripts
  (`refs_common`, `draw_refs`, `select_refs`, `check_captures`, `slice_refs`, `compare13`,
  `routes_refs`, `one_reading`, `compare06i`, `run_refs`, `launch_refs`, `make_part2_plan`,
  `check_progress_wrapper`, `make_rerun_plan`, `run_rerun`, `launch_rerun`, `score_refs`,
  `score_refs_a6`, `score_refs_s3`, `compare_z00_z12`, `stage3_projection`, `harness_equals_census`,
  `kept_list`); `v1/scripts/`: V1's rehearsal-store harness (deviation 15), `v1_rows`, `v1_capture`,
  `v1_compare_groups`, `v1_dev_one`; diffs in `out/scripts_diff_stage1.txt`, `out/scripts_diff_part2.txt`,
  `out/cmd/11`, `49`, `61`;
- `state/`: the predictions (`predictions_20261101T175120Z.md`), the rule files
  (`v1_rule_20261101T175144Z.md`, `v8_rule_20261101T175144Z.md`, `ruling_addendum3_20261101T201347Z.md`),
  the plan and deviation notes (`plan_stage2.md`), launch transcripts, reference copies of every source
  script with SHA-256 lists;
- `out/`: every command and output (`out/cmd/`, `out/commands.log`), the samples (`out/sample/`), the
  captures' checks, every reading (`out/readings/`, `out/readings_06i13/`, `out/rereads/`, `out/cli/`; the
  laptop's `out/readings_fcc_b5626ec_laptop/`), the monitors, the plans and projections, and the three
  scorings (`out/scores_after_stage2/`, `out/scores_addendum6/`, `out/scores_stage3/`);
- **three registered runs**, all `done`, exit 0: part 1 `levin-phase-floor-06g-references-20261101T175231`
  (60 units), part 2 `levin-phase-floor-06g-references-20261101T201415` (380), the rerun batch
  `levin-phase-floor-06g-references-rerun-20261102T185324` (67).

## The stages, in order

1. **Stage 1** (2026-11-01 ~11:10–11:30Z): read; scripts copied and written; nothing computed.
2. **Stage 2** (17:48–17:52Z): trees and registry; harness synced to the census agent's final scripts
   and checked against the census records' hashes; the scratch copy hashed (EQUAL); both census
   checkpoints hashed (g `03adcbba…`, 06h `3104c18e…`); predictions and the V1/V8 rule files written;
   the draw (seed 20261101) and the selection (102 calls).
3. **Part 1** (17:52–18:41Z): 102 captures, their checks, 06i's 13 groups, routes, 52 readings on 06i's 13,
   the command line on two, the comparison with 06i: all pass. The cost projection for part 2 (20.2 h
   summed, 6.07 h at six-at-once) exceeded the 5 h 40 min gate: stopped and asked.
4. **V1's captures** (19:59–20:09Z, in session): the rehearsal stores hashed before and after (EQUAL);
   24 of 24 captures reproduce; 12 of 12 groups equal between p and g.
5. **Part 2** (2026-11-01 20:14Z – 2026-11-02 18:46Z): 380 readings: 363 done, 9 errors (QAWO
   `ZeroDivisionError` on non-monotone Route B segments), 8 stopped by backstop (a) or (b).
6. **The rerun batch** (2026-11-02 18:53Z – 2026-11-03 06:47Z, tool `ff3e8394`): 6 re-reads (bit-identical),
   56 readings of the 14 stationary-point samples, 5 stopped QAWO readings (2 done, 3 stopped).
7. **The laptop's FCC batch** at `b5626ecb` (run `levin-phase-floor-06g-fcc-laptop-20261102T233229`, 32 of
   32; the orchestrator's checks): the 14 samples' FCC readings, superseding the stage-2 ones.
8. **Three scorings**, each an addition beside the earlier: after stage 2 (2026-11-02T20:44:20Z), per
   addendum 6 (2026-11-03T02:08:15Z), the stage-3 recheck (2026-11-03T06:51:58Z).
9. **Final records** (2026-11-03T07:02Z): the scratch copy hashed again (EQUAL), the stores EQUAL.

## The user's rulings, as the addenda quote them

- **Addendum 1** (V1's 12 group calls): "Happy to have the Filon references added – that sounds like a
  good idea."
- **Addendum 2** (V8's sample): "(b) with the V8 sample, please"
- **Addendum 3** (part 2, after the cost gate):
  - "Quibbling over ~8 hours of compute that only has to be run once strikes me as a false economy."
  - "A runtime that exceeds 30 minutes means that we should stop and check, not guillotine it
    immediately. If it is projected to finish in the next 5 minutes, we should finish it. If it's not
    projected to finish for a day, then we would kill it. In between these extremes is the judgement
    call."
  - On scoring: "We are iteratively improving on our algorithm, and we don't know what to expect."
- **Addendum 4** (the hold; the rerun): "Leave the backstop alone for the rest of part 52. We are
  getting enough good results." On a rerun batch of the stopped QAWO readings: "Happy with this".
- **Addendum 5** (the new tool): "I agree with your implementation (1) and (2), so please go ahead with
  that."
- **Addendum 6** (FCC near a stationary point): FCC near a stationary point is "worth fixing"; V1's FCC
  readings re-run at both resolutions "because we don't trust them"; "Can you run the FCC batch here,
  leaving Stage 3 still running on the Mac Studio?"
- **Scoring after stage 2** (relayed by the coordinator): "we can use the result to inform our view on
  whether we should let Stage 3 run to completion. We should then check whether anything in stage 3
  would change those scorings, after it lands."

## Predictions and verdicts

*Copied in at 2026-11-01T17:51:20Z, before any number of the references was computed; the record is
`state/predictions_20261101T175120Z.md` (read-only, SHA-256 `825b6f04…`), with the seed, the count rule
as applied, the agreement rule for 06i's 13 and how each number is formed.* Prompt 06g §5, verbatim:

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

| Prediction | Number scored on | Verdict |
|---|---|---|
| **V3, by its rule as written** (four clause-(i) readings within min(abserr)/100 count; undecided under 45) | 66 of 90 counted; no counted reference violates | **holds** |
| V3, with clauses (ii) and (iii) accepted in place of all_monotone (addendum 6) | 68 counted (z00, z12 added); z12 violates (1,681.35) | **fails** |
| **V3, as the ruling scores it** (addendum 3: every reference with FCC's two usable readings) | 90 of 90 scored (82 two implementations, 8 FCC only); z12 violates; on two-implementation references alone also z12 | **fails** |
| **V4** (z02, z08, z09, z14 ≤ 0.01), as written and as ruled | 0.00121, 2.1e-7, 1.44e-4, 0.00121 (06e's breakdown 0.0012, 0.00046, 0.0015, 0.0012) | **holds** both ways |
| §7 stop: fewer than 39 of the 77 new count | as written 53 of the 77 count (66 less 13 of 06i's); not a stop | — |
| §7 stop: a reading disagrees with 06i's | 13 of 13 agree | — |

The violation, z12 (`g-1e-8-1-2174` #1, G+q+r): v_g `0x1.0c6c2cc40e857p-86`, ref
`0x1.0c5f18f86f178p-86` (1.35492969293809896756067504718e-26), abserr_g `0x1.fdbb1bc1a92b8p-110`;
|v_g − ref|/e_g = 1,681.35; 06h v `0x1.0c6c2cd926acfp-86`, abserr `0x1.febd253628d92p-110`, 1,678.07. Its
captured integrand equals z00's (the same row at rtol 1e-7) by the 06i-13 criterion; z00's Levin value
is 0.0183 of its abserr from the same reference. z00 and z12's row is one of the 720 stationary-point
rows.

### Scored after stage 2 of the rerun batch (2026-11-02T20:44:20Z)

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

### Rescored per addendum 6 (2026-11-03T02:08:15Z)

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

### The stage-3 recheck (2026-11-03T06:51:58Z)

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

1. **The captures, their checks, the group comparison and the tool on 06i's 13 are stages of the
   registered run**, not in-session steps: the captures re-ran about 90 census units, beyond ten
   minutes (CLAUDE.md; README §5 rule 7). STRUCTURALLY REQUIRED.
2. **One reading per process** (`one_reading.py`: the tool's `reading()` with the command line's exact
   arguments), for the 30-minute stop and per-reading wall time; the command line itself on z25 and
   z13 agrees bit for bit. STRUCTURALLY REQUIRED.
3. **The tool on 06i's 13 read the captures at g**, not 06i's inputs: no worktree at 06e's tree here, and
   making one is a git write. STRUCTURALLY REQUIRED.
4. **06i's inputs' provenance** is 06c's depth20-tiling captures at `e5764a6` (06e's `slice_inputs.py`),
   not "made at 06e's census tree" as the brief said (the coordinator recorded it as its slip).
   STRUCTURALLY REQUIRED (the record follows the records).
5. **The count rule also requires Route B's all_monotone and all four readings.** IMPLEMENTATION CHOICE
   (the tool's docstring), fixed in the predictions.
6. **06i's 13 counted under 06g's tolerance**, on 06i's own readings. IMPLEMENTATION CHOICE.
7. **The harness copied in stage 1 was 06h's**; the census agent's final harness was synced at stage 2
   and checked against the census records' hashes. STRUCTURALLY REQUIRED.
8. **"At most six processes" read as six computing processes** (a reading's parent and the driver
   idle); accepted by the coordinator. IMPLEMENTATION CHOICE.
9. **compare13.py checks the integrand as well as the criterion; check_captures.py the region arrays
   and harness hashes.** IMPLEMENTATION CHOICE.
10. **A reading stopped at 30 minutes did not skip its sample's other readings** (part 1's design;
    superseded by addendum 3). IMPLEMENTATION CHOICE.
11. **The draw reads "finite abserr" on 06h's record; V4 is undecided if one of its four does not
    count.** IMPLEMENTATION CHOICE.
12. **The references ran as registered runs split at the cost gate** (part 1, part 2, then the rerun
    batch, slug `references-rerun`). STRUCTURALLY REQUIRED.
13. **V8's 12 were captured in part 1 with V3's 90.** IMPLEMENTATION CHOICE.
14. **`harness_equals_census.py` was first run from the session scratchpad** and copied into `scripts/`
    afterwards, byte-identical. UNINTENDED DRIFT.
15. **The census harness pointed at the rehearsal stores** (`v1/scripts/`: the rehearsal scope's k sample
    and zend in `rebuild.py`, `prov.py`'s depth and store map, `worker.py`'s 1e-8 tolerance optional);
    24 of 24 reproduce. STRUCTURALLY REQUIRED.
16. **The stores reached through symbolic links in `v1/stores/`.** IMPLEMENTATION CHOICE.
17. **V1's bit-for-bit check is against the store's (1 + z_response) form of the group.** STRUCTURALLY
    REQUIRED.
18. **`one_reading.py --progress`** (`pool.imap` with a progress file); checked bit for bit on six
    readings. IMPLEMENTATION CHOICE.
19. **Part 2's driver: no 30-minute stop; the monitor and the backstop of the ruling file.**
    STRUCTURALLY REQUIRED by addendum 3.
20. **A read-only `sqlite3 -readonly` query of R23's primary preceded the first store hash.**
    UNINTENDED DRIFT (the hashes equal the baseline).
21. **V1's 24 captures ran in session** (about 6 minutes), not registered. IMPLEMENTATION CHOICE.
22. **`one_reading.py --progress-order unordered`** and a `--rerun` marker; covered by the six re-reads.
    IMPLEMENTATION CHOICE.
23. **The rerun batch's order: re-reads, the 56 stationary-point readings, then the stopped QAWO
    readings shortest first.** IMPLEMENTATION CHOICE.
24. **The w samples' PYTHONPATH is the tool's tree then `v1/scripts`;** `harness._rebuild_lru_cache`
    resolves from `references/scripts`, whose `harness.py` is byte-identical. IMPLEMENTATION CHOICE.
25. **r24 qawo r2 skipped as the addenda listed it**, though its phase-span estimate was 7.1–11.9 h.
    IMPLEMENTATION CHOICE (reported).
26. **r26 qawo r1 ran**, its estimate under a day at both ends. IMPLEMENTATION CHOICE.
27. **The stage-2 scorer was run twice**, the first a draft whose outputs were overwritten. UNINTENDED
    DRIFT.
28. **"As written" requires clause (i)**, so z00 and z12 cannot count as written. IMPLEMENTATION CHOICE.
29. **The addendum-6 rescoring kept every other sample on its stage-2 readings.** IMPLEMENTATION CHOICE.
30. **The addendum-6 report's V1 heading and clause line kept the stage-2 wording.** UNINTENDED DRIFT
    (cosmetic).
31. **The stage-3 recheck builds on the addendum-6 scoring.** IMPLEMENTATION CHOICE.
32. **Part 2's projections and backstop (b) were wrong for large-phase-span Route B QAWO readings:**
    the model scaled Route B by segment count, not phase span (projections 30–100× low), and (b)
    stopped at max(6 × projection, projection + 1 h), i.e. about 3,905 s; six readings were stopped that
    the ruling's intent would have let run (r24 r1, r26 r1 and r2, r47 r2, r52 r2, r53 r1; r24 r2 and r53
    r2 by (a) on ordered progress). UNINTENDED DRIFT (the orchestrator's r52 report).
33. **The rerun batch's day-or-more projection (`el × total / done`) ignored chunks in flight**, and
    stopped three readings at their first chunk back that would have finished in 3.0, 8.9 and 13.7 h
    (`out/stage3_projection.json`). UNINTENDED DRIFT.

## Verification performed

In `verification.md`: the captures' bit-for-bit checks, 06i's 13, the re-reads under `ff3e8394`, the
cross-host checks, the usable-clause tally, the stores' and the scratch copy's hashes before and after,
Ray never initialised, the trees.

## Observations not acted on

1. **On four of 06i's Route A samples (9, 22, 25, 27) the tool's FCC readings are offset from 06i's** by
   a constant at both resolutions (2.8e-6 to 4.0e-5 of m), within the agreement rule; QAWO and every
   Route B reading bit-identical. Not investigated.
2. **Route A's QAWO cost scales with phase span**; on the 17 large-span calls QAWO was ruled out
   (addendum 3); 7 of them are FCC-only.
3. **The `ff3e839` worktree holds 71 `.pyc` files in 16 `__pycache__/` directories** (`git status
   --ignored`), every one with mtime 2026-11-02T23:12:03–05Z; they include `tools/`, `AdaptiveLevin/`,
   `ComputeTargets/` and `StoreFixtures/` (which `harness.py` imports), so a process that loaded a
   capture with that tree on its path and bytecode writing on. No process of mine started then: the
   batch's reading r53 qawo r1 had been running since 22:21:43Z, its processes all with `-B` and
   `PYTHONDONTWRITEBYTECODE=1`; my session was idle between 20:44Z and 02:08Z. The cause is not
   established. Not deleted. **At the same second (23:12:04Z), `scripts/__pycache__/harness.cpython-313.pyc`
   was written in this records directory**: the same process imported this directory's `harness.py`
   with the `ff3e839` tree on its path. Left in place for the orchestrator (the laptop should not copy
   it as a record).
4. **R24 sits at QAWO's value and FCC's stage-2 r2 was off on w07–w11** before `b5626ecb`; with the
   laptop's FCC, all four readings agree.
5. **The −ΣE correction** helps most where ΣE is large (r51 0.35, r43 0.47, r33 0.54) and worsens r09
   (1.345).
6. The findings for the board are in `issues.md`.

## State handed on

- **Nothing of mine is running.** The three runs are `done`; the registry holds 46 runs, all finished.
- **The scratch copy** is EQUAL to its certification (my after-hash 2026-11-03T07:02Z; the
  orchestrator's ~07:10Z). **The rehearsal stores** R23 and R24 are EQUAL to the rehearsal baseline.
- **The worktrees** `1c5336e`, `4f4f9ed` and `b5626ec` are clean with `--ignored`; `ff3e839` holds the
  `__pycache__/` of observation 3, left in place.
- **Open for the user:** whether to rerun r47 qawo r2, r53 qawo r2 and r26 qawo r1 (projected 3.0, 8.9 and
  13.7 h); no verdict depends on them.
- **Kept on the Mac Studio:** `KEPT_ON_MAC_STUDIO.txt` (404 pickles and large files, 218 MB; the three
  run directories, 15 files).
