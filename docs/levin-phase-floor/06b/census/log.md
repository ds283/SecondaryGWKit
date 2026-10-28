# levin-phase-floor 06b §5 — the census (Mac Studio)

**Subject:** prompt 06b §5 on the Mac Studio (mps026298), Ray-free, no lock. 06a's census re-run
at the code commit `4ddc3c2` against 06a's run as the "before".
- **Trees:** $b$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-92bc62f`,
  detached at Part B's dispatch SHA `92bc62f`, whose code is `4ddc3c2`'s. $p$ is the worktree
  `…-lpf06b-da5c24d` at the code commit's parent `da5c24d`, used for Gate W0's reproduction only.
- **Run:** `levin-phase-floor-06b-census-20261028T161212`. Launched 16:12:12Z (sh 8931, driver 8935);
  the last unit was recorded at 17:02:45Z, and the run ended `done`, exit 0, by 17:03:32Z. That is
  about 51 min, against 06a's 45.5. All 18,560 units are `done`: none over budget, none in error.
- **Commit:** none here; the laptop commits this directory.

**Result:**
- **W0 holds**, all three clauses.
- **W1 holds.** At both rtols, every group call and every row the rule reaches equals 06a's bit
  for bit.
- **W2 fails.** 17 of the 24 sampled calls have $|{\rm value}_b - {\rm ref}| > $ `abserr`$_b$, by up
  to 19.7×. b's values are closer to the reference than 06a's: 4 of 24 are outside their requested
  tolerance under b, against 23 of 24 under 06a. But b's declared error understates its true error.
- **W3 holds.** 44 regions are accepted as phase-limited at 1e-7, all in analytic-block calls. Each
  one's halves' floor sum is at least 0.504 of its own floor.
- **W4 fails.** Three analytic-block calls at 1e-8 that moved reach depth 20, after the premise
  failed. No unit is over budget.
- **The store:** the scratch copy is byte-identical to its certification before, during and after,
  and gives `0e32f24a…`. No worker initialised Ray.

## Before the launch (in order; every command and its output in `out/cmd/`)

1. **Harness copied** (`00`): 06a's 39 scripts were copied from the code worktree's
   `docs/levin-phase-floor/06a/scripts/`, which equals (`diff -r`) `var/levin-phase-floor-06a/scripts/`.
   They went into `scripts/`, and unchanged into `state/scripts_06a_reference/`. The diff against
   06a is saved at preparation and at the end: `out/scripts_diff_against_06a_at_prep.txt` and
   `…_final.txt`.
2. **Rows** (`01`): 06a's `state/rows.json` was copied; its sha256 `aa523d28…` is equal.
3. **The store** (`02`–`05`, `20`, `21`): the scratch copy's five files equal
   `copy_files_certified.txt` (EQUAL), with no journal, `-wal` or `-shm` file. `fingerprint_copy.py`
   gives `0e32f24a…`, problems none, at $b$ and at $p$.
4. **W2's capture list** (`06`): `state/w2_capture.json`, sha256 `217723d1…`. 06a had 1,213
   phase-limited calls at 1e-7; 975 of them, on 529 units, have no sign change of θ′ in 06a's scan.
5. **Development runs** (`07`–`10`, `state/dev/`): smoke1 and smoke2 at $b$, smoke3 at $p$, and
   `w2_reference.py` on two captured calls.
6. **Gate W0** (`12`–`15`): **holds.**
   - **Clause 1** (`15`). `e05c34f..da5c24d` outside `docs/` and `prompts/` changes 8 files:
     - `CLAUDE.md`;
     - three whose ASTs without docstrings equal `e05c34f`'s: `ComputeTargets/primitive_phase.py`,
       `main.py` and `ComputeTargets/tests/test_primitive_phase.py`;
     - four test modules whose ASTs differ: `test_g_phase_reanchor`, `test_phase_groups`,
       `test_quadsource_integral` and `test_tk_source_functions`. None of them is among the 102
       parent-tree modules the harness imported in W0's parent workers
       (`state/W0/p_tree_modules.json`). The only test modules imported are `ComputeTargets.tests`
       and `ComputeTargets.tests.test_main_plumbing`.
   - **Clause 2** (`14`). At $p$, three units are identical to 06a's in every computed value
     (06a's C0 (d) definition) and on C0 (c)'s terms. Their dispatcher counts are equal too
     (descriptive).

     | Row | rtol | Call value/abserr pairs | Regions (× 5 arrays) | Total |
     |---|---|---|---|---|
     | `low-G+r-1` (2-38302) | 1e-7 | 14 | 9 | `0x1.5e6b3999d1ab2p-101` |
     | `ctrl-low-1` (2-2095) | 1e-7 | 14 | 8 | `-0x1.e3272e29a4c5dp-102` |
     | `0-1971` | 1e-8 | 10 | 38 | `0x1.6b3058aae2397p-66` |

   - **Clause 3** (`14`). `b-1e-7-1-34599` was run in two fresh workers. The row is fixed by rule:
     the capture-list unit with the most captured calls. The two runs are identical in every
     computed value, including the analytic blocks, W3 and the captured calls' values.
   - No worker initialised Ray.
7. **Driver smoke test** (`16`, `state/smoke/`): 2 rows × 2 rtols, 4 units done.
8. **Predictions** (`18`, `19`): `state/predictions.md`, sha256
   `8df517969e4973aa222200c0e6ba5d35b796cca1cab323ab44fe17a4476f6e4e`, mtime
   2026-10-28T14:13:55.8Z, before the launch at 16:12:12Z. Part A is the prompt's lines 296–319
   byte for byte. Part B is W2's selection rule and the readings. The file is unchanged since
   (`41`).
9. **Pre-launch** (`22`–`24`): every script equals those W0 and the smoke test ran. The trees are
   clean, the registry held 27 runs, all finished, and the lock was free.
10. **Launch** (`25`) and **one check** (`26`): `running`, driver 8935, sh 8931, stderr empty.

## Rulings recorded before the launch

- **The analytic-block reading (W1's row rule and W4)**, as written in Part B of
  `state/predictions.md`. For an analytic block's call, "phase-limited in 06a" is read from b's
  premise counter, and "moved" is a failed premise check. The user's ruling, relayed by the
  coordinator on 2026-10-28, verbatim: *"For 06b, that reading is reasonable. I don't think it's
  worth spending a lot of time worrying about it."* `state/predictions.md` was not edited for it;
  its Part B already holds the reading. The reading's cross-check holds: on the group calls,
  "phase-limited in 06a" and "b asked the premise" agree on all 74,650 calls (`37`).
- **Deviation 17** below: see the coordinator's note there.

## After the run

- **Status** (`27`): `done`, 18,560 of 18,560 units, exit 0, stderr empty.
  - Every record is at `92bc62f` with empty status, and carries `run_census.py` `5d47334e…`.
  - Every unit's modules are 102 from its own tree, none from $p$ or the main checkout.
  - Every worker reported `ray_initialized` false, with no foreign module.
  - `/tmp/ray-sgk` held 67 entries before W0 and 67 after the run. This is descriptive only:
    R15 and R16 used it in the same period.
- **Over-budget pass:** none was needed, since no unit is over budget.
- **The store** (`28`–`32`): the copy's five files equal the certification after the run and after
  the fingerprint, with no journal file. `fingerprint_copy.py` at $b$ gives `0e32f24a…`, problems
  none.
- **Checkpoint:** `checkpoint.jsonl`, sha256
  `f934ea245c0086529fb9372a9ad457d4d8668114ce346dca5440404c766646ad`. The run directory is 1.8 GB:
  16 region pickles, 5 input pickles (975 captured calls), the events and the worker logs. It is
  kept in full.

## Predictions and verdicts

`state/predictions.md` (sha256 `8df51796…`, mtime 14:13:55Z, before the launch at 16:12:12Z)
holds §5's W1–W4 unchanged. They are copied here unchanged from the prompt:

- **W1 — nothing else moves.**
  - **The rule:** every Levin call with no phase-limited region in 06a equals 06a's call bit for
    bit, in value, abserr, region count and maximum depth. Every row all of whose calls are such
    equals 06a's row in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
- **W2 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that were phase-limited in 06a and whose result moved under
    $b$, with no sign change of $\theta'$ in 06a's scan (`spscan.py`). It is stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's own
    floor reported. The two must agree to within a hundredth of $b$'s `abserr`; a call where they
    do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_b - {\rm ref}| \le$ `abserr`$_b$ for every call left in the sample.
- **W3 — the exit now fires only where bisection would not help.**
  - **The rule:** at `1e-7`, every region $b$ accepts as phase-limited has a halves' floor sum at
    least $\rho$ times its own. Recompute it from the halves' samples in the harness. Do not trust
    the code's flag.
  - **Holds** if no region violates it.
- **W4 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

| | Number | Verdict |
|---|---|---|
| **W0** | see "Before the launch", 6 | holds |
| **W1** | At 1e-7: 36,112 group calls not phase-limited in 06a, all 36,112 equal in value, abserr, regions and depth; 6,777 rows all of whose calls are such, all equal in `total` and `total_abserr`. At 1e-8: 35,605 of 35,605 calls and 6,177 of 6,177 rows equal. A further 1,918 rows at 1e-7 (2,358 at 1e-8) are excluded only by an analytic block's call that was phase-limited in 06a. (`37`) | **holds** |
| **W2** | 24 drawn (12 low, 12 mid, no shortfall; `state/w2_selection.json`). Every call's re-run from its reloaded inputs equals b's call bit for bit. $\|S_9 - S_{10}\|/$`abserr`$_b \le$ 0.0066 on all 24, so none is dropped. $\|{\rm value}_b - S_{10}\| \le$ `abserr`$_b$ on 7; it fails on 17, with err/abserr up to **19.7** (median of all 24: 1.79). (`35`, `37`) | **fails** |
| **W3** | 44 regions accepted phase-limited at 1e-7, in 27 units. All 44 are in analytic-block calls, all Levin, at depth 4–6. Each recomputed own floor equals the stored `phase_err` bit for bit. Halves' sum / own floor ranges from 0.504 to 0.891, so 0 violate the rule, re-scored from the hex floors as well. There are 0 recomputation errors, and the region arrays hold the same 44. (`37`, `out/w3_regions.csv`) | **holds** |
| **W4** | Moved calls: 1e-7, 1,213 group and 12,633 analytic, largest depth 14 and 18. 1e-8: 1,720 group and 17,722 analytic, largest depth 14 and **20**. Three moved calls at 1e-8 reach depth 20, all analytic blocks after the premise failed: `b-1e-8-2-1704` call 16 (Y1, 1,136 regions); `b-1e-8-2-13964` call 5 (J1, 31,885 regions, 127,539 evaluations, unit wall 159 s); `b-1e-8-3-34987` call 9 (J2, 14,293 regions, 69 s). 0 units over budget. (`37`) | **fails** |

**W2, examined further** (descriptive, after the score; `38`, `39`):
- **The reference is sound on the three calls cross-checked:** #21, the worst; #1; and #4.
  - $S_{11}$ agrees with $S_{10}$ to within 0.0018 of `abserr`$_b$.
  - On #1 the adaptive driver at `rtol` 1e-11 (215 regions) agrees with $S_{10}$ to 0.0016 of
    `abserr`$_b$.
  - On #21 and #4 the driver at `rtol` 1e-11 returns b's value exactly, with b's regions (4 and 2),
    each accepted as resolved. Its error against $S_{10}$ stays at 19.7 and 5.7 `abserr`$_b$. The
    error b's value carries is shared by those regions and their halves, so the step-(4) residual
    does not see it, and the regions' eq. (151) floors are below it.
- **Against each call's own request,** max(atol, rtol|value|):
  - b's error exceeds it on 4 of 24, by at most 3.42×;
  - 06a's exceeds it on 23 of 24.

  b's values are better, and its declared errors are smaller than its true errors on 17 of 24.
  06a's declared errors were within the truth on 23 of 24 (`err_06a/abserr_06a`: median 0.068,
  max 1.13). b's `abserr` fell to between 1/10 and 1/16,500 of 06a's.

## Descriptive items (`out/cmd/37`, unless noted)

- **How often the exit fired.**
  - **At 1e-7:** phase-limited group calls fell from 1,213 on 585 rows under 06a to 0 under b.
    `total_phase_limited` rows fell from 585 to 0. 32 analytic-block calls are phase-limited under
    b, on 27 rows.
  - **At 1e-8:** phase-limited group calls fell from 1,720 on 745 rows to 0. 800 analytic-block
    calls are phase-limited under b, on 442 rows.
  - **06a's that would have failed the premise:**
    - at the region level, matching 06a's region arrays to b's by start and end: all 1,213 (1e-7)
      and all 1,720 (1e-8) of 06a's phase-limited group-call regions fail it and are bisected under
      b;
    - by b's premise counter: on the group calls, 1,217 checks at 1e-7 and 1,733 at 1e-8, all
      failed;
    - on 06a's phase-limited analytic-block calls (by the reading): 12,633 at 1e-7 and 17,722 at
      1e-8, every one with a failed premise; 13,031 of 13,075 checks and 23,544 of 24,989 failed.
- **Depth**, row depth over group calls as 06a defines it:
  - at 1e-7 the largest is 12 under 06a and 14 under b; rows at ≥ 11 rise from 50 to 80;
  - at 1e-8 the largest is 13 under 06a and 14 under b;
  - including the analytic blocks, b's largest is 18 at 1e-7 and 20 at 1e-8 (3 rows; W4).

  Full histograms are in `37`.
- **The sample's errors before:** `err_06a/abserr_06a` has median 0.068, max 1.13 (1 of 24
  outside). `err_06a/request` exceeds 1 on 23 of 24 (`39`).
- **Totals** (585 rows moved at 1e-7, 745 at 1e-8):
  - Median $|\Delta{\rm total}|$/`total_abserr`$_{06a}$ by (band, k) is 0.02–0.17. Rows moving by
    more than `total_abserr`$_{06a}$: 32 at 1e-7 and 36 at 1e-8.
  - In every one of them `WKB_Levin` moved, half with `analytic_rad` too. Up to 2.2×10⁴ of
    06a's error, relatively up to 6.9×10⁻³.
  - The largest moves are group calls with an interior stationary point that 06a accepted
    phase-limited at depth 0, and b bisects to depth 11–12. Of the group calls moving more than
    their 06a abserr, 30 of 38 at 1e-7 (35 of 47 at 1e-8) have a sign change of θ′.
  - `total_abserr`$_b$/`total_abserr`$_{06a}$ on the moved rows has median 0.36 at 1e-7 and 0.22
    at 1e-8.
  - `out/moved_calls.csv` and `out/census_rows_06b.csv.gz` hold the details.
- **Stationary-point rows** (06a's 720; 07's, not scored): their totals moved on 323 rows at 1e-7
  and 370 at 1e-8. Row depth is unchanged on 690 (686) and up by 2–6 on the rest. Their
  phase-limited group calls go to 0.
- **Convergence:**
  - `total_converged` false on 2,578 rows under 06a and 2,349 under b at 1e-7; 2,638 and 2,388 at
    1e-8.
  - Unconverged group calls: 6,964 and 5,788 at 1e-7; 7,647 and 6,186 at 1e-8.
  - `convergence_kinds.py` (06a's changed copy, `--tree b`) at 1e-7: bound only 4,018, resolution
    1,770, phase-limited 0. Under 06a: 4,014 / 1,737 / 1,213. At 1e-8: 3,784 / 2,402 / 0, against
    3,745 / 2,182 / 1,720. No row is capped (`40`, `out/convergence_kinds_b.txt`).
- **Cost:**
  - Wall time 51 min against 06a's 45.5.
  - The Levin timed sections sum to 2,477 s at 1e-7 and 3,375 s at 1e-8, against 2,399 s and
    2,760 s.
  - The largest unit is 2.4 s at 1e-7 and 159 s at 1e-8 (the depth-20 call), against 0.8 s and
    1.0 s.
  - Cost is not an argument (README §5 rule 4).

## Deviations from the prompt (numbered and classified)

1. **`prov.py`:** trees `p` and `b`, 06a's run and records, and `WIDER` one level deeper.
   STRUCTURALLY REQUIRED.
2. **`worker.py`, module evidence:** the tree evidence names `AdaptiveLevin.levin_quadrature` and
   the premise check; each unit records its modules by root (`modules`); `bye` lists the tree's
   modules. STRUCTURALLY REQUIRED (the brief; W0 clause 1).
3. **`worker.py`, per-call records:**
   - per call, `n_phase_limited_regions`;
   - the code's premise check, counted by a pass-through at `LQ._roundoff_exit_premise_holds`
     during the timed section, at b only;
   - the analytic blocks' calls summarised (`other_calls`) and their region arrays pickled
     (`other_regions`).

   IMPLEMENTATION CHOICE: 06a recorded no per-call result for the analytic blocks, and W1's row
   rule and W4 need them.
4. **`worker.py`:** W3's recomputed floors. STRUCTURALLY REQUIRED.
5. **`worker.py` and new `make_capture.py`:** W2's inputs, captured for all 975 candidate calls.
   STRUCTURALLY REQUIRED; capturing every candidate is an IMPLEMENTATION CHOICE.
6. **`run_census.py`:** `--tree`, unit ids `<tree>-…`, and `w3` and `capture` in each 1e-7 unit.
   STRUCTURALLY REQUIRED.
7. **`launch.py`:** prompt `06b`, purpose and scope. STRUCTURALLY REQUIRED.
8. **`harness.py`:** `job_name="levin-phase-floor-06b"`, as the brief asks. datastorekit 0.5.0
   stores it and reads it nowhere (`11`). IMPLEMENTATION CHOICE.
9. **`rows.json` copied from 06a** (same sha256). A3 v2 itself is neither read nor hashed; only the
   copy is checked. IMPLEMENTATION CHOICE.
10. **New `w0.py` and `w0_clause1.py`.** IMPLEMENTATION CHOICE throughout:
    - the determinism row is fixed by rule;
    - there is one parent worker per unit;
    - clause 1's import list is taken from W0's parent workers (the reproduction run), not from a
      separate dry run.
11. **`out/cmd/13`: the first W0 check stopped** (ModuleNotFoundError) at the clause-3 capture
    comparison, which unpickles records that need the tree on the path. Clause 2 had already
    printed "holds". It was re-run with `PYTHONPATH` set to b as `14`; nothing else was re-run.
    UNINTENDED DRIFT.
12. **New `w2_reference.py`.** STRUCTURALLY REQUIRED (the prompt allows it). Its method choices are
    IMPLEMENTATION CHOICEs: pieces made by the driver's own bisection, each piece evaluated with
    `_adaptive_levin_subregion`, `math.fsum`, and the re-run check.
13. **Development runs** `07`–`10` and the driver smoke test `16`. IMPLEMENTATION CHOICE.
14. **`out/cmd/17` is a mistaken command:** it wrote the prompt's lines to `/tmp/__x`, deleted the
    file, and checked nothing. `check_predictions.py` (`18`) replaced it. UNINTENDED DRIFT.
15. **`state/predictions.md` was edited once, before the launch:** its header named `out/cmd/17` as
    the byte check, and now names "18 and 19". Part A is unchanged (`19`). UNINTENDED DRIFT.
16. **New `check_predictions.py` and `scripts_equal.py`.** IMPLEMENTATION CHOICE.
17. **The brief says no census process imports `ray`.** 06a's harness and the production modules
    import the `ray` package. Ray is stood in and never initialised: every worker's
    `ray_initialized` is false. STRUCTURALLY REQUIRED reading.
    **The coordinator, 2026-10-28, before the launch:** the brief's wording "no census process
    imports ray" was stricter than the binding rule, which is that no Ray session appears (none
    started by the census, none under /tmp/ray-sgk from it). The laptop classifies this deviation
    as arising from the brief.
18. **Scripts are not `black`-formatted:** they are records of the run. IMPLEMENTATION CHOICE.
19. **No over-budget pass:** no unit was over budget. Recorded for completeness; it is not a
    departure from the prompt.
20. **New `analyse06b.py`.** Its first `select` (`33`) stopped with a KeyError, because it parsed
    the rtol's `-` in a unit id as a separator. That was fixed with `rowkey()`, and the select was
    re-run as `34`. The selection was written only by `34`. UNINTENDED DRIFT.
21. **The first `report` (`36`) stopped** while writing the per-row file, on rows with no group
    call, after printing every score. That was fixed, and two descriptive blocks were added (which
    component of a row's total moved, and the W4 calls' detail). It was re-run as `37`; the scores
    are the same. UNINTENDED DRIFT for the stop; the additions are IMPLEMENTATION CHOICE.
22. **New `w2_crosscheck.py` (`38`) and `w2_request.py` (`39`):** descriptive checks of W2's
    reference and of the errors against each call's request, run after W2 was scored. They are
    not part of the rule. IMPLEMENTATION CHOICE.
23. **`convergence_kinds_06a.py` run with `--tree b --capped-run <this run> --capped-tree b`**
    (`40`). `out/convergence_kinds_b.txt` is its output without the per-record input lines.
    IMPLEMENTATION CHOICE.
24. **`out/cmd/41` ended with exit 1** from GNU `stat` (BSD flags; 06a's deviation 17 again). Every
    other line of it is recorded. UNINTENDED DRIFT.

Deviations 11, 14, 15, 20, 21 and 24 are the UNINTENDED DRIFT. None of them touched a store, a tree,
or any record outside `var/levin-phase-floor-06b/census/` and the run directory.

## Verification performed
- **The trees:** $b$ at `92bc62f`, `status --short --ignored` empty at W0, at the launch, on all
  18,560 records and at the end (`41`). $p$ at `da5c24d`, likewise. The main checkout is at
  `28ebaf0`, status empty, throughout.
- **The copy:** byte-identical to `copy_files_certified.txt` before W0, before the launch, after the
  run and after the fingerprint. `0e32f24a…` before and after.
- **The run:** 18,560 records, every unit `done`, every pickle record matching its sha256 (read in
  `37` and `40`), and `stderr.log` empty.
- **No Ray:** every census worker reported `ray_initialized` false.
- **06a's records were only read:** its checkpoint is still `c13685b4…` (`33`).

## Observations not acted on

All are in `issues.md`:
- **C1:** W2's failure. b's `abserr` understates its error on calls the premise bisects.
- **C2:** W4's failure. Three analytic-block calls chase to depth 20 at 1e-8.
- **C3:** 06a's stationary-point calls move under b by up to 6.2×10⁴ of their 06a abserr (2.5×10⁴ at 1e-7);
  they are 07's.
- **C4:** the analytic blocks are where the exit fired most, and 06a's census did not record them.

## State handed on
- **Nothing is running.** The run is `done`. No lock was taken. Nothing was written in either
  worktree or the main checkout outside `var/levin-phase-floor-06b/census/` and the run directory.
- **The scratch copy is unchanged** (`0e32f24a…`) and is still kept for the user to delete.
- **The run directory** `var/runs/levin-phase-floor-06b-census-20261028T161212/` (1.8 GB) is kept
  in full: checkpoint, region pickles and captured inputs, for 07's census.
- **For the orchestrator:** copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/` and
  `state/` to `docs/levin-phase-floor/06b/census/`. Leave out `out/cmd/40` (3.6 MB), and the
  pickles under `state/dev/`, `state/W0/` and `state/smoke/`, which hold about 50 MB of captured
  inputs.
