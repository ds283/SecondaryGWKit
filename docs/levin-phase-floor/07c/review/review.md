# levin-phase-floor 07c Part C — the Mac Studio orchestrator's review (§C.2)

**Pass.** R32's S0′–S4 hold and the census's Y0–Y9 hold. Each was re-scored by my own readers, which are independent
of the rehearsal and census agents' scripts. R2-28 is converged and dishonest at 1.1434 of its abserr, as §9.5
predicted, and is reported against `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`. No stop of §8, §9.7 or §C.1
arose.

- Code: `67b5af74`, variant (S), parent `cff4237f`.
- Worktree: `SecondaryGWKit-lpf07c-cc3368c`, detached at `cc3368ce`, clean with `--ignored` at the end.
- Dates: 2026-11-19 to 2026-11-20 (UTC).

## The user's words at the stops (relayed by the laptop session)
- **Dispatch:** "Yes, dispatch Part C".
- **Stop 1:** "Yes, continue the rehearsal agent".
- **Stop 2:** "Yes, launch the census".
- **Stop 3:** "Yes, score Y1–Y9 and run the review".

## Protected inputs
These are hashed before dispatch, at stop 2 and at stop 3 by `checks/` and `scripts/hash_protected.sh`:
- 160 recorded files: R31's store, 07b's census run directory (150 files) and A3 v2's scratch copy. All are OK each time.
- 17,277 further files: R29, R30, 07a's census run directory, Part A's `measurement/`, 07b's
  `compare_census_calls.calls.jsonl` and A3 v2. The three listings are byte-identical, and there is no `-wal`/`-shm`.

The census agent hashed the same before and after, with the same result. The second reviewer opened every input
read-only.

## R32 against R31 (`stop1/`, `final/s3_exact.txt`, `final/s124_exact.txt`)
Run `levin-phase-floor-07c-r32-07c-20261119T140417`:
- `done`, exit 0, 14:04–14:18Z;
- fingerprint `1b332ffa…`, re-taken without `--write`;
- R32's files hash the same before and after the fingerprint.

**S0′ holds.** `git diff --stat 5e19c4b6 cff4237f` (code paths) lists only `CLAUDE.md`.

**Outside `QuadSourceIntegral`**, my reading of the `store_value_diff` JSON, which is byte-identical to the agent's
(`c42c6d77…`):
- every value table is identical: 360,462 rows;
- the differences are only in timing, `label`, `GkSource.parents_digest` (291) and the rho-reuse metadata (26).

**Inside** (`stop1/scripts/qsi_compare_07c.py`):
- 864 rows: 559 identical and 305 moved;
- `total_converged` went up 1 and down 0 (861 → 862);
- summed Levin-record regions went 219,776 → 211,594;
- 2 group records flipped `phase_limited` False → True, and each is still converged.

**S3 holds, exactly** (`final/s3_exact.txt`, my reader, Fractions):
- 864 rows, 251 moved, 0 violations of |t32 − t31| ≤ e31 + e32;
- the largest ratio is 5.25051e-6, at k = 1e6, q = 1e5, r = 1e6, z = 437173.17;
- the named record G+q+r at (1e5, 1e6, 1e6), z 144627.7, went from unconverged at 143 regions to converged at 86;
- that record is honest at 0.05331484572900594 against Part A's `ref-R31-3-20363-g1`, and equals (S)'s replay bit for bit.

**S1 holds** (my exact re-score; references checked against 06g's `scores_stage3/scores.json` V1 `fcc_r2`):
- 12 of 12 are converged, located and honest, the worst 0.0670 (w06);
- w00–w07 equal R31's values bit for bit;
- w08–w11 moved, each with one chase stop, at 0.000474, 0.0208, 0.0133 and 0.00195.

**S2 holds:** ratios 0.0202477 and 0.11116, both located.
- Row 1 equals R31.
- Row 2 moved and is scored against the same reference as 07b's reviewed R31 record.
- The p1-reference reading files are not in 07c's rehearsal directory, so I checked the reference string against R31's
  reviewed score file, not against the readings.

**S4 holds:** 4 of 4 blocks are located, guard no leaf, and are honest at 4.37e-5, 5.81e-4, 2.06e-4 and 6.17e-4.
Each block's value, abserr and reference equal R31's reviewed record.

**The replay:** 445 of 445 records reproduce R32. The lock was released at 14:49:20Z, no Ray process is left
(re-checked at the review), and the six readers exit 0 with R32 unchanged.

## Y0 (`stop2/`)
My re-runs, through the agent's harness with output in `stop2/`:
- two K0 units at the parent (`h-1e-8-0-2108`, `h-1e-8-3-14505`) equal 07b's census in every call (hex) and in both
  totals;
- the acting unit `h-1e-8-0-33929`, run twice at the tree, is identical both times and equal to the agent's run;
- its two stopped calls equal (S)'s replay records.

`predictions.md` (`cad4fd7f…`, mode 0444) was written after the last Y0 output and before the census. Its four quoted
blocks equal the prompt's lines exactly.

The projection was 43,312 s central; the census took 43,080 s.

## Y1–Y9, re-scored from the checkpoints

Sources:
- `stop3/` is my own stop-3 reader (`final/scripts/stop3_y1_y8.py`, `stop3_y8_ambiguous.py`).
- `final/` holds the second reviewer's readers (`build_index.py`, `extract.py`, `y4y6.py`, `y235.py`, `y7_conv.py`,
  `y9.py`), adapted from 07b's review. Their SHA-256s are in `final/sha256.txt`.
- Checkpoints, each line checked by its SHA-256 and every pickle record by its own:
  - 07c `33d94ce2…`
  - 07b `396784bc…`
  - 07a `461df2ec…`
  - 07 `03dfe786…` (Y7's population)

| | Verdict | My readers' numbers | Agent |
|---|---|---|---|
| **Y1** | holds | 207,310 untouched calls (1e-7: 73,021 analytic + 24,775 group; 1e-8: 84,152 + 25,362), all equal to 07b's in hex value, hex abserr, `converged`, `num_regions` and `max_depth`. 8,508 units are untouched in every call, and all have hex `total` and `total_abserr` equal. Every call carries the three counts. | same |
| **Y2** | holds | 182,066 scanned calls, 0 Levin leaves holding a sign change. The grid sets are all empty; the scan's 2,734 `ends` listings are all Clenshaw–Curtis leaves. | same |
| **Y3** | holds | 0 violations either way on all 182,066 scanned calls; 784 skipped phases, all `zero_rate` without a sign change. | 0 on its 23,993 |
| **Y4** | holds; R2-28 as predicted | 312 scorings on 306 calls (07b's U4 190; Part A's R4 60, R5 60, R2-28 ×2). R4's and R5's references were recomputed from the readings. Converged and dishonest: only R2-28 (07b's R2-28 and R2-28~o, and `c-1e-7/8-0-13757-5`), at **1.14344** at both rtols, with 1 chase stop; at 07b it was unconverged at 1.1071. Unconverged: v04 (0.000156), `c-1e-8-3-13490-2` (0.0086), `c-1e-8-3-13615-3` (0.0083). The largest converged ratio other than R2-28 is 0.805 (R2-08~o). R4's maximum is 0.416, with 58 of 60 converged (07b: 2 of 60). Readings not agreeing within min(abserr)/100: R2-27, R2-29, R3a-0-33666-a13 (both rtols), all honest. | same; it reports those six as not scored |
| **Y5** | holds | 0 Clenshaw–Curtis regions with a Levin half, 0 non-forced node-partition sums, 0 guarded regions; 18,238 forced partitions in 4,416 calls (= 07b). | same |
| **Y6** | holds | 24 of 24 converged and honest against S10, from 1.58e-5 to 0.828. | same |
| **Y7** | holds | 2,484 pairs, none unpaired, all r_g ≤ 1; the largest is 0.4671 (3-34923 #1, = 07b). | same |
| **Y8** | holds | 453 of 453 ids equal (S)'s replay records in all five outputs. 347 have a unique 07b match. 106 match twin calls, identical in 07b and in 07c, and the replay equals both. | 453 / 448 calls |
| **Y9** | holds | (i) 178 unconverged calls, all group (82 at 1e-7, 96 at 1e-8). 167 have depth-20 leaves (714, all Clenshaw–Curtis), none below a refused leaf. 11 have no depth-20 leaf. (ii) Of 07b's 198 newly unconverged calls, 193 converge; exactly §9.5's five stay unconverged, each untouched, equal to 07b's call, and without a depth-20 leaf. The same reader on 07b's census reproduces 07b's 192 of 198. | same |

**Y9, a reading to note.** For a call with no depth-20 leaf, "every depth-20 leaf is below a refused leaf" is taken
as not true. The census agent fixed that reading in `predictions.md` before the run, and §9.5's own prediction
requires it: the five budget misses stay unconverged while Y9 holds.

Read literally (vacuously true), clause (i) would fail on 11 calls: the five, plus `c-1e-8-0-12591-4`,
`c-1e-8-0-12783-4`, `c-1e-8-0-1989-4`, `c-1e-7-0-33995-5`, `c-1e-8-3-34417-4` and `c-1e-8-3-35036-4`. These are budget
misses with no depth-20 leaf.

"Refused ancestor" is the census instrument's test: a refused rate-rule evaluation at a strictly smaller depth whose
span contains the leaf's. It was implemented independently, in Fractions.

## Where the fix acted (descriptive; the agent's figures, consistent with my Y1 counts)

| rtol, class | acted / calls | stops | stopped leaves | noise bisections |
|---|---|---|---|---|
| 1e-7 analytic | 34,395 / 107,416 | 37,241 | 247,001 | 554,650 |
| 1e-7 group | 12,550 / 37,325 | 13,377 | 277,949 | 113,273 |
| 1e-8 analytic | 23,264 / 107,416 | 26,905 | 182,952 | 502,641 |
| 1e-8 group | 11,963 / 37,325 | 12,653 | 269,387 | 111,249 |

- It acts in 82,172 of 289,482 calls (28.4%).
- 10,800 acted calls still equal 07b's in the five outputs.
- By route, 64.5% of stopped leaves are Levin: mostly at depths 3–4, and 607,044 of all stopped leaves were untrusted.
- Clenshaw–Curtis stopped leaves are mostly at depths 11–13.
- The instrument's counts equal the code's on all 289,482 calls.

## Convergence
- **Against 07b:**
  - 193 group calls newly converged (97 at 1e-7, 96 at 1e-8; all where the stop acted);
  - **none newly unconverged**, so §C.5's list is empty;
  - no analytic call changes;
  - rows' `total_converged`: 136 gained (68 / 68), 0 lost;
  - 0 calls moved past the sum of the two abserrs.
- **Against 07a:**
  - 5 calls lost: §9.5's five, at 1e-8, untouched;
  - 5 gained, at 1e-8: 0-14312 #5, 0-38454 #5, 1-1748 #4, 3-35037 #4 and 1-35186 #6. These are 07b's five newly
    converged calls;
  - rows: 3 lost and 4 gained;
  - unconverged calls are 82 + 96, as at 07a (07b: 179 + 192);
  - 328 group calls moved past the sum of the two abserrs (07b: 326), item 6's population, left to 07d.

## Cost
- **Census wall:** 43,080 s (11.97 h). That is 0.995 of the central projection, 0.83 of the projection with 07b's
  1.197 shortfall applied, and 0.957 of 07b's census.
- **Units:** 0 over budget and 0 in error.
- **Regions:** 0.917 of 07b's (1e-7: analytic 0.965, group 0.883; 1e-8: analytic 0.976, group 0.891).
- **R32's QSI stage:** 11m19s, against R31's 11m28.6s.

## Observations for the records commit (not acted on)
1. **The stop acts far beyond §9.3's basis** (the census agent's `issues.md` 1). It reaches 57,659 analytic calls,
   and most stopped leaves are Levin, many of them untrusted. §9.3 says Part A's sample had no analytic chase.
   - No analytic call is unconverged.
   - The referenced analytic calls (R2, R3) are honest.
   - The agent proposes a reference sample of analytic calls where the stop acts. I agree it is worth an issue.
2. **A stopped untrusted leaf's 1/(1 − r) declaration can overstate by orders of magnitude** (agent's `issues.md` 2).
   On `h-1e-7-1-14470` analytic #7 the abserr is 242× 07b's, and the call is still honest.
3. **R2-28** is converged and dishonest at 1.1434, as §9.5 predicted, against the open issue.
4. **The rehearsal's issues:**
   - two rows are unconverged at depth 20 in both R31 and R32;
   - captures are not byte-identical across trees.
5. **The census copy-back set.** It holds four `out/runs_hashed_*.txt` files of 2,083,08x bytes each. That is under
   2 MiB but over 2,000,000 bytes, and they are near-identical hash listings. The laptop should decide at copy-back
   whether one listing plus the comparisons is enough.

## What stays on the Mac Studio
`KEPT_ON_MAC_STUDIO.txt` lists 34 files with their SHA-256 and size:
- the four checkpoint extracts and indices, and my `stop3/calls_07{b,c}.jsonl`, all derived from checkpoints that stay
  here;
- `stop1/svd_r31_r32.json` (40 MB);
- the three 2.46 MB hash listings;
- the stop-2 raw re-run outputs.

Everything else under `review/` (68 files, the largest `final/y4y6.json` at 236 KB) is meant to be copied back. The census run directory, the
k0 run directory and the agents' large files are on the agents' own KEPT lists.

## State at the end
- Registry: 96 runs, all finished (`final/registry_at_review.txt`).
- Lock: free (`final/lock_at_review.txt`). No Ray process under `/tmp/ray-sgk` (`final/ray_processes_at_review.txt`).
- Worktree: at `cc3368ce`, clean with `--ignored`.
- Nothing is committed or pushed.
