# 07a Part C — the Mac Studio orchestrator's review (§C.2), 2026-11-12

**Verdict: pass.** Every §C.2 check passes. My own readers reproduce every figure the agents report.
- W1, W2, W3 and W5 hold (W5 on gated solves, as the user ruled).
- W4 and W6 fail exactly as §9.5 predicted.
- **W7 fails, which was not predicted.**

None of these is a §8 stop condition.

The records of each step are in this directory:
- `stop1/`, `stop2/` and `stop3/`, with a `notes.md` each;
- `final/`, with `notes_w5_stop.md`;
- `lock_after_rehearsal.txt`.

Every script is my own, written for this review and independent of the agents' scripts. Floats are compared by
`float.hex` or in exact Fractions. Every command used `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B`
or `-I -B`.

## The user's words in Part C, quoted as relayed by the laptop

- **Stop 1:** "Yes, resume the rehearsal agent with the replay".
- **Stop 2:** "Yes, A for Q1 and call's outputs for Q2; launch the census".
- **Stop 3:** "Yes, resume the census agent for scoring".
- **The W5 stop:** "Yes, W5 gated only, and take references for both calls".

## 07's §B.3 checks 1–3 and 8–9, with R30

1. **Scope** (`final/state_at_review.txt`, `final/protected_at_review.txt`). **Pass.**
   - The worktree is at `4bf50d34`, and `git status --short --ignored` is empty in it.
   - The main checkout is at `28ebaf07`, with `status --short` empty.
   - `find var -newer <marker>`, run at 11:34:31Z before the first dispatch, lists nothing outside §C.1's paths,
     `review/` and the references run the user added.
   - All 13,596 protected files are unchanged by hash at each stop and at the end: A3 v2, its scratch copy, R29's
     store, run dir and launch log, 07's census run dir, `var/levin-phase-floor-07/`, and Part A's `measurement/` and
     run dirs.
2. **The runs** (`final/runs_check.txt`). **Pass.**
   - `runregistrykit list` shows 77 runs, all finished.
   - R30 `levin-phase-floor-07a-r30-07a-20261112T113806` is done, and so are
     `levin-phase-floor-07a-k0-20261112T130135` (11/11 commands), `levin-phase-floor-07a-census-20261112T132744`
     (18,560/18,560) and `levin-phase-floor-07a-references-20261112T203555` (16/16).
   - Each manifest names campaign `levin-phase-floor` and prompt `07a`, with `git_dirty` false.
   - R30's manifest names `4bf50d34`.
   - The census-side runs ran with cwd in the main checkout, so their manifests' `git_head` is `28ebaf07`, as 07's
     census manifest's was. Tree h (`4bf50d34`, clean) is named in their purpose and scope, and in every unit record.
3. **The fingerprint.** **Pass.**
   - R30 is `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, with problems none.
   - I re-ran it without `--write` at stop 1 and again here (`final/fingerprint_r30.txt`), from the worktree with
     `PYTHONPATH` set to it.
   - **One slip of mine:** two earlier attempts here ran with cwd in the main checkout. `-m` puts the cwd ahead of
     `PYTHONPATH`, so the main checkout's code ran; it refused the store's schema, and read and wrote nothing. The
     store's mtimes are unchanged and the worktree stayed empty.
8. **The predictions.** **Pass.**
   - `state/predictions.md` is `3ec7b53d…`, unchanged since 13:16:44.807Z.
   - That is after every W0 output (latest 13:16:21Z) and before the census manifest (13:27:44Z).
   - It quotes prompt §6's 31 lines, §9.5's 15 and §10.4's 3 verbatim (`stop2/predictions_check.txt`).
   - Its part 5 wording ("every solve") is superseded by the user's W5 ruling and was not edited; `log.md` says so.
   - The harness diff against 07's copy (`census/out/scripts_diff_at_end.txt`) has 5 files changed: `prov.py`,
     `refs_common.py`, `run_census.py`, `run_k0.py` and `worker.py`. It has 27 new files, 181 + 27 = 208 in all, and
     `log.md` names every one.
   - The rehearsal's five diffs are each named in its log.
9. **The records.** **Pass.**
   - Each `log.md` has §5.1's six sections. The census log adds a references section and a descriptive section.
   - Each deviation is classified:
     - the rehearsal's 21: 12 IMPLEMENTATION CHOICE, 3 STRUCTURALLY REQUIRED, 6 UNINTENDED DRIFT;
     - the census's 29: item 3 carries two classes, one for each of its two parts.
   - Each `verification.md` is written as part of §7a.
   - Each `issues.md` exists: the rehearsal's, and the census's with 7 items.
   - Each `KEPT_ON_MAC_STUDIO.txt` lists what it should:
     - the rehearsal's: R30's five store files with SHA-256, its run dir and launch log;
     - the census's: all 150 files of the census run dir, the k0 and references run dirs, and the large files under
       `census/`.

## The rehearsal values

**`tools.store_value_diff` on R29 and R30**, re-run with `--json`, is byte-identical to the agent's (`f6de2ef1…`).

**Outside `QuadSourceIntegral`:**
- every value table is identical, 360,462 rows;
- the only differences are timing, `label`, `GkSource.parents_digest` (291 rows) and rho reuse (19 rows);
- S0′ holds.

**Inside it** (`stop1/scripts/qsi_compare.py`), of 864 rows:
- 380 moved;
- 0 `total_converged` changes;
- the largest |Δtotal| is 0.0470 of abserr₂₉ (0.0745 of abserr₃₀);
- located rows went 170 → 202, and analytic guards 4 → 0.

**S3, by my reader:** 0 rows have |Δtotal| > abserr₂₉ + abserr₃₀. **Holds.**

**S4, by my reading of R30's four blocks** (`final/scripts/s4_check.py`, `final/s4_check.txt`). **Holds.**
- R29's four guarded 2pt5 J4/Y4 blocks, at k = 3.162e5, z = 1.329e9 and k = 1e6, z = 4.018e9, are each guarded in R29
  and located in R30: 1 root each, with 13, 3, 5 and 12 window pieces, 0 guarded and 0 at depth max.
- The store holds no block value. Each value is the rehearsal agent's R30 replay, accepted because its abserr,
  regions, counts, `converged` and x_span equal the store's block and R29's reference bit for bit, and its root
  equals the reference's.
- |v − ref|/abserr is 4.37e-5, 5.81e-4, 2.06e-4 and 6.17e-4.

The agent's S1 (12/12, worst 0.2176) and S2 (0.0202 and 0.263) are as reported at stop 1. Its 493/493 replays
reproduce R30.

## W0–W7, re-scored from the two checkpoints by my own readers

Both checkpoints were read through my own indices (`stop3/index07*.json`): 07's at `03dfe786…` and 07a's at `461df2ec…`.
Every line was checked by its SHA-256, and every region record against its `pickle_record_sha256`.

**W0 holds** (`stop2/`).
- My parent re-runs of `g-1e-8-3-34739` and `g-1e-7-2-21980` equal 07's census apart from timing, provenance,
  `cache_built` and 07a keys.
- My two runs of `g-1e-7-3-34969` at h are bit-identical.

**W1 holds** (`final/w1_compare.*`, sha `a32eeb05…`), on the user's readings: Q1 is A, and Q2 is the call's outputs.
- At 1e-7, 129,774 untouched calls; at 1e-8, 129,050.
- 0 differ in anything but Rule A's location record.
- 6,691 at each rtol differ in that record alone:
  - 07 has `{phase, variation}`;
  - h has none (12,598) or `{phase, reason: zero_rate}` (784).
- The untouched-only rows are 4,891 at 1e-7 and 4,835 at 1e-8. All have `total` and `total_abserr` equal to 07's in
  hex, with no unit-level difference.

**W2 holds** (`final/w_rescore.*`, sha `022a8d22…`).
- 182,066 scanned calls.
- 0 returned Levin leaves hold a sign change, by my grid test; my grid sets equal the scan's in every call.
- 0 `ends` listings of Levin leaves.

**W3 holds.**
- 0 violations, both ways, on all 182,066 scanned calls.
- 784 skipped phases, all `zero_rate`, none with a sign change.
- No points of kind `end`.

**W4 fails, as predicted** (`final/w4.*`, sha `92fd99d7…`).
- 28 of 33 calls are honest.
- v08, v09, v12, v17 and v22 are at 3.16, 2.21, 3.08, 2.38 and 2.53.
- All 33 calls are bit-equal to 07's: 26 window_both, 6 untouched and 1 own (z24).
- The readings' agreement is the same under 07's tolerance and under min(abserr_h, abserr_07)/100.

**W5 holds** on gated solves, as the user ruled (`final/w5_forced.*`, `final/w_rescore.*`).
- 0 Clenshaw–Curtis regions have a half sent to Levin.
- 0 gated solves take the node-partition sum.
- `num_guarded_regions` is 0 in every call.
- **Descriptive:** 18,236 forced solves take the node-partition sum, 3–10 per call, in 4,416 calls.
  - All 4,416 are among the 4,804 root-located calls; 388 root-located group calls (194 per rtol) have none.
  - The post-unit solves' forced partition counts are 4,416, 3,364, 1,920 and 1,480.

**W6 fails, as predicted.**
- 2,484 pairs, none unpaired.
- r_g ≤ 1 in 2,482 at h, and in 2,482 at 07.
- `3-14501` is at 1.385 and `3-35006` at 1.084, both members converged, the same as at 07.

**W7 fails, not predicted** (`final/w_rescore.*`, `final/stops.*`).
- 4,774 calls are covered: located at both trees, no route or LU change, no new phase.
- 166 window_both calls fall outside W7's cover.
- 25 calls move by more than 1e-5 of abserr_07: 0.0066–0.145, all at 1e-8, all group `G` call 0.
- They are exactly the 25 windows at 1e-8 that stopped at 07 and do not stop at h.
- `converged` is unchanged in all 25, and regions change by 0 or +1.
- 214 covered calls moved at all.
- The census agent's reading (06 or 19 minima windows; iroff1 6 at 07 and 4–5 at h; Part A's basis held only `G+q+r`
  windows) is consistent with my records.

## The user's references (findings, not W verdicts)

Scored by my reader (`final/scripts/refs_check.py`, `final/refs_check.txt`), on 16 readings, all passing 07's checks
(tool `d37e633b…`, input hash, route B, hex).
- Reference: FCC r2. The stated error is the spread of the four readings.
- The spread is 9.2e-8 (`3-678`) and 1.3e-6 (`1-34599`) of the smaller of the two codes' abserrs, so it is far below a
  tenth of both.
- **`1-34599`** (class own), calls #2 and #3:
  - h is honest, at 0.0201 (1e-7) and 0.0150 (1e-8);
  - 07 was not, at 4.03.
- **`3-678`** (class aux_only), calls #2 and #3, both rtols:
  - h is honest at 0.491 of its abserr, which shrank 11×, 1.214e-37 → 1.088e-38;
  - 07 is at 0.135.
- The census agent's numbers equal mine.

## Descriptive, checked

- **Classes** (equal to the agent's):

  | class | 1e-7 | 1e-8 |
  |---|---|---|
  | untouched | 129,774 | 129,050 |
  | aux_only | 8,031 | 7,816 |
  | own | 3,286 | 4,225 |
  | window_both | 2,470 | 2,470 |
  | window_h_only | 1,180 | 1,180 |
  | window_07_only | 0 | 0 |

- **Convergence:** no `converged` flag changes in any class at either rtol.
- **Unconverged calls**, at 07 and at h, at 1e-7 / 1e-8:
  - untouched 62 / 74;
  - aux_only 5 / 5;
  - own 6 / 6;
  - window_both 9 / 11.
- **§10.4's list is empty:** no call goes from converged to unconverged with a route change.
- **The newly located:** 2,360 calls (1,180 per rtol), all converged both before and after.
- **Windows:**
  - per rtol, 3,650 at h against 2,470 at 07;
  - stopped: 1,247 against 1,157 at 1e-7, and 1,236 against 1,195 at 1e-8;
  - stops new at h: 90 (1e-7) and 66 (1e-8). Stops lost at h: 0 and 25;
  - calls left unconverged with a stopped window: 7 and 9, the same as at 07.
- **The raw phase:**
  - the largest |θ| at a gated region's end is 8.87e12 (analytic, row 2-35099) and 4.12e12 (group);
  - the smallest unchanged-route margin to 6π is 4.77e-8 (group, 1-14543), and to 20π 3.0e-4 (group, 3-34129).
- **Cost:**
  - census wall 23,284 s against 07's 22,708 s (+2.5 %);
  - summed unit wall 53,465 s against 51,682 s;
  - regions 1,362,352 against 1,361,060 at 1e-7, and 1,546,782 against 1,543,904 at 1e-8;
  - no unit over budget.

## Not re-checked by me

- The agent's route-change tallies by solve role (24,559 solves; 10,014 C2 changes, all towards Clenshaw–Curtis; 14,545
  LU changes, all towards least squares). I used only the per-call booleans and counts.
- The aux_only abserr ratio distribution.
- The 06m disagreement list.
- The references run's projection refit.
- The rehearsal's S1 and S2 beyond stop 1's agreement.
