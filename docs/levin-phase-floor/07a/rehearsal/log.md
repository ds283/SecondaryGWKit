## Part C §5: rehearsal R30 on the Mac Studio

**Prompt:** [`07a-window-stop-and-phase-variation.md`](../07a-window-stop-and-phase-variation.md) §5 ("as 07 §5"), under
Part C's dispatch (`4bf50d34`) and the user's word at stop 1: *"Yes, resume the rehearsal agent with the replay"*.

**What was measured:** the code commit `1ec734d7`: the C2 gate on the raw phase, Rule A replaced, and the window's count
as `dqage`'s. Its parent is `dadd63a0`. The parent rehearsal is R29, built at `e182bc7a`, whose code is the parent's.

**Host and trees:** mps026298.
- Every build, replay, score and reader ran in the worktree `SecondaryGWKit-lpf07a-4bf50d3`, detached at `4bf50d34`.
  - Its key files: `levin_quadrature.py` `4ea1948b`; `QuadSourceIntegral.py` `33cb161e`; the launcher `b0135218`;
    `main.py` `a5d47f5f`; `store_value_diff.py` `7997af85`. The last four are R29's tree's.
- Nothing was written in the worktree; `status --short --ignored` was empty throughout.
- The main checkout stayed at `28ebaf07`.

**Model:** Claude Opus 5.5, the rehearsal subagent, in four turns:
1. the gate and R30's launch;
2. the end check, the comparisons, and stop 1;
3. the replay's set-up and launch;
4. the trust reading, S1–S4, the readers, the release and these records.

**Date:** 2026-11-12, 11:37–12:35 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE.
- **R30 is `done` with `1b332ffa…`.** S0′ holds.
- **Outside `QuadSourceIntegral` every value table equals R29's.**
- Inside it, 380 of 864 rows moved, and no `total_converged` changed in either direction.
- **S3, S1, S2 and S4 hold.** The readers exit 0 with the store unchanged. The lock is released.
- The 49 abserr-only records are explained, by the replay, as a route or LU change in one auxiliary trust-check solve
  each (`issues.md` item 1).

### What shipped

No code. The records under `var/levin-phase-floor-07a/rehearsal/` (committed by the laptop as
`docs/levin-phase-floor/07a/rehearsal/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and:
- **`scripts/`:**
  - adapted:
    - `rec.sh` (`065afcc6`; R29's, path by sed, `out/rec_sh.diff`);
    - `normalise_jobname_labels.py` (`41da5c72`; 07's, `out/normalise_jobname_labels.diff`);
    - `block_reference_r30.py` (`efac125e`; R29's, `out/block_reference_r30.diff`);
    - `s1_score_r30.py` (`0f2a0469`) and `s2_score_r30.py` (`6fc3fb30`), both R29's, changes marked;
  - unchanged copies:
    - `classify_non_qsi.py` (`26690e61`, 05's);
    - `tabulate_qsi_guard_r29.py` (`47e18cbe`, kept as the source of the verbatim reader);
    - `run_readers_worktree.sh` (`ec96cf09`);
  - new:
    - `tabulate_qsi_r29_r30.py` (`adff0052`) and `qsi_summary_r29_r30.py` (`cc7bc729`);
    - `s3_score.py` (`f12ebafd`);
    - `trust_reading.py` (`49f1456c`);
    - `run_block_reference_r30.sh` (`4a2bd2c3`) and `block_score_r30.py` (`7c92da59`);
    - `capture_compare.py` (`f240a7c2`);
    - `replay_summary_r30.py` (`4ea0b46c`);
    - `peek_qsi_metadata.py` (`e33881f0`, deviation 8);
  - **`capture/`:** R29's on-disk harness (hashes in `state/capture_harness_r29_sha256.txt`), with `prov.py`
    (`b4cd02b9`) and `worker.py` (`63c28b49`) changed where marked "prompt 07a r30". It adds the new `replay07a_r30.py`
    (`c7ff4d85`) and `run_r30_pipeline.sh` (`c4b10450`). Diffs are in `out/capture_harness_r30_diff.txt` and
    `out/run_pipeline_r30.diff`.
- **`out/`:**
  - `commands.log`;
  - the gate, launch, end, stop-1 and final checks;
  - `r30_status.json`;
  - the diff JSON (kept, over 2 MB), its normalised copy, and the classifier outputs;
  - the QSI tabulation, its CSVs and its summary;
  - the replay's stage outputs: `replay_rows.txt`, `replay_run.txt`, `replay_check.txt`, `replay_calls.csv`,
    `replay_calls.checks.json`, `blocks1_run.txt`, `blocks2_*.txt`, `blockcheck.txt`, `blocks.json`,
    `replay_summary.txt`, `r30_pipeline.log`;
  - `trust_reading.txt`;
  - `s1_scores.*`, `s2_scores.*`, `s3_scores.*`;
  - `block_reference/`, `block_reference_run.txt`, `block_scores.*`, `capture_compare.txt`;
  - `readers.txt`, `readers_check.txt`.
- **`state/`:** `lock.log`, the launch lines and markers, every store-hash file, the replay's rows and specs, and the
  store links.
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`, `state/capture/`,
  `state/inputs/`).
- **The store is kept:** `var/datastores/levin-phase-floor-07a-r30*`.

### Predictions and verdicts

| Rule (prompt 07a §5, as 07 §5) | Scored on | Verdict |
|---|---|---|
| 1. S0′: `git diff --stat e182bc7a dadd63a0 -- . ':!docs' ':!prompts'` lists nothing the build imports | `CLAUDE.md` only | **holds** |
| R30 with 07's command line (`--job-name levin-phase-floor-07a`, slug `r30-07a`, `--prompt 07a`, `--cpus 6`) gives `1b332ffa…` | `done`, exit 0, `1b332ffa…`, `classes` equal R29's; 9.7 min | **holds** |
| 2. The lock: `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command of its own | acquired 11:37:54Z, `set-pid` 75161, released 12:31Z; one look-only `status` shared its shell with hashing (deviation 15) | **holds** (with deviation 15) |
| 3. Outside `QuadSourceIntegral`: every value table identical; every other difference of the four volatile kinds; no row in one store only | 360,462 value-table rows identical; 0 unclassified on the normalised copy | **holds** |
| 3. Inside it: every moved row tabulated, against both stores' `total_abserr`, regions, maximum depth, located points, window pieces, guarded regions and `total_converged`; the `total_converged` changes each way | 380 rows tabulated (`out/qsi_rows_r29_r30.csv`); 1,234 records (`out/qsi_records_r29_r30.csv`); `total_converged` changes 0 up, 0 down | tabulated |
| 4. S3: \|t30 − t29\| ≤ e29 + e30 on every row with finite `total_abserr` in both | 864 scored exactly; 0 violations; largest 0.0288 | **holds** |
| 5. S1: V1's twelve locate their point at z ≈ 10211 and are honest against 06g's references | 12 of 12, largest 0.2176 (w11); equal to R29 | **holds** |
| 5. S2: P1's two rows locate within one spacing and are honest against `p1_references` | both; 0.0202 and 0.263 | **holds** |
| 6. S4: R29's four guarded J4/Y4 blocks each locate their root, guard no leaf, and are honest against R29's block references | 4 of 4: the root bit-equal to the reference root, 0 guarded, ratios 4.4e-5 to 6.2e-4 | **holds** |
| 7. The readers on R30 exit 0, store byte-identical; lock released, `ray-processes` empty; R29 and R30 kept | 6 of 6; identical; free; nothing listed; R29 equal to its KEPT hashes | **holds** |

**The inside-QSI tabulation** (`out/qsi_tabulation_r29_r30.txt`, `out/qsi_summary_r29_r30.txt`):
- **Rows:** 484 unmoved and 380 moved.
  - `total` moved in 25 rows, `total_abserr` in 60, `analytic_rad` in 168 and `analytic.abserr` in 339.
  - The other 163 moved only in analytic-block fields or counters.
- **Largest changes:**
  - \|Δtotal\|/abserr is 0.0470 against R29's abserr and 0.0745 against R30's, at (3.162e5)³, z 1.918e8 (regions
    69 → 73).
  - The largest group-record \|Δv\|/abserr is 0.336 (R29) and 0.753 (R30), on G+q+r at (1e6)³, z 2.099e7: regions
    1 → 68, `phase_limited` False → True.
  - \|Δanalytic_rad\|/analytic.abserr is at most 5.5e-5.
- **Regions:** summed, 114,460 → 114,245. They change in 17 rows. Maximum depth changes in 2 rows (17 → 9 and 11 → 15).
  Five rows are at depth 20 in each store.
- **Convergence:** `total_converged` is 862 of 864 in both stores, with 0 changes either way. Group and analytic-block
  `converged` also have 0 changes either way.
- **Location:**

  | | Group records: points / window pieces / guarded | Analytic blocks: points / window pieces / guarded | Rows located | Rows guarded |
  |---|---|---|---|---|
  | R29 | 44 / 562 / 0 | 336 / 2,172 / 4 | 170 | 2 |
  | R30 | 44 / 562 / 0 | 400 / 2,515 / 0 | 202 | 0 |

  - The new points are 64 2pt5 J4/Y4 blocks in 32 rows, which R29's Rule A skipped. Each locates one root, with 2–13
    window pieces.
- **The replay** (`out/replay_summary.txt`): every located record holds exactly one root. There are 44 G+q+r, 200 J4 and
  200 Y4, with 3,077 window pieces and no record over 64 points. The four q-r calls of the replayed rows are skipped with
  reason `zero_rate`.

### Deviations from the prompt

1. **`rec.sh` is R29's** (`b5279dd9`) with its record path changed by sed. IMPLEMENTATION CHOICE.
2. **A GNU-style `stat -f` format string failed** in a read-only gate check. The hashes and `ls -la` carry the same
   information. UNINTENDED DRIFT.
3. **`normalise_jobname_labels.py` is 07's** (`b02fd3fd`), with its two prefixes (`07-` → `07a-`) and its docstring
   changed. IMPLEMENTATION CHOICE.
4. **`classify_non_qsi.py` was run on both the raw and the normalised JSON.** The raw run gives 5,148 unclassified, every
   one a `label` whose job-name prefix differs; the normalised run gives 0. The verdict is the normalised copy's, as 07's
   S0. STRUCTURALLY REQUIRED: the brief sets R30's job name.
5. **The inside-QSI tabulation is a new script**, `tabulate_qsi_r29_r30.py`.
   - R29's script classes a row "untouched" by the newer store's four counts, against a store that lacks them. That
     cannot apply when both stores carry the counts.
   - A row here is "moved" if and only if a value, counter or guard field differs.
   - The reader and the field classification are copied verbatim.
   - IMPLEMENTATION CHOICE.
6. **`qsi_summary_r29_r30.py`**, a summary of the two CSVs for the stop-1 report. IMPLEMENTATION CHOICE.
7. **The classifier prints no provenance header,** so the shell wrote one (script and input SHA-256, the tree's HEAD).
   IMPLEMENTATION CHOICE.
8. **Four read-only exploratory commands at stop 1 printed output that was not saved:** a peek at R30's metadata
   structure (`peek_qsi_metadata.py`; a first one-liner failed on sqlite quoting) and three inline analyses of the CSVs.
   `qsi_summary_r29_r30.py` re-derives and saves their content. UNINTENDED DRIFT.
9. **The capture harness is R29's on-disk copy** (hashes equal to R29's records). `prov.py` was changed to the one tree
   h and R30's store links, with the parents assertion one level shallower. IMPLEMENTATION CHOICE.
10. **`worker.py`: `guard_07`'s skipped-phase entries also record `reason`.** 07a's entry form carries no variation.
    STRUCTURALLY REQUIRED.
11. **`worker.py`: a route diagnostic, `route_07a`, added for the orchestrator's reading.**
    - Pass-throughs at `_adaptive_levin_subregion` (the existing one, extended) and at
      `_Basis_SinCos.build_Levin_data` record, for each solve: V, 07's estimate (the parent's expression on the same
      `theta_prime_Cheb`), `is_direct` and the caller.
    - It returns exactly what it is given. The replay reproducing R30 bit for bit, 493 of 493 records, shows it changed
      no value.
    - IMPLEMENTATION CHOICE.
12. **`replay07a_r30.py` is R29's driver run at one tree.**
    - The 49 abserr-only records' rows are added.
    - The check reports `route_07a`.
    - The blockcheck compares each capture with R29's.
    - IMPLEMENTATION CHOICE.
13. **A first `rows` run failed with a KeyError.** A comment I inserted had commented out the rest of a dict line. It was
    corrected before any output existed or was used. UNINTENDED DRIFT.
14. **Pass 1 of the block check (`blocks1`) ran in session, as the pilot, before the detached pipeline ran the rest.**
    The pipeline is not a registered run, as R29's was not. IMPLEMENTATION CHOICE.
15. **One look-only lock `status` ran in a shell command that had first run `shasum` and `ls`.** That is against "each
    lock command … the only process that command starts". Those processes had ended before it ran, and it changed
    nothing. UNINTENDED DRIFT.
16. **The block captures are not byte-identical to R29's,** as the dispatch's comparison asked.
    - They differ only in the unit id, the tree tag and the call's own result fields (`out/capture_compare.txt`).
    - The integrand's equality is shown instead: 1001-point samples bit-identical, with equal `x_span` and `kw`
      (`out/block_scores.txt`).
    - STRUCTURALLY REQUIRED: a capture records its own run.
17. **S4 used two scripts.**
    - `block_reference_r30.py` is R29's, with the skipped-phase form changed and the located points recorded. It was
      run at h without `--reference`.
    - `block_score_r30.py` is new.
    - R29's references were reused, as §5 item 6 says.
    - IMPLEMENTATION CHOICE.
18. **S3 is scored by a new script, exactly, in `Fraction`s.** IMPLEMENTATION CHOICE.
19. **S1 and S2 are R29's scripts, adapted. Some names were left as R29's:**
    - S1 keeps its key names `v29`/`abserr29`, which hold R30's numbers, so its JSON is byte-identical to R29's
      (`ae704a3c`). R30's twelve calls equal R29's bit for bit.
    - S2 still prints "replay reproduces R29" for R30's check, and its `R30_equals_R29` expression carries a dead
      conditional.
    - Cosmetic. UNINTENDED DRIFT.
20. **`trust_reading.py` and `replay_summary_r30.py` are new.** IMPLEMENTATION CHOICE.
21. **Two read-only looks printed output that was not saved:** the pilot's `emit.jsonl` and the block replays' JSON
    fields. The pipeline's check and `block_score_r30.py` save the same content. UNINTENDED DRIFT.

### Verification performed

- **Gate** (`out/00-gate.txt`, `out/01-*`):
  - the worktree at 4bf50d34, clean; the main checkout at 28ebaf07, clean;
  - S0′ as written;
  - `1ec734d7..4bf50d34` outside docs/prompts: empty;
  - `e182bc7a..4bf50d34` under ComputeTargets, ObjectFactories, RunDrivers, main.py, tools, config, CosmologyModels and
    requirements.txt: empty;
  - 73 runs, all finished; no lock; no Ray process; `auth_token` mode 600;
  - R29 equal to its KEPT hashes.
- **R30** (`out/r30_launch_check.txt`, `out/r30_end_check.txt`):
  - launched 11:38:05Z (sh 75153, driver 75161); manifest at 4bf50d34, clean, launcher `b0135218`;
  - `done` at 11:47:48Z; no Traceback, WARNING or ERROR.
- **Stores:**
  - R29 is byte-identical before, during and after everything (`state/r29_store_sha256_*`, `state/store_sha256_*`,
    `state/stores_sha256_final.txt`).
  - R30 is unchanged after its build: across the comparisons, the replay and the readers.
- **The comparisons:** `tools.store_value_diff` R29 against R30 (15.9 s, exit 1, `f6de2ef1`), run from the worktree.
- **The replay** (`out/replay_check.txt`, `out/blockcheck.txt`):
  - 493 of 493 records reproduce, with Ray uninitialised in all 223 workers;
  - the four block calls reproduce in pass 1 and pass 2;
  - wall time: replay 325 s, blocks 52.5 s each pass.
- **The readers** (`out/readers_check.txt`): 6 of 6 exit 0; R30 identical; 0 Tracebacks; the worktree clean.
- **The lock** (`state/lock.log`): nine commands, each exit 0: ray-processes, acquire, set-pid, ray-processes, status,
  ray-processes, release, status (free), ray-processes (nothing).
- **Final** (`out/final_check.txt`):
  - 74 runs, all finished; `~/.raypoolkit` empty; no Ray process;
  - the worktree and the main checkout clean;
  - `measurement/`, `var/levin-phase-floor-07/`, R29's run directory, A3 v2 and its scratch copy all untouched;
  - no `__pycache__` in the records.

### Observations not acted on

1. **The 49 abserr-only records** (`issues.md` item 1).
   - In each, exactly one auxiliary trust-check solve changes route or LU choice: a quarter in 41, a comparison half
     in 8.
   - No region's own solve changes, so neither does the value.
   - This refines the orchestrator's reading.
   - It bears on §6 W1's diagnostics: those must count the trust check's solves, or W1 will see calls move that it
     predicts are bit-identical.
2. **R30's four S4 block values equal R28's bit for bit.** R30's S1 and S2 calls equal R29's (and R28's) bit for bit.
3. **`route_07a` over 4,614 replayed calls:** 312 solves change C2 route and 430 change LU choice, of 427,437
   (`issues.md` item 3).
4. **The readers subsample k and z at random,** so their file counts differ from R29's (`issues.md` item 4).
5. **The stale comments the dispatch named** were met in passing and not touched (`issues.md` item 5).

### State handed on

- **R30:** `levin-phase-floor-07a-r30-07a-20261112T113806`, `done`, `1b332ffa…`. Its store is kept
  (`KEPT_ON_MAC_STUDIO.txt`).
- **R29** is unchanged and kept.
- **The lock is released.** Nothing of this agent runs. The worktree and the main checkout are clean and unmoved.
- §6, the census, was not touched.
