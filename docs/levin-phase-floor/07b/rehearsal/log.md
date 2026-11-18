## Part C §5: rehearsal R31 on the Mac Studio

**Prompt:** [`07b-levin-leaf-beside-a-window.md`](../07b-levin-leaf-beside-a-window.md) §5 ("as 07a §5"), under Part C's
dispatch (`0a103b5a`). The user's word at stop 1: *"Yes, continue, and settle which store understates"*.

**What was measured:** the code commit `5e19c4b6`, variant (h). After the tree's trust test passes a non-direct Levin
leaf, an edge rule (on a leaf touching a window) or a range rule (on a leaf touching none) on |θ′| can refuse it. Its
parent is `7b82a0cd`. The parent rehearsal is R30, built at `4bf50d34`, whose code is `1ec734d7`'s.

**Host and trees:** mps026298.
- Every build, replay, score and reader ran in the worktree `SecondaryGWKit-lpf07b-0a103b5`, detached at `0a103b5a`.
  - Its key files: `levin_quadrature.py` `8956c7b6`; `QuadSourceIntegral.py` `33cb161e`; the launcher `b0135218`;
    `main.py` `a5d47f5f`; `store_value_diff.py` `7997af85`. The last four are R30's tree's.
- Nothing was written in the worktree; `status --short --ignored` was empty at every check.
- The main checkout stayed at `28ebaf07`, clean.

**Model:** Claude Opus 5.5, the rehearsal subagent, in four turns:
1. the gate and R31's launch;
2. the end check, the comparisons, and stop 1;
3. S3, the replay, S1, S2, S4, the readers, the release, and the references' plan;
4. the references' launch, the scoring, and these records.

**Date:** 2026-11-16, 19:16–20:35 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE.
- **R31 is `done` with `1b332ffa…`.** S0′ holds.
- **Outside `QuadSourceIntegral` every value table equals R30's.**
- Inside it, 569 of 864 rows moved. `total_converged` went down in one row and up in none.
- **S3, S1, S2 and S4 hold.** The readers exit 0 with the store unchanged. The lock is released.
- **Which store understates** (descriptive; the user's word at stop 1): in all 7 group records whose |v31 − v30| exceeds
  e30 + e31, **R30 understates (3.87 to 59.3 of its abserr) and R31 is honest (0.0014 to 0.298)**. Each reference's stated
  error is at most 5.9e-4 of min(e30, e31).

### What shipped

No code. The records under `var/levin-phase-floor-07b/rehearsal/` (committed by the laptop as
`docs/levin-phase-floor/07b/rehearsal/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and:
- **`scripts/`:**
  - adapted from 07a's rehearsal (each diff in `out/`):
    - `rec.sh` (`9a80a313`);
    - `normalise_jobname_labels.py` (`bdc2d44d`);
    - `tabulate_qsi_r30_r31.py` (`a7dc4941`) and `qsi_summary_r30_r31.py` (`50d33417`);
    - `s3_score.py` (`f6396428`);
    - `s1_score_r31.py` (`305d44d0`), `s2_score_r31.py` (`5c499eb2`) and `block_score_r31.py` (`fcb8b436`);
    - `run_block_reference_r31.sh` (`687b4d81`);
  - unchanged copies:
    - `classify_non_qsi.py` (`26690e61`, 05's);
    - `block_reference_r30.py` (`efac125e`);
    - `run_readers_worktree.sh` (`ec96cf09`);
    - `run_06j.py` (`6227aea4`, Part A's plan driver);
    - `refs/variant_loader.py` (`1cb49a13`, Part A's);
  - new:
    - `s0prime.py` (`f7e33882`);
    - `protected_listing.sh` (`8875e3d0`);
    - `launch_r31.py` (`98890b5e`; as it ran the replay: `state/launch_r31_as_replay.py`);
    - `gen/make_replay07b_r31.py` (`01ad6081`) with `gen/replay07b_r31_extra.py.txt` (`afa8e225`);
    - `refs/refcap_replay.py` (`4f6af0d8`), `refs/refs_projection_r31.py` (`5dc49f8a`) and `refs/understates.py`
      (`14e238c5`);
  - **`capture/`:** R30's on-disk harness (hashes in `state/capture_harness_r30_sha256.txt`).
    - `prov.py` (`1c976bf8`) and `worker.py` (`2a253652`) are changed where marked "prompt 07b r31".
    - `replay07b_r31.py` (`ac7319ca`) is generated from R30's `replay07a_r30.py`.
- **`out/`:**
  - `commands.log`;
  - the gate, launch, end, stop-1, readers, rehearsal-end and final checks;
  - `r31_status.json` and `r31_fingerprint_own.txt`;
  - the diff JSON (kept, over 2 MB), its normalised copy, and the classifier outputs;
  - the QSI tabulation, its CSVs and its summary;
  - `s0prime.*` and `s1_scores.*`–`s3_scores.*`;
  - `block_reference/`, `block_scores.*` and `blocks.json`;
  - the replay's outputs: `cmd/r31-replay/`, `replay_calls.csv`, `replay_calls.checks.json` and `refcap.json`;
  - `readers.txt`;
  - `refs/`:
    - the capture replays at both codes;
    - the projection and the tolerances;
    - `readings/` and `understates.{json,txt}`;
  - `cmd/r31-refs/`.
- **`state/`:** `lock.log`, `deviations_draft.md`, the launch lines and markers, every store-hash file, the protected
  listings, the replay's rows and specs, the plans, `refs_plan_r31.md` with its mtime record, and `variants/`.
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt` (the diff JSONs, `extract/`, `state/capture/`, `state/inputs/`).
- **The store is kept:** `var/datastores/levin-phase-floor-07b-r31*`.

### Predictions and verdicts

| Rule (prompt 07b §5, as 07a §5) | Scored on | Verdict |
|---|---|---|
| 1. S0′: `git diff --stat 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'` lists only `CLAUDE.md` and 89c368f2's four files, each AST-equal with docstrings removed | exactly those five; the four AST-equal (Part A's `strip_docstrings`) | **holds** |
| R31 with R30's command line (slug `r31-07b`, `--prompt 07b`, `--job-name levin-phase-floor-07b`, `--cpus 6`) gives `1b332ffa…` | `done`, exit 0, `1b332ffa…` (`status.json`, and again by `runregistrykit store fingerprint`, read only); `classes` equal R30's; 13.6 min | **holds** |
| 2. The lock: `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command of its own | acquired 19:18:38Z; `set-pid` 10624; released 20:04:53Z after `ray-processes` listed nothing; ten commands, each its own shell command, `>>` | **holds** |
| 3. Outside `QuadSourceIntegral`: every value table identical; every other difference of the four volatile kinds; no row in one store only | 360,462 value-table rows identical; 0 unclassified on the normalised copy (timing, `label`, `parents_digest` 291, `rho_reuse` 23) | **holds** |
| 3. Inside it: every moved row tabulated, with the rows whose `total_converged` changes in each direction | 569 rows (`out/qsi_rows_r30_r31.csv`); 9,206 records; `total_converged` 0 up, 1 down | tabulated |
| 4. S3: \|t31 − t30\| ≤ e30 + e31 on every row with finite `total_abserr` in both, or one §9 predicts to move further | 864 scored exactly (Fractions); 0 violations; largest 0.128 | **holds** |
| 5. S1: V1's twelve locate their point at z ≈ 10211 and are honest against 06g's references | 12 of 12; largest 0.0670 (w06); 6 equal R30, 6 moved and more accurate | **holds** |
| 5. S2: P1's two rows locate within one spacing and are honest against `p1_references` | both; 0.0202 (= R30) and 0.111 (R30 0.263) | **holds** |
| 5. S4: R29's four blocks each locate their root, guard no leaf, and are honest against R29's block references | 4 of 4: root bit-equal, 0 guarded, 4.37e-5 to 6.17e-4; each equal to R30 | **holds** |
| 6. The readers on R31 exit 0, store byte-identical; lock released, `ray-processes` empty; R30 and R31 kept | 6 of 6; identical; free; nothing listed; R30 equal to its KEPT hashes | **holds** |

**§9.5's S3 addition** (a row beyond S3's bound reported with its refusals): no row is beyond it.
- The replay records the refusals anyway.
- Over the 4,295 replayed calls: **range rule 591 refusals in 373 of 967 group calls, and 6,733 in 2,667 of 3,328 analytic
  calls; the edge rule refuses nothing.**

**The inside-QSI tabulation** (`out/qsi_tabulation_r30_r31.txt`, `out/qsi_summary_r30_r31.txt`):
- **Rows:** 295 unmoved, 569 moved.
  - `total` moved in 379, `total_abserr` in 386, `analytic_rad` in 548.
  - `phase_limited` flipped False → True in 5,336 analytic-block and 348 group records, never the other way.
- **Largest changes:**
  - \|Δtotal\| is 0.278 of R30's abserr and 0.238 of R31's, at (1e5, 3.162e5, 3.162e5), z 12004.58 (regions 80 → 243).
  - The median over the 379 is 5.8e-3.
  - \|Δanalytic_rad\| is at most 3.1e-3 of R30's `analytic.abserr` and 8.0e-3 of R31's.
- **Regions:** summed `WKB_Levin_num_regions` went 114,245 → 152,117 (1.33×).
  - 386 rows rose and none fell; the median per-row ratio where changed is 1.33, the largest 14.6.
  - Maximum depth changed in 93 rows, and the rows at depth 20 went from 5 to 7.
  - QuadSourceIntegral's stage took 11 m 28.6 s against 7 m 32.7 s (1.52×).
- **Convergence:** `total_converged` went 862 → 861.
  - The one row: (1e5, 1e6, 1e6), z 144627.7067, record `subintervals[1].groups[0]` G+q+r. Its regions went 2 → 143 and
    its depth to 20, with 4 range refusals.
  - That record's \|Δv\| is 0.619 of e30 and 0.897 of e31, and the row's \|Δtotal\| is 3.2e-4 of its abserr.
  - No analytic block's `converged` changed.
- **Location:** unchanged (group 44 / 562 / 0; analytic 400 / 2,515 / 0; 202 rows located).
- **Group records:** 1,053 values moved.
  - \|Δv\| > e30 in 144 of them (largest 59.6), > e31 in 27 (largest 2.37), and > e30 + e31 in 7 (exactly, in Fractions;
    largest 1.53).

**Which store understates** (the user's word at stop 1; descriptive, not scored against any rule;
`out/refs/understates.txt`).
- **The rule** was fixed in `state/refs_plan_r31.md` before any reading (mtime 20:09:13.98Z, `2c2514b0`; the run began
  20:11:20Z):
  - reference = the tool's FCC r2; stated error s = the four readings' scatter;
  - no verdict if s > min(e)/100;
  - a store understates if \|v − ref\| − e > s, and is honest if e − \|v − ref\| > s.
- **The method:** `tools/filon_reference.py` (`d37e633b`, 07's V4 and Part A's tool), Route B, QAWO and FCC at
  resolutions 1 and 2, independent of the Levin driver.
- **The captures:** each captured from R31 by the replay harness. Each reproduces R31 bit for bit at the tree, and R30 bit
  for bit at 07a's module (`git show 1ec734d7`, installed by Part A's `variant_loader`), so the one integrand is both
  stores'.
- **The results:**

  | Record (k, q, r; z) | label | R30: \|v−ref\|/e (regions) | R31: \|v−ref\|/e (regions, range refusals) | s / min(e) | answer |
  |---|---|---|---|---|---|
  | (1e6, 1e5, 1e6; 329.64) | G−q+r | **59.29** (5) | 0.0055 (112, 2) | 5.9e-4 | R30 understates |
  | (3.162e5, 1e5, 3.162e5; 996.42) | G−q−r | **35.60** (3) | 0.0070 (38, 3) | 7.7e-5 | R30 understates |
  | (3.162e5, 1e5, 3.162e5; 6.945e6) | G−q−r | **9.20** (8) | 0.298 (68, 1) | 4.4e-5 | R30 understates |
  | (1e6, 3.162e5, 1e6; 1.918e8) | G−q−r | **5.62** (1) | 0.0150 (45, 1) | 3.5e-5 | R30 understates |
  | (1e6, 1e6, 1e6; 36286.8) | G−q−r | **4.36** (6) | 0.0050 (66, 1) | 2.2e-5 | R30 understates |
  | (1e6, 1e6, 1e6; 331553) | G−q−r | **3.87** (1) | 0.0014 (58, 3) | 2.5e-5 | R30 understates |
  | (1e6, 1e5, 1e6; 36286.8) | G−q+r | **2.91** (58) | 0.0045 (82, 3) | 3.4e-5 | R30 understates |

- Every R30 record here was **converged**, as was every R31 record. Every one of R31's calls had range-rule refusals
  (1–3) and no edge refusal.
- Every reading integrated every segment in t: no direct segment, and every segment monotone.

### Deviations from the prompt

1. **`rec.sh` is R30's** (`065afcc6`) with its record path changed by sed. IMPLEMENTATION CHOICE.
2. **S0′ is scored by a new script, `s0prime.py`,** with Part A's `strip_docstrings` copied verbatim. The raw
   `git diff --stat` is also in `out/00-gate.txt`. IMPLEMENTATION CHOICE.
3. **The lock commands ran from the worktree with `PYTHONPATH` set** (the dispatch's rule), where R30's ran from the main
   checkout. Each entry has a header line written by `echo`, a builtin. IMPLEMENTATION CHOICE.
4. **A listing of every protected file** (mtime, size, path) was taken before, at stop 1, after the rehearsal and at the
   end. IMPLEMENTATION CHOICE.
5. **A first protected-listing one-liner failed:** zsh does not word-split `$P`. It was read-only, and its output was
   overwritten. UNINTENDED DRIFT.
6. **The fingerprint was also re-derived** by `runregistrykit store fingerprint` without `--write`. Read only; the store
   was re-hashed equal. IMPLEMENTATION CHOICE.
7. **`normalise_jobname_labels.py` is 07a's**, with its prefixes (`07a-` → `07b-`) and docstring changed.
   IMPLEMENTATION CHOICE.
8. **The classifier was run on the raw and the normalised JSON.** The raw gives 5,148 UNCLASSIFIED, every one a `label`
   whose job-name prefix differs; the verdict is the normalised copy's. STRUCTURALLY REQUIRED.
9. **The classifier prints no provenance header,** so the shell wrote one. IMPLEMENTATION CHOICE.
10. **Stdlib-only scripts ran with `-B` and `PYTHONPATH`, cwd the worktree** (the dispatch's rule), not with `-I`. None
    imports tree code. IMPLEMENTATION CHOICE.
11. **`tabulate_qsi_r30_r31.py` is 07a's** with the store tags shifted, TREES, RUNS, a comment and the docstring changed;
    the verbatim reader is untouched. IMPLEMENTATION CHOICE. Its first generation aborted on its own assertion before
    writing. UNINTENDED DRIFT, harmless.
12. **`qsi_summary_r30_r31.py` is 07a's**, tags shifted, plus a new `additions_07b`. IMPLEMENTATION CHOICE.
    - One printed label was missed by the regex and corrected, and the function was extended once. It was re-run each
      time, and only the last output is kept. UNINTENDED DRIFT.
    - Its "record-level S3 form" list uses float arithmetic. The exact list (Fractions) is `replay07b_r31.py rows`'s,
      which found the same 7.
13. **One read-only look at stop 1 printed output that was not saved.** Its fields are in the CSV and the summary.
    UNINTENDED DRIFT.
14. **S3 is scored by 07a's `s3_score.py`**, with the tags shifted. IMPLEMENTATION CHOICE.
15. **The capture harness is R30's on-disk copy.** `prov.py` points at the tree and R31; `worker.py` adds `refusals_07b`;
    `route_07a` is kept, descriptive only. IMPLEMENTATION CHOICE.
16. **`replay07b_r31.py` is generated from R30's driver** (`gen/make_replay07b_r31.py`).
    - The 49 abserr-only rows are replaced by the 8 records of interest.
    - It reports the refusals, the blockcheck compares with R30's captures, and two subcommands capture the records of
      interest.
    - IMPLEMENTATION CHOICE. Its first generation aborted before writing. UNINTENDED DRIFT, harmless.
17. **The replay ran as a registered run** (slug `r31-replay`), by Part A's `run_06j.py` and a new launcher, where R30's
    ran as an unregistered shell pipeline. `run_06j.py` appends each command's output to `out/commands.log`, which is why
    that file is over 2 MB. IMPLEMENTATION CHOICE.
18. **Block pass 1 (the pilot) ran in session**, as R30's did. IMPLEMENTATION CHOICE.
19. **S1, S2 and S4 are R30's scripts, adapted.**
    - S1's keys are renamed to R31's.
    - Each adds the refusals and R30's numbers.
    - S4's integrand is still compared with R29's capture, on which the references were taken.
    - IMPLEMENTATION CHOICE.
20. **R30's `capture_compare.py` was not re-run.** The integrand's equality at 1,001 points carries S4, as in R30.
    IMPLEMENTATION CHOICE.
21. **A read-only `tail -3` loop failed** and was repeated. UNINTENDED DRIFT.
22. **A per-call projection total was printed and not saved.** It is re-derivable from `refs_projection_r31.json`.
    UNINTENDED DRIFT.
23. **The readers ran in session** (about 2 minutes), not registered, as R30's did. IMPLEMENTATION CHOICE.
24. **The descriptive addition's new scripts:**
    - the captures re-run at both codes (`refcap_replay.py`);
    - the projection by 07a's model with its γD term (`refs_projection_r31.py`);
    - the scoring (`understates.py`).
    - IMPLEMENTATION CHOICE.
25. **The launcher's first `r31-refs` edit aborted before writing**; `cmp` shows the file unchanged. UNINTENDED DRIFT,
    harmless.
26. **The references are the tool's own command line,** four readings per process, where Part A read each by
    `one_reading.py` for its monitor. They are the same calls of `reading()`. Part A's split-edge FCC was not needed:
    every segment is monotone and none steep. IMPLEMENTATION CHOICE.

### Verification performed

- **Gate** (`out/00-gate.txt`, `out/s0prime.*`, `out/01-registry-list.txt`, `out/02-stores-before.txt`, `state/lock.log`):
  - the worktree at `0a103b5a`, clean with `--ignored`; the main checkout at `28ebaf07`, clean;
  - `5e19c4b6^` = `7b82a0cd`; `5e19c4b6..0a103b5a` outside docs and prompts: empty;
  - 81 runs, all finished; the lock free; no Ray process; `auth_token` mode 600;
  - R29 and R30 equal to their KEPT hashes.
- **R31** (`out/r31_launch_check.txt`, `out/r31_end_check.txt`, `out/r31_fingerprint_own.txt`):
  - launched 19:18:48Z (sh 10620, driver 10624);
  - `done` at 19:32:27Z; no Traceback, WARNING or ERROR.
- **Stores:**
  - R29, R30 and A3 v2 byte-identical throughout (`state/stores_sha256_final.txt`);
  - R31 unchanged after its build, across the fingerprint, the comparisons, the replay, the readers and the references.
- **The replay** (`out/cmd/r31-replay/`):
  - 452 of 452 records reproduce R31 bit for bit, with Ray uninitialised in all 208 workers;
  - the four blocks reproduce in passes 1 and 2;
  - the 8 captures reproduce R31.
- **The captures for the references** (`out/refs/refcap_replay_{h,a07}.json`): all 8 reproduce R31 at the tree and R30 at
  07a's module, bit for bit.
- **The lock:** ten commands, each exit 0, ending free with nothing listed.
- **Final** (`out/final_check.txt`):
  - 84 runs, all finished; `~/.raypoolkit` empty; no Ray process;
  - both trees clean and unmoved, nothing newer in the worktree;
  - the protected listing differs only by the orchestrator's files under `var/levin-phase-floor-07b/review/stop1/`;
  - no `__pycache__` in the records.

### Observations not acted on

1. **R30 had converged and dishonest group calls in the rehearsal store** (the 7 above; `issues.md` item 1). (h) fixes
   each by the range rule alone. The edge rule refused nothing in any replayed call: no window-edge carrier occurs in the
   rehearsal's calls.
2. **The one new unconverged call** (G+q+r, (1e5, 1e6, 1e6), z 144627.7) looks like
   `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`. It goes 2 → 143 regions to depth 20, with \|Δv\| at
   0.62 e30 (`issues.md` item 2). Not referenced; the user asked about the seven.
3. **`phase_limited` flips False → True on 5,684 records** and never back (`issues.md` item 3).
4. **The readers subsample k and z at random**, so their file counts differ from R30's, as 07a recorded.
5. **`out/commands.log` holds the run driver's appended command outputs** (deviation 17).

### State handed on

- **R31:** `levin-phase-floor-07b-r31-07b-20261116T191849`, `done`, `1b332ffa…`. Its store is kept.
- **The replay run** `levin-phase-floor-07b-r31-replay-20261116T195034` and **the references run**
  `levin-phase-floor-07b-r31-refs-20261116T201120`: both `done`, exit 0.
- **R30 and R29** are unchanged and kept.
- **The lock is released.** Nothing of this agent runs. Both trees are clean and unmoved.
- §6, the census, was not touched (the census agent was dispatched after the references ended).
