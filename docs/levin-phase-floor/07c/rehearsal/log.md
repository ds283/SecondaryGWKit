## Part C §5: rehearsal R32 on the Mac Studio

**Prompt:** [`07c-chase-below-a-refused-leaf.md`](../07c-chase-below-a-refused-leaf.md) §5 ("as 07b §5", which is "as 07a
§5"), with §8 and §9 (§9.5's S3 item, §9.7). It ran under Part C's dispatch (`cc3368ce`). The user's word at stop 1:
*"Yes, continue the rehearsal agent"*.

**What was measured:** the code commit `67b5af74`, variant (S).
- **The rule:** below a leaf a rate rule refused, a root stops once its chase has made `_LEVIN_CHASE_STOP_LIMIT = 6`
  noise-level checks within `_LEVIN_CHASE_NOISE_FACTOR = 4` of the yardstick.
- **The parent:** the code commit's parent is `cff4237f`. The parent rehearsal is R31, built at `0a103b5a`, whose code
  is `5e19c4b6`'s.

**Host and trees:** mps026298.
- Every build, replay, score and reader ran in the worktree `SecondaryGWKit-lpf07c-cc3368c`, detached at `cc3368ce`.
  - `levin_quadrature.py` is `d141a80b`.
  - These four are R31's tree's: `QuadSourceIntegral.py` `33cb161e`, the launcher `b0135218`, `main.py` `a5d47f5f`,
    `store_value_diff.py` `7997af85`.
- Nothing was written in the worktree. `status --short --ignored` was empty at every check.
- The main checkout stayed at `28ebaf07`, clean.

**Model:** Claude Opus 5.5, the rehearsal subagent, in three turns:
1. the gate and R32's launch;
2. the end check, the comparisons, and stop 1;
3. S3, the replay, the S3 record, S1, S2, S4, the readers, the release, and these records.

**Date:** 2026-11-19, 14:01–14:52 UTC. **Commit:** none; the laptop commits these records.

**Result:** COMPLETE. Every check of §5 holds.
- **R32 is `done` with `1b332ffa…`, and S0′ holds.**
- **Outside `QuadSourceIntegral` every value table equals R31's** (360,462 rows; 0 unclassified).
- **Inside it, 305 of 864 rows moved.**
  - `total_converged` went up in 1 row and down in none. The one is §9.5's named row.
  - Summed regions went 152,117 → 144,531 (0.950). No row's regions rose.
- **S3 holds.** 0 violations; the largest ratio is 5.25e-6.
- **§9.5's S3 prediction holds.** R31's unconverged record (143 regions) converges in R32:
  - at 86 regions and depth 16;
  - equal bit for bit to Part A's (S) replay record;
  - honest at **0.05331** of its abserr against Part A's `ref-R31-3-20363-g1`.
- **S1, S2 and S4 hold.**
- **The readers** exit 0 with the store unchanged. **The lock is released.**

### What shipped

No code. The records under `var/levin-phase-floor-07c/rehearsal/` (committed by the laptop as
`docs/levin-phase-floor/07c/rehearsal/`): `log.md`, `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and:
- **`scripts/`:**
  - adapted from 07b's rehearsal (each with a diff in `out/`):
    - `rec.sh` (`43c59f3b`) and `protected_listing.sh` (`b4d11baa`);
    - `normalise_jobname_labels.py` (`93a0b27e`);
    - `tabulate_qsi_r31_r32.py` (`8e0c91a7`) and `qsi_summary_r31_r32.py` (`907b9950`);
    - `s3_score.py` (`8355d776`);
    - `s1_score_r32.py` (`8609e7fb`), `s2_score_r32.py` (`4a655c7b`) and `block_score_r32.py` (`6debd7ad`);
    - `run_block_reference_r32.sh` (`6db5cd4f`) and `launch_r32.py` (`145c59c8`);
  - unchanged copies:
    - `classify_non_qsi.py` (`26690e61`, 05's);
    - `block_reference_r30.py` (`efac125e`);
    - `run_readers_worktree.sh` (`ec96cf09`);
    - `run_06j.py` (`6227aea4`, 07b Part A's plan driver);
  - new:
    - `s0prime.py` (`17f73ca4`);
    - `s3_record_r32.py` (`4330f394`), §9.5's S3 prediction;
    - `phase_limited_flips.py` (`90a00f22`);
  - **`gen/`**, the generators that made the adapted scripts:
    - `make_tabulate_qsi_r31_r32.py` (`e8be0f00`), `make_qsi_summary_r31_r32.py` (`7190ccf5`) and `make_s3_score.py`
      (`96954584`);
    - `make_scores_07c.py` (`96976b83`) and `make_replay07c_r32.py` (`8cf49456`);
    - `edit_harness_07c.py` (`3a368bbb`);
  - **`capture/`**, 07b's rehearsal's on-disk harness (hashes in `state/capture_harness_r31_sha256.txt`):
    - `prov.py` (`6b65f1a7`) and `worker.py` (`a5fdb18a`), changed where marked "prompt 07c r32";
    - `replay07c_r32.py` (`a4addf1e`), generated from 07b's `replay07b_r31.py`.
- **`out/`:**
  - `commands.log`;
  - the gate, launch, end, stop-1, readers and final checks;
  - `r32_status.json`, `r32-replay_status.json` and `r32_fingerprint_own.txt`;
  - the classifier outputs, the QSI tabulation, its CSVs and its summary, and `phase_limited_flips.*`;
  - `s0prime.*`, `s1_scores.*`–`s3_scores.*`, `s3_record.*` and `s3_record/`, `block_scores.*`, `block_reference/`;
  - the replay's outputs: `cmd/r32-replay/`, `replay_calls.csv`, `replay_calls.checks.json`, `blocks.json`, `refcap.json`;
  - `readers.txt`, `stop1_unconverged_look.txt` and `unconverged_calls_look.txt`.
- **`state/`:** `lock.log`, `deviations_draft.md`, the launch lines, markers and command files, every store-hash file,
  the protected paths, the replay's rows, specs and plan, and `stores/r32/` (symbolic links).
- **Not for copying back:** see `KEPT_ON_MAC_STUDIO.txt`. It lists the two diff JSONs, the protected listings,
  `state/capture/`, `state/inputs/` and `extract/`.
- **The store is kept:** `var/datastores/levin-phase-floor-07c-r32*`.

### Predictions and verdicts

| Rule (prompt 07c §5, as 07b §5) | Scored on | Verdict |
|---|---|---|
| 1. S0′: `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists nothing the build imports | `CLAUDE.md` only (the `black` paragraph); no `.py`, `.toml`, `requirements.txt` or `config/` | **holds** |
| R32 with R31's command line (slug `r32-07c`, `--prompt 07c`, `--job-name levin-phase-floor-07c`, `--cpus 6`) gives `1b332ffa…` | `done`, exit 0, `1b332ffa…` in `status.json` and again by `runregistrykit store fingerprint` (read only); `classes` equal R31's; about 13.8 min | **holds** |
| 2. The lock: `acquire`, `set-pid` at the launch, `release` after the readers, each a shell command of its own | acquired 14:03:58Z; `set-pid` 19463; released 14:49:20Z after `ray-processes` listed nothing; ten commands, `>>` (deviation 4) | **holds** |
| 3. Outside `QuadSourceIntegral`: every value table identical; every other difference of the four volatile kinds; no row in one store only | 360,462 value-table rows identical; 0 unclassified on the normalised copy (timing, `label` 6,012, `parents_digest` 291, `rho_reuse` 26) | **holds** |
| 3. Inside it: every moved row tabulated, with the rows whose `total_converged` changes in each direction | 305 rows (`out/qsi_rows_r31_r32.csv`); 802 Levin records; `total_converged` 1 up, 0 down | tabulated |
| 4. S3: \|t32 − t31\| ≤ e31 + e32 on every row with finite `total_abserr` in both, or one §9 predicts to move further | 864 scored exactly (Fractions); 0 violations; largest 5.25e-6; the named row 7.0e-8, within the bound | **holds** |
| §9.5's S3 item: R31's record (G+q+r, (1e5, 1e6, 1e6), z 144627.7) converges at 86 regions, honest at 0.053 of its abserr | converged, 86 regions, depth 16; equal to Part A's (S) record; 0.05331484572900594 against `ref-R31-3-20363-g1` (spread 1.1e-6 of abserr, scored); `total_converged` false → true | **holds** |
| 5. S1: V1's twelve locate z ≈ 10211 and are honest against 06g's references | 12 of 12 locate (within 8.52e-6; spacing ≥ 31.0); 12 of 12 honest, largest 0.0670 (w06); 8 equal R31, 4 moved (one chase stop each) | **holds** |
| 5. S2: P1's two rows locate within one spacing and are honest against `p1_references` | \|x − x_s\| 7.1e-9 and 5.6e-9 against 4.12e-3; ratios 0.0202 (= R31) and 0.111 (R31 0.111) | **holds** |
| 5. S4: R29's four blocks each locate their root, guard no leaf, and are honest against R29's block references | 4 of 4: root bit-equal, 0 guarded, 4.37e-5 to 6.17e-4; each equal to R31 and R28 bit for bit | **holds** |
| 6. The readers on R32 exit 0, store byte-identical; lock released, `ray-processes` empty; R31 and R32 kept | 6 of 6; identical; free; nothing listed; R31 equal to its KEPT hashes | **holds** |

**§9.5's S3 item, in detail** (`out/s3_record.txt`, `out/s3_record.json`):

| | R31 (store) | R32 (store) | R32's call (replay at the tree) | Part A's (S) record |
|---|---|---|---|---|
| value | `0x1.44ad9d596748dp-62` | `0x1.44ad9d59665e8p-62` | `0x1.263e9f583ee54p-79` | `0x1.263e9f583ee54p-79` |
| abserr | `0x1.74349fae30cb8p-87` | `0x1.741c57c6295e9p-87` | `0x1.513b2851b5963p-104` | `0x1.513b2851b5963p-104` |
| converged | False | True | True | True |
| regions / max depth | 143 / — | 86 / — | 86 / 16 | 86 / 16 |
| refusals / chase | — | — | range 4, edge 0 / 2 stops, 28 stopped leaves, 12 noise bisections | — |

- Store values are (1 + z) times the call's.
- **The integrand is the reference's.** `block_reference_r30.py`, run on R32's capture and on R31's capture (07b's
  rehearsal, `df2d478f…`, the one Part A read), gives:
  - 1,001 samples bit-identical, with x_span and kw equal;
  - R31's capture, replayed at the tree, reproduces R32's call.
- **The reference** is Part A's split FCC r2, 1.9015112786324759e-24. Its spread over four readings is 7.24e-38, 1.1e-6 of
  the abserr, so the call is scored.
- **The verdict:** \|v − ref\| / abserr = 0.05331484572900594, the same as Part A's (S) ratio. **Honest.**

**The inside-QSI tabulation** (`out/qsi_tabulation_r31_r32.txt`, `out/qsi_summary_r31_r32.txt`):
- **Rows:** 559 unmoved, 305 moved. `total` moved in 251, `total_abserr` in 252, `analytic_rad` in 106.
- **Largest changes:**
  - \|Δtotal\| is at most 1.050e-5 of R31's abserr and of R32's, at (1e6, 1e5, 1e6), z 437173.17. The median over the
    251 is 1.7e-7.
  - \|Δanalytic_rad\| is at most 6.48e-4 of `analytic.abserr`.
  - **Group records:** 477 values moved; \|Δv\| is at most 1.67e-3 of either abserr; none is beyond e31 + e32.
- **Regions:**
  - Over the moved rows, regions fell in 233, rose in 0, and were equal in 72. The per-row ratio where changed runs
    0.634–0.998, median 0.939.
  - Maximum depth changed in 78 rows. The rows at depth 20 went from 7 to 5.
  - Group-record regions over the differing records went 45,439 → 37,853, and analytic-block regions 3,963 → 3,367.
  - QuadSourceIntegral's stage took 11 m 19 s against R31's 11 m 28.6 s.
- **Convergence:**
  - `total_converged` went 861 → 862: (1e5, 1e6, 1e6), z 144627.7, the named row.
  - Unconverged group calls went 3 → 2. The two rows still unconverged were unconverged at R31 too (`issues.md` item 1).
  - No analytic block's `converged` changed.
- **Location:** unchanged. Group records: 44 located points, 562 window pieces, 0 guarded regions. Analytic blocks:
  400 / 2,515 / 0. 202 rows hold a located point.
- **`phase_limited`:** two group records flipped False → True, and none flipped back. They are tabulated below, at the
  orchestrator's note; descriptive (`out/phase_limited_flips.txt`).

  | (k, q, r; z) | record | label | phase_limited | converged | regions | \|Δv\|/e31 | \|Δv\|/e32 | \|Δv\|/(e31+e32) |
  |---|---|---|---|---|---|---|---|---|
  | (1e5, 3.162e5, 3.162e5; 755.687) | `subintervals[1].groups[1]` | G+q−r | False → True | True → True | 38 → 36 | 1.09e-7 | 1.09e-7 | 5.47e-8 |
  | (1e5, 3.162e5, 3.162e5; 755.687) | `subintervals[1].groups[2]` | G−q+r | False → True | True → True | 38 → 36 | 1.09e-7 | 1.09e-7 | 5.47e-8 |

  Both records have the same value, `0x1.5535a1169116bp-67` → `0x1.5535a116911a6p-67`, and the same abserr, because
  q = r makes the two integrands one. abserr32/abserr31 is 1.000000. Neither has a located point.

**The replay** (`out/cmd/r32-replay/`; run `levin-phase-floor-07c-r32-replay-20261119T143558`, 8.5 min):
- **Reproduction:**
  - 445 of 445 records reproduce R32 bit for bit, with Ray uninitialised in all 203 workers.
  - The four blocks reproduce in passes 1 and 2.
  - The capture of the record of interest reproduces R32.
- **Over the 4,180 replayed calls (932 group, 3,248 analytic):**
  - Range-rule refusals: 567 in 359 group calls, and 6,517 in 2,587 analytic calls. The edge rule refuses nothing.
  - **Chase stops:** 180 in 164 group calls, with 4,061 stopped leaves and 1,589 noise bisections. 225 in 225 analytic
    calls, with 1,539 stopped leaves and 8,888 noise bisections.
  - **No call has a chase stop without a rate-rule refusal.** Checked call by call over all 4,180 calls; 0 found
    (`out/chase_without_refusal_look.txt`).

### Deviations from the prompt

1. **`rec.sh` is 07b's** (`9a80a313`) with its record path changed by sed. IMPLEMENTATION CHOICE.
2. **S0′ is scored by a new `s0prime.py`.**
   - "Nothing the build imports" is read as: no listed `*.py`, `*.toml`, `requirements.txt` or `config/` path.
   - The raw `git diff --stat` is also in `out/00-gate.txt`.
   - IMPLEMENTATION CHOICE.
3. **The lock commands ran from the worktree with `PYTHONPATH` set**, each with an `echo` header (a builtin) and output
   `>>` into `state/lock.log`, as 07b's did. IMPLEMENTATION CHOICE.
4. **The `acquire` ran in the same zsh command as an earlier heredoc `cat >> out/commands.log`.**
   - The `cat` had exited before the `acquire` started.
   - Strictly, the acquire was not a lock command in a shell of its own.
   - It acquired normally. UNINTENDED DRIFT, harmless.
5. **A listing of every protected file** (mtime, size, path) was taken before, at stop 1 and at the end.
   - Its paths include `var/levin-phase-floor-07c/review/`, the orchestrator's directory. Its 13 changed lines were
     expected.
   - IMPLEMENTATION CHOICE.
6. **The R32 launch line is 07b's form, run in zsh.** The purpose was held in a shell variable and passed as `sh`'s `$0`;
   its text is also in `state/purpose_r32.txt`. IMPLEMENTATION CHOICE.
7. **The fingerprint was also re-derived** by `runregistrykit store fingerprint` without `--write`. Read only; the store
   re-hashed equal. IMPLEMENTATION CHOICE.
8. **`normalise_jobname_labels.py` is 07b's**, with its prefixes (07a- → 07b-, 07b- → 07c-) and docstring changed.
   IMPLEMENTATION CHOICE.
9. **The classifier was run on the raw and the normalised JSON.**
   - The raw gives 5,148 UNCLASSIFIED, every one a `label` whose job-name prefix differs.
   - The verdict is the normalised copy's.
   - STRUCTURALLY REQUIRED.
10. **The classifier prints no provenance header,** so the shell wrote one. IMPLEMENTATION CHOICE.
11. **Stdlib-only scripts ran with `-B` and `PYTHONPATH`, cwd the worktree, not with `-I`.** The generators ran with
    `-I`. None imports tree code. IMPLEMENTATION CHOICE.
12. **`tabulate_qsi_r31_r32.py` is generated from 07b's** by `gen/make_tabulate_qsi_r31_r32.py`.
    - It changes 72 store tags, TREES, RUNS, the DEPTH_MAX comment and the docstring. The verbatim reader is asserted
      unchanged. IMPLEMENTATION CHOICE.
    - The generator's first run aborted on its own assertion (a wrong line index) before writing anything. It was
      corrected by two asserted replacements and re-run. UNINTENDED DRIFT, harmless.
13. **`qsi_summary_r31_r32.py` is generated from 07b's** (161 tag shifts, the tree path, the docstring).
    - `additions_07b` is kept under its name, and nothing new was added.
    - Its "record-level S3 form" uses float arithmetic and is descriptive only; S3 is the exact `s3_score.py`.
    - IMPLEMENTATION CHOICE.
14. **A read-only look at the unconverged rows was saved** (`out/stop1_unconverged_look.txt`), not printed and lost.
    IMPLEMENTATION CHOICE.
15. **The stage timings of R31 and R32 were diffed** into `out/stage_timings_r31_r32.diff`. IMPLEMENTATION CHOICE.
16. **S3 is scored by 07b's `s3_score.py`,** generated with the tags shifted.
    - §9.5's named row is listed whether or not it is beyond the bound, and is not counted as a violation. It is within
      the bound anyway.
    - IMPLEMENTATION CHOICE.
17. **The capture harness is 07b's on-disk copy.**
    - `prov.py` points at the tree and R32, and `worker.py` adds `chase_07c`.
    - `replay07c_r32.py` is generated from 07b's driver (79 tag shifts).
    - Its records of interest follow 07b's rule. There is 1 record: the converged change.
    - IMPLEMENTATION CHOICE.
18. **`make_s3_score.py`'s first output shifted the reader module's name twice** (`tabulate_qsi_r32_r32`). It was caught
    in the diff before any run; the generator was corrected and re-run. UNINTENDED DRIFT, harmless.
19. **The replay ran as a registered run** (slug `r32-replay`), by 07b's `run_06j.py` and a new `launch_r32.py`, as 07b's
    `r31-replay` did. `run_06j.py` appends each command's output to `out/commands.log`. IMPLEMENTATION CHOICE.
20. **Block pass 1 (the pilot) ran in session**, 56.6 s, as 07b's did. IMPLEMENTATION CHOICE.
21. **§9.5's S3 record is scored against 07c Part A's `ref-R31-3-20363-g1`.**
    - These are Part A's readings of 07b's R31 capture of that call.
    - It is not a reference in 07b's rehearsal: 07b captured the call but read no reference for it.
    - No new reference was made.
    - IMPLEMENTATION CHOICE, accepted by the orchestrator.
22. **The two `phase_limited` flips are tabulated by a new `phase_limited_flips.py`**, at the orchestrator's note.
    IMPLEMENTATION CHOICE.
23. **S1, S2 and S4 are 07b's scripts, generated with the tags shifted** (`gen/make_scores_07c.py`).
    - The parent's numbers are read from 07b's score files.
    - R32's block serials are used (0-10172, 1-20068).
    - Each call's `chase_07c` is recorded.
    - S4's integrand is still compared with R29's capture, on which the references were taken.
    - IMPLEMENTATION CHOICE.
24. **The S3 record's integrand is compared by samples, not by blob.**
    - The plan was byte equality of R32's and R31's captures.
    - The blobs differ in the unit name (it includes the store serial), the worktree's path and the per-process `str`
      hashes. 07a's and 07b's block captures differ from their parents' in the same way (`issues.md` item 2).
    - So `block_reference_r30.py`, unchanged and without `--reference`, was run on both captures. Its 1,001 samples,
      x_span and kw were compared. The byte comparison is still reported, as False.
    - STRUCTURALLY REQUIRED.
25. **`block_reference_r30.py` replayed 07b's R31 capture at this tree.** It read the capture only, and wrote its output
    under `out/s3_record/`. IMPLEMENTATION CHOICE.
26. **The S3 record's ratio is evaluated in exact Fractions** of the double and the decimal readings, where Part A used
    mpmath at 40 digits. Both give 0.05331484572900594. IMPLEMENTATION CHOICE.
27. **The readers ran in session** (136.6 s), not registered, as 07b's did. IMPLEMENTATION CHOICE.
28. **Two read-only looks at the replayed calls were saved:** the unconverged calls (`out/unconverged_calls_look.txt`),
    and the calls with a chase stop but no rate-rule refusal (`out/chase_without_refusal_look.txt`; none).
    IMPLEMENTATION CHOICE.

### Verification performed

- **Gate** (`out/00-gate.txt`, `out/s0prime.*`, `out/01-registry-list.txt`, `out/02-stores-before.txt`, `state/lock.log`):
  - the worktree at `cc3368ce`, clean with `--ignored`; the main checkout at `28ebaf07`, clean;
  - `67b5af74^` = `cff4237f`;
  - `67b5af74..cc3368ce` outside docs and prompts is empty;
  - `0a103b5a..cc3368ce` is empty under every path the build imports outside `AdaptiveLevin/`;
  - 92 runs, all finished; the lock free; no Ray process; `auth_token` mode 600;
  - R29, R30 and R31 equal to 07b's final hashes, and R31 equal to its KEPT list.
- **R32** (`out/r32_launch_check.txt`, `out/r32_end_check.txt`, `out/r32_fingerprint_own.txt`):
  - launched 14:04:16Z (sh 19451, driver 19463);
  - `done` by 14:18:18Z; no Traceback, WARNING or ERROR.
- **Stores:**
  - R29, R30, R31, R31's run directory and A3 v2 byte-identical throughout (`state/stores_sha256_final.txt`);
  - R32 unchanged after its build, across the fingerprint, the comparisons, the replay, the scoring and the readers.
- **The replay:** 445 of 445 records reproduce R32 bit for bit; the blocks and the record capture reproduce.
- **The lock:** ten commands, each exit 0, ending free with nothing listed.
- **Final** (`out/final_check.txt`):
  - 94 runs, all finished; `~/.raypoolkit` empty; no Ray process;
  - both trees clean and unmoved, nothing newer in the worktree;
  - the protected listing differs only under the orchestrator's `var/levin-phase-floor-07c/review/`;
  - no `__pycache__` in the records.

### Observations not acted on

1. **Two rows stay unconverged at depth 20 in both stores** (`issues.md` item 1).
   - One of them, (3.162e5)³ at z 573.11, G−q−r, was replayed: 2 range refusals, 0 chase stops, 5 noise bisections.
     That reads as one check short of the stop.
   - Its bearing is on §6's Y9. The census's instrument decides.
2. **Capture blobs are not byte-reproducible across trees** (`issues.md` item 2).
3. **The two `phase_limited` flips** are tabulated above. Both records stay converged, and they moved by 5.5e-8 of
   e31 + e32.
4. **The readers subsample k and z at random**, so their file counts differ from R31's (3,122 against 3,189), as 07a
   and 07b recorded. They print the same six shard-count warnings and two `MatplotlibDeprecationWarning`s as 07b's.
5. **`out/commands.log` holds the plan driver's appended command outputs** (deviation 19).

### State handed on

- **R32:** `levin-phase-floor-07c-r32-07c-20261119T140417`, `done`, `1b332ffa…`. Its store is kept.
- **The replay run:** `levin-phase-floor-07c-r32-replay-20261119T143558`, `done`, exit 0.
- **R31, R30 and R29** are unchanged and kept.
- **The lock is released.** Nothing of this agent runs. Both trees are clean and unmoved.
- §6, the census, was not touched.
