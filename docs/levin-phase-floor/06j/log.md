# levin-phase-floor 06j — where 06g's slow leaves come from, and which code is right on V2's rows (a measurement)

**Subject:** prompt 06j (`prompts/levin-phase-floor/06j-slow-leaves-and-v2-rows.md`), Part S (the slow
leaves at $r \approx 1/2$) and Part W (V2's rows), on the Mac Studio (mps026298), 2026-11-03.
**Commit:** none; records committed by the orchestrator from the laptop.

## Predictions and verdicts

*Copied in at 2026-11-03T09:05Z, before any probe, reading or sample draw. The record is
`state/predictions_and_rules_20261103T0905Z.md` (read-only, SHA-256 `b41740de…`), which also holds the
sampling rules, the seeds and the operational definitions. Prompt 06j §4, verbatim:*

README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
**undecided** against the rule as written. None is a stop.

**Part S:**
- **S1 — the slow leaves sit on seams.** **Rule:** at least 80 % of the sampled slow leaves hold a
  seam in their span, and the fraction is at least five times the null's.
- **S2 — $r$ stays at 1/2 where the seam is kept.** **Rule:** on at least 80 % of the sampled leaves
  that hold a seam, $r$ stays within $[0.4, 0.6]$ at every one of depths 21–24 along the branch
  holding the seam.
- **S3 — the slow leaves are not noise.** **Rule:** on at least 80 % of the sampled leaves,
  $\Delta_1$ exceeds ten times the larger of S-c's two scatters.
- **S4 — splitting at the seam restores trust.** **Rule:** on at least 80 % of the sampled leaves
  that hold a seam, both sides of the split would be trusted under rule (b).

**Part W:**
- **W1 — the references count.** **Rule:** on every W call, the four readings agree within the
  count tolerance.
- **W2 — 06g is honest on V2's rows.** **Rule:** on every W-5 call, $|v_g - {\rm ref}| \le e_g$.
- **W3 — the parent is the one in error.** **Rule:** on every W-5 call, $|v_{06h} - {\rm ref}| > e_{06h}$.
- **W4 — the pattern is general.** **Rule:** over the W-12 calls, 06g is within its error on at
  least 11, and 06h is outside its error on at least 6.

**The basis of the 80 % and the factors.** S1–S4 are each read as a test of (J) against (N), and
80 % leaves room for a minority of leaves of the other kind. The factor of five over the null and
of ten over the scatter are chosen generously, so that a pass is not a near-miss: README §5's
practice for a cut-off that decides between remedies. Every number behind each verdict is reported,
so that a different reading can be made from the record.

<!-- verdicts are added below this line when scored; the block above is not edited -->

| Prediction | Number scored on | Verdict |
|---|---|---|
| **S1** | 1 of 142 sampled slow leaves hold a seam (0.70 %); null 0.119 % (5.9×) | **fails** |
| **S2** | 1 seam-holding leaf (s07); seam-branch $r$ = 0.494, 0.152, 0.571, 0.078 at depths 21–24 | **fails** (on n = 1) |
| **S3** | $\Delta_1$ > 10 × the larger S-c scatter on 0 of 142 ($\Delta_1$/scatter: median 0.039, max 1.81) | **fails** |
| **S4** | s07: sides at $r$ = 20.4 and 2.42, neither trusted under (b) | **fails** (on n = 1) |
| **W1** | 22 of 22 W calls: four usable readings agreeing within min(e_g, e_06h)/100 (spread/tol 3.1e-8 to 8.2e-3) | **holds** |
| **W2** | 10 of 10 W-5 calls within $e_g$ (worst 0.0463) | **holds** |
| **W3** | 10 of 10 W-5 calls outside $e_{06h}$ (10.6–20.3) | **holds** |
| **W4** | 06g within on 12 of 12 (worst 0.071); 06h outside on 12 of 12 (2.40–18.4) | **holds** |

**Notes on the scoring.**
- **S2 and S4 rest on a single leaf.** Only one sampled slow leaf holds a seam, so each is scored on n = 1 as the
  rule is written. The rules could not anticipate a seam fraction near zero. Both are reported as written, and
  their real content is S1's.
- **Where the numbers are.** `out/results.md` holds every number. The per-leaf table is `out/partS_leaves.csv`,
  the per-call table `out/w_calls.csv` and `out/w_tables.md`.

## Result

- **J0 holds,** with no replacement.
- **Part S:** (J) is not the cause. 141 of 142 slow leaves hold no representation seam, and their $\Delta_1$ is at
  or below the rounding scatter. A descriptive probe added after scoring identifies (N) for half of them: on 75
  leaves, $\Delta_1$ is exactly the difference of the three solves' 06e node errors, which $\phi_P$ declares only
  for the parent.
- **Part W:** 06g is honest on every W call and 06h is outside its error on every one. On every call 06h's error
  sits in a single leaf it trusted at $1/4 \le r < 1$ with $\Delta_1/\phi_P$ between 5.8 and $1.2\times10^7$.

## What shipped

No code. The records are in `var/levin-phase-floor-06j/`:
- `log.md` (this), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt` (138 pickles, 137.6 MB).
- **`state/`:**
  - the predictions, rules and seeds, `predictions_and_rules_20261103T0905Z.md` (read-only, `b41740de…`);
  - the readings plan, `readings_plan_20261103T0935Z.md` (read-only, `49dc3e43…`);
  - the run plans, `plan_run1.json` and `plan_readings.json`, and the launch transcripts;
  - the captures (`capture/`) and the sliced inputs (`inputs/`);
  - 06g's scripts as copied (`scripts_06g_reference/`);
  - the pre-change copies `w_prep_as_run1.py` and `run_06j_as_run1.py`;
  - the before-listings.
- **`scripts/`:** 06g's references scripts, with four changed:
  - `prov.py` and `refs_common.py` (tree and paths);
  - `check_captures.py` (harness check);
  - `one_reading.py` (progress file).

  The diff is in `out/scripts_diff_against_06g.txt`, `out/scripts_diff_one_reading.txt` and
  `out/scripts_diff_run_06j.txt`. 06j's own scripts are draw_06j, replay_06j, slice_06j, seams_06j, probe_s,
  probe_s_nodes, score_s, se_control, w_prep, w_segments, cost_06j, cost_06j_v2, score_w, run_06j, launch_06j and
  dev_probe.
- **`out/`:**
  - every command with its output (`out/cmd/`, `out/commands.log`);
  - the samples (`out/sample/`), the capture checks, the replays and the tool re-read;
  - Part S per call (`out/partS/`), its tables and summaries;
  - the W preparation (`out/w_prep/` from run 1, `out/w_prep_v2/`);
  - the cost projections, the 68 readings (`out/readings/`), the progress files and the monitor log;
  - the W scores and tables, and `out/results.md`.
- **Two registered runs,** both `done` with exit 0:
  - `levin-phase-floor-06j-run1-20261103T091504`: 76 commands, 09:15–09:26Z;
  - `levin-phase-floor-06j-readings-20261103T093537`: 68 readings, 09:35–12:13Z.

## The stages, in order

1. **08:57–09:00Z.** Read the brief, the prompt and its reading list. Copied the scripts. Made the Gate checks
   (`out/cmd/00`–`07b`).
2. **09:04:38Z.** Wrote the predictions, rules and seeds file. At 09:04:52Z, this log with the predictions block.
3. **09:06:41Z.** Drew the samples (`out/cmd/12`). Ran development checks on 06g's capture v02, outside the
   sample (`out/cmd/13`–`19`).
4. **09:15–09:26Z.** Run 1: the captures, the check, slicing, the J0 replays, the tool re-read, Part S's probes and
   the W preparation.
5. **09:27–09:33Z, in session.** J0 comparisons (`out/cmd/40`–`41`). Part S scored (`out/cmd/42`), then the
   descriptive node probe (`43`–`44`). `w_prep` v2 (`45`). Cost projections (`47`–`48`) and the chunks (`49`).
   The plan (`50`–`51`), delivered and checked.
6. **09:35–12:13Z.** The readings run.
7. **12:14–12:20Z, in session.** W scored (`out/cmd/60`). The S-e control redone (`61`). Checks after
   (`62`–`81`). The records.

## Deviations from the prompt

1. **The harness's tree and paths.** `prov.py` and `refs_common.py` now name one tree, `g` (06j's worktree at
   `dfa19d98`), and 06j's records directory, one level shallower than 06g's, so the main-checkout assertion counts
   `parents[2]`. Each change is marked "prompt 06j". **STRUCTURALLY REQUIRED.**
2. **`check_captures.py`'s harness check** compares worker.py's SCRIPTS hashes file by file, every one equal except
   `prov.py` (deviation 1). Both prov.py hashes are recorded. **STRUCTURALLY REQUIRED.**
3. **J0's first clause was checked two ways.** The capture re-ran its unit with the census worker (06g's practice),
   and the captured call was also replayed alone. Both are bit-identical to the census on all 46.
   **IMPLEMENTATION CHOICE.**
4. **Part S's eligibility, fixed in the rules before the draw.** Only lost calls with at least one scoped slow leaf
   are eligible: the 223/227 calls whose untrusted leaves are all $r \ge 1$ have nothing to probe. Only the four
   `G±q±r` groups are used, the `q+r` and `G-r` calls excluded. No two sampled calls share a row.
   **IMPLEMENTATION CHOICE.**
5. **S-b's branch choices.** The prompt does not settle them, so they were fixed in the rules:
   - at a level where both children hold a seam, the one with the larger $\Delta_1$;
   - the off-seam branch follows the larger-$\Delta_1$ child below depth 21;
   - a leaf with no seam gets max and min branches.

   **IMPLEMENTATION CHOICE.**
6. **S-c's scatter.** It is the rms of $V_P$'s change over 16 draws. Nodes move by $k \in \{-4..4\}$ ulp; the
   integrand's inputs move by a hashed $k(x)$, the same at a node in every callable. The halves' and $\Delta_1$'s
   scatters are reported beside it. Fixed in the rules. **IMPLEMENTATION CHOICE.**
7. **S-d's split for a leaf holding several seams** was defined as a split at all of them. It never applied: s07's
   seams are one x with seven kinds. **IMPLEMENTATION CHOICE.**
8. **S-e's control stencil was never taken in run 1.** The rule took the control only if no seam lay within
   3·h_max, with h_max = w/4, of a point w/4 or more from the seam. That window, 0.75w, always held the seam
   itself. **UNINTENDED DRIFT.** It was redone with a per-step window (3h) in `scripts/se_control.py` and written
   to a new file, `out/partS_se_control.json`. Run 1's `out/partS/` is unchanged. The seam stencils recomputed
   there equal run 1's. **IMPLEMENTATION CHOICE.**
9. **06h's leaf values were checked against 06h's value by the code's own arithmetic.** The rules (§5) said
   "exact sum (math.fsum, as _ExactSum.total)", but both codes return the plain floating-point sum of the returned
   leaves in ascending x (`_sum_leaves_ascending`, at 06h's tree and at 06g's). Both readings:
   - **exact sum:** reproduces 06h's value on 20 of 22 calls, c04 and c06 differing in the last bits;
   - **code's arithmetic:** reproduces it on 22 of 22, and 06g's on 22 of 22 (exact sum: 15 of 22).

   The localisation uses the leaf values themselves, so no number in the verdicts depends on which check is used.
   `w_prep.py` v2 checks both (marked "06j v2"; run 1's text is `state/w_prep_as_run1.py`; its output is
   `out/w_prep/`, v2's `out/w_prep_v2/`). **STRUCTURALLY REQUIRED.**
10. **The cost projection has a per-segment term.** The prompt says cost is projected "from each call's phase span
    ..., not from its segment count". The plan's model was wall = a·segments + b·span, fitted to 06g's readings,
    times the largest observed ratio. Its b (0.0571 and 0.233 µs/rad for QAWO r1 and r2) carries the span. Both
    readings, with the actual walls:

    | reading | pure span (0.10 µs/rad r1, 4× at r2) | plan's projection | actual |
    |---|---|---|---|
    | c04 QAWO r1 / r2 | 1419 / 5676 s | 2155 / 6413 s | 830 / 3538 s |
    | c00 QAWO r1 / r2 | 1031 / 4124 s | 1570 / 4663 s | 599 / 2422 s |

    No backstop decision depended on the difference. **IMPLEMENTATION CHOICE** (recorded at the orchestrator's
    request).
11. **The first cost projection was superseded.** `cost_06j.py` (`out/cmd/47`, 105 h) took the largest single
    per-radian rate, from readings dominated by per-segment overhead. It is kept as a record, and `cost_06j_v2.py`
    (`out/cmd/48`) replaced it. **IMPLEMENTATION CHOICE.**
12. **`one_reading.py`: one marked change.** With `--progress-order unordered`, the progress file also lists the
    tool's jobs and the finished chunk indices. No value path changes. Every W reading used 06g's unordered
    progress path (its addendum-4 wrapper, checked bit for bit by 06g); J0's re-read used the plain one.
    **IMPLEMENTATION CHOICE.**
13. **`run_06j.py`'s monitor** (diff in `out/scripts_diff_run_06j.txt`):
    - every reading's progress is weighted by chunk, with weight a·segments + b·span;
    - P_c and P_o are formed as in the plan, with the count-based forms logged beside them;
    - a reading past 30 min or past its projection is reported every 10 min;
    - a reading is stopped only at P_o ≥ 86,400 s.

    **IMPLEMENTATION CHOICE.**
14. **(a) The monitor's "progress unreadable" lines.** The monitor's first check of each reading came in the same
    driver pass that started it, before `one_reading.py` had written its progress file. So `out/reading_monitor.log`
    has one "progress unreadable: [Errno 2]" line per reading, 68 in all, including `127_c04-qawo-r2` at 11:14:06Z.
    - **What happened next:** the monitor kept checking every 30 s, and an error line is logged once per reading.
      The 30-minute reports appeared as planned for the two readings that passed 30 minutes (c00 QAWO r2 at
      11:03:49Z and 11:13:50Z; c04 QAWO r2 at 11:44:16Z, 11:54:21Z and 12:04:27Z).
    - **No decision was affected:** the largest P_o logged was 3,437 s, far from 86,400 s, and no reading was
      stopped. The weighted projections were close to the outcome (c00 r2 P_c 2,424 s against 2,422 s actual; c04
      r2 P_c 3,613 s against 3,538 s); the count-based forms were 3× high.

    **UNINTENDED DRIFT** (cosmetic).
15. **(b) The readings plan's stated correction time is impossible.**
    - **What the file says:** its header (line 2) reads that §7 "was corrected at 09:38Z, before this plan was
      delivered".
    - **What happened:** I wrote the file at about 09:32Z. I corrected its two §7 context lines in place and made it
      read-only at **09:33:15Z**, which is its mtime (`out/cmd/80`), and delivered it in my report a minute or so
      later. The readings run was launched at 09:35:37Z, after the orchestrator's check.
    - **The cause:** "09:38Z" was my wrong estimate of the time, written into the file. Its name's "T0935Z" is
      likewise later than its mtime.
    - The correction was also made in place, not as a new file. The file is not edited now.

    **UNINTENDED DRIFT.**
16. **Gate-stage slips** (each recorded in `out/cmd/`):
    - one `runregistrykit list` ran unrecorded, to /dev/null, before the recorded one;
    - the first `ray-processes` check was piped through `tail`, listed `tail` and exited 1 (`out/cmd/07`); it was
      redone as a command of its own with `>>` (`07b`: none listed, exit 0);
    - a failed `sed` edit left `out/cmd/16` repeating `14`'s development run.

    **UNINTENDED DRIFT.**
17. **Development checks outside the sample.** Before any probe of the sample, `probe_s` and `replay_06j`'s
    functions were tried on 06g's captures v00 and v02, read only from `var/levin-phase-floor-06g/references/state/inputs/`
    (`out/cmd/13`–`19`). They score nothing. **IMPLEMENTATION CHOICE.**
18. **The descriptive node-error probe** (`scripts/probe_s_nodes.py`) was added after S1–S4 were scored. It scores
    nothing. **IMPLEMENTATION CHOICE.**
19. **The log's timestamp.** It was written at 09:04:52Z with the predictions block and extended at the end, so its
    mtime is late. The timestamped record of the predictions is the read-only
    `state/predictions_and_rules_20261103T0905Z.md` (mtime 09:04:38Z). The block above is unchanged: its hash before
    the extension, `273041b2…`, is in `state/log_md_predictions_block_sha256_before_append.txt`.
    **IMPLEMENTATION CHOICE.**
20. **The W-12 rule** excludes, at 1e-8, any (row, index) already picked at 1e-7, and caps picks at two per group
    per rtol. Both are fixed in the rules before the draw. **IMPLEMENTATION CHOICE.**

## Verification performed

`verification.md` has the full text. In brief:
- **The Gate:** all clauses pass (`out/cmd/00`–`07b`).
- **The store and inputs:**
  - the scratch copy is EQUAL to its certification before, after and after the final fingerprint
    (`03`/`04`, `70`/`71`, `73`/`74`), with fingerprint `0e32f24a…` before and after (`05`, `72`);
  - the 06g and 06h checkpoints, `v8_calls.json`, `v8_slow_leaves.json` and `rows.json` match by SHA-256 after
    (`75`);
  - the `var/datastores/` listing is unchanged.
- **No Ray:**
  - every capture worker, replay, probe, w_prep run and reading reports Ray not initialised (`out/cmd/81`, `62`);
  - the lock is free, and no Ray process is listed after (`78`, `79`).
- **The trees:**
  - the worktree is at `dfa19d98`, clean with `--ignored`, before and after (`00`, `76`); no `__pycache__` was
    written anywhere in the records;
  - the main checkout is at `28ebaf0`, clean (`76`);
  - the registry holds 48 runs, all finished, two of them 06j's (`77`).
- **J0:** captures 46/46, replays 46/46, determinism, and the tool re-read (`out/cmd/run1/31`, `33`, `34`, `35`;
  `out/cmd/40`–`41`).
- **The readings:** every one names the tool `d37e633b…`, `b5626ecb`'s by 06g's review. Every one read the pickle
  its `out/inputs.json` entry names, by SHA-256. All 68 are `all_monotone` with no direct segment (`62`, `81`).

## Observations not acted on

1. **A small step in `theta_mod_2pi` at s07's seam.** It is 5e-13 to 1.1e-12 rad and does not shrink with the
   stencil step. The seam is a node of `tau`, `cs_tau` and `friction_F` coinciding with spline knots, at
   $z = 1.59\times10^8$. The control falls to 3e-15. Its effect on the integrand is below the stencil's noise, and
   it was not investigated further.
2. **The second population of slow leaves** (67, $\Delta_1/\phi_P \ge 2$) is not fully explained by the node term.
   Its $\Delta_1$ is well inside the few-ulp input sensitivity, consistent with the abscissa jitter of
   `theta_mod_2pi` 06i described, but that jitter was not measured directly.
3. **On the $\Delta_1/\phi_P$ axis the two kinds of untrusted leaf meet only at the low end.** Part S's slow leaves
   lie at 1.0–25. Part W's wrong 06h leaves lie at 5.8 to $1.2\times10^7$, with c07's at 5.8 the only one below 25.
4. **The five W-5 `G+r` calls share one phase.** They have the same span, segments and phase (345.5 rad), differ in
   their amplitudes, and all have the move in the same 06h leaf.

These are in `issues.md`.

## State handed on

- **Nothing is running.** Both 06j runs are `done`, exit 0. The registry holds 48 runs, all finished. The lock is
  free and no Ray process is listed.
- **Unchanged:** the scratch copy (by SHA-256 and fingerprint), both census checkpoints and every 06g input.
  Nothing was created in `var/datastores/`.
- **The trees:** the worktree is clean with `--ignored`, and the main checkout is clean at `28ebaf0`. 06g's
  worktrees, `var/levin-phase-floor-07-leaves/` and every stash were not touched.
- **Kept on the Mac Studio:** `KEPT_ON_MAC_STUDIO.txt` lists the 138 capture pickles (137.6 MB) with SHA-256 and
  sizes. Both run directories (32 KB each) can be copied whole.
