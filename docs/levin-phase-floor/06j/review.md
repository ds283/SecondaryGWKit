**Verdict: checks 1–6 pass; the review stops on check 7 alone (records), on three findings that change no number and no verdict. J0 holds; S1–S4 fail; W1–W4 hold.**

# levin-phase-floor 06j — the Mac Studio orchestrator's review

Written 2026-11-03 by the Mac Studio orchestrator session (mps026298), worktree
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06j-dfa19d9` at the dispatch SHA `dfa19d98`, against
`prompts/levin-phase-floor/orchestrator/prompt-06j.md` §B.3. Every check's output is in `review/checks/`; the
preflight is `review/preflight/`; the agent's brief is `review/dispatch.md` (SHA-256 `2787672d…`).

## The run

- **§B.1 preflight, 08:46–08:51 UTC.** The registry held 46 runs, all finished. The main checkout was
  `handover-remedial` `28ebaf0`, clean. The worktree was at `dfa19d98`, with `git status --short --ignored` empty.
  `git diff --stat 1c5336e dfa19d98 -- . ':!docs' ':!prompts'` showed only `CLAUDE.md`, `tools/filon_reference.py`
  and its test. `fdc3a2b` is an ancestor of `dfa19d9`, on `origin/handover-remedial`.
  - The scratch copy's five files equal `copy_files_certified.txt` and 06g's preflight, with no journal, `-wal` or
    `-shm`.
  - SHA-256: the 06g checkpoint is `03adcbba…` (16 region pickles, also hashed), 06h's `3104c18e…`,
    `v8_calls.json` `1e9a62b0…` and `v8_slow_leaves.json` `b4643033…`.
  - `~/.raypoolkit/` was empty, and `ray-processes` listed none.
  - Marker `review/dispatch_marker` at 08:51:24.546 UTC. The `var/` listing is in
    `preflight/var_files_before.txt.gz`.
- **Dispatch.** One Opus agent was given `review/dispatch.md`: the prompt, the dispatch SHA, §B.0 verbatim, the
  paths and rules, and a planned stop before any reading.
- **The agent's work.**
  - Gate and samples: 08:58–09:06.
  - Run 1, `levin-phase-floor-06j-run1-20261103T091504` (captures, J0, Part S, W preparation): 09:15–09:26 UTC,
    76/76, exit 0.
  - The planned stop: the readings plan `state/readings_plan_20261103T0935Z.md` (`49dc3e43…`) was checked here
    and passed.
    - The span projection was cross-checked against a pure-span bound: c04 1419/5676 s and c00 1031/4124 s, each
      below the plan's envelope.
    - The backstop stops only at P_o ≥ 86,400 s, with the chunks in flight counted as finished.
    - The fit's per-segment term was asked to be recorded as a deviation; it is deviation 10.
  - The readings run, `levin-phase-floor-06j-readings-20261103T093537`: 09:35–12:13 UTC, 68/68, exit 0, none
    stopped. The longest was c04 QAWO r2 at 3538 s, with largest P_o 3437 s.
  - Both runs were waited on by a background pid wait (`kill -0` only), and the agent was resumed after each.
- **No §7 stop, and no agent question.** The agent asked nothing that needed the user.

## The seven checks

### 1. Scope — passes

- **Under `var/`:** nothing newer than the marker except `var/levin-phase-floor-06j/` and the two 06j run
  directories (`checks/c1_newer.txt`; only the `var/runs` directory's own mtime moved). The full listing differs
  from the preflight only by those two runs' ten files (`checks/c1_var_diff.txt`).
- **The scratch copy:** unchanged by SHA-256 and by `ls` (size and mtime) (`checks/a3v2_copy_after.sha256`,
  `a3v2_copy_ls_after.txt`).
- **The inputs:** both checkpoints, the 16 region pickles, `v8_calls.json` and `v8_slow_leaves.json` are
  unchanged (`checks/inputs_after.sha256`, `census06g_regions_after.sha256`).
- **The trees:** the worktree is at `dfa19d98` with `status --short --ignored` empty, and the main checkout at
  `28ebaf0` is clean (`checks/c1_trees.txt`). The stash list holds only the two old entries.
- **No Ray:** nothing new in `/tmp/ray-sgk` or `/tmp/ray` since the marker (`checks/ray_tmp_newer.txt`), and
  `ray-processes` lists none. `~/.raypoolkit/` is empty. Nothing new is in `var/datastores/`.

### 2. J0 — passes

- **The agent's records:** `out/check_captures.json` has 46/46 `reproduces_census`, with no differing region or
  call key, and only `prov.py` differs in the harness (deviations 1 and 2). `out/replay/` has 46/46. The tool's
  re-read `out/cmd/40` is bit-identical in value_hex, value_30 and imag_30, and `out/cmd/41` shows the
  determinism clause.
- **Re-run by the orchestrator:** s17 (`g-1e-8-1-35480` #2), replayed by my own script
  (`checks/c2_replay_s17.py`, output `.txt`) with nothing wrapped. The census record and region pickle were
  located from the census checkpoint's own unit record (pickle offset, length and SHA-256), not from 06j's
  selection.
  - Value `-0x1.f912b4f680f74p-86`, abserr `0x1.e1172e84c9ff5p-112`, 172 regions and converged False: each
    EQUAL to the census.
  - All 172 regions' start, end, value, depth and convergence ratio are EQUAL.
  - One returned depth-20 leaf, [19.358464120978212, 19.35848015534389], has r `0x1.0249249249249p+4` in both.
  - Ray was not initialised.

### 3. The samples — passes

- **The rules file** `state/predictions_and_rules_20261103T0905Z.md` (read-only, mtime 09:04:38) holds the
  predictions verbatim, both sampling rules, the seeds (20261103; S-c's 2026110301 + 100d) and the W-12 rule. Its
  SHA-256 `b41740de…` was recorded at 09:04:43 and still matches.
- **The samples** `out/sample/{s_sample,w_sample,selection}.json` were written at 09:06:41.
- **The first probe of a sampled leaf** started at 09:25:31 and the first reading at 09:35:37
  (`out/cmd/80_record_mtimes.txt`, confirmed by `ls`). The development probes at 09:11–09:14 used 06g's captures
  v00 and v02, outside the sample (deviation 17), after the draw.

### 4. Part S, re-checked — passes

My own script, written from `_check_leaf` at the tree, is `checks/c4_partS.py` (output `.txt`).
- **S-a, three leaves.** `production_edges` was run on each capture.
  - s07 holds one edge, at 18.886868625123256.
  - s08 holds none; the nearest is 1097.68 widths away.
  - s19 holds none; the nearest is 161.772 widths away.

  All three equal `out/partS_leaves.csv`. Each leaf's census r, read from the region pickle (0.6516, 0.4641,
  0.4991), equals the table's.
- **s07, re-run.**
  - The leaf itself gives Δ1 7.943610669641307e-44, Δ2 5.176e-44, r 0.6515986769570011 and φ_P 9.175e-45: equal
    to the census and the table.
  - S-b's seam branch gives r 0.49378343757064275 at depth 21 and 0.1523809523809524 at depth 22.
  - S-d's split gives r 20.363636… on the left side (width 1.39e-6) and 2.4197530… on the right (1.41e-5),
    neither trusted.

  All equal the table to the last digit.

### 5. Part W, re-checked — passes

My own script is `checks/c5_partW.py` (standard library, Decimal), output `.txt`. It read the readings directly
and 06g's and 06h's values and abserrs directly from the two census checkpoints.
- **f01** (W-5, `g-1e-7-0-12959` #1) and **f06** (its 1e-8 call, sharing f01's reference): |v_g − ref|/e_g is
  0.0203 and |v_06h − ref|/e_06h is 13.0504 against each of the four readings. The four readings' spread is
  4.72e-43, against a tolerance of 5.47e-36.
- **c04** (W-12, `g-1e-7-1-35052` #6): 0.0154 and 5.3006. The spread is 1.73e-38 against 2.27e-35 (ratio
  7.6e-4).
- **The usable clause:** every one of the twelve readings has route B, `all_monotone` True, 0 direct segments and
  no non-monotone segment, so all are usable.
- **The agent's table:** all of these equal `out/w_calls.csv`.

### 6. The predictions — passes

- **Verbatim:** S1–S4 and W1–W4, with the basis paragraph, are verbatim in `log.md` and in the rules file. Each
  was diffed against prompt 06j §4 with no difference.
- **Before the numbers:** `log.md` was born at 09:04:52, before the samples and every number. The read-only rules
  file at 09:04:38 is the timestamped record. But see check 7(c) on deviation 19's supporting file.
- **Re-scored** from `out/partS_leaves.csv` and `out/w_calls.csv` with my own reader (`checks/c6_rescore.txt`):
  - **S1 fails:** 1 of 142 sampled leaves hold a seam (0.70 %, null 0.119 %; the rule needs 80 % and 5× the
    null).
  - **S2 fails:** on n = 1, s07's seam-branch r is 0.494, 0.152, 0.571 and 0.078 at depths 21–24, so it does not
    stay within [0.4, 0.6].
  - **S3 fails:** Δ1 > 10 × the larger scatter on 0 of 142 (median 0.0398, max 1.81). This is robust: scaling
    the ±4-ulp scatter down to a half-ulp would still leave far fewer than 80 % above 10.
  - **S4 fails:** on n = 1, neither side of s07's split is trusted.
  - **W1 holds:** 22/22 calls have four usable readings within tolerance (max spread/tol 8.19e-3).
  - **W2 holds:** 10/10 (worst 0.0463).
  - **W3 holds:** 10/10 (10.61–20.28).
  - **W4 holds:** 06g is within on 12/12 (worst 0.0708) and 06h outside on 12/12 (2.40–18.40).

  Every verdict is scored against the rule as written; S2 and S4 rest on n = 1, which the log states.

### 7. The records — stops, on three findings

**What passes:**
- `log.md` has every README §5.1 heading. Its 20 deviations are numbered and classified.
- `out/results.md` proposes no code. It says so in its header, and its two remedy sections state only what a
  remedy would have to do.
- `verification.md` and `issues.md` (J1–J3) are present.
- `KEPT_ON_MAC_STUDIO.txt` lists 138 pickles (137,574,385 bytes). Every SHA-256 and size was re-checked with 0
  mismatches, and it is complete: every pickle and every file over 2 MB under `var/levin-phase-floor-06j/`
  outside `review/` (`checks/kept_listed.sha256`, `kept_found.txt`). No `__pycache__` was written.
- **Provenance** meets README §5 rule 5 through the command record, as 06f's review read it. Every script prints
  a header (its SHA-256, the tree's HEAD and status, its inputs' SHA-256) into its `out/cmd/` transcript. The
  headerless JSON tables have their SHA-256 printed there and still match: `capture.json` `9134fb08…`,
  `check_captures.json` `65e2225a…`, `inputs.json` `9cbbf131…` and `selection.json` `ae39b182…`.

**The findings.** None changes a number or a verdict. Each is an unrecorded deviation or an inaccurate record. By
the orchestrator's rules they are reported, not repaired.
- **(a) 44 per-call Part W preparation files carry no provenance.** These are `out/w_prep/*.json` and
  `out/w_prep_v2/*.json`. They have no header, and their SHA-256 is printed nowhere. Their runs' transcripts
  (`out/cmd/run1/…w_prep…`, `out/cmd/45`) give the script's hash, the tree and the inputs, and only
  `summary.json` is hashed. This is not in the log's deviations.
- **(b) The 68 readings' header names the wrong prompt.** Its first line reads "levin-phase-floor 06g: Filon
  reading …". 06g's label was kept in the copied `one_reading.py`, whose marked 06j change did not touch it. The
  rest of each header is correct: the 06j script's hash (`3cf12125…`), the tree `dfa19d98`, the input pickle and
  the tool `d37e633b…`. This is not in the log's deviations.
- **(c) Deviation 19 overstates its evidence.** It says the predictions block's hash "before the extension" is in
  `state/log_md_predictions_block_sha256_before_append.txt`. But that file was written at 12:19:03. It holds a
  hash of the whole `log.md` (its line names `log.md`) taken just before the final append, so it is no evidence
  of the 09:05 content. Check 6 rests instead on three things: the log's birth time, the read-only rules file
  with its 09:04:43 hash, and the verbatim diff.

**For the user:** whether to accept (a)–(c) as they stand, to be recorded as deviations in the records commit, as
in 05's and 06b's reviews.

## Slips already recorded by the agent

These were checked against the files and are classified in the log:
- the plan file's impossible "09:38Z" correction time, against its mtime of 09:33:15Z, and its in-place edit
  (deviation 15);
- the 68 "progress unreadable" monitor lines (deviation 14), which affected no backstop decision;
- the S-e control stencil never applied in run 1, redone in a new file (deviation 8);
- the gate-stage slips (deviation 16).

## Kept on the Mac Studio

- **The pickles:** `var/levin-phase-floor-06j/KEPT_ON_MAC_STUDIO.txt` (SHA-256 `6cd7b72a…`) lists them: 138
  pickles under `state/capture/` and `state/inputs/`, 137.6 MB.
- **The run directories** `var/runs/levin-phase-floor-06j-run1-20261103T091504/` and
  `…-readings-20261103T093537/` are 32 KB each, and copyable whole.
- **The orchestrator's `review/`** holds `preflight/`, `checks/` (including `var_files_after.txt.gz`, 0.4 MB compressed from 9.9 MB, and my
  check scripts with their outputs), `dispatch.md` and `dispatch_marker`.
- **The worktree** `SecondaryGWKit-lpf06j-dfa19d9` is clean, kept until the user says. 06g's four worktrees were
  not touched.
