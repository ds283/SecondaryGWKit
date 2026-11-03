# Orchestrator — prompt 06j, where 06g's slow leaves come from, and which code is right on V2's rows

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the measurements are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06j-slow-leaves-and-v2-rows.md`](../06j-slow-leaves-and-v2-rows.md)
**Board item:** 06j · **Closes:** nothing. Its records measure
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` and
`[06g-g-plus-r-rows-move-past-both-errors]`; what follows from them is for the user.
**Model:** Opus.

It is a measurement on the Mac Studio only, with no code commit. The machinery is 06f's note §B (an
orchestrator session on the Mac Studio, through `ssh macstudio-tunnel`, dispatching and reviewing),
with 06g's references practice for any reading (the prompt's §1).

**The user dispatches it, in a fresh session, on their word** (2026-11-03: *"don't dispatch it –
this one is getting long, and I will do that in another session"*).

## 0. The order

1. **This prompt's commit is the dispatch SHA.** Push it, with 06g's records commit before it. The
   laptop then makes the worktree `SecondaryGWKit-lpf06j-<short SHA>` on the Mac Studio, detached,
   with no venv. `git diff --stat 1c5336e <dispatch SHA> -- . ':!docs' ':!prompts'` shows only
   `tools/filon_reference.py`, `tools/tests/test_filon_reference.py` and `CLAUDE.md`.
2. **06j runs** on the Mac Studio. One agent; the Mac Studio orchestrator reviews it.
3. **The laptop** copies the records back and makes 06j's records commit.
4. **The user decides** from 06j's results whether a remedy prompt (06k) is written, and what it
   does. Then 07 resumes. Neither starts here.

## B. On the Mac Studio

### B.0 Facts, checked by the orchestrator on the laptop, 2026-11-03

- **06g's census** `levin-phase-floor-06g-census-20261101T114315` is complete, kept in full on the
  Mac Studio (`docs/levin-phase-floor/06g/KEPT_ON_MAC_STUDIO.md`). Its checkpoint is
  `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, with 16 region pickles. 06h's
  checkpoint is `3104c18ed8b119f1798aec75be98d5b39bce641f3b5623b4c1cdc5b18b5ef372`.
- **Part S's inputs exist on the Mac Studio only:**
  - `var/levin-phase-floor-06g/census/out/v8_calls.json` (`1e9a62b0…`);
  - `…/v8_slow_leaves.json` (`b4643033…`);
  - the census's region pickles, whose `checked_leaves_06g` give every checked leaf's $r$,
    $\Delta_1$, $\phi_P$ and span.

  The summary is `docs/levin-phase-floor/06g/census/out/v8_slow_leaves.md`.
- **Part W's rows:** `census/out/v2_rows.csv.gz` lists V2's rows. The call-level set, 62 calls at
  `1e-7` and 80 at `1e-8`, with each call's ratio, regions and depth, is
  `census/out/cmd/51_values_beyond.txt`. Its `G+r` calls move by $7.0$–$7.2\times10^{-33}$ on every
  row, 06h on 2 regions and 06g on 12–16.
- **The tool** at `b5626ecb` integrates a stationary or near-stationary Route B segment directly.
  Its readings on monotone calls are bit-identical to `1c5336e`'s, on both hosts. 06g's references
  measured QAWO's cost at 0.057–0.10 µs per radian (resolution 1, six workers), about 4× that at
  resolution 2. FCC's cost does not grow with the span.
- **The captures** unpickle with the harness's `scripts/` on `PYTHONPATH`, and are checked against
  the census bit for bit by `check_captures.py`.
- **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### B.1 Before you dispatch

1. `ssh macstudio-tunnel`; `runregistrykit list` shows nothing `running` or stale; the main
   checkout is clean; the worktree is at the dispatch SHA and clean, including ignored files.
2. The scratch copy matches its certification by SHA-256, with no `-wal` or `-shm`. Both census
   checkpoints and `v8_calls.json` and `v8_slow_leaves.json` match their SHA-256.
3. `touch` a marker; record the `var/` listing, as 06g's preflight did.

### B.2 Dispatch

One fresh-context agent, on **Opus**, in the worktree. It is given the prompt, the dispatch SHA and
§B.0, and it does not read `orchestrator/`. It writes only under `var/levin-phase-floor-06j/` and,
for registered runs, `var/runs/`. It never touches `var/levin-phase-floor-07/`, 06g's records, the
scratch copy's directory beyond reading the copy, or any store; it makes no commit, and runs no
`git` write.

**Check the agent's plan for readings before it launches them.** Its cost projection must be by
phase span, and its backstop must stop a reading only at a projected day or more, counting the
chunks in flight (06g's two backstop slips, `06g/review.md`, Slips).

### B.3 The review — on the Mac Studio

1. **Scope:**
   - nothing under `var/` newer than the marker outside `var/levin-phase-floor-06j/` and the 06j
     runs;
   - the scratch copy and both checkpoints unchanged by SHA-256;
   - the worktree clean, including ignored files.
2. **J0:** read the captures' bit-for-bit checks. **Re-run one Part S call's replay yourself**, and
   check its value, abserr, regions and one returned leaf's $r$.
3. **The samples:** each sampling rule and seed was written before its first probe or reading, by
   the files' timestamps.
4. **Part S, re-checked:**
   - for three sampled leaves, recompute with your own reader whether the span holds a seam, from
     `production_edges` on the capture;
   - for one leaf, re-run S-b's first two halvings and S-d's split yourself.
5. **Part W, re-checked:**
   - for two W calls, recompute $|v - {\rm ref}|/e$ for both codes from the readings and the
     census checkpoints;
   - check each reading's usable clause.
6. **The predictions:** S1–S4 and W1–W4 copied verbatim into the log before the numbers, by its
   timestamps; each scored against the rule as written.
7. **The records:**
   - README §5.1's headings, with every deviation classified;
   - `results.md` proposes no code;
   - provenance headers;
   - what is kept on the Mac Studio is listed with SHA-256.

Write `var/levin-phase-floor-06j/review.md`, with the verdict in its first line, and report.

### B.4 On the laptop, after the review

1. Copy `var/levin-phase-floor-06j/` back without pickles or files over 2 MB, checking every file
   by SHA-256.
2. Make the records commit:
   - `docs/levin-phase-floor/06j/`, with its `KEPT_ON_MAC_STUDIO.md`;
   - the log in `logs/`, and `VERIFICATION.md` §6j;
   - the board and `docs/OPEN_ISSUES.md`;
   - the CLAUDE.md black list for the copied scripts.
3. Report the results with what each implies for a remedy (06k) and for 07, and ask the user about
   the worktree and anything kept.

## 5. Stop and ask the user

As 06f's note §5, with prompt 06j's §7. **Never touch 07's stashes or saved work.**

## B.5 Addendum, 2026-11-03: the run, the review and the records

**Dispatched** 2026-11-03 on the user's word: *"Dispatch prompt 06j in `prompts/levin-phase-floor`,
following `orchestrator/prompt-06j.md`, and carry out the review of the agent's work when it is
finished."* The laptop pushed `fdc3a2bd` and `dfa19d98`, and checked on both hosts that
`git diff --stat 1c5336e dfa19d98 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` and the tool and
its test. Both registries showed nothing `running` or stale, the lock was free, and the Mac Studio's
main checkout was clean at `28ebaf07`. The laptop fetched there without moving the checkout, made
the worktree `SecondaryGWKit-lpf06j-dfa19d9` detached at `dfa19d98`, and started a Mac Studio
orchestrator session in it.

**The run and the review on the Mac Studio.**
- **§B.1's preflight passed.** The marker was set at 08:51:24 UTC.
- **The run.** One Opus agent was dispatched (`06j/review/dispatch.md`). Its gate and samples took
  08:58–09:06. Run 1, `levin-phase-floor-06j-run1-20261103T091504`, ran 09:15–09:26: 76 of 76,
  exit 0.
- **The readings plan** was checked before launch (§B.2):
  - the projection was by span, cross-checked against a pure-span bound;
  - the backstop stops a reading only at a projected day or more, with the chunks in flight
    counted;
  - the fit's per-segment term is the log's deviation 10.
- **The readings.** `levin-phase-floor-06j-readings-20261103T093537` ran 09:35–12:13: 68 of 68,
  none stopped. The longest was c04 QAWO r2, at 3,538 s.
- **The waits.** Each run was waited on by a background `kill -0` loop on its pid, and the agent was
  resumed after each. There was no §7 stop, and the agent asked nothing.
- **The review** (`docs/levin-phase-floor/06j/review.md`) passed checks 1–6. It re-ran s17's replay
  and three leaves' seams, s07's leaf, its S-b halvings and its S-d split, and f01, f06 and c04's
  ratios, each with its own scripts.
- **It stopped on check 7 alone:** 44 Part W preparation files without provenance, the readings'
  header line naming 06g, and deviation 19 overstating its evidence. None changes a number.

**The results:**
- J0 holds with no replacement.
- **S1–S4 fail.** The slow leaves are rounding that scales with the leaf (N), not seams (J).
- **W1–W4 hold.** 06g is honest on all 22 W calls and 06h is out on all 22, each time carried by a
  pre-asymptotic leaf that 06h's $r < 1$ trusted and that rule (b) refines.

`VERIFICATION.md` §6j has the numbers.

**The user's rulings** (Decisions, **06j results**):
- (a)–(c) are accepted as they stand: *"(a)-(c) are fine and just need to be recorded, but then we
  can move on."* They are the log's deviations 21–23.
- J1 and J3: *"fine"*.
- On J2, the user asked whether the V2 issue should stay open as a hook for a fix. Nothing is left
  to fix: 06g's code is the honest one, and 06h's mechanism belongs to
  `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, already corrected by rule (b). The user: *"J2: close"*.

**The laptop's checks.**
- **The copy.** The records were copied back by `rsync` without pickles or files over 2 MB: 961
  files, each equal to the Mac Studio's by SHA-256. The two run directories' 10 files were copied
  to `var/runs/` and checked the same way.
- **The re-scoring.** S1, S3 and W1–W4 were re-scored from `partS_leaves.csv` and `w_calls.csv` by
  the orchestrator's own reader, with the same verdicts and numbers.
- **The CSV tables.** The three tables were added past the laptop's `*.csv` exclusion.

**Housekeeping, on the user's word** (*"Please remove the worktree on the Mac Studio. If we don't
need the pickles they can also be cleaned up."*):
- The worktree `SecondaryGWKit-lpf06j-dfa19d9` was removed with `git worktree remove`; it was clean
  with `--ignored`.
- The 138 pickles were checked against `KEPT_ON_MAC_STUDIO.txt` (all equal) and deleted. All are
  reproducible: J0 showed every capture reproduces 06g's census bit for bit, from inputs still
  kept.
- The run directories are kept.

**Issues:**
- Closed: `[06g-g-plus-r-rows-move-past-both-errors]`.
- Narrowed: `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` and
  `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`.
- Opened: none.

`docs/OPEN_ISSUES.md` goes from 127 to 126 open.

**For the user:** whether a remedy prompt (06k) is written for the slow leaves, and what it does.
07 resumes after that.
