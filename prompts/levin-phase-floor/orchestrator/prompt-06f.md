# Orchestrator — prompt 06f, how two Clenshaw–Curtis floor terms add up across a call's leaves

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the measurements are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06f-cc-floor-terms-across-leaves.md`](../06f-cc-floor-terms-across-leaves.md)
**Board item:** 06f · **Closes:** nothing. Its records narrow `[06e-node-errors-add-in-absolute-value]`
and `[06e-cc-declared-phase-term-is-a-crude-proxy]`; what follows from them is for the user.
**Model:** Opus.

It is a measurement on the Mac Studio only, with no code commit. The machinery is 06e's note §B
(an orchestrator session on the Mac Studio, through `ssh macstudio-tunnel`, dispatching and
reviewing), without the rehearsals and without the lock.

## 0. The order

1. **This prompt's commit is the dispatch SHA.** Push it; the laptop then makes the worktree
   `SecondaryGWKit-lpf06f-<short SHA>` on the Mac Studio, detached, with no venv. Its code is
   `f4d8e17`'s: `git diff --stat f4d8e17 <dispatch SHA> -- . ':!docs' ':!prompts'` shows only
   `CLAUDE.md`.
2. **06f runs** on the Mac Studio. One agent; the Mac Studio orchestrator reviews it.
3. **The laptop** copies the records back and makes 06f's records commit.
4. **06g is written** after the user has read 06f's results (held, on the user's word), then **07
   resumes**. Neither starts here.

## B. On the Mac Studio

### B.0 Facts, checked by the orchestrator on the laptop, 2026-10-31

- **06e's census** `levin-phase-floor-06e-census-20261030T081318` is complete; its checkpoint is
  `a0dc4d7f…`, kept in full on the Mac Studio (`docs/levin-phase-floor/06e/KEPT_ON_MAC_STUDIO.md`).
- **The population exists in the checkpoint:** per call, `cc_floor` holds the summed terms
  (`node_term`, `abs_E`, `R_g`, `R_2`, `R_c`, `declared`, `t_value_rounding`, `t_argument`) over
  all and over floor-attributed Clenshaw–Curtis leaves. 06e's report counts node-dominated calls as
  491 and 1,323 and declared-dominated ones as 2,493 and 2,785 at `1e-7` and `1e-8`.
- **Z2's call** reproduces in about 105 s; its `cc_floor` has `declared` 0.0 and `node_term`
  $1.318\times10^{-33}$.
- **The tools exist:** `worker.py`'s `_cc_floor_terms` reproduces each leaf's terms with $E$ signed;
  `x1_capture.py` captures chosen calls and checks them bit for bit; the captures unpickle with the
  census `scripts/` on `PYTHONPATH` (the pickles name `harness` functions).
- **Production supplies `theta_abserr`** only through `BesselPhaseGroup.levin_theta`, so the
  declared term appears on analytic-block calls; the group calls (Z2's among them) carry none.
- **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### B.1 Before you dispatch

1. `ssh macstudio-tunnel`; `runregistrykit list` shows nothing `running` or stale; the main
   checkout is clean; the worktree is at the dispatch SHA and clean.
2. The scratch copy matches its certification by SHA-256, with no `-wal` or `-shm`; 06e's checkpoint
   is `a0dc4d7f…`.
3. `touch` a marker; record the `var/` listing, as 06e's preflight did.

### B.2 Dispatch

One fresh-context agent, on **Opus**, in the worktree. It is given the prompt, the dispatch SHA and
§B.0, and it does not read `orchestrator/`. It writes only under `var/levin-phase-floor-06f/` and,
for registered runs, `var/runs/`. It never touches `var/levin-phase-floor-07/`, the scratch copy's
directory beyond reading the copy, or any store; makes no commit; runs no `git` write.

### B.3 The review — on the Mac Studio

1. **Scope:** nothing under `var/` newer than the marker outside `var/levin-phase-floor-06f/` and the
   06f runs; the scratch copy and 06e's checkpoint unchanged by SHA-256; the worktree clean.
2. **F0:** read the replays' bit-for-bit comparisons; **re-run Z2's call's replay yourself** and
   check its value, abserr and regions.
3. **M1, re-scored:** from the per-leaf tables, with your own reader, recompute $\sum|E|$,
   $|\sum E|$ and the signed-combination error on Z2's call and two other calls; and check, on Z2's
   call, that $\sum|E|$ over its returned leaves equals the checkpoint's `abs_E` over all its
   Clenshaw–Curtis leaves.
4. **M2, re-checked:** recompute $\delta\theta$ yourself in `mpmath` at five nodes of one call; and
   recompute $\delta I$ for one call by a method other than the agent's.
5. **The predictions:** F1–F6 copied verbatim into the log before the numbers, by its timestamps;
   each scored against the rule as written.
6. **The records:** README §5.1's headings, every deviation classified; `results.md` proposes no
   code; provenance headers; what is kept on the Mac Studio is listed with SHA-256.

Write `var/levin-phase-floor-06f/review.md`, with the verdict in its first line, and report.

### B.4 On the laptop, after the review

Copy `var/levin-phase-floor-06f/` back without pickles or files over 2 MB; re-check a point or two
over SSH; make the records commit: `docs/levin-phase-floor/06f/`, its `KEPT_ON_MAC_STUDIO.md`, the
log in `logs/`, `VERIFICATION.md` §6f, the board and `docs/OPEN_ISSUES.md`. Ask the user about the
worktree and anything kept. Report the results with what each implies for 06g and 07.

## 5. Stop and ask the user

As 06e's note §5, with prompt 06f's §7. **Never touch 07's stashes or saved work.**
