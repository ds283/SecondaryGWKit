# Orchestrator — prompt 07c, the chase below a refused leaf, and the Clenshaw–Curtis estimate

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/07c-chase-below-a-refused-leaf.md`](../07c-chase-below-a-refused-leaf.md).
**Board item:** 07c · **Closes, in its records commit, on the user's word:**
`[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`, and, if the fix includes it,
`[06d-cc-nested-estimate-measures-the-coarse-rule]`; **narrows**
`[07-c2-gate-threshold-was-never-measured]` · **Model:** Opus, for every agent. You review on whatever
model you were started with.

The prompt runs in three parts, as 07b's did:
- **Part A (prompt §2), the measurement,** on the Mac Studio: one Claude Code session there, which is
  the measuring agent, in a worktree at the dispatch SHA. It ends at a planned stop. This note's §A.
- **Part B (prompt §1, §3, §4), the code,** on the laptop: a fresh-context subagent in the main
  checkout, one commit. This note's §B.
- **Part C (prompt §5, §6),** on the Mac Studio, through an orchestrator session there: R32 under the
  lock, then the census. This note's §C, completed by an addendum at its dispatch.

## 0. The order, and why

1. **Part A first, because §1 and §3 wait on it.** The fix is chosen on the chase's measured cause and
   on the Clenshaw–Curtis estimate's measured accuracy; neither is known. Part A changes nothing in
   the tree.
2. **At Part A's stop** you relay the agent's tables **as it wrote them**, with your own re-scoring
   beside them (§A.3). The user chooses. You write §1, §3 and §9 from the choice, with the predictions
   it adds, and show the text to the user. It is committed and pushed on their word; that commit is
   Part B's dispatch SHA.
3. **Part B lands before Part C starts.**
4. **Every tree on the Mac Studio is a worktree.** The main checkout there stays at `28ebaf07`; it
   supplies the venv, the stores, `var/` and the registry. Git there is read with
   `--no-optional-locks`, and only the laptop writes git on the Mac Studio (`fetch`, `worktree add`
   and `remove`).
5. **What must not be touched:** A3 v2 and its scratch copy
   `var/levin-phase-floor-06/wider/state/a3v2_copy/` (read-only, kept for 08); R29's, R30's and R31's
   stores; 07's, 07a's and 07b's census and references run directories; `var/levin-phase-floor-07/`,
   `var/levin-phase-floor-07a/` and `var/levin-phase-floor-07b/`.

### 0.1 What 07b taught, which this note carries

07a's lessons (07b's note §0.1) stand. 07b adds:
- **"Beyond both" means beyond the sum.** The census agent counted 605 calls "moved beyond both
  codes' abserrs" by testing each abserr separately; only the 324 beyond the sum show a failing
  declaration. Every prediction and descriptive item that compares a move with two errors names the
  sum (prompt §6 says so).
- **A Result line can slip from its own table.** 07b's rehearsal log gave "3.87 to 59.3" where its
  table gave 2.91. Check every headline number against the table it summarises.
- **Check what the Mac Studio review leaves unchecked.** It did not re-check the census agent's
  descent analysis of the 198 newly unconverged calls; prompt §2.2 makes it Part A's first measurement.
- **zsh.** Two agents lost commands to zsh not word-splitting `$P` and reading `$T:s` as a modifier.
  Agents write commands for zsh or run them under `bash -c`.
- **The projection ran 1.20× short** (37,616 s projected, 45,016 s measured). Read Y0's projection with
  that margin.
- **A session on the Mac Studio may need its folder opened by the user.** At 07b's Part C dispatch the
  first `start_session` was refused until the user opened the session.
- **The code-equals-variant check (07b's U8) worked.** Prompt §6 keeps it as Y8.

---

## A. Part A — the measurement, on the Mac Studio

### A.0 Facts checked by the orchestrator at `db72ca28`, 2026-11-18

Re-check each at dispatch.
- **The code is 07b's** (`5e19c4b6`). `AdaptiveLevin/levin_quadrature.py` is 6,247 lines:
  `DEFAULT_LEVIN_MAX_DEPTH` `:265`, `SIX_PI` `:285`, `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO` `:397`,
  `_LEVIN_TRUST_MAX_RATIO` `:456`, `_LEVIN_EDGE_RATE_RATIO_MAX` `:624`, `_LEVIN_WINDOW_IROFF1_LIMIT`
  `:669`, `_cc_roundoff_floor` `:1010`, `_roundoff_exit_premise_holds` `:1111`, `_cc_leaf_unresolved`
  `:1159`, `_window_bisection_counts` `:1184`, `_trust_noise_yardstick` `:1242`, `_leaf_rate_rule`
  `:1324`, `_adaptive_levin_subregion_cc` `:3272`, `_adaptive_levin` `:4074`, its `_add_leaf` `:4729`,
  `_check_leaf` `:4840`, `_floor_limit_stopped_window_leaf` `:4989`, `_stop_window` `:5018`;
  `adaptive_levin_sincos` `:5983`.
- **On the Mac Studio** (main checkout `28ebaf07`, clean; 86 runs, all finished; the lock free; 116 GiB
  free):
  - 07b's Part C worktree `SecondaryGWKit-lpf07b-0a103b5` is still registered; it is removed at §A.1;
  - 07b's census run directory `var/runs/levin-phase-floor-07b-census-20261117T010200/` (150 files,
    10 GB, checkpoint `396784bc…`) and `var/levin-phase-floor-07b/census/out/stop3/compare_census_calls.calls.jsonl`
    (599 MB), each in 07b's `census/KEPT_ON_MAC_STUDIO.txt`;
  - the captures: R31's record `var/levin-phase-floor-07b/rehearsal/state/inputs/ref-G+q+r-h-3-20363-1.pkl`
    (`df2d478f…`), Part A's `var/levin-phase-floor-07b/measurement/state/refs/inputs/ref-R2-28.pkl`
    (`719ae661…`), and Part A's captures of the 187 (`measurement/KEPT_ON_MAC_STUDIO.txt`);
  - R31: `var/datastores/levin-phase-floor-07b-r31*`, primary `5981563b…`.
- **The populations** (from committed records): 198 newly unconverged calls, 103 integrands
  (`07b/census/out/scores/summary.json`); 324 calls past both errors, 18 referenced
  (`07b/review/final/moves.txt`; the 18 are listed in `VERIFICATION.md` §7b.5).
- **The references' cost** is not known before the draws. 07b's Part A took about 31.5 h; R1-x196's
  direct segment alone took 23.2 h at r1+r2. The chase's integrands lean to high $k$ (3e8 and 9.56e7 in
  84 of the 198), where stationary segments are costly: the projection decides R4's and R5's size.

### A.1 Before you dispatch

1. The prompt and this note are committed and pushed. That commit is the **dispatch SHA**.
2. **On the Mac Studio, read-only:** the main checkout clean at `28ebaf07`; `runregistrykit list` with
   nothing `running` or stale; the lock free; no `var/levin-phase-floor-07c*`; load and free disk
   recorded; the files of §A.0 hashed against their `KEPT_ON_MAC_STUDIO.txt` lines; the scratch copy
   against `copy_files_certified.txt`.
3. **The worktrees:** remove `SecondaryGWKit-lpf07b-0a103b5` (clean, nothing of it uncommitted), then
   `git fetch origin` in the Mac Studio main checkout and
   `git worktree add --detach ../SecondaryGWKit-lpf07c-<dispatch short SHA> <dispatch SHA>`. No venv in
   it.
4. **The marker:** `touch` a file in your scratchpad, and record the time on the Mac Studio.

### A.2 Dispatch

Start one Claude Code session on the Mac Studio over the saved `MacStudio` SSH connection, in the
worktree, on Opus, with `mcp__ccd_session__start_session`. Its brief: prompt §2 is its whole scope;
§A.0; the dispatch SHA; and these rules:
- **Every command** runs with its cwd in the worktree, calls the main checkout's `venv/bin/python` by its
  absolute path, with `PYTHONPATH` set to the worktree and `-B`. Nothing is written in the worktree.
- **It writes only** `var/levin-phase-floor-07c/measurement/` and its own registered run directories.
  No Ray, no lock, no store but A3 v2's scratch copy, read-only, for its captures.
- **`--cpus 6`**, and at most six worker processes.
- **Anything over ten minutes** runs as a registered detached run, launched, checked once, and waited
  on by one background `kill -0` loop.
- **Projections first:** the references' projection reported to you before any reading. Beyond 36 hours
  in all is a stop (prompt §8).
- **Every draw's seed** is written to `state/` before the draw, and every selection saved before any
  reading or solve.
- **Its records:** `log.md` in README §5.1's form, every deviation numbered and classified;
  `verification.md`, its part of `VERIFICATION.md` §7c; `issues.md`; `out/`, `scripts/`, `state/`; and
  `KEPT_ON_MAC_STUDIO.txt`, listing every pickle and file over 2 MB with SHA-256 and size.
- **It stops at prompt §2.7,** and at any of §8's conditions, and reports to the laptop session. It
  recommends nothing.

### A.3 At the stop, before you relay

Read over `ssh macstudio-tunnel`, read-only, with scratch readers of your own:
1. **Scope:** `find var -newer <marker>` lists only the agent's paths; the worktree's
   `git status --short --ignored` is empty; the files of §A.0 and the scratch copy are unchanged.
2. **The replays:** three chase calls and one of the 187, re-run at 07b's code yourself, compared with
   07b's census record in `float.hex`.
3. **The descent analysis (D1):** recount, from the agent's records, the calls whose every depth-20
   leaf descends from a refused leaf, and set it beside the census's 192.
4. **The estimate (D2):** the sample re-drawn from its seed; on three leaves, CC₃₁'s true error
   recomputed yourself against the agent's reference rule; each candidate's minimum ratio re-read from
   the table's inputs.
5. **The references:** the selections re-drawn from their seeds; three readings re-scored in exact
   Fractions, one from each of R4, R5 and R2-28; the agreement rule applied to all.
6. **The variants:** each diff against the tree touches only what it says. Re-run two chase calls under
   two variants yourself. **§2.6's qualification re-scored** by your own reader.

### A.4 After the user's choice

Write prompt §1, §3 and §9 from the user's words, quoted: the fix, its constants with their basis, what
it does to `converged` and the declarations, the leaves it touches, the diagnostics, the tests and
breakages, and §9's additions to §6's predictions (Y1's untouched counts, Y8's replays, and any
prediction for the chase calls), each threshold's basis checked against the census's classes. Show it
to the user. On their word commit and push it, with Part A's records copied back as 07b's were (no
pickles, nothing over 2 MB, by a list made first, each file checked by SHA-256). That commit is Part B's
dispatch SHA.

---

## B. Part B — the code, on the laptop

As 07b's note §B, with:
- **The baselines** at the dispatch SHA; nothing in `AdaptiveLevin/`, `ComputeTargets/` or
  `LiouvilleGreen/` changed since `5e19c4b6`.
- **The commit's paths:** `AdaptiveLevin/levin_quadrature.py`; `AdaptiveLevin/tests/` as §3 and §9
  name; `docs/levin-phase-floor/07c/code/`; the log `logs/07c-chase-below-a-refused-leaf.md` and the
  board; `CLAUDE.md` only for its `black` paragraph. **A path outside that list is a stop.**
- **The review's AST check** names, unchanged unless §9 changes them: `_roundoff_floor`, `_levin_G0_G1`,
  `_trust_noise_yardstick`, `_leaf_rate_rule` and its two constants, the location, the windows' pieces,
  `SIX_PI`, `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`, $\Lambda_c$, `DEFAULT_LEVIN_MAX_DEPTH` and
  `adaptive_levin_sincos`'s body; and whatever §9 says the fix leaves alone.
- **Replays:** R2-28 and one other chase call at the code commit from copies of their captures, each
  equal to the chosen variant's replay record and scored against its reading, and one call the fix
  should not touch, bit-identical to 07b's census.

---

## C. Part C — R32 and the census, on the Mac Studio

### C.0 When it starts

As 07b's note §C.0, with: no `var/levin-phase-floor-07c/rehearsal` or `census` yet, and no
`var/datastores/levin-phase-floor-07c-*`; R31's primary `5981563b…` and 07b's census checkpoint
`396784bc…` intact; the scratch copy certified. The laptop fetches, makes one worktree
`SecondaryGWKit-lpf07c-<Part C dispatch short SHA>`, removes Part A's, and starts the Mac Studio
orchestrator session in it with this note's §C as its brief.

### C.1 What the Mac Studio orchestrator is told

As 07b's note §C.1, with R31 the parent rehearsal (prompt §5 S0′), R32 the new one, and:
- **Planned stop 1** after R32 and its comparison with R31; **stop 2** after Y0, with the predictions
  written; **stop 3** after the census, before any verdict.
- **A call Y1 says is untouched that differs from 07b's is a stop,** before anything else is scored;
  so is Y8 failing.
- **No new references.** Y4 is scored on 07b's U4 references and this prompt's R4, R5 and R2-28; Y6 on
  06b's $S_{10}$.

### C.2 The review — on the Mac Studio

As 07b's note §C.2, with R32 against R31, and Y0–Y8 re-scored from the two checkpoints by your own
readers: Y1's untouched calls bit for bit at both rtols; Y4 on every referenced call; Y6 on all 24; Y7
over all 2,484 pairs; Y8 on every replayed call; and the chase calls still unconverged, recounted.

### C.3 What it reports

Pass or stop on the first line; then R32 and S0′–S4; Y0–Y8 with their numbers; where the fix acted;
convergence against 07b's and 07a's; cost.

### C.4 After the review — on the laptop

As 07b's note §C.4: the copy-back list first, every file checked by SHA-256, and re-checks from the
copies (Y4 on R2-28, one R4 call and one R5 call; Y6 on two calls; one untouched and one moved
rehearsal row, in `float.hex`). Then ask the user about the issues and the agents' `issues.md`, and
make **one records commit**, pushed: the records in `docs/levin-phase-floor/07c/`; the logs appended to
`logs/07c-chase-below-a-refused-leaf.md` with the verdicts; `VERIFICATION.md` §7c; the board, with
07c's row and a Decisions row; the issues on the user's word; `docs/OPEN_ISSUES.md` with its count and
date corrected by counting; this directory's README row and the campaign README's.

**Kept until the user says otherwise:** R29, R30, R31 and R32; 07a's, 07b's and 07c's census run
directories (07c's is 07d's "before"); A3 v2's scratch copy (for 08).

## 5. Stop and ask the user

- At Part A's planned stop, always.
- At any stop condition of prompt §8 that an agent reports, with the agent's words.
- When a premise this note or the prompt states is found false at a dispatch.
- Before any write outside the paths named here, any deletion, and any push.
