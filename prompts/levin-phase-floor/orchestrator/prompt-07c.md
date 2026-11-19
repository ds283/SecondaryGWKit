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

### C.5 Addendum: Part C's dispatch

*Written 2026-11-19, at Part B's landing.*

**Part B** is the code commit **`67b5af74`**, whose parent is **`cff4237f`**. It was reviewed in
07a's note §B.3's checks, as §B adapts them, all passing, and pushed on the user's word: *"Yes, push
it and write the §C addendum"*. One orchestrator commit came between the prompt (`2fd979fd`) and it,
pushed: `cff4237f` amended the prompt at Part A's stop (its §1, §3 and §9: the fix is variant (S);
`[07c-r2-28-trusted-levin-leaf-hides-a-common-error]` opened) and committed Part A's records.

**The tree.** Part C runs at **this addendum's commit**, in the worktree
`SecondaryGWKit-lpf07c-<this commit's short SHA>`. Its code is `67b5af74`'s.
- **S0′ holds by inspection:** `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists
  only `CLAUDE.md`, whose `black` paragraph gained 07b Part C's and 07c Part A's record directories.
  The build imports nothing that changed. R31 is the parent rehearsal. The agent still runs the check
  as prompt §5 writes it.
- `git diff --stat cff4237f 67b5af74 -- . ':!docs' ':!prompts'` touches only:
  - `AdaptiveLevin/levin_quadrature.py` (SHA-256 `d141a80b…`);
  - `AdaptiveLevin/tests/` (`test_levin_chase_stop.py`, new; `test_levin_roundoff_exit.py` and
    `test_levin_trust_yardstick.py`, where the stop acts).

  Nothing under `ComputeTargets/`, `ObjectFactories/` or `LiouvilleGreen/` moved. **So outside
  `QuadSourceIntegral` R32 is predicted to equal R31**, as §5 says.

**What Part B's review found that Part C must carry:**
- **The review's numbers, re-run on the laptop:**
  - the seven roots at the commit: 141 (129 and the 12 added), 781, 39, 156 (1 skipped), 280, 127 and
    64, all OK;
  - the parent's values quoted in tests (b), (c3), (e), (f1) and (f2), reproduced in `float.hex` at
    the parent's own module (`8956c7b6…`, `5e19c4b6`'s). There (b) is unconverged at 100 regions, with
    six depth-20 Clenshaw–Curtis leaves, all untrusted, below its refused leaf;
  - breakages (i), (ii), (iii) and (v) re-applied, each caught by the tests §3 names;
  - **Y8, begun:** R2-28, `c-1e-8-1-2113-1` and `c-1e-8-1-2229-4` replayed at the commit from copies
    of their captures, each checked by SHA-256 against `calls_07c.json`. Each equals (S)'s replay
    record in `float.hex`, leaf by leaf and in `regions_sha256`. The counts (`num_chase_stops`,
    `num_chase_stopped_leaves`, `num_chase_noise_bisections`) are (1, 69, 6), (1, 19, 9) and
    (0, 0, 9). R2-28 is converged at 1.1434 of its abserr, scored in exact fractions, as §9.5
    predicts. `c-1e-8-1-2229-4`, which the stop does not touch, also equals 07b's census. The census
    scores Y8 on every call Part A replayed; these three do not stand in for it.
- **Calls newly unconverged.** §9.5 lists them as descriptive and expects none. In the
  LiouvilleGreen draws the stop acts in 3,051 of 4,320 calls, and **26 go from converged to
  unconverged**, none the other way (the log's Observations 5; the same under (S)). They are
  analytic Bessel calls at rtol $10^{-8}$ and $10^{-10}$, the class §9.3 says Part A's basis does not
  reach. The orchestrator attributed three of them leaf by leaf:
  - in each, essentially all of the call's abserr is one stopped Clenshaw–Curtis leaf;
  - that leaf's $\delta_1$ is $10^{-17}$–$10^{-15}$. Its declaration is §1.1's
    $\max(\text{residual}, \delta_1)/(1-r)$, where the residual is $|{\rm CC}_{47} - {\rm CC}_{24}|$
    (`[06d-cc-nested-estimate-measures-the-coarse-rule]`'s overstatement) and $r$ is 0.81–0.97;
  - at the parent the same interval was refined into two to four trusted Clenshaw–Curtis leaves,
    declaring $10^{-14}$–$10^{-10}$ in all;
  - the abserr goes to 1.008–10.7 times the request, while the value moves at most $4.4\times10^{-4}$
    of the parent's abserr. These are overstatements, not errors.

  On Part A's 454 census calls (S) turns none converged → unconverged. **Whether the census has any is
  the measurement.** A newly unconverged call is not a stop and changes no verdict: Y4, Y6 and Y7 count
  an unconverged call as passing, and Y9 is about depth-20 leaves. Report each one with §9.5's
  instrument: the stopped leaf carrying the most of its abserr, that leaf's route, depth, $r$ and
  share. These go to the user at the records commit, with the agent's Observations 4 and 5.
- **The stop's cost.** In the draws the regions go from 147,029 to 140,896 (0.958×), and the summed
  wall from 1,220.9 s to 1,177.3 s. On Part A's 454 calls, (S) took 0.745× the regions. **The census
  projects its wall from Y0's K0 run before the full run** (§9.4). A projection beyond 48 hours is a
  stop, reported with what would be cut to fit.
- **Y1 reads the counts** (§9.5). 07b's census worker records named result keys only. So the copied
  harness records `num_chase_stops`, `num_chase_stopped_leaves` and `num_chase_noise_bisections` on
  every call, marked "prompt 07c", as the draws' harness
  `docs/levin-phase-floor/07c/code/lg_draws_07c.py` does. A call is untouched if `num_chase_stops` is
  0.
- **Not Part C's:** Part B's Observations not acted on 1–4:
  - 1: on synthetic noise, the trust check chases with no rate-rule refusal;
  - 2: noise above $4\nu_B$ escapes both exits (Part A's proposed
    `[07c-integrand-noise-above-the-cc-floor]`);
  - 3: a range-refused Levin leaf at the noise level is itself stopped, and returned on its
    $r$-based declaration;
  - 4: stopped Clenshaw–Curtis leaves near $6\pi$ declare CC₂₄'s error.

  They are put to the user at the records commit, with Part A's `issues.md` items 1–12 (§C.4). The
  census scores Y4 and Y6 as written and attributes neither.

**§C.0, re-checked read-only on the Mac Studio, 13:42 UTC, 2026-11-19:**
- **The main checkout** is `handover-remedial` at `28ebaf07`, with `git status --short` empty.
  `git worktree list` shows it and Part A's `SecondaryGWKit-lpf07c-2fd979f` (at `2fd979fd`). That
  worktree is removed at dispatch (§C.0).
- **The registry:** 92 runs, all finished. Nothing is `running` or stale.
- **The lock and Ray:** `rehearsal_lock status` reports the lock free. Neither
  `~/.raypoolkit/rehearsal-lock` nor the old `~/ScienceData/.rehearsal-lock` exists, and
  `ray-processes` lists no Ray process of `/tmp/ray-sgk`.
- **Load and disk:** the load is 8.54 / 8.79 / 8.47, all of it system services and applications
  (`CategoriesService`, `WindowServer`, `SetStoreUpdateService`, Mail). 106 GiB is free (89 % used);
  07b's census run directory is 10.9 GB.
- **The 07c paths:**
  - `var/levin-phase-floor-07c/` holds only Part A's `measurement/`, 17,111 files;
  - there is no `rehearsal/` or `census/` yet, and no `var/datastores/levin-phase-floor-07c-*`;
  - `var/runs/` holds Part A's six run directories: `levin-phase-floor-07c-base-…`,
    `-capture-chase-…`, `-capture-r5-…`, `-refs-…`, `-refs2-…` and `-variants-…`. Their slugs do not
    collide with `r32-07c`, `k0` or `census`.
- **The inputs, by SHA-256 against their recorded lists,** 160 files, all OK:
  - R31's five store files against `docs/levin-phase-floor/07b/rehearsal/KEPT_ON_MAC_STUDIO.txt`,
    primary `5981563b…`;
  - 07b's census run directory `levin-phase-floor-07b-census-20261117T010200`: all 150 files against
    `docs/levin-phase-floor/07b/census/KEPT_ON_MAC_STUDIO.txt`, checkpoint `396784bc…` among them,
    and 150 files on disk;
  - the scratch copy's five files against `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`.

  None of the store directories holds a `-wal` or `-shm` file.
- **The venv** is Python 3.13.16, with `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
  0.2.0, `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy` 2.2.4, `scipy` 1.16.3 and `mpmath` 1.3.0.

**The dispatch** waits for the user's word. Then the laptop does three things:
1. It fetches into the Mac Studio's main checkout, which moves no working tree.
2. It repeats §C.0's checks, removes Part A's worktree and makes the one worktree at this commit,
   detached and with no venv.
3. It starts the Mac Studio orchestrator session in that worktree, through the saved `MacStudio`
   connection, with this note's §C as its brief. The folder must be trusted in the app first.

## 5. Stop and ask the user

- At Part A's planned stop, always.
- At any stop condition of prompt §8 that an agent reports, with the agent's words.
- When a premise this note or the prompt states is found false at a dispatch.
- Before any write outside the paths named here, any deletion, and any push.
