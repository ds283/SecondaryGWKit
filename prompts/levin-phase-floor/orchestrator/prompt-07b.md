# Orchestrator — prompt 07b, the Levin leaf beside a window, and how honest the census is

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/07b-levin-leaf-beside-a-window.md`](../07b-levin-leaf-beside-a-window.md).
**Board item:** 07b · **Closes, in its records commit, on the user's word:**
`[07a-levin-leaf-beside-a-window-understates-its-error]`, and, on their conditions,
`[06b-moved-calls-declare-less-than-their-error]` and `[07-iroff1-window-stop-can-leave-a-call-dishonest]`;
**narrows** `[07a-trust-check-auxiliary-route-moves-the-declared-error]` and
`[07-c2-gate-threshold-was-never-measured]` · **Model:** Opus, for every agent. You review on
whatever model you were started with.

The prompt runs in three parts, as 07a's did:
- **Part A (prompt §2), the measurement,** on the Mac Studio: one Claude Code session there, which is
  the measuring agent, in a worktree at the dispatch SHA. It ends at a planned stop. This note's §A.
- **Part B (prompt §1, §3, §4), the code,** on the laptop: a fresh-context subagent in the main
  checkout, one commit. This note's §B.
- **Part C (prompt §5, §6),** on the Mac Studio, through an orchestrator session there: R31 under the
  lock, then the census. This note's §C, completed by an addendum at its dispatch.

## 0. The order, and why

1. **Part A first, because §1 and §3 wait on it.** The fix is chosen on the mechanism, which is not
   known. Part A changes nothing in the tree.
2. **At Part A's stop** you relay the agent's tables **as it wrote them**, with your own re-scoring of
   §2.5 item 4 and of a sample of the references beside them (§A.3). The user chooses. You write §1,
   §3 and §9 from the choice, with the predictions it adds, and show the text to the user. It is
   committed and pushed on their word; that commit is Part B's dispatch SHA.
3. **Part B lands before Part C starts.**
4. **Every tree on the Mac Studio is a worktree.** The main checkout there stays at `28ebaf07`; it
   supplies the venv, the stores, `var/` and the registry. Git there is read with
   `--no-optional-locks`, and only the laptop writes git on the Mac Studio (`fetch`, `worktree add`
   and `remove`).
5. **What must not be touched:** A3 v2 and its scratch copy
   `var/levin-phase-floor-06/wider/state/a3v2_copy/` (read-only, kept for 08); R29's and R30's stores;
   07's and 07a's census and references run directories; `var/levin-phase-floor-07/` and
   `var/levin-phase-floor-07a/`.

### 0.1 What 07a taught, which this note carries

- **A threshold needs a basis that covers what it is applied to.** W7's $10^{-5}$ rested on Part A's
  430 calls, all `G+q+r`, and failed on minima windows none of them held. When you write §9's
  predictions, check each threshold's basis against the census's classes and say where it does not
  reach.
- **A prediction names what it counts.** W1 needed two rulings (which solves count as a route change;
  what "bit for bit" compares) and W5 one (gated or forced solves). Write U1 and every §9 addition so
  that each counts named solves and compares named outputs.
- **Copy predictions by text, not by line.** 07a's log copied §5 and §6 truncated, from a line offset.
  Check the agent's copies by substring.
- **`python -m` puts the cwd ahead of `PYTHONPATH`.** The Mac Studio orchestrator ran the main
  checkout's code twice that way. Every command of every agent runs with its cwd in the worktree.
- **Copy-back by a list made first,** checked by SHA-256 against a list made on the Mac Studio, as 07a.
- **Launch, one check, end the turn.** Lock commands each a shell command of their own, by `>>`.
- **`HEAD` is not the agent's.**

---

## A. Part A — the measurement, on the Mac Studio

### A.0 Facts checked by the orchestrator at `a74e7e10`, 2026-11-13

Re-check each at dispatch.
- **The code is 07a's.** `git diff --stat 1ec734d7 a74e7e10 -- AdaptiveLevin ComputeTargets
  LiouvilleGreen tools` lists `89c368f2`'s four files, docstrings only. `AdaptiveLevin/levin_quadrature.py`
  is 6,003 lines: `DEFAULT_LEVIN_MAX_DEPTH` `:243`, `SIX_PI` `:263`, `_LEVIN_TRUST_MAX_RATIO` `:424`,
  `_CC_THETA_PRIME_REL_ERROR_SUPPLIED` `:457`, `_LEVIN_WINDOW_HALF_PHASE` `:521`,
  `_LEVIN_COALESCENCE_LAMBDA` `:533`, `_declared_endpoint_phase_err` `:995`,
  `_trust_noise_yardstick` `:1154`, `_locate_stationary_points` `:3533`, `_adaptive_levin` `:3911`,
  its `_solve` `:4266`, `_solve_halves` `:4627`, `_check_leaf` `:4645` (the trusted declaration at
  about `:4725`), `_floor_limit_stopped_window_leaf` `:4764`, `_stop_window` `:4793`;
  `adaptive_levin_sincos` `:5751`. The module docstring's stationary-point item (about `:111–121`)
  records the blindness of the residual and the trust check near a stationary point.
- **On the Mac Studio** (main checkout `28ebaf07`, clean; 77 runs, all finished; 131 GiB free):
  - `var/levin-phase-floor-07/census/state/v4/capture/`: 25 directories, `C00`–`C24`;
    `xcheck/capture/`: 10, `X00`–`X09`;
  - `var/levin-phase-floor-07a/measurement/state/pairs_capture/` (the two pairs) and
    `var/levin-phase-floor-07a/census/state/refs/capture_h`, `capture_h2`, `capture_parent`,
    `capture_parent2` (`1-34599` and `3-678`, at 07a's and 07's code);
  - **06b's census state is gone** (`var/levin-phase-floor-06b/census/state/` absent); its 24 W2 calls
    are re-scored from 07a's checkpoint and captured afresh where needed. Their references are
    committed, `docs/levin-phase-floor/06b/census/out/w2_reference.json`;
  - **07a's Part C worktree** `SecondaryGWKit-lpf07a-4bf50d3` is still registered; it is removed at
    §A.1.
- **The references' cost** (re-check from the records): 07's V4 references took 74 readings in about
  25.6 h; the cross-check's model gives about 11,300 CPU-s a class-(c) row, so R1 alone is about 16 h at
  six workers. R2's cost is not known before its draw.
- **07a's census:** `levin-phase-floor-07a-census-20261112T132744`, checkpoint `461df2ec…`, kept whole.
  R30: `var/datastores/levin-phase-floor-07a-r30*`, primary `e5f8b443…`.

### A.1 Before you dispatch

1. The prompt and this note are committed and pushed. That commit is the **dispatch SHA**.
2. **On the Mac Studio, read-only:** the main checkout clean at `28ebaf07`; `runregistrykit list` with
   nothing `running` or stale; the lock free; no `var/levin-phase-floor-07b*`; load and free disk
   recorded; the captures of §A.0 hashed against their `KEPT_ON_MAC_STUDIO.txt` lines; the scratch copy
   against `copy_files_certified.txt`; 07a's census checkpoint and R30's primary hashed.
3. **The worktrees:** remove `SecondaryGWKit-lpf07a-4bf50d3` (clean, nothing of it uncommitted), then
   `git fetch origin` in the Mac Studio main checkout and
   `git worktree add --detach ../SecondaryGWKit-lpf07b-<dispatch short SHA> <dispatch SHA>`. No venv in
   it.
4. **The marker:** `touch` a file in your scratchpad, and record the time on the Mac Studio.

### A.2 Dispatch

Start one Claude Code session on the Mac Studio over the saved `MacStudio` SSH connection, in the
worktree, on Opus, with `mcp__ccd_session__start_session`. Its brief: prompt §2 is its whole scope;
§A.0; the dispatch SHA; and these rules:
- **Every command** runs with its cwd in the worktree, calls the main checkout's `venv/bin/python` by its
  absolute path, with `PYTHONPATH` set to the worktree and `-B`. Nothing is written in the worktree.
- **It writes only** `var/levin-phase-floor-07b/measurement/` and its own registered run directories.
  No Ray, no lock, no store but A3 v2's scratch copy, read-only, for its captures.
- **`--cpus 6`**, and at most six worker processes.
- **Anything over ten minutes** runs as a registered detached run, launched, checked once, and waited
  on by one background `kill -0` loop.
- **Projections first:** §2.2's pilot and §2.4's projection, reported to you before the long runs.
  Beyond 36 hours in all is a stop (prompt §8).
- **Every draw's seed** is written to `state/` before the draw, and every selection saved before any
  reading.
- **Its records:** `log.md` in README §5.1's form, every deviation numbered and classified;
  `verification.md`, its part of `VERIFICATION.md` §7b; `issues.md`; `out/`, `scripts/`, `state/`; and
  `KEPT_ON_MAC_STUDIO.txt`, listing every pickle and file over 2 MB with SHA-256 and size.
- **It stops at prompt §2.6,** and at any of §8's conditions, and reports to the laptop session. It
  recommends nothing.

### A.3 At the stop, before you relay

Read over `ssh macstudio-tunnel`, read-only, with scratch readers of your own:
1. **Scope:** `find var -newer <marker>` lists only the agent's paths; the worktree's
   `git status --short --ignored` is empty; the captures, the scratch copy, 07a's checkpoint and R30
   are unchanged.
2. **The replays:** three calls of §2.1 item 1, two of them failures, re-run at 07a's code yourself,
   compared with 07a's census record in `float.hex`.
3. **06b's 24** re-scored by your own reader from 07a's checkpoint and the committed $S_{10}$.
4. **The mechanism:** on v08's carrying leaf, the true errors of $V_P$, $V_L + V_R$ and $Q$ recomputed
   from the agent's references, and one reference re-read yourself on a quarter's span.
5. **The references:** the selections re-drawn from their seeds; three readings re-scored in exact
   Fractions, one from each of R1, R2 and R3; the agreement rule applied to all.
6. **The variants:** each diff against the tree touches only what it says. Re-run one failure under two
   variants yourself. **§2.5 item 4 re-scored** by your own reader.

### A.4 After the user's choice

Write prompt §1, §3 and §9 from the user's words, quoted: the fix, its constants with their basis, what
it does to `converged`, the leaves it touches, the tests and breakages, and §9's additions to §6's
predictions, each threshold's basis checked against the census's classes (§0.1). Show it to the user.
On their word commit and push it, with Part A's records copied back as 07a's were (no pickles, nothing
over 2 MB, by a list made first, each file checked by SHA-256). That commit is Part B's dispatch SHA.

### A.5 The extension (prompt §2.7), written 2026-11-15

At Part A's stop the user chose to extend it (*"I am going to choose the extension"*; *"Yes, include
the adaptive window variant; write the extension"*). §A.4 waits until the extension stops.

**Already checked at Part A's stop (§A.3), all agreeing with the agent:**
- scope;
- the replays of v00, v08 and v09;
- 06b's 24;
- v08's leaf;
- the draws, the 74 references and the 148 scores;
- §2.5 item 4 for all seven variants;
- v08 under (e) and (f).

**The missed stop.** The agent did not take §8's 36-hour stop: its projection of 9.92 h left out the
direct segments, against about 307 h by 07a's model. It logged this as deviation 16. The run took
about 31.5 h. The user accepted the run as it stands, with the dispatch: *"Yes to both; commit,
push and dispatch the extension"*.

**Before dispatch:**
1. The extension commit is pushed; it is the **extension SHA**. Its code is Part A's:
   `git diff --stat 436ed5d4 <extension SHA> -- AdaptiveLevin ComputeTargets LiouvilleGreen tools` is
   empty.
2. **On the Mac Studio, read-only:**
   - `runregistrykit list` with nothing `running` or stale, and the lock free;
   - the worktree `SecondaryGWKit-lpf07b-436ed5d` clean with `--ignored`;
   - a SHA-256 list of every file under `var/levin-phase-floor-07b/measurement/` saved in your
     scratchpad, to check at the stop that Part A's files are unchanged;
   - a new marker, `/tmp/sgk07b_marker_X`, with its time recorded.
3. `git fetch origin` in the Mac Studio main checkout, so that the agent can read the extension with
   `git --no-optional-locks show <extension SHA>:prompts/levin-phase-floor/07b-levin-leaf-beside-a-window.md`.
   **The worktree stays at `436ed5d4`**, since the code is the same.

**Dispatch:** resume the Part A session, `lpf07b-part-a`, with `SendMessage`. It holds the harness,
the captures and the leaf probes. The message gives:
- the extension SHA, and how to read §2.7 from it;
- §2.7 as its whole scope;
- §A.2's rules, unchanged;
- the 12-hour stop, with the projection made by 07a's model including $\gamma D_f$;
- one reminder from deviation 16: the projection is reported to the laptop **before** any run over
  ten minutes.

**At the stop, before you relay, read-only:**
1. **Scope:** `find var -newer /tmp/sgk07b_marker_X` lists only the agent's paths. Part A's files
   match the saved list, except the three appended records, whose Part A text is unchanged as a
   prefix. The worktree is clean.
2. **The variants:** each new diff against the tree touches only what it says. Replay v08 and one of
   06b's four under two new variants yourself, from captures copied read-only into your scratchpad,
   and compare in `float.hex`.
3. **Table 4 re-scored** by your own reader for every new variant, with (X4)'s counts and the split.
4. **(g)'s family (R)** read from the agent's output, and one case re-run yourself.
5. **(X2)'s separating quantity,** if one is found, recomputed on two leaves from the agent's records.

---

## B. Part B — the code, on the laptop

As 07a's note §B.1–§B.4, with:
- **The baselines** at the dispatch SHA; nothing in `AdaptiveLevin/`, `ComputeTargets/` or
  `LiouvilleGreen/` changed since `1ec734d7` but `89c368f2`'s docstrings.
- **The commit's paths:** `AdaptiveLevin/levin_quadrature.py`; `AdaptiveLevin/tests/` as §3 and §9
  name; `docs/levin-phase-floor/07b/code/`; the log `logs/07b-levin-leaf-beside-a-window.md` and the
  board; `CLAUDE.md` only for its `black` paragraph. **A path outside that list is a stop.**
- **The review's AST check** names, unchanged unless §9 changes them: `_roundoff_floor`, `_levin_G0_G1`,
  `_cc_roundoff_floor`, `_trust_noise_yardstick`, `_window_bisection_counts`, the location, the windows'
  pieces, `SIX_PI`, `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`, $\Lambda_c$, `DEFAULT_LEVIN_MAX_DEPTH` and
  `adaptive_levin_sincos`'s body; and whatever §9 says the fix leaves alone.
- **Replays:** v08 and one control at the code commit from copies of their captures, each scored
  against its reading, and one R2 call the fix should not touch, bit-identical to 07a's census.

---

## C. Part C — R31 and the census, on the Mac Studio

### C.0 When it starts

As 07a's note §C.0, with: no `var/levin-phase-floor-07b/rehearsal` or `census` yet, and no
`var/datastores/levin-phase-floor-07b-*`; R30's primary `e5f8b443…` and 07a's census checkpoint
`461df2ec…` intact; the scratch copy certified. The laptop fetches, makes one worktree
`SecondaryGWKit-lpf07b-<Part C dispatch short SHA>`, removes Part A's, and starts the Mac Studio
orchestrator session in it with this note's §C as its brief.

### C.1 What the Mac Studio orchestrator is told

As 07a's note §C.1, with R30 the parent rehearsal (prompt §5 S0′), R31 the new one, and:
- **Planned stop 1** after R31 and its comparison with R30; **stop 2** after U0, with the predictions
  written; **stop 3** after the census, before any verdict.
- **A call U1 says is untouched that differs from 07a's is a stop,** before anything else is scored.
- **No new references.** U4 and U6 are scored on Part A's references and 06b's $S_{10}$.

### C.2 The review — on the Mac Studio

As 07a's note §C.2, with R31 against R30, and U0–U7 re-scored from the two checkpoints by your own
readers: U1's untouched calls bit for bit at both rtols; U4 on every referenced call; U6 on all 24;
U7 over all 2,484 pairs.

### C.3 What it reports

Pass or stop on the first line; then R31 and S0′–S4; U0–U7 with their numbers; where the fix acted;
convergence against 07a's; cost.

### C.4 After the review — on the laptop

As 07a's note §C.4: the copy-back list first, every file checked by SHA-256, and re-checks from the
copies (U4 on v08 and two R2 calls, U6 on two calls, one untouched and one moved rehearsal row, in
`float.hex`). Then ask the user about the issues and the agents' `issues.md`, and make **one records
commit**, pushed: the records in `docs/levin-phase-floor/07b/`; the logs appended to
`logs/07b-levin-leaf-beside-a-window.md` with the verdicts; `VERIFICATION.md` §7b; the board, with
07b's row and a Decisions row; the issues on the user's word; `docs/OPEN_ISSUES.md` with its count and
date corrected by counting; this directory's README row and the campaign README's.

**Kept until the user says otherwise:** R29, R30 and R31; 07a's and 07b's census run directories
(07b's is 08's "before"); A3 v2's scratch copy (for 08).

### C.5 Addendum: Part C's dispatch

*Written 2026-11-16, at Part B's landing.*

**Part B** is the code commit **`5e19c4b6`**, whose parent is **`7b82a0cd`**. It was reviewed in
07a's note §B.3's checks, as §B adapts them, all passing, and pushed on the user's word: *"Yes, push
it and write the §C.5 addendum"*. Three orchestrator commits came before it, all pushed:
- `a510e4ad` wrote Part A's extension (prompt §2.7), on the user's choice at §2.6's stop;
- `c768c4ad` amended the prompt at the extension's stop (its §1, §3 and §9: the fix is variant (h);
  `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]` opened) and committed Part A's
  records;
- `7b82a0cd` amended it at Part B's stop (its §10). Under the range rule, the 1e-8 rung of
  `test_three_bessel.py`'s `test_quadrature_refinement_is_not_a_certificate_of_phase_accuracy` is
  already at the phase floor, so its tenfold improvement to 1e-14 is gone. The user: *"Yes, write the
  §10 amendment"*. That one assertion is restated as a stall across the rungs.

**The tree.** Part C runs at **this addendum's commit**, in the worktree
`SecondaryGWKit-lpf07b-<this commit's short SHA>`. Its code is `5e19c4b6`'s.
- **S0′ holds by inspection:** `git diff --stat 1ec734d7 7b82a0cd -- . ':!docs' ':!prompts'` lists
  `CLAUDE.md` and `89c368f2`'s four files (`ComputeTargets/tests/test_quadsource_integral.py`,
  `LiouvilleGreen/bessel_phase.py`, `LiouvilleGreen/tests/test_three_bessel.py`,
  `LiouvilleGreen/three_bessel_integrals.py`). Each of the four is AST-equal with docstrings removed,
  checked on the laptop. R30 is the parent rehearsal. The agent still runs the check as prompt §5
  writes it.
- `git diff --stat 7b82a0cd 5e19c4b6 -- . ':!docs' ':!prompts'` touches only:
  - `AdaptiveLevin/levin_quadrature.py` (SHA-256 `8956c7b6…`);
  - `AdaptiveLevin/tests/` (`test_levin_rate_rules.py`, new; `test_levin_leaf_values.py`,
    `test_levin_roundoff_exit.py` and `test_levin_trust_yardstick.py`, where the range rule acts);
  - `LiouvilleGreen/tests/test_three_bessel.py`, §10.3's one test.

  Nothing under `ComputeTargets/`, `ObjectFactories/` or LiouvilleGreen's modules moved. **So outside
  `QuadSourceIntegral` R31 is predicted to equal R30**, as §5 says.

**What Part B's review found that Part C must carry:**
- **The review's numbers, re-run on the laptop:**
  - the seven roots at the commit: 129 (116 and the 13 added), 781, 39, 156 (1 skipped), 280, 127 and
    64, all OK. The review's first ComputeTargets run ended after about seven minutes with no summary
    and no traceback. Run again alone it passed 781, as the agent's run did. The cause was not found;
  - the parent's values quoted in tests (b), (c), (d), (e1) and (e2), reproduced in `float.hex` at
    `c768c4ad` (07a's code) in a scratch worktree. There (b) is at 7.62 and (c) at 102.8 times its
    abserr, both converged;
  - breakages (i), (ii), (iii) and (vi) re-applied, each caught by the tests §3 names, (ii) also by
    §10's test;
  - **U8, begun:** v08, v13, R2-03 and R2-22 replayed at the commit from copies of their captures, each
    equal to (h)'s replay record in `float.hex`, leaf by leaf. v08 is honest at 0.0323 of its abserr
    (3.16 at 07a's code), and v13 at 0.00188. R2-22, which the rules do not touch, also equals 07a's
    census. The census scores U8 on every call Part A replayed; these four do not stand in for it.
- **The range rule's cost.** In the LiouvilleGreen draws it acts in 3,811 of 4,320 calls, with 34,018
  refusals. The regions go from 76,692 to 147,029 (1.92×), and the summed wall from 802.7 s to
  1,271.2 s. On R2, §9.4's estimate is about 5×. **The census projects its wall from U0's K0 run before
  the full run.** A projection beyond 48 hours is a stop (§9.7), reported with what would be cut to
  fit. R31's `QuadSourceIntegral` stage may run longer than R30's; that is cost, not a stop.
- **The edge rule never acted in the draws,** though 423 calls have a window. In Part A it acts on v08
  and v13. The census's refusals by rule (§9.5's descriptive item) show whether it acts in production.
  A zero count is a finding, reported, not a stop.
- **U1 reads the two counts** (§9.5). 07a's census worker records named result keys only, so the
  copied harness records `num_edge_rule_refusals` and `num_range_rule_refusals` on every call, marked
  "prompt 07b", as the draws' harness `docs/levin-phase-floor/07b/code/lg_draws_07b.py` does.
- **Not Part C's:** Part B's Observations not acted on 2 and 3. Observation 2 is marginal dishonesty
  on synthetic no-window phases with large raw values. Observation 3 is the draws' 434 three-Bessel
  results that are converged and dishonest against the test's oracles, which the rule does not reach.
  They are put to the user at the records commit, with Part A's `issues.md` items 1–14 (§C.4). The
  census scores U4 and U6 as written and attributes neither.

**§C.0, re-checked read-only on the Mac Studio, 13:50 UTC, 2026-11-16:**
- **The main checkout** is `handover-remedial` at `28ebaf07`, with `git status --short` empty.
  `git worktree list` shows it and Part A's `SecondaryGWKit-lpf07b-436ed5d` (at `436ed5d4`), which
  is removed at dispatch (§C.0).
- **The registry:** 81 runs, 73 done, 4 failed and 4 killed, all from earlier prompts. Nothing is
  `running` or stale.
- **The lock and Ray:** `rehearsal_lock status` reports the lock free. Neither
  `~/.raypoolkit/rehearsal-lock` nor the old `~/ScienceData/.rehearsal-lock` exists, and
  `ray-processes` lists no Ray process of `/tmp/ray-sgk`.
- **Load and disk:** the load is 4.57 / 4.35 / 4.14, and 133 GiB is free (86 % used).
- **The 07b paths:**
  - `var/levin-phase-floor-07b/` holds only Part A's `measurement/`. Its 7,645 files are the 7,479
    copied back and the 166 kept there, which
    `docs/levin-phase-floor/07b/measurement/KEPT_ON_MAC_STUDIO.txt` lists;
  - there is no `rehearsal/` or `census/` yet, and no `var/datastores/levin-phase-floor-07b-*`;
  - `var/runs/` holds Part A's four run directories, `levin-phase-floor-07b-pilot-…`, `-refs-…`,
    `-leafrefs-…` and `-ext-leafrefs-…`. Their slugs do not collide with `r31-07b`, `k0` or
    `census`.
- **The inputs, by SHA-256 against their recorded lists,** 159 files, all OK:
  - R30's five store files against `docs/levin-phase-floor/07a/rehearsal/KEPT_ON_MAC_STUDIO.txt`,
    primary `e5f8b443…`. R30's run directory and launch log are present;
  - 07a's census run directory `levin-phase-floor-07a-census-20261112T132744`, all 149 files against
    `docs/levin-phase-floor/07a/census/KEPT_ON_MAC_STUDIO.txt`, checkpoint `461df2ec…` among them, and
    149 files on disk;
  - the scratch copy's five files against `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`,
    with no `-wal` or `-shm`.
- **The venv** is Python 3.13.16, with `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
  0.2.0, `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy` 2.2.4, `scipy` 1.16.3 and `mpmath` 1.3.0. The
  laptop's are the same.

**The dispatch** waits for the user's word. Then the laptop does three things:
1. It fetches into the Mac Studio's main checkout, which moves no working tree.
2. It repeats §C.0's checks, removes Part A's worktree and makes the one worktree at this commit,
   detached and with no venv.
3. It starts the Mac Studio orchestrator session in that worktree, through the saved `MacStudio`
   connection, with this note's §C as its brief. The folder must be trusted in the app first.

## 5. Stop and ask the user

- At Part A's planned stop, always, and at the extension's (§A.5).
- At any stop condition of prompt §8 that an agent reports, with the agent's words.
- When a premise this note or the prompt states is found false at a dispatch.
- Before any write outside the paths named here, any deletion, and any push.
