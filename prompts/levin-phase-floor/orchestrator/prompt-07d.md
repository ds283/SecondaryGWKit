# Orchestrator — prompt 07d, the $6\pi$ gate, the Clenshaw–Curtis estimate and the integrand's noise

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/07d-six-pi-gate-and-cc-estimate.md`](../07d-six-pi-gate-and-cc-estimate.md).

**Board item:** 07d. **Model:** Opus, for every agent; you review on whatever model you were started
with.

**Closes, in its records commit, on the user's word,** each under the prompt header's condition:
- `[07-c2-gate-threshold-was-never-measured]`;
- `[06d-cc-nested-estimate-measures-the-coarse-rule]`;
- `[07c-integrand-noise-above-the-cc-floor]`;
- `[07c-chase-stop-acts-on-unreferenced-analytic-calls]`.

Any of the four that does not close is narrowed.

The prompt runs in three parts, as 07b's and 07c's did:
- **Part A (prompt §2), the measurement,** on the Mac Studio: one Claude Code session there, which is
  the measuring agent, in a worktree at the dispatch SHA. It ends at a planned stop. This note's §A.
- **Part B (prompt §1, §3, §4), the code,** on the laptop: a fresh-context subagent in the main
  checkout, one commit. This note's §B.
- **Part C (prompt §5, §6),** on the Mac Studio, through an orchestrator session there: R33 under the
  lock, then the census. This note's §C, completed by an addendum at its dispatch.

## 0. The order, and why

1. **Part A first, because §1 and §3 wait on it.** The estimate, its noise term and the threshold are
   chosen on measurements that do not exist yet. Part A changes nothing in the tree.
2. **At Part A's stop** you relay the agent's tables **as it wrote them**, with your own re-scoring
   beside them (§A.3). The user chooses. You write §1, §3 and §9 from the choice, with the predictions
   it adds, and show the text to the user. It is committed and pushed on their word; that commit is
   Part B's dispatch SHA.
3. **Part B lands before Part C starts.**
4. **Every tree on the Mac Studio is a worktree.** The main checkout there supplies the venv, the
   stores, `var/` and the registry, and never moves while anything runs. Git there is read with
   `--no-optional-locks`. Only the laptop writes git on the Mac Studio: `fetch`, `worktree add` and
   `remove`.
5. **What must not be touched:**
   - A3 v2 and its scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, read-only and kept
     for 08;
   - R29's, R30's, R31's and R32's stores;
   - 07's, 07a's, 07b's and 07c's census and references run directories;
   - `var/levin-phase-floor-07/`, `-07a/`, `-07b/` and `-07c/`.
6. **The Mac Studio is shared.** On 2026-11-21 it was running a large ChamPBH science run. **No part
   is dispatched until the user says the Mac Studio is free.**
   - Part A takes no lock, but it competes for the cores, and its references are hours long.
   - Part C's R33 needs the shared rehearsal lock. Exit 3 from `acquire` means stop and ask (`CLAUDE.md`).

### 0.1 What 07c taught, which this note carries

07a's and 07b's lessons (07c's note §0.1) stand. "Beyond both" means beyond the sum, and a Result line
is checked against its own table. 07c adds:
- **The order in use is 24, not 16** (07c Part A's deviation 5): its labels CC₃₁/CC₁₆ were the audit's.
  The prompt writes ${\rm CC}_{2N-1}$ and ${\rm CC}_N$; check that the agent records $N$ per call.
- **Half of D2's leaves were unresolved,** the reference's stated error above the true error. So the
  prompt fills each stratum to a count of scored leaves (§2.3) rather than drawing a fixed number.
  Check the agent reports every drawn leaf, scored or not.
- **Leaf references need noise-averaging** (07c's deviation 8). On the chase's leaves every
  interpolatory rule disagreed with every other above the floor.
- **D2's population was only the calls already captured** (07c's deviation 9). Its analytic strata held
  13 leaves. This prompt captures by stratum (§2.1 item 1).
- **The references' projection overestimated by a third** (07c Part A's `issues.md` item 9). The census
  ran at 0.995 of its central projection (43,080 s).
- **The LiouvilleGreen draws reach classes the census does not.** Under (S), 26 draw calls became
  unconverged, with no counterpart in the census. The draws are §3 item 7's, and their `converged`
  changes are listed.
- **A clause over "every X" needs its vacuous case written down.** Y9's first clause, read vacuously,
  would have failed on calls with no depth-20 leaf. The census agent fixed the reading in
  `predictions.md` before the run. Ask for the same wherever a prediction quantifies over leaves a call
  may not have.

---

## A. Part A — the measurement, on the Mac Studio

### A.0 Facts checked by the orchestrator at `74ebb54f`, 2026-11-21

Re-check each at dispatch.
- **The code is 07c's** (`67b5af74`): `git diff --stat 67b5af74 74ebb54f -- AdaptiveLevin
  ComputeTargets LiouvilleGreen tools` is empty.
- **`AdaptiveLevin/levin_quadrature.py`** is 6,459 lines, SHA-256 `d141a80b…`. The names, with their
  lines:

  | name | line |
  |---|---|
  | `SIX_PI` | 296 |
  | `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO` | 408 |
  | `_LEVIN_DIRECT_SOLVE_PHASE_SPAN` | 544 |
  | `_LEVIN_CHASE_NOISE_FACTOR` | 688 |
  | `_LEVIN_WINDOW_IROFF1_LIMIT` | 733 |
  | `_DQK21_SATURATION_FACTOR` | 743 |
  | `_levin_G0_G1` | 837 |
  | `_roundoff_floor` | 859 |
  | `_cc_node_term_parts` | 967 |
  | `_cc_roundoff_floor` | 1074 |
  | `_declared_endpoint_phase_err` | 1147 |
  | `_cc_leaf_unresolved` | 1223 |
  | `_window_bisection_counts` | 1248 |
  | `_Basis_SinCos.theta_prime_on_fine_grid` | 2791 |
  | `_adaptive_levin_subregion` | 2842 |
  | `_adaptive_levin_subregion_impl` (the gate at `:3055`) | 2914 |
  | `_adaptive_levin_subregion_cc` | 3355 |
  | `_oriented_cc` | 3688 |
  | `_locate_stationary_points` (it reads no threshold; `:3806` is its docstring's account of the Rule A 07a replaced — corrected at Part A's stop) | 3779 |
  | `_adaptive_levin` | 4157 |
  | `_add_leaf` | 4842 |
  | `_check_leaf` | 4953 |
  | `_floor_limit_stopped_window_leaf` | 5153 |
  | `_stop_window` | 5184 |
  | `adaptive_levin_sincos` | 6182 |

  `ComputeTargets/QuadSourceIntegral.py:71` sets `CHEBYSHEV_ORDER = 24`.
- **On the Mac Studio** (from 07c's records, **not re-checked today**: it was running ChamPBH's science
  run):
  - 07c's census run directory `var/runs/levin-phase-floor-07c-census-20261119T201419/`, checkpoint
    `33d94ce2…`, and `var/levin-phase-floor-07c/census/out/stop3/compare_census_calls.calls.jsonl`
    (`ee36f0d7…`), each in 07c's `census/KEPT_ON_MAC_STUDIO.txt`;
  - 07c Part A's captures and readings, in its `measurement/KEPT_ON_MAC_STUDIO.txt`;
  - R32: `var/datastores/levin-phase-floor-07c-r32*`, run
    `levin-phase-floor-07c-r32-07c-20261119T140417`;
  - Part C's worktree `SecondaryGWKit-lpf07c-cc3368c`, if still registered.
- **The populations** (from committed records):
  - 07c's census: 18,560 units, 289,482 calls;
  - the chase stop acts in 82,172 calls, 57,659 of them analytic
    (`07c/census/out/scores/summary.json`);
  - 07c Part A's regression set: 454 calls, 305 referenced, 299 scored.
- **The references' cost** is not known before the draws. 07c's Part A projected 17.7 h elapsed at two
  readings at a time, and its jobs took 0.65 of 07's model. (Rt) leans to calls with long Levin chains
  above $6\pi$. (Ra) leans to analytic blocks, where Route A applies.

### A.1 Before you dispatch

1. **The dispatch SHA.** The prompt and this note are committed and pushed; that commit is the
   dispatch SHA.
2. **The user's word that the Mac Studio is free** (§0 item 6).
3. **On the Mac Studio, read-only:**
   - the main checkout clean, and where 07c left it;
   - `runregistrykit list` with nothing `running` or stale;
   - the lock free;
   - no `var/levin-phase-floor-07d*`;
   - load and free disk recorded;
   - the files of §A.0 hashed against their `KEPT_ON_MAC_STUDIO.txt` lines;
   - the scratch copy against `copy_files_certified.txt`.
4. **The worktrees:**
   - remove `SecondaryGWKit-lpf07c-cc3368c`, if registered and clean;
   - `git fetch origin` in the Mac Studio main checkout;
   - `git worktree add --detach ../SecondaryGWKit-lpf07d-<dispatch short SHA> <dispatch SHA>`, with no
     venv in it.
5. **The marker:** `touch` a file in your scratchpad, and record the time on the Mac Studio.

### A.2 Dispatch

Start one Claude Code session on the Mac Studio over the saved `MacStudio` SSH connection, in the
worktree, on Opus, with `mcp__ccd_session__start_session`. Its brief: prompt §2 is its whole scope;
§A.0; the dispatch SHA; and these rules:
- **Every command** runs with its cwd in the worktree, calls the main checkout's `venv/bin/python` by its
  absolute path, with `PYTHONPATH` set to the worktree and `-B`. Nothing is written in the worktree.
- **It writes only** `var/levin-phase-floor-07d/measurement/` and its own registered run directories.
  No Ray, no lock, no store but A3 v2's scratch copy, read-only, for its captures.
- **`--cpus 6`**, and at most six worker processes.
- **Anything over ten minutes** runs as a registered detached run, launched, checked once, and waited
  on by one background `kill -0` loop.
- **Projections first:** the references' projection reported to you before any reading. Beyond 36
  hours in all is a stop (prompt §8).
- **Every draw's seed** is written to `state/` before the draw, and every selection saved before any
  reading or solve. The leaf strata are filled in their seeded order, and every drawn leaf is reported.
- **Its records:**
  - `log.md` in README §5.1's form, every deviation numbered and classified;
  - `verification.md`, its part of `VERIFICATION.md` §7d;
  - `issues.md`;
  - `out/`, `scripts/` and `state/`;
  - `KEPT_ON_MAC_STUDIO.txt`, listing every pickle and every file over 2 MB with SHA-256 and size.
- **It stops at prompt §2.8,** and at any of §8's conditions, and reports to the laptop session. It
  recommends nothing.

### A.3 At the stop, before you relay

Read over `ssh macstudio-tunnel`, read-only, with scratch readers of your own:
1. **Scope:** `find var -newer <marker>` lists only the agent's paths; the worktree's
   `git status --short --ignored` is empty; the files of §A.0 and the scratch copy are unchanged.
2. **The replays:** three calls, one of them analytic, re-run at 07c's code yourself, and compared with
   07c's census record in `float.hex`.
3. **The noise (D-N):** on two leaves, $\sigma$ re-measured yourself by the fit residual, and the
   component the agent names as its source re-checked.
4. **The estimate (D-E):**
   - the leaf sample re-drawn from its seed, and the stratum counts checked;
   - on three leaves, the true error recomputed yourself against the agent's reference rule, with each
     candidate's verdict;
   - the agent's cause of understatement checked on two leaves, one by splitting it at its knots
     yourself;
   - each candidate's understatement count re-read from the table's inputs.
5. **The threshold (D-T):** two leaves, one each side of $6\pi$, solved both ways yourself and scored.
6. **The references:**
   - the (Rt) and (Ra) selections re-drawn from their seeds;
   - three readings re-scored in exact `Fraction`s, one each from (Rt), (Ra) and the regression set;
   - the agreement rule applied to all.
7. **The variants:**
   - each diff against the tree touches only what it says;
   - two calls re-run under two variants yourself;
   - **§2.7's qualification re-scored** by your own reader.

### A.4 After the user's choice

**At Part A's stop the user chose a second measurement first** (prompt §9.1), so this section applies at
Part A′'s stop (§A′). Part A's records were copied back in the amendment commit, by the rule of the last
paragraph below.

Write prompt §1, §3 and §9 from the user's words, quoted. They cover:
- the estimate, its noise term and the threshold, with each constant's basis and margin;
- what the change does to the declarations, `converged`, the round-off exit, the window's count and the
  chase stop's leaves;
- the leaves and routes it touches, and the diagnostics;
- the tests and breakages;
- §9's additions to §6's predictions: Z1's untouched counts, Z8's replays, and R2-28's prediction.

**If the chosen noise term reads a quantity from an integrand provider,** the user rules on that scope
at the stop. §9 names every file outside `AdaptiveLevin/`, and §5 states what R33 then moves outside
`QuadSourceIntegral`.

Show the text to the user. On their word commit and push it, with Part A′'s records copied back as
07c's were: no pickles and nothing over 2 MB, by a list made first, each file checked by SHA-256. That
commit is Part B's dispatch SHA.

---

## A′. Part A′ — the second measurement, on the Mac Studio

### A′.0 What it is, and what it inherits

Prompt §9.2, added at Part A's stop on the user's word (§9.1). Its scope is §9.2 and §9.3. Its
inputs are Part A's records on the Mac Studio, read-only:
- `var/levin-phase-floor-07d/measurement/`: its captures, selections, leaf records, call references and
  variant replays;
- its registered runs `levin-phase-floor-07d-*`, all `done`.

The orchestrator's checks at Part A's stop are in `docs/levin-phase-floor/07d/measurement/orchestrator_check/`,
committed. The three facts they found that §9.2 is built on:
- v3's leaf reference understates its error on knotted leaves;
- D-T's halves and quarters ran without the gate;
- `levin_tc` left out the floor.

Give the agent the scripts that show each.

### A′.1 Before you dispatch

1. **The amendment's SHA:** prompt §9 and this section committed and pushed, with Part A's records.
2. **The user's word that the Mac Studio is free** (§0 item 6).
3. **On the Mac Studio, read-only:** as §A.1 item 3, with no `var/levin-phase-floor-07d/measurement-a2`,
   and Part A's records unchanged against its `KEPT_ON_MAC_STUDIO.txt` and the copy-back's SHA-256
   list.
4. **The worktrees:** remove `SecondaryGWKit-lpf07d-49ce964` if clean; fetch; add
   `SecondaryGWKit-lpf07d-<amendment short SHA>` detached at the amendment's SHA, with no venv. Check
   that `git diff --stat 67b5af74 <amendment SHA> -- AdaptiveLevin ComputeTargets LiouvilleGreen tools`
   is empty.
5. **The marker,** as §A.1 item 5.
6. **Part A's session** (`lpf07d-part-a`) is not reused. Archive it only on the user's word.

### A′.2 Dispatch

As §A.2, to a fresh session in the new worktree, with:
- prompt §9.2 and §9.3 as its whole scope, and §9.1 as its background;
- §A′.0;
- Part A's records as its inputs, read-only;
- every write in `var/levin-phase-floor-07d/measurement-a2/` and its own registered run directories.

The projection comes to you before anything long runs.

### A′.3 At the stop, before you relay

As §A.3 items 1, 6 and 7, and:
1. **v4's references:**
   - on four leaves, two of them knotted, re-reference yourself by an independent rule (a different
     Gauss–Legendre order, or a doubling taken one step further);
   - check that each lies within the two references' combined stated errors.
2. **The reproduction of `_check_leaf`:**
   - its check against the instrumented variant re-read from the agent's records;
   - on two leaves, one each side of $12\pi$, the sub-solves' routes at $T = 6\pi$ and $T = 12\pi$
     re-derived yourself from their phase variations.
3. **Item 1's verdicts:** one $T$'s verdict re-scored by your own reader from the per-leaf records,
   with the vacuous cases checked to be named as such.
4. **${\rm CC}_{93}$:** each candidate's understatement count re-read; on three leaves, $s_{\rm own,93}$
   recomputed yourself.
5. **The knots:**
   - on two leaves, the count of each kind re-derived;
   - one jump re-measured yourself;
   - (K)'s bisection points on one call re-derived from the knots.
6. **The cost table:** one leaf's evaluation counts re-derived from the module's counters.

Then relay as §0 item 2, and §A.4 applies.

---

## B. Part B — the code, on the laptop

As 07c's note §B, with:
- **The baselines** at the dispatch SHA. Nothing in `AdaptiveLevin/`, `ComputeTargets/` or
  `LiouvilleGreen/` changed since `67b5af74`.
- **The commit's paths:**
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/` as §3 and §9 name;
  - any provider file §9 names;
  - `docs/levin-phase-floor/07d/code/`;
  - the log `logs/07d-six-pi-gate-and-cc-estimate.md` and the board;
  - `CLAUDE.md`, only for its `black` paragraph.

  **A path outside that list is a stop.**
- **The review's AST check** names these as unchanged, unless §9 changes them:
  - `_roundoff_floor` and `_levin_G0_G1`;
  - `_trust_noise_yardstick`, and `_leaf_rate_rule` with its two constants;
  - the chase stop and its two constants;
  - the location, except where it reads the threshold;
  - the windows' pieces and their stop;
  - `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`, $\Lambda_c$ and `DEFAULT_LEVIN_MAX_DEPTH`;
  - `adaptive_levin_sincos`'s body.
- **Replays at the code commit,** from copies of their captures:
  - one (Ra) call and one route-changed call, each equal to the chosen variant's replay record and
    scored against its reading;
  - R2-28, as §9 predicts;
  - one call the change should not touch, bit-identical to 07c's census.

---

## C. Part C — R33 and the census, on the Mac Studio

### C.0 When it starts

As 07c's note §C.0, with:
- the user's word that the Mac Studio is free;
- no `var/levin-phase-floor-07d/rehearsal` or `census` yet, and no
  `var/datastores/levin-phase-floor-07d-*`;
- R32's store and 07c's census checkpoint `33d94ce2…` intact;
- the scratch copy certified.

The laptop fetches, makes one worktree `SecondaryGWKit-lpf07d-<Part C dispatch short SHA>`, removes
Part A's, and starts the Mac Studio orchestrator session in it with this note's §C as its brief.

### C.1 What the Mac Studio orchestrator is told

As 07c's note §C.1, with R32 as the parent rehearsal (prompt §5 S0′), R33 as the new one, and:
- **Planned stops:** stop 1 after R33 and its comparison with R32; stop 2 after Z0, with the
  predictions written; stop 3 after the census, before any verdict.
- **Before anything else is scored,** a call Z1 says is untouched that differs from 07c's is a stop,
  and so is Z8 failing.
- **No new references.** Z4 is scored on 07b's U4 references, 07c's R4, R5 and R2-28, and this prompt's
  (Rt) and (Ra); Z6 on 06b's $S_{10}$.

### C.2 The review — on the Mac Studio

As 07c's note §C.2, with R33 against R32. Z0–Z8 are re-scored from the two checkpoints by your own
readers:
- Z1's untouched calls, bit for bit at both rtols;
- Z4 on every referenced call;
- Z6 on all 24;
- Z7 over all 2,484 pairs;
- Z8 on every replayed call;
- the route changes and `converged` changes, recounted.

### C.3 What it reports

Pass or stop on the first line. Then:
- R33 and S0′–S4;
- Z0–Z8 with their numbers;
- where the change acted, and the route changes;
- convergence against 07c's and 07b's;
- cost.

### C.4 After the review — on the laptop

As 07c's note §C.4:
1. The copy-back list first, every file checked by SHA-256.
2. Re-checks from the copies:
   - Z4 on one (Rt) call and one (Ra) call;
   - Z6 on two calls;
   - one untouched and one moved rehearsal row, in `float.hex`.
3. Ask the user about the issues and the agents' `issues.md`.
4. Make **one records commit**, pushed on the user's word:
   - the records in `docs/levin-phase-floor/07d/`;
   - the logs appended to `logs/07d-six-pi-gate-and-cc-estimate.md`, with the verdicts;
   - `VERIFICATION.md` §7d;
   - the board, with 07d's row and a Decisions row;
   - the issues, on the user's word;
   - `docs/OPEN_ISSUES.md`, with its count and date corrected by counting;
   - this directory's README row and the campaign README's.

**Kept until the user says otherwise:**
- R29, R30, R31, R32 and R33;
- 07a's, 07b's, 07c's and 07d's census run directories (07d's is 08's "before");
- A3 v2's scratch copy, for 08.

### C.5 Addendum: Part C's dispatch

*Written at Part B's landing.*

## 5. Stop and ask the user

- At Part A's planned stop, always; and at Part A′'s.
- Before any dispatch, for the word that the Mac Studio is free.
- At any stop condition of prompt §8 that an agent reports, with the agent's words.
- When a premise this note or the prompt states is found false at a dispatch.
- Before any write outside the paths named here, any deletion, and any push.
