# Orchestrator — prompt 06, the producers' stored phase: store the residual, reduce exactly

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06-producers-store-the-residual.md`](../06-producers-store-the-residual.md),
as amended before dispatch (its §9: `59d4070`, `7b6c1cd`)
**Board item:** 06 · **Closes, when §5 and §6 hold:** `[02-consumer-phi-below-the-storage-granularity]`
(`phase-representation`; assigned here at G1) · **Model:** Opus, as the prompt recommends, for every
part. You, the orchestrator, review on whatever model you were started with.

The prompt runs in three parts. This note has two, as 05a's had:
- **Part A (prompt §1–§4), the code,** on the laptop: a fresh-context subagent in the **main
  checkout at `HEAD`**, one commit. This note's §A.
- **Part B (prompt §5 and §6), R13 and R14 under the lock, and the re-measurement of 05's 39 rows
  with the producers re-run,** on the Mac Studio, orchestrated there by one session, which
  dispatches two agents. This note's §B.

## 0. The order, and why

1. **Part A first.** R14 and §6's $s$ tree are the code commit. Nothing on the Mac Studio starts
   until Part A has passed its review here and is pushed.
2. **Every tree on the Mac Studio is a worktree** (prompt header). `handover` A4 ran in the Mac
   Studio's main checkout at `28ebaf0` from 15:27 GMT, Ray-free, and that checkout does not move
   for this prompt at all. It still supplies the venv, the store, `var/` and the registry.
   - **The parent tree** ($t_5$, R13) is a detached worktree at the code commit's parent.
   - **The code tree** ($s$, R14) is a detached worktree at Part B's dispatch SHA. That commit
     adds only records, so its code is the code commit's. The Mac Studio orchestrator reads this
     note and the prompt there.
3. **R13 and R14 run one after the other, under the lock**, R13 first. Each is R8's build, about
   3.5 minutes at `--cpus 6`. A4 takes no lock.
4. **§6 takes no lock and uses no Ray**, so its agent may run beside R13 and R14. It reads 05's
   `var/levin-phase-floor-05/wider/state/sample.json` and 05a's run
   `levin-phase-floor-05a-wider-20261026T123047`, which exist only on the Mac Studio.

### 0.1 What 05 and 05a taught, which this note carries

- **The four volatile kinds** of `[05-store-value-diff-compares-volatile-columns]` are named in the
  prompt, and 05's `classify_non_qsi.py` sorts them. Anything it prints as `UNCLASSIFIED` is still
  the stop.
- **"Depth-20 row"** is read as 05a read it: S3 and S4 are over all 39 rows, S5 over the 31 stored
  at depth 20.
- **Every deviation is numbered and classified before an agent reports**, failed attempts included.
- **A question relayed to the user carries the state of the numbers.** In 05a the laptop told the
  user no T1 number existed when both readings had been scored. Before relaying, check what the
  run has already written, and say so.

### 0.2 What changed in the prompt before dispatch

Its §9 lists five amendments. Two bear on the review:
- **R14 must give `1b332ffa…`**, as R13. Any difference is a stop.
- **S1 is scored on pure-WKB windows only.** Each $G$ probe window is tagged by whether every
  source sample's `GkSource` `z_init` is null; numerically seeded windows are reported beside the
  window RMS of their $\Delta\theta$ alone.

---

## A. Part A — the code, on the laptop

### A.0 Facts checked by the orchestrator at `7b6c1cd`

Pass them on.
- **The checkout.** `handover-remedial` at `7b6c1cd`, `git status --short` empty, one worktree
  (itself). The laptop's registry shows nothing `running` or stale (60 runs: 58 finished, 2
  `unknown` pre-registry). Part A launches no run, starts no Ray and opens no production store.
- **The line references** in prompt "Read first" hold: `WKB_phase_function.py:313-314`;
  `GkWKBIntegration.py` `store` `:428`, `deltaTheta` `:504`, `apply_phase_offset` `:516`;
  `TkWKBIntegration.py` `store` `:437`, `:518`, `:532`; `WKBtools.py` `WKB_mod_2pi` `:15`,
  `apply_phase_offset` `:132`; `TkSourceFunctions.py` `_build_WKB` `:478`, `_check_phase_sign`
  `:583`; `primitive_phase.py` `build_phi_samples` `:511`; `ObjectFactories/GkSource.py`'s value
  columns `:885-915`; `check_phases.py` `window_rms` `:106`.
- **The conventions behind §0.2** (amended): `CumulativeTable.delta(a, b)` $= T(b) - T(a)$
  (`cumulative_table.py:420`); `build_phi_samples` forms
  $\theta - {\rm sign}\,k\,L.\delta(z, z_{\rm anchor})$; `GK_PHASE_SIGN = -1`
  (`GkSourcePolicyData.py:142`), anchored at `z_response`; `TK_PHASE_SIGN = +1`
  (`TkSourceFunctions.py:235`), anchored at the producer's `z_init`. `GkSource` stores `z_init`
  null for a pure-WKB value (`GkSource.py:34`).
- **`apply_phase_offset`'s production callers** are the two producers' `store()` alone. Its tests
  are in `test_gk_source_primitive_phase.py` and `test_gk_wkb_phase.py`, and stay.
- **`build_phi_samples`' callers** are `GkSourcePolicyData.py:186` and `TkSourceFunctions.py:517`;
  it is exported by `ComputeTargets/__init__.py:37` and tested in `test_primitive_phase.py:609` and
  `test_gk_source_primitive_phase.py:276`.
- **The two source greps** prompt §2 now names: `test_gk_wkb_phase.py:1150` and
  `test_tk_wkb_phase.py:921`.
- **Modules that name `theta_mod_2pi`**, where stand-ins of a producer's values may live (prompt
  §2's list is "and the like"): in `ComputeTargets/tests/`, `test_tk_source_functions.py`
  (`FakeWKBValue` `:157`, `FakeTkWKB` `:176`), `test_g_phase_reanchor.py` (`_Value` `:644`),
  `test_t_phase_reanchor.py`, `test_phase_groups.py`, `test_quadsource_integral.py`,
  `test_gk_source_primitive_phase.py`, `test_gk_source_policy.py`, `test_gk_wkb_phase.py`,
  `test_tk_wkb_phase.py`, `test_residual_table_reuse.py`, `test_primitive_phase.py`; in
  `ObjectFactories/tests/`, `test_read_path_mechanics.py`, `test_value_reads_parent_key.py`,
  `test_gksource_parent_set.py`. The agent reads each for constructions.
- **The fingerprint** does not digest value columns: each `inventory_spec()` records identities,
  tags, the validated flag and the value-row count (`ObjectFactories/GkWKBIntegration.py:960`).
  `ObjectFactories/tests/test_fingerprint_format_guard.py` pins every spec's digest, and must not
  change.
- **The numeric producer's tolerance**: `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8`
  (`config/defaults.py:86`).
- **`mpmath`** is 1.3.0 in the venv. `TWO_PI = 2.0 * pi` is `LiouvilleGreen/constants.py:3`.
- **The known flake** in the `ComputeTargets` root is `test_tk_wkb_phase …
  test_wall_time_per_object`, a wall-clock assertion. Re-run it alone and record it; it is not a
  stop. Record `uptime` beside each roots run.

### A.1 Before you dispatch

1. Commit this note and `README.md`'s 06 row, and push; that commit is the **dispatch SHA**.
2. **Baselines:** the seven roots serially in the main checkout, into the session scratchpad, at
   the dispatch SHA.
3. **The marker:** `touch` a file in the scratchpad immediately before dispatch.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA,
the baselines and §A.0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Other sessions may commit to this branch while it works. Before staging,
  `git log --oneline <dispatch SHA>..HEAD`; if anything has landed, read it, and merge by reading
  any shared file, never by overwriting it.
- **Its scope is prompt §1–§4,** with §9 part of the prompt. §5 and §6 run later on the Mac Studio,
  and are not its work. It copies §6's predictions, as amended, into its log's **Predictions and
  verdicts** unchanged, and scores nothing.
- **One commit, at the end, staged by explicit path**, not pushed. It may cover:
  - `Quadrature/integrators/WKB_phase_function.py`;
  - `ComputeTargets/GkWKBIntegration.py`, `ComputeTargets/TkWKBIntegration.py`,
    `ComputeTargets/GkSource.py`, `ComputeTargets/GkSourcePolicyData.py`,
    `ComputeTargets/TkSourceFunctions.py`;
  - `ComputeTargets/primitive_phase.py`: a factored helper (prompt §1.4), and `build_phi_samples`
    kept with its docstring corrected, or removed; `ComputeTargets/__init__.py` only for that
    export;
  - `LiouvilleGreen/WKBtools.py`, **docstrings and comments only**;
  - `ObjectFactories/GkWKBIntegration.py`, `ObjectFactories/TkWKBIntegration.py`,
    `ObjectFactories/GkSource.py`: the `theta_residual` column, its write and its read, and
    nothing else;
  - `tools/store_value_diff.py` and a test in `tools/tests/`, only under prompt §1.5;
  - `ComputeTargets/tests/test_stored_phase_residual.py`, new;
  - in existing modules under `ComputeTargets/tests/` and `ObjectFactories/tests/`: the residual on
    a stand-in (prompt §2), an assertion under §2's two exceptions, and nothing else;
  - `prompts/levin-phase-floor/logs/06-producers-store-the-residual.md` and this campaign's board;
  - `docs/OPEN_ISSUES.md` if the commit opens an issue.

  A path outside that list is a stop. `cumulative_table.py`, `BackgroundModel.py`,
  `phase_residual.py`, `GkNumericIntegration.py`, `TkNumericIntegration.py`, `QuadSource.py`,
  `QuadSourceIntegral.py`, `phase_groups.py`, `AdaptiveLevin/`, `StoreFixtures/`, `config/` and
  `main.py` in particular are not touched.
- **No fallback.** A consumer that finds `theta_residual` null raises; it does not fall back to the
  old construction. Old stores are read at their own SHA (`CLAUDE.md`; no version-aware readers).
- **The header's issue does not move in this commit** (prompt §4).
- **No Ray, and no store but the fixtures'** in its tests and its checks. It writes nothing under
  `var/`.
- **Every deviation is numbered and classified** in the log before it reports.
- **Stop and ask** at any of prompt §8's conditions that apply to §1–§4.

### A.3 The review — nine checks, on the laptop

1. **Scope.** `git show --stat <code commit>`: only §A.2's paths. `git diff --stat <dispatch SHA>
   <code commit> --` on §A.2's untouched list is empty. `git status --short` is clean, and
   `find var -newer <marker>` lists nothing of the agent's.
2. **Nothing else moved.**
   - `LiouvilleGreen/WKBtools.py`'s AST, docstrings removed, is identical at the dispatch SHA and
     the code commit.
   - Each `inventory_spec()` is unchanged, and `test_fingerprint_format_guard` passes untouched.
   - Prompt §3 test 6 (a)–(c) exists and passes. If a helper was factored out of
     `PrimitivePhase`, (a) compares before and after on the same $\varphi$ samples, bit for bit.
3. **The construction, read** against prompt §0.2 and §1:
   - the large term $k\,L.\delta$ is formed from node limbs and partials with $k$ as a `Fraction`,
     and the sum reduced against `mpmath`'s $2\pi$ at no fewer than 40 digits, $\Delta\theta$
     inside it;
   - `theta_residual` is formed from $\Delta\theta$ and $\rho.\delta$ alone: no expression in the
     producer subtracts a value of size $k\tau$ to make it;
   - no cross-sample rebase; `G_WKB` and `T_WKB` from the exact remainder;
   - `_build_phase` and `_build_WKB` build $\varphi$ from $r$; $\varphi_G$'s second term is
     $k\,\tau.\delta(z', z_{\rm init})$, zero where `z_init` is null; in `_build_phase` the unwrap
     of $\varphi_G$ is compared with the stored rectifier shifts, and a disagreement prints one
     `!! WARNING` line and uses $\varphi_G$'s own unwrap (prompt §9 point 6);
   - the exact reduction in `WKB_phase_function` is integer arithmetic, checked by test 1 against
     `mpmath`; the wall-time tests' limits did not move; a remainder rounding to $-$`TWO_PI` is
     stored as $0$ with the cycle count raised;
   - no fallback on a null `theta_residual`;
   - the log names `apply_phase_offset`'s remaining callers and says where $\varphi_G$ is formed.
4. **The roots, re-run here.** Six equal to the baselines (`tools` up by the §1.5 test if one was
   added). `ComputeTargets` up by exactly the test methods the new module adds, counted from the
   diff. All OK, with the flake re-run alone.
5. **The tests, read.** Prompt §3's tests 1–7 each exist, named in the log, at the prompt's $k$,
   bounds and fixtures. Test 1's and 2's references are in `mpmath` from the same limbs, partials,
   $\rho$ and $\Delta\theta$. Test 3 uses 04's `window_rms` and records the old construction's
   metric. Test 4 builds at least two wraps. Run the new module alone yourself, with `-v`, and keep
   its printed numbers.
6. **The breakages.** (i)–(v) each name the test that failed and were reverted. If a record is
   thin, re-apply that breakage yourself in the working tree, run the one module, and revert with
   `git checkout -- <file>`; `git status --short` is clean afterwards. **Re-apply (ii) and (iii)
   yourself in any case**: they test what is new in production, the stored residual and $G$'s
   consumer.
7. **The fixtures and assertions.** Every stand-in site is listed in the log by file and line, its
   residual formed from its small terms. Check the diff of each existing module for changed
   `assert`/`self.assert` lines: each is either a pinned value prompt §2 lets move, listed with old
   and new value and its bound, or one of the two source greps, whose `assertIn` now names the
   replacement and whose `assertNotIn` lines are unchanged. No producer source keeps the string
   `apply_phase_offset` in a comment to satisfy the old grep.
8. **The records.**
   - The log has README §5.1's headings and classifies every deviation.
   - Its **Predictions and verdicts** holds prompt §6's predictions as amended, word for word (check
     by substring), marked not yet scored.
   - The board's 06 row says the code landed and §5–§6 are pending, with no commit hash yet. The
     header's issue has not moved, and `docs/OPEN_ISSUES.md` changed only if an issue was opened,
     with count and date correct by counting.
   - `black --check` is clean on every `.py` file touched. The commit message follows `CLAUDE.md`'s
     form.
9. **Nothing else landed unaccounted.** `git log --oneline <dispatch SHA>..HEAD` shows the code
   commit and anything another session landed, each accounted for.

### A.4 Landing

If the review passes: push the code commit. Then, in one residue commit of the orchestrator's own,
the code commit's hash on the board's 06 row and in the log, and §B.5's addendum below; that commit
is Part B's dispatch SHA. Push it.

---

## B. Part B — R13, R14 and the re-measurement, on the Mac Studio

### B.0 When it starts

Only when all hold, each checked by the laptop session over `ssh macstudio-tunnel`, read-only:
- Part A has landed, and the residue commit is pushed;
- **A4 is over:** its run `handover-04-cost-growth-20261026T152704` is `done` (or otherwise
  finished) and its Mac Studio orchestrator, `handover-a4`, has reported. A4's review checks what
  `var/` gained; Part B must not write there while it runs. If A4 is not over, wait for it, and say
  so to the user;
- on the Mac Studio: the main checkout's `git status --short` is empty; `runregistrykit list` shows
  nothing `running` or stale; the lock is free; there is no `var/levin-phase-floor-06/` and no
  `var/datastores/levin-phase-floor-06-*`;
- 05's `sample.json` still has SHA-256 `73aa7876…`, and 05a's wider run directory is intact.

Then the laptop, the only party that writes git on the Mac Studio:
1. fetches in the main checkout (`git fetch origin`), which moves no working tree;
2. makes two detached worktrees, with no venv in either:
   `git worktree add --detach ../SecondaryGWKit-lpf06-<parent short SHA> <parent SHA>` and
   `git worktree add --detach ../SecondaryGWKit-lpf06-<dispatch short SHA> <Part B dispatch SHA>`;
3. starts the Mac Studio orchestrator, a Claude Code session over the saved `MacStudio` SSH
   connection, on this note's §B and the addendum in §B.5, read from the code worktree.

### B.1 Facts, checked 2026-10-26 at 15:43 GMT, read-only

Re-checked at Part B's dispatch and corrected in the addendum where they have moved.
- **The main checkout** is `handover-remedial` at `28ebaf0`, `git status --short` empty, one
  worktree (itself). Python 3.13.16. Load about 7.4 on 10 cores, 117 GiB free.
- **The registry:** 20 runs, one alive: A4's `handover-04-cost-growth-20261026T152704`, 55/60 at
  16 minutes. 05a's three runs are `done`.
- **The lock is free:** `status` says `~/.raypoolkit/rehearsal-lock` does not exist, and
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none. (`status` takes `--ray-temp-dir` only,
  not `--project`.)
- **`var/`** holds `datastores`, `handover-03`, `-04`, `-06`, `-07`, `-08`, `-10`, `-11`, `-12`,
  `levin-phase-floor-01`, `-03`, `-04`, `-05`, `-05a`, `levin-phase-floor-measure`, `py313-venv-02`
  and `runs`; no `levin-phase-floor-06`.
- **05's sample:** `var/levin-phase-floor-05/wider/state/sample.json`, SHA-256
  `73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7`.
- **05a's harness** is `docs/levin-phase-floor/05a/wider/scripts/` in the repository, byte-identical
  to the run, and `var/levin-phase-floor-05a/wider/scripts/` there. 05a's run directory
  `var/runs/levin-phase-floor-05a-wider-20261026T123047/` holds U0's reference $t$ units.
- **A3 v2** is `var/datastores/handover-A3-v2-lambdacdm.sqlite`, digest
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **R8's fingerprint**, which R13 and R14 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R12 gave it.
- **U0's first clause** holds at `7b6c1cd`: `git diff --stat 1fc349c 7b6c1cd -- . ':!docs'
  ':!prompts'` shows `CLAUDE.md`, `ComputeTargets/primitive_phase.py` and
  `ComputeTargets/tests/test_g_phase_reanchor.py`, the last two `bd38e50`'s comments.

### B.2 What the Mac Studio orchestrator is told

**Where things are.**
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at whatever `HEAD` it has
  (`28ebaf0` at writing). It holds the venv, the store, `var/` and the registry; the records go in
  its `var/levin-phase-floor-06/`. **No tree runs here, and it is not moved.**
- **The parent worktree,** `…/SecondaryGWKit-lpf06-<parent>`: R13 and §6's $t_5$.
- **The code worktree,** `…/SecondaryGWKit-lpf06-<dispatch>`: R14, §6's $s$, the value comparison
  (in case §1.5 extended the tool) and the readers. This note and the prompt are read here.
- **In both worktrees,** every command calls the main checkout's `venv/bin/python` by its absolute
  path, with `PYTHONPATH` set to that worktree and `-B` or `PYTHONDONTWRITEBYTECODE=1`. Nothing is
  written in either.
- **The stores' paths are absolute**, into the main checkout's `var/datastores/`, so that they
  outlive the worktrees. Run directories are the main checkout's `var/runs/`.
- **No git command that writes**, in any tree, `fetch` included. The laptop moves them.

**Two agents, each fresh-context, on Opus.** Give each the prompt (its own section is its scope,
with §9), Part B's dispatch SHA, both worktrees' paths, §B.1 and the addendum. Neither reads
`orchestrator/`. They may run at the same time.
- **The rehearsal agent,** prompt §5. It writes only under `var/levin-phase-floor-06/rehearsal/`,
  the two stores `var/datastores/levin-phase-floor-06-r13*` and `-r14*`, their run directories and
  launch logs, the lock and `/tmp/ray-sgk`.
  - **The launches are R11's and R12's** (`docs/levin-phase-floor/05a/rehearsal/out/commands.log`),
    with `--database`, `--job-name levin-phase-floor-06`, `--register` (`r13-parent`,
    `r14-residual`), `--campaign levin-phase-floor --prompt 06`, and a `--purpose` that names the
    host, the tree's SHA and the code commit, and says the fingerprint must equal `1b332ffa…`.
    `--cpus 6`, never more.
  - **The lock:** `acquire` once before R13, `set-pid` with each driver's Python pid, held through
    the comparison and the readers, `release` once `ray-processes` is empty. Each lock command is a
    shell command of its own, output by `>>`, never a pipe.
  - **The comparison** is `tools.store_value_diff` run from the code worktree. 05's
    `classify_non_qsi.py` and `tabulate_qsi.py` are copied unchanged from
    `docs/levin-phase-floor/05/rehearsal/scripts/` into `scripts/`, and recorded by SHA-256 beside
    05's. The per-class tabulation of prompt §5.4 inside the five classes is a new script of the
    agent's, saved in `scripts/`.
  - **If R13 and R14 are value-identical in `GkWKBIntegration`'s values, stop:** then at least one
    tree did not run the code it names.
  - **The repairs' agreement:** `grep` R14's launch log and stdout for `_build_phase`'s
    `!! WARNING` disagreement line (prompt §0.2, §9 point 6). Any is §8's stop; report each with
    its $k$, $z_R$ and samples. Record the count, zero included.
  - **The readers** on R14, with the store's files hashed before and after.
- **The re-measurement agent,** prompt §6. It writes only under `var/levin-phase-floor-06/wider/`
  and its own run directory. It uses no Ray and takes no lock.
  - **The harness** is 05a's, copied from `docs/levin-phase-floor/05a/wider/scripts/` into
    `wider/scripts/`. The diff against it is saved in `out/`, and each change is a deviation.
  - **The sample** is 05's `sample.json`, copied unchanged into `state/`, its SHA-256 checked equal
    to `73aa7876…` before any row runs.
  - **U0** runs before the launch, all four clauses, the producer re-run's faithfulness at the
    parent among them.
  - **The predictions** go in `state/predictions.md`, prompt §6's text as amended, unchanged, after
    U0 passes and before the launch. Their mtime is the review's evidence.
  - **Every checkpoint record** carries the script's SHA-256 and its tree's `HEAD` and
    `git status --short`, and each unit's evidence of where its modules were loaded from.
  - **The budget** is raised from the dispatcher, as 05's was, never by a signal inside `mpmath`.
  - **The probe** reads phase objects and the re-run producers' $\Delta\theta$ only. Its windows,
    samples, each window's pure-WKB tag and the `z_init` it was read from are saved per (row,
    sub-interval, factor, tree), so that S1 and S2 can be re-scored from them.
  - **`convergence_kinds.py`** is copied unchanged from `prompts/levin-phase-floor/measure/`, its
    SHA-256 recorded, and run under $s$ as 05a ran it.
- **Both agents:** `HEAD` is not theirs. They save every command and its output; the review reads
  the outputs, not the log. **Launch, one check, end the turn:** report the run id, the run
  directory, the detached Python pid, the `sh` pid, and the first checkpoint line. You wait on each
  pid once, in the background, and resume the agent with the final `status.json`. **Every deviation
  is numbered and classified before an agent reports**, failed attempts included. They stop and ask
  at any of prompt §8's conditions.
- **Each agent's records** are `log.md` (its section, for appending to the code log),
  `verification.md` (its part of `VERIFICATION.md` §6 "The stored phase"), `issues.md`, and `out/`,
  `scripts/` and `state/`.

**The marker:** `touch` a file in the session scratchpad immediately before the first dispatch.

### B.3 The review — on the Mac Studio, after both agents report

1. **Scope.** All three trees: `git status --short` empty and each `HEAD` as dispatched (the main
   checkout's as it was). `find var -newer <marker>` lists only §B.2's paths. `find <each worktree>
   -newer <marker> -not -path '*/__pycache__/*'` lists nothing. A3 v2's files and sidecar are
   unchanged, and so are 05's and 05a's `var/` directories and run directories.
2. **The runs.** `runregistrykit list` shows R13, R14 and `wider` `done`. Each manifest names
   `levin-phase-floor`, `06` and its slug, `git_dirty: false`, and the SHA of the tree it ran from:
   R13 the parent, R14 the dispatch SHA.
3. **The fingerprints.** R13 and R14's `status.json` both record `1b332ffa…`. Re-run `store
   fingerprint` on each yourself, without `--write`.
4. **The values.** Re-run `tools.store_value_diff` on R13 and R14 yourself, from the code worktree,
   with `--json`, into your scratchpad. It reproduces the saved JSON. Run the copied
   `classify_non_qsi.py` on yours: no `UNCLASSIFIED`; every value table outside the five classes of
   prompt §5.4 identical; no row in one store only; `theta_residual` reported in R14 only.
   Spot-check by a scratch reader (`sqlite3` with `mode=ro`): one changed `GkWKBValue` row, its
   `theta_mod_2pi` in both stores in `float.hex`, against the tabulation, and R14's
   `theta_residual` in $(-\pi - 2\times10^{-3}, \pi + 2\times10^{-3})$; one changed
   `QuadSourceIntegral` total likewise. Every `theta_div_2pi` change lies within the amended §5.4
   bound, or is listed on its own.
5. **The readers and the lock.** The readers all exit 0, and R14's files are identical before and
   after. The lock transcript shows `acquire` 0, `set-pid` 0 with each driver's pid, `release` 0.
   `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing and `status` says free, each run by
   you as a command of its own.
6. **U0 and the store.** U0's four clauses in the saved output, compared with `==` and bit
   patterns, never `isclose`; U0 ran before the launch. **Independently,** by a scratch reader of
   the two `checkpoint.jsonl`s, compare one depth-20 row's $t_5$ unit at `1e-7` with 05a's $t$
   unit: `total`, `total_abserr`, and every call's value and abserr, in `float.hex`. The store
   files' hashes are identical before and after, and `store fingerprint`, run by you, gives
   `0e32f24a…`. No journal, `-wal` or `-shm` file beside any store file.
7. **The trees, read.** The $t_5$ units import this repository's packages from the parent worktree
   and the $s$ units from the code worktree: check `PYTHONPATH` in the launch and one module's
   `__file__` in a saved output. Under $s$, the re-run producer values carry `theta_residual`;
   under $t_5$ they do not exist.
8. **The sample and the predictions.** `state/sample.json`'s SHA-256 is `73aa7876…`.
   `state/predictions.md` is later than U0's output and earlier than the run's `manifest.json`, and
   quotes prompt §6 as amended, word for word. The harness diff against 05a's copy matches the
   deviations.
9. **The verdicts, re-scored.** From the checkpoints and the probe's saved windows, by a scratch
   reader: S1–S5, each holds, fails or undecided against prompt §6's rule as written, with its
   numbers. **S1's tags re-derived** from the saved `z_init`s: a window is pure-WKB only if all 9
   are null. Over-budget units counted as the rules say. The descriptive items are present,
   `convergence_kinds.py`'s output and each numerically seeded window's $\Delta\theta$ RMS among
   them.
10. **The records.** Each `log.md` follows campaign README §5.1 and classifies every deviation, the
    worktrees among them. Each `verification.md` is written as part of `VERIFICATION.md` §6, with
    its links relative to `docs/levin-phase-floor/` in the `06/…` form. Each `issues.md` exists.

Write your verdict on each check to `var/levin-phase-floor-06/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict (pass or stop),
then R13 and R14's fingerprints, the value comparison's summary per class, U0, S1–S5 with their
numbers (S1 with its count of pure-WKB and numerically seeded windows), and the over-budget units.

### B.4 After the review — on the laptop

The laptop copies `var/levin-phase-floor-06/` back with `rsync -a`, without checkpoints or any CSV.
It re-checks over `ssh macstudio-tunnel` itself: check 4's `GkWKBValue` spot-check on one row, and
check 6's comparison with 05a's $t$ unit on one row. Then, in **one records commit**, pushed:
- the records to `docs/levin-phase-floor/06/`, scripts byte-identical and unformatted, with their
  directories added to `CLAUDE.md`'s `black` paragraph if any output records their SHA-256;
- both `log.md`s appended to `logs/06-producers-store-the-residual.md`, and the verdicts into its
  Predictions and verdicts;
- both `verification.md`s as `docs/levin-phase-floor/VERIFICATION.md` §6 "The stored phase";
- the board: 06's row, and anything `issues.md` opens;
- **the header's issue.** If §5 holds and S1, S2 and S3 hold, it is closed: on the
  `phase-representation` board's §4, and its row deleted from `docs/OPEN_ISSUES.md`. If any of
  those fails or is undecided, **ask the user** before closing it. S4 and S5 failing do not by
  themselves keep it open: S4 measures the rows before this prompt, and S5 is 05a's T4 again;
  each is recorded as a finding (an issue if the agent proposes one). The numerically seeded
  windows' noise, if the agent proposes an issue for it, is opened as one;
- `docs/OPEN_ISSUES.md` to match, count and date corrected by counting; this directory's README row
  for 06.

R13 and R14's stores are the user's to keep or retire; ask. The worktrees are removed with `git
worktree remove` on the Mac Studio once the records are committed and the user agrees.

### B.5 Addendum: Part B's dispatch

*Written in the residue commit (§A.4).*

---

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §8 conditions, or any failed check in §A.3 or §B.3:
report it, do not repair it. **Never write to A3 v2** or to 05's or 05a's records on the Mac Studio,
never move the Mac Studio's main checkout, and never end another session's run.
