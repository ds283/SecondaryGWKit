# Orchestrator — prompt 05a, layer 1(a) for $T$: the node-anchored remainder, against $2\pi$

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/05a-reanchor-t-phase.md`](../05a-reanchor-t-phase.md)
**Board item:** 05a · **Closes, when §5 and §6 hold:** `[05-t-remainder-keeps-the-anchoring-floor]`
and `[04-g-phase-is-reduced-against-the-double-two-pi]` (both this campaign; 05 did the second's
$G$ half) · **Model:** Opus, as the prompt recommends, for every part. You, the orchestrator,
review on whatever model you were started with.

The prompt runs in three parts. This note has two, as 05's had:
- **Part A (prompt §1–§4), the code,** on the laptop: a fresh-context subagent in the **main
  checkout at `HEAD`**, one commit. This note's §A.
- **Part B (prompt §5 and §6), R11 and R12 under the lock, and the re-measurement of 05's 39 rows,**
  on the Mac Studio, orchestrated there by one session, which dispatches two agents. This note's §B.

## 0. The order, and why

1. **Part A first.** R11 and R12 compare the code commit with its parent, and §6's $t$ tree is the
   code commit. Nothing on the Mac Studio starts until Part A has passed its review here, and is
   pushed.
2. **R11 and §6's $c$ need the code commit's parent**, so they run from a **detached worktree on the
   Mac Studio** at that SHA, which the laptop makes (§B.0). R12 and $t$ run from the Mac Studio's
   main checkout, which the laptop fast-forwards to Part B's dispatch SHA. That commit adds only
   records, so its code is the code commit's.
3. **R11 and R12 run one after the other, under the lock**, R11 first. Each is R8's build, about
   3.5 minutes at `--cpus 6`.
4. **§6 takes no lock and uses no Ray**, so its agent may run beside R11 and R12 (prompt, "Where it
   runs"). It reads 05's `var/levin-phase-floor-05/wider/state/sample.json` and 05's run
   `levin-phase-floor-05-wider-20261026T020005`, which exist only on the Mac Studio.

### 0.1 What 05 taught, which this note carries

- **§8's value stop** fired on four kinds of column that are not values
  (`[05-store-value-diff-compares-volatile-columns]`). The prompt now names them, and 05's
  `classify_non_qsi.py` sorts them. Anything it prints as `UNCLASSIFIED` is still the stop.
- **"Depth-20 row"** had two readings, and the agent stopped on it. The prompt now names both, and
  each scored rule names its own. T2 and T3 are over all 39 rows; T4 is over the 31 stored at
  depth 20.
- **The Mac Studio review stopped on two unclassified deviations.** Each agent is told that every
  deviation is numbered and classified before it reports, the harness's failed attempts included.

---

## A. Part A — the code, on the laptop

### A.0 Facts checked by the orchestrator at `a2ad0db`

Pass them on.
- **The checkout.** `handover-remedial` at `a2ad0db` (the commit that wrote the prompt; `origin` is
  at `e88bdf9`), `git status --short` empty, one worktree (itself). The laptop's registry shows
  nothing `running` or stale (60 runs: 58 finished, 2 `unknown` pre-registry). Part A launches no
  run, starts no Ray and opens no production store.
- **The line references** in prompt "Read first" hold at `a2ad0db`:
  - `primitive_phase.py`: the docstring's "Anchoring, and the floor it implies" `:56` and "Who uses
    which" `:89`; the constructor's `exact_remainder` branch `:296-321` (the anchor's
    `end_partial` at `:318`, `_anchor_exact` at `:321`); `raw_theta` `:376`; `theta_mod_2pi`
    `:385`; `theta_deriv` `:429`; the `exact_remainder` property `:471`;
  - `TkSourceFunctions.py`: `TK_PHASE_SIGN` `:214` with its comment above; `_sound_horizon_rate`
    `:235`; `__init__` `:276`; `_build_WKB` `:445`, its `PrimitivePhase(` at `:494`;
    `handover_continuity` `:610`, its `theta_mod_2pi` at `:630`; `T_WKB` `:772`, its
    `theta_mod_2pi` at `:780`;
  - `BackgroundModel.py:450-461`, the `CumulativeTable` and its `shifted(cs_tau_init)`;
    `handover.py:61-70`, `handover_switch_node`, whose docstring says `z_init` is a root;
  - `QuadSourceIntegral.py:623`, `Tk_functions_builder=TkSourceFunctions`, and the docstring's
    "production never passes it" at `:637-639`;
  - `test_g_phase_reanchor.py`: `_TableCsTauFixture` `:446`;
    `test_transfer_function_remainders_are_unchanged` `:511`.
- **The fixtures that build a `TkSourceFunctions`** in the test roots: the `ClosedFormPrimitive`
  `cs_tau` at `test_tk_source_functions.py:128`; `test_quadsource_integral.py:425`'s `Tk_builder`,
  which calls `test_tk_source_functions` through the alias `tk_fixtures` (and `:301` patches it);
  `test_phase_groups.py:653`. `test_quadsource_parent_main.py` and `test_primitive_phase.py` name
  the class but were not read for constructions; the agent checks them. Scripts under `docs/`
  that build one are records, and are not edited.
- **05's code** is `120fdfe`. `git diff --stat 120fdfe a2ad0db -- . ':!docs' ':!prompts'` shows
  `CLAUDE.md` alone, so prompt §6's T0 first clause holds at `a2ad0db`.
- **The helpers.** `WKB_mod_2pi` is `LiouvilleGreen/WKBtools.py:15`; `TWO_PI = 2.0 * pi` is
  `LiouvilleGreen/constants.py:3`. `mpmath` is 1.3.0 in the venv.
- **The known flake** in the `ComputeTargets` root is `test_tk_wkb_phase …
  test_wall_time_per_object`, a wall-clock assertion. Re-run it alone and record it; it is not a
  stop. Record `uptime` beside each roots run: the laptop's load was 8–18 when this note was
  written.

### A.1 Before you dispatch

1. Commit this note and `README.md`'s 05a row, and push; that commit is the **dispatch SHA**.
2. **Baselines:** the seven roots serially in the main checkout, into the session scratchpad. They
   were taken at `a2ad0db`, whose code the dispatch SHA shares (it adds only this note).
3. **The marker:** `touch` a file in the scratchpad immediately before dispatch.

### A.2 Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA,
the baselines and §A.0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Other sessions may commit to this branch while it works. Before staging,
  `git log --oneline <dispatch SHA>..HEAD`; if anything has landed, read it, and merge by reading
  any shared file, never by overwriting it.
- **Its scope is prompt §1–§4.** §5 and §6 run later on the Mac Studio, and are not its work. It
  copies §6's predictions into its log's **Predictions and verdicts** unchanged (prompt §4), and
  scores nothing.
- **One commit, at the end, staged by explicit path**, not pushed. It covers:
  - `ComputeTargets/TkSourceFunctions.py`;
  - `ComputeTargets/primitive_phase.py`, **the module docstring's "Who uses which" only**;
  - `ComputeTargets/tests/test_t_phase_reanchor.py`, new;
  - in `ComputeTargets/tests/test_g_phase_reanchor.py`, the one test prompt §2 replaces, and
    nothing else;
  - in the other modules under `ComputeTargets/tests/`, `exact_remainder=False` at a
    `ClosedFormPrimitive` construction, and nothing else;
  - `prompts/levin-phase-floor/logs/05a-reanchor-t-phase.md` and this campaign's board;
  - `docs/OPEN_ISSUES.md` if the commit opens an issue.

  A path outside that list is a stop. `GkSourcePolicyData.py`, `cumulative_table.py`,
  `BackgroundModel.py`, `QuadSourceIntegral.py`, `phase_groups.py`, `spline_wrappers.py`,
  `handover.py`, `AdaptiveLevin/`, `LiouvilleGreen/`, `ObjectFactories/`, `StoreFixtures/`,
  `config/`, `tools/` and `main.py` in particular are not touched.
- **The header's issues do not move in this commit** (prompt §4). They move in 05a's records commit,
  after §5 and §6.
- **No Ray, and no store but the fixtures'** in its tests and its checks. It writes nothing under
  `var/`.
- **Every deviation is numbered and classified** in the log before it reports (§0.1).
- **Stop and ask** at any of prompt §8's conditions that apply to §1–§4.

### A.3 The review — nine checks, on the laptop

1. **Scope.** `git show --stat <code commit>`: only §A.2's paths. `git diff --stat <dispatch SHA>
   <code commit> -- ComputeTargets/GkSourcePolicyData.py ComputeTargets/cumulative_table.py
   ComputeTargets/BackgroundModel.py ComputeTargets/QuadSourceIntegral.py
   ComputeTargets/phase_groups.py ComputeTargets/spline_wrappers.py ComputeTargets/handover.py
   AdaptiveLevin LiouvilleGreen ObjectFactories StoreFixtures config tools main.py` is empty.
   `git status --short` is clean, and `find var -newer <marker>` lists nothing of the agent's.
2. **`primitive_phase.py`'s code did not move.** The module's AST, with its module docstring
   removed, is identical (`ast.dump`) at the dispatch SHA and the code commit. Breakage (iv) was
   reverted.
3. **The keyword, read** against prompt §0.3:
   - `exact_remainder: bool = True` on `TkSourceFunctions.__init__`, documented, stored, and passed
     to the `PrimitivePhase` of `_build_WKB` and nowhere else;
   - no fallback: no `try`, `hasattr`, `isinstance` or other test of the primitive that turns it
     off. A primitive without node accessors raises `PrimitivePhase`'s own `ValueError`;
   - `raw_theta`, `theta_deriv`, `T_WKB`'s arithmetic and `handover_continuity` are not edited;
     the `PrimitivePhase` keeps its other arguments;
   - `QuadSourceIntegral`'s call of `Tk_functions_builder` is unchanged, so production takes the
     default;
   - the docstrings and the `TK_PHASE_SIGN` comment say what is now true, and "Who uses which"
     cites the user's decision of 2026-10-26 and $T$'s non-zero end partial.
4. **The roots, re-run here.** Five equal to the baselines. `ComputeTargets` up by exactly the test
   methods the new module adds, less the one replaced (net of a renamed method, counted from the
   diff). All OK, with the flake re-run alone.
5. **The tests, read.** Prompt §3's tests 1–6 each exist, named in the log.
   - Test 1 has all three clauses, the raise naming the missing accessors.
   - Test 2's reference is in `mpmath` from the same limbs, $p_R$, $p_z$ and $\phi$, against
     $2\pi$, with the bound $4\varepsilon(k|p_z| + 2\pi)$, at the four $k$, with `z_init` on a node
     and off one. **The off-node case has $p_R \neq 0$**, shown in the test or its output.
   - Test 3 samples 65 doubles **in $z$** with `x_is_log=False`, fits in $z$ (`numpy.unwrap`,
     degree 4, RMS), at a recorded $z$ with $k|\Delta_{\tau_s}| \ge 10^9$, with both thresholds,
     and records the $u$-sampled $\sigma_\theta$ beside $\varepsilon\frac{z}{1+z}|d\theta_T/du|$.
   - Test 4 (a)–(c), (b) in the replaced test's place with a note saying so.
   - Test 5's bound is the prompt's, amplitude times the old error plus $10^{-14}$ relative.
   - Test 6 uses `test_phase_groups.py`'s realistic fixture in `Gqr`, with 05's
     `_ProductionPhaseGk`, and states both bounds.
   - Run the new module alone yourself, with `-v`, and keep its printed numbers.
6. **The breakages.** (i)–(iv) each name the test that failed and were reverted. If a record is
   thin, re-apply that breakage yourself in the working tree, run the one module, and revert with
   `git checkout -- <file>`; `git status --short` is clean afterwards. **Re-apply (iv) yourself in
   any case**: it is the one that tests what is new in production.
7. **The fixtures.** Every site changed is listed in the log by file and line, and each is a
   `ClosedFormPrimitive` `cs_tau`. No assertion in an existing module changed except the one
   replaced: check the diff of each existing module for changed `assert`/`self.assert` lines.
8. **The records.**
   - The log has README §5.1's headings and classifies every deviation.
   - Its **Predictions and verdicts** holds prompt §6's predictions word for word (check by
     substring), marked not yet scored.
   - The board's 05a row says the code landed and §5–§6 are pending, with no commit hash yet (that
     is residue). The header's issues have not moved, and `docs/OPEN_ISSUES.md` changed only if an
     issue was opened, with count and date correct by counting.
   - `black --check` is clean on every `.py` file touched. The commit message follows `CLAUDE.md`'s
     form.
9. **Nothing else moved.** `git log --oneline <dispatch SHA>..HEAD` shows the code commit and
   anything another session landed, each accounted for.

### A.4 Landing

If the review passes: push the code commit. Then, in one residue commit of the orchestrator's own,
the code commit's hash on the board's 05a row and in the log, and §B.5's addendum below; that commit
is Part B's dispatch SHA. Push it.

---

## B. Part B — R11, R12 and the re-measurement, on the Mac Studio

### B.0 When it starts

Only when all hold, each checked by the laptop session over `ssh macstudio-tunnel`, read-only:
- Part A has landed, and the residue commit is pushed;
- on the Mac Studio: `git status --short` is empty; `runregistrykit list` shows nothing `running` or
  stale; the lock is free; there is no `var/levin-phase-floor-05a/` and no
  `var/datastores/levin-phase-floor-05a-*`;
- 05's `sample.json` still has SHA-256 `73aa7876…`, and 05's wider run directory is intact.

Then the laptop, the only party that writes git on the Mac Studio:
1. fetches and fast-forwards the main checkout to Part B's dispatch SHA (`git merge --ff-only`);
2. makes the worktree for the parent, detached:
   `git worktree add --detach ../SecondaryGWKit-lpf05a-<parent short SHA> <parent SHA>`, with no
   venv in it;
3. starts the Mac Studio orchestrator, a Claude Code session over the saved `MacStudio` SSH
   connection, on this note's §B and the addendum in §B.5.

### B.1 Facts, checked 2026-10-26 at 11:33 GMT, read-only

Re-checked at Part B's dispatch and corrected in the addendum where they have moved.
- **The checkout** is `handover-remedial` at `53084dc`, `git status --short` empty, one worktree
  (itself): 05's worktree has been removed. Python 3.13.16.
- **The registry:** 16 runs, all finished. The newest are 05's three:
  `levin-phase-floor-05-wider-20261026T020005` (109/109), `…-05-r10-reanchor-20261026T015120` and
  `…-05-r9-parent-20261026T014729`, all `done`.
- **The lock is free:** `status` says so, `~/.raypoolkit/` is empty, and
  `~/ScienceData/.rehearsal-lock` does not exist.
- **`var/`** holds `datastores`, `handover-03`, `-06`, `-07`, `-08`, `-10`, `-11`, `-12`,
  `levin-phase-floor-01`, `-03`, `-04`, `-05`, `levin-phase-floor-measure`, `py313-venv-02` and
  `runs`. `var/datastores/` holds A3 v2's five files and manifest, and the sidecars of retired
  stores (05's R9 and R10 among them).
- **05's sample:** `var/levin-phase-floor-05/wider/state/sample.json`, 26,284 bytes, SHA-256
  `73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7`, the `sample_sha256` every
  header of 05's run records.
- **05's harness** is `var/levin-phase-floor-05/wider/scripts/` there and
  `docs/levin-phase-floor/05/wider/scripts/` in the repository: `analyse.py`, `dispatch.py`,
  `draw_sample.py`, `harness.py`, `hash_store.py`, `l0.py`, `launch.py`, `prov.py`, `rebuild.py`,
  `rec.sh`, `run_wider.py`, `worker.py`.
- **05's run directory** `var/runs/levin-phase-floor-05-wider-20261026T020005/` holds its
  `checkpoint.jsonl`, `spec_{p,c}.json` and 109 `unit_*.pkl`: T0's reference and
  `convergence_kinds.py`'s `--capped-run`.
- **A3 v2** is `var/datastores/handover-A3-v2-lambdacdm.sqlite`, digest
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **R8's fingerprint**, which R11 and R12 must give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9 and R10 gave it.
- **`~/.ray/auth_token`** is mode 600. 118 GiB free; load about 4–5 on 10 cores.

### B.2 What the Mac Studio orchestrator is told

**Where things are.**
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at Part B's dispatch SHA. It
  holds the venv, the store, `var/` and the registry; the records go in its
  `var/levin-phase-floor-05a/`. R12 runs here, and so does §6's $t$ tree.
- **The worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-<parent>`, at the code
  commit's parent. R11 runs here, and so does §6's $c$ tree. It has no venv. Every command there
  calls the main checkout's `venv/bin/python` by its absolute path, with `PYTHONPATH` set to the
  worktree, `-B` or `PYTHONDONTWRITEBYTECODE=1`. Nothing is written in it.
- **R11's store path is absolute**, into the main checkout's `var/datastores/`, so that it outlives
  the worktree. The run directory is the main checkout's `var/runs/`, as for any run launched from a
  worktree.
- **No git command that writes**, in either tree, `fetch` included. The laptop moves both.

**Two agents, each fresh-context, on Opus, in the main checkout.** Give each the prompt (its own
section is its scope), Part B's dispatch SHA, the worktree's path, §B.1 and the addendum. Neither
reads `orchestrator/`. They may run at the same time.
- **The rehearsal agent,** prompt §5. It writes only under `var/levin-phase-floor-05a/rehearsal/`,
  the two stores `var/datastores/levin-phase-floor-05a-r11*` and `-r12*`, their run directories and
  launch logs, the lock and `/tmp/ray-sgk`.
  - **The launches are R9's and R10's** (`docs/levin-phase-floor/05/rehearsal/out/commands.log`),
    with `--database`, `--job-name levin-phase-floor-05a`, `--register` (`r11-parent`,
    `r12-reanchor-t`), `--campaign levin-phase-floor --prompt 05a`, and a `--purpose` that names
    the host, the tree's SHA and the code commit, and says the fingerprint must equal `1b332ffa…`.
    `--cpus 6`, never more. R11 is launched from the worktree, R12 from the main checkout.
  - **The lock:** `acquire` once before R11, `set-pid` with each driver's Python pid, held through
    the comparison and the readers, `release` once `ray-processes` is empty. Each lock command is a
    shell command of its own, output by `>>`, never a pipe.
  - **The comparison** is `tools.store_value_diff` run from the main checkout, as 05 ran it. 05's
    `classify_non_qsi.py` and `tabulate_qsi.py` are copied unchanged from
    `docs/levin-phase-floor/05/rehearsal/scripts/` into `scripts/`, and recorded by SHA-256 beside
    05's (`05/rehearsal/out/script_sha256.txt`). If either needs a change to read R11 and R12, that
    is a deviation, and the original is kept beside the copy.
  - **If R11 and R12 are value-identical in `QuadSourceIntegral`, stop:** then at least one tree did
    not run the code it names.
  - **The readers** on R12, with the store's files hashed before and after.
- **The re-measurement agent,** prompt §6. It writes only under `var/levin-phase-floor-05a/wider/`
  and its own run directory. It uses no Ray and takes no lock.
  - **The harness** is 05's, copied from `var/levin-phase-floor-05/wider/scripts/` into
    `wider/scripts/`. 05's copy is not edited. The diff against it is saved in `out/`, and each
    change is a deviation.
  - **The sample** is 05's `sample.json`, copied unchanged into `state/`, its SHA-256 checked equal
    to `73aa7876…` before any row runs.
  - **The predictions** go in `state/predictions.md`, prompt §6's text unchanged, after T0 passes
    and before the launch. Their mtime is the review's evidence.
  - **Every checkpoint record** carries the script's SHA-256 and its tree's `HEAD` and
    `git status --short`, and each unit's evidence of where its modules were loaded from.
  - **The budget** is raised from the dispatcher, as 05's was, never by a signal inside `mpmath`.
  - **The probe** reads phase objects only. Its windows, $z$ and samples are saved per (row,
    sub-interval, factor, tree), so that T1 can be re-scored from them.
  - **`convergence_kinds.py`** is copied unchanged from `prompts/levin-phase-floor/measure/`, its
    SHA-256 recorded, and run as prompt §6 states, at both tolerances.
- **Both agents:** `HEAD` is not theirs. They save every command and its output; the review reads
  the outputs, not the log. **Launch, one check, end the turn:** report the run id, the run
  directory, the detached Python pid, the `sh` pid, and the first checkpoint line. You wait on each
  pid once, in the background, and resume the agent with the final `status.json`. **Every deviation
  is numbered and classified before an agent reports**, failed attempts included (§0.1). They stop
  and ask at any of prompt §8's conditions.
- **Each agent's records** are `log.md` (its section, for appending to the code log),
  `verification.md` (its part of `VERIFICATION.md` §5a "Layer 1(a) for $T$"), `issues.md`, and
  `out/`, `scripts/` and `state/`.

**The marker:** `touch` a file in the session scratchpad immediately before the first dispatch.

### B.3 The review — on the Mac Studio, after both agents report

1. **Scope.** Both trees: `git status --short` empty and each `HEAD` as dispatched. `find var -newer
   <marker>` lists only §B.2's paths. `find <worktree> -newer <marker> -not -path
   '*/__pycache__/*'` lists nothing. A3 v2's files and sidecar are unchanged, and so are 05's
   `var/levin-phase-floor-05/` and its run directory.
2. **The runs.** `runregistrykit list` shows R11, R12 and `wider` `done`. Each manifest names
   `levin-phase-floor`, `05a` and its slug, `git_dirty: false`, and the SHA of the tree it ran
   from: R11 the parent, R12 the dispatch SHA.
3. **The fingerprints.** R11 and R12's `status.json` both record `1b332ffa…`. Re-run `store
   fingerprint` on each yourself, without `--write`.
4. **The values.** Re-run `tools.store_value_diff` on R11 and R12 yourself, with `--json`, into your
   scratchpad. It reproduces the saved JSON. Run the copied `classify_non_qsi.py` on yours: no
   `UNCLASSIFIED`, every value table outside `QuadSourceIntegral` identical, no row in one store
   only. Spot-check one differing `QuadSourceIntegral` row by a scratch reader (`sqlite3` with
   `mode=ro`): its `total` in both stores, in `float.hex`, against the tabulation.
5. **The readers and the lock.** The readers all exit 0, and R12's files are identical before and
   after. The lock transcript shows `acquire` 0, `set-pid` 0 with each driver's pid, `release` 0.
   `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing and `status` says free, each run by
   you as a command of its own.
6. **T0 and the store.** T0's three clauses in the saved output, compared with `==` and bit
   patterns, never `isclose`; T0 ran before the launch. **Independently,** by a scratch reader of
   the two `checkpoint.jsonl`s, compare one depth-20 row's $c$ unit at `1e-7` with 05's $c$ unit:
   `total`, `total_abserr`, and every call's value and abserr, in `float.hex`. The store files'
   hashes are identical before and after, and `store fingerprint`, run by you, gives `0e32f24a…`.
   No journal, `-wal` or `-shm` file beside any store file.
7. **The trees, read.** The $c$ units import this repository's packages from the worktree and the
   $t$ units from the main checkout: check `PYTHONPATH` in the launch and one module's `__file__`
   in a saved output. Under $t$, $T$'s phase objects report `exact_remainder` true; under $c$,
   false.
8. **The sample and the predictions.** `state/sample.json`'s SHA-256 is `73aa7876…`.
   `state/predictions.md` is later than T0's output and earlier than the run's `manifest.json`, and
   quotes prompt §6 word for word. The harness diff against 05's copy matches the deviations.
9. **The verdicts, re-scored.** From the checkpoints and the probe's saved windows, by a scratch
   reader: T1, T2, T3 and T4, each holds, fails or undecided against prompt §6's rule as written,
   with its numbers. T1 over every window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad, from the
   $z$-sampled $\sigma_\theta$. Over-budget units counted as the rules say. The descriptive items
   are present, the $u$-sampled $\sigma_\theta$ and `convergence_kinds.py`'s output among them.
10. **The records.** Each `log.md` follows campaign README §5.1 and classifies every deviation, the
    worktree among them. Each `verification.md` is written as part of `VERIFICATION.md` §5a, with
    its links relative to `docs/levin-phase-floor/` in the `05a/…` form. Each `issues.md` exists.

Write your verdict on each check to `var/levin-phase-floor-05a/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict (pass or stop),
then R11 and R12's fingerprints, the value comparison's summary, T0, T1, T2, T3 and T4 with their
numbers, and the over-budget units.

### B.4 After the review — on the laptop

The laptop copies `var/levin-phase-floor-05a/` back with `rsync -a`, without checkpoints or any CSV.
It re-checks over `ssh macstudio-tunnel` itself: check 4's spot-check on one row, and check 6's
comparison with 05's $c$ unit on one row. Then, in **one records commit**, pushed:
- the records to `docs/levin-phase-floor/05a/`, scripts byte-identical and unformatted, with their
  directories added to `CLAUDE.md`'s `black` paragraph if any output records their SHA-256;
- both `log.md`s appended to `logs/05a-reanchor-t-phase.md`, and the verdicts into its Predictions
  and verdicts;
- both `verification.md`s as `docs/levin-phase-floor/VERIFICATION.md` §5a "Layer 1(a) for $T$";
- the board: 05a's row, and anything `issues.md` opens;
- **the header's issues.** If §5 holds and T0, T1, T2 and T4 hold, both are closed: on this
  campaign's board §4, and their rows deleted from `docs/OPEN_ISSUES.md`. If any of those fails,
  **ask the user** before closing anything. T3 failing does not by itself keep them open: it is
  what $T$'s errors cost the rows before this prompt, and is recorded as a finding (an issue if
  the agent proposes one). T1's unscored $u$-sampled residue, if the agent proposes an issue for
  the abscissa term, is opened as one.
- `docs/OPEN_ISSUES.md` to match, count and date corrected by counting; this directory's README row
  for 05a.

R11 and R12's stores are the user's to keep or retire; ask. The worktree is removed with `git
worktree remove` on the Mac Studio once the records are committed and the user agrees.

### B.5 Addendum, 2026-10-26 12:20 GMT: Part B's dispatch

**Part A** is the code commit **`1fc349c`**, whose parent is the dispatch SHA **`c4eaefd`**. It
passed all nine checks of §A.3 on the laptop and is pushed. In particular:
- **The roots,** re-run by the laptop at `1fc349c`: AdaptiveLevin 32, ComputeTargets 764 (754 +
  10), CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280, RunDrivers 127,
  tools 55, all OK. The 10 methods were counted from the diff; the replaced test was renamed in
  place.
- **`primitive_phase.py`'s code did not move:** its AST, with the module docstring removed, is
  identical at `c4eaefd` and `1fc349c`. The keyword defaults on, is passed only to `_build_WKB`'s
  `PrimitivePhase`, and has no fallback.
- **Breakage (iv), re-applied by the laptop** ($p_R \to 0$ in the anchor): test 2 failed off a node
  at all four $k$, by exactly $k p_R$ reduced, and passed on a node. Reverted; the tree clean.

**Two commits follow it, on the user's word (2026-10-26):**
- **`bd38e50`** corrects the three comments the log's observation 4 names. Comments and docstrings
  only: both modules' ASTs, docstrings removed, are identical before and after. So R12's and $t$'s
  code is `1fc349c`'s.
- **`2db3f9e`** amends T1 before §6 runs: prompt **§9**. The agent's observations 1 and 2 showed
  that T1 as written could not be scored. A window of relative width $10^{-6}$ at large
  $|\theta_T|$ and $z \gtrsim 1$ is not resolved by its 65 samples, so $\sigma_\theta$ measures
  `numpy.unwrap`. And between table nodes the remainder keeps $\varepsilon k|p_z|$, up to
  $\sim10^{-2}\varepsilon|\theta_T|$, which test 2's bound already admits. §9 narrows each window
  until it resolves, and bounds $\sigma_\theta$ under $t$ by
  $\max(10^{-3}\varepsilon|\theta_T|, \varepsilon(k\max|p_z| + 2\pi), 10^{-13})$. T0, T2–T4 are
  unchanged.

**For the rehearsal agent:**
- **The tool's form** is `PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff R11.sqlite R12.sqlite --json`.
  Exit 1 is expected, since `QuadSourceIntegral` should differ. Exit 2 is a usage error or a store
  that cannot be opened.
- **`QuadSourceIntegral`'s `metadata` is expected to differ** as well as its totals: it stores
  `handover_continuity`, which takes $\sin$ and $\cos$ of $T$'s `theta_mod_2pi` (log, observation
  3). That is inside the class whose values are meant to move.

**For the re-measurement agent:**
- **Prompt §9 is part of its scope.** The probe follows §9's windows, records each window's width,
  largest unwrapped step under both trees, and $\max|p_z|$. `state/predictions.md` quotes §6 with
  §9's T1 in place of §6's, as §9 says.
- **T0's first clause** holds at the parent: `git diff --stat 120fdfe c4eaefd -- . ':!docs'
  ':!prompts'` shows `CLAUDE.md` alone.

**For the review (§B.3), two checks read with §9:**
- **Check 8:** `state/predictions.md` quotes §6's predictions with §9's T1, word for word.
- **Check 9:** T1 is re-scored as §9 states, from the saved windows: each resolved window's
  $\sigma_\theta$ against its own $\max|p_z|$, and each unresolved window counted.

**And §B.4's closing rule** reads T1 as §9's.

**§B.0 re-checked on the Mac Studio at 12:12 GMT**, read-only:
- the checkout is at `53084dc`, `git status --short` empty, one worktree;
- the registry holds 16 runs, all finished;
- the lock is free, `~/.raypoolkit/` is empty and `~/ScienceData/.rehearsal-lock` does not exist;
- there is no `var/levin-phase-floor-05a/` and no `levin-phase-floor-05a-*` store;
- 05's `sample.json` is `73aa7876…`, and its run directory holds 122 files;
- A3 v2 shows `problems: none` and `0e32f24a…`;
- `~/.ray/auth_token` is mode 600; 117 GiB free; the load is about 4.3 on 10 cores.

**The laptop then does three things:**
1. it fast-forwards the Mac Studio checkout to this addendum's commit;
2. it makes the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-c4eaefd`, detached at
   `c4eaefd`;
3. it starts the Mac Studio orchestrator.

**The trees, by SHA:**
- **R11 and $c$** run at `c4eaefd`, in the worktree;
- **R12 and $t$** run at this addendum's commit, in the main checkout. Its code is `1fc349c`'s, up
  to `bd38e50`'s comments. The commit itself adds only records under `prompts/`.

The orchestrator re-reads this state as its Gate, does not fetch, and stops if anything differs.
**Other projects on this host** may take the shared lock. If `acquire` exits 3, that is the stop
the lock's rule gives: ask the user.

### B.6 What happened (2026-10-26)

Part B ran from 12:17 to about 13:30 UTC, in one Mac Studio orchestrator session, with both agents
at once. It was started on the user's word, since the app starts a session on another host only in
reply to the user.

**R11 and R12.**
- Both ended `done` with `1b332ffa…`, and every value table of every class is bit-identical.
- Outside `QuadSourceIntegral`, only the four volatile kinds differ, with none unclassified, so §8
  did not fire.
- 424 of 864 totals move, by at most $8.8\times10^{-5}$ of their error. The readers exited 0 and
  the lock was released.

**The re-measurement.**
- 109 of 109 units, none over budget. T0 held: all 39 $c$ units equal 05's bit for bit.
- **One question, after the launch.** The agent asked whether §9's "largest step of the unwrapped
  samples" means the folded step, which admits aliased windows, or the phase's true step. It had
  fixed both readings before the run.
  - The laptop relayed the question, and told the user that no T1 number existed yet. That was
    wrong: both readings had been scored at 12:35:49Z.
  - The user chose the true step without having seen either number. Once the record of the timing
    had been corrected, the user confirmed it.
- **The verdicts:** T1 holds by the true step (70 of 70), and fails literally (36 aliased). T2 and
  T3 hold. T4 fails on 05's two L3 rows, unchanged.

**The review.** The Mac Studio orchestrator's review passed all ten checks, and its dated addendum
re-read checks 8 and 9 after the ruling.

**The laptop's re-checks**, as §B.4 asks, both pass:
- the QSI row of largest $|\Delta|/$abserr reads the tabulated totals bit for bit in both stores;
- `low-G-r-2`'s $c$ unit equals 05's in all 16 fields.

A3 v2's files are unchanged.

**The header's issues.** Both are closed, on the user's word after §B.4's question, since T4
failed. X2 is opened as `[05a-t-phase-abscissa-rounding-remains]`. X1 is not opened, because
§9's wording is corrected instead.

**What is left.** R11's and R12's stores, and the worktree `SecondaryGWKit-lpf05a-c4eaefd`, are the
user's to keep or retire.

---

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §8 conditions, or any failed check in §A.3 or §B.3:
report it, do not repair it. **Never write to A3 v2** or to 05's records on the Mac Studio, never
move the Mac Studio's checkout while R11, R12 or `wider` runs, and never end another session's run.
