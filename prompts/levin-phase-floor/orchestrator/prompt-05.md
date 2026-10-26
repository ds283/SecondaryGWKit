# Orchestrator — prompt 05, layer 1(a): take $k\tau_R$ out of $G$'s phase

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the deliberate breakages are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/05-reanchor-g-phase.md`](../05-reanchor-g-phase.md)
**Board item:** 05 · **Closes, when §5 and §6 hold:** `[00-consumer-anchoring-floor]`
(`GkTk-remedial`), for $G$ · **Narrows:** `[04-g-phase-is-reduced-against-the-double-two-pi]`
(this campaign) and `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`
(`handover`) · **Model:** Opus, as the prompt recommends, for every part. You, the orchestrator,
review on whatever model you were started with.

The prompt runs in three parts. This note has two, as `handover` 12's had:
- **Part A (prompt §1–§4), the code and `tools/store_value_diff.py`,** on the laptop: a
  fresh-context subagent in the **main checkout at `HEAD`**, one commit. This note's §A.
- **Part B (prompt §5 and §6), R9 and R10 under the lock, and the wider re-measurement,** on the Mac
  Studio, orchestrated there by one session, which dispatches two agents. This note's §B.

## 0. The order, and why

1. **Part A first.** R9 and R10 compare a tree with the commit against its parent, and the comparison
   is made with the tool Part A adds. §6's $c$ tree is the commit. Nothing on the Mac Studio starts
   until Part A has passed its review here, and is pushed.
2. **R9 needs the code commit's parent**, so it runs from a **detached worktree on the Mac Studio**
   at that SHA, which the laptop makes (§B.1). R10 runs from the Mac Studio's main checkout, which
   the laptop fast-forwards to Part B's dispatch SHA. That commit adds only records, so its code is
   the code commit's. The same pair of trees is §6's $p$ and $c$.
3. **R9 and R10 run one after the other, under the lock**, R9 first. Each is R8's build, about 3.5
   minutes at `--cpus 6`.
4. **§6 takes no lock and uses no Ray**, so its agent may run beside R9 and R10 (prompt, "Where it
   runs"). It reads `var/levin-phase-floor-01/out/groups_01.csv`, which exists only on the Mac Studio
   (`*.csv` is excluded from the laptop's copies).

---

## A. Part A — the code, on the laptop

### A.0 Facts checked by the orchestrator at `a130831`

Pass them on.
- **The checkout.** `handover-remedial` at `a130831`, pushed, `git status --short` empty, one
  worktree (itself). The laptop's registry shows nothing `running` or stale (60 runs: 58 finished,
  2 `unknown` pre-registry). Part A launches no run, starts no Ray and opens no production store.
- **The line references** in prompt "Read first" hold at `a130831`: `raw_theta` `:276`,
  `theta_mod_2pi` `:285`, `theta_deriv` `:291`, `build_phi_samples` `:359` (`primitive_phase.py`);
  `_hi_list`/`_lo_list` `:172-173`, `_integral` `:293`, `_locate` `:336`, `delta` `:360`, its
  `table` term at `:387-388` (`cumulative_table.py`); `TablePrimitive` at
  `BackgroundModel.py:638`, whose `.table` property returns the `CumulativeTable`; `_build_phase`
  `:145`, its callers `:237` and `:726` (`GkSourcePolicyData.py`); `_composed_theta_mod_2pi` at
  `phase_groups.py:542`; `GkWKBSplineWrapper`'s two remainder calls at `spline_wrappers.py:161`
  and `:169`; the driver's `sin`/`cos` of a remainder at `levin_quadrature.py:1106-1123`.
- **The helpers.** `WKB_mod_2pi` is `LiouvilleGreen/WKBtools.py:15`; `TWO_PI = 2.0 * pi` is
  `LiouvilleGreen/constants.py:3`. `mpmath` is 1.3.0 in the venv.
- **The existing tests** are `ComputeTargets/tests/test_primitive_phase.py`,
  `test_cumulative_table.py`, `test_phase_groups.py` and `test_gk_source_primitive_phase.py`.
- **The tool's dependencies.** `tools/` holds only `inventory_report.py` and its tests.
  `datastorekit.store_inventory` (`v0.5.0`, in the venv) builds the inventory records that the
  fingerprint digests (`datastorekit/store_fingerprint.py`'s docstring), and loads SQLAlchemy and
  not Ray. `StoreFixtures/real_store_fixtures.py` has `build_real_store` (`:294`) and `RealStore`
  (`:196`).
- **R8's fingerprint**, which R9 and R10 must give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`
  (`docs/handover/HANDOVER-B1.md` §5.3). It was taken at `8a02f0c`. Since then nothing outside
  `docs/` and `prompts/` has changed but `CLAUDE.md`.
- **The known flake** in the `ComputeTargets` root is `test_tk_wkb_phase …
  test_wall_time_per_object`, a wall-clock assertion. Re-run it alone and record it; it is not a
  stop. Record `uptime` beside each roots run.

### A.1 Before you dispatch

1. Commit this note and `README.md`'s 05 row, and push; that commit is the **dispatch SHA**.
2. **Baselines:** the seven roots serially in the main checkout, into the session scratchpad. They
   were taken at `a130831`, whose code the dispatch SHA shares (it adds only this note).
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
  - `ComputeTargets/cumulative_table.py`, `ComputeTargets/BackgroundModel.py` (`TablePrimitive`
    only), `ComputeTargets/primitive_phase.py`, `ComputeTargets/GkSourcePolicyData.py`
    (`_build_phase` only);
  - test modules under `ComputeTargets/tests/`, new or existing;
  - `tools/store_value_diff.py` and `tools/tests/test_store_value_diff.py`;
  - `prompts/levin-phase-floor/logs/05-reanchor-g-phase.md` and this campaign's board;
  - `docs/OPEN_ISSUES.md` if the commit opens an issue.

  A path outside that list is a stop. `TkSourceFunctions.py`, `phase_groups.py`,
  `spline_wrappers.py`, `AdaptiveLevin/`, `ObjectFactories/`, `StoreFixtures/`, `config/` and
  `main.py` in particular are not touched; if a test cannot be written without a change to
  `StoreFixtures/`, stop and ask.
- **The header's issues do not move in this commit** (prompt §4). They move in 05's records commit,
  after §5 and §6.
- **No Ray, and no store but the fixtures'** in its tests and its checks. It writes nothing under
  `var/`.
- **Stop and ask** at any of prompt §8's conditions that apply to §1–§4.

### A.3 The review — nine checks, on the laptop

1. **Scope.** `git show --stat <code commit>`: only §A.2's paths. `git diff --stat <dispatch SHA>
   <code commit> -- ComputeTargets/TkSourceFunctions.py ComputeTargets/phase_groups.py
   ComputeTargets/spline_wrappers.py AdaptiveLevin ObjectFactories StoreFixtures config main.py`
   is empty. `git status --short` is clean, and `find var -newer <marker>` lists nothing of the
   agent's.
2. **The default is today's, read.** The keyword defaults off, and with it off `theta_mod_2pi` is
   the old expression unchanged. `raw_theta` and `theta_deriv` are not edited. `delta`'s
   arithmetic is unchanged, and the accessor reads the same limbs. `TablePrimitive` only passes
   the accessor through. `_build_phase` turns the keyword on, and nothing else does.
3. **The construction, read** against prompt §0.2:
   - $c_n$ is formed from the four limbs exactly, as `mpmath` numbers or `Fraction`s, never through
     `delta`'s double; with $p_R$ and $k$ entered exactly; reduced against `mpmath.pi`'s $2\pi$ at
     no fewer than 40 digits, into $(-2\pi, 0]$, and rounded to a double once;
   - the precision is set locally (`mpmath.workdps` or a private context), never by assigning the
     global `mp.dps`;
   - $c_n$ is memoised per node and computed lazily;
   - $v(z) = k\,p_z + \phi(u)$ is formed in double and reduced with `WKB_mod_2pi`;
   - the sum is wrapped once by an add or subtract of `TWO_PI`;
   - `sign` enters $c_n$ and $v$ as it enters `raw_theta`; the abscissa is taken through `_get_x`
     and `_clamped_u`;
   - without the accessor, it raises; there is no fallback.
4. **The roots, re-run here.** Five equal to the baselines; `ComputeTargets` and `tools` up by
   exactly the test methods added, counted from the diff. All OK, with the flake re-run alone.
5. **The tests, read.** Prompt §3's tests 1–6 and §3A's (t1)–(t5) each exist, named in the log.
   Test 2's reference is in `mpmath` from the same limbs, against $2\pi$, with the bound
   $4\varepsilon(k|p_z| + 2\pi)$, at the four $k$ and both $z_R$. Test 3 uses 04's $\sigma_\theta$
   (65 points, `numpy.unwrap`, degree-4 fit) with both thresholds. Test 4 covers both signs and a
   `TkSourceFunctions` phase built as production builds it. Test 6 uses `test_phase_groups.py`'s
   realistic fixture and states both bounds.
6. **The breakages.** (i)–(iv) each name the test that failed and were reverted. If a record is
   thin, re-apply that breakage yourself in the working tree, run the one module, and revert with
   `git checkout -- <file>`; `git status --short` is clean afterwards.
7. **The tool, read and run.**
   - It matches rows by the inventory's identity key and never by serial, and (t4) is the test of
     that. It never writes: the stores are opened read-only. It imports no Ray.
   - Floats are compared by `float.hex`, with `NaN` equal to `NaN`; JSON is canonicalised.
   - `--json`, `--classes` and the exit codes are as prompt §3A states.
   - A class whose value rows cannot be matched by identity is reported by name. If the log says a
     production class is one, that is prompt §8's stop.
   - **Run it yourself** on two `build_real_store` stores in the scratchpad, from a scratch script:
     exit 0 on two builds, and exit 1 with the class, identity, column and $\Delta$ printed after
     one float is changed by one ulp.
8. **The records.**
   - The log has README §5.1's headings and classifies every deviation.
   - Its **Predictions and verdicts** holds prompt §6's predictions word for word (check by
     substring), marked not yet scored.
   - The board's 05 row says the code landed and §5–§6 are pending, with no commit hash yet (that is
     residue). The header's issues have not moved, and `docs/OPEN_ISSUES.md` changed only if an
     issue was opened, with count and date correct by counting.
   - The module docstring's "Anchoring, and the floor it implies" says what holds for $G$ and for
     $T$, citing G1.
   - `black --check` is clean on every `.py` file touched. The commit message follows `CLAUDE.md`'s
     form.
9. **Nothing else moved.** `git log --oneline <dispatch SHA>..HEAD` shows the code commit and
   anything another session landed, each accounted for.

### A.4 Landing

If the review passes: push the code commit. Then, in one residue commit of the orchestrator's own,
the code commit's hash on the board's 05 row and in the log, and §B.5's addendum below; that commit
is Part B's dispatch SHA. Push it.

---

## B. Part B — R9, R10 and the wider re-measurement, on the Mac Studio

### B.0 When it starts

Only when all hold, each checked by the laptop session over `ssh macstudio-tunnel`, read-only:
- Part A has landed, and the residue commit is pushed;
- on the Mac Studio: `git status --short` is empty; `runregistrykit list` shows nothing `running` or
  stale; the lock is free; there is no `var/levin-phase-floor-05/` and no
  `var/datastores/levin-phase-floor-05-*`.

Then the laptop, the only party that writes git on the Mac Studio:
1. fetches and fast-forwards the main checkout to Part B's dispatch SHA (`git merge --ff-only`);
2. makes the worktree for the parent, detached:
   `git worktree add --detach ../SecondaryGWKit-lpf05-<parent short SHA> <parent SHA>`, with no
   venv in it;
3. starts the Mac Studio orchestrator, a Claude Code session over the saved `MacStudio` SSH
   connection, on this note's §B and the addendum in §B.5.

### B.1 Facts, as at `a130831`

Re-checked at Part B's dispatch and corrected in the addendum where they have moved. Checked
2026-10-26 at 00:56 GMT, read-only:
- **The checkout** is `handover-remedial` at `c8ca155`, `git status --short` empty, one worktree
  (itself): 04's worktree has been removed. Python 3.13.16.
- **The registry:** nothing `running` or stale. The newest runs are
  `levin-phase-floor-04-remove-20261025T231110` and `handover-12-r8-reference-20261025T210551`,
  both `done`.
- **The lock is free:** `status` says so, `~/.raypoolkit/` is empty, and
  `~/ScienceData/.rehearsal-lock` does not exist. `status` takes no `--project`.
- **`var/`** holds `datastores`, `handover-03`, `-06`, `-07`, `-08`, `-10`, `-11`, `-12`,
  `levin-phase-floor-01`, `-03`, `-04`, `py313-venv-02` and `runs`.
- **01's census:** `var/levin-phase-floor-01/out/groups_01.csv`, 37,337 lines, 12.4 MB, headed by
  `group_census.py`'s SHA-256 `aac0b4ed…`.
- **A3 v2** is `var/datastores/handover-A3-v2-lambdacdm.sqlite`, digest
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **R8's store was retired** (`docs/handover/b1/r8/out/retire_r8.txt`). Its digest `1b332ffa…`
  stands in `status.json` of `handover-12-r8-reference-20261025T210551`; `--build` writes no
  sidecar.
- **`~/.ray/auth_token`** is mode 600. 118 GiB free; load about 3.3–4.3 on 10 cores.

### B.2 What the Mac Studio orchestrator is told

**Where things are.**
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at Part B's dispatch SHA. It
  holds the venv, the store, `var/` and the registry; the records go in its
  `var/levin-phase-floor-05/`. R10 runs here, and so does §6's $c$ tree.
- **The worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-<parent>`, at the code commit's
  parent. R9 runs here, and so does §6's $p$ tree. It has no venv. Every command there calls the
  main checkout's `venv/bin/python` by its absolute path, with `PYTHONPATH` set to the worktree.
  Nothing is written in it.
- **R9's store path is absolute**, into the main checkout's `var/datastores/`, so that it outlives
  the worktree. The run directory is the main checkout's `var/runs/`, as for any run launched from a
  worktree.
- **No git command that writes**, in either tree, `fetch` included. The laptop moves both.

**Two agents, each fresh-context, on Opus, in the main checkout.** Give each the prompt (its own
section is its scope), Part B's dispatch SHA, the worktree's path, §B.1 and the addendum. Neither
reads `orchestrator/`. They may run at the same time.
- **The rehearsal agent,** prompt §5. It writes only under `var/levin-phase-floor-05/rehearsal/`,
  the two stores `var/datastores/levin-phase-floor-05-r9*` and `-r10*`, their run directories and
  launch logs, the lock and `/tmp/ray-sgk`.
  - **The launches are R8's** (`docs/handover/b1/r8/` and `HANDOVER-B1.md` §5), with `--database`,
    `--job-name`, `--register` (`r9-parent`, `r10-reanchor`), `--campaign levin-phase-floor
    --prompt 05`, and a `--purpose` that names the host, the tree's SHA and the code commit, and
    says the fingerprint must equal `1b332ffa…`. `--cpus 6`, never more. R9 is launched from the
    worktree, R10 from the main checkout.
  - **The lock:** `acquire` once before R9, `set-pid` with each driver's Python pid, held through
    the comparison and the readers, `release` once `ray-processes` is empty. Each lock command is a
    shell command of its own, output by `>>`, never a pipe.
  - **The comparison** is `tools.store_value_diff` run from the main checkout. Its JSON and the
    tabulation script go in `out/` and `scripts/`.
  - **If R9 and R10 are value-identical in `QuadSourceIntegral`, stop:** then at least one tree did
    not run the code it names, for example through a Ray worker importing from the wrong tree.
  - **The readers** on R10, with the store's files hashed before and after.
- **The re-measurement agent,** prompt §6. It writes only under `var/levin-phase-floor-05/wider/`
  and its own run directory. It uses no Ray and takes no lock.
  - **The harness** is 04's, from `docs/levin-phase-floor/04/scripts/`. Those files are records and
    are not edited: it copies what it needs into `wider/scripts/` and adapts it there.
  - **The rule** goes in `state/sample.md` before any row runs. **The predictions** go in
    `state/predictions.md`, prompt §6's text unchanged, after L0 passes and before the launch.
    Their mtimes are the review's evidence.
  - **Every checkpoint record** carries the script's SHA-256 and its tree's `HEAD` and
    `git status --short`.
  - **The budget** is raised from the dispatcher, never by a signal inside `mpmath` (04's I9).
- **Both agents:** `HEAD` is not theirs. They save every command and its output; the review reads
  the outputs, not the log. **Launch, one check, end the turn:** report the run id, the run
  directory, the detached Python pid, the `sh` pid, and the first checkpoint line. You wait on each
  pid once, in the background, and resume the agent with the final `status.json`. They stop and
  ask at any of prompt §8's conditions.
- **Each agent's records** are `log.md` (its section, for appending to the code log),
  `verification.md` (its part of `VERIFICATION.md` §5 "Layer 1(a)"), `issues.md`, and `out/`,
  `scripts/` and `state/`.

**The marker:** `touch` a file in the session scratchpad immediately before the first dispatch.

### B.3 The review — on the Mac Studio, after both agents report

1. **Scope.** Both trees: `git status --short` empty and each `HEAD` as dispatched. `find var -newer
   <marker>` lists only §B.2's paths. `find <worktree> -newer <marker> -not -path
   '*/__pycache__/*'` lists nothing. A3 v2's files and sidecar are unchanged.
2. **The runs.** `runregistrykit list` shows R9, R10 and `wider` `done`. Each manifest names
   `levin-phase-floor`, `05` and its slug, `git_dirty: false`, and the SHA of the tree it ran
   from: R9 the parent, R10 the dispatch SHA.
3. **The fingerprints.** R9 and R10's `status.json` both record `1b332ffa…`. Re-run `store
   fingerprint` on each yourself, without `--write`.
4. **The values.** Re-run `tools.store_value_diff` on R9 and R10 yourself, with `--json`, into your
   scratchpad. It reproduces the saved JSON. Every class but `QuadSourceIntegral` is value-identical,
   with no row in one store only. Spot-check one differing `QuadSourceIntegral` row by a scratch
   reader (`sqlite3` with `mode=ro`): its `total` in both stores, in `float.hex`, against the
   tabulation.
5. **The readers and the lock.** The readers all exit 0, and R10's files are identical before and
   after. The lock transcript shows `acquire` 0, `set-pid` 0 with each driver's pid, `release` 0.
   `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing and `status` says free, each run by
   you as a command of its own.
6. **L0 and the store.** L0's two checks in the saved output, compared with `==` and bit patterns,
   never `isclose`; L0 ran before the launch. **Independently,** for one depth-20 row, read the
   stored `QuadSource` values yourself and compare them with the recomputed values at $p$, in
   `float.hex`. The store files' hashes are identical before and after, and `store fingerprint`,
   run by you, gives `0e32f24a…`. No journal, `-wal` or `-shm` file beside any store file.
7. **The trees, read.** The $p$ units import this repository's packages from the worktree and the
   $c$ units from the main checkout: check `PYTHONPATH` in the launch and one module's `__file__`
   in a saved output. The patch-free harness calls production's `PrimitivePhase` in both trees.
8. **The sample and the predictions.** `state/sample.md` is older than the first checkpoint and
   states the rule and its bias; the rows drawn follow it, recomputed from `groups_01.csv` by you.
   `state/predictions.md` is later than L0's output and earlier than the run's `manifest.json`, and
   quotes prompt §6 word for word.
9. **The verdicts, re-scored.** From the checkpoints, by a scratch reader: L1, L1q, L2 and L3, each
   holds, fails or undecided against prompt §6's rule as written, with its numbers. Over-budget units
   counted as the rules say. The descriptive items are present.
10. **The records.** Each `log.md` follows campaign README §5.1 and classifies every deviation, the
    worktree among them. Each `verification.md` is written as part of `VERIFICATION.md` §5, with its
    links relative to `docs/levin-phase-floor/` in the `05/…` form. Each `issues.md` exists.

Write your verdict on each check to `var/levin-phase-floor-05/review.md`, with the commands you ran
and what they printed. End the turn with a report whose first line is the verdict (pass or stop),
then R9 and R10's fingerprints, the value comparison's summary, L0, L1, L1q, L2 and L3 with their
numbers, and the over-budget units.

### B.4 After the review — on the laptop

The laptop copies `var/levin-phase-floor-05/` back with `rsync -a`, without checkpoints or any CSV.
It re-checks over `ssh macstudio-tunnel` itself: check 4's spot-check on one row, and check 6's
independent comparison on one row. Then, in **one records commit**, pushed:
- the records to `docs/levin-phase-floor/05/`, scripts byte-identical and unformatted, with their
  directories added to `CLAUDE.md`'s `black` paragraph if any output records their SHA-256;
- both `log.md`s appended to `logs/05-reanchor-g-phase.md`, and the verdicts into its Predictions
  and verdicts;
- both `verification.md`s as `docs/levin-phase-floor/VERIFICATION.md` §5 "Layer 1(a)";
- the board: 05's row, and anything `issues.md` opens;
- **the header's issues.** If §5 holds and L1, L2 and L3 hold, `[00-consumer-anchoring-floor]` is
  closed for $G$ on the `GkTk-remedial` board, and the two others narrowed where they are held. If
  any of those fails, **ask the user** before closing anything. L1q failing does not by itself keep
  the issue open for $G$; it is recorded as the case for re-anchoring $T$.
- `docs/OPEN_ISSUES.md` to match, count and date corrected by counting; this directory's README row
  for 05.

R9 and R10's stores are the user's to keep or retire; ask. The worktree is removed with `git worktree
remove` on the Mac Studio once the records are committed and the user agrees.

### B.5 Addendum, 2026-10-26 01:35 GMT: Part B's dispatch

**Part A** is the code commit **`120fdfe`**, whose parent is the dispatch SHA **`e648951`**. It
passed all nine checks of §A.3 on the laptop and is pushed. In particular:
- **The roots,** re-run by the laptop at `120fdfe`: AdaptiveLevin 32, ComputeTargets 754 (742 + 12),
  CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280, RunDrivers 127, tools 55
  (49 + 6), all OK. The 18 test methods were counted from the diff.
- **The construction, read:** $c_n$ is formed from the limbs and $k$ as `Fraction`s, reduced against
  $2\pi$ in a private 256-bit `mpmath` context and memoised. $v$ is reduced with `WKB_mod_2pi`, and
  the wrap is one add of `TWO_PI`. `TkSourceFunctions` keeps the default.
- **The tool, run by the laptop** on two `build_full_store` stores: exit 0, every class
  value-identical. After a one-ulp change to one `QuadSourceIntegral` row's `total` it exits 1, and
  names the class, the identity, `total` and $\Delta = 2.2\times10^{-16}$. The stores were unchanged,
  and no Ray was imported.

**For the rehearsal agent:**
- **The tool's form** is `PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff R9.sqlite R10.sqlite --json`.
  Exit 1 is expected, since `QuadSourceIntegral` should differ. Exit 2 is a usage error or a store
  that cannot be opened. For each differing row its JSON gives the identity and both values of each
  differing column, in `float.hex` and as numbers, with $\Delta$ and $|\Delta|/|a|$; for `metadata`
  it gives the keys that differ.
- **A class whose rows cannot be matched** makes the result not value-identical, and is named. In
  any class other than `QuadSourceIntegral` that is prompt §8's stop.

**§B.0 re-checked on the Mac Studio at 01:33 GMT**, read-only:
- the checkout is at `c8ca155`, `git status --short` empty, one worktree;
- the registry holds 13 runs, all finished;
- the lock is free, `~/.raypoolkit/` is empty and `~/ScienceData/.rehearsal-lock` does not exist;
- there is no `var/levin-phase-floor-05/` and no `levin-phase-floor-05-*` store;
- A3 v2 shows `problems: none` and `0e32f24a…`;
- `~/.ray/auth_token` is mode 600; 118 GiB free; the load is about 5.3–6.5 on 10 cores.

**The laptop then does three things.** It fast-forwards the Mac Studio checkout to this addendum's
commit. It makes the worktree
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951`, detached at `e648951`. It starts the
Mac Studio orchestrator.

**The trees, by SHA:**
- **R9 and $p$** run at `e648951`, in the worktree;
- **R10 and $c$** run at this addendum's commit, in the main checkout. Its code is `120fdfe`'s: the
  commit adds only records under `prompts/`.

The orchestrator re-reads this state as its Gate, does not fetch, and stops if anything differs.
**Other projects on this host** (ChamKadanoffBaym, ChamLattice) may take the shared lock. If
`acquire` exits 3, that is the stop the lock's rule gives: ask the user.


### B.6 What happened (2026-10-26)

Part B ran from 01:44 to about 02:25 GMT, in one Mac Studio orchestrator session, with both agents at
once.

**R9 and R10.**
- Both ended `done` with `1b332ffa…`. Every value table of every class is bit-identical.
- **Stop 1, at prompt §8.** Outside `QuadSourceIntegral` the differences were only timings, labels
  carrying `datetime.now()`, `GkSource.parents_digest` (a digest of serials) and GkWKB memo
  counters in 25 rows. The user: *"Those columns cannot be expected to be identical. So I think
  that's just an incorrectly-worded prompt."* The lock stayed held through the readers and was then
  released. This opened `[05-store-value-diff-compares-volatile-columns]`.

**The wider run.**
- It finished 109 of 109 units, none over budget, and L0 held.
- **Stop 2, on reading "depth-20 row":** B1 had already uncapped most stored depth-20 rows. The
  user: *"Carry on as you propose. … I guess we only care about rows that are capped under the new
  system"*. The laptop told the agent that $p$ is the parent (B1 without 05).
- **Scored as written,** L1, L1q and L3 fail and L2 holds. On the 11 rows still capped under $p$
  the cap goes on all of them.

**The review.** The Mac Studio orchestrator's review stopped on check 10 alone: deviations 12 and
14 of the wider log were unclassified. The user agreed to classify them on committing (12
STRUCTURALLY REQUIRED, 14 IMPLEMENTATION CHOICE, in the log).

**The laptop's re-checks**, as §B.4 asks, both pass:
- the 28× row reads the tabulated totals bit for bit in both stores;
- `QuadSource` at $p$ on `low-G-r-2` equals A3 v2 at all 666 nodes, plus B1's 3.

**The header's issues.**
- **Closed:** `[00-consumer-anchoring-floor]`, for $G$, on the capped-row reading, by the user's
  decision. §B.4's rule would otherwise have held it open, because L1 and L3 fail as written.
- **Narrowed:** the other two, and two more of this campaign's.
- **Opened:** three issues, among them
  `[05-b1-moves-stored-a3v2-totals-beyond-their-error]`.

**What is left.**
- **The stores.** The user asked whether R9's and R10's stores are needed. The laptop's answer: no.
  R10 equals R8's fingerprint, the comparison's records are kept, and a later pair is built at its
  own trees.
- **Done on the user's word, after `b0207e0`** (2026-10-26, 08:48 GMT). Both stores were retired
  as R8's was:
  - `store create`, then `store fingerprint --write`, which recorded `1b332ffa…` in each;
  - a dry run, which matched that fingerprint and found only each store's own run naming it;
  - then `store retire`.

  Their ten files went, and their sidecars stay as the `retired` record
  ([`retire_r9.txt`](../../../docs/levin-phase-floor/05/rehearsal/out/retire_r9.txt),
  [`retire_r10.txt`](../../../docs/levin-phase-floor/05/rehearsal/out/retire_r10.txt)). The
  worktree `SecondaryGWKit-lpf05-e648951` was clean under `git status --short --ignored`, and was
  removed with `git worktree remove`.

---

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §8 conditions, or any failed check in §A.3 or §B.3:
report it, do not repair it. **Never write to A3 v2**, never move the Mac Studio's checkout while R9,
R10 or `wider` runs, and never end another session's run.
