## Common to both Part B agents of prompt 06b (levin-phase-floor campaign), on the Mac Studio

You are one of two fresh-context agents running Part B of prompt 06b of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you; it reviews your records afterwards. Today is 2026-10-28.

**Your prompt** is `prompts/levin-phase-floor/06b-roundoff-exit-premise.md`, read in the code
worktree (below). Read it in full, and the `CLAUDE.md` of the code worktree, which binds you (the
run registry, the shared rehearsal lock, the venv, `black`). Your scope is the section named in your
own brief below; the other section is the other agent's. **Do not read anything under
`prompts/levin-phase-floor/orchestrator/`.**

### Where things are

- **Part B's dispatch SHA** is `92bc62f09711d10ef0035663d4bbe5629e145e17` (`92bc62f`). It adds only
  records under `prompts/` to the code commit `4ddc3c2`; its code is `4ddc3c2`'s. The code
  commit's parent is `da5c24deb430172f68d983f56577fa48def19903` (`da5c24d`).
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf0`, holds the venv, the stores, `var/` and the run registry. The records go in its
  `var/levin-phase-floor-06b/`. **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-da5c24d`, detached
  at `da5c24d`: R15 and W0's reproduction of 06a run here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-92bc62f`, detached at
  `92bc62f`: R16, the tree $b$ of §5, the value comparison and the readers run here. The prompt is
  read here.
- **In both worktrees:**
  - every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` by that
    absolute path, with `PYTHONPATH` set to that worktree and `-B` (or
    `PYTHONDONTWRITEBYTECODE=1`);
  - nothing is written in either worktree, no `__pycache__` included.
- **Stores' paths are absolute,** into the main checkout's `var/datastores/`. Run directories are in
  the main checkout's `var/runs/` (a run launched from a worktree is recorded there).
- **No git command that writes,** in any tree, `fetch` included. Read git with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06b/`
  back and commits it.
- **`HEAD` is not yours.** Do not assume any tree's `HEAD` reflects your work; check each tree's
  `HEAD` and `git status --short` where you rely on them.

### Facts (from 06a's and earlier records; check what you rely on — no fact here is to be trusted over the code or the store)

- The main checkout is `handover-remedial` at `28ebaf0`, clean. `git worktree list` shows three
  trees: the main checkout and the two worktrees above. 06a's and 06's worktrees no longer exist.
- **A3 v2's scratch copy** is `var/levin-phase-floor-06/wider/state/a3v2_copy/` (five files). Its
  certification is `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`: primary
  `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…` and `ecc3330d…`. Its fingerprint and
  A3 v2's are both `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06a's run** is `var/runs/levin-phase-floor-06a-census-20261027T023143/`: 18,560 units,
  02:31:43Z–03:17:10Z, none over budget or in error. Its records are in the main checkout's
  `var/levin-phase-floor-06a/` and committed in `docs/levin-phase-floor/06a/`. Its harness is
  `docs/levin-phase-floor/06a/scripts/`, byte-identical to the run.
- **R8's fingerprint,** which R15 and R16 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R14 gave it.
- **R13 and R14's launches** are recorded in `docs/levin-phase-floor/06/rehearsal/`. 05's
  `classify_non_qsi.py` and `tabulate_qsi.py` are in `docs/levin-phase-floor/05/rehearsal/scripts/`.
  The readers are `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`
  and work counters (`[05-store-value-diff-compares-volatile-columns]`).
- At dispatch: the registry holds 25 runs, all finished; the rehearsal lock
  `~/.raypoolkit/rehearsal-lock` is free; `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none;
  there is no `var/levin-phase-floor-06b*` and no `var/datastores/levin-phase-floor-06b*`.

### The addendum the orchestrator's note records at Part B's dispatch (verbatim)

### B.5 Addendum, 2026-10-28: Part A landed

**Part A** is the code commit **`4ddc3c2`**, whose parent is **`da5c24d`**. It passed all eight
checks of §A.3 on the laptop, and is pushed.
- **Scope.** Only §A.2's paths. By AST without docstrings, the module changed only in
  `_adaptive_levin`, plus `_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO = 0.5` and
  `_roundoff_exit_premise_holds`. `adaptive_levin_sincos` is unchanged. The stash and
  `var/levin-phase-floor-07/wip/` are intact by SHA-256.
- **The roots,** re-run by the laptop at `4ddc3c2`: AdaptiveLevin 38 (32 + the new module's 6),
  ComputeTargets 780, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ObjectFactories 280,
  RunDrivers 127, tools 58, all OK. The flake did not fire.
- **Test 1, re-run by the laptop against the parent's own module,** extracted from `da5c24d`. Its
  value `0x1.18d9a35c68fc4p-7` and abserr `0x1.047898c70661ep-20` (1 region, phase-limited) are
  bit-identical to the bypass and to the test's comment. Under the new code there are 2 regions,
  the call converges, and the error is $1.5\times10^{-11}$.
- **Test 5's 20 seeded draws, re-run by the laptop** in a scratch worktree at `4ddc3c2`, equal the
  committed `lg_seeded_draws.jsonl` on all 160 units, timings aside. Every recorded SHA-256 matches
  its committed file.
- **Breakage (i), re-applied by the laptop** (the premise always holds): tests 1 and 3 fail, and
  so does the infinite-half contract test. It was reverted, and the tree is clean.

**What Part B must carry:**
- **The census's W4 may fail.** One seeded LiouvilleGreen call, seed 15 `test_JJJ`, runs to
  depth 20 under the new code: from depth 5 and 23 solves to 2,295 solves, with abserr unchanged
  and the call unconverged in both. Its step-(4) residual is flat at 1–2e-15 while the eq. (151)
  floor falls to 8e-20 (`[06b-premise-check-can-chase-a-flat-residual-to-depth-max]`). W4 is
  scored as written, and a failure is a finding for the user.
- **Evaluation counters** differ only in calls that had a phase-limited **fallback** region at the
  parent. There the halves are now solved. Such a row is phase-limited in R15, so §4.3's
  bit-identity rule does not reach it.
- **The trees, by SHA:** R15 and W0's reproduction at `da5c24d`, in the parent worktree. R16 and
  $b$ at the dispatch commit of Part B, whose code is `4ddc3c2`'s.

**Part B starts on the user's word.** §B.0's Mac Studio facts are re-checked by the laptop at that
point, and added here.

**The user gave it on 2026-10-28:** "Please dispatch 06b's Mac Studio part. That is safe to go."
Part B is dispatched from **this addendum's commit**, which adds only records under `prompts/`:
`git diff --stat 4ddc3c2 <this commit> -- . ':!docs' ':!prompts'` is empty. 07's §2 stop is
waiting on the user, and 07's work stays uncommitted in the laptop's checkout and in its stash.
Neither is on the Mac Studio, and Part B does not read them.

**§B.0 re-checked on the Mac Studio at 13:47 UTC**, read-only:
- the main checkout is `handover-remedial` at `28ebaf0`, `git status --short` empty, one
  worktree, itself;
- the registry holds 25 runs, all finished, nothing `running` or stale; 06a's run
  `levin-phase-floor-06a-census-20261027T023143` is `done`, 18,560 of 18,560 units, exit 0;
- the lock is free (`~/.raypoolkit/rehearsal-lock` does not exist), and `ray-processes
  --ray-temp-dir /tmp/ray-sgk` lists none;
- there is no `var/levin-phase-floor-06b*` and no `var/datastores/levin-phase-floor-06b*`;
- the scratch copy's five files match `copy_files_certified.txt` by SHA-256, with no journal,
  `-wal` or `-shm` file beside them;
- `~/.ray/auth_token` is mode 600; 109 GiB free on the data volume. The load is about 9, all
  macOS and desktop processes (WindowServer, ChronoSync, `fileproviderd`, the Claude app), no
  Python.

**The worktrees,** made by the laptop after this commit is pushed, detached, with no venv:
- `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-da5c24d` at `da5c24d`: R15 and W0's
  reproduction of 06a;
- `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-<this commit's short SHA>`: R16, $b$, the
  value comparison and the readers. This note and the prompt are read here.

The records go in the main checkout's `var/levin-phase-floor-06b/`, and the stores are
`var/datastores/levin-phase-floor-06b-r15*` and `-r16*`, with `--job-name levin-phase-floor-06b`,
`--campaign levin-phase-floor --prompt 06b`.



### Rules for both of you

- **Never write to** A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`), its scratch copy (read it
  only, opened read-only), or any earlier prompt's records, stores or runs (`var/levin-phase-floor-0*`
  other than `-06b`, `var/runs/` other than your own runs). Never end another session's run.
  Never touch anything of prompt 07's.
- **Save every command you run and its output** under your `out/` (or `state/` for launch
  transcripts). The review reads the outputs, not your log.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4). Launch detached
  (`nohup sh -c …`, manifest first, stdout/stderr into the run directory, pid recorded), verify in
  **one** check that it started, then end your turn and report: the run id, the run directory, the
  detached Python pid, the `sh` pid and the first checkpoint/heartbeat line. **Do not poll, do not
  `sleep`-wait, never `pgrep -f`.** The orchestrator waits on the pid and resumes you with the
  final `status.json`.
- **Every deviation from the prompt is numbered and classified** (`STRUCTURALLY REQUIRED`,
  `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`) before you report, failed attempts included.
- **Stop and ask** at any of prompt 06b §7's conditions that apply to your section, and at anything
  this brief or the prompt does not settle that changes what you measure. To stop, end your turn
  with the question stated plainly and the state of every file and run you have made. Do not guess
  and do not repair.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a
  deviation to record, not a reason to take it.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-06b/`:
  - `log.md` — your section, in campaign README §5.1's form
    (`prompts/levin-phase-floor/README.md` §5.1), for appending to
    `logs/06b-roundoff-exit-premise.md`;
  - `verification.md` — your part of `docs/levin-phase-floor/VERIFICATION.md` §6b, "The
    round-off-limited exit", with links in the `06b/…` form (the laptop commits your directory as
    `docs/levin-phase-floor/06b/rehearsal/` or `…/06b/census/`);
  - `issues.md` — anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`.


## Your brief: the census agent — prompt 06b §5 (06a's census re-run under $b$)

**Your scope is prompt 06b §5**, and only it. §4 (rehearsals R15 and R16, under Ray and the lock)
is the other agent's, and runs in the same period.

- **Where you write:** only `var/levin-phase-floor-06b/census/` (with `out/`, `scripts/`, `state/`)
  and your own run directory in `var/runs/`. You use no Ray and take no lock.
- **The harness** is 06a's, copied from `docs/levin-phase-floor/06a/scripts/` (in the code
  worktree) into `var/levin-phase-floor-06b/census/scripts/`. Change only what prompt §5 allows.
  The diff against 06a's copy is saved in `out/`, and each change is a numbered deviation.
- **The trees:** W0's reproduction of 06a runs at the parent worktree (`da5c24d`); $b$ is the code
  worktree (`92bc62f`). 06a's own tree was `e05c34f` (in git history; read it with
  `git --no-optional-locks`, never by making a worktree).
- **The store** is the scratch copy, opened read-only, re-checked against its certification and
  fingerprint before the run (with the copied `fingerprint_copy.py`), and hashed (SHA-256) before
  and after, with no journal, `-wal` or `-shm` file beside it.
- **Gate W0 runs before the launch,** all three clauses of prompt §5. The first clause's list of
  modules the harness imports is taken from a dry run at the parent. The other two are compared by
  bit pattern (`float.hex`). Save W0's outputs in `out/`.
- **The predictions** go in `state/predictions.md`, **after W0 and before the launch**:
  - prompt §5's W1–W4 text, unchanged (copied, not paraphrased);
  - W2's selection rule, fixed so that it determines the 24 calls mechanically. Its mtime is the
    review's evidence; do not edit it after the launch. Anything you add after the launch goes in
    another file.
- **What each record carries.** Every checkpoint record carries its script's SHA-256, its tree's
  `HEAD` and `git status --short`, and each unit's evidence of where its modules were loaded from
  (so a reader can see a $b$ unit imported from the code worktree and a parent unit from the parent
  worktree).
- **W3's evidence.** For every region $b$ accepts as phase-limited at `1e-7`, record per region the
  region's own floor and its two halves' floors, recomputed from the halves' samples in the harness
  (prompt §5, W3), so that W3 can be re-scored by an independent reader from your records.
- **W2's inputs.** Save, for each of the 24 calls, the inputs needed to recompute its reference
  (the integrand, phase and span as the call saw them), in a form a separate reader can reload.
  The orchestrator recomputes the reference on two of them.
- **At most six worker processes.** The census starts no Ray: none of its processes imports `ray`
  or starts a Ray session. (R15/R16 legitimately run Ray under `/tmp/ray-sgk` during the same
  period; that is not yours.) If any census process starts Ray, that is a stop.
- **The budget** is 300 s per unit, enforced from the dispatcher, with 06a's over-budget pass.
- **`convergence_kinds.py`** is 06a's changed copy, run under $b$ as 06a ran it.
- **The run** is one registered, detached run with slug `census`
  (`--campaign levin-phase-floor --prompt 06b`), job name `levin-phase-floor-06b`, as 06a's was
  launched.
- **The checkpoint and per-unit records (pickles) are kept in full** in your directory. 07's census
  reads them later. Do not prune them.

### Sequencing

Do your preparation — the harness copy and its diff, the store checks, W0, and
`state/predictions.md` — and then **end your turn before the launch**, reporting that you are
ready, with W0's verdict and the exact launch command you will run. The orchestrator tells you when
to launch (after R15 and R16 are over, so the two do not compete for the CPUs and for the 300 s
budget). Then launch, make one check, and end your turn with the launch report the common brief
lists. You will be resumed with the final `status.json`, and then you score W1–W4 and write the
descriptive items and your records.

If W0 fails, that is a stop (prompt §7): report it, and do not launch.
