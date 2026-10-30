## Common to both Part B agents of prompt 06c (levin-phase-floor campaign), on the Mac Studio

You are one of two fresh-context agents running Part B of prompt 06c of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you; it reviews your records afterwards. Today is 2026-10-29.

**Your prompt** is `prompts/levin-phase-floor/06c-global-error-control.md`, read in the code
worktree (below). Read it in full, §8's five amendments included (they explain the code you are
measuring), and the `CLAUDE.md` of the code worktree, which binds you (the run registry, the shared
rehearsal lock, the venv, `black`). Your scope is the section named in your own brief below; the
other section is the other agent's. **Do not read anything under
`prompts/levin-phase-floor/orchestrator/`.**

**The code under test is 06c's and 06d's together.** Prompt 06d
(`prompts/levin-phase-floor/06d-cc-roundoff-floor.md`) landed after 06c's code and before this
Part B, and has no Part B of its own: it gave every Clenshaw–Curtis fallback solve a round-off floor
of its own in place of the Levin solve's eq. (151), and guarded a division in the depth-18 log.
Read 06d's prompt (§0 and §5) and its log `prompts/levin-phase-floor/logs/06d-cc-roundoff-floor.md`
("Observations not acted on" and "State handed on") **as context** for what you will see. Your
scope stays prompt 06c's §4 or §5; 06d adds no predictions and no checks of its own.

### Where things are

- **Part B's dispatch SHA** is `e5764a6708757cb11704b95bcd66825baab4ba29` (`e5764a6`). It adds only
  records to 06d's code commit `3d70db93a81f1a9da7bec682cbcb8a1782926fe6` (`3d70db9`): `git diff
  --stat 3d70db9 e5764a6 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`. Its code is 06c's
  (`ad9a5f489f241e53effef2037cc6a75fe8cb8542`, `ad9a5f4`) plus 06d's (`3d70db9`).
- **06c's code commit's parent** is `fef401c2614da3dbbf1b4261edd31f5f7f80cdbf` (`fef401c`), whose
  code is 06b's.
- **06b's Part B dispatch commit** (06b's census tree) is
  `92bc62f09711d10ef0035663d4bbe5629e145e17` (`92bc62f`): `git diff --stat 92bc62f fef401c -- .
  ':!docs' ':!prompts'` shows only `CLAUDE.md`.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf0`, holds the venv, the stores, `var/` and the run registry. The records go in its
  `var/levin-phase-floor-06c/`. **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-fef401c`, detached
  at `fef401c`: R17 and X0's reproduction of 06b's census units run here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6`, detached at
  `e5764a6`: R18, the tree $g$ of §5, the value comparison and the readers run here. The prompts
  are read here.
- **In both worktrees:**
  - every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` by that
    absolute path, with `PYTHONPATH` set to that worktree and `-B` (or
    `PYTHONDONTWRITEBYTECODE=1`);
  - nothing is written in either worktree, no `__pycache__` included.
- **Stores' paths are absolute,** into the main checkout's `var/datastores/`. Run directories are in
  the main checkout's `var/runs/` (a run launched from a worktree is recorded there).
- **No git command that writes,** in any tree, `fetch` included. Read git with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06c/`
  back and commits it.
- **`HEAD` is not yours.** Do not assume any tree's `HEAD` reflects your work; check each tree's
  `HEAD` and `git status --short` where you rely on them.

### Facts (from earlier records; check what you rely on — no fact here is to be trusted over the code or the store)

- **Re-checked by the orchestrator on the Mac Studio at 12:32–12:40 UTC today, read-only:**
  - the main checkout is `handover-remedial` at `28ebaf0`, `git status --short` empty;
    `git worktree list` shows three trees, the main checkout and the two worktrees above, both
    clean (`git status --short --ignored` empty) at `fef401c` and `e5764a6`;
  - the registry holds 28 runs, all finished, nothing `running` or stale;
  - the lock is free (`~/.raypoolkit/rehearsal-lock` does not exist), and `ray-processes
    --ray-temp-dir /tmp/ray-sgk` lists none. Note: `status` takes `--ray-temp-dir` and
    `--ray-address` but **not** `--project`;
  - there is no `var/levin-phase-floor-06c*` and no `var/datastores/levin-phase-floor-06c*`;
  - the scratch copy's five files match `copy_files_certified.txt` by SHA-256, with no journal,
    `-wal` or `-shm` file beside them;
  - 06b's census `checkpoint.jsonl` has SHA-256 `f934ea245c0086529fb9372a9ad457d4d8668114ce346dca5440404c766646ad`;
  - `~/.ray/auth_token` is mode 600; 108 GiB free. The load is about 8, from desktop processes,
    with no Python of this project running (some idle `multiprocessing` workers of a Python 3.14
    from 13 days ago, not this project's venv, at 0% CPU; leave them alone).
- **A3 v2's scratch copy** is `var/levin-phase-floor-06/wider/state/a3v2_copy/` (five files). Its
  certification is `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`: primary
  `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…` and `ecc3330d…`. Its fingerprint and
  A3 v2's are both `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06b's Part B** ran on this machine on 2026-10-28. Its records are committed in
  `docs/levin-phase-floor/06b/rehearsal/` and `…/06b/census/` (read them in the code worktree), and
  the uncommitted rest is in the main checkout's `var/levin-phase-floor-06b/`
  (`docs/levin-phase-floor/06b/KEPT_ON_MAC_STUDIO.md` lists it).
  - **R15 and R16** (`levin-phase-floor-06b-r15-parent-20261028T135349`,
    `…-r16-premise-20261028T135745`) both gave `1b332ffa…`. Their launches are in
    `docs/levin-phase-floor/06b/rehearsal/state/`.
  - **06b's census run** `var/runs/levin-phase-floor-06b-census-20261028T161212/` (1.7 GB):
    18,560 of 18,560 units in 51 minutes, none over budget. It is the "before" of §5, read from its
    `checkpoint.jsonl` (`f934ea24…`) and its unit pickles. **Read it only.**
  - 06b's census harness is `docs/levin-phase-floor/06b/census/scripts/`, byte-identical to the
    run.
- **R8's fingerprint,** which R17 and R18 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R16 gave it.
- 05's `classify_non_qsi.py` and `tabulate_qsi.py` are in `docs/levin-phase-floor/05/rehearsal/scripts/`.
  The readers are `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`
  and work counters (`[05-store-value-diff-compares-volatile-columns]`).
- **Every computed value is expected to move** (prompt §0.3; the user accepted it). The store
  fingerprint certifies identities, tags and row counts, not values (`CLAUDE.md` rule 7).

### The addendum the orchestrator's note records at Part B's dispatch (verbatim)

### B.5 Addendum: Part B's dispatch

**Part A** is the code commit **`ad9a5f4`**, whose parent is **`fef401c`**. It re-lands `c85f753`'s
code byte for byte (`git diff c85f753 ad9a5f4 -- AdaptiveLevin ComputeTargets RunDrivers
docs/levin-phase-floor/06c` is empty), so `c85f753`'s review (§A.8) carries over: the tests, the
four breakages, the 20 LiouvilleGreen draws re-run by the orchestrator to 0 differences. §A.9's
re-review: scope as dispatched, stashes and `var/` untouched, the seven roots re-run by the
orchestrator on `ad9a5f4` (AdaptiveLevin 53, the other six equal, all OK). Pushed.

**The trees, by SHA:**
- **R17 and X0's reproduction** at **`fef401c`**, in the parent worktree. Its code is 06b's:
  `git diff --stat 92bc62f fef401c -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`, so X0's
  first clause holds against 06b's census tree.
- **R18 and $g$** at **this addendum's commit**, in the code worktree. Its code is `ad9a5f4`'s:
  the commit adds only records.

**What Part B must carry:**
- **Cost.** On the laptop's draws solves rose ×45 and 230 of 4,320 calls reached depth 20. Where
  a request cannot be met, or round-off sits above the floor, the loop refines to exhaustion.
  06b's 300 s per unit may be exceeded; an over-budget unit counts against X5 as the prompt
  says. **R17 and R18 may run much longer than R15/R16 did.**
- **X1 is the honesty question.** The trust check narrows but does not close
  `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`; an error common to a leaf and its
  descendants, `[06b-moved-calls-declare-less-than-their-error]`'s, is out of its reach.
- **The trust check's descriptive item** reads `used_interval`'s `delta1`, `delta2`,
  `convergence_ratio`, `trusted` and `declared_resolution`, and `num_untrusted_regions`.

**§B.0 re-checked on the Mac Studio at 01:46 UTC, 2026-10-29,** read-only:
- the main checkout is `handover-remedial` at `28ebaf0`, `git status --short` empty, one worktree;
- the registry holds 28 runs, all finished, nothing `running` or stale;
- the lock is free, and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none;
- there is no `var/levin-phase-floor-06c*` and no `var/datastores/levin-phase-floor-06c*`;
- the scratch copy's five files match their certification by SHA-256, with no `-wal` or `-shm`;
- 06b's census checkpoint is `f934ea24…`, as `06b/KEPT_ON_MAC_STUDIO.md` records;
- 108 GiB free; load about 5.

**The worktrees,** made by the laptop after this commit is pushed, detached, with no venv:
`SecondaryGWKit-lpf06c-fef401c` and `SecondaryGWKit-lpf06c-<this commit's short SHA>`.

### B.6 Addendum, 2026-10-29: Part B waits for 06d

Before Part B was started, 07's Part A, rebased onto 06c, stopped a third time and found two
defects in code 06c's census would run: the Clenshaw–Curtis fallback carries the Levin solve's
eq. (151) floor, which grows at a stationary point, and the depth-18 diagnostic can divide by zero
under 06c's loop (prompt 06d §0.1). The user: *"06d prompt before Part B"*, and *"Yes, go with that
ordering."* So:
- **Part B runs after 06d's code lands,** on 06c's and 06d's code together, and its census
  measures both; 06d has no Part B of its own.
- **R17** still runs at `fef401c` (06b's code), and X0 still reproduces 06b's census there.
- **R18 and $g$** run at the commit that records 06d's landing, whose code is 06c's plus 06d's. The
  worktree `SecondaryGWKit-lpf06c-ad63e8a` made for §B.5 is **not** used for them; it is removed
  and a new one made at that commit.
- **The header issues of both prompts** close in 06c's records commit, if 06c's §4 holds and X0–X4
  hold; otherwise ask the user.

**06d landed** on 2026-10-29: code `3d70db9`, reviewed and pushed, then the residue commit that
records it. **R18 and $g$ run at that residue commit**, whose code is `3d70db9`'s (06c's plus
06d's); R17 stays at `fef401c`. The worktree for R18 and $g$ is
`SecondaryGWKit-lpf06c-<the residue commit's short SHA>`, made by the laptop after it is pushed;
`SecondaryGWKit-lpf06c-ad63e8a` is removed first. What Part B must also carry, from 06d's log:
- **The cost moves down where stationary points were chased:** in the laptop's draws, solves
  5.94 M → 3.38 M and depth-20 calls 230 → 101.
- **The node term is a worst-case bound** (06d's log, Observations 2): on `mode_filter_tight` it
  runs about 100× above the realised error and turns a converged call unconverged, honestly.
  Where $|x|$ is large it may bind; X5 and the depth distribution will show it.
- **X1's honesty count** includes the near-singular cases 06d uncovered, if A3 v2 reaches them
  (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, widened by 06d).

### What else Part B must carry (the laptop's dispatch, 2026-10-29)

- **Cost.** On the laptop's LiouvilleGreen draws, 06c alone raised solves about 45× over 06b; 06d
  then cut them by about 40% (5.94 M → 3.38 M) and the calls reaching depth 20 from 230 to 101,
  mostly at stationary points. R18 and the census may still run far longer than 06b's did. An
  over-budget census unit counts against X5, as the prompt says; that is **not** a reason to change
  the budget.
- **Two of 06d's findings bear on the scoring:**
  - 06d's node-rounding term is a worst-case bound. On one problem it runs about 100× above the
    realised error and turns a converged call unconverged, though still honest. Where $|x|$ is
    large it may bind.
  - 06d's draws found near-singular three-Bessel cases reported converged and dishonest
    (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, widened by 06d).

  X1 is scored as written, and anything bearing on either point is a finding to report.

### Rules for both of you

- **Never write to** A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`), its scratch copy (read it
  only, opened read-only), or any earlier prompt's records, stores or runs (`var/levin-phase-floor-0*`
  other than `-06c`, `var/datastores/levin-phase-floor-0*` other than `-06c-r17*`/`-06c-r18*`,
  `var/runs/` other than your own runs). That includes 06b's census run directory, which you only
  read. Never end another session's run. Never touch anything of prompt 07's.
- **Ray.** No Ray session may be started by anything but R17's and R18's own drivers. Production
  modules import the `ray` package, and that is fine; what must never happen outside R17/R18 is Ray
  being **initialised** (a Ray session appearing). Every Ray driver names its address; `ray stop`
  is never run.
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
- **Stop and ask** at any of prompt 06c §7's conditions that apply to your section, and at anything
  this brief or the prompt does not settle that changes what you measure. To stop, end your turn
  with the question stated plainly and the state of every file and run you have made. Do not guess
  and do not repair.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a
  deviation to record, not a reason to take it.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-06c/`:
  - `log.md` — your section, in campaign README §5.1's form
    (`prompts/levin-phase-floor/README.md` §5.1), **every heading of it** (including **What
    shipped**), for appending to `logs/06c-global-error-control.md`;
  - `verification.md` — your part of `docs/levin-phase-floor/VERIFICATION.md` §6c, "Global error
    control", with links in the `06c/…` form (the laptop commits your directory as
    `docs/levin-phase-floor/06c/rehearsal/` or `…/06c/census/`);
  - `issues.md` — anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`.


## Your brief: the census agent — prompt 06c §5 (06b's census re-run under $g$)

**Your scope is prompt 06c §5**, and only it. §4 (rehearsals R17 and R18, under Ray and the lock)
is the other agent's, and runs first.

- **Where you write:** only `var/levin-phase-floor-06c/census/` (with `out/`, `scripts/`, `state/`)
  and your own run directory in `var/runs/`. You take no lock, and no process of yours initialises
  Ray (importing the `ray` package, as production modules do, is fine; a Ray session appearing is
  not).
- **The harness** is 06b's census copy, from `docs/levin-phase-floor/06b/census/scripts/` (in the
  code worktree), copied into `var/levin-phase-floor-06c/census/scripts/`. Change only what prompt
  §5 allows. The diff against 06b's copy is saved in `out/`, and each change is a numbered
  deviation.
- **The trees:** X0's reproduction of 06b runs at the parent worktree (`fef401c`); $g$ is the code
  worktree (`e5764a6`). 06b's census tree was `92bc62f` (in git history; read it with
  `git --no-optional-locks`, never by making a worktree).
- **The "before"** is 06b's census run `var/runs/levin-phase-floor-06b-census-20261028T161212/`,
  read from its `checkpoint.jsonl` (`f934ea24…`) and unit pickles. Read it only, and confirm its
  checkpoint's SHA-256 before and after.
- **The store** is the scratch copy, opened read-only, re-checked against its certification and
  fingerprint before the run (with 06b's copied `fingerprint_copy.py` or its equivalent), and hashed
  (SHA-256) before and after, with no journal, `-wal` or `-shm` file beside it.
- **Gate X0 runs before the launch,** all three clauses of prompt §5:
  - the first clause is against `92bc62f`; the list of modules the harness imports is taken from a
    dry run at the parent;
  - the second (06b's units `low-G+r-1` and `ctrl-low-1` at `1e-7`, `0-1971` at `1e-8`, reproduced
    at the parent) and the third (two runs of one $g$ unit) are compared by bit pattern
    (`float.hex`).
  Save X0's outputs in `out/`.
- **The predictions** go in `state/predictions.md`, **after X0 and before the launch**:
  - prompt §5's X1–X5 text, unchanged (copied, not paraphrased);
  - X1's selection rule, fixed so that it determines the 24 calls mechanically (it may name 06b's
    calls that "moved" only once $g$ has run, so state the rule so that it is mechanical given the
    run). Its mtime is the review's evidence; do not edit it after the launch. Anything you add
    after the launch goes in another file.
- **What each record carries.** Every checkpoint record carries its script's SHA-256, its tree's
  `HEAD` and `git status --short`, and each unit's evidence of where its modules were loaded from
  (so a reader can see a $g$ unit imported from the code worktree and a parent unit from the parent
  worktree). Per call, record what X1–X5 and the descriptive items need, in particular each
  returned leaf's `used_interval` fields of the trust check (`delta1`, `delta2`,
  `convergence_ratio`, `trusted`, `declared_resolution`), its residual, floor and truncation, its
  depth and branch, and the call's `num_untrusted_regions`, so that the orchestrator can re-score
  X2 and X3 from the two checkpoints with its own reader.
- **X1's inputs.** Save, for each of the 24 calls, the inputs needed to recompute its reference
  (the integrand, phase and span as the call saw them), in a form a separate reader can reload. The
  orchestrator recomputes the reference on two of them.
- **At most six worker processes.** A Ray session under `/tmp/ray-sgk` (or anywhere) started by a
  census process is a stop.
- **The budget** is 300 s per unit, enforced from the dispatcher, with 06a's over-budget pass, as
  06b ran it. Over-budget units count against X5; do not change the budget.
- **`convergence_kinds.py`** is 06a's changed copy (as 06b's census carried it), run under $g$ as
  06b ran it.
- **The run** is one registered, detached run with slug `census`
  (`--campaign levin-phase-floor --prompt 06c`), job name `levin-phase-floor-06c`, as 06b's was
  launched.
- **The checkpoint and per-unit records (pickles) are kept in full** in your run directory and
  records. 07's census reads them later. Do not prune them.

### Sequencing

Do your preparation — the harness copy and its diff, the store checks, X0, and
`state/predictions.md` — while the rehearsals run (keep it light: X0's few units only, no bulk
computation), and then **end your turn before the launch**, reporting that you are ready, with
X0's verdict and the exact launch command you will run. The orchestrator tells you when to launch
(after R17 and R18 are over and the lock is released, so the two do not compete for the six cores
and for the 300 s budget). Then launch, make one check, and end your turn with the launch report
the common brief lists. You will be resumed with the final `status.json`, and then you score X1–X5
and write the descriptive items and your records.

If X0 fails, that is a stop (prompt §7): report it, and do not launch.
