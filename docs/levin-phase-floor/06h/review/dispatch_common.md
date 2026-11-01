## Common to the three Part B agents of prompt 06h (levin-phase-floor campaign), on the Mac Studio

You are one of three fresh-context agents running Part B of prompt 06h of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you, and it reviews your records afterwards. Today is 2026-11-01 (UTC).

**Your prompt** is `prompts/levin-phase-floor/06h-group-phase-rounding.md`, read in the code
worktree (below). Read it in full: §0–§3 explain the code you are measuring. Also read the
`CLAUDE.md` of the code worktree, which binds you: the run registry, the shared rehearsal lock, the
venv and `black`. Your scope is the section named in your own brief below. The other sections are
the other agents'.

The prompt points to earlier machinery: "as 06e §4", "06e's census" and "06f's M2". So read:
- `prompts/levin-phase-floor/06e-exact-end-nodes.md` §4 and §5;
- 06e's Part B records in `docs/levin-phase-floor/06e/rehearsal/` and `…/06e/census/`;
- 06f's records in `docs/levin-phase-floor/06f/`, everything the prompt's "Read first" item 7
  names. That includes `docs/levin-phase-floor/06f/review.md` (§4 and Observation 1) and
  `docs/levin-phase-floor/06f/review/scripts/orch_m2.py`.

Read the code log `prompts/levin-phase-floor/logs/06h-group-phase-rounding.md` as context, all of
it. Its "Observations not acted on" and "State handed on" matter most.

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/06e/review/`, `docs/levin-phase-floor/06e/review.md`, or the main
checkout's `var/levin-phase-floor-06h/review/` beyond these brief files. 06f's `review.md` and
`review/scripts/orch_m2.py` are the exceptions above, because the prompt names them.

### Where things are

- **Part B's dispatch SHA** is `c1ec3b2c819cc5df22ddf4b303f69c7129a54d62` (`c1ec3b2`).
  - It adds only records to 06h's code: the code commit `7172a8f52d9b924ccb44f75598ca53d5c4879de2`
    (`7172a8f`) and its tests-only follow-up `ac1c6f415345b15cdc40991cfb134e531512a3aa` (`ac1c6f4`).
  - `git diff --stat ac1c6f4 c1ec3b2 -- . ':!docs' ':!prompts'` is empty.
  - This is the tree $h$.
- **The code commit's parent** is `2c325f56641666b45df483adb8c1202575721c42` (`2c325f5`). Its code
  is 06e's.
- **06e's Part B dispatch tree** is `6d81fad43cbf1f17cff8d0f48fc7b65966e343ab` (`6d81fad`).
  `git diff --stat 6d81fad 2c325f5 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`. The
  orchestrator re-checked this today.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf0`.
  - It holds the venv, the stores, `var/` and the run registry.
  - The records go in its `var/levin-phase-floor-06h/`.
  - **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-2c325f5`, detached at
  `2c325f5`. R21, Gate H's reproduction of 06e's units, and the phase re-measurement's parent column
  run here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-c1ec3b2`, detached at
  `c1ec3b2`. These run here:
  - R22;
  - the tree $h$ of §5: the census, and the phase re-measurement's $h$ column;
  - the value comparison and the readers.

  The prompts are read here.
- **In both worktrees:**
  - every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` by that
    absolute path;
  - `PYTHONPATH` is set to that worktree, plus your copied `scripts/` directory where pickles need
    `harness`;
  - use `-B` or `PYTHONDONTWRITEBYTECODE=1`;
  - nothing is written in either worktree, `__pycache__` included.
- **Stores' paths are absolute,** into the main checkout's `var/datastores/`. Run directories are in
  the main checkout's `var/runs/`; a run launched from a worktree is recorded there.
- **No git command that writes,** in any tree, `fetch` and `stash` included. Read git with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06h/`
  back and commits it.
- **`HEAD` is not yours.** Do not assume any tree's `HEAD` reflects your work. Check each tree's
  `HEAD` and `git status --short` wherever you rely on them.

### Facts (check what you rely on; no fact here is to be trusted over the code or the store)

**Re-checked by the orchestrator on the Mac Studio at 00:13–00:14 UTC today, read-only**
(`var/levin-phase-floor-06h/review/preflight/`):
- **The trees.** The main checkout is `28ebaf0`, `git status --short` empty. Both 06h worktrees are
  clean (`git status --short --ignored` empty) at `2c325f5` and `c1ec3b2`.
- **The registry** holds 36 runs, all finished, nothing `running` or stale.
- **The lock** is free (`~/.raypoolkit/rehearsal-lock` does not exist), and
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none. `status` takes `--ray-temp-dir` and
  `--ray-address` but **not** `--project`.
- **No earlier 06h state.** There was no `var/levin-phase-floor-06h*` before the orchestrator's
  `review/`, and there is no `var/datastores/levin-phase-floor-06h*`.
- **The scratch copy's** five files match `copy_files_certified.txt` by SHA-256, with no journal,
  `-wal` or `-shm` file beside them.
- **06e's census** `checkpoint.jsonl` has SHA-256
  `a0dc4d7f5c47eb28d8691b450559d005a76e1d9f3bd41442e249a419fb15904c` (1.08 GB).
- **The host.** `~/.ray/auth_token` is mode 600, and 96 GiB are free. The load is about 4.5, from
  macOS and desktop processes, with no Python of this project running.
- **`LiouvilleGreen/tests/test_group_phase_rounding.py`,** run by the orchestrator in the code
  worktree on this host: 8 tests, OK.
  - Test 5 re-measured $\varepsilon_{\exp}$'s basis on this host's `np.exp`: worst relative error
    $1.114619\times10^{-16}$ = 1.0040 u, the same as the laptop's.
  - The scalar and array paths are bit-identical.

**The rest:**
- **A3 v2's scratch copy** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, five files.
  - Its certification is `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`: primary
    `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…` and `ecc3330d…`.
  - Its fingerprint and A3 v2's are both
    `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06e's Part B** ran on this machine on 2026-10-30. Its records are committed in
  `docs/levin-phase-floor/06e/`; read them in the code worktree. The uncommitted rest is in the main
  checkout's `var/levin-phase-floor-06e/`, which `docs/levin-phase-floor/06e/KEPT_ON_MAC_STUDIO.md`
  lists. **Read it only.**
  - **R19 and R20** (`levin-phase-floor-06e-r19-parent-20261030T074440`,
    `…-r20-endnodes-20261030T075034`) both gave `1b332ffa…`. Their launches are in
    `docs/levin-phase-floor/06e/rehearsal/state/launch_r19.txt` and `launch_r20.txt`. Their stores
    were retired; the run directories remain.
  - **06e's census run,** `var/runs/levin-phase-floor-06e-census-20261030T081318/` (1.6 GB), is the
    "before" of §5. Read it from its `checkpoint.jsonl` (`a0dc4d7f…`) and its per-unit records.
  - 06e's census harness is `docs/levin-phase-floor/06e/census/scripts/`, byte-identical to the run.
- **06f** ran on this machine on 2026-10-31.
  - Its records are in `docs/levin-phase-floor/06f/`, and in the main checkout's
    `var/levin-phase-floor-06f/`, which is read-only to you.
  - Its 20 M2 calls are `docs/levin-phase-floor/06f/out/sample/`.
  - Its capture pickles were removed, so each call is captured afresh.
- **R8's fingerprint,** which R21 and R22 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R20 gave it.
- **The readers.** 05's `classify_non_qsi.py` is in `docs/levin-phase-floor/05/rehearsal/scripts/`.
  The readers are `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, and 06e's worktree
  adaptation is `docs/levin-phase-floor/06e/rehearsal/scripts/run_readers_worktree.sh`.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest` and
  work counters (`[05-store-value-diff-compares-volatile-columns]`).
- **Every `QuadSourceIntegral` value is expected to move.** The store fingerprint certifies
  identities, tags and row counts, not values (`CLAUDE.md` rule 7).

### The addendum the orchestrator's note records at Part B's dispatch (verbatim)

### B.2 Addendum, 2026-10-31: Part B's dispatch (§B.1)

**Part A** is the code commit **`7172a8f`** and its follow-up **`ac1c6f4`** (tests only), whose
parent is **`2c325f5`**, reviewed (§A.5) and pushed.

**The trees, by SHA:**
- **R21 and Gate H's reproduction** at **`2c325f5`**, in the parent worktree. Its code is 06e's:
  `git diff --stat 6d81fad 2c325f5 -- . ':!docs' ':!prompts'` shows only `CLAUDE.md` (checked on
  the laptop), so Gate H's first clause holds against 06e's Part B dispatch tree `6d81fad`.
- **R22 and $h$** at **this addendum's commit**, in the code worktree. Its code is `ac1c6f4`'s:
  the commit adds only records.
- **06e's census** `levin-phase-floor-06e-census-20261030T081318` is the "before", its
  `checkpoint.jsonl` `a0dc4d7f…` (`docs/levin-phase-floor/06e/KEPT_ON_MAC_STUDIO.md`).
- **06f's 20 M2 calls** are `docs/levin-phase-floor/06f/out/sample/`; 06f's capture pickles were
  removed (the board's Decisions, **06f results**), so each call is captured afresh at $h$ (and at
  the parent, for H1's parent column), as prompt §5 says.

**Before the run, on the Mac Studio:** run `LiouvilleGreen/tests/test_group_phase_rounding.py` in
the code worktree. Its test 5 re-measures $\varepsilon_{\exp}$'s basis on that host's `np.exp`.
**If test 5 or any other test there fails, stop and ask**: the declaration rests on that bound, and
it was measured on the laptop only (the log's Observation 4).

**What Part B must carry:**
- **The declaration rises by orders where the abscissa term dominates** (the definition test:
  $2.66\times10^{-15} \to 2.79\times10^{-10}$ at $x = 5\times10^5$); prompt §0.2 argues a call's
  declared error barely moves, and H3, H4 and the descriptive cost item test that. On the laptop's
  draws no Levin call's convergence changed, regions fell 68,772 → 61,126, and none reached depth
  20.
- **H1's reference is at the exact $e^u$** (prompt §2); 06f's scripts scored at the float $\hat x$
  except its review's Observation 1, so the copied scripts must change there, each change recorded.
- **The near-singular class** moved by one case honest → dishonest and three the other way on the
  draws; if A3 v2 reaches it, report it as such.

### Six cores, across everything

**At most six cores are in use at any time on this host,** across all three agents and the
orchestrator. So the work is strictly sequential, and each stage starts only when the orchestrator
tells you:
1. R21 (Ray, `--cpus 6`), then R22 (Ray, `--cpus 6`);
2. the rehearsal agent's comparison, H0 and readers, then the lock's release;
3. the census agent's Gate H, then the census (six workers);
4. the phase agent's captures, replays and measurement (at most six workers).

While another stage runs, an agent does no computation beyond reading files, copying scripts,
diffing and writing notes. These all count as computation:
- hashing the scratch copy, a store or a checkpoint;
- Gate H;
- fingerprinting;
- the value comparison;
- every unit, capture or replay;
- scoring from a checkpoint.

### Rules for all of you

- **Never write to** any of these:
  - A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`);
  - its scratch copy, which you read only, opened read-only;
  - any earlier prompt's records, stores or runs. That covers `var/levin-phase-floor-0*` other than
    `-06h`, `var/datastores/levin-phase-floor-0*` other than `-06h-r21*`/`-06h-r22*`, and
    `var/runs/` other than your own runs. It includes 06e's census run directory and
    `var/levin-phase-floor-06f/`, which you only read;
  - `var/levin-phase-floor-06h/review/`, the orchestrator's.

  Never end another session's run. Never touch anything of prompt 07's (`var/levin-phase-floor-07*`,
  any stash).
- **Ray.** Only R21's and R22's own drivers may start a Ray session.
  - Production modules import the `ray` package, and that is fine. What must never happen outside
    R21 and R22 is Ray being **initialised**, a Ray session appearing.
  - Every Ray driver names its address.
  - `ray stop` is never run.
- **Save every command you run and its output** under your `out/`, or under `state/` for launch
  transcripts. The review reads the outputs, not your log.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4).
  - Launch detached: `nohup sh -c …`, manifest first, stdout and stderr into the run directory or
    your records, pid recorded.
  - Verify in **one** check that it started.
  - Then end your turn and report the run id (if registered), the run directory, the detached
    Python pid, the `sh` pid and the first checkpoint or heartbeat line.
  - **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The orchestrator waits on the pid and
    resumes you with the final `status.json` (or the exit).
- **Number and classify every deviation** from the prompt before you report: `STRUCTURALLY
  REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`. Failed attempts are deviations too.
- **Stop and ask** at any of prompt 06h §7's conditions that apply to your section. Also stop at
  anything this brief or the prompt does not settle that changes what you measure.
  - To stop, end your turn with the question stated plainly **as its first line**, then the state of
    every file and run you have made.
  - Do not guess and do not repair.
  - Where the prompt's literal wording and the correct measurement part, do not let the wording
    block the right answer: do what is correct, and record it as a deviation with both readings. If
    the two readings would give different verdicts, stop and ask.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-06h/`:
  - `log.md`: your section, in campaign README §5.1's form (`prompts/levin-phase-floor/README.md`
    §5.1), with **every heading of it**, **What shipped** included. It is for appending to
    `logs/06h-group-phase-rounding.md`;
  - `verification.md`: your part of `docs/levin-phase-floor/VERIFICATION.md` §6h, "The group
    phase's rounding", with links in the `06h/…` form. The laptop commits your directory as
    `docs/levin-phase-floor/06h/<your directory>/`;
  - `issues.md`: anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`;
  - `KEPT_ON_MAC_STUDIO.txt`: every pickle and every file over 2 MB you made, with SHA-256 and size.
    The laptop does not copy them.
