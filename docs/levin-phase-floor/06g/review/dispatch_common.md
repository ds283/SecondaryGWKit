## Common to the three Part B agents of prompt 06g (levin-phase-floor campaign), on the Mac Studio

You are one of three fresh-context agents running Part B of prompt 06g of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you, and it reviews your records afterwards. Today is 2026-11-01 (UTC).

**Your prompt** is `prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md`, read in the code
worktree (below). Read it in full: §0–§3 explain the code you are measuring. Also read the
`CLAUDE.md` of the code worktree, which binds you: the run registry, the shared rehearsal lock, the
venv and `black`. Your scope is the section named in your own brief below. The other sections are
the other agents'.

The prompt points to earlier machinery: "as 06h §4", "06h's census", 06i's Filon references and
06c's Z3 calls. So read:
- `prompts/levin-phase-floor/06h-group-phase-rounding.md` §4 and §5;
- 06h's Part B records in `docs/levin-phase-floor/06h/rehearsal/`, `…/06h/census/` and
  `…/06h/phase/` (their `log.md`, `verification.md`, `issues.md`, `scripts/` and `state/`), and
  `docs/levin-phase-floor/06h/KEPT_ON_MAC_STUDIO.md`;
- whatever else your own brief names.

Read the code log `prompts/levin-phase-floor/logs/06g-leaf-values-and-node-sum.md` as context, all
of it. Its deviations 1, 6 and 12, its "Observations not acted on" (1 and 6 above all) and "State
handed on" matter most.

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/*/review/`, `docs/levin-phase-floor/*/review.md`, or the main checkout's
`var/levin-phase-floor-06g/review/` beyond these brief files. The one exception is a reviewer's
script that your own brief names because a record you must reproduce used it.

### Where things are

- **Part B's dispatch SHA** is `1c5336e7e0ab8ed4495601a4033841f2df17ebdd` (`1c5336e`).
  - It adds only records to 06g's code commit `3ae6f3a499a2c7c9d0b38d1f418cc8fe85e66054`
    (`3ae6f3a`). `git diff --stat 3ae6f3a 1c5336e -- . ':!docs' ':!prompts'` is empty.
  - This is the tree $g$.
- **The code commit's parent** is `4f4f9ed2a31fb7437fcfca01263d260eebaff842` (`4f4f9ed`). Its code
  is 06h's: `git diff --stat c1ec3b2 4f4f9ed -- . ':!docs' ':!prompts'` shows only `CLAUDE.md`.
- **06h's census tree** is `c1ec3b2` (code `7172a8f` + tests-only `ac1c6f4`).
  `git diff --stat c1ec3b2 3ae6f3a -- . ':!docs' ':!prompts'` touches `AdaptiveLevin/` (the module
  and four test files), `tools/` (`filon_reference.py` and its test) and `CLAUDE.md`, nothing else.
  The orchestrator re-checked both diffs today.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf0`.
  - It holds the venv, the stores, `var/` and the run registry.
  - The records go in its `var/levin-phase-floor-06g/`.
  - **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-4f4f9ed`, detached at
  `4f4f9ed`. R23 and Gate V's reproduction of 06h's units run here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-1c5336e`, detached at
  `1c5336e`. These run here:
  - R24;
  - the tree $g$ of §5: the census, the captures and the Filon references;
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
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06g/`
  back and commits it.
- **`HEAD` is not yours.** Do not assume any tree's `HEAD` reflects your work. Check each tree's
  `HEAD` and `git status --short` wherever you rely on them.

### Facts (check what you rely on; no fact here is to be trusted over the code or the store)

**Re-checked by the orchestrator on the Mac Studio at 11:00–11:10 UTC today, read-only**
(`var/levin-phase-floor-06g/review/preflight/`):
- **The trees.** The main checkout is `28ebaf0`, `git status --short` empty. Both 06g worktrees are
  clean (`git status --short --ignored` empty) at `4f4f9ed` and `1c5336e`. `git worktree list`
  shows these three trees only.
- **The registry** holds 40 runs, all finished, nothing `running` or stale.
- **The lock** is free (`~/.raypoolkit/rehearsal-lock` does not exist), and
  `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none. `status` takes `--ray-temp-dir` and
  `--ray-address` but **not** `--project`.
- **No earlier 06g state.** There was no `var/levin-phase-floor-06g*` before the orchestrator's
  `review/`, and there is no `var/datastores/levin-phase-floor-06g*`.
- **The scratch copy's** five files match `copy_files_certified.txt` by SHA-256, with no journal,
  `-wal` or `-shm` file beside them.
- **06h's census** `checkpoint.jsonl` has SHA-256
  `3104c18ed8b119f1798aec75be98d5b39bce641f3b5623b4c1cdc5b18b5ef372`; the run directory is 2.3 GB,
  149 files.
- **The host.** 10 cores. `~/.ray/auth_token` is mode 600, and 96 GiB are free. The load was high
  at 11:00 from Spotlight indexing the new worktrees, with no Python of this or another project
  running.

**The rest:**
- **A3 v2's scratch copy** is `var/levin-phase-floor-06/wider/state/a3v2_copy/`, five files.
  - Its certification is `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`: primary
    `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…` and `ecc3330d…`.
  - Its fingerprint and A3 v2's are both
    `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06h's Part B** ran on this machine on 2026-11-01. Its records are committed in
  `docs/levin-phase-floor/06h/`; read them in the code worktree. The uncommitted rest is in the main
  checkout's `var/levin-phase-floor-06h/`, which `docs/levin-phase-floor/06h/KEPT_ON_MAC_STUDIO.md`
  lists. **Read it only.**
  - **R21 and R22** both gave `1b332ffa…`. Their launches are in
    `docs/levin-phase-floor/06h/rehearsal/state/launch_r21.txt` and `launch_r22.txt`. Their stores
    were retired; the run directories remain.
  - **06h's census run,** `var/runs/levin-phase-floor-06h-census-20261101T005037/` (2.3 GB), is the
    "before" of §5. Read it from its `checkpoint.jsonl` (`3104c18e…`) and its per-unit records.
  - 06h's census harness is `docs/levin-phase-floor/06h/census/scripts/`, byte-identical to the run.
- **R8's fingerprint,** which R23 and R24 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R22 gave it.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest` and
  work counters (`[05-store-value-diff-compares-volatile-columns]`).
- **06g reaches `AdaptiveLevin` everywhere it is called,** so `QuadSourceIntegral` values (`total`
  and the analytic oracle) are expected to move. The store fingerprint certifies identities, tags
  and row counts, not values (`CLAUDE.md` rule 7).

### The addendum the orchestrator's note records at Part B's dispatch (verbatim)

### B.2 Addendum, 2026-11-01: Part B's dispatch (§B.1)

**Part A** is the code commit **`3ae6f3a`**, whose parent is **`4f4f9ed`**, reviewed (§A.5) and
pushed.

**The trees, by SHA:**
- **R23 and Gate V's reproduction** at **`4f4f9ed`**, in the parent worktree. Its code is 06h's
  census tree's: `git diff --stat c1ec3b2 4f4f9ed -- . ':!docs' ':!prompts'` shows only
  `CLAUDE.md` (checked on the laptop).
- **R24, $g$ and the references** at **this addendum's commit**, in the code worktree. Its code is
  `3ae6f3a`'s: the commit adds only records. Gate V's second clause, `git diff --stat c1ec3b2
  3ae6f3a -- . ':!docs' ':!prompts'`, touches `AdaptiveLevin/` (the module and four test files),
  `tools/` (the tool and its test) and `CLAUDE.md`, and nothing else (checked on the laptop).
- **06h's census** `levin-phase-floor-06h-census-20261101T005037` is the "before", its
  `checkpoint.jsonl` `3104c18e…` (`docs/levin-phase-floor/06h/KEPT_ON_MAC_STUDIO.md`).
- **Z3's 30 calls** are `docs/levin-phase-floor/06c/depth20-tiling/results.json`'s `calls`; 06i's
  13 references are `docs/levin-phase-floor/06i/out/readings/`.

**What Part B must carry:**
- **The census harness** wraps `_roundoff_exit_premise_holds` to count premise checks (06h's); the
  function now takes `split_could_meet_budget=`, which a wrapper must pass through (the log's
  Observation 6). Each harness change is a deviation, its diff saved.
- **The tool's command line:** `python -m tools.filon_reference PICKLE --tolerance T --unpickler
  ray.cloudpickle --workers 6`, with $g$ and the census harness on `PYTHONPATH` (the captures name
  the harness's functions). Route B's default edges walk the captured integrand's closures as 06i
  did; a capture whose closures differ raises, and its edges must be passed. The tool was not run
  on 06i's 13 captures on the laptop (the log's deviation 12): §5's comparison with 06i's readings
  is that check, and a disagreement is a §7 stop.
- **V8 may fail.** The issue `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`
  shows synthetic fallback calls far from the origin that the parent converges and 06g stops
  node-limited; on the laptop's draws two calls lost convergence to (b), none to `node_limited`.
  The user ruled to run Part B as written: each lost call is reported with its cause, and the §5
  stop "the noise clause or the slow threshold costs convergence on more than a handful of rows"
  (§5 of this note) stands.
- **The near-singular class** moved 6 honest → dishonest and 10 the other way on the draws; if A3 v2
  reaches it, report it as such.

### Six cores, across everything

**At most six cores are in use at any time on this host,** across all three agents and the
orchestrator. So the work is strictly sequential, and each stage starts only when the orchestrator
tells you:
1. R23 (Ray, `--cpus 6`), then R24 (Ray, `--cpus 6`);
2. the rehearsal agent's comparison, V1 and readers, then the lock's release;
3. the census agent's Gate V, then the census (six workers), then its scoring;
4. the references agent's captures, its comparison of 06i's 13, and the references (at most six
   processes), then its scoring.

While another stage runs, an agent does no computation beyond reading files, copying scripts,
diffing and writing notes. These all count as computation:
- hashing the scratch copy, a store or a checkpoint;
- Gate V;
- fingerprinting;
- the value comparison;
- every unit, capture, replay or Filon reading;
- drawing the random sample;
- scoring from a checkpoint.

### Rules for all of you

- **Never write to** any of these:
  - A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`);
  - its scratch copy, which you read only, opened read-only;
  - any earlier prompt's records, stores or runs. That covers `var/levin-phase-floor-0*` other than
    `-06g`, `var/datastores/levin-phase-floor-0*` other than `-06g-r23*`/`-06g-r24*`, and
    `var/runs/` other than your own runs. It includes 06h's census run directory, 06e's
    `var/levin-phase-floor-06e/` (06i's 13 capture pickles are there) and 06c's records, which you
    only read;
  - `var/levin-phase-floor-06g/review/`, the orchestrator's.

  Never end another session's run. Never touch anything of prompt 07's (`var/levin-phase-floor-07*`,
  any stash).
- **Ray.** Only R23's and R24's own drivers, and the readers that §4 runs under the lock, may start
  a Ray session.
  - Production modules import the `ray` package, and that is fine; so is `ray.cloudpickle` as an
    unpickler. What must never happen elsewhere is Ray being **initialised**, a Ray session
    appearing.
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
- **Stop and ask** at any of prompt 06g §7's conditions that apply to your section. Also stop at
  anything this brief or the prompt does not settle that changes what you measure.
  - To stop, end your turn with the question stated plainly **as its first line**, then the state of
    every file and run you have made.
  - Do not guess and do not repair.
  - Where the prompt's literal wording and the correct measurement part, do not let the wording
    block the right answer: do what is correct, and record it as a deviation with both readings. If
    the two readings would give different verdicts, stop and ask.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-06g/`:
  - `log.md`: your section, in campaign README §5.1's form (`prompts/levin-phase-floor/README.md`
    §5.1), with **every heading of it**, **What shipped** included. It is for appending to
    `logs/06g-leaf-values-and-node-sum.md`;
  - `verification.md`: your part of `docs/levin-phase-floor/VERIFICATION.md` §6g, with links in the
    `06g/…` form. The laptop commits your directory as `docs/levin-phase-floor/06g/<your
    directory>/`;
  - `issues.md`: anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`;
  - `KEPT_ON_MAC_STUDIO.txt`: every pickle and every file over 2 MB you made, with SHA-256 and size.
    The laptop does not copy them.
