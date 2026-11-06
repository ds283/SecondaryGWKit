## Common to the two Part B agents of prompt 06m (levin-phase-floor campaign), on the Mac Studio

You are one of two fresh-context agents running Part B of prompt 06m of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you. It holds you at planned stops and reviews your records afterwards. Today is
2026-11-06 (UTC).

**Your prompt** is `prompts/levin-phase-floor/06m-signed-yardstick-in-the-driver.md`, read in the
code worktree (below). Read it in full: §0–§3 explain the code you are measuring. Also read the
code worktree's `CLAUDE.md`, which binds you: the run registry, the shared rehearsal lock, the venv
and `black`. Read the campaign README `prompts/levin-phase-floor/README.md` §4.1 and §5 (§5.1 is the
log's form). Your scope is the section named in your own brief. The other section is the other
agent's.

Read the code log `prompts/levin-phase-floor/logs/06m-signed-yardstick-in-the-driver.md` as
context, all of it. Its R1, M0 and M blocks are the predictions you score, copied there before any
number.

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/*/review/`, any `docs/levin-phase-floor/*/review.md`, or the main checkout's
`var/levin-phase-floor-06m/review/` beyond these brief files.

### Where things are

- **The code commit** is `48cf87c0` (06m Part A: `_trust_noise_yardstick`, used by `_check_leaf`).
  Its parent is **`5ff8a5e44be775df299af0f1754123578905b86e`** (`5ff8a5e4`).
- **Part B's dispatch SHA** is **`38f6a828d17b41893574f7344a543ebfb1e01e85`** (`38f6a828`). Between
  `48cf87c0` and it, only records, one test module (`test_levin_leaf_values.py`) and `CLAUDE.md`
  change. `AdaptiveLevin/levin_quadrature.py` at `38f6a828` is SHA-256 `beaf0e23…`. This is the
  tree $m$.
- `git diff --stat 5ff8a5e4 38f6a828 -- . ':!docs' ':!prompts'` touches
  `AdaptiveLevin/levin_quadrature.py`, `test_levin_leaf_values.py`, the new
  `test_levin_trust_yardstick.py` and `CLAUDE.md`, and nothing else.
- `git diff --stat 1c5336e 5ff8a5e4 -- . ':!docs' ':!prompts'` (06g's R24 tree against R25's) touches
  only `.gitattributes`, `CLAUDE.md`, `tools/filon_reference.py` and its test. At `5ff8a5e4` the
  module is `d89c959e…`, R24's. The launcher `RunDrivers/quadsource_atol_sweep.py` is `b0135218…`
  in both 06m worktrees, as in R23's and R24's manifests.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf07`, clean.
  - It holds the venv, the stores, `var/` and the run registry.
  - The records go in its `var/levin-phase-floor-06m/`.
  - **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-5ff8a5e`, detached at
  `5ff8a5e4`. R25 runs here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-38f6a82`, detached at
  `38f6a828`. Everything else runs here: R26, the value comparisons and the readers; Gate M0 and the
  census. The prompts and records are read here.
- **In both worktrees:**
  - every command calls `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python` by that
    absolute path (neither worktree has a venv);
  - `PYTHONPATH` is set to that worktree, plus your copied `scripts/` directory where needed;
  - use `-B` and `PYTHONDONTWRITEBYTECODE=1`;
  - nothing is written in either worktree, `__pycache__` included.
  - `stat` on this host's PATH is GNU coreutils; use `/usr/bin/stat` for BSD flags.
- **Stores' paths are absolute,** into the main checkout's `var/datastores/`. Run directories are in
  the main checkout's `var/runs/`; a run launched from a worktree is recorded there.
- **No git command that writes,** in any tree, `fetch` and `stash` included. Read git with
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-06m/`
  back and commits it.
- **`HEAD` is not yours.** Check each tree's `HEAD` and `git status --short` wherever you rely on
  them.
- **06g's four worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` exist and are
  kept. You do not touch them and you do not run code from them.

### Facts, re-checked by the orchestrator on the Mac Studio at 09:15–09:20 UTC today, read-only

The records are in `var/levin-phase-floor-06m/review/preflight/` and `review/hashes_before/`. Check
what you rely on yourself; no fact here is to be trusted over the code or the store.
- **The trees.** The main checkout is `28ebaf07`, `git status --short` empty. Both 06m worktrees
  are clean (`git status --short --ignored` empty) at `5ff8a5e4` and `38f6a828`. `git worktree list`
  shows the main checkout, 06g's four worktrees and these two.
- **The registry** holds 58 runs: 50 done, 4 failed, 4 killed. Nothing is `running` or stale (a
  grep for "running" matches only 06h-phase's purpose text "refcheck running"; its state is done).
- **The lock** is free (`~/.raypoolkit/rehearsal-lock` does not exist; no
  `~/ScienceData/.rehearsal-lock`), and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none.
  `status` takes `--ray-temp-dir` and `--ray-address` but **not** `--project`.
- **No earlier 06m state.** There was no `var/levin-phase-floor-06m*` before the orchestrator's
  `review/`, and there is no `var/datastores/levin-phase-floor-06m*`.
- **A3 v2's scratch copy,** `var/levin-phase-floor-06/wider/state/a3v2_copy/`: its five files match
  `var/levin-phase-floor-06/wider/state/copy_files_certified.txt` by SHA-256 and size (primary
  `1542e05c…`; shards `bb007a4c…`, `aae63ccd…`, `5e2bda9b…`, `ecc3330d…`), with no journal, `-wal`
  or `-shm` beside them. Its fingerprint and A3 v2's are both
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **R23 and R24.** Their ten store files (`var/datastores/levin-phase-floor-06g-r23*.sqlite` and
  `-r24*.sqlite`) match 06g's `var/levin-phase-floor-06g/rehearsal/state/stores_sha256_final.txt`.
  Their run directories are `var/runs/levin-phase-floor-06g-r23-parent-20261101T110641` and
  `levin-phase-floor-06g-r24-leafvalues-20261101T111215`. **Read only.**
- **Route B's census** (06l), `var/runs/levin-phase-floor-06l-census-routeB-20261105T100126/`:
  150 files, 4.77 GB, each matching `docs/levin-phase-floor/06l/KEPT_ON_MAC_STUDIO.txt` by SHA-256
  and size. Its `checkpoint.jsonl` is
  `5cb3d3c0e6e628720aa122a1d23d4a90f140b116ef9e58e2f66a3bffb9dd0d39`. **Read only.**
- **06g's census,** `var/runs/levin-phase-floor-06g-census-20261101T114315/`: 149 files; its
  `checkpoint.jsonl` is `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, and its
  16 region pickles match `docs/levin-phase-floor/06g/census/KEPT_ON_MAC_STUDIO.txt`. **Read only.**
- **06g's `rows.json`,** `var/levin-phase-floor-06g/census/state/rows.json`, is `aa523d28…`, the
  rows file 06l's census used. 06l's own copy of it (`var/levin-phase-floor-06l/state/rows.json`)
  was deleted on 2026-11-05 with 06l's other large state (`docs/levin-phase-floor/06l/KEPT_ON_MAC_STUDIO.md`).
- **R8's fingerprint,** which R25 and R26 must both give, is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R9–R24 gave it.
- **The host.** 10 cores, 78 GiB free. `~/.ray/auth_token` is mode 600. **The load is high** (about
  22–27 at 09:17–09:19) from system services: WindowServer, ChronoSync, fileproviderd,
  CategoriesService, DaisyDisk, Spotlight. No compute job of this or another project is running.
  The only other Pythons are idle orphaned workers of another project from 15 October (0 % CPU):
  leave them alone. Bear the load in mind when you time things.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`
  and work counters (`[05-store-value-diff-compares-volatile-columns]`).

### The addendum the orchestrator's note records at Part B's dispatch (verbatim)

### B.1 Addendum, 2026-11-06: Part B's dispatch

**Part A** is the code commit **`48cf87c0`**, whose parent is **`5ff8a5e4`**, reviewed (§A.5) and
pushed. Two commits followed it: `427e88c2` (the review's record and two issues) and `56efc6b9`
(the follow-up). The follow-up restates `test_levin_leaf_values.trust_test()` with $\nu_B$ and
adds 06m's `calibration/scripts/` to `CLAUDE.md`'s `black` list. Neither touches production code.

**The trees, by SHA:**
- **R25** at **`5ff8a5e4`**, in the worktree `SecondaryGWKit-lpf06m-5ff8a5e`. Its production code
  is R24's. `git diff --stat 1c5336e 5ff8a5e4 -- . ':!docs' ':!prompts'` touches only
  `.gitattributes`, `CLAUDE.md`, `tools/filon_reference.py` and its test, and nothing the pipeline
  imports (checked on the laptop). So prompt §4's R25-against-R24 comparison is a reproducibility
  check, as written.
- **R26, Gate M0 and the census $m$** at **this addendum's commit**, in the worktree
  `SecondaryGWKit-lpf06m-<this commit's short SHA>`. Its code is `48cf87c0`'s.
  `git diff --stat 5ff8a5e4 <this commit> -- . ':!docs' ':!prompts'` touches
  `AdaptiveLevin/levin_quadrature.py`, two test modules (`test_levin_trust_yardstick.py`, new, and
  `test_levin_leaf_values.py`) and `CLAUDE.md`, and nothing else.
- **R24** is `levin-phase-floor-06g-r24-leafvalues-20261101T111215`, its store
  `var/datastores/levin-phase-floor-06g-r24*.sqlite` (five files), kept on the Mac Studio. **R23 and
  R24 are read only.**
- **The "before"** is route B's census, `levin-phase-floor-06l-census-routeB-20261105T100126`,
  checkpoint `5cb3d3c0e6e628720aa122a1d23d4a90f140b116ef9e58e2f66a3bffb9dd0d39`. **06g's census**
  `levin-phase-floor-06g-census-20261101T114315`, checkpoint
  `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, is read for the descriptive
  tables. Both are hashed before and after.
- **K0's 100 units** are `docs/levin-phase-floor/06l/out/k0/units.json`; 06l's harness is
  `docs/levin-phase-floor/06l/scripts/`.

**What Part B must carry:**
- **The harness runs with no variant installed.** 06l's `variant_loader` and its `--variant` mode
  are not used at $m$; the tree's own module is the code. The call records at $m$ have no
  `route_06l` key, and prompt §5's comparison excludes it. Each harness change is a deviation, its
  diff saved.
- **The firings are not recorded at $m$.** Route B's census holds them, under `route_06l`. Gate M
  makes the firings at $m$ route B's by identity. Any firing count the descriptive tables need is
  read from route B's census.
- **The agent's test-2 families**, `noisy` and `far` in `test_levin_trust_yardstick.py`, show that
  B fires only at noise level. The rehearsals' `QuadSourceIntegral` rows are where R26 can differ
  from R25, and nowhere else.
- **The open issue `[06m-levin-floor-misses-node-rounding-far-from-the-origin]`** is not measured
  here. If the census reader can count Levin leaves with $\Delta_1/\phi_P > 10$ at no extra cost,
  it reports the count as descriptive. Otherwise it is left for 07's census.

(End of the verbatim addendum. In it, "this addendum's commit" is `38f6a828`, and its worktree is
`SecondaryGWKit-lpf06m-38f6a82`.)

### Six cores, across everything

**At most six cores of computation are in use at any time on this host,** across both agents and
the orchestrator. The work is strictly sequential, and each stage starts only when the orchestrator
tells you:
1. the rehearsal agent: R25 (Ray, `--cpus 6`), then R26 (Ray, `--cpus 6`), then its comparisons,
   the readers and the lock's release;
2. then the census agent: Gate M0, the census (six workers), then its scoring.

While another stage runs, an agent does no computation beyond reading files, copying scripts,
diffing and writing notes.

### Rules for both of you

- **Never write to** any of these:
  - A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`);
  - its scratch copy, which you read only, opened read-only; nothing in its directory is moved,
    retired or deleted;
  - any earlier prompt's records, stores or runs. That covers `var/levin-phase-floor-0*` other than
    `-06m`, `var/datastores/levin-phase-floor-0*` other than `-06m-r25*`/`-06m-r26*`, and
    `var/runs/` other than your own runs. It includes R23 and R24, route B's census, 06g's census
    and 06g's, 06k's and 06l's records, which you only read;
  - `var/levin-phase-floor-06m/review/`, the orchestrator's.

  Never end another session's run. **Never touch anything of prompt 07's**
  (`var/levin-phase-floor-07*`, including `var/levin-phase-floor-07-leaves/`; any stash).
- **Ray.** Only R25's and R26's own drivers, and the readers that §4 runs under the lock, may start
  a Ray session.
  - Production modules import the `ray` package, and that is fine; so is `ray.cloudpickle` as an
    unpickler. What must never happen elsewhere is Ray being **initialised**.
  - Every Ray driver names its address. `ray stop` is never run.
- **Save every command you run and its output** under your `out/`, or under `state/` for launch
  transcripts and lock transcripts. Keep a `out/commands.log` of every command, in order, as 06g's
  rehearsal agent did. The review reads the outputs, not your log.
- **Launch, one check, end the turn** (`CLAUDE.md`, run registry rule 4).
  - Launch detached: `nohup sh -c …`, manifest first, stdout and stderr into the run directory or
    your records, pid recorded.
  - Verify in **one** check that it started.
  - Then end your turn and report the run id, the run directory, the detached Python pid, the `sh`
    pid and the first checkpoint or heartbeat line.
  - **Do not poll, do not `sleep`-wait, never `pgrep -f`.** The orchestrator waits on the pid and
    resumes you with the final `status.json` (or the exit).
- **Planned stops.** Your brief names them. At each one, end your turn with the report it asks
  for, and do nothing further until the orchestrator resumes you.
- **Number and classify every deviation** from the prompt before you report: `STRUCTURALLY
  REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`. Failed attempts are deviations too.
- **Stop and ask** at any of prompt 06m §7's conditions that apply to your section. Also stop at
  anything this brief or the prompt does not settle that changes what you measure.
  - To stop, end your turn with the question stated plainly **as its first line**, then the state
    of every file and run you have made.
  - Do not guess and do not repair.
  - Where the prompt's literal wording and the correct measurement part, do not let the wording
    block the right answer: do what is correct, and record it as a deviation with both readings. If
    the two readings would give different verdicts, stop and ask.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **Formatting:** your copied scripts are records, left as run. Do not run `black` over files
  anywhere outside `var/levin-phase-floor-06m/`.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-06m/`:
  - `log.md`: your section, in campaign README §5.1's form, with **every heading of it**, **What
    shipped** included. It is for appending to `logs/06m-signed-yardstick-in-the-driver.md`;
  - `verification.md`: your part of `docs/levin-phase-floor/VERIFICATION.md` §6m, with links in the
    `06m/…` form. The laptop commits your directory as `docs/levin-phase-floor/06m/<your
    directory>/`;
  - `issues.md`: anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`, with a provenance header (script SHA-256, tree, inputs' SHA-256)
    on every output you write;
  - `KEPT_ON_MAC_STUDIO.txt`: every pickle, every run directory and every file over 2 MB you made,
    with SHA-256 and size. The laptop does not copy them.
