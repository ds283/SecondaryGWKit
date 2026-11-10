## Common to the two Part B agents of prompt 07 (levin-phase-floor campaign), on the Mac Studio

You are one of two fresh-context agents running Part B of prompt 07 of the `levin-phase-floor`
campaign of SecondaryGWKit, on the Mac Studio (`mps026298.phys.susx.ac.uk`). An orchestrator
session dispatched you. It holds you at planned stops and reviews your records afterwards. Today is
2026-11-08 (UTC).

**Your prompt** is `prompts/levin-phase-floor/07-stationary-point-guard.md`, read in the code
worktree (below). Read it in full: §0–§4 explain the code you are measuring, and §9 holds the
amendments the user made at §2's stop. Also read the code worktree's `CLAUDE.md`, which binds you:
the run registry, the shared rehearsal lock, the venv and `black`. Read the campaign README
`prompts/levin-phase-floor/README.md` §4.1 and §5 (§5.1 is the log's form). Your scope is the
section named in your own brief. The other section is the other agent's.

Read the code log `prompts/levin-phase-floor/logs/07-stationary-point-guard.md` as context, all of
it. Its **Predictions and verdicts** holds §5's S0–S2 and §6's V0–V5, copied there before any
number: they are what you score.

**Do not read** anything under `prompts/levin-phase-floor/orchestrator/`,
`docs/levin-phase-floor/*/review/`, any `docs/levin-phase-floor/*/review.md`, or the main checkout's
`var/levin-phase-floor-07/review/` beyond these brief files. The facts you need from the
orchestrator's note are quoted below, verbatim.

### Where things are

- **The code commit** is `ad732887` (07 Part A: location, windows, the nodal guard and §1.3's
  per-window `iroff1` rule in `_adaptive_levin`; the four counts in `QuadSourceIntegral`'s Levin
  records). Its parent is **`91472dbee1daba8374b0018db1195cf6c7898bbb`** (`91472dbe`).
- **Part B's dispatch SHA** is **`746003508bff868fc1b256b29cbd5035609bed2b`** (`74600350`). Between
  `ad732887` and it only `prompts/` changes. This is the tree $g$.
- **Module hashes, as the orchestrator measured them:**
  - at `91472dbe`: `AdaptiveLevin/levin_quadrature.py` `beaf0e23…` (06m's), `QuadSourceIntegral.py`
    `962424c0…`;
  - at `74600350`: `levin_quadrature.py` `dc05596c…`, `QuadSourceIntegral.py` `33cb161e…`;
  - the launcher `RunDrivers/quadsource_atol_sweep.py` is `b0135218…` in both, as in 06m's R25 and
    R26.
- `git diff --stat 91472dbe 74600350 -- . ':!docs' ':!prompts'` touches
  `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_stationary.py` (new),
  `test_levin_trust_yardstick.py`, `ComputeTargets/QuadSourceIntegral.py` (the four counts, 14
  lines), `ComputeTargets/tests/test_quadsource_levin_guard_counts.py` (new) and `CLAUDE.md`, and
  nothing else.
- **The main checkout,** `/Users/ds283/Documents/Code/SecondaryGWKit`, at `handover-remedial`
  `28ebaf07`, clean.
  - It holds the venv, the stores, `var/` and the run registry.
  - The records go in its `var/levin-phase-floor-07/`.
  - **No tree runs from the main checkout, and it is never moved.**
- **The parent worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db`, detached at
  `91472dbe`. R27, S2's capture and the census's K0 run (Gate V0) run here.
- **The code worktree,** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035`, detached at
  `74600350`. Everything else runs here: R28, the value comparisons and the readers; the census $g$
  and V4's references. The prompts and records are read here.
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
  `git --no-optional-locks`. You commit nothing; the laptop copies `var/levin-phase-floor-07/`
  back and commits it.
- **`HEAD` is not yours.** Check each tree's `HEAD` and `git status --short` wherever you rely on
  them.

### Facts, re-checked by the orchestrator on the Mac Studio at 23:59–00:01 UTC, read-only

Check what you rely on yourself; no fact here is to be trusted over the code or the store.
- **The trees.** The main checkout is `28ebaf07`, `git status --short` empty. Both 07 worktrees are
  clean (`git status --short --ignored` empty) at `91472dbe` and `74600350`. `git worktree list`
  shows the main checkout and these two.
- **The registry** holds 62 runs, all finished; nothing is `running` or stale.
- **The lock** is free (`~/.raypoolkit/rehearsal-lock` does not exist), and `ray-processes
  --ray-temp-dir /tmp/ray-sgk` lists none. `status` takes no `--project`.
- **No earlier 07 Part B state.** There is no `var/datastores/levin-phase-floor-07-*`, and no
  `var/levin-phase-floor-07/` but the orchestrator's `review/`. `var/levin-phase-floor-07-leaves/`
  exists: it is 06e's leaf study, not Part B's. **You neither read it nor write to it.**
- **The protected inputs,** hashed by the orchestrator before your dispatch (181 files): A3 v2's six
  files, its scratch copy's five (matching `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`),
  R23–R26's twenty store files (R26's primary `e90e26c2…`), and 06m's census run directory's 150
  files (checkpoint `ea5bc3ab…`).
- **The host.** 10 cores, 135 GiB free. `~/.ray/auth_token` is mode 600. **The load was high**
  (17.8 at 23:59) from system services (audiomxd, Spotlight's `mds_stores` and `mdworker`, configd),
  not from any compute job. Bear it in mind when you time things; it is not by itself a stop.
- **"Value-identical" means the computed values.** It excludes timings, labels, `parents_digest`,
  `rho_reuse` and work counters (`[05-store-value-diff-compares-volatile-columns]`). For R27 against
  R28 it also excludes the four new counts, which only R28 has.
- **05's `classify_non_qsi.py` needs both runs to share a job name** (06m's rehearsal, deviation
  1). S0 compares R26 (job `levin-phase-floor-06m`) with R27 (`levin-phase-floor-07`). The labels
  are normalised in a copy of the JSON, as 06m did, and the classifier is not changed.

### The note's §B.1, verbatim

(Paths in it are the main checkout's.)

```
### B.1 Facts, to be re-checked at Part B's dispatch

- **The scratch copy's certification:** `var/levin-phase-floor-06/wider/state/copy_files_certified.txt`.
  06e's after-hashes are `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`. Its
  fingerprint, and A3 v2's, is `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.
- **06m's census:** `levin-phase-floor-06m-census-20261106T102153`, 150 files, 4.41 GB, checkpoint
  `ea5bc3ab6b618d562b9dee0a7ed4bdba5c1999e8348b2ad33bdadb6135e63c2e`. Its file list and hashes are in
  `docs/levin-phase-floor/06m/census/KEPT_ON_MAC_STUDIO.txt`. Its harness is
  `docs/levin-phase-floor/06m/census/scripts/`, byte-identical to the run.
- **R26:** `var/datastores/levin-phase-floor-06m-r26*.sqlite`, five files, primary `e90e26c2…`, the
  rest in `docs/levin-phase-floor/06m/rehearsal/KEPT_ON_MAC_STUDIO.txt`.
- **The fingerprint** R27 and R28 must both give is
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. R23–R26 gave it.
- **06m's rehearsal record:** its command line, launcher (`b0135218…`), readers script
  (`run_readers_worktree.sh`, `ec96cf09…`), 05's `classify_non_qsi.py` (`26690e61…`) and the
  tabulation are in `docs/levin-phase-floor/06m/rehearsal/`.
- **06g's captures** on the Mac Studio: `var/levin-phase-floor-06g/references/state/capture/`
  (19 MB). Its V1 capture scripts are under `docs/levin-phase-floor/06g/references/v1/`.
```

### The note's §B.5, the addendum at Part B's dispatch, verbatim

In it, "this addendum's commit" is `74600350`, and its worktree is `SecondaryGWKit-lpf07-7460035`.
The worktrees it mentions have been made.

```
### B.5 Addendum: Part B's dispatch

*Written 2026-11-07, at Part A's landing.*

**Part A** is the code commit **`ad732887`**, whose parent is **`91472dbe`**. It was reviewed in
§A.3's eleven checks, all passing, and pushed on the user's word: *"Yes, please land it. Push. Then,
write, commit, and push the dispatch addendum for Part B."* Two orchestrator commits came before it,
both pushed:
- `945c8668` opened `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`;
- `91472dbe` amended the prompt at §2's stop (its §9).

**The user's rulings at §2's stop** are prompt §9 points 1–4, with the user's words, and the board's
Decisions row **07 §2 stop**:
- §2.2 ran at `rtol` $10^{-9}$, and its rules gave **$\Phi_w = 2\pi$, $\Lambda_c = 4$**;
- §1.3 is QUADPACK's `iroff1` counted per window: limit 6, value tolerance $10^{-5}$, error ratio 0.99.
  A stopped window's leaves are returned floor-limited;
- a key-set assertion admits §1.5's six new result keys;
- §2 was re-run once, on the code with §1.3 built.

**The trees, by SHA:**
- **R27 and V0's K0 run** at **`91472dbe`**, in the worktree `SecondaryGWKit-lpf07-91472db`. Its
  production code is the dispatch SHA `bd21bf23`'s, which is 06m's: `git diff --stat bd21bf23
  91472dbe -- . ':!docs' ':!prompts'` is empty. Against R26's tree `38f6a828` it differs only in
  `CLAUDE.md`. So S0 (R27 against R26) is a reproducibility check, as written.
- **R28, the census $g$, V4's references, the comparisons and the readers** at **this addendum's
  commit**, in the worktree `SecondaryGWKit-lpf07-<this commit's short SHA>`. Its code is
  `ad732887`'s. `git diff --stat 91472dbe ad732887 -- . ':!docs' ':!prompts'` touches only:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_stationary.py` (new) and `test_levin_trust_yardstick.py`;
  - `ComputeTargets/QuadSourceIntegral.py`, the four counts;
  - `ComputeTargets/tests/test_quadsource_levin_guard_counts.py` (new);
  - `CLAUDE.md`.

**What Part A's review found that Part B must carry:**
- **Every number in the review was re-run on the laptop:**
  - the seven roots, at 103, 781, 39, 156, 280, 127 and 64, all OK;
  - §2.2's 124 units, bit-identical on the committed module;
  - z12 and w03 replayed bit for bit. z12 is honest at 0.718 of its error, against 1,681× at the
    parent.
- **Windows stop while some of their leaves are still falling** (the agent's O8). This happens on 8
  of §2.3's 14 calls at $2\pi$, and all 8 are honest and converged. **The census reports it, as
  descriptive:** for each call that locates a point, every window `stationary_windows` records as
  stopped, with `leaves_falling_at_stop`, by rtol, and whether the call converged.
- **Location costs time** (the agent's O5). A rate that does not vectorise is evaluated 4,097 times
  per call. The ComputeTargets root took 442 s here against 291 s at dispatch. The census's 300 s
  per-unit budget may be reached more often than in 06m's, which had none over. Over-budget units
  are counted under V5 as the prompt says, and **reported with their wall time against 06m's**. That
  is a finding, not a stop.
- **The bar on window leaves protects no measured value.** Under breakage (iii) test 5's value stays
  honest (0.0057 of its error). Window leaves carry the Clenshaw–Curtis floor, which has no
  $1/\theta'$ in it. Nothing for Part B to do. If V2 finds a returned leaf holding a sign change,
  check whether it is guarded or a window leaf before scoring.
- **The value tables differ from R27 only inside `QuadSourceIntegral`,** and there only in rows
  whose group or analytic calls located a point or guarded a leaf, plus the four new counts. §0.1
  already excludes those counts from "value-identical".
- **Not measured here:** `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`. Production spans
  are non-dyadic, so the parent's leaves carry that floor too. The census does not score it.

**§B.0 and §B.1, re-checked read-only on the Mac Studio, 22:47–22:50 UTC, 2026-11-07:**
- **The main checkout** is `handover-remedial` at `28ebaf07`, with `git status --short` empty.
  `git worktree list` shows the main checkout alone; 06g's and 06m's worktrees were removed
  (`a8c7854c`).
- **The registry** holds 62 runs, all finished, nothing `running` or stale.
- **The lock and Ray:**
  - `rehearsal_lock status`: free (`~/.raypoolkit/rehearsal-lock` does not exist);
  - `ray-processes --ray-temp-dir /tmp/ray-sgk`: none. `/tmp/ray-sgk` holds only session
    directories from 2026-10-25.
- **Load and disk:** the load is 2.75 / 3.04 / 3.17, and 136 GiB is free (85 % used).
- **No 07 paths yet:** there is no `var/levin-phase-floor-07/` and no
  `var/datastores/levin-phase-floor-07-*`. **But `var/levin-phase-floor-07-leaves/` exists** (28
  files, 8 MB, last written 2026-10-29). It is the working directory of 06e's leaf study of 07's
  fourth stop, whose records are committed under `docs/levin-phase-floor/06e/investigations/07-leaves/`.
  It is not Part B's, and Part B's paths (`var/levin-phase-floor-07/…`) do not collide with it.
  **No agent writes or reads it.** Whether it is kept is asked at 07's records commit.
- **The inputs, by SHA-256 against their recorded lists,** 160 files, all OK:
  - the scratch copy's five files, against `06e/review/checks/a3v2_copy_after.sha256`, with no
    `-wal` or `-shm`;
  - 06m's census run directory `levin-phase-floor-06m-census-20261106T102153`, all 150 files
    against `06m/census/KEPT_ON_MAC_STUDIO.txt`, checkpoint `ea5bc3ab…` among them;
  - R26's five store files against `06m/rehearsal/KEPT_ON_MAC_STUDIO.txt`, primary `e90e26c2…`.

  R25's and R26's run directories are present.
- **06g's captures:** `var/levin-phase-floor-06g/references/state/capture/` holds 102 entries,
  19 MB.
- **The venv** is Python 3.13.16, with `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
  0.2.0, `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy` 2.2.4, `scipy` 1.16.3 and `mpmath` 1.3.0. The
  laptop's are the same.

**The worktrees** are made by the laptop on the user's word to dispatch, after this commit is
pushed, detached and with no venv: `SecondaryGWKit-lpf07-91472db` and
`SecondaryGWKit-lpf07-<this commit's short SHA>`. The Mac Studio's main checkout fetches first,
which moves no working tree. §B.0's checks are repeated then. The Mac Studio orchestrator session
is then started in the code worktree, through the saved `MacStudio` connection, with this note as
its brief. Its folder must be trusted in the app first.
```

### Six cores, across everything

**At most six cores of computation are in use at any time on this host,** across both agents and
the orchestrator. The work is strictly sequential, and each stage starts only when the orchestrator
tells you:
1. the rehearsal agent: R27 (Ray, `--cpus 6`), then R28 (Ray, `--cpus 6`), then its comparisons,
   S1, S2's capture, the readers, the lock's release, then S2's references (at most six workers);
2. then the census agent: Gate V0, the census (six workers), then its scoring, then V4's
   references (at most six workers).

While another stage runs, an agent does no computation beyond reading files, copying scripts,
diffing and writing notes.

### Rules for both of you

- **Never write to** any of these:
  - A3 v2 (`var/datastores/handover-A3-v2-lambdacdm*`);
  - its scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, which you read only,
    opened read-only; nothing in its directory is moved, retired or deleted;
  - any earlier prompt's records, stores or runs. That covers `var/levin-phase-floor-0*` other than
    `var/levin-phase-floor-07/` (and there only your own directory), `var/datastores/` other than
    `levin-phase-floor-07-r27*`/`-r28*`, and `var/runs/` other than your own runs. It includes the
    R23–R26 stores and runs, 06m's census, and 06g's captures, which you only read;
  - `var/levin-phase-floor-07/review/`, the orchestrator's;
  - `var/levin-phase-floor-07-leaves/` and `var/levin-phase-floor-07/wip/` (if it exists), and any
    git stash.

  Never end another session's run.
- **Ray.** Only R27's and R28's own drivers, and what prompt §5 runs under the lock (S2's capture,
  the readers), may start a Ray session.
  - Production modules import the `ray` package, and that is fine; so is `ray.cloudpickle` as an
    unpickler. What must never happen elsewhere is Ray being **initialised**.
  - Every Ray driver names its address. `ray stop` is never run.
- **Save every command you run and its output** under your `out/`, or under `state/` for launch
  transcripts and lock transcripts. Keep an `out/commands.log` of every command, in order, as 06m's
  agents did. The review reads the outputs, not your log.
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
- **Stop and ask** at any of prompt 07 §8's conditions that apply to your section. Also stop at
  anything this brief or the prompt does not settle that changes what you measure.
  - To stop, end your turn with the question stated plainly **as its first line**, then the state
    of every file and run you have made.
  - Do not guess and do not repair.
  - Where the prompt's literal wording and the correct measurement part, do not let the wording
    block the right answer: do what is correct, and record it as a deviation with both readings. If
    the two readings would give different verdicts, stop and ask.
- **Correctness is the only objective.** An argument from cost or time for a shortcut is a deviation
  to record, not a reason to take it.
- **No crashing scripts.** Check inputs before long runs; a script that fails part-way is a
  deviation to record.
- **Floats** are compared by bit pattern (`float.hex`), never `isclose`.
- **Formatting:** your copied scripts are records, left as run. Do not run `black` over files
  anywhere outside `var/levin-phase-floor-07/`.
- **Your records** go in your directory under the main checkout's `var/levin-phase-floor-07/`:
  - `log.md`: your section, in campaign README §5.1's form, with **every heading of it**, **What
    shipped** included. It is for appending to `logs/07-stationary-point-guard.md`;
  - `verification.md`: your part of `docs/levin-phase-floor/VERIFICATION.md` §7, "The
    stationary-point guard", with links in the `07/…` form. The laptop commits your directory as
    `docs/levin-phase-floor/07/<your directory>/`;
  - `issues.md`: anything to open on the board, or "none";
  - `out/`, `scripts/` and `state/`, with a provenance header (script SHA-256, tree, `HEAD`,
    `git status --short`, inputs' SHA-256) on every output you write;
  - `KEPT_ON_MAC_STUDIO.txt`: every pickle, every run directory and every file over 2 MB you made,
    with SHA-256 and size. The laptop does not copy them.
