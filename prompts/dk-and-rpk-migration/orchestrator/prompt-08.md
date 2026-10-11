# Orchestrator — prompt 08, rehearse through both packages (R2)

Read [`../README.md`](../README.md) first: §0, §0.1, §2.2 (08's charter), §3, §4 and §4.1, §5 and
§6.2 (C8, C12, C14). Read also `CLAUDE.md`'s "The datastore layer is `datastorekit`; the work queue
is `raypoolkit`" and "Long-running jobs — the run registry", whose rules bind you as much as the
agent. Then read [`prompt-05.md`](prompt-05.md), the note for R1, whose protocol, wait and review
this note carries over except where §0 below says otherwise, and this board's *Orchestrator review
of prompt 05* and *of prompt 07*.

**You do not write code.** You may:
- run the interrupter's dummy tests in the scratchpad;
- run the one wait per launch that §2.1 describes;
- read the two new stores read-only, fingerprint them, and run the census, `PRAGMA
  integrity_check`, `foreign_key_check` and `read_inventory` on them;
- read R0's and R1's stores' files only to hash them, and their run directories;
- run RayPoolKit's measuring script from its root, with its venv;
- run the suites;
- fix small residue in a follow-up commit of your own (§4).

**While a build is running you write nothing in the repository**, outside `var/` and the session
scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop.

**The prompt:** [`08-rehearse-through-both-packages.md`](../08-rehearse-through-both-packages.md)
**Board items:** L1–L4 (R2), G-RPK · **Gate it decides:** RayPoolKit's G2 · **Closes:** nothing ·
**Narrows:** nothing · **Opens:** anything found · **Model:** Opus, as the prompt recommends.

**Gate:**
- 07 landed (`643c3b2`) and was reviewed (`af26914`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive.

## 0. What makes this prompt unusual

**It is R1 again, with the work queue the package's.** R1 ran at `6861a9f`, with `datastorekit`
0.2.2 and SGK's frozen `RayTools/`. 07 swapped `RayTools/` for `raypoolkit` 0.2.0 and changed
nothing else that runs. So **if R2 differs from R1, the work queue is the only cause**. A
fingerprint miss is a RayPoolKit defect (README §5 rule 11, C14): the agent records it and stops,
the user reports it, and the campaign waits for a release and a new pin prompt.

**It runs for half an hour or more, and no conversation may own it.** As at 02 and 05: the agent
launches each build detached, through the registry, checks once, and ends its turn. You run
**one** wait per launch (§2.1) and resume it when the run is no longer alive. Four launches, four
waits.

**It measures one behavioural change, C8.** At `main.py:3247` the `APPLY GKSOURCE POLICIES`
queue's `post_handler`, `postprocess_GkSourcePolicy` (`main.py:2871-2893`), now receives the
object the persist returned, not the driver's local copy. The statistics block it fills
(`main.py:3265-3278`) is the measurement.

**It decides RayPoolKit's G2.** Five parts (prompt §2): `RayTools/` gone, 0 imports and 54 sites
constructing, the pin, the six roots, and R2's fingerprint. The board records it as holding only
when all five do, each naming its commit or run.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `af26914`
and against both installed packages. Each correction below governs, and the agent logs it as an
`IMPLEMENTATION CHOICE` citing this note, or `STRUCTURALLY REQUIRED` where the prompt cannot be
followed as written.

1. **The gate's "no commit after 07's touches code"** (prompt, Gate). `af26914`, 07's review,
   changed one comment in `RunRegistry/tests/test_pipeline_adoption.py` (`:33-37`). Its `ast`
   equals `643c3b2`'s, and nothing a build runs is touched. So the code that runs at the dispatch
   `HEAD` is `643c3b2`'s; the gate holds in substance. The log says so.
2. **The printed lines R1 does not have** (prompt §2, L4, "The printed lines are an interface").
   **R1 printed no progress line at all.** In each of its four `stdout.log`s, `work items
   remaining` and `inflight:` count **0**: every stage finished, or was killed, inside the queues'
   300 s minimum notify interval (`main.py:119`; R1's longest, QSI, took 289.7 s). So:
   - **The completion lines** are compared with R1's same slug: the count of
     `-- ALL WORK ITEMS COMPLETE in time …` (10, 7, 9 and 10 in `reference`, `resumed-a`,
     `resumed-b`, `resumed-c`), each followed by `Queue summary: …`, and the
     `(no compute items were evaluated - all assets read from datastore)` lines, in form and number,
     times and rates aside.
   - **A progress line, if R2 prints one** (its reference QSI stage may pass 300 s), is compared in
     form with the last one the copy printed: `var/runs/a3-v2-readiness-07-reference-20261004T012909/stdout.log:100-101`,
     ```
        -- 2026-10-04 01:40:11 (5m 10.2s running): 0/2 work items remaining = 100.00% complete (may be waiting for compute/store/validate tasks to finish)
           inflight: 0 lookup, 261 compute, 70 store | completed: 864 lookup (avg 167.1/min), 603 compute (avg 116.7/min), 533 store (avg 103.1/min)
     ```
     and with the `status.json` `stage` the registry read from it. A progress line in R2 where R1
     had none is the 300 s threshold and the load, not a finding.
   - **The registry's fields** are as the prompt says. R1's `status.json` `stage`: `reference` and
     `resumed-c` end `'CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete'`; `resumed-a`
     was killed at `'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'` and `resumed-b` at
     `'CALCULATE QUADRATIC SOURCE INTEGRALS'`. `progress` is `null` in all four.
3. **`prune_serials.py` is not run** (inherited through 05 §1.3; README §4.1 L2). It hardcodes
   `a3-v2-readiness` 07's serials. R0 and R1 ran `prune_check.py` in its place, on their own
   `closed_store_state.py` output, and R2 does the same (05's note, correction 1).
4. **The saved measurement waits for the last launch** (prompt §2, G-RPK part 2, and §6).
   `measure/out/08-raypoolkit-measure_client_pools.md` is in the repository, so writing it before
   a launch would make that manifest `git_dirty: true`. The agent runs the script whenever it
   likes, redirecting into `var/dk-and-rpk-migration-08/`, and copies the output into `measure/out/`
   only after the last launch has ended. Run at `HEAD`, it must equal the orchestrator's run below
   but for the commit, the file count and the non-`.py` lines naming `RayWorkPool`.
5. **Checklist item 5 names files 04 deleted** (prompt "Read first" 6). It traces the persisted
   object through SGK's old `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py` and
   `Datastore/object.py`. Trace C8 through the installed `datastorekit` instead, as 07 did (07's
   log, item 5's table; §0 "C8, by reading" below).

**What moves on purpose:**
- Nothing tracked is modified but the board and the verification document, and
  `docs/OPEN_ISSUES.md` if an issue is opened.
- Added: `docs/dk-and-rpk-migration-verification.md` gains §3 (§1 and §2 are not edited; the
  header already names §3); saved outputs under `docs/dk-and-rpk-migration/rehearsal/` named with
  `08`; `measure/out/08-raypoolkit-measure_client_pools.md`; the log.
- No script under `docs/dk-and-rpk-migration/rehearsal/` changes. A new one of the agent's, if any,
  goes beside them and is named in the log.
- **README does not move.** 08 ships no interface.
- Under `var/`: the two stores and their sidecars, four run directories and their launch logs, and
  `var/dk-and-rpk-migration-08/`.
- The index is 108, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-11 at `af26914`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `git diff --stat 643c3b2 af26914 -- . ':!prompts'` is the
  one test comment of correction 1. `RAY_ADDRESS` unset. `pgrep -lf 'gcs_server|raylet|ray::'`
  and the whole-table search find **nothing**. 10 CPUs. Load 10–12 at 22:29, rising to 39 by
  22:35 from processes not of this repository. 72 GB free on `var/`'s volume (93 % used).
- **The registry.** 25 runs: 23 finished, 2 `unknown` (the pre-registry `a3-pilot` pair). None
  `running`. R1's four are `done`, `killed`, `killed`, `done`.
- **The venv, and what the manifests should hold.** `pip show` gives `datastorekit` 0.2.2 and
  `raypoolkit` 0.2.0, both in this venv's `site-packages`; `pip check` is clean.
  `RunRegistry.package_provenance()` gives, with keys sorted:
  `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false,
  "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version":
  "0.2.2"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable":
  false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version":
  "0.2.0"}}`, which is the prompt's §2 record. Compare as JSON, not as text: the manifest's key
  order is the record's, not sorted.
- **What runs is the package.** `RayTools/` does not exist (no `__pycache__` either). In a fresh
  `python -I` with the repository on its path, `import RayTools` and `import RayTools.RayWorkPool`
  raise `ModuleNotFoundError`; `raypoolkit.__file__` is
  `venv/lib/python3.12/site-packages/raypoolkit/__init__.py`; `RayWorkPool.__module__` is
  `raypoolkit.RayWorkPool`; `ray.is_initialized()` is false.
- **The launcher is R1's.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` is
  **`b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`**, R1's; its last change is
  `0c9e2cf`, and it names no pool. Every R2 manifest's `script_sha256` should be this.
- **The build still applies to `main.py`.** `substitute_main_py(filter_cases=False)`, loaded by
  path, replaces each of its four literals exactly once and the result compiles, with Ray down. So
  the four `replaced 1 occurrence` lines recur.
- **The names are free.** `store show` on `var/datastores/dk-and-rpk-migration-08-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either. Nothing under
  `var/datastores/` or `var/runs/` contains `dk-and-rpk-migration-08`; `var/dk-and-rpk-migration-08/`
  does not exist; no `-journal`, `-wal` or `-shm` file under `var/datastores/`.
- **R0's and R1's stores are as 02 and 05 left them.** The 24 files, SHA-256 prefix and mtime. R0's
  twelve equal 05's table (verification §2.8). The agent records the full digests before its first
  launch and after its last step, and they must equal these.

  | File | SHA-256 (16) | mtime |
  |---|---|---|
  | `-02-reference.sqlite` | `9b7a0acc8bcca221` | 14:30:17 |
  | `-02-reference-shard0000.sqlite` | `3e469052e8810e1e` | 14:38:00 |
  | `-02-reference-shard0001.sqlite` | `210f494aedfd154f` | 14:38:00 |
  | `-02-reference-shard0002.sqlite` | `37b12c1592c2f03e` | 14:30:17 |
  | `-02-reference-shard0003.sqlite` | `db35d513a3d6278f` | 14:38:00 |
  | `-02-reference.manifest.json` | `755da82a40a26c4b` | 14:38:55 |
  | `-02-resumed.sqlite` | `bf22863f9ea8b6cf` | 14:45:54 |
  | `-02-resumed-shard0000.sqlite` | `e1340524ecc5bb94` | 14:48:46 |
  | `-02-resumed-shard0001.sqlite` | `35908f3a09e66838` | 14:48:46 |
  | `-02-resumed-shard0002.sqlite` | `fafbe696ba83ac3b` | 14:39:24 |
  | `-02-resumed-shard0003.sqlite` | `09b8aef1061ef5ed` | 14:48:41 |
  | `-02-resumed.manifest.json` | `c93c629616fa92a1` | 14:48:49 |
  | `-05-reference.sqlite` | `6f01ba6041cb3b2f` | 19:28:30 |
  | `-05-reference-shard0000.sqlite` | `1ae3eec6bc377995` | 19:40:08 |
  | `-05-reference-shard0001.sqlite` | `6151d1d673c3e170` | 19:40:08 |
  | `-05-reference-shard0002.sqlite` | `09689a714bba7ddb` | 19:28:29 |
  | `-05-reference-shard0003.sqlite` | `cb2eeb797cbb4e67` | 19:40:08 |
  | `-05-reference.manifest.json` | `bbde995d5c0d0a18` | 19:41:08 |
  | `-05-resumed.sqlite` | `19cd8373ae0a45d6` | 19:50:49 |
  | `-05-resumed-shard0000.sqlite` | `47ec005237b14228` | 19:54:54 |
  | `-05-resumed-shard0001.sqlite` | `430dba4def831f9f` | 19:54:54 |
  | `-05-resumed-shard0002.sqlite` | `b12a4bf7df09a2e4` | 19:42:14 |
  | `-05-resumed-shard0003.sqlite` | `47338a1922fe06ad` | 19:54:54 |
  | `-05-resumed.manifest.json` | `30252d1d9f2a3c98` | 19:54:57 |

- **The recorded fingerprints to compare against**, as run directories for
  `compare_with_recorded.py --against`:
  - R1: `var/runs/dk-and-rpk-migration-05-reference-20261011T192802`;
  - R0: `var/runs/dk-and-rpk-migration-02-reference-20261011T143003`;
  - `a3-v2-readiness` 07: `var/runs/a3-v2-readiness-07-reference-20261004T012909`;
  - `datastore-generic` 09: `var/runs/datastore-generic-09-rehearsal-20261006T152452`.
- **The interrupter passes at `af26914`.** Both dummy tests, run into the scratchpad at
  22:31:13–22:31:39, exit 0 with every verdict as R1's record gives it (five cases; three
  `--build` cases), and no child left alive. The agent runs them again itself.
- **What the package changes at run time, by reading it.** The orchestrator diffed SGK's copy
  (`RayTools/RayWorkPool.py` at `789b72d`, 575 lines) against the installed
  `raypoolkit/RayWorkPool.py` (615 lines). Apart from the licence header and the two imports:
  - **Every `print` string is unchanged.** `format_time` now comes from `raypoolkit._timing`, whose
    `ast` equals `utilities.format_time`'s. So the banner (`:206`), progress (`:585-592`) and
    completion (`:606`, `:609`, `:613`) lines should be byte-for-byte R1's form.
  - **C8**: the store branch's no-validation exit calls `self._post_handler(replacement_obj)`
    (`:529-530`) where the copy passed `obj`. Every other call of the post handler — the lookup
    exit (`:402`), the blob exit (`:439`), the `available` branch (`:458`) and the validate branch
    (`:554`) — passes `obj`, as the copy did.
  - **The constructor** raises `ValueError` on inconsistent handlers or a batch size below 1
    (`:109-142`), where the copy raised `RuntimeError`/`RuntimeWarning` on fewer cases. 07 measured
    all 54 sites `constructs`; a `ValueError` at run time is the prompt's stop.
  - **The in-flight dictionaries are keyed by `ref.hex()`**, a string, where the copy keyed them by
    the bound method `ref.hex`, so by the `ObjectRef`'s identity. Expect nothing visible.
  - `results` is sized by `_num_total_items` (`len(self._todo)`), not `len(task_list)`; equal for
    every SGK call, which passes a list.
- **C8, by reading.** `GkSourcePolicy_queue` (`main.py:3247-3262`) has `compute_handler`,
  `validation_handler=None` and `post_handler=postprocess_GkSourcePolicy`. So:
  - **Where a policy row is computed** (R2's `reference`, and `resumed-b`, which recomputes all 291
    at R1), the handler gets the persisted object. That is the pickled copy the shard actor's
    `object_store` returns (`datastorekit/SQL/ShardedPool.py:3441`, `:3531`;
    `datastorekit/SQL/Datastore.py:661`, `:733`), after the factory's `store` sets `_my_id` and
    nothing else (`Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:150-174`). The handler reads
    `obj.policy.label` (or `store_id` if the label is `None`), `obj.type` and `obj.quality`
    (`ComputeTargets/GkSourcePolicyData.py:585-600`), which the copy carries as the local object
    had them. **So every count should be R1's.**
  - **Where every row is read back** (`resumed-c`), the lookup path is taken and the handler gets
    `obj` as before. `resumed-c`'s block does not exercise C8; say so.
- **What R1's logs held**, against which §2's log comparison is made (verification §2.2.5): each
  killed `stderr.log` the handler's two lines (both `SystemExit(1)`); `reference`'s one SciPy
  `IntegrationWarning`; `resumed-b`'s the same warning; `resumed-c`'s empty. As before, the warning
  falls in whichever launch computes its integral, so the point can move it between `resumed-b`
  and `resumed-c`; that is not a finding. Ray's `[repeated Nx across cluster]` folding varies run
  to run. `SystemExit(1)` against `SystemExit(143)` depends on where the signal lands.
- **The stage times, and the points.** R1's reference (verification §2.5.5), at load 14–78:
  `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` **55.1 s**; `CALCULATE QUADRATIC SOURCE
  INTEGRALS` **289.7 s**; the build 12 m 13 s. R0's, at 7–35: 32.2 s, 212.3 s, 8 m 0 s.
  - **The points come from R2's own reference `stdout.log`**, as the prompt says.
  - `resumed-a`: from the `GkSource` banner, `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$'`,
    with `--interval 5 --hold <S>`, `<S>` putting the signal 10–15 s into the stage and well before
    its end at R0's speed.
  - `resumed-b`: from the QSI banner with `--hold` about a third of R2's reference stage, under
    300 s so no progress line can come during the hold, and **the pattern not anchored at its end**
    (`'^CALCULATE QUADRATIC SOURCE INTEGRALS'`), as R0 and R1 did.
  - **If an interruption misses**, so that a launch meant to be killed ends `done`, **stop and
    ask**. Do not relaunch on the store, and do not build a third store.
- **The commands are R1's**, verbatim in verification §2.1, with `-08-` and `--prompt 08`.
- **No Ray process left: list it properly** (`[07-a-killed-builds-ray-agents-outlive-the-driver]`).
  `pgrep -lf 'gcs_server|raylet|ray::'` and the whole-table search, `ps -axo pid,ppid,lstart,command`
  for lines holding both `SecondaryGWKit/venv` and `ray`. Wait for any listed agent **by pid**,
  never by pattern, and record how long it took, beside R1's (none outlived either killed driver).
- **The measurement, at `af26914`.** RayPoolKit's `measure_client_pools.py sgk --commit af26914`,
  from RayPoolKit's root (clean, `c9b0e11`) with its venv: `RayTools/` absent; **0 imports**; **0
  `import_module` calls**; 2 string literals; **54 calls, `constructs` 54**; the handler table
  as 07's. It equals `measure/out/07-raypoolkit-measure_client_pools.md` but for the commit, 1473
  tracked files (1471) and 638 non-`.py` lines naming `RayTools` or `RayWorkPool` (475), all in
  `prompts/`. `ray.is_initialized()` false after.
- **DatastoreKit's checkout is mid-release** in another session (`389e13b`, `pyproject.toml` and
  records modified, towards `0.2.3`). SGK pins the `v0.2.2` tag and the venv holds it; the agent
  reads nothing there and changes nothing in either package (README §5 rule 12).
- **Shell traps.**
  - `stat` here is GNU coreutils: use `/usr/bin/stat -f '%m %N'` or Python's `os.stat`.
  - `grep` in the agent's shell may wrap `ugrep`: use `git grep` or Python for any regex the log
    relies on.
  - The shell is `zsh`: a bare word beginning `=` (`echo =====`) is command expansion and fails.
    Quote it.
- **Expected counts**, at `af26914` (§1 step 5): Datastore **320** (54 + 103 + 101 + 62),
  RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
  (1 skipped). No count moves at 08.
- **The index:** 108.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `done`/0 and `killed`/143, each with a fingerprint, each
  driver's exit status in its launch log, and no Ray process of this repository left.
- **(R2) R2 is R1.** Both stores equal each other, `39809dca…`, and the fingerprints recorded by
  R1's reference, R0's, 07's and 09's, class by class. Nothing stored was recomputed; the prune
  removed exactly the unvalidated rows.
- **(R3) The stores are sound**, by `store_soundness.py` and the census.
- **(R4) Every reader works, and none touches the store.**
- **(R5) The manifests say both packages ran.** Every `packages` is the prompt's §2 record, every
  `git_dirty` `false`, every `git_head` the dispatch `HEAD`, and every `script_sha256`
  `b7935877…`.
- **(R6) L4 equals R1's**, by the identities of 05's §2, and every log difference from R1 is
  listed and explained.
- **(R7) C8 is measured.** The statistics counts of `reference` and `resumed-b` against R1's, with
  the reason they are (or are not) equal; `resumed-c`'s block named as not exercising it.
- **(R8) The printed lines are the interface.** The completion lines in R1's form and number, any
  progress line in the copy's form, and every `status.json` `stage` R1 had.
- **(R9) G-RPK holds**, part by part, each part resting on a named commit or run.
- **(R10) R0 and R1 were not touched.** Their 24 files keep their hashes and mtimes.

**Conventions.** 05's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §7's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended**. Datastore in the four groups. Under load, the four tests of README §7 are
  known to fail: re-run a failure alone before calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/dk-and-rpk-migration-08/` (`CLAUDE.md` rule 5). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched.
- **The venv.** The agent installs and uninstalls nothing.
- **DatastoreKit and RayPoolKit.** Read only. RayPoolKit's script is run from its root with its
  venv; nothing there is changed, fetched or committed (README §5 rule 12).
- **Never touch a process you did not start.**
- **Stop conditions are real.** A stop leaves the stores, the run directories and the tree as they
  are: no relaunch, no deletion, no fix. It goes to the user through the orchestrator, and the
  agent resumes with `SendMessage`.
- **Usage limits.** A usage stop does not stop a detached build. Resume the agent with
  `SendMessage`, saying exactly where it stopped. If it commits but sends no report, review from the
  commit, its log and its outputs.

## 1. Before you dispatch

Done on 2026-10-11 at `af26914`, before this note was committed. Repeat steps 1–4 at the dispatch
`HEAD` if anything lands in between.

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 643c3b2..HEAD` must touch no code but correction 1's comment.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. Report
   any stale or unknown run, without acting on it.
3. **Ray and the machine.** `pgrep -lf 'gcs_server|raylet|ray::'`, and the whole-table search of
   §0. No process from this repository's `venv`. `RAY_ADDRESS` unset. Record the load, and say so
   if `var/` has less than 10 GB free.
4. **The venv.** `package_provenance()` gives §0's record; `pip check` is clean.
5. **Baselines**, at `af26914`, serially, in the foreground:

   | Root | Tests | Result | Load |
   |---|---|---|---|
   | AdaptiveLevin | 32 | OK | 39 |
   | CosmologyModels | 39 | OK | 39 |
   | LiouvilleGreen | 148 | OK (1 skipped) | 39 |
   | RunRegistry | 286 | OK | 31 |
   | ComputeTargets | 693 | 1 failure: `test_wall_time_per_object` (0.108 s against 0.06 s); alone, failed six times (0.106–0.120 s) | 26; 12–16 alone |
   | Datastore | 320 = 54 + 103 + 101 + 62 | OK | 15–21 |

   Every count is 07's. The wall-clock test is README §7's first; no file it reaches has changed
   since `643c3b2`. The machine was shared throughout: another session ran DatastoreKit's own
   suite, and JetBrains Toolbox held a core. It may pass at the agent's suites, or not; under this
   load it is the known flake, and its failure alone is not a stop.

6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 108.
7. **The names are free**, and **R0's and R1's 24 files** hash as §0's table.
8. **The interrupter.** Both dummy tests at `HEAD`, into the scratchpad. Both pass.
9. **The measurement.** RayPoolKit's script at `HEAD`, as §0.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, and through it 05's;
- the campaign README and this board;
- `CLAUDE.md`'s two sections named above;
- 05's and 07's logs, and verification §2;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, every check and the suites. Its log is
  `logs/08-rehearse-through-both-packages.md`, in README §5.1's form, with the prompt's §6
  additions.
- **The hand-off.**
  - After each launch: one check, then a report giving the run id, the Python pid, the `sh` pid,
    and the interrupter's pid and arguments if there is one, then the end of its turn.
  - You resume it with the run's final `status.json`, the last lines of its launch log, and the
    wait's verdict.
  - It never waits, polls or sleeps on a build itself. Waiting by pid for a killed build's orphaned
    Ray agents, before the next launch, is the one bounded wait it runs.
- **The files it may change** are the prompt's §6 list, nothing else. **It touches none of them, and
  creates nothing in the repository outside `var/`, until the last build launch has ended**, so
  that every manifest stamps `git_dirty: false`. That includes the saved measurement (correction 4).
- **R0's and R1's stores are not opened.** Hash them before the first launch and after the last
  step; compare R2 with them through `compare_with_recorded.py --against` their run directories.
- **The interruption points** come from R2's reference `stdout.log`, as §0 says. A missed
  interruption is a stop.
- **Stop and ask** on any of the prompt's §4 conditions. A fingerprint miss is the one R2 exists
  for: record both fingerprints, `compare_fingerprints` by class, and stop. It is a package defect
  (README §5 rule 11), and nothing is fixed or worked around here. **A statistics count that
  differs is not a stop**: it is C8's measurement, recorded and opened as an issue.

### 2.1 The one wait per launch

05's, unchanged. When the agent reports a launch, run **one** background command that ends when
that run is no longer alive, or after the cap:

```bash
PYTHONPATH=. ./venv/bin/python - <<'EOF'
import json, time
import RunRegistry
status_path = "var/runs/<run id>/status.json"
start, cap, seen_stale = time.time(), 2 * 3600, False
while True:
    status = RunRegistry.read_json(status_path) or {}
    state = RunRegistry.liveness(status, 900)
    if state == "finished" or not RunRegistry.pid_alive(status.get("pid")):
        print("ENDED", state, json.dumps(status)); break
    seen_stale = seen_stale or state == "stale"
    if time.time() - start > cap:
        print("CAP", state, json.dumps(status)); break
    time.sleep(30)
print("heartbeat went stale at least once" if seen_stale else "heartbeat never stale")
EOF
```

- It identifies the run by its pid and heartbeat, never by a command-line match (`CLAUDE.md`
  rule 2). It exits when the build exits. On `CAP` it leaves the build running.
- When it ends, send the agent the run's final `status.json` (less the fingerprint's per-class
  detail), the launch log's last lines, and the verdict line, and nothing else.
  - On `ENDED`, it carries on.
  - On `CAP`, the two-hour stop is met. Tell the agent, and ask the user before anything else
    happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — fourteen checks

1. **Scope.**
   - `git diff --name-status HEAD~1 HEAD`: `A` for the log, the saved measurement and the `*_08*`
     outputs; `M` for this board, the verification document and, if an issue was opened,
     `docs/OPEN_ISSUES.md`. README is not modified. No tracked code, test, tool or rehearsal script
     is modified.
   - In the verification document, the diff adds §3 and touches no line of the header, §1 or §2.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/dk-and-rpk-migration-08-*` yourself.
   - `reference` and `resumed-c` are `done`/0; `resumed-a` and `resumed-b` are `killed`/143, each
     with `fingerprint` and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's lines and no traceback.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on, and the
     handler's line came within the second or two.
   - The next launch's counts show the killed stage had stored some rows and not all.
   - No process the agent started is alive, and no Ray process of this repository.
3. **(R5) The manifests.** For each: `git_head` the dispatch `HEAD`, `git_dirty` `false`;
   `packages` the prompt's §2 record exactly, compared as JSON; `script_sha256` `b7935877…` in all
   four; `argv` holds `--build --rehearsal`, with `--resume` on `resumed-b` and `-c` only; `scope`
   begins `REHEARSAL build, not A3 v2:`; `results` names the right store; each `--purpose` names
   R2, both packages and their commits, and `39809dca`.
4. **(R2) Nothing stored was recomputed.** Run `stage_counts.py` on the four run directories
   yourself, and compare it with the log's table and with R1's `stage_counts_05.txt`.
   - The `reference` column equals R1's, stage by stage.
   - Every stage completed before an interruption computes 0 on the next launch.
   - `resumed-b`'s `GkSource` = 291 − (stored by `resumed-a`) + (pruned); `resumed-c`'s QSI = 864 −
     (stored by `resumed-b`), and nothing else. The totals per class over the chain equal the
     reference's.
5. **(R2) The prune.** Run `prune_check.py` on the agent's saved `closed_store_state.py` output
   after `resumed-a`: every listed serial absent now (or reused by a later row), read `mode=ro`.
   `read_inventory` reports no `duplicate`. `resumed-b`'s integrity warnings count the same rows.
6. **(R2) The fingerprints.** Fingerprint both closed stores yourself, read-only:
   - `compare_stores.py <reference> <resumed>`: equal, each equal to its sidecar;
   - `compare_with_recorded.py` on both primaries `--against` the four run directories of §0: 0
     differences each;
   - format 6, `39809dca…`, 21 classes.
7. **(R3) Integrity.** Run `store_soundness.py` and `replicated_serial_census.py` on both primaries
   yourself: `integrity_check` `ok`, `foreign_key_check` empty, one `version` row per shard (label
   `2025.1.1`), no census divergence, no inventory problem, `replication_in_flight` empty, no
   journal file.
8. **(R4) The readers.** Each reader's log under `var/dk-and-rpk-migration-08/extract/`: exit 0, no
   `ReadOnlyMiss`, `ReadOnlyWrite` or traceback; hashes and mtimes equal around each;
   `extract_tensor_source_data.py` exit 0; the `nomodels extract_Gk_data` run printed the skip line.
   The readers now import the pool from `raypoolkit` (07). Run one reader yourself through
   `run_readers.sh` into the scratchpad, and hash the store before and after.
9. **(R6) L4.**
   - The `** ` banner lines of R2's reference `stdout.log`, in order, equal R1's (numbers and the
     prompt number stripped).
   - The resumed-chain identities of check 4 are in the log as a table beside verification §2.5.3,
     and logged as the `STRUCTURALLY REQUIRED` reading of README §2.2.
10. **(R7) C8.** Run `policy_statistics.py` on R2's `reference`, `resumed-b` and `resumed-c`
    yourself.
    - Each: 14 lines; the counts R1's (291; 109 / 26 (25 complete, 1 minimal) / 156).
    - Against R1's same slug (`policy_statistics_05_reference.txt`; R1's resumed blocks under
      `var/dk-and-rpk-migration-05/checks/`): `diff` empty, or differing only in order and equal
      sorted, as the log says.
    - The log and verification §3's "`main.py:3247`" subsection explain the result from
      `main.py:2871-2893` and `raypoolkit/RayWorkPool.py:529-530`, trace the persisted object
      through the installed `datastorekit`, and say `resumed-c`'s block takes the lookup path.
    - If a count differs: a §3 issue is opened, with the index row, and nothing was fixed.
11. **(R8) The printed lines.** For each slug, against R1's:
    - the count of `ALL WORK ITEMS COMPLETE in time`, `Queue summary:` and
      `(no compute items were evaluated` lines equal, each line's form equal with numbers stripped;
    - any `work items remaining` / `inflight:` pair in R2 in the form of §0 correction 2's lines;
    - each final `status.json` `stage`, and each killed run's `stage` at the signal (the
      interrupter's log), R1's.
12. **(R6) The log differences.** For each slug, compare R2's `stderr.log`, launch log and
    `stdout.log` with R1's yourself, as line kinds with times, pids, ports, paths and numbers
    stripped. Every kind in one and not the other is in the log, with an explanation that holds
    (§0: the `IntegrationWarning`'s launch, Ray's folding, a progress line past 300 s). No
    traceback, no `ValueError`, no `ReadOnlyMiss`/`ReadOnlyWrite`, no refused open.
13. **(R9, R10) G-RPK, R0 and R1.**
    - In a fresh `python -I`, `import RayTools` raises `ModuleNotFoundError`, and
      `raypoolkit.__file__` is in the venv's `site-packages`.
    - Run RayPoolKit's script at the agent's commit yourself: §0's figures; the saved
      `measure/out/08-…md` differs from it only by commit, file count and record lines.
    - G-RPK's five parts are each stated with their commit or run: `643c3b2` and its review
      (`af26914`) for parts 1–2, with the saved measurement at the dispatch `HEAD`; `b6cc119` (06)
      and its review (`cd03ec4`) with R2's manifests for part 3; the six roots at the dispatch
      `HEAD` for part 4; R2's `reference` and `resumed-c` for part 5.
    - R0's and R1's 24 files hash as §0's table, before the agent's first launch (its record) and
      now.
14. **The records, the suites and `black`.**
    - The board: the §1 row for 08, L1–L4 (R2) and G-RPK ✅, the §2 "RayPoolKit G2" row holding
      with its evidence, `### 5.8 After prompt 08`; §3 entries and the index for anything opened,
      the count right by counting. The verification document's §3 has §2's subsections in order,
      plus "`main.py:3247`", every command verbatim, and its numbers match the run directories. The
      log classifies §0's five corrections.
    - All six roots, serially, at the commit, every count as §1 step 5. `black --check` is clean on
      any Python file added.

## 4. After the review

- **Record it on this board** as a §5.8 *Orchestrator review of prompt 08* paragraph, in 05's and
  07's form. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log's header;
  - the board's §2 "RayPoolKit G2" row, if the agent left it ⬜ with every part holding;
  - this note added to the board's list of notes, marked "used for 08".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - R1–R10, each in a line;
  - the fingerprints against `39809dca…`, R1's, R0's, 07's and 09's;
  - C8: the statistics counts against R1's, and why;
  - where each interruption fell, how it was chosen, and what the resume pruned and recomputed;
  - every log difference from R1 and its explanation, and whether any progress line appeared;
  - whether RayPoolKit's G2 holds, and that recording it in RayPoolKit is the user's (C12);
  - anything opened; the stores, which stay (README §3).
- **Then 09**, the close. Its note is written after this review and carries 05–08's results to it.
