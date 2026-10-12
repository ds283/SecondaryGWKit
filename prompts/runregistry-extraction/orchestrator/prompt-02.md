# Orchestrator — prompt 02, rehearse the start tree (R0)

Read [`../README.md`](../README.md) first: §0.1, §0.2, §2.2 (02's charter), §3, §4.1, §5 (rules 9–11,
14, 15) and §8. Read also `CLAUDE.md`'s "Long-running jobs — the run registry" and the freeze
section, whose rules bind you as much as the agent. Then read `dk-and-rpk-migration`'s
[`orchestrator/prompt-08.md`](../../dk-and-rpk-migration/orchestrator/prompt-08.md), the note for
the R2 this prompt repeats, whose protocol, wait and review this note carries over except where §0
below says otherwise, and that board's *Orchestrator review of prompt 08* (§5.8). Then this board's
*Orchestrator review of prompt 01*.

**You do not write code.** You may:
- run the interrupter's dummy tests in the scratchpad;
- run the one wait per launch that §2.1 describes;
- read the two new stores read-only, fingerprint them, and run the census, `PRAGMA
  integrity_check`, `foreign_key_check` and `read_inventory` on them;
- read R2's run directories, and hash the sidecars under `var/datastores/`;
- run the suites;
- fix small residue in a follow-up commit of your own (§4).

**While a build is running you write nothing in the repository**, outside `var/` and the session
scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop.

**The prompt:** [`02-rehearse-the-start-tree.md`](../02-rehearse-the-start-tree.md)
**Board items:** L1–L4 (R0) · **Gate it decides:** R0 · **Closes:** nothing · **Narrows:** nothing ·
**Opens:** anything found · **Model:** Opus, as the prompt recommends.

**Gate:**
- 01 landed (`a45aefc`) and was reviewed (`cf978c6`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/` and `CLAUDE.md` (correction 1).
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive.

## 0. What makes this prompt unusual

**It is `dk-and-rpk-migration`'s R2 again, at a tree whose classes live at new module paths.** R2
ran at `5631f0e` (code `643c3b2`'s). Since then the only change to what a build runs is `511ef5f`'s
move of the factories to `ObjectFactories/`, and `7d76d3e`'s of the store fixtures, which no build
imports. By `ast`, docstrings stripped and the module path normalised, every non-test Python file
those commits touched that a build or a reader runs is unchanged: `main.py`, `config/datastore.py`,
`extract_common.py`, `RunRegistry/stores.py`, `tools/inventory_report.py` and the 21 factory
modules. So **if R0 differs from R2, the module path Ray pickles is the first suspect.** A miss here
is this repository's (README §4.1), not a package's.

**It runs for half an hour or more, and no conversation may own it.** As at `dk-and-rpk-migration`'s
02, 05 and 08: the agent launches each build detached, through the registry, checks once, and ends
its turn. You run **one** wait per launch (§2.1) and resume it when the run is no longer alive. Four
launches, four waits.

**The previous rehearsal's stores are gone.** `718e650` retired all six of `dk-and-rpk-migration`'s
stores; only their sidecars remain (the 14 tombstones of 01's baseline). So, unlike 05 and 08,
there are no earlier store files to hash. R0 is compared with R2 through the fingerprints R2's run
directories recorded, and the agent shows that it left the 14 sidecars alone.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `cf978c6`.
Each correction below governs, and the agent logs it as an `IMPLEMENTATION CHOICE` citing this note,
or `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.

1. **The gate's "no commit after 01's touches code (anything outside `prompts/`)".** `cf978c6`,
   01's review, changed one line of `CLAUDE.md`, adding `a45aefc` to the freeze section. No build,
   test or script reads `CLAUDE.md`; `git diff --stat a45aefc cf978c6 -- . ':!prompts'` lists that
   file alone. The code that runs at the dispatch `HEAD` is `a45aefc`'s, and the gate holds in
   substance. The log says so.
2. **`prune_serials.py` is not run** (inherited through 08, 05 and 02; README §4.1 L2). It
   hardcodes `a3-v2-readiness` 07's serials. R0, R1 and R2 of `dk-and-rpk-migration` ran
   `prune_check.py` in its place on their own `closed_store_state.py` output, and this R0 does the
   same.
3. **Which `store_soundness.py`.** The six scripts of the prompt's §1 are
   `docs/dk-and-rpk-migration/rehearsal/`'s: `stage_counts.py`, `prune_check.py`,
   `policy_statistics.py`, `compare_with_recorded.py`, `store_soundness.py` and `run_readers.sh`.
   `docs/a3-v2-readiness/rehearsal/store_soundness.py` is not run: it raises `TypeError`
   (`[02-07s-store-soundness-calls-read-inventory-without-its-registry]`). The other tools run from
   where R2 ran them: `closed_store_state.py` under `docs/a3-v2-readiness/rehearsal/`;
   `replicated_serial_census.py` under `docs/datastore-integrity-audit/`; `compare_stores.py`,
   `interrupt_when.py` and the two dummy tests' scripts under `docs/datastore-integrity-verification/`
   and `docs/a3-v2-readiness/`. None of them is copied (C10).
4. **The `resumed-a` point** (prompt §1, "the interruption points are R0's own … by the same
   reasoning"). R2 used `--interval 5 --hold 10`, which put its signal 10–15 s into a 41.8 s stage,
   and **stored no `GkSource` row**, so its prune at open held only vacuously (R2's verification
   §3.3.2; that board's §5.8). R1, at the same hold in a 55.1 s stage, stored 101 (45 unvalidated);
   R0 of that campaign, in a 32.2 s stage, 119 (43). The reasoning behind the point is that
   `resumed-a` leaves stored and unvalidated rows for `resumed-b` to prune, and the signal lands well
   before the stage ends. So:
   - **size the hold at about 40 % of this R0's own `reference` `GkSource` stage time**, rounded to
     whole seconds (for a 42 s stage, `--hold 17`, a signal 17–22 s in), and not below 10 s;
   - if `resumed-a` still stores no `GkSource` row, that is recorded as at R2, not a stop: the run
     was killed in the right stage;
   - if the stage completes during the hold and the build runs on to `done`, that is a missed
     interruption and a stop.

   R1's `resumed-b` (`var/runs/dk-and-rpk-migration-05-resumed-b-*`) is then the comparison for
   what a real prune prints: its integrity warnings and Ray's folding of them. If the prune is
   exercised here, `resumed-b`'s logs gain those lines against R2's, explained by the point.
5. **The `resumed-b` point.** From the QSI banner, pattern `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`,
   **not anchored at its end**, `--interval 5`, with `--hold` about a third of this R0's `reference`
   QSI stage, **and at most 250 s**, so that the hold ends before the queue's 300 s minimum notify
   interval (`main.py:119`) can print a progress line.
6. **A progress line may appear.** R2's `reference` QSI stage took 299.2 s, 0.8 s inside the 300 s
   minimum, and no R2 build printed a progress line. Under this machine's load (§1 step 3) a stage
   may pass 300 s and print one. That is the threshold and the load, not a finding. Compare its
   form with the copy's last build line,
   `var/runs/a3-v2-readiness-07-reference-20261004T012909/stdout.log:100-101`, and with R2's GkWKB
   reader's two lines; and a `status.json` `progress` field filled from it, where R2's was `null`,
   is explained the same way.
7. **The saved outputs wait for the last launch.** `docs/runregistry-extraction/` does not exist.
   Every capture goes under `var/runregistry-extraction-02/` until `resumed-c` has ended, and is
   copied into `docs/runregistry-extraction/rehearsal/` only after it.
8. **The verification document's subsections** (prompt §6: "the subsections of
   `docs/dk-and-rpk-migration-verification.md` §3 in the same order"). §3.10 (G-RPK) and §3.11
   (`main.py:3247`) record that campaign's RayPoolKit gate and C8, which have no counterpart here.
   §1 carries §3.1–§3.9, in order. It may add a subsection after them for what is this R0's own
   (the factory move, R7 below), named in the log.

**What moves on purpose:**
- Added: `docs/runregistry-extraction-verification.md`, with the header and §1;
  `docs/runregistry-extraction/rehearsal/*_02*` outputs; the log.
- Modified: this board, and `docs/OPEN_ISSUES.md` only if an issue is opened.
- No script anywhere changes. A new one of the agent's, if any, goes under
  `docs/runregistry-extraction/rehearsal/` and is named in the log.
- **README does not move**, and neither does `prompts/INDEX.md` (the orchestrator's, §4).
- Under `var/`: the two stores and their sidecars, four run directories and their launch logs, and
  `var/runregistry-extraction-02/`. **Nothing else under `var/` changes**, the 14 sidecars
  included.
- The index is 105, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-12 at `cf978c6`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `RAY_ADDRESS` unset. `pgrep -lf 'gcs_server|raylet|ray::'`
  and the whole-table search find **nothing**. 10 CPUs. **The load was 177 at 09:02, falling to 29
  by 09:15**, from macOS services (Fantastical, `eligibilityd`, `contactsd`, Maps, `studentd`) and
  another session's Python 3.13 suite in DatastoreKit, nothing of this repository. 70 GB free on
  `var/`'s volume (93 % used).
- **The registry.** 29 runs: 27 finished, 2 `unknown` (the pre-registry `a3-pilot` pair). None
  `running` or stale. It equals 01's baseline, `measure/out/01-list.txt`.
- **The venv, and what every manifest should hold.** `pip check` is clean; `pip show runregistrykit`
  finds nothing. `RunRegistry.package_provenance()` gives, with keys sorted:
  `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false,
  "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version":
  "0.2.2"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable":
  false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version":
  "0.2.0"}}`, which is R2's record. Compare as JSON, not as text.
- **The launcher is R2's.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` is
  **`b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`**; its last change is
  `0c9e2cf`. Every R0 manifest's `script_sha256` should be this.
- **The build still applies to `main.py`.** `substitute_main_py(filter_cases=False)`, loaded by
  path, replaces each of its four literals exactly once and the result compiles, with Ray down. So
  the four `replaced 1 occurrence` lines recur.
- **The names are free.** `store show` on `var/datastores/runregistry-extraction-02-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either, exit 0. Nothing under
  `var/datastores/` or `var/runs/` contains `runregistry-extraction`; `var/runregistry-extraction-02/`
  does not exist; no `-journal`, `-wal` or `-shm` file under `var/datastores/`.
- **What is under `var/datastores/`**: the 14 sidecars and nothing else.
  `find var/datastores -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256` is
  `023d139fb714e229…`. The agent records the full list before its first launch and after its last
  step; the 14 must be unchanged, and the only new files its two stores' and their sidecars.
- **The recorded fingerprints to compare against**, as run directories for
  `compare_with_recorded.py --against`:
  - R2: `var/runs/dk-and-rpk-migration-08-reference-20261011T224347` and
    `var/runs/dk-and-rpk-migration-08-resumed-c-20261011T230717`. Each `status.json` holds
    `fingerprint_format` 6, digest `39809dca…`, 21 classes, 0 problems.
- **The scripts run at `cf978c6`.**
  - `stage_counts.py` on R2's four run directories reproduces `stage_counts_08.txt` byte for byte;
    `policy_statistics.py` on R2's `reference` reproduces `policy_statistics_08_reference.txt`.
  - On a `StoreFixtures.real_store_fixtures` store built in the scratchpad: `store_soundness.py`
    exits 0, `=> clean`; `closed_store_state.py` exits 0; `replicated_serial_census.py` exits 0,
    `=> no divergence`; `compare_stores.py` and `compare_with_recorded.py` run to a verdict (exit
    1: the fixture has no sidecar and is not the reference store).
  - So a script that fails in the agent's run fails on the build's store, not on an import.
- **The interrupter passes at `cf978c6`.** Both dummy tests, run into the scratchpad at
  09:04:46–09:05:27, exit 0, and each output equals R2's (`interrupter_dummy_test_08.txt`,
  `interrupter_build_dummy_test_08.txt`) once times, pids and paths are stripped. No child left
  alive. The agent runs them again itself.
- **What R2's logs held**, against which the log comparison is made (verification §3.2.5, §3.5.6):
  each killed `stderr.log` the handler's two lines (both `SystemExit(1)`); the one SciPy
  `IntegrationWarning` in `reference` and in whichever resumed launch computes its integral (R2:
  `resumed-c`); no integrity warning in `resumed-b`, because nothing was unvalidated; no progress
  line in any build. Ray's `[repeated Nx across cluster]` folding varies run to run;
  `SystemExit(1)` against `SystemExit(143)` depends on where the signal lands.
- **The stage times.** R2's reference (verification §3.5.5), at load 13–33: `GkSource` **41.8 s**,
  QSI **299.2 s**, the build 11 m 25 s. R1's: 55.1 s, 289.7 s, 12 m 13 s. The earlier
  `dk-and-rpk-migration` R0's: 32.2 s, 212.3 s, 8 m 0 s.
- **The commands are R2's**, verbatim in its verification §3.1 and §3.2.6, with
  `runregistry-extraction-02`, `--campaign runregistry-extraction --prompt 02` and the store names
  of the prompt's §1.
- **No Ray process left: list it properly** (`[07-a-killed-builds-ray-agents-outlive-the-driver]`).
  `pgrep -lf 'gcs_server|raylet|ray::'` and the whole-table search, `ps -axo pid,ppid,lstart,command`
  for lines holding both `SecondaryGWKit/venv` and `ray`. Another session's `zsh -c` whose text
  holds the pattern is not a Ray process. Wait for any listed agent **by pid**, never by pattern,
  and record how long it took, beside R2's (none outlived either killed driver).
- **Shell traps.**
  - `stat` here is GNU coreutils: use `/usr/bin/stat -f '%m %N'` or Python's `os.stat`.
  - `grep` may wrap `ugrep`: use `git grep` or Python for any regex the log relies on.
  - The shell is `zsh`: an unquoted `$var` holding spaces is **not** split into words (use
    `${=var}` or an array), and a bare word beginning `=` is command expansion. Quote it.
- **Expected counts**, at `cf978c6`, serially, load 74–129:

  | Root | Tests | Result |
  |---|---|---|
  | AdaptiveLevin | 32 | OK |
  | CosmologyModels | 39 | OK |
  | ObjectFactories | 271 | OK |
  | tools | 49 | OK |
  | LiouvilleGreen | 148 | OK (1 skipped) |
  | RunRegistry | 286 | OK |
  | ComputeTargets | 693 | 1 failure: `test_wall_time_per_object` (0.073 s against 0.06 s); alone, failed twice (0.073 s) and passed the third time, load 29–33 |

  Every count is 01's. The wall-clock test is README §8's first known flake.
- **The index:** 105.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `done`/0 and `killed`/143, each with a fingerprint, each
  driver's exit status in its launch log, and no Ray process of this repository left.
- **(R2) R0 is `dk-and-rpk-migration`'s R2.** Both stores equal each other, `39809dca…`, and the
  fingerprints recorded by R2's `reference` and `resumed-c`, class by class. Nothing stored was
  recomputed; the prune removed exactly the unvalidated rows, if there were any.
- **(R3) The stores are sound**, by `store_soundness.py` and the census.
- **(R4) Every reader works, and none touches the store.**
- **(R5) The manifests say what ran.** Every `packages` is R2's record, every `git_dirty` `false`,
  every `git_head` the dispatch `HEAD`, and every `script_sha256` `b7935877…`.
- **(R6) L4 equals R2's**: the statistics counts, the reference stage counts, the chain's
  identities, the stage sequence, the printed lines and registry fields. Every log difference from
  R2 is listed and explained.
- **(R7) The factory move is invisible to a build**: no import or unpickling error in any build or
  reader, and the fingerprint unmoved.
- **(R8) Nothing else under `var/` moved.** The 14 sidecars keep their hashes.

**Conventions.** 08's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §8's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended**. Under load, the four tests of README §8 are known to fail: re-run a failure
  alone before calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/runregistry-extraction-02/` (`CLAUDE.md` rule 5). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched.
- **The venv.** The agent installs and uninstalls nothing.
- **`RunRegistry/` is frozen.** The agent changes nothing under it (README §5 rule 14). A defect
  found in it is recorded and the agent stops.
- **DatastoreKit and RayPoolKit.** Not read, not run, not changed (README §5 rule 12).
- **Never touch a process you did not start.**
- **Stop conditions are real.** A stop leaves the stores, the run directories and the tree as they
  are: no relaunch, no deletion, no fix. It goes to the user through the orchestrator, and the
  agent resumes with `SendMessage`.
- **Usage limits.** A usage stop does not stop a detached build. Resume the agent with
  `SendMessage`, saying exactly where it stopped. If it commits but sends no report, review from the
  commit, its log and its outputs.

## 1. Before you dispatch

Done on 2026-10-12 at `cf978c6`, before this note was committed. Repeat steps 1–4 at the dispatch
`HEAD` if anything lands in between.

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline a45aefc..HEAD -- . ':!prompts'` must be `cf978c6` alone (correction 1).
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`. Report
   any stale or unknown run, without acting on it.
3. **Ray and the machine.** `pgrep -lf 'gcs_server|raylet|ray::'`, and the whole-table search of
   §0. No process from this repository's `venv`. `RAY_ADDRESS` unset. Record the load, and say so
   if `var/` has less than 10 GB free.
4. **The venv.** `package_provenance()` gives §0's record; `pip check` is clean.
5. **Baselines**, as §0's table.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 105.
7. **The names are free**, and `var/datastores/` holds the 14 sidecars alone, hashing as §0.
8. **The interrupter.** Both dummy tests at `HEAD`, into the scratchpad. Both pass.
9. **The scripts**, on R2's run directories and a fixture store, as §0.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, and through it `dk-and-rpk-migration`'s 08, 05 and 02;
- the campaign README and this board;
- `CLAUDE.md`'s two sections named above;
- `dk-and-rpk-migration`'s 08 log and its verification §3;
- 01's log;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, every check and the suites. Its log is
  `logs/02-rehearse-the-start-tree.md`, in README §5.1's form, with what `dk-and-rpk-migration`
  08's log records, against R2.
- **The hand-off.**
  - After each launch: one check, then a report giving the run id, the Python pid, the `sh` pid,
    and the interrupter's pid and arguments if there is one, then the end of its turn.
  - You resume it with the run's final `status.json`, the last lines of its launch log, and the
    wait's verdict.
  - It never waits, polls or sleeps on a build itself. Waiting by pid for a killed build's orphaned
    Ray agents, before the next launch, is the one bounded wait it runs.
- **The files it may change** are the prompt's §6 list, nothing else. **It touches none of them, and
  creates nothing in the repository outside `var/`, until the last build launch has ended**, so
  that every manifest stamps `git_dirty: false`.
- **The board's header** says 2 of 9 landed once it commits (README §6.2 C7); `prompts/INDEX.md` is
  the orchestrator's.
- **The interruption points** come from R0's own reference `stdout.log`, as corrections 4 and 5
  say. A missed interruption is a stop.
- **Stop and ask** on any of the prompt's §4 conditions. A fingerprint miss is the one R0 exists
  for: record both fingerprints, `compare_fingerprints` by class, and stop. **A statistics count
  that differs, with the fingerprint holding, is not a stop**: it is a finding, recorded, explained
  if it can be, and opened as an issue.

### 2.1 The one wait per launch

08's, unchanged. When the agent reports a launch, run **one** background command that ends when
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

## 3. The review — thirteen checks

1. **Scope.**
   - `git diff --name-status HEAD~1 HEAD`: `A` for the log, the verification document and the
     `*_02*` outputs; `M` for this board and, if an issue was opened, `docs/OPEN_ISSUES.md`.
     README, `prompts/INDEX.md` and `orchestrator/` are not modified. No tracked code, test, tool
     or script is modified, and nothing under `RunRegistry/`.
   - The verification document's header names §1–§4 as the prompt's §6 says, and holds §1 alone.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/runregistry-extraction-02-*` yourself.
   - `reference` and `resumed-c` are `done`/0; `resumed-a` and `resumed-b` are `killed`/143, each
     with `fingerprint` and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's lines and no traceback.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on, and the
     handler's line came within the second or two. The holds are those corrections 4 and 5 give,
     computed from R0's own reference.
   - The next launch's counts show the killed stage had stored some rows and not all, or, for
     `resumed-a`, none, as the log says.
   - No process the agent started is alive, and no Ray process of this repository.
3. **(R5) The manifests.** For each: `git_head` the dispatch `HEAD`, `git_dirty` `false`;
   `packages` R2's record exactly, compared as JSON; `script_sha256` `b7935877…` in all four;
   `argv` holds `--build --rehearsal`, with `--resume` on `resumed-b` and `-c` only; `scope` begins
   `REHEARSAL build, not A3 v2:`; `results` names the right store; each `--purpose` names R0 of
   this campaign, the start tree's commit and `39809dca`.
4. **(R2) Nothing stored was recomputed.** Run `stage_counts.py` on the four run directories
   yourself, and compare it with the log's table and with R2's `stage_counts_08.txt`.
   - The `reference` column equals R2's, stage by stage.
   - Every stage completed before an interruption computes 0 on the next launch.
   - `resumed-b`'s `GkSource` = 291 − (stored by `resumed-a`) + (pruned); `resumed-c`'s QSI = 864 −
     (stored by `resumed-b`), and nothing else. The totals per class over the chain equal the
     reference's. Any excess equals rows in flight at a `SIGTERM`, as the log says.
5. **(R2) The prune.** Run `prune_check.py` on the agent's saved `closed_store_state.py` output
   after `resumed-a`: every listed serial absent now (or reused by a later row), read `mode=ro`.
   `read_inventory` reports no `duplicate`. `resumed-b`'s integrity warnings count the same rows.
   If `resumed-a` stored none, the clause holds vacuously, as at R2, and the log says so.
6. **(R2, R7) The fingerprints.** Fingerprint both closed stores yourself, read-only:
   - `compare_stores.py <reference> <resumed>`: equal, each equal to its sidecar;
   - `compare_with_recorded.py` on both primaries `--against` R2's two run directories: 0
     differences each;
   - format 6, `39809dca…`, 21 classes.
7. **(R3) Integrity.** Run `store_soundness.py` and `replicated_serial_census.py` on both primaries
   yourself: `integrity_check` `ok`, `foreign_key_check` empty, one `version` row per shard (label
   `2025.1.1`), no census divergence, no inventory problem, `replication_in_flight` empty, no
   journal file.
8. **(R4, R7) The readers.** Each reader's log under `var/runregistry-extraction-02/extract/`: exit 0,
   no `ReadOnlyMiss`, `ReadOnlyWrite`, `ModuleNotFoundError`, unpickling error or traceback; hashes
   and mtimes equal around each; `extract_tensor_source_data.py` exit 0; the `nomodels
   extract_Gk_data` run printed the skip line. Run one reader yourself through `run_readers.sh`
   into the scratchpad, and hash the store before and after.
9. **(R6) L4.**
   - Run `policy_statistics.py` on R0's `reference`, `resumed-b` and `resumed-c` yourself: 14 lines
     each; the counts R2's (291; 109 / 26 (25 complete, 1 minimal) / 156); against R2's same slug,
     `diff` empty or differing only in order and equal sorted, as the log says.
   - The `** ` banner lines of R0's reference `stdout.log`, in order, equal R2's (numbers and the
     campaign prefix stripped).
   - The resumed-chain identities of check 4 are in the log as a table beside R2's verification
     §3.5.3.
10. **(R6) The printed lines.** For each slug, against R2's:
    - the count of `ALL WORK ITEMS COMPLETE in time`, `Queue summary:` and
      `(no compute items were evaluated` lines equal, each line's form equal with numbers stripped;
    - any `work items remaining` / `inflight:` pair in the form of correction 6's lines;
    - each final `status.json` `stage`, and each killed run's `stage` at the signal, R2's.
11. **(R6) The log differences.** For each slug, compare R0's `stderr.log`, launch log and
    `stdout.log` with R2's yourself, as line kinds with times, pids, ports, paths, numbers and the
    campaign prefix stripped. Every kind in one and not the other is in the log, with an
    explanation that holds (§0: the `IntegrationWarning`'s launch, the integrity warnings of a real
    prune, Ray's folding, a progress line past 300 s). No traceback, no `ModuleNotFoundError`, no
    `ReadOnlyMiss`/`ReadOnlyWrite`, no refused open.
12. **(R8) `var/`.** The file list under `var/datastores/`, with hashes, before the agent's first
    launch (its record) and now: the 14 sidecars unchanged, and the only additions R0's two stores
    and their sidecars. A3 v2's guard refused and changed nothing.
13. **The records, the suites and `black`.**
    - The board: the header (2 landed), the §1 row for 02, L1–L4 (R0) ✅, the §2 R0 row holding
      with the two runs that finished `done`, `### 5.2 After prompt 02`; §3 entries and the index
      for anything opened, the count right by counting. The verification document's §1 has R2's
      §3.1–§3.9 in order (correction 8), every command verbatim, and its numbers match the run
      directories. The log classifies §0's eight corrections.
    - All seven roots, serially, at the commit, every count as §0. `black --check` is clean on any
      Python file added.

## 4. After the review

- **Record it on this board** as a §5.2 *Orchestrator review of prompt 02* paragraph, in 01's form.
  Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log's header;
  - the board's §2 R0 row, if the agent left it ⬜ with every part holding;
  - this note added to the board's list of notes, marked "used for 02";
  - `prompts/INDEX.md`'s row: 2 of 9 landed.
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - R1–R8, each in a line;
  - the fingerprints against `39809dca…` and R2's two;
  - the statistics counts against R2's;
  - where each interruption fell, how it was chosen, and what the resume pruned and recomputed;
  - every log difference from R2 and its explanation, and whether any progress line appeared;
  - anything opened; the stores, which stay (README §3).
- **Then 03 waits on E1**, DatastoreKit's release. R0 holding is what lets a miss at R1 be laid on
  the pin.
