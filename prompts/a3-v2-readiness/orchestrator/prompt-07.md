# Orchestrator — prompt 07, rehearse the A3 v2 launch

Read [`../README.md`](../README.md) first: §0, §1, §3, §4 and §5, and §6.1. Read also `CLAUDE.md`'s
"Long-running jobs — the run registry", whose rules bind you as much as the agent. Then read
[`prompt-06a.md`](prompt-06a.md) §0, this board's *Orchestrator review of prompt 06a*, and
`prompts/datastore-integrity/orchestrator/prompt-10.md`, the note for the rehearsal this one
follows.

**You do not write code.** You may:
- read the two new stores read-only, fingerprint them, and run the census, `PRAGMA integrity_check`,
  `foreign_key_check` and `read_inventory` on them;
- run the one wait per launch that §2.1 describes;
- run the dummy tests of the interrupter in the scratchpad;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`07-rehearse-the-a3-v2-launch.md`](../07-rehearse-the-a3-v2-launch.md)
**Board items:** L1–L3 · **Closes:** nothing by name; it is the campaign's last prompt ·
**Model:** Opus.
**Gate:**
- 06a has landed and been reviewed, and its review completed §0's "Carried from the review of
  06a" below;
- **the user has taken D7-1 and D7-2** (§1 item 9);
- `HEAD` is the commit that records them, or a later one that touches no code.

**Never dispatched concurrently with anything else in this checkout**, and never while any Ray
instance of this repository is alive.

*Written 2026-10-03 at `118ca94`, with prompt 07 and before 06a was dispatched, at the user's
request. The facts below were checked then. Re-check every one marked [measured] at dispatch, and
complete the 06a section first.*

## 0. What makes this prompt unusual

**It is the campaign's only real run, and A3 v2's launcher is what runs.** Every earlier prompt was
tested on stand-ins, fixtures or probes. This one builds two stores under `var/datastores/`, the one
exception README §3 and §5 rule 10 allow, through `quadsource_atol_sweep.py --build --rehearsal
--register`. It interrupts one store twice with `SIGTERM`, resumes it with `--build --resume`, and
reads both stores back.

**It runs for an hour or so, and no conversation may own it.** The agent launches each build
detached, through the registry, checks once that it started, and ends its turn. You wait, with
**one** wait per launch (§2.1), and resume the agent when the run is no longer alive. Neither of you
polls. Under D7-1 as recommended there are four launches, so four waits: `reference`, then
`resumed-a`, `-b` and `-c`.

**It is the second rehearsal on this scope, and the first was the defect list.**
`datastore-integrity` 10 built the same scope on 2026-10-01 through `scoped_pipeline_run.py`. Its
five findings are this campaign's §0, items 1–5. So the review reads 07 against it, finding by
finding:

| 2026-10-01 (`docs/datastore-integrity-verification.md`) | What 07 should show instead |
|---|---|
| `resumed-a` recorded `failed`/1 (§2.1.1) | `killed`/143, with a fingerprint |
| `resumed-b` recorded nothing, aborted at 3 s, left `gcs_server` 6426 (§2.1.1) | `killed`/143 within about a second; no Ray process of this repository left |
| 45 `GkSource` rows duplicated under `--no-prune-unvalidated` (§2.1.4) | the unvalidated rows pruned at the next open; no `duplicate` |
| every reader exited 1 on `QCD_Cosmology` (§2.3) | every reader exits 0 with `--models LambdaCDM`, and without it after a skip line |
| the QSI reader wrote to the store it read (issue `[08-…]`) | no file of the store changes under any reader |
| the extractor read at `1e-8` | it reads every `QuadSourceIntegral` row of a `1e-7` store |

**The user's decisions come first.** The prompt's §1.1 states D7-1 and D7-2, each with a
recommendation. Put both to the user before dispatch (§1 item 9). Record the answer, dated, in the
prompt's §1.1 and on the board's §1, in a commit of its own. Give the decisions to the agent as
decided.

**The facts, checked by the orchestrator on 2026-10-03 at `118ca94`.** Pass them on; the agent need
not rediscover them. Each is marked as audit 00 marks its findings.

- **The names are free** [measured]. `store show` on `var/datastores/a3-v2-readiness-07-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either. No run directory
  under `var/runs/` begins `a3-v2`.
- **The scope's cost** [measured on 2026-10-01, at `rtol = 1e-8`]. `datastore-integrity` 10's
  `reference` took 14 m 58 s at a load of 22.5 on 10 CPUs. Its `GkSource` stage took 34 s and
  printed no intermediate progress line. Its `QuadSourceIntegral` stage was 2 work items of 864
  integrals, so every progress line read 100.00 %, the first at 5 m 01 s with 594 stored. At
  `1e-7` the QSI stage should be no slower. The prompt's two-hour cap is eight times the
  reference's wall time.
- **The counts 07's stores should reproduce** [read, verification §2.1.5], by class: `version` 1,
  `store_tag` 10, `redshift` 1159, `wavenumber` 3, `tolerance` 6, `LambdaCDM` 1, `QCD_Cosmology` 1,
  `IntegrationSolver` 7, `GkSourcePolicy` 2, `QuadSourcePolicy` 2, `wavenumber_exit_time` 3,
  `BackgroundModel` 1, `TkNumericIntegration` 3, `TkWKBIntegration` 3, `GkNumericIntegration` 1368,
  `GkWKBIntegration` 3477, `GkSource` 291, `GkSourcePolicyData` 291, `QuadSource` 6,
  `QuadSourceIntegral` 864, `OneLoopIntegral` 0. Same grid, `zend` and sampling (06a's §0
  measured the grid as equal floats). The tolerance rows differ in value, so digests are not
  compared.
- **Ray on this machine** [measured]. The user's ChamPBH cluster is running: `gcs_server` 62525 and
  `raylet` 62574, from `/Users/ds283/Documents/Code/ChamPBH/venv/`. It is the user's, and is left
  alone. So `ray status` finds a cluster, and the 2026-10-01 pre-launch check, "`ray status` finds
  none", cannot pass. The prompt replaces it with "no Ray process from this repository's `venv`".
  - Since 04 (K4), `run_build`'s `ray.init` passes `address="local"`, so the build starts its own
    instance beside ChamPBH's.
  - **The readers do not.** Each `extract_*.py` calls `ray.init(address=args.ray_address)` with
    the default `auto`, which would attach to ChamPBH's cluster. The prompt requires
    `--ray-address local`.
- **What a killed launch prints and records** [read, `scoped_pipeline_run.py`]:
  - the handler's line is `** SIGTERM at <time>: recorded killed, exit code 143; stopping the
    pipeline; the process will exit with status 143`. Its `<time>` is
    `time.strftime("%Y-%m-%d %H:%M:%S %z")`, whole seconds;
  - the interrupter's log uses `RunRegistry.now_iso()`, whole seconds too;
  - so the delay is bounded to a second from the logs. Prompt 04's probe measured 0.001–1.006 s.
- **The driver's exit status is not otherwise observable.** The build is detached. The prompt's
  launch wraps it in `sh -c '…; echo "** driver exit status $?"'`. `RunRegistry.begin` records the
  Python process's pid, which is the one the interrupter signals. The shell is its parent, and only
  writes the line.
- **A3 v2's guard** [read]. `run_build` refuses an existing store without `--resume` before
  `RunRegistry.begin` and before `ray.init`. It has run `import ray` by then, but started nothing.
  So the prompt's foreground check writes no manifest and starts no Ray.
- **`fingerprint_store` refuses a store that a `running` run names** (2026-10-01, log 10 deviation
  10). After a `killed` or `done` finish it does not.
- **Disk** [measured]. 111 GB free on the volume holding `var/`, at 89 % used. 10 CPUs.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch is recorded as it ended.** `done`/0 and `killed`/143, each with a
  fingerprint, each driver's exit status in its launch log, and no Ray process of this repository
  left behind.
- **(R2) The resume is the reference.**
  - Every stage finished before an interruption computes nothing on the next launch.
  - The unvalidated rows are pruned, and nothing is duplicated.
  - The resumed store's fingerprint equals the reference's, class by class.
- **(R3) The stores are sound.** The version row is on every shard, the census finds no divergence,
  `integrity_check` is `ok`, `foreign_key_check` is empty, and the inventory has no problem, on both.
- **(R4) Every reader works, and none touches the store.** All six exit 0 and refuse nothing. Every
  file is byte-identical afterwards, and the QSI extractor reads every row.

**Carried from the review of 06a** (2026-10-04, at `7bd202f`; the board's *Orchestrator review of
prompt 06a*).
- **The scope function** is `build_scope(args)` in `docs/handover/quadsource_atol_sweep.py`. It
  reads `args.rehearsal` only and returns `(k sample, zend string, scope text)`. The text is a
  `str.format` template over `atol`, `rtol`, `cpus` and `database`, which `run_build` fills with
  `args.atol`, `args.rtol`, `args.cpus` and the store's file name.
  - With `--rehearsal`: `[100000.0, 316227.7660168379, 1000000.0]`, `"250"`, and the rehearsal
    text.
  - Without it: `k_sample()`, `"0.1"`, and the production text, unchanged.
  - `run_build` takes the grid and `scope=` from one call. `build_main_argv` takes `--zend` from
    it, so `main.py`'s argv differs from production at the token after `--zend` alone.
- **The `scope` text a rehearsal manifest holds**, as the orchestrator printed it from the
  script's own parser at `--atol 1e-32 --rtol 1e-7` and the default 6 cpus, for a store named
  `store.sqlite`:
  `REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32 rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into store.sqlite`.
  The cpus and the store's name are the launch's own.
- **The line `run_build` prints** for the rehearsal scope, after its `atol=… rtol=…` and
  `main.py argv = …` lines:
  `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250`.
  It goes through the registered stream, so it is in the run's `stdout` copy.
- **`--resume` does not check a store's scope** (log 06a, observation 1). A store built with
  `--rehearsal` and resumed without it would mix the two grids and both `--zend`s, and nothing
  would refuse. **Every launch of 07 passes `--rehearsal`**, the resume included. The check is to
  read each manifest's `scope`, which must begin `REHEARSAL build, not A3 v2:`.
- **`--rehearsal` is refused without `--build`**, by `parser.error` (exit 2) before any mode runs.
  `--prepare --build --rehearsal` is not refused, and `--prepare` runs, because `main()` dispatches
  `--prepare` first. 07 never passes `--prepare`.
- **The interrupter reads the driver from the run's manifest.** `interrupt_when.py` no longer has
  the constant `DRIVER`. `run_driver(run_dir)` takes the basename of the manifest's `argv[0]`,
  which is `quadsource_atol_sweep.py` for a `--build` (`RunRegistry.begin` records `sys.argv` before
  `run_build` replaces it). Before it signals, `ps -p <pid> -o command=` must contain that name and
  `--register <slug>`. A manifest with no `argv[0]` is logged and exits 2 without signalling.
  `interrupter_dummy_test.sh`, unchanged, still passes all five cases. Its refusal line now reads
  `does not name scoped_pipeline_run.py (the run's own driver, its manifest's argv[0]) and --register …`.
- **B3's dummy case**, `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`, as the orchestrator
  re-ran it at `7bd202f` (abridged to its verdict lines):
  ```
  === case A: a --build manifest (argv[0] the sweep script), a child named for it: signals
  interrupter exit=0
  child exit=143 (143 = killed by SIGTERM)
  … sent SIGTERM to pid 60610 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT, 45.00% (11 of 20 work items remaining)' (45% >= 40%); command 'python docs/handover/quadsource_atol_sweep.py --build --register dummy-build-a 120'; exiting
  === case B: the same manifest, a child named for scoped_pipeline_run.py: refuses
  interrupter exit=2
  child 60616 still alive (not signalled)
  … pid 60616's command line does not name quadsource_atol_sweep.py (the run's own driver, its manifest's argv[0]) and --register dummy-build-b: 'python docs/gktk-remedial/scoped_pipeline_run.py --build --register dummy-build-b 30'; exiting without signalling
  === case C: a manifest whose argv[0] is empty: exits without signalling
  interrupter exit=2
  child 60631 still alive (not signalled)
  … the run's manifest names no argv[0]; exiting without signalling
  ```
  The full output is committed as `interrupter_build_dummy_test.txt`.
- **The command line the interrupter sees.** For case A to hold for 07's real build, the Python
  process whose pid `status.json` records must show `quadsource_atol_sweep.py` and
  `--register <slug>` in `ps -o command=`. A launch as
  `./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal … --register <slug> …`
  does, and `run_build`'s replacement of `sys.argv` does not change what `ps` shows.

**Carried from the reviews of 00–06.** All bind the agent, and the brief says so.
- **Staging and reporting.**
  - Stage by explicit path only, never `git add -A`, `.` or `commit -a`. Show
    `git diff --cached --name-only` before committing.
  - Count, then paste. A log states only what the record shows.
- **The SHA, not "this commit".** The agent cannot know its own SHA. Fixing that stays the
  orchestrator's residue.
- **Suites.** README §7's command, output to a file, the verdict grepped from it.
  - Under load, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
    `test_background_tau.test_build_cost` and `test_numeric_phase_cut.test_the_print_is_gated_both_ways`
    fail. Re-run a failure alone, and on an export of the parent commit, before calling it real.
  - At 06's review the first failed at every run, at loads of 15–34, and on an export of `dd7f272`
    too.
- **`var/`.** The agent writes only its two stores, their sidecars, its run directories and launch
  logs, and `var/a3-v2-readiness-07/`. It opens no other store, not even read-only.
- **Scratch.** The agent works in its own subdirectory of the session scratchpad. Long-running
  results go under `var/`, never a scratchpad (`CLAUDE.md` rule 5).
- **Never touch a process you did not start**, ChamPBH's above all.
- **Usage limits.** Resume a stopped agent with `SendMessage`, saying exactly where it stopped. A
  usage stop **does not stop a detached build**, which is the point of detaching it. Your wait
  still ends when the build does. If an agent commits but sends no report, review from the commit,
  its log and its outputs.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`, and say which commit it is.
   - `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows
     this checkout alone.
   - The commits since 06a's review touch no code.
2. **The registry.** `python -m RunRegistry list`: nothing `running`. Report any stale or unknown
   run, without acting on it. At `118ca94` it held 12 runs: 10 finished and 2 `unknown`.
3. **Ray and the machine.**
   - `pgrep -lf 'gcs_server|raylet|ray::'`: no process from this repository's `venv`. Record
     ChamPBH's by pid and start time, so the review can show they were not touched.
   - `RAY_ADDRESS` is unset.
   - `df -h var/`: say so if free space is below 10 GB. Record the load average and
     `sysctl -n hw.ncpu`.
4. **The snapshot.** Snapshot read-only, with a script in the session scratchpad that is never put
   in the tree:
   - what to cover: `var/datastores/`, `var/runs/`, `var/datastore-integrity-10/`, the root's
     `physics-test-n20-lambdacdm-zend0p1*.sqlite`, and `var/bootstrap-a3-resume.log`;
   - what to record: the SHA-256, size and mtime of every file, and the directory list;
   - open files only as bytes, never as databases.

   At `118ca94`, `var/` held 3213 files in 108 directories. **This prompt adds files, by design.**
   The review checks that every file in the snapshot is byte-identical afterwards. It also checks
   that every new path is one of:
   - the two stores' primaries, shards and sidecars;
   - `var/runs/a3-v2-readiness-07-*`, the run directories and their launch logs;
   - `var/a3-v2-readiness-07/`.
5. **Baselines**: those of the board's §5 after 06a's review. Re-measure all six at `HEAD`,
   serially, with README §7's command.
6. **The index.** Count its rows (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`) and tell the agent the
   number. It is **102** at `118ca94`.
7. **The names are free.** `store show` on both store paths: kind `absent`, no runs. No file exists
   by either name.
8. **The interrupter.** Run both dummy tests at `HEAD` yourself, into the scratchpad:
   `interrupter_dummy_test.sh` and 06a's `interrupter_build_dummy_test.sh`. Both must pass before
   anything is launched on the strength of them.
9. **The decisions.** Put D7-1 and D7-2 to the user, as the prompt's §1.1 states them, with the
   recommendations. Commit the answers (the prompt's §1.1 and the board's §1) before dispatch.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt;
- the campaign README's §3, §4, §5 and §5.1;
- `CLAUDE.md`'s run-registry section;
- logs 03, 04, 05 and 06a;
- `HEAD`, the baselines, the index count and the user's decisions;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, every check and the suites. Its log is at
  `logs/07-rehearse-the-a3-v2-launch.md`, in README §5.1's form, with the prompt's §7 additions.
- **The hand-off.**
  - After each launch: one check, then a report giving the run id, the pid and the interrupter's
    pid if there is one, then the end of its turn.
  - You resume it with the run's final `status.json`, the last lines of its launch log, and the
    wait's verdict.
  - It never waits, polls or sleeps on a build itself.
- **The files it may change** are exactly the prompt's §7 list: new files only, plus the log, the
  board and the index. **No tracked file is modified before its commit.**
- **Stop and ask** on any of the prompt's §5 conditions. A stop leaves the stores, the run
  directories and the tree as they are: no relaunch, no deletion, no fix.

### 2.1 The one wait per launch

When the agent reports a launch, run **one** background command that ends when that run is no
longer alive, or after the cap:

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
    time.sleep(60)
print("heartbeat went stale at least once" if seen_stale else "heartbeat never stale")
EOF
```

- It identifies the run by its pid and heartbeat, never by a command-line match (`CLAUDE.md`
  rule 2). It exits when the build exits, so it cannot outlive what it waits on. On `CAP` it leaves
  the build running.
- When it ends, send the agent the run's final `status.json`, the launch log's last lines, and the
  verdict line, and nothing else.
  - On `ENDED`, it carries on.
  - On `CAP`, the prompt's two-hour stop is met. Tell the agent, and ask the user before anything
    else happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — eleven checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the prompt's §7 files.
   - `git diff HEAD~1 HEAD --diff-filter=M --name-only` is the board and `docs/OPEN_ISSUES.md`
     alone: no tracked code, test or tool modified.
   - The diff over every factory, `ComputeTargets/`, `Datastore/`, `main.py`, `tools/`,
     `RunRegistry/`, `config/`, both scoped drivers, `quadsource_atol_sweep.py`, `extract_*.py`,
     `docs/datastore-integrity-verification/` and `orchestrator/` is empty.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log` and launch log under
   `var/runs/a3-v2-readiness-07-*` yourself.
   - States: `reference` and the last resumed launch are `done`/0. The interrupted ones are
     `killed`/143, each with `fingerprint` and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's line and no traceback. No `*** SIGTERM received`
     banner appears.
   - Each manifest:
     - `git_head` is the dispatch `HEAD`, and `script_sha256` is the same in all;
     - `argv` holds `--build --rehearsal`, with `--resume` only on the resumed launches after the
       first;
     - `scope` is the rehearsal text recorded in §0;
     - `results` names the right store.
   - Where `git_dirty` is true, the log's `git status --short` at that launch shows only untracked
     new files.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on.
   - No process the agent started is alive: neither interrupter, no Ray process of this
     repository. ChamPBH's processes have the pids and start times of §1 item 3.
3. **(R2) Nothing stored was recomputed.**
   - Parse the counts from every `stdout.log` yourself with `stage_counts.py`, and compare them with
     the log's tables.
   - Every stage completed before an interruption computes 0 on the next launch.
   - Summed over the resumed launches, less the pruned rows, each stage stores what `reference`
     stored. Any excess equals the rows in flight at a `SIGTERM`, as the log says.
4. **(R2) The prune.** The log's serials of `validated = 0` rows after each killed launch are absent
   from the store now, read `mode=ro`. `read_inventory` reports no `duplicate`.
5. **(R2) The fingerprints.** Fingerprint both closed stores yourself with `fingerprint_store`
   (read-only):
   - `compare_fingerprints` gives nothing;
   - both are at format 6;
   - each equals its sidecar's.

   Compare the per-class counts with §0's table, and read the log's account of any difference.
6. **(R3) Integrity, independently.** On every shard of both stores, opened `mode=ro`:
   - `integrity_check` is `ok`, and `foreign_key_check` is empty;
   - one `version` row, label `2025.1.1`, the same serial on every shard;
   - `replicated_serial_census.py` finds no divergence;
   - `read_inventory` has no problem in any class;
   - the primary's `replication_in_flight` is empty;
   - no `-journal`, `-wal` or `-shm` file is beside any of them.
7. **(R4) The readers.**
   - Read each reader's log under `var/a3-v2-readiness-07/extract/`: exit 0, its run selection,
     what it read, and no `ReadOnlyMiss` or `ReadOnlyWrite`.
   - Compare the log's hashes before and after each reader. Every file of the resumed store is
     byte-identical, and its mtime is unchanged.
   - The QSI extractor's count of rows read equals the store's `QuadSourceIntegral` count.
   - The run without `--models` printed the skip line and exited 0.
   - Run one reader yourself, `extract_TkWKB_data.py --ray-address local --models LambdaCDM`, into
     the scratchpad. Hash the store before and after.
8. **A3 v2's guard.** The log's foreground `--build` refused, naming the existing files. The store's
   hashes are unchanged, and no manifest was written for it.
9. **The records.**
   - The board: the §1 row for 07, L1–L3, §3.2 for anything opened, and §5.16.
   - The header reads "9 of 9 prompts landed. The campaign's work is done; it is closed by the user
     after the orchestrator's review of 07", and does not say it is closed.
   - The index: anything opened added, and the count right by counting.
   - The verification record quotes every command verbatim, and its numbers match the run
     directories. Its A3 v2 checklist names every issue open on this board, and supersedes
     `docs/datastore-integrity-verification.md` §2.6 without editing it.
10. **`var/` and the tree.**
    - Every file in §1's snapshot is byte-identical, and every new path is on §1's list.
    - `main.py` and `quadsource_atol_sweep.py` are byte-identical to `HEAD~1`.
11. **Suites and `black`.**
    - All six run serially in the checkout, every count unchanged from §1 item 5. Treat load
      failures as §0 says.
    - `black --check` is clean on every Python file added.

## 4. After the review

- **Record it on this board**, after the 06a review and in its form. Fix small residue in a
  follow-up commit of its own: "this commit" → the SHA, README §2's status cell for 07, the
  orchestrator notes list (`prompt-07.md` "used").
- **Then report to the user.** Say:
  - whether R1–R4 hold, row by row against §0's table of 2026-10-01;
  - what the run found that no stand-in could, and every issue it opened;
  - the A3 v2 checklist, and what the rehearsal could not exercise.
- **The user decides two things:**
  - whether the campaign closes;
  - D7-2's question, whether the two stores are retired. `store retire` is theirs to run. Tell
    them what each retirement's report should list since 06: no earlier tombstone, and for
    `a3-v2-readiness-07-resumed` nothing unless a sidecar names it. The four older tombstones keep
    the false matches they recorded.
- **On the user's word, write the closure** in its own commit, as `datastore-integrity` did after its
  prompt 10:
  - README's header paragraph and §2 row;
  - the board's header status, now "closed";
  - one sentence in the index's §1.17.

  Issues still open stay on this board's §3.2, each with its index row.
