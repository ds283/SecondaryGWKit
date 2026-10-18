# Orchestrator — prompt 03, rehearse through `v0.2.0` (R3), and the hard-kill route

Read [`../README.md`](../README.md) first: §0.1, §2.2 (03's charter), §3, §4, **§4.1**, §5 (rules
8–11, 14, 15) and §6 (U2, U3, C6–C8). Read also `CLAUDE.md`'s "Long-running jobs — the run
registry" (rule 5 as 02 wrote it), the packages section and "The shared rehearsal lock", whose
rules bind you as much as the agent. Then `runregistry-extraction`'s
[`orchestrator/prompt-08.md`](../../runregistry-extraction/orchestrator/prompt-08.md), the note for
R2, whose protocol, wait and review this note carries over except where §0 below says otherwise,
and this board's §5.1–§5.2. The conventions are [`prompt-01.md`](prompt-01.md) §0's and 08's
note's, carried over; what differs is said here.

**You do not write code.** You may:
- run the one wait per **build** launch that §2.1 describes, six in all;
- read R3's stores read-only once the agent's last check has run, fingerprint them, and run the
  census, `store_soundness.py`, `compare_stores.py` and `compare_with_recorded.py` on them;
- read every run directory, and hash the files under `var/datastores/`;
- run `tools.rehearsal_lock status` and `ray-processes`, never `acquire`, `set-pid` or `release`
  while the agent holds the lock;
- run the suites, and a probe or comparison script from the session scratchpad, never committing
  one;
- fix small residue in a follow-up commit of your own (§4).

**While a build may yet be launched you write nothing in the repository**, outside `var/` and the
session scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop. This note is committed before dispatch for that reason.

**The prompt:** [`03-rehearse-through-v0.2.0.md`](../03-rehearse-through-v0.2.0.md)
**Board items:** L1–L4 (R3), H1, H2 · **Gates it decides:** R3, HK · **Closes:**
`[a3-v2-a-hard-killed-run-blocks-its-store]` · **Opens:** anything found · **Model:** Opus, as the
prompt recommends.

**Gate:**
- 02 landed (`2818780`) and was reviewed (`b84c53d`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (anything outside
  `prompts/` and `docs/`) and no script under `docs/`. If a later commit touches either, re-check
  every fact below that it could move, and hand the agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive or the rehearsal lock is held.

## 0. What makes this prompt unusual

**It is R2 again, through `runregistrykit` 0.2.0, and then the first hard kill of a rehearsal
build.** Between R2's tree (`b9daf44`) and the dispatch `HEAD`, what a build runs changed in two
places: the `runregistrykit` pin (01, `0a27b0c`), and `main.py`'s and the six readers'
`--ray-address` default (`359f3fd`), which no rehearsal reaches (README §0.1). 02's changes are to
`CLAUDE.md`, the interrupter, its dummy test and a new test module. The launcher,
`docs/handover/quadsource_atol_sweep.py`, is unchanged since `7af17bd`: its hash is R2's,
`2a985303…`. **So a miss on the four-launch stores is the pin's** (README §5 rule 11), and a miss on
the `hardkill` store alone is the hard kill's.

**It decides two gates and closes one issue.** R3 (the four launches) and HK (the hard-kill pair),
and `[a3-v2-a-hard-killed-run-blocks-its-store]` on the `handover` board, if acceptance 3 holds.

**It runs for about an hour, and no conversation may own it.** As at R2: the agent launches each
build detached, through the registry, checks once, and ends its turn. You run **one** wait per
build launch (§2.1) and resume it when the driver's pid is dead. Six builds, six waits. The refused
resume is not a build: its check is the agent's own (correction 11).

**The hard kill is the one step nobody has run.** Two things may happen that no rehearsal has seen,
and each has a route below rather than a guess:
- **Ray's processes may outlive a driver that had no chance to shut them down.** On macOS Ray has no
  parent-death signal for the head node's processes. U3 governs: wait by pid for ten minutes after
  the driver's death, then stop and ask (correction 10).
- **A shard may be left with a hot rollback journal**, if a `Datastore` actor died inside a write
  transaction. That changes which read-only tools can read the store before the resume
  (corrections 7–9).

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `b84c53d`.
Each correction below governs, and the agent logs it as an `IMPLEMENTATION CHOICE` citing this note,
or `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.

1. **08's corrections carry over**, as 08's log records them, with R2 in place of R1 and
   `var/rrk-v020-adoption-03/` in place of `var/runregistry-extraction-08/`:
   - `prune_serials.py` is not run; `prune_check.py` runs on R3's own `closed_store_state.py`
     output;
   - the six scripts of `docs/dk-and-rpk-migration/rehearsal/` and `run_readers.sh` are run as they
     stand; the other tools run from where R2 ran them (verification §3.1–§3.4 there), none copied;
   - the pre-launch Ray check is the helper's `ray-processes`, exit 0;
   - **`packages` is compared as parsed JSON, not in order.** The prompt's §3 says "in this order";
     every record is written with `sort_keys`, so the keys on disk are alphabetical whatever order
     the code built them in. `STRUCTURALLY REQUIRED`.
   - `fingerprint_both_ways.py` and RunRegistryKit's `measure_client_registry.py` are **not** run
     (prompt §1; no G2 here).
   - **A3 v2's guard** (verification §3.2.6 there) runs as at R2, after `resumed-c` and the
     readers, on the `resumed` store, with no `--register`; it must exit 1 from `:537` and create
     nothing. Its output is `guard_03.txt`.
2. **The `resumed-a` point.** Hold **about 40 % of R3's own `reference` `GkSource` stage time**,
   rounded to whole seconds and not below 10 s. Pattern
   `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$'`, `--interval 5`. R2's stage took 27.4 s
   (hold 11 s); R1's 102.9 s (41 s). If `resumed-a` stores no `GkSource` row, that is recorded, not
   a stop. If the stage completes during the hold and the build runs on, that is a missed
   interruption, and a stop.
3. **The `resumed-b` point, and `hardkill-a`'s.** Pattern `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`,
   not anchored at its end, `--interval 5`, `--hold` about a third of R3's `reference` QSI stage
   **and at most 250 s**, inside the queue's 300 s minimum notify interval (`main.py:119`). R2's
   stage took 119.8 s (hold 40 s). **`hardkill-a` uses `resumed-b`'s pattern and hold exactly**,
   with `--signal KILL` added, and nothing else changed.
4. **A progress line may or may not appear.** Whether one does is the 300 s threshold and the load,
   not a finding. R2's runs printed none.
5. **No new line naming `runregistrykit` is expected in a build log.** The launcher is R2's; its
   banner's `**   list it with: ./venv/bin/python -m runregistrykit list` is in R2's logs too. Any
   other line naming `runregistrykit` or `datastorekit` outside a path is a finding to explain.
   R2's raylets warned every 10 s that the temp volume was over 95 % full; `var/`'s volume is now
   93 % used (73 GB free), so those lines may be absent. Either way, say which.
6. **The refused resume imports Ray; it must not start it.** The prompt's §2.5 says `begin` "must
   raise before Ray is imported". `run_build` imports `ray` at its first line
   (`quadsource_atol_sweep.py:531`), before the `--resume` pre-check (`:544`, after which `:545`
   prints `** quadsource_atol_sweep --build: resuming into <name>`) and before `begin` (`:556`). Importing
   Ray starts no process. What must not happen is `ray.init`: the launch log must hold no
   `bootstrapping a local Ray instance` line, and `ray-processes` must list nothing after the `sh`
   has exited. `STRUCTURALLY REQUIRED`.
7. **Every file of each store is hashed, journals included.** Where the prompt says "the primary
   and every shard" (§2.4, §2.5), record every file under `var/datastores/` whose name begins
   `rrk-v020-adoption-03-hardkill`, with SHA-256, size and `mtime_ns`, and name any `-journal`,
   `-wal` or `-shm` among them. The `.manifest.json` sidecar is among them once it exists. The
   whole-directory listing before the first launch and after the last step is as R2's (§3.9 there).
8. **`closed_store_state.py` runs after `hardkill-a`**, which the prompt's §2 omits but §2.9 needs:
   `prune_check.py` reads the state it saved. Run it at §2.4, after the Ray wait and before the
   refused resume, into `var/rrk-v020-adoption-03/`, and again after `hardkill-b`, as after every
   R2 launch. It lists any journal beside each file **before** it reads them.
   - If a shard has a hot journal, its `mode=ro` read of that shard fails with
     `sqlite3.OperationalError: attempt to write a readonly database`. That is SQLite's rule, not
     the script's defect: **record it, not a stop**, and do not touch the journal.
     `prune_check.py` for the `hardkill` store then checks what could be read, and the log says
     which shard it could not, with `hardkill-b`'s prune lines as the evidence for that one.
   - Every file is byte-identical across the read, as at R2.
9. **A hot journal is rolled back by the resume, and by nothing else.** The facts are below. No
   command of this prompt removes, renames, copies or opens read-write any file of the `hardkill`
   store; `hardkill-b`'s open is the only read-write open. **One stop is added:** if a `-journal`
   lies beside the `hardkill` **primary** after the Ray wait, do not launch the refused resume, and
   stop. The pre-check (`assert_store_is_self_consistent`, `:669`) reads the primary `mode=ro`, so
   it would fail before `begin` is reached, and §2.5 would show nothing about `begin`. A journal
   beside a shard is not a stop: the pre-check reads the primary alone.
10. **The wait for Ray after `hardkill-a` is the agent's**, bounded, by pid. You resume it once the
    driver's pid is dead (§2.1). It then reads `ray-processes`, and waits for each pid it lists,
    never by pattern, **until the interrupter's `sent SIGKILL` time plus 600 s**. That stamp is the
    driver's death to within a second: `SIGKILL` is not caught. One foreground command, with the
    Bash timeout set to the time remaining (at most 600 000 ms). Record each process's command, its
    parent and when it was gone. **If any is alive at that deadline, stop.** Report each pid, its
    command and its parent. Do not signal it, do not run `ray stop`, and do not release the lock
    (U3).
11. **The refused resume's check is the agent's.** It is not a build, and needs no wait of yours.
    In the one check after launching it, the agent waits for the `sh`'s pid by `kill -0`, bounded
    at 120 s, then records §2.5's evidence in the same turn and carries on to §2.6. It runs no
    `set-pid` for this launch. **If the `sh` is alive at 120 s**, or a run directory has appeared,
    `begin` accepted: that is §2.5's stop. Record the run id, run `set-pid` with its pid, report
    it with its pid, and end the turn. You then run §2.1's wait on it, and ask the user.
12. **The manifest's `host` has no `boot`.** The README's charter (03, acceptance 2) says every
    manifest and status `host` has an `id` and a `boot` that are not `null`. At `v0.2.0` the
    manifest's `host` holds `id`, `name`, `machine`, `system`, `release` and `python`; the
    status's `host` holds `id` and `boot` (release notes §2.1). Check the manifest's `id` and
    `name` against `this_host()`, and the status's `id` and `boot`. `STRUCTURALLY REQUIRED`.
13. **The `end` checks.** `end` writes `state`, `exit_code`, `ended` and **`heartbeat`**, by the
    status writer (release notes §2.2; RunRegistryKit's `test_end.WRITTEN`). "Every key the status
    held before is kept" (§2.6) is checked with those four set aside: every other key present,
    with its value unchanged. `ended.evidence` also holds `pid`, `status_readable` (`true`),
    `heartbeat` (the run's last) and `heartbeat_age`; record them.
14. **The index and its header.** Closing the issue takes `docs/OPEN_ISSUES.md` from 100 to 99, by
    counting (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`). §1.22 is then empty of rows: say so in
    §1.21's form (`:919`, "The campaign has no open issue."), and keep its table out. The header's
    prose says the `handover` board "holds **eight**" and describes this issue as assigned and
    closing at 03: correct each clause the closure makes false, and nothing else. Recount "across
    twenty-one campaigns" and the header's "Two of these hold no open issue" from the index as it
    then stands, and correct them if they move.
15. **The verification document's header** names R3 (§1) and the hard-kill route (§2), says it is
    created by 03 at the dispatch `HEAD`, and that 04 adds §3, the close. §1 has R2's §3 shape:
    §1.1–§1.9 as there, without §3.10 (W1) and §3.11 (G2), which R3 does not repeat. §2 has a
    subsection for each of the prompt's §2.1–§2.9 and one for Ray after the kill.
16. **The board's header** says 3 of 4 landed once the agent commits; README §2's row and
    `prompts/INDEX.md` are the orchestrator's (§4).

**What moves on purpose:**
- Added: `docs/rrk-v020-adoption-verification.md`; `docs/rrk-v020-adoption/rehearsal/*_03*`, with
  `lock_03.txt`, `hardkill_03.txt`, `fingerprint_against_recorded_03.txt` and `run_records_03.txt`
  among them; the log.
- Modified: this board; `prompts/handover/IMPLEMENTATION_STATE.md` (the entry moves from §3 to §4);
  `docs/OPEN_ISSUES.md` (the row leaves §1.22; the count; the header's clauses).
- **README does not move**, and neither does `prompts/INDEX.md` or `orchestrator/`. No script
  anywhere changes, and no code or test.
- Under `var/`: R3's three stores and their sidecars; six run directories and seven launch logs
  (the refused launch's among them); `var/rrk-v020-adoption-03/`. **Nothing else under `var/`
  changes**: the 21 tombstones and every earlier run directory included.
- Under `~/ScienceData/`: `.rehearsal-lock` created and removed again.

**The facts, checked by the orchestrator on 2026-10-18 at `b84c53d`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `git log 2818780..b84c53d` is `b84c53d` alone, which
  touches `prompts/` only; `git diff --stat 2818780 b84c53d -- . ':!prompts' ':!docs'` is empty.
  The last change to the launcher is `7af17bd`; to `docs/datastore-integrity-verification/` and
  `docs/a3-v2-readiness/`, `2818780`; to `docs/dk-and-rpk-migration/rehearsal/`, earlier.
  `RAY_ADDRESS` unset. 10 CPUs. The one-minute load was 17.6 at 23:17 and 9–10 during the
  baselines. **73 GB free on `var/`'s volume (93 % used)**; a rehearsal's stores are about 140 MB
  each. Uptime 13 days.
- **The lock and Ray.** `tools.rehearsal_lock status`: `lock: free`, exit 0. `ray-processes`: none,
  exit 0. `~/ScienceData/` holds `ChamPBH-rehearsals/` alone.
- **The registry.** `./venv/bin/python -m runregistrykit list`: 45 runs, `43 finished, 2 unknown;
  45 with no host recorded`; none `running` or stale.
- **The host.** `runregistrykit.this_host()`: `id` `8d2b247824882486`, `name`
  `mps026088.phys.susx.ac.uk`, `boot` `376461AF-4985-4F42-8E41-F4E914F5E3E6`, `machine` `arm64`,
  `system` `Darwin`, `release` `27.0.0`, `python` `3.12.15`. The `id` and `boot` are 01's reading:
  no reboot since.
- **The venv, and what every manifest's `packages` should hold.** `pip check` clean.
  `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))` gives
  `{"datastorekit": {"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false, "requested_revision": "v0.4.0", "url": "https://github.com/ds283/DatastoreKit", "version": "0.4.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}, "runregistrykit": {"commit": "0bae3a3c501a8264ee4be5bc24f06422af38720f", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RunRegistryKit", "version": "0.2.0"}}`.
  The `datastorekit` and `raypoolkit` entries are R2's exactly. `TERMINAL_STATES` is
  `('done', 'failed', 'killed', 'lost')`.
- **The config.** `runregistrykit.find_client_root()` from the root: `runregistrykit.toml`,
  `runs_root` `var/runs`, `stores_root` `var/datastores`, `distributions`
  `('datastorekit', 'raypoolkit')`, `stores` `{'fingerprinter': 'config.fingerprint:fingerprinter'}`.
- **The records' fields.** R2's manifests have nineteen keys; R3's should have twenty, the new one
  `host`. R2's `status.json` keys are `exit_code`, `fingerprint`, `fingerprint_sidecar`,
  `heartbeat`, `pid`, `stage`, `state`, `units_done`, `units_total`; R3's should add `host`, and
  `hardkill-a`'s after `end` should add `ended` too.
- **The names are free.** `runregistrykit store show` on
  `var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite`: kind `absent`, problems
  none, no runs naming any, exit 0 each. Nothing under `var/runs/` contains `rrk-v020`;
  `var/rrk-v020-adoption-03/` does not exist.
- **What is under `var/datastores/`**: 21 files, all tombstone sidecars (one under
  `backup-pre-resume-20260921T091011/`), no `.sqlite`. `find var/datastores -type f -print0 | sort
  -z | xargs -0 shasum -a 256 | shasum -a 256` is `7ad566f4342d678a…`.
- **The recorded fingerprints to compare against.** `status.json` of
  `var/runs/runregistry-extraction-08-reference-20261015T235716` and
  `…-08-resumed-c-20261016T000959`: each `done`/0, format 6, `encoding_format` 1,
  `39809dcac5c7b231…`, 21 classes, problems `{}`, no `host`.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` is
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`, R2's. Every R3 manifest's
  `script_sha256` should be that.
- **The hot journal, probed in the scratchpad** with this venv's Python 3.12.15. A child process
  opened a write transaction on a small SQLite file, wrote past its page cache, and was sent
  `SIGKILL`; `wait()` gave `-9`, and a 24 kB `-journal` was left beside the file.
  - A `mode=ro` connection's first read raised `sqlite3.OperationalError: attempt to write a
    readonly database` (`SQLITE_READONLY_ROLLBACK`), and left the journal.
  - A read-write connection's first read rolled it back, returned the committed rows, and removed
    the journal.
  - **DatastoreKit's read-write open does exactly that, by design**
    (`datastorekit/SQL/ShardedPool.py:1269-1285`, step 1 of the check at open). It connects
    read-write to each shard before anything else reads it, and prints
    `>> Rolled back the journal left beside "<path>" before comparing the replicated tables` for
    each shard that had one. If `hardkill-b`'s `stdout.log` has that line, record it as H1's
    evidence: it is what a resume does after a hard kill, and A3 v2 inherits it.
  - Every read-only tool refuses a journal: `datastorekit.store_reader.open_read_only` by its
    presence (`store_reader.py:102-107`), and a plain `mode=ro` read as above. **`store create`
    does not open the primary** (`runregistrykit/stores.py:846`, `_primary_problems` checks only
    that it is a regular file), so §2.7 is not affected by a journal.
- **`begin`'s refusal and `end`'s words**, as 02's probe and test pin them (board §5.2):
  - `RuntimeError: Cannot begin run <new id> with results "<primary>": run <dead id> is running
    (stale, pid <pid>) and names this store by its results. A stale run is ended by a person, with
    `python -m runregistrykit end RUN_ID --reason TEXT`, before another run is begun on its store;
    the registry does not decide that a run is dead. No run directory was created`, **with no
    trailing full stop**. The launcher passes `results=str(database)` resolved, so both runs name
    the same path.
  - `end`, accepted: exit 0, stdout `>> end: <id> is now lost, verified: its pid <pid> does not
    answer kill -0 on this host`.
  - `end` again: exit 1, stderr `!! Cannot end <id>: its status says it has already ended, as
    'lost'; … Nothing was written`.
  - `liveness(status, 900)` on a `running` status whose pid is dead, on this host and boot, is
    `stale` at once: the heartbeat's age does not matter.
- **The interrupter at `b84c53d`.** `--signal {TERM,KILL}`, default `TERM`; with `KILL` its send
  line reads `sent SIGKILL to pid … at stage … (…% >= …%); command …; exiting`, and the `watching`
  line is unchanged. Both dummy tests passed at `2818780` in 02's review, case D `child exit=137`.
  The agent runs both again before the first launch.
- **The stage times.** R2's reference (verification §3.5.5 there), at load 6–19: `GkSource`
  **27.4 s**, QSI **119.8 s**, the build 4 m 33 s. R1's: 102.9 s, 494.9 s, 13 m 57 s.
- **The commands are R2's**, verbatim in verification §3.1, §3.1.2, §3.3 and §3.4 there, with
  `rrk-v020-adoption-03`, `--campaign rrk-v020-adoption --prompt 03` and the store names of the
  prompt's §1. `hardkill-a` and `hardkill-b` are `reference`'s command on the `hardkill` store, with
  `--register hardkill-a`, and `--resume --register hardkill-b`.
- **No Ray process left.** `ray-processes` is the test. A killed build's processes are waited for
  **by pid**, never by pattern; how long each took is recorded beside R2's (none outlived either
  `SIGTERM`ed driver there: §3.2.3).
- **Shell traps.**
  - `stat` here is GNU coreutils: use `/usr/bin/stat -f '%m %N'` or Python's `os.stat`.
  - `grep` may wrap `ugrep`: use `git grep` or Python for any regex the log relies on.
  - The shell is `zsh`: an unquoted `$var` holding spaces is **not** split into words (use
    `${=var}` or an array), a bare word beginning `=` is command expansion, and an unmatched glob
    is an error, not a literal (`setopt null_glob`, or Python).
- **Expected counts**, at `b84c53d`, serially, load 8–12 (one-minute):

  | Root | Tests | Result | Wall |
  |---|---|---|---|
  | AdaptiveLevin | 32 | OK | 1 s |
  | CosmologyModels | 39 | OK | 2 s |
  | ObjectFactories | 279 | OK | 14 s |
  | tools | 91 | OK | 10 s |
  | LiouvilleGreen | 148 | OK (1 skipped) | 24 s |
  | RunDrivers | 103 | OK | 6 s |
  | ComputeTargets | 700 | OK | 146 s |

  Every count is 02's. A failure of one of README §8's four is re-run alone, not a stop.
- **The index:** 100.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `reference`, `resumed-c` and `hardkill-b` `done`/0;
  `resumed-a` and `resumed-b` `killed`/143; `hardkill-a` `running` until `end`, then `lost`, with
  `exit_code` `null`. Each launch log ends with its driver exit status: 0, 143, 143, 0, 137, a
  non-zero for the refused launch, 0.
- **(R2) R3 is R2.** Both four-launch stores equal each other, `39809dca…`, their sidecars, and
  R2's two recorded fingerprints. Nothing stored was recomputed; the prune removed exactly the
  unvalidated rows.
- **(R3) The stores are sound**, by `store_soundness.py` and the census, all three.
- **(R4) Every reader works, and none touches the store.**
- **(R5) The manifests say what ran.** Six manifests: every `packages` §0's record as JSON, every
  `git_dirty` `false`, every `git_head` the dispatch `HEAD`, every `script_sha256` `2a985303…`, and
  every `host` this machine's (correction 12); every status `host` this machine's `id` and `boot`.
- **(R6) L4 equals R2's.**
- **(H1) The hard-kill route holds on the real launcher**, step by step: the kill; stale `!!`;
  `begin` refuses with nothing created and nothing changed; `end` verified, then refused; the
  sidecar; `hardkill-b` `done` with `fingerprint_sidecar` `written`; the `hardkill` store
  `39809dca…`.
- **(H2) What Ray did after the `SIGKILL`**, each process to its end, within ten minutes, or the
  stop.
- **(R7) The lock.** Taken before `reference`, `pid` set at each launch that began a run, held
  through the readers and the hard-kill pair, released; free now.
- **(R8) Nothing else under `var/` moved.** The 21 tombstones keep their hashes; `list` shows 51.
- **(R9) The issue closes in the campaign's form**, on both boards and in the index.

**Conventions.** 08's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §8's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended and the lock is released**.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/rrk-v020-adoption-03/` (`CLAUDE.md` rule 6). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched.
- **The venv.** The agent installs and uninstalls nothing.
- **DatastoreKit, RunRegistryKit and RayPoolKit.** Read only, through `git show v<tag>:<path>` or
  the installed package; nothing changed there.
- **The lock.** Every helper command and its exit status in `lock_03.txt`. Never wait for, retry or
  remove a lock; exit 3 or 4 at `acquire`, or a refusal naming another owner, is a stop.
- **Ray addresses.** The driver forces `address="local"`; `run_readers.sh` passes `local`. Nothing
  runs `main.py` or an `extract_*.py` at its default. **`ray stop` is never run.**
- **Never touch a process you did not start**, and never signal one you did, but through the
  interrupter.
- **`end` is run on `hardkill-a` alone**, once accepted and once refused. `Run.finish` is called by
  hand on no run (README §5 rule 15).
- **Stop conditions are real.** A stop leaves the stores, the run directories and the tree as they
  are: no relaunch, no deletion, no fix. It goes to the user through the orchestrator, and the
  agent resumes with `SendMessage`. If it stops while holding the lock, it releases once no Ray
  process of this repository is alive, or says it cannot — **except under correction 10**, where
  the lock stays held for the user.
- **Usage limits.** A usage stop does not stop a detached build. Resume the agent with
  `SendMessage`, saying exactly where it stopped. If it commits but sends no report, review from the
  commit, its log and its outputs.

## 1. Before you dispatch

Done on 2026-10-18 at `b84c53d`, before this note was committed; the results are §0's facts.
Repeat steps 1–3 at the dispatch `HEAD` if anything lands in between.

1. **The tree.** Branch and `HEAD`; `git status --short` empty; `git worktree list` this checkout
   alone; `git diff --stat 2818780 HEAD -- . ':!prompts' ':!docs'` empty, and no `docs/` script
   changed since 02.
2. **The registry.** Nothing `running` or stale; 45 runs.
3. **The lock and Ray.** `status`: free; `ray-processes`: none. `RAY_ADDRESS` unset. Load and free
   space as §0.
4. **The venv and the host.** `package_provenance()` and `this_host()` give §0's records; `pip
   check` is clean.
5. **Baselines**, as §0's table.
6. **The index.** 100.
7. **The names are free**, and `var/datastores/` holds the 21 files, hashing as §0.
8. **The hot-journal probe**, as §0, in the scratchpad's `orch03/hj/`.
9. **The `var/` marker**: immediately before dispatch, `touch` a marker in the scratchpad.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, and through it `runregistry-extraction`'s 08, 04 and 02, and `dk-and-rpk-migration`'s
  08;
- the campaign README and this board;
- `CLAUDE.md`'s three sections named above;
- 08's log, 01's and 02's logs' hand-offs, and `runregistry-extraction` verification §3;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, the readers, the hard-kill pair, the lock's
  release and the suites. Its log is `logs/03-rehearse-through-v0.2.0.md`, in README §5.1's form.
- **The hand-off.**
  - After each build launch: one check, including `set-pid`, then a report giving the run id, the
    Python pid, the `sh` pid, the interrupter's pid and arguments if there is one, and the lock's
    `status`, then the end of its turn.
  - You resume it with the run's final `status.json` (less the fingerprint's per-class detail), the
    last lines of its launch log, and the wait's verdict.
  - It never waits, polls or sleeps on a build itself. Its bounded waits are three: by pid for a
    killed build's Ray processes before the next step; correction 10's, after `hardkill-a`; and
    correction 11's, on the refused launch's `sh`.
- **The files it may change** are the prompt's §7 list, nothing else. **It touches none of them, and
  creates nothing in the repository outside `var/`, until `hardkill-b` has ended**, so that every
  manifest stamps `git_dirty: false`.
- **It retires no store.** R3's three are the user's, after 04's review (README §3).
- **Stop and ask** on any of the prompt's §5 conditions, and on corrections 9, 10 and 11's. A
  fingerprint miss is the one R3 exists for: record, change nothing, stop. **A statistics count that
  differs, with the fingerprint holding, is not a stop**: it is a finding, recorded, explained if it
  can be, and opened as an issue.

### 2.1 The one wait per build launch

When the agent reports a build launch, run **one** background command (`run_in_background`) that
ends when that run's driver pid is dead, or after the cap. It ends on the pid's death, **not** on
`state`: a killed run's `status.json` says `killed` some seconds before its fingerprint and final
heartbeat are written, and `hardkill-a`'s says `running` for ever.

```bash
./venv/bin/python - <<'EOF'
import json, time
import runregistrykit
run_dir, pid = "var/runs/<run id>", <python pid>
status = runregistrykit.read_json(f"{run_dir}/status.json") or {}
assert status.get("pid") == pid, ("status.json names another pid", status.get("pid"))
start, cap, seen_stale, capped = time.time(), 2 * 3600, False, False
while runregistrykit.pid_alive(pid):
    status = runregistrykit.read_json(f"{run_dir}/status.json") or {}
    seen_stale = seen_stale or runregistrykit.liveness(status, 900) == "stale"
    if time.time() - start > cap:
        capped = True
        break
    time.sleep(30)
died = time.strftime("%H:%M:%S")
if not capped:
    time.sleep(10)  # the sh wrapper's exit-status line
status = runregistrykit.read_json(f"{run_dir}/status.json") or {}
fp = status.get("fingerprint") or {}
brief = {k: v for k, v in status.items() if k != "fingerprint"}
brief["fingerprint"] = {k: fp.get(k) for k in ("fingerprint_format", "encoding_format", "digest", "problems")}
brief["fingerprint"]["classes"] = len(fp.get("classes") or {})
print("CAP" if capped else "ENDED", status.get("state"), "pid gone by", died, json.dumps(brief))
print("heartbeat went stale at least once while the pid was alive" if seen_stale else "heartbeat never stale while the pid was alive")
EOF
```

- It identifies the run by its pid and heartbeat, never by a command-line match (`CLAUDE.md`
  rule 2). It exits when the driver exits. On `CAP` it leaves the build running.
- When it ends, send the agent its printed lines and the launch log's last lines, and nothing else.
  - On `ENDED`, it carries on. For `hardkill-a`, `ENDED running` is the expected verdict.
  - On `CAP`, the two-hour stop is met. Tell the agent, and ask the user before anything else
    happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — sixteen checks

1. **Scope.**
   - `git diff --name-status HEAD~1 HEAD`: `A` for the verification document, the log and the
     `*_03*` outputs; `M` for this board, the `handover` board and `docs/OPEN_ISSUES.md`. README,
     `prompts/INDEX.md` and `orchestrator/` are not modified. No code, test, tool or script.
   - The `handover` board's diff moves the one entry from §3 to §4, with a closing paragraph, and
     nothing else. The index's diff is correction 14's.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/rrk-v020-adoption-03-*` yourself.
   - The states and exit codes of R1 above; each fingerprint carrying `encoding_format` 1 and no
     `fingerprint_error`; `hardkill-a` with none, since `finish` never ran.
   - Each launch log ends with its matching driver exit status.
   - Each `SIGTERM`ed `stderr.log` has the handler's lines and no traceback; `hardkill-a`'s has no
     handler line at all.
   - Each interrupter log shows one send, `SIGTERM` twice and `SIGKILL` once, with the pid and the
     stage it acted on; the holds are corrections 2 and 3's, from R3's own reference.
   - No leftover `status.json.*.tmp` in any killed run's directory, or each one named in the log.
   - No process the agent started is alive, and `ray-processes` lists nothing.
3. **(R5) The manifests and statuses.** For each of the six: `git_head` the dispatch `HEAD`,
   `git_dirty` `false`; `packages` §0's record as JSON; `script_sha256` `2a985303…`; twenty keys,
   `host` among them, its `id` and `name` §0's; `argv` holding `--build --rehearsal`, with
   `--resume` on `resumed-b`, `resumed-c` and `hardkill-b` only; `scope` beginning `REHEARSAL
   build, not A3 v2:`; `results` the right store; each `--purpose` naming R3, the tree's commit,
   `runregistrykit` 0.2.0 at `0bae3a3`, and `39809dca` with R2's recording run. Every status's
   `host` is `{"id": "8d2b247824882486", "boot": "376461AF-…"}`.
4. **(R2) Nothing stored was recomputed.** Run `stage_counts.py` on the six run directories
   yourself, and compare it with the log's table and with `stage_counts_08.txt`.
   - The `reference` column equals R2's, stage by stage.
   - Every stage completed before an interruption computes 0 on the next launch, in both chains.
   - The resumed chain's and the hard-kill chain's identities: `GkSource` = 291 − stored + pruned;
     QSI = 864 − stored by the killed launch (+ pruned, if any). The totals per class over each
     chain equal the reference's.
5. **(R2) The prune.** Run `prune_check.py` on the agent's saved `closed_store_state.py` output after
   `resumed-a`, and after `hardkill-a` where it could be read (correction 8): every listed serial
   absent now, or reused by a later validated row, read `mode=ro`. The prune lines of `resumed-b`
   and `hardkill-b` count the same rows, Ray's folding included. If a shard had a journal, the
   rollback line is in `hardkill-b`'s `stdout.log`.
6. **(R2) The fingerprints.** Fingerprint all three closed stores yourself, read-only:
   - `compare_stores.py` on each pair: equal, each equal to its sidecar;
   - `compare_with_recorded.py` on all three primaries `--against` R3's `reference` and `resumed-c`
     and R2's two run directories: 0 differences each;
   - `runregistrykit store fingerprint` on each: format 6, `encoding_format` 1, `39809dca…`, 21
     classes, `recorded: matches`.
7. **(R3) Integrity.** `store_soundness.py` and `replicated_serial_census.py` on all three primaries
   yourself: `integrity_check` `ok`, `foreign_key_check` empty, one `version` row per shard, no
   census divergence, no inventory problem, **no journal file now**.
8. **(R4) The readers.** Each reader's log under `var/rrk-v020-adoption-03/extract/`: exit 0, no
   `ReadOnlyMiss`, `ReadOnlyWrite`, `ModuleNotFoundError`, unpickling error or traceback; hashes and
   mtimes equal around each; the `nomodels extract_Gk_data` run printed the skip line. Each log
   compared with R2's same reader as line kinds. Run no reader yourself unless the lock is free and
   you take it for the run, and release it after.
9. **(R6) L4.** `policy_statistics.py` on R3's `reference`, `resumed-b` and `resumed-c` yourself: 14
   lines each, the counts R2's (291; 109 / 26 (25 complete, 1 minimal) / 156), equal line for line
   or sorted, as the log says. The `** ` banner lines of each run, in order, equal R2's same slug
   with numbers and prefix stripped. The chain's identities in the log beside R2's §3.5.3.
10. **(R6) The printed lines and the log differences.** For each slug, against R2's: completion,
    `Queue summary:` and `(no compute items were evaluated` counts; any progress line; each final
    `stage`. Every kind of line in one and not the other is in the log, explained. Search all the
    build logs and reader logs yourself for `runregistrykit`, `datastorekit` outside paths,
    `DeprecationWarning`, `ConfigError`, `Traceback`, `ModuleNotFoundError`. The refused launch's
    log holds one traceback, `begin`'s, and nothing from Ray (correction 6).
11. **(H1) The hard-kill route.** `hardkill_03.txt` read whole, against the prompt's §2 and
    corrections 6–13, each step against the run directory:
    - `hardkill-a`'s `status.json`: `lost`, `exit_code` `null`; `ended` as correction 13, `at`
      after the interrupter's `sent SIGKILL`, `git_head` the dispatch `HEAD`; every other key
      unchanged from the agent's saved pre-`end` copy; the manifest's SHA-256 as recorded before;
    - `var/runs/`'s count and the `hardkill` files' hashes equal at 2.4 and 2.5;
    - the refused launch's log against §0's refusal text, word for word, ids aside;
    - the second `end`'s exit 1 and its stderr; `status.json` byte-identical around it;
    - `hardkill-b`'s `fingerprint_sidecar` `written`, and its sidecar's `history`.
12. **(H2) Ray after the `SIGKILL`.** The agent's `ray-processes` record and its per-pid times.
    Each pid it names is gone now. If none was listed, the log says so and gives how soon after the
    kill it looked.
13. **(R7) The lock.** `lock_03.txt`: `acquire` exit 0 before `reference`'s manifest was created;
    `set-pid` exit 0 at each of the six build launches, with that run's driver pid, and none for
    the refused launch; `release` exit 0 after §2.9's checks. Now `status`: free; `ray-processes`:
    none. `~/ScienceData/` holds what it held before dispatch.
14. **(R8) `var/`.** The file list under `var/datastores/`, with hashes, before the agent's first
    launch and now: the 21 unchanged, the only additions R3's fifteen store files and three
    sidecars. `find var -newer <marker>` lists only R3's runs, launch logs, stores, sidecars and
    working directory. `runregistrykit list`: 51 runs, 49 finished and 2 `unknown`, nothing
    `running` or stale, "45 with no host recorded"; `run_records_03.txt` agrees.
15. **(R9) The records.** The board: the header (3 of 4 landed), the §1 row for 03, L1–L4 (R3), H1,
    H2 ✅, the §2 rows R3 and HK, the issue moved from §3 to §4, `### 5.3 After prompt 03`. The
    `handover` board's §4 entry cites this commit, verification §2, and `v0.2.0`'s `end` and
    `begin`. The index: the row gone, the count 99 by counting, correction 14's clauses. The
    verification document has correction 15's shape, every command verbatim, its numbers matching
    the run directories. The log classifies §0's corrections and has every §5.1 heading.
16. **The suites.** All seven roots, serially, at the commit, every count as §0.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 03* paragraph in §5.3, in 02's
  form. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board, in the log's header, on the `handover` board and in the
    verification document;
  - the board's §2 rows, if the agent left either ⬜ with every part holding;
  - this note added to the board's list of notes, marked "used for 03";
  - README §2's row for 03; `prompts/INDEX.md`'s row: 3 of 4 landed, R3 and HK hold.
- Anything the review finds that is not residue is opened on the board's §3 and in the index.
- **Report to the user:** R1–R9, H1 and H2, each in a line; the fingerprints against `39809dca…`
  and R2's records; where each interruption fell and what each resume pruned and recomputed; what
  the `SIGKILL` left (Ray's processes, any journal, any temporary status file); the refusal, `end`
  and the resume; every log difference from R2; the lock; the issue closed.
- **Then 04**, the close. Its prompt is written after this review (README §2.1), and states what A3
  v2 inherits from H1 and H2. R3's three stores stay until the user decides, after 04's review.
