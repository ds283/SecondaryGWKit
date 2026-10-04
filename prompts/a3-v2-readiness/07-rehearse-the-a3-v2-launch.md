# Prompt 07 — rehearse the A3 v2 launch on a throwaway store

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-03, after 06 landed and was reviewed, against what 01–06 shipped and 06a's
interface (README §4). **Gate:** 06a landed and reviewed (board §1); `git status` is clean; nothing
is `running` in the registry. **The user decided D7-1 and D7-2 on 2026-10-04, both as
recommended** (§1.1; board §1).
**Closes:** nothing by name. It is the campaign's last prompt. **Opens:** anything you find (§6),
**without fixing it**.
**Recommended model:** **Opus**. It is the first time anything that 02–06a shipped runs under Ray on
a real store. The work is in reading what the runs did.

**Read first:**

1. `CLAUDE.md`, the section "Long-running jobs — the run registry", in full. Its rules bind every
   launch in this prompt.
2. [`README.md`](README.md): §0, §1, §3 (this prompt is the one exception to "nothing under
   `var/`"), §4, §5 and §6.
3. The *State handed to the next prompt* and *Observations not acted on* of every log in `logs/`.
   §0's table lists what each left for this prompt.
4. [`prompts/datastore-integrity/10-build-a-throwaway-store-end-to-end.md`](../datastore-integrity/10-build-a-throwaway-store-end-to-end.md)
   and [`docs/datastore-integrity-verification.md`](../../docs/datastore-integrity-verification.md).
   That prompt rehearsed a build on the same scope through `scoped_pipeline_run.py`, before any of
   this campaign's fixes. Its record holds the counts this prompt's stores should reproduce (§2.2),
   and the five findings this campaign exists to fix.
5. `docs/handover/quadsource_atol_sweep.py`: `main`, `run_build`, `build_main_argv`, the rehearsal
   scope 06a added, and `assert_store_is_self_consistent`.
6. `docs/gktk-remedial/scoped_pipeline_run.py`: `install_killed_handler`, `make_killed_handler`,
   `finish_after_exec` and `SignalLatch` (prompt 04).
7. `RunRegistry/__init__.py` (`begin`, `liveness`, `Run.finish(..., fingerprint=True)`) and
   `RunRegistry/stores.py` (`create_sidecar`, `fingerprint_store`, `compare_fingerprints`). Also
   `python -m RunRegistry store --help`.
8. `docs/datastore-integrity-verification/interrupt_when.py` (as 06a left it, reading the driver
   from the run's manifest), `interrupter_dummy_test.sh`, `stage_counts.py` and `compare_stores.py`;
   06a's `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`; and
   `docs/datastore-integrity-audit/replicated_serial_census.py`. Use them as they stand.

---

## 0. What this prompt is

README §0.1 put it last for one reason: **A3 v2 must not be the first thing to exercise the
result.** Prompts 01–06a changed the build's tolerance, how a new store's version row is written,
how readers open a store, what a `SIGTERM` records, whether a resume prunes, and what a retirement
reports. All of it was tested on stand-ins and fixture stores, or by probes, without a real build.
This prompt runs A3 v2's own launcher, `quadsource_atol_sweep.py --build --register`, on a small
scope. It interrupts the build and resumes it with `--build --resume`. Then it reads the store back
with every reader.

**What earlier prompts left for this one.** Each item is checked by the step named.

| From | First exercised here | Checked in |
|---|---|---|
| 01 | A build at `(1e-32, 1e-7)` read back by `extract_QuadSourceIntegral_data.py` at the default pair, which is now the build's | §2.3: the extractor reads every `QuadSourceIntegral` row the store holds, and none is missed |
| 02 | A new store's first open, writing its version row through the recorded write, on real shards under Ray | §2.2: one `version` row per shard, the same serial and label on each; `replication_in_flight` empty after every close |
| 02 (U1) | The prune at open of a replicated class, by the pool, under a `prune` record | §2.2: what each resumed launch's open did. An unvalidated `BackgroundModel` is not expected, so this path may not run (§2.5) |
| 03 | Every reader opening `read_only=True` on a real store, with `--models` | §2.3: each exits 0, and no file of the store changes |
| 04 | `killed`/143 from a registered `--build`, the fingerprint after the pool closes, `ray.shutdown()` and no Ray process left; `address="local"` beside another cluster | §2.1 |
| 05 | `--build --resume` pruning unvalidated rows at startup | §2.2: the rows left unvalidated by the interruption are gone after the next open, and nothing is duplicated |
| 06 | — | §1.5: the stores stay for the user to retire (D7-2). Their reports will list only real references |
| 06a | `--build --rehearsal`, and its manifest's `scope` text; the interrupter on a `--build` | §2.1: each manifest's `scope` and `argv`; §3 and §1.4 |
| all | The fingerprint at format 6 against an uninterrupted reference; the census, `foreign_key_check` and the inventory on stores this tree wrote | §2.2 |

**What this prompt is not.** It changes no production code: no factory, target, `main.py`,
`Datastore/`, `RunRegistry/`, `config/`, `tools/`, driver, `quadsource_atol_sweep.py` or
`extract_*.py`. It changes no test and no verification tool. **It modifies no tracked file** until
its commit, so every launch runs a tree whose `HEAD` holds all the code that ran. Everything it adds
is a verification record.

---

## 1. The stores and the runs

### 1.1 The decisions (the user's, before dispatch)

Each is stated with the recommendation and its reason. **The orchestrator records the user's
decision here, dated, before dispatch.**

**Decided by the user on 2026-10-04, before dispatch, both as recommended:** D7-1, twice (the
`GkSource` stage, then the `QuadSourceIntegral` stage; four launches); D7-2, both stores stay for
the orchestrator's review, and the user then retires them, or not.

- **D7-1 — where to interrupt.** *Recommended:* twice, by `SIGTERM` to the driver, as
  `datastore-integrity` D10-3 did.
  - Once during `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`. On 2026-10-01 that point left 45
    `GkSource` rows stored but unvalidated (verification §2.1.2). It is where prompt 05's prune at
    startup has real rows to remove.
  - Once during `CALCULATE QUADRATIC SOURCE INTEGRALS`, where many work items are in flight in
    `ray.wait`. That is the state prompt 04's handler was built for, and its ~1 s bound was measured
    in.
  - The second interruption also resumes a store that has already been resumed once.
  - *Alternative:* once, in the `GkSource` stage only. That is the charter's minimum. It leaves
    the handler unexercised in the long stage A3 v2 spends most of its time in.
- **D7-2 — the stores afterwards.** *Recommended:* both stay in `var/datastores/`, with their
  sidecars, for the orchestrator's review. The user then retires them, or not. `store retire` is a
  person's act (`CLAUDE.md`), and this prompt never runs it. Since prompt 06, each retirement's
  report will list only sidecars that name that store, a file of its plan, or its own subdirectory.

**Not decisions** (fixed by README §6.1, D1 and D5, and by 06a):
- the scope is `--rehearsal`;
- the pair is `--atol 1e-32 --rtol 1e-7`;
- `--cpus 6`, as A3 v2's default;
- `main.py`'s own `--prune-unvalidated` default, which is on.

### 1.2 Names

- **The stores.** `var/datastores/a3-v2-readiness-07-reference.sqlite` and
  `var/datastores/a3-v2-readiness-07-resumed.sqlite`.
  - Before the first launch, show that nothing exists by either name: no primary, no shard, no
    sidecar, no tombstone (`store show`: kind `absent`, no runs). If anything does, stop.
  - No other store, tombstone or run uses either name (README §3).
- **The runs.** `--campaign a3-v2-readiness --prompt 07`, with the slugs `reference`, `resumed-a`,
  `resumed-b` and `resumed-c`. Under D7-1's alternative there is no `resumed-c`, and `resumed-b`
  runs to `done`. Every `--purpose` is one line a stranger can read.
- **`--job-name a3-v2-readiness-07`** on every launch, for both stores. The two builds must differ
  in nothing but the interruptions.

### 1.3 How each build is launched — `CLAUDE.md`'s rules, with nothing relaxed

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/a3-v2-readiness-07-<store>.sqlite \
    --job-name a3-v2-readiness-07 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "<one line>" --campaign a3-v2-readiness --prompt 07; \
  echo "** driver exit status $?"' \
  > var/runs/a3-v2-readiness-07-<slug>.launch.log 2>&1 &
disown
```

- **`--resume`** is given to `resumed-b` and `resumed-c` only. Without it, `run_build` refuses a
  store that exists, and that refusal is A3 v2's guard (§2.1).
- **The shell wrapper** exists for one reason. The driver is detached, so nothing else can observe
  its exit status, and prompt 04 says a signalled build exits 143. The signal goes to the Python
  pid in `status.json`, never to the shell.
- **The launch log** holds what is printed before the driver registers, and the exit status line.
  It sits beside the run directory, not in it (`store-retirement`'s
  `[00-a-launch-log-lives-outside-its-run-directory]`). Do not work round that.
- **Before each launch:**
  - `python -m RunRegistry list` shows nothing `running`.
  - **No Ray process of this repository is alive.** `pgrep -lf 'gcs_server|raylet|ray::'` may
    list the user's ChamPBH cluster, started from `…/ChamPBH/venv/`. That is the user's, and is
    left alone. Show that no listed process comes from this repository's `venv`. Do not use
    `ray status`: it finds ChamPBH's cluster, and since prompt 04 (K4) the driver starts its own
    instance beside it.
  - `RAY_ADDRESS` is unset.
  - Record the load average and `git status --short`. Only your own new, untracked files may be
    present: under `docs/a3-v2-readiness/rehearsal/`, the verification record and your log. **No
    tracked file may be modified.** The manifest stamps `git_dirty: true` for untracked files too;
    the record says why.
- **After each launch, one check, then end your turn.** In that check:
  - the manifest exists;
  - `status.json` says `running`, with a pid that answers `kill -0`;
  - `RunRegistry list` shows the run.

  Report the run id, the pid, and the interrupter's pid if there is one, to the orchestrator, and
  **end your turn**. The orchestrator waits, with one wait on the run's liveness, and resumes you
  when the run is no longer alive. **Do not poll.** Never wait with `pgrep -f` (`CLAUDE.md` rule 2).
- **Never touch a process you did not start.** The machine may be heavily loaded by another of the
  user's workspaces. That is not a reason to stop.

### 1.4 How a build is interrupted

Use `interrupt_when.py`, as 06a left it, launched detached right after the build it watches, under
the same rules. Its log is `var/runs/<run id>/interrupter.log`.
- **`GkSource`.** That stage prints no intermediate progress line, so interrupt from its banner,
  with `--hold`. On 2026-10-01 the stage took 34 s, and `--hold 8` put the signal 8–23 s into it,
  with 161 of 291 rows stored.
- **`QuadSourceIntegral`.** At this scope the stage is 2 work items, so every progress line reads
  100.00 %, and the first such line is the only point a percentage can pick (log 10, deviation 5).
- **Read the reference's `stdout.log` first**, for both stages' banners and progress lines, and
  choose each point from it. Afterwards, show from the killed launch's last progress line, and from
  the next launch's counts, that the stage had stored some rows and not all.

### 1.5 The order of work

1. **§3 first**: the interrupter's two dummy tests, before any launch.
2. **The reference build**, `reference`, to `done`. Then:
   - `store create --purpose …` on its closed primary;
   - `store fingerprint --write`.
3. **The resumed build.**
   - `resumed-a`, with the interrupter on the `GkSource` stage, until it is killed. Then
     `store create` on the closed primary.
   - `resumed-b`, with `--resume` and the interrupter on the `QuadSourceIntegral` stage, until it is
     killed.
   - `resumed-c`, with `--resume`, to `done`.
4. **Between launches, on the closed store**, read-only, record:
   - the `*-journal`, `*-wal` and `*-shm` files beside the primary and the shards (expected: none);
   - the rows of the primary's `replication_in_flight`, read with `sqlite3` `mode=ro` (expected:
     none);
   - per shard, the rows of every class with a `validated` column that are `validated = 0`: count
     and serials;
   - the launch's `status.json`, and the last lines of its `stderr.log` and launch log.
5. Then §2.2–§2.4.

---

## 2. What to verify

### 2.1 L1 — every launch ends as recorded

- **States.** `reference` and `resumed-c` are `done` with exit code 0. `resumed-a` and `resumed-b`
  are `killed` with exit code **143**. Paste each final `status.json`.
- **Each killed launch, against prompt 04's account:**
  - `stderr.log` holds the handler's line, `** SIGTERM at <time>: recorded killed, exit code 143;
    stopping the pipeline; …`, and no Python traceback. Ray's `*** SIGTERM received` banner does
    not appear.
  - The delay from the interrupter's `sent SIGTERM` line to the handler's line, from the two
    timestamps. Both are whole seconds, so it is bounded to a second; say so. Prompt 04 measured
    0.001–1.006 s.
  - `status.json` has `fingerprint`, and no `fingerprint_error`. For `resumed-b`,
    `fingerprint_sidecar` says it was written, since the sidecar then exists. For `resumed-a` it
    says why not.
  - The launch log ends `** driver exit status 143`.
  - **No Ray process of this repository is left**, by the same listing as before the launch.
- **Each `done` launch:** `status.json` carries its fingerprint, and the launch log ends
  `** driver exit status 0`.
- **The manifests.** For each: `git_head` is the dispatch `HEAD`; `script` is
  `docs/handover/quadsource_atol_sweep.py`, and `script_sha256` is the same in all four; `argv`
  holds `--build --rehearsal`, and `--resume` exactly where §1.3 says; `results` names the right
  store; `scope` is 06a's rehearsal text, naming the pair, the grid and `zend 250`, and not
  `9280`.
- **Every launch's `stdout.log`** shows `substitute_main_py`'s four replacements, one each; the
  rehearsal scope line; `main.py argv = …` with `--zend 250` and no `--no-prune-unvalidated`; and
  the local Ray bootstrap.
- **A3 v2's guard.** Once, after `resumed-c`, run in the foreground with no `--register`:
  ```bash
  PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
      --database var/datastores/a3-v2-readiness-07-resumed.sqlite \
      --job-name a3-v2-readiness-07 --atol 1e-32 --rtol 1e-7
  ```
  It must refuse, naming the files that exist, before Ray starts. Every file of the store must have
  the same SHA-256 before and after. Paste the message.

### 2.2 L2 — the resume is the reference

- **Nothing stored is recomputed.** Tabulate each stage's final lookup / compute / store counts
  from every launch's `stdout.log`, with `stage_counts.py`.
  - Every stage that finished before an interruption computes **zero** on the next launch.
  - The interrupted stage computes exactly what was missing, plus the rows the next open pruned.
    Rows in flight at the `SIGTERM` were never stored, so they may be computed again; say how many.
  - Summed over the resumed launches, less what was pruned, each stage stores what `reference`
    stored.
- **The prune at startup** (prompt 05).
  - The rows that §1.5 item 4 found `validated = 0` after each killed launch are gone after the
    next launch's open: by serial, read `mode=ro`.
  - The next launch's `stdout.log` shows each actor's integrity warning and
    `** Note: these values have been pruned from the datastore.` Ray may fold repeated lines as
    `[repeated Nx across cluster]`. Say what it printed, and count from the store, not the log.
  - **Nothing is duplicated.** `read_inventory` reports no `duplicate` in any class after each
    resumed launch.
  - Say whether the pool pruned anything of a replicated class, from its output and from the
    store. None is expected (§2.5).
- **The version row** (prompt 02). On every shard of both stores, `SELECT serial, label FROM
  version`, `mode=ro`: exactly one row, label `2025.1.1`, the same serial on every shard of a store.
- **The comparison.**
  - Fingerprint both closed stores read-only (`compare_stores.py`). `compare_fingerprints` must
    be **empty**, both at format 6, and each must equal its sidecar's. Paste the per-class counts
    beside each other.
  - **Against `datastore-integrity` 10.** The scope and sampling are the same, so every class's
    count should equal that record's §2.1.5 table. The tolerance pair is different, so the
    digests are not expected to match. A count that differs is a finding (§6), not a stop. Say
    what differs and why.
- **Integrity, both stores, read-only.**
  - `replicated_serial_census.py` on both primaries: every table `OK`, verdict `no divergence`.
    Check its table list against `config/sharding.py`'s `replicated_tables`, and say what it omits
    or adds.
  - On every shard, opened `mode=ro`: `PRAGMA integrity_check` is `ok`, and
    `PRAGMA foreign_key_check` is empty.
  - `read_inventory(primary)` has no problem in any class. Give each class's record count.

### 2.3 L3 — every reader, read-only, on the resumed store

Run each of `extract_Gk_data.py`, `extract_GkWKB_data.py`, `extract_GkSource_data.py`,
`extract_TkWKB_data.py`, `extract_tensor_source_data.py` and `extract_QuadSourceIntegral_data.py`
once on the resumed store, with:
- `--database var/datastores/a3-v2-readiness-07-resumed.sqlite`;
- **`--ray-address local`**. The scripts' default, `auto`, would attach to the user's ChamPBH
  cluster;
- `--models LambdaCDM`;
- `--output var/a3-v2-readiness-07/extract/<script>/`.

Then run one of them again (*your choice*, the quickest) **without** `--models`. Log 03 says it
prints `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` and exits
0.

For each run:
- **It exits 0.** Record what it read: the model, the objects, and the run it selected and why
  (`resolve_run_selection`'s line).
- **No file of the store changes.** Before and after each reader, take the SHA-256 and mtime of the
  primary and every shard. List the directory too: no `-journal`, `-wal` or `-shm` file may appear.
  This is the proof prompt 03 was written for (D3).
- **It refuses nothing.** A `ReadOnlyMiss` or `ReadOnlyWrite` is a stop (§5). It means a reader
  looks up something the build did not write, or tries to write.
- **`extract_QuadSourceIntegral_data.py` reads every row.** Its count of `QuadSourceIntegral` rows
  read must equal the store's (§2.2). This is prompt 01's point: the extractor's default pair is
  now the build's. On 2026-10-01 a store built at `1e-7` would have yielded none.

Afterwards, `store fingerprint` (without `--write`) on the resumed store must equal its sidecar's.

### 2.4 What A3 v2's launch must now pass

Write it in the verification record as a checklist that supersedes
`docs/datastore-integrity-verification.md` §2.6. Leave that section as it is (additive). Point
each item to where this prompt exercised it. Include at least:
- **the command**: §1.3's, without `--rehearsal`, with A3 v2's store name and job name (the
  `handover` campaign's to choose), on the macstudio. It keeps its own registry, so a `list` here
  sees nothing there (README §3);
- the pre-launch checks of §1.3, for that machine;
- `store create` after the first close, and the fingerprint at finish;
- an interruption, recorded `killed`/143 with its fingerprint and no Ray left behind, and a resume
  with `--build --resume`, which prunes at startup;
- after the build: the census, `foreign_key_check`, the inventory, and every reader with
  `--ray-address local`, each leaving every file of the store as it was;
- **what this prompt could not exercise**, and why: the production grid and `--zend 0.1`; the
  macstudio itself; `QCD_Cosmology`; a crash mid-replicated-write; a signal during `ray.init`
  (`[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]`); the pool's prune of an
  unvalidated `BackgroundModel` if §2.2 did not see one; a `SIGKILL` or power loss;
- **every issue still open on this board** (§3.2), with what it means for A3 v2.

### 2.5 What is expected not to happen

Say for each whether it happened:
- **an unvalidated `BackgroundModel`**: it is written in the first seconds of a build, and neither
  interruption is there, so 02's pool-level prune should find nothing to remove;
- **an in-flight record left by a killed launch**: both interruptions are in sharded stages, which
  make no replicated write;
- **a repair by the check at open**: `_print_reconciliation` prints only when it repairs.

If one does happen, record it in full. It is a finding, not a stop, unless §5 says otherwise.

---

## 3. The interrupter, before its first real use

Since 06a, `interrupt_when.py` checks the pid it is about to signal against the driver named by the
run's own manifest `argv[0]`, so it can interrupt a `--build`. It has never fired on a real one.
Before the first launch, run both dummy tests at `HEAD`, each into a new directory outside `var/`,
and paste their output:
- `docs/datastore-integrity-verification/interrupter_dummy_test.sh`, five cases, as its record says;
- `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`: it signals a `quadsource_atol_sweep.py`
  child, and refuses a `scoped_pipeline_run.py` one under the same manifest.

Leave no `sleep` child of yours alive afterwards.

---

## 4. Acceptance

1. Every launch ended as §2.1 says, through the registry, under `CLAUDE.md`'s rules, with the
   driver's exit status in each launch log.
2. The resume recomputed nothing it had stored, pruned exactly the unvalidated rows, and its store's
   fingerprint equals the reference's.
3. The version row, the census, `integrity_check`, `foreign_key_check` and the inventory are clean
   on both stores.
4. Every reader exits 0 on the resumed store, refuses nothing, and leaves every file of it
   byte-identical. The `QuadSourceIntegral` extractor reads every row.
5. Both dummy tests pass at `HEAD` before the first launch, and both real firings are logged.
6. The verification record holds every measurement, transcribed. The stores are throwaway, so the
   numbers must live in the document (`prompts/run-registry` README §0 item 4). It holds §2.4's
   checklist.
7. Every existing test passes unmodified. The six suites run serially on the staged tree, at the
   pre-dispatch baselines. `black --check` is clean on every Python file added or changed.
8. The board and the index are updated in the same commit.

---

## 5. Stop conditions — stop and ask

Stop, keep everything as it is, and report to the orchestrator. Do not fix code, rescope or
relaunch past any of these.
- Anything exists under either store's name before the first launch.
- A launch refuses at open, raises `ReplicationMismatch` or `ReplicatedDivergence`, or ends with a
  traceback that is not the interruption's.
- **A killed launch is not recorded `killed`/143**, has no fingerprint, does not exit 143, or
  leaves a Ray process of this repository alive.
- A `done` launch is not `done`/0.
- **The two fingerprints differ.** Paste `compare_fingerprints`' output, and classify each
  difference.
- A resumed stage recomputes a row that an earlier launch stored validated, or `read_inventory`
  reports a `duplicate`.
- A row left unvalidated by a killed launch is still there after the next open.
- The census, `integrity_check`, `foreign_key_check`, the version row or the inventory finds
  anything.
- **A reader exits non-zero, raises `ReadOnlyMiss` or `ReadOnlyWrite`, or changes any file of the
  store.**
- Either dummy test fails, or the interrupter refuses a real run's pid.
- **A build is still running after 2 hours of wall time.** The orchestrator's wait reports this to
  you. Do not stop it yourself: report its last `stage` and the load average.
- An existing test fails, other than the load-sensitive ones in the dispatch brief, each re-run
  alone and on an export of the parent commit.

---

## 6. What this prompt does not do

- It changes no tracked file: no production code (§0), no test, no verification tool.
- It writes no store under `var/` but its two, and retires neither (D7-2).
- It opens no other store under `var/`, not even read-only.
- It does not launch A3 v2, choose its store name, or touch the `handover` board.
- It does not fix what it finds. Every finding goes in the log's *Observations not acted on*, and
  opens a §3.2 issue if it is a defect.

---

## 7. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/a3-v2-readiness-verification.md`, new: the record;
- `docs/a3-v2-readiness/rehearsal/`, new: any script of yours, and saved outputs small enough to
  commit, the dummy tests' among them. These include the census, the counts tables, the fingerprint comparison,
  the prune serials, the readers' summaries and the hashes before and after. Large outputs stay
  under `var/`, and the record names them;
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/07-rehearse-the-a3-v2-launch.md`, in README §5.1's form. In addition, it
records:
- the decisions as the brief gave them;
- every run's id, slug, pid, state, exit code, driver exit status, wall time and load average;
- the interrupter's dummy tests and both real firings;
- §2.2's tables;
- the six suites beside the pre-dispatch baselines.

**The verification record**, `docs/a3-v2-readiness-verification.md`, holds §2's measurements and
§2.4's checklist. It quotes every command verbatim. Its first paragraph names the tree, the stores
and the runs.

**The board.**
- The §1 row for 07, and items L1–L3.
- A §3.2 entry for anything opened.
- `### 5.16 After prompt 07 (<date>)`.
- The header's status is **"9 of 9 prompts landed. The campaign's work is done; it is closed by the
  user after the orchestrator's review of 07."** Do not write that it is closed.

**`docs/OPEN_ISSUES.md`.** Add any issue opened, and correct the count and date by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
