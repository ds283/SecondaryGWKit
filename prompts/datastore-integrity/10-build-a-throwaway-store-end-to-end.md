# Prompt 10 — build a throwaway store end to end, interrupt it, resume it, and read it back

**Campaign:** [`README.md`](README.md) · **Board items:** **V1**, **V2**, **V3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-01, after 01–09d landed and were reviewed. **Gate:** 09d landed and reviewed
(board §1); `git status` is clean; nothing is `running` in the registry; **the user has decided
D10-1 to D10-5** (§1), recorded in the orchestrator's dispatch brief. Where the brief's decision
differs from the recommendation written here, the brief is right.
**Closes:** `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]`, assigned here by
the user on 2026-10-01 (§3). Its entry moves to the board's §4, and its index row is deleted.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. This is the campaign's first real run. Nothing that 01–09 shipped
has run under Ray or written a real store, so most of what can go wrong here has never been seen.
The work is in reading what the run did, not in writing code.

**Read first:**

1. `CLAUDE.md`, the section "Long-running jobs — the run registry", in full. Its five rules bind
   every launch in this prompt, and the reasons for them are in `prompts/run-registry/README.md` §0.
2. [`README.md`](README.md): §0, §1, §3 (the one exception to "nothing under `var/`" is this
   prompt), §4, and §5 rules 7–12.
3. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §1 and §4,
   §4.4's census especially.
4. The "State handed to the next prompt" and "Observations not acted on" sections of every log in
   `logs/`. §0's table lists what each left for this prompt.
5. `docs/gktk-remedial/scoped_pipeline_run.py`, the driver you launch through, in full: its
   `--register`, `--allow-existing`, the heartbeat's meaning and `terminal_state`.
6. `RunRegistry/__init__.py` (`begin`, `liveness`, `Run.finish(..., fingerprint=True)`) and
   `RunRegistry/stores.py` (`create_sidecar`, `fingerprint_store`, `fingerprint_of`,
   `compare_fingerprints`). Also `python -m RunRegistry --help` and `store --help`.
7. `Datastore/SQL/ShardedPool.py`: `_reconcile_replicated_tables`, `_print_reconciliation`, and the
   `replication_in_flight` table that prompt 01 added to the primary.
8. `docs/datastore-integrity-audit/replicated_serial_census.py`, and `extract_common.py`'s
   `resolve_run_selection`.
9. `Datastore/tests/test_inventory_report.py`'s
   `test_verbose_renders_every_distinct_record_as_a_distinct_line`, and the board's §3.3 entry for
   `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]`.

---

## 0. What this prompt is

The README's §0.1 put this prompt last for one reason: **the first archival store must not be the
first thing to exercise the result.** Prompts 01–09d changed how replicated rows are written and
repaired, how every lookup is keyed, and how the store is read and reported. All of it was tested
on stand-ins and fixture stores, without Ray. This prompt runs it for real: a scoped pipeline,
interrupted and resumed, then read back by everything that reads a store.

**What earlier prompts left for this one.** Each item is checked by the step named.

| From | First exercised here | Checked in |
|---|---|---|
| 01 (log obs. 4) | `ShardedPool._replicated_write` and `ReplicationMismatch` under real Ray actors | §2.1: no mismatch is raised; `replication_in_flight` is empty after every clean close |
| 02 (log obs. 3, 6) | the check at open, on real shards, after a real interruption; its two refusals (duplicates, shard keys) on real data | §2.1, and the fault copies of §2.5 |
| 03 | exact identity lookups under concurrent actors | §2.1: the resume recomputes nothing already stored |
| 06 (log obs. 3) | a real `QuadSource` through `store()`, and `PRAGMA foreign_key_check` over real rows | §2.2 |
| 07–09c | every keyed stage on a resume: the WKB rows' `numeric_serial`, `QuadSource`'s `Tq`/`Tr`, `GkSource`'s parent set, `QuadSourceIntegral`'s six parents | §2.1 and §2.3 |
| 09c | the `QuadSourceIntegral` stage's parent resolution before its existence query, paid on a resume | §2.1: its cost, measured |
| 09d | `main.py --inventory` on rows the keyed stages wrote | §2.4 |
| all | the fingerprint at format 6, the census, and every `extract_*.py` script on a store this tree wrote | §2.2–§2.4 |

**What this prompt is not.** It changes no production code: no factory, no target, no `main.py`, no
`tools/`, no `RunRegistry/`, no driver. It amends one existing test (§3), which the README's §2 row
for prompt 10 names, as README §5 rule 7 requires. Everything else it adds is a verification record.

---

## 1. The stores and the runs

### 1.1 The decisions (the user's, before dispatch)

Each is stated with a recommendation and its reason. The dispatch brief says which was taken.

- **D10-1 — the scope.** *Recommended:* LambdaCDM only. Three wavenumbers log-spaced over
  $10^5$–$10^6$/Mpc. `--zend 250`, `--source-samples-log10z 100`, the default response
  sparseness, `--shards 4`, and `main.py`'s default tolerances, policies and tags. `--cpus 6`.
  - **LambdaCDM** is A3 v2's model.
  - **The sampling density** is production's. So the `GkSource` parent sets, and the numeric,
    WKB and mixed policy outcomes, are the sizes and kinds A3 v2 will write.
  - **The wavenumbers and `--zend`** keep the `QuadSourceIntegral` stage bounded, by measurement.
    The A3 baseline store, read `mode=ro` on 2026-10-01 with its SHA-256 unchanged, holds 1,060
    rows whose $k$, $q$ and $r$ are all ≤ $1.1\times10^6$. They cost 6,315 CPU-seconds in total,
    the most expensive 533 s. By triangle, the lowest $z_{\rm response}$ measured is 132 to 230. Below that, the
    stage is unmeasured at every tolerance
    (`prompts/handover` board, `[a3-baseline-quadsource-integrals-are-1680-short]`). Its cost
    there grows roughly as $\eta_R^3$, and could run to days. `--zend 250` keeps every row above
    every triangle's measured minimum. The cost peaks at $k \approx 10^7$, which this scope excludes.
  - *Alternative:* the GkTk-remedial verification's five wavenumbers over $10^5$–$10^7$ with
    `--zend 0.1`. That stage did not finish in three hours there
    (`docs/gktk-remedial-verification.md` §4.4).
- **D10-2 — two builds, compared.** *Recommended:* yes.
  - The **reference** build runs straight through.
  - The **resumed** build has the same scope and is interrupted twice.
  - Their fingerprints must be equal: keys, tags, `validated` and value counts, class by class. The
    fingerprint takes no serial, so that equality shows the resume lost nothing, duplicated nothing
    and keyed nothing differently. No other comparison available here says that.
  - *Alternative:* one interrupted build, checked only against itself. That cannot tell a resume
    that silently skipped work from one that did it.
- **D10-3 — where to interrupt.** *Recommended:* twice, by `SIGTERM` to the driver, which Ray's
  handler turns into `SystemExit(15)` and the driver records as `killed`.
  - Once during `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`, where 09b's set-valued key is
    looked up.
  - Once during `CALCULATE QUADRATIC SOURCE INTEGRALS`, where 09c's six parents are resolved
    before the existence query.
  - Each point must leave its stage with some, but not all, of its rows stored (§1.4).
- **D10-4 — fault copies at open.** *Recommended:* yes, on plain copies of the closed resumed store
  in a temporary directory outside `var/` (§2.5). They are the first test of prompt 02's refusals
  and repair on real shards, which log 02 observation 3 recorded as untested.
- **D10-5 — the stores afterwards.** *Recommended:* both stay in `var/datastores/`, with their
  sidecars, for the orchestrator's review. They are retired, or not, by the user afterwards.
  `store retire` is a person's act (`CLAUDE.md`), and this prompt never runs it.

### 1.2 Names

- **The stores.** `var/datastores/datastore-integrity-10-reference.sqlite` and
  `var/datastores/datastore-integrity-10-resumed.sqlite`. Before the first launch, show that nothing
  exists by either name: no primary, no shard, no sidecar, no tombstone (`store show`). If anything
  does, stop.
- **The runs.** `--campaign datastore-integrity --prompt 10`, with the slugs `reference`,
  `resumed-a`, `resumed-b` and `resumed-c`. Every `--purpose` is one line a stranger can read.
- **`--job-name datastore-integrity-10`** on every launch, and the default `--run-label`, for both
  stores. The two must differ in nothing but the interruption.

### 1.3 How each build is launched — `CLAUDE.md` rules 1–5, with nothing relaxed

The command, with D10-1's scope (substitute the brief's if it differs):

```bash
nohup ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py \
    --k-min 1e5 --k-max 1e6 --k-count 3 --cpus 6 --models LambdaCDM \
    --register <slug> --purpose "<one line>" --campaign datastore-integrity --prompt 10 \
    [--allow-existing] \
    -- --database var/datastores/datastore-integrity-10-<store>.sqlite \
       --job-name datastore-integrity-10 --shards 4 --zend 250 \
       --source-samples-log10z 100 --no-prune-unvalidated \
    > var/runs/datastore-integrity-10-<slug>.launch.log 2>&1 &
disown
```

- **`--no-prune-unvalidated`** is A3 v2's flag (`quadsource_atol_sweep.py`'s `run_build`). The
  resume must be the resume A3 v2 will do. If it leaves a row unvalidated that the reference
  validated, that is a finding (§5), not something to work round.
- **`--allow-existing`** is given to `resumed-b` and `resumed-c` only.
- **The launch log.** It holds only what is printed before the driver registers. It sits beside the
  run directory, not in it, as `store-retirement`'s
  `[00-a-launch-log-lives-outside-its-run-directory]` records. Do not work round that.
- **Before each launch:** `python -m RunRegistry list` shows nothing `running`. **One Ray instance at
  a time**: no other build or extraction of yours is alive, `ray status` finds no cluster, and
  `RAY_ADDRESS` is unset. The two builds run one after the other, never together.
- **After each launch, one check, then end your turn.** In that check:
  - the manifest exists;
  - `status.json` says `running`, with a pid that answers `kill -0`;
  - `RunRegistry list` shows the run.

  Report the run id and pid to the orchestrator, and **end your turn**. The orchestrator waits, with
  one wait on the run's liveness, and resumes you when the run is no longer alive. **Do not poll**,
  and never wait with `pgrep -f` (`CLAUDE.md` rule 2).
- **Record `git status --short` at each launch.** The manifest stamps `git_dirty`. Only your own
  new files under `docs/datastore-integrity-verification/`, and your log, may be present, and no
  tracked file may be modified.
- **Never touch a process you did not start.** The machine may be heavily loaded by an unrelated
  job in another workspace. Record the load average at each launch. It is not a reason to stop.

### 1.4 How a build is interrupted

Write `docs/datastore-integrity-verification/interrupt_when.py`, a small one-shot script.
- **Its arguments** are a run directory, a stage regex and a minimum percentage.
- **The loop.** Every 15 s it reads that run's `status.json`.
  - If the run is `finished`, or its pid no longer answers `kill -0`, it logs that and exits
    **without signalling**.
  - While `RunRegistry.liveness(status, 900)` says `stale` with the pid still alive, it waits and
    never signals. A long work item can make the heartbeat go stale, and stale means "go and look".
  - If the run is `alive` and its `stage` field matches the regex at or above the percentage, it
    checks two things. The pid must still answer `kill -0`, and `ps -p <pid> -o command=` must
    name `scoped_pipeline_run.py` and the run's own slug. Only then does it send `SIGTERM`
    **once**, log it, and exit.
- **Its log** is `var/runs/<run id>/interrupter.log`.
- **It is launched** detached, right after the build it watches, under the same rules.

Before its first real use, exercise it once on a dummy: a fake run directory whose `status.json`
names a `sleep` child of yours. Show it signalling when the stage matches, and exiting silently
when the child dies first. Paste both logs.

**Choosing the percentage.** Read the reference build's `stdout.log` for the two stages' progress
lines, and pick each point so that the stage has stored some rows and not all of them. A stage that
prints no intermediate line cannot be interrupted mid-stage this way. Then say so, and interrupt at
its first line. Afterwards, show from the killed launch's last progress line, and from the next
launch's lookup and compute counts, that the stage had stored some rows but not all.

### 1.5 The order of work

1. **The reference build**, `reference`, to `done`. Then:
   - `store create --purpose …` on its closed primary;
   - `store fingerprint --write`.
2. **The resumed build.**
   - `resumed-a`, with the interrupter on the `GkSource` stage, until it is killed. Then
     `store create` on the closed primary.
   - `resumed-b`, `--allow-existing`, with the interrupter on the `QuadSourceIntegral` stage, until
     it is killed.
   - `resumed-c`, `--allow-existing`, to `done`. Then `store fingerprint --write`.
3. **Between launches, on the closed store**, read-only, record:
   - the `*-journal` files beside the shards;
   - the rows of the primary's `replication_in_flight`, read with `sqlite3` `mode=ro`;
   - the launch's `status.json`, including any `fingerprint_error`.

   The next launch's output shows what the check at open did: `_print_reconciliation` prints only
   when it repairs, and the journals that were present are gone.
4. Then §2.2–§2.5, and §3.

---

## 2. What to verify

### 2.1 V1 — the resume is correct

- **Every launch ends as recorded:** `reference` and `resumed-c` are `done` with exit code 0.
  `resumed-a` and `resumed-b` are `killed`, and their `stderr.log` holds no traceback but the
  interruption's.
- **No `ReplicationMismatch` and no `ReplicatedDivergence`** in any launch's output.
- **Nothing stored is recomputed.** Take each stage's final lookup / compute / store counts from
  every launch's `stdout.log`, and tabulate them.
  - Every stage that finished before an interruption computes **zero** on the next launch.
  - The interrupted stage computes exactly what was missing. Rows in flight at the `SIGTERM` were
    never stored, so they may be computed again; say how many.
  - Summed over `resumed-a`–`c`, each stage stores what the reference stored.
- **The `QuadSourceIntegral` stage's resolution cost on a resume.** It is the 09c log's estimate,
  now measured: the time from the stage's banner to its first compute, on `resumed-b` and
  `resumed-c`, against the reference's.
- **D10-2's comparison.**
  - Fingerprint both closed stores read-only (`fingerprint_store`).
  - `compare_fingerprints(reference, resumed)` must be **empty**, and both must be at format 6.
  - Paste the per-class counts beside each other.

### 2.2 V2 — the store's integrity, on both stores, read-only

- **The census of audit §4.4.** Run `replicated_serial_census.py` on both primaries, as it stands;
  do not edit it. Every table must be `OK`, and the verdict `no divergence`.
  - Check its table list against `config/sharding.py`'s `replicated_tables`, and say what it omits
    or adds.
  - If it omits a replicated table, compare that table too, in a script of your own under
    `docs/datastore-integrity-verification/`.
- **Foreign keys and integrity.** On every shard, opened `mode=ro`, run `PRAGMA integrity_check`
  (must be `ok`) and `PRAGMA foreign_key_check` (must be empty). The declared keys include 06's
  `Tq_serial` → `TkNumericIntegration` and 07's `numeric_serial`. The cross-shard references are
  checked by the next item.
- **The inventory.** `read_inventory(primary)` has no problem in any class: no `unresolved-parent`,
  no `duplicate`, no `orphan-member`, no `empty-parent-set`. Give each class's record count.
- **The fingerprint is unchanged by reading.** Fingerprint both stores again after §2.3 and §2.4. It
  must equal the sidecar's (`store fingerprint`, without `--write`).

### 2.3 V2 — every reader, on the resumed store

Run each `extract_*.py` once on the resumed store, which is the one under test:
`extract_Gk_data.py`, `extract_GkWKB_data.py`, `extract_GkSource_data.py`, `extract_TkWKB_data.py`,
`extract_tensor_source_data.py`, `extract_QuadSourceIntegral_data.py`.
- **How:** `--ray-address local`, so that each bootstraps its own Ray instance, with
  `--output var/datastore-integrity-10/extract/<script>/`. First confirm that `local` is accepted
  by the Ray in `venv`; if it is not, say what you used instead.
- **Each must exit 0.** Record what it read: models, objects, and the run it selected and why
  (`resolve_run_selection`'s line).
- **Fingerprint the store after each script.** It must not move. `extract_QuadSourceIntegral_data.py`
  writes to the store it reads (`[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]`).
  Say whether that changed the content, and whether it left an in-flight record.
- **A failure is a finding.** Do not fix a script: record the traceback and stop (§5).

### 2.4 V2 — the inventory report, on both stores

- **The four runs.** `main.py --database <primary> --inventory`, and with `--inventory-verbose`, on
  each store. For each, record:
  - the exit code and wall time;
  - the report's size, and its longest record line by class.
- **Every label is defined once.** Check every `<Class>#<n>` and `S<n>` printed, using the 09d test
  module's `definitions` and `references` helpers. Import them; do not copy them. No 64-hex digest
  may appear (`HEX64`).
- **The two stores' reports are identical**, verbose and not, once each header's timestamp range is
  masked. Labels take no serial, so anything else that differs is a difference in content, and
  §2.1's fingerprint comparison should have caught it first.

### 2.5 D10-4 — the check at open, on real shards

Copy the closed resumed store with plain file copies (`shutil.copy2` of the primary and its shards)
into a temporary directory **outside `var/`**. Shard records resolve beside their primary
(`Datastore/shard_paths.py`), so the copy opens as itself. Run the census on the untouched copy to
show that. Then make one copy per case. Change each with `sqlite3` alone, and open it.
1. **A row missing on one shard.** Delete one replicated row (a `redshift`, say) from one shard. The
   open must refuse with `ReplicatedDivergence`, naming the class, the shard and the serial.
2. **A duplicate.** Duplicate one `BackgroundModelValue` row on one shard. The open must refuse.
3. **A shard key that is not a wavenumber.** Point one of the primary's `shard_keys` at a serial
   that is no `wavenumber`. The open must refuse.
4. **The repair.** Delete one replicated row from one replica, and write the `replication_in_flight`
   record that an interrupted write of it would have left, with another shard as controller. Read
   the record's schema from prompt 01's code and tests. The open must copy the row forward from
   the controller, clear the record and proceed. Run the census on that copy afterwards.

- **The opener.** Open each copy through something that opens a `ShardedPool` and reads. Prefer an
  `extract_*.py` script with `--ray-address local`, which writes nothing after the open. Say which.
- **Record each case's message verbatim.** Then delete the temporary directory; it is not a store
  under `var/`.

### 2.6 V3 — what A3 v2's launch must now pass

Write it in the verification record (§7) as a checklist, each item pointing to where this prompt
exercised it. Include at least:
- `RunRegistry list` clean; the store's name never used, with no sidecar and no tombstone;
- `quadsource_atol_sweep.py`'s `substitute_main_py()`: all five substitutions still match once.
  Run it now, with no Ray and no store, and paste the counts (README §5 rule 12);
- the launch detached through `--build --register`, a sidecar created on the first close, and the
  fingerprint at finish;
- a resume through `--build --resume`, which reaches prompt 02's check at open;
- after the build: the census, `foreign_key_check`, no inventory problem, `--inventory` readable,
  and the extraction scripts, as §2.2–§2.4 ran them;
- **what this prompt could not exercise**, and why. The full production grid and `--zend 0.1` are
  the obvious ones; the A3 v2 tolerance pair is another. Name every issue still open on this
  board, with what it means for A3 v2.

---

## 3. The one test change

**`Datastore/tests/test_inventory_report.py::test_verbose_renders_every_distinct_record_as_a_distinct_line`**
compares whole record lines. Since 09d, each line of a class that another class names ends in its
own label `  #<n>`, which is different for every record. So for those seven classes the test cannot
fail. **Amend it to compare the lines with that label removed**, and change nothing else in the
test. The README's §2 row for prompt 10 names this test and this reason, which is what rule 7
requires.

**Deliberate breakage.** Write a mutation of `tools/inventory_report.py` under which the amended
test fails and the unamended test passes. For example, a labelled class's record line shows its own
label and markers but none of its fields. Then:
- apply it, and show the amended test failing for each labelled class (sub-tests counted);
- apply it to the parent commit's test file, and show that test passing;
- revert it.

Paste the mutation as `git diff`. Replay it twice, as the 09 logs did:
- before the commit, from the final log file;
- after the commit, from `git show HEAD:<log>`, with plain `git apply`.

Each replay runs in a `git archive` export of the tree, made into its own git repository, with
`venv` linked. Report the second replay to the orchestrator, and do not amend the commit for it.

---

## 4. Acceptance

1. Both builds ended as §2.1 says, through the registry, under `CLAUDE.md`'s five rules.
2. The resume recomputed nothing it had stored. Its store's fingerprint equals the reference's.
3. The census, `integrity_check`, `foreign_key_check` and the inventory are clean on both stores.
4. Every `extract_*.py` script exits 0 on the resumed store, and none moves its fingerprint.
5. `main.py --inventory` and `--inventory-verbose` run on both, with every label defined once and no
   digest. The two stores' reports are identical, timestamps masked.
6. The four fault copies refuse, or repair, as §2.5 says.
7. The amended test passes, and the mutation of §3 is caught by it and not by the old one. Every
   other existing test passes unmodified. The six suites run serially on the staged tree, at the
   pre-dispatch baselines (Datastore unchanged in count). `black --check` is clean.
8. The verification record (§7) holds every measurement, transcribed. The stores are throwaway,
   so the numbers must live in the document (`prompts/run-registry` README §0 item 4).
9. The issue is closed, and the board and index are updated in the same commit.

---

## 5. Stop conditions — stop and ask

Stop, keep everything as it is, and report to the orchestrator. Do not fix production code, rescope
or relaunch past any of these.
- Anything exists under either store's name before the first launch.
- A launch refuses at open, raises `ReplicationMismatch` or `ReplicatedDivergence`, or fails with
  a traceback.
- **The two fingerprints differ.** Paste `compare_fingerprints`' output, and classify each
  difference.
- The census, `integrity_check`, `foreign_key_check` or the inventory finds anything.
- A resumed stage recomputes a row that an earlier launch stored validated.
- An extraction script fails, or moves the fingerprint.
- A fault copy opens when it should refuse, or refuses when it should repair.
- **A build is still running after 4 hours of wall time.** The orchestrator's wait reports this to
  you. Do not stop it yourself: report its last `stage` and the load average.
- `[02-the-version-row-is-written-outside-the-in-flight-record]` is reached, that is, an open
  refuses on the `version` table.
- An existing test fails, other than the three load-sensitive ones in the dispatch brief, each
  re-run alone and on an export of the parent commit.

---

## 6. What this prompt does not do

- It changes no production code. That covers factories, targets, `main.py`, `tools/`,
  `RunRegistry/`, `config/`, both scoped drivers, `quadsource_atol_sweep.py`, every `extract_*.py`
  script and every audit probe.
- It writes no store under `var/` but its two, and retires neither (D10-5).
- It opens no other store under `var/`, not even read-only. The A3 baseline measurement in D10-1
  was the orchestrator's.
- It does not choose A3 v2's tolerance pair, or touch the `handover` board.
- It does not fix what it finds. Every finding goes in the log's "Observations not acted on", and
  opens a §3.3 issue if it is a defect.

---

## 7. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `Datastore/tests/test_inventory_report.py`, that one test only;
- `docs/datastore-integrity-verification.md`, new: the record;
- `docs/datastore-integrity-verification/`, new: `interrupt_when.py`, any comparison script of
  yours, and saved outputs small enough to commit (the census output, the counts tables, the
  fingerprint comparison, the fault cases' messages). Large outputs stay in `var/runs/`, and the
  record names them;
- the log, this board, `docs/OPEN_ISSUES.md`, and README §4 if you name anything new.

**The log**, at `logs/10-build-a-throwaway-store-end-to-end.md`, in README §5.1's form. In
addition, it records:
- the decisions as the brief gave them;
- every run's id, slug, pid, state, exit code, wall time and load average;
- the interrupter's dummy test and both real firings;
- §2.1's tables;
- the deliberate-breakage record of §3;
- the six suites beside the pre-dispatch baselines.

**The verification record**, `docs/datastore-integrity-verification.md`, is additive (`CLAUDE.md`
§6). It holds §2's measurements and §2.6's checklist. It quotes every command verbatim. Its first
paragraph names the tree, the stores and the runs.

**The board.**
- The §1 row for 10, and items V1–V3.
- The issue moved to §4, with a closing note naming this commit by subject.
- A §3.3 entry for anything opened.
- `### 5.16 After prompt 10 (<date>)`.
- The header's status is **"12 of 12 prompts landed. The campaign's work is done; it is closed by
  the user after the orchestrator's review of 10."** Do not write that it is closed.

**`docs/OPEN_ISSUES.md`.** Delete the issue's row, and add any opened. Add one sentence to §1.15's
paragraph. Correct the count and date by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
