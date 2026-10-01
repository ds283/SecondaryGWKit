# Datastore integrity campaign — the end-to-end verification (prompt 10)

**Campaign:** [`prompts/datastore-integrity/README.md`](../prompts/datastore-integrity/README.md) ·
**Board:** [`prompts/datastore-integrity/IMPLEMENTATION_STATE.md`](../prompts/datastore-integrity/IMPLEMENTATION_STATE.md) ·
**Prompt:** [`prompts/datastore-integrity/10-build-a-throwaway-store-end-to-end.md`](../prompts/datastore-integrity/10-build-a-throwaway-store-end-to-end.md) ·
**Log:** [`prompts/datastore-integrity/logs/10-build-a-throwaway-store-end-to-end.md`](../prompts/datastore-integrity/logs/10-build-a-throwaway-store-end-to-end.md)

Every measurement below was taken on branch `handover-remedial` at **`d144d04`**. The tree held
nothing else but this prompt's own new files: `git status --short` was
`?? docs/datastore-integrity-verification/` at every launch, which is why each manifest stamps
`git_dirty: true`. Two throwaway stores were built under `var/datastores/`:
**`datastore-integrity-10-reference.sqlite`** (sidecar `store_id` `da35422992af45c7a6a13f5fd6a065e0`),
built straight through by the run **`reference`**; and **`datastore-integrity-10-resumed.sqlite`**
(`store_id` `a71da55b99e94728a9f311849910212e`), built by **`resumed-a`** (interrupted in the
`GkSource` stage), **`resumed-b`** (interrupted in the `QuadSourceIntegral` stage), **`resumed-c`**
(to `done`) and **`resumed-d`** (the user's pruned resume, to `done`). The runs are under
`var/runs/datastore-integrity-10-<slug>-<stamp>/`. Both stores stay, with their sidecars, for the
orchestrator's review (D10-5); neither is retired here. The small outputs quoted below are saved in
[`datastore-integrity-verification/`](datastore-integrity-verification/), and the large ones stay
under `var/` (§6).

**Headline.**
1. **After the user's pruned resume, the resumed store's fingerprint equals the reference's**:
   `compare_fingerprints` is empty, both at format 6, digest
   `10b8725f36bd2aa14f74cc8912efe3dd80961c4fc015fe31729b52aad4f61a1f`, all 21 classes equal.
2. **No stage that finished before an interruption computed anything on the next launch**, and
   each interrupted stage computed what it lacked.
3. **`--no-prune-unvalidated`, the flag A3 v2's launch passes, duplicates rows on a resume.** An
   interruption between a row's store and its validation leaves it unvalidated; under the flag it
   is not pruned, the existence query does not see it, and the resume stores a second row beside
   it. Measured: 45 `GkSource` rows held twice. The user's reading (2026-10-01): pruning is the
   designed recovery, and the flag is the bug. Opened against A3 v2's launch, owned by `handover`
   (§2.1.4).
4. **A `SIGTERM` to the driver is never recorded `killed`.** Delivered while the driver waits in
   `ray.wait`, it reaches Python as `SystemExit(1)` and is recorded `failed`; when the cleanup
   takes more than about 3 s, Ray's C++ signal handler aborts the process and nothing is recorded.
5. **Integrity holds on real shards**: the census, `integrity_check`, `foreign_key_check` and the
   inventory are clean on both stores; prompt 02's three refusals and its repair behave on real
   shards exactly as designed.
6. **Every `extract_*.py` script reads LambdaCDM in full and then exits 1** on `QCD_Cosmology`, which
   a LambdaCDM-only store does not hold. None moves the fingerprint. Accepted by the user's decision
   (2026-10-02) and opened as a defect: A3 v2's store is LambdaCDM-only too.

---

## 1. The runs

### 1.1 How each was launched

D10-1's scope, `CLAUDE.md`'s five rules. The command, verbatim for `reference` (the others differ
only in slug, purpose, store and `--allow-existing`, as the table says):

```bash
PYTHONPATH=. nohup ./venv/bin/python -u docs/gktk-remedial/scoped_pipeline_run.py \
    --k-min 1e5 --k-max 1e6 --k-count 3 --cpus 6 --models LambdaCDM \
    --register reference --purpose "datastore-integrity prompt 10: reference build of a throwaway LambdaCDM store (3 k over 1e5-1e6/Mpc, zend 250), run straight through, whose fingerprint the interrupted-and-resumed build must equal" --campaign datastore-integrity --prompt 10 \
    -- --database var/datastores/datastore-integrity-10-reference.sqlite \
       --job-name datastore-integrity-10 --shards 4 --zend 250 \
       --source-samples-log10z 100 --no-prune-unvalidated \
    > var/runs/datastore-integrity-10-reference.launch.log 2>&1 &
disown
```

`PYTHONPATH=.` is not in the prompt's §1.3 command; the driver's `register()` imports `RunRegistry`,
and `main.py` its packages, from the repository root, as the driver's own usage line says.
`resumed-d` alone omits `--no-prune-unvalidated` (§2.1.4).

Before each launch: `python -m RunRegistry list` showed nothing `running`; `ray status` found no
running Ray instance; `RAY_ADDRESS` was unset; `git status --short` showed only
`?? docs/datastore-integrity-verification/`. After each, one check (the manifest exists,
`status.json` says `running` with a pid answering `kill -0`, `RunRegistry list` shows the run), and
the turn ended; the orchestrator waited, once per launch, on the run's liveness.

| Run | Slug | Pid | `--allow-existing` | Launched | Ended | Wall | State, exit | Load average at launch |
|---|---|---|---|---|---|---|---|---|
| `datastore-integrity-10-reference-20261001T204320` | `reference` | 84388 | no | 20:43:20 | 20:58:18 | 14 m 58 s | `done`, 0 | 22.52 20.59 19.74 |
| `datastore-integrity-10-resumed-a-20261001T210301` | `resumed-a` | 1238 | no | 21:03:01 | 21:05:26 | 2 m 25 s | `failed`, 1 (§2.1.1) | 15.36 18.50 22.48 |
| `datastore-integrity-10-resumed-b-20261001T230813` | `resumed-b` | 6384 | yes | 23:08:13 | 23:15:32 (aborted) | 7 m 19 s | none written; `killed`, `null` by a person at 23:44:51 (§2.1.1) | 10.67 10.13 11.51 |
| `datastore-integrity-10-resumed-c-20261001T234503` | `resumed-c` | 38930 | yes | 23:45:03 | 23:53:17 | 8 m 14 s | `done`, 0 | 12.35 16.12 19.20 |
| `datastore-integrity-10-resumed-d-20261001T235455` | `resumed-d` | 48129 | yes | 23:54:55 | 23:55:28 | 33 s | `done`, 0 | 10.62 19.08 21.16 |

All on 2026-10-01 (BST). `hw.ncpu` is 10. Free space on `var/`'s volume was 45 GB before
`reference`, 36 GB before `resumed-b` (nothing of this prompt ran in between) and 39–40 GB after;
Ray's raylet warned on every run that `/tmp/ray` is "over 95% full". No run was affected.

**Between launches, on the closed store**, read-only: no `*-journal` file beside any shard after
any launch; the primary's `replication_in_flight` had **0 rows** after every launch (`sqlite3
"file:<primary>?mode=ro" "select count(*) from replication_in_flight;"`). No launch's output
contains `Rolled back the journal` or `was opened with a replicated write in flight`, so prompt
02's check at open repaired nothing on any real open; none contains `ReplicationMismatch` or
`ReplicatedDivergence`, and no `stderr.log` or launch log holds a Python traceback.

**Sidecars and fingerprints.** `store create --purpose …` on the closed reference primary at
21:01:35, then `store fingerprint --write` (digest `10b8725f…1a1f`, "it replaced none"). On the
resumed primary, `store create` at 23:08:03, after `resumed-a`; `resumed-c` and `resumed-d` wrote
their fingerprints into the sidecar at finish (`fingerprint_sidecar: written`), and
`store fingerprint --write` after each matched it.

### 1.2 The interrupter

[`interrupt_when.py`](datastore-integrity-verification/interrupt_when.py), launched detached right
after the build it watches, reads the run's `status.json` every 15 s and sends one `SIGTERM` when the
run is `alive` and its `stage` matches a regex at or above a percentage, after checking the pid
answers `kill -0` and that `ps -p <pid> -o command=` names `scoped_pipeline_run.py` and
`--register <slug>`. It waits while the run is `stale`, and exits without signalling when the run
has finished or its pid is gone. `--hold S` waits `S` seconds after the first eligible reading and
signals only if the next reading is eligible too.

**Its dummy test** ([`interrupter_dummy_test.sh`](datastore-integrity-verification/interrupter_dummy_test.sh),
output [`interrupter_dummy_test.txt`](datastore-integrity-verification/interrupter_dummy_test.txt),
polling every 1 s): fake run directories whose `status.json` names a `sleep` child named, with
`exec -a`, as the driver and a slug.
- **A**, signals: a stale heartbeat at a matching stage, waited; fresh at 10 %, waited; at 45 %
  against 40 %, `SIGTERM` once; the child exited 143.
- **B**, silent: the child ended first; "no longer answers kill -0; exiting without signalling".
- **C**: a plain `sleep` at a matching stage is refused, "command line does not name
  scoped_pipeline_run.py and --register dummy-c"; exit 2, the child alive.
- **D**, `--hold 4` at a bare banner: held, read again, signalled; the child exited 143.
- **E**, `--hold 4` and the stage completes during the hold: no signal; exits when the child dies.

**Choosing the points**, from `reference`'s `stdout.log`:
- **`CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` printed no intermediate progress line**: 291
  items in 33.9 s, under the pool's notify interval. So it was interrupted from its first line, with
  `--hold 8`: the signal lands 8–23 s into a 34 s stage.
- **`CALCULATE QUADRATIC SOURCE INTEGRALS`'s percentage counts work items taken off the queue, and
  its 864 integrals are 2 work items**, so every progress line reads 100.00 %. Its first line came
  at 5 m 01 s with 594 of 864 stored. So it was interrupted at its first progress line: regex
  `^CALCULATE QUADRATIC SOURCE INTEGRALS, `, 100 %, no hold.

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py "$D" '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --hold 8 > /dev/null 2>&1 &   # resumed-a, pid 1264
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py "$D" '^CALCULATE QUADRATIC SOURCE INTEGRALS, ' 100 > /dev/null 2>&1 &   # resumed-b, pid 6395
```

**The two firings** ([`interrupter_firings.txt`](datastore-integrity-verification/interrupter_firings.txt)):
`resumed-a`'s saw the banner at 21:05:17, held 8 s and sent `SIGTERM` at 21:05:25; `resumed-b`'s
saw `CALCULATE QUADRATIC SOURCE INTEGRALS, 100.00% (0 of 2 work items remaining)` at 23:15:28 and
sent it at 23:15:29. Each confirmed the pid's command line first.

---

## 2. What was verified

### 2.1 V1 — the resume

#### 2.1.1 How each launch ended

- `reference`, `resumed-c` and `resumed-d`: `done`, exit 0.
- **`resumed-a`: `failed`, exit 1.** The launch log has Ray's `*** SIGTERM received at
  time=1790885125 ***` with a C++ stack in `ray::core::CoreWorker::Wait`, and no Python traceback.
  [`sigterm_in_ray_wait_probe.py`](datastore-integrity-verification/sigterm_in_ray_wait_probe.py)
  shows why: a driver that has called `ray.init` (whose handler is `sys.exit(signum)`) and is
  blocked in `ray.wait` catches **`builtins.SystemExit: code=1`**, not 15, and the driver's
  `terminal_state(1)` is `failed`.
- **`resumed-b`: no terminal state.** The interrupter's `SIGTERM` at 23:15:29.145; the GCS log
  records the driver failing at **23:15:32.164**, 3.0 s later; `status.json` stayed `running`, and
  the orchestrator's wait ended `stale`. [`sigterm_alarm_probe.py`](datastore-integrity-verification/sigterm_alarm_probe.py)
  shows why: after the same `SystemExit(1)`, the driver is alive at 2.5 s of cleanup and gone
  before 3.0 s, **exit status 134 (SIGABRT)**. That is consistent with absl's failure-signal
  handler, which printed the stack and by default arms a 3 s alarm that aborts; the mechanism is
  inferred, the 3 s and the abort are measured.
  `resumed-a` survived because its cleanup finished in 2.2 s. The aborted driver left its
  `gcs_server` (pid 6426, parent 1) alive, hung after its own `SIGTERM`. Both probes' output is in
  [`sigterm_probes.txt`](datastore-integrity-verification/sigterm_probes.txt).
- **By the user's decisions (2026-10-01):** `resumed-a` and `resumed-b` are recorded as their
  evidence shows, in place of §2.1's `killed`: the interrupter's log, Ray's `SIGTERM received`
  banner, no traceback, and the part-stored stage. `gcs_server` 6426 was killed with `SIGKILL`
  after its command line was checked; `ray status` then found no instance. `resumed-b` was ended by
  a person, `RunRegistry.Run(…).finish('killed', exit_code=None)`, after `kill -0 6384` failed:
  `{"exit_code": null, "heartbeat": "2026-10-01T23:44:51+01:00", "pid": 6384, "stage": "CALCULATE
  QUADRATIC SOURCE INTEGRALS, 100.00% (0 of 2 work items remaining)", "state": "killed", …}`. The
  driver's handling is opened as an issue (§5).

#### 2.1.2 Nothing stored is recomputed

Each stage's final lookup / compute / store counts, from each `stdout.log`
([`stage_counts.py`](datastore-integrity-verification/stage_counts.py),
[`stage_counts.txt`](datastore-integrity-verification/stage_counts.txt)); for the stage a launch was
killed in, its last progress line:

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` | `resumed-d` |
|---|---|---|---|---|---|
| exit times, source k | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 | 3/0/0 |
| exit times, response k | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| Tk numeric | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 | 0/0/0 |
| Tk WKB | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 | 0/0/0 |
| quadratic source terms | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 | 0/0/0 |
| Gk numeric | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 | 0/0/0 |
| Gk WKB | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 | 0/0/0 |
| `GkSource` | 291/291/291 | killed; no progress line (161 stored, 45 of them unvalidated) | 175/175/175 | 0/0/0 | 0/0/0 |
| `GkSourcePolicyData` | 291/291/291 | — | 291/291/291 | 0/0/0 | 0/0/0 |
| `QuadSourceIntegral` | 864/864/864 | — | killed at 864/485/451 (last progress, 23:15:24; 485 stored at the abort) | 379/379/379 | 0/0/0 |

- **Every stage that finished before an interruption computes zero on the next launch.**
- **`GkSource`.** `resumed-a` stored 161 rows before the `SIGTERM`, 116 validated and **45 stored
  with all their values and `validated = 0`**: the signal fell between their store and their
  validation. `resumed-b` computed 175 = the 130 missing + the 45 unvalidated, which under
  `--no-prune-unvalidated` stayed in the store unseen by the existence query (§2.1.4).
- **`QuadSourceIntegral`.** `resumed-b` stored 485; the 379 computes in flight at the `SIGTERM`
  were never stored, and `resumed-c` computed exactly those 379.
- **Summed** over `resumed-a`–`c`, less the 45 that `resumed-d` pruned, each stage stores what the
  reference stored: `GkSource` 161 + 175 − 45 = 291; `GkSourcePolicyData` 291; `QuadSourceIntegral`
  485 + 379 = 864; the rest in `resumed-a` alone.

#### 2.1.3 The `QuadSourceIntegral` stage's resolution cost on a resume

The 09c log's estimate, measured ([`qsi_resolution_cost.py`](datastore-integrity-verification/qsi_resolution_cost.py),
[`qsi_resolution_cost.txt`](datastore-integrity-verification/qsi_resolution_cost.txt)). The banner's
time is the pool's start (a progress line's stamp less its running time); the first compute's start
is the earliest `timestamp − compute_time` over the rows the launch stored.

| Launch | Banner | First compute began | Banner to first compute |
|---|---|---|---|
| `reference` (nothing stored) | 20:46:09.65 | 20:46:26.63 | **17.0 s** |
| `resumed-b` (nothing stored) | 23:09:22.69 | 23:09:39.60 | **16.9 s** |
| `resumed-c` (485 of 864 stored) | 23:45:31.90 | 23:45:48.24 | **16.3 s** |
| `resumed-d` (all 864 stored) | — | — | the whole stage, 6.78 s, computing nothing |

At this scope the parent resolution and existence query cost the same whether or not the rows exist;
a resume pays nothing extra.

#### 2.1.4 What `--no-prune-unvalidated` does on a resume, and the user's pruned resume

**Measured.** The resumed store after `resumed-b` (read with `read_inventory`, every file's SHA-256
identical before and after; `fingerprint_store` refuses a store a `running` run names) and after
`resumed-c` (its sidecar fingerprint):
- `GkSource` **336** records against the reference's 291, with the inventory's problem
  `duplicate: GkSource: 45 key-and-tag set(s) are held by more than one record (90 records, all
  kept); e.g. shard/serial #1/561, #1/1065; shard/serial #1/557, #1/1054; …`;
- each of the 45 pairs: one row from `resumed-a` (`validated = 0`, 21:05) and one from `resumed-b`
  (`validated = 1`, 23:08), the same key, `parents_digest` and `z_samples`, 14 on shard 0, 18 on
  shard 1 and 13 on shard 3;
- after `resumed-c`: digest `2067dc3dd3191124ff8fb30420f658effa7507970636ffcd3eb8158852a217c8`,
  problems `{'GkSource': {'duplicate': 1}}`; every class's digest equal to the reference's except
  `GkSource` (336, `5e873d4312c9` against 291, `6a001ba4e735`). `GkSourcePolicyData` (291) and
  `QuadSourceIntegral` (864) equal.

Both `resumed-b` and `resumed-c` printed `!! INTEGRITY WARNING … The following unvalidated GkSource
objects were detected in the datastore` at startup, and pruned nothing.

**The user's decision (2026-10-01).** "The bug here is the `--no-prune-unvaiidated`. Pruning is
there precisely to deal with this situation." So the existence query is not the defect; A3 v2's
launch is. `docs/handover/quadsource_atol_sweep.py`'s `--build` passes the flag, and `--build
--resume` with it; this prompt's §1.3 copied it from there. Opened against that launch, owned by
`handover` (§5).

**`resumed-d`**, the user's pruned resume: the same launch with `--allow-existing`, no interrupter,
and **without** `--no-prune-unvalidated`. At startup it printed, per shard, the unvalidated rows and
`** Note: these values have been pruned from the datastore.`: shard 3 listed its 13, each with its
value count equal to the expected one; shards 0 and 1 appear as `[repeated 32x across cluster]`,
because Ray deduplicates actor output ([`resumed_d_pruning.txt`](datastore-integrity-verification/resumed_d_pruning.txt)).
It then computed and stored **nothing** in any stage. Its fingerprint at finish:
`10b8725f36bd2aa14f74cc8912efe3dd80961c4fc015fe31729b52aad4f61a1f`, problems `{}`. After it, the
pruned rows' children are gone too: `GkSourceValue` 169,653, `GkSource_tags` 2,328 and
`GkSource_parents` 169,653, as in the reference.

#### 2.1.5 D10-2's comparison

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \
    var/datastores/datastore-integrity-10-reference.sqlite \
    var/datastores/datastore-integrity-10-resumed.sqlite
```

Both fingerprinted read-only with `fingerprint_store(write=False)`
([`fingerprint_comparison.txt`](datastore-integrity-verification/fingerprint_comparison.txt)):

```
reference: format 6, digest 10b8725f36bd2aa14f74cc8912efe3dd80961c4fc015fe31729b52aad4f61a1f
  problems none; against its sidecar: 0 difference(s), it matches
resumed: format 6, digest 10b8725f36bd2aa14f74cc8912efe3dd80961c4fc015fe31729b52aad4f61a1f
  problems none; against its sidecar: 0 difference(s), it matches

compare_fingerprints(reference, resumed): 0 difference(s)
```

| Class | Reference | Resumed | Digests |
|---|---|---|---|
| `version` | 1 | 1 | equal |
| `store_tag` | 10 | 10 | equal |
| `redshift` | 1159 | 1159 | equal |
| `wavenumber` | 3 | 3 | equal |
| `tolerance` | 6 | 6 | equal |
| `LambdaCDM` | 1 | 1 | equal |
| `QCD_Cosmology` | 1 | 1 | equal |
| `IntegrationSolver` | 7 | 7 | equal |
| `GkSourcePolicy` | 2 | 2 | equal |
| `QuadSourcePolicy` | 2 | 2 | equal |
| `wavenumber_exit_time` | 3 | 3 | equal |
| `BackgroundModel` | 1 | 1 | equal |
| `TkNumericIntegration` | 3 | 3 | equal |
| `TkWKBIntegration` | 3 | 3 | equal |
| `GkNumericIntegration` | 1368 | 1368 | equal |
| `GkWKBIntegration` | 3477 | 3477 | equal |
| `GkSource` | 291 | 291 | equal |
| `GkSourcePolicyData` | 291 | 291 | equal |
| `QuadSource` | 6 | 6 | equal |
| `QuadSourceIntegral` | 864 | 864 | equal |
| `OneLoopIntegral` | 0 | 0 | equal |

### 2.2 V2 — the stores' integrity, read-only

- **The census of audit §4.4**, as it stands:
  `./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py <reference primary> <resumed primary>`.
  Every table `OK` on both, verdict `no divergence` ([`census.txt`](datastore-integrity-verification/census.txt)).
  Its list is `config/sharding.py`'s thirteen `replicated_tables` plus `BackgroundModel_tags`,
  `BackgroundModel`'s association table; it omits none. No comparison of my own was needed.
- **`PRAGMA integrity_check`** is `ok` and **`PRAGMA foreign_key_check`** returns no row on all ten
  files, each opened `mode=ro` ([`integrity_and_foreign_keys.txt`](datastore-integrity-verification/integrity_and_foreign_keys.txt)).
  The 109 declared keys on a shard ([`declared_foreign_keys.txt`](datastore-integrity-verification/declared_foreign_keys.txt))
  include `QuadSource.Tq_serial -> TkNumericIntegration.serial` (06),
  `TkWKBIntegration.numeric_serial -> TkNumericIntegration.serial` and
  `GkWKBIntegration.numeric_serial -> GkNumericIntegration.serial` (07).
- **The inventory**: `read_inventory` finds **0 problems** on either store, in all 21 classes, so
  no `unresolved-parent`, `duplicate`, `orphan-member` or `empty-parent-set`. The record counts are
  the table in §2.1.5 ([`inventory_problems.txt`](datastore-integrity-verification/inventory_problems.txt)).
- **Reading moves nothing.** After §2.3–§2.5, `store fingerprint` (without `--write`) on each store
  reports digest `10b8725f…1a1f`, problems none, and "matches the fingerprint taken" at
  21:01:40 (reference) and 23:58:21 (resumed)
  ([`fingerprint_after_reading.txt`](datastore-integrity-verification/fingerprint_after_reading.txt)).

### 2.3 V2 — every reader, on the resumed store

Each script once, serially, after `resumed-d`, with its own local Ray instance (Ray 2.43's
`ray.init` accepts `"local"`), and after each `store fingerprint`, the in-flight table and any Ray
process left ([`extractions.txt`](datastore-integrity-verification/extractions.txt)):

```bash
PYTHONPATH=. ./venv/bin/python -u <script>.py --database var/datastores/datastore-integrity-10-resumed.sqlite \
    --ray-address local --output var/datastore-integrity-10/extract/<script>/ \
    > var/datastore-integrity-10/extract/<script>.log 2>&1
```

| Script | Exit | Wall | LambdaCDM read (lookups, last queue) | Fingerprint after |
|---|---|---|---|---|
| `extract_Gk_data.py` | 1 | 28 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | unmoved |
| `extract_GkWKB_data.py` | 1 | 67 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | unmoved |
| `extract_GkSource_data.py` | 1 | 36 s | 10 | unmoved |
| `extract_TkWKB_data.py` | 1 | 15 s | 3 | unmoved |
| `extract_tensor_source_data.py` | 1 | 18 s | 6 | unmoved |
| `extract_QuadSourceIntegral_data.py` | 1 | 22 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 | unmoved |

- **What each read.** Models `LambdaCDM` then `QCD_Cosmology` (`config.model_list`); for
  LambdaCDM, its two exit-time queues (3 each) and the queue above, with `LambdaCDM/` outputs
  written. The run selected, by every script: `** reading run "default", the only run in this
  datastore (no --run-label given)`.
- **Why each exits 1.** Each then printed `>> RUNNING PIPELINE FOR MODEL QCD_Cosmology` and raised,
  verbatim, `RuntimeError: Could not locate suitable background model instance in the datastore`.
  The store holds only LambdaCDM's background (D10-1), and the scripts read every model of
  `build_model_list` with no option to restrict it. **By the user's decision (2026-10-02)** this
  result stands in place of "each must exit 0"; the scripts are opened as a defect (§5), since A3
  v2's store is LambdaCDM-only too.
- **No reader writes the content.** After every script the digest was
  `10b8725f…1a1f`, problems none, matching the sidecar; `replication_in_flight` had 0 rows; no
  journal appeared; no Ray process was left. `extract_QuadSourceIntegral_data.py`, which looks its
  `GkSourcePolicy` up with `object_get`, changed nothing and left no in-flight record here.

### 2.4 V2 — the inventory report, on both stores

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/inventory_reports.py \
    var/datastores/datastore-integrity-10-reference.sqlite \
    var/datastores/datastore-integrity-10-resumed.sqlite var/datastore-integrity-10/inventory
```

It runs `./venv/bin/python main.py --database <primary> --inventory [--inventory-verbose]`
([`inventory_reports.txt`](datastore-integrity-verification/inventory_reports.txt); the four reports
are in `var/datastore-integrity-10/inventory/`).

| Run | Exit | Wall | Size | Lines |
|---|---|---|---|---|
| reference | 0 | 5.45 s | 881,293 bytes | 1,782 |
| reference, verbose | 0 | 5.99 s | 13,282,654 bytes | 7,843 |
| resumed | 0 | 5.80 s | 881,291 bytes | 1,782 |
| resumed, verbose | 0 | 7.16 s | 13,282,652 bytes | 7,843 |

The two bytes are the store's name in the first line. The longest record line by class (characters,
without the bullet), the same on both stores:

| Class | Not verbose | Verbose |
|---|---|---|
| `BackgroundModel` | 275 | 303 |
| `QuadSourceIntegral` | 220 | 266 |
| `QCD_Cosmology` | 130 | 145 |
| `GkWKBIntegration` | 96 | 122 |
| `LambdaCDM` | 92 | 105 |
| `QuadSource` | 91 | 113 |
| `GkSource` | 62 | 86 |
| `TkWKBIntegration` | 59 | 70 |
| `TkNumericIntegration` | 47 | 64 |
| `GkNumericIntegration` | 47 | 68 |
| `store_tag` | 38 | 38 |
| `IntegrationSolver` | 34 | 34 |
| `GkSourcePolicyData` | 31 | 46 |
| `GkSourcePolicy`, `QuadSourcePolicy` | 19 | 19 |
| `tolerance` | 18 | 29 |
| `wavenumber` | 16 | 27 |
| `version` | 14 | 14 |
| `redshift` | 9 | 20 |
| `wavenumber_exit_time` | 8 | 19 |
| `OneLoopIntegral` | — | — |

Most of a report's size is its parent sets' lines (`parent set S<n>: [...]`, one member per
`z_source`), which are not record lines.

- **Every label is defined once.** With prompt 09d's test helpers `definitions` and `references`,
  imported: not verbose, 1,669 row labels and 5 set labels defined, 1,658 and 5 named; verbose,
  5,439 and 291 defined, 5,436 and 291 named. In all four, no label is defined more than once and
  none is named without being defined. **No 64-hex digest** (`HEX64`) appears.
- **The two stores' reports are identical**, verbose and not, once the first line (the store's
  name) and every class header's timestamp range are masked.

### 2.5 D10-4 — the check at open, on real shards

```bash
PYTHONPATH=. ./venv/bin/python -u docs/datastore-integrity-verification/fault_copies.py \
    var/datastores/datastore-integrity-10-resumed.sqlite <work dir>
```

[`fault_copies.py`](datastore-integrity-verification/fault_copies.py) copied the closed resumed
store, primary and shards, with `shutil.copy2`, into a work directory in the session scratchpad,
outside `var/`; changed each case's copy with the `sqlite3` command-line tool alone; and opened it
with **`extract_TkWKB_data.py --ray-address local`**, whose `ShardedPool` open runs the check before
any actor exists. The work directory was deleted afterwards. Output in
[`fault_cases.txt`](datastore-integrity-verification/fault_cases.txt).

- **The untouched copy** opens as itself: the census on it is `no divergence`, every table at the
  original's counts.
- **1. A `redshift` row missing on one shard.** `delete from redshift where serial = 1182;` on
  shard 1. Refused:
  > `ReplicatedDivergence: Cannot open sharded datastore "<work dir>/case1/datastore-integrity-10-resumed.sqlite": its replicated tables differ across shards, and there is no replication_in_flight record, so no interrupted replication explains them. 1 difference(s):`
  > `  - class "redshift", shard [1], serial 1182: held by shard(s) [0, 2, 3], missing on shard(s) [1]`
  > `Nothing was repaired: no shard and not the primary was written, and the replication_in_flight record, if any, was left as it was. A store in this state cannot be put right by this tree: it needs a person, or regeneration.`
- **2. A duplicate `BackgroundModelValue` row** on shard 1 (every column but `serial` copied from
  its first row). Refused:
  > `  - class "BackgroundModelValue", shard 1, key (model_serial=1, z_serial=1): more than one row holds this key on the shard`
  (with the same first and last lines).
- **3. A shard key that is not a wavenumber.** The primary's `shard_keys` were `1|3, 2|0, 3|1`;
  `update shard_keys set key_serial = 1003 where key_serial = (select min(key_serial) from
  shard_keys);`. Refused:
  > `  - class "shard_keys", shard [0, 1, 2, 3], key_serial 1003: the primary assigns wavenumber serial 1003 to shard 3, but it is missing on shard(s) [0, 1, 2, 3]`
- **4. The repair.** On shard 0, the controller, `redshift` serial 1182 carries timestamp
  `2026-10-01 21:03:08.859277` (one timestamp for all 1,159, the driver's, prompt 01). Deleted from
  shard 2; the primary's `replication_in_flight`, empty, given the record an interrupted `get` would
  leave, in prompt 01's schema (`operation`, `class_name`, `controller_shard`, `store_id`,
  `started`), as `test_reconcile_at_open.py` writes it:
  `insert into replication_in_flight (operation, class_name, controller_shard, store_id, started) values ('get', 'redshift', 0, NULL, '2026-10-01 21:03:08.859277');`.
  The open repaired and proceeded:
  > `>> Sharded datastore "<work dir>/case4/datastore-integrity-10-resumed.sqlite" was opened with a replicated write in flight (operation="get", class="redshift", controlling shard=0, store_id=None, started=2026-10-01 21:03:08.859277)`
  > `>>   copied 1 row(s) of "redshift" from shard 0 to shard 2: 1182`
  > `>>   the replication_in_flight record was cleared`
  > `>> Opened existing sharded datastore "<work dir>/case4/datastore-integrity-10-resumed.sqlite" with 4 shards`

  LambdaCDM was then read (`>> RUNNING PIPELINE FOR MODEL LambdaCDM`), and the opener exited 1 on
  `QCD_Cosmology`, the accepted failure of §2.3. Afterwards the record table had 0 rows, shard 2 held
  `1182|249.99999999999994|2026-10-01 21:03:08.859277`, and the census on the copy was `no
  divergence`, `redshift` at 1159 on every shard.

None refused on the `version` table, so `[02-the-version-row-is-written-outside-the-in-flight-record]`
was not reached.

### 2.6 V3 — what A3 v2's launch must now pass

Each item points to where this prompt exercised it.

- [ ] **`RunRegistry list` clean** — nothing `running`, nothing stale (§1.1, before every launch).
- [ ] **The store's name never used**: `store show` reports kind `absent`, no sidecar, no tombstone,
  no file of that name (§1.1; checked for both names before `reference`).
- [ ] **`substitute_main_py()` matches all five substitutions once.** Run on this tree, with no
  Ray and no store ([`substitute_main_py.txt`](datastore-integrity-verification/substitute_main_py.txt)):
  `count=1` for each, at `main.py` lines 4354, 4366, 4255, 4256 and 4148.
- [ ] **Launched detached through `--build --register`**, with a manifest written first, a sidecar
  created on the first close (`store create`), and the fingerprint at finish (§1.1: every run;
  `fingerprint_sidecar: written` once the sidecar exists).
- [ ] **Resume through `--build --resume` without `--no-prune-unvalidated`.** With the flag, a resume
  duplicates every row stored but not validated at the interruption (§2.1.4); without it the
  unvalidated rows are pruned at open and the resume equals an uninterrupted build. The sweep
  script passes the flag today; **this blocks A3 v2's resume as planned** until it does not (§5).
- [ ] **A resume reaches prompt 02's check at open** (§2.1: every resumed launch; nothing to repair
  on a clean close), and the check refuses and repairs as designed on real shards (§2.5).
- [ ] **An interrupted launch is not recorded `killed`.** A `SIGTERM` is recorded `failed`/1 or
  not at all, and may leave a `gcs_server` behind: after any interruption, check `ray status` and
  the run's state by hand, and end a stale run with `Run.finish` (§2.1.1, §5). The sweep's
  `--build` uses the same `terminal_state`.
- [ ] **After the build:** the census, `foreign_key_check`, no inventory problem, `--inventory`
  readable, and the extraction scripts, as §2.2–§2.4 ran them. **The extraction scripts exit 1** on
  a LambdaCDM-only store after reading LambdaCDM in full (§2.3, §5).
- [ ] **No reader moves the fingerprint** (§2.3).

**What this prompt could not exercise, and why.**
- **The full production grid** (50 source and 50 response k over $10^5$–$3\times10^8$/Mpc) and
  **`--zend 0.1`**: unaffordable here; the `QuadSourceIntegral` stage below the A3 baseline's
  measured $z_{\rm response}$ is unmeasured and could run to days (D10-1). So production's batch
  count (this scope had 2 work items of 750), the set sizes at low $z$, and the 09c estimate of
  ~23 minutes of parent resolution over 100 $z_{\rm response}$ are not measured here.
- **The A3 v2 tolerance pair**: not chosen (§6 of the prompt); this ran `main.py`'s defaults.
- **`QCD_Cosmology`**: out of scope (D10-1), which is why every reader exits 1.
- **A crash mid-replicated-write**: both interruptions landed in sharded stages, so the in-flight
  record was never left set by a real run; §2.5's case 4 exercises the repair on a hand-written one.
- **A `SIGKILL`** or power loss: not tried; a `SIGTERM` is what D10-3 decided.

**Every issue still open on the board, and what it means for A3 v2** (board §3):
- `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` — **blocks A3 v2's resume as
  planned**: an interrupted build resumed by `--build --resume` duplicates rows. Owned by `handover`.
- `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]` — an interrupted A3 v2
  launch is indistinguishable in the registry from a real failure, or reads as stale, and may
  orphan a `gcs_server`.
- `[10-extraction-scripts-exit-1-on-a-store-without-every-model]` — every extraction from A3 v2
  exits 1 after reading LambdaCDM; its outputs are written, but a caller that checks the status sees
  a failure.
- `[10-distinct-lines-test-cannot-fail-for-three-labelled-classes-on-the-fixture]` — none; a test's
  reach on its fixture.
- `[02-the-version-row-is-written-outside-the-in-flight-record]` — A3 v2's first open writes the
  `version` row outside the recorded write; a crash in that instant leaves a store that refuses to
  open. Not reached here.
- `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` — the
  `QuadSourceIntegral` reader can write to A3 v2's store; here it changed nothing (§2.3).
- `[09b-gksource-members-are-not-checked-against-their-rows]` — inert in the pipeline, where
  `compute()` asserts the set.
- `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`,
  `[07-drop-actions-list-a-parent-table-before-its-children]` — none on a build.
- `[00-the-inventory-cannot-see-a-serial-split]` — the fingerprint cannot see a split; the census
  can, and is clean here. Run the census on A3 v2.
- `[00-oneloop-factory-cannot-round-trip-a-row]`, `[00-the-5pt0-policy-pair-is-created-and-never-used]`
  — none: `OneLoopIntegral` is not built, and the unused pair is two rows.

---

## 3. The one test change

`Datastore/tests/test_inventory_report.py::test_verbose_renders_every_distinct_record_as_a_distinct_line`
now compares record lines with a record's own label `  #<n>` taken off (`OWN_LABEL`, imported from
prompt 09d's test module). Its deliberate-breakage record is in the log. In short: under a mutation
that prints a labelled class's record without its fields or value count, the amended test fails for
`TkNumericIntegration`, `TkWKBIntegration`, `GkSource` and `GkSourcePolicyData` (4 sub-tests) and the
parent commit's test passes. `QuadSource`, `GkNumericIntegration` and `GkWKBIntegration` cannot fail
on this fixture, which holds one record per tag set of each (§5).

---

## 4. Commands, verbatim

Besides those quoted above:

```bash
PYTHONPATH=. ./venv/bin/python -m RunRegistry list
PYTHONPATH=. ./venv/bin/python -m RunRegistry store show var/datastores/datastore-integrity-10-<store>.sqlite
timeout 60 ./venv/bin/ray status
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" var/datastores/datastore-integrity-10-<store>.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint [--write] var/datastores/datastore-integrity-10-<store>.sqlite
sqlite3 "file:<file>?mode=ro" "PRAGMA integrity_check;"
sqlite3 "file:<file>?mode=ro" "PRAGMA foreign_key_check;"
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/stage_counts.py var/runs/datastore-integrity-10-<run> ...
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/qsi_resolution_cost.py <primary> var/runs/datastore-integrity-10-<run> ...
bash docs/datastore-integrity-verification/interrupter_dummy_test.sh <work dir outside var/>
PYTHONPATH=. ./venv/bin/python -c "import RunRegistry; RunRegistry.Run('var/runs/datastore-integrity-10-resumed-b-20261001T230813').finish('killed', exit_code=None)"
```

---

## 5. Issues opened

On the board's §3.3, each with an index row; none fixed here.
- `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`, assigned to `handover`.
- `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`.
- `[10-extraction-scripts-exit-1-on-a-store-without-every-model]`.
- `[10-distinct-lines-test-cannot-fail-for-three-labelled-classes-on-the-fixture]`.

## 6. What stays under `var/`

- The two stores and their sidecars, in `var/datastores/`.
- The five run directories and their launch logs, `var/runs/datastore-integrity-10-*`
  (`stdout.log`, `stderr.log`, `status.json`, `manifest.json`, and `interrupter.log` for `resumed-a`
  and `resumed-b`).
- `var/datastore-integrity-10/extract/`: each script's log and outputs, and `run_extractions.out`.
- `var/datastore-integrity-10/inventory/`: the four reports.
