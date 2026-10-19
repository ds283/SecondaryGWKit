# rrk-v020-adoption campaign — R3 and the hard-kill route

This is the verification record of the
[`rrk-v020-adoption`](../prompts/rrk-v020-adoption/README.md) campaign. It was created by **prompt 03**
on branch `handover-remedial` at the dispatch `HEAD` **`aab4075`** ("Write the orchestrator note for
rrk-v020-adoption prompt 03"), whose code is 02's (`2818780`).
- **§1, "R3 — through `runregistrykit` 0.2.0"**, is the four-launch rehearsal of
  `runregistry-extraction` R2 run again with the new pin. It has the shape of
  [`runregistry-extraction-verification.md`](runregistry-extraction-verification.md) §3 and is
  compared with it.
- **§2, "The hard-kill route on the A3 v2 launcher"**, is the first `SIGKILL` of a rehearsal build:
  `hardkill-a`, the resume `begin` refuses, `end`, the sidecar and `hardkill-b`, on a third store.
  It also records what Ray's processes did after the kill.
- **04 adds §3**, the close.

The small outputs quoted below are in [`rrk-v020-adoption/rehearsal/`](rrk-v020-adoption/rehearsal/),
under names carrying `03`. The large ones stay under `var/` (§1.9).

## 1. R3 — through `runregistrykit` 0.2.0

**§1's tree, stores and runs.** Every R3 launch ran at **`aab4075`**.
- **Its code is `2818780`'s** (prompt 02). `git log 2818780..aab4075` is `b84c53d` and `aab4075`,
  and `git diff --stat 2818780 aab4075 -- . ':!prompts' ':!docs'` and `-- docs` are both empty.
- **What it changes over R2's tree (`b9daf44`)** in what a build runs is the pin, and nothing else:
  - `requirements.txt:81` is `runregistrykit[stores] @ …@v0.2.0` (01, `0a27b0c`);
  - `main.py`'s and the six readers' `--ray-address` default is `local` (`359f3fd`). No rehearsal
    reaches it: the launcher forces `address="local"`, and `run_readers.sh` passes `local`;
  - the launcher, `docs/handover/quadsource_atol_sweep.py`, is unchanged since `7af17bd`. Its hash
    is R2's, `2a985303…`.
- **The packages.**
  - The registry is the installed **`runregistrykit` 0.2.0** at `0bae3a3c…`, not editable. It
    registers the runs, writes each `status.json` through its new status writer, takes the
    fingerprints at finish and writes the sidecars.
  - The datastore layer is **`datastorekit` 0.4.0** at `c8570e84…`, and the work queue is
    **`raypoolkit` 0.2.0** at `eaef0ce6…`. Both are not editable, and both are as at R2.
- **The tree was clean.** `git status --short` was empty at every launch, checked again in the same
  command that launched.
- **The manifests.** Every manifest stamps `git_dirty: false` and `packages`
  `{"datastorekit": {"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false, "requested_revision": "v0.4.0", "url": "https://github.com/ds283/DatastoreKit", "version": "0.4.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}, "runregistrykit": {"commit": "0bae3a3c501a8264ee4be5bc24f06422af38720f", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RunRegistryKit", "version": "0.2.0"}}`.
  This is compared as JSON. The `datastorekit` and `raypoolkit` entries are R2's exactly.
- **The host.** `runregistrykit.this_host()` read `id` `8d2b247824882486`, `name`
  `mps026088.phys.susx.ac.uk`, `boot` `376461AF-4985-4F42-8E41-F4E914F5E3E6`, `machine` `arm64`,
  `system` `Darwin`, `release` `27.0.0` and `python` `3.12.15`. The `id` and `boot` are 01's reading,
  so there has been no reboot since 01.
- **The lock** was taken before `reference`. It was held through the readers, the guard and the
  hard-kill pair, and released (§1.1.3).

R3's launcher built two throwaway stores for §1:
- **`var/datastores/rrk-v020-adoption-03-reference.sqlite`** (sidecar `store_id`
  `21153431c18c4f1daf30210c8dbec2f8`), built straight through by the run **`reference`**;
- **`var/datastores/rrk-v020-adoption-03-resumed.sqlite`** (`store_id`
  `86482c5d57f1473297efcaef7bf64909`), built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed by `SIGTERM` in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

The third store, `…-hardkill.sqlite`, is §2's. No store is retired: the user retires R3's three
after 04's review, or not (README §3). R2's stores are retired (`5db42bf`). R2 is compared through
the fingerprints its `reference` and `resumed-c` run directories recorded, and through its run
directories and saved outputs for everything else.

**Headline.**
1. **R3 reproduces the reference fingerprint through `runregistrykit` 0.2.0.**
   - Both stores are format 6, **`encoding_format` 1**, digest
     `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, with all 21 classes
     equal. Each equals its sidecar's.
   - `compare_fingerprints` is empty between the two stores. It is also empty between each store
     and four recorded fingerprints: R3's own `reference` and `resumed-c`, and **R2's**
     `reference` and `resumed-c` (§1.3.4).
   - The only change to what runs between R2 and R3 is the pin, so the pin moves nothing that a
     build writes or a fingerprint reads.
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0, and `resumed-a`
   and `resumed-b` are `killed`/143. Each carries a fingerprint with `encoding_format` 1, and each
   driver exited as recorded (§1.2). No `status.json.*.tmp` was left by either `SIGTERM` handler's
   `finish`.
3. **Every manifest and every status carries this machine's `host`** (§1.2.4).
4. **The resume recomputed nothing it had stored, and the prune at open was exercised again**
   (§1.3).
   - `resumed-a` stored 168 `GkSource` rows, 48 of them unvalidated. `resumed-b`'s open pruned
     exactly those 48.
   - `resumed-b` computed 171 = 291 − 168 + 48 `GkSource` rows.
   - `resumed-c` computed the 683 = 864 − 181 `QuadSourceIntegral` rows that were missing.
5. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§1.4). The lookups are R2's.
6. **L4 equals R2's** (§1.5).
   - The statistics counts are R2's, and each block is byte-identical to R2's same slug.
   - The reference stage counts, the chain's identities, the stage banners and the printed lines
     are R2's.
7. **The one line naming `runregistrykit`** in the twelve build logs of §1 is the launcher's banner,
   as at R2. No line names `datastorekit`. No line holds a `DeprecationWarning`, a `ConfigError`, a
   traceback or a refusal. **R2's raylet disk-space warning is absent**: the volume is 93 % used,
   under Ray's 95 % threshold (§1.2.5).
8. **Nothing else under `var/` moved** (§1.9). `list` shows 51 runs, with nothing `running` or
   stale (§1.9).

### 1.1 How each was launched

Each launch was R2's command (`runregistry-extraction-verification.md` §3.1), with
`rrk-v020-adoption-03`, `--campaign rrk-v020-adoption --prompt 03` and this campaign's store names.
It was run from the repository root, detached, with the purpose passed to `sh -c` as its `$0`:

```bash
nohup sh -c "PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/rrk-v020-adoption-03-<store>.sqlite \
    --job-name rrk-v020-adoption-03 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose \"\$0\" --campaign rrk-v020-adoption --prompt 03; \
  echo \"** driver exit status \$?\"" "<one line>" \
  > var/runs/rrk-v020-adoption-03-<slug>.launch.log 2>&1 &
disown
```

The Python `argv` is R2's form exactly (`ps` of each driver, `prelaunch_checks_03.txt`). Each
`--purpose` names:
- R3 of this campaign;
- the tree's commit and its code's (`aab4075`, `2818780`);
- `runregistrykit` 0.2.0 at `0bae3a3`, `datastorekit` 0.4.0 at `c8570e8` and `raypoolkit` 0.2.0;
- the fingerprint it must equal, `39809dca`, and R2's run that recorded it,
  `runregistry-extraction-08-reference-20261015T235716`.

The purposes are in [`run_records_03.txt`](rrk-v020-adoption/rehearsal/run_records_03.txt).

**One check per launch, one wait per launch.** After each launch the agent made one check, and in
the same check ran `tools.rehearsal_lock set-pid` with the driver's pid (§1.1.3). The check
covered:
- the manifest, with its `packages`, `git_dirty`, `git_head`, `script_sha256`, keys and `host`;
- `status.json`, which had to say `running`, with a pid answering `kill -0` and a `host` whose `id`
  and `boot` are this machine's;
- the run's row in `./venv/bin/python -m runregistrykit list`;
- for `resumed-a` and `resumed-b`, the interrupter's launch.

The agent then ended its turn. The orchestrator ran one wait per launch, which ended on the driver
pid's death, and resumed the agent with the final `status.json`. Neither of them polled. Every wait
ended with the pid dead and the heartbeat never stale: `ENDED done`, `ENDED killed`,
`ENDED killed`, `ENDED done`.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load at launch / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `rrk-v020-adoption-03-reference-20261018T232838` | reference | no | 90795 (90789) | `done` | 0 | 0 | 3 m 52 s (23:28:38–23:32:30) | 5.80 10.22 11.29 / 7.34 10.22 11.19 |
| `resumed-a` | `rrk-v020-adoption-03-resumed-a-20261018T233357` | resumed | no | 93122 (93116) | `killed` | 143 | 143 | 2 m 18 s (23:33:57–23:36:15) | 18.62 12.49 11.95 / 13.57 17.72 14.68 |
| `resumed-b` | `rrk-v020-adoption-03-resumed-b-20261018T233734` | resumed | yes | 94708 (94702) | `killed` | 143 | 143 | 1 m 19 s (23:37:34–23:38:53) | 11.93 17.07 14.52 / 15.07 15.86 14.30 |
| `resumed-c` | `rrk-v020-adoption-03-resumed-c-20261018T233941` | resumed | yes | 95847 (95843) | `done` | 0 | 0 | 1 m 46 s (23:39:41–23:41:27) | 15.08 15.80 14.31 / 29.36 20.36 16.30 |

- **Wall time** runs from the manifest's `created` to the final heartbeat.
- **Load** is `uptime`'s on 10 CPUs: at the launch, and at the agent's first check after the run.
  The one-minute load ran from 6 to 29 at these checks, and reached 74 at the end of the readers.
  R2's ran from 6 to 19.

**The sidecars** ([`store_sidecars_03.txt`](rrk-v020-adoption/rehearsal/store_sidecars_03.txt)):
- **The reference store.** `store create` ran on its closed primary at 23:33:28, after `reference`.
  `store fingerprint --write` followed: `39809dca…`, problems none, "it replaced none".
- **The resumed store.** `store create` ran on its closed primary at 23:37, after `resumed-a`. The
  store's five files were unchanged by it.

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`. The commands are R2's:

```bash
./venv/bin/python -m runregistrykit store create --purpose "<one line>" \
    var/datastores/rrk-v020-adoption-03-reference.sqlite
./venv/bin/python -m runregistrykit store fingerprint --write \
    var/datastores/rrk-v020-adoption-03-reference.sqlite
./venv/bin/python -m runregistrykit store create --purpose "<one line>" \
    var/datastores/rrk-v020-adoption-03-resumed.sqlite
```

#### 1.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_03.txt`](rrk-v020-adoption/rehearsal/prelaunch_checks_03.txt)):
- **The tree.**
  - `git rev-parse HEAD` was `aab407529576307c9d3c1e64ccc8b2e980ed2d7d`.
  - `git status --short` was empty.
  - `git worktree list` showed this checkout alone.
  - Every working note and capture went under `var/rrk-v020-adoption-03/`, which is ignored, or
    under the agent's scratchpad, until `hardkill-b` had ended. This record's files were written
    afterwards.
- `RAY_ADDRESS` was unset. The machine has 10 CPUs.
- **The registry.** `./venv/bin/python -m runregistrykit list` showed nothing `running` or stale.
  - Before `reference` it showed 45 runs: "43 finished, 2 unknown; **45 with no host recorded**".
    That is 01's reading.
  - It then showed 46, 47 and 48 runs, all finished but the `a3-pilot` pair, and still "45 with no
    host recorded". R3's runs are not among the 45.
- **No Ray process of this repository.**
  - Before each launch, `tools.rehearsal_lock ray-processes` exited 0 and listed nothing.
  - The machine-wide `pgrep -lf 'gcs_server|raylet|ray::'` was empty (rc 1) every time.
  - Before each later launch, `ps -p` checked the previous run's pids: the `sh` and Python pids,
    the interrupter's, and every Ray pid its logs name. None was alive.
- **The lock** was free before `acquire`, and `~/ScienceData/` held `ChamPBH-rehearsals/` alone.
- **The store names.**
  - Before `reference`, `./venv/bin/python -m runregistrykit store show` on all three names gave
    kind `absent`, problems none, and no run naming any, exit 0.
  - Nothing under `var/datastores/` or `var/runs/` contained `rrk-v020`.
  - There was no `-journal`, `-wal` or `-shm` file under `var/datastores/` before any launch.
- **The packages.**
  - `./venv/bin/pip show` gave `runregistrykit` 0.2.0, `datastorekit` 0.4.0 and `raypoolkit`
    0.2.0, each with `Location` this venv's `site-packages` and no editable location. `pip check`
    was clean.
  - `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))` gave the
    record above, equal as JSON to the orchestrator's. Its `datastorekit` and `raypoolkit` entries
    equal R2's reference manifest's.
  - `TERMINAL_STATES` is `('done', 'failed', 'killed', 'lost')`.
  - The pins are `requirements.txt:17` (`@v0.4.0`), `:76` (`@v0.2.0`) and `:81`
    (`runregistrykit[stores] @ …@v0.2.0`).
- **The configuration.** `runregistrykit.find_client_root()` read `runregistrykit.toml`, with
  `runs_root` `var/runs`, `stores_root` `var/datastores`, `distributions`
  `('datastorekit', 'raypoolkit')` and `stores` `{'fingerprinter': 'config.fingerprint:fingerprinter'}`.
- **What runs.**
  - The three packages' `__file__` are under the venv's `site-packages`.
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`.
  - `ObjectFactories.__file__` is the repository's.
  - `ENCODING_FORMAT` is 1 and `FINGERPRINT_FORMAT` is 6.
  - `ray.is_initialized()` is false.
- **The recorded fingerprints R3 is compared with.** The `status.json` of R2's `reference` and
  `resumed-c` are each `done`/0, format 6, `encoding_format` 1, `39809dca…`, 21 classes, problems
  `{}`, with no `host`.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` was
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`. Its last change is `7af17bd`.
- **`var/datastores/`.** It held 21 files, all tombstone sidecars, one of them under
  `backup-pre-resume-20260921T091011/`. The digest over them was `7ad566f4…`, the orchestrator's.
  The full listing is in §1.9.
- **Disk.** 72 GB was free on `var/`'s volume (93 % used), against R2's 44 GB (96 %).

#### 1.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch.** Each ran into a new directory under the
agent's scratchpad, with R2's commands. Both exited 0
([`interrupter_dummy_test_03.txt`](rrk-v020-adoption/rehearsal/interrupter_dummy_test_03.txt),
[`interrupter_build_dummy_test_03.txt`](rrk-v020-adoption/rehearsal/interrupter_build_dummy_test_03.txt)).
They ran 23:26:46–23:27:12.
- **Against R2's and 02's**, with times, pids and scratch paths stripped:
  - the scoped test's output (40 lines) equals R2's `interrupter_dummy_test_08.txt` and 02's
    `measure/out/02-dummy-scoped.txt`, line for line;
  - the build test's output (28 lines) equals 02's `measure/out/02-dummy-build.txt`;
  - its first 20 lines, cases A–C, equal R2's `interrupter_build_dummy_test_08.txt`.
- **Case D**, new in 02, gave interrupter exit 0, `child exit=137 (137 = killed by SIGKILL)`, and
  the log line `sent SIGKILL to pid …`.
- **No dummy child or interrupter was left alive.** `ps -p` on all 18 pids the two outputs name
  found none.

**The points, from R3's own reference `stdout.log`, sized as the orchestrator's note directs.**
Both queues have a 300 s minimum notify interval (`main.py:119`).
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **24.8 s**
  (`stdout.log:76`; R2: 27.4 s). It was interrupted from the banner with `--interval 5 --hold 10`.
  40 % of 24.8 s is 9.92 s, which rounds to 10 s, equal to the floor.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **1 m 31.4 s (91.4 s)**
  (`stdout.log:100`; R2: 119.8 s). It was interrupted from the banner with `--interval 5 --hold 30`.
  A third of the stage is 30.47 s, rounded to 30 s, under the 250 s cap. The pattern is
  `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its end.

**The real firings** ([`interrupter_firings_03.txt`](rrk-v020-adoption/rehearsal/interrupter_firings_03.txt)):

```
resumed-a (interrupter pid 93128; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10)
2026-10-18T23:34:43+01:00 liveness alive; pid 93122; stage 'CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS'
2026-10-18T23:36:03+01:00 liveness alive; pid 93122; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-18T23:36:03+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 10 s, then reading again
2026-10-18T23:36:13+01:00 sent SIGTERM to pid 93122 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command "…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/rrk-v020-adoption-03-resumed.sqlite … --register resumed-a …"; exiting

resumed-b (interrupter pid 94723; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 30)
2026-10-18T23:38:10+01:00 liveness alive; pid 94708; stage 'APPLY GKSOURCE POLICIES'
2026-10-18T23:38:20+01:00 liveness alive; pid 94708; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-18T23:38:20+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 30 s, then reading again
2026-10-18T23:38:50+01:00 sent SIGTERM to pid 94708 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command "…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/rrk-v020-adoption-03-resumed.sqlite … --register resumed-b …"; exiting
```

- **Each interrupter checked the pid before it signalled**, against its manifest's `argv[0]` and
  `--register <slug>`. It read its stage from `status.json`, which `runregistrykit` 0.2.0 writes.
  Both banners were read. No progress line was read: no stage of R3 ran past 300 s.
- The interrupter reads every 5 s and logs a reading only when the stage changes. So:
  - the `GkSource` stage began between 23:35:58 and 23:36:03, and the signal came 10–15 s into it;
  - the QSI stage began between 23:38:15 and 23:38:20, and the signal came 30–35 s into it.
- Both interrupters were launched detached, 1–2 s after their builds, once the run directory and
  its `status.json` existed, with R2's command and this prompt's run directories and holds.

#### 1.1.3 The lock

Every helper command and its exit status
([`lock_03.txt`](rrk-v020-adoption/rehearsal/lock_03.txt)). Each was run as
`PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock <command>` from the repository root. The
`status` calls inside each pre-launch check are in `prelaunch_checks_03.txt`.

| When | Command | Exit | What it said |
|---|---|---|---|
| 23:27:51 | `acquire --purpose "rrk-v020-adoption 03 (R3)"` | 0 | acquired, for SecondaryGWKit |
| 23:27:51 | `status` | 0 | `HELD`; no pid recorded yet; no Ray process |
| 23:28:45 | `set-pid 90795` (`reference`) | 0 | recorded |
| 23:28:45 | `status` | 0 | `pid: 90795`, alive; 18 Ray processes, the run's |
| 23:34:09 | `set-pid 93122` (`resumed-a`) | 0 | recorded |
| 23:34:09 | `status` | **not recorded** | `pid: 93122`, alive; 21 Ray processes. The record was cut short by the agent's display pipe; see the note in `lock_03.txt` |
| 23:37:46 | `set-pid 94708` (`resumed-b`) | 0 | recorded |
| 23:37:46 | `status` | 0 | `pid: 94708`, alive; 14 Ray processes |
| 23:39:52 | `set-pid 95847` (`resumed-c`) | 0 | recorded |
| 23:39:52 | `status` | 0 | `pid: 95847`, alive; 14 Ray processes |
| 23:42:15 | `status` | 0 | `pid: 95847`, NOT alive; no Ray process (before the readers) |
| 23:46:11 | `status` | 0 | the same (before `hardkill-a`) |
| 23:46:27 | `set-pid 99903` (`hardkill-a`) | 0 | recorded |
| 23:46:27 | `status` | 0 | `pid: 99903`, alive; 18 Ray processes |
| 23:59:29 | `status` | 0 | `pid: 99903`, NOT alive; **1 Ray process**, the GCS server 99918 (§2.3) |
| 00:11:52 | `set-pid 10646` (`hardkill-b`) | 0 | recorded |
| 00:11:52 | `status` | 0 | `pid: 10646`, alive; 14 Ray processes |
| 00:14:55 | `status` | 0 | `pid: 10646`, NOT alive; no Ray process |
| 00:14:55 | `ray-processes` | 0 | none |
| 00:14:55 | `release` | 0 | released |
| 00:14:55 | `status` | 0 | `lock: free`; no Ray process |
| 00:14:55 | `ray-processes` | 0 | none |

- The owner file named `project: SecondaryGWKit`, `purpose: rrk-v020-adoption 03 (R3)`,
  `ray_temp_dir: in-process` and `ray_address: in-process` throughout.
- `acquire` came 47 s before `reference`'s manifest (23:28:38).
- Between launches the lock was held, and its pid was the previous driver's, dead: the protocol,
  not a stale lock.
- From 23:59:15 to 00:10:05 the lock stayed held with a Ray process of this repository alive, by
  correction 10. `release` was run only after §2.9, with no Ray process left, and was not refused.
- Afterwards `~/ScienceData/` holds `ChamPBH-rehearsals/` alone, as before.

### 1.2 L1 — every launch ends as recorded

#### 1.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `encoding_format` 1, `39809dca…`, problems `{}` | not written: "…rrk-v020-adoption-03-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `encoding_format` 1, `c60e0d19…`, problems `{}` | not written: "…rrk-v020-adoption-03-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `encoding_format` 1, `7e41cbe2…`, problems `{}` | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `encoding_format` 1, `39809dca…`, problems `{}` | `written` | `** driver exit status 0` |

- Each fingerprint holds 21 classes. Each `taken` names its run, `git_head` `aab40752…` and
  `git_dirty` false.
- Each `status.json` has R2's nine keys and **`host`**. Its `fingerprint` has R2's keys, with
  `encoding_format`.
- The full digests are:
  - `reference` and `resumed-c`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
  - `resumed-a`: `c60e0d198e78ed4a360801ddb135f42012d638d606550ae34881639bf4834221`;
  - `resumed-b`: `7e41cbe2306c2b8b9ed4a02ddf21a1b1805c669cae0307486ebe0d680d0ba101`.
- The intermediate digests differ from R2's (`87e55e8f…`, `8978b72e…`), as they must: each digests
  a store killed at a different point. By their recorded class counts:
  - `resumed-a`'s store held `GkSource` 168, `GkSourcePolicyData` 0 and `QuadSourceIntegral` 0;
  - `resumed-b`'s held 291, 291 and 181.

#### 1.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 23:36:13 | 23:38:50 |
| Handler's line | `** SIGTERM at 2026-10-18 23:36:13 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-18 23:38:50 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| `stderr.log`'s second line | `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` | `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Fingerprint taken / final heartbeat | 23:36:13 / 23:36:15 | 23:38:50 / 23:38:53 |
| `status.json.*.tmp` left in the run directory | **none** | **none** |
| Stage at the signal | the `GkSource` banner, first read at 23:36:03; signal 10 s later (**168 of 291 rows stored, 48 unvalidated**) | the QSI banner, first read at 23:38:20; signal 30 s later (181 of 864 stored) |

- **Both ended `SystemExit(1)`.** R2's `resumed-b` ended `SystemExit(143)`. As R2 explains (its
  §3.2.2), a signal that lands inside `ray.wait` has Ray's signal check turn the handler's
  `SystemExit(143)` into `SystemExit(1)`, so which it is depends on where the signal falls. The
  latch is what `finish_after_exec` reads, so the record is `killed`/143 and the exit is 143 either
  way.
- **`v0.2.0`'s status writer under the handlers.** Each handler's `finish("killed", …)` went
  through the new writer. Each run directory holds `interrupter.log`, `manifest.json`,
  `status.json`, `stderr.log` and `stdout.log` and nothing else, so no temporary status file was
  left (release notes §2.3).
- The `stderr.log` of `resumed-a` and `resumed-b` holds the handler's two lines and nothing else.
  The `stderr.log` of `reference` and `resumed-c` holds the SciPy `IntegrationWarning` (4 lines), as
  at R2.

#### 1.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 23:33:08 (~38 s after the final heartbeat) | none: `ray-processes` exit 0, `pgrep` empty, pids 90789 and 90795 gone, and the Ray pids its logs name (90861, 90863, 90864) gone |
| `resumed-a` | 23:37:11 (~56 s) | none, by the same checks for 93116, 93122, the interrupter 93128, and the Ray pids 93170, 93172, 93173 |
| `resumed-b` | 23:39:22 (~29 s) | none, for 94702, 94708, the interrupter 94723, and the Ray pids 94755, 94757 |
| `resumed-c` | 23:42:08 (~41 s) | none, for 95843, 95847 and the Ray pids 95879, 95882, 96122 |

No Ray process outlived either `SIGTERM`'d driver, so no wait by pid was run, as at R0, R1 and R2.
§2.3 is the `SIGKILL`'s.

#### 1.2.4 The manifests and the hosts

From [`run_records_03.txt`](rrk-v020-adoption/rehearsal/run_records_03.txt), for all four (and for
§2's two):
- **`git_head`** is `aab407529576307c9d3c1e64ccc8b2e980ed2d7d`, the dispatch `HEAD`, and
  **`git_dirty`** is false.
- **`packages`** equals the record of §1's header, compared as parsed JSON (the orchestrator's
  correction 1): `runregistrykit` 0.2.0 at `0bae3a3c…` (`v0.2.0`), `datastorekit` 0.4.0 at
  `c8570e84…` and `raypoolkit` 0.2.0 at `eaef0ce6…`, none editable. On disk the keys are
  alphabetical, as every record is written with `sort_keys`.
- **`script_sha256`** is `2a985303…` in all six, R2's.
- **The manifests have twenty keys:** R2's nineteen and **`host`**.
  - The manifest's `host` holds `id`, `name`, `machine`, `system`, `release` and `python`, with no
    `boot` (release notes §2.1; the orchestrator's correction 12).
  - Its `id` is `8d2b247824882486` and its `name` `mps026088.phys.susx.ac.uk`, `this_host()`'s, in
    all six.
- **Every status's `host`** holds `id` `8d2b247824882486` and `boot`
  `376461AF-4985-4F42-8E41-F4E914F5E3E6`, `this_host()`'s, at every check and at every run's end.
- `argv` holds `--resume` in `resumed-b` and `resumed-c` only.
- `results_store_id` is `null` for `reference` and `resumed-a`, whose stores had no sidecar at
  launch, and `86482c5d…` for `resumed-b` and `resumed-c`, as at R2.

#### 1.2.5 Every launch's `stdout.log`, and what is new in the logs

Counted in each of the four, as R2's §3.2.5 counted them
([`log_comparison_03.txt`](rrk-v020-adoption/rehearsal/log_comparison_03.txt), section 1):

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4 |
| the rehearsal-scope banner; `main.py argv = …`; `bootstrapping a local Ray instance`; `Started a local Ray instance` (launch log) | 1 each |
| `--no-prune-unvalidated`; `reconcil`, `repair`, `in.flight`, `replicat` | 0 |
| `Traceback`, `ModuleNotFoundError`, `ImportError`, `AttributeError`, `pickle`, `Datastore.SQL`, `ray.kill`, `killed actor`, `ActorDiedError`, `RayActorError`, `ValueError`, `RuntimeWarning`, `did not validate` | 0 |
| `DeprecationWarning`, `_timestamp`, `ConfigError`, `RunRegistry` (not followed by `Kit`) | 0 |
| `datastorekit`, anywhere in a line | 0 |
| `runregistrykit`, anywhere in a line | 1 in `stdout.log` and 1 in the launch log: the banner's `**   list it with: ./venv/bin/python -m runregistrykit list`, and nothing else |
| `INTEGRITY WARNING` | 0, but for `resumed-b`: 2 in `stdout.log` and 2 in the launch log |
| raylet `file_system_monitor` (`… is over 95% full …`) | **0** in every log (R2: 26, 11, 8, 9 in `stderr.log`) |

The launch logs of `resumed-b` and `resumed-c` hold `** quadsource_atol_sweep --build: resuming
into rrk-v020-adoption-03-resumed.sqlite`.

**The comparison with R2, slug by slug.** R2's method was used: each line normalised (ANSI codes,
campaign and prompt, repository path, Ray's `[repeated Nx …]` count and every number replaced) and
compared with R2's same file for the same slug, as a set of line kinds. The script was a throwaway in
the agent's scratchpad, not committed (section 2 of `log_comparison_03.txt`).

| Slug | `stdout.log` | `stderr.log` | launch log |
|---|---|---|---|
| `reference` | 110 lines against 110; **no kind differs** | 4 against 30; R2 has the disk-space kind | 116 against 142; R2 has the disk-space kind |
| `resumed-a` | 84 against 84; no kind differs | 2 against 13; the disk-space kind | 88 against 99; the disk-space kind |
| `resumed-b` | 139 against 131; no kind differs | 2 against 10; the disk-space kind | 144 against 144; the disk-space kind |
| `resumed-c` | 107 against 107; no kind differs | 4 against 13; the disk-space kind | 114 against 123; the disk-space kind |

**Every difference, explained.**
1. **The raylet's disk-space warning is gone.** R2's raylets warned every 10 s that the volume
   holding `/tmp/ray` was over 95 % full (96 % then). R3's volume is 93 % used (72 GB free), under
   the threshold, so no raylet warned. This is the orchestrator's correction 5's expectation.
2. **`resumed-b`'s `stdout.log` is 8 lines longer.** Its integrity warning lists the rows the prune
   removed. Shard 3's block lists its 13 rows in full, two lines each, where R2's listed 9. Ray
   folds shard 0's 18 and shard 1's 17 into one block with `[repeated 35x across cluster]`, as R2's
   35 were. 13 + 35 = 48, the rows `resumed-a` left unvalidated (§1.3.2). The line kinds are R2's.
3. **No new line names `runregistrykit`**: the banner line is R2's. The pin added no line to a build
   log, no warning and no refusal.

### 1.3 L2 — the resume is the reference

#### 1.3.1 Nothing stored is recomputed

Through `dk-and-rpk-migration`'s
[`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py), as it stands, on this
prompt's six run directories ([`stage_counts_03.txt`](rrk-v020-adoption/rehearsal/stage_counts_03.txt)).
Lookup / compute / store:

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **171/171/171** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **683/683/683** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored **168** rows before the signal (58 + 55 + 0 + 55 by shard), of which **48**
    were unvalidated (18 + 17 + 0 + 13) (§1.3.2).
  - `resumed-b` pruned those 48 at open and computed **171 = 291 − 168 + 48**: the 123 never
    stored, and the 48 pruned.
  - Summed, less the pruned: 168 + 171 − 48 = 291, the reference's.
- **`QuadSourceIntegral`.** `resumed-b`'s fingerprint, taken by `finish` on the closed store,
  records **181** rows. `resumed-c` computed **683** = 864 − 181. Summed: 864.
- The `reference` and `resumed-a` columns equal R2's stage by stage.

#### 1.3.2 The prune at startup

The closed store after each launch was read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`, with R2's command
([`closed_store_state_03.txt`](rrk-v020-adoption/rehearsal/closed_store_state_03.txt)). Each file's
SHA-256, size and `mtime_ns` were unchanged by the read.

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **`GkSource` 18 + 17 + 0 + 13 = 48**, of 58 + 55 + 0 + 55 = 168 stored, each listed by serial; `GkNumericIntegration` 1368 and `GkWKBIntegration` 3477 all stored and validated |
| `resumed-b` | none | 0 rows | none (`GkSource` 97 + 97 + 0 + 97 = 291) |
| `resumed-c` | none | 0 rows | none |

`resumed-a`'s signal came 10–15 s into a stage the reference took 24.8 s over. By then 168 of 291
rows were stored and 48 were not yet validated. (R2's left 165 and 44, R1's 207 and 48.)

**By serial**, through `dk-and-rpk-migration`'s
[`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py), unchanged
([`prune_check_03.txt`](rrk-v020-adoption/rehearsal/prune_check_03.txt)):
- **The check against the state after `resumed-a` ran twice:** at 23:39, after `resumed-b` had
  opened and closed the store, and at 23:42, after `resumed-c`. Both times it gave
  `48 serial(s) checked: 48 absent, 0 reused by a validated row, 0 still unvalidated`, exit 0,
  with 18 serials in shard 0, 17 in shard 1 and 13 in shard 3, each `absent`.
- **The check against the state after `resumed-b`**, after `resumed-c`, gave `none: there is
  nothing to check`, exit 0.
- Every file of the store had the same SHA-256 before and after each check.

**So the prune at open removed exactly the rows `resumed-a` left unvalidated**, and no validated row.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§1.3.5), and each finish's fingerprint has `problems` `{}`.

#### 1.3.3 The version row

After every launch there is exactly one row, `(1, '2025.1.1')`, on each of the four shards of each
store. The primary has no `version` table.

#### 1.3.4 The comparison

**The two stores**, by `compare_stores.py` with R2's command
([`fingerprint_comparison_03.txt`](rrk-v020-adoption/rehearsal/fingerprint_comparison_03.txt)).
It exited 0:
- both stores are format 6, digest `39809dca…`, problems none, each with 0 differences against its
  sidecar;
- `compare_fingerprints(reference, resumed): 0 difference(s)`, verdict `equal`;
- all 21 classes are `equal`, with R2's counts (its §3.3.4 table).

**Against the recorded fingerprints**, by `compare_with_recorded.py` as it stands
([`fingerprint_against_recorded_03.txt`](rrk-v020-adoption/rehearsal/fingerprint_against_recorded_03.txt)).
R2's fingerprints carry `encoding_format` 1, so the package's comparison applies to them directly
(README §6.2 C7):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/rrk-v020-adoption-03-reference.sqlite \
    var/datastores/rrk-v020-adoption-03-resumed.sqlite \
    --against var/runs/rrk-v020-adoption-03-reference-20261018T232838 \
    var/runs/rrk-v020-adoption-03-resumed-c-20261018T233941 \
    var/runs/runregistry-extraction-08-reference-20261015T235716 \
    var/runs/runregistry-extraction-08-resumed-c-20261016T000959
```

It exited 0, verdict `equal`. For each store there were 0 differences against each of:
- its sidecar;
- R3's `reference` and `resumed-c` (taken at `aab40752…`, `git_dirty` false);
- **R2's `reference` and `resumed-c`** (taken at `b9daf449…`, `git_dirty` false).

All are format 6, digest `39809dca…`, 21 classes, problems none. The files of both stores had the
same SHA-256, size and `mtime_ns` before and after these reads.

#### 1.3.5 Integrity, both stores, read-only

**The census**, with R2's command ([`census_03.txt`](rrk-v020-adoption/rehearsal/census_03.txt)).
Every table is `OK` on both stores (28 lines), the verdict is `no divergence` on both, and the exit
is 0.

**`integrity_check`, `foreign_key_check` and the inventory**, through `store_soundness.py` as it
stands ([`store_soundness_03.txt`](rrk-v020-adoption/rehearsal/store_soundness_03.txt)):
- it exited 0, `=> clean`;
- on the primary and every shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is
  `['ok']` and `PRAGMA foreign_key_check` returns 0 rows;
- `read_inventory` reads shards `(0, 1, 2, 3)` and reports no problem in any class, with 0
  `duplicate`;
- every file of both stores had the same SHA-256 before and after these reads, and no journal
  appeared.

### 1.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance
(`--ray-address local`). They ran through `run_readers.sh`, unchanged, with R2's two commands and
this prompt's store and output directory (`var/rrk-v020-adoption-03/extract`), the lock held
([`readers_03.txt`](rrk-v020-adoption/rehearsal/readers_03.txt)). Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written (R2) | Traceback / `ReadOnly*` | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 24 s | 116 | 468 (531) | none | identical |
| `extract_GkWKB_data.py` | 0 | 56 s | 348 | 2436 (2436) | none | identical |
| `extract_GkSource_data.py` | 0 | 19 s | 10 | 156 (176) | none | identical |
| `extract_TkWKB_data.py` | 0 | 13 s | 3 | 27 (27) | none | identical |
| `extract_tensor_source_data.py` | 0 | 11 s | 6 | 27 (27) | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 14 s | 24 | 70 (71) | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 24 s | LambdaCDM 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 378 (423) | none | identical |

- **The lookups equal R2's**, queue by queue and reader by reader.
- **The files written differ** in the three readers that draw random subsamples, as between every
  earlier pair. It is not a finding.
- **The reader logs' line kinds against R2's** (section 6 of `log_comparison_03.txt`):
  - three are equal: `extract_TkWKB_data`, `extract_tensor_source_data` and
    `extract_QuadSourceIntegral_data`;
  - the other four differ only by R2's raylet disk-space line, absent here (§1.2.5).
- No reader log names `runregistrykit` or `datastorekit`.
- The readers ran 23:42:21–23:45:08, at a one-minute load of 31–74.
- **No file of the store changed.** Before and after each reader, the SHA-256, mtime and size of
  the primary and of every shard were identical, and no journal appeared. No Ray process of this
  checkout was left after any reader.
- **The fingerprint afterwards.** `./venv/bin/python -m runregistrykit store fingerprint` on the
  resumed store gave `39809dca…`, problems none, "matches the fingerprint taken
  2026-10-18T23:41:25+01:00 by run rrk-v020-adoption-03-resumed-c-20261018T233941, at
  aab407529576307c9d3c1e64ccc8b2e980ed2d7d", exit 0.

**A3 v2's guard**, after the readers, in the foreground, with no `--register`, the lock still held
([`guard_03.txt`](rrk-v020-adoption/rehearsal/guard_03.txt)). It exited 1 within a second
(23:46:05), raising R2's `RuntimeError` from `quadsource_atol_sweep.py:537`. The error names the
five files of `rrk-v020-adoption-03-resumed.sqlite` and refuses to build into an existing store.
Nothing changed: every file of both stores and both sidecars is identical before and after.
Nothing was created: `var/runs/` held 91 entries before and after, and there was no Ray process and
no journal.

### 1.5 L4 — equal to R2

#### 1.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

`policy_statistics.py` was run on each run. It exited 0 on `reference`, `resumed-b`, `resumed-c`,
`hardkill-a` and `hardkill-b`, and 1 on `resumed-a`, which has no block, as at R2. The reference's
block ([`policy_statistics_03_reference.txt`](rrk-v020-adoption/rehearsal/policy_statistics_03_reference.txt)):

```
** GkSourcePolicyData SUMMARY STATISTICS

     * Policy: policy="maximize-WKB"-Levin-threshold="1.5"
        Total instances: 291

        >> numeric: 109 instances = 37.46%
           complete: 109 instances = 100.00% of this type, 37.46% of total

        >> mixed: 26 instances = 8.93%
           complete: 25 instances = 96.15% of this type, 8.59% of total
           minimal: 1 instances = 3.85% of this type, 0.34% of total

        >> WKB: 156 instances = 53.61%
           complete: 156 instances = 100.00% of this type, 53.61% of total
```

| | R2 | R3 |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| `reference` | `complete` first | **byte-identical to R2's `reference`** |
| `resumed-b` | `complete` first | byte-identical to R2's `resumed-b` |
| `resumed-c` | `minimal` first | byte-identical to R2's `resumed-c` |

**Every count is R2's, and every block is R2's same slug byte for byte.** §2's two blocks are in
§2.8.

#### 1.5.2 The reference run's stage counts

The `reference` column of §1.3.1 equals R2's `stage_counts_08.txt` in every stage, and so does the
`resumed-a` column.

#### 1.5.3 The resumed chain, by identity

| Identity | R2 | R3 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated | 44 = 44, all absent, none reused | **48 = 48, all absent, none reused** | ✅ |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | 170 = 291 − 165 + 44 | **171 = 291 − 168 + 48** | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 567 = 864 − 297 | **683 = 864 − 181** | ✅ |
| Totals per class over the chain equal the reference's | 291; 864 | `GkSource` 168 + 171 − 48 = 291; QSI 181 + 683 = 864 | ✅ |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 1.5.4 The stage sequence

The lines of each `stdout.log` that begin `** ` are in section 3 of `log_comparison_03.txt`.
- R3's and R2's hold 26, 23, 26 and 26 such lines, slug by slug.
- Normalised as §1.2.5, each slug's sequence equals R2's, in order, with no exception. The banner's
  `list it with:` line is R2's.
- The stage banners (`** CALCULATE …`, `** APPLY …`, `** BUILD …`) equal R2's verbatim, in order,
  in all four, with nothing normalised.

#### 1.5.5 The stage times

These are not compared (`ALL WORK ITEMS COMPLETE in time …` in each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 26.2 s | 34.3 s | 1.74 s | 1.61 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 1 m 6.82 s | 1 m 18.6 s | 3.64 s | 3.33 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **24.8 s** | killed | 21.4 s | 5.15 s |
| APPLY GKSOURCE POLICIES | 9.67 s | — | 10.4 s | 4.98 s |
| QUADRATIC SOURCE INTEGRALS | **1 m 31.4 s** | — | killed | 1 m 23.4 s |

- Each earlier stage took under 1 s.
- **The reference took 3 m 52 s**, against R2's 4 m 33 s and R1's 13 m 57 s. The `GkSource` stage
  took 24.8 s (R2 27.4 s) and the QSI stage 91.4 s (R2 119.8 s).

#### 1.5.6 The printed lines and the registry's fields

| Line, in `stdout.log` (R3 / R2) | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| `** ` | 26 / 26 | 23 / 23 | 26 / 26 | 26 / 26 |
| `-- ALL WORK ITEMS COMPLETE in time …` | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `Queue summary: …` | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `(no compute items were evaluated - all assets read from datastore)` | 1 / 1 | 1 / 1 | 7 / 7 | 9 / 9 |
| progress lines, `inflight` lines | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

No progress line was printed: no stage ran past 300 s.

**The registry's fields at each run's end** are R2's: `state`, `stage`, `units_done` and
`units_total`. Each `status.json` has R2's keys and `host`.

| Slug | `state` | `stage` | `units_done` / `units_total` |
|---|---|---|---|
| `reference` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` |
| `resumed-a` | `killed` | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | 0 / `null` |
| `resumed-b` | `killed` | `CALCULATE QUADRATIC SOURCE INTEGRALS` | 0 / `null` |
| `resumed-c` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` |

### 1.6 What was expected not to happen

Each of these was looked for, and none happened:
- an unvalidated `BackgroundModel` (§1.3.2);
- an in-flight record left by a killed launch (`replication_in_flight` was empty after every launch);
- a repair by the check at open (§1.2.5);
- a Ray agent outliving a `SIGTERM`'d driver (§1.2.3);
- a line from the pin: a `runregistrykit` or `datastorekit` frame, a `ConfigError`, a
  `DeprecationWarning` or a refusal (§1.2.5);
- a temporary status file left by a handler's `finish` (§1.2.2);
- a record written outside `var/runs/` and `var/datastores/` (§1.9);
- a missed interruption: both killed launches ended `killed`, in the stage their interrupter waited
  for;
- a held lock, or a refused `release` (§1.1.3).

### 1.7 Issues opened

None by §1. Every difference from R2 is explained by:
- the machine: the disk-space warning is gone at 93 %;
- where a launch was killed: the counts of §1.3, and `SystemExit(1)` in `resumed-b`;
- the readers' random subsamples.

§2 opens one issue (§2.10).

### 1.8 The suites

After the last launch, the checks and the release, each root was run with README §8's command, in
the foreground and serially. Each output went to the agent's scratchpad and its verdict was taken by
`grep`.

| Root | Ran | Result | Load (1 min) at start | Wall |
|---|---|---|---|---|
| AdaptiveLevin | 32 | OK | 9.82 | 0 s |
| CosmologyModels | 39 | OK | 9.82 | 2 s |
| ObjectFactories | 279 | OK | 9.19 | 14 s |
| tools | 91 | OK | 8.45 | 10 s |
| LiouvilleGreen | 148 | OK (1 skipped) | 8.76 | 25 s |
| RunDrivers | 103 | OK | 16.78 | 5 s |
| ComputeTargets | 700 | OK | 15.84 | 139 s |

Every count is 02's. No test failed, so none of README §8's four known flakes needed a re-run. No
Python file was added or changed.

### 1.9 What stays under `var/`

- **The run directories:**
  - `var/runs/rrk-v020-adoption-03-{reference,resumed-a,resumed-b,resumed-c,hardkill-a,hardkill-b}-<stamp>/`;
  - their seven launch logs beside them, the refused launch's
    `rrk-v020-adoption-03-hardkill-b-refused.launch.log` among them;
  - the refused launch made no run directory.
- **The stores:** `var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite`, each
  with four shards and a `.manifest.json` sidecar. None is retired.
- **`var/rrk-v020-adoption-03/`**, which holds:
  - the agent's working notes (`NOTES.md`), the pre-launch captures, the lock record, the dummy-test
    outputs, the hard-kill record, and the listings of `var/datastores/`;
  - the closed-store states, the prune checks and the hashes (`state/`), with `hardkill-a`'s
    `status.json` before and after `end` and its `interrupter.log`;
  - the check outputs (`checks/`) and the readers' outputs (`extract/`, `readers.txt`);
  - the orchestrator's stack sample of the GCS server, `gcs_server_99918_sample.txt` (§2.3).

**Nothing else under `var/` moved**
([`var_datastores_03.txt`](rrk-v020-adoption/rehearsal/var_datastores_03.txt)).
- Before the first launch `var/datastores/` held 21 files, all tombstone sidecars, with digest
  `7ad566f4342d678ab441dd540fd97fc3b8b0f094077f92ffe96b2469ef899eaa`.
- After the last step it held 39. The 21 have the same SHA-256, size and `mtime_ns` as before. The
  18 new files are R3's three stores and their sidecars. There is no `-journal`, `-wal` or `-shm`
  file.
- Nothing under `var/` is newer than the first capture but R3's runs, stores and working directory,
  and the two directories holding them.

**The registry after R3** ([`run_records_03.txt`](rrk-v020-adoption/rehearsal/run_records_03.txt)).
`./venv/bin/python -m runregistrykit list` shows **51 runs**: "49 finished, 2 unknown; 45 with no
host recorded".
- Nothing is `running` or stale, and no row is marked `!!`.
- By state: 27 `done`, 18 `killed`, 3 `failed`, 1 `lost` and 2 `unknown`.
- The runs are 01's 45, every id of `measure/out/01-list.txt` present, and R3's six: the four
  launches, `hardkill-a` (`lost`, counted among the finished) and `hardkill-b`.

## 2. The hard-kill route on the A3 v2 launcher

The pair ran after `resumed-c` had ended, its last Ray process was gone, and the readers and the
guard had run, under the same lock. Every command and its output, in order, is in
[`hardkill_03.txt`](rrk-v020-adoption/rehearsal/hardkill_03.txt).

**Headline.**
1. **The route holds on the real launcher.**
   - `hardkill-a` died of signal 9 in the QSI stage, with its status left `running`.
   - `list` marked it stale, `!!`, and named `end`.
   - The resume was refused at `begin`, naming the dead run, `stale` and `end`. No run directory
     was created, and no store file changed.
   - `end` recorded the run `lost`, verified, and a second `end` was refused.
   - `hardkill-b` was then accepted and ended `done`, with its fingerprint recorded and the sidecar
     `written`.
   - The `hardkill` store is `39809dca…`, equal to `reference`, to its sidecar and to R2's recorded
     fingerprints, and it passes every read-only check.
   - `v0.1.0` refused that last fingerprint (`[a3-v2-a-hard-killed-run-blocks-its-store]`, step 4).
     The issue closes.
2. **A store left with no handler having run resumes to the reference.** The kill left no journal
   beside any file. It also left no unvalidated row: the `GkSource` stage had completed and been
   validated, and `QuadSourceIntegral` rows carry no `validated` column. So `hardkill-b` pruned
   nothing and rolled back nothing. It computed the 626 QSI rows that were missing, and nothing
   else.
3. **One Ray process outlived the killed driver by more than ten minutes.** H2 does not hold as
   the prompt states it.
   - The raylet, the workers and the actors were gone within 40 s.
   - The **GCS server**, reparented to pid 1, entered its own shutdown at the kill and deadlocked
     there.
   - The prompt's stop fired at 23:59:15. The user ruled twice; a `SIGTERM` had no effect, and a
     `SIGKILL` ended it at 00:10:05 (§2.3).
   - Opened as `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` (§2.10).

### 2.1 `hardkill-a`

**The launch** was `reference`'s command on the `hardkill` store, with no `--resume`, `--register
hardkill-a`, and the same detached `nohup sh -c '…; echo "** driver exit status $?"'` wrapper. It
was launched at 23:46:18. Its pre-launch check found:
- `ray-processes` empty;
- the store name `absent` to `store show`, with no runs;
- no `rrk-v020-adoption-03-hardkill*` file.

**The interrupter** was launched at 23:46:19 with `resumed-b`'s pattern and hold exactly, and
`--signal KILL` added:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/rrk-v020-adoption-03-hardkill-a-20261018T234619 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 30 --signal KILL > /dev/null 2>&1 &
disown
```

**The one check** (23:46:26), then `set-pid 99903`:
- the run is `rrk-v020-adoption-03-hardkill-a-20261018T234619`, with driver pid 99903, `sh` 99897
  and interrupter 99907;
- the manifest has twenty keys, the record's `packages`, `git_dirty` false and this machine's
  `host`;
- the status is `running`, pid answering, with this host's `id` and `boot`;
- `list` showed it `running`.

### 2.2 After the kill

- **The interrupter's log** ([`interrupter_firings_03.txt`](rrk-v020-adoption/rehearsal/interrupter_firings_03.txt);
  also saved under `var/` before `end`):

  ```
  2026-10-18T23:48:45+01:00 liveness alive; pid 99903; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
  2026-10-18T23:48:45+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 30 s, then reading again
  2026-10-18T23:49:15+01:00 sent SIGKILL to pid 99903 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command "…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/rrk-v020-adoption-03-hardkill.sqlite … --register hardkill-a …"; exiting
  ```

  The stage matches the regex, and the reading (0 %) is at the point (0 %). The QSI stage began
  between 23:48:40 and 23:48:45, so the kill came 30–35 s into it.
- **The launch log** ends with three lines. The middle one is the shell's own report; it names its
  `$0`, which is the purpose:

  ```
  ** CALCULATE QUADRATIC SOURCE INTEGRALS
  <purpose>: line 5: 99903 Killed: 9               PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py … --register hardkill-a …
  ** driver exit status 137
  ```
- **`status.json`** says `state` `running`, `pid` 99903 and `exit_code` `null`. The heartbeat is
  `2026-10-18T23:48:44+01:00`, 31 s before the kill. The stage is `CALCULATE QUADRATIC SOURCE
  INTEGRALS`, and the `host` is `id` `8d2b247824882486` with `boot` `376461AF-…`, this machine's.
  There is no `fingerprint` key: no `finish` ran. At `end`, 00:11:12, the heartbeat's age was
  1348.9 s.
- **The run directory** holds `interrupter.log`, `manifest.json`, `status.json`, `stderr.log` and
  `stdout.log`. **No `status.json.*.tmp`** was left: the kill did not fall inside a status write.
  `stderr.log` is empty.
- **`kill -0 99903`** gave "no such process" (23:49:55, and again at 00:11:12).
- **The store's five files** have no `-journal`, `-wal` or `-shm` beside any of them. Whatever the
  `Datastore` actors were doing at the kill, none left a hot journal. Which of them was mid-write,
  if any, is not recorded.
- **What the kill cut short.** `hardkill-a`'s `stdout.log` lacks the lines Ray prints when its
  log-deduplication buffer flushes at shutdown: the `(Datastore pid=…) … [repeated Nx across
  cluster]` block that ends R2's `resumed-a`. No shutdown ran.

### 2.3 Ray after the `SIGKILL` (U3)

**The reading.** At 23:49:55, 40 s after the kill, `tools.rehearsal_lock ray-processes` exited 1
and listed one process:

```
99918 1 /Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/ray/core/src/ray/gcs/gcs_server --log_dir=/tmp/ray/session_2026-10-18_23-46-20_276571_99903/logs …
```

- This is the GCS server of the dead driver's session, started at 23:46:20 and reparented to
  `launchd` (pid 1).
- Every other Ray process of the run had gone by then: the raylet, the workers, the `Datastore`
  actors and the broker. 18 were listed at 23:46:27.
- The orchestrator's own reading at 23:49:35, 20 s after the kill, listed the same one process.

**The wait** (correction 10). One foreground loop tested `kill -0 99918` every second from
23:50:03 until the deadline, the interrupter's `sent SIGKILL` stamp plus 600 s, 23:59:15. It ended
`!! STILL ALIVE at the deadline 2026-10-18T23:59:15+01:00: [99918]`.

**The stop.** At 23:59:28:
- 99918 was alive in state `S`, with parent 1, no children and no listening TCP socket;
- `ray-processes` exited 1 and listed it alone;
- the lock was held, with its owner pid 99903 dead.

Nothing was signalled, `ray stop` was not run, and the lock was not released. The agent reported to
the orchestrator and ended its turn.

**What the GCS server was doing.** Its `gcs_server.out` ends with these lines:

```
[2026-10-18 23:49:15,151 W 99918 59846240] (gcs_server) server_call.h:331: [1] Not sending reply because executor stopped.
[2026-10-18 23:49:15,152 I 99918 59846167] (gcs_server) gcs_server.cc:267: Stopping GCS server.
[2026-10-18 23:49:21,390 W 99918 59846210] (gcs_server) metric_exporter.cc:105: [1] Export metrics to agent failed: RpcError: …
```

Nothing follows them, and `gcs_server.err` is empty. It began to stop in the second of the kill, and
did not finish.

**The user's rulings**, carried out by the orchestrator (relayed to the agent; the agent signalled
nothing):
1. **00:00:03.** The orchestrator confirmed the stop: 99918 alive, ppid 1, elapsed 13:43, no
   listening socket. The user chose "Claude sends SIGTERM".
2. **00:06:13.** `kill -TERM 99918` was sent once, rc 0. The process was still alive at 00:07:15,
   with nothing new in `gcs_server.out`.
3. **00:07–00:09.** A read-only stack sample, `sample 99918 2`, was saved as
   `var/rrk-v020-adoption-03/gcs_server_99918_sample.txt` (not committed; 30 threads). In all 1386
   samples the main thread was in:

   ```
   main → boost::asio::io_context::run() → … signal_handler<main::$_0, …>::do_complete
     → ray::gcs::GcsServer::Stop() → ray::rpc::GrpcServer::Shutdown()
     → grpc::Server::ShutdownInternal(gpr_timespec) → gpr_cv_wait → _pthread_cond_wait → __psynch_cvwait
   ```

   So the GCS server entered `Stop()` from its own signal handler and is waiting in gRPC's shutdown
   for in-flight calls that will not complete: the log says "Not sending reply because executor
   stopped". A `SIGTERM` goes through the same handler, so it cannot end the process. The user then
   chose "Claude sends SIGKILL".
4. **00:10:05.** `kill -KILL 99918` was sent once, rc 0, and the process was gone by `kill -0` the
   same second. `ray-processes` then exited 0, listing none, and the machine-wide `pgrep` found
   nothing.
5. `ray stop` was never run, and no other process was signalled.

The agent's own reading at 00:10:29 agrees: `kill -0 99918` gives "no such process",
`ray-processes` exits 0 and `pgrep` is empty.

| Process | Gone by | After the kill |
|---|---|---|
| the driver, 99903 | 23:49:15 (the `SIGKILL`) | 0 s |
| the raylet, workers, actors and broker (17 of the 18 listed at 23:46:27) | 23:49:35, the orchestrator's reading | ≤ 20 s |
| the GCS server, 99918 | 00:10:05, by `SIGKILL` on the user's ruling | 20 m 50 s; a `SIGTERM` at 00:06:13 had no effect |

**H2 does not hold as stated.** A Ray process of this repository outlived `hardkill-a` past ten
minutes, and would not have gone by itself. For A3 v2 this is a step in the route after a hard kill,
alongside `end`: check `ray-processes`; a GCS server still listed may need a `SIGKILL` by its pid,
and never `ray stop`. 04 states what A3 v2 inherits. The fix, if there is one, is Ray's or the
launcher's, and is not taken here (§2.10).

### 2.4 The evidence before `end`

At 00:10:41, after the Ray wait:
- **`closed_store_state.py`** read the `hardkill` store `mode=ro`.
  - No journal, WAL or shared-memory file was beside any file, and `replication_in_flight` had 0
    rows.
  - The version row is on every shard.
  - `GkSource` was 97 + 97 + 0 + 97 = 291 rows, **0 unvalidated**, and every other class was all
    validated.
  - The files were identical across the read (correction 8).
- **`./venv/bin/python -m runregistrykit list`** exited 0:

  ```
  !! rrk-v020-adoption-03-hardkill-a-20261018T234619                    running           -      24m  rrk-v020-adoption pr…
  50 run(s) under /Users/ds283/Documents/Code/SecondaryGWKit/var/runs: 47 finished, 1 stale, 2 unknown; 45 with no host recorded.
  ```

  Its stderr read: `!! 1 run(s) say they are running and are not: the pid does not answer kill -0,
  the heartbeat is older than 900 s, or this machine has rebooted since. Nothing here has killed or
  cleaned anything — go and look. Once you know one is over, end it with `python -m runregistrykit
  end RUN_ID --reason TEXT`.`
- **`store show`** on the `hardkill` primary gave kind `absent`, since there is no sidecar yet
  (README §6.2 C8), and problems none. It listed the run that names the store:
  `rrk-v020-adoption-03-hardkill-a-20261018T234619  running   stale     by results`. Exit 0.
- **The store's files** (SHA-256, size, `mtime_ns`):

  | File | SHA-256 | Size | `mtime_ns` |
  |---|---|---|---|
  | `rrk-v020-adoption-03-hardkill.sqlite` | `0eea775f…` | 36864 | 1792363586435920886 |
  | `…-shard0000.sqlite` | `97745599…` | 22474752 | 1792363749770262539 |
  | `…-shard0001.sqlite` | `f8498b68…` | 22315008 | 1792363749746549042 |
  | `…-shard0002.sqlite` | `47c03dd4…` | 1077248 | 1792363586421072878 |
  | `…-shard0003.sqlite` | `e2c5c08a…` | 22384640 | 1792363749732498328 |

  There is no `-journal`, `-wal` or `-shm`.
- **`var/runs/`** held 50 directories (93 entries with the launch logs).

### 2.5 The refused resume

The pre-check found no `-journal` beside the `hardkill` primary (correction 9) and `ray-processes`
empty. The launch at 00:10:54 was `--resume --register hardkill-b`, detached as every launch, into
`var/runs/rrk-v020-adoption-03-hardkill-b-refused.launch.log`, with `sh` pid 10117. One check
followed (correction 11): a wait on the `sh` by `kill -0`, bounded at 120 s. The `sh` was gone in
1.0 s. No `set-pid` was run.

**The launch log**, whole:

```
** quadsource_atol_sweep --build: resuming into rrk-v020-adoption-03-hardkill.sqlite
Traceback (most recent call last):
  File "…/docs/handover/quadsource_atol_sweep.py", line 1055, in <module>
    main()
  File "…/docs/handover/quadsource_atol_sweep.py", line 1039, in main
    run_build(args)
  File "…/docs/handover/quadsource_atol_sweep.py", line 556, in run_build
    run = runregistrykit.begin(
          ^^^^^^^^^^^^^^^^^^^^^
  File "…/venv/lib/python3.12/site-packages/runregistrykit/__init__.py", line 1119, in begin
    raise RuntimeError(
RuntimeError: Cannot begin run rrk-v020-adoption-03-hardkill-b-20261019T001055 with results "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/rrk-v020-adoption-03-hardkill.sqlite": run rrk-v020-adoption-03-hardkill-a-20261018T234619 is running (stale, pid 99903) and names this store by its results. A stale run is ended by a person, with `python -m runregistrykit end RUN_ID --reason TEXT`, before another run is begun on its store; the registry does not decide that a run is dead. No run directory was created
** driver exit status 1
```

- The `--resume` pre-check (`:544`) passed: it read the primary `mode=ro`.
- `begin` (`:556`) raised the words 02's probe and test pin, with no trailing full stop.
- **Ray was imported but not started** (`:531` imports it before the pre-check; the orchestrator's
  correction 6). The log holds no `bootstrapping a local Ray instance` line, and `ray-processes`
  exited 0 afterwards.
- **No run directory was created.** `var/runs/` held 50 directories, as at §2.4; the 94th entry is
  this launch log. No directory named `…-hardkill-b-20261019T001055` exists.
- **Every store file** has §2.4's SHA-256, size and `mtime_ns`, byte for byte.

### 2.6 `end`

`status.json` and `interrupter.log` were saved under `var/rrk-v020-adoption-03/state/` first. The
manifest's SHA-256 was `e4442b60…`. `kill -0 99903` gave "no such process" at 00:11:12. Then:

```bash
./venv/bin/python -m runregistrykit end rrk-v020-adoption-03-hardkill-a-20261018T234619 \
    --reason "rrk-v020-adoption 03: SIGKILLed by interrupt_when.py --signal KILL at CALCULATE QUADRATIC SOURCE INTEGRALS; pid 99903 checked gone with kill -0 at 2026-10-19 00:11:12 +0100"
```

It exited 0 and printed `>> end: rrk-v020-adoption-03-hardkill-a-20261018T234619 is now lost,
verified: its pid 99903 does not answer kill -0 on this host`.

**The record** (`status.json` after; the checks are the orchestrator's correction 13):
- `state` is `lost` and `exit_code` is `null`;
- `ended.verified` is `true` and `ended.not_verified_because` is `null`;
- `ended.evidence` holds `host` `this`, `boot` `this`, `pid_answers` `false`, `state` `running`,
  `pid` 99903, `status_readable` `true`, `heartbeat` `2026-10-18T23:48:44+01:00` and
  `heartbeat_age` 1348.9;
- `ended.at` is `2026-10-19T00:11:12+01:00` and `ended.by` is `ds283`. `ended.host` names this
  machine's `id`, `name` and `boot`. `ended.reason` is the text above;
- **`ended.git_head`** is `aab407529576307c9d3c1e64ccc8b2e980ed2d7d`, the dispatch `HEAD`;
- **every other key is kept with its value.** Setting aside the four `end` writes (`state`,
  `exit_code`, `ended`, `heartbeat`), `host`, `pid`, `stage`, `units_done` and `units_total` are
  unchanged. `heartbeat` moved to `00:11:12`, the write's own;
- the manifest's SHA-256 is unchanged, `e4442b60…`.

**The second `end`** exited 1, with `!! Cannot end rrk-v020-adoption-03-hardkill-a-20261018T234619:
its status says it has already ended, as 'lost'; end records only a run whose end the registry
never saw. Nothing was written`. `status.json` is byte-identical before and after it (`ec5d8a64…`).

**`list` again:** `rrk-v020-adoption-03-hardkill-a-20261018T234619  lost`, with no `!!`; "50
run(s) …: 48 finished, 2 unknown". The run is counted among the finished.

### 2.7 The sidecar

At 00:11:36, after `end`, `./venv/bin/python -m runregistrykit store create --purpose "<one line>"
var/datastores/rrk-v020-adoption-03-hardkill.sqlite` exited 0, with `store_id`
`63ca0315b9404b46a80bc68c6a979db3`. `store show` then gave the sidecar and listed
`rrk-v020-adoption-03-hardkill-a-20261018T234619  lost  finished  by results`. The store's five files
were identical across it.

### 2.8 `hardkill-b`

**The launch** came at 00:11:44, after `ray-processes` had again listed none. It was
`--resume --register hardkill-b`, into `var/runs/rrk-v020-adoption-03-hardkill-b.launch.log`. The run
is `rrk-v020-adoption-03-hardkill-b-20261019T001145`, with driver pid 10646 and `sh` 10642.

**The one check** (00:11:51) found:
- `begin` accepted the run;
- the manifest as every other, with `results_store_id` `63ca0315…`;
- the status `running`, with this host's `id` and `boot`.

`set-pid 10646` followed. The orchestrator's wait ended `ENDED done`, the pid gone by 00:13:47.

**At open.** `stdout.log` holds no `INTEGRITY WARNING`, no prune line and no `Rolled back the
journal` line.
- No journal lay beside any file (§2.4), so DatastoreKit's check at open had nothing to roll back.
- No row was unvalidated, so the prune had nothing to remove. `hardkill-a` was killed after the
  `GkSource` stage had completed and validated its 291 rows, in the QSI stage, whose rows carry no
  `validated` column.

The prune at open is therefore not exercised by this pair. R3's `resumed-b` exercised it (§1.3.2).

**What it computed** (`stage_counts_03.txt`): every stage 0/0/0 up to `APPLY GKSOURCE POLICIES`,
then **`QuadSourceIntegral` 626/626/626**.
- So `hardkill-a` had stored 864 − 626 = **238** QSI rows when it was killed. That figure is
  inferred: no fingerprint was taken of the store between the two runs.
- The store now holds 864 rows, and `read_inventory` reports no duplicate (§2.9). So no row stored
  before the kill was computed again.

**At finish** (`status.json`):
- `state` is `done` and `exit_code` is `0`;
- the `fingerprint` has `fingerprint_format` 6, **`encoding_format` 1**, digest `39809dca…`, 21
  classes and problems `{}`, taken at 00:13:29 at `aab40752…` with `git_dirty` false;
- **`fingerprint_sidecar` is `written`**, and there is no `fingerprint_error`.

This is the fingerprint `v0.1.0` refused, at step 4 of `[a3-v2-a-hard-killed-run-blocks-its-store]`.
- The launch log ends `** driver exit status 0`.
- `stderr.log` holds the SciPy `IntegrationWarning`, as `resumed-c`'s does.
- Its `stdout.log` matches R2's `resumed-c` in line kinds and in the `** ` sequence, but for the
  store and run names.
- Its statistics block has R2's counts, `minimal` first under `mixed`, like R2's and R3's
  `resumed-c` (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
- `hardkill-a`'s block is byte-identical to R3's `reference`.

**After it** (00:14:19), no Ray process was left: `ray-processes` exited 0, `pgrep` was empty, and
10642, 10646 and the Ray pids its logs name (10686, 10690, 10902) were gone. `hardkill-b`'s Ray
shut down through the driver, as every `done` run's does.

### 2.9 The checks, read-only, on the `hardkill` store

At 00:14:38, after `hardkill-b` and its Ray processes had gone (`hardkill_03.txt`, §2.9):
- **`compare_with_recorded.py`**, against R3's `reference` and `resumed-c` and R2's `reference` and
  `resumed-c`, exited 0 with verdict `equal`. It found 0 differences against the sidecar and
  against each of the four recorded fingerprints: format 6, `39809dca…`, 21 classes, problems none.
- **`compare_stores.py`** of `reference` against `hardkill` exited 0: each matches its sidecar,
  `compare_fingerprints` has 0 differences, all 21 classes are equal, and the verdict is `equal`.
- **`store_soundness.py`** exited 0, `=> clean`.
  - On the primary and every shard, opened `mode=ro`, `PRAGMA integrity_check` is `['ok']` and
    `foreign_key_check` returns 0 rows.
  - The version row is on each shard.
  - `read_inventory` reads shards `(0, 1, 2, 3)` with no problem and 0 `duplicate`.
- **The census:** every table `OK`, `no divergence`, exit 0.
- **`closed_store_state.py`** after `hardkill-b` found no journal, 0 in-flight rows and no
  unvalidated row.
- **`prune_check.py`** against the states after `hardkill-a` and after `hardkill-b` each gave
  `none: there is nothing to check`, exit 0. That is so because `hardkill-a` left no unvalidated
  serial, as §2.8 says; it is not a prune that was exercised.
- **Every file of the store**, the sidecar included, had the same SHA-256, size and `mtime_ns`
  before and after these checks.

Then the lock was released (§1.1.3).

### 2.10 The issue opened

**`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`** is on this campaign's board, §3,
and indexed in `docs/OPEN_ISSUES.md` §1.22.
- After a `SIGKILL` of the driver, Ray's GCS server is left reparented to pid 1, inside
  `GcsServer::Stop()`, waiting in gRPC's shutdown, and does not exit.
- A `SIGTERM` cannot end it; a `SIGKILL` does.
- It bears on A3 v2's route after a hard kill: after `end`, a person also checks `ray-processes`,
  and may need to `SIGKILL` a GCS server by its pid.
- The next step is 04's, which states what A3 v2 inherits. Any fix is Ray's or the launcher's.

**`[a3-v2-a-hard-killed-run-blocks-its-store]`** closes: acceptance 3 holds (§2.1–§2.9). Its entry
moves to the `handover` board's §4, and its row leaves `docs/OPEN_ISSUES.md` §1.22.

## 3. The close

**§3's tree.** Prompt 04 ran on branch `handover-remedial` at **`aa0b57c`** ("Write
rrk-v020-adoption prompt 04 and its orchestrator note"), clean. Its code is `2818780`'s: `aa0b57c`
differs from `ebb70a3` in `prompts/` alone, and `ebb70a3`, `17916cb` and `aab4075` touch no code, test or
script after `2818780` (`git diff --stat 2818780 HEAD -- . ':!prompts' ':!docs'` is empty). 04 changes
records, `CLAUDE.md` rule 5 and nothing else: no code, no test, no script, nothing under `var/`.
**Every measurement of §3.2 was taken at `aa0b57c` before any edit**, and 04's own commit changes no
file any of them reads. Each is saved whole, with its command and the `HEAD`, under
`prompts/rrk-v020-adoption/measure/out/04-*`.

### 3.1 R3's six runs, as one table

Read from each run directory's `manifest.json` and `status.json` under `var/runs/`, at `aa0b57c`.
Every run has `git_head` `aab407529576307c9d3c1e64ccc8b2e980ed2d7d`, `git_dirty` `false`,
`script_sha256` `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`, twenty manifest
keys with `host`, and `packages` equal as JSON to `package_provenance` (§3.2): `runregistrykit` 0.2.0,
`datastorekit` 0.4.0, `raypoolkit` 0.2.0, none editable. Every fingerprint is `fingerprint_format` 6,
`encoding_format` 1.

| Slug | Run id (`rrk-v020-adoption-03-…`) | `--resume` | End state, exit code | Digest | `fingerprint_sidecar` |
|---|---|---|---|---|---|
| `reference` | `reference-20261018T232838` | no | `done`, 0 | `39809dca…` | not written: no sidecar existed yet |
| `resumed-a` | `resumed-a-20261018T233357` | no | `killed`, 143 | `c60e0d19…` | not written: no sidecar existed yet |
| `resumed-b` | `resumed-b-20261018T233734` | yes | `killed`, 143 | `7e41cbe2…` | `written` |
| `resumed-c` | `resumed-c-20261018T233941` | yes | `done`, 0 | `39809dca…` | `written` |
| `hardkill-a` | `hardkill-a-20261018T234619` | no | `lost`, `null` (by `end`, verified) | none: `finish` never ran | none |
| `hardkill-b` | `hardkill-b-20261019T001145` | yes | `done`, 0 | `39809dca…` | `written` |

- The full `done` digest is `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, as in
  §1.2.1. No run has a `fingerprint_error`.
- Three digests are `39809dca…`, in three different stores' runs (`reference`, `resumed-c`,
  `hardkill-b`), and the three stores' sidecars carry the same digest (§3.2).
- The two `not written` rows are what §1.2.1 records: `store create` came after those two runs.

### 3.2 Z1 — the final measurements

Taken at `aa0b57c` before any edit, in the order of the prompt. Each is in a file under
`prompts/rrk-v020-adoption/measure/out/` with its command and the `HEAD`.

**The seven roots** ([`04-roots.txt`](../prompts/rrk-v020-adoption/measure/out/04-roots.txt)), each
once, in the foreground, serially, against 02's counts, at one-minute load 11–26 (a Time Machine
backup was running):

| Root | Ran | Result | Wall | 02's count |
|---|---|---|---|---|
| `AdaptiveLevin/tests` | 32 | OK | 0.2 s | 32 |
| `CosmologyModels/tests` | 39 | OK | 2.2 s | 39 |
| `ObjectFactories/tests` | 279 | OK | 25.7 s | 279 |
| `tools/tests` | 91 | OK | 17.6 s | 91 |
| `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 40.8 s | 148 (1 skipped) |
| `RunDrivers/tests` | 103 | OK | 7.1 s | 103 |
| `ComputeTargets/tests` | 700 | FAILED (failures=1) | 303.0 s | 700 |

- **Every count is 02's.** The one failure is `test_tk_wkb_phase.TestCost.test_wall_time_per_object`,
  the first of README §8's four known wall-clock tests. In the full run it failed on `0.1397 not less
  than or equal to 0.06` (`COST_COLD_WALL_TIME_LIMIT`). Re-run alone it failed again, `0.0759 not less
  than or equal to 0.05`, at one-minute load 21.8, with `WindowServer` (72 %), `backupd` (37 %) and
  `mds_stores` (35 %) the heaviest processes (`ps -Ao pcpu,comm -r`). The orchestrator found the same
  test failing the same way at load 40 under a backup at `ebb70a3`, and passing at load 4–6 at 03's
  review. No code has changed since, and 04 changes no code a test reads, so the failure cannot be
  04's (the orchestrator's ruling 9). No other test failed.

**The packages** ([`04-pip.txt`](../prompts/rrk-v020-adoption/measure/out/04-pip.txt)).
- `pip freeze` has 101 lines and **equals 01's after-install freeze exactly** (the block headed
  `pip freeze … after the install` in `01-pip.txt`, blank lines dropped; `diff` exit 0).
- `pip check`: `No broken requirements found.`
- `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))`, as JSON
  with sorted keys, **equals the `packages` of each of R3's six manifests** (six of six): `datastorekit`
  0.4.0 at `c8570e84…` (`v0.4.0`), `raypoolkit` 0.2.0 at `eaef0ce6…` (`v0.2.0`), `runregistrykit` 0.2.0
  at `0bae3a3c…` (`v0.2.0`), none editable.

**The registry** ([`04-list.txt`](../prompts/rrk-v020-adoption/measure/out/04-list.txt)).
`./venv/bin/python -m runregistrykit list` exits 0 with **stderr empty**. Its last line reads
`51 run(s) under …/var/runs: 49 finished, 2 unknown; 45 with no host recorded.` By state the 51 are 27
`done`, 18 `killed`, 3 `failed`, 1 `lost` and 2 `unknown` (the pre-registry `a3-pilot` pair). None is
`running`, stale or `foreign`, and no row carries `!!` or `??`.

**The stores** ([`04-store-show.txt`](../prompts/rrk-v020-adoption/measure/out/04-store-show.txt)).
`find var/datastores -name '*.manifest.json'` gives **24** sidecars (a glob of the top level finds 23,
one being in `backup-pre-resume-20260921T091011/`). `runregistrykit store show` on each one's store
exits 0, kind `registry`, **`problems: none`**: 24 of 24.
- **The 21 tombstones** read `state: retired`, and each reads **line for line as `01-store-show.txt`'s
  after-block** for the same store (21 of 21 equal, the `exit=` lines aside).
- **R3's three** (`rrk-v020-adoption-03-{reference,resumed,hardkill}`) read kind `registry`, problems
  none, live, and each sidecar's fingerprint digest is `39809dca…`. Each lists its runs: `reference`
  its `reference` run, `resumed` the three, `hardkill` the two.

**The machine** ([`04-disk.txt`](../prompts/rrk-v020-adoption/measure/out/04-disk.txt)). `df -h /`:
927G, **863G used, 64G free, 94 %**. The orchestrator read 859G used, 69G free and 93 % at `ebb70a3`;
the 4 GB between the two readings is not 04's, which wrote nothing under `var/`.

### 3.3 Z2 — RunRegistryKit's G1, drafted

Not committed in RunRegistryKit. The user makes that commit, or has a session there make it
(README §6.3).

RunRegistryKit's `prompts/interrupted-runs/IMPLEMENTATION_STATE.md` §2 (`:1176-1188` at `f6d4b80`) has
G1's status cell as `not yet (SGK adopts `v0.2.0`)`, and says "After the close, a gate is recorded on
this board when it holds, in a records commit of its own". The gate's own words are in that
campaign's `README.md` §7 (`:388-402`): SGK "has adopted `v0.2.0` in a campaign of its own, its pin
moved in a commit of its own". So the draft is the **row of G1**, in the form of the `extraction`
board's G2 cell (`prompts/extraction/IMPLEMENTATION_STATE.md:1128`): `**holds** <date>: …`. It
replaces `| **G1** | not yet (SGK adopts `v0.2.0`) |`. The date is 2026-10-18, the day `0a27b0c`
moved the pin in a commit of its own, which is when the gate's condition held (the orchestrator's
ruling 1); the cell names 04's commit, `34bf4f7`, which 04 wrote as "this commit" and the
orchestrator substituted in its follow-up.

```markdown
| **G1** | **holds** 2026-10-18: SGK adopted `v0.2.0` in a campaign of its own, `rrk-v020-adoption`, its pin moved in a commit of its own. 01 (`0a27b0c`, 2026-10-18) moved the one line `requirements.txt:81` from `@v0.1.0` to `@v0.2.0`; `pip install --dry-run` said `Would install runregistrykit-0.2.0` and nothing else, and the venv then held `runregistrykit` 0.2.0 at `0bae3a3c…` (`requested_revision` `v0.2.0`, not editable), `datastorekit` 0.4.0 and `raypoolkit` 0.2.0 unmoved, `pip check` clean. 02 (`2818780`, 2026-10-18) adopted the route after a hard kill in SGK's `CLAUDE.md` and tested it: the interrupter gained `--signal {TERM,KILL}`, and a new driver test, `RunDrivers/tests/test_hard_killed_predecessor.py` (4 tests), `SIGKILL`s a real child and shows `scoped_pipeline_run.register` refused, `end` exit 0, and a second registration accepted. 03 (`17916cb`, 2026-10-19) rehearsed it on A3 v2's own launcher, `docs/handover/quadsource_atol_sweep.py` (R3, six runs): `rrk-v020-adoption-03-reference-20261018T232838` and `-resumed-c-20261018T233941` (both `done`, both `39809dca…`, equal to SGK's earlier R2, `runregistry-extraction` 08) through `v0.2.0`, and the hard-kill pair `-hardkill-a-20261018T234619` (`SIGKILL`ed, `lost` by `end`, verified) and `-hardkill-b-20261019T001145` (`done`, `fingerprint_sidecar` `written`, `39809dca…`). 04 (`34bf4f7`, 2026-10-19) took the final measurements and closed the campaign. Each of the adoption note's sections, [`docs/adoption/v0.2.0/secondarygwkit.md`](../../docs/adoption/v0.2.0/secondarygwkit.md), is answered by SGK's `docs/rrk-v020-adoption-verification.md` (§1–§3) and `prompts/rrk-v020-adoption/measure/out/`: (§1, the pin) as above, `requirements.txt:17`, `:75`, `:76` and `:88` unchanged, `pip freeze` differing from before by the one line (`01-pip.txt`) and equal to it at 04 (`04-pip.txt`), and `package_provenance` equal to the `packages` of all six R3 manifests; (§2, its runs) `list` read SGK's 45 run directories written under `v0.1.0` as before, same rows and states, gaining "45 with no host recorded" (`01-list.txt`), and at 04 gives 51 runs, 49 finished (27 `done`, 18 `killed`, 3 `failed`, 1 `lost`) and 2 `unknown`, nothing `running` or stale, exit 0, stderr empty (`04-list.txt`), with one checkout and no worktree sharing the runs root; (§3, `liveness`'s values) `interrupt_when.py` still waits on `alive` and now takes `--signal`, its dummy tests unchanged for `TERM` and a new `KILL` case exiting 137 (02), and R3 used it three times, `SIGTERM` for `resumed-a` and `resumed-b` and `SIGKILL` for `hardkill-a` (verification §1.1.2, §2.2), `tools/rehearsal_lock.py`'s own `pid_alive` being untouched; (§4, `begin` with `results`) the launcher's `run_build` (`quadsource_atol_sweep.py:556`) refused the resume of `hardkill-a`'s store at `begin` with `RuntimeError`, naming the dead run, `stale` and `python -m runregistrykit end`, ending `No run directory was created`, no run directory added and the store's five files byte-identical (§2.5), then `end` and the resume accepted (§2.6, §2.8); (§5, the handlers' `finish("killed", …)`) `resumed-a` and `resumed-b` each ended `killed`, exit 143, through the new status writer, with a fingerprint recorded and no `status.json.*.tmp` left (§1.2.2); (§6, `TERMINAL_STATES`) `RunDrivers/tests/test_pipeline_adoption.py:249-253` passes with `lost` added, and `lost` is `hardkill-a`'s recorded state; (§7, refusal fragments and two runs on one store) `test_quadsource_atol_sweep_prepare` passes unchanged, and R3's refused resume is the one real second `begin` on a store an unfinished run names; (§8, `pid_alive` and key sets) all seven SGK roots passed at 01, 02 and 03, and at 04 six pass and `ComputeTargets` has one failure, a wall-clock assertion under machine load that does not touch the registry (`test_tk_wkb_phase.TestCost.test_wall_time_per_object`), at the same counts throughout (AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279, tools 91, LiouvilleGreen 148 with 1 skipped, RunDrivers 99 and then 103 with 02's four tests, ComputeTargets 700), and every R3 manifest has twenty keys with `host` and every status a `host` with `id` and `boot` (§1.2.4). Of §9, the points that bear on the route were each shown on the real launcher in §2: the defect (`hardkill-a` left `running` with its pid dead; `list` marked it stale, `!!`, and named `end` on stderr, §2.4), `end` (exit 0, "now lost, verified", evidence `host` `this`, `boot` `this`, `pid_answers` `false`; a second `end` refused and the status byte-identical, §2.6), `begin`'s refusal (§2.5), and the resumed run's fingerprint (`hardkill-b` `done`, `fingerprint_sidecar` `written`, no `fingerprint_error`, `39809dca…` equal to `reference` and to R2's two recorded fingerprints, §2.8–§2.9), which `v0.1.0` recorded as `fingerprint_error`. **No defect of this package was met.** The one finding of R3, `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`, is Ray's process behaviour after a `SIGKILL` of a driver, not the package's, and is not a report to this repository; neither of this board's two open issues that bear on a long run was met (`[04-a-killed-process-not-yet-reaped-answers-kill-0]`: `hardkill-a`'s driver was reaped by its `sh` and `end` found `pid_answers` `false`; `[03-a-run-whose-manifest-cannot-be-read-names-no-store]`: every manifest was readable). **Read here** 2026-10-19 through `git` only, at RunRegistryKit `f6d4b80` and `v0.2.0`: the adoption note, the release notes and the `interrupted-runs` board's §2–§4, none changed. SGK's record is `docs/rrk-v020-adoption-verification.md` §1 (R3), §2 (the hard-kill route) and §3 (the close) at 04's commit |
```

Each clause of the draft is read from this repository:
- **The commits** exist and say what the draft claims (`git show --stat`): `0a27b0c` "Pin
  runregistrykit at v0.2.0" (`requirements.txt`, one line, with `CLAUDE.md`'s two pin mentions and
  `measure/out/01-*`); `2818780` "Adopt the hard-kill route in CLAUDE.md, the interrupter and a
  driver test" (`CLAUDE.md`, `interrupt_when.py`, the dummy test, `test_hard_killed_predecessor.py`);
  `17916cb` "Rehearse through runregistrykit v0.2.0 and the hard-kill route" (this document's §1–§2).
- **The adoption note's §1–§8** are each answered by a clause of the draft, as the table below
  maps; **§9's** four points that bear on the route are answered by §2.4–§2.8 of this document.

| Adoption note (`docs/adoption/v0.2.0/secondarygwkit.md`) | SGK's evidence, in the draft | Where |
|---|---|---|
| §1 the pin | `requirements.txt:81`, one line; `:17`, `:75`, `:76`, `:88` unchanged | 01, `01-pip.txt`; `04-pip.txt` |
| §2 its runs | 45 old runs read as before; 51 now; one checkout | `01-list.txt`; `04-list.txt` |
| §3 `liveness`'s values | `interrupt_when.py --signal`; three firings in R3 | 02; §1.1.2, §2.2 |
| §4 `begin` with `results`, after a hard kill | the driver test; the launcher's real refusal, `end`, the resume | 02; §2.5–§2.8 |
| §5 the handlers' `finish("killed", …)` | `resumed-a`, `resumed-b`; no temporary status file | §1.2.2 |
| §6 `TERMINAL_STATES` | `test_pipeline_adoption.py:249-253` passes; `lost` recorded | `RunDrivers` 103 at 02, 03, 04 |
| §7 refusal fragments, two runs on one store | `test_quadsource_atol_sweep_prepare` passes; one real second `begin` | `RunDrivers` 103; §2.5 |
| §8 `pid_alive` and key sets | all roots at 02's counts; twenty keys with `host` | §1.8, §3.2; §1.2.4 |
| §9, the defect, `end`, `begin`'s refusal, the resumed fingerprint | each shown on the real launcher | §2.4, §2.6, §2.5, §2.8 |

### 3.4 Z3 — the issues RunRegistryKit has since closed

Read with `git show f6d4b80:<path>` (RunRegistryKit's `HEAD`, clean) and `git show v0.2.0:<path>`.

| Issue (SGK's board entry) | RunRegistryKit's entry (`prompts/interrupted-runs/IMPLEMENTATION_STATE.md` §4 at `f6d4b80`) | `v0.2.0`'s release notes |
|---|---|---|
| `[04-runs-do-not-say-which-machine-produced-them]` (`run-registry` §4, `:377`) | `:1523`; **Resolved (2026-10-16, by 01, `387c58f`)** at `:1597` | §2.1, the `host` records (`:33`), and §2.4 (`:112`) |
| `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` (`a3-v2-readiness` §4, `:1685`) | `:1641`; **Resolved (2026-10-17, by 02, `2bf9c72`)** at `:1763` | §2.3, `status.json`'s temporary names (`:97`) |

- **Both are resolved on RunRegistryKit's board and released in `v0.2.0`**, which SGK has run since
  01 (`0a27b0c`). The release notes name each change, and the tag `v0.2.0` (tag object `895e3c70…`, on
  `0bae3a3c…`) contains both prompts' commits.
- **Where RunRegistryKit's change differs from the SGK entry's next step.** For the machine: SGK asked
  for a `machine` field from `platform.node()` and a lister that refuses to judge a foreign record;
  the package records a `host` (an `id` that is a digest of the OS's identity, a `boot`, and a name for
  reading) and judges a run on another host `foreign`, one from an earlier boot `stale`. For the
  heartbeat: the SGK entry's next step was a unique temporary name per write, which RunRegistryKit
  adopted for `status.json` alone, and it also discards a write that a later one has superseded; its
  board had already corrected the next step on hand-over. In each case RunRegistryKit's board is right.
- **The index** (`docs/OPEN_ISSUES.md`) said in §1.11, §1.17 and §1.21's lead-in that both "stay open"
  or are "still open" in RunRegistryKit. Each sentence now says where and when the issue was closed,
  that `v0.2.0` released it, and that SGK adopted it at `0a27b0c`; §1.21 says which of the four
  transferred issues are still open there (§1.14's two) and which closed (§1.11's, §1.17's). The
  rows were already deleted, and no row is added or deleted now: the index stays at 100.
- **The two boards** each gain one dated paragraph after their `Transferred (2026-10-16)` paragraph,
  and nothing else changes (`git diff -U0`: one hunk each).
- **The three left alone are still open in RunRegistryKit.** Its `docs/OPEN_ISSUES.md` §1.2 at
  `f6d4b80` (`:28-30`) lists `[00-a-copy-carries-its-sources-present-tense-fields]`,
  `[01-cross-filesystem-move-advice-says-delete-by-hand]` and
  `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]`, each with owner `none`. Their SGK
  records are not touched.

### 3.5 Z4 — what A3 v2 inherits

For `handover`, which owns the A3 v2 launch (`a3-v2-readiness` README §1). It starts from
[`runregistry-extraction-verification.md`](runregistry-extraction-verification.md) §4.5 and states
what is different, from the tree at `aa0b57c` and the records.

**The launcher, unchanged.** `docs/handover/quadsource_atol_sweep.py --build --register <slug>
--purpose "<one line>"`.
- Its SHA-256 at `aa0b57c` is `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`,
  **equal to all six of R3's manifests' `script_sha256`** (checked here against the six run directories)
  and to R2's. Its last commit is `7af17bd`. Its line numbers in §4.5 are still right.
- **The flags R3 ran it with:** `--build --rehearsal [--resume] --database
  var/datastores/<store>.sqlite --job-name rrk-v020-adoption-03 --atol 1e-32 --rtol 1e-7 --cpus 6
  --register <slug> --purpose "…" --campaign rrk-v020-adoption --prompt 03` (§1.1). **A production
  build drops `--rehearsal`**, which selects a three-wavenumber, `--zend 250` scope that "is not A3 v2"
  (`:989-993`); `--job-name` defaults to `handover-A3-baseline-v2` (`:1000`) and `--campaign` to
  `handover` (`:1014`).
- **It forces `address="local"`** at the `ray.init` it wraps (`:610`), and `run_build` registers the
  run with `runregistrykit.begin(…)` at `:556`, after the `--resume` pre-check and before Ray starts, so
  a refused `begin` ends the launch before any Ray process exists (§2.5).

**The pins,** by line of `requirements.txt`: `datastorekit @ …@v0.4.0` (`:17`), `raypoolkit @ …@v0.2.0`
(`:76`) and `runregistrykit[stores] @ …@v0.2.0` (`:81`), tags and never editable. `ray==2.43.0` (`:75`) and
`SQLAlchemy==2.0.39` (`:88`) are SGK's own pins. The only change since §4.5 is `:81`, from `v0.1.0`.

**The commands,** as R3 ran them, in their current spelling:
```bash
./venv/bin/python -m runregistrykit list                                    # discovery, before anything
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock acquire --purpose "<text>"
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes
./venv/bin/python -m runregistrykit store show var/datastores/<store>.sqlite   # absent before the build
# the launch above, then at once:
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock set-pid <driver pid>
# after a hard kill, once a person has checked that the run is over:
./venv/bin/python -m runregistrykit end <RUN_ID> --reason "<text>"             # on the host it ran on
# after the build ends, and its last Ray process is gone:
./venv/bin/python -m runregistrykit store create --purpose "<one line>" var/datastores/<store>.sqlite
./venv/bin/python -m runregistrykit store fingerprint --write var/datastores/<store>.sqlite
./venv/bin/python -m runregistrykit store show var/datastores/<store>.sqlite
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock release
./venv/bin/python -m runregistrykit store retire --dry-run --reason "<text>" var/datastores/<store>.sqlite
```
- **What `list` now prints:** `!!` before a run that says `running` and is not (the pid does not answer
  `kill -0`, the heartbeat is older than 900 s, or this machine has rebooted since), with the
  explanation on stderr; `??` before a `foreign` run, which is not tested here; and a summary line
  that ends "*N* with no host recorded", which counts the runs written under `v0.1.0`: 45, for ever.
- **The lock's five commands** and its rule are as §4.5 says: a held lock (exit 3) means stop and ask
  the user; never wait, retry or remove it; every driver names its address (`local`); `ray stop` is
  never run.
- **A `SIGTERM`'d build** needs none of this: its handler records `killed` through the new writer, with
  a fingerprint, and leaves no temporary status file (§1.2.2).

**The route after a hard kill, as R3 ran it** (§2, step by step):
1. `list` shows the run `stale`, with `!!`, and stderr says `!! 1 run(s) say they are running and are
   not … end it with python -m runregistrykit end RUN_ID --reason TEXT` (§2.4).
2. The resume's `begin` is **refused**, with `RuntimeError` naming the dead run, `stale` and `python -m
   runregistrykit end`, ending `No run directory was created`: no run directory is added, Ray is not
   started and the store's files are byte-identical (§2.5). The launch log is the record.
3. A person checks that the run is over (`kill -0` on its pid, on that host), and runs `end` **on the host
   it ran on**: exit 0, "now lost, verified", with the evidence recorded; a second `end` is refused
   (§2.6).
4. `tools.rehearsal_lock ray-processes` (below).
5. `store create`, if the store has no sidecar yet (R3's `hardkill` store had none, README §6.2 C8)
   (§2.7).
6. The resume is begun with `--resume --register <slug>`. It is accepted, runs to `done`, and its
   `finish` records the fingerprint, with `fingerprint_sidecar` `written` (§2.8). Under `v0.1.0` it
   recorded `fingerprint_error`.

**Ray after a `SIGKILL`,** as R3 found it (§2.3). The raylet, workers, actors and broker were gone
within 20 s. **The GCS server was not:** it was left reparented to pid 1, inside `GcsServer::Stop()`
waiting in gRPC's shutdown, which a `SIGTERM` cannot end (it goes through the same handler) and a
`SIGKILL` did, 20 m 50 s after the driver's death.
- Until it is gone `ray-processes` is not empty, so the lock's `release` **refuses** (it prints the
  processes and returns 1, `tools/rehearsal_lock.py:376-380`) and a next `acquire` refuses too (it
  removes the lock again and returns 4, `:293-299`).
- It is one observation of one kill. Whether every `SIGKILL` leaves the GCS server so, and what
  signalled it into `Stop()`, are not established. Hence `CLAUDE.md` rule 5 (below), and the issue
  `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`, which stays open with no owner.

**Ray after a `SIGTERM`:** `[07-a-killed-builds-ray-agents-outlive-the-driver]` (the `a3-v2-readiness`
board; index §1.17) is still open: after a killed build exits 143 its dashboard and runtime-env agents
may outlive it by 1–2 minutes, so "no Ray process left" is true only after waiting by pid. R3's §1.2.3
found none: nothing outlived either `SIGTERM`'d driver, at load 6–29.

**What the hard kill did not test** (03's log, observation 2). `hardkill-a` was killed in the QSI stage,
after `GkSource` had completed and been validated; `QuadSourceIntegral` rows carry no `validated`
column. So no row was left unvalidated and no journal was left: **neither the prune at open nor
DatastoreKit's journal rollback ran after a `SIGKILL`** (§2.4, §2.8). A kill inside the `GkSource`
stage, or inside a write, would test them. R3's `resumed-b` exercised the prune after a `SIGTERM`
(§1.3.2).

**RunRegistryKit's open issues that bear on a long run,** both still open at `f6d4b80` (its
`docs/OPEN_ISSUES.md` §1.3), both with owner none:
- **`[04-a-killed-process-not-yet-reaped-answers-kill-0]`.** A killed process its parent has not
  reaped still answers `kill -0`, so its run is `alive` in `list` and `end` refuses it. A detached
  launch through `nohup sh -c '…; echo "** driver exit status $?"'` is reaped by its `sh`: `hardkill-a`
  died with its launch log ending `137`, and `end` found `pid_answers` `false`. A launch that is not
  reaped by a parent that waits could meet it.
- **`[03-a-run-whose-manifest-cannot-be-read-names-no-store]`.** A live run whose `manifest.json` is
  missing or torn names no store, so the store operations proceed on the store it is writing. The
  manifest is written once, by `begin`, and nothing rewrites it; no known cause tears it.

**The issues that no longer bear on it:**
- `[a3-v2-a-hard-killed-run-blocks-its-store]`, closed by 03 (the `handover` board's §4);
- `[00-main-and-the-readers-default-to-ray-address-auto]`, closed by `handover` 05 (`359f3fd`): `main.py`
  and the readers default to `local`;
- `[04-runs-do-not-say-which-machine-produced-them]` and
  `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`, closed in RunRegistryKit
  and adopted here (§3.4). A second-machine build is now judged `foreign` rather than by this
  machine's process table; a handler's terminal state can no longer be lost to a heartbeat.

**Its store.** The A3 v2 store is **the first live store recorded in this layout that is meant to be
kept**. At `aa0b57c` every sidecar under `var/datastores/` is a tombstone (21 `retired`) but R3's three
(`rrk-v020-adoption-03-{reference,resumed,hardkill}`, live, `39809dca…`), which are throwaway and the
user's to retire, or not (README §3, §6.3). It is a new name, and `store show` must say `absent` for it
before the first launch.

**The disk.** `df -h /` is **94 % used, 64 GB free** (§3.2), against R2's 97 % (36 GB) and R3's 93 %.
Ray's spilling threshold is 95 %: about 17 GB of headroom on a 927 GB volume. R3's three stores and
their working files are on it. A3 v2 is long and wide and may spill; it is the machine's, not the
code's, as at R2.

**`CLAUDE.md` rule 5** (the user's ruling, README §6.1 **U4**) gains two sentences, after its sentence
on `end` and the resume and the unreaped process:

> After `end`, check `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes`: a
> `SIGKILL`ed driver's GCS server may be left alive, reparented to pid 1, stuck in its own shutdown
> where a `SIGTERM` cannot end it (`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`). One
> still listed ten minutes after the kill is ended **by a person** with `kill -KILL <pid>`, by its pid,
> never by pattern and never with `ray stop`; until then the rehearsal lock cannot be released.

Nothing else in `CLAUDE.md` changes.

## 4. After the close: clean refusals (05)

**The tree.** Prompt 05 ran on branch `handover-remedial` at **`affaf84`**, clean, after the campaign's
close (§3) and on the user's reversal of README §6.2 **C3** (§6.1 **U5**). It changes the two drivers
and adds one test module. Nothing above this section changes.

**What changed.** A refusal is a check that stops a driver before any work starts. Each driver now
defines `Refusal(RuntimeError)`, raises it at the refusal sites below, and prints it from its
`__main__` block:

```
!! <the refusal's message, unchanged>
```

as one line on **stderr** (a multi-line message prints as it is, with `!! ` before its first line
only), with **no traceback** and exit status **1**, which is the status a traceback gives. Everything
else keeps its traceback and its type: a failure during the work, and an internal invariant that does
not hold. `Refusal` is a `RuntimeError`, so every caller and test that catches `RuntimeError` still
does. `main()` itself still raises.

**The refusal sites** (`raise Refusal(`, by enclosing function):

| Driver | Function | Sites |
|---|---|---|
| `docs/handover/quadsource_atol_sweep.py` | `run_build` | 3: the existing store, the not-a-directory, `begin`'s |
| | `assert_store_is_self_consistent` | 1 |
| | `prepare` | 2: `--force`, `copy_store`'s |
| | `sweep` | 1: `begin`'s |
| `docs/gktk-remedial/scoped_pipeline_run.py` | `main` | 2: the existing datastore, the not-a-directory |
| | `register` | 2: the missing package, `begin`'s |

**What stays a `RuntimeError`** (a failure or an invariant): in the launcher the case selector, `substitute_main_py`,
the two model filters and the sweep's child exit (5); in the scoped driver the model filter and the
`main.py` literal count (2).

**`begin`'s refusals.** At each of the three `runregistrykit.begin(` calls (`run_build`, `sweep`,
`register`), a `RuntimeError` is re-raised as `Refusal(str(e))` from `e`, **after** a clause
`except runregistrykit.ConfigError: raise`. `runregistrykit.ConfigError` subclasses `RuntimeError` at
`v0.2.0` (`client.py:84`) and is not a refusal, so it, an `OSError` and a `FileExistsError` keep their
tracebacks. Read at the tag, `begin` raises a `RuntimeError` of its own in two places only, both
before the run directory is created and both ending `No run directory was created`: a retired store
(`__init__.py:1102`) and a store a run that has not finished names (`:1119`).

**The launcher's SHA-256**, beside §3.5's:

| Script | Before 05 (`13d993c`; the launcher's is R3's, §3.5) | After 05 |
|---|---|---|
| `docs/handover/quadsource_atol_sweep.py` | `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d` | `9b40e457f444db10d2b406b749e0b4391d53753e93c928fa431856b4c645e649` |
| `docs/gktk-remedial/scoped_pipeline_run.py` | `244ad70e7e0901c612bb01eb4f0fec10ee9bbd8e3bb3a17904024e111b6646df` | `d2e38510acf5cff09dc63219e5c59cead2eb28042ba58c770ab8abe90d972c83` |

The launcher's hash is **no longer R3's `2a985303…`**; this is expected, and A3 v2's manifests will
carry the new one. Nothing R3 measured is changed by it: the change is confined to what runs before
`begin` returns.

**What a log shows now.** A detached launch that is refused ends its `stdout.log`/`stderr.log` with the
one `!!` line, where it ended with a traceback's last line. A `--child` of the sweep that meets
`assert_store_is_self_consistent`'s refusal prints one `!!` line and exits 1, as before; the parent's
"child … exited 1" `RuntimeError` is unchanged and keeps its traceback. `CLAUDE.md` rule 5's route is
unchanged.

**The test** is `RunDrivers/tests/test_clean_refusals.py` (12 tests; `RunDrivers` is 103 → 115). Its
output of the seven roots is `prompts/rrk-v020-adoption/measure/out/05-roots.txt`.
