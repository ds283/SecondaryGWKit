# runregistry-extraction campaign — the three rehearsals

**Campaign:** [`prompts/runregistry-extraction/README.md`](../prompts/runregistry-extraction/README.md) ·
**Board:** [`prompts/runregistry-extraction/IMPLEMENTATION_STATE.md`](../prompts/runregistry-extraction/IMPLEMENTATION_STATE.md) ·
**Protocol:** README §4.1, which is `dk-and-rpk-migration`'s
([`dk-and-rpk-migration-verification.md`](dk-and-rpk-migration-verification.md) §1–§3) with the
names changed

This record holds the campaign's three rehearsal rebuilds, and its close, one section each:
- **§1, R0**: the start tree (prompt 02);
- §2, R1: through DatastoreKit's fingerprint release (prompt 04);
- §3, R2: through both packages, after the RunRegistryKit swap (prompt 08);
- §4: the close (prompt 09).

Each section quotes its commands verbatim and transcribes every measurement, because the stores
are throwaway. Each is compared with the rehearsal before it. R0's is `dk-and-rpk-migration`'s R2,
§3 of that campaign's record. Sections are added, never rewritten (`CLAUDE.md`, rule 6).

---

## 1. R0 — the start tree

**§1's tree, stores and runs.** Every R0 launch ran on branch `handover-remedial` at **`fe16d8e`**
("Write the orchestrator note for runregistry-extraction prompt 02").
- **Its code is `a45aefc`'s** (prompt 01). `cf978c6`, 01's review, changed two lines of `CLAUDE.md`.
  `git diff --stat a45aefc fe16d8e -- . ':!prompts'` lists that file alone, and no build, test or
  script reads it.
- **What it adds over `dk-and-rpk-migration`'s R2 (`5631f0e`)**:
  - `511ef5f`, which moved the object factories from `Datastore/SQL/ObjectFactories/` to
    `ObjectFactories/`;
  - `7d76d3e`, which moved the store fixtures and the inventory-report tests.

  No build has run through either before this one (README §0.2).
- **The packages.**
  - The datastore layer is the installed **`datastorekit` 0.2.2**, at `6d8ee9ae…`, not editable.
  - The work queue is the installed **`raypoolkit` 0.2.0**, at `eaef0ce6…`, not editable.
  - `runregistrykit` is not installed.
- **The tree was clean.** `git status --short` was empty at every launch.
- **The manifests.** Every manifest stamps `git_dirty: false` and `packages`
  `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false, "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version": "0.2.2"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}}`.
  That is R2's record, compared as JSON.

R2's launcher built two throwaway stores:
- **`var/datastores/runregistry-extraction-02-reference.sqlite`** (sidecar `store_id`
  `85b282b5425f43569f2b2386edefd688`), built straight through by the run **`reference`**;
- **`var/datastores/runregistry-extraction-02-resumed.sqlite`** (`store_id`
  `836df36b0e3f4c45b8a8dddffaa9f212`), built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

The runs are `var/runs/runregistry-extraction-02-<slug>-<stamp>/`. Both stores stay, with their
sidecars (README §3). The small outputs quoted below are in
[`runregistry-extraction/rehearsal/`](runregistry-extraction/rehearsal/), under names carrying `02`.
The large ones stay under `var/` (§1.9).

`dk-and-rpk-migration`'s six stores were retired at `718e650`. R0 is compared with R2 through the
fingerprints R2's `reference` and `resumed-c` run directories recorded (§1.3.4). The 14 sidecars
under `var/datastores/` were hashed before R0's first launch and after its last step (§1.9).

**Headline.**
1. **R0 reproduces the reference fingerprint at the start tree.**
   - Both stores are format 6, digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`,
     all 21 classes equal, and each equals its sidecar's.
   - `compare_fingerprints` is empty between the two stores.
   - It is also empty between each store and the fingerprints recorded by R2's
     `dk-and-rpk-migration-08-reference` and `dk-and-rpk-migration-08-resumed-c` (`5631f0e`)
     (§1.3.4).
   - The only change to what runs between R2 and R0 is the factory move. So the move is invisible
     to a build (§1.10).
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0. `resumed-a` and
   `resumed-b` are `killed`/143, each with its fingerprint, and each driver exited 143 (§1.2).
3. **The resume recomputed nothing it had stored, and this time the prune at open was exercised.**
   - `resumed-a` stored 164 `GkSource` rows before its signal, 63 of them unvalidated.
   - `resumed-b`'s open pruned exactly those 63: every serial is absent afterwards and none was
     reused.
   - `resumed-b` then computed 190 = 291 − 164 + 63.
   - `resumed-c` computed the 378 = 864 − 486 `QuadSourceIntegral` rows not stored when `resumed-b`
     was killed, and nothing else.
   - Nothing is duplicated (§1.3).
4. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§1.4).
5. **L4 equals R2's** (§1.5):
   - the statistics counts are equal, in `reference` and in both resumed runs that print the block;
   - the reference's stage counts are equal, stage by stage;
   - the resumed chain's identities hold, the prune's now with 63 rows;
   - the reference's `** ` lines are equal, in order;
   - every completion line has R2's form and number, and the registry's `stage` fields are R2's.

   The statistics block has R2's 14 lines, but `complete` and `minimal` under `mixed` are in the
   other order, which is R1's order. That is
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]` again, not a new finding.
6. **The first registered builds to print a pool progress line.** Under a load of 26–340, the
   reference's QSI stage took 20 m 30.1 s and printed three progress lines, and `resumed-a`'s
   GkWKB stage printed one. Each is in the copy's form, and the registry read one into
   `status.json`'s `stage` (§1.5.6).
7. **The logs differ from R2's only where the interruption points and the load put them** (§1.2.5).
   No R0 log holds a traceback, an import or unpickling error, or a warning that R2 or R1 did not
   print.
8. **Nothing else under `var/` moved.** The 14 sidecars have the same SHA-256, mtime and size before
   the first launch and after the last step (§1.9).

### 1.1 How each was launched

Each launch was README §4.1's command, verbatim but for the store, the slug, the purpose and
`--resume`, run from the repository root, detached. The purpose was handed to `sh -c` as its `$0`,
so that its parentheses and semicolons needed no quoting inside the command string. The Python
`argv` is R2's form exactly:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/runregistry-extraction-02-<store>.sqlite \
    --job-name runregistry-extraction-02 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "$0" --campaign runregistry-extraction --prompt 02; \
  echo "** driver exit status $?"' "<one line>" \
  > var/runs/runregistry-extraction-02-<slug>.launch.log 2>&1 &
disown
```

- `reference`: the reference store, no `--resume`.
- `resumed-a`: the resumed store, no `--resume`.
- `resumed-b` and `resumed-c`: the resumed store, with `--resume`.

Each `--purpose` names R0, this campaign, the start tree's commit and its code's, the factory move,
both packages and their commits, and the fingerprint it must equal
([`run_records_02.txt`](runregistry-extraction/rehearsal/run_records_02.txt)).

**One check per launch, one wait per launch.** After each launch the agent made one check:
- the manifest existed;
- `status.json` said `running`, with a pid answering `kill -0`;
- `RunRegistry list` showed the run.

The agent then ended its turn. The orchestrator ran one wait per launch, on the run's liveness, and
resumed the agent when the run was no longer alive. Neither polled. Every wait ended
`ENDED finished`, with the heartbeat never stale.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `runregistry-extraction-02-reference-20261012T091921` | reference | no | 69779 (69774) | `done` | 0 | 0 | 29 m 19 s (09:19:21–09:48:40) | 69.76 54.73 65.04 / 157.91 339.39 312.04 |
| `resumed-a` | `runregistry-extraction-02-resumed-a-20261012T095041` | resumed | no | 90851 (90847) | `killed` | 143 | 143 | 12 m 14 s (09:50:41–10:02:55) | 127.20 309.05 302.19 / 73.20 182.33 241.64 |
| `resumed-b` | `runregistry-extraction-02-resumed-b-20261012T100353` | resumed | yes | 99176 (99172) | `killed` | 143 | 143 | 6 m 37 s (10:03:53–10:10:30) | 57.90 169.94 235.46 / 103.45 109.75 175.56 |
| `resumed-c` | `runregistry-extraction-02-resumed-c-20261012T101145` | resumed | yes | 4791 (4615) | `done` | 0 | 0 | 5 m 31 s (10:11:45–10:17:16) | 90.21 106.44 172.86 / 92.01 107.30 146.96 |

- **Wall time** is from the manifest's `created` to the final heartbeat.
- **Load** is `uptime`'s on 10 CPUs: at the pre-launch check, and at the agent's first check after
  the run. The machine was shared, and heavily loaded by processes not of this repository. The
  one-minute load at the checks ran from 58 to 158. The five-minute load reached 339 at 09:49, during
  `reference`'s QSI stage. At the orchestrator's first reading, at 09:02, it was 177. R2's checks
  ran at 13–33. The load slowed every stage (§1.5.5), and changed nothing a build wrote.

**The sidecars** ([`store_sidecars_02.txt`](runregistry-extraction/rehearsal/store_sidecars_02.txt)):
- **The reference store.** `store create` ran on its closed primary at 09:49:54, after `reference`.
  `store fingerprint --write` followed: `39809dca…`, problems none, "it replaced none".
- **The resumed store.** `store create` ran on its closed primary at 10:03:38, after `resumed-a`.

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`. The commands are R2's,
with `runregistry-extraction-02` for `dk-and-rpk-migration-08`:

```bash
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" \
    var/datastores/runregistry-extraction-02-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint --write \
    var/datastores/runregistry-extraction-02-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

#### 1.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_02.txt`](runregistry-extraction/rehearsal/prelaunch_checks_02.txt)):
- **The tree.**
  - `git rev-parse HEAD` was `fe16d8e76e6d40d5e53eb6649a03834dda82be93`.
  - `git status --short` was empty, and was checked again in the same shell command that launched.
  - `git worktree list` showed this checkout alone.
  - Every working note and capture went under `var/runregistry-extraction-02/`, which is ignored,
    until the last launch had ended. This record's files were written afterwards.
- `RAY_ADDRESS` was unset.
- **The registry.** `python -m RunRegistry list` showed nothing `running`:
  - 29 runs before `reference`: 27 finished, and 2 `unknown`, the pre-registry `a3-pilot` pair.
    That is 01's baseline (`measure/out/01-list.txt`).
  - Then 30, 31 and 32 runs, all finished but those two.
- **No Ray process of this repository.**
  - The whole-table search, `ps -axo pid,ppid,lstart,command` for lines holding both
    `SecondaryGWKit/venv` and `ray`, was empty before each launch.
  - `pgrep -lf 'gcs_server|raylet|ray::'` was empty before `reference`, `resumed-a` and `resumed-b`.
  - Before `resumed-c` (10:11:13), `pgrep` listed three processes: 4084 `gcs_server`, 4173 `raylet`
    and 4178 `ray::DashboardAgent`.
    - They were a Ray cluster of **StochasticInstantons'** venv (Python 3.13), started at 10:11:04
      by another session. None of them was this repository's, and none was touched.
    - The build cannot join that cluster. It starts its own instance with
      `ray.init(address="local", …)` (`docs/handover/quadsource_atol_sweep.py:601-618`), and
      `/tmp/ray/ray_current_cluster` did not exist.
    - That cluster was gone by the check after `resumed-c`.
  - Before each later launch, the previous run's pids were checked by `ps -p`: the `sh` and Python
    pids, the interrupter's, and every Ray pid its logs name. None was alive.
- **The store names.**
  - Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run
    naming either.
  - Nothing under `var/datastores/` or `var/runs/` contained `runregistry-extraction`, and there
    was no `-journal`, `-wal` or `-shm` file under `var/datastores/`.
  - Before `resumed-a`, no `runregistry-extraction-02-resumed*` file existed.
- **The packages.**
  - `./venv/bin/pip show datastorekit raypoolkit` gave 0.2.2 and 0.2.0, each with `Location` this
    venv's `site-packages`.
  - `pip show runregistrykit` found nothing. `pip check` was clean.
  - `RunRegistry.package_provenance()` gave the record above. It equals, as JSON, the `packages` of
    R2's reference manifest.
  - `requirements.txt:17` is `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2`.
  - `requirements.txt:76` is `raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0`.
- **What runs.** In a fresh `python -I` with the repository on `sys.path`:
  - `datastorekit.__file__` and `raypoolkit.__file__` are under the venv's `site-packages`;
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`;
  - `ObjectFactories.__file__` is the repository's `ObjectFactories/__init__.py`;
  - `import Datastore.SQL.ObjectFactories` raises `ModuleNotFoundError: No module named 'Datastore'`;
  - `import runregistrykit` raises `ModuleNotFoundError`;
  - `ray.is_initialized()` is false.

  `Datastore/SQL/ObjectFactories` does not exist in the tree.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` was `b7935877…`, R2's.
  Its last change is `0c9e2cf`.
- **`var/datastores/`.** The 14 sidecars and nothing else. Their full SHA-256, mtime and size were
  recorded before `reference` (§1.9).
- 65–68 GB free on `var/`'s volume.

#### 1.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch.** Each ran into a new directory under the
agent's scratchpad, outside `var/` and the repository, with R2's commands:

```bash
bash docs/datastore-integrity-verification/interrupter_dummy_test.sh <scratch>/work/dummy-scoped
bash docs/a3-v2-readiness/interrupter_build_dummy_test.sh <scratch>/work/dummy-build
```

Both exited 0 ([`interrupter_dummy_test_02.txt`](runregistry-extraction/rehearsal/interrupter_dummy_test_02.txt),
[`interrupter_build_dummy_test_02.txt`](runregistry-extraction/rehearsal/interrupter_build_dummy_test_02.txt)).
- **The five cases, 09:18:33–09:18:58.**
  - A signalled at 45 % ≥ 40 % (child 143).
  - B exited silently when the child died first.
  - C refused `'sleep 30'` with exit 2, naming "`scoped_pipeline_run.py` (the run's own driver,
    its manifest's argv[0])", and the child lived.
  - D held 4 s and signalled (child 143).
  - E did not signal when the stage completed during the hold.
- **The `--build` cases, 09:18:58–09:18:59.**
  - A signalled the `quadsource_atol_sweep.py --build --register dummy-build-a` child
    (interrupter 0, child 143).
  - B refused the `scoped_pipeline_run.py` child (exit 2, child alive).
  - C: "the run's manifest names no argv[0]", exit 2, child alive.
- **Against R2's.** With times, pids and scratch paths stripped, each output equals R2's
  `interrupter_dummy_test_08.txt` (40 lines) and `interrupter_build_dummy_test_08.txt` (20 lines),
  line for line.
- **No dummy child or interrupter was left alive.** `ps -p` on all 16 pids the two outputs name,
  every `pid N` and every `child N`, found none.

**The points, from R0's own reference `stdout.log`, sized as the orchestrator's note directs.**
Both queues have a 300 s minimum notify interval (`main.py:119`).
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **1 m 18 s (78 s)** (R2:
  41.8 s; R1: 55.1 s; `dk-and-rpk-migration` R0: 32.2 s). It was interrupted from the banner with
  `--interval 5 --hold 31`.
  - That is 40 % of 78 s, rounded to whole seconds, and above the 10 s floor.
  - It puts the signal 31–36 s into the stage.
  - The hold is longer than R2's 10 s on purpose. R2's signal, 10–15 s into its stage, came before
    any `GkSource` row was stored, so R2's prune at open held only vacuously
    (`dk-and-rpk-migration-verification.md` §3.3.2). The point exists to leave stored and
    unvalidated rows for `resumed-b` to prune.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **20 m 30.1 s (1230.1 s)** (R2:
  299.2 s). It was interrupted from the banner with `--interval 5 --hold 250`.
  - A third of the stage is 410 s. The hold is capped at 250 s, so that it ends before the queue's
    300 s minimum can print a progress line.
  - The pattern is `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its end, as at R0, R1
    and R2. So a progress line during the hold could not have made the reading after it ineligible.

**The real firings** ([`interrupter_firings_02.txt`](runregistry-extraction/rehearsal/interrupter_firings_02.txt)):

```
resumed-a (interrupter pid 90922; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 31)
2026-10-12T10:00:50+01:00 liveness alive; pid 90851; stage 'CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS, 66.67% (8 of 24 work items remaining)'
2026-10-12T10:02:20+01:00 liveness alive; pid 90851; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-12T10:02:20+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 31 s, then reading again
2026-10-12T10:02:51+01:00 sent SIGTERM to pid 90851 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/runregistry-extraction-02-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 99219; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 250)
2026-10-12T10:05:42+01:00 liveness alive; pid 99176; stage 'APPLY GKSOURCE POLICIES'
2026-10-12T10:06:07+01:00 liveness alive; pid 99176; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-12T10:06:07+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 250 s, then reading again
2026-10-12T10:10:17+01:00 sent SIGTERM to pid 99176 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/runregistry-extraction-02-resumed.sqlite … --register resumed-b …'; exiting
```

- **Each interrupter checked the pid before it signalled.** It confirmed that the pid's command line
  named `quadsource_atol_sweep.py`, its manifest's `argv[0]`, and `--register <slug>`. It refused
  no real run's pid.
- **Each interrupter read its stage from `status.json`**, whose `stage` the registry takes from the
  pool's printed lines. Both banners were read, as at R2.
- **`resumed-a`'s interrupter also read a progress line.** Its reading at 10:00:50 is
  `CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS, 66.67% (8 of 24 work items remaining)`: the
  registry's form of the pool's progress line at `stdout.log:72` (§1.5.6).
- **The interrupter logs a reading only when the stage changes**, and reads every 5 s. So:
  - the `GkSource` stage began between 10:02:15 and 10:02:20, and the signal came 31–36 s into it;
  - the QSI stage began between 10:06:02 and 10:06:07, and the signal came 250–255 s into it.
- **No progress line appeared during `resumed-b`'s hold.**

Both interrupters were launched detached, right after their builds, once the run directory and its
`status.json` existed:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/runregistry-extraction-02-resumed-a-20261012T095041 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 31 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/runregistry-extraction-02-resumed-b-20261012T100353 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 250 > /dev/null 2>&1 &
disown
```

### 1.2 L1 — every launch ends as recorded

#### 1.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | not written: "…runregistry-extraction-02-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `9bb753bd…`, problems `{}` | absent | not written: "…runregistry-extraction-02-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `a0d053af…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint holds 21 classes, and its `taken` names its run, `git_head` `fe16d8e7…` and
`git_dirty` false.

The full digests:
- `reference` and `resumed-c`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `9bb753bd48450e64c6ad8dd3254c35ce91786363eb91d7735fa29c43b98fae18`;
- `resumed-b`: `a0d053af1167bd327bebfd4d5c5d207563c49ce37db8c1d9af3baca8ff744225`.

The two intermediate digests differ from R2's (`ff6c4257…`, `aa860ed8…`), as they must: each
digests a store killed at a different point. By their recorded class counts:
- `resumed-a`'s store held `GkSource` 164, `GkSourcePolicyData` 0 and `QuadSourceIntegral` 0;
- `resumed-b`'s held 291, 291 and 486.

#### 1.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 10:02:51 | 10:10:17 |
| Handler's line | `** SIGTERM at 2026-10-12 10:02:51 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-12 10:10:18 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s (equal whole-second stamps) | at most about 1 s (stamps one second apart) |
| `stderr.log`'s handler lines | the line above, then `** SIGTERM: the pipeline ended with SystemExit(143); recording killed, exit code 143` | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Stage at the signal | the `GkSource` banner, first read 10:02:20, signal 31 s later (**164 of 291 rows stored, 63 unvalidated**) | the `QuadSourceIntegral` banner, first read 10:06:07, signal 250 s later (486 of 864 stored) |

- **`SystemExit(143)` in `resumed-a`, `SystemExit(1)` in `resumed-b`.**
  - At R2 both were `SystemExit(1)`; at `dk-and-rpk-migration`'s R0, `resumed-a` was `143` and
    `resumed-b` `1`, as here.
  - A signal that lands during `ray.wait` has Ray's signal check convert the handler's
    `SystemExit(143)`; one that lands elsewhere does not. The latch is what `finish_after_exec`
    reads, so the record is `killed`/143 and the exit is 143 either way.
- **`stderr.log` of `resumed-a`** holds only the handler's two lines.
- **`stderr.log` of `resumed-b`** holds the handler's two lines and, before them, the one SciPy
  `IntegrationWarning` from a `compute_QuadSource_integral` worker (pid 99267). At R1 it was also in
  `resumed-b`; at R2 it was in `resumed-c` (§1.2.5).
- **`stderr.log` of `reference`** holds only that `IntegrationWarning` ("The occurrence of roundoff
  error is detected…", worker pid 69964), as at R0, R1 and R2.
- **`stderr.log` of `resumed-c` is empty.**

#### 1.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 09:49:43 (~63 s after the final heartbeat; load 157.91) | none: `pgrep` empty, whole-table search empty, pids 69774 and 69779 gone, and the Ray pids its logs name (69821, 69824, 69825, 69964) gone |
| `resumed-a` | 10:03:18 (~23 s; load 73.20) | none, by the same checks for pids 90847, 90851, the interrupter 90922, and the Ray pids 90983, 90995, 90996 |
| `resumed-b` | 10:10:54 (~24 s; load 103.45) | none, by the same checks for pids 99172, 99176, the interrupter 99219, and the Ray pids 99264, 99265, 99266, 99267 |
| `resumed-c` | 10:17:45 (~29 s; load 92.01) | none, for pids 4615, 4791 and the Ray pids 4898, 4901. `pgrep`, with other repositories' venvs excluded by path, was empty, and the StochasticInstantons cluster of §1.1.1 was gone |

- **No Ray process outlived either killed driver**, so no wait by pid was run. The agents' exit time
  is "none listed at the first check", as at `dk-and-rpk-migration`'s R0, R1 and R2.
- `[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur, at loads of 73–103 at the
  checks. Its record has the agents outliving the driver at loads of 65–123. The issue stays open;
  it is not this campaign's.
- No launch log or `stderr.log` holds a `ray.kill`, killed-actor, `ActorDiedError` or
  `RayActorError` line (§1.2.5).

#### 1.2.4 The manifests

From [`run_records_02.txt`](runregistry-extraction/rehearsal/run_records_02.txt), for all four:
- `git_head` `fe16d8e76e6d40d5e53eb6649a03834dda82be93`, the dispatch `HEAD`; `git_dirty` false.
- **`packages`** equals, compared as JSON, R2's manifests' `packages` and the record of §1's
  header: `datastorekit` 0.2.2 at `6d8ee9ae…`, `raypoolkit` 0.2.0 at `eaef0ce6…`, neither
  editable, each with its URL and `requested_revision`.
- `script` is `docs/handover/quadsource_atol_sweep.py`. **`script_sha256`** is
  `b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`, the same in all four, and
  R2's.
- `argv[0]` is `docs/handover/quadsource_atol_sweep.py`. `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only.
- `results` names the reference store for `reference`, and the resumed store for the other three.
  `results_store_id` is `null` for `reference` and `resumed-a`, whose stores had no sidecar at
  launch, and `836df36b…` for `resumed-b` and `resumed-c`, as at R2.
- `scope`, in each, is R2's with `runregistry-extraction-02` for `dk-and-rpk-migration-08`:
  `REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32
  rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into
  runregistry-extraction-02-<store>.sqlite`.

#### 1.2.5 Every launch's `stdout.log`, and what is new in the logs

Counted in each of the four, as R2's §3.2.5
([`log_comparison_02.txt`](runregistry-extraction/rehearsal/log_comparison_02.txt)):

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--zend', '250', …]` | 1 |
| `--no-prune-unvalidated`, in `stdout.log` or the launch log | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |
| `reconcil`, `repair`, `in.flight`, `replicat` (case-insensitive), in `stdout.log` or the launch log | 0 |
| `Traceback`, `ModuleNotFoundError`, `ImportError`, `AttributeError`, `pickle`/`unpickl`, `Datastore.SQL`/`Datastore/SQL`, `ray.kill`, `killed actor`, `ActorDiedError`, `RayActorError`, `ValueError`, `RuntimeWarning`, `did not validate`, in any of the three logs | 0 |
| `INTEGRITY WARNING`, in any of the three logs | 0, but for `resumed-b`: 2 in `stdout.log` and 2 in the launch log |

The launch logs of `resumed-b` and `resumed-c` hold `** quadsource_atol_sweep --build: resuming
into runregistry-extraction-02-resumed.sqlite`.

**The comparison with R2, slug by slug.**
- **The method** is R2's §3.2.5. Each line of each R0 `stderr.log`, launch log and `stdout.log` was
  normalised and compared, as a set of line kinds, with R2's same file for the same slug. The
  normalisation:
  - ANSI colour codes stripped;
  - the campaign and prompt and the repository path replaced;
  - Ray's `[repeated Nx across cluster]` count replaced;
  - every number replaced by `N`, which removes times, pids, serials and counts.
- **The script** was a throwaway in the agent's scratchpad, not committed; the normalisation is the
  one above.

| Slug | `stderr.log` | launch log | `stdout.log` |
|---|---|---|---|
| `reference` | equal (4 lines, 4 kinds each) | R0 has 3 kinds R2 has not, the progress lines; 122 lines against 116 | the same 3 kinds; 116 lines against 110 |
| `resumed-a` | equal (2, 2) | R0 has 2 kinds R2 has not, one progress line and its `inflight` line; R2 has 1 kind R0 has not, a `Queue summary` in `/sec` that R0 printed in `/min`; 90 lines against 88 | the same 3 kinds; 86 lines against 84 |
| `resumed-b` | R0 has 4 kinds R2 has not: the `IntegrationWarning`'s four lines; 6 lines against 2 | R0 has 17 kinds R2 has not: the same 4, the integrity warning's 12, and a completion time in `Nm Ns`; 162 lines against 108 | R0 has 13 kinds R2 has not: the integrity warning's 12 and the completion time; 153 lines against 103 |
| `resumed-c` | R2 has 4 kinds R0 has not: the `IntegrationWarning`'s four lines; 0 lines against 4 | the same 4 kinds; 110 lines against 114 | equal (107 lines, 58 kinds each) |

**Every difference, explained.**
1. **The pool's progress lines**, in `reference` (three, in the QSI stage) and `resumed-a` (one, in
   the GkWKB stage). Each stage ran past the queue's 300 s minimum interval under this load, so the
   pool printed its periodic line, as it does in any stage that long (§1.5.6). The orchestrator's
   note anticipated this (correction 6). The threshold and the load explain it; it is not a finding.
2. **`resumed-a`'s `Queue summary` rate unit.** The GkNumeric stage's summary reads
   `1368 lookup (avg 305.3/min), …` where R2's read `1368 lookup (avg 15.91/sec), …`: about 5 per
   second here against 16 at R2. The line's form is the same, and the pool chooses the unit from the
   rate, as its other summaries show (`3 lookup (avg 52.54/sec)` in the same log). The slower rate
   is the load again.
3. **The SciPy `IntegrationWarning`** ("The occurrence of roundoff error is detected…", from a
   `compute_QuadSource_integral` worker).
   - It is in `reference` at every rehearsal.
   - In the resumed chain it is in `dk-and-rpk-migration`'s R0 `resumed-c`, R1 `resumed-b`, R2
     `resumed-c`, and here in `resumed-b`.
   - It belongs to one integral, which falls in whichever launch computes it. R0's `resumed-b`
     stored 486 QSI rows before its signal, against R2's 281 and R1's 334, so here the integral
     fell before the point.
   - The point moves it. It is not a finding.
4. **`resumed-b`'s integrity warning**, which R2's lacked and R1's had.
   - At its open, `resumed-b` printed each affected shard's `!! INTEGRITY WARNING`, the list of
     unvalidated `GkSource` objects with their `contains N z-sample values` lines, and
     `** Note: these values have been pruned from the datastore.`
   - Ray folds the three shards' blocks: shard 1's in full at `stdout.log:13`, then shard 0's at
     `:142` with `[repeated 2x across cluster]`. The folding varies run to run.
   - Here `resumed-a` left 63 rows unvalidated (§1.3.2); at R2 it left none. So the point puts these
     12 kinds, and the 50 lines `stdout.log` is longer by, in this log.
   - The prune is the package's shard actors' at open, the same `datastorekit` 0.2.2 as R1's and
     R2's.
5. **`resumed-b`'s `GkSource` completion time** is `in time 1m 10s`, where R2's was `30.4s`. It took
   longer under the load and computed fewer rows. The line's form is the pool's, which prints
   minutes past 60 s.
6. **The handler's `SystemExit(143)` in `resumed-a`** is where the signal landed (§1.2.2).

**Nothing else differs.** No line from the factory move appears: no import, unpickling or attribute
error, and no module path naming `Datastore.SQL`. No warning or traceback appears that no earlier
rehearsal printed. Every pool line, banner, completion and `Queue summary` alike, has R2's form
(§1.5.6).

#### 1.2.6 A3 v2's guard

After `resumed-c` and the readers, in the foreground, with no `--register`
([`guard_02.txt`](runregistry-extraction/rehearsal/guard_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/runregistry-extraction-02-resumed.sqlite \
    --job-name runregistry-extraction-02 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (10:28:47–10:28:48), with:

```
RuntimeError: quadsource_atol_sweep --build: runregistry-extraction-02-resumed.sqlite, runregistry-extraction-02-resumed-shard0000.sqlite, runregistry-extraction-02-resumed-shard0001.sqlite, runregistry-extraction-02-resumed-shard0002.sqlite, runregistry-extraction-02-resumed-shard0003.sqlite already exist(s). Refusing to build into an existing store: a comparator that silently inherited earlier rows is one nobody can interpret. Pass --resume to continue an interrupted build at the SAME tolerance pair, or choose a path that does not exist.
```

- It names the five files that exist.
- **Nothing changed.** Afterwards, `shasum -a 256 -c` gave `OK` on every file of both stores and
  both sidecars:
  - against the hashes taken just before it, after the readers;
  - and against the hashes taken after `resumed-c`, before the readers.
- **Nothing was created.** No run directory appeared (63 entries under `var/runs/` before and after),
  and no Ray process of this repository and no journal appeared.

### 1.3 L2 — the resume is the reference

#### 1.3.1 Nothing stored is recomputed

Through `dk-and-rpk-migration`'s
[`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py), unchanged
([`stage_counts_02.txt`](runregistry-extraction/rehearsal/stage_counts_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/stage_counts.py \
    var/runs/runregistry-extraction-02-reference-20261012T091921 \
    var/runs/runregistry-extraction-02-resumed-a-20261012T095041 \
    var/runs/runregistry-extraction-02-resumed-b-20261012T100353 \
    var/runs/runregistry-extraction-02-resumed-c-20261012T101145
```

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
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **190/190/190** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **378/378/378** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored **164** `GkSource` rows before the signal (57 + 51 + 0 + 56 by shard), of
    which **63** were unvalidated (23 + 20 + 0 + 20) (§1.3.2).
  - `resumed-b` pruned those 63 at open, and computed **190 = 291 − 164 + 63**: the 127 never
    stored, and the 63 pruned. No validated row was computed again.
  - Summed, less the pruned: 164 + 190 − 63 = 291, the reference's.
- **`QuadSourceIntegral`.**
  - `resumed-b` printed no progress line in the stage. Its fingerprint, taken by `finish` on the
    closed store, records **486** rows.
  - `resumed-c` computed **378** = 864 − 486, exactly what was missing.
  - Summed: 486 + 378 = 864, the reference's. No stored row was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b`, because `resumed-a` never reached it, and
  0/0/0 in `resumed-c`.

#### 1.3.2 The prune at startup

The closed store after each launch, read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`. Each file's SHA-256 was unchanged by the
read ([`closed_store_state_02.txt`](runregistry-extraction/rehearsal/closed_store_state_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/a3-v2-readiness/rehearsal/closed_store_state.py \
    var/datastores/runregistry-extraction-02-<store>.sqlite
```

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **`GkSource` 23 + 20 + 0 + 20 = 63**, of 57 + 51 + 0 + 56 = 164 stored, each listed by serial. `GkNumericIntegration` 456 + 406 + 0 + 506 = 1368 and `GkWKBIntegration` 3 × 1159 = 3477 are all stored and validated |
| `resumed-b` | none | 0 rows | none (`GkSource` 97 + 97 + 0 + 97 = 291) |
| `resumed-c` | none | 0 rows | none |

**`resumed-a` stored and left unvalidated rows, as the point intended** (§1.1.2). The signal came
31–36 s into a stage the reference took 78 s over. 164 of 291 rows were stored by then, and 63 of
them were not yet validated. R1's `resumed-a` left 101 rows (45 unvalidated); R2's left none.

**By serial**, through `dk-and-rpk-migration`'s
[`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py), unchanged
([`prune_check_02.txt`](runregistry-extraction/rehearsal/prune_check_02.txt)):

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/runregistry-extraction-02/state/after-resumed-a.txt \
    var/datastores/runregistry-extraction-02-resumed.sqlite
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/runregistry-extraction-02/state/after-resumed-b.txt \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

- **The first ran twice:** at 10:11:06, after `resumed-b` had opened and closed the store, and again
  after `resumed-c`. Both times it gave:
  - `63 serial(s) checked: 63 absent, 0 reused by a validated row, 0 still unvalidated`, exit 0;
  - by shard, 23 serials in shard 0, 20 in shard 1 and 20 in shard 3, each `absent`.
- **The second**, after `resumed-c`, gave `none: there is nothing to check`, exit 0.
- Every file of the store had the same SHA-256 before and after each check.

**So the prune at open removed exactly the rows `resumed-a` left unvalidated**, and no validated row.
Each pruned serial is gone, and none was reused, so the 63 recomputed rows took new serials. R2 held
this only vacuously; R0 exercises it, as R1 did, through the same `datastorekit` 0.2.2.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§1.3.5), so no `duplicate`. Each finish's fingerprint is built from the package's
`read_inventory`, and its `problems` is `{}` for all four launches.

**The pool's prune of a replicated class.** None. No launch printed a prune of a replicated class.
After every launch each shard's `BackgroundModel` row is validated, and the census finds the
replicated tables equal (§1.3.5).

#### 1.3.3 The version row

From `closed_store_state.py`, after every launch: exactly one row, `(1, '2025.1.1')`, on each of the
four shards of both stores. The primary has no `version` table.

#### 1.3.4 The comparison

**The two stores**, by `compare_stores.py`
([`fingerprint_comparison_02.txt`](runregistry-extraction/rehearsal/fingerprint_comparison_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \
    var/datastores/runregistry-extraction-02-reference.sqlite \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

Exit 0:

```
reference: …/var/datastores/runregistry-extraction-02-reference.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches
resumed: …/var/datastores/runregistry-extraction-02-resumed.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches

compare_fingerprints(reference, resumed): 0 difference(s)
verdict: equal
```

All 21 classes are `equal`. With the campaign and prompt normalised, the output equals R2's
`fingerprint_comparison_08.txt` line for line.

**Against R2's reference and `resumed-c`.** R2's stores are retired. Each R0 store was
fingerprinted read-only again and compared with the fingerprint recorded in each R2 run's
`status.json`, by `dk-and-rpk-migration`'s
[`compare_with_recorded.py`](dk-and-rpk-migration/rehearsal/compare_with_recorded.py), unchanged
([`fingerprint_against_recorded_02.txt`](runregistry-extraction/rehearsal/fingerprint_against_recorded_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/runregistry-extraction-02-reference.sqlite \
    var/datastores/runregistry-extraction-02-resumed.sqlite \
    --against var/runs/dk-and-rpk-migration-08-reference-20261011T224347 \
    var/runs/dk-and-rpk-migration-08-resumed-c-20261011T230717
```

It exited 0, verdict `equal`. For each store there were 0 differences against:
- its sidecar;
- R2's reference (taken at `5631f0ed…`, `git_dirty` false);
- R2's `resumed-c` (taken at `5631f0ed…`, `git_dirty` false).

All are format 6, digest `39809dca…`, 21 classes, problems none. The files of both stores had the
same SHA-256 before and after these reads (`after-resumed-c.sha256`, all `OK`).

| Class | Reference | Resumed | Digests | R2's reference |
|---|---|---|---|---|
| `version` | 1 | 1 | equal | 1 |
| `store_tag` | 10 | 10 | equal | 10 |
| `redshift` | 1159 | 1159 | equal | 1159 |
| `wavenumber` | 3 | 3 | equal | 3 |
| `tolerance` | 7 | 7 | equal | 7 |
| `LambdaCDM` | 1 | 1 | equal | 1 |
| `QCD_Cosmology` | 1 | 1 | equal | 1 |
| `IntegrationSolver` | 7 | 7 | equal | 7 |
| `GkSourcePolicy` | 2 | 2 | equal | 2 |
| `QuadSourcePolicy` | 2 | 2 | equal | 2 |
| `wavenumber_exit_time` | 3 | 3 | equal | 3 |
| `BackgroundModel` | 1 | 1 | equal | 1 |
| `TkNumericIntegration` | 3 | 3 | equal | 3 |
| `TkWKBIntegration` | 3 | 3 | equal | 3 |
| `GkNumericIntegration` | 1368 | 1368 | equal | 1368 |
| `GkWKBIntegration` | 3477 | 3477 | equal | 3477 |
| `GkSource` | 291 | 291 | equal | 291 |
| `GkSourcePolicyData` | 291 | 291 | equal | 291 |
| `QuadSource` | 6 | 6 | equal | 6 |
| `QuadSourceIntegral` | 864 | 864 | equal | 864 |
| `OneLoopIntegral` | 0 | 0 | equal | 0 |

#### 1.3.5 Integrity, both stores, read-only

**The census** ([`census_02.txt`](runregistry-extraction/rehearsal/census_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py \
    var/datastores/runregistry-extraction-02-reference.sqlite \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

- Every table is `OK` on both stores (28 lines), and the verdict is `no divergence` on both. Exit 0.
- With the campaign and prompt normalised, the output equals R2's `census_08.txt` line for line.

**`integrity_check`, `foreign_key_check` and the inventory**
([`store_soundness_02.txt`](runregistry-extraction/rehearsal/store_soundness_02.txt)), through
`dk-and-rpk-migration`'s [`store_soundness.py`](dk-and-rpk-migration/rehearsal/store_soundness.py),
unchanged:

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py \
    var/datastores/runregistry-extraction-02-reference.sqlite \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

- It exited 0, `=> clean`.
- On the primary and every shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is
  `['ok']` and `PRAGMA foreign_key_check` returns 0 rows.
- The package's `read_inventory(primary, factories)`, given the factories from their new home
  `ObjectFactories/`, reads shards `(0, 1, 2, 3)`. It reports no problem in any class of either
  store, and 0 `duplicate`.
- Each class's record count is §1.3.4's table.
- With the campaign and prompt normalised, the output equals R2's `store_soundness_08.txt` line for
  line.
- Every file of both stores had the same SHA-256 before and after these reads, and no journal
  appeared.

### 1.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance
(`--ray-address local`), through `dk-and-rpk-migration`'s
[`run_readers.sh`](dk-and-rpk-migration/rehearsal/run_readers.sh), unchanged
([`readers_02.txt`](runregistry-extraction/rehearsal/readers_02.txt)):

```bash
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/runregistry-extraction-02-resumed.sqlite var/runregistry-extraction-02/extract
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/runregistry-extraction-02-resumed.sqlite var/runregistry-extraction-02/extract \
    nomodels extract_Gk_data
```

Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` / traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 67 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 396 | none | identical |
| `extract_GkWKB_data.py` | 0 | 201 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 58 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 210 | none | identical |
| `extract_TkWKB_data.py` | 0 | 34 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | **0** | 35 s | `EXTRACT TENSOR SOURCE DATA`: 6, all six `QuadSource` rows | 27 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 52 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 | 76 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 66 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 459 | none | identical |

- **The lookups equal R2's, queue by queue and reader by reader**
  ([`log_comparison_02.txt`](runregistry-extraction/rehearsal/log_comparison_02.txt), section 4).
- **The readers take the factories from their new home.** Each reader imports `factories` from
  `config.datastore`, which imports them from `ObjectFactories/` since `511ef5f`, and hands them to
  its pool. None raised on an import or an unpickling (§1.10).
- **The files written differ** in the readers that draw random subsamples, as between R0, R1 and R2.
  Against R2:
  - `extract_Gk_data.py`: 396 against 441; without `--models`, 459, equal to R2's;
  - `extract_GkSource_data.py`: 210 against 174;
  - `extract_QuadSourceIntegral_data.py`: 76 against 71.

  Each draws its subsample with `random.sample`, so the count varies run to run. It is not a finding.
- **`extract_GkWKB_data.py`'s queue printed two progress lines**, as R2's did. Normalised, they have
  R2's form:

  ```
     -- 2026-10-12 10:22:03 (1m 3.95s running): 238/348 work items remaining = 31.61% complete
        inflight: 0 lookup | completed: 110 lookup (avg 103.2/min)
     -- 2026-10-12 10:23:04 (2m 5.58s running): 108/348 work items remaining = 68.97% complete
        inflight: 0 lookup | completed: 240 lookup (126.6/min, avg 114.7/min)
  ```
- **The reader logs' line kinds against R2's**, normalised as §1.2.5:
  - `extract_Gk_data`, `extract_TkWKB_data`, `extract_tensor_source_data`,
    `extract_QuadSourceIntegral_data` and `extract_Gk_data` without `--models`: equal.
  - `extract_GkWKB_data`: R2 has four kinds R0 has not. They are R2's two matplotlib `UserWarning`s
    from one `plot_Gk` worker, which R2's record traced to a random draw
    (`dk-and-rpk-migration-verification.md` §3.4). R0's draw did not produce them.
  - `extract_GkSource_data`: R0 has one kind R2 has not. It is the same
    `MatplotlibDeprecationWarning` R2's log prints (`extract_GkSource_data.py:649`, "The x parameter
    as float was deprecated…"), folded by Ray with `[repeated Nx across cluster]`. R0's log prints it
    five times, R2's twice. The folding varies run to run.
- **The load.** The readers ran 10:19:29–10:28:15, at a one-minute load of 41–106 from processes
  not of this repository.
- **The model and the run.**
  - Every run printed `** reading run "default", the only run in this datastore (no --run-label
    given)`.
  - With `--models LambdaCDM` each printed `>> RUNNING PIPELINE FOR MODEL LambdaCDM` and nothing
    about `QCD_Cosmology`.
  - Without `--models`, `extract_Gk_data.py` read LambdaCDM, then printed `>> RUNNING PIPELINE FOR
    MODEL QCD_Cosmology` and the skip line, and exited 0.
- **`extract_tensor_source_data.py` exits 0.**
- **No file of the store changed.**
  - Before and after each reader, the SHA-256, mtime and size of the primary and of every shard were
    identical, and no `-journal`, `-wal` or `-shm` file appeared.
  - The SHA-256s are those taken after `resumed-c` (`after-resumed-c.sha256`).
  - No Ray process of this checkout was left after any reader.
- **The shard-count warning.** Each log opens `!! WARNING: number of shards read from database (=4)
  does not match specified number of shards (=10)`, as at every earlier rehearsal.
- **The fingerprint afterwards.** `store fingerprint` (without `--write`) on the resumed store gave
  `39809dca…`, problems none, "matches the fingerprint taken 2026-10-12T10:17:15+01:00 by run
  runregistry-extraction-02-resumed-c-20261012T101145, at fe16d8e76e6d40d5e53eb6649a03834dda82be93".

### 1.5 L4 — equal to `dk-and-rpk-migration`'s R2

#### 1.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/policy_statistics.py \
    var/runs/runregistry-extraction-02-reference-20261012T091921
```

Exit 0. The reference run's block, lines 83–96 of its `stdout.log`, verbatim
([`policy_statistics_02_reference.txt`](runregistry-extraction/rehearsal/policy_statistics_02_reference.txt)):

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

**Against R2's `policy_statistics_08_reference.txt`.**
- **Line for line, it differs.** `diff` of R2's block against R0's:

  ```
  10d9
  <            minimal: 1 instances = 3.85% of this type, 0.34% of total
  11a11
  >            minimal: 1 instances = 3.85% of this type, 0.34% of total
  ```

  Under `>> mixed`, R2 printed `minimal` then `complete`; R0 prints `complete` then `minimal`.
- **Sorted, the blocks are equal.**
- **R0's block equals `dk-and-rpk-migration`'s R1 `policy_statistics_05_reference.txt` line for
  line.**

This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]`: the order is Ray's
completion order, and the counts are the measurement.

| | R2 `reference` | R0 `reference` |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| Order under `mixed` | `minimal`, `complete` | `complete`, `minimal` |

**R0's resumed runs**, against R2's same slug
([`policy_statistics_02_resumed-b.txt`](runregistry-extraction/rehearsal/policy_statistics_02_resumed-b.txt),
[`policy_statistics_02_resumed-c.txt`](runregistry-extraction/rehearsal/policy_statistics_02_resumed-c.txt)):
- **`resumed-b`** computed all 291 policy rows, and its block is at `stdout.log:125`. It has every
  count R2's `resumed-b` had, and prints `complete` first where R2's printed `minimal` first. Sorted,
  the two are equal. It equals R0's own reference line for line.
- **`resumed-c`** computed none, and its block is at `stdout.log:83`. It equals R2's `resumed-c`
  line for line, `minimal` first.
- **`resumed-a`** has no block, and the script exits 1 on it, as at R2.

#### 1.5.2 The reference run's stage counts

From [`stage_counts_02.txt`](runregistry-extraction/rehearsal/stage_counts_02.txt), against R2's
`stage_counts_08.txt`, stage by stage:

| Stage | R2 `reference` | R0 `reference` |
|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | 291/291/291 |
| APPLY GKSOURCE POLICIES | 291/291/291 | 291/291/291 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | 864/864/864 |

Equal in every stage. The `resumed-a` column is also R2's, entry for entry, since both were killed
in the same stage.

#### 1.5.3 The resumed chain, by identity

The raw counts of `resumed-b` and `resumed-c` depend on where each launch was killed, and R0 chose
its own points (§1.1.2). So L4 compares what holds whatever the point:

| Identity | R2 | R0 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated (`prune_check.py`) | 0 = 0, nothing to check | **63 = 63, all absent, none reused** | ✅ |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | pruned 0; computed 291 − 0 + 0 = 291 | **190 = 291 − 164 + 63** | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 583 = 864 − 281 | 378 = 864 − 486 | ✅ |
| Totals per class over the chain equal the reference's | `GkSource` 0 + 291 − 0 = 291; QSI 281 + 583 = 864 | `GkSource` 164 + 190 − 63 = 291; QSI 486 + 378 = 864 | ✅ |

**R0 exercises the `GkSource` prune at open**, which R2 did not (§1.3.2).

What each resumed launch pruned and recomputed:

| Launch | Pruned at open | Computed, stage by stage | Of which recomputing a pruned row |
|---|---|---|---|
| `resumed-a` | nothing (new store) | everything up to the `GkSource` stage, as `reference`; **164 `GkSource` stored** at the signal, 63 unvalidated | — |
| `resumed-b` | **63 `GkSource`** (the unvalidated ones), in shards 0, 1 and 3 | `GkSource` 190; `GkSourcePolicyData` 291; `QuadSourceIntegral` 486 stored at the signal; every earlier stage 0 | 63 `GkSource` |
| `resumed-c` | nothing | `QuadSourceIntegral` 378; every other stage 0 | — |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 1.5.4 The stage sequence

The lines of each reference `stdout.log` that begin `** `
([`log_comparison_02.txt`](runregistry-extraction/rehearsal/log_comparison_02.txt)):
- R2's and R0's each hold 26 such lines.
- With the campaign and prompt, the repository path and every number normalised, the two sequences
  are equal, in order.
- Of the 26, 15 are the pipeline's stage banners, `** BUILD ARRAY OF WAVENUMBERS TO SAMPLE` through
  `** CALCULATE QUADRATIC SOURCE INTEGRALS`. They are equal verbatim, in order, with nothing
  normalised. The other 11 are the launcher's own lines.

#### 1.5.5 The stage times

Not compared (`ALL WORK ITEMS COMPLETE in time …` in each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1 m 53.5 s | 4 m 28.9 s | 5.44 s | 5.2 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 4 m 46.9 s | 6 m 32 s, one progress line | 13.7 s | 12.9 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **1 m 18 s**, no progress line | killed | 1 m 10 s | 14.5 s |
| APPLY GKSOURCE POLICIES | 24.6 s | — | 24.6 s | 12 s |
| QUADRATIC SOURCE INTEGRALS | **20 m 30.1 s**, three progress lines | — | killed, no progress line | 4 m 24 s |

- Each earlier stage took under 4 s.
- **The reference took 29 m 19 s**, against R2's 11 m 25 s, R1's 12 m 13 s and
  `dk-and-rpk-migration` R0's 8 m 0 s.
- **The `GkSource` stage took 78 s and the QSI stage 1230.1 s**, against R2's 41.8 s and 299.2 s.
  The five-minute load during the reference's QSI stage reached 339, against R2's 13–33 at its
  checks. The stores are equal, so the time is the machine's.
- `resumed-c`'s QSI stage, 378 rows in 264 s, ran at a load near 100, and printed no progress line.

#### 1.5.6 The printed lines and the registry's fields

The pool's printed lines are an interface: `StageTracker` and `interrupt_when.py` read them. Against
R2's same slug ([`log_comparison_02.txt`](runregistry-extraction/rehearsal/log_comparison_02.txt)):

| Line, in `stdout.log` | `reference` R0 / R2 | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| `** ` (banners and the launcher's lines) | 26 / 26 | 23 / 23 | 26 / 26 | 26 / 26 |
| `-- ALL WORK ITEMS COMPLETE in time …` | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `Queue summary: …`, one after each completion line | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `(no compute items were evaluated - all assets read from datastore)` | 1 / 1 | 1 / 1 | 7 / 7 | 9 / 9 |
| progress: `n/m work items remaining = p% complete` | **3** / 0 | **1** / 0 | 0 / 0 | 0 / 0 |
| progress: `inflight: … \| completed: …` | **3** / 0 | **1** / 0 | 0 / 0 | 0 / 0 |

- **Every completion line is followed by its `Queue summary` line**, in all four.
- **The forms are R2's.** Normalised as §1.2.5, the completion lines of R0's four runs make 2
  kinds, the `Queue summary` lines 7 and the `(no compute items …)` line 1. Each set equals R2's.
- **R0's builds printed progress lines, and R2's did not.** These are the first progress lines any
  registered build has printed under `raypoolkit`; R2's observation 4 recorded that none had.
  - In the reference's QSI stage, at 5 m 21.5 s, 10 m 30.1 s and 16 m 21 s:

    ```
       -- 2026-10-12 09:33:22 (5m 21.5s running): 0/2 work items remaining = 100.00% complete (may be waiting for compute/store/validate tasks to finish)
          inflight: 0 lookup, 608 compute, 76 store | completed: 864 lookup (avg 161.2/min), 256 compute (avg 47.77/min), 180 store (avg 33.59/min)
    ```

    The later two print two rates for each kind, as in `864 lookup (0/day, avg 82.27/min)`. The
    first is 0 because no lookup completed after the first line. R2's GkWKB reader's second line has
    the same two-rate form, `250 lookup (87.21/min, avg 119.5/min)`.
  - In `resumed-a`'s GkWKB stage, at 5 m 0.6 s: `8/24 work items remaining = 66.67% complete`,
    then its `inflight: … validation | completed: …` line.
  - **The form is the copy's.** It matches the last build line of the copy,
    `var/runs/a3-v2-readiness-07-reference-20261004T012909/stdout.log:100-101`, field for field:
    `-- 2026-10-04 01:40:11 (5m 10.2s running): 0/2 work items remaining = 100.00% complete (may be
    waiting for compute/store/validate tasks to finish)`, then `inflight: 0 lookup, 261 compute,
    70 store | completed: 864 lookup (avg 167.1/min), …`. It also matches R2's GkWKB reader's two
    lines.
  - **The registry read them.** `resumed-a`'s interrupter, reading `status.json`, logged the stage
    `CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS, 66.67% (8 of 24 work items remaining)` at
    10:00:50, two seconds after the line at `stdout.log:72`. That is `StageTracker`'s form for a
    progress line (`docs/gktk-remedial/scoped_pipeline_run.py:142-160`).
  - **Why.** Each of those stages ran past the queue's 300 s minimum notify interval (`main.py:119`)
    under this load. That is the threshold and the load, as the orchestrator's note expected
    (correction 6), not a finding.
- **The registry's fields at each run's end**, R0 against R2:

| Slug | `state` | `stage` | `units_done` / `units_total` | R2's `stage` |
|---|---|---|---|---|
| `reference` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` | the same |
| `resumed-a` | `killed` | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | 0 / `null` | the same |
| `resumed-b` | `killed` | `CALCULATE QUADRATIC SOURCE INTEGRALS` | 0 / `null` | the same |
| `resumed-c` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` | the same |

  Each `status.json` has R2's keys exactly. The killed runs' `stage` is the stage they were killed
  in, which the interrupter read from it. The registry's progress, `units_done`/`units_total`, is
  0/`null` in all four, as at R2: a `main.py` pipeline reports no total, and a progress line moves
  `stage`, not those fields. `RunRegistry list` shows `-` under PROGRESS for each. No field R2
  filled is empty at R0.

### 1.6 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen. After every launch each shard's
  `BackgroundModel` row is validated (§1.3.2), and no resumed launch printed a prune of it.
- **An in-flight record left by a killed launch.** Did not happen. `replication_in_flight` had 0 rows
  after every launch, the two killed ones included.
- **A repair by the check at open.** Did not happen. No `stdout.log` or launch log holds `reconcil`,
  `repair`, `in.flight` or `replicat` (§1.2.5).
- **A Ray agent outliving a killed driver.** Did not happen (§1.2.3).
- **An import or unpickling failure from the factory move.** Did not happen (§1.10):
  - no `ModuleNotFoundError`, `ImportError`, `AttributeError` or pickle error, in any build or
    reader;
  - no traceback in any `stderr.log` or launch log;
  - no reader that refused or failed;
  - no `ReadOnlyMiss` or `ReadOnlyWrite`.
- **A printed line the registry or the interrupter could not read.** Did not happen (§1.5.6).
- **A missed interruption.** Did not happen. Both killed launches ended `killed`, in the stage the
  interrupter waited for.

### 1.7 Issues opened

None. Every difference from R2 is explained by:
- where a launch was killed;
- Ray's completion order (an issue already open);
- the readers' random subsamples;
- or the load, with the pool's 300 s notify threshold.

### 1.8 The suites

After the last launch, the readers and the guard: README §8's command, each root in the
foreground, serially, its output in the agent's scratchpad and its verdict taken by `grep`. Every
count is 01's and the dispatch baseline's.

| Root | Ran | Result | Load (1 min) at start |
|---|---|---|---|
| AdaptiveLevin | 32 | OK | 67.9 |
| CosmologyModels | 39 | OK | 63.5 |
| ObjectFactories | 271 | OK | 63.5 |
| tools | 49 | OK | 51.1 |
| LiouvilleGreen | 148 | OK (1 skipped) | 39.3 |
| RunRegistry | 286 | OK | 29.0 |
| ComputeTargets | 693 | 1 failure: `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (0.0658 s against 0.06 s) | 24.9 |

**The one failure is README §8's first known wall-clock flake**, as at dispatch (0.073 s). Run alone
three times at 10:38, at a load of 17, it passed, passed, and failed (0.0709 s). It is a wall-clock
assertion with about 1× margin, not a defect of this tree. Every other test passed.

No Python file was added, so `black --check` had nothing to check.

### 1.9 What stays under `var/`

- `var/runs/runregistry-extraction-02-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`. Each
  holds the run's manifest, `status.json` with its full fingerprint, `stdout.log` and `stderr.log`.
  For `resumed-a` and `resumed-b` it also holds `interrupter.log`.
- `var/runs/runregistry-extraction-02-<slug>.launch.log`, beside each run directory.
- `var/datastores/runregistry-extraction-02-{reference,resumed}.sqlite`, four shards each, and each
  store's `.manifest.json` sidecar. Shard 2 of each, 1 077 248 bytes, holds only replicated rows.
- `var/runregistry-extraction-02/`, which holds:
  - the agent's working notes (`NOTES.md`);
  - the pre-launch captures, the dummy-test outputs and the listings of `var/datastores/`;
  - the closed-store states, the prune checks and the hashes (`state/`);
  - the check outputs (`checks/`), among them every run's statistics block;
  - the readers' outputs and logs (`extract/`, `readers.txt`).

**Nothing else under `var/` moved**
([`var_datastores_02.txt`](runregistry-extraction/rehearsal/var_datastores_02.txt)).
- Before the first launch, `var/datastores/` held the 14 sidecars and nothing else.
  `find var/datastores -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256` was
  `023d139fb714e229f96c6fb61894daf526b768fd603307d153d4d91810cd2626`.
- After the last step it held 26 files. The 14 sidecars have the same SHA-256, mtime (to the
  nanosecond) and size as before, and the same digest over them, `023d139f…`. The 12 new files are
  this rehearsal's two stores and their sidecars.
- No `-journal`, `-wal` or `-shm` file is under `var/datastores/`.
- `RunRegistry list` shows 33 runs: 01's 29, and this rehearsal's four, all finished but the two
  `unknown` of the `a3-pilot` pair.

### 1.10 The factory move, seen by a build

The orchestrator's R7: the move `511ef5f` must be invisible to a build. This section gathers the
evidence.

**What moved.** `511ef5f` moved `Datastore/SQL/ObjectFactories/` to `ObjectFactories/` with
`git mv`, deleted the empty `Datastore` packages, and rewrote every import. `7d76d3e` gave the store
fixtures and the inventory-report tests their own homes; no build or reader imports either.

**What a build hands Ray.** `main.py` and every reader take `factories` from `config.datastore`,
which imports all of them from `ObjectFactories/` (`config/datastore.py:28-86`). The pool hands them
to its shard actors, so Ray pickles them by their module path.
- In a fresh `python -I`, all 38 registered factories have a module under `ObjectFactories`, and
  none under `Datastore`.
- `import Datastore.SQL.ObjectFactories` raises `ModuleNotFoundError: No module named 'Datastore'`.
- `git grep` finds no `Datastore.SQL` or `Datastore/SQL` in `main.py`, `config/`, `extract_common.py`,
  the six readers, or `ObjectFactories/` outside its tests.

**What the record shows.**
- **No import or unpickling error anywhere.** In the twelve build logs (§1.2.5) and the seven reader
  logs (§1.4) there are 0 lines of `Traceback`, `ModuleNotFoundError`, `ImportError`,
  `AttributeError`, `pickle` or `unpickl`, and none naming `Datastore.SQL`.
- **Every class was stored and read back through the new path.** All 21 fingerprinted classes have
  their reference counts in both stores (§1.3.4). `read_inventory` read every class with no problem
  (§1.3.5). The resumed launches looked up every stored class in a store an earlier process wrote
  (§1.3.1), and every reader read its classes back (§1.4).
- **The fingerprint did not move.** Both stores are `39809dca…`, equal class by class to the
  fingerprints R2 recorded at `5631f0e`, before the move.

So the factory move changes nothing a build writes, reads or records. R1 can isolate the
DatastoreKit pin.

---

## 2. R1 — through DatastoreKit `v0.4.0`

**§2's tree, stores and runs.** Every R1 launch ran on branch `handover-remedial` at **`89ff8b7`**
("Write the orchestrator note for runregistry-extraction prompt 04").
- **Its code is `5b81f2e`'s** (prompt 03). `git diff --stat 5b81f2e 89ff8b7 -- . ':!prompts' ':!docs'`
  is empty: `e150335` and `89ff8b7` touch `prompts/` only.
- **What it adds over R0's tree (`fe16d8e`)**:
  - `5b81f2e`, which moved `requirements.txt:17` from `datastorekit` `v0.2.2` to `v0.4.0`;
  - `c863055` (02a), which added `tools/rehearsal_lock.py`, its tests and `CLAUDE.md`'s lock
    section. No build or reader imports any of them.

  `git diff --stat fe16d8e 5b81f2e -- . ':!prompts' ':!docs'` lists those four files alone.

  Under `datastorekit/`, `v0.2.2` → `v0.4.0` changes `SQL/factory_base.py` (`v0.2.3`) and
  `object.py` (`v0.3.0`), and adds `store_fingerprint.py`. `store_inventory.py`, which SGK's
  fingerprint reads, is unchanged (board §5.5).
- **The packages.**
  - The datastore layer is the installed **`datastorekit` 0.4.0**, at `c8570e84…`, not editable.
  - The work queue is the installed **`raypoolkit` 0.2.0**, at `eaef0ce6…`, not editable, as at R0.
  - `runregistrykit` is not installed.
- **The tree was clean.** `git status --short` was empty at every launch.
- **The manifests.** Every manifest stamps `git_dirty: false` and `packages`
  `{"datastorekit": {"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false, "requested_revision": "v0.4.0", "url": "https://github.com/ds283/DatastoreKit", "version": "0.4.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}}`,
  compared as JSON. The `raypoolkit` entry is R0's exactly.
- **The lock.** R1 is the first rehearsal under the shared rehearsal lock (README §4.1). It was
  taken before `reference`, its `pid` set at each launch, held through the readers and the guard,
  and released (§2.1.3).

R1's launcher built two throwaway stores:
- **`var/datastores/runregistry-extraction-04-reference.sqlite`** (sidecar `store_id`
  `0ab77b25f9e74b20977a7dba409c1e32`), built straight through by the run **`reference`**;
- **`var/datastores/runregistry-extraction-04-resumed.sqlite`** (`store_id`
  `bc2ac248bfab423cb8ef47be37efdcdd`), built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

The runs are `var/runs/runregistry-extraction-04-<slug>-<stamp>/`. Both stores stay, with their
sidecars. After this prompt's review the orchestrator retires them and R0's two (D6, board §5.5).
The small outputs quoted below are in
[`runregistry-extraction/rehearsal/`](runregistry-extraction/rehearsal/), under names carrying `04`.
The large ones stay under `var/` (§2.9).

R1 is compared with R0 through R0's stores, which are still live, and through the fingerprints R0's
`reference` and `resumed-c` run directories recorded (§2.3.4). The 26 files under `var/datastores/`
were hashed before R1's first launch and after its last step (§2.9).

**Headline.**
1. **R1 reproduces the reference fingerprint through `datastorekit` 0.4.0.**
   - Both stores are format 6, digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`,
     all 21 classes equal, and each equals its sidecar's.
   - `compare_fingerprints` is empty between the two stores.
   - It is also empty between each store and the fingerprints recorded by R0's
     `runregistry-extraction-02-reference` and `runregistry-extraction-02-resumed-c` (`fe16d8e`)
     (§2.3.4).
   - The only change to what runs between R0 and R1 is the pin. So the three releases between
     `v0.2.2` and `v0.4.0` move nothing a build writes or reads.
2. **The fingerprint both ways holds on all four live stores (W1, §2.10).**
   `fingerprint_both_ways.py` exits 0 on R1's two stores and R0's two. On each, the package's
   fingerprint equals SGK's in every field but `encoding_format`: the same keys in the same order,
   the overall digest, all 21 classes in order with every count and digest, and every tag set. The
   listings are 7542 lines each, equal but for the first. Each of the five deliberate breakages
   makes it exit 1, naming what failed.
3. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0. `resumed-a` and
   `resumed-b` are `killed`/143, each with its fingerprint, and each driver exited 143 (§2.2).
4. **The resume recomputed nothing it had stored, and the prune at open was exercised again.**
   - `resumed-a` stored 207 `GkSource` rows before its signal, 48 of them unvalidated.
   - `resumed-b`'s open pruned exactly those 48: every serial is absent afterwards and none was
     reused.
   - `resumed-b` then computed 132 = 291 − 207 + 48.
   - `resumed-c` computed the 440 = 864 − 424 `QuadSourceIntegral` rows not stored when `resumed-b`
     was killed, and nothing else.
   - Nothing is duplicated (§2.3).
5. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§2.4).
6. **L4 equals R0's** (§2.5):
   - the statistics counts are equal, in `reference` and in both resumed runs that print the block;
   - the reference's stage counts are equal, stage by stage;
   - the resumed chain's identities hold, the prune's with 48 rows;
   - the `** ` lines of every run are equal, in order;
   - every completion line has R0's form and number, and the registry's `stage` fields are R0's.

   The reference's statistics block prints `minimal` before `complete` under `mixed`, R2's order
   and not R0's. That is `[02-the-policy-statistics-block-orders-its-lines-by-completion]` again,
   not a new finding.
7. **No log line names `datastorekit`**, in any of the twelve build logs, as in R0's. None holds a
   `DeprecationWarning` or `_timestamp` (§2.2.5). The logs differ from R0's only where the
   interruption points and the load put them.
8. **The lock held for the whole rehearsal, and is free again** (§2.1.3).
9. **Nothing else under `var/` moved.** The 14 tombstones and R0's twelve store files have the same
   SHA-256, mtime and size before the first launch and after the last step (§2.9).

### 2.1 How each was launched

Each launch was §1.1's command, with `runregistry-extraction-04`, `--prompt 04` and this prompt's
store names, run from the repository root, detached, the purpose passed to `sh -c` as its `$0`:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/runregistry-extraction-04-<store>.sqlite \
    --job-name runregistry-extraction-04 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "$0" --campaign runregistry-extraction --prompt 04; \
  echo "** driver exit status $?"' "<one line>" \
  > var/runs/runregistry-extraction-04-<slug>.launch.log 2>&1 &
disown
```

- `reference`: the reference store, no `--resume`.
- `resumed-a`: the resumed store, no `--resume`.
- `resumed-b` and `resumed-c`: the resumed store, with `--resume`.

Each `--purpose` names R1 of this campaign, the tree's commit and its code's, `datastorekit` 0.4.0
at `c8570e8` and `raypoolkit` 0.2.0, and the fingerprint it must equal
([`run_records_04.txt`](runregistry-extraction/rehearsal/run_records_04.txt)).

**One check per launch, one wait per launch.** After each launch the agent made one check:
- the manifest existed;
- `status.json` said `running`, with a pid answering `kill -0`;
- `RunRegistry list` showed the run;
- in the same check, `tools.rehearsal_lock set-pid` with that pid (§2.1.3), and, for `resumed-a`
  and `resumed-b`, the interrupter's launch.

The agent then ended its turn. The orchestrator ran one wait per launch on the run's liveness, and
resumed the agent when the run was no longer alive. Neither polled. Every wait ended
`ENDED finished`, with the heartbeat never stale. The wait on `resumed-a` ended when
`status.json` first said `killed`, 5–6 s before the fingerprint was written; the orchestrator's
waits on `resumed-b` and `resumed-c` ended on the pid's death (§2.6).

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `runregistry-extraction-04-reference-20261014T154651` | reference | no | 3243 (3239) | `done` | 0 | 0 | 13 m 57 s (15:46:51–16:00:48) | 11.37 31.55 34.93 / 100.13 94.44 62.59 |
| `resumed-a` | `runregistry-extraction-04-resumed-a-20261014T160226` | resumed | no | 11561 (11558) | `killed` | 143 | 143 | 8 m 49 s (16:02:26–16:11:15) | 59.78 84.55 60.76 / 20.53 61.19 67.51 |
| `resumed-b` | `runregistry-extraction-04-resumed-b-20261014T161222` | resumed | yes | 17571 (17565) | `killed` | 143 | 143 | 4 m 16 s (16:12:22–16:16:38) | 16.46 56.99 65.82 / 56.62 50.74 59.30 |
| `resumed-c` | `runregistry-extraction-04-resumed-c-20261014T161739` | resumed | yes | 20688 (20685) | `done` | 0 | 0 | 7 m 12 s (16:17:39–16:24:51) | 40.63 47.46 57.87 / 38.73 59.13 59.23 |

- **Wall time** is from the manifest's `created` to the final heartbeat.
- **Load** is `uptime`'s on 10 CPUs: at the pre-launch check, and at the agent's first check after
  the run. The machine was shared, and loaded by processes not of this repository: the one-minute
  load at the checks ran from 11 to 100, and reached 161 at 16:29, during the readers. R0's checks
  ran at 58–158, with a five-minute load up to 339. The load slowed the stages (§2.5.5), and changed
  nothing a build wrote.

**The sidecars** ([`store_sidecars_04.txt`](runregistry-extraction/rehearsal/store_sidecars_04.txt)):
- **The reference store.** `store create` ran on its closed primary at 16:01:53, after `reference`.
  `store fingerprint --write` followed: `39809dca…`, problems none, "it replaced none".
- **The resumed store.** `store create` ran on its closed primary at 16:12:07, after `resumed-a`.

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`. The commands are §1.1's,
with `runregistry-extraction-04` for `runregistry-extraction-02`.

#### 2.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_04.txt`](runregistry-extraction/rehearsal/prelaunch_checks_04.txt)):
- **The tree.**
  - `git rev-parse HEAD` was `89ff8b728b61ba22452110de88718094361e4e15`.
  - `git status --short` was empty, and was checked again in the same shell command that launched.
  - `git worktree list` showed this checkout alone.
  - `git diff --stat 5b81f2e HEAD -- . ':!prompts' ':!docs'` was empty.
  - Every working note and capture went under `var/runregistry-extraction-04/`, which is ignored,
    until the last launch had ended. This record's files, and the W1 script, were written
    afterwards.
- `RAY_ADDRESS` was unset.
- **The registry.** `python -m RunRegistry list` showed nothing `running`: 33 runs before
  `reference` (31 finished, 2 `unknown`, the pre-registry `a3-pilot` pair); then 34, 35 and 36, all
  finished but those two.
- **No Ray process of this repository.** Before each launch `tools.rehearsal_lock ray-processes`
  exited 0, listing nothing. This is the test (README §4.1). The machine-wide
  `pgrep -lf 'gcs_server|raylet|ray::'` was also recorded, and was empty (rc 1) before every launch:
  no other project's cluster was up. Before each later launch the previous run's pids were checked
  by `ps -p`: the `sh` and Python pids, the interrupter's, and every Ray pid its logs name. None was
  alive.
- **The lock** was free before `reference` (`status`: `lock: free`, exit 0), and
  `~/ScienceData/` held `ChamPBH-rehearsals/` alone.
- **The store names.**
  - Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run
    naming either.
  - Nothing under `var/datastores/` or `var/runs/` contained `runregistry-extraction-04`, and there
    was no `-journal`, `-wal` or `-shm` file under `var/datastores/`.
  - Before `resumed-a`, no `runregistry-extraction-04-resumed*` file existed.
- **The packages.**
  - `./venv/bin/pip show datastorekit raypoolkit` gave 0.4.0 and 0.2.0, each with `Location` this
    venv's `site-packages`.
  - `pip show runregistrykit` found nothing. `pip check` was clean.
  - `RunRegistry.package_provenance()` gave the record above, equal as JSON to the expected record,
    and its `raypoolkit` equal to R0's reference manifest's.
  - `requirements.txt:17` is `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.4.0`;
    `requirements.txt:76` is `raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0`.
- **What runs.** In a fresh `python -I` with the repository on `sys.path`:
  - `datastorekit.__file__` and `raypoolkit.__file__` are under the venv's `site-packages`;
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`;
  - `ObjectFactories.__file__` is the repository's `ObjectFactories/__init__.py`;
  - `datastorekit.store_fingerprint.ENCODING_FORMAT` is `1`;
  - `import runregistrykit` raises `ModuleNotFoundError`;
  - `ray.is_initialized()` is false.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` was `b7935877…`, R0's.
  Its last change is `0c9e2cf`.
- **`var/datastores/`.** 26 files: the 14 tombstones and R0's twelve. Their full SHA-256, mtime and
  size were recorded before `reference` (§2.9).
- 72 GB free on `var/`'s volume (93 % used).

#### 2.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch.** Each ran into a new directory under the
agent's scratchpad, with §1.1.2's commands. Both exited 0
([`interrupter_dummy_test_04.txt`](runregistry-extraction/rehearsal/interrupter_dummy_test_04.txt),
[`interrupter_build_dummy_test_04.txt`](runregistry-extraction/rehearsal/interrupter_build_dummy_test_04.txt)):
the five cases 15:45:47–15:46:11, the `--build` cases 15:46:11–15:46:12.
- **Against R0's.** With times, pids and scratch paths stripped, each output equals R0's
  `interrupter_dummy_test_02.txt` (40 lines) and `interrupter_build_dummy_test_02.txt` (20 lines),
  line for line.
- **No dummy child or interrupter was left alive.** `ps -p` on all 16 pids the two outputs name
  found none.

**The points, from R1's own reference `stdout.log`, sized as the orchestrator's note directs.**
Both queues have a 300 s minimum notify interval (`main.py:119`).
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **1 m 42.9 s (102.9 s)**
  (R0: 78 s; `dk-and-rpk-migration` R2: 41.8 s). It was interrupted from the banner with
  `--interval 5 --hold 41`.
  - That is 40 % of 102.9 s, rounded to whole seconds, and above the 10 s floor.
  - It puts the signal 41–46 s into the stage, so that `resumed-a` leaves stored and unvalidated
    rows for `resumed-b` to prune, as at R0.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **8 m 14.9 s (494.9 s)** (R0:
  1230.1 s). It was interrupted from the banner with `--interval 5 --hold 165`.
  - A third of the stage is 165 s, under the 250 s cap that keeps the hold inside the queue's 300 s
    minimum.
  - The pattern is `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its end, as at R0.

**The real firings** ([`interrupter_firings_04.txt`](runregistry-extraction/rehearsal/interrupter_firings_04.txt)):

```
resumed-a (interrupter pid 11805; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 41)
2026-10-14T16:06:28+01:00 liveness alive; pid 11561; stage 'CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS'
2026-10-14T16:10:28+01:00 liveness alive; pid 11561; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-14T16:10:28+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 41 s, then reading again
2026-10-14T16:11:09+01:00 sent SIGTERM to pid 11561 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/runregistry-extraction-04-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 17691; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 165)
2026-10-14T16:13:22+01:00 liveness alive; pid 17571; stage 'APPLY GKSOURCE POLICIES'
2026-10-14T16:13:37+01:00 liveness alive; pid 17571; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-14T16:13:37+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 165 s, then reading again
2026-10-14T16:16:22+01:00 sent SIGTERM to pid 17571 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/runregistry-extraction-04-resumed.sqlite … --register resumed-b …'; exiting
```

- **Each interrupter checked the pid before it signalled**: its command line named
  `quadsource_atol_sweep.py`, its manifest's `argv[0]`, and `--register <slug>`. It refused no
  real run's pid.
- **Each interrupter read its stage from `status.json`.** Both banners were read, as at R0. No
  progress line was read: `resumed-a`'s GkWKB stage took 4 m 0.4 s here and printed none.
- **The interrupter logs a reading only when the stage changes**, and reads every 5 s. So:
  - the `GkSource` stage began between 16:10:23 and 16:10:28, and the signal came 41–46 s into it;
  - the QSI stage began between 16:13:32 and 16:13:37, and the signal came 165–170 s into it.
- **No progress line appeared during `resumed-b`'s hold.**

Both interrupters were launched detached, about 10 s after their builds, once the run directory and
its `status.json` existed, in the one check after the launch:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/runregistry-extraction-04-resumed-a-20261014T160226 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 41 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/runregistry-extraction-04-resumed-b-20261014T161222 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 165 > /dev/null 2>&1 &
disown
```

#### 2.1.3 The lock

Every helper command and its exit status
([`lock_04.txt`](runregistry-extraction/rehearsal/lock_04.txt)), each run as
`PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock <command>` from the repository root:

| When | Command | Exit | What it said |
|---|---|---|---|
| 15:46:42 | `acquire --purpose "runregistry-extraction 04 (R1)"` | 0 | acquired, for SecondaryGWKit |
| 15:46:42 | `status` | 0 | `HELD`; `pid: -` (no pid recorded yet); no Ray process |
| 15:47:01 | `set-pid 3243` (`reference`) | 0 | recorded |
| 15:47:01 | `status` | 0 | `pid: 3243`, alive; `reference`'s own Ray processes |
| 16:01:26 | `status` | 0 | `pid: 3243`, NOT alive; no Ray process |
| 16:02:37 | `set-pid 11561` (`resumed-a`) | 0 | recorded |
| 16:02:38 | `status` | 0 | `pid: 11561`, alive; 14 Ray processes, the run's |
| 16:11:46 | `status` | 0 | `pid: 11561`, NOT alive; no Ray process |
| 16:12:32 | `set-pid 17571` (`resumed-b`) | 0 | recorded |
| 16:12:32 | `status` | 0 | `pid: 17571`, alive; 14 Ray processes, the run's |
| 16:17:05 | `status` | 0 | `pid: 17571`, NOT alive; no Ray process |
| 16:17:48 | `set-pid 20688` (`resumed-c`) | 0 | recorded |
| 16:17:49 | `status` | 0 | `pid: 20688`, alive; 14 Ray processes, the run's |
| 16:25:50 | `status` | 0 | `pid: 20688`, NOT alive; no Ray process |
| 16:33:42 | `status` | 0 | `pid: 20688`, NOT alive; no Ray process (after the readers and the guard) |
| 16:33:42 | `ray-processes` | 0 | none |
| 16:33:42 | `release` | 0 | released |
| 16:33:42 | `status` | 0 | `lock: free`; no Ray process |
| 16:33:42 | `ray-processes` | 0 | none |

- The owner file named `project: SecondaryGWKit`, `purpose: runregistry-extraction 04 (R1)`,
  `ray_temp_dir: in-process` and `ray_address: in-process` throughout.
- Between launches the lock was held and its pid was the previous driver's, dead: the protocol, not a
  stale lock.
- `release` was not refused: the recorded pid was dead and no Ray process of this repository was
  alive. No wait by pid was needed before it (§2.2.3).
- Afterwards `~/ScienceData/` holds `ChamPBH-rehearsals/` alone, as before.

### 2.2 L1 — every launch ends as recorded

#### 2.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | not written: "…runregistry-extraction-04-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `a410cce0…`, problems `{}` | absent | not written: "…runregistry-extraction-04-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `8146dab8…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint holds 21 classes, and its `taken` names its run, `git_head` `89ff8b72…` and
`git_dirty` false.

The full digests:
- `reference` and `resumed-c`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `a410cce07d9fb7a424e64ebd0b7819a096b9075ad7394ed5e78599b9f3cf8fe6`;
- `resumed-b`: `8146dab80c46050bff5e33e77f3ec288b3dd644b0fe4448ac9fb6f96b15ff446`.

The two intermediate digests differ from R0's (`9bb753bd…`, `a0d053af…`), as they must: each
digests a store killed at a different point. By their recorded class counts:
- `resumed-a`'s store held `GkSource` 207, `GkSourcePolicyData` 0 and `QuadSourceIntegral` 0;
- `resumed-b`'s held 291, 291 and 424.

#### 2.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 16:11:09 | 16:16:22 |
| Handler's line | `** SIGTERM at 2026-10-14 16:11:09 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-14 16:16:23 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s (equal whole-second stamps) | at most about 1 s (stamps one second apart) |
| `stderr.log`'s handler lines | the line above, then `** SIGTERM: the pipeline ended with SystemExit(143); recording killed, exit code 143` | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Stage at the signal | the `GkSource` banner, first read 16:10:28, signal 41 s later (**207 of 291 rows stored, 48 unvalidated**) | the `QuadSourceIntegral` banner, first read 16:13:37, signal 165 s later (424 of 864 stored) |

- **`SystemExit(143)` in `resumed-a`, `SystemExit(1)` in `resumed-b`**, as at R0. The record is
  `killed`/143 and the exit is 143 either way (§1.2.2).
- **`stderr.log` of `resumed-a` and of `resumed-b`** each hold only the handler's two lines.
- **`stderr.log` of `reference`** holds only the SciPy `IntegrationWarning` ("The occurrence of
  roundoff error is detected…", worker pid 3309), as at every rehearsal.
- **`stderr.log` of `resumed-c`** holds the same warning (worker pid 20790). At R0 it was in
  `resumed-b` (§2.2.5).

#### 2.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 16:01:25 (~37 s after the final heartbeat; load 100.13) | none: `ray-processes` exit 0, `pgrep` empty, pids 3239 and 3243 gone, and the Ray pids its logs name (3289, 3291, 3293, 3309) gone |
| `resumed-a` | 16:11:46 (~31 s; load 20.53) | none, by the same checks for pids 11558, 11561, the interrupter 11805, and the Ray pids 11779, 11782, 11783 |
| `resumed-b` | 16:17:05 (~27 s; load 56.62) | none, by the same checks for pids 17565, 17571, the interrupter 17691, and the Ray pids 17643, 17645, 17646, 17648 |
| `resumed-c` | 16:25:50 (~59 s; load 38.73) | none, for pids 20685, 20688 and the Ray pids 20785, 20787, 20790 |

- **No Ray process outlived either killed driver**, so no wait by pid was run. The agents' exit time
  is "none listed at the first check", as at R0.
- `[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur. The issue stays open; it is
  not this campaign's.
- No launch log or `stderr.log` holds a `ray.kill`, killed-actor, `ActorDiedError` or
  `RayActorError` line (§2.2.5).

#### 2.2.4 The manifests

From [`run_records_04.txt`](runregistry-extraction/rehearsal/run_records_04.txt), for all four:
- `git_head` `89ff8b728b61ba22452110de88718094361e4e15`, the dispatch `HEAD`; `git_dirty` false.
- **`packages`** equals, compared as JSON, the record of §2's header: `datastorekit` 0.4.0 at
  `c8570e84…`, `requested_revision` `v0.4.0`, and `raypoolkit` 0.2.0 at `eaef0ce6…`, R0's, neither
  editable, each with its URL.
- `script` is `docs/handover/quadsource_atol_sweep.py`. **`script_sha256`** is
  `b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`, the same in all four, and
  R0's.
- `argv[0]` is `docs/handover/quadsource_atol_sweep.py`. `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only.
- `results` names the reference store for `reference`, and the resumed store for the other three.
  `results_store_id` is `null` for `reference` and `resumed-a`, whose stores had no sidecar at
  launch, and `bc2ac248…` for `resumed-b` and `resumed-c`, as at R0.
- `scope` is R0's with `runregistry-extraction-04` for `runregistry-extraction-02`. The manifests
  have R0's keys exactly.

#### 2.2.5 Every launch's `stdout.log`, and what is new in the logs

Counted in each of the four, as §1.2.5
([`log_comparison_04.txt`](runregistry-extraction/rehearsal/log_comparison_04.txt), section 2):

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--zend', '250', …]` | 1 |
| `--no-prune-unvalidated`, in `stdout.log` or the launch log | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |
| `reconcil`, `repair`, `in.flight`, `replicat` (case-insensitive), in `stdout.log` or the launch log | 0 |
| `Traceback`, `ModuleNotFoundError`, `ImportError`, `AttributeError`, `pickle`/`unpickl`, `Datastore.SQL`/`Datastore/SQL`, `ray.kill`, `killed actor`, `ActorDiedError`, `RayActorError`, `ValueError`, `RuntimeWarning`, `did not validate`, in any of the three logs | 0 |
| **`datastorekit`**, anywhere in a line, `DeprecationWarning`, `_timestamp`, in any of the three logs | **0**, as in R0's twelve |
| `INTEGRITY WARNING`, in any of the three logs | 0, but for `resumed-b`: 2 in `stdout.log` and 2 in the launch log |

The launch logs of `resumed-b` and `resumed-c` hold `** quadsource_atol_sweep --build: resuming
into runregistry-extraction-04-resumed.sqlite`.

**The three releases' signatures are absent.** No line names `datastorekit`, even through a path:
a traceback or warning raised inside the package would. No `DeprecationWarning` and no
`_timestamp`, the attribute `v0.3.0`'s `DatastoreObject` added, appear. The counts of every search
equal R0's but for `IntegrationWarning` and the progress lines, below.

**The comparison with R0, slug by slug.** The method is §1.2.5's: each line of each R1 log
normalised (ANSI codes stripped, the campaign and prompt and the repository path replaced, Ray's
`[repeated Nx across cluster]` count and every number replaced by `N`) and compared, as a set of
line kinds, with R0's same file for the same slug. The script was a throwaway in the agent's
scratchpad, not committed.

| Slug | `stderr.log` | launch log | `stdout.log` |
|---|---|---|---|
| `reference` | equal (4 lines, 4 kinds each) | R0 has 2 kinds R1 has not: a `Queue summary` in `/sec`, and the two-rate `inflight` line of a later progress line; 118 lines against 122 | the same 2 kinds; 112 lines against 116 |
| `resumed-a` | equal (2, 2) | R1 has 1 kind R0 has not, a `Queue summary` in `/sec` that R0 printed in `/min`; R0 has 2 kinds R1 has not, its one progress line and its `inflight` line; 88 lines against 90 | the same 3 kinds; 84 lines against 86 |
| `resumed-b` | R0 has 4 kinds R1 has not: the `IntegrationWarning`'s four lines; 2 lines against 6 | R0 has 5 kinds R1 has not: the same 4, and a completion time in `Nm Ns`; 148 lines against 162 | R0 has 1 kind R1 has not, the completion time in `Nm Ns`; 143 lines against 153 |
| `resumed-c` | R1 has 4 kinds R0 has not: the `IntegrationWarning`'s four lines; 4 lines against 0 | R1 has 6 kinds R0 has not: the same 4, and a progress line and its `inflight` line; 116 lines against 110 | R1 has 2 kinds R0 has not: the progress line and its `inflight` line; 109 lines against 107 |

**Every difference, explained.**
1. **The pool's progress lines.** R1's `reference` printed one in its QSI stage (8 m 14.9 s), where
   R0's printed three in a 20 m 30.1 s stage; R1's `resumed-a` printed none, its GkWKB stage taking
   4 m 0.4 s, under 300 s; R1's `resumed-c` printed one in its QSI stage (5 m 43.8 s), where R0's
   (4 m 24 s) printed none. A stage that runs past the queue's 300 s minimum notify interval prints
   one, as the orchestrator's note expected (correction 4). The later lines of a stage print two
   rates per kind (§1.5.6); R1 printed no stage's second line, so has none of that kind.
2. **The `Queue summary` rate unit**, `/sec` against `/min`, in `reference` and `resumed-a`. The
   pool chooses the unit from the rate. The line's form is the same (§1.2.5, item 2).
3. **The SciPy `IntegrationWarning`.** It belongs to one integral, which falls in whichever resumed
   launch computes it. R1's `resumed-b` stored 424 QSI rows before its signal, against R0's 486, so
   here the integral fell after the point, in `resumed-c`, as at `dk-and-rpk-migration`'s R0 and
   R2. It is in `reference`, as at every rehearsal. The point moves it; it is not a finding.
4. **`resumed-b`'s `GkSource` completion time** is `in time 32.5s`, where R0's was `1m 10s`.
5. **`resumed-b`'s integrity warning** is in both, with equal line kinds: the prune at open of the
   rows `resumed-a` left unvalidated (§2.3.2). Shard 3's block lists its 15 rows in full
   (`stdout.log:13-46`); Ray folds shard 1's 19 and shard 0's 14 into one block with
   `[repeated 33x across cluster]` (`:132-137`). 15 + 33 = 48.

**Nothing else differs.** No warning or traceback appears that no earlier rehearsal printed. Every
pool line, banner, completion and `Queue summary` alike, has R0's form (§2.5.6).

#### 2.2.6 A3 v2's guard

After `resumed-c` and the readers, in the foreground, with no `--register`, the lock still held
([`guard_04.txt`](runregistry-extraction/rehearsal/guard_04.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/runregistry-extraction-04-resumed.sqlite \
    --job-name runregistry-extraction-04 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (16:33:33–16:33:34), with R0's `RuntimeError`, naming the five files of
`runregistry-extraction-04-resumed.sqlite` that exist and refusing to build into an existing store.
- **Nothing changed.** Afterwards, `shasum -a 256 -c` gave `OK` on every file of both stores and
  both sidecars, against the hashes taken just before it, after the readers, and against those
  taken after `resumed-c`, before the readers.
- **Nothing was created.** No run directory appeared (71 entries under `var/runs/` before and
  after), and no Ray process of this repository and no journal appeared.

### 2.3 L2 — the resume is the reference

#### 2.3.1 Nothing stored is recomputed

Through `dk-and-rpk-migration`'s
[`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py), unchanged
([`stage_counts_04.txt`](runregistry-extraction/rehearsal/stage_counts_04.txt)), with §1.3.1's
command and this prompt's run directories. Lookup / compute / store:

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **132/132/132** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **440/440/440** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored **207** `GkSource` rows before the signal (67 + 71 + 0 + 69 by shard), of
    which **48** were unvalidated (14 + 19 + 0 + 15) (§2.3.2).
  - `resumed-b` pruned those 48 at open, and computed **132 = 291 − 207 + 48**: the 84 never
    stored, and the 48 pruned. No validated row was computed again.
  - Summed, less the pruned: 207 + 132 − 48 = 291, the reference's.
- **`QuadSourceIntegral`.**
  - `resumed-b` printed no progress line in the stage. Its fingerprint, taken by `finish` on the
    closed store, records **424** rows.
  - `resumed-c` computed **440** = 864 − 424, exactly what was missing.
  - Summed: 424 + 440 = 864, the reference's. No stored row was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b`, because `resumed-a` never reached it, and
  0/0/0 in `resumed-c`.

#### 2.3.2 The prune at startup

The closed store after each launch, read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`, with §1.3.2's command. Each file's SHA-256
was unchanged by the read
([`closed_store_state_04.txt`](runregistry-extraction/rehearsal/closed_store_state_04.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **`GkSource` 14 + 19 + 0 + 15 = 48**, of 67 + 71 + 0 + 69 = 207 stored, each listed by serial. `GkNumericIntegration` 456 + 406 + 0 + 506 = 1368 and `GkWKBIntegration` 3 × 1159 = 3477 are all stored and validated |
| `resumed-b` | none | 0 rows | none (`GkSource` 97 + 97 + 0 + 97 = 291) |
| `resumed-c` | none | 0 rows | none |

**`resumed-a` stored and left unvalidated rows, as the point intended** (§2.1.2). The signal came
41–46 s into a stage the reference took 102.9 s over. 207 of 291 rows were stored by then, and 48
of them were not yet validated. R0's `resumed-a` left 164 rows (63 unvalidated).

**By serial**, through `dk-and-rpk-migration`'s
[`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py), unchanged, with §1.3.2's
commands and this prompt's state files
([`prune_check_04.txt`](runregistry-extraction/rehearsal/prune_check_04.txt)):
- **The first ran twice:** at 16:17:20, after `resumed-b` had opened and closed the store, and again
  after `resumed-c`. Both times it gave:
  - `48 serial(s) checked: 48 absent, 0 reused by a validated row, 0 still unvalidated`, exit 0;
  - by shard, 14 serials in shard 0, 19 in shard 1 and 15 in shard 3, each `absent`.
- **The second**, after `resumed-c`, gave `none: there is nothing to check`, exit 0.
- Every file of the store had the same SHA-256 before and after each check.

**So the prune at open removed exactly the rows `resumed-a` left unvalidated**, and no validated row,
now through `datastorekit` 0.4.0's shard actors. Each pruned serial is gone, and none was reused.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§2.3.5). Each finish's fingerprint has `problems` `{}`, for all four launches.

**The pool's prune of a replicated class.** None. After every launch each shard's `BackgroundModel`
row is validated, and the census finds the replicated tables equal (§2.3.5).

#### 2.3.3 The version row

From `closed_store_state.py`, after every launch: exactly one row, `(1, '2025.1.1')`, on each of the
four shards of both stores. The primary has no `version` table.

#### 2.3.4 The comparison

**The two stores**, by `compare_stores.py`, with §1.3.4's command and this prompt's stores
([`fingerprint_comparison_04.txt`](runregistry-extraction/rehearsal/fingerprint_comparison_04.txt)).
It exited 0: both stores format 6, digest `39809dca…`, problems none, each with 0 differences
against its sidecar; `compare_fingerprints(reference, resumed): 0 difference(s)`; `verdict: equal`.
All 21 classes are `equal`. With the campaign and prompt normalised, the output equals R0's
`fingerprint_comparison_02.txt` line for line (the command line and the exit line aside).

**Against R0's reference and `resumed-c`**, by `compare_with_recorded.py`, unchanged
([`fingerprint_against_recorded_04.txt`](runregistry-extraction/rehearsal/fingerprint_against_recorded_04.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/runregistry-extraction-04-reference.sqlite \
    var/datastores/runregistry-extraction-04-resumed.sqlite \
    --against var/runs/runregistry-extraction-02-reference-20261012T091921 \
    var/runs/runregistry-extraction-02-resumed-c-20261012T101145
```

It exited 0, verdict `equal`. For each store there were 0 differences against:
- its sidecar;
- R0's reference (taken at `fe16d8e7…`, `git_dirty` false);
- R0's `resumed-c` (taken at `fe16d8e7…`, `git_dirty` false).

All are format 6, digest `39809dca…`, 21 classes, problems none. Both sides are SGK's fingerprints,
so SGK's comparison applies. The files of both stores had the same SHA-256 before and after these
reads (`after-resumed-c.sha256`, all `OK`).

| Class | Reference | Resumed | Digests | R0's reference |
|---|---|---|---|---|
| `version` | 1 | 1 | equal | 1 |
| `store_tag` | 10 | 10 | equal | 10 |
| `redshift` | 1159 | 1159 | equal | 1159 |
| `wavenumber` | 3 | 3 | equal | 3 |
| `tolerance` | 7 | 7 | equal | 7 |
| `LambdaCDM` | 1 | 1 | equal | 1 |
| `QCD_Cosmology` | 1 | 1 | equal | 1 |
| `IntegrationSolver` | 7 | 7 | equal | 7 |
| `GkSourcePolicy` | 2 | 2 | equal | 2 |
| `QuadSourcePolicy` | 2 | 2 | equal | 2 |
| `wavenumber_exit_time` | 3 | 3 | equal | 3 |
| `BackgroundModel` | 1 | 1 | equal | 1 |
| `TkNumericIntegration` | 3 | 3 | equal | 3 |
| `TkWKBIntegration` | 3 | 3 | equal | 3 |
| `GkNumericIntegration` | 1368 | 1368 | equal | 1368 |
| `GkWKBIntegration` | 3477 | 3477 | equal | 3477 |
| `GkSource` | 291 | 291 | equal | 291 |
| `GkSourcePolicyData` | 291 | 291 | equal | 291 |
| `QuadSource` | 6 | 6 | equal | 6 |
| `QuadSourceIntegral` | 864 | 864 | equal | 864 |
| `OneLoopIntegral` | 0 | 0 | equal | 0 |

#### 2.3.5 Integrity, both stores, read-only

**The census**, with §1.3.5's command and this prompt's stores
([`census_04.txt`](runregistry-extraction/rehearsal/census_04.txt)): every table is `OK` on both
stores (28 lines), and the verdict is `no divergence` on both. Exit 0. With the campaign and prompt
normalised, the output equals R0's `census_02.txt` line for line.

**`integrity_check`, `foreign_key_check` and the inventory**, through `dk-and-rpk-migration`'s
[`store_soundness.py`](dk-and-rpk-migration/rehearsal/store_soundness.py), unchanged
([`store_soundness_04.txt`](runregistry-extraction/rehearsal/store_soundness_04.txt)):
- It exited 0, `=> clean`.
- On the primary and every shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is
  `['ok']` and `PRAGMA foreign_key_check` returns 0 rows.
- The package's `read_inventory(primary, factories)`, now `datastorekit` 0.4.0's, reads shards
  `(0, 1, 2, 3)`, and reports no problem in any class of either store, and 0 `duplicate`.
- With the campaign and prompt normalised, the output equals R0's `store_soundness_02.txt` line for
  line.
- Every file of both stores had the same SHA-256 before and after these reads, and no journal
  appeared.

### 2.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance
(`--ray-address local`), through `dk-and-rpk-migration`'s
[`run_readers.sh`](dk-and-rpk-migration/rehearsal/run_readers.sh), unchanged, with §1.4's two
commands and this prompt's store and output directory, the lock held
([`readers_04.txt`](runregistry-extraction/rehearsal/readers_04.txt)). Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` / traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 52 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 315 | none | identical |
| `extract_GkWKB_data.py` | 0 | 125 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 41 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 176 | none | identical |
| `extract_TkWKB_data.py` | 0 | 26 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | 0 | 25 s | `EXTRACT TENSOR SOURCE DATA`: 6 | 27 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 33 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 | 73 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 54 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 369 | none | identical |

- **The lookups equal R0's, queue by queue and reader by reader**
  ([`log_comparison_04.txt`](runregistry-extraction/rehearsal/log_comparison_04.txt), section 4).
- **The files written differ** in the readers that draw random subsamples, as between R0, R2 and
  R1: against R0's 396, 210, 76 and 459, R1 wrote 315, 176, 73 and 369. Each draws its subsample with
  `random.sample`; it is not a finding.
- **The reader logs' line kinds against R0's**, normalised as §2.2.5: six are equal.
  `extract_GkWKB_data`'s queue printed one progress line, where R0's printed two; R0's second, in
  the two-rate form, is the one kind R1's log lacks. The first has R0's form:

  ```
     -- 2026-10-14 16:28:19 (1m 2.21s running): 168/348 work items remaining = 51.72% complete
        inflight: 0 lookup | completed: 180 lookup (avg 173.6/min)
  ```
- **The load.** The readers ran 16:26:11–16:32:19, at a one-minute load of 36–161 from processes
  not of this repository.
- **The model and the run.** Every run printed `** reading run "default", the only run in this
  datastore (no --run-label given)`. With `--models LambdaCDM` each printed `>> RUNNING PIPELINE FOR
  MODEL LambdaCDM`; without it, `extract_Gk_data.py` read LambdaCDM, then printed the
  `QCD_Cosmology` skip line, and exited 0.
- **No file of the store changed.** Before and after each reader, the SHA-256, mtime and size of the
  primary and of every shard were identical, and no `-journal`, `-wal` or `-shm` file appeared. No
  Ray process of this checkout was left after any reader.
- **The fingerprint afterwards.** `store fingerprint` (without `--write`) on the resumed store gave
  `39809dca…`, problems none, "matches the fingerprint taken 2026-10-14T16:24:51+01:00 by run
  runregistry-extraction-04-resumed-c-20261014T161739, at 89ff8b728b61ba22452110de88718094361e4e15".

### 2.5 L4 — equal to R0

#### 2.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

`policy_statistics.py`, with §1.5.1's command, on each of R1's runs. Exit 0 on `reference`,
`resumed-b` and `resumed-c`; exit 1 on `resumed-a`, which has no block, as at R0. The reference's
block, lines 83–96 of its `stdout.log`, verbatim
([`policy_statistics_04_reference.txt`](runregistry-extraction/rehearsal/policy_statistics_04_reference.txt)):

```
** GkSourcePolicyData SUMMARY STATISTICS

     * Policy: policy="maximize-WKB"-Levin-threshold="1.5"
        Total instances: 291

        >> numeric: 109 instances = 37.46%
           complete: 109 instances = 100.00% of this type, 37.46% of total

        >> mixed: 26 instances = 8.93%
           minimal: 1 instances = 3.85% of this type, 0.34% of total
           complete: 25 instances = 96.15% of this type, 8.59% of total

        >> WKB: 156 instances = 53.61%
           complete: 156 instances = 100.00% of this type, 53.61% of total
```

| | R0 | R1 |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| `reference`'s order under `mixed` | `complete`, `minimal` | `minimal`, `complete` |
| `reference`, against R0's | — | differs in the order of those two lines; equal sorted |
| `resumed-b` (`stdout.log:115`) | `complete` first | equal to R0's `resumed-b` line for line |
| `resumed-c` (`stdout.log:83`) | `minimal` first | equal to R0's `resumed-c` line for line |

**Every count is R0's**, in every run that prints the block. The order is Ray's completion order:
`[02-the-policy-statistics-block-orders-its-lines-by-completion]`, with one more instance and no
new fact. R1's reference block equals `dk-and-rpk-migration` R2's line for line.

#### 2.5.2 The reference run's stage counts

From [`stage_counts_04.txt`](runregistry-extraction/rehearsal/stage_counts_04.txt), against R0's
`stage_counts_02.txt`: the `reference` column is equal in every stage, and so is the `resumed-a`
column, since both were killed in the same stage. (The table is §2.3.1's.)

#### 2.5.3 The resumed chain, by identity

| Identity | R0 | R1 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated (`prune_check.py`) | 63 = 63, all absent, none reused | **48 = 48, all absent, none reused** | ✅ |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | 190 = 291 − 164 + 63 | **132 = 291 − 207 + 48** | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 378 = 864 − 486 | 440 = 864 − 424 | ✅ |
| Totals per class over the chain equal the reference's | `GkSource` 164 + 190 − 63 = 291; QSI 486 + 378 = 864 | `GkSource` 207 + 132 − 48 = 291; QSI 424 + 440 = 864 | ✅ |

What each resumed launch pruned and recomputed:

| Launch | Pruned at open | Computed, stage by stage | Of which recomputing a pruned row |
|---|---|---|---|
| `resumed-a` | nothing (new store) | everything up to the `GkSource` stage, as `reference`; **207 `GkSource` stored** at the signal, 48 unvalidated | — |
| `resumed-b` | **48 `GkSource`** (the unvalidated ones), in shards 0, 1 and 3 | `GkSource` 132; `GkSourcePolicyData` 291; `QuadSourceIntegral` 424 stored at the signal; every earlier stage 0 | 48 `GkSource` |
| `resumed-c` | nothing | `QuadSourceIntegral` 440; every other stage 0 | — |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 2.5.4 The stage sequence

The lines of each `stdout.log` that begin `** `
([`log_comparison_04.txt`](runregistry-extraction/rehearsal/log_comparison_04.txt), section 3):
- R1's and R0's hold 26, 23, 26 and 26 such lines, slug by slug.
- With the campaign and prompt, the repository path and every number normalised, each slug's
  sequence equals R0's, in order.
- In the reference, 15 of the 26 are the pipeline's stage banners, `** BUILD ARRAY OF WAVENUMBERS
  TO SAMPLE` through `** CALCULATE QUADRATIC SOURCE INTEGRALS`. They equal R0's verbatim, in order,
  with nothing normalised. The other 11 are the launcher's own lines.

#### 2.5.5 The stage times

Not compared (`ALL WORK ITEMS COMPLETE in time …` in each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1 m 1.85 s | 3 m 19.5 s | 4.27 s | 13 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 1 m 43.8 s | 4 m 0.42 s, no progress line | 8.6 s | 23.1 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **1 m 42.9 s**, no progress line | killed | 32.5 s | 14.4 s |
| APPLY GKSOURCE POLICIES | 54.7 s | — | 13.9 s | 13.2 s |
| QUADRATIC SOURCE INTEGRALS | **8 m 14.9 s**, one progress line | — | killed, no progress line | 5 m 43.8 s, one progress line |

- Each earlier stage took under 5 s.
- **The reference took 13 m 57 s**, against R0's 29 m 19 s and `dk-and-rpk-migration` R2's
  11 m 25 s.
- **The `GkSource` stage took 102.9 s and the QSI stage 494.9 s**, against R0's 78 s and 1230.1 s.
  The `GkSource` stage was slower than R0's and the QSI stage faster; the load moved between them.
  The stores are equal, so the time is the machine's.

#### 2.5.6 The printed lines and the registry's fields

Against R0's same slug ([`log_comparison_04.txt`](runregistry-extraction/rehearsal/log_comparison_04.txt), section 3):

| Line, in `stdout.log` | `reference` R1 / R0 | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| `** ` (banners and the launcher's lines) | 26 / 26 | 23 / 23 | 26 / 26 | 26 / 26 |
| `-- ALL WORK ITEMS COMPLETE in time …` | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `Queue summary: …`, one after each completion line | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `(no compute items were evaluated - all assets read from datastore)` | 1 / 1 | 1 / 1 | 7 / 7 | 9 / 9 |
| progress: `n/m work items remaining = p% complete` | **1** / 3 | **0** / 1 | 0 / 0 | **1** / 0 |
| progress: `inflight: … \| completed: …` | **1** / 3 | **0** / 1 | 0 / 0 | **1** / 0 |

- **Every completion line is followed by its `Queue summary` line**, in all four.
- **The forms are R0's.** Normalised as §2.2.5, the completion lines of R1's four runs make 2
  kinds, the `Queue summary` lines 7 and the `(no compute items …)` line 1. Each set equals R0's.
- **The progress lines** are R0's first-line form, field for field:

  ```
     -- 2026-10-14 15:57:46 (5m 21.8s running): 0/2 work items remaining = 100.00% complete (may be waiting for compute/store/validate tasks to finish)
        inflight: 0 lookup, 226 compute, 40 store | completed: 864 lookup (avg 161.1/min), 638 compute (avg 119/min), 598 store (avg 111.5/min)
  ```

  in the reference's QSI stage (`stdout.log:100-101`), and in `resumed-c`'s QSI stage at
  5 m 43.4 s (`stdout.log:100-101`). The orchestrator's note expected that whether they appear is
  the 300 s threshold and the load (correction 4).
- **The registry's fields at each run's end**, R1 against R0: `state`, `stage`, `units_done` and
  `units_total` are R0's in all four, and each `status.json` has R0's keys exactly.

| Slug | `state` | `stage` | `units_done` / `units_total` | R0's `stage` |
|---|---|---|---|---|
| `reference` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` | the same |
| `resumed-a` | `killed` | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | 0 / `null` | the same |
| `resumed-b` | `killed` | `CALCULATE QUADRATIC SOURCE INTEGRALS` | 0 / `null` | the same |
| `resumed-c` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` | the same |

### 2.6 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen (§2.3.2).
- **An in-flight record left by a killed launch.** Did not happen: `replication_in_flight` had 0
  rows after every launch.
- **A repair by the check at open.** Did not happen (§2.2.5).
- **A Ray agent outliving a killed driver.** Did not happen (§2.2.3).
- **A line from the three releases**: a `datastorekit` frame, a `DeprecationWarning`, or
  `_timestamp`. Did not happen (§2.2.5).
- **A printed line the registry or the interrupter could not read.** Did not happen (§2.5.6).
- **A missed interruption.** Did not happen. Both killed launches ended `killed`, in the stage the
  interrupter waited for.
- **A held lock, or a refused `release`.** Did not happen (§2.1.3).

One thing about the record's timing was seen by the orchestrator's wait, not by the build. A killed
run's `status.json` says `killed` from the moment the driver's handler records it (16:11:09 for
`resumed-a`); `finish_after_exec` adds the fingerprint once the pipeline has unwound (16:11:14) and
the final heartbeat follows (16:11:15). That wait, which ended on the state, ended 5–6 s before
the record was complete; the waits on `resumed-b` and `resumed-c` ended on the pid's death, as
`CLAUDE.md`'s liveness rule reads it. This is the registry's and the launcher's ordering
(`docs/gktk-remedial/scoped_pipeline_run.py`, `RunRegistry/__init__.py` `Run.finish`'s docstring),
documented and frozen; no issue is opened.

### 2.7 Issues opened

None. Every difference from R0 is explained by:
- where a launch was killed;
- Ray's completion order (an issue already open);
- the readers' random subsamples;
- or the load, with the pool's 300 s notify threshold.

### 2.8 The suites

After the last launch, the readers, the guard and the release: README §8's command, each root in
the foreground, serially, its output in the agent's scratchpad and its verdict taken by `grep`.

| Root | Ran | Result | Load (1 min) at start |
|---|---|---|---|
| AdaptiveLevin | 32 | OK | 29.5 |
| CosmologyModels | 39 | OK | 29.5 |
| ObjectFactories | 271 | OK | 29.5 |
| tools | 91 | OK | 25.5 |
| LiouvilleGreen | 148 | OK (1 skipped) | 19.0 |
| RunRegistry | 286 | OK | 14.2 |
| ComputeTargets | 693 | 2 failures: `test_background_tau.TestBackgroundTau.test_build_cost` (0.653 s against 0.5 s) and `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (0.0923 s against 0.06 s) | 7.8 |

**Both failures are README §8's known wall-clock tests.** Run alone three times each at 16:41, at a
load of 11–13: `test_build_cost` passed three times; `test_wall_time_per_object` failed three times
(0.0805 s, 0.105 s, 0.113 s), as it failed alone twice at 03's run. It is a wall-clock assertion with
about 1× margin; no code it runs has changed since `a45aefc`. Every count is 03's.

`docs/runregistry-extraction/fingerprint_both_ways.py`, the one Python file added, is clean under
`black --check`.

### 2.9 What stays under `var/`

- `var/runs/runregistry-extraction-04-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`. Each
  holds the run's manifest, `status.json` with its full fingerprint, `stdout.log` and `stderr.log`.
  For `resumed-a` and `resumed-b` it also holds `interrupter.log`.
- `var/runs/runregistry-extraction-04-<slug>.launch.log`, beside each run directory.
- `var/datastores/runregistry-extraction-04-{reference,resumed}.sqlite`, four shards each, and each
  store's `.manifest.json` sidecar. Shard 2 of each, 1 077 248 bytes, holds only replicated rows.
- `var/runregistry-extraction-04/`, which holds:
  - the agent's working notes (`NOTES.md`), the pre-launch captures, the lock record, the dummy-test
    outputs and the listings of `var/datastores/`;
  - the closed-store states, the prune checks and the hashes (`state/`);
  - the check outputs (`checks/`), among them every run's statistics block;
  - the readers' outputs and logs (`extract/`, `readers.txt`);
  - the five breakages' diffs and outputs (`breakage/`), and the suites' outputs (`suites/`).

**Nothing else under `var/` moved**
([`var_datastores_04.txt`](runregistry-extraction/rehearsal/var_datastores_04.txt)).
- Before the first launch, `var/datastores/` held 26 files: the 14 tombstones and R0's twelve.
  `find var/datastores -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256` was
  `3e0e6badf16a57c786772255f2b5276274db5c913a4fcadcd27539e9464b01c9`; over the 14 tombstones alone,
  `023d139fb714e229f96c6fb61894daf526b768fd603307d153d4d91810cd2626`.
- After the last step it held 38. The 26 have the same SHA-256, mtime (to the nanosecond) and size
  as before, and the same two digests. The 12 new files are this rehearsal's two stores and their
  sidecars. R0's stores were read by the comparisons and by W1, and none of their files moved.
- No `-journal`, `-wal` or `-shm` file is under `var/datastores/`.
- `RunRegistry list` shows 37 runs: R0's 33 and this rehearsal's four, all finished but the two
  `unknown` of the `a3-pilot` pair.

### 2.10 W1 — the fingerprint both ways

**The script.** [`runregistry-extraction/fingerprint_both_ways.py`](runregistry-extraction/fingerprint_both_ways.py),
written after `resumed-c` had ended and the readers had run. For each primary, read-only and with no
Ray, it reads `inventory = read_inventory(primary, factories)` (`datastorekit.store_inventory`,
`config.datastore`), takes `sgk = RunRegistry.stores.fingerprint_of(inventory)` and
`pkg = datastorekit.store_fingerprint.fingerprint_of(inventory, RunRegistry.stores.FINGERPRINT_FORMAT)`,
and prints one line per check:
1. `pkg["encoding_format"]` is `ENCODING_FORMAT`, and is `1`; `"encoding_format"` is not in `sgk`;
2. with `encoding_format` removed from `pkg`, `pkg == sgk` as dictionaries, every differing path
   printed with both values (the class, the tag set by its tags, the field);
3. **the order**, which `dict` equality does not compare: `list(pkg_without_encoding) == list(sgk)`
   and `list(pkg["classes"]) == list(sgk["classes"])`, the first differing position printed if not.
   `tag_sets` are lists, so (2) already compares their order (the orchestrator's correction 5);
4. `fingerprint_format` 6 and `taken` `None` in both;
5. the overall digest, printed, `39809dca…` in both;
6. the listings: equal in length, equal line for line but the first, and the package's first line
   SGK's with ` encoding_format 1` appended (the addendum's item 4);
7. the package's `compare_fingerprints(pkg, pkg)` and SGK's `compare_fingerprints(sgk, sgk)` empty,
   and the package's `compare_fingerprints(sgk, pkg)` exactly one entry, of kind `encoding`;
8. SGK's `compare_fingerprints` of the sidecar's recorded fingerprint with `sgk` empty.

It exits 0 if and only if every check holds on every store, and writes nothing.

**The run** ([`fingerprint_both_ways_04.txt`](runregistry-extraction/rehearsal/fingerprint_both_ways_04.txt)),
at 16:33:54, after the lock was released:

```bash
PYTHONPATH=. ./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py \
    var/datastores/runregistry-extraction-04-reference.sqlite \
    var/datastores/runregistry-extraction-04-resumed.sqlite \
    var/datastores/runregistry-extraction-02-reference.sqlite \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

It exited 0 in 11 s: `summary: 4 store(s); every check holds on every store`. On each store all 20
checks hold:

| Store | SGK's and the package's digest | Classes, in order | Listings | `compare_fingerprints(sgk, pkg)` (package's) | Against the sidecar |
|---|---|---|---|---|---|
| R1 reference | `39809dca…`, equal | 21, equal | 7542 lines each, equal but the first | one `encoding` entry: recorded `None`, current 1 | 0 differences (taken 16:02:03 by `store fingerprint --write`) |
| R1 resumed | `39809dca…`, equal | 21, equal | 7542, equal but the first | the same | 0 (taken 16:24:51 by `resumed-c`) |
| R0 reference | `39809dca…`, equal | 21, equal | 7542, equal but the first | the same | 0 (taken 2026-10-12 09:50:07 by `store fingerprint --write`) |
| R0 resumed | `39809dca…`, equal | 21, equal | 7542, equal but the first | the same | 0 (taken 2026-10-12 10:17:15 by R0's `resumed-c`) |

- **On every store the package's fingerprint has SGK's five keys plus `encoding_format`**, second.
  With it removed, the dictionaries are equal, key for key and in order: `fingerprint_format` 6, the
  overall digest, `problems` `{}`, `taken` `None`, and for each of the 21 classes, in the same
  order, its `count`, `digest` and every tag set's `tags`, `count` and `digest`.
- **So DatastoreKit's `fingerprint_of` is SGK's on real stores**, the check its own tests cannot
  make. 06 can make SGK depend on it.
- **The one `encoding` entry** is `[00-recorded-fingerprints-lack-the-encoding-format]`, measured:
  once SGK compares through the package, every fingerprint SGK now holds compares as that one entry,
  whatever the stores hold. The issue is assigned to the retirement of these four stores after this
  prompt's review (D6).
- **The stores were untouched.** Every file of the four stores (24: each primary, its four shards and
  its sidecar) had the same SHA-256, `mtime_ns` and size before and after the run, and no journal
  appeared.

**The deliberate breakages** (README §5 rule 8). Each was applied to a copy of the script in the
agent's scratchpad, never committed, and run on R1's reference store. Each exits 1 and names what
failed. The diffs, exactly as applied, are in the log
([`logs/04-rehearse-through-the-new-datastorekit.md`](../prompts/runregistry-extraction/logs/04-rehearse-through-the-new-datastorekit.md)).

| # | Breakage | Exit | The line that names the failure |
|---|---|---|---|
| 1 | `encoding_format` not removed from `pkg` before the comparison | 1 | `encoding_format: package 1, SGK '<absent>'`, and the key order differs at position 1 |
| 2 | `GkSource`'s `digest` in `pkg` altered after it is taken | 1 | `classes.GkSource.digest: package '0000…', SGK '6a001ba4…'` |
| 3 | format `5` passed to the package's `fingerprint_of` | 1 | `fingerprint_format: package 5, SGK 6` and `digest: package 'b40736b5…', SGK '39809dca…'` |
| 4 | one tag set's `count` in `sgk` incremented | 1 | `classes.GkSource.tag_sets[0] tags [GkOneLoopDensity, …, SourceRedshiftGrid_1159_7139cb9c].count` with both values |
| 5 | the listing check compares the first lines unmodified | 1 | `[FAIL] package's first line is SGK's + ' encoding_format 1'` |

The reference store's files were unchanged by the five runs.

---

## 3. R2 — through both packages

**§3's tree, stores and runs.** Every R2 launch ran on branch `handover-remedial` at **`b9daf44`**
("Write the orchestrator note for runregistry-extraction prompt 08").
- **Its code is `a9ed928`'s** (prompt 07). `git diff --stat a9ed928 b9daf44 -- . ':!prompts'` is
  empty: `319cad0` and `b9daf44` touch `prompts/` only.
- **What it changes over R1's tree (`89ff8b7`)** is the registry, and nothing else that a build runs:
  - `RunRegistry/` is gone (06, `740bafc`). `runregistrykit.toml` and `config/fingerprint.py` are
    new; `requirements.txt` gains the `runregistrykit[stores]` pin (05, `3dcab1b`).
  - The launcher, `docs/handover/quadsource_atol_sweep.py`, now registers through
    `runregistrykit.begin`. Its banner's `list it with:` line names `runregistrykit`, and its hash
    is `2a985303…` (06; 06's review, `7af17bd`).
  - `main.py`, every `extract_*.py`, `run_readers.sh`, every factory and `StoreFixtures/` are
    untouched since R1 (the orchestrator's note).
- **The packages.**
  - The registry is the installed **`runregistrykit` 0.1.0**, at `9f8ebe43…`, not editable. It
    registers the runs, takes the fingerprints at finish through `config/fingerprint.py`'s
    fingerprinter, and writes the sidecars.
  - The datastore layer is **`datastorekit` 0.4.0** at `c8570e84…`, and the work queue
    **`raypoolkit` 0.2.0** at `eaef0ce6…`, both not editable, both as at R1.
- **The tree was clean.** `git status --short` was empty at every launch.
- **The manifests.** Every manifest stamps `git_dirty: false` and `packages`
  `{"datastorekit": {"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false, "requested_revision": "v0.4.0", "url": "https://github.com/ds283/DatastoreKit", "version": "0.4.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}, "runregistrykit": {"commit": "9f8ebe431503fb8c79ee1407cc460501f1ca2d92", "editable": false, "requested_revision": "v0.1.0", "url": "https://github.com/ds283/RunRegistryKit", "version": "0.1.0"}}`,
  compared as JSON. The `datastorekit` and `raypoolkit` entries are R1's exactly.
- **The lock.** It was taken before `reference`, its `pid` set at each launch, held through the
  readers and the guard, and released (§3.1.3).

R2's launcher built two throwaway stores:
- **`var/datastores/runregistry-extraction-08-reference.sqlite`** (sidecar `store_id`
  `6bce4480eba24f4291bfe72ea55b4d23`), built straight through by the run **`reference`**;
- **`var/datastores/runregistry-extraction-08-resumed.sqlite`** (`store_id`
  `86b70a5d966347fbbbac46140ba8e57e`), built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

The runs are `var/runs/runregistry-extraction-08-<slug>-<stamp>/`. Both stores stay, with their
sidecars, and neither is retired: the user retires them after 09's review, or not (README §3). The
small outputs quoted below are in [`runregistry-extraction/rehearsal/`](runregistry-extraction/rehearsal/),
under names carrying `08`. The large ones stay under `var/` (§3.9).

R0's and R1's stores are retired (board §5.8). R2 is compared with them through the fingerprints
their `reference` and `resumed-c` run directories recorded, field by field (§3.10), and with R1's
run directories and saved outputs for everything else. The 18 files under `var/datastores/` were
hashed before R2's first launch and after its last step (§3.9).

**Headline.**
1. **R2 reproduces the reference fingerprint through `runregistrykit` 0.1.0 and `datastorekit`
   0.4.0.**
   - Both stores are format 6, **`encoding_format` 1**, digest
     `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes equal, and
     each equals its sidecar's.
   - `compare_fingerprints` is empty between the two stores, and between each store and the
     fingerprints R2's own `reference` and `resumed-c` recorded (§3.3.4).
   - The only change to what runs between R1 and R2 is the registry. So the swap moves nothing a
     build writes or a fingerprint reads.
2. **W1 holds across stores (§3.10).** `fingerprint_both_ways.py` exits 0, 20 of 20 checks on each
   store. Field by field, with `encoding_format` and `taken` set aside, each R2 store's fingerprint
   equals each of the four fingerprints SGK's own code recorded for R0 and R1. Against each record,
   the package's comparison is exactly one `encoding` entry. The stores were untouched by it.
3. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0; `resumed-a` and
   `resumed-b` are `killed`/143. Each has a fingerprint carrying `encoding_format` 1, and each driver
   exited as recorded (§3.2).
4. **The resume recomputed nothing it had stored, and the prune at open was exercised a third time.**
   - `resumed-a` stored 165 `GkSource` rows before its signal, 44 of them unvalidated.
   - `resumed-b`'s open pruned exactly those 44: every serial is absent afterwards and none was
     reused.
   - `resumed-b` then computed 170 = 291 − 165 + 44.
   - `resumed-c` computed the 567 = 864 − 297 `QuadSourceIntegral` rows not stored when `resumed-b`
     was killed, and nothing else.
   - Nothing is duplicated (§3.3).
5. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§3.4).
6. **L4 equals R1's** (§3.5):
   - the statistics counts are equal, in `reference` and in both resumed runs that print the block;
   - the reference's stage counts are equal, stage by stage;
   - the resumed chain's identities hold, the prune's with 44 rows;
   - the stage banners of every run are R1's, in order;
   - every completion line has R1's form and number, and the registry's `stage` fields are R1's.

   The reference's statistics block prints `complete` before `minimal` under `mixed`, R0's order and
   not R1's: `[02-the-policy-statistics-block-orders-its-lines-by-completion]` again.
7. **The one line naming `runregistrykit`** in the twelve build logs is the launcher's banner,
   `**   list it with: ./venv/bin/python -m runregistrykit list`, one in each `stdout.log` and
   launch log, where R1's named `RunRegistry` (06). No line names `datastorekit`, and none holds a
   `DeprecationWarning`, a `ConfigError` or a traceback (§3.2.5).
8. **One new kind of line, from the machine:** Ray's raylet warns every 10 s that its temp
   directory's volume "is over 95% full" (54 lines over the four `stderr.log`s). The volume is 96 %
   used, against 93 % at R1. Nothing a build wrote or read changed (§3.2.5).
9. **The lock held for the whole rehearsal, and is free again** (§3.1.3).
10. **Nothing else under `var/` moved.** The 18 tombstones have the same SHA-256, mtime and size
    before the first launch and after the last step; the only new files are R2's two stores and
    their sidecars (§3.9).
11. **RunRegistryKit's G2 holds**, each of its checklist item 9's six clauses with its evidence
    (§3.11).

### 3.1 How each was launched

Each launch was §2.1's command, with `runregistry-extraction-08`, `--prompt 08` and this prompt's
store names, run from the repository root, detached, the purpose passed to `sh -c` as its `$0`:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/runregistry-extraction-08-<store>.sqlite \
    --job-name runregistry-extraction-08 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "$0" --campaign runregistry-extraction --prompt 08; \
  echo "** driver exit status $?"' "<one line>" \
  > var/runs/runregistry-extraction-08-<slug>.launch.log 2>&1 &
disown
```

- `reference`: the reference store, no `--resume`.
- `resumed-a`: the resumed store, no `--resume`.
- `resumed-b` and `resumed-c`: the resumed store, with `--resume`.

Each `--purpose` names R2 of this campaign, the tree's commit and its code's, `runregistrykit` 0.1.0
at `9f8ebe4`, `datastorekit` 0.4.0 at `c8570e8` and `raypoolkit` 0.2.0, and the fingerprint it must
equal ([`run_records_08.txt`](runregistry-extraction/rehearsal/run_records_08.txt)).

**One check per launch, one wait per launch.** After each launch the agent made one check:
- the manifest existed, with its `packages`, `git_dirty` and keys checked;
- `status.json` said `running`, with a pid answering `kill -0`;
- `./venv/bin/python -m runregistrykit list` showed the run;
- in the same check, `tools.rehearsal_lock set-pid` with that pid (§3.1.3), and, for `resumed-a`
  and `resumed-b`, the interrupter's launch.

The agent then ended its turn. The orchestrator ran one wait per launch, ending on the driver pid's
death, and resumed the agent with the final `status.json`. Neither polled. Every wait ended with the
pid dead and the heartbeat never stale (`ENDED done`, `ENDED killed`, `ENDED killed`, `ENDED done`).

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `runregistry-extraction-08-reference-20261015T235716` | reference | no | 76434 (76430) | `done` | 0 | 0 | 4 m 33 s (23:57:16–00:01:49) | 5.95 7.17 8.92 / 18.79 23.61 16.60 |
| `resumed-a` | `runregistry-extraction-08-resumed-a-20261016T000342` | resumed | no | 83931 (83928) | `killed` | 143 | 143 | 2 m 2 s (00:03:42–00:05:44) | 11.97 21.38 16.07 / 5.94 15.15 14.49 |
| `resumed-b` | `runregistry-extraction-08-resumed-b-20261016T000708` | resumed | yes | 87907 (87904) | `killed` | 143 | 143 | 1 m 27 s (00:07:08–00:08:35) | 7.17 14.57 14.31 / 10.71 14.62 14.45 |
| `resumed-c` | `runregistry-extraction-08-resumed-c-20261016T000959` | resumed | yes | 91423 (91420) | `done` | 0 | 0 | 1 m 40 s (00:09:59–00:11:39) | 8.54 13.71 14.13 / 5.59 11.12 13.04 |

- **Wall time** is from the manifest's `created` to the final heartbeat.
- **Load** is `uptime`'s on 10 CPUs: at the pre-launch check, and at the agent's first check after
  the run. The one-minute load at the checks ran from 6 to 19, and reached 56 at the end of the
  readers. R1's checks ran at 11–100, R0's at 58–158. The machine was lightly loaded, and every stage
  ran faster than at R0 or R1 (§3.5.5); that changed nothing a build wrote.

**The sidecars** ([`store_sidecars_08.txt`](runregistry-extraction/rehearsal/store_sidecars_08.txt)):
- **The reference store.** `store create` ran on its closed primary at 00:03:15, after `reference`.
  `store fingerprint --write` followed: `39809dca…`, problems none, "it replaced none". The
  fingerprint in the sidecar carries `encoding_format` 1.
- **The resumed store.** `store create` ran on its closed primary at 00:06:52, after `resumed-a`.
  The store's five files were unchanged by it.

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`. The commands are §1.1's,
through `runregistrykit` (06's log, §7.8):

```bash
./venv/bin/python -m runregistrykit store create --purpose "<one line>" \
    var/datastores/runregistry-extraction-08-reference.sqlite
./venv/bin/python -m runregistrykit store fingerprint --write \
    var/datastores/runregistry-extraction-08-reference.sqlite
./venv/bin/python -m runregistrykit store create --purpose "<one line>" \
    var/datastores/runregistry-extraction-08-resumed.sqlite
```

#### 3.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_08.txt`](runregistry-extraction/rehearsal/prelaunch_checks_08.txt)):
- **The tree.**
  - `git rev-parse HEAD` was `b9daf449722f3e443f8dd592c1a890752cbeed6c`.
  - `git status --short` was empty, and was checked again in the same shell command that launched.
  - `git worktree list` showed this checkout alone.
  - `git log --oneline a9ed928..HEAD` is `b9daf44` and `319cad0`, and
    `git diff --stat a9ed928 HEAD -- . ':!prompts'` was empty.
  - Every working note and capture went under `var/runregistry-extraction-08/`, which is ignored,
    or the agent's scratchpad, until the last launch had ended. This record's files were written
    afterwards.
- `RAY_ADDRESS` was unset. The machine has 10 CPUs.
- **The registry.** `./venv/bin/python -m runregistrykit list` showed nothing `running` or stale:
  39 runs before `reference` (37 finished, 2 `unknown`, the pre-registry `a3-pilot` pair); then 40,
  41 and 42, all finished but those two.
- **No Ray process of this repository.** Before each launch `tools.rehearsal_lock ray-processes`
  exited 0, listing nothing. The machine-wide `pgrep -lf 'gcs_server|raylet|ray::'` was also
  recorded, and was empty (rc 1) before every launch. Before each later launch the previous run's
  pids were checked by `ps -p`: the `sh` and Python pids, the interrupter's, and every Ray pid its
  logs name. None was alive.
- **The lock** was free before `reference` (`status`: `lock: free`, exit 0), and `~/ScienceData/`
  held `ChamPBH-rehearsals/` alone.
- **The store names.**
  - Before `reference`, `./venv/bin/python -m runregistrykit store show` on both names gave kind
    `absent`, problems none, and no run naming either, exit 0.
  - Nothing under `var/datastores/` or `var/runs/` contained `runregistry-extraction-08`, and there
    was no `-journal`, `-wal` or `-shm` file under `var/datastores/`.
  - Before `resumed-a`, no `runregistry-extraction-08-resumed*` file existed, and `store show` on
    the name again gave `absent`.
- **The packages.**
  - `./venv/bin/pip show runregistrykit datastorekit raypoolkit` gave 0.1.0, 0.4.0 and 0.2.0, each
    with `Location` this venv's `site-packages` and no editable location. `pip check` was clean.
  - `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))` gave the
    record above. It equals, as JSON, the orchestrator's record and 06's worktree-check manifest's
    `packages`; its `datastorekit` and `raypoolkit` equal R1's reference manifest's.
  - `requirements.txt:17` is `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.4.0`;
    `:76` is `raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0`; `:81` is
    `runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0`.
- **The configuration.** `runregistrykit.find_client_root()` from the root read
  `runregistrykit.toml`, with `runs_root` `var/runs`, `stores_root` `var/datastores`,
  `distributions` `('datastorekit', 'raypoolkit')` and `stores`
  `{'fingerprinter': 'config.fingerprint:fingerprinter'}`.
- **What runs.** In a fresh `python -I` with the repository on `sys.path`:
  - `runregistrykit.__file__`, `datastorekit.__file__` and `raypoolkit.__file__` are under the
    venv's `site-packages`;
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`;
  - `ObjectFactories.__file__` is the repository's `ObjectFactories/__init__.py`;
  - `datastorekit.store_fingerprint.ENCODING_FORMAT` is `1`, and
    `config.fingerprint.FINGERPRINT_FORMAT` is `6`;
  - `import RunRegistry` raises `ModuleNotFoundError`;
  - `ray.is_initialized()` is false.
- **The recorded fingerprints R2 is compared with.** The `status.json` of R0's and R1's
  `reference` and `resumed-c` are each `done`/0, format 6, `39809dca…`, 21 classes, problems `{}`,
  with no `encoding_format`; all four are equal once `taken` is set aside.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` was
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`. Its last change is `7af17bd`.
- **`var/datastores/`.** 18 files, all tombstone sidecars, one of them under
  `backup-pre-resume-20260921T091011/`. Their full SHA-256, mtime and size were recorded before
  `reference` (§3.9).
- 44 GB free on `var/`'s volume (96 % used) before `reference`, 40 GB after it.

#### 3.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch.** Each ran into a new directory under the
agent's scratchpad, with §1.1.2's commands. Both exited 0
([`interrupter_dummy_test_08.txt`](runregistry-extraction/rehearsal/interrupter_dummy_test_08.txt),
[`interrupter_build_dummy_test_08.txt`](runregistry-extraction/rehearsal/interrupter_build_dummy_test_08.txt)):
the five cases 23:56:18–23:56:42, the `--build` cases at 23:56:42.
- **Against R1's.** With times, pids and scratch paths stripped, each output equals R1's
  `interrupter_dummy_test_04.txt` (40 lines) and `interrupter_build_dummy_test_04.txt` (20 lines),
  line for line.
- **No dummy child or interrupter was left alive.** `ps -p` on all 16 pids the two outputs name
  found none.

**The points, from R2's own reference `stdout.log`, sized as the orchestrator's note directs.**
Both queues have a 300 s minimum notify interval (`main.py:119`).
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **27.4 s**
  (`stdout.log:76`; R1: 102.9 s; R0: 78 s). It was interrupted from the banner with
  `--interval 5 --hold 11`.
  - 40 % of 27.4 s is 10.96 s, rounded to 11 s: just above the 10 s floor, which therefore did not
    bind.
  - It puts the signal 11–16 s into a stage of about 27 s, so that `resumed-a` leaves stored and
    unvalidated rows for `resumed-b` to prune, as at R0 and R1. It did (§3.3.2).
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **1 m 59.8 s (119.8 s)**
  (`stdout.log:100`; R1: 494.9 s). It was interrupted from the banner with `--interval 5 --hold 40`.
  - A third of the stage is 39.9 s, rounded to 40 s, well under the 250 s cap.
  - The pattern is `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its end, as at R0 and
    R1.

**The real firings** ([`interrupter_firings_08.txt`](runregistry-extraction/rehearsal/interrupter_firings_08.txt)):

```
resumed-a (interrupter pid 84032; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 11)
2026-10-16T00:04:22+01:00 liveness alive; pid 83931; stage 'CALCULATE WKB PART OF TENSOR GREEN FUNCTIONS'
2026-10-16T00:05:32+01:00 liveness alive; pid 83931; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-16T00:05:32+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 11 s, then reading again
2026-10-16T00:05:43+01:00 sent SIGTERM to pid 83931 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/runregistry-extraction-08-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 88011; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 40)
2026-10-16T00:07:42+01:00 liveness alive; pid 87907; stage 'APPLY GKSOURCE POLICIES'
2026-10-16T00:07:52+01:00 liveness alive; pid 87907; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-16T00:07:52+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 40 s, then reading again
2026-10-16T00:08:32+01:00 sent SIGTERM to pid 87907 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/runregistry-extraction-08-resumed.sqlite … --register resumed-b …'; exiting
```

- **Each interrupter checked the pid before it signalled**: its command line named
  `quadsource_atol_sweep.py`, its manifest's `argv[0]`, and `--register <slug>`. It refused no
  real run's pid.
- **Each interrupter read its stage from `status.json`**, which `runregistrykit` now writes. Both
  banners were read, as at R0 and R1. No progress line was read: no stage of R2 ran past 300 s.
- **The interrupter logs a reading only when the stage changes**, and reads every 5 s. So:
  - the `GkSource` stage began between 00:05:27 and 00:05:32, and the signal came 11–16 s into it;
  - the QSI stage began between 00:07:47 and 00:07:52, and the signal came 40–45 s into it.
- **No progress line appeared during `resumed-b`'s hold.**

Both interrupters were launched detached, about 5 s after their builds, once the run directory and
its `status.json` existed, in the one check after the launch:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/runregistry-extraction-08-resumed-a-20261016T000342 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 11 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/runregistry-extraction-08-resumed-b-20261016T000708 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 40 > /dev/null 2>&1 &
disown
```

#### 3.1.3 The lock

Every helper command and its exit status
([`lock_08.txt`](runregistry-extraction/rehearsal/lock_08.txt)), each run as
`PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock <command>` from the repository root:

| When | Command | Exit | What it said |
|---|---|---|---|
| 23:57:01 | `acquire --purpose "runregistry-extraction 08 (R2)"` | 0 | acquired, for SecondaryGWKit |
| 23:57:02 | `status` | 0 | `HELD`; `pid: -` (no pid recorded yet); no Ray process |
| 23:57:37 | `set-pid 76434` (`reference`) | 0 | recorded |
| 23:57:37 | `status` | 0 | `pid: 76434`, alive; 19 Ray processes, the run's |
| 00:03:09 | `status` | 0 | `pid: 76434`, NOT alive; no Ray process |
| 00:03:52 | `set-pid 83931` (`resumed-a`) | 0 | recorded |
| 00:03:52 | `status` | 0 | `pid: 83931`, alive; 18 Ray processes, the run's |
| 00:06:45 | `status` | 0 | `pid: 83931`, NOT alive; no Ray process |
| 00:07:16 | `set-pid 87907` (`resumed-b`) | 0 | recorded |
| 00:07:16 | `status` | 0 | `pid: 87907`, alive; 14 Ray processes, the run's |
| 00:09:38 | `status` | 0 | `pid: 87907`, NOT alive; no Ray process |
| 00:10:03 | `set-pid 91423` (`resumed-c`) | 0 | recorded |
| 00:10:03 | `status` | 0 | `pid: 91423`, alive; 14 Ray processes, the run's |
| 00:12:38 | `status` | 0 | `pid: 91423`, NOT alive; no Ray process |
| 00:16:30 | `status` | 0 | `pid: 91423`, NOT alive; no Ray process (after the readers and the guard) |
| 00:16:31 | `ray-processes` | 0 | none |
| 00:16:31 | `release` | 0 | released |
| 00:16:31 | `status` | 0 | `lock: free`; no Ray process |
| 00:16:31 | `ray-processes` | 0 | none |

- The owner file named `project: SecondaryGWKit`, `purpose: runregistry-extraction 08 (R2)`,
  `ray_temp_dir: in-process` and `ray_address: in-process` throughout.
- `acquire` came 15 s before `reference`'s manifest (23:57:16).
- Between launches the lock was held and its pid was the previous driver's, dead: the protocol, not a
  stale lock.
- `release` was not refused: the recorded pid was dead and no Ray process of this repository was
  alive. No wait by pid was needed before it (§3.2.3).
- Afterwards `~/ScienceData/` holds `ChamPBH-rehearsals/` alone, as before.

### 3.2 L1 — every launch ends as recorded

#### 3.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `encoding_format` 1, `39809dca…`, problems `{}` | absent | not written: "…runregistry-extraction-08-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `encoding_format` 1, `87e55e8f…`, problems `{}` | absent | not written: "…runregistry-extraction-08-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `encoding_format` 1, `8978b72e…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `encoding_format` 1, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint holds 21 classes, and its `taken` names its run, `git_head` `b9daf449…` and
`git_dirty` false. Each `status.json` has R1's nine keys exactly; its `fingerprint` has R1's five
keys and `encoding_format`.

The full digests:
- `reference` and `resumed-c`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `87e55e8fad8d95010f5739d0bf1aaa445a185030797fdbfe4adc7fd70b3cd87d`;
- `resumed-b`: `8978b72e8d41837d45118740b9a7f414a5ea651db7d3758137f4c52725eb8c41`.

The two intermediate digests differ from R1's (`a410cce0…`, `8146dab8…`), as they must: each
digests a store killed at a different point. By their recorded class counts:
- `resumed-a`'s store held `GkSource` 165, `GkSourcePolicyData` 0 and `QuadSourceIntegral` 0;
- `resumed-b`'s held 291, 291 and 297.

#### 3.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 00:05:43 | 00:08:32 |
| Handler's line | `** SIGTERM at 2026-10-16 00:05:43 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-16 00:08:33 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s (equal whole-second stamps) | at most about 1 s (stamps one second apart) |
| `stderr.log`'s handler lines | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` | the line above, then `** SIGTERM: the pipeline ended with SystemExit(143); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Fingerprint taken / final heartbeat | 00:05:43 / 00:05:44 | 00:08:33 / 00:08:35 |
| Stage at the signal | the `GkSource` banner, first read 00:05:32, signal 11 s later (**165 of 291 rows stored, 44 unvalidated**) | the `QuadSourceIntegral` banner, first read 00:07:52, signal 40 s later (297 of 864 stored) |

- **`SystemExit(1)` in `resumed-a`, `SystemExit(143)` in `resumed-b`**: the other way round from R0
  and R1. §1.2.2 gives the reason: a signal that lands during `ray.wait` has Ray's signal check
  convert the handler's `SystemExit(143)` into `SystemExit(1)`; one that lands elsewhere does not.
  Which it is depends on where in the pool's loop the signal falls. The latch is what
  `finish_after_exec` reads, so the record is `killed`/143 and the exit is 143 either way, as here.
- **`stderr.log` of `resumed-a` and of `resumed-b`** hold the handler's two lines and the raylet's
  disk-space lines (11 and 8, §3.2.5), and nothing else.
- **`stderr.log` of `reference`** holds the SciPy `IntegrationWarning` ("The occurrence of roundoff
  error is detected…", worker pid 76517), as at every rehearsal, and 26 disk-space lines.
- **`stderr.log` of `resumed-c`** holds the same warning (worker pid 92163), as at R1, and 9
  disk-space lines.

#### 3.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 00:03:00 (~71 s after the final heartbeat; load 18.79) | none: `ray-processes` exit 0, `pgrep` empty, pids 76430 and 76434 gone, and the Ray pids its logs name (76517, 76518, 76519, 76521) gone |
| `resumed-a` | 00:06:28 (~44 s; load 5.94) | none, by the same checks for pids 83928, 83931, the interrupter 84032, and the Ray pids 83985, 83989, 83990 |
| `resumed-b` | 00:09:22 (~47 s; load 10.71) | none, by the same checks for pids 87904, 87907, the interrupter 88011, and the Ray pids 87999, 88000, 88001 |
| `resumed-c` | 00:12:38 (~59 s; load 5.59) | none, for pids 91420, 91423 and the Ray pids 91543, 91545, 92163 |

- **No Ray process outlived either killed driver**, so no wait by pid was run, as at R0 and R1.
- `[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur. The issue stays open; it is
  not this campaign's.
- No launch log or `stderr.log` holds a `ray.kill`, killed-actor, `ActorDiedError` or
  `RayActorError` line (§3.2.5).

#### 3.2.4 The manifests

From [`run_records_08.txt`](runregistry-extraction/rehearsal/run_records_08.txt), for all four:
- `git_head` `b9daf449722f3e443f8dd592c1a890752cbeed6c`, the dispatch `HEAD`; `git_dirty` false.
- **`packages`** equals, compared as JSON, the record of §3's header: `runregistrykit` 0.1.0 at
  `9f8ebe43…` (`v0.1.0`), `datastorekit` 0.4.0 at `c8570e84…` (`v0.4.0`) and `raypoolkit` 0.2.0 at
  `eaef0ce6…` (`v0.2.0`), none editable, each with its URL. On disk the keys are alphabetical, as
  every record is written with `sort_keys`; the comparison is of the parsed dictionaries, not their
  order (the orchestrator's correction 5).
- `script` is `docs/handover/quadsource_atol_sweep.py`. **`script_sha256`** is
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`, the same in all four. R1's was
  `b7935877…`: 06 rewrote the launcher's registry calls and 06's review its `--force` message.
- `argv[0]` is `docs/handover/quadsource_atol_sweep.py`. `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only.
- `results` names the reference store for `reference`, and the resumed store for the other three.
  `results_store_id` is `null` for `reference` and `resumed-a`, whose stores had no sidecar at
  launch, and `86b70a5d…` for `resumed-b` and `resumed-c`, as at R0 and R1.
- `scope` is R1's with `runregistry-extraction-08` for `runregistry-extraction-04`. The manifests
  have R1's nineteen keys exactly.

#### 3.2.5 Every launch's `stdout.log`, and what is new in the logs

Counted in each of the four, as §1.2.5
([`log_comparison_08.txt`](runregistry-extraction/rehearsal/log_comparison_08.txt), section 2):

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--zend', '250', …]` | 1 |
| `--no-prune-unvalidated`, in `stdout.log` or the launch log | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |
| `reconcil`, `repair`, `in.flight`, `replicat` (case-insensitive), in `stdout.log` or the launch log | 0 |
| `Traceback`, `ModuleNotFoundError`, `ImportError`, `AttributeError`, `pickle`/`unpickl`, `Datastore.SQL`/`Datastore/SQL`, `ray.kill`, `killed actor`, `ActorDiedError`, `RayActorError`, `ValueError`, `RuntimeWarning`, `did not validate`, in any of the three logs | 0 |
| `DeprecationWarning`, `_timestamp`, **`ConfigError`**, `RunRegistry` (not followed by `Kit`), in any of the three logs | **0** |
| **`datastorekit`**, anywhere in a line | **0**, as in R0's and R1's twelve |
| **`runregistrykit`**, anywhere in a line | 1 in `stdout.log` and 1 in the launch log: the banner's `list it with:` line, and nothing else |
| `INTEGRITY WARNING`, in any of the three logs | 0, but for `resumed-b`: 2 in `stdout.log` and 2 in the launch log |
| raylet `file_system_monitor` (`… is over 95% full …`) | 0 in `stdout.log`; in `stderr.log` and again in the launch log, 26 (`reference`), 11 (`resumed-a`), 8 (`resumed-b`), 9 (`resumed-c`); R1: 0 |

The launch logs of `resumed-b` and `resumed-c` hold `** quadsource_atol_sweep --build: resuming
into runregistry-extraction-08-resumed.sqlite`.

**The swap's one expected line.** The launcher's banner says, in every `stdout.log` and launch log,

```
**   list it with: ./venv/bin/python -m runregistrykit list
```

where R1's said `**   list it with: PYTHONPATH=. ./venv/bin/python -m RunRegistry list`. 06 changed
it with the launcher's registry calls (the orchestrator's correction 6). It is the only line in the
twelve build logs that names `runregistrykit`, and no line names `datastorekit`, even through a path:
a traceback or warning raised inside either package would. The counts of every other search equal
R1's but for `SystemExit` and the disk-space lines, below.

**The comparison with R1, slug by slug.** The method is §1.2.5's: each line of each R2 log
normalised (ANSI codes stripped, the campaign and prompt and the repository path replaced, Ray's
`[repeated Nx across cluster]` count and every number replaced by `N`) and compared, as a set of
line kinds, with R1's same file for the same slug. The script was a throwaway in the agent's
scratchpad, not committed.

| Slug | `stderr.log` | launch log | `stdout.log` |
|---|---|---|---|
| `reference` | R2 has 1 kind R1 has not, the disk-space line; 30 lines against 4 | R2 has 3 kinds R1 has not: the banner line, the disk-space line, and a `Queue summary` in `/sec`; R1 has 3 kinds R2 has not: its banner line, and its one progress line and `inflight` line; 142 lines against 118 | R2 has the banner line and the `/sec` summary; R1 its banner line and its progress pair; 110 lines against 112 |
| `resumed-a` | R2 has the disk-space kind; 13 lines against 2 | the banner lines, and the disk-space kind; 99 lines against 88 | the banner lines only; 84 lines against 84 |
| `resumed-b` | R2 has the disk-space kind; 10 lines against 2 | the banner lines, and the disk-space kind; 144 lines against 148 | the banner lines only; 131 lines against 143 |
| `resumed-c` | R2 has the disk-space kind; 13 lines against 4 | the banner lines, the disk-space kind, and R1's progress pair; 123 lines against 116 | the banner lines and R1's progress pair; 107 lines against 109 |

**Every difference, explained.**
1. **The banner's `list it with:` line** (above), in every `stdout.log` and launch log: the swap's
   one expected line.
2. **Ray's disk-space warning**, `(raylet) file_system_monitor.cc:116: /tmp/ray/session_… is over
   95% full, available space: 39.9 GB; capacity: 926.4 GB. Object creation will fail if spilling is
   required.`, every 10 s from each raylet, into `stderr.log` and the launch log. Ray warns when the
   volume holding its temp directory passes 95 %. The volume was 96 % used throughout R2 (44 GB free
   before `reference`, 40 GB after it), against 93 % at R1 (72 GB). It concerns object spilling, and
   no object was spilled: no line says so, nothing failed, and both stores are `39809dca…`. It is the
   machine's, not the swap's, and changed nothing a build wrote or read.
3. **The pool's progress lines.** None of R2's stages ran past the queue's 300 s minimum notify
   interval (the reference's QSI stage took 119.8 s, `resumed-c`'s 74.6 s), so R2 printed none.
   R1's `reference` and `resumed-c` printed one each, in QSI stages of 494.9 s and 343.8 s. Whether a
   stage prints one is the threshold and the load (the orchestrator's correction 4).
4. **The `Queue summary` rate unit**, `/sec` against `/min`, in `reference`. The pool chooses the unit
   from the rate; R2's stages ran faster. The line's form is the same (§1.2.5, item 2).
5. **`SystemExit(1)` and `SystemExit(143)`**, swapped between `resumed-a` and `resumed-b` against
   R1 (§3.2.2). Ray's signal check converts the exit when the signal lands inside `ray.wait`, and
   where it lands is the point's.
6. **`resumed-b`'s `stdout.log` is 12 lines shorter.** Its integrity warning lists the rows the
   prune removed: shard 3's block lists its 9 rows in full (`stdout.log:13-34`), where R1's listed
   15, two lines each; Ray folds shard 0's 12 and shard 1's 23 into one block with
   `[repeated 35x across cluster]` (`:120-125`). 9 + 35 = 44, the rows `resumed-a` left unvalidated
   (§3.3.2). The line kinds are R1's.
7. **The SciPy `IntegrationWarning`** is in `reference` and `resumed-c`, as at R1. `resumed-b`
   stored 297 QSI rows before its point, so the one integral fell after it.

**Nothing else differs.** No warning or traceback appears that no earlier rehearsal printed, but the
machine's disk-space line. Every pool line, banner, completion and `Queue summary` alike, has R1's
form (§3.5.6).

#### 3.2.6 A3 v2's guard

After `resumed-c` and the readers, in the foreground, with no `--register`, the lock still held
([`guard_08.txt`](runregistry-extraction/rehearsal/guard_08.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/runregistry-extraction-08-resumed.sqlite \
    --job-name runregistry-extraction-08 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (00:16:25), with R0's and R1's `RuntimeError` (now raised from
`quadsource_atol_sweep.py:537`, reached from `:1055` and `:1039`, which 06's lines moved), naming the
five files of `runregistry-extraction-08-resumed.sqlite` that exist and refusing to build into an
existing store.
- **Nothing changed.** Afterwards, `shasum -a 256 -c` gave `OK` on every file of both stores and
  both sidecars, against the hashes taken just before it, after the readers.
- **Nothing was created.** No run directory appeared (81 entries under `var/runs/` before and
  after), and no Ray process of this repository and no journal appeared.

### 3.3 L2 — the resume is the reference

#### 3.3.1 Nothing stored is recomputed

Through `dk-and-rpk-migration`'s
[`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py), as 06 left it
([`stage_counts_08.txt`](runregistry-extraction/rehearsal/stage_counts_08.txt)), with §1.3.1's
command and this prompt's run directories. Lookup / compute / store:

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **170/170/170** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **567/567/567** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored **165** `GkSource` rows before the signal (55 + 57 + 0 + 53 by shard), of
    which **44** were unvalidated (12 + 23 + 0 + 9) (§3.3.2).
  - `resumed-b` pruned those 44 at open, and computed **170 = 291 − 165 + 44**: the 126 never
    stored, and the 44 pruned. No validated row was computed again.
  - Summed, less the pruned: 165 + 170 − 44 = 291, the reference's.
- **`QuadSourceIntegral`.**
  - `resumed-b` printed no progress line in the stage. Its fingerprint, taken by `finish` on the
    closed store, records **297** rows.
  - `resumed-c` computed **567** = 864 − 297, exactly what was missing.
  - Summed: 297 + 567 = 864, the reference's. No stored row was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b`, because `resumed-a` never reached it, and
  0/0/0 in `resumed-c`.
- The `reference` and `resumed-a` columns equal R1's stage by stage.

#### 3.3.2 The prune at startup

The closed store after each launch, read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`, with §1.3.2's command. Each file's SHA-256
was unchanged by the read
([`closed_store_state_08.txt`](runregistry-extraction/rehearsal/closed_store_state_08.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **`GkSource` 12 + 23 + 0 + 9 = 44**, of 55 + 57 + 0 + 53 = 165 stored, each listed by serial. `GkNumericIntegration` 456 + 406 + 0 + 506 = 1368 and `GkWKBIntegration` 3 × 1159 = 3477 are all stored and validated |
| `resumed-b` | none | 0 rows | none (`GkSource` 97 + 97 + 0 + 97 = 291) |
| `resumed-c` | none | 0 rows | none |

**`resumed-a` stored and left unvalidated rows, as the point intended** (§3.1.2). The signal came
11–16 s into a stage the reference took 27.4 s over. 165 of 291 rows were stored by then, and 44 of
them were not yet validated. R1's `resumed-a` left 207 rows (48 unvalidated), R0's 164 (63).

**By serial**, through `dk-and-rpk-migration`'s
[`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py), unchanged, with §1.3.2's
commands and this prompt's state files
([`prune_check_08.txt`](runregistry-extraction/rehearsal/prune_check_08.txt)):
- **The first ran twice:** at 00:09:45, after `resumed-b` had opened and closed the store, and again
  at 00:12:48, after `resumed-c`. Both times it gave:
  - `44 serial(s) checked: 44 absent, 0 reused by a validated row, 0 still unvalidated`, exit 0;
  - by shard, 12 serials in shard 0, 23 in shard 1 and 9 in shard 3, each `absent`.
- **The second**, after `resumed-c`, gave `none: there is nothing to check`, exit 0.
- Every file of the store had the same SHA-256 before and after each check.

**So the prune at open removed exactly the rows `resumed-a` left unvalidated**, and no validated row,
with the runs now registered through `runregistrykit`. Each pruned serial is gone, and none was
reused.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§3.3.5). Each finish's fingerprint has `problems` `{}`, for all four launches.

**The pool's prune of a replicated class.** None. After every launch each shard's `BackgroundModel`
row is validated, and the census finds the replicated tables equal (§3.3.5).

#### 3.3.3 The version row

From `closed_store_state.py`, after every launch: exactly one row, `(1, '2025.1.1')`, on each of the
four shards of both stores. The primary has no `version` table.

#### 3.3.4 The comparison

**The two stores**, by `compare_stores.py`, with §1.3.4's command and this prompt's stores
([`fingerprint_comparison_08.txt`](runregistry-extraction/rehearsal/fingerprint_comparison_08.txt)).
It exited 0: both stores format 6, digest `39809dca…`, problems none, each with 0 differences
against its sidecar; `compare_fingerprints(reference, resumed): 0 difference(s)`; `verdict: equal`.
All 21 classes are `equal`. With the campaign and prompt normalised, the output equals R1's
`fingerprint_comparison_04.txt` line for line.

**Against R2's own `reference` and `resumed-c`**, by `compare_with_recorded.py`, as 06 left it
([`fingerprint_against_recorded_08.txt`](runregistry-extraction/rehearsal/fingerprint_against_recorded_08.txt)).
It is run against R2's own run directories only: after 06 it compares through the package, which
gives one `encoding` entry for every fingerprint SGK's code recorded (README §4). R0 and R1 are
compared field by field in §3.10.

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/runregistry-extraction-08-reference.sqlite \
    var/datastores/runregistry-extraction-08-resumed.sqlite \
    --against var/runs/runregistry-extraction-08-reference-20261015T235716 \
    var/runs/runregistry-extraction-08-resumed-c-20261016T000959
```

It exited 0, verdict `equal`. For each store there were 0 differences against:
- its sidecar;
- R2's reference (taken at `b9daf449…`, `git_dirty` false);
- R2's `resumed-c` (taken at `b9daf449…`, `git_dirty` false).

All are format 6, digest `39809dca…`, 21 classes, problems none. The files of both stores had the
same SHA-256 before and after these reads.

| Class | Reference | Resumed | Digests | R1's reference |
|---|---|---|---|---|
| `version` | 1 | 1 | equal | 1 |
| `store_tag` | 10 | 10 | equal | 10 |
| `redshift` | 1159 | 1159 | equal | 1159 |
| `wavenumber` | 3 | 3 | equal | 3 |
| `tolerance` | 7 | 7 | equal | 7 |
| `LambdaCDM` | 1 | 1 | equal | 1 |
| `QCD_Cosmology` | 1 | 1 | equal | 1 |
| `IntegrationSolver` | 7 | 7 | equal | 7 |
| `GkSourcePolicy` | 2 | 2 | equal | 2 |
| `QuadSourcePolicy` | 2 | 2 | equal | 2 |
| `wavenumber_exit_time` | 3 | 3 | equal | 3 |
| `BackgroundModel` | 1 | 1 | equal | 1 |
| `TkNumericIntegration` | 3 | 3 | equal | 3 |
| `TkWKBIntegration` | 3 | 3 | equal | 3 |
| `GkNumericIntegration` | 1368 | 1368 | equal | 1368 |
| `GkWKBIntegration` | 3477 | 3477 | equal | 3477 |
| `GkSource` | 291 | 291 | equal | 291 |
| `GkSourcePolicyData` | 291 | 291 | equal | 291 |
| `QuadSource` | 6 | 6 | equal | 6 |
| `QuadSourceIntegral` | 864 | 864 | equal | 864 |
| `OneLoopIntegral` | 0 | 0 | equal | 0 |

The last column's digests are compared in §3.10, field by field.

#### 3.3.5 Integrity, both stores, read-only

**The census**, with §1.3.5's command and this prompt's stores
([`census_08.txt`](runregistry-extraction/rehearsal/census_08.txt)): every table is `OK` on both
stores (28 lines), and the verdict is `no divergence` on both. Exit 0. With the campaign and prompt
normalised, the output equals R1's `census_04.txt` line for line.

**`integrity_check`, `foreign_key_check` and the inventory**, through `dk-and-rpk-migration`'s
[`store_soundness.py`](dk-and-rpk-migration/rehearsal/store_soundness.py), as 06 left it
([`store_soundness_08.txt`](runregistry-extraction/rehearsal/store_soundness_08.txt)):
- It exited 0, `=> clean`.
- On the primary and every shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is
  `['ok']` and `PRAGMA foreign_key_check` returns 0 rows.
- The package's `read_inventory(primary, factories)` reads shards `(0, 1, 2, 3)`, and reports no
  problem in any class of either store, and 0 `duplicate`.
- With the campaign and prompt normalised, the output equals R1's `store_soundness_04.txt` line for
  line.
- Every file of both stores had the same SHA-256 before and after these reads, and no journal
  appeared.

### 3.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance
(`--ray-address local`), through `dk-and-rpk-migration`'s
[`run_readers.sh`](dk-and-rpk-migration/rehearsal/run_readers.sh), unchanged, with §1.4's two
commands and this prompt's store and output directory, the lock held
([`readers_08.txt`](runregistry-extraction/rehearsal/readers_08.txt)). Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` / traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 27 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 531 | none | identical |
| `extract_GkWKB_data.py` | 0 | 55 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 16 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 176 | none | identical |
| `extract_TkWKB_data.py` | 0 | 12 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | 0 | 12 s | `EXTRACT TENSOR SOURCE DATA`: 6 | 27 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 15 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 | 71 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 22 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 423 | none | identical |

- **The lookups equal R1's, queue by queue and reader by reader**
  ([`log_comparison_08.txt`](runregistry-extraction/rehearsal/log_comparison_08.txt), section 4).
- **The files written differ** in the readers that draw random subsamples, as between R0 and R1:
  against R1's 315, 73 and 369, R2 wrote 531, 71 and 423. Each draws its subsample with
  `random.sample`; it is not a finding.
- **The reader logs' line kinds against R1's**, normalised as §3.2.5: three are equal
  (`extract_TkWKB_data`, `extract_tensor_source_data`, `extract_QuadSourceIntegral_data`). The other
  four differ only by:
  - the raylet's disk-space line (§3.2.5), in `extract_Gk_data` (2), `extract_GkWKB_data` (4),
    `extract_GkSource_data` (1) and the `nomodels` run (1);
  - `extract_GkWKB_data`'s queue, which finished in 44.3 s and so printed neither R1's progress line
    and its `inflight` line, nor a completion time in `Nm Ns`;
  - Matplotlib's warnings, which both print: the `RuntimeWarning` "More than 20 figures have been
    opened" in `extract_GkWKB_data` (R2 2 lines, R1 5) and the `MatplotlibDeprecationWarning` in
    `extract_GkSource_data` (R2 2, R1 3). Ray folds repeats into `[repeated Nx across cluster]`
    lines, and R1's logs hold one folded form of each that R2's do not.

  No reader log names `runregistrykit` or `datastorekit`, and none holds a `Traceback`,
  `ReadOnlyMiss`, `ReadOnlyWrite`, `ModuleNotFoundError`, unpickling error or `_timestamp`.
- **The load.** The readers ran 00:13:01–00:15:48, at a one-minute load of 26–56.
- **The model and the run.** Every run printed `** reading run "default", the only run in this
  datastore (no --run-label given)`. With `--models LambdaCDM` each printed `>> RUNNING PIPELINE FOR
  MODEL LambdaCDM`; without it, `extract_Gk_data.py` read LambdaCDM, then printed the
  `QCD_Cosmology` skip line, and exited 0.
- **No file of the store changed.** Before and after each reader, the SHA-256, mtime and size of the
  primary and of every shard were identical, and no `-journal`, `-wal` or `-shm` file appeared. No
  Ray process of this checkout was left after any reader.
- **The fingerprint afterwards.** `./venv/bin/python -m runregistrykit store fingerprint` (without
  `--write`) on the resumed store gave `39809dca…`, problems none, "matches the fingerprint taken
  2026-10-16T00:11:37+01:00 by run runregistry-extraction-08-resumed-c-20261016T000959, at
  b9daf449722f3e443f8dd592c1a890752cbeed6c", exit 0. Every file of the store has the hashes taken
  after `resumed-c`, before the readers.

### 3.5 L4 — equal to R1

#### 3.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

`policy_statistics.py`, with §1.5.1's command, on each of R2's runs. Exit 0 on `reference`,
`resumed-b` and `resumed-c`; exit 1 on `resumed-a`, which has no block, as at R0 and R1. The
reference's block, lines 83–96 of its `stdout.log`, verbatim
([`policy_statistics_08_reference.txt`](runregistry-extraction/rehearsal/policy_statistics_08_reference.txt)):

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

| | R1 | R2 |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| `reference`'s order under `mixed` | `minimal`, `complete` | `complete`, `minimal` |
| `reference`, against R1's | — | differs in the order of those two lines; equal sorted; equal to R0's `reference` byte for byte |
| `resumed-b` (`stdout.log:103`) | `complete` first | equal to R1's `resumed-b` byte for byte |
| `resumed-c` (`stdout.log:83`) | `minimal` first | equal to R1's `resumed-c` byte for byte |

**Every count is R1's**, in every run that prints the block. The order is Ray's completion order:
`[02-the-policy-statistics-block-orders-its-lines-by-completion]`, with one more instance and no
new fact.

#### 3.5.2 The reference run's stage counts

From [`stage_counts_08.txt`](runregistry-extraction/rehearsal/stage_counts_08.txt), against R1's
`stage_counts_04.txt`: the `reference` column is equal in every stage, and so is the `resumed-a`
column, since both were killed in the same stage. (The table is §3.3.1's.)

#### 3.5.3 The resumed chain, by identity

| Identity | R1 | R2 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated (`prune_check.py`) | 48 = 48, all absent, none reused | **44 = 44, all absent, none reused** | ✅ |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | 132 = 291 − 207 + 48 | **170 = 291 − 165 + 44** | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 440 = 864 − 424 | **567 = 864 − 297** | ✅ |
| Totals per class over the chain equal the reference's | `GkSource` 207 + 132 − 48 = 291; QSI 424 + 440 = 864 | `GkSource` 165 + 170 − 44 = 291; QSI 297 + 567 = 864 | ✅ |

What each resumed launch pruned and recomputed:

| Launch | Pruned at open | Computed, stage by stage | Of which recomputing a pruned row |
|---|---|---|---|
| `resumed-a` | nothing (new store) | everything up to the `GkSource` stage, as `reference`; **165 `GkSource` stored** at the signal, 44 unvalidated | — |
| `resumed-b` | **44 `GkSource`** (the unvalidated ones), in shards 0, 1 and 3 | `GkSource` 170; `GkSourcePolicyData` 291; `QuadSourceIntegral` 297 stored at the signal; every earlier stage 0 | 44 `GkSource` |
| `resumed-c` | nothing | `QuadSourceIntegral` 567; every other stage 0 | — |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 3.5.4 The stage sequence

The lines of each `stdout.log` that begin `** `
([`log_comparison_08.txt`](runregistry-extraction/rehearsal/log_comparison_08.txt), section 3):
- R2's and R1's hold 26, 23, 26 and 26 such lines, slug by slug.
- With the campaign and prompt, the repository path and every number normalised, each slug's
  sequence equals R1's, in order, but for one line: the third, the banner's `list it with:` line,
  which names `runregistrykit` where R1's named `RunRegistry` (§3.2.5).
- In the reference, 15 of the 26 are the pipeline's stage banners, `** BUILD ARRAY OF WAVENUMBERS
  TO SAMPLE` through `** CALCULATE QUADRATIC SOURCE INTEGRALS`. They equal R1's verbatim, in order,
  with nothing normalised. The other 11 are the launcher's own lines.

#### 3.5.5 The stage times

Not compared (`ALL WORK ITEMS COMPLETE in time …` in each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 29.7 s | 25.4 s | 1.61 s | 1.73 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 1 m 11.2 s | 1 m 11.4 s, no progress line | 3.58 s | 3.71 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **27.4 s**, no progress line | killed | 20.5 s | 5.64 s |
| APPLY GKSOURCE POLICIES | 10.4 s | — | 10.4 s | 5.46 s |
| QUADRATIC SOURCE INTEGRALS | **1 m 59.8 s**, no progress line | — | killed, no progress line | 1 m 14.6 s, no progress line |

- Each earlier stage took under 1.1 s.
- **The reference took 4 m 33 s**, against R1's 13 m 57 s, R0's 29 m 19 s and
  `dk-and-rpk-migration` R2's 11 m 25 s.
- **The `GkSource` stage took 27.4 s and the QSI stage 119.8 s**, against R1's 102.9 s and
  494.9 s. The machine was lightly loaded (§3.1). The stores are equal, so the time is the
  machine's.

#### 3.5.6 The printed lines and the registry's fields

Against R1's same slug ([`log_comparison_08.txt`](runregistry-extraction/rehearsal/log_comparison_08.txt), section 3):

| Line, in `stdout.log` | `reference` R2 / R1 | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| `** ` (banners and the launcher's lines) | 26 / 26 | 23 / 23 | 26 / 26 | 26 / 26 |
| `-- ALL WORK ITEMS COMPLETE in time …` | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `Queue summary: …`, one after each completion line | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `(no compute items were evaluated - all assets read from datastore)` | 1 / 1 | 1 / 1 | 7 / 7 | 9 / 9 |
| progress: `n/m work items remaining = p% complete` | **0** / 1 | 0 / 0 | 0 / 0 | **0** / 1 |
| progress: `inflight: … \| completed: …` | **0** / 1 | 0 / 0 | 0 / 0 | **0** / 1 |

- **Every completion line is followed by its `Queue summary` line**, in all four.
- **The forms are R1's.** Normalised as §3.2.5, the completion lines of R2's four runs make 2
  kinds, the `Queue summary` lines 7 and the `(no compute items …)` line 1. Each set equals R1's.
- **No progress line** was printed: no stage ran past 300 s (§3.2.5, item 3).
- **The registry's fields at each run's end**, R2 against R1: `state`, `stage`, `units_done` and
  `units_total` are R1's in all four, and each `status.json` has R1's keys exactly. `runregistrykit`
  read the pool's printed lines as `RunRegistry` did.

| Slug | `state` | `stage` | `units_done` / `units_total` | R1's `stage` |
|---|---|---|---|---|
| `reference` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` | the same |
| `resumed-a` | `killed` | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | 0 / `null` | the same |
| `resumed-b` | `killed` | `CALCULATE QUADRATIC SOURCE INTEGRALS` | 0 / `null` | the same |
| `resumed-c` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` | the same |

### 3.6 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen (§3.3.2).
- **An in-flight record left by a killed launch.** Did not happen: `replication_in_flight` had 0
  rows after every launch.
- **A repair by the check at open.** Did not happen (§3.2.5).
- **A Ray agent outliving a killed driver.** Did not happen (§3.2.3).
- **A line from the swap** but the banner's: a `runregistrykit` or `datastorekit` frame, a
  `ConfigError`, a `DeprecationWarning`. Did not happen (§3.2.5).
- **A record written outside `var/runs/` and `var/datastores/`.** Did not happen: `git status
  --short` stayed empty to the last launch, and nothing under `var/` but R2's own runs, stores and
  working directory is newer than the first capture (§3.9).
- **A printed line the registry or the interrupter could not read.** Did not happen (§3.5.6).
- **A missed interruption.** Did not happen. Both killed launches ended `killed`, in the stage the
  interrupter waited for.
- **A held lock, or a refused `release`.** Did not happen (§3.1.3).

**A killed run's record, seen by the waits.** The orchestrator's waits ended on the driver pid's
death, as `CLAUDE.md`'s liveness rule reads it (board §5.7), and each resumed the agent with the
complete record: `resumed-a`'s handler recorded `killed` at 00:05:43, the fingerprint was taken at
00:05:43 and the final heartbeat came at 00:05:44; `resumed-b`'s at 00:08:33, 00:08:33 and 00:08:35.

### 3.7 Issues opened

None. Every difference from R1 is explained by:
- the swap's one expected line, the launcher's banner;
- the machine: the disk-space warning, from a volume 96 % used, and the light load, with the
  pool's 300 s notify threshold;
- where a launch was killed, and where in Ray's wait the signal fell;
- Ray's completion order (an issue already open);
- or the readers' random subsamples.

### 3.8 The suites

After the last launch, the readers, the guard and the release: README §8's command, each root in
the foreground, serially, its output in the agent's scratchpad and its verdict taken by `grep`.

| Root | Ran | Result | Load (1 min) at start | Wall |
|---|---|---|---|---|
| AdaptiveLevin | 32 | OK | 7.9 | 1 s |
| CosmologyModels | 39 | OK | 7.9 | 2 s |
| ObjectFactories | 279 | OK | 7.4 | 14 s |
| tools | 91 | OK | 6.9 | 10 s |
| LiouvilleGreen | 148 | OK (1 skipped) | 6.3 | 23 s |
| RunDrivers | 99 | OK | 5.6 | 4 s |
| ComputeTargets | 693 | OK | 14.3 | 140 s |

Every count is 07's. No test failed, so none of README §8's four known wall-clock or chance tests
needed a re-run. No Python file was added or changed, so `black` had nothing to check.

### 3.9 What stays under `var/`

- `var/runs/runregistry-extraction-08-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`. Each
  holds the run's manifest, `status.json` with its full fingerprint, `stdout.log` and `stderr.log`.
  For `resumed-a` and `resumed-b` it also holds `interrupter.log`.
- `var/runs/runregistry-extraction-08-<slug>.launch.log`, beside each run directory.
- `var/datastores/runregistry-extraction-08-{reference,resumed}.sqlite`, four shards each, and each
  store's `.manifest.json` sidecar. Shard 2 of each, 1 077 248 bytes, holds only replicated rows.
  Neither store is retired.
- `var/runregistry-extraction-08/`, which holds:
  - the agent's working notes (`NOTES.md`), the pre-launch captures, the lock record, the dummy-test
    outputs and the listings of `var/datastores/`;
  - the closed-store states, the prune checks and the hashes (`state/`);
  - the check outputs (`checks/`), among them every run's statistics block and the log comparisons;
  - the readers' outputs and logs (`extract/`, `readers.txt`).

**Nothing else under `var/` moved**
([`var_datastores_08.txt`](runregistry-extraction/rehearsal/var_datastores_08.txt)).
- Before the first launch, `var/datastores/` held 18 files, all tombstone sidecars.
  `find var/datastores -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256` was
  `1fbaf5b766d2a006e611ba65939ce5c742b11ef7d1f37c238ea2f03995ea8a6e`, the orchestrator's.
- After the last step it held 30. The 18 have the same SHA-256, mtime (to the nanosecond) and size
  as before. The 12 new files are this rehearsal's two stores and their sidecars. The digest over
  the 30 is `23f8da12e7842ce83306b260ec60a882bc90f62fbc1cf9db2e5b23237b1feed0`.
- No `-journal`, `-wal` or `-shm` file is under `var/datastores/`.
- Nothing under `var/` is newer than the first listing (23:56:07) but R2's runs, stores and working
  directory, and the two directories holding them.
- **The registry after R2** ([`run_records_08.txt`](runregistry-extraction/rehearsal/run_records_08.txt)):
  `./venv/bin/python -m runregistrykit list` shows **43 runs**, nothing `running` or stale: 23
  `done`, 15 `killed`, 3 `failed`, 2 `unknown` (the `a3-pilot` pair). They are 01's 29, whose ids and
  states equal `measure/out/01-list.txt`'s, and this campaign's fourteen: R0's, R1's and R2's twelve
  launches, and 06's two worktree runs.

### 3.10 W1 across stores

**What it carries** (README §4, amended 2026-10-14). 04's W1 showed that SGK's own `fingerprint_of`
and DatastoreKit's agree on the same store. SGK's is deleted (06), and R0's and R1's stores are
retired (board §5.8). So R2 carries W1's claim across stores: the package's fingerprint of each R2
store equals, in every field but `encoding_format`, the fingerprints SGK's own code recorded for R0
and R1.

**The script** is [`runregistry-extraction/fingerprint_both_ways.py`](runregistry-extraction/fingerprint_both_ways.py)
as 06 rewrote it, unchanged. For each primary, read-only and with no Ray, it takes the store's
fingerprint through `config.fingerprint.fingerprinter`, the one `runregistrykit` uses, and checks:
1. the store's `encoding_format` is `datastorekit.store_fingerprint.ENCODING_FORMAT`, 1, and its
   digest is `--expect-digest`, here its default, `39809dca…`;
2. the sidecar, which the package wrote, records a fingerprint, and `compare_fingerprints(sidecar's,
   current)` is empty;
3. for each recorded fingerprint: it lacks `encoding_format`; its digest is `39809dca…`; **field by
   field, with `encoding_format` and `taken` set aside on both sides**, it equals the store's, every
   differing path printed with both values; and the package's `compare_fingerprints(recorded,
   current)` is exactly one entry, of kind `encoding`.

That is four checks of its own per store and four per recorded fingerprint: with four run
directories, 20 per store (the orchestrator's correction 7). It exits 0 if and only if every check
holds, and writes nothing.

**The run** ([`fingerprint_both_ways_08.txt`](runregistry-extraction/rehearsal/fingerprint_both_ways_08.txt)),
at 00:17:21, after the readers, the guard and the release:

```bash
PYTHONPATH=. ./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py \
    var/datastores/runregistry-extraction-08-reference.sqlite \
    var/datastores/runregistry-extraction-08-resumed.sqlite \
    --against \
    var/runs/runregistry-extraction-02-reference-20261012T091921 \
    var/runs/runregistry-extraction-02-resumed-c-20261012T101145 \
    var/runs/runregistry-extraction-04-reference-20261014T154651 \
    var/runs/runregistry-extraction-04-resumed-c-20261014T161739
```

It exited 0 in 5 s, **20 of 20 checks on each store**, with the summary `2 store(s) against 4
recorded fingerprint(s); every check holds: each store's fingerprint equals every recorded one in
every field but encoding_format and taken`.

| R2 store | Its fingerprint | Against its sidecar | Against each of R0's and R1's four records |
|---|---|---|---|
| reference | format 6, `encoding_format` 1, `39809dca…`, 21 classes | 0 entries (taken 00:03:17 by `store fingerprint --write`) | the record lacks `encoding_format`; `39809dca…`; 0 differing paths; the package's comparison one `encoding` entry, recorded `None`, current 1 |
| resumed | format 6, `encoding_format` 1, `39809dca…`, 21 classes | 0 entries (taken 00:11:37 by `resumed-c`) | the same, for all four |

- **In each of the eight pairs every field is equal but `encoding_format` and `taken`**:
  `fingerprint_format` 6, the overall digest, `problems` `{}`, and for each of the 21 classes its
  `count`, its `digest` and every tag set's `tags`, `count` and `digest`, the tag sets in the same
  order.
- **So `runregistrykit`'s fingerprint, through SGK's fingerprinter and DatastoreKit's
  `fingerprint_of`, is what SGK's own code recorded** for R0 and R1. With 04's W1, which showed the
  two implementations equal on one store, the claim reaches the tree where only the package's is
  left.
- **The one `encoding` entry** against each record is
  `[00-recorded-fingerprints-lack-the-encoding-format]`, measured (closed at 06): the package's
  comparison stops at the encoding and compares nothing else, which is why the script compares the
  content itself.
- **The stores were untouched.** Every file of R2's two stores and their sidecars (12) had the same
  SHA-256, size and `mtime_ns` before and after the run.
- This prompt adds no breakage of the script: 06 named and ran them, on the same code (06's log).

### 3.11 RunRegistryKit's G2

RunRegistryKit's SGK checklist, item 9 ("Its acceptance", `docs/adoption/secondarygwkit.md` at
`aaab408`, read with `git show`), clause by clause, with its evidence. The measure ran from
RunRegistryKit's root, with its own venv, at its `HEAD` `caad864` (clean); it reads SGK through git
only, and nothing in RunRegistryKit changed
([`g2_measure_08.txt`](runregistry-extraction/rehearsal/g2_measure_08.txt)).

| # | Clause | Evidence | Holds |
|---|---|---|---|
| 1 | No `RunRegistry` import, string literal or `python -m RunRegistry` outside `prompts/` and dated `docs/` prose; `measure_client_registry.py sgk --commit <its commit>` reports 0 imports in live code, and the staying tests are no longer under `RunRegistry/` | **Here**, at the dispatch `HEAD` `b9daf44`: `./venv/bin/python docs/adoption/measure_client_registry.py sgk --commit b9daf449722f3e443f8dd592c1a890752cbeed6c`, exit 0, 422 tracked `.py` files parsed: import statements 0, names used 0, calls bound 0, string literals naming it 0, `-m RunRegistry` 0, in live code and in the staying tests (48 name-alike calls, none the registry's). `git grep -nP 'RunRegistry(?!Kit)' -- . ':!prompts' ':!docs/*.md' ':!docs/**/*.md' ':!docs/**/*.txt'` prints `CLAUDE.md:82` and `config/fingerprint.py:33`, both prose naming the old package's history (06's ruling 1), and nothing else. `git ls-tree HEAD RunRegistry` is empty and `import RunRegistry` raises `ModuleNotFoundError`. The staying driver tests are under `RunDrivers/tests/` (06, board item A3) | ✅ |
| 2 | Its roots pass, `RunDrivers` with the moved driver tests and `ObjectFactories` with `TestGolden` | 06 (§7.5), 07 (`07-roots.txt`), and **here** at `b9daf44` (§3.8): all seven, RunDrivers 99 and ObjectFactories 279 among them | ✅ |
| 3 | `python -m runregistrykit list` gives its 29 runs, with the same states and purposes as its 01's baseline, and `store show` reads each of its sidecars as that baseline does | 06 (§7.1: every run's id, state, purpose and stage, and every `store show` block, equal to the `RunRegistry` readings byte for byte). **Here**: `list` exit 0, 43 runs, 01's 29 by id and state equal to `01-list.txt`; `store show` on all 18 tombstones exit 0, kind `registry`, problems none, `retired` | ✅ |
| 4 | A run begun from a linked worktree is recorded under the main checkout's `var/runs/` | 06 (§7.4): `runregistry-extraction-06-worktree-check-20261015T224836`, begun in a linked worktree at the probe `ec781383…`, recorded in this checkout's `var/runs/` with `git_dirty` false; `list` shows it `done` here | ✅ |
| 5 | A fingerprint taken through the new path carries `encoding_format` beside the digest, and its digest equals `fingerprint_of`'s on 06's test store | **Here**: all four R2 runs' `status.json` fingerprints and both sidecars carry `encoding_format` 1 beside the digest (§3.2.1, §3.10). 06's `TestConfigFingerprint` test 2 shows the fingerprinter's fingerprint equals `fingerprint_of(read_inventory(primary, factories), 6)` on the test store; it passes here in ObjectFactories' 279 | ✅ |
| 6 | `requirements.txt` holds the pin, installed from it, not editable; every new manifest's `packages` names `runregistrykit`, `datastorekit` and `raypoolkit` | `requirements.txt:81` `runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0` (05); `pip show` and `package_provenance` give 0.1.0 at `9f8ebe43…`, `requested_revision` `v0.1.0`, not editable (§3.1.1). **Here**: all four R2 manifests' `packages` name the three, at their pins, none editable, equal as JSON to the record (§3.2.4) | ✅ |

**Every clause holds, so RunRegistryKit's G2 holds.** It rests on 06 (`740bafc`), 07 (`a9ed928`)
and this prompt's runs `runregistry-extraction-08-reference-20261015T235716` and
`runregistry-extraction-08-resumed-c-20261016T000959`, both `done` and `39809dca…`. It is recorded on
this repository's board; 09 drafts the record for RunRegistryKit's (README §6.2 C11).

---

## 4. The close

**§4's tree.** Prompt 09 ran on branch `handover-remedial` at **`ef34ed0`** ("Write the orchestrator note
for runregistry-extraction prompt 09"), clean. Its code is `a9ed928`'s: `git diff --stat b9daf44 33a7346 --
. ':!prompts' ':!docs/runregistry-extraction-verification.md' ':!docs/runregistry-extraction'` was empty at
the orchestrator's check (2026-10-16), and `33a7346`, `c07ab17` and `ef34ed0` touch `prompts/` alone. 09
changes records only: no code, no test and no script. **Every measurement below that reads the committed
tree (RunRegistryKit's measure, the `git grep`) therefore measures the tree 09 started from, and 09's own
commit changes no file either of them reads.** The measurements are saved whole, each with its command,
under `prompts/runregistry-extraction/measure/out/09-*`.

### 4.1 The three rehearsals, as one table

Twelve runs, each read from its run directory's `manifest.json` and `status.json`
([`09-runs.txt`](../prompts/runregistry-extraction/measure/out/09-runs.txt), with the full digests). Every
`git_dirty` is `false`. Every `fingerprint_format` is 6, and every `done` run's overall digest is
`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`.

| Rehearsal | Run (`runregistry-extraction-NN-…`) | Dispatch `git_head` (code) | Packages in `packages` | `script_sha256` | Digest | `encoding_format` | End state |
|---|---|---|---|---|---|---|---|
| R0 (02, §1) | `02-reference-20261012T091921` | `fe16d8e` (`a45aefc`) | `datastorekit` 0.2.2, `raypoolkit` 0.2.0 | `b7935877…` | `39809dca` | none | `done`, 0 |
| R0 | `02-resumed-a-20261012T095041` | `fe16d8e` | same | `b7935877…` | `9bb753bd` | none | `killed`, 143 |
| R0 | `02-resumed-b-20261012T100353` | `fe16d8e` | same | `b7935877…` | `a0d053af` | none | `killed`, 143 |
| R0 | `02-resumed-c-20261012T101145` | `fe16d8e` | same | `b7935877…` | `39809dca` | none | `done`, 0 |
| R1 (04, §2) | `04-reference-20261014T154651` | `89ff8b7` (`5b81f2e`) | `datastorekit` 0.4.0, `raypoolkit` 0.2.0 | `b7935877…` | `39809dca` | none | `done`, 0 |
| R1 | `04-resumed-a-20261014T160226` | `89ff8b7` | same | `b7935877…` | `a410cce0` | none | `killed`, 143 |
| R1 | `04-resumed-b-20261014T161222` | `89ff8b7` | same | `b7935877…` | `8146dab8` | none | `killed`, 143 |
| R1 | `04-resumed-c-20261014T161739` | `89ff8b7` | same | `b7935877…` | `39809dca` | none | `done`, 0 |
| R2 (08, §3) | `08-reference-20261015T235716` | `b9daf44` (`a9ed928`) | `datastorekit` 0.4.0, `raypoolkit` 0.2.0, `runregistrykit` 0.1.0 | `2a985303…` | `39809dca` | 1 | `done`, 0 |
| R2 | `08-resumed-a-20261016T000342` | `b9daf44` | same | `2a985303…` | `87e55e8f` | 1 | `killed`, 143 |
| R2 | `08-resumed-b-20261016T000708` | `b9daf44` | same | `2a985303…` | `8978b72e` | 1 | `killed`, 143 |
| R2 | `08-resumed-c-20261016T000959` | `b9daf44` | same | `2a985303…` | `39809dca` | 1 | `done`, 0 |

- **The packages' commits.** `datastorekit` is `6d8ee9ae…` (0.2.2) at R0 and `c8570e84…` (0.4.0) at R1 and R2;
  `raypoolkit` is `eaef0ce6…` throughout; `runregistrykit` is `9f8ebe43…` at R2. None is editable.
- **The pattern is the same in all three.** The three `done` references and the three `resumed-c` runs
  carry `39809dca…`; each interrupted run's digest differs from it, because it records a store mid-build.
  The digest does not depend on which of the three registries and datastore layers built the store.
- **`encoding_format` appears at R2 and not before**, on every R2 fingerprint, `done` or `killed`: the
  fingerprint was taken through `datastorekit.store_fingerprint` (06). R0's and R1's were taken by
  SGK's own code and do not carry it, which is `[00-recorded-fingerprints-lack-the-encoding-format]`
  (closed at 06; board §4).
- **The launcher's hash changed once**, between R1 and R2, when 06 made it register through
  `runregistrykit.begin`; it is the same in all four R2 manifests and equals the file's hash now (§4.5).
- **The code commit, and the dispatch commit beside it**: R0 `a45aefc` (01) at `fe16d8e`; R1 `5b81f2e` (03)
  at `89ff8b7`; R2 `a9ed928` (07) at `b9daf44`. Each dispatch commit adds records to `prompts/` only.

### 4.2 Z1 — the final measurements

Taken at `ef34ed0`, clean. Each is in a file under `prompts/runregistry-extraction/measure/out/` with the command
that made it. Where a figure equals the orchestrator's or 08's, it says so.

**The seven roots** ([`09-roots.txt`](../prompts/runregistry-extraction/measure/out/09-roots.txt)), each once, in the
foreground, serially, against 08's counts:

| Root | Ran | Result | Wall | 08's count |
|---|---|---|---|---|
| `AdaptiveLevin/tests` | 32 | OK | 0.05 s | 32 |
| `CosmologyModels/tests` | 39 | OK | 0.7 s | 39 |
| `ObjectFactories/tests` | 279 | OK | 13.0 s | 279 |
| `tools/tests` | 91 | OK | 9.1 s | 91 |
| `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 23.0 s | 148 (1 skipped) |
| `RunDrivers/tests` | 99 | OK | 3.1 s | 99 |
| `ComputeTargets/tests` | 693 | OK | 138.3 s | 693 |

No failure, and none of README §8's four wall-clock tests failed. They ran at one-minute load 8–22, after
every record edit of 09 and before its commit; 09 changes no code.

**The packages** ([`09-pip.txt`](../prompts/runregistry-extraction/measure/out/09-pip.txt)).
- `pip freeze` has 101 lines. It equals 05's "before" list (lines 4–103 of `05-pip.txt`, 100 lines) plus
  the one line 05's own diff adds,
  `runregistrykit @ git+https://github.com/ds283/RunRegistryKit@9f8ebe431503fb8c79ee1407cc460501f1ca2d92`;
  the `diff` against the "before" list is that `81a82` line alone, and nothing was installed or
  removed since.
- `pip check`: `No broken requirements found.`
- `runregistrykit.package_provenance(["datastorekit", "raypoolkit", "runregistrykit"])`, as JSON with sorted
  keys, **equals the `packages` of each of R2's four manifests** (`json.load` equality, four of four):
  `datastorekit` 0.4.0 at `c8570e84…` (`v0.4.0`), `raypoolkit` 0.2.0 at `eaef0ce6…` (`v0.2.0`),
  `runregistrykit` 0.1.0 at `9f8ebe43…` (`v0.1.0`), none editable. (`package_provenance` takes the
  names; called with none it raises `TypeError`.)

**Every `RunRegistry` reference outside `prompts/` and dated `docs/` prose**
([`09-measure.txt`](../prompts/runregistry-extraction/measure/out/09-measure.txt)):
- RunRegistryKit's `docs/adoption/measure_client_registry.py sgk --commit ef34ed06…`, run from its root
  (`caad864`, clean) with its own venv, exit 0: **422** tracked `.py` files parsed; every live-code
  and staying-test figure is **0** (import statements, those at module level, files importing it, names
  used, calls bound, string literals naming it, `-m RunRegistry`), the 48 name-alike calls in live code
  being none the registry's; campaign records: 6 `.py` files import it, 7 name it. It equals the
  measure at `b9daf44` (§3.11) and the orchestrator's at `33a7346`, but for the commit.
- `git grep -nP 'RunRegistry(?!Kit)' -- . ':!prompts' ':!docs/*.md' ':!docs/**/*.md' ':!docs/**/*.txt'`
  prints `CLAUDE.md:82` and `config/fingerprint.py:33` and nothing else: both name the old package's
  history. `git ls-tree HEAD RunRegistry` is empty and `import RunRegistry` raises `ModuleNotFoundError`.

**The registry and the stores.**
- `./venv/bin/python -m runregistrykit list`
  ([`09-list.txt`](../prompts/runregistry-extraction/measure/out/09-list.txt)), exit 0: **43 runs**, 23 `done`,
  15 `killed`, 3 `failed`, 2 `unknown` (the pre-registry `a3-pilot` pair), **none `running` and none
  stale**. 01's 29 are all present, each in the state `01-list.txt` recorded (compared by id and state).
- `store show` on every sidecar under `var/datastores/`
  ([`09-store-show.txt`](../prompts/runregistry-extraction/measure/out/09-store-show.txt)): **twenty** sidecars, found
  with `find var/datastores -name '*.manifest.json'` (a glob of the top level finds nineteen, one being
  in `backup-pre-resume-20260921T091011/`). Each exits 0, kind `registry`, **`problems: none`**: 18 are
  `retired` (the tombstones) and two are live, R2's `runregistry-extraction-08-reference` and `-resumed`.
- **The machine** ([`09-disk.txt`](../prompts/runregistry-extraction/measure/out/09-disk.txt)): `/` is **97 % used,
  36 GB free** (`df -h /`; `/tmp` and the repository are on it), against 96 % and 40–44 GB at R2 and 35 GB at the
  orchestrator's check. `RAY_ADDRESS` is unset; `tools.rehearsal_lock status` says the lock is free and
  no Ray process of this repository is up. The load was 10–28 during the measurements (§4.5, "What bears").

### 4.3 Z2 — the two gate records, drafted

Neither is committed in its repository. The user makes those commits, or has a session there make them
(README §6.2 C11, §6.3). Each is the text to paste, in its board's own form.

#### 4.3.1 RunRegistryKit's G2

RunRegistryKit's `prompts/extraction/IMPLEMENTATION_STATE.md` §2 says a gate is "recorded on this board when
it holds, in a records commit of its own; nothing else on the board changes for it". So the draft is the
**status cell of the G2 row**, in E1's and E2's form (`**holds** <date>: …`), keeping the checklist
pointer. It replaces
`not yet (SGK adopts `v0.1.0` in its `runregistry-extraction` 05–08); its checklist is …`.

```markdown
| **G2** | **holds** 2026-10-16: SGK adopted `v0.1.0` in its `runregistry-extraction` 05–08 (05 `3dcab1b` pinned `runregistrykit[stores] @ v0.1.0`; 06 `740bafc` adopted it and deleted `RunRegistry/`, reviewed at `7af17bd`; 07 `a9ed928` guarded the fingerprint format; 08 `957d658` rehearsed through it), and every clause of checklist item 9 holds, each read in SGK's `docs/runregistry-extraction-verification.md` (§3.11 at `957d658`, §4.2 at 09's commit). (1) **No import:** `measure_client_registry.py sgk --commit ef34ed06…` (this repository at `caad864`) reports 0 imports, names, bound calls, string literals and `-m RunRegistry` in live code and in the staying tests, 422 `.py` files parsed, and at `b9daf44` the same; `git grep` prints only the two prose lines naming the old package's history; `RunRegistry/` is gone from the tree; the staying driver tests are `RunDrivers/tests/`. (2) **The roots pass**, `RunDrivers` (99, with the moved driver tests) and `ObjectFactories` (279, with `TestGolden`) among all seven, at 06, 07 and 08, and at 09's commit. (3) **`list` gives its runs**: 06 showed 01's 29 equal to the `RunRegistry` baseline byte for byte, `store show` reading each sidecar as the baseline did; at 09, `list` gives 43 runs, 01's 29 each in its recorded state, none `running`, and `store show` on all 20 sidecars exits 0, kind `registry`, problems none. (4) **A linked-worktree run is recorded in the main checkout**: `runregistry-extraction-06-worktree-check-20261015T224836`, begun from a linked worktree at the probe `ec781383…`, recorded under the main checkout's `var/runs/`, `git_dirty` false, `done`. (5) **A fingerprint through the new path carries `encoding_format`** beside the digest: all four R2 runs' fingerprints and both R2 sidecars carry `encoding_format` 1 beside `39809dca…`, and 06's `TestConfigFingerprint` shows the digest equal to `fingerprint_of`'s on its test store. (6) **The pin is installed, and manifests name all three:** `requirements.txt:81` holds `runregistrykit[stores] @ …@v0.1.0`; `pip freeze` has it at `9f8ebe43…`, not editable, `pip check` clean; all four R2 manifests' `packages` name `runregistrykit` 0.1.0, `datastorekit` 0.4.0 and `raypoolkit` 0.2.0, equal as JSON to `package_provenance`. R2 reproduced SGK's reference fingerprint through `runregistrykit` and `datastorekit` together: `runregistry-extraction-08-reference-20261015T235716` and `-resumed-c-20261016T000959`, both `done`, both `39809dca…`. No defect of this package was met in the rehearsal; SGK's five inherited issues are in its `runregistry-extraction` brief, and the one found in adoption is `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]`; its checklist is [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md) (07, `aaab408`) |
```

The checklist's six clauses, each with the SGK evidence the draft names:

| # | Checklist clause (`docs/adoption/secondarygwkit.md` §9 at `aaab408`) | SGK evidence | Where |
|---|---|---|---|
| 1 | No `RunRegistry` import, string literal or `-m RunRegistry` outside `prompts/` and dated `docs/` prose; the measure reports 0 imports; the staying tests are not under `RunRegistry/` | the measure and the `git grep` at `ef34ed0` (§4.2) and `b9daf44` (§3.11); `RunRegistry/` gone from the tree; `RunDrivers/tests/` | §3.11 row 1, §4.2 |
| 2 | Its roots pass, `RunDrivers` and `ObjectFactories` with `TestGolden` | 06, 07, 08 and 09's suites | §3.11 row 2, §4.2 |
| 3 | `list` gives its 29 runs in 01's states, and `store show` reads each sidecar as the baseline does | 06 (`740bafc`) byte for byte; 09: 43 runs, 01's 29 by id and state; 20 sidecars | §3.11 row 3, §4.2 |
| 4 | A linked-worktree run is recorded under the main checkout's `var/runs/` | `runregistry-extraction-06-worktree-check-20261015T224836` (06) | §3.11 row 4 |
| 5 | A fingerprint through the new path carries `encoding_format`, and its digest equals `fingerprint_of`'s on the test store | R2's four fingerprints and two sidecars (08); `TestConfigFingerprint` (06) | §3.11 row 5, §3.10 |
| 6 | The pin is installed, not editable, and every new manifest's `packages` names the three | `requirements.txt:81`; 09's `pip freeze`, `pip check` and provenance; R2's four manifests | §3.11 row 6, §4.2 |

#### 4.3.2 DatastoreKit's

DatastoreKit's `prompts/fingerprint/IMPLEMENTATION_STATE.md` §2 is one paragraph, with one dated italic note,
`*(2026-10-14: `v0.4.0` is made … are given to the user for SGK's E1.)*`, and says SGK's R1 result "is recorded
here when SGK reports it, in a records commit of its own; a difference is a defect of this package, opened in
§3". So the draft is a **second dated italic note appended to that paragraph**, reporting R1's both-ways
result first, then what has adopted the release since. "No difference" is the finding; no issue is opened.

```markdown
*(2026-10-16: SGK's R1 result, and what has adopted `v0.4.0` since. **The fingerprint taken both ways is equal**, so this package's `fingerprint_of` is SGK's on real stores. SGK's `runregistry-extraction` 04 (`ec61569`) read each of four stores, read-only and with no Ray, and took the fingerprint with SGK's own code and with `datastorekit.store_fingerprint.fingerprint_of(inventory, 6)`, on R1's two stores (`runregistry-extraction-04-reference`, `-resumed`, built through `v0.4.0`) and R0's two (built through `v0.2.2`): on each, all 20 checks hold, with the package's `encoding_format` 1 the only extra key. With it removed, the two fingerprints are equal key for key and in order (format 6, overall digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21 classes, each class's `count`, `digest` and tag sets), the listings are equal line for line but the first, and the package's `compare_fingerprints(sgk, pkg)` is exactly one entry, of kind `encoding`, as the addendum says it should be. The four stores' files were byte-identical before and after. **Adopted since.** SGK pinned `datastorekit` at `v0.4.0` (03, `5b81f2e`) and rebuilt through it (R1: `runregistry-extraction-04-reference-20261014T154651` and `-resumed-c-20261014T161739`, both `done`, `39809dca…`). SGK's fingerprinter, `config/fingerprint.py`, is RunRegistryKit's `DatastoreKitFingerprinter` over SGK's factories, so every fingerprint SGK takes goes through `datastorekit.store_fingerprint` and records `encoding_format` beside the digest (06, `740bafc`); all four R2 runs' fingerprints and both R2 sidecars carry `encoding_format` 1. SGK's R2 (08, `957d658`; runs `runregistry-extraction-08-reference-20261015T235716` and `-resumed-c-20261016T000959`, both `done`, `39809dca…`) repeated the both-ways check across stores: each R2 store's fingerprint equals each of the four fingerprints SGK recorded for R0 and R1 in every field but `encoding_format` and `taken`, 20 of 20 checks on each, against each record the package's comparison one `encoding` entry, and the stores untouched. 07 (`a9ed928`) pins the digest of every inventory spec so that a spec change without a format bump fails. No difference, and no defect of this package, was found; nothing is opened in §3. Record: SGK's `docs/runregistry-extraction-verification.md` §2.10, §3.10 and §4.)*
```

### 4.4 Z3 — the registry issues, placed

Each of the four issues `[00-open-issues-about-the-registry-code-stay-on-sgk-boards]` names, and the two it
leaves, was read against `runregistrykit` at `v0.1.0` (`git show v0.1.0:<path>`, RunRegistryKit read with `git`
only, its `HEAD` `caad864`, whose `git diff --stat v0.1.0 caad864 -- runregistrykit pyproject.toml` is
empty) and against SGK at `ef34ed0`. Line numbers are the tag's for the package.

| Issue (SGK board, line at `ef34ed0`) | Whose | The code | Still there |
|---|---|---|---|
| `[04-runs-do-not-say-which-machine-produced-them]` (`run-registry` `:61`) | the package's; **a decided change**, not a defect | `begin` `runregistrykit/__init__.py:604`; the manifest `:707-723` (`argv` `:712`, `cwd` `:713`), `git_head` `:152`, `packages` `:725`; `pid_alive` `:258`, `liveness` `:294`. `git grep -n -i 'machine\|platform\|node()' v0.1.0 -- runregistrykit` finds one unrelated comment (`stores.py:240`) and one in a test | yes: no `machine`, not built |
| `[00-a-copy-carries-its-sources-present-tense-fields]` (`store-retirement` `:508`) | the package's | `copy_store` `runregistrykit/stores.py:1138`; docstring "every other field, unknown ones included, is the source's, verbatim"; `fields = _copy.deepcopy(reading.fields)` `:1160`, seven assignments `:1161-1172` | yes |
| `[01-cross-filesystem-move-advice-says-delete-by-hand]` (`store-retirement` `:525`) | **two halves**: the advice text is DatastoreKit's (`datastorekit/SQL/ShardedPool.py:3029`, in its `docs/OPEN_ISSUES.md` §1.2); **the relaying half is the package's** | `_store_files` `stores.py:1119-1129` (re-raise `:1125-1128`); `move_store` `:1179`, calling `_store_files("move", ShardedPool.move_store, src, dst)` at `:1195` | yes |
| `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` (`a3-v2-readiness` `:1375`) | the package's, though SGK's handler triggers it: nothing in SGK can avoid the shared inode | `write_json_atomic` `__init__.py:242-251`, `tmp = path + ".tmp"` at `:245`; `Run._update` `:398-403` (write at `:402`); `heartbeat` `:405`, `finish` `:424` | yes |
| `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]` (this board) | the package's | `sys.stdout.flush()` `runregistrykit/__main__.py:563`, unguarded; `main` `:408` | yes |
| `[00-a-sigterm-pipeline-run-is-recorded-as-failed]` (`store-retirement` `:471`) | **SGK's** | `terminal_state` `docs/gktk-remedial/scoped_pipeline_run.py:256`, `install_killed_handler` `:412`, `finish_after_exec` `:429`; `git grep -nP "def (terminal_state\|finish_after_exec\|install_killed_handler)\(" v0.1.0` in RunRegistryKit is empty, and the package has no `SIGTERM` or `signal.` outside its tests | n/a |
| `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]` (`a3-v2-readiness` `:1392`) | **SGK's** | the window is between `ray.init` (Ray's handler) and `install_killed_handler` (`:412`), installed after it returns; the package is not involved | n/a |

- **So five are the package's and two are SGK's.** The two SGK's stay where they are, untouched, and the brief
  names them as SGK's, with these lines, so that RunRegistryKit is not asked for them.
- **The brief** is
  [`prompts/runregistry-extraction/runregistrykit-inherited-issues-brief.md`](../prompts/runregistry-extraction/runregistrykit-inherited-issues-brief.md):
  the five, each complete in RunRegistryKit's board-entry form, with a proposed `docs/OPEN_ISSUES.md` row each.
- **Nothing is transferred.** RunRegistryKit's `docs/OPEN_ISSUES.md` holds its own two and none of these. So each
  of the four stays in its SGK board's §3 with one line, `**Drafted for RunRegistryKit (2026-10-16,
  runregistry-extraction prompt 09):**`, giving the package's lines and the brief; this board's two stay,
  each with a `**Narrowed (2026-10-16, prompt 09):**` line. The four move to their boards' §4 as transferred,
  and this board's two close, in a records commit after the user has handed the brief over.

### 4.5 Z4 — what A3 v2 inherits

For `handover`, which owns the A3 v2 launch (`a3-v2-readiness` README §1). From the tree at `ef34ed0` and the records.

**The launcher.** `docs/handover/quadsource_atol_sweep.py --build --register <slug> --purpose "<one line>"`.
- Its SHA-256 at `ef34ed0` is `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`
  (`2a985303…`), **equal to every R2 manifest's `script_sha256`**: it has not changed since R2. Its last commit
  is `7af17bd`.
- **The flags R2 ran it with:** `--build --rehearsal [--resume] --database var/datastores/<store>.sqlite
  --job-name <job> --atol 1e-32 --rtol 1e-7 --cpus 6 --register <slug> --purpose "…" --campaign <campaign>
  --prompt <NN>`.
  **A production build drops `--rehearsal`**, which selects a three-wavenumber, `--zend 250` scope that "is not
  A3 v2" (`--help`; `:989-993`). `--job-name` defaults to `handover-A3-baseline-v2` and `--campaign` to `handover`.
  `--build` needs `--database`, `--atol` and `--rtol` (`:1031-1037`), and `--register` needs `--purpose`.
- **It forces `address="local"`** at the `ray.init` it wraps (`:610`), "never another `ray start` cluster this
  machine may be running", and installs the driver's signal handler after `ray.init` returns. It starts a fresh
  local Ray instance of `--cpus` CPUs with no dashboard.
- Launched detached, as R2's were (§3.1): `nohup sh -c '… ; echo "** driver exit status $?"' "<purpose>" >
  var/runs/<id>.launch.log 2>&1 &` then `disown`, one check that it started, and the turn ends.

**The commands**, as R2 ran them (§3.1, §3.1.3), each in its current spelling:
```bash
./venv/bin/python -m runregistrykit list                                    # discovery, before anything
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock acquire --purpose "<text>"
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes
./venv/bin/python -m runregistrykit store show var/datastores/<store>.sqlite   # absent before the build
# the launch above, then at once:
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock set-pid <driver pid>
# after the build ends, and its last Ray process is gone:
./venv/bin/python -m runregistrykit store create --purpose "<one line>" var/datastores/<store>.sqlite
./venv/bin/python -m runregistrykit store fingerprint --write var/datastores/<store>.sqlite
./venv/bin/python -m runregistrykit store show var/datastores/<store>.sqlite
PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock release
# retiring a store, when it is no longer wanted: a dry run first
./venv/bin/python -m runregistrykit store retire --dry-run --reason "<text>" var/datastores/<store>.sqlite
./venv/bin/python -m runregistrykit store retire --reason "<text>" var/datastores/<store>.sqlite
```
- **The lock's five commands** are `acquire --purpose`, `set-pid`, `status`, `ray-processes` and `release`. **A held
  lock (exit 3) means stop and ask the user**: never wait, retry or remove it. Every driver names its address
  (`local`), and `ray stop` is never run.
- R2's `store create` ran on the closed primary after the build (§3.1); `resumed-b` and `resumed-c` then each
  recorded `fingerprint_sidecar: written`. 09 ran only `list` and `store show`.

**Its store.** The A3 v2 store is **the first live store recorded in the new layout that is meant to be kept**.
- Every sidecar under `var/datastores/` is a tombstone except R2's two (§4.2: 18 `retired`, two live), and R2's two
  are throwaway: `runregistry-extraction-08-reference` and `-resumed`, retired by the user after this review, or not
  (README §3, §6.3).
- It is a new name under `var/datastores/`, and `store show` must say `absent` for it before the first launch.

**What bears on it**, each pointing at its record:
- **`[00-main-and-the-readers-default-to-ray-address-auto]`** (this board's §3, `:205`; `docs/OPEN_ISSUES.md` §1.1;
  **assigned to `handover`**): `main.py` and the six `extract_*.py` default to `--ray-address auto`. The launcher
  forces `local` and `run_readers.sh` passes it, so a build is safe; a reader launched by hand without
  `--ray-address local` attaches to any Ray instance it finds. Until it is fixed, every run states its address.
- **`[07-a-killed-builds-ray-agents-outlive-the-driver]`** (`a3-v2-readiness` board `:1439`; index §1.17): after a
  killed build exits 143 its dashboard and runtime-env agents outlive it by 1–2 minutes, so "no Ray process left"
  is true only after waiting by pid for every process from the venv. R2 found none to wait for (§3.2.3), at light load.
- **The disk.** The volume Ray spills to is **97 % used, 36 GB free** (§4.2,
  [`09-disk.txt`](../prompts/runregistry-extraction/measure/out/09-disk.txt)): past Ray's 95 % threshold, where every
  raylet warned every 10 s at R2 that "object creation will fail if spilling is required". R2 spilled nothing, and both
  stores were `39809dca…`. A3 v2 is long and wide and may need to spill. It was recorded at 08's review (board §5.14) and
  not opened as an issue, at the orchestrator's ruling: it is the machine's, not the code's.
- **The registry's issues do not block it.** The five handed to RunRegistryKit (§4.4) are all open on their boards:
  none is a defect a build meets in the normal case. The one a second-machine A3 build meets is
  `[04-runs-do-not-say-which-machine-produced-them]` (a decided change, not built): until it is, run directories
  are transferred only when terminal, and `list` is read on the machine that produced them.
- **The pins** are `datastorekit` v0.4.0, `raypoolkit` v0.2.0 and `runregistrykit[stores]` v0.1.0, tags and never
  editable (`requirements.txt:17`, `:76`, `:81`); `ray` and `SQLAlchemy` are SGK's own pins.
