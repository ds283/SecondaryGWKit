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
