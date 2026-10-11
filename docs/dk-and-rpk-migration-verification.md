# dk-and-rpk-migration campaign — the three rehearsals

**Campaign:** [`prompts/dk-and-rpk-migration/README.md`](../prompts/dk-and-rpk-migration/README.md) ·
**Board:** [`prompts/dk-and-rpk-migration/IMPLEMENTATION_STATE.md`](../prompts/dk-and-rpk-migration/IMPLEMENTATION_STATE.md) ·
**Protocol:** README §4.1, which is `a3-v2-readiness` prompt 07's
([`a3-v2-readiness-verification.md`](a3-v2-readiness-verification.md)) with the names changed

This record holds the campaign's three rehearsal rebuilds, one section each: **§1 R0**, the start
tree (prompt 02); §2 R1, through `datastorekit` (prompt 05); §3 R2, through both packages (prompt
08). 09 adds §4, the close. Each section quotes its commands verbatim, and transcribes every
measurement, because the stores are throwaway.

**§1's tree, stores and runs.** Every R0 launch ran on branch `handover-remedial` at **`a5392fd`**
("Write the orchestrator note for dk-and-rpk-migration prompt 02"), whose code is `f54f1c5`'s
(prompt 01). Its datastore layer is `b510bc9`'s, byte for byte, and its work queue is the frozen
`RayTools/`. `git status --short` was empty at every launch, and every manifest stamps
`git_dirty: false` and `packages` `{"datastorekit": null, "raypoolkit": null}`. A3 v2's own launcher,
`docs/handover/quadsource_atol_sweep.py --build --rehearsal --register`, built two throwaway stores:
- **`var/datastores/dk-and-rpk-migration-02-reference.sqlite`** (sidecar `store_id`
  `b9a1b4132a4f4066887c077bd4adae5a`), built straight through by the run **`reference`**;
- **`var/datastores/dk-and-rpk-migration-02-resumed.sqlite`** (`store_id`
  `157d877a6a7e440ead4211e5c1425ba4`), built by **`resumed-a`** (killed by `SIGTERM` in the
  `GkSource` stage), **`resumed-b`** (`--resume`, killed in the `QuadSourceIntegral` stage) and
  **`resumed-c`** (`--resume`, to `done`).

The runs are `var/runs/dk-and-rpk-migration-02-<slug>-<stamp>/`. Both stores stay, with their
sidecars; no prompt of this campaign retires a store (README §3). The small outputs quoted below are
in [`dk-and-rpk-migration/rehearsal/`](dk-and-rpk-migration/rehearsal/), under names carrying `02`;
the large ones stay under `var/` (§1.9).

---

## 1. R0 — the start tree

**Headline.**
1. **R0 reproduces the reference fingerprint.** Both stores are format 6, digest
   `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes equal, each
   equal to its sidecar's. `compare_fingerprints` is empty between the two stores, and between each
   store and the fingerprints recorded by `a3-v2-readiness-07-reference` (`dbd8a51`) and by
   `datastore-generic-09-rehearsal` (`53d4e18`) (§1.3.4). So the three layer commits no rehearsal
   had run, `fee1fbf`, `4c34f13` and `086c81a` (README §0.1), change nothing a build writes. R1 is to
   be compared with this.
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0; `resumed-a` and
   `resumed-b` are `killed`/143, each with its fingerprint, each driver exiting 143 (§1.2).
3. **The resume recomputed nothing it had stored.** At its open, `resumed-b` pruned the 43 `GkSource`
   rows `resumed-a` left unvalidated, and computed 215 = 291 − 119 + 43. `resumed-c` computed the
   532 = 864 − 332 `QuadSourceIntegral` rows not stored when `resumed-b` was killed, and nothing
   else. Nothing is duplicated (§1.3).
4. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§1.4).
5. **L4's baseline is recorded** (§1.5). R0's `GkSourcePolicyData SUMMARY STATISTICS` block holds the
   same 14 lines as `datastore-generic` 09's, with the same numbers, but **two of its lines are in the
   other order**. The block's order is Ray's completion order, not a property of the store: R0's own
   `resumed-b` prints 09's order. Opened as
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]` (§1.7).
6. **Two findings, neither a stop.**
   - 07's `store_soundness.py`, which README §4.1 names, raises `TypeError` at this tree:
     `read_inventory` has needed its registry as an argument since `19f07ee`. R0 ran a copy that
     passes it (§1.3.5). Opened as `[02-07s-store-soundness-calls-read-inventory-without-its-registry]`.
   - 07's finding `[07-a-killed-builds-ray-agents-outlive-the-driver]` did not recur: no Ray
     process of this repository was listed 44 s after `resumed-a` exited, nor 15 s after
     `resumed-b` (§1.2.3). The issue stays open; it is not this campaign's.

### 1.1 How each was launched

Each launch was README §4.1's command, verbatim but for the store, the slug, the purpose and
`--resume`, run from the repository root, detached:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/dk-and-rpk-migration-02-<store>.sqlite \
    --job-name dk-and-rpk-migration-02 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "<one line>" --campaign dk-and-rpk-migration --prompt 02; \
  echo "** driver exit status $?"' \
  > var/runs/dk-and-rpk-migration-02-<slug>.launch.log 2>&1 &
disown
```

- `reference`: the reference store, no `--resume`.
- `resumed-a`: the resumed store, no `--resume`.
- `resumed-b` and `resumed-c`: the resumed store, with `--resume`.

Each `--purpose` is in its manifest
([`run_records_02.txt`](dk-and-rpk-migration/rehearsal/run_records_02.txt)). After each launch the
agent made one check, that the manifest existed, that `status.json` said `running` with a pid
answering `kill -0`, and that `RunRegistry list` showed it, and ended its turn. The orchestrator ran
one wait per launch, on the run's liveness, and resumed the agent when the run was no longer alive.
Neither polled. Every wait ended `finished`, with the heartbeat never stale.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `dk-and-rpk-migration-02-reference-20261011T143003` | reference | no | 65761 (65752) | `done` | 0 | 0 | 8 m 0 s (14:30:03–14:38:03) | 30.74 35.73 25.37 / 13.66 26.01 27.30 |
| `resumed-a` | `dk-and-rpk-migration-02-resumed-a-20261011T143917` | resumed | no | 71276 (71273) | `killed` | 143 | 143 | 2 m 27 s (14:39:17–14:41:44) | 9.17 23.05 26.14 / 6.82 16.56 22.84 |
| `resumed-b` | `dk-and-rpk-migration-02-resumed-b-20261011T144253` | resumed | yes | 73797 (73794) | `killed` | 143 | 143 | 2 m 8 s (14:42:53–14:45:01) | 6.38 15.81 22.43 / 17.43 17.75 22.19 |
| `resumed-c` | `dk-and-rpk-migration-02-resumed-c-20261011T144549` | resumed | yes | 75882 (75880) | `done` | 0 | 0 | 3 m 0 s (14:45:49–14:48:49) | 15.51 17.26 21.88 / 22.54 22.44 23.09 |

Wall time is from the manifest's `created` to the final heartbeat. The load is `uptime`'s on 10
CPUs, at the pre-launch check and at the agent's first check after the run. The machine was shared,
and the load at the first pre-launch check was 51.96.

**The sidecars** ([`store_sidecars_02.txt`](dk-and-rpk-migration/rehearsal/store_sidecars_02.txt)):
`store create` on the reference's closed primary at 14:38:52, after `reference`, then `store
fingerprint --write` (39809dca…, problems none); `store create` on the resumed store's closed primary
at 14:42:42, after `resumed-a`.

```bash
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" \
    var/datastores/dk-and-rpk-migration-02-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint --write \
    var/datastores/dk-and-rpk-migration-02-reference.sqlite
PYTHONPATH=. ./venv/bin/python -m RunRegistry store create --purpose "<one line>" \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite
```

#### 1.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_02.txt`](dk-and-rpk-migration/rehearsal/prelaunch_checks_02.txt)):
- `git rev-parse HEAD` was `a5392fd1c4ae1c88aa929525ed3f1e6a55c51c82`, and `git status --short` was
  empty. Every working note and capture until the last launch had ended went under
  `var/dk-and-rpk-migration-02/`, which is ignored; this record's files were written afterwards.
- `RAY_ADDRESS` was unset.
- `python -m RunRegistry list` showed nothing `running`: 17 runs before `reference` (15 finished, 2
  `unknown`, the pre-registry `a3-pilot` pair), then 18, 19 and 20, all finished but those two.
- **No Ray process at all.** `pgrep -lf 'gcs_server|raylet|ray::'` was empty, and so was the
  whole-table search, `ps -axo pid,ppid,lstart,command` for lines holding both
  `SecondaryGWKit/venv` and `ray`. The user's ChamPBH cluster was not up.
- Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run naming
  either; nothing under `var/datastores/` or `var/runs/` began `dk-and`. Before `resumed-a`, no
  `dk-and-rpk-migration-02-resumed*` file existed.
- `./venv/bin/pip show datastorekit raypoolkit` found neither.
- 82–84 GB free on `var/`'s volume.

#### 1.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch**, each into a new directory under the agent's
scratchpad, outside `var/` and the repository:

```bash
bash docs/datastore-integrity-verification/interrupter_dummy_test.sh <scratch>/work/dummy-scoped
bash docs/a3-v2-readiness/interrupter_build_dummy_test.sh <scratch>/work/dummy-build
```

Both exited 0, with every verdict as 07's record gives it
([`interrupter_dummy_test_02.txt`](dk-and-rpk-migration/rehearsal/interrupter_dummy_test_02.txt),
[`interrupter_build_dummy_test_02.txt`](dk-and-rpk-migration/rehearsal/interrupter_build_dummy_test_02.txt)):
- **five cases, 14:29:20–14:29:45.** A signalled at 45 % ≥ 40 % (child 143). B exited silently when
  the child died first. C refused `'sleep 30'` with exit 2, naming "`scoped_pipeline_run.py` (the
  run's own driver, its manifest's argv[0])", and the child lived. D held 4 s and signalled (child
  143). E did not signal when the stage completed during the hold.
- **the `--build` cases, 14:29:45.** A signalled the `quadsource_atol_sweep.py --build --register
  dummy-build-a` child (interrupter 0, child 143). B refused the `scoped_pipeline_run.py` child
  (exit 2, child alive). C: "the run's manifest names no argv[0]", exit 2, child alive.
- No `sleep` child was left alive.

**The points, from R0's reference `stdout.log`.** Neither stage printed an intermediate progress
line:
- `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **32.2 s**. `resumed-a` was interrupted from
  the banner with `--interval 5 --hold 10`, to put the signal 10–15 s into the stage (07 held 15 s
  in a stage of 62 s; 09's stage took 24.9 s).
- `CALCULATE QUADRATIC SOURCE INTEGRALS` took **3 m 32.3 s**. 07's trigger, the stage's 100 %
  progress line, would never have fired. `resumed-b` was interrupted from the banner with
  `--interval 5 --hold 70`, about a third of the stage.
  - The pattern is `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, **not anchored at the end**, so it
    matches both the bare banner and a progress line. With `$`, a progress line printed during the
    hold would make the reading after it ineligible, and the interrupter would never signal.
  - Whether a line can come inside 75 s was argued both ways before the launch. `RayWorkPool`
    prints one once more than `notify_min_time_interval` has passed since the last **and** either
    `notify_time_interval` has passed or more than `notify_batch_size` queue items have completed
    (`RayTools/RayWorkPool.py:521-533`). Its defaults are 60 s, 300 s and 500
    (`RayTools/RayWorkPool.py:15-17`), under which a line could come about 60 s in. But `main.py`
    passes this queue `notify_min_time_interval=MIN_NOTIFY_INTERVAL`, which is 300 s, and
    `notify_batch_size=2000` (`main.py:119`, `:4155-4156`), so in this stage no line can come
    before 300 s; 07's came at 5 m 10.2 s. The un-anchored pattern is right either way, and costs
    nothing.
  - It cannot fire between stages. `interrupt_when.progress()` gives `None` for the stage's final
    `…, all work items complete` reading, which is never eligible. This was checked by calling
    `progress()` on the bare banner (0.0), on `…, 100.00% (0 of 2 work items remaining)` (100.0)
    and on `…, all work items complete` (`None`).

**The real firings** ([`interrupter_firings_02.txt`](dk-and-rpk-migration/rehearsal/interrupter_firings_02.txt)):

```
resumed-a (interrupter pid 71349; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10)
2026-10-11T14:41:32+01:00 liveness alive; pid 71276; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-11T14:41:32+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 10 s, then reading again
2026-10-11T14:41:42+01:00 sent SIGTERM to pid 71276 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/dk-and-rpk-migration-02-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 73873; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 70)
2026-10-11T14:43:43+01:00 liveness alive; pid 73797; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-11T14:43:43+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 70 s, then reading again
2026-10-11T14:44:53+01:00 sent SIGTERM to pid 73797 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume … --register resumed-b …'; exiting
```

Each confirmed, before it signalled, that the pid's command line named `quadsource_atol_sweep.py`,
its manifest's `argv[0]`, and `--register <slug>`. The interrupter refused no real run's pid. No
progress line appeared during `resumed-b`'s hold. Both interrupters were launched detached,
right after their builds, once the run directory existed:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/dk-and-rpk-migration-02-resumed-a-20261011T143917 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/dk-and-rpk-migration-02-resumed-b-20261011T144253 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 70 > /dev/null 2>&1 &
disown
```

### 1.2 L1 — every launch ends as recorded

#### 1.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | not written: "…dk-and-rpk-migration-02-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `eaf590af…`, problems `{}` | absent | not written: "…dk-and-rpk-migration-02-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `fc02616e…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint's `taken` names its run and `git_head` `a5392fd1…`, `git_dirty` false. The full
digests: `reference` and `resumed-c`
`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`; `resumed-a`
`eaf590af20c10c9e92af848143754cbdf29ce2fd684d76715b2c3ca34d5ee4ca`; `resumed-b`
`fc02616e9cb3e653e81580ef2f7827da476082a3908ddec06513cb66a83a90b4`.

#### 1.2.2 Each killed launch, against prompt 04's account

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 14:41:42 | 14:44:53 |
| Handler's line | `** SIGTERM at 2026-10-11 14:41:42 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-11 14:44:54 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s (equal whole-second stamps) | under 2 s (stamps 1 s apart) |
| `stderr.log`'s second line | `** SIGTERM: the pipeline ended with SystemExit(143); recording killed, exit code 143` | `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Stage at the signal | the `GkSource` banner, first read 14:41:32, signal 10 s later (119 of 291 rows stored, 43 unvalidated) | the `QuadSourceIntegral` banner, first read 14:43:43, signal 70 s later (332 of 864 stored) |

- **The delay.** Both stamps are whole seconds, the handler's from `time.strftime`, the
  interrupter's from `RunRegistry.now_iso()`. They bound the delay; they do not measure it. In
  `resumed-b` they differ by one second, so the delay was under 2 s, and may have been a few
  milliseconds across a second boundary. Prompt 04 measured 0.001–1.006 s.
- **`SystemExit(1)` in `resumed-b`** is the case 07 recorded: a signal during `ray.wait`, where Ray's
  signal check converts the handler's `SystemExit(143)`. The latch is what `finish_after_exec`
  reads, so the record is `killed`/143 and the exit is 143.
- `stderr.log` of `reference` and of `resumed-c` holds only one `IntegrationWarning` from SciPy's
  `quad` in a `compute_QuadSource_integral` worker ("The occurrence of roundoff error is
  detected…"), and no traceback.

#### 1.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 14:39:11 (~68 s after exit, at the next pre-launch check) | none |
| `resumed-a` | 14:42:28 (~44 s after exit; load 6.82) | none: `pgrep` empty, whole-table search empty, nothing in process group 71276 |
| `resumed-b` | 14:45:16 (~15 s after exit; load 17.43) | none, by the same three checks for process group 73797 |
| `resumed-c` | 14:49:35 (~46 s after exit) | none |

07's orphaned dashboard and runtime-env agents
(`[07-a-killed-builds-ray-agents-outlive-the-driver]`) did not appear after either kill, so there
was nothing to wait for by pid, and no wait was run. 07 saw them at loads of 65–123; here the loads
were 7–17. That is consistent with 07's inference, that the agents exit when a periodic check finds
their raylet gone, but it does not establish it. The issue is not this campaign's, and stays open.

#### 1.2.4 The manifests

From [`run_records_02.txt`](dk-and-rpk-migration/rehearsal/run_records_02.txt), for all four:
- `git_head` `a5392fd1c4ae1c88aa929525ed3f1e6a55c51c82`, the dispatch `HEAD`; `git_dirty` false;
- **`packages` `{"datastorekit": null, "raypoolkit": null}`** (prompt 01's field);
- `script` `docs/handover/quadsource_atol_sweep.py`; `script_sha256`
  `d745783e65bfff99b9c704117f3fd63527f8b70b9a1a450a053b718ad120e9eb`, the same in all four (07's was
  `3d0e6ecd…`; the script has changed since `dbd8a51`);
- `argv[0]` `docs/handover/quadsource_atol_sweep.py`; `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only;
- `results` names the reference store for `reference`, the resumed store for the other three;
- `scope`, in each, the rehearsal text, naming the pair, the grid and `zend 250`, and not `9280`:
  `REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32
  rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into
  dk-and-rpk-migration-02-<store>.sqlite`.

#### 1.2.5 Every launch's `stdout.log`

Counted in each of the four:

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--zend', '250', …]` | 1 |
| `--no-prune-unvalidated`, in `stdout.log` or the launch log | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |
| `reconcil`, `repair`, `in.flight`, `replicat` (case-insensitive), in `stdout.log` or the launch log | 0 |

The launch logs of `resumed-b` and `resumed-c` begin `** quadsource_atol_sweep --build: resuming
into dk-and-rpk-migration-02-resumed.sqlite`.

#### 1.2.6 A3 v2's guard

After `resumed-c`, in the foreground, with no `--register`
([`guard_02.txt`](dk-and-rpk-migration/rehearsal/guard_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/dk-and-rpk-migration-02-resumed.sqlite \
    --job-name dk-and-rpk-migration-02 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (14:52:26–14:52:27), with:

```
RuntimeError: quadsource_atol_sweep --build: dk-and-rpk-migration-02-resumed.sqlite, dk-and-rpk-migration-02-resumed-shard0000.sqlite, dk-and-rpk-migration-02-resumed-shard0001.sqlite, dk-and-rpk-migration-02-resumed-shard0002.sqlite, dk-and-rpk-migration-02-resumed-shard0003.sqlite already exist(s). Refusing to build into an existing store: a comparator that silently inherited earlier rows is one nobody can interpret. Pass --resume to continue an interrupted build at the SAME tolerance pair, or choose a path that does not exist.
```

It names the five files that exist. Every file of both stores, and both sidecars, had the same
SHA-256 afterwards as at 14:49, after `resumed-c`; no run directory was created; no Ray process of
this repository appeared.

### 1.3 L2 — the resume is the reference

#### 1.3.1 Nothing stored is recomputed

Through S1's [`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py)
([`stage_counts_02.txt`](dk-and-rpk-migration/rehearsal/stage_counts_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/stage_counts.py \
    var/runs/dk-and-rpk-migration-02-reference-20261011T143003 \
    var/runs/dk-and-rpk-migration-02-resumed-a-20261011T143917 \
    var/runs/dk-and-rpk-migration-02-resumed-b-20261011T144253 \
    var/runs/dk-and-rpk-migration-02-resumed-c-20261011T144549
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
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **215/215/215** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **532/532/532** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.** `resumed-a` stored 119 rows (39 + 34 + 0 + 46 by shard) before the signal, 43 of
  them unvalidated (§1.3.2). `resumed-b` pruned the 43 at open and computed 215 = 291 − 119 + 43:
  what was missing, plus the pruned rows. No validated row was computed again. Summed, less the
  pruned: 119 + 215 − 43 = 291, the reference's.
- **`QuadSourceIntegral`.** `resumed-b` printed no progress line in the stage, so the counts in
  flight at the signal are not in its log. The closed store held **332** rows (106 + 102 + 0 + 124 by
  shard), read `mode=ro`. `resumed-c` computed **532** = 864 − 332, exactly what was missing. Summed:
  332 + 532 = 864, the reference's. No stored row was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b` because `resumed-a` never reached it; it
  is 0/0/0 in `resumed-c`.

#### 1.3.2 The prune at startup (prompt 05)

The closed store after each launch, read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`, each file's SHA-256 unchanged by the read
([`closed_store_state_02.txt`](dk-and-rpk-migration/rehearsal/closed_store_state_02.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **43 `GkSource`**: shard 0 serials 214, 215, 219, 224, 227, 234, 235, 236, 237, 238, 239; shard 1 524, 532, 535, 537, 538, 541, 542, 543, 544, 545; shard 3 19, 21, 23, 24, 26, 27, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46. Every other class with a `validated` column: none |
| `resumed-b` | none | 0 rows | none |
| `resumed-c` | none | 0 rows | none |

**By serial**, through [`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py), which reads
the serials from the saved `closed_store_state.py` output rather than naming them
([`prune_check_02.txt`](dk-and-rpk-migration/rehearsal/prune_check_02.txt)):

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/dk-and-rpk-migration-02/state/after-resumed-a.txt \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/dk-and-rpk-migration-02/state/after-resumed-b.txt \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite
```

After `resumed-c`: **all 43 serials absent**, none reused by a later row (07 saw three reused), and
0 rows `validated = 0` on any shard. The same was read with `sqlite3` `mode=ro` straight after
`resumed-b`. After `resumed-b` no serial was unvalidated, so the second run checks nothing. Per
shard, `GkSource`'s 97 rows hold 97 distinct keys `(wavenumber_exit_serial, model_serial,
z_response_serial, parents_digest)`.

**What `resumed-b` printed.** Each actor's integrity warning, at open (14:42:58):
- shard 0 in full: `!! INTEGRITY WARNING … datastore "shard0000-store" …`, `The following
  unvalidated GkSource objects were detected in the datastore:`, eleven entries, store_ids 214, 215,
  219, 224, 227, 234–239, and `** Note: these values have been pruned from the datastore.`;
- shards 1 and 3 folded by Ray: `!! INTEGRITY WARNING … "shard0003-store" … [repeated 2x across
  cluster]`, one entry shown with `[repeated 32x across cluster]`, and the note `[repeated 2x across
  cluster]`. 11 + 32 = 43, the count from the store.

`resumed-c` printed no integrity warning; nothing was unvalidated. `QuadSourceIntegral` has no
`validated` column, so the rows `resumed-b` stored need no prune.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§1.3.5), so no `duplicate`. Each finish's fingerprint is built from `read_inventory`, and its
`problems` is `{}` for `resumed-a`, `resumed-b` and `resumed-c`.

**The pool's prune of a replicated class.** None. No launch printed a prune of a replicated class,
and after every launch each shard's `BackgroundModel` row is validated and the census finds the
replicated tables equal (§1.3.5).

#### 1.3.3 The version row (prompt 02 of `a3-v2-readiness`)

`SELECT serial, label FROM version`, `mode=ro`, on every shard, after every launch: exactly one row,
`(1, '2025.1.1')`, on each of the four shards of both stores. The primary has no `version` table.

#### 1.3.4 The comparison

**The two stores**, by `compare_stores.py`
([`fingerprint_comparison_02.txt`](dk-and-rpk-migration/rehearsal/fingerprint_comparison_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \
    var/datastores/dk-and-rpk-migration-02-reference.sqlite \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite
```

Exit 0:

```
reference: …/var/datastores/dk-and-rpk-migration-02-reference.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches
resumed: …/var/datastores/dk-and-rpk-migration-02-resumed.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches

compare_fingerprints(reference, resumed): 0 difference(s)
verdict: equal
```

**Against 07's reference, and 09's rehearsal.** 07's stores are retired; its run directory is not.
Each store was fingerprinted read-only again and compared with the fingerprint recorded in each
earlier run's `status.json`, by [`compare_with_recorded.py`](dk-and-rpk-migration/rehearsal/compare_with_recorded.py)
([`fingerprint_against_recorded_02.txt`](dk-and-rpk-migration/rehearsal/fingerprint_against_recorded_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/dk-and-rpk-migration-02-reference.sqlite \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite \
    --against var/runs/a3-v2-readiness-07-reference-20261004T012909 \
    var/runs/datastore-generic-09-rehearsal-20261006T152452
```

Exit 0, verdict `equal`: for each store, 0 differences against its sidecar, against 07's reference
(taken at `dbd8a511…`) and against 09's rehearsal (taken at `53d4e188…`). All are format 6, digest
`39809dca…`, 21 classes, problems none.

| Class | Reference | Resumed | Digests | 07's reference |
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

**The census** (`docs/datastore-integrity-audit/replicated_serial_census.py`, both primaries;
[`census_02.txt`](dk-and-rpk-migration/rehearsal/census_02.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py \
    var/datastores/dk-and-rpk-migration-02-reference.sqlite \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite
```

Every table `OK` on both, verdict `no divergence` on both. Its table list is `config/sharding.py`'s
13 `replicated_tables`, in the same order, plus `BackgroundModel_tags`, which it adds; it omits
none. Each table has equal rows on the four shards: `version` 1, `store_tag` 10, `redshift` 1159,
`wavenumber` 3, `wavenumber_exit_time` 3, `tolerance` 7, `LambdaCDM` 1, `QCD_Cosmology` 1,
`IntegrationSolver` 7, `BackgroundModel` 1, `BackgroundModelValue` 1159, `GkSourcePolicy` 2,
`QuadSourcePolicy` 2, `BackgroundModel_tags` 4.

**`integrity_check`, `foreign_key_check` and the inventory**
([`store_soundness_02.txt`](dk-and-rpk-migration/rehearsal/store_soundness_02.txt)).
- **07's script fails at this tree.** `docs/a3-v2-readiness/rehearsal/store_soundness.py`, run on
  both primaries, printed the pragmas of the reference store, then raised
  `TypeError: read_inventory() missing 1 required positional argument: 'factories'`. Since `19f07ee`
  ("Give the datastore layer its registries instead of importing them"), `read_inventory` takes the
  registry of storable classes as an argument; 07's script, a record of `dbd8a51`, calls it with the
  primary alone. Its output is kept at
  `var/dk-and-rpk-migration-02/checks/store_soundness_07_original_traceback.txt`. The script was not
  changed (it is 07's record). Opened as
  `[02-07s-store-soundness-calls-read-inventory-without-its-registry]` (§1.7).
- **The copy.** [`store_soundness.py`](dk-and-rpk-migration/rehearsal/store_soundness.py) is 07's
  with two changes: it passes `config.datastore.factories`, exactly as
  `RunRegistry.stores.fingerprint_store` does, and it reads the shards from the primary's `shards`
  table instead of a glob.

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py \
    var/datastores/dk-and-rpk-migration-02-reference.sqlite \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite
```

Exit 0, `=> clean`. On the primary and every shard of both stores, opened `mode=ro`, `PRAGMA
integrity_check` is `['ok']` and `PRAGMA foreign_key_check` returns 0 rows. `read_inventory(primary,
factories)` reads shards `(0, 1, 2, 3)` and reports no problem in any class of either store, and 0
`duplicate`; each class's record count is §1.3.4's table, identical in the two. Every file of both
stores had the same SHA-256 before and after these reads, and no journal appeared.

### 1.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance,
through S1's [`run_readers.sh`](dk-and-rpk-migration/rehearsal/run_readers.sh)
([`readers_02.txt`](dk-and-rpk-migration/rehearsal/readers_02.txt)):

```bash
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite var/dk-and-rpk-migration-02/extract
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/dk-and-rpk-migration-02-resumed.sqlite var/dk-and-rpk-migration-02/extract \
    nomodels extract_Gk_data
```

which runs, for each reader,

```bash
PYTHONPATH=. ./venv/bin/python <script>.py --database var/datastores/dk-and-rpk-migration-02-resumed.sqlite \
    --ray-address local --models LambdaCDM --output var/dk-and-rpk-migration-02/extract/<script>/ \
    > var/dk-and-rpk-migration-02/extract/<script>.log 2>&1
```

and, in the second form, the same without `--models`. Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` / traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 22 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 324 | none | identical |
| `extract_GkWKB_data.py` | 0 | 57 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 20 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 144 | none | identical |
| `extract_TkWKB_data.py` | 0 | 11 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | **0** | 11 s | `EXTRACT TENSOR SOURCE DATA`: 6, all six `QuadSource` rows | 27 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 17 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 (6 `(k, q, r)`, 4 tasks each) | 74 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 25 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 423 | none | identical |

- **The model and the run.** Every run printed `** reading run "default", the only run in this
  datastore (no --run-label given)`, from `resolve_run_selection`. With `--models LambdaCDM` each
  printed `>> RUNNING PIPELINE FOR MODEL LambdaCDM` and nothing about `QCD_Cosmology`. Without
  `--models`, `extract_Gk_data.py` read LambdaCDM, then printed `>> RUNNING PIPELINE FOR MODEL
  QCD_Cosmology` and the skip line, and exited 0.
- **`extract_tensor_source_data.py` exits 0**, as expected since 07a fixed its random pairing; 07's
  first pass exited 1 there.
- **`extract_QuadSourceIntegral_data.py`** draws 6 `(k, q, r)` triples at random by design (07
  §2.3.3), so it does not read all 864 rows. Every lookup it made was found.
- **No file of the store changed.** Before and after each reader, the SHA-256, the mtime and the
  size of the primary and of every shard were identical, and no `-journal`, `-wal` or `-shm` file
  appeared. The SHA-256s are those taken after `resumed-c`: `bf22863f…` (primary), `e1340524…`,
  `35908f3a…`, `fafbe696…`, `09b8aef1…` (shards 0–3). The mtimes (14:45:54 for the primary,
  14:48:41–14:48:46 for shards 0, 1 and 3, 14:39:24 for shard 2) all predate the first reader
  (14:53:25). No Ray process of this checkout was left after any reader.
- Each log opens `!! WARNING: number of shards read from database (=4) does not match specified
  number of shards (=10)`, the readers' default `--shards`; 07's logs hold the same line.
- Afterwards `store fingerprint` (without `--write`) on the resumed store gave `39809dca…`,
  problems none, "matches the fingerprint taken 2026-10-11T14:48:49+01:00 by run
  dk-and-rpk-migration-02-resumed-c-20261011T144549".

### 1.5 L4 — the campaign's baseline

R0 has no earlier rehearsal of this campaign to equal. These are what R1 (05) and R2 (08) compare
with, produced by S1's scripts.

#### 1.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/policy_statistics.py \
    var/runs/dk-and-rpk-migration-02-reference-20261011T143003
```

Exit 0. The reference run's block, lines 83–96 of its `stdout.log`, verbatim
([`policy_statistics_02_reference.txt`](dk-and-rpk-migration/rehearsal/policy_statistics_02_reference.txt)):

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

The script ends the block before the first later line that, after leading whitespace, begins `** `
or `@@ `. Here that is line 97, `   @@ QuadSourceIntegral triangle filter: kept 864 of 1728 (k,q,r)
triples (50.00%)`, then the `** CALCULATE QUADRATIC SOURCE INTEGRALS` banner.

#### 1.5.2 Against `datastore-generic`'s 09 rehearsal — same lines, two in the other order

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/policy_statistics.py \
    var/runs/datastore-generic-09-rehearsal-20261006T152452
```

Exit 0, 14 lines. **It is not R0's line for line.** `diff` of R0's block against 09's:

```
10d9
<            minimal: 1 instances = 3.85% of this type, 0.34% of total
11a11
>            minimal: 1 instances = 3.85% of this type, 0.34% of total
```

Under `>> mixed`, R0 prints `minimal` then `complete`; 09 prints `complete` then `minimal`. Every
line is otherwise identical, and the two blocks sorted are equal. 07's reference block is 09's,
byte for byte.

**Why.** It is not a property of the store, and not of the tree.
- `postprocess_GkSourcePolicy` (`main.py:2871`) is the `APPLY GKSOURCE POLICIES` queue's
  `post_handler` (`main.py:3253`). It fills `GkSourcePolicy_statistics[policy]["statistics"][type]
  [quality]`, plain dictionaries, inserting each key the first time an object of that type or
  quality arrives (`:2892`).
- The block iterates those dictionaries (`:3270`, `:3275`), so it prints in insertion order, and
  `RayWorkPool` calls the `post_handler` as `ray.wait` returns completed work, in completion order.
- So the order of `minimal` and `complete` under `mixed` is whichever mixed object completed first.
  The order of the three types could move the same way, though it has not been seen to.
- **Within R0, on one tree,** the order differs: `resumed-b`, which recomputed all 291 policy rows,
  prints 09's order (`complete` first); `resumed-c` prints the reference's.

**What it means.** The counts are the baseline; the order is noise. 08's acceptance 3 reads "R1's,
line for line", and README §0.2 says that comparison measures `main.py:3247` (C8). A line-for-line
comparison can fail on order alone, at any of R1, R2, or their resumed runs. Opened as
`[02-the-policy-statistics-block-orders-its-lines-by-completion]` (§1.7). The scripts are unchanged
by it: `policy_statistics.py` prints the block verbatim, as specified, and a comparison can sort it.

#### 1.5.3 The stage counts

§1.3.1's table, from [`stage_counts_02.txt`](dk-and-rpk-migration/rehearsal/stage_counts_02.txt).

#### 1.5.4 What each resumed launch pruned and recomputed

| Launch | Pruned at open | Computed, stage by stage | Of which recomputing a pruned row |
|---|---|---|---|
| `resumed-a` | nothing (new store) | everything up to the `GkSource` stage, as `reference`; 119 `GkSource` stored (43 unvalidated) at the signal | — |
| `resumed-b` | **43 `GkSource`** (11 + 10 + 22, shards 0, 1, 3) | `GkSource` 215; `GkSourcePolicyData` 291; `QuadSourceIntegral` 332 stored at the signal; every earlier stage 0 | 43 `GkSource` |
| `resumed-c` | nothing | `QuadSourceIntegral` 532; every other stage 0 | — |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 1.5.5 The stage times

Not part of L4's comparison, but what R1 and R2 will choose their interruption points from
(`ALL WORK ITEMS COMPLETE in time …` in each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1 m 34.7 s | 45 s | 1.53 s | 1.49 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 1 m 45.4 s | 1 m 19.9 s | 3.38 s | 3.26 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **32.2 s**, no progress line | killed | 22.3 s | 5.42 s |
| APPLY GKSOURCE POLICIES | 11.1 s | — | 13.5 s | 5.09 s |
| QUADRATIC SOURCE INTEGRALS | **3 m 32.3 s**, no progress line | — | killed, no progress line | 2 m 36.9 s |

The earlier stages each took under 5 s.

### 1.6 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen. After every launch each shard's
  `BackgroundModel` row is validated (§1.3.2), and no resumed launch printed a prune of it.
- **An in-flight record left by a killed launch.** Did not happen. `replication_in_flight` had 0 rows
  after every launch, the two killed ones included.
- **A repair by the check at open.** Did not happen. No `stdout.log` or launch log holds `reconcil`,
  `repair`, `in.flight` or `replicat` (§1.2.5).
- **A Ray agent outliving a killed driver** (07's finding). Did not happen (§1.2.3).

### 1.7 Issues opened

On the board's §3 and in `docs/OPEN_ISSUES.md` §1.20:
- `[02-the-policy-statistics-block-orders-its-lines-by-completion]` (§1.5.2).
- `[02-07s-store-soundness-calls-read-inventory-without-its-registry]` (§1.3.5).

### 1.8 The suites

Serially, in the foreground, after the last launch, with README §7's command; Datastore in four
groups. Every count is the dispatch baseline.

| Root | Ran | Result | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 | OK | 26.7 |
| CosmologyModels | 39 | OK | 26.7 |
| LiouvilleGreen | 148 | OK (1 skipped) | 26.7 |
| RunRegistry | 286 | OK | 20.5 |
| ComputeTargets | 693 | OK | 20.3 |
| Datastore `[a-f]` / `[g-o]` / `[p-r]` / `[s-z]` | 163 / 185 / 200 / 191 = 739 | OK | 6.4 / 5.9 / 12.0 / 8.8 |

### 1.9 What stays under `var/`

- `var/runs/dk-and-rpk-migration-02-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`: each run's
  manifest, `status.json` with its full fingerprint, `stdout.log`, `stderr.log`, and, for
  `resumed-a` and `resumed-b`, `interrupter.log`.
- `var/runs/dk-and-rpk-migration-02-<slug>.launch.log`, beside each run directory.
- `var/datastores/dk-and-rpk-migration-02-{reference,resumed}.sqlite`, four shards each (shard 2 of
  each, 1 077 248 bytes, holds only replicated rows), and each store's `.manifest.json` sidecar.
- `var/dk-and-rpk-migration-02/`: the agent's working notes (`notes.txt`), the pre-launch captures
  and dummy-test outputs (`prelaunch/`), the closed-store states and hashes (`state/`), the check
  outputs (`checks/`, among them 07's `store_soundness.py` traceback and the other runs' statistics
  blocks), and the readers' outputs and logs (`extract/`, `extract-summary-*.txt`).
