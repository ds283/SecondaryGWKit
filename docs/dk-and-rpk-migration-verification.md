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

---

## 2. R1 — through `datastorekit`

**§2's tree, stores and runs.** Every R1 launch ran on branch `handover-remedial` at **`6861a9f`**
("Write the orchestrator note for dk-and-rpk-migration prompt 05"), whose code is `0c9e2cf`'s
(prompt 04). Its datastore layer is the installed package **`datastorekit` 0.2.2**, at commit
`6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`, not editable. Its work queue is still the frozen
`RayTools/`. `git status --short` was empty at every launch. Every manifest stamps `git_dirty: false`
and `packages`
`{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false, "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version": "0.2.2"}, "raypoolkit": null}`.
The same launcher as at R0 built two throwaway stores:
- **`var/datastores/dk-and-rpk-migration-05-reference.sqlite`** (sidecar `store_id`
  `ce780c5daaf44e1286e23bd63d273be0`), built straight through by the run **`reference`**;
- **`var/datastores/dk-and-rpk-migration-05-resumed.sqlite`** (`store_id`
  `8ea1f07ab83b43d8a0c5d8a990aed895`), built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

The runs are `var/runs/dk-and-rpk-migration-05-<slug>-<stamp>/`. Both stores stay, with their
sidecars (README §3). The small outputs quoted below are in
[`dk-and-rpk-migration/rehearsal/`](dk-and-rpk-migration/rehearsal/), under names carrying `05`.
The large ones stay under `var/` (§2.9).

R0's stores were not opened. R1 is compared with R0 through the fingerprint R0's `reference` run
recorded (§2.3.4). R0's twelve files were hashed before R1's first launch and after its last step.

**Headline.**
1. **R1 reproduces the reference fingerprint through the package.** Both stores are format 6,
   digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes
   equal, and each equals its sidecar's. `compare_fingerprints` is empty between the two stores. It
   is also empty between each store and the fingerprints recorded by
   `dk-and-rpk-migration-02-reference` (R0, `a5392fd`), `a3-v2-readiness-07-reference` (`dbd8a51`)
   and `datastore-generic-09-rehearsal` (`53d4e18`) (§2.3.4). The only change to what runs between
   R0 and R1 is the package swap. So `datastorekit` 0.2.2 writes what SGK's own layer wrote.
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0. `resumed-a` and
   `resumed-b` are `killed`/143, each with its fingerprint, and each driver exited 143 (§2.2).
3. **The resume recomputed nothing it had stored.**
   - At its open, `resumed-b` pruned the 45 `GkSource` rows `resumed-a` left unvalidated, and
     computed 235 = 291 − 101 + 45.
   - `resumed-c` computed the 530 = 864 − 334 `QuadSourceIntegral` rows not stored when `resumed-b`
     was killed, and nothing else.
   - Nothing is duplicated (§2.3).
4. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§2.4).
5. **L4 equals R0's** (§2.5):
   - the statistics counts are equal;
   - the reference's stage counts are equal, stage by stage;
   - the resumed chain's identities hold;
   - the reference's `** ` lines are equal, in order.

   The statistics block holds R0's 14 lines, with `complete` and `minimal` under `mixed` in the
   other order, which is 09's order. That is
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]` again, not a new finding.
6. **The logs differ from R0's only where the interruption points put them** (§2.2.5). No R1 log
   holds a traceback, a warning R0 did not print, or a line from the package's new close
   behaviour.
7. **DatastoreKit's G2 holds** (§2.10).
8. **R0 was not touched.** Its twelve files have the same SHA-256 and mtime before R1's first launch
   and after its last step.

### 2.1 How each was launched

Each launch was README §4.1's command, verbatim but for the store, the slug, the purpose and
`--resume`. Each was run from the repository root, detached:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/dk-and-rpk-migration-05-<store>.sqlite \
    --job-name dk-and-rpk-migration-05 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "<one line>" --campaign dk-and-rpk-migration --prompt 05; \
  echo "** driver exit status $?"' \
  > var/runs/dk-and-rpk-migration-05-<slug>.launch.log 2>&1 &
disown
```

- `reference`: the reference store, no `--resume`.
- `resumed-a`: the resumed store, no `--resume`.
- `resumed-b` and `resumed-c`: the resumed store, with `--resume`.

Each `--purpose` names R1, the campaign, the package and its commit, and the fingerprint it must
equal ([`run_records_05.txt`](dk-and-rpk-migration/rehearsal/run_records_05.txt)).

**One check per launch, one wait per launch.** After each launch the agent made one check:
- the manifest existed;
- `status.json` said `running`, with a pid answering `kill -0`;
- `RunRegistry list` showed the run.

The agent then ended its turn. The orchestrator ran one wait per launch, on the run's liveness, and
resumed the agent when the run was no longer alive. Neither polled. Every wait ended
`ENDED finished`, with the heartbeat never stale.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `dk-and-rpk-migration-05-reference-20261011T192802` | reference | no | 83632 (83628) | `done` | 0 | 0 | 12 m 13 s (19:28:02–19:40:15) | 13.84 29.00 32.74 / 38.56 42.53 43.83 |
| `resumed-a` | `dk-and-rpk-migration-05-resumed-a-20261011T194149` | resumed | no | 92759 (92751) | `killed` | 143 | 143 | 4 m 23 s (19:41:49–19:46:12) | 23.81 38.25 42.20 / 23.93 46.87 47.11 |
| `resumed-b` | `dk-and-rpk-migration-05-resumed-b-20261011T194703` | resumed | yes | 96222 (96219) | `killed` | 143 | 143 | 2 m 51 s (19:47:03–19:49:54) | 21.18 45.52 46.62 / 35.93 44.64 45.99 |
| `resumed-c` | `dk-and-rpk-migration-05-resumed-c-20261011T195044` | resumed | yes | 98713 (98710) | `done` | 0 | 0 | 4 m 13 s (19:50:44–19:54:57) | 31.43 43.39 45.53 / 13.79 30.05 38.99 |

- **Wall time** is from the manifest's `created` to the final heartbeat.
- **Load** is `uptime`'s on 10 CPUs: at the pre-launch check, and at the agent's first check after
  the run. The machine was shared. At `reference`'s launch, 90 s after its pre-launch check, the
  load was 77.69 48.35 39.94. Over the four builds it ran from 14 to 78, against R0's 6 to 36, so
  R1's stages are slower than R0's (§2.5.5).

**The sidecars** ([`store_sidecars_05.txt`](dk-and-rpk-migration/rehearsal/store_sidecars_05.txt)):
- the reference store: `store create` on its closed primary at 19:41:00, after `reference`, then
  `store fingerprint --write` (`39809dca…`, problems none, "it replaced none");
- the resumed store: `store create` on its closed primary at 19:46:56, after `resumed-a`.

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`. The commands are §1.1's,
with `-05-` for `-02-`.

#### 2.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_05.txt`](dk-and-rpk-migration/rehearsal/prelaunch_checks_05.txt)):
- **The tree.** `git rev-parse HEAD` was `6861a9fc0eb8cd60bb25cda8529ab45a51e50632`.
  `git status --short` was empty, and was checked again in the same shell command that launched.
  Every working note and capture, until the last launch had ended, went under
  `var/dk-and-rpk-migration-05/`, which is ignored. This record's files were written afterwards.
- `RAY_ADDRESS` was unset.
- **The registry.** `python -m RunRegistry list` showed nothing `running`: 21 runs before
  `reference` (19 finished, 2 `unknown`, the pre-registry `a3-pilot` pair), then 22, 23 and 24, all
  finished but those two.
- **No Ray process at all.** `pgrep -lf 'gcs_server|raylet|ray::'` was empty before each launch.
  So was the whole-table search, `ps -axo pid,ppid,lstart,command` for lines holding both
  `SecondaryGWKit/venv` and `ray`.
- **The store names.**
  - Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run
    naming either.
  - Nothing under `var/datastores/` or `var/runs/` contained `dk-and-rpk-migration-05`, and there
    was no journal file under `var/datastores/`.
  - Before `resumed-a`, no `dk-and-rpk-migration-05-resumed*` file existed.
- **The package.**
  - `./venv/bin/pip show datastorekit` gave 0.2.2, `Location` this venv's `site-packages`.
  - `pip show raypoolkit` found nothing, and `pip check` was clean.
  - `RunRegistry.package_provenance()` gave the record above.
  - `requirements.txt:17` is `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2`.
- **What runs is the package.**
  - In a fresh `python -I` with the repository on `sys.path`, each of the 15 layer modules 04
    deleted raises `ModuleNotFoundError` (15 of 15). They are `Datastore.{object, contract,
    replication, shard_paths, store_reader, store_inventory}`, `Datastore.SQL.{schema, ShardedPool,
    Datastore, ClientPool, SerialPoolBroker, ProfileAgent}`, `Datastore.SQL.ObjectFactories.base`
    and `tools.{sharded_store, shard_key_audit}`.
  - `Datastore/__init__.py` and `Datastore/SQL/__init__.py` are 0 bytes.
  - `datastorekit.__file__`, `datastorekit.SQL.ShardedPool.__file__` and
    `datastorekit.store_inventory.__file__` are under
    `venv/lib/python3.12/site-packages/datastorekit/`.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` was `b7935877…` (§2.2.4).
- **R0's twelve files.** Their full SHA-256, mtime and size were recorded before `reference`. They
  equal the orchestrator's table (§2.8).
- 73–75 GB free on `var/`'s volume.

#### 2.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch.** Each ran into a new directory under the
agent's scratchpad, outside `var/` and the repository, with §1.1.2's commands. Both exited 0, with
every verdict as R0's
([`interrupter_dummy_test_05.txt`](dk-and-rpk-migration/rehearsal/interrupter_dummy_test_05.txt),
[`interrupter_build_dummy_test_05.txt`](dk-and-rpk-migration/rehearsal/interrupter_build_dummy_test_05.txt)):
- **The five cases, 19:27:20–19:27:44.**
  - A signalled at 45 % ≥ 40 % (child 143).
  - B exited silently when the child died first.
  - C refused `'sleep 30'` with exit 2, naming "`scoped_pipeline_run.py` (the run's own driver,
    its manifest's argv[0])", and the child lived.
  - D held 4 s and signalled (child 143).
  - E did not signal when the stage completed during the hold.
- **The `--build` cases, 19:27:45.**
  - A signalled the `quadsource_atol_sweep.py --build --register dummy-build-a` child
    (interrupter 0, child 143).
  - B refused the `scoped_pipeline_run.py` child (exit 2, child alive).
  - C: "the run's manifest names no argv[0]", exit 2, child alive.
- No dummy child was left alive (`ps -p` on all eight child pids found none).

**The points, from R1's own reference `stdout.log`.** Neither interrupted stage printed an
intermediate progress line. Both queues have a 300 s minimum notify interval (`main.py:119`,
`:4155-4156`).
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **55.1 s** (R0: 32.2 s).
  It was interrupted from the banner with `--interval 5 --hold 10`, R0's arguments.
  - That puts the signal 10–15 s into the stage.
  - It is well before the stage's end, even if the stage ran at R0's speed.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **4 m 49.7 s** (289.7 s; R0:
  3 m 32.3 s). It was interrupted from the banner with `--interval 5 --hold 95`.
  - That is about a third of the stage, and before its end even at R0's 212.3 s.
  - 95 s is under the queue's 300 s minimum, so no progress line can come during the hold.
  - The pattern is `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its end, as R0's was
    (§1.1.2 gives the reasons).

**The real firings** ([`interrupter_firings_05.txt`](dk-and-rpk-migration/rehearsal/interrupter_firings_05.txt)):

```
resumed-a (interrupter pid 92862; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10)
2026-10-11T19:46:00+01:00 liveness alive; pid 92759; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-11T19:46:00+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 10 s, then reading again
2026-10-11T19:46:10+01:00 sent SIGTERM to pid 92759 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/dk-and-rpk-migration-05-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 96313; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 95)
2026-10-11T19:48:10+01:00 liveness alive; pid 96222; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-11T19:48:10+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 95 s, then reading again
2026-10-11T19:49:45+01:00 sent SIGTERM to pid 96222 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/dk-and-rpk-migration-05-resumed.sqlite … --register resumed-b …'; exiting
```

- Each interrupter confirmed, before it signalled, that the pid's command line named
  `quadsource_atol_sweep.py`, its manifest's `argv[0]`, and `--register <slug>`.
- It refused no real run's pid, and no progress line appeared during `resumed-b`'s hold.
- In `resumed-b` the reading before the banner (19:47:55) was `APPLY GKSOURCE POLICIES`. So the
  stage began between 19:48:05 and 19:48:10, and the signal came 95–100 s into it.

Both interrupters were launched detached, right after their builds, once the run directory existed:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/dk-and-rpk-migration-05-resumed-a-20261011T194149 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/dk-and-rpk-migration-05-resumed-b-20261011T194703 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 95 > /dev/null 2>&1 &
disown
```

### 2.2 L1 — every launch ends as recorded

#### 2.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | not written: "…dk-and-rpk-migration-05-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `b6b89d84…`, problems `{}` | absent | not written: "…dk-and-rpk-migration-05-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `09170e40…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint holds 21 classes, and its `taken` names its run, `git_head` `6861a9fc…` and
`git_dirty` false.

The full digests:
- `reference` and `resumed-c`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `b6b89d84c5dafd9ced8964f74a74eead764d8112c31591137092e9058eb1ce8a`;
- `resumed-b`: `09170e40ffbc01f9662267a59e83e3df989093b16e83b61566de441370166419`.

The two intermediate digests differ from R0's (`eaf590af…`, `fc02616e…`), as they must: each
digests a store killed at a different point.

#### 2.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 19:46:10 | 19:49:45 |
| Handler's line | `** SIGTERM at 2026-10-11 19:46:10 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-11 19:49:45 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s (equal whole-second stamps) | under 1 s (equal whole-second stamps) |
| `stderr.log`'s handler lines | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Stage at the signal | the `GkSource` banner, first read 19:46:00, signal 10 s later (101 of 291 rows stored, 45 unvalidated) | the `QuadSourceIntegral` banner, first read 19:48:10, signal 95 s later (334 of 864 stored) |

- **`SystemExit(1)` in both.** At R0, `resumed-a` printed `SystemExit(143)` and `resumed-b`
  `SystemExit(1)`. `SystemExit(1)` is the case 07 recorded: the signal lands during `ray.wait`,
  whose signal check converts the handler's `SystemExit(143)`. Which one a run prints depends on
  where in the driver's loop the signal lands, not on the layer. The latch is what
  `finish_after_exec` reads, so the record is `killed`/143 and the exit is 143 either way.
- **`resumed-b`'s `stderr.log` also holds the SciPy `IntegrationWarning`** (worker pid 96405), ahead
  of the handler's lines. At R0 it was in `resumed-c` (§2.2.5).
- **`stderr.log` of `reference`** holds only the one `IntegrationWarning` from SciPy's `quad` in a
  `compute_QuadSource_integral` worker ("The occurrence of roundoff error is detected…"), as at R0.
  **`resumed-c`'s `stderr.log` is empty.**

#### 2.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 19:40:53 (~38 s after the final heartbeat; load 38.56) | none: `pgrep` empty, whole-table search empty, pids 83628 and 83632 gone, nothing in process group 83632 |
| `resumed-a` | 19:46:45 (~33 s; load 23.93) | none, by the same checks for pids 92751, 92759, 92862 and process group 92759 |
| `resumed-b` | 19:50:24 (~30 s; load 35.93) | none, by the same checks for pids 96219, 96222, 96313 and process group 96222 |
| `resumed-c` | 19:55:37 (~40 s; load 13.79) | none, for pids 98710 and 98713 |

- **No Ray process outlived either killed driver,** so no wait by pid was run, and the agents' exit
  time is "none listed at the first check". The same was true at R0.
- **What `v0.2.2`'s close changes.** `ShardedPool._kill_actors()` calls `ray.kill` on each shard
  actor and the broker at close. R0's actors died with the driver's Ray session instead.
  - Here the end state is the same as R0's: nothing left.
  - No launch log or `stderr.log` holds a `ray.kill`, killed-actor, `ActorDiedError` or
    `RayActorError` line (§2.2.5).
  - So the new close leaves no trace R1 can see. Whether it shortens the life of an orphan,
    `[07-a-killed-builds-ray-agents-outlive-the-driver]`, cannot be said, since neither rehearsal
    produced one.

#### 2.2.4 The manifests

From [`run_records_05.txt`](dk-and-rpk-migration/rehearsal/run_records_05.txt), for all four:
- `git_head` `6861a9fc0eb8cd60bb25cda8529ab45a51e50632`, the dispatch `HEAD`; `git_dirty` false.
- **`packages`** equals, key for key, the prompt's §2 record:
  `{"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit", "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false}, "raypoolkit": null}`.
- `script` is `docs/handover/quadsource_atol_sweep.py`. **`script_sha256`** is
  `b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`, the same in all four.
  - R0's was `d745783e65bfff99…`.
  - 04 rewrote the script's one layer import (`:693`, `from Datastore.shard_paths` →
    `from datastorekit.shard_paths`), its only change since R0.
- `argv[0]` is `docs/handover/quadsource_atol_sweep.py`. `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only.
- `results` names the reference store for `reference`, and the resumed store for the other three.
- `scope`, in each, is R0's with `-05-`:
  `REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32
  rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into
  dk-and-rpk-migration-05-<store>.sqlite`.

#### 2.2.5 Every launch's `stdout.log`, and what is new in the logs

Counted in each of the four, as §1.2.5:

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--zend', '250', …]` | 1 |
| `--no-prune-unvalidated`, in `stdout.log` or the launch log | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |
| `reconcil`, `repair`, `in.flight`, `replicat` (case-insensitive), in `stdout.log` or the launch log | 0 |
| `Traceback`, `ray.kill`, `killed actor`, `ActorDiedError`, `RayActorError` | 0 |

The launch logs of `resumed-b` and `resumed-c` begin `** quadsource_atol_sweep --build: resuming
into dk-and-rpk-migration-05-resumed.sqlite`.

**The comparison with R0, slug by slug**
([`log_comparison_05.txt`](dk-and-rpk-migration/rehearsal/log_comparison_05.txt)).
- **The method.** Each line of each R1 `stderr.log`, launch log and `stdout.log` was normalised and
  compared, as a set of line kinds, with R0's same file for the same slug. The normalisation:
  - ANSI colour codes stripped;
  - the prompt number and the repository path replaced;
  - Ray's `[repeated Nx across cluster]` count replaced;
  - every number replaced by `N`, which removes times, pids, serials and counts.
- **The result.** It was run by a throwaway script in the agent's scratchpad, not committed; the
  normalisation is the one above.

| Slug | `stderr.log` | launch log | `stdout.log` |
|---|---|---|---|
| `reference` | equal (4 lines, 4 kinds each) | equal (96 lines, 79 kinds each) | equal (90 lines, 73 kinds each) |
| `resumed-a` | equal (2, 2) | equal (75, 65) | equal (71, 61) |
| `resumed-b` | R1 has 4 kinds R0 has not: the `IntegrationWarning`'s four lines | the same 4 kinds; 132 lines against 120 | equal kinds; 123 lines against 115 |
| `resumed-c` | R0 has 4 kinds R1 has not: the `IntegrationWarning`'s four lines | the same 4 kinds; 90 lines against 94 | equal (87, 57) |

**Every difference, explained.**
1. **The SciPy `IntegrationWarning`** ("The occurrence of roundoff error is detected…", from a
   `compute_QuadSource_integral` worker).
   - It is in `reference` at both R0 and R1.
   - In the resumed chain it is in R1's `resumed-b` and R0's `resumed-c`.
   - It belongs to one integral, which falls in whichever launch computes it. R1's `resumed-b` ran
     about 95 s of the QSI stage and stored 334 rows, against R0's 70 s and 332. Which rows
     complete first is completion order, so the integral fell before R1's point and after R0's.
   - The point moves it. It is not a finding.
2. **`resumed-b`'s eight extra lines** (12 in the launch log, which also holds the stderr lines) are
   the integrity warning's entries.
   - R1 pruned 45 rows at open where R0 pruned 43.
   - R1 shows shard 0 in full, with 15 entries, against R0's 11.
   - Ray's folding of shards 1 and 3 into one shown entry `[repeated 30x across cluster]` (R0:
     `32x`) varies run to run (§1.3.2).
   - The count of entries is 15 + 30 = 45, as the store says (§2.3.2).
3. **`resumed-c`'s launch log is four lines shorter** by the warning of item 1.
4. **The handler's `SystemExit(1)` in `resumed-a`**, where R0 had `SystemExit(143)` (§2.2.2). The
   normalisation hides it, being a number. It is a value, not a new kind of line, and depends on
   where the signal lands.

**Nothing else differs.** No line from the package's new close behaviour appears, and no warning or
traceback appears that R0 did not print.

#### 2.2.6 A3 v2's guard

After `resumed-c` and the readers, in the foreground, with no `--register`
([`guard_05.txt`](dk-and-rpk-migration/rehearsal/guard_05.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/dk-and-rpk-migration-05-resumed.sqlite \
    --job-name dk-and-rpk-migration-05 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (20:07:16–20:07:17), with:

```
RuntimeError: quadsource_atol_sweep --build: dk-and-rpk-migration-05-resumed.sqlite, dk-and-rpk-migration-05-resumed-shard0000.sqlite, dk-and-rpk-migration-05-resumed-shard0001.sqlite, dk-and-rpk-migration-05-resumed-shard0002.sqlite, dk-and-rpk-migration-05-resumed-shard0003.sqlite already exist(s). Refusing to build into an existing store: a comparator that silently inherited earlier rows is one nobody can interpret. Pass --resume to continue an interrupted build at the SAME tolerance pair, or choose a path that does not exist.
```

- It names the five files that exist.
- Afterwards, `shasum -a 256 -c` gave `OK` on every file of both stores and both sidecars, against
  the hashes taken at 19:55, after `resumed-c`.
- No run directory was created (47 entries under `var/runs/` before and after), and no Ray process
  of this repository appeared.

### 2.3 L2 — the resume is the reference

#### 2.3.1 Nothing stored is recomputed

Through S1's [`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py)
([`stage_counts_05.txt`](dk-and-rpk-migration/rehearsal/stage_counts_05.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/stage_counts.py \
    var/runs/dk-and-rpk-migration-05-reference-20261011T192802 \
    var/runs/dk-and-rpk-migration-05-resumed-a-20261011T194149 \
    var/runs/dk-and-rpk-migration-05-resumed-b-20261011T194703 \
    var/runs/dk-and-rpk-migration-05-resumed-c-20261011T195044
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
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **235/235/235** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **530/530/530** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored 101 rows before the signal (35 + 30 + 0 + 36 by shard), 45 of them
    unvalidated (§2.3.2).
  - `resumed-b` pruned the 45 at open and computed 235 = 291 − 101 + 45: what was missing, plus
    the pruned rows. No validated row was computed again.
  - Summed, less the pruned: 101 + 235 − 45 = 291, the reference's.
- **`QuadSourceIntegral`.**
  - `resumed-b` printed no progress line in the stage. The closed store held **334** rows
    (118 + 93 + 0 + 123 by shard), read `mode=ro`, every file's SHA-256 unchanged by the read.
  - `resumed-c` computed **530** = 864 − 334, exactly what was missing.
  - Summed: 334 + 530 = 864, the reference's. No stored row was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b`, because `resumed-a` never reached it,
  and 0/0/0 in `resumed-c`.

#### 2.3.2 The prune at startup

The closed store after each launch, read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`, with each file's SHA-256 unchanged by the
read ([`closed_store_state_05.txt`](dk-and-rpk-migration/rehearsal/closed_store_state_05.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **45 `GkSource`** (every other class with a `validated` column: none):<br>shard 0: 219, 220, 221, 222, 223, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235;<br>shard 1: 513, 519, 528, 529, 530, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541;<br>shard 3: 5, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 |
| `resumed-b` | none | 0 rows | none |
| `resumed-c` | none | 0 rows | none |

**By serial**, through [`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py)
([`prune_check_05.txt`](dk-and-rpk-migration/rehearsal/prune_check_05.txt)):

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/dk-and-rpk-migration-05/state/after-resumed-a.txt \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/dk-and-rpk-migration-05/state/after-resumed-b.txt \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite
```

- **The first ran twice:** at 19:50:54, once `resumed-c` had opened the store, and again after
  `resumed-c`. Both gave "45 serial(s) checked: 45 absent, 0 reused by a validated row, 0 still
  unvalidated".
- The closed-store read after `resumed-b` already showed 0 unvalidated rows and 97 + 97 + 0 + 97
  `GkSource` rows.
- **The second** reports "none: there is nothing to check": after `resumed-b` no serial was
  unvalidated.

**What `resumed-b` printed.** Each actor's integrity warning, at open (19:47:08):
- **shard 0 in full:** `!! INTEGRITY WARNING … datastore "shard0000-store" …`, then `The following
  unvalidated GkSource objects were detected in the datastore:`, fifteen entries (`store_id` 219–223
  and 226–235), and `** Note: these values have been pruned from the datastore.`;
- **shards 1 and 3 folded by Ray:** `!! INTEGRITY WARNING … "shard0001-store" … [repeated 2x across
  cluster]`, one entry shown with `[repeated 30x across cluster]`, and the note `[repeated 2x across
  cluster]`.

15 + 30 = 45, the count from the store. `resumed-c` printed no integrity warning, since nothing was
unvalidated. `QuadSourceIntegral` has no `validated` column, so the rows `resumed-b` stored need no
prune.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§2.3.5), so no `duplicate`. Each finish's fingerprint is built from the package's
`read_inventory`, and its `problems` is `{}` for all four launches.

**The pool's prune of a replicated class.** None. No launch printed a prune of a replicated class.
After every launch each shard's `BackgroundModel` row is validated, and the census finds the
replicated tables equal (§2.3.5).

#### 2.3.3 The version row

From `closed_store_state.py`, after every launch: exactly one row, `(1, '2025.1.1')`, on each of the
four shards of both stores. The primary has no `version` table.

#### 2.3.4 The comparison

**The two stores**, by `compare_stores.py`
([`fingerprint_comparison_05.txt`](dk-and-rpk-migration/rehearsal/fingerprint_comparison_05.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \
    var/datastores/dk-and-rpk-migration-05-reference.sqlite \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite
```

Exit 0:

```
reference: …/var/datastores/dk-and-rpk-migration-05-reference.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches
resumed: …/var/datastores/dk-and-rpk-migration-05-resumed.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches

compare_fingerprints(reference, resumed): 0 difference(s)
verdict: equal
```

**Against R0's reference, 07's reference and 09's rehearsal.** R0's stores were not opened. Each R1
store was fingerprinted read-only again and compared with the fingerprint recorded in each earlier
run's `status.json`, by [`compare_with_recorded.py`](dk-and-rpk-migration/rehearsal/compare_with_recorded.py)
([`fingerprint_against_recorded_05.txt`](dk-and-rpk-migration/rehearsal/fingerprint_against_recorded_05.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/dk-and-rpk-migration-05-reference.sqlite \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite \
    --against var/runs/dk-and-rpk-migration-02-reference-20261011T143003 \
    var/runs/a3-v2-readiness-07-reference-20261004T012909 \
    var/runs/datastore-generic-09-rehearsal-20261006T152452
```

It exited 0, verdict `equal`. For each store there were 0 differences against:
- its sidecar;
- R0's reference (taken at `a5392fd1…`);
- 07's reference (taken at `dbd8a511…`);
- 09's rehearsal (taken at `53d4e188…`).

All are format 6, digest `39809dca…`, 21 classes, problems none. The files of both stores had the
same SHA-256 before and after these reads.

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

**The census** ([`census_05.txt`](dk-and-rpk-migration/rehearsal/census_05.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py \
    var/datastores/dk-and-rpk-migration-05-reference.sqlite \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite
```

- Every table is `OK` on both stores (28 lines), and the verdict is `no divergence` on both.
- With the prompt number normalised, the output equals R0's `census_02.txt` line for line: the same
  14 tables, with the same rows per shard.

**`integrity_check`, `foreign_key_check` and the inventory**
([`store_soundness_05.txt`](dk-and-rpk-migration/rehearsal/store_soundness_05.txt)), through this
campaign's [`store_soundness.py`](dk-and-rpk-migration/rehearsal/store_soundness.py) as 04 left it
(its `:30` imports `datastorekit.store_inventory`):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py \
    var/datastores/dk-and-rpk-migration-05-reference.sqlite \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite
```

- It exited 0, `=> clean`.
- On the primary and every shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is
  `['ok']` and `PRAGMA foreign_key_check` returns 0 rows.
- The package's `read_inventory(primary, factories)` reads shards `(0, 1, 2, 3)`, and reports no
  problem in any class of either store and 0 `duplicate`.
- Each class's record count is §2.3.4's table.
- With the prompt number normalised, the output equals R0's `store_soundness_02.txt` line for line.
- Every file of both stores had the same SHA-256 before and after these reads, and no journal
  appeared.

### 2.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance,
through S1's [`run_readers.sh`](dk-and-rpk-migration/rehearsal/run_readers.sh)
([`readers_05.txt`](dk-and-rpk-migration/rehearsal/readers_05.txt)):

```bash
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite var/dk-and-rpk-migration-05/extract
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/dk-and-rpk-migration-05-resumed.sqlite var/dk-and-rpk-migration-05/extract \
    nomodels extract_Gk_data
```

Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` / traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 52 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 450 | none | identical |
| `extract_GkWKB_data.py` | 0 | 250 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 77 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 176 | none | identical |
| `extract_TkWKB_data.py` | 0 | 28 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | **0** | 36 s | `EXTRACT TENSOR SOURCE DATA`: 6, all six `QuadSource` rows | 27 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 49 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 | 75 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 68 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 513 | none | identical |

- **The lookups equal R0's, reader by reader** (§1.4).
- **The files written differ.** For `extract_Gk_data.py` it is 450 against 324, and 513 against
  423 without `--models`; for `extract_GkSource_data.py` 176 against 144; for
  `extract_QuadSourceIntegral_data.py` 75 against 74. Each of these readers draws its subsample
  with `random.sample` (`extract_Gk_data.py:390`, `:406`; `extract_GkSource_data.py:839`, `:855`;
  `extract_QuadSourceIntegral_data.py:1132`, `:1138`). Which plots a drawn object yields depends on
  its branch type, so the file count varies run to run. 02's review recorded the same of
  `extract_Gk_data.py`. It is not a finding.
- **The wall times are 2.4–4.4 times R0's.** The load rose from 7 to 130–157 during the six,
  from processes not of this repository. The top CPU users at the end were PyCharm and a
  `python -m unittest discover -s datastorek…` process; neither was started by this prompt.
  - `extract_GkWKB_data.py` ran 250 s, so its queue printed progress lines that R0's 57 s run did
    not. They are the only line kinds in any reader log that R0's logs lack.
  - `extract_Gk_data.py`'s `Queue summary: … (avg …/min)` appears where R0 had `(avg …/sec)`: a
    slower rate, which the queue prints in per-minute units.
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
  - The SHA-256s are those taken after `resumed-c`: `19cd8373…` (primary), then `47ec0052…`,
    `430dba4d…`, `b12a4bf7…` and `47338a19…` (shards 0–3).
  - The mtimes all predate the first reader (19:56:44): 19:50:49 for the primary, 19:54:54 for
    shards 0, 1 and 3, and 19:42:14 for shard 2.
  - No Ray process of this checkout was left after any reader.
- **The shard-count warning.** Each log opens `!! WARNING: number of shards read from database (=4)
  does not match specified number of shards (=10)`, as at R0 and 07.
- **The fingerprint afterwards.** `store fingerprint` (without `--write`) on the resumed store gave
  `39809dca…`, problems none, "matches the fingerprint taken 2026-10-11T19:54:56+01:00 by run
  dk-and-rpk-migration-05-resumed-c-20261011T195044".

### 2.5 L4 — equal to R0

#### 2.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/policy_statistics.py \
    var/runs/dk-and-rpk-migration-05-reference-20261011T192802
```

Exit 0. The reference run's block, lines 83–96 of its `stdout.log`, verbatim
([`policy_statistics_05_reference.txt`](dk-and-rpk-migration/rehearsal/policy_statistics_05_reference.txt)):

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

**Against R0's `policy_statistics_02_reference.txt`.**
- **Line for line, it differs.** `diff` of R0's block against R1's:

  ```
  10d9
  <            minimal: 1 instances = 3.85% of this type, 0.34% of total
  11a11
  >            minimal: 1 instances = 3.85% of this type, 0.34% of total
  ```

  Under `>> mixed`, R0 printed `minimal` then `complete`; R1 prints `complete` then `minimal`.
- **Sorted, the blocks are equal.**
- **R1's block equals `datastore-generic` 09's line for line**, and so 07's.

This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]` (§1.5.2): the order is
Ray's completion order, and the counts are the measurement.

| | R0 `reference` | R1 `reference` |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| Order under `mixed` | `minimal`, `complete` | `complete`, `minimal` |

**R1's own resumed runs** (the outputs are under `var/dk-and-rpk-migration-05/checks/`):
- `resumed-b`, which recomputed all 291 policy rows, prints R1's reference order.
- `resumed-c`, which computed none, prints the block with R0's reference order.
- `resumed-a` has no block, and the script exits 1 on it, as at R0.

All have the same counts.

#### 2.5.2 The reference run's stage counts

From [`stage_counts_05.txt`](dk-and-rpk-migration/rehearsal/stage_counts_05.txt), against R0's
`stage_counts_02.txt`, stage by stage:

| Stage | R0 `reference` | R1 `reference` |
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

Equal in every stage. The `resumed-a` column is also R0's, entry for entry, since both were killed
in the same stage.

#### 2.5.3 The resumed chain, by identity

The raw counts of `resumed-b` and `resumed-c` depend on where each launch was killed, and R1 chose
its own points (§2.1.2). So L4 compares what holds whatever the point:

| Identity | R0 | R1 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated (`prune_check.py`) | 43 = 43, all absent | 45 = 45, all absent | ✅ |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | 215 = 291 − 119 + 43 | 235 = 291 − 101 + 45 | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 532 = 864 − 332 | 530 = 864 − 334 | ✅ |
| Totals per class over the chain equal the reference's | `GkSource` 119 + 215 − 43 = 291; QSI 332 + 532 = 864 | `GkSource` 101 + 235 − 45 = 291; QSI 334 + 530 = 864 | ✅ |

What each resumed launch pruned and recomputed, beside §1.5.4:

| Launch | Pruned at open | Computed, stage by stage | Of which recomputing a pruned row |
|---|---|---|---|
| `resumed-a` | nothing (new store) | everything up to the `GkSource` stage, as `reference`; 101 `GkSource` stored (45 unvalidated) at the signal | — |
| `resumed-b` | **45 `GkSource`** (15 + 15 + 15, shards 0, 1, 3) | `GkSource` 235; `GkSourcePolicyData` 291; `QuadSourceIntegral` 334 stored at the signal; every earlier stage 0 | 45 `GkSource` |
| `resumed-c` | nothing | `QuadSourceIntegral` 530; every other stage 0 | — |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 2.5.4 The stage sequence

The lines of each reference `stdout.log` that begin `** `
([`log_comparison_05.txt`](dk-and-rpk-migration/rehearsal/log_comparison_05.txt)):
- R0's and R1's each hold 26 such lines.
- With the prompt number and the run's timestamp normalised, the two sequences are equal, in order.
- Of the 26, 15 are the pipeline's stage banners, `** BUILD ARRAY OF WAVENUMBERS TO SAMPLE` through
  `** CALCULATE QUADRATIC SOURCE INTEGRALS`, equal in order. The other 11 are the launcher's own
  lines (`** quadsource_atol_sweep …` and the two `**   ` lines under its registration).

#### 2.5.5 The stage times

Not compared, but recorded for 08 to choose its points from (`ALL WORK ITEMS COMPLETE in time …` in
each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1 m 46.2 s | 1 m 21.5 s | 1.65 s | 1.61 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3 m 37.2 s | 2 m 15.1 s | 3.38 s | 4.05 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **55.1 s**, no progress line | killed | 36.9 s | 5.61 s |
| APPLY GKSOURCE POLICIES | 21.5 s | — | 13.9 s | 5.17 s |
| QUADRATIC SOURCE INTEGRALS | **4 m 49.7 s**, no progress line | — | killed, no progress line | 3 m 48.2 s |

- Each earlier stage took under 5 s.
- At loads of 14–78, every stage is slower than R0's at 6–36. The reference took 12 m 13 s against
  8 m 0 s.
- The QSI stage, at 289.7 s, came within 10 s of the 300 s at which its queue may first print a
  progress line. At a higher load a reference QSI stage may print one. 08 should read its own
  reference for that, as here.

### 2.6 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen. After every launch each shard's
  `BackgroundModel` row is validated (§2.3.2), and no resumed launch printed a prune of it.
- **An in-flight record left by a killed launch.** Did not happen. `replication_in_flight` had 0 rows
  after every launch, the two killed ones included.
- **A repair by the check at open.** Did not happen. No `stdout.log` or launch log holds `reconcil`,
  `repair`, `in.flight` or `replicat` (§2.2.5).
- **A Ray agent outliving a killed driver.** Did not happen (§2.2.3).
- **Anything the package raises that the layer did not.** Did not happen:
  - no refused open;
  - no traceback in any `stderr.log` or launch log;
  - no reader that refused or failed;
  - no `ReadOnlyMiss` or `ReadOnlyWrite`.

### 2.7 Issues opened

None. Every difference from R0 is explained by where a launch was killed, by Ray's completion order
(an issue already open), by the readers' random subsamples, or by the load.

### 2.8 The suites, and R0's files

**The suites.** Serially, in the foreground, after the last launch and the guard, with README §7's
command; Datastore in four groups. Every count is the dispatch baseline.

| Root | Ran | Result | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 | OK | 74.5 |
| CosmologyModels | 39 | OK | 74.5 |
| LiouvilleGreen | 148 | OK (1 skipped) | 72.6 |
| RunRegistry | 286 | OK | 40.5 |
| ComputeTargets | 693 | OK | 13.4 |
| Datastore `[a-f]` / `[g-o]` / `[p-r]` / `[s-z]` | 54 / 103 / 101 / 62 = 320 | OK | 7.4 / 7.3 / 8.1 / 8.1 |

`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which failed at dispatch, passed here.

**R0's twelve files** (`var/datastores/dk-and-rpk-migration-02-*`), SHA-256, mtime and size, before
R1's first launch (19:26) and after its last step (20:16, and again before the commit), are
identical:

| File | SHA-256 | mtime |
|---|---|---|
| `-reference.sqlite` | `9b7a0acc8bcca22171ada0a9ba08b892aa6eb470e24d8fbe35c587efbdc682a8` | 14:30:17 |
| `-reference-shard0000.sqlite` | `3e469052e8810e1efec8dee53c3a89ea4a2a343d3346cd2a64681170fd478bd5` | 14:38:00 |
| `-reference-shard0001.sqlite` | `210f494aedfd154fde34754d0dffb8030c42a1ae701fb3b10c9f94434293b8f8` | 14:38:00 |
| `-reference-shard0002.sqlite` | `37b12c1592c2f03e22a862797c30dc6f2b01c4e8c3af4f83bd672d79fed21ec9` | 14:30:17 |
| `-reference-shard0003.sqlite` | `db35d513a3d6278f09729dcdb7c0ba3ca645562dfaf1df4de5b9f4070099f1ce` | 14:38:00 |
| `-reference.manifest.json` | `755da82a40a26c4b530f4ad6451939cc6863f9370ba42ef006f275c621f595f2` | 14:38:55 |
| `-resumed.sqlite` | `bf22863f9ea8b6cf39ab9e4bbcebe5ae0e303983b9106be63abb53ece5508062` | 14:45:54 |
| `-resumed-shard0000.sqlite` | `e1340524ecc5bb94224b6aafae357df17e9e4ecc09bbd7af892f6ef7010acb4f` | 14:48:46 |
| `-resumed-shard0001.sqlite` | `35908f3a09e6683843fc5f62e7da968a6693a473df90fc448420e5dcd8ecae3b` | 14:48:46 |
| `-resumed-shard0002.sqlite` | `fafbe696ba83ac3bf6dfea5c96ac84a1a8485b885bc96adad95f25618826060b` | 14:39:24 |
| `-resumed-shard0003.sqlite` | `09b8aef1061ef5ed2182833353402af30c9f99cd39c7318a6358682acf2640ea` | 14:48:41 |
| `-resumed.manifest.json` | `c93c629616fa92a11c2a324f0382d45df5b91b7ebbb4e64822967c6cc53b58da` | 14:48:49 |

Each equals the orchestrator's dispatch table. No `-journal`, `-wal` or `-shm` file is under
`var/datastores/`.

### 2.9 What stays under `var/`

- `var/runs/dk-and-rpk-migration-05-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`. Each holds
  the run's manifest, `status.json` with its full fingerprint, `stdout.log` and `stderr.log`. For
  `resumed-a` and `resumed-b` it also holds `interrupter.log`.
- `var/runs/dk-and-rpk-migration-05-<slug>.launch.log`, beside each run directory.
- `var/datastores/dk-and-rpk-migration-05-{reference,resumed}.sqlite`, four shards each, and each
  store's `.manifest.json` sidecar. Shard 2 of each, 1 077 248 bytes, holds only replicated rows.
- `var/dk-and-rpk-migration-05/`, which holds:
  - the agent's working notes (`notes.txt`);
  - the pre-launch captures and dummy-test outputs (`prelaunch/`);
  - the closed-store states and hashes (`state/`);
  - the check outputs (`checks/`, among them every run's statistics block and 09's);
  - R0's hashes after the last step (`r0_hashes_after.txt`);
  - the readers' outputs and logs (`extract/`, `extract-summary-*.txt`).

### 2.10 G-DK — DatastoreKit's G2

DatastoreKit's G2 (README §0) holds when its four parts hold (prompt 05 §2). Each rests on a named
commit or run:

| Part | What | Rests on | Holds |
|---|---|---|---|
| 1 | The 17 files are gone, two emptied | `0c9e2cf` (prompt 04, reviewed, board §5.4). Re-checked at `6861a9f`: each of the 15 deleted modules raises `ModuleNotFoundError` in a fresh `python -I`, and both `__init__.py` are 0 bytes (§2.1.1) | ✅ |
| 2 | The pin is in place and the venv installed from it | `9ec603a` (prompt 03, reviewed, board §5.3). Re-checked at `6861a9f`: `requirements.txt:17`, `pip show` 0.2.2 in this venv, `pip check` clean. All four R1 manifests record `datastorekit` 0.2.2 at `6d8ee9ae…`, `editable: false` (§2.2.4) | ✅ |
| 3 | The six roots pass | This prompt's suites at `6861a9f`, whose code is `0c9e2cf`'s: Datastore 320, RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), all OK (§2.8) | ✅ |
| 4 | R1 reproduces `39809dca…` | The runs `dk-and-rpk-migration-05-reference-20261011T192802` and `dk-and-rpk-migration-05-resumed-c-20261011T195044`, both `done`, both `39809dca…` with all 21 classes. They are equal to each other, to R0's, 07's and 09's (§2.3.4) | ✅ |

**DatastoreKit's G2 holds.** Recording it on DatastoreKit's own extraction board is a commit in that
repository, not here (README §6.2 C12).

---

## 3. R2 — through both packages

**§3's tree, stores and runs.** Every R2 launch ran on branch `handover-remedial` at **`5631f0e`**
("Write the orchestrator note for dk-and-rpk-migration prompt 08").
- Its code is `643c3b2`'s (prompt 07), but for one test comment. `af26914`, 07's review, changed a
  comment in `RunRegistry/tests/test_pipeline_adoption.py`, and that file's `ast` is equal at the two
  commits. `git diff --stat 643c3b2 5631f0e -- . ':!prompts'` lists that file alone.
- Its datastore layer is the installed **`datastorekit` 0.2.2**, at `6d8ee9ae…`, not editable.
- Its work queue is the installed **`raypoolkit` 0.2.0**, at `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`,
  not editable. `RayTools/` does not exist.
- `git status --short` was empty at every launch.
- Every manifest stamps `git_dirty: false` and `packages`
  `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false, "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version": "0.2.2"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}}`.

The same launcher as at R0 and R1 built two throwaway stores:
- **`var/datastores/dk-and-rpk-migration-08-reference.sqlite`** (sidecar `store_id`
  `90908c95f9ff48ca87cd63885976fb1d`), built straight through by the run **`reference`**;
- **`var/datastores/dk-and-rpk-migration-08-resumed.sqlite`** (`store_id`
  `d0e2c2c076814c78a53512d63e2274ba`), built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

The runs are `var/runs/dk-and-rpk-migration-08-<slug>-<stamp>/`. Both stores stay, with their
sidecars (README §3). The small outputs quoted below are in
[`dk-and-rpk-migration/rehearsal/`](dk-and-rpk-migration/rehearsal/), under names carrying `08`.
The large ones stay under `var/` (§3.9).

R0's and R1's stores were not opened. R2 is compared with them through the fingerprints their
`reference` runs recorded (§3.3.4). Their 24 files were hashed before R2's first launch and after its
last step.

**Headline.**
1. **R2 reproduces the reference fingerprint through both packages.**
   - Both stores are format 6, digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`,
     all 21 classes equal, and each equals its sidecar's.
   - `compare_fingerprints` is empty between the two stores.
   - It is also empty between each store and the fingerprints recorded by:
     - `dk-and-rpk-migration-05-reference` (R1, `6861a9f`);
     - `dk-and-rpk-migration-02-reference` (R0, `a5392fd`);
     - `a3-v2-readiness-07-reference` (`dbd8a51`);
     - `datastore-generic-09-rehearsal` (`53d4e18`) (§3.3.4).
   - The only change to what runs between R1 and R2 is the work queue. So `raypoolkit` 0.2.0
     drives the build to the store SGK's own `RayTools/` built.
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0. `resumed-a` and
   `resumed-b` are `killed`/143, each with its fingerprint, and each driver exited 143 (§3.2).
3. **The resume recomputed nothing it had stored.**
   - **`resumed-a` stored no `GkSource` row before its signal.** So `resumed-b` pruned nothing at
     open, and computed 291 = 291 − 0 + 0. R2 does not exercise the `GkSource` prune at open
     (§3.3.2).
   - `resumed-c` computed the 583 = 864 − 281 `QuadSourceIntegral` rows not stored when `resumed-b`
     was killed, and nothing else.
   - Nothing is duplicated (§3.3).
4. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§3.4). Each reader's queues are now `raypoolkit`'s.
5. **L4 equals R1's** (§3.5):
   - the statistics counts are equal, in `reference` and in `resumed-b`;
   - the reference's stage counts are equal, stage by stage;
   - the resumed chain's identities hold, the prune's vacuously;
   - the reference's `** ` lines are equal, in order;
   - every completion line has R1's form and number, and the registry's `stage` fields are R1's.

   The statistics block holds R1's 14 lines, with `complete` and `minimal` under `mixed` in the
   other order, which is R0's order. That is
   `[02-the-policy-statistics-block-orders-its-lines-by-completion]` again, not a new finding.
6. **`main.py:3247` is measured** (§3.11). The post handler, now given the persisted object, counts
   what it counted from the local copy: 291; numeric 109; mixed 26 (25 complete, 1 minimal); WKB 156.
   This holds in `reference` and in `resumed-b`, the two launches that computed the policy rows.
7. **The logs differ from R1's only where the interruption points put them** (§3.2.5). No R2 log
   holds a traceback, a warning R1 did not print, or a line from the new pool.
8. **RayPoolKit's G2 holds** (§3.10).
9. **R0 and R1 were not touched.** Their 24 files have the same SHA-256 and mtime before R2's first
   launch and after its last step.

### 3.1 How each was launched

Each launch was README §4.1's command, verbatim but for the store, the slug, the purpose and
`--resume`. Each was run from the repository root, detached:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/dk-and-rpk-migration-08-<store>.sqlite \
    --job-name dk-and-rpk-migration-08 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "<one line>" --campaign dk-and-rpk-migration --prompt 08; \
  echo "** driver exit status $?"' \
  > var/runs/dk-and-rpk-migration-08-<slug>.launch.log 2>&1 &
disown
```

- `reference`: the reference store, no `--resume`.
- `resumed-a`: the resumed store, no `--resume`.
- `resumed-b` and `resumed-c`: the resumed store, with `--resume`.

Each `--purpose` names R2, the campaign, both packages and their commits, and the fingerprint it
must equal ([`run_records_08.txt`](dk-and-rpk-migration/rehearsal/run_records_08.txt)).

**One check per launch, one wait per launch.** After each launch the agent made one check:
- the manifest existed;
- `status.json` said `running`, with a pid answering `kill -0`;
- `RunRegistry list` showed the run.

The agent then ended its turn. The orchestrator ran one wait per launch, on the run's liveness, and
resumed the agent when the run was no longer alive. Neither polled. Every wait ended
`ENDED finished`, with the heartbeat never stale.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load before / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `dk-and-rpk-migration-08-reference-20261011T224347` | reference | no | 55979 (55974) | `done` | 0 | 0 | 11 m 25 s (22:43:47–22:55:12) | 12.82 15.94 14.77 / 33.02 29.06 22.79 |
| `resumed-a` | `dk-and-rpk-migration-08-resumed-a-20261011T225623` | resumed | no | 64487 (64482) | `killed` | 143 | 143 | 4 m 41 s (22:56:23–23:01:04) | 22.83 26.98 22.25 / 32.48 32.31 26.83 |
| `resumed-b` | `dk-and-rpk-migration-08-resumed-b-20261011T230305` | resumed | yes | 69056 (69051) | `killed` | 143 | 143 | 2 m 48 s (23:03:05–23:05:53) | 16.79 27.73 25.57 / 19.22 27.64 26.17 |
| `resumed-c` | `dk-and-rpk-migration-08-resumed-c-20261011T230717` | resumed | yes | 71947 (71942) | `done` | 0 | 0 | 5 m 39 s (23:07:17–23:12:56) | 15.16 25.83 25.56 / 32.77 57.25 43.64 |

- **Wall time** is from the manifest's `created` to the final heartbeat.
- **Load** is `uptime`'s on 10 CPUs: at the pre-launch check, and at the agent's first check after
  the run. The machine was shared. At the checks the one-minute load ran from 13 to 33, and the
  five-minute load reached 57 at 23:13, from processes not of this repository. During the readers it
  reached 252 (§3.4).

**The sidecars** ([`store_sidecars_08.txt`](dk-and-rpk-migration/rehearsal/store_sidecars_08.txt)):
- the reference store: `store create` on its closed primary at 22:55:56, after `reference`, then
  `store fingerprint --write` (`39809dca…`, problems none, "it replaced none");
- the resumed store: `store create` on its closed primary at 23:02:54, after `resumed-a`.

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`. The commands are §1.1's,
with `-08-` for `-02-`.

#### 3.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_08.txt`](dk-and-rpk-migration/rehearsal/prelaunch_checks_08.txt)):
- **The tree.**
  - `git rev-parse HEAD` was `5631f0ed5d9cefd8a3c79ba42480f2b9c59eb925`.
  - `git status --short` was empty, and was checked again in the same shell command that launched.
  - Every working note and capture, until the last launch had ended, went under
    `var/dk-and-rpk-migration-08/`, which is ignored. This record's files, and the saved
    measurement under `prompts/`, were written afterwards.
- `RAY_ADDRESS` was unset.
- **The registry.** `python -m RunRegistry list` showed nothing `running`: 25 runs before
  `reference` (23 finished, 2 `unknown`, the pre-registry `a3-pilot` pair), then 26, 27 and 28, all
  finished but those two.
- **No Ray process at all.**
  - The whole-table search, `ps -axo pid,ppid,lstart,command` for lines holding both
    `SecondaryGWKit/venv` and `ray`, was empty before each launch.
  - Before `reference`, `pgrep -lf 'gcs_server|raylet|ray::'` listed two pids, 50609 and 53388.
    Both were `zsh -c` shells of another session, working in DatastoreKit's scratchpad, whose
    command text holds that pattern itself. Neither was a Ray process, and neither was touched.
  - With `zsh -c` and `grep` lines excluded, no `gcs_server`, `raylet` or `ray::` process existed.
    From `resumed-a` on, `pgrep` was empty.
  - Before each later launch, the pids of the previous run were checked by `ps -p`: the `sh` and
    Python pids, the interrupter's, and every Ray pid its logs name. None was alive.
- **The store names.**
  - Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run
    naming either.
  - Nothing under `var/datastores/` or `var/runs/` contained `dk-and-rpk-migration-08`, and there
    was no journal file under `var/datastores/`.
  - Before `resumed-a`, no `dk-and-rpk-migration-08-resumed*` file existed.
- **The packages.**
  - `./venv/bin/pip show datastorekit raypoolkit` gave 0.2.2 and 0.2.0, each with `Location` this
    venv's `site-packages`. `pip check` was clean.
  - `RunRegistry.package_provenance()` gave the record above.
  - `requirements.txt:17` is `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2`.
  - `requirements.txt:76` is `raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0`.
- **What runs is the package.** In a fresh `python -I` with the repository on `sys.path`:
  - `import RayTools` and `import RayTools.RayWorkPool` raise `ModuleNotFoundError`;
  - `raypoolkit.__file__` is `venv/lib/python3.12/site-packages/raypoolkit/__init__.py`;
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`;
  - `datastorekit.__file__` is under the same `site-packages`;
  - `ray.is_initialized()` is false.

  No `RayTools*` path exists outside `venv/` and `var/`, `__pycache__` included.
  `git ls-tree 643c3b2 RayTools` is empty.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` was `b7935877…`, R1's
  (§3.2.4).
- **R0's and R1's 24 files.** Their full SHA-256, mtime and size were recorded before `reference`.
  All 24 equal the orchestrator's table (§3.8).
- 68–72 GB free on `var/`'s volume.

#### 3.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch.** Each ran into a new directory under the
agent's scratchpad, outside `var/` and the repository, with §1.1.2's commands. Both exited 0, with
every verdict as R0's and R1's
([`interrupter_dummy_test_08.txt`](dk-and-rpk-migration/rehearsal/interrupter_dummy_test_08.txt),
[`interrupter_build_dummy_test_08.txt`](dk-and-rpk-migration/rehearsal/interrupter_build_dummy_test_08.txt)):
- **The five cases, 22:43:01–22:43:26.**
  - A signalled at 45 % ≥ 40 % (child 143).
  - B exited silently when the child died first.
  - C refused `'sleep 30'` with exit 2, naming "`scoped_pipeline_run.py` (the run's own driver,
    its manifest's argv[0])", and the child lived.
  - D held 4 s and signalled (child 143).
  - E did not signal when the stage completed during the hold.
- **The `--build` cases, 22:43:26.**
  - A signalled the `quadsource_atol_sweep.py --build --register dummy-build-a` child
    (interrupter 0, child 143).
  - B refused the `scoped_pipeline_run.py` child (exit 2, child alive).
  - C: "the run's manifest names no argv[0]", exit 2, child alive.
- No dummy child or interrupter was left alive (`ps -p` on all 16 pids the two outputs name found
  none).

**The points, from R2's own reference `stdout.log`.** Neither interrupted stage printed an
intermediate progress line. Both queues have a 300 s minimum notify interval (`main.py:119`).
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **41.8 s** (R1: 55.1 s;
  R0: 32.2 s). It was interrupted from the banner with `--interval 5 --hold 10`, R0's and R1's
  arguments.
  - That puts the signal 10–15 s into the stage.
  - It is well before the stage's end, even if the stage ran at R0's speed.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **4 m 59.2 s** (299.2 s; R1:
  289.7 s; R0: 212.3 s). It was interrupted from the banner with `--interval 5 --hold 100`.
  - That is about a third of the stage, and before its end even at R0's 212.3 s.
  - 100 s is under the queue's 300 s minimum, so no progress line can come during the hold.
  - The pattern is `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its end, as R0's and
    R1's were (§1.1.2 gives the reasons).

**The real firings** ([`interrupter_firings_08.txt`](dk-and-rpk-migration/rehearsal/interrupter_firings_08.txt)):

```
resumed-a (interrupter pid 64545; '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10)
2026-10-11T23:00:52+01:00 liveness alive; pid 64487; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-11T23:00:52+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 10 s, then reading again
2026-10-11T23:01:02+01:00 sent SIGTERM to pid 64487 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/dk-and-rpk-migration-08-resumed.sqlite … --register resumed-a …'; exiting

resumed-b (interrupter pid 69114; '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 100)
2026-10-11T23:04:04+01:00 liveness alive; pid 69056; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-11T23:04:04+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 100 s, then reading again
2026-10-11T23:05:44+01:00 sent SIGTERM to pid 69056 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command '/opt/local/…/Python -u docs/handover/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/dk-and-rpk-migration-08-resumed.sqlite … --register resumed-b …'; exiting
```

- Each interrupter confirmed, before it signalled, that the pid's command line named
  `quadsource_atol_sweep.py`, its manifest's `argv[0]`, and `--register <slug>`.
- It refused no real run's pid, and no progress line appeared during `resumed-b`'s hold.
- **Each interrupter read the stage it waits for from `status.json`**, whose `stage` the registry
  takes from the pool's printed lines. Both banners were read, as at R1, so the pool's banner line
  is one `StageTracker` and `interrupt_when.py` can read.
- In `resumed-b` the reading before the banner (23:03:49) was `APPLY GKSOURCE POLICIES`. The
  interrupter reads every 5 s and logs a reading only when the stage changes, so the stage began
  between 23:03:59 and 23:04:04, and the signal came 100–105 s into it.

Both interrupters were launched detached, right after their builds, once the run directory existed:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/dk-and-rpk-migration-08-resumed-a-20261011T225623 \
    '^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10 > /dev/null 2>&1 &
disown
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/dk-and-rpk-migration-08-resumed-b-20261011T230305 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 100 > /dev/null 2>&1 &
disown
```

### 3.2 L1 — every launch ends as recorded

#### 3.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_error` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | not written: "…dk-and-rpk-migration-08-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `ff6c4257…`, problems `{}` | absent | not written: "…dk-and-rpk-migration-08-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `aa860ed8…`, problems `{}` | absent | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `39809dca…`, problems `{}` | absent | `written` | `** driver exit status 0` |

Each fingerprint holds 21 classes, and its `taken` names its run, `git_head` `5631f0ed…` and
`git_dirty` false.

The full digests:
- `reference` and `resumed-c`: `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
- `resumed-a`: `ff6c425751dbf2b78ee7b41b32d50172d7c143bb470c4300bdc5724c59d7164c`;
- `resumed-b`: `aa860ed8779a51703a9a5bde954a1b86dffd39016afe2da8fa1d9f85565bd9be`.

The two intermediate digests differ from R1's (`b6b89d84…`, `09170e40…`), as they must: each
digests a store killed at a different point.

#### 3.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 23:01:02 | 23:05:44 |
| Handler's line | `** SIGTERM at 2026-10-11 23:01:02 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-11 23:05:45 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` |
| Delay | under 1 s (equal whole-second stamps) | at most about 1 s (stamps one second apart) |
| `stderr.log`'s handler lines | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` | the line above, then `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` |
| Python traceback | none | none |
| Ray's `*** SIGTERM received` banner | absent (`stderr.log` and launch log) | absent |
| Stage at the signal | the `GkSource` banner, first read 23:00:52, signal 10 s later (**0 of 291 rows stored**) | the `QuadSourceIntegral` banner, first read 23:04:04, signal 100 s later (281 of 864 stored) |

- **`SystemExit(1)` in both**, as at R1: the signal lands during `ray.wait`, whose signal check
  converts the handler's `SystemExit(143)`. The latch is what `finish_after_exec` reads, so the
  record is `killed`/143 and the exit is 143 either way.
- **Each killed `stderr.log` holds only the handler's two lines.** R1's `resumed-b` also held the
  SciPy `IntegrationWarning`; R2's has it in `resumed-c` (§3.2.5).
- **`stderr.log` of `reference`** holds only the one `IntegrationWarning` from SciPy's `quad` in a
  `compute_QuadSource_integral` worker ("The occurrence of roundoff error is detected…"), as at R0
  and R1. **`resumed-c`'s `stderr.log`** holds the same warning, from worker pid 72838, and nothing
  else.

#### 3.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 22:55:45 (~33 s after the final heartbeat; load 33.02) | none: `pgrep` empty (other sessions' shells excluded), whole-table search empty, pids 55974 and 55979 gone, and the Ray pids its logs name (56026, 56028, 56029, 56349) gone |
| `resumed-a` | 23:01:45 (~41 s; load 32.48) | none, by the same checks for pids 64482, 64487, the interrupter 64545, and the Ray pids 64535, 64536, 64539 |
| `resumed-b` | 23:06:41 (~48 s; load 19.22) | none, by the same checks for pids 69051, 69056, the interrupter 69114, and the Ray pid 69106 |
| `resumed-c` | 23:13:33 (~37 s; load 32.77) | none, for pids 71942, 71947 and the Ray pids 72067, 72068, 72838 |

- **No Ray process outlived either killed driver,** so no wait by pid was run. The agents' exit
  time is "none listed at the first check", as at R0 and R1.
- No launch log or `stderr.log` holds a `ray.kill`, killed-actor, `ActorDiedError` or
  `RayActorError` line (§3.2.5).

#### 3.2.4 The manifests

From [`run_records_08.txt`](dk-and-rpk-migration/rehearsal/run_records_08.txt), for all four:
- `git_head` `5631f0ed5d9cefd8a3c79ba42480f2b9c59eb925`, the dispatch `HEAD`; `git_dirty` false.
- **`packages`** equals, compared as JSON, the prompt's §2 record:
  `{"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit", "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false}, "raypoolkit": {"version": "0.2.0", "url": "https://github.com/ds283/RayPoolKit", "commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "requested_revision": "v0.2.0", "editable": false}}`.
  R1's held `raypoolkit: null`; this is the only field that differs from R1's in kind.
- `script` is `docs/handover/quadsource_atol_sweep.py`. **`script_sha256`** is
  `b7935877cdbc4e54f6e85a9769f16c2182f7840eaa4bfcb0af2da5bfb9bd3cac`, the same in all four, and R1's.
  The launcher has not changed since `0c9e2cf`, and names no pool.
- `argv[0]` is `docs/handover/quadsource_atol_sweep.py`. `argv` holds `--build --rehearsal`, and
  `--resume` in `resumed-b` and `resumed-c` only.
- `results` names the reference store for `reference`, and the resumed store for the other three.
- `scope`, in each, is R1's with `-08-`:
  `REHEARSAL build, not A3 v2: main.py's own unfiltered QuadSourceIntegral work list at atol=1e-32
  rtol=1e-07, 6 cpus, LambdaCDM, 3 log-spaced k over 100000-1e+06 /Mpc, zend 250, into
  dk-and-rpk-migration-08-<store>.sqlite`.

#### 3.2.5 Every launch's `stdout.log`, and what is new in the logs

Counted in each of the four, as §1.2.5 and §2.2.5
([`log_comparison_08.txt`](dk-and-rpk-migration/rehearsal/log_comparison_08.txt)):

| Line | Each launch |
|---|---|
| `** quadsource_atol_sweep: replaced 1 occurrence of …` | 4, one per substitution |
| `** quadsource_atol_sweep --build: rehearsal scope, NOT A3 v2: k = [100000.0, 316227.7660168379, 1000000.0], zend 250` | 1 |
| `main.py argv = [… '--zend', '250', …]` | 1 |
| `--no-prune-unvalidated`, in `stdout.log` or the launch log | 0 |
| `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, then `Started a local Ray instance.` (launch log) | 1 each |
| `reconcil`, `repair`, `in.flight`, `replicat` (case-insensitive), in `stdout.log` or the launch log | 0 |
| `Traceback`, `ray.kill`, `killed actor`, `ActorDiedError`, `RayActorError`, `ValueError`, `RuntimeWarning`, `did not validate`, in any of the three logs | 0 |
| `INTEGRITY WARNING`, in any of the three logs | 0 |

The launch logs of `resumed-b` and `resumed-c` begin `** quadsource_atol_sweep --build: resuming
into dk-and-rpk-migration-08-resumed.sqlite`.

**The comparison with R1, slug by slug.**
- **The method** is §2.2.5's. Each line of each R2 `stderr.log`, launch log and `stdout.log` was
  normalised and compared, as a set of line kinds, with R1's same file for the same slug. The
  normalisation:
  - ANSI colour codes stripped;
  - the prompt number and the repository path replaced;
  - Ray's `[repeated Nx across cluster]` count replaced;
  - every number replaced by `N`, which removes times, pids, serials and counts.
- **The result.** It was run by a throwaway script in the agent's scratchpad, not committed; the
  normalisation is the one above.

| Slug | `stderr.log` | launch log | `stdout.log` |
|---|---|---|---|
| `reference` | equal (4 lines, 4 kinds each) | equal (116 lines, 80 kinds each) | equal (110 lines, 74 kinds each) |
| `resumed-a` | equal (2, 2) | equal (88, 66) | equal (84, 62) |
| `resumed-b` | R1 has 4 kinds R2 has not: the `IntegrationWarning`'s four lines | R1 has 16 kinds R2 has not: the same 4, and the integrity warning's 12; 108 lines against 152 | R1 has 12 kinds R2 has not, the integrity warning's; 103 lines against 143 |
| `resumed-c` | R2 has 4 kinds R1 has not: the `IntegrationWarning`'s four lines | the same 4 kinds; 114 lines against 110 | equal (107, 58) |

**Every difference, explained.**
1. **The SciPy `IntegrationWarning`** ("The occurrence of roundoff error is detected…", from a
   `compute_QuadSource_integral` worker).
   - It is in `reference` at R0, R1 and R2.
   - In the resumed chain it is in R0's `resumed-c`, R1's `resumed-b` and R2's `resumed-c`.
   - It belongs to one integral, which falls in whichever launch computes it. R2's `resumed-b`
     stored 281 QSI rows before its signal, against R1's 334, so the integral fell after R2's
     point, as it did at R0 (332).
   - The point moves it. It is not a finding.
2. **`resumed-b`'s missing integrity warning.** R1's `resumed-b` printed, at its open, each shard's
   `!! INTEGRITY WARNING`, the list of unvalidated `GkSource` objects (with their
   `contains N z-sample values` lines) and `** Note: these values have been pruned from the
   datastore.` R2's `resumed-a` stored no `GkSource` row (§3.3.2). So there was nothing unvalidated
   at `resumed-b`'s open and nothing to warn of. That is all 12 kinds, and the 40 lines `stdout.log`
   is shorter by.
3. **`resumed-c`'s launch log is four lines longer** by the warning of item 1.
4. **The handler's `SystemExit(1)`** in both killed launches is R1's (§3.2.2).

**Nothing else differs.** No line from the new pool appears, and no warning or traceback appears
that R1 did not print. Every pool line, banner, completion and `Queue summary` alike, has R1's
form (§3.5.6).

#### 3.2.6 A3 v2's guard

After `resumed-c` and the readers, in the foreground, with no `--register`
([`guard_08.txt`](dk-and-rpk-migration/rehearsal/guard_08.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --build --rehearsal \
    --database var/datastores/dk-and-rpk-migration-08-resumed.sqlite \
    --job-name dk-and-rpk-migration-08 --atol 1e-32 --rtol 1e-7
```

It exited 1 within a second (23:22:58–23:22:59), with:

```
RuntimeError: quadsource_atol_sweep --build: dk-and-rpk-migration-08-resumed.sqlite, dk-and-rpk-migration-08-resumed-shard0000.sqlite, dk-and-rpk-migration-08-resumed-shard0001.sqlite, dk-and-rpk-migration-08-resumed-shard0002.sqlite, dk-and-rpk-migration-08-resumed-shard0003.sqlite already exist(s). Refusing to build into an existing store: a comparator that silently inherited earlier rows is one nobody can interpret. Pass --resume to continue an interrupted build at the SAME tolerance pair, or choose a path that does not exist.
```

- It names the five files that exist.
- Afterwards, `shasum -a 256 -c` gave `OK` on every file of both stores and both sidecars, against
  the hashes taken just before it. It also gave `OK` on the resumed store's five files against the
  hashes taken at 23:13, after `resumed-c` and before the readers.
- No run directory was created (55 entries under `var/runs/` before and after), and no Ray process
  of this repository appeared.

### 3.3 L2 — the resume is the reference

#### 3.3.1 Nothing stored is recomputed

Through S1's [`stage_counts.py`](dk-and-rpk-migration/rehearsal/stage_counts.py)
([`stage_counts_08.txt`](dk-and-rpk-migration/rehearsal/stage_counts_08.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/stage_counts.py \
    var/runs/dk-and-rpk-migration-08-reference-20261011T224347 \
    var/runs/dk-and-rpk-migration-08-resumed-a-20261011T225623 \
    var/runs/dk-and-rpk-migration-08-resumed-b-20261011T230305 \
    var/runs/dk-and-rpk-migration-08-resumed-c-20261011T230717
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
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **291/291/291** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **583/583/583** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored **no** `GkSource` row before the signal: 0 + 0 + 0 + 0 by shard, so none
    unvalidated (§3.3.2).
  - `resumed-b` pruned nothing at open, and computed 291 = 291 − 0 + 0: everything, since nothing
    was there. No validated row was computed again, since there was none.
  - Summed, less the pruned: 0 + 291 − 0 = 291, the reference's.
- **`QuadSourceIntegral`.**
  - `resumed-b` printed no progress line in the stage. The closed store held **281** rows
    (79 + 92 + 0 + 110 by shard), read `mode=ro`, every file's SHA-256 unchanged by the read.
  - `resumed-c` computed **583** = 864 − 281, exactly what was missing.
  - Summed: 281 + 583 = 864, the reference's. No stored row was computed again.
- `APPLY GKSOURCE POLICIES` is 291/291/291 in `resumed-b`, because `resumed-a` never reached it,
  and 0/0/0 in `resumed-c`.

#### 3.3.2 The prune at startup

The closed store after each launch, read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`, with each file's SHA-256 unchanged by the
read ([`closed_store_state_08.txt`](dk-and-rpk-migration/rehearsal/closed_store_state_08.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | none: **`GkSource` 0 rows on every shard**. `GkNumericIntegration` 456 + 406 + 0 + 506 = 1368 and `GkWKBIntegration` 3 × 1159 = 3477 are all stored and validated |
| `resumed-b` | none | 0 rows | none (`GkSource` 97 + 97 + 0 + 97 = 291) |
| `resumed-c` | none | 0 rows | none |

**`resumed-a` stored no `GkSource` row.** The record shows:
- its interrupter first read the `GkSource` banner at 23:00:52 and sent `SIGTERM` at 23:01:02,
  after a 10 s hold, so the signal came 10–15 s into the stage (§3.1.2);
- the closed store after it holds 0 `GkSource` rows on all four shards;
- R1's `resumed-a`, at the same `--hold 10`, had stored 101 (45 unvalidated), and R0's 119 (43
  unvalidated).

The run was killed in the right stage, so the interruption did not miss (prompt 08 §1, 05 §4). What
the record does not show is why no row had landed 10–15 s in. **An inference, not a measurement:**
the machine's load. R2's `resumed-a` took 2 m 50.7 s over the WKB stage before it, against R1's
2 m 15.1 s, and the load at the check after it was 32. The pool's scheduling code is the same
as the copy's (§3.11), so nothing read points at the pool. A third store, to try again, is
forbidden (the orchestrator's note).

**By serial**, through [`prune_check.py`](dk-and-rpk-migration/rehearsal/prune_check.py)
([`prune_check_08.txt`](dk-and-rpk-migration/rehearsal/prune_check_08.txt)):

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/dk-and-rpk-migration-08/state/after-resumed-a.txt \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/prune_check.py \
    var/dk-and-rpk-migration-08/state/after-resumed-b.txt \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite
```

- **The first ran twice:** at 23:06:54, after `resumed-b` had opened and closed the store, and
  again after `resumed-c`. Both gave `none: there is nothing to check`, exit 0.
- **The second**, after `resumed-c`, gave the same.

**So R2 does not exercise the `GkSource` prune at open.** L2's "the prune at open removed only
unvalidated rows" holds at R2 vacuously: there was nothing unvalidated to remove. `resumed-b`
printed no integrity warning, and `resumed-c` none either. **What does cover that path** is R1, run
with the same `datastorekit` 0.2.2 (§2.3.2). There, `resumed-b`'s open pruned the 45 `GkSource`
rows `resumed-a` left unvalidated, all 45 serials then absent and none reused. The prune runs in
the shard actors at open (the package's `Datastore`), not in the work queue, so R2's change of
queue does not reach it.

**Nothing is duplicated.** `read_inventory` after `resumed-c` reports no problem in any class
(§3.3.5), so no `duplicate`. Each finish's fingerprint is built from the package's
`read_inventory`, and its `problems` is `{}` for all four launches.

**The pool's prune of a replicated class.** None. No launch printed a prune of a replicated class.
After every launch each shard's `BackgroundModel` row is validated, and the census finds the
replicated tables equal (§3.3.5).

#### 3.3.3 The version row

From `closed_store_state.py`, after every launch: exactly one row, `(1, '2025.1.1')`, on each of the
four shards of both stores. The primary has no `version` table.

#### 3.3.4 The comparison

**The two stores**, by `compare_stores.py`
([`fingerprint_comparison_08.txt`](dk-and-rpk-migration/rehearsal/fingerprint_comparison_08.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-verification/compare_stores.py \
    var/datastores/dk-and-rpk-migration-08-reference.sqlite \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite
```

Exit 0:

```
reference: …/var/datastores/dk-and-rpk-migration-08-reference.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches
resumed: …/var/datastores/dk-and-rpk-migration-08-resumed.sqlite
  format 6, digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
  problems none
  against its sidecar: 0 difference(s), it matches

compare_fingerprints(reference, resumed): 0 difference(s)
verdict: equal
```

All 21 classes are `equal`. With the prompt number normalised, the output equals R1's
`fingerprint_comparison_05.txt` line for line.

**Against R1's reference, R0's reference, 07's reference and 09's rehearsal.** R0's and R1's stores
were not opened. Each R2 store was fingerprinted read-only again and compared with the fingerprint
recorded in each earlier run's `status.json`, by
[`compare_with_recorded.py`](dk-and-rpk-migration/rehearsal/compare_with_recorded.py)
([`fingerprint_against_recorded_08.txt`](dk-and-rpk-migration/rehearsal/fingerprint_against_recorded_08.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/dk-and-rpk-migration-08-reference.sqlite \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite \
    --against var/runs/dk-and-rpk-migration-05-reference-20261011T192802 \
    var/runs/dk-and-rpk-migration-02-reference-20261011T143003 \
    var/runs/a3-v2-readiness-07-reference-20261004T012909 \
    var/runs/datastore-generic-09-rehearsal-20261006T152452
```

It exited 0, verdict `equal`. For each store there were 0 differences against:
- its sidecar;
- R1's reference (taken at `6861a9fc…`);
- R0's reference (taken at `a5392fd1…`);
- 07's reference (taken at `dbd8a511…`);
- 09's rehearsal (taken at `53d4e188…`).

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

#### 3.3.5 Integrity, both stores, read-only

**The census** ([`census_08.txt`](dk-and-rpk-migration/rehearsal/census_08.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/datastore-integrity-audit/replicated_serial_census.py \
    var/datastores/dk-and-rpk-migration-08-reference.sqlite \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite
```

- Every table is `OK` on both stores (28 lines), and the verdict is `no divergence` on both.
- With the prompt number normalised, the output equals R1's `census_05.txt` line for line.

**`integrity_check`, `foreign_key_check` and the inventory**
([`store_soundness_08.txt`](dk-and-rpk-migration/rehearsal/store_soundness_08.txt)), through this
campaign's [`store_soundness.py`](dk-and-rpk-migration/rehearsal/store_soundness.py) as 04 left it:

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/store_soundness.py \
    var/datastores/dk-and-rpk-migration-08-reference.sqlite \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite
```

- It exited 0, `=> clean`.
- On the primary and every shard of both stores, opened `mode=ro`, `PRAGMA integrity_check` is
  `['ok']` and `PRAGMA foreign_key_check` returns 0 rows.
- The package's `read_inventory(primary, factories)` reads shards `(0, 1, 2, 3)`, and reports no
  problem in any class of either store and 0 `duplicate`.
- Each class's record count is §3.3.4's table.
- With the prompt number normalised, the output equals R1's `store_soundness_05.txt` line for line.
- Every file of both stores had the same SHA-256 before and after these reads, and no journal
  appeared.

### 3.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance,
through S1's [`run_readers.sh`](dk-and-rpk-migration/rehearsal/run_readers.sh)
([`readers_08.txt`](dk-and-rpk-migration/rehearsal/readers_08.txt)). Every reader's queues are now
`raypoolkit`'s (07 rewrote their imports):

```bash
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite var/dk-and-rpk-migration-08/extract
bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh \
    var/datastores/dk-and-rpk-migration-08-resumed.sqlite var/dk-and-rpk-migration-08/extract \
    nomodels extract_Gk_data
```

Both invocations exited 0.

| Script | Exit | Wall | What it read (last queue's lookups) | Files written | `ReadOnlyMiss` / `ReadOnlyWrite` / traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 44 s | `GENERATING GkNumeric DATA PRODUCTS`: 116 | 441 | none | identical |
| `extract_GkWKB_data.py` | 0 | 192 s | `GENERATING GkNumeric/GkWKB DATA PRODUCTS`: 348 | 2436 | none | identical |
| `extract_GkSource_data.py` | 0 | 52 s | `GENERATING GkSource DATA PRODUCTS`: 10 | 174 | none | identical |
| `extract_TkWKB_data.py` | 0 | 23 s | `GENERATING TkNumeric/TkWKB DATA PRODUCTS FOR LambdaCDM`: 3 | 27 | none | identical |
| `extract_tensor_source_data.py` | **0** | 25 s | `EXTRACT TENSOR SOURCE DATA`: 6, all six `QuadSource` rows | 27 | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 28 s | `GENERATING QuadSourceIntegral DATA PRODUCTS`: 24 | 71 | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 48 s | LambdaCDM: 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 459 | none | identical |

- **The lookups equal R1's, reader by reader** (§2.4).
- **The files written differ**, as they did between R0 and R1. Against R1:
  - `extract_Gk_data.py`: 441 against 450, and 459 against 513 without `--models`;
  - `extract_GkSource_data.py`: 174 against 176;
  - `extract_QuadSourceIntegral_data.py`: 71 against 75.

  Each of these readers draws its subsample with `random.sample` (§2.4), so the count varies run to
  run. It is not a finding.
- **`extract_GkWKB_data.py`'s queue printed two progress lines**, against R1's three. Each is the
  package's, in R1's form:

  ```
     -- 2026-10-11 23:16:37 (1m 3.52s running): 188/348 work items remaining = 45.98% complete
        inflight: 0 lookup | completed: 160 lookup (avg 151.1/min)
     -- 2026-10-11 23:17:39 (2m 5.49s running): 98/348 work items remaining = 71.84% complete
        inflight: 0 lookup | completed: 250 lookup (87.21/min, avg 119.5/min)
  ```

  These are the only progress lines R2 printed anywhere (§3.5.6).
- **`extract_GkWKB_data.py`'s log also holds four kinds of line R1's did not.** They are two
  matplotlib `UserWarning`s from one `plot_Gk` worker, each with its source line:
  - `Data has no positive values, and therefore cannot be log-scaled.` (`ax.set_yscale("log")`);
  - `No artists with labels found to put in legend.` (`ax.legend(loc="best")`).

  The reader draws its source redshifts at random (`sample`, `extract_GkWKB_data.py:439-441`).
  Each plot task takes the two `object_get` refs straight from the task builder (`:448-479`), not
  through the work queue's results. So which `(k, z_source)` pairs are plotted, and whether one has
  no positive values, varies run to run, and the queue cannot change what a plot is given. The
  files written are 2436, R1's. Neither warning appears in any of R0's, R1's or 07's reader logs.
  Every other reader log has R1's line kinds exactly
  ([`log_comparison_08.txt`](dk-and-rpk-migration/rehearsal/log_comparison_08.txt), last section).
- **The load.** The readers ran 23:14:37–23:21:38. The one-minute load reached 252 at 23:18, from
  processes not of this repository; the top CPU users were OrbStack, `runningboardd` and the
  system's `deleted` cache daemon. Even so, every reader's wall time is at or under R1's.
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
  - The SHA-256s are those taken after `resumed-c` (§3.3.2's state, `after-resumed-c.sha256`).
  - No Ray process of this checkout was left after any reader.
- **The shard-count warning.** Each log opens `!! WARNING: number of shards read from database (=4)
  does not match specified number of shards (=10)`, as at R0, R1 and 07.
- **The fingerprint afterwards.** `store fingerprint` (without `--write`) on the resumed store gave
  `39809dca…`, problems none, "matches the fingerprint taken 2026-10-11T23:12:56+01:00 by run
  dk-and-rpk-migration-08-resumed-c-20261011T230717".

### 3.5 L4 — equal to R1

#### 3.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

```bash
./venv/bin/python docs/dk-and-rpk-migration/rehearsal/policy_statistics.py \
    var/runs/dk-and-rpk-migration-08-reference-20261011T224347
```

Exit 0. The reference run's block, lines 83–96 of its `stdout.log`, verbatim
([`policy_statistics_08_reference.txt`](dk-and-rpk-migration/rehearsal/policy_statistics_08_reference.txt)):

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

**Against R1's `policy_statistics_05_reference.txt`.**
- **Line for line, it differs.** `diff` of R1's block against R2's:

  ```
  10d9
  <            complete: 25 instances = 96.15% of this type, 8.59% of total
  11a11
  >            complete: 25 instances = 96.15% of this type, 8.59% of total
  ```

  Under `>> mixed`, R1 printed `complete` then `minimal`; R2 prints `minimal` then `complete`.
- **Sorted, the blocks are equal.**
- **R2's block equals R0's `policy_statistics_02_reference.txt` line for line.**

This is `[02-the-policy-statistics-block-orders-its-lines-by-completion]` (§1.5.2): the order is
Ray's completion order, and the counts are the measurement.

| | R1 `reference` | R2 `reference` |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| Order under `mixed` | `complete`, `minimal` | `minimal`, `complete` |

**R2's own resumed runs**, against R1's same slug (saved as
[`policy_statistics_08_resumed-b.txt`](dk-and-rpk-migration/rehearsal/policy_statistics_08_resumed-b.txt)
and [`policy_statistics_08_resumed-c.txt`](dk-and-rpk-migration/rehearsal/policy_statistics_08_resumed-c.txt)):
- **`resumed-b`**, which computed all 291 policy rows (block at `stdout.log:87`), has every count
  R1's `resumed-b` had. It prints `minimal` first, where R1's `resumed-b` printed `complete` first.
  Sorted, the two are equal. It equals R2's reference line for line.
- **`resumed-c`**, which computed none (block at `stdout.log:83`), equals R1's `resumed-c` line for
  line, `minimal` first.
- **`resumed-a`** has no block, and the script exits 1 on it, as at R0 and R1.

#### 3.5.2 The reference run's stage counts

From [`stage_counts_08.txt`](dk-and-rpk-migration/rehearsal/stage_counts_08.txt), against R1's
`stage_counts_05.txt`, stage by stage:

| Stage | R1 `reference` | R2 `reference` |
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

Equal in every stage. The `resumed-a` column is also R1's, entry for entry, since both were killed
in the same stage.

#### 3.5.3 The resumed chain, by identity

The raw counts of `resumed-b` and `resumed-c` depend on where each launch was killed, and R2 chose
its own points (§3.1.2). So L4 compares what holds whatever the point:

| Identity | R1 | R2 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated (`prune_check.py`) | 45 = 45, all absent | 0 = 0, nothing to check | ✅, vacuously at R2 |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | 235 = 291 − 101 + 45 | pruned 0; computed 291 − 0 + 0 = 291 | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 530 = 864 − 334 | 583 = 864 − 281 | ✅ |
| Totals per class over the chain equal the reference's | `GkSource` 101 + 235 − 45 = 291; QSI 334 + 530 = 864 | `GkSource` 0 + 291 − 0 = 291; QSI 281 + 583 = 864 | ✅ |

**R2 does not exercise the `GkSource` prune at open** (§3.3.2). The second identity holds only in
that 0 = 0. R1 exercised that path with the same `datastorekit` 0.2.2.

What each resumed launch pruned and recomputed, beside §2.5.3:

| Launch | Pruned at open | Computed, stage by stage | Of which recomputing a pruned row |
|---|---|---|---|
| `resumed-a` | nothing (new store) | everything up to the `GkSource` stage, as `reference`; **0 `GkSource` stored** at the signal | — |
| `resumed-b` | **nothing** (no unvalidated row) | `GkSource` 291; `GkSourcePolicyData` 291; `QuadSourceIntegral` 281 stored at the signal; every earlier stage 0 | — |
| `resumed-c` | nothing | `QuadSourceIntegral` 583; every other stage 0 | — |

No replicated class was pruned at any open. No `QuadSourceIntegral` row was pruned: the class has no
`validated` column.

#### 3.5.4 The stage sequence

The lines of each reference `stdout.log` that begin `** `
([`log_comparison_08.txt`](dk-and-rpk-migration/rehearsal/log_comparison_08.txt)):
- R1's and R2's each hold 26 such lines.
- With the prompt number, the repository path and every number normalised, the two sequences are
  equal, in order.
- Of the 26, 15 are the pipeline's stage banners, `** BUILD ARRAY OF WAVENUMBERS TO SAMPLE` through
  `** CALCULATE QUADRATIC SOURCE INTEGRALS`. They are equal verbatim, in order, with nothing
  normalised. The other 11 are the launcher's own lines.

#### 3.5.5 The stage times

Not compared (`ALL WORK ITEMS COMPLETE in time …` in each `stdout.log`):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1 m 48.6 s | 1 m 26 s | 1.69 s | 5.13 s |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3 m 15.7 s | 2 m 50.7 s | 3.49 s | 9.17 s |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **41.8 s**, no progress line | killed | 30.4 s | 8.34 s |
| APPLY GKSOURCE POLICIES | 23.1 s | — | 14.2 s | 8.4 s |
| QUADRATIC SOURCE INTEGRALS | **4 m 59.2 s**, no progress line | — | killed, no progress line | 4 m 59 s |

- Each earlier stage took under 2 s.
- The reference took 11 m 25 s, against R1's 12 m 13 s and R0's 8 m 0 s.
- **The QSI stage, at 299.2 s, came within 0.8 s of the 300 s** at which its queue may first print
  a progress line, and printed none. `resumed-c`'s QSI stage, 583 rows in 299 s, printed none
  either.

#### 3.5.6 The printed lines and the registry's fields

RayPoolKit's checklist item 5 says the pool's printed lines are an interface: `StageTracker` and
`interrupt_when.py` read them. Against R1's same slug
([`log_comparison_08.txt`](dk-and-rpk-migration/rehearsal/log_comparison_08.txt)):

| Line, in `stdout.log` | `reference` R2 / R1 | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| `** ` (banners and the launcher's lines) | 26 / 26 | 23 / 23 | 26 / 26 | 26 / 26 |
| `-- ALL WORK ITEMS COMPLETE in time …` | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `Queue summary: …`, one after each completion line | 10 / 10 | 7 / 7 | 9 / 9 | 10 / 10 |
| `(no compute items were evaluated - all assets read from datastore)` | 1 / 1 | 1 / 1 | 7 / 7 | 9 / 9 |
| progress: `n/m work items remaining = p% complete` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| progress: `inflight: … \| completed: …` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- **Every completion line is followed by its `Queue summary` line**, in all four.
- **The forms are R1's.** Normalised as §3.2.5, the completion lines of R2's four runs make 2
  kinds, the `Queue summary` lines 7 and the `(no compute items …)` line 1. Each set equals R1's.
- **No build printed a progress line**, as at R1: every stage finished, or was killed, inside the
  queues' 300 s minimum interval. **The progress lines R2 did print** are the GkWKB reader's two
  (§3.4). They are in the form of R1's reader and of the copy's last build line
  (`var/runs/a3-v2-readiness-07-reference-20261004T012909/stdout.log:100-101`): `-- <stamp>
  (<elapsed> running): n/m work items remaining = p% complete`, then `inflight: … | completed: …`.
  No registered run printed one, so no `status.json` `progress` field was filled from one, at R1 or
  R2.
- **The registry read them.** Each run's `status.json` at its end, R2 against R1:

| Slug | `state` | `stage` | `progress` | R1's `stage` |
|---|---|---|---|---|
| `reference` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | `null` | the same |
| `resumed-a` | `killed` | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | `null` | the same |
| `resumed-b` | `killed` | `CALCULATE QUADRATIC SOURCE INTEGRALS` | `null` | the same |
| `resumed-c` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | `null` | the same |

  The killed runs' `stage` is the stage they were killed in, which the interrupter read from it.
  No field R1 filled is empty at R2.

### 3.6 What was expected not to happen

- **An unvalidated `BackgroundModel`.** Did not happen. After every launch each shard's
  `BackgroundModel` row is validated (§3.3.2), and no resumed launch printed a prune of it.
- **An in-flight record left by a killed launch.** Did not happen. `replication_in_flight` had 0 rows
  after every launch, the two killed ones included.
- **A repair by the check at open.** Did not happen. No `stdout.log` or launch log holds `reconcil`,
  `repair`, `in.flight` or `replicat` (§3.2.5).
- **A Ray agent outliving a killed driver.** Did not happen (§3.2.3).
- **Anything the pool raises that the copy did not.** Did not happen:
  - no `ValueError` from a constructor, in any build or reader;
  - no traceback in any `stderr.log` or launch log;
  - no reader that refused or failed;
  - no `ReadOnlyMiss` or `ReadOnlyWrite`.
- **A printed line the registry or the interrupter could not read.** Did not happen (§3.5.6).

### 3.7 Issues opened

None. Every difference from R1 is explained by:
- where a launch was killed;
- Ray's completion order (an issue already open);
- the readers' random subsamples;
- or the load.

`resumed-a`'s storing no `GkSource` row is an observation, not an issue: nothing in the record
points at the pool (§3.3.2).

### 3.8 The suites, and R0's and R1's files

**The suites.** Serially, in the foreground, after the last launch, the readers and the guard, with
README §7's command; Datastore in four groups. Every count is the dispatch baseline.

| Root | Ran | Result | Load (1 min) |
|---|---|---|---|
| AdaptiveLevin | 32 | OK | 25.6 |
| CosmologyModels | 39 | OK | 25.6 |
| LiouvilleGreen | 148 | OK (1 skipped) | 23.7 |
| RunRegistry | 286 | OK | 19.8 |
| ComputeTargets | 693 | OK | 42.6 |
| Datastore `[a-f]` / `[g-o]` / `[p-r]` / `[s-z]` | 54 / 103 / 101 / 62 = 320 | OK | 6.6 / 6.2 / 8.2 / 8.5 |

`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which failed at dispatch, passed here.

**R0's and R1's 24 files** (`var/datastores/dk-and-rpk-migration-{02,05}-*`), SHA-256, mtime
(to the nanosecond) and size, were taken before R2's first launch (22:42) and after its last step
(23:31). They are identical, and each equals the orchestrator's dispatch table. R0's twelve are §2.8's
table. R1's twelve:

| File | SHA-256 | mtime |
|---|---|---|
| `-reference.sqlite` | `6f01ba6041cb3b2fca06…` | 19:28:30 |
| `-reference-shard0000.sqlite` | `1ae3eec6bc37799547d8…` | 19:40:08 |
| `-reference-shard0001.sqlite` | `6151d1d673c3e170c5ef…` | 19:40:08 |
| `-reference-shard0002.sqlite` | `09689a714bba7ddb1288…` | 19:28:29 |
| `-reference-shard0003.sqlite` | `cb2eeb797cbb4e67783f…` | 19:40:08 |
| `-reference.manifest.json` | `bbde995d5c0d0a181369…` | 19:41:08 |
| `-resumed.sqlite` | `19cd8373ae0a45d62eed…` | 19:50:49 |
| `-resumed-shard0000.sqlite` | `47ec005237b142280f6a…` | 19:54:54 |
| `-resumed-shard0001.sqlite` | `430dba4def831f9fdfce…` | 19:54:54 |
| `-resumed-shard0002.sqlite` | `b12a4bf7df09a2e49deb…` | 19:42:14 |
| `-resumed-shard0003.sqlite` | `47338a1922fe06adc06c…` | 19:54:54 |
| `-resumed.manifest.json` | `30252d1d9f2a3c9839ec…` | 19:54:57 |

The full digests are in [`prelaunch_checks_08.txt`](dk-and-rpk-migration/rehearsal/prelaunch_checks_08.txt).
No `-journal`, `-wal` or `-shm` file is under `var/datastores/`.

### 3.9 What stays under `var/`

- `var/runs/dk-and-rpk-migration-08-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`. Each holds
  the run's manifest, `status.json` with its full fingerprint, `stdout.log` and `stderr.log`. For
  `resumed-a` and `resumed-b` it also holds `interrupter.log`.
- `var/runs/dk-and-rpk-migration-08-<slug>.launch.log`, beside each run directory.
- `var/datastores/dk-and-rpk-migration-08-{reference,resumed}.sqlite`, four shards each, and each
  store's `.manifest.json` sidecar. Shard 2 of each, 1 077 248 bytes, holds only replicated rows.
- `var/dk-and-rpk-migration-08/`, which holds:
  - the agent's working notes (`notes.txt`);
  - the pre-launch captures and dummy-test outputs (`prelaunch/`);
  - the closed-store states and hashes, and each run's state after it (`state/`);
  - the check outputs (`checks/`, among them every run's statistics block);
  - R0's and R1's hashes after the last step (`r0_r1_hashes_after.txt`);
  - RayPoolKit's measurement as first written (`08-raypoolkit-measure_client_pools.md`);
  - the readers' outputs and logs (`extract/`, `readers.txt`).

### 3.10 G-RPK — RayPoolKit's G2

RayPoolKit's G2 (README §0; its checklist item 8) holds when its five parts hold (prompt 08 §2).
Each rests on a named commit or run:

| Part | What | Rests on | Holds |
|---|---|---|---|
| 1 | `RayTools/` is gone | `643c3b2` (prompt 07, reviewed, board §5.7): `git ls-tree 643c3b2 RayTools` is empty, and so is `git ls-tree 5631f0e RayTools`. Re-checked at `5631f0e`: no `RayTools*` path in the tree, `import RayTools` raises `ModuleNotFoundError` in a fresh `python -I` (§3.1.1) | ✅ |
| 2 | 0 imports and 0 `import_module` calls naming it; every one of the 54 sites `constructs` | RayPoolKit's `measure_client_pools.py sgk --commit 5631f0e`, run from RayPoolKit's root (clean, `c9b0e11`) with its own venv, saved as [`measure/out/08-raypoolkit-measure_client_pools.md`](../prompts/dk-and-rpk-migration/measure/out/08-raypoolkit-measure_client_pools.md): `RayTools/` absent, **0 imports**, **0 `import_module` / `__import__` calls**, 2 string literals, **54 calls, `constructs` 54**, `ray.is_initialized()` false after. It equals 07's saved run but for the commit, the tracked-file count (1474 against 1471) and the non-`.py` lines naming `RayTools` or `RayWorkPool` (651 against 475), all 176 more of them in `prompts/` | ✅ |
| 3 | The pin is in place and the venv installed from it | `b6cc119` (prompt 06, reviewed, board §5.6): `requirements.txt:76`. Re-checked at `5631f0e`: `pip show` 0.2.0 in this venv, `pip check` clean. All four R2 manifests record `raypoolkit` 0.2.0 at `eaef0ce6…`, `editable: false` (§3.2.4) | ✅ |
| 4 | The six roots pass | This prompt's suites at `5631f0e`, whose code is `643c3b2`'s but for one test comment: Datastore 320, RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), all OK (§3.8) | ✅ |
| 5 | R2 reproduces `39809dca…` | The runs `dk-and-rpk-migration-08-reference-20261011T224347` and `dk-and-rpk-migration-08-resumed-c-20261011T230717`, both `done`, both `39809dca…` with all 21 classes. They are equal to each other, and to R1's, R0's, 07's and 09's (§3.3.4) | ✅ |

**RayPoolKit's G2 holds.** Recording it on RayPoolKit's own extraction board is a commit in that
repository, not here (README §6.2 C12).

### 3.11 `main.py:3247`

**What changed (C8; RayPoolKit checklist item 5).** In the store branch of the pool's loop, where an
item has no validation handler, the post handler is the item's exit:
- SGK's copy called `self._post_handler(obj)`, the driver's local object from before the persist
  (`RayTools/RayWorkPool.py:490` at `789b72d`);
- `raypoolkit` 0.2.0 calls `self._post_handler(replacement_obj)`, the object the store returned
  (`raypoolkit/RayWorkPool.py:529-530`).

Every other call of the post handler passes `obj`, in both: the lookup exit (`:402`), the blob exit
(`:439`), the `available` branch (`:458`) and the validate branch (`:554`). A diff of the two
files gives nothing else that runs but:
- the constructor's checks (`:109-142`);
- the in-flight keys, `ref.hex()` for `ref.hex`;
- `results` sized by `_num_total_items`.

The loop's scheduling is the same.

**The queue it reaches.** `GkSourcePolicy_queue` (`main.py:3247-3262`, `APPLY GKSOURCE POLICIES`) has
`compute_handler=compute_GkSourcePolicy_work`, `validation_handler=None`,
`post_handler=postprocess_GkSourcePolicy` and `store_results=False`. So every policy row it
computes leaves the loop through `:529-530`, and every row it reads back leaves through the lookup
exit with `obj`, as before.

**What the handler reads.** `postprocess_GkSourcePolicy` (`main.py:2871-2893`):
- reads `obj.policy`;
- reads `policy.label`, or `policy.store_id` if the label is `None`;
- reads `obj.type` and `obj.quality`;
- counts them into `GkSourcePolicy_statistics`;
- keeps no reference to `obj`, and returns `None`.

The block at `main.py:3265-3278` prints those counts.

**What the persisted object is**, traced through the installed `datastorekit` 0.2.2 (the orchestrator's
note, correction 5; 07's log, item 5):
- `ShardedPool.object_store` (`datastorekit/SQL/ShardedPool.py:3441`) routes the object to one shard,
  since `GkSourcePolicyData` is sharded on `k` (`config/sharding.py:32`). It returns
  `[self._shards[shard_id].object_store.remote(item)]` (`:3531`).
- The shard's `Datastore.object_store` (`datastorekit/SQL/Datastore.py:661`) returns what the
  factory's `store` returns (`:733`).
- `sqla_GkSourcePolicyData_factory.store` (`Datastore/SQL/ObjectFactories/GkSourcePolicyData.py:150-174`)
  inserts the row, sets `obj._my_id` (`:172`) and returns `obj` (`:174`).
- The pool's `ray.get(ref)` (`raypoolkit/RayWorkPool.py:495`) gives the pickled copy, which must
  satisfy `available` (`:498`).

So the persisted object is the local object as Ray carried it to the actor and back, with `store_id`
set. `policy`, `type` and `quality` read `_policy`, `_type` and `_quality`
(`ComputeTargets/GkSourcePolicyData.py:584-600`). `_type` and `_quality` are set by `store()` before
the persist (`:574-575`). No class on that path defines `__getstate__`, `__setstate__` or
`__reduce__`: `git grep` finds none under `ComputeTargets/` or `Datastore/`. In the installed
`datastorekit` the only ones are in `replication.py` and `SQL/schema.py`, on classes of their own.
**By reading, the handler counts what it counted.**

**The measurement.** The statistics block of every launch that computed the policy rows, against
R1's same slug (§3.5.1):

| Launch | Policy rows computed (through `:529-530`) | Block's counts | R1's same slug | Equal |
|---|---|---|---|---|
| `reference` | 291 (stage counts 291/291/291) | 291; numeric 109 (109 complete); mixed 26 (25 complete, 1 minimal); WKB 156 (156 complete) | the same | ✅ counts; ✅ sorted lines; order differs under `mixed` |
| `resumed-b` | 291 (291/291/291) | the same | the same | ✅ counts; ✅ sorted lines; order differs under `mixed` |
| `resumed-c` | 0 (0/0/0; all 291 read back through the lookup exit, with `obj`) | the same | the same | ✅ line for line |

- In `reference` and `resumed-b` the handler was called 291 times with the persisted object, since
  the stage stored 291 and the block totals 291. **Every count is R1's.**
- **`resumed-c`'s block does not exercise C8.** It read every row back, and the handler got `obj`
  from the lookup exit, as with the copy.
- **The order** of `minimal` and `complete` under `mixed` is Ray's completion order
  (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`), and has now been seen both
  ways under the copy (R0, R1) and under the package (R2's `minimal` first, R1's `complete` first).
  It is not C8's.

**C8 is measured, and changes nothing the block records.** No issue is opened.

---

## 4. The close

**§4's tree.** Prompt 09 ran on branch `handover-remedial` at **`b95bf19`** ("Write the orchestrator note
for dk-and-rpk-migration prompt 09"), whose code is `f539015`'s (prompt 08): `ede4062` and `b95bf19`
touch only `prompts/`. Its measurements were taken there, because both packages' scripts read the
committed tree through `git show`. 09's own changes (§4.3) are comment text and six `docs/` import
statements, and the orchestrator re-runs both measurements at 09's commit. Before the commit, both
scripts were also run on the working tree as a commit object (`git stash create`, `43458c2`, tracked
changes only), and each differs from the saved output only by the commit line. `git status --short` was
empty at the start, nothing was `running` in the registry (29 runs: 27 finished, 2 `unknown`), and no
Ray process was up (`RAY_ADDRESS` unset; `pgrep -lf 'gcs_server|raylet|ray::'` found nothing).

**Headline.**
1. **Both packages' G2 hold**, and §4.4 and §4.5 draft each record for its own repository (C12). Neither
   is committed there.
2. **Nothing outside `prompts/` and dated `docs/` names a deleted layer file as a current path** (§4.3.2):
   the regex printed 23 lines before 09's sweep and none after.
3. **Both packages' measurements are as expected** (§4.2): DatastoreKit's script finds the 15 layer files
   absent and **0 imports, 0 string literals and 0 `import_module` calls outside `prompts/`**; RayPoolKit's
   finds `RayTools/` absent, **0 imports**, 0 `import_module` calls and **54 calls, all `constructs`**.
4. **The six roots are at 08's counts** (§4.2.3), and the venv is 06's, byte for byte (§4.2.4).
5. **The four inherited issues are handed over** (§4.3.1): three transferred whole, one narrowed.

### 4.1 R0, R1 and R2

| | R0 (prompt 02) | R1 (prompt 05) | R2 (prompt 08) |
|---|---|---|---|
| **SGK tree** | `a5392fd`, code `f54f1c5`'s | `6861a9f`, code `0c9e2cf`'s | `5631f0e`, code `643c3b2`'s but for one test comment |
| **Layer** | SGK's own, `b510bc9`'s | `datastorekit` 0.2.2 | `datastorekit` 0.2.2 |
| **Work queue** | SGK's own `RayTools/` | SGK's own `RayTools/` | `raypoolkit` 0.2.0 |
| **`packages` in every manifest** | `{"datastorekit": null, "raypoolkit": null}` | `datastorekit` 0.2.2 at `6d8ee9ae…`, `editable: false`; `raypoolkit` `null` | `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit` 0.2.0 at `eaef0ce6…`, both `editable: false` |
| **`git_dirty`, every manifest** | `false` | `false` | `false` |
| **Reference fingerprint** | format 6, `39809dca…`, 21 classes | the same | the same |
| **Rehearsal stores** | `var/datastores/dk-and-rpk-migration-02-{reference,resumed}.sqlite` | `…-05-…` | `…-08-…` |

`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2` is the reference digest in every
row. The twelve runs, read from each run directory's `manifest.json` and `status.json` for this record
(`state`, `exit_code`, the first eight characters of the fingerprint's digest, `git_head`):

| Run | `state` | `exit_code` | Fingerprint |
|---|---|---|---|
| `dk-and-rpk-migration-02-reference-20261011T143003` | `done` | 0 | `39809dca` |
| `dk-and-rpk-migration-02-resumed-a-20261011T143917` | `killed` | 143 | `eaf590af` |
| `dk-and-rpk-migration-02-resumed-b-20261011T144253` | `killed` | 143 | `fc02616e` |
| `dk-and-rpk-migration-02-resumed-c-20261011T144549` | `done` | 0 | `39809dca` |
| `dk-and-rpk-migration-05-reference-20261011T192802` | `done` | 0 | `39809dca` |
| `dk-and-rpk-migration-05-resumed-a-20261011T194149` | `killed` | 143 | `b6b89d84` |
| `dk-and-rpk-migration-05-resumed-b-20261011T194703` | `killed` | 143 | `09170e40` |
| `dk-and-rpk-migration-05-resumed-c-20261011T195044` | `done` | 0 | `39809dca` |
| `dk-and-rpk-migration-08-reference-20261011T224347` | `done` | 0 | `39809dca` |
| `dk-and-rpk-migration-08-resumed-a-20261011T225623` | `killed` | 143 | `ff6c4257` |
| `dk-and-rpk-migration-08-resumed-b-20261011T230305` | `killed` | 143 | `aa860ed8` |
| `dk-and-rpk-migration-08-resumed-c-20261011T230717` | `done` | 0 | `39809dca` |

Each `killed` run's digest differs from its namesake's in the other rehearsals, as it must: each digests
a store killed at a different point (§1.2.1, §2.2.1, §3.2.1). Each rehearsal's `reference` and
`resumed-c` end `done`/0 at `39809dca…`, and equal each other and every earlier rehearsal's.

### 4.2 The final measurements (Z4)

Saved under [`prompts/dk-and-rpk-migration/measure/out/`](../prompts/dk-and-rpk-migration/measure/out/),
names beginning `09-`.

#### 4.2.1 DatastoreKit's `measure_client_imports.py`

`cd /Users/ds283/Documents/Code/DatastoreKit && ./venv/bin/python docs/extraction/measure_client_imports.py sgk --commit b95bf19`,
DatastoreKit clean at `83068cb`. DatastoreKit has since released `v0.2.3`; SGK stays on `v0.2.2`, and the
script is unchanged since `v0.2.2` (`git diff v0.2.2 HEAD -- docs/extraction/measure_client_imports.py` is
empty), so this is the tag's script. Saved as
[`09-datastorekit-measure_client_imports.md`](../prompts/dk-and-rpk-migration/measure/out/09-datastorekit-measure_client_imports.md).

| | Count |
|---|---|
| Layer files absent | **15** of the 15 deleted; `Datastore/__init__.py` and `Datastore/SQL/__init__.py` present, 0 bytes |
| Tracked `.py` files read | 423, 0 unparsed |
| Import statements naming a layer module | **51 in 31 files, all in `prompts/`**; **0 outside** |
| String literals naming a layer module | 20, all in `prompts/`; 0 outside |
| `import_module` / `__import__` calls naming a layer module | 1, in `prompts/`; 0 outside |
| Import statements naming a factory module | 99 (97 outside `Datastore/SQL/ObjectFactories/`; 15 in `docs/`) |

So the checklist's "0 imports" reads "0 outside `prompts/`", the record of the campaigns that wrote the
layer. The output equals 04's saved
[`04-datastorekit-measure_client_imports.md`](../prompts/dk-and-rpk-migration/measure/out/04-datastorekit-measure_client_imports.md)
but for the commit line and 425 → 423 files read (07 deleted `RayTools/`'s two). The 99 factory imports
are 04's.

#### 4.2.2 RayPoolKit's `measure_client_pools.py`

`cd /Users/ds283/Documents/Code/RayPoolKit && ./venv/bin/python docs/adoption/measure_client_pools.py sgk --commit b95bf19`,
RayPoolKit clean at `c9b0e11`, verdicts by `raypoolkit 0.2.0`'s own constructor. Saved as
[`09-raypoolkit-measure_client_pools.md`](../prompts/dk-and-rpk-migration/measure/out/09-raypoolkit-measure_client_pools.md).

| | Count |
|---|---|
| `RayTools/` at the commit | absent |
| Imports of a `RayTools` module | **0** |
| `import_module` / `__import__` calls naming `RayTools` | **0** |
| String literals naming `RayTools` | 2: `docs/tolerance-convergence/inventory.py:382` (kept, C10) and one in `prompts/` |
| `RayWorkPool(...)` calls | **54**, verdicts `constructs` **54**, no flag |
| `ray.is_initialized()` after the verdicts | False |
| Tracked files; `.py` parsed | 1496; 425, 0 unparsed |
| Lines of non-`.py` files naming `RayTools` or `RayWorkPool` | 793 |

The output equals 08's saved
[`08-raypoolkit-measure_client_pools.md`](../prompts/dk-and-rpk-migration/measure/out/08-raypoolkit-measure_client_pools.md)
but for the commit (`5631f0e` → `b95bf19`), the tracked-file count (1474 → 1496) and the non-`.py` lines
(651 → 793). The 142 new lines are 23 in `docs/` (this record's §3 and the rehearsal outputs) and 119 in
`prompts/`.

#### 4.2.3 The six roots

Serially, in the foreground, README §7's command, on the working tree with 09's changes (Z2, Z3, Z7 and
the records), at `b95bf19` plus those. Datastore in the four groups. Each full output is saved as
`09-suite-<root>.txt` in the directory above.

| Root | Ran | 08's | Result | Load (1 min) at the start |
|---|---|---|---|---|
| AdaptiveLevin | 32 | 32 | OK | 43.8 |
| CosmologyModels | 39 | 39 | OK | 43.8 |
| LiouvilleGreen | 148 | 148 | OK (1 skipped) | 44.1 |
| RunRegistry | 286 | 286 | OK | 54.4 |
| ComputeTargets | 693 | 693 | 1 failure, below | 24.8 |
| Datastore `[a-f]` / `[g-o]` / `[p-r]` / `[s-z]` | 54 / 103 / 101 / 62 = 320 | 320 | OK | 8.2 / 7.9 / 7.3 / 7.5 |

**The one failure** is `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, README §7's known
wall-clock flake: `0.0613 not less than or equal to 0.06`. Re-run alone at load 8.2, three times, it passed,
passed, and failed once (`0.0658 not less than or equal to 0.06`). The assertion has
a margin of about one, nothing 09 changed is on its path (comments and six `docs/` statements), and it
passed in R1's and R2's root runs (§2.8, §3.8). `black --check` is clean on the 24 Python files 09
touches.

#### 4.2.4 The venv

`pip freeze` is saved as [`09-pip-freeze.txt`](../prompts/dk-and-rpk-migration/measure/out/09-pip-freeze.txt)
and `diff`s empty against 06's `var/dk-and-rpk-migration-06/freeze-after.txt` (100 lines each).
`pip check` reports no broken requirements. `RunRegistry.package_provenance()`, saved as
[`09-package-provenance.json`](../prompts/dk-and-rpk-migration/measure/out/09-package-provenance.json):

- `datastorekit` 0.2.2, commit `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`, requested revision `v0.2.2`,
  `editable: false`;
- `raypoolkit` 0.2.0, commit `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, requested revision `v0.2.0`,
  `editable: false`.

### 4.3 What 09 changed

#### 4.3.1 The hand-over of the four inherited issues (Z1, C11)

DatastoreKit's `docs/OPEN_ISSUES.md` §1.2 (at `83068cb`) lists all four, with SGK's board for each.
- **Three transferred whole** to their boards' §4, each entry's text verbatim, its heading noting
  "transferred 2026-10-12 to DatastoreKit by dk-and-rpk-migration 09", and a line saying the issue is
  **still open there**: a transfer is not a resolution.
  - `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` and
    `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`
    (`prompts/a3-v2-readiness`);
  - `[00-the-inventory-cannot-see-a-serial-split]` (`prompts/datastore-integrity`).
- **One narrowed**, at the user's decision, and kept on `prompts/store-retirement`'s §3:
  `[01-cross-filesystem-move-advice-says-delete-by-hand]`. The advice text is the package's
  (`ShardedPool._failure_message`, installed `datastorekit/SQL/ShardedPool.py:3026-3030`, with `delete_store`
  at `:2704`); `RunRegistry.stores.move_store` (`RunRegistry/stores.py:1109-1123`) relays it through
  `_store_files` (`:1052-1060`) with the sidecar listing appended, adding and shaping no advice. What stays
  SGK's is that `store move` passes on, for a registered store, advice that decision 6.1.1 forbids (a
  registered store's source is retired, never deleted by hand). The test that asserted the text moved to the
  package with 04.

#### 4.3.2 The sweep of prose (Z2) and the regex

`[04-comments-name-the-layers-old-module-paths]`: its 23 lines, in 18 files, each rewritten to name what is
current and nothing else (`datastorekit.store_inventory`, `.shard_paths`, `.SQL.schema`, `.SQL.Datastore._build_schema`;
`config.datastore.factories` for the `_factories` that left the layer at `19f07ee`; file paths
`datastorekit/…`). Three lines were re-wrapped, where the longer name passed 100 columns. For every one of
the 18 files, `ast.dump` of the module with docstrings removed is identical to `HEAD`'s.

The regex (the orchestrator's, over `git ls-files` outside `prompts/` and `docs/`, skipping import
statements and `import_module` calls; quoted verbatim in log 09) printed **23 lines before** the sweep,
exactly the board's, and **0 after**. `import RunRegistry.stores` in a fresh `python -I` still loads neither
`ray`, `sqlalchemy` nor `datastorekit.store_inventory` (`RunRegistry/stores.py:114`).

#### 4.3.3 The six `docs/` probes (Z7, chosen by the user on 2026-10-12)

`[04-six-docs-probes-import-factories-from-the-layer]`: in each of `docs/datastore-integrity-audit/fk_sweep_probe.py`,
`lookup_keys/common.py`, `lookup_keys/probe_03_column_existence.py`, `tq_fk_probe.py`,
`docs/tolerance-convergence/inventory.py` and `order_audit.py`, the one statement importing `_factories`
from `datastorekit.SQL.Datastore` is now `from config.datastore import factories as _factories`, and
nothing else changed. No probe was run. Each file's import statements alone were executed
(an `ast` extraction, log 09): before the change each of the six failed on its `_factories` line
(`ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore'`); after it each is
`all OK`, at 4, 6, 6, 4, 10 and 25 statements, and the six that fail through `lookup_keys/common.py`
(`probe_01`, `01b`, `04`, `04b`, `05`, `06`) are `all OK` at 14, 6, 9, 7, 14 and 7.

#### 4.3.4 `CLAUDE.md`'s test command (Z3)

The one `unittest discover` line for `ComputeTargets/tests` is now the command for each of the six roots,
and the Datastore root's four groups, with the reason for them. Nothing else in the section changed.

### 4.4 DatastoreKit's G2, drafted

For DatastoreKit's `prompts/extraction/IMPLEMENTATION_STATE.md` §2, row **G2**. It is its G1 cell's form
(`✅ <date>: …`), followed by the cell's existing notes **verbatim**. Not committed there (C12): the user
makes that commit, or has a session in that repository make it. "This commit" is SGK's 09 commit.
Each part of DatastoreKit's checklist, `docs/adoption/secondarygwkit.md` §9, rests on:

| Checklist part | Rests on |
|---|---|
| 1. the 17 layer files deleted, and the pinned tag imported | `0c9e2cf` (prompt 04, reviewed, board §5.4); the pin at `9ec603a` (prompt 03); §4.2.1, the script at `b95bf19` |
| 2. its remaining suites pass | §2.8 at `6861a9f` (R1); §4.2.3 at 09 |
| 3. a rehearsal reproduces the reference fingerprint | R1's runs `dk-and-rpk-migration-05-reference-20261011T192802` and `dk-and-rpk-migration-05-resumed-c-20261011T195044` (§2.3.4, §2.10); §4.1 |

````markdown
| **G2**: SGK adopted `v0.2.2` or later (U37; `actor-names` U3; `factory-defaults` U5), fingerprint reproduced | ✅ 2026-10-12: holds. SGK adopted `v0.2.2` (the pin at SGK `9ec603a`, installed at `6d8ee9ae…`, not editable) and a rehearsal through the package reproduces the reference fingerprint. Against the checklist, [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md) §9: (1) **the 17 layer files are deleted and the tag imported**: 15 removed and the two `__init__.py` emptied (U3) by SGK `0c9e2cf`; `docs/extraction/measure_client_imports.py sgk --commit b95bf19` finds the 15 layer files absent and, outside SGK's `prompts/` record, 0 import statements, 0 string literals and 0 `import_module` calls naming a layer module (the 51 statements, 20 literals and 1 call it counts are all in `prompts/`), unchanged at SGK's 09 commit (this commit); (2) **its remaining suites pass**: at SGK `6861a9f` (R1), AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 320 (54 + 103 + 101 + 62), all OK; at SGK's 09 commit the same counts, all OK but one wall-clock assertion, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, a known load-sensitive flake: it failed once in the root run at load 25 and once in three runs alone, passed in the other two, and passed in R1's and R2's root runs; (3) **a rehearsal rebuild through the package reproduces the reference fingerprint**, run under SGK's run-registry rules: R1's `dk-and-rpk-migration-05-reference-20261011T192802` and `dk-and-rpk-migration-05-resumed-c-20261011T195044`, both `done`, format 6, `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes equal, each manifest recording `datastorekit` 0.2.2 at `6d8ee9ae…`; equal to each other and to R0's (SGK `a5392fd`, before the swap), to `a3-v2-readiness` 07's and to `datastore-generic` 09's. R2 (SGK `5631f0e`, with both packages) reproduced it again. SGK's record is `docs/dk-and-rpk-migration-verification.md` §2.10 and §4 (campaign `dk-and-rpk-migration`). SGK stays on `v0.2.2`; moving to `v0.2.3` is the user's (`factory-defaults` U5). *(2026-10-09: the adoption checklist is [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md), measured at SGK `b510bc9` by 07a.)* *(2026-10-09: `v0.1.0` is made, annotated, on 05's commit `68db557` (tag object `0ece4aa`) and pushed, after CI passed there at both ends, [run 37922626418](https://github.com/ds283/DatastoreKit/actions/runs/37922626418): Python 3.12.15 / Ray 2.43.0 / SQLAlchemy 2.0.39 and 3.13.16 / 2.55.1 / 2.0.46, SQLite 3.45.1 at both, each `Ran 444 tests … OK`. SGK may now adopt it.)* *(2026-10-10: `v0.2.1` is ready to be tagged on extraction prompt 10's commit once CI passes there at both ends, as U23 and U28; 10 makes no tag and pushes nothing. SGK's addendum is [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md#addendum-v021-extraction-prompt-10-2026-10-10).)* *(2026-10-10: `v0.2.1` is made, annotated, on 10's commit `33778b0` (tag object `0ba2e4d`) and pushed, after CI passed there at both ends, [run 38054748108](https://github.com/ds283/DatastoreKit/actions/runs/38054748108): Python 3.12.15 / Ray 2.43.0 / SQLAlchemy 2.0.39 and 3.13.16 / 2.55.1 / 2.0.46, SQLite 3.45.1 at both, each `Ran 486 tests … OK`. SGK may now adopt it.)* *(2026-10-11: `v0.2.2` is ready to be tagged on `actor-names` prompt 02's commit once CI passes there at both ends, as U23 and U28; 02 makes no tag and pushes nothing. SGK's addendum is [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md#addendum-v022-actor-names-prompt-02-2026-10-11).)* *(2026-10-11: `v0.2.2` is made, annotated, on 02's commit `6d8ee9a` (tag object `c32d027`) and pushed, after CI passed there at both ends, [run 38131949852](https://github.com/ds283/DatastoreKit/actions/runs/38131949852): Python 3.12.15 / Ray 2.43.0 / SQLAlchemy 2.0.39 and 3.13.16 / 2.55.1 / 2.0.46, SQLite 3.45.1 at both, each `Ran 494 tests … OK`. SGK may now adopt it.)* *(2026-10-11: `v0.2.3` is ready to be tagged on `factory-defaults` prompt 02's commit once CI passes there at both ends, as U23 and U28; 02 makes no tag and pushes nothing. SGK may stay on `v0.2.2` or move to `v0.2.3`, as it chooses (`factory-defaults` U5). SGK's addendum is [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md#addendum-v023-factory-defaults-prompt-02-2026-10-11).)* *(2026-10-11: `v0.2.3` is made, annotated, on `factory-defaults` 02's commit `0de2b16` (tag object `8dac562`) and pushed, after CI passed there at both ends, [run 38177954330](https://github.com/ds283/DatastoreKit/actions/runs/38177954330): Python 3.12.15 / Ray 2.43.0 / SQLAlchemy 2.0.39 and 3.13.16 / 2.55.1 / 2.0.46, SQLite 3.45.1 at both, each `Ran 504 tests … OK`. SGK may adopt it, or stay on `v0.2.2` (`factory-defaults` U5).)* |
````

### 4.5 RayPoolKit's G2, drafted

For RayPoolKit's `prompts/extraction/IMPLEMENTATION_STATE.md` §2, row **G2**. It is its G1 cell's form
(`**holds** <date>, …`), keeping the checklist pointer. Not committed there (C12). Each part of its
checklist, `docs/adoption/secondarygwkit.md` §8, items 1-5 (item 6 is this recording), rests on:

| Checklist item | Rests on |
|---|---|
| 1. `RayTools/` is gone | `643c3b2` (prompt 07, reviewed, board §5.7); `git ls-tree HEAD RayTools` is empty at `b95bf19` |
| 2. no import of `RayTools` | §4.2.2, the script at `b95bf19`, run from RayPoolKit's root; and §3.10 at `5631f0e` |
| 3. the pin, and the venv installed from it | `b6cc119` (prompt 06, board §5.6), `requirements.txt:76`; §4.2.4; every R2 manifest |
| 4. SGK's suites pass | §3.8 at `5631f0e` (R2); §4.2.3 at 09 |
| 5. every site constructs | §4.2.2: 54 calls, `constructs` 54; R2's `main.py:3247` measurement, §3.11 |

````markdown
| **G2** | **holds** 2026-10-12: SGK adopted `v0.2.0`, and its five parts of the checklist's item 8 hold. (1) `RayTools/` is gone: `git ls-tree 643c3b2 RayTools` is empty (SGK's prompt 07), and so it is at `5631f0e` and `b95bf19`. (2) No import of it: `docs/adoption/measure_client_pools.py sgk --commit b95bf19`, run from this repository (clean, `c9b0e11`) with its own venv, gives 0 imports and 0 `import_module` / `__import__` calls naming `RayTools`, and 2 string literals, `docs/tolerance-convergence/inventory.py:382` (item 9's kept `"RayTools"`) and one in SGK's `prompts/` record; it is unchanged at SGK's 09 commit (this commit). (3) The pin is in place and the venv installed from it: `requirements.txt:76` at SGK `b6cc119`, `raypoolkit` 0.2.0 at `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, not editable, recorded in every R2 manifest. (4) SGK's suites pass: at SGK `5631f0e` (R2), AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693 (the five modules of item 6 among them), Datastore 320 (54 + 103 + 101 + 62), all OK; at SGK's 09 commit the same counts, all OK but one wall-clock assertion, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, a known load-sensitive flake: it failed once in the root run at load 25 and once in three runs alone, passed in the other two, and passed in R1's and R2's root runs. (5) Every site constructs: the same script gives 54 `RayWorkPool(...)` calls, every one `constructs`. R2, SGK `5631f0e`, reproduced the reference fingerprint `39809dca…` (21 classes) through both packages, in runs `dk-and-rpk-migration-08-reference-20261011T224347` and `dk-and-rpk-migration-08-resumed-c-20261011T230717`, and measured item 5 (`main.py:3247`): the counts the post handler records are unchanged. SGK's record is `docs/dk-and-rpk-migration-verification.md` §3.10 and §4 (campaign `dk-and-rpk-migration`). Checklist: [`docs/adoption/secondarygwkit.md`](../../docs/adoption/secondarygwkit.md), measured at SGK `b510bc9`; its item 8 is the acceptance |
````

### 4.6 Left for the user

Not issues, and not done here (README §6.3):
- retiring the six rehearsal stores `var/datastores/dk-and-rpk-migration-{02,05,08}-{reference,resumed}.sqlite`
  (README §3), after 09's review;
- the two G2 commits in DatastoreKit and RayPoolKit (§4.4, §4.5);
- whether to rename the `Datastore` package (U3), which now holds only the factories and the tests;
- whether SGK moves to `datastorekit` `v0.2.3`, released 2026-10-11 (`factory-defaults` U5). SGK stays pinned
  at `v0.2.2`, which DatastoreKit's G2 admits, and 09 moves no pin.
