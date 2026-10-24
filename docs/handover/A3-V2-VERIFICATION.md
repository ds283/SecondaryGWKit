# A3 v2 — verification record

This document is the `handover` campaign's record of the A3 v2 build: the host made ready and
rehearsed there (§1, prompt 06), the launch (§2, prompt 07), and the checks on the closed store
(§3, prompt 08). Each section is written by its prompt and is additive (`CLAUDE.md`, campaign
conventions, rule 6).

**Where the records are.** The prompts run on the Mac Studio and commit nothing there; each writes
under that checkout's `var/handover-NN/`, which the orchestrator copies here. **A path
`var/handover-06/<dir>/` in §1 is that directory on the Mac Studio.** Its `out/`, `roots/`,
`scripts/` and `state/` are committed at [`a3-v2/rehearsal/`](a3-v2/rehearsal/), and the links below
point there. `recorded/` (copies of R3's two run directories, which are on the laptop under
`var/runs/`), `work/` (the interrupter's dummy-test directories) and `extract/` (207 MB of reader
output) stay on the Mac Studio, as do R4's run directories (`var/runs/handover-06-*`) and its two
stores.

## 1. R4 — the Mac Studio

**§1's tree, host, venv, stores and runs.** Every R4 launch ran on the Mac Studio, **`mps026298`**, in
`/Users/ds283/Documents/Code/SecondaryGWKit`, at the dispatch SHA
**`3152af943a4f8dd5cf5920f7ed6b209f1c7e217b`** on `handover-remedial` ("Write the handover orchestrator
note for prompt 06").
- **Its code is `c081eed`'s.** `git log c081eed..HEAD` is `304960f` and `3152af9`, both prompts and
  notes; `git diff --stat c081eed HEAD -- . ':!prompts' ':!docs'` is empty.
- **What it changes over R3's tree (`aab4075`)** in what a build runs:
  - **the launcher**, `RunDrivers/quadsource_atol_sweep.py`, SHA-256
    `b2fa4ef1bbdbb09fea1780952710eeb53dee401ef6c0febcd385dd98aa0526ef`. It was moved from
    `docs/handover/` by `031bcb4`, which is its last change, and its refusals were made clean by
    `rrk-v020-adoption` 05. R3 ran `2a985303…`;
  - **`datastorekit` 0.5.0** at `f47b9a255e9ad58e0d0eaf85644f53b575884d31` (`requirements.txt:17`,
    `@v0.5.0`; `dk-v050-adoption`, `542ddd1`). R3 ran 0.4.0 at `c8570e8…`;
  - **the host** and the venv built on it, below.
- **The host** ([`host_06.txt`](a3-v2/rehearsal/out/host_06.txt)):
  - macOS 27.0.1 (26A434), Apple M1 Max, 10 CPUs, 32 GiB;
  - up 15 days, booted 2026-10-07 10:23:52; `pmset` `sleep 0`;
  - `df -h ~` 927 G, **122 G free, 87 % used** (R3's laptop volume: 72 G, 93 %).
  - The CPU at §1.1 was led by `corespotlightd` (141 %), `CategoriesService` (67 %) and OrbStack's
    helper (the website's object store; 10 %).
  - `runregistrykit.this_host()`: `id` **`d120d1cf03551e6f`**, `name` `mps026298.phys.susx.ac.uk`,
    `boot` **`C58A0F92-923F-471F-B32E-B1B9F5D0A848`**, `machine` `arm64`, `system` `Darwin`,
    `release` `27.0.0`, `python` `3.12.15`. The `id` and `boot` were the same at §1.7, before every
    launch and at every run's end: the machine did not reboot.
- **The checkout before.** On `main` at `9678f28`, clean. `git fetch origin` and
  `git switch handover-remedial` gave `HEAD` `3152af94…`, the dispatch SHA, and `git status --short`
  was empty.
- **R3's recorded run directories** were staged by the orchestrator at `~/handover-06-recorded/`
  (outside the checkout: `main`'s `.gitignore` does not ignore `var/`). After the switch they were
  moved to `var/handover-06/recorded/`. Each `manifest.json` and `status.json` has the laptop's
  SHA-256: `13e8fcaa…`, `a64cd267…`, `a0742b57…` and `3d30a3cf…`.
- **The old venv** ([`old_venv_06.txt`](a3-v2/rehearsal/out/old_venv_06.txt)) was Python **3.13.16**
  (`/opt/local/bin/python3.13 -m venv`), with `ray==2.45.0` and none of the three packages. It was
  **moved, not deleted**, to `~/Documents/Code/retired-venvs/SecondaryGWKit-py313-20261022`.
- **The new venv.** `/opt/local/bin/python3.12 -m venv venv` gave **Python 3.12.15**, the laptop's
  patch version, with pip 25.0.1.
  - `./venv/bin/pip install -r requirements.txt` exited 0, with pip not upgraded
    ([`pip_install_06.txt`](a3-v2/rehearsal/out/pip_install_06.txt)). It printed two `WARNING: The candidate
    selected … is a yanked version`, for `aiohttp` 3.11.14 and `aiohttp-cors` 0.8.0.
  - **`venv` was not ignored.** `git check-ignore -v venv` exited 1 and `git status --short` showed
    `?? venv/`. Nothing tracked ignores it: `.gitignore` names `var/` only, and `.git/info/exclude`
    did not name it. The old 3.13 venv had been ignored by its own `venv/.gitignore` (`*`), which
    `python3.13 -m venv` writes and 3.12's does not.
  - `runregistrykit` stamps `git_dirty = bool(git status --porcelain)`, so every manifest would have
    said `git_dirty: true`. Nothing was launched, and the orchestrator ruled: append `venv` to
    `.git/info/exclude`, as the laptop's line 40 does. Afterwards `git check-ignore -v venv` gives
    `.git/info/exclude:28:venv` and `git status --short` is empty.
  - **`pip freeze` is identical to the laptop's** second freeze in
    `prompts/dk-v050-adoption/measure/out/01-pip.txt`: 101 lines, `diff` exit 0
    ([`pip_06.txt`](a3-v2/rehearsal/out/pip_06.txt)). `pip check` is clean.
  - `runregistrykit.package_provenance(("datastorekit", "raypoolkit", "runregistrykit"))` gives
    `datastorekit` 0.5.0 at `f47b9a2…` (`v0.5.0`), `raypoolkit` 0.2.0 at `eaef0ce6…` (`v0.2.0`) and
    `runregistrykit` 0.2.0 at `0bae3a3c…` (`v0.2.0`), **all `editable: false`**. `ray` is 2.43.0.
- **The seven roots** at the dispatch SHA ([`roots_06.txt`](a3-v2/rehearsal/out/roots_06.txt)), serially, from
  19:46:43 to 19:50:09:

  | Root | Ran | Result | Load (1 min) at start | Wall |
  |---|---|---|---|---|
  | AdaptiveLevin | 32 | OK | 32.94 | 2 s |
  | ComputeTargets | 700 | OK | 32.94 | 146 s |
  | CosmologyModels | 39 | OK | 18.48 | 2 s |
  | LiouvilleGreen | 148 | OK (skipped=1) | 17.40 | 28 s |
  | ObjectFactories | **280** | OK | 14.10 | 14 s |
  | RunDrivers | 115 | OK **(skipped=1)** | 12.35 | 4 s |
  | tools | 91 | OK | 11.52 | 10 s |

  - **ObjectFactories 280** is `01-roots.txt`'s AFTER count (0.5.0) and the `dk-v050-adoption`
    board's "ObjectFactories at `HEAD`: 280". The prompt's 279 is that file's BEFORE count.
  - **RunDrivers' skip** is `test_pipeline_adoption.…test_the_stage_is_the_hand_written_manifests_own_field`,
    which skips when the gitignored `var/datastores/handover-A3-baseline-lambdacdm.manifest.json` is
    absent, as on any fresh `var/` (`RunDrivers/tests/test_pipeline_adoption.py:101-102`).
  - No test failed. The `ComputeTargets` wall-clock flake did not fire.
- **The registry and the lock before** (§1.7):
  - there was no `var/runs/`, and `list` said `no runs under …/var/runs`;
  - `~/ScienceData` did not exist, and the lock was `free`;
  - `ray-processes` listed nothing and `RAY_ADDRESS` was unset.
- **The tree was clean at every launch**, checked again in the same command that launched.
  `git status --short` is empty after the last step, at `HEAD` `3152af94…`.

R4's launcher built two throwaway stores:
- **`var/datastores/handover-06-reference.sqlite`** (sidecar `store_id`
  `d3ce4fae5a8b4121b6ec2c1bc016feb1`), built straight through by the run **`reference`**;
- **`var/datastores/handover-06-resumed.sqlite`** (`store_id` `1c41fb1357b542cc99b5c19cb8684dfb`),
  built by three runs:
  - **`resumed-a`**, killed by `SIGTERM` in the `GkSource` stage;
  - **`resumed-b`**, with `--resume`, killed by `SIGTERM` in the `QuadSourceIntegral` stage;
  - **`resumed-c`**, with `--resume`, run to `done`.

Neither store is retired: they are the user's to retire after the review, or not. R3 is compared
through the fingerprints its `reference` and `resumed-c` run directories recorded (copied to
`var/handover-06/recorded/`), through their `stdout.log` and `stderr.log`, and through R3's saved
outputs (`docs/rrk-v020-adoption/rehearsal/*_03.txt`) for everything else. R3's `resumed-a`,
`resumed-b` and launch logs are not on this machine.

**Headline.**
1. **R4 reproduces the reference fingerprint on the Mac Studio.**
   - Both stores are format 6, **`encoding_format` 1**, digest
     `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, with all 21 classes equal.
     Each equals its sidecar's.
   - `compare_fingerprints` is empty between the two stores. It is also empty between each store and
     **R3's recorded `reference` and `resumed-c`** (§1.3.4).
   - So the host, the venv, the launcher as it now stands and the `datastorekit` 0.5.0 pin change
     nothing a build writes or a fingerprint reads.
2. **Every launch ended as recorded.** `reference` and `resumed-c` are `done`/0, and `resumed-a` and
   `resumed-b` are `killed`/143. Each carries a fingerprint with `encoding_format` 1, and each driver
   exited as recorded (§1.2). No `status.json.*.tmp` was left.
3. **Every manifest and every status carries the Mac Studio's `host`**, `datastorekit` 0.5.0 at
   `f47b9a2…`, the launcher's path and `b2fa4ef1…`, `git_head` the dispatch SHA and `git_dirty` false
   (§1.2.4).
4. **The resume recomputed nothing it had stored, and the prune at open was exercised** (§1.3).
   - `resumed-a` stored 164 `GkSource` rows, 43 of them unvalidated. `resumed-b`'s open pruned
     exactly those 43.
   - `resumed-b` computed 170 = 291 − 164 + 43 `GkSource` rows, and stored 286 QSI rows before its
     signal.
   - `resumed-c` computed the 578 = 864 − 286 QSI rows that were missing.
5. **Every reader exits 0, refuses nothing, and leaves every file of the store byte-identical**
   (§1.4). Its lookups are R3's.
6. **L4 equals R3's** (§1.5), with two log differences in `resumed-b`/`resumed-c` that come from
   where the signal fell (§1.2.5).
7. **No line names `datastorekit`**, and the one line naming `runregistrykit` is the launcher's
   banner, as at R3. No `DeprecationWarning`, `ConfigError`, traceback or refusal appears in a build
   log. No raylet disk-space warning appears: the volume is 87 % used.
8. **The lock** was taken before `reference`, held through the readers and the guard, and released.
   `set-pid` failed at `reference` by the agent's slip, so that run's pid was never recorded in the
   lock (§1.1.3). `list` shows R4's four runs and nothing else (§1.9).

### 1.1 How each was launched

Each launch was R3's command (`rrk-v020-adoption-verification.md` §1.1), with the launcher's new
path, `handover-06`, `--campaign handover --prompt 06` and this prompt's store names. It was run
from the repository root, detached, with the purpose passed to `sh -c` as its `$0`
(`var/handover-06/scripts/launch.sh`):

```bash
nohup sh -c "PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/handover-06-<store>.sqlite \
    --job-name handover-06 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose \"\$0\" --campaign handover --prompt 06; \
  echo \"** driver exit status \$?\"" "<one line>" \
  > var/runs/handover-06-<slug>.launch.log 2>&1 &
disown
```

- **Each `--purpose`** names:
  - R4 of `handover` prompt 06 on the Mac Studio (`mps026298`);
  - the tree, `3152af9`;
  - `datastorekit` 0.5.0 `f47b9a2`, `raypoolkit` 0.2.0 `eaef0ce6` and `runregistrykit` 0.2.0 `0bae3a3`;
  - the launcher and its `b2fa4ef1`;
  - the fingerprint it must equal, `39809dca`, and R3's run that recorded it,
    `rrk-v020-adoption-03-reference-20261018T232838`.

  The purposes are in [`run_records_06.txt`](a3-v2/rehearsal/out/run_records_06.txt).
- **Before the first launch** `var/datastores/` and `var/runs/` did not exist; both were made empty
  (`mkdir -p`), since the launcher refuses a `--database` whose directory is not one.

**One check per launch, one wait per launch.** After each launch the agent made one check
(`var/handover-06/scripts/after_launch.py`), and in the same step ran `tools.rehearsal_lock set-pid`
with the driver's pid. The check covered:
- the manifest: `packages` against the record as JSON, `git_dirty`, `git_head`, `script`,
  `script_sha256`, its keys against R3's reference manifest's, and `host`;
- `status.json`: `running`, a pid answering `kill -0`, and a `host` whose `id` and `boot` are
  `this_host()`'s;
- the run's row in `list`, and `liveness(status, 900)` `alive`;
- for `resumed-a` and `resumed-b`, the interrupter's pid and log.

Then **one background command** waited until `kill -0 <driver pid>` failed
(`while kill -0 <pid> 2>/dev/null; do sleep 15; done`). There was no foreground poll and no
`pgrep -f`. Every wait ended with the pid dead: `ENDED` 19:56:24, 19:59:16, 20:01:31, 20:03:35.

| Slug | Run id | Store | `--resume` | Python pid (`sh` pid) | State | Exit code | Driver exit status | Wall time | Load at launch / after |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `handover-06-reference-20261022T195228` | reference | no | 78053 (78047) | `done` | 0 | 0 | **3 m 41 s** (19:52:28–19:56:09) | 4.15 10.96 8.90 / 5.76 8.62 8.41 |
| `resumed-a` | `handover-06-resumed-a-20261022T195718` | resumed | no | 78548 (78545) | `killed` | 143 | 143 | 1 m 47 s (19:57:18–19:59:05) | 4.83 8.06 8.21 / 4.73 7.13 7.83 |
| `resumed-b` | `handover-06-resumed-b-20261022T200009` | resumed | yes | 78952 (78950) | `killed` | 143 | 143 | 1 m 13 s (20:00:09–20:01:22) | 4.26 6.71 7.65 / 6.17 6.67 7.52 |
| `resumed-c` | `handover-06-resumed-c-20261022T200159` | resumed | yes | 79208 (79203) | `done` | 0 | 0 | 1 m 27 s (20:01:59–20:03:26) | 5.62 6.53 7.46 / 5.53 6.26 7.25 |

- **Wall time** runs from the manifest's `created` to the final heartbeat. R3's were 3 m 52 s,
  2 m 18 s, 1 m 19 s and 1 m 46 s.
- **Load** is `uptime`'s on 10 CPUs, at the launch and at the first check after the run. The
  one-minute load ran from 4 to 6 at these checks (R3: 6 to 29), and reached 17 at the end of the
  readers (R3: 74).

**The sidecars** ([`store_sidecars_06.txt`](a3-v2/rehearsal/out/store_sidecars_06.txt)):
- **The reference store.** `store create` ran on its closed primary at 19:56:52, after `reference`.
  `store fingerprint --write` followed: `39809dca…`, problems none, "it replaced none".
- **The resumed store.** `store create` ran on its closed primary at 19:59:45, after `resumed-a`.
  - Every one of the store's five files has an mtime before the sidecar's `created`, so it wrote
    none of them.
  - The before/after hash comparison planned for this step was lost to a shell slip (log,
    deviation 9).

`resumed-b` and `resumed-c` then each record `fingerprint_sidecar: written`, and `reference` and
`resumed-a` "not written: … is not a problem-free registry sidecar (there is none)", as at R3.

#### 1.1.1 The pre-launch checks

Before each launch ([`prelaunch_checks_06.txt`](a3-v2/rehearsal/out/prelaunch_checks_06.txt)):
- **The tree.** `HEAD` was `3152af943a4f8dd5cf5920f7ed6b209f1c7e217b` and `git status --short` was
  empty. `git worktree list` showed this checkout alone. Every working file went under
  `var/handover-06/`.
- **The environment.** `RAY_ADDRESS` was unset, and the machine has 10 CPUs.
- **The registry.** `list` showed nothing `running` or stale. It showed 0, 1, 2 and 3 runs, all
  finished, and "0 with no host recorded" throughout.
- **No Ray process of this repository.**
  - Before each launch, `tools.rehearsal_lock ray-processes` exited 0 and listed nothing.
  - Before each later launch, `ps -p` checked the previous run's pids: the `sh` and Python pids, the
    interrupter's, and every Ray pid its logs name. None was alive.
- **The lock** was free before `acquire`, and `~/ScienceData/` did not exist.
- **The store names.**
  - Before `reference`, `store show` on both names gave kind `absent`, problems none, and no run
    naming either, exit 0.
  - Nothing under `var/` but `var/handover-06/` contained `handover-06`.
  - No `-journal`, `-wal` or `-shm` file existed under `var/datastores/` before any launch.
- **The packages.**
  - `pip show` gave `runregistrykit` 0.2.0, `datastorekit` 0.5.0 and `raypoolkit` 0.2.0, each in
    this venv's `site-packages`. `pip check` was clean.
  - `package_provenance` equalled the record above as JSON. Its `raypoolkit` and `runregistrykit`
    entries equal R3's reference manifest's; its `datastorekit` is the one change (R3: 0.4.0
    `c8570e8`).
  - `TERMINAL_STATES` is `('done', 'failed', 'killed', 'lost')`.
- **The configuration.** `find_client_root()` read `runregistrykit.toml`: `runs_root` `var/runs`,
  `stores_root` `var/datastores`, `distributions` `('datastorekit', 'raypoolkit')`, and the
  fingerprinter `config.fingerprint:fingerprinter`.
- **What runs.**
  - The three packages' `__file__` are under the venv's `site-packages`.
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`, and `ObjectFactories.__file__` is the
    repository's.
  - `ENCODING_FORMAT` is 1 and `FINGERPRINT_FORMAT` is 6.
  - `ray.__version__` is 2.43.0 and `ray.is_initialized()` is false.
- **The recorded fingerprints R4 is compared with.** R3's `reference` and `resumed-c` `status.json`
  are each `done`/0: `fingerprint_format` 6, `encoding_format` 1, `39809dca…`, 21 classes,
  problems `{}`, host `8d2b247824882486` (the laptop).
- **The launcher.** `shasum -a 256 RunDrivers/quadsource_atol_sweep.py` was
  `b2fa4ef1bbdbb09fea1780952710eeb53dee401ef6c0febcd385dd98aa0526ef`. Its last change is `031bcb4`,
  and `git log c081eed..HEAD` touches it in no commit.
- **The pins** are `requirements.txt:17` (`@v0.5.0`), `:75` (`ray==2.43.0`), `:76` (`@v0.2.0`) and
  `:81` (`runregistrykit[stores] @ …@v0.2.0`).
- **Disk.** 121–122 GB was free on `var/`'s volume (87 %).

#### 1.1.2 The interrupter

**The dummy tests, at `HEAD`, before the first launch**, each into a new directory under
`var/handover-06/work/`, with R3's commands. Both exited 0
([`interrupter_dummy_test_06.txt`](a3-v2/rehearsal/out/interrupter_dummy_test_06.txt),
[`interrupter_build_dummy_test_06.txt`](a3-v2/rehearsal/out/interrupter_build_dummy_test_06.txt)). They ran
19:50:48–19:51:14.
- With times, pids and paths normalised, the scoped output (40 lines) and the build output (28 lines,
  cases A–D) equal R3's `interrupter_dummy_test_03.txt` and `interrupter_build_dummy_test_03.txt`,
  line for line.
- Case D gave interrupter exit 0 and child exit 137.
- No dummy child or interrupter was left alive: `ps -p` on all the pids named found none.

**The points, from R4's own `reference` `stdout.log`, by R3's reasoning.**
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **22.6 s** (`stdout.log:76`;
  R3: 24.8 s). 40 % of it is 9.04 s, which is under the 10 s floor, so the hold is **10 s**, from the
  banner: `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 10`.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **1 m 28.7 s (88.7 s)**
  (`stdout.log:100`; R3: 91.4 s). A third of it is 29.57 s, which rounds to **30 s**, under the
  250 s cap: `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 30`.
- **Both holds are R3's.**

**The real firings** ([`interrupter_firings_06.txt`](a3-v2/rehearsal/out/interrupter_firings_06.txt)):

```
resumed-a (interrupter pid 78556)
2026-10-22T19:58:54+01:00 liveness alive; pid 78548; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
2026-10-22T19:58:54+01:00 eligible at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'; holding 10 s, then reading again
2026-10-22T19:59:04+01:00 sent SIGTERM to pid 78548 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); command "…/Python -u RunDrivers/quadsource_atol_sweep.py --build --rehearsal --database var/datastores/handover-06-resumed.sqlite … --register resumed-a …"; exiting

resumed-b (interrupter pid 78964)
2026-10-22T20:00:50+01:00 liveness alive; pid 78952; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
2026-10-22T20:00:50+01:00 eligible at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'; holding 30 s, then reading again
2026-10-22T20:01:20+01:00 sent SIGTERM to pid 78952 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); command "…/Python -u RunDrivers/quadsource_atol_sweep.py --build --rehearsal --resume --database var/datastores/handover-06-resumed.sqlite … --register resumed-b …"; exiting
```

- Each interrupter matched the run by the basename of its manifest's `argv[0]`. That is
  `quadsource_atol_sweep.py` at the launcher's new path too, so the move to `RunDrivers/` changes
  nothing here.
- Both banners were read. No progress line was read: no stage ran past 300 s.
- Both interrupters were launched detached about 1 s after their builds, once the run directory and
  its `status.json` existed.

#### 1.1.3 The lock

Every helper command and its exit status ([`lock_06.txt`](a3-v2/rehearsal/out/lock_06.txt)):

| When | Command | Exit | What it said |
|---|---|---|---|
| 19:52:00 | `acquire --purpose "SecondaryGWKit handover 06 (R4 on the Mac Studio)"` | 0 | acquired, for SecondaryGWKit |
| 19:52:00 | `status` | 0 | `HELD`, host `mps026298`; no pid recorded yet; no Ray process |
| 19:52:34 | `set-pid 78053` (`reference`) | **2** | **not run**: the agent's loop passed `set-pid 78053` as one argument, and argparse rejected it |
| 19:52:34 | `status` | 0 | `HELD`; `pid: -`; 14 Ray processes, the run's |
| 19:57:26 | `set-pid 78548` (`resumed-a`) | **2** | the same slip |
| 19:57:26 | `status` | 0 | `HELD`; `pid: -`; 18 Ray processes |
| 19:57:38 | `set-pid 78548` (`resumed-a`), re-run correctly | 0 | recorded |
| 19:57:38 | `status` | 0 | `pid: 78548`, alive; 19 Ray processes |
| 20:00:18 | `set-pid 78952` (`resumed-b`) | 0 | recorded |
| 20:00:18 | `status` | 0 | `pid: 78952`, alive; 14 Ray processes |
| 20:02:00 | `set-pid 79208` (`resumed-c`) | 0 | recorded |
| 20:02:00 | `status` | 0 | `pid: 79208`, alive; 2 Ray processes |
| 20:04:23 | `status` (before the readers) | 0 | `pid: 79208`, NOT alive; no Ray process |
| 20:07:26 | `status` | 0 | the same, after the readers and the guard |
| 20:07:26 | `ray-processes` | 0 | none |
| 20:07:26 | `release` | 0 | released |
| 20:07:26 | `status` | 0 | `lock: free`; no Ray process |
| 20:07:27 | `ray-processes` | 0 | none |

- The owner file named `project: SecondaryGWKit`, `ray_temp_dir: in-process` and
  `ray_address: in-process` throughout.
- `acquire` came 28 s before `reference`'s manifest (19:52:28). The lock was held through all four
  launches, the checks, the readers and the guard.
- **`reference` ran with the lock held but no pid in it** (`pid: -`), because of the slip above. It
  was not back-filled after the run had ended.
- Afterwards `~/ScienceData/` exists and is empty: `acquire` made it.

### 1.2 L1 — every launch ends as recorded

#### 1.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `encoding_format` 1, `39809dca…`, problems `{}` | not written: "…handover-06-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `encoding_format` 1, `7c165c78…`, problems `{}` | not written: "…handover-06-resumed.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `encoding_format` 1, `82318bfa…`, problems `{}` | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `encoding_format` 1, `39809dca…`, problems `{}` | `written` | `** driver exit status 0` |

- Each fingerprint holds 21 classes. Each `taken` names `git_head` `3152af94…` and `git_dirty` false.
- Each `status.json` has R3's keys (`exit_code`, `fingerprint`, `fingerprint_sidecar`, `heartbeat`,
  `host`, `pid`, `stage`, `state`, `units_done`, `units_total`).
- The full intermediate digests are `resumed-a`
  `7c165c7852801b8170b27dba6c5307c64076e3fb40577c8190fc84b9f291fa80` and `resumed-b`
  `82318bfa43c805589489bd8642f37a36e870277b213d3ac89a8f72f7d9378405`. They differ from R3's
  (`c60e0d19…`, `7e41cbe2…`), as they must: each digests a store killed at a different point. By
  their class counts:
  - `resumed-a`'s store held `GkSource` 164, `GkSourcePolicyData` 0 and QSI 0;
  - `resumed-b`'s held 291, 291 and 286.

#### 1.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 19:59:04 | 20:01:20 |
| Handler's line | `** SIGTERM at 2026-10-22 19:59:04 +0100: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-22 20:01:20 +0100: recorded killed, exit code 143; …` |
| `stderr.log`'s handler lines | `** SIGTERM: the pipeline ended with SystemExit(1); recording killed, exit code 143` | `** SIGTERM: the pipeline ended with **SystemExit(143)**; recording killed, exit code 143` |
| Python traceback | none | none |
| Fingerprint taken / final heartbeat | 19:59:04 / 19:59:05 | 20:01:21 / 20:01:22 |
| `status.json.*.tmp` left in the run directory | **none** | **none** |
| Stage at the signal | the `GkSource` banner, first read at 19:58:54; signal 10 s later (**164 of 291 rows stored, 43 unvalidated**) | the QSI banner, first read at 20:00:50; signal 30 s later (**286** of 864 stored) |

- **`resumed-b` ended `SystemExit(143)`**, where R3's ended `SystemExit(1)`. R2's `resumed-b` ended
  `SystemExit(143)` too. As R2 explains (its §3.2.2), a signal that lands inside `ray.wait` has Ray's
  signal check turn the handler's `SystemExit(143)` into `SystemExit(1)`, so which one appears
  depends on where the signal falls. The record is `killed`/143 and the exit is 143 either way.
- Each run directory holds `interrupter.log`, `manifest.json`, `status.json`, `stderr.log` and
  `stdout.log` and nothing else.
- `resumed-a`'s `stderr.log` holds the handler's two lines and nothing else. **`resumed-b`'s holds
  SciPy's `IntegrationWarning` (4 lines) before them**: see §1.2.5.

#### 1.2.3 No Ray process of this repository left

| After | When | Processes of this repository |
|---|---|---|
| `reference` | 19:56:40 (~31 s after the final heartbeat) | none: `ray-processes` exit 0; pids 78047 and 78053 and the Ray pids its logs name (78089, 78090, 78091, 78117) gone |
| `resumed-a` | 19:59:32 (~27 s) | none, for 78545, 78548, the interrupter 78556, and the Ray pids 78576, 78578 |
| `resumed-b` | 20:01:41 (~19 s) | none, for 78950, 78952, the interrupter 78964, and the Ray pids 78981, 78984, 78985, 79021 |
| `resumed-c` | 20:03:39 (~13 s) | none, for 79203, 79208 and the Ray pid 79275 |

No Ray process outlived either `SIGTERM`'d driver, so no wait by pid was needed, as at R0–R3.

#### 1.2.4 The manifests and the hosts

From [`run_records_06.txt`](a3-v2/rehearsal/out/run_records_06.txt), for all four:
- **`git_head`** is `3152af943a4f8dd5cf5920f7ed6b209f1c7e217b`, the dispatch SHA, and **`git_dirty`**
  is false.
- **`packages`** equals, as parsed JSON:
  `{"datastorekit": {"commit": "f47b9a255e9ad58e0d0eaf85644f53b575884d31", "editable": false, "requested_revision": "v0.5.0", "url": "https://github.com/ds283/DatastoreKit", "version": "0.5.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}, "runregistrykit": {"commit": "0bae3a3c501a8264ee4be5bc24f06422af38720f", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RunRegistryKit", "version": "0.2.0"}}`.
  Against R3's, only `datastorekit` differs.
- **`script`** is `RunDrivers/quadsource_atol_sweep.py` and **`script_sha256`** is `b2fa4ef1…` in all
  four (R3: `docs/handover/…`, `2a985303…`). `argv[0]` is the same path.
- **The manifests have twenty keys, R3's twenty.**
  - Each `host` is `{"id": "d120d1cf03551e6f", "machine": "arm64", "name": "mps026298.phys.susx.ac.uk", "python": "3.12.15", "release": "27.0.0", "system": "Darwin"}`,
    which is `this_host()`'s.
  - Each `scope` begins `REHEARSAL build, not A3 v2`.
- **Every status's `host`** is `{"boot": "C58A0F92-923F-471F-B32E-B1B9F5D0A848", "id": "d120d1cf03551e6f"}`,
  `this_host()`'s, at every check and at every run's end. The `boot` is the one `uptime` and
  `kern.boottime` (2026-10-07 10:23:52) describe.
- `argv` holds `--resume` in `resumed-b` and `resumed-c` only.
- `results_store_id` is `null` for `reference` and `resumed-a`, and `1c41fb13…` for `resumed-b` and
  `resumed-c`, as at R3.

#### 1.2.5 Every launch's `stdout.log`, and what is new in the logs

The full counts are in [`log_comparison_06.txt`](a3-v2/rehearsal/out/log_comparison_06.txt), section 1, with R3's
beside each (`var/handover-06/scripts/log_comparison.py`). For each launch:
- **Equal to R3's:**
  - `replaced 1 occurrence` 4 and 4; the rehearsal banner, `main.py argv` and `bootstrapping a local
    Ray instance` 1 each; `Started a local Ray instance` 1 in the launch log;
  - `--no-prune-unvalidated`, `reconcil|repair|in.flight|replicat`, `Traceback`,
    `ModuleNotFoundError|ImportError|AttributeError`, `pickle`, `Datastore.SQL`, `ray.kill` and the
    actor errors, `ValueError`, `RuntimeWarning`, `did not validate`, `DeprecationWarning`,
    `_timestamp`, `ConfigError` and `RunRegistry` (not `Kit`): **0**;
  - `datastorekit`, anywhere: **0**;
  - `runregistrykit`: 1 in `stdout.log` and 1 in the launch log, the banner's `**   list it with:
    ./venv/bin/python -m runregistrykit list`, and nothing else;
  - `INTEGRITY WARNING`: 0, but for `resumed-b`, 2 and 2, as at R3;
  - `resuming into`: 1 in the launch logs of `resumed-b` and `resumed-c`;
  - raylet `file_system_monitor`: **0**.
- **Different from R3's** (marked `!` in section 1):
  1. **`resumed-b`: `SystemExit(143)` 1 / 1 in `stderr.log` and the launch log, and
     `SystemExit(1)` 0** (R3: the reverse). Explained in §1.2.2.
  2. **`IntegrationWarning`**: `resumed-b` 1 / 1 and **`resumed-c` 0** (R3: `resumed-b` 0, `resumed-c`
     1). `reference` has it once, as at R3. The warning comes from one `QuadSourceIntegral`, which
     emits it when it is computed.
     - At R3, `resumed-b` stored 181 QSI rows before its signal, so `resumed-c` computed that
       integral.
     - At R4, `resumed-b` stored 286, which include it, so `resumed-c` did not compute it again.
     - It follows from where the signal fell, as the counts of §1.3 do.

**Against R3's same slug, line kind by line kind** (section 2; each line normalised: ANSI codes, the
repository path, campaign and run names, the launcher's path, the purpose, Ray's `[repeated Nx …]`
count and every number). R3's raw logs are here for `reference` and `resumed-c` only:

| Slug | `stdout.log` | `stderr.log` |
|---|---|---|
| `reference` | 110 lines against 110; **no kind differs** | 4 against 4; no kind differs |
| `resumed-a` | R3's not on this machine: by counts (section 1) and the `** ` counts (§1.5.4) | the same |
| `resumed-b` | the same | the same |
| `resumed-c` | 107 against 107; **no kind differs** | **0 against 4**; R3's 4 are the `IntegrationWarning` (difference 2) |

**No new line names `datastorekit`**, and there is no `DeprecationWarning` and no refusal. The pin
move to 0.5.0, the launcher's move and the host added no line to a build log.

### 1.3 L2 — the resume is the reference

#### 1.3.1 Nothing stored is recomputed

Through `dk-and-rpk-migration`'s `stage_counts.py`, as it stands, on R4's four run directories
([`stage_counts_06.txt`](a3-v2/rehearsal/out/stage_counts_06.txt)). Lookup / compute / store:

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
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **578/578/578** |

- **Every stage that finished before an interruption computed zero on the next launch.**
- **`GkSource`.**
  - `resumed-a` stored **164** rows before the signal (53 + 55 + 0 + 56 by shard), of which **43**
    were unvalidated (8 + 23 + 0 + 12).
  - `resumed-b` pruned those 43 at open and computed **170 = 291 − 164 + 43**.
  - Summed, less the pruned: 164 + 170 − 43 = 291.
- **`QuadSourceIntegral`.** `resumed-b`'s fingerprint records **286** rows. `resumed-c` computed
  **578** = 864 − 286. Summed: 864.
- **The `reference` and `resumed-a` columns equal R3's stage by stage.**

#### 1.3.2 The prune at startup

The closed store after each launch was read `mode=ro` by
`docs/a3-v2-readiness/rehearsal/closed_store_state.py`
([`closed_store_state_06.txt`](a3-v2/rehearsal/out/closed_store_state_06.txt)):

| After | `-journal`/`-wal`/`-shm` | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 rows | none |
| `resumed-a` | none | 0 rows | **`GkSource` 8 + 23 + 0 + 12 = 43**, of 53 + 55 + 0 + 56 = 164 stored, each listed by serial |
| `resumed-b` | none | 0 rows | none (`GkSource` 97 + 97 + 0 + 97 = 291) |
| `resumed-c` | none | 0 rows | none |

`resumed-a`'s signal came 10–15 s into a stage the reference took 22.6 s over. (R3 left 168 and 48,
R2 165 and 44.)

**By serial**, through `prune_check.py`, unchanged ([`prune_check_06.txt`](a3-v2/rehearsal/out/prune_check_06.txt)):
- **Against the state after `resumed-a`**, run twice: at 20:01:51, after `resumed-b`, and after
  `resumed-c`. Both times it gave `43 serial(s) checked: 43 absent, 0 reused by a validated row,
  0 still unvalidated`, exit 0, with 8 serials in shard 0, 23 in shard 1 and 12 in shard 3.
- **Against the state after `resumed-b`**, after `resumed-c`: `none: there is nothing to check`,
  exit 0.

**So the prune at open removed exactly the rows `resumed-a` left unvalidated**, and no validated row.
`resumed-b`'s `INTEGRITY WARNING` lists those rows (2 lines in `stdout.log`, 2 in the launch log, as
at R3).

#### 1.3.3 The version row

After every launch there is exactly one row, `(1, '2025.1.1')`, on each of the four shards of each
store. The primary has no `version` table.

#### 1.3.4 The comparison

**The two stores**, by `compare_stores.py`
([`fingerprint_comparison_06.txt`](a3-v2/rehearsal/out/fingerprint_comparison_06.txt)). It exited 0:
- both stores are format 6, digest `39809dca…`, problems none, each with 0 differences against its
  sidecar;
- `compare_fingerprints(reference, resumed): 0 difference(s)`, verdict `equal`.

**Against R3's recorded fingerprints**, by `compare_with_recorded.py` as it stands
([`fingerprint_against_recorded_06.txt`](a3-v2/rehearsal/out/fingerprint_against_recorded_06.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/handover-06-reference.sqlite var/datastores/handover-06-resumed.sqlite \
    --against var/handover-06/recorded/rrk-v020-adoption-03-reference-20261018T232838 \
    var/handover-06/recorded/rrk-v020-adoption-03-resumed-c-20261018T233941
```

It exited 0, verdict `equal`. For each store there were 0 differences against each of:
- its sidecar;
- **R3's `reference`** and **R3's `resumed-c`**, each taken at `aab40752…`, `git_dirty` false, on
  the laptop.

All are format 6, digest `39809dca…`, 21 classes, problems none.

**The store files** (both stores, both sidecars) had the same SHA-256, size and mtime before and
after the closed-state read, the prune checks, both comparisons, the census and the soundness check
(`var/handover-06/state/store_files_{before,after}_checks.txt`).

#### 1.3.5 Integrity, both stores, read-only

**The census** ([`census_06.txt`](a3-v2/rehearsal/out/census_06.txt)). Every table is `OK` on both stores (28 lines),
the verdict is `no divergence` on both, and the exit is 0.

**`store_soundness.py`** ([`store_soundness_06.txt`](a3-v2/rehearsal/out/store_soundness_06.txt)) exited 0,
`=> clean` on both:
- on the primary and every shard, opened `mode=ro`, `PRAGMA integrity_check` is `['ok']` and
  `PRAGMA foreign_key_check` returns 0 rows;
- `read_inventory` reads shards `(0, 1, 2, 3)` with no problem in any class and 0 `duplicate`. The
  counts are `GkSource` 291, `GkSourcePolicyData` 291, `QuadSource` 6, `QuadSourceIntegral` 864,
  `GkNumericIntegration` 1368 and `GkWKBIntegration` 3477.

### 1.4 L3 — every reader, read-only, on the resumed store

Each reader ran once, serially, on the closed resumed store, with its own local Ray instance
(`--ray-address local`). They ran through `run_readers.sh`, unchanged, with R3's two commands and
output under `var/handover-06/extract/`, the lock held ([`readers_06.txt`](a3-v2/rehearsal/out/readers_06.txt)).
Both invocations exited 0.

| Script | Exit | Wall | Lookups (last queue) | Files written (R3) | Traceback / `ReadOnly*` | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 19 s | 116 | 423 (468) | none | identical |
| `extract_GkWKB_data.py` | 0 | 41 s | 348 | 2436 (2436) | none | identical |
| `extract_GkSource_data.py` | 0 | 14 s | 10 | 156 (156) | none | identical |
| `extract_TkWKB_data.py` | 0 | 11 s | 3 | 27 (27) | none | identical |
| `extract_tensor_source_data.py` | 0 | 11 s | 6 | 27 (27) | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 12 s | 24 | 73 (70) | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 19 s | LambdaCDM 116; then `>> SKIPPING MODEL QCD_Cosmology: the datastore holds no background model for it` | 441 (378) | none | identical |

- **The lookups equal R3's**, reader by reader.
- **The files written differ** in the readers that draw random subsamples, as between every earlier
  pair. This is not a finding.
- No reader log names `runregistrykit` or `datastorekit`.
- The readers ran 20:04:31–20:06:45, at a one-minute load of 4.7–17.
- **No file of the store changed.** Before and after each reader the SHA-256, mtime and size of the
  primary and every shard were identical, and no journal appeared. No Ray process of this checkout
  was left after any reader.
- **The fingerprint afterwards.** `store fingerprint` on the resumed store gave `39809dca…`,
  problems none, "matches the fingerprint taken 2026-10-22T20:03:25+01:00 by run
  handover-06-resumed-c-20261022T200159, at 3152af943a4f8dd5cf5920f7ed6b209f1c7e217b", exit 0.

**A3 v2's guard**, after the readers, in the foreground, with no `--register`, the lock held
([`guard_06.txt`](a3-v2/rehearsal/out/guard_06.txt)):
- It exited 1 within a second (20:07:08) and printed one line on stderr: `!! quadsource_atol_sweep
  --build: handover-06-resumed.sqlite, … already exist(s). Refusing to build into an existing store:
  …`.
- There was **no traceback**. That is `rrk-v020-adoption` 05's clean refusal; R3, before 05, printed
  a `RuntimeError` traceback. The message is R3's.
- Nothing changed: every file of both stores and both sidecars is identical before and after.
- Nothing was created: `var/runs/` held 8 entries before and after, and there was no Ray process and
  no journal.

### 1.5 L4 — equal to R3

#### 1.5.1 The `GkSourcePolicyData SUMMARY STATISTICS` block

`policy_statistics.py` exited 0 on `reference`, `resumed-b` and `resumed-c`, and 1 on `resumed-a`,
which has no block, as at R3 ([`policy_statistics_06_*.txt`](a3-v2/rehearsal/out/)).

| | R3 | R4 |
|---|---|---|
| Total instances | 291 | 291 |
| numeric | 109 (109 complete) | 109 (109 complete) |
| mixed | 26 (25 complete, 1 minimal) | 26 (25 complete, 1 minimal) |
| WKB | 156 (156 complete) | 156 (156 complete) |
| `reference` | `complete` first under `mixed` | `minimal` first; **sorted lines identical to R3's** |
| `resumed-b` | `complete` first | `minimal` first; sorted lines identical |
| `resumed-c` | `minimal` first | **byte-identical to R3's `resumed-c`** |

**Every count is R3's.** The order of the two `mixed` lines varies run to run
(`[02-the-policy-statistics-block-orders-its-lines-by-completion]`), and sorted, every block equals
R3's same slug.

#### 1.5.2 The reference run's stage counts

The `reference` column of §1.3.1 equals R3's `stage_counts_03.txt` in every stage, and so does the
`resumed-a` column.

#### 1.5.3 The resumed chain, by identity

| Identity | R3 | R4 | Holds |
|---|---|---|---|
| Every stage finished before a kill computes 0 on the next launch | yes | yes | ✅ |
| `GkSource` rows pruned at `resumed-b`'s open = rows `resumed-a` left unvalidated | 48 = 48, all absent, none reused | **43 = 43, all absent, none reused** | ✅ |
| `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned) `GkSource` | 171 = 291 − 168 + 48 | **170 = 291 − 164 + 43** | ✅ |
| `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral`, and nothing else | 683 = 864 − 181 | **578 = 864 − 286** | ✅ |
| Totals per class over the chain equal the reference's | 291; 864 | `GkSource` 164 + 170 − 43 = 291; QSI 286 + 578 = 864 | ✅ |

#### 1.5.4 The stage sequence

Section 3 of `log_comparison_06.txt`:
- **Counts.** R4 and R3 hold the same counts of `** ` lines, `ALL WORK ITEMS COMPLETE`, `Queue
  summary`, no-compute lines, progress lines and `inflight` lines, slug by slug: (26, 10, 10, 1, 0,
  0), (23, 7, 7, 1, 0, 0), (26, 9, 9, 7, 0, 0) and (26, 10, 10, 9, 0, 0).
- **`reference` and `resumed-c`**, whose R3 logs are here:
  - the `** ` sequence, normalised, equals R3's in order;
  - the stage banners (`** CALCULATE …`, `** APPLY …`, `** BUILD …`) equal R3's verbatim (13/13).
- **`resumed-a` and `resumed-b`** are compared by those counts, and their banners are listed in
  section 3.

#### 1.5.5 The stage times

These are not compared. `07` scales A3 v2's wall-clock estimate from them (`ALL WORK ITEMS COMPLETE
in time …`; R3's in brackets):

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 22.9 s (26.2 s) | 22.5 s (34.3 s) | 1.54 s (1.74 s) | 1.55 s (1.61 s) |
| WKB PART OF TENSOR GREEN FUNCTIONS | 1 m 4.75 s (1 m 6.82 s) | 1 m 4.09 s (1 m 18.6 s) | 3.23 s (3.64 s) | 3.13 s (3.33 s) |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **22.6 s** (24.8 s) | killed (killed) | 18.2 s (21.4 s) | 4.77 s (5.15 s) |
| APPLY GKSOURCE POLICIES | 8.77 s (9.67 s) | — | 8.59 s (10.4 s) | 4.75 s (4.98 s) |
| QUADRATIC SOURCE INTEGRALS | **1 m 28.7 s** (1 m 31.4 s) | — | killed (killed) | 1 m 6.87 s (1 m 23.4 s) |

- Each earlier stage took under 1 s.
- **The reference took 3 m 41 s, against R3's 3 m 52 s** (R2: 4 m 33 s). Stage by stage the Mac
  Studio at `--cpus 6` is 2–13 % faster than the laptop was at R3, at a lower load (4–6 against
  6–29).
- `resumed-c`'s QSI time is shorter than R3's because it computed 578 rows, not 683.

#### 1.5.6 The printed lines and the registry's fields

The printed-line counts are those of §1.5.4, each equal to R3's. No progress line was printed: no
stage ran past 300 s.

**The registry's fields at each run's end** are R3's: `state`, `stage`, `units_done` and
`units_total`.

| Slug | `state` | `stage` | `units_done` / `units_total` |
|---|---|---|---|
| `reference` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` |
| `resumed-a` | `killed` | `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` | 0 / `null` |
| `resumed-b` | `killed` | `CALCULATE QUADRATIC SOURCE INTEGRALS` | 0 / `null` |
| `resumed-c` | `done` | `CALCULATE QUADRATIC SOURCE INTEGRALS, all work items complete` | 0 / `null` |

### 1.6 What was expected not to happen

Each of these was looked for, and none happened:
- an unvalidated `BackgroundModel`;
- an in-flight record left by a killed launch;
- a repair by the check at open;
- a Ray agent outliving a `SIGTERM`'d driver;
- a line from the pin or the move: a `datastorekit` frame, a `ConfigError`, a `DeprecationWarning`
  or a refusal;
- a temporary status file;
- a record written outside `var/` (bar the `.git/info/exclude` line, §1 above);
- a missed interruption;
- a held lock, or a refused `release`.

### 1.7 Issues opened

**One**, on the orchestrator's instruction: `[06-nothing-tracked-ignores-venv]` (see `issues.md`).
Every difference from R3 is explained by:
- the host: the disk is under Ray's threshold, and the times are faster;
- where a launch was killed: the counts of §1.3, `SystemExit(143)` and the `IntegrationWarning`'s
  move;
- the readers' random subsamples;
- `rrk-v020-adoption` 05's clean refusal at the guard;
- the order of the statistics block's lines.

### 1.8 The suites

The seven roots ran before the first launch, at the dispatch SHA (§1 above,
[`roots_06.txt`](a3-v2/rehearsal/out/roots_06.txt)). No Python file was added or changed afterwards.

### 1.9 What stays under `var/`

- **The run directories:** `var/runs/handover-06-{reference,resumed-a,resumed-b,resumed-c}-<stamp>/`,
  with their four launch logs beside them.
- **The stores:** `var/datastores/handover-06-{reference,resumed}.sqlite`, each with four shards and
  a `.manifest.json` sidecar. Neither is retired.
- **`var/handover-06/`**, which holds:
  - `out/`, the saved outputs;
  - `log.md`, `verification.md` and `issues.md`;
  - `recorded/`, R3's two run directories;
  - `scripts/`, the agent's throwaway helpers;
  - `state/`, the closed-store states, the hashes, the two freezes and `.git/info/exclude` as it was;
  - `extract/`, the readers' output; `roots/`, the suites' full output; and `work/`, the dummy tests.
- **Nothing else.** [`var_datastores_06.txt`](a3-v2/rehearsal/out/var_datastores_06.txt) lists every file under
  `var/datastores/` and `var/runs/` with its SHA-256, size and `mtime_ns`: 34 files. There is no
  `-journal`, `-wal` or `-shm` file.

**The registry after R4** ([`run_records_06.txt`](a3-v2/rehearsal/out/run_records_06.txt)). `list` shows **4 runs**:
"4 finished; 0 with no host recorded". Nothing is `running` or stale, and no row is marked `!!`. By
state: 2 `done` and 2 `killed`.

**Outside the checkout:**
- `~/Documents/Code/retired-venvs/SecondaryGWKit-py313-20261022` (the old venv, moved);
- `~/ScienceData/` (made by `acquire`, now empty);
- `~/handover-06-recorded/`, the orchestrator's staging folder, emptied by the move and left in
  place.

## 2. The launch

**Where the records are.** A path `var/handover-07/…` in this section is that directory on the Mac
Studio. Its `out/` files, its scripts (`launch_07.sh`, the launch as run; `prelaunch.sh`;
`launch_check.py`) and `freeze_07.txt` are committed at [`a3-v2/launch/`](a3-v2/launch/), the scripts
under `scripts/`.

**The tree.** A3 v2 was launched on the Mac Studio (`mps026298`) at the dispatch SHA
**`b90752dda46e57d588f8107cbd4d4557134fadb2`** on `handover-remedial` ("Record prompt 06's commit and
write the orchestrator note for 07").
- `git merge --ff-only origin/handover-remedial` took the checkout from `3152af9` (R4's) to it, by
  `319b14f` and `b90752d`.
- `git diff --stat 3152af9 b90752d -- . ':!prompts' ':!docs'` is empty, so A3 v2 builds R4's code.
- `git status --short` was empty before the launch and in the launch's own command.
  `git check-ignore -v venv` gave `.git/info/exclude:28:venv`.

### 2.1 The pre-launch checks (prompt §1)

From [`prelaunch_checks_07.txt`](launch/prelaunch_checks_07.txt), from 21:05:43:

| Check | Found | |
|---|---|---|
| 1. `HEAD`, tree, the Gate's diff | `b90752dd…`; `git status --short` empty; diff empty | ✅ |
| 2. The launcher's SHA-256 | `b2fa4ef1bbdbb09fea1780952710eeb53dee401ef6c0febcd385dd98aa0526ef`, 06's | ✅ |
| 3. `pip freeze` against 06's | 101 lines; identical to `docs/handover/a3-v2/rehearsal/out/pip_06.txt`'s freeze (`diff` rc 0) and to 06's local copy | ✅ |
| 4. `list` | R4's four runs, "4 finished; 0 with no host recorded"; nothing `running` or stale | ✅ |
| 5. The store name | `store show var/datastores/handover-A3-v2-lambdacdm.sqlite`: `absent`, problems none, no runs; `ls var/datastores/handover-A3-v2-lambdacdm*` finds nothing | ✅ |
| 6. Lock, Ray, `RAY_ADDRESS` | `lock: free`; `ray-processes` none; `unset` | ✅ |
| 7. Load, disk, power, CPU | load **4.19** 3.69 3.62 at 21:05; re-sampled at 21:06: 3.82, then **3.75** 3.63 3.60, with CPU 67 % then 84 % idle (`top -l 2 -s 5`). Disk 121 G free (88 %). `pmset` `sleep 0`. Top CPU: `mediaanalysisd` 84 %, `mds_stores` 11 %, OrbStack 10 % | ✅ recorded |

- **The load.** The first one-minute reading, 4.19, was above prompt §5's "loaded beyond 4 cores".
  The 5- and 15-minute averages were under 4, and the re-sample a minute later gave 3.75 with about
  1.6 cores busy. The launch went ahead on the re-sample (log, deviation 2).
- **The Gate: macOS updates.**
  - The user confirmed in chat that automatic installation is off.
  - `defaults read /Library/Preferences/com.apple.SoftwareUpdate` gives
    **`AutomaticallyInstallMacOSUpdates = 0`**.
  - It also gives `AutomaticDownload = 1`, **`CriticalUpdateInstall = 1`**, `ConfigDataInstall = 1`
    and `LastUpdatesAvailable = 1`, with `RecommendedUpdates` empty.
  - `AutomaticCheckEnabled` is not set.
- **The host.** `this_host()` is `{"boot": "C58A0F92-923F-471F-B32E-B1B9F5D0A848", "id":
  "d120d1cf03551e6f", "machine": "arm64", "name": "mps026298.phys.susx.ac.uk", "python": "3.12.15",
  "release": "27.0.0", "system": "Darwin"}`, which is 06's. The machine has not rebooted.
- **The packages.** `package_provenance` gives the three pins, none editable, equal to R4's
  manifests'.

### 2.2 The lock and the launch (prompt §2)

From [`lock_07.txt`](launch/lock_07.txt):

| When | Command | Exit | What it said |
|---|---|---|---|
| 21:06:24 | `acquire --purpose "SecondaryGWKit handover 07: A3 v2 build, about 40 h"` | 0 | acquired, for SecondaryGWKit |
| 21:06:24 | `status` | 0 | `HELD`, host `mps026298`; no pid yet; no Ray process |
| 21:12:00 | `set-pid 84436` | 0 | recorded |
| 21:12:00 | `status` | 0 | `pid: 84436`, alive; 14 Ray processes |

**The launch.** This is prompt §2.2's command verbatim, with `<SHA>` = `b90752d` and `<hours>` =
`39-44`, written to `var/handover-07/launch_07.sh` and run by `bash` at 21:11:54:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py --build \
    --database var/datastores/handover-A3-v2-lambdacdm.sqlite \
    --job-name handover-A3-v2 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register a3-v2 \
    --purpose "handover prompt 07: A3 v2, the LambdaCDM store rebuilt from scratch at QuadSourceIntegral atol 1e-32 rtol 1e-7 (main.py 9280-item work list, 8 k over 1e5-3e8/Mpc, zend 0.1, 6 cpus) on the Mac Studio at b90752d; R4 (handover 06) reproduced 39809dca on this host; estimated 39-44 h" \
    --campaign handover --prompt 07; \
  echo "** driver exit status $?"' \
  > var/runs/handover-07-a3-v2.launch.log 2>&1 < /dev/null &
disown
```

- **The pids.** `sh` 84431 (parent 1); driver **84436** (parent 84431).
- **The first attempt was refused** by the session's permission classifier, as a "production
  deploy", before anything ran. Between 21:06:24 and the user's approval in chat the lock was held
  with no pid and no build. The command then ran unchanged.
- **The estimate, 39–44 h.**
  - Prompt §0 gives 40–45 wall-hours at 6 CPUs at the laptop's speed (about 250 CPU-hours).
  - Scaled by R4 against R3, it is about 0.97 of that. QSI dominates A3 v2, and R4's QSI stage took
    88.7 s against R3's 91.4 s (×0.970). The whole reference build took 3 m 41 s against 3 m 52 s
    (×0.953), and the other stages were 2–13 % faster.
  - 40–45 h × 0.97 = 38.8–43.7 h, written as 39–44 h.

### 2.3 The one check (prompt §3)

From [`launch_check_07.txt`](launch/launch_check_07.txt), at 21:12:18, 24 s after the launch:
- **Run id** `handover-07-a3-v2-20261022T211155`.
- **`scope`**: `full build: main.py's own 9280-item QuadSourceIntegral work list at atol=1e-32
  rtol=1e-07, 6 cpus, LambdaCDM, 8 log-spaced k over 100000-3e+08 /Mpc, into
  handover-A3-v2-lambdacdm.sqlite`. It begins `full build:` and names every item prompt §3 lists.
- **The manifest.**
  - `git_dirty` **false**; `git_head` **`b90752dda46e57d588f8107cbd4d4557134fadb2`**, the dispatch
    SHA.
  - `host` `id` `d120d1cf03551e6f` with `name` `mps026298.phys.susx.ac.uk`, 06's.
  - `packages` equal as JSON to R4's: `datastorekit` 0.5.0 `f47b9a2…`, `raypoolkit` 0.2.0
    `eaef0ce6…` and `runregistrykit` 0.2.0 `0bae3a3c…`, none editable.
  - `script` `RunDrivers/quadsource_atol_sweep.py`, with `script_sha256` `b2fa4ef1…`, 06's.
  - No `--rehearsal` and no `--resume` in `argv`. `results_store_id` is `null`, and there are 20
    keys.
- **`status.json`.**
  - `state` **`running`**, pid 84436, stage `CALCULATE NUMERICAL PART OF TENSOR GREEN FUNCTIONS`,
    heartbeat 21:12:07.
  - `host` `{"boot": "C58A0F92-…", "id": "d120d1cf03551e6f"}`, `this_host()`'s.
  - `kill -0 84436` ok, and `liveness(status, 900)` is `alive`.
- **`stdout.log`.**
  - It has four `replaced 1 occurrence` lines: the two `np.logspace(np.log10(1e5), np.log10(3e8),
    …)` k grids, `tol=DEFAULT_QUADRATURE_ATOL` and `tol=DEFAULT_QUADRATURE_RTOL`.
  - Its `main.py argv` is `['--database', '…/var/datastores/handover-A3-v2-lambdacdm.sqlite',
    '--job-name', 'handover-A3-v2', '--shards', '4', '--zend', '0.1', '--source-samples-log10z',
    '100']`. That has **`--zend 0.1`** and **no `--no-prune-unvalidated`**.
  - It shows `bootstrapping a local Ray instance (num_cpus=6, include_dashboard=False)`, and the
    launch log shows `Started a local Ray instance`.
  - There is no `!!` line and no traceback, and `stderr.log` is empty.
- **`list`**: 5 runs, "1 alive, 4 finished; 0 with no host recorded". `handover-07-a3-v2-20261022T211155`
  is `running`.
- The load right after the check was 4.83 3.53 3.47, the build's six workers starting.

**The session then ended its turn**, with no wait and no poll (`CLAUDE.md` rule 4). The registry
owns the run. The lock stays held with pid 84436 until prompt 08 releases it. The checkout stays at
`b90752d`, clean, and is not moved while the build runs.

## 3. After the build

**Where the records are.** A path `var/handover-08/…` in this section is that directory on the Mac
Studio. Its `out/`, `scripts/` and `state/` are committed at [`a3-v2/after/`](a3-v2/after/); its
`extract/` (360 MB of reader output, with `extract_GkWKB_data.log`) stays on the Mac Studio, beside
the store, which is kept there (the user, 2026-10-25).

**§3 hit a §8 stop**: `extract_GkWKB_data.py` exited 1. The user ruled, on 2026-10-25, **"Release
the lock, close 08"**. The failure stays an open issue, to be fixed when the reader is re-run, and §6
was then run (§3.8). Every other step is complete.

**The tree.** Dispatch SHA `087d420b222740221450819cb74dedcb0cf3cde2`, read with `git show`. The
checkout stayed at **`b90752d`**, clean; `git diff --stat b90752d 087d420 -- . ':!prompts' ':!docs'`
is empty.

### 3.1 Before anything ([`prechecks_08.txt`](after/prechecks_08.txt))

Every Gate item is as stated:
- **The run.** `handover-07-a3-v2-20261022T211155` is `done`, exit 0, and the launch log ends
  `** driver exit status 0`. `kill -0 84436` gives no such process, and the final heartbeat was
  21:56:27.
- **The fingerprint.** Format 6, `encoding_format` 1, `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`,
  21 classes, problems `{}`, taken at `b90752d` with `git_dirty` false. QSI 9280, `GkSource` 1160,
  `QuadSource` 36. `fingerprint_sidecar` is "not written … (there is none)".
- **The store.** Four shards of 105.8–107.6 MB, last written 2026-10-24T21:56:16, and a 36,864-byte
  primary. No journal.
- **The registry, lock and Ray.** `list` shows 5 runs, all finished. The lock is held, with owner
  84436 not alive. There is no Ray process and `RAY_ADDRESS` is unset.
- **The store files' hashes** are in `state/store_files_before.txt`, and every later comparison is
  against them.

### 3.2 The store's record ([`store_record_08.txt`](after/store_record_08.txt))

- `store create` exited 0, with the prompt's purpose; sidecar `handover-A3-v2-lambdacdm.manifest.json`.
- `store fingerprint --write` gave **`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`**,
  format 6, problems none, "it replaced none". `compare_fingerprints` between the run's recorded
  fingerprint and the sidecar's finds **0 differences** (`datastorekit.store_fingerprint`).
- `store show`: kind `registry`, problems none, and the run named
  (`handover-07-a3-v2-20261022T211155 done finished by results`).
- **Every store file is unchanged** (SHA-256, size, mtime) by `store create` and `fingerprint --write`.

### 3.3 The after-build checks (one store)

| Check | Result |
|---|---|
| `closed_store_state.py` ([`closed_store_state_08.txt`](after/closed_store_state_08.txt)) | one `version` row `(1, '2025.1.1')` per shard; none on the primary; no journal, WAL or shm; `replication_in_flight` 0 rows; no row `validated = 0` |
| `replicated_serial_census.py` ([`census_08.txt`](after/census_08.txt)) | **`no divergence`**; 14 tables `OK`; rc 0 |
| `store_soundness.py` ([`store_soundness_08.txt`](after/store_soundness_08.txt)) | **`=> clean`**: `integrity_check ['ok']` and `foreign_key_check` 0 rows on the primary and all four shards; `read_inventory` over shards 0–3, 0 problems, 0 `duplicate`; counts equal the run's fingerprint in all 21 classes (`redshift` 1740, `wavenumber` 8, `GkNumericIntegration` 4549, `GkWKBIntegration` 13920, `GkSource` 1160, `GkSourcePolicyData` 1160, `QuadSource` 36, `QuadSourceIntegral` 9280, …) |
| store files before and after | identical (`state/store_files_after_checks.txt`) |

### 3.4 The readers ([`readers_08.txt`](after/readers_08.txt))

`run_readers.sh` was run on the A3 v2 primary into `var/handover-08/extract`: the six readers, then
`extract_Gk_data` with `nomodels`, as R4 ran them. They ran 00:35:23–00:39:48, with the lock held.

| Script | Exit | Wall | Lookups (last queue) | Files | Size | Traceback | Store files after |
|---|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 57 s | 522 | 1593 | 100.5 MB | none | identical |
| `extract_GkWKB_data.py` | **1** | 24 s | 8 | 344 | 18.2 MB | **`OverflowError` in `ray::plot_Gk()`** | identical |
| `extract_GkSource_data.py` | 0 | 33 s | 45 | 873 | 70.6 MB | none | identical |
| `extract_TkWKB_data.py` | 0 | 19 s | 8 | 72 | 6.8 MB | none | identical |
| `extract_tensor_source_data.py` | 0 | 22 s | 36 | 172 | 7.5 MB | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 33 s | 240 | 681 | 56.1 MB | none | identical |
| `extract_Gk_data.py`, **no `--models`** | 0 | 56 s | 522; then `>> SKIPPING MODEL QCD_Cosmology: …` | 1593 | 100.5 MB | none | identical |

- **The output was 360 MB in all**, far under the 20 GB limit; R4's was 208 MB.
- No `ReadOnlyMiss` or `ReadOnlyWrite` appeared, and no Ray process was left after any reader.
- **`store fingerprint` afterwards** (no `--write`) found problems none and "matches the fingerprint
  taken 2026-10-25T00:34:14+01:00 by a person, at b90752d…".
- **The failure** (`issues.md`) came from plotting: the full-range PDF of `k = 1e5`, `z_source = 1.3633e16`
  (serial 19) was left as a 1,039-byte fragment with no PNG or CSV. A plotted value there is presumably
  `0`, `inf` or `nan` on a log axis; that was not confirmed against the data. The store is not implicated.

### 3.5 The depth census ([`depth_census_08.txt`](after/depth_census_08.txt))

**The script** is `scripts/depth_census.py`: read-only, no Ray, every file `mode=ro`, following
`report()`. It reads `DEFAULT_LEVIN_MAX_DEPTH = 20` from `AdaptiveLevin/levin_quadrature.py:154`.
**Its test on R4's reference store** ([`depth_census_r4_08.txt`](after/depth_census_r4_08.txt))
counted 864 rows, all at (1e-32, 1e-7), and left every file identical.

**The population.** **9280** rows at (1e-32, 1e-7), 2320 per shard. There are 0 rows at any other
pair, and every store file is unchanged by the census.

| Bin (`WKB_Levin_max_depth`) | A3 v2 rows | mean `compute_time` | CPU-hr | unconverged | Baseline (rtol 1e-8; 7600 rows) |
|---|---|---|---|---|---|
| NULL (no Levin call) | 2489 | 0.035 s | 0.024 | 0 | 2489 at 0.062 s |
| 0 (not in the baseline's bins) | 99 | 0.121 s | 0.003 | 65 | — |
| 1–19 | 4427 | 0.421 s | 0.517 | 1018 | 3054 at 0.776 s |
| **20** | **2265** | **454.2 s** | **285.78** | **1179 (52.1 %)** | **2057 at 170.6 s; 97.5 CPU-hr; 889 unconverged (43.2 %)** |
| **store** | **9280** | — | **286.33** | **2262** | **7600; 98.2 CPU-hr** |

- **`rtol = 1e-7` did not remove the depth-20 cliff.** The depth-20 rows carry 285.8 of the store's
  286.3 CPU-hours (99.8 %; the baseline's 99.3 %). Their mean cost is 454 s, against 0.4 s for
  depth 1–19, and 52.1 % of them are stored unconverged.
- **Rows over 60 s:** 679, of which 524 (77.2 %) are unconverged. The baseline had 195, at 74.4 %.
- **Totals.** `total_phase_limited = 1` on 525 rows. The maximum `WKB_Levin_num_regions` is
  966,508 (baseline: 1,530,190). `SUM(WKB_Levin_elapsed)` is 1,029,462 s of 1,030,782 s of
  `compute_time`, against 2.6 s of `numeric_quad_compute_time`.

**By band:**

| Band | rows | CPU-hr | depth-20 rows (mean) | unconverged at depth 20 | rows > 60 s (unconverged) |
|---|---|---|---|---|---|
| seam, $z > 10^5$ | 6080 | 0.47 | 594 (0.68 s) | 191 | 0 |
| mid, $9 \le z \le 10^5$ | 2112 | 36.73 | 945 (139.4 s) | 456 | 185 (76.2 %) |
| low, $z < 9$ | 1088 | **249.13** | 726 (**1235 s**) | 532 (73.3 %) | 494 (77.5 %) |
| $z > 174.1$ (both stores hold it all) | 7552 | 2.50 | 1153 (6.3 s) | 415 | 31 (64.5 %) |

- **The 20 most expensive rows** took 6,251–7,559 s each. All are at depth 20 and unconverged, with
  822k–967k regions. Every one has the baseline's geometry, `r == k ≫ q`, with `k` = 9.70e6 or
  3.05e7 /Mpc and `z_response` 0.1–2.09. The worst is `(9.7035e6, 9.8506e5, 9.7035e6)` at
  $z = 0.302$: 7,558.6 s, `abserr/|total|` 3.6e-5.
- **Not like-for-like.**
  - The pair differs, which is the change under test.
  - The baseline lacked 1680 items, all at $z \le 174.1$, including the whole low band.
  - No per-band baseline figure exists.
  - At $z > 174.1$ (7552 rows) v2 costs 2.50 CPU-hr, with 1153 rows at depth 20 averaging 6.3 s.
    There is no baseline number for that set. The baseline's 7600 rows were these 7552 plus 48 of
    the 64 at $z = 132.08$.

### 3.6 From the run's logs ([`run_logs_08.txt`](after/run_logs_08.txt))

- **The tail.**
  - The QSI stage ran 21:22:59 Thu → 21:56:16 Sat, **48.55 h** (the log says `2d 33m 16.8s`).
  - Every batch had been handed out after 6 h 35 m. The rest was the six workers draining the
    in-flight items, at full occupancy in the session's snapshots.
  - The last progress line with six or more items in flight was 19:25:17 (52 compute). The next,
    and final, line is 21:56:15, with 0 compute and 52 store.
  - The five-workers-idle state therefore began after 19:25:17 and before 21:48, when a `ps`
    snapshot found five `ray::IDLE` and one at 99 %. So it lasted **at most 2.52 h (5.2 % of the
    stage)**, under §4's 10 %.
  - The last 52 items were stored together at 21:56:16, so the store's timestamps cannot order
    them. The longest of them is `(3.0455e7, 3.0917e6, 3.0455e7)` at $z = 0.174$, 6,580.6 s.
- **The `IntegrationWarning`s.** There are 15 events: 6 "maximum number of subdivisions (100)", 6
  "probably divergent, or slowly convergent" and 3 roundoff. All are SciPy `quad`'s at
  `Quadrature/simple_quadrature.py:92`, in `compute_QuadSource_integral` tasks on six worker pids,
  all within the QSI stage's first 1.6 min (`stderr.log` was last written at 21:24:37). The
  orchestrator's 13 did not count Ray's two `[repeated 2x across cluster]` lines. **No warning names
  a row**, so none can be tied to a stored `QuadSource` row.

### 3.7 The run's work ([`run_record_08.txt`](after/run_record_08.txt))

- **Wall time:** `created` 2026-10-22T21:11:55 → final heartbeat 2026-10-24T21:56:27, **48.74 h**,
  against 39–44 h estimated.
- **CPU:** **286.33 CPU-hr** of `compute_time`, against the ~250 projection (×1.145). By band: low
  249.13 (projected ~233), mid 36.73 (~16), seam 0.47 (~1).
- **Occupancy:** at 6 CPUs, 286.33 CPU-hr is 47.7 h fully occupied, against the 48.74 h it took.
- `list` shows 5 runs, all finished.

### 3.8 The lock ([`lock_08.txt`](after/lock_08.txt))

- **At the stop** (00:40:41) the lock was left as found: held by SecondaryGWKit, purpose "… 07: A3 v2
  build …", owner pid 84436 (not alive), with `ray-processes` empty.
- **On the user's ruling**, at 00:51:03:
  - `ray-processes`: empty, exit 0;
  - **`release`: "released the rehearsal lock", exit 0**;
  - `status`: **`lock: free`** (`/Users/ds283/ScienceData/.rehearsal-lock does not exist`), "Ray
    processes of this repository: none";
  - `ray-processes`: empty.
- `~/ScienceData/` remains, empty.
- **The freeze on the checkout ends here.** It stays at `b90752d`, clean, and is not moved by this
  prompt.
