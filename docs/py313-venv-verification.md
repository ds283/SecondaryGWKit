# py313-venv campaign — the venv on Python 3.13

This is the verification record of the [`py313-venv`](../prompts/py313-venv/README.md) campaign. The
campaign moves the venv on the laptop (`mps026088`) and on the Mac Studio (`mps026298`) from Python
3.12.15 to **3.13.16**, and moves two pins: `ray` 2.43.0 → **2.59.0** (`requirements.txt:75`) and
`SQLAlchemy` 2.0.39 → **2.0.55** (`:88`). Every other pin stays. `runregistrykit.toml`'s
`distributions` gains `ray`, `SQLAlchemy`, `numpy` and `scipy`, so that a manifest says which venv
wrote a store.
- **§1, "R6 — the laptop on Python 3.13"**, is written by **prompt 01**, on branch
  `handover-remedial` at the dispatch `HEAD` **`55c6a54`** ("Write the py313-venv orchestrator note
  for prompt 01"), whose code is `ebfae11`'s. It has the shape of
  [`rrk-v020-adoption-verification.md`](rrk-v020-adoption-verification.md) §1 and §2 (R3), and is
  compared with it. Its §1.10 is R3's hard-kill pair, run again.
- **§2** is prompt 02's (the Mac Studio, R7).

The saved outputs quoted below are in
[`../prompts/py313-venv/measure/out/`](../prompts/py313-venv/measure/out/), under names starting
`01-`. The large ones stay under `var/` (§1.9). Times are the laptop's, GMT: BST ended at 02:00 on
2026-10-25.

## 1. R6 — the laptop on Python 3.13

### 1.0 The venv

| | Before | After |
|---|---|---|
| Interpreter | MacPorts Python **3.12.15** | MacPorts Python **3.13.16** (`/opt/local/bin/python3.13 -m venv venv`) |
| `ray` | 2.43.0 | **2.59.0** |
| `SQLAlchemy` | 2.0.39 | **2.0.55** |
| SQLite (`sqlite3.sqlite_version`) | 3.53.4 | **3.53.4**, unchanged |
| `pip freeze` | 101 lines | 101 lines; **`diff` is exactly the `ray` and `SQLAlchemy` lines** |

([`01-pip.txt`](../prompts/py313-venv/measure/out/01-pip.txt))
- **The baselines were taken first**, under 3.12, before the venv moved: the freeze, the provenance,
  the seven roots and `black`.
- **The old venv was moved, not deleted**, to
  **`~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025`** (817 MB, `pyvenv.cfg`
  `version = 3.12.15`). It is the rollback (README §4.1). Retiring it is the user's.
- **The install.** `./venv/bin/pip install -r requirements.txt` exited 0, with the pip that
  `python3.13 -m venv` wrote (26.2.1), not upgraded. Its only warnings are the two yanked pins
  3.12 resolves too (`aiohttp` 3.11.14, `aiohttp-cors` 0.8.0). `pip check` is clean.
  `python3.13 -m venv` writes `venv/.gitignore` (`*`), and `.gitignore:7` names `venv/` as well, so
  `git status` did not show the venv.
- **The three git packages are at their commits and not editable**: `datastorekit` 0.5.0
  `f47b9a25…` (`v0.5.0`), `raypoolkit` 0.3.0 `67a75b97…` (`v0.3.0`) and `runregistrykit` 0.2.0
  `0bae3a3c…` (`v0.2.0`). `package_provenance` gives `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy`
  2.2.4 and `scipy` 1.16.3, with `url` and `commit` null.
- **`this_host()`** is `id` `8d2b247824882486`, `boot` `376461AF-…`, both R3's, with `python`
  now `3.13.16`.

### 1.0.1 The suites, under 3.12 and 3.13

([`01-roots.txt`](../prompts/py313-venv/measure/out/01-roots.txt)) README §7's command for each
root, serially, in the foreground, at the dispatch `HEAD`. The load was not this repository's: a
ChamPBH agent's test suite ran beside every root at about 99 % of one CPU, and the one-minute load
ran from 8 to 54.

| Root | 3.12 Ran | 3.12 wall | 3.13 Ran | 3.13 wall | Result, both |
|---|---|---|---|---|---|
| AdaptiveLevin | 32 | 1 s | 32 | 4 s | OK |
| CosmologyModels | 39 | 7 s | 39 | 20 s | OK |
| LiouvilleGreen | 148 | 59 s | 148 | 110 s | OK (skipped=1) |
| ObjectFactories | 280 | 35 s | 280 | 61 s | OK |
| RunDrivers | 120 | 16 s | 120 | 27 s | OK |
| tools | 49 | 31 s | 49 | 46 s | OK |
| ComputeTargets | 737 | 272 s | 737 | 330 s | 1 failure, the known wall-clock test |

- **Every count is 3.12's, with the same skip.**
- **The one failure** was `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, README §7's
  known test, and it failed under **both** interpreters. Re-run alone three times each, it failed
  each time, on its cold limit of 0.06 s:
  - 3.12 printed 0.114–0.148 s with the table built and 0.039–0.054 s cached;
  - 3.13 printed 0.078–0.084 s built and 0.027–0.029 s cached.

  So 3.13 is faster here, not slower. The ComputeTargets run with `-W default` under 3.13 passed,
  737 OK, the test included. This is not a finding against 3.13.
- **`test_ray_address_default`**, run on its own under 3.13: 7 tests OK. `ray._private.services`
  still resolves `local` to a new instance under 2.59.0.
- **`black --check --extend-exclude '^/prompts/' .`**: 369 files unchanged, under 3.12, under 3.13,
  and again before the commit ([`01-black.txt`](../prompts/py313-venv/measure/out/01-black.txt)).
- **Warnings.**
  - ComputeTargets under `-W default` gives the same counts as without the flag and as under 3.12:
    - 4 `DeprecationWarning` lines, all this repository's own `bessel_phase` deprecation, raised by
      `ComputeTargets/tests/test_quadsource_integral.py:497-498`;
    - 6 SciPy `IntegrationWarning`;
    - 0 `FutureWarning`;
    - none naming a file under `site-packages`.
  - Across the seven roots the warning kinds are equal but in ObjectFactories. Under 3.13 it has
    **one `ResourceWarning: unclosed database in <sqlite3.Connection …>`**, reported at
    `sqlalchemy/sql/selectable.py:4765`. That is where the collector found the connection, not
    where it was opened.
    - Python 3.13 newly warns when a `sqlite3.Connection` is collected unclosed.
    - `PYTHONTRACEMALLOC` traced it to `ObjectFactories/tests/test_quadsource_parent_key.py:443-444`,
      where `_read` makes `sqla.create_engine("sqlite://")` and never disposes it.
      *Corrected in the review of 01 (board §5.2):* `_read` does dispose of it after its `with`
      block. One test of the class expects `_read` to raise, and the raise skips the call.
    - It is a test's in-memory engine, not a package's and not a store's. It is opened as
      `[01-a-test-leaves-an-in-memory-sqlite-engine-undisposed]` (§1.7).

### 1.0.2 The tree and the manifests

- **The code is `ebfae11`'s.** `git log ebfae11..55c6a54` is the planning commit `1fd1c36` and
  the note `55c6a54`, and `git diff --stat ebfae11 HEAD -- . ':!prompts' ':!docs'` is empty.
- **The working tree was dirty at every launch, in exactly these paths:** ` M CLAUDE.md`,
  ` M requirements.txt`, ` M runregistrykit.toml` and `?? prompts/py313-venv/measure/`. **No `.py`
  file was dirty** at any launch (`01-prelaunch_checks.txt` records `git status --short` at each).
  So every manifest records **`git_dirty: true`**, as the corrected prompt expects. R3's said false.
- **The launcher** is `RunDrivers/quadsource_atol_sweep.py`, SHA-256 `5b984c75…`, last changed in
  `40d10b1` (`rpk-v030-adoption` 01's temp dir). R3 ran `docs/handover/…` at `2a985303…`, and R4
  ran `b2fa4ef1…`.
- **Every manifest has R3's twenty keys.**
  - Its `packages` names **seven** distributions, where R3's named three: `SQLAlchemy`,
    `datastorekit`, `numpy`, `ray`, `raypoolkit`, `runregistrykit` and `scipy`. It equals the
    record taken after the install, as JSON.
  - Against R3, `runregistrykit` is equal. `datastorekit` and `raypoolkit` differ by the pins moved
    since (`dk-v050-adoption`, `rpk-v030-adoption`), and the other four are new.
  - Its `host.python` is **`3.13.16`**, and every status's `host` `id` and `boot` are this
    machine's ([`01-run_records.txt`](../prompts/py313-venv/measure/out/01-run_records.txt)).

### 1.1 How each was launched

Each launch was R3's command (`rrk-v020-adoption-verification.md` §1.1), with R4's launcher path,
`py313-venv-01`, `--campaign py313-venv --prompt 01` and this prompt's store names, detached, with
the purpose as `sh -c`'s `$0`:

```bash
nohup sh -c "PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/py313-venv-01-<store>.sqlite \
    --job-name py313-venv-01 --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose \"\$0\" --campaign py313-venv --prompt 01; \
  echo \"** driver exit status \$?\"" "<one line>" \
  > var/runs/py313-venv-01-<slug>.launch.log 2>&1 &
disown
```

Each `--purpose` names:
- R6 of `py313-venv` prompt 01 on the laptop, and the tree, `55c6a54`;
- Python 3.13.16, `ray` 2.59.0 and `SQLAlchemy` 2.0.55, and the three packages;
- the fingerprint to equal, `39809dca`, and R3's run that recorded it,
  `rrk-v020-adoption-03-reference-20261018T232838`.

None names `/tmp/ray-sgk`.

**One check per launch, and the waits were the orchestrator's.** After each launch the agent made
one check, then ended its turn and reported the driver's pid, the `sh` pid and the run id. The
check covered the manifest, the status, `list` and, for the killed launches, the interrupter. The
orchestrator waited on the pid in the background and resumed the agent with the final
`status.json`. Neither of them polled, and nobody ran `pgrep -f`.

| Slug | Run id | Store | `--resume` | Python pid (`sh`) | State | Exit | Driver exit status | Wall time | Load at launch |
|---|---|---|---|---|---|---|---|---|---|
| `reference` | `py313-venv-01-reference-20261025T125115` | reference | no | 87658 (87651) | `done` | 0 | 0 | **16 m 33 s** (12:51:15–13:07:48) | 12.6, then 39.6 a minute in |
| `resumed-a` | `py313-venv-01-resumed-a-20261025T131024` | resumed | no | 91483 (91477) | `killed` | 143 | 143 | 3 m 52 s (13:10:24–13:14:16) | 8.7 |
| `resumed-b` | `py313-venv-01-resumed-b-20261025T131514` | resumed | yes | 92639 (92634) | `killed` | 143 | 143 | 3 m 18 s (13:15:14–13:18:32) | 8.2 |
| `resumed-c` | `py313-venv-01-resumed-c-20261025T131935` | resumed | yes | 94127 (94121) | `done` | 0 | 0 | 2 m 29 s (13:19:35–13:22:04) | 8.7 |

R3's wall times were 3 m 52 s, 2 m 18 s, 1 m 19 s and 1 m 46 s. **`reference` ran each stage 3–4
times slower than R3's** at a one-minute load of 12–40: the ChamPBH suite ran beside it, with
Spotlight's indexer. Single-threaded, 3.13 timed faster than 3.12 (§1.0.1), and the later
launches, at a load of about 8–20, came closer to R3's times. The stage times are recorded, not
compared (§1.5.5).

**The sidecars** ([`01-store_sidecars.txt`](../prompts/py313-venv/measure/out/01-store_sidecars.txt)):
- **The reference store.** `store create` (store_id `7fc02cda82be4c8c9a67f012139ccf51`) ran after
  `reference`, then `store fingerprint --write`: `39809dca…`, problems none, "it replaced none".
- **The resumed store.** `store create` (store_id `23b16cdbf76c43c1878558ab46f7f8cc`) ran after
  `resumed-a`.
- The store files were identical across each.

#### 1.1.1 The pre-launch checks

([`01-prelaunch_checks.txt`](../prompts/py313-venv/measure/out/01-prelaunch_checks.txt))
- **The tree.** `HEAD` was `55c6a547db490794cf7e3f0cb478c7b9d29aa744` and `git worktree list` showed
  this checkout alone. `git status --short` was as §1.0.2 says, with no dirty `.py` file.
- **The environment.** `RAY_ADDRESS` was unset, and the machine has 10 CPUs.
- **The registry.** Before `reference`, `list` showed **53 runs: "51 finished, 2 unknown; 45 with
  no host recorded"**. Nothing was `running` or stale before any launch.
- **No Ray process of this repository.** Before each launch, `ray-processes --ray-temp-dir
  /tmp/ray-sgk` exited 0 (`01-lock.txt`). `ps -p` checked the previous run's pids, those of the
  `sh`, the driver, the interrupter and every Ray pid its logs name, and found none alive. R3's
  machine-wide `pgrep -lf` was not run (deviation 2).
- **The store names.** All three were `absent` to `store show`, with no runs. No file under `var/`
  outside `var/py313-venv-01/` contained `py313-venv`, and there was no journal under
  `var/datastores/`.
- **The packages and the configuration.** `pip show` gives the seven distributions, each in this
  venv's `site-packages`, and `pip check` is clean.
  - `find_client_root()` reads `distributions` as `('datastorekit', 'raypoolkit', 'ray',
    'SQLAlchemy', 'numpy', 'scipy')`, and the fingerprinter is `config.fingerprint:fingerprinter`.
  - `TERMINAL_STATES` is `('done', 'failed', 'killed', 'lost')`.
  - `ENCODING_FORMAT` is 1 and `FINGERPRINT_FORMAT` is 6.
  - `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`, and `ray.is_initialized()` is false.
- **R3's recorded fingerprints.** R3's `reference` and `resumed-c` are each `done`/0, format 6,
  `encoding_format` 1, `39809dca…`, 21 classes, problems `{}`.
- **`var/datastores/`** held 26 files, digest over the listing `3e4ebfda…` (§1.9).
- **Disk.** `df` said 95 % used, with 55–56 GB free. R3's disk was at 93 %.

#### 1.1.2 The interrupter

**The dummy tests**, under 3.13, before the first launch, each into a new directory in the agent's
scratchpad, both exited 0 (12:49:41–12:50:08;
[`01-interrupter_dummy_test.txt`](../prompts/py313-venv/measure/out/01-interrupter_dummy_test.txt),
[`01-interrupter_build_dummy_test.txt`](../prompts/py313-venv/measure/out/01-interrupter_build_dummy_test.txt)).
- With times, pids and paths normalised, the scoped output (40 lines, cases A–E) equals R3's and
  R4's.
- The build output (28 lines, cases A–D) equals theirs but for one pid in bash's own
  `line 77: <pid> Killed: 9` line.
- Case D gave child exit 137. No dummy pid was left alive.

**The points, from R6's own `reference` `stdout.log`, by R3's and R4's reasoning**
([`01-interrupter_firings.txt`](../prompts/py313-venv/measure/out/01-interrupter_firings.txt)):
- **`resumed-a`.** `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT` took **82.5 s**
  (`stdout.log:78`; R3 24.8 s). 40 % of that is 33 s, over the 10 s floor:
  `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$' 0 --interval 5 --hold 33`.
- **`resumed-b`.** `CALCULATE QUADRATIC SOURCE INTEGRALS` took **351.6 s** (`stdout.log:104`; R3
  91.4 s). A third is 117 s, under the 250 s cap: `'^CALCULATE QUADRATIC SOURCE INTEGRALS' 0
  --interval 5 --hold 117`.
- The holds are longer than R3's because the stage times were.

**The firings:**

```
resumed-a (interrupter pid 91499)
13:13:40 liveness alive; pid 91483; stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT'
13:13:40 eligible …; holding 33 s, then reading again
13:14:13 sent SIGTERM to pid 91483 at stage 'CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT' (0% >= 0%); … exiting

resumed-b (interrupter pid 92656)
13:16:29 liveness alive; pid 92639; stage 'CALCULATE QUADRATIC SOURCE INTEGRALS'
13:16:29 eligible …; holding 117 s, then reading again
13:18:26 sent SIGTERM to pid 92639 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%); … exiting
```

- Each signal came at the hold, 33–38 s and 117–122 s into its stage. Each interrupter checked the
  pid's command line first, then exited.
- `resumed-b`'s QSI stage began 75 s after its launch, because its earlier stages were lookups. It
  was earlier than the agent had estimated, but where the hold put it.

#### 1.1.3 The lock

Every lock command, its output and its exit status are in
[`01-lock.txt`](../prompts/py313-venv/measure/out/01-lock.txt). Each was run as a **shell command of
its own**, the lock command the only process that command started. The header was zsh's builtin
`print -P`, the output went by `>> … 2>&1`, and the exit was recorded with the builtin `echo`. No
pipe was used, and no `--purpose` names the temp dir.

| When | Command | Exit | What it said |
|---|---|---|---|
| 12:12:5x | `status`, `ray-processes` (the Gate) | 0, 0 | `lock: free`; none |
| 12:51:03 | `acquire --purpose "SecondaryGWKit py313-venv 01 (R6)"` | 0 | acquired, for SecondaryGWKit |
| 12:51:06 | `status` | 0 | `HELD`, `ray_temp_dir: /tmp/ray-sgk`; no pid yet; none |
| 12:51:41 | `set-pid 87658` (`reference`) | 0 | recorded |
| 12:51:43 | `ray-processes` | **1** | **the 2.59.0 instance: 22 lines, `gcs_server` 87699 and `raylet` 87745 among them** |
| 12:51:46 | `status` | 0 | `pid: 87658`, alive; 22 Ray processes of `/tmp/ray-sgk` |
| 13:08:43 | `ray-processes` (after `reference`) | 0 | none |
| 13:10, 13:10:32, 13:10:34 | `ray-processes`; `set-pid 91483` (`resumed-a`); `status` | 0, 0, 0 | none; recorded; alive, 13 |
| 13:14:43 | `ray-processes` | 0 | none |
| 13:15, 13:15:2x | `ray-processes`; `set-pid 92639` (`resumed-b`); `status` | 0, 0, 0 | none; recorded; alive, 13 |
| 13:18:58 | `ray-processes` | 0 | none |
| 13:19, 13:19:5x | `ray-processes`; `set-pid 94127` (`resumed-c`); `status` | 0, 0, 0 | none; recorded; alive, 13 |
| 13:22–13:29 | `ray-processes`; `status` before the readers; `status` before `hardkill-a` | 0 | none; pid NOT alive; none |
| 13:29:3x | `set-pid 98329` (`hardkill-a`); `status` | 0, 0 | recorded; alive, 17 |
| 13:35:50 | `ray-processes` (after the `SIGKILL`) | **1** | **one: `98369 1 …/gcs_server --log_dir=/tmp/ray-sgk/session_2026-10-25`** (§1.10.3) |
| 15:31:32 | `ray-processes` (after the user's kill) | 0 | none |
| 15:32, 15:32, 15:33 | `ray-processes`, before the refused resume, after it, before `hardkill-b` | 0 | none |
| — | `set-pid` for `hardkill-b` | **not run** | the agent's slip (deviation 6) |
| 15:35:4x | `ray-processes` (after `hardkill-b`) | 0 | none |
| 15:37:08 | `release` | 0 | released |
| 15:37:09 | `status` | 0 | `lock: free`; `Ray processes of /tmp/ray-sgk: none` |
| 15:37:10 | `ray-processes` | 0 | none |

- **The lock sees a Ray 2.59.0 instance by its temp dir.** Its `ray-processes` listed `reference`'s
  instance, `gcs_server` and `raylet` among the lines, and every count `status` gave was of
  `/tmp/ray-sgk`. 2.59.0's processes include two names that 2.43.0 does not show as separate lines,
  `ray::DashboardAgent` and `ray::RuntimeEnvAgent`.
- The owner file named `project: SecondaryGWKit`, `purpose: SecondaryGWKit py313-venv 01 (R6)`,
  `host: mps026088`, `ray_temp_dir: /tmp/ray-sgk` and `ray_address: in-process` throughout.
- Between launches the recorded pid was the previous driver's, dead: that is the protocol, not a
  stale lock. From 13:35:14 to 15:31:14 the lock was held with a Ray process of this repository
  alive (§1.10.3).
- **`hardkill-b` ran without its pid in the lock** (deviation 6). For its 105 s the lock was held,
  but its owner pid named `hardkill-a`'s dead driver, 98329, not `hardkill-b`'s live one. It was not
  back-filled, as R4's was not.
- After the release, `~/.raypoolkit/` is empty and `~/ScienceData/.rehearsal-lock` does not exist.

### 1.2 L1 — every launch ends as recorded

#### 1.2.1 The states

| Slug | `state` | `exit_code` | `fingerprint` | `fingerprint_sidecar` | Launch log ends |
|---|---|---|---|---|---|
| `reference` | `done` | 0 | format 6, `encoding_format` 1, **`39809dca…`**, problems `{}` | not written: "…py313-venv-01-reference.manifest.json is not a problem-free registry sidecar (there is none)" | `** driver exit status 0` |
| `resumed-a` | `killed` | 143 | format 6, `encoding_format` 1, `55f890d9…`, problems `{}` | not written: "… (there is none)" | `** driver exit status 143` |
| `resumed-b` | `killed` | 143 | format 6, `encoding_format` 1, `7f69cbaa…`, problems `{}` | `written` | `** driver exit status 143` |
| `resumed-c` | `done` | 0 | format 6, `encoding_format` 1, **`39809dca…`**, problems `{}` | `written` | `** driver exit status 0` |

- Each fingerprint holds 21 classes, and each `taken` names its run, `git_head` `55c6a547…` and
  `git_dirty` true.
- The full intermediate digests are `resumed-a`
  `55f890d90f094f36c38fce5580adf9d804ce2fe095cae390bd733b098a2a1a8d` and `resumed-b`
  `7f69cbaa3d72271904d3bd277781160e9ded852d1d7f4a3e68dfef25224acbbe`. They differ from R3's
  (`c60e0d19…`, `7e41cbe2…`), as they must, since each digests a store killed at a different
  point.
  - `resumed-a`'s store held `GkSource` 180, `GkSourcePolicyData` 0 and QSI 0.
  - `resumed-b`'s held 291, 291 and 541.

#### 1.2.2 Each killed launch

| | `resumed-a` | `resumed-b` |
|---|---|---|
| Interrupter's `sent SIGTERM` | 13:14:13 | 13:18:26 |
| Handler's line | `** SIGTERM at 2026-10-25 13:14:13 +0000: recorded killed, exit code 143; stopping the pipeline; the process will exit with status 143` | `** SIGTERM at 2026-10-25 13:18:26 +0000: recorded killed, exit code 143; …` |
| `stderr.log`'s handler line | `… the pipeline ended with SystemExit(143); …` | `… the pipeline ended with SystemExit(1); …` |
| Python traceback | none | none |
| `status.json.*.tmp` left | **none** | **none** |
| Stage at the signal | `GkSource`, 33–38 s in (**180 of 291 stored, 42 unvalidated**) | QSI, 117–122 s in (**541** of 864 stored) |
| Driver gone (the orchestrator's wait) | 13:14:24 | 13:18:40 |
| Ray processes of `/tmp/ray-sgk` gone | by 13:14:43, **≤ 30 s after the signal** | by 13:18:58, **≤ 32 s** |

- `SystemExit(143)` against `SystemExit(1)` depends on whether the signal lands inside `ray.wait`
  (R2 §3.2.2). R3 had `SystemExit(1)` in both, and R4 had 1 and 143. The record is `killed`/143
  and the exit is 143 either way.
- **No Ray process outlived either `SIGTERM`ed driver**, as at R0–R4, so no wait by pid was needed.
  R3's were gone within about 56 s and 29 s of the final heartbeat.

#### 1.2.3 No Ray process of this repository left

After each launch, `ray-processes` exited 0 and `ps -p` found none alive of:
- the `sh`, the driver and the interrupter;
- the Ray pids the logs name: `reference` 87773–87775 and 87812; `resumed-a` 91556, 91557 and
  91560; `resumed-b` 92710–92713 and 92832; `resumed-c` 94248 and 94251.

§1.10.3 covers the `SIGKILL`.

#### 1.2.4 The manifests and the hosts

As §1.0.2 says, for all six runs ([`01-run_records.txt`](../prompts/py313-venv/measure/out/01-run_records.txt)):
- `git_head` is the dispatch `HEAD` and `git_dirty` is true;
- `packages` is the seven-distribution record, equal as JSON;
- `script` is `RunDrivers/quadsource_atol_sweep.py`, with `5b984c75…`;
- there are twenty keys, and `host` is this machine's with `python` `3.13.16`;
- `argv` holds `--resume` in `resumed-b`, `resumed-c` and `hardkill-b` only;
- `results_store_id` is null for `reference`, `resumed-a` and `hardkill-a`, and the store's
  sidecar id for the resumes.

#### 1.2.5 Every launch's logs, and what is new in them

([`01-log_comparison.txt`](../prompts/py313-venv/measure/out/01-log_comparison.txt), against R3's
same slug, whose raw logs are all on this machine.) Section 1's counts are equal to R3's for every
search, slug and file but four:
- `SystemExit(1)`/`SystemExit(143)` in `resumed-a` (§1.2.2);
- `IntegrationWarning`, which moves from R3's `resumed-c` and `hardkill-b` to R6's `resumed-b` and
  `hardkill-a`. SciPy warns once, from the one `QuadSourceIntegral` that raises it, in whichever run
  computes it. R6's `resumed-b` (541 stored) and `hardkill-a` (536) computed it before their
  signals, and R3's (181, 238) did not. This is R4's explanation.

Zero in every log, as at R3:
- `Traceback`, `ModuleNotFoundError`, `ImportError`, `AttributeError`, `pickle`, `Datastore.SQL`;
- the actor errors, `ValueError`, `RuntimeWarning`, `did not validate`;
- `DeprecationWarning`, `FutureWarning`, `ResourceWarning`, `ConfigError`, `RunRegistry` (not
  `Kit`);
- `datastorekit`;
- the raylet's disk-space line (`df` said 95 %, but Ray did not warn).

`runregistrykit` appears once, in the banner, and `INTEGRITY WARNING` twice in `resumed-b`, both as
at R3.

**The line kinds, normalised as R3's and R4's were** (section 2):

| Slug | `stdout.log` | `stderr.log` | launch log |
|---|---|---|---|
| `reference` | 114 against 110: **the progress and `inflight` kinds** (2 + 2) | 4 against 4; no kind differs | 122 against 116: the progress kinds, and **Ray's token-auth lines** |
| `resumed-a` | 84 against 84; no kind differs | 2 against 2; no kind differs | 89 against 88: the token-auth `WARNING` |
| `resumed-b` | 139 against 139; no kind differs | 6 against 2: the `IntegrationWarning` | 149 against 144: the same, and the token-auth `WARNING` |
| `resumed-c` | 107 against 107; no kind differs | 0 against 4: the `IntegrationWarning` moved | 111 against 114: the same, and the token-auth `WARNING` |

**Every difference, explained:**
1. **The progress lines in `reference`.** Two stages ran past the 300 s notify interval, the Gk WKB
   stage (5 m 18 s) and the QSI stage (5 m 51.6 s). Each printed one progress line and one
   `inflight` line. R3's stages ran under 300 s, so R3 printed none. The cause is the load (§1.1).
2. **Ray 2.59.0's token authentication, the one new line from the upgrade.** Every launch log, and
   three reader logs, carry
   `WARNING authentication_token_setup.py:… -- Token authentication is enabled for this Ray
   cluster. Set RAY_AUTH_MODE=disabled to opt out.`
   - `reference`'s, the first Ray start under 2.59.0, also carries `INFO … Generated new
     authentication token and saved to /Users/ds283/.ray/auth_token`. That file was written at
     12:51, 64 bytes, mode 644, **outside the checkout**.
   - It bears on nothing a build writes or a fingerprint reads, since §1.3 is unchanged. But it is
     a new file outside `var/` that every Ray 2.59.0 instance on this machine will read, and a
     warning in every log.
   - Opened as `[01-ray-2.59-enables-token-auth-and-writes-a-token-into-home]` (§1.7).
3. **The `IntegrationWarning`'s move**, as above.
4. **No disk-space warning**, as at R3 and R4.

No new line names `runregistrykit`, `datastorekit` or `raypoolkit`. No `DeprecationWarning`,
traceback or refusal came from the interpreter or the pins.

### 1.3 L2 — the resume is the reference

#### 1.3.1 Nothing stored is recomputed

([`01-stage_counts.txt`](../prompts/py313-venv/measure/out/01-stage_counts.txt), lookup / compute / store)

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| HORIZON EXIT TIMES, SOURCE K-SAMPLE | 3/3/3 | 3/3/3 | 3/0/0 | 3/0/0 |
| HORIZON EXIT TIMES, RESPONSE K-SAMPLE | 3/0/0 | 3/0/0 | 3/0/0 | 3/0/0 |
| NUMERICAL PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| WKB PART OF MATTER TRANSFER FUNCTIONS | 3/3/3 | 3/3/3 | 0/0/0 | 0/0/0 |
| QUADRATIC SOURCE TERMS | 6/6/6 | 6/6/6 | 0/0/0 | 0/0/0 |
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 1368/1368/1368 | 1368/1368/1368 | 0/0/0 | 0/0/0 |
| WKB PART OF TENSOR GREEN FUNCTIONS | 3477/3477/3477 | 3477/3477/3477 | 0/0/0 | 0/0/0 |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | 291/291/291 | (no progress line) | **153/153/153** | 0/0/0 |
| APPLY GKSOURCE POLICIES | 291/291/291 | — | 291/291/291 | 0/0/0 |
| QUADRATIC SOURCE INTEGRALS | 864/864/864 | — | (no progress line) | **323/323/323** |

- Every stage that finished before an interruption computed zero on the next launch.
- **`GkSource`.** `resumed-a` stored **180** (59 + 60 + 0 + 61), **42** of them unvalidated
  (15 + 14 + 0 + 13). `resumed-b` pruned those 42 at open and computed **153 = 291 − 180 + 42**.
- **QSI.** `resumed-b` stored **541**, and `resumed-c` computed **323 = 864 − 541**.
- The `reference` and `resumed-a` columns equal R3's (and R4's) stage by stage.

#### 1.3.2 The prune at startup

([`01-closed_store_state.txt`](../prompts/py313-venv/measure/out/01-closed_store_state.txt),
[`01-prune_check.txt`](../prompts/py313-venv/measure/out/01-prune_check.txt)) The closed store
was read `mode=ro` after each launch:

| After | journal | `replication_in_flight` | Rows `validated = 0` |
|---|---|---|---|
| `reference` | none | 0 | none |
| `resumed-a` | none | 0 | **`GkSource` 15 + 14 + 0 + 13 = 42**, of 180, each listed by serial |
| `resumed-b` | none | 0 | none (`GkSource` 97 + 97 + 0 + 97) |
| `resumed-c` | none | 0 | none |

- `prune_check.py` against the state after `resumed-a` gave `42 serial(s) checked: 42 absent, 0
  reused by a validated row, 0 still unvalidated`, after `resumed-b` and again after `resumed-c`,
  exit 0. Against the state after `resumed-b` it gave `none: there is nothing to check`.
- **So the prune at open removed exactly the rows `resumed-a` left unvalidated**, and no validated
  row. `resumed-b`'s `INTEGRITY WARNING` lists them, in 2 lines of `stdout.log` and 2 of the
  launch log, as at R3. (R3 left 168 and 48; R4 164 and 43.)

#### 1.3.3 The version row

Exactly one row, `(1, '2025.1.1')`, is on each shard of each store. The primary has no `version`
table.

#### 1.3.4 The comparison

**The two stores**, by `compare_stores.py`
([`01-fingerprint_comparison.txt`](../prompts/py313-venv/measure/out/01-fingerprint_comparison.txt)):
both are format 6, `39809dca…`, problems none, each with 0 differences against its sidecar.
`compare_fingerprints(reference, resumed): 0 difference(s)`, verdict `equal`, exit 0.

**Against R3's recorded fingerprints**, by `compare_with_recorded.py` as it stands, run through
`bash` ([`01-fingerprint_against_recorded.txt`](../prompts/py313-venv/measure/out/01-fingerprint_against_recorded.txt)):

```bash
PYTHONPATH=. ./venv/bin/python docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py \
    var/datastores/py313-venv-01-reference.sqlite var/datastores/py313-venv-01-resumed.sqlite \
    --against var/runs/rrk-v020-adoption-03-reference-20261018T232838 \
    var/runs/rrk-v020-adoption-03-resumed-c-20261018T233941
```

Exit 0, verdict `equal`. Each store has 0 differences against its sidecar and against **R3's
`reference` and `resumed-c`**, which were taken at `aab40752…` under 3.12.15 and `ray` 2.43.0.

**FP6 holds: Python 3.13.16, `ray` 2.59.0 and `SQLAlchemy` 2.0.55 change nothing a build writes
or a fingerprint reads.** The store files and the sidecars were identical before and after all
of §1.3's reads.

#### 1.3.5 Integrity, both stores, read-only

- **The census** ([`01-census.txt`](../prompts/py313-venv/measure/out/01-census.txt)): every table
  `OK` on both stores (28 lines), `no divergence`, exit 0.
- **`store_soundness.py`** ([`01-store_soundness.txt`](../prompts/py313-venv/measure/out/01-store_soundness.txt)):
  - on the primary and every shard, opened `mode=ro`, `integrity_check` is `['ok']` and
    `foreign_key_check` returns 0 rows;
  - `read_inventory` reports 0 problems and 0 `duplicate` on both stores;
  - `=> clean`, exit 0.

### 1.4 L3 — every reader, read-only, on the resumed store

Through `run_readers.sh`, unchanged, with R3's two commands, the output in
`var/py313-venv-01/extract/` and the lock held
([`01-readers.txt`](../prompts/py313-venv/measure/out/01-readers.txt)). Both invocations exited 0
(13:23:30–13:28:37).

| Script | Exit | Wall | Lookups (last queue) | Files written (R3) | Traceback | Store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 39 s | 116 | 540 (468) | none | identical |
| `extract_GkWKB_data.py` | 0 | 94 s | 348 | 2436 (2436) | none | identical |
| `extract_GkSource_data.py` | 0 | 42 s | 10 | 186 (156) | none | identical |
| `extract_TkWKB_data.py` | 0 | 24 s | 3 | 27 (27) | none | identical |
| `extract_tensor_source_data.py` | 0 | 23 s | 6 | 27 (27) | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 28 s | 24 | 69 (70) | none | identical |
| `extract_Gk_data.py`, no `--models` | 0 | 39 s | 116; then `>> SKIPPING MODEL QCD_Cosmology …` | 405 (378) | none | identical |

- **The lookups equal R3's.** The files written differ where a reader draws a random subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader `run_readers.sh` counts 0 Ray processes of this checkout's venv.
- `extract_GkSource_data`'s log has 3 `MatplotlibDeprecationWarning` lines from its own `:671`, where
  R3's had 2. The count follows the subsample, so it is not a new kind of line.
- **The fingerprint afterwards**: `store fingerprint` gives `39809dca…`, "matches the fingerprint
  taken 2026-10-25T13:22:02+00:00 by run py313-venv-01-resumed-c-20261025T131935, at 55c6a547…
  (dirty)".

**A3 v2's guard** ([`01-guard.txt`](../prompts/py313-venv/measure/out/01-guard.txt)), in the
foreground with no `--register` and the lock held, exited 1 within a second. It printed the clean
refusal `!! quadsource_atol_sweep --build: py313-venv-01-resumed.sqlite, … already exist(s).
Refusing to build into an existing store: …`, with no traceback, as at R4. Every file of R6's
stores and sidecars was identical, and `var/runs/` held 108 entries before and after.

### 1.5 L4 — equal to R3, and to R4

#### 1.5.1 The statistics block

`policy_statistics.py` exited 0 on every run but `resumed-a`, which has no block, as at R3
([`01-policy_statistics_*.txt`](../prompts/py313-venv/measure/out/)). Every count is R3's and R4's:
291 instances; numeric 109 (109 complete); mixed 26 (25 complete, 1 minimal); WKB 156 (156
complete).
- `reference`, `resumed-c` and `hardkill-b` are byte-identical to R3's same slug.
- `resumed-b` and `hardkill-a` differ from R3's only in the order of the two `mixed` lines, and are
  equal when sorted (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).

#### 1.5.2 The reference stage counts

The `reference` column of §1.3.1 equals R3's and R4's in every stage.

#### 1.5.3 The resumed chain, by identity

| Identity | R3 | R4 | R6 | Holds |
|---|---|---|---|---|
| A stage finished before a kill computes 0 on the next launch | yes | yes | yes | ✅ |
| `GkSource` pruned at `resumed-b`'s open = left unvalidated by `resumed-a` | 48 = 48 | 43 = 43 | **42 = 42**, all absent, none reused | ✅ |
| `resumed-b` computes 291 − stored + pruned | 171 | 170 | **153 = 291 − 180 + 42** | ✅ |
| `resumed-c` computes 864 − stored by `resumed-b` | 683 | 578 | **323 = 864 − 541** | ✅ |
| Totals per class equal the reference's | 291; 864 | 291; 864 | 180 + 153 − 42 = 291; 541 + 323 = 864 | ✅ |

#### 1.5.4 The stage sequence and the printed lines

Section 3 of `01-log_comparison.txt` covers this.
- In all six runs the `** ` sequence, normalised, equals R3's in order, and the stage banners equal
  R3's verbatim (13/13, and 11/11 in `resumed-a`).
- The line counts (`** `, `ALL WORK ITEMS COMPLETE`, `Queue summary`, no-compute, progress,
  `inflight`) are:

  | Slug | R6 | R3 and R4 |
  |---|---|---|
  | `reference` | **(26, 10, 10, 1, 2, 2)** | (26, 10, 10, 1, 0, 0) |
  | `resumed-a` | (23, 7, 7, 1, 0, 0) | the same |
  | `resumed-b` | (26, 9, 9, 7, 0, 0) | the same |
  | `resumed-c` | (26, 10, 10, 9, 0, 0) | the same |

  `reference`'s progress lines are difference 1 of §1.2.5.

#### 1.5.5 The stage times

These are not compared. R3's are in brackets.

| Stage | `reference` | `resumed-a` | `resumed-b` | `resumed-c` |
|---|---|---|---|---|
| NUMERICAL PART OF TENSOR GREEN FUNCTIONS | 3 m 5.05 s (26.2 s) | 56.5 s (34.3 s) | 3.38 s (1.74 s) | 3.52 s (1.61 s) |
| WKB PART OF TENSOR GREEN FUNCTIONS | 5 m 18.1 s (1 m 6.82 s) | 1 m 53.3 s (1 m 18.6 s) | 7.61 s (3.64 s) | 7.67 s (3.33 s) |
| GREENS FUNCTIONS FOR SOURCE REDSHIFT | **1 m 22.5 s** (24.8 s) | killed | 33.7 s (21.4 s) | 9.49 s (5.15 s) |
| APPLY GKSOURCE POLICIES | 21.6 s (9.67 s) | — | 13.1 s (10.4 s) | 10.1 s (4.98 s) |
| QUADRATIC SOURCE INTEGRALS | **5 m 51.6 s** (1 m 31.4 s) | — | killed | 1 m 39.1 s (1 m 23.4 s) |

`hardkill-b`, at a load of about 5–12, ran each stage within 10 % of R3's `hardkill-b`: QSI took
1 m 18 s against R3's 1 m 22.5 s. So the slow `reference` was the machine's load, not 3.13 or Ray
2.59.0.

#### 1.5.6 The registry's fields at each run's end

These are R3's. `reference` and `resumed-c` end `done` at `CALCULATE QUADRATIC SOURCE INTEGRALS,
all work items complete`. `resumed-a` ends `killed` at `CALCULATE GREENS FUNCTIONS FOR SOURCE
REDSHIFT`, and `resumed-b` ends `killed` at `CALCULATE QUADRATIC SOURCE INTEGRALS`. `units_done` is
0 and `units_total` is `null` in every run.

### 1.6 What was expected not to happen

Each of these was looked for, and none happened:
- an unvalidated row left after a completed stage;
- an in-flight record;
- a repair at open;
- a Ray agent outliving a `SIGTERM`ed driver;
- a temporary status file;
- a missed interruption;
- a refused `release`;
- a dirty `.py` file at a launch;
- a line naming a package frame, a `ConfigError` or a refusal in a build log.

**One record was written outside `var/` by the build, not by the agent:** Ray 2.59.0's
`~/.ray/auth_token` (§1.2.5, difference 2).

### 1.7 Issues opened

Both are on the [board](../prompts/py313-venv/IMPLEMENTATION_STATE.md) §3 and indexed in
`docs/OPEN_ISSUES.md` §1.23.
- **`[01-ray-2.59-enables-token-auth-and-writes-a-token-into-home]`**: Ray 2.59.0 turns on token
  authentication for a local cluster. It warns at every start and writes `~/.ray/auth_token`, mode
  644. It changes nothing stored.
- **`[01-a-test-leaves-an-in-memory-sqlite-engine-undisposed]`**: Python 3.13's new
  `ResourceWarning`, from `ObjectFactories/tests/test_quadsource_parent_key.py:443-444`.

§1.10 narrows `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` (§1.10.10). Every other
difference from R3 is explained by:
- the load: the stage times and `reference`'s progress lines;
- where the signals fell: the counts of §1.3, `SystemExit` and the `IntegrationWarning`;
- the readers' random subsamples;
- `rrk-v020-adoption` 05's clean refusals.

### 1.8 The suites

These are §1.0.1, before the first launch. No `.py` file was added or changed afterwards, and
`black --check` was clean again before the commit.

### 1.9 What stays under `var/`

- **Runs:** `var/runs/py313-venv-01-{reference,resumed-a,resumed-b,resumed-c,hardkill-a,hardkill-b}-<stamp>/`,
  with their seven launch logs beside them. One of the logs is the refused launch's
  `py313-venv-01-hardkill-b-refused.launch.log`, which made no run directory.
- **Stores:** `var/datastores/py313-venv-01-{reference,resumed,hardkill}.sqlite`, each with four
  shards and a sidecar. None is retired; they are the user's to retire.
- **`var/py313-venv-01/`:**
  - `scripts/`, the agent's throwaway helpers;
  - `state/`, holding the freezes, the closed-store states, the hashes and `hardkill-a`'s status
    before and after `end`;
  - `roots/`, the suites' full outputs, including the traced `ResourceWarning`;
  - `extract/`, the readers' output.
- **Nothing else moved** ([`01-var_datastores.txt`](../prompts/py313-venv/measure/out/01-var_datastores.txt)).
  `var/datastores/` held 26 files before and 44 after. The 26 are unchanged in SHA-256, size and
  `mtime_ns`, and the 18 new files are R6's three stores and their sidecars. There is no journal.
- **The registry after R6** ([`01-run_records.txt`](../prompts/py313-venv/measure/out/01-run_records.txt)):
  - 59 runs: "57 finished, 2 unknown; 45 with no host recorded";
  - nothing `running` or stale;
  - by state, 32 `done`, 20 `killed`, 3 `failed`, 2 `lost` and 2 `unknown`.
- **Outside the checkout:**
  - `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025`, the old venv, moved;
  - `~/.ray/auth_token`, written by Ray;
  - `~/.raypoolkit/`, empty.

### 1.10 The hard-kill route on Ray 2.59.0

R3's §2, with `01` in place of `03`, run after `resumed-c`, the readers and the guard, under the
same lock, on `var/datastores/py313-venv-01-hardkill.sqlite`. Every command and its output, in
order, is in [`01-hardkill.txt`](../prompts/py313-venv/measure/out/01-hardkill.txt) (prompt 01
§2.9).

**Headline.**
1. **The route holds under Ray 2.59.0 and Python 3.13, every step as at R3.**
   - `hardkill-a` died of signal 9 in the QSI stage, with its status left `running`.
   - `list` marked it stale, `!!`, and named `end`.
   - The resume was refused at `begin`, with no run directory created and no store file changed.
   - `end` recorded the run `lost`, verified, and a second `end` was refused.
   - `hardkill-b` ended `done` with its fingerprint and the sidecar `written`.
   - The `hardkill` store is **`39809dca…`**, equal to R3's and R6's recorded fingerprints, and
     passes every read-only check.
2. **The GCS server deadlocked again.** It was alive **about 1 h 56 m** after the `SIGKILL`, until a
   person killed it on the user's ruling. Its main thread was in R3's stack frame for frame, and Ray
   2.59.0 logs the `SIGTERM` that put it there. H2 does not hold, as at R3.

#### 1.10.1 `hardkill-a`

**The launch** (13:29:26) was `reference`'s command on the `hardkill` store, with no `--resume` and
`--register hardkill-a`. Its pre-launch check found the store name `absent` with no runs, no file
of it, and `ray-processes` empty. **The interrupter** (pid 98353, 13:29:27) ran with `resumed-b`'s
point and `--signal KILL`:

```bash
nohup ./venv/bin/python -u docs/datastore-integrity-verification/interrupt_when.py \
    var/runs/py313-venv-01-hardkill-a-20261025T132927 \
    '^CALCULATE QUADRATIC SOURCE INTEGRALS' 0 --interval 5 --hold 117 --signal KILL > /dev/null 2>&1 &
```

**The one check**, then `set-pid 98329`:
- the run is `py313-venv-01-hardkill-a-20261025T132927`, with driver 98329 and `sh` 98325;
- the manifest is as every other;
- the status is `running` with its pid answering;
- `status` showed 17 Ray processes of `/tmp/ray-sgk`.

#### 1.10.2 After the kill

- **The interrupter's log.** It read the QSI banner at 13:33:17, held 117 s, and logged `sent
  SIGKILL to pid 98329 at stage 'CALCULATE QUADRATIC SOURCE INTEGRALS' (0% >= 0%)` at
  **13:35:14**, then exited. The load had fallen, so the build reached QSI 3 m 50 s after its
  launch.
- **The launch log** ends with the shell's own report, which names its `$0` (the purpose), `…:
  line 5: 98329 Killed: 9 …`, then `** driver exit status 137`.
- **`status.json`**:
  - `state` `running`, `pid` 98329, `exit_code` `null`, stage `CALCULATE QUADRATIC SOURCE
    INTEGRALS`;
  - heartbeat `13:33:14`, which was 176.9 s old at 13:36, and `liveness` `stale`;
  - this machine's `host` `id` and `boot`;
  - **no `fingerprint` key**, because no `finish` ran.
- **The run directory** holds `interrupter.log`, `manifest.json`, `status.json`, `stderr.log` and
  `stdout.log`. **There is no `status.json.*.tmp`.** `stderr.log` is SciPy's 4-line
  `IntegrationWarning`.
- **`kill -0 98329`** gives "no such process". The `sh` and the interrupter are gone.
- **No `-journal`, `-wal` or `-shm`** lies beside any file of the store.

#### 1.10.3 Ray after the `SIGKILL` (U3)

**The reading.** The orchestrator's reading at 13:35:29 found one process alive. At 13:35:50
`ray-processes` exited 1 and listed one process:

```
98369 1 /Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.13/site-packages/ray/core/src/ray/gcs/gcs_server --log_dir=/tmp/ray-sgk/session_2026-10-25
```

- This is the GCS server of the dead driver's session, started at 13:29:28 and reparented to
  `launchd`.
- The raylet, the workers, the actors and the broker were gone within about 15 s.
- At 13:36:11 the GCS server was in state `S`, with no children and no listening TCP socket.

**What the GCS server was doing.** Its `gcs_server.out` ends in the second of the kill:

```
[2026-10-25 13:35:14,977 I 98369 …] (gcs_server) gcs_server_main.cc:247: GCS server received SIGTERM, shutting down...
[2026-10-25 13:35:14,980 I 98369 …] (gcs_server) gcs_server.cc:395: Stopping GCS server.
[2026-10-25 13:35:14,980 I 98369 …] (gcs_server) ray_event_recorder_base.cc:69: Stopping RayEventRecorder and flushing remaining events.
[2026-10-25 13:35:14,982 I 98369 …] (gcs_server) gcs_node_manager.cc:178: HandleUnregisterNode() for node … grpc_peer=ipv4:127.0.0.1:61874
[2026-10-25 13:35:14,984 I 98369 …] (gcs_server) gcs_node_manager.cc:656: , death reason = EXPECTED_TERMINATION, death message = received SIGTERM …
```

Nothing follows these lines, and `gcs_server.err` is empty. **Unlike 2.43.0, 2.59.0 logs that the
GCS server received a `SIGTERM`** in the second of the driver's `SIGKILL`. The sender is not
recorded. R3's log showed only `Stopping GCS server.`, after "Not sending reply because executor
stopped".

**The wait and the stop.**
- The orchestrator waited on 98369 by pid in the background, with a deadline 11 minutes from
  13:35:29. R3's rule puts the deadline at the `sent SIGKILL` stamp + 600 s, 13:45:14.
- 98369 was alive at 13:46:44 (ppid 1, etime 17:16, stat `S`), so prompt §2.9's stop fired.
- Nothing was signalled, `ray stop` was not run, and the lock was not released. The orchestrator
  put the stop to the user.

**The user's ruling (2026-10-25): sample it, then the user kills it.**
1. **The sample**, read-only, at 15:28:43: `sample 98369 10 -file
   prompts/py313-venv/measure/out/01-gcs-sample.txt`, exit 0. At that point 98369 had ppid 1,
   etime 01:59:15 and stat `S`. The sample records 30 threads, 8276 samples each, parent `launchd`
   [1], footprint 93.5 MB.
   - **The main thread, in every sample**, ran:

     ```
     main → boost::asio::io_context::run() → scheduler::do_run_one
       → signal_handler<main::$_1, …>::do_complete → ray::gcs::GcsServer::Stop()
       → ray::rpc::GrpcServer::Shutdown() → grpc::Server::ShutdownInternal(gpr_timespec)
       → gpr_cv_wait → _pthread_cond_wait → __psynch_cvwait
     ```

     That is R3's stack frame for frame (R3: `signal_handler<main::$_0, …>`).
   - **The gRPC threads**:
     - `server.poll0` and `server.poll1` wait in `ray::rpc::GrpcServer::PollEventsFromCompletionQueue
       → grpc::CompletionQueue::AsyncNextInternal`, on the server's completion queues that
       `Shutdown` is waiting to drain;
     - five `client.poll` threads wait in `ClientCallManager::PollEventsFromCompletionQueue`;
     - a `SyncRequestThreadManager` worker waits in `PollForWork`;
     - the event engine's pool waits in `PollPoller::Work`;
     - the executor and the rest are idle in `gpr_cv_wait`.
   - So the GCS server's own `SIGTERM` handler is stuck in gRPC's server shutdown, waiting for
     in-flight calls that nothing will complete, because the raylet, the workers and the driver are
     gone. A `SIGTERM` goes through the same handler and cannot end it.
2. **The kill.** The user ran `kill -KILL 98369` in their own terminal pane, by its pid, by 15:31:14.
   The exact second is not recorded. At 15:31:14 the orchestrator found `kill -0 98369` gave "no
   such process" and `ps -p` listed nothing. The agent's own reading agrees, and `ray-processes`
   exited 0 at 15:31:32. The agent signalled nothing, and `ray stop` was never run.

| Process | Gone by | After the kill | R3 |
|---|---|---|---|
| the driver, 98329 | 13:35:14 (the `SIGKILL`) | 0 s | 0 s |
| the raylet, workers, actors and broker | 13:35:29 | ≤ 15 s | ≤ 20 s |
| **the GCS server, 98369** | **by 15:31:14, by the user's `kill -KILL`** | **about 1 h 56 m**; it would not have gone by itself | 20 m 50 s, by `SIGKILL` on the user's ruling |

**H2 does not hold, as at R3.** Sixteen minor releases of Ray have not changed it, and the
lifetime is bounded only by how long it takes a person to rule.

#### 1.10.4 The evidence before `end`

At 15:31:45:
- **`closed_store_state.py`**: no journal, 0 in-flight rows, `GkSource` 97 + 97 + 0 + 97 = 291 and
  **0 unvalidated**, and every other class validated. The files were identical across the read.
- **`list`** exited 0: `!! py313-venv-01-hardkill-a-20261025T132927 running …` and "58 run(s) …:
  55 finished, 1 stale, 2 unknown". Its stderr named `python -m runregistrykit end RUN_ID --reason
  TEXT`.
- **`store show`**: kind `absent` (no sidecar yet), and `py313-venv-01-hardkill-a-20261025T132927
  running stale by results`.
- **The store's five files**, with SHA-256, size and `mtime_ns`, are saved in `01-hardkill.txt`.
  `var/runs/` held 58 directories.

#### 1.10.5 The refused resume

Launched detached at 15:32:09, `--resume --register hardkill-b`, into
`var/runs/py313-venv-01-hardkill-b-refused.launch.log`. The one check was the `sh` (14973) gone,
by `kill -0` within 120 s: it went in 2 s. No `set-pid` was run. **The launch log, whole:**

```
** quadsource_atol_sweep --build: resuming into py313-venv-01-hardkill.sqlite
!! Cannot begin run py313-venv-01-hardkill-b-20261025T153210 with results "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/py313-venv-01-hardkill.sqlite": run py313-venv-01-hardkill-a-20261025T132927 is running (stale, pid 98329) and names this store by its results. A stale run is ended by a person, with `python -m runregistrykit end RUN_ID --reason TEXT`, before another run is begun on its store; the registry does not decide that a run is dead. No run directory was created
** driver exit status 1
```

- It is `begin`'s refusal, in `rrk-v020-adoption` 05's clean form, with no traceback. R3's was a
  `RuntimeError` traceback with the same words.
- Ray was not started: there is no `bootstrapping a local Ray instance` line, and `ray-processes`
  exited 0.
- **No run directory was created**, and `var/runs/` holds 58 directories, as at §1.10.4.
- **Every store file** has §1.10.4's SHA-256, size and `mtime_ns`.

#### 1.10.6 `end`

The manifest's SHA-256 was `35cde564…`, and `kill -0 98329` gave "no such process" at 15:32:36.
Then:

```bash
./venv/bin/python -m runregistrykit end py313-venv-01-hardkill-a-20261025T132927 \
    --reason "py313-venv 01: SIGKILLed by interrupt_when.py --signal KILL at CALCULATE QUADRATIC SOURCE INTEGRALS; pid 98329 checked gone with kill -0 at 2026-10-25 15:32:36 +0000"
```

It exited 0: `>> end: py313-venv-01-hardkill-a-20261025T132927 is now lost, verified: its pid 98329
does not answer kill -0 on this host`.
- `state` is `lost` and `exit_code` is `null`. `ended.verified` is `true` and
  `not_verified_because` is `null`.
- `ended.evidence` is `host` `this`, `boot` `this`, `pid_answers` `false`, `state` `running`, `pid`
  98329, `status_readable` `true`, `heartbeat` `13:33:14` and `heartbeat_age` 7162.4.
- `ended.git_head` is `55c6a547…`, the dispatch `HEAD`, and `ended.by` is `ds283`.
- `host`, `pid`, `stage`, `units_done` and `units_total` are kept with their values. Only `state`
  and `heartbeat` changed, and `ended` was added. The manifest's SHA-256 is unchanged.
- **The second `end`** exited 1: `!! Cannot end …: its status says it has already ended, as 'lost';
  … Nothing was written`. `status.json` is byte-identical across it (`d0ef4c2e…`).
- **`list` again**: the run is `lost`, with no `!!`; "56 finished, 2 unknown".

#### 1.10.7 The sidecar

`store create --purpose "<one line>"` exited 0 (store_id `8bf413f9711c45f0854942d265e12c48`).
`store show` then gave kind `registry`, problems none, and
`py313-venv-01-hardkill-a-20261025T132927 lost finished by results`. The store's five files were
identical across `end` and the sidecar.

#### 1.10.8 `hardkill-b`

**The launch** came at 15:33:04, `--resume --register hardkill-b`: run
`py313-venv-01-hardkill-b-20261025T153304`, driver 15262, `sh` 15257. It **ended `done` at
15:34:49, before the agent's check** (deviation 6). The check, made after its end, found the
manifest as every other, with `--resume` and `results_store_id` `8bf413f9…`.

- **At open**, there was no `INTEGRITY WARNING`, no prune line and no `Rolled back the journal`
  line. `hardkill-a` left no journal and no unvalidated row: it was killed after `GkSource` had
  completed and validated, in the QSI stage, whose rows carry no `validated` column. So the prune
  is not exercised by this pair, as at R3. §1.3.2 exercised it.
- **What it computed**: every stage 0/0/0 up to the policies, then **`QuadSourceIntegral`
  328/328/328**. So `hardkill-a` had stored 864 − 328 = **536** QSI rows when it was killed. That
  figure is inferred.
- **At finish**: `done`/0, with a fingerprint of format 6, `encoding_format` 1, **`39809dca…`**, 21
  classes and problems `{}`. `fingerprint_sidecar` is `written`, with no `fingerprint_error`. The
  launch log ends `** driver exit status 0`.
- Its statistics block is byte-identical to R3's `hardkill-b`, and its `stdout.log` matches R3's in
  line kinds and in the `** ` sequence.
- **After it**, no Ray process was left: `ray-processes` exited 0, and `ps -p` found none of 15257,
  15262, 15362 or 15363. Its Ray shut down through the driver, as every `done` run's does.

#### 1.10.9 The checks, read-only, on the `hardkill` store

At 15:35:5x, after `hardkill-b` and its Ray processes had gone:
- **`compare_with_recorded.py`**, against R3's `reference` and `resumed-c` and R6's `reference` and
  `resumed-c`: 0 differences against the sidecar and against each of the four, verdict `equal`,
  exit 0.
- **`compare_stores.py`**, reference against hardkill: 0 differences, `equal`.
- **`store_soundness.py`**: `integrity_check` `['ok']` and `foreign_key_check` 0 on the primary and
  every shard; 0 problems and 0 `duplicate`; `=> clean`.
- **The census**: `no divergence`, exit 0.
- **`closed_store_state.py`**: no journal, 0 in-flight rows, no unvalidated row.
- **`prune_check.py`** against the states after `hardkill-a` and after `hardkill-b`: `none: there
  is nothing to check`.
- **Every file of the store**, the sidecar included, was identical before and after these checks.

Then the lock was released (§1.1.3).

#### 1.10.10 The issue narrowed

**`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`**, on the `rrk-v020-adoption`
board's §3 and indexed in `docs/OPEN_ISSUES.md` §1.1, is narrowed with this measurement:
- it **is not Ray 2.43.0's alone**: under 2.59.0 the same `SIGKILL` leaves the same GCS server in
  the same stack, frame for frame;
- it lived about 1 h 56 m, until a person's `SIGKILL` on the user's ruling (sample, then kill);
- 2.59.0 logs the `SIGTERM` it received in the second of the driver's death.

What A3 v2 inherits is unchanged: after `end`, a person checks `ray-processes` and ends a listed GCS
server with `kill -KILL <pid>` (`CLAUDE.md` rule 5). The cause and any fix are still Ray's or the
launcher's.

**`[a3-v2-origin-main-moved-to-python-3.13-without-this-branch]`** closes on the `handover` board
(§4): this branch is now on 3.13.16 with `ray` 2.59.0.

## 2. R7 — the Mac Studio

Written by prompt 02.
