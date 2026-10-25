# Log — py313-venv prompt 01, move the laptop's venv to Python 3.13

- **Subject:** [`01-move-the-laptop-venv-to-python-3.13.md`](../01-move-the-laptop-venv-to-python-3.13.md),
  board items B1, P1, V1, G1, C1, S1, R6 and K1, and gate FP6.
- **Commit:** the commit that adds this log, whose parent is `55c6a54`, the dispatch `HEAD`.
- **Result:** landed. Acceptance 1–6 hold.
  - The stop of §2.9, a stop condition of §4, fired: the GCS server was alive past ten minutes. It
    was taken as written, and the user ruled.
  - In ComputeTargets, README §7's known wall-clock test failed under both interpreters, under
    another project's load. It is recorded below.
  - Every other test of the seven roots passes at 3.12's counts.
- **Model:** Opus 5.5.
- **Date:** 2026-10-25, 12:12–15:37 GMT, on the laptop (`mps026088`), in the main checkout, on
  `handover-remedial`.

## What shipped

- **The pins.** `requirements.txt:75` `ray==2.43.0` → `ray==2.59.0`, and `:88` `SQLAlchemy==2.0.39`
  → `SQLAlchemy==2.0.55`. No other line moved.
- **The venv**, which is not in the commit.
  - The 3.12.15 venv was moved to `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025`.
  - `/opt/local/bin/python3.13 -m venv venv` then `./venv/bin/pip install -r requirements.txt`
    built the new one, with the pip that the venv brought (26.2.1), not upgraded. It is Python
    3.13.16.
  - Its freeze is 101 lines and differs from 3.12's in exactly the `ray` and `SQLAlchemy` lines.
  - The three git packages are at their commits and not editable, and `pip check` is clean.
  - SQLite is 3.53.4 under both interpreters.
- **`runregistrykit.toml`.** `distributions` is now `["datastorekit", "raypoolkit", "ray",
  "SQLAlchemy", "numpy", "scipy"]`, under a one-line comment saying the last four are there so that
  a manifest says which venv wrote a store.
- **`CLAUDE.md`.** One bullet opens "Repository mechanics":
  - the venv is Python 3.13, MacPorts' 3.13.16, on both hosts;
  - it is built with `python3.13 -m venv venv`, then `pip install -r requirements.txt`, never by
    hand;
  - it moved from 3.12.15 on 2026-10-25 in `py313-venv` (the laptop in 01, the Mac Studio in 02).

  The packages section's `ray` and `SQLAlchemy` sentence is unchanged.
- **`docs/py313-venv-verification.md`** (new): the header, §1 "R6 — the laptop on Python 3.13" with
  R3's hard-kill pair as its §1.10, and a one-line §2 for prompt 02.
- **`measure/out/01-*`:**
  - `pip`, `roots`, `black`;
  - `prelaunch_checks`, `lock`, `interrupter_dummy_test`, `interrupter_build_dummy_test`,
    `interrupter_firings`;
  - `store_sidecars`, `closed_store_state`, `prune_check`, `fingerprint_comparison`,
    `fingerprint_against_recorded`, `census`, `store_soundness`;
  - `readers`, `guard`, `stage_counts`, `policy_statistics_{reference,resumed-b,resumed-c,hardkill-a,hardkill-b}`,
    `log_comparison`, `run_records`, `var_datastores`;
  - `hardkill`, `gcs-sample`.
- **The boards.**
  - This campaign's board: the header, the §1 row, the eight items, FP6, §3 (two issues opened), §4
    (the closed issue) and §5.1.
  - The `handover` board: the issue moved from §3 to §4, its header count, and its pointer to the
    GCS issue.
  - The `rrk-v020-adoption` board: the GCS issue narrowed.
  - `docs/OPEN_ISSUES.md`: one row deleted, two added, one narrowed; 103 → **104**, and twenty-two
    → twenty-three campaigns.

## Deviations from the prompt

1. **The waits were the orchestrator's, not the agent's** (§2.8's "waits" bullet). After each
   launch the agent made its one check, ended its turn, and reported the driver's pid, the `sh`
   pid, the run id, any interrupter pid and the lock's owner file. The orchestrator waited on the
   pid in the background and resumed the agent with the final `status.json`. For the `SIGKILL`ed
   launch it also waited on the GCS server's pid. The agent never waited on a job, never
   sleep-polled and never ran `pgrep -f`. — **STRUCTURALLY REQUIRED**: a subagent's turn is not
   resumed by its own background command (the orchestrator's brief).
2. **R3's machine-wide `pgrep -lf 'gcs_server|raylet|ray::'`** was not run in the pre-launch checks.
   The lock's `ray-processes --ray-temp-dir /tmp/ray-sgk` and `ps -p` on the previous run's pids
   were the test. — **STRUCTURALLY REQUIRED** by the orchestrator's rule (never `pgrep -f`).
3. **Every lock command was a shell command of its own**, the only process it started. Output went
   by `>> … 2>&1`, the exit was recorded with the builtin `echo` and the header with zsh's builtin
   `print -P`. R3 ran `set-pid` and `status` inside its check script. — **STRUCTURALLY REQUIRED** by
   `CLAUDE.md`'s lock section and the brief.
4. **The first `compare_with_recorded.py` call ran in zsh**, which does not word-split an unquoted
   `$A`. It raised `FileNotFoundError` on the two run directories joined as one path. It was re-run
   through `bash` at once, exit 0, `equal`, and its file overwritten. — **UNINTENDED DRIFT**, the
   agent's. It wrote nothing anywhere else and read nothing twice that mattered. Every later command
   with variable arguments ran through `bash`, as §2 says.
5. **§2.9's ten-minute stop fired and was taken as written.** The GCS server 98369 was alive at
   13:46:44, 11.5 min after the `SIGKILL`. Nothing was signalled, `ray stop` was not run, and the
   lock stayed held. The orchestrator put it to the user, who ruled: sample it, then the user kills
   it.
   - The agent ran `sample 98369 10 -file …/01-gcs-sample.txt` (read-only).
   - The user ran `kill -KILL 98369` in their own pane, by 15:31:14.
   - The prompt then went on with R3 §2.4 onwards.
   - — **STRUCTURALLY REQUIRED**: the prompt's own stop, resumed by the user's ruling.
6. **`set-pid` was not run for `hardkill-b`.** The agent's `wait_started.sh` looked for the run
   directory with the glob `var/runs/py313-venv-01-hardkill-b-*`. That also matched the refused
   launch's log file, `…-hardkill-b-refused.launch.log`, which sorts after the run directory. So
   the script took the log for the run directory, sat out its 120 s bound and printed no pid. By
   then `hardkill-b`, which only had 328 QSI rows to compute, had ended `done` (15:33:04–15:34:49).
   - **What it cost:** for those 105 s the lock was held, but its owner pid named `hardkill-a`'s
     dead driver, 98329, not `hardkill-b`'s live one, 15262. A `status` or `release` in that window
     would have read the owner as dead while a build ran.
   - It was **not back-filled**, as R4's slip was not.
   - The one check was made after the run's end. It finds the manifest as every other run's.
   - The glob is fixed in the helper scripts.
   - — **UNINTENDED DRIFT**, the agent's.
7. **`CLAUDE.md`'s bullet says which prompt moves which host**: "(the laptop in its prompt 01, the
   Mac Studio in 02)". The prompt asked for "on the laptop and on the Mac Studio … moved from 3.12 on
   <date> in the `py313-venv` campaign". Until 02 lands, the Mac Studio is still 3.12, and the
   parenthesis keeps the sentence true. — **IMPLEMENTATION CHOICE**.
8. **The verification record puts the hard-kill pair at §1.10**, inside §1, not as a §2 of its own
   as in R3's document. §2.11 leaves §2 to prompt 02 and asks for §1 "in the shape of R3's §1 and
   §2". — **IMPLEMENTATION CHOICE**.
9. **Measurements beyond the prompt's:**
   - `pip check` and the SQLite version in §2.1's block, so that §2.4's comparison had a baseline;
   - the ObjectFactories root re-run under `PYTHONTRACEMALLOC=40`, to trace the new
     `ResourceWarning`. It was stopped with `kill -TERM` on its own unittest pid once the trace was
     printed, and is not a Ray process;
   - each root's warnings by kind, 3.12 against 3.13;
   - the `hardkill` store also compared against R6's own recorded `reference` and `resumed-c`;
   - A3 v2's guard after the readers, as R3 and R4 ran it.

   — **IMPLEMENTATION CHOICE**.
10. **Two issues opened** (board §3) for new lines that README §5 rule 5 makes findings: Ray 2.59.0's
    token authentication, and 3.13's `ResourceWarning`. Neither bears on a stored value. Both are
    opened rather than only recorded, under `CLAUDE.md` rule 4. — **IMPLEMENTATION CHOICE**.

## Verification performed

**The venv (acceptance 1).**
- Python 3.13.16. The freezes, 101 lines each, differ in exactly `ray` and `SQLAlchemy`.
- `datastorekit` 0.5.0 `f47b9a25…`, `raypoolkit` 0.3.0 `67a75b97…` and `runregistrykit` 0.2.0
  `0bae3a3c…`, none editable. `pip check` is clean.
- The old venv was moved, and its path is recorded in `01-pip.txt`.

**The suite (acceptance 2).**

| Root | 3.12 | 3.13 |
|---|---|---|
| AdaptiveLevin | 32 OK, 1 s | 32 OK, 4 s |
| CosmologyModels | 39 OK, 7 s | 39 OK, 20 s |
| LiouvilleGreen | 148 OK (skipped=1), 59 s | 148 OK (skipped=1), 110 s |
| ObjectFactories | 280 OK, 35 s | 280 OK, 61 s |
| RunDrivers | 120 OK, 16 s | 120 OK, 27 s |
| tools | 49 OK, 31 s | 49 OK, 46 s |
| ComputeTargets | 737, 1 failure, 272 s | 737, 1 failure, 330 s; 737 OK in the `-W default` run |

- **The one failure** is `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, in both. Re-run
  alone three times under each interpreter, it failed each time on its cold limit (0.06 s): 3.12
  gave 0.114–0.148 s and 3.13 gave 0.078–0.084 s.
- The load was a ChamPBH agent's suite (`…/ChamPBH/…/agent04/suite.sh`) at ~99 % CPU, with
  Spotlight, at a one-minute load of 8–54. Since it fails under both and 3.13 is faster, this is not
  a finding against 3.13.
- `test_ray_address_default` alone under 3.13: 7 OK.
- `black --check --extend-exclude '^/prompts/' .`: 369 unchanged, under 3.12, 3.13 and before the
  commit.

**R6's four launches (acceptance 3).**
- `reference` `done`/0; `resumed-a` `killed`/143 in `GkSource` (hold 33 s); `resumed-b`
  `killed`/143 in QSI (hold 117 s); `resumed-c` `done`/0.
- The points were chosen from R6's own `reference`, by R3's reasoning.
- **Both stores are `39809dca…`**, 21 classes, `encoding_format` 1, each equal to its sidecar.
  `compare_fingerprints` between them is empty, and each has 0 differences against R3's recorded
  `reference` and `resumed-c`.
- **The lock listed the running 2.59.0 instance by its temp dir**: `ray-processes` exited 1 with 22
  lines, `gcs_server` and `raylet` among them.
- The chain: `resumed-a` stored 180 `GkSource` rows, 42 unvalidated. `resumed-b` pruned exactly
  those 42 (by serial: 42 absent, 0 reused) and computed 153 = 291 − 180 + 42. It then stored 541
  QSI rows, and `resumed-c` computed 323 = 864 − 541.
- No Ray process outlived either `SIGTERM`ed driver: all were gone within 30 and 32 s of the
  signals, where R3's went within 56 s and 29 s of the final heartbeat.
- Every manifest has the seven-distribution `packages`, `host.python` 3.13.16, the dispatch
  `git_head` and `git_dirty` true. No `.py` file was dirty at any launch.

**The hard-kill pair (acceptance 4).** R3's route, every step:
- `hardkill-a` `SIGKILL`ed at 13:35:14 in QSI, status left `running`, `list` `!!` stale;
- the resume refused at `begin`, cleanly, with no traceback, no run directory and every store file
  byte-identical;
- `end` exit 0, `lost`, verified, `ended.git_head` the dispatch `HEAD`, and a second `end` refused
  with `status.json` byte-identical;
- the sidecar made;
- `hardkill-b` `done`, with fingerprint `39809dca…` and `fingerprint_sidecar` `written`;
- the `hardkill` store equal to R3's and R6's recorded fingerprints, and clean on soundness, the
  census, closed state and prune check.

**Ray's last process was not gone within ten minutes, so §2.9's stop was taken.** The GCS server
lived about 1 h 56 m, until the user's kill.

**The readers (acceptance 5).**
- All seven exit 0 with R3's lookups (116, 348, 10, 3, 6, 24; 116 and the `QCD_Cosmology` skip).
- Every store file is identical before and after, and no Ray process is left after any of them.
- The store's fingerprint afterwards matches `resumed-c`'s.
- `release` exit 0 at 15:37:08. Afterwards `status` says `lock: free` and `ray-processes` lists
  nothing.

**Acceptance 6**: `git diff --cached --name-only` was shown before the commit (below).

**The registry after R6:** 59 runs, "57 finished, 2 unknown; 45 with no host recorded", nothing
`running` or stale. That is 53 before and R6's six.

**`var/datastores/`:** 26 files before, 44 after. The 26 are unchanged, and the 18 new files are
R6's three stores and sidecars. There is no journal.

## What is new in the logs

Compared with R3's same slug, whose raw logs are all on this machine (`01-log_comparison.txt`):
1. **Ray 2.59.0's token authentication**, the one line the upgrade added. Every launch log and three
   reader logs carry `WARNING authentication_token_setup.py:… -- Token authentication is enabled
   for this Ray cluster. Set RAY_AUTH_MODE=disabled to opt out.` The first start
   (`reference`) also logged `Generated new authentication token and saved to
   /Users/ds283/.ray/auth_token`, a 64-byte, mode-644 file outside the checkout. Opened as
   `[01-ray-2.59-enables-token-auth-and-writes-a-token-into-home]`.
2. **The GCS server's log names the `SIGTERM`** (`gcs_server_main.cc:247: GCS server received
   SIGTERM, shutting down...`) in the second of the driver's `SIGKILL`. 2.43.0's did not.
3. **`reference` printed two progress lines and two `inflight` lines**, because two stages ran past
   the 300 s notify interval under the load. R3's did not.
4. **`IntegrationWarning`** moved from `resumed-c` and `hardkill-b` to `resumed-b` and `hardkill-a`.
   It follows which run computed the one QSI that raises it, as at R4.
5. **`SystemExit(143)`** in `resumed-a`, where R3 had `SystemExit(1)`. It depends on where the signal
   falls.
6. **The process list** under 2.59.0 shows `ray::DashboardAgent` and `ray::RuntimeEnvAgent` among
   the instance's processes. These are in the lock's listings, not in any log.
7. **Under 3.13 the suites print one new `ResourceWarning`**, in ObjectFactories. Opened as
   `[01-a-test-leaves-an-in-memory-sqlite-engine-undisposed]`.

What is absent:
- no line names `datastorekit`, and `runregistrykit` appears only in the banner;
- no `DeprecationWarning` or `FutureWarning` comes from the interpreter, Ray or SQLAlchemy;
- there is no traceback and no refusal;
- there is no raylet disk-space warning (at 95 % used by `df`).

## Observations not acted on

- **The machine's load.** R6's `reference` took 16 m 33 s against R3's 3 m 52 s. The cause was a
  ChamPBH agent's suite running alongside, not the venv: `hardkill-b`, at a lower load, ran within
  10 % of R3's times. The interruption holds (33 s, 117 s) are longer than R3's for the same reason.
  A rehearsal under the lock is not protected from another project's test suite, which needs no
  lock.
- **The orchestrator's `origin/handover-remedial` was two commits behind** at dispatch (`1fd1c36`
  and `55c6a54` were unpushed). The push after this commit carries them.
- **`~/.ray/auth_token`** is left in place. It is Ray's, and the issue above records it.
- **The GCS server's `SIGTERM`**: who sends it at the driver's death is not established. Its
  `log_dir` and the sample are under `/tmp/ray-sgk/session_2026-10-25_13-29-28_397917_98329/` and
  in `01-gcs-sample.txt`.

## State handed on

- **To 02 (README §4.2):**
  - the new `requirements.txt` and `runregistrykit.toml`, pushed;
  - the laptop's freeze, the last block of `measure/out/01-pip.txt`, 101 lines;
  - the roots' counts: 32, 39, 148 (skipped=1), 280, 120, 49 and 737;
  - R6's fingerprints, all three `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`.
- **The venv** is 3.13.16. The 3.12 venv is at
  `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025`, and retiring it is the user's.
- **Under `var/`:**
  - R6's six run directories and seven launch logs;
  - the stores `py313-venv-01-{reference,resumed,hardkill}.sqlite` with their sidecars (store_ids
    `7fc02cda…`, `23b16cdb…`, `8bf413f9…`), none retired;
  - `var/py313-venv-01/` (scripts, state, roots, extract).
- **The lock** is free. `ray-processes` lists nothing. Nothing is `running` or stale.
