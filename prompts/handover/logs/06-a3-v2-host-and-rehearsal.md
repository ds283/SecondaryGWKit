# Log 06 — the A3 v2 host: the Mac Studio's checkout and venv, and rehearsal R4 there

> **Committed on the laptop** by the orchestrator (`CLAUDE.md`; prompt 06 §7). A path
> `var/handover-06/…` below is that directory on the Mac Studio; its `out/`, `roots/`, `scripts/` and
> `state/` are committed at `docs/handover/a3-v2/rehearsal/`, and `verification.md` is
> `docs/handover/A3-V2-VERIFICATION.md` §1.


**Prompt:** prompts/handover/06-a3-v2-host-and-rehearsal.md
**Commit:** none on the Mac Studio (prompt §7). The orchestrator makes the one commit on the laptop.
Executed at dispatch SHA `3152af943a4f8dd5cf5920f7ed6b209f1c7e217b` ("Write the handover orchestrator
note for prompt 06").
**Model:** Claude Opus 5.5, in a Claude Code session on the Mac Studio (`mps026298`).
**Date:** 2026-10-22 (19:42–20:08 BST).
**Result:** COMPLETE WITH DEVIATIONS. **R4 holds**: both stores are `39809dca…`, equal to each other,
to their sidecars and to R3's recorded `reference` and `resumed-c`. One stop-and-ask, on `venv` not
being ignored, was resolved by the orchestrator before any launch.

The record is `verification.md` (§1, "R4 — the Mac Studio"), and this log points into it. The saved
outputs are under `out/`.

## What shipped

Nothing tracked. Every record is under `var/handover-06/`:
- **`out/`**, named with `06`:
  - the host, venv and suites: `host_06.txt`, `old_venv_06.txt`, `pip_install_06.txt`,
    `pip_06.txt`, `roots_06.txt`;
  - the launches: `prelaunch_checks_06.txt`, `lock_06.txt`, `interrupter_dummy_test_06.txt`,
    `interrupter_build_dummy_test_06.txt`, `interrupter_firings_06.txt`, `store_sidecars_06.txt`;
  - the stores: `closed_store_state_06.txt`, `prune_check_06.txt`, `fingerprint_comparison_06.txt`,
    **`fingerprint_against_recorded_06.txt`**, `census_06.txt`, `store_soundness_06.txt`;
  - L4 and the readers: `stage_counts_06.txt`,
    `policy_statistics_06_{reference,resumed-b,resumed-c}.txt`, `log_comparison_06.txt`,
    `readers_06.txt`, `guard_06.txt`;
  - the registry: `run_records_06.txt` and `var_datastores_06.txt`.
- **`verification.md`**, **`issues.md`** and this log.
- **The working files:**
  - `recorded/`, R3's two run directories;
  - `scripts/`, the agent's throwaway helpers (`prelaunch.sh`, `launch.sh`,
    `launch_with_interrupter.sh`, `after_launch.py`, `check_and_setpid.sh`, `after_run.sh`,
    `checks.sh`, `roots.sh`, `log_comparison.py`, `run_records.py`);
  - `state/`, `extract/`, `roots/` and `work/`.
- **Under `var/`, outside `var/handover-06/`:** `var/runs/` (four run directories and four launch
  logs) and `var/datastores/` (two stores, each with four shards and a sidecar). Nothing is retired.
- **Outside `var/`:**
  - the new `venv/` (Python 3.12.15);
  - one appended line `venv` in `.git/info/exclude` (deviation 2);
  - the old venv, moved to `~/Documents/Code/retired-venvs/SecondaryGWKit-py313-20261022`;
  - `~/ScienceData/`, made by `acquire`, empty after `release`.

## The runs

| Slug | Run id | Python pid (`sh`) | Interrupter | State | Exit code | Driver exit status | Wall time (R3) |
|---|---|---|---|---|---|---|---|
| `reference` | `handover-06-reference-20261022T195228` | 78053 (78047) | — | `done` | 0 | 0 | 3 m 41 s (3 m 52 s) |
| `resumed-a` | `handover-06-resumed-a-20261022T195718` | 78548 (78545) | 78556, `--hold 10` | `killed` | 143 | 143 | 1 m 47 s (2 m 18 s) |
| `resumed-b` | `handover-06-resumed-b-20261022T200009` | 78952 (78950) | 78964, `--hold 30` | `killed` | 143 | 143 | 1 m 13 s (1 m 19 s) |
| `resumed-c` | `handover-06-resumed-c-20261022T200159` | 79208 (79203) | — | `done` | 0 | 0 | 1 m 27 s (1 m 46 s) |

- **Fingerprints.** `reference` and `resumed-c` are
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`; `resumed-a` is `7c165c78…`
  and `resumed-b` `82318bfa…`. All are format 6, `encoding_format` 1, 21 classes, problems `{}`.
- **Every manifest** has `git_head` the dispatch SHA, `git_dirty` false, `packages` equal as JSON
  to the pinned record (`datastorekit` 0.5.0 `f47b9a2…`), `script`
  `RunDrivers/quadsource_atol_sweep.py` with `script_sha256` `b2fa4ef1…`, twenty keys (R3's), and
  `host.id` `d120d1cf03551e6f`. Every status's `host` is `this_host()`'s `id` and `boot`.
- **The interruptions.** The points come from R4's own `reference`: `GkSource` 22.6 s, QSI 88.7 s,
  so the holds are 10 s and 30 s, R3's (verification §1.1.2).
  - `resumed-a` left 164 `GkSource` rows, 43 unvalidated.
  - `resumed-b` pruned the 43, computed 170, and stored 286 QSI rows.
  - `resumed-c` computed 578 QSI rows.
- **The stage times beside R3's** are in verification §1.5.5. The Mac Studio is 2–13 % faster per
  stage at `--cpus 6`.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — R3's recorded run directories were staged outside the checkout.**
   - **Assumed:** prompt §2 says the orchestrator copies them into `var/handover-06/recorded/` before
     dispatch.
   - **Found:** `main`'s `.gitignore` does not ignore `var/`, so a copy there would have dirtied
     `git status` before §1.2's check.
   - **Done:** they were staged at `~/handover-06-recorded/`. After `git switch handover-remedial`,
     whose `.gitignore` ignores `var/`, they were moved into `var/handover-06/recorded/`.
     `git status --short` stayed empty. The SHA-256 of each `manifest.json` and `status.json`
     matches the laptop's: `13e8fcaa…`, `a64cd267…`, `a0742b57…` and `3d30a3cf…` (`host_06.txt`).
     `~/handover-06-recorded/` is left in place, empty.
2. **STRUCTURALLY REQUIRED — `venv` appended to `.git/info/exclude`.**
   - **Assumed:** §1.4 says to check that `venv` is ignored and `git status --short` is still empty.
   - **Found:** it was not ignored. `git check-ignore -v venv` exited 1 and the status showed
     `?? venv/`. The Python 3.13 venv carried its own `venv/.gitignore` (`*`); a 3.12 `venv` does
     not write one, and nothing tracked names `venv`. `runregistrykit` stamps
     `git_dirty = bool(git status --porcelain)`, so every manifest would have recorded
     `git_dirty: true`.
   - **Done:** the agent stopped before any launch and asked the orchestrator. On its ruling, the
     line `venv` was appended to the checkout's `.git/info/exclude` (line 28), as on the laptop
     (its line 40). That file is written outside `var/`. The check afterwards:
     `git check-ignore -v venv` → `.git/info/exclude:28:venv`, and `git status --short` empty.
     The original file is kept at `state/git_info_exclude_before.txt`. Opened as
     `[06-nothing-tracked-ignores-venv]`.
3. **STRUCTURALLY REQUIRED — `var/runs/` and `var/datastores/` were created.** This checkout had no
   `var/`, and the launcher refuses a `--database` whose directory is not one
   (`RunDrivers/quadsource_atol_sweep.py`, `run_build`). Both were made empty with `mkdir -p`
   before the first launch.
4. **STRUCTURALLY REQUIRED — `ObjectFactories` is 280, not the prompt's 279.**
   - **Assumed:** §1.6 quotes 279 as `01-roots.txt`'s count.
   - **Found:** that is the file's BEFORE section, on `datastorekit` 0.4.0. Its AFTER section (0.5.0)
     and the `dk-v050-adoption` board ("ObjectFactories at `HEAD`: 280") give 280.
   - **Done:** the run gave 280, OK. Not a stop: the count equals `01-roots.txt`'s count for this
     tree.
5. **IMPLEMENTATION CHOICE — `RunDrivers` `OK (skipped=1)`, accepted.** 115 ran, the laptop's count,
   with one skip the laptop does not have. The skip comes from a test that reads a gitignored
   `var/` file absent from a fresh `var/` (`test_pipeline_adoption.py:101-102`). No test failed, so
   it is not §5's "a test fails"; it is recorded.
6. **IMPLEMENTATION CHOICE — the comparison with R3 is against the two recorded directories, and
   against R3's saved outputs for the other two slugs.** Only R3's `reference` and `resumed-c` run
   directories were copied over. R3's `resumed-a`, `resumed-b` and launch logs are not on this
   machine. So:
   - the line-kind comparison (verification §1.2.5) is full for `reference` and `resumed-c`;
   - for the other two it is by `log_comparison_03.txt`'s recorded counts (section 1) and `** `
     counts (section 3).

   The alternative was to ask for the other two to be copied; the counts were enough to show that
   nothing new appears.
7. **IMPLEMENTATION CHOICE — working directories inside `var/handover-06/`**: the dummy tests ran
   into `var/handover-06/work/`, and R3's "scratchpad" files went to `state/` and `scripts/`.
   Prompt §0.3 says everything produced goes under `var/handover-06/`.
8. **UNINTENDED DRIFT — `reference`'s pid was never recorded in the lock.**
   - **What happened:** the agent's first two `set-pid` calls (19:52:34 for `reference`, 19:57:26
     for `resumed-a`) passed `set-pid <pid>` as one argument. zsh does not word-split an unquoted
     `$c`. argparse rejected the call with exit 2, and the agent misread the following `status`'s
     exit 0 as the `set-pid`'s.
   - **Effect:** `reference` ran its whole 3 m 41 s under the held lock with `pid: -`.
   - **Repair:** the slip was found at `resumed-a`'s check and `set-pid 78548` was re-run correctly
     at 19:57:38. `resumed-b` and `resumed-c` were set correctly through a bash helper. It was not
     back-filled for `reference`, which had ended. The note is in `lock_06.txt`.
   - **Consequence:** none to the build or its record. The lock was held, so no other project could
     acquire it. Only its owner file lacked the pid for those four minutes.
9. **UNINTENDED DRIFT — the resumed store's before/after hash at `store create` was lost.**
   - **What happened:** the step's comparison used `[ "$a" == "$b" ]` in zsh, which reads `==` as an
     `=command` expansion (`(eval):1: = not found`). The command stopped after `store create`
     (rc 0) and before the comparison or the `resumed-b` launch that followed it.
   - **Done instead:** every store file's mtime was shown to be earlier than the sidecar's
     `created` (19:59:45), so `store create` wrote none of them. That is recorded in
     `store_sidecars_06.txt`. Later steps used `bash`.
10. **UNINTENDED DRIFT — two more shell slips, each re-run before anything depended on it:**
    - the first Ray-pid extraction after `reference` used the wrong pattern, and was re-run with
      `pid=` (as R3's first was);
    - the first `stage_counts.py` call passed its four directories as one argument (zsh again), and
      was re-run through `bash`.

    Both are visible in the saved outputs.

## Verification performed

All of this was run here, and the outputs are saved (verification §1 has each in full):
- **§1.1–§1.2.** The host was recorded. The checkout went from `main` `9678f28` clean to
  `handover-remedial` `3152af94…` clean.
- **§1.3–§1.5, the venv.**
  - The old venv (3.13.16, `ray` 2.45.0) was recorded and moved.
  - The new venv is Python 3.12.15, with `pip install -r requirements.txt` rc 0 and pip not
    upgraded.
  - **`pip freeze` is identical to the laptop's** second freeze in `01-pip.txt`: 101 lines, `diff`
    rc 0. `pip check` is clean.
  - `package_provenance` gives the three pins, none editable.
- **§1.6, the roots**, at the dispatch SHA: AdaptiveLevin 32 OK, ComputeTargets 700 OK,
  CosmologyModels 39 OK, LiouvilleGreen 148 OK (skipped=1), ObjectFactories 280 OK, RunDrivers 115
  OK (skipped=1), tools 91 OK. The flake did not fire.
- **§1.7.** `list` showed no runs. `this_host()` gave `d120d1cf03551e6f` / `C58A0F92-…`. The lock
  was free, `ray-processes` listed nothing, and `RAY_ADDRESS` was unset.
- **The launcher's SHA-256** is `b2fa4ef1…`, unchanged since `031bcb4`; no commit in
  `c081eed..HEAD` touches it.
- **L1.** States, exits and fingerprints are as recorded. No temporary status file, no Ray process
  left after any run, and every manifest and status is as §3 says.
- **L2.** The prune removed exactly 43 and reused none. The chain's identities hold. The two stores
  are equal (`compare_stores.py`, verdict `equal`).
- **L3.**
  - **`compare_with_recorded.py`, both stores against R3's two recorded runs: every comparison
    empty, verdict `equal`, rc 0.**
  - The census shows `no divergence` and `store_soundness` is `clean`.
  - Every reader exits 0, with R3's lookups, and the store is byte-identical before and after each.
    `store fingerprint` afterwards matches `resumed-c`'s.
- **The guard** refuses cleanly (`!!`, exit 1) and changes and creates nothing.
- **L4.** Statistics counts are R3's, and sorted lines are identical. Stage counts, identities, the
  `** ` counts and the verbatim banners equal R3's. The status fields are R3's. Every log difference
  is explained.
- **The lock** was acquired at 19:52:00 and released at 20:07:26 (rc 0). Afterwards `status` is free
  and `ray-processes` lists nothing.
- **The tree.** `git status --short` was empty before the first launch and after the last step, and
  `HEAD` is still the dispatch SHA.
- **The registry.** `list` shows 4 runs, "4 finished; 0 with no host recorded", nothing running or
  stale.

## Observations not acted on

1. **Nothing tracked ignores `venv/`** (deviation 2). Opened as `[06-nothing-tracked-ignores-venv]`
   (`issues.md`).
2. **`aiohttp` 3.11.14 and `aiohttp-cors` 0.8.0 are yanked on PyPI.** pip warns and installs them.
   The pins are the laptop's, and the freeze is identical. Not opened: a pin move with its own
   rehearsal, not this prompt's.
3. **`RunDrivers`' adoption test skips on a fresh `var/`** (deviation 5). One test loses its
   evidence on any machine without the hand-written A3 manifest. By design; not opened.
4. **Prompt §1.6's 279 is the stale count** (deviation 4). For the orchestrator's review of the
   prompt; not opened.
5. **The log differences from R3** (`SystemExit(143)` in `resumed-b`; the `IntegrationWarning` moved
   from `resumed-c` to `resumed-b`; the policy block's line order) all follow from where the signal
   fell, or from `[02-the-policy-statistics-block-orders-its-lines-by-completion]`. Not findings.
6. **The guard's refusal no longer prints a traceback**, as `rrk-v020-adoption` 05 intended. The
   message text is R3's.
7. **Host load.** `corespotlightd` used ~140 % CPU and `CategoriesService` ~67 % at §1.1, and the
   load was 33 during the suites (the tests' own workers). During R4 the one-minute load was 4–6. It
   bears on 07's 40-hour estimate only as background.

## State handed to the next prompt

For 07, on this checkout:
- **`HEAD`** is `3152af943a4f8dd5cf5920f7ed6b209f1c7e217b` on `handover-remedial`, with
  `git status --short` empty. 07's checkout fast-forwards to the orchestrator's commit of these
  records. That commit changes only `docs/` and `prompts/`, so the launcher's hash stays
  `b2fa4ef1…`.
- **The venv** is `venv/`, Python 3.12.15 (MacPorts `/opt/local/bin/python3.12`), with a freeze
  identical to the laptop's. It is ignored by `.git/info/exclude:28` only, so keep that line if the
  venv is rebuilt here.
- **The host.** `this_host()` is `{"boot": "C58A0F92-923F-471F-B32E-B1B9F5D0A848", "id":
  "d120d1cf03551e6f", "machine": "arm64", "name": "mps026298.phys.susx.ac.uk", "python": "3.12.15",
  "release": "27.0.0", "system": "Darwin"}`, booted 2026-10-07 10:23:52. A different `boot` at 07
  means the machine rebooted.
- **The registry and the lock.** `var/runs/` holds R4's four runs, all finished. The lock is free;
  `~/ScienceData/` exists and is empty.
- **Disk.** 121 GB free (87 %), under Ray's 95 % threshold.
- **Stage times at `--cpus 6`, rehearsal scope:**
  - `reference` 3 m 41 s (R3 3 m 52 s);
  - `GkSource` 22.6 s and QSI 88.7 s (R3 24.8 s and 91.4 s);
  - Tk/Gk numerics 22.9 s and Gk WKB 1 m 4.75 s.

  07 scales A3 v2's wall-clock estimate from these.
- **Stores** `var/datastores/handover-06-{reference,resumed}.sqlite` (`store_id` `d3ce4fae…`,
  `1c41fb13…`) are the user's to retire. A3 v2's store name must be new (`store show` → `absent`).
- **The launch helper's pattern** (`var/handover-06/scripts/launch.sh`) is R3's command with the
  launcher at `RunDrivers/`. Use bash, not zsh, for any helper that word-splits or uses `==`.
