# Log — py313-venv prompt 02 (Mac Studio), STOPPED at step 8

- **Result:** stopped and asked at R7's `reference`, which failed at Ray start-up
  (`issues.md`). The lock is held. Owner file: pid 93203, dead. No Ray process is left.
- **Dispatch:** `aac52bc`. Model Opus 5.5. 2026-10-25, 16:18–16:34 GMT, on `mps026298`.

## What was done (steps 1–7)
- **1** Fast-forwarded `57f67a9`→`aac52bc`, clean. The Gate's diff lists exactly the 3 allowed files
  (`out/host_02.txt`, `out/gate_02.txt`).
- **2** 3.12.15 baselines (`out/pip_02.txt`): the freeze equals the laptop's 3.12 freeze; `pip check`
  is clean; SQLite 3.53.4.
  - Roots (`out/roots_02.txt`): 32, 737, 39, 148 (s=1), 280, 124 (s=1), 49, all OK.
- **3** Venv moved to `~/Documents/Code/retired-venvs/SecondaryGWKit-py312-20261025` (817 MB).
- **4** `python3.13 -m venv` + `pip install -r requirements.txt`, pip 26.2.1 not upgraded.
  - 3.13.16. The freeze is identical to the laptop's last freeze.
  - `pip check` is clean, and the provenance equals the laptop's as JSON. ray 2.59.0, SQLAlchemy
    2.0.55, SQLite 3.53.4.
  - Token: generated with stdout to /dev/null, `chmod 600`, 64 bytes (`out/token_02.txt`).
- **5** 3.13 roots: same counts, all OK. `test_ray_address_default` 7 OK.
  - Warnings by kind equal to 3.12's. ObjectFactories `ResourceWarning` 0, where the laptop's 3.13 had 1.
  - Load reached 24 (Time Machine, Spotlight).
- **6** A3 v2 `store show`: registry, problems none, `0e32f24a…`. Files identical to handover 10's
  after-state (`state/store_files_before.txt`, `out/store_02.txt`).
- **7** `acquire` exit 0, 16:31:52 (`out/lock_02.txt`).
- **8** Dummy tests equal R4's, normalised. Environment checks as expected
  (`out/prelaunch_checks_02.txt`). `reference` launched 16:32:01 (sh 93198, driver 93203).
  `set-pid` 93203. `ray-processes` exit 1 listed `gcs_server` 93233. The second listing, 18 s later,
  was empty: the run had failed.
  - Token checks: no `Token authentication is enabled` line and no `Generated new` line in any of
    the run's logs. The driver's `ps -E` was not made (driver dead). The token is still 600/64 B.

## Deviations
1. The waits are the orchestrator's: one check per launch, then the turn ends. —
   STRUCTURALLY REQUIRED.
2. Every lock command ran as a shell command of its own. The header was printed with the `print -P`
   builtin and records the temp dir as `<the temp dir>`. — STRUCTURALLY REQUIRED.
3. The dummy tests' work dir is under `var/py313-venv-02/work/`, as R4's was. — IMPLEMENTATION CHOICE.
4. Extra measurements:
   - the `ps -E` probe, and the auth-mode import-order probe (neither starts Ray, neither prints
     the token);
   - 3.12 freeze against the laptop's 3.12 freeze;
   - warning counts per root.

   — IMPLEMENTATION CHOICE.
5. Two record slips, both corrected in place: two `pip_02.txt` diff commands broke on a parenthesised
   comment and were re-run; the `ps -E` probe's first run read the wrong field and was re-run. —
   UNINTENDED DRIFT.
6. A second `ray-processes` call during `reference`'s check, because the first came before the
   raylet was up. — IMPLEMENTATION CHOICE.

## Not done
Steps 8 (the rest), 9, 10 and 11, and `verification.md`. The lock is held, awaiting the orchestrator.
