# Log 11 — A3 v2's seam record: the policy summary and the clamp-gap census

> **Written on the Mac Studio, committed on the laptop** by the orchestrator (prompt 11 §2). A path
> `var/handover-11/…` below is that directory on the Mac Studio; its `out/`, `scripts/` and `state/`,
> with `issues.md`, are committed at `docs/handover/a3-v2/seam/`, and `verification.md` is
> `docs/handover/A3-V2-VERIFICATION.md` §5.

**Prompt:** prompts/handover/11-a3-v2-seam-record.md (read at the dispatch SHA `757ef01`)
**Commit:** none on the Mac Studio (prompt header; dispatch rules). The checkout was **not** moved:
it was at `757ef016fc37be996c342d162c05f17b3c430e4d` at the start and at the end (deviation 1).
**Model:** Claude Sonnet 5.5, in a Claude Code session on the Mac Studio (`mps026298`), as a dispatched
sub-agent.
**Date:** 2026-10-25 (18:12–18:16 UTC).
**Result:** COMPLETE WITH DEVIATIONS. No §3 stop was hit, and acceptance 1–5 hold. The deviations are
in how step 1 was done and in how things were recorded, not in what was measured.

The record is `verification.md`, §5 "The seam record".

## What shipped

Nothing tracked, and no commit. Under `var/handover-11/`:
- `out/`:
  - `gate_11.txt`, the Gate;
  - `step1_checkout.txt`, the step-1 check;
  - `freeze_vs_10_after.txt`, the diff of this session's `pip freeze` against 10's;
  - `policy_11.txt`, step 3 (the block verbatim with its line numbers, the table count, the
    cell-by-cell comparison, the non-`complete` row);
  - `seam_census_11.txt`, step 4's census, (a) to (d);
  - `subintervals_11.csv`, 19,289 records (5,050,755 bytes, under the 10 MB limit);
  - `skipped_check_11.txt`, an extra read (deviation 5);
  - `store_after_11.txt`, step 6: the compare, the fingerprint, the registry and `HEAD`.
- `state/`: `store_files_before.txt`, `store_files_after.txt`, `freeze_11.txt`.
- `scripts/`: `rec.sh` (records a command, its output and its exit), `storefiles.sh`, `policy_11.py`,
  `seam_census.py`, `skipped_check.py`.
- This log, `verification.md` and `issues.md`.

**Outside `var/handover-11/`:** nothing was written. In particular nothing under
`var/levin-phase-floor-01/` (not read either), and nothing in the store, its sidecar or
`var/runs/handover-07-…/`. The store's SQLite files were opened `mode=ro`.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — step 1 became a check: parallel dispatch.**
   - **The rule:** `git fetch origin && git merge --ff-only origin/handover-remedial`, and the
     `pip freeze` saved before and after the move.
   - **Why it could not be followed as written:** a second session (`levin-phase-floor` 01) runs at the
     same time in this checkout, and the dispatch forbids any git command that writes. The laptop
     orchestrator had already fast-forwarded the checkout to the dispatch SHA.
   - **Done:** `HEAD` is `757ef01…` and `git status --short` is empty (`gate_11.txt`,
     `step1_checkout.txt`, and again in `store_after_11.txt`). The `pip freeze` was taken once, since
     there was no move to take it either side of.
   - **The Gate's "the dispatch commit is pushed"** was checked read-only and without fetching:
     `git rev-parse origin/handover-remedial` (the local remote-tracking ref), `git branch -r
     --contains 757ef01`, and `git ls-remote origin refs/heads/handover-remedial`, which reads the
     remote and writes nothing locally. All three give `757ef01…`.
2. **IMPLEMENTATION CHOICE — a separate `policy_11.py` for step 3.**
   - **Found:** step 3 (the policy summary) and step 4 (`seam_census.py`) are separate steps, and
     step 4 does not list the policy among the script's contents.
   - **Done:** step 3 has its own read-only script, which prints the block, counts the rows and compares
     them cell by cell. Exit status 1 on a disagreement or a missing block. It uses the same `mode=ro`
     pattern.
3. **IMPLEMENTATION CHOICE — what the census tabulates beyond the text.**
   - **Stratum "clamped factors".** The prompt gives 0 to 2. `source` is a third factor that can be
     clamped, and 392 rows have all three clamped, so the table has a column `3`. The same table
     without `source` (Stratum 1b: `Tq` and `Tr` alone, 0 to 2) and the two strata crossed are added.
   - **Stratum "oscillatory factors in the last Levin sub-interval".** The prompt gives 0 to 3. A row
     with no Levin sub-interval has no such sub-interval, and 2489 rows are like that; they are placed
     in stratum 0 (every factor smooth), and the output says so.
   - **A "clamped factor in the row"** means a factor with a non-zero gap on any sub-interval of the
     row.
   - **Added:** the table at the grid values of $z_{\rm response}$ nearest 7.63e9 and 1.45e9
     (8.374e9 and 1.595e9), to set against the 2026-09 figures, which were taken at those two
     redshifts. The nearest grid values are not those exactly.
   - **Also in the CSV:** the row's `shard`/`serial`, `sub_index`, `n_osc`, `total` and `analytic_rad`.
4. **IMPLEMENTATION CHOICE — how the script is run and where it finds the constant.**
   - **The rule:** `./venv/bin/python -I scripts/seam_census.py <primary>`.
   - **Done:** run from the repository root as `./venv/bin/python -I
     var/handover-11/scripts/seam_census.py var/datastores/handover-A3-v2-lambdacdm.sqlite --expect
     9280`. The script locates the repository from its own path, reads `HANDOVER_CLAMP_MAX_GRID_STEPS`
     from `ComputeTargets/QuadSourceIntegral.py` with a regex (it does not import it), and reads the
     fingerprint it writes in its header from the sidecar. The fingerprint was then recomputed by
     `store fingerprint` (no `--write`) in step 6. The run took 1.4 s, so it ran in the foreground.
5. **IMPLEMENTATION CHOICE — extra read-only checks.**
   - **Done:**
     - `skipped_check.py`: which rows carry a `skipped` entry, against $q = r$. The census reports 3798
       rows with a `skipped` entry, and the prompt asks only for the count;
     - a `pip freeze` diff against 10's `freeze_after.txt` (the `ray` and `SQLAlchemy` lines differ,
       which is the `py313-venv` campaign's move, and not this prompt's);
     - a third hash of the store files after `skipped_check.py`: identical;
     - a closing `runregistrykit list` and `git rev-parse HEAD`.
   - **Why:** each is read-only, and each either strengthens "nothing was written" or reports a figure
     the census prints without explaining.
6. **UNINTENDED DRIFT — the first `storefiles.sh` run wrote garbage into
   `state/store_files_before.txt`.**
   - **What happened:** `stat -f` resolved to GNU `stat` in the session's `PATH`, which printed
     errors and no sizes or mtimes, so the file held hashes followed by junk.
   - **What was done about it:** the script was changed to call `/usr/bin/stat`, and re-run at once,
     overwriting the file, before the census ran. The hashes in the two runs were the same. The first,
     bad output is in this log only, and not in `state/`.
7. **UNINTENDED DRIFT — `out/store_after_11.txt` starts with a stray line, `diff rc 0`.**
   - **What happened:** a `diff … | tee out/store_after_11.txt` that was meant to be a scratch check
     wrote its status line to the file, before `rec.sh` took it over.
   - **What was done about it:** left as written. The file's recorded commands follow it.

## Verification performed

All of the following was run, and its output is saved.

- **Gate** (`gate_11.txt`): 10's board row is "reviewed" (`423aefd`); `origin/handover-remedial` and
  `HEAD` are `757ef01…`; `runregistrykit list`: 10 runs, 10 finished; `git status --short` empty;
  `store show`: kind `registry`, problems none, fingerprint `0e32f24a2dc2f4c9…`.
- **Step 2** (`state/store_files_before.txt`): SHA-256, size and mtime of the primary, four shards,
  sidecar and `stdout.log`.
- **Step 3** (`policy_11.txt`): `stdout.log` lines 89–102 verbatim; 1160 rows from the shards; **every
  cell agrees** (numeric/complete 363, mixed/complete 71, mixed/minimal 1, WKB/complete 725); no `fail`
  or `incomplete`; the one non-`complete` row is shard 2, serial 460, $k = 3.091682\times10^6$/Mpc,
  $z_{\rm response} = 3.963729\times10^{10}$, mixed, minimal, `crossover_z` $4.2472907\times10^{10}$.
- **Steps 4–5** (`seam_census_11.txt`): 9280 rows at `(1e-32, 1e-7)`, 2320 per shard, every partition
  parses; 19,289 sub-intervals; no gap above its row's allowance; the gap census; the oracle comparison.
  The script's SHA-256 is `240cc0638500048808deb81d3884c0855e5215dd0668edfb9d0a633aa9c4dcae`.
- **Step 6** (`store_after_11.txt`): the before and after store files and `stdout.log` are identical;
  `store fingerprint` (no `--write`) is `0e32f24a2dc2f4c9…`, problems none, "matches the fingerprint
  taken 2026-10-25T00:34:14+01:00 … at b90752d".
- **Acceptance 1–5:** hold (1: identical, fingerprint; 2: block transcribed, agrees, non-`complete` row
  listed; 3: 9280 rows with the three factors beside 2026-09; 4: the oracle comparison tabulated with
  its limits stated; 5: the records exist).
- **The test suites were not run.** The prompt does not ask for them and nothing here changes code.

## Observations not acted on

Each of these is in `issues.md`, for the orchestrator to open or drop.

1. **The census's strata are tied to $z_{\rm response}$.** The 2489 rows with no clamped factor are
   exactly the rows with one sub-interval and no Levin sub-interval, and occur only at
   $z_{\rm response}\ge10^9$ (2188 of them at $\ge10^{12}$). So the oracle comparison cannot be read as
   a clean effect of the clamp, over and above the oracle's own deficit.
2. **The `skipped` entries.** 3798 rows have one, always for `Tq`, always `log_width` 0.0, always with
   $q = r$ and equal `Tq` and `Tr` hand-over redshifts. The 1422 other $q = r$ rows have none.
3. **A `minimal` $G_k$ row exists** (shard 2, serial 460).
4. **Large maxima in the oracle comparison** (up to 96.9, at the $10^{10}$ decade; 20.5 at $10^8$). They
   are in the table and are not read.
5. **`Tq` and `Tr` gaps have identical quartiles overall** (Q1 6.925e-03, median 1.152e-02, Q3
   1.611e-02), and the same minimum and maximum.

## State handed to the next step

- **The checkout** is at `757ef01…`, clean, not moved.
- **The store** `var/datastores/handover-A3-v2-lambdacdm.sqlite` (with its four shards and sidecar) is
  byte-identical; its fingerprint is `0e32f24a…`, equal to the sidecar's. `stdout.log` is identical.
- **The registry** holds 10 runs, all finished. No run was registered by this prompt. No lock was
  taken, and no Ray process was started.
- **`var/handover-11/`** holds the records above, ready to be copied to `docs/handover/a3-v2/seam/`.
