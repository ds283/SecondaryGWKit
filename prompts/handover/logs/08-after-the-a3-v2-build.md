# Log 08 — after the A3 v2 build: the store's record, its checks, its readers, its depth census

> **Committed on the laptop** by the orchestrator (prompt 08 §7). A path `var/handover-08/…` below is
> that directory on the Mac Studio; its `out/`, `scripts/` and `state/` are committed at
> `docs/handover/a3-v2/after/`, and `verification.md` is `docs/handover/A3-V2-VERIFICATION.md` §3.
> `extract/` stays on the Mac Studio.


**Prompt:** prompts/handover/08-after-the-a3-v2-build.md (read at `087d420` with `git show`)
**Commit:** none on the Mac Studio (prompt §7). The checkout stays at `b90752d`.
**Model:** Claude Opus 5.5, in the Mac Studio session that ran 06 and 07.
**Date:** 2026-10-25 (00:33–00:45 BST).
**Result:** COMPLETE WITH DEVIATIONS. §3 hit a §8 stop (`extract_GkWKB_data.py` exited 1). The user
ruled "Release the lock, close 08" (deviation 6), and §6 then released the lock at 00:51:03. Every
other step is complete.

The record is `verification.md` (§3, "After the build").

## What shipped

Nothing tracked. Under `var/handover-08/`:
- `out/`: `prechecks_08.txt`, `store_record_08.txt`, `closed_store_state_08.txt`, `census_08.txt`,
  `store_soundness_08.txt`, `readers_08.txt`, `depth_census_r4_08.txt`, `depth_census_08.txt`,
  `run_logs_08.txt`, `run_record_08.txt`, `lock_08.txt`;
- `scripts/`: `depth_census.py`, `run_logs.py`, `storefiles.sh`, `prechecks.sh`, `store_record.sh`,
  `checks.sh`;
- `state/`: the store files' hashes before, after the record, after the checks and after the
  readers;
- `extract/`, 360 MB, which stays on the Mac Studio;
- this log, `verification.md` and `issues.md`.

Outside `var/handover-08/`: the sidecar `var/datastores/handover-A3-v2-lambdacdm.manifest.json`,
which `store create` writes, as §2 asks.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — stopped at §3.** `extract_GkWKB_data.py` exited 1, which is §8's "a
   reader that exits non-zero".
   - The running reader was not interrupted. The last, `extract_Gk_data` without `--models`,
     finished (exit 0), and the post-reader `store fingerprint` ran, because both are read-only and
     the readers ran as one command.
   - §5 (a read of the run record and the registry) was then completed, since it changes nothing.
   - §6 was not run at the stop, and the lock was left as found until the user's ruling
     (deviation 6), after which §6 ran.
2. **IMPLEMENTATION CHOICE — a fifth bin in §4.**
   - **Found:** 99 rows have `WKB_Levin_max_depth = 0`, which fits none of the prompt's bins (NULL,
     1–19, 20).
   - **Done:** the census reports them as their own bin, and its bins sum to 9280.
   - **Alternative:** folding them into 1–19, which would change that bin's mean from 0.421 s to
     0.418 s and its unconverged count by 65. Not chosen, because the baseline's definition is
     unknown.
3. **IMPLEMENTATION CHOICE — the census reads the shards from the primary's `shards` table**, as
   `run_readers.sh` does, rather than by `report()`'s name pattern. The result is the same here.
4. **IMPLEMENTATION CHOICE — the tail is bounded, not timed.**
   - **Found:** no progress line falls between 19:25:17 (52 in flight) and 21:56:15 (0 in flight),
     and the store's timestamps are those of the store batches.
   - **Done:** the moment fewer than six items remained cannot be read, so the tail is bounded at
     most 2.52 h, using the session's own `ps` snapshot at 21:48.
6. **STRUCTURALLY REQUIRED — the §8 stop resolved by the user's ruling.**
   - **The ruling**, given in chat with the orchestrator on 2026-10-25 and relayed to this session:
     **"Release the lock, close 08"**.
   - **What it means:** the GkWKB reader's failure stays recorded as the open issue
     `[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`, and is
     fixed later as its own work, when the reader is re-run.
   - **Done:** §6 was run as written at 00:51:03. `ray-processes` was empty; `release` exit 0; then
     `status` "lock: free" and `ray-processes` empty (`out/lock_08.txt`).
   - **Why this class:** the prompt could not complete as written, since a reader failed, and the
     way past was the user's decision, not the agent's.
5. **UNINTENDED DRIFT — the first `run_logs.py` output took the wrong "last line in flight".** It
   used the final 21:56:15 line, which has 0 compute in flight, and printed a 1-second tail. It was
   caught on reading and corrected to the last line with six or more compute items in flight
   (19:25:17). The saved output is the corrected run.

## Verification performed

- §1: every Gate item as stated (verification §3.1).
- §2: sidecar digest `0e32f24a…`, equal to the run's, `compare_fingerprints` 0 differences; the
  store files unchanged.
- §3, store checks: closed state clean, `no divergence`, `=> clean`, counts equal to the
  fingerprint's, store files unchanged.
- §3, readers: six exited 0 and one exited 1 (`extract_GkWKB_data.py`). Every store file was
  unchanged by every reader, the fingerprint afterwards matches, no Ray process was left, and the
  output totals 360 MB.
- §4:
  - the census, tested on R4 (864, unchanged), then run on A3 v2: 9280 rows at the pair, store
    unchanged;
  - **the cliff remains**: 2265 rows at depth 20, mean 454 s, 285.8 of 286.3 CPU-hr, 1179
    unconverged;
  - the tail is at most 5.2 % of the stage;
  - there are 15 warnings, none tied to a row.
- §5: 48.74 h of wall time and 286.33 CPU-hr.
- §6: `release` exit 0 on the user's ruling; afterwards the lock is free and `ray-processes` lists nothing.
- `git status --short` is empty throughout, with `HEAD` at `b90752d`.

## Observations not acted on

1. **The GkWKB reader's failure** is opened as
   `[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`. The failing
   pair is `k = 1e5`, `z_source = 1.3633e16` (serial 19), identified by its partial full-range PDF,
   which has no PNG or CSV, at `:285`'s `savefig`. Why it overflows (presumably a `0`, `inf` or `nan`
   on a log axis) was not confirmed against the data. An earlier suspicion of the one-point Green's
   function at `z_source = 0.1259` was checked and dropped: its plots were written.
2. **The 99 depth-0 rows** (deviation 2). Not opened.
3. **15 warnings, not 13**: Ray's dedup multipliers. Not opened, since none ties to a row.
4. **The cost overran the projection by 14.5 %.** The mid band took 36.7 CPU-hr against ~16 projected,
   and the low band 249 against ~233. For the orchestrator's reading of the projection; not opened.
5. During the census's comparison I found that the baseline's 7600 rows are the 7552 at
   $z > 174.1$ plus 48 of the 64 at $z = 132.08$. That set is not quite the "$z > 174.1$" population
   prompt §4 names, which is recorded as specified.

## State handed to the next step

- **The store** `var/datastores/handover-A3-v2-lambdacdm.sqlite` (+ 4 shards) now has its sidecar,
  `store_id` per `store_record_08.txt`, with fingerprint `0e32f24a…` written. It is unchanged since
  the build.
- **The lock is released** (00:51:03, exit 0); `status` says free and `ray-processes` lists
  nothing. ChamPBH and StochasticInstantons may take it. `~/ScienceData/` is left, empty.
- **The checkout** is at `b90752d`, clean. The freeze ended with §6, and this prompt does not move
  the checkout.
- **`extract/`** holds every reader's output, the partial GkWKB output among it (344 files).
