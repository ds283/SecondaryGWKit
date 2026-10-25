# Log 10 — re-read A3 v2 with the fixed readers

> **Committed on the laptop** by the orchestrator (prompt 10 §2). A path `var/handover-10/…` below
> is that directory on the Mac Studio; its `out/`, `scripts/` and `state/`, with `issues.md` and the
> orchestrator's `review.md`, are committed at `docs/handover/a3-v2/reread/`, and `verification.md`
> is `docs/handover/A3-V2-VERIFICATION.md` §4, its links corrected from `reread/…` to
> `a3-v2/reread/…` on the user's decision (review check 9). `extract/` stays on the Mac Studio.

**Prompt:** prompts/handover/10-reread-a3-v2.md (read at `57f67a9` with `git show`, before the move)
**Commit:** none on the Mac Studio (prompt header; dispatch rules). The checkout was fast-forwarded
`b90752d` → `57f67a9` (step 1).
**Model:** Claude Opus 5.5, in a Claude Code session on the Mac Studio (`mps026298`).
**Date:** 2026-10-25 (10:47–11:00 GMT).
**Result:** COMPLETE WITH DEVIATIONS. No §3 stop was hit, and acceptance 1–4 hold. The deviations
are in how things were recorded, not in what was run.

The record is `verification.md`, §4 "Re-read with the fixed readers".

## What shipped

Nothing tracked, and no commit. Under `var/handover-10/`:
- `out/`:
  - `gate_10.txt`, the Gate;
  - `step1_checkout.txt`, the fetch, the fast-forward and the post-move checks;
  - `pip_10.txt`: the freeze compare, the dry run, the install, the freeze after, `pip check` and
    the provenance;
  - `store_10.txt`: `store show`, and `store fingerprint` before the readers;
  - `lock_10.txt`, every lock command, with its exit status;
  - `lock_help_10.txt`, the module's `--help` text;
  - `readers_10.txt`, the readers;
  - `gkwkb_readback_10.txt`, step 6;
  - `reader_logs_scan_10.txt`: the error and warning scan of the reader logs, the file kinds, and
    `HEAD` and status at the end;
  - `store_after_10.txt`: step 7's store files, fingerprint and Ray-process check;
  - `registry_after_10.txt`.
- `state/`:
  - `store_files_before.txt` and `store_files_after.txt`;
  - `freeze_before.txt`, `freeze_after.txt`, and `freeze_06.txt`, which is lines 7–107 of
    `docs/handover/a3-v2/rehearsal/out/pip_06.txt`.
- `scripts/`:
  - `rec.sh`, which records a command, its output and its exit;
  - `storefiles.sh`;
  - `readers_10.sh`, the wrapper round `run_readers.sh`;
  - `gkwkb_readback.py`, step 6's counts and the dropped points.
- `extract/`, 879 MB, the readers' output and logs. It stays on the Mac Studio.
- This log, `verification.md` and `issues.md`.

**Outside `var/handover-10/`:**
- the fast-forward of the checkout to `57f67a9` (step 1);
- `./venv`'s `raypoolkit` moved 0.2.0 → 0.3.0 (step 2, the prompt's own amendment);
- the lock directory `~/.raypoolkit/rehearsal-lock`, taken and released, which leaves
  `~/.raypoolkit/` empty.

Nothing else was written. The store, its shards and its sidecar are byte-identical before and after.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — the exit status of each lock command is recorded in the same tool
   call.**
   - **The rule:** each lock command is its own shell command, appending with
     `>> var/handover-10/out/lock_10.txt 2>&1`, and `echo $?` comes "in the next command".
   - **Why it could not be followed as written:** each Bash tool call here is a fresh shell, so a
     separate call cannot see the previous call's `$?`.
   - **Done:** every lock command was run as `<lock command> >> lock_10.txt 2>&1; rc=$?; echo
     "[exit $rc]" >> lock_10.txt; echo $rc`. The lock command is the only process started, since
     everything after it is a shell builtin, and there is no pipe and no `tee`.
   - **The headers** above each lock command were written by separate tool calls, which named
     nothing that starts Ray.
2. **UNINTENDED DRIFT — the first lock `status` is incomplete in the record.**
   - **What happened:** it was run without `--ray-temp-dir`, and without its exit status recorded.
     The tool reported success. Its output, "lock: free", and "Ray processes: no temp dir given",
     is the first two lines of `lock_10.txt`, annotated afterwards.
   - **What was done about it:** it was re-run at once with `--ray-temp-dir /tmp/ray-sgk` and its
     exit recorded: free, none, 0. Kept as written, since the record is additive.
3. **IMPLEMENTATION CHOICE — `status` and `ray-processes` were given `--ray-temp-dir
   /tmp/ray-sgk`.**
   - **Found:** `status --help` shows that `status` takes an optional `--ray-temp-dir`, so that it
     lists the processes.
   - **Done:** `status` was always given it, and `ray-processes` was given it alone, as the note
     says. No `--project` was passed to either.
4. **IMPLEMENTATION CHOICE — the Gate's "lock free" was checked at both locations.**
   - **Done:** both `~/.raypoolkit/rehearsal-lock` and `~/ScienceData/.rehearsal-lock` were checked,
     and so was the old module, `tools.rehearsal_lock status`, at `b90752d`, before step 1 removed
     it. All three said free.
   - **Why:** the lock in force at the Gate was the one at `b90752d`, and step 2's text names the
     new one "from here".
5. **IMPLEMENTATION CHOICE — the `pip_06.txt` comparison uses that file's freeze lines alone.**
   - **Found:** the file also holds headers, a `diff`, `pip check` and a provenance block.
   - **Done:** its lines 7–107, the 101-line `pip freeze`, were compared, and they are saved as
     `state/freeze_06.txt`.
6. **IMPLEMENTATION CHOICE — extra read-only checks.**
   - **Done:**
     - a `store fingerprint` (no `--write`) before the readers as well as after, to give a
       "fingerprint before";
     - the store files compared with 08's `store_files_before.txt` too (identical);
     - `storefiles.sh` also records the sidecar's hash, size and mtime, on a `SIDECAR` line;
     - a scan of every reader log for errors and warnings;
     - a closing `runregistrykit list`.
   - **Why:** each is read-only, and each strengthens "nothing was written".
7. **IMPLEMENTATION CHOICE — the readers were run under the tool's background mode.**
   - **The rule:** the dispatch allows this when the run may exceed 600 s.
   - **Done:** the readers ran as one wrapper (`scripts/readers_10.sh`) in the background, and the
     completion notification was awaited. They took 6 m 52 s. The wrapper itself is a plain
     foreground `bash` script, and it is not detached.
   - **The checks while waiting:** one look at `readers_10.txt`, and one at the first reader's log
     tail, about a minute in. No poll loop was run.
8. **IMPLEMENTATION CHOICE — the dropped-point count's scope.**
   - **Found:** the reader does not report how many points it dropped.
   - **Done:**
     - the count was taken for the pair from its CSV, as the prompt says, by calling
       `extract_common.sensible_log_points` itself on the CSV's columns;
     - it was also summed over all 1392 CSVs this run wrote, labelled as this run's random sample
       and not the store.
   - **Alternative considered:** a store-wide count, from a read-only pass over every
     `GkWKBIntegration`/`GkNumericIntegration` row. Not chosen: it is outside the prompt, and the
     prompt names the fallback.

## Verification performed

All of the following was run, and its output is saved.

- **Gate** (`gate_10.txt`):
  - 09's board row is "reviewed", with `39809dca…`;
  - `bacbab6` is in `b90752d..57f67a9`, and `origin/handover-remedial` = `57f67a9`;
  - `git diff --stat … -- ObjectFactories config main.py` gives exactly `config/ray_local.py`
    (+7) and `main.py` (+8 −2), which is the allowed diff, and the full diff is saved;
  - the registry has 5 runs, all finished;
  - the lock is free (both locations, and the old module);
  - `git status --short` is empty.
- **Step 1:** fast-forward `b90752d..57f67a9` (80 files). `HEAD` = `57f67a9`, `git status --short`
  is empty, and the Gate diff is unchanged.
- **Step 2:**
  - the freeze equals `pip_06.txt`'s (rc 0);
  - the dry run says "Would install raypoolkit-0.3.0" alone;
  - after the install, the freeze differs on the one `raypoolkit` line (`eaef0ce6…` →
    `67a75b97a656accb98195a4bda93f64e6a5f3fb2`);
  - `pip check` is clean.
- **Step 3:**
  - `store show`: kind `registry`, problems none, fingerprint `0e32f24a…`;
  - `store fingerprint` before the readers matches it;
  - the store files are identical to 08's.
- **Step 4:** `acquire` exit 0 at 10:49:36Z. `status` then reports HELD, `pid: -`, exit 0.
- **Step 5:** see the table below. All seven exit 0, both `run_readers.sh` invocations return rc 0,
  and every reader leaves the store files identical. No reader log has a Traceback, `ReadOnly*` or
  `FATAL`, and no Ray process was left after any reader.

  | Script | Exit (08) | Wall (08) | Files (08) | KiB (08) |
  |---|---|---|---|---|
  | `extract_Gk_data` | 0 (0) | 54 s (57 s) | 1251 (1593) | 78,616 (102,080) |
  | `extract_GkWKB_data` | **0 (1)** | 171 s (24 s) | **9744 (344)** | 551,064 (18,680) |
  | `extract_GkSource_data` | 0 (0) | 34 s (33 s) | 847 (873) | 69,684 (72,268) |
  | `extract_TkWKB_data` | 0 (0) | 19 s (19 s) | 72 (72) | 6,936 (6,936) |
  | `extract_tensor_source_data` | 0 (0) | 22 s (22 s) | 172 (172) | 7,672 (7,672) |
  | `extract_QuadSourceIntegral_data` | 0 (0) | 31 s (33 s) | 687 (681) | 58,512 (57,444) |
  | `extract_Gk_data`, no `--models` | 0 (0) | 58 s (56 s) | 1980 (1593) | 127,032 (102,944) |

  In total, 879 MB against 360 MB.
- **Step 6:**
  - **The counts.** There are 1392 full-range PDFs, 1392 PNGs and 1392 CSVs, and their pair sets
    are equal. The work list is 1392 pairs, read from the reader's log: line 32 `792/1392 work items
    remaining`, line 34 `252/1392 …` and line 37 `Queue summary: 1392 lookup`. It agrees with
    `extract_GkWKB_data.py:509`'s `product`: 8 `k` × 174 `z_source`.
  - **The pair** `k=1e+05-k-serial=1/zsource=1.3633e+16-z-serial=19` has its PDF (30,366 B), its
    PNG (128,806 B) and its CSV (34,150 B, 154 rows).
  - **The dropped points:** 35 of the pair's 154, all above the ceiling of 4.0099e+24
    (= 10³ × max $|G|$ 4.0099e+21). The largest dropped value is 7.5709e+243. Over all 1392 sampled
    pairs, 5162 of 109,594 points were dropped: 5034 above the ceiling and 128 exact zeros.
- **Step 7:**
  - the store files are identical to step 3's (diff rc 0);
  - `store fingerprint`: `0e32f24a…`, problems none, "matches the fingerprint taken
    2026-10-25T00:34:14+01:00 … at b90752d";
  - no Ray process is left, by both `ps` and `ray-processes`.
- **Step 8:** `ray-processes` empty (exit 0), then `release` exit 0 at 10:57:50Z, then `status`
  "lock: free" (exit 0), then `ray-processes` empty (exit 0).
- **The test suites were not run.** The prompt does not ask for them, and 09 ran them on the
  laptop at this tree's code.

## Observations not acted on

Each of these is in `issues.md`, for the orchestrator to open or drop.

1. **The readers draw unseeded random subsamples.** `extract_GkWKB_data.py:472` uses
   `random.sample` at about 10 % of the `z_source` values. The failing pair was in this run's draw,
   but its presence was a roughly 1-in-10 chance, not a property of the procedure. The other
   readers' file counts vary between runs for the same reason.
2. **`status` exited 0, not 3,** while the lock was held with no pid recorded, against the
   orchestrator's note §0.
3. **The theta figure is never closed.** `extract_GkWKB_data.py:322` opens a figure after the
   `plt.close()` at `:319`, and matplotlib now warns "More than 20 figures have been opened" in the
   GkWKB log. This is pre-existing, and it was not visible in 08 because that run failed early.
4. **The pre-existing warnings recur.** "!! WARNING: number of shards read from database (=4) does
   not match specified number of shards (=10)" is printed by every reader, as in 08, and
   `extract_GkSource_data.py:671` prints a `MatplotlibDeprecationWarning`. Not opened, since 08 had
   both.

## State handed to the next step

- **The checkout** is at `57f67a9`, clean, with `HEAD` = `origin/handover-remedial` as fetched at
  10:47Z.
- **The venv:** `raypoolkit` 0.3.0 (`67a75b97…`); the rest is as `pip_06.txt`.
- **The store** `var/datastores/handover-A3-v2-lambdacdm.sqlite` (and its 4 shards and sidecar) is
  unchanged. Its fingerprint is `0e32f24a…`, equal to the sidecar's.
- **The lock is released.** `~/.raypoolkit/rehearsal-lock` does not exist, and no process is under
  `/tmp/ray-sgk`. `/tmp/ray-sgk/` holds seven Ray session directories from the readers.
- **The registry** holds 5 runs, all finished. The readers register no run.
- **`extract/`** (879 MB) holds the complete GkWKB read-back and every other reader's output.
