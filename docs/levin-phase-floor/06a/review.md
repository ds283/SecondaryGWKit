# levin-phase-floor 06a — the orchestrator's review (Mac Studio)

**Verdict: STOP, on check 9 alone.** One record names a script SHA-256 that no kept file has. The
first `c0.py`, which wrote `out/cmd/17_C0_check.txt` (`b33229d8…`), was changed (deviation 15) and
its text was not kept. Every other check passes, and C0's verdict rests on `out/cmd/18`, whose
`c0.py` (`2c1b8067…`) is the file in `scripts/`. The orchestrator does not repair this. The user
decides whether it is recorded as it stands.

- **Reviewed:** 2026-10-27, after the agent's final report, by the orchestrator, in its own
  session on the Mac Studio (mps026298).
- **The marker:** `<scratchpad>/marker`, touched 2026-10-27T02:10:14Z, immediately before the
  dispatch.
- **The dispatch:** SHA `e05c34f`. The worktree is
  `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06a-e05c34f`, and the main checkout is at
  `28ebaf0`.
- **The run:** `levin-phase-floor-06a-census-20261027T023143`.

The orchestrator's readers live in its own scratchpad: `rv_read.py` for checks 2–4, 6 and 7, and
`rv_read2.py` for check 7 from the pickles and for check 8. They import nothing from either tree
and run with `-I -B`. Floats are compared by `float.hex` / `.hex` strings, and arrays by
`dtype`, `shape` and `tobytes()`.

## The orchestrator's Gate (before the dispatch, 02:08–02:10Z), read-only

Every fact in the note's §1 held:
- **The main checkout:** `28ebaf0` on `handover-remedial`, `status --short` empty. Two worktrees:
  itself and `…-lpf06a-e05c34f` at `e05c34f`.
- **The worktree:** `status --short --ignored` empty. `diff --stat f5cee4c HEAD -- . ':!docs'
  ':!prompts'` shows `CLAUDE.md` alone (6+, 3−).
- **The registry and lock:** `runregistrykit list` shows 24 runs, all finished, none running or
  stale. The lock is free, and `ray-processes --ray-temp-dir /tmp/ray-sgk` lists none.
- **`var/`:** as §1 lists it, with no `levin-phase-floor-06a`.
- **The copy:** its five SHA-256, sizes and mtimes equal `copy_files_certified.txt`.
- **The baselines it saved:**
  - SHA-256, size and mtime for every file of A3 v2 and of the copy, in `gate_stores_before.txt`;
  - `/tmp/ray-sgk` with 59 entries, the newest from 2026-10-26 22:14;
  - 06's `var/levin-phase-floor-06/` with 3,413 files, and 06's run directory with 124.

## Check 1 — scope: **pass**

```
$ git --no-optional-locks rev-parse HEAD; git --no-optional-locks status --short   (main checkout)
28ebaf0743e0555fb2c4512ce714884005a60eff
(empty)
$ git -C <worktree> rev-parse HEAD; git -C <worktree> status --short --ignored; find <worktree> -newer marker
e05c34f2fcc6ef20fc3323cb3a6f81d130c5a01c
(empty)
(empty)
$ find var -newer marker | awk -F/ '{print $2"/"$3}' | sort | uniq -c
  var/ itself, var/runs/ itself (directory mtimes), levin-phase-floor-06a/ (155 entries),
  runs/levin-phase-floor-06a-census-20261027T023143 (134 entries)
$ find var/levin-phase-floor-06 -type f | wc -l; find <06 run dir> -type f | wc -l; find both -newer marker
3413
124
(empty)
```

## Check 2 — the run: **pass**

- **The registry:** `runregistrykit list` shows
  `levin-phase-floor-06a-census-20261027T023143  done  18560/18560`. 25 runs, all finished.
- **The manifest:**
  - `campaign: levin-phase-floor`, `prompt: 06a`, slug `census` (in the run id), `git_dirty: False`;
  - `git_head` is the main checkout's `28ebaf0`, as the purpose string explains.
- **The purpose string** names:
  - the dispatch SHA `e05c34f` and the worktree;
  - the copy's fingerprint `0e32f24a`;
  - "every A3 v2 QuadSourceIntegral row (9280)";
  - "rtol 1e-7 and 1e-8 (atol 1e-32)".
- **`status.json`:** state `done`, exit 0, units 18560/18560, heartbeat 03:17:10Z.
- **The checkpoint and the row list** (`rv_read.py`):
  ```
  checkpoint lines 18560; distinct units 18560 dups 0
  rows.json n rows 9280 sha aa523d28…; distinct row ids 9280
  1e-7 rows 9280 max count 1 set==rows.json True
  1e-8 rows 9280 max count 1 set==rows.json True
  statuses {('1e-7','done'): 9280, ('1e-8','done'): 9280}
  rows_sha in every record True
  ```
- **No over-budget re-runs** exist, so none sit beside the units.

## Check 3 — the tree, read: **pass**

- **Every record:**
  - `tree_head` is `e05c34f2…`;
  - `tree_status_short` is `""`;
  - `git_dirty` is `False`;
  - `script_sha256` is `83c4deec…`;
  - exactly one `scripts_sha256` map appears.
- **Module evidence:** each of the 16 `worker_*_hello.json` names 14 module files. Every path
  starts with the worktree's path. Each also gives `head` `e05c34f2`, an empty status, and
  `PYTHONPATH` equal to the worktree. The orchestrator checked the path prefixes itself, without
  reading `all_in_tree`.
- **`stdout.log`, from its `pid`/`exit` lines:** at most 6 workers were alive at once, and none
  was left alive. All eight `1e-7` workers exited before the first `1e-8` worker started.
- **The ledger:** the last `1e-7` record is line 9279 (02:55:54Z) and the first `1e-8` record is
  line 9280 (02:56:55Z).

## Check 4 — C0 and the order of work: **pass**

- **C0's five clauses are in the saved outputs:**
  - (a) `out/cmd/07`;
  - (b) `02` and `15` (the copy's fingerprint at $h$: `0e32f24a…`, problems none);
  - (c)–(e) `16` (the run) and `18` (the check). `18` says IDENTICAL for `low-G+r-1` at
    `1e-7` and `1e-8` and for `ctrl-low-1` at `1e-7`. Order-independence holds on the target row
    and on the seven others. `/tmp/ray-sgk` went 59 → 59, and no worker initialised Ray.
- **How C0 compares:** `c0.py` compares `total`, `total_abserr` and call values and abserrs by
  their `hex` strings, `x_span` by `float.hex`, and region arrays by
  `dtype`/`shape`/`tobytes()`. There is no `isclose`.
- **The order, by mtime:**

  | step | output | mtime |
  |---|---|---|
  | C0 | `16`, `18` | 02:25:22Z, 02:25:44Z |
  | the pilot | `19`, `20` | 02:28:23Z, 02:28:26Z |
  | the predictions | `state/predictions.md` | 02:28:40Z |
  | the launch | the run's `manifest.json` | 02:31:43Z |

- **The predictions file:** `state/predictions.md` (1,678 bytes, sha256 `3ad9b43d…`) equals the
  prompt's bytes from `## 4. Predictions` up to `## 5. Descriptive`, as read from the worktree.
- **Independently, `ctrl-low-1` at `1e-7`:** census unit `h-1e-7-2-2095` against 06's
  `s-1e-7-ctrl-low-1` (shard 2, serial 2095), with 7 calls on each side.
  - `total` is `-0x1.e3272e29a4c5dp-102` on both.
  - `total_abserr`, every call's value and abserr, `num_regions`, `num_simple_regions`,
    `max_depth`, `depth_hist_levin_cc`, `converged` and `phase_limited` show **no difference**.
- **Note:** a driver smoke test (`23`, 02:29–02:31Z, into `state/smoke`) ran after the
  predictions and before the launch. It is the agent's deviation 13. It was not registered and
  touched no store.

## Check 5 — the stores: **pass**

- **The store files:** SHA-256, size and mtime of A3 v2's five `.sqlite` files and the copy's five,
  re-taken after the run, equal the Gate's baseline (`diff` is empty).
- **No journal, `-wal` or `-shm` file** sits beside either.
- **The fingerprints**, run by the orchestrator without `--write`:
  ```
  $ ./venv/bin/python -B -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite
  digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56
  problems: none
  recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752d…
  $ PYTHONPATH=<worktree> python -B var/levin-phase-floor-06a/scripts/fingerprint_copy.py <copy primary>   (cwd: scratchpad)
  digest: "0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56"
  problems: none
  ```
  The worktree's `status --short --ignored` was still empty afterwards.
- **The lock**, each command run on its own: `rehearsal_lock status` gives
  `lock: free (…/rehearsal-lock does not exist)`, and
  `rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing (exit 0).
- **`/tmp/ray-sgk`:** 59 entries, and `find /tmp/ray-sgk -maxdepth 1 -newer marker` is empty.

## Check 6 — C4, re-scored: **pass**

- **`rv_read.py`:** the 39 rows with `names05` against 06's `s-1e-7-<name>` units, and the 31
  `s-1e-8-<name>` units 06 has. The same (shard, serial) was checked for each.
- **What was compared:** `total` and `total_abserr` (`hex`); each call's value and abserr
  (`hex`); `num_regions`, `num_simple_regions`, `max_depth`, `depth_hist_levin_cc`,
  `converged` and `phase_limited`.
  ```
  names05 count 39; 06 s-1e-8 units 31; compared 1e-7 39, 1e-8 31; failures []
  ```
- **Region arrays, from both runs' pickles:** each census record was first checked against its
  `pickle_record_sha256`. Then every ndarray 06 keeps (`start`, `end`, `depth`, `type`, `abserr`)
  was compared by `dtype`/`shape`/`tobytes()`, and `x_span` by `float.hex`:
  ```
  regions 1e-7 mid-Gqr-1: calls 5/5 arrays compared 25 differing 0
  regions 1e-8 R3:        calls 7/7 arrays compared 35 differing 0
  ```
- **C4 holds, 70 of 70.** The agent's verdict.

## Check 7 — C1–C3, re-scored: **pass** (the verdicts are the agent's)

- **Method:** row depth was taken from the checkpoint's per-call `max_depth`. It was then taken
  again from the depth arrays of all 18,560 pickle records, each sha256-checked. The two agree on
  every unit, and no record's sha256 mismatches.
  ```
  1e-7: ge18 0, max 12, >11 19, over budget 0, errored 0, no-Levin rows 2489
  1e-8: ge18 0, max 13, >11 29, over budget 0, errored 0, no-Levin rows 2489
  ```
- **C1 holds** (0 of 9,280 at `1e-7`). **C2 holds** (0 of 9,280 at `1e-8`).
- **C3 fails**: 19 rows exceed 11 at `1e-7`, all at depth 12; 29 at `1e-8`, 28 at 12 and one at 13.
- **The histograms equal the agent's**, bin for bin.

## Check 8 — the descriptive items: **pass**

- **`out/census_summary.txt`** (518 lines) has its predictions section and sections 1–9.
- **`out/census_rows.csv.gz`:** 8 comment lines, a header and **18,560 data lines**, 1.65 MB.
- **Spot check of the deepest row at `1e-7`, `3-38028`:**
  - its checkpoint record gives call depths `[8, 12, 0, 0, 0]`, regions summing to 33,
    `WKB_Levin_data` max_depth 12 with 33 regions, and `total` `0x1.26eb82499dce2p-64`;
  - its CSV line gives `h_depth` 12, `regions` 33 and `total_hex` `0x1.26eb82499dce2p-64`.
- **The stationary-point scan** (summary item 7) states its counts with the grid spacing beside
  them:
  - calls scanned: 37,325;
  - grid spacing $h$ in $\log(1+z')$: median 0.00253, min 1.69e-05, max 0.00712;
  - calls with an interior sign change: 720 (in 720 rows);
  - calls accepting a region across one: 720 (720 regions, 707 by the grid and 13 by the region's
    ends and midpoint).

  The limit is stated beside them: a pair of stationary points closer than $h$ is not seen.

## Check 9 — the harness: **STOP on one clause**

- **The diff against 06's copy:** `state/scripts_06_reference/` equals the worktree's
  `docs/levin-phase-floor/06/wider/scripts/` (`diff -r` empty). The orchestrator's own
  `diff -ru state/scripts_06_reference scripts` equals `out/scripts_diff_against_06_final.txt`
  (comment lines aside).
- **What it shows:**
  - changed files: `launch.py`, `prov.py`, `rebuild.py`, `rec.sh`, `rerun.py` and `worker.py`;
  - new files: `run_census.py`, `spscan.py`, `make_rows.py`, `c0.py`, `pilot.py`, the analysis and
    check scripts, `compare_hashes.sh` and `extract_predictions.sh`.

  Each change is a numbered, classified deviation (3–12, 14, 16, 20–22). The launch-time diff
  `out/scripts_diff_against_06_at_launch.txt` was also saved.
- **`convergence_kinds.py`** in `scripts/` is `cmp`-equal to
  `prompts/levin-phase-floor/measure/convergence_kinds.py`. The changed copy is
  `convergence_kinds_06a.py` (sha256 `e04a0a1e…`). Its diff, `out/convergence_kinds_06a.diff`
  (normal `diff` format), is current.
- **Every script SHA-256 in the records matches the file in `scripts/`: fails on one record.**
  Every `<name>.py|.sh … <sha256>` pair in `log.md`, `out/` and `state/` (outside
  `scripts_06_reference`), and in the run directory's JSON and `stdout.log`, was matched against
  `scripts/`:
  ```
  distinct (script,hash) pairs found 18; pairs naming a current script with a different hash: 2
   ('launch.py', 'c78bb528…', ['out/cmd/13_dev_smoke1.txt', 'out/cmd/23_smoke_driver.txt', 'out/commands.log'])
   ('c0.py',     'b33229d8…', ['out/cmd/17_C0_check.txt', 'out/commands.log'])
  ```
  - **`launch.py` `c78bb528…`: explained.** It is 06's `launch.py`, kept in
    `state/scripts_06_reference/launch.py` and in the worktree's `docs/`. The two smoke runs hashed
    it before it was changed (deviations 13 and 18). The run itself records `b7c57f45…`, the file
    in `scripts/`.
  - **`c0.py` `b33229d8…`: no kept file has this hash.** It is the version whose comparator
    included per-group `elapsed` timings and reported C0 (d) as failing (deviation 15, UNINTENDED
    DRIFT). The comparator was then changed in place, and that text was not kept. `out/cmd/18`, the
    C0 result the log relies on, records `2c1b8067…`, which is the file in `scripts/`. **This is
    the stop.** It is a gap in the record of a superseded check, not in any number C0–C4 rests on.

## Check 10 — the records: **pass**

- **`log.md`** follows campaign README §5.1:
  - subject, commit ("none here"; records by the orchestrator's commit) and result;
  - What shipped;
  - Predictions and verdicts: each rule as written, the number it was scored on, and the verdict;
  - Deviations, numbered 1–24 and each classified;
  - Verification performed, Observations not acted on (X1–X10), and State handed on.
- **Its deviations:** UNINTENDED DRIFT for 1, 2, 15, 17 and the first run in 21. Deviation 19 is
  a note that no over-budget pass was needed.
- **`verification.md`** is headed `## 6a. The depth census`, with subsections 6a.1–6a.4. Its links
  are in the form `06a/…`.
- **`issues.md`** exists, with X1–X10.
- **The run directory's size** is recorded in the log and in the agent's report: 220 MB,
  219,778,859 bytes. `du` gives 229,284 KiB on disk, and the directory holds 133 files.

## Notes for the laptop (not checks)

- **"All 48 rows"** in the log and X1 are 48 (rtol, row) entries over **29 distinct rows**. The 19
  rows deeper than 11 at `1e-7` are among the 29 at `1e-8`. The scan ran at `1e-7` only, and each
  of the 29 rows holds an interior sign change of θ′ in that scan.
- **What to leave out of the copy:**
  - `out/cmd/43_convergence_kinds_rerun.txt` is 3.9 MB, over the 2 MB limit;
    `out/convergence_kinds_h.txt` is its content without the per-record input lines;
  - the pickles under `state/C0`, `state/pilot`, `state/smoke` and `state/dev`.
- **The run:** wall time 02:31:43Z–03:17:10Z (45 min 27 s), against 0.84 h projected.
