## §4 — rehearsals R19 and R20 on the Mac Studio

**Prompt:** [`06e-exact-end-nodes.md`](../06e-exact-end-nodes.md) §4, under the orchestrator's
Part B dispatch (`6d81fad`) and its addendum B.2. The code measured is 06e's code commit
(`f4d8e17`), against its parent (`50277f3`), whose code is 06d's.
**Host:** mps026298, the Mac Studio.
- **R19** ran in the worktree `SecondaryGWKit-lpf06e-50277f3`, detached at `50277f3`. That is
  `f4d8e17`'s parent. Its code is the code 06c's census ran: `git diff --stat e5764a6 50277f3`
  outside `docs/` and `prompts/` shows only `CLAUDE.md`.
- **R20**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06e-6d81fad`,
  detached at `6d81fad`. Its code is `f4d8e17`'s: `git diff --stat f4d8e17 6d81fad` outside
  `docs/` and `prompts/` is empty.
  - **On code paths,** `50277f3..6d81fad` changes only `AdaptiveLevin/levin_quadrature.py` and
    four AdaptiveLevin tests.
  - **The launcher is the same file in both trees** (`b0135218…`), and so is
    `ComputeTargets/QuadSourceIntegral.py` (`962424c0…`).
- **The main checkout** stayed at `28ebaf0` and clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in three turns:
1. the gate, the lock and R19's launch;
2. R19's checks and R20's launch;
3. R20's checks, the comparison, the tabulation and Z1, the readers, the release and these records.

**Date:** 2026-10-30, 07:43–08:05 UTC.
**Commit:** none. The records are committed on the laptop, in 06e's records commit.
**Result:** COMPLETE. **Z1 HOLDS.** No §7 stop condition fired.
- **The fingerprints.** R19 and R20 both ended `done`, exit 0, with fingerprint `1b332ffa…`. Each
  equals R8's class by class, and R20's `classes` object equals R19's. `problems` is `{}` in both.
- **Outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` or the residual-memo counters
    (`rho_reuse`, 32 `GkWKBIntegration` rows);
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`,** all 864 rows are tabulated.
  - **`total`** is bit-identical in 361 rows: the 291 with no WKB Levin call, and 70 of the 573
    that have one.
  - **Every value field** is bit-identical in 205 rows, all without a Levin call. The other 86 such
    rows differ only in the analytic oracle (`analytic_rad` and its metadata), which uses Levin.
  - **No row with a Levin call** is bit-identical in every value field.
- **Z1 holds.**
  - **Scored:** all 864 rows, each with finite `total_abserr` in both stores. None of R8's
    rehearsal rows is one of 06a's 720 stationary-point rows; none is any A3 v2 row at all.
  - **Violations:** 0.
  - **The largest $|\Delta T|/(e_{19}+e_{20})$** is 0.0031. The largest $|\Delta T|/e_{19}$ is
    0.0062, and the largest $|\Delta T|/e_{20}$ is 0.0063, both at $k=3.16\times10^5$, $q=10^5$,
    $r=3.16\times10^5$, $z=437173$.
- **Convergence and depth:**
  - **`total_converged`** went from 863 to 864 of 864 rows.
  - **The one row at depth 20 in R19** (06c's R18 row, $k=10^6$, $q=3.16\times10^5$, $r=10^6$,
    $z=3971.4$) is at depth 7 in R20, with 394 → 92 regions, and is converged.
  - **No row reaches depth 20 in R20.**
- **The readers** exited 0, six of six. R20 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6e, "Exact end nodes".

### What shipped

No code. The records are under the main checkout's `var/levin-phase-floor-06e/rehearsal/`, for
the laptop to commit as `docs/levin-phase-floor/06e/rehearsal/`:
- `log.md` (this), `verification.md` and `issues.md`.
- **`scripts/`:**
  - `rec.sh` (`e5dc47a7…`): 06c's, with the record path changed;
  - `classify_non_qsi.py` (`26690e61…`): 05's, copied unchanged. It is byte-identical to 05's and
    06c's copies;
  - `run_readers_worktree.sh` (`ec96cf09…`): 06c's (06b's, 06's), copied unchanged;
  - `tabulate_qsi_endnodes.py` (`6b34c725…`): 06c's `tabulate_qsi_global.py`, adapted. Its diff
    against 06c's copy is `out/tabulate_qsi_endnodes.diff`.
- **`out/`:**
  - `commands.log`, every command;
  - the gate checks `00`–`02`;
  - each build's launch and end checks, and both `status.json`. The failed first form of R20's
    launch check is kept as `r20_launch_check.attempt1.txt`;
  - `store_value_diff.json` (`cdafa6fc…`, 42.1 MB) and `classify_non_qsi.txt` (`54ea89e8…`);
  - `qsi_rows.csv`, every row (`4ae92ce3…`); `qsi_tabulation.txt` (`b51f483b…`); and
    `qsi_extra_summary.txt`;
  - `r19_vs_06c_r18_csv_check.txt` and `z1_on_06c_y1_rows.txt`, read-only cross-readings against
    06c's committed tabulation;
  - `readers.txt`, `readers_check.txt` and `run_readers_worktree.diff`;
  - `final_check.txt`.
- **`state/`:**
  - `lock.log`;
  - both launch lines and the three markers;
  - every hash file of the stores: after R19's build, at R20's launch, before and after the
    comparison, before and after the readers, and final.
- **Not for copying back:** the readers' `extract/` (180 MB). `store_value_diff.json` is large, and
  is the orchestrator's to keep or drop.
- **The stores are kept:** `var/datastores/levin-phase-floor-06e-r19*` and `…-r20*`, five files
  each.

### Predictions and verdicts

| Rule (prompt 06e §4, as written) | Scored on | Verdict |
|---|---|---|
| §4.2: R19 and R20 each `done`; **both fingerprints must equal `1b332ffa…`** | R19 and R20 each `done`, exit 0, `1b332ffa…`. Classes equal R8's in both, and R20's equal R19's. `problems {}` | **holds** |
| §4.3: outside `QuadSourceIntegral` every value table is identical | 7 tables, 360,462 value rows, all identical. `value tables not identical: none` in every class | **holds** |
| §4.3: every other difference outside it is of the four volatile kinds | timing, label, `parents_digest`, and `rho_reuse` (32 `GkWKBIntegration` rows). **0 unclassified** | **holds** |
| §4.3: no row in one store only | 0 only-in-A, 0 only-in-B and 0 unmatched, in every class | **holds** |
| §4.3: inside `QuadSourceIntegral` every row is tabulated, with $\lvert T_{20}-T_{19}\rvert$ against `total_abserr` in each, the region counts, the maximum depths and `total_converged` in each | 864 of 864 rows, in `out/qsi_rows.csv` | **done** |
| **§4.3 Z1:** every row outside 06a's 720 stationary-point rows with finite `total_abserr` in both stores has $\lvert T_{20}-T_{19}\rvert \le e_{19}+e_{20}$. Rows with an infinite `total_abserr` in either are counted separately. The stationary-point rows are tabulated, not scored. Holds if no row violates it | 0 stationary-point rows in the store (deviation 4); 864 scored; **0 violate** (largest ratio 0.0031); 0 non-finite | **holds** |
| §4.4: every reader exits 0, and the store is byte-identical before and after | 6 of 6 exit 0. The 5 files are identical by SHA-256, size and mtime; no journal | **holds** |
| §4.5: lock released; `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing | released, exit 0; `status` free; `ray-processes` empty before and after the release | **holds** |
| §4.6: the stores are kept | both stores in place, unchanged since the comparison | **holds** |

**§6 item 2 (acceptance):** "R19 and R20 both `done` with `1b332ffa…`; outside
`QuadSourceIntegral`, every value table identical; inside it every row tabulated and Z1 scored;
the readers exit 0; the lock released." Every clause is met.

### Where the region counts and maximum depths are stored (§4.3)

`ComputeTargets/QuadSourceIntegral.py` and `ObjectFactories/QuadSourceIntegral.py` are the same
files at `50277f3` and `6d81fad` (`962424c0…`). They are the same at 06c's `e5764a6` too, whose
reading (06c's rehearsal log) this repeats, re-checked by grep:
- `ObjectFactories/QuadSourceIntegral.py` columns 255, 272 and 280;
- `_aggregate_LevinData`, which sums `num_regions` and takes the maximum of `max_depth`.

Everything is in each shard's table `QuadSourceIntegral`:
- **Region count, the call total:** column `WKB_Levin_num_regions`. SQLAlchemy `Integer`, SQLite
  integer, nullable. It is the **sum** of the calls' `num_regions`, over every
  `adaptive_levin_sincos` call of the row's WKB Levin sub-intervals (every phase group of every
  sub-interval). It is NULL when the row made no WKB Levin call (291 rows in both stores).
- **Region count, per call:** column `metadata`, a JSON string written by `json.dumps`, at
  `WKB_Levin.subintervals[i].groups[j].regions` (a JSON int). The script asserts that their sum
  equals `WKB_Levin_num_regions` on every row of both stores.
- **Maximum depth:** column `WKB_Levin_max_depth`. `Integer`, nullable. It is the **maximum** of
  the calls' `max_depth`, over the same calls. **No per-call depth is stored.**
- **The analytic oracle's region counts:** `metadata` at
  `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions` (a JSON int, `len(regions)` of each
  three-Bessel Levin block). These feed `analytic_rad`, not `total`. **No depth is stored for
  them.**
- **`total_converged`:** a column, `Boolean`, SQLite integer 0/1, nullable. It is the AND of every
  WKB Levin call's `converged`. The script asserts that it equals the AND of the stored group
  flags.
- **`total_abserr`** and **`total`:** columns, `Float(64)`.

**Work counters, not values** (06b's and 06c's classification, kept):
- **The columns:** `WKB_Levin_num_regions`, `WKB_Levin_evaluations`, `WKB_Levin_simple_regions`,
  `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`,
  `WKB_Levin_max_depth`, `numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`.
- **The metadata leaves:** `regions`, `simple_regions` and `evaluations`.
- **Why they are counters.** They count the partition the driver chose and the work it did. None
  of them is the integral or a bound on its error. 06e moves the partition through the values its
  leaves compute (§0.3, "through the loop so do region lists, depths"), without changing what any
  counter means. They are compared and tabulated, region counts and depth included, but a
  difference in one is not a value difference.

**Not compared:** timings, the label and store bookkeeping. **Everything else is a value**,
including `total_converged`, `total_phase_limited` and every `converged` and `phase_limited` leaf.

### Deviations from the prompt

1. **`run_readers.sh` cannot run from the code worktree as written.** It `cd`s to its own tree and
   calls `./venv/bin/python`, which the worktree lacks. 06c's `run_readers_worktree.sh` was used,
   copied unchanged (`cmp` identical, `ec96cf09…`). Its diff against `run_readers.sh`
   (`out/run_readers_worktree.diff`) is byte-identical to 06c's. `STRUCTURALLY REQUIRED`.
2. **The readers start Ray.** Each reader calls `ray.init(address="local",
   _temp_dir=/tmp/ray-sgk)` (`extract_QuadSourceIntegral_data.py` 132–134).
   - **The conflict:** the common brief says no Ray session may be started by anything but R19's
     and R20's drivers. Prompt §4.4 and this agent's brief both require the readers, with the lock
     held through them.
   - **What was done:** they ran under the lock, as 06b's and 06c's did. Each left no Ray process,
     and `ray-processes` was empty after them.
   - Recorded so that the orchestrator can confirm the reading. `STRUCTURALLY REQUIRED`.
3. **The tabulation is 06c's `tabulate_qsi_global.py`, adapted** as `tabulate_qsi_endnodes.py`
   (the brief allows this; diff in `out/tabulate_qsi_endnodes.diff`).
   - **Kept:** the `mode=ro` reader, the physical-key pairing, the field kinds, the `float.hex`
     comparison and the cross-check against the tool.
   - **Changed:**
     - R17/R18 became R19/R20, and Y1 became Z1;
     - 06a's 720 stationary-point rows are identified, tabulated and excluded from scoring
       (deviation 4);
     - the largest $|\Delta T|/e$ in each store is printed with its row;
     - one stray "18" in a label became "20", and "T17" became "T19".
   - `IMPLEMENTATION CHOICE`.
4. **How 06a's stationary-point rows are found in the rehearsal store.**
   - **The method:** `analyse06c.py`'s `scan06a` rule. That is the 1e-7 records of 06a's census
     checkpoint (`c13685b4…`) with a call whose `n_sign_changes > 0`, asserted to be 720. Each is
     mapped through 06a's `state/rows.json` (`aa523d28…`) to its physical key (k, q, r,
     z_response, parsed with `float` from the stored repr strings). A rehearsal row is among them
     iff its own physical key, read from the store, is equal as floats.
   - **The result is 0.** No rehearsal row is any A3 v2 row. Only $k=10^5$ is shared, and none of
     the rehearsal's 96 `z_response` values is among A3 v2's. So Z1 scores all 864 rows.
   - **The consequence:** the rehearsal store's own stationary-point calls, if it has any, are not
     identifiable by 06a's scan. They are scored with the rest (Observations 1).
   - `IMPLEMENTATION CHOICE` (the method). The prompt does not say how to carry an A3 v2 row set
     to R8's scope.
5. **One `ray-processes` look ran in a shared shell command.** It was the look after R19. It ran
   in the same Bash invocation as a read-only `python -I` (already exited) and two `tail`s. It
   listed nothing, exit 0. It was repeated at once as a command of its own (nothing, exit 0), and
   both results are in `state/lock.log` (lines 6 and 7). `UNINTENDED DRIFT`.
6. **The first form of R20's launch check failed to parse.** It used bash process substitution
   under `sh`: a syntax error, exit 2, and nothing ran. Its record is kept as
   `out/r20_launch_check.attempt1.txt`; the second form ran. `UNINTENDED DRIFT`.
7. **Two read-only cross-readings against 06c's committed tabulation** (not asked for):
   - **`out/r19_vs_06c_r18_csv_check.txt`:** R19 equals 06c's R18 on all 864 rows, same code, in
     `total`, `total_abserr`, `total_converged`, regions, depth, evaluations and `analytic_rad`, by
     the CSVs' `float.hex` strings. So the "before" is reproducible across runs at 06d's code.
   - **`out/z1_on_06c_y1_rows.txt`:** Z1's ratio on 06c's 30 Y1-violating rows.
   - CSV files were read only; no store was opened. `IMPLEMENTATION CHOICE`.
8. **Record slips, not deviations from the prompt:**
   - `out/01-gate-registry-lock.txt` checked the auth token with GNU `stat` and the wrong flags.
     It was re-checked with `/usr/bin/stat` in `out/02-gate-launcher.txt` (mode 600).
   - `out/r19_end_check.txt` has four lines appended after `rec.sh`'s footer: the fingerprint's
     `problems`, `taken`, formats and class counts.
   - In `out/final_check.txt`, the second `runregistrykit list | head -3` raised `BrokenPipeError`
     once `head` had closed the pipe. The first, `| tail -1`, gives the complete count (33 runs,
     all finished).

   `UNINTENDED DRIFT`, in records only.
9. **`black`.** `black --check` would reformat `tabulate_qsi_endnodes.py` and the copied
   `classify_non_qsi.py` (`out/final_check.txt`). Both are left as run, as records, as 06b's and
   06c's were; the copy must stay byte-identical to 05's. `IMPLEMENTATION CHOICE`.

### Verification performed

**The gate,** read-only (`out/00-gate-trees.txt`, `01-gate-registry-lock.txt`,
`02-gate-launcher.txt`):
- the main checkout is at `28ebaf0`, clean;
- both worktrees are clean (`--ignored` included), at `50277f3` and `6d81fad`;
- `f4d8e17^` is `50277f3`, and the code-path diffs are as the brief states;
- the registry held 31 runs, all finished;
- the lock was free, with no Ray processes and no `~/ScienceData/.rehearsal-lock`;
- `~/.ray/auth_token` is mode 600, and 97 GiB were free;
- the venv is 3.13.16 with the pinned packages;
- no 06e store or run existed.

**R19** (`out/r19_launch_check.txt`, `out/r19_end_check.txt`, `out/r19_status.json`):
- launched 07:44:39Z from the parent worktree (`sh` 98913, driver 98917), then `set-pid 98917`;
- run `levin-phase-floor-06e-r19-parent-20261030T074440`; manifest `git_head 50277f3…`,
  `git_dirty` false, cwd the parent worktree (`lsof`);
- `done`, exit 0, last heartbeat 07:49:32Z; the driver exited 07:49:46Z;
- digest `1b332ffa…`, equal to R8's and to the literal; classes equal R8's and 06c's R17's;
  `problems {}`;
- no Traceback, no `!! WARNING`, and `stderr.log` empty;
- the worktrees unwritten; `ray-processes` empty after it.

**R20** (`out/r20_launch_check.txt`, `out/r20_end_check.txt`, `out/r20_status.json`):
- launched 07:50:33Z from the code worktree (`sh` 99670, driver 99675), then `set-pid 99675`;
- run `levin-phase-floor-06e-r20-endnodes-20261030T075034`; manifest `git_head 6d81fad…`,
  `git_dirty` false, cwd the code worktree;
- `done`, exit 0, last heartbeat 07:55:07Z; the driver exited 07:55:38Z;
- digest `1b332ffa…`; classes equal R8's and R19's; `problems {}`;
- no Traceback, no `!! WARNING`;
- `ray-processes` empty after it.
- **Both `status.json`** carry `fingerprint_sidecar: "not written: … is not a problem-free
  registry sidecar (there is none)"`, as 06c's R17 did.

**The comparison** ran from the code worktree in 16 s: `python -B -m tools.store_value_diff R19 R20
--json`. It exited 1 because `QuadSourceIntegral` differs. The tool opens every file `mode=ro`.
- `classify_non_qsi.py` (`out/classify_non_qsi.txt`) reports 0 unclassified. Its per-class counts
  equal 06c's except for timing counts and `rho_reuse` (32 against 24).
- `tabulate_qsi_endnodes.py` (`out/qsi_tabulation.txt`, exit 0) checks its own reading:
  - 864 rows in each store, the same key set;
  - 48 columns, equal;
  - tolerances (atol 1e-32, rtol 1e-7) equal row by row;
  - the tool's 864 differing rows equal the stores' key set;
  - **0** `total` mismatches against the tool;
  - the metadata field sets are equal on every row;
  - 3,105 Levin calls in each store, the same per row;
  - `WKB_Levin_num_regions` equals the sum of the group counts, and `total_converged` equals the
    AND of the group flags, on every row of both.

**The stores:**
- They are identical before and after the comparison (`state/store_sha256_before_diff.txt` =
  `…_after_diff.txt`).
- R19 is unchanged since its build (`r19_store_sha256_after_build.txt`, and again at R20's
  launch).
- The final hashes (`state/stores_sha256_final.txt`) equal the pre-comparison ones.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- six of six exit 0, walls 24, 43, 19, 15, 13 and 16 s;
- 405, 2,436, 164, 27, 27 and 68 files;
- each leaves the store identical (SHA-256, mtime, size) with no journal, and 0 Ray processes
  after;
- 0 Tracebacks;
- six `!! WARNING: number of shards read from database (=4) does not match specified number of
  shards (=10)`, as in 06b and 06c.

**The lock** (`state/lock.log`). Every command exited 0, in this order:
1. `ray-processes` (empty);
2. `acquire`;
3. `set-pid 98917`;
4. `ray-processes` (empty; deviation 5);
5. `ray-processes` (empty);
6. `set-pid 99675`;
7. `ray-processes` (empty);
8. `ray-processes` (empty);
9. `release`;
10. `status` (free, none);
11. `ray-processes` (empty).

**At the end** (`out/final_check.txt`):
- the registry holds 33 runs, all finished;
- the main checkout is at `28ebaf0`, clean;
- both worktrees are clean, with nothing newer than the R19 marker;
- `~/.raypoolkit` is empty;
- 06a's checkpoint and `rows.json`, which were only read, have unchanged SHA-256.

### Z1 and the tabulation in detail

From `out/qsi_tabulation.txt`, `out/qsi_extra_summary.txt` and `out/qsi_rows.csv`.

**Z1.**
- 864 rows scored, 0 violations, 0 non-finite and 0 stationary-point rows.
- $|\Delta T|/(e_{19}+e_{20})$: median $8.3\times10^{-9}$, p90 $4.4\times10^{-7}$, p99
  $9.5\times10^{-6}$, max 0.0031.
- **The largest, in each store,** is at $k=3.16\times10^5$, $q=10^5$, $r=3.16\times10^5$,
  $z=437173.17$:
  - $|\Delta T|/e_{19}=0.00624$ and $|\Delta T|/e_{20}=0.00626$;
  - total19 `0x1.49a315251da54p-54`, total20 `0x1.49a31528f01e6p-54`;
  - abserr19 `0x1.32806dde5be7fp-77`, abserr20 `0x1.31684a68e80c9p-77`.
- $|\Delta T|/|T_{19}|$ on the rows with a Levin call: median $9.9\times10^{-15}$, max
  $1.3\times10^{-9}$.
- **On 06c's 30 Y1-violating rows** (`out/z1_on_06c_y1_rows.txt`), Z1's ratio is at most
  $1.8\times10^{-6}$. That includes the 12 equilateral `G+q+r` rows, where 06c's Y1 ratio was
  336–16,763. 06e leaves those totals where 06c and 06d put them.

**Bit-identity, by `float.hex`:**
- **`total`:** 361 of 864 rows (291 without a Levin call, 70 with one);
- **`total_abserr`:** 295;
- **every value field:** 205, all without a Levin call;
- **`analytic_rad`:** 320. The largest $|\Delta\,{\rm analytic\_rad}|$ over the summed analytic
  abserr is $7.7\times10^{-5}$.

**`total_abserr`:** $e_{20}/e_{19}$ runs from 0.987 to 1.006 (median 1.000). It is larger in 217
rows and smaller in 352.

**`total_converged`:**
- R19 has 863 of 864 rows converged, and R20 has 864. The one change (0→1) is the depth-20 row
  above.
- Unconverged Levin calls: 2 in R19, 0 in R20.
- **`total_phase_limited`:**
  - **By row:** 173 rows stay set, 58 drop, 22 are new, and 611 stay clear.
  - **Phase-limited Levin calls:** 474 in R19, 406 in R20.
  - **Phase-limited analytic blocks:** 3,506 in R19, 3,832 in R20, of 10,072.

**Regions and depth** (rows with a Levin call):
- **`WKB_Levin_num_regions` summed:** 34,892 in R19, 31,216 in R20 (−10.5 %).
- **Per row:**
  - regions are equal in 319 rows, more in R20 in 89 and fewer in 165;
  - R20/R19 runs from 0.195 to 1.55.
- **Evaluations:** 276,031 in R19, 246,623 in R20.
- **Analytic-block regions:** 35,996 in R19, 28,785 in R20.
- **The maximum depth** is equal in 463 rows, deeper in R20 in 35 and shallower in 75. Rows at
  depth ≥ 12: 46 in R19, 11 in R20. The depth histograms:

  | depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 18 | 19 | 20 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | R19 | 4 | 1 | 4 | 6 | 4 | 21 | 132 | 228 | 83 | 24 | 12 | 8 | 15 | 12 | 9 | 4 | 4 | 1 | 0 | 1 |
  | R20 | 4 | 1 | 4 | 6 | 4 | 23 | 145 | 247 | 75 | 33 | 12 | 8 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 0 |

- **The ten R20 rows deeper than 12** (`out/qsi_extra_summary.txt`) are all equilateral
  ($k=q=r$):
  - $k=q=r=10^5$: $z$ 329.6, 573.1, 755.7, 2284 and 3012;
  - $k=q=r=3.16\times10^5$: $z$ 1313.8, 1732.4 (depth 19), 3011.9 (depth 18) and 9104.3;
  - $k=q=r=10^6$: $z$ 1313.8 (depth 18).

  Each is converged in both stores, and each has $|\Delta T|/e_{19} \le 1.3\times10^{-5}$.

### Observations not acted on

1. **Z1's exclusion of 06a's stationary-point rows is vacuous on R8's rehearsal scope.** No
   rehearsal row is an A3 v2 row (deviation 4).
   - **The deep rows:** R20's ten rows deeper than 12 are all equilateral, at $z \le 9{,}105$. 06c's Y1 found its large `G+q+r` disagreements on that same class.
   - **Whether they hold stationary points:** they may, but no scan of this store was run. That
     would be computation outside §4.
   - **The effect here is none:** Z1 holds with them included.
2. **06e leaves 06c's twelve large Y1 disagreements where 06c put them.** On those `G+q+r` rows,
   R20 moves R19's total by at most $1.8\times10^{-6}$ of the summed declared errors. So this
   comparison cannot say whether 06b's or 06c/06d's value is right there.
   - That stays with 06c's N1 / `[06c-rehearsal-totals-disagree-beyond-both-declared-errors]`, if
     it was opened.
   - It may also stay with the near-singular class the addendum names
     (`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`, widened by 06e).
3. **The declared errors barely move at this scope.** $e_{20}/e_{19}$ is within 1.3 % of 1 on
   every row. On these rows `total_abserr` is evidently not dominated by Clenshaw–Curtis node
   terms, which are the term 06e tightens; the store does not record the split.
4. **Cost** (descriptive; README §5 rule 4):
   - Levin regions fell by 10.5 % and evaluations by 10.7 %;
   - analytic-block regions fell from 35,996 to 28,785;
   - rows at depth ≥ 12 fell from 46 to 11;
   - build wall was 5 m 07 s (R19) against 5 m 05 s (R20), with the host's load at about 11 at the
     end (desktop processes; 06c's R18 took 4 m 51 s at load 5).
5. **The readers' file counts differ from 06c's** (405/164/68 against 432/156/71 for `Gk`,
   `GkSource` and `QuadSourceIntegral`). `extract_Gk_data.py`, `extract_GkSource_data.py` and
   `extract_QuadSourceIntegral_data.py` draw unseeded `random.sample` subsamples, so the files
   differ from run to run. Not a store difference.
6. **`rho_reuse` differs in 32 `GkWKBIntegration` rows** (06c: 24). This is the residual memo's
   reuse counter, one of the four volatile kinds. `GkWKBIntegration`'s code is the same in both
   trees, so its count varies from run to run.
7. **`QuadSourceIntegral` stores no per-call depth,** as 06c noted. Depth is visible here only as
   the row maximum.

### State handed on

- **The stores** `var/datastores/levin-phase-floor-06e-r19*` and `…-r20*` are kept, unchanged since
  their builds. The final SHA-256 values are in `state/stores_sha256_final.txt`:
  - R19's primary `0e194eee…`;
  - R20's primary `34ca43d5…`.
- **The runs** `levin-phase-floor-06e-r19-parent-20261030T074440` and
  `…-r20-endnodes-20261030T075034` are both `done`. Their launch logs are
  `var/runs/levin-phase-floor-06e-r{19,20}.launch.log`.
- **The lock** is released, and `/tmp/ray-sgk` has no Ray process. The census agent may proceed;
  its census is Ray-free and needs no lock.
- **For the orchestrator:**
  - §4 holds in every clause, and Z1 holds.
  - Deviation 2 (the readers start Ray, under the lock) and deviation 4 (no stationary-point row in
    the rehearsal store) ask for a confirming reading.
