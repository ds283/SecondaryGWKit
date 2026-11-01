## §4 — rehearsals R21 and R22 on the Mac Studio

**Prompt:** [`06h-group-phase-rounding.md`](../06h-group-phase-rounding.md) §4, under the
orchestrator's Part B dispatch (`c1ec3b2`) and its addendum B.2. The code measured is 06h's code
commit `7172a8f` with its tests-only follow-up `ac1c6f4`, against their parent `2c325f5`, whose code
is 06e's (`f4d8e17`).

**Host:** mps026298, the Mac Studio.
- **R21** ran in the worktree `SecondaryGWKit-lpf06h-2c325f5`, detached at `2c325f5`, `7172a8f`'s
  parent. Its code is the code 06e's Part B ran: `git diff --stat 6d81fad 2c325f5` outside `docs/`
  and `prompts/` shows only `CLAUDE.md`.
- **R22**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06h-c1ec3b2`,
  detached at `c1ec3b2`. Its code is `ac1c6f4`'s: `git diff --stat ac1c6f4 c1ec3b2` outside
  `docs/` and `prompts/` is empty.
  - **On code paths,** `2c325f5..c1ec3b2` changes `LiouvilleGreen/three_bessel_integrals.py`, a
    docstring sentence of `LiouvilleGreen/bessel_phase.py`, and three test files.
  - **The same file in both trees:** the launcher (`b0135218…`, R19's and R20's), `main.py`
    (`a5d47f5f…`), `ComputeTargets/QuadSourceIntegral.py` (`962424c0…`) and
    `AdaptiveLevin/levin_quadrature.py` (`8285c1e4…`).
  - **`three_bessel_integrals.py`** is `862a18fe…` at the parent and `fea67d56…` at 06h. These are
    the hashes the code log records.
- **The main checkout** stayed at `28ebaf0`, clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in three turns:
1. the gate, the lock and R21's launch;
2. R21's checks and R22's launch;
3. R22's checks, the comparison, the tabulation and H0, the readers, the release and these records.

**Date:** 2026-11-01, 00:18–00:36 UTC.

**Commit:** none. The records are committed on the laptop, in 06h's records commit.

**Result:** COMPLETE. **H0 HOLDS.** No §7 stop condition fired.
- **The fingerprints.**
  - R21 and R22 both ended `done`, exit 0, with fingerprint `1b332ffa…`.
  - Each equals R8's class by class, and R22's `classes` object equals R21's (and R19's).
  - `problems` is `{}` in both.
- **Outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` or the residual-memo counters
    (`rho_reuse`, 24 `GkWKBIntegration` rows);
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`,** all 864 rows are tabulated.
  - **`total` is bit-identical in all 864 rows, and so are `total_abserr`, `total_converged`,
    `total_phase_limited`, `numeric_quad`, `WKB_Levin`, and every WKB Levin region count, depth
    and evaluation count.**
  - **The reason is structural.** `total` is `numeric_quad + WKB_Levin`
    (`ComputeTargets/QuadSourceIntegral.py` 799). The WKB Levin calls use
    `ComputeTargets/phase_groups.py`'s groups, which 06h does not change (§0.3).
    `BesselPhaseGroup` is used only by `_three_bessel_Levin` (1192–1195), the analytic oracle.
    The oracle's result is `analytic_rad` (815) and the metadata `analytic.*`, which are not in
    `total` (Observation 1).
  - **What moved** is the analytic oracle:
    - `analytic_rad` moved in 464 rows, all with a three-Bessel Levin block. It is bit-identical
      in 400: the 205 rows with no such block, and 195 of the 659 with one;
    - `analytic.abserr` moved in all 659 rows with a block;
    - the blocks' region count fell from 28,785 to 24,428;
    - every block converged under both codes.
  - **Every value field** is bit-identical in 205 rows, all without a three-Bessel Levin block.
- **H0 holds** for all 864 rows, scored with finite `total_abserr` in both stores.
  - **Stationary-point rows:** none. No rehearsal row is one of 06a's 720 stationary-point rows,
    or any A3 v2 row.
  - **Violations:** 0. Every $|\Delta T|$ is 0, so the largest $|\Delta T|/e_{21}$ and
    $|\Delta T|/e_{22}$ are both 0.
  - **The same rule applied to `analytic_rad` and `analytic.abserr`** (descriptive, not H0) also
    has 0 violations. The largest $|\Delta\,{\rm analytic\_rad}|/(e_{21}+e_{22})$ is
    $2.49\times10^{-5}$.
- **The readers** exited 0, six of six. R22 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6h, "The group phase's rounding".

### What shipped

No code. The records are under the main checkout's `var/levin-phase-floor-06h/rehearsal/`, for
the laptop to commit as `docs/levin-phase-floor/06h/rehearsal/`:
- `log.md` (this), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.
- **`scripts/`:**
  - `rec.sh` (`a24f685e…`): 06e's, with the record path changed by `sed`;
  - `classify_non_qsi.py` (`26690e61…`): 05's, copied unchanged. It is byte-identical to 05's and
    to 06e's copy;
  - `run_readers_worktree.sh` (`ec96cf09…`): 06e's (06c's, 06b's, 06's), copied unchanged;
  - `tabulate_qsi_groupphase.py` (`dea79d01…`): 06e's `tabulate_qsi_endnodes.py` (`6b34c725…`),
    adapted. Its diff against 06e's copy is `out/tabulate_qsi_groupphase.diff`;
  - `qsi_analytic_summary.py` (`7cb08e8f…`): new, descriptive. It covers the analytic oracle, which
    H0's `total` does not reach. It imports `tabulate_qsi_groupphase.read`.
- **`out/`:**
  - `commands.log`, every command;
  - the gate checks `00-gate-trees.txt` and `01-gate-registry-lock.txt`;
  - each build's launch and end checks (`r21_launch_check`, `r21_end_check`, `r21_end_check2`,
    `r22_launch_check` and `r22_end_check`), both `status.json`, and the registry listings;
  - `store_value_diff.json` (`68b9c378…`, 41.0 MB; kept on the Mac Studio) and its `.stderr`;
  - `classify_non_qsi.txt` (`571a48c7…`);
  - `qsi_rows.csv`, every row (`840ef35c…`), and `qsi_tabulation.txt` (`6a206e53…`);
  - `qsi_analytic_rows.csv`, every row (`f98ffebb…`), and `qsi_analytic_summary.txt`
    (`41091fa8…`);
  - `r21_vs_06e_r20_csv_check.txt`, a read-only cross-reading against 06e's committed tabulation;
  - `readers.txt`, `readers_check.txt` and `run_readers_worktree.diff`;
  - `final_check.txt`.
- **`state/`:**
  - `lock.log`;
  - both launch lines and the three markers;
  - every hash file of the stores: after R21's build, at R22's launch, before and after the
    comparison, before and after the readers, and final.
- **Not for copying back:**
  - the readers' `extract/` (172 MB, 3,099 files);
  - `out/store_value_diff.json` (41.0 MB), listed in `KEPT_ON_MAC_STUDIO.txt`.
- **The stores are kept:** `var/datastores/levin-phase-floor-06h-r21*` and `…-r22*`, five files
  each. Their run directories are `var/runs/levin-phase-floor-06h-r21-parent-20261101T001852/` and
  `…-r22-groupphase-20261101T002423/`, with launch logs
  `var/runs/levin-phase-floor-06h-r{21,22}.launch.log`.

### Predictions and verdicts

| Rule (prompt 06h §4, as written) | Scored on | Verdict |
|---|---|---|
| §4.2: R21 and R22 each run as R19 and R20 ran, registered `--campaign levin-phase-floor --prompt 06h`, slugs `r21-parent` and `r22-groupphase`; **both fingerprints must equal `1b332ffa…`** | Both `done`, exit 0, `1b332ffa…`. Classes equal R8's in both, and R22's equal R21's. `problems {}`. Same launcher (`b0135218…`) and arguments as R19 and R20 | **holds** |
| §4.3: outside `QuadSourceIntegral` every value table is identical | 7 tables, 360,462 value rows, all identical. `value tables not identical: none` in every class | **holds** |
| §4.3: every other difference outside it is of the four volatile kinds | timing, label, `parents_digest`, and `rho_reuse` (24 `GkWKBIntegration` rows). **0 unclassified** | **holds** |
| §4.3: no row in one store only | 0 only-in-A, 0 only-in-B and 0 unmatched, in every class | **holds** |
| §4.3: inside `QuadSourceIntegral` every row is tabulated, with $\lvert T_{22}-T_{21}\rvert$ against `total_abserr` in each, the region counts, the maximum depths and `total_converged` in each | 864 of 864 rows, in `out/qsi_rows.csv` | **done** |
| **§4.3 H0:** every row outside 06a's 720 stationary-point rows with finite `total_abserr` in both stores has $\lvert T_{22}-T_{21}\rvert \le e_{21}+e_{22}$. Rows with an infinite `total_abserr` in either are counted separately, and the stationary-point rows are tabulated, not scored. Holds if no row violates it | 0 stationary-point rows in the store; 864 scored; **0 violate** ($\lvert\Delta T\rvert = 0$ in every row); 0 non-finite | **holds** (Observation 1: it cannot fail here) |
| §4.4: every reader exits 0, and the store is byte-identical before and after | 6 of 6 exit 0. The 5 files are identical by SHA-256, size and mtime; no journal | **holds** |
| §4.5: lock released; `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing | released, exit 0; `status` free; `ray-processes` empty before and after the release | **holds** |
| §4.6: the stores are kept | both stores in place, unchanged since the comparison | **holds** |

**§6 item 2 (acceptance):** "R21 and R22 both `done` with `1b332ffa…`; outside
`QuadSourceIntegral`, every value table identical; inside it every row tabulated and H0 scored; the
readers exit 0; the lock released." Every clause is met.

### Where the region counts and maximum depths are stored (§4.3)

`ComputeTargets/QuadSourceIntegral.py` (`962424c0…`) is the same file at `2c325f5` and `c1ec3b2`,
and the same as at 06e's `50277f3` and `6d81fad`. So 06e's reading holds unchanged. The script
re-checks it by assertion on every row of both stores.

Everything is in each shard's table `QuadSourceIntegral`:
- **Region count, the call total:** column `WKB_Levin_num_regions`. SQLAlchemy `Integer`, SQLite
  integer, nullable. It is the **sum** of the calls' `num_regions`, over every
  `adaptive_levin_sincos` call of the row's WKB Levin sub-intervals (every phase group of every
  sub-interval; `_aggregate_LevinData`). It is NULL when the row made no WKB Levin call (291 rows
  in both stores).
- **Region count, per call:** column `metadata`, a JSON string written by `json.dumps`, at
  `WKB_Levin.subintervals[i].groups[j].regions` (a JSON int). The script asserts that their sum
  equals `WKB_Levin_num_regions` on every row of both stores.
- **Maximum depth:** column `WKB_Levin_max_depth`, `Integer`, nullable. It is the **maximum** of
  the calls' `max_depth`, over the same calls. **No per-call depth is stored.**
- **The analytic oracle's region counts:** `metadata` at
  `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions` (a JSON int, `len(regions)` of each
  three-Bessel Levin block). Each block also stores `abserr` (JSON float), `converged` and
  `phase_limited` (JSON bool), and `elapsed` (a timing). The row stores `analytic.abserr` and
  `analytic.converged`, and the column `analytic_rad` (`Float(64)`).
  - These blocks feed `analytic_rad`, not `total`.
  - **No depth and no per-block value is stored for them.**
  - **They are the only place 06h's code reaches in this table.**
- **`total_converged`:** a column, `Boolean`, SQLite integer 0/1, nullable. It is the AND of every
  WKB Levin call's `converged`. The script asserts that it equals the AND of the stored group
  flags.
- **`total_abserr`** and **`total`:** columns, `Float(64)`.

**Work counters, not values** (06b's, 06c's and 06e's classification, kept):
- **The columns:** `WKB_Levin_num_regions`, `WKB_Levin_evaluations`, `WKB_Levin_simple_regions`,
  `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`,
  `WKB_Levin_max_depth`, `numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`.
- **The metadata leaves:** `regions`, `simple_regions` and `evaluations`.
- **Why they are counters.** They count the partition the driver chose and the work it did. None
  of them is the integral or a bound on its error.
  - 06h moves the partition through the declared phase error (§0.3, "What moves": "regions, depth").
  - It changes no driver code, so no counter changes its meaning.
  - They are compared and tabulated, region counts and depth included, but a difference in one is
    not a value difference.

**Not compared:** timings, the label and store bookkeeping. **Everything else is a value**,
including `total_converged`, `total_phase_limited`, and every `converged`, `phase_limited` and
`abserr` leaf.

### Deviations from the prompt

1. **`run_readers.sh` cannot run from the code worktree as written.** It `cd`s to its own tree and
   calls `./venv/bin/python`, which the worktree lacks. 06e's `run_readers_worktree.sh` was used,
   copied unchanged (`cmp` identical, `ec96cf09…`). Its diff against `run_readers.sh`
   (`out/run_readers_worktree.diff`) is byte-identical to 06e's. `STRUCTURALLY REQUIRED`.
2. **The readers start Ray.** Each reader calls `ray.init(address="local",
   _temp_dir=/tmp/ray-sgk)` (`extract_QuadSourceIntegral_data.py` 132–134).
   - **The conflict:** the common brief says only R21's and R22's drivers may start a Ray session.
     Prompt §4.4 and this agent's brief both require the readers, with the lock held through them.
   - **What was done:** they ran under the lock, as 06e's, 06c's and 06b's did. Each reader left no
     Ray process ("Ray processes from this checkout's venv after: 0"), and `ray-processes` was
     empty after them.
   - Recorded so that the orchestrator can confirm the reading. `STRUCTURALLY REQUIRED`.
3. **The tabulation is 06e's `tabulate_qsi_endnodes.py`, adapted** as `tabulate_qsi_groupphase.py`.
   The brief allows this. The diff is `out/tabulate_qsi_groupphase.diff`.
   - **Kept:** the `mode=ro` reader, the physical-key pairing, the field kinds, the `float.hex`
     comparison, the cross-check against the tool, and 06a's stationary-point identification.
   - **Changed:**
     - by `sed`: R19/R20 became R21/R22, the suffixes 19/20 became 21/22 (`total19`, `abserr20`,
       the rule's subscripts, the non-finite labels), and Z1 became H0;
     - by asserted single replacements: the docstring header, "prompt 06e" in H0's paragraph
       (now "06h"), one stray "abserr21+20" (now "+22"), "T19" (now "T21"), the `DEPTH_MAX`
       comment, the usage line, and the counters' justification paragraph (which now cites 06h
       §0.3).
   - `DEPTH_MAX = 20` is unchanged.
   - `IMPLEMENTATION CHOICE`.
4. **How 06a's stationary-point rows are found in the rehearsal store:** 06e's method, unchanged.
   - **The method:** `analyse06c.py`'s `scan06a` rule. That is the 1e-7 records of 06a's census
     checkpoint (`c13685b4…`) with a call whose `n_sign_changes > 0`, asserted to be 720.
     - Each is mapped through 06a's `state/rows.json` (`aa523d28…`) to its physical key
       (k, q, r, z_response), parsed with `float`.
     - A rehearsal row is among them iff its own physical key, read from the store, is equal as
       floats.
   - **The result is 0.** No rehearsal row is any A3 v2 row. Only $k=10^5$ is shared, and none of
     the rehearsal's 96 `z_response` values is among A3 v2's. So H0 scores all 864 rows.
   - **The consequence:** the rehearsal store's own stationary-point calls, if it has any, are
     not identifiable by 06a's scan, and are scored with the rest.
   - `IMPLEMENTATION CHOICE` (the method, 06e's). The prompt does not say how to carry an A3 v2
     row set to R8's scope.
5. **A descriptive table of the analytic oracle was added** (`scripts/qsi_analytic_summary.py`,
   `out/qsi_analytic_rows.csv` and `out/qsi_analytic_summary.txt`).
   - **Why:** H0 scores `total`, which 06h's code cannot reach in this table (Observation 1), so
     the tabulation alone would not show what moved.
   - **What it gives:** per row, `analytic_rad`, $\lvert\Delta\,{\rm analytic\_rad}\rvert$ against
     `analytic.abserr` in each store, and the H0 form on that pair. Per block, `abserr`,
     `regions`, `converged` and `phase_limited`.
   - Not scored, and nothing in H0 depends on it. `IMPLEMENTATION CHOICE`.
6. **A read-only cross-reading against 06e's committed tabulation** (not asked for):
   `out/r21_vs_06e_r20_csv_check.txt`.
   - **R21 against 06e's R20,** the same code: equal on all 864 rows, by the CSVs' `float.hex`
     strings. This covers `total`, `total_abserr`, `total_converged`, regions, depth,
     evaluations, `analytic_rad`, and the analytic regions and abserr. So the "before" reproduces
     across runs at 06e's code.
   - **R22 against 06e's R20:** `total` and `total_abserr` are equal on 864/864, and
     `analytic_rad` on 400/864.
   - Only CSV files were read; no store was opened. `IMPLEMENTATION CHOICE`.
7. **R21's launch check printed a `BrokenPipeError` traceback.** It came from
   `runregistrykit list | head -2` in the check's own command line: `head` closed the pipe once
   the list grew to 37 runs.
   - The header and R21's row had printed before it.
   - Every later check redirects the listing to a file in `out/` instead.
   - `UNINTENDED DRIFT`, in records only.
8. **The registry step of `out/r21_end_check.txt` was mis-written.**
   - It redirected to `/tmp/../$PWD/…`, a path that does not exist. The redirect failed, nothing
     was written anywhere, and `head`/`tail`/`rm` reported the missing file.
   - It was redone in `out/r21_end_check2.txt`, which also compares R21's fingerprint with R19's
     and R8's class by class.
   - `UNINTENDED DRIFT`, in records only.
9. **The first `cat` of the two brief files was truncated in the tool's display.** Both were then
   read in full. `UNINTENDED DRIFT`, no effect.
10. **`black`.** `black --check` would reformat `tabulate_qsi_groupphase.py`,
    `qsi_analytic_summary.py` and the copied `classify_non_qsi.py` (`out/final_check.txt`).
    - All three are left as run, as records, as 06b's, 06c's and 06e's were.
    - The copy must stay byte-identical to 05's.
    - `IMPLEMENTATION CHOICE`.

### Verification performed

**The gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`), 00:18 UTC:
- **Trees:**
  - the main checkout is `28ebaf0`, `handover-remedial`, `status --short` empty;
  - the parent worktree is at `2c325f5` and the code worktree at `c1ec3b2`, both detached, with
    `status --short --ignored` empty;
  - `7172a8f^` is `2c325f5`;
  - `2c325f5..c1ec3b2` holds `7172a8f`, `ac1c6f4` and `c1ec3b2`;
  - outside `docs/` and `prompts/`: `diff --stat ac1c6f4 c1ec3b2` is empty, and `6d81fad..2c325f5`
    changes only `CLAUDE.md`.
- **The registry and the lock:**
  - 36 runs, all finished;
  - the lock is free (`~/.raypoolkit` empty), with no old lock at `~/ScienceData/.rehearsal-lock`;
  - `ray-processes` lists none.
- **The host:**
  - `~/.ray/auth_token` is `-rw-------`;
  - 96 GB free, load 4.7;
  - venv Python 3.13.16, ray 2.59.0, SQLAlchemy 2.0.55;
  - datastorekit `f47b9a2`, raypoolkit `67a75b9`, runregistrykit `0bae3a3`, the same as 06e's.
- **No earlier state:** no 06h store or run.

**R21** (`out/r21_launch_check.txt`, `out/r21_end_check.txt`, `out/r21_end_check2.txt`,
`out/r21_status.json`):
- **Launch:** at 00:18:51Z from the parent worktree (`sh` 67079, driver 67087), then
  `set-pid 67087`.
- **Manifest:** `git_head` `2c325f5`, `git_dirty` false, `script_sha256` `b0135218…`. The driver's
  cwd is the parent worktree.
- **End:** `done`, exit 0, at 00:23:36Z (heartbeat 00:23:30Z, fingerprint taken at 00:23:28Z).
  - The fingerprint is `1b332ffa…`. Its classes, digest and formats equal R19's and R8's.
  - `problems {}`. QuadSourceIntegral has 864 rows.
  - `fingerprint_sidecar` reads "not written … (there is none)", R19's note.
- **Logs:** no Traceback in the launch log, `stdout.log` or `stderr.log`, and no `!! WARNING`. The
  launch log ends "** driver exit status 0".
- **Trees:** both worktrees are clean, with nothing newer than the launch marker.
- **The store:** 5 files, no journal (`state/r21_store_sha256_after_build.txt`). It was unchanged
  at R22's launch (`state/r21_store_sha256_at_r22_launch.txt`).

**R22** (`out/r22_launch_check.txt`, `out/r22_end_check.txt`, `out/r22_status.json`):
- **Launch:** at 00:24:22Z from the code worktree (`sh` 68065, driver 68069), then
  `set-pid 68069`.
- **Manifest:** `git_head` `c1ec3b2`, `git_dirty` false, `script_sha256` `b0135218…`. The driver's
  cwd is the code worktree.
- **End:** `done`, exit 0, at 00:28:57Z (heartbeat 00:28:54Z, fingerprint taken at 00:28:52Z).
  - The fingerprint is `1b332ffa…`. Its classes and digest equal R21's, R19's and R8's.
  - `problems {}`. QuadSourceIntegral has 864 rows.
- **Logs:** no Traceback and no `!! WARNING` in either launch log, R22's `stdout.log` or its
  `stderr.log`.
- **Trees:** both worktrees are clean, with nothing newer than R21's marker.
- **The registry** holds 38 runs, all finished.

**The comparison** (`out/store_value_diff.json`, `out/classify_non_qsi.txt`):
- **The tool:** `tools.store_value_diff` from the code worktree (`7997af85…`, 06e's), which opens
  every file `mode=ro`. It exited 1, because `QuadSourceIntegral` differs.
- **The classifier:** 05's `classify_non_qsi.py`, unchanged (`26690e61…`), exited 0.
- **Outside `QuadSourceIntegral`:**
  - `not_compared: []`;
  - 0 only-in-A, only-in-B, unmatched or unresolvable in every class;
  - value tables not identical: none in every class. Bit-identical value rows: `BackgroundModel`
    1,159, `TkNumericIntegration` 1,478, `TkWKBIntegration` 2,209, `GkNumericIntegration` 35,874,
    `GkWKBIntegration` 147,642, `GkSource` 169,653 and `QuadSource` 2,447: 360,462 in all;
  - rows identical outright in the ten bookkeeping classes and in `GkSourcePolicyData` (291);
  - the other differences are `timing`, `label`, `parents_digest` (291 `GkSource`) and `rho_reuse`
    (24 `GkWKBIntegration` rows);
  - **UNCLASSIFIED: 0.**
- **Against 06e's** `classify_non_qsi.txt`, with digits removed, the output differs only in one
  extra timing line (`BackgroundModel` `min_RHS_time`).
- **The stores:** both unchanged by the comparison (`state/store_sha256_before_diff.txt` equals
  `…_after_diff.txt`).

**The tabulation and H0** (`out/qsi_tabulation.txt`, `out/qsi_rows.csv`):
- 864 rows in each store, the same (k, q, r, z_response) set, and 48 columns.
- Tolerances: (log10 atol, log10 rtol) = (−32, −7) in every row.
- **The tool's report:** 864 rows differ (timings and labels), and its key set equals the stores'.
  The cross-check of `total` against it has 0 mismatches.
- **The stores' structure:** no row's field set differs. 573 rows have a WKB Levin call (3,105
  calls in each store) and 291 have none.
- **Bit-identical by `float.hex`:**
  - `total`: 864/864;
  - `total_abserr`: 864/864;
  - every value field: 205/864, all among the rows with no WKB Levin call;
  - `analytic_rad`: 400/864.
- **The value fields that differ** are only `analytic_rad` (464 rows), `analytic.abserr` (659),
  each block's `abserr` (564–643 rows per block) and `phase_limited` (47–83 rows per block).
- **H0:**
  - 0 stationary-point rows, 864 scored, 0 non-finite;
  - **0 violations**, every $|\Delta T|/(e_{21}+e_{22})$ = 0;
  - the largest $|\Delta T|/e_{21}$ and $|\Delta T|/e_{22}$ are both 0.
- **`total_converged`:** (1, 1) in all 864 rows, and 0 WKB Levin calls unconverged in either store.
- **`total_phase_limited`:** (1, 1) in 195 rows and (0, 0) in 669. 406 phase-limited calls in each
  store.
- **Regions:** `WKB_Levin_num_regions` sums to 31,216 in both, and evaluations to 246,623 in both.
  Regions are equal in 573/573 rows.
- **Maximum depth:** the same distribution in both, equal in every row. The deepest is 19, in one
  row. No row reaches 20.

**The analytic oracle, descriptive** (`out/qsi_analytic_summary.txt`,
`out/qsi_analytic_rows.csv`):
- **Scope:** 659 rows have a three-Bessel Levin block, 10,072 blocks in all (5,272 at order 1/2+b
  and 4,800 at 5/2+b).
- **`analytic_rad`:**
  - moved in 464 rows;
  - $|\Delta|/(e_{21}+e_{22})$ (with $e$ the row's `analytic.abserr`): median
    $9.1\times10^{-9}$, 99th percentile $1.15\times10^{-5}$, maximum $2.485\times10^{-5}$;
  - the maximum is at $k=3.16\times10^5$, $q=10^5$, $r=3.16\times10^5$, $z=1.36\times10^{10}$:
    `-0x1.60509ba2ceb16p-39` → `-0x1.60509ba2ceb18p-39`, abserr `0x1.3638d1213ffcfp-76` and
    `0x1.3e961353b521dp-76`;
  - **0 rows** have $|\Delta| > e_{21}+e_{22}$;
  - the relative movement $|\Delta|/|{\rm analytic\_rad}|$ is at most $1.03\times10^{-11}$, with
    median $6.2\times10^{-16}$.
- **`analytic.abserr`,** R22/R21 by row: median 1, minimum 0.258, maximum 3.07, 1st–99th percentile
  0.891–1.138. It rose in 456 rows and fell in 203.
- **`analytic.converged`:** (True, True) in all 864 rows.
- **Per block:**
  - `abserr` R22/R21: median 1, 1st–99th percentile 0.78–1.68, extremes $5.9\times10^{-7}$ and
    $1.15\times10^5$. It is bit-identical in 428 blocks;
  - converged in all 10,072 blocks under both codes;
  - `phase_limited`: T→T 3,312, F→F 5,656, F→T 584 and T→F 520;
  - regions fell from 28,785 to 24,428 (−15.1%): 538 blocks gained regions, 2,456 lost them and
    7,078 kept the same count;
  - the largest per-block count fell from 14 to 11. Blocks with more than 8 regions went from 370
    to 7.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- They ran from the code worktree with the lock held, on R22.
- Six of six exited 0 (walls 24, 43, 18, 15, 13 and 15 s).
- Each printed "store files: identical before and after (SHA-256, mtime, size); no journal".
- R22's hashes before and after the readers are identical (`state/r22_store_sha256_*_reads.txt`).
- No Traceback in any reader log. Each log has the "shards (=4) … (=10)" warning, as 06e's did.

**The lock** (`state/lock.log`, every command exit 0):
1. `ray-processes`: none;
2. `acquire`;
3. `set-pid 67087`;
4. `ray-processes` after R21: none;
5. `set-pid 68069`;
6. `ray-processes` after R22: none;
7. `ray-processes` before the release: none;
8. `release` ("released the rehearsal lock");
9. `status` ("lock: free … does not exist", "Ray processes of /tmp/ray-sgk: none");
10. `ray-processes`: none.

**Final** (`out/final_check.txt`):
- both stores unchanged since the comparison (`state/stores_sha256_final.txt`);
- the main checkout `28ebaf0` is clean, and both worktrees are clean with nothing newer than R21's
  launch marker;
- `~/.raypoolkit` is empty;
- 38 runs, all finished;
- no `__pycache__` in the records.

### Observations not acted on

1. **H0 cannot detect 06h in the rehearsal store, and the same may hold for §5's H3.**
   - **Why:** `total` is `numeric_quad + WKB_Levin`. Neither uses `BesselPhaseGroup`, which 06h
     changes.
   - **What does move:** 06h's code reaches `QuadSourceIntegral` only through `_three_bessel_Levin`,
     the analytic oracle (`analytic_rad`, `analytic.*`).
   - **The consequence:**
     - `total`, `total_abserr`, `total_converged` and every WKB Levin counter are bit-identical
       in all 864 rows, so H0 holds with $|\Delta T| = 0$, not as a test of 06h;
     - the brief's "No row is expected to be bit-identical" and "Every `QuadSourceIntegral` value
       is expected to move" are not borne out by `total`;
     - they are borne out by the oracle (464 rows' `analytic_rad`, 659 rows' `analytic.abserr`);
     - prompt §0.3's "Every `QuadSourceIntegral` row can move" is true only of those columns.
   - **In H0's form,** the oracle's movement is inside its declared error by more than four orders at worst
     (deviation 5).
   - **For §5, not mine to score:**
     - H3 ("the rows agree", on `total`) will be decided the same way, unless the census records
       `total` differently;
     - H4 ("no Levin call that 06e reported converged is unconverged") is a test of 06h only if
       the census's Levin calls include the analytic blocks;
     - the descriptive "per analytic-block call" item of §5 suggests they do.
   - Recorded for the orchestrator. Neither reading of H0 changes its verdict here.
2. **The analytic blocks' declared errors move in both directions, and widely per block:**
   $5.9\times10^{-7}$ to $1.15\times10^5$ in R22/R21.
   - Per row the spread is 0.26–3.07, median 1.
   - This fits §0.2's "the declaration should rise by orders in the tail while the declared error
     of a call barely moves" at the row level.
   - It is not examined further here; the census and the phase measurement are §5's.
3. **The blocks' regions fell 15.1%** (28,785 → 24,428), and no block reached more than 11 regions.
   This is the same direction as the code log's draws (68,772 → 61,126). No block's convergence
   changed, and none is unconverged.
4. **`runregistrykit list | head` raises `BrokenPipeError`** (deviation 7). 06e's `final_check`
   saw the same. It is a command-line nicety of the pinned package, with no effect on any record.
   Not opened (`issues.md`).

### State handed on

- **The lock is released.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing, and
  `~/.raypoolkit` is empty. The six cores are free for §5.
- **The stores are kept:**
  - `var/datastores/levin-phase-floor-06h-r21{,-shard000[0-3]}.sqlite`, R21 at `2c325f5`;
  - `var/datastores/levin-phase-floor-06h-r22{,-shard000[0-3]}.sqlite`, R22 at `c1ec3b2`.
  - Both are `1b332ffa…`, with their hashes in `state/stores_sha256_final.txt`.
- **The runs:**
  - `levin-phase-floor-06h-r21-parent-20261101T001852` and
    `levin-phase-floor-06h-r22-groupphase-20261101T002423`, both `done`;
  - launch logs `var/runs/levin-phase-floor-06h-r21.launch.log` and `…-r22.launch.log`.
- **The main checkout and both worktrees** are clean and unmoved.
- **Kept on the Mac Studio, not for copying back:** `out/store_value_diff.json` (41.0 MB) and
  `extract/` (172 MB). See `KEPT_ON_MAC_STUDIO.txt`.
