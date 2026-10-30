## §4 — rehearsals R17 and R18 on the Mac Studio

**Prompt:** [`06c-global-error-control.md`](../06c-global-error-control.md) §4, under the
orchestrator's Part B dispatch (`e5764a6`) and its addenda B.5 and B.6. The code measured is 06c's
(`ad9a5f4`) and 06d's (`3d70db9`) together.
**Host:** mps026298, the Mac Studio.
- **R17** ran in the worktree `SecondaryGWKit-lpf06c-fef401c`, detached at `fef401c`. That is 06c's
  code commit `ad9a5f4`'s parent, and its code is 06b's: `git diff --stat 92bc62f fef401c` outside
  `docs/` and `prompts/` shows only `CLAUDE.md`.
- **R18**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06c-e5764a6`,
  detached at `e5764a6`. Its code is `3d70db9`'s, which is 06c's plus 06d's:
  `git diff --stat 3d70db9 e5764a6` outside `docs/` and `prompts/` shows only `CLAUDE.md`.
- **The main checkout** stayed at `28ebaf0` and clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in three turns:
1. the gate, the lock and R17's launch;
2. R17's checks and R18's launch;
3. R18's checks, the comparison, the tabulation and Y1, the readers, the release and these records.

**Date:** 2026-10-29, 12:36–12:57 UTC.
**Commit:** none. The records are committed on the laptop, in 06c's records commit.
**Result:** COMPLETE, **Y1 FAILS**. No §7 stop condition fired. Prompt §4.3 says a Y1 violation
is "a finding for the user, not by itself a stop". **It bears on B.6:** the header issues close
"if 06c's §4 holds and X0–X4 hold; otherwise ask the user". §4's Y1 does not hold.
- **The fingerprints.** R17 and R18 both ended `done`, exit 0, with fingerprint `1b332ffa…`. Each
  equals R8's class by class, and R18's `classes` object equals R17's.
- **Outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` or the residual-memo counters
    (`rho_reuse`);
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`,** all 864 rows are tabulated:
  - `total` is bit-identical in 295 rows: the 291 with no WKB Levin call, and 4 that have one;
  - every value field is bit-identical in 205 rows, all of them without a Levin call. The other 86
    such rows differ only in the analytic oracle's metadata, which uses Levin.
- **Y1 fails.** 30 of the 838 rows with finite `total_abserr` in both stores violate the rule.
  - **In 12, the excess is large.** $|\Delta T|$ is 336–16,763 times $e_{17}+e_{18}$. These are all
    equilateral rows ($k=q=r$ at 1e5 (5), 3.16e5 (6) and 1e6 (1); $z$ 250–6,905).
  - **The cause in each of the 12** is the `G+q+r` phase group of the all-oscillatory
    sub-interval. Its value moves by 0.03–1.0 % between the codes, while each code declares an
    error of at most about $2\times10^{-7}$ of it: $|\Delta V|/(e_{17}+e_{18})$ is $4\times10^3$ to $3.4\times10^5$
    on that call. At least one code is dishonest there; which one, this comparison cannot say.
  - **The other 18** exceed by 1.007–20.4×.
  - **Infinite in R17:** 26 rows have `total_abserr = inf` in R17 and finite in R18. They are
    counted separately: all are R17-unconverged and R18-converged, with
    $|\Delta T|/e_{18} \le 0.087$.
- **`total_converged`** went from 516 to 863 of 864 rows.
- **The readers** exited 0, six of six, and R18 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6c, "Global error control".

### What shipped

No code. The records, under the main checkout's `var/levin-phase-floor-06c/rehearsal/`, for the
laptop to commit as `docs/levin-phase-floor/06c/rehearsal/`:
- `log.md` (this), `verification.md` and `issues.md`;
- **`scripts/`:**
  - `rec.sh`, 06b's with the record path changed;
  - `classify_non_qsi.py`, 05's, copied unchanged (`26690e61…`, byte-identical to 05's and to
    06b's copy);
  - `run_readers_worktree.sh`, 06b's (and 06's), copied unchanged (`ec96cf09…`);
  - **new:** `tabulate_qsi_global.py` (`3046d5c0…`), the §4.3 tabulation and Y1, written from 06b's
    `tabulate_qsi_phase_limited.py`;
  - **new:** `y1_violations_groups.py` (`0ac64339…`), which localises each Y1 violation to calls;
- **`out/`:**
  - `commands.log`, every command;
  - the gate checks `00`–`02`;
  - each build's launch and end checks, and both `status.json`;
  - `store_value_diff.json` (`57313b01…`, 42.8 MB) and `classify_non_qsi.txt`;
  - `qsi_rows.csv`, every row (`6e2f3a8c…`), `qsi_tabulation.txt` (`2cd75189…`) and
    `qsi_extra_summary.txt`;
  - `y1_violations_groups.txt` (`c03ab0d0…`) and `y1_violations_worst_calls.txt`;
  - the R16(06b)-against-R17 determinism check: `store_value_diff_r16_06b_vs_r17.json` (40.2 MB),
    `classify_non_qsi_r16_06b_vs_r17.txt`, `r16_06b_vs_r17_qsi_summary.txt` and
    `r16_06b_vs_r17_labels.txt`;
  - `readers.txt`, `readers_check.txt` (with its failed first form, `readers_check.attempt1.txt`),
    `run_readers_worktree.diff`, `final_check.txt` and `black_check.txt`;
- **`state/`:** `lock.log`, both launch lines, the markers, `progress.md`, and every hash file of
  the stores (and of 06b's R16, which was only read).
- **Not for copying back:** the readers' `extract/` (177 MB). The two `store_value_diff` JSONs are
  large, and are the orchestrator's to keep or drop.
- **The stores are kept:** `var/datastores/levin-phase-floor-06c-r17*` and `…-r18*`, five files
  each.

### Predictions and verdicts

| Rule (prompt 06c §4, as written) | Scored on | Verdict |
|---|---|---|
| §4.2: R17 and R18 each `done`; **both fingerprints must equal `1b332ffa…`** | R17 `done`, exit 0, `1b332ffa…`; R18 `done`, exit 0, `1b332ffa…`. Classes equal R8's in both, and R18's equal R17's. `problems {}` | **holds** |
| §4.3: outside `QuadSourceIntegral` every value table is identical | 7 tables, 360,462 value rows, all identical; `value tables not identical: none` in every class | **holds** |
| §4.3: every other difference outside it is of the four volatile kinds | timing, label, `parents_digest`, `rho_reuse` (24 `GkWKBIntegration` rows); **0 unclassified** | **holds** |
| §4.3: no row in one store only | 0 only-in-A, 0 only-in-B, 0 unmatched, in every class | **holds** |
| §4.3: inside `QuadSourceIntegral` every row is tabulated (the four quantities named below) | 864 of 864 rows, in `out/qsi_rows.csv` | **done** |
| **§4.3 Y1:** every row with finite `total_abserr` in both stores has $\lvert T_{18}-T_{17}\rvert \le e_{17}+e_{18}$; rows with an infinite `total_abserr` in either are counted separately; holds if no row violates it | 838 rows finite in both; **30 violate** (ratios 1.007–16,763). 26 rows are infinite in R17 (none in R18), counted separately | **fails** (a finding for the user, not by itself a stop) |
| §4.4: every reader exits 0, and the store is byte-identical before and after | 6 of 6 exit 0; the 5 files are identical by SHA-256, size and mtime; no journal | **holds** |
| §4.5: lock released; `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing | released, exit 0; `status` free; `ray-processes` empty, before and after the release | **holds** |
| §4.6: the stores are kept | both stores in place, unchanged since the comparison | **holds** |

**§6 item 2 (acceptance):** "R17 and R18 both `done` with `1b332ffa…`; outside
`QuadSourceIntegral`, every value table identical; inside it every row tabulated and Y1 scored;
the readers exit 0; the lock released". Every clause is met, and **Y1 is scored as failing**.

### Where the region counts and maximum depths are stored (§4.3)

From `ComputeTargets/QuadSourceIntegral.py` (`evaluate_QuadSource_integral` 670–838,
`_aggregate_LevinData` 533–549, `phase_group_Levin_integral` 940–971, `_three_bessel_Levin`
1290–1330) and `ObjectFactories/QuadSourceIntegral.py` (columns 249–288), at `e5764a6`. The same
holds at `fef401c`: the module's AST without comments and docstrings is equal at the two trees
(`out/02-gate-launcher.txt`). Everything is in each shard's table `QuadSourceIntegral`:

- **Region count, the call total:** column `WKB_Levin_num_regions`. SQLAlchemy `Integer`, SQLite
  integer, nullable. It is the **sum**, over every `adaptive_levin_sincos` call of the row's WKB
  Levin sub-intervals (every phase group of every sub-interval), of the call's `num_regions`. It
  is NULL when the row made no WKB Levin call (291 rows in both stores).
- **Region count, per call:** column `metadata`, a JSON string written by `json.dumps`, at
  `WKB_Levin.subintervals[i].groups[j].regions` (a JSON int). The script asserts that their sum
  equals `WKB_Levin_num_regions` on every row of both stores.
- **Maximum depth:** column `WKB_Levin_max_depth`. `Integer`, nullable. It is the **maximum** over
  the same calls of the call's `max_depth`. **No per-call depth is stored.**
- **The analytic oracle's region counts:** `metadata` at
  `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions` (a JSON int, `len(regions)` of each
  three-Bessel Levin block). These feed `analytic_rad`, not `total`. **No depth is stored for
  them.**
- **`total_converged`:** a column, `Boolean`, SQLite integer 0/1, nullable. It is the AND of every
  WKB Levin call's `converged`. The script asserts that it equals the AND of the stored group
  flags.
- **`total_abserr`:** a column, `Float(64)`. It is the linear sum of numeric_quad's and every WKB
  Levin call's `abserr`. `total` is a column, `Float(64)`.
- **Not stored:** 06c's `num_untrusted_regions`, and the trust check's per-leaf fields (`delta1`,
  `delta2`, `convergence_ratio`, `trusted`, `declared_resolution`). `QuadSourceIntegral` persists
  none of them, so no rehearsal store can show the trust check.

**Work counters, not values** (06b's classification, kept):
- **the columns:** `WKB_Levin_num_regions`, `WKB_Levin_evaluations`, `WKB_Levin_simple_regions`,
  `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`,
  `WKB_Levin_max_depth`, `numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`;
- **the metadata leaves:** `regions`, `simple_regions` and `evaluations`.

**Why they are counters.** They count the partition the driver chose and the work it did. None of
them is the integral or a bound on its error. 06c changes the partition by construction (§0.2)
without changing what any counter means (§0.3). They are compared and tabulated, region counts
and depth included, but a difference in one is not a value difference.
- **Timings:** columns ending `compute_time`, `RHS_time` or `elapsed`, and the leaves `elapsed`
  and `compute_time`.
- **The label:** the `label` column.
- **Store bookkeeping:** `serial`, every `*_serial`, and `timestamp`.

These three kinds are not compared. **Everything else is a value**, including `total_converged`,
`total_phase_limited` and every `converged` and `phase_limited` leaf.

### Deviations from the prompt

1. **R18's launcher has a different SHA-256 from R17's.** It is `b0135218…` in the code worktree,
   against `5b984c75…` in the parent worktree, the main checkout, R15 and R16. 06c changed the
   docstring of `RunDrivers/quadsource_atol_sweep.py` (and a comment in
   `ComputeTargets/QuadSourceIntegral.py`). The ASTs without docstrings are equal at the two trees
   for both files. Each build ran its own tree's launcher, which is what running "at the code
   commit" means. `STRUCTURALLY REQUIRED`.
2. **`run_readers.sh` cannot run from the code worktree as written.** It `cd`s to its own tree and
   calls `./venv/bin/python`, which the worktree lacks. 06b's `run_readers_worktree.sh` was used,
   copied unchanged (`cmp` identical, `ec96cf09…`). Its diff against `run_readers.sh`
   (`out/run_readers_worktree.diff`) is byte-identical to 06b's. `STRUCTURALLY REQUIRED`.
3. **The readers start Ray.** Each reader calls `ray.init(address="local",
   _temp_dir=/tmp/ray-sgk)`.
   - **The conflict:** the common brief says "No Ray session may be started by anything but R17's
     and R18's own drivers". Prompt §4.4 and this agent's brief both require the readers, with the
     lock "held through the comparison and the readers".
   - **What was done:** they ran under the lock, as 06b's did, and each left no Ray process.
   - Recorded so the orchestrator can confirm the reading. `STRUCTURALLY REQUIRED`.
4. **The `QuadSourceIntegral` tabulation is a new script,** `tabulate_qsi_global.py`, written from
   06b's `tabulate_qsi_phase_limited.py`.
   - **Kept:** its mode=ro reader, physical-key pairing, field kinds, `float.hex` comparison and
     cross-check against the tool.
   - **Replaced:** its 06b §4.3 rule, by 06c's tabulation and Y1.
   - Brief: "by a new script of yours". `IMPLEMENTATION CHOICE`.
5. **A descriptive localisation of Y1's violations,** `y1_violations_groups.py`. It prints each
   violating row's calls side by side from `metadata`. The prompt does not ask for it. It is
   read-only and scores nothing; it is what shows that the 12 large violations are each one
   `G+q+r` call. `IMPLEMENTATION CHOICE`.
6. **A read-only determinism check of the "before":** `tools.store_value_diff` of 06b's R16
   (`levin-phase-floor-06b-r16`, 06b's code, which is R17's) against R17.
   - **Not asked for.** It shows that R17 reproduces 06b's R16 in every value, so the comparison's
     "before" is not a run-to-run artefact.
   - **R16's store** was hashed before and after, and is unchanged
     (`state/r16_06b_sha256_before_read.txt`, `out/final_check.txt`). Nothing was written to it.
   - **The classifier** reports 5,148 labels `UNCLASSIFIED` there. 05's label rule requires the
     labels to be equal once the time stamp is removed, and the job names differ (`…-06b` against
     `…-06c`). An inline check (`out/r16_06b_vs_r17_labels.txt`) shows that all 6,012 label
     differences vanish once the stamp and the job name are removed. In `QuadSourceIntegral`,
     only timings, labels and the metadata leaves `elapsed`/`compute_time` differ.
   - `IMPLEMENTATION CHOICE`.
7. **The first form of the readers check failed to parse.** It used bash process substitution
   under `sh`, with stray text. It was a syntax error with exit 2, and nothing ran. Its record is
   kept as `out/readers_check.attempt1.txt`; the second form ran. `UNINTENDED DRIFT`.
8. **The post-reader hash file has two more lines than the pre-reader one.**
   `state/r18_store_sha256_after_reads.txt` carries a trailing `journal/wal/shm: none`, which the
   `grep r18` extraction of the pre-reader file dropped. So the `diff` in `out/readers_check.txt`
   prints those two lines. The 5 SHA-256 lines and the 5 size/mtime lines are identical (checked
   and appended there). `UNINTENDED DRIFT`, in a record only.
9. **`black`.** `black --check` would reformat `tabulate_qsi_global.py`, `y1_violations_groups.py`
   and the copied `classify_non_qsi.py` (`out/black_check.txt`). They are left as run, as records,
   as 06b's were; the copy must stay byte-identical to 05's. `IMPLEMENTATION CHOICE`.

### Verification performed

**The gate,** read-only (`out/00-gate-trees.txt`, `01-gate-registry-lock.txt`,
`02-gate-launcher.txt`):
- the main checkout is at `28ebaf0`, clean;
- both worktrees are clean (`--ignored` included), at `fef401c` and `e5764a6`;
- `git worktree list` shows the three trees;
- `ad9a5f4^` is `fef401c`;
- the code-path diffs are as the brief states;
- the registry holds 28 runs, all finished;
- the lock is free, and there are no Ray processes;
- `~/.ray/auth_token` is mode 600, and 108 GiB are free;
- the venv is 3.13.16 with the pinned packages;
- no 06c store or run existed (the gate file's exit 2 is that `ls`, intended).

**R17** (`out/r17_launch_check.txt`, `out/r17_end_check.txt`, `out/r17_status.json`):
- launched 12:37:19Z from the parent worktree (`sh` 96787, driver 96791), with `set-pid 96791`;
- run `levin-phase-floor-06c-r17-parent-20261029T123720`; manifest `git_head fef401c…`,
  `git_dirty` false, cwd the parent worktree;
- `done`, exit 0, last heartbeat 12:40:09Z;
- digest `1b332ffa…`, equal to R8's and to the literal; classes equal to R8's; `problems {}`;
- no Traceback, no `!! WARNING`, and `stderr.log` empty;
- the worktrees unwritten;
- `ray-processes` empty after it.

**R18** (`out/r18_launch_check.txt`, `out/r18_end_check.txt`, `out/r18_status.json`):
- launched 12:40:57Z from the code worktree (`sh` 97466, driver 97470), with `set-pid 97470`;
- run `levin-phase-floor-06c-r18-global-20261029T124058`; manifest `git_head e5764a6…`,
  `git_dirty` false;
- `done`, exit 0, last heartbeat 12:45:49Z;
- digest `1b332ffa…`; classes equal to R8's and to R17's; `problems {}`;
- no Traceback, no `!! WARNING`;
- `ray-processes` empty after it.

**The comparison** ran from the code worktree, in 16 s:
`python -B -m tools.store_value_diff R17 R18 --json`, exit 1 because `QuadSourceIntegral` differs.
- `classify_non_qsi.py` (`out/classify_non_qsi.txt`) reports 0 unclassified.
- `tabulate_qsi_global.py` (`out/qsi_tabulation.txt`, exit 0) checks its own reading:
  - 864 rows in each store, the same key set;
  - 48 columns, equal;
  - tolerances (atol 1e-32, rtol 1e-7) equal row by row;
  - the tool's 864 differing rows equal the stores' key set;
  - **0** `total` mismatches against the tool;
  - the metadata field sets are equal on every row;
  - 3,105 Levin calls in each store, the same per row;
  - `WKB_Levin_num_regions` equals the sum of the group counts, and `total_converged` equals the
    AND of the group flags, on every row of both.

**The stores** are identical before and after the comparison (`state/store_sha256_before_diff.txt`
= `…_after_diff.txt`). R17 is unchanged since its build (`r17_store_sha256_after_build.txt`), and
the final hashes equal the pre-comparison ones (`state/stores_sha256_final.txt`). Every reader
opened them `mode=ro`.

**The determinism check** (deviation 6): R16 (06b) against R17. Every value table outside
`QuadSourceIntegral` is identical, and inside it only timings, labels and `elapsed`/`compute_time`
leaves differ. So the "before" is reproducible across runs at 06b's code.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- six of six exit 0; walls 24, 48, 19, 15, 13 and 17 s;
- 432, 2,436, 156, 27, 27 and 71 files;
- each leaves the store identical (SHA-256, mtime, size) with no journal, and 0 Ray processes
  after;
- 0 Tracebacks;
- six `!! WARNING: number of shards read from database (=4) does not match specified number of
  shards (=10)`, as in 06b.

**The lock** (`state/lock.log`): ray-processes (empty), acquire, set-pid 96791, ray-processes
(empty), set-pid 97470, ray-processes (empty), ray-processes (empty), release, status (free, none),
ray-processes (empty). Each was exit 0.

**At the end** (`out/final_check.txt`):
- the registry holds 30 runs, all finished;
- the main checkout is at `28ebaf0`, clean;
- both worktrees are clean, with nothing newer than the R17 marker;
- `~/.raypoolkit` is empty;
- 06b's R16 is unchanged.

### Y1 in detail

The rule as written is $\lvert T_{18}-T_{17}\rvert \le e_{17}+e_{18}$, in float64, on rows finite
in both (`out/qsi_tabulation.txt`, `out/y1_violations_groups.txt`,
`out/y1_violations_worst_calls.txt`).
- **Finite in both:** 838 rows. Of these, 808 hold and 30 violate.
- **Infinite in R17, finite in R18:** 26 rows. All are at $z \ge 1.33\times10^9$; R17 is
  unconverged and R18 converged in all; $|\Delta T|/e_{18}$ has a median of 0.0017 and a maximum
  of 0.087. None is infinite in R18. This is §0.2's "an infinite floor is chased".
- **$|\Delta T|/(e_{17}+e_{18})$ over the finite rows:** median 0.014, p90 0.31, p99 7,090,
  max 16,763.
- **The 30 violations by row type:** equilateral 1e5³ (6), 3.16e5³ (9) and 1e6³ (2); the rest are
  squeezed or near-squeezed. By `total_converged` (R17→R18): 19 went 0→1 and 11 stayed 1→1.
- **The 12 large violations (336–16,763×)** each sit in one call: the `G+q+r` group of the
  all-oscillatory sub-interval `[True, True, True]`.
  - **The movement:** its value moves by 0.03–1.0 % of itself (0.64–1.04 % in eleven of them),
    against declared errors of about $2\times10^{-9}$–$2\times10^{-7}$ of it in each code. On that call $|\Delta V|/(e_{17}+e_{18})$ runs from
    $4.3\times10^3$ to $3.4\times10^5$.
  - **Regions R17 → R18:** 12→2, 13→7, 13→2, 13→8, 13→10, 13→3, 13→4, 2→21, 12→2, 12→2, 1→66 and
    18→7.
  - **Converged:** both codes call the call converged in 11 of the 12; the twelfth is R17-unconverged.
  - **The direction is mixed,** so the comparison cannot say which code is dishonest. R18 uses fewer
    regions in ten (consistent with `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`
    or with an error common to a leaf and its descendants,
    `[06b-moved-calls-declare-less-than-their-error]`), and far more in two.
  - **Example,** $k=q=r=10^5$, $z=5236.553$: total17 `-0x1.5ff6eb7403dbcp-55`, total18
    `-0x1.60d489de69404p-55`, abserr17 `0x1.1eff05d8001dfp-79`, abserr18 `0x1.21b46d538f125p-78`.
    Every violating row is listed in `float.hex` in `out/qsi_tabulation.txt`.
- **The other 18 (1.007–20.4×)** are mostly the `G` group of the `[True, False, False]`
  sub-interval, or a `G±r` group. R17 is often unconverged there, and R18 uses more regions.

### Observations not acted on

1. **Y1's 12 large violations need a reference.** One of the two codes misdeclares the `G+q+r`
   call of equilateral rows by up to $3.4\times10^5$. An independent evaluation of those 12 calls
   (for example §5's W2 reference: Levin forced to $2^9$ and $2^{10}$ equal pieces) would say
   which. That is outside §4; proposed in `issues.md`.
2. **`QuadSourceIntegral` stores neither `num_untrusted_regions` nor the trust check's per-leaf
   fields,** and no per-call `max_depth`. So the rehearsal stores cannot show the trust check, and
   depth is visible only as the row maximum. §5's census records them per call. (06b's N2, on the
   analytic blocks' phase-limited ORs, is the same kind of gap.)
3. **One row reaches depth 20 under R18:** $k=10^6$, $q=3.16\times10^5$, $r=10^6$, $z=3971.4$. It
   has 31 → 394 regions, depth 6 → 20, unconverged in both, with 2 calls unconverged and 2
   phase-limited in R18. Y1 holds there (0.033). It is the only R18-unconverged row.
4. **Phase-limited calls:** R17 has 0 (06b's code had none on this scope, as R16 showed), and R18
   has 474 calls on 231 rows, plus 3,506 of 10,072 analytic blocks. This is 06c's meaning of the
   flag ("a leaf the loop chose while the budget was unmet, and found floor-limited"), and is not
   a change of the counters' meaning.
5. **Cost** (descriptive; README §5 rule 4):
   - Levin regions: 13,577 → 34,892 (×2.57);
   - Levin evaluations: 30,231 → 276,031 (×9.1);
   - analytic-block regions: 16,405 → 35,996;
   - row max depth: 279 rows deeper, 64 shallower and 230 equal;
   - build wall: about 2 m 50 s (R17) against 4 m 51 s (R18).
6. **`total_abserr`** moved in 573 of 573 rows with a Levin call. $e_{18}/e_{17}$ (both finite)
   has a median of 1.05 and runs from 0.035 to 5.9.

### State handed on

- **The stores** `var/datastores/levin-phase-floor-06c-r17*` and `…-r18*` are kept, unchanged
  since their builds. The final SHA-256 values are in `state/stores_sha256_final.txt`:
  - R17's primary `b9f56fa3…`;
  - R18's primary `150612c5…`.
- **The runs** `levin-phase-floor-06c-r17-parent-20261029T123720` and
  `…-r18-global-20261029T124058` are both `done`. Their launch logs are
  `var/runs/levin-phase-floor-06c-r1{7,8}.launch.log`.
- **The lock** is released. `/tmp/ray-sgk` has no Ray process. The census agent may now take the
  lock if it needs it; its census is Ray-free.
- **For the orchestrator:**
  - Y1 fails. Under B.6, the header issues do not close on this record without the user.
  - The 12 `G+q+r` calls are the place to put a reference.
  - Deviation 3 (the readers start Ray, under the lock) asks for a confirming reading.
