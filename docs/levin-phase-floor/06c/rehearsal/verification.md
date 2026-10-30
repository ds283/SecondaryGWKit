### 6c.x Rehearsals R17 and R18, and the value comparison (prompt 06c §4)

**Prompt 06c §4, run 2026-10-29, 12:36–12:57 UTC, on the Mac Studio** (mps026298, Darwin 27.0.0).
The code measured is 06c's (`ad9a5f4`) and 06d's (`3d70db9`) together (addendum B.6).
- **The trees.** Both builds ran from git worktrees, so the main checkout did not move:
  - **R17** ran in `SecondaryGWKit-lpf06c-fef401c`, detached at `fef401c`. That is the parent of
    06c's code commit `ad9a5f4`, and its code is 06b's: `92bc62f..fef401c` touches only
    `CLAUDE.md` outside `docs/` and `prompts/`.
  - **R18** ran in `SecondaryGWKit-lpf06c-e5764a6`, detached at Part B's dispatch SHA `e5764a6`.
    Its code is `3d70db9`'s: `3d70db9..e5764a6` touches only `CLAUDE.md` outside `docs/` and
    `prompts/`.
  - **On code paths,** `fef401c..e5764a6` changes:
    - `AdaptiveLevin/levin_quadrature.py` and the AdaptiveLevin and LiouvilleGreen tests;
    - comments only in `ComputeTargets/QuadSourceIntegral.py`, and the docstring only in
      `RunDrivers/quadsource_atol_sweep.py`. For both, the ASTs without docstrings are equal.
- **The main checkout** stayed at `28ebaf0` and clean. It holds the venv, the stores, the registry
  and these records.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0 and `runregistrykit` 0.2.0.
- **The load** was about 5 on 10 cores, from desktop processes.

**Records** are in [`06c/rehearsal/`](06c/rehearsal/): `out/`, `scripts/`, `state/`, the agent's
`log.md`, this section as first written, and `issues.md`.
- **Commands.** Every command is in [`out/commands.log`](06c/rehearsal/out/commands.log).
- **Scripts, by SHA-256** (from [`out/final_check.txt`](06c/rehearsal/out/final_check.txt)):
  - `classify_non_qsi.py` is `26690e61…`: 05's, copied unchanged, and byte-identical to 05's and
    06b's copies;
  - `run_readers_worktree.sh` is `ec96cf09…`, 06b's (06's), copied unchanged;
  - `run_readers.sh` is `7c63a5ba…`;
  - the launcher is `5b984c75…` in the parent worktree and the main checkout, and `b0135218…` in
    the code worktree (docstring only);
  - `tools/store_value_diff.py` is `7997af85…` in both worktrees;
  - the new `tabulate_qsi_global.py` is `3046d5c0…`, and the new `y1_violations_groups.py` is
    `0ac64339…`.
- **Not copied back:** the readers' `extract/` (177 MB).

#### The two builds

| | R17 (before) | R18 (after) |
|---|---|---|
| tree | worktree at `fef401c` (parent of `ad9a5f4`; code 06b's) | worktree at `e5764a6` (code `ad9a5f4` + `3d70db9`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `-B -u` and `PYTHONDONTWRITEBYTECODE=1`; `nohup sh -c`, detached | the same, in its own worktree |
| run | `levin-phase-floor-06c-r17-parent-20261029T123720` | `levin-phase-floor-06c-r18-global-20261029T124058` |
| `sh` / driver pid | 96787 / 96791 | 97466 / 97470 |
| wall | 12:37:19 launch → 12:40:09 last heartbeat | 12:40:57 → 12:45:49 |
| manifest | `git_head` `fef401c2…`, `git_dirty` false, cwd the parent worktree, script SHA `5b984c75…` | `git_head` `e5764a67…`, `git_dirty` false, cwd the code worktree, script SHA `b0135218…` |
| cwd of the driver (`lsof`) | the parent worktree | the code worktree |
| end | `done`, exit 0; no Traceback; `stderr.log` empty; no `!! WARNING` | the same |
| store | `var/datastores/levin-phase-floor-06c-r17.sqlite` + 4 shards | `var/datastores/levin-phase-floor-06c-r18.sqlite` + 4 shards |

**The scope** was R8's rehearsal scope, launched as R15 and R16 were:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- `--job-name levin-phase-floor-06c`, `--register r17-parent` / `r18-global`, and
  `--campaign levin-phase-floor --prompt 06c`.

**The worktrees were not written.** After each build, after the readers and at the end, each
worktree's `git status --short --ignored` was empty, and `find -newer` the pre-R17 marker listed
nothing in either.

#### The lock

[`state/lock.log`](06c/rehearsal/state/lock.log). Each lock command was a shell command of its
own, with its output appended by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |
| 2 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 06c section 4, rehearsals R17 … and R18 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 3 | `set-pid … 96791` (R17) | recorded | 0 |
| 4 | `ray-processes` (after R17) | nothing | 0 |
| 5 | `set-pid … 97470` (R18) | recorded | 0 |
| 6 | `ray-processes` (after R18) | nothing | 0 |
| 7 | `ray-processes` (after the readers) | nothing | 0 |
| 8 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 9 | `status --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | free; no Ray processes | 0 |
| 10 | `ray-processes` | nothing | 0 |

#### The fingerprints (prompt §4.2)

| | digest | problems | classes |
|---|---|---|---|
| **R17** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | equal to R8's |
| **R18** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | equal to R8's and R17's |

R17's was taken at `fef401c` and R18's at `e5764a6`, both with `git_dirty` false
([`out/r17_end_check.txt`](06c/rehearsal/out/r17_end_check.txt),
[`out/r18_end_check.txt`](06c/rehearsal/out/r18_end_check.txt)). **Holds.**

#### The value comparison (prompt §4.3)

`python -B -m tools.store_value_diff R17 R18 --json`, run from the code worktree: 16 s, exit 1
(`QuadSourceIntegral` differs). The JSON is
[`out/store_value_diff.json`](06c/rehearsal/out/store_value_diff.json), `57313b01…`, 42.8 MB. Every
store file was opened `mode=ro`. Both stores' SHA-256 values are identical before and after
(`state/store_sha256_before_diff.txt`, `…_after_diff.txt`).

##### Outside `QuadSourceIntegral`

[`out/classify_non_qsi.txt`](06c/rehearsal/out/classify_non_qsi.txt), from 05's
`classify_non_qsi.py`, unchanged:
- **every value table is bit-identical:** 7 tables, 360,462 rows;
- **every other difference** is a timing, a label, `parents_digest` (291 `GkSource`) or
  `rho_reuse` (24 `GkWKBIntegration` rows: `metadata` and `stage_1_RHS_evaluations`);
- **0 unclassified differences,** and no row is in one store only or unmatched, in any class.

**Holds.**

##### Inside `QuadSourceIntegral`: where the region counts and depths are stored

All in each shard's table `QuadSourceIntegral` (`ComputeTargets/QuadSourceIntegral.py`,
`ObjectFactories/QuadSourceIntegral.py`):

| quantity | where | encoding |
|---|---|---|
| region count, row | column `WKB_Levin_num_regions` | Integer, nullable. The **sum** over every WKB Levin call (every phase group of every sub-interval) of `num_regions`. NULL if the row made none (291 rows) |
| region count, per call | `metadata`, `WKB_Levin.subintervals[i].groups[j].regions` | a JSON int. Asserted to sum to the column on every row |
| maximum depth, row | column `WKB_Levin_max_depth` | Integer, nullable. The **maximum** over the same calls of `max_depth`. **No per-call depth is stored** |
| analytic-oracle regions | `metadata`, `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions` | a JSON int, `len(regions)` per three-Bessel block. Feeds `analytic_rad`, not `total`. No depth stored |
| `total_converged` | column | Boolean (SQLite 0/1). The AND of the calls' `converged`, asserted equal to the stored group flags |
| `total`, `total_abserr` | columns | Float(64). `total_abserr` is the linear sum of numeric_quad's and every Levin call's `abserr` |

**Not stored:** `num_untrusted_regions` and the trust check's per-leaf fields.

**Work counters,** compared and tabulated but not values:
- the columns `WKB_Levin_num_regions`, `_evaluations`, `_simple_regions`, `_SVD_errors`,
  `_order_changes`, `_chebyshev_min_order`, `_max_depth`, `numeric_quad_compute_steps` and
  `numeric_quad_RHS_evaluations`;
- the metadata leaves `regions`, `simple_regions` and `evaluations`.

They count the partition and the work. 06c changes those by construction without changing what
they mean.

**Not compared:** timings, `label`, `serial`, `*_serial` and `timestamp`.

##### Inside `QuadSourceIntegral`: per row

[`out/qsi_rows.csv`](06c/rehearsal/out/qsi_rows.csv) holds all 864 rows. The summary is
[`out/qsi_tabulation.txt`](06c/rehearsal/out/qsi_tabulation.txt), from
`scripts/tabulate_qsi_global.py`.

**The reading is checked:**
- 864 rows in each store, the same physical (k, q, r, z_response) set;
- 48 columns, equal;
- tolerances equal row by row;
- 0 `total` mismatches against the tool;
- equal metadata field sets on every row;
- 3,105 Levin calls in each store, the same per row.

**Bit-identity** (`float.hex`):

| | rows |
|---|---|
| `total` bit-identical | **295** of 864: the 291 rows with no WKB Levin call, and 4 rows with one |
| `total_abserr` bit-identical | 291 |
| every value field bit-identical | 205, all without a Levin call. The other 86 such rows differ only in the analytic oracle's metadata |
| `analytic_rad` bit-identical | 455 |

**The four quantities:**

| | R17 | R18 |
|---|---|---|
| `total_converged` rows | 516 of 864 | **863** of 864. R17→R18: 347 went 0→1, 516 stayed 1→1, 1 stayed 0→0 |
| Levin calls unconverged | 830 | 2 |
| Levin calls phase-limited (rows) | 0 (0) | 474 (231) |
| `WKB_Levin_num_regions`, summed | 13,577 | 34,892. Per row R18/R17: median 2.26, range 0.83–13.7 |
| Levin evaluations, summed | 30,231 | 276,031 |
| rows at `depth_max` = 20 | 0 | **1** |
| analytic-block regions | 16,405 | 35,996 |
| analytic blocks phase-limited, unconverged | 0, 132 | 3,506, 2 |

**`WKB_Levin_max_depth` over the 573 rows with a Levin call:**
- R17: `{0: 5, 1: 3, 2: 3, 3: 12, 4: 2, 5: 17, 6: 231, 7: 259, 8: 17, 10: 2, 11: 13, 12: 9}`;
- R18: `{0: 4, 1: 1, 2: 4, 3: 6, 4: 4, 5: 21, 6: 132, 7: 228, 8: 83, 9: 24, 10: 12, 11: 8, 12: 15,
  13: 12, 14: 9, 15: 4, 16: 4, 18: 1, 20: 1}`;
- deeper in R18 in 279 rows, shallower in 64, and equal in 230.

**$|\Delta T|$ against `total_abserr`,** over the rows with a Levin call:
- $|\Delta T|/e_{17}$: median 0.089, p90 1.5, max $5.1\times10^4$;
- $|\Delta T|/e_{18}$: median 0.095, p90 1.1, max $2.6\times10^4$;
- $e_{18}/e_{17}$: median 1.05, range 0.035–5.9;
- $|\Delta T|/|T_{17}|$: median $1.2\times10^{-8}$, max $2.5\times10^{-3}$.

##### Y1 — consistency

**Rule:** every row with finite `total_abserr` in both stores has $\lvert T_{18}-T_{17}\rvert \le
e_{17}+e_{18}$. Rows infinite in either are counted separately.

| | rows |
|---|---|
| finite in both | 838 |
| **violating** | **30** |
| `total_abserr` infinite in R17 (finite in R18) | 26, counted separately. All at $z \ge 1.33\times10^9$, R17-unconverged and R18-converged, $\lvert\Delta T\rvert/e_{18} \le 0.087$ |
| infinite in R18 | 0 |

**Y1 fails** (30 of 838). A violation is a finding for the user, not by itself a stop.
- **Every violating row is printed** with both totals and both `total_abserr` in `float.hex` in
  `out/qsi_tabulation.txt`.
- **Each is localised to its calls** in
  [`out/y1_violations_groups.txt`](06c/rehearsal/out/y1_violations_groups.txt) and
  [`out/y1_violations_worst_calls.txt`](06c/rehearsal/out/y1_violations_worst_calls.txt).

**The 12 large violations** ($|\Delta T|/(e_{17}+e_{18})$ = 336–16,763):
- **Where:** all equilateral ($k=q=r$ at $10^5$: 5 rows, $3.16\times10^5$: 6, $10^6$: 1;
  $z$ 250–6,905).
- **The cause:** each comes from one call, the `G+q+r` phase group of the all-oscillatory
  sub-interval. Its value moves by 0.03–1.0 % (0.64–1.04 % in eleven), against declared errors of
  about $2\times10^{-9}$–$2\times10^{-7}$ of it in each code. On that call
  $|\Delta V|/(e_{17}+e_{18})$ = $4.3\times10^3$–$3.4\times10^5$.
- **Both codes** report that call converged in 11 of the 12.
- **Regions R17→R18:** fewer in R18 in ten (for example 12→2 and 13→2), more in two (2→21, 1→66).
- **Example,** $k=q=r=10^5$, $z=5236.553$:
  - totals `-0x1.5ff6eb7403dbcp-55` → `-0x1.60d489de69404p-55`;
  - `total_abserr` `0x1.1eff05d8001dfp-79` → `0x1.21b46d538f125p-78`.

**What the comparison shows:** at least one of the two codes misdeclares those calls by up to
$3.4\times10^5$. Which one it is, the comparison cannot say.

**The other 18** violate by 1.007–20.4×. They are mostly the `G` group of the
`[True, False, False]` sub-interval, or a `G±r` group, with R17 often unconverged there.

##### Determinism of the "before" (not asked; read-only)

`tools.store_value_diff` of 06b's R16 (06b's code, which is R17's) against R17:
- every value table outside `QuadSourceIntegral` is identical;
- inside it, only timings, labels and the `elapsed`/`compute_time` leaves differ;
- all 6,012 label differences vanish once the time stamp and the job name (`…-06b` against
  `…-06c`) are removed ([`out/r16_06b_vs_r17_labels.txt`](06c/rehearsal/out/r16_06b_vs_r17_labels.txt)).

R16's store is unchanged by the read.

#### The readers (prompt §4.4)

- **Where they ran.** On R18, from the code worktree, with the lock held, through 06b's
  `run_readers_worktree.sh` (`READERS_TREE` the code worktree, `READERS_PYTHON` the venv python).
  Each reader started a local Ray session in `/tmp/ray-sgk` and left no Ray process.
- **The result** ([`out/readers.txt`](06c/rehearsal/out/readers.txt)):

| reader | exit | wall (s) | files |
|---|---|---|---|
| `extract_Gk_data` | 0 | 24 | 432 |
| `extract_GkWKB_data` | 0 | 48 | 2,436 |
| `extract_GkSource_data` | 0 | 19 | 156 |
| `extract_TkWKB_data` | 0 | 15 | 27 |
| `extract_tensor_source_data` | 0 | 13 | 27 |
| `extract_QuadSourceIntegral_data` | 0 | 17 | 71 |

- **The store.** R18's five files are byte-identical before and after each reader (SHA-256,
  mtime, size), with no journal. They are also identical against the hashes taken before the
  comparison ([`out/readers_check.txt`](06c/rehearsal/out/readers_check.txt)).
- **The logs.** 0 Tracebacks; six `!! WARNING: number of shards read from database (=4) does not
  match specified number of shards (=10)`, as in 06b.

**Holds.**

#### At the end

([`out/final_check.txt`](06c/rehearsal/out/final_check.txt))
- the lock is released, and `ray-processes` lists nothing;
- the registry holds 30 runs, all finished;
- the main checkout is at `28ebaf0`, clean;
- both worktrees are clean and unwritten;
- **the stores are kept,** and their final hashes equal those before the comparison:
  - R17's primary `b9f56fa3…`, shards `faaa8e6b…`, `4e945603…`, `04a1f342…`, `ad760544…`;
  - R18's primary `150612c5…`, shards `fec45664…`, `324bb1da…`, `858a1bee…`, `436531a0…`.

**§6 item 2:**
- R17 and R18 are `done` with `1b332ffa…`;
- every value table outside `QuadSourceIntegral` is identical;
- every row inside it is tabulated, and Y1 is scored: **fails, 30 of 838**;
- the readers exit 0;
- the lock is released.
