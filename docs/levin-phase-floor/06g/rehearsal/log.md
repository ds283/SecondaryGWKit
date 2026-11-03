## §4 — rehearsals R23 and R24 on the Mac Studio

**Prompt:** [`06g-leaf-values-and-node-sum.md`](../06g-leaf-values-and-node-sum.md) §4 ("As 06h §4"),
under the orchestrator's Part B dispatch (`1c5336e`) and its addendum B.2. The code measured is
06g's code commit `3ae6f3a`, against its parent `4f4f9ed`, whose code is 06h's (`7172a8f` +
`ac1c6f4`, the code of 06h's census tree `c1ec3b2`).

**Host:** mps026298, the Mac Studio.
- **R23** ran in the worktree `SecondaryGWKit-lpf06g-4f4f9ed`, detached at `4f4f9ed`, `3ae6f3a`'s
  parent. `git diff --stat c1ec3b2 4f4f9ed` outside `docs/` and `prompts/` shows only `CLAUDE.md`.
- **R24**, the comparison and the readers ran in the worktree `SecondaryGWKit-lpf06g-1c5336e`,
  detached at `1c5336e`. `git diff --stat 3ae6f3a 1c5336e` outside `docs/` and `prompts/` is
  empty.
  - **On code paths,** `4f4f9ed..3ae6f3a` changes `AdaptiveLevin/levin_quadrature.py`, five
    `AdaptiveLevin/tests/` files, and adds `tools/filon_reference.py` with its test.
  - **The same file in both trees:** the launcher (`b0135218…`, R21's and R22's), `main.py`
    (`a5d47f5f…`), `ComputeTargets/QuadSourceIntegral.py` (`962424c0…`) and
    `LiouvilleGreen/three_bessel_integrals.py` (`fea67d56…`).
  - **`levin_quadrature.py`** is `8285c1e4…` at the parent (06h's) and `d89c959e…` at 06g, the
    final module the code log records.
- **The main checkout** stayed at `28ebaf0`, clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in three turns:
1. the gate, the lock and R23's launch;
2. R23's checks and R24's launch;
3. R24's checks, the comparison, the tabulation and V1, the readers, the release and these records.

**Date:** 2026-11-01, 11:05–11:35 UTC.

**Commit:** none. The records are committed on the laptop, in 06g's records commit.

**Result:** COMPLETE. **V1 FAILS on `total`** (12 violating rows of 864) **and holds on the
oracle.** Under the prompt, a violation is a finding for the user, not a stop. No §7 stop condition
fired.
- **The fingerprints.**
  - R23 and R24 both ended `done`, exit 0, with fingerprint `1b332ffa…`.
  - Each `classes` object equals R8's, R21's and R22's. `problems` is `{}` in both.
- **Outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` (291 `GkSource` rows) or the
    residual-memo counters (`rho_reuse`, 31 `GkWKBIntegration` rows);
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`,** all 864 rows are tabulated (`out/qsi_rows.csv`).
  - **R23 reproduces 06h's R22** on every tabulated column of all 864 rows (`total`,
    `total_abserr`, the converged and phase-limited flags, region counts, depths, evaluations,
    `analytic_rad`, `analytic.abserr`), read from 06h's committed CSV
    (`out/r23_vs_06h_r22_csv_check.txt`).
  - **`total` moved in 570 rows.** It is bit-identical (`float.hex`) in 294: the 291 rows with no
    WKB Levin call, and 3 of the 573 with one. `total_abserr` is bit-identical in the same 294.
  - **`analytic_rad` moved in 636 rows.** It is bit-identical in 228: the 205 rows with no
    three-Bessel Levin block, and 23 of the 659 with one.
  - **Convergence was lost:** `total_converged` went from 864 rows to 797. 67 rows went converged →
    unconverged, and none the other way. That is 97 WKB Levin calls; R23 has 0 unconverged calls.
  - **Depth and refinement rose:** 76 rows reach `depth_max` (20) in R24 against 0 in R23. Every
    one of the 67 rows that lost convergence is among them.
    - WKB Levin regions went from 31,216 to 134,011 (4.29×), and evaluations from 246,623 to
      1,068,983.
    - 558 rows have more regions and none fewer. 494 rows go deeper and none shallower.
  - **The analytic oracle's blocks** went from 24,428 regions to 29,228. All 10,072 blocks
    converged under both codes.
- **V1 on `total` FAILS: 12 of 864 scored rows violate it.**
  - **The largest $|\Delta T|/(e_{23}+e_{24})$ is 13,101.5**, at $k=q=r=10^5$, $z=5236.553$.
  - **No stationary-point rows and no non-finite rows.** No rehearsal row is one of 06a's 720
    stationary-point rows, or any A3 v2 row. No row has a non-finite `total_abserr`.
  - **Where the violations sit.** In every one of the 12 rows, the whole move is in a single call:
    sub-interval 1's `G+q+r` group, regime (True, True, True).
    - In R23 that call has 2–12 regions, converged, not phase-limited, except one row.
    - In R24 it has 39–208 regions and is phase-limited. It is unconverged at depth 20 in 10
      rows and converged in 2.
    - Its value moves by 2,500–321,000 times its two errors' sum (`out/v1_violations_detail.txt`).
- **V1 on the oracle HOLDS: 0 of 864 scored rows violate it.** The largest
  $|\Delta\,{\rm rad}|/(e_{23}+e_{24})$ is 0.4986.
- **The readers** exited 0, six of six. R24 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6g.

### What shipped

No code. The records are under the main checkout's `var/levin-phase-floor-06g/rehearsal/`, for the
laptop to commit as `docs/levin-phase-floor/06g/rehearsal/`:
- `log.md` (this), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.
- **`scripts/`:**
  - `rec.sh` (`94bd9fdf…`): 06h's (`a24f685e…`), with the record path changed by `sed`;
  - `classify_non_qsi.py` (`26690e61…`): 05's, copied unchanged. It is byte-identical to 05's and
    to 06h's copy;
  - `run_readers_worktree.sh` (`ec96cf09…`): 06h's, copied unchanged. Its diff against
    `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, `out/run_readers_worktree.diff`, equals
    06h's;
  - `tabulate_qsi_leafvalues.py` (`3268a0bc…`): 06h's `tabulate_qsi_groupphase.py` (`dea79d01…`),
    adapted. V1's oracle half is added to it (deviation 3). Its diff against 06h's is
    `out/tabulate_qsi_leafvalues.diff`;
  - `qsi_analytic_summary.py` (`e44a838c…`): 06h's (`7cb08e8f…`), adapted. Its diff is
    `out/qsi_analytic_summary.diff`;
  - `v1_violations_detail.py` (`7fb60969…`): new and descriptive. It breaks V1's violating rows
    down group by group.
- **`out/`:**
  - `commands.log`, every command;
  - the gate checks: `00-gate-trees.txt`, `01-gate-registry-lock.txt` and `01-registry-list.txt`;
  - each build's launch and end checks (`r23_launch_check`, `r23_end_check`, `r24_launch_check`,
    `r24_end_check`), both `status.json`, and the registry listings;
  - `store_value_diff.json` (`36cd46a4…`, 42.3 MB; kept on the Mac Studio) and its `.stderr`;
  - `classify_non_qsi.txt` (`8796d2cb…`);
  - `qsi_rows.csv`, every row on `total` and on the oracle (`325228f9…`), and `qsi_tabulation.txt`
    (`51517042…`), which holds both V1 verdicts;
  - `qsi_analytic_rows.csv`, every row (`ad7eb0d8…`), and `qsi_analytic_summary.txt`
    (`bbaa15df…`);
  - `v1_violations_detail.txt` (`aaa4a95f…`), `qsi_unconverged_groups.txt` (`32c0753c…`) and
    `qsi_lost_convergence_crosscount.txt`, all descriptive;
  - `r23_vs_06h_r22_csv_check.txt` (`2c28e4c5…`), a read-only cross-reading against 06h's
    committed tabulation;
  - `readers.txt`, `readers_check.txt` and `run_readers_worktree.diff`;
  - `final_check.txt`.
- **`state/`:**
  - `lock.log`;
  - both launch lines and the three markers;
  - every hash file of the stores: after R23's build, at R24's launch, before and after the
    comparison, before and after the readers, and final.
- **Not for copying back:**
  - the readers' `extract/` (182 MB, 3,187 files);
  - `out/store_value_diff.json` (42.3 MB), listed in `KEPT_ON_MAC_STUDIO.txt`.
- **The stores are kept:** `var/datastores/levin-phase-floor-06g-r23*` and `…-r24*`, five files
  each.
  - Their run directories are `var/runs/levin-phase-floor-06g-r23-parent-20261101T110641/` and
    `…-r24-leafvalues-20261101T111215/`.
  - Their launch logs are `var/runs/levin-phase-floor-06g-r{23,24}.launch.log`.

### Predictions and verdicts

| Rule (prompt 06g §4, as written; "as 06h §4") | Scored on | Verdict |
|---|---|---|
| R23 and R24 run as 06h's R21 and R22 ran, slugs `r23-parent` and `r24-leafvalues`, `--prompt 06g`; **both fingerprints `1b332ffa…`** | Both `done`, exit 0, `1b332ffa…`. `classes` equal R8's, R21's and R22's in both. `problems {}`. Same launcher (`b0135218…`) and arguments as R21 and R22 | **holds** |
| Outside `QuadSourceIntegral`: every value table is identical | 7 tables, 360,462 value rows, all identical. `value tables not identical: none` in every class | **holds** |
| …every other difference is of the four volatile kinds | timing, label, `parents_digest` (291), `rho_reuse` (31 `GkWKBIntegration` rows). **0 unclassified** | **holds** |
| …and no row is in one store only | 0 only-in-A, 0 only-in-B and 0 unmatched, in every class | **holds** |
| Inside it, every row is tabulated: $\lvert T_{24}-T_{23}\rvert$ against `total_abserr` in each, and the same on the analytic oracle (`analytic_rad` against `analytic.abserr`); region counts, maximum depths and `total_converged` in each | 864 of 864 rows, in `out/qsi_rows.csv` (both quantities) and `out/qsi_analytic_rows.csv` | **done** |
| **V1 — consistency.** Every row outside 06a's 720 stationary-point rows with finite `total_abserr` in both stores has $\lvert T_{24}-T_{23}\rvert \le e_{23}+e_{24}$, and the same on the oracle. Rows with an infinite error in either are counted separately; the stationary-point rows are tabulated, not scored. Holds if no row violates either | **On `total`:** 0 stationary-point rows; 864 scored; **12 violate**, largest ratio 13,101.5; 0 non-finite. **On the oracle:** 864 scored; 0 violate, largest ratio 0.4986; 0 non-finite | **fails** (on `total`; a finding for the user, not a stop) |
| The readers run on R24, exit 0, and the store is byte-identical before and after | 6 of 6 exit 0. The 5 files are identical by SHA-256, size and mtime; no journal | **holds** |
| The lock is released, and `ray-processes` lists nothing | released, exit 0; `status` free; `ray-processes` empty before and after the release | **holds** |
| The stores are kept | both stores in place, unchanged since the comparison | **holds** |

**§6 item 2 (acceptance):** "R23 and R24 both `done` with `1b332ffa…`; outside `QuadSourceIntegral`
every value table identical; V1 scored; the readers exit 0; the lock released." Every clause is met.
V1 is scored, and it fails.

**V1's 12 violating rows**, each with both totals and both `total_abserr` in `float.hex`. All are at
(atol, rtol) = (1e-32, 1e-7), with $k=q=r$:

| $k$ | $z_{\rm response}$ | `total` R23 | `total` R24 | `total_abserr` R23 | `total_abserr` R24 | ratio |
|---|---|---|---|---|---|---|
| 1e5 | 3971.4126 | `-0x1.620fef0efb626p-56` | `-0x1.6264cbb02c333p-56` | `0x1.53b4899ee9003p-79` | `0x1.56ce16ca76decp-79` | 4,074.3 |
| 1e5 | 5236.553 | `-0x1.60d489de69402p-55` | `-0x1.5ff6ede3f1fadp-55` | `0x1.21b44281c283ep-78` | `0x1.088f35de11bf5p-78` | 13,101.5 |
| 1e5 | 6904.7189 | `-0x1.856721d9b93e5p-55` | `-0x1.85621fd5aeb33p-55` | `0x1.729c4e92c90ebp-78` | `0x1.6a4705e03b114p-78` | 223.9 |
| 316228 | 434.65133 | `-0x1.bcb9578284e92p-64` | `-0x1.bbf87768382dap-64` | `0x1.eff7071a8f92ap-88` | `0x1.e64ede7779d01p-88` | 12,868.4 |
| 316228 | 573.11465 | `0x1.01dcb4297e779p-63` | `0x1.0163814429fcbp-63` | `0x1.91a4ed615374dp-87` | `0x1.7888456c4574ep-87` | 10,207 |
| 316228 | 996.42016 | `-0x1.2179faec40418p-62` | `-0x1.21132816cf33ep-62` | `0x1.b9223a4398a9cp-86` | `0x1.b1c6747755c13p-86` | 7,702.1 |
| 316228 | 6904.7189 | `0x1.2b8433694421fp-60` | `0x1.2b4c92dd0131bp-60` | `0x1.3585cfdb5c416p-84` | `0x1.2b3a0f49a3a98p-84` | 5,988.6 |
| 1e6 | 329.64046 | `-0x1.a4001d0eea20ap-70` | `-0x1.a4a7b36307b24p-70` | `0x1.749052d5bed9ap-93` | `0x1.50fe6c9753b0bp-93` | 7,739.3 |
| 1e6 | 573.11465 | `-0x1.4b8ed0563fd6ep-68` | `-0x1.4bb0ed52fd428p-68` | `0x1.14f5d284d513dp-91` | `0x1.f651f176a5f76p-92` | 2,116.6 |
| 1e6 | 996.42016 | `-0x1.199b2af471ea1p-67` | `-0x1.19b37534bf06bp-67` | `0x1.a4df28a0efa98p-91` | `0x1.d527ab2931c68p-91` | 1,788.6 |
| 1e6 | 1732.3814 | `-0x1.e6ea7941509c2p-67` | `-0x1.e712f5a4fc756p-67` | `0x1.55e5071cf810ap-90` | `0x1.637e2d61dc783p-90` | 1,902.3 |
| 1e6 | 6904.7189 | `0x1.fb6aca7818b1ep-65` | `0x1.fbac854c4b16cp-65` | `0x1.0e8e346b764b8p-87` | `0x1.d25b550ba8510p-88` | 2,137.9 |

- **Largest ratios on `total`:** $\lvert\Delta T\rvert/e_{23}$ is 25,486.2 ($k=316228$,
  $z=434.65$). $\lvert\Delta T\rvert/e_{24}$ is 27,448.3 ($k=10^5$, $z=5236.55$).
- **The distribution of $\lvert\Delta T\rvert/(e_{23}+e_{24})$** over the 864 rows: p50 0.019,
  p90 0.159, p99 1,900, p100 13,100.
- **Largest ratios on the oracle:** $\lvert\Delta\,{\rm rad}\rvert/e_{23}$ and
  $\lvert\Delta\,{\rm rad}\rvert/e_{24}$ are both 0.9972. Over the 636 rows that moved, the
  distribution of $\lvert\Delta\,{\rm rad}\rvert/(e_{23}+e_{24})$ is p50 0.106, p90 0.432,
  p100 0.4986.

### Where the region counts, maximum depths and the oracle are stored (§4)

`ComputeTargets/QuadSourceIntegral.py` (`962424c0…`) is the same file at `4f4f9ed` and `1c5336e`,
and the same as at 06h's `2c325f5` and `c1ec3b2`, so 06h's reading holds unchanged. The script
re-checks it by assertion on every row of both stores.

Everything is in each shard's table `QuadSourceIntegral`:
- **Region count, the call total:** column `WKB_Levin_num_regions`. SQLAlchemy `Integer`, SQLite
  integer, nullable.
  - It is the **sum** of the calls' `num_regions`, over every `adaptive_levin_sincos` call of the
    row's WKB Levin sub-intervals (`_aggregate_LevinData`).
  - It is NULL when the row made no WKB Levin call (291 rows in both stores).
- **Region count, per call:** column `metadata`, a JSON string written by `json.dumps`, at
  `WKB_Levin.subintervals[i].groups[j].regions` (a JSON int). Each group there also stores `value`,
  `abserr`, `converged`, `phase_limited`, `simple_regions` and `evaluations`. The script asserts
  that the `regions` sum to `WKB_Levin_num_regions` on every row of both stores.
- **Maximum depth:** column `WKB_Levin_max_depth`, `Integer`, nullable. It is the **maximum** of the
  calls' `max_depth`, over the same calls. **No per-call depth is stored.**
- **The analytic oracle's value:** column `analytic_rad`, `Float(64)`.
- **The oracle's error:** `metadata` at `analytic.abserr`, a JSON number (`json.dumps` of a Python
  float). The flag `analytic.converged` (JSON bool) sits beside it.
- **The oracle's per-block records:** `metadata` at
  `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data`, with `regions` (a JSON int, `len(regions)` of the
  block), `abserr` (JSON float), `converged` and `phase_limited` (JSON bool), and `elapsed` (a
  timing). **No depth and no per-block value is stored for them.**
- **`total_converged`:** a column, `Boolean`, SQLite integer 0/1, nullable. It is the AND of every
  WKB Levin call's `converged`. The script asserts that it equals the AND of the stored group flags.
- **`total_abserr`** and **`total`:** columns, `Float(64)`.
- **06g's new result keys are not stored.** `QuadSourceIntegral` persists only the keys above. So
  `node_limited`, `node_error_signed_sum`, `node_error_abs_sum`, `abserr_node_signed` and the
  per-leaf `value_kind`, $r$ and trust are not in either store (Observation 4).

**Work counters, not values** (06b's, 06c's, 06e's and 06h's classification, kept):
- **The columns:** `WKB_Levin_num_regions`, `WKB_Levin_evaluations`, `WKB_Levin_simple_regions`,
  `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`,
  `WKB_Levin_max_depth`, `numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`.
- **The metadata leaves:** `regions`, `simple_regions` and `evaluations`.
- **Why they are counters.** They count the partition the driver chose and the work it did. None
  of them is the integral or a bound on its error.
  - 06g moves the partition (§0.3: refinement rises where slow leaves were trusted).
  - Its diff to `levin_quadrature.py` leaves the result keys `num_regions`, `num_simple_regions`,
    `evaluations`, `max_depth`, `num_SVD_errors`, `num_order_changes` and `chebyshev_min_order`,
    and their assignments, untouched. `ComputeTargets/` is unchanged (§0.3). So no counter changes
    its meaning.
  - They are compared and tabulated, region counts and depth included, but a difference in one is
    not a value difference.

**Not compared:** timings, the label and store bookkeeping. **Everything else is a value**,
including `total_converged`, `total_phase_limited`, and every `converged`, `phase_limited` and
`abserr` leaf.

**The stationary-point rows.** 06a's 720 are identified as 06c's, 06e's and 06h's tabulations
identified them, from 06a's record rather than by re-running `spscan.py` (deviation 5).
- **The rule:** `analyse06c.py`'s `scan06a` rule, applied to the 1e-7 records of 06a's census
  checkpoint (`var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl`,
  `c13685b4…`). A row is a stationary-point row if its `scan` has a call with
  `n_sign_changes > 0`; the script asserts there are 720 of them.
- **The mapping:** each is mapped to its physical key (k, q, r, z_response) through 06a's
  `state/rows.json` (`aa523d28…`).
- **The rehearsal store:** a row is among them iff its own physical key is equal, as floats, to
  one of theirs. **0 of the 864 rows are.** No rehearsal row equals any A3 v2 row. They share only
  $k=10^5$ and no $z_{\rm response}$.

### Deviations from the prompt

1. **The gate's registry check grepped the listing for `running|stale`, and matched one row's
   purpose text.** UNINTENDED DRIFT.
   - The row was `levin-phase-floor-06h-phase-20261101T033514`, whose stage reads "refcheck
     running". Its state is `done`.
   - The state column was then tallied: 34 done, 2 failed, 4 killed. The footer reads "40 finished".
   - No record is affected.
2. **R23's end check printed every nested `digest` key.** IMPLEMENTATION CHOICE.
   - The check searched `status.json` recursively for `digest`, so it printed the per-class
     digests too.
   - Only the last one printed, the top-level digest, is compared with the literal, and it is
     `True`. The `classes` objects were compared whole with R21's, R22's and R8's.
   - R24's end check read `fingerprint.digest` directly.
3. **The tabulation scores V1's oracle half, which 06h's script did not.** IMPLEMENTATION CHOICE.
   - **The change.** The tabulation is 06h's `tabulate_qsi_groupphase.py`, with:
     - the names R21/R22 → R23/R24, and H0 → V1;
     - the counters' paragraph restated for 06g;
     - the depth comment updated.
     It also adds the oracle rule: `analytic_rad` with `analytic.abserr`, the same exclusions, the
     violations in `float.hex`, and the largest ratios.
   - **Why.** In 06h the oracle form was descriptive only, in `qsi_analytic_summary.py`. 06g's V1
     scores both.
   - **Reading.** "Finite `total_abserr`" is read, for the oracle half, as finite
     `analytic.abserr`. A non-finite `analytic_rad` is also counted as non-finite. No row was
     non-finite either way.
   - **Diffs.** `out/tabulate_qsi_leafvalues.diff` and `out/qsi_analytic_summary.diff`; the summary
     changes only its names, its import and its docstring.
4. **Descriptive outputs beyond the brief.** IMPLEMENTATION CHOICE.
   - **The outputs:**
     - `v1_violations_detail.py` → `out/v1_violations_detail.txt`;
     - `out/qsi_unconverged_groups.txt`;
     - `out/qsi_lost_convergence_crosscount.txt`;
     - `out/r23_vs_06h_r22_csv_check.txt`.
   - **What they do.** Each is read-only on the stores (sqlite `mode=ro`) or on CSVs, and none is
     scored. They locate V1's violations and the lost convergence, and check R23 against 06h's
     R22.
   - **When they ran:** while the lock was held and no other stage was running.
5. **The stationary-point rows come from 06a's census checkpoint record, not from running
   `spscan.py`.** IMPLEMENTATION CHOICE. The brief allows "from 06a's `spscan.py` or its record". The
   method is 06c's, 06e's and 06h's, and gives the 720 by assertion.
6. **`black` was not applied to the scripts.** IMPLEMENTATION CHOICE.
   - `black --check` would reformat four of them (`out/final_check.txt`). They are records, left as
     run.
   - `classify_non_qsi.py` must stay byte-identical to 05's.

### Verification performed

**The gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`):
- **Trees:** the main checkout is `28ebaf0`, clean. `4f4f9ed` and `1c5336e` are detached, with
  `status --short --ignored` empty. `git worktree list` shows three trees.
- **The diffs:** `3ae6f3a^` is `4f4f9ed`, and the diffs are as stated above.
- **Registry:** 40 runs, all finished, nothing `running` or stale.
- **Lock:** `~/.raypoolkit` was empty and the old lock location was absent.
- **Host:** `~/.ray/auth_token` mode 600, 96 GB free.
- **Packages:** venv 3.13.16, ray 2.59.0, SQLAlchemy 2.0.55.
- **Clean start:** no 06g store or run existed.

**R23** (`out/r23_launch_check.txt`, `out/r23_end_check.txt`, `out/r23_status.json`):
- **Launch:** at 11:06:40Z from the parent worktree (`sh` 44456, driver 44460), then `set-pid 44460`.
  Ray started at 11:06:48 with `num_cpus=6`.
- **Manifest:** `git_head` `4f4f9ed`, `git_dirty` false, `script_sha256` `b0135218…`. The driver's
  cwd is the parent worktree.
- **End:** `done`, exit 0, heartbeat 11:11:17Z. Fingerprint `1b332ffa…`, taken at `4f4f9ed`,
  `problems {}`.
- **Logs:** no Traceback and no `!! WARNING` in the launch log, `stdout.log` or `stderr.log`. The
  launch log ends "** driver exit status 0".
- **Trees:** both worktrees are clean, with nothing newer than the launch marker.
- **The store:** 5 files, no journal (`state/r23_store_sha256_after_build.txt`). It was unchanged
  at R24's launch, and again before and after the comparison.

**R24** (`out/r24_launch_check.txt`, `out/r24_end_check.txt`, `out/r24_status.json`):
- **Launch:** at 11:12:14Z from the code worktree (`sh` 45376, driver 45379), then `set-pid 45379`.
  Ray started at 11:12:21.
- **Manifest:** `git_head` `1c5336e`, `git_dirty` false, `script_sha256` `b0135218…`. The driver's
  cwd is the code worktree.
- **End:** `done`, exit 0, heartbeat 11:22:45Z. Fingerprint `1b332ffa…` (`True` against the
  literal), taken at `1c5336e`, `problems {}`. Its `classes` equal R23's, R21's, R22's and R8's.
  `QuadSourceIntegral` holds 864 rows.
- **Logs:** no Traceback and no `!! WARNING` in either launch log, R24's `stdout.log` or its
  `stderr.log`.
- **Trees:** both worktrees are clean, with nothing newer than R23's marker.
- **The registry** holds 42 runs, all finished.
- **Wall time:** about 10.5 min, against R23's 4.6 min (descriptive; cost is not an argument).

**The comparison** (cwd the code worktree, so `-m` imports its tool):
- **The tool:** `tools/store_value_diff.py` (`7997af85…`, 06h's), which opens every file through
  `open_read_only`. It exited 1, because `QuadSourceIntegral` differs, in 16.1 s.
- **Outside `QuadSourceIntegral`** (`out/classify_non_qsi.txt`): all value tables are identical
  (1,159 + 1,478 + 2,209 + 35,874 + 147,642 + 169,653 + 2,447 = 360,462 rows), and the differences
  are classified as above.
- **The stores** are byte-identical before and after the comparison and the tabulations
  (`state/store_sha256_before_diff.txt`, `…_after_diff.txt`).

**The tabulation** (`out/qsi_tabulation.txt`):
- The stores' key sets are equal, with 48 columns each and the tolerances equal row by row.
- The tool reports 864 rows differing, 0 only in one store and 0 unmatched. Its keys equal the
  stores' set.
- The cross-check of `total` against the tool finds 0 mismatches.
- The field sets of every row are equal in both stores.
- The WKB Levin call count is 3,105 in both, equal row by row.

| | R23 | R24 |
|---|---|---|
| `total` bit-identical (`float.hex`) | 294 / 864 (291 with no WKB Levin call + 3) | |
| `total_abserr` bit-identical | 294 / 864 | |
| `analytic_rad` bit-identical | 228 / 864 (205 with no three-Bessel block + 23) | |
| `analytic.abserr` bit-identical | 378 / 864 | |
| every value field bit-identical | 208 / 864 (all with no WKB Levin call) | |
| `total_converged` | 864 | 797 |
| WKB Levin calls unconverged | 0 | 97 |
| `total_phase_limited` | 195 | 478 |
| WKB Levin calls phase-limited | 406 | 1,411 |
| `WKB_Levin_num_regions`, summed | 31,216 | 134,011 |
| `WKB_Levin_evaluations`, summed | 246,623 | 1,068,983 |
| rows at `WKB_Levin_max_depth` = 20 | 0 | 76 |
| analytic-block regions, summed | 24,428 | 29,228 |
| analytic blocks converged | 10,072 / 10,072 | 10,072 / 10,072 |
| analytic blocks phase-limited | 3,896 | 3,768 |

- **`total_abserr`, R24/R23,** over the rows with a WKB call: p0 0.186, p50 0.955, p99 1.40,
  p100 2.05. It rose in 129 rows and fell in 441.
- **$\lvert\Delta T\rvert/\lvert T_{23}\rvert$:** p50 1.2e-8, p99 7.3e-4, p100 2.45e-3.
- **Maximum depth,** R23 → R24: equal in 79 rows, deeper in 494, shallower in 0.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- They ran on R24 from the code worktree, under the lock, through 06h's wrapper unchanged.
- **Exits:** six of six exit 0, with 0 Tracebacks.
- **The store:** six "identical before and after" lines, and R24's hash files are equal before and
  after the reads.
- **Warnings:** six `!! WARNING … shards (=4) … (=10)` lines, as 06h's.
- **Ray:** no Ray process from this checkout's venv is left after any reader.

**The lock** (`state/lock.log`): `ray-processes` (empty), `acquire`, `set-pid 44460`,
`ray-processes` (empty), `set-pid 45379`, `ray-processes` (empty), `release`, `status` (free, Ray
processes none), `ray-processes` (empty). Each exited 0, and each was a command of its own.

**Final** (`out/final_check.txt`):
- the store hashes are equal to those after the comparison;
- 42 runs, all finished, and `~/.raypoolkit` is empty;
- the main checkout and both worktrees are clean and unmoved;
- there is no `__pycache__` in the records.

### Observations not acted on

1. **V1 fails on `total` in 12 rows, and every violation is one call: sub-interval 1's `G+q+r`
   group.** This is the highest-$z$ oscillatory sub-interval, from $z\sim10^9$–$10^{10}$ down to
   $z_{\rm response}$ (`out/v1_violations_detail.txt`).
   - **The call in each store.**
     - In R23 it has 2–12 regions and is converged; it is not phase-limited in 11 of the 12.
     - In R24 it has 39–208 regions and is phase-limited in all 12. It is unconverged at depth 20 in
       10 and converged in 2.
   - **The size of the move.** The group's value moves by 0.1–3.4 % (for example
     `0x1.4d0cf4361bcb7p-59` → `0x1.4a6610ee6fad8p-59`). Both stores declare errors near
     $10^{-7}$ of it, so the move is 2,500–321,000 times the group's two errors' sum (224–13,100
     times at row level).
   - **What it implies.** At least one of the two stores' errors understates the true error by that
     factor.
   - **What cannot be told from here.** This rehearsal has no reference, so it cannot say which
     store is right. The pattern fits (b): R23 trusts a coarse leaf that 06g refuses to trust and
     refines.
   - **Nothing was changed.** It is opened for the orchestrator in `issues.md`. §5's Filon tool
     could referee these calls, but their $k$ are not A3 v2's.
2. **Convergence was lost in 67 rows (97 calls), and every one of those rows reaches depth 20 in
   R24** (`out/qsi_lost_convergence_crosscount.txt`, `out/qsi_unconverged_groups.txt`).
   - **Where.** The lost calls are all in the regime (True, True, True) groups `G±q±r` of
     sub-intervals 1 and 2: 68 in sub-interval 1 and 29 in sub-interval 2.
   - **Spread across $k$:** 17 rows at $k=10^5$, 25 at 316,228 and 25 at $10^6$.
   - **Depth 20.** 76 rows reach depth 20 under 06g; 9 of them still converge.
   - **The likely cause.** A call that reaches `depth_max` is flagged unconverged, and 06g's
     node-limited stop ends below `depth_max`. So the cause in these rows is refinement reaching the
     cap, consistent with (b)'s untrusted leaves. That is an inference: no per-leaf data are stored.
   - **What it bears on.** §5's V5 (no call at `depth_max`) and V8 (no convergence lost), and the
     issue `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`, whose mechanism this
     is not.
3. **The phase-limited flag spread.**
   - `total_phase_limited` went false → true in 283 rows, and never true → false.
   - WKB calls phase-limited rose from 406 to 1,411.
   - On the oracle, 139 blocks went phase-limited → not and 11 the other way.
4. **06g's new driver keys are not persisted.** `node_limited`, the signed node sums and the
   per-leaf trust records are not in `QuadSourceIntegral`. A rehearsal cannot attribute a lost
   convergence to (b), to the node-limited stop or to the exit; only a replay can (§5's census
   harness records them).
5. **The readers' file counts differ from 06h's.** `extract_Gk_data` wrote 432 files against 396,
   and `extract_GkSource_data` 186 against 134. Both readers draw unseeded `random.sample`
   subsamples (`extract_Gk_data.py` 416 and 432, `extract_GkSource_data.py` 861 and 877). So their
   outputs are not reproducible file for file; this is not a store difference.
6. **R23 reproduces 06h's R22 exactly** on every tabulated column of all 864 rows. The parent tree
   at `4f4f9ed` builds what 06h's code built at `c1ec3b2`.

### State handed on

- **R23** `levin-phase-floor-06g-r23-parent-20261101T110641` and **R24**
  `levin-phase-floor-06g-r24-leafvalues-20261101T111215`: both `done`, `1b332ffa…`.
- **Their stores are kept,** unchanged since the comparison
  (`state/stores_sha256_final.txt`):
  - `var/datastores/levin-phase-floor-06g-r23*`, primary `88cc355f…`;
  - `var/datastores/levin-phase-floor-06g-r24*`, primary `6e4815c8…`.
- **The lock is released.** `~/.raypoolkit` is empty, and `ray-processes` lists nothing. The six
  cores are free for §5.
- **The trees.** The main checkout and both worktrees are clean and unmoved. Nothing was written in
  either worktree.
- **For the orchestrator:** V1 fails on `total` (Observation 1), and convergence is lost in 67
  rehearsal rows (Observation 2). Both are in `issues.md`.
- **Kept on the Mac Studio, not for copying back:** `out/store_value_diff.json` (42.3 MB) and
  `extract/` (182 MB). See `KEPT_ON_MAC_STUDIO.txt`.
