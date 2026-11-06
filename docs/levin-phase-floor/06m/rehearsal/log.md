## §4 — rehearsals R25 and R26 on the Mac Studio

**Prompt:** [`06m-signed-yardstick-in-the-driver.md`](../06m-signed-yardstick-in-the-driver.md) §4
("As 06g §4"), under the orchestrator's Part B dispatch (`38f6a828`) and its addendum B.1. The code
measured is 06m's code commit `48cf87c0`, route B's signed trust yardstick. It is measured against
its parent `5ff8a5e4`, whose production code is R24's (06g's `3ae6f3a`).

**Host:** mps026298, the Mac Studio.
- **R25** ran in the worktree `SecondaryGWKit-lpf06m-5ff8a5e`, detached at `5ff8a5e4`, which is
  `48cf87c0`'s parent.
  - `git diff --stat 1c5336e 5ff8a5e4` outside `docs/` and `prompts/` touches only `.gitattributes`,
    `CLAUDE.md`, `tools/filon_reference.py` and its test.
  - `levin_quadrature.py` is `d89c959e…` there, R24's.
- **R26**, the comparisons and the readers ran in the worktree `SecondaryGWKit-lpf06m-38f6a82`,
  detached at `38f6a828`.
  - `git diff --stat 5ff8a5e4 38f6a828` outside `docs/` and `prompts/` touches
    `AdaptiveLevin/levin_quadrature.py` (`beaf0e23…`), `test_levin_leaf_values.py`, the new
    `test_levin_trust_yardstick.py` and `CLAUDE.md`, and nothing else.
  - **The same file in both trees:**
    - the launcher `RunDrivers/quadsource_atol_sweep.py` (`b0135218…`, R23's and R24's);
    - `main.py` (`a5d47f5f…`);
    - `ComputeTargets/QuadSourceIntegral.py` (`962424c0…`);
    - `LiouvilleGreen/three_bessel_integrals.py` (`fea67d56…`);
    - `tools/store_value_diff.py` (`7997af85…`).
- **The main checkout** stayed at `28ebaf07`, clean. It ran no tree code. It held the venv, the
  stores, the registry and these records.

**Model:** Claude Opus 5.5, the rehearsal subagent of the Mac Studio orchestrator, in four turns:
1. the gate, the lock and R25's launch;
2. R25's checks, R25 against R24, and R26's launch;
3. R26's checks, R26 against R25, the tabulation and R1 (planned stop 1, lock held);
4. the readers, the release and these records.

**Date:** 2026-11-06, 09:23–10:00 UTC.

**Commit:** none. The records are committed on the laptop, in 06m's records commit.

**Result:** COMPLETE. **R1 HOLDS on `total` and on the oracle**: 0 violating rows of 864 on each.
No §7 stop condition fired.
- **The fingerprints.**
  - R25 and R26 both ended `done`, exit 0, with fingerprint `1b332ffa…`.
  - Each `classes` object equals R24's and R23's. `problems` is `{}` in both.
- **R25 against R24: value-identical.**
  - Every value table outside `QuadSourceIntegral` is identical: 7 tables, 360,462 rows.
  - Inside it, every value and counter field of all 864 rows is bit-identical (`float.hex`). That
    includes `total`, `total_abserr`, `analytic_rad`, `analytic.abserr` and `total_converged`.
  - The rehearsal is reproducible. The only differences are timings, labels, `parents_digest`,
    `rho_reuse` and store bookkeeping.
- **R26 against R25, outside `QuadSourceIntegral`:**
  - every value table is bit-identical: 7 tables, 360,462 rows;
  - every other difference is a timing, a label, `parents_digest` (291 `GkSource` rows) or the
    residual-memo counters (`rho_reuse`, 24 `GkWKBIntegration` rows);
  - 0 differences are unclassified, and no row is in one store only.
- **Inside `QuadSourceIntegral`,** all 864 rows are tabulated (`out/qsi_rows.csv`).
  - **`total` moved in 451 rows.** It is bit-identical (`float.hex`) in 413: the 291 rows with no
    WKB Levin call, and 122 of the 573 with one. `total_abserr` is bit-identical in 403.
  - **`analytic_rad` moved in 4 rows.** It is bit-identical in 860. `analytic.abserr` is
    bit-identical in 848.
  - **Convergence was gained, and none lost.**
    - `total_converged` went from 797 rows to 845: 48 rows went unconverged → converged, and none
      the other way.
    - Unconverged WKB Levin calls fell from 97 to 20.
  - **Refinement fell.**
    - WKB Levin regions went from 134,011 to 114,826 (0.857×), and evaluations from 1,068,983 to
      915,503.
    - 442 rows have fewer regions, and none more.
    - 275 rows are shallower, and none deeper.
    - Rows at `depth_max` (20) fell from 76 to 23.
- **R1 on `total` HOLDS: 0 of 864 scored rows violate it.** The largest
  $|\Delta T|/(e_{25}+e_{26})$ is $4.93\times10^{-6}$. No stationary-point row, no non-finite row.
- **R1 on the oracle HOLDS: 0 of 864 scored rows violate it.** The largest
  $|\Delta\,{\rm rad}|/(e_{25}+e_{26})$ is $6.68\times10^{-8}$.
- **The readers** exited 0, six of six. R26 is byte-identical before and after them.
- **The lock** is released, and `ray-processes` lists nothing.

The measurements are in `verification.md`, written for `docs/levin-phase-floor/VERIFICATION.md`
§6m.

### What shipped

No code. The records are under the main checkout's `var/levin-phase-floor-06m/rehearsal/`, for the
laptop to commit as `docs/levin-phase-floor/06m/rehearsal/`:
- `log.md` (this), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`.
- **`scripts/`:**
  - `rec.sh` (`0a13627b…`): 06g's (`94bd9fdf…`), with the record path changed by `sed`;
  - `classify_non_qsi.py` (`26690e61…`): 05's, copied unchanged, byte-identical to 05's and to
    06g's copy;
  - `run_readers_worktree.sh` (`ec96cf09…`): 06g's (= 06h's), copied unchanged. Its diff against
    `docs/dk-and-rpk-migration/rehearsal/run_readers.sh`, `out/run_readers_worktree.diff`, equals
    06g's;
  - `tabulate_qsi_signedyardstick.py` (`3a4d8748…`): 06g's `tabulate_qsi_leafvalues.py`
    (`3268a0bc…`), adapted (deviation 4). Its diff is `out/tabulate_qsi_signedyardstick.diff`;
  - `qsi_identity_r24_r25.py` (`e63aae0c…`): new. It checks R25 against R24 inside
    `QuadSourceIntegral` by bit pattern over every row and field (deviation 2);
  - `normalise_jobname_labels.py` (`08dc45a8…`): new. It normalises the job-name prefix of the
    labels in R24 against R25's JSON (deviation 1).
- **`out/`:**
  - `commands.log`, every command;
  - the gate checks: `00-gate-trees.txt`, `01-gate-registry-lock.txt` and `01-registry-list.txt`;
  - each build's launch and end checks (`r25_launch_check`, `r25_end_check`, `r26_launch_check`,
    `r26_end_check`), both `status.json`, and the registry listings;
  - **R25 against R24:**
    - `store_value_diff_r24_r25.json` (`d9445335…`, 40.1 MB; kept on the Mac Studio) and its
      `.stderr`;
    - `classify_non_qsi_r24_r25.txt` (on the raw JSON);
    - `normalise_jobname_labels.txt`;
    - `store_value_diff_r24_r25.labels_normalised.json` (`117f0991…`, 26.9 MB; kept);
    - `classify_non_qsi_r24_r25_labels_normalised.txt`;
    - `qsi_identity_r24_r25.txt`;
    - `qsi_tool_metadata_keys_r24_r25.txt`;
  - **R26 against R25:**
    - `store_value_diff.json` (`d67e696f…`, 41.1 MB; kept) and its `.stderr`;
    - `classify_non_qsi.txt` (`f78d2966…`);
    - `qsi_rows.csv`, every row on `total` and on the oracle (`d4ed0096…`);
    - `qsi_tabulation.txt` (`d7abb598…`), which holds both R1 verdicts and the 48 rows whose
      `total_converged` changed;
  - `readers.txt`, `readers_check.txt`, `run_readers_worktree.diff`, `final_check.txt` and
    `final_registry_list.txt`.
- **`state/`:**
  - `lock.log`;
  - the launch lines, `launch_r25.txt` and `launch_r26.txt`;
  - the markers `r25_launch_marker`, `r26_launch_marker` and `readers_marker`;
  - every hash file of the stores:
    - R24 and R25 before and after their comparison;
    - R25 at R26's launch;
    - all three before and after R26's comparison;
    - R26 before and after the readers;
    - the final hashes, `stores_sha256_final.txt`.
- **Not for copying back:**
  - the three diff JSONs;
  - the readers' `extract/` (179 MB, 3,171 files).

  Both are listed in `KEPT_ON_MAC_STUDIO.txt`.
- **The stores are kept:** `var/datastores/levin-phase-floor-06m-r25*` and `…-r26*`, five files
  each.
  - Their run directories are `var/runs/levin-phase-floor-06m-r25-parent-20261106T092420/` and
    `…-r26-signedyardstick-20261106T093937/`.
  - Their launch logs are `var/runs/levin-phase-floor-06m-r{25,26}.launch.log`.

### Predictions and verdicts

| Rule (prompt 06m §4, as written; "as 06g §4") | Scored on | Verdict |
|---|---|---|
| R25 and R26 run as 06g's R23 and R24 ran, slugs `r25-parent` and `r26-signedyardstick`, `--prompt 06m`; **both fingerprints `1b332ffa…`** | Both `done`, exit 0, `1b332ffa…`. `classes` equal R24's and R23's in both. `problems {}`. Same launcher (`b0135218…`) and arguments as R23 and R24, with `--job-name levin-phase-floor-06m` | **holds** |
| **R25 against R24:** every value table identical, every other difference of the four volatile kinds | 7 tables, 360,462 rows, identical. In `QuadSourceIntegral` all 864 rows are bit-identical in every value and counter field. Other differences: timing, label (job-name prefix and stamp, deviation 1), `parents_digest` (291), `rho_reuse` (27 rows), store serials and timestamps. 0 only-in-one-store | **holds** (no difference: the rehearsal is reproducible) |
| Outside `QuadSourceIntegral` (R26 against R25): every value table is identical | 7 tables, 360,462 value rows, all identical. `value tables not identical: none` in every class | **holds** |
| …every other difference is of the four volatile kinds | timing, label, `parents_digest` (291), `rho_reuse` (24 `GkWKBIntegration` rows). **0 unclassified** | **holds** |
| …and no row is in one store only | 0 only-in-A, 0 only-in-B and 0 unmatched, in every class | **holds** |
| Inside it, every row is tabulated: $\lvert T_{26}-T_{25}\rvert$ against `total_abserr` in each, and the same on the oracle; region counts, maximum depths and `total_converged` in each; the rows whose `total_converged` changes, in each direction | 864 of 864 rows, in `out/qsi_rows.csv`. The 48 changed rows (all unconverged → converged; 0 the other way) are listed in `out/qsi_tabulation.txt` | **done** |
| **R1 — consistency.** Every row outside 06a's 720 stationary-point rows with finite `total_abserr` in both stores has $\lvert T_{26}-T_{25}\rvert \le e_{25}+e_{26}$, and the same on the oracle. Rows with an infinite error in either are counted separately; the stationary-point rows are tabulated, not scored. Holds if no row violates either | **On `total`:** 0 stationary-point rows; 864 scored; **0 violate**; largest ratio $4.93\times10^{-6}$; 0 non-finite. **On the oracle:** 864 scored; **0 violate**; largest ratio $6.68\times10^{-8}$; 0 non-finite | **holds** |
| The readers run on R26, exit 0, and the store is byte-identical before and after | 6 of 6 exit 0. The 5 files are identical by SHA-256, size and mtime; no journal | **holds** |
| The lock is released, and `ray-processes` lists nothing | released, exit 0; `status` free; `ray-processes` empty before and after the release | **holds** |
| The stores are kept | both stores in place, unchanged since the comparison | **holds** |

**§6 item 2 (acceptance):** "R25 and R26 both `done` with `1b332ffa…`; outside `QuadSourceIntegral`
every value table identical; R1 scored; the readers exit 0; the lock released." Every clause is met,
and R1 holds.

**R1's numbers** (`out/qsi_tabulation.txt`):
- **On `total`:**
  - The largest $\lvert\Delta T\rvert/(e_{25}+e_{26})$ is $4.93\times10^{-6}$.
  - The largest $\lvert\Delta T\rvert/e_{25}$ is $9.870\times10^{-6}$, and the largest
    $\lvert\Delta T\rvert/e_{26}$ is $9.869\times10^{-6}$. Both are at $k=10^6$, $q=10^5$, $r=10^6$,
    $z=573.11465$: `total` `0x1.570fd96dfd9b4p-69` → `0x1.570fd96dff072p-69`, `total_abserr`
    `0x1.1941f1bdc138fp-92` → `0x1.194b57be74495p-92`.
  - The distribution over the 864 rows: p50 $3.3\times10^{-9}$, p90 $2.9\times10^{-7}$, p99
    $2.2\times10^{-6}$, p100 $4.9\times10^{-6}$.
  - $\lvert\Delta T\rvert/\lvert T_{25}\rvert$ is at most $2.3\times10^{-10}$.
- **On the oracle:**
  - The largest $\lvert\Delta\,{\rm rad}\rvert/(e_{25}+e_{26})$ is $6.677\times10^{-8}$, at
    $k=q=r=10^6$, $z=996.42016$: `analytic_rad` `-0x1.b497f0c217becp-67` → `-0x1.b497f0c217c0bp-67`,
    `analytic.abserr` `0x1.bac05c053b652p-92` → `0x1.bac05e6bcc796p-92`.
  - The largest $\lvert\Delta\,{\rm rad}\rvert/e$ in either store is $1.335\times10^{-7}$.

### Where the region counts, maximum depths and the oracle are stored (§4)

`ComputeTargets/QuadSourceIntegral.py` (`962424c0…`) is the same file at `5ff8a5e4` and `38f6a828`,
and the same as at 06g's `4f4f9ed` and `1c5336e`, so 06g's reading holds unchanged. The script
re-checks it by assertion on every row of both stores.

Everything is in each shard's table `QuadSourceIntegral`:
- **Region count, the row total:** column `WKB_Levin_num_regions`. SQLAlchemy `Integer`, SQLite
  integer, nullable.
  - It is the **sum** of the calls' `num_regions`, over every `adaptive_levin_sincos` call of the
    row's WKB Levin sub-intervals (`_aggregate_LevinData`).
  - It is NULL when the row made no WKB Levin call (291 rows in both stores).
- **Region count, per call:** column `metadata`, a JSON string written by `json.dumps`, at
  `WKB_Levin.subintervals[i].groups[j].regions` (a JSON int).
  - Each group there also stores `value`, `abserr`, `converged`, `phase_limited`,
    `simple_regions`, `evaluations` and `elapsed`.
  - The script asserts that the `regions` sum to `WKB_Levin_num_regions` on every row of both
    stores.
- **Maximum depth:** column `WKB_Levin_max_depth`, `Integer`, nullable. It is the **maximum** of
  the calls' `max_depth`, over the same calls. **No per-call depth is stored.**
- **`total_converged`:** a column, `Boolean`, SQLite integer 0/1, nullable. It is the AND of every
  WKB Levin call's `converged`. The script asserts that it equals the AND of the stored group flags.
- **`total`** and **`total_abserr`:** columns, `Float(64)`.
- **The analytic oracle's value:** column `analytic_rad`, `Float(64)`.
- **The oracle's error:** `metadata` at `analytic.abserr`, a JSON number (`json.dumps` of a Python
  float). The flag `analytic.converged` (JSON bool) sits beside it.
- **The oracle's per-block records:** `metadata` at `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data`.
  - Each holds `regions` (a JSON int, `len(regions)` of the block), `abserr` (JSON float),
    `converged` and `phase_limited` (JSON bool), and `elapsed` (a timing).
  - **No depth and no per-block value is stored for them.**
- **06m's yardstick $\nu_B$ is not stored,** nor is any per-leaf trust record (prompt 06m §0.4).

**Work counters, not values** (06b's, 06c's, 06e's, 06h's and 06g's classification, kept):
- **The columns:** `WKB_Levin_num_regions`, `WKB_Levin_evaluations`, `WKB_Levin_simple_regions`,
  `WKB_Levin_SVD_errors`, `WKB_Levin_order_changes`, `WKB_Levin_chebyshev_min_order`,
  `WKB_Levin_max_depth`, `numeric_quad_compute_steps` and `numeric_quad_RHS_evaluations`.
- **The metadata leaves:** `regions`, `simple_regions` and `evaluations`.
- **Why they are counters.** They count the partition the driver chose and the work it did. None
  of them is the integral or a bound on its error.
  - 06m moves the partition (§0.3: only in calls where some leaf is trusted under $\nu_B$ and not
    under $\phi_P$).
  - **Checked:** no `+`/`-` line of `git diff 5ff8a5e4 38f6a828 --
    AdaptiveLevin/levin_quadrature.py` names `num_regions`, `num_simple_regions`, `evaluations`,
    `max_depth`, `num_SVD_errors`, `num_order_changes` or `chebyshev_min_order`. `ComputeTargets/`
    is unchanged. So no counter changes its meaning.
  - They are compared and tabulated, region counts and depth included, but a difference in one is
    not a value difference.

**Not compared:**
- timings: columns ending `compute_time`, `RHS_time` or `elapsed`, and the metadata leaves
  `elapsed` and `compute_time`;
- the label;
- store bookkeeping: `serial`, every `*_serial`, `timestamp`. Rows are paired by physical key.

**Everything else is a value**, including `total_converged`, `total_phase_limited`, and every
`converged`, `phase_limited`, `value` and `abserr` leaf.

**The stationary-point rows.** 06a's 720 are identified as 06g's rehearsal agent identified them
(and 06c's, 06e's and 06h's), from 06a's record rather than by re-running `spscan.py`.
- **The rule:** `analyse06c.py`'s `scan06a` rule, applied to the 1e-7 records of 06a's census
  checkpoint (`var/runs/levin-phase-floor-06a-census-20261027T023143/checkpoint.jsonl`,
  `c13685b4…`). A row is a stationary-point row if its `scan` has a call with
  `n_sign_changes > 0`; the script asserts there are 720 of them.
- **The mapping:** each is mapped to its physical key (k, q, r, z_response) through 06a's
  `state/rows.json` (`aa523d28…`).
- **The rehearsal store:** a row is among them iff its own physical key is equal, as floats, to
  one of theirs.
  - **0 of the 864 rows are.**
  - No rehearsal row equals any A3 v2 row. They share only $k=10^5$ and no $z_{\rm response}$.

### Deviations from the prompt

1. **R25 against R24: 05's classifier reports the labels as UNCLASSIFIED, because the job names
   differ.** STRUCTURALLY REQUIRED.
   - **The cause.** The brief's `--job-name levin-phase-floor-06m` must differ from R24's
     `levin-phase-floor-06g`, and `main.py` puts the job name at the head of every label. 05's
     `classify_non_qsi.py` calls a label difference a `label` only if the two labels are equal once
     the trailing stamp is removed.
   - **Read literally,** the unchanged classifier on the raw JSON reports 5,148 label differences
     outside `QuadSourceIntegral` as UNCLASSIFIED (`out/classify_non_qsi_r24_r25.txt`).
   - **Read correctly,** every one of the 6,012 differing labels (`QuadSourceIntegral`'s 864
     included) differs only by that prefix and the stamp.
     - `normalise_jobname_labels.py` rewrites side a's prefix, asserting both prefixes, into a copy
       of the JSON.
     - The classifier, unchanged, then reports 0 unclassified
       (`out/classify_non_qsi_r24_r25_labels_normalised.txt`).
   - **The verdict is the same either way:** every difference is a label. The orchestrator
     confirmed the reading at planned stop 1.
   - **R26 against R25** shares one job name, and the classifier ran on its raw JSON.
2. **R25 against R24 inside `QuadSourceIntegral`: a new identity script.** IMPLEMENTATION CHOICE.
   - `qsi_identity_r24_r25.py` keeps 06g's reader and field classification verbatim. It compares
     every column and every flattened `metadata` leaf of all 864 rows by `float.hex` (floats) and by
     type and value (others). The brief asked how identity inside `QuadSourceIntegral` is
     established.
   - **As corroboration,** the tool's own `QuadSourceIntegral` report was read for the `metadata`
     keys it found differing: only `elapsed` (13,177) and `compute_time` (1,728)
     (`out/qsi_tool_metadata_keys_r24_r25.txt`).
3. **06g's rehearsal `verification.md` was read in two parts,** the second after turn 1's
   hand-back. UNINTENDED DRIFT.
   - The brief said to read it in full before anything else.
   - The first 3,000 bytes were read before the launch, and the rest after it.
   - No action depended on the unread part.
4. **The tabulation is 06g's, adapted.** IMPLEMENTATION CHOICE.
   - **The script:** `tabulate_qsi_signedyardstick.py`, from 06g's `tabulate_qsi_leafvalues.py`.
   - **The changes:**
     - the names R23/R24 → R25/R26 and V1 → R1 (by `sed`), and the suffixes 23/24 → 25/26 (one
       regex, 69 replacements);
     - the header and the counters' paragraph restated for 06m;
     - a provenance header printed: the script, the store files, the JSON, both trees' HEAD, and
       both runs' state and fingerprint;
     - the rows whose `total_converged` changes, listed in each direction (the prompt asks for
       them).
   - Nothing in the method changed. The diff is `out/tabulate_qsi_signedyardstick.diff`.
5. **06g's `qsi_analytic_summary.py` and `v1_violations_detail.py` were not copied.**
   IMPLEMENTATION CHOICE.
   - The brief allows but does not require them.
   - The oracle's counts (blocks, regions, phase-limited, converged) and R1's oracle half are in
     `qsi_tabulation.txt`, and every row's oracle fields are in `qsi_rows.csv`.
   - There are no violations to break down.
6. **`classify_non_qsi.py`'s outputs and the readers' output carry no provenance header of their
   own.** IMPLEMENTATION CHOICE.
   - The classifier must stay byte-identical to 05's, and the readers' wrapper to 06g's.
   - Their SHA-256s, their inputs' SHA-256s and the trees are recorded in `commands.log` and in the
     outputs' neighbours (`normalise_jobname_labels.txt`, `qsi_tabulation.txt`, `readers.txt`'s
     store-state lines).
7. **The first `final_check` command failed** (UNINTENDED DRIFT).
   - It used `<( … )` process substitution, which `rec.sh`'s `sh -c` does not support. It exited 2
     with a syntax error, before any check ran, and wrote `out/final_check.txt`.
   - It was rewritten for plain `sh` (string comparison in place of `cmp` against process
     substitution) and rerun. The rerun's record replaced the failed one in `out/final_check.txt`.
   - The failed command was read-only.
8. **`ls -d var/levin-phase-floor-07*` was run once,** in the final check, to confirm this agent
   wrote nothing there. UNINTENDED DRIFT.
   - It lists the name only (`var/levin-phase-floor-07-leaves`) and reads nothing inside.
   - The brief says never to touch anything of 07's; nothing was opened or written.
9. **`black` was not applied to the scripts.** IMPLEMENTATION CHOICE.
   - `black --check` would reformat four of them (`out/final_check.txt`). They are records, left
     as run.
   - `classify_non_qsi.py` must stay byte-identical to 05's.

### Verification performed

**The gate** (`out/00-gate-trees.txt`, `out/01-gate-registry-lock.txt`):
- **Trees:**
  - the main checkout is `28ebaf07`, with `status --short` empty;
  - `5ff8a5e4` and `38f6a828` are detached, with `status --short --ignored` empty;
  - `git worktree list` shows seven trees: the main checkout, 06g's four and 06m's two.
- **The diffs:** `48cf87c0^` is `5ff8a5e4`. `5ff8a5e4..38f6a828` is `48cf87c0`, `427e88c2`,
  `56efc6b9`, `38f6a828`. The diff stats are as stated above.
- **Registry:** 58 runs: 50 done, 4 failed, 4 killed. Nothing was `running` or stale.
- **Lock:** `~/.raypoolkit` was empty, and the old lock location was absent.
- **Host:** `~/.ray/auth_token` mode 600, 78 GB free, load about 18–22.
- **Packages:** venv 3.13.16, ray 2.59.0, SQLAlchemy 2.0.55.
- **Clean start:** no 06m store or run existed.

**R25** (`out/r25_launch_check.txt`, `out/r25_end_check.txt`, `out/r25_status.json`):
- **Launch:** at 09:24:19Z from the parent worktree (`sh` 55642, driver 55646), then
  `set-pid 55646`. Ray started at 09:24:29 with `num_cpus=6`.
- **Manifest:** `git_head` `5ff8a5e4`, `git_dirty` false, `script_sha256` `b0135218…`. The driver's
  cwd is the parent worktree.
- **End:** `done`, exit 0, heartbeat 09:36:10Z. Fingerprint `1b332ffa…`, taken at `5ff8a5e4`,
  `problems {}`. Its 21 `classes` equal R24's and R23's.
- **Logs:** no Traceback and no `!! WARNING` in the launch log, `stdout.log` or `stderr.log`
  (empty). The launch log ends "** driver exit status 0".
- **Wall time:** about 12 min.

**R25 against R24** (cwd the code worktree, so `-m` imports its tool):
- **The tool:** `tools/store_value_diff.py` (`7997af85…`), which opens every file through
  `open_read_only`. It exited 1 in 17.8 s.
- **Outside `QuadSourceIntegral`:** all 7 value tables are identical (360,462 rows), and every
  difference is classified, as in deviation 1.
- **Inside it** (`out/qsi_identity_r24_r25.txt`):
  - **Structure:** 864 rows in each, equal key sets, 48 columns each, equal field sets, and equal
    tolerances.
  - **Compared:** 118,776 value fields and 27,163 counter fields, with **0 differences**.
  - **Bit-identical in 864 / 864:** `total`, `total_abserr`, `analytic_rad`, `analytic.abserr` and
    `total_converged`.
  - **What differs:** timing (18,733 fields), `label` (864), and serials and timestamps (5,325;
    store bookkeeping, which the tool resolves and does not report).
- **The stores** (R24 and R25) are byte-identical before and after
  (`state/r24_r25_store_sha256_{before,after}_diff.txt`). R24's five files equal 06g's
  `stores_sha256_final.txt`.

**R26** (`out/r26_launch_check.txt`, `out/r26_end_check.txt`, `out/r26_status.json`):
- **Launch:** at 09:39:36Z from the code worktree (`sh` 57109, driver 57112), then `set-pid 57112`.
  Ray started at 09:39:46.
- **Manifest:** `git_head` `38f6a828`, `git_dirty` false, `script_sha256` `b0135218…`. The driver's
  cwd is the code worktree.
- **End:** `done`, exit 0, heartbeat 09:49:23Z. Fingerprint `1b332ffa…`, taken at `38f6a828`,
  `problems {}`. Its 21 `classes` equal R25's, R24's and R23's. `QuadSourceIntegral` holds 864
  rows.
- **Logs:** no Traceback and no `!! WARNING` in either launch log, R26's `stdout.log` or its
  `stderr.log`.
- **Trees:** both worktrees are clean, with nothing newer than R25's marker.
- **The registry** holds 60 runs, all finished.
- **Wall time:** about 10 min, against R25's 12 min (descriptive; cost is not an argument).

**R26 against R25** (cwd the code worktree):
- **The tool** exited 1, because `QuadSourceIntegral` differs, in 17.6 s.
- **Outside `QuadSourceIntegral`** (`out/classify_non_qsi.txt`): all value tables are identical
  (1,159 + 1,478 + 2,209 + 35,874 + 147,642 + 169,653 + 2,447 = 360,462 rows), and every difference
  is classified, as above.
- **The stores** (R24, R25 and R26, 15 files) are byte-identical before and after the comparison and
  the tabulation (`state/store_sha256_before_diff.txt`, `…_after_diff.txt`).

**The tabulation** (`out/qsi_tabulation.txt`):
- The stores' key sets are equal, with 48 columns each and the tolerances (1e-32, 1e-7) equal row
  by row.
- The tool reports 864 rows differing, 0 only in one store and 0 unmatched. Its keys equal the
  stores' set.
- The cross-check of `total` against the tool finds 0 mismatches.
- The field sets of every row are equal in both stores.
- The WKB Levin call count is 3,105 in both, equal row by row.

| | R25 | R26 |
|---|---|---|
| `total` bit-identical (`float.hex`) | 413 / 864 (291 with no WKB Levin call + 122) | |
| `total_abserr` bit-identical | 403 / 864 | |
| `analytic_rad` bit-identical | 860 / 864 | |
| `analytic.abserr` bit-identical | 848 / 864 | |
| every value field bit-identical | 402 / 864 (291 with no WKB Levin call + 111) | |
| `total_converged` | 797 | 845 |
| WKB Levin calls unconverged | 97 | 20 |
| `total_phase_limited` | 478 | 464 |
| WKB Levin calls phase-limited | 1,411 | 1,322 |
| `WKB_Levin_num_regions`, summed | 134,011 | 114,826 |
| `WKB_Levin_evaluations`, summed | 1,068,983 | 915,503 |
| rows at `WKB_Levin_max_depth` = 20 | 76 | 23 |
| analytic-block regions, summed | 29,228 | 29,199 |
| analytic blocks converged | 10,072 / 10,072 | 10,072 / 10,072 |
| analytic blocks phase-limited | 3,768 | 3,762 |

- **`total_converged`, R25 → R26:** 797 stayed converged, 19 stayed unconverged, **48 went
  unconverged → converged, and 0 converged → unconverged.**
  - All 48 had `WKB_Levin_max_depth` 20 in R25, with 1–4 unconverged calls each (73 in all, 0 in
    R26). In R26 their
    maximum depth is 11–20, and only 2 of them are still at 20.
  - Each satisfies R1, the largest at $4.8\times10^{-6}$.
  - They are listed row by row in `qsi_tabulation.txt`, with both totals and both errors in hex.
- **`total_phase_limited`:** went true → false in 14 rows, and never false → true.
- **`total_abserr`, R26 against R25:** rose in 445 rows and fell in 16. The ratio is 1.00 to 3
  significant figures in every row.
- **Regions per row,** R26/R25: p0 0.385, p50 0.929, p100 1. Fewer in 442 rows, equal in 131, more
  in 0.
- **Maximum depth,** R25 → R26: equal in 298 rows, shallower in 275, deeper in 0.

**The readers** (`out/readers.txt`, `out/readers_check.txt`):
- They ran on R26 from the code worktree, under the lock, through 06g's wrapper unchanged.
- **Exits:** six of six exit 0, with 0 Tracebacks.
- **The store:** six "identical before and after" lines, and R26's hash files are equal before and
  after the reads (`state/r26_store_sha256_{before,after}_reads.txt`).
- **Warnings:** six `!! WARNING … shards (=4) … (=10)` lines, as 06g's.
- **Ray:** no Ray process from this checkout's venv is left after any reader.

**The lock** (`state/lock.log`), ten commands:
1. `ray-processes` (empty)
2. `acquire`
3. `set-pid 55646`
4. `ray-processes` (empty)
5. `set-pid 57112`
6. `ray-processes` (empty)
7. `ray-processes` (empty)
8. `release`
9. `status` (free, Ray processes none)
10. `ray-processes` (empty)

Each exited 0, and each was a command of its own, appended by `>>`.

**Final** (`out/final_check.txt`):
- the store hashes are equal to those after the comparison;
- R23 and R24 equal 06g's final hashes (R23's five files were read only by `shasum`, here);
- 60 runs, all finished, and `~/.raypoolkit` is empty;
- the main checkout and both worktrees are clean and unmoved, with nothing newer than R25's marker;
- there is no `__pycache__` in the records.

### Observations not acted on

1. **06m gains convergence in the rehearsal and loses none.** This is descriptive, not scored.
   - **What moved.** 48 rows (73 WKB Levin calls) went unconverged → converged; 4 more calls
     converged in rows that stay unconverged, so unconverged calls fell by 77 (97 → 20). Every one had
     reached `depth_max` in R25. Rows at depth 20 fell from 76 to 23, and regions fell 14 %.
   - **Against 06g's rehearsal.** 06g's R24 had lost 67 rows' convergence against R23, all at depth
     20 (06g rehearsal, observation 2). Here 48 of those losses are recovered.
   - **The size of the moves.** Every one stays within R1: the largest
     $\lvert\Delta T\rvert/(e_{25}+e_{26})$ is $4.9\times10^{-6}$. The values move at the
     $10^{-10}$ level relative, and the declared errors by under 1 %.
   - **What it fits.** This is §0.3's picture: B trusts slow leaves at noise level that 06g's
     $\phi_P$ refused.
2. **19 rows (20 calls) stay unconverged in R26,** 23 rows reach depth 20, and 2 of the 48 rows that
   now converge still reach depth 20.
   - The rehearsal stores no per-leaf data, so these cannot be attributed here to an untrusted
     leaf at `depth_max` or to anything else.
   - The census (§5) records the calls, and §0.4's proposed issue
     `[06m-calls-flagged-unconverged-within-their-request]` is about such calls.
3. **05's classifier and job names.** `classify_non_qsi.py` classes a label difference as `label`
   only when both runs share a job name.
   - A cross-prompt comparison needs the job-name prefix normalised first (deviation 1). 06g's
     R23/R24 shared one, so the case had not arisen.
   - Recorded in `issues.md`, not opened.
4. **The readers' file counts differ from 06g's.** `extract_Gk_data` wrote 450 files against 432,
   and `extract_GkSource_data` 152 against 186. This is the unseeded `random.sample` 06g recorded,
   not a store difference.
5. **4 rows' `analytic_rad` moved,** by at most $6.7\times10^{-8}$ of the two errors' sum, and 16
   rows' `analytic.abserr`.
   - The oracle's three-Bessel Levin blocks also run through `adaptive_levin_sincos`, so the
     changed trust test reaches them.
   - All 10,072 blocks converge in both stores.

### State handed on

- **R25** `levin-phase-floor-06m-r25-parent-20261106T092420` and **R26**
  `levin-phase-floor-06m-r26-signedyardstick-20261106T093937`: both `done`, `1b332ffa…`.
- **Their stores are kept,** unchanged since the comparison (`state/stores_sha256_final.txt`):
  - `var/datastores/levin-phase-floor-06m-r25*`, primary `d74ac31d…`;
  - `var/datastores/levin-phase-floor-06m-r26*`, primary `e90e26c2…`.
- **R24** (06g's, read only) is unchanged; R23 was only hashed, in the final check, and equals 06g's final hashes.
- **The lock is released.** `~/.raypoolkit` is empty, and `ray-processes` lists nothing. The six
  cores are free for §5.
- **The trees.** The main checkout and both worktrees are clean and unmoved. Nothing was written in
  either worktree.
- **Kept on the Mac Studio, not for copying back:**
  - the three diff JSONs (41.1, 40.1 and 26.9 MB);
  - `extract/` (179 MB).

  See `KEPT_ON_MAC_STUDIO.txt`.
