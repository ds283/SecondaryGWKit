### 5.x Rehearsals R9 and R10, and the value comparison (prompt §5)

**Prompt 05 §5, run 2026-10-26, 01:46–02:10 UTC, on the Mac Studio (mps026298, arm64, Darwin
27.0.0).** The records were written in the main checkout at `53084dc` (`handover-remedial`, clean
throughout). Its code is the code commit `120fdfe`'s. Python 3.13.16 from the main checkout's
venv: `ray` 2.59.0, `SQLAlchemy` 2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
0.2.0. The load average was about 5–7 on 10 cores.

**Records** are in [`05/rehearsal/`](05/rehearsal/): `out/`, `scripts/` and `state/`, the agent's
`log.md`, this section as first written, and `issues.md`. Every command is in
[`out/commands.log`](05/rehearsal/out/commands.log). The scripts' SHA-256 are in
[`out/script_sha256.txt`](05/rehearsal/out/script_sha256.txt). The readers' `extract/` was not
copied back.

#### The two trees

| | R9 (before) | R10 (after) |
|---|---|---|
| tree | worktree `SecondaryGWKit-lpf05-e648951`, detached at `e648951` (the code commit's parent) | main checkout at `53084dc` (code = `120fdfe`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `PYTHONDONTWRITEBYTECODE=1` | R8's launch as written: `PYTHONPATH=.`, `./venv/bin/python` |
| run | `levin-phase-floor-05-r9-parent-20261026T014729` | `levin-phase-floor-05-r10-reanchor-20261026T015120` |
| `sh` / driver pid | 47763 / 47770 | 49749 / 49752 |
| wall | 01:47:29 → 01:50:23 | 01:51:20 → 01:54:10 |
| manifest | `git_head` `e6489519…`, `git_dirty` false, cwd the worktree | `git_head` `53084dcb…`, `git_dirty` false, cwd the main checkout |
| launcher SHA-256 | `5b984c75…` (R8's) | `5b984c75…` |
| end | `done`, exit 0, no Traceback, `stderr.log` empty | `done`, exit 0, no Traceback, `stderr.log` empty |
| store | `var/datastores/levin-phase-floor-05-r9.sqlite` + 4 shards | `var/datastores/levin-phase-floor-05-r10.sqlite` + 4 shards |

Both builds were R8's rehearsal scope: 3 k over 1e5–1e6/Mpc, zend 250, atol 1e-32, rtol 1e-7, 6
CPUs, the same `--job-name levin-phase-floor-05`, and `--campaign levin-phase-floor --prompt 05`.

At R9's launch, `lsof` showed the cwd of the driver and of a Ray `Datastore` actor to be the
worktree ([`out/r9_launch_check.txt`](05/rehearsal/out/r9_launch_check.txt)). After both runs, the
worktree's `git status --short --ignored` was empty, and `find -newer` listed nothing in it.

#### The lock

[`out/lock.log`](05/rehearsal/out/lock.log). Each command was a shell command of its own, output by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 05, rehearsals R9 … and R10 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid … 47770` (R9) | recorded | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after R9) | nothing | 0 |
| 4 | `set-pid … 49752` (R10) | recorded | 0 |
| 5 | `status --ray-temp-dir /tmp/ray-sgk` (while stopped for the question below) | HELD; owner 49752 not alive; no Ray processes | 0 |
| 6 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after the readers) | nothing | 0 |
| 7 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 8 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |

#### The fingerprints

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R9** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 digests equal R8's |
| **R10** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 equal R8's; the whole `classes` object equals R9's |

The fingerprints are the ones the launcher took at the end of each build, read from `status.json`
([`out/r9_end_check.txt`](05/rehearsal/out/r9_end_check.txt),
[`out/r10_end_check.txt`](05/rehearsal/out/r10_end_check.txt)). There is no sidecar: `--build`
writes none.

#### The value comparison (prompt §5.4)

`PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff var/datastores/levin-phase-floor-05-r9.sqlite var/datastores/levin-phase-floor-05-r10.sqlite --json`,
run in the main checkout at `53084dc`. It exited **1** in 16 s. The text form, `--max-rows 1000000`,
also exited 1. Outputs: [`out/store_value_diff.json`](05/rehearsal/out/store_value_diff.json),
[`out/store_value_diff.txt`](05/rehearsal/out/store_value_diff.txt),
[`out/store_value_diff_summary.txt`](05/rehearsal/out/store_value_diff_summary.txt).

- **Structure.** 21 classes were compared, and `not_compared` is empty. In every class, 0 rows are
  in one store only, 0 rows cannot be matched by identity and 0 columns are unresolvable.
- **Every value table of every class is bit-identical between R9 and R10**, as the tool's JSON
  shows:

  | value table | rows identical | differ / only in one / unmatched |
  |---|---|---|
  | BackgroundModelValue | 1159 | 0 / 0 / 0 |
  | TkNumericValue | 1478 | 0 / 0 / 0 |
  | TkWKBValue | 2209 | 0 / 0 / 0 |
  | GkNumericValue | 35874 | 0 / 0 / 0 |
  | GkWKBValue | 147642 | 0 / 0 / 0 |
  | GkSourceValue | 169653 | 0 / 0 / 0 |
  | QuadSourceValue | 2447 | 0 / 0 / 0 |

- **Row columns outside `QuadSourceIntegral`.** 12 classes are reported value-identical. They are
  `version`, `store_tag`, `redshift`, `wavenumber`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`,
  `IntegrationSolver`, `GkSourcePolicy`, `QuadSourcePolicy`, `GkSourcePolicyData` and
  `OneLoopIntegral`, which has no rows. Eight are reported not value-identical, in their row
  columns only. [`scripts/classify_non_qsi.py`](05/rehearsal/scripts/classify_non_qsi.py) assigns
  every one of their differing columns to one of four kinds, and **0 are left unclassified**
  ([`out/classify_non_qsi.txt`](05/rehearsal/out/classify_non_qsi.txt)):

  | kind | columns | classes (rows) |
  |---|---|---|
  | wall-clock timing | `compute_time`, `stage_1_compute_time`, `{min,mean,max}_RHS_time` | wavenumber_exit_time (3), BackgroundModel (1), TkNumericIntegration (3), TkWKBIntegration (3), GkNumericIntegration (1368), GkWKBIntegration (3474), QuadSource (6) |
  | `label` | `<job>-…-<datetime.now()>`; equal once the trailing timestamp is removed | TkNumeric (3), TkWKB (3), GkNumeric (1368), GkWKB (3477), GkSource (291), QuadSource (6) |
  | `GkSource.parents_digest` | see below | GkSource (291 of 291) |
  | GkWKB work counters | `metadata` keys `rho_evals`, `rho_reused`, and `stage_1_RHS_evaluations` | GkWKBIntegration (25 of 3477) |

  **`parents_digest`.** It is the SHA-256 (hex) of the compact JSON
  (`json.dumps(…, separators=(",", ":"))`) of the GkSource's parent set. That set is the list of
  triples (z_source **serial**, GkNumericIntegration **serial** or None, GkWKBIntegration
  **serial** or None), sorted by z_source serial (`ObjectFactories/GkSource.py`
  `GkSource_parents_digest`; `ComputeTargets/GkSource.py` `GkSource_parent_set`). It is a function
  of row serials, and of nothing else.
  [`scripts/parent_serials.py`](05/rehearsal/scripts/parent_serials.py)
  ([`out/parent_serials.txt`](05/rehearsal/out/parent_serials.txt)) checks this on the stores:
  - the redshift serials are the same in both stores (0 of 1159 differ);
  - the GkNumericIntegration serials differ in 1346 of 1368 rows, and the GkWKBIntegration serials
    in 3466 of 3477. Rows were paired by label without its timestamp;
  - in each store, recomputing the digest from that GkSource's own `GkSource_parents` rows
    reproduces the stored column for 291 of 291 rows.

  The tool matched every GkSource row by identity, so the parent sets agree by identity, and only
  the serials differ.

  **The GkWKB work counters**
  ([`scripts/gkwkb_reuse_rows.py`](05/rehearsal/scripts/gkwkb_reuse_rows.py),
  [`out/gkwkb_reuse_rows.txt`](05/rehearsal/out/gkwkb_reuse_rows.txt)). In each of the 25 rows,
  one store reused the per-process residual table (`rho_reused` true, `rho_evals` 0 or 4) and the
  other built it (`rho_evals` 4632–4636). The swaps go both ways within every $k$: 11 rows were
  reused in R9 only, and 14 in R10 only. No other metadata key differs in those rows. Every
  `GkWKBValue` row, and every other GkWKB value column (`sin_coeff`, `cos_coeff`, `G_init`, …), is
  identical.

- **The user's decision (2026-10-26), relayed:** "Those columns cannot be expected to be identical.
  So I think that's just an incorrectly-worded prompt." The four kinds above are therefore not
  prompt §8's stop. A difference outside `QuadSourceIntegral` of any other kind would still be a
  stop, and there is none.

#### `QuadSourceIntegral`, tabulated per row

[`scripts/tabulate_qsi.py`](05/rehearsal/scripts/tabulate_qsi.py) reads both stores (`mode=ro`) and
the tool's JSON. Its outputs are [`out/qsi_tabulation.txt`](05/rehearsal/out/qsi_tabulation.txt)
and one row per integral in [`out/qsi_rows.csv`](05/rehearsal/out/qsi_rows.csv). A = R9 (parent),
B = R10 (code).

- Rows were paired by physical (k, q, r, z_response), giving 864 = 864. The pairing has the same
  key set as the tool's 864 identity-matched rows, and gives 0 mismatches in `total`'s `float.hex`
  against the tool.
- All 864 rows differ in some column. Wall-clock columns differ in every row, and `label` in
  every row.

| | rows |
|---|---|
| `total` differs | **542** of 864 (322 bit-identical) |
| `WKB_Levin` differs | 542, exactly the rows where `total` differs |
| `numeric_quad` / `analytic_rad` differ | **0** / **0** |
| `total_abserr` differs | 544 |
| rows with no `WKB_Levin` part (`max_depth` None) | 291, none of which differs in `total` |

Over the 542 rows whose `total` differs:

| statistic | median | 90 % | max |
|---|---|---|---|
| $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}_A\rvert$ | 1.43e-11 | 7.02e-09 | 4.03e-05 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ | 8.71e-05 | 3.62e-02 | 28.2 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_B$ | 8.71e-05 | 3.66e-02 | 29.4 |

- Rows with $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ > 0.1: 22. Above 1: **one**, at
  $(k,q,r,z_{\rm resp}) = (10^6, 3.162\times10^5, 10^6, 329.64)$. There the ratio is 28.2, the
  relative change 4.0e-5, the depth 6 → 7, the regions 36 → 39, and `total_converged` is 0 in both.
- **`total_converged`:** no row changes (0→0: 355, 1→1: 509). **`total_phase_limited`:** no row
  changes (0→0: 833, 1→1: 31).
- **`WKB_Levin_max_depth`:** 15 rows change: 6→7 ×5, 7→6 ×3, 8→7 ×5, 8→6 ×1, 14→12 ×1. No row
  is at depth ≥ 15 in either store.
- **`WKB_Levin_num_regions`:** 42 rows change, with the total going from 13,726 to 13,652. The
  largest falls are on $(10^6, 10^5, 10^6)$: 79 → 55, 67 → 50, 63 → 56 and 60 → 53.
- **By depth under A**, the median $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ rises with
  depth: 3e-5 to 9e-5 at depths 5–7, 4.8e-4 at depth 8, and 1.2e-2 to 6.4e-2 at depths 10–14.
- **Metadata.** The non-wall-clock keys that differ are the `WKB_Levin` `value`/`abserr` tree:
  `WKB_Levin.abserr` in 544 rows, `subintervals[i].value` and `.abserr`, and
  `subintervals[i].groups[j].value` and `.abserr`. In a few rows `groups[j].regions`,
  `.simple_regions` and `.evaluations` also differ. There are 41 distinct keys, in 546 rows. In the
  other 318 rows only wall-clock keys (`…elapsed`, `…compute_time`) differ.

These are recorded, not judged. Prompt §6 scores the effect on the depth-20 rows of A3 v2.

#### The readers (prompt §5.5)

`bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/levin-phase-floor-05-r10.sqlite var/levin-phase-floor-05/rehearsal/extract`,
with the lock held, from 02:06:12 to 02:08:36 UTC ([`out/readers.txt`](05/rehearsal/out/readers.txt)).
Overall exit **0**.

| reader | exit | wall | lookups | files written (R8) | store files after |
|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 26 s | 116 | 450 (531) | identical |
| `extract_GkWKB_data.py` | 0 | 48 s | 348 | 2436 (2436) | identical |
| `extract_GkSource_data.py` | 0 | 19 s | 10 | 142 (160) | identical |
| `extract_TkWKB_data.py` | 0 | 16 s | 3 | 27 (27) | identical |
| `extract_tensor_source_data.py` | 0 | 14 s | 6 | 27 (27) | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 15 s | 24 | 74 (76) | identical |

- The lookups equal R8's.
- File counts differ only where a reader draws an unseeded subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader, 0 Ray processes of this venv were left, and no reader log has a Traceback.

**R10's store, before and after the readers**
([`state/r10_store_sha256_before_reads.txt`](05/rehearsal/state/r10_store_sha256_before_reads.txt),
[`state/r10_store_sha256_after_reads.txt`](05/rehearsal/state/r10_store_sha256_after_reads.txt);
`diff` empty; no `-journal`, `-wal` or `-shm` file):

```
557066beae8bf5e5416947f07687cd33771e126c31407fef2016ec826afb50cc  levin-phase-floor-05-r10.sqlite            36864
ee4552a062a0adbd7c192ffc79a0d9e33527df8a6acd6e2405054d891173062e  levin-phase-floor-05-r10-shard0000.sqlite  24076288
83c0e9b27b76d479495a19388f23dc65d864254fd28aa3a8966432666cb7cadb  levin-phase-floor-05-r10-shard0001.sqlite  24289280
5b80e1bda653a68e1c3e5d1b1f877e9f5d0c723f8555cee52892a8b6e9a96689  levin-phase-floor-05-r10-shard0002.sqlite   1077248
9b90fb154fd8051e0b3ffe2cd07fa0a484d436deb496808a400929c171a8f8d6  levin-phase-floor-05-r10-shard0003.sqlite  23543808
```

**Both stores**, all ten files, are byte-identical from before the comparison to the end
([`state/store_sha256_before_diff.txt`](05/rehearsal/state/store_sha256_before_diff.txt),
[`state/stores_sha256_final.txt`](05/rehearsal/state/stores_sha256_final.txt)). R9's equal its
hashes taken just after its build
([`state/r9_store_sha256_after_build.txt`](05/rehearsal/state/r9_store_sha256_after_build.txt)).
R9's primary is `cfba7ac4c218b7c25f0fef9114537b556530b5f71c6b96c6699818cc5fb7d2de`.

#### Against prompt §7.2

- R9 and R10 both end `done` with fingerprint `1b332ffa…`: **yes**.
- Every non-`QuadSourceIntegral` value is identical: **yes, on the user's reading.** Every value
  table is bit-identical. The row-column differences are only wall-clock timings, timestamped
  labels, the serial-derived `parents_digest`, and the GkWKB residual-reuse counters.
- The `QuadSourceIntegral` differences are tabulated: **yes**. 542 of 864 `total`s moved, all
  through `WKB_Levin`, with `numeric_quad` and `analytic_rad` untouched. R9 and R10 are not
  value-identical in `QuadSourceIntegral`, so each tree ran the code it names.
- The readers exit 0: **yes**, and the store is byte-identical.
- The lock is released: **yes**.
