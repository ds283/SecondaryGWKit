### 5a.x Rehearsals R11 and R12, and the value comparison (prompt 05a §5)

**Prompt 05a §5, run 2026-10-26, 12:20–12:33 UTC, on the Mac Studio** (mps026298, arm64, Darwin
27.0.0).
- **The checkout.** The records were written in the main checkout at `149c393`
  (`handover-remedial`, clean throughout). Its code is the code commit `1fc349c`'s, up to
  `bd38e50`'s comment-only changes.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit` 0.2.0.
- **The load** was about 5–6 on 10 cores. Prompt §6's Ray-free re-measurement ran beside the
  readers, from 12:30:47.

**Records** are in [`05a/rehearsal/`](05a/rehearsal/): `out/`, `scripts/` and `state/`, the
agent's `log.md`, this section as first written, and `issues.md`.
- Every command is in [`out/commands.log`](05a/rehearsal/out/commands.log).
- The scripts' SHA-256 are in [`out/script_sha256.txt`](05a/rehearsal/out/script_sha256.txt). All
  four (`classify_non_qsi.py`, `tabulate_qsi.py`, `tools/store_value_diff.py`, `run_readers.sh`)
  equal 05's.
- The readers' `extract/` was not copied back.

#### The two trees

| | R11 (before) | R12 (after) |
|---|---|---|
| tree | worktree `SecondaryGWKit-lpf05a-c4eaefd`, detached at `c4eaefd` (the code commit's parent) | main checkout at `149c393` (code = `1fc349c`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `PYTHONDONTWRITEBYTECODE=1` | R10's launch: `PYTHONPATH=.`, `./venv/bin/python` |
| run | `levin-phase-floor-05a-r11-parent-20261026T122041` | `levin-phase-floor-05a-r12-reanchor-t-20261026T122434` |
| `sh` / driver pid | 22677 / 22684 | 24894 / 24897 |
| wall | 12:20:41 → 12:23:39 | 12:24:34 → 12:27:25 |
| manifest | `git_head` `c4eaefda…`, `git_dirty` false, cwd the worktree | `git_head` `149c3938…`, `git_dirty` false, cwd the main checkout |
| launcher SHA-256 | `5b984c75…` (R8's) | `5b984c75…` |
| end | `done`, exit 0, no Traceback, `stderr.log` empty | `done`, exit 0, no Traceback, `stderr.log` empty |
| store | `var/datastores/levin-phase-floor-05a-r11.sqlite` + 4 shards | `var/datastores/levin-phase-floor-05a-r12.sqlite` + 4 shards |

**The scope.** Both builds were R8's rehearsal scope:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- the same `--job-name levin-phase-floor-05a`, and `--campaign levin-phase-floor --prompt 05a`.

**The worktree.** At R11's launch, `lsof` showed that the driver's cwd, and that of the Ray
`Datastore` actor 22808, was the worktree
([`out/r11_launch_check.txt`](05a/rehearsal/out/r11_launch_check.txt)). After both runs, the
worktree's `git status --short --ignored` was empty, and `find -newer` listed nothing in it.

#### The lock

[`out/lock.log`](05a/rehearsal/out/lock.log). Each command was a shell command of its own, output by
`>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 05a, rehearsals R11 … and R12 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid … 22684` (R11) | recorded | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after R11) | nothing | 0 |
| 4 | `set-pid … 24897` (R12) | recorded | 0 |
| 5 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after the readers) | nothing | 0 |
| 6 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 7 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |
| 8 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |

#### The fingerprints

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R11** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 digests equal R8's |
| **R12** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 equal R8's; the whole `classes` object equals R11's |

The fingerprints are the ones the launcher took at the end of each build, read from `status.json`
([`out/r11_end_check.txt`](05a/rehearsal/out/r11_end_check.txt),
[`out/r12_end_check.txt`](05a/rehearsal/out/r12_end_check.txt); the files themselves are
[`out/r11_status.json`](05a/rehearsal/out/r11_status.json) and
[`out/r12_status.json`](05a/rehearsal/out/r12_status.json)). There is no sidecar, because
`--build` writes none.

#### The value comparison (prompt §5.4)

`PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff var/datastores/levin-phase-floor-05a-r11.sqlite var/datastores/levin-phase-floor-05a-r12.sqlite --json`,
run in the main checkout at `149c393`. It exited **1**, as expected, in 17 s. The text form, with
`--max-rows 1000000`, also exited 1.

Outputs:
- [`out/store_value_diff.json`](05a/rehearsal/out/store_value_diff.json);
- [`out/store_value_diff.txt`](05a/rehearsal/out/store_value_diff.txt);
- [`out/store_value_diff_summary.txt`](05a/rehearsal/out/store_value_diff_summary.txt);
- [`out/store_value_diff.meta.txt`](05a/rehearsal/out/store_value_diff.meta.txt).

- **Structure.** 21 classes were compared, and `not_compared` is empty. In every class:
  - 0 rows are in one store only;
  - 0 rows cannot be matched by identity;
  - 0 columns are unresolvable.
- **Every value table of every class is bit-identical between R11 and R12**, as the tool's JSON
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

- **Row columns outside `QuadSourceIntegral`.** As in 05, 12 classes are reported value-identical:
  `version`, `store_tag`, `redshift`, `wavenumber`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`,
  `IntegrationSolver`, `GkSourcePolicy`, `QuadSourcePolicy`, `GkSourcePolicyData`, and
  `OneLoopIntegral`, which has no rows.
- **Eight classes differ, in row columns only.** 05's
  [`scripts/classify_non_qsi.py`](05a/rehearsal/scripts/classify_non_qsi.py), copied unchanged
  (`26690e61…`), assigns every differing column to one of the four volatile kinds of
  `[05-store-value-diff-compares-volatile-columns]`. **0 are left unclassified**
  ([`out/classify_non_qsi.txt`](05a/rehearsal/out/classify_non_qsi.txt)):

  | kind | columns | classes (rows) | column-differences |
  |---|---|---|---|
  | wall-clock timing | `compute_time`, `stage_1_compute_time`, `{min,mean,max}_RHS_time` | wavenumber_exit_time (3), BackgroundModel (1), TkNumericIntegration (3), TkWKBIntegration (3), GkNumericIntegration (1368), GkWKBIntegration (3474), QuadSource (6) | 8607 |
  | `label` | equal once the trailing `datetime.now()` stamp is removed | TkNumeric (3), TkWKB (3), GkNumeric (1368), GkWKB (3477), GkSource (291), QuadSource (6) | 5148 |
  | `GkSource.parents_digest` | the SHA-256 of the parents' serials (05 §5.x) | GkSource (291 of 291) | 291 |
  | GkWKB work counters | `metadata` keys `rho_evals`, `rho_reused`; `stage_1_RHS_evaluations` | GkWKBIntegration (31 of 3477) | 62 |

  Nothing outside `QuadSourceIntegral` differs in any other way, so prompt §8's stop does not fire.

#### `QuadSourceIntegral`, tabulated per row

05's [`scripts/tabulate_qsi.py`](05a/rehearsal/scripts/tabulate_qsi.py), copied unchanged
(`b7508351…`), reads both stores (`mode=ro`) and the tool's JSON. Its docstring still names R9 and
R10; **here A = R11 (parent `c4eaefd`) and B = R12 (code `1fc349c`)**. Its outputs are
[`out/qsi_tabulation.txt`](05a/rehearsal/out/qsi_tabulation.txt), with one row per integral in
[`out/qsi_rows.csv`](05a/rehearsal/out/qsi_rows.csv). Further counts from the CSV are in
[`out/qsi_extra_counts.txt`](05a/rehearsal/out/qsi_extra_counts.txt).

- **The pairing.** Rows were paired by physical (k, q, r, z_response), giving 864 = 864, with one
  atol and one rtol, (−32, −7). The pairing has the same key set as the tool's 864 identity-matched
  rows, and 0 mismatches in `total`'s `float.hex` against the tool.
- **All 864 rows differ in some column.** Wall-clock columns and `label` differ in every row, and
  so does `metadata` (below).

| | rows |
|---|---|
| `total` differs | **424** of 864 (440 bit-identical) |
| `WKB_Levin` differs | 435. Every row whose `total` differs is among them; in 11 more, the change rounds away in `total` |
| `numeric_quad` / `analytic_rad` differ | **0** / **0** |
| `total_abserr` differs | 476 |
| rows with no `WKB_Levin` part (`max_depth` None) | 291, none of which differs in `total` |

Over the 424 rows whose `total` differs:

| statistic | median | 90 % | max |
|---|---|---|---|
| $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}_A\rvert$ | 1.38e-15 | 2.84e-14 | 2.92e-11 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ | 8.25e-09 | 4.36e-08 | 8.81e-05 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_B$ | 8.25e-09 | 4.36e-08 | 8.81e-05 |

- Rows with $\lvert\Delta{\rm total}\rvert/$`total_abserr` > 0.1: **0** (and so 0 above 1).
- **The largest ratio** is at $(k,q,r,z_{\rm resp}) = (3.162\times10^5, 3.162\times10^5,
  3.162\times10^5, 6904.72)$. There the ratio is 8.8e-5 and the relative change 3.5e-12, at depth
  11, with 32 regions and `total_converged` 0 in both stores. The 15 largest ratios are all on the
  equal-$k$ diagonals $k = q = r = 10^5$ and $3.162\times10^5$, at depths 10–12.
- **`total_converged`:** no row changes (0→0: 355, 1→1: 509). **`total_phase_limited`:** no row
  changes (0→0: 833, 1→1: 31).
- **`WKB_Levin_max_depth`:** no row changes. **`WKB_Levin_num_regions`:** no row changes, with
  13,652 regions in both stores.
- **By depth under A,** the median $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ is:
  - 2e-9 to 1.1e-8 at depths 2–8;
  - 1.5e-6 to 1.7e-5 at depths 10–14;
  - the maximum anywhere is 8.8e-5, at depth 11.
- **`total_abserr` = inf in both stores:** 26 rows, at depths 0–8. 12 of their `total`s differ, by
  1.1e-16 to 1.0e-15 relative, and the tabulation prints their ratio as 0. This is the open issue
  `[04-roundoff-floor-can-be-infinite]`.
- **Metadata.** The non-wall-clock keys that differ, 29 distinct, are two families:
  - `partition.handover.T{q,r}.handover_continuity[i].d{T,Tprime}_rel`, **in every one of the 864
    rows**. This is expected (§B.5). The diagnostic takes $\sin$ and $\cos$ of $T$'s
    `theta_mod_2pi`, which is what this prompt changes. In 292 rows these are the only
    non-wall-clock keys that differ: the 291 rows with no WKB part, and one at depth 0;
  - the `WKB_Levin` `value`/`abserr` tree (`WKB_Levin.abserr` in 476 rows, `subintervals[i]`, and
    `subintervals[i].groups[j]`), in 572 rows. No `regions`, `simple_regions` or `evaluations` key
    differs.
- **R11 and R12 are not value-identical in `QuadSourceIntegral`.** So each tree ran the code it
  names.

These are recorded, not judged. Prompt §6 scores the effect on A3 v2's rows.

**Beside 05's R9/R10, for orientation only.** 05 changed $G$'s phase, and moved 542 `total`s by up
to 4.0e-5 relative and 28× `total_abserr`. It changed 15 depths and 42 region counts. This prompt
changes $T$'s phase, and moves 424 `total`s by at most 2.9e-11 relative and 8.8e-5 ×
`total_abserr`. It changes no depth, region count or convergence flag.

#### The readers (prompt §5.5)

`bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/levin-phase-floor-05a-r12.sqlite var/levin-phase-floor-05a/rehearsal/extract`,
with the lock held, from 12:29:51 to 12:32:26 UTC
([`out/readers.txt`](05a/rehearsal/out/readers.txt)). Overall exit **0**.

| reader | exit | wall | lookups | files written (05's R10) | store files after |
|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 28 s | 116 | 387 (450) | identical |
| `extract_GkWKB_data.py` | 0 | 57 s | 348 | 2436 (2436) | identical |
| `extract_GkSource_data.py` | 0 | 20 s | 10 | 162 (142) | identical |
| `extract_TkWKB_data.py` | 0 | 15 s | 3 | 27 (27) | identical |
| `extract_tensor_source_data.py` | 0 | 14 s | 6 | 27 (27) | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 17 s | 24 | 73 (74) | identical |

- The lookups equal 05's and R8's.
- File counts differ only where a reader draws an unseeded subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader, 0 Ray processes of this venv were left, and no reader log has a Traceback.

**R12's store, before and after the readers.** The `diff` is empty, and there is no `-journal`,
`-wal` or `-shm` file
([`state/r12_store_sha256_before_reads.txt`](05a/rehearsal/state/r12_store_sha256_before_reads.txt),
[`state/r12_store_sha256_after_reads.txt`](05a/rehearsal/state/r12_store_sha256_after_reads.txt)):

```
94dae1cca60ef5d58e9ef298d8fdd4fbf937a7eebbdd3f1495e9ca0b6300c749  levin-phase-floor-05a-r12.sqlite            36864
b42aa513f7d75b563ed196e4e8d6065c20606e8357565ebec912d36dbacb4e04  levin-phase-floor-05a-r12-shard0000.sqlite  24076288
a5d54bbbe8d6b1a64c47ea252f5b11be1dad640e2d061746493276e55254a73f  levin-phase-floor-05a-r12-shard0001.sqlite  24203264
af01699e932eeda9b24d8e4661fb284c3bec8ad1f0531f6c4f8bcb4d4897b273  levin-phase-floor-05a-r12-shard0002.sqlite   1077248
5b249c8510b5bd91ece9cb67bc060e17cd12bbac475c71246836bb861835c9da  levin-phase-floor-05a-r12-shard0003.sqlite  23560192
```

**Both stores, all ten files, are byte-identical from before the comparison to the end**
([`state/store_sha256_before_diff.txt`](05a/rehearsal/state/store_sha256_before_diff.txt),
[`state/stores_sha256_final.txt`](05a/rehearsal/state/stores_sha256_final.txt)). R11's files equal
the hashes taken just after its build
([`state/r11_store_sha256_after_build.txt`](05a/rehearsal/state/r11_store_sha256_after_build.txt)).
R11's primary is `8553800554d70f69d0fd90655899ed819bb5bf2456d62456da293f660e1a7cb8`.

#### Against prompt §7.2

- R11 and R12 both end `done` with fingerprint `1b332ffa…`: **yes**.
- Every value table outside `QuadSourceIntegral` is identical, and every other difference there is
  of the four kinds: **yes**.
  - All 7 value tables are bit-identical: 360,462 rows.
  - 0 differences are unclassified, and 0 rows are in one store only.
- The `QuadSourceIntegral` differences are tabulated: **yes**.
  - 424 of 864 `total`s moved, all through `WKB_Levin`, by at most 2.9e-11 relative and 8.8e-5 ×
    `total_abserr`.
  - No depth, region count or convergence flag changed.
  - `handover_continuity` moved in every row.
- The readers exit 0: **yes**, and the store is byte-identical.
- The lock is released: **yes**.
