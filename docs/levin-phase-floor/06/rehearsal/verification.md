### 6.x Rehearsals R13 and R14, and the value comparison (prompt 06 §5)

**Prompt 06 §5, run 2026-10-26, 21:55–22:15 UTC, on the Mac Studio** (mps026298, arm64, Darwin
27.0.0).
- **The trees.** Both builds ran from git worktrees, so that the main checkout did not move:
  - R13 ran in `SecondaryGWKit-lpf06-9f66622`, detached at `9f66622`, the parent of the code commit
    `dc9d28c`;
  - R14 ran in `SecondaryGWKit-lpf06-f5cee4c`, detached at Part B's dispatch SHA `f5cee4c`. Its code
    is `dc9d28c`'s; `dc9d28c..f5cee4c` touches only `prompts/`.
- **The main checkout** stayed at `28ebaf0` and clean throughout. It holds the venv, the stores,
  the registry and these records.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit` 0.2.0, `numpy` 2.2.4 and
  `scipy` 1.16.3.
- **The load** was about 5–10 on 10 cores. Prompt §6's Ray-free re-measurement ran beside these
  builds.

**Records** are in [`06/rehearsal/`](06/rehearsal/): `out/`, `scripts/` and `state/`, the agent's
`log.md`, this section as first written, and `issues.md`.
- Every command is in [`out/commands.log`](06/rehearsal/out/commands.log).
- The scripts' SHA-256 are in [`out/script_sha256.txt`](06/rehearsal/out/script_sha256.txt):
  - `classify_non_qsi.py` (`26690e61…`) and `tabulate_qsi.py` (`b7508351…`) are 05's, copied
    unchanged and byte-identical (`cmp`);
  - `run_readers.sh` (`7c63a5ba…`) and the launcher (`5b984c75…`) equal 05's;
  - `tools/store_value_diff.py` is `7997af85…` in the code worktree. That is the code commit's
    §1.5 extension. The parent worktree and the main checkout still hold 05's `5c92d8e3…`.
- The readers' `extract/` (176 MB) was not copied back, and neither was the 1.09 GB
  `out/store_value_diff.json` (see the log's observations).

#### The two builds

| | R13 (before) | R14 (after) |
|---|---|---|
| tree | worktree at `9f66622` (parent of `dc9d28c`) | worktree at `f5cee4c` (code `dc9d28c`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `PYTHONDONTWRITEBYTECODE=1` | the same, in its own worktree |
| run | `levin-phase-floor-06-r13-parent-20261026T215550` | `levin-phase-floor-06-r14-residual-20261026T215945` |
| `sh` / driver pid | 97736 / 97740 | 99934 / 99938 |
| wall | 21:55:50 → 21:58:44 (heartbeat) | 21:59:45 → 22:02:40 (heartbeat) |
| manifest | `git_head` `9f66622d…`, `git_dirty` false, cwd the parent worktree | `git_head` `f5cee4c7…`, `git_dirty` false, cwd the code worktree |
| cwd of the driver and of the `Datastore` actor (`lsof`) | the parent worktree (97740, 97834) | the code worktree (99938, 133) |
| end | `done`, exit 0; no Traceback; `stderr.log` empty | `done`, exit 0; no Traceback; `stderr.log` empty |
| store | `var/datastores/levin-phase-floor-06-r13.sqlite` + 4 shards | `var/datastores/levin-phase-floor-06-r14.sqlite` + 4 shards |

**The scope.** Both builds were R8's rehearsal scope, launched as 05a launched R11:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- `--job-name levin-phase-floor-06` and `--campaign levin-phase-floor --prompt 06`.

**The worktrees were not written.** After both builds, after the readers, and at the end, each
worktree's `git status --short --ignored` was empty, and `find -newer` the pre-R13 marker listed
nothing in either ([`out/r13_end_check.txt`](06/rehearsal/out/r13_end_check.txt),
[`out/r14_end_check.txt`](06/rehearsal/out/r14_end_check.txt),
[`out/final_check.txt`](06/rehearsal/out/final_check.txt)).

#### The repairs' agreement, and `_check_phase_sign` (prompt §0.2, §8; dispatch §B.5)

**R14 printed no `!! WARNING (GkSourcePolicyData` line.** It printed no other `!! WARNING` line
either, and `_check_phase_sign` did not raise.
- The count is 0 in R14's launch log, `stdout.log` and `stderr.log`, for each of these patterns:
  `!! WARNING (GkSourcePolicyData`, any `!! WARNING`, `_check_phase_sign`, `Error`, `Exception`
  and `Traceback`.
- **Ray's own worker logs.** Only the `Datastore` actor's output reaches the launch log, so the
  same patterns were also searched in Ray's worker logs for both sessions: 12 `worker-*.out` and 12
  `worker-*.err` each, under `/tmp/ray-sgk/session_*_97740` and `…_99938`. Every count is 0
  ([`out/ray_worker_logs_check.txt`](06/rehearsal/out/ray_worker_logs_check.txt)).
- Those logs record the same tasks in both builds: 291 `apply_GkSource_policy`, 291
  `assemble_GkSource_values`, 3480 `WKB_phase_function`, 1371 `numeric_with_phase_cut`, 864
  `compute_QuadSource_integral`, 6 `compute_quad_source`, 3 `find_horizon_exit_time` and 1
  `compute_background`.

#### The lock

[`out/lock.log`](06/rehearsal/out/lock.log). Each lock command was a shell command of its own, with
its output appended by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 06 section 5, rehearsals R13 … and R14 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid … 97740` (R13) | recorded | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after R13) | nothing | 0 |
| 4 | `set-pid … 99938` (R14) | recorded | 0 |
| 5 | `ray-processes` (after R14) | nothing | 0 |
| 6 | `ray-processes` (after the readers) | nothing | 0 |
| 7 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 8 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |
| 9 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |

#### The fingerprints (prompt §5.2, §5.3)

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R13** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 digests equal R8's |
| **R14** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 equal R8's; the whole `classes` object equals R13's |

The fingerprints are the ones the launcher took at the end of each build, read from `status.json`
(fingerprint format 6, encoding format 1;
[`out/r13_status.json`](06/rehearsal/out/r13_status.json),
[`out/r14_status.json`](06/rehearsal/out/r14_status.json)). There is no sidecar, because `--build`
writes none. As §5.3 says, the added column does not enter the fingerprint.

#### The value comparison (prompt §5.4)

The comparison was run from the code worktree, so that `-m` imports that tree's tool:

```
cd <code worktree> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<code worktree> <main>/venv/bin/python -B -m tools.store_value_diff <main>/var/datastores/levin-phase-floor-06-r13.sqlite <main>/var/datastores/levin-phase-floor-06-r14.sqlite --json
```

It exited **1**, as the addendum expects, in 29 s, and wrote 1.09 GB of JSON
(`out/store_value_diff.json`, SHA-256 `7e42eadc…`).
- The summary is [`out/store_value_diff_summary.txt`](06/rehearsal/out/store_value_diff_summary.txt),
  from [`scripts/summarise_diff.py`](06/rehearsal/scripts/summarise_diff.py).
- The exit status is in [`out/store_value_diff.meta.txt`](06/rehearsal/out/store_value_diff.meta.txt).
- R13 was read without `theta_residual`. The tool reports that column under `columns_only_in_b` in
  `GkWKBValue`, `TkWKBValue` and `GkSourceValue`, and in no other table.

**Structure.** 21 classes were compared, and `not_compared` is empty. In every class:
- 0 rows are in one store only;
- 0 rows cannot be matched by identity;
- 0 columns are unresolvable.

The value tables:

| value table | class | rows identical | differ | only in one / unmatched | column in one store only |
|---|---|---|---|---|---|
| BackgroundModelValue | outside | 1159 | 0 | 0 / 0 | — |
| TkNumericValue | outside | 1478 | 0 | 0 / 0 | — |
| GkNumericValue | outside | 35874 | 0 | 0 / 0 | — |
| QuadSourceValue | outside | 2447 | 0 | 0 / 0 | — |
| TkWKBValue | **inside** | 47 | 2162 | 0 / 0 | `theta_residual` (R14) |
| GkWKBValue | **inside** | 1391 | 146251 | 0 / 0 | `theta_residual` (R14) |
| GkSourceValue | **inside** | 23402 | 146251 | 0 / 0 | `theta_residual` (R14) |

There are 360,462 value rows in all. **R13 and R14 are not value-identical in `GkWKBIntegration`'s
values**, so the dispatch's stop does not fire: each tree ran its own code.

##### Outside the five classes

- **Every value table is bit-identical:** `BackgroundModelValue`, `TkNumericValue`,
  `GkNumericValue` and `QuadSourceValue`, 40,958 rows.
- **10 classes are value-identical:** `version`, `store_tag`, `redshift`, `wavenumber`,
  `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`, `GkSourcePolicy` and
  `QuadSourcePolicy`. `OneLoopIntegral` has no rows.
- **Five classes differ in row columns only.** 05's
  [`scripts/classify_non_qsi.py`](06/rehearsal/scripts/classify_non_qsi.py), copied unchanged,
  sorts each of their differences into one of the volatile kinds
  ([`out/classify_non_qsi.txt`](06/rehearsal/out/classify_non_qsi.txt)). **0 are unclassified:**

  | class | rows | columns (kind) |
  |---|---|---|
  | wavenumber_exit_time | 3 | `compute_time` (timing) |
  | BackgroundModel | 1 | `compute_time`, `{min,mean,max}_RHS_time` (timing) |
  | TkNumericIntegration | 3 | `compute_time`, `*_RHS_time` (timing); `label` |
  | GkNumericIntegration | 1368 | `compute_time`, `*_RHS_time` (timing; `min_RHS_time` in 1047); `label` |
  | QuadSource | 6 | `compute_time` (timing); `label` |

  Nothing outside the five classes differs in any other way, so prompt §8's stop does not fire.
- **The script exits 1, reporting 185 problems.** All 185 are inside the five classes, because it
  exempts only `QuadSourceIntegral` (log, deviation 6):
  - 3 are the value tables of `TkWKBIntegration`, `GkWKBIntegration` and `GkSource`;
  - 182 are `GkSourcePolicyData` rows whose `metadata` differs.
- **Its row-column kinds inside the five classes** are the same volatile kinds as before:
  - `TkWKBIntegration`: 3 `label` and 3 `stage_1_compute_time`;
  - `GkWKBIntegration`: 3477 `label`, 3474 `stage_1_compute_time`, and 25 rows with the
    residual-memo counters `rho_evals`, `rho_reused` and `stage_1_RHS_evaluations`;
  - `GkSource`: 291 `label` and 291 `parents_digest`.

##### Inside the five classes, tabulated per class

The new script is [`scripts/tabulate_phase_classes.py`](06/rehearsal/scripts/tabulate_phase_classes.py)
(`81effdbb…`).
- **How it reads.** It pairs rows with the code worktree's own `tools.store_value_diff.read_store`,
  which is the tool's identity matching, and so has every column of both sides.
- **The cross-check.** Its differing-row counts per column equal the tool's JSON for all three
  value tables.
- **Outputs:** [`out/phase_classes_tabulation.txt`](06/rehearsal/out/phase_classes_tabulation.txt),
  with one row per value row in `out/phase_classes_rows.csv`.
- **The outliers** come from [`scripts/examine_outliers.py`](06/rehearsal/scripts/examine_outliers.py),
  in [`out/examine_outliers.txt`](06/rehearsal/out/examine_outliers.txt).
- **Notation:** $\varepsilon = 2^{-52}$, and $|\theta| = |{\rm div}\cdot2\pi + {\rm mod}|$ from
  R14's stored cycle count and remainder.

**`theta_residual`, present in R14 only:**
- `GkWKBValue`: non-null in all 147,642 rows; max $|r|$ 1.571.
- `TkWKBValue`: non-null in all 2,209 rows; max $|r|$ 1.764.
- `GkSourceValue`: non-null in all 147,642 rows with a WKB phase; max $|r|$ 1.571.
- In no table is $|r|$ greater than $\pi + 2\times10^{-3}$.
- In every table, R14's `theta_mod_2pi` lies in $(-2\pi, 0]$.

**`theta_mod_2pi`: $|\Delta|$ against $\varepsilon|\theta|$**, over the rows whose cycle count did
not change, which is every row:

| table | rows | bit-identical | $\lvert\theta\rvert$ median / max | $\lvert\Delta\rvert$ max (rad) | $\lvert\Delta\rvert/\varepsilon\lvert\theta\rvert$ median / 90 % / 99 % / max | > 1 | > 4 |
|---|---|---|---|---|---|---|---|
| GkWKBValue | 147642 | 1391 | 7.65e5 / 7.67e8 | 2.28e-7 | 0.324 / 0.790 / 1.26 / 272 | 5442 | 52 |
| TkWKBValue | 2209 | 47 | 7.01e4 / 2.11e8 | 5.29e-8 | 0.395 / 0.877 / 1.43 / 8.41 | 120 | 2 |
| GkSourceValue | 147642 | 1391 | 7.65e5 / 7.67e8 | 2.28e-7 | 0.309 / 0.766 / 1.19 / 7.18 | 4553 | 2 |

- **All 54 producer rows above 4** (52 $G$, 2 $T$) have $|\theta| < 1$ rad, and
  $|\Delta| \le 8.9\times10^{-16}$, which is one ulp of $2\pi$.
- **Where they lie.** These are samples close to the producer's own anchor. There $\varepsilon
  |\theta|$ is smaller than the granularity of the remainder itself, and smaller than the error of
  the old cancelling sum $k\,L.\delta$, whose terms are of size $k\tau$.
- **Against `GkSource`'s rectified cycle count.** The same remainders sit at $|\theta|$ of 0.09–26
  rad there. Against that, every ratio is at most 7.2.
- This is recorded, not judged (log, observations).

**`theta_div_2pi` changes: none.**
- There are 0 in `GkWKBValue`, 0 in `TkWKBValue` and 0 in `GkSourceValue`, and 0 changes of
  `GkSourceValue.raw_theta_div_2pi`.
- So no change needs to be tested against the $4\varepsilon|\theta|$ bound of §5.4 (§9 point 3),
  and none is listed on its own.

**`G_WKB` and `T_WKB`.** Here $A = \sqrt{H_{\rm ratio}/\omega}\,{\rm hypot}(s, c)$, times
$e^{\rm friction}$ for $T$, is the Liouville–Green amplitude.

| column | rows differ | $\lvert\Delta\rvert/\lvert a\rvert$ median / 90 % / 99 % / max | $\lvert\Delta\rvert/(A\,\lvert\Delta\theta_{\rm mod}\rvert)$ median / 90 % / max |
|---|---|---|---|
| GkWKBValue.G_WKB | 145772 | 3.7e-11 / 1.1e-8 / 2.0e-7 / 3.6e-4 | 0.70 / 0.99 / 2.73 |
| TkWKBValue.T_WKB | 2154 | 5.0e-12 / 3.2e-9 / 8.0e-8 / 7.7e-6 | 0.72 / 0.99 / 1.08 |
| GkSourceValue.G_WKB | 145772 | as GkWKBValue | as GkWKBValue |
| GkSourceValue.new_G_WKB | 145772 | as GkWKBValue | as GkWKBValue |

- The largest relative changes are where $G$ is near a zero of its oscillation.
- **Exceeding $A|\Delta\theta_{\rm mod}|$.** 615 `GkWKBValue` rows and 14 `TkWKBValue` rows exceed
  it. Every one is within $A|\Delta\theta| + 4\varepsilon A$: that excess is the rounding of $G$
  itself, and most of these rows have $|\Delta\theta| \le 10^{-14}$. So every change to `G_WKB`
  and `T_WKB` is accounted for by the change of the remainder.
- No other value column of the three tables differs.

**`GkSourcePolicyData`'s changed fields:**
- `Levin_z`, `crossover_z`, `type` and `quality` are unchanged in all 291 rows.
- Only `metadata` differs, in 182 rows, and only in its key `Levin_z_dtheta_dlogz`:
  - $|\Delta|$ has median 3.0e-10 and max 4.0e-6;
  - $|\Delta|/|a|$ has median 2.0e-10 and max 2.6e-6;
  - no value goes to or from null.

##### `QuadSourceIntegral`, tabulated per row

05's [`scripts/tabulate_qsi.py`](06/rehearsal/scripts/tabulate_qsi.py), copied unchanged, reads both
stores (`mode=ro`) and the tool's JSON. Its docstring still names R9 and R10; **here A = R13
(parent `9f66622`) and B = R14 (code `dc9d28c`)**.
- Its output is [`out/qsi_tabulation.txt`](06/rehearsal/out/qsi_tabulation.txt), with one row per
  integral in `out/qsi_rows.csv`.
- Further counts are in [`out/qsi_extra_counts.txt`](06/rehearsal/out/qsi_extra_counts.txt).

**The pairing.** Rows were paired by physical (k, q, r, z_response): 864 = 864, with one atol and
one rtol, (−32, −7). It has the same key set as the tool's, with 0 mismatches in `total`'s
`float.hex` against the tool.

| | rows |
|---|---|
| `total` differs | **554** of 864 (310 bit-identical) |
| `WKB_Levin` differs | 568. Every row whose `total` differs is among them |
| `numeric_quad` / `analytic_rad` differ | **0** / **0** |
| rows with no `WKB_Levin` part (`max_depth` None) | 291, none of which differs in `total` |

Over the 554 rows whose `total` differs:

| statistic | median | 90 % | max |
|---|---|---|---|
| $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}_A\rvert$ | 2.79e-11 | 1.37e-08 | 1.57e-03 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ | 1.90e-04 | 9.82e-02 | 1.82e+04 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_B$ | 1.90e-04 | 9.58e-02 | 1.28e+04 |

- **Above `total_abserr`.** 4 rows exceed $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ > 1,
  and 2 exceed it against `total_abserr`$_B$. 55 rows exceed 0.1 against $A$, and 54 against $B$.
- **The two largest are at $z_{\rm resp} = 250$, the end of the range:**

  | $(k, q, r)$ | rel. change | $/$`total_abserr`$_A$ | depth | regions |
  |---|---|---|---|---|
  | $(3.162\times10^5, 3.162\times10^5, 3.162\times10^5)$ | 1.6e-3 | 1.8e4 | 12 → 6 | 31 → 14 |
  | $(10^5, 3.162\times10^5, 3.162\times10^5)$ | 2.8e-5 | 484 | 14 → 6 | 30 → 15 |

  Both rows have `total_converged` 0 in both stores. The other two rows above 1 are at
  $k = q = r = 10^6$: $z_{\rm resp}$ 755.7 at 1.25, and 996.4 at 1.02. Both are below 1 against
  `total_abserr`$_B$.
- **`WKB_Levin_max_depth`:** 3 rows change (12 → 6, 14 → 6 and 4 → 3).
- **`WKB_Levin_num_regions`:** 59 rows change, and the total falls from 13,652 to 13,495.
- **`total_converged`:** no row changes (0→0: 355, 1→1: 509).
- **`total_phase_limited`:** 13 rows change (4 0→1, 9 1→0).
- **By depth under A,** the median $\lvert\Delta\rvert/$`total_abserr`$_A$ is:
  - 1e-5 to 2.4e-4 at depths 5–8;
  - 0.04 to 0.13 at depths 10–12.
- **`total_abserr` = inf in both stores:** 26 rows. 9 of their `total`s differ. This is the open
  issue `[04-roundoff-floor-can-be-infinite]`.
- **Metadata.** 60 distinct non-wall-clock keys differ, and every row has at least one. They form
  two families:
  - `partition.handover.T{q,r}.handover_continuity[i].d{T,Tprime}_rel`, from $T$'s changed
    remainder, as in 05a;
  - the `WKB_Levin` `value`/`abserr` tree.

These are recorded, not judged. Prompt §6 scores the effect on A3 v2's rows.

#### The readers (prompt §5.5)

**Why they did not run from `run_readers.sh` itself.** The script changes directory to its own
tree and calls `./venv/bin/python`. The code worktree has no venv, and the main checkout's copy
would run the main checkout's code (`28ebaf0`), not the code commit's.
- So they ran through [`scripts/run_readers_worktree.sh`](06/rehearsal/scripts/run_readers_worktree.sh),
  a copy changed in three lines only ([`out/run_readers_worktree.diff`](06/rehearsal/out/run_readers_worktree.diff)).
- Each reader's cwd was the code worktree, run with the main venv's python, `PYTHONPATH=.` and
  `PYTHONDONTWRITEBYTECODE=1`.
- The lock was held, from 22:12:15 to 22:14:35 UTC
  ([`out/readers.txt`](06/rehearsal/out/readers.txt)). Overall exit **0**.

| reader | exit | wall | lookups | files written (05a's R12) | store files after |
|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 25 s | 116 | 396 (387) | identical |
| `extract_GkWKB_data.py` | 0 | 47 s | 348 | 2436 (2436) | identical |
| `extract_GkSource_data.py` | 0 | 19 s | 10 | 176 (162) | identical |
| `extract_TkWKB_data.py` | 0 | 14 s | 3 | 27 (27) | identical |
| `extract_tensor_source_data.py` | 0 | 14 s | 6 | 27 (27) | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 16 s | 24 | 70 (73) | identical |

- The lookups equal 05's and 05a's.
- File counts differ only where a reader draws an unseeded subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader, 0 Ray processes of the venv were left.
- No reader log has a Traceback. Each has one line `!! WARNING: number of shards read from
  database (=N) does not match specified number of shards (=N)`, the readers' own note on the shard
  count.

**R14's store, before and after the readers.** The `diff` is empty, and there is no `-journal`,
`-wal` or `-shm` file
([`state/r14_store_sha256_before_reads.txt`](06/rehearsal/state/r14_store_sha256_before_reads.txt),
[`state/r14_store_sha256_after_reads.txt`](06/rehearsal/state/r14_store_sha256_after_reads.txt)):

```
1c0f3cf63cc95208c53e558d152255f34fa6b17c9b82f9270a2b43f2f126f7da  levin-phase-floor-06-r14-shard0000.sqlite  25038848
866511b7ef81f27da9bcf5306bd9a6efa9c367475f3c7959a23a78d07c943fcd  levin-phase-floor-06-r14-shard0001.sqlite  25255936
dfaad9499d3d7f7364532cd5cb8091b1f32a51d1fed0a1a43f10213fdaff656b  levin-phase-floor-06-r14-shard0002.sqlite   1077248
410bf34b370512c5426428f1c6f3c6ea6008e3bd9b0d371f60e08bbce9d12a57  levin-phase-floor-06-r14-shard0003.sqlite  24485888
fff6cf7fac9106897efba04547710ec99ce051f4f2457108dc8cdb6fd9689acd  levin-phase-floor-06-r14.sqlite              36864
```

**Both stores, all ten files, are byte-identical from before the comparison to the end**
([`state/store_sha256_before_diff.txt`](06/rehearsal/state/store_sha256_before_diff.txt),
[`state/stores_sha256_final.txt`](06/rehearsal/state/stores_sha256_final.txt)). R13's files equal
the hashes taken just after its build
([`state/r13_store_sha256_after_build.txt`](06/rehearsal/state/r13_store_sha256_after_build.txt)).
R13's primary is `766839d657e7f943f2e3ef1f3058268544b42ffcfeb1d8f753a68d9c768989c9`.

#### Against prompt §7.2

- **R13 and R14 both end `done` with fingerprint `1b332ffa…`: yes.** Each has 21/21 class digests
  equal to R8's.
- **Outside the classes of §0.3, every value table is identical and every other difference is of
  the volatile kinds: yes.**
  - The 4 value tables there are bit-identical: 40,958 rows.
  - 0 differences there are unclassified.
  - 0 rows are in one store only, and no column is in one store only.
- **Inside them, the differences are tabulated: yes.**
  - `theta_residual` is in R14 only, and non-null wherever a remainder is stored.
  - `theta_mod_2pi` moved in 146,251 $G$ and 2,162 $T$ rows. The 99th percentile of the move is
    1.2–1.4 $\varepsilon|\theta|$.
  - `theta_div_2pi`: no change.
  - `G_WKB` and `T_WKB` moved by the remainder's change only.
  - `GkSourcePolicyData`: only `Levin_z_dtheta_dlogz`, by at most 2.6e-6 relative.
  - `QuadSourceIntegral`: 554/864 `total`s moved, two by $10^2$–$10^4$ × `total_abserr`$_A$, with
    their depth halved.
- **The repairs agree in R14:** 0 `!! WARNING (GkSourcePolicyData` lines, and
  `_check_phase_sign` did not raise.
- **The readers exit 0: yes,** and the store is byte-identical.
- **The lock is released: yes.**
