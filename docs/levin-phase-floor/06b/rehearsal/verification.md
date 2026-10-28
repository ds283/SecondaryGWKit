### 6b.x Rehearsals R15 and R16, and the value comparison (prompt 06b §4)

**Prompt 06b §4, run 2026-10-28, 13:52–14:12 UTC, on the Mac Studio** (mps026298, Darwin 27.0.0).
- **The trees.** Both builds ran from git worktrees, so the main checkout did not move:
  - **R15** ran in `SecondaryGWKit-lpf06b-da5c24d`, detached at `da5c24d`, the parent of the code
    commit `4ddc3c2`;
  - **R16** ran in `SecondaryGWKit-lpf06b-92bc62f`, detached at Part B's dispatch SHA `92bc62f`.
    Its code is `4ddc3c2`'s: `4ddc3c2..92bc62f` is empty outside `docs/` and `prompts/`.
  - **On code paths,** `da5c24d..92bc62f` touches only `AdaptiveLevin/levin_quadrature.py` and
    `AdaptiveLevin/tests/test_levin_roundoff_exit.py`.
- **The main checkout** stayed at `28ebaf0` and clean. It holds the venv, the stores, the registry
  and these records.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0 and `runregistrykit` 0.2.0.
- **The load** was about 9 on 10 cores, all of it macOS and desktop processes.

**Records** are in [`06b/rehearsal/`](06b/rehearsal/): `out/`, `scripts/`, `state/`, the agent's
`log.md`, this section as first written, and `issues.md`.
- **Commands.** Every command is in [`out/commands.log`](06b/rehearsal/out/commands.log).
- **Scripts, by SHA-256** (from [`out/final_check.txt`](06b/rehearsal/out/final_check.txt)):
  - `classify_non_qsi.py` is `26690e61…`. It is 05's, copied unchanged, and byte-identical to 05's
    and 06's copies;
  - `run_readers_worktree.sh` is `ec96cf09…`, 06's, copied unchanged;
  - `run_readers.sh` is `7c63a5ba…`;
  - the launcher is `5b984c75…` in both worktrees and in the main checkout;
  - `tools/store_value_diff.py` is `7997af85…` in both worktrees;
  - the new `tabulate_qsi_phase_limited.py` is `ecde056d…`. Its failed first attempt is kept as
    `state/tabulate_qsi_phase_limited.attempt1.py`, `03176ba4…`.
- **Not copied back:** the readers' `extract/` (180 MB).

#### The two builds

| | R15 (before) | R16 (after) |
|---|---|---|
| tree | worktree at `da5c24d` (parent of `4ddc3c2`) | worktree at `92bc62f` (code `4ddc3c2`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `-B -u` and `PYTHONDONTWRITEBYTECODE=1`; `nohup sh -c`, detached | the same, in its own worktree |
| run | `levin-phase-floor-06b-r15-parent-20261028T135349` | `levin-phase-floor-06b-r16-premise-20261028T135745` |
| `sh` / driver pid | 96490 / 96498 | 97356 / 97360 |
| wall | 13:53:48 launch → 13:56:59 last heartbeat; driver gone 13:57:07 | 13:57:43 → 14:01:04; driver gone 14:01:13 |
| manifest | `git_head` `da5c24de…`, `git_dirty` false, cwd the parent worktree, script SHA `5b984c75…` | `git_head` `92bc62f0…`, `git_dirty` false, cwd the code worktree, script SHA `5b984c75…` |
| cwd of the driver (`lsof`) | the parent worktree | the code worktree |
| end | `done`, exit 0; no Traceback; `stderr.log` empty; no `!! WARNING` in the launch log | the same |
| store | `var/datastores/levin-phase-floor-06b-r15.sqlite` + 4 shards | `var/datastores/levin-phase-floor-06b-r16.sqlite` + 4 shards |

**The scope** was R8's rehearsal scope, launched as R13 and R14 were:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- `--job-name levin-phase-floor-06b`, `--register r15-parent` / `r16-premise`, and
  `--campaign levin-phase-floor --prompt 06b`.

**The worktrees were not written.** After each build, after the readers and at the end, each
worktree's `git status --short --ignored` was empty, and `find -newer` the pre-R15 marker listed
nothing in either.

#### The lock

[`state/lock.log`](06b/rehearsal/state/lock.log). Each lock command was a shell command of its
own, with its output appended by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (before) | nothing | 0 |
| 2 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 06b section 4, rehearsals R15 … and R16 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 3 | `set-pid … 96498` (R15) | recorded | 0 |
| 4 | `ray-processes` (after R15) | nothing | 0 |
| 5 | `set-pid … 97360` (R16) | recorded | 0 |
| 6 | `ray-processes` (after R16) | nothing | 0 |
| 7 | `ray-processes` (after the readers) | nothing | 0 |
| 8 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 9 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |
| 10 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |

#### The fingerprints (prompt §4.2)

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R15** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | `classes` equals R8's |
| **R16** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | `classes` equals R8's and R15's |

These are the fingerprints the launcher took at the end of each build, read from `status.json`:
[`out/r15_status.json`](06b/rehearsal/out/r15_status.json) and
[`out/r16_status.json`](06b/rehearsal/out/r16_status.json). R15's was taken at `da5c24d` and
R16's at `92bc62f`, both with `git_dirty` false. There is no sidecar, because `--build` writes
none.

#### The value comparison (prompt §4.3)

The comparison was run from the code worktree, so that `-m` imports that tree's tool:

```
cd <code worktree> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<code worktree> <main>/venv/bin/python -B -m tools.store_value_diff <main>/var/datastores/levin-phase-floor-06b-r15.sqlite <main>/var/datastores/levin-phase-floor-06b-r16.sqlite --json
```

- **The run.** It exited **1**, in 18 s, and wrote 40.3 MB of JSON (`out/store_value_diff.json`,
  SHA-256 `51b9b94a…`; [`out/store_value_diff.meta.txt`](06b/rehearsal/out/store_value_diff.meta.txt)).
- **The structure.** `not_compared` is empty. Every class has 0 rows in one store only, 0 rows
  that cannot be matched and 0 unresolvable columns.
- **The stores did not change.** Both stores' ten files are byte-identical before and after it
  ([`state/store_sha256_before_diff.txt`](06b/rehearsal/state/store_sha256_before_diff.txt),
  [`state/store_sha256_after_diff.txt`](06b/rehearsal/state/store_sha256_after_diff.txt)).

##### Outside `QuadSourceIntegral`

The source is 05's [`scripts/classify_non_qsi.py`](06b/rehearsal/scripts/classify_non_qsi.py),
unchanged, with output in [`out/classify_non_qsi.txt`](06b/rehearsal/out/classify_non_qsi.txt). It
exits 0 and reports **0 unclassified differences or structural problems**.

- **Every value table is bit-identical:** 7 tables, 360,462 rows.

  | value table | rows identical | differ |
  |---|---|---|
  | BackgroundModelValue | 1,159 | 0 |
  | TkNumericValue | 1,478 | 0 |
  | TkWKBValue | 2,209 | 0 |
  | GkNumericValue | 35,874 | 0 |
  | GkWKBValue | 147,642 | 0 |
  | GkSourceValue | 169,653 | 0 |
  | QuadSourceValue | 2,447 | 0 |

- **11 classes are row-identical:** `version`, `store_tag`, `redshift`, `wavenumber`,
  `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`, `GkSourcePolicy`,
  `QuadSourcePolicy` and `GkSourcePolicyData`. `OneLoopIntegral` has no rows.
- **Eight classes differ in row columns of the volatile kinds only:**

  | class | rows | columns (kind) |
  |---|---|---|
  | wavenumber_exit_time | 3 | `compute_time` (timing) |
  | BackgroundModel | 1 | `compute_time`, `{min,mean,max}_RHS_time` (timing) |
  | TkNumericIntegration | 3 | `compute_time`, `*_RHS_time` (timing); `label` |
  | TkWKBIntegration | 3 | `stage_1_compute_time` (timing); `label` |
  | GkNumericIntegration | 1368 | `compute_time`, `*_RHS_time` (timing; `min_RHS_time` in 1022); `label` |
  | GkWKBIntegration | 3477 | `stage_1_compute_time` (timing, 3474); `label`; `metadata` (`rho_evals`/`rho_reused` only) and `stage_1_RHS_evaluations` in 23 rows (the residual-memo work counter) |
  | GkSource | 291 | `label`; `parents_digest` |
  | QuadSource | 6 | `compute_time` (timing); `label` |

##### Inside `QuadSourceIntegral`: where the phase-limited flags are stored

The flags were established from `ComputeTargets/QuadSourceIntegral.py` and
`ObjectFactories/QuadSourceIntegral.py` at `92bc62f`, and confirmed on the stores. All three live
in the shard table `QuadSourceIntegral`.

| flag | column | encoding | set by |
|---|---|---|---|
| **(a)** `total_phase_limited` | `total_phase_limited` | SQLAlchemy `Boolean`, nullable. Stored as SQLite `integer` 0/1: `typeof` is `integer` on all 864 rows of both stores | `evaluate_QuadSource_integral`: the OR of (b) over every Levin sub-interval |
| **(b)** each group record's flag | `metadata` (a JSON string, `json.dumps` / `json.loads`), key `WKB_Levin.subintervals[i].groups[j].phase_limited` | JSON `true`/`false` | `phase_group_Levin_integral`: one record per phase group, holding `label`, `value`, `abserr`, `converged`, `phase_limited`, `regions`, `simple_regions`, `evaluations` and `elapsed` |
| **(c)** each analytic block's flag | `metadata`, key `analytic.{0pt5,2pt5}.{J,Y}{1..4}_data.phase_limited` (three-Bessel integral all Levin) or `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.phase_limited` (split quad + Levin) | JSON `true`/`false` | `_three_bessel_Levin`: one record per block, holding `elapsed`, `regions`, `abserr`, `converged` and `phase_limited` |

- **The ORs of (b)** are `WKB_Levin.phase_limited` and `WKB_Levin.subintervals[i].phase_limited`.
  The tabulation asserts, on every row of both stores, that each is the OR of its groups and that
  (a) is the OR of all (b).
- **The ORs of (c)** are not stored.
- **No other** metadata leaf named `phase_limited` exists.
- **"Phase-limited in R15"** is (a) or any (b) or any (c), on R15's row.

##### Inside `QuadSourceIntegral`: per row

The new script is
[`scripts/tabulate_qsi_phase_limited.py`](06b/rehearsal/scripts/tabulate_qsi_phase_limited.py).
- **How it reads.** It opens every store file with sqlite `mode=ro`, and pairs rows by physical
  $(k, q, r, z_{\rm resp})$.
- **The cross-checks.** The pairing gives the tool's key set, and `total`'s `float.hex` agrees with
  the tool on all 864 rows. Both stores have one tolerance pair, (atol, rtol) = (1e-32, 1e-7), and
  their metadata key sets are equal on every row.
- **Outputs:** [`out/qsi_tabulation.txt`](06b/rehearsal/out/qsi_tabulation.txt), with one line per
  row in `out/qsi_rows.csv`.
- **Bit-identity** is by `float.hex` for floats, and by type and `==` otherwise.

**Field kinds:**
- **Not compared:**
  - timings: columns ending `compute_time`, `RHS_time` or `elapsed`, and metadata leaves `elapsed`
    and `compute_time`;
  - `label`;
  - store identities: `serial`, every `*_serial`, and `timestamp`.
- **Work counters,** compared and reported separately:
  - the columns `WKB_Levin_{num_regions, evaluations, simple_regions, SVD_errors, order_changes,
    chebyshev_min_order, max_depth}`, `numeric_quad_compute_steps` and
    `numeric_quad_RHS_evaluations`;
  - the metadata leaves `regions`, `simple_regions` and `evaluations`.
- **Values:** everything else, every metadata leaf included.

**The counts:**

| | R15 | R16 |
|---|---|---|
| rows phase-limited (any flag) | **82** of 864 | **0** |
| (a) `total_phase_limited` true | 26 | 0 |
| rows with any (b) | 26 | 0 |
| (b) flags set, of 3,105 group records | 48 | 0 |
| rows with any (c) | 70 | 0 |
| (c) flags set, of 10,072 analytic blocks | 143 | 0 |

Of the 82 rows, 12 are (a/b) only, 56 are (c) only and 14 are both.

**The rule: a row with no phase-limited flag in R15 is bit-identical.** It holds on all 782 such
rows:
- **The named set:** 0 rows differ in `total`, `total_abserr`, `analytic_rad`, any `analytic`
  metadata leaf or any field of any group record.
- **Every value field** of the row, columns and metadata: 0 rows differ.
- **Every work counter:** 0 rows differ.

**The other 82 rows are tabulated in full** in [`out/qsi_tabulation.txt`](06b/rehearsal/out/qsi_tabulation.txt).
- **`total`** moved in exactly the 26 rows with a (b) flag in R15, and in no other
  ([`out/qsi_set_check.txt`](06b/rehearsal/out/qsi_set_check.txt)).
- **`analytic_rad`** moved in exactly the 70 rows with a (c) flag.
- **The 56 (c)-only rows** keep `total` and `total_abserr` bit for bit.

The 26 rows whose total moved:

| (k, q, r) /Mpc | z_resp | flags R15 a/b/c | `total_abserr` R15 → R16 | R16/R15 | \|ΔT\|/err15 | \|ΔT\|/err16 | Levin regions | max depth | converged |
|---|---|---|---|---|---|---|---|---|---|
| (1e5, 1e5, 1e5) | 755.687 | 1/1/0 | 3.561e-25 → 3.507e-25 | 0.985 | 0.00182 | 0.00185 | 27 → 29 | 11 → 11 | 0 → 0 |
| (1e5, 3.16e5, 3.16e5) | 755.687 | 1/2/0 | 5.089e-26 → 4.608e-26 | 0.905 | 0.0642 | 0.0709 | 15 → 17 | 6 → 6 | 0 → 0 |
| (1e5, 3.16e5, 3.16e5) | 36286.8 | 1/1/0 | 2.174e-24 → 2.200e-24 | 1.012 | 0.00661 | 0.00653 | 23 → 28 | 7 → 7 | 0 → 0 |
| (1e5, 3.16e5, 3.16e5) | 83186 | 1/1/0 | 5.036e-24 → 5.044e-24 | 1.002 | 0.0048 | 0.0048 | 21 → 24 | 7 → 7 | 0 → 0 |
| (1e5, 1e6, 1e6) | 434.651 | 1/2/3 | 3.054e-27 → 2.579e-27 | 0.844 | 0.0492 | 0.0582 | 15 → 17 | 6 → 6 | 0 → 1 |
| (1e5, 1e6, 1e6) | 755.687 | 1/1/3 | 5.664e-27 → 5.336e-27 | 0.942 | 0.024 | 0.0255 | 14 → 15 | 6 → 6 | 0 → 0 |
| (1e5, 1e6, 1e6) | 1313.84 | 1/2/3 | 8.130e-27 → 7.024e-27 | 0.864 | 0.0435 | 0.0503 | 10 → 12 | 3 → 3 | 0 → 1 |
| (1e5, 1e6, 1e6) | 3011.93 | 1/2/2 | 1.517e-26 → 1.349e-26 | 0.889 | 0.0399 | 0.0448 | 15 → 17 | 6 → 6 | 0 → 1 |
| (1e5, 1e6, 1e6) | 9104.3 | 1/1/1 | 4.172e-26 → 4.057e-26 | 0.973 | 0.000935 | 0.000962 | 15 → 17 | 6 → 6 | 0 → 0 |
| (3.16e5, 3.16e5, 3.16e5) | 329.64 | 1/3/0 | 3.248e-27 → 1.653e-27 | 0.509 | 0.0439 | 0.0863 | 32 → 35 | 12 → 12 | 0 → 1 |
| (3.16e5, 3.16e5, 3.16e5) | 1732.38 | 1/1/0 | 1.504e-26 → 1.425e-26 | 0.947 | 0.00699 | 0.00738 | 33 → 34 | 12 → 12 | 0 → 0 |
| (3.16e5, 3.16e5, 3.16e5) | 2284.25 | 1/1/0 | 1.828e-26 → 1.735e-26 | 0.949 | 0.00127 | 0.00134 | 32 → 35 | 12 → 12 | 0 → 0 |
| (3.16e5, 3.16e5, 3.16e5) | 3011.93 | 1/1/0 | 2.345e-26 → 2.246e-26 | 0.958 | 0.00356 | 0.00372 | 30 → 32 | 12 → 12 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 250 | 1/2/2 | 8.930e-28 → 5.711e-28 | 0.640 | 0.36 | 0.563 | 12 → 14 | 6 → 6 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 573.115 | 1/1/1 | 1.454e-27 → 1.408e-27 | 0.968 | 0.0351 | 0.0362 | 14 → 15 | 7 → 7 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 755.687 | 1/3/3 | 2.362e-27 → 1.882e-27 | 0.797 | 0.124 | 0.155 | 14 → 17 | 7 → 7 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 1732.38 | 1/4/1 | 4.007e-27 → 2.921e-27 | 0.729 | 0.152 | 0.208 | 12 → 20 | 6 → 6 | 0 → 1 |
| (3.16e5, 1e6, 1e6) | 2284.25 | 1/4/0 | 4.882e-27 → 3.653e-27 | 0.748 | 0.107 | 0.143 | 12 → 18 | 6 → 6 | 0 → 1 |
| (3.16e5, 1e6, 1e6) | 3011.93 | 1/4/1 | 6.076e-27 → 4.916e-27 | 0.809 | 0.0896 | 0.111 | 12 → 16 | 6 → 6 | 0 → 1 |
| (3.16e5, 1e6, 1e6) | 15828.8 | 1/2/0 | 3.035e-26 → 2.953e-26 | 0.973 | 0.00275 | 0.00283 | 20 → 25 | 6 → 6 | 0 → 0 |
| (1e6, 3.16e5, 1e6) | 250 | 1/1/3 | 5.501e-28 → 3.678e-28 | 0.668 | 0.046 | 0.0688 | 35 → 36 | 7 → 7 | 0 → 0 |
| (1e6, 3.16e5, 1e6) | 329.64 | 1/1/1 | 6.076e-28 → 4.110e-28 | 0.677 | 0.0249 | 0.0367 | 39 → 40 | 7 → 7 | 0 → 0 |
| (1e6, 3.16e5, 1e6) | 573.115 | 1/1/2 | 1.153e-27 → 9.348e-28 | 0.811 | 0.0405 | 0.05 | 31 → 32 | 6 → 6 | 0 → 0 |
| (1e6, 1e6, 1e6) | 329.64 | 1/3/1 | 5.416e-28 → 1.310e-28 | 0.242 | 0.144 | 0.597 | 16 → 27 | 7 → 7 | 0 → 0 |
| (1e6, 1e6, 1e6) | 755.687 | 1/1/0 | 8.133e-28 → 5.230e-28 | 0.643 | 0.333 | 0.517 | 14 → 15 | 7 → 7 | 0 → 0 |
| (1e6, 1e6, 1e6) | 20871.2 | 1/2/0 | 6.169e-27 → 6.072e-27 | 0.984 | 0.0138 | 0.014 | 17 → 25 | 8 → 8 | 0 → 0 |

**Summaries of the 82 flagged rows:**
- **The 26 rows whose total moved:**
  - $|\Delta T|$ is at most 0.36 × `total_abserr`$_{R15}$ and at most 0.597 ×
    `total_abserr`$_{R16}$;
  - `total_abserr` fell in 24, by a factor of 0.242–0.985, and rose in 2, by a factor of 1.002 and
    1.012;
  - `WKB_Levin_max_depth` did not change in any of them;
  - their Levin regions rose from 530 to 612, and their `WKB_Levin_evaluations` from 1314 to 1638;
  - `total_converged` went 0 → 1 in 7, and stayed 0 in 19.
- **The 70 rows whose `analytic_rad` moved:** $|\Delta$`analytic_rad`$|$ against R15's analytic
  `abserr` has a median of 0.058 and a maximum of 0.748. Their analytic-block regions rose from
  2203 to 2355.
- **Over all 82:** `total_converged` is 0 → 0 in 71, 0 → 1 in 7 and 1 → 1 in 4.

#### The readers (prompt §4.4)

- **Where they ran.** On R16 from the code worktree, with the lock held, through 06's
  `run_readers_worktree.sh`. That script is `run_readers.sh` with only its tree and its python
  changed ([`out/run_readers_worktree.diff`](06b/rehearsal/out/run_readers_worktree.diff), equal
  to 06's).
- **The results** are in [`out/readers.txt`](06b/rehearsal/out/readers.txt) and
  [`out/readers_check.txt`](06b/rehearsal/out/readers_check.txt).

| reader | exit | wall | files | store before = after |
|---|---|---|---|---|
| extract_Gk_data | 0 | 27 s | 468 | yes |
| extract_GkWKB_data | 0 | 50 s | 2436 | yes |
| extract_GkSource_data | 0 | 20 s | 180 | yes |
| extract_TkWKB_data | 0 | 15 s | 27 | yes |
| extract_tensor_source_data | 0 | 14 s | 27 | yes |
| extract_QuadSourceIntegral_data | 0 | 18 s | 76 | yes |

- **The outcome:** 0 Tracebacks, and no Ray process left after any reader.
- **The warning.** Each reader prints one
  `!! WARNING: number of shards read from database (=4) does not match specified number of shards
  (=10)`, as in 06.
- **The store.** R16's five files are byte-identical before and after the readers
  ([`state/r16_store_sha256_before_reads.txt`](06b/rehearsal/state/r16_store_sha256_before_reads.txt),
  [`state/r16_store_sha256_after_reads.txt`](06b/rehearsal/state/r16_store_sha256_after_reads.txt)),
  and equal to the hashes taken before the comparison.
- **The file counts** differ from 06's run on R14. The readers draw their subsample with unseeded
  `random.sample`.

#### At the end

- **The stores** are kept; their ten files' final hashes are
  [`state/stores_sha256_final.txt`](06b/rehearsal/state/stores_sha256_final.txt), equal to those
  before the comparison.
- **The registry** holds 27 runs, all finished.
- **The lock** is free, and `ray-processes` lists nothing.
- **The trees:** the main checkout is at `28ebaf0` and clean, and both worktrees are clean.

**§6.2 acceptance holds:**
- R15 and R16 are `done` with `1b332ffa…`;
- outside `QuadSourceIntegral`, every value table is identical;
- inside it, every row with no phase-limited flag in R15 is bit-identical, and the other 82 are
  tabulated;
- the readers exit 0, and the lock is released.
