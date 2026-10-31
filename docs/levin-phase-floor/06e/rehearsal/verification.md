### 6e.x Rehearsals R19 and R20, and the value comparison (prompt 06e §4)

**Prompt 06e §4, run 2026-10-30, 07:43–08:05 UTC, on the Mac Studio** (mps026298, Darwin 27.0.0).
The code measured is 06e's code commit `f4d8e17` (exact Chebyshev end nodes, and the measured
Clenshaw–Curtis node term), against its parent `50277f3`, whose code is 06d's.
- **The trees.** Both builds ran from git worktrees, so the main checkout did not move.
  - **R19** ran in `SecondaryGWKit-lpf06e-50277f3`, detached at `50277f3`. That is
    `f4d8e17`'s parent, and its code is the code 06c's census ran: `e5764a6..50277f3` touches only
    `CLAUDE.md` outside `docs/` and `prompts/`.
  - **R20** ran in `SecondaryGWKit-lpf06e-6d81fad`, detached at Part B's dispatch SHA `6d81fad`.
    Its code is `f4d8e17`'s: `f4d8e17..6d81fad` touches nothing outside `docs/` and `prompts/`.
  - **On code paths,** `50277f3..6d81fad` changes only `AdaptiveLevin/levin_quadrature.py` and
    four AdaptiveLevin tests.
- **The main checkout** stayed at `28ebaf0` and clean. It holds the venv, the stores, the registry
  and these records.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0 and `runregistrykit` 0.2.0.
- **The load** was 4–6 at the gate and about 11 at the end, on 10 cores, from desktop processes.

**Records** are in [`06e/rehearsal/`](06e/rehearsal/): `out/`, `scripts/`, `state/`, the agent's
`log.md`, this section as first written, and `issues.md`.
- **Commands.** Every command is in [`out/commands.log`](06e/rehearsal/out/commands.log).
- **Scripts, by SHA-256** (from [`out/02-gate-launcher.txt`](06e/rehearsal/out/02-gate-launcher.txt)
  and [`out/final_check.txt`](06e/rehearsal/out/final_check.txt)):
  - `classify_non_qsi.py` is `26690e61…`: 05's, copied unchanged, and byte-identical to 05's and
    06c's copies;
  - `run_readers_worktree.sh` is `ec96cf09…`, 06c's, copied unchanged;
  - `run_readers.sh` is `7c63a5ba…`;
  - the launcher `RunDrivers/quadsource_atol_sweep.py` is `b0135218…` in both worktrees;
  - `tools/store_value_diff.py` is `7997af85…` in both;
  - `ComputeTargets/QuadSourceIntegral.py` is `962424c0…` in both;
  - `AdaptiveLevin/levin_quadrature.py` is `ed0648dc…` in the parent worktree and `8285c1e4…` in
    the code worktree;
  - the tabulation `tabulate_qsi_endnodes.py` is `6b34c725…`. It is 06c's `tabulate_qsi_global.py`
    adapted, with the diff in
    [`out/tabulate_qsi_endnodes.diff`](06e/rehearsal/out/tabulate_qsi_endnodes.diff).
- **Not copied back:** the readers' `extract/` (180 MB).

#### The two builds

| | R19 (before) | R20 (after) |
|---|---|---|
| tree | worktree at `50277f3` (parent of `f4d8e17`; code 06d's) | worktree at `6d81fad` (code `f4d8e17`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `-B -u` and `PYTHONDONTWRITEBYTECODE=1`; `nohup sh -c`, detached | the same, in its own worktree |
| run | `levin-phase-floor-06e-r19-parent-20261030T074440` | `levin-phase-floor-06e-r20-endnodes-20261030T075034` |
| `sh` / driver pid | 98913 / 98917 | 99670 / 99675 |
| wall | 07:44:39 launch → 07:49:32 last heartbeat | 07:50:33 → 07:55:07 |
| manifest | `git_head` `50277f3e…`, `git_dirty` false, cwd the parent worktree, script SHA `b0135218…` | `git_head` `6d81fad4…`, `git_dirty` false, cwd the code worktree, script SHA `b0135218…` |
| cwd of the driver (`lsof`) | the parent worktree | the code worktree |
| end | `done`, exit 0; no Traceback; `stderr.log` empty; no `!! WARNING` | the same |
| store | `var/datastores/levin-phase-floor-06e-r19.sqlite` + 4 shards | `var/datastores/levin-phase-floor-06e-r20.sqlite` + 4 shards |

**The scope** was R8's rehearsal scope, launched as R17 and R18 were:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- `--job-name levin-phase-floor-06e`, `--register r19-parent` / `r20-endnodes`, and
  `--campaign levin-phase-floor --prompt 06e`.

**The worktrees were not written.** After each build, after the readers and at the end, each
worktree's `git status --short --ignored` was empty. `find -newer` the pre-R19 marker listed
nothing in either.

#### The lock

[`state/lock.log`](06e/rehearsal/state/lock.log). Each lock command's output was appended by `>>`.
Each was a shell command of its own, except row 4 (log, deviation 5).

| # | command | output | exit |
|---|---|---|---|
| 1 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |
| 2 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 06e section 4, rehearsals R19 … and R20 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 3 | `set-pid … 98917` (R19) | recorded | 0 |
| 4 | `ray-processes` (after R19; in a shared Bash call) | nothing | 0 |
| 5 | `ray-processes` (after R19, repeated on its own) | nothing | 0 |
| 6 | `set-pid … 99675` (R20) | recorded | 0 |
| 7 | `ray-processes` (after R20) | nothing | 0 |
| 8 | `ray-processes` (after the readers) | nothing | 0 |
| 9 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 10 | `status --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | free; no Ray processes | 0 |
| 11 | `ray-processes` | nothing | 0 |

#### The fingerprints (prompt §4.2)

| | R19 | R20 |
|---|---|---|
| `state` / `exit_code` | `done` / 0 | `done` / 0 |
| digest | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | the same |
| equal to R8's (`handover-12-r8-reference-20261025T210551`) | digest and every class | digest and every class |
| equal to each other | — | `classes` equal to R19's |
| `problems` | `{}` | `{}` |
| `taken` | `50277f3…`, `git_dirty` false, 07:49:30Z | `6d81fad…`, `git_dirty` false, 07:55:06Z |
| `QuadSourceIntegral` rows | 864 | 864 |

**Both fingerprints equal `1b332ffa…`: holds.** The fingerprint certifies identities, tags and row
counts, not values (`CLAUDE.md` rule 7). The values follow.

#### Outside `QuadSourceIntegral` (prompt §4.3)

`tools.store_value_diff` R19 against R20, `--json` (from the code worktree, 16 s, exit 1 because
`QuadSourceIntegral` differs), classified by 05's `classify_non_qsi.py`:
[`out/classify_non_qsi.txt`](06e/rehearsal/out/classify_non_qsi.txt).

| class | rows identical / differ | value-table rows identical | differing columns (kind: rows) |
|---|---|---|---|
| `version`, `store_tag`, `redshift`, `wavenumber`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`, `GkSourcePolicy`, `QuadSourcePolicy`, `GkSourcePolicyData` | all identical | — | none |
| `wavenumber_exit_time` | 0 / 3 | — | `compute_time` (timing: 3) |
| `BackgroundModel` | 0 / 1 | 1,159 | `compute_time`, `max_RHS_time`, `mean_RHS_time` (timing: 1 each) |
| `TkNumericIntegration` | 0 / 3 | 1,478 | timings (3), `min_RHS_time` (timing: 3), `label` (3) |
| `TkWKBIntegration` | 0 / 3 | 2,209 | `label` (3), `stage_1_compute_time` (timing: 3) |
| `GkNumericIntegration` | 0 / 1,368 | 35,874 | timings (1,368; `min_RHS_time` 1,094), `label` (1,368) |
| `GkWKBIntegration` | 0 / 3,477 | 147,642 | `label` (3,477), `stage_1_compute_time` (timing: 3,474), `metadata` and `stage_1_RHS_evaluations` (`rho_reuse`: 32) |
| `GkSource` | 0 / 291 | 169,653 | `label` (291), `parents_digest` (291) |
| `QuadSource` | 0 / 6 | 2,447 | `compute_time` (timing: 6), `label` (6) |
| `OneLoopIntegral` | 0 rows | — | — |

- **Every value table is identical:** 360,462 value rows in 7 tables, with `value tables not
  identical: none` in every class.
- **Every other difference** is a timing, a label, `parents_digest` or `rho_reuse`: **0
  unclassified**.
- **No row is in one store only:** 0 only-in-A, 0 only-in-B and 0 unmatched, in every class.
- **Holds.**

#### Inside `QuadSourceIntegral` (prompt §4.3)

Every row is tabulated in [`out/qsi_rows.csv`](06e/rehearsal/out/qsi_rows.csv): 864 rows, with
`float.hex` of both totals and both `total_abserr`, $|\Delta T|$ against each, the region counts,
the maximum depths, `total_converged` and `total_phase_limited` in each. The summary is
[`out/qsi_tabulation.txt`](06e/rehearsal/out/qsi_tabulation.txt), and
[`out/qsi_extra_summary.txt`](06e/rehearsal/out/qsi_extra_summary.txt) lists the rows at depth
≥ 12 in either store and the one row whose `total_converged` changed.

**The reading is checked by the script itself:**
- 864 rows in each store, the same physical-key set;
- 48 columns, equal;
- tolerances equal row by row;
- 0 `total` mismatches against the tool's identity-matched report;
- the metadata field sets are equal on every row;
- 3,105 Levin calls in each store, the same per row;
- `WKB_Levin_num_regions` equals the sum of the per-call `regions`, and `total_converged` the AND
  of the per-call `converged`, on every row of both.

**Where the region counts and maximum depths are stored** (each shard's table
`QuadSourceIntegral`, the same files at both trees):
- `WKB_Levin_num_regions` (`Integer`, nullable): the sum of `num_regions` over every WKB Levin
  call of the row;
- `WKB_Levin_max_depth` (`Integer`, nullable): the maximum of the calls' `max_depth`. **No per-call
  depth is stored;**
- the per-call region counts are in `metadata` (JSON) at
  `WKB_Levin.subintervals[i].groups[j].regions`;
- the analytic oracle's region counts are at `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.regions`
  (no depth);
- `total_converged` (`Boolean`, 0/1) is the AND of the calls' `converged`;
- `total` and `total_abserr` are `Float(64)`.

Work counters (the `WKB_Levin_*` counts, depth included, the `numeric_quad_*` step counts, and the
metadata leaves `regions`, `simple_regions` and `evaluations`) are compared and tabulated, but are
not values. The log gives the reasons.

| | R19 | R20 |
|---|---|---|
| rows / rows with a WKB Levin call | 864 / 573 | 864 / 573 |
| `total` bit-identical to the other store (`float.hex`) | 361 rows: 291 without a Levin call, 70 with one | — |
| every value field bit-identical | 205 rows, all without a Levin call; the other 86 such rows differ only in the analytic oracle | — |
| `total_converged` | 863 | 864 (the one change, 0→1, is R19's depth-20 row) |
| unconverged Levin calls | 2 | 0 |
| rows with `total_phase_limited` | 231 | 195 (173 kept, 58 cleared, 22 new) |
| phase-limited Levin calls / analytic blocks (of 10,072) | 474 / 3,506 | 406 / 3,832 |
| `WKB_Levin_num_regions`, summed | 34,892 | 31,216 |
| `WKB_Levin_evaluations`, summed | 276,031 | 246,623 |
| analytic-block regions, summed | 35,996 | 28,785 |
| rows at depth ≥ 12 / at depth 20 | 46 / 1 | 11 / 0 |

- **Per row,** regions are equal in 319, more in R20 in 89 and fewer in 165. The maximum depth is
  equal in 463, deeper in R20 in 35 and shallower in 75.
- **R19's depth-20 row** ($k=10^6$, $q=3.16\times10^5$, $r=10^6$, $z=3971.4$, 06c's R18 row) is
  at depth 7 in R20, with 394 → 92 regions. It is converged in R20 and not in R19.
- **R20's ten rows deeper than 12** are all equilateral ($k=q=r$, $z \le 9{,}105$), converged in
  both stores. The deepest is 19 ($k=q=r=3.16\times10^5$, $z=1732.4$).
- **$e_{20}/e_{19}$** runs from 0.987 to 1.006, with median 1.000.

#### Z1 — consistency (prompt §4.3)

**The rule:** every row outside 06a's 720 stationary-point rows with finite `total_abserr` in both
stores has $|T_{20}-T_{19}| \le e_{19}+e_{20}$, in float64 as written. Rows infinite in either are
counted separately. The stationary-point rows are tabulated, not scored.

**How the stationary-point rows were identified.**
- **The rule used:** the 1e-7 records of 06a's census checkpoint (`c13685b4…`) with a call whose
  scan has `n_sign_changes > 0`. These are 720 rows, the rule of `analyse06c.py`'s `scan06a`.
- **Their keys:** each is mapped to its physical key (k, q, r, z_response) through 06a's
  `state/rows.json` (`aa523d28…`).
- **The match:** a rehearsal row is among them iff its own physical key is equal as floats.
- **The store holds 0 of them, and 0 rows of A3 v2 at all.** Only $k=10^5$ is shared, and none of
  the rehearsal's 96 response redshifts.

| | count |
|---|---|
| stationary-point rows (tabulated, not scored) | 0 |
| rows scored (finite `total_abserr` in both) | 864 |
| **violations** | **0** |
| rows with a non-finite `total_abserr` in either | 0 |

- $|\Delta T|/(e_{19}+e_{20})$: median $8.3\times10^{-9}$, p90 $4.4\times10^{-7}$, p99
  $9.5\times10^{-6}$, maximum 0.0031.
- **The largest $|\Delta T|/e$ in each store** is at $k=3.16\times10^5$, $q=10^5$,
  $r=3.16\times10^5$, $z=437173.17$:
  - $|\Delta T|/e_{19} = 0.00624$ and $|\Delta T|/e_{20} = 0.00626$;
  - total19 `0x1.49a315251da54p-54`, total20 `0x1.49a31528f01e6p-54`;
  - abserr19 `0x1.32806dde5be7fp-77`, abserr20 `0x1.31684a68e80c9p-77`.
- $|\Delta T|/|T_{19}|$, on rows with a Levin call: median $9.9\times10^{-15}$, maximum
  $1.3\times10^{-9}$.
- **Z1 holds.**

**Against 06c's comparison** (read-only cross-readings of 06c's committed `qsi_rows.csv`):
- **R19 reproduces 06c's R18 on all 864 rows** (the same code), in `total`, `total_abserr`,
  `total_converged`, regions, depth, evaluations and `analytic_rad`
  ([`out/r19_vs_06c_r18_csv_check.txt`](06e/rehearsal/out/r19_vs_06c_r18_csv_check.txt)).
- **On 06c's 30 Y1-violating rows,** Z1's ratio is at most $1.8\times10^{-6}$
  ([`out/z1_on_06c_y1_rows.txt`](06e/rehearsal/out/z1_on_06c_y1_rows.txt)). That includes the 12
  equilateral `G+q+r` rows, whose Y1 ratio was 336–16,763. 06e leaves those totals where 06c and
  06d put them.

#### The readers (prompt §4.4)

From the code worktree, on R20, under the lock, with 06c's `run_readers_worktree.sh`
([`out/readers.txt`](06e/rehearsal/out/readers.txt),
[`out/readers_check.txt`](06e/rehearsal/out/readers_check.txt)):

| reader | exit | wall | files | store before = after |
|---|---|---|---|---|
| `extract_Gk_data` | 0 | 24 s | 405 | yes |
| `extract_GkWKB_data` | 0 | 43 s | 2,436 | yes |
| `extract_GkSource_data` | 0 | 19 s | 164 | yes |
| `extract_TkWKB_data` | 0 | 15 s | 27 | yes |
| `extract_tensor_source_data` | 0 | 13 s | 27 | yes |
| `extract_QuadSourceIntegral_data` | 0 | 16 s | 68 | yes |

- **The store** is identical by SHA-256, size and mtime, with no journal, before and after each
  reader. `state/r20_store_sha256_before_reads.txt` equals `…_after_reads.txt`:

  | file | SHA-256 |
  |---|---|
  | primary | `34ca43d5…` |
  | shard0000 | `e909e835…` |
  | shard0001 | `244ffe2e…` |
  | shard0002 | `7f811f62…` |
  | shard0003 | `e90db3b7…` |

- **Ray:** each reader starts a local Ray in `/tmp/ray-sgk`, and none left a Ray process.
- **Logs:** 0 Tracebacks. The six `!! WARNING … shards (=4) … (=10)` are as in 06b and 06c.
- **The file counts differ from 06c's** because three readers draw unseeded random subsamples.
- **Holds.**

#### The release, and the stores (prompt §4.5, §4.6)

- **The lock:** released, exit 0. `status` says free with no Ray processes, and `ray-processes`
  lists nothing. **Holds.**
- **The stores are kept,** unchanged since the comparison (`state/stores_sha256_final.txt`):
  - R19's primary `0e194eee…`, shards `25de48d3…`, `37b64f25…`, `cd972580…` and `a07918a5…`;
  - R20's, as above.

#### §4 acceptance (prompt §6 item 2)

R19 and R20 are both `done` with `1b332ffa…`. Outside `QuadSourceIntegral` every value table is
identical. Inside it every row is tabulated, and Z1 is scored: it **holds**. The readers exit 0,
and the lock is released. **§4 holds in every clause.**
