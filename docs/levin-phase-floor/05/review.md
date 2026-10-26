# levin-phase-floor 05, Part B — the Mac Studio orchestrator's review (§B.3)

Host mps026298 (the Mac Studio), 2026-10-26. Main checkout `/Users/ds283/Documents/Code/SecondaryGWKit`
at `53084dcb14f7835e22fe8ffde8c0ac09c20a6fca`; worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951`
at `e6489519ee662434d1f8e430e1c172246678ac8c`. Marker touched 01:44:30.712Z, immediately before the
first dispatch. The outputs named below are in `review/` beside this file. My scratch scripts
(`check6_qs.py`, `check8_sample.py`, `check8_null.py`, `check9_score.py`) are copied there too.

**Verdict: stop, on check 10 alone.** Checks 1–9 pass. In check 10, the wider agent's log has two
deviations (12 and 14) that carry none of the three classifications. Item 14 says it is not a
deviation of method. Item 12 is recorded as one: "Recorded as the user's decision". It is reported
here and not repaired.

One reading was applied, on the user's ruling. It was relayed verbatim by the laptop orchestrator
after the rehearsal agent's §8 stop:

> "R9/R10 differences: Those columns cannot be expected to be identical. So I think that's just an
> incorrectly-worded prompt. Do whatever you were already planning to do with the lock. Only this
> project is running code on the Mac Studio right now."

So check 4's "every class but QuadSourceIntegral is value-identical" is read as follows: identical
except in the timing columns, `label`, `GkSource.parents_digest` and the GkWKBIntegration work
counters (`rho_evals`, `rho_reused`, `stage_1_RHS_evaluations`), with every value table identical.

## The Gate (before dispatch, 01:44Z)

- `git rev-parse HEAD`: main `53084dcb…`; worktree `e6489519…`. `git status --short` is empty in
  both. `git rev-parse 120fdfe^` is `e6489519…`. `git diff --stat 120fdfe 53084dc -- . ':!prompts'`
  is empty.
- `runregistrykit list`: 13 runs, 13 finished, none running or stale.
- `raypoolkit.rehearsal_lock status`: `lock: free (… does not exist)`, exit 0. `~/.raypoolkit/` is
  empty, and `~/ScienceData/.rehearsal-lock` does not exist.
- There is no `var/levin-phase-floor-05`, and no `var/datastores/levin-phase-floor-05-*`.
- `store show var/datastores/handover-A3-v2-lambdacdm.sqlite`: `problems: none`, digest `0e32f24a2dc2f4c9…`.
- `~/.ray/auth_token` is mode 600. 118 GiB free; load 3.4–5.2.
- A3 v2 hashed: `review/a3v2_sha256_before.txt`.

## 1. Scope — pass

`review/check1_scope.txt`:
- main `53084dcb…` and worktree `e6489519…`, each with `git status --short` empty. The worktree's
  `git status --short --ignored` is also empty.
- `find <worktree> -newer marker -not -path '*/__pycache__/*'` prints nothing. It prints nothing
  with `__pycache__` included, either.
- `find var -newer marker` prints only these:
  - `var/datastores/levin-phase-floor-05-r9*` and `-r10*` (10 files);
  - `var/levin-phase-floor-05/{rehearsal,wider,review}`;
  - the run directories `var/runs/levin-phase-floor-05-{r9-parent-20261026T014729,r10-reanchor-20261026T015120,wider-20261026T020005}`;
  - `var/runs/levin-phase-floor-05-r9.launch.log` and `-r10.launch.log`.
- A3 v2: `shasum -a 256 var/datastores/handover-A3-v2-lambdacdm*` gives the same six hashes before
  (`review/a3v2_sha256_before.txt`), after the agents (`review/a3v2_sha256_after.txt`) and after my
  check 6 (`review/a3v2_sha256_after_check6.txt`). `diff` prints nothing, and there is no journal,
  `-wal` or `-shm` file.

## 2. The runs — pass

`runregistrykit list` (`review/check2_list.txt`) shows all three `done`:
`levin-phase-floor-05-wider-20261026T020005` (109/109),
`levin-phase-floor-05-r10-reanchor-20261026T015120` and
`levin-phase-floor-05-r9-parent-20261026T014729`.

| manifest | campaign | prompt | slug | git_head | git_dirty | cwd |
|---|---|---|---|---|---|---|
| R9 | levin-phase-floor | 05 | r9-parent | `e6489519…` (parent) | false | worktree |
| R10 | levin-phase-floor | 05 | r10-reanchor | `53084dcb…` (dispatch SHA) | false | main checkout |
| wider | levin-phase-floor | 05 | wider | `53084dcb…` | false | main `var/levin-phase-floor-05/wider` |

- R9 and R10 both ran the launcher `RunDrivers/quadsource_atol_sweep.py`, SHA-256 `5b984c75…` (R8's),
  with `--build --rehearsal --atol 1e-32 --rtol 1e-7 --cpus 6`.
- R9's `--database` is the main checkout's `var/datastores/levin-phase-floor-05-r9.sqlite`, by
  absolute path.
- The wider manifest records the main checkout because its driver lives there. The p tree is
  recorded per checkpoint (check 7).

## 3. The fingerprints — pass

- Both `status.json` files give fingerprint digest `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, problems `{}`:
  - R9: exit 0, `done`, pid 47770, `taken.git_head` `e6489519…`;
  - R10: exit 0, `done`, pid 49752, `taken.git_head` `53084dcb…`.
- I re-ran `./venv/bin/python -m runregistrykit store fingerprint var/datastores/levin-phase-floor-05-{r9,r10}.sqlite`,
  without `--write` (`review/fingerprint_r9.txt`, `fingerprint_r10.txt`). Each exits 0 with
  `digest: 1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` and `problems: none`.
  The two listings differ only in the `store:` line.

## 4. The values — pass (on the user's ruling above)

- **The R9/R10 store files** hash the same as the agent's after its comparison and at its end
  (`review/r9r10_sha256_before_review.txt`). The primaries are R9 `cfba7ac4…` and R10 `557066be…`.
- **The comparison re-run.** I ran `PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff var/datastores/levin-phase-floor-05-r9.sqlite var/datastores/levin-phase-floor-05-r10.sqlite --json`
  into the scratchpad. It exits 1 in 16 s. Its SHA-256 is `55a5451c3f336c10…`, equal to the saved
  `rehearsal/out/store_value_diff.json` (41,065,812 bytes each), so the JSON is reproduced byte for
  byte.
- **The classes, from my JSON** (`review/check4_classes.txt`). `not_compared: []`. In every class,
  0 rows are only in one store, 0 cannot be matched and 0 columns are unresolvable.
  - Twelve classes are identical.
  - Eight classes outside QuadSourceIntegral differ only in these columns:
    - wavenumber_exit_time: `compute_time`;
    - BackgroundModel: `compute_time`, `max_RHS_time`, `mean_RHS_time`;
    - TkNumericIntegration and GkNumericIntegration: `compute_time`, `label` and the `*_RHS_time` columns;
    - TkWKBIntegration: `label`, `stage_1_compute_time`;
    - GkWKBIntegration: `label`, `stage_1_compute_time`, and in 25 rows `stage_1_RHS_evaluations`
      and `metadata` keys `rho_evals`/`rho_reused` only;
    - GkSource: `label`, `parents_digest`;
    - QuadSource: `compute_time`, `label`.
  - Every value table is identical: BackgroundModelValue 1159, TkNumericValue 1478, TkWKBValue 2209,
    GkNumericValue 35874, GkWKBValue 147642, GkSourceValue 169653, QuadSourceValue 2447.
  - My classifier's list of differences outside the ruled column kinds is `BAD []`.
  - QuadSourceIntegral: 864/864 rows differ, among them `total` (542 rows) and `total_abserr`.
    So R9 and R10 are not value-identical there.
- **The spot-check** (`review/check4_spot.txt`). I read row (k, q, r, z_response) =
  (1e5, 1e5, 1e5, 249.99999999999994) with `sqlite3` `mode=ro`:
  - R9 (shard 3, serial 92): `total` `-0x1.1f665c75480dap-59`;
  - R10 (shard 3, serial 103): `total` `-0x1.1f665c7aeff2cp-59`.
  - These equal the first row of `rehearsal/out/qsi_rows.csv` (`total_A`, `total_B`) bit for bit.
    The serials differ, so the pairing is by identity, not by serial.

## 5. The readers and the lock — pass

- **The readers** (`rehearsal/out/readers.txt`): six readers, each `exit 0`, and
  `run_readers.sh exit 0`.
- **R10's five files** have the same SHA-256 before and after the readers
  (`rehearsal/state/r10_store_sha256_{before,after}_reads.txt`), equal to my hashes in check 4.
- **The lock transcript** (`rehearsal/out/lock.log`):
  - `acquired …`, exit 0;
  - `recorded pid 47770`, exit 0 (R9's driver), and `recorded pid 49752`, exit 0 (R10's driver);
  - `ray-processes` none, exit 0;
  - `released the rehearsal lock`, exit 0;
  - `lock: free`.
- **My own look** (`review/check5_lock.txt`), each a command of its own with output by `>>`:
  - `ray-processes --ray-temp-dir /tmp/ray-sgk`: exit 0, `Ray processes of /tmp/ray-sgk: none`;
  - `status --ray-temp-dir /tmp/ray-sgk`: `lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)`, exit 0.

## 6. L0 and the store — pass

- **The saved L0.** `wider/out/L0.txt` (01:57:24Z) and its re-run `wider/out/L0_final.txt`
  (01:59:59Z) are the same apart from the ledger path and wall times. Both runs were before the
  launch (manifest 02:00:05Z).
  - (a) on `low-G+r-1` and `ctrl-low-1`: 360 stored nodes, 363 recomputed, 360 common, 0
    (node, column) pairs differing in bits; added `[361, 362, 363]`, which are B1's
    (`HANDOVER_OVERLAP_NODES = 2`); removed `[]`.
  - (b) the two runs are identical in the result (68 floats), the calls, the regions and the rest.
  - Bits are compared with `float.hex` and `==`, never `isclose`.
- **Independently, on R1** (shard 2, QuadSourceIntegral serial 37527, depth 20 stored and under
  p). `review/check6_qs.py` and its output `review/check6_out.txt`:
  - I ran the agent's own `rebuild_row` once at p, with `PYTHONPATH` the worktree, `-B` and
    `PYTHONDONTWRITEBYTECODE=1`, wrapping `recompute_QuadSource` to capture its values.
  - I read A3 v2's `QuadSourceValue` rows for the stored QuadSource (serial 5, the shard whose
    `Tq_serial`/`Tr_serial` are (1, 601), which is shard 3) with `sqlite3` `mode=ro`.
  - Result: `common nodes 559; (node,column) pairs compared 5031; differing in float.hex 0;
    recomputed-only z serials [471, 472, 473]; stored-only []`.
- **The store afterwards.**
  - Its hashes are unchanged (check 1), and there is no journal, `-wal` or `-shm` file.
  - `./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite`
    (`review/fingerprint_a3v2.txt`): exit 0, `digest: 0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`,
    `problems: none`.
  - The worktree is still empty under `git status --short --ignored`.

## 7. The trees — pass

- `run_wider.py` `worker_env` sets `PYTHONPATH` to the tree, `PYTHONDONTWRITEBYTECODE=1`, and
  `OMP_NUM_THREADS=1`.
- **The run's `worker_p_hello.json`:**
  - `PYTHONPATH /Users/ds283/Documents/Code/SecondaryGWKit-lpf05-e648951`, head `e6489519…`, status empty;
  - every module from the worktree: `ComputeTargets/__init__.py`, `primitive_phase.py`,
    `QuadSource.py`, `QuadSourceIntegral.py`, `GkSourcePolicyData.py`, `StoreFixtures`, `main.py`;
  - `PrimitivePhase_has_exact_remainder False`.
- **`worker_c_hello.json`:** the same modules from the main checkout, head `53084dcb…`,
  `PrimitivePhase_has_exact_remainder True`.
- **The dispatcher** (`wider/scripts/dispatch.py`) replaces `PrimitivePhase.theta_mod_2pi` with a
  pass-through. It counts calls, raises the budget at entry, and otherwise returns
  `ORIG_THETA_MOD_2PI(phase, x, x_is_log)`, the tree's own production method. The original is
  restored in a `finally`, and the worker asserts it after every unit.
- **Over the checkpoint's 109 records:**
  - calls: p G `exact_remainder=None` 24,192,372, p T 14,924,742; c G `exact_remainder=True`
    268,136, c T `exact_remainder=False` 167,562;
  - every p record has head `e6489519…` and every c record `53084dcb…`, all with empty status;
  - every record carries `script_sha256` `491d2608…`.

## 8. The sample and the predictions — pass

- **`state/sample.md`** (01:51:40Z) is older than its script `draw_sample.py` (01:52:16Z), than
  `sample.json` and than the first checkpoint (02:00:08Z). It states the rule (quoting §6) and,
  under "The rule's bias", that it prefers cheap rows.
- **The rows, redrawn by me.** `review/check8_sample.py` reads `groups_01.csv`;
  `review/check8_null.py` reads the store with `mode=ro`. Output: `review/check8_sample_out.txt`.
  - The 24 cell rows and 6 band controls are exactly the agent's, in order: `MATCH`.
  - The `NULL` controls: 2489 candidates, taken `[(3, 452), (1, 20111)]`, equal to the agent's.
- **`state/predictions.md`** (01:57:36Z) is later than L0's output `out/L0.txt` (01:57:24Z) and
  earlier than the run's `manifest.json` (02:00:05Z).
  - Prompt lines 292–315 at `53084dc` appear in it verbatim: `True`, 1576 characters.
  - L0 was re-run after it (`L0_final.txt`, 01:59:59Z), following a fix to `worker.py`'s handling
    of a `None` max depth. The body is identical, and the predictions are the prompt's text.
    The log records this as deviation 9.

## 9. The verdicts, re-scored — pass (my scoring reproduces the agent's)

My script is `review/check9_score.py`, with output `review/check9_score_out.txt`. It reads
`checkpoint.jsonl` and the unit pickles. A scored group is one with at least 10 regions at depth
≥ 18 under p; a row's regions are `WKB_Levin_data.num_regions`.

- **Units:** 109, all `done`. **Over-budget units: none**, and no unit errored.
- **L1: fails** (9/24). Clause 1 is true on all 24 rows. Clause 2 (c ≤ p/10) is false on 15:
  - 14 rows not capped under p, with c/p from 0.73 to 1.09;
  - R5, with c/p 0.3455 (p: depth 20, 55 regions, and no scored group).
- **L1q: fails** (1/7). R7 is true with c/p 0.0016. The six drawn `Gqr` rows are false: none is
  capped under p, and c/p runs from 1.0 to 1.35.
- **L2: holds** (8/8). The ratios are 0.161, 0.652, 0.0109, 0.00409, 4.29e-12 and 6.58e-10, and
  the two `NULL` controls are bit-identical.
- **L3: fails** (29/31). It is false on `mid-G-r-3` (1.294) and `mid-G-3` (1.698). No row was out
  of budget.
- **The rows capped under p (p max depth 20), not scored:** 11 rows. Clause 1 is true on all 11,
  and the rule on 10/11; R5 fails clause 2 only. On these rows L3's largest ratio is 0.166, and it
  holds on all of them.

| row | p depth, regions | c depth, regions | c/p |
|---|---|---|---|
| low-G+r-3 | 20, 1340 | 8, 33 | 0.0246 |
| low-G-r-2 | 20, 15068 | 7, 42 | 0.0028 |
| low-G-r-3 | 20, 6822 | 8, 40 | 0.0059 |
| low-G-1 | 20, 3902 | 11, 34 | 0.0087 |
| low-G-3 | 20, 831 | 7, 39 | 0.0469 |
| R1 | 20, 22852 | 8, 63 | 0.0028 |
| R3 | 20, 10457 | 5, 18 | 0.0017 |
| R4 | 20, 42830 | 8, 83 | 0.0019 |
| R5 | 20, 55 | 5, 19 | 0.3455 |
| R6 | 20, 5130 | 8, 28 | 0.0055 |
| R7 (Gqr) | 20, 16660 | 8, 27 | 0.0016 |

- **The descriptive items** are all present in `wider/out/wider_05.txt`:
  - N4, wider;
  - cost and regions by class and band, which covers low against mid;
  - `total_converged` from p to c at 1e-7 and at 1e-8;
  - R1–R7 against 04's arms;
  - for the board: the stored total against p's total, in stored errors.

## 10. The records — **fails** (the wider log's deviations 12 and 14)

- **The two `log.md`s** each have README §5.1's headings: What shipped, Predictions and verdicts,
  Deviations from the prompt, Verification performed, Observations not acted on, State handed on.
  - The rehearsal log's 10 deviations are all classified: 2 STRUCTURALLY REQUIRED (the §8 ruling,
    with the user's words; the worktree), 6 IMPLEMENTATION CHOICE and 2 UNINTENDED DRIFT.
  - The wider log has 14 deviations; 1–11 and 13 are classified (2 SR, 7 IC, 1 UD), the worktree
    being deviation 5 (SR). **Deviation 12**, reading "depth-20 row" as the sample's rows with the
    user's decision quoted, is marked "**Recorded as the user's decision** … it changes no rule"
    and carries no classification. **Deviation 14**, R9/R10 sharing the host, is marked
    "**Recorded; not a deviation of method.**"
- **Both `verification.md`s** are written as `### 5.x …` subsections of `VERIFICATION.md` §5,
  with every link in the `05/rehearsal/…` or `05/wider/…` form.
- **`issues.md`:**
  - rehearsal: P1 `[05-store-value-diff-compares-volatile-columns]` (proposed) and P2;
  - wider: W1–W9.
