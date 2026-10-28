# Review — prompt 06b Part B (§4 rehearsals R15/R16, §5 census), Mac Studio

**Reviewer:** the Mac Studio orchestrator session, 2026-10-28, under orchestrator note
`prompt-06b.md` §B, which falls back to `prompt-07.md` §B.3 where it is silent. The note was read
in the code worktree at `92bc62f`.

**Verdict: STOP, on check 10 alone** (the census log's records form, two points below). Checks 1–9
pass, and every measurement re-checked here agrees with the agents.

The orchestrator's readers and their outputs are in [`review/`](review/). Every float comparison
below is by bit pattern (`float.hex`), never `isclose`. The scratch readers are the orchestrator's
own, not the agents' scripts, except where a check says to re-run one.

## Dispatch facts

- **Marker:** `touch`ed at 2026-10-28T13:51:55.228Z in the session scratchpad, before the first
  dispatch.
- **The briefs** are [`review/dispatch_rehearsal.md`](review/dispatch_rehearsal.md) (sha256
  `74b78530…`) and [`review/dispatch_census.md`](review/dispatch_census.md) (sha256 `926f43d3…`).
  Each is the common brief, plus 06b §B.5 verbatim, plus the agent's own scope.
- **Pre-flight, read-only, before the marker:**
  - three trees at `28ebaf0`, `da5c24d` and `92bc62f`, all clean;
  - the registry held 25 runs, all finished;
  - the lock was free, and `ray-processes` listed none;
  - there was no `var/levin-phase-floor-06b*`;
  - the scratch copy matched `copy_files_certified.txt`, with no journal file;
  - the load average of about 20 came from Spotlight indexing the new worktrees, with no Python
    running.
- **The baseline:** 3,878 protected files were hashed before dispatch (A3 v2, the scratch copy,
  `var/levin-phase-floor-06{,a}/`, and 06's and 06a's run directories), and the `var/` file list
  was saved.
- **Sequencing, an orchestrator choice:**
  - the census agent prepared and ran Gate W0 while R15 and R16 ran;
  - its launch was held until the lock was released, so that the 300 s budget and the wall time
    were not shared with Ray builds;
  - the note says the two "may" run together, and does not require it.
- **One question was relayed verbatim** (the census agent's reading of W1/W4 for analytic-block
  calls). The user ruled, via the laptop: *"For 06b, that reading is reasonable. I don't think it's
  worth spending a lot of time worrying about it."* The ruling was passed to the agent verbatim
  before the launch.
- **The brief's own wording drifted from the note.** It said "no census process imports `ray`",
  where the note's rule is that no Ray session appears. The census agent recorded this as its
  deviation 17. The laptop classifies it as arising from the brief.

## Check 1 — Scope: **pass**

- **Trees** ([`review/check1a.txt`](review/check1a.txt)):
  - the main checkout is at `28ebaf0`, and `git status --short` is empty;
  - the parent worktree is at `da5c24d`, and the code worktree at `92bc62f`;
  - both worktrees are empty under `--ignored`, and `find <worktree> -newer <marker>` lists
    nothing, with no `__pycache__`.
- **`find var -newer <marker> -type f`** lists only:
  - `var/datastores/levin-phase-floor-06b-r1{5,6}*` (10 files);
  - `var/levin-phase-floor-06b/{rehearsal,census}`;
  - `var/runs/levin-phase-floor-06b-{r15-parent-…,r16-premise-…,census-…}`;
  - `var/runs/levin-phase-floor-06b-r1{5,6}.launch.log`.

  (`review/` and this file were written after that listing.)
- **Protected files** ([`review/check1b.txt`](review/check1b.txt)): `shasum -a 256 -c` over the
  3,878 baseline files is all OK. No file under `var/` was removed.

## Check 2 — The runs: **pass** (one reading noted)

From [`review/check2.txt`](review/check2.txt). `runregistrykit list` holds 28 runs, all finished.

| Run | State | Manifest | `git_head` | Status |
|---|---|---|---|---|
| `levin-phase-floor-06b-r15-parent-20261028T135349` | `done` | `levin-phase-floor`, `06b` | `da5c24d…`, `git_dirty` False, cwd the parent worktree | exit 0 |
| `levin-phase-floor-06b-r16-premise-20261028T135745` | `done` | `levin-phase-floor`, `06b` | `92bc62f…`, `git_dirty` False, cwd the code worktree | exit 0 |
| `levin-phase-floor-06b-census-20261028T161212` | `done`, 18560/18560 | `levin-phase-floor`, `06b` | `28ebaf0` (the main checkout's), `git_dirty` False | exit 0 |

- **Reading noted:** the census manifest's `git_head` is the main checkout's, because the harness
  runs from `var/levin-phase-floor-06b/census`. The same was true of 06a's census, which 06a's
  review accepted (06a review check 2).
- The census names the tree it ran from in two places:
  - the purpose string names `92bc62f` and its worktree;
  - every one of the 18,560 checkpoint records carries `tree_head` `92bc62f…` with empty
    `tree_status_short` (check 7).

## Check 3 — The fingerprints: **pass**

- R15 and R16's `status.json` both record
  `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`, with problems `{}`. They were
  taken at `da5c24d` and `92bc62f` respectively, with `git_dirty` false.
- The orchestrator re-ran the fingerprint on each store, without `--write`
  ([`review/fp_r15.txt`](review/fp_r15.txt), [`review/fp_r16.txt`](review/fp_r16.txt)). Each
  printed `digest: 1b332ffa…` and exited 0:
  ```
  cd <code worktree>; PYTHONPATH=. <venv>/python -B -m runregistrykit store fingerprint var/datastores/levin-phase-floor-06b-r15.sqlite
  cd <code worktree>; PYTHONPATH=. <venv>/python -B -m runregistrykit store fingerprint var/datastores/levin-phase-floor-06b-r16.sqlite
  ```

## Check 4 — The values: **pass**

- **`tools.store_value_diff` re-run** from the code worktree with `--json`. It exits 1, as expected,
  because `QuadSourceIntegral` differs. Its JSON is **byte-identical** to the agent's
  `rehearsal/out/store_value_diff.json` (`cmp`; both sha256 `51b9b94a…`).
- **Outside `QuadSourceIntegral`** ([`review/svd_summary.txt`](review/svd_summary.txt)):
  - all 7 value tables are identical, 360,462 rows: BackgroundModelValue 1,159, TkNumericValue
    1,478, TkWKBValue 2,209, GkNumericValue 35,874, GkWKBValue 147,642, GkSourceValue 169,653,
    QuadSourceValue 2,447;
  - no class has a row in one store only, an unmatched row or an unresolvable column;
    `not_compared` is empty;
  - the object-column differences are all timings, `label`, `parents_digest`, and GkWKB's
    residual-memo counters on 23 rows (`stage_1_RHS_evaluations`, `metadata.rho_evals` and
    `rho_reused`). These are the volatile kinds.
- **Inside `QuadSourceIntegral`**, by the orchestrator's own `mode=ro` reader
  ([`review/qsi_reader.py`](review/qsi_reader.py), [`review/qsi_reader.out`](review/qsi_reader.out)):
  - **Row matching.** Row serials differ between the stores, so rows are matched by physical
    identity: $k$, $q$ and $r$ (via `wavenumber_exit_time` → `wavenumber.k_inv_Mpc`), atol and rtol
    (`log10_tol`), both policies' `Levin_threshold`, and $z_{\rm response}$ and
    $z_{\rm source,max}$. The identity sets are equal: 864 rows in each.
  - **Phase-limited in R15, re-derived from the three flags:**
    - (a) `total_phase_limited`: 26 rows;
    - (b) `metadata.WKB_Levin.subintervals[*].groups[*].phase_limited`: 26 rows, 48 flags;
    - (c) `metadata.analytic.*.phase_limited`: 70 rows, 143 flags;
    - **82 rows flagged in all**, which equals the agent's set and counts.
  - **R16 has no flagged row.**
  - **The 782 unflagged rows are bit-identical:** 0 violations in `b`, `total`, `total_abserr`,
    `total_converged`, `numeric_quad`, `WKB_Levin`, `analytic_rad`, `eta_*`, and every
    `analytic.*` and `WKB_Levin.*` metadata leaf. The work counters and timings (`regions`,
    `simple_regions`, `evaluations`, `elapsed`, `compute_time`) were set aside.
  - **`total` moved on 26 rows,** all with a (b) flag. The largest change is $|\Delta T| = 0.360\times$
    R15's `total_abserr` and $0.597\times$ R16's. The agent's tabulation of the 82 rows
    (`rehearsal/out/qsi_tabulation.txt`, `qsi_rows.csv`) is complete.
  - **Spot-check, the largest mover,** shard 0 (serials R15 10286, R16 10091):

    | | `total` | `total_abserr` |
    |---|---|---|
    | R15 | `0x1.13fa730737be4p-67` | `0x1.1afeca4a03b83p-90` |
    | R16 | `0x1.13fa723b8dee4p-67` | `0x1.69f54490326c6p-91` |

## Check 5 — Readers and lock: **pass**

(P1's rows belong to 07 and do not apply to 06b.)
- **The readers:** six, run on R16 from the code worktree through 06's `run_readers_worktree.sh`,
  copied unchanged (the agent's deviation 7, STRUCTURALLY REQUIRED). All exit 0
  (`rehearsal/out/readers.txt`).
- **R16 unchanged by the readers:** its hashes are identical before and after
  (`diff state/r16_store_sha256_{before,after}_reads.txt`: identical).
- **All ten store files** now equal `rehearsal/state/stores_sha256_final.txt`.
- **The lock transcript** (`rehearsal/state/lock.log`):
  - `acquire` 0;
  - `set-pid 96498` (R15's driver) 0;
  - `set-pid 97360` (R16's driver) 0;
  - `release` 0, then `status` free.
- **Run by the orchestrator,** each as a shell command of its own with output by `>>`
  ([`review/check5_lock.txt`](review/check5_lock.txt)):
  - `ray-processes --ray-temp-dir /tmp/ray-sgk` exits 0 with nothing listed;
  - `status` prints `lock: free (… does not exist)`.

## Check 6 — W0 and the store: **pass**

- **W0's three clauses are in the saved outputs**, and all hold:
  - `census/out/cmd/15_W0_clause1.txt`;
  - `census/out/cmd/14_W0_check_rerun.txt`, which ends at 14:10:17Z.
- **W0 ran before the launch:** the outputs' mtimes are 14:10:17Z and 14:10:24Z, and the manifest
  was created at 16:12:12Z.
- **Independent comparison:** `ctrl-low-1`'s parent unit at `1e-7` (`p-1e-7-2-2095`, from
  `state/W0/P2/emit.jsonl`, `tree_head da5c24d`) against 06a's `h-1e-7-2-2095` in 06a's
  `checkpoint.jsonl`:
  - `total` is `-0x1.e3272e29a4c5dp-102` in both;
  - `total_abserr` is `0x1.ef96f9f3976bcp-107` in both;
  - all 7 calls' value and abserr are equal.
- **The copy:**
  - its five files equal the certification by sha256 after the run (orchestrator, in check 1's
    baseline re-check), with no journal, `-wal` or `-shm` file;
  - the agent's `fingerprint_copy.py` at $b$ gives `0e32f24a…`, problems none
    (`census/out/cmd/30`).
  - The orchestrator did not re-run `fingerprint_copy.py` itself. The byte-identity of the copy's
    files to the certified files, whose fingerprint is `0e32f24a…`, carries it.

## Check 7 — The trees, read: **pass**

- **The parent unit `p-1e-7-2-2095`:**
  - modules by root are 102 from its own tree, with `foreign` empty;
  - `AdaptiveLevin.levin_quadrature` was loaded from
    `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-da5c24d/AdaptiveLevin/levin_quadrature.py`.
- **The $b$ unit `b-1e-7-3-34455`:**
  - modules by root are 102 from its own tree, with `foreign` empty;
  - `levin_quadrature` was loaded from `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06b-92bc62f/…`.
- **All 18,560 $b$ records** carry `tree_head 92bc62f…` and empty `tree_status_short`
  ([`review/rescore.out`](review/rescore.out)).
- **$b$'s results carry the premise counters** (`premise_checks`, `premise_failed`) and `w3` per
  1e-7 unit.

## Check 8 — The predictions: **pass**

- **Timing:** `state/predictions.md` (sha256 `8df51796…`) has mtime 14:13:55.8Z. That is later than
  W0's outputs (14:10:24Z) and earlier than the manifest (16:12:12Z).
- **The text:** prompt 06b's lines 297–319, from "W1 — nothing else moves" to the end of W4,
  1,555 chars, are a **substring** of `predictions.md`. The prompt is quoted word for word.
- **W2's selection rule** is fixed in Part B, mechanically. The selection gives 12 low and 12 mid,
  every selected call moved, and the strata are spread over every `k_key`.
- **The harness diff** (`census/out/scripts_diff_against_06a_final.txt`):
  - changed: `prov.py`, `worker.py`, `run_census.py`, `launch.py` and `harness.py`;
  - new: `make_capture.py`, `w0.py`, `w0_clause1.py`, `w2_reference.py`, `check_predictions.py`,
    `scripts_equal.py`, `analyse06b.py`, `w2_crosscheck.py` and `w2_request.py`.

  Each is named in the census log's deviations, 1–8, 10, 12, 16, 20 and 22.

## Check 9 — The verdicts, re-scored: **pass** (each agrees with the agent)

By [`review/rescore.py`](review/rescore.py) from the two `checkpoint.jsonl`s, and
[`review/w2_ref_mine.py`](review/w2_ref_mine.py) for W2. The outputs are
[`review/rescore.out`](review/rescore.out) and [`review/w2_ref_mine.out`](review/w2_ref_mine.out).
- **The run:** 18,560 units in each run, the same keys, every $b$ unit `done`, none over budget or
  in error. The largest $b$ unit wall is 158.9 s, against a budget of 300 s.
- **W1 — holds.**
  - At `1e-7`: 36,112 of 36,112 group calls not phase-limited in 06a equal 06a's in value, abserr,
    `num_regions` and `max_depth`. 6,777 of 6,777 rows with no phase-limited call (group or analytic,
    per the ruled reading) equal 06a's in `total` and `total_abserr`.
  - At `1e-8`: 35,605 of 35,605 calls and 6,177 of 6,177 rows.
  - The ruled reading's cross-check on group calls (premise asked in $b$ ⇔ phase-limited in 06a)
    has 0 disagreements at both rtols.
- **W2 — fails (17 of 24).**
  - The orchestrator recomputed the reference itself, from the saved inputs, on two calls:
    - **#21, `b-1e-7-3-34455` call 1.** The re-run equals the census bit for bit.
      $S_{10}$ = `0x1.a296675645038p-89`, identical to the agent's. $|S_9-S_{10}|$ = 0.0031
      abserr$_b$, and $|S_{10}-S_{11}|$ = 0.0018 abserr$_b$. $|{\rm value}_b - S_{10}|$/abserr$_b$ =
      **19.65**: the call does not hold. Under 06a the same ratio was 1.13 of 06a's abserr.
    - **#9, `b-1e-7-2-39035` call 2.** The re-run equals the census. $S_{10}$ =
      `0x1.6186143375987p-87`, identical to the agent's. The ratio is **0.154**: the call holds.
  - The agent's table: every re-run is equal and every call agrees ($|S_9-S_{10}| \le$
    abserr$_b$/100). Seven hold; 17 fail, with a median ratio of 1.79 and a maximum of 19.65.
- **W3 — holds.**
  - From `census/out/w3_regions.csv`'s hex floors, re-evaluated: 44 regions, 0 violations of
    `L + R >= 0.5 * own`. The ratio runs from 0.5045 to 0.8908. The recomputed own floor equals the
    stored `phase_err` in all 44.
  - From the checkpoint: phase-limited regions counted in $b$'s 1e-7 calls are 44, and `w3.n_regions`
    is 44, with 0 premise failures and 0 errors.
- **W4 — fails.** At 1e-7, 13,846 calls moved (1,213 group, 12,633 analytic) and none reaches depth
  20. At 1e-8, 19,442 moved (1,720 group, 17,722 analytic), and 3 reach depth 20. No unit is over
  budget.

  | Row (1e-8) | Call | Unit wall |
  |---|---|---|
  | `2-1704` | analytic Y1, call 16 | 6.1 s |
  | `2-13964` | analytic J1, call 5 | 158.9 s |
  | `3-34987` | analytic J2, call 9 | 69.4 s |
- **The descriptive items are present** in the census log, "Descriptive items".

**Row depth** (maximum `max_depth` over the row's group calls; −1 means no group call):

| depth | 06a 1e-7 | b 1e-7 | 06a 1e-8 | b 1e-8 |
|---|---|---|---|---|
| −1 | 2489 | 2489 | 2489 | 2489 |
| 0 | 291 | 291 | 284 | 284 |
| 1 | 530 | 530 | 516 | 516 |
| 2 | 1786 | 1786 | 1790 | 1790 |
| 3 | 654 | 654 | 567 | 567 |
| 4 | 99 | 99 | 69 | 69 |
| 5 | 96 | 96 | 91 | 91 |
| 6 | 511 | 510 | 396 | 396 |
| 7 | 2010 | 2006 | 529 | 529 |
| 8 | 732 | 707 | 1565 | 1564 |
| 9 | 29 | 29 | 923 | 890 |
| 10 | 3 | 3 | 8 | 8 |
| 11 | 31 | 36 | 24 | 31 |
| 12 | 19 | 38 | 28 | 46 |
| 13 | – | 3 | 1 | 7 |
| 14 | – | 3 | – | 3 |

Counting the analytic blocks too, which 06a did not record:
- **1e-7:** $b$'s largest is 18, on one row.
- **1e-8:** the distribution's tail is 15:11, 17:1, 18:8, 19:3, 20:3.

## Check 10 — The records: **FAIL (stop)**

- **Rehearsal `log.md`** follows README §5.1:
  - the subject and result;
  - What shipped;
  - Predictions and verdicts;
  - Deviations, 10, each classified: 1–5, 9 and 10 IMPLEMENTATION CHOICE; 6 and 8 UNINTENDED
    DRIFT; 7 STRUCTURALLY REQUIRED;
  - Verification performed;
  - Observations not acted on;
  - State handed on.

  Its `verification.md` (headed "6b.x Rehearsals R15 and R16…") and `issues.md` exist.
- **Census `log.md`** — two departures from README §5.1:
  1. **It has no "What shipped" section.** Its headings are:
     - the subject and result;
     - Before the launch;
     - Rulings recorded before the launch;
     - After the run;
     - Predictions and verdicts;
     - Descriptive items;
     - Deviations;
     - Verification performed;
     - Observations not acted on;
     - State handed on.

     06a's census log had "What shipped". The content is mostly present under "Subject/Result" and
     "After the run", but the template section is absent.
  2. **Deviation 19 carries no classification.** Its text: *"No over-budget pass: no unit was over
     budget. Recorded for completeness; it is not a departure from the prompt."* The other 23 are
     classified: 6 UNINTENDED DRIFT (11, 14, 15, 20, 21, 24), and the rest STRUCTURALLY REQUIRED or
     IMPLEMENTATION CHOICE.

  Its `verification.md` (headed "6b.5 The census…") and `issues.md` exist.
- **Run directory sizes:**

  | Directory | Size |
  |---|---|
  | census run | 1.8 G |
  | R15 run | 28 K |
  | R16 run | 28 K |
  | `var/levin-phase-floor-06b/census` | 62 M |
  | `var/levin-phase-floor-06b/rehearsal` | 219 M, of which `extract/` is about 180 M |

  The census agent's report records the sizes; whether its log does was not checked.

Per orchestrator README rule 6, this is reported and not repaired. Both points are in the records'
form only. Neither touches a measurement, a verdict or a store. Whether to accept them as they
stand, or to have the log amended, is the user's call (compare 05's stop on check 10).

## Summary for the report

- **R15/R16 fingerprints:** both `1b332ffa…`.
- **The value comparison:**
  - outside QSI, 360,462 value rows are identical;
  - inside QSI, 82 of 864 rows are phase-limited in R15 and 0 in R16;
  - 782 rows are bit-identical;
  - `total` moved on 26 rows, by at most 0.360 × R15 abserr and 0.597 × R16 abserr;
  - `analytic_rad` moved on 70 rows (agent: by ≤ 0.748 × R15 analytic abserr; not re-derived).
- **W0** holds, on all three clauses.
- **W1** holds. **W2** fails, 17 of 24, with a maximum of 19.65 × abserr$_b$. **W3** holds, 44
  regions with a minimum ratio of 0.5045. **W4** fails, 3 analytic calls at 1e-8 reaching depth 20.
- **Over-budget units:** none.
- **Wall time:** 16:12:12Z to the last unit at 17:02:45Z, about 50.5 min to the last unit and 51
  min to exit, against 06a's 45.5 min.
