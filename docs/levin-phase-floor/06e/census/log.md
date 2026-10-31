# levin-phase-floor 06e §5 — the census (Mac Studio)

**Subject:** prompt 06e §5 on the Mac Studio (`mps026298`), Ray-free, with no lock. 06c's census is
re-run under 06e's code (exact Chebyshev end nodes; the Clenshaw–Curtis node term from the measured
displacement), against 06c's run as the "before". This run is also 07's "before".
- **Trees:**
  - $g$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06e-6d81fad`, at Part B's
    dispatch SHA `6d81fad`. Its code is 06e's `f4d8e17`.
  - $p$ is `…-lpf06e-50277f3`, at the code commit's parent `50277f3`. Its code is 06c's census's,
    and it is used for Gate Z0 only.
  - 06c's census tree `e5764a6` is read only through git history.
- **Runs:**
  - The census, `levin-phase-floor-06e-census-20261030T081318`: launched 2026-10-30 08:13:18Z
    (sh 3892, driver 3896), `done`, exit 0, at 10:00:05Z. About **1 h 47 min**, against 06c's
    2 h 30 min.
  - Z3's references for samples 3 and 15, `levin-phase-floor-06e-references-20261030T192730`:
    19:27:30Z to 05:02:41Z, `done`, exit 0.
  - The eight short references ran in-session (10:02–19:25Z, all exit 0).
- **Commit:** none here. The records are committed by the orchestrator, from the laptop, as
  `docs/levin-phase-floor/06e/census/`.

**Result:**
- **Z0 holds** on all three clauses.
- **Z2 fails at 1e-8 only.** One call outside the 720 rows reaches depth 20:
  `g-1e-8-0-33995` call 5, G−q+r. It is unconverged and floor-limited, with 8,706 regions. At 1e-7,
  0 calls reach depth 20; 06c had 109 and 153.
- **Z3 fails.**
  - 13 samples were referenced. 6 count: 2, 3, 8, 9, 14 and 15. 7 are dropped on scatter: 1, 10,
    13, 22, 23, 25 and 27.
  - One counted sample violates: **sample 3** (`g-1e-7-3-34370` call 4, G−q−r) is at **1.245**
    × `abserr`$_g$. 06c was at 1.197 on the same call.
- **Z4 holds** at both rtols: 8,560 rows each, 0 violations.
- **Z5 fails at 1e-8 only.** Of 173 calls 06c reported unconverged, 1 is still unconverged: the same
  `g-1e-8-0-33995` call 5. At 1e-7 all 85 converge.
- **The store:** byte-identical to its certification throughout, fingerprint `0e32f24a…`.
  06c's checkpoint is `b61cb034…` throughout. No worker initialised Ray.

## What shipped

No code. Records only, in `var/levin-phase-floor-06e/census/` (this directory) and two run
directories:
- **`scripts/`:** 06c's census harness, changed as deviations 1–10 say. 06c's copy is kept
  unchanged in `state/scripts_06c_reference/`. The diffs are
  `out/scripts_diff_against_06c_at_prep.txt` and `…_final.txt`. New: `z0.py`, `z0_clause1.py`,
  `dev_line_tiling.py`, `checks_after_06e.py` and `analyse06e.py`.
- **`references/`:** `orch_ref.py`, byte-identical to 06c's orchestrator's (sha256 `8a0d291d…`);
  `slice_inputs.py`, `replay_g.py`, `run_short.sh`, `launch_short.py`, `run_refs.py`,
  `launch_refs.py` and `score_z3.py`; `inputs/` (the 13 captures and `inputs.json`); `out/` (the 20
  orch_ref outputs, `short_progress.txt`, `z3_scores.json`).
- **`state/`:**
  - `predictions.md` (fixed before the launch) and `drafts/predictions_partB_draft.md` (written
    before any computation);
  - `Z0/` (Gate Z0's five workers) and `smoke/`;
  - `rows.json`;
  - `launch_plan.txt`;
  - `drafts/log_working_draft.md` (this log as drafted along the way).
- **`out/`:**
  - every command and its output (`out/cmd/00`–`62`, `out/commands.log`);
  - `census_rows_06e.csv.gz` (one line per unit, both runs side by side);
  - `deep_calls_06e.json` (every call past depth 12);
  - the references' script hashes.
- **Run `levin-phase-floor-06e-census-20261030T081318`:** 1.6 GB, 149 files. It holds the
  checkpoint (sha256 `a0dc4d7f…`, 18,560 records), 16 region pickles, the events and the worker
  logs. **Kept in full on the Mac Studio for 07's §6.**
- **Run `levin-phase-floor-06e-references-20261030T192730`:** 12 KB.

**Where each per-call field lives.** Checkpoint `data.calls` (group) and `data.other_calls`
(analytic blocks); per call:

| Need | Field(s) |
|---|---|
| Z2 | `max_depth`, `n_at_depth_max`, `depth_max` |
| Z3 | `index`, `notify_label`, `x_span`, `value`, `abserr` |
| Z5 | `converged` |
| Cost | `num_regions`, `num_subregion_solves`, `solves_by_purpose`, `budget` (the loop's counters) |
| Budget attribution | `E_share` and `E_largest_share` (06c's attribution) |
| Clenshaw–Curtis floor terms | `cc_floor`: `n_cc_leaves`, `n_reproduced`, `sums_all_cc_leaves`, `sums_floor_attributed_cc_leaves` |

- **Per unit** (checkpoint):
  - Z4: `result.total`, `result.total_abserr`, `result.total_converged`;
  - `status`, `wall_s`;
  - `tree_head`, `tree_status_short`, `scripts_sha256`;
  - `modules` (module provenance by root, foreign modules listed);
  - `deep_lines` (06c's: depth_max calls) and `deep_lines_06e` (every call past depth 12, per level
    `[type, residual, floor, truncation, e, (e_L+e_R)/e_P, r]` and `tiling`
    `[tile N, exact N, tile 2N−1, exact 2N−1]`).
- **Per leaf** (the worker pickle `regions_<worker>.pkl`, at `pickle_offset` / `pickle_length`,
  SHA-checked):
  - `regions` and `other_regions`: start, end, depth, type (0 Levin, 1 Clenshaw–Curtis), abserr
    (residual), phase_err (floor), abserr_truncation, total_err, relerr, phase_limited, delta1,
    delta2, convergence_ratio, trusted, declared_resolution;
  - `cc_floor_terms`: per Clenshaw–Curtis leaf `t_value_rounding`, `t_argument`, `node_term`, `E`,
    `R_g`, `R_2`, `R_c`, `floor_returned`, `declared`, `phase_err_resolve`, `abserr_resolve`,
    `reproduces`;
  - `deep_lines` and `deep_lines_06e`, in full.
- **Registry stamp:** the registry stamps every checkpoint record with `run_census.py`'s SHA-256
  (`a6d9bac4…`).

## Before the launch (every command and its output in `out/cmd/`)

1. **Preparation** (no computation; `00`–`02`): the harness copied and diffed; `rows.json` copied
   (`aa523d28…`); a dry import of `launch.py`.
2. **State** (`10`–`12`, `29`–`31`): the trees are clean at `28ebaf0`, `50277f3` and `6d81fad`;
   33 runs, all finished; the lock is free; no Ray process under `/tmp/ray-sgk`.
3. **The store** (`13`–`16`): the copy is EQUAL to `copy_files_certified.txt`, with no journal,
   `-wal` or `-shm` file. Fingerprint `0e32f24a…`, no problems, at $g$ and at $p$.
4. **06c's checkpoint** (`17`): `b61cb034…`. **job_name** (`18`): stored, never read
   (datastorekit 0.5.0).
5. **Gate Z0 holds** (`19`–`21`):
   - **Clause 1.** `git diff --stat e5764a6 50277f3 -- . ':!docs' ':!prompts'` shows `CLAUDE.md`
     only. The harness imported 102 modules from $p$ (`state/Z0/p_tree_modules.json`), so no
     file needed the AST test.
   - **Clause 2.** At $p$, three units are identical to 06c's `g-` units in every value 06c
     recorded, on 06c's keys:

     | Unit | Total |
     |---|---|
     | `p-1e-7-2-38302` (`low-G+r-1`) | `0x1.5e6b398fe7c77p-101` |
     | `p-1e-7-2-2095` (`ctrl-low-1`) | `-0x1.e3272e29c8b8bp-102` |
     | `p-1e-8-0-1971` | `0x1.6ad248e2d3adap-66` |

     The comparison covers group and analytic calls, the budget counters, the solves by purpose,
     every region array with the trust numbers, the slim and full results, `GkPolicy`, the scan,
     W3, and the deep lines. These units have none in either run. Dispatcher counts are equal too
     (descriptive).
   - **Clause 3.** `g-1e-7-0-14164` (06c's X0 unit) is identical in two fresh workers, on every
     key the worker records, 06e's new ones included. Its 6 fallback leaves' floor terms
     reproduce.
   - No worker initialised Ray.
6. **Development checks:**
   - `21b`: `_line_tiling` on a synthetic 21-level line. Every level tiles at $g$; at $p$ some do
     not.
   - `22`, `22b`: the driver smoke test, 2 rows × 2 rtols, 4 units done; 64 of 64 fallback leaves
     reproduced.
7. **The copy after Z0** (`23`, `24`): EQUAL.
8. **Predictions** (`26`): `state/predictions.md`, sha256 `f27771cc…`, mtime
   2026-10-30T08:12:47Z. Part A is prompt lines 345–375, byte for byte. Part B is the preparation
   draft, with the over-budget sentence's basis added.
9. **Scripts** (`27`, `34`): every run script equals those Z0 and the smoke test ran. 06c's
   checkpoint `b61cb034…` (`28`). The copy before the launch (`32`, `33`) is EQUAL.
10. **Launch** (`36`) and one check (`37`): `running`, six 1e-7 workers started, stderr empty.

## After the run

- **Status** (`38`): `done`, exit 0, 18,560 of 18,560 units, `stderr.log` empty. All 16 workers'
  `bye` messages report `ray_initialized` false and no foreign modules.
- **Every record** (`39`):
  - is at `6d81fad` with empty status, carries `run_census.py` `a6d9bac4…`, and has scripts equal
    to `scripts/`;
  - **none is over budget, none in error, and none has `wall_s` > 300 s.** So no over-budget pass
    was run;
  - 278,863 of 278,863 Clenshaw–Curtis leaves' floor terms reproduce bit for bit;
  - 200 deep lines (calls past depth 12, both rtols), every one reproducing its leaf, no error.
- **The store** (`40`–`44`, `59`, `60`): the copy is EQUAL to its certification after the run,
  after the fingerprint, and at the end, with no journal. Fingerprint `0e32f24a…`.
- **06c's checkpoint** (`45`, `61`): `b61cb034…` after the run and at the end.
- **Z3's inputs** (`46`): the 13 captures, sliced and SHA-checked against `captured.json`, are
  identical to the orchestrator's copies where it has them.
- **The replay** (`48`): on all 13 samples, $g$'s driver replayed on the captured input gives
  $g$'s census value, abserr, regions, depth and converged bit for bit, with label and span equal.
- **References.**
  - Short batch (`49`–`51`): 16 passes, all exit 0. Walls 305–6,095 s per pass, about 2–3.5×
    depth20-tiling's estimates.
  - Registered run (`52`–`54`): 4 passes, all exit 0. s03 3,984 s + 11,131 s; s15 5,081 s +
    14,301 s.
- **Scores:**
  - `55`, `analyse06e.py`: Z2, Z4, Z5 and the descriptive items.
  - `57`, `score_z3.py`: Z3. `58` is the Z2/Z5 call in detail.
  - `56`: the Y1 lookup.
- **Final state** (`61`, `62`): trees clean; 35 runs, all finished; `state/predictions.md`
  unchanged (`f27771cc…`, mtime 08:12:47Z).

## Predictions and verdicts

`state/predictions.md` (sha256 `f27771cc…`, mtime 2026-10-30T08:12:47Z, before the launch at
08:13:18Z) holds §5's Z2–Z5 unchanged. They are copied here unchanged from the prompt:

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **Z2 — the cap.**
  - **The rule:** outside the stationary-point rows, no call under $g$ reaches `depth_max`, at
    either rtol. An over-budget unit counts as false.
  - **Holds** if true at both rtols. 06c's census had 109 and 153 such calls among those that
    moved.
- **Z3 — honest where referenced.**
  - **The sample** is `depth20-tiling`'s 30 calls, matched by row, rtol and call index. Their
    references are computed by `orchestrator_check/orch_ref.py`'s method at three resolutions, as
    copied into `var/levin-phase-floor-06e/census/references/`. The samples:
    - 1, 9 and 25, already run, re-scored against $g$;
    - 2, 8, 10, 13, 14, 22, 23 and 27, each estimated at under 45 minutes;
    - 3 and 15, estimated at 2–2.6 h, in one registered run with slug `references`.

    The 18 estimated at 20 h or more are not run.
  - **A reference counts** only if its resolutions agree to within 1/100 of $\min({\rm abserr}_{06c},
    {\rm abserr}_g)$. Those that do not are reported with their scatter and dropped.
  - **The rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every counted reference. The census
    side, $|v_{06c} - {\rm ref}|/{\rm abserr}_{06c}$, is reported beside it.
  - **Holds** if no counted reference violates it. Fewer than five counted is **undecided**, not
    holds.
- **Z4 — the rows agree.**
  - **The rule:** outside the stationary-point rows, every row with finite `total_abserr` under both
    has $|{\rm total}_g - {\rm total}_{06c}| \le {\rm total\_abserr}_{06c} + {\rm
    total\_abserr}_g$.
  - **Holds** if true at both rtols.
- **Z5 — budget misses converge.**
  - **The rule:** every Levin call outside the stationary-point rows that 06c reported unconverged
    is `converged` under $g$.
  - **Holds** if true at both rtols.

| | Number | Verdict |
|---|---|---|
| **Z0** | See "Before the launch", item 5: all three clauses. | holds |
| **Z2** | The 720 rows check out at 720. No unit is over budget. 1e-7: 0 calls outside the 720 reach depth 20 (11 inside, all G+q+r call 1, reported). 1e-8: **1**, `g-1e-8-0-33995` call 5 (G−q+r, k=3.14e5, q=1e5, r=3.14e5, z_response 4.809e6), unconverged (17 inside, all G+q+r call 1). 06c: 109 and 153. (`55`, `58`) | **fails** (1e-8) |
| **Z3** | 13 referenced. **Counted 6:** 2 (0.9985), 3 (**1.245**), 8 (0.9995), 9 (0.9979), 14 (0.9985), 15 (0.7811); these are $\|v_g-{\rm ref}\|/{\rm abserr}_g$. **Dropped 7** on scatter (range of the three resolutions / min abserr): 1 (0.961), 10 (0.0234), 13 (17.6), 22 (0.989), 23 (0.0673), 25 (65.7), 27 (0.766). Every capture stands for its $g$ call. Table below. (`57`, `references/out/z3_scores.json`) | **fails** (sample 3) |
| **Z4** | 1e-7: 8,560 rows scored (0 non-finite), 0 violate. 1e-8: 8,560, 0 violate. Exact (`Fraction`). (`55`) | **holds** |
| **Z5** | 1e-7: 85 Levin calls outside the 720 that 06c reported unconverged; all converged under $g$. 1e-8: 173; **1** not converged, `g-1e-8-0-33995` call 5 (the Z2 call). 0 unmatched. (`55`) | **fails** (1e-8) |

**Z3 in full** (ref = S128 at P = 64; ratios exact; "orch" is the orchestrator's each-against-ref reading):

| # | unit, call | block | range / min(abserr) (orch) | counted | \|v_g−ref\|/e_g | \|v_06c−ref\|/e_06c |
|---|---|---|---|---|---|---|
| 1 | g-1e-7-2-1444 #4 | G+q−r | 0.961 (0.583) | dropped | 0.626 | 0.122 |
| 2 | g-1e-7-2-1627 #3 | G−q+r | 5.0e-6 | yes | 0.9985 | 0.9985 |
| 3 | g-1e-7-3-34370 #4 | G−q−r | 0.00833 | yes | **1.245** | 1.197 |
| 8 | g-1e-7-3-37319 #6 | Y1 | 0.00080 | yes | 0.9995 | 0.9993 |
| 9 | g-1e-7-1-35340 #18 | Y2 | 0.00666 (0.00452) | yes | 0.9979 | 0.991 |
| 10 | g-1e-7-1-35332 #10 | Y3 | 0.0234 (0.0190) | dropped | 0.985 | 0.950 |
| 13 | g-1e-8-3-34220 #4 | G+q−r | 17.6 (10.6) | dropped | 10.4 | 0.068 |
| 14 | g-1e-8-2-1627 #3 | G−q+r | 5.0e-6 | yes | 0.9985 | 0.9985 |
| 15 | g-1e-8-2-1549 #4 | G−q−r | 0.00890 | yes | 0.7811 | 0.7804 |
| 22 | g-1e-8-2-21874 #12 | Y3 | 0.989 | dropped | 0.216 | 0.029 |
| 23 | g-1e-8-2-1402 #12 | Y4 | 0.0673 (0.0351) | dropped | 0.969 | 0.948 |
| 25 | g-1e-8-0-33962 #12 | Y4 | 65.7 | dropped | 9.19 | 4.9e-5 |
| 27 | g-1e-8-3-34419 #21 | J4 | 0.766 | dropped | 2.03 | 1.38 |

On the dropped samples the $g$ ratios are not scored. Samples 13, 25 and 27 exceed 1 against
an unconverged reference; sample 25's 65.7 is 06c's own non-converging reference, now against
$g$'s 4e4× smaller abserr.

## Descriptive items (`out/cmd/55`)

- **Convergence.**

  | | 1e-7, $g$ | 1e-7, 06c | 1e-8, $g$ | 1e-8, 06c |
  |---|---|---|---|---|
  | Group calls unconverged | 9 | 15 | 15 | 49 |
  | Analytic calls unconverged | 0 | 89 | 0 | 151 |
  | Rows with `total_converged` false | 9 | 13 | 15 | 45 |
  | Rows with `total_converged` false, outside the 720 | **0** | 9 | **1** | 33 |

  There are 37,325 group and 107,416 analytic calls per rtol in both runs.
- **Depth.**
  - Calls at depth 20: 11 against 109 at 1e-7 (all 11 are G+q+r call 1 of a stationary-point row);
    18 against 153 at 1e-8 (17 inside the 720, plus `g-1e-8-0-33995` #5).
  - Row depth (max over the row's calls), $g$/06c, at 1e-7: depth 4: 1,190/422; 8: 777/978;
    12: 13/247; 16: 10/82; 20: 11/63. At 1e-8: depth 4: 1,614/383; 12: 30/293; 20: 18/81. The full
    histograms are in `55`.
  - The mass moves from depths 8–20 to 3–7.
- **Cost.**

  | | 1e-7, $g$ | 1e-7, 06c | 1e-8, $g$ | 1e-8, 06c |
  |---|---|---|---|---|
  | Regions | 625,878 | 973,698 (×0.64) | 799,186 | 2,086,346 (×0.38) |
  | Solves | 4.86 M | 7.26 M (×0.67) | 6.21 M | 12.44 M (×0.50) |
  | Unit wall summed | **2.28 h** | 3.21 h (×0.71) | **2.97 h** | 5.77 h (×0.52) |
  | Largest unit | 6.0 s | 426 s | 105 s (`g-1e-8-0-33995`) | 1,185 s |

  - The run took 1 h 47 min, against 2 h 30 min.
  - Cost is not an argument (README §5 rule 4).
- **Where the budget went.**
  - Per call, the largest share of $E$:
    - group calls: 1e-7 resolution 30,908 / floor 6,417; 1e-8 31,059 / 6,266;
    - analytic: 1e-7 61,997 / 45,392 / truncation 27; 1e-8 54,303 / 53,113.
  - Over the Clenshaw–Curtis leaves whose floor carries their error (5,628 calls at 1e-7, 7,326 at
    1e-8), the floor sums to 1.58e-17 and 1.97e-17. Of that, **the declared-phase term is 99.4 %**;
    (n+4)uS is 0.46 %; the node term $\phi_{\rm node}$ 0.06 % (|E| 0.05 %, R_g 0.01 %, R_2 and
    R_c ≈ 0); the argument term 0.03 %.
  - Per call, the largest floor term is (n+4)uS on 2,644 and 3,216 calls, the declared term on
    2,493 and 2,785, and the node term on 491 and 1,323.
- **Anything still deep** (outside the 720, past depth 12; `out/deep_calls_06e.json`):
  - 14 calls at 1e-7 and 24 at 1e-8, all at depths 13–15 and converged, except `g-1e-8-0-33995` #5
    (depth 20).
  - **Every level of every such line tiles** (`every_level_tiles_at_N` true on all).
  - Residual over floor over each line's last five levels falls to 0.004–1.6 at the leaf (in `55`).
  - Inside the 720: 59 and 103 such calls.
- **The Z2/Z5 call** `g-1e-8-0-33995` #5 (G−q+r; no sign change of θ′ in 06a's scan, min|θ′|/max|θ′|
  = 0.0028; `58`):
  - **Under $g$:** 8,706 regions (8,600 Clenshaw–Curtis), 5,642 floor-limited, 3,064 at depth
    20 and unchecked; 30,707 solves; unit wall 105 s.
    - Floor sum 1.33e-33 against atol 6.7e-34 (`E_share` floor 1.33e-33, resolution 5.8e-38).
    - abserr `0x1.badaaed87034dp-110` (1.33e-33).
  - **Under 06c:** depth 13, 688 regions all floor-limited, unconverged, abserr 4.99e-32, wall
    9.2 s.
  - **At 1e-7 under $g$:** converged at depth 11.
  - So the floor fell 37× but still exceeds the request, and the loop now chases to depth 20.
- **The 720 stationary-point rows** (07's; not scored):
  - Totals moved on 642 (1e-7) and 630 (1e-8); by more than `total_abserr`$_{06c}$ on 7 and 8, by
    up to 261 and 2,290 of it.
  - **`*-1-2174` call 1:**
    - at 1e-7: depth 20, 94 regions, unconverged, value moved 1.4e-5 `abserr`;
    - at 1e-8: **depth 7, 18 regions, converged, value moved 2,295 × `abserr`$_{06c}$** (1,678 ×
      `abserr`$_g$). This is bit for bit depth20-tiling's pinned counterfactual value
      (`0x1.0c6c2cd926acfp-86`), which that study judged most likely dishonest, unverified.
  - **Y1's 12 equilateral rows are not in A3 v2.** y1-references' Y01–Y12 come from the R17/R18
    rehearsal stores. Their (k, q, r, z_response) match no `rows.json` row: k = 3.16e5 and 1e6 do
    not occur with q = r = k, and k = 1e5 rows exist only at other redshifts (`56`). So they cannot
    be reported from this census.
- **Over-budget units:** none at either rtol. No unit's `wall_s` exceeds 300 s, so 06c's deviation
  22 does not arise.

## Deviations from the prompt (numbered and classified)

1. **`prov.py`:** trees `p` = `…-lpf06e-50277f3` and `g` = `…-lpf06e-6d81fad`; 06c's run,
   records and checkpoint SHA-256; depth20-tiling's directory; `WIDER` =
   `var/levin-phase-floor-06e/census`. STRUCTURALLY REQUIRED.
2. **`harness.py`:** `job_name="levin-phase-floor-06e"`, which is stored and never read (`18`).
   IMPLEMENTATION CHOICE.
3. **`worker.py`, tree evidence:** 06e's `_chebyshev_node_displacements` and
   `_cc_node_term_parts`, 06d's `_cc_node_error_bound`, and the SHA-256 of
   `levin_quadrature.py`. STRUCTURALLY REQUIRED (the brief: evidence of where modules were loaded
   from).
4. **`worker.py`, `_cc_floor_terms`** (the Clenshaw–Curtis floor's terms):
   - each returned fallback leaf is re-solved after the timed section;
   - pass-throughs at `_cc_node_term_parts`, `_cc_roundoff_floor` and
     `_declared_endpoint_phase_err` are installed for the re-solve only;
   - the floor's first two terms are read from the code's own function on zeroed inputs, so each
     is exact;
   - each leaf is checked to reproduce bit for bit (all 278,863 do).

   IMPLEMENTATION CHOICE: the result does not return the terms, and recomputing after the timed
   section leaves the budget as 06c's.
5. **`worker.py`, `deep_lines_06e`:** 06c's `_deep_line` for every call past depth 12, with
   per-level tiling at N and 2N−1 (`_line_tiling`). 06c's `deep_lines` key is unchanged. Required by
   the prompt's "anything still deep"; IMPLEMENTATION CHOICE in form.
6. **Nothing in the timed section changes and no 06c key changes;** every addition is a new key.
   IMPLEMENTATION CHOICE, so that Z0's clause 2 compares on 06c's keys.
7. **`run_census.py`:** docstring only; the trees move through `prov.py`. STRUCTURALLY REQUIRED.
8. **`launch.py`:** prompt `06e`, purpose, scope, imports. STRUCTURALLY REQUIRED.
9. **New `z0.py`** (from 06c's `x0.py`): P units against 06c's `g-` units on 06c's keys, deep
   lines included; the determinism unit is 06c's X0 unit `g-1e-7-0-14164`; no capture.
   IMPLEMENTATION CHOICE.
10. **New `z0_clause1.py`;** `scripts_equal.py` (state/Z0) and `check_predictions.py` (06e, lines
    345–375) adapted. IMPLEMENTATION CHOICE.
11. **Z3, how `orch_ref.py` was run:** unchanged, with `GIT_OPTIONAL_LOCKS=0`. Its plain
    `git status` would otherwise refresh the worktree's index, which is a write in a tree.
    STRUCTURALLY REQUIRED. Also `PYTHONDONTWRITEBYTECODE=1` and one BLAS/OpenMP thread per process,
    for the six-core rule. IMPLEMENTATION CHOICE.
12. **Z3:** the eight short references ran in one detached sequential script (`run_short.sh`),
    never two at once. The brief allows this if recorded; the coordinator confirmed it.
    IMPLEMENTATION CHOICE.
13. **Z3, the three resolutions:** S96(P128), S96(P64) and S128(P64); the reference is S128(P64).
    "Agree" means the range ≤ min(abserr)/100, exact. The orchestrator's reading is reported
    beside it; it changes no count. IMPLEMENTATION CHOICE.
14. **Z3, input identity:** a capture stands for $g$'s call only if $g$'s driver replay on it
    reproduces $g$'s census call bit for bit (`replay_g.py`; all 13 do). IMPLEMENTATION CHOICE.
15. **Scripts are not `black`-formatted;** they are records. IMPLEMENTATION CHOICE.
16. **06b's W3 evidence and 06a's scan are kept** in each 1e-7 unit, as in 06c. IMPLEMENTATION
    CHOICE.
17. **`out/cmd/00` was written by hand,** since rec.sh did not exist before the copy. Its `# cwd`
    is the shell's directory at the time, the orchestrator's `review/`, where nothing was written.
    UNINTENDED DRIFT.
18. **`out/cmd/11` piped rec.sh's output through `sed`** in the same shell, so the lock's
    process listing showed that `sed` (its own child). `12`, `30` and `31` use `>>` only and list
    nothing. UNINTENDED DRIFT.
19. **`out/cmd/25` failed:** my writer raised on a non-ASCII bytes literal before writing anything,
    so `state/predictions.md` did not exist and the check failed. It was rewritten and checked as
    `26`. UNINTENDED DRIFT.
20. **`out/cmd/22b` and `56` are inline reads** (the smoke ledger summary, the Y1 lookup), not run
    through rec.sh. Each notes its command or is a pure read. New `dev_line_tiling.py`, because Z0
    had no call past depth 12. IMPLEMENTATION CHOICE.
21. **More than twelve over-budget units would have been a stop before the pass.** This is my
    reading, not the prompt's; its basis is 06c's Part B, which fixed the same threshold for 06a's
    pass. It did not arise (none over budget). IMPLEMENTATION CHOICE.
22. **`out/cmd/47` failed** with `ModuleNotFoundError: harness`: the captured group calls' pickles
    name `harness._rebuild_lru_cache` (06c's census harness). I re-ran it as `48` with the census
    `scripts/` after the tree on PYTHONPATH, as depth20-tiling's own commands did.
    `run_short.sh` and `run_refs.py` got the same change before their launch, and `orch_ref.py` is
    unchanged. UNINTENDED DRIFT.
23. **New `checks_after_06e.py`, `analyse06e.py` and `score_z3.py`.** IMPLEMENTATION CHOICE.
24. **`out/cmd/61`'s `stat` printed errors** (BSD flags on a GNU `stat`, as 06c's deviation 23).
    The mtime is recorded by `62` instead. UNINTENDED DRIFT.

None of the UNINTENDED DRIFT items (17, 18, 19, 22, 24) touched a store, a tree, or any record
outside this directory and the two run directories.

## Verification performed

- **The trees:**
  - $g$ is at `6d81fad`. `status --short --ignored` is empty at Z0, at the launch, on all 18,560
    records, and at the end. No `__pycache__` was written.
  - $p$ is at `50277f3`, likewise.
  - The main checkout is at `28ebaf0`, status empty, throughout.
- **The copy:**
  - It is byte-identical to `copy_files_certified.txt` before Z0, after Z0, at the launch, after
    the run, after the fingerprint, and at the end, with no journal, `-wal` or `-shm` file.
  - It gives `0e32f24a…` at $g$ and $p$ before and at $g$ after.
- **06c's run was only read:** its checkpoint is `b61cb034…` before, after preparation, after the
  run, and at the end.
- **The run:**
  - 18,560 records, every unit `done`, every record at $g$, clean.
  - Every worker reported `ray_initialized` false, with no foreign modules.
  - Every Clenshaw–Curtis leaf's re-solve and every deep line reproduces.
- **Z3:**
  - Every capture's SHA-256 matches; every replay at $g$ equals the census call.
  - Every new reference output names tree `6d81fad` and `orch_ref.py` `8a0d291d…`.
  - 06c's values agree between its checkpoint and depth20-tiling's capture records.
- **Z4** was evaluated exactly on the recorded doubles.

## Observations not acted on

All are in `issues.md`:
- **D1:** Z2 and Z5 fail on one call, `g-1e-8-0-33995` #5, whose floor still exceeds its request.
- **D2:** Z3 fails on sample 3 by 1.245×; 06c was dishonest there too (1.197×).
- **D3:** on 6 counted samples and on several dropped ones, $|v-{\rm ref}|/{\rm abserr}$ sits at
  0.95–0.9995 in both codes.
- **D4:** Clenshaw–Curtis floors are 99.4 % the declared-phase term.
- **D5:** `*-1-2174` call 1 at 1e-8 converges onto the pinned value, 2.3e3 × `abserr`$_{06c}$
  from 06c's.
- **D6:** Y1's 12 rows are not in A3 v2.

## State handed on

- **Nothing is running.** Both runs are `done`. No lock was taken. Nothing was written in either
  worktree, or in the main checkout outside this directory and the two run directories.
- **The scratch copy is unchanged** (`0e32f24a…`) and is still kept for the user to delete.
- **Run directories kept in full:**
  - `var/runs/levin-phase-floor-06e-census-20261030T081318/` (1.6 GB) for 07's §6;
  - `…-06e-references-20261030T192730/` (12 KB).
- **The header issues:** Z0 and Z4 hold; Z2, Z3 and Z5 fail. Whether
  `[06c-chebyshev-end-nodes-miss-the-span]` and `[06d-cc-node-term-is-pessimistic]` close is for
  the orchestrator and the user (prompt "Closes, when §4 and §5 hold").
- **For the orchestrator:** copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/`,
  `state/` and `references/` to `docs/levin-phase-floor/06e/census/`. Things to leave out and list
  as kept on the Mac Studio:
  - the pickles under `state/Z0/` and `state/smoke/` (`regions_*.pkl`);
  - `references/inputs/*.pkl` (copies of depth20-tiling's captures, about 7.8 MB).
