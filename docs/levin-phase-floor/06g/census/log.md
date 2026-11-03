# levin-phase-floor 06g §5 — the census (Mac Studio)

**Subject:** prompt 06g §5's census on the Mac Studio (`mps026298`). 06h's census was re-run under 06g's code,
Ray-free and with no lock, against 06h's run as the "before". This record covers:
- Gate V;
- the census run;
- V2 and V5–V8;
- the descriptive items, plus the slow-leaf breakdown the user asked for (addendum).

§4 (R23, R24) and §5's references belong to the other Part B agents. **This census is 07's "before".**
- **Trees:**
  - $g$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-1c5336e` at Part B's dispatch SHA
    `1c5336e`. Its code is 06g's code commit `3ae6f3a`.
  - $p$ is `…-lpf06g-4f4f9ed`, the code commit's parent, whose code is 06h's census tree's. It is used for Gate V
    only.
  - 06h's census tree `c1ec3b2` is read only, through git history.
- **Run:** `levin-phase-floor-06g-census-20261101T114315`.
  - Launched at 11:43:15Z (sh 51178, driver 51183). It ended `done`, exit 0, with its last heartbeat at
    17:22:36Z; the driver exited at about 17:23:11Z.
  - That is **5 h 40 min of wall time against 06h's 2 h 33 min** (a finding; see Cost).
- **Commit:** none here. The records are committed by the orchestrator from the laptop, as
  `docs/levin-phase-floor/06g/census/`.

**Result:**
- **Gate V holds** on all three parts.
- **V8 fails at both rtols:** 801 calls lost in 470 rows at 1e-7, and 830 calls in 476 rows at 1e-8. Every lost
  call is a `G±q±r`-type group call. The note's §5 stop on V8 was reached and put to the user, who ruled to score
  the rest (below).
- **V2 fails at both rtols** on `total`: 5 rows at each rtol, the worst ratio 2.16. On the oracle it holds, with
  0 violations.
- **V5 fails:** 966 and 996 calls outside the 720 reach depth 20, against 0 and 1 under 06h.
- **V6 fails:** Z2's call ends node-limited, but at depth 20.
- **V7 holds:** H4's call converges.
- **The store** stayed byte-identical to its certification throughout (fingerprint `0e32f24a…`). 06h's checkpoint
  is `3104c18e…` throughout. No worker started Ray.

## The note's §5 stop on V8, and the user's ruling

The orchestrator's note §5 says: "Also stop if the noise clause or the slow threshold costs convergence on more
than a handful of rows (V8), before scoring the rest." After the run I scored V8 alone (`out/v8.md`) and stopped.
- The stop was reached: V8's lost calls are (b) on a non-noise leaf in 470 rows (1e-7) and 474 rows (1e-8).
- The orchestrator recorded it in `var/levin-phase-floor-06g/review/stop_v8.md` and put it to the user.
- The user ruled to score the rest of Part B for the record: *"(b) with the V8 sample, please"*, relayed by the
  laptop orchestrator.
- The stop stays in the record, and V8's verdict, fails at both rtols, does not change.
- `out/v8.md`, `out/v8_calls.json` (`1e9a62b0…`, from which the orchestrator drew a V8 reference sample) and
  `state/predictions.md` were not touched after the stop.

## What shipped

No code. Records only, in `var/levin-phase-floor-06g/census/` (this directory) and one run directory.
- **`scripts/`:** 06h's census harness, changed as deviations 1–15 say. 06h's copy is kept unchanged in
  `state/scripts_06h_reference/`, with its SHA-256s in `state/scripts_06h_reference.sha256`.
  - The diffs are `out/scripts_diff_against_06h_at_prep.txt` and `…_final.txt`.
  - The new files are diffed against their 06h sources in
    `out/scripts_diff_new_files_against_06h_sources_*.txt`.
  - New for the gate and the checks: `gate_v.py`, `gate_v_clause1.py`, `checks_after_06g.py`, `static_names.py`
    and `smoke_summary.py`.
  - New for scoring, after the run: `v8_06g.py`, `v8_breakdown.py`, `analyse06g.py`, `v2_detail06g.py`,
    `values_beyond06g.py`, `v8_slow_leaves.py` and `v8_slow_leaves_coverage.py`.
- **`state/`:**
  - `predictions.md` (sha256 `930ce6fd…`, mtime 11:42:36.6Z, fixed before the launch), with its drafts in
    `drafts/`;
  - `V/` (Gate V's five workers), `smoke/`, `rows.json`, `launch_plan.txt`;
  - `drafts/deviations_stage1.md`, the deviations as they were recorded stage by stage.
- **`out/`:**
  - every command and its output (`out/cmd/00`–`57`, `out/commands.log`);
  - `v8.md` and `v8_calls.json` (V8);
  - `analyse06g` output in `out/cmd/49`, plus `v2_rows.csv.gz` (every row scored for V2, both readings),
    `v5_depth_max_calls.csv` and `descriptive_06g.json`;
  - `v8_slow_leaves.md` and `v8_slow_leaves.json` (the addendum).
- **Run `levin-phase-floor-06g-census-20261101T114315`:** 4.5 GB, 149 files.
  - It holds the checkpoint (sha256 `03adcbba…`, 2,083,192,147 bytes, 18,560 records), 16 region pickles, the
    events and the worker logs.
  - **It is kept in full on the Mac Studio for 07's census and the references agent**
    (`KEPT_ON_MAC_STUDIO.txt`).

### Where each per-call field lives

The checkpoint's `data.calls` holds the group calls and `data.other_calls` the analytic-block calls. Every key 06h
recorded is kept as it was (06h's log, "Where each per-call field lives"). 06g's new per-call keys:

| Need | Field(s) |
|---|---|
| Matching (the references agent) | row `data.row`, `data.shard`, `data.serial`; rtol `data.rtol`; `index`, `notify_label`, `x_span` (hex) |
| Value, abserr | `value`, `abserr` (each `{repr, hex}`) |
| ΣE, Σ\|E\|, \|ΣE\| | `node_error_signed_sum`, `node_error_abs_sum`, `abserr_node_signed` (`{repr, hex}`; None at $p$) |
| Clenshaw–Curtis flag | `has_cc_leaf` (also `num_simple_regions`) |
| Node-limited stop | `node_limited` (bool; None at $p$); also `budget.node_limited` |
| The call's error reassembled | `abserr_eprime_sum` (Σe′ ascending, as the driver sums it); `abserr_reassembles` (Σe′ + \|ΣE\| == abserr bit for bit) |
| Values returned | `values_06g`: `n_quarters`, `n_own`, `sum_abs_q_minus_own`, `value_reassembles` |
| The exit | 06h's `premise_checks`/`premise_failed`, and `premise_06g`: `n_checks`, `n_fired`, `n_not_fired`, `n_halving_failed`, `n_split_could_meet_budget`, `n_halving_held_but_split_could_meet_budget` |
| Checked leaves | `checked_06g`: `n_made`, `n_checked`, `n_checked_returned`, `n_checked_bisected`, `n_checked_then_floor_limited`, `classes_all` and `classes_returned` (test 3's eight classes), `n_b_untrusted`, `n_b_untrusted_returned`, `n_noise_clause_trusted`, `n_trust_g_differs_from_trusted_final` |
| The budget record | `budget`: 06h's 13 keys plus `node_error_signed_sum`, `node_error_abs_sum`, `node_limited` |
| Errors in 06g's additions | `error_06g` (absent when none; absent on every call of the run) |

- **Per unit** (checkpoint), as in 06h, plus `checked_06g_s` (post-timed seconds).
- **Per leaf** (pickle `regions_<worker>.pkl` at `pickle_offset`/`pickle_length`, SHA-checked):
  - the region arrays add `value`, `value_own`, `value_kind` (0 own, 1 quarters, −1 none), `phase_err_declared`,
    `phase_err_roundoff`, `node_error_signed`, `phase_err_less_node` and `budget_err`;
  - `checked_leaves_06g` (per call, every checked leaf, returned or bisected) holds start, end, depth, delta1,
    delta2, convergence_ratio, phase_err, phase_err_roundoff, abserr, declared_resolution, value, value_own,
    node_error_signed, budget_err, total_err, type, trusted_final, value_kind, returned, phase_limited, trust_g and
    trust_parent;
  - `premise_06g` (per call, each premise check) holds the result, the budget clause as passed, the halving test
    and the three floors in hex.

## Before the launch (every command and its output in `out/cmd/`)

1. **Preparation**, stage 1, no computation (`00`–`05`):
   - the harness was copied and diffed, and `rows.json` copied (`aa523d28…`, cmp identical);
   - a syntax check (`04`) and a static undefined-name check (`04d`);
   - the trees are clean.
2. **State** (`10`–`12`, `02`): the trees are clean at `28ebaf0`, `4f4f9ed` and `1c5336e`, and `git worktree list`
   shows these three only. The registry holds 42 runs, all finished. The lock is free, and there is no Ray process
   under `/tmp/ray-sgk`. `02` is the launcher's dry import.
3. **The store** (`13`–`16`): the copy is EQUAL to `copy_files_certified.txt`, with no journal, `-wal` or `-shm`
   file. Its fingerprint is `0e32f24a…` with no problems, at $g$ and at $p$.
4. **06h's checkpoint** (`17`): `3104c18e…`. `18`: `job_name` is stored and never read.
5. **Gate V** (`19`–`21`; it holds, all three parts):
   1. **The parent reproduces 06h.** At $p$, `p-1e-7-2-38302` (`low-G+r-1`), `p-1e-7-2-2095` (`ctrl-low-1`) and
      `p-1e-8-0-1971` are IDENTICAL to 06h's units on every key 06h recorded.
      - That covers the group and analytic calls (value, abserr, regions, depth, converged and every other key),
        the slim result (total, total_abserr, analytic_rad), `analytic_06h.h` (the oracle's value, abserr and
        converged), every region array, `cc_floor_terms`, `declared_terms_06h`, the deep lines, W3, the scan and
        GkPolicy.
      - Calls equal: 23/23, 23/23 and 21/21. Totals: `0x1.5e6b398fe7c40p-101`, `-0x1.e3272e29c8b71p-102` and
        `0x1.6ad248e2d3ab7p-66`.
      - The module list is the P workers': 102 modules from $p$ (`state/V/p_tree_modules.json`).
   2. **The integrand is 06h's.** `git diff --stat c1ec3b2 3ae6f3a -- . ':!docs' ':!prompts'` touches
      `AdaptiveLevin/levin_quadrature.py`, five AdaptiveLevin test files, `tools/filon_reference.py`, its test and
      `CLAUDE.md`, all allowed. Of these the harness imports only `AdaptiveLevin.levin_quadrature`. Also:
      `c1ec3b2..4f4f9ed` changes only `CLAUDE.md`, and `3ae6f3a..1c5336e` changes nothing.
   3. **The new code is deterministic.** `g-1e-7-0-14164` is IDENTICAL in two fresh workers on every key, 06g's
      new ones included, timings excepted.
      - It qualifies: 2 calls have Clenshaw–Curtis leaves with ΣE ≠ 0, and 107 returned leaves return their
        quarters.
      - In 23 of 23 calls the abserr and the value reassemble bit for bit.
      - At $g$, 8 of 8 fallback floor-term leaves reproduce.
   - No worker started Ray.
6. **The smoke test** (`22`, `22b`): 06h's two rows at both rtols, 4 units `done` at $g$.
   - No 06g-record error; abserr and value reassembled in every call; every budget record parsed with 06g's keys.
   - Every fallback floor-term and declared-term leaf reproduced.
7. **Predictions** (`25`): `state/predictions.md`, sha256 `930ce6fd…`, mtime 11:42:36.6Z.
   - Part A is prompt lines 401–422 byte for byte.
   - Part B fixes how they are read: the 720, call matching, V8's five cause classes (extended before the launch
     at the coordinator's request so as to separate the depth cap from (b)), and V2 and V5–V7.
   - `26`: every run script equals those Gate V and the smoke test ran.
8. **Pre-launch** (`28`–`34`): all as before; the copy is EQUAL.
9. **Launch** (`35`) and one check (`36`): `running`, six 1e-7 workers started, stderr empty.

## After the run

- **Status** (`38`): `done`, exit 0, 18,560 of 18,560 units, `stderr.log` empty. All 16 workers' `bye` messages
  report `ray_initialized` false and no foreign modules. No census worker process remained.
- **Every record** (`39`, `checks_after_06g.py`):
  - all 18,560 are at `1c5336e` with empty status, carry `run_census.py` `5f21f072…` and have scripts equal to
    `scripts/`;
  - **none is over budget, none is in error, and none has `wall_s` over 300 s**, so no over-budget pass was run;
  - 289,482 calls, none with an error in 06g's records; every abserr and value reassembles and every budget record
    parses;
  - 1,381,881 of 1,381,881 fallback floor-term leaves and 877,835 of 877,835 declared-term leaves reproduce;
  - 15,770 deep lines past depth 12, all reproducing;
  - 18,560 oracle counterfactuals were computed (06h's record, kept; see deviation 15);
  - 06g's trust test agrees with the final flag on all 5,759,997 checked leaves.
- **The store** (`40`–`44`, `53`, `54`): EQUAL after the run, after the fingerprint (`0e32f24a…`, no problems) and
  at the end.
- **06h's checkpoint** (`45`, `55`): `3104c18e…`.
- **Scores:**
  - `46` (V8, `v8_06g.py`) and `47` (its breakdown);
  - `49` (`analyse06g.py`: V2, V5, V6, V7 and the descriptive items);
  - `50` (V2's rows in detail) and `51` (calls whose value moved beyond both errors);
  - `56`, `56b` and `57` (the slow-leaf addendum).
- **Final state** (`48`, `52`): the trees are clean; 43 runs, all finished; no census worker process.
  `state/predictions.md` (`930ce6fd…`), `out/v8.md` (`784ab528…`) and `out/v8_calls.json` (`1e9a62b0…`) are
  unchanged since the stop.

## Predictions and verdicts

`state/predictions.md` (sha256 `930ce6fd…`, mtime 2026-11-01T11:42:36.6Z, before the launch at 11:43:15Z) holds
§5's predictions unchanged in its Part A. They are copied here unchanged from the prompt:

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **V2 — the rows agree.** **Rule:** outside the stationary-point rows, every row with finite
  `total_abserr` under both has $|{\rm total}_g - {\rm total}_{06h}| \le {\rm
  total\_abserr}_{06h} + {\rm total\_abserr}_g$, and the same on the oracle. **Holds** if true at
  both rtols.
- **V3 — honest where referenced.** **Rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every
  counted reference, Z3's and the random sample's. $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ is
  reported beside it. **Holds** if no counted reference violates it; fewer than 45 of the 90 counted is
  **undecided**.
- **V4 — the trusted leaf's value.** **Rule:** on Z3's samples 2, 8, 9 and 14,
  $|v_g - {\rm ref}|/{\rm abserr}_g \le 0.01$ (06e's breakdown, with the quarters: 0.0012, 0.00046,
  0.0015). **Holds** if true on all four.
- **V5 — the cap.** **Rule:** outside the stationary-point rows no call under $g$ reaches
  `depth_max`, at either rtol. An over-budget unit counts as false. **Holds** if true at both.
- **V6 — Z2's call.** **Rule:** `g-1e-8-0-33995` call 5 ends below `depth_max`, converged or
  `node_limited`. **Holds** if so; its $\sum E$, $\sum|E|$, regions and depth are reported.
- **V7 — H4's call.** **Rule:** `g-1e-8-3-1072` call 13 converges. **Holds** if so.
- **V8 — no convergence lost.** **Rule:** outside the stationary-point rows, no call that 06h
  reported converged is unconverged under $g$. **Holds** if true at both rtols. A failure is a
  finding for the user: each such call is reported with its cause ((b) on a non-noise leaf,
  `node_limited`, or other).

V3 and V4 are the references agent's, and are not scored here.

| | Number | Verdict |
|---|---|---|
| **Gate V** | Git clauses hold (`21`); P1–P3 IDENTICAL to 06h's units; G1 = G2 on every key, and the unit qualifies; no Ray (`20`) | **holds** |
| **V2**, on `total` | 8,560 rows scored per rtol (0 non-finite, 0 without a result). **5 violations at each rtol**, in the same rows: `0-12959` 2.164, `0-12778` 2.076, `0-1873` 1.813, `0-33730` 1.777, `0-2108` 1.160 (ratio \|Δtotal\|/(e_06h + e_g), exact). Below the violations, the ratio's p99 is 0.499. Each row's values and errors in hex are in `out/cmd/49`. | **fails** (both rtols) |
| **V2**, on the oracle | 8,560 rows per rtol, 0 violations; max ratio 0.499 (1e-7) and 0.498 (1e-8) | (holds on this reading) |
| **V5** | Outside the 720, calls at depth 20 under $g$: **966 in 577 rows (1e-7), 996 in 582 rows (1e-8)**, all group calls (G−q+r 270/279, G+q+r 244/250, G+q−r 233/240, G−q−r 212/220, G−r 6/6, q+r 1/1); 165 and 168 of them still converged. 06h: 0 and 1. Over-budget units: 0. Every call is in `out/v5_depth_max_calls.csv`. | **fails** (both rtols) |
| **V6** | `g-1e-8-0-33995` #5 (G−q+r) ends **`node_limited`, at depth 20** (32 leaves at the cap, 15 of them untrusted). Abserr 7.164e-34 against a request of 6.689e-34; Σe′ 6.592e-34; ΣE −5.712e-35; Σ\|E\| 6.510e-34; 963 regions, 7,703 solves. Under 06h: unconverged at depth 20, 8,706 regions, 30,707 solves, abserr 1.333e-33, Σ\|E\| 1.312e-33, ΣE +2.431e-35 (fsum over 8,600 of 8,600 reproduced leaves). | **fails** |
| **V7** | `g-1e-8-3-1072` #13 "analytic J1" **converges**: abserr 1.544e-26 against a request of 1.629e-26, 2 regions at depth 1. Premise checks 2, firings 1. At the root the halving test failed (floors 1.945e-26 against 3.577e-27 + 3.577e-27) and the split could meet the budget. On a half the halving test held, the split could not meet the budget, and the exit fired. Under 06h: unconverged, 1 region, depth 0, abserr 2.359e-26, 1 premise check, which held. | **holds** |
| **V8** | `out/v8.md`. Scored: 33,725 group and 95,896 analytic calls (1e-7); 33,724 and 95,895 (1e-8). Lost: **801 calls in 470 rows (1e-7), 830 in 476 (1e-8)**. By class: "(b), refined to the depth cap" 801 / 470 rows and 827 / 474 rows; "node_limited" 0 and 3 / 3 rows; every other class 0. Gained: 1 (V7's call). 0 unmatched. | **fails** (both rtols); the note's §5 stop reached; the user ruled to score the rest |

### V8 in more detail (`out/v8.md`, `out/cmd/47`)

- Every lost call is a group call, in the regime groups (at 1e-7: G−q+r 217, G+q+r 213, G+q−r 195, G−q−r 174,
  G−r 1, q+r 1).
- Every (b) call still meets its request on abserr (abserr/request 5.7e-9 to 0.987, median 0.035, at 1e-7). Each
  returns at least one untrusted leaf at depth 20 (median 6, max 469), and the driver withholds `converged` while
  one is returned. Regions per call: median 7 → 135 (1e-7) and 9 → 140 (1e-8).
- The three node-limited calls (1e-8) are just above their requests:
  - `0-12783` #4: 8.951e-34 against 8.593e-34;
  - `0-33997` #4: 9.657e-34 against 9.645e-34;
  - `2-34662` #2: 3.39999e-34 against 3.39997e-34.
- **A caveat on V8's (b) class, found by the addendum** (`out/v8_slow_leaves.md`, "Coverage"). Part B's (b) test
  counts (b)-class leaves among every checked leaf, returned or bisected.
  - In 223 lost calls at 1e-7 and 227 at 1e-8, the untrusted leaves returned at depth 20 are not themselves in the
    (b) class: r ≥ 1 or not finite. Their returned (b)-class leaves were exempted floor-limited.
  - V8's classification is as fixed and is not changed.
- Inside the 720 (not scored): 316 calls lost in 212 rows (1e-7) and 340 in 228 rows (1e-8).

### V2 in more detail (`out/cmd/50`, `51`)

- In each violating row, the call that moves beyond both errors is the `G+r` call (index 1). Its value moves by
  7.1–7.2e-33 against abserrs of 3–7e-34 under each code, a ratio of 6–11. 06h returned 2 leaves at depth 1;
  $g$ returns 12–16 at depth 5–7. Both codes report it converged.
- At call level, outside the 720, **62 calls in 59 rows (1e-7) and 80 calls in 75 rows (1e-8)** move by more than
  abserr_06h + abserr_g, all group calls, every one on a different partition. This is not scored. Which side is
  wrong is for the references.

## The slow leaves behind V8's lost calls (the brief's addendum; descriptive)

`out/v8_slow_leaves.md` and `out/v8_slow_leaves.json`, from `scripts/v8_slow_leaves.py` (`out/cmd/56b`) and
`…_coverage.py` (`57`). It was computed from `out/v8_calls.json` and the region pickles' `checked_leaves_06g`, with
nothing rerun. The addendum's brief is `review/dispatch_census_addendum.md` (`36010d19…`).
- **The scope:** every untrusted (b) leaf a lost call returns, outside the 720: returned, 1/4 ≤ r < 1,
  Δ1 > φ_P, `trusted_final` 0.
  - 1e-7: **4,342 leaves in 578 calls**. A further 20,848 returned (b)-class leaves were exempted floor-limited.
  - 1e-8: **4,478 leaves in 600 calls** (22,349 exempted).
  - Every scoped leaf is at depth 20.
- **1. r** (1e-7; 1e-8 within a few per cent):
  - [1/4, 1/2) 1,980; [1/2, 3/4) 1,534; [3/4, 1) 828.
  - Quantiles: p10 0.421, p25 0.494, p50 0.501, p75 0.674, p90 0.852, max 0.999.
  - **r clusters at 1/2:** 1,763 of 4,342 are within 0.01 of 1/2, against 30 near 1/4 and 70 near 3/4. The
    histogram in 1/32 bins puts 1,380 in [0.469, 0.5) and 715 in [0.5, 0.531), and 67–135 in every other bin.
- **2. Δ1/φ_P:** median 2.27, p75 6.3, p90 10.4, p99 23.0, max 44.6, min 1.004. By r bin, the medians are 1.50
  ([1/4, 1/2)), 2.94 ([1/2, 3/4)) and 6.04 ([3/4, 1)); at most 2 in 1,373, 686 and 66 leaves respectively.
- **3. r against depth.** The chains are recorded: a bisected leaf that was checked before its split is in
  `checked_leaves_06g`.
  - Recorded ancestors per leaf: 16–20 of 20 levels (median 19), with every level from 10 to 19 recorded.
  - Over the recorded ancestors at depth ≥ 10, max |log2(r_anc/r_leaf)| has median 4.47; 7 of 4,342 chains stay
    within a factor 2.
  - Over depth ≥ 16, the median is 1.68; 651 stay within a factor 2 and 77 within 10 %.
  - The sample chains show r wandering, with values from 0.3 to 100 at depths 0–15, and settling near 0.4–0.5 over
    the last two to four levels.
  - Over every checked leaf of these calls, the median r by depth falls from 2.0 (depth 0) to 0.24 (depth 7),
    rises to 1.5 (depth 11), and falls to 0.50 at depths 19–20.
- **4. Position.** The rule was fixed before looking: consecutive depth-20 leaves join a cluster if their gap is at
  most 64 depth-20 widths of the call.
  - Clusters per call: **median 2, max 29** (264 of 578 calls have one).
  - Leaves per cluster: median 2, max 26.
  - **No cluster is within 2⁻¹⁰ of an end of the call's x_span** (0 of 1,556).
  - Cluster centres lie mostly in the upper half of the interval (u in [0.5, 0.9): 1,116 of 1,556).
  - Other sub-interval or partition endpoints are not in the call's records, so they were not tested.
- **5. By group.** The r distribution is the same in every group: median 0.500–0.501, and 39–44 % of leaves
  within 0.01 of 1/2.
  - G−q−r has the smallest Δ1/φ_P (median 1.51) and the widest late-depth wander (median max |log2| 2.17 at
    depth ≥ 16).
  - G+q+r's and G−q−r's cluster centres concentrate in u ∈ [0.7, 0.9); G+q−r's and G−q+r's are spread over
    [0.4, 0.9).
  - The q+r call has one leaf.

## Descriptive items (`out/cmd/49`, outside the 720 unless said)

**The values** (|v_g − v_06h|/abserr_g per call with finite abserr_g). At 1e-7, then at 1e-8:

| Cause | Calls | Median ratio | p99 | Max | Ratio > 1 |
|---|---|---|---|---|---|
| Different partition | 43,715 / 50,213 | 0.608 / 0.472 | 2.12 / 1.87 | 26.7 / 26.7 | 1,805 / 1,560 |
| Same partition, quarters returned | 80,502 / 73,965 | 0.864 / 0.545 | 1 / 1 | 1 / 1 | 0 / 0 |
| Same partition, no quarters | 5,404 / 5,443 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- On the same partition, abserr is unchanged in every call without a CC leaf (78,578 and 71,742 calls).
- With a CC leaf (ΣE in the error), abserr_g/abserr_06h runs from 0.277 to 1.56, median 1, with 2,610 of 7,328
  calls (1e-7) unchanged.

**Slow leaves**, test 3's classes, as (Δ1 > φ_P)/(Δ1 ≤ φ_P), all rows:

| | r < 1/16 | [1/16, 1/4) | [1/4, 1) | ≥ 1 | Total |
|---|---|---|---|---|---|
| 1e-7, $g$, every checked leaf | 371,479/128,095 | 331,317/36,794 | 592,745/176,712 | 558,406/536,925 | 2,732,473 |
| 1e-7, $g$, returned | 369,107/128,095 | 330,247/36,793 | 130,597/176,648 | 105,204/252,397 | 1,529,088 |
| 1e-7, 06h, returned (bisected not recorded) | 147,165/108,651 | 50,979/15,142 | 59,285/83,385 | 791/111,027 | 576,425 |
| 1e-8, $g$, every checked leaf | 397,438/139,109 | 370,160/45,935 | 637,283/224,593 | 585,919/627,087 | 3,027,524 |
| 1e-8, $g$, returned | 392,990/139,109 | 367,684/45,932 | 146,723/224,467 | 115,931/296,221 | 1,729,057 |
| 1e-8, 06h, returned | 176,113/115,237 | 77,956/21,187 | 74,843/117,043 | 3,312/144,258 | 729,949 |

- Calls outside the 720 with a (b) leaf: 22,845 at 1e-7 (20,793 group, 2,052 analytic). They add **776,371
  regions and 6.21 M solves** against 06h, and 22,044 of them converge.
- At 1e-8: 21,362 calls (35 analytic), adding 789,433 regions and 6.35 M solves; 20,531 converge.

**The node sum:**
- Calls with a CC leaf: 24,983 (1e-7) and 26,441 (1e-8).
- Calls where Σ|E| ≥ 1 % of abserr_g: 3,110 and 3,431, none with ΣE = 0 exactly.
- Σ|E|/|ΣE| over those calls: median 1, p90 10.5 and 11, p99 98.7 and 111, max 1.76e4.

**The exit:**
- Firings: 06h 90,509 → $g$ 402,760 (1e-7); 127,498 → 461,974 (1e-8).
- Premise checks: 260,979 → 691,088 and 339,559 → 808,455.
- Calls whose firing count changed: 24,890 at 1e-7 (22,067 more, 2,823 fewer; 15,222 analytic, 9,668 group), and
  29,224 at 1e-8 (25,565 more, 3,659 fewer).

**Cost — a finding.** The census took **5 h 40 min of wall time against 06h's 2 h 33 min**. The 1e-7 set ran
11:43–14:30Z and the 1e-8 set 14:30–17:23Z.

| | 1e-7, 06h | 1e-7, $g$ | 1e-8, 06h | 1e-8, $g$ |
|---|---|---|---|---|
| **Unit wall summed (`wall_s`)** | 2.223 h | **7.616 h (×3.43)** | 2.919 h | **8.643 h (×2.96)** |
| Largest unit | 5.7 s | 47.1 s (`g-1e-7-0-36806`) | 105.2 s (`g-1e-8-0-33995`) | 46.8 s (`g-1e-8-0-36806`) |
| Post-timed, summed | 1.970 h | 3.380 h | 1.892 h | 2.602 h |
| Group regions, all rows | 227,373 | 1,124,662 (×4.95) | 329,525 | 1,255,648 (×3.81) |
| Group solves, all rows | 1.782 M | 8.960 M (×5.03) | 2.560 M | 10.008 M (×3.91) |
| Analytic regions, all rows | 349,052 | 404,426 (×1.16) | 408,628 | 473,409 (×1.16) |
| Analytic solves, all rows | 2.685 M | 3.128 M (×1.17) | 3.162 M | 3.680 M (×1.16) |

Cost is not an argument (README §5 rule 4).

**Depth and convergence,** outside the 720:
- Group calls converged: 33,725 → 32,924 (1e-7) and 33,724 → 32,894 (1e-8).
- Analytic calls converged: 95,896 → 95,896 and 95,895 → 95,896.
- Rows with `total_converged` false: 0 → 470 and 1 → 476.
- Group calls at depth 20: 0 → 966 and 1 → 996. Group calls at depth 13–19: 14 → 5,553 (1e-7).
- Analytic calls reach depth 15 at 1e-7 under $g$ (06h: 8) and 6 at 1e-8 (06h: 6).
- The full histograms are in `out/cmd/49`.

**The 720 stationary-point rows** (07's; not scored):
- Every total and oracle moved, at both rtols.
- Group calls converged: 3,591 → 3,275 (1e-7) and 3,586 → 3,246 (1e-8); calls at depth 20: 11 → 357 and 17 → 385.
- Rows with `total_converged` false: 9 → 220 and 14 → 241.
- Calls lost: 316 and 340; gained: 0.

**The near-singular class:** A3 v2 reaches it. 06h's rule, |K|/(k + q c_s + r c_s) ≤ 10⁻³, picks out 435 rows
(06h: 435), none in the 720.
- Analytic calls lost: 0 at both rtols; gained: 0 and 1.
- Group calls lost: 43 at each rtol.
- The oracle's convergence is unchanged: 133 of 435 at 1e-7, 131 at 1e-8.
- The oracle's |Δ|/(e_06h + e_g) is at most 1.6e-7.

## Deviations from the prompt (numbered and classified)

1. **`prov.py`:**
   - the trees: `p` is `…-lpf06g-4f4f9ed` and `g` is `…-lpf06g-1c5336e`;
   - 06h's run, records, rows and checkpoint SHA-256 (`RUN06H`, `REC06H`, `ROWS06H`, `CHECKPOINT06H_SHA256`);
   - 06h's trees are kept by name.

   STRUCTURALLY REQUIRED.
2. **`harness.py`:** `job_name="levin-phase-floor-06g"` (the brief); stored, never read. STRUCTURALLY REQUIRED.
3. **`run_census.py`:** docstring only. STRUCTURALLY REQUIRED.
4. **`launch.py`:** prompt `06g`, purpose, scope, imports. STRUCTURALLY REQUIRED.
5. **`worker.py`, tree evidence:** 06g's markers and the premise's signature. The worker refuses a signature that
   is neither 06h's nor 06g's. STRUCTURALLY REQUIRED (the brief: check the signature the wrapper forwards).
6. **`worker.py`, the premise pass-through.**
   - It forwards keywords (`split_could_meet_budget=`). Without that, 06h's wrapper would have raised TypeError at
     every premise check at $g$ (the code log's observation 6). STRUCTURALLY REQUIRED.
   - It records each check's detail: the result, the budget clause, the halving test (the pure original called
     once more with no keyword) and the floors. IMPLEMENTATION CHOICE, allowed by the brief.
7. **`worker.py`, every checked leaf:** a pass-through at `used_interval.__init__` keeps every region the driver
   makes; checked leaves are recorded after the call, returned or bisected.
   - The pass-through is a class attribute, installed for the timed section only, then restored and asserted.
   - It runs inside the timed section, as 06h's premise and solve counters do.

   IMPLEMENTATION CHOICE, allowed by the brief.
8. **`worker.py`, per-call fields:** `node_*`, `has_cc_leaf`, `abserr_eprime_sum`/`abserr_reassembles`,
   `values_06g`, `premise_06g`, `checked_06g` and `error_06g`. Allowed by the brief.
9. **`worker.py`, returned-leaf arrays:** value, value_own, value_kind, the floor's parts, E and e′. Allowed by the
   brief.
10. **`worker.py`, `_budget_record`:** parses 06g's 16-argument record; 06h's parser kept it as raw text.
    STRUCTURALLY REQUIRED.
11. **`worker.py`, `_cc_floor_terms` and `_declared_terms_06h`:** they keep the first `_cc_roundoff_floor` capture
    of each re-solve. 06g's fallback solve calls it twice, and the second call overwrote the capture, so every
    fallback leaf at $g$ would have recorded "not reproduced". At $p$ nothing changes. STRUCTURALLY REQUIRED.
12. **`gate_v.py`** (from `gate_h.py`):
    - the before is 06h's census;
    - the comparison at $p$ also covers `declared_terms_06h`, `analytic_06h.h` and `n_captured`;
    - the brief's list is printed call by call in hex;
    - 06g's new pickle keys are compared between G1 and G2, and every pickle key is accounted for;
    - 06g's qualification test for the determinism unit;
    - 06h's counterfactual check is made descriptive.

    IMPLEMENTATION CHOICE.
13. **`gate_v_clause1.py`:** new; adds clauses 2 (`c1ec3b2..4f4f9ed`) and 3 (`3ae6f3a..1c5336e`) beside the
    prompt's. IMPLEMENTATION CHOICE.
14. **Helper scripts:** `scripts_equal.py` (state/V), `check_predictions.py` (06g, lines 401–422),
    `checks_after_06g.py`, `static_names.py`, `smoke_summary.py`. IMPLEMENTATION CHOICE.
15. **06h's oracle counterfactual** (06e's BesselPhaseGroup) is still computed at $g$, unchanged. At $g$ it means
    06e's group phase under 06g's driver, and no 06g score uses it. It costs post-timed time outside `wall_s`, and
    is part of the run's wall time. IMPLEMENTATION CHOICE.
16. **`out/cmd/00` was recorded by an inline equivalent of rec.sh.** IMPLEMENTATION CHOICE.
17. **`04b` and `04c` ran `static_names.py` while it held only its docstring** (a BSD sed error while writing it).
    Both are annotated; `04d` is the check. UNINTENDED DRIFT.
18. **Scripts are not `black`-formatted;** they are records. IMPLEMENTATION CHOICE.
19. **Tooling for the edits.** The copied scripts were edited by string replacement under
    `/opt/local/bin/python3` (3.14.8), and the static check runs under it; neither imports any tree module. The
    parent's module was read via `git show` into the session scratchpad. IMPLEMENTATION CHOICE.
20. **`gate_v.py`'s P-unit lines still print the labels "vs 06e", "06e deep lines" and "CC floor-term leaves
    06e".** The comparison is with 06h's run. UNINTENDED DRIFT (cosmetic).
21. **`out/cmd/11` piped rec.sh through sed,** so the lock's listing shows that sed (as 06h's deviation 18). `29`,
    `12` and `30` (empty: none) do not. UNINTENDED DRIFT.
22. **The smoke summary (22b) is `smoke_summary.py`.** A first inline attempt failed on shell quoting before
    rec.sh ran, and wrote nothing. IMPLEMENTATION CHOICE.
23. **Part B's V8 classification was extended before the launch** (the coordinator's message after R24): the
    ending is read beside (b), giving five classes mapped onto the prompt's three causes. The stage-1 draft is
    kept in `state/drafts/`. IMPLEMENTATION CHOICE.
24. **The determinism unit `g-1e-7-0-14164` qualified,** so the re-choice rule in the launch plan was not used.
    (Recorded for completeness.)
25. **New scoring scripts** `v8_06g.py`, `v8_breakdown.py`, `analyse06g.py`, `v2_detail06g.py` and
    `values_beyond06g.py`. `out/v8.md`'s last section ("Descriptive, from the same lost calls") was appended by
    hand after the scoring run, before the stop was reported, and is marked so. IMPLEMENTATION CHOICE.
26. **The fingerprint after the run was taken at $g$ only,** as 06h's was. IMPLEMENTATION CHOICE.
27. **The addendum's scripts:**
    - `v8_slow_leaves.py` and `…_coverage.py`;
    - the depth ≥ 16 window of item 3 was added after a first look at the output (`56`, re-run as `56b`); the
      clustering rule was not changed;
    - `out/v8_slow_leaves.md`'s "Coverage" section was appended by hand from `57`.

    IMPLEMENTATION CHOICE.
28. **"Calls whose exit changed" is reported as counts by direction and kind.** The list itself is not written
    out: 24,890 and 29,224 calls. IMPLEMENTATION CHOICE.

None of the UNINTENDED DRIFT items (17, 20, 21) touched a store, a tree, or any record outside this directory and
the run directory.

## Verification performed

- **The trees:** $g$ at `1c5336e` and $p$ at `4f4f9ed`, each with empty `status --short --ignored`, checked at
  preparation, before Gate V, at the launch, on all 18,560 records, and at the end. No `__pycache__` was written.
  The main checkout was at `28ebaf0` with empty status throughout.
- **The copy:** byte-identical to `copy_files_certified.txt` before Gate V, after Gate V and the smoke test, at
  the launch, after the run, after the fingerprint and at the end, with no journal, `-wal` or `-shm` file. It gives
  `0e32f24a…` at $g$ and $p$ before, and at $g$ after.
- **06h's run was only read:** its checkpoint is `3104c18e…` before, after preparation, after the run, in each
  scoring script, and at the end.
- **The run:**
  - 18,560 records, every unit `done`, every one at $g$ and clean;
  - every worker reported `ray_initialized` false and no foreign modules;
  - every fallback floor-term leaf, declared-term leaf and deep line reproduces, and every call's abserr and value
    reassemble bit for bit.
- **The scoring:**
  - V2 was evaluated exactly (`Fraction`) on the recorded doubles;
  - calls were matched by kind and index with equal label and span, 0 unmatched;
  - every pickle record read was SHA-checked;
  - the 720 count was asserted.

## Observations not acted on

All of these are in `issues.md`:
- **C1:** V8 fails, through refinement to depth 20.
- **C2:** V5 fails.
- **C3:** V2 fails, with call-level moves beyond both errors.
- **C4:** V6 fails, through the cap.
- **C5:** the cost.
- **C6:** the node-limited stop costs three calls at 1e-8.
- **C7:** r clusters at 1/2 in the slow leaves (descriptive).
- **W3 (not an issue):** W3's `premise_holds` (1e-7 units) is still 06b's formula on whole floors, not 06g's
  premise, which `premise_06g` records. It is a harness record, not a defect of the code, so I propose no board
  entry.

## State handed on

- **Nothing of mine is running.** The run is `done`, no census worker process remains, no lock was taken, and Ray
  was never started. Nothing was written in either worktree, or in the main checkout outside this directory and the
  run directory.
- **The scratch copy is unchanged** (`0e32f24a…`).
- **The run directory is kept in full** (4.5 GB) for 07's census and the references agent:
  `var/runs/levin-phase-floor-06g-census-20261101T114315/`, checkpoint `03adcbba…`.
- **For the orchestrator:**
  - copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/` and `state/` to
    `docs/levin-phase-floor/06g/census/`;
  - leave out the files `KEPT_ON_MAC_STUDIO.txt` lists: the pickles under `state/V/` and `state/smoke/`,
    `state/rows.json` (7.8 MB), `out/v8_calls.json` (6.9 MB, `1e9a62b0…`, the V8 sample's source) and
    `out/v8_slow_leaves.json` (15.0 MB, `b4643033…`).
- V3, V4 and "the correction −ΣE" are the references agent's.
