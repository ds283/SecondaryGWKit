# levin-phase-floor 06h §5 — the census (Mac Studio)

**Subject:** prompt 06h §5's census on the Mac Studio (`mps026298`): 06e's census re-run under 06h's
code, Ray-free, with no lock, against 06e's run as the "before". It covers Gate H, H3, H4 and the
descriptive items; H1 and H2 (the phase re-measurement) and §4 are the other Part B agents'. This run
is 06g's "before".
- **Trees:**
  - $h$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-c1ec3b2` at Part B's
    dispatch SHA `c1ec3b2`; its code is `7172a8f` + `ac1c6f4` (tests only). The scripts call it `g`.
  - $p$ is `…-lpf06h-2c325f5`, the code commit's parent, whose code is 06e's. It is used for Gate H
    only.
  - 06e's census tree `6d81fad` is read only through git history.
- **Runs:** the census, `levin-phase-floor-06h-census-20261101T005037`, launched at 00:50:37Z
  (sh 73326, driver 73329). It ended `done`, exit 0, with its last heartbeat at 03:23:50Z: about
  **2 h 33 min**, against 06e's 1 h 47 min. The difference is the post-timed instrumentation
  (deviation 20), not the code: the unit wall is ×0.97 and ×0.98 of 06e's.
- **Commit:** none here. The records are committed by the orchestrator from the laptop, as
  `docs/levin-phase-floor/06h/census/`.

**Result:**
- **Gate H holds** on both parts (git clause, bit-for-bit reproduction at the parent, determinism at h).
- **H3 holds** at both rtols. As written, on `total`, 8,560 rows are scored per rtol with 0
  violations; in fact every `total` and `total_abserr` outside the 720 is bit-identical to 06e's.
  On **the second reading** (the analytic oracle, the orchestrator's request), 8,560 rows per rtol
  are scored with 0 violations; the worst ratio is 0.0756 at 1e-7 and 1.35e-4 at 1e-8. **The two
  readings agree.**
- **H4 fails at 1e-8 on one call:** `g-1e-8-3-1072` call 13, "analytic J1", converged under 06e,
  unconverged under $h$. At 1e-7 it holds. Scored calls per rtol: 95,896 analytic-block and 33,725
  group calls. No call gained convergence.
- **The store** stayed byte-identical to its certification throughout, fingerprint `0e32f24a…`.
  06e's checkpoint is `a0dc4d7f…` throughout. No worker started Ray.

## What shipped

No code. Records only, in `var/levin-phase-floor-06h/census/` (this directory) and one run
directory.
- **`scripts/`:** 06e's census harness, changed as deviations 1–9, 15 and 16 say. 06e's copy is kept
  unchanged in `state/scripts_06e_reference/`.
  - The diffs are `out/scripts_diff_against_06e_at_prep.txt` and `…_final.txt`. The new files are
    diffed against their 06e sources in `out/scripts_diff_new_files_against_06e_sources_*.txt`.
    The worker's change after the first Gate H is `out/worker_diff_gate_h_1_to_final.txt`.
  - New for the gate and the checks: `gate_h.py`, `gate_h_clause1.py` and `checks_after_06h.py`.
  - New for scoring, after the run: `analyse06h.py`, `h4_detail06h.py` and `nearsingular06h.py`.
- **`state/`:**
  - `predictions.md` (sha256 `6ff4f05b…`, mtime 00:49:57Z, fixed before the launch), with its drafts
    in `drafts/`;
  - `H/` and `H2/` (Gate H's two runs), `smoke/`, `rows.json`, `launch_plan.txt`;
  - `drafts/worker_at_gate_h_1.py` (the worker the first Gate H ran).
- **`out/`:**
  - every command and its output (`out/cmd/00`–`55`, `out/commands.log`);
  - `census_rows_06h.csv.gz`, one line per unit with both runs side by side, the oracle and its
    counterfactual included;
  - `h4_changed_calls_06h.json`.
- **Run `levin-phase-floor-06h-census-20261101T005037`:** 2.3 GB, 149 files. It holds the checkpoint
  (sha256 `3104c18e…`, 1,642,892,823 bytes, 18,560 records), 16 region pickles, the events and the
  worker logs. **It is kept in full on the Mac Studio for 06g's census** (`KEPT_ON_MAC_STUDIO.txt`).

**Where each per-call field lives.** The checkpoint's `data.calls` holds the group calls and
`data.other_calls` the analytic-block calls. Per call:

| Need | Field(s) |
|---|---|
| H4 and matching | `index`, `notify_label`, `x_span` (hex), `converged` |
| The call | `value`, `abserr` (repr + hex), `num_regions`, `num_subregion_solves`, `max_depth`, `depth_hist_levin_cc` ([Levin, CC] leaves per depth), `rtol`, `atol`, `premise_checks`, `premise_failed`, `budget` |
| Declared term, Clenshaw–Curtis branch (06e and $h$; 06f's method) | `cc_floor.sums_floor_attributed_cc_leaves.declared`, `cc_floor.sums_all_cc_leaves.declared`, `cc_floor.n_floor_attributed` |
| Declared term by branch ($h$ only, new) | `declared_06h`: `request`, `has_parts`, `n_leaves`, `n_reproduced`, `n_errors`, and for each of `levin`, `levin_floor_attributed`, `cc`, `cc_floor_attributed`: `n`, `n_reproduced` and the sums of `declared`, `declared_constituents`, `declared_abscissa`, `declared_phi_eval`, `floor` and `phase_err_resolve`, with the maximum endpoint `theta_abserr` |

- **Per unit** (checkpoint):
  - H3: `result.total`, `result.total_abserr`, `result.total_converged`, `result.analytic_rad`;
  - H3's second reading (new): `analytic_06h.h` (value, abserr, converged) and
    `analytic_06h.parent_counterfactual` (06e's oracle recomputed: value, abserr, converged);
  - `status`, `wall_s` (the timed section only), and the post-timed seconds (`declared_06h_s`,
    `analytic_06h.seconds`, `cc_floor_s`, `scan_s`, W3 and deep lines);
  - `tree_head`, `tree_status_short`, `scripts_sha256`, `modules` (counts by root, foreign modules,
    the tree's files, `LiouvilleGreen.three_bessel_integrals` among them);
  - the registry's `run_census.py` stamp `e11dcd72…` on every record.
- **Per leaf** (pickle `regions_<worker>.pkl` at `pickle_offset`/`pickle_length`, SHA-checked):
  - 06e's `regions` and `other_regions` arrays (depth, and branch as `type`: 0 Levin, 1 CC), and
    06e's `cc_floor_terms`;
  - new `declared_terms_06h`: per leaf `branch`, `floor_attributed`, `declared`, `floor`,
    `theta_abserr_a/_b`, `p_l1_a/_b`, the three parts of `theta_abserr` at each end, the declared term
    split by part, and `reproduces`.

## Before the launch (every command and its output in `out/cmd/`)

1. **Preparation** (`00`–`04`, no computation): the harness copied and diffed, `rows.json` copied
   (`aa523d28…`), a syntax check.
2. **State** (`10`–`12`, `28`–`30`): the trees are clean at `28ebaf0`, `2c325f5` and `c1ec3b2`; 38
   runs, all finished; the lock is free; no Ray process under `/tmp/ray-sgk`. `02` is the launcher's
   dry import.
3. **The store** (`13`–`16`): the copy is EQUAL to `copy_files_certified.txt`, with no journal, `-wal`
   or `-shm` file. Its fingerprint is `0e32f24a…` with no problems, at $h$ and at $p$.
4. **06e's checkpoint** (`17`): `a0dc4d7f…`. `18`: `job_name` is stored and never read.
5. **The coordinator's two facts** (`05`): see deviations 15 and 20.
6. **Gate H** (`19`–`21` on the first worker, `19b`–`21b` on the final one; both hold):
   - **The git clause.** `git diff --stat 6d81fad 2c325f5 -- . ':!docs' ':!prompts'` shows
     `CLAUDE.md` only. The harness imported 102 modules from $p$ (`state/H2/p_tree_modules.json`,
     equal to `state/H/`'s), so no file needed the AST test.
   - **The parent reproduces 06e bit for bit.** At $p$ these three units are IDENTICAL to 06e's, on
     every key 06e recorded (group and analytic calls with `cc_floor`, budget counters, solves by
     purpose, every region array, `cc_floor_terms`, deep lines, the slim and full results, GkPolicy,
     the scan, W3):

     | Unit | Total |
     |---|---|
     | `p-1e-7-2-38302` (`low-G+r-1`) | `0x1.5e6b398fe7c40p-101` |
     | `p-1e-7-2-2095` (`ctrl-low-1`) | `-0x1.e3272e29c8b71p-102` |
     | `p-1e-8-0-1971` | `0x1.6ad248e2d3ab7p-66` |

     Every value is compared as a `float.hex` string or as array bytes.
   - **The new code is deterministic.** `g-1e-7-0-14164` is IDENTICAL in two fresh workers at $h$ on
     every key, 06h's new keys (`declared_06h`, `declared_terms_06h`, `analytic_06h`) included.
     - It qualifies because its 16 analytic-block calls are each handed a `BesselPhaseGroup` that
       supplies `theta_abserr`, and all 16 have every leaf reproduced and a non-zero declared term.
     - Its `total` is bit-identical to 06e's; its oracle moved
       (`0x1.7221577ef7fcep-84` against 06e's `0x1.7221577ef7858p-84`).
   - **06e's oracle recomputed as a counterfactual** (not part of Gate H): on the G unit it equals
     06e's recorded `analytic_rad` bit for bit.
   - No worker started Ray.
7. **The smoke test** (`22`, `22b`): 2 rows × 2 rtols, 4 units `done` at $h$; 297 of 297 declared
   leaves reproduced; counterfactual present; no foreign module.
8. **Predictions** (`25`): `state/predictions.md`, sha256 `6ff4f05b…`, mtime 00:49:57Z. Part A is
   prompt lines 334–347 byte for byte. `26`, `33`: every run script equals those Gate H's second run
   and the smoke test ran.
9. **Pre-launch** (`28`–`34`): all as before; the copy is EQUAL (`31`, `32`).
10. **Launch** (`35`) and one check (`36`): `running`, six 1e-7 workers started, stderr empty.

## After the run

- **Status** (`38`): `done`, exit 0, 18,560 of 18,560 units, `stderr.log` empty. All 16 workers'
  `bye` messages report `ray_initialized` false and no foreign modules.
- **Every record** (`39`):
  - all are at `c1ec3b2` with empty status, carry `run_census.py` `e11dcd72…`, and have scripts
    equal to `scripts/`;
  - **none is over budget, none is in error, and none has `wall_s` over 300 s**, so no over-budget
    pass was run;
  - 278,727 of 278,727 Clenshaw–Curtis floor-term leaves reproduce;
  - 757,680 of 757,680 declared-term leaves reproduce (214,832 calls supplying `theta_abserr`, all
    with parts; 74,650 calls supplying none);
  - 18,560 of 18,560 oracle counterfactuals were computed, each from exactly one captured
    `analytic_integral` call;
  - 200 deep lines past depth 12, all reproducing.
- **The store** (`40`–`44`, `53`, `54`): EQUAL after the run, after the fingerprint (`0e32f24a…`, no
  problems) and at the end.
- **06e's checkpoint** (`45`, `55`): `a0dc4d7f…`.
- **Scores:** `46` (`analyse06h.py`), `47`–`48` (the H4 call in detail), `49`–`50` (the near-singular
  class).
- **Final state** (`51`, `52`): the trees are clean; 39 runs, all finished; no census worker process;
  `state/predictions.md` is unchanged (`6ff4f05b…`, mtime 00:49:57Z).

## Predictions and verdicts

`state/predictions.md` (sha256 `6ff4f05b…`, mtime 2026-11-01T00:49:57Z, before the launch at
00:50:37Z) holds §5's predictions unchanged in its Part A. They are copied here unchanged from the
prompt:

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "The
stationary-point rows" are 06a's 720 (`spscan.py`).
- **H1 — the declaration is honest.** **Rule:** on every one of the 20 M2 calls, at every point
  sampled, $|\delta\theta(u)| \le \texttt{theta\_abserr}_h(u)$, the reference at the exact $e^u$.
  **Holds** if no point violates it. The parent's declaration is scored on the same points beside it.
- **H2 — the declared term covers the response.** **Rule:** on every returned leaf of the 20 calls,
  the first-order response to the measured phase error is at most that leaf's declared term.
  **Holds** if no leaf violates it.
- **H3 — the rows agree.** **Rule:** outside the stationary-point rows, every row with finite
  `total_abserr` under both has $|{\rm total}_h - {\rm total}_{06e}| \le {\rm total\_abserr}_{06e} +
  {\rm total\_abserr}_h$. **Holds** if true at both rtols.
- **H4 — no convergence is lost.** **Rule:** outside the stationary-point rows, no Levin call that
  06e reported converged is unconverged under $h$. **Holds** if true at both rtols. A failure is a
  finding for the user: §0.2's estimate of what the declaration costs the driver is wrong.

H1 and H2 are the phase agent's, and are not scored here.

| | Number | Verdict |
|---|---|---|
| **Gate H** | Git clause: `CLAUDE.md` only. P1–P3 IDENTICAL to 06e's units. G1 = G2 on every key. No Ray. (`20b`, `21b`) | **holds** |
| **H3** (as written, on `total`) | 1e-7: 8,560 rows scored (0 non-finite), 0 violations. 1e-8: 8,560, 0 violations. Exact (`Fraction`). Every `total` and `total_abserr` outside the 720 is bit-identical to 06e's at both rtols. (`46`) | **holds** |
| **H3, second reading** (the analytic oracle; orchestrator's request; Part B) | The oracle moved in 4,203 rows outside the 720 at 1e-7 and 3,920 at 1e-8. 06e's counterfactual equals 06e's recorded `analytic_rad` bit for bit on all 8,560 rows at each rtol, so all are scored, 0 non-finite and 0 violations. $\|a_h-a_{06e}\|/({\rm e}_{06e}+{\rm e}_h)$: median 0, p99 1.02e-6, max **0.0756** at 1e-7; p99 2.18e-6, max 1.35e-4 at 1e-8. (`46`) | **holds**; the two readings agree |
| **H4** | Scored outside the 720 at each rtol: **95,896 analytic-block calls and 33,725 group calls**. 06e converged: at 1e-7, all of both kinds; at 1e-8, 95,896 analytic and 33,724 group. **Lost at 1e-7: 0. Lost at 1e-8: 1**, analytic, `g-1e-8-3-1072` call 13 "analytic J1": 06e converged (2 regions, depth 1, abserr 7.16e-27) and $h$ unconverged (1 region, depth 0, abserr 2.36e-26, against a request of 1.63e-26). Gained: 0 at both. 0 unmatched. (`46`, `47`, `48`) | **fails** (1e-8) |

**The H4 call in detail** (`47`, `48`):
- **The row:** `3-1072`, k = 1e5, q = r = 3e8, z_response 1.21e9, band seam, outside the 720.
- **Under 06e** the driver tested the round-off exit's premise twice, and it failed once: at the
  root, so the root was bisected. One half was then accepted floor-limited. Each half's floor was
  about 4.5e-27, giving the call 7.16e-27 against its request of 1.63e-26: converged.
- **Under $h$** the premise was tested once and held, at the root. The root (the whole span) was
  accepted floor-limited at depth 0, with no split. Its `phase_err` is 2.36e-26: the eq. (151) floor
  of 1.94e-26 plus the declared term of 4.14e-27.
  - The declared term is 0.25 of the request, almost all of it (4.136e-27) the abscissa part.
  - The constituents' part is 1.9e-30, which is what 06e's declaration would charge here.
- **So the root's floor already exceeded the request under both codes.** The larger declaration
  made `_roundoff_exit_premise_holds` (halves' floors ≥ ½ the root's) pass at the root. Under 06e it
  failed, and the bisection reached a floor below the request.
- I infer that the halves' floors under $h$ satisfied the premise from the premise counters and the
  budget record. The halves' $h$ floors themselves are not recorded at 1e-8 (W3 runs at 1e-7 only).
- **Effect on the row:** the row's `total` does not depend on this call, and is bit-identical. The
  oracle's value is bit-identical too (`-0x1.9f88a8d93a945p-58` in both), and the oracle is
  unconverged under both codes (abserr 4.5e-19 against |value| 5.6e-18).
- **The prompt's reading:** this is the finding H4 names, "§0.2's estimate of what the declaration
  costs the driver is wrong", in a narrow form. The declared term itself stays a quarter of the
  request. But on a span whose floor sits near the request, raising it can flip the round-off exit's
  premise and stop a bisection that 06e's code used to converge.

## Descriptive items (`out/cmd/46`, `49`, `50`)

**Fact 1, checked here.** 06h's code moves only the analytic oracle:
- **Totals:** every `total` and `total_abserr` outside the 720, and every total inside it, is
  bit-identical to 06e's at both rtols.
- **Group calls:** every one of the 37,325 group calls per rtol is bit-identical to 06e's in value,
  abserr, converged, regions and depth.

**What the declaration costs** (analytic-block calls, 107,416 per rtol).
- **The Clenshaw–Curtis branch, 06f's method, on both checkpoints.** Floor-attributed fallback leaves:

  | | 1e-7, 06e | 1e-7, $h$ | 1e-8, 06e | 1e-8, $h$ |
  |---|---|---|---|---|
  | Calls with a declared term > 0 | 5,224 | 5,299 | 6,055 | 6,117 |
  | Declared-dominated calls | 2,493 | **5,137** | 2,785 | **5,939** |
  | D ≥ request / ≥ 0.1 / ≥ 0.01 request | 0 / 0 / 1 | 0 / 0 / 2 | 0 / 1 / 22 | 0 / 2 / 23 |
  | Unconverged | 0 | 0 | 0 | 0 |

  Declared term over all fallback leaves, $h$ over 06e on matched calls: median 1.95 and 1.97, p99
  5.4 and 5.2, max 31.8.
- **By branch under $h$** (`declared_06h`; 06e did not record the Levin branch, deviation 12). Total
  declared over the request per call:

  | | 1e-7 | 1e-8 |
  |---|---|---|
  | Median | 5.3e-6 | 6.3e-5 |
  | p99 | 3.7e-4 | 4.0e-3 |
  | Max | 0.034 | **0.341** |

  - **At 1e-7:** no call at ≥ 0.1 of its request, 6 at ≥ 0.01.
  - **At 1e-8:** 6 calls at ≥ 0.1 of the request, 349 at ≥ 0.01, and 7,055 at ≥ 0.001.
  - The Levin branch carries the declared term on 102,603 calls, the CC branch on 6,166 and 6,960.
  - **The abscissa part's share** of a call's declared term: median 0.26 and 0.28, p90 0.97.
  - **Declared over its constituents' part** (what 06e's declaration would charge on $h$'s leaves):
    median 2.4, p90 62–74, p99 2.4e3–2.7e3, max 1.9e4.
  - So the declaration rose by up to four orders of magnitude on some calls, and still stays
    below 0.35 of every call's request. Prompt §0.2's "the declaration should rise by orders in the
    tail while the declared error of a call barely moves" holds in magnitude; H4's one call is the
    exception through the round-off exit's premise.
- **Convergence and depth:**

  | | 1e-7, $h$ | 1e-7, 06e | 1e-8, $h$ | 1e-8, 06e |
  |---|---|---|---|---|
  | Analytic calls unconverged | 0 | 0 | **1** | 0 |
  | Group calls unconverged | 9 | 9 | 15 | 15 |
  | Rows with `total_converged` false | 9 | 9 | 15 | 15 |
  | Calls at depth 20 (all group) | 11 | 11 | 18 | 18 |

  - The analytic calls get shallower, with the mass moving from depth 4–5 to depth 3. At 1e-7:
    depth 3 25,353/9,101, depth 4 15,520/30,546, depth 5 1,147/4,087 ($h$/06e); maximum depth 8
    (06e: 9). At 1e-8: depth 3 32,906/13,833, depth 4 17,235/38,780, depth 5 142/1,693.
  - Group call depths are identical. The full row and call histograms are in `46`.
- **Cost:**

  | | 1e-7, $h$ | 1e-7, 06e | 1e-8, $h$ | 1e-8, 06e |
  |---|---|---|---|---|
  | Analytic regions | 349,052 | 398,505 (×0.876) | 408,628 | 469,661 (×0.870) |
  | Analytic solves | 2.685 M | 3.081 M (×0.872) | 3.162 M | 3.650 M (×0.866) |
  | Group regions and solves | equal | | equal | |
  | Unit wall summed (`wall_s`) | 2.223 h | 2.282 h (×0.974) | 2.919 h | 2.974 h (×0.981) |
  | Largest unit | 5.7 s | 6.0 s | 105.2 s (`g-1e-8-0-33995`) | 105.1 s |

  - **Post-timed, outside `wall_s`, under $h$** (h of CPU): the oracle counterfactual 1.31 and 1.56;
    the declared terms 0.21 and 0.24; `cc_floor` 0.04 and 0.09; the scan 0.27 (1e-7 only); W3 0.14
    (1e-7 only).
  - The run's 2 h 33 min against 06e's 1 h 47 min is that instrumentation. Cost is not an argument
    (README §5 rule 4).
- **The 720 stationary-point rows** (07's; not scored): no total moved at either rtol. The oracle
  moved on 462 rows (1e-7) and 421 (1e-8), none beyond the second reading's bound (max ratios
  5.7e-6 and 2.9e-6). No call gained or lost convergence.
- **The near-singular class:** A3 v2 reaches it, by Part B's pre-fixed rule
  $|K|/(k + q c_s + r c_s) \le 10^{-3}$ (`49`, `50`).
  - Every row has b = 0. The rule picks out **435 rows**, 301 of them with analytic Levin calls.
  - All 435 are in the (+,+,−) group with q = r ≫ k, where K = k exactly: rel 2.89e-4 (145 rows,
    k = 1e5, q = r = 3e8) and 9.05e-4 (290 rows). By band: 285 seam, 99 mid, 51 low.
  - The second-reading ratio is at most 5.0e-14 over these rows at both rtols.
  - The oracle is unconverged in 302 of them **under both codes**, a condition that predates 06h.
  - **The H4 call's row `3-1072` is one of them** (rel 2.89e-4). The call that lost convergence is
    J1, whose group is (+,+,+) and not the near-singular one.
  - **Over budget:** none at either rtol. No unit's `wall_s` exceeds 300 s.

## Deviations from the prompt (numbered and classified)

1. **`prov.py`:** trees `p` = `…-lpf06h-2c325f5` and `g` = `…-lpf06h-c1ec3b2`; 06e's run, records,
   rows and checkpoint SHA-256 (`RUN06E`, `REC06E`, `ROWS06E`, `CHECKPOINT06E_SHA256`, named as 06f's
   prov.py names them); `WIDER` = `var/levin-phase-floor-06h/census`. STRUCTURALLY REQUIRED.
2. **$h$ keeps the scripts' key `g`,** so units are named `g-<rtol>-<shard>-<serial>`, as in 06e's
   census. IMPLEMENTATION CHOICE.
3. **`harness.py`:** `job_name="levin-phase-floor-06h"` (the brief); stored, never read. STRUCTURALLY
   REQUIRED.
4. **`run_census.py`:** docstring only. STRUCTURALLY REQUIRED.
5. **`launch.py`:** prompt `06h`, purpose, scope, imports. STRUCTURALLY REQUIRED.
6. **`worker.py`, tree evidence:**
   - `LiouvilleGreen.three_bessel_integrals`, `LiouvilleGreen.bessel_phase` and the file QSI's
     `BesselPhaseGroup` comes from must all be in the tree;
   - it records whether 06h's markers are present (`_declared_error_parts`, `_two_product_error`,
     `EXP_RELERR`) and both modules' SHA-256.

   STRUCTURALLY REQUIRED (the brief: evidence of where modules were loaded from).
7. **`worker.py`, `_declared_terms_06h`:** after the timed section, every returned leaf of every call
   supplying `theta_abserr` is re-solved with pass-throughs, and each is checked to reproduce bit for
   bit. The copied harness recorded the declared term on fallback leaves only. Gate H's reproduction
   on 06e's keys certifies that this changes no computed value. IMPLEMENTATION CHOICE in form,
   allowed by the brief.
8. **`gate_h.py`** (from 06e's `z0.py`):
   - the "before" is 06e's census, and 06e's `deep_lines_06e`, `cc_floor_terms` and `cc_floor` are
     compared too;
   - every pickle key 06e recorded is checked to be compared;
   - the determinism unit is `g-1e-7-0-14164`, with a qualification test.

   IMPLEMENTATION CHOICE.
9. **`gate_h_clause1.py`** (commits `6d81fad`, `2c325f5`), `scripts_equal.py` (`state/H2`),
   `check_predictions.py` (06h, lines 334–347) and `checks_after_06h.py`. IMPLEMENTATION CHOICE.
10. **`out/cmd/00` was recorded by an inline equivalent of rec.sh** (rec.sh did not exist before the
    copy). IMPLEMENTATION CHOICE.
11. **A stray `cp -p out/cmd/00_harness_copied.txt /dev/null`** in `00`'s shell failed and wrote
    nothing. UNINTENDED DRIFT.
12. **The Levin branch has no 06e column:** 06e's checkpoint does not separate a Levin leaf's declared
    term from its eq. (151) floor. The Clenshaw–Curtis branch is given under both codes, and the
    Levin branch under $h$, with the constituents' part as a labelled counterfactual. STRUCTURALLY
    REQUIRED.
13. **The launcher's dry import ran at stage 2** (`02`), not at preparation, under the six-core rule.
    IMPLEMENTATION CHOICE.
14. **Scripts are not `black`-formatted;** they are records. IMPLEMENTATION CHOICE.
15. **H3's second reading needed 06e's oracle abserr, which 06e did not record.** The worker
    (`analytic_06h`) records $h$'s oracle abserr and converged. At $h$ it also recomputes 06e's oracle
    after the timed section:
    - `analytic_integral`'s arguments are kept by a pass-through installed with the timed section's
      wrappers;
    - the oracle is re-run with QSI's `BesselPhaseGroup` replaced by the parent's class, from git at
      `2c325f5`, SHA-256 `862a18fe…` checked, executed into an unregistered module.

    A row counts only where the recomputed value equals 06e's `analytic_rad` bit for bit, which held
    on all 18,560 units. Gate H was re-run on this worker. IMPLEMENTATION CHOICE (the orchestrator's
    request; the method is mine).
16. **Gate H ran twice:** `state/H` on the worker before deviation 15, `state/H2` on the final one.
    Both hold, and H2 certifies the run. IMPLEMENTATION CHOICE.
17. **Part B of the predictions adds H3's second reading and H4's split by kind** (the orchestrator's
    request), and fixes that a disagreement between the two H3 readings is a stop. It did not arise.
    IMPLEMENTATION CHOICE.
18. **`out/cmd/11` piped rec.sh's output through `sed`,** so the lock's listing shows that sed, as in
    06e's deviation 18. `29` and `30` do not. UNINTENDED DRIFT.
19. **`launch.py`'s purpose does not mention the oracle counterfactual;** launch.py was not edited
    after Gate H. IMPLEMENTATION CHOICE.
20. **The run's wall time includes the post-timed passes,** the oracle counterfactual chief among
    them. The unit-wall comparison is on `wall_s`. STRUCTURALLY REQUIRED by the instrumentation.
21. **New `analyse06h.py`, `h4_detail06h.py` and `nearsingular06h.py`.** IMPLEMENTATION CHOICE.
22. **The first write of `KEPT_ON_MAC_STUDIO.txt` used BSD `stat -f %z` on this host's GNU `stat`,**
    as 06e's deviation 24, and listed no sizes. It was rewritten with `wc -c`. UNINTENDED DRIFT.
23. **The near-singular rule** ($|K|/(k+qc_s+rc_s) \le 10^{-3}$) is Part B's, not the prompt's: the
    prompt and brief only say "if A3 v2 reaches it". Fixed before the run. IMPLEMENTATION CHOICE.

None of the UNINTENDED DRIFT items (11, 18, 22) touched a store, a tree, or any record outside this
directory and the run directory.

## Verification performed

- **The trees:** $h$ at `c1ec3b2` and $p$ at `2c325f5`, each with empty `status --short --ignored`,
  at preparation, before Gate H, at the launch, on all 18,560 records, and at the end. No
  `__pycache__` was written. The main checkout was at `28ebaf0` with empty status throughout.
- **The copy:** byte-identical to `copy_files_certified.txt` before Gate H, after it, at the launch,
  after the run, after the fingerprint and at the end, with no journal, `-wal` or `-shm` file. It
  gives `0e32f24a…` at $h$ and $p$ before, and at $h$ after.
- **06e's run was only read:** its checkpoint is `a0dc4d7f…` before, after preparation, after the run
  and at the end.
- **The run:**
  - 18,560 records, every unit `done`, every record at $h$ and clean.
  - Every worker reported `ray_initialized` false and no foreign modules.
  - Every Clenshaw–Curtis floor-term leaf, every declared-term leaf and every deep line reproduces.
  - Every oracle counterfactual reproduces 06e's `analytic_rad`.
- **H3 and the second reading** were evaluated exactly (`Fraction`) on the recorded doubles. **H4**
  matched every call by row, rtol and index with equal label, `x_span` and kind; 0 problems.

## Observations not acted on

All of these are in `issues.md`:
- **D1:** H4 fails at 1e-8 on one analytic call. The larger declaration flipped the round-off exit's
  premise at the root of a span whose eq. (151) floor already exceeded the request.
- **D2:** A3 v2 reaches the near-singular class (435 rows by Part B's rule); nothing moved there.
  Its oracles are unconverged in 302 rows under both codes.
- **D3:** on the Clenshaw–Curtis branch the declared term now dominates the floor in about twice as
  many analytic calls (2,493 → 5,137 and 2,785 → 5,939), still at most 0.34 of a request.

## State handed on

- **Nothing of mine is running.** The run is `done`, no census worker process remains, no lock was
  taken, and Ray was never started. Nothing was written in either worktree, or in the main checkout
  outside this directory and the run directory.
- **The scratch copy is unchanged** (`0e32f24a…`).
- **The run directory is kept in full** (2.3 GB) for 06g's census:
  `var/runs/levin-phase-floor-06h-census-20261101T005037/`, checkpoint `3104c18e…`.
- **For the orchestrator:**
  - copy `log.md`, `verification.md`, `issues.md`, `scripts/`, `out/` and `state/` to
    `docs/levin-phase-floor/06h/census/`;
  - leave out the files `KEPT_ON_MAC_STUDIO.txt` lists: the pickles under `state/H/`, `state/H2/` and
    `state/smoke/`, and `state/rows.json` (7.8 MB);
  - H3 holds and H4 fails. Whether `[06f-theta-abserr-omits-the-product-rounding]` closes also
    depends on H1 and H2.
