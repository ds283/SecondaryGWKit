## Part C §6: the census at 07a's code (Mac Studio), the census agent

**Subject:** prompt 07a §6, as amended by §9.5 and §10.4: the census re-run at the code commit `1ec734d7`, against 07's
census as the "before". On the Mac Studio (`mps026298`), 2026-11-12, by Part C's census agent. Covers Gate W0, the
predictions, the census, stop 3's comparison, W1–W7, the descriptive items and the records. Ray-free, no lock, no store
built.
- **The tree h:** the worktree `SecondaryGWKit-lpf07a-4bf50d3`, detached at Part C's dispatch SHA `4bf50d34`, whose code is
  `1ec734d7`'s.
  - `AdaptiveLevin/levin_quadrature.py` `4ea1948b…`; `ComputeTargets/QuadSourceIntegral.py` `33cb161e…`.
  - Clean with `--ignored` throughout. The main checkout stayed at `28ebaf07`, clean.
- **The parent:** there is no parent worktree. W0's K0 run ran in h with the parent's module installed in place of h's by
  `variant_loader.py`.
  - The module is `state/variants/levin_quadrature_parent.py`, from `git show dadd63a0`, `4e5da371…`.
  - It is byte-identical to 07's census tree `e182bc7a`'s.
- **The "before":** 07's census, `levin-phase-floor-07-census-20261108T171137`, checkpoint `03dfe786…`.
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07a/census/`.

**The user's rulings, relayed by the laptop orchestrator:**
- At STOP 2: *"Yes, A for Q1 and call's outputs for Q2; launch the census"*. The orchestrator's reading is recorded in
  `state/ruling_stop2.md`:
  - Q1, reading A: every solve counts as "a region whose route or LU choice differs" in W1 and W7.
  - Q2: "bit for bit" compares the call's outputs. The `location` and `harness` categories are set aside as `phase_span`
    is.
- At STOP 3: *"Yes, resume the census agent for scoring"*.
- At the W5 stop: *"Yes, W5 gated only, and take references for both calls"*. The orchestrator's reading is recorded in
  `state/ruling_w5.md`:
  - W5's counts are made over gated solves only. The forced solves' count is descriptive.
  - `state/predictions.md` part 5's wording ("every solve … forced included") is superseded by this ruling. The file is not
    edited.
  - References are taken, by 07's V4 method, for rows `3-678` and `1-34599`. The ruling overrides §C.1's "no new references
    are taken" for these two calls only. The results are findings, not W verdicts.

**Result:**
- **Gate W0 holds**, all four clauses.
- **The census** `levin-phase-floor-07a-census-20261112T132744` is `done`: 18,560 of 18,560 units, 0 in error, 0 over budget,
  wall 23,284 s (07: 22,708 s). Checkpoint `461df2ec…`.
- **W1 holds** under the ruled readings: 0 of 258,824 untouched calls differ in the call's outputs, and all 9,726 units made
  only of such calls have `total` and `total_abserr` equal to 07's.
- **W2 holds** (0 Levin leaves listed). **W3 holds** (0 violations; the 784 skipped phases are all `zero_rate`).
- **W4 fails, as §9.5 predicted:** exactly v08, v09, v12, v17 and v22, at 3.16, 2.21, 3.08, 2.38 and 2.53. All 33 calls are
  bit-identical to 07's.
- **W5 holds** (gated solves, as ruled): 0 node-partition sums, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded
  regions. Descriptive: 18,236 forced solves took the node-partition sum, 3–10 per call, all within the root-located calls
  (4,416 of the 4,804). Their V routes nothing.
- **W6 fails, as §9.5 predicted:** 2,482 of 2,484 pairs have r_g ≤ 1. The two that do not are `3-14501` (1.39) and
  `3-35006` (1.08), as at 07.
- **W7 fails:** at 1e-7, 2,407 covered calls move by at most 1.80e-7 of abserr_07. At 1e-8, 25 of 2,367 move by
  0.0066–0.145 of abserr_07, beyond the 1e-5 threshold.
  - All 25 are lone-`G` group calls with minima windows (6 or 19 minima).
  - At 07 one window in each stopped (iroff1 6). At h none did (iroff1 4–5), because the value test is now at rtol 1e-8.
  - Their abserr and `converged` are unchanged.
  - Part A's basis (430 calls, all `G+q+r`) held no such call.
- **The store and 07's census are unchanged**, before, after the census and at the end.
- **The user's references** (findings, not verdicts):

  | row | class | h | 07 |
  |---|---|---|---|
  | `1-34599` (`G±q∓r`) | own | **honest**: 0.020 (1e-7) and 0.015 (1e-8) of abserr_h | **not honest**: 4.03× abserr_07 |
  | `3-678` (`G±q∓r`) | aux_only | **honest**: 0.491 of its abserr, 11× smaller than 07's and from 1 region | honest: 0.135 |

  - At `1-34599` the own C2 route change corrected a dishonest call of 07.
  - At `3-678` the shrunken declaration, the product of an auxiliary trust-check route change, is honest.
  - In every case the readings' spread is at least 7.8e5 below both abserrs.

### What shipped

No code. The records are in `var/levin-phase-floor-07a/census/`:
- **The records:**
  - `log.md` (this file), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
  - `state/predictions.md`: SHA-256 `3ec7b53d…`, mtime 2026-11-12T13:16:44.807Z, read-only, written after W0 and before the
    launch;
  - `state/ruling_stop2.md`.
- **`scripts/`:** 07's census harness, copied from h's `docs/levin-phase-floor/07/census/scripts/`. It is byte-identical to
  07's Mac Studio copy (`out/cmd/00`). Changes are marked "prompt 07a". The diffs are `out/scripts_diff_before_k0.txt`,
  `out/scripts_diff_at_stop2.txt` and `out/scripts_diff_at_end.txt`.
  - **Changed:**
    - `prov.py`: the tree h, the parent's module and 07's census;
    - `worker.py`: tree h; section 6's diagnostics `route_07a` with its on/off switch; the post-call tallies;
    - `run_k0.py`: variant `parent`, `--no-instrument`;
    - `run_census.py`: tree h;
    - `refs_common.py`: the ROWS path.
  - **New:**
    - `hash_runs_07a.py`, `ast_lg_check_07a.py`, `index_07_07a.py`, `compare_07a.py`;
    - `select_w0_units_07a.py`, `w0a_imports_07a.py`, `launch_07a.py`, `launch_census_07a.py`, `census_projection_07a.py`;
    - `write_predictions_07a.py`, `stop3_classes_07a.py`, `score_v4_07a.py`, `xcheck_07a.py`, `xcheck_06m_from07_07a.py`;
    - the references: `refs_select_07a.py`, `x1_capture_07a.py`, `check_captures_07a.py`, `refs_integrand_07a.py`,
      `refs_slice_07a.py`, `refs_projection_07a.py`, `score_refs_07a.py`;
    - `score_07a.py`, `summarise_07a.py`, `w7_detail_07a.py`, `moves_beyond_07a.py`, `w5_forced_detail_07a.py`,
      `kept_list_07a.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`), with these results:
  - W0: `out/w0/`, `out/k0/compare_w0b.json`, `out/w0c3/compare_instr.json`, `out/w0c4/compare_det.json`,
    `out/k0/compare_k0_h_calls.json`;
  - stop 3: `out/stop3/compare_census_calls.json`, `.calls.jsonl` and `classes.json`;
  - the scores: `out/scores/summary.json`, `w4_scores.json`, `w4_report.md`, `w7_detail.json`, `moves_beyond.json`,
    `w5_forced.json`;
  - the cross-checks: `out/xcheck/`;
  - the user's references: `out/refs/` (captures, checks, integrands, projection, readings, `scores.json`, `report.md`);
  - before and after: `out/runs_hashed_{before,after,end}.txt`, and `out/census_projection.json`.
- **Three registered runs, all `done`:**
  - `levin-phase-floor-07a-k0-20261112T130135`: 11 of 11 commands, checkpoint `97731e2b…`;
  - `levin-phase-floor-07a-census-20261112T132744`: 18,560 of 18,560, exit 0, checkpoint `461df2ec…`;
  - `levin-phase-floor-07a-references-20261112T203555`: 16 of 16 readings, exit 0.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census's checkpoint and region records in full (6.7 GB, 08's "before");
  - the k0 and references run directories;
  - every pickle and every file over 2 MB under `census/`.

### Predictions and verdicts

`state/predictions.md` holds the prompt's §6 W1–W6 and descriptive items (lines 525–556), §9.5 (716–731) and §10.4's item
(823–825), each verbatim by line range. Then come this agent's descriptive additions (part 4), how each rule is read
(part 5, with Q1 and Q2 left to the user), and V4's 33 calls (part 6). Verdicts are as ruled at STOP 2.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **W0 (1)** the parent's code is 07's (S0′) | **holds** | `git diff --stat e182bc7a dadd63a0` (outside docs and prompts) lists `CLAUDE.md` only. The K0 workers loaded 101 files of h. None is in S0′'s diff. Of the parent→h diff only the two LiouvilleGreen modules were loaded, both AST-equal without docstrings. h's own `levin_quadrature.py` was never loaded (`out/w0/w0a_imports.txt`, `out/w0/w0a_ast_lg.txt`) |
| **W0 (2)** the harness reproduces 07's census at the parent | **holds** | 100 of 100 K0 units equal 07's bit for bit: 2,110 calls, 2,110 region records, 32,956 leaves. Only 07a's four keys are set aside. 0 of 07a's keys in 07's records; `phase_span` 0 times as a key in either (`out/k0/compare_w0b.json`) |
| **W0 (3)** the instrument changes nothing | **holds** | `h-1e-8-1-2174` (a root window) and `h-1e-7-1-11214` (an own solve's C2 route changed; chosen by `select_w0_units_07a.py`'s fixed rule), diagnostics on and off: IDENTICAL (`out/w0c3/compare_instr.json`) |
| **W0 (4)** the new code is deterministic | **holds** | `h-1e-8-1-2174` run twice: identical on every key, 07a's included (`out/w0c4/compare_det.json`) |
| **W1** nothing else moves | **holds** (reading A; the call's outputs) | 1e-7: 129,774 untouched calls; 123,083 strictly equal; 6,691 differ only in `location`; 0 in `call` or `harness`. 1e-8: 129,050; 122,359; 6,691; 0. Rows made only of untouched calls: 4,891 and 4,835, totals equal in all. Strictly: fails (13,382 `location`-only calls). Under reading B: fails (15,847 aux_only calls differ in `call`; totals differ in 217 + 207 rows) |
| **W2** no Levin leaf across a stationary point | **holds** | 182,066 calls scanned (37,325 group at each rtol, 107,416 analytic at 1e-7), 0 scan errors. Listed leaves: 0 Levin; window leaves 1,682 (analytic, 1e-7), 526 (group, each rtol) |
| **W3** every phase with something to locate is located | **holds** | 181,282 scanned calls scored, 0 violations. 784 set aside (392 group calls per rtol, the q = r rate): every one `reason: zero_rate`, no scan sign change, min/max \|θ′\| undefined (rate zero) |
| **W4** honest where referenced | **fails** (as §9.5 predicted) | 28 of 33 honest. v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53, to three figures as predicted. All 33 calls' value and abserr equal 07's in hex (26 window_both, 6 untouched, 1 own: z24). Tolerance min(abserr_07, abserr_h)/100. Every call's readings agree |
| **W5** the gate is consistent; the guard a backstop | **holds** (gated solves, as ruled at the W5 stop) | Gated solves: 0 Clenshaw–Curtis regions with a half sent to Levin (also 0 on 07's estimate); 0 node-partition sums; 0 guarded regions. Descriptive: 18,236 forced solves took the node-partition sum (`out/scores/w5_forced.json`), 3–10 per call, all within the root-located calls (4,416 of the 4,804; 388 root-located group calls, 194 per rtol, have none). In each such call this is the window piece whose end node is the root, and its forced halves and quarters; their V routes nothing. Before the ruling it was undecided, because it fails if forced solves count (part 5's superseded text) |
| **W6** the pairs, on r_g | **fails** (as §9.5 predicted) | 2,482 of 2,484 pairs r_g ≤ 1, 0 unpaired. `3-14501` 1.385 and `3-35006` 1.084, as at 07, all members converged. h's own population: 3,650 pairs, 3,648 ≤ 1, the same two |
| **W7** the stop's change is small | **fails** | Reading A: at 1e-7, 2,407 covered calls, 0 violating, max move 1.80e-7 of abserr_07. At 1e-8, 2,367 covered, 25 violating (0.0066–0.145 of abserr_07), no `converged` change. Reading B: also fails (38 and 97) |

### The user's references (findings, not W verdicts; `out/refs/`)

Method: 07's V4 reference method, ruled at the W5 stop.
- **The tool and readings:** `tools/filon_reference.py` (`d37e633b`, as 07's). QAWO r1 and r2 and FCC r1 and r2 on each call;
  each QAWO reading projected at 30 minutes or less. The reference is FCC r2. 07's reading-usability check passed on all 16.
- **The calls:** the eight flagged calls (row × group calls 2 and 3 × 1e-7 and 1e-8).
- **The captures:** each captured at h, and checked bit for bit against 07a's census (`out/refs/check_captures_h.json`,
  8 of 8). Each captured again in h with the parent's module, and checked against 07's census
  (`out/refs/check_captures_07.json`, 8 of 8).
- **The integrands:** equal by bit pattern at 1001 points between the two captures and across the rtols
  (`out/refs/integrands.json`). So one reference per row and index serves both rtols.
- **The run:** `levin-phase-floor-07a-references-20261112T203555`, `done`, 16 of 16 readings, wall about 13 minutes
  (projected 737 s).
- **The stated error:** the readings' spread.

| call | ref (FCC r2) | stated error | ÷ min abserr | abserr_h | abserr_07 | \|v_h − ref\|/abserr_h | \|v_07 − ref\|/abserr_07 | regions h / 07 |
|---|---|---|---|---|---|---|---|---|
| `1-34599` #2, #3, 1e-7 (own) | −3.162788408760542e-26 | 5.7e-40 | 1.3e-6 | 1.524e-33 | 4.462e-34 | **0.0201** | **4.03** | 16 / 6 |
| `1-34599` #2, #3, 1e-8 (own) | (the same) | 5.7e-40 | 1.3e-6 | 1.118e-33 | 4.462e-34 | **0.0150** | **4.03** | 17 / 6 |
| `3-678` #2, #3, 1e-7 and 1e-8 (aux_only) | 1.431843199210653e-30 | 1.0e-45 | 9.2e-8 | 1.088e-38 | 1.214e-37 | **0.491** | 0.135 | 1 / 12 |

- Every call's stated error is at least 10× below both abserrs, so every call is scored. The readings also agree within
  07's own tolerance (min/100).
- Calls #2 (G+q−r) and #3 (G−q+r) give the same values, since q = r.
- `1-34599`: h is honest, and 07 was 4.03× its abserr from the reference.
- `3-678`: h's eleven-times-smaller declaration is honest, at 0.491 of it.

### Descriptive (not scored)

- **Route changes:**
  - 24,559 solves changed decision, of 22,945,838 recorded:
    - 10,014 C2 route changes, every one towards Clenshaw–Curtis (V < 6π ≤ estimate);
    - 14,545 LU changes, every one towards least squares (V ≤ 20π < estimate).
  - By class:

    | class | to CC | to least squares |
    |---|---|---|
    | own | 3,007 | 4,689 |
    | comparison half | 3,864 | 5,251 |
    | trust-check quarter | 3,143 | 4,605 |

  - Estimate/V on the changed solves: 1.00001–2.29 (median 1.064) for C2 changes, 1.00001–2.39 (median 1.150) for LU
    changes. 07's estimate was always above V on a changed solve.
  - By rtol and kind: analytic 3,402 + 5,776 (1e-7) and 3,695 + 6,224 (1e-8); group 1,449 + 1,260 and 1,468 + 1,285.
  - Every band and k is touched (`out/scores/summary.json`).
  - Calls: own 1,380 analytic + 1,906 group (1e-7) and 2,186 + 2,039 (1e-8); aux_only 7,371 + 660 and 7,251 + 565.
- **Moves:** 3,091 units' `total` or `total_abserr` differ from 07's.
  - Relative: at most 3.3e-4 (90th percentile 2.3e-7).
  - Against total_abserr_07: at most 0.749, 0 above 1.
  - No `total_converged` and no call's `converged` changed.
  - **Calls moved beyond an abserr** (`out/scores/moves_beyond.json`):
    - **beyond both codes' abserrs:** `1-34599` `G+q-r`/`G-q+r` at 1e-7 (3.96× abserr_07, 1.16× abserr_h) and 1e-8 (3.99×,
      1.59×), class own (one own C2 route change; regions 16/17 against 6);
    - **beyond abserr_07 only:** `1-2179` `G±q∓r` (4.66× abserr_07, 0.34× abserr_h; abserr_h 13.6× abserr_07), class own;
      `3-843` analytic J2 (1.09×, 0.58×) and `3-13508` `G±q∓r` (1.08×, 0.60×), class aux_only.
- **The aux_only calls** (reading A sets them outside W1): 15,847 calls (`out/scores/summary.json`, `aux_only`).
  - The changed solves are comparison halves in 8,878 calls and quarters in 7,500.
  - abserr equal in 1,609, smaller at h in 6,697, larger in 7,541.
    - The ratio h/07 runs 0.090–5.10, median 1.0, 99th percentile 1.28.
    - It shrank by more than 2× in **47** calls and grew by more than 2× in 70.
  - Value: unchanged in 8,792. The move over abserr_07 is at most 1.09, 90th percentile 3.3e-5.
  - 4 calls moved by more than their own abserr at h: `3-678` `G+q-r` and `G-q+r`, at 1e-7 and at 1e-8.
    - Their abserr shrank to 0.0897 of 07's and their regions went 12 → 1.
    - Their value moved by 2.00× abserr_h (0.179× abserr_07).
    - These are among the 47.
  - No `converged` change; regions changed in 432 calls.
  - **Referenced aux_only calls: none.** V4's 33 calls are window_both (26), untouched (6) and own (1). The cross-check's
    2,484 pairs are all calls with a window. So no shrinking declaration among them is tested against a reference.
- **The newly located:** 2,360 calls (window_h_only; at each rtol 1,012 group plus 84 J4 and 84 Y4).
  - Points: 2,024 minima and 336 roots. Pieces 2–59, median 14.
  - All converged at both codes.
  - 07's Rule A variation on them was 0.009–18.58.
  - No call locates more than 64 points (the most is 64 pieces, in a call of the old population).
- **Windows:**
  - 3,650 windows per rtol at h (07: 2,470).
  - Stopped: 1,247 at 1e-7, 1,243 with a leaf still falling (07: 1,157/1,153); 1,236 at 1e-8, 1,233 falling (07:
    1,195/1,192).
  - Calls with a stopped window left unconverged: 7 (1e-7) and 9 (1e-8).
- **Convergence and the flag:** unconverged group calls at h equal 07's in every class.
  - 1e-7: 30 in the 720, 52 outside. 1e-8: 34 and 62.
  - No analytic call is unconverged.
  - Row depth 20: 77 (1e-7) and 83 (1e-8) at both codes.
- **The disagreement list:**
  - h against 07: 0 listed calls, on 07's population and on h's.
  - h against 06m (06m's values from 07's cross-check output; 06m's checkpoint was trimmed on 2026-11-10): 843 listed calls,
    class (c) 786, (a) 31, (e) 26, all `G+q+r`, each in the same class as at 07.
- **§10.4's item:** **0** calls go from converged to unconverged with a solve whose route changed and no window, under
  either reading.
- **The largest |θ| at a gated region's end:**
  - analytic: 8.874e12 (ulp 1.95e-3 rad), `h-*-2-35099` J1, both rtols;
  - group: 4.119e12 (ulp 4.9e-4 rad), the same row's `G+q+r`.
- **The smallest margins among unchanged solves:**
  - |V − 6π|: 4.77e-8 (group, a trust-check quarter, `*-1-14543`, V 18.8495560, estimate 18.8496054); analytic 1.38e-3
    (1e-7) and 3.10e-4 (1e-8).
  - |V − 20π| on the Levin route: 3.02e-4 (group, `*-3-34129`, an own solve); analytic 4.53e-3 and 2.94e-3.
- **Cost:**
  - Census wall 23,284 s against 22,708 s.
  - Unit walls 24,952 / 28,513 s against 24,203 / 27,479 s.
  - Group-call regions 952,395 / 1,065,828 against 952,275 / 1,065,288 (1e-7 / 1e-8).
  - The harness's own re-solves after the timed section: 06h's counterfactual made 6,906,084 solves, 7,097 with a C2 change
    and 11,999 with an LU change (`route_07a_post_unit`). They are not the call's.

### Deviations from the prompt

1. **The harness changes,** each within §6's list (`out/scripts_diff_at_end.txt`). STRUCTURALLY REQUIRED.
   - The tree and its paths (`prov.py`, `refs_common.py`, `run_census.py`, `run_k0.py`).
   - The diagnostics (`worker.py`'s `route_07a` and its on/off switch).
   - The comparison with 07's census (`compare_07a.py`).
   - V4's re-scoring (`score_v4_07a.py`).
2. **The parent ran as a variant from `state/variants/` in h**, not as a parent worktree, as the dispatch directs.
   STRUCTURALLY REQUIRED.
3. **06g's `rows.json` is no longer on this host.** 07's copy (same SHA-256) is read. The first copy command failed
   (`out/cmd/07`). STRUCTURALLY REQUIRED; the failed attempt is UNINTENDED DRIFT.
4. **New scripts** (listed under What shipped). IMPLEMENTATION CHOICE.
5. **Development smoke tests** (`state/smoke/`, `out/cmd/11`–`17`), not scored. IMPLEMENTATION CHOICE.
6. **K0 at h, its call comparison, the unit selection and W0's clauses 3 and 4 ran inside the `k0` run**, as 07 did.
   IMPLEMENTATION CHOICE.
7. **W0 (3)'s units:** z12's row fixed by name; the route unit chosen by a rule fixed before the run. IMPLEMENTATION CHOICE.
8. **The diagnostics also tally the harness's own re-solves after the timed section** (`route_07a_post`). IMPLEMENTATION
   CHOICE.
9. **The comparison puts each differing path in one category** (`call`, `location`, `harness`). It was put to the user as
   Q2 and ruled. IMPLEMENTATION CHOICE.
10. **`predictions.md` gained parts 4 and 5.** Part 6's column labelled "30 digits" holds 07's stored reading strings (17
    digits). The file is fixed and was not rewritten. The label is UNINTENDED DRIFT (cosmetic).
11. **The first edit of `prov.py` ran through the system `python3`** (an editing script; no repository code). UNINTENDED
    DRIFT (cosmetic).
12. **Redundant background waits on the k0 driver** were started and stopped. UNINTENDED DRIFT (cosmetic).
13. **No lock taken, and no lock `status` or `ray-processes` run**, as the dispatch says. IMPLEMENTATION CHOICE.
14. **The A3 v2 fingerprint was not recomputed;** the copy is checked by SHA-256 against its certification, as 07 did.
    IMPLEMENTATION CHOICE.
15. **The stop-3 comparison and every scoring script ran in the session through `rec.sh`**, not as registered runs. The
    dispatch names only the `k0` and `census` slugs. The longest took 103 s. IMPLEMENTATION CHOICE.
16. **`score_v4_07a.py`'s first run failed on a broken docstring** (`out/cmd/50`); it was fixed and re-run (`50b`).
    UNINTENDED DRIFT (a failed attempt).
17. **A first loop of the three cross-checks did not run** (zsh does not word-split `set -- $args`; `rec.sh` could not open
    its output). It was re-run under `bash`. Nothing was written but the shell's errors. UNINTENDED DRIFT (a failed attempt).
18. **The disagreement list against 06m used 07's cross-check output for 06m's values,** because 06m's census checkpoint was
    trimmed on 2026-11-10. The first attempt, against 06m's checkpoint, failed (`out/cmd/51_xcheck_h_vs_06m_pop07`).
    STRUCTURALLY REQUIRED.
19. **`xcheck_06m_from07_07a.py`'s output carries a vestigial empty key** `class_h_vs_07`. The real table is
    `class_07_to_h`. UNINTENDED DRIFT (cosmetic).
23. **References were taken for rows `3-678` and `1-34599`,** on the user's word at the W5 stop. This overrides §C.1's "no new
    references are taken" for these two calls only. STRUCTURALLY REQUIRED (by the ruling).
24. **The reference harness is 07's V4 method, copied with changes marked "prompt 07a"** (`out/scripts_diff_at_end.txt`).
    IMPLEMENTATION CHOICE.
    - `x1_capture_07a.py`: tree h, the parent variant, the match to 07's record.
    - `check_captures_07a.py`: the census named on the command line; `route_07a*` set aside.
    - `refs_slice_07a.py`: the 1e-7 captures only.
    - `launch_07a.py`'s `references` plan.
    - `one_reading.py` and `run_06j.py` are unchanged.
    - New: `refs_select_07a.py`, `refs_integrand_07a.py`, `score_refs_07a.py`.
25. **The projection was refitted on 07's own 74 readings,** because 07's calibration (06g's readings) is gone from this host.
    07's feature function and model form are unchanged (`refs_projection_07a.py`). STRUCTURALLY REQUIRED.
26. **One reference per row and index serves both rtols,** on the integrands' shown identity, as 07's cross-check did.
    IMPLEMENTATION CHOICE.
27. **The parent captures (`out/refs/captures_parent.json`) were made so that each call could be checked against 07's census.**
    IMPLEMENTATION CHOICE.
28. **The stated error of a reference is the readings' spread** (07's agreement measure). IMPLEMENTATION CHOICE.
29. **Two failed attempts, each re-run.** UNINTENDED DRIFT (failed attempts).
    - The first captures (`out/cmd/71`, `72`) could not write their JSON because `out/refs/` did not exist; re-run as `71b` and
      `72b` into new directories.
    - A zsh `$H:s…` substitution error stopped one command; re-run with the literal path.
20. **W4's tolerance** is min(abserr_07, abserr_h)/100: 07's rule with "the before" now 07 (part 5). 07's own scores are
    shown beside it. IMPLEMENTATION CHOICE.
21. **W3's "scored" count** is every scanned call less the 784 set aside. `score_07a.py` keeps in detail only calls with a
    sign change, a point or a listed leaf; the others satisfy V3 vacuously. IMPLEMENTATION CHOICE.
22. **`kept_list_07a.py` was made from 07's by `sed`.** A substitution miswrote its own source's name in its docstring; this
    was corrected before it ran. UNINTENDED DRIFT (cosmetic).

### Verification performed

`verification.md` has the account. In brief:
- **The trees:** h at `4bf50d34`, clean with `--ignored` at every check and at the end, with no `__pycache__`. The main
  checkout is at `28ebaf07`, clean.
- **Every census worker:** module `4ea1948b`, no variant, h clean at its SHA, no Ray, no foreign module.
- **The scratch copy:** EQUAL to its certification before, after the census and at the end. Before equals end.
- **07's census:** 150 of 150 files equal its kept list. Its 850 working-record files are byte-identical before, after and at
  the end.
- **The registry:** 76 runs, all finished. No lock was taken.

### Observations not acted on

1. **W7's 25 calls:** at 1e-8, minima windows of lone-`G` calls that stopped at 07 no longer stop under 07a's value test
   (min(1e-5, rtol)). Values move by up to 0.145 of their abserr, abserr and `converged` unchanged. §9.5's basis (Part A's
   430 `G+q+r` calls) did not sample them.
2. **W5's forced partition sums** (descriptive under the ruling): in a root-located call, the window piece whose end node is
   the root takes the node-partition sum.
   - θ′ changes sign at that node.
   - So do its forced halves and quarters there.
   - V routes nothing on a forced solve, but it is the `phase_span` reported for that region.
3. **Shrinking declarations on auxiliary-only calls:** 47 aux_only calls' abserr shrank by more than 2×.
   - The one the user referenced, `3-678`, is honest at 0.491 of its eleven-times-smaller abserr.
   - The other 46 are unreferenced.
4. **`1-34599`, where 07 and h disagreed beyond both errors:** the reference shows h honest (0.020 and 0.015) and 07 not
   (4.03). An own C2 route change corrected it.
5. **06m's census checkpoint was trimmed on 2026-11-10,** and 06g's census records are gone from this host (`issues.md`).
6. **§10.4's set is empty** in the census: no call loses convergence with a route change and no window.

### State handed on

- **Nothing is running.** All three registered runs are `done`: k0, census and references. No lock was taken.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (149 files, 6.7 GB; 08's "before");
  - the k0 and references run directories;
  - every pickle and every file over 2 MB under `census/` (the K0, W0 and smoke runs' region pickles; the stop-3 and
    scoring outputs).
- **Read only and unchanged:** the scratch copy and 07's census, by hash.
- **Issues:** proposed in `issues.md`, for the laptop and the user.
