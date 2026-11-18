## Part C §6: the census at 07b's code (Mac Studio), the census agent

**Subject:** prompt 07b §6, as amended by §9 (§9.4, §9.5, §9.7). The census was re-run at the code commit `5e19c4b6` (variant
(h), the rate rules), with 07a's census as the "before". It ran on the Mac Studio (`mps026298`), 2026-11-16 to 2026-11-18,
by Part C's census agent. This log covers Gate U0, the predictions, the projection, the census, stop 3's two checks
(U1's untouched calls and U8), U1–U7, the descriptive items and the records. Ray-free, no lock, no store built.
- **The tree h:** the worktree `SecondaryGWKit-lpf07b-0a103b5`, detached at Part C's dispatch SHA `0a103b5a`.
  - Its code is `5e19c4b6`'s: `AdaptiveLevin/levin_quadrature.py` `8956c7b6…`.
  - It stayed clean with `--ignored` throughout, with no `__pycache__`. The main checkout stayed at `28ebaf07`, clean.
- **The parent:** there is no parent worktree. U0's K0 run ran in h with the parent's module installed in place of h's by
  `variant_loader.py`: `state/variants/levin_quadrature_parent.py`, from `git show 7b82a0cd`, `4ea1948b…`, blob `0701d2a6`,
  the same as 07a's code commit `1ec734d7` and 07a's census tree `4bf50d34`.
- **The "before":** 07a's census, `levin-phase-floor-07a-census-20261112T132744`, checkpoint `461df2ec…`.
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07b/census/`.

**The user's rulings, relayed by the orchestrator:**
- At stop 2: *"Yes, launch the census"*.
- At stop 3: *"Yes, score U1–U7"*.

**Result:**
- **Gate U0 holds**, all four clauses.
- **The projection** was 37,616 s central and 65,989 s at its widest margin, within §9.4's 48 hours.
- **The census** is `levin-phase-floor-07b-census-20261117T010200`, `done`:
  - 18,560 of 18,560 units, 0 in error, 0 over budget;
  - wall 45,016 s (12.5 h; 07a: 23,284 s), checkpoint `396784bc…`.
- **Stop 3's two checks pass:**
  - every one of the 60,160 untouched calls equals 07a's in U1's five outputs;
  - all 187 U8 calls equal (h)'s replay records.
- **U1–U8 all hold.** U4, U6 and U7 hold as §9.5 predicted.
- **The store, 07a's census and Part A's records are unchanged**, before, after the census and at the end.

### What shipped

No code. The records are in `var/levin-phase-floor-07b/census/`.
- **The records:**
  - `log.md` (this file), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`;
  - `state/predictions.md`: SHA-256 `0c6055ac…`, mtime 2026-11-16T21:03:31.278528Z, read-only. It was written after U0 and before
    the launch. Its prompt blocks are copied by text and checked by substring.
- **`scripts/`:** 07a's census harness, copied from h's `docs/levin-phase-floor/07a/census/scripts/`. It is byte-identical to
  07a's Mac Studio copy (`out/cmd/00`). Changes are marked "prompt 07b". The diffs are `out/scripts_diff_before_k0.txt`,
  `_at_stop2.txt`, `_before_census.txt` and `_at_end.txt`.
  - **Changed:**
    - `prov.py`: the tree h, the parent's module and 07a's census;
    - `worker.py`: the two counts on every call; the 07b tree evidence; the rate rules' instrument (`rate_rule_07b`) and
      its switch;
    - `run_k0.py`: the parent variant 07b's; `--no-instrument` turns off both instruments;
    - `run_census.py`: the backstop stops at a projected 48 h, not a day.
  - **New:**
    - `ast_check_07b.py`, `u0a_imports_07b.py`, `index_07a_07b.py`, `compare_07b.py`, `select_u0_units_07b.py`;
    - `launch_07b.py`, `launch_census_07b.py`, `census_projection_07b.py`, `write_predictions_07b.py`, `hash_runs_07b.py`;
    - `score_u8_07b.py`, `stop3_cost_07b.py`, `score_v4_07b.py`, `score_refs_07b.py`, `xcheck_07b.py`, `score_07b.py`;
    - `summarise_07b.py`, `kept_list_07b.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`), with these results:
  - U0: `out/u0/u0a_ast.txt`, `out/u0/u0a_imports.txt`, `out/k0/compare_u0b.json`, `out/u0c3/compare_instr.json`,
    `out/u0c4/compare_det.json`, and `out/k0/compare_k0_h_calls.json` (descriptive);
  - the projection: `out/census_projection.json`;
  - stop 3: `out/stop3/compare_census_calls.json` (and `.calls.jsonl`), `out/stop3/u8.json`, `out/stop3/cost.json`;
  - the scores: `out/scores/summary.json`, `u4_v4_scores.json`, `u4_v4_report.md`, `u4_u6_scores.json`, and
    `out/xcheck/u7_vs_07a_pop07/xcheck.json`;
  - before and after: `out/runs_hashed_{before,after,end}.txt`, `out/cmd/04b`, `41`–`43b`, `60`–`64`.
- **Two registered runs, both `done`:**
  - `levin-phase-floor-07b-k0-20261116T204402`: 11 of 11 commands, checkpoint `3f2f055f…`;
  - `levin-phase-floor-07b-census-20261117T010200`: 18,560 of 18,560 units, exit 0, checkpoint `396784bc…`.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (11 GB). It is **prompt 08's "before"**;
  - the k0 run directory;
  - every pickle and every file over 2 MB under `census/`.

### Predictions and verdicts

`state/predictions.md` holds three parts of the prompt, each copied verbatim **by text** and checked as a substring of the
prompt:
- §6's U1–U7 and descriptive items (prompt lines 607–622);
- §9.5's additions, U8 among them (725–748);
- §9.7's stops (765–769).

Part 4 is this agent's reading of each rule, fixed before the run. Verdicts are by that file, unchanged.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **U0 (1)** the parent's code is 07a's (S0′) | **holds** | `git diff --stat 1ec734d7 7b82a0cd` lists `CLAUDE.md` and 89c368f2's four files, each AST-equal without docstrings. The K0 workers at the parent loaded 101 files of h. Of 07a's census tree → h diff, only the two LiouvilleGreen modules were loaded, both AST-equal. h's own `levin_quadrature.py` was not loaded. The variant was the parent's `4ea1948b` (`out/u0/`) |
| **U0 (2)** the harness reproduces 07a's census at the parent | **holds** | 100 of 100 K0 units are bit-identical to 07a's census: 2,110 calls, 32,850 returned regions. Only 07b's four new keys were set aside, 0 of them in 07a's records (`out/k0/compare_u0b.json`) |
| **U0 (3)** the instrument changes nothing | **holds** | `h-1e-8-3-14501` (the fix acts: edge 2 and range 3 on call #1, range 2–7 on the others) and `h-1e-7-1-20582` (11 calls, all counts 0; the one fully untouched K0 unit) are IDENTICAL with the instruments on and off. The instrument's refusals equal the code's counts on every call (`out/u0c3/compare_instr.json`) |
| **U0 (4)** the new code is deterministic | **holds** | `h-1e-8-3-14501`, run twice in fresh workers, is identical on every key (`out/u0c4/compare_det.json`) |
| §9.4 the projection within 48 h | **holds** | Central 37,616 s (schedule model, ρ by rtol and the 720). Margins up to 65,989 s. No unit was projected over budget (`out/census_projection.json`). Measured: 45,016 s |
| **U1** nothing else moves | **holds** | 60,160 untouched calls (sum of counts 0): 29,758 at 1e-7 and 30,402 at 1e-8. All equal 07a's in value, abserr, `converged`, `num_regions` and max depth (hex), and on the whole call record. The 2,844 + 2,844 rows untouched in every call equal 07a's in `total` and `total_abserr`. 0 calls lack the counts. Coverage 20.8% of calls, about a fifth as §9.5 expected |
| **U2** no Levin leaf across a stationary point | **holds** | 0 Levin leaves listed. Window leaves listed: 1,682 (analytic, 1e-7), 526 (group, each rtol) |
| **U3** every phase with something to locate is located | **holds** | 23,998 calls scored, 0 violations. 784 set aside (392 group calls per rtol), every one `zero_rate` with no sign change |
| **U4** honest where referenced | **holds** (as §9.5 predicted) | 190 scorings, 0 converged and dishonest, 181 honest, 3 unconverged, 6 reported and not scored (stated error above min(abserr)/100; ratio ≤ 0.035 on each). The 190 are: V4's 33 on 07's readings; Part A's R1 60, R2 60, R3 (W7 6, aux 12, own 6, pairs 4) and item 3, on Part A's references; 07a's rows 1-34599 and 3-678 (8 calls) on 07a's references. **Unconverged: exactly v04, R2-28 and R2-28~o**, R2-28 at **1.1071** and 1.1071 of its abserr, as predicted. Largest honest ratio 0.805 |
| **U5** the gate is consistent; the guard a backstop (gated solves, 07a's ruling) | **holds** | 0 node-partition sums on gated solves, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded regions. Descriptive: 18,238 forced solves took the node-partition sum (07a: 18,236) |
| **U6** 06b's 24 honest or unconverged | **holds** (as predicted) | 24 of 24 honest; ratios 1.8×10⁻⁵–0.83, all converged and all scored. The four dishonest at 07a (`0-12977` #5, `1-2174` #2, `2-14345` #3, `2-34473` #3) are honest |
| **U7** the pairs | **holds** (as predicted) | 2,484 of 2,484 pairs have r_g ≤ 1, 0 unpaired; largest 0.467. `3-14501` 0.00046 (07a 1.385); `3-35006` 0.00057 (07a 1.084). 14 pairs have an unconverged member |
| **U8** the code is (h) | **holds** | 187 of 187 equal (h)'s replay record in all five outputs. Each record also equals its summary row. They are 185 distinct census calls: `R1-x019-8`/`v12` and `R1-x317-7`/`v07` are each the same call |

### Descriptive (not scored)

- **Where the fix acts:** 229,322 of 289,482 calls (79.2%).

  | rtol, kind | calls | acted | edge (calls / refusals) | range (calls / refusals) |
  |---|---|---|---|---|
  | 1e-7 analytic | 107,416 | 93,932 | 0 / 0 | 93,932 / 351,789 |
  | 1e-7 group | 37,325 | 21,051 | 340 / 438 | 21,049 / 53,794 |
  | 1e-8 analytic | 107,416 | 93,665 | 0 / 0 | 93,665 / 323,256 |
  | 1e-8 group | 37,325 | 20,674 | 334 / 430 | 20,667 / 53,162 |

  - By band: seam 68% of calls acted, mid 91%, low 92%. Every k acts at 78–83%.
  - By 07a's class: untouched 80%, aux_only 97%, own 54%, window_both 47%, window_h_only (the minima windows 07a newly
    located) 0%.
  - Tables by rtol × kind × band × k: `out/scores/summary.json` `where_the_fix_acts`.
- **The edge rule:**
  - 868 refusals in 674 calls, every one a `G+q+r` call with a root window, all in the 720. 636 refusals are at depth 0
    (the seed leaf beside the window). Median q 656, range 2.0006–5464.
  - It was evaluated 2,802 times on analytic-call windows and refused none.
  - It never acted on a minima window.
- **The range rule:**
  - 782,001 refusals: 675,045 on analytic calls, 106,956 on group calls.
  - Only at depths 0–3: 62,145, 184,209, 302,056 and 233,591.
  - q from just above 10 to above 10⁹.
  - The instrument recorded every evaluation: 0 unrecorded, and in every call the instrument's refusals equal the counts.
- **`converged`:**
  - **198 group calls newly unconverged** (97 at 1e-7, 101 at 1e-8; 4 with an edge refusal).
    - In 192 of them every depth-20 leaf (668 leaves) is a Clenshaw–Curtis leaf descending from a leaf the rate rules
      refused. This is `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`.
    - The other 6 have no leaf at depth 20 and no untrusted leaf. Their abserr is 1.001–1.33× 07a's (the budget not met).
  - 5 group calls (1e-8) became converged.
  - No analytic call is unconverged under either code.
  - Rows: 139 `total_converged` true → false, 4 false → true.
  - Unconverged group calls: 07a 82 + 96, 07b 179 + 192 (1e-7 + 1e-8).
  - Row depth 20: 182 / 185 rows (07a 77 / 83).
- **abserr against 07a's:**
  - Untouched calls: equal.
  - Acted calls: median 0.0010 (1e-7 analytic), 0.0081 (1e-8 analytic), 0.40 / 0.51 (group). It grew in about 4% of
    acted analytic calls and about 20% of acted group calls.
  - 738 acted calls' abserr grew by more than 100×, up to 4.6×10⁵. These are analytic calls of one region at 07a, two at
    07b.
- **Values:**
  - 2,672 group calls moved by more than 07a's abserr. **605 moved beyond both codes' abserrs**; 603 of them are converged
    under both codes. No analytic call moved beyond 07a's abserr. None of the 605 is referenced.
  - Row totals: 11,235 rows differ; the move over 07a's `total_abserr` has median 6.6×10⁻⁴ and maximum 1.065, 2 rows above 1.
  - 07b against 07a on 07's 2,484 pairs: 220 calls listed, all class (c).
- **Windows:** stopped 1,247 (1e-7) and 1,236 (1e-8), as at 07a.
- **Cost:**

  | | 07b | 07a | ratio |
  |---|---|---|---|
  | census wall | 45,016 s | 23,284 s | 1.93× |
  | unit wall, 1e-7 | 52,154 s | 24,952 s | 2.09× |
  | unit wall, 1e-8 | 54,667 s | 28,513 s | 1.92× |
  | unit time with post sections, 1e-7 | 95,096 s | 50,696 s | 1.88× |
  | unit time with post sections, 1e-8 | 77,572 s | 39,379 s | 1.97× |
  | regions, all calls | 6,039,639 | 2,909,134 | 2.08× |
  | regions, 1e-7 analytic | | | 2.64× |
  | regions, 1e-7 group | | | 2.00× |
  | regions, 1e-8 analytic | | | 2.22× |
  | regions, 1e-8 group | | | 1.86× |
  | largest unit wall | 40.1 s | 38.4 s | |

  The wall came in 1.20× the central projection, inside its margin. 0 units were over budget.

### Deviations from the prompt

1. **The harness changes** (`prov.py`, `worker.py`, `run_k0.py`, `run_census.py`; `out/scripts_diff_at_end.txt`).
   STRUCTURALLY REQUIRED for the tree, the parent and the two counts. Item 16 covers the backstop.
2. **The parent ran as a variant from `state/variants/` in h**, as the dispatch directs. STRUCTURALLY REQUIRED.
3. **New scripts** (listed under What shipped). IMPLEMENTATION CHOICE.
4. **U0 (3)/(4)'s units:** `h-1e-8-3-14501` was fixed by name (Part A's `p14501_8` and `n0`). The quiet unit was chosen
   from K0 at h by a rule fixed before the run: every call untouched, then the most calls. Only one K0 unit qualified.
   IMPLEMENTATION CHOICE.
5. **The rate rules' instrument** records every evaluation, not only refusals. It reads a leaf's depth from the calling
   frame's `leaf`. IMPLEMENTATION CHOICE.
6. **`--no-instrument` turns off 07a's and 07b's instruments together.** Mode `instr` sets aside both instruments' keys but
   not the two counts. IMPLEMENTATION CHOICE.
7. **`compare_07b.py` adds an `oracle` category** (`analytic_rad`, `analytic_06h`) for the unit's LiouvilleGreen Levin
   calls, which are not recorded calls. IMPLEMENTATION CHOICE.
8. **The projection** adds a schedule simulation, calibrated on 07a's own wall, and stated margins to 07a's model.
   IMPLEMENTATION CHOICE.
9. **`predictions.md` adds §9.7's block and parts 4–5.** IMPLEMENTATION CHOICE.
10. **Smoke tests ran in the session** (`state/smoke/`, `out/cmd/11`–`17`), not scored. IMPLEMENTATION CHOICE.
11. **Part A's whole `measurement/` was hashed** (7,645 files) in place of "the captures you use". U8 reads Part A's replay
    records, and no Part A capture is used. IMPLEMENTATION CHOICE.
12. **The registry has no job-name field.** The run ids carry the prefix `levin-phase-floor-07b`. STRUCTURALLY REQUIRED.
13. **The A3 v2 fingerprint was not recomputed.** The copy is checked by SHA-256 against its certification, as 07a did.
    IMPLEMENTATION CHOICE.
14. **Failed attempts, each re-run.** UNINTENDED DRIFT (failed attempts).
    - zsh did not word-split `$P` (`out/cmd/04`, `05`; re-run as `04b`, `05b`).
    - Two commands written with `$T:scripts` never ran, because zsh reads `:s` as a modifier. They left no record.
    - The first smoke instr comparison set aside the wrong keys (`15`, re-run as `15b`).
    - `compare_07b` raised a KeyError on a leftover reading key (`16`, re-run as `16b`).
    - `score_v4_07b.py` failed because `out/scores/` did not exist (`50`, re-run as `50b`).
15. **Cosmetic:** a GNU `stat` call failed, and `runregistrykit list | head` raised BrokenPipe in `out/cmd/27`. UNINTENDED
    DRIFT (cosmetic).
16. **`run_census.py`'s backstop stops at a projected 48 h** (§9.4) rather than 06k's day. It never acted; the largest
    projection it reported was 44,854 s. IMPLEMENTATION CHOICE.
17. **`launch_census_07b.py`**, 07a's launcher with the 07b module check, prompt and purpose. IMPLEMENTATION CHOICE.
18. **U4's sets.**
    - "07a's census's other eight V4 calls" is read as z00 and Z3's seven, scored with z12 on 06g's readings
      (`score_v4_07b.py`, 07a's W4 method).
    - Part A's 06b W2 calls are scored under U6, not U4.
    - `i3-34599-h` (Part A's item 3) and 07a's `r1-34599-1e-7-2` are the same census call, scored once under each source.
    - IMPLEMENTATION CHOICE.
19. **U4's "scored" rule** is the reference's stated error ≤ min(abserr_07b, abserr_07a)/100, 07a's W4 form. Six calls are
    therefore reported, not scored; Part A's (h) column leaves the same six unscored. IMPLEMENTATION CHOICE.
20. **"Class" in the descriptive tables** is read as 07a's W1 class of the call, beside its kind, a window at 07b, and the
    720. IMPLEMENTATION CHOICE.
21. **`score_07b.py` reads every unit's pickle** (07a's read them only when needed), to aggregate the rate rules'
    evaluations. IMPLEMENTATION CHOICE.
22. **U7's population is 07's 2,484 pairs** (07's census marks), with 07a's census as the before, as 07a scored W6.
    IMPLEMENTATION CHOICE.
23. **Stop 3 and the scoring ran in the session through `rec.sh`**, not as registered runs. The dispatch names only `k0`
    and `census`. The longest took about 3 minutes. IMPLEMENTATION CHOICE.

### Verification performed

`verification.md` has the account. In brief:
- **The trees:** h is at `0a103b5a`, clean with `--ignored` at every check and at the end. The main checkout is at
  `28ebaf07`, clean.
- **Every census worker:** module `8956c7b6` from the tree, no variant, h clean at its SHA, no Ray, no foreign module
  (18,560 of 18,560 units' evidence; 16 workers).
- **The scratch copy:** EQUAL to its certification and to its before-hash, after the census and at the end. `shasum -c`
  against 06e's after-hashes is OK. No `-wal` or `-shm` files.
- **07a's census and Part A's records:** 07a's census is 149 of 149 files equal to 07a's KEPT list; Part A's 7,645 files
  and 07a's 653 working-record files are byte-identical before, after and at the end.
- **The registry:** 86 runs, all finished. No lock was taken.

### Observations not acted on

1. **The chase, measured in the census.** 192 of the 198 newly unconverged group calls end with every depth-20 leaf
   (668 in all) a Clenshaw–Curtis leaf below a leaf the rate rules refused. 139 rows lose `total_converged`.
2. **605 group calls moved beyond both codes' abserrs**, 603 of them converged under both codes.
   - Examples: `h-1e-7-0-12710` `G-q-r` (63.6× 07a's abserr, 1 region → 52) and `h-1e-8-1-34595` `G-q+r`.
   - Under one of the two codes each such call is dishonest. None is referenced. U4's referenced calls, which include
     07a's dishonest R1 rows, are all honest at 07b.
3. **abserr grew by more than 100× in 738 acted calls**, up to 4.6×10⁵. They are analytic calls with one region at 07a and
   two at 07b. Whether 07a's one-region declaration was dishonest is not measured.
4. **The edge rule's reach:**
   - It acts only on `G+q+r` root windows.
   - It was evaluated 2,802 times on analytic windows and refused none.
   - It never acted on a minima window: 07a's 2,360 newly located minima-window calls are untouched.
   - This matches §9.3's "does not reach".
5. **The range rule refuses only at depths 0–3.**
6. **One row untouched in every call** differs from 07a only in its LiouvilleGreen oracle (`analytic_rad`/`analytic_06h`),
   which is not among U1's outputs.
7. **The projection underestimated the wall by 1.20×**, inside its margin. 07a's census wall underlies it, and the acted
   analytic calls weigh more than K0's sample showed.

### State handed on

- **Nothing is running.** Both registered runs are `done`: k0 and census. No lock was taken.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (150 files, 11 GB; **prompt 08's "before"**);
  - the k0 run directory;
  - every pickle and every file over 2 MB under `census/`.
- **Read only and unchanged:** the scratch copy, 07a's census and Part A's records, by hash.
- **Issues:** proposed in `issues.md`, for the laptop and the user.
