## Part C §6: the census at 07c's code (Mac Studio), the census agent

**Subject:** prompt 07c §6, as amended by §9 (§9.4, §9.5, §9.7). The census was re-run at the code commit `67b5af74`
(variant (S), the stop below a refused leaf), with 07b's census as the "before". It ran on the Mac Studio (`mps026298`),
2026-11-19 to 2026-11-20, by Part C's census agent. This log covers Gate Y0, the predictions, the projection, the census,
stop 3's two checks (Y1's untouched calls and Y8), Y1–Y9, the descriptive items and the records. Ray-free, no lock, no
store built. The sections below are in the order the work was done; each stop's section is kept as written then.
- **The tree h:** the worktree `SecondaryGWKit-lpf07c-cc3368c`, detached at Part C's dispatch SHA `cc3368ce`.
  - Its code is `67b5af74`'s: `AdaptiveLevin/levin_quadrature.py` `d141a80b…`.
  - It stayed clean with `--ignored`, with no `__pycache__` (`out/cmd/01`, `29`). The main checkout stayed at `28ebaf07`,
    clean.
- **The parent:** there is no parent worktree. Y0's K0 run at the parent ran in h with the parent's module installed in
  place of h's by `variant_loader.py`: `state/variants/levin_quadrature_parent.py`, from `git show cff4237f`, `8956c7b6…`,
  blob `adb7b7df`, the same as 07b's code commit `5e19c4b6` and 07b's census tree `0a103b5a` (`out/cmd/03`).
- **The "before":** 07b's census, `levin-phase-floor-07b-census-20261117T010200`, checkpoint `396784bc…`.
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07c/census/`.

**The user's rulings, relayed by the orchestrator:**
- At stop 2: *"Yes, launch the census"*.
- At stop 3: *"Yes, score Y1–Y9 and run the review"*.

**Result:**
- **Gate Y0 holds**, all five clauses.
- **The projection** was 43,312 s central and 55,392 s at its widest margin times 07b's shortfall, within §9.4's 48 hours.
- **The census** is `levin-phase-floor-07c-census-20261119T201419`, `done`:
  - 18,560 of 18,560 units, 0 in error, 0 over budget;
  - wall 43,080 s (11.97 h; 07b: 45,016 s; 0.995 of the central projection).
- **Stop 3's two checks pass:**
  - all 207,310 untouched calls equal 07b's in Y1's five outputs;
  - all 453 Y8 ids equal (S)'s replay records.
- **Y1–Y9 all hold.** Y4 holds with R2-28 converged and dishonest at 1.1434 of its abserr, as §9.5 predicted, reported
  against `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]`.
- **No call is newly unconverged** against 07b's census. 193 group calls are newly converged.
- **The store, 07b's census and K0 runs, 07b's records and Part A's records are unchanged**, before, at each stop and at
  the end.

### What shipped (at stop 2; the final additions are in "Y1–Y9 scored" below)

No code. The records are in `var/levin-phase-floor-07c/census/`.
- **`scripts/`:** 07b's census harness, copied from h's `docs/levin-phase-floor/07b/census/scripts/`, byte-identical to
  07b's Mac Studio copy (`out/cmd/00`: 232 files, list hash `c74a7107…` for all three). Changes are marked "prompt 07c".
  The diffs are `out/scripts_diff_before_k0.txt` and `out/scripts_diff_at_stop2.txt`.
  - **Changed:**
    - `prov.py`: the tree h (`cc3368ce`), the parent's module, 07b's census and K0 runs;
    - `worker.py`: the three counts on every call; the 07c tree evidence; the stop's instrument (`chase_07c`) and its
      switch;
    - `run_k0.py`: the parent variant 07c's; `--no-instrument` also turns off the stop's instrument;
    - `run_census.py`: its docstring only (07b's 48 h backstop kept).
  - **New:** `ast_check_07c.py`, `y0a_imports_07c.py`, `index_07b_07c.py`, `compare_07c.py`, `select_y0_units_07c.py`,
    `launch_07c.py`, `census_projection_07c.py`, `write_predictions_07c.py`, `hash_runs_07c.py`, and, not yet run on the
    census, `launch_census_07c.py`, `score_y8_07c.py`, `kept_list_07c.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`):
  - Y0: `out/y0/y0a_ast.txt`, `out/y0/y0a_imports.txt`, `out/k0/compare_y0b.json`, `out/y0c3/compare_instr.json`,
    `out/y0c4/compare_det.json`, and `out/k0/compare_k0_h_calls.json` (descriptive);
  - the projection: `out/census_projection.json`;
  - before and at stop 2: `out/runs_hashed_{before,stop2}.txt`, `out/cmd/04`–`06`, `27`–`28b`.
- **One registered run, `done`:** `levin-phase-floor-07c-k0-20261119T151009`, 11 of 11 commands, exit 0, checkpoint
  `1cdc934e…`.

### The stop's instrument (worker.py, "prompt 07c")

§9.5 asks, for each stopped leaf, its route, depth, declaration, `r` and its share of its call's abserr. The stopped leaf
is made inside `_adaptive_levin`'s closures (`_stop_chase_leaf`), which no module attribute reaches. So:
- a pass-through at the module attribute `_chase_at_noise_level` (looked up in the module's globals at call time) returns
  exactly what the original returns;
- on its first call within a recorded Levin call, it finds that call's `_adaptive_levin` frame and replaces, in the
  frame's cell, `_stop_chase_leaf` by a pass-through that calls the original with the same leaf;
- after the original returns, it records the leaf if the original stopped it (its region not floor-limited before, and
  floor-limited after: the original's own test): seq, span, depth, the region's type (Levin or `direct`, the
  Clenshaw–Curtis pair), `forced_direct`, `window_index`, `guarded`, the trust before the stop, `r`, δ₁, δ₂, the check's
  ν_B, the residual, the declared resolution and e′ after the stop, and the root;
- after the call it reads, never writes, the frame's `chase_stopped`, `chase_counts` and `chase_root`. The share is the
  stopped region's e′ as returned (`budget_err`) over the call's abserr. Per call it records the leaves by route and depth,
  the stopped leaf carrying the most of the abserr, and whether the instrument's counts equal the three result counts
  (stopped leaves, roots stopped, bisections counted).

Shown to change nothing by Y0 (3). Its counts equal the result's on every call of Y0 (3)'s two units.

### Gate Y0 and the predictions (at stop 2)

`state/predictions.md` holds four parts of the prompt, each copied verbatim **by text** and checked as a substring:
- §6's Y1–Y8 and descriptive items (prompt lines 502–521);
- §9.5's additions, Y8 and Y9 among them (663–697);
- §9.7's stops (720–724);
- §8's Part C stops (538–539).

Part 5 is this agent's reading of each rule, fixed before the run. Part 6 states Y0 and the projection as they stood.
Y1–Y9 are not yet scored.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **Y0 (1)** the parent's code is 07b's (S0′) | **holds** | `git diff --stat 5e19c4b6 cff4237f -- . ':!docs' ':!prompts'` lists `CLAUDE.md` alone. `levin_quadrature.py` at `cff4237f` is byte-identical to `5e19c4b6`'s (`8956c7b6…`). The K0 workers at the parent loaded 101 files of h, none differing between `cff4237f` and h or between 07b's census tree `0a103b5a` and h; h's own `levin_quadrature.py` was not loaded; the variant was the parent's `8956c7b6` (`out/y0/`) |
| **Y0 (2)** the harness reproduces 07b's census at the parent | **holds** | 100 of 100 K0 units are bit-identical to 07b's census: 2,110 calls (566 group), 55,032 returned regions. Only 07c's five new keys were set aside, 0 of them in 07b's records; 15 workers' evidence all holds (`out/k0/compare_y0b.json`) |
| **Y0 (3)** the instrument changes nothing | **holds** | `h-1e-8-0-33929` (the fix acts: calls #2 and #3 each 1 stop, 10 stopped leaves, 6 bisections counted) and `h-1e-8-3-14505` (21 calls, every `num_chase_stops` 0, 20 with a counted bisection; chosen by the rule fixed before the run) are IDENTICAL with the instruments on and off. Every call of the run ON carries the stop's instrument with its counts equal to the result's; no call of the run OFF does (`out/y0c3/compare_instr.json`) |
| **Y0 (4)** the new code is deterministic | **holds** | `h-1e-8-0-33929`, run twice in two fresh workers, is identical on every key (`out/y0c4/compare_det.json`) |
| **Y0 (5)** §9.4 the projection within 48 h | **holds** | Central 43,312 s (schedule model, ρ by rtol and the 720; ρ_h 0.957–0.994 by class, 0.965 overall). Highest margin 46,285 s. Times 07b's shortfall 1.197: 51,833 s central, 55,392 s highest. No unit projected over budget (`out/census_projection.json`) |

**Descriptive, from K0 at h** (`out/k0/compare_k0_h_calls.json`; not a prediction): the fix acts in 591 of 2,110 K0
calls: 163 of 566 group calls and 428 of 1,544 analytic calls. All 1,519 untouched calls equal 07b's in Y1's five outputs
and in the whole record; the 19 units untouched in every call equal 07b's in both totals. 79 acted calls equal 07b's in
the five outputs. No call changes `converged`. Regions 51,687 against 07b's 55,032 (0.939).

### Deviations from the prompt (so far)

1. **The harness changes** (`prov.py`, `worker.py`, `run_k0.py`, `run_census.py`'s docstring;
   `out/scripts_diff_at_stop2.txt`). STRUCTURALLY REQUIRED for the tree, the parent and the three counts.
2. **The parent ran as a variant from `state/variants/` in h**, as the dispatch directs. STRUCTURALLY REQUIRED.
3. **New scripts** (listed under What shipped). IMPLEMENTATION CHOICE.
4. **The stop's instrument replaces a closure in the running call's frame** (described above). No module attribute
   reaches `_stop_chase_leaf`, and the alternative (reconstructing which leaves were stopped from the returned regions)
   cannot tell a leaf the stop floor-limited from one already floor-limited. Y0 (3) shows it changes nothing.
   IMPLEMENTATION CHOICE.
5. **Y0 (3)/(4)'s units:** `h-1e-8-0-33929` was fixed by name (two of Part A's chase calls, `c-1e-8-0-33929-2` and
   `-3`, which (S)'s replays touched; the cheapest such unit at 07b, 6.3 s). The quiet unit was chosen from K0 at h by a
   rule fixed before the run: every call untouched; then the most calls with a counted bisection (so the instrument's
   pass-through ran without a stop), the most calls, the most group calls. IMPLEMENTATION CHOICE.
6. **`--no-instrument` turns off 07a's, 07b's and 07c's instruments together.** Mode `instr` sets aside the three
   instruments' keys but not the counts. IMPLEMENTATION CHOICE.
7. **The projection** is 07b's model, re-based on 07b's census, with 07b's shortfall (measured over central, 1.197)
   applied as the dispatch asks. IMPLEMENTATION CHOICE.
8. **`predictions.md` adds §9.7's and §8's Part C blocks, and parts 5–6.** IMPLEMENTATION CHOICE.
9. **Smoke tests ran in the session** (`state/smoke/`, `out/cmd/11`–`17b`), not scored. They include a dry run of the Y8
   scorer against 07b's census (`17b`: 128 of 453 equal, as expected of a code that is not (S); it exits 1 by design).
   IMPLEMENTATION CHOICE.
10. **Part A's whole `measurement/` was hashed** (17,111 files), with 07b's K0 run directory and 07b's census working
    records, in place of "Part A's `measurement/`" alone. IMPLEMENTATION CHOICE.
11. **The registry has no job-name field.** The run ids carry the prefix `levin-phase-floor-07c`. STRUCTURALLY REQUIRED.
12. **The A3 v2 fingerprint was not recomputed.** The copy is checked by SHA-256 against its certification, as 07b did.
    IMPLEMENTATION CHOICE.
13. **Failed attempts, each re-run.** UNINTENDED DRIFT (failed attempts). The first two smoke comparisons were written
    with `$T:scripts` and never ran: zsh read `:s` as a modifier ("bad substitution"). They left no record, and were
    re-run with the path quoted as `out/cmd/14` and `15`.
14. **The readings of Y4, Y8 and Y9 in `predictions.md` part 5** (fixed before the run). IMPLEMENTATION CHOICE:
    - Y4's "scored" test takes min(abserr) over this census and 07b's (07b's U4 form, one code on), with 07a's as well for
      R5, as Part A's R5 rule took it;
    - Y8's population is (S)'s 454 replay records less R31's record (not a census call): 453 ids, 448 census calls;
    - Y9 (i) does not count a call with no depth-20 leaf as "every depth-20 leaf below a refused leaf".
15. **Smoke runs 12 and 13 ran concurrently** (two workers in all). IMPLEMENTATION CHOICE.

### Verification performed (so far)

`verification.md` has the account. In brief:
- **The trees:** h is at `cc3368ce`, clean with `--ignored`, no `__pycache__`. The main checkout is at `28ebaf07`, clean.
- **The scratch copy:** EQUAL to its certification and to its before-hash at stop 2. `shasum -c` against 06e's after-hashes
  is OK. No `-wal` or `-shm` files.
- **07b's census and K0 runs, 07b's records and Part A's records:** 150 of 150 and 5 of 5 files equal to 07b's KEPT list;
  07b's 54 kept working files equal; all 17,111 Part A files and 517 07b working files byte-identical before and at
  stop 2.
- **The registry:** 95 runs, all finished. No lock was taken.

### Observations not acted on (so far)

1. **The stop acts on analytic calls in K0:** 428 of 1,544 analytic K0 calls, more than on group calls (163 of 566).
   §9.3 says Part A's sample had no chase on an analytic call, and scores the stop there only by R2's 21 and R3's
   analytic calls. None of the 428 changes `converged` in K0.
2. **The machine has 10 cores** and ran with a background load of about 3–4 cores (Spotlight, CategoriesService) during
   K0. The parent's K0 units ran 1.018× 07b's census unit time, so the projection carries about 2% for it.

### State handed on (at stop 2)

- **Nothing is running.** The k0 run is `done`. No lock was taken.
- **The census is not launched.** It waits on the user's word: `scripts/launch_census_07c.py 43312 46285`.
- **Read only and unchanged:** the scratch copy, 07b's census and K0 runs and records, Part A's records, by hash.

### The launch (after stop 2)

- **The user's word at stop 2**, relayed by the orchestrator: *"Yes, launch the census"*.
- `out/cmd/31`: the registry before the launch, 95 runs, all finished.
- `out/cmd/32`: `scripts/launch_census_07c.py 43312 46285` at 2026-11-19T20:14:19Z.
  - Run `levin-phase-floor-07c-census-20261119T201419`; `sh` pid 90730; driver pid 90737.
  - `run_census.py` `a4ac7ba7…`, `worker.py` `b1fd6cf8…`; module `d141a80b…`, no variant.
- `out/cmd/33`: one check, 36 s after the launch.
  - `state` running; pid 90737 answers `kill -0`; heartbeat 2026-11-19T20:14:20Z.
  - Six 1e-7 workers started; stderr empty.
  - No checkpoint line yet.

### Planned stop 3 (the census `done`; the two checks of §C.1, and the descriptive list)

**The run.** `levin-phase-floor-07c-census-20261119T201419` is `done`, exit 0 (`out/cmd/40`):
- 18,560 of 18,560 units, 0 in error, 0 over budget;
- `stderr.log` empty; checkpoint 3,498,056,160 bytes;
- wall 43,080 s (11.97 h), from the manifest's `created` (2026-11-19T20:14:19Z) to the last heartbeat
  (2026-11-20T08:12:19Z).

**Check 1: Y1's untouched calls** (`out/stop3/compare_census_calls.json`, `.calls.jsonl`; `out/cmd/46`). Every one equals
07b's call in all five outputs. No stop.

| rtol, kind | untouched (`num_chase_stops` 0) | of these equal to 07b's | touched | touched but equal in the five |
|---|---|---|---|---|
| 1e-7 analytic | 73,021 | 73,021 | 34,395 | 5,707 |
| 1e-7 group | 24,775 | 24,775 | 12,550 | 206 |
| 1e-8 analytic | 84,152 | 84,152 | 23,264 | 4,684 |
| 1e-8 group | 25,362 | 25,362 | 11,963 | 203 |
| **all** | **207,310** | **207,310** | **82,172** | **10,800** |

- Every untouched call's whole record also equals 07b's (category `none` on all 207,310).
- 8,508 units are untouched in every Levin call. All 8,508 equal 07b's in `total` and `total_abserr`.
  - 8,495 of them equal 07b's whole.
  - 13 differ only in the harness's own post-section re-solves: `analytic_06h.parent_counterfactual` (11) and
    `route_07a_post_unit.n_solves` (11). These run Levin at the new code after the timed section, and are not among Y1's
    outputs.
- Every call carries the three counts (0 `no_counts`).
- All 18,560 units: module `d141a80b` from the tree, no variant, h clean at its SHA, no foreign module. All 16 workers'
  evidence holds.

**Check 2: Y8** (`out/stop3/y8.json`; `out/cmd/47`).
- 453 of 453 ids, naming 448 distinct census calls, equal (S)'s replay records in value, abserr (hex), `converged`,
  `num_regions` and max depth.
- Each record equals its `S_summary_vplain.json` row. All were replayed under module `09088683…`.
- The stop acted in 327 of the 453. No stop.

**The stop's instrument** (`out/cmd/48b`): on all 289,482 calls its counts equal the three result counts. None is in
error, none failed to find its frame, and every stopped leaf is among the returned regions.

**Descriptive: `converged` against 07b's census** (`out/stop3/stop3.json`, `out/stop3/newly_unconverged.md`;
`out/cmd/48`). Nothing is scored.
- **Newly unconverged (converged at 07b, unconverged here): none,** at either rtol, analytic or group. The list is empty.
- **Newly converged:** 193, all group calls: 97 at 1e-7 and 96 at 1e-8.
- **Unconverged under both codes:** 82 group calls at 1e-7 and 96 at 1e-8.
- **No analytic call is unconverged** under either code (107,416 at each rtol).

**Cost** (descriptive):

| | 07c | | | 07b's census |
|---|---|---|---|---|
| census wall | 43,080 s | 0.995 of the central projection (43,312 s); 0.831 of central × 1.197 (51,833 s); 0.931 of the highest margin (46,285 s) | | 45,016 s (07c / 07b 0.957) |
| unit wall, 1e-7 | 49,405 s | | | 52,154 s |
| unit wall, 1e-8 | 49,325 s | | | 54,667 s |
| largest unit wall | 40.0 s | | | 40.1 s |
| regions, all calls | 5,536,209 | | | 6,039,639 (0.917) |
| regions by rtol and kind | 1e-7 analytic 0.965, 1e-7 group 0.883, 1e-8 analytic 0.976, 1e-8 group 0.891 | | | |

**Read only and unchanged after the census** (`out/cmd/41`–`45`):
- The scratch copy is EQUAL to its certification and to its before-hash; `shasum -c` gives 5 OK.
- 07b's census and K0 runs, 07b's records and Part A's records all hold, and are EQUAL before and after.
- h is clean with `--ignored`, with no `__pycache__`; the main checkout is at `28ebaf07`, clean (`out/cmd/49`).
- The registry holds 96 runs, all finished.

**Deviations at stop 3** (numbered on from 15):

16. **`stop3_07c.py`** is new, after 07b's `stop3_cost_07b.py`: Y1's counts, `converged` against 07b, the newly
    unconverged list with each call's carrying stopped leaf, and cost. IMPLEMENTATION CHOICE.
17. **The instrument's count check** (`out/cmd/48b`) was first run outside `rec.sh` into `out/stop3/`. That file was
    deleted and the check re-run through `rec.sh`. IMPLEMENTATION CHOICE.
18. **Stop 3's checks ran in the session through `rec.sh`**, not as registered runs. The longest took 2.4 minutes.
    IMPLEMENTATION CHOICE.

**Observation not acted on:**

3. **The 13 untouched-in-every-call units that differ from 07b's whole record do so only in the harness's own
   post-section re-solves** (`analytic_06h.parent_counterfactual`, `route_07a_post_unit.n_solves`). These run at the
   new code and are not census outputs. compare_07c.py files `route_07a_post_unit` under category `call` because 07b's
   regex does not name it; that is a categorisation gap, not a change in a census call.

### Y1–Y9 scored (after stop 3, on the user's word)

State: `state/predictions.md` (`cad4fd7f…`, unchanged since 2026-11-19T15:30:24Z). Each verdict is against its rule as
written there (part 5's reading) and in the prompt. Scorers are listed under deviation 19. The summary is
`out/scores/summary.json` (`out/cmd/55`).

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **Y1** nothing else moves | **holds** | 207,310 untouched calls (`num_chase_stops` 0): 97,796 at 1e-7 (73,021 analytic, 24,775 group) and 109,514 at 1e-8 (84,152, 25,362). All equal 07b's in value, abserr, `converged`, `num_regions` and max depth (hex), and in the whole call record. The 4,233 + 4,275 = 8,508 units untouched in every call equal 07b's in `total` and `total_abserr`. 0 calls lack the counts. Coverage 71.6% of calls. Every one of 07b's 60,160 calls with no rate-rule refusal is untouched (§9.5) |
| **Y2** no Levin leaf across a stationary point | **holds** | 0 Levin leaves listed. Window leaves listed: 1,682 (analytic, 1e-7), 526 (group, each rtol), as at 07b |
| **Y3** every phase with something to locate is located | **holds** | 23,993 calls scored, 0 violations. 784 set aside (392 group calls per rtol), each `zero_rate` with no sign change |
| **Y4** honest where referenced | **holds** (R2-28 as §9.5 predicted) | 312 scorings: 299 honest, 3 unconverged, 6 reported and not scored, 4 converged and dishonest, all four R2-28. **(a)** 07b's U4's 190: 181 honest; v04 unconverged (0.000156); 6 not scored (R2-27, R2-29, R3a-0-33666-a13, each at both rtols; stated error above min(abserr)/100; ratio ≤ 0.035); R2-28 and R2-28~o converged and dishonest at **1.1434**. **(b)** Part A's 122: R4 58 honest and 2 unconverged (`c-1e-8-3-13490-2` 0.0086, `c-1e-8-3-13615-3` 0.0083); R5 60 honest; R2-28's census call at both rtols converged and dishonest at **1.1434** (`c-1e-7-0-13757-5`, `c-1e-8-0-13757-5`). No other converged dishonest call. Largest honest ratio 0.805. (b)'s 122 verdicts equal Part A's (S) verdicts on the same calls |
| **Y5** the gate is consistent; the guard a backstop (gated solves) | **holds** | 0 node-partition sums on gated solves, 0 Clenshaw–Curtis regions with a Levin half, 0 guarded regions. Descriptive: 18,238 forced solves took the node-partition sum (07b: 18,238) |
| **Y6** 06b's 24 honest or unconverged | **holds** (as predicted) | 24 of 24 honest and converged; ratios 1.58×10⁻⁵–0.828 |
| **Y7** the pairs | **holds** (as predicted) | 2,484 of 2,484 pairs have r_g ≤ 1, 0 unpaired; largest 0.467. `3-14501` 0.00046, `3-35006` 0.00057 (07b 0.00046, 0.00057). 11 pairs have an unconverged member (07b: 14) |
| **Y8** the code is (S) | **holds** | 453 of 453 ids (448 distinct census calls) equal (S)'s replay records in all five outputs. The stop acts in 327 of them |
| **Y9** the chase ends | **holds** | **(i)** 0 chase calls. The census has 178 unconverged calls, all group (82 at 1e-7, 96 at 1e-8). 11 have no leaf at depth 20; in the other 167, not every depth-20 leaf lies below a refused leaf. **(ii)** Of 07b's 198 newly unconverged calls, exactly five are unconverged here: `c-1e-8-1-2229-4`, `c-1e-8-3-13490-2`, `c-1e-8-3-13490-3`, `c-1e-8-3-13615-2`, `c-1e-8-3-13615-3`. Each is untouched (stops 0), equal to 07b's call, and has no leaf at depth 20 |

#### Descriptive (§6 and §9.5; not scored)

- **Where the fix acts** (`num_chase_stops` > 0): 82,172 of 289,482 calls (28.4%).

  | rtol, kind | calls | acted | with a counted bisection | stops | stopped leaves | bisections counted |
  |---|---|---|---|---|---|---|
  | 1e-7 analytic | 107,416 | 34,395 | 89,981 | 37,241 | 247,001 | 554,650 |
  | 1e-7 group | 37,325 | 12,550 | 18,093 | 13,377 | 277,949 | 113,273 |
  | 1e-8 analytic | 107,416 | 23,264 | 88,453 | 26,905 | 182,952 | 502,641 |
  | 1e-8 group | 37,325 | 11,963 | 17,949 | 12,653 | 269,387 | 111,249 |
  | **all** | **289,482** | **82,172** | **214,476** | **90,176** | **977,289** | **1,281,813** |

  - **By band:** seam 18,290 of 149,482 calls acted (12%), mid 39,754 of 92,400 (43%), low 24,128 of 47,600 (51%).
  - **By k:** from 20% of calls (k = 10⁵) to 32% (k ≈ 9.6×10⁷).
  - **By 07b's class of the call** (the rate rules' refusals at 07b): range-refused 81,527 of 228,648; edge and range
    638 of 665; edge only 7 of 9; no refusal 0 of 60,160.
  - **Windows:** 1,447 of 7,300 calls with a window acted. **The 720:** 15,655 of 30,240.
  - Tables by rtol × kind × band and rtol × kind × k: `out/scores/summary.json` `where_the_fix_acts`.
- **The stopped leaves, by the stop's instrument** (977,289; the instrument's counts equal the result's on every call, 0
  errors, every stopped leaf returned):

  | rtol, kind | Clenshaw–Curtis untrusted / trusted | Levin untrusted / trusted |
  |---|---|---|
  | 1e-7 group | 98,250 / 77,695 | 62,331 / 39,673 |
  | 1e-7 analytic | 8 / 105 | 167,632 / 79,256 |
  | 1e-8 group | 95,287 / 75,341 | 60,117 / 38,642 |
  | 1e-8 analytic | 0 / 8 | 123,419 / 59,525 |

  - **By route:** Clenshaw–Curtis 346,694; Levin 630,595 (64.5%).
  - **By depth:**
    - Clenshaw–Curtis leaves lie at depths 3–17, most at 11–13 (72,781, 145,304, 82,134).
    - Levin leaves lie at depths 1–15, most at 3 and 4 (149,757 and 349,120).
  - **By r:**
    - r ≥ 1: 180,906 Clenshaw–Curtis and 88,693 Levin (group); 278,168 Levin (analytic).
    - r < 1/4: 44,868 Clenshaw–Curtis (group); 50,801 Levin (group); 69,622 Levin (analytic).
  - **Share of the call's abserr** (a stopped leaf's e′ as returned over the call's abserr), carried by one stopped leaf:
    - at least 0.1: 605 Clenshaw–Curtis and 5,358 Levin leaves on group calls; 113 Clenshaw–Curtis and 45,121 Levin leaves
      on analytic calls;
    - the largest single share is 0.99999: `h-1e-7-1-35361` analytic #13, a trusted Clenshaw–Curtis leaf at depth 3,
      r 0.684.
- **`converged` against 07b's census:**
  - **newly unconverged: none**, at either rtol, analytic or group, so no carrying stopped leaf is listed
    (`out/stop3/newly_unconverged.md` is empty);
  - **newly converged: 193**, all group calls the fix acts in: 97 at 1e-7, 96 at 1e-8.
- **`converged` against 07a's census** (from 07b's stop-3 call list):
  - newly unconverged: 5, the five of Y9 (ii), all group at 1e-8 (untouched here, as at 07b);
  - newly converged: 5, the group calls at 1e-8 that 07b converged.
- **Unconverged calls:** 07c 82 + 96 (1e-7 + 1e-8), all group; 07b 179 + 192; 07a 82 + 96.
- **The calls still unconverged with every depth-20 leaf below a refused leaf:** none (Y9 (i)).
- **Rows (units) with a leaf at depth 20:** 77 at 1e-7 and 83 at 1e-8 (07b 182 and 185; 07a 77 and 83).
- **Rows' `total_converged`:** 136 false → true, 0 true → false.
- **abserr against 07b's:**
  - untouched calls: equal.
  - acted group calls: median 1.000; p01 0.62–0.63; p99 2.7; maximum 7.6 (1e-7) and 8.1 (1e-8); above 2× in 302 and
    307.
  - acted analytic calls: median 1.00 (1e-7) and 1.02 (1e-8); 24 above 2× at 1e-7, the largest 242×.
    - The largest is `h-1e-7-1-14470` analytic #7 (J1), converged under both codes.
    - Its abserr is carried (share 0.998) by an untrusted stopped Levin leaf at depth 3, with r = 0.99986.
    - That leaf declares max(residual, δ₁)/(1 − r) = 2.46×10⁻²⁰, from δ₁ = 3.5×10⁻²⁴.
- **Values moved past the sum** of the two codes' abserrs:
  - against 07b's: **0 calls**;
  - against 07a's: 328 group calls (07b's census: 326).
- **Row totals against 07b's:** 8,872 rows differ; the move over 07b's `total_abserr` has median 2.1×10⁻⁸ and maximum
  9.9×10⁻⁵; 0 rows above 1.
- **Cost** (`out/stop3/stop3.json`):

  | | 07c | 07b | ratio |
  |---|---|---|---|
  | census wall | 43,080 s | 45,016 s | 0.957 |
  | against the projection | 0.995 of the central 43,312 s; 0.931 of the highest margin 46,285 s | | |
  | unit wall, 1e-7 | 49,405 s | 52,154 s | 0.947 |
  | unit wall, 1e-8 | 49,325 s | 54,667 s | 0.902 |
  | regions, all calls | 5,536,209 | 6,039,639 | 0.917 |
  | regions, 1e-7 analytic / group | | | 0.965 / 0.883 |
  | regions, 1e-8 analytic / group | | | 0.976 / 0.891 |
  | largest unit wall | 40.0 s | 40.1 s | |

  0 units were over budget.

#### Deviations since stop 3

19. **Scorers, each copied from 07b's census harness and changed where marked "prompt 07c":**
    - `score_07c.py` (from `score_07b.py`; adds the stop's aggregation `r07c`);
    - `score_v4_07c.py` and `score_refs_07c.py` (Y4 (a) and Y6, with 07b's census as the other code);
    - `xcheck_07c.py` (Y7, before = 07b);
    - `summarise_07c.py` (from `summarise_07b.py`).

    New: `score_refs_partA_07c.py` (Y4 (b)). IMPLEMENTATION CHOICE.
20. **Arithmetic:**
    - Y4 (a)'s ratios are 07b's scripts', in mpmath at 40 digits.
    - Y4 (b), every value move and every abserr ratio are in exact Fractions.
    - R2-28 is scored both ways and agrees: 1.1434356254583664 at 1e-7 in each.

    IMPLEMENTATION CHOICE.
21. **"Class" in the descriptive tables** is read as 07b's class of the call by the rate rules, beside kind, window, the
    720, band and k. IMPLEMENTATION CHOICE.
22. **Scoring ran in the session through `rec.sh`**, not as registered runs. The longest took about 2 minutes.
    IMPLEMENTATION CHOICE.
23. **`KEPT_ON_MAC_STUDIO.txt` keeps, by judgment and not by size, the raw worker streams and unit records** of the Y0
    and smoke runs (`emit.jsonl`, `units.jsonl` under `state/`) and the smoke dry run of the Y8 scorer. Their verdicts are
    in `out/`. IMPLEMENTATION CHOICE.

#### Verification performed (final)

`verification.md` has the account. In brief:
- **The trees:** h is at `cc3368ce`, clean with `--ignored`, no `__pycache__`, at every check and at the end. The main
  checkout is at `28ebaf07`, clean.
- **Every census worker:** module `d141a80b` from the tree, no variant, h clean at its SHA, no Ray, no foreign module
  (18,560 of 18,560 units' evidence; 16 workers).
- **The scratch copy:** EQUAL to its certification and to its before-hash, after the census and at the end. `shasum -c`
  against 06e's after-hashes OK (5). No `-wal` or `-shm` files.
- **07b's census and K0 runs, 07b's records and Part A's records:** before, at stop 2, after the census and at the end,
  every hash equal (`out/runs_hashed_{before,stop2,after,end}.txt`).
- **The registry:** 96 runs, all finished. No lock was taken.

#### Observations not acted on (final; proposed in `issues.md`)

1. **The stop acts mostly on Levin leaves, and on analytic calls.**
   - 630,595 of the 977,289 stopped leaves are Levin leaves, most at depth 3–4, 413,499 of them untrusted.
   - 57,659 of the 82,172 acted calls are analytic: 429,832 stopped Levin leaves.
   - §9.3's basis was D1's chase on group calls: "A Levin leaf is rarely at the noise level (both differences within
     4 ν_B in 2,460 of 15,445 Levin bisections)", and "The analytic calls have no chase in the sample". The census reaches
     far beyond that basis.
   - No referenced call is converged and dishonest but R2-28, and no analytic call is unconverged. But the stopped Levin
     leaves' declarations are scored only through the referenced analytic calls (07b's R2's 21 and R3's).
2. **A stopped untrusted leaf's declaration can overstate by the factor 1/(1 − r).** On `h-1e-7-1-14470` #7, r = 0.99986
   gives 2.46×10⁻²⁰ from δ₁ = 3.5×10⁻²⁴; the call's abserr grows 242× against 07b's, and it stays converged. This is the
   pattern the LiouvilleGreen draws showed, at census scale. A stopped leaf declares what the tree declares at
   `depth_max`, as §1.1 sets.
3. **The 324 calls of item 6 are unchanged in kind:** 328 group calls move past the sum of 07c's and 07a's abserrs (07b:
   326 against 07a). None moves past the sum against 07b. They remain 07d's.
4. **13 units untouched in every call** differ from 07b's whole record only in the harness's own post-section re-solves
   (`analytic_06h.parent_counterfactual`, `route_07a_post_unit.n_solves`), not census outputs (stop 3).
5. **Unconverged calls and rows at depth 20 return to 07a's counts** (82 + 96 calls; 77 / 83 rows). The 178 unconverged
   calls all predate 07b, and 167 of them have depth-20 leaves not all below a refused leaf.

#### State handed on

- **Nothing is running.** Both registered runs are `done`: `levin-phase-floor-07c-k0-20261119T151009` (11/11) and
  `levin-phase-floor-07c-census-20261119T201419` (18,560/18,560). No lock was taken.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory in full (149 files, 10.7 GB): **prompt 07d's "before"**;
  - the k0 run directory;
  - every pickle, every file over 2 MB, and the raw worker streams under `census/`.
- **Read only and unchanged:** the scratch copy, 07b's census and K0 runs and records, and Part A's records, by hash.
- **Issues:** proposed in `issues.md`, for the laptop and the user.
- **No commit, no push.**
