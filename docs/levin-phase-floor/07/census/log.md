# levin-phase-floor 07 §6 — the census at 07's code (Mac Studio), the census agent

**Subject:** prompt 07 §6, as amended by §10 (V3 sets aside a phase Rule A skipped), on the Mac Studio (`mps026298`),
2026-11-08 to 2026-11-10, by Part B's census agent. Gate V0, the census, V1–V5, V4's references, the descriptive items,
and a cross-check beyond §6 that the user asked for at stop 4. Ray-free; no store built.
- **The tree g:** the worktree `SecondaryGWKit-lpf07-e182bc7` at Part B's resumed dispatch SHA `e182bc7a`, whose code is
  `adffa21b`'s (07's `ad732887` with Rule A): `AdaptiveLevin/levin_quadrature.py` `4e5da371…`,
  `ComputeTargets/QuadSourceIntegral.py` `33cb161e…`. No variant was installed.
- **The parent p:** the worktree `SecondaryGWKit-lpf07-91472db` at `91472dbe` (06m's code, `beaf0e23…`), for V0's K0 run.
- **The "before":** 06m's census, `levin-phase-floor-06m-census-20261106T102153` (checkpoint `ea5bc3ab…`).
- **The store:** 06's certified scratch copy of A3 v2, read only.

**Commit:** none here. The orchestrator commits these records from the laptop as `docs/levin-phase-floor/07/census/`.

**Result:**
- **Gate V0 holds**, all three clauses.
- **The census** `levin-phase-floor-07-census-20261108T171137` is `done`: 18,560 of 18,560 units, 0 in error, 0 over budget,
  wall 22,708 s (06m: 18,465 s).
- **V1 holds:** all 284,514 calls that locate nothing and guard nothing equal 06m's bit for bit, and all 16,104 units
  made only of such calls equal 06m's whole, `total` and `total_abserr` included.
- **V2 holds:** no returned Levin leaf holds a sign change of θ′ (182,066 calls scanned).
- **V3 holds:** 0 violations in 170,582 scored calls; 11,484 set aside under §10.
- **V4 fails:** 28 of 33 scored calls are honest. z00, z12 and Z3's seven are honest, and 19 of the 24. Not honest:
  v08, v09, v12, v17 and v22, at 2.2–3.2× their abserr.
- **V5 fails:** 23 calls that locate a point hold leaves at `depth_max`.
- **The store and 06m's census are unchanged**, before against after.

## What shipped

No code. The records are in `var/levin-phase-floor-07/census/`:
- **The records:** `log.md` (this file), `verification.md`, `issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and
  `state/predictions.md` (SHA-256 `92e22911…`, mtime 2026-11-08T14:10:10Z, not rewritten).
- **`scripts/`:** 06m's census harness, copied from g's `docs/levin-phase-floor/06m/census/scripts/`, changed where marked
  "prompt 07". The diffs are `out/scripts_diff_at_copy.txt` (empty), `out/scripts_diff_before_k0.txt`,
  `out/scripts_diff_at_stop2.txt` and `out/scripts_diff_at_stop4.txt`.
  - **Changed:**
    - `prov.py` and `refs_common.py`: the trees and the "before";
    - `worker.py`: §1.5's diagnostics, Rule A's key, the scan;
    - `run_k0.py`: `--tree`;
    - `run_census.py`: 06a's over-budget pass;
    - `run_06j.py`: the references' run backstop;
    - `x1_capture.py` and `check_captures.py`: V4's captures, against 07's census.
  - **New:**
    - `compare_07.py`, `hash_runs_07.py`, `index_06m_07.py`, `v0a_imports_07.py`, `select_v4_07.py`;
    - `census_projection_07.py`, `launch_07.py`, `launch_census_07.py`, `stop3_guarded_07.py`;
    - `score_07.py`, `summarise_07.py`, `extra_descriptive_07.py`, `v4_capture_selection_07.py`, `slice_v4_07.py`;
    - `refs_projection_07.py`, `score_v4_07.py`;
    - the cross-check's `xcheck_07.py`, `xcheck_capture_07.py`, `xcheck_projection_07.py` and `xcheck_report_07.py`;
    - `kept_list_07.py`.
- **`out/`:** every command and its output (`out/cmd/`, `out/commands.log`), with these results:
  - V0: `out/k0/compare_v0b.json`, `out/v0c/compare_det.json`, `out/v0/v0a_imports.txt`, `out/k0/compare_k0_g_calls.json`;
  - stop 3: `out/stop3/compare_census_calls.json`, `out/stop3/guard_only.json`;
  - the scores: `out/scores/summary.json`, `out/scores/extra_descriptive.json`;
  - V4: `out/v4/` (captures, checks, inputs, projection, readings, `v4_scores.json`, `report.md`, `scripts_run.txt`);
  - the cross-check: `out/xcheck/`;
  - before and after: `out/runs_hashed_before.txt`, `out/runs_hashed_after.txt`, `out/census_projection.json`.
- **Three registered runs, all `done`:**
  - `levin-phase-floor-07-k0-20261108T135240`: 7 of 7 commands;
  - `levin-phase-floor-07-census-20261108T171137`: 18,560 of 18,560, exit 0, checkpoint `03dfe786…`;
  - `levin-phase-floor-07-references-20261109T005328`: 74 of 74 readings, exit 0, checkpoint `40a9d871…`.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`): the census's checkpoint and region records in full (08 compares
  against them), the references and k0 run directories, and every pickle and every file over 2 MB under `census/`.

## Predictions and verdicts

`state/predictions.md` holds prompt §6's predictions (lines 765–803) and §10's last bullet (lines 1017–1018), verbatim,
then V4's selection and how each rule is read. It was written after V0 and before the census.

| Prediction | Verdict | The numbers it was scored on |
|---|---|---|
| **V0 (a)** the parent's code is 06m's | **holds** | `git diff --stat 48cf87c0 91472dbe` (outside docs and prompts) touches `AdaptiveLevin/tests/test_levin_leaf_values.py` and `CLAUDE.md`; none of the 102 files of p the K0 workers loaded is among them |
| **V0 (b)** K0 reproduces 06m at the parent | **holds** | 100 of 100 units equal: 2,110 calls (566 group, 1,544 analytic), 2,110 region records, 32,518 returned leaves; 0 differing |
| **V0 (c)** the new code is deterministic | **holds** | `g-1e-8-1-2174` (z12's row, which locates one root at g) run twice in two fresh workers: identical on every key but timing and provenance, 07's new keys included |
| **V1** nothing else moves | **holds** at both rtols | At each rtol: 9,280 of 9,280 units; 142,257 calls that locate and guard nothing, all equal; 8,052 units made only of such calls, `total` and `total_abserr` equal in hex |
| **V2** no Levin leaf across a stationary point | **holds** | 182,066 calls scanned (37,325 group at each rtol; 107,416 analytic at 1e-7), 0 scan errors. Listed leaves: 0 Levin. The listed leaves are window leaves (526 group at each rtol, 1,514 analytic) and 168 ordinary Clenshaw–Curtis leaves of analytic calls whose phase Rule A skipped |
| **V3** the location finds what the scan finds (§10's set-aside) | **holds** | 170,582 scored calls; 3,584 with a sign change or a located root; 0 violations; no point of kind `end`. Set aside: 11,484 scanned calls with a phase Rule A skipped (V from 0 to 18.81): 3,260 group calls at each rtol, none with a sign change; 4,964 analytic calls at 1e-7, of which 168 show a scan sign change (J4 and Y4 of 84 rows; 14 of them guarded, O9's calls). In all, counting the calls not scanned, 16,448 calls have a skipped phase |
| **V4** honest where referenced | **fails** | 33 scored. Honest: z00 (0.718), z12 (0.718), Z3's seven (at most 0.0021), and 19 of the 24. Not honest: v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53, |v_g − ref| / abserr_g. Every call's readings agree, the widest spread being 0.0039 of the tolerance. Basis: 13 calls on four readings, 11 on FCC r1 and r2 (the ruling at stop 4) |
| **V5** the cap is not reached where a point was handled | **fails** | 23 calls (10 at 1e-7, 13 at 1e-8), each a `G+q+r` call #1 that locates its root, hold 2–54 leaves at `depth_max` 20 (`g-*-0-38999` holds 54). Three of them are converged (`g-1e-7-1-34766`, `g-1e-8-1-34766`, `g-1e-8-0-14167`); 20 are not. 0 units over budget |

**The five V4 failures.**
- Each is a converged `G+q+r` call of class (c) in the cross-check.
- Each call's maximum region depth at g is 6, 7, 9, 11 or 14 (v17, v12, v08, v09, v22).
- Each one's window was stopped by §1.3's `iroff1` rule while 4–13 of its leaves were still falling; the windows'
  `max_depth_at_stop` was 5–6.
- 06m was 497–30,400× its abserr from the reference on the same calls.
- This is the case O8 and the note's B.5 anticipated (windows stopped while leaves still fell). On these five calls the
  value is not within its declared error.

**The comparison** with 06m is `scripts/compare_07.py`: 06l's `compare_k0.diff`, applied unchanged.
- **Set aside:** 06m's exclusions (timing and provenance) and 07's new keys, each named and why:
  - `guard_07`, `window_07`, `guarded_07`, `guard_at_depth_max_07`: §1.5's diagnostics;
  - `scan_07`, `scan_07_error`, and `scan_07_s` (timing): §6's scan;
  - QuadSourceIntegral's four counts (`stationary_points`, `window_pieces`, `guarded_regions`, `guard_at_depth_max`),
    which reach the pickle's `result_full…groups[]` at g.
- No 07 key name occurs in any 06m record.

## Descriptive (not scored)

- **Depth of the rows:**

  | | rows at depth 20, g / 06m | rows at depth ≥ 18, g / 06m |
  |---|---|---|
  | 1e-7 | 77 / 192 | 283 / 384 |
  | 1e-8 | 83 / 213 | 297 / 411 |

  Of the 720 rows, 24 are at depth 20 at g against 139 at 06m (1e-7), and 27 against 157 (1e-8).
- **Locations:** 4,940 calls locate, the same 2,470 at each rtol.
  - Group calls: 720 roots (one per row of the 720), and 2,404 minima in lone-`G` calls (160 calls with 6 minima and 76
    with 19, at each rtol), λ from 1.6e-7 to 2.3e-6.
  - Analytic calls: 1,514 roots per rtol (757 J4, 757 Y4).
  - No point at an end of a span; no minimum left unlocated.
  - `num_guarded_regions`: 28 in all, O9's calls, one region each; `num_guard_at_depth_max` 0.
- **Windows:** 2 to 32 pieces per call; widths in x from 4.7e-5 to 14.6 (median 2.58).
  - θ′²/|θ″| at the own window edges: group roots 12.52–12.62 (median 12.566 = 2Φ_w); analytic roots 1.18–8.09;
    minima up to 2π.
- **§1.3's firings (B.5):**
  - 1e-7: 1,157 of 2,470 windows stopped; 1,153 with a leaf still falling; 1,146 of those calls converged.
  - 1e-8: 1,195 of 2,470 stopped; 1,192 falling; 1,183 of those calls converged.
  - `leaves_falling_at_stop` 0–29. Every stopped window is listed in `out/scores/summary.json`.
- **Convergence:**
  - Located calls: 262 went from unconverged (06m) to converged, and 15 the other way.
  - Unconverged group calls in the 720: 64 at g (62 within their request, each with an untrusted leaf at `depth_max`;
    2 over, `node_limited`, at most 1.0038×; largest share of a row's `total_abserr` 18.7 %) against 311 at 06m
    (largest share 53.0 %).
  - Outside the 720: 114, identical to 06m's.
  - No analytic call is unconverged.
- **Totals in the 720:** 643 of 1,440 row totals moved by more than 06m's `total_abserr`; the largest move is 2.6e5 of
  it (`g-1e-8-0-2065`); relative moves at most 0.86 %.
- **Cost:**

  | | g | 06m |
  |---|---|---|
  | census wall | 22,708 s | 18,465 s |
  | unit walls, 1e-7 / 1e-8 | 24,203 / 27,479 s | 23,970 / 27,070 s |
  | group-call regions, 1e-7 / 1e-8 | 952,275 / 1,065,288 | 939,882 / 1,053,755 |

  - The median unit-wall ratio is 1.000.
  - 254 tiny units (under 0.3 s) take more than twice 06m's time; 0 units are over budget.
- **The stationary-point calls before:** 06m is honest on 17 of the 33 V4 calls. On the other 16 it is 356–5.2e5× its
  abserr from the reference (z12 1,681×).
- **V4's references:** 74 readings, all usable.
  - Run wall about 25.6 h, against 89,230 s projected.
  - The longest readings were FCC r2 on v18 (20,509 s) and on v13 (19,373 s).

### The cross-check (beyond §6, interim; the user's request at stop 4)

The rule is `state/xcheck_rule_20261109.md` (SHA-256 `e06a1058…`), fixed before any result. The results are
`out/xcheck/xcheck.json` and `out/xcheck/report.md`. Both checkpoints were unchanged before and after.

- **(A) The population and the cross-rtol ratios:**
  - 4,968 calls that locate or guard, i.e. 2,484 pairs; all paired.
  - r_g ≤ 1 in 2,482 pairs and r_06m ≤ 1 in 2,432.
  - Two pairs have r_g > 1: `3-14501` (1.39) and `3-35006` (1.08), both `G+q+r`, band low, k 1e5, class (e).
- **(B) The disagreement list**, |v_g − v_06m| > max(abserr): 843 calls, all `G+q+r`.
  - Classes: (a) 31, (b) 0, (c) 786, (d) 0, (e) 26.
  - 3-38439's `G+q+r` is class (c), the known case.
- **The cost of referencing class (c):** one capture per row; all 396 reproduce the census bit for bit.
  - All four readings: 2.6e7 s; FCC only: 4.47e6 s.
  - 145 rows project at 30 min or less by FCC alone.
- **These are interim results.** The user is to choose a sample of references from them.

## Deviations from the prompt

1. **The harness changes,** each within §6's list: the tree and its paths; §1.5's diagnostics per call and per returned
   leaf, with Rule A's `location_skipped_phases`; the scan on every group call at both rtols and every analytic call at
   1e-7; the comparison with the new keys set aside. STRUCTURALLY REQUIRED.
2. **K0 at p runs as `p-<rtol>-…`.** `compare_07.py whole` compares its `unit` and `tree` under 06m's names (`g-…`, `g`)
   rather than excluding them. IMPLEMENTATION CHOICE.
3. **06h's analytic counterfactual is computed at p as at g,** so that K0 reproduces 06m's records. STRUCTURALLY REQUIRED.
4. **My first `run_k0.py` edit broke the spec line** (a comment swallowed `"units": units,`). I found it by reading and
   fixed it before any run. UNINTENDED DRIFT (a failed attempt).
5. **Development self-tests outside the scored set** (`out/cmd/10`–`13`, `21`; `state/smoke/`). IMPLEMENTATION CHOICE.
6. **K0 at g, its call-by-call comparison and V0 (c) ran inside the k0 run,** so that the census projection had unit
   timings at g. IMPLEMENTATION CHOICE.
7. **Hand entries in `out/commands.log` first had estimated times,** later corrected. UNINTENDED DRIFT (cosmetic).
8. **06a's over-budget pass lives in `run_census.py`,** in the census's own run. It was added after K0, so K0's records
   name the earlier `run_census.py`. IMPLEMENTATION CHOICE.
9. **V4 is stratified over all eight k,** not "all six that carry them": all eight carry stationary-point calls. The
   user ruled at stop 2: "Please keep all 8 `k` and launch the census". IMPLEMENTATION CHOICE.
10. **`state/predictions.md` gained a part 4,** how each rule is read, beyond the brief's three parts. IMPLEMENTATION
    CHOICE.
11. **`census_projection_07.py`'s first version counted the producer rebuild** in K0's ratio. It was superseded, and its
    output is kept in `state/superseded/`. UNINTENDED DRIFT (a failed attempt).
12. **The predictions check wrote and removed an unused `/tmp/x765.txt`.** UNINTENDED DRIFT (cosmetic).
13. **The census held the shared rehearsal lock,** on the orchestrator's direction, though the brief said no lock was
    needed. IMPLEMENTATION CHOICE.
14. **The lock commands and the hashing are hand entries in `out/commands.log`,** each lock command a shell command of its
    own. IMPLEMENTATION CHOICE.
15. **`stop3_guarded_07.py`,** a new reader for stop 3. IMPLEMENTATION CHOICE.
16. **`score_07.py`'s first full run crashed after 17 s** (an unpacking bug). Fixed, checked on five units, then re-run.
    UNINTENDED DRIFT (a failed attempt).
17. **`summarise_07.py`'s first run crashed** on a `None` row depth in a sort. Fixed and re-run. UNINTENDED DRIFT (a failed
    attempt).
18. **V4's capture scripts:** `x1_capture.py` (its title) and `check_captures.py` (07's census) were edited, and
    `refs_common.py` gained `CENSUS07`. STRUCTURALLY REQUIRED (§6's "V4's references").
19. **A 25th capture, z12's call,** made for the projection only. IMPLEMENTATION CHOICE.
20. **`out/v4/` was made after `x1_capture.py` had started,** but before it wrote there. UNINTENDED DRIFT (caught before
    it failed).
21. **`refs_projection_07.py`'s first version failed:** 06g's captures are no longer on this host. Rewritten to take the
    readings' own segment tables; the features were checked equal on z12's and P1's rows. UNINTENDED DRIFT (a failed
    attempt).
22. **That first attempt was started with a plain `&`;** it failed after 3 s. UNINTENDED DRIFT (cosmetic).
23. **The projection model's choices:** γ is the largest calibration value; the margin is ×1.5; the laptop's FCC times
    are scaled to this host by r55's readings. IMPLEMENTATION CHOICE.
24. **New readers and launchers for scoring and V4.** IMPLEMENTATION CHOICE.
25. **V3 covers the scanned calls only,** as §6's scan does: analytic calls at 1e-8 locate (1,514) but are not scanned.
    IMPLEMENTATION CHOICE (recorded as read).
26. **The reading subset.** FCC r1 and r2 on all 24; QAWO r1 and r2 on the 13 whose QAWO readings each project at 30
    minutes or less (`state/v4_reading_list_20261109.json`, `ce594095…`); 74 readings in place of 96. The user: "Take
    your recommended option of FCC at both resolutions plus QAWO where it is projected to be 30 minutes or less. It
    costs only a few hours for an independent check of many calls. Calls without QAWO should be scored on whatever
    references we do have, either one or two FCC readings." STRUCTURALLY REQUIRED (by the ruling).
27. **V4's scoring rule, as the ruling states it.** Four readings: as written. Without QAWO: on the FCC readings it has,
    FCC r2 the reference and the two in agreement within the same tolerance, or one reading with no agreement test.
    Each call's basis is reported; readings that disagree are reported, not scored. STRUCTURALLY REQUIRED (by the
    ruling).
28. **`run_06j.py` gained the run backstop:** it reports past 30 minutes and stops only at a projected finish of 48 h. It
    was tested on a scratch copy with a stub registry. IMPLEMENTATION CHOICE.
29. **`launch_07.py`'s `references` plan was written for the ruled subset.** IMPLEMENTATION CHOICE.
30. **The cross-check, beyond §6.** The user: "Yes, please do the cross-check on `rtol`s and the list of calls where
    `06m` and `07` differ by more than the stated error. We can decide on a sample of references to sort of what these
    difference really show, later." IMPLEMENTATION CHOICE.
31. **Class (e) added to the cross-check's four classes,** named in the rule before any result so that the classes are
    exhaustive. IMPLEMENTATION CHOICE.
32. **The cross-check's captures were grouped one worker per (rtol, k),** one at a time under `nice`, one row per
    class-(c) pair (the integrand is the same at both rtols). IMPLEMENTATION CHOICE.
33. **The cross-check's capture chain was started with `&` inside a background shell,** and awaited by an until-loop on
    its last command's record. UNINTENDED DRIFT.
34. **Development runs of the cross-check on two rows** (`state/dev/`). IMPLEMENTATION CHOICE.
35. **A reading-usability check in V4's scorer:** the tool's and the capture's SHA-256, Route B, Ray not initialised,
    every non-monotone segment integrated directly. All 74 readings pass. IMPLEMENTATION CHOICE.
36. **z00, z12 and Z3's seven are scored on 06g's own readings, two to four of them,** with this rule's tolerance and
    agreement test. IMPLEMENTATION CHOICE.
37. **Readings are compared as exact Fractions of their 30-digit strings** (06g used mpmath at 50 digits). IMPLEMENTATION
    CHOICE.
38. **Two figures in my stop-4 report were misstated;** the JSON was right throughout. The leaves at `depth_max` in V5's
    23 calls number 2–54, not 2–14 (`g-*-0-38999` holds 54). Three of the 23 are converged, not two (`g-1e-7-1-34766`,
    `g-1e-8-1-34766`, `g-1e-8-0-14167`). Corrected here. UNINTENDED DRIFT.
39. **My V4 report to the orchestrator said "3–14 regions deep"** of the five failures. Their maximum region depth at g
    is 6, 7, 9, 11 and 14, and their windows' `max_depth_at_stop` is 5–6. The phrase was in the report message only; it
    is in no file. Corrected here. UNINTENDED DRIFT.
40. **The first before-against-after comparison of 06m's run directory failed:** `sh` has no process substitution. Its
    record is kept in `state/superseded/`, and it was re-run with `bash` (`out/cmd/99b`). UNINTENDED DRIFT (a failed
    attempt).

## Verification performed

`verification.md` has the full account. In brief:
- **The trees, before and after:** g at `e182bc7a`, p at `91472dbe` and `-7460035` at `74600350`, all clean with
  `--ignored` and with no `__pycache__`. The main checkout is at `28ebaf07`, clean.
- **The module hashes:** g `4e5da371…` (equal to `git show` at `e182bc7a` and `adffa21b`), p `beaf0e23…`. Every census
  worker loaded g's module from the tree, with no variant, no foreign module and no Ray.
- **The scratch copy:** EQUAL to its certification and to 06e's after-hashes, before and after; before equals after.
- **06m's census:** 150 of 150 files equal its kept list, before and after.
- **Captures:** 25 + 396 captures reproduce 07's census bit for bit (values, regions, call records, harness).
- **The registry:** 69 runs, all finished. The lock is free.

## Observations not acted on

1. **Lone-`G` group calls locate minima of |θ′|** with λ about 1e-7–1e-6, 6 or 19 per call, 236 calls at each rtol.
   Every one is converged. Not looked into further.
2. **The class-(c) calls of the cross-check** (786 calls, 396 rows) are where both codes are self-consistent across
   rtols and disagree beyond their errors. All five V4 failures are among them, and so is 3-38439. Which code
   is right there is undecided without references.
3. **The references run took about 25.6 h** against a projection of 89,230 s (+3 %). The backstop reported and stopped
   nothing.

## State handed on

- **Nothing is running.** The lock is free; all three of my registered runs are `done`.
- **Kept on the Mac Studio** (`KEPT_ON_MAC_STUDIO.txt`):
  - the census run directory (150 files, 4.8 GB), the references and k0 run directories;
  - every pickle and every file over 2 MB under `census/`: the K0, V0 (c), V4 and cross-check captures (`state/`,
    917 MB), `state/score_07_units.json`, the comparison outputs.
- **The census's checkpoint and region records stay in full; 08 compares against them.**
- **Read only and unchanged:** the scratch copy and 06m's census.
- **Issues:** proposed in `issues.md`, for the laptop and the user to decide.
