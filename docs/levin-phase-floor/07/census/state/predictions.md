# levin-phase-floor 07 §6 — the census's predictions, fixed before the census

Written at 2026-11-08T14:09:44Z on mps026298.phys.susx.ac.uk, after Gate V0 (all three clauses held: out/k0/compare_v0b.json, out/v0c/compare_det.json,
out/v0/v0a_imports.txt) and before the census is launched. Not to be rewritten: a correction is a new file and a deviation.

Source: the prompt in the tree g, `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7/prompts/levin-phase-floor/07-stationary-point-guard.md`
(HEAD e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f, file SHA-256 1be67cc1ea1847282272831feb3620037f018f75326183be98c307f3eae78bff).
Part 1 is lines 765-803 (§6, "Predictions, fixed before the run", V1-V5), copied by `sed -n 765,803p`; part 2 is
lines 1017-1018 (§10's last bullet, on V3), copied by `sed -n 1017,1018p`. Both unchanged, word for word.
Part 3 is V4's selection rule and its 24 calls (state/v4_selection.json, written by scripts/select_v4_07.py).

## 1. Prompt §6, the predictions (verbatim)


**Predictions, fixed before the run** (README §5 rule 3). Copy them into
`state/predictions.md` unchanged.
- **V1 — nothing else moves.**
  - **The rule:** every Levin call, group or analytic, in which $g$ locates nothing and guards
    nothing equals 06m's call bit for bit, in its record and its region record, by 06m's
    comparison with the new keys set aside. Every row all of whose calls are such equals 06m's row
    in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
- **V2 — no Levin leaf is returned across a stationary point.**
  - **The rule:** the scan finds no returned **Levin** leaf holding a sign change of $\theta'$, in
    any group call at either rtol or any analytic call at `1e-7`.
  - **Holds** if none does. At 06a's code there were 677 such regions, at `1e-7` alone.
- **V3 — the location finds what the scan finds.**
  - It runs on the scan's grid, so the two should agree exactly.
  - **The rule:** every call in which the scan finds a sign change has a located `root` within one
    scan spacing of each, and every located `root` lies within one scan spacing of a sign change
    the scan found.
- **V4 — honest where it is referenced.**
  - **The set:**
    - z00 and z12, against 06g's reference (`rows`);
    - Z3's seven analytic calls in the 720 (z07, z16, z18, z21, z24, z28, z29), against 06g's;
    - 24 more stationary-point group calls, stratified by band (low, mid), by $k$ (all six that
      carry them), by rtol and by 06m's depth, with the deepest four included. The selection rule
      is fixed in `state/predictions.md` before the run.
  - **The 24's references** are `tools/filon_reference.py`'s, four readings (QAWO and FCC at two
    resolutions), FCC at resolution 2 the reference, from captures at $g$'s tree, as 06g's were
    made. Each capture is checked bit for bit against the census record. They run as one registered
    run with slug `references`, after the census, at most six workers. The 30-minute reading limit
    is a check, not a cutoff (06g's ruling). A projection beyond a day is a stop.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call in the set whose
    four readings agree within 06g's tolerance, here $\min({\rm abserr}_{06m}, {\rm abserr}_g)/100$.
    A call whose readings do not agree is reported, not scored.
  - **Holds** if every scored call is honest.
- **V5 — the cap is not reached where a point was handled.**
  - **The rule:** at both rtols, no leaf reaches `depth_max` in a call that locates a point or
    guards a leaf. An over-budget unit counts as false.
  - A failure is a finding for the user, not a fault. It reopens G2's L2 (Decisions, **G2 L2,
    decided**).

## 2. Prompt §10, its last bullet, on V3 (verbatim)

     - **V3 sets aside a phase Rule A skipped:** its calls are counted and listed, with any sign
       change the scan finds in them; V2 still covers every returned Levin leaf.

## 3. V4's selection rule and its 24 calls

Fixed here from `state/v4_selection.json` (SHA-256 `15a4391b8e49c547a0557bc20829faf5c2323f0e826420ef0e21d694b134d6c8`), written by `scripts/select_v4_07.py` from 06m's census
checkpoint (`ea5bc3ab...`) and `state/rows.json` (`aa523d28...`) alone: it reads no result at g, so the selection is
decidable without g's results (the census has not run). The rule, as the script applies it:

- **The population** (1,438 calls): every group call of 06m's census at 1e-7 whose 06a scan entry in 06m's census
  (the unit's `scan`, made on the call's own theta_deriv, so independent of the driver) shows a sign change of theta'
  (`n_sign_changes` > 0), and the same call (same row, index, label and span, checked) at 1e-8; less z00 and z12
  (`g-1e-7-1-2174` #1, `g-1e-8-1-2174` #1), which V4 scores already. These are exactly the 720 rows, one call each
  (#1, `G+q+r`), at both rtols. Band and k are `state/rows.json`'s `band` and `k`.
- **The deepest four:** sorted by 06m's `max_depth` (deepest first), then 06m's `n_at_depth_max` (most first), then
  (rtol, shard, serial, index); the first four of distinct rows.
- **The other twenty:** four cells, band x rtol (low/1e-7, low/1e-8, mid/1e-7, mid/1e-8), five calls each. The eight k
  are ordered by value (k1 = 1e5 ... k8 = 3e8); k1, k3, k5, k7 go in low/1e-7 and mid/1e-8, and k2, k4, k6, k8 in
  low/1e-8 and mid/1e-7; each cell's fifth slot takes the k with the most calls of the cell not yet slotted there
  (ties: smaller k). The five slots, in ascending k, take the cell's 06m-depth quantiles 0.1, 0.5, 0.9, 0.3, 0.7 in
  that order (quantile q: the element at index floor(q (n - 1) + 1/2) of the cell's depths sorted ascending); each
  slot takes its cell's call of that k whose 06m depth is nearest the target (ties: shard, serial, index), not among
  the deepest four and of a row not already chosen.
- **How it meets the strata:** bands low and mid (the population has no seam call); **all eight k** (the prompt says
  "all six that carry them"; in 06m's census all eight k carry stationary-point calls -- see the log's deviation);
  both rtols, twelve each; 06m's depth from 0 to 20, the deepest four (depth 20, the most leaves at depth_max)
  included; 24 distinct rows.

Population by cell: {"low/1e-7": 411, "low/1e-8": 411, "mid/1e-7": 308, "mid/1e-8": 308}; by k: {"k1": 206, "k2": 206, "k3": 204, "k4": 206, "k5": 206, "k6": 184, "k7": 142, "k8": 84};
k values: {"k1": "100000.0", "k2": "313856.84721559234", "k3": "985061.2054411147", "k4": "3091682.0425413917", "k5": "9703455.784651041", "k6": "30454960.396664713", "k7": "95584978.52172911", "k8": "300000000.0000001"}.
Cells: {"low/1e-7": {"n": 411, "slots": ["k1", "k2", "k3", "k5", "k7"], "fifth": "k2", "targets": [0, 10, 20, 4, 15]}, "low/1e-8": {"n": 411, "slots": ["k1", "k2", "k4", "k6", "k8"], "fifth": "k1", "targets": [0, 11, 20, 3, 16]}, "mid/1e-7": {"n": 308, "slots": ["k1", "k2", "k4", "k6", "k8"], "fifth": "k1", "targets": [0, 9, 20, 0, 15]}, "mid/1e-8": {"n": 308, "slots": ["k1", "k2", "k3", "k5", "k7"], "fifth": "k2", "targets": [0, 12, 20, 1, 16]}}.

| sid | unit | rtol | kind | index | band | k | 06m depth | 06m leaves at depth_max | 06m converged | why | label |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v00 | `g-1e-7-3-37343` | 1e-7 | group | 1 | mid | k5 (9703455.784651041) | 20 | 132 | False | deepest four | k=9.7e+06/Mpc, q=9.7e+06/Mpc, r=9.7e+06/Mpc @ z_response=914.24, z in (1.6721e+11, 914.24), regime Gqr, group G+q+r |
| v01 | `g-1e-7-2-34834` | 1e-7 | group | 1 | low | k4 (3091682.0425413917) | 20 | 130 | False | deepest four | k=3.09e+06/Mpc, q=3.09e+06/Mpc, r=3.09e+06/Mpc @ z_response=0.91255, z in (5.2861e+10, 0.91255), regime Gqr, group G+q+r |
| v02 | `g-1e-8-3-39137` | 1e-8 | group | 1 | low | k1 (100000.0) | 20 | 122 | False | deepest four | k=1e+05/Mpc, q=9.85e+05/Mpc, r=9.85e+05/Mpc @ z_response=0.13184, z in (1.6711e+10, 0.13184), regime Gqr, group G+q+r |
| v03 | `g-1e-8-1-34611` | 1e-8 | group | 1 | mid | k3 (985061.2054411147) | 20 | 100 | False | deepest four | k=9.85e+05/Mpc, q=9.85e+05/Mpc, r=9.85e+05/Mpc @ z_response=398.99, z in (1.6711e+10, 398.99), regime Gqr, group G+q+r |
| v04 | `g-1e-7-3-2326` | 1e-7 | group | 1 | low | k1 (100000.0) | 1 | 0 | True | cell low/1e-7, k1, depth quantile 0.1 (target 0) | k=1e+05/Mpc, q=1e+05/Mpc, r=1e+05/Mpc @ z_response=0.22914, z in (1.709e+09, 0.22914), regime Gqr, group G+q+r |
| v05 | `g-1e-7-0-38985` | 1e-7 | group | 1 | low | k2 (313856.84721559234) | 10 | 0 | True | cell low/1e-7, k2, depth quantile 0.5 (target 10) | k=3.14e+05/Mpc, q=3.09e+06/Mpc, r=3.09e+06/Mpc @ z_response=2.7567, z in (5.2861e+10, 2.7567), regime Gqr, group G+q+r |
| v06 | `g-1e-7-1-2120` | 1e-7 | group | 1 | low | k3 (985061.2054411147) | 20 | 22 | False | cell low/1e-7, k3, depth quantile 0.9 (target 20) | k=9.85e+05/Mpc, q=9.85e+05/Mpc, r=9.85e+05/Mpc @ z_response=0.52504, z in (1.6711e+10, 0.52504), regime Gqr, group G+q+r |
| v07 | `g-1e-7-3-14592` | 1e-7 | group | 1 | low | k5 (9703455.784651041) | 4 | 0 | True | cell low/1e-7, k5, depth quantile 0.3 (target 4) | k=9.7e+06/Mpc, q=9.7e+06/Mpc, r=9.7e+06/Mpc @ z_response=0.13184, z in (1.6721e+11, 0.13184), regime Gqr, group G+q+r |
| v08 | `g-1e-7-1-38834` | 1e-7 | group | 1 | low | k7 (95584978.52172911) | 16 | 0 | True | cell low/1e-7, k7, depth quantile 0.7 (target 15) | k=9.56e+07/Mpc, q=9.56e+07/Mpc, r=9.56e+07/Mpc @ z_response=3.6343, z in (1.6351e+12, 3.6343), regime Gqr, group G+q+r |
| v09 | `g-1e-8-3-34962` | 1e-8 | group | 1 | low | k1 (100000.0) | 2 | 0 | True | cell low/1e-8, k1, depth quantile 0.1 (target 0) | k=1e+05/Mpc, q=3.09e+06/Mpc, r=3.09e+06/Mpc @ z_response=0.39825, z in (5.2861e+10, 0.39825), regime Gqr, group G+q+r |
| v10 | `g-1e-8-0-14314` | 1e-8 | group | 1 | low | k2 (313856.84721559234) | 12 | 0 | True | cell low/1e-8, k2, depth quantile 0.5 (target 11) | k=3.14e+05/Mpc, q=3.09e+06/Mpc, r=3.09e+06/Mpc @ z_response=6.3166, z in (5.2861e+10, 6.3166), regime Gqr, group G+q+r |
| v11 | `g-1e-8-2-2189` | 1e-8 | group | 1 | low | k4 (3091682.0425413917) | 20 | 44 | False | cell low/1e-8, k4, depth quantile 0.9 (target 20) | k=3.09e+06/Mpc, q=3.09e+06/Mpc, r=3.09e+06/Mpc @ z_response=0.30208, z in (5.2861e+10, 0.30208), regime Gqr, group G+q+r |
| v12 | `g-1e-8-0-14299` | 1e-8 | group | 1 | low | k6 (30454960.396664713) | 2 | 0 | True | cell low/1e-8, k6, depth quantile 0.3 (target 3) | k=3.05e+07/Mpc, q=3.05e+07/Mpc, r=3.05e+07/Mpc @ z_response=6.3166, z in (5.169e+11, 6.3166), regime Gqr, group G+q+r |
| v13 | `g-1e-8-2-34841` | 1e-8 | group | 1 | low | k8 (300000000.0000001) | 15 | 0 | True | cell low/1e-8, k8, depth quantile 0.7 (target 16) | k=3e+08/Mpc, q=3e+08/Mpc, r=3e+08/Mpc @ z_response=8.3275, z in (5.1721e+12, 8.3275), regime Gqr, group G+q+r |
| v14 | `g-1e-7-3-13816` | 1e-7 | group | 1 | mid | k1 (100000.0) | 0 | 0 | True | cell mid/1e-7, k1, depth quantile 0.1 (target 0) | k=1e+05/Mpc, q=3.14e+05/Mpc, r=3.14e+05/Mpc @ z_response=75.993, z in (5.406e+09, 75.993), regime Gqr, group G+q+r |
| v15 | `g-1e-7-0-1973` | 1e-7 | group | 1 | mid | k2 (313856.84721559234) | 9 | 0 | True | cell mid/1e-7, k2, depth quantile 0.5 (target 9) | k=3.14e+05/Mpc, q=9.85e+05/Mpc, r=9.85e+05/Mpc @ z_response=25.156, z in (1.6711e+10, 25.156), regime Gqr, group G+q+r |
| v16 | `g-1e-7-2-2092` | 1e-7 | group | 1 | mid | k4 (3091682.0425413917) | 20 | 88 | False | cell mid/1e-7, k4, depth quantile 0.9 (target 20) | k=3.09e+06/Mpc, q=3.09e+06/Mpc, r=3.09e+06/Mpc @ z_response=10.979, z in (5.2861e+10, 10.979), regime Gqr, group G+q+r |
| v17 | `g-1e-7-0-1772` | 1e-7 | group | 1 | mid | k6 (30454960.396664713) | 0 | 0 | True | cell mid/1e-7, k6, depth quantile 0.3 (target 0) | k=3.05e+07/Mpc, q=3.05e+07/Mpc, r=3.05e+07/Mpc @ z_response=3641, z in (5.169e+11, 3641), regime Gqr, group G+q+r |
| v18 | `g-1e-7-2-13649` | 1e-7 | group | 1 | mid | k8 (300000000.0000001) | 15 | 0 | True | cell mid/1e-7, k8, depth quantile 0.7 (target 15) | k=3e+08/Mpc, q=3e+08/Mpc, r=3e+08/Mpc @ z_response=526.01, z in (5.1721e+12, 526.01), regime Gqr, group G+q+r |
| v19 | `g-1e-8-3-1921` | 1e-8 | group | 1 | mid | k1 (100000.0) | 9 | 0 | True | cell mid/1e-8, k1, depth quantile 0.1 (target 0) | k=1e+05/Mpc, q=3.14e+05/Mpc, r=3.14e+05/Mpc @ z_response=100.19, z in (5.406e+09, 100.19), regime Gqr, group G+q+r |
| v20 | `g-1e-8-0-1896` | 1e-8 | group | 1 | mid | k2 (313856.84721559234) | 12 | 0 | True | cell mid/1e-8, k2, depth quantile 0.5 (target 12) | k=3.14e+05/Mpc, q=9.85e+05/Mpc, r=9.85e+05/Mpc @ z_response=229.56, z in (1.6711e+10, 229.56), regime Gqr, group G+q+r |
| v21 | `g-1e-8-1-1747` | 1e-8 | group | 1 | mid | k3 (985061.2054411147) | 20 | 8 | False | cell mid/1e-8, k3, depth quantile 0.9 (target 20) | k=9.85e+05/Mpc, q=9.85e+05/Mpc, r=9.85e+05/Mpc @ z_response=914.24, z in (1.6711e+10, 914.24), regime Gqr, group G+q+r |
| v22 | `g-1e-8-3-14096` | 1e-8 | group | 1 | mid | k5 (9703455.784651041) | 1 | 0 | True | cell mid/1e-8, k5, depth quantile 0.3 (target 1) | k=9.7e+06/Mpc, q=9.7e+06/Mpc, r=9.7e+06/Mpc @ z_response=10.979, z in (1.6721e+11, 10.979), regime Gqr, group G+q+r |
| v23 | `g-1e-8-1-34772` | 1e-8 | group | 1 | mid | k7 (95584978.52172911) | 14 | 0 | True | cell mid/1e-8, k7, depth quantile 0.7 (target 16) | k=9.56e+07/Mpc, q=9.56e+07/Mpc, r=9.56e+07/Mpc @ z_response=43.723, z in (1.6351e+12, 43.723), regime Gqr, group G+q+r |

## 4. How the rules will be read (fixed here, before the census)

- **"Locates nothing and guards nothing"** (V1, and stop 3's comparison): the call's `stationary_points` is empty and
  its `num_guarded_regions` and `num_guard_at_depth_max` are 0 (the harness's `guard_07`: `locates` and `guards` both
  false). A call with a phase Rule A skipped and nothing located or guarded is such a call. "06m's comparison with the
  new keys set aside" is `scripts/compare_07.py calls`: compare_k0.diff, with 06m's exclusions (timing, provenance)
  and the new keys listed in its docstring set aside, nothing else.
- **V2:** a returned leaf of branch Levin listed by the scan (`spscan.scan_call`, "grid" or "ends") in any group
  call at either rtol or any analytic call at 1e-7. A listed leaf that is not Levin (window, guarded-at-cap, or
  ordinary Clenshaw-Curtis) is reported, not counted.
- **V3:** on every scanned call (group at both rtols, analytic at 1e-7) with no phase Rule A skipped: each scan sign
  change (abscissa `sign_change_x`, the midpoint of its bracketing samples) has a located point of kind `root`
  within one scan spacing h, and each located `root` lies within h of a scan sign change. A located point of kind
  `end` (found as a root within h/2 of an end) is reported apart, with whether counting it as a root would change
  the verdict; if it would, that is put to the orchestrator before V3 is scored. Calls with a phase Rule A skipped
  are set aside, counted and listed with the sign changes the scan finds in them (§10).
- **V4:** a call's four readings agree when their spread (largest minus smallest) is within
  min(abserr_06m, abserr_g)/100; the reference is FCC at resolution 2; honest is |value_g - ref| <= abserr_g, in
  exact arithmetic on the hex values and 30-digit readings. For z00, z12 and Z3's seven the readings are 06g's
  (`scores_stage3/scores.json`, `rows`), under this tolerance.
- **V5:** a leaf at `depth_max` (the call's `n_at_depth_max` > 0) in any call, group or analytic, at either rtol,
  that locates a point or guards a leaf, counts against it; an over-budget unit counts as false (06a's over-budget
  pass re-runs it once at 3,600 s; it is reported with both results).
