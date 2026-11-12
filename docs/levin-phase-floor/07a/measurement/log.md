# levin-phase-floor 07a, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07a Part A, the replays of section 2.1's calls at 07's code and at variants of the window's stop,
on the Mac Studio (mps026298), worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd` at the dispatch
SHA `f93aafd0173bd5de1cdc4107656dbe07c4c1c2a3` (code 07's), records in
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/measurement/`.
**Commit:** none (records committed by the orchestrator, on the laptop).
**Result (2026-11-11): the planned stop of section 2.5.** Stop 1 (2026-11-10) asked two cost questions. The user's
rulings (deviations 16, 17, 20, 21) were applied, and every part of section 2 they left in scope is done:
- the Gate (Mac Studio clauses) holds before and after;
- section 2.1 item 2's captures check, 9 of 9;
- replays at 07's code reproduce 07's census, 430 of 430 bit for bit;
- variants (a), (c)-(f) ran on all 430 calls, and (b) and (g) (as b2 and g2) on items 1-2 and the no-window calls;
- the no-window check passes, 5 of 5 for every variant;
- all 24 attributions are done, and every one closes;
- the O9 check covers the five failures;
- the final tables are written.

**No variant changes any of the five failures.** In every failure the whole excess error is in H3, the Levin leaves
outside the windows. In three of them it sits in a single seed region beside the window that was never bisected.
(b2) and (g2) qualify only by leaving all 29 window calls unconverged. Item 3 is undecided by construction.

*Stop 1's result line, kept:* stopped before section 2.5 with a question on section 2.2's cost rule (the attribution
projects over 12 h) and on the cost of variants (b) and (g), which the prompt does not bound.

## What shipped

Nothing in the tree (its `git status --short --ignored` empty throughout; the main checkout at `28ebaf07`, clean).
Records only, under `measurement/`:
- `scripts/`: 07's census harness copied from `docs/levin-phase-floor/07/census/scripts/` (diff in
  `out/scripts_diff_against_07.txt`: `prov.py` and `refs_common.py` only, each change marked "prompt 07a"), and 07a's
  own scripts (`*_07a.py`, `capture_pairs_07a.py`, `gate_hashes.py`, `launch_07a.py`, `launch_attr_07a.py`);
- `scripts/variants/levin_quadrature_07a_{a,b,c,d,e,f,g,b2,g2}.py`, each with its diff against the tree in
  `out/variants/<v>.diff` and its SHA-256 in `out/variants/variants.json` (the first index kept as
  `variants_v1.json`);
- `state/`: `census_refs.pkl` (07's census records of every call), `calls_07a.json` (the manifest),
  `pairs_capture/` (section 2.1 item 2's captures), the plans;
- `out/`: the replays (`replays/<variant>/<sid>.json`, instrument records `*.instrument.json` for items 1-2), the
  interim tables (`tables_interim_1.{json,md}`), the attribution (`attr/`), the (b) cost probe (`b_scaling/`), the
  captures' check (`pairs_capture/`), the Gate's hashes (`gate_hashes_*.json`);
- registered runs `levin-phase-floor-07a-x396-fast-20261110T215229` (done, 2,376 of 2,376),
  `levin-phase-floor-07a-attr-1-20261110T215423` (done, 8 of 8), `levin-phase-floor-07a-bg2-items12-20261110T223735`
  (done, 68 of 68) and `levin-phase-floor-07a-attr-2-20261111T102500` (done, 15 of 15, 10 h 55 min);
- after stop 1: `out/tables_final.{json,md}` (`tables_07a_final.py`), `out/tables_final_rg.{json,md}`
  (`tables_07a_final_rg.py`, items 2 and 4 on r_g: **the tables of record**), `out/attr/attr_tables_final.{json,md}`
  (all 24 calls), `out/o9/` (`o9_check_07a.py`, `o9_table_07a.py`: the O9 check), `out/gate_hashes_after_final.json`,
  and `out/stop_2_report.md` (the section 2.5 report).

## Predictions and verdicts

Part A has no predictions (section 2.4's tables are descriptive; section 2.5 recommends nothing). The checks it
carries:

| check (prompt) | scored on | verdict |
|---|---|---|
| Gate: code is 07's | `git diff --stat adffa21b f93aafd0 -- AdaptiveLevin ComputeTargets LiouvilleGreen tools` empty | holds |
| Gate: 07's captures match `KEPT_ON_MAC_STUDIO.txt` | 78 of 78 lines (V4 `C00`-`C24`, cross-check `X00`-`X09`), before and after | holds |
| Gate: the scratch copy matches its certification | 5 of 5 files, SHA-256, size and mtime, no extra file, before and after | holds |
| Gate: nothing `running` or stale | Mac Studio registry, 69 runs finished at start (the laptop's: the orchestrator's check) | holds |
| 2.1 item 2: each capture checks against 07's census | 9 of 9 (four pair calls, five no-window calls) by 07's `check_captures.py`: value, abserr, label, converged, regions bit for bit | holds |
| 2.1: a replay at 07's code reproduces its census record | 430 of 430 at the tree and 430 of 430 at (a): value, abserr, converged, num_regions, max_depth, every region array, section 1.5's entries | holds |
| 2.3: a variant does not change a call with no window | five calls, (a), (b2), (c), (d), (e), (f), (g2): 5 of 5 each, bit for bit | holds |
| 2.2: the attribution's closure check, |sum ref_i - ref| <= tol | all 24 calls with a stopped window: 6.2e-6 to 3.3e-3 of tol | holds (no call reported unattributed) |

## Deviations from the prompt

1. **`STRUCTURALLY REQUIRED`** — "07's `make_capture.py`" is 06b's writer of a capture list from 06a's checkpoint; 07's
   capture runner is `xcheck_capture_07.py` (after 06c's `x1_capture.py`: the census `worker.py` re-runs the unit with
   `capture` in its spec). Section 2.1 item 2's captures were made by `capture_pairs_07a.py`, a copy of
   `xcheck_capture_07.py` changed only to merge several calls of one unit into one run of it (07's made one unit entry
   per call), and checked by 07's `check_captures.py` unchanged.
2. **`STRUCTURALLY REQUIRED`** — the harness's trees and paths: `prov.py` and `refs_common.py` name this worktree as
   `g`; `refs_common.ROWS` is 07's census copy of `rows.json` (`var/levin-phase-floor-07/census/state/rows.json`, the
   same SHA-256 `aa523d28...`), since 06g's was removed in the housekeeping of 2026-11-10.
3. **`STRUCTURALLY REQUIRED`** — section 2.3's five calls with no window: every call of section 2.1 has one, so five
   were captured from the pair units in the same two worker runs as item 2 (`g-1e-8-3-14501` calls 0, 2, 4;
   `g-1e-7-3-35006` calls 3 and 7, the last an analytic J2 block), each checked bit for bit like the pairs. Which five:
   **`IMPLEMENTATION CHOICE`**.
4. **`STRUCTURALLY REQUIRED`** — the 396: 07 captured 390 of them at 1e-7 and 6 at 1e-8 (one call per row; the integrand
   is the same at both rtols). Section 2.1 item 3 names the calls at 1e-8, so a 1e-7 capture is replayed with the 1e-8
   census call's `rtol` and `atol` and compared with that call's record; all 396 reproduce it bit for bit.
5. **`IMPLEMENTATION CHOICE`** — the variants are one instrumented copy of the module, differing only in the constant
   `_VARIANT_07A`; (a) is that copy, and 07's code itself is replayed too (`tree`), so the instrument is shown to change
   nothing (430 of 430 at both). The instrument records section 2.2's quantities at each window bisection and stop, and
   every returned leaf at the end; it is on for items 1-2 only (at (a), (c)-(f)).
6. **`IMPLEMENTATION CHOICE`** — the raw estimate e of a leaf is its own solve's error, `used_interval`'s budget error
   with no trust-check declaration; **the analogue of `defab.eq.error`** is `200 |CC_{2N-1} - CC_N| >= resasc`, with
   dqk21's `resasc` formed on the fine Clenshaw-Curtis rule (`sum_j |w_j| |g_j - gbar|`), both positive. The
   Clenshaw-Curtis solve returns `resasc_07a` in its dict and the region keeps it; nothing reads them but the window code
   and the instrument (the no-window check).
7. **`IMPLEMENTATION CHOICE`** — (d)'s value test: `|ΔV| <= min(1e-5, rtol) |V_L + V_R|`.
8. **`IMPLEMENTATION CHOICE`** — (e)'s "request unmet": the call's running error above `max(atol, rtol |V|)` once the
   halves of the bisection that reached the count are in, before the window's leaves are floor-limited (dqage's
   `errsum > errbnd`, tested before it sets `ier = 2`).
9. **`IMPLEMENTATION CHOICE`** — the two extra variants: (f), dqage in full, (c)'s count with (e)'s outcome (section
   0.1's default); (g), 07's count with no count while another leaf of the window fell at its last bisection.
10. **`UNINTENDED DRIFT`**, corrected — in (b) and (g) as first written, the list of the window's other falling leaves
    was formed at every window bisection, a pass over every leaf; read only by (g) and the instrument, it changes no
    value, but makes (b) quadratic in its leaves. `b2` and `g2` (`make_variants_07a_bg2.py`) form it only where read;
    `b2` equals `b` bit for bit at depth_max 12 (v08) at the same wall, so the driver's own passes dominate. (b) and (g)
    are to be run as `b2` and `g2`; no result of `b` or `g` is used.
11. **`IMPLEMENTATION CHOICE`** — the attribution's method: `tools.filon_reference` (Route B, 07's tool `d37e633b`),
    one reading per resolution with every leaf end added to the production edges, each leaf's reference the in-order
    sum of its own jobs (`attr_07a.py`'s docstring); window leaves fall to the tool's direct `mpmath` Gauss-Legendre
    path. Reference: resolution 2; stated accuracy `max(|ref_2 - ref_1|, fit-noise 1 sigma)`.
12. **`STRUCTURALLY REQUIRED`** — section 2.2's "stated accuracy at least 100x finer than the leaf's declared error"
    cannot be met on leaves whose e' is below about 100x the integrand's own double-precision noise (the readings agree
    to about 1e-15 of the local magnitude): 80 of v22's 124 leaves, 32 of v21's 62, 83 of v03's 161. Such leaves are
    listed per call; the parts' summed accuracies are given beside each part.
13. **`IMPLEMENTATION CHOICE`** — part H3 ("the Levin leaves outside every window") holds every leaf outside every
    window, Clenshaw-Curtis fallback leaves included (listed by type); each part's declared error is `sum e' + |sum E|`
    over its leaves, as the call's abserr is formed.
14. **`IMPLEMENTATION CHOICE`** — the (b) cost probe (`b_scaling_07a.py`: v08 and v12 at variant (b) with `depth_max`
    6-18, scratch, not a section 2.4 table), run to project (b)'s cost before committing it.
15. **`STRUCTURALLY REQUIRED`** — the projection rule as written passed (the pilot, v22, projects 9.4 h) but the next two
    calls cost 2.6-2.7x what the pilot's ratio predicts (07's own readings' CPU is the better guide: about 25 h). The
    20 remaining attributions were held, and reported with the calls to drop (stop 1).
16. **`STRUCTURALLY REQUIRED`** — the user's ruling on stop 1's first question, relayed by the orchestrator on
    2026-11-11 (the user's words: "run all the calls needed for the H3 attribution in full"): every remaining
    attribution of section 2.2 is run, v13, v18 and v23 included, past section 2.2's twelve-hour rule (projection
    about 25 h on six workers). Run as `levin-phase-floor-07a-attr-2-20261111T102500` (15 calls, serial, six spawn
    workers each); the earlier `attr-1` ran eight calls with one worker each beside the (b2)/(g2) run.
17. **`STRUCTURALLY REQUIRED`** — the user's ruling on stop 1's second question ("no (b) or (g) on the 396 class-c
    captures"): (b2) and (g2) are run on items 1-2 and the no-window calls only (68 of 68 ok), not on the 396.
    Section 2.4 item 3 is therefore **undecided by construction** for every variant; (b2) converged on none of the 29
    window calls of items 1-2 (only on the five no-window calls, where it equals the tree). Qualification (item 4) is
    judged on items 1-2 with item 3 undecided.
18. **`IMPLEMENTATION CHOICE`** — the final tables are written by `tables_07a_final.py`, a copy of `tables_07a.py`
    (left as run for the interim tables) changed only for deviation 17 and for the pairs' r_g (deviation 19), each
    change marked "07a final"; it reads b2 and g2 as (b) and (g) (deviation 10).
19. **`IMPLEMENTATION CHOICE`** — at the orchestrator's instruction (2026-11-11; the pairs' formula is a question still
    with the user), item 2 tabulates the prompt's r = |v_7 - v_8| / (e_7 + e_8) and, labelled beside it, the
    cross-check's r_g = |v_7 - v_8| / e_7.
20. **`STRUCTURALLY REQUIRED`** — the user's ruling of 2026-11-11 on the pairs' formula, relayed by the orchestrator
    ("happy to follow your recommendation of using `r_g`"): items 2 and 4 are scored on the cross-check's
    r_g = |v(1e-7) - v(1e-8)| / abserr(1e-7) (07's `xcheck_rule_20261109.md`); the prompt's r is kept beside it,
    labelled. The prompt's formula contradicted its own "as the cross-check scored them" (the orchestrator's error, in
    its words). Written by `tables_07a_final_rg.py`, a copy of `tables_07a_final.py` (left as run) changed only for
    this, each change marked "07a r_g". Deviation 19's tables are superseded for items 2 and 4, and kept.
21. **`STRUCTURALLY REQUIRED`** — an addition at the user's word (2026-11-11, on the orchestrator's proposal): the O9
    check on the H3 leaves of the five failures, a measurement only, at 07's code as it is
    (`scripts/o9_check_07a.py`, `out/o9/<sid>.{json,md}`): for each leaf, its parent and its sibling, 07's gate
    estimate `phase_span` from `_Basis_SinCos.build_Levin_data` on the region's own grid, |theta(b) - theta(a)| from
    the raw phase, whether theta' changes strict sign on the nodes, the node-partition sum, and the route each would
    take against `SIX_PI` and `_LEVIN_DIRECT_SOLVE_PHASE_SPAN`.
22. **`IMPLEMENTATION CHOICE`** — the O9 check's details. (i) The grid is the call's own `chebyshev_order` (24 on
    v22), not 16 as the instruction said: "the leaf's own nodes" are the ones its solve used. (ii) The leaves
    checked: every H3 leaf whose |true error| exceeds its declared e' + |E|, and, labelled, the three H3 leaves of
    largest |true error|. (iii) The node-partition sum is given for every region; it is the true variation used for
    the route only where theta' changes sign, else |theta(b) - theta(a)|. (iv) Parent and sibling are found by
    descending from the leaf's seed region (the range between windows that holds it) by `break_point`, the driver's
    own split, and the leaf must be reached exactly at its depth. (v) It is run for each failure once the
    attribution run has ended, not as each call lands: watching for each would mean polling a long run (CLAUDE.md
    rule 4); its result does not depend on when it runs.

## Verification performed

- The Gate's hash clauses before and after (`out/gate_hashes_before.json`, `out/gate_hashes_after_interim_1.json`):
  78 of 78 captures, 5 of 5 store files; 07's census checkpoint `03dfe786...` before and after; R29's store and 07's
  run directories not opened.
- 9 of 9 captures by `check_captures.py` (`out/pairs_capture/check_captures.json`).
- 430 of 430 replays reproduce 07's census at the tree and at (a) (`out/tables_interim_1.md`).
- The no-window check, (a), (c)-(f): 5 of 5 each.
- `b2` = `b` bit for bit on v08 at depth_max 12.
- The attribution's closure check on all 24 calls: |sum ref_i - ref| / tol between 6.2e-6 (v09) and 3.3e-3 (v08); holds
  on every call (`out/attr/attr_tables_final.md`).
- After the runs (`out/gate_hashes_after_final.json`): 78 of 78 captures and 5 of 5 store files match again. 07's
  census checkpoint is still `03dfe786...` (hashed read-only).
- The O9 check: each region was reached by descent at exactly its own depth; no family error on any of the five
  calls.
- The worktree's `git status --short --ignored` empty; the main checkout at `28ebaf07`, clean; no Ray initialised in
  any replay (`ray_initialized` false in every output).

## Observations not acted on

- The requests of several calls are atol-dominated (v08: `atol` 1.86e-33 against `rtol |V|` 3.4e-38; abserr 1.94e-36):
  converged is far inside the request; the honesty failures are of the declared error, not of the request.
- In every window of items 1-2 the bisections are driven by untrusted leaves (issues.md item 2).
- On v22 the excess error is in one Levin leaf beside the window (issues.md item 1).
- (d) changes 58 of the 396 (value, abserr or regions); (e) and (f) leave 35 more of the 396 unconverged, and v03 and
  the two 1e-8 pair members; no variant run so far changes a V4 failure.
- 16 of the 396 have a window that never stopped.
- (b2) and (g2) are equal bit for bit on all 34 calls: (g)'s rule never stops a window here, because at every
  bisection that reaches the count another leaf of the window also fell. Both leave all 29 window calls unconverged:
  764,283 regions and about 50,000 s, against (a)'s 2,386 regions and 48 s on items 1-2.
- In three failures (v08, v12, v17) the leaf carrying all of the excess is a **depth-0 seed region**, the range from
  the window's upper edge to the end of the span (about [9.23, 27-28]). It was solved as one Levin leaf, with an
  estimated phase span of 1.3e8-4.1e8, and never bisected; its true error is 2.9-7.4x its e'. In v09 the leaf is at
  depth 1 and in v22 at depth 5 (issues.md item 1).
- The O9 check finds no route change and no strict sign change of theta' on any leaf, parent or sibling checked.
  07's estimate of the phase span overstates the true variation on the big seed leaves (true / estimate 0.73-0.75),
  but both sides are far above 20 pi.
- The H2 part's |true| / declared exceeds 1 on two controls, v02 (1.31) and v10 (1.04). Both parts are about 1e-7 of
  their call's abserr.

## State handed on

At the section 2.5 stop: nothing running. All four of 07a's registered runs are `done`, and the lock was not used.
The scratch copy, 07's census and its captures are unchanged. The worktree's `git status --short --ignored` is empty,
and the main checkout is at `28ebaf07`, clean. `KEPT_ON_MAC_STUDIO.txt` lists 61 files: every pickle, and every file
over 2 MB, chiefly the (b2)/(g2) replay records.
*At stop 1 (kept):* `attr-1` was running, and nothing else.
