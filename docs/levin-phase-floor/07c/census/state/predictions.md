# levin-phase-floor 07c §6 — the census's predictions, fixed before the census

Provenance (the writer's header):

```
levin-phase-floor 07c section 6: the census's predictions, fixed before the census
written 2026-11-19T15:30:24Z on mps026298.phys.susx.ac.uk
script write_predictions_07c.py sha256 6287f2f8ad0df083d6c8d114bf824e6f217d2b0663e4977b41b5dc2e7f07384a
tree h: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c HEAD cc3368ce69d42fd6871d6264fd30efa65dd00cea (status --short: empty)
main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
input /Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c/prompts/levin-phase-floor/07c-chase-below-a-refused-leaf.md sha256 c13e8d3573ab9aebd1fe59328793b728e4d349118abb0863d77c233d191b503b
```

Written after Gate Y0 (all its clauses) and before the census is launched. Not to be rewritten: a correction is a new file and a deviation.

Source: `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c/prompts/levin-phase-floor/07c-chase-below-a-refused-leaf.md` (file SHA-256 c13e8d3573ab9aebd1fe59328793b728e4d349118abb0863d77c233d191b503b). Parts 1-4 are copied **by text** (each from a fixed opening string to a fixed closing string, both unique in the file; the line numbers are given for reference only) and checked to be substrings of the prompt. Parts 5 and 6 are this agent's.

## 1. Prompt §6, the predictions and the descriptive items (verbatim; prompt lines 502-521)

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md` unchanged,
with §9's additions:
- **Y1 — nothing else moves.** Every call in which the fix does not act (§9 names the counts) equals
  07b's call bit for bit in value, abserr, `converged`, `num_regions` and maximum depth. Every row all
  of whose calls are such equals 07b's in `total` and `total_abserr`. **Holds** if true at both rtols.
- **Y2 — no Levin leaf is returned across a stationary point** (07a's W2).
- **Y3 — every phase with something to locate is located** (07a's W3).
- **Y4 — honest where it is referenced.** Every call referenced in 07b (U4's 190) and in this prompt's
  Part A (R4, R5, R2-28), scored from the census against its reference by 07's rule. **Holds** if every
  one is honest, or unconverged under §9's rule.
- **Y5 — the gate is consistent, and the guard is a backstop** (07a's W5, on gated solves).
- **Y6 — 06b's 24 are honest,** against $S_{10}$ by 06b's rule, or unconverged.
- **Y7 — the pairs:** $r_g$ over all 2,484 pairs; every pair has $r_g \le 1$, or a member unconverged.
- **Y8 — the code is the chosen variant:** every census call Part A replayed equals the variant's
  replay record bit for bit in value, abserr, `converged`, regions and maximum depth.

**Descriptive, not scored:** where the fix acts, by class, band, $k$ and rtol; `converged` changes
against 07b's and against 07a's, in each direction, and the calls still unconverged with every
depth-20 leaf below a refused leaf; rows at depth 20; the abserr ratio against 07b's; values moved past
the sum of the two codes' abserrs (**the sum**, not each separately); regions and cost against 07b's.

## 2. Prompt §9.5, the additions to §5's and §6's predictions (Y9 among them) (verbatim; prompt lines 663-697)

### 9.5 Additions to §5's and §6's predictions

- **Y1 counts** a call untouched if `num_chase_stops` is 0, and compares value, abserr, `converged`,
  `num_regions` and the maximum depth. Every call with no rate-rule refusal is untouched (07b's U1:
  60,160 calls); on Part A's 454, 126 are, and on R2's 60 random calls, 47.
- **Y4, on the user's ruling.** R2-28 is predicted converged and dishonest at 1.143 of its abserr at
  both rtols (its census call `c-1e-7-0-13757-5` and `c-1e-8-0-13757-5`). Its error is the carrying
  leaf's (§9.2), not the stop's, and belongs to the issue opened in §9.6. **Y4 holds** if every other
  referenced call is honest or unconverged; R2-28 is reported with its ratio against that issue. Any
  other referenced call converged and dishonest fails Y4.
- **Y6 and Y7 are predicted to hold.** 06b's W2 calls and the two pairs that Part A replayed are honest
  under S. The rest were not replayed; the prediction rests on S leaving no other referenced call
  converged and dishonest.
- **Y8 — the code is (S).** Every census call that Part A replayed (§2.1's, both rtols) equals (S)'s
  replay record bit for bit in value, abserr, `converged`, regions and maximum depth. **Holds** if all
  do.
- **Y9 — the chase ends.** No call is unconverged with every depth-20 leaf below a leaf a rate rule
  refused. Of 07b's 198 newly unconverged calls, exactly five stay unconverged: `c-1e-8-1-2229-4`,
  `c-1e-8-3-13490-2`, `c-1e-8-3-13490-3`, `c-1e-8-3-13615-2` and `c-1e-8-3-13615-3`, each untouched
  and equal to 07b's call. **Holds** if both are true. Y9 is the header's closing condition for
  `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`, and is scored with Y1–Y8 (§7).
- **S3 (§5).** R31's record (`G+q+r` at $(10^5, 10^6, 10^6)$, $z = 144627.7$) is predicted to converge
  at 86 regions, honest at 0.053 of its abserr. Its row is named as predicted to move, with
  `total_converged` possibly false → true. Any other row beyond S3's bound is reported with the stops
  in its calls.
- **Descriptive, added:** the stops by class, band, $k$ and rtol; by the harness's instrument (marked
  "prompt 07c"), each stopped leaf's route (Clenshaw–Curtis or Levin), depth and declaration, and its
  share of its call's abserr; `converged` changes against 07b's in each direction, with the calls
  newly unconverged listed (none is expected).

**Part B's review** replays from copies of their captures at the code commit:
- R2-28, equal to (S)'s replay record in `float.hex` (value `-0x1.654fbbd987ff2p-96`, abserr
  `0x1.11de231bf47dap-122`, 100 regions, depth 13, converged);
- one other chase call, `c-1e-8-1-2113-1`, equal to (S)'s record;
- `c-1e-8-1-2229-4` (untouched), equal to (S)'s record and to 07b's census.

## 3. Prompt §9.7, the stops added (for reference; not predictions) (verbatim; prompt lines 720-724)

### 9.7 Stops added (into §8)

- **In Part B:** an output of the code differing from (S)'s replay record on a call the review replays;
  no call found for §3 (b).
- **In Part C:** Y8 failing, before anything else is scored.

## 4. Prompt §8, Part C's stops (for reference; not predictions) (verbatim; prompt lines 538-539)

- In Part C: S0′ failing; the census's projection beyond 48 hours; a call Y1 says is untouched
  differing from 07b's, or Y8 failing, before anything else is scored; a lock held.

## 5. How each rule is read (this agent's, fixed here before the census)

Throughout: a census call is named by its unit `h-<rtol>-<shard>-<serial>`, its kind (group, in the unit's `calls`; or
analytic, in `other_calls`) and its index; 07b's census call is the one of the same name in
`levin-phase-floor-07b-census-20261117T010200` (checkpoint `396784bc…`). Floats are compared by `float.hex`; ratios are
computed in exact `Fraction`s from the hex values and the references' decimal strings. "Unconverged" is the call's
`converged` false. A unit not `done` in either census is listed and not compared.

- **Y1.** A call is untouched iff `num_chase_stops` is 0 (§9.5), as the call's result returns it (the census records the
  three counts on every call; a missing count is a fault, reported). Each untouched call is compared with 07b's call in
  value, abserr, `converged`, `num_regions` and `max_depth`. A "row all of whose calls are such" is read as a unit (one
  row at one rtol) every one of whose Levin calls, group and analytic, is untouched; it is compared with 07b's unit in
  `result.total` and `result.total_abserr`. **Holds** if every untouched call is equal in all five at both rtols and
  every such unit is equal in both totals. Beside it, descriptive: each untouched call's whole record diffed against 07b's
  by category (compare_07c.py mode `calls`). **Stop** (§8): any untouched call differing in the five, before anything
  else is scored.
- **Y8.** The calls are every record Part A wrote of (S)'s replays, `var/levin-phase-floor-07c/measurement/out/replays/S/
  <id>.json` (454), less `r31-3-20363-1` (R31's record, a rehearsal row and not a census call): 453 ids, 448 distinct
  census calls (five census calls are named by two ids each; each id is compared). Each id's census call is read from
  Part A's mappings (`out/census_refs_07c.json` `calls`; `state/census_wanted_07c.json`, `census_wanted_07c_chase.json`,
  `census_wanted_07c_r5.json` `wanted`). (S)'s replay record is the file's `replay`, checked to equal
  `S_summary_vplain.json`'s row. **Holds** if all 453 equal the census call in value and abserr (hex), `converged`,
  `num_regions` and `max_depth`. **Stop** (§9.7): any failure, before anything else is scored (scripts/score_y8_07c.py).
- **Y2, Y3, Y5** as 07b's census scored U2, U3 and U5 (07a's W2, W3 and W5; score_07b.py), Y5 on gated solves only
  (07a's W5 ruling), the forced solves' count descriptive.
- **Y4.** Two sets, each scored from this census against its reference by 07's rule: a call is scored if its reference's
  stated error is at most min(abserr)/100 over this census's and 07b's census's abserr for the call (07c's form of 07b's
  U4 rule, min(abserr_07b, abserr_07a)/100; for R5, 07a's census abserr as well, as Part A's R5 rule takes it), otherwise
  reported and not scored; a scored call is honest iff |v − ref| ≤ abserr; a call unconverged here counts as
  "unconverged under §9's rule".
  - (a) **07b's U4's 190 scorings**: the same calls, on the same references, by 07b's census scripts (score_v4_07b.py:
    V4's 25 and z00 and Z3's seven, with z12, on 07's and 06g's readings; score_refs_07b.py: 07b Part A's R1 60, R2 60,
    R3 (W7 6, aux 12, own 6, pairs 4) and item 3, on 07b Part A's references, and 07a's rows 1-34599 and 3-678, 8 calls,
    on 07a's census references), copied with this census in place of 07b's and 07b's census in place of 07a's as the
    other code.
  - (b) **07c Part A's R4, R5 and R2-28**: every entry of Part A's `out/tables/scores_07c.json` `scored` whose
    `ref_key` is `ref-R4-*` (sets `chase` and `chase~o`, 60) or `ref-R5-*` (sets `r5` and `r5~o`, 60), and R2-28's census
    call at both rtols (`c-1e-7-0-13757-5`, `c-1e-8-0-13757-5`, `ref-R2-28`), each id's census call by Part A's mappings
    (as Y8), on Part A's reference and stated error (`ref`, `spread` there).
  - **Holds** if every scored call in (a) and (b) is honest or unconverged, except R2-28: its census call at both rtols in
    (b), and its two scorings in (a) (07b Part A's `R2-28` and `R2-28~o`, the same integrand), which §9.5 predicts
    converged and dishonest at 1.143 of the abserr; each is reported with its ratio against
    `[07c-r2-28-trusted-levin-leaf-hides-a-common-error]` and does not fail Y4. Any other scored call converged and
    dishonest fails Y4. A call named in both sets is scored once under each. Nothing is attributed (the dispatch).
- **Y6.** 06b's 24 calls (`docs/levin-phase-floor/06b/census/out/w2_reference.json`; unit `b-1e-7-X-N` is this census's
  `h-1e-7-X-N`, the group call of the same index) against $S_{10}$ by 06b's rule, as 07b's score_refs_07b.py scored U6:
  scored if $|S_9 - S_{10}| \le$ abserr/100; honest iff $|v - S_{10}| \le$ abserr; **holds** if every one is honest or
  unconverged. Nothing is attributed.
- **Y7.** $r_g$ over 07's 2,484 pairs (07's census marks), by 07b's xcheck_07b.py with this census and 07b's census as
  the before; **holds** if every pair has $r_g \le 1$ or a member unconverged.
- **Y9.** Two clauses, both over this census's Levin calls at both rtols.
  - (i) **A chase call** is a call that is unconverged, returns at least one leaf at depth 20, and every one of whose
    depth-20 leaves lies inside the span of a leaf a rate rule refused in the same call at a smaller depth (07b's census
    rule, score_07b.py's `r07b_unit`: the rate-rule instrument's refused evaluations, `lo ≤ start`, `end ≤ hi`, refused
    depth below the leaf's). A call unconverged with no leaf at depth 20 (a budget miss) is not a chase call: "every
    depth-20 leaf" is not read as true of a call with none. Clause (i) is true iff there is no chase call.
  - (ii) **07b's 198 newly unconverged calls** are the list of 07b's census `out/scores/summary.json`
    `newly_unconverged`. Clause (ii) is true iff, of the 198, exactly the five named in §9.5 are unconverged here, and
    each of the five has `num_chase_stops` 0 and equals 07b's call in Y1's five outputs.
  - **Holds** iff (i) and (ii) are both true. A call newly unconverged against 07b is not a stop and changes no verdict
    (Y4, Y6 and Y7 count an unconverged call as passing; Y9 is about depth-20 leaves).
- **The projection** (§9.4): within 48 hours at its highest margin; stated also times 07b's shortfall (1.197).
- **Descriptive** (§6 and §9.5), not scored:
  - where the fix acts (`num_chase_stops` > 0) by kind, 07b's class of the call (the rate rules' refusals), window,
    the 720, band, $k$ and rtol; the stops, stopped leaves and counted bisections by the same;
  - by the stop instrument (`chase_07c`, marked "prompt 07c"): each stopped leaf's route (Levin or the Clenshaw–Curtis
    pair, `type` `direct`), its trust before the stop, depth, declaration (declared resolution and e'), $r$, and its share
    of its call's abserr (its e' as returned over the call's abserr); the instrument's counts against the three result
    counts on every call;
  - `converged` changes against 07b's and against 07a's (07b's census `out/stop3/compare_census_calls.calls.jsonl`
    carries 07a's), in each direction; **every call newly unconverged against 07b listed with the stopped leaf carrying
    the most of its abserr** (route, depth, $r$, share), or "no stopped leaf" if the stop did not act in it;
  - the calls still unconverged with every depth-20 leaf below a refused leaf (Y9 (i)'s list, expected empty);
  - rows (units) with a leaf at depth 20, each rtol, against 07b's;
  - the abserr ratio against 07b's, untouched and acted calls apart;
  - values moved past **the sum**: $|v_{07c} - v_{07b}| > {\rm abserr}_{07c} + {\rm abserr}_{07b}$, in exact
    arithmetic; never each separately;
  - regions and cost against 07b's (census wall, unit walls, unit time with post sections, regions by rtol and kind).

## 6. Gate Y0 and the projection, as they stood when this file was written

- Y0 (1): `out/y0/y0a_ast.txt`: verdict: HOLDS
- Y0 (1): `out/y0/y0a_imports.txt`: verdict: HOLDS (the code the K0 workers executed is the parent code, and the parent is 07bs)
- Y0 (2): `out/k0/compare_y0b.json`: 100 of 100 K0 units equal 07b's census at the parent; totals {'n_calls': 2110, 'n_group_calls': 566, 'n_regions': 2110, 'n_regions_returned': 55032}; worker evidence all hold True; new keys in 07b's records 0
- Y0 (3): `out/y0c3/compare_instr.json`: h-1e-8-0-33929 IDENTICAL (stops [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 1, 10, 6, 10], [3, 1, 10, 6, 10], [4, 0, 0, 3, 0], [5, 0, 0, 2, 0], [6, 0, 0, 3, 0], [7, 0, 0, 2, 0], [8, 0, 0, 1, 0], [9, 0, 0, 2, 0], [10, 0, 0, 1, 0], [11, 0, 0, 5, 0], [12, 0, 0, 3, 0], [13, 0, 0, 3, 0], [14, 0, 0, 4, 0], [15, 0, 0, 5, 0], [16, 0, 0, 6, 0], [17, 0, 0, 5, 0], [18, 0, 0, 6, 0], [19, 0, 0, 0, 0], [20, 0, 0, 0, 0]]); h-1e-8-3-14505 IDENTICAL (stops [[0, 0, 0, 0, 0], [1, 0, 0, 7, 0], [2, 0, 0, 3, 0], [3, 0, 0, 3, 0], [4, 0, 0, 5, 0], [5, 0, 0, 7, 0], [6, 0, 0, 7, 0], [7, 0, 0, 4, 0], [8, 0, 0, 8, 0], [9, 0, 0, 4, 0], [10, 0, 0, 8, 0], [11, 0, 0, 12, 0], [12, 0, 0, 6, 0], [13, 0, 0, 3, 0], [14, 0, 0, 7, 0], [15, 0, 0, 5, 0], [16, 0, 0, 7, 0], [17, 0, 0, 5, 0], [18, 0, 0, 7, 0], [19, 0, 0, 7, 0], [20, 0, 0, 9, 0]])
- Y0 (4): `out/y0c4/compare_det.json`: h-1e-8-0-33929 IDENTICAL
- The projection (`out/census_projection.json`): central 43312 s (12.03 h); times 07b's shortfall 1.197: 51833 s; highest margin 46285 s, times the shortfall 55392 s; within 48 h: True (times the shortfall: True)
