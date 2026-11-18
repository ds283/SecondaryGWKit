# levin-phase-floor 07b §6 — the census's predictions, fixed before the census

Provenance (the writer's header):

```
levin-phase-floor 07b section 6: the census's predictions, fixed before the census
written 2026-11-16T21:03:31Z on mps026298.phys.susx.ac.uk
script write_predictions_07b.py sha256 50e27a06c59f0f7a060d14c457160fc2fa90a1169b6b3443dd3c505e04ad6b79
tree h: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5 HEAD 0a103b5ae0de32b568f44904a5c3dcbfca922a2e (status --short: empty)
main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
input /Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5/prompts/levin-phase-floor/07b-levin-leaf-beside-a-window.md sha256 33c0a19505e0cf8747c76738f89280cab8ff96544db9861ba0bb1fd400f078fb
```

Written after Gate U0 (all four clauses) and before the census is launched. Not to be rewritten: a correction is a new file and a deviation.

Source: `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5/prompts/levin-phase-floor/07b-levin-leaf-beside-a-window.md` (file SHA-256 33c0a19505e0cf8747c76738f89280cab8ff96544db9861ba0bb1fd400f078fb). Parts 1-3 are copied **by text** (each from a fixed opening string to a fixed closing string, both unique in the file; the line numbers are given for reference only) and checked to be substrings of the prompt. Parts 4 and 5 are this agent's.

## 1. Prompt §6, the predictions and the descriptive items (verbatim; prompt lines 607-622)

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md` unchanged,
with §9's additions:
- **U1 — nothing else moves.** Every call in which the fix does not act equals 07a's call bit for bit,
  in the call's outputs. Every row all of whose calls are such equals 07a's in `total` and
  `total_abserr`. **Holds** if true at both rtols.
- **U2 — no Levin leaf is returned across a stationary point** (07a's W2).
- **U3 — every phase with something to locate is located** (07a's W3).
- **U4 — honest where it is referenced.** Every call referenced in Part A (V4's 25 and 07a's census's
  other eight V4 calls, R1, R2, R3, and `1-34599` and `3-678`), scored from the census against its
  reference by 07's rule. **Holds** if every one is honest, or unconverged under §9's rule.
- **U5 — the gate is consistent, and the guard is a backstop** (07a's W5, on gated solves).
- **U6 — 06b's 24 are honest,** against $S_{10}$ by 06b's rule, or unconverged.
- **U7 — the pairs:** $r_g$ over all 2,484 pairs; every pair has $r_g \le 1$, or a member unconverged.

**Descriptive, not scored:** the calls where the fix acts, by class, band, $k$ and rtol; `converged`
changes in each direction; the abserr ratio against 07a's; regions and cost against 07a's.

## 2. Prompt §9.5, the additions to §5's and §6's predictions (U8 among them) (verbatim; prompt lines 725-748)

### 9.5 Additions to §5's and §6's predictions

- **U1 counts** a call untouched if `num_edge_rule_refusals + num_range_rule_refusals` is 0, and compares
  value, abserr, `converged`, `num_regions` and the maximum depth. Expected to cover about a fifth of
  the calls (R2: 11 of 60).
- **U4 is predicted to hold,** with exactly v04 and R2-28 (both rtols) unconverged among the referenced
  calls, R2-28 at 1.107 of its abserr.
- **U6 is predicted to hold.** The four of 06b's 24 dishonest at 07a are honest under (h)'s replays;
  the other 20 were not replayed, and the prediction for them rests on (h) leaving no referenced call
  converged and dishonest.
- **U7 is predicted to hold:** both pairs pass under (h)'s replays; the other 2,482 were not replayed.
- **U8 — the code is (h).** Every census call that Part A replayed (§2.1's and §2.4's, both rtols)
  equals (h)'s replay record bit for bit in value, abserr, `converged`, regions and maximum depth.
  **Holds** if all do.
- **S3 (§5)** names no row. A row beyond S3's bound is reported with the refusals in its calls, and S3
  is scored as written.
- **Descriptive, added:** the refusals by rule, class, band, $k$ and rtol, and, by the harness's
  instrument (marked "prompt 07b"), each refused leaf's span, depth and $q$; every call newly
  unconverged with its depth-20 leaves' routes and whether each descends from a refused leaf (X3's
  measures, for the issue in §9.6); regions and wall against 07a's.

**Part B's review** replays v08, v13 (a control the edge rule acts on), R2-22 (untouched: 45 trust checks, all
passing the range rule) and R2-03 from copies of their captures at the code commit: each equal to (h)'s
replay record in `float.hex`, and R2-22 equal to 07a's census too.

## 3. Prompt §9.7, the stops added (for reference; not predictions) (verbatim; prompt lines 765-769)

### 9.7 Stops added (into §8)

- In Part B: an output of the code differing from (h)'s replay record on a call the review replays; no
  dishonest-and-converged call found at the parent for §3 (b) or (c).
- In Part C: the census's projection beyond 48 hours; U8 failing, before anything else is scored.

## 4. How each rule is read (this agent's, fixed here before the census)

- **U1.** A call is untouched iff `num_edge_rule_refusals + num_range_rule_refusals` is 0 (§9.5), the two counts as the
  call's result returns them (the census records them on every call; None would be a fault). Each untouched call is
  compared with 07a's census call (the same unit `h-<rtol>-<shard>-<serial>`, the same kind and index) in value,
  abserr, `converged`, `num_regions` and `max_depth`, floats by `float.hex`. **Holds** if every untouched call is equal
  in all five at both rtols, and every unit all of whose Levin calls (group and analytic) are untouched equals 07a's in
  `result.total` and `result.total_abserr` by hex. Beside it, descriptive: each untouched call's whole record diffed
  against 07a's by category (`call`, `location`, `harness`, `oracle`), compare_07b.py mode `calls`. A unit not `done`
  in either census is listed and not compared. **Stop** (§8): any untouched call differing in the five, before anything
  else is scored.
- **U8.** The calls are the 187 Part A replayed at both rtols, `var/levin-phase-floor-07b/measurement/state/
  census_wanted_07b.json` less `i3-34599-07` (replayed at 07's module): each id's census call is that file's (unit,
  kind, index). (h)'s replay record is `out/ext/replays/h/<id>.json`'s `replay` (and `h_summary_all.json` /
  `h_summary_i3.json`, checked to agree with it). **Holds** if every one equals in value and abserr (hex), `converged`,
  `num_regions` and `max_depth`. **Stop** (§9.7): any failure, before anything else is scored.
- **U2, U3, U5** as 07a's W2, W3, W5 were scored (07a's census `log.md`), U5 on gated solves only (07a's W5 ruling,
  `var/levin-phase-floor-07a/census/state/ruling_w5.md`), the forced solves' count descriptive.
- **U4.** The referenced calls and their references are Part A's (`var/levin-phase-floor-07b/measurement/out/tables/
  scores_final.json` and `tables_07b.json`: V4's 25, R1, R2, R3, item 3), 07's V4 readings for 07a's census's other
  eight V4 calls (`docs/levin-phase-floor/07/census/out/v4/v4_scores.json`), and 07a's census references for `1-34599`
  and `3-678` (`var/levin-phase-floor-07a/census/out/refs/scores.json`), each scored from this census against its
  reference by 07's rule as those records apply it. **Holds** if every one is honest, or unconverged. §9.5's prediction
  (exactly v04 and R2-28 at both rtols unconverged, R2-28 at 1.107 of its abserr) is scored beside it.
- **U6.** 06b's 24 calls (`docs/levin-phase-floor/06b/census/out/w2_reference.json`; unit `b-1e-7-X-NNNNN` is this
  census's `h-1e-7-X-NNNNN`, the same call index) against $S_{10}$ by 06b's rule, $|v - S_{10}|/{\rm abserr} \le 1$, or
  unconverged.
- **U7.** $r_g$ over all 2,484 pairs, by 07's cross-check as 07a's `xcheck_07a.py` computed it; every pair $r_g \le 1$
  or a member unconverged.
- **No new references** are taken; U4 and U6 are scored as written and attributed to nothing (the dispatch).

## 5. Gate U0 and the projection, as they stood when this file was written

- U0 (1): `out/u0/u0a_ast.txt` and `out/u0/u0a_imports.txt` (both HOLD).
- U0 (2): `out/k0/compare_u0b.json` (100 of 100 K0 units equal 07a's census at the parent).
- U0 (3): `out/u0c3/compare_instr.json` (`h-1e-8-3-14501`, where the fix acts, and `h-1e-7-1-20582`, where it does not:
  IDENTICAL on and off).
- U0 (4): `out/u0c4/compare_det.json` (`h-1e-8-3-14501` twice: IDENTICAL).
- The projection: `out/census_projection.json`.
