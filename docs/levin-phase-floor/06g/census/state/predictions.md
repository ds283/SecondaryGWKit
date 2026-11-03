# levin-phase-floor 06g section 5, the census: predictions, fixed before the run

Written 2026-11-01T11:42:36Z on mps026298 by the census agent, after Gate V held (out/cmd/20, 21) and before the launch. Not edited after the launch; anything added after it goes in another file.

Part A is prompt 06g's lines 401-422 (section 5, "Predictions, fixed before the run", V2-V8), byte for byte, from /Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-1c5336e/prompts/levin-phase-floor/06g-leaf-values-and-node-sum.md (sha256 f20d7c9486bca452adf2971edab9a6de86d37214b04e66377096dc8697baf3ea) at 1c5336e. Part B fixes how they are read on the two checkpoints (Part B's census brief).

## Part A

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

## Part B — readings fixed before the launch (census agent; written after Gate V, before the launch)

These fix how Part A's rules are read on the two checkpoints. They add no rule. Where a reading here and
Part A's words part, Part A is the rule; the reading is recorded beside it.

**The two checkpoints.** "06h" is run `levin-phase-floor-06h-census-20261101T005037`, `checkpoint.jsonl`
SHA-256 `3104c18e…`, read only. "g" is this census's run. A unit is `g-<rtol>-<shard>-<serial>` in both; units are
matched by (rtol, shard, serial).

**1. The stationary-point rows.** 06a's 720, identified exactly as 06h's `analyse06h.py` (`scan06a_sp`) and 06e's
did: the (shard, serial) of every 06a 1e-7 unit (run `levin-phase-floor-06a-census-20261027T023143`) whose scan
has a call with `n_sign_changes > 0`. The count must be 720, or the scoring stops. The same 720 (shard, serial)
are excluded at both rtols.

**2. Matching calls.** Within a matched unit, a call is matched by (kind, index): kind "group" for the unit's
`calls`, "analytic" for its `other_calls`; `index` is the call's position among the unit's
`adaptive_levin_sincos` calls. A match also requires an equal `notify_label` and `x_span` equal by bit pattern
(hex). A call in one checkpoint only, or whose label or span differs, is unmatched: every one is listed with its
unit, kind and index. Where the call lists differ in length or order, only the calls matched this way are
compared, and the rest are listed as unmatched; no call is matched by position alone.

**3. V8 — scored calls and causes.**
- Scored: every matched call outside the 720 whose 06h `converged` is True. Lost: its g `converged` is False.
  Gained (listed, not scored): 06h False, g True. Both kinds of call are scored.
- **The cause of a lost call**, from g's record of that call. Two facts are read first:
  - **its ending:** "`node_limited`" if `node_limited` is True; else "depth cap" if `n_at_depth_max` ≥ 1 (a
    returned leaf at depth ≥ its `depth_max`, 20: refinement reached the cap); else "other ending";
  - **(b):** whether `checked_06g.n_b_untrusted` ≥ 1: a leaf the trust check was taken on (returned or later
    bisected) with 1/4 ≤ r < 1 and Δ1 > φP (`delta1 > phase_err`), which 06g's trust test untrusts and 06h's
    (r < 1) trusted — a non-noise slow leaf.

  The call's class is then, in this order:
  1. "`node_limited`" — the ending is `node_limited` (whatever (b) says; (b) is reported beside it);
  2. "(b), refined to the depth cap" — the ending is the depth cap and (b) holds;
  3. "depth cap, without (b)" — the ending is the depth cap and (b) does not hold;
  4. "(b), below the depth cap" — (b) holds and no returned leaf is at the cap;
  5. "other" — none of these;
  and "unreadable" if the call's `checked_06g` or `error_06g` records an error.

  **Against Part A's three causes:** "(b) on a non-noise leaf" is classes 2 and 4; "`node_limited`" is class 1;
  "other" is classes 3 and 5. Both levels are reported. For each lost call the table also gives
  `num_untrusted_regions` (untrusted leaves returned, each at the cap), `n_at_depth_max`,
  `n_phase_limited_regions`, `max_depth`, regions and solves under both, `premise_06g` (checks, firings, halving
  failures, split-could-meet-budget) and 06h's `premise_checks`/`premise_failed`, so that classes 3 and 5 are
  described, not left blank.
- Rows are counted per rtol by cause (a row with lost calls of two causes is counted under each, and once in the
  total).
- **Over budget.** A 06h-converged call that g did not complete because its unit stopped over budget is "unread".
  The over-budget pass's re-run (3,600 s), where it completes, supplies the call; it stays over budget for V5. If
  any call is unread after the pass, V8 at that rtol is **undecided** unless it already fails on the calls read.
- An unmatched 06h-converged call outside the 720 makes V8 at that rtol undecided unless it already fails.

**4. V2.** Per rtol, every row outside the 720 that is `done` in both with a finite `total_abserr` in both:
|total_g − total_06h| ≤ total_abserr_06h + total_abserr_g, evaluated exactly (`Fraction` of the recorded
doubles). "The same on the oracle": the oracle's value and declared error under each are `analytic_06h.h`'s
`value` and `abserr` (06h recorded them per unit), on rows with a finite oracle abserr in both. Rows without a
result under either, or with a non-finite error, are counted and listed, not scored. Every violating row is
listed with both values and both errors in `float.hex`; the largest ratio is reported on each.

**5. V5.** Outside the 720, at each rtol: false if any g call has `n_at_depth_max` > 0 (a returned leaf at depth
≥ its `depth_max`, 20), or any unit is over budget (whatever its re-run gives). Inside the 720, both are
reported, not scored.

**6. V6.** `g-1e-8-0-33995` call 5 is the call of index 5 in that unit, matched to 06h's by rule 2. Its ending is
"converged" if `converged`, "`node_limited`" if `node_limited`, "neither" otherwise; it holds if
`max_depth` < `depth_max` and the ending is not "neither". Reported under both: ending, depth, regions,
`node_error_signed_sum`, `node_error_abs_sum` (06h: Σ|E| from `cc_floor.sums_all_cc_leaves.abs_E`, and ΣE as
`math.fsum` of the pickle's `cc_floor_terms` E over the call's reproduced leaves).

**7. V7.** `g-1e-8-3-1072` call 13 ("analytic J1"), matched by rule 2. Holds if its g `converged` is True.
Reported under both: converged, abserr, the request max(atol, rtol|value|), regions, depth, premise checks and
firings (g: `premise_06g` and the pickle's per-check detail; 06h: `premise_checks`, `premise_failed`).

**8. The descriptive items** (not scored), from the same matched calls outside the 720 unless said otherwise:
- **The values.** |v_g − v_06h| / abserr_g per call with finite abserr_g, split by cause: "different partition"
  (the returned leaves' spans differ between the two pickles), "same partition, quarters returned"
  (`values_06g.n_quarters` > 0), "same partition, no quarters". The abserr moves are split by `has_cc_leaf`
  (ΣE in the error).
- **Slow leaves.** Test 3's eight classes summed over every checked leaf (`checked_06g.classes_all`) at each rtol,
  and over the returned ones; for calls with `n_b_untrusted` ≥ 1, the regions and solves added against 06h.
- **The node sum.** Σ|E| / |ΣE| per call where `node_error_abs_sum` ≥ 0.01 × abserr_g.
- **The exit.** Firings and premise checks under 06h (`premise_checks` − `premise_failed`, `premise_checks`) and
  g (`premise_06g.n_fired`, `n_checks`); every call whose firing count changed.
- **Cost.** Regions, solves and `wall_s` against 06h; the depth distribution and converged counts at both rtols.
- **The 720 rows.** Totals and convergence moved, at each rtol.
- **The near-singular class.** 06h's rule, |K|/(k + q c_s + r c_s) ≤ 10⁻³ (`nearsingular06h.py`): convergence and
  honesty movement there, reported as such.
