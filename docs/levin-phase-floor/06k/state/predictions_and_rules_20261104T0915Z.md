# levin-phase-floor 06k — the predictions, Gate K0's draw rule and seed, and the operational definitions

Written by the 06k agent on mps026298 before any K0 unit, any Part L probe and any census was run, and
before 06g's census checkpoint was read for any purpose but its SHA-256 (the Gate). Made read-only once
written; a correction is a new file, and a deviation.

## 1. The predictions (prompt 06k §4, copied verbatim)

## 4. Predictions, fixed before the run

README §5 rule 3. Copy them into the log unchanged. Each is scored per route, **holds**, **fails**
or **undecided**, against the rule as written. None is a stop. In each rule, "outside the 720"
means outside the 720 stationary-point rows.

- **L1 — the route does not trust 06h's kind of leaf.** **Rule:** of the 17 06h carrying leaves
  (L-b), the route trusts or leaves none.
- **L2 — the route reaches the slow leaves.** **Rule:** the route trusts or leaves at least 80 % of
  06j's 142 sampled slow leaves (L-a).
- **C1 — convergence comes back.** **Rule:** at least 90 % of V8's lost calls outside the 720, at
  each rtol, converge under the route.
- **C2 — nothing converged is lost.** **Rule:** no call outside the 720 that 06g converges is
  unconverged under the route.
- **H1 — no referenced call gets worse.** **Rule:** on every referenced call outside the 720,
  $|v_{\rm route} - {\rm ref}|/e_{\rm route} \le \max(1,\ |v_g - {\rm ref}|/e_g)$.
- **S1 — no move past both errors.** **Rule:** no call outside the 720 moves by more than
  $e_{\rm route} + e_g$ against 06g's census.
- **K1 — routes 1 and 2 fire only on noise.** **Rule:** every leaf the route trusts that 06g
  refined has $\Delta_1/\phi_P < 5.8$ (c07's carrying leaf, the smallest of 06h's kind). Scored for
  routes 1 and 2. Route 3 decides by size, not by noise. Its firings are reported by
  $\Delta_1/\phi_P$ and by $D$ against the request, and K1 is not scored for it.

**The basis.**
- **L1, H1, S1 and K1** are the safety constraint of §0.1, stated four ways: at leaf level on the
  known bad leaves, on the references, census-wide, and on every firing. Route 3's safety rests on
  L1, H1 and S1 alone, and on $\eta$'s margin.
- **L2 and C1** are what the remedy is for. The 80 % follows 06j; 90 % leaves room for calls held
  by something other than slow leaves.
- **C2** holds by construction if the route only adds trust, and is a check that it does.

A route passes as a candidate remedy if L1, C2, H1 and S1 hold, and K1 for routes 1 and 2. L2 and
C1 say how much it recovers. Every number behind each verdict is reported, so that a different
reading can be made from the record.

## 2. Gate K0's units: the draw rule and its seed

K0 runs on 100 units.

- **U46, 06j's 46 calls' units.** The `g_unit` of every entry of 06j's `out/sample/selection.json`
  (`docs/levin-phase-floor/06j/out/sample/selection.json`, the committed copy; 46 entries on 46
  distinct units, 23 at each rtol).
- **U54, 54 more, drawn by this rule:**
  - **Population:** the 18,560 units of 06g's census checkpoint
    (`var/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl`, SHA-256 `03adcbba…`),
    less U46.
  - **The 720:** 06a's stationary-point rows, as 06g's `v8_06g.py` forms them (`scan06a_sp()`: the
    rows of 06a's run's 1e-7 checkpoint with a scan of `n_sign_changes > 0`), which must number 720.
  - **The V8 units:** the units holding at least one lost call of `v8_calls.json`
    (`var/levin-phase-floor-06g/census/out/v8_calls.json`, SHA-256 `1e9a62b0…`), the "lost" list.
  - **Three strata per rtol, disjoint, 9 units each (27 per rtol, 54 in all):**
    1. `V8out`: V8 units whose row is outside the 720;
    2. `SP720`: units whose row is in the 720;
    3. `rest`: every other unit.
  - **The draw:** in each (rtol, stratum) the candidate uids are sorted as strings ascending, and 9
    are drawn without replacement by
    `numpy.random.Generator(numpy.random.PCG64(seed)).choice(n, size=9, replace=False)`, with
    `seed = 20261104 * 100 + 10 * s + t`, `s` = 1, 2, 3 for `V8out`, `SP720`, `rest`, `t` = 1 for
    1e-7 and 2 for 1e-8. The drawn indices are taken in the order `choice` returns them.
  - A stratum with fewer than 9 candidates is a stop.
- **Each K0 unit is run as 06g's census ran it:** the unit spec of `run_census.plan` (budget 300 s;
  `scan` and `w3` at 1e-7), in a census worker (`worker.py`) at the tree, grouped one worker per
  (variant, rtol, k_key), its units in rows.json's order; at most six workers at once.

## 3. Operational definitions, fixed now

- **Loading a variant.** Before anything imports `AdaptiveLevin`, the worker executes the variant file
  as the module `AdaptiveLevin.levin_quadrature` (in `sys.modules`), then imports the package, whose
  `__init__` binds `adaptive_levin_sincos` from it. Every production import, and every wrapper the
  harness installs at `LQ.<name>`, then sees the variant. The identity variant is a byte-identical
  copy of the tree's file loaded the same way.
- **A firing** is a trust check (routes 1 and 2) or a stop test (route 3) at which the route trusts or
  leaves a leaf that 06g's rule (b) finds untrusted: refined if it is refinable, returned untrusted if
  it is at `depth_max`. Each firing is recorded with the leaf's depth, $r$, $\Delta_1$, $\phi_P$, and
  the route's own numbers ($\nu_1$; $\Delta_{1c}$, $\Delta_{2c}$, $r_c$; $D$ and the request).
- **Route 1** trusts if $r < 1/4$, or $r < 1$ and $\Delta_1 \le \nu_1$,
  $\nu_1 = (\phi_P + \phi_L) + \phi_R$, each the solve's `phase_err` as `_make_leaf` reads it.
  $\nu_1 \ge \phi_P$, so it only adds trust; everything else is 06g's.
- **Route 2, as built:** the corrected rule is evaluated where 06g's rule (b) finds the leaf
  untrusted. If it trusts, the leaf is trusted, declares $\max(\text{residual},\Delta_{1c})/(1-r_c)$
  and returns $Q$ (uncorrected); a leaf 06g's rule trusts keeps 06g's decision and declaration.
  This is the reading under which Gate K0's second clause and §0.2's "06g's trust for any leaf with
  $r < 1/4$" unchanged can hold. **The literal reading** (every checked leaf decided and declared on
  the corrected numbers) is measured at leaf level in Part L on every leaf (L-a, L-b, L-c), and is
  put to the orchestrator at planned stop 1.
- **Route 3, as built:** at each stop test (the budget passed, every leaf checked or floor-limited),
  the untrusted leaves that are refinable (below `depth_max`, not floor-limited) are ranked by $D$
  ascending (ties by start, then sequence number), and the longest prefix whose $D$, summed
  correctly rounded (`math.fsum`), is at most $\eta$ times the request
  $\max(\text{atol}, \text{rtol}\,|V|)$ of the returned sums in ascending x is the left set. The left
  set is recomputed at every stop test over every untrusted refinable leaf, those left before
  included. A left leaf declares $D$, is not a candidate for refinement and does not block the stop;
  the call is converged if its budget is met and no returned leaf is untrusted other than the left
  ones. An untrusted leaf at `depth_max` is not ranked (the prompt's "untrusted refinable"): it keeps
  06g's declaration and keeps the call unconverged. $\eta = 10^{-3}$.
  - **The smallest $\eta$** recorded per call is
    $\eta_{\min} = \sum_{\text{left}} D / \text{request}$ on the returned partition (0 if none left),
    and, beside it, the same with the untrusted leaves at `depth_max` added
    (the other reading of "untrusted refinable"); both are put to the orchestrator at stop 1.
- **K0's comparison.** Every key of each unit's checkpoint record and of its region-pickle record is
  compared with 06g's, floats by bit pattern, arrays element by element with NaN equal to NaN,
  except timing (`wall_s`, `rebuild_s`, every key ending `_s`, `seconds`, `elapsed`, `compute_time`),
  provenance (`tree_head`, `tree_status_short`, `scripts_sha256`, `worker`, `worker_pid`,
  `worker_tree`, `pickle`, `pickle_offset`, `pickle_length`, `pickle_record_sha256`, `modules`,
  `rows_sha256`, `w2_capture_sha256`) and the new key `route_06k`. Every differing path is listed.
  Clause 2 compares per call (the call's summary, its region arrays, checked leaves and premise
  checks) for the calls with no firing; the unit-level keys only on units with no firing in any
  call.
