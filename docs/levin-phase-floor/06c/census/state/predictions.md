# levin-phase-floor 06c, section 5 (the census): predictions, fixed before the run

Written 2026-10-29T12:58:35Z on mps026298.phys.susx.ac.uk, after Gate X0 held (out/cmd/08, 09 and 14) and before the
launch. Part A is prompt 06c's §5 "Predictions, fixed before the run" and X1–X5, copied unchanged
(lines 363–392 of `prompts/levin-phase-floor/06c-global-error-control.md` at `e5764a6`; out/cmd/16
checks the bytes). Part B fixes X1's selection rule, so that it determines the 24 calls
mechanically given the run, and the readings each rule is scored by. Nothing in Part B changes a
rule of Part A; where Part A is silent, Part B says what is counted. This file is not edited after
the launch; anything added then goes in another file.

## Part A — prompt 06c §5, unchanged

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
"Moved" means the call's value, abserr, region count or maximum depth differs from 06b's.
- **X1 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that moved, with no sign change of $\theta'$ in 06a's scan
    (`spscan.py`): 12 that 06b reported unconverged with no region at `depth_max`, none
    phase-limited and a finite abserr (or all such, if fewer), and the rest stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** W2's, Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's
    own floor reported. The two must agree to within a hundredth of $g$'s `abserr`; a call where
    they do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_g - {\rm ref}| \le$ `abserr`$_g$ for every call left in the sample.
- **X2 — budget misses converge.**
  - **The rule:** every Levin call that 06b reported unconverged, with no region at `depth_max`, no
    phase-limited region and a finite abserr, is `converged` under $g$, outside 06a's stationary-
    point rows.
  - **Holds** if true at both rtols.
- **X3 — no infinite declared error.**
  - **The rule:** no call under $g$ declares an infinite abserr unless one of its leaves is at
    `depth_max`.
  - **Holds** if true at both rtols.
- **X4 — the rows agree.**
  - **The rule:** outside 06a's 720 stationary-point rows, every row with finite `total_abserr`
    under both has $|{\rm total}_g - {\rm total}_{06b}| \le {\rm total\_abserr}_{06b} + {\rm
    total\_abserr}_g$.
  - **Holds** if true at both rtols.
- **X5 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

## Part B — X1's selection rule, and the readings by which X1–X5 are scored

**Terms.**
- *06b* is run `levin-phase-floor-06b-census-20261028T161212` (its checkpoint, SHA-256
  `f934ea24…`, and its per-worker pickles), read only. *g* is this run, at the worktree `e5764a6`
  (code `ad9a5f4` plus `3d70db9`). A g unit `g-<rtol>-<shard>-<serial>` is compared with 06b's
  `b-<rtol>-<shard>-<serial>`. Calls are matched by `index` with their `notify_label` equal; a
  call that cannot be matched so, or a unit whose number of calls differs, is reported, and
  counted against the prediction that needs it as "undecided" unless the prediction fails anyway.
- *A Levin call* is one `adaptive_levin_sincos` call of a row: its group calls (the checkpoint's
  `calls`) and its analytic blocks' calls (`other_calls`). Both runs recorded both.
- *Moved* is the prompt's: the call's value or abserr differs from 06b's by bit pattern (hex), or
  its `num_regions` or `max_depth` differs.
- *depth_max* is the driver's default, `DEFAULT_LEVIN_MAX_DEPTH = 20` (QuadSourceIntegral passes
  none; each call's record carries the `depth_max` it ran under). A call *reaches* depth_max if a
  leaf has depth ≥ depth_max (`max_depth ≥ 20`, equivalently `n_at_depth_max > 0`).
- *06b reported unconverged with no region at depth_max, none phase-limited and a finite abserr*
  (X1's first stratum, X2's premise): the 06b call has `converged` false, `max_depth < 20`,
  `n_phase_limited_regions == 0`, and a finite `abserr`. Call such a 06b call *X2-type*.
- *06a's stationary-point rows* are the 720 rows (by shard and serial) whose 06a unit at `1e-7`
  (run `levin-phase-floor-06a-census-20261027T023143`) has a group call whose scan has
  `n_sign_changes > 0` (06b's reading, `analyse06b.py`). The same rows are excluded at both rtols.
- *No sign change of θ′ in 06a's scan*, for a call: 06a's `1e-7` unit of the row has a scan entry
  for the call's `index` with `n_sign_changes == 0`. 06a scanned group calls only, so only group
  calls can be in X1's sample.
- *Over budget* is a unit still over budget after 06a's over-budget pass: after both sets, every
  over-budget unit is re-run once at g, with 3,600 s (if more than twelve units are over budget,
  stop and ask before the pass). A unit that completes in the pass is scored on the pass's record.
  An error in a unit (not over budget) is a stop (prompt 06c §7).

**X1. The selection rule.**
1. *Eligible:* a group call at `1e-7` whose g unit is `done` in this run, which moved, whose
   06a scan shows no sign change of θ′, and whose label matches 06b's.
2. *Category 1:* the eligible calls whose 06b call is X2-type, on rows of any band (low, mid,
   seam). *Category 2:* the other eligible calls, on rows whose `rows.json` band is `low` or
   `mid`.
3. *Order.* A stratum is a (band, `k_key`) pair. Strata are ordered band first (low, mid, seam),
   then by the row's `k` ascending (numerically). Within a stratum, calls are ordered by the
   SHA-256 (hex, ascending) of the string `"<g unit id>:<call index>"`.
4. *Category 1's draw:* $n_1 = \min(12, |\text{category 1}|)$ calls, round-robin over its strata
   in order: one call from each stratum that has one left, pass after pass, until $n_1$ are taken.
5. *Category 2's draw:* $n_2 = 24 - n_1$ calls, $\lceil n_2/2\rceil$ from low and
   $\lfloor n_2/2\rfloor$ from mid. Within a band, round-robin over its strata in order, as in 4.
   If one band has fewer than its share, the shortfall is taken from the other band by continuing
   its round-robin. If fewer than $n_2$ are eligible, all are taken and the shortfall reported.
6. *The inputs* (`scripts/x1_capture.py`, after the run, at g): each selected call's unit is re-run
   in a fresh census worker with the call in its capture list; the worker pickles the call's span,
   `f`, `theta` and keyword arguments as the call saw them. The capture counts only if its value
   and abserr (hex), `num_regions`, `max_depth` and `converged` equal this run's call. Gate X0's
   third clause (two runs of one g unit bit-identical) is why they are expected to.

**X1. The reference and the score** (`scripts/x1_reference.py`, at g, on each captured call).
First the call is re-run from its reloaded inputs and must equal this run's value and abserr bit
for bit. The span is cut into $2^9$ and into $2^{10}$ equal pieces by repeated bisection at the
driver's own break point, `(e + s) / 2.0`; each piece is evaluated once by
`_adaptive_levin_subregion` with the call's `chebyshev_order` and `rtol`, as the driver evaluates
a region (a piece below the Levin gate takes the Clenshaw–Curtis branch, with 06d's floor); the
values are summed (`math.fsum`), and each piece's `phase_err` is kept and their sum and maximum
reported. $S_9$, $S_{10}$ are the sums; the reference is $S_{10}$. A call with
$|S_9 - S_{10}| > $ `abserr`$_g/100$ is reported and dropped, as the prompt says; no replacement is
drawn. X1 **fails** if $|{\rm value}_g - S_{10}| > $ `abserr`$_g$ on any call left; **undecided**
if no call fails but a call's capture does not match, its re-run does not reproduce it, its
reference could not be computed, or no call is left; **holds** otherwise. 06b's value and abserr
for each call are reported against the same reference (descriptive).

**X2.** At each rtol, every Levin call (group and analytic) whose 06b call is X2-type, on a row
outside 06a's 720 stationary-point rows, must have `converged` true in g's record. **Fails** if
one does not, at either rtol; **undecided** if none fails but a unit holding such a call is over
budget; **holds** otherwise.

**X3.** At each rtol, on every row (the stationary-point rows included), every Levin call of g
whose `abserr` is `+inf` must have a leaf at depth_max. **Fails** if one has none; **undecided** if
none fails but a unit is over budget; **holds** otherwise. A NaN `abserr`, if any, is not infinite;
each is reported beside the verdict, with what the verdict would be if it were counted.

**X4.** At each rtol, on every row outside the 720 whose `total_abserr` is finite in both 06b and
g: $|{\rm total}_g - {\rm total}_{06b}| \le {\rm total\_abserr}_{06b} + {\rm total\_abserr}_g$,
evaluated exactly on the recorded doubles (`fractions.Fraction` of each hex value), so that no
rounding of the test decides it. Rows with a non-finite `total_abserr` in either are counted
separately. **Fails** if a row violates it; **undecided** if none does but a unit is over budget;
**holds** otherwise.

**X5.** At both rtols: no Levin call (group or analytic) that moved reaches depth_max. **Fails** if
one does, or if any unit is over budget; **holds** otherwise.

**What else is fixed now.** 06b's three depth-20 calls are `b-1e-8-2-1704` call 16,
`b-1e-8-2-13964` call 5 and `b-1e-8-3-34987` call 9; they are reported by name under g whatever
their depth. "A residual flat over the last five levels" is read from each depth_max call's
recomputed deepest line (the leftmost leaf at the call's greatest depth; worker.py `_deep_line`)
as the step-(4) residual (or the nested pair, on a fallback node) of the line's last five levels;
the ratios $(e_L + e_R)/e_P$ along the whole line are tabulated, with $e$ = max(residual, floor) +
truncation, unwidened. The trust check's descriptive item reads each returned leaf's `delta1`,
`delta2`, `convergence_ratio`, `trusted` and `declared_resolution`, the call's
`num_untrusted_regions`, and the loop's own counters (`budget`: checks, passes, leaves found
untrusted) and its solves by purpose (`solves_by_purpose`). None of these is scored.
