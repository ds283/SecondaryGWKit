## Part B — the readings by which H3 and H4 are scored, and what the descriptive items count

(Draft, written at preparation on 2026-11-01 before any computation of this census: no store hash,
no fingerprint, no unit, no Gate H. It becomes Part B of `state/predictions.md` after Gate H and
before the launch. Part A is prompt 06h §5's "Predictions, fixed before the run" with H1–H4,
lines 334–347 of `prompts/levin-phase-floor/06h-group-phase-rounding.md` at `c1ec3b2`, copied byte
for byte and checked by `scripts/check_predictions.py`. H1 and H2 are the phase agent's
(Part B's phase brief); this census scores H3 and H4 only. Nothing here changes a rule of Part A;
where Part A is silent, this says what is counted.)

**Terms.**
- *06e* is run `levin-phase-floor-06e-census-20261030T081318` (its checkpoint, SHA-256
  `a0dc4d7f…`, and its per-worker pickles), read only. *h* is this run, at the worktree `c1ec3b2`
  (code `7172a8f` + `ac1c6f4`), whose scripts call the tree `g`. Both runs name their units
  `g-<rtol>-<shard>-<serial>`; a unit of this run is compared with 06e's unit of the same rtol,
  shard and serial (the same row of `state/rows.json`, `aa523d28…`).
- *A Levin call* is one `adaptive_levin_sincos` call of a row: its cosmological group calls (the
  checkpoint's `calls`) and its analytic blocks' calls (`other_calls`, labels "analytic J1" …
  "analytic Y4"). Calls are matched across the runs by **row, rtol and call `index`**, and the match
  holds only if the two calls' `notify_label`s are equal and their `x_span`s are equal bit for bit.
  Where a unit's call lists differ in length, the calls are matched by index over the shorter list
  under the same test, and every call without a match is listed. A call that cannot be matched is
  reported and counted as undecided for the prediction that needs it, unless that prediction fails
  anyway.
- *The stationary-point rows* are 06a's 720 (prompt §5; `spscan.py`): the rows, by shard and serial,
  whose 06a unit at `1e-7` (run `levin-phase-floor-06a-census-20261027T023143`) has a group call
  whose scan has `n_sign_changes > 0` — the reading of 06b, 06c and 06e (`analyse06e.py`,
  `scan06a_sp`). The count is checked to be 720. The same rows are excluded at both rtols.
- *Over budget* is as 06e's Part B read it: a unit recorded `over budget` by the dispatcher (300 s,
  raised at the first `PrimitivePhase.theta_mod_2pi` entry past the deadline). After both sets,
  every over-budget unit is re-run once at h with 3,600 s (06a's over-budget pass, as 06c and 06e
  would have run it). If more than twelve units are over budget, I stop and ask before the pass
  (06c's and 06e's threshold, from 06a's pass being sized for at most twelve units). A unit that
  completes in the pass is scored on the pass's record; one still over budget stays over budget.
  A unit `done` with `wall_s` > 300 s (the dispatcher's blind spot: the analytic blocks call no
  `PrimitivePhase`) is not over budget; each is reported. A unit in error (not over budget) is a
  stop (prompt 06h §7). A Ray session started by a census process is a stop.

**H3.** At each rtol, on every row outside the 720 whose `total_abserr` is finite in both 06e and h:
$|{\rm total}_h - {\rm total}_{06e}| \le {\rm total\_abserr}_{06e} + {\rm total\_abserr}_h$,
evaluated exactly on the recorded doubles (`fractions.Fraction` of each `float.hex`). Rows with a
non-finite `total_abserr` in either are counted separately. **Fails** if a row violates it (every
violating row is listed with both totals and both `total_abserr` in `float.hex`); **undecided** if
none does but a unit outside the 720 is still over budget after the pass (it has no total);
**holds** otherwise, at both rtols.

**H4.** At each rtol, every Levin call (group and analytic) on a row outside the 720 whose 06e record
has `converged` true must have `converged` true in h's matched record. **Fails** if one does not.
**Undecided** if none fails but: a unit outside the 720 is still over budget after the pass and a
call 06e reported converged has no completed counterpart under h (an over-budget unit's completed
calls are recorded and scored; the call in flight and those after it cannot be read, and count as
unresolved); or a call 06e reported converged cannot be matched. **Holds** otherwise, at both
rtols. Every call that lost convergence and every call that gained it is listed (unit, index,
label, 06e and h value, abserr, regions, depth).

**What else is fixed now (descriptive, not scored).**
- *What the declaration costs* (06f's `declared_vs_request_06f.py` method, on both checkpoints).
  Per analytic-block call, the request is $\max({\rm atol}, {\rm rtol}\,|v|)$ with the call's own
  `atol`, `rtol` and value.
  - **The Clenshaw–Curtis branch, under 06e and h alike:** the declared term summed over the call's
    floor-attributed fallback leaves (`cc_floor.sums_floor_attributed_cc_leaves.declared`, calls
    with `n_floor_attributed` > 0), and over all its fallback leaves
    (`cc_floor.sums_all_cc_leaves.declared`), against the request: 06f's counts (declared-dominated
    calls; calls with a declared term > 0; D/request ≥ 1, ≥ 0.1, ≥ 0.01; unconverged), and the
    distribution of the ratio of h's sum to 06e's on matched calls.
  - **The Levin branch, under h only:** 06e's checkpoint does not record a Levin leaf's declared term
    (its `phase_err` is the eq. (151) floor plus the declared term, recorded together). Under h,
    `declared_06h.levin` gives the declared term summed over the call's Levin leaves, all and
    floor-attributed, and split by the three parts of `theta_abserr` (constituents, abscissa,
    $\phi_{\rm eval}$), each against the request. The constituents' share is reported beside it as
    what 06e's declaration (the constituents' sum alone) would charge on h's leaves with h's |p|: a
    counterfactual on h's leaves, not 06e's number.
  - Also under h: `declared_06h.cc`, the same split on the fallback branch.
- *Convergence and depth:* converged calls (group and analytic) and rows, `total_converged`, and the
  row depth distribution (max over the row's Levin calls) and the call depth distribution under 06e
  and h, at both rtols, inside and outside the 720.
- *Cost:* regions (`num_regions`), solves (`num_subregion_solves`) and unit wall time (`wall_s`)
  summed, against 06e, split into group and analytic-block calls; the run's wall time against 06e's
  1 h 47 min, and unit walls summed against 06e's 2.28 h and 2.97 h. Also: group calls bit-identical
  between 06e and h (06h's code changes only the Bessel group phase, which the cosmological group
  calls do not use), counted, and any that is not listed.
- *The stationary-point rows:* how the 720 moved against 06e at both rtols: totals moved, and by more
  than `total_abserr`$_{06e}$; convergence gained and lost; depth.
- *The near-singular class:* the LiouvilleGreen draws' near-singular cases are three-Bessel calls
  whose third wavenumber lies within $\epsilon \le 10^{-4}$ of $|k-q|$ or $k+q$, i.e. a sign group
  with $|K| = |k + e_q\,q c_s + e_r\,r c_s|$ small against the coefficients. For each analytic-block
  call of h, $|K|/(k + q c_s + r c_s)$ is formed from the row's $k, q, r$ (`state/rows.json`) and
  its $c_s = \sqrt{(1-b)/(1+b)/3}$ (the row's `b`, the pickle's `result_full`), as
  `_three_bessel_Levin` forms them; its distribution is reported, and every call with
  $|K|/(k + q c_s + r c_s) \le 10^{-3}$ is reported as reaching the near-singular class, with its
  H3 and H4 standing and its declared term against the request. None at or below $10^{-3}$ means A3
  v2 does not reach the class, and that is said.
