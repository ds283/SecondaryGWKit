## Part B — the readings by which Z2–Z5 are scored, and Z3's sample and matching rule

(DRAFT, written at preparation on 2026-10-30 before any computation; the final text goes into
state/predictions.md after Z0 and before the launch. Any difference between the two is listed in
log.md.)

**Terms.**
- *06c* is run `levin-phase-floor-06c-census-20261029T130214` (its checkpoint, SHA-256 `b61cb034…`,
  and its per-worker pickles), read only. *g* is this run, at the worktree `6d81fad` (code
  `f4d8e17`). Both runs name their units `g-<rtol>-<shard>-<serial>` (each run's own tree `g`); a
  unit of this run is compared with 06c's unit of the same rtol, shard and serial.
- *A Levin call* is one `adaptive_levin_sincos` call of a row: its group calls (the checkpoint's
  `calls`) and its analytic blocks' calls (`other_calls`). Calls are matched across the runs by
  `index` with equal `notify_label`; a call that cannot be matched so, or a unit whose number of
  calls differs, is reported and counted as undecided for the prediction that needs it, unless that
  prediction fails anyway.
- *depth_max* is the driver's default, `DEFAULT_LEVIN_MAX_DEPTH = 20` (each call's record carries
  the `depth_max` it ran under). A call *reaches* depth_max if a leaf has depth ≥ depth_max
  (`max_depth ≥ depth_max`, equivalently `n_at_depth_max > 0`).
- *The stationary-point rows* are 06a's 720 (prompt §5; `spscan.py`): the rows, by shard and
  serial, whose 06a unit at `1e-7` (run `levin-phase-floor-06a-census-20261027T023143`) has a group
  call whose scan has `n_sign_changes > 0` — 06b's and 06c's reading (`analyse06c.py`,
  `scan06a`). The count is checked to be 720. The same rows are excluded at both rtols.
- *Over budget* is as 06c's Part B read it: a unit recorded `over budget` by the dispatcher (300 s,
  raised at the first `theta_mod_2pi` entry past the deadline). After both sets, every over-budget
  unit is re-run once at g with 3,600 s (06a's over-budget pass; if more than twelve units are over
  budget, stop and ask before the pass). A unit that completes in the pass is scored on the pass's
  record; one still over budget stays over budget. A unit `done` with `wall_s` > 300 s (the
  dispatcher's blind spot, 06c's deviation 22) is not over budget; each is reported. A unit in
  error (not over budget) is a stop (prompt 06e §7).

**Z2.** At each rtol, every Levin call (group and analytic) of g on a row outside the 720 has
`max_depth < depth_max`. **Fails** if one reaches depth_max at either rtol, or if a unit on a row
outside the 720 is over budget (the prompt: "an over-budget unit counts as false"); **holds**
otherwise. Calls of the 720 that reach depth_max are reported, not scored.

**Z3. The sample** is depth20-tiling's 30 calls (`var/levin-phase-floor-06c/depth20-tiling/state/
sample.json` and `captured.json`), numbered 0–29 by `sample_no` as in its `results.md`. Referenced:
- 1, 9 and 25: not re-run. Their resolutions are the orchestrator's existing outputs,
  `orchestrator_check/02-ref-sNN.txt` (P = 128, 64/96 nodes) and `04-ref-sNN-P64.txt` (P = 64,
  96/128 nodes), whose S lines (hex) are read as they stand;
- 2, 8, 10, 13, 14, 22, 23 and 27: `references/orch_ref.py` (06c's, byte-identical, sha256
  `8a0d291d…`) run at the tree g on `references/inputs/sNN.pkl`, at `--P 128 --n1 64 --n2 96` and
  at `--P 64 --n1 96 --n2 128` (`--cell 1e-3`, `--workers 6`), in-session, one at a time;
- 3 and 15: the same two invocations each, in one registered run with slug `references`.
The other 17 are not run.

**Matching (row, rtol, call index).** depth20-tiling's call `(unit, index)` with unit
`g-<rtol>-<shard>-<serial>` (06c's census) is this run's unit `g-<rtol>-<shard>-<serial>` (same
rtol, shard and serial, so the same row) and its Levin call with the same `index` (`calls` or
`other_calls`). The match holds only if that call's `notify_label` equals the captured record's
and its `x_span` equals the captured `x_span` bit for bit. The inputs `references/inputs/sNN.pkl`
are depth20-tiling's captures (`slice_inputs.py`: sliced at `captured.json`'s offset and length,
SHA-256 checked, compared with the orchestrator's copies where it has them). **They stand for g's
call** only if the driver at g, replayed on them (`replay_g.py`, and each orch_ref.py invocation's
own replay line), returns g's census value and abserr for the call bit for bit. A sample whose
match or replay fails is reported and not counted.

**Z3. Agreement, and the score.** The three resolutions are S96 at P = 128, S96 at P = 64 and S128
at P = 64 (S64 at P = 128 is printed by orch_ref.py and reported, not used, as in the
orchestrator's `05-honesty.txt`). The reference is S128 at P = 64. A reference **counts** iff
every pair of the three agrees to within min(`abserr`$_{06c}$, `abserr`$_g$)/100, that is, their
range max − min ≤ min(…)/100, evaluated exactly on the doubles (`fractions.Fraction` of each hex
value). `abserr`$_{06c}$ is the call's abserr in 06c's census (depth20-tiling's `census_abserr`,
checked against 06c's checkpoint); `abserr`$_g$ is this run's. The orchestrator's own reading
(each of S96(P128) and S96(P64) against the reference) is reported beside it. Those that do not
count are reported with their scatter (range, and range / min(abserr)) and dropped. The rule
$|v_g - {\rm ref}| \le {\rm abserr}_g$ is evaluated exactly for every counted reference;
$|v_{06c} - {\rm ref}|/{\rm abserr}_{06c}$ is reported beside it. **Fails** if any counted
reference violates the rule; **undecided** if none does and fewer than five count; **holds**
otherwise.

**Z4.** At each rtol, on every row outside the 720 whose `total_abserr` is finite in both 06c and
g: $|{\rm total}_g - {\rm total}_{06c}| \le {\rm total\_abserr}_{06c} + {\rm total\_abserr}_g$,
evaluated exactly on the recorded doubles (`fractions.Fraction`). Rows with a non-finite
`total_abserr` in either are counted separately. **Fails** if a row violates it; **undecided** if
none does but a unit outside the 720 is over budget (it has no total); **holds** otherwise.

**Z5.** At each rtol, every Levin call (group and analytic) on a row outside the 720 whose 06c
record has `converged` false must have `converged` true in g's record. **Fails** if one does not;
**undecided** if none fails but a unit holding such a call is over budget or a call cannot be
matched; **holds** otherwise.

**What else is fixed now (descriptive, not scored).**
- *Convergence and depth:* converged calls and rows, `total_converged`, and the row depth
  distribution (max over the row's Levin calls) under 06c and g, at both rtols.
- *Cost:* regions (`num_regions`), solves (`num_subregion_solves`) and unit wall time (`wall_s`)
  summed, against 06c; the run's wall time against 06c's 2 h 30 min, and unit walls summed against
  06c's 3.21 h and 5.77 h.
- *Where the budget went:* per call, 06c's attribution (`E_share`, `E_largest_share`), and, over
  each call's Clenshaw–Curtis leaves whose floor carries their error (`cc_floor`, the worker's
  `_cc_floor_terms`), the floor's terms: (n + 4)uS, the argument term, |E|, R_g, R_2, R_c and the
  declared term.
- *Anything still deep:* every call outside the 720 with `max_depth > 12`, with its deepest line's
  residual over floor at each level (`deep_lines_06e`) and whether its ends tile at each level.
- *The stationary-point rows:* how the 720 moved against 06c, and by name Y1's 12 equilateral rows
  and the `*-1-2174` call 1 (units `g-1e-7-1-2174` and `g-1e-8-1-2174`, call index 1). The twelve
  are `docs/levin-phase-floor/06c/y1-references/results.json`'s rows Y01–Y12 (rank 1–12,
  `equilateral` true), found in `state/rows.json` by equal k, q, r and z_response (each must match
  exactly one row; one that does not is reported), and reported at both rtols.
