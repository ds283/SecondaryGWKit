# levin-phase-floor 06b, section 5 (the census): predictions, fixed before the run

Written 2026-10-28T14:13:38Z on mps026298.phys.susx.ac.uk, after Gate W0 held (out/cmd/13-15) and before the launch.
Part A is prompt 06b's §5 W1-W4, copied unchanged (lines 296-319 of
`prompts/levin-phase-floor/06b-roundoff-exit-premise.md` at `92bc62f`; out/cmd/18 and 19 check the bytes).
Part B fixes W2's selection rule, so that it determines the 24 calls mechanically, and the
readings each rule is scored by. Nothing in Part B changes a rule of Part A; where Part A is
silent, Part B says what is counted. Anything added after the launch goes in another file.

## Part A — prompt 06b §5, unchanged

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **W1 — nothing else moves.**
  - **The rule:** every Levin call with no phase-limited region in 06a equals 06a's call bit for
    bit, in value, abserr, region count and maximum depth. Every row all of whose calls are such
    equals 06a's row in `total` and `total_abserr`.
  - **Holds** if true at both rtols.
- **W2 — honest where it moved.**
  - **The sample:** 24 calls at `1e-7` that were phase-limited in 06a and whose result moved under
    $b$, with no sign change of $\theta'$ in 06a's scan (`spscan.py`). It is stratified by band
    (low, mid) and by $k$. The selection rule is fixed in `state/predictions.md` before the run.
  - **The reference:** Levin forced to $2^9$ and to $2^{10}$ equal pieces, with each piece's own
    floor reported. The two must agree to within a hundredth of $b$'s `abserr`; a call where they
    do not is reported, and dropped from the rule.
  - **The rule:** $|{\rm value}_b - {\rm ref}| \le$ `abserr`$_b$ for every call left in the sample.
- **W3 — the exit now fires only where bisection would not help.**
  - **The rule:** at `1e-7`, every region $b$ accepts as phase-limited has a halves' floor sum at
    least $\rho$ times its own. Recompute it from the halves' samples in the harness. Do not trust
    the code's flag.
  - **Holds** if no region violates it.
- **W4 — the cap.**
  - **The rule:** at both rtols, no call that moved reaches `depth_max`. An over-budget unit counts
    as false.
  - A failure is a finding for the user, not a fault. It bears on G2's L2 (Decisions, **G2 L2,
    decided**).

## Part B — W2's selection rule, and the readings by which W1–W4 are scored

**Terms.**
- *06a* is run `levin-phase-floor-06a-census-20261027T023143` (its checkpoint and per-worker
  pickles); *b* is this run, at the worktree `92bc62f` (code `4ddc3c2`). A b unit
  `b-<rtol>-<shard>-<serial>` is compared with 06a's `h-<rtol>-<shard>-<serial>`; calls are matched
  by `index`, with their `notify_label` equal.
- *A Levin call* is one `adaptive_levin_sincos` call of a row: its group calls (06a's and b's
  checkpoint `calls`) and its analytic blocks' calls (`analytic J*`/`Y*`; b's `other_calls`; 06a
  recorded no per-call result for them).
- *Phase-limited in 06a:* for a group call, 06a's `phase_limited` (true iff some region of the call
  was accepted phase-limited). For an analytic block's call, b's premise counter: the call asked
  the code's premise check (`premise_checks`) at least once. The two are the same thing: b's
  bisection is 06a's until a region meets the old exit's conditions; there 06a accepted the region
  as phase-limited and b asks the premise, and every region b visits that 06a did not lies under
  such a region. The reading is cross-checked on the group calls, where both are known, and every
  disagreement is reported.
- *Moved:* a group call moved if its value or abserr differs from 06a's by bit pattern. An analytic
  block's call moved if its premise check failed at least once in b (only then can its bisection
  differ from 06a's).
- *Over budget:* a unit still over budget after 06a's over-budget pass (one re-run with 3,600 s).
- `depth_max` is the driver's default, `DEFAULT_LEVIN_MAX_DEPTH = 20` (QuadSourceIntegral does not
  pass one). A call *reaches* it if a region of the call has depth ≥ 20.

**W1.** Per call: every group call not phase-limited in 06a equals 06a's call in value and abserr
(bit pattern), `num_regions` and `max_depth`. Per row: every row none of whose Levin calls (group
and analytic) was phase-limited in 06a equals 06a's row in `total` and `total_abserr` (bit pattern).
**Fails** if either is false anywhere; **undecided** if neither, with a unit over budget or in
error; **holds** otherwise, at both rtols.

**W2. The selection rule.**
1. *Eligible:* a group call at `1e-7` listed in `state/w2_capture.json` (06a phase-limited, and its
   06a scan `n_sign_changes == 0`), on a row whose `rows.json` band is `low` or `mid`, whose b unit is
   `done`, which moved, and whose inputs were captured without error.
2. *Strata:* (band, `k_key`), the strata of a band ordered by the row's `k` ascending (numerically).
   Within a stratum, calls are ordered by the SHA-256 (hex, ascending) of the string
   `"<b unit id>:<call index>"`.
3. *Draw:* twelve per band. Within a band, round-robin over its strata in order: one call from each
   stratum that has one left, pass after pass, until twelve are taken or the band is exhausted. If
   one band has fewer than twelve eligible, the shortfall is taken from the other band by continuing
   its round-robin, to twenty-four in all. If fewer than twenty-four are eligible, all are taken and
   the shortfall reported.

**W2. The reference and the score** (`scripts/w2_reference.py`, at b, on each call's captured
inputs). First the call is re-run from its reloaded inputs and must equal b's value and abserr bit
for bit. The span is cut into $2^9$ and into $2^{10}$ equal pieces by repeated bisection at the
driver's own break point, `(e + s) / 2.0`; each piece is evaluated once by
`_adaptive_levin_subregion` with the call's `chebyshev_order` and `rtol`, as the driver evaluates a
region; the pieces' values are summed (`math.fsum`), and each piece's `phase_err` is kept and their
sum reported. $S_9$, $S_{10}$ are the sums; the reference is $S_{10}$. A call with
$|S_9 - S_{10}| > $ `abserr`$_b/100$ is reported and dropped, as the prompt says. W2 **fails** if
$|{\rm value}_b - S_{10}| > $ `abserr`$_b$ on any call left; **undecided** if no call fails but some
call's reference could not be computed, or its re-run does not reproduce b's call, or no call is
left; **holds** otherwise.

**W3.** "Every region b accepts as phase-limited at `1e-7`" is every region with `phase_limited`
true in the result of every Levin call (group and analytic) of every `1e-7` unit. For each, the worker
recomputed after the timed section, with the module's own `_adaptive_levin_subregion` and the
call's arguments, the region's own floor on its own samples and each half's floor on the half's own
samples (halves split at `(end + start) / 2.0`). The rule is evaluated on those numbers:
`phase_err_L + phase_err_R >= 0.5 * phase_err_own`. A region whose recomputed own floor differs
from the stored `phase_err` is reported. **Fails** if any region violates the rule; **undecided** if
none does but a recomputation errored or a `1e-7` unit is over budget or in error; **holds**
otherwise.

**W4.** At both rtols: no call that moved (group or analytic, as defined above) reaches `depth_max`.
**Fails** if one does, or if any unit is over budget; **undecided** if neither, with a unit in error;
**holds** otherwise.
