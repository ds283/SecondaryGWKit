# levin-phase-floor 05 §6 (the wider re-measurement): found and not acted on

These are for the orchestrator to place on the boards. Nothing here was changed. The numbers are in
`out/wider_05.txt`, from the run `levin-phase-floor-05-wider-20261026T020005`. In the text, $p$ is
the code commit's parent `e648951` (B1's code, without 05's change) and $c$ is the code commit
`120fdfe` (05's change).

## W1. B1 moves A3 v2's stored totals far beyond their declared error

**Proposed tag:** `[05-b1-moves-stored-a3v2-totals-beyond-their-error]`. Asked for by the laptop
orchestrator (its note 3), and to be opened there. Descriptive.

**What was measured.** For every row of the sample, $|{\rm total}_p - {\rm total}_{\rm stored}|$ /
the stored `total_abserr` (`out/wider_05.txt`, the last table before VERDICTS). Under $p$, the
`QuadSource` row is recomputed in-process as `main.py` computes it at this tree. It is bit-identical
to the stored row at every stored node, and it adds B1's 3 nodes (gate L0 (a), which held in all
109 units). The partition is B1's. Everything else is read from the store.
- **36 of the 39 rows** are outside the stored error. The ratio runs from 5.19 (`ctrl-mid-2`) to
  1.51e5 (`mid-G-r-3`), and most rows are between 20 and 1e5.
- **The relative change** runs from 5.7e-6 (`mid-G-3`) to 0.75 (R4). It reaches 13.2 on `low-G-r-2`, where
  the total changes sign: stored −3.05e-24, $p$ +3.73e-23.
- **The three rows within error:**
  - `low-Gqr-3`, whose stored `total_abserr` is infinite;
  - the two `NULL` controls, which have no Levin sub-interval. They are bit-identical, so B1 does
    not move a row that has no hand-over in its range.
- **03's rows:** R1 1.21e4, R2 5.72e3, R3 272, R4 6.18e3, R5 9.53e3, R6 84, R7 8.82e3.
- **The L0 control** `ctrl-low-1` changes by 41.8 times its stored error: stored −6.93e-31,
  $p$ −3.82e-31.

**Impact.**
- The stored A3 v2 `QuadSourceIntegral` values are not what this tree computes. The difference is
  not within the declared error on any row that has a Levin sub-interval.
- This agrees with R8's finding of a change of up to 0.80 relative on one triple
  (`docs/handover/b1/r8/verification.md`), now on 39 A3 v2 rows.
- Any reader of A3 v2's totals at this tree reads pre-B1 values.

**Next step.** For the board: whether A3 v2 is rebuilt at B1 or later before any physics is read
from it.

## W2. Under B1's code, most of the census's cheap depth-20 rows are no longer capped

**Proposed tag:** `[05-b1-uncaps-most-cheap-depth-20-rows]`. Descriptive.

- **11 of the 31** depth-20 rows of the sample reach depth 20 under $p$:
  - low band: `low-G+r-3`, `low-G-r-2`, `low-G-r-3`, `low-G-1`, `low-G-3`;
  - 03's rows: R1, R3, R4, R5, R6, R7.
- **20 are not capped under $p$.** Their maximum depth is 2–10 and their region counts are 9–72,
  against stored counts of 39–29,782. They are:
  - all 12 drawn mid-band rows;
  - all 6 drawn `Gqr` rows;
  - `low-G+r-1`, `low-G+r-2`, `low-G-r-1` and `low-G-2`;
  - and 03's R2, which drops from 29,782 regions to 15.
- So the stored `WKB_Levin_max_depth`, and 01's census built from it, describe the pre-B1 code. The
  drawing rule (`state/sample.md`) said that the classes and depths were the stored rows', and it
  did not assume them under $p$. This is the measurement.
- **The likely mechanism, not investigated:** the pre-B1 clamp at the hand-over. A clamp edge is a
  kink in the integrand, and 04's I6 named it as a candidate for a short chain to the cap. B1
  removed the clamp. These rows' depth-20 chains were short: 39–506 regions stored on 19 of the 20 (R2, 29,782, is the exception).

**Bearing.**
- On which rows any layer-1 result is about.
- On G2's Q4 (the depth cap).
- On any later census: it should be re-taken at the current code, not read from A3 v2.

## W3. L1 and L1q as written fail on rows that are not capped under $p$

**Proposed tag:** `[05-l1-rule-presumes-the-row-is-capped-under-p]`. Descriptive, about the
predictions themselves.

- L1's rule needs $c$'s region count to be at most a tenth of $p$'s. On a row that $p$ does not
  cap, $c$ and $p$ make about the same regions (the floor was not what drove them), so the rule
  fails by construction.
- **L1 fails on 15 of 24 rows**, and 14 of those are rows not capped under $p$.
- **L1q fails on 6 of 7 rows**, and all 6 are rows not capped under $p$.
- **Clause 1** (no group of $p$'s scored set left with 10 or more regions at depth ≥ 18 under $c$)
  **holds on every one of the 31 rows.**
- On the rows capped under $p$ (the second reading, which the user said is the one that matters),
  L1's rule holds on 9 of 10 and L1q's on 1 of 1. The one exception is R5 (W4).

The user's answer, relayed by the laptop orchestrator: *"I guess we only care about rows that are
capped under the new system"*.

## W4. R5: capped under $p$ by a single short chain, so L1's tenfold clause fails

R5 reaches depth 20 under $p$, but with only 55 regions, against 26,273 stored. The deepest chain is
in its `G+r` group: 42 regions, 7 of them at depth ≥ 18. That is below 03's threshold of 10, so R5
has no scored group under $p$.
- Under $c$ it has 19 regions at maximum depth 5, with none at depth ≥ 18.
- The ratio $c/p$ is 0.345, against the clause's 0.1.

So the change removes R5's cap, but the tenfold clause cannot be met when $p$'s cap costs about 50
regions. Descriptive; it bears on how a rule like L1 is posed.

## W5. L3 fails on two rows that the change barely touches

**Proposed tag:** `[05-declared-error-at-1e-7-misses-the-1e-8-change-on-uncapped-rows]`.

- **`mid-G-r-3`:** $|{\rm total}_c(10^{-7}) - {\rm total}_c(10^{-8})| = 1.29 \times$ `total_abserr`$_c(10^{-7})$,
  with a relative change of 2.7e-6.
- **`mid-G-3`:** 1.70× the declared error, with a relative change of 3.0e-7.

Neither row is capped under $p$. On both, $c$ is close to $p$: $|c-p|$/`total_abserr`$_p$ is 0.015
and 0.53. So this is the error estimate at `rtol = 1e-7` understating the change to `1e-8` on
ordinary rows. It is not something 05's change did.
- On the 11 rows capped under $p$, L3's rule holds on all 11. The largest ratio there is 0.166.
- On all 31 rows, the largest ratio otherwise is 0.615.

Bearing: `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`),
which 05 narrows, and G2.

## W6. `low-G-3`: on a capped row, $c$ moves by 2.48 times $p$'s declared error

This is 04's N4 widened. On `low-G-3` (capped under $p$, 831 regions, its lone `G` group with 689
regions at depth ≥ 18), $|{\rm total}_c - {\rm total}_p|$ = 2.48 × `total_abserr`$_p$, a relative
change of 1.05e-6.
- The jitter's effect, which $c$ removes, exceeds the error $p$ declares there. 04's I8 found the
  same on R3, R6 and R7, at the pre-B1 tree.
- On the other 10 capped rows the ratio is 0.0225–0.634. At $p$, R3, R6 and R7 are now within it:
  0.406, 0.403 and 0.17.
- Descriptive. It narrows 04's `[04-stored-abserr-understates-the-jitter-on-3-of-7-rows]`.

## W7. `total_converged` is False on most depth-20 rows under $c$ at both tolerances

- **Under $c$ at `1e-7`,** 21 of the 31 depth-20 rows are unconverged. At `1e-8`, 19 of the 31
  are.
- **On the capped rows,** 9 of 11 stay unconverged at both tolerances. The exceptions are R3 and R5.
  This is so although the depth drops from 20 to 5–11 and the regions fall by 95–99.8 %.
- **The unconverged pieces are Levin groups,** often with 1–17 regions at low depth. They are
  frequently `Gqr` groups such as `G+q-r` and `G-q+r`. No `numeric_quad` sub-interval is
  unconverged.
- **3 rows have an infinite `total_abserr`** under both $p$ and $c$: `low-G+r-2`, `low-G-r-1` and
  `low-Gqr-3`, with one non-converged `Gqr` group each.
- **Not investigated.** It may be the driver's round-off floor (`[04-roundoff-floor-can-be-infinite]`)
  or the group-level `atol` scaling. It bears on G2 and on how a stored row's `total_converged` is
  read.

## W8. Region counts under $c$ on R1–R7 are about a third to a half of 04's arm A

| Row | 04's A regions (pre-B1 tree) | $c$ regions |
|---|---|---|
| R1 | 133 | 63 |
| R2 | 89 | 12 |
| R3 | 66 | 18 |
| R4 | 167 | 83 |
| R5 | 130 | 19 |
| R6 | 89 | 28 |
| R7 | 88 | 27 |

The trees differ by B1 (partition and `QuadSource`), by the modulus (A used `TWO_PI`, $c$ uses $2\pi$)
and by the anchoring (A anchored at the top of the $\phi$ samples, $c$ per node). Descriptive: it is
not a comparison of like with like, and nothing here separates the three.

## W9. scipy `IntegrationWarning` in both trees

`stderr.log` holds 15 scipy `IntegrationWarning`s from `Quadrature/simple_quadrature.py:92`:
- 5 from the $p$ tree and 10 from the $c$ tree;
- the messages are "maximum number of subdivisions (100)" and "probably divergent, or slowly
  convergent".

They are production's numeric quadrature, not this harness, and no unit failed. Not investigated:
which rows raise them, and why the $c$ tree raises more, is not known. `stderr.log` does not name
the unit.
