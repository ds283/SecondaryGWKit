# Issues found by the rehearsal agent, levin-phase-floor 06b §4 (Mac Studio)

These are proposals for the board. The orchestrator decides whether to open them.

## Proposed

None that needs a new board entry from §4 alone. Two notes for the orchestrator to weigh:

### N1. On the rehearsal scope, the premise check removes every phase-limited acceptance

- **What happened.** R15 carries 48 group flags on 26 rows and 143 analytic-block flags on 70
  rows. R16 carries none (`out/qsi_tabulation.txt`).
- **Which case it exercises.** No production row on this scope takes the exit with its premise
  holding: the case of §2 test 2, with $G_1/G_0 < 3$.
- **Proposed handling.** Weigh it beside §5's descriptive count ("how often the exit fired … and
  how many of 06a's would have failed the premise") and W3. If A3 v2 shows the same, the exit may
  now be dead code on production phases. That bears on what `phase_limited` still means for a
  consumer.
- Not scored here.

### N2. The analytic blocks' phase-limited ORs are computed but not stored

- **What happens.** `_three_bessel_Levin` (`ComputeTargets/QuadSourceIntegral.py`) forms
  `J_phase_limited` and `Y_phase_limited` and returns them in its `"phase_limited"`, but
  `_three_bessel_integrals` drops them. Only the per-block flags reach the row's `metadata`.
- **Why it matters.** Nothing in a `QuadSourceIntegral` row's columns says that `analytic_rad` was
  accepted on a floor. A reader must walk `analytic.{0pt5,2pt5}[.Levin].{J,Y}{1..4}_data`.
- **What it is not.** Not a defect of 06b's, and not a value error. It is recorded because §4.3
  had to find these flags by reading the code.
- **Proposed handling:** a note on the board, or none, at the orchestrator's discretion.
