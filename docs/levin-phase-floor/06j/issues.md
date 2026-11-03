# levin-phase-floor 06j: issues for the board

These are proposals; the orchestrator names them and decides what is opened, narrowed or closed. Each one's
measurements are in `out/results.md` and the files it names.

**J1. Narrow `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` with Part S's cause: the slow
leaves are rounding (N), not seams (J).**
- **Seams:** 141 of 142 sampled slow leaves hold no representation seam (S1 fails: 0.70 % against a null of
  0.12 %; trusted deep leaves 1.7 % against 1.6 %).
- **Rounding:** $\Delta_1$ is at or below the few-ulp scatter of the leaf's value on every leaf (S3 fails).
- **75 leaves:** $\Delta_1 = |E_P - (E_L + E_R)|$ to 0.6 %, the difference of three solves' 06e node errors. Rule
  (b)'s noise clause compares it with $\phi_P$, which carries only $|E_P|$; this gives $\Delta_1/\phi_P \approx 1.5$
  and $r = ½$.
- **67 leaves:** $\Delta_1$ is inside the integrand's input sensitivity at the nodes.
- **What a remedy would have to do:** give the trust test a noise level that covers every solve in $\Delta_1$ and
  $\Delta_2$, or form them from node-corrected values. It must keep refining leaves with $\Delta_1 \gg \phi_P$
  (J2). Seams as breakpoints would not address these leaves.
- Bears on `[06e-node-errors-add-in-absolute-value]` (06g's signed node sum treats the call's total, not the trust
  test's differences).

**J2. Close or narrow `[06g-g-plus-r-rows-move-past-both-errors]`: the parent is the one in error.**
- **06g is honest:** within its error on all 22 W calls (worst 0.071).
- **06h is not:** outside its error on all 22 (2.4–20×), reported converged.
- **Where 06h's error sits:** on every call, in one leaf 06h trusted at $1/4 \le r < 1$ with $\Delta_1/\phi_P$ from
  5.8 to $1.2\times10^7$. On the `G+r` calls this is the depth-1 leaf $[23.539, 25.255]$, which carries the whole
  $7.1\times10^{-33}$.
- This is `[06c-budget-can-stop-on-a-pre-asymptotic-residual]`'s mechanism in 06h's trust test (r < 1), and 06g's
  rule (b) corrects it. No remedy on 06g's side is indicated.

**J3. Observation, descriptive: a small `theta_mod_2pi` step at a coinciding table node.**
- At s07's seam ($x = 18.8868686$, $z = 1.59\times10^8$; nodes of `tau`, `cs_tau` and `friction_F` coinciding with
  phase and amplitude spline knots), `theta_mod_2pi` shows a step of 5e-13 to 1.1e-12 rad that does not shrink with
  the stencil step. The control falls to 3e-15.
- Below the integrand's stencil noise, and not investigated.
- Proposed as a note on the board, not an issue, unless the orchestrator prefers one.

**Recorded in the log, not proposed:** the slips in deviations 8, 14, 15 and 16. In particular, the readings
plan's header gives a correction time (09:38Z) that its mtime (09:33:15Z) rules out.
