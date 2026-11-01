# Issues from 06h §4 (rehearsal agent)

**None to open.** No defect was found in this repository's code or in a pinned package that bears
on a result.

For the orchestrator's judgement, not opened:

1. **H0, and possibly §5's H3, cannot test 06h's change** (log, Observation 1).
   - **Why:** `total` is `numeric_quad + WKB_Levin`. `BesselPhaseGroup`, the only class 06h
     changes, reaches `QuadSourceIntegral` only through the analytic oracle (`analytic_rad`,
     `analytic.*`).
   - **In the rehearsal,** `total` and `total_abserr` are bit-identical in all 864 rows, so H0
     holds with $\lvert\Delta T\rvert = 0$.
   - **This is a property of the prediction, not a defect.** It changes no verdict here. The H0
     form applied to `analytic_rad` also holds, with a largest ratio of $2.5\times10^{-5}$.
   - **If the board should carry it,** a possible hook is: "06h's §4/§5 row rules score `total`,
     which the Bessel group phase does not reach".
2. **`runregistrykit list | head` raises `BrokenPipeError`** (log, deviation 7; 06e's rehearsal saw
   the same). It is a command-line nicety of the pinned package (`v0.2.0`), with no effect on any
   record. If it were opened, it would be reported to RunRegistryKit, not patched here.
