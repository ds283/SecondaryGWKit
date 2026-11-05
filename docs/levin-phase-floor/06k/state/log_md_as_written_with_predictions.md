# levin-phase-floor 06k — three ways to stop refining the slow leaves, measured on A3 v2 (a measurement)

**Subject:** prompt 06k (`prompts/levin-phase-floor/06k-slow-leaf-trust-routes.md`), on the Mac Studio
(mps026298), 2026-11-04.
**Commit:** none; records committed by the orchestrator from the laptop.

## Predictions and verdicts

*Copied in at 2026-11-04T09:00Z, before any K0 unit, Part L probe or census. The record is
`state/predictions_and_rules_20261104T0915Z.md` (read-only, SHA-256 `e315d74d…`, mtime 08:59:36Z),
which also holds K0's draw rule and seed and the operational definitions. Prompt 06k §4, verbatim:*

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

<!-- verdicts are added below this line when scored; the block above is not edited -->
