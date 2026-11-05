# 06l predictions, copied verbatim from prompts/levin-phase-floor/06l-signed-trust-yardstick.md §4 (lines 256-298) at af4c4dc1, before any number of this prompt was taken

Written: 2026-11-05T09:07:47Z. Source SHA-256: dda3c8fd2d88f7cbd650b13f9b11a8b23488df9e73bc217cfb785ddcd99babe8

## 4. Predictions, fixed before the run

README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
**undecided** against the rule as written. None is a stop. "Outside the 720" means outside the 720
stationary-point rows.

- **L1 — B does not trust 06h's kind of leaf.** **Rule:** of the 17 06h carrying leaves (L-b), B
  trusts none.
- **L2 — B trusts every first-population slow leaf.** **Rule:** B trusts all 75 of 06j's sampled slow
  leaves with $\Delta_1/\phi_P < 2$.
- **C1 — convergence comes back.** **Rule:** at least 90 % of V8's lost calls outside the 720, at
  each rtol, converge under B.
- **C2 — nothing converged is lost but to the node-limited stop.** **Rule:** every call outside the
  720 that 06g converges and B does not ends `node_limited`.
- **H1 — no referenced call gets worse.** **Rule:** on every one of the 115 matched references
  outside the 720, $|v_B - {\rm ref}|/e_B \le \max(1,\ |v_g - {\rm ref}|/e_g)$.
- **S1 — no move past both errors.** **Rule:** no call outside the 720 moves by more than $e_B + e_g$
  against 06g's census.
- **K1 — B trusts Levin parents only at noise level.** **Rule:** every trust firing on a Levin
  parent, outside the 720, has $\Delta_1/\phi_P < 5.8$.

**Descriptive, not scored:** the same counts for Clenshaw–Curtis parents, by $\Delta_1/\phi_P$ and by
$\Delta_1/(|E_P| + |E_L + E_R|)$; how many of 06j's second population B trusts, and why.

**The basis.**
- **L1, H1, S1 and K1** are 06k's safety constraint.
  - L1: 06h's leaves are Levin over Levin, where $\nu_B = \nu_1$ and route 1 trusted none
    ($\Delta_1/\nu_1 \ge 5.7$).
  - K1: on a Levin parent, $\nu_B$ holds only Levin floors. In 06k's L-c sample a Levin half's
    floor is at most 0.96 of its parent's, so $\nu_B \le 3\phi_P$ there.
  - K1 is scored on Levin parents only. On a Clenshaw–Curtis parent $\phi_P$ understates the
    rounding in $\Delta_1$ (§0.1), so $\Delta_1/\phi_P$ against 5.8, a threshold taken from Levin
    leaves, is not a test of noise there.
- **L2** is §0.2's guarantee, checked. 06j measured $r$ at 0.489–0.506 on these leaves, so the
  $r < 1$ condition holds.
- **C1** keeps 06k's rule for comparison. The leaf-level estimate (about 60 % of sampled slow leaves
  trusted, against route 1's 80 %) makes a shortfall likely, and the recovered fraction itself is the
  measure.
- **C2** follows the user's ruling that losses to the node-limited stop are
  `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`'s, not the route's.

B passes as a candidate remedy if L1, L2, C2, H1, S1 and K1 hold. C1 and the comparison with route 1
say how much it recovers, and what A's extra recovery rests on.
