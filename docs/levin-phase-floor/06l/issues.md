# levin-phase-floor 06l — issues for the board

**New issues: none.** Route B is a scratch variant, not code in the tree. As with 06k's route observations, what 06l
finds about it bears on a remedy. So it is proposed as a narrowing of the issue it measures, not as an issue of its own.

**Proposed narrowing of `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`** (for the
orchestrator to word; the numbers are in `06l/out/results.md`):

- **Route B** (ν_B = φ_P + the Levin halves' φ + the Clenshaw–Curtis halves' φ⁻ + |Σ E| over them; a Clenshaw–Curtis
  half contributing nothing on a Levin parent) **holds every safety rule on A3 v2's census:**
  - L1: none of 06h's carrying leaves trusted;
  - C2: 4 losses, all `node_limited`;
  - H1: worst v03 1.182;
  - S1: 0 moves;
  - K1: no Levin-parent firing at Δ1/φ_P ≥ 5.8; max 2.85, against route 1's 1.1 × 10⁴.
- **It recovers less than route 1:**
  - C1: 93.5 % and 93.1 %, against 98.9 % and 98.6 %;
  - it leaves 72 and 76 group calls at `depth_max`, against 14 and 16;
  - unit wall 0.86 of 06g's on a loaded host, and group-call regions 0.84, against route 1's 0.78 and 0.78.
- **It fails L2:** 63 of 75 first-population slow leaves are trusted.
  - §0.2's triangle-inequality guarantee is tight on the 12 others (E_P and E_L + E_R of opposite signs).
  - There Δ1 exceeds the first-order node error by 0.2–0.6 %, which no floor declares.
  - **A constraint on any remedy built on ν_B:** it needs a margin for the node term's remainder to reach these
    leaves. That margin is not measured here.
- **Route 1's extra recovery rests partly on what the user's modification removes:**
  - of the 91 calls route 1 converges and B does not, 14 rest wholly on Levin parents with a Clenshaw–Curtis half;
  - 75 rest partly on them, and 2 not at all.
  - So A, route 1 with the modification, recovers at most 77 of them. How many is not settled by route 1's census.
- **Next step:** for the user. Whether 06m implements B, B with a margin, A, or none.
