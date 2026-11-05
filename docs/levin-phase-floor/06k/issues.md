# levin-phase-floor 06k — proposed for the board

## To narrow

**`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`**, narrowed by 06k's three routes, the second
measured two ways (`docs/levin-phase-floor/06k/out/results.md`).
- **Every route brings convergence back:** 98.6–100 % of V8's lost calls outside the 720.
- **Every route costs less:** 0.30–0.78 of 06g's unit wall.
- **None passes as a candidate remedy:**
  - every route loses a few calls 06g converges (C2: 7, 5, 53 and 1);
  - route 1's trust and both route 2s' trust reach leaves far above noise (K1);
  - both route 2s and route 3 move values past both errors (S1: 10, 10 and 46 calls);
  - route 3 misses two references (H1) and leaves one of 06h's carrying leaves (L1).
- **Route 1** fails only C2 and K1.

## New

1. **`[06k-route3-left-leaf-declaration-understates]`** — under route 3, a leaf left at D ≤ 10⁻³ of the request can be
   off by more than the call's whole error.
   - **Where seen:** route 3's census.
     - 46 calls outside the 720 move past e_route + e_g, worst 1.645. Each move is carried wholly by one depth-0 or
       depth-1 leaf that 06g refined into 7–200 leaves (`06k/out/partC/movers_route3.json`).
     - Two references are missed: c01 at 3.93× (06g 0.0014) and r56 at 1.49× (06g 0.0059).
     - z00, in the 720, goes from 0.018 to 1,681×.
   - **Impact:** route 3's basis, that a left leaf's declaration would have to understate its error by 10³, is
     contradicted. As built, the route would make honest calls dishonest.
   - **Next step:** none proposed. This is a measurement of a route not adopted.
2. **`[06k-route2-corrected-declaration-omits-the-node-error-of-q]`** — under route 2 (as built and literal), a leaf
   trusted on the corrected numbers declares max(residual, Δ1c)/(1 − r_c) and returns Q uncorrected. Δ1c has the node
   errors taken out, so the declaration does not cover Q − V_P's node part.
   - **Where seen:** 10 calls outside the 720 move past both errors, worst 1.323
     (`06k/out/partC/movers_route2.json`). Eight are single-region analytic J1, Y1 and J2 calls; two are one
     G+q+r leaf at depth 11.
   - **Impact:** a declared error short of the value's move. The route is not adopted.
   - **Next step:** none proposed.

No other issue. The observations in the log's "Observations not acted on" 3–6 describe the routes and are not proposed
as issues.
