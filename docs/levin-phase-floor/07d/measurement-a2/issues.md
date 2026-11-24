# levin-phase-floor 07d Part A′ — issues proposed

Proposals only. This Part opens no issue and touches no board; the orchestrator decides what is opened, narrowed or
closed. Each item points at the record that holds its numbers.

1. **f₀ jumps in value at the nearest-node switch midpoints.** This is an integrand defect, apart from the quadrature
   (the orchestrator's second message).
   - **Where:** at each midpoint between table nodes, the nearest node changes, and f₀ steps.
   - **Measured** on 16 group leaves, 82 midpoints (`out/repro/midjump_a2.json`, `scripts/midjump_a2.py`):
     - The direct step |f₀(x + u) − f₀(x − u)| has median 2.9 × 10⁻¹⁵ of |f₀| and max 1.6 × 10⁻¹⁴, about 13 ulps of
       f₀ at the median.
     - It is the same at 1, 4 and 64 ulps (medians 2.9, 3.4 and 3.2 × 10⁻¹⁵), so it is a step, not a slope.
     - At knot-free control points on the same leaves the step is 9.5 × 10⁻¹⁶ (1 ulp) and 6.3 × 10⁻¹⁶ (64 ulps).
     - Item 4's fit-based value jump agrees: median 1.9 × 10⁻¹⁵, max 5.2 × 10⁻¹⁵ of |f₀|.
   - **f₁:** no step is resolved beyond its own slope over the step. The fit gives a median of 3.1 × 10⁻¹⁵ of |f₁|.
   - **Effect on the quadrature** (table 5): the step is above 10× the control at 26–33 % of the midpoints. Splitting at
     the knots lowers CC₄₇'s true error on about 76 % of group leaves; the midpoints are one of the kinds split at.
   - **Proposed owner:** whichever campaign owns the tables' interpolation (the nearest-node evaluation).

2. **The knot readers do not see knots in the sine component of a three-Bessel J call.**
   - `tools/filon_reference.production_edges` and Part A's analytic reader walk `f[0]`'s closure.
   - On the sine component `f = [lambda x: 0.0, lambda x: -Levin_f(x)]` (`ComputeTargets/QuadSourceIntegral.py`,
     `_three_bessel_Levin`), `f[0]` has no closure. Both raise TypeError, and the knots in `f[1]` are never read.
   - **Size:** 706 of 1,464 analytic leaves, every analytic stratum, calls Y1–Y4 (`out/repro/unknown_knots_a2.json`).
   - **Effect here:** v4 takes these leaves whole; 12 are unsettled (1.7 %, against 0.6 % for the others), and no verdict
     changes. Any tool that splits at knots inherits the gap.
   - **Proposed owner:** the campaign that owns `tools/filon_reference.py`.

3. **The leaf reference v4 understates its own error on some leaves.**
   - The leaves: 84 leaves where a declaration understates were checked against a second rule, 48-point Gauss–Legendre at
     256 and 512 panels a piece (`out/repro/second_ref_a2.json`).
   - **On 7 of the 84 the two references disagree** by 1.11–2.29 eps_v4, so eps_v4 there is 0.44–0.90 of v4's own
     error.
   - Every disagreeing leaf has a large eps_v4 relative to the error it judges, or 68+ knots.
   - **Four of them are all of table 4's returned-leaf Levin understatements**
     (`out/tables/understate_ref2.md`). Every value — V_P, CC₉₃, CC₁₈₅ and the split forms — agrees with the second
     reference far better than with v4.
   - So table 4's failures at 4π and 6π (group) and at 16π and 20π (analytic) are reference failures. On the second
     reference those sides have no understating leaf.
   - **The verdicts this decides:** V93, V93tail, G and K are each refused on (c) at 6π.
   - **Proposed owner:** this campaign, for the reference rule that any re-run uses.

4. **The tree's Clenshaw–Curtis declaration max(|CC₄₇ − CC₂₄|, floor) understates** on 74 of 899 scored leaves on v4
   (table 2, `out/tables/t2_de_v4.md`), all group.
   - Causes: knots 55, an error common to CC₂₄ and CC₄₇ 19.
   - All 74 still understate on the second reference.
   - In D-T (table 4) it makes the CC side fail for group calls at every T (34–52 leaves).
   - Part A's issue 2 carried forward on the new reference: narrower in count, same causes.

5. **A trusted Levin leaf understates on the visited band:** `band_20_32v|group|1e-7#79`, at 8π, under the CC₄₇
   fallback. D/te is 0.91, and both references agree. It is visited, not returned, so it is in no pooled row. Under the
   CC₉₃ fallback at 8π the same leaf is untrusted (r = 7.3). One leaf; recorded for whoever owns the trust check.

6. **The tree's own Levin declaration at making understates widely on the visited band:** group leaves with 5+ knots,
   36–44 of about 100 at each T (table 4, "Levin own decl" column, `band_20_32v` rows). These regions were untrusted and
   bisected, so nothing returned carries the declaration. It bears on any change that would return a region without its
   check.

7. **On knotted group leaves Q beats CC₉₃, which beats V_P.**
   - Group leaves with 5+ knots: CC₉₃ is closer than V_P on 515 of 516, but Q is closer than CC₉₃ on 310 against 37
     (table 4b).
   - A route that returns CC₉₃ on such a leaf in place of a trusted Q gives up accuracy. Recorded for the owner of the
     gate.

8. **(K) and (KT) make three regression calls dishonest that are honest at the tree:**
   - `R:c-1e-7-1-1759-5` and `R:c-1e-8-1-1759-5`, at 1.19 each;
   - `R:m-1e-8-3-35006-g4`, at 1.56 under K and 1.24 under KT.

   They also make R2-28's four honest (table 7). The cause has not been measured.
