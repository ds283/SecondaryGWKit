# levin-phase-floor 07b Part A — issues proposed (not opened)

Measuring agent, Mac Studio (mps026298), worktree `SecondaryGWKit-lpf07b-436ed5d` at `436ed5d4`. Each item is a
proposal for the orchestrator; none is opened here, and no board is touched.

1. **A trusted Levin leaf can carry an error common to its halves and quarters, beside a window and away from one.**
   This is the measured mechanism of section 2.2. The issue is presumably the one this prompt already owns.
   - **Beside a window (v08, v12, v17; v13, v18, v23 smaller):** the depth-0 seed leaf's grid has its first spacing
     285–1010 ℓ at the edge, with ε = 1/(4π) there. S1, Q and S3 share one error (spread 0.8–6% of err(Q)). It lies in
     the eighth beside the window, and falls about as Φ_w^(-3/2) when the window widens.
   - **Away from a window (v09 leaf 38, item 3's leaf 5, item 4's carriers):** the ends are resolved (rate ratio
     1.02–1.06 across the first spacing), and the error is still common to V_P, S1 and Q, with r 0.08–0.38. The eighths
     either remove it (item 3, 2174) or grow it (12977, 14345).
   - **In both, θ′ agrees with the captured phase** to at most 4×10⁻⁹. Every dishonest scored leaf spans an
     end-to-end |θ′| ratio of at least 49.
   - Proposed name: `[07b-trusted-levin-leaf-carries-a-common-error-across-a-wide-rate-range]`.
2. **Class (c) is not honest at 07a's (or 07's) code: 9 of 30 sampled rows converged and dishonest at both rtols**
   (1.19–10.8). The rows are listed in `log.md` observation 1. 07 is nearer the reference than 06m on all 60 (row,
   rtol). The cross-check's undecided class is decided for 07 on this sample, but it is not honest.
   - Proposed name: `[07b-class-c-rows-dishonest-at-07a]`.
3. **R2 found no dishonest call in 30 drawn uniformly (60 scored with the other rtol),** the largest ratio 0.805. This
   is the first honesty rate on random calls (§0.4). It is small and not stratified, and 21 of the 30 are analytic.
   Recorded, not an issue.
4. **R3-aux: `3-13508` G+q−r #2 at 1e-7 is dishonest at 07's code (1.04) and honest at 07a's (0.020).** This is
   evidence for `[07a-trust-check-auxiliary-route-moves-the-declared-error]`, whose call moved by more than 07's abserr.
   The other aux and shrinker calls are honest at both codes.
5. **The pairs are the 1e-7 member's dishonesty, and a window variant can make a pair agree while both members are
   wrong.** Under (b4)–(e), `3-14501`'s members agree to 10⁻⁶ e₇ and both are about 2× their abserr. A pair test (r_g)
   is not an honesty test. Bears on how 07's cross-check pairs are read.
6. **`_CC_THETA_PRIME_REL_ERROR_SUPPLIED = 2⁻¹⁰` is 10⁵–10⁶× the measured inconsistency of θ′ with the captured
   phase** (at most 4×10⁻⁹ integrated). At the declared error, the M3 perturbation moves the carrying leaves 5.7–10⁴×
   their true error, so it cannot discriminate. Recorded; the constant is used elsewhere as a bound, and changing it is
   not this prompt's.
7. **The projection method for split-FCC readings omits the direct segment** (`log.md` deviation 16). A stationary
   point's segment is integrated directly whatever the split, at a cost proportional to its phase span. Any later
   projection of split readings must keep 07a's γ D_f term. R1-x196's direct segment (5.1×10⁶ rad) took 23.2 h at r1+r2
   on two workers: about 0.016 s per rad, far less than 07a's γ (0.057 s per rad at r2) but not negligible.
8. **References on deep leaves of `2-34473` cannot reach e′/100.** Those leaves are at e′ ≈ 10⁻⁴¹; the reading's
   accuracy is 2–45 e′. Leaf-level attribution there needs a finer method. Recorded.

## Extension (§2.7)

9. **No quantity a leaf's own solves give separates the dishonest trusted leaves away from windows from the honest
   ones** (X2, 18 candidates, best margin 0.76), and a higher collocation order does not remove their error. The six
   carriers, and (e)'s eleven calls, rest on a common error of a trusted Levin leaf that bisection cannot see, trusted by
   r < 1/4 or by the noise clause. Proposed name: `[07b-no-leaf-local-test-sees-the-common-error-away-from-windows]`.
10. **The noise clause trusts leaves 1.2–2.7 e′ wrong:** the pairs' and v09's carriers (r 0.38–0.81, δ₁ ≤ ν_B).
    Proposed name: `[07b-noise-clause-trusts-a-wrong-leaf]`.
11. **(f)'s chase is outside 07a's window stop:** noise-level fallback and Levin leaves inside (f)-untrusted spans, beside
    window edges, reach depth 20 (X3: 4,374 leaves in 61 calls, 0 countable by the window rule). A stop for them must
    reach leaves made after it, as `_add_leaf` does for a stopped window; (i2) did not, and (i), counting whole subtrees,
    stops Levin leaves not at the noise level and returns them dishonest. Proposed name:
    `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`.
12. **The adaptive window seldom resolves the edge on the census:** 89 of 98 calls with a window reach 8π with the edge
    test failing (spacing/ℓ up to 2.2×10³), while family (R) is resolved by 8π. Family (R) is not the census's windows.
    Recorded.
13. **R2-28 under the rate-range rule:** honest in one region at (a), unconverged at 1.107 of its abserr under (f), (h),
    g2h and (i2). Recorded, not attributed.
14. **v19 is dishonest whenever its window widens** ((b4), (b8), g1, g2, g2h), honest at (a), (e), (h). Not attributed.
