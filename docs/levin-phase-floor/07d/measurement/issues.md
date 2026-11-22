# levin-phase-floor 07d Part A — issues proposed

Proposals only. This Part opens no issue and touches no board; the orchestrator decides what is opened, narrowed or
closed. Each item points at the table that holds its numbers.

1. **The Levin route's own declaration understates on about a fifth of the leaves in every band up to 20π.** On table 3
   (out/tables/d_t.md), 408 band leaves understate on the Levin route's own declaration, all group calls, 348 of them with
   knots inside the leaf. The ratio of true error to declaration has median 1.22, p90 72 and p99 326. After the trust check
   (`levin_tc`) 5 to 40 leaves per band still understate. Analytic band leaves understate on neither. No threshold below
   20π is therefore bounded on both sides, under any candidate estimate. Proposed owner: whichever campaign takes the
   Levin solve's error model.

2. **The Clenshaw–Curtis fallback's declaration max(|CC₄₇ − CC₂₄|, floor) understates on 114 of 928 scored leaves, all
   group** (table 2, out/tables/d_e.md). Causes, tested in the order knots → noise: interior knots on 68 (the split at the
   knots removes the error), integrand noise on 46 (32 of them window pieces). On 07c's 176 truncation-dominated chase
   leaves (out/tables/trunc176.md) the tree's declaration understates on 22 and is undecided on 126. Causes: knot 8,
   noise 4, 3–4 s_CC 10 (4 distinct leaves). This is the measurement 07d asked for: the error is visible to estimates from
   the leaf's own samples (nest3N4 understates on none), so no §8 class was found.

3. **The group integrands' noise is `theta_mod_2pi`'s scatter, node-independent, and undeclared.** Table 1
   (out/tables/d_n.md):
   - The phase-predicted scatter over the measured σ_g is 1.42 ≈ √2, and the amplitude-predicted scatter is ~1e-3 of it.
   - Lag-1 autocorrelation of g is −0.016.
   - σ/|g| has median 2.75e-12 on group leaves and 5.6e-15 on analytic leaves.
   - The raw phase's own rounding (0.7 u|θ|) is not the source.
   - Group calls declare no phase error at all.
   - The analytic blocks' declared `theta_abserr` overpredicts their scatter by a median 4.1×.
   Proposed: a declared phase error for group calls, from the composition that forms `theta_mod_2pi`.

4. **Window pieces carry the largest noise:** group windows σ/|g| median 3.3e-10 (p90 6e-8), against 2.75e-12 for group
   leaves at large. 32 of the 46 noise-caused understatements of item 2 are window pieces.

5. **Group leaves labelled `G`, and group leaves with 5+ knots, show structure, not noise, at the finest sampling.**
   Plateau185/LS is 6.2e5 (label G, 59 leaves) and 8.6e4 (group, 5+ knots, 162 leaves). The shift test's spread is
   2.1e5 and 1.1e4 times s_CC. No coefficient plateau is found at 47 nodes. These leaves cross many table knots, so the
   pieces between knots are each smooth. The candidates nest3N4 and pairDecl bound them, and the five other candidates
   understate on 5–7 leaves, all with 59–82 knots. Proposed: whether the fallback should split at known knots (the
   production edges) before it estimates.

6. **Analytic leaves show step structure in `theta_mod_2pi` at its 2π wraps.** Plateau47/LS is 1.7 and plateau185/LS 3.5
   (white noise would give 1), and lag-1 autocorrelation of θ is 0.18. On analytic leaves with 1–4 knots the
   phase-predicted scatter exceeds σ_g (1.9 and 1.3). It is structure, not
   noise, and no table here measures its size against an analytic call's abserr. Proposed: check `BesselPhaseGroup`'s reduction at the wrap.

7. **Route A QAWO cannot read analytic blocks at large |K|.** At |K| ~1e7 to 3e8, QAWO's panels of phase span ≤ 2¹²
   (r1) and 2⁹ (r2) rad number 1e5 to 8e9 per integrand, so QAWO was dropped on 43 of the 46 (Ra) and (Rt) Route A
   integrands' readings (out/refs/refs_projection_07d.json). Their references rest on FCC r1 and r2 alone. Proposed for
   `tools/filon_reference.py`'s owner: a Route A rule whose cost does not scale with |K| · span.

8. **D-N's "common to CC_N and CC_{2N-1}" test is too loose to name a cause.** The test is same sign and
   |e₄₇| ≥ ½|e₂₄|, as table 2 first used it. A noise-limited leaf can pass it by chance (two independent errors of
   one sign). It is reported, not used
   as a cause. Proposed: drop it from future cause tables, or test it against s_CC.

9. **The prompt and the orchestrator note describe Rule A as reading `SIX_PI`; at 07c's code it reads no threshold.** The
   prompt and note describe Rule A as moving with the threshold. At 07c's code it reads no threshold (log deviation 2;
   the orchestrator will correct this in §9's amendment). Recorded here so that it is not lost.
