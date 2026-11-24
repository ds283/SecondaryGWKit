# levin-phase-floor 07d Part A′ addendum: issues proposed

These are proposals only. Nothing is opened, and the board is the orchestrator's.

1. **A trust check can be fooled by an error that its Levin estimates share.** (Item B; `out/check_b_region.json`,
   `out/quarter_b_*.json`.)
   - **The case:** on `R:a:W2-0-12977-5`, region V = 59.4π at depth 4.
   - **What the check saw:** V_P, the halves' sum and the remaining quarters carry nearly the same error, about
     −3.6×10⁻³¹ to −4.0×10⁻³¹.
     - While one quarter's Levin solve was also wrong (by +2.13×10⁻³¹), the check refused.
     - When that quarter took the accurate CC₉₃ route (at 12π), the check trusted the region, with D 0.12 of its true
       error. That makes the call converged and dishonest (2.79).
   - **Measured:** one call. Its frequency elsewhere is not measured.
   - **Bears on:** the choice of gate T, and on the trust check's design.
2. **On trusted checks near the edge, D tracks Q's error by construction.** (`out/edge_vp_add.json`.)
   - **The class:** 49 near-edge checks with D/te ∈ [0.95, 1.05].
   - **What was found:**
     - The returned Q carries the halves' error (ratio 1.07), and D ≈ δ₁ = |V_P − halves|.
     - V_P is about 35× more accurate than Q on all 49.
     - Whether D ≥ te on such a leaf comes down to the (1 − r) factor. Neither reference here can decide them.
3. **One returned leaf understates under the CC₄₇ fallback.** `band_8_10|group|1e-8#23`, at 6π and 8π: D/te 0.51 on
   the smooth-phase reference; v4 could not see it. Under the CC₉₃ fallback the same check is untrusted.
4. **High-V Levin leaves understate on their own, far below budget.** On `R:a:W2-0-12977-5`, three regions with
   V = 662π–1,530π have decl/true 0.0096–0.51 under the tree and every variant. Their errors are about 10⁵ below the
   call's budget.
5. **The smooth-phase reference scatters by more than te on some band_20_32 group leaves at 6π** (`tables_add_supp.md`).
   For example, `band_20_32|group|1e-7#94` has scatter 3.3×10⁻⁴⁷, with eps_s 1.3×10⁻⁴⁶ against te 7.8×10⁻⁴⁷. So
   neither reference used here can decide those leaves.
