# levin-phase-floor 07c Part A — issues proposed (not opened)

Measuring agent, Mac Studio (mps026298), worktree `SecondaryGWKit-lpf07c-2fd979f` at `2fd979fd` (code 07b's). Each item is
a proposal for the orchestrator. None is opened here, and no board is touched.

1. **The integrand as evaluated carries noise the Clenshaw-Curtis floor does not model.** On the chase's CC leaves the
   per-sample scatter is about 3–6×10⁻¹² of |g|. No `theta_abserr` is declared for these integrands, so
   `_cc_roundoff_floor` sees only the rounding of the sum.
   - On R31's record's 122 chase leaves every interpolatory rule (CC_93, CC_185, composite Gauss-Legendre) disagreed with
     every other above the leaf's floor. The leaf references needed a least-squares fit (deviation 8).
   - D2 classes 22,700 of 22,876 measured chase leaves noise-dominated (`out/d2/d2_regimes.json`): true error within
     3σ√Σw² of the noise.
   - Proposed name: `[07c-integrand-noise-above-the-cc-floor]`.
2. **No Clenshaw-Curtis error estimate tried bounds CC_47's true error** (D2, `out/d2/d2_table.md`). The tree declares
   max(|CC_47 − CC_24|, floor), and that falls below the true error on 3 to 18 of 25 leaves in each group stratum:
   - gate q1 3, q2 11, q3 7, q4 5, chase 5;
   - window pieces 18.

   The minimum ratios to the true error are:
   - pair 0.020;
   - nest3 0.004;
   - tail_sum4 0.199;
   - tail_max4x2 0.118.

   On the 176 truncation-dominated chase leaves every candidate understates (pair median 0.23).
   - Proposed name: `[07c-cc-fallback-abserr-understates-its-true-error]`.
3. **Window pieces are noise-dominated far above their floor:** s_CC/floor has median 484 on the sampled window pieces
   (D2), against 0.36–1.5 on gate and chase leaves.
   - Proposed name: `[07c-window-pieces-noise-far-above-floor]`.
4. **The chase is mostly Levin bisection, not Clenshaw-Curtis.**
   - Of the 24,109 bisections below refused leaves, 15,445 are Levin.
   - 14,037 Levin leaves chosen below refused leaves had a residual above their floor, so the round-off exit was not
     considered.
   - Their δ₁/ν_B has median 66 (`out/d1/d1_tables.md` §2).
   - The trust clause that failed is r ≥ 1 for 6,734 of them, the noise clause for 7,213 and a rate rule for 88.

   Recorded for the orchestrator. It bears on whether a stop at the noise level reaches the chase.
5. **06d's "2.7×10⁻⁶ against 5.4×10⁻¹⁷ at 6π" names no case in the records found.**
   - The audit's own table (N = 13) reproduces to its printed digits.
   - The same integrand at N = 16 gives 9.8×10⁻⁸ at 6π and 3.8×10⁻¹² at 3π (`out/d2/reproduce.json`).

   Recorded, not an issue.
6. **The descent comparison's "nearest" column** (`out/d1/d1_tables.md` §1) disagrees with the census agent on 59 calls.
   This is a reading, not a disagreement: the census agent's `refused_ancestor_depths` is the *shallowest* refused
   ancestor's depth, which agrees on 198 of 198. Recorded, not an issue.

7. **R2-28 carries an error that every code shares, about 1.1× the abserr every code declares**
   (`out/tables/scores_07c.json`, table 3).
   - **The reference** is 07b Part A's, read by fccsplit and QAWO at both resolutions: −1.7616814886668874×10⁻²⁹, with
     spread 8.4×10⁻⁴².
   - **Every value is off by the same amount.** The value is −2.30×10⁻³⁷ from the reference under the tree and under
     every variant (1.31×10⁻⁸ relative). The codes' values differ from one another by at most 2×10⁻⁴¹.
   - **Every abserr is below that error.** The abserr is 2.01–2.09×10⁻³⁷ under every code, so |v − ref|/abserr is
     1.10–1.14 everywhere.
   - **At the tree** the call is unconverged (296 regions), with ratio 1.107.
   - **Under N, Eg and Ea** it stays unconverged, with ratio 1.10–1.11.
   - **Under S, S2, SEg and SEa** it converges (100–133 regions), and is dishonest at 1.12–1.14. This is the sole cause of
     their failing the qualification, through four rows that are R2-28's two calls scored twice (deviation 21).
   - **What the error is not:** an error common to every code's value is not something the chase or the CC abserr
     changes. Where it comes from was not measured in this Part.
   - Proposed name: `[07c-r2-28-common-error-above-every-abserr]`.
8. **(E) makes seven referenced calls unconverged that converge at the tree.** Under Eg and Ea these are:
   - `a:v05`;
   - `a:R1-x029-7/8`;
   - `a:R1-x098-7/8`;
   - R5's `m-1e-7/1e-8-2-38635-g6`.

   Under SEg and SEa the first five are. All seven have ratios ≤ 0.015 under every code. They are honest at the tree,
   and unconverged, not dishonest, under (E). Recorded for the orchestrator. It bears on table 3's honest column, not on
   the qualification rule as written.
9. **The references' projection overestimated by a third.** The jobs' summed wall time was 0.65 of 07's projection
   model's sum: 19.09 h against 29.56 h. The run took 6 h 22 min against 9.85 h projected without the margin. Recorded,
   not an issue.
10. **Table 5's "calls touched" counts fewer than `out/variants/reach.md` for every variant:**
    - S 326 against 328;
    - S2 309 against 311;
    - N 429 against 438;
    - Eg 385 against 394;
    - Ea 396 against 404;
    - SEg 395 against 405;
    - SEa 401 against 411.

    This is a definition, not a disagreement. `score_07c.py` counts a call touched if value, abserr, converged,
    num_regions or max_depth differ. `variant_reach_07c.py` also counts refusal counts and the region arrays' SHA-256.
    Recorded, not an issue.

## Extension: R2-28 (2026-11-19)

Item 7 above says R2-28's reference was read "by fccsplit and QAWO at both resolutions". The scored reference has two
readings, fccsplit r1 and r2, so that wording is a slip. It is left as written; see `log.md` deviation 25.

11. **A trusted Levin leaf can return a quarters' sum whose error exceeds its e′, because the halves and quarters share
    an error that its own solve does not** (07b's M2, now measured on R2-28; `out/r2-28/mech.md`).
    - R2-28's leaf [22.3566866, 23.5393356] at depth 4: S1 errs by −2.35×10⁻³⁷ and Q by −2.34×10⁻³⁷, against V_P's
      −4.8×10⁻³⁸.
    - So r = 0.0048, the leaf is trusted, and e′ ≈ δ₁ = 1.86×10⁻³⁷ is below Q's error: 1.247 e′.
    - The same leaf, with the same hex value, is in every 07b-lineage code's partition, and it carries the call's whole
      error: every code is dishonest or unconverged on R2-28 because of it.
    - Proposed name: `[07c-trusted-levin-leaf-shares-error-in-halves-and-quarters]`. It may be the same issue as 07b's
      observation 14 (trusted Levin carriers 1.17–2.66 e′ wrong); the orchestrator to judge.
12. **Levin solves at low |θ′| scatter at about 10⁻³⁷, 10³–10⁴ × their floor.** On the carrying leaf's 15 pieces at N
    and N ± 8, the median |error| is 2.5×10⁻³⁸ (2.0×10⁴ × `phase_err`) and the largest 2.4×10⁻³⁷. On its neighbour at
    higher |θ′| the median is 8.4×10⁻⁴⁰ (2.3×10³ × the floor).
    - The scatter does not fall with order or depth, so it is not truncation. The floor does not model it.
    - Its source, the integrand's noise (item 1) or the collocation, was not separated.
    - Proposed name: `[07c-levin-solve-scatter-above-its-floor]`.
