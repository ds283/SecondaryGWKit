## Addendum to the census agent's brief — the slow leaves behind V8's lost calls (descriptive)

**The source:** the user's request of 2026-11-01, relayed by the laptop orchestrator. The user's
words, replying to the laptop's proposal: "Yes, please do that". This is a descriptive item and
scores nothing.

**The purpose:** to tell apart two explanations of leaves that were refined to depth 20 and still
show r ≥ 1/4:
- (i) a non-smooth point (a kink or an integrable singularity) in the integrand or at a boundary,
  giving algebraic convergence at a steady ratio;
- (ii) Δ1 and Δ2 are rounding noise above a φ_P that is too low.

**The scope:** every untrusted (b) leaf that a lost call returns, outside the 720 stationary-point
rows, at both rtols. Compute it from the census records already held: the checkpoint and the
per-unit region pickles. **Rerun nothing.** If an item cannot be computed from what is recorded,
say so and skip it.

**Report, per rtol:**
1. **Counts by r:** the leaves' r binned into [1/4, 1/2), [1/2, 3/4) and [3/4, 1), with counts and
   quantiles of r. If r clusters (near 1/2 or 1/4, say), say so.
2. **Δ1/φ_P:** its distribution, from the median to the maximum, and its count in each r bin.
3. **r against depth:**
   - If the records hold the r of a returned leaf's ancestors, give r along the chain for a sample
     of leaves, and say whether r stays steady with depth or wanders.
   - If they do not, report r against depth across all the returned leaves, and state that the
     chains are not recorded.
4. **Position:** where the depth-20 slow leaves sit in each call's interval.
   - Do they cluster at a few points per call, and are those points at sub-interval or region
     endpoints? Or are they spread out?
   - Give the number of distinct clusters per call, as median and maximum, and a few example calls
     with their points.
   - State the clustering rule you use, before you look at the result.
5. **By group:** split everything by group (G±q±r), and note any difference.

**Output:** `out/v8_slow_leaves.md` and its data, with the script in `scripts/`. Its own section
goes in your records. Report the numbers; interpretation stays with the laptop.
