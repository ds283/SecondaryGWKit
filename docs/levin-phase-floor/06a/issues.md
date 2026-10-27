# levin-phase-floor 06a — observations not acted on

Each with its numbers. The orchestrator decides which are opened on the board. Sources:
[`out/census_summary.txt`](out/census_summary.txt), `out/cmd/40`–`45`.

**X1. Every row deeper than 11 is a stationary-point row, and its deep regions are at the
stationary point.**
- All 19 rows deeper than 11 at 1e-7 (all at depth 12), and all 29 at 1e-8 (28 at 12, one at 13),
  are rows whose 1e-7 scan found an interior stationary point.
- Their regions at depth ≥ 11 lie within 1–3.5 grid spacings of the nearest sign change of θ′
  (median 1; 137 regions at 1e-7 and 163 at 1e-8; `out/cmd/41`).
- Every region at depth ≥ 12 is in a `Gqr` call (1e-7: 13 Levin, 25 Clenshaw–Curtis; 1e-8: 17 and
  42, plus 2 Clenshaw–Curtis at depth 13).
- The deepest rows are almost all equilateral (q = r = k), k = 1e5 or 3.14e5/Mpc, mid band, class
  `Gqr`.
- **Impact:** this is why C3 fails as written. What is left of the depth is the driver bisecting
  towards a stationary point, which is 07's subject, not the input floor that layer 2 addresses.
  No row comes within five levels of the cap.

**X2. The scan's "accepts a region across one" count equals its stationary-point count by
construction.**
- 720 Levin calls in 720 rows (7.8 % of 9,280; 720 of 37,325 calls) have a sign change of θ′.
  Exactly those 720 calls list a region across it: 707 found by the grid, 13 by the region's ends
  and midpoint.
- The counts are equal because the accepted regions tile each call's `x_span`, so every interior
  stationary point lies inside some accepted region. The count of "accepts a region across one"
  therefore carries no information beyond the stationary-point count. What does carry information
  is the region that holds the point:
  - 623 of the 720 such regions are at depth 0, meaning the whole call was accepted as one region.
    The others are 40 at depth 1, 2 at depth 2, 2 at depth 3, 3 at depth 10, 32 at depth 11 and
    18 at depth 12;
  - 677 are Levin regions and 43 Clenshaw–Curtis; 238 are phase-limited;
  - |value|/abserr (1/relerr) has median 8.2e5 and range 2.7e3–1.1e13.
- **Where they are:** every one is in group `G+q+r` of regime `Gqr`. 412 are in the low band, 308 in
  mid, none in seam. By class, 236 are `G` and 484 `Gqr`. Over these calls min|θ′|/max|θ′| has
  median 2.6e-9.
- **Grid:** spacing h from 1.7e-5 to 7.1e-3 in log(1+z′), median 2.5e-3. A pair of stationary
  points closer together than h is not seen.
- **Impact:** this is the extent of `[06-levin-accepts-a-region-across-a-stationary-point]` at the
  current code. The 623 depth-0 regions are P1's pattern: one region of the whole call accepted
  across the point. `out/scan_listed_regions.csv` lists every region, for 07.

**X3. A3 v2's stored totals are not the current code's, across the whole store.**
- At 1e-7, |total_h − total_A3v2| exceeds the stored `total_abserr` on 6,187 of the 8,722 rows that
  have a finite, non-zero stored error. It exceeds 10× on 5,677 rows and 100× on 4,818. The median
  ratio is 286 and the largest 8.6e5 (`2-1524`).
- Relative to the stored total, the change has median 1.2e-3, 99th percentile 0.87 and maximum 42.6.
- 2,377 rows are bit-identical; these are mostly rows with no Levin call.
- **Cross-check:** 05's `p` units (B1's code only) against the same stored totals give median
  2.97e3× and 36 of 38 beyond 1× (`out/cmd/41`). So the size of this effect is B1's, already seen
  on 39 rows, and is not an artefact of the census.
- **Impact:** this widens `[05-b1-moves-stored-a3v2-totals-beyond-their-error]` from 39 rows to
  the whole store. A3 v2's stored totals should not be used as values at the current code.

**X4. Infinite declared errors.** At both rtols, 634 rows have `total_abserr = +inf`.
- **Against the stored values:** 554 were also +inf in the store, 80 are newly +inf, and 4 that were
  +inf in the store are now finite.
- **Where:** 392 in seam (all class none), 181 in low and 61 in mid. Every one has a Levin call.
- **Impact:** `[04-roundoff-floor-can-be-infinite]`, now measured over the whole store.

**X5. Convergence, the "before" for 08.**
- **`total_converged` False:**
  - at 1e-7 on 2,578 of 9,280 rows. The store holds 2,262. Of these, 544 rows converged in the
    store but not at h, and 228 the reverse;
  - at 1e-8 on 2,638 rows.
- **Unconverged Levin calls**, sorted by `convergence_kinds.py` (copy changed; diff
  `out/convergence_kinds_06a.diff`):
  - 1e-7: 6,964 of 37,325 calls. 4,014 are bound only, 1,737 resolution and 1,213 phase-limited;
  - 1e-8: 7,647 calls. 3,745 are bound only, 2,182 resolution and 1,720 phase-limited;
  - no call is capped at either rtol.

**X6. `GkSourcePolicyData.Levin_z` differs from A3 v2's on 22 of the 1,160.**
- Two values of k are affected, both in the low band: k = 3.045e7 (12 z_response, 9.199e12 against
  9.413e12) and k = 3e8 (10 z_response, 9.205e13 against 8.995e13 or 9.419e13).
- type and quality are unchanged on all 1,160.
- 06's W10 saw this on one row.

**X7. The producers re-run against the stored values (06's W9 over the whole store).**
- Per k, only 270–554 of 105,813–121,785 G values are bit-identical in every stored column. This is
  expected: 06 changed the stored phase columns.
- `theta_div_2pi` changes on 0–2 values per k, each one cycle near a 2π boundary.
- Coefficients are bit-identical on every G and T object. All 128 T re-runs are coefficient-identical
  to the store, and all 128 `QuadSource` recomputations hold L0a.

**X8. 1e-7 against 1e-8 at h** (descriptive, for 08's Q5).
- |total(1e-7) − total(1e-8)| / `total_abserr`(1e-7) has median 0, 90th percentile 0.34 and maximum
  3.9e4, over 8,646 rows.
- It exceeds 1× on 341 rows.

**X9. Regions.**
- At 1e-7, h uses 108,879 regions in all, against 145,313,734 stored (0.07 %). At 1e-8 it uses
  157,229.
- Per row with a Levin call: median 12, maximum 76 at 1e-7; median 13, maximum 136 at 1e-8.
- 74 rows (1e-7) and 125 rows (1e-8) have more regions than the store.

**X10. Record notes.**
- `out/cmd/11`'s lock listing names a `sed` (deviation 2).
- `out/cmd/43` (3.9 MB) is over 2 MB and stays on the Mac Studio. `out/convergence_kinds_h.txt`
  (1.7 MB) is the same output without the per-record input lines.
- `out/census_rows.csv.gz` is 1.65 MB, under 2 MB.
- The run directory (220 MB) and `state/C0/`, `state/pilot/`, `state/dev/` and `state/smoke/` (pickles)
  stay on the Mac Studio.
