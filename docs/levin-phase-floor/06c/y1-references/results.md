# Y1 references: which code was right on the 30 Y1 rows of 06c Part B

Mac Studio (`mps026298`), 2026-10-29. R17 = tree `fef401c` (06b's Levin code); R18 = tree
`e5764a6` (06c's global budget plus 06d's CC floor). Brief: `dispatch.md`. Log and deviations:
`log.md`. Per-call numbers, all in hex: `results.json`. Full per-call table:
`state/results_table.md` (written by `scripts/assemble.py`, output `out/30-assemble.txt`).

## Conclusion

1. **The twelve worst rows are near-singular, and they are 06d's class.** In every one, the
   `G+q+r` phase has an exact stationary point inside the span. θ′ changes sign at x = log(1+z) =
   9.23135, z = 10211.3, the same point for every k: it is where 2c_s(z) = 1 for k = q = r. θ′ is
   exactly 0 at a double there (|θ′| ≤ 7.5e-9 at the two bracketing doubles). Across the span,
   max|θ′| is 3.0e5 to 7.3e7. The stationary width 1/√|θ″| is 1.43e-3 at k = 1e5, 8.0e-4 at
   3.16e5 and 4.5e-4 at 1e6. No other moved call of the 137 referenced has a zero of θ′. None of
   rows Y13–Y30 has one: Y27, the one other equilateral row, has z_response = 15829 > 10211.
2. **Whichever code leaves the stationary point inside a coarse region is the dishonest one.**
   Both codes call the `G+q+r` call converged, except R17 on Y11. That code's error on the call is
   4.5e3 to 1.6e6 times its own declared abserr.
   - **R18 is the dishonest code in 10 rows** (Y01–Y07, Y09, Y10, Y12). Its region containing x0
     is at depth 1, 6–8 wide in x, which is 4e3–1e4 stationary widths.
   - **R17 is the dishonest code in Y08 and Y11.** Its region is at depth 1 on Y08 (7.9 wide) and
     depth 0 on Y11 (the whole span, 16.4 wide, flagged unconverged).
   - The other code bisected to depth 10–12 around x0, leaving regions 4–11 stationary widths
     wide. It is within its declared error, or within a factor 0.08–10.9 of it.
3. **Row verdicts.** In the top twelve: R17 within on 6 rows (Y03, Y04, Y05, Y06, Y07, Y09); R18
   within on 2 (Y08, Y11); neither on 4 (Y01, Y02, Y10, Y12).
   - On the four "neither" rows R18 is off by 443–2.5e4× and R17 by only 1.4–3.2×. R17 misses
     there on its sub-0 `G` call (2.2–3.3×) and on `G+q+r` (Y01 2.3×, Y12 3.5×).
   - In the other 18 rows: **R18 within on every row**, R17 outside by 1.5–68×. R17 is driven by
     its sub-0 `G` call (up to 73×) and by `G±r` (up to 192×). None of these 18 rows is
     near-singular.
4. **R18's other misses are small.** Outside the `G+q+r` stationary calls, R18 is outside its
   declared error on 13 calls, all by at most 3.53× (Y19 `G-q-r`). R17 is outside on 66 calls.

## Per-row table (verdict: is |T − T_ref| ≤ the code's declared total_abserr)

Columns: the calls that moved (referenced / moved); per call, |value − ref| / abserr for each
code, and the reference's convergence difference |S384 − S192| over min(abserr17, abserr18); per
row, |T − T_ref| / e for each code.

A row with unreferenced moved calls shows two figures, "a / b". In "a" those calls are taken at
R17's value, in "b" at R18's. The verdict is the same under both in every row (state/results_table.md).

| row | k = q = r (/Mpc) | z_resp | moved calls (ref'd/moved) | G+q+r: R17 / R18 err/abserr | row: R17 \|T−Tref\|/e17 | R18 \|T−Tref\|/e18 | verdict |
|---|---|---|---|---|---|---|---|
| Y01 | 1e5 | 5236.55 | 3/3 | 2.27 / 2.44e5 | 2.16 | 2.51e4 | **neither** |
| Y02 | 1e5 | 996.42 | 5/5 | 0.0775 / 2.64e5 | 2.10 | 2.37e4 | **neither** |
| Y03 | 3.162e5 | 434.651 | 2/5 | 0.433 / 5.97e5 | 0.70 / 0.75 | 2.55e4 | **R17 within** |
| Y04 | 3.162e5 | 573.115 | 2/5 | 1.69 / 3.96e5 | 0.24 / 0.28 | 1.98e4 | **R17 within** |
| Y05 | 3.162e5 | 996.42 | 2/5 | 10.9 / 2.83e5 | 0.010 / 0.021 | 1.53e4 | **R17 within** |
| Y06 | 3.162e5 | 5236.55 | 5/5 | 0.551 / 1.86e5 | 0.48 | 1.45e4 | **R17 within** |
| Y07 | 3.162e5 | 755.687 | 2/5 | 0.601 / 1.01e5 | 0.075 / 0.106 | 1.83e4 | **R17 within** |
| Y08 | 1e5 | 250 | 3/3 | 1.62e6 / 0.283 | 3.40e4 | 0.105 | **R18 within** |
| Y09 | 3.162e5 | 6904.72 | 5/5 | 0.723 / 8.58e4 | 0.049 | 1.18e4 | **R17 within** |
| Y10 | 1e5 | 3971.41 | 5/5 | 1.06 / 8.87e4 | 1.41 | 8.19e3 | **neither** |
| Y11 | 1e6 | 1313.84 | 2/5 | 1.60e4 / 0.403 | 4.20e3 | 0.39 / 0.30 | **R18 within** |
| Y12 | 1e5 | 6904.72 | 5/5 | 3.50 / 4.49e3 | 3.20 | 443 | **neither** |

| row | k, q, r (/Mpc) | z_resp | moved (ref'd/moved) | R17 \|T−Tref\|/e17 | R18 \|T−Tref\|/e18 | verdict |
|---|---|---|---|---|---|---|
| Y13 | 1e6, 1e5, 1e6 | 20871.2 | 7/7 | 25.8 | 0.045 | **R18 within** |
| Y14 | 1e5, 1e6, 1e6 | 1313.84 | 3/5 | 67.6 | 0.37 | **R18 within** |
| Y15 | 1e5, 1e6, 1e6 | 1.0022e6 | 5/5 | 49.3 | 0.30 | **R18 within** |
| Y16 | 1e6, 1e5, 1e6 | 1.45467e8 | 7/7 | 13.0 | 0.033 | **R18 within** |
| Y17 | 1e5, 1e6, 1e6 | 4.8124e7 | 5/5 | 13.0 | 0.20 | **R18 within** |
| Y18 | 1e5, 3.162e5, 3.162e5 | 3.33477e8 | 5/5 | 28.4 | 0.36 | **R18 within** |
| Y19 | 3.162e5, 1e5, 3.162e5 | 9104.3 | 6/6 | 12.4 | 0.23 | **R18 within** |
| Y20 | 3.162e5, 1e5, 3.162e5 | 1.00802e9 | 7/7 | 8.66 | 0.12 | **R18 within** |
| Y21 | 1e5, 1e6, 1e6 | 1.91807e8 | 5/5 | 14.9 | 0.23 | **R18 within** |
| Y22 | 1e6, 1e5, 1e6 | 2.29751e6 | 6/6 | 4.31 | 0.030 | **R18 within** |
| Y23 | 3.162e5, 1e6, 1e6 | 5.79784e8 | 5/5 | 2.62 | 0.0058 | **R18 within** |
| Y24 | 1e6, 3.162e5, 1e6 | 4.01763e9 | 7/7 | 1.99 | 0.24 | **R18 within** |
| Y25 | 3.162e5 (all) | 2.31084e9 | 5/5 | 2.03 | 0.030 | **R18 within** |
| Y26 | 3.162e5 (all) | 1.00802e9 | 5/5 | 2.27 | 0.088 | **R18 within** |
| Y27 | 1e5 (all) | 15828.8 | 5/5 | 2.69 | 0.25 | **R18 within** |
| Y28 | 1e6, 1e5, 1e6 | 3.04698e9 | 3/3 | 1.53 | 0.036 | **R18 within** |
| Y29 | 3.162e5 (all) | 3.04698e9 | 5/5 | 1.72 | 0.37 | **R18 within** |
| Y30 | 1e6 (all) | 1.59206e7 | 5/5 | 1.84 | 0.016 | **R18 within** |

## Near-singular evidence: the `G+q+r` calls of the top twelve

x0 = 9.23135 (z = 10211.3) in every row. θ′ = 0 at the double `0x1.276737eb9bfeap+3` (k = 1e5),
`...bc58p+3` (3.16e5) or `...bbfep+3` (1e6). In the table below, "width" is the region's width in
x, and "(n w)" is that width in stationary widths.

| row | stationary width | max\|θ′\| on span | R17 region at x0: depth, width (n w) | R18 region at x0: depth, width (n w) | R17 / R18 regions, max depth |
|---|---|---|---|---|---|
| Y01 | 1.43e-3 | 6.96e5 | 10, 0.0124 (8.7) | 1, 6.35 (4.4e3) | 12/10 vs 2/1 |
| Y02 | 1.43e-3 | 9.96e6 | 11, 0.0070 (4.9) | 1, 7.18 (5.0e3) | 13/11 vs 7/5 |
| Y03 | 8.0e-4 | 6.92e7 | 12, 0.0040 (5.0) | 1, 8.17 (1.0e4) | 13/12 vs 2/1 |
| Y04 | 8.0e-4 | 5.44e7 | 11, 0.0078 (9.8) | 1, 8.03 (1.0e4) | 13/11 vs 8/6 |
| Y05 | 8.0e-4 | 3.15e7 | 12, 0.0038 (4.7) | 1, 7.75 (9.7e3) | 13/12 vs 10/7 |
| Y06 | 8.0e-4 | 2.20e6 | 11, 0.0068 (8.4) | 1, 6.92 (8.6e3) | 13/11 vs 3/2 |
| Y07 | 8.0e-4 | 4.19e7 | 12, 0.0039 (4.8) | 1, 7.89 (9.8e3) | 13/12 vs 4/3 |
| Y08 | 1.43e-3 | 3.37e7 | **1, 7.87 (5.5e3)** | 10, 0.0154 (10.8) | 2/1 vs 21/10 |
| Y09 | 8.0e-4 | 9.55e5 | 11, 0.0066 (8.3) | 1, 6.79 (8.5e3) | 12/11 vs 2/1 |
| Y10 | 1.43e-3 | 1.31e6 | 11, 0.0063 (4.4) | 1, 6.49 (4.5e3) | 12/11 vs 2/1 |
| Y11 | 4.5e-4 | 7.27e7 | **0, 16.4 (3.6e4)** (unconverged) | 12, 0.0040 (8.9) | 1/0 vs 66/16 |
| Y12 | 1.43e-3 | 3.02e5 | 11, 0.0061 (4.3) | 1, 6.21 (4.4e3) | 18/12 vs 7/4 |

In every row, the code whose x0 region is coarse misses the reference by 4.5e3–1.6e6 of its
declared abserr. On Y01, for example, R18's depth-1 region [8.564, 14.912] declares abserr 3.8e-29
(total_err 7.1e-29), and the call's true error is 1.8e-23 (raw, before the 1+z scale). Its summed
error, 7.3e-29, sat under its budget of rtol·|value| = 1.7e-28, so the global budget stopped. The
trust check one level down did not catch the stationary point, because at depths 1–3 the region
containing x0 is still thousands of stationary widths wide.

## How each reference's accuracy was established

- **Method.** Composite Gauss–Legendre of f_sin·sin(θ_mod) + f_cos·cos(θ_mod), on the captured
  callables themselves (it is not the Levin driver), as log.md deviation 4 describes.
  - Panels: equal steps of Φ/512 + x/0.005, with Φ the integral of |θ′|. The oscillation count was
    taken first, by `probe.py`.
  - Two resolutions on the same panels: 192 and 384 nodes per panel.
  - Exact summation: Shewchuk partials, one final rounding.
- **Estimate.** The reference's error estimate is |S384 − S192|.
  - It is at most 0.0028 of the smaller declared abserr on all 12 `G+q+r` calls, and at most
    1.5e-8 on all 30 `G` calls.
  - It meets the brief's 1/100 on 122 of the 137 referenced calls.
  - **14 calls reach only 0.010–0.099, and one reaches 0.76** (Y22 `G+q-r`). These are `G±q±r`
    calls whose R17 abserr is tiny.
  - A 384/768 re-run of Y16-06 (out/26) gives |S768 − S384| = 2.8e-36, no smaller than
    |S384 − S192|. That is the evaluation-noise floor of the double-precision integrand, ~1e-14 of
    Σ|terms|, so the 1/100 target is not reachable there.
  - On Y22-04, |S768 − S384| = 3.6e-36. Against R18's abserr that is 4e-4, so R17's abserr is the
    limit.
  - None of these 15 calls is a `G+q+r` stationary call. On none of them could a verdict change: the
    reference uncertainty is far below each code's |v − ref| or margin (results.json
    `ref_conv_over_min_abserr`).
- **Row totals.** The reference total's error is the scaled sum of the calls' convergence
  differences. It is at most 6.9e-28, against row errors of 1e-27 to 4e-20 (results.json,
  `Tref_error`).

## How the reference was shown to integrate each driver's function

1. **Capture.** `capture.py` re-ran every row in-process in each tree, on its own store copy,
   through the census harness with Ray stood in. Every `rebuild_row` check against the stored row
   passed.
   - total and total_abserr were bit-identical to the store, all 30 rows in both trees.
   - Each group call's (1+z)·value, (1+z)·abserr and region count were bit-identical to the store's
     metadata.
   - The moved calls' span, f, theta dict and kwargs were pickled (out/04–08, 16–17).
2. **Replay.** Each pickled call was replayed in a fresh process in its own tree. Value and abserr
   reproduced its capture and its store bit for bit for all 154 calls in both trees (out/10, 11,
   18, 19), and again at the start of every reference step.
3. **Cross-tree comparison.** R17's callables, evaluated at R18's reference nodes on a dense sample
   (every 16th panel, at most 3000 panels per call, both node sets), are bit-identical to R18's.
   - Five quantities were compared: f_sin, f_cos, θ_mod_2pi, θ and θ_deriv.
   - This covers 50.8 million nodes across all 137 calls, with zero differing bits, plus θ_mod at
     every sampled panel end.
   - So one reference serves both codes.
4. **Limits and phase convention.**
   - Code: `phase_group_Levin_integral` passes x_span = (log(1+z_min), log(1+z_max)) with
     a < b, and f = [f_sin, f_cos]. `adaptive_levin_sincos` integrates f[0]·sin θ + f[1]·cos θ over
     x_span (docstring). `_Basis_SinCos.eval_basis` and `build_Levin_data` take sin and cos of
     `theta_mod_2pi` when supplied (both trees).
   - Scale: the reference is compared with the raw call value. The store holds (1+z_response)
     times it, and the capture checked that product.
   - Numbers: on the calls with no stationary point, both codes agree with the reference to
     within a few times their ~1e-8-relative declared errors (e.g. Y24 `G-r`: 0.0013 and 0.0053 of abserr). A swapped sin/cos,
     reversed limits or a wrong phase would give O(1) relative errors.
   - θ_deriv against θ: the GL integral of θ_deriv across each sampled panel matches the change in
     θ_mod_2pi to ≤ 1.2e-8 rad per panel.
