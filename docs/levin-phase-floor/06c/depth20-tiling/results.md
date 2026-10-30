# Depth-20 lines against the end-node tiling mechanism: results

Mac Studio (`mps026298`), 2026-10-29. Tree g = `e5764a6`, unpatched, clean throughout. Brief:
`dispatch.md`; log and deviations: `log.md`; per-call numbers in hex: `results.json`; every
command's output: `out/NN_*.txt`.

**STOPPED at step 4 under the brief's stop rule:** the first reference attempted (sample 1,
`g-1e-7-2-1444` call 4) did not converge to 1/100 of the smaller declared error
(|S384 − S192| = 2.72e-36 = 0.30 × the pinned abserr 9.07e-36, 0.059 × the census abserr). No
further reference was launched. Steps 0–3 and the counterfactual runs are complete for all 30
sampled calls; the honesty half of the question is **not** answered by a reference.

## The answer

**The production depth-20 chase is, overwhelmingly, the tiling mechanism.** Over all 262 lines'
last five levels, 1,179 of the 1,222 Levin parents with residual > 10× floor have computed end
nodes that do not tile (96.5 %), and all 17 Levin parents with residual ≤ 10× floor do tile. On
the 30 sampled calls, pinning the two end nodes to a and b (nothing else changed) removes the
deep residual: on 127 of 138 flagged deep levels it falls to ≤ 10× the floor (median
residual/floor 3e4 → 1.4e-3; pinning removes ≥ 99 % of the signed residual in mp on 129), and
the measured change matches the first-order offset terms g(b)δb − g(a)δa (within 10 % on 122 of
138, test (iii) on 123; within 34 % on all but sample 0/12's level 18, off by 1.33). 134 of the
138 do not tile; the 4 that tile are all Levin-parent/CC-halves levels at the bottom of a line.
The strict pre-fixed rule puts 15 calls in (a), 3 in (b) and 12 in (c); but 10 of the 12 (c) and
all 3 (b) calls are tiling-held on every deep level except either the very bottom level, where a
Levin parent is compared with Clenshaw–Curtis halves at the 6π gate (pinning does not move that
residual), or levels where pinning removes ≥ 99 % of the residual but the first-order prediction
is off by 10–33 %. The three (b) verdicts are artefacts of the 1e-6 threshold on analytic spans
whose |θ′| spans six decades (θ′ ≈ 6e5–9e5 there, no sign change). **Pinning stops the chase on 29
of 30 calls** (max depth 1–7; regions 10,181 → 274; solves 55,694 → 2,162; 29 converged against 1).
The exception is one G+q+r row (`*-1-2174` call 1, both tolerances), whose span holds a θ′ zero
at x = 5.63671 (z ≈ 279.5): pinned at 1e-7 it still reaches depth 20; pinned at 1e-8 it converges
at depth 7 but its value moves by 1.7e3× its own declared abserr (1.9e-4 relative) away from the
census value, which agrees at both tolerances and with the pinned 1e-7 run — so pinning there
most likely returns a dishonest error (Y1's stationary-point class), **unverified**. Elsewhere the
pinned value moves from the census value by 5e-10 to 0.05 of the census abserr. **Whether
pinning keeps the declared error honest is not established**: the one reference attempted failed
the convergence criterion and stopped the step; 18 of the other references are estimated at 20 h
to ~1 year at six workers with Y1's method, and 2 more at 2.0–2.6 h.

## Cheap pass (step 1, all 262 calls; `out/02_cheap_pass.txt`)

- Every span reconstructed from the checkpoint's `x_span` with the driver's `(b + a) / 2.0`;
  262/262 reproduce the recorded leaf bit for bit, and every level's (start, end) equals the
  census pickle's full line.
- Last five levels (1,310): Levin, residual > 10 floor: 1,222 (1,179 not tiling, 43 tiling);
  Levin, residual ≤ 10 floor: 17 (all tile); direct (CC): 71 (all ≤ 10 floor). residual/floor
  median 1.97e4, p10 861, p90 2.52e5, max 8.28e5. Parent's computed ends equal a and b on 567,
  not on 743.
- All levels (5,502): Levin residual > 10 floor 3,867, of which 3,777 do not tile (97.7 %).
- Per call (last five levels): on 222 calls every Levin level with residual > 10 floor fails to
  tile; on 40 some do and some do not.
- End nodes are (a+b)/2 ± (b−a)/2 at every order, so tiling at the solve's own order (24 Levin,
  47 CC) and at 24 agree on every level.

### Classification, per sampled call

Rule class: (a) tiling, (b) stationary point, (c) other, by `scripts/classify.py`'s fixed rule. Deep levels = depths 16–20; flagged = Levin parent with residual > 10 floor; attributed = no tile, pinned residual ≤ 10 floor, and the residual's change within 10 % (+ floor) of the offset terms. Labels of non-attributed levels are the post-hoc reading.

| # | rtol | block | unit, call | rule class | flagged / attributed (d16–20) | non-attributed deep levels | (b) test: min\|θ′\| / max_span\|θ′\| on the d16 region (θ′ there) | θ′ sign change in span | reading: tiling holds the deep levels |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1e-7 | G+q+r | g-1e-7-1-2174 #1 | **c** | 2 / 0 | d18: other: end-node offset (mp, computed - pinned); d19: Levin P vs CC halves | 0.00014 (-3.79e+05 at x=5.63489) | x=5.63671 (z=279.539), θ″=2.07e+08 | no |
| 1 | 1e-7 | G+q-r | g-1e-7-2-1444 #4 | **c** | 5 / 4 | d20: Levin P vs CC halves | 0.55 (-4.04e+06 at x=13.2175) | none | yes |
| 2 | 1e-7 | G-q+r | g-1e-7-2-1627 #3 | **a** | 5 / 5 | — | 0.24 (-2.13e+07 at x=11.091) | none | yes |
| 3 | 1e-7 | G-q-r | g-1e-7-3-34370 #4 | **c** | 4 / 3 | d19: Levin P vs CC halves | 0.0023 (-6.75e+05 at x=15.8648) | none | yes |
| 4 | 1e-7 | J1 | g-1e-7-3-13510 #5 | **a** | 5 / 5 | — | 0.36 (4.01e+10 at x=4.75152) | none | yes |
| 5 | 1e-7 | J2 | g-1e-7-2-34485 #17 | **c** | 5 / 4 | d20: offset-dominated, prediction off 0.27 | 0.064 (1.1e+09 at x=2.16011) | none | yes |
| 6 | 1e-7 | J3 | g-1e-7-1-38737 #11 | **b** | 4 / 3 | d19: Levin P vs CC halves | 6.3e-07 (6.02e+05 at x=-5.52271) | none | yes |
| 7 | 1e-7 | J4 | g-1e-7-2-34718 #11 | **c** | 5 / 3 | d18: offset-dominated, prediction off 0.10; d19: offset-dominated, prediction off 0.10 | 2.6e-05 (-2.92e+06 at x=-2.76716) | none | yes |
| 8 | 1e-7 | Y1 | g-1e-7-3-37319 #6 | **c** | 5 / 0 | d16: offset-dominated, prediction off 0.11; d17: offset-dominated, prediction off 0.11; d18: offset-dominated, prediction off 0.33; d19: offset-dominated, prediction off 0.33; d20: offset-dominated, prediction off 0.33 | 0.032 (4.73e+06 at x=2.32511) | none | yes |
| 9 | 1e-7 | Y2 | g-1e-7-1-35340 #18 | **c** | 5 / 4 | d20: offset-dominated, prediction off 0.14 | 0.57 (4.61e+06 at x=2.04234) | none | yes |
| 10 | 1e-7 | Y3 | g-1e-7-1-35332 #10 | **c** | 4 / 3 | d19: Levin P vs CC halves | 0.0049 (7.15e+05 at x=-4.89602) | none | yes |
| 11 | 1e-7 | Y4 | g-1e-7-3-36020 #20 | **a** | 5 / 5 | — | 0.68 (-3.17e+10 at x=4.51623) | none | yes |
| 12 | 1e-8 | G+q+r | g-1e-8-1-2174 #1 | **c** | 2 / 0 | d18: other: end-node offset (mp, computed - pinned); d19: Levin P vs CC halves | 0.00014 (-3.79e+05 at x=5.63489) | x=5.63671 (z=279.539), θ″=2.07e+08 | no |
| 13 | 1e-8 | G+q-r | g-1e-8-3-34220 #4 | **a** | 4 / 4 | — | 0.47 (-1.08e+07 at x=13.3914) | none | yes |
| 14 | 1e-8 | G-q+r | g-1e-8-2-1627 #3 | **a** | 5 / 5 | — | 0.24 (-2.13e+07 at x=11.091) | none | yes |
| 15 | 1e-8 | G-q-r | g-1e-8-2-1549 #4 | **a** | 5 / 5 | — | 0.86 (-3.26e+08 at x=9.7609) | none | yes |
| 16 | 1e-8 | J1 | g-1e-8-2-34708 #5 | **a** | 5 / 5 | — | 0.057 (3.1e+09 at x=5.37808) | none | yes |
| 17 | 1e-8 | J2 | g-1e-8-3-13913 #9 | **a** | 5 / 5 | — | 0.49 (1.08e+10 at x=7.51528) | none | yes |
| 18 | 1e-8 | J3 | g-1e-8-1-2368 #9 | **b** | 5 / 4 | d20: Levin P vs CC halves | 6.9e-07 (8.99e+05 at x=-4.66652) | none | yes |
| 19 | 1e-8 | J4 | g-1e-8-3-36019 #19 | **c** | 4 / 2 | d16: offset-dominated, prediction off 0.18; d17: offset-dominated, prediction off 0.18 | 0.13 (-1.97e+09 at x=2.88373) | none | yes |
| 20 | 1e-8 | Y1 | g-1e-8-0-12932 #14 | **a** | 5 / 5 | — | 0.0091 (4.28e+07 at x=-2.09215) | none | yes |
| 21 | 1e-8 | Y2 | g-1e-8-1-14572 #8 | **b** | 5 / 4 | d20: Levin P vs CC halves | 7.4e-07 (9.46e+05 at x=-4.61595) | none | yes |
| 22 | 1e-8 | Y3 | g-1e-8-2-21874 #12 | **a** | 5 / 5 | — | 0.26 (8.75e+06 at x=-3.97869) | none | yes |
| 23 | 1e-8 | Y4 | g-1e-8-2-1402 #12 | **a** | 5 / 5 | — | 0.16 (-8.42e+06 at x=-1.70639) | none | yes |
| 24 | 1e-8 | J1 | g-1e-8-2-34718 #5 | **a** | 5 / 5 | — | 0.04 (6.29e+10 at x=4.57841) | none | yes |
| 25 | 1e-8 | Y4 | g-1e-8-0-33962 #12 | **c** | 4 / 3 | d19: Levin P vs CC halves | 0.62 (-1.3e+06 at x=-5.58439) | none | yes |
| 26 | 1e-8 | Y1 | g-1e-8-0-37212 #8 | **a** | 5 / 5 | — | 0.91 (1.39e+10 at x=5.66632) | none | yes |
| 27 | 1e-8 | J4 | g-1e-8-3-34419 #21 | **c** | 5 / 3 | d16: offset-dominated, prediction off 0.33; d17: offset-dominated, prediction off 0.33 | 0.47 (2.51e+07 at x=1.85505) | none | yes |
| 28 | 1e-7 | J1 | g-1e-7-2-34708 #5 | **a** | 5 / 5 | — | 0.057 (3.1e+09 at x=5.37808) | none | yes |
| 29 | 1e-7 | J3 | g-1e-7-1-14572 #9 | **a** | 5 / 5 | — | 0.036 (4.61e+10 at x=6.17829) | none | yes |

### Classification totals

| rtol | block | calls | (a) | (b) | (c) | reading: tiling holds |
|---|---|---|---|---|---|---|
| 1e-7 | G+q+r | 1 | 0 | 0 | 1 | 0 |
| 1e-7 | G+q-r | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | G-q+r | 1 | 1 | 0 | 0 | 1 |
| 1e-7 | G-q-r | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | J1 | 2 | 2 | 0 | 0 | 2 |
| 1e-7 | J2 | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | J3 | 2 | 1 | 1 | 0 | 2 |
| 1e-7 | J4 | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | Y1 | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | Y2 | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | Y3 | 1 | 0 | 0 | 1 | 1 |
| 1e-7 | Y4 | 1 | 1 | 0 | 0 | 1 |
| 1e-7 | all | 14 | 5 | 1 | 8 | 13 |
| 1e-8 | G+q+r | 1 | 0 | 0 | 1 | 0 |
| 1e-8 | G+q-r | 1 | 1 | 0 | 0 | 1 |
| 1e-8 | G-q+r | 1 | 1 | 0 | 0 | 1 |
| 1e-8 | G-q-r | 1 | 1 | 0 | 0 | 1 |
| 1e-8 | J1 | 2 | 2 | 0 | 0 | 2 |
| 1e-8 | J2 | 1 | 1 | 0 | 0 | 1 |
| 1e-8 | J3 | 1 | 0 | 1 | 0 | 1 |
| 1e-8 | J4 | 2 | 0 | 0 | 2 | 2 |
| 1e-8 | Y1 | 2 | 2 | 0 | 0 | 2 |
| 1e-8 | Y2 | 1 | 0 | 1 | 0 | 1 |
| 1e-8 | Y3 | 1 | 1 | 0 | 0 | 1 |
| 1e-8 | Y4 | 2 | 1 | 0 | 1 | 2 |
| 1e-8 | all | 16 | 10 | 2 | 4 | 15 |
| all | all | 30 | 15 | 3 | 12 | 28 |

Sample, Levin parents at depths 16–20 (144 levels): residual/floor median 3e+04, max 7.78e+05. Flagged levels with the ends pinned (138): residual/floor median 0.00139, max 3.14e+04; ≤ 10 on 127.

### Counterfactual: the call re-run with the end nodes pinned, against R18's census call

| # | rtol | block | census: depth / regions / solves / wall s / conv | pinned: depth / regions / solves / wall s / conv | census value; abserr | pinned value; abserr | \|Δv\| / abserr_census | ref? | census \|v−ref\|/abserr | pinned \|v−ref\|/abserr | ref conv / min abserr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1e-7 | G+q+r | 20 / 50 / 399 / 0.68 / False | 20 / 84 / 671 / 1.47 / False | `0x1.0c5f181ecdbffp-86`; `0x1.75833833809f1p-110` | `0x1.0c5f181ecc74cp-86`; `0x1.758e9b6e3aac7p-110` | 1.35e-05 | no | — | — | — |
| 1 | 1e-7 | G+q-r | 20 / 234 / 1871 / 3.21 / False | 1 / 2 / 15 / 0.02 / True | `0x1.c427c1ab729c6p-89`; `0x1.e75e77abcc7b6p-115` | `0x1.c427c1ab41d00p-89`; `0x1.81cb7dded8c1ap-117` | 0.00156 | NOT CONVERGED | 0.128 | 0.638 | 0.3 |
| 2 | 1e-7 | G-q+r | 20 / 101 / 807 / 0.97 / False | 6 / 7 / 55 / 0.07 / True | `-0x1.8274be9da55efp-95`; `0x1.f350a57b8ff13p-115` | `-0x1.8274be9da55f4p-95`; `0x1.f35087217bd59p-115` | 5.97e-10 | no | — | — | — |
| 3 | 1e-7 | G-q-r | 20 / 103 / 823 / 1.96 / False | 2 / 4 / 31 / 0.04 / True | `0x1.2549f2489da62p-88`; `0x1.ddc88087c8490p-112` | `0x1.2549f248a5a85p-88`; `0x1.cb647a7f43ea2p-112` | 6.55e-05 | no | — | — | — |
| 4 | 1e-7 | J1 | 20 / 48 / 383 / 0.48 / False | 4 / 6 / 47 / 0.06 / True | `-0x1.63f97fdd91ecfp-49`; `0x1.01b82debfe9c4p-79` | `-0x1.63f97fddc689ap-49`; `0x1.de35914b57f3dp-75` | 0.051 | no | — | — | — |
| 5 | 1e-7 | J2 | 20 / 55 / 439 / 0.57 / False | 4 / 7 / 55 / 0.08 / True | `-0x1.676baac06a186p-54`; `0x1.75b944db98124p-80` | `-0x1.676baac22d0c2p-54`; `0x1.3824adc888c37p-81` | 0.0189 | no | — | — | — |
| 6 | 1e-7 | J3 | 20 / 214 / 1711 / 3.27 / False | 4 / 8 / 63 / 0.08 / True | `0x1.e0c532935ac56p-46`; `0x1.22955c1cb99dbp-73` | `0x1.e0c532936ff44p-46`; `0x1.005c17040b0b4p-70` | 0.00228 | no | — | — | — |
| 7 | 1e-7 | J4 | 20 / 67 / 535 / 0.71 / False | 2 / 3 / 23 / 0.04 / True | `-0x1.11ee760ebdd3fp-42`; `0x1.707f83c1a5e7bp-74` | `-0x1.11ee760ebdd1ap-42`; `0x1.5f79a9fac3d26p-72` | 2.45e-05 | no | — | — | — |
| 8 | 1e-7 | Y1 | 20 / 141 / 1127 / 1.45 / False | 2 / 3 / 23 / 0.03 / True | `-0x1.5957d1ae87874p-30`; `0x1.fec7202556ba5p-56` | `-0x1.5957d1ae877cbp-30`; `0x1.feb0798665f4ep-56` | 1.26e-06 | no | — | — | — |
| 9 | 1e-7 | Y2 | 20 / 67 / 535 / 0.75 / False | 4 / 8 / 63 / 0.10 / True | `0x1.9e838ef109fc4p-33`; `0x1.90820b21d2a6ep-65` | `0x1.9e838ef109ff8p-33`; `0x1.8dc0e9e93e321p-65` | 3.17e-05 | no | — | — | — |
| 10 | 1e-7 | Y3 | 20 / 77 / 615 / 1.07 / False | 2 / 4 / 31 / 0.05 / True | `0x1.53f0186022c00p-44`; `0x1.0892b0dd35d3ap-77` | `0x1.53f0186022d19p-44`; `0x1.fec92197c6400p-78` | 0.000519 | no | — | — | — |
| 11 | 1e-7 | Y4 | 20 / 112 / 895 / 1.10 / False | 1 / 2 / 15 / 0.02 / True | `0x1.714f485ba5c17p-52`; `0x1.e037d82ed6005p-78` | `0x1.714f485cc7c26p-52`; `0x1.de907ba32fa42p-76` | 0.00944 | no | — | — | — |
| 12 | 1e-8 | G+q+r | 20 / 48 / 383 / 0.85 / False | 7 / 18 / 143 / 0.23 / True | `0x1.0c5f181ecdc0bp-86`; `0x1.758384c2f61c3p-110` | `0x1.0c6c2cd926acfp-86`; `0x1.febd253628d92p-110` | 2.3e+03 | no | — | — | — |
| 13 | 1e-8 | G+q-r | 20 / 364 / 2911 / 3.45 / False | 1 / 2 / 15 / 0.02 / True | `0x1.f022b8fa455f2p-94`; `0x1.8e28ba0bdd213p-116` | `0x1.f022b8fae2452p-94`; `0x1.4e9792d3061f5p-123` | 0.000385 | no | — | — | — |
| 14 | 1e-8 | G-q+r | 20 / 102 / 815 / 0.97 / False | 3 / 4 / 31 / 0.04 / True | `-0x1.8274be9da55e1p-95`; `0x1.f350a0c941ba3p-115` | `-0x1.8274be9da55d8p-95`; `0x1.f350869124fbdp-115` | 1.07e-09 | no | — | — | — |
| 15 | 1e-8 | G-q-r | 20 / 86 / 687 / 0.79 / False | 2 / 3 / 23 / 0.03 / True | `-0x1.38011d9663c89p-91`; `0x1.a6a5ba739c9cbp-116` | `-0x1.38011d966319fp-91`; `0x1.a63eefc9a64b1p-116` | 1.26e-05 | no | — | — | — |
| 16 | 1e-8 | J1 | 20 / 592 / 4735 / 5.98 / False | 4 / 6 / 47 / 0.06 / True | `-0x1.526e64cfec909p-37`; `0x1.9a6c1160f6134p-69` | `-0x1.526e64cfec47ap-37`; `0x1.532c6c545320ap-67` | 0.000694 | no | — | — | — |
| 17 | 1e-8 | J2 | 20 / 437 / 3495 / 5.23 / False | 4 / 7 / 55 / 0.07 / True | `0x1.131542d88bb7bp-37`; `0x1.8e45dd4c0f674p-66` | `0x1.131542d88b927p-37`; `0x1.8ba3e2f2ce41dp-66` | 4.57e-05 | no | — | — | — |
| 18 | 1e-8 | J3 | 20 / 106 / 847 / 1.24 / False | 4 / 5 / 39 / 0.06 / True | `0x1.88383b7f1dd4dp-41`; `0x1.87cfbff6b54a6p-81` | `0x1.88383b7f1dd5fp-41`; `0x1.e8d7207eff5a6p-70` | 0.00287 | no | — | — | — |
| 19 | 1e-8 | J4 | 20 / 99 / 791 / 1.22 / False | 4 / 6 / 47 / 0.06 / True | `0x1.6db5f5ecfadb1p-48`; `0x1.adabbade77fffp-76` | `0x1.6db5f5ed1bde8p-48`; `0x1.7c1d9e018e6ddp-75` | 0.0048 | no | — | — | — |
| 20 | 1e-8 | Y1 | 20 / 89 / 711 / 0.89 / False | 4 / 10 / 79 / 0.10 / True | `-0x1.986e91189e408p-52`; `0x1.2b5e67536288cp-79` | `-0x1.986e91185feb6p-52`; `0x1.3a89e2777b216p-83` | 0.00651 | no | — | — | — |
| 21 | 1e-8 | Y2 | 20 / 210 / 1679 / 3.14 / True | 5 / 9 / 71 / 0.11 / True | `0x1.0abac5abbf371p-42`; `0x1.e625586ed3c0ap-70` | `0x1.0abac5abbf544p-42`; `0x1.e54c1dd94fd79p-70` | 1.47e-05 | no | — | — | — |
| 22 | 1e-8 | Y3 | 20 / 146 / 1167 / 1.45 / False | 4 / 5 / 39 / 0.05 / True | `-0x1.35f747459e2b1p-46`; `0x1.343feabeaf356p-78` | `-0x1.35f747459d339p-46`; `0x1.6e343f1eb5419p-81` | 0.00314 | no | — | — | — |
| 23 | 1e-8 | Y4 | 20 / 103 / 823 / 1.05 / False | 4 / 7 / 55 / 0.08 / True | `-0x1.1202ddb9e5bd6p-42`; `0x1.a8cef17512fefp-80` | `-0x1.1202ddb9e5bd2p-42`; `0x1.9f6ad7e2b35e7p-80` | 0.000147 | no | — | — | — |
| 24 | 1e-8 | J1 | 20 / 83 / 663 / 0.86 / False | 5 / 16 / 127 / 0.17 / True | `0x1.7059b3f13e60fp-48`; `0x1.0f4ed55763796p-83` | `0x1.7059b3f13e68ep-48`; `0x1.2b8d171ad5beap-83` | 0.000914 | no | — | — | — |
| 25 | 1e-8 | Y4 | 20 / 5394 / 17427 / 37.56 / False | 3 / 6 / 47 / 0.06 / True | `-0x1.856da4f5d0ee7p-52`; `0x1.0c4b38939c8e0p-70` | `-0x1.856da4f8bbbc3p-52`; `0x1.a2517359ee192p-86` | 0.00017 | no | — | — | — |
| 26 | 1e-8 | Y1 | 20 / 91 / 727 / 0.92 / False | 4 / 8 / 63 / 0.08 / True | `0x1.dd62f5b583ec1p-43`; `0x1.30d14947287b0p-71` | `0x1.dd62f5b5897b3p-43`; `0x1.245523bfed0cdp-71` | 0.00114 | no | — | — | — |
| 27 | 1e-8 | J4 | 20 / 87 / 695 / 0.91 / False | 4 / 7 / 55 / 0.08 / True | `0x1.b5ae337651871p-42`; `0x1.faa615e5ea6fcp-74` | `0x1.b5ae337655536p-42`; `0x1.57495e0df31f8p-74` | 0.0075 | no | — | — | — |
| 28 | 1e-7 | J1 | 20 / 591 / 4727 / 5.97 / False | 4 / 6 / 47 / 0.06 / True | `-0x1.526e64cfec909p-37`; `0x1.9a76934cff23ep-69` | `-0x1.526e64cfec47ap-37`; `0x1.532c6c545320ap-67` | 0.000694 | no | — | — | — |
| 29 | 1e-7 | J3 | 20 / 284 / 2271 / 2.94 / False | 5 / 11 / 87 / 0.11 / True | `0x1.88383b27a9b0cp-41`; `0x1.5e09144a13fddp-67` | `0x1.88383b27a9b3ap-41`; `0x1.5de06ce537441p-67` | 5.01e-07 | no | — | — | — |

### Depth before and after pinning (the 30 sampled calls)

| max depth | census calls | pinned calls |
|---|---|---|
| 1 | 0 | 3 |
| 2 | 0 | 5 |
| 3 | 0 | 2 |
| 4 | 0 | 14 |
| 5 | 0 | 3 |
| 6 | 0 | 1 |
| 7 | 0 | 1 |
| 20 | 30 | 1 |

Converged: census 1 of 30, pinned 29 of 30. Regions summed: census 10181, pinned 274. Solves summed: census 55694, pinned 2162. Wall summed: census calls 91.6 s, pinned 3.5 s.

References attempted: [1]; converged to 1/100 of the smaller abserr: [].


## The reference that stopped step 4 (verbatim, `out/17_ref_01.txt`, `out/ref_batch_progress.txt`)

```
sample 01 conv_diff 2.7209433343957533e-36 min abserr 9.069994260877289e-36 ratio 0.2999939422379052 converged False
STOP: sample 01 reference did not converge
```
S192 = `0x1.c427c194a62cfp-89`, S384 = `0x1.c427c19be1f42p-89` (2.853501101367909e-27), 16,058
panels (D = 512 rad, hmax = 0.005), cancellation Σ|terms|/|S| = 1.5e4, linear rounding bound
4.2e-38. Descriptively only (not a valid reference): taking S384, census |v−ref| = 0.128 abserr,
pinned 0.638 abserr. A likely cause: 192 Gauss–Legendre nodes across a 512-rad panel is about 2.4
nodes per period, below GL's ~π-per-period threshold, so S192 itself is unconverged and the pair
measures S192's error; a 384/768 pair would test this. Not run (stop rule).

## References not launched (estimated over 45 min at six workers; `out/16_ref_size.txt`)

Command for each: `PYTHONPATH=<tree g>:<scripts> python -B scripts/refq.py run state/captured.json N`.
Samples 3 (7.0e3 s), 15 (9.2e3 s): 2–2.6 h. Samples 0, 12, 20 (7–10e4 s), 5, 17, 19, 26 (2.3–3.5e5 s),
4, 7, 11, 16, 28 (7e5–1.7e6 s), 6, 18, 21, 24, 29 (1.6–2.8e7 s): 20 h to ~1 year. Under 45 min but
not run because of the stop: 2, 8, 9, 10, 13, 14, 22, 23, 25, 27.
