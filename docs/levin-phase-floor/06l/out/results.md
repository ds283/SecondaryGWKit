# levin-phase-floor 06l — results

<!-- provenance: written 2026-11-05 by the 06l agent on mps026298 from out/k0/summary.json (k0_summary_06l.py,
out/cmd/81), out/k0/compare_identity.json and compare_routeB.json (compare_k0.py, out/cmd/k0partL/51-52),
out/partL_summary.json and out/partL_leaves.csv (score_l_06l.py, out/cmd/80), out/partL_l2/account.json
(l2_account_06l.py, out/cmd/85), out/lg_draws/compare.txt (lg_compare_06l.py, out/cmd/84), out/census_projection.json
(out/cmd/86), out/partC/routeB.json and summary.json (analyse_c_06l.py, out/cmd/101), out/partC/against_A.json
(against_a_06l.py, out/cmd/104) and out/partC/h1.json (score_h1_06l.py, out/cmd/103). Tree af4c4dc1 (production code
1c5336e's, 06g's); scratch copy fingerprint 0e32f24a… before and after; 06g's census checkpoint 03adcbba…, route 1's
36e3dd4c…, route B's 5cb3d3c0…. Route B's variant is scripts/variants/levin_quadrature_routeB.py (5aa6178a…). Every
number below is in those files. This file proposes no code. -->

**Route B.** One change in `_check_leaf`'s noise clause. A checked leaf is trusted if r < 1/4, or if r < 1 and
Δ1 ≤ ν_B, where ν_B = φ_P + Σ_{Levin halves} φ_h + Σ_{CC halves} φ⁻_h + |Σ_{CC halves} E_h|. On a Levin parent a
Clenshaw–Curtis half contributes nothing (the user's modification).

**The verdict in one line.** L1, C1, C2, H1, S1 and K1 hold, and **L2 fails** (63 of 75). So by §4's rule (L1, L2,
C2, H1, S1 and K1) **B does not pass as a candidate remedy as written**. Its one failure is at leaf level, by
0.04–0.41 % of ν_B on 12 leaves (below). Census-wide it holds every safety rule.

## Gate K0 — holds

Run `levin-phase-floor-06l-k0partL-20261105T091942`, on 06k's 100 units (`out/k0/units.json`, the same file as 06k's).

| | units equal to 06g | what was compared |
|---|---|---|
| identity variant | 100 of 100 | every key, compared whole, counters included |
| route B | 100 of 100 | the 1,862 calls with no firing (16 units compared whole) |

- Route B fires in 248 of 2,110 calls on K0: 4,393 firings, and 0 withdrawals (a leaf 06g trusts that B does not).
- No firing on K0 is outside route 1's yardstick.

## Part L — leaf level

Run `levin-phase-floor-06l-partL-20261105T095407`.
- **The probes:** 46 of 46 captures reproduce 06g's census bit for bit. The probes reproduce the census's Δ1, Δ2, r
  and φ_P on 142 of 142 L-a leaves and 1,882 of 1,882 L-c leaves.
- **Against 06k's table:** every one of 06k's 42 columns equals 06k's `partL_leaves.csv` on all 2,046 rows.

| | L-b: 06h carrying leaves trusted (17 distinct, 22 evaluations) | L-a: slow leaves trusted (of 142) | first population, Δ1/φ_P < 2 (of 75) | second population (of 67) | slow leaves trusted at Δ1/φ_P ≥ 5.8 | L-c: 06g-trusted leaves whose decision changes (of 1,882) |
|---|---|---|---|---|---|---|
| route 1 (06k) | 0 | 114 | 63 | 51 | 30 | 0 |
| **route B** | **0** | **85** | **63** | **22** | **7** | **0** |

**Gate L3 holds on all 2,046 leaves.**
- ν_B ≤ ν1·(1 + 10⁻¹²) everywhere. The largest ν_B/ν1 is 1 + 2.2 × 10⁻¹⁶; on 78 leaves ν_B exceeds ν1 by about 1 ulp.
- B trusts no leaf that route 1 does not.
- The variant's ν_B and an exact fsum of its terms agree to 2.2 × 10⁻¹⁶ relative.

**L2's 12 untrusted leaves** are listed in `out/partL_l2/account.json`.
- **What they share:** every one is Clenshaw–Curtis over Clenshaw–Curtis, with r 0.491–0.501, Δ1/φ_P 1.488–1.503 and
  Δ1/ν_B 1.0004–1.0041.
- **Why ν_B falls short:**
  - On each, E_P and E_L + E_R have opposite signs. So the node-error difference |E_P − (E_L + E_R)| equals
    |E_P| + |E_L + E_R|, and the triangle inequality of §0.2 holds with equality.
  - Δ1 exceeds that difference by 0.20–0.61 %, which is 06j's "to 0.6 %".
  - The rest of ν_B is too small to cover the excess: per leaf, (φ_P − |E_P|) is 0.07–0.15 % of ν_B, and
    φ⁻_L + φ⁻_R is 0.05–0.12 %.
- **What that means for §0.2's guarantee:** it holds to first order only. Where the bound is tight, the excess of Δ1
  over the first-order node error is not declared by any floor.
- **The trusted 63:** 56 of them have opposite signs too. Their Δ1/ν_B has median 0.996 and maximum 0.99999.

**The second population** (descriptive): B trusts 22 of 67.
- On those 22, Δ1/(|E_P| + |E_L + E_R|) runs from 0.30 to 4.2 (median 1.24), and Δ1/ν_B from 0.22 to 0.986.
- On the same 22, the halves' node errors cancel to 0.07–3.7 % of |E_L| + |E_R|, and Δ1/ν1 is 0.002–0.037.
- So B trusts them where the signed node errors and the floors' other terms happen to cover Δ1, not on the
  cancelling term route 1 adds.
- The orchestrator's estimate from 06k's table was 85 trusted in all, 7 of them at Δ1/φ_P ≥ 5.8. Both match. Its split,
  75 + 10, does not: the measured split is 63 + 22.

**The LiouvilleGreen draws** (06g's 20 seeded draws, 240 units; `out/lg_draws/compare.txt`). Every test passes under
every mode. The identity variant equals the tree bit for bit on all 4,320 Levin calls and every case.

| | Levin calls firing | convergence changes | near-singular cases (ε ≤ 10⁻⁴, 560) | other cases (520) |
|---|---|---|---|---|
| route 1 (06k) | 290 | 2 gained (seed 14, `test_JJJ` #9, `test_YJJ` #6) | no honesty change | no honesty change |
| route B | 122 (290 firings; 0 withdrawals) | none | no honesty change (dishonest 432 → 432; true error changed on 3) | no honesty change (true error changed on 13) |

## Part C — the census under B

Run `levin-phase-floor-06l-census-routeB-20261105T100126`.
- It finished `done`, 18,560 of 18,560 units, exit 0.
- Every call matched its 06g and route 1 call by kind, index, label and span.
- The 720 are 06a's stationary-point rows; "outside" means outside them.

### Convergence (C1, C2)

| V8's lost calls converged / lost under 06g | 1e-7 outside the 720 | 1e-8 outside | 1e-7 in the 720 | 1e-8 in the 720 |
|---|---|---|---|---|
| route 1 (06k) | 792 / 801 (98.9 %) | 818 / 830 (98.6 %) | 213 / 316 | 218 / 340 |
| **route B** | **749 / 801 (93.5 %)**: 51 untrusted leaf, 1 node_limited | **773 / 830 (93.1 %)**: 53 untrusted leaf, 4 node_limited | 181 / 316 | 188 / 340 |

**Calls 06g converges that B does not** (C2).
- Outside the 720 there are 4, all group calls at 1e-8, all `node_limited`, with abserr 1.0009–1.032× 06g's:
  - `g-1e-8-0-12591` #4;
  - `g-1e-8-1-35186` #6;
  - `g-1e-8-1-1748` #4;
  - `g-1e-8-0-14312` #5.
- In the 720 there is 1, also `node_limited` (`g-1e-8-3-35036` #4, 1.054×).
- Route 1 lost 7, all `node_limited`. Two are lost under B too (`g-1e-8-0-12591` #4, `g-1e-8-1-35186` #6), and B converges the other five.

Against 06g, B converges 1,522 calls outside the 720 that 06g does not, and 369 in it.

### Depth

| group calls with a leaf at `depth_max` | outside the 720 (1e-7 / 1e-8) | in the 720 |
|---|---|---|
| 06g | 966 / 996 | 357 / 385 |
| route 1 | 14 / 16 | 118 / 139 |
| **route B** | **72 / 76** | 155 / 176 |

### Moves (S1)

- **Outside the 720,** no call moves by more than e_B + e_g: 0 calls, in 0 units and 0 rows (shard-serial).
- **Row totals:** none outside the 720 moves past both total abserrs.
- **In the 720:**
  - one call moves, in one row: `0-2316`'s `G+q+r` call #1, at both rtols;
  - its ratios are 2,513 (1e-7) and 7,406 (1e-8);
  - B converges it in 68 and 73 regions, against 06g's 201 and 206, which do not converge;
  - so do that row's totals, at 48 and 290;
  - B and route 1 agree on it within their errors.

### Firings

A firing trusts a leaf that 06g's rule leaves untrusted. Totals: 232,781 outside the 720 and 38,249 in it.

| parent / halves | outside the 720 | in the 720 |
|---|---|---|
| Clenshaw–Curtis / CC + CC | 222,942 | 36,348 |
| Levin / Levin + Levin | 9,833 | 1,901 |
| Levin / CC + Levin (the Levin half's floor covers Δ1) | 6 | 0 |

- **Δ1/ν_B, outside the 720:**
  - < 0.5: 6,275;
  - [0.5, 0.9): 56,244;
  - [0.9, 1]: 170,262.
- **r, outside the 720:**
  - < 0.5: 88,939;
  - [0.5, 0.75): 91,949;
  - [0.75, 1): 51,893.
- **Depth:** depths 9–15 hold 205,337 of the firings.
- **Inclusion:** no firing is outside route 1's yardstick, and none withdraws trust.

**K1 holds.**
- B makes 9,839 firings on Levin parents outside the 720, and none at Δ1/φ_P ≥ 5.8.
- The largest Δ1/φ_P among them is 2.85.
- In the 720 there are 1,901 such firings, again none at ≥ 5.8.
- Route 1 had 48,131 at ≥ 5.8 on Levin parents, up to 1.1 × 10⁴.

**Clenshaw–Curtis parents** (descriptive, outside the 720; not a test of noise there, §4):

| Δ1/φ_P | [1, 2) | [2, 5.8) | [5.8, 25) | [25, 100) | [100, 1,000) |
|---|---|---|---|---|---|
| firings | 93,354 | 64,719 | 48,501 | 14,148 | 2,220 |

- The largest Δ1/φ_P on a Clenshaw–Curtis parent is 862.
- **Against the signed node errors,** Δ1/(|E_P| + |E_L + E_R|) on CC + CC firings:
  - < 1: 153,188;
  - [1, 1.01): 52,169;
  - [1.01, 2): 17,011;
  - [2, 5.8): 464;
  - ≥ 5.8: 110.
- So 69 % of these firings have Δ1 within the two node-error terms, and 92 % within 1 % of them.

### Cost

Each census wall runs from the run's manifest `created` to its last heartbeat; the ratios are against 06g.

| | census wall | unit wall (1e-7 + 1e-8) | group-call regions (1e-7 + 1e-8) |
|---|---|---|---|
| 06g | 20,361 s (5 h 39 min) | 58,535 s | 2,380,310 |
| 06h | 9,193 s (2 h 33 min) | 18,509 s (0.32) | 556,898 (0.23) |
| route 1 | 16,758 s (4 h 39 min) | 45,803 s (0.78) | 1,847,162 (0.78) |
| **route B** | **18,340 s (5 h 6 min)** | **50,495 s (0.86)** | **1,993,637 (0.84)** |

- **Per rtol,** route B's unit walls are 24,063 s at 1e-7 and 26,432 s at 1e-8 (06g: 27,419 and 31,116 s; route 1:
  21,896 and 23,907 s).
- **The projection was 17,347 s,** and the run took 5.7 % more. The backstop reported twice, its last P_o 18,033 s,
  and stopped nothing.
- **The host was busier this time.** Today's identity run on K0 took 1.11× 06g's census unit walls on the same units,
  with a load of about 15 from system services. So the unit-wall ratio overstates B's cost against censuses run on
  quieter days. The region ratio, 0.84, does not depend on load.

### Against route 1's census

| | 1e-7 outside the 720 | 1e-8 outside | in the 720 |
|---|---|---|---|
| route 1 converges, B does not | 43 | 48 | 66 |
| B converges, route 1 does not | 0 | 6 (route 1 `node_limited`) | 3 |
| calls moving by more than e_B + e_route1 | 0 | 0 | 0 |

**What holds B's 91 calls outside the 720.**
- 87 end on an untrusted leaf, and 4 end `node_limited`.
- B's holding leaves (untrusted at `depth_max`) number 199, all Clenshaw–Curtis over Clenshaw–Curtis at depth 20:
  - r runs from 0.27 to 26.8, median 1.46;
  - Δ1/φ_P from 0.11 to 32.6, median 0.53;
  - Δ1/ν_B from 0.037 to 11.8, median 0.34;
  - Δ1/ν1 from 10⁻⁴ to 11.7, median 0.33;
  - the halves' node errors |E_L + E_R|/(|E_L| + |E_R|) have median 0.9999, so they barely cancel.
- 167 of the 199 are leaves route 1's yardstick would not trust either. Most have r ≥ 1.
- These are the leaves B's refinement reached after it withheld trust higher up. Route 1 trusted that ancestor instead,
  and never made them.

**A's difference from route 1, as far as route 1's census shows it** (`out/partC/against_A.json`).
- **Method:** each of route 1's firings in these calls is matched by span to the leaf where B withheld trust that
  route 1's yardstick would give, which supplies the halves' kinds. A's decision is then computed from route 1's own
  numbers.
- **Matched firings:** 640.
  - A withholds 409. All are on Levin parents, 407 with two Clenshaw–Curtis halves and 2 with one. Their Δ1/φ_P has
    median 141, up to 1,029, and (φ_L + φ_R)/φ_P has median 239.
  - A keeps 231, all on Clenshaw–Curtis parents. There B's signed yardstick is short by Δ1/ν_B 1.000004–3.1
    (median 1.010).
- **Unmatched:** 2,134 of route 1's firings lie where B's refinement had already left route 1's tree.
- **Of the 91 calls:**

| | calls |
|---|---|
| route 1 recovered them only through Levin parents with a CC half (A loses them as B does) | 14 (7 at 1e-7, 7 at 1e-8) |
| only through Clenshaw–Curtis parents (A decides as route 1 there) | 2 |
| through both (A withholds part of route 1's trust; its outcome is not in route 1's census) | 75 |

  In the 720 the same split is 11, 14 and 41 of 66.
- **So A recovers at most 77 of these 91 calls,** and at least 14 of route 1's extra recoveries rest wholly on the
  modification's leaves.
- **Where B and A part:** on the Clenshaw–Curtis parents of the 77, by 0–3 % of ν_B on most of the leaves concerned.

## The predictions

| | rule | route B | number |
|---|---|---|---|
| **L1** | of the 17 06h carrying leaves, B trusts none | **holds** | 0 of 17 |
| **L2** | B trusts all 75 first-population slow leaves | **fails** | 63 of 75; the 12 others at Δ1/ν_B 1.0004–1.0041 |
| **C1** | ≥ 90 % of V8's lost calls outside the 720 converge, at each rtol | **holds** | 93.5 % (749/801), 93.1 % (773/830) |
| **C2** | every call outside the 720 that 06g converges and B does not ends `node_limited` | **holds** | 4 calls, all `node_limited` (1e-8) |
| **H1** | on the 115 matched references outside the 720, \|v_B − ref\|/e_B ≤ max(1, 06g's ratio) | **holds** | 115 of 115; 114 within own error; worst v03 1.182 (06g 1.203, route 1 1.185) |
| **S1** | no call outside the 720 moves by more than e_B + e_g | **holds** | 0 calls |
| **K1** | every trust firing on a Levin parent outside the 720 has Δ1/φ_P < 5.8 | **holds** | 9,839 firings, max 2.85 |

**H1 and the references.**
- 124 references are matched (06g's 102, 06j's 22), with 06g's ratio reproduced on 124 of 124.
- V1's 12 are unscored, as in 06k (deviation 24 there, on the user's ruling).
- **In the 720** (not scored), B leaves all nine as 06g has them: z00 at 0.018, z12 at 1,681.3, and the rest at most
  0.002.
- **Where B moves a ratio:** B's ratio differs from 06g's on 31 of the 115 outside the 720, and none crosses its
  bound.

## B beside route 1 and A (no code is proposed)

| | L2 (leaf level) | C1 at 1e-7 / 1e-8 | C2 losses (all `node_limited`) | H1 worst | S1 | K1: Levin-parent firings at Δ1/φ_P ≥ 5.8 | unit wall / regions vs 06g |
|---|---|---|---|---|---|---|---|
| route 1 | 63 / 75 (06k's rule: 80.3 % of 142, holds) | 98.9 / 98.6 % | 7 | 1.185 | 0 | 48,131 (fails) | 0.78 / 0.78 |
| **route B** | **63 / 75 (fails)** | **93.5 / 93.1 %** | **4** | **1.182** | **0** | **0 (holds)** | **0.86 / 0.84** |
| A (from route 1's census) | — | between B's and route 1's on the calls above: at most 77 of route 1's 91 extra | — | — | — | 0 by construction | — |

- **What B is.** B trusts only where ν_B, a sum of declared floors and the halves' signed node errors, covers Δ1. Its
  census holds every safety rule (L1, C2, H1, S1, K1).
- **What B gives up.** It recovers about 5 points less of V8's lost calls than route 1. It leaves 72 and 76 group calls
  at `depth_max` outside the 720 (route 1: 14 and 16).
- **Where its losses sit.** They are on Clenshaw–Curtis leaves whose Δ1 exceeds the signed node-error yardstick by a
  fraction of a per cent, and on their descendants.
- **L2 as written fails on 12 such leaves.** §0.2's triangle-inequality guarantee leaves no margin for the 0.2–0.6 %
  by which Δ1 exceeds the first-order node error.
- **What route 1's extra recovery rests on.** Partly on Levin parents with Clenshaw–Curtis halves, which the
  modification (A) removes: at least 14 of its 91 extra calls rest wholly on them, and 75 more in part.
- **What decides how much A keeps.** A keeps route 1's Clenshaw–Curtis-parent trust, which is not bounded by any floor
  (§0.1). How much of route 1's recovery A keeps is not determined by route 1's census alone.

Which route, if any, becomes code is for the user.
