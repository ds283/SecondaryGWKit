# levin-phase-floor 06k — results

<!-- provenance: written 2026-11-05 by the 06k agent on mps026298 from out/k0/summary.json (k0_summary.py, out/cmd/96),
out/k0/compare_*.json (compare_k0.py, out/cmd/k0partL/50-54, out/cmd/route2lit/55), out/partL_summary.json (score_l.py,
out/cmd/80b), out/partL_route2lit_withdrawals.csv and out/partL_route2lit_by_solve_kind.json (withdrawals_l.py,
out/cmd/98), out/lg_draws/compare.txt (lg_compare_06k.py, out/cmd/97), out/partC/summary.json and out/partC/<route>.json
(analyse_c.py, out/cmd/101), out/partC/movers_*.json (movers_c.py, out/cmd/102), out/partC/h1.json (score_h1.py,
out/cmd/103) and out/cmd/122 (eta_route3.py). Tree 89370609 (production code 1c5336e's, 06g's); scratch copy fingerprint
0e32f24a… before and after; 06g census checkpoint 03adcbba…, 06h's 3104c18e…; the four 06k census checkpoints
36e3dd4c… (route 1), 8363ae82… (route 2), 7fb9c41e… (route 2 literal), 89afd6da… (route 3). Every number below is
in those files. This file proposes no code. -->

**The four routes measured.** Route 2 is measured two ways, on the user's ruling at planned stop 1: *"We had better
run that fourth census"*.
- **Route 1:** the noise clause compares Δ1 with ν1 = φP + φL + φR.
- **Route 2, as built:** where rule (b) finds a leaf untrusted, rule (b) is applied to values corrected by −E. A
  firing declares max(residual, Δ1c)/(1 − r_c). 06g's trusted leaves keep 06g's decision and declaration.
- **Route 2, literal:** every checked leaf is decided and declared on the corrected values, so trust at r < 1/4 can
  be withdrawn.
- **Route 3:** at each stop test, the longest prefix of untrusted refinable leaves, ranked by D, whose D sums to at
  most 10⁻³ of the request is left. It stays as built, on the user's ruling: *"stick with what has been built."*

**The verdict in one line.** No route passes as a candidate remedy by §4's rule (L1, C2, H1, S1, and K1 for
routes 1 and 2). Each fails C2, and each other than route 1 fails S1 or H1 as well.

## Gate K0 — holds

Each run below took 100 units: 06j's 46 calls' units, plus 54 drawn by the seeded rule written before any unit ran
(`state/predictions_and_rules_20261104T0915Z.md` §2, drawn into `out/k0/units.json`).

| Run | Units equal to 06g | What was compared |
|---|---|---|
| tree's module (control) | 100 of 100 | every key |
| identity variant | 100 of 100 | every key |
| route 1 | 100 of 100 | the 1,855 calls with no firing |
| route 2 | 100 of 100 | the 1,837 calls with no firing |
| route 2 literal | 100 of 100 | the 1,032 calls with no firing of any kind |
| route 3 | 100 of 100 | the 683 calls with no firing |

- **What "every key" covers:** value, abserr, regions, region arrays, `converged`, checked leaves, the premise
  counter, solves by purpose, the budget record, dispatcher counts, scans, deep lines and the floor's terms.
- **What may differ:** timing, provenance and `cache_built` (what a worker's producer cache built). `cache_built`
  differs on 95 units under the control as under the identity variant, so it reflects the worker grouping, not the
  variant.
- **Route 2 literal:** clause 2 does not apply, by the orchestrator's ruling, because the variant changes every
  checked Clenshaw–Curtis leaf's declaration. Every such change counts as a firing, so its calls with no firing were
  still compared, and all 1,032 are equal.

## Part L — leaf level, and the LiouvilleGreen draws

Probes of each route's rule in isolation, on the leaf's own span in 06g's code. They reproduce the census's Δ1, Δ2, r
and φP on 142 of 142 L-a leaves and 1,882 of 1,882 L-c leaves (`out/partL_leaves.csv`).

| | L1: 06h carrying leaves trusted or left (17 distinct, 22 evaluations) | L2: slow leaves trusted or left (of 142) | L-c: 06g-trusted leaves whose decision changes (of 1,882) | slow leaves accepted at Δ1/φP ≥ 5.8 |
|---|---|---|---|---|
| route 1 | 0 | 114 (80.3 %) | 0 | 30 (max 24.8) |
| route 2 | 0 | 57 (40.1 %) | 0 | 6 (max 14.1) |
| route 2 literal | 0 | 57 (40.1 %) | **326 withdrawn** | 6 |
| route 3 | **1** (c01) | 142 (100 %, all left) | 0 | 34 |

**By population.** Of the slow leaves with Δ1/φP < 2 and ≥ 2 respectively:
- route 1 trusts 63 of 75 and 51 of 67;
- route 2 trusts 45 of 75 and 12 of 67.

**c01 under route 3.** This is 06h's carrying leaf on `2-33416` #6: depth 0, r = 0.55, Δ1/φP = 1.26 × 10⁴. Its D is
1.7 × 10⁻⁴ of η times the call's request, because the request is set by atol, far above the call's error.

**Route 2 literal's withdrawals** are listed in full in `out/partL_route2lit_withdrawals.csv`:
- 265 at r < 1/4;
- 300 with r_c ≥ 1 (with −E removed, Δ1c is tiny);
- 303 on Clenshaw–Curtis parents and 23 on Levin parents;
- 44 at `depth_max`.

**Which leaves the correction touches.** It is per solve, so a leaf is unchanged only if all seven of its solves
are Levin (`out/partL_route2lit_by_solve_kind.json`):

| L-c leaves | count | changed |
|---|---|---|
| Levin parents, all seven solves Levin | 621 | none |
| Levin parents with a Clenshaw–Curtis solve among the six others | 388 | all corrected; 23 change decision |
| Clenshaw–Curtis parents | 873 | 872 corrected |

**The LiouvilleGreen draws** (06g's 20 seeded draws, against the tree's run; `out/lg_draws/compare.txt`). Every test
passes under every mode. The identity variant equals the tree bit for bit on all 4,320 Levin calls and every case.

| | Levin calls firing | convergence changes | near-singular cases (ε ≤ 10⁻⁴, 560) | other cases (520) |
|---|---|---|---|---|
| route 1 | 290 | 2 gained (seed 14, `test_JJJ` #9 and `test_YJJ` #6) | no honesty change | no honesty change |
| route 2 | 2,294 | none | 6 dishonest → honest (reported error up 1.1–4.2×, true error unchanged) | no change |
| route 2 literal | 3,761 | none | 10 dishonest → honest; 2 honest → dishonest (true/reported 1.05, 1.09) | no change |
| route 3 | 3,379 | **2 lost** (seed 13, `test_JJJ` #5 and #6; abserr 1.007× and 1.015× the request) | 1 honest → dishonest (true/reported 2.21, flagged converged) | no change |

## Part C — the four censuses against 06g's

Every census finished `done`, 18,560 of 18,560 units, exit 0. Every call matched its 06g call by kind, index,
label and span.

### Convergence (C1, C2)

**V8's lost calls**, given as converged under the route / lost under 06g:

| | 1e-7 outside the 720 | 1e-8 outside | 1e-7 in the 720 | 1e-8 in the 720 |
|---|---|---|---|---|
| route 1 | 792 / 801 (98.9 %): 8 untrusted leaf, 1 node_limited | 818 / 830 (98.6 %): 9, 3 | 213 / 316 | 218 / 340 |
| route 2 | 801 / 801 (100 %) | 828 / 830 (99.8 %): 2 node_limited | 227 / 316 | 239 / 340 |
| route 2 literal | 794 / 801 (99.1 %): 7 untrusted | 821 / 830 (98.9 %): 7 untrusted, 2 node_limited | 203 / 316 | 212 / 340 |
| route 3 | 801 / 801 (100 %) | 829 / 830 (99.9 %): 1 node_limited | 316 / 316 | 339 / 340 |

**Calls 06g converges that the route does not.** Every one is a group call. The prompt's 720 are 06a's
stationary-point rows, as everywhere in this file.

| | outside the 720 | why | in the 720 |
|---|---|---|---|
| route 1 | 7, all at 1e-8 | `node_limited`; abserr 1.00–1.08× 06g's | 0 |
| route 2 | 5, all at 1e-8 | `node_limited` | 1 |
| route 2 literal | **53** (25 at 1e-7, 28 at 1e-8) | 51 untrusted leaf at `depth_max`, 2 node_limited | 30 |
| route 3 | 1 at 1e-8 | `node_limited` | 0 |

### Depth (V5's count)

Group calls with a leaf at `depth_max`, outside the 720 at 1e-7 and 1e-8, then in the 720 at both rtols:

| | outside the 720 (1e-7 / 1e-8) | in the 720 (1e-7 / 1e-8) |
|---|---|---|
| 06g | 966 / 996 | 357 / 385 |
| 06h | 0 / 1 | 11 / 17 |
| route 1 | 14 / 16 | 118 / 139 |
| route 2 | 2 / 2 | 105 / 122 |
| route 2 literal | 42 / 43 | 144 / 166 |
| route 3 | 0 / 0 | 0 / 1 |

### Moves (S1)

Calls whose value moves by more than e_route + e_g against 06g; leaves in `out/partC/movers_<route>.json`.

| | calls outside the 720 (rows) | worst ratio | in the 720 | row `total`s moving past both errors |
|---|---|---|---|---|
| route 1 | 0 | — | 2 | 0 outside, 2 in the 720 |
| route 2 | 10 (10) | 1.323 | 0 | 0 |
| route 2 literal | 10 (10), the same calls | 1.323 | 0 | 0 |
| route 3 | 46 (44) | 1.645 | 130 | 0 outside, 108 in the 720 |

- **Routes 2 and 2 literal.**
  - Eight of the movers are single-region analytic J1, Y1 and J2 calls (rows `3-662`, `3-754`, `2-33231`,
    `1-20532`, each at both rtols). The route trusts the root leaf and returns Q. Its corrected declaration leaves
    out the node-error difference between Q and V_P, so the move exceeds both errors.
  - The other two are one G+q+r leaf at depth 11 (`2-21787` #3, both rtols), carrying 75 % of the move.
- **Route 3.**
  - Every mover is a coarse leaf (depth 0 or 1) left at D ≤ 10⁻³ of the request, where 06g refined the same span
    into 7–200 leaves.
  - The left leaf carries 100 % of each move, which is up to 1.6 × (e_route + e_g). So D understated that leaf's
    error by at least about 10³.
  - Examples: `1-10767` #5 (41 → 1 regions), `3-34929` #6 (104 → 2), `2-21853` #6 (200 → 1), and c01's call
    `2-33416` #6 (23 → 1).

### Firings

By depth, Δ1/φP and r, in `out/partC/summary.json`. Here "trust" firings are those that trust a leaf 06g's rule
finds untrusted.

| | firings | trust firings outside the 720 | of them at Δ1/φP ≥ 5.8 | largest Δ1/φP | where they sit |
|---|---|---|---|---|---|
| route 1 | 267,578 | 229,338 | **89,171** | 1.1 × 10⁴ | depths 9–15; r < 0.5: 101,023, [0.5, 0.75): 101,228, [0.75, 1): 65,327 |
| route 2 | 196,599 | 167,281 | **82,565** | 3.7 × 10⁵ | 122,247 at r ≥ 1 (trusted on r_c) |
| route 2 literal | 1,064,182 | 191,623 trust additions | **88,186** | 3.7 × 10⁵ | see below |
| route 3 | 293,871 left | — | — | — | 249,353 at r ≥ 1; 211,133 at depths 0–2 |

- **Route 2 literal by kind:**

  | kind | firings |
  |---|---|
  | `declaration_changed` | 802,091 |
  | `trust_added` | 226,879 |
  | `trust_withdrawn` | 30,158 |
  | `declaration_changed_untrusted` | 5,054 |

- **Route 3:**
  - D against the cap: < 10⁻⁶: 78,195; < 10⁻³: 91,660; < 10⁻¹: 69,053; up to the cap: 54,963.
  - 173,207 calls return left leaves. Their η_min has median 6.7 × 10⁻⁸, p90 4.1 × 10⁻⁴ and maximum 10⁻³.
  - At η = 10⁻⁴, 26,699 of those calls' left sets would not be accepted on their returned partitions
    (`out/cmd/122`).
  - The reading the user ruled against (an untrusted leaf at `depth_max` ranked too) differs on 1 call.

### Cost

Census wall from each run's manifest `created` to its last heartbeat. Ratios are against 06g.

| | census wall | unit wall (1e-7 + 1e-8) | group-call regions (1e-7 + 1e-8) |
|---|---|---|---|
| 06g | 20,361 s (5 h 39 min) | 58,535 s | 2,380,310 |
| 06h | 9,193 s (2 h 33 min) | 18,509 s (0.32) | 556,898 (0.23) |
| route 1 | 16,758 s (4 h 39 min) | 45,803 s (0.78) | 1,847,162 (0.78) |
| route 2 | 13,621 s (3 h 47 min) | 34,986 s (0.60) | 1,557,755 (0.65) |
| route 2 literal | 13,968 s (3 h 53 min) | 36,047 s (0.62) | 1,589,783 (0.67) |
| route 3 | 7,626 s (2 h 7 min) | 17,573 s (0.30) | 644,329 (0.27) |

Per rtol the unit walls are as follows (06g: 27,419 and 31,116 s):
- route 1: 21,896 and 23,907 s;
- route 2: 16,177 and 18,809 s;
- route 2 literal: 16,652 and 19,395 s;
- route 3: 7,346 and 10,227 s.

### Against the references (H1)

`out/partC/h1.json` and `h1_calls.csv` score 124 references, each matched to its census call:
- 06g's 102 (V3's 90 and V8's 12);
- 06j's 22.

Each is scored with the reference value its source used (`ref_30`), and 06g's own ratio is reproduced on 124 of 124.
115 lie outside the 720. V1's 12 are calls of 06g's rehearsal store R24, not census calls; they are unscored, on the
user's ruling (*"(a) as recommended, please"*).

| | within its own error (of 115) | worst \|v − ref\|/e | violations of ≤ max(1, 06g's ratio) |
|---|---|---|---|
| route 1 | 114 | 1.185 (v03; 06g's is 1.203) | 0 |
| route 2 | 114 | 1.202 (v03) | 0 |
| route 2 literal | 114 | 1.200 (v03) | 0 |
| route 3 | 112 | 3.935 (c01; 06g's is 0.0014) | **2**: c01 3.93; r56 (`g-1e-8-1-10794` #5) 1.49 against 0.0059 |

**In the 720** (not scored): routes 1, 2 and 2 literal leave all nine references as 06g has them, z12 at about 1,681
in each. Route 3 takes z00 from 0.018 to 1,681 × its error, the same as z12.

## The predictions, per route

| | rule | route 1 | route 2 | route 2 literal | route 3 |
|---|---|---|---|---|---|
| **L1** | of the 17 carrying leaves, none trusted or left | **holds** (0) | **holds** (0) | **holds** (0) | **fails** (1: c01) |
| **L2** | ≥ 80 % of 142 slow leaves trusted or left | **holds** (80.3 %) | **fails** (40.1 %) | **fails** (40.1 %) | **holds** (100 %) |
| **C1** | ≥ 90 % of V8's lost calls outside the 720 converge, at each rtol | **holds** (98.9 %, 98.6 %) | **holds** (100 %, 99.8 %) | **holds** (99.1 %, 98.9 %) | **holds** (100 %, 99.9 %) |
| **C2** | no call outside the 720 that 06g converges is lost | **fails** (7) | **fails** (5) | **fails** (53) | **fails** (1) |
| **H1** | every referenced call outside the 720 at ≤ max(1, 06g's ratio) | **holds** (115 of 115) | **holds** | **holds** | **fails** (2 of 115) |
| **S1** | no call outside the 720 moves past e_route + e_g | **holds** (0) | **fails** (10, worst 1.32) | **fails** (10, worst 1.32) | **fails** (46, worst 1.65) |
| **K1** | every trust firing at Δ1/φP < 5.8 | **fails** (89,171 of 229,338) | **fails** (82,565 of 167,281) | **fails** (88,186 of 191,623) | not scored |

For K1, the counts are of trust firings outside the 720. Within the 720 they are 15,491 of 38,240 (route 1),
13,781 of 29,318 (route 2) and 15,047 of 35,256 (route 2 literal).

A candidate remedy needs L1, C2, H1 and S1 to hold, and K1 for routes 1 and 2. None qualifies.

## What each route would do as a remedy (no code is proposed)

- **Route 1** is the closest.
  - It recovers 98.6–98.9 % of V8's lost calls, and no value moves past both errors.
  - No reference is worse than under 06g (114 of 115 within its own error; v03 at 1.185 against 06g's 1.203), and it keeps refining every one of 06h's carrying leaves.
  - Its cost falls to 0.78 of 06g's.
  - **Two failures.** It loses 7 calls at 1e-8 to the node-limited stop (abserr 0–8 % above 06g's). And its noise
    yardstick passes many leaves far above the noise level, K1's 5.8 included: when a Levin parent's halves are
    Clenshaw–Curtis solves, φL + φR can be hundreds of times φP (about 660× on the development unit's first firing, `out/cmd/16`, `state/dev/route1_1449/`); its census firings reach Δ1/φP = 1.1 × 10⁴. So it does not fire only on noise, as K1 asks.
- **Route 2, as built,** recovers more (99.8–100 %) at lower cost (0.60).
  - **Its declaration is short where it trusts on r_c alone.** It omits the node error by which Q differs from V_P,
    and S1 finds 10 calls that move past both errors. The analytic ones have one region.
  - Its trust firings run up to Δ1/φP ≈ 3.7 × 10⁵ (K1).
- **Route 2, literal,** recovers as much as route 1 at 0.62 of the cost.
  - **It withdraws trust**: 30,158 withdrawals in the census, many at r < 1/4. It loses 53 calls outside the 720,
    51 of them to an untrusted leaf at `depth_max`.
  - It shares route 2's S1 moves and K1 failure.
- **Route 3** recovers the most (100 %) at the least cost (0.30, near 06h's 0.32). Inside the 720 it converges all
  but one lost call.
  - **Its premise fails.** A leaf left at D ≤ 10⁻³ of the request can be off by more than the call's error:
    46 calls move past both errors, two references are missed (c01 at 3.9×, r56 at 1.5×), and inside the 720 z00
    goes to 1,681× its error.
  - At leaf level it leaves 06h's c01 leaf (L1).
  - In the LiouvilleGreen draws it is the one route that loses convergence (2 calls) and makes a case dishonest.
- **Side by side:**

  | | recovery (C1, at 1e-7 / 1e-8) | cost (unit wall) | honesty | safety (L1, S1) |
  |---|---|---|---|---|
  | route 3 | 100 / 99.9 % | 0.30 | fails H1 | fails |
  | route 2 | 100 / 99.8 % | 0.60 | holds H1 | fails S1 |
  | route 2 literal | 99.1 / 98.9 % | 0.62 | holds H1 | fails S1 |
  | route 1 | 98.9 / 98.6 % | 0.78 | holds H1 | holds |

  Route 1 alone keeps both L1 and S1. Every route, route 1 included, loses some convergence 06g had (C2).

Which route, if any, becomes code is for the user.
