# levin-phase-floor 07d, Part A′ (section 9.2): the second measurement — log of the measuring agent

**Subject:** prompt 07d section 9.2 items 0–6 and section 9.3's tables 1–7. **Commit:** none; the records are
committed by the orchestrator. **Result:** at section 9.3's stop, with every table written and no stop condition met.
- **Variants that qualify (section 9.3):** V93T8 and V93T10, and no other.
- **Every understatement of a returned leaf on the Levin side is a failure of v4's reference.** The second reference
  shows it. On the second reference, no returned leaf understates on the Levin side at any T.

Run on the Mac Studio (mps026298), Python 3.13.16, in the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3`,
detached at the amendment's SHA `c2c57a36`, whose code is 07c's (`67b5af74`; `levin_quadrature.py` `d141a80b…`).
Records are in `var/levin-phase-floor-07d/measurement-a2/` (below, `R/`). Started 2026-11-23T02:13:36Z
(`state/started_utc.txt`); stopped about 05:45Z. No Ray, no lock, and no store opened. Part A's records were read only.

## What shipped

All of it is under `R/`. Nothing was written in a tree or in Part A's records.
- **The gate:**
  - `out/cmd/01_gate.txt`, at the start: GATE HOLDS.
  - `out/cmd/99_gate_end.txt`, at the end: GATE HOLDS. It is identical to the start except the registry's run count,
    103 → 107, which are this Part's four runs.
- **v4's rule:** `state/refs_rule_leaf_v4.txt`, with the seeds in `state/seeds_a2.json` (BANDS2 707501, RT2 707502,
  BANDS2V 707503). Both were written before any draw.
- **The projection,** in `out/projection/`, and the orchestrator's go-ahead and second message, in substance, in
  `state/goahead_orchestrator_20261123.md`.
- **Nine variants** in `scripts/variants/`. Each has its diff in `out/variants/*.diff` and its registry entry in
  `out/variants/variants.json`; each is specified in `state/variants_a2.json` before it was built.
  - I6, the instrument;
  - P0, a plumbing check;
  - V93, V93tail, V93T8, V93T10, V93T12, K, G and KT.
- **Four registered runs** (`var/runs/levin-phase-floor-07d-a2-*`), all done, all exit 0:

  | run | what it ran | result | time |
  |---|---|---|---|
  | `a2-instrument-20261123T022323` | I6 on 4,220 calls | all equal the tree | 295 s |
  | `a2-leaves-20261123T023302` | 1,803 leaf specs | 0 errors | 1,499 s |
  | `a2-variants1-20261123T030234` | seven variants × 4,220 calls | all written | 1 h 47 m |
  | `a2-variants2-20261123T045048` | KT × 4,220 calls | all written | 9 m |

- **Tables** in `out/tables/`:
  - `t1_v4_v3`, `kappa93`, `t2_de_v4`, `t3_cc93`, `t4_dt`, `t4b_routes_by_kind_knots`, `t5_knots`, `t6_cost`;
  - `t7_scores` (section 9.3 table 7 and the qualification);
  - `strata_K_KT`;
  - `understate_ref2`.
- **Checks** in `out/repro/`:
  - `verify_repro_a2` (488/488);
  - `check_cc93_a2` (125/125);
  - `decl93_V93` and `decl93_V93tail`;
  - `second_ref_a2`;
  - `midjump_a2`;
  - `unknown_knots_a2`.
- Every script is in `scripts/` with its SHA-256 in each output's header. Every command's output is in `out/cmd/`.

## Predictions and verdicts

The prompt makes no numerical prediction for Part A′. Section 9.3's tables and verdicts follow.

### Table 1 — v4 against v3 (`out/tables/t1_v4_v3.md`)

- **Settling:** 3,673 leaves measured, 3,642 settled. No stratum has more than a tenth unsettled; the worst is 5 of
  100, `band_20_32v|analytic`. So the section 9.3 stop on settling is not met.
- **Equality checks, all equal:**
  - CC₄₇ against Part A's, 2,873;
  - σ̂, 2,873;
  - s_own, 2,873;
  - Part A's D, 10,542 of 10,542;
  - v3's verdicts re-derived, 0 mismatches.
- **Verdict changes, v3 → v4:**
  - table 2's tree: bounded→undecided 180, undecided→bounded 247, undecided→understates 3, understates→undecided 54;
  - table 3's: bounded→undecided 150, undecided→bounded 199, undecided→understates 19, understates→bounded 1,
    understates→undecided 172.
- **Where the references disagree:** v4 and v3 differ beyond both errors most on group leaves with 5+ knots,
  128 of 593.

### The unknown-knot leaves (the go-ahead's item 3)

- **What they are:** 706 leaves, all analytic, about 48 % of the 1,464 analytic leaves measured. They spread over every
  analytic stratum:
  - 12–37 a stratum in the gate, chase and window families;
  - 20–30 in each returned band, and 49–55 in each 20–32π stratum;
  - 6 in `chase|analytic|1e-8`.

  By call: Y1 125, Y2 217, Y3 173, Y4 191 (`out/repro/unknown_knots_a2.json`).
- **Why neither reader finds their knots:**
  - Every one is the sine component of a three-Bessel J call, `f=[lambda x: 0.0, lambda x: -Levin_f(x)]`
    (`ComputeTargets/QuadSourceIntegral.py`, `_three_bessel_Levin`).
  - Both readers walk `f[0]`'s closure: `production_edges` takes `f[0].__closure__[2]`, and Part A's analytic reader
    takes `f[0].__closure__`.
  - `f[0]` here is the zero lambda, which has no closure, so `production_edges` raises
    `TypeError("'NoneType' object is not subscriptable")` and the analytic reader raises
    `TypeError("'NoneType' object is not iterable")`.
  - The knots are in `f[1]`'s `Levin_f`, which neither reader looks at. This is a reader failure, not a knot-free
    integrand.
- **v4 on them:**
  - It splits nothing, taking the whole leaf, as the rule says where no knots are known.
  - Settled 694, unsettled 12 (1.7 %). The other leaves: settled 2,948, unsettled 19 (0.6 %).
  - Ten of table 1's verdict changes fall on them.
- **Does the class decide any verdict? No.**
  - None of table 2's or table 3's understatements is on an unknown-knot leaf.
  - Table 4's one analytic Levin understatement (`band_20_32|analytic|1e-8#21`) has known knots (0), and is a v4 failure
    (below).
  - Neither V93's nor V93tail's own declaration understates on any leaf.
  - On `band_20_32v`, the tree's own Levin declaration at making understates on 2–3 of the 98 unknown-knot leaves.
    These are rows of their own, not a qualification input.

### κ₉₃ (`out/tables/kappa93.md`)

- κ₉₃ = 4.
- The group ratio σ_own,93/σ̂ has p05 0.838 and median 1.01, over 1,089 leaves.
- The analytic ratio has p05 1.05 and median 1.81, over 909 leaves.

### Table 2 — Part A's CC₄₇ candidates on v4 (`out/tables/t2_de_v4.md`)

- **Sample:** 1,682 settled leaves drawn, 899 scored.
- **Understatements:**

  | candidate | understates |
  |---|---|
  | tree max(pair, floor) | 74 |
  | pairN4 | 6 |
  | pairN3 | 8 |
  | tail4N4, chopN4, pairtailN4 | 5 each |
  | nest3N4 | 0 (min D/te 1.38) |
  | pairDecl | 0 |

- **Qualify:** nest3N4 and pairDecl.
- **The tree's 74, by cause:** knots 55, and an error common to CC₂₄ and CC₄₇ 19.
- **On the second reference all 74 still understate.** The references disagree on 3 of them, with the verdict
  unchanged (`out/tables/understate_ref2.md`).

### Table 3 — the CC₉₃ candidates (`out/tables/t3_cc93.md`)

- **All five qualify,** on the returned strata and with `band_20_32v`: nest3N4, pair93, chop93, nest93, tail93.
- **Median D/te:**

  | candidate | cc strata, group | band, group |
  |---|---|---|
  | pair93 | 5.85 | 34.8 |
  | nest93 | 5.96 | |
  | tail93 | 6.55 | 45.9 |
  | nest3N4 | 7.23 | |
  | chop93 | 1,970 | |

- **CC₉₃ against CC₄₇:** CC₉₃ is closer than CC₄₇ mostly on knotted and band leaves.
- **dot against fsum (the go-ahead's item 2):** the largest |dot − fsum|/|CC₉₃| on the scored leaves is 4.89 × 10⁻¹³.
- **Table 3 used the 47-point floor; the variants use the 93-point floor (deviation 10).** Each variant's own declaration
  was checked on all 3,642 settled leaves (`out/repro/decl93_*.json`):
  - V93 (pair93): bounded 3,636, undecided 6, understates 0;
  - V93tail (tail93): bounded 3,640, undecided 2, understates 0;
  - both against table 3's formula: 3 undecided → bounded each;
  - their values equal the record's CC₉₃ bit for bit, 3,642 of 3,642;
  - D_variant/D_table3: median 1.00, range 0.57–1.52.

### Table 4 — D-T as the tree runs it (`out/tables/t4_dt.md`, returned leaves)

- **Under the CC₉₃ fallback and each CC₉₃ declaration:**

  | T | group, Levin side | group, CC side | analytic, Levin side | analytic, CC side |
  |---|---|---|---|---|
  | 4π, 6π | fails (2 understate) | bounded | bounded | bounded |
  | 8π–12π | bounded | bounded | bounded | bounded |
  | 16π, 20π | bounded | bounded | fails (1 leaf) | bounded |

- **Under the tree's own CC₄₇ declaration** the CC side fails for group calls at every T, with 34–52 understating.
- **Hence V93-T = {8, 10, 12}π** (state/variants_a2.json).
- **The causes of the Levin-side understatements (`out/tables/understate_ref2.md`).** Thirteen rows, five leaves.
  - Four of the five are returned leaves:
    - `band_20_32|group|1e-7#63`, `band_20_32|group|1e-8#52` and `band_6_8|group|1e-8#40`, which make 4π and 6π
      fail;
    - `band_20_32|analytic|1e-8#21`, which makes 16π and 20π fail.
  - **On those four, v4 and the second reference disagree by 1.17–1.28 eps_v4.** The second reference is
    48-point Gauss–Legendre, 256 and 512 panels a piece.
  - **Every value agrees with the second reference far better than with v4.** Errors against v4 and the second
    reference:

    | leaf | V_P error, v4 | V_P error, second ref | CC₁₈₅ error, v4 | CC₁₈₅ error, second ref |
    |---|---|---|---|---|
    | #63 | 7.3e−42 | 8.5e−44 | 1.2e−41 | 4.7e−42 |
    | #21 | 7.0e−27 | 3.2e−29 | 7.2e−27 | 1.5e−28 |

  - **On the second reference all four are undecided or bounded, at every T, under both fallbacks.**
  - **The fifth leaf is a genuine understatement:** `band_20_32v|group|1e-7#79`, visited, not returned.
    - The references agree to 0.03 eps.
    - It understates at 8π under the CC₄₇ fallback only: D/te 0.91.
    - Under the CC₉₃ fallback at 8π it is untrusted, with r = 7.3.
    - It is kept out of the pooled rows, as the go-ahead requires.
- **The new band's population (`out/leaves_a2/new_band_population.json`).** 3,938 Levin regions with
  V ∈ [20π, 32π) made on the `d:` calls, by fate in the tree:

  | fate | group | analytic | all |
  |---|---|---|---|
  | returned | 1,723 | 508 | 2,231 |
  | untrusted | 1,388 | 32 | 1,420 |
  | rate rule | 9 | 228 | 237 |
  | bisected unchecked (depth or forced) | 46 | 0 | 46 |
  | trusted but bisected | 4 | 0 | 4 |

  - The table's rows for `band_20_32v` score Levin on the tree's own solve's declaration, with the floor ("Levin own
    decl").
  - Where the tree took the check, they also score it as `_check_leaf` returned it.
- **"Made" (the go-ahead's item 1), as read:** every region `_make_leaf` built and solved by Levin with V in the band,
  whatever followed. Its depth and fate are recorded. The visited set is these less the ones returned. I believe this is
  the orchestrator's reading. The one difference from "solved by Levin and then bisected": the 46 "bisected unchecked"
  were bisected without a trust check, and they are included.

### Table 4b and table 5 — knots (`out/tables/t4b_routes_by_kind_knots.md`, `t5_knots.md`)

- **Table 4b, on group leaves with 5+ knots:** CC₉₃ is closer than V_P on 515 of 516, but Q is closer than CC₉₃ on 310
  against 37.
- **Table 5, the jumps at each kind of knot:**
  - The phase-spline, G-spline and table-node knots coincide. They carry a third-derivative jump in f₀ and f₁: median
    1.3 × 10³ and 2.2 × 10³ times the control, above 10× on 92–95 %.
  - The nearest-node midpoints carry a value jump: median 3.75 and 2.97 times the control, above 10× on 26–33 %.
  - Splitting at the knots lowers CC₄₇'s true error on about 76 % and V_P's on 70–73 %.
  - No declared break lies inside any leaf.
- **The midpoint jump's size (the orchestrator's second message; `out/repro/midjump_a2.json`).** On 16 group leaves,
  82 midpoints:
  - f₀'s direct step across the midpoint, d = |f(x+u) − f(x−u)|, has median 2.9 × 10⁻¹⁵ of |f| and max 1.6 × 10⁻¹⁴.
  - It is flat from 1 to 64 ulps: median 3.4 × 10⁻¹⁵ at 4 ulps, 3.2 × 10⁻¹⁵ at 64.
  - At the control points the step is 9.5 × 10⁻¹⁶ (1 ulp) and 6.3 × 10⁻¹⁶ (64 ulps).
  - Item 4's fit-based jump agrees: median 1.9 × 10⁻¹⁵ of |f|.
  - f₁'s step is no larger than its own slope over the step, at knots and controls alike (median 7.3 × 10⁻¹⁵ against
    7.2 × 10⁻¹⁵ at 1 ulp, both growing linearly). No f₁ jump is resolved by this measure; the fit gives a median of
    3.1 × 10⁻¹⁵.

### Table 6 — cost (`out/tables/t6_cost.md`)

Median per leaf:
- CC₄₇ solve 3.5–7 ms;
- CC₉₃ 3.8–12 ms;
- CC₁₈₅ 7.6–19 ms;
- one Levin solve 0.55–2.1 ms;
- a trust check 9–21 ms (7 solves, 4–6 of them CC sub-solves).

### Table 7 — honesty, convergence and qualification (`out/tables/t7_scores.md`)

- **Honesty, on 417 referenced and scored calls:**

  | code | honest | dishonest | unconverged |
  |---|---|---|---|
  | tree | 410 | 4 | 3 |
  | V93, V93tail, V93T8, V93T10 | 412 | 4 | 1 |
  | V93T12 | 411 | 5 | 1 |
  | K, KT | 413 | 3 | 1 |
  | G | 416 | 0 | 1 |

  - The tree's 4 dishonest calls are R2-28's, at ratio 1.14, and stay dishonest under the V93 family (1.10–1.13).
  - K and KT make them honest (0.053). G makes them honest (5 × 10⁻⁸).
- **New converged-dishonest calls:**
  - V93T12: `R:a:W2-0-12977-5`, ratio 2.79 (tree 0.0485).
  - K: `R:c-1e-7-1-1759-5` and `R:c-1e-8-1-1759-5` (1.19 each), and `R:m-1e-8-3-35006-g4` (1.56).
  - KT: the same three (1.19, 1.19, 1.24).
- **Convergence, reach and cost, over all 4,220 calls (against the tree):**

  | variant | touched | conv lost / gained | regions | solves | wall | route changes (calls) | knot breaks or gates (calls) |
  |---|---|---|---|---|---|---|---|
  | V93 | 3,397 | 0 / 7 | 0.912 | 0.908 | 2.03 | 0 | – |
  | V93tail | 3,397 | 1 / 7 | 0.992 | 0.966 | 1.52 | 0 | – |
  | V93T8 | 3,513 | 0 / 7 | 0.921 | 0.916 | 2.23 | 22,490 (2,388) | – |
  | V93T10 | 3,625 | 0 / 7 | 0.907 | 0.901 | 2.32 | 37,574 (3,158) | – |
  | V93T12 | 3,687 | 0 / 7 | 0.888 | 0.884 | 2.27 | 48,669 (3,328) | – |
  | K | 3,406 | 0 / 7 | 0.789 | 0.789 | 1.57 | 0 | breaks 220,326 (721) |
  | G | 3,406 | 0 / 7 | 4.519 | 4.550 | 10.35 | 0 | gates 305,221 (686) |
  | KT | 3,522 | 0 / 7 | 0.791 | 0.788 | 1.68 | 17,010 (2,379) | breaks 218,166 (721) |

  - The tree has 9 calls unconverged. Every variant has 2, except V93tail and KT, which have 3.
- **Qualification (section 9.3: (a) no new converged-dishonest call, (b) the declaration qualifies, (c) the threshold
  bounded on both sides):**

  | variant | qualifies | why not |
  |---|---|---|
  | V93T8 | yes | |
  | V93T10 | yes | |
  | V93, V93tail | no | (c) at 6π |
  | V93T12 | no | (a), one call |
  | K | no | (a), three calls, and (c) at 6π |
  | KT | no | (a), the same three calls |
  | G | no | (c) at 6π only |

  - Every (c) failure at 6π rests on the v4 failures of table 4. On the second reference the 6π Levin side has no
    understating leaf, only undecided ones.
- **K's and KT's changed calls by (Rt) stratum (the orchestrator's second message; `out/tables/strata_K_KT.md`).**
  - **K against V93:** every changed `d:` call is "layout changed, routes unchanged", 315 calls in 8 group strata.
    Each stratum has 2–4 referenced calls from Part A's (Rt).
  - **KT against V93T8:**
    - Its 270 Levin→CC calls fall in 8 strata, each with 2–4 referenced (Rt) calls.
    - 41 "layout changed, routes unchanged" calls fall in 8 strata (9, 9, 1, 2, 2, 2, 8, 8), **none referenced**.
  - **The regression set:** 401 changed calls, 253 of them referenced.
  - **No qualification verdict depends on KT's empty strata.** KT does not qualify on (a), on referenced calls, and
    calls referenced in those strata cannot remove that. So no projection was sent and no reference was read.
  - Direction is judged against the tree's 6π gate (`num_a2_route_changed`); "changed" against the variant's base.

## Deviations from the prompt

1. **The gate's first record misread the registry.**
   - `gate_a2.py` v1 matched "running" inside a killed run's description (`levin-phase-floor-07c-refs-20261118T112553`,
     status `killed`), and wrote GATE FAILS.
   - v2 reads the status column; its record (`out/cmd/01_gate.txt`) holds.
   - v1's output is kept as `out/cmd/01_gate_v1_status_parse_defect.txt`.

   **UNINTENDED DRIFT.**
2. **Part A's `scripts/harness.py` is copied byte for byte** into `scripts/` (`out/cmd/02_harness_copy_sha256.txt`). The
   group captures unpickle objects of module `harness`. **STRUCTURALLY REQUIRED.**
3. **v4 splits an analytic leaf at the knots Part A's v3 analytic reader finds:** near-region interpolant breakpoints and
   x_star crossings. `production_edges` raises on analytic blocks. The prompt's "the whole leaf where none are known" is
   kept where neither reader reads one; that is the 706 sine-component leaves above. **IMPLEMENTATION CHOICE.**
4. **v4's settling rule reads "two successive doublings agree" as both of the last two doublings agreeing** within 3×
   their combined noise (state/refs_rule_leaf_v4.txt item 4), so one accidental agreement does not settle a piece.
   **IMPLEMENTATION CHOICE.**
5. **v4's noise term takes σ̂ from Part A's v3 least squares,** re-run by the same code on the same pieces and checked
   equal to Part A's σ̂ on 2,873 of 2,873 leaves. The prompt does not say where the noise comes from.
   **IMPLEMENTATION CHOICE.**
6. **The new band's first draw rested on a misreading.**
   - I told the orchestrator that no returned leaf has V ≥ 20π. In fact Part A's `population.json` leaves out
     out-of-band leaves by construction.
   - The band was redrawn from the returned leaves (section 2.4's construction) as `band_20_32`.
   - The approved "visited, not returned" set was kept apart as `band_20_32v`.
   - The first, mixed draw (`state/superseded/`, `out/leaves_a2/superseded/`) was superseded before any solve.

   **UNINTENDED DRIFT.**
7. **The first reproduction differed from I6 on one field, 80 of 488 leaves.**
   - The cause: I6 records the rate rule only where it refuses, and the reproduction recorded the rule it applied.
   - The fix: the reproduction now records `rule` only where it refuses, and `rule_applied` beside it. After that, 488 of
     488 are equal.
   - v1 is kept (`out/repro/verify_repro_a2_v1_rule_field.json`).

   **UNINTENDED DRIFT** (a comparison convention; no value differed).
8. **CC₉₃ is computed by `np.dot(w₉₃, g₉₃)`** in item 1 and in the V93 family, and the largest |dot − fsum|/|CC₉₃| is
   recorded. **IMPLEMENTATION CHOICE**, approved (go-ahead, item 2).
9. **Tables 1–6 score every leaf of Part A's that this Part measured,** not only Part A's scored prefix.
   **IMPLEMENTATION CHOICE.**
10. **Table 3 scored each CC₉₃ candidate with the 47-point floor,** max(E, floor₄₇) + κ s₉₃, where the V93 family
    declares with its floor and node term on the 93-point grid (go-ahead, item 2).
    - `decl93_a2.py` then scored V93's and V93tail's own declarations on all 3,642 settled leaves: none understates.
    - Table 3's verdicts move only undecided → bounded, 3 leaves each.
    - Its other candidates (nest3N4, nest93, chop93) were not built and are not re-checked.

    **UNINTENDED DRIFT.**
11. **Item 4 measures the jumps on f₀, f₁ and θ_raw as the integrand receives them,** not per table ingredient.
    **IMPLEMENTATION CHOICE.**
12. **V93tail (tail93) is my choice:** the second built V93 declaration, beside pair93. It gives the CC₉₃ family a
    declaration that needs no second rule. **IMPLEMENTATION CHOICE.**
13. **A seventh process, briefly.**
    - While batch 1 held six workers, two reader commands ran, about a second each:
      - one read Part A's census index's first row;
      - one gave a first look at item 4's midpoint-jump records.
    - Neither wrote anything.
    - Earlier, while the leaf run held six workers, only `ps` and `ls` ran.

    **UNINTENDED DRIFT**, as Part A's deviation 11.
14. **The patch to `make_variants_a2.py` was truncated** by a nested triple quote, and repaired before any build. No
    variant was affected; each diff is saved. **UNINTENDED DRIFT.**
15. **G's basis text was first written with unchecked figures,** and recomputed and corrected before G was built:
    - n = 516;
    - CC₉₃ closer than V_P on 515;
    - Q closer than CC₉₃ on 310 against 37.

    **UNINTENDED DRIFT.**
16. **(K), (KT) and (G) take a call's knots from `leaf_a2.typed_knots`** (production_edges' walk), supplied by the
    harness. On analytic calls the walk raises and no knots are supplied, so those calls bisect and gate as the tree
    does. **STRUCTURALLY REQUIRED.**
17. **`midjump_a2.py`'s first run failed after measuring,** while writing its header: it passed a directory as an input.
    - The header now lists the leaf files it read, and it was re-run; the measurement is unchanged.
    - The failed run's output is kept as `out/cmd/midjump_a2_v1_header_defect.txt`.
    - Likewise, `gate_a2.py` was first called at the end without its output argument. It wrote nothing, and its stdout
      file was removed before the correct call.

    **UNINTENDED DRIFT.**
18. **Each new check's selection was fixed in its docstring before it ran:** second reference, midpoint jumps, own
    declarations and unknown-knot reader. Two are measurements the prompt does not name:
    - the re-scoring on the second reference (`rescore_ref2_a2.py`);
    - the unknown-knot diagnosis (`unknown_knots_a2.py`).

    They answer the go-ahead's item 3 and the "causes of any understatement". **IMPLEMENTATION CHOICE.**

## Verification performed

See `verification.md`. In short:
- the gate held at the start and the end;
- I6 equals the tree on 4,220 of 4,220 calls;
- P0 equals the tree on 211 of 211;
- the `_check_leaf` reproduction equals I6 on 488 of 488 checked leaves (338 group, 150 analytic);
- V93's CC₉₃ equals item 1's on 125 of 125, and on 3,642 of 3,642 in the declaration check;
- every registered run exited 0 and wrote every output;
- the worktree's `git --no-optional-locks status --short --ignored` is empty, and the main checkout's status is empty.

## Observations not acted on

- **The second reference disagrees with v4 on 7 of 84 leaves checked** (8 %), by 1.11–2.29 eps_v4. On all seven, v4's
  stated error eps_v4 is 0.44–0.90 of its own error measured against the second reference. Proposed in `issues.md`.
- **The knot readers see only `f[0]`.** About half the analytic leaves (the sine components) therefore read as knotless
  (`issues.md`).
- **On the visited band, the tree's own Levin declaration at making understates on many regions:**
  - group, 5+ knots: 36–44 of about 100 at every T;
  - analytic: 2–5.

  These regions were untrusted and bisected, so the declaration is not returned. Recorded only.
- **On group leaves with 5+ knots, Q beats CC₉₃** (310 against 37) though CC₉₃ beats V_P (515 of 516). Recorded only.
- **G is the only variant that makes every referenced call honest,** at 10.35× the tree's wall and 4.5× its regions.
  Recorded only.

## State handed on

- **Nothing is running.** The registry lists 107 runs, all finished; no run is running or stale.
- No lock was taken. No Ray was started.
- `KEPT_ON_MAC_STUDIO.txt`, written last, lists every pickle and every file over 2 MB in `R/` and in this Part's run
  directories. Everything else in `R/` is for the orchestrator to copy back.

## Addendum (2026-11-24): settling the near-edge leaves (A) and dissecting V93T12's new dishonest call (B)

**Subject:** the orchestrator's addendum, recorded in substance in `addendum/state/orchestrator_addendum_20261124.md`.
**Records:** `addendum/` (its own `scripts/`, `state/`, `out/`), committed by the orchestrator. **Result:** A and B done,
no stop condition met.
- **A:** on the references below no near-edge check of the CC₉₃ fallback understates on a returned leaf. One returned leaf
  understates under the CC₄₇ fallback, and 77 checks stay undecided.
- **B:** V93T12's understatement is carried by one region trusted through a change in its trust check's sub-solves, not
  by a CC₉₃ leaf.

Nothing else in `R/` was written. I recommend nothing.

### What ran

- **The selection** (`addendum/state/selection_add.json`, by `scripts/select_add.py`) was fixed before any reading:
  - every trusted Levin-side check at T ∈ {6, 8, 10, 12}π, under both fallbacks, undecided on v4;
  - every undecided CC-side pair93 and tail93 verdict of table 4 at those T.

  That is 461 checks on 154 leaves (returned group 100, returned analytic 38, visited group 16; 320 pieces).

  | | Levin, CC₄₇ fallback | Levin, CC₉₃ fallback | CC side, pair93 and tail93 |
  |---|---|---|---|
  | returned | 235 | 177 | 24 (3 a T each) |
  | visited | 16 | 9 | 0 (no visited leaf has V < 12π) |

- **The rules** (`addendum/state/rule_ref48_add.md`) were fixed before any reference was read.
- **A.2, ref48:**
  - The rule: 48-point Gauss–Legendre at 512 and 1,024 panels a piece, and 2,048 where |I₁₀₂₄ − I₅₁₂| exceeds the
    noise term 2σ̂√Σw².
  - The run: registered, `levin-phase-floor-07d-add-ref48-20261124T103141`, six workers, 7 min, all exit 0, 154 of
    154 leaves. The projection, 4–10 min, is in `addendum/out/projection_ref48.json`.
  - **The noise term dominates the stated error on all 154 leaves.** One leaf needed 2,048 panels; none still moves at
    2,048.
- **A.3, the smooth-phase reference:**
  - The tool: `tools/filon_reference.reference` on the call with the leaf's span: route B for group leaves, edges at the
    leaf's knots; route A for analytic leaves.
  - Its stated error: the four readings' scatter + 2 × FCC resolution 2's fit-noise 1σ.
  - Where it ran: on the 27 leaves (55 checks) where the noise term decides the verdict, that is, undecided on eps48 and
    decided on the change alone.
  - Its cost was projected first: 3 leaves in 8.5 s, so about 80 s for all, far under two hours. The probe's three
    values equal the run's bit for bit.
- **A supplementary pass, not in the fixed rule** (addendum deviation 3): the same smooth-phase reference on the 17 other
  leaves with a check still undecided.

### A — tables (`addendum/out/tables_add.md`, `tables_add_supp.md`, `edge_vp_add.json`)

**Final verdicts** (the last reference taken on each leaf: ref48, or the smooth-phase reference where taken):

| set | fallback | T | bounded | undecided | understates |
|---|---|---|---|---|---|
| returned | CC₄₇ | 6π | 64 | 29 | 1 |
| returned | CC₄₇ | 8π | 46 | 13 | 1 |
| returned | CC₄₇ | 10π | 44 | 12 | 0 |
| returned | CC₄₇ | 12π | 23 | 2 | 0 |
| returned | CC₉₃ | 6π | 53 | 14 | 0 |
| returned | CC₉₃ | 8π | 41 | 2 | 0 |
| returned | CC₉₃ | 10π | 42 | 3 | 0 |
| returned | CC₉₃ | 12π | 21 | 1 | 0 |
| returned | CC side, pair93 and tail93 | each T | 6 | 0 | 0 |
| visited | CC₄₇ | 8–12π | 15 | 1 | 0 |
| visited | CC₉₃ | 6–12π | 8 | 0 | 1 |

- **The CC side:** all 24 are bounded on ref48, under table 4's formula. Under the variants' own declarations
  (`decl93_*`) they are bounded too.
- **The returned understatement, `band_8_10|group|1e-8#23`** (V = 9.9π, 1 knot):
  - Under the CC₄₇ fallback at 6π and 8π: D = 1.65e-43 against te = 3.25e-43 on the smooth-phase reference (D/te 0.51),
    eps_s 1.15e-43. ref48 agrees with the smooth-phase reference to 2.3e-44.
  - Under the CC₉₃ fallback the same leaf's check is untrusted (r = 1.37), so it is bisected.
  - v4 could not see it (eps_v4 1.8e-42).
- **The visited understatement, `band_20_32v|group|1e-7#43`:** at 12π under the CC₉₃ fallback, on ref48 (D/te 0.76).
- **What stays undecided: 77 checks,** every one with the smooth-phase reference taken (`tables_add_supp.md`). Two kinds.
  - **(i) 49 checks with D/te in [0.95, 1.05], at the edge by construction** (`edge_vp_add.json`):
    - V_P is closer to the reference than Q on all 49, with |V_P error|/|Q error| of median 0.028.
    - Q's error equals the halves' error (median ratio 1.07).
    - D ≈ δ₁ = |V_P − halves| (median D/δ₁ 1.10).
    - So the check's declaration measures the very error its returned value shares with the halves. D sits on te within
      the (1 − r) factor, and |D − te| is below the reference's stated error.
    - This class includes every edge leaf the orchestrator named:

      | leaf | T | D | te_s | eps_s |
      |---|---|---|---|---|
      | `band_10_12|group|1e-7#7`, `…1e-8#24` | 6, 8, 10π | 6.73e-41 | 6.77e-41 | 6.0e-42 |
      | `band_16_20|group|1e-8#32` | 10, 12π | 4.03e-42 / 4.94e-42 | 3.86e-42 / 4.79e-42 | 3.1e-43 |
      | `band_6_8|group|1e-7#40`, `…1e-8#59` | 6π | 3.90e-45 | 3.89e-45 | 8.8e-47 |

    - `band_6_8|group|1e-8#8` at 6π became bounded on the smooth-phase reference: D 1.21e-42, te 1.18e-42, eps 2.8e-44.
  - **(ii) 28 checks where the reference's own stated error exceeds |D − te|:**
    - Most are `band_20_32` group leaves at 6π, where the smooth-phase reference's four readings scatter by more than te.
      For example, `band_20_32|group|1e-7#94`: D 3.85e-48, te_s 7.8e-47, eps_s 1.3e-46, D/te 0.05.
    - On these, D/te ranges from 0.05 to 4.0. No reference taken here can decide them.
- **"At the noise floor" (A.4) is not informative as I defined it.** The rule I fixed beforehand labels a check when
  2 s₄₇ ≥ |D − te|. Every undecided check meets it, with 2 s₄₇/|D − te| of at least 38, so the label separates nothing.
  The table reports both numbers, and I force no verdict (addendum deviation 4).
- **v4 against ref48:** the two disagree beyond both errors on 8 leaves (34 checks).
- **The orchestrator's 31-leaf check:** its checks (trusted, CC₉₃ fallback, undecided on v4, T ∈ {6, 8, 10, 12}π)
  are a subset of this selection. On my references none of them on a returned leaf understates; one visited check of this selection does
  (`band_20_32v|group|1e-7#43`, 12π).

### B — `R:a:W2-0-12977-5` (`addendum/out/dissect_b.md`, `check_b_region.json`, `quarter_b_tree.json`, `quarter_b_variant.json`)

- **The call:** an analytic Gqr call, group G−q+r, k = 3.14×10⁵/Mpc. Part A's reference is −1.6152007252×10⁻²⁴.

  | code | ratio |v − ref|/abserr | regions returned | differ from the others |
  |---|---|---|---|
  | tree | 0.0485 | 36 | 21 |
  | V93T8 | 0.0538 | 32 | 17 |
  | V93T10 | 0.0498 | 30 | 15 |
  | V93T12 | 2.79 | 21 | 6 |

  - 15 regions are common to all four codes.
  - Every differing region was measured on ref48 by a registered run, `levin-phase-floor-07d-add-dissect-b-20261124T105107`:
    six workers, 29 regions, 14 min.
- **What carries it.** Under V93T12 one Levin region carries the whole understatement:
  - span `[0x1.3744465540140p+4, 0x1.45b4e8df10b9ep+4]`, depth 4, V = 59.4π;
  - returned trusted, declared total_err 4.46e-32 against a true error of −3.62e-31 (decl/true 0.12);
  - the call's error is −3.58e-31.

  Under the tree, V93T8 and V93T10 the same span is not trusted, and is bisected to depths 7–9.
- **Why it is trusted at 12π only** (`check_b_region.json`, the reproduction that is bit for bit with I6):
  - **The check:** V_P, the halves and the four quarters.
    - At 6, 8 and 10π all six sub-solves are Levin, r = 4.98, untrusted.
    - At 12π the sixth sub-solve, the right half's right quarter (V = 10.26π), takes the CC₉₃ route, and r falls to
      0.048 (δ₁ 4.3e-32, δ₂ 2.1e-33), so the region is trusted.
  - **The quarters** (`quarter_b_*.json`):

    | quarter | V | Levin error | CC₉₃ error | CC₉₃ declared (V93T12) |
    |---|---|---|---|---|
    | sixth, the right half's right quarter | 10.26π | +2.13e-31 (its own solve's max(abserr, phase_err) is 3.5e-37) | −9.5e-35 | 4.1e-33, bounded |
    | fifth, its sibling | 12.85π, so still Levin | −1.44e-31 | −6.2e-35 | |

  - **On the region:**
    - V_P, the halves' sum and Q all carry nearly the same error: −4.03e-31, −3.60e-31, and −1.49e-31 at T ≤ 10π
      (−3.62e-31 at 12π).
    - At T ≤ 10π the sixth quarter's Levin error, +2.13e-31, made Q disagree with the halves, and the check refused.
    - At 12π the accurate CC₉₃ value removes that disagreement. Q then agrees with the halves, both carrying the error
      common to the region's Levin solves, and the check trusts it.
- **So the mechanism is a change in a trust check's sub-solves.**
  - The CC₉₃ sub-solve is the accurate one, and its pair93 declaration bounds its error (43×).
  - What fails is the check: it sees only differences between Levin estimates that share an error. An inaccurate quarter
    had been hiding that agreement.
  - No CC₉₃ leaf with V in [10, 12)π is returned by V93T12 on this call.
- **Is the mechanism present below budget under V93T10 or V93T8?** Not on this call.
  - Under both, the same region's check refuses (r = 4.98).
  - Every differing region they return is bounded, with decl/true of 3.2 to 2.1×10⁴. That includes the CC leaves with V
    in [8, 10)π under V93T10: 436 and 2.07×10⁴.
  - The exceptions are three Levin regions with V = 662π, 701π and 1,530π, returned under all four codes, the tree
    included. Their decl/true is 0.0096–0.51, but their errors (≤ 2.7e-36) are about 10⁵ below the call's budget. That
    is a different mechanism: high-V Levin leaves understating on their own.
  - I measured only this call. Whether the sub-solve mechanism occurs below budget on other calls under V93T8 or V93T10
    is not measured here.

### Addendum deviations

1. **`dissect_b.py` first ran as an unregistered background command** and passed ten minutes.
   - I stopped it at about 12 min, with no output written.
   - It was restructured into `measure I N` and `combine` modes, the same rule, and rerun as the registered run above.
   - Its first stdout is kept as `addendum/out/cmd/09_dissect_b_v1_killed_over_10min.txt`.

   **UNINTENDED DRIFT.**
2. **`check_b_region.py`'s first version** would have re-measured the region's reference, about 14 min unregistered. I
   stopped it within a minute, before any output, and rewrote it to read the region's ref48 from the registered run. Its
   stdout is kept as `addendum/out/cmd/11_check_b_region_v1_stopped.txt`. **UNINTENDED DRIFT.**
3. **The supplementary smooth-phase pass** (`addendum/state/supplementary_pass_add.json`) was written after the A.4
   tables were read.
   - The reason: on those 17 leaves the last doubling's change itself exceeds |D − te₄₈|, so the fixed rule's "noise
     decides" test excludes them although ref48 cannot resolve them.
   - Its results are reported apart, in `tables_add_supp.md`, and change no fixed-pass verdict: all its checks stay
     undecided.
   - `smooth_add.py` gained a `--list` option for it; each output records the script's SHA-256.

   **IMPLEMENTATION CHOICE.**
4. **The "noise floor" proxy, 2 s₄₇,** was fixed before reading and turned out uninformative (above). It is reported, not
   relied on. **IMPLEMENTATION CHOICE.**
5. **Two quarters that are sub-solves, not returned regions, were measured for B.2** (`quarter_b.py`, registered run
   `levin-phase-floor-07d-add-quarter-b-20261124T110740`, two processes, 5 min). They go beyond "the differing leaves",
   to say which value carries the error. **IMPLEMENTATION CHOICE.**
6. **The driver and launcher are copied byte for byte into `addendum/scripts/`** (`out/cmd/02_driver_copy_sha256.txt`),
   so that the runs' records land under `addendum/`. **STRUCTURALLY REQUIRED.**

### Addendum verification

- **Four registered runs,** all done and exit 0. Nothing is running: the registry lists 110 runs, all finished.
  - add-ref48: 6 of 6 commands;
  - add-dissect-b: 7 of 7;
  - add-quarter-b: 2 of 2.
- **Processes:** at most six at once. No Python ran while add-ref48 or add-dissect-b held six workers; the scoring and
  supplementary readers ran while add-quarter-b held two.
- **Checks:**
  - the smooth probe equals the run on 3 of 3 leaves;
  - the selection's D equals table 4's D on every Levin check, which `select_add.py` asserts;
  - the trust-check reproduction is the one verified bit for bit (488 of 488).
- **Files:** no file in `addendum/` is over 2 MB and none is a pickle. `addendum/KEPT_ON_MAC_STUDIO.txt` lists the run
  directories' large files, if any.
- The worktree's `git --no-optional-locks status --short --ignored` is empty.
