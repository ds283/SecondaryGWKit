# levin-phase-floor 07d Part A′: the scoring rules of section 9.3's tables

Fixed 2026-11-23, while the leaf run (`levin-phase-floor-07d-a2-leaves-*`) ran, before any of its outputs was read.
Verdicts are v4's (`state/refs_rule_leaf_v4.txt` item 6). An unsettled leaf is reported and never scored.

## Table 1, v4 against v3

- **Every candidate verdict Part A's tables 2 and 3 recorded is re-taken on v4.** That is every candidate on CC₄₇ in a
  Clenshaw–Curtis stratum, and every route in a band.
  - The declaration D is Part A's, recomputed from its leaf record by Part A's formulas (`d_e_table_07d.py`), and checked
    equal to its recorded D.
  - The value is the same CC₄₇ (or V_P), checked equal.
- **A change** is any leaf whose verdict differs between v3 and v4, counted by table, stratum and direction.
- Also tabulated:
  - |ref_v4 − ref_v3| against eps_v3 + eps_v4, by knot class (none, 1–4, 5+, unknown);
  - for the leaves whose knots are unknown (the orchestrator's item 3): their kind and stratum, why neither reader finds
    knots, and v4's settled and unsettled counts.

## Table 2, D-E on v4

- **The leaves:** Part A's candidates on CC₄₇, on every measured leaf of Part A's Clenshaw–Curtis strata, not only the
  prefix to the 40th scored leaf (Part A's form).
- **Scoring:** a leaf is scored if the tree's verdict on v4 is not undecided. Counts and ratios are as Part A's.
- **Qualification:** a candidate qualifies if it understates on no scored leaf of any stratum it applies to.
- **The cause** of each tree understatement, by Part A's rule in order:
  1. knots, if the leaf split at all its knots (item 4's "all" split, CC₄₇) leaves less than half the true error, decided
     beyond eps;
  2. common, if the signed errors of CC₂₄ and CC₄₇ have one sign and |e₄₇| ≥ ½|e₂₄|;
  3. noise, if the true error is at most 4 s_CC (s_CC = σ̂ √Σw² of the 47 rule);
  4. else unexplained.

  Part A's order was knots, then noise; common is reported beside, as Part A did.

## Table 3, CC₉₃

**κ for s_own,93 (fixed before reading):**
- On chopped group leaves (standardChop chops the 93 coefficients), take q = p05 of σ_own,93 / σ̂.
- κ₉₃ is the smallest of {3, 4, 5, 6, 8} with κ₉₃ · q ≥ 2.8 (Part A's basis: P(|Z| > 2.8) = 0.5%).
- Analytic leaves' p05 is reported beside it.

**The candidates for CC₉₃'s error.** Each is max(E, floor₄₇) + κ s:

| name | E | s | extra evaluations beyond 93 |
|---|---|---|---|
| `nest3N4` | \|CC₉₃ − CC₄₇\| | 4 s_own,47 | 0 (Part A's, as it stands) |
| `pair93` | \|CC₁₈₅ − CC₉₃\| | κ₉₃ s_own,93 | 92 |
| `chop93` | chop tail of the 93 coefficients (Part A's chopN4 form) | κ₉₃ s_own,93 | 0 |
| `nest93` | \|CC₉₃ − CC₄₇\| | κ₉₃ s_own,93 | 0 |
| `tail93` | tail_sum4 of the 93 coefficients | κ₉₃ s_own,93 | 0 |

- **The floor is CC₄₇'s** (the tree's solve). The V93 variant forms the 93 rule's own; both are about 10⁻¹⁶ relative,
  far below κ s_own.
- **Leaves:** every settled leaf of Part A's D-E and D-T samples, and of item 3's two band families (reported apart).
- **Decisions:** a candidate's verdict counts where it is not undecided.
- **Qualification:** a candidate qualifies if it understates on no leaf of any stratum.
- **Also reported:**
  - CC₄₇'s and CC₉₃'s true errors, and which is closer where v4 can tell (the difference of the two absolute errors
    above 2 eps_ref);
  - the largest |dot − fsum| / |CC₉₃| (the orchestrator's item 2).

## Table 4, D-T as the tree runs it

The grid is T ∈ {4, 6, 8, 10, 12, 16, 20} π.

**The leaves:**
- the band strata (Part A's five bands and band_20_32);
- the gate strata gate_q1..q4 (Part A's section 2.4 quarters);
- band_20_32v, "visited, not returned", in rows of its own and never pooled.

**The Levin side (V ≥ T),** from item 1's check at T under the CC₄₇ fallback (tree) and the CC₉₃ fallback:
- A trusted leaf is scored on Q with D = total_err_trusted = max(max(residual, δ₁)/(1 − r), phase_err) + truncation.
- An untrusted leaf is counted, with r and δ₁; the depth-cap form is reported beside it.
- A leaf whose own solve goes to Clenshaw–Curtis or is guarded is counted apart.
- On band_20_32v, the tree's own declaration at making (own_err) is also scored, on V_P.

**The Clenshaw–Curtis side (V < T):**
- CC₄₇ with the tree's declaration max(pair, floor), and with nest3N4;
- CC₉₃ with each table 3 candidate.

**Per T, by call kind, and by knot class for item 3:** each side is
- *bounded* if it has scored leaves and none understates;
- *fails* if any understates;
- *vacuous* if it has no scored leaf.

**Also, per row:**
- the true error against the leaf's share of the request;
- which route is more accurate, where v4 can tell.

## Table 5, knots by kind (group leaves)

- **Counts:** each kind's count inside each leaf, by band.
- **Continuity:** per kind and ingredient (f₀, f₁, θ_raw), the jump in the value and in the first three derivatives,
  against the control's (same h).
  - The jump is "above the noise" if it exceeds 10 times the control's.
  - The control is the same fit at the leaf's largest knot-free gap.
- **The error it carries:** for CC₄₇, CC₉₃ and V_P, the true error of the leaf split at that kind's knots alone, against
  the unsplit true error.
  - The split "lowers" it if the error falls below half the unsplit one, beyond eps_ref.
  - It "raises" it if it rises above twice the unsplit one.
- A kind whose split lowers some rule's error on at least one leaf in ten, with jumps above the noise, is a kind that
  carries error.

## Table 6, cost

Per band, medians and p90 of:
- the walls of CC₄₇ (the solve), CC₉₃ and CC₁₈₅, and of one Levin solve;
- the Levin leaf as the tree checks it at each T: wall, and solves.
