# levin-phase-floor 07 census agent: the cross-rtol check and the disagreement list — the rule, fixed before any result

Written 2026-11-09 on mps026298, before any number of this check was read. Not to be rewritten: a correction is a new
file and a deviation. The user, relayed verbatim: "Yes, please do the cross-check on `rtol`s and the list of calls
where `06m` and `07` differ by more than the stated error. We can decide on a sample of references to sort of what these
difference really show, later."

## Inputs (read only, each checked by SHA-256 before and after)
- 07's census `levin-phase-floor-07-census-20261108T171137`, checkpoint `03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10`.
- 06m's census `levin-phase-floor-06m-census-20261106T102153`, checkpoint `ea5bc3ab6b618d562b9dee0a7ed4bdba5c1999e8348b2ad33bdadb6135e63c2e`.
- `state/rows.json` (`aa523d28…`) for band and k; `state/v4_selection.json` (`15a4391b…`) and 06g's
  `scores_stage3/scores.json` (`rows`) for V4's set.

## Notation
A call is a Levin call of a unit's record (`calls`: group; `other_calls`: analytic), identified by its row, kind, call
index, `notify_label` and `x_span` (both ends by hex). For one call: v_g(ρ), e_g(ρ) are 07's value and abserr at rtol ρ;
v_m(ρ), e_m(ρ) are 06m's. Every comparison below is made exactly, on the doubles as fractions (Python `Fraction` of
the float); ratios are reported as doubles.

## A. The population and the pairing
- **Population:** every call that 07's record at either rtol marks as locating a point or guarding a leaf
  (`guard_07.locates` or `guard_07.guards` true).
- **Pairing:** the same call at the other rtol is the call of the same row and kind with the same index, label and span
  (hex); a pair that does not match by all four is reported, not used.
- **Part A ratios, per pair:**
  - r_g = |v_g(1e-7) - v_g(1e-8)| / e_g(1e-7)
  - r_m = |v_m(1e-7) - v_m(1e-8)| / e_m(1e-7)
  - d8_g = |v_g(1e-8) - v_m(1e-8)| / e_g(1e-8),  d8_m = |v_g(1e-8) - v_m(1e-8)| / e_m(1e-8)
  A ratio with a zero or non-finite denominator is infinite when its numerator is non-zero, and 0 when it is zero.

## B. The disagreement list
- **The rule:** a call of the population, at rtol ρ, is listed if |v_g(ρ) - v_m(ρ)| > max(e_g(ρ), e_m(ρ)).
- **"Agrees", in terms of part A (fixed here):**
  - 07 agrees with itself across rtols: r_g <= 1.
  - 06m agrees with itself across rtols: r_m <= 1.
  - The two codes' 1e-8 values agree: |v_g(1e-8) - v_m(1e-8)| <= max(e_g(1e-8), e_m(1e-8)), i.e. d8_g <= 1 or d8_m <= 1.
- **The classes of a listed call** (by its pair; the same for both rtols of a pair):
  - **(a) settled for 07:** r_g <= 1, r_m > 1, and the 1e-8 values agree.
  - **(b) settled for 06m:** r_m <= 1, r_g > 1, and the 1e-8 values agree.
  - **(c) undecided:** r_g <= 1 and r_m <= 1 (each code agrees with itself; the two disagree at the listed rtol).
  - **(d) neither:** r_g > 1 and r_m > 1.
  - **(e) the remainder, named here so that the classes are exhaustive:** exactly one of r_g <= 1, r_m <= 1 holds and
    the 1e-8 values do not agree (one code is self-consistent, but the codes disagree at 1e-8).
- A listed call whose pair is not matched is class "unpaired".

## C. What is reported
- Part A: counts with r_g <= 1 and with r_m <= 1, overall and by kind (group / analytic block label), band and k; the
  distribution (quantiles 0, 0.1, 0.5, 0.9, 0.99, 1 and bins <= 0.01, (0.01, 0.1], (0.1, 1], (1, 10], > 10) of r_g,
  r_m, d8_g, d8_m; every pair with r_g > 1 in full.
- Part B: per listed call its unit, index, kind/group, band, k, rtol, both values and abserrs, |Δ|/e_g and |Δ|/e_m, the
  maximum depth and region count under each code, whether it is in V4's set (z00, z12, Z3's seven, the 24), its class;
  the counts per class by band and k; for each listed call's unit (a row of the 720 or not), its row totals under each
  code, |Δ total| / total_abserr under each.
- For classes (c) and (d): the reference cost by refs_projection_07.py's model, unchanged, where a capture can be made
  without Ray, under `nice`, in one process, each checked bit for bit against 07's census; a call the model does not
  cover (Route A, an analytic block) is reported as not projected.
