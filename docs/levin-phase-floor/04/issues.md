# levin-phase-floor 04 — found and not acted on

These are for the orchestrator to place on the boards. Nothing here was changed.

The first dispatch stopped at step 5 (I1). The user then amended the two gates (prompt §8, commit
`d170e6d`). The resume passed them and ran `levin-phase-floor-04-remove-20261025T231110` to `done`.
I1–I5 were written at the stop, and their text is kept. I6–I9 were written after the run.

## I1. E1 (c) and E2 cannot hold at round-off: the bounds leave out two round-off terms

**Resolved for this prompt by §8 (`d170e6d`).** The user amended E1 (c) and E2 with the two terms.
Under the amended gates the worst ratios are 0.379 for E1 (c), and at most 0.772 for E2 (A) and
0.203 for E2 (X). The text below is the record at the stop.

**Proposed tag:** `[04-e1-e2-bounds-omit-abscissa-and-remainder-rounding]`.

**What was measured** (`out/phases_04.txt`, `out/diag_abscissa_rounding_04.txt`):

- **E1 (a) and (b) hold everywhere.** That is 2000 abscissae per factor per Levin sub-interval on
  all seven rows. The worst ratios are 1.000 for (a) and 0.870 for (b).
- **E1 (c) fails on the lone-`G` sub-interval of six rows** (R1, R2, R4, R5, R6 and R7), at 23–160
  of 2000 abscissae each.
  - The largest |wrap(ρ^A − ρ^X)| is 1.776e-15, which is 2⁻⁴⁹. The bound there is about 1e-15,
    because Φ^A_G is O(1) at the top of the source range.
  - Every failing abscissa has |ρ^A| ≥ 8. There, the double ρ^A has an ulp of 1.78e-15. The arm's
    sum ρ^A = rem + c_G lies in (−4π, 0] and is not re-reduced, as prompt §2 specifies.
  - So the arm's own representation is coarser than the bound. On top of that come the roundings
    of c_G and of `WKB_mod_2pi`'s `r − TWO_PI` (each ≤ 4.4e-16) and ρ^X's own (≤ 4.4e-16).
  - The wrap is evaluated exactly, on the two doubles, so the measurement adds no error.
  - Every other G factor passes (c). That includes R3's lone `G`, where Φ^A is larger, with a
    worst ratio of 0.666; the rest are at most 0.394.
- **E2's P clause holds** in all 29 scored groups. The ratio σ_θ/(ε ΣΦ^P) is 0.21–0.34.
- **E2's A clause fails in 5 of 29 scored groups.** These are the four lone `G` groups (R1, R4, R6
  and R7, ratios 2.57–6.57) and `G+q+r` on R6 (1.28). The other 24 have ratios 0.21–0.72.
  - For the lone `G`, ε ΣΦ^A is 3.7e-16 to 1.0e-15. The measured σ_θ is 1.9e-15 to 3.8e-15. The
    median |composed phase| under A is about 9.4, where a double's ulp is 1.78e-15.
- **E2's X clause "≤ ε ΣΦ^X" fails in 19 of 29 scored groups**, by ratios of 1.00–21.4 (R1 `G+r`
  is 2.93e-15 against 2.93e-15, over by less than the printed digits). The other 10 have ratios
  0.43–0.94.
  - The clause "≤ 1e-2 F" holds in all 29, with σ_θ^X at most 6.5e-14, against a 1e-2 F of 1.7e-8
    to 9.2e-7.
- **What the X jitter is.** It is the abscissa's own rounding. `PrimitivePhase._get_x` forms
  z = expm1(x) as a double, and the leading term is evaluated at that z. That carries
  (dθ/du)·δz/(1+z), about ε|dθ/du|, into the phase.
  - `diag_abscissa_rounding.py` predicts the X residual from the exact expm1 alone. On R3 `G−r`
    and R4 `G−r` the correlation with the measured residual is 0.9996–1.0000, and what remains is
    0.4–3e-15. On R1 `G+q+r` the correlation is 0.94–0.97.
  - The prompt's Φ^X = k|p| + 2π does not include this term. In every failing group,
    ε Σ|dθ_i/du| is at or above the measured σ_θ^X.
  - P and A carry the term too. There it is far below ε Φ^P, and below ε Φ^A wherever Φ^A is not
    O(1).

**Why the gates are not met, in one line each.** E1 (c)'s bound is below one ulp of the value
arm A produces. E2's model εΣΦ leaves out the abscissa-rounding term ε|dθ/du| (X and A) and the
granularity of a remainder sum of magnitude up to 4π (A). Every failure is between 4e-16 and
6.5e-14 rad. F is 1.7e-6 to 9.2e-5. Under X the composed jitter is at most 4e-8 F.

**Impact.**
- Prompt §5: *"A gate that fails is a stop, not a verdict."* §7: *"E1, E2 or M0 fails"* is a stop
  and ask. So step 6 (the predictions file) and step 7 (the run) were not done.
- Nothing measured says the arms are wrong:
  - E1 (a) and (b) hold;
  - ρ^X is the correctly rounded exact remainder at all 80,000 E1 abscissae (checked with
    `fractions`);
  - ρ^A − ρ^X stays within 1.8e-15 everywhere;
  - P reproduces 03's printed medians in all 29 groups.

**Next step.** The user decides how to proceed. The options as the agent sees them, none taken:
1. treat the failures as round-off below the scale the gates protect, and launch as written;
2. re-pose E1 (c) and E2 with the two missing terms (ε|dθ/du|, and ½ ulp of the arm's composed
   value). README §5 rule 3 forbids re-posing a scored rule after the numbers are in, but these
   are gates;
3. change arm A to re-reduce ρ^A into (−TWO_PI, 0]. That departs from prompt §2, and would only
   remove part of (c)'s excess.

## I2. G's residual φ is at the floor's own scale, in the stored samples and in the consumer

**Proposed tag:** `[04-gk-phi-samples-carry-the-floor]`. This is descriptive, from step 5.

- **Production's φ_G at its samples is quantised.** `build_phi_samples` forms
  φ = float(div·TWO_PI + mod) − s·k·Δ in double, with θ_G ~ 1e10–4e11. So φ_G is a multiple of
  ulp(θ_G). Its max |φ| is 1.9e-6 to 1.2e-4, and in over half the 9-sample windows it is exactly
  zero.
  - Over the Levin windows, its window RMS is 0 at the median, 0.25–0.41 F at the 90th
    percentile, and 0.46–0.78 F at the maximum.
- **Rebuilt in mpmath from the stored div/mod and Δ^ext, φ_G still carries noise.** It is at
  0.15–0.21 F (median) on every row, against an F of 1.7e-6 to 9.2e-5.
  - So most of the noise is **in the stored samples** (the producer's θ_mod_2pi), not only the
    consumer's subtraction.
- **What follows.** The φ spline interpolates this noise smoothly between samples. So every arm,
  X included, carries an O(0.2–0.7 F) smooth phase error that varies on the sample spacing. X is
  therefore "not the truth" by about F, as prompt §2 anticipates, and this measures how far.
- **T's φ is far quieter.** Over the Levin windows its window RMS is ≤ 3e-5 F at the median and
  ≤ 0.06 F at the maximum. Over all windows, the largest is 0.39 F (R2's `r`, outside its Levin
  sub-intervals).
- **Bearing.** On Q3 (whether layer 1 must change `PrimitivePhase`, and what is stored), and on how
  N4 and R2 are read once they are run.

## I3. The modulus error of G is 0.175 F on every row

This is descriptive. max |θ_G| / TWO_PI × (2π − TWO_PI) is 0.175 F on all seven rows: 2.96e-7 rad
on R1, up to 1.61e-5 on R3. It is a smooth phase error that P, A and X share, as prompt §1 and §2
say. For T it is 1.9e-4 F to 0.051 F. Recorded for the remedy prompts.

## I4. `runregistrykit` takes `git_head` from the script's directory

**Settled for this prompt by §8.** The run used `script=` the main checkout's path, so the manifest
records `c8ca155`. Every payload records the worktree's `98e68d3` and its empty status. The
observation stands for any later harness that runs from a worktree.

`runregistrykit.begin` (v0.2.0) records `git_provenance(start)`, and `start` is the directory of
`script` when one is given (`__init__.py`, `begin`). This prompt's scripts must live in the main
checkout's `var/levin-phase-floor-04/scripts/`, since nothing may be written in the worktree. So a
`begin(..., script=<that path>)` records the main checkout's `HEAD` (the records), not `98e68d3`
(the code).

The only way to record `98e68d3` is `script=None` with the launcher's working directory in the
worktree. That loses the registry's own `script_sha256` stamp on every checkpoint record.

This was found while planning step 7, which was not reached. **Not a defect** of the package: it
does what its docstring says. The launch design is for whoever resumes (log, State handed on).

## I5. The registry run was never begun; `black` wrote its cache outside the allowed area

`python -m black` was run once on four of this prompt's scripts. It wrote its cache file under
`~/Library/Caches/black/25.1.0/` at 22:50Z, which is outside `var/levin-phase-floor-04/`.

It is harmless, but it is a write the orchestrator's rules forbid. It is recorded in the log as
UNINTENDED DRIFT. No further `black` run was made.

## I6. With every remainder exact, X at rtol 1e-8 still sends one short chain of regions to the cap

**Proposed tag:** `[04-x-at-1e-8-still-reaches-depth-20]`. This is descriptive.

- **At rtol 1e-7,** no group under X or A reaches depth 20. The deepest are the `Gqr` groups on
  R4–R7: 2–3 regions at depth 18–19.
- **At rtol 1e-8,** X reaches depth 20 on R1, R4, R5, R6 and R7, always in the same pattern:
  exactly 2 regions at depth 20 and 4 at depth 18 or more, in each affected group.
  - That is up to five groups per row, and 4–10 depth-20 regions per row.
  - R2 and R3 do not reach it.
- **The pattern** is one bisection chain to the cap at a single point. The likely candidates are a
  kink of the integrand, such as a clamp edge or a sub-interval boundary, or the remaining
  abscissa-rounding jitter near ε|dθ/du|. **Not investigated.**
- **Cost and impact.** It costs nothing measurable: 1–3 s per row. It does not affect R2's
  agreement, where every ratio is ≤ 0.47. **Bearing:** on G2's Q4 (the depth cap) and on layer 2.

## I7. Arm A matched X on every row, including the groups whose F^A_g is about F

**Proposed tag:** `[04-layer-1a-reaches-the-exact-limit-on-these-rows]`. This is descriptive.

- **A and X are the same, region for region.** On every row, A at 1e-7 has exactly X's region
  count (66–167), its maximum depth, and its depth-18 counts in every group.
  - The totals differ by at most 1.17e-12 relative (R4).
  - A ran faster: 0.4–1.1 s, against X's 0.8–2.1 s, because it does no mpmath.
- **F^A_g did not predict the depth.** In the `Gqr` groups, F^A_g (step 5) is 1.1–1.3 F.
  - That is the maximum over the sub-interval, which reaches z_response, where T's phase is large.
  - Yet A sent 0–3 of their regions to depth 18 or deeper, as X did.
  - So the deep regions those groups had under P sit where Φ^A is small. A1's covered set (F^A ≤
    1e-8) is therefore conservative on these rows.
- **Bearing:** G1's Q2. On these seven rows, layer 1(a) alone reaches what an exact composition
  reaches.

## I8. N4: the stored total_abserr does not cover the jitter's effect on three rows

**Proposed tag:** `[04-stored-abserr-understates-the-jitter-on-3-of-7-rows]`. It narrows
`[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` (`handover`).

**What was measured.** The ratio |total_P − total_X(1e-8)| / `total_abserr`_P on each row (stored
total = P, by M0):

| Row | Ratio | Relative difference |
|---|---|---|
| R1 | 0.440 | — |
| R2 | 0.0795 | — |
| R3 | **1.161** | 2.71e-4 |
| R4 | 0.245 | — |
| R5 | 0.445 | — |
| R6 | **1.533** | 6.6e-7 |
| R7 | **1.942** | 7.3e-7 |

- **The stored rows understate their error on R3, R6 and R7.** R3's large relative error comes
  with a small, cancelling total of 7.0e-29.
- **On R6 and R7,** P's `total_converged` is False, while X at 1e-8 converges with a declared error
  30–41× smaller.
- **Caveat (I2):** X keeps φ's O(F) sample noise and the modulus error, so X is not the truth.
  P − X is the effect of the jitter alone.

## I9. mpmath's `from_float` swallows any exception and returns NaN

`mpmath/libmp/libmpf.py:418-424` (mpmath 1.3.0) wraps `math.frexp(x)` in a bare `except:`. On any
exception it returns `fnan` for a finite `x`.

**How it was found.** An exception raised by a signal handler inside it is swallowed, and the value
silently becomes NaN. That happened once in this prompt's first smoke test of a SIGALRM budget.

**What it means.** It is a hazard for any timeout or interrupt implemented by a signal around
mpmath code. It is not this repository's code and is not a defect of it. This prompt's run raises
its budget exception from the dispatcher instead (log deviation 10). Recorded for whoever writes the
next such harness.

