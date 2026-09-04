# Log 05 — Make `atol` a global tolerance

**Prompt:** prompts/levin-refactor/05-global-tolerance.md
**Commit:** Distribute atol across subintervals so it bounds the total error (SHA intentionally omitted — see README §5 rule 5)
**Date:** 2026-09-04
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

- New helper `_local_atol(atol, a, b, x_span_width)` (`AdaptiveLevin/levin_quadrature.py:1329`,
  just before `_adaptive_levin`): returns `atol * |b - a| / x_span_width`, or `atol` unscaled if
  `x_span_width == 0.0` (see Deviations). `x_span_width = |x_span[1] - x_span[0]|` is computed once
  before the driver loop (`:1416`, right after `regions = [...]`).
- Inside the loop, `local_atol = _local_atol(atol, a, b, x_span_width)` is computed once per popped
  region, immediately after `a`/`b` are read off `current_region` (`:1477-1481`), and reused by both
  the fallback branch and the Levin branch below it.
- **The five `atol` sites**, all re-audited against the current code (line numbers are post-prompt-04,
  not the audit's pre-campaign ones the prompt cites):
  1. **Acceptance test** — `resolved = abserr < atol or relerr < rtol` → `abserr < local_atol or
     relerr < rtol`, in both the fallback branch (`:1583`, was the direct-quadrature `:1551`
     pre-edit) and the Levin branch (`:1740`, was `:1707`). **Scaled**, as directed.
  2. **Relative-error denominator floor** — fallback branch `relerr_denom = max(np.fabs(estimate),
     atol)` → `..., local_atol)` (`:1578`); Levin branch `relerr_denom = max(min(np.fabs(estimate),
     np.fabs(refined_estimate)), atol)` → `..., local_atol)` (`:1703`). **Scaled**, as directed and
     re-verified against the item-5 degenerate case below.
  3. **`phase_limited` conditions** — `phase_err > atol` → `phase_err > local_atol` in both branches
     (`:1595` fallback, `:1731` Levin), keeping the guard threshold aligned with the acceptance test
     it guards. **Scaled**, as directed.
  4. **Aggregate relative error** — `relerr_total = abserr_total / max(np.fabs(val), atol)` (`:1850`).
     **Left global**, unchanged, as directed.
  5. **`converged`** — `requested_total = max(atol, rtol * np.fabs(val))` /
     `converged = abserr_total <= requested_total` (`:1857-1858`). **Left global**, unchanged, as
     directed.
- **Diagnostics-only `atol` use left alone.** `_write_progress_data()` (`:2010` onward) recomputes its
  own `abserr_status`/`relerr_status` against the raw `atol` parameter it is passed (`:2106, 2110`).
  This is a separate, `emit_diagnostics`-only function that re-solves a region for a plot dump, not
  part of the accept/reject decision the five sites above implement; it is prompt 07's territory
  (diagnostics hygiene), not this prompt's, and rule 6 says not to touch it here.
- **Docstring and comments updated** to state the new contract:
  - `adaptive_levin_sincos()`'s `:param atol:` and `:param rtol:` (`:2227-2238`): atol is now a bound
    on the *summed* result via length-proportional distribution; rtol stays per-region, unscaled, with
    the reason stated (no additive length-proportional analogue).
  - The `:return:` `"converged"` bullet (`:2262-2263`): now usually True.
  - The in-loop and post-loop comments referencing "per-region tolerance" / "prompt 05 changes this"
    (`:1477-1480`, `:1700`, `:1852-1858`, `:1933-1937`, `:1869-1872`) rewritten to describe the current
    behaviour rather than announce a still-pending change; the `converged`-warning print body
    (`:1869-1872`) rewritten from "atol is currently a per-region tolerance" to name the two things
    that can still make it False (depth_max reached unresolved; a region's round-off floor exceeding
    its length-scaled share of atol).
- **Test updated.** `AdaptiveLevin/tests/test_levin_quadrature.py::test_converged_flag` asserted
  `assertFalse(data_bump["converged"])` on the audit's C3 reproducer (Gaussian bump × sin(3e4 x) on
  [0, 3] at atol=1e-10) — that was the pre-prompt-05 defect this prompt fixes, so the assertion is now
  inverted to `assertTrue(...)` plus `assertLess(data_bump["abserr"], 1e-10)`, with the comment
  rewritten to explain both the old and new behaviour.

## Numerical evidence

All runs `PYTHONPATH=. ./venv/bin/python`, `chebyshev_order=12`, `rtol=1e-13` (so acceptance is
governed by `atol`, matching the prompt's tolerance-matrix framing) unless stated otherwise. "BEFORE"
loads `AdaptiveLevin/levin_quadrature.py` at `HEAD` (prompt 04's state, `13773b9`) from a scratch copy
via `importlib`; "AFTER" is the working tree. Problems, five as required, three with closed forms used
throughout the campaign (README §5.2) and two constructed the same way:

- `sin_1_100`: `∫₁^100 sin(x) dx`, closed form `cos(1) − cos(100)`.
- `grz_100`, `grz_1000`: `∫₋₁¹ cos(λ·atan x)/(1+x²) dx`, closed form `(2/λ)·sin(πλ/4)`, λ = 100, 1000.
- `expsin_30000`, `expsin_1e6`: `∫_{1/3}^{7/3} e^{−x} sin(ωx) dx`, closed form as in README §5.2, ω =
  3×10⁴ (the audit's own C3 frequency) and 1×10⁶.

### Item 2 — the contract holds, atol=1e-8/1e-10/1e-12

| Problem | atol | BEFORE regions/evals | BEFORE abserr | BEFORE converged | AFTER regions/evals | AFTER abserr | AFTER converged | AFTER contract (abserr ≤ max(atol,rtol·\|val\|)) |
|---|---|---|---|---|---|---|---|---|
| sin_1_100 | 1e-8 | 1/3 | 2.43e-14 | True | 1/3 | 2.43e-14 | True | ✅ |
| sin_1_100 | 1e-10 | 1/3 | 2.43e-14 | True | 1/3 | 2.43e-14 | True | ✅ |
| sin_1_100 | 1e-12 | 1/3 | 2.43e-14 | True | 1/3 | 2.43e-14 | True | ✅ |
| grz_100 | 1e-8 | 2/7 | 5.11e-9 | True | 2/7 | 5.11e-9 | True | ✅ |
| grz_100 | 1e-10 | 4/15 | 2.67e-12 | True | 4/15 | 2.67e-12 | True | ✅ |
| grz_100 | 1e-12 | 24/51 | 2.24e-12 | **False** | 32/67 | 4.43e-13 | **True** | ✅ |
| grz_1000 | 1e-8 | 2/7 | 6.40e-11 | True | 2/7 | 6.40e-11 | True | ✅ |
| grz_1000 | 1e-10 | 2/7 | 6.40e-11 | True | 2/7 | 6.40e-11 | True | ✅ |
| grz_1000 | 1e-12 | 4/15 | 1.20e-13 | True | 4/15 | 1.20e-13 | True | ✅ |
| expsin_30000 | 1e-8/1e-10/1e-12 | 1/3 | 2.15e-16 | True | 1/3 | 2.15e-16 | True | ✅ |
| expsin_1e6 | 1e-8/1e-10/1e-12 | 1/3 | 2.15e-16 | True | 1/3 | 2.15e-16 | True | ✅ |

The C3 reproducer specifically (`∫₀³ e^{−400(x−1.1)²} sin(3×10⁴x) dx`, atol=1e-10):
`converged` False → True, `abserr` 1.27e-10 (over the request) → 6.22e-13 (well under). This is the
regression check `test_converged_flag` now encodes.

**`converged` is True in every AFTER cell above**, including the one case (`grz_100` at 1e-12) where
it was False before. That is the one problem/tolerance combination in this matrix where scaling
changes anything (1 or 2-region problems have `local_atol == atol` identically, since the single
region's length fraction is 1 or, for `grz`'s symmetric 2-region case at low `depth`, close enough
that the acceptance test lands the same way — see item 3 for why).

### Item 3 — cost at matched delivered accuracy: BEFORE@atol=1e-12 vs AFTER@atol=1e-10

| Problem | BEFORE@1e-12 regions/evals | BEFORE@1e-12 abserr | BEFORE@1e-12 true error | AFTER@1e-10 regions/evals | AFTER@1e-10 abserr | AFTER@1e-10 true error |
|---|---|---|---|---|---|---|
| sin_1_100 | 1/3 | 2.43e-14 | 1.17e-14 | 1/3 | 2.43e-14 | 1.17e-14 |
| grz_100 | 24/51 | 2.24e-12 | 3.78e-13 | 4/15 | 2.67e-12 | 3.49e-13 |
| grz_1000 | 4/15 | 1.20e-13 | 1.20e-13 | 2/7 | 6.40e-11 | 6.42e-11 |
| expsin_30000 | 1/3 | 2.15e-16 | 2.65e-17 | 1/3 | 2.15e-16 | 2.65e-17 |
| expsin_1e6 | 1/3 | 2.15e-16 | 1.00e-16 | 1/3 | 2.15e-16 | 1.00e-16 |

**I cannot cleanly reproduce "free at matched accuracy" on this problem set, and I am reporting that
with numbers rather than asserting the audit's claim by citation.** `sin_1_100` and both `expsin_*`
problems land in 1 region regardless of tolerance in this range, so BEFORE@1e-12 and AFTER@1e-10 are
identical by construction (no region-count effect to measure). `grz_1000` diverges the other way:
BEFORE@1e-12 delivers ~1.2e-13 true error at 15 evaluations; AFTER@1e-10, on only 2 regions, has
`local_atol` within a factor of 2 of the nominal `atol=1e-10` (the length fractions are close to 0.5
each), so it behaves almost like the *unscaled* scheme at 1e-10 — 500× looser (6.4e-11) at less than
half the cost. `grz_100` is the one case that supports the audit's framing: BEFORE@1e-12 needed 24
regions/51 evaluations to reach 3.78e-13 true error (and still failed its own converged check, per
item 2); AFTER@1e-10 reaches an *extremely close* true error (3.49e-13) at under a third the cost
(4 regions/15 evaluations). The direction of the audit's claim — that a caller can loosen the nominal
`atol` by roughly the region-count factor and land on comparable accuracy — reproduces cleanly on the
one problem in this set with enough regions for the effect to be visible (`grz_100`, 4–32 regions
across this tolerance range); it does not transfer to problems that resolve in 1–2 regions, where
`local_atol ≈ atol` regardless of the nominal value chosen. This is expected from the definition of
`_local_atol()` — the scaling factor for an N-region problem where all regions have comparable width is
of order 1/N — and is consistent with the audit's own qualifier "on three of the five problems"
(README §2.2, not all five).

### Item 4 — cost at fixed nominal atol

This is the same matrix as item 2, read the other way: at fixed `atol`, `sin_1_100`, `grz_1000` and
both `expsin_*` problems show **zero** evaluation-count change (1–2 regions throughout this tolerance
range; `local_atol` differs from `atol` by less than the factor that would flip an accept/bisect
decision). `grz_100` at `atol=1e-12` is the one combination that moves: evaluations 51 → 67
(**1.31×**, inside the audit's 1.1–1.5× band), for a reported error bound 2.24e-12 → 4.43e-13
(**5.05×** tighter — the *reported* abserr, which is what the contract is measured against; the
*true* error against the closed form is essentially unchanged, 3.78e-13 → 3.78e-13, because the
underlying resolution was already fine — what changed is that the delivered result is now honestly
certified rather than accepted while over budget). This is a smaller accuracy gain than the audit's
10–100× band, on the one measured case where cost changed at all; I am reporting the measured number
rather than the audit's range because they differ, and the reason (this case's true error was already
below both tolerances; only the *certification* changed) is visible in the numbers above.

### Item 5 — degenerate case: identically-zero integrand

`f = [0, 0]`, `theta(x) = 1e5·x`, `x_span=(1.0, 2.0)`, `atol=1e-15`, `rtol=1e-10`, `depth_max=8`
(prompt 01 item 4's case, re-run because item 2 above changes the relerr-denominator floor that this
case depends on):

```
value=0.0, abserr=0.0, num_regions=1, evaluations=3, converged=True
```

Unchanged from the pre-prompt-05 baseline (1 region / 3 solves) — the scaling is a no-op here because
there is exactly one region, so `local_atol == atol` by the definition in `_local_atol()`.

### Item 6 — reversed span

`f=[1, 0]`, `theta(x) = 100x`, `atol=rtol=1e-12`:

```
forward (1.0, 5.0):  value= 0.017461681457191774, converged=True, regions=1
reversed (5.0, 1.0): value=-0.01746168145719162,  converged=True, regions=1
sum = 1.53e-16  (round-off)
```

Correctly negated, both converge, and `_local_atol()`'s `np.fabs(b - a)` keeps the scale factor
positive regardless of region orientation.

### Item 7 — three-Bessel oracle (J000)

Running the full 7-oracle, `max_x=1e12` matrix used elsewhere in this campaign was not attempted here
on cost grounds (it is what makes `LiouvilleGreen/tests/test_3bessel_analytic.py` slow, and it was
already exercised, unaffected by this prompt's kind of change, in earlier prompts' logs). Instead:
`J000` at `(k, q, s) = (1.3, 1.7, 2.1)`, `max_x = 1e6` (still exercises `_Levin_JJJ`'s full four-phase-
group decomposition, `phase_atol=1e-25, phase_rtol=5e-14, quad_atol=1e-14, quad_rtol=1e-10`, matching
the test file's own defaults except `max_x`), `adaptive_levin_sincos` monkeypatched between the
before/after module while the shared `bessel_phase` splines (independent of this module, computed
once) are reused:

```
analytic J000(k=1.3, q=1.7, s=2.1) = 0.16923037349654133
BEFORE (prompt 04): value=1.692305e-01  abserr_vs_analytic=1.017e-07  relerr=6.010e-07
AFTER  (prompt 05): value=1.692305e-01  abserr_vs_analytic=1.017e-07  relerr=6.010e-07
```

Identical to the digits printed (both runs land well inside the test suite's own `REL_TOLERANCE=1e-5`).
No regression on the production three-Bessel code path.

## Deviations from the prompt

### STRUCTURALLY REQUIRED — the "zero-width span returns 0.0 before the loop" premise does not hold

The prompt's final bullet under "Reversed spans" states: "A zero-width span returns 0.0 before the
loop; confirm the scaling cannot divide by zero." I looked for this early-return and could not find
one — there is no zero-width guard anywhere in `_adaptive_levin` or `adaptive_levin_sincos`. I
confirmed empirically that a zero-width call currently **raises**, not returns 0.0:

```
adaptive_levin_sincos((5.0, 5.0), [lambda x: 1.0, lambda x: 0.0], {"theta": lambda x: 3.0*x},
                       atol=1e-10, rtol=1e-10)
  -> ValueError: sampled phase derivative theta' contains non-numeric values (np.nan, np.inf, or np.-inf)
```

raised from `build_Levin_data()` (`:831`), reached via `_adaptive_levin_subregion_impl` *before* the
driver loop's acceptance test — i.e. before any of this prompt's code runs at all, since the crash
happens while producing `data` for the one and only region, and `local_atol` is only consulted once
`data` comes back. This is pre-existing behaviour, unrelated to anything prompt 05 changes, and out of
this prompt's scope per rule 6 ("do not fix things the prompt did not ask for") — I did not add a
zero-width guard to `_adaptive_levin`. What I did do: `_local_atol()` still special-cases
`x_span_width == 0.0` to return `atol` unscaled rather than dividing by zero, both because it costs
nothing and because it is the mathematically correct answer if that upstream crash is ever relaxed (a
zero-width original span can only ever contain a zero-width region, so the length fraction is 1
either way). Recorded as an open item on the status board (§3) for whoever next touches input
validation or the zero-width path.

### IMPLEMENTATION CHOICE — `local_atol` computed once per region, ahead of the branch split

The prompt describes the acceptance test, denominator floor and `phase_limited` guard as though each
is edited independently. I introduced a single `local_atol = _local_atol(atol, a, b, x_span_width)`
right after `a`/`b` are read off the popped region (before the fallback/Levin branch split), and both
branches read that one local variable. Alternative considered: compute it separately inside each
branch (closer to a literal per-site edit). Rejected because `a`/`b` are identical for both branches at
that point in the loop (they describe the same popped region, before its `is_direct` outcome is known)
and a single computation makes it visually obvious that the fallback and Levin branches are using the
*same* scaled tolerance, which is exactly the self-consistency the prompt asks for in its
"Recommended: scale it" notes for sites 2 and 3.

### None beyond the above

No other deviations. The acceptance test's structure (`abserr < ... or relerr < ...`) is untouched;
no global worst-first balancing was introduced; the total-variation gate, the round-off floor and
`p_use` are untouched.

## Verification performed

1. `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — **21 tests,
   OK** (16 pre-existing + `test_converged_flag`'s assertions updated in place, no test added or
   removed). Actually run, not reasoned about.
2. Items 2–7 above: all actually run, via a scratch comparison script
   (`/private/tmp/.../scratchpad/verify_05.py` and `verify_05_bessel.py`, not committed — outside the
   repository) that loads the pre-prompt-05 module from `git show HEAD:...` as a separate Python module
   and the working tree as the current one, and calls both directly. Not reasoned about; every number
   in the tables above is copy-pasted from actual runs.
3. Manually re-derived, by reading the code, that `_write_progress_data()`'s own `atol` use is
   independent of the five driver-loop sites (different function, different purpose) — not run, since
   it requires `emit_diagnostics=True` and writes to disk; reasoning only, consistent with rule 6 and
   with prompt 07's stated scope for that function.

## Observations not acted on

- `_write_progress_data()` (`:2010`) computes its own `abserr`/`relerr` against the raw, unscaled
  `atol` it is passed, and its docstring-adjacent comments were not touched. This is diagnostics-only
  (`emit_diagnostics=True`), prompt 07's territory, not this one's.
- The zero-width `x_span` crash documented above under Deviations is a real, reachable defect (any
  caller that passes `x_span=(x, x)`), but it predates this campaign's prompt 01 input validation (which
  checks `len(x_span) == 2` and finiteness, not distinctness) and is not something prompt 05 was asked
  to fix. Left as a status-board item.

## State handed to the next prompt

- **C3 is now fully closed** (prompt 01 closed the reporting half; this closes the distribution half).
  `converged` is a meaningful, usually-True signal for prompt 06 and any later prompt to build on.
- **`_local_atol()` is the pattern for anything else that needs a region's share of `atol`.** Prompt 06
  (mode filter / `p_use`) adds a discarded-mode contribution to `abserr`; if it needs a tolerance
  reference for that contribution, `local_atol` (already computed once per region in the loop, in scope
  at both branch sites) is the length-scaled quantity to compare against, not the raw `atol` parameter.
- **Open status-board item**: zero-width `x_span` raises from `build_Levin_data()` rather than
  returning a defined result; not blocking, not this campaign's stated scope, but worth a line so a
  future input-validation pass (or a future edit to `_local_atol()`) doesn't have to rediscover it.
