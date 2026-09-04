# Log 01 — Refuse or report: finiteness, validation, and an honest aggregate

**Prompt:** prompts/levin-refactor/01-refuse-or-report.md
**Commit:** Make Levin quadrature refuse bad input rather than certifying it; SHA intentionally omitted — see README §5 rule 5
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

All seven items in the prompt, in `AdaptiveLevin/levin_quadrature.py` and
`AdaptiveLevin/tests/test_levin_quadrature.py`.

1. **C1 — non-finite propagation (item 1).** Four checks added, each printing a warning naming the
   region/label then raising `ValueError`:
   - `_adaptive_levin_subregion_impl:663-670` (post-formatting line numbers will drift; see the
     commit diff) — `f_Cheb`, immediately after sampling (`levin_quadrature.py`, right after the
     `np.hstack` that builds it). Names "sampled amplitude f".
   - `_Basis_SinCos.build_Levin_data` — `theta_prime_Cheb`, immediately after it is obtained
     (covers both the `theta_deriv`-supplied branch and the spectral-differentiation branch).
     Names "sampled phase derivative theta'". `build_Levin_data` gained an optional `label`
     parameter, threaded from `_adaptive_levin_subregion_impl`'s existing `label` variable, purely
     so this warning can name the region the same way the pre-existing `LevinL` check does.
   - `_adaptive_levin_subregion_impl` — the solved `p`, checked once immediately before
     `P = p.reshape(...)`, i.e. after all three solve exits (direct LU, `lstsq`, `pinv`) have had
     their chance to produce a value. Names "solved Levin antiderivative p".
   - `p_use = [np.isfinite(r) and r > rtol for r in p_ratios]` — replaces
     `p_use = [r > rtol for r in p_ratios]`, so a non-finite ratio is rejected because it is
     non-finite, not because `NaN > rtol` happens to be `False`.
   - `p_mean_max == 0.0` guard: when every component of `p` is identically zero, `p_ratios` is now
     `[1.0, 1.0, ...]` (all modes kept) instead of `pm / 0.0` producing `nan` with a
     `RuntimeWarning`. See *Deviations* below.

2. **C3, reporting half (item 2).** After `relerr_total` is computed, `requested_total =
   max(atol, rtol * |val|)` and `converged = abserr_total <= requested_total` are computed; a
   warning is printed in the same two-line style as the existing depth-limit and phase-limited
   health checks when `converged` is `False`. `"converged": bool(converged)` added to the returned
   dict. Acceptance logic (`resolved`, `phase_limited`, the per-region loop) is untouched.

3. **C6 (item 3).** `if current_region.depth > max_depth: max_depth = current_region.depth` moved
   from inside the "region accepted" branch (after the `if resolved or phase_limited or ...:`
   test) to immediately after `current_region = regions.pop()`, before the direct-quadrature
   branch's `continue`. The old copy was deleted, not duplicated.

4. **C8 (item 4).** `_adaptive_levin` now rejects `atol <= 0` with `ValueError` before doing
   anything else. Took the audit's "reject" option, not the "floor at 1e-300" option — see
   *Deviations*.

5. **C9, validation (item 5).** Added to `_adaptive_levin`: `rtol < 0`, `depth_max < 0`,
   `len(x_span) != 2`, and non-finite `x_span` entries all raise `ValueError` naming the offending
   parameter. Added to `adaptive_levin_sincos`: `len(f) != 2` raises `ValueError` naming the
   sin/cos basis's two-component contract (covers both the "wrong length" and "empty" rows of the
   audit's table, since both fail the same `!= 2` test). Added to `_Basis_SinCos.__init__`: a bare
   callable `theta` (rather than a dict) raises `TypeError` naming the mistake explicitly
   (`"did you mean theta={'theta': theta}?"`). `chebyshev_order < 8` now prints a warning naming
   the clamped value, once per call to `_adaptive_levin`, rather than clamping silently; the
   clamping behaviour itself (in `_adaptive_levin_subregion`) is unchanged.

6. **C9, message quality and docstring (item 6).** `adaptive_levin_sincos` gained a full
   docstring: what it computes, the two-component `f` contract, the three `theta` dict keys and
   their accuracy consequences, the `atol`-is-per-region / `converged` caveat (with a note that
   prompt 05 changes this), and every key in the returned dict. The NaN-endpoint row of the audit's
   table is now handled by the new `x_span` finiteness check in `_adaptive_levin`, which fires
   before the `LevinL` super-operator is ever built — this is a strict improvement on the "message
   misattributes the cause" complaint (item 6), since the NaN is now caught and named at the point
   it enters (`x_span`) rather than three layers downstream.

7. **Tests (item 7).** Added `test_nan_amplitude_raises`, `test_all_nan_amplitude_raises`,
   `test_input_validation` (one assertion per row of the audit's table, including the
   `chebyshev_order < 8` warn-not-raise row via `contextlib.redirect_stdout`), `test_atol_zero_rejected`,
   `test_converged_flag`. Suite grew from 4 to 9 tests, runtime 0.011s (was 0.008s).

## Numerical evidence

Required by README §5 rule 9. Three problems, computed on the `c8a1918` baseline (via `git stash`)
and again after the full change (via `git stash pop`), using `PYTHONPATH=. ./venv/bin/python`:

| Problem | Metric | Before | After |
|---|---|---|---|
| `∫₁^100 sin(100x)/x dx` (existing test, `test_SincIntegral`) | `value` | `0.00866607849680814` | `0.00866607849680814` |
| | `abserr` | `1.569602979674854e-13` | `1.569602979674854e-13` |
| | `num_regions` | `10` | `10` |
| `∫_{1/3}^{7/3} e^{-x} sin(10⁶x) dx` | `value` | `-3.347268410464638e-07` | `-3.347268410464638e-07` |
| | `abserr` | `1.898176214528212e-16` | `1.898176214528212e-16` |
| | `num_regions` | `1` | `1` |
| Three-Bessel `J000` oracle, `k=1.3, q=1.7, s=2.1, max_x=1e5` (via `quad_JJJ`, phase built with `bessel_phase(atol=1e-25, rtol=5e-14)`, quad `atol=1e-14, rtol=1e-10`) | `value` | `0.16923010755911763` | `0.16923010755911763` |

All three are **bit-identical**, not merely close, confirming this commit does not touch arithmetic
on valid input.

The closed form for the second problem is
`[-e^{-x}(sin(ωx)+ω cos(ωx))/(1+ω²)]_{1/3}^{7/3}` at `ω=10⁶`: `-3.3472684114661845e-07`, giving
`relerr ≈ 2.99e-10` against the delivered value — consistent with the ~10⁻¹⁰ accuracy this problem
was already known to deliver at this frequency (README §5.2), unaffected by this commit.

C3 reproducer, exactly as measured in README §2.2: `∫₀³ e^{-400(x-1.1)²} sin(3×10⁴x) dx` at
`atol=1e-10` (default `rtol=1e-7`) now prints

```
!! WARNING (adaptive_levin, ...): the aggregate error estimate exceeds what was requested |
abserr_total=1.26e-10 > requested=1e-10 (atol=1e-10, rtol=1e-07)
```

and returns `converged=False` — `abserr_total=1.26e-10` matches the audit's and the plan's
reproduction exactly.

## Deviations from the prompt

### Implementation choice: `p_mean_max == 0` behaviour (item 1)

The prompt flags this as an open judgement call and recommends "an all-ones ratio vector and an
all-`True` `p_use`" as the behaviour-preserving choice. Took exactly that: when `p_mean_max <= 0.0`
(every component of the solved `p` is identically zero on this region), `p_ratios = [1.0, ...]`
rather than `pm / p_mean_max` (which was `0.0 / 0.0 = nan` with a `RuntimeWarning`). Combined with
the new `np.isfinite(r) and r > rtol` gate in `p_use`, an all-ones `p_ratios` gives an all-`True`
`p_use`, so every (identically-zero) mode is kept — the region correctly contributes exactly zero
via `lower_limit`/`upper_limit`, computed from genuinely-zero `p` values, not discarded by a
divide-by-zero artefact. This is reachable in production: `QuadSourceIntegral.py` passes
`f = [Levin_f, lambda x: 0.0]`, and a region where the amplitude underflows gives `p ≡ 0` for that
component and possibly for both.

### Implementation choice: reject vs. floor `atol <= 0` (item 4)

Took the audit's recommended "reject" option over "floor at ~1e-300 and floor `relerr_denom` at a
positive number derived from the running total". Reasoning, beyond what the prompt already gives:
flooring would need a second new floor (on `relerr_denom`, itself already using `atol` as its own
floor at `:relerr_denom = max(min(...), atol)`) whose magnitude has no principled derivation yet —
"derived from the running total" is underspecified until the aggregate-tracking machinery from
item 2 exists, and even then it is a second free parameter to tune and defend. Rejecting is a single
`if not (atol > 0): raise` with no new constant, is honest about the module's actual contract (the
phase-rounding floor is absolute, so a purely-relative request is not answerable), and gives the
caller an actionable message. Not revisited; if a caller genuinely needs `atol=0` semantics later,
that is new scope, not a bug in this choice.

### Implementation choice: where to validate `len(f) != 2` (item 5)

The prompt says the two-component contract "belongs to `_Basis_SinCos`" but also that
`_adaptive_levin`/`_adaptive_levin_subregion_impl` "are written for general `m` and must stay that
way" — and `_Basis_SinCos.__init__` does not receive `f` (only `theta`); `f` is only ever passed to
`_adaptive_levin`. Rather than changing `_Basis_SinCos.__init__`'s signature to take `f` solely for
validation (which would tie a data class to a value it doesn't otherwise use), the check was placed
at the top of `adaptive_levin_sincos` — the sin/cos-specific public entry point that is the actual
locus of the "two components" decision, since it is what chooses to construct `_Basis_SinCos` and
assumes `f` has the matching shape. This keeps `_Basis_SinCos` unchanged in its parameters, keeps
`_adaptive_levin`/`_adaptive_levin_subregion_impl` fully general in `m` (untouched by this
decision), and puts the validation at the one place that logically owns the "m=2" assumption. The
error message names the basis (`"the sin/cos basis _Basis_SinCos is fixed at two components"`) as
the prompt requires.

## Verification performed

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .` — ran and
  passed: `Ran 9 tests in 0.011s`, `OK`. Actually executed, output inspected (see the C1/C3
  reproducer warnings appearing in the test output, confirming the new code paths fire during the
  test run itself, not just in the standalone checks below).
- `./venv/bin/black --check AdaptiveLevin/levin_quadrature.py AdaptiveLevin/tests/test_levin_quadrature.py`
  — initially reported one file would be reformatted (whitespace only, a single line-wrap in the
  pre-existing `f_scale` comprehension that black's line-length rule now disagreed with after
  nearby edits shifted indentation); ran `black AdaptiveLevin/levin_quadrature.py`, then re-ran
  `--check` on both files: clean. Re-ran the full test suite after formatting: still 9/9 `OK`.
- Numerical no-change check: see *Numerical evidence* above. Actually run, both before (`git
  stash`) and after (`git stash pop`), with `repr()`-precision output compared for exact equality.
- Both C1 reproducers (partial-NaN, all-NaN) actually raise `ValueError` naming "amplitude" —
  verified via the new unit tests, which pass.
- Both C8 checks actually verified: `atol=0` raises `ValueError` naming "atol" (test and manual
  check); `atol=1e-15` on the existing `test_SinIntegral`-style problem still gives the same
  region/evaluation counts as before (no regression from the validation itself, since it only
  rejects `atol<=0` and nothing else).
- C6 reproducer actually constructed and run standalone: a narrow Gaussian bump
  (`exp(-2000(x-0.5)²)`) modulating `sin(40x)` on `(0,1)` at `atol=rtol=1e-30`, `depth_max=3`. The
  bump forces bisection for resolution while the modest total phase span (40, just over
  `2·SIX_PI≈37.7`) means sub-intervals fall below the `SIX_PI` net-phase-diff gate after two
  bisections, so every accepted region ends up on the direct-quadrature branch
  (`num_simple_regions == num_regions == 4`). On the pre-fix code (`git stash` back to baseline)
  this reports `max_depth=0`; on the fixed code it reports `max_depth=2`, matching the actual
  bisection depth reached. Both runs performed and compared directly, not reasoned about.
- C3 reproducer: actually run standalone (see *Numerical evidence*) and observed inside the new
  `test_converged_flag` test, which passed.

## Observations not acted on

- `_adaptive_levin_subregion`'s SVD-failure step-down loop (`levin_quadrature.py`, around
  `working_order = working_order - 2`) prints its own warning without a finiteness check on
  anything upstream of it; out of scope here, untouched.
- `used_interval.__str__`'s f-string nested-quote construction
  (`f"[{", ".join(...)}]"`) relies on PEP 701 (Python 3.12+) nested-quote support; noticed while
  reading the file, not touched — this prompt does not ask for it and the venv is 3.12 per
  README §5, so it already works here.
- The `_Basis_SinCos.__init__` missing-`"theta"`-key branch still raises `RuntimeError` rather than
  `ValueError` (pre-existing; the audit's table does not list this row as needing a type change,
  only a message fix, which was made). Left as `RuntimeError` to avoid an unrequested exception-type
  change on an untested path.
- `black` reformatted one pre-existing line unrelated to this prompt's edits (the `f_scale`
  comprehension). This is a one-line whitespace change forced by `black`'s check failing on the
  file as a whole after nearby edits; not a deliberate content change and does not affect behaviour.

## State handed to the next prompt

- The four finiteness checks (`f_Cheb`, `theta'`, `p`, and the `p_use` non-finite gate) are now in
  place and raise `ValueError`. Prompt 02 rewrites the solve (`np.linalg.solve` /
  `np.linalg.lstsq` / `pinv` block) to a complexified `N×N` system; it must carry the `p`
  finiteness check (currently sitting immediately before `P = p.reshape(...)`) forward onto
  whatever the new solve's output variable is called, at the same logical point (after all solve
  exits, before the ratios are computed).
- `build_Levin_data` now takes an optional `label` parameter purely for the `theta'` finiteness
  warning; any future caller of `build_Levin_data` (there is currently only the one, in
  `_adaptive_levin_subregion_impl`) should pass a label if one is available, but it is optional and
  defaults to `None` so this is not a breaking change.
- `_adaptive_levin`'s new input-validation block sits before `driver_start = time.perf_counter()`;
  prompt 03's restructuring of `_adaptive_levin_subregion_impl` and the weakly-oscillatory gate does
  not touch `_adaptive_levin`'s entry, so this should be unaffected, but worth checking that no new
  parameter needing validation is introduced without a corresponding check here.
- The `converged` key is new in the returned dict; the benchmark harness reads by key and does not
  break on new keys (standing note 8), confirmed by inspection, not by running the harness itself
  (out of scope for this prompt).
