# Log 03 — The closed-form tail and the remainder-tested crossover

**Prompt:** prompts/transfer-remedial/03-closed-form-tail.md
**Commit:** *(this commit; SHA not self-embedded)* — Add the closed-form Bessel tail and its crossover test
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

> **Read deviation 1 first.** The third coefficient of DLMF 10.18.18 as printed in
> `DRAFT-PLAN.md` §7.2, in this prompt's §2 and in its §5 reproduction script has denominator
> **15360**, and that is wrong by a factor of three: the correct denominator is **5120**. The
> shipped module uses 5120 (and 229376 for the fourth coefficient), both determined numerically
> against `mpmath` and cross-checked against the `(8x)^k` grouping of Abramowitz & Stegun 9.2.29.
> Nothing else in the campaign is affected — `bessel_reference.tail_residual_series` carries only
> the first two coefficients, 8 and 384, which are correct.

---

## What shipped

### `LiouvilleGreen/bessel_tail.py` (new, 434 lines)

A pure, self-contained module: no state, no interpolation, no sampling, no branch tracking, and
no `scipy.special` import at all (asserted by a test that checks `sys.modules` in a subprocess).

Module constants:

```python
TAIL_SERIES_MAX_TERMS = 3            # terms the module will evaluate
TAIL_SERIES_COEFFICIENT_COUNT = 4    # ... plus the omitted-term estimator
DEFAULT_TAIL_TERMS = 3
DEFAULT_CROSSOVER_SAFETY = 0.25      # see deviation 3
SAFE_WINDOW_MAX_FACTOR = 4800.0      # DRAFT-PLAN.md 5.3's window top, as a multiple of max(nu, 1)
DEGENERATE_MIN_X = 1e-5
```

Public functions:

```python
construction_min_x(nu: float) -> float
tail_crossover_max_x(nu: float) -> float
tail_series_coefficients(nu: float) -> Tuple[float, float, float, float]

tail_residual(nu, x, n_terms=3)            # r_nu(x)
tail_residual_deriv(nu, x, n_terms=3)      # dr/dx
tail_residual_log_deriv(nu, x, n_terms=3)  # dr/d(log x) = x r'
tail_amplitude(nu, x, n_terms=3)           # a_nu = (1 + r')^(-1/2)
tail_first_omitted_term(nu, x, n_terms=3)  # |c_n| / x^(2n+1)

tail_crossover(nu, phase_atol, amplitude_rtol, n_terms=3, x_max=None,
               safety=DEFAULT_CROSSOVER_SAFETY) -> TailCrossover
```

`TailCrossover` is a `NamedTuple` with fields
`(nu, x_star, first_omitted_at_x_star, terms_used, phase_atol, amplitude_rtol, safety, x_phase,
x_amplitude, amplitude_error_at_x_star, binding)`; `binding` is `"phase"`, `"amplitude"` or
`"domain"`.

The five evaluators accept a scalar or an array of `x` and return the matching type (`float` for
a scalar, `np.ndarray` for an array). All of them raise `ValueError` on `x <= 0` or on
`n_terms` outside `[1, 3]`.

Series (repository convention, `mu = 4 nu^2`), with the coefficients as shipped:

    r_nu(x) ~ (mu-1)/(8x) + (mu-1)(mu-25)/(384 x^3)
              + (mu-1)(mu^2-114mu+1073)/(5120 x^5)
              + (mu-1)(5mu^3-1535mu^2+54703mu-375733)/(229376 x^7) + ...

with the fourth term used only as the omitted-term estimator. Evaluation is by **Horner in
`w = 1/x^2`** (`_horner`), stated and justified in the module docstring: `x**7` overflows above
`x ~ 1.1e44` where `w**3` merely underflows; one rounding per term instead of a power plus a
division; and the small corrections are accumulated onto the large leading term last, so the
additions are "big plus tiny" rather than cancelling.

`tail_amplitude` raises rather than returning `nan` when `1 + r' <= 0` (deviation 4);
`tail_crossover` raises when the budget cannot be met below `x_max`, and separately when the
omitted-term estimator vanishes identically at an isolated order (deviation 5).

### `LiouvilleGreen/tests/test_bessel_tail.py` (new, 19 tests, 0.22 s)

Scored entirely against `LiouvilleGreen.tests.bessel_reference` (prompt 01). Tier per order,
chosen from measurement rather than preference: `exact` for `nu` in `{0.5, 1.5, 2.5}`, `scipy`
for `{1.75, 20.5}`, `mpmath` for `{100.5, 1000.5}` — see "Verification performed" for why the
last is not optional. Two module-level helpers, `reference_residual(nu, x)` and
`reference_a_and_theta_deriv(nu, x)`, recover the reference residual **on its correct branch**
using an exact 50-digit reduction of `x + c_nu` mod `2 pi`; a double-precision reduction errs by
`~4e-17 x`, which is 8e-12 at `x = 2e5` and would swamp the three-term series error.

Tests, in the order of the prompt's §4 plus four of my own:

| test | what it pins |
|---|---|
| `test_half_order_is_identically_exact` | §4.1 — `r == 0`, `a == 1`, exact equality, 7 values of `x`, 1–3 terms |
| `test_half_order_crossover_is_the_domain_lower_bound` | §3.2 — `x_star == construction_min_x(1/2)`, `binding == "domain"` |
| `test_reference_helper_agrees_with_cached_corners` | validates this module's own reference machinery against prompt 01's cached corners at `100 nu`, every order |
| `test_series_accuracy_against_references` | §4.2 — `|dr|` and `|da/a|` at `50/100/200 nu`, 2 and 3 terms, four orders |
| `test_first_omitted_term_estimates_the_true_error` | the estimator tracks the truth within 2x plus a 1e-15 reference floor |
| `test_wronskian_holds_by_construction` | §4.3 first half — `a^2 (1 + r') = 1` to 4e-16 |
| `test_theta_deriv_against_the_independent_oracle` | §4.3 second half — `a^-2` vs `(2/pi)/(x(J^2+Y^2))` to 1e-11 at `x >= 100 nu` |
| `test_log_derivative_is_x_times_the_derivative` | `dr/dlog x = x r'` |
| `test_crossover_is_a_real_remainder_test` | §4.4 — bounds on `x_star`, omitted term below budget, **measured** `|dr|` and `|da/a|` below budget |
| `test_crossover_phase_condition_binds` | §3.2 — `binding == "phase"` at every order and budget |
| `test_crossover_is_monotone_in_the_budget` | §4.4 — five budgets, non-increasing |
| `test_crossover_raises_when_the_budget_is_unreachable` | §4.5 — message names `nu`, both budgets and `x_max` |
| `test_estimator_vanishes_at_an_isolated_order` | deviation 5 |
| `test_amplitude_fails_loudly_below_the_series_region` | deviation 4 |
| `test_residual_is_not_reduced_mod_two_pi` | §4.6 — `r(100 nu) = 5.002540` at `nu = 1000.5`, `r > pi` at both crossovers |
| `test_module_does_not_import_a_scipy_bessel_routine` | §6 — subprocess check on `sys.modules` |
| `test_vectorized_and_scalar_agree` | §3.1 |
| `test_invalid_arguments_raise` | `n_terms`, `x <= 0`, non-positive budgets, `safety` |
| `test_coefficients_match_the_published_series` | deviation 1 — pins 5120 and 229376 so nobody "corrects" them back to the plan text |

### Campaign bookkeeping

`IMPLEMENTATION_STATE.md` — prompt 03's row, M6 (⬜ → 🟡, 05 still to consume it), M4 and M5
(unchanged 🟡; the tail is the mechanism by which both are avoided, but nothing consumes it yet),
a new §3 issue `[03-draft-plan-tail-coefficient-wrong]`, and standing note 14.
`docs/OPEN_ISSUES.md` — one new row in §4, count 22 → 23.

---

## Deviations from the prompt

### 1. The third series coefficient in the prompt is wrong by a factor of three — `STRUCTURALLY REQUIRED`

**What the prompt assumed.** §2 and the §5 reproduction script both give

    t5 = (mu - 1)(mu^2 - 114 mu + 1073) / 15360

as the third term of DLMF 10.18.18 in this repository's convention. `DRAFT-PLAN.md` §7.2 prints
the same denominator.

**What is actually there.** The correct denominator is **5120**. Determined by fitting, not
guessed: with the (correct) first two terms subtracted from a 120-digit `mpmath` residual on its
resolved branch, `(r_true - two-term series) * x^5` converges to

| nu | limit of `(r_true - r_2) x^5` | numerator `(mu-1)(mu^2-114mu+1073)` | implied denominator |
|---|---:|---:|---:|
| 3/2 | 0.199999975 (at `x = 2400`) | 1024 | 5120.00 |
| 5/2 | −5.39999928 (at `x = 4000`) | −27648 | 5120.00 |
| 20.5 | 864675.13 (at `x = 32800`) | 4427136000 | 5120.00 |

and `4427136000 / 5120 = 864675.0` exactly. The value is confirmed structurally by Abramowitz &
Stegun 9.2.29's grouping, in which the terms are `(mu-1)/(8x)`, `4(mu-1)(mu-25)/(3(8x)^3)`,
`32(mu-1)(mu^2-114mu+1073)/(5(8x)^5)`: `4/(3·8^3) = 1/384` reproduces the prompt's *second*
denominator exactly, and `32/(5·8^5) = 1/5120` is the third. So the prompt's 384 is right and its
15360 is the same coefficient with the wrong power of 8 folded in.

**What was done instead.** 5120 shipped. The symptom of the prompt's value, had it been used, is
unmistakable and was measured before the correction: the "three-term" series would have removed
only one third of the two-term error (`|dr|` 1.769e-10 → 1.179e-10 at `nu = 5/2, x = 125`,
instead of 1.769e-10 → 2.42e-14), and the crossover would have been sized by an estimator that
does not describe the remainder.

**The fourth coefficient was obtained the same way**, so that the prompt's preferred
"three terms plus a fourth as the estimator" could be delivered rather than its fallback: with
the corrected three-term series subtracted, `(r_true - r_3) x^7` converges to −0.14285721 (nu=3/2),
11.57142857 (5/2) and 142855744 (20.5) against numerators −32768, 2654208 and 32767657574400,
i.e. denominator **229376** in all three cases, `= 7·8^7/64` exactly, matching A&S 9.2.29's
`64(mu-1)(5mu^3-1535mu^2+54703mu-375733)/(7(8x)^7)`. Agreement to eight significant figures at
three orders, with the residual drift in the fitted constant behaving as the next term of the
expansion should, is not a guess; and `test_coefficients_match_the_published_series` pins both
denominators against the `(8x)^k` form so a later reader cannot silently "restore" the plan's
number.

**Not fixed here:** `DRAFT-PLAN.md` is deliberately not edited by this campaign (README §5 item
9, `RECONCILIATION.md` §4), so the plan text still prints 15360. Board issue
`[03-draft-plan-tail-coefficient-wrong]` hands the correction to prompt 09, which is where the
campaign records plan-versus-tree corrections in `docs/`. This is a *new* conflict of exactly the
kind README §5 item 9 asks to be recorded; it is not load-bearing on any design decision —
one series still supplies both the phase and the amplitude, the Wronskian amplitude is unchanged,
the two-region structure is unchanged — only on the arithmetic inside it.

### 2. The `[50 nu, 4800 nu]` bound on `x_star` in §4.4 cannot hold, and was replaced — `STRUCTURALLY REQUIRED`

**What the prompt assumed.** §3.2's sanity anchor and §4.4's assertion both come from
`DRAFT-PLAN.md` §5.3, which measures **two** terms reaching machine precision for
`x ~ 50–100 nu`, and conclude that `x_star` should land in `[50 nu, 4800 nu]`.

**What is actually there.** The prompt also instructs (§3.1) that *three* terms ship, and
(§3.2) that `x_star` be the **smallest** admissible `x`, on the explicit reasoning that every
e-fold below it must be sampled and branch-tracked. Those three instructions are jointly
unsatisfiable: a third term buys two to four orders, so the smallest admissible `x` at the
campaign's budgets is `4.4 nu` to `58 nu`, below `50 nu` at eleven of the twelve non-degenerate
(order, budget) pairs. Keeping the `50 nu` floor would mean pushing `x_star` outwards
*past* what the remainder test requires, which is precisely the "safe large value" §3.2 forbids.

**What was done instead.** `test_crossover_is_a_real_remainder_test` asserts

    construction_min_x(nu) <= x_star <= 100 nu        (and <= 4800 nu)

instead. The upper bounds are the plan's nominal crossover and the top of §5.3's safe matching
window respectively; both hold at every order and budget with margin. This keeps everything the
window bound was there to protect — the tail is never asked to start before the domain does, and
never so far out that the near region becomes unaffordable or the §5.2 cancellation floor matters
— while allowing the third term to do the job it was implemented for. The measured
`|delta r|`-below-budget assertion, which is the substance of §4.4, is unchanged and is what
actually certifies each `x_star`.

### 3. A safety factor of 4 on both budgets — `IMPLEMENTATION CHOICE`

**The prompt left open** how strictly "the first omitted term is below `phase_atol`" should be
read. Read as equality-at-the-crossover, the measured error at `x_star` comes out at 1.01–1.02
times the budget for the `1e-6` row (2.53e-7 at `nu = 20.5`, 2.52e-7 at 100.5, 2.51e-7 at 1000.5
against a `2.5e-7` budget) — i.e. §4.4's "the measured `|delta r|` at `x_star` is below the
budget" fails, narrowly, at three orders. That is not a defect in the estimator; it is what an
asymptotic remainder does, since the first omitted term is an *estimator* and the terms beyond it
add a little more.

**Alternatives considered.** (a) Assert the measured error against `1.05 x budget` — rejected: it
weakens the acceptance test rather than the construction, and the number 1.05 would then be load
bearing at prompt 05 with nothing behind it. (b) Ship four terms and estimate with a fifth —
rejected: it needs a fifth DLMF coefficient I would be taking on trust, and the prompt's "do not
guess a coefficient" applies. (c) Require the first omitted term below `safety * budget` with
`safety = 0.25`, so that the *measured* error clears the budget by a factor of about four.

**Chosen (c)**, exposed as `DEFAULT_CROSSOVER_SAFETY` and as a keyword of `tail_crossover`, so a
caller who disagrees can set `safety=1.0` and get the earlier crossover. The price is
`4^(1/7) = 1.219` in `x_star`, i.e. **0.20 of an e-fold** of extra near region per order — for
instance `nu = 1000.5` at `1e-11` goes from 3.86 to 4.06 e-folds of sampled domain. That is a
cheap price for turning "measured error 1.02x the budget" into "measured error 0.25x the budget",
and it is the *measured* error, not the estimator, that prompt 05 and prompt 08 will quote.

A later reader who wants the tail as wide as possible should note that the four is not sacred:
the measured ratio `|delta r| / first omitted term` is 1.000 to three figures everywhere it is
above the reference floor (table below), so `safety = 0.5` would also work and would cost
0.10 e-folds instead of 0.20.

### 4. `tail_amplitude` raises instead of returning `nan` when `1 + r' <= 0` — `IMPLEMENTATION CHOICE`

**The prompt did not specify** the behaviour of `a = (1 + r')^(-1/2)` when the base is
non-positive, which happens once the asymptotic series diverges. NumPy's answer is a silent
`nan`, and a silent `nan` reaching an interpolant is the exact failure mode README §2 (e) and
`DRAFT-PLAN.md` §4.4 exist to remove, so the module raises `ValueError` naming `nu`, the value of
`1 + r'` and the number of terms.

Worth recording where the guard actually bites, because it is further out than I expected and the
answer bears on prompt 05: **it does not bite anywhere in the supported domain**. At leading
order `r' -> -(mu-1)/(8x^2)`, so even at the turning point `x = nu` the leading contribution is
only `-1/2`, and the three-term `1 + r'` is still 0.31 at `nu = 1000.5, x = nu`. It takes an
argument well below the domain lower bound — `0.3 nu` at that order — to drive it negative. So
this is a guard against a caller handing the tail a near-region argument, not against anything
the two-region construction does on its own. Prompt 05 should not rely on it as the region check.

### 5. `tail_crossover` refuses when the omitted-term estimator vanishes at an isolated order — `IMPLEMENTATION CHOICE`

**The prompt asked** for `nu = 1/2` to be special-cased "explicitly on `mu - 1 = 0` rather than
relying on the numeric test". Implementing that literally exposed a neighbouring case the prompt
does not mention: at `nu = 5/2`, `mu - 25 = 0` exactly, so the **second** coefficient vanishes
identically while the series does not. A one-term crossover at that order would therefore be
sized by an estimator of zero and would return the domain lower bound — the `nu = 1/2` answer, at
an order where it is wrong by ten orders of magnitude (`|delta r|` with one term at
`nu = 5/2, x = 50 nu` is 1.77e-10, not 0).

The degenerate branch is therefore keyed on `4 nu^2 - 1 == 0` and nothing else, and a zero
estimator that is *not* that case raises with a message saying so. At the shipped default of
three terms the estimator is `c3`, whose polynomial factor has no root at any order the campaign
uses, so this path is unreachable in normal use; it is a guard against a future `n_terms`
change. `test_estimator_vanishes_at_an_isolated_order` covers both halves.

### 6. Public names beyond the prompt's list — `IMPLEMENTATION CHOICE`

`tail_series_coefficients`, `construction_min_x` and `tail_crossover_max_x` are exported in
addition to the five evaluators and `tail_crossover` the prompt names.

`construction_min_x` was needed because §3.2 requires `nu = 1/2` to return "the domain lower
bound" and the module had to have a definition of it; it duplicates
`bessel_reference.construction_min_x` under **the same name** rather than a new one, deliberately,
so prompt 05 cannot end up with two subtly different lower bounds. (It is a duplication, not an
import: production code must not depend on a test module, and `bessel_reference` itself duplicates
the same three lines from `bessel_phase` for the mirror-image reason.) `tail_series_coefficients`
exists so that the coefficient test can pin the numbers of deviation 1 without reaching into a
private name. `tail_crossover_max_x` exposes the default ceiling so a caller can ask for it
before calling.

---

## Verification performed

Everything below was run, from the repository root, with `PYTHONPATH=. ./venv/bin/python`.
Environment: Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2, mpmath 1.3.0, Darwin 25.5.0 (arm64).

### The prompt's §5 reproduction, and why it was extended

Run first, as instructed. It reproduces `RECONCILIATION.md` §1's amplitude column exactly
(`|da/a|` with two terms: 3.5e-12 at `nu = 5/2, x = 50 nu`, 1.9e-12 at 20.5, 2.0e-12 at 100.5;
5.5e-14, 2.9e-14, 3.1e-14 at `100 nu`). As the prompt notes, it checks the amplitude only. It was
extended to check `r` as well, with the branch resolved against the series prediction rather than
folded, which is what exposed deviation 1: with the prompt's `15360` the third term removed only
a third of the two-term error at every order tested, and the fitted coefficient came out at
exactly three times the prompt's value.

### Test suite

```
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_tail -v
    Ran 19 tests in 0.224s -- OK
```

Per-module runs of the rest of `LiouvilleGreen/tests`, as standing note 13 directs (the discovery
run is dominated by `test_3bessel_analytic`, which pre-dates this campaign and did not finish in
15 minutes on prompt 01's baseline machine):

| module | result |
|---|---|
| `test_bessel_phase` | **Ran 4 tests in 0.202s — OK** (including `test_phase_derivative`, the standing 1e-6 gate) |
| `test_bessel_reference` | Ran 12 tests in 0.369s — OK |
| `test_range_reduce` | Ran 4 tests in 0.000s — OK |
| `test_scipy_bessel_domain` | Ran 7 tests in 0.032s — OK |
| `test_three_bessel` | Ran 2 tests in 7.208s — OK |
| `test_3bessel_analytic` | **not completed** — still running, with no output, after 9 minutes; abandoned. Consistent with prompt 01's baseline (">15 min"). Pre-existing and provably unaffected: see below |

`./venv/bin/python -m black --check LiouvilleGreen/` — 18 files unchanged, clean.

**The prompt's §6 asks for the full `discover -s LiouvilleGreen/tests -t .` run to pass, and that
was not done.** It cannot be, on this tree: `test_3bessel_analytic` did not finish in 15 minutes on
prompt 01's baseline machine, and a bounded re-run for this commit was still running with no output
after 9 minutes and was abandoned. What was run instead is the per-module substitute standing note
13 prescribes, over the five modules that do finish — all OK. This is a reasoned substitution, not
a measurement: I did **not** observe `test_3bessel_analytic` pass. The reasoning is that
`bessel_tail` is a new module and no existing file imports it — `grep -rn "bessel_tail"
--include='*.py' .` returns the module, its own test, and one *docstring* mention in
`test_scipy_bessel_domain.py:9`, and nothing else — while `LiouvilleGreen/bessel_phase.py` and
every other pre-existing file are untouched, so no existing test's behaviour can have changed.

`git diff HEAD~1 --stat` touches only: `LiouvilleGreen/bessel_tail.py` (new),
`LiouvilleGreen/tests/test_bessel_tail.py` (new), this log,
`prompts/transfer-remedial/IMPLEMENTATION_STATE.md`, `docs/OPEN_ISSUES.md`.
`LiouvilleGreen/bessel_phase.py` is untouched, and nothing imports `bessel_tail` yet.

### Reference tier: `jv`/`yv` are not good enough above `nu = 20.5` here

Measured before choosing tiers, against 50-digit `mpmath` on the same `(nu, x)`:

| nu | max `|delta r|` (scipy vs mpmath) | max `|delta a/a|` |
|---|---:|---:|
| 0.5, 1.5, 1.75, 2.5 | 2.9e-15 | 2.2e-15 |
| 20.5 | 7.6e-15 | 9.0e-15 |
| 100.5 | **1.2e-12** | **7.5e-13** |
| 1000.5 | **8.4e-12** | **1.7e-11** |

over `x` in `[5 nu, 100 nu]`. The three-term series error at those orders is 7.1e-13 (100.5) and
7.1e-12 (1000.5) at `50 nu`, i.e. *below* the SciPy floor, so scoring the series against `jv`/`yv`
there would be measuring Amos. This is well inside `scipy_reference_max_x(nu)`, which is a
boundary against silent catastrophic failure and says nothing about the 1e-12 regime; the tier
choice is recorded in the test module's docstring. `mpmath` at these points costs under 10 ms
each, because they are far from the turning point.

`test_reference_helper_agrees_with_cached_corners` closes the loop: at `x = 100 nu`, every order,
the residual this module's helper reconstructs agrees with prompt 01's independently branch-walked
cached corner to **≤ 3.8e-12** (worst at `nu = 1000.5`), the amplitude to ≤ 1.7e-11 and `theta'`
to ≤ 3.4e-11 — with the bound asserted at 1e-11/1e-13/1e-13 for the mpmath-tier orders where the
agreement is at the 1e-16 level.

### §4.2 — series accuracy, measured

`|delta r|` against the references. Two-term column reproduces `RECONCILIATION.md` §1 (1.77e-10,
7.64e-10, 4.01e-9 at `50 nu`) to three figures.

| nu | x/nu | x | `|dr|` n=2 | `|dr|` n=3 | `|da/a|` n=2 | `|da/a|` n=3 |
|---|---:|---:|---:|---:|---:|---:|
| 5/2 | 50 | 125 | 1.769e-10 | 2.419e-14 | 3.539e-12 | 8.882e-16 |
| 5/2 | 100 | 250 | 5.529e-12 | 1.596e-16 | 5.551e-14 | 0 |
| 5/2 | 200 | 500 | 1.728e-13 | 1.301e-17 | 6.661e-16 | 2.220e-16 |
| 20.5 | 50 | 1025 | 7.644e-10 | 1.203e-13 | 1.865e-12 | 4.441e-16 |
| 20.5 | 100 | 2050 | 2.388e-11 | 9.298e-16 | 2.909e-14 | 0 |
| 20.5 | 200 | 4100 | 7.461e-13 | 2.567e-16 | 6.661e-16 | 2.220e-16 |
| 100.5 | 50 | 5025 | 4.009e-09 | 7.125e-13 | 1.995e-12 | 4.441e-16 |
| 100.5 | 100 | 10050 | 1.253e-10 | 5.884e-15 | 3.131e-14 | 0 |
| 100.5 | 200 | 20100 | 3.915e-12 | 5.551e-17 | 4.441e-16 | 0 |
| 1000.5 | 50 | 50025 | 4.003e-08 | 7.146e-12 | 2.001e-12 | 4.441e-16 |
| 1000.5 | 100 | 100050 | 1.251e-09 | 5.596e-14 | 3.131e-14 | 0 |
| 1000.5 | 200 | 200100 | 3.908e-11 | 4.441e-16 | 4.441e-16 | 0 |

The `|da/a|` column is two to four orders below the corresponding `|delta r|` at every point,
which is §3.2's "the amplitude condition is weaker by a factor `~2x`" made concrete.

**Estimator quality**, the basis of the remainder test: `|delta r| / first omitted term` is
**1.000** to three figures at every point where the truth is above the double-precision reference
floor (four orders × three multiples of `nu` × 1–3 terms = 36 points, of which one — `nu = 5/2`
with one term — has an identically vanishing estimator and is covered by deviation 5 instead).
The three points where the ratio is not 1.000 — 8.8, 35.0 and 1.28 — all have a true error at or
below 2.6e-16, i.e. the reference floor, not the series.
`test_first_omitted_term_estimates_the_true_error` asserts `error < 2 * estimate + 1e-15`.

### §4.3 — the Wronskian, both halves

`|a^2 (1 + r') - 1| <= 4e-16` at every order in `{1/2, 3/2, 7/4, 5/2, 20.5, 100.5, 1000.5}`, every
`n_terms` in `{1, 2, 3}`, at `x/max(nu,1)` in `{50, 100, 1000}`. Tautological by construction, as
the prompt says; asserted as a differentiation guard.

`theta' = a^-2` against the independent oracle `(2/pi)/(x(J^2+Y^2))` at `x >= 100 nu`: required
`1e-11`, **measured worst case 1.110e-15**, at `nu = 7/4, x = 175`. Every other point is
0, 2.220e-16 or 4.441e-16, i.e. at the double-precision floor of the reference itself.

### §4.4 / §7 — the `x_star` table

Shipped defaults: three terms, estimator `c3`, `safety = 0.25`, `x_max = 4800 max(nu, 1)`.
`x_0 = construction_min_x(nu)`. "e-folds" is `log(x_star / x_0)`, the sampled span prompt 04 has to
cover. `|dr|` and `|da/a|` are **measured against the references at `x_star`**, not estimated.

| nu | budget | `x_star` | `x_star/nu` | first omitted at `x_star` | measured `|dr|` | measured `|da/a|` | `x_amplitude` | binding | e-folds |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1/2 | 1e-11 | 1.0e-05 | — | 0 | 0 | 0 | 1.0e-05 | domain | 0.000 |
| 1/2 | 1e-06 | 1.0e-05 | — | 0 | 0 | 0 | 1.0e-05 | domain | 0.000 |
| 3/2 | 1e-11 | 34.4119 | 22.941 | 2.500e-12 | 2.498e-12 | 2.545e-13 | 25.860 | phase | 3.192 |
| 3/2 | 1e-06 | 6.6439 | 4.429 | 2.500e-07 | 2.457e-07 | 1.317e-07 | 6.1324 | phase | 1.547 |
| 7/4 | 1e-11 | 54.5456 | 31.169 | 2.500e-12 | 2.491e-12 | 1.601e-13 | 38.697 | phase | 3.482 |
| 7/4 | 1e-06 | 10.5311 | 6.018 | 2.500e-07 | 2.292e-07 | 7.541e-08 | 9.1764 | phase | 1.837 |
| 5/2 | 1e-11 | 64.4688 | 25.788 | 2.500e-12 | 2.500e-12 | 1.359e-13 | 44.791 | phase | 3.270 |
| 5/2 | 1e-06 | 12.4470 | 4.979 | 2.500e-07 | 2.499e-07 | 7.166e-08 | 10.622 | phase | 1.626 |
| 20.5 | 1e-11 | 664.389 | 32.409 | 2.500e-12 | 2.501e-12 | 1.310e-14 | 344.85 | phase | 3.479 |
| 20.5 | 1e-06 | 128.273 | 6.257 | 2.500e-07 | 2.527e-07 | 7.007e-09 | 81.776 | phase | 1.834 |
| 100.5 | 1e-11 | 4199.98 | 41.791 | 2.500e-12 | 2.501e-12 | 2.220e-15 | 1731.2 | phase | 3.733 |
| 100.5 | 1e-06 | 810.889 | 8.069 | 2.500e-07 | 2.521e-07 | 1.099e-09 | 410.54 | phase | 2.088 |
| 1000.5 | 1e-11 | 58122.8 | 58.094 | 2.500e-12 | 2.499e-12 | 0 | 17251 | phase | 4.062 |
| 1000.5 | 1e-06 | 11221.8 | 11.216 | 2.500e-07 | 2.511e-07 | 7.872e-11 | 4090.8 | phase | 2.417 |

Every row: `x_star <= 100 nu` (deviation 2), measured `|dr|` and `|da/a|` a factor ~4 below the
budget, and **the phase condition binds at every order and both budgets** — `x_phase / x_amplitude`
runs from 1.08 (`nu = 3/2`, 1e-6) to 3.37 (`nu = 1000.5`, 1e-11). `x_star` is monotone
non-increasing across five budgets `{1e-13, 1e-11, 1e-9, 1e-6, 1e-4}` at every order.

Note the shape of the `x_star/nu` column: it **grows** with order, 4.4 → 11.2 at `1e-6` and
22.9 → 58.1 at `1e-11`, because the coefficient grows like `nu^8` while the power of `x` is fixed
at 7. This is the sense in which "a fixed multiple of `nu`" would have been wrong, and it is
what `DRAFT-PLAN.md` §7.2 is guarding against.

### §4.5 — the failure path

`tail_crossover(1000.5, 1e-30, 1e-30, x_max=1e6)` raises `ValueError`:

    tail_crossover: no crossover below x_max=1e+06 for nu=1000.5 at phase_atol=1e-30,
    amplitude_rtol=1e-30 with 3 series term(s) and safety=0.25. The phase condition alone needs
    x >= 3.01046e+07 and the amplitude condition x >= 4.09083e+06; the best first omitted term
    available below the ceiling is 5.60229e-21 at x=1e+06. Widen x_max, loosen the budget, or
    extend the series.

`tail_crossover(1000.5, 1e-40, 1e-40)` (default ceiling 4.8024e6) also raises;
`tail_crossover(1000.5, 1e-30, 1e-30, x_max=1e9)` succeeds at `x_star = 3.0105e7`.

### §4.6 — branch sanity

`tail_residual(1000.5, 100050)` = **5.002540**, reproducing `RECONCILIATION.md` C2's 5.002540 to
six decimal places, and greater than `pi`. At the two crossovers `r(x_star)` is **8.611292**
(1e-11) and **44.630484** (1e-6), both above `pi`; and `r(5 nu) = 100.387 > r(100 nu)`, so the
residual increases towards the turning point as C2's table requires. Any fold into `(-pi, pi]`
fails all four.

### §4.1 — the free exactness test

`r`, `r'`, `dr/dlog x` and the first omitted term are **exactly** `0.0` and `a` is **exactly**
`1.0` at `nu = 1/2`, for `x` in `{1e-3, 1, 10, 1e3, 1e7, 1e12, 1e15}` and 1, 2 and 3 terms
(7 × 3 × 5 = 105 exact-equality assertions).
`tail_series_coefficients(0.5) == (0.0, 0.0, 0.0, 0.0)`.
`4 * 0.5 * 0.5 == 1.0` exactly in binary, so this is a property of the arithmetic and not a
tolerance.

---

## Observations not acted on

1. **`DRAFT-PLAN.md` §7.2 still prints the wrong third coefficient**, as does this prompt.
   Deviation 1; board issue `[03-draft-plan-tail-coefficient-wrong]`; the plan is not edited by
   this campaign, and prompt 09 owns recording plan corrections in `docs/`. Prompt 04 and prompt
   05 must take the coefficients from `bessel_tail.tail_series_coefficients`, never from the plan
   text.

2. **`bessel_reference.tail_residual_series` is a second, independent two-term implementation of
   the same series** (`bessel_reference.py:201-218`). It is correct — 8 and 384 — and it is used
   only to anchor the residual branch when regenerating the cached table, so it does not need the
   third term. But it is now a duplicate of `bessel_tail.tail_residual(nu, x, n_terms=2)`, and a
   future change to one will not reach the other. Not consolidated here: `bessel_reference` is a
   test module and `bessel_tail` is production, the dependency would have to run the wrong way,
   and prompt 01 owns that file. Worth a note in prompt 08 or 09 if either revisits the harness.

3. **The `tail_amplitude` guard is unreachable inside the supported domain.** Deviation 4 records
   the measurement. If prompt 05 wants a cheap "is this argument in the tail" assertion it needs
   its own, comparing against `x_star`; `1 + r' > 0` will not serve.

4. **`test_3bessel_analytic.py` remains the reason the `LiouvilleGreen/tests` discovery run is
   unusable**, and therefore the reason prompt 03 could not satisfy its own §6 acceptance
   criterion literally. Prompt 01's baseline records that it did not finish in 15 minutes; a
   bounded re-run for this commit produced no output in 9 minutes and was abandoned. Nothing in
   this commit can affect it (see "Verification performed"), so the five modules that do finish
   are the complete set of what could regress. Do not read the discovery run's duration as a
   regression (standing note 13). Whoever eventually fixes this — its per-case `bessel_phase`
   rebuild at `:47-56`, `:504-512`, `:572-604` — will be doing every later prompt in this campaign
   a favour; prompt 08 has to touch that file's tolerances and will meet the same wall.

5. **`safety = 0.25` is conservative by a factor of two.** Deviation 3 records that the measured
   estimator ratio is 1.000, so `safety = 0.5` would give the same acceptance result at half the
   e-fold cost. Left at 0.25 because the cost is 0.10 of an e-fold either way and prompt 04's
   sampling budget is not yet known. If prompt 04 finds the near region expensive at
   `nu = 1000.5`, this is the cheapest knob in the design.

6. **The series is not evaluated with a term-count that adapts to `x`.** A caller deep in the tail
   (`x = 1e12`) pays for three terms where one is already at the double-precision floor. The cost
   is three multiplies and it is not worth a branch, but a future vectorized consumer evaluating
   millions of points might disagree.

---

## State handed to the next prompt

### Import path and public API

    from LiouvilleGreen import bessel_tail

```python
# --- constants -------------------------------------------------------------------------------
TAIL_SERIES_MAX_TERMS = 3            # terms the module will evaluate
TAIL_SERIES_COEFFICIENT_COUNT = 4    # ... plus the omitted-term estimator, c3
DEFAULT_TAIL_TERMS = 3
DEFAULT_CROSSOVER_SAFETY = 0.25
SAFE_WINDOW_MAX_FACTOR = 4800.0
DEGENERATE_MIN_X = 1e-5

# --- geometry --------------------------------------------------------------------------------
construction_min_x(nu: float) -> float          # sqrt(nu^2 - 1/4), or 1e-5 for nu <= 1/2
tail_crossover_max_x(nu: float) -> float        # 4800 * max(nu, 1)
tail_series_coefficients(nu: float) -> Tuple[float, float, float, float]   # (c0, c1, c2, c3)

# --- the series; each accepts a scalar or an array of x and returns the matching type ---------
tail_residual(nu, x, n_terms=3)            # r_nu(x) = theta_nu(x) - x - c_nu
tail_residual_deriv(nu, x, n_terms=3)      # dr/dx
tail_residual_log_deriv(nu, x, n_terms=3)  # dr/d(log x) = x r'
tail_amplitude(nu, x, n_terms=3)           # a_nu = (1 + r')^(-1/2); A_nu = sqrt(2/(pi x)) a_nu
tail_first_omitted_term(nu, x, n_terms=3)  # |c_n| / x^(2n+1)

# --- the remainder test ------------------------------------------------------------------------
tail_crossover(nu, phase_atol, amplitude_rtol, n_terms=3, x_max=None,
               safety=DEFAULT_CROSSOVER_SAFETY) -> TailCrossover
```

`TailCrossover` is a `NamedTuple`:

| field | meaning |
|---|---|
| `nu` | the order the test was run for |
| `x_star` | the crossover. Use the tail for `x >= x_star`; everything below is the near region |
| `first_omitted_at_x_star` | `tail_first_omitted_term(nu, x_star, terms_used)` — the estimated remainder. Compare against `phase_atol`, **not** against `safety * phase_atol` |
| `terms_used` | the `n_terms` the crossover was sized for. Pass the same value to the evaluators or the result means nothing |
| `phase_atol`, `amplitude_rtol` | the budgets as requested |
| `safety` | the margin factor applied to both (default 0.25) |
| `x_phase`, `x_amplitude` | the smallest `x` satisfying each condition alone |
| `amplitude_error_at_x_star` | estimated `|delta a / a|` at `x_star` |
| `binding` | `"phase"`, `"amplitude"` or `"domain"` — which condition set `x_star` |

**Raising behaviour**, all `ValueError`: `x <= 0`; `n_terms` outside `[1, 3]`; a non-positive
budget or `safety` outside `(0, 1]`; `1 + r' <= 0` in `tail_amplitude`; no crossover below
`x_max`; and a first-omitted-term coefficient that vanishes at an order that is not `nu = 1/2`.

### Series terms shipped, and the estimator

**Three terms are summed** (`c0/x`, `c1/x^3`, `c2/x^5`); the **fourth coefficient `c3`** is the
omitted-term estimator, so the prompt's preferred configuration was delivered rather than its
two-plus-third fallback. The coefficients are

    c0 = (mu-1)/8
    c1 = (mu-1)(mu-25)/384
    c2 = (mu-1)(mu^2-114mu+1073)/5120          <-- 5120, NOT the plan's 15360
    c3 = (mu-1)(5mu^3-1535mu^2+54703mu-375733)/229376

with `mu = 4 nu^2`. **Take these from `tail_series_coefficients`, never from `DRAFT-PLAN.md` §7.2
or from prompt 03 §2, both of which print 15360 for the third denominator.** Deviation 1 has the
numerical determination; `test_coefficients_match_the_published_series` pins them.

### The `x_star` table — prompt 04 sizes its sampled domain from this

Shipped defaults (`n_terms=3`, `safety=0.25`). `x_0 = construction_min_x(nu)`; "e-folds" is
`log(x_star/x_0)`, the span the near-region sampler has to cover and branch-track.

| nu | budget | `x_star` | first omitted at `x_star` | `x_star/nu` | measured `|dr|` | measured `|da/a|` | e-folds |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/2 | 1e-11 | 1.0e-05 (`= x_0`) | 0 | — | 0 | 0 | 0.000 |
| 1/2 | 1e-06 | 1.0e-05 (`= x_0`) | 0 | — | 0 | 0 | 0.000 |
| 3/2 | 1e-11 | 34.4119 | 2.500e-12 | 22.941 | 2.498e-12 | 2.545e-13 | 3.192 |
| 3/2 | 1e-06 | 6.6439 | 2.500e-07 | 4.429 | 2.457e-07 | 1.317e-07 | 1.547 |
| 7/4 | 1e-11 | 54.5456 | 2.500e-12 | 31.169 | 2.491e-12 | 1.601e-13 | 3.482 |
| 7/4 | 1e-06 | 10.5311 | 2.500e-07 | 6.018 | 2.292e-07 | 7.541e-08 | 1.837 |
| 5/2 | 1e-11 | 64.4688 | 2.500e-12 | 25.788 | 2.500e-12 | 1.359e-13 | 3.270 |
| 5/2 | 1e-06 | 12.4470 | 2.500e-07 | 4.979 | 2.499e-07 | 7.166e-08 | 1.626 |
| 20.5 | 1e-11 | 664.389 | 2.500e-12 | 32.409 | 2.501e-12 | 1.310e-14 | 3.479 |
| 20.5 | 1e-06 | 128.273 | 2.500e-07 | 6.257 | 2.527e-07 | 7.007e-09 | 1.834 |
| 100.5 | 1e-11 | 4199.98 | 2.500e-12 | 41.791 | 2.501e-12 | 2.220e-15 | 3.733 |
| 100.5 | 1e-06 | 810.889 | 2.500e-07 | 8.069 | 2.521e-07 | 1.099e-09 | 2.088 |
| 1000.5 | 1e-11 | 58122.8 | 2.500e-12 | 58.094 | 2.499e-12 | 0 | 4.062 |
| 1000.5 | 1e-06 | 11221.8 | 2.500e-07 | 11.216 | 2.511e-07 | 7.872e-11 | 2.417 |

Four things prompt 04 and prompt 05 should take from this table:

1. **The sampled span is 3.2–4.1 e-folds at the tight budget**, 1.5–2.4 at the loose one — smaller
   than `DRAFT-PLAN.md` §1's "a fixed ≈4.6 e-folds", because three terms move the crossover in.
2. **`x_star/nu` grows with order** (4.4 → 11.2 and 22.9 → 58.1). A fixed multiple of `nu` would
   be wrong at one end or the other; call `tail_crossover`.
3. **`nu = 1/2` returns `x_star = x_0 = 1e-5`**: the near-region sampler must never run at that
   order, and prompt 05 should assert that it does not.
4. **The phase condition binds everywhere**, by 1.08x to 3.37x in `x`, so prompt 05's crossover
   check on the phase does cover the amplitude — the property `DRAFT-PLAN.md` §7.2 relies on.

### Accuracy at the crossover, and the reference floor prompt 05 will hit

- Measured `|delta r|` at `x_star` is 2.49–2.53e-12 for the 1e-11 budget and 2.29–2.53e-07 for
  the 1e-6 budget, i.e. a factor ~4 inside budget at every order (that is `safety = 0.25` at
  work; the underlying estimator is accurate to 1.000).
- `theta' = a^-2` against `(2/pi)/(x(J^2+Y^2))` at `x >= 100 nu`: worst 1.110e-15 over seven
  orders. The tail is not the limiting factor in anything prompt 05 measures.
- **Above `nu = 20.5`, SciPy `jv`/`yv` are not an adequate reference for this module**, well below
  `scipy_reference_max_x(nu)`: measured floors are 1.2e-12 (`nu = 100.5`) and 8.4e-12
  (`nu = 1000.5`) in `r`, 1.7e-11 in `a`. Prompt 05's crossover-agreement test must use the
  `mpmath` tier or the cached corners at those orders, or it will be scoring Amos. `mpmath` is
  cheap here (<10 ms per point) because `x_star >> nu`.
- Recovering a reference residual needs the mod-`2 pi` reduction of `x + c_nu` done in extended
  precision. `test_bessel_tail.reference_residual(nu, x)` is a working implementation to copy;
  a double-precision reduction errs by `~4e-17 x` and is useless above `x ~ 1e4`.

### Facts prompt 05 can rely on without re-deriving

- `a^2 (1 + r') = 1` to 4e-16 by construction, at every order and term count. It is a tautology;
  the independent checks are `a^-2` against `1 + r_u/x` and against the reference `theta'`
  (README §2 (f)).
- `r` from this module is **never** reduced mod `2 pi`: 5.002540 rad at `nu = 1000.5, x = 100 nu`,
  8.61 rad at that order's tight-budget `x_star`, 570.82 rad at `x_0`.
- Nothing in `bessel_tail` imports `scipy.special`, and a test enforces it. The tail therefore
  carries the supported `x_max` past the `2.5e15` Amos cliff on its own.
- `construction_min_x` in this module and in `bessel_reference` are the same function with the
  same name; keep them that way.
