# Log 02 — Bracket the equality solve

**Prompt:** prompts/background-solver-robustness/02-bracket-the-equality-solve.md
**Commit:** *(this commit)* — Bracket the equality solve and make its guard reachable
**Model:** Claude Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

Four deviations. **D1 is the substantive one and it was escalated before any code was written:**
the prompt's acceptance — "the four equality redshifts, bit-identical, or ≤ 1 ulp" — is not
attainable by any solver, because the residual this method root-finds is a cancellation between two
densities of order $10^{112}$ and its sign change is not localised to a single float. The prompt's
recommended tolerance, `rtol=1e-14` (README §7 **D1**), was measured to break **prompt 01's**
existing test. The run was stopped, the measurement put to the user, and the user's decision of
2026-09-16 is what shipped: the $\sqrt2$ expansion at `rtol=8.9e-16`, Brent's own $4\varepsilon$
floor, with prompt 02 §4's first acceptance row amended to "≤ 4 ulp against prompt 01's reference".

---

## What shipped

**One production method, three new tests, no other production file in the diff.**

- `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, `_find_rho_equality` only
  (`:999-1021` before → `:999-1132` after). **Signature unchanged**; the two call sites at
  `:501-508` and the two prints at `:516-517` are untouched.

  - **Before:** `root = root_scalar(match_rho, x0=init_z, xtol=1e-6, rtol=1e-4)` — an unbracketed
    secant — followed by a `if not root.converged` guard that log 01 §4.1 measured never firing.
  - **After:** the guess is clamped into the $T(z)$ representation's tabulated range and evaluated;
    a bracket is expanded about it multiplicatively in $1+z$ by `BRACKET_EXPANSION_FACTOR =
    sqrt(2.0)`, capped at `BRACKET_EXPANSION_MAX_STEPS = 140` and clamped to
    `self._T_z_spline._min_z`, `self._T_z_spline._max_z`, until the residual changes **sign**;
    then `root_scalar(match_rho, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=8.9e-16)`.
    A failure to bracket raises `_find_rho_equality`'s own `RuntimeError`; the `converged` guard
    is kept and now names the species pair and the bracket.
  - No new module-level or public symbol. The two constants are function-local (D3).
  - The docstring gains a paragraph saying `init_z` is a guess and not an answer.

- `CosmologyModels/tests/test_rho_equality.py` — **prompt 01's four test methods are untouched,
  character for character.** Added: the module-level helper
  `solve_recording_evaluations(model, pair: str, init_z: float) -> (float, list[float])`, the
  constants `DISPLACED_GUESS_FRACTIONS = (0.8, 0.7, 0.5, 0.2)` and `TRUNCATED_MAX_Z = 100.0`, and
  three test methods:
  - `test_a_displaced_guess_now_finds_the_root_instead_of_failing`
  - `test_a_failure_to_bracket_raises_the_methods_own_error`
  - `test_the_bracket_does_not_leave_the_tabulated_range`

- `prompts/background-solver-robustness/IMPLEMENTATION_STATE.md` — board row 02, item rows (b),
  (c), (e), `[00-equality-solve-is-unbracketed-and-loose]` moved to §4, one new §3 issue.
- `docs/OPEN_ISSUES.md` — that row deleted from §1.8, one row added; **count stays 72**.

`T_Z_REPRESENTATION_VERSION` is **6** before and **6** after (`LambdaCDM_GenericEOS.py:419`,
untouched). Nothing in the diff touches the representation, `_rho_fluid`, `_solve_T_z`,
`_temperature_crossing_log1pz`, `_bisect_temperature_crossing_log1pz`, `main.py` or
`ComputeTargets/`.

## The two equality redshifts

All values at 17 significant figures and as `float.hex()`, on `QCD_Cosmology(max_z=1e12)` and the
`PureRadiationEOS` stand-in (`max_z=1e6`), at `Planck2018()` and `Mpc_units()`. "Before" is
`7fdc49b`, log 01's table; "after" is this commit.

| Model | Pair | Before | After | Move |
|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.6689742499498` | `3406.6689742499511` | **+3.0 ulp** |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.30342303299640738` | `0.30342303299640738` | **no bit moved** |
| pure-radiation stand-in | matter = radiation | `3403.1059638279453` | `3403.1059638279457` | **+1.0 ulp** |
| pure-radiation stand-in | matter = $\Lambda$ | `0.30342303299640738` | `0.30342303299640738` | **no bit moved** |

As `float.hex()`, because the decimals differ in their last digit only:

| Quantity | Before | After |
|---|---|---|
| QCD, matter = radiation | `0x1.a9d5683cafacdp+11` | `0x1.a9d5683cafad0p+11` |
| QCD, matter = $\Lambda$ | `0x1.36b4870e4a718p-2` | `0x1.36b4870e4a718p-2` |
| stand-in, matter = radiation | `0x1.a963640e40f2bp+11` | `0x1.a963640e40f2cp+11` |
| stand-in, matter = $\Lambda$ | `0x1.36b4870e4a718p-2` | `0x1.36b4870e4a718p-2` |

**Both $\Lambda$ roots are bit-identical. Both matter–radiation roots moved, by +3 and +1 ulp, and
both moved *onto* prompt 01's independent bracketed reference**, which they previously sat 3 and 1
ulp below. Scored against that reference, the separations go

| Model, pair | Before | After |
|---|---|---|
| QCD, matter = radiation | −3.0 ulp | **0.0 ulp** |
| QCD, matter = $\Lambda$ | −2.0 ulp | −2.0 ulp |
| stand-in, matter = radiation | −1.0 ulp | **0.0 ulp** |
| stand-in, matter = $\Lambda$ | −2.0 ulp | −2.0 ulp |

so the solve is **closer to the reference after the change than before it, on every case that
moved**, and prompt 01's `SOLVE_VS_REFERENCE_ULP = 4` is met with 4 ulp of margin on two of the
four and 2 on the other two. **`SOLVE_VS_REFERENCE_ULP` was not widened** — nor were
`LAMBDA_CLOSED_FORM_ULP` or `MATTER_RADIATION_CLOSED_FORM_ULP`.

**The two printed banner lines are character-identical.** `:.4g` of every value above gives
`3407`, `3403` and `0.3034`; the `CosmologyModels` and `ComputeTargets` suite output at this commit
carries `|  matter-radiation equality at z = 3407` and `|  matter-Lambda equality at z = 0.3034`
exactly as before (board §6 note 5).

**Why "bit-identical" was not attainable, and why 3 ulp is not a loss of accuracy** — see D1.

## Deviations from the prompt

### D1 — `rtol=8.9e-16`, not README §7 D1's `rtol=1e-14`; and the acceptance is 4 ulp against the reference, not bit-identity — **STRUCTURALLY REQUIRED**

**What the prompt assumed.** §2.2: "Ship `xtol=1e-300, rtol=1e-14` (README §7 **D1**), identical to
`_solve_T_z:583`", with the only sanctioned deviation being `scipy` rejecting `xtol=1e-300`. §4's
first row: the four redshifts "**bit-identical**; if not, ≤ 1 ulp". §5's first stop condition:
"Either root moves by more than 1 ulp … stop."

**What is actually there.** Two measured facts, both taken at `7fdc49b` before any production line
was written.

*First, the root is a band a few ulp wide, so no algorithm can be bit-identical to the secant.*
`match_rho` is a difference between two densities of order $10^{112}$; near the root the
cancellation leaves a residual quantised at about $7\times10^{100}$, roughly **0.55 quanta per ulp
of $z$, with ±1 to 2 quanta of evaluation noise**. Stepping float by float through `match_rho` on
`QCD_Cosmology` at matter–radiation equality, relative to the closed form `0x1.a9d5683cafac9p+11`:

| offset | −2 | −1 | **0** | **+1** | +2 | +3 | +4 | +5 | +6 | **+7** | +8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `match_rho` | +1.75e101 | +1.05e101 | +1.40e101 | **0.0** | +7.00e100 | +1.75e101 | +7.00e100 | +1.05e101 | +3.50e100 | **−7.00e100** | −7.00e100 |

The residual is **exactly zero at +1 ulp, non-zero either side of it, and changes sign between +6
and +7** — it is not monotone at this scale. On the matter–$\Lambda$ pair it is **exactly zero
across a five-float plateau** (−2 … +2 ulp of the shipped value) on both models. Bit-identity
between two solvers is therefore not a property this problem has; where a solver stops inside that
band is set by its tolerance.

*Second, `rtol=1e-14` measurably breaks prompt 01's test.* 1e-14 is 75 ulp of slack at
$z\sim3.4\times10^3$, and Brent stops early inside it. Run at `7fdc49b` with the $\sqrt2$
expansion and `rtol=1e-14`:

```
FAIL: test_equality_redshifts_match_a_bracketed_reference
      (model='QCD_Cosmology', pair='matter_radiation')
AssertionError: 3.183231456205249e-12 not less than or equal to np.float64(1.8189894035458565e-12)
: QCD_Cosmology, matter_radiation: the equality redshift has moved away from the bracketed
reference. solve = 3406.668974249948, reference = 3406.668974249951, separation = -7.000 ulp,
budget = 4 ulp
Ran 4 tests in 0.111s
FAILED (failures=1)
```

The full sweep, ulp against the shipped float / ulp against prompt 01's reference, every expansion
straddling at step 1:

| factor | rtol | QCD m=r | QCD m=Λ | rad m=r | rad m=Λ | prompt 01 |
|---|---|---|---|---|---|---|
| 1.01 | 1e-14 | +0 / −3 | +5 / +3 | +0 / −1 | +5 / +3 | passes |
| 1.01 | 1e-15 | +4 / +1 | +0 / −2 | +0 / −1 | +0 / −2 | passes |
| 1.01 | 8.9e-16 | +3 / +0 | +2 / +0 | +0 / −1 | +2 / +0 | passes |
| $\sqrt2$ | 1e-14 | −4 / **−7** | −1 / −3 | −2 / −3 | −1 / −3 | **FAILS** |
| $\sqrt2$ | 1e-15 | +4 / +1 | +0 / −2 | +2 / +1 | +0 / −2 | passes |
| **$\sqrt2$** | **8.9e-16** | **+3 / +0** | **+0 / −2** | **+1 / +0** | **+0 / −2** | **passes** |
| 2 | 1e-14 | −8 / −11 | −1 / −3 | −6 / −7 | −1 / −3 | **FAILS** |
| 2 | 1e-15 | +4 / +1 | +0 / −2 | −2 / −3 | +0 / −2 | passes |
| 2 | 8.9e-16 | +3 / +0 | −2 / −4 | +2 / +1 | −2 / −4 | passes |

**No configuration reaches ≤ 1 ulp against the shipped floats**; the best any of them does is 3
ulp on the QCD matter–radiation root, and the shipped row is one of the two that achieve it.

**What was done instead.** The run was **stopped before any production line was committed** and the
measurement was put to the user, who decided README §7 **D1** on 2026-09-16:

> Ship the √2 expansion at rtol=8.9e-16 (Brent's own 4*eps floor, which is what prompt 01's own
> reference uses) instead of D1's rtol=1e-14, and amend prompt 02 §4's first acceptance row from
> "bit-identical / <= 1 ulp" to "<= 4 ulp against prompt 01's reference", with the cancellation
> measurement as the justification.

That is what shipped, and both load-bearing measurements above were independently reproduced by
the orchestrator before the decision was taken. The value, the competing floor, the reason the
absolute component is disabled and the reason for the floor rather than 1e-14 are all in the
comment at the point of use, to the standard of `:574-582`.

**Why this is not a loss of accuracy.** The two roots that moved moved **onto** the independent
reference, not away from it, and the campaign's own oracle says the direction is right where it can
be checked at all: see D4 and "Observations not acted on" item 1. `_find_rho_equality`'s results
remain printed with `:.4g` and discarded, so no computed quantity in the pipeline moves — README
§0.2's framing is untouched.

### D2 — the three tests are not the three the prompt specifies, because a bracketed solve cannot fail on a displaced guess — **STRUCTURALLY REQUIRED**

**What the prompt assumed.** §3 item 1: call `_find_rho_equality` with the matter–radiation guess
displaced by −50 % and "**Assert `RuntimeError`**". §3 item 2: the same at −80 % "must not produce
a message mentioning `TemperatureRepresentation`, and must not raise `ValueError`". §3 item 3: "with
a guess so bad that no bracket exists", choosing the guess from the measurements.

**What is actually there.** Once the expansion is clamped to the tabulated range and both density
ratios are monotone across it (prompt 01's standing test), **no displacement of the guess can fail
any more** — which is the entire point of the change. Measured on `QCD_Cosmology`,
matter = radiation, at this commit:

| guess | ×0.01 | ×0.2 | ×0.5 | ×0.7 | ×0.8 | ×0.9 | ×0.95 | ×1.2 | ×2 | ×5 | ×100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| root, ulp vs. reference | +1 | +1 | +1 | +1 | −1 | 0 | −1 | −1 | +1 | +1 | +1 |
| `_rho_fluid` evals | 54 | 37 | 18 | 29 | 24 | 14 | 25 | 28 | 34 | 46 | 83 |

Every one of them recovers the root to ±1 ulp, including the three offsets that fail on `7fdc49b`
(−30 % `ValueError`, −50 % and −80 % bounds errors) and the −20 % offset that returns a *wrong*
answer there. §3's items 1 and 2 as written would assert a failure that a correct implementation
does not have, and item 3's "guess so bad that no bracket exists" does not exist.

**What was done instead.** Three tests that make the same three statements against a tree on which
the bracket works:

1. `test_a_displaced_guess_now_finds_the_root_instead_of_failing` — the four fractions
   `(0.8, 0.7, 0.5, 0.2)` of the analytic guess, i.e. exactly audit §3.2's −20 / −30 / −50 / −80 %
   rows, each asserted to recover the bracketed reference within `SOLVE_VS_REFERENCE_ULP`. This is
   §3 item 1 inverted, and it is a **stronger** statement than "it raises its own error".
2. `test_a_failure_to_bracket_raises_the_methods_own_error` — the guard is reached instead by a
   cosmology whose tabulated range **cannot contain** its own equality redshift
   (`TRUNCATED_MAX_Z = 100.0`, ceiling 105.05, root at $z\sim3.4\times10^3$), which is now the only
   way to reach it and is a realistic user error. Asserts `RuntimeError`, `assertIn` on
   `_find_rho_equality`, `matter` and `radiation` — never on the whole message, which is prose —
   and `assertNotIn` on `TemperatureRepresentation`, which is §3 item 2's distinguishing assertion.
3. `test_the_bracket_does_not_leave_the_tabulated_range` — §3 item 3, name kept. Asserted on the
   **arguments** `_rho_fluid` is called with, recorded by shadowing it on the instance, rather than
   on the failure message: from `init_z = 0.01 z_eq` and from `init_z = 10 z_ceil`, every
   evaluation lies in $[-0.24,\,1.05\times10^{12}]$. Asserting the arguments rather than the
   exception makes the test independent of what the representation chooses to do when asked out of
   range.

Alternatives considered for item 2's failure case. (i) A contrived species pair —
`_find_rho_equality("matter", "T")` has no root, since `_rho_fluid` returns `"T"` as a key and
$\rho_m \gg T$ everywhere. Rejected: it abuses the API and would pin behaviour nothing relies on.
(ii) Shrinking `BRACKET_EXPANSION_MAX_STEPS` in the test so the cap binds. Rejected: it tests the
cap, which is a belt-and-braces guard behind the clamp, not the clamp. (iii) The truncated
cosmology, **chosen**: it fails for the physical reason the guard exists to report, it exercises
the clamp at both ends (the failure message quotes `z_lo=-0.24` and `z_hi=105.05`, both the
representation's own bounds), and on `7fdc49b` the same construction raises the
`TemperatureRepresentation` bounds error that the new message replaces, so the test distinguishes
the trees.

### D3 — the two bracket constants are function-local, not module-level — **IMPLEMENTATION CHOICE**

§2.1 item 2 requires "a named constant with a comment saying what range it covers, not a magic
number in a `while`". The prompt also restricts the diff to "`_find_rho_equality` only,
`:999-1021`". Module-level constants would sit with `DEFAULT_T_Z_SPLINE_SAMPLES` and friends at the
top of the file, outside that range. `BRACKET_EXPANSION_FACTOR` and `BRACKET_EXPANSION_MAX_STEPS`
are therefore defined inside the method, in upper case, with the comment §2.1 asks for.

Alternatives. (i) Module scope, matching the file's convention for `DEFAULT_*` — rejected only on
the scope restriction; nothing outside this method reads them, and a later prompt that wants them
public can lift them in one line. (ii) Default keyword arguments on the method — rejected: it
widens the signature, which §2.1's "the signature does not change" forbids. A reader who expects
module constants will find them one screen up from where they are used, which is the cost.

### D4 — the sign test compares signs instead of `f_lo * f_hi <= 0` — **IMPLEMENTATION CHOICE**

`_solve_T_z:568` writes its bracket check as a product, and copying it would have been the
house-style choice. It cannot be copied here: that residual is a difference of temperatures and is
$O(1)$, while this one is a difference of energy densities that reaches ~$10^{183}$ at the top of
the tabulated range at the default `max_z = 1e20`, so the product overflows to `+inf` and a genuine
sign change is silently missed. The shipped test is `f_lo == 0.0 or f_hi == 0.0 or (f_lo < 0.0) !=
(f_hi < 0.0)`, with the reason in a comment beside it. `scipy`'s `brentq` accepts a bracket whose
endpoint residual is exactly zero (it tests `f(a)*f(b) > 0`), so the zero cases are safe to pass
through, and on the matter–$\Lambda$ pair they are not hypothetical — the residual is exactly zero
over five consecutive floats.

## Verification performed

Everything below was **run**. Nothing in this section is reasoning about what would happen.

### 1. The test module, after

```
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
```
→ `Ran 7 tests in 0.406s / OK`. Prompt 01's four pass unchanged. Printed:

```
  equality redshifts against a bracketed brentq reference (17 digits):
             QCD_Cosmology  matter_radiation: solve = 3406.668974249951   reference = 3406.668974249951   (+0.0 ulp)
             QCD_Cosmology     matter_lambda: solve = 0.3034230329964074  reference = 0.3034230329964075  (-2.0 ulp)
   pure-radiation stand-in  matter_radiation: solve = 3403.1059638279457  reference = 3403.1059638279457  (+0.0 ulp)
   pure-radiation stand-in     matter_lambda: solve = 0.3034230329964074  reference = 0.3034230329964075  (-2.0 ulp)

  failure to bracket:
    LambdaCDM_GenericEOS._find_rho_equality: could not bracket the redshift at which rho[matter] =
    rho[radiation]. Expanding multiplicatively in 1+z about the initial guess z=3403.1, clamped to
    z=105.05 (residual 7.9666e+111), reached z_lo=-0.24 (residual 3.0257e+105) and z_hi=105.05
    (residual 7.9666e+111), which have the same sign, so no root is enclosed. The search is clamped
    to the tabulated range of T(z), z in [-0.24, 105.05].
```

The message names the species pair, the guess, the clamped guess, both endpoints of the range
searched and the residual at each, per §2.3.

### 2. The three new tests on `HEAD~1` (`7fdc49b`) — README §2 (e)

Production file reverted to `7fdc49b`, the new tests in place:

```
git checkout HEAD -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
```
→ `Ran 7 tests in 0.142s / FAILED (failures=2, errors=5)`. **All three new tests fail; prompt 01's
four pass**, which is the evidence that the four are the "nothing moved" half and the three are the
"something changed" half.

`test_a_displaced_guess_now_finds_the_root_instead_of_failing`, all four subtests:

```
FAIL (fraction=0.8):
AssertionError: 8.863517450663494e-06 not less than or equal to np.float64(1.8189894035458565e-12)
: a guess displaced to 0.8 x the analytic value (2725.3351793999586) did not recover the
matter/radiation equality redshift: solve = 3406.6689831134686, reference = 3406.668974249951,
separation = +19491081.000 ulp, budget = 4 ulp

ERROR (fraction=0.7):
  File ".../LambdaCDM_GenericEOS.py", line 1013, in _find_rho_equality
    root = root_scalar(match_rho, x0=init_z, xtol=1e-6, rtol=1e-4)
  File ".../LambdaCDM_GenericEOS.py", line 320, in __call__
    log_z = log(1.0 + z)
ValueError: math domain error

ERROR (fraction=0.5):
RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.25719 (min allowed
z=-0.24, recommended limit is z >= -0.2376)

ERROR (fraction=0.2):
RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.38344 (min allowed
z=-0.24, recommended limit is z >= -0.2376)
```

The +19,491,081 ulp on the −20 % row is log 01 §4.1's `converged=True` wrong answer, 2.7e-9
relative, quoted here in the units this campaign scores in.

`test_a_failure_to_bracket_raises_the_methods_own_error`:

```
FAIL: AssertionError: '_find_rho_equality' not found in 'TemperatureRepresentation: evaluated T(z)
out of bounds @ z=3403.1 (max allowed z=105.05, recommended limit is z <= 104)' : a failure to
bracket the equality redshift must name '_find_rho_equality'; the message was:
TemperatureRepresentation: evaluated T(z) out of bounds @ z=3403.1 (max allowed z=105.05,
recommended limit is z <= 104)
```

**Note that `assertRaises(RuntimeError)` alone passes on both trees** — `7fdc49b` does raise a
`RuntimeError` here. The assertion has to bite on the message text, and it does.

`test_the_bracket_does_not_leave_the_tabulated_range`, both subtests:

```
ERROR (init_z=34.06668974249948):
RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.39498 (min allowed
z=-0.24, recommended limit is z >= -0.2376)

ERROR (init_z=10500000000000.5):
RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=1.05e+13 (max allowed
z=1.05e+12, recommended limit is z <= 1.0395e+12)
```

The second is the guess itself being out of range — the case that motivated clamping `init_z` and
not only the expanded endpoints (see "Observations not acted on" item 2).

### 3. Acceptance table

| Check | Threshold | Measured |
|---|---|---|
| The four equality redshifts, against prompt 01's reference | ≤ 4 ulp (D1; prompt said bit-identical / ≤ 1) | **0, −2, 0, −2** |
| The four, against the `7fdc49b` floats | — (recorded) | **+3, 0, +1, 0 ulp**; both Λ roots bit-identical |
| Prompt 01's four test methods | pass **unchanged** | **pass, file untouched** |
| §3 items 1 and 2 on `HEAD~1` | **fail**, output quoted | **2 failures, 5 errors**; quoted above |
| `_rho_fluid` evaluation count, four production call sites | quoted before and after | **3, 1, 1, 1 → 23, 25, 21, 25** (+20, +24, +20, +24) |
| Wall-clock cost of constructing a `QCD_Cosmology` | quoted before and after | **49.6 ms → 46.7 ms** (mean of 7; before range 47.0–58.5, after 46.4–47.1; a repeat of the "after" run gave 50.5 ms mean, 46.9–65.3). **Unmeasurable against the 3,000-node build**, as the prompt expects |
| `CosmologyModels` suite | 34 → 37, OK | **34 → 37**, OK, 1.444 s |
| `ComputeTargets` suite | **447 → 447**, OK | **447**, OK, 215.4 s, no failures, no re-run needed |
| `T_Z_REPRESENTATION_VERSION` | **6** before and after | **6 → 6** |
| The two printed banner lines | character-identical at `:516-517` | **`z = 3407`, `z = 0.3034`**, unchanged |
| `black --check` | clean | **clean** |

Bracket-expansion behaviour at the four production call sites, confirming §2.1 item 4: the loop
runs **one step** in every case and the first expanded pair straddles immediately, because
`match_rho(init_z)` is at the rounding floor but not zero. Ranges of $z$ visited:
QCD m=r `[2408.59, 4818.17]`, QCD m=Λ `[-0.0783407, 0.843319]`, stand-in m=r `[2406.07, 4813.13]`,
stand-in m=Λ `[-0.0783407, 0.843319]` — all inside `[-0.24, 1.05e12]` and `[-0.24, 1.05e6]`
respectively.

### 4. Formatting

`./venv/bin/python -m black CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
CosmologyModels/tests/test_rho_equality.py` → both left unchanged on the final run; the tree is
clean under `--check`.

## Observations not acted on

1. **Prompt 01's bracketed reference is 1.494 ulp from the exact matter–$\Lambda$ root, and the
   closed form is 0.506 ulp from it — so `LAMBDA_CLOSED_FORM_ULP` measures the reference's own
   error.** `match_rho` for that pair is exactly $\rho_{m0}(1+z)^3-\rho_\Lambda$, with no
   temperature dependence, so the root is available in closed form from the model's own float
   constants. Evaluated at 60 decimal digits from `QCD_Cosmology`'s `rho_m0 =
   6.894224419906764e+105` and `rho_cc = 1.52665741011693e+106`:

   | | value | ulp from exact |
   |---|---|---|
   | exact | `0.303423032996407410561312801228` | — |
   | closed form / this solve / the `7fdc49b` solve | `0.30342303299640738` | **−0.506** (the nearest double) |
   | prompt 01's `bracketed_reference` | `0.30342303299640749` | **+1.494** |

   The reference is **the second-nearest double, on the wrong side**. This does not weaken anything
   asserted: prompt 01's tests all pass, and the direction of the finding is that the shipped
   answer is *better* than the thing it is scored against. But README §3.1 calls the bracketed
   reference "the anchor every measurement is scored against" and item (a) is written as agreement
   with it, and on the one pair where an oracle exists the anchor is the less accurate of the two.
   **Opened as a §3 issue** on this board and indexed. It is not acted on here because README §3.1
   is a planning statement and prompt 02 may not edit it, and because changing the reference would
   change what prompt 01's four tests assert, which §3 of this prompt forbids.

2. **The guess is clamped as well as the expansion, which the prompt did not ask for.** §2.1 item 3
   requires only that "the expansion must never propose $z$ below the representation's floor".
   A first implementation did exactly that and still evaluated `match_rho(init_z)` unclamped, which
   left defect (ii) alive for a guess that is itself out of range: the truncated-cosmology case of
   D2 raised `TemperatureRepresentation: evaluated T(z) out of bounds @ z=3403.1` rather than the
   method's own error. Clamping `init_z` into the tabulated range before evaluating it is what
   makes the guard reachable at all, so it is inside the prompt's intent rather than beyond it, but
   it is a line the prompt does not name and it is recorded here for that reason. No issue opened.

3. **The cap `BRACKET_EXPANSION_MAX_STEPS = 140` cannot be reached, because the clamp always stops
   the expansion first.** It is kept as a guard against a future representation that reports no
   finite bounds. Nothing tests it, and a test would have to construct a model with an unbounded
   $T(z)$, which does not exist. No issue opened; recorded so a later reader does not mistake the
   absence of coverage for an oversight.

4. **Stale CPU-saturating processes were running on this machine throughout.** Ten
   `while :; do :; done` shells left over from a loaded-machine benchmark run at 12:55 in this
   session's scratchpad were still alive at 14:20, 1 h 27 m later, saturating every core; the
   `kill $LOADPIDS` that should have reaped them did not. They do not affect any pass/fail result
   or any float in this log, and the construction timing above was taken with both trees under the
   same load, but they inflate every wall-clock figure here (the `ComputeTargets` suite took 215 s
   against log 01's 181 s on the same tree) and they will corrupt
   `[06-t-photon-call-cost-needs-a-quiet-machine]`, whose whole point is a quiet machine, if they
   are still running when **prompt 05** takes its five-run mean. They were **not killed** — they
   are not this prompt's to reap — but prompt 05 must check for them before measuring. No issue
   opened; this is an environment observation, not a repository one.

5. **`measure_rho_equality.py` reproduces the shipped *call shape*, not the shipped *method*, so
   it is unaffected by this change and will go on reporting the pre-change behaviour.** Its §2,
   §3.1 and §3.2 all build `root_scalar(f, x0=guess, xtol=1e-6, rtol=1e-4)` directly
   (`:74`, `:117`, `:141`) rather than calling `_find_rho_equality`. Re-run at this commit it
   still exits 0 and still prints `-3.66e-16`, `ValueError: math domain error` at −30 % and the
   two `TemperatureRepresentation` bounds errors — every figure `RECONCILIATION.md` §2 confirmed.
   That is **correct for an audit artefact** (CLAUDE.md: verification documents are additive, and
   the script was right for the tree it was taken on) and it is why it was not edited, which this
   prompt's file list forbids anyway. Recorded because a later reader who runs it expecting to see
   prompt 02's behaviour will not, and the log 01 §4.1 table it produced is now a record of
   `7fdc49b` rather than of the method. No issue opened.

## State handed to the next prompt

**The four equality redshifts at this commit**, at 17 significant figures and in hex. These replace
log 01's table as the campaign's reference values from here on; prompt 03 scores the three
closed-form sites against **these**.

| Model | Pair | `_find_rho_equality` | `float.hex()` |
|---|---|---|---|
| `QCD_Cosmology(max_z=1e12)` | matter = radiation | `3406.6689742499511` | `0x1.a9d5683cafad0p+11` |
| `QCD_Cosmology(max_z=1e12)` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` |
| `PureRadiationEOS(max_z=1e6)` | matter = radiation | `3403.1059638279457` | `0x1.a963640e40f2cp+11` |
| `PureRadiationEOS(max_z=1e6)` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` |

**Two of these moved from log 01's table** (+3 ulp and +1 ulp, both matter–radiation, both onto
prompt 01's reference); the two $\Lambda$ roots did not move a bit. **Prompt 03's `main.py:526`
correction must be re-taken against this table, not against log 01's**: the closed form
$\Omega_m/\Omega_r-1$ is `3406.6689742499480` (`0x1.a9d5683cafac9p+11`), so on `QCD_Cosmology` it
now sits **7 ulp = −9.3e-16 relative** below the solve, where `RECONCILIATION.md` §6 measured
−9.34e-16 against the reference. The $\Lambda$ closed form and the solve are **the same float**.

**The bracket-expansion policy and its constants**, for prompt 06's provenance entry:

- expansion variable: $1+z$, multiplicative;
- `BRACKET_EXPANSION_FACTOR = sqrt(2.0)`, `BRACKET_EXPANSION_MAX_STEPS = 140`
  ($\sqrt2^{140}=2^{70}=1.2\times10^{21}$, which spans the tabulated range in one direction at the
  default `max_z = 1e20`), both **function-local** to `_find_rho_equality`;
- clamped to `self._T_z_spline._min_z`, `self._T_z_spline._max_z` — the representation's own
  buffered bounds, `-0.24` and `1.05 × (1+max_z) − 1`. **The guess is clamped too**, not only the
  endpoints;
- sign test, not a product test (D4);
- solve: `root_scalar(match_rho, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=8.9e-16)`.

**The tolerance, for `docs/TOLERANCE-PROVENANCE.md`.** `xtol=1e-300, rtol=8.9e-16` — Brent's own
$4\varepsilon = 8.881784\times10^{-16}$ floor, which `scipy` 1.15.2 accepts and below which it
raises. Chosen by the user on 2026-09-16 (README §7 **D1**, amended), on the measurement in D1
above; the competing value was `rtol=1e-14`, `_solve_T_z:584`'s, which is 75 ulp of slack at
$z\sim3.4\times10^3$ and lands 7 ulp from the independent reference. **`_solve_T_z`'s own
`rtol=1e-14` is not re-opened by this and must not be**: it is per-node in a 3,000-node tabulation
where uncorrelated scatter is the thing being bought, not a single value.

**Achieved accuracy and cost.** 0 / −2 / 0 / −2 ulp against prompt 01's bracketed reference.
**23 / 25 / 21 / 25** `_rho_fluid` evaluations at the four production call sites, against
**3 / 1 / 1 / 1** at `7fdc49b` — **+20 to +24 each, two per model construction**. `AUDIT.md` §3.1's
"+6 to +9" is for tightening the secant and does **not** survive bracketing; the comment in the
file quotes the measured figure and says so. Construction cost **49.6 → 46.7 ms** mean of 7, i.e.
inside the run-to-run scatter.

**The failure surface, for prompt 04 and prompt 06.** `_find_rho_equality` now raises its own
`RuntimeError`, prefixed `LambdaCDM_GenericEOS._find_rho_equality: `, in two places: a failure to
bracket (naming both species, the guess, the clamped guess, both endpoints and both residuals) and
a non-converged Brent (naming both species and the bracket). **No displacement of the guess can
reach either** — the only route is a cosmology whose tabulated range does not contain the root, as
`TRUNCATED_MAX_Z = 100.0` in the test module demonstrates. A later prompt that wants a failing
case must use that route.

**Names prompt 03 and later can use.** `CosmologyModels/tests/test_rho_equality.py` now also
exports `solve_recording_evaluations(model, pair, init_z) -> (root, evaluations)`,
`DISPLACED_GUESS_FRACTIONS` and `TRUNCATED_MAX_Z`, alongside log 01's list. The recording helper
shadows `model._rho_fluid` on the instance and restores it in a `finally`, so it is safe to use on
the class-level models.

**Open, and prompt 03 should read it before scoring anything in ulp:**
`[02-bracketed-reference-is-not-the-exact-root]` — the campaign's reference is 1.494 ulp from the
exact matter–$\Lambda$ root while the closed form is 0.506 ulp from it. See "Observations not acted
on" item 1 and the board §3 entry.

**Suite counts to carry forward:** `CosmologyModels` **37**, `ComputeTargets` **447**.
`T_Z_REPRESENTATION_VERSION` = **6**.

**Reproduction commands.**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```
