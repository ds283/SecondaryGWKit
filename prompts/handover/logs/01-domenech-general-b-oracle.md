# Log 01 — the Domènech general-$b$ oracle

**Prompt:** [`prompts/handover/01-domenech-general-b-oracle.md`](../01-domenech-general-b-oracle.md)
**Commit:** *(this commit)* — "Land the Domenech general-b oracle with its tests"
**Model:** Claude Opus 5
**Date:** 2026-09-20
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

Two new files. **No existing file in the tree is edited** — `ComputeTargets/QuadSourceIntegral.py`,
`QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/`, `main.py`, `config/`, every factory, every
schema, the six `extract_*.py`, `docs/spec/`, `ComputeTargets/tests/test_quadsource_integral.py`,
`ComputeTargets/tests/kohri_terada.py` and everything under `docs/radiation-oracle/` are untouched.
The diff outside these two files is the log, the board, `docs/OPEN_ISSUES.md` and one §3→§4 move on
the `radiation-oracle` board.

### `ComputeTargets/tests/domenech.py` (new, 782 lines)

The general-$b$ kernel, beside `kohri_terada.py` and shaped like it: a module docstring naming the
two papers **with their arXiv versions**, the two sign errors in the review, the provenance of every
object, and a "what its agreement with itself does and does not prove" section in the shape of
`KOHRI-TERADA-ORACLE.md` §0. Public symbols:

| Symbol | What |
|---|---|
| `w_of_b(b)`, `cs_of_b(b)`, `rho_of_b(b)` | $w = c_s^2 = \frac{1-b}{3(1+b)}$, $c_s$, $\rho = \frac{b+2}{b+1}$ |
| `two_c_squared(b)` | $2c^2 = 2\big(\frac{2+b}{3+2b}\big)^2$ — the constant the 2020 paper folds into its $I$ and the review does not, so $I_{\rm KT} = 2c^2 I_{\rm rev}$ |
| `kt_convention_norm(b)` | $N(b) = -\frac{(3+2b)^2}{2(2+b)^2}$, recon §8's boxed result |
| `kernel_constant(b)` | $K(b) = 4^b\Gamma^2[b+\frac32]\frac{2b+3}{b+2}$; the exact kernel carries $\mathcal N = \pi K$ |
| `Kinematics`, `kinematics(v, u, b)` | namedtuple `(cs, delta, one_plus_y, one_minus_y, y, on_cut)`; $1\pm y$ formed from $\delta = c_s(u+v)-1$ and $c_s(u-v)$ directly, per recon §5.3 |
| `OutsideGervoisNavelet(ValueError)` | raised for $c_s\lvert u-v\rvert\ge1$ ($y\ge1$); its docstring is the whole of the §2 item 5 decision |
| `A_coefficient(one_plus_y, one_minus_y, b)` | recon §3.5's elementary $A(y)$ — no special function at all |
| `B_coefficient(...)`, `C_coefficient(...)` | the on-cut and off-cut $\sin$ coefficients |
| `B_at_resonance(b)` | $B(-1) = -\frac{2^b\Gamma(b)(3+2b)(1+b+b^2)}{(1+b)\Gamma(2b+3)} = -C(+1)$, $b>0$ |
| `ferrers_P`, `ferrers_Q`, `olver_Q` | the six raw `mpmath` calls of recon §3.2, exposed **only** so the closed forms can be scored against them; nothing in the module's own path calls them |
| `I_target(v, u, x, b)` | the **corrected** (4.10) with the $x\to\infty$ coefficients (3.3)/(3.4) substituted, $J_{b+1/2}(x)$ and $Y_{b+1/2}(x)$ kept exact |
| `I_asymptotic(v, u, x, b)` | the review's (4.12) **with its $\cos$ sign flipped** |
| `I_quadrature(v, u, x, b, x_lo=0.0, x_hi=None, epsabs=1e-16, epsrel=1e-13)` | the corrected (4.10) with (4.11)'s **finite-$x$** integrals by `scipy.quad`, returning `(value, declared_error)`; the range broken at every period of $1$, $c_s(u\pm v)$, $c_su$, $c_sv$ exactly as `kohri_terada.I_RD_quadrature` breaks it |
| `small_x_limit(x, b)` | $+x^2/(2(2+b))$ |
| `total_from_I_rev(k, q, r, tau_response, tau_source_max, b, ...)` | the mapping onto `total`, returning `total`, `total_error`, `u`, `v`, `x`, `x_min`, `I_rev_trunc`, `I_rev_trunc_error`, `head`, `head_error`, `head_over_I` |
| `MPMATH_DPS = 30` | the working precision every coefficient is evaluated at, applied with `mp.workdps` so the answer does not depend on the caller's `mp.dps` |

`_R_nu`, `_S_nu`, `_y_from`, `_require_defined`, `_require_on_cut`, `_Isimpledef_integrand`,
`_breakpoints` are private.

**No tolerance moves the answer.** `I_target`, `I_asymptotic` and the three coefficients are closed
forms; the only tolerances in the module are `scipy.quad`'s inside `I_quadrature`, which returns its
own declared error beside the value so every comparison is scored against it.

### `ComputeTargets/tests/test_domenech_oracle.py` (new, 1164 lines, **22 test methods**)

| Prompt §3 group | Class | Methods |
|---|---|---|
| 1 Legendre mapping | `TestLegendreMapping` | `test_ferrers_wronskian`, `test_olver_wronskian`, `test_ferrers_P_closed_forms`, `test_parity_at_mu_plus_nu_zero_or_two`, `test_special_values_at_b_zero_and_half`, `test_olver_phase_conversion` |
| 2 $A$, $B$, $C$ vs raw calls | `TestCoefficients` | `test_A_against_raw_ferrers_P`, `test_B_against_raw_ferrers_Q`, `test_C_against_raw_olver_Q` |
| 3 target vs exact kernel | `TestTargetAgainstExactKernel` | `test_difference_falls_like_one_over_x` |
| 4 doubly asymptotic | `TestAsymptoticForm` | `test_asymptotic_approaches_target_like_one_over_x`, `test_asymptotic_equals_target_at_b_zero` |
| 5 the $b=0$ tie | `TestKohriTeradaTie` | `test_quadrature_is_nine_eighths_of_eq22`, `test_target_is_nine_eighths_of_eq25`, `test_refuses_the_T_first_shape_at_b_zero` |
| 6 small $x$ | `TestSmallX` | `test_small_x_limit` |
| 7 the resonance | `TestResonance` | `test_coefficients_at_the_resonance`, `test_target_one_sided_limits_agree_like_delta_to_the_b`, `test_exact_kernel_is_regular_at_the_resonance` |
| 8 $N$ over the fixtures | `TestNormalisation` | `test_N_is_constant_over_the_nine_b_02_fixture_cases` |
| acceptance 1 | `TestOracleIsFixed` | `test_answer_does_not_depend_on_the_callers_mp_dps`, `test_two_c_squared_bridges_the_two_conventions` |

Two module-level helpers the tests share: `envelope(v, u, x, b)`, recon §2.4's unit of distance, and
`eq25_rounding_scale(v, u, x)`, an absolute rounding estimate for KT eq. (25) built to the same
recipe `kohri_terada.I_RD_with_rounding_scale` uses for eq. (22) — see the deviations.

---

## Deviations from the prompt

### 1. The off-cut coefficient uses the 2020 paper's $1/\tilde y^2$ form, not recon §5.2's second box — **IMPLEMENTATION CHOICE**, and it settles recon §10 item 5

**What the prompt assumed.** Recon §9's brief gives $C(\tilde y) = S_b + 2\rho S_{b+2}$ with

$$S_\nu(\tilde y) = -\frac{\pi}{2\sin(b\pi)\Gamma(\nu-b+1)}\Big[(\tilde y-1)^b\mathbf F\big(\nu+1,-\nu;1+b;\tfrac{1-\tilde y}{2}\big) - \tfrac{\Gamma(\nu-b+1)}{\Gamma(\nu+b+1)}(\tilde y+1)^b\mathbf F\big(\nu+1,-\nu;1-b;\tfrac{1-\tilde y}{2}\big)\Big].$$

**What is actually there.** That form is mathematically correct — it reproduces `legenq(type=3)` to
$10^{-48}$ term by term at 50 digits — but it is **catastrophically ill-conditioned away from
$\tilde y = 1$**. At $b = 0.2$, $\tilde y = 150$ its two terms are each $2.2426\times10^{5}$ and
their difference is $1.02\times10^{-8}$: a **13-order cancellation**. The cancellation is not cured
by working precision, because what it amplifies is the $1.1\times10^{-17}$ by which a Python
`float` `b = 0.2` differs from the number meant; the two terms are then each accurate to
$\sim10^{-15}$ *relative*, and $10^{-15}\times2.2\times10^{13}$ is a per-cent error. Measured
against the raw Olver call, identical at `mp.dps` 30, 60 **and** 120:

| $\tilde y$ | recon §5.2's off-cut box, relative error | cancellation factor |
|---|---|---|
| 1.5 | 6.2e-15 | 1.5 |
| 40 | 9.5e-10 | ~1e+9 |
| 150 | 8.6e-08 | 2.2e+13 |

Recon §10 item 5 left exactly this untested: *"not tested at large $\lvert y\rvert$ ($\gtrsim100$ on
`q-smooth`), where the $\tilde y^{-\nu-1}$ decay and the $(\tilde y\pm1)^b$ factors may cancel."*
**`q-smooth` at $b = 0.2$ sits at $\tilde y = 149.5$** — it is a fixture shape, not a corner.

**What was done instead.** `_S_nu` transcribes the 2020 paper's **own** $\lvert x\rvert>1$
definition of $Q^\mu_\nu$ — the $1/x^2$ hypergeometric at `1912.tex:743-747`, which is DLMF 14.3.7 —
with $\mathcal Q = e^{-\mu\pi i}Q/\Gamma[\mu+\nu+1]$ at `1912.tex:749-751`. At $\mu = -b$ the
$(x^2-1)^{\mu/2}$ cancels the $(\tilde y^2-1)^{b/2}$ exactly and what is left is

$$(\tilde y^2-1)^{b/2}\mathcal Q^{-b}_\nu(\tilde y) = \frac{\sqrt\pi}{2^{\nu+1}\Gamma(\nu+\tfrac32)}\,\tilde y^{\,b-\nu-1}\,{}_2F_1\Big(\tfrac{\nu-b+2}{2},\tfrac{\nu-b+1}{2};\nu+\tfrac32;\tfrac{1}{\tilde y^2}\Big),$$

which has no cancellation anywhere: the argument is in $[0,1]$, the series converges at $1$ itself
for $b>0$ ($c-a-b = b$), and it is regular at $b = 0$, so **$C$ needs no $b = 0$ branch at all**
where $B$ does. It agrees with `legenq(type=3)` to rounding at every $\tilde y$ tried, 0.00e+00
relative at $\tilde y = 40$ and $150$.

**Why this is not "resolving a disagreement with the recon by choosing".** It is not a disagreement:
both forms are the same function, the recon flagged the conditioning as untested, and the
replacement is transcribed from the paper the recon cites rather than inferred. Recon §5.2's box is
still the right form near the resonance and is still what `_R_nu` uses **on-cut**, where the range
is bounded and it is exact to rounding (recon §5.3's table). **A §3 issue is opened** against the
recon's §9 brief, because anyone re-implementing from that document would inherit the defect.

### 2. The $y>1$ continuation is **refused**, not implemented — **IMPLEMENTATION CHOICE** (prompt §2 item 5, which requires the choice to be stated)

`I_target` and `I_asymptotic` raise `OutsideGervoisNavelet` for $c_s\lvert u-v\rvert\ge1$. Three
reasons, all recorded on the exception class itself:

1. **Campaign README §5.1 forbids it.** Recon §6.2's rule ($\mathcal I^\infty_J = 0$ and
   $\mathcal I^\infty_Y = -[\text{off-cut form at }+y]$) was matched numerically on three shapes and
   **not read from a source** (recon §10 item 2). §5.1 requires a transcription to be scored against
   the paper's own defining integral before it goes anywhere, and there is no paper here.
2. **There is nothing to score it against where it would be used.** The only shape that reaches it
   is `T-first` at $b = 0$ — $\lvert q-r\rvert = 2000 > k = 1000$, not a closable triangle — and
   there Kohri & Terada's eq. (25) is *equally* inapplicable: `KOHRI-TERADA-ORACLE.md` §8 item 4
   records that for $\lvert v-u\rvert>\sqrt3$ its $\cos$ term is absent, and with the term kept the
   difference is $O(1)$ at every $x$. So the one external reference that could pin the continuation
   is itself undefined there.
3. **Nothing needs it.** No physical configuration reaches $y>1$: momentum conservation puts $(u,v)$
   on $\lvert u-v\rvert\le1\le u+v$, and $c_s<1$ then makes $c_s\lvert u-v\rvert<1$ always
   (recon §6.2). `I_quadrature` — the decisive instrument — **never forms $y$** and is perfectly
   well defined on `T-first` at $b = 0$: it ties to $\tfrac98$ eq. (22) there to **2.18e-15**, and
   that point is in test 5. At $b = 0.2$ the same shape has $y = +0.9979$, is on-cut, and is one of
   the nine cases test 8 scores.

This is **not** prompt §6's last stop condition. That condition is "you cannot decide item 5 without
reading Gervois–Navelet"; the decision to refuse needs no source, and §2 item 5 offers it explicitly.
`test_refuses_the_T_first_shape_at_b_zero` pins the refusal, and asserts that the same shape at
$b = 0.2$ is on-cut and finite.

### 3. `eq25_rounding_scale` is built in the test module — **STRUCTURALLY REQUIRED**

Test 5's second half scores `I_target(b=0)` against $\tfrac98\,$`kohri_terada.I_RD_asymptotic`, and
needs eq. (25)'s own declared error to build a bound from (campaign README §5 rule 8).
`kohri_terada.py` has `I_RD_with_rounding_scale` for eq. (22) but **no equivalent for eq. (25)**,
and this prompt does not edit that file. The estimate is therefore built in
`test_domenech_oracle.py` to the same recipe: every term's magnitude plus the error its inputs'
rounding induces. It is needed: at $u = q/k = 0.01$ eq. (25)'s $\sin$ bracket is
$-4uv = -4.4\times10^{-2}$ against $d\log\lvert\cdot\rvert = +4.4\times10^{-2}$, cancelling to
$1.4\times10^{-5}$, and the bound there is 2.6e-10 relative against ~7e-15 elsewhere — the same
$1/(u^3v^3)$ conditioning `KOHRI-TERADA-ORACLE.md` §7.2 traced eq. (22)'s 3.18e-10 to.

### 4. Two Wronskian tests were anchored to the module's helpers after breakage 4 exposed a gap — **STRUCTURALLY REQUIRED** (prompt §3: "a break that no test catches is a missing test")

`test_ferrers_wronskian` and `test_olver_wronskian` need derivatives, so they evaluate
`mp.legenp`/`mp.legenq` inline rather than through the module's float-returning helpers. Breakage 4
showed that this made them test the *convention* without testing the *module's wiring*:
`ferrers_Q` could be switched from `type=2` to `type=3` and `test_ferrers_wronskian` still passed.
Both tests now assert `ferrers_P(nu, b, y) == float(P(y))`, `ferrers_Q(nu, b, y) == float(Q(y))`
and `olver_Q(nu, b, x) == float(Q(x))` at the evaluation point, which closes the gap with no loss of
precision. Re-run with the breakage in place, `test_ferrers_wronskian` now fails.

### 5. Three test bounds are set by the *reference's* conditioning, not the module's — **IMPLEMENTATION CHOICE**, each stated in the test that carries it

- `test_special_values_at_b_zero_and_half` evaluates the papers' reduced forms (`1912.tex:804-822`)
  **in `mpmath` at 30 digits, not in double**. Several of them are the ill-conditioned way to write
  their function: $\mathcal Q^0_0 + 4\mathcal Q^0_2 = \tfrac32\tilde y^2\ln\frac{\tilde y+1}{\tilde y-1} - 3\tilde y$
  cancels by 213 at $\tilde y = 6$ and 21597 at $\tilde y = 60$, and
  $\mathcal Q^{-1/2}_{5/2}$'s numerator is $846.0 - 846.0$ at $\tilde y = 6$. In double they are
  1.2e-13 to 2.7e-11 out and it is the **reference** that is wrong. Evaluated at 30 digits the
  comparison is against the module's double output, as it should be, and holds at 1e-13.
- `test_small_x_limit` calls `I_quadrature` with `epsabs=0.0`. The module's `1e-16` default (taken
  from `kohri_terada.I_RD_quadrature`, to match its shape) is an **absolute** floor, and at
  $x = 0.01$ the integral is $2.3\times10^{-5}$, so that floor lets `quad` stop with a declared error
  of 5.5e-09 relative on a value whose measured accuracy is 1e-13. Only the declared error moves:
  $\alpha$ is identical to six figures either way. The default is left as it is, so the two oracle
  modules keep the same signature.
- The two Wronskians assert 1e-14, not 1e-20. Written with $\nu = b+j$ and the right-hand side in
  the **integer** $j$ — $\Gamma(j+1)/(\Gamma(2b+j+1)(1-y^2))$ and $-1/(\Gamma(j+1)(x^2-1))$ — the
  identity holds to **1.1e-16**, which is $\nu = b+2$'s own double representation inside the
  `mpmath` call. Written in $\nu-b+1$ and $\nu+b+1$ instead it holds only to 2.5e-16, because
  $b+2-b$ is not $2$ in floating point and $\Gamma$ of an argument off by $4\times10^{-16}$ moves by
  $5\times10^{-16}$. 1e-14 is still five orders inside any normalisation slip.

### 6. Nothing else

No other deviation. The module is a pure function of $(v, u, x, b)$; it uses `mpmath` for the
Legendre sector and `scipy.special.jv`/`yv` only for the outer $J_{b+1/2}(x)$, $Y_{b+1/2}(x)$ and
the source's $J_{b+1/2}$, $J_{b+5/2}$ — orders at most $2.5$, so
`[01-scipy-jv-yv-high-order-boundary]`'s Amos boundary at $\nu\approx86$ is not approached; the
docstring says which regime the module is good for and why. `mp.dps` never leaks: `mp.workdps` is
applied in each coefficient and `test_answer_does_not_depend_on_the_callers_mp_dps` asserts bit
equality at `mp.dps` 15, 30 and 80.

---

## The deliberate-breakage record

Each break applied alone to a pristine tree, the test module run, the tree restored and the SHA-256
of `domenech.py` checked back to `cea0b13ac981333aa99c298a3b509d69793a7d2723ac6cfa6c231b63b3b7d056`.
"Caught" names the **test methods** that failed, not the subtest count.

### Breakage 1 — the printed order $\big(J_{b+1/2}\mathcal I_Y - Y_{b+1/2}\mathcal I_J\big)$

`I_quadrature`, `I_target` and `I_asymptotic` all negated, which is what transcribing
`review.tex:650` as printed produces.

**Caught by 4 of 22:**

- `TestSmallX.test_small_x_limit` — *"I_rev is positive at small x; (4.10) as printed is not"*;
  $I = -2.234\times10^{-2}$ at $b = 0$, $x = 0.3$. **The cheapest discriminator, and the first to
  fire.**
- `TestKohriTeradaTie.test_quadrature_is_nine_eighths_of_eq22`
- `TestKohriTeradaTie.test_target_is_nine_eighths_of_eq25`
- `TestNormalisation.test_N_is_constant_over_the_nine_b_02_fixture_cases` — and this is the
  instructive one. **$N$ stays constant**, at $+1.194214876033$ on all nine cases, spread 5.3e-13;
  $\lvert N - N(b)\rvert = 2.39$ everywhere. It is exactly `KOHRI-TERADA-ORACLE.md` §0's *benign*
  failure — a wrong normalisation is constant but wrong — and it is why the test asserts the value
  as well as the constancy.

**Not caught by the other 18**, and the reason is structural rather than accidental: all of
`TestLegendreMapping`, `TestCoefficients`, `TestResonance`, `TestOracleIsFixed`,
`TestTargetAgainstExactKernel.test_difference_falls_like_one_over_x` and both of
`TestAsymptoticForm` compare two of the module's **own** objects, and an overall sign flips both
sides. Only the derived small-$x$ limit and the two external ties can see it. This is the
quantitative form of the module docstring's provenance paragraph.

### Breakage 2 — $\Gamma[1]$ in place of $\Gamma[3]$ on the off-cut second term

`C_coefficient`'s `2.0 * rho_of_b(b)` changed to `1.0 * rho_of_b(b)`.

**Caught by 7 of 22:**

- `TestCoefficients.test_C_against_raw_olver_Q`
- `TestLegendreMapping.test_special_values_at_b_zero_and_half`
- `TestTargetAgainstExactKernel.test_difference_falls_like_one_over_x` — **the intended
  discriminator.** The off-cut rows stop falling like $1/x$; recon §4.2 measured the stall at
  $8\times10^{-2}$ of the envelope.
- `TestKohriTeradaTie.test_target_is_nine_eighths_of_eq25` — the external tie, through the two
  off-cut points among the five
- `TestResonance.test_coefficients_at_the_resonance` — $C(1)\ne-B(-1)$
- `TestResonance.test_target_one_sided_limits_agree_like_delta_to_the_b`
- `TestResonance.test_exact_kernel_is_regular_at_the_resonance`

**Not caught:** `test_N_is_constant_over_the_nine_b_02_fixture_cases` and
`test_quadrature_is_nine_eighths_of_eq22`, both correctly — they score `I_quadrature`, which does
not use $C$ at all — and the on-cut tests.

### Breakage 3 — `I_target` in place of `I_quadrature` in test 8

`total_from_I_rev` changed to take the full integral from `I_target(v, u, x, b)` and subtract the
quadratured head, instead of integrating (4.11) between the code's own limits.

**Caught by 1 of 22:** `TestNormalisation.test_N_is_constant_over_the_nine_b_02_fixture_cases`, and
it fails on **the constancy assertion**, not the value:

| shape | $x_{\rm resp}$ | $N$ |
|---|---|---|
| together | 980 / 100 / 30 | $-1.19636$ / $-1.09561$ / $-1.13995$ |
| T-first | 980 / 100 / 30 | $-1.23419$ / $-1.19862$ / $-1.03191$ |
| q-smooth | 980 / 100 / 30 | $-1.03710$ / $+0.82784$ / $-4.23430$ |

min $-4.234$, max $+0.828$, **spread 5.06** about a mean of $-1.260$. This is prompt §2 item 4's
warning made concrete: *"Getting this backwards will look like a pipeline defect and is not one."*
The worst rows are `q-smooth`, $u = 0.01$, exactly where recon §6.4 says the target's $O(1/x)$ is
controlled by $c_s\min(u,v)x$ rather than by $x$.

**Not caught by the other 21**, correctly — nothing else in the module changed.

### Breakage 4 — `type=2` swapped for `type=3` in one Legendre call

`ferrers_Q`'s `mp.legenq(nu, -b, y, type=2)` changed to `type=3`.

**Caught by 3 of 22**, all as `ERROR` rather than `FAIL` — at $\lvert y\rvert<1$ `legenq(type=3)`
returns a complex number and `float()` raises `TypeError`:

- `TestCoefficients.test_B_against_raw_ferrers_Q`
- `TestLegendreMapping.test_parity_at_mu_plus_nu_zero_or_two`
- `TestLegendreMapping.test_ferrers_wronskian` — **only after the test was fixed.** On the first
  run it did **not** fire: it evaluated `mp.legenq(..., type=2)` inline, so it checked the
  convention without checking that the module's helper is the call being checked. That is a missing
  test, and it was added rather than recorded: see deviation 4. The record here is of the state
  after the fix, and the state before is stated so the gap is not hidden.

**Supplementary probe, the other direction** — `olver_Q`'s `type=3` changed to `type=2`, so that the
remaining three of the six calls are exercised too. **Caught by 3:**
`TestLegendreMapping.test_olver_wronskian`, `TestLegendreMapping.test_olver_phase_conversion`
(the residual imaginary part after the $e^{+ib\pi}$ phase stops vanishing) and
`TestCoefficients.test_C_against_raw_olver_Q`.

---

## Verification performed

### Suite counts

| Suite | Before (`162f6df`) | After | Verdict |
|---|---|---|---|
| `ComputeTargets/tests` | **530** | **552** | OK — `+22`, exactly the 22 test methods added |
| `CosmologyModels/tests` | **39** | **39** | OK |
| `LiouvilleGreen/tests` | **148** (skipped=1) | **148** (skipped=1) | OK — control, untouched |

`ComputeTargets` runs in 161 s; the new module contributes 19 s. `test_tk_wkb_phase`'s known
wall-clock flake did not fire on any run here.

`./venv/bin/python -m black --check .` — *"269 files would be left unchanged"*, clean.

### Prompt §5 acceptance, item by item

**1. Pure function of $(v,u,x,b)$, no tolerance moves the answer.** Asserted by
`test_answer_does_not_depend_on_the_callers_mp_dps` (bit-identical at `mp.dps` 15, 30, 80 for $A$,
$B$, $C$, `I_target`, `I_asymptotic`). No datastore, no Ray, no global state.

**2. All eight test groups pass; the four breakages are recorded above with the tests each caught.**

**3. Test 5's $b=0$ tie**, at recon §7's own $(u, v, x)$. Both figures quoted as acceptance asks:

| $(u, v, x)$ | `I_quadrature` vs $\tfrac98$ eq. (22) | its bound | `I_target` vs $\tfrac98$ eq. (25) | its bound |
|---|---|---|---|---|
| $(0.7, 1.1, 20)$ | **4.02e-15** | 5.5e-14 | 4.65e-15 | 7.2e-15 |
| $(0.7, 1.1, 200)$ | 5.16e-16 | 1.3e-13 | 1.69e-15 | 7.1e-15 |
| $(1.2, 0.5, 100)$ | 2.67e-16 | 1.5e-13 | 9.47e-16 | 9.6e-15 |
| $(0.3, 1.6, 300)$ | 3.38e-15 | 1.0e-13 | 1.20e-16 | 6.9e-15 |
| $(0.01, 1.1, 50)$ | **3.85e-10** | 7.3e-09 | 2.26e-11 | 2.6e-10 |
| $(10, 12, 141)$ | 2.18e-15 | 7.5e-14 | *refused, $y = 1.0042$* | — |

So **$\le4.0\times10^{-15}$ except at $u = 0.01$, where it is $3.9\times10^{-10}$** — recon §7's
two figures, $\le4.5\times10^{-15}$ and $\sim4\times10^{-10}$, reproduced. The $3.9\times10^{-10}$
is **eq. (22)'s own double-precision rounding**, not this module's: its $1/(u^3v^3)$ prefactor is
$\sim10^6$ at $u = 0.01$ and multiplies terms that nearly cancel
(`KOHRI-TERADA-ORACLE.md` §7.2). The $I_{\rm target}$ column is an **exact identity**, not an
$O(1/x)$ approach — at $b = 0$ the order-$\tfrac12$ Bessel asymptotics are exact, so the target
object and the corrected (4.12) coincide and both equal $\tfrac98\times$ eq. (25) — and
`test_asymptotic_equals_target_at_b_zero` asserts the coincidence separately at $32\epsilon$.

**4. Test 8's $N$, with the reference's own error beside it** (README §5 rule 8). Nine $b = 0.2$
cases, exact flavour, at the reference pair $(10^{-45}, 10^{-12})$:

| shape | $x_{\rm resp}$ | $u$ | $v$ | $x = k\tau$ | $y$ | branch | head/$I$ | oracle's own | pipeline's own | $N$ |
|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 0.9091 | 1.0909 | 1.9057e+03 | −1.2521 | off-cut | 6.9e-06 | 8.6e-14 | 1.8e-12 | **−1.194214876033** |
| together | 100 | 0.9091 | 1.0909 | 1.9445e+02 | −1.2521 | off-cut | 7.5e-06 | 1.6e-13 | 2.0e-12 | **−1.194214876033** |
| together | 30 | 0.9091 | 1.0909 | 5.8336e+01 | −1.2521 | off-cut | 7.2e-06 | 6.8e-14 | 2.0e-12 | **−1.194214876033** |
| T-first | 980 | 10.0000 | 12.0000 | 1.7324e+02 | +0.9979 | on-cut | 1.2e-06 | 6.4e-14 | 1.0e-12 | **−1.194214876033** |
| T-first | 100 | 10.0000 | 12.0000 | 1.7678e+01 | +0.9979 | on-cut | 1.7e-06 | 1.7e-14 | 1.0e-12 | **−1.194214876033** |
| T-first | 30 | 10.0000 | 12.0000 | 5.3033e+00 | +0.9979 | on-cut | 1.8e-06 | 1.8e-14 | 1.0e-12 | **−1.194214876033** |
| q-smooth | 980 | 0.0100 | 1.1000 | 1.8899e+03 | −149.5409 | off-cut | 1.6e-05 | 6.9e-12 | 3.8e-12 | **−1.194214876033** |
| q-smooth | 100 | 0.0100 | 1.1000 | 1.9285e+02 | −149.5409 | off-cut | 2.0e-05 | 3.9e-12 | 1.1e-11 | **−1.194214876033** |
| q-smooth | 30 | 0.0100 | 1.1000 | 5.7854e+01 | −149.5409 | off-cut | 4.0e-06 | 2.5e-13 | 1.0e-12 | **−1.194214876033** |

- **min** −1.194214876033, **max** −1.194214876033, **spread 5.285e-13** (4.425e-13 relative about
  the mean).
- The derived value $-\frac{(3+2b)^2}{2(2+b)^2} = -1.194214876033$; **worst
  $\lvert N - N(b)\rvert = 4.130e-13$**.
- Against $I_{\rm rev}$ rather than the Kohri–Terada convention, $N_{\rm rev} = -1$ exactly:
  worst $\lvert N_{\rm rev}+1\rvert = $ **3.461e-13**.
- **The reference's own error is the "oracle's own" column**: the declared error of `scipy.quad` on
  the two (4.11) integrals, carried through the same prefactor — 1.7e-14 to 6.9e-12 relative. The
  pipeline's own declared error is 1.0e-12 to 1.1e-11. Every deviation is inside the sum of the two.
- **Which of the two the test asserts on.** The load-bearing assertion is **constancy**: the
  per-case assertion is $\lvert N - N(b)\rvert \le \lvert N(b)\rvert\times$(the two declared errors),
  and the class-level assertion is that the spread over the nine is $<10^{-11}$ of the mean. A
  second, separate assertion pins the **value** against recon §8's derivation. They are deliberately
  distinct: a drift is prompt §6's first stop condition and a constant-but-wrong $N$ is its second,
  and breakage 1 above shows the module failing the value while passing the constancy. The
  constancy statistic is the one that would catch a wrong kernel, measure or Green's function
  (`KOHRI-TERADA-ORACLE.md` §0), and the nine cases span $x$ from 5.3 to 1.9e+03, $u$ from 0.01 to
  10, and both Gervois–Navelet branches.
- **The head is computed and reported, not subtracted.** $I_{\rm rev,trunc}$ is (4.11) integrated
  between the code's own limits directly, which avoids differencing two numbers $10^5$ apart at
  small $x$; `head_over_I` is 1.2e-06 to 2.0e-05 and, as README §2 (k) says, is set by how early the
  integral starts rather than how late it ends.

**5, 6, 7.** Counts and `black` above; board and index below.

### Recon tables reproduced

Recon §2.4, $\lvert$target $-$ exact$\rvert$/envelope at $x = 100/1600/6400$ — the figure prompt §3
test 3 names:

| $b$ | $(u,v)$ | measured here | recon §2.4 |
|---|---|---|---|
| 0.2 | (1.3, 1.2) | 1.6e-02 / 1.5e-03 / **9.8e-06** | 1.6e-2 / 1.5e-3 / 9.9e-6 |
| 0.2 | (1.0, 1.3) | 2.5e-02 / 1.8e-03 / 1.9e-05 | 2.5e-2 / 1.8e-3 / 1.9e-5 |
| 0.2 | (2.5, 3.0) | 4.9e-04 / 2.1e-04 / 3.4e-05 | 4.9e-4 / 2.1e-4 / 3.4e-5 |
| 0.5 | (2.5, 3.0) | 6.2e-04 / 6.2e-05 / 1.8e-06 | 6.2e-4 / 6.2e-5 / 1.8e-6 |

Recon §4.2's off-cut table at $x = 6400/25600$, $b = 0.2$: (0.909, 1.091) **1.4e-04 / 1.7e-05**,
(1.0, 1.0) **3.2e-05 / 1.8e-05**, (0.6, 0.9) **6.9e-05 / 1.1e-04** — all three rows to the digit.

Recon §4.1's nine-case table: `I_quadrature` at $x = 30$ gives $+1.1724885558$e-01,
$+1.2003036441$e-01, $+1.0447050704$e-02 and $-7.2141335647$e-01 at $(b; u, v) = $
$(0.2; 0.909, 1.091)$, $(0.2; 0.3, 1.6)$, $(0.5; 0.7, 1.1)$, $(-0.3; 0.3, 1.6)$ — every printed
digit of the recon's **E** column. Recon §4.1's small-$x$ ratios 0.99992 / 0.99994 / 0.99995 at
$x = 0.03$ for $b = 0, 0.2, 0.5$: measured 0.999930 / 0.999943 / 0.999955.

Recon §5.2's resonance figure: $B(-1) = -6.214780789768604$ at $b = 0.2$ against the recon's
$\mp6.2147807897686$, and $C(+1) = +6.214780789768601$, agreeing to 1e-15 — so
$B(-1) = -C(1)$ holds numerically as well as in closed form. $A(-1) = 0$ exactly for $b>0$.

Recon §6.4's Bessel-correction figures: $\lvert$(4.12) $-$ target$\rvert$/envelope on
$(0.909, 1.091)$ at $b = 0.2$ is 2.00e-03 at $x = 50$ falling to 9.90e-07 at $x = 3200$, against the
recon's *"$2\times10^{-3}$ at $x = 50$ falling to $10^{-6}$ at $3200$"*.

### Independent check of the whole chain, run in scratch and not landed

Before writing any test, `I_quadrature` was scored against a **separate** quadrature of the review's
*defining* integral $\int_0^x G_x(x,\bar x)f(\bar x)\,d\bar x$, built only from (4.7) `eq:hgreen`
and (4.9) `eq:fsimple` and sharing no code with `domenech.py` — the object recon §0 calls **E**.
They agree to **6.2e-16 to 6.6e-15** relative at $x = 30$ for $b \in \{-0.3, 0, 0.2, 0.5\}$ on four
$(u,v)$. That is the step (4.9) $\to$ (4.11)'s weight $\bar x^{1/2-b}$ and the corrected (4.10),
checked directly. It is not landed, because test 5 covers the same chain against an object this
campaign did not write and is the stronger check; it is recorded here because campaign README §5.1
asks for the paper's own defining integral and this is it.

### New-agent behaviours worth stating

- The exponent of the resonance approach is asserted, not a tolerance:
  $\lvert I_{\rm target}(\delta) - I_{\rm target}(0)\rvert \propto \lvert\delta\rvert^b$ from both
  sides, measured at $100^{-b}$ per two decades of $\delta$ to within 8% at $b \in \{0.2, 0.5, 0.8\}$
  (0.3981, 0.1000, 0.0251 predicted; 0.3981, 0.1000, 0.0258 worst measured).
- At the resonance the exact kernel is finite with declared error 4.4e-16, and its four consecutive
  first differences on a $10^{-6}$ spacing in $\delta$ agree to 2e-04 of each other — no kink.
- Also at the resonance, the target's accuracy degrades as $b$ falls, exactly as recon §5.1 implies:
  $\lvert$target $-$ exact$\rvert$/envelope at $x = 400/1600/6400$ is 4.5e-01 / 3.4e-01 / 2.5e-01 at
  $b = 0.2$ but 4.3e-02 / 1.4e-02 / 4.5e-03 at $b = 0.8$, where it recovers the ordinary $1/x$.
  **No fixture case is near the resonance** (recon §5.4), so nothing here is scored there; this is
  campaign README §7 **D6**'s subject.

---

## Observations not acted on

1. **Recon §9's off-cut brief is ill-conditioned at large $\tilde y$.** Deviation 1 above.
   **→ §3 issue `[01-recon-off-cut-closed-form-is-ill-conditioned]`.**

2. **Recon §7 prints "$\mathcal N = \tfrac{3\pi}{8}$" where its own §2.2 definition gives
   $\tfrac{3\pi^2}{8}$.** §2.2 sets $\mathcal N \equiv \pi4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}$,
   which at $b = 0$ is $\pi\cdot\frac\pi4\cdot\frac32 = \frac{3\pi^2}{8}$. §7's $\frac{3\pi}{8}$ is
   the same constant **without** the $\pi$ — i.e. this module's `kernel_constant(b)`, not
   $\mathcal N$. Harmless: §7's displayed reduction to $\tfrac98\times$ KT eq. (25) is correct as
   written and was reproduced to rounding here, so the slip is in the symbol and not in any
   arithmetic that uses it. But a reader hand-checking §7 against §2.2 will lose a factor of $\pi$.
   **→ §3 issue `[01-recon-section-7-N-symbol-drops-a-pi]`.**

3. **`legenp` fails at $b = \tfrac12$ too, not only `legenq`.** Recon §3.4 records
   `legenq(2.5, -0.5, y, type=2)` raising `hypsum() failed to converge`; so does
   `legenp(2.5, -0.5, -0.999, type=2)`, at `mp.dps = 30` with mpmath 1.3.0. The consequence is the
   same and the recon's instruction ("special-case $b = 0, \pm\tfrac12$ and never route them
   through `legenq`") already covers it — the module never calls either at those $b$, and
   `test_special_values_at_b_zero_and_half` covers them from the papers' closed forms instead. Not
   opened as an issue: it widens a recorded fact rather than contradicting one, and nothing depends
   on it.

4. **`scipy.quad` emits `IntegrationWarning` on individual sub-intervals at $x\gtrsim10^3$.** Both
   "roundoff error is detected" and "extremely bad integrand behavior" fire on pieces whose own
   integral happens to be near zero relative to `epsrel = 1e-13`. **`kohri_terada.I_RD_quadrature`
   does the same at the same $x$** (13 warnings at $(v,u,x) = (1.2, 1.3, 1600)$), with the same
   breakpoint construction and the same tolerance pair, so this is the established behaviour of that
   construction rather than something this module introduced. It matters only through
   `declared_error`, which the warnings say may be *underestimated* — and the declared error is
   demonstrably not underestimated where it is used: the $b = 0$ tie lands at 4.0e-15 inside a
   5.5e-14 bound and $N$ at 4.1e-13 inside bounds of 1.0e-12 to 1.5e-11. Not opened: it is a
   property of `kohri_terada.py` as much as of this module, this prompt does not edit that file,
   and no measurement here is affected.

5. **Several of the papers' own reduced forms are the ill-conditioned way to write their function.**
   Deviation 5's first bullet. This is a property of `1912.tex:804-822`, not of the tree; it is
   recorded in `test_special_values_at_b_zero_and_half`'s docstring so that a later reader who
   re-derives those references in double precision knows why they disagree.

6. **The campaign README's "no board yet" prose is now stale.** `prompts/handover/README.md` §3 says
   prompt 01 is *"written 2026-09-19, not run"*, and `docs/OPEN_ISSUES.md` §1.1 says the campaign has
   *"no prompt written and no board yet"*. Both are superseded by this commit. The index's §1.1 and
   header prose are corrected here because the maintenance rule requires the index to be right; the
   README's §3 row is **not** edited, because it is the record of what was asked and the board is
   now the place that says what landed. **Not an issue** — the board's §1 carries the status.

---

## State handed to the next prompt

- **Module:** `ComputeTargets/tests/domenech.py`. Import it as
  `from ComputeTargets.tests.domenech import ...`. The signatures are in "What shipped".
  `I_quadrature` returns **`(value, declared_error)`**; `I_target` and `I_asymptotic` return a
  float and **raise `OutsideGervoisNavelet`** for $c_s\lvert u-v\rvert\ge1$.
- **The instrument to score against is `I_quadrature`, not `I_target`,** at any $x$ below
  $\sim10^5$. Breakage 3 above prices the mistake: $N$ drifts over a range of 5.06 on the nine
  fixture cases. `I_target` is for large-$x$ regression; at $x\sim10^7$ (campaign README §2 (j))
  it is sharp everywhere.
- **`I_quadrature`'s own ceiling is plain quadrature's.** `KOHRI-TERADA-ORACLE.md` §8 Table 8.2 has
  `scipy.quad` of the analogous eq. (15) failing above $x\approx3\times10^4$: the breakpoint list
  caps at 4,000 and quad's 200-subdivision limit runs out, with a declared error of order
  $\lvert I\rvert$. It reports the failure — `declared_error` is how a caller sees it — but **A2,
  D2 and E2 must not assume `I_quadrature` is usable at production $x$.** Above that, `I_target`
  with the outer Bessels taken from the repository's own two-region machinery is the instrument,
  and a Levin evaluation of (4.11) is campaign README §7 **D6**.
- **Cost:** `I_quadrature` is 0.02 s at $x = 100$, 0.06 s at 400, 0.22 s at 1600, 0.82 s at 6400 on
  one core (Apple M1 Pro). `total_from_I_rev` is two such calls. The whole new test module is 19 s.
- **$N(b)$ is now measured, not predicted:** $-1.194214876033$ at $b = 0.2$, constant over the nine
  fixture cases to 5.3e-13, equal to $-\frac{(3+2b)^2}{2(2+b)^2}$ to 4.1e-13. Against
  $I_{\rm rev}$ it is $-1$ to 3.5e-13. **Campaign README §6 acceptance item 3 is met.**
- **What is still not covered.** Varying $w$ (campaign README §0.3 and §2 (g) — every oracle here
  is constant-$w$); the resonance for $b\le0$ (**D6**); $y>1$ (deviation 2); the oscillation average
  (4.14) `eq:kernelaverage`, which recon §10 item 7 says needs only $A$, $B$, $C$ — all three are
  now in the tree and separately callable, so that is a short piece of work if a later prompt wants
  it; and the realistic flavour at large $x$, which is **A2**.
- **Fixture kinematics at $b = 0.2$**, for any prompt that needs to know which branch a shape is on:
  `together` $(0.909, 1.091)$ $y = -1.2521$ off-cut; `T-first` $(10, 12)$ $y = +0.9979$ on-cut;
  `q-smooth` $(0.01, 1.1)$ $y = -149.5409$ off-cut. At $b = 0$, `T-first` is $y = +1.0042$ and is
  refused.
- **Board created.** `prompts/handover/IMPLEMENTATION_STATE.md` now exists, with prompt 00 recorded
  as landed-without-a-log (by its own §8) and rows for the ten groupings of README §3. Prompt 03
  still owns opening the two §2 (o) issues and the one §2 (p) issue formally; they remain
  *"none yet"* in `docs/OPEN_ISSUES.md` §1.1 and are **not** moved onto this board by this prompt.
