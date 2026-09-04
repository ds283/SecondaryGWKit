# Prompt 04 — Replace the endpoint phase floor with the paper's round-off bound

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §3.1, §3.2, §3.3 (C4), §3.4; §5 recommendations 5 and 10
**Depends on:** prompt 03 (the region taxonomy — Levin cells and Clenshaw–Curtis cells — must be settled)
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_quadrature.py`,
plus the log and the status board.

---

## Character of this commit

This changes **every error bar the module reports**. It does not change any returned value except
through the acceptance test — a different floor arms `phase_limited` at a different threshold, which
changes how much subdivision happens, which changes the value at the last digit or two.

It closes C4, the last of the audit's critical-and-high findings.

---

## The defect (C4)

`levin_quadrature.py:521` sets `theta_scale = TWO_PI` whenever `theta_mod_2pi` is supplied, on the
reasoning that a range-reduced phase hands `sin`/`cos` an `O(2π)` argument.

**That reasoning is incomplete, and the measured consequence is severe.** Reproduced at HEAD on
`∫_{1/3}^{7/3} e^{−x} sin(ωx) dx` against its closed form, `atol = 1e-18`, `rtol = 1e-13`:

| ω | raw: true err | raw: reported | reduced: true err | reduced: reported | reduced is optimistic by |
|---|---|---|---|---|---|
| 10⁴ | 1.4×10⁻¹⁷ | 1.90×10⁻¹⁶ | 7.7×10⁻¹⁸ | 7.6×10⁻¹⁸ | 1.0× |
| 10⁶ | 1.0×10⁻¹⁶ | 1.90×10⁻¹⁶ | 4.5×10⁻¹⁸ | 8.0×10⁻²⁰ | 56× |
| 10⁸ | 1.1×10⁻¹⁶ | 1.90×10⁻¹⁶ | 4.1×10⁻¹⁷ | 1.3×10⁻²¹ | 3.2×10⁴ |
| 10¹⁰ | 2.3×10⁻¹⁷ | 1.90×10⁻¹⁶ | 1.8×10⁻¹⁷ | 1.6×10⁻²⁴ | 1.1×10⁷ |
| 10¹² | 1.3×10⁻¹⁶ | 1.90×10⁻¹⁶ | 9.9×10⁻¹⁷ | 1.2×10⁻²⁶ | **8.1×10⁹** |

The raw path is flat in ω and never exceeded — exactly what the paper predicts in prose (v3, p. 30).
The reduced path is optimistic by the oscillation count. Note also that **the reduced path is not
more accurate than the raw path** (both sit at ~10⁻¹⁶ absolute); it just claims to be.

## Why the obvious diagnosis is wrong

"The product `ωx` is formed in binary64 before reduction, so `dθ ≈ eps·|θ|`" is only half the story.
The audit repeated the experiment with an **exactly** reduced phase (residue computed in 60-digit
arithmetic from the double-precision `x`, then rounded) and the delivered error was essentially
unchanged — `1.0e-18` against `3.9e-18` at ω = 10⁶, `8.6e-17` against `9.7e-17` at ω = 10¹².

So the residual floor is **not endpoint-phase rounding at all**. It is the round-off floor that
Chen et al. derive, eq. (151):

> `|I₁ − I| ≲ ε (1 + G₁/G₀ + max{G₁, k²}/G₀) · |W| min(1, 1/|W|) · ‖f‖_{L∞}`

With `G₀ = min|g′|`, `G₁ = max|g′|` on the rescaled interval, the bracket tends to a small constant
when `g′` is roughly constant and large, and `|W| min(1, 1/|W|) ≤ 1`, leaving `≈ ε‖f‖_∞`. For the
table above, `ε·‖f‖_∞·(h/2) = 2.2×10⁻¹⁶ × 0.717 × 1 = 1.6×10⁻¹⁶` — matching every measured value.

This also *derives* the earlier campaign's empirical `eps·θ_max` relative bound, and explains why the
raw-path endpoint bound works numerically: `Σ|p| ≈ 2‖f‖_∞/θ′` and `theta_scale ≈ θ′b`, so
`eps·θ_scale·Σ|p| ≈ 2b·eps·‖f‖_∞` — the same quantity up to a geometric factor.

## Why C4 is scored high rather than medium

`phase_limited` (`:1075-1092`) is the branch that closes the "tolerance trap", and its first
condition is `phase_err > atol`. Understating `phase_err` by the oscillation count pushes the
threshold at which the guard arms down by the same factor. Measured (audit §0.2, §3.3) on
`∫₁¹⁰⁰ sin(10³x)/x dx`: the raw path arms it from `atol ≈ 10⁻²²` downwards, the reduced path only
from `atol ≈ 10⁻³⁰`, and the reduced path pays ~2× the regions and solves in between. The window is
widened by roughly eight decades of `atol` **on exactly the path `three_bessel_integrals.py` and
`QuadSourceIntegral.py` use** (standing note 7).

---

## What to do

### 1. Implement eq. (151) as the round-off floor

```
round_off_err = C_roundoff · eps · max_grid|f| · (h/2) · (1 + G1/G0 + max(G1, k²)/G0)
```

Everything it needs is already sampled, so it costs no extra evaluations of anything.

**Four things that will silently produce a wrong bound if you get them wrong:**

**(a) `G₀` and `G₁` are on the *rescaled* interval `[-1, 1]`.** So `G₀ = min|θ′|·(h/2)` and
`G₁ = max|θ′|·(h/2)`, where `h = |b − a|`. The ratio `G₁/G₀` is scale-invariant and will look right
either way; **`max(G₁, k²)/G₀` is not**. Getting this wrong gives a bound that is right on some
problems and wrong by `(h/2)²` on others.

**(b) `G₀ = min|θ′|` can be exactly zero** — at a stationary point of the phase, which is precisely
the C2 case prompt 03 exists to handle. The bound then evaluates to `inf`. Decide what to do
(clamp `G₀` from below? fall through to the step-(4) residual? report the region as unbounded and
force bisection?) and **state the decision and its consequence**. Note the interaction with prompt
03: after that commit a region with an interior stationary point and small total variation goes to
Clenshaw–Curtis, where eq. (151) does not apply at all — so the Levin branch only sees `G₀ ≈ 0` for
a region with *large* total variation and an interior turning point, which is a real and reachable
configuration.

**(c) `max_grid|f|` for a two-component `f`.** The paper's `‖f‖_∞` is for the scalar problem. After
prompt 02 the natural reading is `max |f₁ + i f₂| = max sqrt(f₁² + f₂²)` over the grid. Choose,
justify, and say what the alternative (`max_i max_grid |f_i|`, or the sum) would have given on your
validation set.

**(d) `k` is the Chebyshev collocation order** (`chebyshev_order`), not the region index or the wave
number. This creates a dependency: **prompt 08 raises the default order, which changes every
reported floor through the `max(G₁, k²)` term.** Prompt 08 is sequenced after this one for exactly
that reason.

**Measured validity** (audit §3.3, `e29_paperbound.py`): across 26 resolved cells — three problem
families × five decades of ω × two orders, excluding four `sinc` cells that are under-resolved as a
single region where the step-(4) residual and not round-off dominates — the worst true/bound ratio
is **0.22 with `C = 1`**, against **1.24** for the endpoint model. It is both valid and tighter, it
is the paper's own bound rather than a local invention, and **it is independent of whether the phase
is range-reduced**, which removes C4 entirely.

*Caveat, from the audit and important:* it bounds **round-off only**. Where the region is
under-resolved the step-(4) residual dominates, and `total_err = max(residual, round_off_err)` is
still the right aggregate. Keep that structure — it is what `used_interval.total_err` (`:271-277`)
already does.

**A further argument the audit does not make, measured during this campaign's planning:**
eq. (151) **aggregates gracefully under subdivision and the endpoint model does not.** Summing the
eq. (151) bound over a cell decomposition of `∫_{1/3}^{7/3} e^{−x} sin(ωx) dx` gives a total that is
invariant under the number of cells (it is `≈ 3·eps·‖f‖_∞·(h_total/2)` once `G₁ > k²`), whereas the
raw endpoint bound summed over 400 cells is 400× larger — `2.5e-14` against `9.3e-17`. So the
current floor makes a heavily-subdivided integral look *less* accurate the harder the driver works
on it. (Spot check: 10 cells of one problem family; eq. (151) valid, worst true/bound ratio 0.001.
This is a sanity check, not a reproduction of the audit's 26-cell validation.)

### 2. Retire the endpoint model as the default, keep it for a declared phase error

`_phase_error(theta_scale, p_endpoint_l1)` (`:126-160`), `theta_scale` (both branches, `:503-528`)
and `_Basis_SinCos.phase_scale` (`:540-549`) all exist to serve the endpoint model. The derivation in
`_phase_error`'s docstring is **correct** (audit §3.2 verifies it, and notes it is conservative by up
to √2 since the true sensitivity is `sqrt(p₁² + p₂²)`, not `|p₁| + |p₂|`). What is wrong is
*inferring* `dθ` — as `TWO_PI` or as `max|θ_Cheb|` — rather than being told it.

So:

- **Remove `theta_scale` and the `TWO_PI` special case.** Recommendation 5.3 says to retain `TWO_PI`
  "only when the phase function declares that it reduces exactly. Never as a default inferred from
  the mere presence of `theta_mod_2pi`." Since eq. (151) supersedes it and no caller declares
  anything, the simplest correct action is to delete the inference. If you keep a declared-exact
  path, it must be opt-in and it must be documented.
- **Keep `_phase_error` and drive it from a declared `theta_abserr`** (item 3). When no
  `theta_abserr` is supplied, the endpoint term is absent and the floor is eq. (151) alone.
- **Preserve the docstring's derivation** — it is correct, it explains why the step-(4) residual is
  blind to this error (common-mode between parent and children, `:145-150`), and that explanation is
  still needed. Rewrite the parts that describe the *inferred* `theta_scale`.

### 3. Add an optional `theta_abserr` to the phase contract

Recommendation 5.2. Accept an optional `theta_abserr` in the `theta` dict: a scalar, or a callable
of `x`, in **radians**. When present, the region's floor gains an endpoint term
`_phase_error(theta_abserr_at_endpoints, p_endpoint_l1)`.

Follow `_Basis_SinCos`'s existing convention for optional keys (standing note 11): `__init__` sets
the attribute only when the key is present, and consumers test with `hasattr`. Do not mix that with
an explicit-`None` default.

**Why this matters even though nothing supplies it yet.** This is the only way to represent the
error a *phase spline* introduces. `bessel_phase`'s construction accuracy is a property of the
spline and is invisible from inside the quadrature. The earlier campaign measured a uniform
**2×10⁻⁸ relative floor** on all seven three-Bessel oracles attributable to phase construction; the
module cannot currently express it and reports ~10⁻¹⁵ instead. **`three_bessel_integrals.py:12-25`
independently corroborates this** — its own retuning comment records that the delivered relative
error is "set by the accuracy of the phase and modulus splines, not by the spectral order".

Nothing in `LiouvilleGreen/` computes such a number today (`phase_spline.py` has no accuracy API), so
this parameter ships unused by production callers. That is expected and is recorded in README §6 as
the natural follow-up. **Document it clearly** so the follow-up is obvious, and add a test that
exercises the path with a synthetic `theta_abserr`.

### 4. The safety factors (recommendation 10)

The audit's recommendation 10 says to raise `_LEVIN_PHASE_ERROR_SAFETY` from 1.0 to 4–8, because
the *endpoint* bound was exceeded by 1.24× in one of 30 cells (§3.2, `e29_paperbound.py`, adding
`∫₋₁¹ cos(λ atan x)/(1+x²) dx` at λ = 10⁴…10¹²).

But recommendation 5 **replaces** the endpoint model with eq. (151), which has **4.5× margin at
`C = 1`** on the same validation set. A single shared constant cannot serve both.

**Introduce two separately-named constants**, and set each from the audit's own measurement for its
own model:

- `_LEVIN_ROUNDOFF_SAFETY` — the `C` in eq. (151). The evidence supports `1.0`; if you choose more,
  say what it buys.
- `_LEVIN_PHASE_ERROR_SAFETY` — retained, now applying only to the declared-`theta_abserr` endpoint
  term. The 1.24×-exceeded measurement applies to the *inferred* `theta_scale`, not to a declared
  `dθ`, so the case for 4–8 is weaker here than the audit's numbering suggests. State your reasoning.

Keep both comments honest about what they were measured against. The existing comment block at
`:94-108` contains a genuinely useful caveat about intervals where `θ` is exactly representable —
preserve whatever of it still applies.

### 5. Return the error components separately (§3.4)

The aggregate currently mixes a resolution residual, a round-off floor and `quad`'s own estimate
into a single scalar. Returning the components costs nothing and lets a caller see *why* they cannot
get more digits, which is the actionable information:

- `abserr_resolution` — the summed step-(4) residual;
- `abserr_roundoff` — the summed eq. (151) floor (plus any declared-`theta_abserr` endpoint term);
- `abserr_fallback` — the summed Clenshaw–Curtis nested-pair estimates from prompt 03's cells.

`abserr` stays as it is: the sum over regions of `max(...)`. These are **new keys**, so the
benchmark harness is unaffected (standing note 8). Add the same breakdown to `used_interval` so
per-region diagnostics carry it too.

### 6. A safety factor on the residual (§3.1)

Audit §3.1 records that the step-(4) residual "is a difference of two estimates, not a bound, and it
can under-report the collocation error" — measured 2.4×10⁻¹⁷ reported against a true 2.8×10⁻¹⁵, a
factor 120. It suggests "a safety factor on the residual (as QUADPACK applies to its own) would be
prudent".

This is a suggestion in prose, not a numbered recommendation, and it is backed by one measurement
against a same-core high-effort reference — which the audit's own §7 caveats say is legitimate for
comparing policies and not for establishing absolute accuracy. **Do not apply a residual safety
factor on that basis.** Instead: note it in your log as an observation, say what would settle it (a
factor-120 under-report measured against a *closed form*, not a same-core reference), and leave the
residual alone. If you believe the evidence is stronger than this reading, make the case in the log
and act on it — but do not do it silently.

---

## Do not

- Do not change the acceptance test or the tolerance semantics — prompt 05.
- Do not change `p_use`'s selection rule — prompt 06.
- Do not change the default Chebyshev order — prompt 08, and it depends on this commit through the
  `k²` term.
- Do not touch `phase_spline.py`, `bessel_phase.py` or `range_reduce_mod_2pi.py`. README §6: they are
  out of scope for this campaign and warrant their own audit.
- Do not remove the `phase_limited` flag or the warning at `:1213-1224`. Fixing the floor is what
  makes them arm at the right threshold; they become more useful, not less.

---

## Verification

1. `AdaptiveLevin/tests/` passes.
2. **C4 is fixed.** Re-run the ω-ladder table at the top of this prompt, both phase modes. The
   reported floor must (a) bound the true error in every cell, and (b) be **the same for the raw and
   reduced paths**, since eq. (151) does not depend on how the phase is presented. Report the full
   table.
3. **The `phase_limited` guard arms at the right threshold.** On `∫₁¹⁰⁰ sin(10³x)/x dx`, sweep `atol`
   downwards for both phase modes and report the `atol` at which the guard first arms, and the
   region/solve counts. Before this commit: raw arms from ~10⁻²², reduced only from ~10⁻³⁰, with the
   reduced path paying ~2× in between. After: they should agree.
4. **Validity of eq. (151) across a wider set.** At minimum: `∫_{1/3}^{7/3} f(x) sin(ωx) dx` for
   `f ∈ {e^{−x}, 1/x, 1}` at ω = 10⁴…10¹², and `∫₋₁¹ cos(λ atan x)/(1+x²) dx` at λ = 10⁴…10¹²
   (closed form `(2/λ) sin(πλ/4)`). Report the worst true/bound ratio. The audit's figure to beat is
   **0.22**; anything above 1.0 in a *resolved* cell is a failure and must be investigated, not
   papered over with a larger `C`.
5. **The subdivision-invariance property.** Show the summed floor for the same integral computed
   with a few regions and with many. It should be roughly flat, unlike the endpoint model.
6. **The `theta_abserr` path works.** A synthetic `theta_abserr = 1e-8` on a problem with a closed
   form must raise the reported `abserr` to ~`1e-8 × Σ|p endpoints|` and must not change the value.
7. **Three-Bessel oracles.** At least two, before and after: the reported `abserr` should now be much
   larger (the reduced path's optimism is gone) and the value should be essentially unchanged. If the
   value moves more than the previous `abserr`, investigate — the acceptance threshold has shifted
   and you need to know by how much.
8. **`grep -n "TWO_PI" AdaptiveLevin/levin_quadrature.py`** — any surviving use is deliberate and
   explained.

---

## Finish

1. Write `prompts/levin-refactor/logs/04-roundoff-floor.md`. Under *Numerical evidence*, items 2–7.
   Judgement calls that **must** be recorded: the `G₀ = 0` handling (step 1b); the `‖f‖_∞` definition
   for two components (step 1c); whether a declared-exact `TWO_PI` path was retained (step 2); both
   safety-factor values and their justification (step 4); and the residual-safety-factor decision
   (step 6). Also record, plainly, **how much the reported `abserr` moved on the production three-
   Bessel path** — a user who reads only one line of this log should learn that number.
2. Update `IMPLEMENTATION_STATE.md`. **C4 closes here**, and part of rec 11 (the `abserr` components).
3. Commit in one commit. Suggested message:

```
Report the round-off floor Chen et al. derive, not an endpoint phase model

The per-region accuracy floor was modelled as rounding of the phase at the
two region endpoints, eps * theta_scale * sum|p(endpoints)|, with theta_scale
inferred from how the phase was presented: max|theta| for a raw phase, and a
hardwired 2*pi whenever a range-reduced phase function was supplied, on the
reasoning that reduction hands sin/cos an O(2*pi) argument.

That reasoning is incomplete. Repeating the experiment with an exactly
reduced phase -- residue computed in 60-digit arithmetic and then rounded --
leaves the delivered error unchanged, so the residual floor is not endpoint
phase rounding at all. It is the round-off floor of Chen et al. eq. (151),
eps * ||f||_inf * (h/2) * (1 + G1/G0 + max(G1, k^2)/G0), which matches every
measured value.

The practical consequence of the old model was severe on exactly the path
production uses. Measured on int_{1/3}^{7/3} exp(-x) sin(w x) dx against its
closed form, the reduced-phase floor was optimistic by 56x at w = 1e6 and by
8.1e9 at w = 1e12, while the raw-phase floor stayed flat at 1.9e-16 and was
never exceeded. Because the phase-limited acceptance branch arms on
phase_err > atol, understating the floor by the oscillation count pushed that
guard down by the same factor, widening the window in which the driver
subdivides pointlessly by about eight decades of atol.

Equation (151) is valid with 4.5x margin where the endpoint model was
exceeded, needs no evaluations the solve does not already make, and is
independent of whether the phase is range-reduced. It also aggregates
sensibly: the summed floor is invariant under how finely the interval is
divided, where the endpoint model grew linearly with the region count.

The endpoint model is retained but is no longer inferred. A phase dictionary
may now declare theta_abserr, a scalar or callable in radians, which is the
only way to express the error a phase spline's own construction introduces --
measured at a uniform 2e-8 relative floor across the three-Bessel oracles,
and currently reported by this module as 1e-15. Nothing in LiouvilleGreen
supplies such a number yet.

The returned error estimate is also broken out into its resolution,
round-off and fallback components, so a caller can see which one is stopping
them from getting more digits.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
