# Prompt 09 — Propagate the error estimate to callers

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §3.4; §5 recommendation 15
**Depends on:** prompt 04 (the error components you are propagating)
**Files you may touch:** `LiouvilleGreen/three_bessel_integrals.py`, `ComputeTargets/QuadSourceIntegral.py`,
`LiouvilleGreen/tests/test_3bessel_analytic.py`, `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`,
plus the log and the status board.

---

## Character of this commit

**Source-incompatible.** It changes the return type of `quad_JJJ` and `quad_YJJ` and threads an error
estimate through code that currently discards it. It is the first prompt in the campaign to touch
files outside `AdaptiveLevin/`.

It is also the only prompt whose *value* depends on work that is out of scope (README §6). Read the
scoping note below before starting.

---

## Why this matters

Audit §3.4: `three_bessel_integrals.py` combines four phase groups as `(−G₁ + G₂ + G₃ − G₄)/4` and
keeps only `value` (`:208`). **Because that combination is cancellative, the relative error of the
total is amplified by the cancellation factor**, and only the summed `abserr` can reveal it. The same
argument applies to the `J`/`Y` recombination and to `QuadSourceIntegral`'s eight-way structure.

A caller that gets four values of size 1 which cancel to 10⁻⁶, each carrying an absolute error of
10⁻¹², has a result with a relative error of 10⁻⁶ and no way to know.

## Scoping: read this before starting

> **Correction to the audit.** Recommendation 15 names `quad_JJJ`/`quad_YJJ`. Those two functions
> have **no production caller** — only `LiouvilleGreen/tests/test_3bessel_analytic.py` (four sites)
> and `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`. The production consumer of Levin
> quadrature is **`ComputeTargets/QuadSourceIntegral.py`**, which calls `adaptive_levin_sincos`
> **directly at nine sites** (`:473`–`:611`, `:1013`) and discards `abserr` at every one.
>
> Do both. Treat `QuadSourceIntegral.py` as the load-bearing one.

> **Honesty about what this buys today.** The propagated `abserr` will be dominated by whatever
> prompt 04's floor reports. On the three-Bessel path the *real* accuracy limit is the phase and
> modulus splines — measured at a uniform 2×10⁻⁸ relative floor across all seven oracles, and
> independently corroborated by `three_bessel_integrals.py:12-25` — which the quadrature cannot see.
> Prompt 04 added an optional `theta_abserr` for exactly this, and nothing supplies it (README §6).
>
> So this commit builds the pipe. It will carry a truthful *quadrature* error and an incomplete
> *total* error until `phase_spline` learns to report its own fit accuracy. **Say this plainly in
> the docstrings you write, and in the log** — a caller who trusts a propagated `1e-15` on a path
> whose real floor is `2e-8` is worse off than one who had no error bar at all and knew it.
>
> If, on reflection, you judge that shipping a knowingly-incomplete error bar is worse than shipping
> none, **stop and put that argument in the log and a §3 issue** rather than proceeding. It is a
> legitimate position and the campaign would rather have it argued than have the pipe built and
> quietly mistrusted.

---

## What to do

### 1. `three_bessel_integrals.py`

- `_Levin_3bessel` (`:159-212`) sums four groups. Accumulate `data["abserr"]` alongside
  `data["value"]`, combining the error **linearly** (not in quadrature) for the same reason the
  module itself sums regions linearly (`:1166-1172`): the four groups share a phase construction, so
  a systematically inaccurate phase produces a common drift and the contributions are not
  independent. Apply `|combination[index]|` and the `norm_factor`.
- Also carry `converged` (prompt 01), `phase_limited`, and the components from prompt 04 if you can
  do so without an unwieldy return type. **`converged` is the one a caller most needs**: an
  aggregate that missed its request is actionable in a way that a raw number is not.
- `_direct_JJJ`/`_direct_YJJ` (`:78-108`, `:288-318`) use `simple_quadrature`, which returns its own
  `abserr`. `quad_JJJ` returns `numeric + Levin` (`:75`); propagate both halves' errors.
- **Return type.** `quad_JJJ`/`quad_YJJ` return bare floats today. A dict matches
  `adaptive_levin_sincos` and `simple_quadrature`; a `NamedTuple` is friendlier to unpack and static
  analysis. Pick one, use it for both, and update all six call sites (four in
  `test_3bessel_analytic.py`, two via the `ORACLES` table in `bessel_tier.py:68-69`).

### 2. `ComputeTargets/QuadSourceIntegral.py`

Nine call sites. Trace what happens to each value and propagate the error along the same path.
Pay attention to:

- The `J`/`Y` pairs (`J1_data`/`Y1_data`, …). Each pair is `f = [Levin_f, 0]` against
  `f = [0, -Levin_f]` on the *same* phase, so they are the sine and cosine parts of one complex
  integral and their errors are correlated. Linear combination again.
- The `:1013` call is structurally different (a single Green's-function integral, not a
  four-group decomposition). Handle it on its own terms.
- **Where the error should end up.** Look at what `QuadSourceIntegral` stores. If there is an
  existing metadata or tolerance field the error belongs in, use it; if storing it means a schema
  change, **do not make one** — that is a Datastore change well outside this campaign's scope.
  Compute and log the error, note in the log where it would need to go to be persisted, and open a
  §3 issue. Check `ComputeTargets/QuadSourceIntegral.py`'s storable-class definition before deciding.

### 3. Tests and the benchmark harness

- `test_3bessel_analytic.py` calls `quad_JJJ`/`quad_YJJ` at `:320`, `:365`, `:410`, `:501`. Update
  for the new return type. **Then use it:** add an assertion that the reported `abserr` bounds the
  actual error against the analytic oracle. That is the audit's C12 item "any check of the returned
  `abserr` against the true error", applied where the oracles already exist, and it is the single
  most valuable test in this commit.
  **Expect this assertion to be interesting.** If the reported `abserr` does *not* bound the true
  error against the closed form, that is the phase-spline floor showing through — the scoping note
  above predicts exactly this. Do not weaken the assertion to make it pass. Record the measured
  ratio, mark the test `skip`/`expectedFailure` with a comment pointing at the `theta_abserr`
  follow-up, and open a §3 issue.
- `bessel_tier.py:68-69` builds `ORACLES` from the two functions. Update it, and have it record the
  propagated error alongside the measured one — the harness exists to compare reported against true,
  so this is a natural fit.

---

## Do not

- Do not change `Datastore` schemas.
- Do not touch `phase_spline.py`, `bessel_phase.py` or `range_reduce_mod_2pi.py`. README §6.
- Do not change `AdaptiveLevin/levin_quadrature.py` at all. If you find you need to, that is a gap in
  prompt 04 — open a §3 issue and work around it, or stop.
- Do not combine errors in quadrature. See step 1 for why.
- Do not weaken a failing `abserr`-bounds-truth assertion to make the suite green.

---

## Verification

1. `AdaptiveLevin/tests/` passes; `LiouvilleGreen/tests/` passes (or fails only where step 3
   predicted, with the failure documented).
2. **All six `quad_JJJ`/`quad_YJJ` call sites updated.** `grep -rn "quad_JJJ\|quad_YJJ"` across the
   repo and confirm every one matches the new signature.
3. **The propagated error is right in size.** For at least three analytic oracles, report the
   propagated `abserr`, the true error against the closed form, and their ratio. State whether the
   propagated value bounds the truth, and if not, by how much and why.
4. **The cancellation is visible.** Report the four group values, their individual errors, and the
   combined result with its error, for one configuration. The point of this commit is that a reader
   can see the cancellation factor; show it.
5. `QuadSourceIntegral.py` imports cleanly and its nine call sites are consistent.
6. The benchmark harness runs.

---

## Finish

1. Write `prompts/levin-refactor/logs/09-caller-propagation.md`. Under *Numerical evidence*, items 3
   and 4. **The scoping note's honesty requirement is the most important thing in this log**: state
   plainly what the propagated number does and does not include, and what would have to change
   (`phase_spline` reporting its fit accuracy) for it to be complete. Record the return-type choice,
   where the `QuadSourceIntegral` error ends up (or does not), and the outcome of the
   `abserr`-bounds-truth assertion including any `skip`.
2. Update `IMPLEMENTATION_STATE.md`.
3. Commit in one commit. Suggested message:

```
Carry the Levin error estimate out to the three-Bessel callers

quad_JJJ and quad_YJJ combine four phase groups as (-G1 + G2 + G3 - G4)/4 and
kept only the value. That combination is cancellative, so the relative error
of the total is amplified by the cancellation factor and only the summed
absolute error can reveal it: four group values of order one cancelling to
1e-6, each accurate to 1e-12, give a result whose relative error is 1e-6 with
nothing in the return value to say so.

Both functions now return the estimate alongside the value, together with the
convergence flag, and QuadSourceIntegral's nine direct calls propagate the
same. Errors are combined linearly rather than in quadrature, for the reason
the quadrature itself sums regions linearly: the groups share a phase
construction, so an inaccurate phase produces a common drift and the
contributions are not independent.

The analytic three-Bessel tests now assert that the reported error bounds the
measured error against the closed form.

What this number does not yet include: the accuracy of the phase and modulus
splines themselves, measured at a uniform 2e-8 relative floor across the seven
oracles and invisible from inside the quadrature. The quadrature accepts an
optional theta_abserr for exactly this, and nothing in LiouvilleGreen supplies
one yet.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
