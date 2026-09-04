# Prompt 05 — Make `atol` a global tolerance

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §1.4 fix 2 (C3); §5 recommendation 8
**Depends on:** prompt 04 (`phase_limited` arms on `phase_err > atol`; the floor must be settled first)
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_quadrature.py`,
plus the log and the status board.

---

## Character of this commit

A one-line change to the acceptance test that makes `atol` mean what a caller assumes it means, plus
the consequential tidying. The audit's measurement is that it is **free at matched delivered
accuracy** — a relabelling rather than a cost.

Prompt 01 made the module *report* that the aggregate missed the request. This makes the aggregate
meet it.

---

## The defect

Acceptance at `levin_quadrature.py:1083` is `abserr < atol or relerr < rtol` **per region**.
`abserr_total` (`:1173-1180`) is the sum over accepted regions, so the delivered absolute error
scales like `N_regions × atol`.

The paper has the same structure — their step 4 uses a per-interval `ϵ` — and gets away with it
because they run at `ϵ = 10⁻¹³` with few intervals. **For a library API this is a broken contract.**

**Reproduced at HEAD.** `∫₀³ e^{−400(x−1.1)²} sin(3×10⁴x) dx` at `atol = 1e-10`:

```
reported abserr = 1.26e-10        (exceeds the requested atol; audit's delivered error: 1.27e-10)
```

At `atol = 1e-12` the same problem reports `6.22e-13` and does not exceed. So it is not a systematic
factor — it is a region-count-dependent overshoot, which is exactly what makes it hard for a caller
to reason about.

---

## What to do

Accept when

```
abserr < atol * (b - a) / (b0 - a0)
```

where `(b0, a0)` is the **original** `x_span`. This makes `Σ abserr < atol` true by construction,
because the length fractions sum to one.

**Measured cost** (audit §1.4, `e16_scaled.py`, five problems × three tolerances): this is **free at
matched delivered accuracy**. `local-h` at `atol = 10⁻¹²` and `local-h-scaled` at `atol = 10⁻¹⁰`
produce identical evaluation counts, solve counts, region counts and errors on three of the five
problems. At fixed nominal `atol` it costs 1.1–1.5× more evaluations and buys 10–100× accuracy.

### Everything else that reads `atol` per region

`atol` appears in five places inside the driver loop. Each needs a decision, and **each decision
needs recording** — this is the part of the prompt where drift is most likely, because four of the
five look like they should obviously follow the acceptance test and one of them probably should not.

1. **The acceptance test** (`:1083`) — scaled. This is the change.
2. **The relative-error denominator floor** (`:1059`,
   `relerr_denom = max(min(|estimate|, |refined|), atol)`). This floor exists to stop pointless
   subdivision at an accidental zero of a region's contribution (the comment at `:1052-1058`
   explains it, and it is correct). A *local* atol keeps that reasoning intact and self-consistent
   with the acceptance test. **Recommended: scale it.** But note the consequence — a very short
   region gets a very small floor, which re-admits some of the behaviour the floor was added to
   prevent. Check that the `atol = 1e-15` / identically-zero-integrand case from prompt 01's item 4
   still terminates in 1 region / 3 solves, and say so.
3. **The `phase_limited` conditions** (`:1075-1077`, `phase_err > atol and phase_err > rtol *
   relerr_denom and abserr <= phase_err`). This is the guard that closes the tolerance trap. It
   should use the same local atol as the acceptance test, or the guard fires at a different
   threshold from the test it is guarding. **Recommended: scale it.**
4. **The aggregate relative error** (`:1183`, `relerr_total = abserr_total / max(|val|, atol)`).
   This is computed once, over the whole integral, after the loop. **It must use the global `atol`.**
   Scaling here would be a category error.
5. **`converged`** (added by prompt 01, item 2). Also global, also after the loop. Unchanged.

### Consequential updates

- **The `converged` flag should now almost always be true** for a run that terminates normally. That
  is the point, and it is a good cross-check: if it is still false on the C3 reproducer after this
  commit, the scaling is not doing what you think.
- **The docstring** (prompt 01, item 6) says `atol` is a per-region tolerance and that this is
  scheduled to change. Update it. State plainly that `atol` is now a bound on the summed absolute
  error, and that `rtol` remains a **per-region** relative test — because it does, and the asymmetry
  will confuse someone if it is not written down.
- **`rtol` is deliberately not scaled.** A relative tolerance is not additive over regions the way an
  absolute one is, and there is no length-proportional analogue that means anything. Say so in the
  docstring rather than leaving the asymmetry unexplained.
- **Reversed spans.** `x_span` may have `b < a` (audit §1.1 verifies this works and returns the
  correctly negated value). Use `|b − a| / |b0 − a0|` so the fraction is positive. A zero-width span
  returns 0.0 before the loop; confirm the scaling cannot divide by zero.

---

## Do not

- Do not change the acceptance test's *structure* (`abserr < ... or relerr < ...`). Only the
  absolute threshold changes.
- Do not implement global worst-first error balancing. Audit §2.3 measured it and it is **unsound**
  with the step-(4) residual: the residual is a difference of two rules, not a per-region bound that
  decreases under refinement, so a heap driven by it need not converge — 8 192 regions and 827 353
  evaluations against 3 665 for the local scheme. The audit is explicit that the contract benefit
  which motivated it "is obtained far more cheaply by §1.4's length-proportional tolerance, which is
  free at matched accuracy" — i.e. by exactly this commit.
- Do not change the floor, the gate, or `p_use`.

---

## Verification

1. `AdaptiveLevin/tests/` passes.
2. **The contract holds.** On at least five problems at three tolerances each, `abserr_total <=
   max(atol, rtol·|val|)` and `converged` is `True`. Include the C3 reproducer at `atol = 1e-10`,
   which must now come in under `1e-10`.
3. **Cost at matched accuracy.** Reproduce the audit's key claim: `local-h` at `atol = 1e-12` and
   `local-h-scaled` at `atol = 1e-10` should give comparable evaluation counts, solve counts, region
   counts and delivered errors. Report both, side by side, on at least three problems. If you cannot
   reproduce "free at matched accuracy", say so with numbers — the recommendation rests on it.
4. **Cost at fixed nominal `atol`.** Same five problems at the same `atol`, before and after:
   evaluations, solves, regions, delivered error against a closed form. The audit's expectation is
   1.1–1.5× more evaluations for 10–100× accuracy.
5. **The degenerate cases still terminate.** Identically-zero integrand at `atol = 1e-15`,
   `depth_max = 8`: still 1 region / 3 solves (this is the prompt 01 item 4 check, and item 2 above
   is why it needs re-running).
6. **Reversed span** still returns the correctly negated value.
7. **Three-Bessel oracles**, before and after: value, `abserr`, region count.

---

## Finish

1. Write `prompts/levin-refactor/logs/05-global-tolerance.md`. Under *Numerical evidence*, items 2–4
   and 7. **Every one of the five `atol` sites above must appear in the log** with what you did and
   why — including the ones you left global. That list is the whole content of this commit and a
   later reader needs it without reading the diff.
2. Update `IMPLEMENTATION_STATE.md`. **C3 closes here** (prompt 01 closed its reporting half).
3. Commit in one commit. Suggested message:

```
Distribute atol across subintervals so it bounds the total error

The step-(4) acceptance test compared each region's resolution residual
against the caller's atol directly, so the delivered absolute error scaled
with the number of regions rather than being bounded by what was asked for.
Chen et al. have the same structure and get away with it because they run at
a fixed 1e-13 with few intervals; for a library API it means atol does not
mean what a caller assumes.

Measured on int_0^3 exp(-400 (x - 1.1)^2) sin(3e4 x) dx at atol = 1e-10, the
reported and delivered errors were both about 1.27e-10 -- over the request,
with nothing to say so. The same problem at atol = 1e-12 came in under, so
the overshoot is region-count dependent rather than a fixed factor, which is
what makes it hard to reason about.

A region is now accepted when its residual falls below atol scaled by its
share of the original interval length, which makes the sum over regions
bound atol by construction. The relative-error denominator floor and the
phase-limited guard are scaled the same way, so all three thresholds inside
the loop agree; the aggregate relative error and the converged flag are
computed once at the end and stay global.

This is close to free. Measured across five problems at three tolerances,
the scaled scheme at atol = 1e-10 produces the same evaluation, solve and
region counts and the same delivered error as the unscaled scheme at
atol = 1e-12 on most of them: it is a relabelling that makes atol mean what
a caller assumes it means.

rtol is deliberately left as a per-region test. A relative tolerance is not
additive over regions and has no length-proportional analogue.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
