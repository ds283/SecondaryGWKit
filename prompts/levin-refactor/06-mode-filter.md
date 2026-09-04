# Prompt 06 — Fix the `p_use` mode filter

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §1.5 (C5); §5 recommendation 9
**Depends on:** prompt 02 (the storage layout of `p` is what this indexes) and prompt 04 (the
`abserr` component structure this adds to)
**Files you may touch:** `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_quadrature.py`,
plus the log and the status board.

---

## Character of this commit

Two small fixes to a heuristic that is *defensible but not self-consistent*. Neither changes the
returned value by more than `rtol`; the point is that the change they remove is currently invisible
to the error estimate.

The third problem the audit lists under C5 — that this filter is the mechanism behind C1 — was
closed by prompt 01. **Do not undo it.**

---

## The heuristic

`levin_quadrature.py:764-789`. After the solve, modes whose relative amplitude is below `rtol` are
dropped from the endpoint sum, on the reasoning (from commit `af85ef2`) that they cannot be computed
accurately anyway — especially when associated with small singular values that the SVD is not
handling well — so they act as numerical noise that pollutes the `abserr` estimate and can prevent
convergence of the bisection step. That reasoning is sound.

## Problem (a): it gates on the wrong quantity

`p_ratios` is built from `mean|P[j,:]|` over the **collocation points** (`:757-759`), but what the
estimate consumes is `P[j, 0]` and `P[j, −1]` — the **endpoint** values (`:772-777`). A mode with a
small mean and a large endpoint value would be discarded wrongly.

The audit could not find a case where this actually bites: over 400 randomised spike-amplitude
problems the worst perturbation from the gate was 4.1×10⁻⁷ against `rtol = 10⁻⁷`, i.e. bounded by
`rtol` as intended. **So this is a correctness-of-reasoning fix, not a bug fix, and the log should
say so** rather than implying a defect was found in the field.

**The fix is free.** Gate on the quantity actually used:

```
ratio_i = (|p_i(a)| + |p_i(b)|) / Σ_j (|p_j(a)| + |p_j(b)|)
```

Note this is a *fraction of the total*, whereas the current `p_ratios` is a *ratio to the maximum*.
They differ by a factor of at most `m`. **Decide deliberately which normalisation to keep** — the
threshold `r > rtol` means something different under each — and record the choice. Keeping
ratio-to-maximum preserves the meaning of the existing threshold and of the `p_ratios` values
recorded in `used_interval` and printed by its `__str__` (`:210`), which is the conservative option.

`p_ratios` is also carried in `p_ratios_history` and printed by the depth-18 diagnostic at
`:900-925`. If you change its meaning, that diagnostic's output changes meaning too. Either keep the
meaning or update the label.

## Problem (b): it makes the returned value a function of `rtol` at fixed mesh

Measured (audit §1.5b), one region, order 16, mode ratios `[1.06×10⁻⁶, 1]`:

```
rtol = 1e-2, 1e-4      ->  value = 1.81743030831524e-07
rtol <= 1e-6           ->  value = 1.81736473688358e-07     (relative jump 3.6e-5)
```

The jump is bounded by `rtol`, so it is defensible. **But it is invisible to `abserr`**, because
parent and children use identical gating and the discarded amount is common-mode in the step-(4)
residual — the same blindness that makes the residual unable to see endpoint phase rounding.

A caller with `atol = 10⁻²⁰, rtol = 10⁻³` — a legal combination — gets a value perturbed at the
10⁻³ level with an error bar near machine precision.

**Fix:** add the discarded endpoint contribution

```
Σ_{i not used} (|p_i(a)| + |p_i(b)|)
```

to the region's `abserr`. A handful of flops, and it makes the heuristic self-consistent: the value
is perturbed by an amount the error bar now knows about.

**Where it goes.** Prompt 04 broke the error estimate into components. This is neither a resolution
residual nor a round-off floor; it is a *deliberate truncation*. Either add a fourth component
(`abserr_truncation`) or fold it into `abserr_roundoff` with a comment. Adding a component is
cleaner and costs nothing — the audit's §3.4 point is precisely that a caller wants to see *which*
term is stopping them. Choose and record.

## An open question this prompt must answer

After prompt 02, the `(sin, cos)` path always has `m = 2`, and the two "modes" are the real and
imaginary parts of the single complex antiderivative `q = p₁ + i p₂`. **Dropping one of them is
dropping the real or imaginary part of `q`.** The audit does not address this, and it is not obvious
that the original reasoning — "modes associated with small singular values are noise" — survives the
reinterpretation, because the small-singular-value argument was about the `2N×2N` operator's
spectrum, not about `Re q` versus `Im q`.

Note also that `QuadSourceIntegral.py` passes `f = [Levin_f, lambda x: 0.0]` and
`f = [lambda x: 0.0, lambda x: -Levin_f(x)]` — one identically-zero component — at every one of its
nine call sites. `f₂ ≡ 0` does **not** make `p₂ ≡ 0` (the system couples them), so the filter is
live on the production path.

**You must reach and record a conclusion on this**, with evidence:

- If the filter still earns its keep at `m = 2`, say what it protects against and show a case where
  turning it off degrades a result. Then implement (a) and (b) as described.
- If it does not, the honest change is to remove it and let both components through — which would
  make (b) moot and (a) unnecessary. That is a *larger* change than this prompt asks for, so **do
  not do it unilaterally**: implement (a) and (b), and open a §3 issue on the board recommending
  removal, with the evidence.

The measurement that would settle it: run the campaign's standard problem set and the three-Bessel
oracles with the filter disabled, and compare values, region counts and delivered errors against a
closed form. If nothing changes anywhere, the filter is inert on real problems and should go.

---

## Do not

- **Do not remove the non-finite rejection prompt 01 added to `p_use`.** It is load-bearing (C1).
- Do not remove the filter in this commit, even if you conclude it should go. Open an issue.
- Do not change the floor, the gate, the acceptance test, or the solve.

---

## Verification

1. `AdaptiveLevin/tests/` passes.
2. **The `rtol`-dependence is now visible in `abserr`.** Reproduce the audit's §1.5b case (one
   region, order 16, a problem with mode ratios around `[1e-6, 1]`) at `rtol` = 1e-2, 1e-4, 1e-6,
   1e-8. Before: the value jumps by 3.6e-5 with `abserr` unchanged. After: the reported `abserr` must
   bound the jump. Report the actual table.
3. **The endpoint gate does not change results on ordinary problems.** The five-problem A/B: value,
   `abserr`, regions. Perturbations should be bounded by `rtol`, as the audit's 400-problem sweep
   found.
4. **The filter-disabled experiment** from the open question above, with its numbers.
5. **Three-Bessel oracles**, before and after — this is where `f₂ ≡ 0` makes the filter live.
6. **Prompt 01's C1 reproducers still raise.**

---

## Finish

1. Write `prompts/levin-refactor/logs/06-mode-filter.md`. Under *Numerical evidence*, items 2–5.
   **The open question must be answered in its own clearly-headed section** with the evidence, the
   conclusion, and — if the conclusion is "remove it" — a pointer to the §3 issue you opened. Also
   record the normalisation choice (problem (a)) and where the truncation term went (problem (b)).
   Be explicit in the log that problem (a) is a fix to the *reasoning* and that no field case was
   found where it changed a result.
2. Update `IMPLEMENTATION_STATE.md`. **C5 closes here** (prompt 01 closed its C1-mechanism part).
3. Commit in one commit. Suggested message:

```
Make the Levin mode filter self-consistent with the estimate it feeds

Modes whose relative amplitude falls below rtol are dropped from the endpoint
sum, on the grounds that they cannot be computed accurately and behave as
noise that pollutes the resolution residual. Two things about that heuristic
did not line up with the estimate it serves.

It selected on the mean of |p| over the collocation points, while the
estimate consumes only p at the two endpoints, so a mode with a small mean
and a large endpoint value could be discarded wrongly. No case was found
where this changed a result -- over 400 randomised spike-amplitude problems
the worst perturbation stayed inside rtol, as intended -- but the gate now
tests the quantity it is gating.

More consequentially, the discarded contribution was invisible to the
reported error. Parent and children use identical gating, so the dropped
amount is common-mode in the step-(4) comparison and subtracts out exactly:
the returned value became a function of rtol at fixed mesh, measured jumping
by 3.6e-5 between rtol = 1e-4 and rtol = 1e-6, with the error bar unmoved. A
caller asking for atol = 1e-20 and rtol = 1e-3 -- a legal combination -- got
a value perturbed at the 1e-3 level and an error bar near machine precision.
The discarded endpoint contribution is now added to the region's estimated
error.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
