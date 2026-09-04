# Prompt 10 — Test matrix and campaign verification

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit sections:** §1.12 (C12); §5 recommendation 16
**Depends on:** all nine preceding prompts
**Files you may touch:** `AdaptiveLevin/tests/`, `LiouvilleGreen/tests/`, a new
`docs/adaptive-levin-verification.md`, plus the log and the status board.
**Files you may NOT touch:** any production source file. See *If you find a defect* below.

---

## Character of this commit

Two jobs, in this order:

1. **Complete the test matrix** the audit's C12 asks for, minus whatever earlier prompts already
   added.
2. **Verify the campaign**: check every claim the nine preceding commits made, and hand the user an
   explicit list of what is confirmed, what is not, and what would need a real compute run.

This is the analogue of the `backport-modules` campaign's prompt 10. Its output is a document as much
as it is code.

---

## Part 1 — the test matrix

C12 lists what `AdaptiveLevin/tests/test_levin_quadrature.py` did not cover at the start of the
campaign. Earlier prompts were each asked to add tests for their own change, so **start by
inventorying what exists** rather than by writing.

| C12 gap | Expected owner | Your job |
|---|---|---|
| The `theta_mod_2pi` path — the branch production actually uses | none | **write it** |
| The `theta_deriv` path (all four original tests use spectral differentiation) | none | **write it** |
| A region taking the fallback deliberately | 03 | verify, extend |
| Non-monotonic phase (C2) | 03 | verify |
| Non-finite amplitude (C1) | 01 | verify |
| Reversed span | none | **write it** |
| `m ≠ 2` | none | **write it** — see below |
| `atol = 0` | 01 | verify |
| `abserr` checked against the true error, on every problem | 09 (three-Bessel only) | **extend to all** |

### Notes on the harder rows

**`m ≠ 2`.** `_adaptive_levin` and `_adaptive_levin_subregion_impl` are written for general `m`
(standing note 10) but `_Basis_SinCos` imposes `m = 2` and prompt 01 added validation saying so.
There is no public entry point for `m ≠ 2`. So this test must construct a minimal alternative basis
object satisfying whatever contract prompt 02 settled (`build_Levin_data`, `eval_basis`, and the
complexification predicate) and drive `_adaptive_levin` directly. **This is the only test of the
generic path in the repository**, and the generic path exists precisely so that the future bases in
the prior review's §8.3 have somewhere to land. Make it a real test: a basis for which you can write
a closed form.

If prompt 02's complexification predicate makes a non-`_Basis_SinCos` basis impossible to construct
from outside the module, that is a design problem worth recording — say so and open a §3 issue.

**`abserr` against the true error, everywhere.** For every problem with a closed form (README §5.2),
assert `|value − truth| <= abserr`. Two cautions:

- **Do not weaken an assertion to make it pass.** If the reported error does not bound the truth on
  some problem, that is a finding, and it is exactly what the campaign exists to prevent. Record the
  ratio, mark the case `expectedFailure` with a comment naming the cause, and open a §3 issue.
- The existing four tests assert `|value − truth| < 1e-10`, eight orders of magnitude looser than the
  module delivers (standing note 4). **Tighten them** to something near what is actually delivered,
  so they can catch a regression. Choose thresholds from measurement, with a stated margin, and say
  what margin you used and why — a threshold set exactly at the measured value will flake.

**`_GRZIntegral(10.0)`.**

> **Correction to the audit.** C12 says `_GRZIntegral(10.0)` "takes [the fallback] by accident:
> 10·(atan 1 − atan −1) = 15.7 < 6π, so **that test never exercises Levin at all**". `test_GRZIntegral`
> calls `_GRZIntegral` at λ = 10, **100 and 1000**; only λ = 10 falls through. The test does exercise
> Levin, twice.

The narrower true point still stands: one of the three sub-cases silently never reaches the Levin
rule, and nothing says so. Make it explicit — assert which path each λ takes, using
`num_simple_regions` — so that the coverage is visible rather than accidental. After prompt 03 the
gate is total variation, so re-derive which path λ = 10 now takes rather than assuming it is
unchanged.

### Keep the suite fast

Baseline was 0.008 s for 4 tests. A test matrix this size will be slower; keep it under a few
seconds. Put anything genuinely slow (a three-Bessel sweep, an order sweep) behind an explicit
marker or a separate module that is not run by default, and say in the log how to run it.

---

## Part 2 — campaign verification

Write `docs/adaptive-levin-verification.md`, modelled on
[`docs/backport-modules-verification.md`](../../docs/backport-modules-verification.md).

Go through **every claim made by prompts 01–09** — their commit messages and their logs — and record
for each: what it claimed, how you checked it, and what you found. Do not take a log's word for it;
the point of this pass is independent confirmation.

The claims that most need re-checking, because they are the ones a reader will rely on:

1. **C1** — no input produces a silent zero. Try the audit's cases and at least two more (a spline
   evaluated outside its range; an overflow in `x^{3/2}·m_μ m_ν m_σ`; a division by zero in a user
   amplitude — audit §1.2 names all three as realistic triggers).
2. **C2** — the stationary-phase integral. Report the final relative error against the oracle
   `-6.879079716900e-04` and compare with the audit's three data points (module as committed:
   1590% wrong; interval split by hand: 12 digits; the audit's total-variation prototype: 1.1e-8).
3. **C3** — `converged` is true whenever it should be, and `abserr_total <= max(atol, rtol|val|)` on
   a spread of problems and tolerances.
4. **C4** — the ω-ladder in both phase modes. The reported floor must bound the truth and must not
   depend on whether the phase is range-reduced.
5. **The speedup** — end-to-end wall time at `c8a1918` versus the campaign tip, on a fixed problem
   set. The audit's expectation from complexification alone is 1.4–1.8×; prompts 03, 07 and 08 add
   more. Report the composite. **Use the same machine state and best-of-N**; a single-run comparison
   is not worth publishing.
6. **The `lstsq` share after complexification** (prompt 02 step 6, re-derivable from the API after
   prompt 07 item 4) — this is the standing input to the deferred rank-revealing-QR decision
   (README §6). Restate it prominently.
7. **The three-Bessel oracles** — all seven, at `c8a1918` and at the tip. Any change in the delivered
   error is the campaign's most important single result, in either direction.

Then a section on **what is not verified**, with the same honesty the `backport-modules` verification
document uses:

- Anything that needs a real compute pipeline (`QuadSourceIntegral` end to end under Ray) — say what
  it would take and roughly what it would cost.
- The phase-spline accuracy floor, which no test in this repository can currently see (README §6).
- Any `expectedFailure` you left, and why.

## If you find a defect

**Do not fix it here.** This prompt's commit must contain no production-source changes, so that a
red test is unambiguously a statement about the code rather than about this commit.

Instead: write the failing test (marked `expectedFailure` with a comment), record it in
`docs/adaptive-levin-verification.md`, and open a §3 issue on the board naming the prompt whose work
it belongs to. If it is severe enough to warrant an immediate fix, say so in the log and let the user
decide — do not decide for them by quietly patching it.

---

## Verification

1. The full suite passes (or fails only where explicitly marked, with each marker explained).
2. Every row of the C12 table above is either covered by a named test or explained as not covered.
3. `docs/adaptive-levin-verification.md` exists and covers every claim from prompts 01–09.
4. `IMPLEMENTATION_STATE.md` §2's item table has no ⬜ left, and every ⚠️ has a one-line reason.
5. The suite's runtime is stated.

---

## Finish

1. Write `prompts/levin-refactor/logs/10-test-matrix.md`. Keep it short — the substance belongs in
   `docs/adaptive-levin-verification.md`. The log records what you added, what you found, and what
   you deliberately left.
2. Update `IMPLEMENTATION_STATE.md`: mark the campaign complete, fill in earlier prompts' commit
   SHAs (verified reachable with `git merge-base --is-ancestor <sha> HEAD`) since you are editing the
   file anyway, and leave §3 accurate — an issue that is still open at the end of the campaign should
   stay open and visible, not be tidied away.
3. Commit in one commit. Suggested message:

```
Complete the Levin test matrix and verify the refactor campaign

The suite covered four integrals, none of which exercised the range-reduced
phase path that production uses, the supplied-derivative path, a
non-monotonic phase, a non-finite amplitude, a reversed span, a basis with
other than two components, or the returned error estimate against a known
truth. One of the three Gradshteyn & Ryzhik cases silently never reached the
Levin rule at all, because its net phase change fell below the fallback
threshold.

The gaps are now covered, the four original assertions are tightened from
1e-10 -- eight orders of magnitude looser than the module delivers -- to
thresholds set by measurement, and every problem with a closed form asserts
that the reported error bounds the measured one.

docs/adaptive-levin-verification.md records what each commit in this campaign
claimed and what an independent check found, together with an explicit list of
what could not be verified here and what it would take.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
