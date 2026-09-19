# Test-suite runtime — a one-prompt campaign

**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)

> **This campaign is retrospective.** The change landed first, at `07c6041`, in answer to a direct
> question from the author about where the suite's wall clock goes. The README, the prompt and the
> log were written afterwards to put it on the record in the usual form. **The prompt file is
> therefore a reconstruction, not an instruction that was executed** — it says what the work would
> have been asked for, and the log is the honest account of what was actually done. Nothing here
> should be read as evidence that a prompt was followed; the evidence is the diff and the
> measurements in the log. See §4.

## 1. Why this exists

`prompts/transfer-remedial` prompt 08 ran `test_3bessel_analytic` to completion for the first time
and opened `[08-3bessel-plot-cost-dominates-the-suite]` on that campaign's board: the module spends
its whole wall clock evaluating 250-point grids of three-Bessel integrals to draw convergence
figures, not on assertions. Prompt 08 was allowed to propose only, so nothing was implemented, and
the issue sat open with a fully specified next step.

It was re-measured on 2026-09-19, on `75db3c5`, by timing every test module in the repository in
its own interpreter:

| | seconds |
|---|---:|
| `LiouvilleGreen.tests.test_3bessel_analytic` | **1121.5** |
| `ComputeTargets.tests.test_quadsource_integral` | 52.5 |
| `ComputeTargets.tests.test_source_grid` | 21.3 |
| `LiouvilleGreen.tests.test_three_bessel` | 10.5 |
| the other 45 modules, none over 9.7 s | 195.2 |

**One module was 85 % of a 1321 s suite.** The split inside it was then measured directly at the
fixed triple $(k,q,s) = (1.3, 1.7, 2.1)$, `max_x = 1e12`, `atol = 1e-14`, `rtol = 1e-10`:

| | seconds |
|---|---:|
| the 250-point diagnostic grid | **42.5** |
| the single evaluation at `max_x` that the assertion reads | **0.14** |

A factor of ~300, and `plot_and_compute_3Bessel` ran 47 times per run — 5 in `test_JJJ`, 2 in
`test_YJJ`, 40 in `test_YJJ_log_singularity` — writing 110 files and 4.2 MB of figures.

`test_YJJ_log_scaling` was found to contain **no assertion of any kind**: 40 near-singular
evaluations and four figures, reporting a pass whatever those numbers came out as.

### Why it is worth having

Prompt 08's own impact statement is the argument, and it is a verification argument rather than a
convenience one: *the per-commit `LiouvilleGreen/tests` discovery run is dominated by figure
drawing, so in practice nobody runs it*, which is how
`[07-abserr-bounds-truth-is-now-an-unexpected-success]` — a whole module reporting `FAILED` —
survived three prompts unnoticed. A suite nobody runs asserts nothing.

## 2. Scope

**In scope**

- `LiouvilleGreen/tests/test_3bessel_analytic.py`, and nothing else.

**Out of scope, and none of it was touched**

- Every tolerance, band and assertion in that module. `ABS_TOLERANCE`, `REL_TOLERANCE`,
  `SINGULARITY_ABS_TOLERANCE`, `SINGULARITY_REL_TOLERANCE` and `MAX_X` are unchanged, and the
  module docstring's account of how they were measured stands.
- The unseeded `uniform(0.1, 5.0)` draws. The docstring defends them explicitly — the tolerances
  are sized for the worst oracle over random draws — so making the module reproducible is a
  separate decision for the author, not a performance change.
- `LiouvilleGreen/bessel_phase.py`, `three_bessel_integrals.py`, `AdaptiveLevin/`, and every other
  test module. This campaign removes diagnostic cost; it does not touch the thing being measured.
- `DEFAULT_3BESSEL_CHEBYSHEV_ORDER`, which is
  `[08-3bessel-chebyshev-order-is-now-the-limit]` on `transfer-remedial`'s board and is deliberately
  left alone.

## 3. The prompt

| # | Prompt | Covers | Model |
|---|---|---|---|
| 01 | [Gate the three-Bessel diagnostic plots](01-gate-the-three-bessel-diagnostic-plots.md) | Board item **T1**; closes `transfer-remedial`'s `[08-3bessel-plot-cost-dominates-the-suite]` | Opus |

## 4. Rules

The invariants in the repository's [`CLAUDE.md`](../../CLAUDE.md) apply, **with one departure that
is recorded rather than hidden**: invariant 3 asks that a prompt update
`IMPLEMENTATION_STATE.md` and [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) *in its own
commit*, and here the code commit `07c6041` landed before this campaign existed, so the paperwork
is a second commit. The revert boundary is therefore the pair, not the single commit. This is the
cost of documenting retrospectively and it is not a precedent to reuse when a prompt is written
first.

Invariant 4 holds unchanged: nothing the work did not call for was fixed. The one thing noticed and
not acted on is in the log's "Observations not acted on" and in board §5.

Conventions this campaign inherits and must not "correct":

- **The tolerances are measurements, not guesses.** The module docstring records where every one
  came from, including which oracle limits it and why raising the Chebyshev order makes five of the
  seven *worse*. A runtime change may not move any of them.
- **A performance change that alters a number is not a performance change.** The asserted values
  must be bit-for-bit what they were, and the log must say how that was established rather than
  assert it.

### 4.1 Log format

As `prompts/radiation-oracle/README.md` §4.1: front matter with **Prompt**, **Commit**, **Model**,
**Date**, **Result**; then `## What shipped`, `## Deviations from the prompt`,
`## Verification performed` (quote the numbers, not pass/fail), `## Observations not acted on`,
`## State handed to the next prompt`.

## 5. Acceptance

The campaign is done when prompt 01's row is ✅ or ⚠️; `test_3bessel_analytic` runs in seconds with
its assertions intact and its figures available behind a documented switch; the diagnostic path is
shown still to work rather than assumed to; every test package passes at no lower a count than the
campaign started with; and `transfer-remedial`'s `[08-3bessel-plot-cost-dominates-the-suite]` is
closed on that board and removed from the project-wide index.
