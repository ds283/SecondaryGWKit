# Prompt 01 — gate the three-Bessel diagnostic plots

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **T1**, and `transfer-remedial`'s
`[08-3bessel-plot-cost-dominates-the-suite]`. **Opens:** nothing.
**Recommended model:** **Opus** — the edit is mechanical but the obligation is not. The whole value
is in establishing that no asserted number moved, and the file is dense with tolerances whose
provenance must survive.

> **Reconstructed after the fact.** This prompt was written after `07c6041` had landed, to record
> what the work was. It is not the instruction that produced the commit — see README §4 and the
> log. Read it as a specification of the finished state, and the log as the account of how that
> state was reached.

**Read first:** the module docstring of `LiouvilleGreen/tests/test_3bessel_analytic.py` **in full**
— it records where every tolerance came from and which oracle limits it, and none of it may move;
`prompts/transfer-remedial/IMPLEMENTATION_STATE.md`'s
`[08-3bessel-plot-cost-dominates-the-suite]`, which specifies the change and whose "next step" this
implements; and campaign README §2 for what is out of scope.

**It changes no tolerance, no assertion and no number.**

---

## 1. What to build

Three changes to `LiouvilleGreen/tests/test_3bessel_analytic.py`, and nothing outside it.

**1. Separate the diagnostic from the assertion.** `plot_and_compute_3Bessel` currently evaluates a
250-point `logspace` grid of full three-Bessel integrals, draws a PDF and a PNG, and only then
performs the single evaluation at `max_x` that its callers assert on. Split it: a function that
returns the asserted evaluation, and a separate one that draws the figure. The asserted call must
come out of the split textually unchanged.

**2. Put the figures behind a switch, defaulting to off.** An environment variable — the issue
suggests that or a module flag. Drawing must be opt-in; a default run must write no files. Import
`seaborn` and `matplotlib` inside the plotting function rather than at module scope, so a default
run does not pay for the import either.

**3. Decide what to do with `test_YJJ_log_scaling`.** It asserts nothing, so `unittest` runs 40
near-singular evaluations and four figures on every discovery for a result that cannot fail. The
issue offers two options: keep it in discovery, or move it to `docs/` as a script. Pick one,
**justify it in the log**, and make the outcome visible — a test that cannot fail must not report
as a pass.

While in the file: `test_YJJ_log_singularity` rebuilds `mu_phase` and `nu_phase` inside the helper
for all 20 of its s-values although `k` and `q` are fixed per oracle. Hoisting them is in scope
because it is the same helper's call signature. **If `bessel_phase`'s own parameter defaults are
not the `DEFAULT_*` constants the helper passes explicitly, pass them explicitly in the hoisted
calls too** — do not assume the two agree.

## 2. What must not change

- Every tolerance constant, `MAX_X`, and the docstring's account of how they were measured.
- The set of test methods. Nothing added, nothing removed, nothing renamed.
- The unseeded `uniform(0.1, 5.0)` draws (README §2).
- Any file other than `test_3bessel_analytic.py`.

## 3. Verification the log must carry

Numbers, not pass/fail:

1. **The module both ways.** Default: the test count, the result line, the wall clock against the
   1121.5 s baseline, and confirmation that no figure directory was created. With the switch set:
   that the plotting path still runs and how many figures it wrote. **The diagnostic path must be
   exercised, not assumed** — it is the half that the default run no longer covers.
2. **That the test set is unchanged**, established mechanically against `HEAD` rather than by
   reading the diff.
3. **The full suite** by discovery, per package, with counts and wall clock.
4. **`black --check`** clean.

## 4. What would make this wrong

- A moved tolerance, or an assertion that now reads a different number. The grid never fed into the
  asserted value; if the split changes a result, the split is wrong.
- A default run that still writes files, or still imports `matplotlib`.
- `test_YJJ_log_scaling` left reporting as an ordinary pass.
- Hoisted phase builds that silently use different tolerances from the ones the helper passed.
