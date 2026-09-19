# Log 01 — gate the three-Bessel diagnostic plots

**Prompt:** `prompts/test-suite-runtime/01-gate-the-three-bessel-diagnostic-plots.md`
*(reconstructed after the fact — see "Deviations" below; it did not drive the work)*
**Commit:** `07c6041` — "Put the three-Bessel convergence figures behind a flag"
**Model:** Opus 5
**Date:** 2026-09-19
**Result:** **DONE.** `LiouvilleGreen.tests.test_3bessel_analytic` runs in **9.4 s** against a
**1121.5 s** baseline, a factor of **119**, with every assertion intact and no tolerance touched.
The whole suite by discovery is **149.4 s** against ~22 min. `plot_and_compute_3Bessel` is now
`compute_3Bessel`, which builds the three phase objects and returns the evaluation at `max_x`; the
250-point grid and the figures moved to `_plot_convergence`, called only when
`THREE_BESSEL_DIAGNOSTIC_PLOTS` is set to something other than `"0"` or `"false"`.
`test_YJJ_log_scaling`, which contains no assertion, is `skipUnless` the same flag and now reports
as **skipped** rather than as a pass that could not have failed. One file in the diff; `black
--check` clean.

**The order of events matters and is not the usual one.** The author asked where the suite's wall
clock goes; the measurement below answered it; the author then asked for the change directly, and
it landed at `07c6041`. The campaign, its README and the prompt file were written afterwards. So
this log is the primary record and the prompt is a reconstruction of it, not its source. Read §2.

## What shipped

**One file: `LiouvilleGreen/tests/test_3bessel_analytic.py`** (+163 / −37).

1. **`compute_3Bessel`** (was `plot_and_compute_3Bessel`) builds `mu_phase`, `nu_phase`,
   `sigma_phase` and returns `evaluator(...)` at `max_x`. That final call came out of the split
   textually unchanged, and the grid never fed into it, which is why no asserted number moves.
   It takes optional `mu_phase` / `nu_phase` so a caller sweeping `s` at fixed `k`, `q` can supply
   them.
2. **`_plot_convergence`** holds the 250-point `logspace` grid, the figure and both `savefig`
   calls, and is called from `compute_3Bessel` only under the flag. `seaborn` and
   `matplotlib.pyplot` are imported **inside** it and inside `test_YJJ_log_scaling`, and are gone
   from module scope. `_figure_path` factors out the path construction the two plotting sites
   shared.
3. **`DIAGNOSTIC_PLOTS`** reads `THREE_BESSEL_DIAGNOSTIC_PLOTS`, off unless set to something other
   than `""`, `"0"` or `"false"` (case-folded). A comment above it carries the 42.5 s / 0.14 s
   measurement so the next reader does not have to rediscover why the switch exists.
4. **`test_YJJ_log_scaling` is `@unittest.skipUnless(DIAGNOSTIC_PLOTS, ...)`**, with a docstring
   stating plainly that it asserts nothing and pointing at `test_YJJ_log_singularity` as the
   assertion that does cover the near-singular region.
5. **`mu_phase` / `nu_phase` hoisted** out of the eps sweep in `test_YJJ_log_singularity` — they
   depend only on `(mu, k)` and `(nu, q)`, fixed across that oracle's 20 s-values, and were being
   rebuilt 20 times each.
6. The module docstring gained a paragraph pointing at the switch.

## Deviations from the prompt

- **`UNINTENDED DRIFT` — the prompt did not exist when the work was done.** The campaign is
  retrospective; `07c6041` landed in answer to a direct request, and README, prompt and board came
  after, in a second commit. This breaks CLAUDE.md invariant 3's "in its own commit" and makes the
  revert boundary the **pair** of commits rather than one. Classified as drift rather than as a
  choice because the invariant is not one the work was free to trade away — it is recorded here,
  in README §4 and in board §5 rather than smoothed over. It is not a precedent.
- **`IMPLEMENTATION CHOICE` — `test_YJJ_log_scaling` kept under `skipUnless`, not moved to
  `docs/`.** The issue offered both. Keeping it costs one skip line in the report and leaves the
  eps-scaling figures reachable by the same switch as the convergence ones, which is one mechanism
  instead of two; moving it to `docs/` would have meant a second copy of the phase-building and
  oracle code, or an import from a test package into `docs/`. The decisive point is that a reader
  who wants these figures wants both sets, and they now come from one flag.
- **`IMPLEMENTATION CHOICE` — an environment variable rather than a module flag.** The issue
  allowed either. The variable needs no edit to the file to turn on, so the figures can be had from
  a clean tree.
- **`STRUCTURALLY REQUIRED` — explicit `phase_atol` / `amplitude_rtol` in the hoisted builds.** The
  first version of the hoist relied on `bessel_phase`'s own defaults. They are **`None`**, not
  `DEFAULT_PHASE_ATOL` / `DEFAULT_AMPLITUDE_RTOL` (both `1e-11`), which the helper had been passing
  explicitly. Whether `None` resolves to the same constants inside `bessel_phase` was not
  established, so the hoisted calls pass both explicitly, as the helper did. Caught before commit.

## Verification performed

**The baseline.** Every test module in the repository timed in its own interpreter on `75db3c5`,
2026-09-19. `test_3bessel_analytic` **1121.5 s**; the next four
`ComputeTargets.tests.test_quadsource_integral` 52.5 s, `ComputeTargets.tests.test_source_grid`
21.3 s, `LiouvilleGreen.tests.test_three_bessel` 10.5 s,
`ComputeTargets.tests.test_numeric_phase_cut` 9.7 s; the remaining 44 modules 195.2 s between them,
of which roughly 55 s is 49 × interpreter startup. Total **1321 s**, of which one module is
**84.9 %**. *(A concurrent `discover` run of `LiouvilleGreen/tests` was competing for CPU
throughout, so these are an upper bound; the ranking is unaffected.)*

**The split inside the module**, measured directly at $(k,q,s) = (1.3,1.7,2.1)$, `max_x = 1e12`,
`atol = 1e-14`, `rtol = 1e-10`: the 250-point grid **42.5 s**, the single asserted evaluation at
`max_x` **0.14 s**. A factor of **304**. The helper ran **47** times per run — 5 + 2 + 40 — and one
run wrote **110 files, 4.2 MB**.

**Default run, after.** `Ran 5 tests in 8.767s`, `OK (skipped=1)`, wall clock **9.4 s**. No
`test_3bessel_analytic/` directory created. Per-method: `test_JJJ` ok, `test_YJJ` ok,
`test_YJJ_log_singularity` ok, `test_abserr_bounds_truth` ok, `test_YJJ_log_scaling` *skipped
'diagnostic only; set THREE_BESSEL_DIAGNOSTIC_PLOTS to run it'*.

**Diagnostic run, after** — the half the default run no longer covers, so it was exercised rather
than assumed. `THREE_BESSEL_DIAGNOSTIC_PLOTS=1` over `test_YJJ` and `test_YJJ_log_scaling`:
`Ran 2 tests in 68.144s`, `OK` — `test_YJJ_log_scaling` **ran rather than skipped** — and **10
PDFs** written, which is 2 convergence figures from `test_YJJ` and 8 from `test_YJJ_log_scaling`
(2 oracles × 2 configs × 2 figure kinds), as expected.

**That no assertion changed.** `test_abserr_bounds_truth` runs at the fixed triple and prints the
true error against the closed form for all seven oracles; its values after the change — JJJ(1,1,0)
3.8121e-14, JJJ(2,2,0) 2.1233e-15, JJJ(2,2,2) 3.1225e-15, JJJ(2,3,1) 9.2704e-15, YJJ(0,0,0)
5.4489e-12, YJJ(0,2,2) 1.5432e-14, each bounding — are the same figures the baseline run printed,
`bounds=True` on 7 of 7 both times. The other four tests draw fresh wavenumbers on every run
(README §2), so they cannot be compared value-by-value by construction; the structural argument is
that the `evaluator(...)` call is textually unchanged and nothing read the grid.

**That the test set is unchanged**, established mechanically: the `test*` methods were extracted by
`ast` from `git show HEAD:...` and from the working file and compared as sets — identical, five
methods. So `LiouvilleGreen` was 148 before and is 148 now, one skipped, without needing a 20-minute
baseline discovery run.

**Full suite by discovery, after:** AdaptiveLevin `Ran 32 tests` OK, 0.5 s; CosmologyModels
`Ran 39 tests` OK, 2.2 s; LiouvilleGreen `Ran 148 tests` `OK (skipped=1)`, 24.4 s; ComputeTargets
`Ran 530 tests` OK, 122.3 s. **749 tests, all OK, 149.4 s.**

**`black --check`**: `1 file would be left unchanged.`

## Observations not acted on

- **`set_xlabel("$\epsilon$")` is an invalid escape sequence**, at four sites in
  `test_YJJ_log_scaling` (lines 801 and 855 of the file at `HEAD` before this change, 917 and 971
  after, each appearing twice under `ast.parse`). Python emits `SyntaxWarning: invalid escape
  sequence '\e'` on every import of the module. It **predates this change** — confirmed against
  `git show HEAD:` — and the fix is an `r` prefix. Not touched: it is unrelated to runtime and
  CLAUDE.md invariant 4 forbids folding it in. Recorded in board §5 rather than opened as a §3
  issue, because it is a lint nit with a one-character fix that affects no number and belongs in no
  section of the project-wide index. Anyone editing those lines for another reason should fix it in
  passing.
- **`ComputeTargets` is now the suite's head**, at 122.3 s of 149.4 s, with
  `test_quadsource_integral` 52.5 s and `test_source_grid` 21.3 s the two largest. **Nothing
  suggests this is diagnostic rather than verification cost** — unlike `test_3bessel_analytic`,
  neither module plots, and no test module outside `test_3bessel_analytic` imports `matplotlib` or
  calls `savefig` anywhere in the repository. Recorded as board §5 note 2 as the place a future
  runtime question would start, not as a defect.
- **The figures land in the current working directory**, under `test_3bessel_analytic/<timestamp>/`.
  Unchanged by this work, and now written only on request, which removes most of the reason to care.

## State handed to the next prompt

None — this is a one-prompt campaign and README §5's acceptance is met. What it hands to the
repository:

- `transfer-remedial`'s `[08-3bessel-plot-cost-dominates-the-suite]` is **closed**, its next step
  implemented as specified. Moved to that board's §4 and removed from `docs/OPEN_ISSUES.md`.
- **`THREE_BESSEL_DIAGNOSTIC_PLOTS=1` is how the convergence figures are obtained.** Anyone
  re-measuring the three-Bessel oracles, or chasing
  `[08-3bessel-chebyshev-order-is-now-the-limit]`, wants it set.
- The per-module timing baseline in "Verification performed" is the measurement any future runtime
  work should be scored against.
