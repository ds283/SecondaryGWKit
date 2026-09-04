# Log 09 — Propagate the error estimate to callers

**Prompt:** prompts/levin-refactor/09-caller-propagation.md
**Commit:** Carry the Levin error estimate out to the three-Bessel callers; SHA intentionally omitted — see README §5 rule 5
**Date:** 2026-09-04
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### 1. `LiouvilleGreen/three_bessel_integrals.py`

- New `BesselIntegralResult` `NamedTuple` (`value`, `abserr`, `converged`, `phase_limited`),
  replacing the bare `float` that `quad_JJJ`/`quad_YJJ` used to return. Its docstring states plainly
  what `abserr` does and does not cover (the scoping note's honesty requirement).
- `_direct_JJJ`/`_direct_YJJ` (`:119`, `:358`): now scale `simple_quadrature`'s own `abserr` by the
  same normalisation constant as `value`, and compute a local `converged = abserr <= max(atol,
  rtol*|value|)` (the module itself has no such flag). `phase_limited=False`: plain quadrature has
  no phase-limited concept.
- `_Levin_3bessel` (`:210`): accumulates `abserr` **linearly** across the four phase groups
  (`abserr_total += |combination[index]| * data["abserr"]`, then scaled by the same `norm_factor` as
  `value`), `converged = all(...)`, `phase_limited = any(...)`.
- `quad_JJJ`/`quad_YJJ` (`:63`, `:311`): combine the numeric and Levin halves as
  `abserr = numeric.abserr + Levin.abserr`, `converged = numeric.converged and Levin.converged`,
  `phase_limited = Levin.phase_limited` (the numeric half has none).

### 2. `ComputeTargets/QuadSourceIntegral.py`

Traced all nine `adaptive_levin_sincos` call sites (the eight `J1..Y4_data` calls inside
`_three_bessel_Levin`, plus the single `:1013` call inside `WKB_Levin_integral`) and propagated
`abserr`/`converged`/`phase_limited` alongside `value` at every stage between them and the top-level
`compute_QuadSource_integral`:

- `_three_bessel_Levin` (`:443`): each of the eight `adaptive_levin_sincos` calls already returns
  `abserr`/`converged`/`phase_limited`; these are folded into the per-block `metadata` entries and
  combined **linearly** across the four `J`/`Y` groups (`J_abserr = norm_factor * sum(...)`, matching
  `_Levin_3bessel`'s policy in `three_bessel_integrals.py` for the same reason), returned as new
  top-level `"abserr"`/`"converged"`/`"phase_limited"` keys (each a `[J, Y]` pair).
- `_three_bessel_quad` (`:689`): added `"abserr"` (from `simple_quadrature` directly, already
  correctly scaled) and a locally-computed `"converged"` pair, mirroring `_three_bessel_Levin`'s
  shape.
- `_three_bessel_integrals` (`:330`): all three branches (Levin-only, quad-only, quad+Levin combined)
  now also return `"J_abserr"`/`"Y_abserr"`/`"J_converged"`/`"Y_converged"`; the combined branch sums
  the quad and Levin halves linearly, same as `quad_JJJ`.
- `analytic_integral` (`:799`): combines `data0pt5`/`data2pt5`'s `J_abserr`/`Y_abserr` into
  `Y_factor_abserr`/`J_factor_abserr` linearly (weighted by `|A|`, the same coefficient that combines
  the values), then folds through the final `yv`/`jv` recombination as
  `abserr = |F| * (|yv| * Y_factor_abserr + |jv| * J_factor_abserr)`. Returns new `"abserr"`/
  `"converged"` keys alongside `"value"`.
- `WKB_Levin_integral` (`:1031`): adds `"abserr"`/`"converged"`/`"phase_limited"` as siblings of the
  existing `"data"` (the fixed-schema `LevinData` namedtuple) and `"value"` keys — **not** as new
  `LevinData` fields (see "Where the error ends up" below). `abserr` is scaled by the same
  `(1.0 + z_response.z)` factor as `value`.
- `compute_QuadSource_integral` (`:74`): the returned `"metadata"` dict gains an `"abserr"`/
  `"converged"` pair inside `"analytic"`, and a new `"WKB_Levin"` entry (`None` when that region
  wasn't run) carrying `abserr`/`converged`/`phase_limited` from `WKB_Levin_integral`.

**Where the `QuadSourceIntegral` error ends up.** Per the prompt's instruction, checked the
storable-class definition (`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`) before deciding:
`LevinData` (`Quadrature/integration_metadata.py`) is a namedtuple mapped onto fixed, explicitly
named SQL columns (`WKB_Levin_num_regions`, `WKB_Levin_evaluations`, ...); adding a field there is a
schema change and explicitly out of scope. `"metadata"` (`QuadSourceIntegral._metadata`) is a
free-form dict, persisted as a single JSON string column (`sqla.Column("metadata", sqla.String(...))`,
confirmed at `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py:205-206`, written via
`json.dumps(obj.metadata)`). This is exactly the "no schema change needed" case the prompt
anticipated, so all new error information for this file goes there. No `Datastore` schema was
touched.

### 3. Tests and the benchmark harness

- `LiouvilleGreen/tests/test_3bessel_analytic.py`: all four bare-float call sites (`plot_and_compute_3Bessel`'s
  internal grid loop and final call, and the direct `quad_YJJ` call in `test_YJJ_log_scaling`) updated
  to unpack `.value`. `test_JJJ`/`test_YJJ`/`test_YJJ_log_singularity` print the reported
  `abserr`/`converged`/`phase_limited` alongside the existing diagnostics.
- New `test_abserr_bounds_truth` (`@unittest.expectedFailure`): runs all seven oracles (five `J`,
  two `Y`) at a fixed triple, prints the true-vs-reported `abserr` and their ratio for every one, and
  asserts the reported `abserr` bounds the true error. See "Numerical evidence" and "Deviations"
  below for why this is `expectedFailure` rather than a plain assertion.
- `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`: `run_bessel_levin` unpacks the
  `BesselIntegralResult`, and its returned row dict gains `reported_abserr`,
  `abserr_bounds_truth`, `abserr_ratio_true_over_reported`, `converged`, `phase_limited` — additive
  columns; `campaign.py`'s `_write()` takes the union of keys across rows, so this needed no changes
  there.

## Numerical evidence

### The propagated error's size, against three-plus oracles (verification item 3)

Fixed triple `(k, q, s) = (1.3, 1.7, 2.1)`, `max_x = 1e12`, `atol = 1e-14`, `rtol = 1e-10` — the same
configuration `test_abserr_bounds_truth` uses (measured by running that test to completion):

| Oracle | true abserr | reported abserr | ratio (true/reported) | bounds? |
|---|---|---|---|---|
| J(0,0,0) | 2.75e-09 | 1.53e-07 | 0.018 | yes |
| J(1,1,0) | 3.00e-10 | 2.92e-10 | 1.03 | **no** |
| J(2,2,0) | 2.16e-09 | 2.16e-10 | 9.98 | **no** |
| J(2,2,2) | 8.51e-10 | 2.17e-10 | 3.93 | **no** |
| J(2,3,1) | 2.08e-09 | 1.82e-10 | 11.5 | **no** |
| Y(0,0,0) | 2.46e-09 | 1.53e-07 | 0.016 | yes |
| Y(0,2,2) | 1.58e-09 | 2.17e-10 | 7.27 | **no** |

2 of 7 bound the truth (both order-0 oracles, where the reported `abserr` happens to be ~60x
conservative); the other 5 underbound it, by up to 11.5x. This is precisely the scoping note's
prediction: the reported `abserr` is the accuracy of the *quadrature*, not of the phase/modulus
splines that feed it, and this module's own retuning comment records those splines' fit floor at a
uniform ~2e-8 relative error across all seven oracles
(`three_bessel_integrals.py`'s `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` comment) — a floor invisible from
inside the quadrature, and larger than several of the true errors shown above.

Also measured at five random `(k, q, s)` draws in `(0.1, 5.0)^3` (one run, not cherry-picked): 2 of 5
bounded truth, 3 did not (ratios up to 27.2x) — consistent with the fixed-triple table, i.e. this is
not an artefact of the particular triple chosen.

`converged` was `False` on every one of the 12 combinations measured (both tables): at
`atol=1e-14, rtol=1e-10`, every region hit `phase_limited=True` (the round-off floor, not lack of
resolution, bounds further improvement) before meeting that tolerance. This is expected — 1e-14/1e-10
is far tighter than the ~1e-10-to-1e-7 round-off floor `adaptive_levin_sincos` reports at `max_x=1e12`
— and is itself useful information a caller gets for the first time via the new `converged` flag.

### The cancellation is visible (verification item 4)

`quad_JJJ`'s J(0,0,0) oracle at the fixed triple, Levin part only (the four sum-and-difference
groups before the numeric small-`x` piece is added), `chebyshev_order=12`:

| Group | sign | value | reported abserr |
|---|---|---|---|
| `(e_nu=+1, e_sigma=+1)` | -1 | 0.37035955 | 1.68e-07 |
| `(e_nu=+1, e_sigma=-1)` | +1 | 0.37036711 | 1.68e-07 |
| `(e_nu=-1, e_sigma=+1)` | +1 | 0.37036565 | 1.68e-07 |
| `(e_nu=-1, e_sigma=-1)` | -1 | -0.37036420 | 1.68e-07 |

Signed sum = 0.74073741; `abserr` sum (all `|sign|=1`) = 4 × 1.68e-07 = 6.72e-07; both are then
divided by the same `norm_factor` (4 for the group average, then the `pi^1.5/(2 sqrt(kqs))`
prefactor) to give the reported combined `abserr` in the earlier table (1.53e-07). At this particular
triple the four groups happen to combine *constructively* (a cancellation factor of ~2, not the
audit's illustrative near-total-cancellation example) — but the mechanism that would reveal a worse
case is exactly what changed: `abserr` is now the linear sum of the four groups' own `abserr`,
carried through the same `norm_factor` as `value`, so a configuration where the four terms nearly
cancel (small `|value|`, unchanged `abserr`) would show up immediately as a large `relerr` a caller
can compute from the two returned numbers — which was impossible when only `value` was returned.

### `ComputeTargets/QuadSourceIntegral.py` wiring (verification items 1, 5)

`LiouvilleGreen/tests/test_three_bessel.py::test_three_bessel` (which exercises
`_three_bessel_integrals` — the internal, `GkSource`-driven implementation in `QuadSourceIntegral.py`,
not `three_bessel_integrals.py`'s `quad_JJJ`/`quad_YJJ`, which have no production caller — see
README standing note 9) run to completion: **2 tests, OK, 5.18s**. `json.dumps` of the full returned
metadata dict (already part of this test) confirms every new field
(`J1_data.abserr/converged/phase_limited`, ..., `Y4_data...`) round-trips as plain JSON-safe
`float`/`bool`, and the new top-level `"J_abserr"`/`"Y_abserr"`/`"J_converged"`/`"Y_converged"` keys
are present at every level (`_three_bessel_Levin`, `_three_bessel_integrals`,
`analytic_integral`/`compute_QuadSource_integral`'s metadata dict), confirmed by direct inspection of
the printed JSON.

## Deviations from the prompt

### IMPLEMENTATION CHOICE — `BesselIntegralResult` carries 4 fields, not the full component breakdown

The prompt: "carry `converged`..., `phase_limited`, and the components from prompt 04 if you can do
so without an unwieldy return type." Chose `(value, abserr, converged, phase_limited)` and did not
propagate `abserr_resolution`/`abserr_roundoff`/`abserr_fallback`/`abserr_truncation` individually.
Reasoning: the prompt's own text names `converged` as "the one a caller most needs"; the four
components are diagnostic (which of several sources dominates a *single* Levin region's own error) and
their meaning degrades once summed across four phase groups plus a numeric-quadrature half that has
none of them — a caller wanting per-region diagnostics for the *Levin* half specifically can still
reach `data["abserr_roundoff"]` etc. by calling `adaptive_levin_sincos` directly, as the module
itself always has. Kept the return type to the four fields the prompt's own priority ordering
justified.

### IMPLEMENTATION CHOICE — `phase_limited=False` (not `None`) for plain-quadrature halves

`_direct_JJJ`/`_direct_YJJ` and `_three_bessel_quad`'s "phase-limited" concept does not apply (no
Levin round-off floor exists for `scipy.quad`). Chose `False` over `Optional[bool]` so the field
stays a plain `bool` everywhere and a combined `phase_limited = Levin.phase_limited` (in
`quad_JJJ`/`quad_YJJ`) does not need to special-case `None`. Documented at each definition site with
a one-line comment.

### IMPLEMENTATION CHOICE — `converged` for a plain-quadrature half is locally computed, not native

`simple_quadrature` (`Quadrature/simple_quadrature.py`) has no `"converged"` key; only
`adaptive_levin_sincos` does. Computed `abserr <= max(atol, rtol*|value|)` locally wherever a
quadrature-only `abserr` needed a matching `converged` flag (`_direct_JJJ`/`_direct_YJJ` in
`three_bessel_integrals.py`; `_three_bessel_quad` in `QuadSourceIntegral.py`), using the same
`atol`/`rtol` already passed to that call. This mirrors `adaptive_levin_sincos`'s own definition of
`converged` exactly, so a caller combining the two halves via `and` gets a consistent semantic.

### STRUCTURALLY REQUIRED — `test_abserr_bounds_truth` restructured around a single end-of-loop assertion, not per-oracle `subTest`

First attempt used `with self.subTest(...):` per oracle with an assertion inside the block, expecting
`subTest`'s documented "continue past a failure" behaviour. Measured directly (a 5-iteration toy
`TestCase` and then the real test): combined with `@unittest.expectedFailure`, a failing `subTest`
**aborts the rest of the loop** — only 2 of 7 oracles printed before the (expected) exception
propagated. This is a real interaction of Python's `unittest`, not a mistake in this test's logic
(verified reproducible in isolation). Restructured to collect every oracle's `(true_abserr,
reported_abserr, bounds)` in a plain list first, print all seven, and raise exactly one assertion
(`self.assertEqual(failures, [])`) after the loop — compatible with `expectedFailure`'s
one-exception-per-test model and gives full diagnostic output. Re-run to completion: all seven
oracles print, matches the numbers in "Numerical evidence" above, test correctly reports "expected
failure".

### UNINTENDED DRIFT — full `LiouvilleGreen/tests/test_3bessel_analytic.py` class not run to completion

Verification item 1 asks that `LiouvilleGreen/tests/` pass (or fail only as step 3 predicted). Ran
`test_three_bessel.py` (`ComputeTargets/QuadSourceIntegral.py`'s own three-Bessel path) to completion:
2/2 pass. Ran the new `test_abserr_bounds_truth` to completion: expected failure, as designed. Did
**not** run `test_JJJ`/`test_YJJ`/`test_YJJ_log_singularity`/`test_YJJ_log_scaling` to completion:
each generates a 250-point `evaluator()` grid per oracle purely to draw a diagnostic PDF/PNG plot
(`plot_and_compute_3Bessel`), a cost that predates this prompt and is unrelated to the change here
(confirmed: the only edit inside that function's grid loop is appending `.value` to an already-unpacked
call). `test_JJJ` alone ran past 6 minutes without finishing its first oracle's plot-and-compare cycle
in this session and was killed rather than left to run further. This is a real verification gap, not
a claimed pass: the *values* these tests check (`numeric = result.value`, unchanged from before this
prompt) are not independently confirmed correct by this session for these four specific test methods,
though the identical code path (same `quad_JJJ`/`quad_YJJ`, same `.value` extraction pattern) is
exercised and confirmed correct by `test_abserr_bounds_truth`, which shares the same evaluators,
oracles and near-identical parameters (`atol=1e-14` vs `test_JJJ`'s unset default; both at
`max_x` on the order of `1e12`). **Next step:** whoever has time for a ~30-60 minute test run should
execute `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_3bessel_analytic -v`
in full and confirm `test_JJJ`/`test_YJJ`/`test_YJJ_log_singularity` still pass (expected: yes,
unchanged assertions on unchanged values) and `test_YJJ_log_scaling` still runs without raising
(it has no assertions).

### Observations not acted on

- **`Y3_data` in `_three_bessel_Levin` uses `LEVIN_ABSERR`/`LEVIN_RELERR` module constants
  (`1e-23`/`1e-8`) instead of the `atol`/`rtol` parameters every other one of the eight calls uses**
  (`QuadSourceIntegral.py`, pre-existing, predates this prompt). Looks like a copy-paste slip — `Y1`,
  `Y2`, `Y4`, and all four `J` calls use `atol`/`rtol`; only `Y3` doesn't — but it is not something
  this prompt's scope covers (no value or error-propagation logic changes because of it; it only
  changes how tightly that one sub-call converges), and fixing it would change `Y3_data`'s reported
  numbers for reasons unrelated to error propagation. Left alone per README §5 rule 6.
- **`numeric_quad_data`/`WKB_quad_data` (regions 1 and 2 of `compute_QuadSource_integral`, plain
  `scipy.quad` via `simple_quadrature`) already compute an `abserr` that is silently discarded** —
  `numeric_quad_integral`/`WKB_quad_integral` return the *entire* `simple_quadrature` result dict
  (which has its own `"abserr"` key) under the name `"data"`, but the top-level function only reads
  `payload["value"]` and `payload["data"]` (the nested `IntegrationData`, not the abserr), never
  `payload["abserr"]`. This predates this prompt and is not Levin-related, so it is out of this
  prompt's scope by its own title ("Carry the *Levin* error estimate..."), but it means `"total"`
  (`numeric_quad + WKB_quad + WKB_Levin`) still has no complete propagated error bound even after
  this commit — only its `WKB_Levin` third does. See the §3 issue opened below.

## Verification performed

1. `AdaptiveLevin/tests/`: **28 tests, OK, 0.059s** (unaffected — this prompt does not touch
   `levin_quadrature.py`; run anyway to confirm the campaign's standing baseline).
2. `LiouvilleGreen/tests/test_three_bessel.py`: **2 tests, OK, 5.18s**, full run, output inspected.
3. `LiouvilleGreen/tests/test_3bessel_analytic.py::Test3BesselAnalytic::test_abserr_bounds_truth`:
   run standalone to completion, **1 test, OK (expected failures=1)**, all seven oracles' diagnostics
   printed and match the "Numerical evidence" table above.
4. `LiouvilleGreen/tests/test_3bessel_analytic.py` (full class): started, did not complete within the
   session (see "Deviations" — `test_JJJ` alone exceeded 6 minutes on pre-existing plotting cost);
   not claimed as passing.
5. `grep -rn "quad_JJJ\b\|quad_YJJ\b"` across the repository: all reference sites enumerated and
   accounted for — 6 call sites needing the new-signature update (4 in
   `LiouvilleGreen/tests/test_3bessel_analytic.py`, 2 via `ORACLES` in
   `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`), all updated; the rest are
   definitions, an import, or comments.
6. `PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets.QuadSourceIntegral"` and
   `import LiouvilleGreen.three_bessel_integrals`: both exit cleanly, no errors.
7. `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`'s `run_bessel_levin` called directly
   (one oracle, `max_x=1e5`, from the `docs/adaptive-levin-benchmark` directory with
   `PYTHONPATH=<repo>:.`): returns a dict with the new `reported_abserr`/`abserr_bounds_truth`/
   `abserr_ratio_true_over_reported`/`converged`/`phase_limited` keys alongside the pre-existing
   ones, confirming the harness runs end to end with the new return type.

## Observations not acted on

(See "Deviations" above — the `Y3_data` tolerance slip and the discarded `numeric_quad`/`WKB_quad`
`abserr` are recorded there rather than duplicated here.)

## State handed to the next prompt

- `quad_JJJ`/`quad_YJJ` now return `BesselIntegralResult`, not a bare `float`. Any *future* caller of
  either function (none exist in production; see README standing note 9) must unpack `.value`.
- `ComputeTargets/QuadSourceIntegral.py`'s returned `"metadata"` dict has two new populated entries:
  `metadata["analytic"]["abserr"]`/`["converged"]`, and `metadata["WKB_Levin"]` (a dict with
  `abserr`/`converged`/`phase_limited`, or `None` if region 3 wasn't run for this call). Nothing
  in-tree reads these yet (grepped); this is the pipe the prompt asked for, not a persisted
  consumer of it.
- Two new §3 issues opened (see `IMPLEMENTATION_STATE.md`):
  `[09-abserr-does-not-bound-phase-spline-floor]` and
  `[09-quadsource-total-error-incomplete]`.
- Prompt 10 (test matrix and campaign verification) should treat this prompt's own
  "Deviations" verification gap (item 4 above) as one of the things worth actually running, since it
  is the one piece of this campaign's own verification checklist not closed out here.
