# Log 06 — The compatibility adapter, `theta_abserr`, and consumer migration

**Prompt:** prompts/transfer-remedial/06-evaluation-and-compatibility.md
**Commit:** *(this commit; SHA not self-embedded)* — Adapt the Bessel phase interface to its consumers
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

## The three decisions, up front

The orchestrator has to surface all three (prompt §7).

1. **`Q` — removed.** The prompt's recommended resolution, taken. Reading `data["Q"]` now raises a
   `KeyError` whose message names `phase.residual`, `phase.log_amplitude` and `phase.raw_theta`
   rather than being a bare `KeyError: 'Q'`. The alternatives are weighed in Deviation 1.
2. **`phi` — kept, and identically `0.0`.** Prompt 05 already shipped this; prompt 06 confirms it,
   documents why the key exists at all, and tests it. There *is* a live reader
   (`docs/transfer-remedial/measure_bessel_phase.py:283`), so keeping it is not hypothetical.
3. **`plot_besssel_phase.py` — repaired, not deleted.** Repair was the prompt's default and
   deletion is a stop condition, so it was repaired: `x_min` → `min_x`, `phase(x)` →
   `phase.raw_theta(x)`, and the `Q` panel replaced by a residual panel. **It now runs**: 12 PDFs
   for `nu` = 1/2, 3/2, 5/2, with the reconstructed `J`/`Y` agreeing with `jv`/`yv` to 5.1e-15
   (`nu = 1/2`) and 2.4e-13 (`nu` = 3/2, 5/2, where the difference is SciPy's own error near the
   turning point). The user may still prefer deletion; nothing imports it.

## What shipped

### `LiouvilleGreen/bessel_phase.py`

* **`Q` removed.** `_DeprecatedQ` (the `XSplineWrapper` subclass prompt 05 shipped, `:694-714`)
  is deleted, and the `"Q"` key is gone from the returned mapping.
* **New `_REMOVED_KEYS` and `_BesselPhaseData(dict)`.** `bessel_phase()` now returns a
  `_BesselPhaseData`, a module-level `dict` subclass whose only added behaviour is `__missing__`:
  a removed key raises `KeyError` with the migration text, any other missing key raises the
  ordinary `KeyError(key)`. It pickles exactly as a `dict` does (verified, including that the
  revived object still raises the informative `KeyError`).
* **New accessors on `BesselPhaseFunction`:**
  * `log_amplitude(x, x_is_log=False)` — `ell = log a_nu`, the amplitude counterpart of the
    existing `residual`, and the second half of the diagnostic pair that replaces `Q`.
  * `theta_abserr_at(x, x_is_log=False)` — the **callable** form of the declared absolute phase
    error, for `AdaptiveLevin`'s `theta_abserr`. Scalars in, scalars out; arrays in, arrays out.
    Model: below `x_star` the constant `theta_abserr`; at and above it
    `2 |c_n|/x^(2n+1) + 4 eps + eps |r_nu(x)|`, capped at `theta_abserr`.
* **New constant `DECLARED_SERIES_SAFETY = 2.0`**, applied to the tail series remainder wherever it
  enters a *declared* error and nowhere else (Deviation 6). `accuracy` gains a
  `"declared_series_safety"` key and its two `tail_*` entries now report the **raw** estimator, so
  both the measurement and the margin are visible (19 keys, was 18).
* `phase.theta_abserr` keeps its name, meaning and type (a `float`, the domain-wide bound); its
  comment now points at `theta_abserr_at`. The constructor docstring gains the four-key
  `AdaptiveLevin` phase dictionary verbatim, and the note that `theta` is required but never
  evaluated while `theta_deriv` carries double duty (conditioning *and* subdivision).
* Module docstring: a new "What was removed, and why" entry for `Q`.

### `main.py` — one hunk, `:516-536` (was `:516-528`), inside `run_pipeline`

`atol=1e-25, rtol=5e-14` → `phase_atol=1e-12, amplitude_rtol=1e-12` on both builds, so production
does not depend on the deprecation shim. The comment — which described `Q -> 1` and the
Dormand–Prince stepper, both gone — is replaced by what is actually requested, what is achieved,
and the fact that the cost no longer depends on `largest_x_with_clearance`.

### `ComputeTargets/QuadSourceIntegral_debug.py`

`bessel_function_plot`: `Q = data["Q"]` deleted; `Q_grid` → `residual_grid` from
`phase.residual(x)`; the panel is labelled `$r_{\nu}$` and written to `Bessel_residual.pdf`/`.png`
instead of `Q.pdf`/`.png` (Deviation 7). Everything else in the file is untouched.

### `plot_besssel_phase.py`

Repaired as described above; `Q_plot_nu=*.pdf` → `residual_plot_nu=*.pdf`.

### Tests

* **New `LiouvilleGreen/tests/test_bessel_compatibility.py`** (20 tests, 2.2 s): the adapter
  surface by `inspect.signature`; raw/log agreement at matched arguments; array/scalar equality;
  `XSplineWrapper` still importable and `mod` still an instance; `theta_abserr` present, finite,
  positive and never under-reporting (cached corners at all seven orders, plus a dense `mpmath`
  sweep of both regions at five); the tail declaration far below the near one; a real
  `adaptive_levin_sincos` call that converges, reproduces a 40-digit `mpmath` value, reports the
  declared phase error, prefers the callable to the scalar, never evaluates `theta`, and still
  requires the `theta` key; `Q` gone with an informative `KeyError`; `phi` zero; the deprecation
  shim; and the pickle / `ray.cloudpickle` round trips bit-identical on a grid.
* `LiouvilleGreen/tests/test_bessel_two_region.py`: `"Q"` dropped from
  `test_public_surface_is_preserved`'s key list, and its `Q(x) == raw_theta(x)/x` assertion
  replaced by `assertNotIn("Q", data)` with a pointer to the new module. Nothing else changed;
  `test_declared_accuracy_bounds_the_corner_measured_error` still passes with the larger declared
  values.

## The final adapter surface, verbatim

```python
# phase -- LiouvilleGreen.bessel_phase.BesselPhaseFunction
raw_theta(x, x_is_log=False)                          # x + c_nu + r_nu; eps*theta limited
theta_mod_2pi(x, x_is_log=False)                      # atan2(sin, cos) of the split pair, (-pi, pi]
theta_deriv(x, x_is_log=False, log_derivative=False)  # exp(-2 ell) below x_star, 1 + r' above
residual(x, x_is_log=False)                           # r_nu, tail-continuous branch, never mod 2pi
residual_log_deriv(x, x_is_log=False)                 # dr_nu/d log x
log_amplitude(x, x_is_log=False)                      # NEW in 06: ell = log a_nu
sin_cos_theta(x, x_is_log=False) -> (sin theta, cos theta)
theta_deriv_from_residual(x, x_is_log=False)          # 1 + r_u/x; the check route
theta_abserr_at(x, x_is_log=False) -> float           # NEW in 06: declared |delta theta| at x, rad
bessel_j(x, is_log=False)
bessel_y(x, is_log=False)
# attributes: nu, min_x, max_x, x_star, c_nu, c_nu_reduced, theta_abserr

# mod -- BesselAmplitude (a XSplineWrapper)
__call__(x, is_log=False)      # A_nu = sqrt(2/(pi x)) a_nu
a(x, is_log=False)             # a_nu
log_deriv(x, is_log=False)     # d log A/dx

# the returned mapping (a _BesselPhaseData, i.e. a dict)
phase, mod, phi, bessel_j, bessel_y, min_x, max_x,          # pre-existing meanings
nu, x_star, theta_abserr, amplitude_relerr, theta_deriv_relerr,
accuracy, accuracy_met, crossover, near_region              # added by 05
# "Q": REMOVED. data["Q"] raises KeyError naming phase.residual / phase.log_amplitude.
```

`theta_abserr`'s **type and semantics**: the dict key and the `phase.theta_abserr` attribute are a
`float`, the largest declared absolute phase error anywhere on `[min_x, max_x]`, in radians.
`phase.theta_abserr_at` is a **callable of `x`** returning the declared error *at* `x`, in radians,
and is what a consumer should put in the `AdaptiveLevin` phase dictionary. `max_x` of
`theta_abserr_at` over the domain equals `theta_abserr` by construction (both tail terms decay
monotonically in `x`, so the tail maximum is at `x_star`, and the near-region value is the constant
scalar).

**The deprecation translation and precedence, as shipped** (unchanged from prompt 05; re-tested
here): `atol` and `rtol` are accepted, **ignored**, and warned about with one
`DeprecationWarning(stacklevel=2)` naming `phase_atol` and `amplitude_rtol`; neither is translated,
because they named tolerances of an ODE solve that no longer exists. Defaults are `None`, so a
caller passing only the new arguments gets no warning (asserted). If old and new are supplied
together the new win and the warning text contains "take precedence" (asserted). Production no
longer relies on any of it.

## Deviations from the prompt

### 1. `Q` removed rather than retained — IMPLEMENTATION CHOICE

The prompt recommended removal and permitted departure with reasons; prompt 05 had shipped the
other option (`Q` as an `XSplineWrapper` returning `raw_theta(x)/x`, on the argument that with
`phi = 0` and no cycle rebasing this *is* the old pre-offset ODE state).

Alternatives weighed:

* **Keep `Q` as `raw_theta/x`.** Numerically defensible — the two quantities do coincide now. But
  the member's *meaning* was "the state the ODE advances, before the offset", and there is no ODE,
  no offset and no state. A consumer reading `Q` to diagnose the construction would be reading a
  quantity that no longer has any privileged relationship to it, and it inherits `raw_theta`'s
  `eps*theta` limit (2.2e-1 rad at `x = 1e15`) while looking like a smooth O(1) diagnostic.
  `DRAFT-PLAN.md` §8.1's prohibition is on exactly this shape of survival.
* **Keep it and deprecate it with a warning.** Rejected: a `DeprecationWarning` on a `dict`
  lookup needs the same `dict` subclass this uses, and the only two consumers are diagnostics that
  this commit edits anyway. A deprecation period buys nothing when the migration is two lines.
* **Remove it (taken).** Both live consumers are migrated to `phase.residual` in this commit, and
  the removal is loud: `_BesselPhaseData.__missing__` names the replacement. The one cost is that
  the returned object is no longer a bare `dict`; `_BesselPhaseData` is a module-level subclass
  adding no state, `isinstance(x, dict)` still holds, and both pickle codecs round-trip it.

### 2. `theta_abserr_at` is a new method, not a callable `phase.theta_abserr` — IMPLEMENTATION CHOICE

The prompt asked for "a `theta_abserr` accessor … may be a scalar or a callable". Making
`phase.theta_abserr` itself callable would have shadowed the `float` attribute prompt 05 published
and handed over, and a value that is both a number and a function of `x` is the kind of member that
reads as a bug. The alternative considered was an object with `__call__` **and** `__float__`;
rejected as cleverness. Instead the scalar keeps its name and the callable gets a distinct one,
matching `AdaptiveLevin`'s own internal spelling (`_Basis_SinCos.theta_abserr_at`) so the two ends
of the contract are named the same thing.

### 3. `residual` is not aliased as `r_nu` — IMPLEMENTATION CHOICE

Prompt §2.2 says to "expose `r_nu(x, x_is_log=False)` and `log_amplitude(x, …)`". `log_amplitude`
did not exist and was added. `r_nu` **does** exist, under the name `residual`, which prompt 05
shipped and which board standing note 19 already tells prompt 07 to use. Adding a second public
spelling of one quantity invites the two to drift apart and doubles what prompt 07 has to be told;
the accessor prompt 06 was asked to provide is provided, under the name the tree already uses.

### 4. A new test module rather than extending `test_bessel_two_region.py` — IMPLEMENTATION CHOICE

The prompt offered the choice. `test_bessel_two_region.py` was already 783 lines and is the
*numerical acceptance* module (references, error metrics, the acceptance table); the adapter
surface, serialization and Levin integration are a different concern and would have doubled it.
`test_bessel_compatibility.py` also builds only cheap orders (1/2, 3/2, 5/2, 20.5 plus the corner
sweep), so it runs in 2.2 s and can be run on its own while iterating. Two consequences: `phi` and
the deprecation shim are now asserted in both modules (cheap, and a test file should stand alone),
and prompt 08 inherits two files to consider rather than one.

### 5. `python -c "import main"` cannot be run — STRUCTURALLY REQUIRED

Prompt §6 lists it as an acceptance check. `main.py` parses `sys.argv` at module scope
(`main.py:88-...`, `parser.parse_args()`), so importing it exits through `argparse` before reaching
anything else — as `CLAUDE.md` and log 05 both record. Substituted, and all three pass:

* `python -m py_compile main.py` — OK.
* an `ast` walk of `main.py` for `bessel_phase` calls: `[(524, ['phase_atol', 'amplitude_rtol']),
  (532, ['phase_atol', 'amplitude_rtol'])]`, i.e. both production call sites migrated and no
  `atol`/`rtol` left.
* the two calls executed with production-shaped arguments
  (`largest_x_with_clearance = 1.075e15`), reported below.

### 6. `theta_abserr` is not fed from prompt 05's fields unmodified — STRUCTURALLY REQUIRED

Prompt §2.1 says to feed the accessor "from prompt 05's achieved-accuracy fields", and prompt §5
item 2 requires the declared value never to under-report. **Doing the first exactly fails the
second.** Two measurements:

* The tail term prompt 05 declares is `crossover.first_omitted_at_x_star`, and the tail model needs
  the same estimator at general `x`. `bessel_tail.tail_first_omitted_term`'s own docstring records
  the true `|delta r|` as **0.98–1.02×** it. Measured here over 300 points on `[x_star, 3 x_star]`
  at `nu` = 3/2, 7/4, 5/2, 20.5, 100.5 (1500 points, 50-digit `mpmath`), the worst
  measured/estimator ratio was **0.99994** — always below 1, with 0.006 % to spare. Declaring an
  asymptotic remainder *estimator* 1:1 as a bound is not a margin, and nothing obliges the sign of
  that 0.006 % to survive a change of `tail_terms` or an untested order.
* Combining the tail terms with `max()` rather than by summation under-reported outright:
  **3 %** at `nu = 20.5, x = 2050` (measured 9.714e-16 against a max-model 9.389e-16), where the
  series remainder and the 4-eps arithmetic floor are the same size.

Shipped: the tail terms are **summed**, and the series remainder carries
`DECLARED_SERIES_SAFETY = 2.0` wherever it is *declared* (never where it sizes the crossover, which
`bessel_tail` owns and which already applies its own `safety = 0.25`). The visible consequence is
that the low-order declared `theta_abserr` moves from prompt 05's **2.5e-12 to 5.0e-12** — still a
factor of two inside the 1e-11 the caller requested and 200× inside the campaign's 1e-6 high-order
target — and that the worst declared/measured ratio at any cached corner rises from 1.04 to
**2.05**. `amplitude_relerr` and `theta_deriv_relerr` are unaffected in practice: their tail terms
never bind (the near-region sampling floor is 3e-13 against a tail amplitude term of ~1e-15).

Nothing in the orchestrator's stop list is touched: the zero point, the sign convention, the
Wronskian relation, the exactness of README §2 (a), which series supplies the tail, the two-region
structure and the `theta' = e^{-2 ell}` route are all unchanged. What changed is a declared error
becoming larger, i.e. more conservative.

### 7. Two diagnostic output filenames changed — IMPLEMENTATION CHOICE

`Q.pdf` → `Bessel_residual.pdf` in `QuadSourceIntegral_debug.py`, and `Q_plot_nu=*.pdf` →
`residual_plot_nu=*.pdf` in `plot_besssel_phase.py`. Keeping the old filenames for a panel that now
plots a different quantity is the same failure mode as keeping the `Q` key, one directory further
out: someone comparing an old `Q.pdf` against a new one would be comparing `theta/x` with `r_nu`.
Nothing consumes these paths programmatically (grepped).

### 8. `accuracy` gains a key — IMPLEMENTATION CHOICE

`accuracy` is now 19 keys, not log 05's 18: `"declared_series_safety"` was added, and
`"tail_first_omitted_at_x_star"`/`"tail_amplitude_at_x_star"` now hold the **raw** estimator rather
than the declared value. Without that, a reader seeing `theta_abserr = 5.0e-12` and
`tail_first_omitted_at_x_star = 5.0e-12` could not tell that the measurement was 2.5e-12 and the
rest is margin. Additive; nothing in the tree reads `accuracy` by key except the tests.

### 9. Production accuracy set to 1e-12, not the 1e-11 default — IMPLEMENTATION CHOICE

The prompt asked for "values that meet README §6's low-order row (1e-11) with margin". The default
1e-11 achieves a declared 5.0e-12, i.e. a factor of two — thin for a headline claim. 1e-12 achieves
a declared 5.0e-13 (`nu = 5/2`) at a cost of 3.4 ms and `x_star` moving 64.5 → 89.6, a factor of
20 of margin. 1e-13 was measured and rejected: it buys **nothing** (the declaration floors at
5.0e-13, the amplitude at 3.0e-13, both set by the scaled-Hankel sampling floor) while widening the
sampled near region by another 0.33 e-fold.

### 10. The dense `theta_abserr` sweep is scored against `mpmath`, not SciPy — IMPLEMENTATION CHOICE

The first version of that test scored against `bessel_reference.scipy_reference` and failed at
`nu = 1/2, x = 16.2576`, reporting a measured error of **1.07e-14** against a declared 8.9e-16.
At `nu = 1/2` the representation is *exact* (`r == 0`, `a == 1`) and agrees with the closed form to
an ulp, so the whole 1.07e-14 is Amos's own error — SciPy is not an adequate reference for a
declaration this small, two orders of `nu` below where standing note 15 measured that floor. The
test now uses `TIER_MPMATH` throughout, which costs ~2.7 ms per point (1000 points, 0.8 s).

## Verification performed

Everything below was run. Nothing here is "I reasoned that this is correct".

### Tests

| Module | Result |
|---|---|
| `LiouvilleGreen.tests.test_bessel_compatibility` (new) | **20 tests, OK**, 2.2 s |
| `LiouvilleGreen.tests.test_bessel_two_region` | **24 tests, OK**, 3.8 s |
| `LiouvilleGreen.tests.test_bessel_phase` | **4 tests, OK**, 0.18 s — including `test_phase_derivative` at 1e-6, unchanged and untouched (`RECONCILIATION.md` C4) |
| `LiouvilleGreen.tests.test_three_bessel` | **2 tests, OK**, 4.9 s |
| `LiouvilleGreen.tests.test_bessel_near_region` | 27 tests, OK, 0.26 s |
| `LiouvilleGreen.tests.test_bessel_reference` | 12 tests, OK, 0.31 s |
| `LiouvilleGreen.tests.test_bessel_tail` | 19 tests, OK, 0.15 s |
| `LiouvilleGreen.tests.test_range_reduce` | 4 tests, OK |
| `LiouvilleGreen.tests.test_scipy_bessel_domain` | 7 tests, OK, 0.03 s |
| `LiouvilleGreen.tests.test_3bessel_analytic` | **started, made normal progress, not seen to completion** — the pre-existing multi-hour module of `RECONCILIATION.md` §3.4 and issue `[05-3bessel-analytic-not-run-to-completion]`. Only the expected `DeprecationWarning`s (prompt 08 owns those call sites). |
| `unittest discover -s ComputeTargets/tests -t .` | **97 tests, OK**, 125.9 s — the workstream-B fixtures (`test_tk_source_functions`, `test_phase_groups`) are the canary for an adapter slip, and `test_quadsource_integral` exercises `BesselPhaseProxy`'s consumer |

`unittest discover -s LiouvilleGreen/tests -t .` as a single run is therefore reported per module,
per `RECONCILIATION.md` §3.4's instruction: every module was run and passed except the pre-existing
multi-hour one.

### The declared error against the measured phase-pair error

Worst `theta_abserr_at(x) / E_theta(x)` at the same `x`, per order and per region, against the
committed 40-digit `mpmath` corners. **Ratios are declared/measured, so > 1 is the safe
direction**; `-` means that order has no corner in that region.

| nu | region | worst declared/measured (at x) | declared there | measured there |
|---|---|---:|---:|---:|
| 0.5 | near | - (no near region) | - | - |
| 0.5 | tail | 8.00 (x=10) | 8.882e-16 | 1.110e-16 |
| 1.5 | near | 22518 (x=15) | 5.000e-12 | 2.220e-16 |
| 1.5 | tail | 4.00 (x=1e+12) | 8.882e-16 | 2.220e-16 |
| 1.75 | near | 6320.8 (x=17.5) | 5.000e-12 | 7.910e-16 |
| 1.75 | tail | 3.21 (x=175) | 2.319e-15 | 7.216e-16 |
| 2.5 | near | 9481.3 (x=3.674) | 5.000e-12 | 5.274e-16 |
| 2.5 | tail | 2.67 (x=1000) | 8.889e-16 | 3.331e-16 |
| 20.5 | near | 1441.2 (x=30.74) | 5.000e-12 | 3.469e-15 |
| 20.5 | tail | 2.16 (x=1000) | 2.866e-13 | 1.324e-13 |
| 100.5 | near | 58.07 (x=100.5) | 5.000e-12 | 8.610e-14 |
| 100.5 | tail | 2.54 (x=1.005e+04) | 1.213e-14 | 4.774e-15 |
| 1000.5 | near | 20.47 (x=1e+04) | 5.000e-12 | 2.443e-13 |
| 1000.5 | tail | 2.05 (x=1e+05) | 1.137e-13 | 5.547e-14 |

**Overall worst ratio 2.05**, at `nu = 1000.5, x = 1e5` — just above that order's `x_star`, i.e. it
is the series remainder and the 2.0 is `DECLARED_SERIES_SAFETY` doing exactly the job Deviation 6
introduced it for. On the dense `mpmath` sweep (both regions, `nu` = 1/2, 3/2, 7/4, 5/2, 20.5,
1000 points), the worst ratio is **2.01** at `nu = 20.5, x = 979.97`. Both are asserted.

The spread across regions is the argument for the callable: at `nu = 5/2` the near region declares
5.000e-12 and the tail at `x = 1e7` declares 8.882e-16, a factor of **5.6e3**.

### The Levin call, before and after declaring `theta_abserr`

`adaptive_levin_sincos` on `integral A_nu sin theta_nu dx = integral J_nu dx`, `nu = 5/2`, with
`{theta: raw_theta, theta_mod_2pi, theta_deriv}` and `atol = rtol = 1e-10`. The reference is
40-digit `mpmath`, split at 2 pi intervals.

| span | region | declared | `abserr` | `abserr_roundoff` | `abserr_resolution` | converged |
|---|---|---|---:|---:|---:|---|
| (10, 60) | near | none | 3.536275e-13 | 2.437887e-14 | 2.151057e-15 | True |
| (10, 60) | near | `theta_abserr_at` | **2.538774e-11** | 2.538774e-11 | 2.151057e-15 | True |
| (1000, 1100) | tail | none | 8.982385e-16 | 8.982385e-16 | 1.578598e-16 | True |
| (1000, 1100) | tail | `theta_abserr_at` | **9.420688e-16** | 9.420688e-16 | 1.578598e-16 | True |
| (1000, 1100) | tail | `theta_abserr` (scalar) | 2.474591e-13 | 2.474591e-13 | 1.578598e-16 | True |

* **The concrete demonstration the prompt asks for:** on the near-region span the reported `abserr`
  rises from 3.536e-13 to **2.539e-11** when the object declares its own phase error, and the
  reported error is then dominated by that declaration (`abserr_roundoff` exceeds
  `abserr_resolution` by 1.2e4). The value is bit-identical between the two runs — declaring an
  error is bookkeeping, not a change of quadrature.
* **Why a callable and not the scalar:** on the tail span the scalar declaration inflates the
  reported error by **262.7×** (2.475e-13 against 9.421e-16) for no reason at all, since nothing
  above `x_star` interpolates anything or touches `hankel1e`.
* Values against `mpmath`: 8.2275350693216759e-02 vs 8.2275350693218757e-02 (difference
  **1.998e-15**) on (10, 60); 7.5784805462043650e-03 vs 7.5784805462043676e-03 (difference
  **2.602e-18**) on (1000, 1100).
* **`need_theta_Cheb` did not fire.** Both spans were re-run with a `theta` callable that raises
  `AssertionError`, and both produced bit-identical `value` and `abserr`. That is the campaign's
  evidence for `RECONCILIATION.md` §1's reading of `levin_quadrature.py:1038`. Deleting the
  `"theta"` key instead raises `RuntimeError`, confirming it is required rather than optional
  (`:948-952`).

### Serialization

**Bit-identical, both codecs.** `pickle.dumps`/`loads` and `ray.cloudpickle.dumps`/`loads` were
each round-tripped and every accessor compared with `assertEqual` (exact, not `almostEqual`) at
`x` = 5, 30, 64, 100, 1e4, 1e7 in both raw and logarithmic input modes, for `nu` = 1/2, 5/2 and
20.5: `raw_theta`, `theta_mod_2pi`, `theta_deriv` (both flavours), `residual`,
`residual_log_deriv`, `log_amplitude`, `sin_cos_theta`, `theta_abserr_at`, `bessel_j`, `bessel_y`,
`mod`, `mod.a`, `mod.log_deriv`, and the `phi`/`min_x`/`max_x`/`nu`/`x_star`/`theta_abserr`/
`accuracy`/`crossover` members. **What carries the interpolants:** `scipy` `BSpline`-family
objects held on `NearRegionData` (`r_interp`, `log_a_interp`), reached through the module-level
classes `_TwoRegionCorrections`, `BesselPhaseFunction` and `BesselAmplitude`. There are no closures
over locals anywhere in the returned object, which is why plain `pickle` suffices — log 05's
observation 6, now exercised through the `ray.cloudpickle` codec `BesselPhaseProxy` actually uses.
The `theta_abserr_at` callable is a bound method of a module-level class and round-trips like the
rest; the revived mapping still raises the informative `KeyError` for `"Q"`.

### The two migrated diagnostics

* `import ComputeTargets.QuadSourceIntegral_debug` — OK. `bessel_function_plot` was then run to
  completion for `nu` = 1/2 and 5/2 over `[min_x, 1e4]`, 300 grid points: **34 files** written,
  including the new `Bessel_residual.pdf`/`.png`, with no exception.
* `plot_besssel_phase.py` — **runs to completion**, 12 PDFs (`bessel_J_plot`, `bessel_Y_plot`,
  `phase_plot`, `residual_plot` for each of `nu` = 1/2, 3/2, 5/2). Reconstruction checked
  numerically on the same grid: `max|our J - jv|` = 5.1e-15 (1/2), 2.4e-13 (3/2), 2.1e-13 (5/2),
  and `max|our Y - yv|` = 2.2e-16, 3.2e-13, 2.3e-13; the residual panel spans
  [0.0, 0.0] (1/2, exact), [0.0100, 0.6155] (3/2) and [0.0300, 1.1832] (5/2), matching
  `RECONCILIATION.md` C2's `r(x_0)` figures.

### `main.py`

Cannot be imported (Deviation 5). `py_compile` OK; the `ast` walk shows both call sites carrying
`phase_atol`/`amplitude_rtol` and nothing else; `git diff HEAD~1 --stat` shows **one hunk**, at
`main.py:514-536`, inside `run_pipeline`'s Bessel construction stage. The two calls were executed
directly with `largest_x_with_clearance = 1.075e15`:

| call | `x_star` | declared `theta_abserr` | declared `amplitude_relerr` | declared `theta_deriv_relerr` | build |
|---|---:|---:|---:|---:|---:|
| `bessel_phase(0.5, 1.075e15, phase_atol=1e-12, amplitude_rtol=1e-12)` | 1e-05 (= `min_x`) | 8.882e-16 | 8.882e-16 | 1.776e-15 | 0.4 ms |
| `bessel_phase(2.5, 1.075e15, phase_atol=1e-12, amplitude_rtol=1e-12)` | 89.5791 | 5.000e-13 | 3.006e-13 | 6.011e-13 | 3.4 ms |

Both meet README §6's low-order row (1e-11) with a factor of 20 and 1.1e4 respectively.

### The consumer inventory, re-checked

`RECONCILIATION.md` §3.3 is complete for the members it lists, with **three additions** found by
grep (all pass, none needing a change here):

* `ComputeTargets/tests/test_quadsource_integral.py:244, 482-483` — builds `bessel_phase`, two of
  them with `atol`/`rtol`; part of the 97-test `ComputeTargets` run, and the only place
  `BesselPhaseProxy`'s consumer is exercised by a test.
* `docs/source-remediation-verification/run_quadsource_integrals.py:218-221` and
  `docs/spec-code-audit/scripts/QI_02_analytic_numeric.py:60-61` — both pass `atol`/`rtol`; both
  noted in log 05 observation 9 and owned by nobody. They warn and still run.

No consumer outside the two diagnostics read `Q`, and no consumer outside
`docs/transfer-remedial/measure_bessel_phase.py:283` reads `phi`.

## Observations not acted on

1. **`AdaptiveLevin/levin_quadrature.py:2750` is stale**, exactly as `DRAFT-PLAN.md` §8.1 and
   prompt §2.1 predicted: the `theta` docstring says it is "always used to decide whether a
   subinterval is oscillatory enough for the Levin rule (via the total phase change across it)",
   but `phase_span` is computed from `theta_prime_Cheb` (`:1090`) and `need_theta_Cheb` is `False`
   whenever `theta_mod_2pi` and `theta_deriv` are both supplied (`:1038`). This commit's
   exploding-`theta` test is direct evidence. `AdaptiveLevin/` may not be edited here; opened as
   `[06-levin-theta-docstring-stale]`.
2. **`docs/transfer-remedial/measure_bessel_phase.py` no longer runs against the current module.**
   `:276` and `:307` read `data["phase"].num_chunks`, which existed on `phase_spline` and does not
   exist on `BesselPhaseFunction`; prompt 05 removed it. The script's historical-module path
   (`:130`, which loads the old `bessel_phase` from git) is unaffected. Opened as
   `[06-measure-bessel-phase-num-chunks]`; `docs/` is prompt 09's.
3. **`QuadSourceIntegral_debug.three_bessel_plot` is dead in the same way
   `plot_besssel_phase.py` was.** `:271-273` and `:438-462` call `phase_A(x1)` / `phase_B(x2)`,
   i.e. they call the phase object — which neither `phase_spline` nor `BesselPhaseFunction`
   defines `__call__` for. It has been broken since before this campaign. Prompt 06 was told to
   migrate `Q` "and leave everything else alone", so it is left alone and opened as
   `[06-three-bessel-plot-calls-a-non-callable-phase]`. The fix is one line per call site
   (`.raw_theta`), but it is a second repair-or-delete judgement and it is not this prompt's.
4. **`ComputeTargets/QuadSourceIntegral.py:681`'s docstring is now doubly stale**: it lists `Q`
   among the members the returned dict "carries", as well as claiming there is no `"nu"` key.
   Already recorded as `[05-quadsource-order-check-docstring-stale]`; the `Q` half is added to that
   entry rather than opened separately. No functional impact — the numeric order check still runs
   and `test_quadsource_integral` passes.
5. **`_three_bessel_Levin` is now the only Bessel-phase consumer in the tree that could pass a
   `theta_abserr` and does not.** Before this commit nothing could. `ComputeTargets/` is out of
   scope (README §4.2); this sharpens `[00-qsi-three-bessel-levin-excluded]`, which prompt 09 hands
   to `source-remediation`, and the hand-off text should now mention `theta_abserr_at` by name
   alongside the missing `theta_deriv`.
6. **`theta_mod_2pi` costs a `sin`, a `cos` and an `atan2` more than the old `fmod`.** Not
   measured against a production profile here. It is the accessor Levin calls twice per region
   (`:1103-1104`), not per collocation point, so it is very unlikely to matter; noted because
   nothing has looked.

## State handed to the next prompt

### What prompt 07 needs to assemble `K t + C + R(t)`

For `theta_mu(k t) + eps_nu theta_nu(q t) + eps_sigma theta_sigma(s t)`, everything below is on the
object `bessel_phase()` returns under `"phase"`. **Do not difference `raw_theta` values** — it is
`eps * theta`-limited by construction, 2.2e-1 rad at `x = 1e15`.

```python
phase.c_nu                                  # float attribute: pi/4 - pi nu/2, as defined
phase.c_nu_reduced                          # float attribute: the same mod 2 pi -- sum THIS one
                                            #   when only sin/cos of the group are needed
phase.residual(x, x_is_log=False)           # R's constituents; tail-continuous branch, never
                                            #   reduced mod 2 pi (570.82 rad at nu=1000.5, x_0)
phase.residual_log_deriv(x, x_is_log=False) # dr/d log x, if a group derivative is wanted from r
phase.theta_deriv(x, x_is_log=False, log_derivative=False)   # theta'; sum these for K + R'
phase.sin_cos_theta(x, x_is_log=False)      # (sin, cos) of one factor, by angle addition
phase.theta_abserr_at(x, x_is_log=False)    # declared |delta theta| at x, in radians
phase.theta_abserr                          # the same as a domain-wide float
phase.nu, phase.x_star, phase.min_x, phase.max_x
```

* **`K` is exact**: `K = k + eps_nu q + eps_sigma s` in the *arguments*, formed before any
  multiplication by `t`. **`C` is exact**: `c_mu + eps_nu c_nu + eps_sigma c_sigma`, and for
  trigonometric use sum `c_nu_reduced` instead — each term is a single rounding of `pi q` with
  `|q| <= 2`, below 7e-16, whereas summing `c_nu` at high order injects that order's ulp
  (2.3e-13 at `nu = 1000.5`).
* **`R(t)` is the only sampled part**: `r_mu(kt) + eps_nu r_nu(qt) + eps_sigma r_sigma(st)`.
* **The declared error of a group** adds: `theta_abserr_at` of each factor at its own argument, and
  they are independent, so a sum (not a max) is the honest combination. That sum is what
  `adaptive_levin_sincos`'s `theta_abserr` key wants for a phase group.

### The `AdaptiveLevin` phase dictionary, settled

```python
theta = {
    "theta":         phase.raw_theta,        # REQUIRED (:948-952) though never evaluated (:1038)
    "theta_mod_2pi": phase.theta_mod_2pi,    # (-pi, pi] from atan2 of the split pair
    "theta_deriv":   phase.theta_deriv,      # double duty: basis conditioning AND subdivision
    "theta_abserr":  phase.theta_abserr_at,  # callable of x; prefer it to the scalar
}
```

Supplying `theta_abserr` is measured to raise the reported `abserr` by 71.8× in the near region and
1.05× in the tail (numbers above). That is the accuracy claim reaching the consumer, and it is a
*larger* reported error than before — a prompt-07 test that expects `abserr` to shrink is expecting
the wrong thing.

### Numbers a later prompt should not re-derive

* Declared `theta_abserr` at the shipped defaults (`phase_atol = amplitude_rtol = 1e-11`):
  8.882e-16 at `nu = 1/2`; **5.000e-12** at every other order the campaign covers (it is the tail
  series term at `x_star`, `2 x 0.25 x phase_atol`, which dominates the near-region terms at all of
  them). Prompt 05's log records 2.500e-12 for these; Deviation 6 is why they moved.
* Production (`main.py`, `phase_atol = amplitude_rtol = 1e-12`): declared `theta_abserr` 8.882e-16
  at `nu = 1/2` and 5.000e-13 at `nu = 5/2`; `x_star` = `min_x` and 89.5791; builds 0.4 ms and
  3.4 ms at `largest_x_with_clearance = 1.075e15`.
* `DECLARED_SERIES_SAFETY = 2.0` is applied to the series remainder in the declared errors only.
  `bessel_tail.DEFAULT_CROSSOVER_SAFETY = 0.25` is unchanged and still the only thing that sizes
  `x_star`; the two are independent and must not be conflated.
* `accuracy` is now 19 keys. `tail_first_omitted_at_x_star` and `tail_amplitude_at_x_star` are the
  **raw** estimators; multiply by `accuracy["declared_series_safety"]` to recover the declared
  contribution.
* `data` is a `_BesselPhaseData`, a `dict` subclass. `isinstance(data, dict)` is `True`,
  `data.get("Q")` is `None`, `data["Q"]` raises a `KeyError` naming the replacement, and both
  `pickle` and `ray.cloudpickle` round-trip it with the behaviour intact.
* `test_bessel_compatibility.py` is 2.2 s and builds only `nu` = 1/2, 3/2, 5/2, 20.5 plus the
  corner sweep at all seven orders; `test_bessel_two_region.py` is 3.8 s. Prompt 08 owns
  `test_bessel_phase.py` and `test_3bessel_analytic.py`, whose `atol`/`rtol` call sites still emit
  one `DeprecationWarning` each.
