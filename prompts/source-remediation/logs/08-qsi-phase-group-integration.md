# Log 08 — Region partition and phase-group Levin integration in `QuadSourceIntegral` (A4 part 2, A2 part 3)

**Prompt:** prompts/source-remediation/08-qsi-phase-group-integration.md
**Commit:** *(this commit)* — "Partition the source time integral and Levin-integrate its phase groups"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Fable 5.1
**Date:** 2026-09-09
**Result:** COMPLETE WITH DEVIATIONS

Two of the deviations below are ones the orchestrator's stop rules single out, and both are
measurements rather than choices: the §6 cost ratio is **above 3×** on the $b=0$ cases (deviation
10, board issue `[08-levin-fallback-cost-ratio]`), and the realistic-fixture oracle cannot meet the
prompt's $10^{-5}$ (deviation 1). A third finding — the hand-over clamp the board prescribes costs
**5.1e-3 of `total`** for a one-grid-step gap, the largest single error term in the chain
(deviation 2, board issue `[08-handover-clamp-error]`) — is not covered by any stop rule but needs
the user's eye.

## What shipped

### `ComputeTargets/QuadSourceIntegral.py` (1502 → 1949 lines)

Old line numbers refer to the pre-commit file (`39ed7fc`).

- **Constants** (`:26-27` before → `:26-85` after). `LEVIN_MIN_2PI_CYCLES`/`LEVIN_MIN_PHASE_DIFF`
  kept, now unused, under a comment explaining the retired Green's-function-only gate (audit QI-6)
  and that they wait on prompt 10's `QuadSourcePolicy` decision. `CHEBYSHEV_ORDER`, `LEVIN_RELERR`,
  `LEVIN_ABSERR` unchanged. New: `LEVIN_USE_THETA_DERIV = True` (§4 decision, deviation 9),
  `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` and `MIN_SUBINTERVAL_LOG_WIDTH = DEFAULT_FLOAT_PRECISION`
  (deviations 2 and 5).
- **`get_z`** (`:63-70` → `:98-106`): also accepts any object exposing `.z` (the audit's `QI_*`
  stand-ins and the new tests).
- **New clamping adapters** `_ClampedPhase`, `_ClampedTk`, `_ClampedSource` (`:122-217`). They
  present the `TkSourceFunctions` protocol `phase_groups` reads (`T`, `dT_dz`; `M`, `dlnM_dz`,
  `omega`, `phase`) and the source spline, with $\log(1+z')$ pinned into the evaluable range of
  the representation in use (`numeric_region`, open above, for a smooth factor; `WKB_region` for
  an oscillatory one; `source.numeric_region` for $f$).
- **New `build_partition(GkPolicy, Tq_f, Tr_f, source, z_response, z_source_max)`** (`:282-495`).
  Breakpoints: `GkPolicy.crossover_z` for type `mixed` (type `numeric` = $G$ smooth throughout,
  `WKB` = oscillatory throughout), `Tq_f.crossover_z`, `Tr_f.crossover_z`; clipped to the range,
  sorted descending, deduplicated at `MIN_SUBINTERVAL_LOG_WIDTH`. Each sub-interval gets a regime
  `(G_osc, q_osc, r_osc)` (oscillatory iff the sub-interval lies below the factor's breakpoint),
  a method (`"quad"` for `(F,F,F)`, `"Levin"` otherwise), and a validity check: $G$'s regions are
  checked strictly (as before); each $T$'s representation may fall short of a sub-interval end by
  at most `HANDOVER_CLAMP_MAX_GRID_STEPS` mean source-grid steps in $\log(1+z)$ (bridged by the
  clamp, gap recorded), else `RuntimeError`; `z_response` below an oscillatory $T$'s lowest LG
  sample raises. Returns the sub-interval list and the JSON record stored as
  `metadata["partition"]`.
- **New `evaluate_QuadSource_integral(model, k, q, r, source, GkPolicy, z_response, z_source_max,
  b, Bessel_0pt5, Bessel_2pt5, Tq_numeric, Tq_WKB, Tr_numeric, Tr_WKB, atol, rtol,
  Tk_functions_builder=TkSourceFunctions)`** (`:574-781`): the task body as a plain function.
  Builds `Tq_f`/`Tr_f` via `Tk_functions_builder(model, q.k, Tq_numeric, Tq_WKB)`, checks their
  `crossover_z` against `source.crossover_z_q/_r` when those are not `None`, partitions,
  integrates each sub-interval (quad → `numeric_quad_integral` with the clamped source; Levin →
  `phase_group_Levin_integral`), distributing `atol` by $\log(1+z')$-length share, then calls
  `analytic_integral` exactly as before.
- **`compute_QuadSource_integral`** (`@ray.remote`, `:784-826`): now a thin wrapper that resolves
  `model_proxy.get()` and the two `BesselPhaseProxy.get()` and calls the plain function. Four new
  keyword parameters `Tq_numeric, Tq_WKB, Tr_numeric, Tr_WKB` between `Bessel_2pt5` and `atol`.
- **New `phase_group_Levin_integral(model, k, q, r, regime, Gk_f, Tq_f, Tr_f, z_response, max_z,
  min_z, atol, rtol)`** (`:829-915`): `build_phase_groups(regime, Gk=Gk_f if G_osc else
  Gk_f.numeric_Gk, Tq=..., Tr=..., model_functions=model.functions,
  w_background=model.functions.wBackground)`, then one `adaptive_levin_sincos(x_span,
  [g.f_sin, g.f_cos], theta=g.levin_theta(include_deriv=LEVIN_USE_THETA_DERIV),
  atol=atol/n_groups, rtol=rtol, chebyshev_order=CHEBYSHEV_ORDER, notify_label=...)` per group.
  Values and `abserr` summed linearly, `converged = all`, `phase_limited = any`, all scaled by
  $(1+z_{\rm resp})$. Returns per-group metadata and the raw driver dicts for aggregation.
- **Aggregators** `_aggregate_IntegrationData` (sum times/steps/evaluations, evaluation-weighted
  mean RHS time, min/max RHS times) and `_aggregate_LevinData` (sum `num_regions`, `evaluations`,
  `num_simple_regions`, `num_SVD_errors`, `num_order_changes`; min `chebyshev_min_order`; max
  `max_depth`; sum `elapsed`) (`:526-571`).
- **`numeric_quad_integral`** (`:910-977` → `:1500-1581`): unchanged integrand and measure;
  gains `source_f: Optional[Callable] = None` (a callable $f(\log(1+z'),\,{\tt z\_is\_log=True})$
  standing in for `source.functions.source`) and now scales the returned `"abserr"` by
  $(1+z_{\rm resp})$ like `"value"`. The `GkPolicy._source_proxy` reads in its error messages go
  through `_Gk_diagnostics`, which tolerates stand-ins.
- **Deleted** `WKB_quad_integral` (`:980-1047`) and `WKB_Levin_integral` (`:1050-1173`)
  (deviation 4).
- **`compute_QuadSource_integral`'s three-region body** (`:96-278`) replaced as above. **Return
  dict keys unchanged**; meanings per prompt §5:

  | key | value now |
  |---|---|
  | `numeric_quad` | sum over `(F,F,F)` sub-intervals |
  | `WKB_quad` | `0.0` |
  | `WKB_Levin` | sum over every sub-interval with an oscillatory factor |
  | `total` | `numeric_quad + WKB_Levin` |
  | `numeric_quad_data` | aggregate `IntegrationData`, or `None` if no smooth sub-interval |
  | `WKB_quad_data` | `None` |
  | `WKB_Levin_data` | aggregate `LevinData`, or `None` if no Levin sub-interval |
  | `WKB_phase_spline_chunks` | `getattr(Gk_f.phase, "num_chunks", None)` |
  | `metadata["analytic"]` | unchanged |
  | `metadata["numeric_quad"]` | new: `{"abserr", "subintervals": [{z_max, z_min, value, abserr}]}` |
  | `metadata["WKB_Levin"]` | `{"abserr", "converged", "phase_limited", "theta_deriv_supplied", "subintervals": [{z_max, z_min, regime, value, abserr, converged, phase_limited, groups: [{label, value, abserr, converged, phase_limited, regions, simple_regions, evaluations, elapsed}]}]}` — always a dict (deviation 6) |
  | `metadata["partition"]` | new: `{z_source_max, z_response, G_type, breakpoints: [{factor, z, inside_range}], crossover_z_q, crossover_z_r, Levin_z_unused, max_clamp_gap_log1pz, subintervals: [{z_max, z_min, regime, method, clamp_gaps_log1pz}]}` |

- **`QuadSourceIntegral.compute()`** (`:1398-1465` → `:1807-1908`): new class constant
  `REQUIRED_PAYLOAD_KEYS`; a missing key raises `RuntimeError` naming every missing key (checked
  first, before any object is touched); after the existing checks, each of the four `Tk` objects
  must have `Tk.k.store_id` equal to the matching `q`/`r` exit-time's `k.store_id`; the four
  objects are forwarded to the remote call.
- **Untouched:** `_three_bessel_integrals`, `_three_bessel_Levin`, `_three_bessel_quad`,
  `analytic_integral`, `_extract_z` (`:330-907`, byte-identical), `BesselPhaseProxy`, the
  `QuadSourceIntegral` constructor, properties and `store()`.

### `ComputeTargets/tests/test_quadsource_integral.py` (new, 1125 lines, 15 tests, ~140 s)

Offline, no Ray, no datastore. Stand-ins `FakeWavenumber`, `FakeExitTime`, `FakeZ`,
`FakeZSample`, `FakeQuadSource` (spline of the exact $f$ on the production grid, or the exact $f$),
`FakeGkPolicy` (a `GkSourceFunctions` around `ExactGk`/`OffsetBesselPhaseGk`),
`ExactTkFunctions` (log 07's `ExactTk` plus region bookkeeping, injected through
`Tk_functions_builder`), `exact_Tk_inputs(fixture)` (captures the `(Tk_numeric, Tk_WKB)`
stand-ins prompt 05's `Fixture.exact_functions()` builds), three `Shape`s, three response
redshifts, and a `Case` class that assembles one integral in either the exact or the realistic
flavour and calls `evaluate_QuadSource_integral` directly. Tests: oracle 1 (exact and realistic),
oracle 2, regime coverage, partition structure, `GkSourcePolicyData` types, $q=r$, output
population, artificial split, hand-over seam, clamp measurement and its two failure modes,
old-regime regression against the pre-commit file, `compute()` payload check.

### Not touched

`main.py`, `Datastore/`, `MetadataConcepts/QuadSourcePolicy.py`, `ComputeTargets/phase_groups.py`,
`ComputeTargets/TkSourceFunctions.py`, `ComputeTargets/QuadSource.py`, `AdaptiveLevin/`,
`LiouvilleGreen/`, `docs/`.

## Deviations from the prompt

### 1. The realistic-fixture oracle is asserted at the representation floor, not at $10^{-5}$ — STRUCTURALLY REQUIRED

§7 item 1 asks `total` vs `analytic_rad` to $10^{-5}$ "with the threshold justified from the
fixture floors measured in prompts 05 and 07 (state the arithmetic)". The arithmetic does not
give $10^{-5}$. The floors on the board are pointwise, relative to the local envelope:
4.5e-4 for the QuadSource spline of $f$ at the hand-over (`[06-source-spline-residual-vs-handover]`;
here the hand-over is at $x\approx19$, ~6 cycles of $f$, slightly worse), 7e-6 ($b=0$) to
1.4e-4 ($b=0.25$) for the LG closed forms just below it (`[07-lg-derivative-truncation-at-handover]`),
6e-6 for the re-splined phase. Integrated, the smooth-region error is that floor times the
smooth-region contribution, and **`numeric_quad` and `WKB_Levin` cancel** — by a factor 4.8 at
$b=0$, "together", $x_{\rm resp}=980$ and 5.0 at $b=0.2$, "q-smooth", $x_{\rm resp}=100$ — so the
relative error of `total` is the floor times $|{\tt numeric\_quad}/{\tt total}|$: measured
2.38e-3 on the first of those cases, of which 2.21e-3 is the smooth part
($4.55\times10^{-4}\times4.8$, matching the board's number exactly) and 1.7e-4 the Levin part.

Shipped: the exact-ingredient test asserts $10^{-5}$ (met: worst 1.36e-6, which is
`analytic_rad`'s own floor, audit QI-1) and the realistic test asserts on the two parts
separately against the exact-ingredient run of the same case, normalised by
`scale = max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)`: smooth part $<10^{-3}$ (met: 4.55e-4),
Levin part $<5\times10^{-4}$ (met: 1.30e-4), `total` $<1.5\times10^{-3}$ (met: 4.91e-4). The full
attribution table is under "Verification performed". The integrator itself is at the
$10^{-13}$ level (oracle 2).

### 2. The hand-over gap is bridged by clamping, with a tolerance; the clamp costs 5.1e-3 — IMPLEMENTATION CHOICE (board-directed), plus a measurement the user must see

§3 says to "assert, with informative errors" that every sub-interval lies inside each factor's
chosen representation. Board §5 note 6 (binding) says `WKB_region[0]` sits up to one grid step
*below* `crossover_z` in production (the WKB grid is `z_source_sample.truncate(z_init,
keep="lower")`, `main.py:695-697`, and `TkWKBIntegration` stores no sample at `z_init`), and
prescribes "partition on `crossover_z`, but clamp quadrature nodes to `numeric_region`/`WKB_region`".
A strict assertion would make every production integral fail. Shipped: the partition is at
`crossover_z`; each $T$'s accessors are clamped into the range of the representation in use
(`_ClampedTk`); the shortfall is allowed up to `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` mean
source-grid steps in $\log(1+z)$ (computed from `source.z_sample`) and raises beyond that; the gap
is recorded per sub-interval in `metadata["partition"]["subintervals"][i]["clamp_gaps_log1pz"]`.
The same clamp is applied to the source spline (`_ClampedSource`) for the sliver log 06 deviation 1
can leave between `max(crossover_z_q, crossover_z_r)` and `source.numeric_region[1]`.
Alternatives: (a) strict assertion — rejected, blocks production; (b) first-order Taylor
extension of the LG phase and amplitude across the gap using the closed-form `omega`/`dlnM_dz`
— would cut the error below by ~100× (phase error $\tfrac12\theta''\Delta^2\approx5\times10^{-3}$ rad
against $0.44$ rad for the clamp at $x\approx19$), but extends `TkSourceFunctions`'s representation
outside what prompt 05 certified and is not what the board says; not done.

**Measured** (`TestHandOverClamp`, realistic fixtures, $b=0$, "together", $x_{\rm resp}=100$,
`Fixture(drop_first_WKB_sample=True)` for $T_r$): gap 2.30e-2 in $\log(1+z)$ = 1.00 grid step;
`total` with the gap vs the same case without: **5.14e-3 relative**; with the gap vs
`analytic_rad`: 5.11e-3 (without: 2.6e-5). The error scales as gap$^2$, so a uniformly
distributed gap averages ~1/3 of this — still an order of magnitude above every other floor.
Recorded as board issue `[08-handover-clamp-error]`; the remedy is upstream (an LG sample at
`z_init`, or the overlap of `docs/lg-phase-and-handover-followup-2026-09.md` §1.4), or option (b)
above if the user prefers it done here.

### 3. The task body is a plain function with a `Tk_functions_builder` hook — IMPLEMENTATION CHOICE

§7 says to "call the underlying function, not the Ray remote — check how `QI_02` did it". `QI_02`
called `analytic_integral`, a plain function; the region logic lived inside the `@ray.remote`
body and was reachable only through Ray's private `RemoteFunction._function`. Shipped:
`evaluate_QuadSource_integral` (plain) does everything; the remote resolves the proxies and calls
it. The keyword `Tk_functions_builder=TkSourceFunctions` lets a test substitute an exact
two-region representation (deviation 11) — production never passes it. Alternative: patching
`ComputeTargets.QuadSourceIntegral.TkSourceFunctions` in tests; rejected as more fragile than an
explicit seam.

### 4. `WKB_quad_integral` and `WKB_Levin_integral` are deleted — IMPLEMENTATION CHOICE

§1 replaces the three-region scheme and §5 sets `WKB_quad` to 0.0. Both functions became dead
code and re-implement the rejected scheme (direct quadrature of an oscillatory $G$; Levin with
$f$ in the amplitude slot). Kept would have invited a caller to reach for them. The old
`WKB_Levin_integral` integrand survives as the `G`-only phase group (log 07 item 3), and the
regression test runs the pre-commit functions from `git show 39ed7fc:...`. Cost: the audit's
`docs/spec-code-audit/scripts/QI_03_measure.py` sections 2–4 no longer run (observation 2).

### 5. The region-nonempty guard is in $\log(1+z)$ — STRUCTURALLY REQUIRED (touches B11's sites)

Audit B11 (`:194, :220, :246`, a ratio in $z$) is assigned to prompt 09. Those lines no longer
exist: the partition is new code, and it has to decide when two breakpoints coincide. It does so
with `MIN_SUBINTERVAL_LOG_WIDTH = DEFAULT_FLOAT_PRECISION` in $\log(1+z)$, which is B11's fix in
substance. Nothing else of B11 was done; prompt 09 should confirm and close it.

### 6. `metadata["WKB_Levin"]` is always a dict; two metadata keys added — IMPLEMENTATION CHOICE

Before, `metadata["WKB_Levin"]` was `None` when no Levin region ran. §5 says to "extend" it; a
consumer (prompt 09's error propagation, B8) is simpler if the shape is fixed, so it is now
always `{"abserr": 0.0, "converged": True, "phase_limited": False, "theta_deriv_supplied",
"subintervals": []}` at minimum. `metadata["numeric_quad"]` (the quad error estimates §4 asks to
return; nowhere else to put them without a schema change) and `"theta_deriv_supplied"` are new.
The persisted `metadata` column is free-form JSON, so the factory is unaffected.

### 7. `TkSourceFunctions` receives `q.k`, not `q` — STRUCTURALLY REQUIRED

§2 implies the exit-time objects. `wavenumber_exit_time` has no `__float__`
(`CosmologyConcepts/wavenumber.py:107-`; only `wavenumber` has one, `:47`), and
`TkSourceFunctions` needs `float(k)`. It is given `q.k`; `z_exit` for its consistency check is
then resolved from `Tk_numeric.z_exit`, which log 05 deviation 4 provides for.

### 8. `atol` distribution — IMPLEMENTATION CHOICE (the prompt asked which)

Each sub-interval receives `atol × (its log(1+z') width) / (total width)`, mirroring what
`adaptive_levin_sincos` does internally across its own sub-regions since levin-refactor prompt 05
(audit C3), so the partition is transparent to the driver's contract; within a Levin sub-interval
each group receives `atol_sub / n_groups`, so the linear sum of group error bounds is bounded by
`atol_sub`. `rtol` is passed unchanged everywhere (a relative tolerance has no additive share).
The alternative `atol / (n_groups · n_subintervals)` over-constrains short sub-intervals for no
gain. At the production `atol = 1e-25` none of this binds; `rtol` does.

### 9. `theta_deriv` is passed to the driver — IMPLEMENTATION CHOICE, with the numbers

§4 asks for an end-to-end validation at $|\theta|\sim10^5$–$10^6$. Exact ingredients (log 07's
`ExactTk`/`ExactGk`), all-oscillatory regime, four groups, reference an independent `scipy.quad`
of the exact integrand ($10^{-11}$ relative), `rtol = 10^{-8}`:

| $b$ | $x_r$ range | $\sum\lvert\theta\rvert$ | without `theta_deriv`: rel err / reported abserr | with: rel err / reported abserr | solves (both) |
|---|---|---|---|---|---|
| 0 | $10^3$–$3\times10^3$ | 5.0e3 | 3.3e-12 / 1e-12 | 2.4e-12 / 4e-13 | 12 |
| 0 | $10^4$–$1.2\times10^4$ | 2.1e4 | 3.7e-11 / 2e-12 | 3.6e-11 / 3e-13 | 12 |
| 0 | $10^5$–$1.03\times10^5$ | 1.9e5 | 2.5e-10 / 7e-11 | 2.4e-10 / 3e-13 | 12 |
| 0 | $10^6$–$1.003\times10^6$ | 1.8e6 | 1.7e-11 / 1e-9 | 7.9e-12 / 6e-10 | 12 |
| 0.2 | $10^3$–$3\times10^3$ | 5.7e3 | 8.7e-12 / 5e-12 | 4.4e-12 / 4e-13 | 12 |
| 0.2 | $10^4$–$1.2\times10^4$ | 2.2e4 | 4.5e-10 / 6e-10 | 2.0e-10 / 5e-10 | 12 |
| 0.2 | $10^5$–$1.03\times10^5$ | 1.9e5 | 2.9e-11 / 5e-10 | 2.1e-11 / 4e-13 | 12 |
| 0.2 | $10^6$–$1.003\times10^6$ | 1.8e6 | 9.1e-10 / 9e-10 | 1.5e-9 / 5e-13 | 12 |

Accuracy is the same to within a factor 2 either way (both at $10^{-9}$ or better), cost is
identical (12 solves, 0.01 s), and the *reported* error floor is up to three orders lower with
the derivative supplied (the spectral route's floor inherits $\epsilon\lvert\theta\rvert$). Neutral
on value, a win on the error estimate; passed (`LEVIN_USE_THETA_DERIV = True`). With realistic
fixtures the phase-spline fit error (~$h^4x/384$) dominates both routes equally.

### 10. The §6 cost ratio exceeds 3× in wall-clock — measurement, board issue opened

"$G$ oscillatory, both $T$ smooth", realistic fixtures, `rtol = 1e-8`, sub-interval from $G$'s
hand-over to $T_r$'s, best of three timings; (a) the pre-commit `WKB_quad_integral` (loaded from
`git show 39ed7fc`), (b) the new single-group Levin call. Integrand evaluations are counted as
distinct `Hubble` calls (one per abscissa in both paths):

| $b$ | $\theta_G$ span (cycles) | (a) time / evals / rel err | (b) time / evals / rel err | regions (all CC fallback) | **time ratio** | eval ratio |
|---|---|---|---|---|---|---|
| 0 | 3.52 | 19.2 ms / 1365 / 8.7e-5 | 63.5 ms / 1840 / 1.3e-4 | 21 | **3.31** | 1.35 |
| 0 | 4.15 | 17.0 ms / 1197 / 6.7e-5 | 59.6 ms / 1722 / 1.0e-4 | 20 | **3.51** | 1.44 |
| 0 | 4.47 | 24.9 ms / 1785 / 1.1e-4 | 88.8 ms / 2451 / 1.6e-4 | 28 | **3.56** | 1.37 |
| 0.2 | 4.47 | 19.9 ms / 1575 / 3.6e-5 | 63.6 ms / 1907 / 4.6e-5 | 22 | 3.19 | 1.21 |
| 0.2 | 5.11 | 24.6 ms / 1701 / 2.3e-5 | 69.2 ms / 1907 / 2.9e-5 | 22 | 2.82 | 1.12 |
| 0.2 | 5.42 | 26.0 ms / 1995 / 1.1e-4 | 68.9 ms / 1998 / 1.4e-4 | 23 | 2.65 | 1.00 |

The total-variation gate routes every region to Clenshaw–Curtis, as intended, but the driver's
per-region overhead (20–28 regions, each a nested pair of order-24 Chebyshev rules with the
acceptance bookkeeping) makes the call 2.7–3.6× slower in wall-clock while doing only 1.0–1.4×
the integrand work. On these cheap fixtures the overhead dominates; on production integrands
(GenericEOS `Hubble`, several spline reads per abscissa) the ratio would move towards the
evaluation ratio. Both accuracies are at the realistic-fixture floor. Since the $b=0$ rows are
above the prompt's 3× trigger, board issue `[08-levin-fallback-cost-ratio]` is opened and the
user decides whether prompt 10 reinstates a cycle-count threshold via
`QuadSourcePolicy.Levin_threshold`.

### 11. Test design differences from §7 — STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE

- **Exact ingredients need a seam** (STRUCTURALLY REQUIRED). §7 item 2 asks for $10^{-8}$ against
  `scipy.quad` of the exact integrand; the realistic `TkSourceFunctions` carry $10^{-5}$–$10^{-4}$
  of representation error (log 07), so $10^{-8}$ is only reachable with exact $T$ — hence
  `Tk_functions_builder` (deviation 3) and `ExactTkFunctions`. Achieved: $\le3.5\times10^{-13}$.
- **The realistic $G$ phase is offset by two cycles** (STRUCTURALLY REQUIRED for the fixture).
  Log 07's `BesselPhaseGk` samples $\theta_G=\vartheta(k\eta')-\vartheta(k\eta_{\rm resp})$, which
  vanishes at $z_{\rm resp}$; an integral down to $z_{\rm resp}$ needs the phase grid to reach it,
  and a rounding-level positive value there gets `wrap_theta`'s `div2pi = +1` against $\le0$
  everywhere else, which `phase_spline`'s logarithmic chunking refuses ("initial and final div2pi
  values must have the same sign"). `OffsetBesselPhaseGk` shifts every sample by $-4\pi$:
  sin/cos, `theta_deriv` and every phase difference are unchanged. Log 07 avoided this by keeping
  its grid 5 % above $z_{\rm resp}$. See observation 1 for the production question this raises.
- **Seam check at one abscissa** (IMPLEMENTATION CHOICE). §7 item 4 says "just above … and just
  below"; at $x\approx19$ the integrand changes by $10^{-4}$ over $2\times10^{-6}$ in $\log(1+z)$,
  which would swamp the seam. Both representations are evaluated *at* the hand-over (the spline's
  last node and the LG region's first sample) on the "T-first" shape, where the regime below
  $T_r$'s hand-over is `(F,F,T)` for both $b$. Exact: $10^{-15}$; realistic: 5.6e-6 ($b=0$),
  4.0e-5 ($b=0.2$) of the envelope, asserted at 2e-3 (log 07 item 5's seam number; the prompt's
  "prompt 06's spline accuracy" 4.5e-4 is the between-node number, not the node one).
- **Artificial split at `rtol = 1e-10`** (IMPLEMENTATION CHOICE): the $10^{-10}$ agreement §7
  item 4 asks for is not guaranteed by two Levin runs at `rtol = 1e-8`; asserted at $10^{-10}$
  with `rtol = 1e-10`, achieved 3.1e-13 / 4.2e-14.
- **Old-regime regression uses `TkSourceFunctions`, an exact $G$ and a constant source**
  (IMPLEMENTATION CHOICE). "Both $T$ super-horizon throughout" is realised with $q=r=10^4$ five
  to six e-folds outside the horizon (where `TkSourceFunctions.T` is exactly 1) and $k=10^8$ so
  that $\theta_G$ turns over 4.4 cycles; the pre-commit `numeric_quad_integral +
  WKB_Levin_integral` is run from `git show 39ed7fc:ComputeTargets/QuadSourceIntegral.py`
  loaded as a module (skipped if git is unavailable). Agreement 2.2e-13 (target $10^{-10}$).
  A first attempt with the *exact* `ExactTk` stand-ins disagreed at 4e-6 because their $T$ is
  the analytic $1-x^2/10$, not 1 — the test's inconsistency, not the code's; recorded so nobody
  chases it.
- **"$T_q$ only" and "$G$ and $T_q$" are not separately exercised**; with $q<r$ throughout,
  $T_r$ always hands over first, and README §6's rows are stated per *number* of oscillatory
  factors. `(T,F,T)`, `(F,F,T)`, `(F,T,T)`, `(T,F,F)`, `(T,T,T)` all occur.
- **Oracle 2's short range** for the "q-smooth" shape has two regimes, not three ($G$ has
  already handed over at its top); the assertion is `>= 2`.

### 12. Small robustness edits outside the prompt's list — IMPLEMENTATION CHOICE

`get_z` accepts anything with `.z`; `WKB_phase_spline_chunks` uses `getattr(..., "num_chunks",
None)`; `_Gk_diagnostics` tolerates a `GkPolicy` without `_source_proxy`. All three exist so that
duck-typed stand-ins (the audit's and the tests') can drive the function; production objects
behave identically.

## Verification performed

All from the repository root with `PYTHONPATH=.` and `./venv/bin/python`.

**`python -m unittest discover -s ComputeTargets/tests -t .`: Ran 60 tests in 149.4 s — OK**
(15 new, 45 pre-existing). `python ComputeTargets/tests/sympy_phase_groups.py`: "all residuals
are identically zero". `./venv/bin/black` reports both touched files formatted. The new module
alone runs in ~138 s, of which most is `bessel_phase` construction for the analytic oracle (three
per case, 36 cases).

**Oracle 1, exact ingredients** (`total` vs `analytic_rad`; worst **1.358e-6**, threshold 1e-5).
Wavenumbers in $H_0=1$ units; regimes listed top to bottom:

| case | `total` | `analytic_rad` | rel | regimes | Levin solves |
|---|---|---|---|---|---|
| b=0 together x=980 | +9.2632298681e-13 | +9.2632424502e-13 | 1.358e-06 | -, r, Gr, Gqr | 15 |
| b=0 together x=100 | -1.4546904149e-10 | -1.4546904412e-10 | 1.810e-08 | -, r, Gr, Gqr | 15 |
| b=0 together x=30 | -1.9958705760e-10 | -1.9958701583e-10 | 2.093e-07 | -, r, Gr, Gqr | 9 |
| b=0 T-first x=980 | +7.4690421983e-11 | +7.4690421246e-11 | 9.866e-09 | -, r, qr, Gqr | 15 |
| b=0 T-first x=100 | -8.4352152992e-09 | -8.4352153430e-09 | 5.187e-09 | -, r, qr, Gqr | 13 |
| b=0 T-first x=30 | +2.3437794771e-08 | +2.3437794777e-08 | 2.477e-10 | -, r, qr, Gqr | 7 |
| b=0 q-smooth x=980 | -8.5701377060e-12 | -8.5701378837e-12 | 2.074e-08 | -, G, Gr | 19 |
| b=0 q-smooth x=100 | +1.5578803620e-10 | +1.5578803658e-10 | 2.446e-09 | -, G, Gr | 11 |
| b=0 q-smooth x=30 | +7.4905262167e-10 | +7.4905262167e-10 | 9.546e-12 | -, G, Gr | 9 |
| b=0.2 together x=980 | +4.9847338684e-12 | +4.9847328504e-12 | 2.042e-07 | -, G, Gr, Gqr | 15 |
| b=0.2 together x=100 | -4.2083425529e-11 | -4.2083420823e-11 | 1.118e-07 | -, G, Gr, Gqr | 13 |
| b=0.2 together x=30 | +3.1040426690e-10 | +3.1040420569e-10 | 1.972e-07 | -, G, Gr, Gqr | 9 |
| b=0.2 T-first x=980 | -6.8179016559e-11 | -6.8179009728e-11 | 1.002e-07 | -, r, qr, Gqr | 13 |
| b=0.2 T-first x=100 | +5.6035620689e-09 | +5.6035620782e-09 | 1.656e-09 | -, r, qr, Gqr | 13 |
| b=0.2 T-first x=30 | +2.1035156913e-08 | +2.1035156884e-08 | 1.411e-09 | -, r, qr, Gqr | 7 |
| b=0.2 q-smooth x=980 | -2.5224741948e-12 | -2.5224725332e-12 | 6.587e-07 | -, G, Gr | 19 |
| b=0.2 q-smooth x=100 | +2.4408179425e-11 | +2.4408177596e-11 | 7.494e-08 | -, G, Gr | 11 |
| b=0.2 q-smooth x=30 | +5.6787830863e-10 | +5.6787830861e-10 | 3.597e-11 | -, G, Gr | 9 |

Every Levin call converged; every group solved in one Levin region (3 solves) or one CC region.
Type `numeric` and type `WKB` policies on "T-first" x=100: 5.24e-9 and 5.19e-9. $q=r$: 9.1e-8.

**Oracle 1, realistic fixtures** (attribution against the exact run of the same case, normalised
by `scale`; worst: `total` vs `analytic_rad` **2.379e-3** relative, and by scale: smooth part
**4.553e-4**, Levin part **1.301e-4**, total **4.910e-4**):

| case | rel(total, analytic) | $\lvert$numeric_quad/total$\rvert$ | smooth-part dev | Levin-part dev | total dev | solves | time |
|---|---|---|---|---|---|---|---|
| b=0 together x=980 | 2.379e-03 | 4.8 | 4.553e-04 | 3.591e-05 | 4.910e-04 | 401 | 0.81 s |
| b=0 together x=100 | 2.560e-05 | 1.3 | 1.460e-05 | 4.962e-06 | 1.955e-05 | 105 | 0.29 s |
| b=0 together x=30 | 4.253e-04 | 0.6 | 4.008e-04 | 2.473e-05 | 4.253e-04 | 17 | 0.12 s |
| b=0 T-first x=980 | 2.992e-04 | 1.9 | 1.014e-04 | 2.283e-06 | 1.037e-04 | 139 | 0.31 s |
| b=0 T-first x=100 | 4.776e-05 | 0.9 | 4.153e-05 | 6.229e-06 | 4.776e-05 | 57 | 0.18 s |
| b=0 T-first x=30 | 7.238e-05 | 0.9 | 6.528e-05 | 7.098e-06 | 7.238e-05 | 13 | 0.10 s |
| b=0 q-smooth x=980 | 3.718e-04 | 3.1 | 1.6e-09 | 1.200e-04 | 1.200e-04 | 351 | 0.57 s |
| b=0 q-smooth x=100 | 2.054e-04 | 1.6 | 1.2e-11 | 1.301e-04 | 1.301e-04 | 87 | 0.19 s |
| b=0 q-smooth x=30 | 1.426e-04 | 0.9 | 1.2e-09 | 7.341e-05 | 7.341e-05 | 47 | 0.08 s |
| b=0.2 together x=980 | 5.836e-05 | 1.4 | 6.308e-06 | 3.519e-05 | 4.135e-05 | 435 | 0.92 s |
| b=0.2 together x=100 | 1.304e-04 | 1.5 | 4.056e-05 | 4.717e-05 | 8.765e-05 | 69 | 0.21 s |
| b=0.2 together x=30 | 5.740e-05 | 1.5 | 4.921e-06 | 3.391e-05 | 3.870e-05 | 25 | 0.17 s |
| b=0.2 T-first x=980 | 2.505e-05 | 0.5 | 1.021e-05 | 1.473e-05 | 2.505e-05 | 121 | 0.44 s |
| b=0.2 T-first x=100 | 8.706e-06 | 0.8 | 6.497e-06 | 1.520e-05 | 8.706e-06 | 57 | 0.18 s |
| b=0.2 T-first x=30 | 2.524e-06 | 0.9 | 6.815e-06 | 9.338e-06 | 2.524e-06 | 13 | 0.08 s |
| b=0.2 q-smooth x=980 | 6.670e-06 | 2.4 | 1.6e-09 | 2.503e-06 | 2.775e-06 | 365 | 0.87 s |
| b=0.2 q-smooth x=100 | 4.332e-04 | 5.0 | 6.4e-10 | 7.167e-05 | 7.165e-05 | 67 | 0.16 s |
| b=0.2 q-smooth x=30 | 1.122e-04 | 0.9 | 7.9e-10 | 1.122e-04 | 1.122e-04 | 57 | 0.11 s |

Read: the smooth part is the $f$-spline floor (zero on "q-smooth", where $T_q$ is far
super-horizon and $f$ barely oscillates in the smooth region); the Levin part is the LG
closed-form/phase-respline floor; the largest relative errors of `total` are cancellation.

**Oracle 2** (exact ingredients, short range $x_r\in[4,60]$ straddling every hand-over, vs
`scipy.quad` of the exact integrand at `epsrel = 1e-12`, `rtol = 1e-10`; threshold 1e-8):
4.1e-15, 1.5e-15, 4.1e-14 ($b=0$; together, T-first, q-smooth), 1.1e-13, 2.6e-15, 3.5e-13
($b=0.2$). The integrator and partition contribute nothing measurable.

**Regime coverage** (`build_partition` over all 18 cases): rows reached `G only`, `one T only`,
`G and one T`, `Tq and Tr, G smooth`, `all three` (and `none`). Sub-intervals are contiguous,
descending, start at `(F,F,F)`/`quad`, and no factor ever returns from oscillatory to smooth.

**Seams**: artificial split (exact, `(T,T,T)`, $x_r\in[40,980]$): 3.1e-13 ($b=0$), 4.2e-14
($b=0.2$). Hand-over seam at $T_r$'s hand-over on "T-first": exact 1.3e-15 / 8.2e-15; realistic
**5.6e-6 / 4.0e-5** of the envelope.

**Hand-over clamp**: gap 2.301e-2 in $\log(1+z)$ (1.00 grid steps) recorded in the partition
metadata for exactly one sub-interval; `total` with vs without the gap **5.140e-3**; with gap vs
`analytic_rad` 5.114e-3. A five-step gap raises "the Liouville-Green representation of Tr is not
evaluable over the whole sub-interval … shortfall 1.150e-01 … at most 3.455e-02 (1.5 mean
source-grid steps) can be bridged"; an LG region ending above $z_{\rm resp}$ raises "lies below
the lowest Liouville-Green sample".

**Old-regime regression**: $\theta_G$ span 4.4 cycles; old total −1.426544368746e-16
(numeric +6.084616e-17, Levin −2.035006e-16), new −1.426544368746e-16 (same parts); **rel
2.19e-13** (target 1e-10); numeric parts agree to 1e-12. Independently, both Levin values agree
with `quad` of the same integrand to 1e-13.

**`compute()` payload**: a payload lacking the four `Tk` keys raises `RuntimeError` whose message
names `Tq_numeric`, `Tq_WKB`, `Tr_numeric`, `Tr_WKB`; not a `KeyError`.

**QI_03 re-run** (adapted copy of `docs/spec-code-audit/scripts/QI_03_measure.py`, section 1
only, plus the same call with `source_f` overriding the spline): `numeric_quad_integral =
5.815078635181e-09`, rel vs the R28 measure **4.267e-16** on both paths, 9.93e-1 vs the
no-Jacobian variant. The measure is unchanged. The script as committed was not modified
(README §5 item 5); its sections 2–4 call the deleted functions (observation 2).

**`theta_deriv`** and **cost**: the tables in deviations 9 and 10.

**Not verified here, and needing a pipeline run (prompt 12):** that real `GkSourcePolicyData`,
`TkNumericIntegration`, `TkWKBIntegration` and `QuadSource` rows satisfy the protocols (the
attribute names match by inspection; nothing here touched a datastore); the actual size of the
hand-over gap on production rows and hence the clamp error; whether the real $\theta_G$ samples
near $z_{\rm resp}$ trip `phase_spline`'s sign rule (observation 1); the wall-clock cost on
production integrands.

## Observations not acted on

1. **The Green's-function phase at $z_{\rm resp}$.** A phase grid that reaches $z_{\rm resp}$
   with $\theta_G\to0$ there gets mixed `div2pi` signs from a rounding-level positive value and
   `phase_spline(chunk_logstep=125)` refuses it (deviation 11). `GkSourcePolicyData._create_functions`
   builds exactly such a spline over `WKB_data` down to `z_sample.min`; whether the stored
   `theta_div_2pi` of a real `GkSource` has this property depends on the constant
   `GkWKBIntegration` fixes at matching and was not checked here (no rows). Prompt 12 should
   look, since a failure there would be a construction-time `RuntimeError` in production, not a
   wrong number.
2. **`docs/spec-code-audit/scripts/QI_03_measure.py` sections 2–4 no longer run** (they call the
   deleted `WKB_quad_integral`/`WKB_Levin_integral`). Section 1 runs and agrees (4.27e-16). The
   script is the record of the pre-fix state and was left alone.
3. **Levin driver overhead in the few-cycle regime** (deviation 10): the 2.7–3.6× wall-clock
   ratio is 20–28 CC-fallback regions of overhead on a ~1.3× evaluation count. If the user wants
   the fallback cheaper rather than gated, the lever is in `AdaptiveLevin/` (a coarser first
   bisection, or a plain adaptive quadrature when the *whole* call's phase span is below $6\pi$),
   out of scope here.
4. **`ComputeTargets/phase_groups.py`'s docstrings cite `WKB_Levin_integral` by pre-commit line
   number** (`:103`, `:124`, `:197`); they describe what the `G`-only group reproduces and are
   now historical. Left, since that file is prompt 07's.
5. **Cancellation between `numeric_quad` and `WKB_Levin`** reaches 5× on these fixtures, so the
   error of `total` is that factor times the parts' floors. Prompt 09's error bound (B8) should
   sum the parts' absolute errors, not scale a relative one; `metadata["numeric_quad"]["abserr"]`
   and `metadata["WKB_Levin"]["abserr"]` are there for it.
6. **Realistic $G$ fixtures at $x_{\rm resp}=30$ have 4–17 phase samples** (the LG region of $G$
   is a fraction of an e-fold there); `phase_spline` warns, harmlessly. The production analogue
   (a `mixed` policy with its crossover just above $z_{\rm resp}$) would warn the same way.
7. **Test wall-clock** is dominated by `bessel_phase` for the analytic oracle (three
   constructions per case); caching by $(\nu, x_{\max})$ would halve it. Not done; 140 s is inside
   the prompt's budget.
8. **The `Levin_z` machinery in `GkSourcePolicyData._classify_Levin` and the `Levin_z` column**
   are now write-only from this module's point of view (read into `metadata["partition"]
   ["Levin_z_unused"]`). `extract_common.py` and `extract_GkSource_data.py` still plot it.
   Prompt 10 decides its fate together with `QuadSourcePolicy.Levin_threshold`.

## State handed to the next prompt

Prompts 09 and 10 program against the following.

```python
from ComputeTargets.QuadSourceIntegral import (
    evaluate_QuadSource_integral,      # plain function; the Ray task calls it
    compute_QuadSource_integral,       # @ray.remote wrapper
    build_partition, phase_group_Levin_integral, numeric_quad_integral,
    LEVIN_USE_THETA_DERIV, HANDOVER_CLAMP_MAX_GRID_STEPS,
)

# QuadSourceIntegral.compute(payload) now requires
payload = {
    "source": QuadSource, "GkPolicy": GkSourcePolicyData, "b": float,
    "Bessel_0pt5": BesselPhaseProxy, "Bessel_2pt5": BesselPhaseProxy,
    "Tq_numeric": TkNumericIntegration, "Tq_WKB": TkWKBIntegration,   # for q
    "Tr_numeric": TkNumericIntegration, "Tr_WKB": TkWKBIntegration,   # for r
}
# missing keys -> RuntimeError naming them (QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS);
# each Tk's .k.store_id must equal the matching q/r exit time's .k.store_id.
```

1. **Prompt 10 supplies the four payload keys** `Tq_numeric`, `Tq_WKB`, `Tr_numeric`, `Tr_WKB`
   from `main.py`'s QuadSourceIntegral stage (the `TkNumericIntegration`/`TkWKBIntegration`
   objects for $q$ and $r$, as the QuadSource and TkWKB stages already look them up). Until then
   `compute()` raises; board issue `[08-pipeline-non-runnable-until-10]`.
2. **`WKB_quad` is identically `0.0` and `WKB_quad_data` is `None`.** Prompt 09 decides whether
   to drop the columns; nothing reads them.
3. **`numeric_quad_data`/`WKB_Levin_data` are aggregates**: `IntegrationData` summed over the
   smooth sub-intervals (evaluation-weighted mean RHS time, min/max of the extremes);
   `LevinData` summed over every Levin call (`num_regions`, `evaluations`, `num_simple_regions`,
   `num_SVD_errors`, `num_order_changes`), `min(chebyshev_min_order)`, `max(max_depth)`,
   `sum(elapsed)`. Either is `None` when no sub-interval of that kind ran.
4. **Error estimates for B8** live in `metadata["numeric_quad"]["abserr"]` (scipy's estimates,
   scaled by $1+z_{\rm resp}$, summed) and `metadata["WKB_Levin"]["abserr"]` (driver `abserr`
   summed linearly over groups and sub-intervals, scaled likewise); `metadata["WKB_Levin"]` also
   carries `converged`, `phase_limited`, `theta_deriv_supplied` and the per-sub-interval,
   per-group breakdown, and is always a dict. Sum the two absolute errors for a bound on
   `total`; do not scale a relative error — the parts cancel by up to 5× (observation 5).
5. **`metadata["partition"]`** records breakpoints (which factor, where, whether inside the
   range), the regimes and methods of every sub-interval, the clamp gaps in $\log(1+z)$, the
   unused `Levin_z`, and `crossover_z_q/_r` as recomputed from the `Tk` objects.
6. **`atol` is distributed by $\log(1+z')$-length share** across sub-intervals and then equally
   across a sub-interval's groups; `rtol` is passed unchanged. The stored `atol_serial`/
   `rtol_serial` therefore still describe what the caller asked for the whole integral.
7. **The §6 cost ratio is 2.65–3.56× in wall-clock (1.00–1.44× in integrand evaluations)**, above
   the 3× trigger at $b=0$; board issue `[08-levin-fallback-cost-ratio]`. The user decides whether
   prompt 10 reinstates a cycle-count threshold via `QuadSourcePolicy.Levin_threshold`
   (`MetadataConcepts/QuadSourcePolicy.py`, threaded through `main.py`, currently read by
   nothing). If reinstated, the natural place is `build_partition`: a `(T,F,F)` sub-interval
   whose $\theta_G$ span is below the threshold would take `method = "quad"` with the integrand
   from `phase_groups.smooth_source` on the `TkSourceFunctions` numeric accessors — **not** the
   old `WKB_quad_integral`, which read $f$ from the QuadSource spline below its range.
8. **The hand-over clamp costs ~5e-3 of `total` per one-grid-step gap** (board issue
   `[08-handover-clamp-error]`), and every production integral has such a gap for each $T$
   (`main.py:695-697`). This is now the dominant error term; prompt 12 must expect it and should
   report the recorded `clamp_gaps_log1pz`. Remedies are upstream or the extrapolation option of
   deviation 2; neither belongs to prompts 09 or 10.
9. **Accuracy floors for prompt 12**: exact ingredients reproduce `analytic_rad` to
   $\le1.4\times10^{-6}$ (that is `analytic_rad`'s own floor) and `scipy.quad` to
   $\le3.5\times10^{-13}$; realistic fixtures give smooth-part $\le4.6\times10^{-4}$ and
   Levin-part $\le1.3\times10^{-4}$ of `max(|numeric_quad|, |WKB_Levin|, |analytic_rad|)`, and
   up to $2.4\times10^{-3}$ relative in `total` where the parts cancel — before the clamp error
   of item 8. None of these is a physics defect.
10. **`LEVIN_USE_THETA_DERIV = True`**: the composed $d\Psi/d\log(1+z')$ is passed. Neutral on
    value, identical cost, up to $10^3\times$ lower reported error floor (deviation 9).
11. **Region guards are in $\log(1+z)$** (`MIN_SUBINTERVAL_LOG_WIDTH`); B11's original sites are
    gone. Prompt 09 should close B11 by confirming this rather than by editing.
12. **Reusable test machinery** in `ComputeTargets/tests/test_quadsource_integral.py`: `Case(b,
    shape, x_resp, exact, G_type, z_source_max)` with `.run()`, `.partition()`,
    `.integrand_exact()`, `.bessel_phase_data()`; `Shape`, `SHAPES`, `X_RESP_VALUES`;
    `FakeQuadSource`, `FakeGkPolicy`, `OffsetBesselPhaseGk`, `ExactTkFunctions`,
    `exact_Tk_inputs(fixture)`, `load_pre_commit_module()` (the file at `39ed7fc`),
    `regimes_of(result)`, `regime_row(regime)`.
13. **No schema, no `Datastore`, no `main.py`, no `QuadSourcePolicy` change.**
