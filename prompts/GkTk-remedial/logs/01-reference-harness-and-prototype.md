# Log 01 — Reference harness, primitive prototype and throughput benchmark

**Prompt:** prompts/GkTk-remedial/01-reference-harness-and-prototype.md
**Commit:** *(SHA intentionally not embedded — writing it and amending changes it again; the
repository's campaign convention since `source-remediation` log 01 deviation 4)* — "Add WKB phase
references and a measured primitive prototype", the single commit that adds this log
**Model:** Claude Opus 5
**Date:** 2026-09-10
**Result:** COMPLETE WITH DEVIATIONS

> **First line, per prompt §4.** The interval accessor costs **102.2 µs per call on
> `QCDModel` when *both* endpoints are off-grid**, above README §4.3's 50 µs stop threshold
> (47.7 µs with one endpoint off-grid). The production case is not that case: the background
> model is built on the source grid (`main.py:476`, `RECONCILIATION.md` §1 item 11), so every
> WKB sample is a node and both endpoints are on-grid, where the call costs **4.39 µs and zero
> Hubble evaluations**. Only the per-object anchor $z_{\rm init}$ is off-grid — one endpoint,
> once per object.
>
> **Second, per prompt §4.** On `QCDModel`, Gauss orders 4 and 8 disagree by
> $2.5\times10^{-7}$ relative on $\tau$ at the production nodes, seven orders above the
> $10^{-13}$ "say so prominently" threshold. The cause is **not** the $T(z)$ spline knots the
> review anticipated: it is the two *branch boundaries* of `QCD_EOS.G(T)` at $T=10^{-5}$ GeV
> ($z\approx4.19\times10^7$) and $T=0.12$ GeV ($z\approx8.58\times10^{11}$), where the
> Saikawa–Shirai fit is switched for an asymptotic constant. Fixed-order Gauss converges only
> as $N^{-2}$ there: at order 4 that one interval carries $1.6\times10^{-8}$ Mpc, i.e.
> **4.8 rad of phase at $k=3\times10^8$**; order 20 still leaves 0.26 rad. Prompt 02 must
> decide, and raising the order is not among its options.

---

## What shipped

No production code was changed. `git diff HEAD~1 --stat` touches only new files plus the campaign
board, its log and `docs/OPEN_ISSUES.md`.

### `ComputeTargets/tests/wkb_reference.py` (new, 456 lines)

The reference harness. Imports nothing from `WKB_phase_function.py`, `phase_spline.py`,
`cumulative_table.py`, `phase_residual.py` or `primitive_phase.py`, and does not import `mpmath`
(a test asserts the latter).

Production geometry, read off `main.py` on `9ff59d5`:
`PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z = 100`, `PRODUCTION_RESPONSE_SPARSENESS = 12`,
`PRODUCTION_Z_END = 0.1`, `PRODUCTION_SMALLEST_K_INV_MPC = 1.0e5`,
`PRODUCTION_LARGEST_K_INV_MPC = 3.0e8`, `PRODUCTION_SUPERHORIZON_EFOLDS = 5`,
`REFERENCE_K_VALUES = (1.0e5, 1.0e7, 3.0e8)`,
`MODEL_KEYS = ("RadiationModel", "LambdaCDMModel", "QCDModel")`, `REFERENCE_DATA_PATH`.

Grid helpers:

```python
horizon_exit_z(cosmology, k_inv_Mpc: float, efolds_subh: float = 0.0) -> float
production_source_z_values(z_init, z_end=0.1, samples_per_log10z=100) -> np.ndarray
to_redshift_array(z_values) -> redshift_array
production_source_grid(z_init, z_end=0.1, samples_per_log10z=100) -> redshift_array
production_response_grid(source_grid, sparseness=12) -> redshift_array
```

`horizon_exit_z` mirrors `CosmologyConcepts.wavenumber._solve_horizon_exit` (positive
`efolds_subh` = inside the horizon) and reproduces the review's geometry table: $z_{\rm exit}$ =
1.390e14, $z_{e3}$ = 6.923e12 for $k=3\times10^8$ (review §1: 1.4e14, 6.92e12); $z_{e3}$ =
2.3076e9 for $k=10^5$ (review §4: 2.31e9).

Stand-in models:

* `RadiationModel(H0=1.0)` — `.functions` (a `ModelFunctions` with `tau` wired), plus closed forms
  `Hubble(z)`, `tau(z) = 1/(H0 s)`, `tau_delta(z_a, z_b)`, `cs_tau(z)`, `cs_tau_delta(z_a, z_b)`,
  `friction_F_delta(z, z_ref)`, `theta_G(k, z, z_init)`, `rho_G(...) == 0`,
  `rho_T(k, z, z_init)`, `x_T(k, z)`.
* `LambdaCDMModel(cosmology=None)` — `.cosmology`, `.functions`; the analytic-derivative branch of
  `BackgroundModel._create_functions`, as `realbg.py` builds it. `functions.tau is None`.
* `QCDModel(z_sample, cosmology=None, atol=1e-10, rtol=1e-8)` — `.cosmology`, `.functions`,
  `.background_payload`, `.z_sample`; built by calling the **undecorated** `compute_background` and
  splining its derivative samples through `_model_functions_from_background`, which reproduces
  `BackgroundModel._create_functions`. `functions.tau is None`.

Error definitions (README §6, unchanged by every later prompt):

```python
phase_error(theta, theta_ref) -> float            # |theta - theta_ref|, unwrapped, radians
difference_error(delta, delta_ref) -> float       # |delta - delta_ref| / |delta_ref|
envelope_relative_error(value, value_ref, envelope) -> float
load_references(path: Optional[Path] = None) -> dict
```

### `ComputeTargets/tests/wkb_reference_data.json` (new, 36 kB)

The cached references. Schema described in §"State handed to the next prompt" below and, in the
file itself, in its `"schema"` block.

### `ComputeTargets/tests/test_wkb_reference.py` (new, 10 tests, 0.23 s)

`TestReferenceJSON.test_json_is_complete`, `.test_radiation_closed_forms`,
`.test_baselines_block`, `.test_reference_module_does_not_import_mpmath`;
`TestErrorDefinitions` × 3; `TestStandInModels.test_production_grid_shape`,
`.test_epsilon_in_the_radiation_era`, `.test_radiation_model_is_self_consistent`.

### `docs/gktk-remedial/` (new)

* `reference_lib.py` — the two reference methods and the double-precision integrands.
  `C_G_double(functions, z)` / `C_T_double(functions, z)` re-implement the `B + C` terms of
  `Gk_omegaEff_sq` / `Tk_omegaEff_sq` (never $\omega^2-\omega_0^2$; `RECONCILIATION.md` §2 item 3).
  `MpLambdaCDM`, `MpRadiation`, `mp_increment`, `gauss_bisect`, `quad_sum_over_intervals`,
  `gauss_sum_over_intervals`, `select_checkpoints`, `short_baseline_locations`,
  `fractional_point`.
* `generate_references.py` — writes the JSON (152 s).
* `prototype_primitive.py` — `CumulativeTablePrototype(z_nodes, integrand_u, order,
  partial_order=8, double_double=True)` with `value(z)` and `delta(z_a, z_b)`, plus the
  measurement driver (~90 s).
* `baseline_k1e5.py` — the two production baselines (~10 s); appends a `"baselines"` block to the
  JSON.
* `PROTOTYPE-MEASUREMENTS.md` — the results document, with every table quoted below.

---

## Deviations from the prompt

### 1. The production grid is log-spaced in $z$, not in $1+z$ — STRUCTURALLY REQUIRED

The prompt (§2.1) describes the source grid as "100 per decade of $1+z$ … mirroring
`wavenumber_exit_time.populate_z_sample`". Those two are not the same thing:
`populate_z_sample` (`CosmologyConcepts/wavenumber.py:288-292`) computes
`num = round(spld*(log10(z_init) - log10(z_end)) + 0.5)` and returns
`logspace(log10(z_init), log10(z_end), num)` — log-spaced **in $z$**.

`production_source_z_values` mirrors the code, because that is the grid the tables of prompts 03
and 04 will be built on. The difference is invisible for $z\gg1$ and large at the bottom: near
$z=0.1$ the spacing in $u=\log(1+z)$ is 0.0021, ten times finer than the 0.023 it is at high $z$.
Anything a later prompt asserts about "the grid interval" must take the location into account.

### 2. The residual references are anchored at each $k$'s own 3-e-fold sub-horizon point, not at the top node — STRUCTURALLY REQUIRED

The prompt (§2.2) asks for $\rho_G(z_j; z_{\rm top})$ and $\rho_T(z_j; z_{\rm top})$. That is not
evaluable. The grid top is the 5-e-fold **super**-horizon point of $k=3\times10^8$, so for every
$k$ in the list the mode is far outside the horizon there and $\omega_T^2 = c_s^2(k/H)^2 + C_T$ is
negative ($C_T=-2/s^2$ in radiation, and $x_T^2 < 2$); `mp.quad` returns a complex number.

Each $k$'s residual is therefore anchored at $z_{e3}(k)$, the 3-e-folds-sub-horizon root of
$k(1+z)/H = e^3$, which is where the production WKB region starts and where review §4 and §12.2
tabulate $\rho$. Only checkpoints strictly below the anchor carry an entry. The anchor is
deliberately **off** the node grid (it is a `root_scalar` root), which is the case
`RECONCILIATION.md` §2 item 5 says prompt 03's tests must exercise; to keep the JSON
self-sufficient the block `primitives_at_rho_anchor[k]` records $\tau$, $c_s\tau$ and $F$ at the
anchor as well.

### 3. The sign of the exact-radiation $\rho_T$ — STRUCTURALLY REQUIRED

The prompt (§2.1) gives the `RadiationModel` closed form as $\rho_T = 1/x_i - 1/x$. In the
convention this campaign actually uses — README §2 (a) and (c),
$\theta = -k\,\tau.\mathrm{delta}(z_i, z) - \Delta\rho$, which is what review §4's and §12.3's own
reference scripts compute — the residual is $\rho_T = 1/x - 1/x_i$, the negative of the quoted
expression. Review §12.4 quotes it in the opposite convention
($\rho_T = \theta - (x_i - x)$). **This does not touch README §2 (c)'s `delta` convention; it
applies it.** The JSON and `RadiationModel.rho_T` use the campaign convention, and the JSON's
`"schema"` block says so explicitly.

`RadiationModel.rho_T` is *exact*, not the $1/x-1/x_i$ asymptote: the antiderivative of
$C_T/(\omega_T + kc_s/H)$ in exact radiation is $g(s) = -2s/(\sqrt{a^2-2s^2}+a) +
\sqrt2\arcsin(\sqrt2 s/a)$ with $a = kc_s/H_0$, written in that cancellation-free form. Without
this the prompt's test 1 ("closed forms agree to $10^{-15}$") could not pass: the asymptote is
only correct to $O(1/x_i^3)$, i.e. $3\times10^{-4}$ relative at $x_i=24$.

### 4. `epsabs=0, epsrel=1e-14` is rejected by QUADPACK — STRUCTURALLY REQUIRED

The prompt (§2.2) specifies SciPy `quad` with `epsabs=0, epsrel=1e-14`. SciPy 1.15.2 raises
`ValueError: If 'epsabs'<=0, 'epsrel' must be greater than both 5e-29 and 50*(machine epsilon)`,
and $50\varepsilon = 1.11\times10^{-14}$. The primary QCD reference uses `epsrel=1.5e-14`, the
tightest pure-relative tolerance available; the cross-check at `epsrel=1e-12` is as specified.

### 5. The Gauss–Legendre cross-check bisects to at most 8 levels — IMPLEMENTATION CHOICE

The prompt asks for "composite Gauss–Legendre order 40 with bisection until two successive
refinements agree to $10^{-14}$". On `QCD_Cosmology` that criterion is never met on the two
branch-boundary intervals of §3 below, and an unbounded bisection spends $4096\times40$ evaluations
there. The cross-check is capped at 8 levels (256 panels) and the worst per-interval relative
change is recorded alongside the result, so a reader can see which intervals did not converge.
Alternatives considered: capping at 4 levels (tried; the totals then agreed only to
$2\times10^{-10}$, weakening the cross-check) and dropping the Gauss cross-check for a third
`quad` tolerance (rejected: it would not be an independent method). At 8 levels the totals agree
with `quad` to $\le6.3\times10^{-15}$ for the primitives.

### 6. `RadiationModel`'s $\epsilon$ smoke test uses different tolerances for the two models — IMPLEMENTATION CHOICE

The prompt's test 4 asks that `epsilon(1e10)` be "within $10^{-6}$ of 2" for both real-background
stand-ins. `LambdaCDMModel` gives $1.9999998298$ ✓. `QCD_Cosmology` gives $1.9979580$ — and that
is physically right, not a wiring error: $z=10^{10}$ is $T\approx2.4$ MeV, in the middle of
$e^+e^-$ annihilation, where $g_*(T)$ genuinely varies (at $z=10^9$, $\epsilon = 1.9517$; review §6
quotes departures up to 0.150 at the QCD transition). The QCD assertion is
`QCD_EPSILON_TOLERANCE = 6e-2`, with the measured values in a comment. A tighter bound would be
asserting that the QCD equation of state is radiation.

### 7. `docs/gktk-remedial/reference_lib.py` is a fourth script, not one of the three named — IMPLEMENTATION CHOICE

The prompt allows "new files under `docs/gktk-remedial/` (scripts and one results document)". The
reference quadratures are needed by both `generate_references.py` and `prototype_primitive.py`
(which builds its own off-grid references), so they live in a sibling module picked up with the
`sys.path.insert` idiom of `docs/gk-wkb-review-fable-2026-09-09/common.py`. It is script support,
not importable machinery: nothing in `ComputeTargets/` or elsewhere imports it.

### 8. The prototype table is anchored at the top node — IMPLEMENTATION CHOICE

`value(z)` returns $\tau(z) - \tau(z_{\rm top})$ rather than an arbitrary primitive, so that it can
be compared with the JSON directly and so that `delta(z_a, z_b) = value(z_b) - value(z_a)` in exact
arithmetic. Prompt 03 is free to anchor elsewhere; the sign convention of `delta` is what matters
and it is README §2 (c)'s.

### 9. The commit SHA is not embedded in the log or the board — IMPLEMENTATION CHOICE

README §5.1's template has a `<sha>` field. Filling it requires `git commit --amend`, which
changes the SHA the log just recorded; there is no fixed point. The repository's other campaigns
settled this the same way (`prompts/source-remediation/logs/01-genericeos-sound-speed.md`
deviation 4, and its board rows 01 and 03): record the commit *subject*, which is unique and
stable, and let `git log --grep` supply the SHA. The alternative — a follow-up commit carrying
only the SHA — would break the one-commit-per-prompt rollback boundary.

---

## Verification performed

Everything below was **run**, not reasoned about. Full tables are in
`docs/gktk-remedial/PROTOTYPE-MEASUREMENTS.md`.

### Tests

* `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_wkb_reference -v` —
  **10 tests, OK, 0.23 s** (acceptance: <10 s). No `mpmath` import (asserted by
  `test_reference_module_does_not_import_mpmath`, which scans `wkb_reference.py` for
  `import mpmath` / `from mpmath`).
* `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` —
  **107 tests, OK, 258 s**. No production code changed.
* `./venv/bin/python -m black` clean on every new file.

Measured closed-form agreements behind test 1 (`RadiationModel`, against the 40-digit mpmath
references): $\tau$ **1.76e-16**, $c_s\tau$ **1.82e-16**, $F$ **2.34e-16**, $\rho_G$ **exactly 0**
at every checkpoint and every $k$, $\rho_T$ **8.34e-16** (worst at $k=3\times10^8$,
$z=9.906\times10^6$); short baselines 1.40e-16 (full) and 0.0 (37 % fraction). Assertion
thresholds are 1e-15 and, for $\rho_T$, 1e-14.

### Reference floors

* `RadiationModel`, `LambdaCDMModel`: mpmath at `mp.dps = 40` against closed-form backgrounds;
  the reference's own floor is ~1e-38 relative and the JSON's doubles are the 1-ulp limit.
* `QCDModel`: `quad` at `epsrel=1.5e-14` per production interval, `fsum`-summed. Agreement with
  `quad` at `epsrel=1e-12`: **≤ 8.8e-15** relative over all nine quantities (0.0 for $\tau$).
  Agreement with composite Gauss–Legendre 40 with bisection: **≤ 6.3e-15** for the three
  primitives, ≤ 2.1e-10 for $\rho_G$ (whose absolute size is 5.4e-4 rad, so 1.1e-13 rad). 29–68 of
  1,731 intervals raise a SciPy `IntegrationWarning` — see §3 of the measurements document.

Cross-checks against the review's own numbers, all reproduced:
$\tau(0.1)-\tau(z_{\rm top}) = 13728.0095$ Mpc on LambdaCDM, so $k\tau = 1.37280\times10^9$ rad at
$k=10^5$ (review §4: $\theta_{\rm ref}(0.1) = -1.3728\times10^9$);
$k\,c_s\tau = 6.1747\times10^7$ (review §12.2: 6.175e7); $\rho_G = -2.588\times10^{-7}$ rad
(review §6: $-2.5\times10^{-7}$); $\rho_T = -0.08634$ (review §12.2: $-0.0863$); QCD
$\rho_G(k=10^7) = +3.003\times10^{-5}$ (review §6: 3.0e-5), $\rho_T(k=3\times10^8) = -0.09310$
(review §12.2: $-0.0933$); $F$ from $z_{e3}(10^5)$ to $0.1$ = 38.9 (review §12.2: 38.9).

### Prototype — build and node accuracy

| model | order | Hubble evals | build | max rel err at nodes |
|---|---|---|---|---|
| LambdaCDM | 4 | 6,924 | 0.037 s | **1.81e-15** (at $z=1.005\times10^7$) |
| LambdaCDM | 8 | 13,848 | 0.041 s | 1.96e-15 |
| LambdaCDM | 12 | 20,772 | 0.047 s | 1.96e-15 |
| QCD | 4 | 6,924 | 0.084 s | **3.45e-07** (at $z=1.005\times10^7$) |
| QCD | 8 | 13,848 | 0.129 s | 9.21e-08 |
| QCD | 12 | 20,772 | 0.222 s | 4.46e-08 |

**Acceptance (LambdaCDM order 4 $\le2\times10^{-14}$): PASS at 1.81e-15.** Review §7 measured
5.7e-15 on a shorter grid. The QCD figure is reported without a threshold, per the prompt; orders
4 and 8 disagree by 2.5e-7 ≫ 1e-13, which the prompt asks be said prominently — see the Result
banner and §3 of the measurements document.

`QCDModel` construction (undecorated `compute_background` on 1,732 nodes + derivative splines):
**0.19–0.43 s** over five runs.

### Prototype — off-grid, 25 random points

| method | LambdaCDM | as rad at $k=3\times10^8$ | QCD |
|---|---|---|---|
| nearest node + Gauss 8 (order 4 table) | **5.84e-16** | 2.5e-3 | 2.16e-07 |
| cubic spline of the nodes | 7.33e-10 | 3.1e3 | 7.56e-08 |
| quintic spline of the nodes | 9.91e-15 | 4.2e-2 | 7.50e-08 |

Reproduces review §7's 2.1e-16 / 1.4e-9 / 1.8e-14. On QCD every method is dominated by the same
accumulated table error, so the row says nothing about the accessor.

### Prototype — short baselines, double-double against single-double

**Acceptance (dd over one interval $\le10^{-13}$ relative): PASS, worst 8.76e-15** (LambdaCDM,
$z=100.19$). The 37 %-fraction rows reach 3.37e-14, also inside 1e-13.

The single-double control shows the ~1e-12 Mpc absolute floor **only where $\tau$ is large**: at
$z\approx1$ ($\tau=1.08\times10^4$ Mpc) it is 1.05e-14 relative on $\Delta\tau=56.98$ Mpc, i.e.
$6.0\times10^{-13}$ Mpc, $1.8\times10^{-4}$ rad at $k=3\times10^8$ (QCD: 1.52e-14 → 2.6e-4 rad).
At $z\approx10^6$, where $\tau=0.046$ Mpc, one ulp is $7\times10^{-18}$ Mpc and the control is
indistinguishable from the design — the review's $9\times10^{-4}$ rad is a statement about the
*absolute* $\tau$ at low $z$, not about grid intervals everywhere.

The claim of review §13.3 in its own terms — a pointwise accessor cannot carry a short baseline
*however close the endpoints are* — is the third control, `value(z_b) - value(z_a)`:

| model | width / interval | $\Delta\tau$ [Mpc] | `delta` dd [rad] | pointwise difference [rad] |
|---|---|---|---|---|
| LambdaCDM | 1e-1 | 5.6926 | 2.5e-5 | 1.2e-4 |
| LambdaCDM | 1e-3 | 5.6920e-2 | 4.2e-6 | 2.0e-4 |
| QCD | 1e-1 | 5.6926 | 2.7e-7 | 1.5e-4 |
| QCD | 1e-3 | 5.6920e-2 | 4.2e-9 | 6.7e-5 |
| QCD | 1e-5 | 5.6920e-4 | **0.0** | 4.9e-5 |

### Prototype — throughput ($10^5$ calls, best of 3)

| call | LambdaCDM | QCD | integrand calls per call |
|---|---|---|---|
| `delta`, both on-grid | 3.03 µs | **4.39 µs** | **0** |
| `delta`, one off-grid | 8.20 µs | 47.7 µs | 8 |
| `delta`, both off-grid | 13.1 µs | **102.2 µs** | 16 |
| `value`, on-grid | 1.47 µs | 1.44 µs | 0 |
| `value`, off-grid | 7.15 µs | 46.7 µs | 8 |
| cubic spline lookup | 3.29 µs | 2.38 µs | — |

On-grid calls are short-circuited — **zero** integrand evaluations, confirmed by the counter. A
Hubble evaluation costs 0.23 µs on LambdaCDM and 8.7 µs on QCD. Against a spline lookup the
off-grid accessor is 4× / 43×, at the optimistic end of review §13.3's "perhaps 50–100×".

### Baselines (prompt §2.5, acceptance 5 %)

| quantity | this tree | review (`f06f587`) | agreement |
|---|---|---|---|
| $\theta_G$ error at $z=0.1$, $k=10^5$ | **13.90 rad** | 13.9 rad | **0.0 %** |
| $G_k$ stage 1 / resets / stage 2 / time | 46,984 / 676 / 1,298 / 1.72 s | 46,702 / 676 / 1,271 / 1.3 s | — |
| $\theta_T$ error at $z=0.1$, $k=10^5$ | **2.012 rad** | 2.01 rad | **0.1 %** |
| $T_k$ stage 1 / resets / stage 2 / friction / time | 35,653 / 508 / 1,061 / 1,814 / 1.12 s | 35,626 / 508 / 1,061 / 1,811 / 1.1 s | — |
| $T_k$ friction $\delta F$ at $z=0.1$ | $-2.26\times10^{-7}$ | $-2.3\times10^{-7}$ | 2 % |

**PASS on both.** $k=3\times10^8$ was not run, per the prompt.

---

## Observations not acted on

1. **A baseline narrower than ~$10^{-9}$ of a decade is not resolvable in $u=\log(1+z)$.** The
   LambdaCDM $10^{-5}$-of-an-interval row *rises* to 1.2e-5 rad, worse than the $10^{-3}$ row,
   because ${\rm ulp}(u)/\Delta u = 1.1\times10^{-16}/2.3\times10^{-7} = 4.8\times10^{-10}$ of the
   width: the endpoint's own representation, not the accessor, sets the error there. Not
   actionable — it is a property of doubles, the other face of `CLAUDE.md`'s redshift-arithmetic
   rule — but a trap for a prompt-03 test that tries to demonstrate the design on an arbitrarily
   short baseline.
2. **The production grid is denser in $u$ at low $z$** (deviation 1). At $z=0.1$ the interval is
   0.0021 in $u$, at $z=10^{10}$ it is 0.023. Nothing depends on it here.
3. **`quad` at `epsrel=1.5e-14` warns on 29–68 of QCD's 1,731 production intervals**
   ("Extremely bad integrand behavior"), the same intervals §3 of the measurements document
   identifies. The totals are unaffected (they agree with two other methods to 1e-14), but a later
   prompt that runs `quad` on QCD should expect the warnings.
4. **The `LambdaCDM.Hubble` rounding floor** is issue `[01-lambdacdm-hubble-rounding-floor]` below.
5. Two latent defects in the `BackgroundModelValue` factory's `build()` path
   (`RECONCILIATION.md` §2 item 11) were **not** looked at; prompt 03 owns them.

### Issues opened on the board (§3)

* `[01-qcd-eos-branch-boundaries]`
* `[01-offgrid-accessor-cost-on-qcd]`
* `[01-lambdacdm-hubble-rounding-floor]`

---

## State handed to the next prompt

**Module path:** `ComputeTargets/tests/wkb_reference.py`. Every public name:

```python
# constants
PRODUCTION_SOURCE_SAMPLES_PER_LOG10Z = 100
PRODUCTION_RESPONSE_SPARSENESS       = 12
PRODUCTION_Z_END                     = 0.1
PRODUCTION_SMALLEST_K_INV_MPC        = 1.0e5
PRODUCTION_LARGEST_K_INV_MPC         = 3.0e8
PRODUCTION_SUPERHORIZON_EFOLDS       = 5
REFERENCE_K_VALUES                   = (1.0e5, 1.0e7, 3.0e8)
MODEL_KEYS                           = ("RadiationModel", "LambdaCDMModel", "QCDModel")
REFERENCE_DATA_PATH                  = Path(__file__).parent / "wkb_reference_data.json"

# grids
horizon_exit_z(cosmology, k_inv_Mpc: float, efolds_subh: float = 0.0) -> float
production_source_z_values(z_init: float, z_end: float = 0.1,
                           samples_per_log10z: int = 100) -> np.ndarray   # descending
to_redshift_array(z_values: Sequence[float]) -> redshift_array
production_source_grid(z_init: float, z_end: float = 0.1,
                       samples_per_log10z: int = 100) -> redshift_array
production_response_grid(source_grid: redshift_array, sparseness: int = 12) -> redshift_array

# stand-ins
class RadiationModel:
    name = "RadiationModel"
    def __init__(self, H0: float = 1.0)
    .H0, .cosmology (None), .functions            # ModelFunctions, with tau wired
    Hubble(z), tau(z), tau_delta(z_a, z_b), cs_tau(z), cs_tau_delta(z_a, z_b),
    friction_F_delta(z, z_ref), theta_G(k, z, z_init), rho_G(k, z, z_init) -> 0.0,
    rho_T(k, z, z_init), x_T(k, z)
class LambdaCDMModel:
    name = "LambdaCDMModel"
    def __init__(self, cosmology=None)
    .cosmology, .functions                        # functions.tau is None
class QCDModel:
    name = "QCDModel"
    def __init__(self, z_sample: redshift_array, cosmology=None,
                 atol: float = 1e-10, rtol: float = 1e-8)
    .cosmology, .functions, .background_payload, .z_sample     # functions.tau is None

# errors (README §6)
phase_error(theta, theta_ref) -> float
difference_error(delta, delta_ref) -> float
envelope_relative_error(value, value_ref, envelope) -> float
load_references(path: Optional[Path] = None) -> dict
```

**JSON schema** (`ComputeTargets/tests/wkb_reference_data.json`, 36 kB, schema_version 1). Top
level: `schema_version`, `generated`, `generator`, `campaign`, `environment`, `schema` (prose),
`k_values`, `k_keys`, `rho_anchor_efolds_subh` (= 3), `models`, `baselines`.

`k` keys are `f"{k:.6e}"`: `"1.000000e+05"`, `"1.000000e+07"`, `"3.000000e+08"`.

Per model (`"RadiationModel"`, `"LambdaCDMModel"`, `"QCDModel"`):

| key | content |
|---|---|
| `method` | prose: which reference method, and why |
| `reference_floor` | mpmath models: a note. QCD: per-quantity `{"agreement": {...}, "diagnostics": {...}}` |
| `grid` | `z_init`, `z_end`, `samples_per_log10z`, `num_nodes`, `superhorizon_efolds`, `largest_k_inv_Mpc` |
| `z_top` | = `grid.z_init` |
| `checkpoints` | list of `{index, z}`, descending; `index` is into the descending node array |
| `tau_minus_top` | parallel list: $\tau(z_j)-\tau(z_{\rm top}) = \int_{z_j}^{z_{\rm top}}dz/H > 0$ = `tau.delta(z_top, z_j)` |
| `cs_tau_minus_top` | parallel list: $\int_{z_j}^{z_{\rm top}}c_s\,dz/H > 0$, $c_s^2=$ `wPerturbations` |
| `friction_F_minus_top` | parallel list: $F(z_j)-F(z_{\rm top}) = -\int_{z_j}^{z_{\rm top}}\tfrac32(1+c_s^2)\,dz/(1+z) < 0$ — the primitive of `friction_RHS`, which **decreases** towards low $z$ while $\tau$, $c_s\tau$ increase |
| `rho_anchor_z` | per $k$: $z_{e3}(k)$, the off-grid root of $k(1+z)/H=e^3$ |
| `primitives_at_rho_anchor` | per $k$: `{tau_minus_top, cs_tau_minus_top, friction_F_minus_top}` at that anchor |
| `rho_G`, `rho_T` | per $k$: list of `{index, z, value}` for the checkpoints **strictly below** the anchor; `value` $=\int_z^{z_{\rm anchor}}C/(\omega+\omega_0)\,dz$ |
| `short_baseline` | three records `{index, z_node_hi, z_node_lo, z_fraction_offset (0.37), z_fraction, delta_tau_full, delta_tau_fraction}`, near $z=10^6$, $10^2$, 1; both deltas positive, the fraction taken in $u=\log(1+z)$ |
| `H0` | `RadiationModel` only (1.0) |
| `generation_time_seconds`, `build_time_seconds` | timings |

Units are `Mpc_units` (`Mpc = 1.0`): $\tau$, $c_s\tau$ in Mpc; $F$, $\rho$ dimensionless (radians).

**Phase reconstruction from the JSON** (used by `baseline_k1e5.py`, and what prompts 06 and 07
should score against):

```
theta_G(z_j; z_anchor) = -( k * (tau_minus_top[j] - primitives_at_rho_anchor[k].tau_minus_top)
                            + rho_G[k][j].value )
theta_T(z_j; z_anchor) = -( k * (cs_tau_minus_top[j] - primitives_at_rho_anchor[k].cs_tau_minus_top)
                            + rho_T[k][j].value )
F(z_j) - F(z_anchor)   = friction_F_minus_top[j] - primitives_at_rho_anchor[k].friction_F_minus_top
```

**Grids actually used.** LambdaCDM and QCD share one grid: $z_{\rm init} = 2.0636395964161516\times10^{16}$
(5 e-folds super-horizon for $k=3\times10^8$), **1,732 nodes** down to $z=0.1$. `RadiationModel`
($H_0=1$) has its own: $z_{\rm init} = 4.4523947729772961\times10^{10}$, **1,165 nodes**.
$\rho$ anchors: LambdaCDM 2.3076e9 / 2.3076e11 / 6.9227e12; QCD 2.4944e9 / 2.5982e11 / 1.0235e13;
Radiation 4977.7 / 4.9787e5 / 1.4936e7.

**Prototype design, for prompt 03 to move into `ComputeTargets/cumulative_table.py`.**
`docs/gktk-remedial/prototype_primitive.py:CumulativeTablePrototype`. Nodes ascending in
$u=\log(1+z)$; increments by Gauss–Legendre of the build order per production interval;
`hi[i] = fsum(increments[i:])`, `lo[i] = fsum([*increments[i:], -hi[i]])` (anchored at the top
node); `_nearest(u)` by `np.searchsorted` with an **exact** `u == u_node` short-circuit;
`_partial(u, node)` a Gauss-8 panel over at most half an interval;
`delta(z_a, z_b) = (hi[n_b] - hi[n_a]) + (lo[n_b] - lo[n_a]) + partial(u_b,n_b) - partial(u_a,n_a)`.

**Measured costs and accuracies, for prompt 03's acceptance and prompt 02's decision:**

* Build, both models, order 4: 6,924 Hubble evaluations, 0.037 s (LambdaCDM) / 0.084 s (QCD).
* $\tau$ at nodes, order 4: **1.81e-15** LambdaCDM (target $\le2\times10^{-14}$ — met with an
  order of magnitude to spare), **3.45e-07** QCD.
* $\Delta\tau$ over one production interval, double-double: **≤ 8.76e-15** relative (target
  $\le10^{-13}$).
* Off-grid `value`, order 4: 5.84e-16 LambdaCDM.
* `QCDModel` construction: **0.19–0.43 s**.
* Interval accessor: **4.4 µs / 0 Hubble evaluations on-grid**; 47.7 µs one endpoint off-grid;
  102.2 µs both off-grid, on QCD. LambdaCDM 3.0 / 8.2 / 13.1 µs.
* Hubble evaluation: 0.23 µs (LambdaCDM), 8.7 µs (QCD).

**Gauss orders are NOT decided here** — that is prompt 02 (README §7 D3). What prompt 02 inherits:
order 4 is at the double-precision floor for $\tau$ on LambdaCDM and raising it buys nothing; on
`QCD_Cosmology` **no fixed order works**, because `QCD_EOS.G(T)` switches branches at
`T_LO = 1e-5` GeV ($z = 4.191\times10^7$) and `T_120_MEV = 0.12` GeV ($z = 8.579\times10^{11}$)
and the error there falls only as $N^{-2}$ (6.33e-5 at order 4, 1.69e-5 at 8, 8.20e-6 at 12,
3.47e-6 at 20 — that is 4.78, 1.28, 0.62, 0.26 rad at $k=3\times10^8$). The number of production
intervals with a per-interval error above $10^{-12}$ is 301 / 163 / 38 / 8 at those orders (of
1,731); the tail is the 500-point $T(z)$ spline's knots, which fall every ~4 production intervals.
The obvious repairs, none of them taken here: adaptive quadrature on the offending intervals, or
subdividing the table at the two known branch-boundary constants.

**Baselines** (in the JSON's `"baselines"` block): $\theta_G$ error 13.90 rad and $\theta_T$ error
2.012 rad at $z=0.1$, $k=10^5$, LambdaCDM, production tolerances — 0.0 % and 0.1 % from the
review. The `"baselines"` block is written by `baseline_k1e5.py` and carried forward by
`generate_references.py`; re-run the baseline script after any regeneration.
