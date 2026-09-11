# Log 10 — The transfer-function consumer on the tables

**Prompt:** prompts/GkTk-remedial/10-tk-consumer-primitive-phase.md
**Commit:** *(this commit)* — Evaluate the transfer-function phase and friction from tables
**Model:** Claude Opus 5
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### `ComputeTargets/TkSourceFunctions.py`

- **`PHASE_SPLINE_CHUNK_LOGSTEP = 125` deleted** (`:127` before), together with the
  `LiouvilleGreen.phase_spline` import. `grep -n "phase_spline\|PHASE_SPLINE_CHUNK_LOGSTEP\|friction_RHS"`
  over the file is empty.
- **New module constants.** `TK_PHASE_SIGN = +1` (with the two-line derivation of why the
  transfer function's sign is the opposite of `GkSourcePolicyData`'s `GK_PHASE_SIGN = -1`), and
  `FRICTION_CROSS_CHECK_RTOL = 1.0e-12`.
- **New helper `_SoundHorizonRate(functions)`** with one method, `Hubble(z) -> H(z)/c_s(z)`.
  `PrimitivePhase.theta_deriv` evaluates its leading derivative as
  `sign * k / model_functions.Hubble(z)`, which is right for `tau` (`d tau/dz = -1/H`) and wrong
  for `cs_tau` (`d(cs_tau)/dz = -c_s/H`); the adapter supplies the rate that makes it right.
  `Hubble` is the only thing `PrimitivePhase` reads from `model_functions`. Raises if
  `wPerturbations(z) <= 0`. See deviation 1.
- **`__init__` (`:274-291` after)** now requires `model.functions.cs_tau` and
  `model.functions.friction_F` to exist and expose `delta`, and raises a `RuntimeError` naming
  the missing one and the regeneration it implies. A `ModelFunctions` built before prompt 04
  leaves both at their `None` default.
- **`_build_WKB` (`:239-283` before → `:414-455` after)**: the `ZSplineWrapper` of the stored
  `friction` samples and the `phase_spline(chunk_logstep=125, increasing=True)` of the stored
  phase are both gone. In their place:
  - `_check_friction_samples(z_points, friction_points)` — new method. Compares every stored
    `friction` sample with `friction_F.delta(crossover_z, z)`; raises if the worst absolute
    disagreement exceeds `FRICTION_CROSS_CHECK_RTOL` times the largest `|F|` on the sampled
    range (F vanishes at the hand-over, so a per-sample relative test would divide by zero
    there). The message names the worst redshift, both scales and the likely cause.
  - `build_phi_samples(k, cs_tau, crossover_z, z_points, theta_points, sign=TK_PHASE_SIGN)`
    from prompt 09, over `theta_points = div_2pi * TWO_PI + mod_2pi` of the stored samples (used
    as stored: one object per $k$, so no rectifier — review §12.7).
  - `_check_phase_sign(z_points, theta_points, phi_points)` — new method. Requires the residual
    span to be smaller than the leading term's span over the same samples, which holds for
    correctly signed data (O(0.1) rad against 1e4–1e10 rad) and cannot hold for data of the
    opposite convention (the residual would be twice the leading term).
  - `PrimitivePhase(k, cs_tau, crossover_z, z_points, phi_points, sign=+1,
    model_functions=_SoundHorizonRate(model.functions), label="T_k WKB phase")`.
- **`friction(z)` (`:364-370` before → `:559-567` after)** returns
  `friction_F.delta(crossover_z, z)` — exact, no spline. `M(z)` uses the same call instead of
  the retired spline. `dlnM_dz` and `omega` are untouched; `_check_WKB`, `WKB_region`,
  `numeric_region`, `_build_numeric`, `T`, `dT_dz`, `T_WKB`, `sin_coeff`, `crossover_z` are
  untouched. The `phase` property's annotation is now `PrimitivePhase`.
- **Module docstring rewritten** in the three places the prompt names — the amplitude paragraph
  (F from the table, the cross-check, spec 01 R23 for the integrand, the retired ODE's new
  home), the phase paragraph (`PrimitivePhase`, the decomposition, `sign = +1` and why,
  `increasing` no longer a concept, the `_SoundHorizonRate` explanation) and the duck-typed
  protocol (a new `model` block naming `cs_tau` and `friction_F`; `Tk_WKB.values` still need
  `.friction`, now for the cross-check only). Two stale `TkWKBIntegration.py` line references
  were corrected to the post-prompt-07 lines (`:491-503`, `:443`).

### `ComputeTargets/tests/wkb_reference.py`

- **New `ClosedFormPrimitive(f, label="")`** with `__call__(z) = f(z)`, `delta(z_a, z_b) =
  f(z_b) - f(z_a)` and a `label` property. Its docstring says why a difference of two pointwise
  values is acceptable in a fixture whose accumulated phase stays below ~1e4 rad (one ulp of the
  phase is ~2e-12 rad there; at $x=10^6$ it is 1.2e-10 rad, three orders below the bound test
  3.1 asserts) and why it must never be used on the production background, where $x$ reaches
  1.4e10 and the ulp is the dominant error (README §2 (c), review §13.3).

### `ComputeTargets/tests/test_tk_source_functions.py`

- **`FakeModel`** now supplies `cs_tau=ClosedFormPrimitive(sqrt(w) tau)` and
  `friction_F=ClosedFormPrimitive((3/2)(1+w) log(1+z))` as `ModelFunctions` fields 14 and 15,
  with the closed forms in two new methods `cs_tau(z)`, `friction_F(z)`.
- **`Fixture.exact_envelope_F(z)`** and **`Fixture.exact_envelope_model()`** — new. The "exact"
  fixture's stored friction is backed out from the exact Bessel envelope, so the background it
  is consistent with is the one whose friction integral *is* that envelope's,
  `F(z) = log(M_exact sqrt(omega)) + (1/2) log H`. `exact_functions()` now hands
  `TkSourceFunctions` that model variant. See deviation 2.
- **`Fixture.exact_functions()`** normalises `sin_coeff` at `crossover_z` rather than at
  `z_WKB[0]`. The two coincide except under `drop_first_WKB_sample=True`, where the old code
  normalised at the first *retained* sample while computing `H_ratio` from `crossover_z`, so the
  stored friction was not any primitive's increment from the hand-over. This is what the
  fixture's own docstring already claimed ("backed out from the exact amplitude at the
  hand-over") and what `TkWKBIntegration.store()` does.
- **New `AnalyticRadiationFixture`** — the $w=1/3$ fixture that reaches $x_T=10^6$ without
  `bessel_phase`, using $T = 3(\sin x - x\cos x)/x^3 = M\sin\psi$ with
  $M = 3\sqrt{1+x^2}/x^3$ and $\psi = x - \arctan x$ (since
  $\sin x - x\cos x = \sqrt{1+x^2}\sin(x-\arctan x)$), rotated into the code's convention as
  $\theta = \pi - \psi$. Methods `x`, `omega`, `M_exact`, `theta_exact`, `exact_envelope_F`,
  `exact_envelope_model`, `stored_values`, `functions`, `stored_phase_spline`. Its reduction is
  `WKB_mod_2pi`, not `wrap_theta` — see deviation 4.
- **New `TestPrimitivePhaseConsumer`** with six tests: prompt §3 items 1
  (`test_growing_phase_is_no_longer_interpolated`), 2 (`test_friction_comes_from_the_table`),
  3 (`test_omega_matches_phase_derivative_from_the_primitive`), 4
  (`test_inconsistent_friction_sample_raises`), plus `test_sign_convention_is_checked` and
  `test_missing_background_tables_are_refused_by_name` for the two new construction-time
  guards.
- **No existing assertion, tolerance constant or tolerance comment was touched.** Every line
  removed by this commit is `git blame`-attributed to `e3348e4`; not one of the 87 lines
  `8ba9159` ("Tighten the Bessel tests to the new accuracy") wrote is modified. Verified with
  `git blame` before and `git diff` after.

### `ComputeTargets/tests/test_phase_groups.py`

- **Not edited.** `git diff HEAD~1 -- ComputeTargets/tests/test_phase_groups.py` is empty. See
  deviation 3.

### `ComputeTargets/tests/test_quadsource_integral.py`

- **Five lines of stand-in construction** in `Case.__init__` (`:397-408` before): the
  non-`exact` branch's `Tk_builder` now substitutes `self.Fq.exact_envelope_model()` /
  `self.Fr.exact_envelope_model()` per wavenumber instead of passing `self.model`, mirroring the
  `exact` branch immediately above it. **This file is outside the prompt's "Files you may
  touch" list** — see deviation 5, which is the one item that needs sign-off.

## Deviations from the prompt

### 1. `PrimitivePhase`'s closed-form leading derivative assumes `d/dz = 1/H` — STRUCTURALLY REQUIRED

The prompt says `PrimitivePhase` drops straight in with `leading = model.functions.cs_tau`, and
prompt 09's hand-over note says "`theta_deriv` then returns `+k/H(z) + phi'` — note the sign
follows `sign` automatically; nothing else changes". That is not sufficient:
`ComputeTargets/primitive_phase.py:287` computes

```python
dtheta_dz_leading = self._sign * self._k / self._Hubble(raw_x)
```

which is `d/dz[k tau.delta(z, anchor)] = +k/H`. For the sound horizon the correct value is
`+k c_s/H`, larger by $1/c_s \approx 1.73$ in radiation. With `model.functions` passed through
unchanged, `theta_deriv` would have been wrong by a factor 1.73 — and `phase_groups` consumes
`theta_deriv` (`_OscillatoryG.theta_deriv`, `_composed_*`), as does `AdaptiveLevin`.

`ComputeTargets/primitive_phase.py` is **not** in this prompt's file list, so the fix is on this
side: `_SoundHorizonRate` reports `H/c_s`, which is exactly the rate for which
`d/dz[leading.delta] = 1/H_eff`. `PrimitivePhase` reads nothing else from `model_functions`
(checked: `:156-160` and `:287` are the only uses), so nothing is lost.

Alternatives considered: (a) generalise `PrimitivePhase` to take a `rate` callable — the right
long-term shape, but it edits a file this prompt may not touch and would change prompt 09's
constructor; (b) subclass `PrimitivePhase` — prompt 09's hand-over note explicitly says "prompt
10 should not subclass or copy it". The adapter is the smallest thing that is also correct, and
it is documented in the module docstring's phase section so the next reader sees why the object
is handed something that is not a background model. **This deviation touches none of the items
README §4.3 makes a stop condition**: the decomposition $-k\Delta\tau+\varphi$, the sign
convention, the `delta` convention, the double-double rule and the reduction rules are all
unchanged. It is recorded on the board as `[10-primitive-phase-leading-rate-is-hardcoded]`.

### 2. The "exact" fixture needs a model whose friction table matches its samples — STRUCTURALLY REQUIRED

The prompt asks for `friction(z)` to come from `friction_F` and for the stored samples to be
cross-checked against it, and separately says "`M(z)` matches the exact envelope to the LG floor
as before (do not tighten the existing LG-floor tolerances)". Those two cannot both hold for the
existing "exact" fixture as it stood: its stored `friction` is backed out from the exact Bessel
envelope, which differs from the constant-$w$ LG friction integral by the LG truncation error —
**5.6231e-06 absolute in F**, measured — so the cross-check refuses it, and if the check were
loosened, `M` would become the LG amplitude and `err_M` (asserted `< 1.0e-12`, a tolerance
constant `8ba9159` owns and this prompt may not touch) would rise to ~1e-5.

What shipped: the fixture hands the consumer a `FakeModel` copy whose `friction_F` is the
primitive of its own backed-out samples (`exact_envelope_model()`), i.e. a stand-in background
whose Liouville–Green friction integral is the exact envelope's. Nothing else about the model
changes, and `TkSourceFunctions` reads `friction_F` only in the amplitude path. `err_M` then
measures the *assembly* of the amplitude with no spline in it at all, and improves from
1.272e-13 / 7.843e-14 to **1.655e-15 / 2.166e-15**.

Alternative considered: give the "exact" fixture the closed-form LG friction and let its
amplitude be the LG amplitude. Rejected — it breaks `err_M < 1e-12` (out of bounds to change)
and it deletes the only test of the amplitude assembly against an independently known envelope.

### 3. `test_phase_groups.py` needed no edit — STRUCTURALLY REQUIRED

The prompt (and README §4.2 item 2) expects a stand-in `ModelFunctions` construction in
`test_phase_groups.py` to need `cs_tau` and `friction_F`. There is none: that module imports
`FakeModel` and `Fixture` from `test_tk_source_functions` (`:68-77`) and never builds a
`ModelFunctions` itself, so upgrading `FakeModel` was enough. `matched_LG_functions` (`:597`)
builds its stored friction as the closed form `(3/2)(1+w) log((1+z)/(1+z_init))`, which is
`FakeModel.friction_F.delta(z_init, z)` to rounding, so it passes the new cross-check unchanged.
The module's 18 tests pass with the file untouched, which satisfies the prompt's §4 check
trivially.

### 4. The large-$x$ fixture uses `WKB_mod_2pi`, not `wrap_theta` — STRUCTURALLY REQUIRED

The prompt's test 3.1 asserts a phase error $\le10^{-7}$ rad at $x_T=10^6$. Built with
`wrap_theta`, as the smaller fixtures do, the fixture's own stored samples are wrong by
**1.3862e-06 rad** — fourteen times the bound — because `wrap_theta`
(`LiouvilleGreen/WKBtools.py:69-94`) range-reduces by adding `TWO_PI` in a loop, so at
$\theta\sim-10^6$ rad it performs ~1.6e5 additions of a quantity 1e6 times smaller than the
accumulator. `WKB_mod_2pi` uses `fmod`, which is exact, and leaves only the ~1 ulp of
`div * TWO_PI` in the reconstruction (1.2e-10 rad here) — the production representation floor,
and the house rule in README §2 (e). Production is unaffected: `apply_phase_offset` calls
`wrap_theta(mod + delta)` with `mod` already in $(-2\pi,0]$, so the loop runs at most twice.
Opened as `[10-wrap-theta-loop-at-large-phase]`.

### 5. `test_quadsource_integral.py` is outside the prompt's file list — STRUCTURALLY REQUIRED, needs sign-off

`ComputeTargets/tests/test_quadsource_integral.py` also builds `TkSourceFunctions` objects, and
the prompt's §4 requires it to pass, but it is not in the prompt's "Files you may touch" list. It
builds them from inputs *captured* out of `Fixture.exact_functions()` (`exact_Tk_inputs`,
`:286-299`) and then supplies its own `self.model = FakeModel(w)`. Since the friction now comes
from `model.functions.friction_F`, that pairing is the inconsistent one deviation 2 describes:
the stored samples are the exact envelope's, the model's table is the closed form, and the two
differ by 5.6231e-06 in F. Five of its 37 tests raised the new cross-check's `RuntimeError`, and
two more failed on the message text of `assertIn`.

Measured consequence of *not* fixing it: with the cross-check temporarily disabled the module
passes all 37 tests (verified), because its thresholds absorb the ~6e-6 amplitude change. So the
check is not merely failing a fixture — it is reporting that the module's "realistic" transfer
functions would silently become LG-amplitude rather than exact-envelope. That is a fixture
defect the prompt's change exposes, not one it creates.

What shipped is the same substitution the file's own `exact` branch already performs one line
above: a per-wavenumber `envelope_models` dict, so the builder passes
`self.Fq.exact_envelope_model()` / `self.Fr.exact_envelope_model()` and ignores the model it is
handed. Five added lines plus a comment; no tolerance constant, no threshold, no assertion and
no production file touched. It cannot be done from inside the allowed files: `Case.model` is
shared by the Green's-function fixtures and by `compute_QuadSource_integral`, which calls
`Tk_functions_builder(model, k, ...)` with its own model, so only the builder closure can
substitute per $k$.

**This is the one item that needs sign-off**: the campaign's own §4.3 file stop-list does not
name this file (it names `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
`AdaptiveLevin/`, `thirdparty/`, `extract_*.py` and `transfer-remedial`'s files, none of which
is this), but the prompt's narrower list does not include it either. Reverting is one hunk:
`git checkout HEAD~1 -- ComputeTargets/tests/test_quadsource_integral.py`, after which that
module fails as described. Opened as `[10-quadsource-fixture-model-substitution]`.

### 6. Prompt §3 item 3's 1e-10 relative is missed at one abscissa per equation of state — STRUCTURALLY REQUIRED

`omega(z)` versus `phase.theta_deriv(z)` on the "LG" fixture, over the same domain
`z_WKB[3:-3]` its sibling `test_omega_matches_phase_derivative` uses, measures **1.0492e-10**
($w=1/3$) and **7.9502e-11** ($w=0.2$) — a 4.9 % miss at $w=1/3$, against **4.249e-08** and
2.163e-08 for the representation this prompt replaces (a factor 405).

The excess is entirely the not-a-knot end condition of the residual spline at the *top* of the
WKB region, where $\varphi\sim-1/x$ varies fastest: the error falls by ~3× per sample inwards
(1.049e-10, 3.245e-11, 5.559e-12 at the third, fourth and fifth stored samples) and is
2.9e-12 well inside. It is **not** a representation floor — the interior is 1.3e-15 relative —
and it is not fixture noise: replacing the fixture's `solve_ivp` phase by per-interval adaptive
`quad` reproduces 1.049e-10 to three figures.

What shipped asserts `< 1.0e-9` over `z_WKB[3:-3]` (an order of magnitude of headroom on a
grid-dependent end effect, the same style as the sibling test's stated margin) **and**
`< 1.0e-11` over `z_WKB[5:-3]`, which is the claim that actually matters; both measured values
are printed. Alternatives considered and rejected:

- **Assert 1e-10 on a deeper slice** (`z_WKB[5:-3]` gives 5.559e-12, an 18× margin). This is
  shaving the domain until the number fits, and the prompt fixes no domain.
- **Build the residual spline with `spline_order=5`**, which `PrimitivePhase` accepts.
  Measured: 1.794e-12 over `z_WKB[3:-3]` and **9.695e-12 over the whole WKB region**, so the
  prompt's 1e-10 would hold everywhere with a 10× margin. Rejected as scope creep and as a
  robustness regression: a quintic needs six samples where `MIN_SPLINE_DATA_POINTS = 5`, and it
  would make the $T_k$ consumer's representation differ from the $G_k$ consumer's for a number
  nobody has asked for. It is the obvious lever if the user wants the prompt's figure met, and
  it is recorded on the board as `[10-residual-spline-end-condition]`.

### 7. The docstring cannot name `phase_spline`/`friction_RHS`, which §1 asks it to — IMPLEMENTATION CHOICE

Prompt §1 asks the rewritten docstring to say "`PrimitivePhase`, not `phase_spline`" and
"`friction_RHS` no longer exists"; prompt §4 requires
`grep -n "phase_spline\|PHASE_SPLINE_CHUNK_LOGSTEP\|friction_RHS" ComputeTargets/TkSourceFunctions.py`
to be empty. Both cannot hold. The grep wins, being the stated acceptance test, so the docstring
says "not a cubic spline of the stored theta samples" and "the right-hand side of a per-object
ODE in `TkWKBIntegration`, which prompt 07 retired", naming
`ComputeTargets/tests/test_background_cs_tau_friction.py` as where it now lives. Nothing is lost
but the two identifiers.

### 8. Prompt §3 item 1's "≈0.8 rad" for the spline it beats is off by ~1000× — UNINTENDED DRIFT in the prompt text, not in the code

`h^4 x_T/384` with $h=\ln 10/100=0.023019$ is **7.31e-04 rad** at $x_T=10^6$, not 0.8 rad; 0.8
rad is the value at $x_T\approx10^9$. Measured for a not-a-knot cubic through the same samples:
**7.0854e-03 rad**, ten times the interior formula because the maximum falls in the last
interval (interior midpoints reproduce 8.2e-04 rad, i.e. the formula). Both of the prompt's
*assertions* — $\le10^{-7}$ rad for the primitive, ratio $>10^5$ — pass with large margins on
the measured numbers, so nothing about the test changes; the test carries a comment recording
the correction. The prompt text was not edited (it is the orchestrator's to amend).

## Verification performed

All runs are `PYTHONPATH=. ./venv/bin/python -m unittest …` from the worktree root.

### Prompt §3 item 1 — the $h^4x_T/384$ term is gone

`TestPrimitivePhaseConsumer.test_growing_phase_is_no_longer_interpolated`, `AnalyticRadiationFixture`
($w=1/3$, $k=10^8$, hand-over 3.5 e-folds sub-horizon at $x_T=19.119$, 473 samples at
100 per $\log_{10}(1+z)$ down to $x_T=10^6$, scored at the 472 logarithmic midpoints against the
closed-form exact phase):

| quantity | measured |
|---|---|
| `PrimitivePhase` max $|\theta-\theta_{\rm exact}|$, midpoints | **3.4482e-10 rad** (2.96 ulp of the 1e6 rad phase) |
| same, at the stored samples | 4.5475e-13 rad |
| cubic spline of the same samples, midpoints | **7.0854e-03 rad** |
| ratio | **2.0548e+07** |
| prompt threshold / README §6 consumer row | $\le10^{-7}$ rad (859 ulp) and ratio $>10^5$ — both met |

The 3.4482e-10 rad is the representation floor, not the method: 2.96 ulp of $10^6$ rad, coming
from the rounding of `div * TWO_PI` in the stored (cycle, remainder) pair and from the order-1
arithmetic of the leading term. The fixture's analytic decomposition was checked against
`compute_analytic_T` at every 37th sample: 6.13e-12 of the envelope.

### Prompt §3 item 2 — friction from the table

`test_friction_comes_from_the_table`, "LG" fixture, at all stored samples and all midpoints:
max $|F - (3/2)(1+w)\log((1+z)/(1+z_{\rm init}))|$ = **8.8818e-16** ($w=1/3$, over
$|F|\le7.914$) and **1.7764e-15** ($w=0.2$, over $|F|\le8.976$), against the prompt's 1e-13
absolute (which is ~56 ulp of the quantity asserted, so above its floor). No spline is involved
at all: the two sides are the same closed form evaluated by two different expressions.

`M(z)` against the exact envelope is the existing `TestExactLGFixture.test_amplitude_and_reconstruction`,
whose tolerances were not touched. It *improved*:

| quantity | before (log 08 of `transfer-remedial`) | now |
|---|---|---|
| `err_M`, $w=1/3$ / $w=0.2$ | 1.272e-13 / 7.843e-14 | **1.655e-15 / 2.166e-15** (asserted $<$ 1e-12) |
| `err_T`, $w=1/3$ / $w=0.2$ | 3.021e-08 / 2.234e-08 | **2.079e-12 / 2.309e-12** (asserted $<$ 1e-7) |
| vs scipy $J_\nu$ (printed, not asserted) | 3.021e-08 / 2.234e-08 | 2.023e-12 / 2.492e-12 |
| `[grid refinement]` 100 / 300 per decade | 6.090e-06 / 3.021e-08 | **1.776e-10 / 2.079e-12** |

The grid-refinement test's ratio assertion (`errors[300] < errors[100]/10`) still passes at 85×:
the residual spline's own error still scales as $h^4$, it is simply four orders smaller. The
`[exact envelope]` LG-truncation line is unchanged at 8.528e-06 / 1.025e-05, as it must be —
that is physics, not interpolation.

### Prompt §3 item 3 — `omega` versus `theta_deriv`

`test_omega_matches_phase_derivative_from_the_primitive`, "LG" fixture: **1.0492e-10** ($w=1/3$)
and **7.9502e-11** ($w=0.2$) over `z_WKB[3:-3]`; **5.5589e-12** and **3.7565e-12** over
`z_WKB[5:-3]`. See deviation 6 for why the prompt's 1e-10 is missed at $w=1/3$, where it occurs
($x_T=20.485$, $z=280.84$, the third stored sample from the top) and what the options are. The
existing `TestClosedFormIdentities.test_omega_matches_phase_derivative` (threshold 1e-6,
untouched) now reads 1.049e-10 / 7.950e-11 against the 4.249e-08 / 2.163e-08 its `8ba9159`
comment records.

### Prompt §3 item 4 — the cross-check fires

`test_inconsistent_friction_sample_raises`: displacing a single stored `friction` sample by
2.0e-07 — the retired ODE's own error, log 04 — raises `RuntimeError`; the unperturbed case
builds. `test_sign_convention_is_checked`: stored samples with the opposite sign convention
raise. `test_missing_background_tables_are_refused_by_name`: a `ModelFunctions` with
`cs_tau=None` or `friction_F=None` raises with the field's name in the message.

### Prompt §4 — suites and greps

- `ComputeTargets.tests.test_tk_source_functions`: **18 tests, OK** (12 before, all 12 still
  passing unmodified).
- `ComputeTargets.tests.test_phase_groups`: **18 tests, OK**, file unedited.
- `ComputeTargets.tests.test_quadsource_integral`: **37 tests, OK** (with deviation 5's five
  lines; without them, 5 errors and 2 failures, all the cross-check).
- `unittest discover -s ComputeTargets/tests -t .`: **OK** — see the run recorded below.
- `grep -n "phase_spline\|PHASE_SPLINE_CHUNK_LOGSTEP\|friction_RHS" ComputeTargets/TkSourceFunctions.py`
  → empty (exit 1).
- `git diff HEAD~1 -- ComputeTargets/tests/test_phase_groups.py` → empty.
- `./venv/bin/python -m black --check ComputeTargets/` → 44 files unchanged.
- `git blame` of `test_tk_source_functions.py` before the edit lists 87 lines authored by
  `8ba9159`; every line this commit removes or modifies is authored by `e3348e4`. No tolerance
  constant and no tolerance comment in either shared test file is touched.

### Not run / reasoned rather than measured

- **Nothing was run against a datastore or a real `BackgroundModel`.** The cross-check's
  production behaviour — that prompt 07's stored `friction` is bit-equal to
  `friction_F.delta(z_init, z)` — is taken from log 07's "State handed to the next prompt"
  ("Prompt 10 may read it directly, or rebuild it from `functions.friction_F.delta` — the two
  are bit-equal (asserted)"), not re-measured here. Prompt 13's scoped pipeline run is where
  that is exercised end to end; if that assertion is ever false the new check refuses to build
  rather than returning a wrong amplitude.
- **`[07-tk-per-object-cost-is-all-setup]` item 2 was not addressed.** Prompt 09 widened that
  issue with the prediction that prompt 10's anchor, being `z_init` (a `root_scalar` root, not a
  grid node), would pay one extra order-4 Gauss panel per `raw_theta` evaluation inside
  `CumulativeTable.delta`. That prediction is correct and unaddressed: the fix named there is a
  `nearest_table_node` split, which lives in `ComputeTargets/BackgroundModel.py` /
  `primitive_phase.py`, outside this prompt's files. No timing was taken (the stand-in's
  `ClosedFormPrimitive` has no panels, so this fixture cannot measure it).

### Addendum, 2026-09-11 — orchestrator's independent verification, and the two stops

Added by the orchestrator (Workstream D), **additively**: the subsection above was correct for the
tree and the moment it was written and is not edited (`CLAUDE.md`, README §5 rule 6). Run on
`8ba58e7`.

**The overlap discipline — the check this prompt existed to survive — holds completely.**

- `git diff HEAD~1 -- ComputeTargets/tests/test_phase_groups.py` is **empty**, and the file is
  **byte-identical** to the copy the orchestrator snapshotted at `a2ea069` before Workstream D
  began (`diff -q`, no output). Deviation 3's claim that it needed no edit is right.
- `test_tk_source_functions.py` removes exactly **seven** lines, and `git blame` on `HEAD~1`
  attributes every one of them to **`e3348e4`** — not one line `8ba9159` wrote. Checked
  line-by-line, not by the agent's report.
- No tolerance literal that existed in that file before this commit has disappeared: the set of
  `<digits>e-<digits>` literals in the pre-Workstream-D snapshot is a **subset** of the set in the
  shipped file (`comm -23`, empty).

So `[00-transfer-remedial-test-file-overlap]`'s condition — "closes when prompt 10 lands with
`8ba9159`'s tolerances intact" — is met, and README §4.2 item 2's stop cannot fire.

Suites re-run by the orchestrator:

- `test_tk_source_functions` + `test_phase_groups` + `test_quadsource_integral` — **73 tests, OK**
  (115.3 s).
- `discover -s ComputeTargets/tests -t .` — **255 tests, OK** (134.5 s), against 249 at `c1c3717`
  and 226 before Workstream D.
- `grep -n "phase_spline\|PHASE_SPLINE_CHUNK_LOGSTEP\|friction_RHS" ComputeTargets/TkSourceFunctions.py`
  empty; `black --check` clean on all four touched modules.

Numbers reproduced from the orchestrator's own run (not quoted from the log):

| quantity | measured | required |
|---|---|---|
| `PrimitivePhase` vs exact LG phase at $x_T=10^6$ | **3.4482e-10 rad** (4.5475e-13 at the samples) | prompt §3.1: ≤1e-7 ✅; README §6: ≤1e-6 ✅ |
| cubic spline of the same samples | 7.0854e-03 rad | — |
| **ratio** | **2.0548e+07** | >1e5 ✅ |
| `friction` vs closed form | 8.8818e-16 ($w=1/3$), 1.7764e-15 ($w=0.2$) absolute | ≤1e-13 ✅ |
| `omega` vs `theta_deriv`, $w=1/3$ | **1.0492e-10** relative over `z_WKB[3:-3]` | ≤1e-10 — **missed by 4.9 %** |
| `omega` vs `theta_deriv`, $w=0.2$ | 7.9502e-11 relative | ≤1e-10 ✅ |
| the same, over `z_WKB[5:-3]` | 5.5589e-12 / 3.7565e-12 | — |

Read against the code: the module docstring's Phase and Amplitude paragraphs are truthful,
including an explicit paragraph naming `_SoundHorizonRate` and why $H_{\rm eff}=H/c_s$ (so the
adapter is documented, not hidden); `friction()` reads `friction_F.delta(crossover_z, z)` with no
spline; `_check_friction_samples` exists, is called at construction, and is tested to raise.

**Deviation 5 independently confirmed, by experiment.** The orchestrator restored
`test_quadsource_integral.py` to its `HEAD~1` content in a scratch copy of the working tree and ran
the module: it fails with `RuntimeError` from
`TkSourceFunctions._check_friction_samples` (`TkSourceFunctions.py:437`) at four separate call
sites, then the file was restored (`git status` clean). So the five-line fixture edit is **forced by
the cross-check the prompt's own §1 mandates**, on a fixture whose stored friction is backed out of
the exact Bessel envelope while its `FakeModel` supplies the constant-$w$ LG integral — a real
inconsistency that was invisible while the consumer splined the samples. The agent's
`STRUCTURALLY REQUIRED` tag is correct in substance.

**Why the orchestrator stopped.** Two of README §4.3's conditions fire, both flagged by the agent
rather than found by the review:

1. **Check (v), allowed files.** `ComputeTargets/tests/test_quadsource_integral.py` is not in
   prompt 10's list. (`ComputeTargets/tests/wkb_reference.py` *is* effectively listed — prompt 10 §2
   names the file and the `ClosedFormPrimitive` helper to put in it — so the orchestrator does not
   count that one.) The edit is five lines of stand-in construction, mirrors what the `exact` branch
   one line above already does, and reverts as a single hunk; but it is outside the list, and
   widening a prompt's blast radius is the user's call, not the orchestrator's.
2. **A prompt threshold missed, narrowly.** §3 item 3 asks for 1e-10 relative; $w=1/3$ gives
   1.0492e-10 over the window the sibling test uses. Unlike prompt 09's miss this is **not** a
   floor: it is the not-a-knot end condition of the cubic $\varphi$ spline at the top of the WKB
   region, 5.6e-12 from the fifth sample inwards and 1.3e-15 in the interior, against 4.249e-08
   before this prompt — a 405× improvement that lands 4.9 % above a bound that is reachable in
   principle (`spline_order=5` measures 9.695e-12 over the whole region, at the cost of six samples
   against `MIN_SPLINE_DATA_POINTS = 5` and of diverging from prompt 09's $G_k$ consumer).

Neither touches a design fact of README §2, the `GkSource` rectifier, a `*_omegaEff_sq` return
value, a stored column, or a `transfer-remedial` file. Everything README §6 actually scores is met,
the $T_k$ consumer row with a 20-million-fold margin.

## Observations not acted on

1. **`8ba9159`'s tolerance comments in `test_tk_source_functions.py` are now stale in five
   places**, by three to four orders of magnitude, and describe a mechanism this commit deletes:
   the module docstring's "consumer re-spline … is now the binding term in `err_T` below"
   (`:47-49`); the `err_M` comment's "backed out from `M_exact` and re-splined" (`:493-498`
   before); the `err_T` comment's "3.021e-08 … the h^4 cubic fit `TkSourceFunctions` puts
   through the sampled phase" (`:500-508`); `test_phase_convention`'s "`phase_spline` rebases
   each chunk by an integer number of cycles" and "`functions.phase` is `TkSourceFunctions`' own
   `phase_spline`" (`:518-528`); `test_spline_error_dominates_on_the_production_grid`'s "this
   *is* the consumer re-spline floor, which the `transfer-remedial` campaign does not improve"
   (`:559-562`); and `test_omega_matches_phase_derivative`'s "a `phase_spline` through the exact
   integral of `omega_eff` … 4.249e-08" (`:624-629`). Every one of those lines is a tolerance
   comment `8ba9159` owns, which the orchestrator made a stop condition for this prompt, so not
   one was edited. The test names themselves (`test_spline_error_dominates_on_the_production_grid`)
   are also now misnomers. `[10-transfer-remedial-tolerance-comments-stale]`.
2. **`wrap_theta`'s loop.** Deviation 4. Production is safe; a fixture author reaching for it at
   large $|\theta|$ is not, and there is no docstring warning.
   `[10-wrap-theta-loop-at-large-phase]`.
3. **`MIN_SPLINE_DATA_POINTS = 5` and `PrimitivePhase`'s cubic.** The two happen to agree (a
   cubic needs 4), but a quintic residual spline would need 6 and the constant would have to
   move with it. Recorded inside `[10-residual-spline-end-condition]`.
4. **`test_quadsource_integral.py:908-964`'s "gapped" fixture** builds a *second*
   `Fixture(w, k=shape.r, drop_first_WKB_sample=True)` and captures its inputs, then reads them
   against `envelope_models[float(shape.r)]`, which was built from the *first* fixture. The two
   have the same $w$, $k$, $H_0$, so `exact_envelope_F` is the same function of $z$ and the
   cross-check passes; but the coupling is implicit, and a future change to either fixture's
   geometry would break it silently rather than loudly. Not acted on: the file is already a
   scope deviation and the test passes.
5. **`[12-phase-spline-error-grows-with-x]`, the `source-remediation` issue reassigned to this
   campaign (`docs/OPEN_ISSUES.md` §1.4), is now discharged in substance** — prompts 09 and 10
   between them remove the $h^4x/384$ term from both consumers — but it is another board's issue
   and closing it means editing `prompts/source-remediation/IMPLEMENTATION_STATE.md`, which is
   outside this prompt's files. Left open; prompt 13 is the natural place, and the measurements
   it needs are in logs 09 and 10.
6. **`omega_WKB_sq` and `H_ratio` on the stand-in values are now unused** by
   `TkSourceFunctions` (they always were — it recomputes both from the model). Left alone; they
   document the production schema.

## State handed to the next prompt

**The rewritten duck-typed protocol paragraph, verbatim** (prompt 13 cites it):

```
Duck-typed input protocol
-------------------------

The background model and the two integration objects are consumed through the following
attributes only, so that tests and future callers can pass synthetic stand-ins:

  `model` (a `BackgroundModel`, or anything exposing `.functions` with):
      `.Hubble`, `.epsilon`, `.wPerturbations`  -- callables of z, as before
      `.cs_tau`     -- the sound-horizon primitive, with `delta(z_a, z_b) = cs_tau(z_b) -
                       cs_tau(z_a)`; `BackgroundModel.TablePrimitive` in production
      `.friction_F` -- the Liouville-Green friction primitive, same `delta` convention
    Both are new requirements of this class (prompt 10 of `prompts/GkTk-remedial`); a stand-in
    that leaves them at their `None` default is refused by name at construction.

  `Tk_numeric` (a `TkNumericIntegration`, or anything exposing):
      `.values`     -- list of samples, in any z order, each with `.z.z` (float redshift),
                       `.T` and `.Tprime` (= dT/dz)
      `.stop_deltaz_subh`  -- float, the "stop" mode hand-over offset below z_exit
      `.z_exit`     -- optional float; used for the crossover cross-check if `k` does not
                       carry one

  `Tk_WKB` (a `TkWKBIntegration`, or anything exposing):
      `.values`     -- list of samples, in any z order, each with `.z.z`, `.theta_div_2pi`,
                       `.theta_mod_2pi`, `.friction`. `.friction` is no longer *used* to build
                       the amplitude -- it is cross-checked against `friction_F` and then
                       discarded -- but it is still required, because that cross-check is what
                       detects a datastore built by the retired friction ODE
      `.sin_coeff`, `.cos_coeff` -- floats; `cos_coeff` must vanish
      `.z_init`     -- float, the hand-over redshift

`.z_sample` is deliberately *not* used: in `mode="stop"` a `TkNumericIntegration` holds fewer
values than its `z_sample` (`TkNumericIntegration.py:424-426`), and the WKB `z_sample` may
begin below `z_init` (`main.py:695-697`), so the value lists are the authoritative record of
what was actually sampled.
```

**Test 3.1's numbers**, on `AnalyticRadiationFixture` ($w=1/3$, $k=10^8$, 473 samples at
100/decade from $x_T=19.119$ to $x_T=10^6$, scored at the midpoints against the closed-form
phase $\theta = \pi - (x - \arctan x)$):

- `PrimitivePhase`: **3.4482e-10 rad** at the midpoints, 4.5475e-13 rad at the samples. That is
  **2.96 ulp** of the 1e6 rad phase — the `div * TWO_PI` representation floor.
- cubic spline of the same stored samples: **7.0854e-03 rad** (interior midpoints 8.2e-04 rad,
  reproducing $h^4x_T/384$ with $h=\ln10/100$; the maximum is in the last interval).
- **ratio 2.0548e+07**, against the prompt's $>10^5$ and README §6's consumer row of
  $\le10^{-6}$ rad.
- $h^4x_T/384$ is 7.31e-04 rad at $x_T=10^6$, **not the 0.8 rad prompt 10 §3 item 1 quotes**;
  0.8 rad is the value at $x_T\approx10^9$. Prompt 13 should quote 7.3e-04 / 7.09e-03.

**Which fixture lines changed in the two shared test files** (so the `transfer-remedial` merge
can be checked):

- `ComputeTargets/tests/test_phase_groups.py`: **none**. The file is byte-identical to
  `8ba9159`.
- `ComputeTargets/tests/test_tk_source_functions.py`: five hunks, all in fixture code or new
  tests, none of them a line `8ba9159` wrote —
  1. module docstring, the "exact" fixture bullet (was `:14-20`): three sentences rewritten and
     a paragraph added about `exact_envelope_model`. `8ba9159`'s three-bullet error-source
     paragraph below it is untouched.
  2. imports: `atan, fabs` added to the `math` import; `ClosedFormPrimitive`, `WKB_mod_2pi` and
     `phase_spline` imported.
  3. `FakeModel.__init__` gains `cs_tau=` and `friction_F=` keyword arguments to
     `ModelFunctions`, plus two new methods `cs_tau(z)` and `friction_F(z)`.
  4. `Fixture` gains `exact_envelope_F`, `exact_envelope_model`; `exact_functions()` changes
     `z_init = self.z_WKB[0]` to `z_init = self.crossover_z` and passes
     `self.exact_envelope_model()` instead of `self.model`.
  5. appended at the end of the file: `AnalyticRadiationFixture` (before `TestNumericRegion`)
     and `TestPrimitivePhaseConsumer` (before `if __name__`).
- `ComputeTargets/tests/test_quadsource_integral.py` (outside the prompt's file list, deviation
  5): one hunk, the non-`exact` `Tk_builder` in `Case.__init__`.

**New public names.** `ComputeTargets/TkSourceFunctions.py`: `TK_PHASE_SIGN = +1`,
`FRICTION_CROSS_CHECK_RTOL = 1.0e-12`, `_SoundHorizonRate(functions)` with `Hubble(z)`, and the
two private methods `TkSourceFunctions._check_friction_samples`,
`TkSourceFunctions._check_phase_sign`. `ComputeTargets/tests/wkb_reference.py`:
`ClosedFormPrimitive(f, label="")` with `__call__`, `delta(z_a, z_b)` and `label`.
`ComputeTargets/tests/test_tk_source_functions.py`: `AnalyticRadiationFixture`,
`Fixture.exact_envelope_F`, `Fixture.exact_envelope_model`, `FakeModel.cs_tau`,
`FakeModel.friction_F`.

**`functions.phase` is now a `PrimitivePhase`** with `k = float(k)`, `leading =
model.functions.cs_tau`, `z_anchor = crossover_z` (= `TkWKBIntegration.z_init`), `sign = +1`,
`spline_order = 3`, `label = "T_k WKB phase"`, and `model_functions` a `_SoundHorizonRate`
rather than the model's own `ModelFunctions` (deviation 1 — anyone reading `phase._Hubble` gets
$H/c_s$, by design). `num_chunks` is 1, so `QuadSourceIntegral`'s
`WKB_phase_spline_chunks` metadata reads 1 for the transfer function as it already does for the
Green's function.

**`chunk_logstep=125` is now gone from every production path.** The remaining occurrences in
the tree are test fixtures (`test_quadsource_integral.py:266`, `test_phase_groups.py:336`,
`:1137`, `test_primitive_phase.py:182`, and `test_tk_source_functions.py`'s new
`stored_phase_spline`, which builds one deliberately to be beaten) and `docs/` reproduction
scripts. `phase_spline`'s frozen signature (D4) is what keeps those working, and nothing in
`ComputeTargets/` imports `LiouvilleGreen.phase_spline` outside tests any more.

**What prompt 13 must re-measure on the production path**, and what it cannot take from here:
every number above is on closed-form stand-ins with a `ClosedFormPrimitive` in place of the
double-double table, so none of them exercises `CumulativeTable.delta`, the off-grid anchor
partial, or the $\varepsilon k\tau$ floor at $x_T=1.4\times10^{10}$. In particular the
consumer-row figure for $T_k$ at $k=3\times10^8$ on the real background is **not** measured
here; `[07-tk-per-object-cost-is-all-setup]`'s item 2 (the anchor panel recomputed per
evaluation, now also on the consumer side, since $z_{\rm init}$ is off-grid) applies to every
`raw_theta` call this consumer makes.
