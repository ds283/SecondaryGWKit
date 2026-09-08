# Log 07 — Phase-group decomposition of the source integrand (A4, part 1)

**Prompt:** prompts/source-remediation/07-phase-group-algebra.md
**Commit:** *(this commit)* — "Add the phase-group decomposition of the source integrand"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Fable 5.1
**Date:** 2026-09-08
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

- **new `ComputeTargets/phase_groups.py`** (575 lines, ~150 of them the module docstring, which
  fixes the definitions of prompt §2 and states the algebra of §3 in full). Pure functions; no
  Ray, no datastore, no integration. Public surface:

  | name | kind | meaning |
  |---|---|---|
  | `PhaseGroup(label, f_sin, f_cos, theta, theta_mod_2pi, theta_deriv, signs)` | frozen dataclass | one `f_sin sin Ψ + f_cos cos Ψ` term; `signs = (s_G, s_q, s_r)` with `0` for a smooth factor, `label` like `"G+q-r"` |
  | `PhaseGroup.levin_theta(include_deriv=True)` | method | the `theta` dict for `adaptive_levin_sincos`; `include_deriv=False` reproduces what `WKB_Levin_integral` passes today |
  | `PhaseGroup.value(log_z)`, `.envelope(log_z)` | methods | the term's value; `sqrt(f_sin²+f_cos²)` |
  | `build_phase_groups(regime, *, Gk, Tq, Tr, model_functions, w_background)` | function | the decomposition for `regime = (G_osc, q_osc, r_osc)`; 1, 2 or 4 groups; raises `ValueError` for `(False, False, False)` |
  | `evaluate_sum(groups, log_z)` | function | `Σ [f_sin sin(Ψ mod 2π) + f_cos cos(Ψ mod 2π)]` — tests and boundary checks only |
  | `evaluate_envelope(groups, log_z)` | function | `Σ sqrt(f_sin²+f_cos²)`, the natural scale for a residual |
  | `group_signs(regime)`, `signs_label(signs)` | functions | the sign tuples a regime produces, in order, and their labels |
  | `source_coefficients`, `both_oscillatory_coefficients`, `one_oscillatory_coefficients`, `smooth_source`, `phase_group_terms` | pure algebra | plain arithmetic on their arguments (no `math` calls) so that `tests/sympy_phase_groups.py` verifies the module's *own* code path on sympy symbols |
  | `INGREDIENT_CACHE_SIZE = 16384` | constant | size of the per-`build_phase_groups` memo of ingredient evaluations |

  Regime → groups, in the order returned (matching `_three_bessel_Levin`'s `phase1…phase4`):
  `(T,T,T)` → `G+q+r, G+q-r, G-q+r, G-q-r`; `(T,T,F)` → `G+q, G-q`; `(T,F,T)` → `G+r, G-r`;
  `(T,F,F)` → `G`; `(F,T,T)` → `q+r, q-r`; `(F,T,F)` → `q`; `(F,F,T)` → `r`.

  The coefficient formulas are exactly prompt §3's (verified, not trusted — see below):
  `c_SS = αM_qM_r + β(−a_qM_r − a_rM_q + a_qa_r)`, `c_SC = β(−b_rM_q + a_qb_r)`,
  `c_CS = β(−b_qM_r + b_qa_r)`, `c_CC = βb_qb_r`; `P± = (c_CC ∓ c_SS)/2`, `Q± = (c_SC ± c_CS)/2`;
  one-oscillatory `c_S = αM_qT_r + β(−a_qT_r − DT_rM_q + a_qDT_r)`, `c_C = βb_q(DT_r − T_r)`;
  `G` smooth multiplies amplitudes by `G/H²`; `G = A_G sin θ_G` splits each group into
  `Ψ = θ_G ± Θ` with `f_sin = A_GP/(2H²)`, `f_cos = ∓A_GQ/(2H²)`, the constant-phase (both-`T`-smooth)
  group re-combining into the single `Ψ = θ_G`, `f_sin = A_G f/H²`, `f_cos = 0`.
  Phases: `theta` = signed sum of `raw_theta`; `theta_mod_2pi` = signed sum of `theta_mod_2pi`
  (in `(−6π, 6π)`, not re-reduced); `theta_deriv` = signed sum of `omega·(1+z')` for `T` factors
  and `phase.theta_deriv(..., log_derivative=True)` for `G`.
- **`ComputeTargets/__init__.py`**: exports `PhaseGroup, build_phase_groups, evaluate_sum,
  evaluate_envelope, group_signs`, inserted after the `QuadSource` block.
- **new `ComputeTargets/tests/sympy_phase_groups.py`** (212 lines). Starts from
  `QuadSource.source_function` called on sympy symbols (float literals rationalised with
  `nsimplify`), substitutes `T_i = M_i sin θ_i`, `dT_i/dz = M_i(dlnM_i/dz) sin θ_i + M_iω_i cos θ_i`,
  and checks (1) `smooth_source` ≡ `source_function`; (2) `D[M sin θ] = aS + bC`; (3) each of
  `c_SS, c_SC, c_CS, c_CC, c_S, c_C` implied by the module's coefficient functions against the
  coefficient sympy extracts from the expanded kernel, plus "no stray monomials"; (4) for all
  seven regimes, `Σ f_sin sin Ψ + f_cos cos Ψ − G f ≡ 0` using the module's own
  `phase_group_terms`. Residuals reduced with `cancel(expand(expand_trig(·)))`; non-zero exit on
  failure.
- **new `ComputeTargets/tests/test_phase_groups.py`** (1452 lines, 18 tests, 3.5 s). Imports
  `Fixture, FakeModel, FakeWKBValue, FakeTkWKB, log_grid` and the grid constants from
  `test_tk_source_functions.py`. Adds: exact scipy-based stand-ins `ExactTk`/`ExactGk` (`m =
  sqrt(J²+Y²)`, `ϑ = atan2(J, −Y)` unwrapped, `dϑ/dx = 2/(πxm²)`, `d ln m/dx` from the
  recurrences — valid at every `z`); a realistic `BesselPhaseGk` whose phase is a real
  `phase_spline` (`chunk_logstep=125`, `increasing=False`) through `bessel_phase(0.5+b)` samples;
  `matched_LG_functions(fixture)`, an LG `TkSourceFunctions` matched at the hand-over exactly as
  `TkWKBIntegration.store()` does (`:436-462`); and a `Case` per `w` holding regime ranges on the
  realistic fixtures (`q = 1e4`, `r = 2e4`, `k = 1.2e4`).

Nothing in `QuadSourceIntegral.py`, `QuadSource.py`, `TkSourceFunctions.py`, `AdaptiveLevin/`,
`LiouvilleGreen/` or `Datastore/` was touched.

## Deviations from the prompt

### 1. Oracle 2's 1e-8 is met with exact Bessel stand-ins, not with `bessel_phase` fixtures — STRUCTURALLY REQUIRED

§6 asks for Oracle 2 "with `T` from `bessel_phase(1.5+b)` and `G` from `bessel_phase(0.5+b)`",
the LG form of `G` asserted against `compute_analytic_G` to 1e-10, and `evaluate_sum` against
`compute_analytic_G · source_function(analytic T)/H²` to 1e-8 relative-to-envelope. Neither
threshold is reachable through `bessel_phase`: its phase is accurate to ~`x·1e-8`
(`docs/lg-phase-and-handover-followup-2026-09.md` §2.4; log 05 measured `T_WKB` vs scipy
saturating at 2.0e-6 of envelope), and — the larger effect, found here — the *derivative*
pieces a `TkSourceFunctions` supplies are the LG closed forms `ω = sqrt(Tk_omegaEff_sq)` and the
R23/R24 `d ln M/dz`, which differ from the exact `dθ/dz` and `d ln M/dz` by the LG truncation
(`O(x⁻⁴)` relative; log 05 deviation 5 measured 8.5e-6 at `w=1/3`, 7.0e-5 at `w=0.2` in
`d ln M/dz` at the hand-over). Those pieces dominate `f` through `DT_qDT_r`, so on the "exact"
fixture the residual against scipy is grid-independent: **7.0e-6 (`w=1/3`), 1.38e-4 (`w=0.2`)**
at 100/decade and 6.3e-6, 1.08e-4 at 300/decade. Replacing only `ω` and `d ln M/dz` by their
exact values (keeping the `bessel_phase` amplitude and re-splined phase) drops these to 1.5e-6
and 1.4e-6 — `bessel_phase`'s own floor — which pins the attribution
(`test_LG_truncation_is_the_realistic_floor`).

Shipped: (a) the 1e-10 and 1e-8 assertions run on stand-ins whose `m, ϑ` are `bessel_phase`'s
*definitions* (`bessel_phase.py:92-99`, `J = m sin ϑ`) evaluated exactly from scipy rather than
through its ODE and splines — measured 6.4e-13 / 3.0e-13 for the LG form of `G`, and ≤ 3.4e-14
in all seven regimes for `evaluate_sum`; (b) the `bessel_phase`-and-`phase_spline` fixtures are
tested at **5e-4** with the measured values printed, together with the `x_q > 100` subset
(3.0e-6, 6.5e-6) showing the floor is a hand-over effect. The realistic floor is the number
prompt 08 needs; the exact one shows the module contributes nothing to it.

### 2. Boundary-consistency threshold — STRUCTURALLY REQUIRED

§6 asks for agreement "to the spline's accuracy (~1e-6 at 3 e-folds sub-horizon, prompt 06 §3.4
gives the number)". I may not read prompt 06; the board's `[06-source-spline-residual-vs-handover]`
gives 4.5e-4 of envelope for the `f` spline at the hand-over and log 05 gives 2.7e-4 for the
`dT/dz` spline between grid points, so 1e-6 was never the right number for `f`. Measured, `T_r`
in its LG form just below its hand-over (`x_r` 19 → 28) against `source_function` with `T_r` from
a numeric spline that extends 0.4 e-folds further in, `T_q` numeric, `G` exact:

| fixture for `T_r` | `w` | `G` | at the numeric spline's nodes | at log-midpoints |
|---|---|---|---|---|
| exact (`M`, `θ` exact; `ω`, `dlnM/dz` LG) | 1/3 | smooth / osc | 8.7e-06 / 7.8e-06 | 1.03e-03 / 8.3e-04 |
| exact | 0.2 | smooth / osc | 1.51e-04 / 1.07e-04 | 7.2e-04 / 7.0e-04 |
| matched LG (production representation) | 1/3 | smooth / osc | 4.1e-05 / 2.4e-05 | 1.07e-03 / 8.6e-04 |
| matched LG | 0.2 | smooth / osc | 1.02e-03 / 6.6e-04 | 1.06e-03 / 8.3e-04 |

At the nodes the numeric side is exact, so the residual is the LG side's derivative truncation
(deviation 1) — and, for the matched-LG fixture, the truncation of the representation itself,
consistent with audit TK-8(e) (3.7e-4–5.6e-3 of envelope in `T` at 3 e-folds). Between nodes the
numeric `dT/dz` spline's fit error is added. Shipped thresholds: 5e-4 (nodes) and 2e-3
(midpoints) on the exact fixture; 5e-2 on the matched-LG fixture, which is documentation for
prompt 08 rather than a test of this module.

### 3. The 1e-10 phase-composition assertion is on exact-remainder constituents — STRUCTURALLY REQUIRED

§6 asks that the composed `theta_mod_2pi` agree with `Ψ mod 2π` to 1e-10 at `|Ψ| ~ 1e6` "for a
synthetic pair of `phase_spline`s". With real `phase_spline` objects holding *exact* `(div, mod)`
samples of quadratic phases (cubic-exact, so no fit error), the remainder route gives
**3.6e-9 rad at 2.2e5 rad and 3.8e-8 rad at 2.2e6 rad**, the same as reducing the raw sum
(3.7e-9, 4.3e-8) and the same as each constituent's own `raw_theta` error. The cause is
`phase_spline`'s chunking: `chunk_logstep=125` is *geometric* in the cycle count
(`phase_spline.py:460-476`: chunks `[0,126]`, `[94,11751]`, `[8813, 1.1e6]`, …), so the top chunk
spans essentially the whole range and its rebased phase is as large as the raw one; the spline
then rounds at ~20 ε|θ|. That is `LiouvilleGreen/` (README §5 item 8), not this module. Shipped:
the 1e-10 assertion runs on constituents whose raw, remainder and derivative are each correctly
rounded from `mpmath` — measured **1.8e-15 rad** for the remainder route against **3.8e-10 rad**
for the reduce-the-raw-sum route at 2.2e6 rad, i.e. exactly the ε|Ψ| loss the prompt describes
(its "~1e-10·1e6" is a slip; the loss is ~ε·|Ψ| ≈ 1e-10, not 1e-4). The `phase_spline` variant is
kept as a measurement with bounds 2× the constituents' own error and 1e-6 rad absolute.
`theta_deriv` agrees with a centred finite difference to 2.3e-12 (exact) and 1.7e-10 (spline).

### 4. Extra public helpers — IMPLEMENTATION CHOICE

§5 names `build_phase_groups`, `evaluate_sum` and the `PhaseGroup` dataclass. Added:
`PhaseGroup.levin_theta()`, `.value()`, `.envelope()`, `evaluate_envelope`, `group_signs`,
`signs_label`, and the five pure algebra functions as public names. Reasons: `levin_theta()` is
the one-line adapter prompt 08 would otherwise write four times, with the `include_deriv` switch
being exactly its §4 decision; `evaluate_envelope` is the only sensible normalisation for a
residual of an oscillatory sum and prompt 08's boundary check needs it; the algebra functions are
public so that the sympy script verifies the *module's* code path rather than a re-transcription
(the alternative — deriving in sympy and re-typing the formulas into the module — is precisely the
failure mode the prompt warns about). None of them integrates or reads a datastore.

### 5. Ingredient memoisation — IMPLEMENTATION CHOICE

Each group's `f_sin`/`f_cos` reads one entry of a shared per-abscissa evaluation of every
ingredient (`H`, `w_0`, `G` or `A_G`, `(M, a, b)` or `(T, DT)` for each `T`), memoised with
`functools.lru_cache(16384)` keyed on `log_z`. The Levin driver samples every group's `f_sin` and
`f_cos` at the same Chebyshev nodes, so without this a node would cost `2 × (groups)` evaluations
of the `TkSourceFunctions` accessors (each `M` is one `Hubble`, one `wPerturbations` and several
spline reads — log 05 observation 3) instead of one. Alternatives: no cache (8× the cost in the
four-group regime), or a last-value cache (defeated by the driver sampling `f_sin` over a whole
grid before `f_cos`). Hashing is on the float value, so `numpy.float64` abscissae hit the same
entries; an array argument raises `TypeError` inside the cache, which the driver's vectorisation
probe treats as "does not vectorize" (`levin_quadrature.py:846-886`). Tested with numpy scalars.

### 6. Smooth-`G` input accepts a callable or an object exposing `numeric_Gk` — IMPLEMENTATION CHOICE

§5's phrasing ("for a smooth `Gk`, its `numeric_Gk`") can be read as passing the `ZSplineWrapper`
or the `GkSourceFunctions`. Both are accepted: a callable is used as `G(x, z_is_log=True)`,
otherwise `.numeric_Gk` is. Smooth `T` factors are always the `TkSourceFunctions` object itself
(its `T`/`dT_dz` are used), since there is no separate object to pass. Protocol violations raise
`TypeError` at build time, a missing WKB representation (`sin_amplitude is None`) `ValueError`.

### 7. The `G`-only integrand equals the current `WKB_Levin_integral` one to rounding, not bitwise — IMPLEMENTATION CHOICE

§3 says the `G`-only group "must reproduce [the current integrand] exactly". `f` for two smooth
factors is computed by `smooth_source` (the `α, β` form with `DT = (1+z)dT/dz`) rather than by
calling `source_function`, so that the one pure `phase_group_terms` is the whole algebra and the
sympy script covers it. The two are the same function (sympy residual zero) evaluated with a
different association order; measured agreement 2.5e-16 relative to the kernel's term envelope.
Bitwise equality was judged not worth a second code path.

### 8. Test configuration — IMPLEMENTATION CHOICE

`q = 1e4 < r = 2e4` (so `T_r` hands over 0.69 e-folds above `T_q` and the "one `T` oscillatory"
row is populated), `k = 1.2e4` (a triangle, though nothing here needs one), `z_resp` 5 % below
the lowest LG sample. With `q < r` the "`T_q` oscillatory, `T_r` numeric" regime is empty on the
realistic fixtures, so it is tested on the exact stand-ins only (Oracle 1 and Oracle 2). A `q = r`
test passes the same object twice and checks that the `q−r` group has identically zero phase and
still sums correctly. `matched_LG_functions` (~45 lines) mirrors `TkWKBIntegration.store()`'s
matching because `Fixture.LG_functions()` starts its phase at zero with an arbitrary `sin_coeff`
and is therefore a *different solution*, unusable for a seam comparison (it gave 0.999 of
envelope before this was understood).

## Verification performed

All from the repository root, `PYTHONPATH=.`, `./venv/bin/python`:

- `python -m unittest discover -s ComputeTargets/tests -t .`: **Ran 45 tests in 4.836 s — OK**
  (27 pre-existing, 18 new).
- `python ComputeTargets/tests/sympy_phase_groups.py`: 21 residuals, every one
  `residual = zero`; exit 0. The first run reported four non-zero residuals of the form
  `4X/(6w+6) − 2X/(3w+3)`; that was `expand` not combining denominators, fixed in the script's
  reduction with `cancel`, and every residual is then identically zero — **the prompt's §3
  formulas are correct as written**; no formula was changed.
- `./venv/bin/black` on the three new files.

Measured maxima (relative to `max(|reference|, Σ envelopes, Σ|kernel terms|)`, 200 log-uniform
points per cell, seed 20260908):

**Oracle 1** — `evaluate_sum` vs `G·source_function(T, T′ reconstructed from the same accessors)/H²`
(algebra and sign bookkeeping only; threshold 1e-12):

| regime | exact stand-ins `w=1/3` | `w=0.2` | realistic fixtures `w=1/3` | `w=0.2` |
|---|---|---|---|---|
| `G, q, r` | 4.3e-16 | 4.8e-16 | 4.0e-16 | 4.5e-16 |
| `G, q` | 5.3e-16 | 5.5e-16 | — (unpopulated) | — |
| `G, r` | 4.8e-16 | 5.5e-16 | 5.0e-16 | 4.1e-16 |
| `G` | 3.5e-16 | 4.0e-16 | 3.5e-16 | 3.3e-16 |
| `q, r` | 6.7e-16 | 4.9e-16 | 4.2e-16 | 4.6e-16 |
| `q` | 4.4e-16 | 5.7e-16 | — | — |
| `r` | 5.3e-16 | 4.3e-16 | 3.9e-16 | 4.6e-16 |

`q = r` degenerate case (`G,q,r` and `q,r`): passes at 1e-12 with the `q−r` phase identically 0.

**Oracle 2** — vs `compute_analytic_G · source_function(analytic T)/H²` (scipy only):

| regime | exact stand-ins `w=1/3` | `w=0.2` |
|---|---|---|
| `G, q, r` | 4.2e-15 | 8.2e-15 |
| `G, q` | 4.2e-15 | 1.5e-14 |
| `G, r` | 1.1e-15 | 1.0e-15 |
| `G` | 6.0e-16 | 2.3e-15 |
| `q, r` | 4.8e-15 | 1.2e-14 |
| `q` | 7.4e-15 | 3.4e-14 |
| `r` | 1.0e-15 | 7.5e-16 |

LG form of `G` vs `compute_analytic_G`: 6.4e-13 (`w=1/3`), 3.0e-13 (`w=0.2`) (threshold 1e-10).

Realistic fixtures (`TkSourceFunctions` from prompt 05's exact fixture + `phase_spline`-backed
`G`), all-oscillatory regime, `x_q ∈ [19, 1000]`:

| `w` | 100/decade | of which `x_q > 100` | 300/decade | of which `x_q > 100` | with exact `ω`, `dlnM/dz` |
|---|---|---|---|---|---|
| 1/3 | **7.0e-06** | 3.0e-06 | 6.3e-06 | 1.7e-06 | 1.5e-06 |
| 0.2 | **1.38e-04** | 6.5e-06 | 1.08e-04 | 5.7e-06 | 1.4e-06 |

(The `G` fixture contributes nothing visible: "T fixtures only" columns agree to <1 %.)

**Structure**: `(F,F,F)` raises `ValueError`; group counts `2^(n−1)`, labels and signs as tabled;
`G`-only `f_cos ≡ 0` and `f_sin` vs `A_G f/H²` 2.5e-16 / 2.3e-16; `theta` for the `G`-only group is
`raw_theta` of `G` bitwise; `theta_deriv` uses `omega·(1+z)` for `T` (a deliberately wrong `omega`
propagates) and the spline log-derivative for `G`.

**Phase composition** (`|Ψ|` up to 2.2e6 rad): see deviation 3 — exact constituents 1.8e-15 rad
(remainder route) vs 3.8e-10 rad (raw route); `phase_spline` constituents 3.8e-8 rad either route;
`theta_deriv` vs finite difference 2.3e-12 / 1.3e-10.

**Boundary consistency**: table in deviation 2.

Not verified here, and needing a pipeline run: that a real `GkSourceFunctions` and real
`TkSourceFunctions` built from datastore rows satisfy the protocol (they do by inspection of
`GkSourcePolicyData.py:19-32, 640-700` and `TkSourceFunctions.py`, and the fixtures use the same
attribute names), and any cost measurement — prompt 08 owns both. This module has not been
exercised against a live pipeline.

## Observations not acted on

1. **`phase_spline`'s chunking does not protect precision beyond the second chunk** (deviation 3).
   With `chunk_logstep=125` the chunk boundaries in cycles are `0, 126, 11751, 1.1e6, …`, so for
   `|θ| ≳ 7e4` rad the rebased spline values are as large as `raw_theta` and `theta_mod_2pi` is
   no more accurate than `fmod(raw_theta, 2π)`: 3.8e-8 rad at 2.2e6 rad on exact data. Harmless
   for `sin`/`cos` at any accuracy this pipeline reaches (the followup document's fit error is
   ~1e-2 rad at production `x`), but the reconciliation document §3.1's rationale for summing
   remainders is weaker than stated. A linear `chunk_step` would restore it. `LiouvilleGreen/` is
   out of scope; the module sums remainders as the prompt specifies regardless.
2. **The LG closed forms for `ω` and `d ln M/dz` are the accuracy floor of the oscillatory
   integrand near the hand-over** (deviation 1): 7e-6 (`w=1/3`) to 1.4e-4 (`w=0.2`) of envelope
   at `x ≈ 20`, falling as `x⁻⁴`. This is the representation `TkSourceFunctions` was specified to
   supply (log 05 §4, "closed forms, no splined products") and is not a defect in it; a
   hand-over deeper inside the horizon (the followup document §1.4) would reduce it. Recorded on
   the board for prompts 08 and 12.
3. **Every callable this module produces is scalar-only** (it calls `ZSplineWrapper` and
   `phase_spline`, which are scalar-only — levin-refactor log 08), so the Levin driver will sample
   them in a Python loop. If prompt 08's cost measurement finds this matters, vectorising would
   have to start in `LiouvilleGreen/` and `spline_wrappers.py`, not here.
4. **`source_function`'s `None`-propagating signature** (returns `None` if any `T` is `None`) has no
   analogue here: a `None` from a factor accessor would raise inside the arithmetic. Prompt 08's
   region partition should never produce one.
5. **`w_background` is a separate argument although `model_functions.wBackground` exists**, as §5
   specifies; the tests pass `model.functions.wBackground`. Kept as specified so a caller can
   substitute a fixed-`w` kernel for an oracle comparison without a stand-in model.

## State handed to the next prompt

Prompt 08 programs against the following, verbatim.

```python
from ComputeTargets import PhaseGroup, build_phase_groups, evaluate_sum, evaluate_envelope, group_signs

groups = build_phase_groups(
    (G_osc, q_osc, r_osc),          # regime; (False, False, False) raises ValueError
    Gk=Gk_f,                        # GkSourceFunctions if G_osc, else Gk_f.numeric_Gk (or Gk_f itself)
    Tq=Tq_f, Tr=Tr_f,               # TkSourceFunctions; WKB accessors used if oscillatory, T/dT_dz if not
    model_functions=model.functions,
    w_background=model.functions.wBackground,
)
for g in groups:                    # 1, 2 or 4 groups; order = group_signs(regime)
    g.label                         # "G+q-r" etc.
    g.signs                         # (s_G, s_q, s_r), 0 for a smooth factor
    g.f_sin(log_z), g.f_cos(log_z)  # amplitudes, 1/H^2 included, (1+z_resp) NOT included
    g.theta(log_z)                  # signed sum of raw_theta
    g.theta_mod_2pi(log_z)          # signed sum of remainders, in (-6 pi, 6 pi)
    g.theta_deriv(log_z)            # d Psi / d log(1+z'), closed-form omega (1+z) for T, spline log-derivative for G
    g.levin_theta(include_deriv=True)   # {"theta", "theta_mod_2pi"[, "theta_deriv"]} for adaptive_levin_sincos
    adaptive_levin_sincos(x_span, f=[g.f_sin, g.f_cos], theta=g.levin_theta(), ...)

evaluate_sum(groups, log_z)         # pointwise sum, for the region-boundary consistency check only
evaluate_envelope(groups, log_z)    # sum of sqrt(f_sin^2 + f_cos^2): normalise residuals by this
```

1. **Every callable takes `log(1+z')`** and is scalar-only. Range checking is the factor
   objects' own: `TkSourceFunctions` raises outside `numeric_region`/`WKB_region`,
   `phase_spline` outside its samples, `ZSplineWrapper` 1 % outside its range. Partition on
   `crossover_z` and clamp nodes to the regions (board §5 notes 6–7); this module does not clamp.
2. **How a smooth factor is passed.** A smooth `T` is the *same* `TkSourceFunctions` object; the
   regime flag selects `T`/`dT_dz` instead of `M`/`dlnM_dz`/`omega`/`phase`. A smooth `G` is
   `Gk_f.numeric_Gk` (or `Gk_f`; both work). An oscillatory `G` must have `sin_amplitude` and
   `phase` non-`None`, else `ValueError`. Wrong shapes raise `TypeError` at build time, before
   any integration starts.
3. **The `G`-only regime is the current `WKB_Levin_integral` integrand** (`f_sin = A_G f/H²`,
   `f_cos ≡ 0`, `theta = θ_G` bitwise), to 2.5e-16 in `f_sin`. Passing `include_deriv=False`
   reproduces exactly the `theta` dict that function passes today. `theta_deriv` is provided
   everywhere; whether to pass it is prompt 08's call (the module docstring points at the
   `:1128-1145` comment).
4. **Group order and the resonance.** Groups come sorted by `signs` descending:
   `G+q+r, G+q-r, G-q+r, G-q-r`. With the code's conventions (`θ_G` decreasing with `z'`, `θ_q, θ_r`
   increasing — board §5 note 5, log 05 deviation 2) the slowly varying group is `G+q+r`; check
   `g.theta_deriv` rather than assuming which label is stationary. For `q = r` (same object
   passed twice) the `q−r` phase and its derivative are identically zero and that group is a
   smooth integrand; the driver's total-variation gate will route it to Clenshaw–Curtis.
5. **Accuracy floors for the error budget** (all relative to the envelope of `G f/H²`):
   - the decomposition itself: ≤ 7e-16 (Oracle 1) — nothing to budget;
   - exact ingredients end to end: ≤ 3.4e-14 (Oracle 2, exact stand-ins);
   - **realistic `TkSourceFunctions` in the all-oscillatory regime: 7.0e-6 (`w=1/3`), 1.4e-4
     (`w=0.2`) at the production grid, dominated by `x ≈ 20` just below the hand-over and falling
     to 3e-6 / 6.5e-6 for `x > 100`; grid-independent** (LG truncation of `ω`, `dlnM/dz`; board
     issue `[07-lg-derivative-truncation-at-handover]`). Use ~5e-4 as the acceptance threshold for
     an exact-fixture comparison in the oscillatory regions, not 1e-8;
   - the seam between the numeric and LG representations of one factor (`x_r` 19 → 28): 8.7e-6 /
     1.5e-4 at grid nodes and ~1e-3 between nodes on the exact fixture; with the production LG
     representation (matched at the hand-over as `TkWKBIntegration.store()` does) 4e-5 / 1e-3 at
     nodes. A region-boundary consistency check at the hand-over should therefore use ~2e-3, and a
     mismatch of that size is the representation, not a bug (audit TK-8(e); followup §4).
6. **Composed remainders are accurate to the constituents' own `theta_mod_2pi`**: 1.8e-15 rad on
   exact constituents; through `phase_spline` ~2e-14·|θ| (3.8e-8 rad at 2e6 rad) — observation 1.
   Fine for `sin`/`cos`; not a reason to re-reduce.
7. **Test fixtures reusable by prompt 08**: `ComputeTargets/tests/test_phase_groups.py` exposes
   `ExactTk(k, w, model)`, `ExactGk(k, w, z_resp, model)` (exact stand-ins valid at every `z`, with
   `numeric_Gk`, `sin_amplitude`, `phase`), `BesselPhaseGk(k, w, z_resp, model, z_grid)`,
   `matched_LG_functions(fixture)`, `Case.get(w[, samples])` (with `.both_WKB`, `.r_only`,
   `.both_numeric` regime ranges, `.scipy_oracle(log_z)`, `.term_envelope(...)`), and
   `max_relative_residual(groups, reference, log_z_values, extra_scale=None)`. The scipy oracle is
   a *pointwise integrand*; an integral oracle is `analytic_integral` / spec 04 R14 as in `QI_02`.
8. **Cost**: not measured (no driver here). Each abscissa costs one memoised ingredient
   evaluation (one `Hubble`, one `wBackground`, and per oscillatory `T` one `M`, `dlnM_dz`,
   `omega` — see log 05 observation 3 for what those cost) plus, per group, three phase-spline
   reads for `theta`, `theta_mod_2pi` and (for `G`) `theta_deriv`.
9. **No schema, no persistence, no `Datastore` change**; `QuadSourcePolicy` untouched.
