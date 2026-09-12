# Reconciling `sigw-resonance-scaffolding.md` against the codebase

**Subject.** The design proposal *"Adaptive scaffolding of the SIGW one-loop integral for a general
time-dependent equation of state"* (`~/Downloads/sigw-resonance-scaffolding.md`), read against the
repository at commit `a47664a`.

**Scope.** Two questions. (1) Which of the proposal's Steps 0–6 already exist in the code, in what
form, and where the concepts diverge. (2) What would be involved in building a Liouville–Green
(LG) phase representation of the *quadratic source integral*, so that it can be used the way the
proposal's Steps 2–3 want.

**Method.** Reading of `ComputeTargets/`, `LiouvilleGreen/`, `AdaptiveLevin/`, `CosmologyModels/`
and `main.py`, plus the transcribed derivation in `docs/spec/` (unsigned-off; the spec→code audit
pass has not been run). Two arithmetic checks were run against the actual `main.py` grid. No
repository files were modified other than this document.

---

## 0. Summary

### 0.1 Verdict

The proposal was written without sight of the code and assumes considerably less machinery than
exists. Steps 0 and 4 are substantially **already built, and in a stronger form than the proposal
describes** — the code does not have to reconstruct phases from `c_s(η)`, because it solves LG
phase functions directly for both the transfer function and the tensor Green's function, and it
already runs adaptive Levin quadrature against them. Steps 1, 2, 3 and 5 (outer) are **absent**,
and — more importantly — the current object graph has no place to put them, because the outer
`(s,d)` integral does not exist yet.

| Proposal step | Status in code | Where |
|---|---|---|
| **0** Background tables `a, H, w, c_s²` | **done**, on a shared `z` grid, persisted | `ComputeTargets/BackgroundModel.py`; `ModelFunctions` namedtuple |
| **0** Mode tables as amplitude + phase | **done** for both scalar and tensor sectors | `TkWKBIntegration`, `GkWKBIntegration`/`GkSource`, `LiouvilleGreen/phase_spline.py` |
| **0** `c_s²` independent of `w` | **partly** — `wBackground`/`wPerturbations` are already separate, but `c_s² ≡ wPerturbations` is hardwired | `CosmologyModels/base.py`; `ComputeTargets/WKB_Tk.py:17` (`A = w * k_over_H_2`) |
| **0** Wronskian check on the tensor pair | **not applicable as described** — the code builds `Gr_k` directly, not two homogeneous solutions | `GkNumericIntegration`, `GkSource` |
| **1** Resonance scan | **absent** — but every ingredient is a closed-form function of `z` already | would consume `Tk_omegaEff_sq`, `Gk_omegaEff_sq` |
| **2** Uniform-asymptotic surrogate kernel | **absent**, and there is no kernel object to be a surrogate *for* | — |
| **3** Equidistributed, per-`k` `(s,d)` grid | **absent, and contradicted**: one fixed `(q,r)` grid is shared by every `k` | `main.py:2489` |
| **4** Levin/Filon inner time integral | **done for the Green's-function phase only**; the source is treated as a slowly varying amplitude | `QuadSourceIntegral.WKB_Levin_integral` |
| **4** Stationary-point safety of the Levin rule | **done** (the C2 total-variation gate) | `AdaptiveLevin/levin_quadrature.py:1260`; `docs/adaptive-levin-verification.md` §4.2 |
| **5** Error control, inner | **done** (`abserr`, `converged`, `phase_limited`) | `adaptive_levin_sincos` |
| **5** Error control, outer `(s,d)` | **absent** | `OneLoopIntegral` is a stub |
| **6** Validation ladder | **partly** — a fixed-`w` analytic three-Bessel oracle exists and is computed alongside | `QuadSourceIntegral.analytic_integral` |

### 0.2 The four findings that matter

**F1 — `QuadSource` currently splines an oscillating function.** `QuadSource` is built from
`TkNumericIntegration` only (`ComputeTargets/QuadSource.py:9`), and `_create_functions()` fits a
plain `make_interp_spline` through the *sampled source values* in `log(1+z)`. Sub-horizon, that is
a spline through an oscillation — precisely the failure the proposal's Step 0 warns about ("*
interpolating the oscillating `φ` directly is hopeless once the mode is deep inside the horizon*"),
one level up. The scale of the mismatch: the source is sampled at
`DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z = 100` points per decade in `z`, while this module's own Levin
tuning log records a deep-sub-horizon configuration of **~4.5×10⁶ cycles**
(`QuadSourceIntegral.py:35`, `CHEBYSHEV_ORDER` comment). The planned refactor onto
`TkWKBIntegration` is therefore not a tidy-up; it is the correctness fix that everything else here
depends on.

**F2 — the Levin call is given the wrong phase.** `WKB_Levin_integral` hands
`adaptive_levin_sincos` the phase of the **Green's function alone** (`Gk_f.phase`), with
`Levin_f = sin_amplitude · f / H²`, i.e. with the source `f` in the *amplitude* slot. Once `T_q`
and `T_r` are sub-horizon, `f` oscillates at `θ_q ± θ_r` and is not an amplitude at all. The Levin
rule will still converge — it will just bisect until each region resolves the oscillation it was
told was smooth, losing the entire advantage, and inheriting F1's spline error as its accuracy
floor. This is also exactly the regime in which the resonance lives, so the current code is
least trustworthy precisely where the proposal says the answer is decided.

**F3 — 92% of scheduled `QuadSourceIntegral` work is on triples that are not triangles.**
`main.py:2489` forms `itertools.product(z_response_sample,
combinations_with_replacement(source_k_exit_times, 2), response_k_exit_times)` with no
`|q−r| ≤ k ≤ q+r` filter. On the shipped grid (50 log-spaced `k` from 1e5 to 3e8/Mpc, used for both
source and response) that is 63,750 `(k,q,r)` triples of which **5,133 (8.1%)** close. Worse, the
surviving triples induce an uncontrolled `s = (q+r)/k` sampling: for a mid-grid
`k = 5.9×10⁶/Mpc` only 109 pairs survive, spread over `s ∈ [1.015, 101]`, with **three** nodes
within ±0.05 of the radiation resonance `s = √3`. This is the proposal's §2.3 "*a fixed `(s,d)` mesh
cannot remain correct across decades in `k`*", made concrete — and here it is not even a mesh.

**F4 — in this code `c_s → 0` is guaranteed, and `c_s(z)` is non-monotone.** Both of the
proposal's §5 pitfalls are live, not hypothetical:
- `wPerturbations` for `LambdaCDM` is `(1/3)·Ω_r(1+z)/(Ω_m + Ω_r(1+z))`, which **→ 0** as `z → 0`.
  So `R = 1/c_s` is unbounded below matter–radiation equality and the resonance band is
  `[1/c_s^max, ∞)`, not a compact interval. Whether that matters is decided by the amplitude
  weight, which is the proposal's argument for putting `P_ζ P_ζ W` inside the Step-3 monitor.
- `QCD_EOS.w(T) = 4G_s/(3G) − 1` (Saikawa–Shirai fits) dips below 1/3 at the QCD crossover and
  again at `e⁺e⁻` annihilation. So `dR/dz` has interior zeros, `z_*(s)` is **multi-valued**, and
  `QCD_Cosmology` is exactly the proposal's "interior fold → Airy caustic" row. Contributions from
  the several stationary points must be summed as complex amplitudes, not as `|·|²`.

---

## 1. Step-by-step reconciliation

### 1.1 Step 0 — background and mode tables

**Background.** `BackgroundModel` solves once and tabulates on a single shared `z` grid: `Hubble`,
`tau` (conformal time, as `a₀τ`), `rho`, `wBackground`, `wPerturbations`, `epsilon`, and `z`
derivatives of `w` and `ε` up to second order. It is persisted and handed to workers as a
`ModelProxy`. This is the proposal's `BackgroundTables` interface, already built and already
shared — so the §4 question "*is there an existing time grid shared between background and modes*"
answers **yes** for the background; each mode solver then owns its own `z_sample`, with splines
providing dense output.

**`c_s²` vs `w`.** The proposal asks that these be carried independently. The code is halfway
there: `BaseCosmology` requires **both** `wBackground` (which drives `H` through `ρ`) and
`wPerturbations` (radiation-only fraction, excluding Λ and dust), and they genuinely differ. But
`Tk_omegaEff_sq` writes the gradient term as `A = w · (k/H)²` with `w = wPerturbations`
(`ComputeTargets/WKB_Tk.py:22`), i.e. it assumes `c_s² = w_pert` adiabatically. Making `c_s²`
independent is a small, local change: add `csSquared(z)` (plus its two `z` derivatives) to
`BaseCosmology`/`ModelFunctions`, default it to `wPerturbations`, and use it in `Tk_omegaEff_sq`
and `Tk_d_ln_omegaEff_dz`. Nothing downstream needs to know. The proposal's caveat about
non-adiabatic pressure would then become an explicit, per-model choice rather than an implicit one.

**Modes in amplitude–phase form.** Already done, and this is the code's main asset. Both sectors
solve an LG phase function (`Quadrature/integrators/WKB_phase_function.py`) and store
`(theta_div_2pi, theta_mod_2pi)` per sample point, so the cycle count is carried exactly and the
argument handed to `sin`/`cos` never loses precision to a large phase. `LiouvilleGreen/phase_spline.py`
reassembles this into a chunked, rebased spline with `raw_theta`, `theta_mod_2pi` and `theta_deriv`.
Concretely, after the `deltaTheta` rotation in `TkWKBIntegration.store()` the transfer function is a
**single-phase LG form**

```
T_q(z) = M_q(z) · sin θ_q(z),     M_q(z) = sin_coeff · sqrt( (H_init/H) / ω_q ) · exp(friction)
```

with `cos_coeff ≡ 0`. The Green's function has the same shape in its WKB region, with amplitude
`sin_coeff · sqrt(H_ratio / sqrt(omega_WKB_sq))` assembled in
`GkSourcePolicyData._create_functions()`. So the proposal's `ModeTables` interface exists; `M` and
`θ` are the objects to interpolate in `ln q`, exactly as it recommends.

**The tensor Wronskian check.** Not available as written. The code does not build two
Wronskian-normalised homogeneous solutions; `GkSource` assembles the retarded Green's function
directly — a numeric ODE solve at small `|z_source − z_response|` and a WKB branch further out,
with `GkSourcePolicyData` choosing the crossover. A cheap equivalent check does exist in principle
(the LG amplitude of the tensor mode should satisfy `M² θ' = const` in the region where
`a''/a` is negligible), but nothing computes it.

### 1.2 Step 1 — resonance scan

Absent, and it is the cheapest missing piece: **no new solves are required**, because the two
frequencies are already closed-form functions of `z`:

```python
Gk_omegaEff_sq(model, k, z)   # ComputeTargets/WKB_Gk.py — tensor,  → (k/H)²  sub-horizon
Tk_omegaEff_sq(model, q, z)   # ComputeTargets/WKB_Tk.py — scalar,  → w (q/H)² sub-horizon
```

Two remarks on translating the proposal into the code's variables.

*Use the `ω`'s, not `1/c_s`.* The proposal offers `R(η;q,r)` from the stored `ω`'s as a refinement
"if horizon corrections matter". Here it is the **cheaper** route as well as the more accurate one,
because those functions already exist and the horizon-correction terms `B`, `C` are already in
them. The stationary-phase condition is

```
ω_k^G(z_*) = ω_q^T(z_*) + ω_r^T(z_*)
```

which sub-horizon reduces to `k = c_s(q+r)`, i.e. `s = 1/c_s`, recovering the proposal's §2.1.

*The variable change is harmless.* The code works in `x = log(1+z)`, and `dθ/dz = ω_eff`, so
`dΨ/dx = (1+z)[ω_k^G ∓ ω_q^T ∓ ω_r^T]`. The location `z_*` of the stationary point is therefore
variable-independent, and at that point

```
Ψ''(x)|_{x_*} = (1+z_*)² · (dΨ/dz)'|_{z_*}
```

with no `Ψ'` term surviving. So the proposal's width formulas transfer with a single Jacobian
factor. `Ψ''` itself is available in closed form: `dω/dz = ω · d ln ω/dz`, and both
`Gk_d_ln_omegaEff_dz` and `Tk_d_ln_omegaEff_dz` exist.

*One derivative short.* `Ψ'''` — needed for the Airy/CFU normal form at a fold, and for the
`(c_s''/k²)^{1/3}` width in the proposal's §2.3 table — requires `d²ω/dz²`, hence **third**
`z` derivatives of `w` and `ε`. `BackgroundModel` currently carries `d2_wPerturbations_dz2` and
`d2_epsilon_dz2` and stops there. Extending `_build_derivative` one level is mechanical (it already
differentiates a sampled series to produce the second derivative) but is a schema change to the
`BackgroundModel` value table.

### 1.3 Step 2 — surrogate kernel

Absent, and — this is the substantive point — **there is nothing for it to be a surrogate of.** The
proposal assumes a kernel `K(s,d;k)`, time-averaged and squared. The code's corresponding object is
`QuadSourceIntegral`, which is the *un-squared* time integral at a *specific* `z_response`. Per
`docs/spec/05-one-loop.md` R23+R31 the squaring happens inside the loop integral, and no
oscillation-average over `z_response` is taken anywhere in the notes. So before Step 2 can be
written down, a convention has to be fixed:

> Is `P_h(k)` wanted at a fixed `z_response`, or averaged over the tensor oscillation?

That choice determines what `OneLoopIntegral` is, and it is currently unmade. It also interacts
with F4: at a caustic, `⟨J²⟩` and `J²` differ by more than a factor of two, because the
stationary-point contribution is not in phase with the endpoint contributions.

### 1.4 Step 3 — the grid

Absent and contradicted; see **F3**. The current grid is built once, at the top of `main.py`, as
`combinations_with_replacement` over a single 50-point log-spaced `source_k_array`, and is reused
for every response `k`. It is looped, not vectorised over grid points (the `RayWorkPool` batches
`(z_response, k, q, r)` tuples), so a **ragged per-`k` grid would not break any array shapes** —
which answers the proposal's §4 question favourably. What it *would* break is the object cache:
per-`k` `s` nodes mean per-`k` values of `q` and `r`, and every distinct `q` needs its own
`TkNumericIntegration`, `TkWKBIntegration` and (per pair) `QuadSource`. On the shipped grid that
multiplies the scalar-sector solve count by up to the number of response `k`'s. §3 below is about
how to avoid paying that.

### 1.5 Step 4 — the inner time integral

Largely done, and the proposal underestimates it. `compute_QuadSource_integral` already partitions
`[z_source_max, z_response]` into up to three regions — numeric quadrature, ordinary WKB
quadrature, and WKB Levin — with the boundaries chosen by `GkSourcePolicyData` (`crossover_z`,
`Levin_z`) and a `LEVIN_MIN_2PI_CYCLES = 10` gate. `AdaptiveLevin` is a full Bremer–Chen–Yang
adaptive Levin implementation with a Clenshaw–Curtis fallback.

Critically for this proposal, the **stationary-point hazard is already closed**. The audit finding
C2 (`docs/adaptive-levin-audit-2026-09.md` §0.1) was that the weakly-oscillatory gate tested the
*net* phase change, so a phase with an interior stationary point was handed wholesale to `quad` and
came back 1590% wrong. The fix — gate on total variation
(`AdaptiveLevin/levin_quadrature.py:1260`) — is in, and `docs/adaptive-levin-verification.md` §4.2
records it verified on exactly the sum-and-difference phase-group family that the resonance
produces. So the machinery is safe to point at a resonant phase, at a measured cost of ~3.4× more
evaluations on such a case.

The gap is **F2**: the phase supplied is the Green's function's alone. Fixing that is §3.

### 1.6 Step 5 — error control

Inner: done. `adaptive_levin_sincos` returns an aggregate `abserr`, `converged` and
`phase_limited`, and `LiouvilleGreen/three_bessel_integrals.py::BesselIntegralResult` documents why
the four phase groups' errors must be summed **linearly** rather than in quadrature (they share a
phase construction, so an inaccurate phase drifts them together) and why relative error is amplified
by the cancellation between groups. That reasoning carries over verbatim to the numerical case.
Note its caveat: `abserr` is the quadrature error only and excludes the phase/modulus spline fit
error, measured at ~2e-8 relative on this module's own oracles.

Outer: absent, because there is no outer integral.

### 1.7 Step 6 — the validation ladder

Rungs 1–2 are partly built already, by a route the proposal does not anticipate:
`QuadSourceIntegral.analytic_integral` evaluates the fixed-`w` result of `docs/spec` R31 by Levin
integration of three-Bessel products, using `LiouvilleGreen/bessel_phase.py` for the analytic
phases, and stores it as `analytic_rad` alongside every numeric result. `LiouvilleGreen/tests`
carries seven closed-form oracles (`J000, J110, J220, J222, J231, Y000, Y022`).

Two adjustments to the ladder as written:

- The proposal's `w → 1/3` guard (SIGWfast nudges `w` by 1e-4 to dodge `Γ(1±b)`/`1/sin(πb)` poles)
  is **not needed on this route**. Those poles come from the closed-form general-`w` kernel; the
  code's analytic branch instead integrates `J_{1/2+b} J_{1/2+b} J_{1/2+b}`-type products
  numerically and passes smoothly through `b = 0`. If SIGWfast is ever used as a cross-check
  (rung 2), the guard applies to *that* side only.
- Rung 4 (monochromatic `P_ζ`) is the cheapest real test available and needs no `P_ζ` machinery at
  all: it is a single `(s,d)` evaluation.

### 1.8 The `d` direction

Agreed and easy. At fixed `k`, `(s,d) ↔ (q,r)` is linear — `q,r = k(s ± d)/2` — so a
Gauss–Legendre rule in `d` at each `s` node maps to a pair of wavenumbers with no extra machinery.
The `d → 1` edge singularity the proposal mentions coincides with `r → 0` or `q → 0`, i.e. the IR
end of the source grid, which is also where `T → 1` and the LG representation is *invalid* (the
mode is super-horizon). That region must be served by the numeric branch, not the WKB branch — a
constraint the existing `GkSourcePolicyData` crossover logic already expresses for the Green's
function and which would need an analogue for the source.

---

## 2. The proposal's §4 questions, answered

**Where does the background come from?** A solved model (`BackgroundModel.compute()` integrates for
`τ`) on top of an analytic or tabulated cosmology (`LambdaCDM`, `LambdaCDM_GenericEOS`,
`QCD_Cosmology` with the Saikawa–Shirai `g(T)`, `g_s(T)` fits). Persisted and shared.

**Is `c_s²` available independently of `w`?** Not yet, but the harder half of that split
(`wBackground` vs `wPerturbations`) is already done. See §1.1.

**Are the transfer functions numerical, and stored as fields or as amplitude and phase?** Both.
`TkNumericIntegration` stores the field and its derivative; `TkWKBIntegration` stores amplitude and
phase (with exact cycle counts). `QuadSource` currently consumes only the former — that is F1.

**Shared time grid?** Yes for the background; each mode solver owns its own `z_sample` and exposes
dense output through splines.

**Is the `(s,d)` grid per-`k`?** No — one fixed `(q,r)` grid for all `k`, and not in `(s,d)`
coordinates at all. See F3.

**Vectorised or looped?** Looped over `(z_response, k, q, r)` work items through `RayWorkPool`, with
vectorisation only in the datastore reads. A ragged per-`k` grid is therefore structurally fine.

**Where would `resonance_scan` sit?** It depends only on `BackgroundModel` plus `k, q, r`, so it
belongs next to `WKB_Gk.py`/`WKB_Tk.py` — say `ComputeTargets/resonance_map.py` — and can be
imported by both a grid builder in `main.py` and by `QuadSourceIntegral` itself, which wants to know
whether its integration range contains a stationary point.

---

## 3. An LG representation of the quadratic source integral

This is the substantive design question. It splits into two tiers that are worth keeping separate,
because the first is a refactor of proven machinery and the second is genuinely new.

### 3.1 Tier 1 — put the source's phases into the Levin call

Once `QuadSource` is refactored onto `TkWKBIntegration`, each factor is a single-phase LG form
(§1.1). Write `S_q = sin θ_q`, `C_q = cos θ_q`, and use `D ≡ d/d log(1+z) = (1+z) d/dz`, which is
the variable the integral is already performed in. Then

```
D T_q = (D M_q) S_q + M_q (D θ_q) C_q
```

and the source of `QuadSource.source_function`,

```
f = α T_q T_r + β [ −(D T_q) T_r − (D T_r) T_q + (D T_q)(D T_r) ],
α = (5+3w)/(3(1+w)),  β = 2/(3(1+w)),
```

is a bilinear in `{S_q, C_q} × {S_r, C_r}` with smooth coefficients built from `M`, `DM`, `Dθ`, `w`.
Product-to-sum collapses it to two phase groups,

```
f(z) = P₊ cos Θ₊ + Q₊ sin Θ₊ + P₋ cos Θ₋ + Q₋ sin Θ₋,      Θ± = θ_q ± θ_r
```

with `P±, Q±` non-oscillatory. Multiplying by the Green's function `G = A_G sin θ_G` and the measure
`1/H²` and expanding once more gives **four** total phases

```
Ψ_j = θ_G ± θ_q ± θ_r ,      j = 1…4
```

each carrying its own smooth sin- and cos-amplitude. That is precisely the signature
`adaptive_levin_sincos(x_span, f=[f_sin, f_cos], theta={...})` accepts, so Tier 1 is **four Levin
calls in place of one**.

The code already contains the template. `QuadSourceIntegral._three_bessel_Levin` (lines 461–730)
builds exactly these four sign combinations — `phase1 … phase4`, each with its `_mod_2pi` partner —
for the *analytic* fixed-`w` case, using `bessel_phase()` for the phases and moduli. Tier 1 is that
function with the analytic phase objects swapped for the numerical ones:

| analytic branch (exists) | numerical branch (Tier 1) |
|---|---|
| `phase_data_Gk["phase"]` from `bessel_phase(0.5+b, …)` | `GkPolicy.functions.phase` (`phase_spline`) |
| `phase_data_Tk["phase"]` from `bessel_phase(1.5+b, …)` | `TkWKBIntegration` phase, as a `phase_spline` |
| `mod_Gk(x)`, `mod_Tk(x)` | `Gk_f.sin_amplitude`, `M_q`, `M_r` splines |
| argument `x = k·η`, `q·c_s·η` | argument `log(1+z)`, common to all three |

Two practical notes. First, the numerical case is *simpler* than the analytic one in one respect:
all three phases are functions of the same variable `log(1+z)`, so there is no per-factor argument
rescaling and no `x_cut` bookkeeping. Second, `phase_spline` supports composing the phases exactly:
each contributor carries `(div_2pi, mod_2pi)`, so `Ψ_j` can be assembled with the cycle counts added
as integers and the residues added and re-reduced (`LiouvilleGreen/WKBtools.wrap_theta` already does
this carry). Summing `raw_theta` values instead would throw away the precision the split exists to
protect.

**What Tier 1 buys.** It fixes F1 and F2 together, it makes the evaluation cost independent of how
deeply sub-horizon the modes are, and it makes the resonance an *explicit, locatable object*: the
group `Ψ = θ_G − θ_q − θ_r` is stationary exactly where §1.2's frequency condition holds. The C2
total-variation gate then routes the neighbourhood of `z_*` to Clenshaw–Curtis and bisects it,
so the answer stays correct there without any special-casing. Tier 1 alone therefore gives a
trustworthy evaluator at any single `(k,q,r)` — but it does **not** give anything interpolable.

### 3.2 Tier 1.5 — the resonance map falls out for free

With `Ψ_j` in hand, the proposal's Step 1 is a few lines: bracket `dΨ/dz = 0` on the background
grid (multi-valued in `QCD_Cosmology`, per F4), evaluate `Ψ''` from `d ln ω/dz`, and report the
band edges as the `s` values where `dR/dz = 0` or where the integration range ends. Cost is table
lookups, as the proposal says. The natural return object is its `ResonanceMap`:

```python
ResonanceMap(band=(s_min, s_max),
             stationary_points=[(z_star, Psi_dd, Psi_ddd), …],   # per (k,q,r)
             caustics=[(s_c, width_s, kind)],                    # kind ∈ {fold, endpoint}
             ...)
```

with the caveat from §1.2 that `Ψ'''` needs one more derivative level in `BackgroundModel`. Until
that lands, folds can be detected (`Ψ'' = 0`) and located but not *widened* analytically; a
finite-difference `Ψ'''` off the existing second derivatives would be adequate for grid design,
which is all Step 2/3 use it for.

### 3.3 Tier 2 — carry an amplitude and an endpoint phase, not a float

This is what actually dissolves F3, and it is the direct answer to "*what would be involved in
generating an LG-type representation of the quadratic source integral*".

**Why it is needed.** `J(k,q,r;z_resp)` is itself a rapidly oscillating function of `q` at fixed
`k, r, z_resp` — its phase is the accumulated `θ_q(z_resp)`, which runs to millions of cycles across
the `q` grid. So interpolating `J` in `ln q` is hopeless for exactly the reason the proposal gives
for not interpolating `φ`. But if `J` is written in LG form,

```
J(k,q,r;z_resp) = Σ_j  𝒜_j(k,q,r) · sin( Ψ_j(z_resp) + φ_j(k,q,r) )
```

then the slow content is in `(𝒜_j, φ_j)` and the fast content is in `Ψ_j`, which can be evaluated
**exactly** at any `(q,r)` from the existing phase splines. Interpolate the former; never
interpolate the latter. A coarse `(ln q, ln r)` grid of amplitudes then supports evaluation of `J`
at arbitrary `(q,r)` — hence a per-`k` `(s,d)` grid at no additional solve cost. That is the whole
point.

**Where the split comes from.** Two routes, and they are complementary rather than alternatives.

*(a) From the Levin solution itself.* The Levin rule computes `∫ f e^{iΨ}` as a boundary term
`[p e^{iΨ}]` where `p` solves `p' + iΨ' p = f`. So `adaptive_levin_sincos` is **already** producing
an (amplitude, phase) split — it just sums the regions and returns a float. Exposing the outermost
endpoint values would give `𝒜_j`, `φ_j` at the `z_response` end essentially free. The honest
caveat: with adaptive subdivision the total is `Σ_regions [p_i e^{iΨ}]_{a_i}^{b_i}` and `p_i` is not
continuous across regions, so the interior terms do not telescope exactly. For a strongly
oscillatory, non-stationary integrand they are asymptotically small, and route (a) then yields a
*surrogate* — which is exactly what the proposal's Step 2 wants and is happy with. It should not be
used for the final answer.

*(b) From uniform asymptotics.* Per phase group, the standard decomposition is
endpoint (Fresnel) contributions plus a stationary-point (Airy, uniform/CFU) contribution. Its
inputs are `Ψ_j`, `Ψ_j'`, `Ψ_j''`, `Ψ_j'''` and the amplitude, all at `z_*` and at the endpoints —
i.e. exactly the `ResonanceMap` of Tier 1.5. This is the proposal's Step 2 kernel `K̃`, and it is
the route that stays finite *at* the caustic.

**What breaks, and why (b) is not optional.** A single (amplitude × single phase) form is precisely
what fails at a caustic — that is the definition. So `𝒜_j` for the resonant group is **not** smooth
in `(ln q, ln r)` across the band: it has an Airy-type transition of width `Δs/s ~ (c_s''/k²)^{1/3}`.
Interpolating the raw `𝒜_j` across the band would smear the caustic away. The fix is to interpolate
the **CFU coefficients** — the two slowly varying Airy multipliers and the argument `ζ` — which
*are* smooth through the fold, rather than the amplitude they multiply. This is the proposal's §5
"coalescing stationary points" pitfall, re-expressed one level up, and it is the single hardest part
of the whole scheme.

A defensible staging that avoids that difficulty on the first pass:

1. Interpolate `(𝒜_j, φ_j)` only **outside** the band (where a single non-stationary form is valid
   and smooth).
2. Inside the band — a small `s` interval, located exactly by Tier 1.5 — evaluate `J` directly with
   Tier 1's Levin calls at every quadrature node.

That gives a per-`k` grid with correct resolution of the feature, at a solve cost proportional to
the number of *in-band* nodes rather than to the whole grid, and it defers all uniform-asymptotic
machinery. It is also directly checkable: refine the in-band nodes until the local contribution
stabilises, which is the proposal's Step 5.

### 3.4 What this means for the object graph

The user's note that `QuadSource`/`QuadSourceIntegral` can be restructured is well placed; Tier 2
requires it.

- **`QuadSource`** becomes a *phase-group* object rather than a sampled-value object: for the WKB
  region it stores the smooth amplitudes `P±, Q±` (or, more economically, `M_q`, `M_r` and their
  log-derivatives, letting the consumer assemble `P±, Q±`) plus references to the two phase splines.
  For the region where either mode is super-horizon it keeps the present sampled form, since the LG
  representation is invalid there. That mirrors the numeric/WKB/mixed split `GkSourcePolicyData`
  already performs for the Green's function, and suggests a `QuadSourcePolicyData` alongside it —
  note `MetadataConcepts/QuadSourcePolicy.py` already exists with a `Levin_threshold` and a
  `numeric_policy`, and is threaded through `main.py`, but nothing consumes it yet.

- **`QuadSourceIntegral`** returns, per `(k, q, r, z_response)`, four `(𝒜_j, φ_j)` pairs plus the
  scalar total, instead of a single `value`. That is a schema change in
  `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` — roughly 8–12 extra float columns, plus
  the existing `LevinData` per group. The phase `Ψ_j(z_resp)` need not be stored, since it is
  reconstructible from the phase splines; storing `(div_2pi, mod_2pi)` anyway makes each row
  self-contained and is cheap.

- **`OneLoopIntegral`** (currently a stub with `compute()` unimplemented, and with an inverted
  guard at `OneLoopIntegral.py:104` — `if self._value is None: raise "…already been computed"`,
  which should be `is not None` — that will need
  fixing regardless) becomes the owner of the per-`k` grid. It consumes a *coarse* amplitude grid
  plus a `ResonanceMap`, builds its own `s` nodes by the proposal's Step 3, evaluates `J` by
  amplitude interpolation outside the band and by direct Levin inside it, squares, weights by
  `P_ζ(uk)P_ζ(vk)Q_s²`, and integrates. The triangle filter (F3) belongs here too.

---

## 4. Risks and open questions

1. **The `z_response` convention (§1.3).** Fixed-time or oscillation-averaged `P_h`? Unmade, and it
   changes what `OneLoopIntegral` computes and how much of the LG split survives the squaring.
2. **Validity boundary of the source LG form.** The Green's function has `GkSourcePolicyData` to
   decide where WKB is trustworthy. The source needs the same for `T_q` *and* `T_r` — and the
   binding constraint is the **smaller** of `q, r`, which for a `d ≈ 1` triangle can be far
   super-horizon while the other factor is deeply sub-horizon. That mixed case has no analogue in
   the Green's-function logic and will need its own treatment: one factor an amplitude, the other a
   phase.
3. **Cancellation between phase groups.** `BesselIntegralResult`'s docstring records the analytic
   case: four `O(1)` groups cancelling to `1e-6` give a relative error of `1e-6` with nothing in a
   bare float to say so. The numerical case will be worse, because the phase splines are fitted
   rather than exact. Tier 1 must return summed absolute errors per group, as the analytic branch
   already does, and Tier 2's `𝒜_j` must carry them too.
4. **Overall normalisation (proposal §5).** Not settled here, and worth an explicit check rather
   than inspection. The prefactor in `analytic_integral` (`B·C·D·E` at
   `QuadSourceIntegral.py:870–873`) and the transcribed target `docs/spec/05-one-loop.md` R31 do not
   obviously agree once written in the same variables — the code's `1/((3+2b)(2+b))` against R31's
   `(2+b)/(3+2b)³`. That may well be absorbed by the differing normalisation of `f` between
   `QuadSource.source_function` and spec R28, since the code's `T_k → 1` convention places the
   `S*`/`Φ*` factors elsewhere. It is exactly the kind of factor the proposal warns is
   "convention-sensitive", and the spec→code audit pass that the transcription campaign defers
   (`prompts/spec-transcription/README.md` §1) is the right place to settle it. Flagged, not
   claimed.
5. **Polarisations (proposal §5).** `docs/spec/05-one-loop.md` records the notes as **per
   polarisation** ("for any `s`"), with no sum and no `Ω_GW`. Whatever `OneLoopIntegral` ends up
   returning should say which it is in its docstring.

---

## 5. Suggested increments

Ordered so that each is separately testable and each leaves the tree working.

1. **Refactor `QuadSource` onto `TkWKBIntegration`** (fixes F1). Keep the numeric branch for the
   super-horizon region; add the WKB branch with `(M, θ)` output. Test: the reconstructed source
   agrees with the present sampled source wherever both are valid, and the analytic radiation source
   in the deep sub-horizon regime where the present one does not.
2. **Add the triangle filter and an `(s,d)` view of the grid** (fixes F3's waste). Independent of
   everything else, and immediately recovers ~92% of the scheduled `QuadSourceIntegral` work.
3. **`ComputeTargets/resonance_map.py`** — the proposal's Steps 0–1, consuming only
   `BackgroundModel` plus `(k,q,r)`. Self-contained and testable against the constant-`w` answer
   `s = 1/√w`, as the proposal's §6 first-session goal proposes. This is the natural first
   increment, exactly as the proposal argues.
4. **Tier 1: phase-group Levin evaluation in `QuadSourceIntegral`** (fixes F2), modelled on
   `_three_bessel_Levin`. Test: against `analytic_rad` on a fixed-`w` model, which is already
   computed alongside every result — a ready-made oracle the proposal did not know about.
5. **Plotting harness** (proposal §6): `R(z) = 1/c_s(z)` and the band; flagged caustics with
   predicted widths at three `k`; the true `J(s, d=const; k)` from a brute-force dense grid. With
   step 4 in place this is cheap enough to run on a real dense `s` scan.
6. **Tier 2 + `OneLoopIntegral`**: amplitude storage, per-`k` grid construction, direct in-band
   evaluation. Only after 1–5 have agreed with each other.

The proposal's own scope advice holds up well against this code: Steps 0–2 really are the
self-contained first increment. The one adjustment is that **Step 4 is not optional here** — it
must be done before Steps 2–3 can be trusted, because the current inner integral is not accurate
in the resonant region for the reason in F2, and a grid designed against an inaccurate kernel
would be tuned to the wrong feature.
