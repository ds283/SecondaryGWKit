# GK — audit of the tensor Green's function (spec `docs/spec/02-greens-function.md`)

Agent GK. Repository at HEAD `e9a5539` (branch `main`, clean). Read-only; all scripts under
`audit/GK_*.py` in the scratch directory. Python `venv/bin/python` (numpy 2.2, scipy 1.15,
sympy 1.13).

**Headline.** The physics of spec 02 is implemented correctly. Every result-bearing formula with a
code counterpart was reproduced either symbolically (sympy, difference exactly 0) or numerically
(relative difference ≤ 2×10⁻¹¹ against the analytic oracle). The redshift ODE, the unit-jump
initial data, ω_eff², the NUM 10 correction to dω_eff/dz, the WKB matching coefficients, the
pure-sine phase shift, the R38 `Q(u)` phase representation and the (div 2π, mod 2π) carry are all
exact. Three code defects were found; two are diagnostic-only, one can change the stored
`crossover_z` in a corner case. No defect changes the value of the Green's function.

---

## 0. Summary of findings

| ID | Severity | One line | Code |
|---|---|---|---|
| GK-1 | **DEFECT** (stored diagnostic) | `GkWKBValue.analytic_G_w` / `analytic_Gprime_w` return the *radiation* oracle, so the stored `analytic_G_w`, `analytic_Gprime_w` columns of every WKB value are wrong whenever w ≠ 1/3 | `ComputeTargets/GkWKBIntegration.py:583`, `:588` |
| GK-2 | **DEFECT** (policy) | `"WKB_minimal"` is computed from `numeric_clearance`, not `WKB_clearance`; the last classification band therefore never tests the WKB end of the spline, and can select a `crossover_z` with no WKB clearance | `ComputeTargets/GkSourcePolicyData.py:364` |
| GK-3 | **DEFECT** (dead guard) | the pre-flight WKB-criterion warning omits `fabs`; `d ln ω/dz` is negative everywhere tested, so the warning can never fire (the real guard, with `fabs`, is in `WKB_phase_function`) | `ComputeTargets/GkWKBIntegration.py:312` |
| GK-4 | CONVENTION | spec R37 writes every ω_eff in the phase-shifted amplitude *starred* (open question Q9); the code keeps ω_eff(z) unstarred, i.e. it follows R34/R35/R46. The code is the self-consistent choice; R37's stars are a transcription-level slip | `ComputeTargets/GkWKBIntegration.py:459` |
| GK-5 | CONVENTION | R18's jump `[dG/dz] = −1/(a₀H(z'))` is not implemented; the code uses the unit jump `dG/dz|_{z'} = +1` of NUM 03/06 — exactly as the §0 sign-off states | `ComputeTargets/GkNumericIntegration.py:343-344` |
| GK-6 | UNVERIFIED / design note | `GkSourcePolicyData` implements **no** WKB-validity criterion: the crossover is chosen purely from spline-clearance geometry. R38's `Q → −1` is never used as a criterion, and `has_WKB_violation` is stored but never used to reject data | `GkSourcePolicyData.py:202-219, 348-365` |
| GK-7 | Observation | the R38 premise \|Q\| → 1 does not hold in practice (measured \|Q\| ≈ 28–50 over 300 → 5 in redshift), because ω_eff = k/H evolves strongly; harmless, since the reassembly is algebraically exact | `Quadrature/integrators/WKB_phase_function.py:262-303` |
| GK-8 | Observation (cosmetic) | attribute typo `_init_efolds_suph` in the un-populated branch vs `_init_efolds_subh` used everywhere else → `AttributeError` rather than the intended `RuntimeError` | `ComputeTargets/GkWKBIntegration.py:81` |
| GK-9 | Observation | the `analytic_*_w` oracle evaluates `wBackground` at the **response** redshift while the closed form assumes w constant between source and response; it is a fixed-w oracle used on a non-fixed-w background | `GkNumericIntegration.py:408-413`, `GkWKBIntegration.py:442-447` |
| GK-10 | Observation | the θ spline used to pick `Levin_z` is built with `chunk_logstep=None`, the one used for evaluation with `chunk_logstep=125` | `GkSourcePolicyData.py:170` vs `:669` |

---

## 1. Formula map

`k` in the code always means `k_phys = k/a₀` (`CosmologyConcepts/wavenumber.py`); `tau` always means
`τ = a₀η` (`BackgroundModel.py:56`). See §3.

| spec result | meaning | code location | status | how checked |
|---|---|---|---|---|
| R1 / R11 | `G'' + (k² − a''/a)G = δ(η−η')` | basis of the redshift ODE | AGREES | GK_01: derived a''/a = a₀²H²(2−ε)/(1+z)², sympy difference 0 |
| R2 | `b = (1−3w)/(1+3w)`, `a ∝ η^{1+b}` | `analytic_Gk.py:5,30` | AGREES | inspection + GK_03 (b = 0, 0.25, 1) |
| R3 | reduced conformal equation | — (intermediate) | n/a | expected: only the final R9 form is coded |
| R4 | homogeneous `√η J_{b+1/2}`, `√η Y_{b+1/2}` | `analytic_Gk.py:18` (`n0 = 0.5 + b`) | AGREES | index appears as `jv(n0,·)`, `yv(n0,·)` |
| R5–R8 | matching algebra for α, β; Wronskian | — | n/a | intermediate steps; superseded by the closed form R9 |
| **R9** | retarded `Gr_k(η,η')` (Bessel J/Y, index 1/2+b) | `analytic_Gk.compute_analytic_G` | **AGREES** | maps exactly under `G_code = −a₀H(z')Gr_k`: `A*B*(C−D) = −H_s(π/2)√(ττ_s)[J(kτ_s)Y(kτ) − J(kτ)Y(kτ_s)]`, = R9's `(π/2)√(ηη'){−Y(kη')J(kη) + J(kη')Y(kη)}` times `−H_s`. GK_03: max rel. difference vs the code's own ODE **1.5×10⁻¹¹** over z = 19…0.5 for w = 1/3, 0.2, 0 |
| R9 (d/dz) | derivative of the oracle | `analytic_Gk.compute_analytic_Gprime` | AGREES | analytic: `d/dτ[√τ Y_n(kτ)] = τ^{-1/2}[(1+b)Y_n − kτY_{n+1}]` with `n = 1/2+b` reproduces `C1·C2 + D1·D2` and the `−(1/H)` Jacobian gives the `H_s/H` prefactor; GK_03 rel. diff ≤ 1.7×10⁻¹⁰ |
| R10 | `G_k[h] = (a'/a)G_k[χ]` | none | none, expected | the pipeline works with χ = a·h throughout |
| R12 | `d/dη = −(1+z)aH d/dz` | `BackgroundModel.py:56` (`dτ/dz = −1/H`) | AGREES | GK_03 checks the stub background satisfies it; identity `(1+z)aH = a₀H` used in GK_01 |
| R13–R16 | intermediate redshift algebra, explicit a₀ | — | n/a | folded into R17; the minus sign of Tier 2.6 is what produces the ε friction, confirmed in GK_01 |
| **R17 / R19 / R32** | `G'' + ε/(1+z) G' + (k²/H² − (2−ε)/(1+z)²)G = 0` | `GkNumericIntegration.py:64-66` | **AGREES** | GK_01: sympy difference between the code's `dGprime_dz` and R17 solved for `G''` is **exactly 0** |
| R18 | jump `−1/(a₀H(z'))` | not implemented | CONVENTION (GK-5) | `initial_value=0.0, initial_deriv=1.0` at `:343-344`; §0 item 2/3 sanctions this |
| R20 | friction elimination, `ω² = f''/f + (ε/(1+z))f'/f + k²/H² + (ε−2)/(1+z)²` | `WKB_Gk.py:4-19` | AGREES | GK_02: derived from `dln f/dz = −ε/2/(1+z)`, difference from R22 is 0 |
| R21 | `ε = (1+z)dlnH/dz`, `ε' = dlnH/dz + (1+z)d²lnH/dz²` | `BackgroundModel.py:318-334` | AGREES | inspection; used by all of GK_02–GK_09 |
| **R22 / R32** | `ω_eff² = k²/H² − ε'/2/(1+z) + (3ε/2 − ε²/4 − 2)/(1+z)²` | `WKB_Gk.Gk_omegaEff_sq` | **AGREES** | GK_02: sympy difference **0** (both against R22 and against the from-scratch R20 construction) |
| R23 | `A ∝ ω^{-1/2}`, `Θ = ∫ω dz` | `WKB_phase_function.py:96` (`dtheta_dz = +omega`), `GkWKBIntegration.py:459` | AGREES | GK_09 reproduces Θ; GK_04(c) reproduces G |
| R24 | ΛCDM `dlnH/dz`, `d²lnH/dz²` | `CosmologyModels/LambdaCDM/LambdaCDM.py:131-171` | AGREES | GK_07: sympy difference 0 |
| R25 / R34 | LG modes anchored at the matching point | `GkWKBIntegration.py:459-466`; `spline_wrappers.py:118-132` | AGREES | GK_04(a): amplitude `ω^{-1/2}(H*/H)^{1/2}` verified by reproducing (G*, G'*) |
| **R26 / R36** | `α = G*ω*^{1/2}`, `β = [G'* + (G*/2)(ω'/ω + ε/(1+z))*]/ω*^{1/2}` | `GkWKBIntegration.py:392-396` | **AGREES** | GK_04(a): 2000 random quadruples; worst rel. error reproducing G* = 1.9×10⁻¹⁶, G'* = 4.1×10⁻⁸ (catastrophic cancellation in the random test, not a code error) |
| R27 | old (uncorrected) `ω'/ω` | not used | AGREES (correct version used) | GK_02: the code matches R31, which differs from R27 by the `−2εk²/H²/(1+z)` term |
| R28 | `ε'`, `ε''` from `lnH` derivatives | `BackgroundModel.py:327-343` | AGREES | inspection |
| R29 | ΛCDM `d³lnH/dz³` | `LambdaCDM.py:173-197` | AGREES | GK_07: sympy difference 0 |
| R30 / R35 | `exp(−½∫ε/(1+z)) = (H*/H)^{1/2}` | `GkWKBIntegration.py:458` (`H_ratio = H_init/H`) | AGREES | GK_04(c) end-to-end |
| R33 | `Θ' = ±ω_eff` | `+` branch taken, `WKB_phase_function.py:96` | CONVENTION/AGREES | phase integrated from `z_init` **downward**, so Θ ≤ 0; matches §0.2 item 2.8 |
| **R31** | NUM 10 corrected `2ω dω/dz` (adds `−2εk²/H²/(1+z)`) | `WKB_Gk.Gk_d_ln_omegaEff_dz:36-43` | **AGREES** | GK_02: sympy difference **0**, and the code's `numerator/(2ω²)` equals `ω'/ω` exactly |
| R37 | `B sin(Θ+ΔΘ)`, `tanΔΘ = α/β`, `B² = (α²+β²)…` | `GkWKBIntegration.py:400-410` | AGREES for the shift; **CONVENTION** for the stars (GK-4) | GK_04(a),(b): `sin_coeff·sin(Θ+ΔΘ) + cos_coeff·cos(Θ+ΔΘ) = α cosΘ + β sinΘ` to 3×10⁻¹² for 10 000 random Θ; `cos_coeff` set identically to 0 |
| **R38** | `Θ = Θ_i + ω_i(1+u)Q`, `dQ/du = −(ω/ω_i)/(1+u) − Q/(1+u)`, `Q(0)=0` | `WKB_phase_function.py:266-303` | **AGREES** (verbatim) | GK_09: `θ_Q` matches a direct `∫ω dz` to 2.9×10⁻¹² for u > 1 |
| R39–R42, R44–R46 | transfer-function WKB (NUM 11 Steps 6–8) | `ComputeTargets/WKB_Tk.py`, `TkWKBIntegration.py` | out of scope (Group 1) | see spec 01; noted below |
| R43 | transfer-function `ω_eff²` | `WKB_Tk.Tk_omegaEff_sq` | AGREES | with `c_s² = wPerturbations` (so `3c_sc_s' = (3/2)w'`) and a₀ absorbed, the code is R43 term by term; GK_08 also confirms `Tk_d_ln_omegaEff_dz` is exactly `d(ω²)/dz` (i.e. the commit `641bb51` fix is right: the pre-fix form differs by `+9(1+w)w'/4/(1+z)²`) |

R## with **no** code counterpart: R3, R5, R6, R7, R8 (intermediate algebra for the Bessel
normalisation — the code only ever needs the assembled R9), R10 (the h-field Green's function; the
pipeline never leaves χ), R13–R16 (intermediate redshift algebra), R18 (superseded by the unit-jump
convention), R27 (superseded by R31). All expected.

---

## 2. Findings in detail

### GK-1 — DEFECT: `analytic_G_w` / `analytic_Gprime_w` of a WKB value return the radiation oracle

`ComputeTargets/GkWKBIntegration.py:582-588`

```python
    @property
    def analytic_G_w(self) -> Optional[float]:
        return self._analytic_G_rad          # <-- should be self._analytic_G_w

    @property
    def analytic_Gprime_w(self) -> Optional[float]:
        return self._analytic_Gprime_rad     # <-- should be self._analytic_Gprime_w
```

The constructor stores `self._analytic_G_w` correctly (`:527-528`), and the value *is* computed with
`wBackground(z)` at `:442-447`; only the accessors are wrong. The corresponding class for the numeric
branch, `GkNumericValue` (`GkNumericIntegration.py:503-509`), is correct — so this is an isolated
copy-paste slip in the WKB class.

Consequences for stored numbers:
* `Datastore/SQL/ObjectFactories/GkWKBIntegration.py:564` serialises `value.analytic_G_w`, i.e. the
  `analytic_G_w` and `analytic_Gprime_w` columns of the GkWKB value table contain the **radiation**
  values.
* `GkSource.assemble_GkSource_values` (`GkSource.py:359-368`) copies `WKB.analytic_G_w` into
  `GkSourceValue` whenever there is no numeric datum at that source redshift, so the same columns of
  `GkSourceValue` are wrong in the WKB-only region.
* Downstream use is diagnostic only (`extract_GkWKB_data.py:164-206`, `extract_GkSource_data.py`);
  `analytic_G_*` never enters a physical result. The practical effect is that any plot or
  spot-check of "WKB vs fixed-w analytic" is silently comparing against w = 1/3.

Magnitude: for the constant-w backgrounds of GK_03, `analytic_G_rad` and `analytic_G_w` differ by
O(1) relative (e.g. w = 0.2, z_source = 20, z = 1: −162.92 for w = 0.2 vs −192.76 for w = 1/3, a
15 % difference), so the corruption is not subtle where w departs from 1/3.

### GK-2 — DEFECT: `WKB_minimal` tests the numeric clearance

`ComputeTargets/GkSourcePolicyData.py:363-364`

```python
                    "numeric_minimal": numeric_clearance > 0.0,
                    "WKB_minimal": numeric_clearance > 0.0,     # <-- should be WKB_clearance
```

`CLASSIFICATION_BANDS[-1]` is `("minimal", ["numeric_minimal", "WKB_minimal"])`, so that band
degenerates to a single condition. A point in the overlap region with
`numeric_clearance > 0` but `WKB_clearance ≤ 0` — i.e. sitting at or above the top of the WKB
spline's domain — would be accepted as a `minimal`-quality crossover. With
`numeric_policy="maximize-WKB"` (the only policy used in `main.py:2621-2637`) the *largest*
qualifying redshift is chosen (`:392-396`), which is precisely the direction that makes a bad WKB
clearance more likely.

Reachability: only if none of the nine earlier bands matched, i.e. every overlap point has
`WKB_clearance ≤ 0.01`. When that happens the intended fallback is the explicit
`crossover_z = primary_WKB_largest_z, quality = "minimal"` at `:412-416`; the bug pre-empts that
fallback with an essentially equivalent (but unaudited) choice. Consequence if hit: a different
stored `crossover_z`, hence a different `Levin_z` (`_classify_Levin:149`) and a different
numeric/WKB split of the `QuadSourceIntegral` regions (`QuadSourceIntegral.py:152-190`). Because
both branches represent the same G (see GK-11 below / §2 last item), the integral would still be
right up to the WKB error evaluated near the edge of its spline. Cannot be exercised without a
database run; flagged as a defect on the code, not a measured error.

Note also that `numeric_clearance` at `:350` is `(logz − numeric_logz_low)/numeric_logz_low`, a
*relative* clearance in `log(1+z)`; the thresholds 0.05/0.025/0.01 are therefore fractions of
`log(1+z_min)`, not e-folds. That is a deliberate choice, but it makes the thresholds
z-range-dependent.

### GK-3 — DEFECT (dead guard): missing `fabs` in the pre-flight WKB criterion

`ComputeTargets/GkWKBIntegration.py:312-318`

```python
        WKB_criterion_init = d_ln_omega_init / sqrt(omega_sq_init)
        if WKB_criterion_init > 1.0:
            print(f"!! Warning (GkWKBIntegration) ...")
```

Every other evaluation of the same criterion takes the modulus:
`GkWKBIntegration.py:456`, `GkNumericIntegration.py:416-418`,
`WKB_phase_function.py:89`, `:285`, `:662`. Measured (GK_03/side test): `Gk_d_ln_omegaEff_dz` is
**negative at every point tested** (w = 0, 0.2, 1/3; k = 1 and 10⁴; z = 1, 20, 200; values
−0.0075 … −1.17), because the dominant term is `−2ε(k/H)²/(1+z) < 0`. So
`WKB_criterion_init > 1.0` is unreachable and the warning is dead code. There is no numerical
consequence: `WKB_phase_function.py:662-666` recomputes the criterion **with** `fabs` and raises a
`RuntimeError`, so the actual protection is in place. But the comment at
`GkWKBIntegration.py:371-372` ("the WKB criterion < 1 … has been checked in `compute()`") is not
true of `compute()`.

### GK-4 — CONVENTION: R37's starred ω_eff (spec open question Q9)

Spec R37 as transcribed has `B² = (α²+β²)H*/(ω*_eff H)`, i.e. no `1/ω_eff(z)`. The code uses

```python
            H_ratio = H_init / H
            norm_factor = sqrt(H_ratio / omega)        # GkWKBIntegration.py:458-459
```

with `omega = sqrt(Gk_omegaEff_sq(model, k, current_z))` at the *evaluation* redshift, i.e. it keeps
`ω_eff^{-1/2}(z)(H*/H)^{1/2}` of R34/R35 and of the parallel transfer-function result R46. This is
the correct reading: GK_04(c) shows that with the unstarred amplitude the reconstruction agrees with
the exact solution to 3.8×10⁻¹² (radiation, where the LG form is exact) and to 1.8×10⁻³ at 5+
e-folds inside the horizon for w = 0.2 (genuine WKB error). With the starred amplitude the
reconstruction would drift by the factor `(ω*/ω(z))^{1/2}`, which over z = 150 → 19 in the same test
is ≈ 4.4 — a factor-of-four error. **Q9 should be closed in favour of the unstarred form; the stars
on NUM 11 p.5 are a slip and the code is right.**

The sign-fixing step at `GkWKBIntegration.py:404-410`
(`sin_coeff = sgn(sin ΔΘ)·sgn(G_init)·B`) is a no-op: `sin ΔΘ ∝ raw_cos_coeff = ω^{1/2}G_init`, so
the two sign factors always agree and `sin_coeff = +B`. GK_04 confirms it (`sin_coeff == +B` in
every case) and that the resulting single-sine form is algebraically identical to
`α cos Θ + β sin Θ`. With the unit-jump start (`G_init = 0`, `Gprime_init = 1`) one gets `ΔΘ = 0`
and `sin_coeff = 1/√ω_i` exactly (GK_06).

### GK-5 — CONVENTION: the integration variable and the unit jump

`GkNumericIntegration.RHS` is integrated by `numeric_with_phase_cut`
(`Quadrature/integrators/numeric_with_phase_cut.py:126-140`) with
`t_span=(z_init.z, z_min)` — the independent variable is **z itself**, not `log(1+z)`. The initial
state is `[initial_value, initial_deriv] = [0.0, 1.0]`
(`GkNumericIntegration.py:343-344`), so `dG/dz|_{z=z'} = +1` **in z**, as spec §0 item 2 requires.
`log(1+z)` appears only (i) in the splines built later (`GkSourcePolicyData.py:603, 646, 658`),
(ii) in the oscillation-resolution diagnostic (`RHS:76` with
`NumericIntegrationSupervisor.report_wavelength:80`, which correctly converts `Δlog z → (1+z)Δlog z`
before comparing with the z-space wavelength `2π/ω_eff`), and (iii) in the Levin frequency test.

State layout is `state[0] = G`, `state[1] = dG/dz` (`VALUE_INDEX`, `DERIV_INDEX`), and **both** are
stored (`GkNumericValue.G`, `.Gprime`, `GkNumericIntegration.py:388-389, 421-434`). The stored
`Gprime` is genuinely `dG/dz` (verified against `compute_analytic_Gprime` to ≤ 1.7×10⁻¹⁰ in GK_03).

Measured sign/normalisation (GK_03): with z_source = 20, radiation,
`G/(z − z') = 1.00005, 1.0005, 1.0048` at `Δz = 10⁻³, 10⁻², 10⁻¹`. So
`G_code ≈ z − z' < 0` just after the source, exactly §0 item 2.

### GK-6 — the WKB-validity criterion actually implemented (spec R38 / Tier 2.8)

`GkSourcePolicyData` contains **no** WKB-validity test. What exists, and where:

| test | threshold | where | effect |
|---|---|---|---|
| `\|d ln ω_eff/dz\|/ω_eff > 1` at z_init | 1.0 | `WKB_phase_function.py:662-666` | hard `RuntimeError` |
| same, during stage 1 and stage 2 | 1.0 | `WKB_phase_function.py:89-93`, `:285-289` | sets `has_WKB_violation`, records z and e-folds |
| same, at z_init (local) | 1.0 | `GkWKBIntegration.py:312` | dead (GK-3) |
| response redshift within 3 e-folds of re-entry | `z_exit_subh_e3` | `GkWKBIntegration.py:112-122` | `ValueError` |
| source redshift more than 4 e-folds inside the horizon for the numeric solve | `z_exit_subh_e4` | `GkNumericIntegration.py:302-306` | `ValueError` |
| WKB datum with `z_response ≥ z_exit_subh_e3` | — | `GkSource.py:111-117` | warning |
| numeric/WKB hand-over window for the initial condition | `z_exit_subh_e3` … `z_exit_subh_e6` | `GkNumericIntegration.py:120-121` | search window for the fixed-phase cut |
| choice of `crossover_z` | `CLEARANCE_GOOD=0.05`, `CLEARANCE_ACCEPTABLE=0.025`, `CLEARANCE_MARGINAL=0.01` (relative clearance in `log(1+z)`), plus `numeric_policy="maximize-WKB"` | `GkSourcePolicyData.py:202-219, 348-416` | spline-geometry only |
| choice of `Levin_z` | `policy.Levin_threshold` = **1.5** or **5.0** on `\|dθ/dlog(1+z)\|` | `GkSourcePolicyData.py:181-191`; values set in `main.py:2627, 2633` | switch to Levin quadrature |

So the R38 `Q → −1` fixed point is *not* used as a validity criterion anywhere. Consistent with the
author sign-off (§0.2 item 2.8: "nothing in it assumes the sign of Q; the supervisor only records
its extremes") — `QSupervisor.update_Q` (`Quadrature/supervisors/WKB.py:173-178`) tracks
`largest_Q`/`smallest_Q` and these are written to metadata
(`WKB_phase_function.py:485-486`) and nowhere else. `has_WKB_violation` is likewise stored but never
consulted when assembling `GkSource` or choosing `crossover_z`; a mode that violated the criterion
mid-integration is used anyway. Whether that matters cannot be decided without a database run —
flagged UNVERIFIED.

### GK-7 — the |Q| ≈ 1 premise of R38 does not hold in practice

GK_09, constant-w background, k = 10⁴, phase integrated from z = 300 to z = 5:

| w | ω_eff(z_i) | Θ(z_end) | Q range | max rel. error of `Θ = Θ_i + ω_i(1+u)Q` |
|---|---|---|---|---|
| 1/3 | 0.1104 | −1633.4 | [−50.0, 0] | 2.2×10⁻¹² |
| 0.2 | 0.3456 | −2851.2 | [−27.9, 0] | 2.9×10⁻¹² |

`Q` runs to ≈ −50, not ≈ −1, because `ω_eff ≈ k/H` grows by a large factor as z decreases (the
spec's own proviso, "if ω_eff does not evolve much"). This is harmless — the representation is
algebraically exact and `WKB_product_mod_2pi` reassembles `ω_i(1+u)Q` to 2.2×10⁻¹⁶ relative
(GK_05) — but the metadata fields `stage_2_largest_Q` / `stage_2_smallest_Q` should not be read as
"close to ±1" diagnostics.

### GK-8 — attribute typo

`GkWKBIntegration.py:81` initialises `self._init_efolds_suph = None` while `:100`, `:245-248` and
`:369` all use `_init_efolds_subh`. Reading `init_efolds_subh` on a freshly-constructed
(un-populated) object raises `AttributeError` rather than the intended
`RuntimeError("init_efolds_subh has not yet been populated")`. Cosmetic.

### GK-9 — the fixed-w oracle is evaluated with w at the response redshift

`GkNumericIntegration.py:408-413` and `GkWKBIntegration.py:442-447` call
`compute_analytic_G(k, wBackground(z_response), tau_source, tau, H_source)`. The closed form R9
requires w constant between source and response (it is derived from `a ∝ η^{1+b}`), and it also
assumes the constant-w relation between τ and z; the code supplies the *true* background τ and the
*instantaneous* w at the response redshift. So `analytic_G_w` is only a meaningful oracle inside a
single constant-w era (as `analytic_G_rad` is inside radiation domination). Not a defect — but it
means a non-zero `G − analytic_G_w` across e.g. the matter/radiation transition is expected and must
not be read as a code error. (In GK_03 the background genuinely has constant w, and the agreement
is 10⁻¹¹.)

### GK-10 — chunking inconsistency between the two θ splines

`_classify_Levin` builds `phase_spline(..., chunk_logstep=None)` with the comment that chunking
risks edge effects in the derivative (`GkSourcePolicyData.py:159-172`), while `_create_functions`
builds the θ spline used for evaluation with `chunk_logstep=125` (`:662-671`). The `Levin_z`
selection therefore uses a different spline from the one that will be evaluated. Not measured.

### GK-11 (verification, not a finding) — the assembled `GkSource` function is the unit-jump G on both sides of the crossover

* **Numeric branch.** `GkSourcePolicyData.py:602-603` splines `(log(1+z_source), v.numeric.G)`
  directly. `v.numeric.G` is `GkNumericValue.G`, which is `data["value_sample"]` from the ODE with
  `G = 0, dG/dz = +1` at the source. No sign flip, no re-normalisation anywhere in the chain
  (`GkNumericIntegration.py:388, 421-434` → `GkSource.py:326` → `GkSourcePolicyData.py:603`).
* **WKB branch.** `GkSourcePolicyData.py:644-648` splines
  `sin_coeff · sqrt(H_ratio/sqrt(omega_WKB_sq))`, and `GkWKBSplineWrapper.__call__`
  (`spline_wrappers.py:118-124`) returns `sin_amplitude · sin(theta_mod_2pi)` — with
  `cos_amplitude_spline = None` (`GkSourcePolicyData.py:676`), consistent with
  `cos_coeff ≡ 0` set at `GkWKBIntegration.py:409`. This is exactly the `G_WKB` of
  `GkWKBIntegration.py:463-466`, whose coefficients were matched to `(G, G')` taken either from the
  numeric solution's fixed-phase cut (`main.py:1303-1305`, via
  `numeric_with_phase_cut.py:205-211`, a minimum of G so `G' ≈ 0`) or from the unit-jump initial
  data (`main.py:1375-1376`). Both seeds are the *same* unit-jump G.
* **Measured continuity** (GK_06, w = 0.2, k = 10⁴, z_response = 20, both branches built with the
  code's own formulae):

| z_source | e-folds subh. | numeric branch | WKB branch | rel. difference |
|---|---|---|---|---|
| 200 | 4.97 | 0.6358017 | 0.6362086 | 6.4×10⁻⁴ |
| 150 | 5.20 | −0.8255160 | −0.8255478 | 3.9×10⁻⁵ |
| 100 | 5.52 | 0.2091322 | 0.2091882 | 2.7×10⁻⁴ |
| 60 | 5.92 | 0.0487358 | 0.0487222 | 2.8×10⁻⁴ |
| 40 | 6.24 | −0.0734694 | −0.0734708 | 2.0×10⁻⁵ |
| 30 | 6.46 | 0.0340134 | 0.0340142 | 2.4×10⁻⁵ |
| 25 | 6.60 | −0.0279843 | −0.0279840 | 9.2×10⁻⁶ |

  Same sign, same normalisation, residual consistent with the intrinsic WKB error at 5–6.6 e-folds
  inside the horizon. And with z_source → z_response from above (GK_06):
  `G = −1.000001×10⁻⁴, −9.997533×10⁻⁴, −9.717272×10⁻³` at `Δz = 10⁻⁴, 10⁻³, 10⁻²`, i.e.
  `G → (z_response − z_source) → 0⁻`. **The assembled function is `G_code`, not `−G_code` and not
  `Gr_k`.**
* **Consumer.** `QuadSourceIntegral.py:140-190` uses region 1 = numeric for
  `z_source > crossover_z`, region 2 = direct WKB quadrature between `crossover_z` and `Levin_z`,
  region 3 = Levin below `Levin_z` — so the crossover is a change of representation of one object,
  as intended.

---

## 3. Numerical convention notes

* **a₀ is absorbed, and the covariance test is structural.** The Green's-function code touches `k`
  only through `k_float = k.k` = `k_phys = k/a₀` (in `k/H`, `Gk_omegaEff_sq`, `Gk_d_ln_omegaEff_dz`)
  and through `k·tau` with `tau = a₀η` in `analytic_Gk.py`. There is no other appearance of a
  momentum or a time. Hence `a₀ → λa₀` with comoving `k → λk` and `η → η/λ` leaves every input to
  every formula unchanged, and the covariance test is passed identically rather than
  approximately — GK_03 confirms `G(z=5)` is bit-identical (`−52.3760799078`) for λ = 1, 3, 0.1.
  The model classes offer no dial for `a₀`, which is exactly the point: it cannot be varied because
  it never appears. In GK_01 the derivation was done with `a₀` explicit and the a₀'s cancel to give
  R17 in `k_phys`.
* **Integration variable.** The Green's-function ODE and the WKB phase ODE are both integrated in
  **z** (`numeric_with_phase_cut.py:126-131`; `WKB_phase_function.py:108-118`); the stage-2 phase
  ODE is integrated in `u = z_init − z` (`:305-314`). Only the splines and the Levin frequency test
  work in `log(1+z)`.
* **Phase direction.** `dΘ/dz = +ω_eff` with `Θ(z_init) = 0`, integrated downward, so `Θ < 0` and
  `Θ` becomes more negative the further the response is from the source. `phase_spline` is
  constructed with `increasing=False` (`GkSourcePolicyData.py:171, 670`) to match. `WKB_mod_2pi`
  fixes `mod 2π ∈ (−2π, 0]` (`WKBtools.py:19-22`). GK_05 verifies:
  `WKB_mod_2pi` round-trip 4.4×10⁻¹⁶; `WKB_product_mod_2pi` round-trip 2.2×10⁻¹⁶;
  `shift_theta_sample` preserves `θ + ΔΘ` up to a constant 2π offset (spread 2.3×10⁻¹⁵) so
  `sin` is unchanged to 6.2×10⁻¹⁵; `phase_spline` reassembles a known analytic phase with
  `max|sin(spline mod 2π) − sin(θ_exact)| = 7.2×10⁻¹³`.
* **Signs that are conventions.** (i) the `−1/(a₀H)` source of R17/R18 vs the code's `+1` unit jump
  (§0 item 3); (ii) `Θ' = +ω_eff` rather than `−ω_eff` (R33's `±`), which fixes the sign of Q
  (§0.2 item 2.8) and is what makes `Q → −1` rather than `+1`; (iii) `G_code < 0` just after the
  source. None of these changes a physical result on its own.
* **ε and its derivatives** come from `d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3`, which `LambdaCDM`
  supplies analytically (R24/R29, verified) and which `LambdaCDM_GenericEOS`/`QCD_Cosmology` obtain
  from spline differentiation (`BackgroundModel.py:120-157`). `ω_eff²` needs `ε'` and
  `d ln ω_eff/dz` needs `ε''`, i.e. up to `d³lnH/dz³` — three spline derivatives for a generic EOS.
  Accuracy of that chain is a property of the background model, not of spec 02; not audited here.

---

## 4. Scripts run

All in `<scratch>/audit/`, runnable with `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python`
(GK_04 and GK_06 import GK_03, so run them from that directory).

| script | checks | result |
|---|---|---|
| `GK_01_sympy_ode.py` | derives `a''/a` and the redshift ODE from R1/R12 and compares with R17 and with the code's `dGprime_dz` | both differences exactly **0** |
| `GK_02_sympy_omega.py` | R20→R22 construction, R22 vs `Gk_omegaEff_sq`, R31 vs `d(ω²)/dz`, R31 vs the code numerator, and `numerator/(2ω²) = ω'/ω` | all differences **0** |
| `GK_03_numeric_analytic.py` | integrates the code's own `RHS` on exact constant-w backgrounds (w = 1/3, 0.2, 0) and compares G and dG/dz with `compute_analytic_G`/`Gprime`; unit-jump sign; a₀ covariance; ω_eff diagnostics | max rel. difference **1.5×10⁻¹¹** (G), **1.7×10⁻¹⁰** (G'); `G/(z−z') → 1`; G identical under λ-rescaling |
| `GK_04_wkb_matching.py` | (a) 2000 random `(G*,G'*,ω,ω',ε)` quadruples reproduce G*, G'* from the code's coefficients; (b) the `B sin(Θ+ΔΘ)` rewriting; (c) end-to-end WKB reconstruction vs the exact solution | (a) 1.9×10⁻¹⁶ / 4.1×10⁻⁸; (b) 3.2×10⁻¹²; (c) 3.8×10⁻¹² (radiation, exact case), 1.8×10⁻³ at 5+ e-folds subhorizon for w = 0.2 |
| `GK_05_phase_reassembly.py` | `WKB_mod_2pi`, `WKB_product_mod_2pi`, `shift_theta_sample`, `phase_spline` carry | 4.4×10⁻¹⁶, 2.2×10⁻¹⁶, 6.2×10⁻¹⁵, 7.2×10⁻¹³ |
| `GK_06_gksource_stitch.py` | at fixed z_response, numeric branch vs WKB branch as `GkSourcePolicyData` assembles them; unit-jump sign in the source variable | branches agree to 9×10⁻⁶ … 6×10⁻⁴; `G → (z_response − z_source) → 0⁻` |
| `GK_07_sympy_lnH.py` | R24, R29 vs `LambdaCDM.d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3` | all differences **0** |
| `GK_08_sympy_Tk_omega.py` | (cross-check) R43 and `Tk_d_ln_omegaEff_dz` = `d(ω²)/dz`; confirms the `641bb51` fix | difference **0**; pre-fix form off by `+9(1+w)w'/4/(1+z)²` |
| `GK_09_stage2_Q.py` | R38 `Q(u)` representation vs a direct `∫ω dz`; range of Q | θ reproduced to 2.9×10⁻¹²; Q ∈ [−50, 0] |
