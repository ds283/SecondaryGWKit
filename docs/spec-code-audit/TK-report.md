# Audit TK — background model and scalar transfer function

Scope: `docs/spec/01-transfer-function.md` (all of it) against `ComputeTargets/BackgroundModel.py`,
`CosmologyModels/*`, `ComputeTargets/TkNumericIntegration.py`, `ComputeTargets/TkWKBIntegration.py`,
`ComputeTargets/WKB_Tk.py`, `ComputeTargets/analytic_Tk.py`, plus the integrators
(`Quadrature/integrators/{WKB_phase_function,numeric_with_phase_cut}.py`) and
`CosmologyConcepts/wavenumber.py`. Repository HEAD e9a5539, branch `main`, read-only.

**Headline.** The transfer-function physics agrees with spec 01. Every result-bearing formula that
has a code counterpart was checked symbolically or numerically and reproduces the spec exactly
(R14/R21, R23/R24/R26, R27/R29, R30-corrected, R11, R18/R20, R28/R31). The 641bb51 fix to
`Tk_d_ln_omegaEff_dz` is present and the full derivative is now exact. One genuine physics defect
was found, and it is **not** in spec 01's formulas but in one of the two cosmology classes that
supply `c_s^2`: `LambdaCDM_GenericEOS.wPerturbations` divides by the **total** density, Λ included,
contradicting the author's Λ-unperturbed convention and the `LambdaCDM` sibling (factor 3.21 too
small at z=0). The remaining findings are diagnostic/bookkeeping.

---

## 0. Summary table

| ID | Severity | One line | Code:line |
|---|---|---|---|
| TK-1 | **DEFECT** | `LambdaCDM_GenericEOS.wPerturbations` denominator includes ρ_Λ; c_s² too small by ×3.21 at z=0, ×1.28 at z=1, <1% for z≳5 | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:290` |
| TK-2 | **DEFECT** (stored diagnostic) | `TkWKBValue.analytic_T_w` / `analytic_Tprime_w` return the `_rad` members, so the TkWKB `analytic_*_w` DB columns store the radiation oracle | `ComputeTargets/TkWKBIntegration.py:632,636` |
| TK-3 | **DEFECT** (dead guard) | `WKB_criterion_init` computed without `fabs`; d ln ω_eff/dz < 0 throughout, so the warning can never fire | `ComputeTargets/TkWKBIntegration.py:356` |
| TK-4 | **DEFECT** (wrong exception) | `_init_efolds_suph` initialised where the property and `store()` use `_init_efolds_subh` → `AttributeError` not `RuntimeError` | `ComputeTargets/TkWKBIntegration.py:115` vs `:136,285,413` |
| TK-5 | **DEFECT** (accuracy, ends only) | `_build_derivative` spline stack is grid-end biased: ε′ rel. err 9.4e-4 and ε″ 3.0e-1 at the z=0.1 end vs 1.4e-6/6.3e-5 interior; propagates to ω_eff² at ≤9e-7 and d ln ω/dz at ≤6e-5 | `ComputeTargets/BackgroundModel.py:129-149` |
| TK-6 | CONVENTION (measured) | τ_init uses spec R20's radiation-only closed form but evaluates it on the *total* ρ; rel. error 8.5e-4 at z_init=1e6, 8.5e-8 at 1e10, entering as a constant absolute offset in a₀τ | `ComputeTargets/BackgroundModel.py:61-64` |
| TK-7 | CONVENTION (measured) | T=1, T′=0 at z_exit_suph_e5 is a super-horizon approximation; measured relative error in T_k is 2.5e-6 (radiation), scaling as (k c_s τ_init)² | `ComputeTargets/TkNumericIntegration.py:376-377` |
| TK-8 | CONVENTION | dθ/dz = +ω_eff with θ(z_init)=0 and integration towards smaller z, so θ *decreases* (goes negative); the `sgn_sin_deltaTheta*sgn_T` factor is provably always +1; stored `theta_div_2pi` is rebased by an arbitrary integer | `Quadrature/integrators/WKB_phase_function.py:96-101`, `ComputeTargets/TkWKBIntegration.py:454-459`, `LiouvilleGreen/WKBtools.py:103-108` |
| TK-9 | AGREES (information) | c_* = 3(1+w*)/(5+3w*) is applied nowhere in the T_k path; T_k→1 is a normalisation of φ/φ* only, and the ζ*→φ* constant belongs to the still-stubbed one-loop prefactor | `ComputeTargets/OneLoopIntegral.py:97-110` |

No other disagreement with spec 01 was found.

---

## 1. Formula map

`c_s^2` below always means `wPerturbations(z)` (spec 01 Tier 3 block). "k" in the code is `k.k`,
the physical wavenumber today k/a₀ (spec 02 §0.1 item 1).

| Spec result | Meaning | Code location | Status | How checked |
|---|---|---|---|---|
| R1 | Linearised G^η_η, G^i_j | — | no counterpart (expected: derivation only) | — |
| R2 | ηη Einstein equation | — | no counterpart (expected) | — |
| R3 | diagonal ij equation | — | no counterpart (expected) | — |
| R4 | closure δp = c_s²δρ = wδρ | `wPerturbations` in `TkNumericIntegration.py:62`, `WKB_Tk.py:9`, `TkWKBIntegration.py:47` | AGREES (but see TK-1 for one implementation of `wPerturbations`) | code reading + TK_05(d) |
| R5 | φ equation in conformal time | ancestor of the coded ODE | AGREES | `TK_01`: R5 + R12/R13 → R14 identically (residual 0) |
| R6 | ρ ∝ a^{-3(1+w)}, a ∝ η^{2/(1+3w)}, 1+b = 2/(1+3w) | `analytic_Tk.py:6,20` (`b = (1-3w)/(1+3w)`) | AGREES | algebraic identity with R7 |
| R7 | b, H_conf, w↔b shorthand | `analytic_Tk.py:6,20` | AGREES | as above |
| R8 | constant-w φ equation, vanishing mass term | — | no counterpart (expected: intermediate) | — |
| R9 | χ normal form | — | no counterpart (expected) | — |
| R10 | general Bessel solution, order 3/2+b | only the J branch survives, in `analytic_Tk.py:10-16` | AGREES (β=0 by R11) | code reading |
| **R11** | φ = 2^{3/2+b}Γ(5/2+b)φ*(kc_sη)^{-3/2-b}J_{3/2+b} | `ComputeTargets/analytic_Tk.py:5-16` | **AGREES** | term-by-term identity; `k*cs*tau = (k/a₀)c_s(a₀η)`; independently confirmed as the exact solution of the coded ODE by `TK_03` (residual → 0 as (kc_sτ_i)²) |
| (R11 derivative) | dT/dz of R11 | `analytic_Tk.py:19-36` | AGREES | analytic: dT/dx = 2^{1/2+b}Γ x^{-5/2-b}[xJ_{1/2+b} − (3+2b)J_{3/2+b} − xJ_{5/2+b}], dx/dz = −k c_s/H; matches line-for-line, and `TK_03` measures agreement with the ODE's dT/dz to the same 1e-6–1e-12 level |
| R12 | dz = −(1+z)aH dη, d/dη = −(1+z)aH d/dz | implicit in the coded z-space ODE | AGREES | `TK_01` |
| R13 | H_conf = aH, H_conf′ = a²H²(1−ε) | implicit | AGREES | `TK_01` (residual 0) |
| **R14** | boxed φ equation in redshift | `ComputeTargets/TkNumericIntegration.py:73-80` | **AGREES** | `TK_01`: `code RHS − spec RHS = 0` symbolically. Note (1+z)²a²H² = a₀²H², so `wPerturbations*k_over_H_2` is exactly c_s²k²/((1+z)²a²H²) — the R29/Q8 form |
| R15 | ε = (1+z)H⁻¹dH/dz = d ln H/d ln(1+z) | `BackgroundModel.py:318-325` (`epsilon = (1+z)*d_lnH_dz`); `ZSplineWrapper` line 61 divides the log-spline derivative by (1+z), so `d_lnH_dz` is a genuine z-derivative | AGREES | code reading + `TK_05(a)`: `LambdaCDM.d_lnH_dz` vs sympy d/dz ln H, max rel diff 9.6e-16 |
| R16 | w(z) for m+r+Λ | `LambdaCDM.wBackground:199-213`, `wPerturbations:215-224`; `LambdaCDM_GenericEOS:269-292` | **split — see TK-1** | `TK_05(a)` rel diff ≤2.8e-15 against sympy; `TK_05(d)` for the Λ-denominator discrepancy |
| R17 | continuity equation for total ρ | — | no counterpart (expected: consistency statement; the code never integrates ρ, it evaluates it) | — |
| R18 | dz = −a₀H dτ, a₀(τ−τ_init) = −∫dz/H | `BackgroundModel.py:53-58` (`da0_tau_dz = -1.0/H`) | AGREES | code reading; sign and orientation match (integration runs z_init → z_stop with z decreasing) |
| R19 | τ→0 as z→∞ convention | realised through the R20 initial condition | AGREES | `TK_05(b)`: code τ_init vs ∫_z^∞ dz/H |
| **R20** | a₀τ₁ = (3M_P²/ρ₁)^{1/2}(1+z₁) | `BackgroundModel.py:61-64` | **AGREES with R20; R20 itself assumes radiation domination** | `TK_05(b)`: identity `sqrt(3)M_P(1+z)/sqrt(ρ) ≡ (1+z)/H` verified to 7e-21; deviation from the exact integral 8.5e-4 at z=1e6 → 8.6e-10 at z=1e12 (TK-6) |
| R21 | R14 in Fourier space | same as R14 | AGREES | `TK_01` |
| R22 | ϑ equation before choosing P | — | no explicit counterpart (its two brackets appear separately as R23 and R27) | `TK_02` reproduces R27 from it |
| **R23** | 2P′/P = −(ε−3(1+c_s²))/(1+z); P = P₀√(H_init/H)exp[3/2∫(1+c_s²)dz/(1+z)] | friction: `TkWKBIntegration.py:49` (`(3/2)(1+cs2)/(1+z)`, initial state 0 at z_init, `WKB_phase_function.py:567`); amplitude √(H_init/H): `TkWKBIntegration.py:494-495` | **AGREES** | code reading (dF/dz is exactly the R23 integrand); `TK_04` reconstructs T_k from P·ω^{-1/2}·sin θ and matches the exact Bessel solution to 4e-4–6e-3 of the local envelope, 3 e-folds inside the horizon |
| **R24** | P′/P and P″/P − (P′/P)² | consumed in `raw_sin_coeff`, `TkWKBIntegration.py:438-445` | **AGREES** | `TK_02` "R24 check: 0" (with 2c_sc_s′ = w′ from R28); and the matching condition α = (T′ − T[(P′/P) − ½d lnω/dz])/√ω expands to the coded expression exactly |
| R25 | normal-form ϑ equation | equivalent to R27; no separate code | AGREES | `TK_02` (R27 definition ≡ compact form) |
| R26 | P with P₀ dropped | `TkWKBIntegration.py:494` gives H_ratio=1 and `friction_sample[0]`=0 at z_init, i.e. P(z_init)=1 | AGREES | code reading |
| **R27** | boxed ω_eff² | `ComputeTargets/WKB_Tk.py:4-25` | **AGREES** | `TK_02`: `code Tk_omegaEff_sq − R29 = 0`, and R27's *definition* (P″/P + …) − R27 compact form = 0 |
| R28 | c_s² = w(z), 2c_sc_s′ = w′, w and w′ for m+r | `LambdaCDM.wPerturbations:215-224`, `d_wPerturbations_dz:226-239` | AGREES | `TK_05(a)`: rel diff 1.7e-15 (w) and ≤3.5e-5 (w′ — the 3.5e-5 is cancellation in the *sympy reference* at z=1e14, not in the code, which uses the pre-simplified closed form) |
| **R29** | ω_eff² in terms of w with a₀ explicit | `WKB_Tk.py:17-23` | **AGREES** | `TK_02` (residual 0) |
| **R30 (corrected)** | 2ω_eff dω_eff/dz | `WKB_Tk.py:28-69` | **AGREES — the 641bb51 fix is present and correct** | `TK_02`: `code numerator − d(ω_eff²)/dz = 0` symbolically (differentiating the code's own `Tk_omegaEff_sq` with dH/dz = Hε/(1+z)). The pre-fix expression differs by exactly `+9(1+w)w′/(4(1+z)²)`, i.e. the (3/2)(1+w) → 3(1+w) slip recorded at R30 |
| R31 | w″(z) for m+r | `LambdaCDM.d2_wPerturbations_dz2:241-254` | AGREES | `TK_05(a)` (same cancellation caveat as R28) |

Checks section (§4) and Corrections (§5) of spec 01 concern the reading of the pages, not the code;
the one substantive item, Q7/R30, is covered above. Open questions Q1, Q3, Q5, Q6, Q9, Q10 have no
code counterpart. Q2 and Q8 are settled and consistent with the code (Q8 explicitly: the code uses
the R29 form `w*(k/a₀)²/H²`, never `a → a₀` inside the (1+z)^{-2} bracket). Q4 is exactly the split
that produces TK-1.

---

## 2. Findings in detail

### TK-1 — DEFECT: `LambdaCDM_GenericEOS.wPerturbations` includes ρ_Λ in the denominator

**Spec.** Spec 01's Tier 3 block: "*Wherever this group writes c_s² it means w(z) of the perturbed
fluid: δp = c_s²δρ with Λ unperturbed. In the code this is `wPerturbations(z)` (radiation + matter,
Λ excluded)*". Spec 03 §0.5 repeats: "`wPerturbations(z)`, Λ unperturbed". Spec 01 R28 writes the
Λ-free form
w(z) = w_r Ω_r(1+z)/(Ω_m + Ω_r(1+z)).
(Spec 01 R16 as transcribed literally *does* carry Ω_cc in the denominator; Q4 flags this as an
unresolved inconsistency on the page, and the author's sign-off resolves it in favour of "Λ
unperturbed". The GenericEOS code is a faithful transcription of the literal R16, which is almost
certainly how the defect arose.)

**Code.** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:281-292`:

```python
def wPerturbations(self, z: float) -> float:
    rho = self._rho_fluid(z)
    T = rho["T"]
    # perturbations w(z) includes contributions from radiation and matter, but not the cosmological
    # constant, which we take not to have perturbations. ...
    numerator = self._eos.w(T) * rho["radiation"]
    denominator = self.rho(z)          # <-- rho_m + rho_r + rho_cc
    return numerator / denominator
```

`self.rho(z)` is defined at line 250-257 as `rho["matter"] + rho["radiation"] + rho["lambda"]`. The
comment states the intent (Λ excluded) but the denominator contradicts it. The sibling class does it
correctly — `CosmologyModels/LambdaCDM/LambdaCDM.py:215-224` has
`denominator = self.omega_m + self.omega_r*one_plus_z`, which after multiplying through by (1+z)³ is
ρ_m + ρ_r with no ρ_Λ.

**Measurement** (`TK_06`… actually `TK_05` section (d), Planck2018 Ω values, Ω_m=0.3111,
Ω_cc=0.6889, Ω_r=9.139e-5):

| z | w_P, Λ-excluded (correct) | w_P, Λ-included (GenericEOS form) | ratio |
|---|---|---|---|
| 0 | 9.7892e-05 | 3.0460e-05 | 3.214 |
| 0.5 | 1.4682e-04 | 8.8667e-05 | 1.656 |
| 1 | 1.9573e-04 | 1.5331e-04 | 1.277 |
| 2 | 2.9350e-04 | 2.7128e-04 | 1.082 |
| 5 | 5.8649e-04 | 5.8055e-04 | 1.010 |
| 10 | 1.0737e-03 | 1.0719e-03 | 1.002 |
| 100 | 9.6050e-03 | 9.6050e-03 | 1.000 |

**Consequence.** `wPerturbations` is the c_s² of the transfer-function ODE
(`TkNumericIntegration.py:62,74-78`), of ω_eff² (`WKB_Tk.py:9,17-23`), of the Liouville–Green
friction integrand (`TkWKBIntegration.py:47-49`) and — via the spline stack, since
`LambdaCDM_GenericEOS` supplies no analytic `d_wPerturbations_dz`/`d2_wPerturbations_dz2` — of w′
and w″ in `Tk_d_ln_omegaEff_dz`. It also feeds `compute_analytic_T(..., wPerturbations, ...)`
(`TkNumericIntegration.py:436`). The absolute size of c_s² at z≲2 is ~1e-4, so the *absolute*
change in the ODE coefficients is small; but the friction term carries 3(1+c_s²) and the mass term
3(1+c_s²), so what is actually wrong at the 1e-4 level is the departure of the late-time coefficients
from their matter-dominated values, i.e. exactly the small radiation correction the quantity exists
to describe. `wBackground` is unaffected in both classes (both correctly divide by the total ρ and
put p_Λ = −ρ_Λ in the numerator). Only the `LambdaCDM_GenericEOS`/QCD models are affected; the plain
`LambdaCDM` models are correct.

### TK-2 — DEFECT: `TkWKBValue`'s `_w` analytic oracles alias the `_rad` ones

`ComputeTargets/TkWKBIntegration.py:630-636`:

```python
    @property
    def analytic_T_w(self) -> Optional[float]:
        return self._analytic_T_rad          # should be self._analytic_T_w

    @property
    def analytic_Tprime_w(self) -> Optional[float]:
        return self._analytic_Tprime_rad     # should be self._analytic_Tprime_w
```

The constructor (lines 551-554, 571-572) stores the correct `_analytic_T_w`/`_analytic_Tprime_w`
computed at lines 482-485 from `wPerturbations(z)`, so the values are computed and then discarded on
read. `Datastore/SQL/ObjectFactories/TkWKBIntegration.py:613-614` writes
`"analytic_T_w": value.analytic_T_w` — i.e. **the persisted `analytic_T_w`/`analytic_Tprime_w`
columns of the TkWKB value table contain the radiation-oracle values**, duplicating
`analytic_T_rad`/`analytic_Tprime_rad`. `extract_TkWKB_data.py:142,180-181,201,243-244` plots them.

Consequence: diagnostics only. `TkNumericValue` (`TkNumericIntegration.py:529-535`) has the correct
properties, and `ComputeTargets/QuadSource.py:111-112,130-131` reads from `TkNumericValue`, so the
source-term / one-loop chain is unaffected. Detectable in an existing database as
`analytic_T_w == analytic_T_rad` on every TkWKB row.

### TK-3 — DEFECT: WKB-validity warning at handover is unreachable (missing `fabs`)

`ComputeTargets/TkWKBIntegration.py:356-362`:

```python
        WKB_criterion_init = d_ln_omega_WKB_init / sqrt(omega_WKB_sq_init)
        if WKB_criterion_init > 1.0:
            print(f"!! Warning (TkWKBIntegration) ...")
```

Every other use of the same criterion takes the absolute value: line 442-444, line 490-492, and
`Quadrature/integrators/WKB_phase_function.py:89,285,662`. `TK_08` evaluates the sign directly on a
constant-w background: d ln ω_eff/dz is negative throughout the sub-horizon regime (e.g. radiation,
k=1e4: −0.369 at z=1000, −0.035 at z=100, −0.0038 at z=10 in units of the criterion), because
ω_eff ≈ c_s k/H grows as z decreases. So `WKB_criterion_init` is always negative and the branch is
dead. Consequence: no stored number changes; the intended warning about starting the WKB branch
where the approximation is poor is silently lost. The corresponding hard error in
`WKB_phase_function.py:662-666` *does* use `fabs` and would raise, so the condition is not
unguarded — only the earlier, softer warning is.

### TK-4 — DEFECT: mis-spelled attribute in `TkWKBIntegration.__init__`

Line 115 (payload-is-None branch) initialises `self._init_efolds_suph = None`, but the property
(line 284-288) and `store()` (line 413) both use `self._init_efolds_subh`. Reading
`init_efolds_subh` before `store()` raises `AttributeError` instead of the intended
`RuntimeError("init_efolds_subh has not yet been populated")`, and `_init_efolds_suph` is dead. No
stored number changes.

### TK-5 — DEFECT (accuracy at grid ends): `_build_derivative` spline stack

`ComputeTargets/BackgroundModel.py:120-149`. When the cosmology supplies an analytic method the code
takes it (line 126-127); otherwise it builds a not-a-knot cubic `make_interp_spline` in log(1+z),
differentiates it, and divides by (1+z). `d2_lnH_dz2` is built from the `d_lnH_dz` *samples*,
`d3_lnH_dz3` from the `d2_lnH_dz2` samples, and `d2_wPerturbations_dz2` from the
`d_wPerturbations_dz` samples — a stack up to three deep.

Which cosmologies take which branch:

| attribute | `LambdaCDM` | `LambdaCDM_GenericEOS` |
|---|---|---|
| `d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3` | analytic | **spline (1×, 2×, 3× stacked)** |
| `d_wPerturbations_dz`, `d2_wPerturbations_dz2` | analytic | **spline (1×, 2× stacked)** |

So this affects the QCD/GenericEOS models only. `TK_06` runs the identical code path on the
Planck2018 `LambdaCDM` background (where exact derivatives are available as a reference) on the grid
`main.py` actually builds — z ∈ [0.1, 1e12], 100 samples per decade, n = 1300:

| quantity | stack depth | z=0.1 (end) | 2nd point | 3rd point | interior median | z=1e12 (end) |
|---|---|---|---|---|---|---|
| ε | 1 | 1.25e-05 | 3.25e-06 | 8.13e-07 | 6.78e-13 | 3.80e-13 |
| ε′ | 2 | 9.40e-04 | 2.03e-06 | 6.83e-05 | 1.42e-06 | (ref→0)† |
| ε″ | 3 | **3.01e-01** | 5.29e-02 | 1.28e-02 | 6.33e-05 | (ref→0)† |
| w′ | 1 | 1.74e-06 | 4.57e-07 | 1.19e-07 | 1.04e-09 | 1.08e-06 |
| w″ | 2 | **3.78e-01** | 1.04e-03 | 2.60e-02 | 1.82e-08 | 5.85e-05 |

† at high z the background is radiation-dominated, ε → 2 exactly and ε′, ε″ → 0, so the *relative*
error there is meaningless; the absolute error is what matters and is bounded by the ω_eff² test
below.

The end bias is 4–7 orders of magnitude worse than the interior, is worst at the **low-z** end
(z=0.1), and worsens with stack depth — the classic not-a-knot end-condition error, amplified by
each differentiation. `TK_07` measures the propagation into the two quantities the WKB branch
consumes, by building two `ModelFunctions` over the same background (analytic vs spline stack) and
evaluating `Tk_omegaEff_sq` / `Tk_d_ln_omegaEff_dz`:

| z | rel. diff in ω_eff² | rel. diff in d ln ω_eff/dz |
|---|---|---|
| 0.1 (low-z end) | 2.1e-07 | 2.6e-04 |
| 0.12 (2nd) | 6.8e-10 | 1.4e-06 |
| 1079 (interior) | 2.4e-13 | 1.1e-10 |
| 9.6e11 (near top) | 6.0e-08 | 4.5e-06 |
| 1e12 (high-z end) | 8.7e-07 | 5.7e-05 |

**Consequence.** Bounded: ≤9e-7 in ω_eff² and ≤6e-5 in d ln ω_eff/dz, and only within the first two
or three grid points of each end (three to five orders of magnitude worse than the interior).
`d_ln_omega_dz` enters `raw_sin_coeff` (`TkWKBIntegration.py:442`) at the numeric→WKB handover
redshift, which sits 3–6 e-folds inside the horizon — deep in the grid interior — so the practical
exposure is small. It matters if any consumer evaluates ε′, ε″, w′ or w″ at or adjacent to the grid
boundary. Two contributing choices worth noting: (i) the derivative is taken of *sampled values* on
the production grid rather than of a finer private grid, so the accuracy is tied to
`source_samples_log10z`; (ii) nothing pads the grid, whereas `LambdaCDM_GenericEOS._build_T_z_spline`
(line 180-197) deliberately does add a 5% buffer for exactly this reason, and
`DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2` extends into the future so that derivatives at z=0 are
clean. The same padding is not applied in `_build_derivative`.

### TK-6 — CONVENTION (measured): the conformal-time initial condition

`ComputeTargets/BackgroundModel.py:61-64`:

```python
        rho_init = cosmology.rho(z_init)
        tau_init = sqrt(3.0) * cosmology.units.PlanckMass / sqrt(rho_init) * (1.0 + z_init)
```

This is spec 01 **R20** verbatim, a₀τ₁ = (3M_P²/ρ₁)^{1/2}(1+z₁). Since ρ = 3H²M_P² identically
(`LambdaCDM.Hubble:113-121`), it equals (1+z_init)/H(z_init) — `TK_05(b)` verifies the identity to
7e-21. R20 is derived *assuming radiation domination at and before z₁*; the code evaluates it with
the total ρ, so it is an approximation to the exact R19 statement a₀τ = ∫_z^∞ dz′/H(z′), with
fractional error ≈ (1/3)Ω_m/(Ω_r(1+z_init)):

| z_init | code τ_init | exact ∫_z^∞ dz/H | rel. diff |
|---|---|---|---|
| 1e6 | 0.46270247 | 0.463095573 | 8.49e-04 |
| 1e8 | 4.63481914e-03 | 4.63485859e-03 | 8.51e-06 |
| 1e10 | 4.63489729e-05 | 4.63489768e-05 | 8.51e-08 |
| 1e12 | 4.63489807e-07 | 4.63489807e-07 | 8.64e-10 |

Because the ODE `da0_tau_dz = -1/H` (line 56) is then exact, the error enters as a **constant
absolute offset** in a₀τ at every lower redshift: at z_init=1e6 the offset is −3.9e-4 Mpc against
a₀τ(z=0.1) ≈ 1.4e4 Mpc, i.e. 3e-8 relative by the end of the run, but ~8e-4 relative in the first
decade below z_init. τ is used only by the analytic oracles (`analytic_Tk`, `analytic_Gk`) and by
`main.py:415`'s Levin-domain sizing, not by the numeric ODEs, so this is a bounded oracle-accuracy
statement rather than a physics error. Recorded because spec 01 §3.3 gives R20 as a
radiation-domination result while the code applies it unconditionally.

### TK-7 — CONVENTION (measured): the super-horizon normalisation T→1

`ComputeTargets/TkNumericIntegration.py:376-377` passes `initial_value=1.0, initial_deriv=0.0`, and
`main.py:506` sets z_init = `k_exit.z_exit_suph_e5`, i.e. k/(aH) = e^{-5} = 6.74e-3
(`CosmologyConcepts/wavenumber.py:514-518` solves log(k(1+z)/H) = −N, which is a₀-invariant). Spec 01
§2.8 requires φ → φ*_k as kc_sη → 0, so T=1, T′=0 is exact only in the limit.

`TK_03` measures the resulting error by comparing the integration of the *actual* coded RHS against
the exact solution R11:

- radiation, k=1e4, z_init at exactly 5 e-folds super-horizon, z_stop at 6 e-folds sub-horizon:
  worst relative difference in T over the whole run **2.53e-06**; for w=0.2, **2.05e-06**.
- the residual is a constant multiplicative offset (a mixture of the correct J solution with a small
  admixture from the imperfect initial data), not a growing error: pushing z_init back scales it as
  (k c_s τ_init)² — measured ratios 0.063, 0.076 for successive ×4 increases in (1+z_init) (exact
  (1/16) = 0.0625; the third step reaches the 1e-12 integrator tolerance floor).

This confirms both that the coded ODE has R11 as its exact solution and that the normalisation error
budget at the production z_init is a few parts in 1e6.

**Identity of the normalisation (question 5 of the assignment).** T_k → 1 corresponds to
φ_k → φ*_k, and spec 03 §0.4 fixes φ* = 3(1+w*)/(5+3w*) ζ*. The constant
c_* = 3(1+w*)/(5+3w*) is applied **nowhere** in the code: `grep` for `5 + 3`/`5.0 + 3.0` over the
repository finds exactly one hit, `ComputeTargets/QuadSource.py:35`, which is the *different*
(5+3w)/(3(1+w)) coefficient of the source term f evaluated at `wBackground(z')` (spec 03 §0.2), not
c_*. `ComputeTargets/OneLoopIntegral.py:97-110` `compute()` is a stub — no 648π², no
((1+w*)/(5+3w*))⁴. So the ζ*→φ* translation is neither applied nor double-applied anywhere in the
T_k path; it is outstanding build work in the one-loop layer, exactly as spec 03 §0.2 records. **This
is consistent, not a defect**, but it means the stored T_k, T_k^WKB and QuadSource quantities are all
in φ/φ* units and must not be interpreted as being in ζ* units.

### TK-8 — CONVENTION: phase direction, the sign fix-up, and `theta_div_2pi`

**(a) Phase direction.** `Quadrature/integrators/WKB_phase_function.py:96-97` sets
`dtheta_dz = +omega_value` with `state = [0.0]` at z_init (line 101) and `t_span = (z_init, z_target)`
with z_target < z_init. Since ω_eff > 0 and z *decreases* along the integration, **θ decreases and
becomes negative** as time moves forward. `TK_04` confirms this directly (θ runs 0 → −1710 over the
test range). This is the convention spec 02 §0.2 item 2.8 signs off on ("with the page's (and the
code's) convention dΘ/dz = +ω_eff … the phase *decreases* with u"), and it is internally consistent:
`WKB_mod_2pi` (`LiouvilleGreen/WKBtools.py:11-23`) enforces a **negative** mod-2π remainder, matching.
Stage 2 is consistent too: with θ = θ_init + ω_init(1+u)Q and
`dQ_du = -omega/omega_init/(1+u) - Q/(1+u)` (line 293), dθ/du = ω_init(Q + (1+u)Q′) = −ω, i.e.
dθ/dz = +ω, and Q(0)=0 gives θ(u=0)=θ_init. So the assignment's expectation that "θ increases in the
direction the code integrates" is **not** what happens — it decreases — but that is the documented
convention and nothing depends on the sign, since θ is only ever consumed through sin/cos.

**(b) The `sgn` fix-up is a provable no-op.** `ComputeTargets/TkWKBIntegration.py:449-459`:

```python
        deltaTheta = atan2(raw_cos_coeff, raw_sin_coeff)
        B = sqrt(raw_cos_coeff*raw_cos_coeff + raw_sin_coeff*raw_sin_coeff)
        sin_deltaTheta = sin(deltaTheta)
        sgn_sin_deltaTheta = +1 if sin_deltaTheta >= 0.0 else -1
        sgn_T = +1 if self._T_init >= 0.0 else -1
        self._cos_coeff = 0.0
        self._sin_coeff = sgn_sin_deltaTheta * sgn_T * B
```

With α = `raw_sin_coeff`, β = `raw_cos_coeff` = √ω_init·T_init: δ = atan2(β,α) gives
sin δ = β/B, so sgn(sin δ) = sgn(β) = sgn(T_init) (as ω_init > 0), hence
`sgn_sin_deltaTheta * sgn_T = +1` identically, including the T_init = 0 edge case where both are +1
by the `>= 0.0` convention. `TK_04` reports `sgn correction factor = +1` in all six configurations
tested. The correct coefficient is +B: at z_init, θ+δ = δ and
ω^{-1/2}·B·sin δ = β/√ω_init = T_init ✓.

**(c) Matching condition.** α = (T′_init − T_init[(P′/P) − ½ d lnω/dz])/√ω_init with R24's
P′/P = −½(ε−3(1+c_s²))/(1+z) expands to exactly
`(Tprime_init + (T_init/2)*(d_ln_omega + (eps - 3(1+cs2))/(1+z)))/sqrt(omega)` —
`TkWKBIntegration.py:438-445`, verbatim. AGREES.

**(d) Stored `theta_div_2pi` carries an arbitrary integer offset.**
`LiouvilleGreen/WKBtools.py:103-108` rebases the div-2π shift to the first sample
(`theta_div_2pi_shift_base`), so the stored `theta_div_2pi` (and hence the `TkWKBValue.theta`
property, line 602-604) is θ/2π only up to a constant integer per object. Differences of θ within
one object are exact; the absolute value is not meaningful. `T_WKB` (line 499-506) correctly uses
only `theta_mod_2pi`. Recorded as a convention.

**(e) Amplitude form.** `norm_factor = sqrt(H_ratio/omega)` with `H_ratio = H_init/H`, times
`exp(friction_sample[i])` — i.e. exactly P·ω^{-1/2} with P from R23/R26. `TK_04` validates the whole
assembly against the exact Bessel solution starting 3 e-folds inside the horizon:

| w | worst |err|/envelope over the run |
|---|---|
| 1/3 | 3.67e-04 |
| 0.2 | 5.60e-03 |
| 0.5 | 4.36e-03 |

(k-independent, as it must be for constant w. The raw *relative* differences reach 2%–50% but only
at zero crossings of T, where the relative measure is meaningless — hence the envelope-normalised
column.) These are the expected Liouville–Green truncation errors at 3 e-folds inside the horizon,
not formula disagreements.

---

## 3. Numerical convention notes

1. **Independent variable is z, not log(1+z).** The transfer-function ODE
   (`numeric_with_phase_cut.py:126-141`), the phase function (`WKB_phase_function.py:108-118`), the
   friction integral (`:569-583`) and the background τ integration (`BackgroundModel.py:68-77`) all
   step in z. log(1+z) appears only in (i) `_build_derivative`/`_build_func` spline abscissae, (ii)
   the horizon-crossing solve (`wavenumber.py:401-403`), (iii) `ε = d ln H/d ln(1+z)` as a *reading*
   of R15, and (iv) the unresolved-oscillation diagnostic, which correctly converts
   `grid_spacing = (1+z)*delta_logz` before comparing with the Δz wavelength 2π/ω_eff
   (`Quadrature/supervisors/numeric.py:73-90`). `ZSplineWrapper.__call__` with `deriv=True`
   (`spline_wrappers.py:58-61`) divides the log-spline derivative by (1+z), so every quantity named
   `d…_dz` really is a z-derivative — verified against `LambdaCDM`'s analytic `d_lnH_dz` in
   `TK_05(a)`.
2. **a₀ is absorbed and the covariance test passes by inspection.** In this whole subsystem k appears
   only as (i) `k/H` in `TkNumericIntegration.py:70` and `WKB_Tk.py:14,41`, (ii) `k*cs*tau` in
   `analytic_Tk.py:8` and the overall `cs*k/H` of `compute_analytic_Tprime`, (iii) `(1+z)*k/H` in the
   e-folds diagnostics (`BackgroundModel.py:261`, `TkNumericIntegration.py:335`,
   `wavenumber.py:401`). The stored `k.k` is k/a₀ and the stored `tau` is a₀η, so under
   a₀ → λa₀ with comoving k → λk and η → η/λ every one of these is invariant. No stray power of a₀
   survives. In particular the code's `wPerturbations * (k/a₀)²/H²` is exactly R21's
   c_s²k²/((1+z)²a²H²), which is spec 01 Q8's warning taken correctly.
3. **Sign of ε.** ε = +(1+z)d ln H/dz > 0 for a decelerating universe, matching spec §2.3. Radiation
   gives ε=2, matter ε=3/2, as required.
4. **`wBackground` vs `wPerturbations`.** Both classes' `wBackground` includes p_Λ = −ρ_Λ in the
   numerator and ρ_Λ in the denominator (`LambdaCDM.py:208-211`,
   `LambdaCDM_GenericEOS.py:276-279`) — the spec 03 §0.5 convention. The transfer-function ODE and
   ω_eff² use `wPerturbations` throughout; `wBackground` appears in this subsystem only where
   `BackgroundModel` samples it for storage. That division of labour is exactly what spec 01's Tier 3
   block prescribes. The one departure is TK-1.
5. **`GenericEOSBase.w(T) = 4G_S/(3G) − 1`** (`GenericEOS.py:56-72`) is p/ρ for a relativistic bath
   with p = sT − ρ, s = (2π²/45)g_S T³, ρ = (π²/30)g T⁴; correct, and reduces to 1/3 when g_S = g.
   Its own docstring records that it is not valid after e⁺e⁻ annihilation, which is a modelling
   caveat outside spec 01.
6. **Phase sign.** See TK-8(a). θ < 0 and decreasing; `WKB_mod_2pi`'s negative-remainder convention
   matches.

---

## 4. Scripts run

All under
`docs/spec-code-audit/scripts/`,
run with `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python`.

| Script | What it checks | Result |
|---|---|---|
| `TK_01_Tk_ODE.py` | sympy: R5 + R12/R13 → R14/R21, and the coded RHS of `TkNumericIntegration.py:73-80` | all residuals identically 0; variable change is z, coefficient of φ″ is a₀²H² |
| `TK_02_omegaEff.py` | sympy: R24; R27 definition ≡ R27/R29 compact; `Tk_omegaEff_sq` ≡ R29; `Tk_d_ln_omegaEff_dz` numerator ≡ d(ω_eff²)/dz | all 0; pre-641bb51 form differs by exactly 9(1+w)w′/(4(1+z)²) |
| `TK_03_numeric_vs_analytic.py` | integrates the real `RHS` on constant-w stand-ins vs `compute_analytic_T`/`Tprime` (R11) | worst rel. diff 2.5e-06 at the production z_init; residual scales as (kc_sτ_init)² (ratio 0.063 per ×4 in 1+z_init, exact 1/16) |
| `TK_04_WKB_reconstruction.py` | reassembles T_WKB exactly as `TkWKBIntegration.store()` does (real `Tk_omegaEff_sq`, `Tk_d_ln_omegaEff_dz`, `friction_RHS`, phase, coefficients) vs exact Bessel | |err|/envelope ≤ 5.6e-3 starting 3 e-folds sub-horizon; `sgn` factor = +1 in every case; θ negative and decreasing |
| `TK_05_background_derivatives.py` | `LambdaCDM` analytic methods vs sympy (R15/R16/R28/R31); τ_init (R20) vs exact ∫dz/H (R19); first look at the spline path; `wPerturbations` denominators | ≤2.9e-14 for H/w quantities; τ_init 8.5e-4 → 8.6e-10 for z_init 1e6 → 1e12; wPerturbations ratio 3.214 at z=0 |
| `TK_06_spline_end_bias.py` | `_build_derivative` on `main.py`'s actual grid (z∈[0.1,1e12], 100/decade, n=1300) vs analytic ε, ε′, ε″, w′, w″ | end/interior degradation 4–7 orders of magnitude; ε″ 3.0e-1 at the z=0.1 end vs 6.3e-5 interior |
| `TK_07_omegaEff_spline_impact.py` | same, but propagated into `Tk_omegaEff_sq` and `Tk_d_ln_omegaEff_dz` | ≤8.7e-07 in ω_eff², ≤5.7e-05 in d ln ω/dz, confined to the outermost 2–3 grid points |
| `TK_08_criterion_sign.py` | sign of `Tk_d_ln_omegaEff_dz` and reachability of the `TkWKBIntegration.py:357` warning | negative throughout; the warning is dead code |

`TK_03`'s `FakeModel`/`DummySupervisor` are imported by `TK_04` and `TK_08`. No repository file was
modified.
