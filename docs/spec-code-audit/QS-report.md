# QS — audit of the quadratic source term (`docs/spec/03-source-term.md` → `ComputeTargets/QuadSource.py`)

Agent QS. Repository at HEAD `e9a5539`, branch `main`, clean. Read-only; nothing in the repository
was modified. Python `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python`.

Scope as assigned: spec 03 (both readings) → `ComputeTargets/QuadSource.py`,
`ComputeTargets/TkNumericIntegration.py` (only far enough to fix the meaning of `T`, `Tprime`),
`ComputeTargets/analytic_Tk.py`, `Datastore/SQL/ObjectFactories/QuadSource.py` (column list only),
`main.py` scheduling. `ComputeTargets/QuadSourceIntegral.py` was read only for the prefactor
bookkeeping of §2 QS-4 (a separate agent audits that file).

---

## 0. Summary of findings

| ID | Severity | One line | Code |
|---|---|---|---|
| QS-1 | **AGREES** | `source_function` is *algebraically identical* to the spec-03 R22 / §0.3 kernel `f`, term by term, including which `w` sits in which coefficient | `ComputeTargets/QuadSource.py:35-45` |
| QS-2 | **AGREES** | The derivative fed into `f` is the plain `dT/dz`, exactly as the spec's `(T-(1+z)dT/dz)` requires; no Jacobian factor is needed and none is applied | `TkNumericIntegration.py:32,68,92`; `analytic_Tk.py:19-36`; `QuadSource.py:108,127` |
| QS-3 | **AGREES** | `w` inside `f` is `wBackground(z)` (Λ included), evaluated at the source sample redshift — spec 03 §0.5 / Q5 | `QuadSource.py:142-162` |
| QS-4 | **UNVERIFIED (build gap, not a defect)** | Of the spec's overall prefactor chain, *only* `f` and the `(1+z)/(1+z')·dz'/H²` weight exist in code. `q²_phys`, `Q_s`, the `36(...)²` per `h_s` and the whole `648π²((1+w*)/(5+3w*))⁴ ∫dq/q ∫dθ sin⁵θ 𝒫*𝒫*/r³` live nowhere. Table in §2 | `OneLoopIntegral.py:98-112` (stub) |
| QS-5 | **DEFECT (F1, known; here quantified)** | `_create_functions` splines the *oscillating* source in `log(1+z)` on a 100-per-decade grid. Measured: error reaches **100 % of the local envelope** once `k c_s η ≳ 300`, and pointwise relative error `10⁴–10⁷`. The shipped grid runs to ~4.5×10⁶ cycles | `QuadSource.py:294-309` |
| QS-6 | **DEFECT (new)** | `compute_quad_source` iterates the **full** source grid but `TkNumericIntegration.z_sample` is truncated at *both* ends (and further shortened by `mode="stop"`). Reproduced: unguarded `IndexError` at `QuadSource.py:87`. Only the *leading* (high-z) shortfall is handled | `QuadSource.py:82-102` vs `main.py:505-507,528` |
| QS-7 | **CONVENTION / oracle hygiene** | The `analytic_source_w` column mixes two `w`'s: `T` from a constant-`w = wPerturbations(z)` solution, coefficients from `wBackground(z)`. Identical above `z≈100`; coefficient ratio 1.12 at `z=1`, 1.70 at `z=0.1` | `TkNumericIntegration.py:436-439` + `QuadSource.py:142-162` |
| QS-8 | **CONVENTION** | `analytic_source_rad`/`analytic_source_w` are *not* independent oracles of `f` — they are the same `source_function` fed analytic `T`,`T'`. They test `T`, not `f` | `QuadSource.py:147-162` |
| QS-9 | **AGREES** | `f(q,r)=f(r,q)` exactly (symbolically and bit-for-bit); `main.py` schedules only unordered pairs `q.k ≤ r.k`, so the R31 symmetry reduction is respected and nothing is double counted | `QuadSource.py:35-45`; `main.py:878,952-958` |
| QS-10 | **CONVENTION (cosmetic)** | `ZSplineWrapper` is constructed with `label="T_k"` inside `QuadSource`, so its out-of-bounds message reads `GkSource.function: evaluated T_k out of bounds` for a *source* spline | `QuadSource.py:304` |
| QS-11 | **AGREES (measured)** | `compute_analytic_Tprime` loses precision to cancellation super-horizon (rel. error `≈1e-16/x²`·O(10), reaching O(1) at `x≲5e-7`), but this is **harmless inside `f`**: measured `f` accuracy is `≤1e-13` everywhere tested | `analytic_Tk.py:28-36` |

No defect was found in the *formula* the source term implements. The two defects are a sampling
defect (QS-5) and a grid-coverage defect (QS-6); both sit exactly where the reconciliation document
already says the refactor onto `TkWKBIntegration` has to happen.

---

## 1. Formula map

Result-bearing `R##` of spec 03 with a counterpart in the audited files. `MAIN` 10 results (R1–R11)
and the `NUM` 03 derivation steps R12–R21, R23–R27 and R29–R35 are *derivation* or *post-source*
results; whether each has a code counterpart is stated explicitly.

| spec result | meaning | code location | status | how checked |
|---|---|---|---|---|
| R1–R11 (`MAIN` 10) | perfect-fluid `T^a_b`, `ψ=φ`, `v_i(φ'+ℋφ)`, quadratic stress `4M_P⁴/(a⁴(ρ₀+p₀))∂φ̃∂φ̃` | **none** | expected — the code starts from R22; these are the algebra that *produces* the `2/(3(1+w₀))` coefficient | inspection |
| R12 | starting sourced tensor equation, incl. `8M_P²/(ρ₀+p₀)` velocity term | **none** (folded into R22) | expected | inspection |
| R13 | `3H²M_P²=ρ₀`, `d/dη = -(1+z)aH d/dz` | `BackgroundModel.py:56` (`d(a₀τ)/dz=-1/H`, equivalent form); used implicitly by `analytic_Tk` | AGREES | `QS_02`, `QS_02b`: `compute_analytic_Tprime = dT/dz` to 60-digit reference |
| R14, R17, R18, R19 | `χ_s=ah_s` rescaling, `a''/a=(aH)²(2-ε)`, redshift form | `TkNumericIntegration.RHS` is the *scalar* analogue; the tensor operator lives in `GkNumericIntegration` (other agent) | out of scope | inspection |
| R15 | `e_s^{lm}q_l r_m → -e_s^{lm}q_l q_m`, sign cancels R12's minus | **none** | deferred to `OneLoopIntegral` (stub) | grep: no polarisation algebra anywhere in `ComputeTargets/` |
| R16 | `φ'+ℋφ = aH[φ-(1+z)dφ/dz]` | the `(T-(1+z)dT/dz)` grouping inside `source_function` | AGREES | `QS_01` (sympy, exact) |
| R20 / §0.4 | `φ_k = 3(1+w*)/(5+3w*) T_k ζ*_k`, `T_k→1` at `z_init` | `TkNumericIntegration.compute` sets `initial_value=1.0, initial_deriv=0.0` (`:376-377`); the `3(1+w*)/(5+3w*)` factor is **nowhere** | AGREES on `T_k→1`; the `ζ*→φ*` constant is a build gap (see QS-4) | inspection; cross-read spec 01 §2.8 |
| R21 | `Q_s(q) ≡ e_s^{lm}(k) q_l q_m` | **none** | build gap (QS-4) | grep |
| **R22 / §0.3** `f` | `f = T_qT_r + 2/(3(1+w₀))(T-(1+z)dT/dz)_q(T-(1+z)dT/dz)_r` | `QuadSource.py:23-51` `source_function` | **AGREES, exactly** | `QS_01` sympy: `spec_f - code_f == 0`; `QS_02(b)` numeric on the analytic radiation `T`, worst rel. diff `2.3e-16` |
| R22 prefactor `36(1+w*)²/(5+3w*)²` | `ζ*→φ*` squared, ×4 | **none** | build gap (QS-4) | grep for `36`, `1296`, `2592`, `648` in `ComputeTargets/` → no hits |
| R22 `1/(a H²(1+z)²)` | source normalisation of the `χ_s` equation | absorbed: the code's `Ḡ_k` is already the `h_s`-level unit-jump function, so only `1/H²` and `(1+z)/(1+z')` survive (R26) | AGREES | see R26/R28 row |
| R23, R24 | Green's-function solution for `χ_s`, then `h_s` with `a₀` made explicit and cancelled | `QuadSourceIntegral` (other agent) | out of scope | — |
| R25 / §0.1 | unit-jump causal `Ḡ_k`: `Ḡ(z',z')=0`, `dḠ/dz|_{z'}=+1` | `GkNumericIntegration` (other agent) | out of scope | — |
| **R26 / R28** `I_s` | `∫_z^{z_init} dz' Ḡ_k(z,z') (1+z)/(1+z') · Q_s/(a₀²H²(z')) · f(z')` | `QuadSourceIntegral.py:954-975`: `(1+z_response) · ∫ dlog(1+z') [G·f/H²]`; `Q_s/a₀²` **absent** | AGREES on the measure/Jacobian; `Q_s/a₀²` is a build gap | `QS_04(b)` sympy: `spec integrand·dz'/du − code integrand == 0` |
| R27 | dimension check | — | n/a | — |
| R29 | `Q_s/(a₀²H²) ∝ q²_phys/H²`; `a₀` bookkeeping | code works in `k/a₀` (`wavenumber.k`) and `a₀η` (`BackgroundModel` `tau`), both individually `a₀`-invariant under `a₀→λa₀`, comoving `k→λk`, `η→η/λ` | AGREES | inspection; `BackgroundModel.py:56,62-64` |
| R30, R31, R32 | Wick contraction, `1296`; symmetry collapse `→ 2 I_s²`, `2592` | R31's symmetry premise ("`f` symmetric, depends only on `|q|,|k−q|`") is **satisfied** by the code; the coefficients are **nowhere** | AGREES (symmetry); build gap (coefficients) | `QS_01`, `QS_02(d)` |
| R33 | `Q_± = (1/√2)q²_phys sin²θ{cos2φ,sin2φ}`, `∫cos²2φ dφ = π` | **none** | build gap | grep |
| R34, R35 / §0.3 | `648π²((1+w*)/(5+3w*))⁴ ∫dq/q ∫₀^π dθ sin⁵θ 𝒫*(q)𝒫*(r)/r³ {…}²` | `OneLoopIntegral.py` is a stub (`compute()` sets nothing, `:98-112`) | build gap; spec 03 §0.2 already labels this a build specification | inspection |
| §0.5 `w₀` vs `w*` vs `c_s²` | `w₀=wBackground(z')` inside `f`; `w*=w(z_init)` in the prefactor; `c_s²=wPerturbations` in the `T_k` equation | `QuadSource.py:142` uses `wBackground(z.z)`; `TkNumericIntegration.py:62,74-79` uses `wPerturbations` | AGREES | inspection; `QS_05(b)` measures how far apart the two `w`'s are |
| stored columns | `source`, `undiff`, `diff`, `analytic_{source,undiff,diff}_{rad,w}` | `Datastore/SQL/ObjectFactories/QuadSource.py:530-538` | AGREES | inspection |

---

## 2. Findings in detail

### QS-1 — `source_function` is exactly the spec's `f` (AGREES)

Spec 03 R22 / §0.3 (`NUM` 03 p. 4):

```
f(z' | q, k-q) = T_q T_r + 2/(3(1+w(z'))) · (T - (1+z')dT/dz')_q · (T - (1+z')dT/dz')_r
```

Code, `ComputeTargets/QuadSource.py:35-45`:

```python
undiff_part = (5.0 + 3.0 * w) / (3.0 * (1.0 + w)) * Tq * Tr
diff_part = (
    2.0 / (3.0 * (1.0 + w))
    * ( -one_plus_z * Tq * Tr_prime
        -one_plus_z * Tr * Tq_prime
        +one_plus_z_2 * Tq_prime * Tr_prime )
)
source_term = undiff_part + diff_part
```

`scripts/QS_01_sympy_f.py` gives `spec_f - code_f == 0` identically in `(T_q,T_r,T_q',T_r',z,w)`. The
regrouping is the single identity `1 + 2/(3(1+w)) = (5+3w)/(3(1+w))`: the code has moved the
`2/(3(1+w))·T_qT_r` cross term out of the "diff" bracket and into "undiff", which is why the stored
`undiff` column carries `(5+3w)/(3(1+w))·T_qT_r` rather than `T_qT_r`. The `undiff`/`diff` split is
therefore a **reporting** convention, not a physics statement; only `source = undiff + diff` is
spec-defined. This is exactly what spec 03 §0.2's "Code status" paragraph asserts, now verified
symbolically.

Numerically (`scripts/QS_02_deriv_and_f.py`, part b), evaluating both an independent transcription of
R22 and `source_function` on the analytic constant-`w` transfer function over
`w ∈ {1/3, 0.1, 0.5}`, `(q,r) ∈ {(1,1),(1,30),(30,30),(300,700)}`, `z ∈ {10⁵…1}`: worst relative
difference `2.3e-16` (float round-off).

Downstream consequence: none — the kernel is right.

### QS-2 — the derivative is `dT/dz` (AGREES)

The spec writes `dT/dz` (spec 03 §2.2: "From p. 2 onwards `z`-derivatives are always written out as
`d/dz`"). The code:

- `TkNumericIntegration.py:30-32` declares `state[1] = dT/dz`; `RHS` returns `[Tprime, dTprime_dz]`
  and `numeric_with_phase_cut.py:126-141` integrates `solve_ivp(..., t_span=(z_init, z_min))`, i.e.
  the independent variable **is** `z`. So `deriv_sample` → `TkNumericValue.Tprime` is `dT/dz`.
- `QuadSource.py:108,127` reads `Tq_.Tprime` and passes it straight to `source_function`. **No
  conversion factor is applied, and none is needed.**

For the analytic oracle, `analytic_Tk.py:19-36` returns `-(1/H) · dT/dτ` with `τ = a₀η`. Since
`BackgroundModel.py:56` defines `τ` by `dτ/dz = -1/H`, `-(1/H)dT/dτ = (dτ/dz)(dT/dτ) = dT/dz`. ✓
`scripts/QS_02b_deriv_mpmath.py` confirms this against a 50-digit `mpmath` derivative of the exact
`T(z)` on the exact constant-`w` background: agreement to `1e-16 … 1e-11` wherever the float64
bracket is not cancellation-limited (see QS-11).

Jacobian chain, for the record (`w`-general, `H = H₀(1+z)^{3(1+w)/2}`):

```
d/dη        = -(1+z) a H d/dz            (spec 03 R13)
d/d(a0 η)   = -H d/dz                    (a0 η, code's tau: BackgroundModel.py:56)
d/dlog(1+z) = (1+z) d/dz
```
so a `d/dlog(1+z)` reading of `Tprime` would be wrong by `(1+z)` and an `H⁻¹ dT/dη` reading by
`1/(1+z)` — factors of `10¹`–`10⁵` on the shipped grid, i.e. easily detectable. Measured at
`w=1/3, k=30, z=10³`: `dT/dz = 5.982e-08`, `(1+z)dT/dz = 5.988e-05`, `dT/dz/(1+z) = 5.976e-11`. The
code matches the first.

### QS-3 — which `w` multiplies which term (AGREES)

`QuadSource.py:142` evaluates `wBackground(z.z)` at the *source sample* redshift and passes it as the
single `w` used by both coefficients. This is spec 03 §0.5 / Q5 exactly: `w₀ = w(z')`,
`wBackground` (Λ included). It is also the physically required choice: the coefficient descends from
`2M_P²ℋ²/(a²(ρ₀+p₀))` with `3H²M_P² = ρ₀^tot`, so `1+w₀` must reconstruct `(ρ₀+p₀)/ρ₀` for the
*total* background — which is what `wBackground` is (`CosmologyModels/LambdaCDM/LambdaCDM.py:199-213`
puts `-Ω_cc` in the numerator and all three components in the denominator). `wPerturbations` would
be wrong here, and the code does not use it here.

No singularity risk: `1+wBackground = 0` requires `(4/3)Ω_r(1+z)⁴ + Ω_m(1+z)³ = 0`, i.e. only
`z = -1`.

### QS-4 — where each factor of the spec's prefactor chain lives (build gap)

Reference: spec 03 §0.3 build form (= R35 with the §0.2 coefficient and the completed measure).

| factor in the spec build form | where it is in the code | status |
|---|---|---|
| `f = T_qT_r + 2/(3(1+w(z')))(T-(1+z')T')_q(T-(1+z')T')_r` | `QuadSource.py:35-45` (`source_function`); both coefficients inside | **implemented** |
| `T_k → 1` at `z_init` | `TkNumericIntegration.py:376-377` (`initial_value=1.0, initial_deriv=0.0`) | **implemented** |
| `Ḡ_k(z,z')`, unit-jump causal in `z` | `GkNumericIntegration` / `GkSource` / `GkSourcePolicyData` | implemented (other agent) |
| `dz'` and `(1+z)/(1+z')` | `QuadSourceIntegral.py:975` `(1+z_response)·∫dlog(1+z')` — verified equal, `QS_04(b)` | **implemented** |
| `1/H²(z')` | `QuadSourceIntegral.py:957-961` (inside the integrand) | **implemented** |
| `q²_phys = (q/a₀)²` (i.e. `Q_s/a₀²` with the angular part stripped by R33/R34) | **nowhere.** No `q.k*q.k` factor in `QuadSourceIntegral`; `source_function` has no `q` dependence beyond `T_q` | **absent** |
| `Q_s` polarisation structure `(1/√2)sin²θ{cos2φ,sin2φ}`, the `½` from `Q_±²`, the `π` from `∫cos²2φ dφ` | **nowhere** (already folded into `648π²` in the spec) | **absent** |
| `36((1+w*)/(5+3w*))²` per `h_s` → `2592(...)⁴` for `⟨hh⟩` → `648π²(...)⁴` | **nowhere.** grep for `36`, `1296`, `2592`, `648` and for `wBackground(z_init)`/`w_star` in `ComputeTargets/` returns nothing | **absent** |
| `∫₀^∞ dq/q`, `∫₀^π dθ sin⁵θ`, `𝒫*(q)𝒫*(r)/r³`, `r=√(k²+q²-2kq cosθ)` | **nowhere.** `OneLoopIntegral.compute()` (`:98-112`) sets no value; `main.py` builds a fixed `(q,r)` product grid with no `|q−r| ≤ k ≤ q+r` filter (reconciliation doc F3) | **absent** |
| `8M_P²/(a²(ρ₀+p₀))` (`NUM` 03 R12) vs `4M_P⁴/(a⁴(ρ₀+p₀))` (`MAIN` 10 R11) | neither appears; the code enters at R22, where all of this has already collapsed into the single `2/(3(1+w₀))` | **implemented, in collapsed form** |
| `ζ* → φ*` constant `3(1+w*)/(5+3w*)` (spec 03 §0.4, spec 01 §2.8) | **nowhere**; `T_k = φ_k/φ*_k` only. This is one of the two `c_*` factors that make up the `36` above | **absent** |
| `a₀` powers | none needed: the code's `wavenumber.k` is `k/a₀` and `ModelFunctions.tau` is `a₀η`, each separately invariant under `a₀→λa₀`, `k→λk`, `η→η/λ`. Spec §0.1's `Q_s/a₀²` ↔ `a₀²` cancellation is therefore automatic | **implemented by construction** |

Net: what `QuadSourceIntegral` stores is `I_s` of R28 **divided by `Q_s/a₀²`** — i.e. the spec's brace
with `q²_phys` stripped — and the entire numerical prefactor chain that spec 03 §0.2 settled
(`1296 → 2592 → 1296π → 648π²`) is unimplemented. Consistent with spec 03 §0.2's own "Code status"
note; recorded here as the factor-by-factor ledger requested.

### QS-5 — DEFECT: the source spline cannot represent a sub-horizon source (quantified F1)

`ComputeTargets/QuadSource.py:294-309`:

```python
source_data = [(log(1.0 + v.z.z), v.source) for v in self.values]
...
source_spline = make_interp_spline(source_x_data, source_y_data)   # default cubic
```

The grid is `CosmologyConcepts/wavenumber.populate_z_sample` → `logspace(log10 z_init, log10 z_end,
…)` with `samples_per_log10z = DEFAULT_SOURCE_SAMPLES_PER_LOG10_Z = 100` (`main.py:69,321`), i.e.
`Δlog(1+z) ≤ 0.0231` (measured). `f` for `q=r=k` oscillates at twice the transfer-function phase, so
the node count per half-cycle at the low-`z` end is `π/(0.0231 · x)` with `x = k c_s (a₀η)`.

`scripts/QS_03_spline_error.py` builds the *exact* analytic radiation source on that grid, fits the
same `make_interp_spline`, and compares against the exact function at 39 interior points of every
node interval (`z_init = 10⁸`, `z_end = 0.1`, 900 nodes):

| `x_end` | cycles | nodes per half-cycle | max err / local envelope | worst `|Δ/f|` |
|---|---|---|---|---|
| 3 | 0.95 | 45.4 | 1.5e-09 | 1.6e-09 |
| 10 | 3.2 | 13.6 | 3.2e-06 | 4.1e-06 |
| 30 | 9.5 | 4.54 | 1.8e-04 | 4.1e-03 |
| 100 | 31.8 | 1.36 | 5.2e-03 | 8.8e+00 |
| 300 | 95.5 | 0.454 | **8.8e-01** | 1.2e+04 |
| 1e3 | 318 | 0.136 | **1.28** | 9.8e+04 |
| 1e4 | 3183 | 0.0136 | **1.21** | 7.3e+06 |

So the spline is accurate only while the mode is within ~10 cycles of horizon crossing; by
`x ≈ 300` the interpolant bears no relation to the function (error ≥ the oscillation envelope), and
the pointwise relative error is `10⁴`–`10⁷`. The shipped configuration is far past the bottom of this
table: `main.py:2671` uses source `k` up to `3×10⁸/Mpc` down to `z_end = 0.1`, and this module's own
Levin tuning note records a "**~4.5e6-cycle deep-sub-horizon case**"
(`ComputeTargets/QuadSourceIntegral.py:34`) — i.e. `x ~ 1.4×10⁷`, three decades below the last row.

This is finding **F1** of `docs/resonance-scaffolding/sigw-resonance-reconciliation.md` §0.2,
now measured. Downstream: every `QuadSourceIntegral` value (numeric, WKB-quadrature and Levin paths
alike) reads `source_f.source(log_z_source, z_is_log=True)`
(`QuadSourceIntegral.py:958, 1028, 1102`), so this spline error is the accuracy floor of the whole
source integral in the sub-horizon regime — including the resonance region. It is *not* a formula
disagreement with spec 03: the sampled node values are correct (QS-1); it is the dense-output
representation that fails.

### QS-6 — DEFECT: `compute_quad_source`'s grid alignment cannot survive the `Tk` truncation

`ComputeTargets/QuadSource.py:82-102` walks `z_sample` (the **full** `z_source_sample`, passed at
`main.py:910`) against `Tq.z_sample`/`Tr.z_sample`, advancing `q_idx`/`r_idx` only on a `store_id`
match. A mismatch is tolerated **only when `q_idx == 0`**, in which case `T=1, T'=0` is substituted
(`:116-121`, the correct super-horizon default). Any mismatch later raises; and once
`q_idx == len(Tq.z_sample)` the read at `:87`

```python
Tq_z: redshift = Tq_zsample[q_idx]
```

is out of range.

But `main.py:505-507` truncates the `TkNumericIntegration` grid at **both** ends —

```python
source_zs = z_source_sample.truncate(k_exit.z_exit_suph_e5, keep="lower") \
                           .truncate(0.85 * k_exit.z_exit_subh_e6, keep="higher-include")
```

— so the `Tk` grid stops at `z ≈ 0.85 z_exit_subh_e6`, far above `z_end = 0.1`, for every `k`; and
`main.py:528` additionally passes `mode="stop"`, which terminates the integration at a phase minimum
3–6 e-folds inside the horizon and stores only the achieved samples
(`Datastore/SQL/ObjectFactories/TkNumericIntegration.py:336-346,420` reconstruct `z_sample` from
`len(values)`). `QuadSource` also consumes `TkNumericIntegration` **only** — never
`TkWKBIntegration` (`QuadSource.py:9-12`; `main.py:854`) — so there is no continuation to fill the
low-`z` tail.

`scripts/QS_04_coverage_and_jacobian.py` drives the loop body directly with mock objects shaped that
way (12 source redshifts, `Tk` grid missing 2 leading and 4 trailing):

```
(a) len(z_sample) = 12   len(Tq.z_sample) = 6
    RAISED IndexError: list index out of range      # QuadSource.py:87
    control, Tk missing only the 2 leading high-z samples:
      OK, 12 values; first two use the T=1,T'=0 default: [1.5, 1.5, 1.5]
```

Consequence: with the shipped `main.py` settings the `--quad-source-queue` stage cannot complete for
any `(q,r)`; it fails loudly rather than storing a wrong number, so no bad data is at risk. The fix
is the same refactor QS-5 needs (feed `QuadSource` the WKB continuation, or truncate the
`QuadSource` `z_sample` to `Tq ∩ Tr`), which is why this is reported alongside F1 rather than as an
independent problem. `git log -- ComputeTargets/QuadSource.py` shows the file untouched since
`8e96750`, before the `z_exit_subh_e6` / `mode="stop"` truncation landed in `main.py`, so this looks
like an unnoticed regression rather than a design decision.

### QS-7 — the `analytic_source_w` column mixes two different `w`'s (convention / oracle hygiene)

`TkNumericIntegration.py:436-439` builds `analytic_T_w`/`analytic_Tprime_w` from
`compute_analytic_T(k, wPerturbations(z), tau)` — a *constant-`w`* Bessel solution evaluated with
`w = wPerturbations(z)`. `QuadSource.py:155-162` then feeds those into `source_function` with
`w = wBackground(z)`. Since spec 03 §0.5 fixes `w₀ = wBackground` inside `f`, the *numeric* column is
right; the point is only that the oracle column is internally inconsistent.

Measured (`scripts/QS_05_oracle_quality.py`, `Ω_m=0.3, Ω_r=9.1e-5, Ω_Λ=0.7`):

| `z` | `wBackground` | `wPerturbations` | ratio of `(5+3w)/(3(1+w))` |
|---|---|---|---|
| ≥ 10³ | 0.0776 … 0.332 | identical to 6 d.p. | 1.0000 |
| 100 | 0.009906 | 0.009909 | 1.0000 |
| 10 | −0.000638 | 0.001109 | 1.0007 |
| 1 | −0.225544 | 0.000202 | 1.1166 |
| 0.1 | −0.636651 | 0.000111 | 1.7009 |

So the two agree wherever the source term actually matters (radiation era through recombination) and
differ by up to 70 % in the coefficient below `z ≈ 1`, where the constant-`w` oracle is meaningless
anyway. Recorded as a convention note, not a defect.

### QS-8 — `analytic_source_*` are not independent oracles of `f` (convention)

`QuadSource.py:144-162` calls the *same* `source_function` three times, with numeric `T`, with
`analytic_T_rad`, and with `analytic_T_w`. So `analytic_source_rad ≡ f(analytic T)` by construction:
the comparison `source` vs `analytic_source_rad` tests `TkNumericIntegration` against
`analytic_Tk`, and says nothing about whether `f` itself is right. The independent check on `f` is
the one done here (`QS_01` sympy + `QS_02(b)` re-transcription). Worth stating so that a green
`source ≈ analytic_source_rad` comparison is not mistaken for validation of the kernel.

### QS-9 — symmetry and pair scheduling (AGREES)

`source_function` is exactly symmetric under `(T_q,T_q') ↔ (T_r,T_r')`: symbolically
(`QS_01`: difference `0`) and bit-for-bit in floating point (`QS_02(d)`: `f(q,r)` and `f(r,q)`
identical to the last bit, because the code's term ordering
`-(1+z)T_qT_r' - (1+z)T_rT_q' + (1+z)²T_q'T_r'` is itself symmetric under the swap). This is the
premise R31 uses to collapse the two Wick terms to `2 I_s²`.

`main.py:952-958` schedules `itertools.combinations_with_replacement(source_k_exit_times, 2)`, and
`source_k_exit_times` is built from an ascending `np.logspace` (`main.py:2671`), so the
`assert q.k <= r.k` at `main.py:878` always holds: only unordered pairs are computed, with the
reflection `r < q` obtained by symmetry (comment at `main.py:946-948`). Correct, and no double
counting. Note the spec's `f(z' | **q**, **k**−**q**)` (§0.5, Q3) depends on `q` and `r` only through
`|q|` and `|r|`, which is what the code stores — the `(k, k−q)` form on `NUM` 03 pp. 5, 7 is the
notational slip the spec already identifies, and the code does not follow it.

### QS-10 — cosmetic: wrong label on the source spline wrapper

`QuadSource.py:304` passes `"T_k"` as the `ZSplineWrapper` label. `spline_wrappers.py:41-53` uses it
in both out-of-bounds messages, already prefixed `"GkSource.function:"`, so a source-spline range
error is reported as `GkSource.function: evaluated T_k out of bounds`. Diagnostics only.

### QS-11 — float64 cancellation in `compute_analytic_Tprime`, harmless inside `f` (AGREES, measured)

`analytic_Tk.py:28-36` evaluates
`D = x J_{b+1/2}(x) − (3+2b) J_{b+3/2}(x) − x J_{b+5/2}(x)`, which is `O(x^{b+7/2})` as `x → 0`
while its individual terms are `O(x^{b+3/2})`. `scripts/QS_02c_cancellation.py` measures the resulting
double-precision loss against a 60-digit reference (`w=1/3, k=1`):

| `z` | `x = k c_s a₀η` | rel. err. of `Tprime` | rel. err. of `T` |
|---|---|---|---|
| 1 | 2.9e-01 | 2.5e-14 | ~1e-15 |
| 10² | 5.7e-03 | 1.9e-10 | ~1e-15 |
| 10⁴ | 5.8e-05 | 1.9e-06 | ~1e-15 |
| 10⁶ | 5.8e-07 | 2.8e-02 | ~1e-15 |
| 10⁷ | 5.8e-08 | 5.1e+00 | ~1e-15 |

`T` itself is accurate everywhere; only the derivative degrades, and it degrades exactly where the
mode is deep super-horizon. `scripts/QS_05_oracle_quality.py` shows this does **not** propagate: inside
`f` the `T'` terms are `O(x²)`-suppressed relative to the `T_qT_r` term, and the measured relative
error of the float64 analytic `f` against the 60-digit `f` is `≤ 1.0e-13` over
`z ∈ {1, 10², 10⁴, 10⁵, 10⁶, 10⁷}` and `(q,r) ∈ {(1,1),(30,30),(1,1000)}`. Recorded so that the
`analytic_Tprime_*` columns are not trusted on their own super-horizon.

---

## 3. Numerical convention notes

1. **`a₀` is absorbed, not set to 1** (spec §0.1). The two `a₀`-invariant combinations are exactly
   what the code carries: `wavenumber.k` is `k/a₀` and `ModelFunctions.tau` is `a₀η`
   (`BackgroundModel.py:56` integrates `d(a₀τ)/dz = -1/H`, with
   `tau_init = √3 M_P/√ρ · (1+z) = (1+z)/H`, which is `a₀/(aH)`). Under `a₀→λa₀`, comoving `k→λk`,
   `η→η/λ`, both are individually invariant, so every product formed in `analytic_Tk`
   (`k·c_s·tau`) and in `QuadSourceIntegral`'s Bessel arguments is automatically covariant.
   `source_function` contains no `a₀` at all — `f` is dimensionless and `a₀`-free, correctly.
2. **Derivative variable.** `T`, `Tprime` are `(T, dT/dz)`; the spec's `d/dz` is used directly with
   no Jacobian. The `log(1+z)` variable appears only twice, and both are conversions, not physics:
   the source spline is *fitted* in `log(1+z)` (`QuadSource.py:295`) and the source integral is
   *evaluated* in `log(1+z)` with the `(1+z')` Jacobian absorbed against the spec's `1/(1+z')`
   (verified in `QS_04(b)`). `ZSplineWrapper(..., deriv=True)` would divide by `(1+z)` — not used
   by `QuadSource`, which builds only the undifferentiated `source` function.
3. **Which `w`.** `wBackground` (Λ included) inside `f`; `wPerturbations` (Λ unperturbed) inside the
   `T_k` equation. Both match spec 03 §0.5 / spec 01's Tier 3 block. `w*` never appears in code.
4. **Signs.** `source_function` has no sign convention of its own: `f` is a sum of products, and the
   only relative sign, the `-(1+z)T T'`, is fixed by R16's `φ'+ℋφ = aH[φ-(1+z)dφ/dz]` and verified
   symbolically. The `Ḡ = -G` / `∫dz'` orientation conventions of §0.1 do not touch this file.
5. **Grid variable.** The redshift grid is log-spaced in `z` (`logspace(log10 z_init, log10 z_end)`),
   while the spline variable is `log(1+z)`. These coincide for `z ≫ 1`; below `z ≈ 1` the node
   spacing in `log(1+z)` shrinks (0.0231 at high `z`, ~0.002 at `z = 0.1`), which is why the worst
   spline error in QS-5 sits at `z ~ 1`, not at the very end of the range.
6. **`undiff` / `diff` columns.** Reporting-only split; `undiff` carries the `(5+3w)/(3(1+w))`
   grouping, which is *not* the spec's `T_qT_r` term. Do not compare column-by-column against the
   spec — only `source` is spec-defined.

---

## 4. Scripts run

All under `docs/spec-code-audit/scripts/`, run from the repository root with the project venv.

| script | what it checks | result |
|---|---|---|
| `QS_01_sympy_f.py` | sympy: spec 03 R22 `f` vs `QuadSource.source_function`; the `(5+3w)/(3(1+w))` regrouping identity; `q↔r` symmetry | all three differences identically `0` |
| `QS_02_deriv_and_f.py` | (a) `compute_analytic_Tprime` vs float FD `dT/dz`; (b) `source_function` vs independent R22 transcription on the analytic constant-`w` `T`; (c) shows `analytic_source_rad` is the same function; (d) numeric `f(q,r)=f(r,q)` | (b) worst rel. diff `2.3e-16`; (d) bit-identical; (a) superseded by `QS_02b` where FD was cancellation-limited |
| `QS_02b_deriv_mpmath.py` | 50-digit `mpmath` confirmation that `compute_analytic_Tprime = dT/dz` exactly | agrees to `1e-16 … 1e-11` outside the cancellation region; the two alternative readings differ by `(1+z)^{±1}` |
| `QS_02c_cancellation.py` | locates the float64 cancellation in `analytic_Tk.py:28-36` | rel. err. of `Tprime` `≈ 1e-16/x²`·O(10); `T` accurate to `2e-15` everywhere |
| `QS_03_spline_error.py` | spline error of `_create_functions` on the exact analytic radiation source, on the code's own 100-per-decade grid | error reaches 100 % of the local envelope for `x ≳ 300`; table in QS-5 |
| `QS_04_coverage_and_jacobian.py` | (a) drives `compute_quad_source`'s alignment loop with a realistically-truncated `Tk` grid; (b) sympy Jacobian of `dz' → dlog(1+z')` in the source integral | (a) `IndexError` at `QuadSource.py:87`; (b) difference `0` |
| `QS_05_oracle_quality.py` | (a) `f` accuracy vs 60-digit reference; (b) `wBackground` vs `wPerturbations` in ΛCDM | (a) `≤ 1.0e-13`; (b) identical above `z≈100`, coefficient ratio 1.70 at `z=0.1` |
