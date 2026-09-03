# `AdaptiveLevin`: API review and reuse assessment

**Subject.** `/Users/ds283/Documents/Code/SecondaryGWKit/AdaptiveLevin` — an adaptive Levin-collocation
quadrature for oscillatory integrals — together with the `LiouvilleGreen` layer that drives it.

**Question.** Is this reusable, as-is or after extraction, in the scale-dependent-bias pipeline described
in `png-tilt-forecast-design.md`, given the gaps identified in `LEGACY-CODE-AUDIT.md`?

**Review date:** 1 September 2026. Unlike the legacy audit, this one is *not* a static read: the code was
run, benchmarked against `scipy` references, and probed on a synthetic non-Limber projection integral.
Two defects were found by doing so, one of them blocking. Reproduction recipes are in §9.

---

## 0. Verdict

**Extract it. It is the best-in-class piece of the SecondaryGWKit codebase for this purpose, and it
already solves — correctly, and more generally than the legacy C++ — the exact problem
`LEGACY-CODE-AUDIT.md` identifies as the hardest to re-derive.**

Three findings drive that conclusion.

1. **`AdaptiveLevin` itself is small, sound, self-contained, and fast.** ~1,000 lines, four external
   dependencies (three of which are unnecessary), and a dependency footprint inside SecondaryGWKit of
   about 250 lines of trivially portable helper code. On a synthetic non-Limber integral with ~2,500
   oscillations it converged in **9 subintervals and 51 linear solves (0.057 s)** to a relative accuracy
   of 4×10⁻⁸, against 1.9 s for `scipy.integrate.quad` — a ~33× speedup, with the gap widening as the
   oscillation count rises. It behaves as advertised: cost is set by the *smoothness of the amplitude*,
   not by the number of oscillations.

2. **The `LiouvilleGreen/three_bessel_integrals.py` layer is a working, numerical Fabrikant integrator,
   and it is more general than either legacy implementation.** `quad_JJJ` computes
   ∫₀^X x² j_μ(kx) j_ν(qx) j_σ(sx) dx for *arbitrary* (μ,ν,σ) — no closed form, no diagonal restriction,
   no μ ≤ 4 cap. `LSSEFT-analytic` throws outside λ=0, μ=ν and warns above μ=4; `DonoughLSSEFT` supplies
   nine hard-coded closed forms. This code needs none of them. It also supplies `quad_YJJ` for the
   Y·J·J family, which is what the one-Rayleigh-momentum reduction needs and which the audit does not
   record as existing anywhere else.

   I verified the off-diagonal claim independently: for (μ,ν,σ)=(2,3,1) at (k,q,s)=(1.3,1.7,2.1),
   `quad_JJJ` agrees with brute-force `scipy` integration to 1.1×10⁻⁵, and with the corrected closed form
   to 1.1×10⁻⁶. See §7.2 — **the disagreement in the repository's own test is a typo in the test's
   reference formula, not an error in the integrator.**

3. **The blocking defect is not in `AdaptiveLevin` but in its companion phase function.**
   `LiouvilleGreen/bessel_phase.py:122` sets the Liouville–Green initial condition as
   `init_jv / m(min_x)` where `m = J² + Y²`; it should be `init_jv / sqrt(m(min_x))`. The quantity
   exceeds 1 for ν ≳ 5.5, so `np.asin` returns NaN and the ODE solve raises. **`bessel_phase` therefore
   works only for spherical Bessel order ℓ ≲ 5 as committed.** That is invisible in the current
   application (the Fabrikant work needs μ ≤ 4) and fatal for non-Limber C_ℓ, which needs ℓ up to ~2000.
   The one-character fix restores correct behaviour: with it, the ℓ=400 projection integral evaluates to
   2.5×10⁻⁶ at default settings and 5.5×10⁻⁹ with a denser phase grid.

**Where it does *not* help.** Levin quadrature is the wrong tool for two of the pipeline's other needs,
and it is worth saying so plainly so effort is not misdirected: the Legendre projection of μ^{0,2,4,6,8}
through the Matsubara exp[−k²(X+Yμ²)] damping is a *non-oscillatory* integral over μ ∈ [−1,1] — that is
the ~350 lines of closed-form special-function work in `LSSEFT/cosmology/multipole_Pk_calculator.cpp`,
and it must still be ported by hand. The 1-loop (q,x) loop integrals themselves are also non-oscillatory.
`AdaptiveLevin` addresses the Bessel-integral substrate underneath the angular reduction, and the
angular observable — nothing else.

---

## 1. What the code is

An implementation of Levin's method (Levin 1996) with adaptive bisection following the scheme in
Chen et al. (the source comments cite specific equations: (166)–(168) for the collocation system,
(172)–(173) and the step below it for the adaptive loop).

The problem solved is

$$I = \int_a^b \sum_{i=1}^{m} f_i(x)\, w_i(x)\, dx, \qquad w'(x) = A(x)\, w(x),$$

where the `f_i` are slowly varying and the vector `w` of oscillatory basis functions satisfies a linear
ODE system. Instead of resolving the oscillations, one seeks an antiderivative vector `p` with
`(p·w)' = f·w`, i.e. `p' + Aᵀp = f`, and then `I = p(b)·w(b) − p(a)·w(a)` — two boundary evaluations.

Mechanically:

| Step | Where |
|---|---|
| Chebyshev extremal grid + spectral differentiation matrix `D`, Weideman–Reddy construction | `chebyshev_matrices()`, `AdaptiveLevin/levin_quadrature.py:174` |
| Build `Aᵀ` for the chosen basis, and `w` at both endpoints | `_Basis_SinCos.build_Levin_data()` |
| Assemble `L = blockdiag(D,…,D) + Aᵀ`, solve `L p = f` by `lstsq` (falling back to `pinv`) | `_adaptive_levin_subregion_impl()` |
| Evaluate `p·w` at the endpoints, differenced | same, lines 456–466 |
| Bisect until \|est − (estL + estR)\| passes `atol` **or** `rtol` | `_adaptive_levin()` |

Two design decisions in the adaptive loop are worth knowing about because they dominate the observed
behaviour:

- **Direct-quadrature escape.** If the total phase change across a subinterval is below `6π`
  (`AdaptiveLevin/levin_quadrature.py:600`), the region is handed to `scipy.quad` instead. This is
  correct — Levin has no advantage on a nearly non-oscillatory interval, and the collocation system
  becomes ill-conditioned there — but it means a run that fails to converge degrades into ordinary
  quadrature rather than into wrong answers. Graceful, and expensive; see §6.2.
- **SVD-failure recovery.** `_adaptive_levin_subregion()` steps the Chebyshev order *down* by 2 on a
  linear-algebra failure and retries, down to a floor of 8. Changing the order changes the matrix, which
  gives the decomposition another chance. This is an unusual and, in my judgement, genuinely good idea —
  it is the kind of robustness that only gets written after the failure has been hit in production.

The only exported entry point is `adaptive_levin_sincos`, for the basis `w = (sin θ, cos θ)`.

---

## 2. The API

```python
from AdaptiveLevin import adaptive_levin_sincos

data = adaptive_levin_sincos(
    x_span,                    # (a, b) tuple of floats
    f,                         # list of m scalar callables, f[i](x) -> float
    theta,                     # dict, see below
    atol=1e-15,
    rtol=1e-7,
    chebyshev_order=12,
    depth_max=20,
    build_p_sample=False,
    notify_interval=300,
    notify_label=None,
    emit_diagnostics=False,
)
```

computing `∫_a^b [f[0](x) sin θ(x) + f[1](x) cos θ(x)] dx`.

### 2.1 The phase-function contract

`theta` is a dict with one required and two optional keys:

| Key | Required | Meaning |
|---|---|---|
| `theta` | **yes** | `θ(x)`, the raw (unwrapped) phase. Used only for the `6π` interval test and, if the others are absent, for everything else. |
| `theta_mod_2pi` | no | `θ(x) mod 2π`. If present, all `sin`/`cos` evaluations go through it. |
| `theta_deriv` | no | `θ'(x)`. If present, avoids differentiating `θ` spectrally. |

**`theta_mod_2pi` is the load-bearing option, and its presence is what makes this implementation viable
at large argument.** For a Bessel projection integral with kχ ~ 10⁵–10¹¹, `θ` is a huge number and
`sin(θ)` computed naively loses every significant digit. The repository takes this seriously enough to
carry a bespoke range reduction (`LiouvilleGreen/range_reduce_mod_2pi.py`, 79 lines) that factorises the
integer part of the argument with `sympy.factorint` and folds prime factors into the reduced residue one
at a time, so that `fmod` is never applied to a large number. The phase spline
(`LiouvilleGreen/phase_spline.py`) then stores `(div_2π, mod_2π)` as a *pair* and splines the residue
rebased to zero within each chunk. This is careful, unusual work, and it is the reason the code can be
pointed at `MAX_X = 1e12` — the value used in the repository's own three-Bessel tests.

If you supply only `theta`, you get the naive path and you should not trust it beyond θ ~ 10⁶ or so.

`theta_deriv` is supported but disabled at the one call site that defines it
(`ComputeTargets/QuadSourceIntegral.py:1019`, commented out) — treat that path as less exercised.

### 2.2 Return value

A dict. `value` is the integral (a `float`). The rest is diagnostics:

| Key | Use |
|---|---|
| `num_regions`, `regions` | accepted subintervals; each `used_interval` carries start/end/depth/type and its own `abserr`, `relerr`, `p_ratios` |
| `num_simple_regions` | how many fell through to direct quadrature — **the health metric to watch** |
| `evaluations` | linear solves ÷ 3 … see §7.4 |
| `max_depth` | bisection depth reached; equal to `depth_max` means silent non-convergence |
| `num_SVD_errors`, `num_order_changes`, `chebyshev_min_order` | conditioning health |
| `elapsed` | wall time |
| `p_points` | Levin antiderivative samples, if `build_p_sample=True` |

**There is no aggregate error estimate.** Per-region `abserr` is recorded and could be summed, but the
caller is not handed one. For a Fisher pipeline this matters — add it on extraction (§8).

### 2.3 The extension point

`_adaptive_levin()` is written generically in `m`: it takes a `BasisData` object satisfying an implicit
three-method protocol —

```python
class Basis:
    def raw_theta(self, x) -> float: ...              # for the 6π interval test
    def build_Levin_data(self, grid, Dmat) -> (AmatT, w0, wk): ...
    def eval_basis(self, x) -> list[float]: ...       # for the direct-quadrature fallback
```

— and only `_Basis_SinCos` implements it. This is the right seam, and it is undocumented. Two extensions
follow directly (§8.3): a `(j_ℓ, y_ℓ)` basis using the exact Bessel recursion for `Aᵀ`, which removes the
need for a precomputed phase function altogether; and an Airy/turning-point basis if the ℓ ≈ kχ region
ever needs special handling.

---

## 3. What comes with it: the `LiouvilleGreen` layer

`AdaptiveLevin` alone integrates against `sin θ`/`cos θ`. Everything that makes it useful for *Bessel*
integrals lives one directory over, and should be extracted with it.

| Module | Lines | What it does |
|---|---|---|
| `bessel_phase.py` | 289 | Liouville–Green phase/modulus for `J_ν`, `Y_ν` by the Bremer (2022, arXiv:2209.14561) modulus method: integrates `dQ/dlog x = (2/π)/(x m(x)) − Q` with `m = J² + Y²`, then fixes the additive phase offset by root-finding against `jv`. Returns `phase`, `mod`, `Q`, `bessel_j`, `bessel_y`, `min_x`, `max_x`. |
| `phase_spline.py` | 686 | Chunked spline of `(div_2π, mod_2π)` with derivative support; keeps the residue near zero within each chunk to avoid catastrophic cancellation. |
| `range_reduce_mod_2pi.py` | 79 | Prime-factor-assisted `big × small mod 2π`. Small and clever. |
| `three_bessel_integrals.py` | 440 | `quad_JJJ` and `quad_YJJ`: **the Fabrikant integrals, numerically.** |

### 3.1 How the three-Bessel integral is done

This is the part worth studying, because it is the template for every other multi-Bessel integral the
pipeline will need. `quad_JJJ` computes

$$\int_0^{X} x^2\, j_\mu(kx)\, j_\nu(qx)\, j_\sigma(sx)\, dx$$

by splitting at `x_cut = max(min_x)/min(k,q,s)`:

- **below the cut**, where the Liouville–Green representation is not valid (`min_x = √(ν²−¼)` is the
  turning point), it uses ordinary `scipy.quad` on the exact Bessel functions;
- **above the cut**, it writes each `J_{ν+½}(·) = m(·) sin θ(·)`, expands the triple product of sines
  into **four** sum-and-difference phases `θ_μ ± θ_ν ± θ_σ`, and calls `adaptive_levin_sincos` once per
  phase group with a common slowly-varying amplitude `x^{3/2} m_μ m_ν m_σ`, combining as
  `(−G₁ + G₂ + G₃ − G₄)/4`. Integration is in `log x`.

`quad_YJJ` is identical with `Y_{μ+½} = −m cos θ`, i.e. the amplitude moves to the cosine slot.

**The generalisation is mechanical.** A product of *n* Bessel factors decomposes into 2^{n−1} phase
groups. For the two-Bessel integrals needed by non-Limber C_ℓ the count drops to two. Nothing new has to
be invented; the pattern is already written down twice in this repository
(`three_bessel_integrals.py` and `ComputeTargets/QuadSourceIntegral.py:455–620`).

### 3.2 Against the legacy audit's Fabrikant discussion

`LEGACY-CODE-AUDIT.md` §1.3, §4.2 and §6 treat the Fabrikant integrals as a closed-form problem: which
(λ,μ,ν) cases have hard-coded expressions, and where to find more of them. Two of its concerns dissolve
if this code is adopted:

- *"Fabrikant integrals stop at μ = 4 … check it stays fine if ∇²δ or new operators raise the angular
  order."* Not a constraint here. `quad_JJJ` takes μ, ν, σ as floats.
- *"`LSSEFT-analytic` refuses λ ≠ 0 and μ ≠ ν … the closed forms already exist in `DonoughLSSEFT`."*
  Also not a constraint. The audit's advice to harvest Regan's nine closed forms remains excellent, but
  the role changes: they become **test oracles for a general numerical routine**, not the computational
  path. That is a strictly better position — the closed forms are exactly where the transcription errors
  live (§7.2 documents one), and a numerical integrator validated against several of them is more
  trustworthy than a lookup table that must be extended by hand for every new operator.

The audit's Route A ("reimplement the symbolic engine in SymPy") is also cheaper than it estimates, for
the same reason: the symbolic reduction can emit an *unevaluated* `FabJ(λ,μ,ν,s,t,u)` node for every case
and let `quad_JJJ` evaluate it numerically at run time, instead of needing a closed form to exist for
every index combination the reduction happens to produce. That removes the "throw if not in the table"
failure mode entirely.

---

## 4. Dependency footprint and extraction cost

`levin_quadrature.py` imports, from inside SecondaryGWKit:

```
Quadrature.simple_quadrature.simple_quadrature   -> Quadrature.integration_metadata.IntegrationData
                                                 -> Quadrature.supervisors.base (IntegrationSupervisor, RHS_timer)
                                                 -> utilities.format_time
utilities.format_time
```

`Quadrature/integration_metadata.py` imports `Datastore.DatastoreObject`, which looks alarming but is a
20-line base class with no further imports. **There is no real coupling to the SQLite/Ray/datastore
layer.** Total transitive Python to port or stub: ~250 lines, all of it trivial.

External: `numpy`, `scipy` — plus `seaborn` and `matplotlib`, imported unconditionally at module level
(`AdaptiveLevin/levin_quadrature.py:10`) but used only inside `_write_progress_data()`. Move them
behind the `emit_diagnostics` branch.

**Python ≥ 3.12 is required** — `levin_quadrature.py:78` uses a nested-same-quote f-string (PEP 701).
`LiouvilleGreen` additionally needs NumPy ≥ 2.0 (`np.pow`, `np.asin`) and `sympy` (for `factorint` in the
range reduction). All are one-line fixes if a lower floor is wanted.

**Estimated extraction effort: half a day** for `AdaptiveLevin` + helpers as a standalone package with
its four unit tests; **two to three days** including the `LiouvilleGreen` phase-function layer, the
`bessel_phase` fix, and a regression suite against the `DonoughLSSEFT` closed forms.

---

## 5. Test coverage as found

`AdaptiveLevin/tests/test_levin_quadrature.py` — 4 tests, **all pass in 0.027 s**:

| Test | Result |
|---|---|
| `∫₁^{50000} sin x dx` | correct to 1e-10, **1 region, 3 integrand evaluations** |
| `∫₁^{500000} cos x dx` | correct to 1e-10, **1 region, 3 evaluations**, 2 ms |
| `∫₁^{100} sin(100x)/x dx` | correct to 1e-10, 10 regions, 57 evaluations |
| `∫₋₁^{+1} cos(λ arctan x)/(1+x²) dx`, λ = 10, 100, 1000 | correct to 1e-10 against `(2/λ)sin(πλ/4)` |

The first two are the headline demonstration: ~80,000 oscillations resolved in three integrand
evaluations. That is what Levin quadrature is for.

`LiouvilleGreen/tests/` adds `test_bessel_phase.py`, `test_three_bessel.py`, `test_range_reduce.py`, and
`test_3bessel_analytic.py`. The last is the important one: it checks `quad_JJJ` against five closed forms
(J000, J110, J220, J222, J231) and `quad_YJJ` against two (Y000, Y022) — i.e. **it already covers the
off-diagonal cases**. Its `J220` reference agrees term-by-term with the `Fabrik022` expression the legacy
audit quotes from `DonoughLSSEFT`, which is a useful three-way consistency check between this code,
`LSSEFT-analytic`, and Regan's independent implementation.

Caveat on state: the working tree has this file modified, narrowing `Jintegrals` to `[J220]` at a fixed
right-angle configuration, with comments about reconciling against an external result. Read it as
mid-investigation rather than as a passing suite — and see §7.2, which is almost certainly what that
investigation is about.

---

## 6. Measured performance

Test problem: a synthetic non-Limber projection integral `∫ W(χ) j_ℓ(kχ) dχ` over χ ∈ [1, 1600] with a
broad-spectrum weight `W = 1/(1+χ)`, driven through `bessel_phase` (patched per §7.1), compared against
`scipy.integrate.quad` on `spherical_jn`.

### 6.1 With well-matched tolerances

ℓ = 2, k = 10 (≈2,546 oscillations), `chebyshev_order=32`:

| `atol` | `rtol` | rel. error | regions | of which direct | solves | time |
|---|---|---|---|---|---|---|
| 1e-12 | 1e-8 | 4.3e-08 | 9 | 0 | 51 | **0.057 s** |
| 1e-10 | 1e-6 | 8.7e-07 | 8 | 0 | 45 | 0.051 s |

`scipy.quad` on the same integral: **1.92 s**. Speedup ≈ 33×, and the Levin cost is essentially flat in
the oscillation count while `quad`'s is linear.

Chebyshev order scan at `atol=1e-12, rtol=1e-8`:

| order | rel. error | regions | solves | time |
|---|---|---|---|---|
| 12 | 4.2e-08 | 11 | 57 | 0.029 s |
| 32 | 4.3e-08 | 9 | 51 | 0.057 s |
| 64 | 4.3e-08 | 7 | 39 | 0.102 s |
| 96 | 4.2e-08 | 6 | 33 | 0.200 s |

Higher order buys fewer regions but each solve is O((mN)³), so wall time *rises*. **Order 12–32 is the
sweet spot**; the `chebyshev_order=64` used throughout `three_bessel_integrals.py` is likely costing
about 2× for nothing, and is worth re-tuning on the real integrands.

### 6.2 The tolerance trap

The same integral with `atol=1e-30, rtol=1e-11` — i.e. tighter than the phase representation can deliver:

| `atol` | `rtol` | rel. error | regions | of which direct | solves | time |
|---|---|---|---|---|---|---|
| 1e-30 | 1e-11 | 4.2e-08 | **892** | **856** | 2781 | **4.53 s** |

Same answer, 80× the cost, and 96% of the regions have fallen through to direct quadrature. At ℓ=2,
k=100 (~25,000 oscillations) this becomes 35,283 solves and 81 s, against 14.5 s for plain `quad` — the
Levin path is then *slower* than not using it at all.

**This is the single most important operational fact about the code.** The acceptance test is
`abserr < atol OR relerr < rtol`, and `relerr` is normalised by `min(|est|, |refined|)`, which blows up
near a zero crossing of a region's contribution. If neither tolerance is attainable, the bisection runs
to the `6π` floor and silently converts into `scipy.quad` with enormous overhead. It never returns a
wrong answer; it just stops being a Levin method.

Practical rule: **`atol` is the real control.** Set it to roughly (target relative accuracy) ×
(expected magnitude of the whole integral) ÷ (expected region count), and set `rtol` no tighter than the
accuracy of the phase function. Then check `num_simple_regions` and `max_depth` in the returned dict —
a large `num_simple_regions`, or `max_depth == depth_max`, means the tolerances are misconfigured.

### 6.3 Accuracy is limited by the phase function, not by Levin

ℓ = 400, k = 0.4, varying the `bessel_phase` sample density (default: 250 points per e-fold in log x):

| `sample_points` | rel. error | phase build time |
|---|---|---|
| default (~3,200) | 2.5e-06 | 0.03 s |
| 20,000 | 5.5e-09 | 0.40 s |
| 60,000 | 5.5e-09 | 1.04 s |

The Levin core will deliver whatever the phase/modulus splines give it. At default density the floor is
~10⁻⁸ for low ℓ and ~10⁻⁶ by ℓ = 400. For Fisher-forecast work that is likely fine; for anything
claiming sub-10⁻⁸, raise the density and amortise the phase build across all k at fixed ℓ (which the
current call pattern does *not* do — see §7.5).

---

## 7. Defects and sharp edges

### 7.1 `bessel_phase` initial condition — **blocking for non-Limber** (`LiouvilleGreen/bessel_phase.py:122`)

```python
init_jv  = jv(nu, min_x)
init_sin = init_jv / m(min_x)        # m = J^2 + Y^2 ; should be sqrt(m)
init_phase = np.asin(init_sin)
```

`J_ν = √m · sin θ`, so the argument of `asin` must be `J/√m`, not `J/m`. Measured at the turning point
`min_x = √(ν²−¼)`:

| ν | `J/m` (as written) | `J/√m` (correct) |
|---|---|---|
| 2.5 | 0.699 | 0.472 |
| 20.5 | **1.523** | 0.498 |
| 100.5 | **2.597** | 0.500 |
| 400.5 | **4.119** | 0.500 |

The crossover is near ν ≈ 5.5. Above it `asin` returns NaN and `solve_ivp` raises
`ValueError: All components of the initial state y0 must be finite`. **`bessel_phase` is unusable for
ℓ ≳ 5 as committed.**

Below the crossover the wrong initial condition is largely harmless — the ODE `dQ/dlog x = … − Q` is
contracting, so the error decays, and the residual constant offset is removed anyway by the later
`root_scalar` phase match against `jv`. That is why the existing Fabrikant tests (μ ≤ 4) pass and the bug
has gone unnoticed. It is nonetheless a bug, and it is exactly the wall you hit on the first non-Limber
C_ℓ call.

With `init_jv / np.sqrt(m(min_x))` substituted, ℓ = 2, 20, 100, 400 all evaluate correctly (§6).

### 7.2 `J231` reference formula is wrong — the integrator is right (`LiouvilleGreen/tests/test_3bessel_analytic.py:241`)

```python
k6 = s4 * s_sq          # names k^6, computes s^6
...
    - 5.0 * k6
```

Verified numerically at (μ,ν,σ) = (2,3,1), (k,q,s) = (1.3,1.7,2.1):

| Quantity | Value |
|---|---|
| Brute-force `scipy` integration, tail-averaged | −0.06216562 |
| `quad_JJJ` (max_x = 1e6) | −0.06216491 |
| Closed form with the `−5k⁶` term | −0.06216498 |
| Closed form **as written**, with `−5s⁶` | −0.30765570 |

`quad_JJJ` agrees with brute force to 1.1×10⁻⁵ and with the corrected closed form to 1.1×10⁻⁶. **The
Levin/Fabrikant path reproduces the off-diagonal three-Bessel integral correctly; the test's oracle has
a transcription typo.** Fixing `k6 = k4 * k_sq` should restore the full `Jintegrals` list.

This is a small illustration of the §3.2 argument. It is also worth cross-checking the corrected form
against `DonoughLSSEFT`'s `Fabrik123` before treating the matter as closed.

### 7.3 The tolerance trap

See §6.2. Not a bug, but a usability hazard severe enough to warrant a guard: warn when
`num_simple_regions / num_regions` exceeds, say, 0.5, or when `max_depth == depth_max`.

### 7.4 One third of the linear solves are redundant (`AdaptiveLevin/levin_quadrature.py:735`)

Each loop iteration performs three Levin solves: the parent estimate on `(a,b)` and the two halves
`(a,c)`, `(c,b)` used as the refinement. On bisection the two children are pushed back onto the queue as
*intervals*, and their already-computed estimates are discarded — each is re-solved from scratch when
popped. Caching the child estimate on `_levin_interval` reduces the cost from `3N` solves to `2N+1`. On
the §6.1 benchmark that is 51 → ~35. Cheap, safe, worth doing on extraction.

### 7.5 Recomputed Chebyshev matrices (`AdaptiveLevin/levin_quadrature.py:374`)

`chebyshev_matrices(x_span, order)` is called on every subregion evaluation. The grid and `D` on [−1,1]
depend only on `order`; only the affine rescaling depends on the span. Cache per order and rescale —
an O(N²) construction currently repeated thousands of times per integral.

### 7.6 Scalar-only integrand evaluation

`f_Cheb = np.hstack([[func(x) for x in grid] for func in f])` — every amplitude is called once per
collocation point in a Python loop, and the phase likewise. For a pipeline evaluating C_ℓ(z_i,z_j) over a
grid of ℓ and redshift-bin pairs, this Python-call overhead will dominate. The natural fix is to allow
`f` and `theta` to be either scalar callables or NumPy-vectorised ones, detected once at entry.

Related: the current `three_bessel_integrals.py` call pattern rebuilds `bessel_phase` for every
`(order, k·max_x)` combination. Phase functions should be built once per order over the widest needed
range and cached — they are independent of the amplitude and reusable across every k, q, s.

### 7.7 Smaller items

- **Diagnostics path is cwd-relative**: `SlowLevinData/{uuid}/{timestamp}`
  (`AdaptiveLevin/levin_quadrature.py:835`). Under a parallel driver this scatters output into
  whatever directory each worker happens to be in. Make it a parameter.
- **Unconditional `print` on warnings**, and a 5-minute progress notifier that writes to stdout. Route
  through `logging`.
- **`matplotlib`/`seaborn` imported at module scope** (§4) — a hard dependency for a numerical routine
  that only needs them for optional PDF diagnostics.
- **`lstsq` called without `rcond`** (`AdaptiveLevin/levin_quadrature.py:411`). Fine under
  NumPy ≥ 2.0 (the installed version is 2.2.4), but pin the intent explicitly.
- **No aggregate error estimate returned** (§2.2).
- **`p_use` filtering**: p-modes with mean relative amplitude below `rtol` are zeroed
  (`AdaptiveLevin/levin_quadrature.py:459`). The rationale in the comment is sound (they are
  associated with small singular values and pollute the error estimate), but it is a heuristic that
  couples the *solution* to `rtol`, and it deserves a test that it never discards a genuinely needed
  mode when the two components differ greatly in magnitude.
- **Descending grid convention.** `chebyshev_matrices` returns `x` with `x[0] = b` and `x[-1] = a`,
  and the endpoint extraction depends on it. Verified correct; documented only in comments. Any
  refactor must preserve it or the result silently changes sign.
- **No complex-valued `f`.** Not needed if sum/difference phase groups are used, but worth knowing.

---

## 8. Recommendation

### 8.1 Extract as a standalone package

Take `AdaptiveLevin/` plus the ~250 lines of helpers, and `LiouvilleGreen/{bessel_phase, phase_spline,
range_reduce_mod_2pi, three_bessel_integrals}.py`, into a small package (`levinquad`, or similar) with no
SecondaryGWKit dependencies. Carry the existing tests across; they are good and they run in
milliseconds.

Changes to make during extraction, in priority order:

1. **Fix `bessel_phase.py:122`** (`/ m` → `/ np.sqrt(m)`) and add a test at ℓ = 2, 20, 100, 400, 1000.
2. **Fix `test_3bessel_analytic.py:241`** (`k6 = k4 * k_sq`) and restore the full `Jintegrals` list.
3. Return an aggregate error estimate; warn on `max_depth == depth_max` or a high direct-quadrature
   fraction.
4. Cache child subregion estimates (§7.4) and Chebyshev matrices (§7.5).
5. Move `matplotlib`/`seaborn` behind `emit_diagnostics`; parameterise the diagnostics path; move
   `print` to `logging`.
6. Add vectorised-integrand support (§7.6).
7. Document the `BasisData` protocol (§2.3).

### 8.2 Regression baseline

`LEGACY-CODE-AUDIT.md` §8 recommends standing up `DonoughLSSEFT` as the regression baseline before
writing any Python, and that advice applies here with a twist: **transcribe Regan's nine `Fabrik*` closed
forms into Python as pure test oracles** (they are ~100 lines of arithmetic, no C++ build needed) and
check `quad_JJJ`/`quad_YJJ` against all nine. That gives the numerical integrator an independent,
multi-case validation, and simultaneously validates the closed forms against each other — §7.2 shows
that transcription errors in this family are real and not hypothetical.

### 8.3 Two extensions worth scoping

- **A `(j_ℓ, y_ℓ)` basis.** Implementing `build_Levin_data` from the exact spherical-Bessel recursion
  (`w' = A w` with `A` built from `j_ℓ' = j_{ℓ−1} − (ℓ+1)j_ℓ/x`) removes the phase function from the
  critical path entirely: no Liouville–Green ODE, no spline, no range reduction, and the §6.3 accuracy
  ceiling disappears. This is the standard formulation for Levin-based non-Limber solvers and it is the
  single highest-value addition. The seam already exists (§2.3).
- **Two-Bessel integrals for C_ℓ(z_i,z_j).** Mechanically the `quad_JJJ` pattern with 2^{n−1} = 2 phase
  groups. Straightforward once §8.1(1) is done.

### 8.4 Build versus adopt, for non-Limber C_ℓ

The design doc (§10 steps 3 and 4, and H1) commits to evaluating the low-k analysis as C_ℓ(z_i,z_j) with
non-Limber integration below ℓ ~ 100. That is a solved problem in the public literature, with
FFTLog-family and Levin-family solvers both in circulation and both benchmarked in the community
comparison exercises. **Before building, check whether an existing public non-Limber solver can be
driven with the required magnification, Doppler and wide-angle kernels** — the physics content the design
doc cares about is in the kernels, not in the quadrature.

The case for using this code instead is narrower but real: it is *already in the author's hands*, it
composes with the Fabrikant work that has no public equivalent, and — unlike an FFTLog approach — it
imposes no requirement that the radial kernels be power-law-decomposable, which matters for the
relativistic and wide-angle terms the design doc insists cannot be dropped. My reading is that the
Fabrikant/three-Bessel use is the *unambiguous* case for extraction, and the non-Limber use is a strong
secondary one that should be sanity-checked against a public solver rather than assumed.

### 8.5 What this does not solve

To keep the audit trail honest, restating §0: the Legendre projection of `μ^{0,2,4,6,8}` through the
Matsubara `exp[−k²(X+Yμ²)]` damping — item 2 on the legacy audit's salvage list, ~350 lines of
hand-derived special functions in `multipole_Pk_calculator.cpp` — is not an oscillatory-integral problem
and gains nothing from this code. Neither do the 1-loop (q,x) integrands, the wiggle/no-wiggle filter, or
the growth ODEs. `AdaptiveLevin` addresses the Bessel-integral substrate and the angular observable.
Everything else in the legacy audit's ranking still has to be ported on its own terms.

---

## 9. Reproduction

All measurements above were made with the SecondaryGWKit virtualenv
(`/Users/ds283/Documents/Code/SecondaryGWKit/venv`, Python 3.12, NumPy 2.2.4, SciPy 1.15.2).

Unit tests:

```bash
./venv/bin/python -m unittest AdaptiveLevin.tests.test_levin_quadrature -v
```

The `bessel_phase` initial-condition diagnostic:

```bash
cd /Users/ds283/Documents/Code/SecondaryGWKit && ./venv/bin/python -c "
from scipy.special import jv, yv
import numpy as np
for nu in [2.5, 20.5, 100.5, 400.5]:
    x = np.sqrt(nu*nu - 0.25); m = jv(nu,x)**2 + yv(nu,x)**2
    print(f'nu={nu}: J/m={jv(nu,x)/m:.4g}  J/sqrt(m)={jv(nu,x)/np.sqrt(m):.4g}')"
```

The `J231` cross-check:

```bash
cd /Users/ds283/Documents/Code/SecondaryGWKit && ./venv/bin/python -c "
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn
k, q, s = 1.3, 1.7, 2.1
f = lambda x: x*x*spherical_jn(2,k*x)*spherical_jn(3,q*x)*spherical_jn(1,s*x)
Xs = np.linspace(500.0, 700.0, 401)
tot = quad(f, 0, Xs[0], limit=200000, epsabs=1e-14, epsrel=1e-12)[0]
vals = [tot]
for a, b in zip(Xs[:-1], Xs[1:]):
    tot += quad(f, a, b, limit=20000, epsabs=1e-14, epsrel=1e-12)[0]; vals.append(tot)
print('brute force =', np.mean(vals))"
```

The non-Limber benchmarks of §6 used a patched copy of `bessel_phase.py` (§7.1) shadowing the repository
version on `PYTHONPATH`; **no files in SecondaryGWKit were modified by this review.**
