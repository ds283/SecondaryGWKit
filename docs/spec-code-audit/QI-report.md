# QI — spec 04 / spec 05 vs `ComputeTargets/QuadSourceIntegral.py`

**Subject.** `ComputeTargets/QuadSourceIntegral.py` (1502 lines), `ComputeTargets/OneLoopIntegral.py`
(123 lines), `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`, and the work-item grid in
`main.py`, at HEAD `e9a5539`.

**Specs.** `docs/spec/04-source-integral.md` (whole file), `docs/spec/05-one-loop.md` (whole file),
`docs/spec/03-source-term.md` §0.2–§0.3 (prefactor chain, completed build form R35).

**Method.** Every formula claim below is either a sympy derivation or a numerical measurement
against an independent implementation of the spec formula. Scripts in §4. No repository file was
modified.

---

## 0. Summary

**Verdict.** The source time integral is *correct against its spec*, and more tightly so than the
reconciliation doc's open question suggested. Measured:

- `analytic_integral` reproduces **spec 04 R14 with the signed-off $a_0^2$** to 7×10⁻⁹–4×10⁻⁷
  relative, on both its quadrature and its Levin branches, at $b = 0$ and $b = 0.2$ (§2, QI-1).
- All three numerical regions implement **exactly** the spec 03 R28 / spec 04 R1 measure
  $\int dz'\,G\,\frac{1+z}{1+z'}\,f/H^2$ — to 4×10⁻¹⁶ (quadrature) and 8×10⁻¹⁴ (Levin) — and agree
  with each other to 10⁻¹² on a common integrand (§2, QI-2).
- **Reconciliation doc §4 item 4 is closed and the code is right.** The code's
  $1/((3+2b)(2+b))$ against R31's $(2+b)/(3+2b)^3$ is exactly $1/c^2$, $c = (2+b)/(3+2b)$, and it is
  *entirely* accounted for by the normalisation of `QuadSource.source_function`: measured
  $f_{\rm code}/f_{\rm R28} = 1/c^2$ to 14 digits at three values of $b$ (§2, QI-3). Code
  `analytic_rad` $= -\frac{1}{c^2}\,a_0^{-2}\times$ (spec 05 R31), i.e. it is $-Q_s^{-1}\times$
  spec 03 R28's $I_s$ — precisely what §0.1 of specs 03/04/05 asserts. Not a defect.

| ID | Severity | One line | Code:line |
|---|---|---|---|
| QI-1 | AGREES | `analytic_integral` = spec 04 R14 with $a_0^2$; prefactor, orders, kernels, weight and limits all verify | `QuadSourceIntegral.py:868-878` |
| QI-2 | AGREES | numeric / WKB-quad / WKB-Levin all implement the spec 03 R28 measure and agree with each other | `:955-975`, `:1025-1045`, `:1094-1170` |
| QI-3 | AGREES (closes recon §4.4) | code/R31 prefactor ratio is $-1/c^2$, absorbed exactly by `source_function`'s $f$ normalisation | `:871`, `QuadSource.py:35-45` |
| QI-4 | CONVENTION | integration variable is $\log(1+z')$ throughout, with the $1/(1+z')$ Jacobian and the $(1+z)$ pulled out afterwards; orientation is $\int_{z_{\rm resp}}^{z_{\rm src,max}}$ (spec's $\int_z^{z_{\rm init}}$) | `:955-975` |
| QI-5 | DEFECT (known, F2 — confirmed) | the Levin call receives the **Green's-function phase only**; the oscillating source $f$ sits in the amplitude slot | `:1099-1112` |
| QI-6 | DEFECT (known, F2/§5 — confirmed) | no property of $T_q$, $T_r$ enters the Levin decision, at either gate | `:126-129`, `:160-163`, `GkSourcePolicyData.py:179-183` |
| QI-7 | UNVERIFIED | continuity of $G$ across `crossover_z` is never measured; the crossover is chosen on *spline clearance*, not on numeric/WKB agreement | `GkSourcePolicyData.py:346-410` |
| QI-8 | DEFECT (minor) | one of the eight Levin calls in `_three_bessel_Levin` (`Y3`) uses `LEVIN_ABSERR`/`LEVIN_RELERR` instead of the passed-in tolerances | `:622-623` |
| QI-9 | DEFECT (minor) | `analytic_integral`'s `atol`/`rtol` arguments are dead: the three-Bessel calls hardwire `1e-21`/`1e-8` | `:832-833`, `:844-845` |
| QI-10 | DEFECT (provenance) | `b` is not persisted; `analytic_rad` is stored with no record of the $b$ (and hence $c_s$, Bessel orders) it was computed at | `ObjectFactories/QuadSourceIntegral.py:173` |
| QI-11 | UNVERIFIED (by design) | `total`'s error bar excludes the two `quad` regions' reported errors; only the Levin region's `abserr` is propagated | `:314-322` |
| QI-12 | DEFECT (known, F3 — arithmetic confirmed) | 63,750 $(k,q,r)$ triples scheduled, **5,133 (8.05%)** close a triangle; no filter | `main.py:2488-2492` |
| QI-13 | not-yet-built | **none** of spec 05 R23/R31/R35's outer ingredients exists anywhere in the code | see §2 QI-13 |

---

## 1. Formula map

### spec 04 (NUM 06 / NUM 07)

| spec result | meaning | code location | status | how checked |
|---|---|---|---|---|
| R1 | source redshift integral $\int_z^{z_{\rm init}}dz'\,G_k\frac{1+z}{1+z'}\frac{f}{H^2}$ | `QuadSourceIntegral.py:955-975` (`numeric_quad_integral`), `:1025-1045`, `:1099-1170` | **AGREES** | QI_03: three branches vs independent `scipy.quad` of R1; rel. 4e-16 / 4e-16 / 8e-14 |
| R2, R3 | $G_{\rm me} = -H(z')G_{\rm them}$; explicit $G_{\rm them}$ | not in this module; `GkNumericIntegration.py` / `analytic_Gk.py` (other agent) | out of scope | — |
| R4 | $T_k = 2^{3/2+b}\Gamma(\frac52+b)(kc_s\eta)^{-3/2-b}J_{3/2+b}$ | not in this module; enters via `QuadSource` | used in QI_05 | reproduces $f$ to 1e-13 |
| R5, R6 | $f$ before/after the Friedmann substitution | `QuadSource.py:35-45` | **AGREES** | QI_05: code $f$ = R11 (the fixed-$w$ reduction of R6) to 5e-15 |
| R7, R8 | fixed-$w$ background, $w\!\leftrightarrow\!b$ | implicit; $c_s^2=(1-b)/(3(1+b))$ at `:347`, `:471`, `:743`, `:817` | **AGREES** | QI_01(d): $(1-b)/(3(1+b)) \equiv w$ under $b=(1-3w)/(1+3w)$ |
| R9, R10 | Bessel derivative and order-lowering identities | not coded explicitly; consumed into R11 | **AGREES** (indirect) | QI_05 |
| R11 (final boxed $f$) | $\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma^2(qc_s\eta)^{-\frac12-b}(rc_s\eta)^{-\frac12-b}\{J_{\frac12+b}J_{\frac12+b}+\frac{2+b}{1+b}J_{\frac52+b}J_{\frac52+b}\}$ | prefactor at `:871-872`, $(2+b)/(1+b)$ at `:853`, kernels at `:757-770` / `:493-501` | **AGREES** | QI_05, QI_02 |
| R12 | $dz = -a_0H\,d\eta$, $\frac{1+z}{1+z'} = (\eta'/\eta)^{1+b}$ | absorbed into the $(\eta')^{1/2-b}$ weight at `:500`, `:757`, `:769` | **AGREES** | QI_02 (the weight is what R14 demands) |
| R13 | conformal-time form before pulling factors out | intermediate; not coded | expected absence | — |
| **R14** (final analytic source integral, with the §0 $a_0^2$) | $-\frac{a_0^2\pi}{2}\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma^2(qrc_s^2\eta)^{-\frac12-b}\{Y_{b+\frac12}(k\eta)I_J - J_{b+\frac12}(k\eta)I_Y\}$ | `analytic_integral`, `:868-878` | **AGREES** | QI_01(a): symbolic ratio code/R14 $= a_0^{-2}$ exactly. QI_02: 6.8e-9 … 4e-7 relative on 6 configurations |
| R15, R16 | Liouville normal form, Riccati/Kummer phase equations | `LiouvilleGreen/bessel_phase.py` (Bremer modulus method) | out of scope (see `docs/adaptive-levin-audit-2026-09.md`) | order check only: `bessel_phase(0.5+b)`, `bessel_phase(2.5+b)` at `main.py:429-436` |
| R17 | three-Bessel integral in LG form, $(2/\pi)^{3/2}(x_1x_2x_3)^{-1/2}(\beta_1\beta_2\beta_3)^{-1/2}\prod\cos\gamma_i$ | `_three_bessel_Levin` `:493-501` (amplitude = $\eta^{3/2-b}\,m_1m_2m_3$) | **CONVENTION** | `bessel_phase` returns $m$ with $J_\nu = m\sin\theta$ (`bessel_phase.py:274-282`), i.e. $m^2 = \frac{2}{\pi x}\frac{1}{\theta'}$ of R22. The $(2/\pi)^{3/2}(x\beta)^{-1/2}$ of R17 *is* $m$; the code carries $m$ directly rather than re-deriving it, and uses $\sin$ where R17 writes $\cos$ (Q7's undocumented $\gamma = \Theta - \pi/2$). Verified by QI_02 |
| R18 | $\cos\cos\cos$ → four cosines | superseded by the $\sin\sin\sin$ route | expected absence | — |
| R19 | $\sin\cos\cos$ | superseded | expected absence | — |
| **R20** | $\sin\gamma_1\sin\gamma_2\sin\gamma_3 = \frac14(-S_{+++}+S_{++-}+S_{-+}-S_{--})$ | `:679-681`, phases at `:503-528`, `:544-566`, `:585-607`, `:628-648` | **AGREES** | coefficient-by-coefficient: `norm_factor*(-J1+J2+J3-J4)` with phases $\theta_1+\theta_2+\theta_3$, $\theta_1+\theta_2-\theta_3$, $\theta_1-\theta_2+\theta_3$, $\theta_1-\theta_2-\theta_3$ — identical to R20. Verified numerically by QI_02 |
| **R21** | $\cos\gamma_1\sin\gamma_2\sin\gamma_3$ | same four phases with `f=[0, -Levin_f]` (`:539`, `:580`, `:620`, `:661`) and `-Y1+Y2+Y3-Y4` at `:681` | **AGREES** | $Y_\nu = -m\cos\theta$ (`bessel_phase.py:279`), so $Y J J = -m_1m_2m_3\cos\theta_1\sin\theta_2\sin\theta_3$; the extra minus in `f=[0,-Levin_f]` supplies it. Verified by QI_02 |
| R22 | $J_\nu = m\sin\Theta$, $m^2 = \frac{2}{\pi x}\frac{1}{\Theta'}$ | `bessel_phase.py:121-134`, `:274-277` | **AGREES** | out-of-scope module; convention confirmed by inspection + QI_02 |
| R23–R26, R28 (NUM 07 Fabrikant) | $\mathcal J^\mu_{\nu\sigma}$ in Levin form | `LiouvilleGreen/three_bessel_integrals.py` (separate module) | not used by this file | `QuadSourceIntegral.py` has its **own** three-Bessel machinery (`_three_bessel_*`), independent of `LiouvilleGreen/three_bessel_integrals.py` |
| R27 | product-to-sum identities | consumed into R20/R21 | expected absence | — |
| R29 | closed-form $\mathcal J^0_{00}$, $\mathcal J^1_{10}$, $\mathcal J^2_{20}$ | `LiouvilleGreen/tests/test_3bessel_analytic.py` | out of scope | grep only |

### spec 05 (MAIN 11 / MAIN 14) and spec 03 §0.2–§0.3

| spec result | meaning | code location | status | how checked |
|---|---|---|---|---|
| R1–R22 | derivation of the field equation, Wick contraction, source in $b$-form | no code counterpart (derivation) | expected absence | — |
| **R23** | $P^h_{22}(k) = 32\int\frac{d^3q}{(2\pi)^3}Q_s^2P_*(q)P_*(r)(\text{time integral})^2$ | **nowhere** | not built | `OneLoopIntegral.compute()` is a no-op (`OneLoopIntegral.py:92-107`); grep for `Q_s`, `projector`, `sin^5`, `648`, `2592`, `1296`, `P_zeta` returns nothing outside unrelated files |
| R25, R27 | $f$ in $b$-form / four-Bessel form | `QuadSource.py:35-45` is R25's expanded form **divided by $c^2$** | **CONVENTION** | QI_05: $f_{\rm code}/f_{\rm R28} = 1/c^2$ exactly (see QI-3) |
| **R28** | completed-square $f$, prefactor $\frac{2+b}{(3+2b)^3}2^{3+2b}\Gamma^2$ | code's equivalent is spec 04 R11 | **CONVENTION** (ratio $1/c^2$) | QI_05, three values of $b$, 14 digits |
| R29, R30 | intermediate Step-6 lines | not coded | expected absence | — |
| **R31 (TARGET)** | $\pi 2^{2+2b}\frac{2+b}{(3+2b)^3}\Gamma^2(c_s^2qr\eta)^{-\frac12-b}(Y_{b+\frac12}I_J - J_{b+\frac12}I_Y)$, $I_{J/Y} = \int d\eta'(\eta')^{\frac12-b}\{J,Y\}_{\frac12+b}(k\eta')(\dots)$ | `analytic_integral` `:820-878` computes $-\frac{1}{c^2}\times$ this (in $a_0$-absorbed variables) | **AGREES up to the documented $-Q_s/c^2$** | QI_01(b): symbolic, R14/R31 $=-a_0^2(3+2b)^2/(2+b)^2 = -a_0^2/c^2$ exactly. $I_J$, $I_Y$ themselves are byte-for-byte R31: weight $(\eta')^{1/2-b}$ (`:500`,`:757`,`:769`), orders $\frac12+b$ / $\frac52+b$, coefficient $\frac{2+b}{1+b}$ (`:853`), limits $[\tau(z_{\rm src,max}), \tau(z_{\rm resp})]$ (`:820-824`) |
| R32, R33 | Step 7 Fabrikant form | deliberately not used | expected absence (per README §2) | — |
| spec 03 R35 / §0.3 build form: $648\pi^2\left(\frac{1+w^*}{5+3w^*}\right)^4\int\frac{dq}{q}\int d\theta\sin^5\theta\,\mathcal P^*(q)\frac{\mathcal P^*(r)}{r^3}\{\dots\}^2$ | the deliverable | **nowhere** | not built | see QI-13 |
| spec 03 R28's $Q_s(\mathbf k,\mathbf q)/a_0^2$ inside $I_s$ | spin-2 projector / $q_{\rm phys}^2$ | **nowhere** | not built | `total` is $I_s$ with the $Q_s/a_0^2$ *omitted* — see QI-13 |

---

## 2. Findings

### QI-1 (AGREES) — `analytic_integral` is spec 04 R14 with the signed-off $a_0^2$

**Spec.** spec 04 R14, read with the author's §0 item 4 correction $a_0^1 \to a_0^2$:

$$\int_z^{z_{\rm init}}\!dz'\,G_k\frac{1+z}{1+z'}\frac{f}{H^2} = -\frac{a_0^2\pi}{2}\,\frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2\!\left(qrc_s^2\eta\right)^{-\frac12-b}\left\{Y_{b+\frac12}(k\eta)I_J - J_{b+\frac12}(k\eta)I_Y\right\}$$

**Code** (`QuadSourceIntegral.py:868-878`):

```python
B = pi / 2.0
C = pow(2.0, 3.0 + 2.0 * b) / (3.0 + 2.0 * b) / (2.0 + b)
D = gamma(2.5 + b) * gamma(2.5 + b)
E = pow(q.k * r.k * cs_sq * eta_response, -0.5 - b)
F = -B * C * D * E
...
value = F * (Y_bessel * Y_factor - J_bessel * J_factor)
```

with `cs_sq = (1-b)/(1+b)/3` (`:817`), `Y_factor = data0pt5["J"] + A*data2pt5["J"]` and
`J_factor = data0pt5["Y"] + A*data2pt5["Y"]`, `A = (2+b)/(1+b)` (`:853-857`) — i.e. `Y_factor` is
$I_J$ and `J_factor` is $I_Y$, matching R14's pairing (the $Y$ outside multiplies the $J$-kernel
integral). `eta_response = tau(z_response)` (`:824`) is R14's unprimed $\eta$, as C6 requires.

**Derivation** (`QI_01_prefactor.py`): symbolically,
`F_code / F_R14 = a_0**(-2)` exactly. Since the code's variables *are* $k/a_0$ and $a_0\eta$
(spec 04 §0 item 1), a code expression that equals R14 with $a_0 \to 1$ is R14 written in the
invariants. Covariance test: R14's right-hand side in comoving variables scales as $\lambda^{-2}$
under $a_0 \to \lambda a_0$, matching the $a_0^2$ prefactor and leaving the left-hand side invariant
— consistent with §0 item 4, and inconsistent with the page's $a_0^1$.

**Measurement** (`QI_02_analytic_numeric.py`): `analytic_integral` was called with stand-in
model/wavenumber/redshift objects and *real* `bessel_phase()` splines built as `main.py:429-436`
builds them, then compared with a direct `scipy.quad` of the R14 integrand:

| $b$ | $k$ | $q$ | $r$ | $[\eta_{\rm init},\eta]$ | branch exercised | rel. diff |
|---|---|---|---|---|---|---|
| 0 | 1 | 1.3 | 0.8 | [1e-3, 5] | quad + Levin | 6.85e-09 |
| 0 | 10 | 7 | 5 | [1e-3, 3] | quad + Levin | 1.64e-07 |
| 0 | 50 | 30 | 25 | [1e-3, 2] | quad + Levin | 3.80e-07 |
| 0 | 200 | 120 | 90 | [1e-3, 1] | quad + Levin | 1.56e-07 |
| 0 | 200 | 120 | 90 | [0.5, 2] | **Levin only** | 5.58e-08 |
| 0.2 | 50 | 30 | 25 | [1e-3, 2] | quad + Levin | 1.32e-06 |
| 0.2 | 50 | 30 | 25 | [0.4, 2] | **Levin only** | 8.32e-08 |

The residual is the phase/modulus-spline fit floor (~2e-8 relative, per
`three_bessel_integrals.py`'s own caveat) plus `scipy.quad` round-off on the oscillatory reference
(the reference itself raises `IntegrationWarning` on the wide-range cases). Every prefactor,
Bessel order, kernel choice, weight and limit in R14 is therefore confirmed.

**Bessel orders.** `Bessel_0pt5 = bessel_phase(0.5 + b_value, …)` and
`Bessel_2pt5 = bessel_phase(2.5 + b_value, …)` (`main.py:429-436`) with `b_value = 0.0`
(`main.py:418`). The Green's-function order is $\frac12+b$ (`:472`/`:759`: `jv(0.5+b, x1)`;
`phase_data_Gk = phase_data["0pt5"]`) and the source orders are $\frac12+b$ and $\frac52+b$ — so
spec 04 Tier 2.1's "every $J_{5/2}$ is $J_{5/2+b}$" is honoured. Note the coupling: the Levin
branch uses the *caller-supplied* splines while `_three_bessel_quad` recomputes `jv(nu+b, ·)`
(`:759-771`); these agree only because `main.py` builds the splines with the same `b_value` it
passes as `b`. That is correct at HEAD but is an unguarded invariant.

### QI-2 (AGREES) / QI-4 (CONVENTION) — the measure, the Jacobian and the region consistency

**Spec.** spec 03 R28 / spec 04 R1: $I_s \propto \int_z^{z_{\rm init}}dz'\,\bar G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{f(z')}{H(z')^2}$.

**Code.** All three regions build the identical integrand and identical measure:

```python
# :955-960 (numeric)              # :1025-1030 (WKB quad)
def integrand(log_z_source):      def integrand(log_z_source):
    Green = Gk_f.numeric_Gk(...)      Green = Gk_f.WKB_Gk(...)
    H = model_f.Hubble(exp(log_z_source) - 1.0); H_sq = H * H
    f = source_f.source(log_z_source, z_is_log=True)
    return Green * f / H_sq
```
integrated over `[log(1+min_z), log(1+max_z)]` (`:962-963`, `:1032-1033`, `:1094-1096`) and then
scaled by `(1.0 + z_response.z)` (`:975`, `:1045`, `:1167`).

**The exact measure the code uses.** $d\log(1+z') = dz'/(1+z')$ supplies the $1/(1+z')$; the
post-multiplication supplies the $(1+z)$. So

$$\texttt{value} \;=\; (1+z_{\rm resp})\int \frac{dz'}{1+z'}\,\frac{G\,f}{H^2} \;=\; \int_{z_{\rm resp}}^{z_{\rm src,max}}\!dz'\;G\,\frac{1+z}{1+z'}\,\frac{f}{H^2},$$

which is spec 03 R28 / spec 04 R1 with **no** factor of $H(z')^2$, $(1+z')$ or $a_0$ missing.

**Measurement** (`QI_03_measure.py`), with synthetic smooth $G$, $f$, $H$ (and an oscillatory
$G = A(z)\sin\theta(z)$ for the Levin branch), against independent `scipy.quad`:

```
reference  int dz' G (1+z)/(1+z') f / H^2 = 5.815078635181e-09
           int dz' G (1+z)        f / H^2 = 8.673919877674e-07   <- NOT what the code computes

numeric_quad_integral  = 5.815078635181e-09   rel vs R28 measure = 4.27e-16
WKB_quad_integral      = 5.815078635181e-09   rel vs numeric branch = 0.00e+00
WKB_Levin_integral     = 1.599619724845e-10   rel vs R28 measure (osc G) = 8.27e-14
WKB_quad with same osc G = 1.599619724847e-10  rel vs Levin = 1.08e-12
```

The 0.993 relative discrepancy against the no-Jacobian variant shows the $1/(1+z')$ is
unambiguously present. **Region consistency (check 4):** the numeric and WKB-quad branches are
character-for-character the same integrand differing only in the `Gk_f` slot
(`:957` `numeric_Gk` vs `:1027` `WKB_Gk`), both take $f$ from the same
`source_f.source` spline, and `WKB_Gk` is `sin_amplitude·sin(theta_mod_2pi)`
(`spline_wrappers.py:118-122`), which is exactly the `sin_ampl·f/H²` amplitude plus
`f=[Levin_f, 0]`-times-$\sin\theta$ that the Levin branch hands `adaptive_levin_sincos`
(`:1099-1105`, `:1128`; the sin/cos basis is documented at `levin_quadrature.py:2724`). Measured
identity of the quad and Levin routes on a common oscillatory $G$: 1.1e-12. The underlying $G$
normalisations also agree by construction: the WKB `sin_coeff` is fixed from the numeric solution's
$G$, $G'$ at the matching point (`GkWKBIntegration.py:389-410`), with a defensive re-evaluation
check in `GkSource.py:265-296`.

**QI-4 (CONVENTION).** Variable choice $\log(1+z')$ (not $z'$, not $\log(1+z)$-of-response) and
integration orientation `a=log(1+min_z)` → `b=log(1+max_z)`, i.e. from $z_{\rm response}$ up to
$z_{\rm source,max}$, matching the spec's $\int_z^{z_{\rm init}}$ orientation. Together with
`analytic_integral`'s own $[\tau(z_{\rm src,max}), \tau(z_{\rm resp})] = [\eta_{\rm init}, \eta]$
(`:820-824`), `total` and `analytic_rad` are like-for-like: **same range, same $f$ normalisation,
same overall sign** — `analytic_rad` carries R14's explicit minus in `F` (`:875`) and `total`
computes R14's left-hand side, so the two are directly comparable with no sign flip. That is the
oracle relation the module intends and it is correctly wired.

One wart: the region-nonempty guards test `get_z(max_z)/get_z(min_z) > 1 + rtol` (`:194`, `:220`,
`:246`) — a ratio in $z$, not in $1+z$, so the effective minimum interval width varies by orders
of magnitude across the redshift range, and the test would divide by zero at $z_{\rm resp} = 0$
(not reachable at `DEFAULT_ZEND = 0.1`, `main.py:71`). Convention/robustness, not a numerical
defect.

### QI-3 (AGREES — closes reconciliation §4 item 4)

**The question.** `docs/resonance-scaffolding/sigw-resonance-reconciliation.md` §4 item 4: the
code's `1/((3+2b)(2+b))` (`:871`) against spec 05 R31's `(2+b)/(3+2b)^3`. Flagged, not claimed.

**Answer: consistent, and the discrepancy is exactly the $f$ normalisation.** Two independent
measurements.

*(a) Symbolic* (`QI_01_prefactor.py`):

```
(a) code prefactor / R14 prefactor = a_0**(-2)
(b) R14 prefactor / R31 prefactor  = -a_0**2*(2*b + 3)**2/(b + 2)**2   ==  -a_0^2/c^2   (True)
(c) f_R11 / f_R28                  =  (2*b + 3)**2/(b + 2)**2          ==   1/c^2       (True)
(c')c_*(w) = 3(1+w)/(5+3w) in b    =  (b + 2)/(2*b + 3)                 =    c
```

So, in the code's variables,

$$\boxed{\;\texttt{analytic\_rad} \;=\; -\frac{1}{c^2}\times(\text{spec 05 R31}),\qquad c=\frac{2+b}{3+2b}=\frac{3(1+w)}{5+3w}=c_*\;}$$

and equivalently `analytic_rad` $= a_0^{-2}\times$ (spec 04 R14 with $a_0^2$). The minus is the
`NUM` 02/03/06 redshift-orientation convention (spec 04 §0 item 3): R14 and R31 are differently defined
intermediates ($\bar G_k$ with $\int_z^{z_{\rm init}}dz'$ vs ${\rm Gr}_k$ with $\int d\eta'$), and each chain
followed consistently gives the same $h_s$ with the same sign. This report verified the code against the
`NUM` 03/06 chain only; it did not re-trace the `MAIN` 14 prefactor signs, and the correctness of the
code's sign does not depend on squaring. The $1/c^2$ is not a discrepancy: spec 03 R28's $I_s$ carries an
explicit $Q_s(\mathbf k,\mathbf q)/a_0^2$ that the code omits (it is the one-loop layer's job), and
spec 03 §0.1 states $I_s = -\frac{Q_s}{c^2}\times$R31. The code's `analytic_rad` is therefore
$I_s$ with $Q_s/a_0^2$ stripped, i.e. it is **exactly right**.

*(b) Numerical, at the level of $f$ itself* (`QI_05_f_normalisation.py`). `QuadSource.source_function`
was evaluated on a constant-$w$ background with $T$ from spec 04 R4 and compared with both
$f$ normalisations:

| $b$ | $w$ | $\eta$ | $f_{\rm code}$ | $f_{\rm R11}$ (spec 04) | rel diff | $f_{\rm code}/f_{\rm R28}$ (spec 05) | $1/c^2$ |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.3333 | 0.4 | 7.6910406e-01 | 7.6910406e-01 | 5.1e-15 | 2.2500000 | 2.2500000 |
| 0.00 | 0.3333 | 3 | -6.5437148e-03 | -6.5437148e-03 | 1.5e-13 | 2.2500000 | 2.2500000 |
| 0.20 | 0.2222 | 0.4 | 1.0430314e+00 | 1.0430314e+00 | 4.9e-15 | 2.3884298 | 2.3884298 |
| 0.20 | 0.2222 | 3 | -1.1402464e-02 | -1.1402464e-02 | 2.0e-13 | 2.3884298 | 2.3884298 |
| -0.15 | 0.4510 | 0.4 | 5.4927044e-01 | 5.4927044e-01 | 4.0e-16 | 2.1300219 | 2.1300219 |
| -0.15 | 0.4510 | 3 | -4.6605417e-02 | -4.6605417e-02 | 3.0e-14 | 2.1300219 | 2.1300219 |

`QuadSource.source_function` **is** spec 04 R11 (to 1e-13), it carries no $36\left(\frac{1+w^*}{5+3w^*}\right)^2$
and no $Q_s$, and its ratio to spec 05 R28's $f$ is $1/c^2$ to 14 digits. The reconciliation doc's
conjecture ("*That may well be absorbed by the differing normalisation of `f`*") is confirmed
quantitatively.

**Which spec factors are absent from the code, and where they must be supplied.**

| Spec factor | Where in the spec | In the code? | Where it must be supplied |
|---|---|---|---|
| $36\left(\frac{1+w^*}{5+3w^*}\right)^2 = 4c_*^2$ on $h_s$ | spec 03 R26/R28 | **no** | one-loop layer (as $648\pi^2\left(\frac{1+w^*}{5+3w^*}\right)^4$ after squaring) |
| $Q_s(\mathbf k,\mathbf q)/a_0^2 \to q_{\rm phys}^2\times\text{(spin-2 angular factor)}$ | spec 03 R28/R35, spec 05 R23 | **no** | one-loop layer, inside the brace before squaring |
| $\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma^2(\tfrac52+b)$ | spec 04 R11/R14 | **yes**, `:871-872` | — |
| $1/a_0^2$ | spec 03 R28 | absorbed (`k/a_0`, $a_0\eta$) — QI-1 | — |
| $c_s$ powers, $(qrc_s^2\eta)^{-1/2-b}$ | spec 04 R14 | **yes**, `:817`, `:873` | — |
| $1/H(z')^2$ | spec 04 R1 | **yes**, `:958-959`, `:1028-1029`, `:1101-1102` | — |
| $(1+z)/(1+z')$ Jacobian | spec 04 R1 | **yes**, QI-2 | — |
| $\mathcal P^*(q)\mathcal P^*(r)/r^3$, $\int dq/q$, $\int d\theta\sin^5\theta$, $648\pi^2$, per-$s$ label | spec 03 R35 / §0.3 | **no** | one-loop layer (QI-13) |

### QI-5 (DEFECT — confirms F2) — the Levin call is given the Green's-function phase only

`QuadSourceIntegral.py:1099-1112`:

```python
def Levin_f(log_z_source: float) -> float:
    H = model_f.Hubble(exp(log_z_source) - 1.0)
    H_sq = H * H
    f = source_f.source(log_z_source, z_is_log=True)      # <-- oscillates as theta_q +/- theta_r
    sin_ampl = Gk_f.sin_amplitude(log_z_source, z_is_log=True)
    return sin_ampl * f / H_sq                            # <-- amplitude slot

def Levin_phase(log_z_source: float) -> float:
    return Gk_f.phase.raw_theta(log_z_source, x_is_log=True)   # <-- Green's function only
```

`Gk_f.phase` is the phase spline built from `v.WKB.theta_div_2pi/theta_mod_2pi` of the *Green's
function* (`GkSourcePolicyData.py:659-670`); nothing about $T_q$ or $T_r$ enters. Once both source
modes are sub-horizon, `source_f.source` is a `make_interp_spline` through the sampled *oscillating*
source (that is F1), so the "amplitude" handed to the Levin rule oscillates at $\theta_q \pm \theta_r$.
`adaptive_levin_sincos` will still converge — it bisects until each subregion resolves what it was
told was smooth — but the frequency-independent accuracy the method exists for is lost, and the
accuracy floor becomes the source spline's. **Confirmed as described in F2; not redesigned here.**
Contrast the analytic branch, which does this correctly: `_three_bessel_Levin` puts the full
three-phase sum $\theta_{G} \pm \theta_{T_q} \pm \theta_{T_r}$ in the phase slot (`:503-528` and the
three siblings) and only the moduli in the amplitude (`:493-501`).

### QI-6 (DEFECT — confirms the known Levin-decision issue)

Two gates decide Levin, and **neither** consults $T_q$ or $T_r$.

1. *Where Levin becomes available*, `GkSourcePolicyData.py:179-190`: the first `z_source`
   (descending) at which `|theta_spline.theta_deriv(z, log_derivative=True)| > policy.Levin_threshold`,
   where `theta_spline` is built from the Green's-function WKB phase alone
   (`GkSourcePolicyData.py:154-169`). This is $|d\theta_G/d\log(1+z)|$ — the code's comment at
   `:174-177` correctly notes it must be the frequency with respect to $\log(1+z)$, the eventual
   integration variable.
2. *Whether Levin is worth it*, `QuadSourceIntegral.py:126-137` (WKB) and `:158-186` (mixed):

```python
phase_diff = Gk_f.phase.raw_theta(z_response.z) - Gk_f.phase.raw_theta(Levin_z)
if phase_diff > LEVIN_MIN_PHASE_DIFF:     # = 10 * 2*pi  (:26-27)
```

So the triggering quantity is the **Green's-function phase change across `[Levin_z, z_response]`,
in units of $2\pi$, thresholded at `LEVIN_MIN_2PI_CYCLES = 10`**. It is a *net* difference rather
than a total variation — harmless for the monotone Green's-function phase, but it means a region in
which $\theta_G$ turns over ten cycles while $\theta_q - \theta_r$ turns over $10^6$ (the resonant
regime) is classified as *not worth Levin* and handed to `scipy.quad`. This is the same class of
error as the audit's C2, one level up. **Known issue; confirmed with line numbers, not redesigned.**

### QI-7 (UNVERIFIED) — continuity of $G$ at `crossover_z` is not measured

`crossover_z` is chosen by *spline clearance* — how far a candidate $z$ sits from the ends of the
numeric and WKB splines in $\log(1+z)$ (`GkSourcePolicyData.py:346-410`, `classify_point`) — and by
the `maximize-WKB`/`minimize-WKB` policy, **not** by any comparison of the two representations'
values at that point. Nothing in `compute_QuadSource_integral` checks that
$G_{\rm numeric}(\texttt{crossover\_z}) \approx G_{\rm WKB}(\texttt{crossover\_z})$, so the region
sum `numeric_quad + WKB_quad + WKB_Levin` (`:297`) could carry a step discontinuity in $G$ without
any diagnostic. The normalisations *are* the same object by construction (QI-2), and
`GkSource.py:265-296` re-checks the WKB rectification against the stored `G_WKB`, so a gross error
would be caught; a small mismatch would not. **What would be needed:** a database run, evaluating
both splines at `crossover_z` for a sample of $(k, z_{\rm response})$ and reporting the relative
difference. Cannot be checked here.

### QI-8, QI-9 (DEFECT, minor) — tolerance plumbing in the analytic branch

- `:622-623`: of the eight `adaptive_levin_sincos` calls in `_three_bessel_Levin`, the `Y3` call
  alone passes `atol=LEVIN_ABSERR, rtol=LEVIN_RELERR` (`= 1e-23, 1e-8`, `:48-49`); the other seven
  pass the function's own `atol`/`rtol`. Since the caller supplies `1e-21`/`1e-8`, `Y3` is computed
  to a 100× *tighter* absolute tolerance than its siblings. Not a wrong answer — it wastes work and
  makes the four per-group `abserr`s summed at `:690-691` non-uniform, which matters because the
  groups cancel (the summed-linearly reasoning at `:686-689`). `LEVIN_ABSERR`/`LEVIN_RELERR` are
  otherwise dead constants; the asymmetry looks like a leftover.
- `:832-833`, `:844-845`: `analytic_integral` receives `rtol`/`atol` from the caller (which come from
  the `QuadSourceIntegral` `tolerance` objects, `:1462-1463`) and then **ignores them**, hardwiring
  `atol=1e-21, rtol=1e-8` on both `_three_bessel_integrals` calls. The stored `atol_serial`/
  `rtol_serial` therefore do not describe `analytic_rad`.

### QI-10, QI-11 — what is persisted

`QuadSourceIntegral.compute()` (`:1394-1466`) validates that the supplied `GkSourcePolicyData`,
`QuadSource`, $k$, $q$, $r$ and $z_{\rm response}$ all match, then dispatches
`compute_QuadSource_integral.remote(...)` with `b`, `Bessel_0pt5`, `Bessel_2pt5` from the payload.
`store()` (`:1469-1502`) copies the payload onto the object.

Columns (`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py:92-208`): `label`; FKs
`model_serial`, `k/q/r_wavenumber_exit_serial`, `atol_serial`, `rtol_serial`, `policy_serial`,
`source_serial`, `data_serial`, `z_response_serial`, `z_source_max_serial`; values `total`,
`numeric_quad`, `WKB_quad`, `WKB_Levin`, `analytic_rad`, `eta_source_max`, `eta_response`; timing
and Levin diagnostics (`numeric_quad_*`, `WKB_quad_*`, `WKB_Levin_num_regions/evaluations/
simple_regions/SVD_errors/order_changes/min_order/max_depth/elapsed`, `WKB_phase_spline_chunks`,
`compute_time`, `analytic_compute_time`); and a free-form JSON `metadata` string carrying the
analytic branch's `abserr`/`converged` plus the Levin region's `abserr`/`converged`/`phase_limited`.

- **QI-10.** There is **no `b` column**. `analytic_rad` is a $b$-dependent quantity (it fixes $c_s$,
  the Bessel orders and the $\eta'$ weight), and the field name asserts $b = 0$ while the code path
  accepts any $b$. At HEAD this is safe (`main.py:418` hardwires `b_value = 0.0`) but a future run at
  $b \ne 0$ would write indistinguishable rows.
- **QI-11.** `total = numeric_quad + WKB_quad + WKB_Levin` (`:297`). Only the Levin region's
  `abserr` reaches `metadata`; the two `quad` regions' `scipy.quad` errors are dropped, as the
  comment at `:314-322` states. So `total` has no error bound. Also stored: `analytic_rad`'s
  `abserr` excludes the phase/modulus spline fit error (~2e-8 relative, per §1.6 of the
  reconciliation doc and my QI-1 measurements).

### QI-12 (DEFECT — F3 arithmetic re-derived and confirmed)

`main.py:2488-2492`:

```python
qsi_work_items = itertools.product(
    z_source_integral_response_sample,
    itertools.combinations_with_replacement(source_k_exit_times, 2),
    response_k_exit_times,
)
qsi_work_items = [(z, k, q, r) for z, (q, r), k in qsi_work_items]
```

Grid: `source_k_array = np.logspace(log10(1e5), log10(3e8), 50)` (`main.py:2671`) and an identical
`response_k_array` (`main.py:2682`). `combinations_with_replacement(50, 2) = 1275` $(q,r)$ pairs
$\times$ 50 response $k$ = **63,750** $(k,q,r)$ triples, each replicated over
`z_source_integral_response_sample` (`main.py:2153`). Re-running the arithmetic on the actual grid
(`QI_04_triangle.py`):

```
combinations_with_replacement pairs = 1275
(k,q,r) triples = 63750
triples satisfying |q-r| <= k <= q+r : 5133 (8.05%)

mid-grid k = 5.94e+06/Mpc: 109 surviving pairs, s=(q+r)/k in [1.015, 101]
  nodes within +-0.05 of s=sqrt(3): 3 -> [1.6977, 1.6985, 1.7212]
```

**F3's numbers are exactly right**: 63,750 / 5,133 = 8.05%, so 91.95% of scheduled work is on
non-triangles; 109 surviving pairs at the mid-grid $k$, $s \in [1.015, 101]$, three nodes within
±0.05 of the radiation resonance $s = \sqrt3$. No `|q-r| <= k <= q+r` test appears anywhere in
`main.py` or in `QuadSourceIntegral.py`. Note that a non-triangle $(k,q,r)$ is not merely wasted:
it has no $\theta \in [0,\pi]$ that realises it, so the row is not a point of the spec 03 §0.3
integrand at all.

### QI-13 — spec 05 R23/R31/R35 outer ingredients: none exist

`ComputeTargets/OneLoopIntegral.py` is a stub with an inverted guard (already recorded; preamble
item 8): `compute()` (`:92-107`) raises if `self._value is None` — i.e. it refuses precisely when
there is work to do — and its body after the label assignment is empty, so `_compute_ref` is never
set and `store()` (`:109-123`) always raises. Its SQL factory has a single `value` column
(`ObjectFactories/OneLoopIntegral.py:121`). It is registered in the datastore
(`Datastore/SQL/Datastore.py:38-40,120`), sharded on `k` (`config/sharding.py:34`) and listed in the
inventory report (`tools/inventory_report.py:67`), but **`main.py` never instantiates it** (the only
`OneLoop` hits in `main.py` are the store-tag labels `TkOneLoopDensity`/`GkOneLoopDensity` at
`main.py:352-353`).

Checked by grep across the repository (excluding `venv/`):

| Ingredient | spec | Present? |
|---|---|---|
| $648\pi^2$ (or $512\pi^2$, $1296$, $2592$, $1024$) | spec 03 R35 / §0.2, spec 05 §0.2 | **no** — the only hit for `648` is `0.0648056` in `QCD_EOS.py:104` |
| $\left(\frac{1+w^*}{5+3w^*}\right)^4$, $w^*$, $c_*$ | spec 03 R26/R35 | **no** — no `w_star`/`wstar`; the only `5.0 + 3.0*w` is `QuadSource.py:35`, which is the $w(z')$ *inside* $f$, a different quantity (spec 03 §0.2) |
| projector $Q_s(\mathbf k,\mathbf q)$, $Q_s^2 = \frac12 q^4\sin^4\theta\{\cos^2,\sin^2\}2\varphi$ | spec 05 R23, spec 03 R34 | **no** |
| $\int d\theta\,\sin^5\theta$ measure, $\theta$ variable, $r=\sqrt{k^2+q^2-2kq\cos\theta}$ | spec 03 §0.3 | **no** — $r$ is an independent grid point (QI-12); nothing computes the implied $\theta$, and the $d\theta\,\sin^5\theta \leftrightarrow r\,dr$ Jacobian ($r\,dr = kq\sin\theta\,d\theta$) appears nowhere |
| $\int_0^\infty dq/q$ measure | spec 03 §0.3 | **no** |
| $\mathcal P^*(q) = \mathcal P_\zeta(q)$, $\mathcal P^*(r)/r^3$ | spec 03 §0.3–§0.4 | **no** — no primordial spectrum object anywhere |
| per-polarisation label $s$ / $\Omega_{\rm GW}$ separation | spec 05 §0.2 | **no** — nothing is labelled by $s$ |
| squaring of the time integral | spec 05 R23 | **no** |

**Expected answer confirmed: none of them exist.** Everything the specs demand *below* the loop
integral (the unit-jump $\bar G_k$, $T_k \to 1$ at $z_{\rm init}$, $f$, the R28 measure, the R14
analytic oracle) is built and verified; everything *above* it is unbuilt. Per spec 03 §0.2's own
note, the corrected $648\pi^2$ chain is therefore a **build specification**, not an audit finding.

---

## 3. Numerical convention notes

1. **$a_0$.** Confirmed absorbed, not set to one. `analytic_integral` carries no $a_0$, and
   symbolically that expression equals spec 04 R14's $a_0^2$ form written in $k/a_0$ and $a_0\eta$
   (QI-1). The covariance test $a_0 \to \lambda a_0$, $q,r,k \to \lambda(\cdot)$,
   $\eta \to \eta/\lambda$ passes: R14's right-hand side scales as $\lambda^{-2}$, cancelling the
   $a_0^2$, and the code's arguments $k\eta$, $qc_s\eta$, $rc_s\eta$ and the products
   $qr c_s^2\eta$, $\eta^{3/2-b}$ are built only from the invariants. `total` is likewise
   $a_0$-free and is $I_s$ (spec 03 R28) with the $Q_s/a_0^2$ omitted, so the one-loop layer must
   supply $q_{\rm phys}^2 = (q/a_0)^2$ and the angular factor.
2. **Sign.** `analytic_rad` carries R14's explicit $-$ (`:875`, `F = -B*C*D*E`). `total` computes
   R14's left-hand side. So the two are same-sign comparable, and both are $-1/c^2$ times spec 05
   R31 — the `NUM` 02/03/06 vs `MAIN` 14 orientation convention between differently defined intermediates (see QI-3); not a sign error in $h_s$.
3. **Variable.** Every integral in the file is in $\log$ of the natural variable: $\log(1+z')$ for
   the three numerical regions (`:962`, `:1032`, `:1094`) and $\log\eta'$ for the analytic branch
   (`:471`, `:743`), with the Jacobian absorbed into the amplitude
   (`A = pow(eta, 1.5 - b)` at `:500`, `:757`, `:769`, giving $d\eta'\,(\eta')^{1/2-b}$ — exactly
   R14's weight).
4. **$c_s$.** The analytic branch hardwires the fixed-$w$ value $c_s^2 = (1-b)/(3(1+b))$
   (`:347`, `:471`, `:743`, `:817`); QI_01(d) confirms this is identically $w$ under
   $b = (1-3w)/(1+3w)$, consistent with $c_s^2 = $ `wPerturbations` for a single-fluid constant-$w$
   epoch. The *numerical* branches inherit whatever $c_s(z)$ `QuadSource`/`Tk` used, so the two are
   comparable only for a genuinely constant-$w$ model.
5. **$\eta_{\rm init}$.** `analytic_integral` uses the finite $\tau(z_{\rm source,max})$
   (`:822`), as spec 04 §0.2's last bullet says it does; it never takes $\eta_0 \to 0$.
6. **LG phase convention.** `bessel_phase` supplies $J_\nu = m\sin\theta$, $Y_\nu = -m\cos\theta$
   (`bessel_phase.py:274-282`), i.e. NUM 07's R22 convention with $\gamma = \Theta - \pi/2$ relative
   to NUM 06 p.10's $\cos\gamma$ — spec 04 Q7's undocumented identification. The code is
   self-consistent in it (QI-1, QI-2 rows R20/R21).

---

## 4. Scripts run

All under
`docs/spec-code-audit/scripts/`,
run with `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python`.

| Script | Checks | Result |
|---|---|---|
| `QI_01_prefactor.py` | sympy: code prefactor vs spec 04 R14 vs spec 05 R31 vs $f$ normalisations; $c_*(w) = (2+b)/(3+2b)$; $c_s^2 = w$ | code/R14 $= a_0^{-2}$; R14/R31 $= -a_0^2/c^2$; $f_{\rm R11}/f_{\rm R28} = 1/c^2$ — all exact |
| `QI_02_analytic_numeric.py` | `analytic_integral` (real `bessel_phase` splines, stand-in model/wavenumber/redshift) vs `scipy.quad` of spec 04 R14, seven $(b,k,q,r,\eta)$ configurations covering quad-only, hybrid and Levin-only branches | relative difference 6.8e-9 … 1.3e-6, at the phase-spline floor |
| `QI_03_measure.py` | `numeric_quad_integral`, `WKB_quad_integral`, `WKB_Levin_integral` on synthetic $G$, $f$, $H$ vs `scipy.quad` of the spec 03 R28 measure, and against each other | 4.3e-16, 0.0, 8.3e-14 vs R28; quad-vs-Levin 1.1e-12; 0.993 off the no-Jacobian variant |
| `QI_04_triangle.py` | re-derives the F3 grid arithmetic from `main.py:2671/2682/2488` | 63,750 triples, 5,133 (8.05%) close; mid-grid $k$: 109 pairs, $s\in[1.015,101]$, 3 near $\sqrt3$ |
| `QI_05_f_normalisation.py` | `QuadSource.source_function` vs spec 04 R11 and spec 05 R28, at $b = 0, 0.2, -0.15$ | $f_{\rm code} = f_{\rm R11}$ to 5e-15…3e-13; $f_{\rm code}/f_{\rm R28} = 1/c^2$ to 14 digits |
