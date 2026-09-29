# Audit — the perturbation sound speed is $p/\rho$, not $dp/d\rho$

**Taken:** 2026-09-29 at `b9713ea` (`handover-remedial`, clean).
**Subject:** the quantity `wPerturbations(z)` feeds to the transfer-function sector as $c_s^2$, on
`QCD_Cosmology`.
**Measurements:** [`measure_cs2.py`](measure_cs2.py) — needs neither Ray nor a datastore. Every
figure below comes from that script; re-run it rather than trusting this document.
**Provenance:** found while reviewing arXiv:2609.29078 (Franciolini, Pritchard and Tada, "QCD
crossover transfer functions for scalar-induced gravitational waves in the PTA band"), whose
scalar equation carries a term $3\mathcal H^2(c_s^2 - w)$ that our closure sets to zero. The
review is [`docs/one-loop-planning/QCD-CROSSOVER-KERNELS-REVIEW.md`](../../docs/one-loop-planning/QCD-CROSSOVER-KERNELS-REVIEW.md).
**Status:** audit only. No production code has been changed. This folder is seeded with the audit
and its script; the campaign README, prompts and board do not exist yet.

---

## 0. The one-paragraph version

The transfer-function sector closes the scalar system with $\delta p = c_s^2\,\delta\rho$ and takes
$c_s^2$ from `wPerturbations(z)`. On every model the code had before the QCD background, that
function returns $p/\rho$ of the perturbed fluid, and $p/\rho$ is the right sound speed for a fluid
whose $w$ is constant. On the QCD background it is not: an equilibrium plasma with running
$g_{*\rho}(T)$ and $g_{*s}(T)$ has $\delta p/\delta\rho = dp/d\rho = w + \rho\,dw/d\rho$, and the
second term is what the crossover *is*. Measured on the code's own Saikawa–Shirai fits, the
adiabatic $c_s^2$ dips to **0.217** where $w$ dips to **0.250**, the gap reaches **0.050**, which
is **23 %** of $c_s^2$ and **11 %** of $c_s$, and it changes sign across the transition. This enters
the $c_s^2k^2$ term, so every source mode inside the horizon during the crossover oscillates at
the wrong frequency: the accumulated sound-phase error is **0.05–0.6 rad** for wavenumbers in the
PTA band and **17 rad** at $k = 3\times10^8$/Mpc, against a declared phase floor of $5\times10^{-12}$
rad. It also sets the mass term $3\mathcal H^2(c_s^2 - w)$ identically to zero, which is **7 %** of
$\omega^2$ three e-folds inside the horizon at the bottom of the dip. The fix is small — one
log-derivative of a fit the code already has — but it changes every stored $T_k$ on the QCD
background, and the fit's branch joins make the derivative need care.

---

## 1. What the code does

**The closure.** Spec 01 R4 (`docs/spec/01-transfer-function.md:232-236`) writes
$\delta p = c_s^2\,\delta\rho = w\,\delta\rho$, and the Tier 3 sign-off at the head of that file
reads "$c_s^2 = w(z)$ of the *perturbed* fluid ... In the code this is `wPerturbations(z)`".
`CLAUDE.md:93` records the same as a convention. The equation it closes is R5,

$$
\phi'' + 3\mathcal H(1+c_s^2)\phi' + \{2\mathcal H' + \mathcal H^2(1+3c_s^2)\}\phi - c_s^2\nabla^2\phi = 0,
$$

whose bracket is $3\mathcal H^2(c_s^2 - w)$ once $\mathcal H' = -\tfrac12(1+3w)\mathcal H^2$ is used.
In the code's redshift form the bracket is $\{3(1+c_s^2) - 2\epsilon\}/(1+z)^2$, which is the same
thing, because $\epsilon = \tfrac32(1+w)$ exactly.

**The consumers.** Every one of them reads the same function.

| Where | What it does with `wPerturbations` |
|---|---|
| `ComputeTargets/TkNumericIntegration.py:66-81` | friction $3(1+w)$, mass $3(1+w) - 2\epsilon$, and $w\,k^2/H^2$ |
| `ComputeTargets/WKB_Tk.py:20-21, 46, 97-99` | $\omega_{\rm eff}^2$ and its derivative, with $w$, $w'$, $w''$ |
| `ComputeTargets/TkWKBIntegration.py:445, 513-520` | the initial frequency, and the constant-$w$ analytic comparators |
| `ComputeTargets/BackgroundModel.py:308-321` | `_sound_speed_sq`, the integrand of the `cs_tau` sound-horizon table and the friction table |
| `main.py:703-711` | finite-difference `d_wPerturbations_dz` and `d2_...` when the cosmology supplies none |

**The producers.**

| Where | What it returns |
|---|---|
| `CosmologyModels/LambdaCDM/LambdaCDM.py:248-257` | $\tfrac13\,\Omega_r(1+z)/(\Omega_m + \Omega_r(1+z))$: $p/\rho$ of radiation + matter |
| `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:1182-1193` | `eos.w(T)`$\times\rho_r/(\rho_r+\rho_m)$ |
| `CosmologyModels/GenericEOS/GenericEOS.py:138-152` | `w(T)` $= 4g_s/(3g_\rho) - 1$, i.e. $p/\rho$ with $p = Ts - \rho$ |
| `CosmologyModels/GenericEOS/QCD_EOS.py:271-292` | the same, with $T$ clamped at `EOS_T_LO` $= 2\times10^{-3}$ GeV |

`LambdaCDM_GenericEOS` supplies no `d_wPerturbations_dz`, so on the QCD background the
derivatives `WKB_Tk.py` needs come from `main.py`'s finite differences. Nothing in the tree
computes $dp/d\rho$.

---

## 2. Why $p/\rho$ is the wrong quantity on this background

For a plasma in thermal equilibrium at zero chemical potential, $\rho + p = Ts$ and $dp = s\,dT$,
so $d\rho = T\,ds$. With $s = \tfrac{2\pi^2}{45}g_s T^3$,

$$
c_s^2 \equiv \frac{\delta p}{\delta\rho}\bigg|_{\rm adiabatic} = \frac{dp}{d\rho}
= \frac{s}{T\,ds/dT} = \frac{1}{3 + d\ln g_s/d\ln T},
\qquad
w = \frac{p}{\rho} = \frac{4g_s}{3g_\rho} - 1 .
$$

Writing $p = w\rho$, $\delta p = w\,\delta\rho + \rho\,\delta w = (w + \rho\,dw/d\rho)\,\delta\rho$: the
closure $\delta p = w\,\delta\rho$ drops $\rho\,dw/d\rho$, which vanishes when $g_\rho$ and $g_s$ are
constant and only then. That is why every check the project has ever run passed: on radiation,
on ΛCDM and on any constant-$w$ fixture the two definitions coincide to rounding, and the
$3\mathcal H^2(c_s^2 - w)$ bracket is genuinely zero. The QCD background is the first model in the
tree on which they differ, and it is the model whose whole purpose is the running of $g_*$.

The literature the project measures itself against uses $dp/d\rho$: Franciolini, Pritchard and
Tada solve $\Phi'' + 3\mathcal H(1+c_s^2)\Phi' + [c_s^2k^2 + 3\mathcal H^2(c_s^2-w)]\Phi = 0$ with
$c_s^2 = p'(T)/\rho'(T)$, show $w$ and $c_s^2$ separately in their Fig. 1, and attribute the
dominant $k$-dependent feature of their scalar transfer function to the sign change of
$c_s^2 - w$ across the transition.

---

## 3. Measurements

All on the code's own equation of state, `QCD_EOS`, i.e. the Saikawa–Shirai fits for $g_\rho(T)$
and $g_s(T)$, with $c_s^2$ from a centred difference of $\ln g_s$ in $\ln T$ at $h = 10^{-5}$.

### 3.1 $w$ against $c_s^2$ across the thermal history

| $T$ [GeV] | $w$ | $c_s^2$ | $c_s^2 - w$ | $(c_s^2 - w)/c_s^2$ |
|---|---|---|---|---|
| 0.03 | 0.3107 | 0.3043 | −0.0065 | −2.1 % |
| 0.09 | 0.3054 | 0.3003 | −0.0051 | −1.7 % |
| 0.11 | 0.2964 | 0.2831 | −0.0133 | −4.7 % |
| 0.13 | 0.2801 | 0.2393 | −0.0408 | −17.0 % |
| 0.14 | 0.2722 | 0.2224 | −0.0498 | −22.4 % |
| 0.15 | 0.2650 | 0.2172 | −0.0478 | −22.0 % |
| 0.17 | 0.2545 | 0.2260 | −0.0286 | −12.6 % |
| 0.20 | 0.2504 | 0.2498 | −0.0006 | −0.3 % |
| 0.25 | 0.2617 | 0.2753 | +0.0136 | +4.9 % |
| 0.30 | 0.2756 | 0.2881 | +0.0125 | +4.3 % |
| 0.50 | 0.3014 | 0.3054 | +0.0040 | +1.3 % |
| 30 | 0.3232 | 0.3186 | −0.0046 | −1.5 % |

Minimum $w$ **0.2503** at $T = 0.195$ GeV; minimum $c_s^2$ **0.2171** at $T = 0.151$ GeV; largest
gap **0.0501** at $T = 0.143$ GeV, **22.8 %** of $c_s^2$; largest relative error in $c_s$ itself
**10.8 %**. $c_s^2$ leads $w$: it is below $w$ on the high-temperature side of the dip and above it
on the low-temperature side, so the gap changes sign near $T = 0.2$ GeV. For comparison, the
lattice-spline equation of state of arXiv:2609.29078 has minimum $w \approx 0.231$ and a gap of
about 0.02; the difference is the input fit, not the method.

### 3.2 The derivative at the parametrisation's branch joins

`QCD_EOS.break_temperatures_GeV` declares four joins; `discontinuity_temperatures_GeV` says
$g_\rho$ and $g_s$ *jump* at three of them. A derivative of $g_s$ is worse behaved there than
$g_s$ is:

| Join | $T$ [GeV] | $c_s^2$ just below | $c_s^2$ just above | $w$ there | centred difference straddling the join |
|---|---|---|---|---|---|
| `EOS_T_LO` | 0.002 | 0.3324 | 0.3324 | 0.3336 | 0.3324 |
| `T_120_MEV` | 0.12 | 0.2714 | 0.2700 | 0.2881 | **0.5491** |

At `T_120_MEV` the two one-sided values differ by 0.5 %, so $c_s^2$ acquires a step there in
addition to the steps `H(z)` already carries, and a finite difference that straddles the join
returns a number that is simply wrong. Any implementation must differentiate each fit piece on
its own side of the join, and the quadrature and ODE break-point machinery
(`integration_break_points`, the per-sector policy of `prompts/qcd-background-audit`) must treat
`T_120_MEV` as a jump in $c_s^2$ as it already does for $H$.

### 3.3 Accumulated sound-phase error on `QCD_Cosmology`

The error that matters for the source is in the $c_s^2 k^2$ term. The phase a source mode
accumulates after horizon entry is $\int c_s\,k\,dz/H$, and the closure's error in it is
$\int (c_s^{\rm ad} - c_s^{w})\,k\,dz/H$, taken here from horizon entry ($k = H/(1+z)$) down to
the bottom of the window, with $c_s^2$ clamped below `EOS_T_LO` exactly as `w` is, because the
equilibrium formula is invalid there for both.

| $k$ [1/Mpc] | $T$ at entry [GeV] | sound phase after entry [rad] | phase error, window $T \ge 0.002$ GeV | phase error, crossover only, $T \ge 0.02$ GeV |
|---|---|---|---|---|
| $10^5$ | 0.0085 | 1.9 | −0.002 | — |
| $3\times10^5$ | 0.025 | 6.8 | −0.012 | −0.002 |
| $10^6$ | 0.079 | 24 | −0.045 | −0.013 |
| $2\times10^6$ | 0.145 | 49 | −0.11 | −0.044 |
| $5\times10^6$ | 0.32 | 123 | −0.29 | −0.13 |
| $10^7$ | 0.63 | 246 | −0.58 | −0.26 |
| $3\times10^7$ | 1.85 | 740 | −1.7 | −0.77 |
| $10^8$ | 6.1 | 2470 | −5.8 | −2.6 |
| $3\times10^8$ | 18.3 | 7410 | −17 | −7.7 |

For every mode that enters before the crossover the error is a fixed fraction, about
$2.3\times10^{-3}$, of the sound phase accumulated across the transition, and it grows linearly
with $k$. The project's declared phase floor is `BesselPhaseFunction.theta_abserr` $= 5\times10^{-12}$
rad, and the largest recorded representation error in this sector is
`[00-tk-lg-truncation-floor]` at $1.4\times10^{-4}$ of the envelope. The closure error is nine to
thirteen orders above the first and three to five above the second across the production range.

### 3.4 The mass term the closure sets to zero

At the bottom of the dip, $3(c_s^2 - w)/c_s^2 = -0.68$, so the term $3\mathcal H^2(c_s^2 - w)$ is
$-0.68\,(aH/k)^2$ of $c_s^2k^2$:

| $k/(aH)$ | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| relative to $c_s^2k^2$ | −0.68 | −0.076 | −0.0068 | −0.00076 |

This is the term arXiv:2609.29078 identifies as the origin of the boost-then-dip in $|\Phi|^2$
relative to radiation for modes crossing the horizon near the transition. Under the closure it is
identically zero for every $k$.

---

## 4. Impact

- **What is wrong.** Every stored `TkNumericIntegration`, `TkWKBIntegration`, `TkSourceFunctions`,
  and the `cs_tau` and friction tables of `BackgroundModel`, on `QCD_Cosmology`. Through them,
  every `QuadSource` and `QuadSourceIntegral` on that model. The size of the error is §3.3: a
  sound-phase error of order 0.1–1 rad for the modes the QCD background exists to describe.
- **What is not affected.** Radiation, ΛCDM and every constant-$w$ fixture, on which the two
  definitions agree to rounding, so every test and oracle in the tree remains valid as a test of
  what it tests. The tensor sector, which never reads `wPerturbations`. The source term's $w_0$,
  which is `wBackground(z')`, the background $p/\rho$, and is correct as such (spec 03 §0.2). The
  constant-$w$ analytic comparators `compute_analytic_T` and friends, which are exact for the
  constant $w$ they are given.
- **Datastore.** Nothing to migrate: the schema is unchanged. The QCD products are wrong and must
  be recomputed; whether the existing QCD stores are retired through `store retire` is the user's
  call.
- **Relation to the source term's $(1+3w)$ finding.** Independent. That one is theirs, in the
  source coefficients; this one is ours, in the scalar equation.

---

## 5. Complications an implementation must face

1. **Derivatives across joins** (§3.2). `WKB_Tk.py` needs $c_s^{2\prime}$ and $c_s^{2\prime\prime}$;
   `main.py:703` builds them by finite differences of the value function. A step in $c_s^2$ at
   `T_120_MEV` turns those into spikes unless the differences are one-sided at the join. The clean
   way is for the equation-of-state object to supply $d\ln g_s/d\ln T$ and its derivative
   analytically, piece by piece, as it supplies $g_s$.
2. **Below `EOS_T_LO`.** The equilibrium formula is invalid once the neutrinos decouple, for
   $c_s^2$ as for $w$. `QCD_EOS.w` clamps at `EOS_T_LO`; the same clamp on $c_s^2$ is the minimal
   consistent choice, and §3.3 used it. A sector-by-sector treatment is out of scope.
3. **The matter admixture.** `LambdaCDM_GenericEOS.wPerturbations` weights the plasma value by
   $\rho_r/(\rho_r+\rho_m)$. For an adiabatic radiation-plus-dust mixture the sound speed is
   $c_{s,\rm pl}^2/(1 + 3\rho_m/4\rho_r)$, not $c_{s,\rm pl}^2/(1 + \rho_m/\rho_r)$. The difference is
   $O(\rho_m/\rho_r) \sim 10^{-9}$ at the QCD epoch and is the author's ΛCDM convention, which
   `LambdaCDM.wPerturbations` and `CosmologyModels/tests/test_wPerturbations.py` pin. Not this
   audit's to change; recorded so the decision is explicit.
4. **The name.** `wPerturbations` appears in 26 modules and tests outside `docs/`. A rename to
   something that says "squared sound speed" is honest but wide; keeping the name and changing
   what it returns is narrow but leaves a misleading name in a convention line of `CLAUDE.md`.
5. **The spec.** Spec 01 R4 and its Tier 3 sign-off state the closure as $c_s^2 = w$. That is
   right for what the notes derived, a constant-$w$ epoch, and wrong as a statement about the
   QCD build. The sign-off note needs an addendum, not a rewrite (CLAUDE.md rule 6).
6. **The `cs_tau` table** is built once per model from `_sound_speed_sq` at the break-point
   knots, so it inherits the fix for free, but its quoted increment error ($4.55\times10^{-15}$ for
   $c_s/H$, `BackgroundModel.py:37`) was measured for the smooth $p/\rho$ and must be re-measured
   for a $c_s$ with a step at `T_120_MEV`.

---

## 6. A campaign sketch, for the README to take or leave

Correctness is the only objective; the order is by epistemic dependency.

| # | Prompt | Why in this position |
|---|---|---|
| 01 | Give `GenericEOSBase` a `cs2(T)` with an analytic $d\ln g_s/d\ln T$ per fit piece, and `QCD_EOS` its clamp; test against the finite-difference definition away from joins and against the one-sided values at them. | Everything downstream reads it. |
| 02 | Route `LambdaCDM_GenericEOS.wPerturbations` to it, with analytic first and second derivatives so `main.py:703`'s finite differences are not used on this model; keep `LambdaCDM` byte-identical. | Consumers must not see a spiky derivative. |
| 03 | Re-run the transfer-function verification harness (`docs/gktk-remedial/`) on `QCD_Cosmology` and record the change in $T_k$ against a stored QCD run; re-measure the `cs_tau` increment error. | The measurement that says the fix did what §3 predicts. |
| 04 | Spec 01 R4 addendum, `CLAUDE.md:93`, the `_sound_speed_sq` docstring, this audit's close-out. | Record last, once the numbers exist. |

**Decisions for the user:** D1 the name (item 4); D2 the clamp policy below `EOS_T_LO` (item 2);
D3 whether the ΛCDM matter weighting is touched (item 3: recommended no); D4 whether the existing
QCD stores are retired.

---

## 7. What was not done

No production file was edited. No transfer function was recomputed with the corrected sound
speed, so the size of the change in $T_k$ itself is predicted from the phase error in §3.3, not
measured; that is prompt 03's job. The e$^\pm$ annihilation region below `EOS_T_LO` was not
examined beyond confirming that the unclamped $c_s^2$ there differs from the clamped $w$ by up to
$7\times10^{-4}$, which is why §3.3 clamps.
