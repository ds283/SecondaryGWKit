# arXiv:2609.29078 against this project — what the QCD-crossover kernel paper teaches about the one-loop build

**Date:** 2026-09-29 · **Tree:** `handover-remedial` at `b9713ea`, clean · **Author:** Claude Fable
5.1, from the paper's LaTeX source (`arXiv-2609.29078v1.tar.gz`), its public code
(`github.com/gfranciolini/QCD-SIGW`, read on 2026-09-28) and this repository ·
**Status:** fact-finding for the `OneLoopIntegral` campaign, which does not exist yet. Nothing here
is a plan, and no production file was changed to write it.

**The paper.** G. Franciolini, X. Pritchard and Y. Tada, *QCD Crossover Transfer Functions for
Scalar-Induced Gravitational Waves in the PTA Band*, arXiv:2609.29078. It solves the tensor and
scalar equations across the Standard Model thermal history, tabulates the oscillation-averaged
Kohri–Terada kernel on a $(t, s, k)$ grid, and fits NANOGrav 15-year data with it. The same
physics this project computes, by a different route.

**Scripts.** [`fpt_rd_quadrature.py`](fpt_rd_quadrature.py) (their grid rule and the reference,
in exact radiation), [`fpt_trapezoid_scan.py`](fpt_trapezoid_scan.py) (the small-$t$ scan) and
[`fpt_levin_comparison.py`](fpt_levin_comparison.py) (the same integrals through
`AdaptiveLevin`). Run from the repository root with `PYTHONPATH=.`; none needs Ray or a datastore.
Every number in §4.2 and §5.2 comes from them.

**Companion.** The one defect this review found on our side is audited separately, in
[`prompts/perturbation-sound-speed/AUDIT.md`](../../prompts/perturbation-sound-speed/AUDIT.md).

---

## 0. Summary

1. **Their architecture** is the Kohri–Terada kernel on the numerical background: scalar and
   tensor modes solved in conformal time and continued by leading-order WKB, the source time
   integral by the trapezoid rule on a geometric grid, and the response-time dependence factored
   out *analytically* through the Wronskian decomposition of the Green's function, so that the
   oscillation average touches only the two tensor mode functions.
2. **That factorisation exists in our Liouville–Green representation too, in a better basis**, and
   it removes the need to sample `z_response` at all. At a single late response redshift, two
   Levin integrals per $(k, q, r)$ — the existing one and its quadrature partner, with
   $\sin\theta_G \to \cos\theta_G$ — give the period-averaged $\langle h^2\rangle$ exactly as
   $A(z_r)^2(J_1^2 + J_2^2)/2$. §3 derives it and says what it does to `QuadSourceIntegral`,
   `OneLoopIntegral` and the observable layer.
3. **Their source-term coefficients are inconsistent away from $w = 1/3$**: they are Kohri–Terada's
   constant-$w$ coefficients, which already contain $\mathcal H = 2/((1+3w)\eta)$, with a numerical
   $1/\mathcal H$ re-inserted. The derivative terms are low by $(1+3w)/2$ and $((1+3w)/2)^2$, and
   the normalisation is evaluated at the running $w$. At their $w_{\min} = 0.231$ the three
   coefficients are 0.95, 0.80 and 0.68 of the correct ones. Invisible to their radiation-era
   validation. Our `QuadSource.source_function` has this right (§4.1).
4. **Their time quadrature** is 10⁻³ to 10⁻² relative in the region $t \lesssim 2$ that carries any
   spectrum, 10⁻⁵ in the middle, and 3–5 × 10⁻³ where their point cap bites — measured, §4.2. Our
   Levin driver on the same integrals is 10⁻¹¹ to 10⁻¹⁶ at flat cost (§5.2).
5. **The one thing to fix on our side** is the perturbation sound speed: `wPerturbations` is
   $p/\rho$, and on the QCD background that is not $dp/d\rho$. The paper's dominant scalar effect
   lives in the term our closure sets to zero. Audited in the companion document; it must land
   before any QCD production run, but it does not block designing the one-loop target.

---

## 1. What the paper computes, and how

Everything below is from `main.tex` and the public `core/` modules; nothing is paraphrased from
the abstract.

- **Background.** $g_{*\rho}(T)$, $g_{*s}(T)$ from a cubic spline of lattice QCD (Borsanyi et al.),
  sector-by-sector after neutrino decoupling; $w = p/\rho$, $c_s^2 = p'(T)/\rho'(T)$;
  $w_{\min} \approx 0.231$, $c_{s,\min}^2 \approx 0.21$. Friedmann and continuity integrated for
  $T(\eta)$, $a(\eta)$. Neutrino free-streaming damping neglected.
- **Scalar mode.** $\Phi'' + 3\mathcal H(1+c_s^2)\Phi' + [c_s^2k^2 + 3\mathcal H^2(c_s^2 - w)]\Phi = 0$
  from $\Phi = 1$, $\Phi' = 0$ at $\eta_{\rm in} = 10^{-2}/(\max(u,v,1)k)$, Dopri8 at
  `rtol=1e-9, atol=1e-11`, then leading-order WKB: with the friction removed,
  $\omega^2 = k^2f_3 + f_2 - \tfrac12f_1' - \tfrac14f_1^2$, and the continuation
  $X \simeq e^{-y/2}f_3^{-1/4}[C_+e^{ikz} + C_-e^{-ikz}]$, $y = \int f_1$, $z = \int\sqrt{f_3}$, keeps
  **only** $k^2f_3$ in the phase. Matching where
  $|f_2| + \tfrac12|f_1'| + \tfrac14f_1^2 < 10^{-3}k^2f_3$ holds for the rest of the evolution:
  $k\eta_{\rm WKB} \sim 10^2$ in radiation, up to $10^4$ for modes that meet the crossover late.
- **Tensor modes.** Two basis solutions $g_1, g_2$ of $g'' + (k^2 - a''/a)g = 0$ from Cauchy data
  at $\eta_{\rm in}$ with unit Wronskian, the same WKB continuation (exact in radiation, where
  $a''/a = 0$), matched shortly after horizon crossing.
- **Green's function and kernel.** $G(\eta,\bar\eta) = g_1(\eta)g_2(\bar\eta) - g_2(\eta)g_1(\bar\eta)$,
  so the kernel is $I(\eta) = g_1(\eta)I_2 - g_2(\eta)I_1$ with
  $I_i = \frac{k^2}{a(\eta)}\int^{\eta}d\bar\eta\,g_i(\bar\eta)a(\bar\eta)f(\bar\eta)$: the response
  time enters only through $g_{1,2}(\eta)$, $1/a(\eta)$ and the upper limit.
- **Source.** $f = c_1\Phi_1\Phi_2 + c_2(\Phi_1'\Phi_2 + \Phi_1\Phi_2')/\mathcal H + c_3\Phi_1'\Phi_2'/\mathcal H^2$
  with $c_1 = 6(1+w)/(5+3w)$, $c_2 = 6(1+3w)(1+w)/(5+3w)^2$, $c_3 = 3(1+3w)^2(1+w)/(5+3w)^2$, $w$ and
  $\mathcal H$ numerical at $\bar\eta$, $\Phi' = d\Phi/d\bar\eta$ (`core/kernel.py`, `source_f`).
- **Time integral.** `cumulative_trapezoid` of $g_i\,a\,f$ on
  `geomspace(eta_l, eta_c, n)`, $\eta_l = 10^{-2}/(\max(u,v,1)k)$, $\eta_c = 400/k$,
  $n = \mathrm{clip}(45\cdot\max(u,v)\cdot k\eta_c/2\pi,\ 800,\ 2\times10^5)$ (`export_kernels.py`).
- **Oscillation average.** Over the last 8 periods of the external mode before $\eta_c$, on a
  uniform window, by trapezoid, with weight $(k\eta)^2(a_c/a)^2$: the three bilinear averages
  $\overline{g_1^2}$, $\overline{g_2^2}$, $\overline{g_1g_2}$ once per $k$, then
  $\overline{I^2} = I_2^2\,\overline{g_1^2} - 2I_1I_2\,\overline{g_1g_2} + I_1^2\,\overline{g_2^2}$.
  The cross term is kept because $g_1, g_2$ are not in quadrature after a non-radiation epoch.
- **Spectrum and observable.** $\overline{\mathcal P_h} = \tfrac14\int_0^\infty dt\int_0^1 ds\,
  \overline{J^2}/(uv)^2\,\mathcal P_\zeta(ku)\mathcal P_\zeta(kv)$, $J = t(t+2)(1-s^2)I$, on a fixed
  grid: $t \in [10^{-3}, 3\times10^3]$ with 1444 points clustered at the resonance, 24 points in
  $s$, 110 nodes in $k$. Then $\Omega_{\rm GW}h^2 = \Omega_{r,0}h^2\,c_g(k)\,\overline{\mathcal P_h}(k,\eta_c)$
  with $c_g = \tfrac{1}{24}(k/\mathcal H_c)^2\,(g_{*\rho}/g_{*\rho}^0)(g_{*s}^0/g_{*s})^{4/3}$
  carrying the propagation from $\eta_c$ to today.
- **Validation.** Radiation-era benchmark $x_c^2\overline{\mathcal P_h}/A_\zeta^2 = 19.7$ for a
  flat spectrum, reproduced "at the few-per-mille level"; table interpolation floor a few
  $\times10^{-4}$; "the time quadrature contributes an additional uncertainty below the percent
  level" for the most hierarchical configurations.

---

## 2. The map onto this project

| | arXiv:2609.29078 | This project |
|---|---|---|
| Background | Lattice spline, $w_{\min}$ 0.231 | Saikawa–Shirai fits in `QCD_EOS`, $w_{\min}$ 0.250 |
| Scalar sound speed | $dp/d\rho$ | $p/\rho$ via `wPerturbations` — **defect**, companion audit |
| Scalar mode | In $\eta$; leading-order WKB with $k^2f_3$ only in the phase; $k\eta_{\rm WKB} \sim 10^2$–$10^4$ | In $z$; LG with the full $\omega_{\rm eff}^2$ under the root; hand-over $x_T \approx 15.5$ |
| Tensor mode | Cauchy basis $g_1, g_2$ at $\eta_{\rm in}$ | Unit-jump $G_k(z \leftarrow z')$ per $(k, z_{\rm source})$; LG at fixed $z_{\rm response}$ |
| Source term | Kohri–Terada constant-$w$ coefficients with numerical $\mathcal H$ — **inconsistent**, §4.1 | Spec 03 R22: $c_*$ at $z_{\rm init}$, $w_0(z')$ inside $f$, numerical $H(z')$ |
| Time integral | Trapezoid, geometric grid, ≤ 2×10⁵ points | Adaptive Levin over phase groups $\pm\theta_G \pm\theta_q \pm\theta_r$ |
| Response time | Factored analytically; $\eta_c = 400/k$ | One full Levin integral per $(z_{\rm response}, k, q, r)$ over the winnowed response grid to $z_{\rm end}$ |
| Oscillation average | Three window averages of tensor products, once per $k$ | Unplanned; the working idea was to sample $z_{\rm response}$ and fit |
| Loop integral | Fixed $(t, s)$ tables | $(q, \theta)$ per polarisation, spec 03 R35; `OneLoopIntegral` is a stub |
| Observable | $c_g(k)$ with explicit $g_*$ dilution | Separate layer, not built; the dilution is inside $H(z)$ |

---

## 3. The late-time average, and the factorisation we already have

### 3.1 Why the response-redshift dependence factorises in our representation

Two properties of `GkWKBIntegration`'s representation at fixed source redshift $z$, as a function
of response redshift $z'$, and the transposition into `GkSource` cannot undo either.

- **The phase is a definite integral of a source-independent integrand.** Since prompt 06 of
  `GkTk-remedial`, `Quadrature/integrators/WKB_phase_function.py` evaluates
  $\theta(z'; z) = -[k\,\tau.\mathrm{delta}(z_h, z') + \rho.\mathrm{delta}(z_h, z')]$ from the model's
  double-double $\tau$ table and the per-$(\text{model}, k)$ residual table $\rho$; $z_h$ is the
  hand-over point for source $z$. So $\theta(z'; z) = \Theta(z') - \Theta(z_h(z))$ with
  $\Theta \equiv -(k\tau + \rho)$ a single primitive. The initial-data shift $\delta\theta$
  (`GkWKBIntegration.py:457`) is a function of $z$ alone.
- **The amplitude is a product.** `GkWKBIntegration.py:513` gives
  $B(z)\sqrt{H_{\rm init}(z)} \times [H(z')\,\omega(z')]^{-1/2}$.

Hence, exactly within the LG representation,

$$
G_k(z' \leftarrow z) = b(z)\,A(z')\,\sin\!\big(\Theta(z') - \Theta_h(z)\big)
= A(z')\big[\sin\Theta(z')\cdot b\cos\Theta_h - \cos\Theta(z')\cdot b\sin\Theta_h\big],
$$

and the source integral is $A(z')[\sin\Theta(z')\,J_C - \cos\Theta(z')\,J_S]$, with $J_C$ and $J_S$
the integrals of the source against $b\cos\Theta_h$ and $b\sin\Theta_h$, which depend on $z'$
only through the upper limit. `GkSourcePolicyData.py:136` writes the phase at fixed response
redshift as $-k\,\tau.\mathrm{delta}(z_s, z_r) + \varphi(z_s)$: $\Theta(z_r)$ is an additive constant
and everything else is a function of $z_s$, so the factorisation survives the "complicated
remapping". The cross term vanishes for us and not for them because our basis is the late-time
LG basis, in quadrature by construction; theirs is the Cauchy basis at $\eta_{\rm in}$.

### 3.2 An implementation that needs no knowledge of $\Theta(z_r)$

At one late response redshift $z_r$:

1. the existing integral, with $G = A_G\sin\theta_G$, gives $J_1 = \sin\Theta_r J_C - \cos\Theta_r J_S$;
2. the quadrature partner, with $G_\perp = A_G\cos\theta_G$ and the same amplitude and phase
   objects, gives $J_2 = \cos\Theta_r J_C + \sin\Theta_r J_S$;
3. $J_1^2 + J_2^2 = J_C^2 + J_S^2$, and the period-averaged $\langle h^2\rangle$ is
   $A(z_r)^2(J_1^2 + J_2^2)/2$ times the existing prefactors.

In `ComputeTargets/phase_groups.py` the partner is the "multiplying by $G$" step with the roles of
$P$ and $Q$ swapped and signs adjusted: $\cos\theta_G\cos\Theta = \tfrac12[\cos(\theta_G+\Theta) +
\cos(\theta_G-\Theta)]$, $\cos\theta_G\sin\Theta = \tfrac12[\sin(\theta_G+\Theta) - \sin(\theta_G-\Theta)]$.
Two conditions on $z_r$: below the hand-over of every source redshift, so every $G$ is LG in the
response variable (the lower branch of the hand-over mechanism guarantees this a few e-folds
inside the horizon); and past the emission window. A built-in check: $(J_1^2 + J_2^2)\,H(z_r)\,\omega(z_r)$
is independent of $z_r$ once the source has decayed, so two response redshifts per $k$ verify
convergence of the emission and the LG regime at once.

### 3.3 What it does to the object graph

- **`QuadSourceIntegral`.** Keyed today on $(z_{\rm response}, k, q, r)$ and computed for every
  response redshift in the winnowed grid down to $z_{\rm end}$ (`main.py:3036`). Below the emission
  window every one of those rows carries the same two numbers rotated by $\Theta(z_r)$. The
  rebuilt target computes the pair $(J_1, J_2)$ at one policy-chosen $z_c(k)$ per wavenumber, plus
  one check redshift, and stores $J_1^2 + J_2^2$ with its error. That is two Levin calls per
  $(k, q, r)$ instead of one per $(z_{\rm response}, k, q, r)$. Schema changes are free; the
  `(1+z_{\rm response})` prefactor and the `z_response` key become a policy row.
- **`OneLoopIntegral`.** Consumes $J_1^2 + J_2^2$ per $(k, q, r)$ and does the $(q, \theta)$ integral
  of spec 03 R35, per polarisation, as spec 03 §0.3 fixed. The response redshift disappears from
  its key too. The resonance-scaffolding documents in this folder are about *that* integral's
  nodes; nothing here changes their question, it only fixes what the integrand is.
- **The observable layer.** With $\langle h^2\rangle(z_r) = A(z_r)^2(J_1^2+J_2^2)/2$ and $A(z)^2 \propto
  1/(H(z)\,\omega(z))$ on the true background, $\mathcal P_h$ at any $z$ deep inside the horizon
  follows analytically from the stored pair, and $\Omega_{\rm GW}(z) = \tfrac{1}{24}(k/aH)^2\,\overline{\mathcal P_h}(z)$
  can be evaluated today directly. The paper's $c_g$ factor is what this replaces: its $g_*$
  ratios are the statement that $a(z)$ and $T(z)$ are related by entropy conservation, which
  `QCD_Cosmology` already imposes when it builds $H(z)$. No Green's function needs to be carried
  to $z = 0.1$.
- **The emission-converged criterion** is the one thing the paper supplies that we do not have:
  $\eta_c = 400/k$, six e-folds inside the horizon. Their Fig. 8 shows that for squeezed
  configurations the running integral grows as $\ln(k_*\bar\eta)$ until $k\bar\eta \approx 1$ and
  then saturates by cancellation, so the same criterion serves the infrared tail. Our policy
  should measure this rather than adopt it: the check in §3.2 is the instrument.

---

## 4. Findings about their calculation

### 4.1 The source coefficients

Their source term $S = 2\Phi\Phi + \tfrac{4}{3(1+w)}(\Phi'/\mathcal H + \Phi)(\Phi'/\mathcal H + \Phi)$
is exact for any instantaneous $w$; it comes from $2\mathcal H^2M_P^2/(a^2(\rho+p)) = 2/(3(1+w))$.
Expanding it fixes the coefficient ratios without reference to any normalisation:

| Term | From $S$, any $w(\bar\eta)$ | Paper eq. for $f$ and `source_f` |
|---|---|---|
| $\Phi\Phi$ | $N\cdot 2(5+3w)/(3(1+w))$ | $6(1+w)/(5+3w)$ |
| $(\Phi'\Phi + \Phi\Phi')/\mathcal H$ | $N\cdot 4/(3(1+w))$ | $6(1+3w)(1+w)/(5+3w)^2$ |
| $\Phi'\Phi'/\mathcal H^2$ | $N\cdot 4/(3(1+w))$ | $3(1+3w)^2(1+w)/(5+3w)^2$ |

The mixed-to-$\Phi\Phi$ ratio is $2/(5+3w)$ from $S$ and $(1+3w)/(5+3w)$ from the paper; they agree
only at $w = 1/3$. The origin is Kohri–Terada eq. (16), written in $\bar x\partial_{\bar x}$ form
"using $\mathcal H = 2/((1+3w)\eta)$", so that $\Phi'/\mathcal H = \tfrac{1+3w}{2}\bar x\partial_{\bar x}\Phi$
and the $(1+3w)/2$ migrates into the coefficient. Reading $\bar x\partial_{\bar x}$ back as
$\partial_{\bar\eta}/\mathcal H$ with a numerical $\mathcal H$ restores the whole of $1/\mathcal H$ and
counts $(1+3w)/2$ twice per derivative. Separately, their $\Phi\Phi$ coefficient equals
$[3(1+w)/(5+3w)]^2 \times 2(5+3w)/(3(1+w))$, the $\zeta$-to-$\Phi$ factor at the running $w(\bar\eta)$,
whereas $\Phi_k \to 1$ at $\eta_{\rm in}$ in radiation fixes $N = (2/3)^2$ once. Neither $\Phi'$ nor
$\mathcal H$ is computed wrongly; the coefficients in front of them are constant-$w$ objects.

At their $w_{\min} = 0.231$: ratios of their coefficients to the correct ones **0.947, 0.801,
0.678**, the last two being $0.947 \times (1+3w)/2$ and $0.947 \times ((1+3w)/2)^2$. The
$\Phi'\Phi'/\mathcal H^2$ term dominates sub-horizon emission, so the contamination is of the same
order as the ±35–55 % modulation the paper reports and acts during the epoch that produces it.
Every factor is 1 at $w = 1/3$, so the radiation-era benchmark cannot see it. The correct $f$ in
their normalisation is
$\tfrac49\big[\tfrac{2(5+3w)}{3(1+w)}\Phi_1\Phi_2 + \tfrac{4}{3(1+w)}\big((\Phi_1'\Phi_2 + \Phi_1\Phi_2')/\mathcal H + \Phi_1'\Phi_2'/\mathcal H^2\big)\big]$,
which is `QuadSource.source_function` with $c_*^2$ pulled outside (spec 03 §0.2, audit QS-1).

### 4.2 The time quadrature, measured

In exact radiation every ingredient is closed form. `fpt_rd_quadrature.py` replicates their grid
rule exactly and scores the basis-invariant $J_s^2 + J_c^2$ against composite Gauss–Legendre with
8 panels per period of the fastest leg and 16 nodes, itself checked against 12 panels to
$10^{-14}$.

| $t$ ($s = 0$) | Their points | Points per period in the last decade | Relative error |
|---|---|---|---|
| 0.1 | 1.6k | 6 | $6\times10^{-2}$ |
| 0.3 | 1.9k | 7 | $2\times10^{-2}$ |
| 0.73, the resonance | 2.5k | 9 | $1\times10^{-3}$ |
| 1.5 | 3.6k | 13 | $8\times10^{-3}$ |
| 3 | 5.7k | 18 | $2\times10^{-6}$ |
| 10 | 16k | 16 | $2\times10^{-4}$ |
| 30 to 300 | 44k to 200k, capped | 16 to 18 | $1$–$3\times10^{-5}$ |
| 1000 | 200k, cap short 7× | — | $2\times10^{-3}$ |
| 3000 | 200k, cap short 21× | — | $5\times10^{-3}$ |

$s = 0.7$ is the same picture. Reading "fastest frequency" as $\max(u, v, 1)$ halves the small-$t$
numbers and changes nothing else. The mechanism is the grid, not the cap: 45 points per period
is a budget over the whole span, and a geometric grid over 4.6 decades leaves 6 to 13 of them in
the last decade, where the resonant and late contributions live. The errors alternate in sign
and average down over a flat spectrum, which is consistent with their few-per-mille benchmark;
they exceed the few $\times10^{-4}$ table floor the paper quotes, and they are far below the
paper's headline effect.

### 4.3 The WKB continuation

Both codes are leading-order Liouville–Green: amplitude $\omega^{-1/2}$, phase $\int\omega$. The
difference is what is under the root. We keep $\omega_{\rm eff}^2 = \omega_0^2 + C$, where $C$ is
their $f_2 - \tfrac12f_1' - \tfrac14f_1^2$; they drop $C$ from the phase after matching.

- Tensor sector: exact in radiation for both (their $a''/a = 0$; our $\epsilon = 2$ makes the LG
  form reproduce $\sin k\tau$ with constant amplitude, audit GK-4's $3.8\times10^{-12}$). On QCD
  both truncate, ours one order higher.
- Scalar sector: neither exact in radiation. Keeping $C$ matches the Bessel phase through $O(1/x)$
  and errs at $O(x^{-3})$, the `[00-tk-lg-truncation-floor]` $1.4\times10^{-4}$ at $x_T \approx 15.5$.
  Dropping $C$ errs at $O(1/x)$: for order $3/2$ exactly $1/x$, about $7\times10^{-3}$ rad at their
  radiation-era matching point $x \approx 134$. The neglected term decays as $x^{-2}$, so the phase
  error is set by where the hand-over sits, for both codes. Their decade-shift test covers the
  tensor modes only.

### 4.4 Smaller items

Both codes neglect neutrino free-streaming damping. Their $\eta_{\rm in} = 10^{-2}/(\max(u,v,1)k)$ is
4.6 e-folds before entry, against our `OutsideHorizonEfolds` of 5. Their input equation of state
has $w_{\min}$ 0.231 to our fits' 0.250; that is the fixture, not the method.

---

## 5. Findings about ours

### 5.1 The sound speed

`wPerturbations` returns $p/\rho$; the adiabatic sound speed of the equilibrium plasma is
$dp/d\rho = 1/(3 + d\ln g_s/d\ln T)$, computable from the $g_s$ fit we already have. On our own
fit the gap reaches 0.050, 23 % of $c_s^2$, and the accumulated sound-phase error after horizon
entry is 0.05–0.6 rad across the PTA band. The companion audit has the measurements, the
consumers, the complications at the fit's branch joins and a campaign sketch. It is exact for
constant $w$ and harmless in ΛCDM, which is why nothing in the tree ever saw it.

### 5.2 The Levin driver on the same integrals

`fpt_levin_comparison.py` builds the radiation-era integrand as phase groups with the hand-over
at $q\eta = 15.5$ per leg, integrates in $\log\eta$, and runs `adaptive_levin_sincos` at production
settings (`quad_atol = 1e-32`, `quad_rtol = 1e-8`, Chebyshev order 24), both quadrature partners.

| $t$ | Their time | Their error | Levin time | Subregion solves | Levin error |
|---|---|---|---|---|---|
| 0.1 | 0.4 ms | $6\times10^{-2}$ | 0.7 s | 36 | $3\times10^{-14}$ |
| 0.73 | 0.5 ms | $1\times10^{-3}$ | 0.5 s | 28 | $2\times10^{-16}$ |
| 10 | 2 ms | $2\times10^{-4}$ | 0.5 s | 28 | $6\times10^{-12}$ |
| 100 | 16 ms | $1\times10^{-5}$ | 0.6 s | 34 | $2\times10^{-12}$ |
| 3000 | 25 ms | $5\times10^{-3}$ | 0.6 s | 40 | $1\times10^{-12}$ |

Levin sits at the double-precision floor at a cost flat in $t$; the driver's own reported error
is conservative by two to five orders. Both timings flatter their sides: theirs omits the
mode-function evaluation on every grid point, ours uses linear phases and rational amplitudes,
so it measures the driver's floor, not production. Production is the handover board's A3 baseline:
on squeezed triples the bisection runs to the depth cap and a row costs 171 s against 0.8 s,
with a median achieved relative error of $2\times10^{-6}$, because the *phase representation*
delivers about six digits. The quadrature method is not our bottleneck; the representation
feeding it is.

### 5.3 The cost structure the factorisation removes

The production design computes a full Levin integral per $(z_{\rm response}, k, q, r)$ over a
response grid of about a dozen redshifts per decade of $1+z$ down to $z_{\rm end}$. Below the
emission window every row is redundant with every other, up to a rotation. The A3 baseline's
unreachable tolerances on squeezed triples are therefore paid roughly a hundred times per
$(k, q, r)$; §3 pays them twice.

---

## 6. Bearing on the `OneLoopIntegral` campaign

Not a plan. A list of what this review fixes, what it leaves open, and what stands in the way.

**Fixed by this review, subject to the author's agreement.**
1. The late-time average is $A(z_r)^2(J_1^2 + J_2^2)/2$ from two Levin integrals at one late $z_r$
   per $k$; there is no oscillation to fit out (§3.2).
2. `QuadSourceIntegral` is rebuilt around the pair; `z_response` leaves its key and becomes a
   policy (§3.3). The existing phase-group machinery is reused with one extension.
3. `OneLoopIntegral` consumes $J_1^2 + J_2^2$ per $(k, q, r)$ and owns only the momentum integral
   (spec 03 R35, per polarisation). The response redshift leaves its key too.
4. The observable is $\Omega_{\rm GW}(k, z)$ from $\mathcal P_h$ at $z_c$ and the analytic LG
   propagation on the true background; no $c_g$, no Green's function to $z_{\rm end}$ (§3.3).

**Open, and measurable before planning.**
5. The emission-converged policy $z_c(k)$, and its check redshift; the paper's 400/k is a
   starting point, the invariance of $(J_1^2+J_2^2)H\omega$ is the instrument.
6. Whether the $(q, \theta)$ integral's node placement (the resonance-scaffolding documents in this
   folder) changes when the integrand is the averaged pair rather than the oscillating $h$. It
   should not; it should be checked.
7. Whether `OneLoopIntegral`'s error budget can be stated from the stored per-$(k,q,r)$ errors,
   which the pair's Levin `abserr` supplies but the A3 baseline shows are often unconverged.

**In the way.**
8. `prompts/handover` is in progress. Its A3 remedy is sized for the per-response-redshift
   design; under §3 the same unreachable tolerance is paid twice, not a hundred times, which
   changes the remedy's priority but not its correctness.
9. `prompts/perturbation-sound-speed` must land before any QCD production run. It does not block
   the design.
10. The `OneLoopIntegral` datastore rows on the boards:
    `[02-oneloopintegral-is-a-ninth-keyed-object-type]` (tolerance-convergence, unassigned, the
    user decides) and `[00-oneloop-factory-cannot-round-trip-a-row]` (datastore-integrity). A
    rebuilt key makes the first moot and the second part of the build.
11. `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]` (tolerance-convergence,
    post-campaign): irrelevant to the late-time pair, which reads only the LG form, but it still
    bounds any $h(z)$ wanted *during* emission.

---

## 7. Reproduction

```bash
PYTHONPATH=. ./venv/bin/python docs/one-loop-planning/fpt_rd_quadrature.py
PYTHONPATH=. ./venv/bin/python docs/one-loop-planning/fpt_trapezoid_scan.py
PYTHONPATH=. ./venv/bin/python docs/one-loop-planning/fpt_levin_comparison.py
```

The first two take seconds; the third about half a minute. The coefficient ratios of §4.1 are
one line each in the paper's own notation and need no script.
