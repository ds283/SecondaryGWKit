# Spec 05 — One-loop (P22) induced tensor power spectrum

Transcribed 2026-09-07 from the handwritten notes listed below (campaign README
`prompts/spec-transcription/README.md`, Group 5). Transcription only: nothing here has been
checked against, or adjusted to match, the code.

The **build target** (README §2) is the endpoint of Step 6 of `MAIN` 14 (p.11), *before* the
conversion to "Fabrikant form" in Step 7. It is marked `TARGET` below (R31). Steps 7 of `MAIN` 14
(R32–R33) are transcribed so the audit can see what is **not** to be used.

Sign-off: **Tier 1.1 (Green's-function normalisation), Tier 1.2 (numerical prefactor), Tier 1.3 (form of the one-loop integral), Tier 1.4 (seed $S^* \equiv \zeta^*$), Tier 2.3, 2.4 and the Tier 3 notation items signed off by the author 2026-09-07; see §0. **All review-queue items for this file are closed.**
Remaining Tier 1 and Tier 2 items: *pending author review.*

---

## 0. Author sign-off notes

### 0.1 Tier 1.1 — Green's-function normalisation: **signed off 2026-09-07**

Definitive project convention, fixed by what the code does (full statement and code references in
`02-greens-function.md` §0 and `04-source-integral.md` §0):

- **$a_0$ is absorbed, not set to one.** The code works in $k/a_0$ (physical wavenumber today) and
  $a_0\eta$; every formula is written in these two $a_0$-invariant combinations. Where the notes
  "drop" $a_0$ it is being held back to combine with a comoving momentum into a physical one. Any
  comoving-variable expression must be invariant under $a_0 \to \lambda a_0$, comoving momenta
  $\to \lambda\times$, $\eta \to \eta/\lambda$. Do **not** read an absent $a_0$ as $a_0 = 1$.
- **The code's Green's function** (`ComputeTargets/GkNumericIntegration.py`) is the unit-jump
  causal function for $\chi_s = a h_s$ in redshift: $G(z',z') = 0$, $dG/dz|_{z=z'} = +1$, zero for
  $z > z'$, source $-\delta(z-z')$. It is $\bar G_k$ of `NUM` 03 (spec 03 R25) $= G_{\rm me}$ of
  `NUM` 06 (spec 04 R1–R2), and
  $G_{\rm code}(z,z') = -a_0H(z')\,{\rm Gr}_k(\eta(z),\eta(z'))$ for any $a_0$, where ${\rm Gr}_k$
  is the retarded conformal-time function of `MAIN` 13/14 (spec 05 R18, R21; source
  $+\delta(\eta-\eta')$).
- **Signs.** `NUM` 02's $-\frac{1}{a_0H}\delta(z-z')$ is a convention paired with the orientation
  of $\int dz'$, not a slip. The relative minus sign between the `NUM` 03/06 source integral and the
  `MAIN` 14 target is the same convention and enters the one-loop result squared.
- **$a_0$ hand-off.** `NUM` 06 R14's prefactor should be $a_0^2$ in comoving variables (the page
  writes $a_0^1$; the power was absorbed in R2 while R13 kept it explicit). With $a_0^2$ the
  $Q_s/a_0^2$ of spec 03 R28 cancels exactly and the `NUM` 03/06 source integral is
  $-Q_s/c^2$ times the `MAIN` 14 target, $c^2 = (2+b)^2/(3+2b)^2$.

### 0.2 Tier 1.2 and 1.3 — prefactor and form of the one-loop integral: **signed off 2026-09-07**

- **Prefactor (1.2).** The 32 of R12/R14/R23 is $2\times4^2$ with $c^4 = \big(\tfrac{3(1+w)}{5+3w}\big)^4$
  inside $f^2$. `NUM` 03 pulls $c_*^2$ outside $f$ and has $2\times(4c_*^2)^2 = 2592\big(\tfrac{1+w^*}{5+3w^*}\big)^4$;
  $32\times81 = 2592$. Full chain and the two-$w$ decision in spec 03 §0.2.
- **Form (1.3).** R23 is the loop integral before the measure is split. Splitting $d^3q$, inserting
  $Q_\pm^2 = \tfrac12q^4\sin^4\theta\{\cos^2,\sin^2\}2\varphi$ and doing $\int d\varphi$ gives
  $P^h_{22} = 8\pi^2\int_0^\infty q^3dq\int_0^\pi d\theta\,\sin^5\theta\,\mathcal P^*(q)\mathcal P^*(r)r^{-3}\,(\text{R31})^2$,
  which is `NUM` 03 R35 with the corrected $648\pi^2$ exactly (spec 03 §0.3). **The build form is
  `NUM` 03 R35 as completed in spec 03 §0.3**; R23/R31 here are its conformal-time, fixed-$w$
  counterpart and the analytic check on it.
- **Deliverable.** Per-polarisation $P^h_{22}(k)$ labelled by $s$ ("for any $s$", p. 6) is the stored
  quantity. $\Omega_{\rm GW}$ is built in a separate, decoupled layer (spec 03 §0.3 for the reason).

### 0.3 Tier 1.4 — identity of the primordial seed: **signed off 2026-09-07**

The seed written $S^*$ in this transcription is $\zeta^*$, the primordial curvature perturbation,
identical to the glyph `NUM` 03 p. 4 writes (spec 03 R20); the letter was mis-read from the
handwriting (diff-05 D1 resolved in favour of the duplicate's $\zeta^*$). $P_*(q) = P_\zeta(q)$. The
linear relation is $\phi = \tfrac{3(1+w^*)}{5+3w^*}\,T\,\zeta^*$ with $T \to 1$ at $z_{\rm init}$; the
initial spectrum is supplied in $\zeta$ so that `PyTransport`/`CppTransport` output can be fed in. Sign
convention immaterial (only $P_\zeta$ enters). Full statement in spec 03 §0.4.

**All four Tier 1 items are now signed off for this file.**

### 0.4 Tier 2 and Tier 3 items — **signed off 2026-09-07**

- **2.3 (MAIN 14 p.10–11).** The exponent of 2 was changed on the page from $2+3b$ to $2+2b$; $2^{2+2b}$ confirmed
  (and forced by `NUM` 06's $\tfrac\pi2\,2^{3+2b}$).
- **2.4 (MAIN 14 p.8 bottom, p.9, p.10).** The Bessel orders of the four-term $f$ are as read: the third
  term is the $q\leftrightarrow r$ exchange of the second, $J_{5/2+b}(q\eta c_s)J_{3/2+b}(r\eta c_s)$. Every
  $J_{5/2}$ on p.10 is $J_{5/2+b}$. **$k$-for-$r$:** the author could not locate this slip. The transcribers'
  location (§5 table) is the *first* completed-square line on **p.9**, prefactor written
  $(q\eta c_s)(k\eta c_s)\,J_{\frac12+b}(q\eta c_s)J_{\frac12+b}(r\eta c_s)$, where the second factor should
  be $(r\eta c_s)$ and the next line has $r$; separately, on p.10 a $k$ in $(r\eta'c_s)^{-1/2-b}$ is
  over-written to $r$ on the page (duplicate C4). Neither affects R28–R31.
  **Resolved 2026-09-07:** the author has re-read p.9; the page reads $(r\eta c_s)$, as it should. The "$k$" was a
  **transcription error**, not a slip on the page. The §5 table row for MAIN 14 p.9 is annotated accordingly.
- **Tier 3, prime on the transfer function (Q1).** `MAIN` 11's $\Phi' = d\Phi/d\eta$ is an anomaly; treat it
  as such. `MAIN` 14 and `NUM` 06 use $d/dx$, and the project converges on $T_k(z)$ in any case.
- **Tier 3, arguments of $f$.** $f(\mathbf q,\mathbf k-\mathbf q,\eta')$ as written here is the correct form;
  `NUM` 03's occasional $(\mathbf k,\mathbf k-\mathbf q)$ is the slip (spec 03 §0.5).
- **$\eta_0$ / $z_{\rm init}$.** Not a spec constant: it is initial data in the numerical pipeline,
  set per mode as the redshift a fixed number of e-folds before horizon exit
  (`CosmologyConcepts/wavenumber.py`, `z_exit_suph_eN`). The spec records only the conditions it must
  satisfy: (i) $w$ constant at $z_{\rm init}$ (so $w^*$ is defined, spec 03 §0.2); (ii) every mode
  entering the calculation — $k$, $q$ and $r$ — is super-horizon there, so $T\to1$ and $\phi$ is
  constant. The `MAIN` 14 target is insensitive to $\eta_0$: near $\eta'\to0$ its $Y$-kernel integrand
  behaves as $(\eta')^{1}$ and its $J$-kernel integrand as $(\eta')^{2+2b}$, so the $\eta_0$-dependence
  is $O((k\eta_0)^2)$ and the analytic formula may be read with $\eta_0\to0$; the code's
  `analytic_integral` uses the finite $\eta_{\rm init}=\tau(z_{\rm init})$ in any case.

All review-queue items for this file are closed.

---

## 1. Source documents

| Tag | File | Pages | Date on p.1 | Role |
|---|---|---|---|---|
| MAIN 11 | `BASE/Green's function formula for P22/11 - 2022:07:18 - 22 power spectrum.pdf` | 10 | 18 July 2022 | "Build evolution equation for the tensor perturbation" — first derivation of the P22 formula |
| MAIN 14 | `BASE/Green's function formula for P22/14 - 2022:08:30 - recheck P22 power spectrum.pdf` | 12 | 30 Aug 2022 | "Recheck formula for the induced tensor power spectrum" — recheck of MAIN 11 plus Steps 5–7 (explicit Bessel form of the source, collection of the time integral, Fabrikant form). **Primary build target.** |

`BASE = /Users/ds283/Library/CloudStorage/Box-Box/Research projects/SIGWs/4D/Calculations`.

Step labels used in this file are the author's own ("STEP 1", …). Note the two documents number
their steps differently: MAIN 11 has Steps 1–5, MAIN 14 has Steps 1–7, and e.g. MAIN 11 Step 4
(Green's function) corresponds to MAIN 14 Step 2.

---

## 2. Conventions in force

| Item | Convention | Where stated | Notes |
|---|---|---|---|
| Time variable | Conformal time $\eta$ throughout both documents. | MAIN 11 p.1 ($\partial_\eta$), p.6; MAIN 14 p.1 | Never $t$, $z$ or $\log(1+z)$. |
| Prime on $h_s$, $\phi$, $a$ | $d/d\eta$. | MAIN 11 p.2–3 ($\phi'+\mathcal H\phi$), p.6 ($a'/a$); MAIN 14 p.1–2 | |
| Prime on the transfer function $\Phi$ | **MAIN 14:** $\Phi'(x) = d\Phi/dx$ with $x = q\eta$; chain-rule factors $q$, $r$ written explicitly (MAIN 14 p.1, and $\Phi'(x)$ computed explicitly on p.8). **MAIN 11:** pp.5–6 write $\Phi'(q\eta)$ with factors of $\eta$ but *without* factors of $q$, $r$, i.e. there the prime behaves as $d/d\eta$. | MAIN 14 p.1, p.8; MAIN 11 p.5–6 | Not stated explicitly in either document; inferred. See Checks §4 and Open questions §6. |
| $\mathcal H$ vs $H$ | $\mathcal H = a'/a$, $H = \dot a/a = a'/a^2 = \mathcal H/a$. | MAIN 11 p.4 | |
| Background | $3H^2 M_P^2 = 3\mathcal H^2 M_P^2/a^2 = \rho_0 \Rightarrow \mathcal H^2 M_P^2/a^2 = \rho_0/3$; $p_0 = w\rho_0$ used to get $2\mathcal H^2M_P^2/(a^2(\rho_0+p_0)) = 2/(3(1+w))$. | MAIN 11 p.4 | $M_P$ is the reduced Planck mass (inferred from $3H^2M_P^2=\rho_0$). |
| Fixed-$w$ epoch | $\mathcal H = \dfrac{2}{1+3w}\dfrac{1}{\eta}$; hence $a(\eta')/a(\eta) = (\eta'/\eta)^{1+b}$ (used without comment). | MAIN 11 p.6; MAIN 14 p.1, p.10 | $\epsilon$ is not used. |
| Scale-factor normalisation | Only the ratio $a(\eta')/a(\eta)$ ever appears. $a_0$ is never mentioned. | MAIN 11 p.7; MAIN 14 p.3, p.10 | Not stated; no normalisation is needed for the results. **Author note (2026-09-07):** the code matches this by absorbing $a_0$ into $k/a_0$ and $a_0\eta$, never by setting $a_0 = 1$ (§0). |
| Green's function | For $\chi_s \equiv a\,h_s$: $\mathrm{Gr}_k''(\eta,\eta') + \big(k^2 - a''/a\big)\mathrm{Gr}_k(\eta,\eta') = +\delta(\eta-\eta')$. First argument $\eta$ = response time, second $\eta'$ = source time; retarded ($=0$ for $\eta<\eta'$). Explicit form R23. Depends only on $k=|\mathbf k|$ (so $\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}$). | MAIN 14 p.3–4, p.6; MAIN 11 p.7–8 | $h_s = \int_{\eta_0}^{\eta} d\eta'\,\frac{a(\eta')}{a(\eta)}\mathrm{Gr}_k(\eta,\eta')\,[\text{source}]$. No "literature" Green's function is referenced. |
| Equation of state | $w$; $b = \dfrac{1-3w}{1+3w}$; derived identities $1+3w = \dfrac{2}{1+b}$, $1+w = \dfrac{2(2+b)}{3(1+b)}$, $5+3w = \dfrac{2(3+2b)}{1+b}$. | MAIN 14 p.4, p.7 | $c_s$ appears in $\Phi(x)$ (MAIN 14 p.4, p.8) but is **not defined** in either document (no $c_s^2 = w$ or $c_s^2 = (1-b)/(3(1+b))$ is written). |
| Fourier convention | $h_{ij}(\mathbf x) = \displaystyle\int\frac{d^3k}{(2\pi)^3}\, e^{i\mathbf k\cdot\mathbf x}\sum_s e^s_{ij}(\mathbf k)\, h_s(\mathbf k)$. | MAIN 11 p.2–3 | Same convention implied for $\phi_{\mathbf q}$ ($\int d^3q\,d^3r/(2\pi)^6\, e^{i\mathbf q\cdot\mathbf x}e^{i\mathbf r\cdot\mathbf x}$, MAIN 11 p.3). |
| Polarisation tensors | $e_+^{lm} = \tfrac{1}{\sqrt2}(e^l e^m - \bar e^l\bar e^m)$, $e_\times^{lm} = \tfrac{1}{\sqrt2}(e^l\bar e^m + \bar e^l e^m)$, with $\mathbf e,\bar{\mathbf e}$ unit vectors $\perp\mathbf k$; normalisation $e_s^{ij}e_{ij\,s'} = \delta_{ss'}$. | MAIN 11 p.3, p.9; MAIN 14 p.3 | The author notes $h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$ relative to Kohri & Terada (MAIN 11 p.4) and that Adshead et al.'s $h_{ij}$ "is half the one used here" (MAIN 11 p.10). See Open questions. |
| Power-spectrum convention | Dimensionful $P$: $\langle S_{\mathbf q}S_{\mathbf q'}\rangle = (2\pi)^3\delta(\mathbf q+\mathbf q')P(q)$ (inferred from the four-point function, R12/R24); $\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,\delta_{ss'}\,P^h_{22}(k)$. No dimensionless $\mathcal P = k^3P/2\pi^2$ is used anywhere. | MAIN 11 p.7–8; MAIN 14 p.5–6 | Two-point convention never written explicitly; inferred. MAIN 11 writes $P_S$; MAIN 14 writes $P_*$. |
| Scalar variable and transfer function | $\phi$ is the scalar metric perturbation of the author's own metric (nonlinear definition differs from Adshead et al.; MAIN 11 p.2, p.4). Linear solution $\phi_{\mathbf q}(\eta) = \dfrac{3(1+w)}{5+3w}\Phi(q\eta)\,S^*_{\mathbf q}$, with $\Phi(q\eta)\to1$ as $q\eta\to0$. Explicit $\Phi(x) = 2^{3/2+b}\Gamma(\tfrac52+b)(xc_s)^{-3/2-b}J_{3/2+b}(xc_s)$. | MAIN 11 p.5 (with $S(\mathbf k)$); MAIN 14 p.1, p.4, p.8 | $S^*$ (MAIN 14) / $S$ (MAIN 11) is **not defined** in either document; it is the early-time seed amplitude such that $\phi\to\frac{3(1+w)}{5+3w}S^*$. Its relation to $\zeta$ or $\mathcal R$ is not stated. MAIN 14 p.10 remarks that Domènech uses "$\Phi^*$ rather than $S^*$ as we do". |
| Momentum labels | External $\mathbf k$; loop momentum $\mathbf q$; $\mathbf r = \mathbf k - \mathbf q$ (MAIN 14 p.1 "where $\mathbf r = \mathbf k-\mathbf q$"); second copy $\mathbf t$, $\mathbf u = \mathbf k'-\mathbf t$ (MAIN 14 p.4); $\mathbf k' = -\mathbf k$ after the $\delta$-function. Angles: $\Theta$ = polar angle of $\mathbf q$ relative to $\mathbf k$, $\phi$ = azimuth in the plane $\perp\mathbf k$ (MAIN 11 p.9). | MAIN 11 p.3, p.7–9; MAIN 14 p.1, p.4, p.6 | **The angular reduction of $\int d^3q/(2\pi)^3$ is not carried out in either document**: the final results are left as $\int d^3q/(2\pi)^3$ with $Q_s(\mathbf k,\mathbf q)^2 = \tfrac12 q^4\sin^4\Theta\cos^2 2\phi$ (or $\sin^2 2\phi$). Only the statement that the $\phi$-integral over $2\pi$ yields $\delta_{ss'}$ is made (MAIN 14 p.7). No $(u,v)$ or $(q,r)$ change of variables appears. |
| Polarisation sum | Results are **per polarisation** $s$ ("for any $s$", MAIN 14 p.6). No sum over $s$, no $\Omega_{\rm GW}$, and no $h_{ij}$ total spectrum is formed. | MAIN 14 p.6 | |
| Symmetry factor | Factor $32 = 2\times 16 = 2\times 4^2$: the $4$ from the source normalisation, the $2$ from the two Wick pairings. | MAIN 11 p.8; MAIN 14 p.5–6 | |
| Lower limit $\eta_0$ | All time integrals run $\int_{\eta_0}^{\eta}$; $\eta_0$ not defined. | MAIN 11 p.10; MAIN 14 p.3–6, p.10–12 | |

---

## 3. Results

### 3.1 MAIN 11 — `11 - 2022:07:18 - 22 power spectrum.pdf` (10 pp.)

**R1** — MAIN 11 p.1 (Step 1). Tensor kinetic terms in the Einstein tensor, with $\gamma_{ij} = (e^h)_{ij}$, ${\rm tr}\,h = 0$ (so $\det\gamma = 1$), dropping non-transverse terms ($R_{jk}\to -\tfrac12\partial^2 h_{jk}$, $R\to 0$):
$$
G^i{}_j \supset \frac{1}{2a^2}\Big\{\partial_\eta^2 h^i{}_j + 2\mathcal H\,\partial_\eta h^i{}_j - \partial^2 h^i{}_j\Big\}.
$$
What: $O(h)$ part of $G^i{}_j$. Confidence: high.

**R2** — MAIN 11 p.2 (Step 2). The $G^i{}_j$ equation, after applying the transverse-traceless projector $\mathcal P_{ij}{}^{ab}$:
$$
\partial_\eta^2 h_{ij} + 2\mathcal H\,\partial_\eta h_{ij} - \partial^2 h_{ij}
= \mathcal P_{ij}{}^{ab}\left\{4\,\partial_a\phi\,\partial_b\phi + \frac{8M_P^2}{a^2(\rho_0+p_0)}\,\partial_a(\phi'+\mathcal H\phi)\,\partial_b(\phi'+\mathcal H\phi)\right\}.
$$
What: real-space sourced tensor equation. Author's annotation: "This apparently: MATCHES Domènech (2021) Eq. (3.15); DOES NOT MATCH Adshead et al. (2021) Eq. (A.6) because of our different definition for $\phi,\Psi$ at nonlinear level. They have an extra term involving $\phi\,\partial_i\partial_j\phi$ from $e^{2\Psi}(\nabla^i\nabla_j\Psi - \nabla^i\nabla_j\phi)$ (recall the nonlinear parts of $\Psi$ and $\phi$ have opposite sign)." Confidence: high for this line. (The preceding unprojected line has the second term as $\frac{4M_P^2}{a^{?}}\frac{1}{\rho_0+p_0}\partial^i(\phi'+\mathcal H\phi)\partial_j(\phi'+\mathcal H\phi)$ with the power of $a$ written as something like $a^{2+}$, presumably $a^4$; medium — but only the projected line above is used later.)

**R3** — MAIN 11 p.2–3 (Step 3). Fourier decomposition and polarisation normalisation:
$$
h_{ij} = \int\frac{d^3k}{(2\pi)^3}\,e^{i\mathbf k\cdot\mathbf x}\,\Big\{e_{ij}(\mathbf k)h_{\mathbf k}(\eta) + \bar e_{ij}(\mathbf k)\bar h_{\mathbf k}(\eta)\Big\}
= \int\frac{d^3k}{(2\pi)^3}\,e^{i\mathbf k\cdot\mathbf x}\sum_s e^s_{ij}(\mathbf k)\,h_s(\mathbf k),
\qquad e_s^{ij}e_{ij\,s'} = \delta_{ss'} ,
$$
with $e_{ij}\to e^+_{ij}$, $\bar e_{ij}\to e^\times_{ij}$, $h_{\mathbf k}\to h^+_{\mathbf k}$, $\bar h_{\mathbf k}\to h^\times_{\mathbf k}$ (arrows on the page). "Fixes normalisation of $e^s_{ij}$." Confidence: high.

**R4** — MAIN 11 p.3. Fourier-space equation for a single polarisation (no sum on $s$), after $\int d^3x$, the $\delta(\mathbf k-\mathbf q-\mathbf r)$, and using $e_s^{lm}q_l(k-q)_m = -e_s^{lm}q_lq_m$:
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s
= 4\int\frac{d^3q}{(2\pi)^3}\,e_s^{lm}(\mathbf k)\,q_lq_m\left\{\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q} + \frac{2\mathcal H^2M_P^2}{a^2(\rho_0+p_0)}\Big(\phi+\frac{\phi'}{\mathcal H}\Big)_{\mathbf q}\Big(\phi+\frac{\phi'}{\mathcal H}\Big)_{\mathbf k-\mathbf q}\right\}.
$$
Confidence: high.

**R5** — MAIN 11 p.4. Background relations: $\mathcal H = a'/a$, $H = \dot a/a = a'/a^2 = \mathcal H/a$, $3H^2M_P^2 = 3\mathcal H^2M_P^2/a^2 = \rho_0 \Rightarrow \mathcal H^2M_P^2/a^2 = \rho_0/3$. Hence
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s
= 4\int\frac{d^3q}{(2\pi)^3}\,e_s^{lm}(\mathbf k)\,q_lq_m\left\{\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q} + \frac{2}{3(1+w)}\Big(\phi+\frac{\phi'}{\mathcal H}\Big)_{\mathbf q}\Big(\phi+\frac{\phi'}{\mathcal H}\Big)_{\mathbf k-\mathbf q}\right\}.
$$
Author: "Matches Kohri & Terada Eq. (8) after adjusting for the normalization of $h_{ij}$, $h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$." Notes on p.4–5: Domènech uses basically the same metric and gets the same source term; Adshead et al. use $e^{2\phi}\to(1+2\phi)$ and get $G^i{}_j\supset 4\phi\partial^i\partial_j\phi + 2\partial^i\phi\partial_j\phi$, which is equivalent because $=4\partial^i(\phi\partial_j\phi) - 2\partial^i\phi\partial_j\phi$ and the first term $\sim k^i(\phi\partial_j\phi)$ contracts to zero with a polarisation tensor; Kohri & Terada use a metric like Adshead et al. but write a source term like ours. Confidence: high.

**R6** — MAIN 11 p.5. Expanded source (green annotation: the factor $e_s^{lm}(\mathbf k)q_lq_m$ is "$Q_s$ in notation of Adshead et al."):
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s
= 4\int\frac{d^3q}{(2\pi)^3}\,e_s^{lm}(\mathbf k)\,q_lq_m\left\{\frac{5+3w}{3(1+w)}\,\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q}
+ \frac{2}{3(1+w)}\left(\frac{\phi_{\mathbf q}\phi'_{\mathbf k-\mathbf q} + \phi'_{\mathbf q}\phi_{\mathbf k-\mathbf q}}{\mathcal H} + \frac{\phi'_{\mathbf q}\phi'_{\mathbf k-\mathbf q}}{\mathcal H^2}\right)\right\}.
$$
Confidence: high.

**R7** — MAIN 11 p.5. Transfer-function substitution ("Now if"):
$$
\phi(\mathbf k,\eta) = \frac{3(1+w)}{5+3w}\,\Phi(k\eta)\,S(\mathbf k),
$$
giving
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s
= 4\int\frac{d^3q}{(2\pi)^3}\,e_s^{lm}(\mathbf k)q_lq_m\,S(\mathbf q)S(\mathbf k-\mathbf q)
\left\{\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta) + \frac{6(1+w)}{(5+3w)^2}\left[\frac{\Phi(q\eta)\Phi'(r\eta) + \Phi'(q\eta)\Phi(r\eta)}{\mathcal H} + \frac{\Phi'(q\eta)\Phi'(r\eta)}{\mathcal H^2}\right]\right\}.
$$
Here $r = |\mathbf k-\mathbf q|$ (not defined on this page; defined in MAIN 14 p.1). $S(\mathbf k)$ is not defined. Confidence: high for the formula; see Conventions for the meaning of $\Phi'$ in this document.

**R8** — MAIN 11 p.6. Assuming $\mathcal H = \dfrac{2}{1+3w}\dfrac1\eta$:
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s
= 4\int\frac{d^3q}{(2\pi)^3}\,e_s^{lm}(\mathbf k)q_lq_m\,S(\mathbf q)S(\mathbf k-\mathbf q)
\left\{\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta) + \frac{3(1+w)(1+3w)\,\eta}{(5+3w)^2}\Big[\Phi'(q\eta)\Phi(r\eta) + \Phi(q\eta)\Phi'(r\eta)\Big] + \frac32\frac{(1+w)(1+3w)^2}{(5+3w)^2}\,\eta^2\,\Phi'(q\eta)\Phi'(r\eta)\right\}
$$
$$
\equiv 4\int\frac{d^3q}{(2\pi)^3}\,Q_s\,S(\mathbf q)S(\mathbf k-\mathbf q)\,f(\mathbf q,\mathbf r,\eta).
$$
What: defines $f(\mathbf q,\mathbf r,\eta)$ (MAIN 11 form; contrast R20). The denominator of the middle term originally read $(5+3w)^2(1+3w)$ with the $(1+3w)$ struck through (see §5). Note the absence of factors $q$, $r$ multiplying $\eta$: in this document $\Phi'$ evidently means $d/d\eta$ (Conventions; Checks). Confidence: high for the glyphs.

**R9** — MAIN 11 p.6–7 (Step 4). Rescaling: let $h_s = f(\eta)\chi_s(\eta)$ and choose $f'+\frac{a'}{a}f = 0 \Rightarrow f = 1/a$. Then $f''/f + 2\frac{a'}{a}\frac{f'}{f} = -a''/a$ and
$$
\chi_s'' + \Big(k^2 - \frac{a''}{a}\Big)\chi_s = 4a(\eta)\int\frac{d^3q}{(2\pi)^3}\,Q_s\,S_{\mathbf q}S_{\mathbf k-\mathbf q}\,f(q,\eta,\dots).
$$
Confidence: high.

**R10** — MAIN 11 p.7. Green's-function solution:
$$
h_s = \int d\eta'\,\frac{1}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\,4a(\eta')\int\frac{d^3q}{(2\pi)^3}Q_sS_{\mathbf q}S_{\mathbf k-\mathbf q}f(q,\eta',\dots)
= 4\int d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\int\frac{d^3q}{(2\pi)^3}\,Q_s\,S_{\mathbf q}S_{\mathbf k-\mathbf q}\,f(q,r,\eta').
$$
(The page marks $\mathbf k-\mathbf q$ with "$r$" underneath.) $\mathrm{Gr}_k$ is not defined explicitly in MAIN 11; see R23 for its definition in MAIN 14. Confidence: high.

**R11** — MAIN 11 p.7. Two-point function of $h_s$ and the four-point function of $S$:
$$
\langle h_s(\mathbf k)h_s(\mathbf k')\rangle = 16\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)Q_s(\mathbf k',\mathbf t)\,\langle S_{\mathbf q}S_{\mathbf k-\mathbf q}S_{\mathbf t}S_{\mathbf k'-\mathbf t}\rangle
\int d\eta'\frac{a(\eta')}{a(\eta)}\mathrm{Gr}_k(\eta,\eta')f(\mathbf q,\mathbf k-\mathbf q,\eta')\int d\eta''\frac{a(\eta'')}{a(\eta)}\mathrm{Gr}_{k'}(\eta,\eta'')f(\mathbf t,\mathbf k'-\mathbf t,\eta''),
$$
$$
\langle S_{\mathbf q}S_{\mathbf k-\mathbf q}S_{\mathbf t}S_{\mathbf k'-\mathbf t}\rangle
= (2\pi)^6\,\delta(\mathbf q+\mathbf t)\,\delta(\mathbf k+\mathbf k'-\mathbf q-\mathbf t)\,P_S(q)P_S(|\mathbf k-\mathbf q|)
+ (2\pi)^6\,\delta(\mathbf q+\mathbf k'-\mathbf t)\,\delta(\mathbf k-\mathbf q+\mathbf t)\,P_S(q)P_S(|\mathbf k-\mathbf q|).
$$
What: fixes the (unstated) two-point convention $\langle S_{\mathbf q}S_{\mathbf q'}\rangle = (2\pi)^3\delta(\mathbf q+\mathbf q')P_S(q)$. A plain "P" is written above each $P_S$ symbol (annotation; see §5). Confidence: high.

**R12** — MAIN 11 p.8. After the $\delta$-functions,
$$
\langle h_s(\mathbf k)h_s(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,32\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)Q_s(\mathbf k',-\mathbf q)\,P_S(q)P_S(|\mathbf k-\mathbf q|)
\left(\int d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\,f(\mathbf q,\mathbf k-\mathbf q,\eta')\right)^2,
$$
"since $\mathbf k' = -\mathbf k$ and: $\mathrm{Gr}_{\mathbf k}$ does not depend on the orientation of $\mathbf k$; $f(\mathbf q,\mathbf r,\eta)$ depends only on $q = |\mathbf q|$, $r = |\mathbf r|$; $f(q,r,\eta) = f(r,q,\eta)$." (The intermediate line has the two Wick terms with $Q_s(\mathbf k',-\mathbf q)f(-\mathbf q,\mathbf k'+\mathbf q,\eta'')$ and $Q_s(\mathbf k',\mathbf q-\mathbf k)f(\mathbf q-\mathbf k,\mathbf k'+\mathbf k-\mathbf q,\eta'')$, the last argument marked "$=-\mathbf q$".) Confidence: high.

**R13** — MAIN 11 p.9 (Step 5). Polarisation factors:
$$
Q_s = e_s^{lm}(\mathbf k)q_lq_m,\qquad
e_+^{lm}(\mathbf k) = \tfrac{1}{\sqrt2}\big(e^le^m - \bar e^l\bar e^m\big),\qquad
e_\times^{lm}(\mathbf k) = \tfrac{1}{\sqrt2}\big(e^l\bar e^m + \bar e^le^m\big),
$$
with $\mathbf e,\bar{\mathbf e}$ unit vectors in the 2-dimensional subspace $\perp\mathbf k$; $\Theta,\phi$ polar angles with $\Theta$ the polar angle relative to $\mathbf k$:
$$
e_+^{lm}(\mathbf k)q_lq_m = \tfrac{1}{\sqrt2}q^2\big\{(\hat{\mathbf q}\cdot\mathbf e)^2 - (\hat{\mathbf q}\cdot\bar{\mathbf e})^2\big\} = \tfrac{1}{\sqrt2}q^2\sin^2\Theta\,(\cos^2\phi-\sin^2\phi) = \tfrac{1}{\sqrt2}\,q^2\sin^2\Theta\,\cos2\phi,
$$
$$
e_\times^{lm}(\mathbf k)q_lq_m = \tfrac{1}{\sqrt2}q^2\big\{(\hat{\mathbf q}\cdot\mathbf e)(\hat{\mathbf q}\cdot\bar{\mathbf e}) + (\hat{\mathbf q}\cdot\bar{\mathbf e})(\hat{\mathbf q}\cdot\mathbf e)\big\} = \tfrac{1}{\sqrt2}\,q^2\sin^2\Theta\,\sin2\phi.
$$
"The $\cos2\phi$, $\sin2\phi$ factors reflect the spin 2 representation of the little group. NOTE $\cos2\phi$, $\sin2\phi$ factors missing in Baumann et al." Confidence: high (there is a small stray mark after "$\cos$" in the $+$ result that could be read as a superscript; MAIN 14 p.4 has plainly $\cos2\phi$).

**R14** — MAIN 11 p.10. Final per-polarisation spectrum:
$$
P_s(k) = 32\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)^2\,P_S(q)\,P_S(|\mathbf k-\mathbf q|)
\left(\int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\,f(\mathbf q,\mathbf k-\mathbf q,\eta')\right)^2.
$$
"Matches Eq. (2.28) of Adshead et al., bearing in mind that: their $f(\mathbf q,\mathbf k-\mathbf q,\eta')$ is twice the one used here; their $h_{ij}$ is half the one used here. These two changes compensate for each other, so we end up with the Adshead et al. result." Note $Q_s(\mathbf k',-\mathbf q)\to Q_s(\mathbf k,\mathbf q)$ has been applied silently here (made explicit in MAIN 14 p.6). Confidence: high.

### 3.2 MAIN 14 — `14 - 2022:08:30 - recheck P22 power spectrum.pdf` (12 pp.)

**R15** — MAIN 14 p.1 (Step 1). Starting point, "the sourced equation for a single polarization":
$$
h_s'' + 2\mathcal H h_s' + k^2h_s = 4\int\frac{d^3q}{(2\pi)^3}\,e_s^{lm}(\mathbf k)q_lq_m
\left\{\frac{5+3w}{3(1+w)}\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q} + \frac{2}{3(1+w)}\left(\frac{\phi_{\mathbf q}\phi'_{\mathbf k-\mathbf q}+\phi'_{\mathbf q}\phi_{\mathbf k-\mathbf q}}{\mathcal H} + \frac{\phi'_{\mathbf q}\phi'_{\mathbf k-\mathbf q}}{\mathcal H^2}\right)\right\}.
$$
Identical to R6. Confidence: high.

**R16** — MAIN 14 p.1. Linear solution for $\phi_{\mathbf q}$:
$$
\phi_{\mathbf q}(\eta) = \frac{3(1+w)}{5+3w}\,\Phi(q\eta)\,S^*_{\mathbf q},\qquad \Phi(q\eta)\to1 \text{ as } q\eta\to0 .
$$
What: defines the transfer function $\Phi$ and its early-time normalisation; $S^*_{\mathbf q}$ is the seed variable (undefined; see Conventions). Confidence: high (the star is written as a superscript on $S$ and as a subscript on $P_*$).

**Author sign-off (2026-09-07):** the seed letter is $\zeta^*$, the primordial curvature perturbation, as the duplicate
transcription read it (diff-05 D1); $P_*(q) = P_\zeta(q)$. See §0.3 and spec 03 §0.4.

**R17** — MAIN 14 p.1–2. Source in terms of $\Phi$, with chain-rule factors, "where $\mathbf r = \mathbf k-\mathbf q$":
$$
h_s'' + 2\mathcal H h_s' + k^2h_s = 4\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,S^*_{\mathbf q}S^*_{\mathbf k-\mathbf q}
\left\{\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta) + \frac{6(1+w)}{(5+3w)^2}\left(\frac{r}{\mathcal H}\Phi(q\eta)\Phi'(r\eta) + \frac{q}{\mathcal H}\Phi'(q\eta)\Phi(r\eta) + \frac{qr}{\mathcal H^2}\Phi'(q\eta)\Phi'(r\eta)\right)\right\}.
$$
Then "take $\mathcal H$ to correspond to an epoch of fixed $w$", $\mathcal H = \dfrac{2}{1+3w}\dfrac1\eta$, giving (p.2)
$$
h_s'' + \frac{4}{1+3w}\frac{h_s'}{\eta} + k^2h_s = 4\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,S^*_{\mathbf q}S^*_{\mathbf k-\mathbf q}
\left\{\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta) + \frac{3(1+w)(1+3w)}{(5+3w)^2}\left(r\eta\,\Phi(q\eta)\Phi'(r\eta) + q\eta\,\Phi'(q\eta)\Phi(r\eta) + \frac{1+3w}{2}\,q\eta\,r\eta\,\Phi'(q\eta)\Phi'(r\eta)\right)\right\}.
$$
Confidence: high.

**R18** — MAIN 14 p.2–3 (Step 2). Green's-function formula. Homogeneous equation $h_s'' + 2\frac{a'}{a}h_s' + k^2h_s = 0$. Choosing $h_s(\eta) = \chi_s(\eta)/a(\eta)$:
$$
\chi_s'' + \Big(k^2 - \frac{a''}{a}\Big)\chi_s = 4a(\eta)\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,S^*_{\mathbf q}S^*_{\mathbf k-\mathbf q}\,f(q,r,\eta),
$$
$$
f(q,r,\eta) = \frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta) + \frac{3(1+w)(1+3w)}{(5+3w)^2}\left(r\eta\,\Phi(q\eta)\Phi'(r\eta) + q\eta\,\Phi'(q\eta)\Phi(r\eta) + \frac{1+3w}{2}\,qr\eta^2\,\Phi'(q\eta)\Phi'(r\eta)\right).
$$
Let $\mathrm{Gr}_k(\eta,\eta')$ be the Green's function for the $\chi_s$ equation,
$$
\mathrm{Gr}''(\eta,\eta') + \Big(k^2 - \frac{a''}{a}\Big)\mathrm{Gr}_k(\eta,\eta') = \delta(\eta-\eta').
$$
Then
$$
h_s = \int_{\eta_0}^{\eta}d\eta'\;4\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,S^*_{\mathbf q}S^*_{\mathbf k-\mathbf q}\,f(q,r,\eta').
$$
(The factor $\mathrm{Gr}_k(\eta,\eta')$ is inserted with an arrow into the $h_s$ line.) What: source function $f$ in $w$-form, Green's function definition, formal solution. Confidence: high.

**Author note (2026-09-07):** this $\mathrm{Gr}_k$ (defined here on p.2–3, explicit retarded form R21 on p.4;
every later $\mathrm{Gr}_k$ in `MAIN` 14 is this object) is the project's reference Green's function. The code
computes the redshift-space unit-jump function $G_{\rm code} = -a_0H(z')\,\mathrm{Gr}_k(\eta(z),\eta(z'))$
(spec 03 R25, spec 04 R2); the factors $a(\eta')/a(\eta) = (1+z)/(1+z')$ and $d\eta' = -dz'/(a_0H)$ are
supplied by the source-integral code. See §0.

**R19** — MAIN 14 p.3–4 (Step 3 ①). Polarisation factors $Q_s(\mathbf k,\mathbf q) = e_s^{lm}(\mathbf k)q_lq_m$ with $e_\pm^{lm}$ as in R13 and $e_s^{lm}e_{s'\,lm} = \delta_{ss'}$:
$$
Q_+(\mathbf k,\mathbf q) = \frac{q^2}{\sqrt2}\sin^2\Theta\,\cos2\phi,\qquad
Q_\times(\mathbf k,\mathbf q) = \frac{q^2}{\sqrt2}\sin^2\Theta\,\sin2\phi .
$$
Confidence: high.

**R20** — MAIN 14 p.4 (Step 3 ②). Transfer function:
$$
\Phi(k\eta) = 2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)\,(kc_s\eta)^{-3/2-b}\,J_{3/2+b}(kc_s\eta),\qquad b = \frac{1-3w}{1+3w}.
$$
(Restated on p.8 as $\Phi(x) = 2^{3/2+b}\Gamma(\tfrac52+b)(xc_s)^{-3/2-b}J_{3/2+b}(xc_s)$.) Confidence: high.

**R21** — MAIN 14 p.4 (Step 3 ③). Green's function:
$$
\mathrm{Gr}_k(\eta,\eta') = \frac{\pi}{2}\sqrt{\eta\eta'}\,\Big\{J_{b+\frac12}(k\eta')\,Y_{b+\frac12}(k\eta) - Y_{b+\frac12}(k\eta')\,J_{b+\frac12}(k\eta)\Big\}\ \ \text{if } \eta>\eta';\qquad 0\ \ \text{if } \eta<\eta'.
$$
Confidence: high.

**R22** — MAIN 14 p.4–5 (Step 4). Two-point function:
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = 16\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,\mathrm{Gr}_k(\eta,\eta')\mathrm{Gr}_{k'}(\eta,\eta'')\,\frac{a(\eta')}{a(\eta)}\frac{a(\eta'')}{a(\eta)}
\times\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k',\mathbf t)\,\langle S^*_{\mathbf q}S^*_{\mathbf k-\mathbf q}S^*_{\mathbf t}S^*_{\mathbf k'-\mathbf t}\rangle\,f(q,r,\eta')\,f(t,u,\eta'').
$$
Wick's theorem: $\langle S^*_{\mathbf q}S^*_{\mathbf k-\mathbf q}S^*_{\mathbf t}S^*_{\mathbf k'-\mathbf t}\rangle = \langle S^*_{\mathbf q}S^*_{\mathbf t}\rangle\langle S^*_{\mathbf k-\mathbf q}S^*_{\mathbf k'-\mathbf t}\rangle + \langle S^*_{\mathbf q}S^*_{\mathbf k'-\mathbf t}\rangle\langle S^*_{\mathbf k-\mathbf q}S^*_{\mathbf t}\rangle + \text{disconnected}$, which on p.5 is written as
$$
(2\pi)^6\,P_*(q)P_*(|\mathbf k-\mathbf q|)\Big\{\delta(\mathbf q+\mathbf t)\delta(\mathbf k-\mathbf q+\mathbf k'-\mathbf t) + \delta(\mathbf q+\mathbf k'-\mathbf t)\delta(\mathbf k-\mathbf q+\mathbf t)\Big\},
$$
so that (p.5, bottom)
$$
\langle h_sh_{s'}\rangle = 16\int\!\!\int d\eta'd\eta''\,\frac{a(\eta')a(\eta'')}{a(\eta)^2}\mathrm{Gr}_k\mathrm{Gr}_{k'}\int\frac{d^3q}{(2\pi)^3}(2\pi)^3\delta(\mathbf k+\mathbf k')P_*(q)P_*(|\mathbf k-\mathbf q|)
\Big\{Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k',-\mathbf q)f(\mathbf q,\mathbf k-\mathbf q,\eta')f(-\mathbf q,\mathbf k'+\mathbf q,\eta'') + Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k',\mathbf k'+\mathbf q)f(\mathbf q,\mathbf k-\mathbf q,\eta')f(\mathbf k'+\mathbf q,-\mathbf q,\eta'')\Big\} + \text{disconnected}.
$$
Confidence: high for the structure; **medium** for the second argument of the second $f$ on p.4, read as $u$ (i.e. $\mathbf u = \mathbf k'-\mathbf t$; the letter is small and could also be read as $s$ — it is not used again).

**R23** — MAIN 14 p.6. Definition of $P^h_{22}$ and its final form. "We always want to drop the disconnected contributions. Then ① $Q_{s'}(\mathbf k',\mathbf k'+\mathbf q) = Q_{s'}(\mathbf k',\mathbf q)$ since $\mathbf e,\bar{\mathbf e}\perp\mathbf k'$. ② $f(q,r,\eta) = f(r,q,\eta)$." Hence
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,P^h_{22}(k),
$$
$$
P^h_{22}(k) = 32\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,\frac{a(\eta')a(\eta'')}{a(\eta)^2}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\mathrm{Gr}_{-\mathbf k}(\eta,\eta'')
\int\frac{d^3q}{(2\pi)^3}P_*(q)P_*(|\mathbf k-\mathbf q|)\,Q_s(\mathbf k,\mathbf q)Q_{s'}(-\mathbf k,-\mathbf q)\,f(\mathbf q,\mathbf k-\mathbf q,\eta')f(-\mathbf q,\mathbf q-\mathbf k,\eta'').
$$
"Now $Q_{s'}(-\mathbf k,-\mathbf q) = Q_{s'}(\mathbf k,\mathbf q)$ and $f(\mathbf q,\mathbf k-\mathbf q,\eta') = f(-\mathbf q,\mathbf q-\mathbf k,\eta')$ & $\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}$", giving
$$
P^h_{22}(k) = 32\int\!\!\int d\eta'd\eta''\,\frac{a(\eta')a(\eta'')}{a(\eta)^2}\mathrm{Gr}_k(\eta,\eta')\mathrm{Gr}_k(\eta,\eta'')\;\delta_{ss'}\int\frac{d^3q}{(2\pi)^3}P_*(q)P_*(|\mathbf k-\mathbf q|)\,Q_s(\mathbf k,\mathbf q)^2\,f(\mathbf q,\mathbf k-\mathbf q,\eta')f(\mathbf q,\mathbf k-\mathbf q,\eta'')
$$
$$
\boxed{\;P^h_{22}(k) = 32\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)^2\,P_*(q)\,P_*(|\mathbf k-\mathbf q|)\left(\int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\,f(\mathbf q,\mathbf k-\mathbf q,\eta')\right)^2\;}\qquad\text{"for any } s\text{."}
$$
(Box added by the transcriber for visibility; not boxed on the page.) p.7 note: "$\int\frac{d^3q}{(2\pi)^3}Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k,\mathbf q)\cdots\propto\delta_{ss'}$ from integration of the $\phi$ dependence over $2\pi$ *if* $f$ is independent of $\phi$." Confidence: high. This is the outer structure into which the `TARGET` time integral (R31) is inserted.

**Author note (2026-09-07):** the 32 here is $2\times4^2$ with $c^4 = \big(\tfrac{3(1+w)}{5+3w}\big)^4$ hidden inside
$f^2$ (R16–R18). `NUM` 03 pulls $c_*^2$ outside $f$ and correspondingly has $2\times(4c_*^2)^2 = 2592\big(\tfrac{1+w^*}{5+3w^*}\big)^4$;
$32\times81 = 2592$, so the two agree. The single $w$ of `MAIN` 11/14 is the fixed-$w$ restriction stated on
`MAIN` 14 p. 2; for a running background $w$ inside $f$ is $w(\eta')$ and $w$ in $c_*$ is $w(z_{\rm init})$.
See spec 03 §0.2 (Tier 1.2, signed off). Splitting the measure here and doing the $\varphi$ integral gives
`NUM` 03 R35 exactly; that is the build form, stored per polarisation $s$ (spec 03 §0.3, Tier 1.3, signed off).

**R24** — MAIN 14 p.7 (Step 5). Identities for rewriting in terms of $b$:
$$
1+3w = \frac{2}{1+b},\qquad b = \frac{1-3w}{1+3w},\qquad 1+w = \frac{2(2+b)}{3(1+b)},\qquad 5+3w = \frac{2(3+2b)}{1+b}.
$$
Confidence: high.

**R25** — MAIN 14 p.7. Source function in $b$-form:
$$
f(q,r,\eta) = \frac{2+b}{3+2b}\,\Phi(q\eta)\Phi(r\eta) + \frac{2+b}{(3+2b)^2}\left\{r\eta\,\Phi(q\eta)\Phi'(r\eta) + q\eta\,\Phi'(q\eta)\Phi(r\eta) + \frac{1}{1+b}\,qr\eta^2\,\Phi'(q\eta)\Phi'(r\eta)\right\}.
$$
Confidence: high.

**R26** — MAIN 14 p.8. Derivative of the transfer function. Using $2Z'_\alpha(x) = Z_{\alpha-1}(x) - Z_{\alpha+1}(x)$ and $\frac{2\alpha}{x}Z_\alpha(x) = Z_{\alpha-1}(x)+Z_{\alpha+1}(x)$ for any Bessel function $Z_\alpha$, so $Z'_\alpha(x) - \frac{\alpha}{x}Z_\alpha(x) = -Z_{\alpha+1}(x)$:
$$
\Phi'(x) = 2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)(xc_s)^{-3/2-b}\,c_s\left\{J'_{3/2+b}(xc_s) - \Big(\tfrac32+b\Big)\frac{1}{xc_s}J_{3/2+b}(xc_s)\right\}
= -\,2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)(xc_s)^{-3/2-b}\,c_s\,J_{5/2+b}(xc_s).
$$
Confidence: high.

**R27** — MAIN 14 p.8. Source function as Bessel products (four terms):
$$
f(q,r,\eta) = \frac{2+b}{3+2b}\,2^{3+2b}\,\Gamma^2\!\left(\tfrac52+b\right)(q\eta c_s)^{-3/2-b}(r\eta c_s)^{-3/2-b}
\Big\{J_{\frac32+b}(q\eta c_s)J_{\frac32+b}(r\eta c_s) - \frac{r\eta c_s}{3+2b}J_{\frac32+b}(q\eta c_s)J_{\frac52+b}(r\eta c_s) - \frac{q\eta c_s}{3+2b}J_{\frac52+b}(q\eta c_s)J_{\frac32+b}(r\eta c_s) + \frac{1}{(3+2b)(1+b)}(q\eta c_s)(r\eta c_s)J_{\frac52+b}(q\eta c_s)J_{\frac52+b}(r\eta c_s)\Big\}.
$$
Confidence: high.

**R28** — MAIN 14 p.9–10. "Complete the square" using $Z_\alpha(x) - \frac{x}{2\alpha}Z_{\alpha+1}(x) = \frac{x}{2\alpha}Z_{\alpha-1}(x)$ (with $2\alpha = 3+2b$), yielding the final two-term form of the source (p.10, top):
$$
\boxed{\;f(q,r,\eta) = \frac{2+b}{(3+2b)^3}\,2^{3+2b}\,\Gamma\!\left(\tfrac52+b\right)^2(q\eta c_s)^{-\frac12-b}(r\eta c_s)^{-\frac12-b}
\Big\{J_{\frac12+b}(q\eta c_s)J_{\frac12+b}(r\eta c_s) + \frac{2+b}{1+b}\,J_{\frac52+b}(q\eta c_s)J_{\frac52+b}(r\eta c_s)\Big\}\;}
$$
(Box added by the transcriber.) The p.9 intermediate line writes the coefficient of the second term as $\big(\frac{3+2b}{1+b}-1\big)$, which p.10 simplifies to $\frac{2+b}{1+b}$. Author: "Matches Domènech Eq. (4.9) up to normalization, which appears to come from him using $\Phi^*$ rather than $S^*$ as we do." Confidence: high. (On p.9 the first "completed" line reads $(q\eta c_s)(k\eta c_s)$ where $(r\eta c_s)$ is clearly intended — the next line has $r$; see §5.)

**R29** — MAIN 14 p.10 (Step 6 "Collect the time integral", first line). Substituting $a(\eta')/a(\eta) = (\eta')^{1+b}/\eta^{1+b}$, R21 and R28:
$$
\int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\,f(q,r,\eta')
= \int_{\eta_0}^{\eta}d\eta'\,\frac{(\eta')^{1+b}}{\eta^{1+b}}\,\frac{\pi}{2}\sqrt{\eta\eta'}\,\Big(J_{b+\frac12}(k\eta')Y_{b+\frac12}(k\eta) - Y_{b+\frac12}(k\eta')J_{b+\frac12}(k\eta)\Big)
\frac{2+b}{(3+2b)^3}\,2^{3+2b}\,\Gamma\!\left(\tfrac52+b\right)^2(q\eta'c_s)^{-\frac12-b}(r\eta'c_s)^{-\frac12-b}
\Big\{J_{\frac12+b}(q\eta'c_s)J_{\frac12+b}(r\eta'c_s) + \frac{2+b}{1+b}J_{\frac52+b}(q\eta'c_s)J_{\frac52+b}(r\eta'c_s)\Big\}.
$$
Confidence: high, with two remarks: (i) the last factor of the Green's function runs off the right margin and is read as $J_{b+\frac12}(k\eta)$ by comparison with R21; (ii) on this line the last two Bessel orders are written $J_{5/2}$ (without "$+b$"); the following line and p.11 write $J_{\frac52+b}$, so the omission is an abbreviation/slip.

**Author sign-off (2026-09-07):** $J_{\frac52+b}$ confirmed (Tier 2.4); see §0.4.

**R30** — MAIN 14 p.10 (Step 6, second line). Pulling out the $\eta'$-independent factors:
$$
= \pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\,\eta^{-\frac12-b}
\int_{\eta_0}^{\eta}d\eta'\,(\eta')^{\frac32+b}\,(q\eta'c_s)^{-\frac12-b}(r\eta'c_s)^{-\frac12-b}
\Big(J_{b+\frac12}(k\eta')Y_{b+\frac12}(k\eta) - Y_{b+\frac12}(k\eta')J_{b+\frac12}(k\eta)\Big)
\Big(J_{\frac12+b}(q\eta'c_s)J_{\frac12+b}(r\eta'c_s) + \frac{2+b}{1+b}J_{\frac52+b}(q\eta'c_s)J_{\frac52+b}(r\eta'c_s)\Big).
$$
Confidence: high for all glyphs except the exponent of 2, which has been over-written (see §5): reading $2+2b$, consistent with $\frac{\pi}{2}\cdot2^{3+2b} = \pi\,2^{2+2b}$ — medium on the glyph, high on the value.

**R31 — `TARGET`** — MAIN 14 p.11 (Step 6, endpoint; top of page). This is the build target (README §2): the time integral that is squared inside R23.
$$
\boxed{\;
\int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_k(\eta,\eta')\,f(q,r,\eta')
= \pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\,\big(c_s^2\,q\,r\,\eta\big)^{-\frac12-b}
\Big(Y_{b+\frac12}(k\eta)\,I_J - J_{b+\frac12}(k\eta)\,I_Y\Big)
\;}
$$
$$
\boxed{\;
I_{J/Y} = \int_{\eta_0}^{\eta}d\eta'\,(\eta')^{\frac12-b}
\begin{Bmatrix} J_{\frac12+b}(k\eta') \\[2pt] Y_{\frac12+b}(k\eta') \end{Bmatrix}
\Big(J_{\frac12+b}(q\eta'c_s)\,J_{\frac12+b}(r\eta'c_s) + \frac{2+b}{1+b}\,J_{\frac52+b}(q\eta'c_s)\,J_{\frac52+b}(r\eta'c_s)\Big)
\;}
$$
where the upper (lower) entry of the brace defines $I_J$ ($I_Y$). Here $q = |\mathbf q|$, $r = |\mathbf k-\mathbf q|$, $b = (1-3w)/(1+3w)$, $c_s$ is the (undefined in these notes) sound speed appearing in $\Phi$, and $\eta_0$ is the (undefined) lower limit. The two boxes are on the page as displayed equations (the boxes themselves are the transcriber's). Confidence: **high** for every glyph on p.11; the only inherited uncertainty is the over-written exponent $2+2b$ noted in R30 and §5, which is arithmetically forced.

*Transcriber's assembly (not written as a single formula anywhere in the notes):* inserting R31 into R23 gives the per-polarisation one-loop spectrum
$$
P^h_{22}(k) = 32\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)^2\,P_*(q)\,P_*(r)\;
\Big[\pi\,2^{2+2b}\,\tfrac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\,(c_s^2qr\eta)^{-\frac12-b}\Big]^2
\Big(Y_{b+\frac12}(k\eta)\,I_J - J_{b+\frac12}(k\eta)\,I_Y\Big)^2 ,
\qquad Q_s^2 = \tfrac12 q^4\sin^4\Theta\,\{\cos^2 2\phi \text{ or } \sin^2 2\phi\},
$$
with $r = |\mathbf k-\mathbf q|$. The angular integration over $\Theta,\phi$ and any change of loop variables are **not** performed in the notes.

---

*Everything from here to the end of §3.2 is Step 7 ("Rewrite in Fabrikant form"). Per README §2 it is **NOT to be used** for the build; it is transcribed for the audit's reference only.*

**R32** — MAIN 14 p.11 (Step 7, not to be used). "First we wish to exchange the Bessel functions for spherical Bessel functions ⇒ time integral $I_{J/Y}$ factor is"
$$
I_{J/Y} = \Big(\frac{2}{\pi}\Big)^{3/2}\int_{\eta_0}^{\eta}d\eta'\,(\eta')^{\frac12-b}\sqrt{k\eta'}
\begin{Bmatrix} j_b(k\eta') \\[2pt] y_b(k\eta') \end{Bmatrix}
(qrc_s^2)^{1/2}\,\eta'\,\Big(j_b(q\eta'c_s)j_b(r\eta'c_s) + \frac{2+b}{1+b}\,j_{2+b}(q\eta'c_s)j_{2+b}(r\eta'c_s)\Big)
$$
$$
= \Big(\frac{2}{\pi}\Big)^{3/2}(kqrc_s^2)^{1/2}\int_{\eta_0}^{\eta}d\eta'\,(\eta')^{2-b}
\begin{Bmatrix} j_b(k\eta') \\[2pt] y_b(k\eta') \end{Bmatrix}
\Big(j_b(q\eta'c_s)j_b(r\eta'c_s) + \frac{2+b}{1+b}\,j_{2+b}(q\eta'c_s)j_{2+b}(r\eta'c_s)\Big).
$$
(Uses $J_{n+1/2}(x) = \sqrt{2x/\pi}\,j_n(x)$, not written on the page.) Confidence: high.

**R33** — MAIN 14 p.12 (Step 7, not to be used). Definition
$$
I_{j,y} = \int_{\eta_0}^{\eta}d\eta'\,(\eta')^{2-b}
\begin{Bmatrix} j_b(k\eta') \\[2pt] y_b(k\eta') \end{Bmatrix}
\Big(j_b(q\eta'c_s)j_b(r\eta'c_s) + \frac{2+b}{1+b}\,j_{2+b}(q\eta'c_s)j_{2+b}(r\eta'c_s)\Big),
$$
"Then the time integral can be written"
$$
\frac{4}{\pi^2}\,\pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\,(c_s^2qr\eta)^{-\frac12-b}\cdot(qrk^2c_s^2)^{1/2}\,\eta^{1/2}\,\Big(y_b(k\eta)\,I_j \mp j_b(k\eta)\,I_y\Big)
= \frac{2^{4+2b}}{\pi}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)\,k\,(c_s^2qr\eta)^{-b}\,\Big(y_b(k\eta)\,I_j - j_b(k\eta)\,I_y\Big).
$$
Confidence: **medium**. (a) The exponent in $I_{j,y}$ is written "$2\bar{+}b$" — a $+$ over-written to $-$; read as $2-b$, consistent with R32. (b) The sign between the two terms in the first line is written $\mp$ (or a $-$ with a mark above); the final line has a plain $-$. (c) The final line has $\Gamma(\tfrac52+b)$ **without** the square, whereas every preceding line (and the arithmetic) has $\Gamma^2$; apparent slip. (d) The last factor was first written "$g_b(k\eta)$" and corrected to $j_b(k\eta)$ (a "j" written beneath). Not to be used in any case.

---

## 4. Checks (MAIN 14 as a recheck of MAIN 11)

The author does not write any explicit "agrees/disagrees" remarks in MAIN 14; the comparison below is the transcriber's, line by line.

| MAIN 11 result | MAIN 14 counterpart | Outcome |
|---|---|---|
| R6 (p.5) source equation in $w$ | R15 (p.1) | Identical, term for term. MAIN 14 takes it as its starting point without rederiving R1–R5. |
| R7 (p.5) $\phi = \frac{3(1+w)}{5+3w}\Phi\,S$ | R16 (p.1) | Same prefactor. MAIN 14 renames $S\to S^*$ and adds the normalisation $\Phi(q\eta)\to1$ as $q\eta\to0$. |
| R8 (p.6) $f(q,r,\eta)$ | R18 (p.3) | Same coefficients $\frac{3(1+w)}{5+3w}$, $\frac{3(1+w)(1+3w)}{(5+3w)^2}$ and $\frac{3}{2}\frac{(1+w)(1+3w)^2}{(5+3w)^2}$ (MAIN 14 writes the last as $\frac{3(1+w)(1+3w)}{(5+3w)^2}\cdot\frac{1+3w}{2}$). **Notational discrepancy, not commented on by the author:** MAIN 11 has $\eta\,\Phi'(q\eta)\Phi(r\eta)$, $\eta^2\Phi'\Phi'$; MAIN 14 has $q\eta\,\Phi'(q\eta)\Phi(r\eta)$, $qr\eta^2\Phi'\Phi'$. The two agree iff MAIN 11's prime means $d/d\eta$ and MAIN 14's means $d/d(q\eta)$. MAIN 14 is the one that is internally consistent with its explicit $\Phi'(x)$ (R26), and it is the build target. |
| R9–R10 (p.6–7) $h_s = \chi_s/a$, Green's-function solution | R18 (p.2–3) | Agree. MAIN 14 adds the defining equation for $\mathrm{Gr}_k$ (source $+\delta$) and the limits $\int_{\eta_0}^{\eta}$. |
| R11–R12 (p.7–8) Wick contraction, factor 32, symmetry arguments | R22–R23 (p.4–6) | Agree: same two Wick pairings, same $(2\pi)^6$, same factor $32 = 2\times16$, same symmetry statements ($\mathrm{Gr}_{\mathbf k}=\mathrm{Gr}_{-\mathbf k}$, $f$ depends only on $q,r$, $f(q,r)=f(r,q)$). MAIN 14 additionally: keeps $s\ne s'$ and shows $\propto\delta_{ss'}$; states $Q_{s'}(\mathbf k',\mathbf k'+\mathbf q) = Q_{s'}(\mathbf k',\mathbf q)$ and $Q_{s'}(-\mathbf k,-\mathbf q) = Q_{s'}(\mathbf k,\mathbf q)$ explicitly (MAIN 11 applied the latter silently between p.8 and p.10); writes "+ disconnected" and drops it explicitly. |
| R13 (p.9) $Q_\pm$ | R19 (p.4) | Identical: $Q_+ = \frac{q^2}{\sqrt2}\sin^2\Theta\cos2\phi$, $Q_\times = \frac{q^2}{\sqrt2}\sin^2\Theta\sin2\phi$. |
| R14 (p.10) $P_s(k)$ | R23 (p.6) | Identical formula, renamed $P_s\to P^h_{22}$, $P_S\to P_*$. |
| — | R20, R21, R24–R33 | New in MAIN 14 (explicit $\Phi$, explicit $\mathrm{Gr}_k$, $b$-form, Bessel-product form, time integral, Fabrikant form); no MAIN 11 counterpart. |

Net: the recheck **confirms** MAIN 11's final formula (R14 ≡ R23) with no discrepancy noted by the author; the only difference found by the transcriber is the implicit change in the meaning of $\Phi'$ (R8 vs R18).

---

## 5. Corrections and cross-outs

| Where | Original | Correction | Used later |
|---|---|---|---|
| MAIN 11 p.1 | In $R_{jk}$, the term $\partial_j\partial_i h_{mk}$ struck through (both occurrences) with the annotation "0 traceless" pointing at $\partial_i\partial_k h_{jm}$-type terms. | — (intermediate algebra) | Only $R_{jk}\to-\tfrac12\partial^2h_{jk}$ is used. |
| MAIN 11 p.6 | Coefficient of the mixed $\Phi\Phi'$ term written $\dfrac{3(1+w)(1+3w)\eta}{(5+3w)^2\,(1+3w)}$ | the trailing $(1+3w)$ in the denominator is struck through | Later steps (R8's $f$, and MAIN 14 R18) use $\dfrac{3(1+w)(1+3w)}{(5+3w)^2}$, i.e. the corrected form. |
| MAIN 11 p.6 | Leading "$f''\chi_s$" at the start of the collected $\chi_s$ equation struck through | replaced by $f\chi_s''$ | Intermediate. |
| MAIN 11 p.7 | $P_S(q)P_S(k-q)$ in the four-point function | a plain roman "P" is written above each $P_S$ | Read as clarifying that $P$ is the dimensionful spectrum (not script $\mathcal P$). Interpretation is the transcriber's. |
| MAIN 11 p.9 | "$\cos2\phi$" with a small mark after "cos" | — | Read as $\cos2\phi$ (confirmed MAIN 14 p.4). |
| MAIN 14 p.5 | In the second-to-last display, a factor $Q_s(\mathbf k,\mathbf q)$ between $\delta(\mathbf k+\mathbf k')$ and $P_*(q)$ struck through | the $Q_s$ factors appear inside the braces instead | Braces form used (R22). |
| MAIN 14 p.6 | $f(-\mathbf q,\,\text{[struck symbol]}\,\mathbf q-\mathbf k,\eta'')$ | struck symbol illegible (possibly a "$\mathbf k$" or "+") | $f(-\mathbf q,\mathbf q-\mathbf k,\eta'')$ used. |
| MAIN 14 p.9 | First "completed-square" line: $(q\eta c_s)(k\eta c_s)J_{\frac12+b}(q\eta c_s)J_{\frac12+b}(r\eta c_s)$ | not corrected on the page; next line has $(r\eta c_s)$ | $r$ (R28). **Author 2026-09-07: transcription error — the page reads $(r\eta c_s)$; there is no slip on the page.** |
| MAIN 14 p.10 | Bessel orders $J_{5/2}$ in the first Step-6 line | not corrected on the page; next lines have $J_{\frac52+b}$ | $\frac52+b$ (R29–R31). |
| MAIN 14 p.10 and p.11 | Exponent of 2 in $\pi\,2^{2+\cdots}$: something over-written with "$2b$" written above (original possibly "$3b$" or "$2b$" re-inked) | $2^{2+2b}$ | $2^{2+2b}$; arithmetically forced by $\frac\pi2\cdot2^{3+2b}$. Enters R30–R31 (`TARGET`) and R33. |
| MAIN 14 p.12 | $I_{j,y}$ exponent "$(\eta')^{2\bar{+}b}$" ($+$ over-written) | $2-b$ | $2-b$ (consistent with R32). Fabrikant part only. |
| MAIN 14 p.12 | "$\mp$" between $y_bI_j$ and $j_bI_y$ in the first line | plain "$-$" in the final line | $-$. Fabrikant part only. |
| MAIN 14 p.12 | "$g_b(k\eta)$" in the final line | "j" written beneath → $j_b(k\eta)$ | $j_b$. Fabrikant part only. |

---

## 6. Open questions

1. **Meaning of the prime on $\Phi$ in MAIN 11 (pp.5–6).** MAIN 11's $f$ (R8) has no $q$, $r$ factors multiplying $\eta$, MAIN 14's (R18) does. Consistent only if MAIN 11's prime is $d/d\eta$. The author never comments. The build uses MAIN 14, where $\Phi'(x) = d\Phi/dx$ is explicit (R26).
   **Closed 2026-09-07:** MAIN 11's prime is $d/d\eta$, an anomaly; the project uses $d/dx$ and, in the numerical branch, $T_k(z)$ (§0.4).
2. **$S$ / $S^*$ is never defined** in either document. From R16 it is the quantity to which $\frac{5+3w}{3(1+w)}\phi_{\mathbf q}$ tends as $q\eta\to0$. Whether $S^*$ equals $\zeta$, $\mathcal R$, or $-\zeta$ etc. is not stated; nor is the relation of $P_*(q)$ to a primordial $\mathcal P_\zeta$. (MAIN 14 p.10 only says Domènech uses "$\Phi^*$ rather than $S^*$".) Presumably defined in `MAIN` 10/12 (Groups 1, 3).
   **Closed 2026-09-07:** $S^* \equiv \zeta^*$, the primordial curvature perturbation; $P_*(q) = P_\zeta(q)$; sign
   convention immaterial. See §0.3.
3. **$c_s$ is never defined** in either document; it enters only through $\Phi(x)$ (R20, R26–R31). Presumably from `MAIN` 12 (Group 1).
4. **$\eta_0$ is never defined**; all time integrals are $\int_{\eta_0}^{\eta}$.
5. **Angular reduction not done.** Neither document reduces $\int d^3q/(2\pi)^3\,Q_s^2\,\cdots$ to scalar integrals; the final results (R14, R23, `TARGET` R31) are left with $\int d^3q$ and $Q_s^2 = \frac12q^4\sin^4\Theta\{\cos^2,\sin^2\}2\phi$. Any $(q,r)$, $(u,v)$ or $(\Theta,\phi)$ reduction in the code has no counterpart in these notes and must be audited against the build target R23+R31 as written.
6. **Per-polarisation only.** $P^h_{22}$ is "for any $s$"; no sum over polarisations and no $\Omega_{\rm GW}$ or $h_{ij}$-total spectrum appears. Whether the code's deliverable is $P^h_{22}$, $2P^h_{22}$, or a dimensionless $\mathcal P_h$ is not fixed by these notes.
7. **MAIN 11 p.2, unprojected Einstein equation:** the power of $a$ in $\frac{4M_P^2}{a^{?}(\rho_0+p_0)}$ is written as something like "$a^{2+}$"; presumably $a^4$ (since the projected line that follows, R2, has $a^2$ after multiplication by $2a^2$). Does not affect any result.
8. **MAIN 14 p.4, R22:** second argument of the second $f$ read as $u$ ($\mathbf u = \mathbf k'-\mathbf t$); could be another letter. Not used again.
9. **MAIN 14 p.10, R29:** the Green's-function factor runs off the right margin ("$J_{b+\frac12}(k\eta$" cut); completed from R21.
10. **MAIN 14 p.12, R33 (Fabrikant part, not to be used):** final line has $\Gamma(\tfrac52+b)$ where all preceding lines have $\Gamma(\tfrac52+b)^2$ — apparent slip; and the $\mp$ vs $-$ sign. Recorded for completeness only.
11. **Statements about $h_{ij}$ normalisation relative to the literature.** MAIN 11 p.4: "$h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$" (Kohri & Terada); MAIN 11 p.10: Adshead et al.'s "$h_{ij}$ is half the one used here". Transcribed as written; the transcriber has not checked whether these two statements are mutually consistent given the papers' respective conventions.
12. **$\delta_{ss'}$ step (MAIN 14 p.6–7)** is stated to hold "if $f$ is independent of $\phi$" (the azimuth); the author asserts this condition rather than proving it. $f$ depends only on $q$, $r=|\mathbf k-\mathbf q|$, which are $\phi$-independent, so the condition is met, but this is the transcriber's remark.
