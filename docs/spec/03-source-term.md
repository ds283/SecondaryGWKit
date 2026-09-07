# Spec 03 — Source term for the induced tensor equation

Transcribed 2026-09-07 from the handwritten notes listed below (spec-transcription campaign,
Group 3). This file records what is on the pages; it does not correct the physics and was
written without reading the code.

Sign-off: **Tier 1.1 (Green's-function normalisation) Tier 1.2 (numerical prefactor chain), Tier 1.3 (form of the one-loop integral), Tier 1.4 (seed = $\zeta^*$), Tier 2.9, 2.10 and the Tier 3 notation items signed off by the author 2026-09-07; see §0. **All review-queue items for this file are closed.**
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

### 0.2 Tier 1.2 — numerical prefactor chain of `NUM` 03 pp. 7–10: **signed off 2026-09-07**

**Decision (author).** The typed red annotations of 10 July 2025 are right and the blue-ink chain is a
slip. Taking the prefactor $36\big(\tfrac{1+w^*}{5+3w^*}\big)^2$ of $h_s$ (R26–R28) as correct, the
coefficients read:

| Page (result) | Blue ink, as written | Red annotation | **Correct** | Step |
|---|---|---|---|---|
| p. 7 (R30) | 1024 | 1296 | **1296** | $36^2$ from the Wick contraction of two factors of $h_s$ |
| p. 8 (R30, cont.) | 1024 | — | **1296** | carried |
| p. 9 (R32) | 2048 | 2592 | **2592** | $\times 2$: the two Wick terms collapse by symmetry (R31) |
| p. 9 (R34, both lines) | $1024\pi$ | "1292" (first line only) | **$1296\pi$** | $\times\tfrac12$ from $Q_\pm^2 = \tfrac12 q^4_{\rm phys}\sin^4\theta$; $\times\pi$ from $\int_0^{2\pi}d\varphi\cos^2 2\varphi$ |
| p. 10 (R35) | $512\pi^2$ | "646" | **$648\pi^2$** | $\times\tfrac{\pi}{2}$ from $4\pi^4/(2\pi)^3$ on converting $P^*$ to $\mathcal P^*/q^3$ |

"1292" and "646" are typos for 1296 and 648. **The final formula R35 is to be read with
$648\pi^2\big(\tfrac{1+w^*}{5+3w^*}\big)^4$.** Open question Q2 is closed.

**Origin of 36 versus 32.** `MAIN` 11/14 and `NUM` 03 start from the same field equation with the
same leading factor 4. They differ only in where the super-horizon constant
$c_* \equiv 3(1+w^*)/(5+3w^*)$ sits. `MAIN` 11 R7 and `MAIN` 14 R16 substitute
$\phi = c\,\Phi\,S^*$ and fold $c^2$ into $f$, so the $h_s$ prefactor stays 4 and $\langle hh\rangle$
carries $2\times4^2 = 32$ with $c^4$ hidden inside $f^2$. `NUM` 03 R22 pulls $c_*^2$ outside, so the
$h_s$ prefactor is $4c_*^2 = 36\big(\tfrac{1+w^*}{5+3w^*}\big)^2$ (the 36 is $4\times3^2$, the 3 being
the numerator of $c_*$) and $\langle hh\rangle$ carries $2\times(4c_*^2)^2 = 2592\big(\tfrac{1+w^*}{5+3w^*}\big)^4$.
The two agree exactly: $32\times 81 = 2592$, i.e. $f_{14} = c^2 f_{03}$ (`cross-spec-check.md` A5).
The 1024 of p. 7 is $32^2$: the `MAIN` 11 coefficient was copied and squared without noticing that in
`NUM` 03 the $c^2$ is no longer inside $f$. (Consistent with the struck leading "4" recorded at R22.)

**Two different $w$'s — `NUM` 03 is correct to separate them.**
- $w_0$ inside $f$ (the $2/(3(1+w_0))$) comes from $2M_P^2\mathcal H^2/(a^2(\rho_0+p_0))$ in the
  Einstein equation and is the background at the **source** time: $w_0 = w(z')$. The coefficient
  $(5+3w)/(3(1+w))$ on $\phi\phi$ in `MAIN` 11 R6 / `MAIN` 14 R15 is the *same* $w$; it is what
  appears when the complete square of R5 is expanded, not a second ingredient.
- $w^*$ in $c_*$ (R20, $\phi_{\mathbf k} = c_*\,T_k(z)\,\zeta^*_{\mathbf k}$) is the adiabatic
  super-horizon relation at the time the initial condition is set. It requires $w$ to be constant
  at $z_{\rm init}$ for long enough that $\phi$ has settled to its constant value; with running $w$
  it has no reason to equal $w(z')$. **Requirement for the build:** $z_{\rm init}$ must sit deep
  inside an epoch of constant $w^*$. For radiation, $w^* = \tfrac13$, $c_* = \tfrac23$
  (the standard $\phi = \tfrac23\zeta$, sign convention-dependent and irrelevant since only $P^*$
  enters), $4c_*^2 = \tfrac{16}{9}$.
- `MAIN` 11 p. 6 and `MAIN` 14 p. 2 both say "take $\mathcal H$ to correspond to an epoch of fixed
  $w$", so their single $w$ is a restriction, not an error; it becomes wrong only when the formula
  is carried to a time-varying background, which is the case the numerical branch is built for.
- The normalisation $T_k \to 1$ at $z_{\rm init}$ is a normalisation of the transfer function only
  and carries **no** information about which seed is intended; the $\zeta^*\to\phi^*$ translation
  is the separate constant $c_*$ applied outside. (Identity of the seed: Tier 1.4, still open.)

**Code status.** `ComputeTargets/QuadSource.py` (`source_function`) is the $f$ of R22 exactly: it is
written in the expanded form $\tfrac{5+3w}{3(1+w)}T_qT_r + \tfrac{2}{3(1+w)}[-(1+z)(T_qT_r'+T_rT_q') + (1+z)^2T_q'T_r']$,
which reduces algebraically to $T_qT_r + \tfrac{2}{3(1+w)}(T-(1+z)T')_q(T-(1+z)T')_r$, and it
evaluates $w$ at the source sample redshift (`wBackground(z)`), i.e. $w_0 = w(z')$.
`ComputeTargets/TkNumericIntegration.py` integrates $T_k$ from $T_k = 1$, $dT_k/dz = 0$ at
$z_{\rm init}$. `ComputeTargets/OneLoopIntegral.py` `compute()` is a **stub**: the $648\pi^2$
prefactor, the $\big(\tfrac{1+w^*}{5+3w^*}\big)^4$ factor and the $\theta$ integral are not yet
implemented anywhere. For this item the corrected chain is therefore a **build specification**, not an
audit finding. Everything that does exist (source $f$, $T_k$ normalisation, unit-jump $\bar G_k$)
follows `NUM` 03's conventions.

### 0.3 Tier 1.3 — form of the one-loop integral: **signed off 2026-09-07**

**Decision (author).** There are not two forms. `MAIN` 14 R23 (spec 05) is the 3-D loop measure
before anything has been done to it; to evaluate it one must split $d^3q = q^2dq\,\sin\theta\,d\theta\,d\varphi$,
insert the spin-2 projector factors $Q_\pm^2 = \tfrac12q^4\sin^4\theta\{\cos^2,\sin^2\}2\varphi$ and do the
$\varphi$ integral, and the result *is* `NUM` 03 R35. Nothing else happens between them. **The build
form is `NUM` 03 R35 with the Tier 1.2 coefficient and the measure completed:**

$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,\delta_{ss'}\,P^h_{22}(k),\qquad
P^h_{22}(k) = 648\pi^2\left(\frac{1+w^*}{5+3w^*}\right)^4
\int_0^\infty\frac{dq}{q}\int_0^\pi d\theta\,\sin^5\theta\;\mathcal P^*(q)\,\frac{\mathcal P^*(r)}{r^3}
\left\{\int_z^{z_{\rm init}}dz'\,\bar G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{q_{\rm phys}^2}{H(z')^2}\,f(z'\,|\,\mathbf q,\mathbf k-\mathbf q)\right\}^2,
$$
$$
r = |\mathbf k-\mathbf q| = \sqrt{k^2+q^2-2kq\cos\theta},\qquad q_{\rm phys} = q/a_0,\qquad
f(z'\,|\,\mathbf q,\mathbf k-\mathbf q) = T_qT_r + \frac{2}{3(1+w(z'))}\Big(T-(1+z')\frac{dT}{dz'}\Big)_q\Big(T-(1+z')\frac{dT}{dz'}\Big)_r .
$$

with $\theta$ the angle between $\mathbf q$ and $\mathbf k$, $\bar G_k$ the unit-jump Green's function
of §0.1, $w^* = w(z_{\rm init})$ and $w(z')$ inside $f$ as in §0.2, and $T_k \to 1$ at $z_{\rm init}$.
The $d\theta$, the limits and $r(\theta)$ were not written on p. 10; they are supplied here by the
reduction. Any change of variables (e.g. $(u,v) = (q/k, r/k)$ or $(q,r)$) is an implementation choice
and does not alter the spec.

**Check** *[author-confirmed]*: carrying the same steps out on spec 05 R23 gives
$P^h_{22} = 8\pi^2\int q^3dq\int d\theta\,\sin^5\theta\,\mathcal P^*(q)\mathcal P^*(r)r^{-3}\,(\text{TARGET})^2$
with TARGET = spec 05 R31. Inserting $\big(\tfrac{1+w^*}{5+3w^*}\big)^4 = c_*^4/81$, $648/81 = 8$, and
the §0.1 relation between the redshift brace and TARGET (the $q^2_{\rm phys}$ in the brace supplies the
$q^4$), the two coincide exactly for $w^* = w$. The match requires the corrected 648, so it is an
independent confirmation of §0.2. For a running background only the `NUM` 03 form applies.

**Deliverable.** The stored quantity is the **per-polarisation** spectrum $P^h_{22}(k)$, labelled by
$s$, exactly as both documents define it ("for any $s$", `MAIN` 14 p. 6). The observable
$\Omega_{\rm GW}$ is built in a **separate layer** that consumes the per-polarisation spectra; for
the present calculation the two polarisations carry equal power and the total tensor spectrum is the
sum over $s$, but the layers are to be kept decoupled. Reason: cosmological-collider-type models can
give unequal power in the two polarisations; the present calculation does not handle that case, and
if it is needed later a new compute layer can be shipped for $P^h_{22,s}$ while the layer that builds
the observable is kept unchanged.

**Method (build question, not spec).** How the $q$ and $\theta$ integrals are actually performed is
open. Any scheme has to handle the Kohri–Terada resonance in the inner time integral, possibly by a
stationary-phase treatment. `ComputeTargets/OneLoopIntegral.py` is currently a stub.

### 0.4 Tier 1.4 — identity and normalisation of the primordial seed: **signed off 2026-09-07**

**Decision (author).** The seed is the primordial curvature perturbation $\zeta^*$ on both
`MAIN` 14 p. 1 and `NUM` 03 p. 4. The "5-like glyph" of spec 03 and the "$S^*$" of spec 05 are the
same handwritten $\zeta^*$; the transcribers had trouble with the handwriting. The linear relation is
$$
\phi_{\mathbf k}(z) = \frac{3(1+w^*)}{5+3w^*}\,T_k(z)\,\zeta^*_{\mathbf k},\qquad T_k \to 1 \text{ at } z_{\rm init},
$$
(`NUM` 03 R20), correct at linear order for adiabatic perturbations with $w = w^*$ constant at
$z_{\rm init}$ (§0.2). Consequently **$P^*(q)$ is $P_\zeta(q)$ and $\mathcal P^*(q)$ is the dimensionless
$\mathcal P_\zeta(q)$**, and the $\big(\tfrac{1+w^*}{5+3w^*}\big)^4$ in R35 is the translation from
$\zeta^*$ to the potential, applied outside the transfer functions.

**Why $\zeta$.** The initial spectrum is to be supplied in terms of $\zeta$ so that output from
`PyTransport` or `CppTransport`, which compute the $\zeta$ two-point function, can be fed in
directly. Spec 01's $\phi^*$ is not in conflict: it is the early-time value of the potential itself,
$\phi^* = \tfrac{3(1+w^*)}{5+3w^*}\zeta^*$, and $T_k = \phi_k/\phi^*_k$.

**Sign.** The sign convention relating $\zeta$ to the potential is not fixed by the notes and does
not need to be: only $P_\zeta$ enters $P^h_{22}$. At linear order on super-horizon scales the
adiabatic $\zeta$ and $\mathcal R$ coincide up to convention, so a $\mathcal R$ spectrum may be fed
in equally.

**All four Tier 1 items are now signed off for this file.**

### 0.5 Tier 2 and Tier 3 items — **signed off 2026-09-07**

- **2.9 (p. 7, R28).** The denominator under the ink blot is $a_0^2H^2(z')$; both transcribers' reading confirmed.
- **2.10 (p. 9, R34).** The struck factor outside the braces ($q^4_{\rm phys}/2$, possibly $/4$) is crossed
  out and unused; the remaining expression is correct.
- **Arguments of $f$ (Q3).** $f(z'\,|\,\mathbf q,\mathbf k-\mathbf q)$ throughout; the $(\mathbf k,\mathbf k-\mathbf q)$
  form on pp. 5, 7 is a notational slip. `MAIN` 14's form is the correct one.
- **$z$ vs $z'$ in $f$ (Q4).** $z'$, the source time integrated over, on pp. 9–10.
- **$w_0$, $w^*$, $c_s^2$.** $w_0 = p_0/\rho_0$ is the background equation of state (code:
  `wBackground(z)`, $\Lambda$ included), evaluated at the source time inside $f$; $w^*$ is $w_0$ at the
  initial time; elsewhere $c_s^2 = w(z)$ of the perturbed fluid is intended (code: `wPerturbations(z)`,
  $\Lambda$ unperturbed). See §0.2 and spec 01's Tier 3 block.
- **Overloaded symbols** (confirmed as traps, not errors): $\epsilon$ = slow-roll parameter
  $-\dot H/H^2$ and, in jump conditions, an infinitesimal; $\eta$ = conformal time and, in `NUM` 05 p.2
  only, the second slow-roll parameter $d\ln\epsilon/dN$; $\omega_{\rm eff}$ = the Green's-function
  frequency (spec 02 R22) and the transfer-function frequency (spec 01 R27), different functions;
  $Q_s$ written with comoving $q$ (`MAIN` 14) or with $q_{\rm phys}=q/a_0$ (`NUM` 03 p.9), differing
  by $a_0^{-2}$ (absorbed, spec 02 §0 item 1).

All review-queue items for this file are closed.

---

## 1. Source documents

| Ref | File | Pages | Date on first page |
|---|---|---|---|
| `MAIN` 10 | `Green's function formula for P22/10 - 2022:07:18 - contribution from energy-momentum tensor.pdf` | 4 | 18 July 2022 (heading "CALCULATION 10. Extract contribution to the source term from the energy-momentum tensor") |
| `NUM` 03 | `Numerical implementation/03 - 2024-09-20 - SIGW source term expressed in redshift.pdf` | **10** (see note) | 20 Sep 2024 (heading "SIGW source term expressed in redshift") |

**Page-count note for `NUM` 03.** The campaign README lists this document as 46 pages. The PDF's
live page tree has `/Count 10`, the Read tool renders 10 pages and rejects any page beyond 10, and
the file contains 16 `%%EOF` markers (incremental saves, consistent with the typed red annotations
dated "DS 10 July 2025" on pp. 7, 9, 10). A naive count of `/Type /Page` objects in the raw bytes
gives 46 because each incremental save re-appends page objects. The document is 10 pages long and
all 10 were read.

`NUM` 03 also carries typed red annotations (boxes and text) added later by the author, dated
10 July 2025; these are recorded in §5 and are distinguished from the original blue ink below.

`MAIN` 10 has one typed red annotation on p. 2 (undated); recorded in §5.

---

## 2. Conventions in force

### 2.1 `MAIN` 10

- **Time variable.** Conformal time $\eta$ throughout the perturbation algebra (p. 1: $v^i = dx^i/d\eta$;
  pp. 3–4: $\phi'$, $\mathcal{H}'$). Cosmic time $t$ appears only in the background check on pp. 3–4
  (dots: $\dot H$, $\ddot a$). Proper time along the fluid flow is $\tau$ (p. 1).
- **Prime.** $d/d\eta$ (pp. 3–4). Dot $= d/dt$ (pp. 3–4). Not stated explicitly; inferred from
  $\mathcal{H}' \to \dot H$ conversion on p. 3.
- **Hubble rates.** $\mathcal{H}$ (script) is the conformal Hubble rate, $H$ the cosmic-time rate; the
  background equation is written in both forms on p. 3 ($\tfrac{1}{a^2}(-2\mathcal{H}'-\mathcal{H}^2)$ and
  $-(2\dot H+3H^2)$). $\epsilon$ is not used.
- **Scale factor.** No $a_0$ appears; $a$ is generic. Nothing says $a_0=1$.
- **Metric ansatz (inferred, not written down).** From $u^au_a=-1$ on p. 1, the metric used is
  $g_{\eta\eta} = -a^2 e^{2\phi}$, $g_{ij} = a^2 e^{-2\psi}\gamma_{ij}$ (so $\phi$ is the lapse
  potential, $\psi$ the curvature potential, both in exponential form). Anisotropic stress is ignored
  (p. 1). Step 2 concludes $\psi=\phi$ at first order (p. 3) and Step 3 uses it (p. 4).
- **Einstein equation and Planck mass.** $G^a{}_b = M_P^{-2}\,T^a{}_b$ with $3H^2M_P^2=\rho_0$ (reduced
  Planck mass; inferred from p. 3 "$\tfrac{1}{a^2}(-2\mathcal{H}'-\mathcal{H}^2)=p_0M_P^{-2}$" and p. 4
  "$H^2M_P^2=\rho_0/3$").
- **Background quantities.** $\rho_0$, $p_0$; perturbations $\delta\rho$, $\delta p$, velocity $v^i$;
  orders counted as $O(\delta,v)$ and $O(\delta,v)^2$.
- **Green's function, $w$, Fourier, transfer function, loop momenta.** Not used in this document.

### 2.2 `NUM` 03

- **Time variable.** Starts in conformal time $\eta$ (p. 1, quoting `MAIN` 11 p. 3), converts to
  redshift $z$ on pp. 1–4. Cosmic time $t$ is used transiently on pp. 2–3 (dots, $\dot H$) to evaluate
  $a''/a$ and $d(aH)/dt$. From p. 4 onwards everything is in $z$. Neither $\tau$ nor $\log(1+z)$ is used.
- **Prime.** On pp. 1–3 a prime on a field or on $a$ means $d/d\eta$ ($h_s'$, $\chi_s'$, $a'$, $a''$,
  $\phi'$). From p. 2 onwards $z$-derivatives are always written out as $d/dz$. **From p. 5 onwards a
  prime on $z$ (i.e. $z'$) is the source-time integration variable, not a derivative.** The
  conversion rule is stated on p. 1: $\dfrac{d}{d\eta} = -(1+z)\,aH\,\dfrac{d}{dz}$.
- **Hubble rates.** $\mathcal{H}$ (script) $=a'/a$ on pp. 1–2; $H$ is the cosmic-time rate from p. 2
  onwards, with $\mathcal{H}=aH$ (used, not stated). $\epsilon$ is introduced on p. 3 through
  $a''/a = 2a^2H^2 + a^2\dot H = (aH)^2(2-\epsilon)$, i.e. **$\epsilon = -\dot H/H^2$** (inferred; no
  explicit definition is written). On p. 6 the symbol $\epsilon$ is reused as an infinitesimal
  interval $z'\pm\epsilon$ (unrelated).
- **Scale-factor normalisation.** $a_0$ appears explicitly (pp. 4, 5, 7) with $a = a_0/(1+z)$ used
  implicitly ($a_0/a(z') = 1+z'$ on p. 5). **$a_0$ is not set to 1.** The notation $k_{\rm phys}$,
  $q_{\rm phys}$ (pp. 4, 5, 7, 9, 10) means the comoving wavenumber divided by $a_0$:
  $k^2_{\rm phys}/H^2 \equiv k^2/(a_0^2 H^2)$ — inferred from p. 4, where
  $\tfrac{k^2}{a^2H^2}\tfrac{1}{(1+z)^2}$ is rewritten as $\tfrac{k^2}{H^2a_0^2}\tfrac{a_0^2}{a^2(1+z)^2}$
  and then as $k^2_{\rm phys}/H^2$, and from p. 7, $Q_s/(a_0^2H^2) \propto q^2_{\rm phys}/H^2$.
- **Green's function** (pp. 5–6, stated). For $\chi_s = a\,h_s$: $G^\chi_k(z,z')$ satisfies the
  homogeneous operator of R30 with source **$+\delta(z-z')$**; first argument $z$ is the response
  (observation) redshift, second $z'$ the source redshift. Then $\bar G_k \equiv -G_k$ is introduced
  (p. 5) so that the source integral can be written with limits $\int_z^{z_{\rm init}}dz'$ (p. 6).
  Causal boundary conditions (p. 6): $\bar G_k(z,z') = d\bar G_k/dz = 0$ for $z>z'$ (i.e. before the
  source), and $d\bar G_k/dz\big|_{z=z'} = +1$ approached from $z<z'$. No relation to a "literature"
  Green's function is stated. Note $G_k$ depends only on $k=|\mathbf{k}|$ (p. 8).
- **Equation of state.** $w_0 \equiv p_0/\rho_0$, the background equation-of-state parameter at the
  time appearing in the equation (p. 2 writes $\rho_0(1+p_0/\rho_0)$, p. 3 writes $1+w_0$; inferred).
  $w^*$ is "the fixed value of $w$ at the initial time" (p. 4, stated). The shorthands $b$ and
  $c_s^2$ are **not** used. Whether $w_0$ inside the $z'$-integral means $w(z')$ is not stated (see §6).
- **Fourier / power-spectrum conventions.**
  - Convolution measure (p. 1, quoted): $\int \dfrac{d^3q\,d^3r}{(2\pi)^3}\,\delta(\mathbf{k}-\mathbf{q}-\mathbf{r})$,
    reduced to $\int \dfrac{d^3q}{(2\pi)^3}$ with $\mathbf{r}=\mathbf{k}-\mathbf{q}$ (p. 2).
  - Two-point function (inferred from the Wick contraction on pp. 7–8):
    $\langle \zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{t}}\rangle = (2\pi)^3\delta(\mathbf{q}+\mathbf{t})\,P^*(q)$
    with $P^*$ dimensionful.
  - Dimensionless spectrum (inferred from p. 9, where $P^*(q)P^*(r)$ becomes
    $4\pi^4\,\mathcal{P}^*(q)\mathcal{P}^*(r)/(q^3r^3)$): $\mathcal{P}(q) = \dfrac{q^3}{2\pi^2}P(q)$.
  - Tensor two-point function written as
    $\langle h_s(\mathbf{k})h_{s'}(\mathbf{k}')\rangle = (2\pi)^3\delta(\mathbf{k}+\mathbf{k}')\,\delta_{ss'}\,(\dots)$
    (pp. 8–10). No dimensionless tensor spectrum is defined in this document.
  - Polarisation tensors: $e_s^{\ell m}(\mathbf{k})$, $s=\pm$, transverse — "$e$ annihilates $\mathbf{k}$"
    (p. 2). Normalisation inferred from p. 9, $Q_\pm = \tfrac{1}{\sqrt2}q^2_{\rm phys}\sin^2\theta\,\{\cos 2\varphi,\ \sin 2\varphi\}$:
    this corresponds to $e_+ = \tfrac{1}{\sqrt2}(\hat{x}\hat{x}-\hat{y}\hat{y})$,
    $e_\times = \tfrac{1}{\sqrt2}(\hat{x}\hat{y}+\hat{y}\hat{x})$ with $\mathbf{k}\parallel\hat z$, i.e.
    $e_s^{\ell m}e^{s'}_{\ell m} = \delta_{ss'}$. Not written explicitly.
  - The relation between $h_s$ and $h_{ij}$ is not stated in this document (it comes from `MAIN` 11).
- **Transfer function** (p. 4, stated): $\phi_{\mathbf{k}} = \dfrac{3(1+w^*)}{5+3w^*}\,T_k(z)\,\zeta^*_{\mathbf{k}}$,
  so $T_k$ is the transfer function of the **Newtonian potential $\phi$** (the lapse potential of
  `MAIN` 10's metric), and $\zeta^*$ is the primordial curvature perturbation at the initial time. The
  early-time normalisation of $T_k$ is not stated here; the prefactor is the standard constant-$w$
  super-horizon relation, which implies $T_k\to 1$ at the initial time (inferred).
- **Momentum labels.** External $\mathbf{k}$; loop momentum $\mathbf{q}$; $\mathbf{r}=\mathbf{k}-\mathbf{q}$
  (pp. 1–2), later $r\equiv|\mathbf{k}-\mathbf{q}|$ (pp. 9–10); a second loop momentum $\mathbf{t}$ for
  the four-point contraction (p. 7). Angular reduction (p. 9): $\theta$ is the angle between
  $\mathbf{q}$ and $\mathbf{k}$, $\varphi$ the azimuth about $\mathbf{k}$; the $\varphi$ integral is
  done analytically, the $\theta$ integral is left implicit (see §6).
- **Tensor variable.** $h_s$ is the polarisation-$s$ amplitude of the tensor mode in conformal time
  (p. 1); the rescaled variable $\chi_s \equiv a\,h_s$ is introduced on p. 1 and used through p. 6.

---

## 3. Results

Confidence flags refer to the *reading* of the handwriting, not to the physics.

### 3.1 `MAIN` 10 — contribution from the energy-momentum tensor

**R1** (`MAIN` 10 p. 1). Perfect-fluid energy-momentum tensor and 4-velocity.
$$T^a{}_b = (\rho+p)\,u^a u_b + p\,\delta^a{}_b,\qquad
u^a = \frac{dx^a}{d\tau} = \frac{d\eta}{d\tau}\,(1, v^i),\qquad v^i = \frac{dx^i}{d\eta}.$$
Anisotropic stress ignored; $v^i$ is the 3-velocity ("if $\eta$ is conformal time, this is the local
physical velocity, neglecting gravitational effects"); $d\eta/d\tau$ is "the analogue of the Lorentz
factor". Confidence: high.

**R2** (`MAIN` 10 pp. 1–2). Normalisation of the 4-velocity.
$$\frac{d\eta}{d\tau} = \frac{1}{a}\,e^{-\phi}\,\gamma_{SR},\qquad
\gamma_{SR} = \frac{1}{\sqrt{1-\gamma_{ij}v_p^i v_p^j}} = 1+O(v^2),\qquad
v_p^i = e^{-(\phi+\psi)}\,v^i \ \text{(true physical velocity)}.$$
Obtained from $(d\eta/d\tau)^2 a^2 e^{2\phi}\{1 - e^{-2(\phi+\psi)}\gamma_{ij}v^iv^j\} = 1$.
Confidence: high.

**R3** (`MAIN` 10 p. 2). Component (A), to $O(\delta,v)$.
$$T^0{}_0 = (\rho+p)u^0u_0 + p = -(\rho+p)\gamma_{SR}^2 + p = -\rho_0 + \delta\rho \quad\text{(as written)},$$
using $u_0 = g_{\eta\eta}u^0 = -a e^{\phi}\gamma_{SR}$. The final expression is red-boxed with the
typed annotation "sign flip on delta \rho; should be negative", i.e. the corrected reading is
$T^0{}_0 = -\rho_0 - \delta\rho$. Not used later in this document. Confidence: high (both the original
and the annotation are clear).

**R4** (`MAIN` 10 p. 2). Component (B), to $O(\delta,v)$.
$$T^0{}_i = (\rho+p)u^0u_i = (\rho+p)\,e^{-2\phi-2\psi}\,\gamma_{SR}^2\,\gamma_{ij}v^j = (\rho_0+p_0)\,v_i .$$
Confidence: high.

**R5** (`MAIN` 10 pp. 2–3). Component (C).
$$T^i{}_j = (\rho+p)\,e^{-2(\phi+\psi)}\,v^i\gamma_{jk}v^k + p\,\delta^i{}_j = (p_0 + p)\,\delta^i{}_j
\quad\text{up to } O(\delta,v)^2 .$$
Confidence: **medium** — the last bracket reads "$(p_0+p)$" and is almost certainly intended as
$(p_0+\delta p)$; the glyph is a plain $p$ with no visible $\delta$.

**R6** (`MAIN` 10 p. 3). $G^i{}_j$ Einstein tensor to first order (Step 2), after collecting terms:
$$G^i{}_j \to \frac{1}{a^2}\delta^i{}_j\left(-2\mathcal{H}'-\mathcal{H}^2\right)
+ \frac{1}{a^2}\delta^i{}_j\Big\{2\psi'' + 2\mathcal{H}\phi' + 4\mathcal{H}\psi' + 2\phi\,(2\mathcal{H}'+\mathcal{H}^2)
+ \nabla^2\phi - \nabla^2\psi\Big\}
+ \frac{1}{a^2}\Big\{\nabla^i\nabla_j\psi - \nabla^i\nabla_j\phi\Big\} .$$
(The first form on the page has the overall factor $(1-2\phi)$ multiplying the background bracket
and the $\phi',\psi',\psi''$ terms; the expanded form above is what the page derives from it.)
Confidence: high.

**R7** (`MAIN` 10 p. 3). Traceless condition. "At $O(\delta,v)$, $\nabla^i\nabla_j(\psi-\phi)=0$ since
$T^i{}_j = p\,\delta^i{}_j$." Hence $\psi=\phi$, used in Step 3. Confidence: high.

**R8** (`MAIN` 10 pp. 3–4). Background equation from the trace, three equivalent forms:
$$\frac{1}{a^2}\left(-2\mathcal{H}'-\mathcal{H}^2\right) = \frac{p_0}{M_P^2}
\;\Longleftrightarrow\; 2\dot H + 3H^2 = -\frac{p_0}{M_P^2}
\;\Longleftrightarrow\; 2M_P^2\frac{\ddot a}{a} = -p_0 - H^2M_P^2 = -\tfrac13(\rho+3p),$$
"which is the expected Friedmann acceleration equation" (uses $H^2M_P^2=\rho_0/3$). Confidence: high.

**R9** (`MAIN` 10 p. 4). $G^0{}_i$ equation (Step 3), to $O(\delta,v)$, with $\psi=\phi$:
$$G^0{}_i \to -\frac{2}{a^2}\partial_i\psi' + \frac{2}{a^2}(-\mathcal{H})\partial_i\phi
= -\frac{2}{a^2}\partial_i\left(\phi'+\mathcal{H}\phi\right) = M_P^{-2}(\rho_0+p_0)\,v_i .$$
Confidence: high.

**R10** (`MAIN` 10 p. 4). Fluid velocity in terms of the potential:
$$v_i = -\frac{2M_P^2}{a^2}\,\frac{1}{\rho_0+p_0}\,\partial_i\left(\phi'+\mathcal{H}\phi\right).$$
Confidence: high (a superscript on $v_i$ is struck through; see §5).

**R11** (`MAIN` 10 p. 4). **Final result of the document** — quadratic velocity contribution to the
spatial stress:
$$T^i{}_j\Big|_{O(\delta,v)^2} = (\rho_0+p_0)\,\frac{4M_P^4}{a^4}\,\frac{1}{(\rho_0+p_0)^2}\,
\partial^i(\phi'+\mathcal{H}\phi)\,\partial_j(\phi'+\mathcal{H}\phi)
= \frac{4M_P^4}{a^4}\,\frac{1}{\rho_0+p_0}\,\partial^i(\phi'+\mathcal{H}\phi)\,\partial_j(\phi'+\mathcal{H}\phi).$$
Confidence: high for the structure; **medium** for the power of $a$ in the last denominator, which
is over-written (reads "$a^{4}$" with a stray mark; $a^4$ is consistent with $(a^2)^2$ from R10).

### 3.2 `NUM` 03 — SIGW source term expressed in redshift

**R12** (`NUM` 03 p. 1). Starting point, quoted "from (11), page 3" (i.e. `MAIN` 11 p. 3):
$$h_s'' + 2\mathcal{H}h_s' + k^2h_s = -\int\frac{d^3q\,d^3r}{(2\pi)^3}\,\delta(\mathbf{k}-\mathbf{q}-\mathbf{r})\,
e_s^{\ell m}(\mathbf{k})\left\{4\,q_\ell r_m\,\phi_{\mathbf{q}}\phi_{\mathbf{r}}
+ \frac{8M_P^2}{\rho_0+p_0}\,\frac{q_\ell r_m}{a^2}\,\big(\phi'_{\mathbf{q}}+\mathcal{H}\phi_{\mathbf{q}}\big)\big(\phi'_{\mathbf{r}}+\mathcal{H}\phi_{\mathbf{r}}\big)\right\}.$$
Confidence: high.

**R13** (`NUM` 03 p. 1). Auxiliary relations used throughout:
$$3H^2M_P^2 = \rho_0,\qquad \frac{d}{d\eta} = -(1+z)\,aH\,\frac{d}{dz}.$$
Confidence: high.

**R14** (`NUM` 03 pp. 1–2). Rescaling. With $h_s = \chi_s/a$,
$$\chi_s'' + \left(k^2 - \frac{a''}{a}\right)\chi_s = a\times(\text{source term}),$$
i.e. the $2\mathcal{H}h_s'$ friction term is removed (the $\{2\mathcal{H}^2 - a''/a - 2\mathcal{H}^2\}$
bracket on p. 2 collapses to $-a''/a$). Confidence: high.

**R15** (`NUM` 03 p. 2). Momentum reduction. Integrating out $\mathbf{r}$ sets $\mathbf{r}=\mathbf{k}-\mathbf{q}$ and
$$e_s^{\ell m}(\mathbf{k})\,q_\ell r_m \;\to\; e_s^{\ell m}(\mathbf{k})\,q_\ell k_m - e_s^{\ell m}(\mathbf{k})\,q_\ell q_m
= -\,e_s^{\ell m}(\mathbf{k})\,q_\ell q_m \quad\text{"since $e$ annihilates $\mathbf{k}$".}$$
The overall minus sign cancels the minus on the right of R12, so the source becomes $+4a\int\dots$.
Confidence: high.

**R16** (`NUM` 03 pp. 2–3). Conversion of the velocity factor (used inside the source):
$$\phi' + \mathcal{H}\phi \;\to\; -(1+z)\,aH\frac{d\phi}{dz} + aH\,\phi = aH\left[\phi - (1+z)\frac{d\phi}{dz}\right].$$
Confidence: high.

**R17** (`NUM` 03 p. 3). Second derivative of the scale factor in terms of $\epsilon$:
$$\frac{a''}{a} = \frac{d}{dt}\left(a^2H\right) = 2a^2H^2 + a^2\dot H = (aH)^2\,(2-\epsilon).$$
(This is the only place $\epsilon$ is defined; it implies $\epsilon=-\dot H/H^2$.) Confidence: high.

**R18** (`NUM` 03 p. 3). Tensor equation in redshift, before dividing through:
$$(1+z)^2(aH)^2\frac{d^2\chi_s}{dz^2} + (1+z)(aH)^2\,\epsilon\,\frac{d\chi_s}{dz}
+ \Big(k^2 - (aH)^2(2-\epsilon)\Big)\chi_s
= 4a\int\frac{d^3q}{(2\pi)^3}\,e_s^{\ell m}(\mathbf{k})q_\ell q_m
\left\{\phi_{\mathbf{q}}\phi_{\mathbf{k}-\mathbf{q}} + \frac{2}{3(1+w_0)}
\Big(\phi_{\mathbf{q}}-(1+z)\frac{d\phi_{\mathbf{q}}}{dz}\Big)\Big(\phi_{\mathbf{k}-\mathbf{q}}-(1+z)\frac{d\phi_{\mathbf{k}-\mathbf{q}}}{dz}\Big)\right\}.$$
The coefficient $\tfrac{2}{3(1+w_0)}$ arises as $\tfrac{2M_P^2}{3H^2M_P^2}\tfrac{1}{a^2}\tfrac{1}{1+w_0}(aH)^2$
where the "2" is the over-written "8" (8/4 after the overall 4 was pulled out; see §5). The
first-derivative coefficient $\epsilon$ arises from $(1+z)(aH)^2 - (1+z)(a^2H^2 + a^2\dot H)$.
Confidence: high.

**R19** (`NUM` 03 p. 4). Same equation divided by $(1+z)^2(aH)^2$:
$$\frac{d^2\chi_s}{dz^2} + \frac{\epsilon}{1+z}\frac{d\chi_s}{dz}
+ \left(\frac{k^2}{a^2H^2}\frac{1}{(1+z)^2} - \frac{2-\epsilon}{(1+z)^2}\right)\chi_s
= \frac{4}{a}\frac{1}{H^2}\int\frac{d^3q}{(2\pi)^3}\frac{1}{(1+z)^2}\,e_s^{\ell m}(\mathbf{k})q_\ell q_m
\left\{\phi_{\mathbf{q}}\phi_{\mathbf{k}-\mathbf{q}} + \frac{2}{3(1+w_0)}
\Big(\phi-(1+z)\frac{d\phi}{dz}\Big)_{\mathbf{q}}\Big(\phi-(1+z)\frac{d\phi}{dz}\Big)_{\mathbf{k}-\mathbf{q}}\right\}.$$
Confidence: high.

**R20** (`NUM` 03 p. 4). Inflationary initial condition / transfer-function substitution:
$$\phi_{\mathbf{k}} = \frac{3(1+w^*)}{5+3w^*}\,T_k(z)\,\zeta^*_{\mathbf{k}},\qquad
\text{"where $w^*$ is the fixed value of $w$ at the initial time".}$$
Confidence: high for the prefactor and $T_k(z)$; **medium** for the symbol $\zeta^*$ — the glyph is
a "5"-like character with subscript $\mathbf{k}$ and superscript $*$, read as $\zeta$ (curvature
perturbation); alternatives would be $S$ or $\mathcal{S}$. The same glyph appears in the dimension
check on p. 6 ("$h_s$ and $\zeta$ should be dimensionless") and as $P^*$ on pp. 7–10, supporting $\zeta$.

**Author note (2026-09-07):** $w^*$ is the value of $w$ at $z_{\rm init}$, which must lie deep inside an epoch of
constant $w$; for radiation the prefactor is $\tfrac23$. It is distinct from $w_0 = w(z')$ inside $f$ (R22). The code's
$T_k$ is normalised to 1 at $z_{\rm init}$, which is a normalisation of $T_k$ only and says nothing about the
seed; the translation from the seed to $\phi^*$ is this prefactor. See §0.2. **The glyph is $\zeta^*$, the primordial curvature perturbation (Tier 1.4, signed off 2026-09-07, §0.4); $P^* = P_\zeta$.**

**R21** (`NUM` 03 p. 4). Definition (under-brace on the page):
$$Q_s(\mathbf{q}) \equiv e_s^{\ell m}(\mathbf{k})\,q_\ell q_m .$$
On p. 7 it is also written $Q_s(\mathbf{k},\mathbf{q})$; on p. 9 the explicit $Q_\pm$ are written with
$q_{\rm phys}$ (i.e. including the $1/a_0^2$; see R33 and §6). Confidence: high.

**R22** (`NUM` 03 p. 4). **Source term in redshift, with transfer functions** (last equation on the
page; the curly bracket is red-boxed on the page):
$$\frac{d^2\chi_s}{dz^2} + \frac{\epsilon}{1+z}\frac{d\chi_s}{dz}
+ \left(\frac{k^2_{\rm phys}}{H^2} + \frac{\epsilon-2}{(1+z)^2}\right)\chi_s
= \frac{1}{a}\frac{1}{H^2}\frac{1}{(1+z)^2}\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf{q})\,
\frac{36(1+w^*)^2}{(5+3w^*)^2}\,\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}
\left\{T_qT_{k-q} + \frac{2}{3(1+w_0)}\Big(T-(1+z)\frac{dT}{dz}\Big)_q\Big(T-(1+z)\frac{dT}{dz}\Big)_{k-q}\right\}.$$
Here $k^2_{\rm phys}/H^2$ replaces $\tfrac{k^2}{H^2a_0^2}\tfrac{a_0^2}{a^2(1+z)^2}$ of the previous line
(so $k_{\rm phys}=k/a_0$), and $36 = 4\times 9$ (a leading "4" is struck out; see §5). The curly bracket
is what later pages call $f$:
$$f(z\,|\,\mathbf{q},\mathbf{k}-\mathbf{q}) \equiv T_q(z)\,T_{|\mathbf{k}-\mathbf{q}|}(z)
+ \frac{2}{3(1+w_0)}\Big(T_q - (1+z)\frac{dT_q}{dz}\Big)\Big(T_{|\mathbf{k}-\mathbf{q}|} - (1+z)\frac{dT_{|\mathbf{k}-\mathbf{q}|}}{dz}\Big).$$
The identification of $f$ with this bracket is not written as an equation anywhere in the document;
it is the only reading consistent with pp. 5–10 and with the red box. Confidence: high for the
equation as written; the $f$ identification is an inference (flagged in §6).

**R23** (`NUM` 03 p. 5). Green's function solution for $\chi_s$ (Step 3):
$$\chi_s(z) = \frac{36(1+w^*)^2}{(5+3w^*)^2}\int_{z_{\rm init}}^{z}dz'\,G^\chi_k(z,z')\,
\frac{1}{a(z')}\frac{1}{H(z')^2}\frac{1}{(1+z')^2}\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf{q})\,
\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}\,f(z'\,|\,\mathbf{k},\mathbf{k}-\mathbf{q}).$$
(Arguments of $f$ written as $(\mathbf{k},\mathbf{k}-\mathbf{q})$ on this page; see §6.) Confidence: high.

**R24** (`NUM` 03 p. 5). Back to $h_s = \chi_s/a$, with $a_0$ factors made explicit and then cancelled:
$$h_s(z) = \frac{36(1+w^*)^2}{(5+3w^*)^2}\int_{z_{\rm init}}^{z}dz'\,G^\chi_k(z,z')\,
\frac{1}{H(z')^2}\,\frac{a_0}{a(z')}\frac{a_0}{a(z)}\frac{1}{(1+z')^2}
\int\frac{d^3q}{(2\pi)^3}\,\frac{Q_s(\mathbf{q})}{a_0^2}\,\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}\,f(z'\,|\,\mathbf{k},\mathbf{k}-\mathbf{q})$$
$$\phantom{h_s(z)} = \frac{36(1+w^*)^2}{(5+3w^*)^2}\int_{z_{\rm init}}^{z}dz'\,G^\chi_k(z,z')\,
\frac{1}{H(z')^2}\,\frac{1+z}{1+z'}
\int\frac{d^3q}{(2\pi)^3}\,\frac{Q_s(\mathbf{q})}{a_0^2}\,\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}\,f(z'\,|\,\mathbf{k},\mathbf{k}-\mathbf{q}).$$
Confidence: high.

**R25** (`NUM` 03 pp. 5–6). Green's function definition and sign convention. "$G^\chi_k$ is the usual
Green's function satisfying"
$$\frac{d^2G_k}{dz^2} + \frac{\epsilon}{1+z}\frac{dG_k}{dz} + \left(\frac{k^2}{H^2} + \frac{\epsilon-2}{(1+z)^2}\right)G_k = \delta(z-z').$$
Defining $\bar G_k \equiv -G_k$: integrating across $z'$ gives $\big[d\bar G_k/dz\big]_{z'-\epsilon}^{z'+\epsilon} = -1$;
imposing $\bar G_k(z,z') = d\bar G_k(z,z')/dz = 0$ for $z>z'$ ("to get the causal Green fn") gives
$$\frac{d\bar G_k}{dz}\Big|_{z=z'} = +1 .$$
Note the $k^2/H^2$ in the operator is written without the subscript "phys" that R22 uses for the same
term. Confidence: high.

**Author note (2026-09-07):** $\bar G_k$ defined here is **the Green's function the code computes**
(`ComputeTargets/GkNumericIntegration.py`: $G(z',z') = 0$, $dG/dz|_{z'} = +1$, zero for $z > z'$). The $k^2/H^2$
is $k^2/(a_0^2H^2)$ with $a_0$ absorbed into $k/a_0$, not $a_0 = 1$. Relation to the conformal-time
function of `MAIN` 13/14: $\bar G_k = -a_0H(z')\,{\rm Gr}_k$. See §0.

**R26** (`NUM` 03 p. 6). **Final form of $h_s(z)$ with the causal Green's function** (sign of $\bar G$
absorbed by reversing the limits):
$$h_s(z) = 36\left(\frac{1+w^*}{5+3w^*}\right)^2\int_{z}^{z_{\rm init}}dz'\,\bar G_k(z,z')\,\frac{1+z}{1+z'}
\int\frac{d^3q}{(2\pi)^3}\,\frac{Q_s(\mathbf{q})}{a_0^2}\,\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}\,
\frac{f(z'\,|\,\mathbf{q},\mathbf{k}-\mathbf{q})}{H^2(z')}.$$
Confidence: high.

**R27** (`NUM` 03 p. 6). Dimension check (Step 4). "In real space $h_s$ and $\zeta$ should be
dimensionless, so $h^s_{\mathbf{k}}$ and $\zeta_{\mathbf{k}}$ have dimension $[M^{-3}]$." RHS:
$\int d^3q\;[M^3]\times Q_s\;[M^2]\times\zeta^*\zeta^*\;[M^{-6}]\times H^{-2}\;[M^{-2}] = [M^{-3}]$ ✓.
Confidence: high.

**R28** (`NUM` 03 p. 7). **Definition of the source redshift integral** (Step 5):
$$I_s(z\,|\,\mathbf{k},\mathbf{q}) = \int_{z}^{z_{\rm init}}dz'\,\bar G_k(z,z')\,\frac{1+z}{1+z'}\,
\frac{Q_s(\mathbf{k},\mathbf{q})}{a_0^2\,H^2(z')}\,f(z'\,|\,\mathbf{k},\mathbf{k}-\mathbf{q}),$$
$$h_s(\mathbf{k}) = 36\left(\frac{1+w^*}{5+3w^*}\right)^2\int\frac{d^3q}{(2\pi)^3}\,
\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}\,I_s(z\,|\,\mathbf{k},\mathbf{q}).$$
Confidence: high for the structure; **medium** for the denominator "$a_0^2H^2(z')$" — a small mark
after $H^2$ could be read as a subscript "0", but $H(z')$ with an argument is the consistent reading.
**Author sign-off (2026-09-07):** $a_0^2H^2(z')$ confirmed (Tier 2.9).

**Author note (2026-09-07):** the $Q_s/a_0^2$ here is exactly cancelled by the $a_0^2$ that
$dz'\,\bar G_k = (-a_0H\,d\eta')(-a_0H(z')\,{\rm Gr}_k)$ produces on conversion to conformal time, so
$I_s = -\dfrac{Q_s}{c^2}\times$ (`MAIN` 14 target, spec 05 R31) with $c^2 = (2+b)^2/(3+2b)^2$, independent of $a_0$ (§0).

**R29** (`NUM` 03 p. 7). Remarks on $a_0$-dependence: $\dfrac{Q_s}{a_0^2H^2(z')} \propto \dfrac{q^2_{\rm phys}}{H^2(z')}$;
"$\int d^3q\,\zeta^*_{\mathbf{q}}$ is independent of $a_0$"; "$\zeta^*_{\mathbf{k}-\mathbf{q}}$ gives $h_s(\mathbf{k})$
the right dependence on $a_0$ to be a comoving Fourier transform." Confidence: high.

**R30** (`NUM` 03 pp. 7–8). Two-point function by Wick contraction:
$$\langle h_s(\mathbf{k})h_{s'}(\mathbf{k}')\rangle = 1024\left(\frac{1+w^*}{5+3w^*}\right)^4
\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\,
\big\langle\zeta^*_{\mathbf{q}}\zeta^*_{\mathbf{k}-\mathbf{q}}\zeta^*_{\mathbf{t}}\zeta^*_{\mathbf{k}'-\mathbf{t}}\big\rangle\,
I_s(z\,|\,\mathbf{k},\mathbf{q})\,I_{s'}(z\,|\,\mathbf{k}',\mathbf{t})$$
$$= 1024\left(\frac{1+w^*}{5+3w^*}\right)^4\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}
\Big\{(2\pi)^6\delta(\mathbf{q}+\mathbf{t})\,\delta(\mathbf{k}-\mathbf{q}+\mathbf{k}'-\mathbf{t})
+ (2\pi)^6\delta(\mathbf{q}+\mathbf{k}'-\mathbf{t})\,\delta(\mathbf{k}-\mathbf{q}+\mathbf{t})\Big\}
P^*(q)P^*(|\mathbf{k}-\mathbf{q}|)\,I_s(z\,|\,\mathbf{k},\mathbf{q})\,I_{s'}(z\,|\,\mathbf{k}',\mathbf{t})$$
$$= (2\pi)^3\delta(\mathbf{k}+\mathbf{k}')\;1024\left(\frac{1+w^*}{5+3w^*}\right)^4
\int\frac{d^3q}{(2\pi)^3}\,P^*(q)P^*(|\mathbf{k}-\mathbf{q}|)
\Big(I_s(z\,|\,\mathbf{k},\mathbf{q})\,I_{s'}(z\,|\,\mathbf{k}',-\mathbf{q}) + I_s(z\,|\,\mathbf{k},\mathbf{q})\,I_{s'}(z\,|\,\mathbf{k}',\mathbf{k}'+\mathbf{q})\Big).$$
**The coefficient 1024 is red-boxed on both pages with the typed annotation (p. 7) "DS 10 July 2025
— Looks like this should be 1296 = 36^2, not 1024".** (Arithmetically, $36^2=1296$ and $32^2=1024$;
the prefactor of $h_s$ in R26–R28 is 36.) Confidence: high (the page's original is unambiguously
1024).

**Author sign-off (2026-09-07):** the annotation is right, **1296** on both p. 7 and p. 8. The 1024 is $32^2$, the
`MAIN` 11 coefficient squared; see §0.2 for why `MAIN` 11/14 have 32 and `NUM` 03 has 36.

**R31** (`NUM` 03 p. 8). Symmetry reduction of the two $I I$ terms. "$I_s$ should be regarded as a
function of $\mathbf{q}$ and $\mathbf{k}-\mathbf{q}$, not $\mathbf{k}$ and $\mathbf{q}$ separately, except
for the Green's function"; "the Green's function only depends on $k=|\mathbf{k}|=|\mathbf{k}'|$":
$$I_s(\mathbf{q},\mathbf{k}-\mathbf{q})\,I_{s'}(-\mathbf{q},\mathbf{q}-\mathbf{k}) + I_s(\mathbf{q},\mathbf{k}-\mathbf{q})\,I_{s'}(\mathbf{q}-\mathbf{k},-\mathbf{q})
= 2\,I_s(\mathbf{q},\mathbf{k}-\mathbf{q})\,I_{s'}(-\mathbf{q},\mathbf{q}-\mathbf{k}) = 2\,I_s(\mathbf{q},\mathbf{k}-\mathbf{q})^2,$$
"because source $f$ is symmetric and $Q^s$ is blind to $\mathbf{k}$" and "because source only depends on
$|\mathbf{q}|$ and $|\mathbf{k}-\mathbf{q}|$". (The step $I_{s'}\to I_s$, i.e. the $\delta_{ss'}$, is
justified on p. 9 by the $\varphi$-integrals of R33.) Confidence: high.

**R32** (`NUM` 03 p. 9). Two-point function after the symmetry reduction:
$$\langle h_s(\mathbf{k})h_s(\mathbf{k}')\rangle = (2\pi)^3\delta(\mathbf{k}+\mathbf{k}')\;2048\left(\frac{1+w^*}{5+3w^*}\right)^{4}
\int\frac{d^3q}{(2\pi)^3}\,P^*(q)P^*(|\mathbf{k}-\mathbf{q}|)\,I_s(z\,|\,\mathbf{q},\mathbf{k}-\mathbf{q})^2 .$$
"2048" red-boxed, typed annotation "DS 10 July 2025 should be 2592" ($=2\times1296$). The exponent on
the bracket is written over a struck "2" and reads 4. Confidence: high.

**Author sign-off (2026-09-07):** **2592** $= 2\times1296$, as annotated (§0.2).

**R33** (`NUM` 03 p. 9). Explicit polarisation projections and azimuthal integrals:
$$Q_+ = \frac{1}{\sqrt2}\,q^2_{\rm phys}\sin^2\theta\,\cos2\varphi,\qquad
Q_- = \frac{1}{\sqrt2}\,q^2_{\rm phys}\sin^2\theta\,\sin2\varphi,$$
$$\int_0^{2\pi}d\varphi\,Q_+Q_- = 0,\qquad \int_0^{2\pi}d\varphi\cos^22\varphi = \int_0^{2\pi}d\varphi\sin^22\varphi = \pi .$$
(Written $\phi$ on the page for the azimuth; rendered $\varphi$ here to avoid a clash with the
potential $\phi$.) Confidence: high.

**R34** (`NUM` 03 p. 9). After the $\varphi$ integral (two equivalent forms on the page):
$$\langle h_s(\mathbf{k})h_{s'}(\mathbf{k}')\rangle = (2\pi)^3\delta(\mathbf{k}+\mathbf{k}')\,\delta_{ss'}\;1024\left(\frac{1+w^*}{5+3w^*}\right)^4\pi
\int\frac{d^3q}{(2\pi)^3}\,P^*(q)P^*(|\mathbf{k}-\mathbf{q}|)
\left\{\int_z^{z_{\rm init}}dz'\,\bar G_k(z,z')\,\frac{1+z}{1+z'}\,\sin^2\theta\,\frac{q^2_{\rm phys}}{H(z')^2}\,f(z'\,|\,\mathbf{q},\mathbf{k}-\mathbf{q})\right\}^2$$
$$= (2\pi)^3\delta(\mathbf{k}+\mathbf{k}')\,\delta_{ss'}\;1024\pi\left(\frac{1+w^*}{5+3w^*}\right)^4
\int\frac{q^2\,dq\,\sin\theta\;4\pi^4}{(2\pi)^3}\,\frac{\mathcal{P}^*(q)}{q^3}\frac{\mathcal{P}^*(r)}{r^3}\,\sin^4\theta
\left\{\int_z^{z_{\rm init}}dz'\,\bar G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{q^2_{\rm phys}}{H(z')^2}\,f(z\,|\,\mathbf{q},\mathbf{k}-\mathbf{q})\right\}^2 .$$
Here $r=|\mathbf{k}-\mathbf{q}|$, $\mathcal{P}^*$ is the dimensionless spectrum, and the factor
$\tfrac12$ from $Q_\pm^2$ has been combined with 2048 to give 1024 (a struck "$q^4_{\rm phys}/2$"
appears outside the braces on the first line; see §5). "1024" red-boxed, typed annotation
"DS 10 July 2025 should be 1292" (sic — see §6). Confidence: high for the structure; **medium** for
(i) the $d\theta$, which is not written ("$q^2\,dq\,\sin\theta$" only), and (ii) the argument of $f$ in
the last brace, which appears to lack the prime on $z$.

**Author sign-off (2026-09-07):** both red-boxed 1024s on this page are **1296**, i.e. the coefficient is $1296\pi$;
the annotation "1292" is a typo for 1296 (§0.2). The argument of $f$ is $z'$ (Q4). The struck
$q^4_{\rm phys}/2$ outside the braces is crossed out and unused; the remaining expression is correct (Tier 2.10).

**R35** (`NUM` 03 p. 10). **Final formula of the document:**
$$\langle h_s(\mathbf{k})h_{s'}(\mathbf{k}')\rangle = (2\pi)^3\delta(\mathbf{k}+\mathbf{k}')\,\delta_{ss'}\;512\pi^2\left(\frac{1+w^*}{5+3w^*}\right)^4
\int\frac{dq}{q}\,\sin^5\theta\;\mathcal{P}^*(q)\,\frac{\mathcal{P}^*(r)}{r^3}
\left\{\int_z^{z_{\rm init}}dz'\,G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{q^2_{\rm phys}}{H(z')^2}\,f(z\,|\,\mathbf{q},\mathbf{k}-\mathbf{q})\right\}^2 .$$
"512" red-boxed, typed annotation "DS 10 July 2025 should be 646" (sic — see §6). The $\theta$ integral
is implicit (no $d\theta$ written); $r=|\mathbf{k}-\mathbf{q}|=\sqrt{k^2+q^2-2kq\cos\theta}$ is not
written out. Confidence: high for the coefficients and structure; **medium** for (i) whether the
Green's function carries a bar ($\bar G_k$ as on p. 9, or $G_k$ as it appears to be written here),
(ii) the missing $d\theta$, (iii) the argument of $f$, written $z$ rather than $z'$.

**Author sign-off (2026-09-07):** the coefficient is **$648\pi^2$**; "646" is a typo for 648 (§0.2). The
Green's function is the barred $\bar G_k$ (Q6; it is the function the code computes, §0.1) and the argument of
$f$ is $z'$ (Q4). The missing $d\theta$, the $r(\theta)$ and the limits are supplied in §0.3 (Tier 1.3, signed off):
$\int_0^\infty dq/q\int_0^\pi d\theta$, $r = \sqrt{k^2+q^2-2kq\cos\theta}$. **This, with $648\pi^2$, is the build form.**

---

## 4. Checks

Neither document is a recheck of another document. `NUM` 03 p. 1 quotes its starting equation from
`MAIN` 11 p. 3 without re-deriving it. `MAIN` 10 p. 4 checks its background equation against the
Friedmann acceleration equation ("which is the expected Friedmann acceleration equation" — agreed).
`NUM` 03 p. 6 performs a dimension check of R26 (agreed, R27).

---

## 5. Corrections and cross-outs

### `MAIN` 10

| Page | Item | Original | Correction / later use |
|---|---|---|---|
| 2 | Typed red box + annotation on $T^0{}_0$ | $-\rho_0 + \delta\rho$ | Annotation: "sign flip on delta \rho; should be negative" → $-\rho_0-\delta\rho$. Not used later in this document. |
| 2 | Struck line in component (B) | "$(\rho+p)\,\tfrac{1}{a}e^{-\phi}\gamma_{SR}$" (incomplete) | Rewritten in full on the next line (R4). |
| 2 | Component (B), second line | "$a\,e^{-2\psi}$" with "$e^{-\phi}$" inserted above | Read as $a\,e^{-\phi-2\psi}$ (R4). |
| 4 | Superscript on $v_i$ | a small struck superscript | Read as plain $v_i$ (R10). |
| 4 | Denominator in R11 | "$a^{4}$" with an over-written mark | Read as $a^4$. |

### `NUM` 03

| Page | Item | Original | Correction / later use |
|---|---|---|---|
| 2, 3 | Coefficient of the velocity term after pulling out the overall 4 | "$8M_P^2$" | "2" written above the 8 (i.e. $8/4$). Later steps use 2 → $\tfrac{2}{3(1+w_0)}$ (R18–R22). |
| 4 | Last equation | leading "4 =" | "4" struck; absorbed into $36 = 4\times 9$ (R22). |
| 4 | Red box (typed, no text) around the curly bracket of R22 | — | Highlights the source function $f$. |
| 5 | Page number | circled "5" beside a scribbled-out circle | Page renumbered; no content affected. |
| 7 | Wick contraction, first $\delta\delta$ term | "$P^*(q)P^*(|\mathbf{k}-\mathbf{q}|)$" struck after the first term | Rewritten once after the closing brace, applying to both terms (R30). |
| 7 | Typed red box + annotation on 1024 | 1024 | "DS 10 July 2025 — Looks like this should be 1296 = 36^2, not 1024". |
| 8 | Typed red box on 1024 | 1024 | Same annotation (no new text). |
| 9 | Typed red box + annotation on 2048 | 2048 | "DS 10 July 2025 should be 2592". |
| 9 | Exponent on $(\tfrac{1+w^*}{5+3w^*})$ | "2" struck | "4" (R32). |
| 9 | Struck "$q^4_{\rm phys}/2$" outside the braces (first line of R34) | — | The $q^4\sin^4\theta$ moved inside the squared brace; the $\tfrac12$ combined $2048\to1024$. |
| 9 | Typed red box + annotation on 1024 (second form) | 1024 | "DS 10 July 2025 should be 1292" (sic). |
| 9 | Typed red box on "1024π" | — | No new text. |
| 10 | Typed red box + annotation on 512 | 512 | "DS 10 July 2025 should be 646" (sic). |

**Which values do later steps use?** The blue-ink chain on pp. 7–10 uses 1024 → 2048 → 1024π →
512π². The July 2025 annotations propose 1296 → 2592 → "1292" → "646"; the last two are not
arithmetically consistent with the chain ($2592/2 = 1296$, $2592/4 = 648$) and look like typos for
1296 and 648. The annotations do not restate the final formula, so no corrected final formula
exists on the pages.

---

## 6. Open questions

1. **Page count.** README says 46 pages for `NUM` 03; the PDF has 10 (see §1). If a longer scan
   exists elsewhere, it was not the file at the stated path.
2. **Coefficient annotations (`NUM` 03 pp. 7–10).** The typed annotations give 1296, 2592, "1292",
   "646". The third and fourth do not follow from the first two by the page's own steps
   ($\times\tfrac12\pi$ then $\times\tfrac{4\pi^4}{(2\pi)^3}=\tfrac{\pi}{2}$ would give $1296\pi$ and
   $648\pi^2$). Whether "1292" and "646" are typos, or whether the author intended a different
   correction, cannot be decided from the page. The original 1024 appears to arise from
   $32^2$ rather than $36^2$; the notes do not say where 32 would come from.
   **Closed 2026-09-07:** typos for 1296 and 648; the 32 is the `MAIN` 11/14 coefficient, whose $c^2$ sits inside
   $f$ rather than outside. Corrected chain and explanation in §0.2.
3. **Arguments of $f$.** Written $f(z'|\mathbf{k},\mathbf{k}-\mathbf{q})$ on pp. 5 and 7, but
   $f(z'|\mathbf{q},\mathbf{k}-\mathbf{q})$ on pp. 6 and 9, and p. 8 states $I_s$ "should be regarded as
   a function of $\mathbf{q}$ and $\mathbf{k}-\mathbf{q}$". Read as a notational slip on pp. 5, 7; the
   $(\mathbf{q},\mathbf{k}-\mathbf{q})$ form is the one consistent with R22.
   **Closed 2026-09-07:** $f(z'\,|\,\mathbf q,\mathbf k-\mathbf q)$; the other form is a slip (§0.5).
4. **$z$ vs $z'$ in $f$.** In the last brace of R34 and in R35 the argument of $f$ appears as $z$
   (no prime) although it sits inside the $dz'$ integral. Almost certainly $z'$; flagged because the
   prime is genuinely absent or too faint to see.
   **Closed 2026-09-07:** $z'$, the source time (§0.5).
5. **Definition of $f$.** Never written as "$f = \dots$"; identified with the red-boxed bracket of
   R22 by consistency. Also, $w_0$ inside $f$ is the background $p_0/\rho_0$ from the derivation on
   pp. 2–3; whether it is to be evaluated at the source redshift $z'$ (i.e. $w_0 = w(z')$) is not
   stated.
   **Closed 2026-09-07:** $w_0 = w(z')$ at the source time; $w^*$ in the prefactor is $w(z_{\rm init})$. The
   $f$ identification with the R22 bracket is confirmed, and it is what `ComputeTargets/QuadSource.py` implements. See §0.2.
6. **Bar on the Green's function, p. 10.** $\bar G_k$ (barred) on pp. 6, 7, 9; the p. 10 symbol looks
   unbarred. The limits $\int_z^{z_{\rm init}}$ are those of the barred form (R26), so $\bar G_k$
   is presumably meant.
7. **$k_{\rm phys}$ / $q_{\rm phys}$.** Never defined; inferred to mean $k/a_0$ (§2.2). On p. 4 $Q_s$
   is defined with comoving $q$ and the $1/a_0^2$ is carried separately (pp. 5–7), whereas on
   p. 9 $Q_\pm$ are written directly with $q_{\rm phys}$, so the $1/a_0^2$ has been absorbed into
   $Q_\pm$ there. Consistent, but the same symbol is used for both.
8. **Missing $d\theta$.** The measures on pp. 9–10 read "$q^2\,dq\,\sin\theta$" and "$\tfrac{dq}{q}\sin^5\theta$"
   with no $d\theta$; the $\theta$ integration over $[0,\pi]$ is implied by the preceding $d^3q$.
9. **$H^2(z')$ in R28.** Possible stray subscript; read as $H(z')$.
10. **`MAIN` 10 R5.** "$(p_0+p)\,\delta^i{}_j$" — presumably $p_0+\delta p$.
11. **Cross-document factor (for the orchestrator's cross-spec check, not an inconsistency within
    either document).** `MAIN` 10 R11 gives the quadratic stress as
    $\tfrac{4M_P^4}{a^4(\rho_0+p_0)}\,\partial^i(\phi'+\mathcal{H}\phi)\partial_j(\phi'+\mathcal{H}\phi)$,
    while the source quoted in `NUM` 03 R12 (from `MAIN` 11 p. 3) has
    $\tfrac{8M_P^2}{(\rho_0+p_0)a^2}\,q_\ell r_m(\phi'+\mathcal{H}\phi)_q(\phi'+\mathcal{H}\phi)_r$. The
    conversion (factor $M_P^{-2}$ from the Einstein equation, $a^2$ from the tensor equation's
    normalisation, and the factor 2, plus the relative sign of R12's $4q_\ell r_m\phi\phi$ term) is
    done in `MAIN` 11 (Group 5) and is not on these pages.
12. **"$Q^s$ is blind to $\mathbf{k}$" (p. 8).** Transcribed verbatim; presumably means that
    $Q_s(\mathbf{q}) = e_s^{\ell m}(\mathbf{k})q_\ell q_m$ is unchanged under $\mathbf{k}\to-\mathbf{k}$ (and
    $\mathbf{q}\to-\mathbf{q}$), but the page does not elaborate.
