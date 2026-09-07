# Spec 02: Tensor Green's function

Transcribed 2026-09-07 from the handwritten notes listed below (spec-transcription campaign, Group 2).
Sign-off: **Tier 1.1 (Green's-function normalisation), Tier 2.6, 2.7, 2.8 and the Tier 3 notation items signed off by the author 2026-09-07; see §0. All review-queue items for this file are closed** (Tier 1.2–1.4 do not concern this file).

Notation used in this file: `MAIN 13` = document 13 of the main line, `NUM nn` = document nn of the
numerical-implementation folder; page references are of the form `NUM 05 p.3`. Results are numbered
`R1, R2, …` continuously across the whole file. Confidence flags refer to the *reading* of the page,
not to the physics.

---

## 0. Author sign-off notes

### 0.1 Tier 1.1 — Green's-function normalisation: **signed off 2026-09-07**

The review queue (`REVIEW-QUEUE.md` §1.1) asked which of three redshift-space Green's functions the
code computes and whether the $a_0$ bookkeeping between `NUM` 03, `NUM` 06 and `MAIN` 14 closes.
Both are settled below. What follows is the **definitive project convention**, fixed by what the
code actually does. Later readers and agents should take it as authoritative over any wording in
the handwritten notes or in the transcription sections of these spec files.

**(1) Normalisation of the scale factor: $a_0$ is absorbed, not set to one.**
The code's wavenumber variable is the physical wavenumber today, $k/a_0$
(`CosmologyConcepts/wavenumber.py`, "the normalization $a_0$ of $a(z)$ is absorbed into
$k/a_0 = k_{\rm phys}$"), and its conformal-time variable is $a_0\eta$
(`ComputeTargets/BackgroundModel.py` integrates $d(a_0\eta)/dz = -1/H$). Every formula in the code
is therefore written in the two $a_0$-invariant combinations $k/a_0$ and $a_0\eta$; their product
$k\eta$ is what every Bessel function sees. The handwritten notes "drop" or "hold back" $a_0$ in
exactly this sense: a power of $a_0$ is never discarded, it is understood to combine with a comoving
momentum into a physical one. This is deliberate. It forces a check that no stray power of $a_0$
survives in a physical result: *any expression written in comoving variables must be invariant under
$a_0 \to \lambda a_0$ with $q, r, k \to \lambda q, \lambda r, \lambda k$ and
$\eta, \eta' \to \eta/\lambda, \eta'/\lambda$ (physical momenta and $H(z)$ fixed).*
**Do not read the absence of $a_0$ in the code, or in a formula, as the assumption $a_0 = 1$; and do
not read a "missing" $a_0$ in a comoving-variable formula as an error until this covariance test has
been applied.**

**(2) The Green's function the code computes.** `ComputeTargets/GkNumericIntegration.py`
integrates the homogeneous operator
$$
\frac{d^2G}{dz^2} + \frac{\epsilon}{1+z}\frac{dG}{dz}
+ \Big(\frac{k^2}{a_0^2H^2} - \frac{2-\epsilon}{(1+z)^2}\Big)G = 0
$$
from $z = z'$ downward (forward in time) with $G(z',z') = 0$ and $dG/dz\big|_{z=z'} = +1$; the
function is zero for $z > z'$. This is the **unit-jump causal Green's function for $\chi_s = a\,h_s$
in redshift**: the operator above acting on it gives $-\delta(z-z')$. It is exactly $\bar G_k$ of
`NUM` 03 (spec 03 R25) and $G_{\rm me}$ of `NUM` 06 (spec 04 R1–R2). It is related to the retarded
conformal-time Green's function ${\rm Gr}_k(\eta,\eta')$ of `MAIN` 13/14 (spec 02 R9, spec 05 R18,
R21), which has source $+\delta(\eta-\eta')$ and unit jump in $d/d\eta$, by
$$
G_{\rm code}(z,z') = -\,a_0\,H(z')\;{\rm Gr}_k\big(\eta(z),\eta(z')\big),
$$
valid for **any** $a_0$. Because ${\rm Gr}_k \propto \sqrt{\eta\eta'}$, the right-hand side written
in the code's variables is $-H(z')\,\frac{\pi}{2}\sqrt{\tau\tau'}\{\dots\}$ with $\tau = a_0\eta$ and
no $a_0$ left over; that is what `ComputeTargets/analytic_Gk.py` evaluates, and it is the meaning of
the code comment "$G_{\rm us} = -H(z')\,G_{\rm them}$". $G_{\rm code}$ is negative just after the
source ($G_{\rm code} \approx z - z'$ for $z \lesssim z'$). First argument = response redshift,
second = source redshift.

**(3) Sign of the redshift Jacobian (`NUM` 02).** `NUM` 02 transforms $\delta(\eta-\eta')$ with
$dz/d\eta = -a_0H$ *without* taking the absolute value, giving source $-\frac{1}{a_0H}\delta(z-z')$
(spec 02 R17–R18). This is a convention, not a slip: it is paired with the orientation of the $z'$
integration measure ($\int_{z_{\rm init}}^{z}dz'$), and the two together reproduce the same
$\chi_s$. `NUM` 03 and `NUM` 06 later adopt the unit jump $dG/dz|_{z'} = +1$, with the source
$-\delta(z-z')$ chosen so that the jump is natural for $z$ decreasing to the future; the code follows
`NUM` 03/06. The relative minus sign between the code's source integral and the `MAIN` 14 target is
this orientation convention; it enters the one-loop result squared.

**(4) Consequence for `NUM` 06 R14 and the $a_0$ hand-off.** Under the covariance test in (1), the
right-hand side of spec 04 R14 written in comoving variables scales as $\lambda^{-2}$
($(qrc_s^2\eta)^{-1/2-b} \to \lambda^{-1/2-b}$, $\int d\eta'(\eta')^{1/2-b} \to \lambda^{-3/2+b}$),
while the left-hand side is invariant. The prefactor must therefore be $a_0^2$, not the $a_0^1$
written on the page. The dropped power enters through R2, which is written in the absorbed
convention ($G_{\rm me} = -H(z')G_{\rm them}$ with $G_{\rm them}$ in the variable $a_0\eta$), while
R13 keeps $a_0$ explicit ($dz' = -a_0H\,d\eta'$). In the code's variables `analytic_integral` in
`ComputeTargets/QuadSourceIntegral.py` carries no $a_0$, which *is* the $a_0^2$ form. With $a_0^2$
the source integral cancels the $Q_s/a_0^2$ of spec 03 R28 exactly (i.e. $Q_s$ is built from
physical momenta), $h_s$ is $a_0$-independent as it must be, and the `NUM` 03/06 source integral
equals $-Q_s/c^2$ times the `MAIN` 14 target (spec 05 R31), with $c^2 = (2+b)^2/(3+2b)^2$, as
`cross-spec-check.md` §2 found. **The three "differently normalised" Green's functions of the review
queue are one object.**

### 0.2 Tier 2 and Tier 3 items — **signed off 2026-09-07**

- **2.6 (NUM 02 p.2, R13).** The $a(aH^2+a\dot H)\,(1+z)\,dG_k/dz$ term carries a **minus** sign, as
  transcribed; this is what makes the friction coefficient collapse to $\epsilon$ overall (R15–R17).
- **2.7 (MAIN 13 p.3, Q8).** The "$\alpha$" is the symbol $\propto$: "both sides are proportional to".
  The $Y_{b+1/2}(k\eta')$ factor is temporarily suppressed and restored later; R9 is unaffected.
- **2.8 (NUM 11 p.5–6, R38, Q10).** The intent is $\Theta-\Theta_i\sim\omega_{\rm eff}u$, so
  $|Q|\to1$ for $u\gg1$. With the page's (and the code's) convention $d\Theta/dz=+\omega_{\rm eff}$
  and $u=z_i-z$, the phase *decreases* with $u$ and the fixed point is $Q=-1$; "close to unity" is to
  be read in magnitude. The code (`Quadrature/integrators/WKB_phase_function.py`, stage 2) implements
  R38 verbatim, starts from $Q(0)=0$, and nothing in it assumes the sign of $Q$ (the supervisor only
  records its extremes), so this is a wording point only. Spec statement: $Q\to-1$ under the
  $+\omega_{\rm eff}$ convention; equivalently $Q\to+1$ with the lower sign of R33's $\pm$.
- **Tier 3, $k$ vs $k_{\rm phys}$ (Q6).** The bare $k^2/H^2$ of NUM 05/10/11 is
  $k_{\rm phys}^2/H^2=k^2/(a_0^2H^2)$, as NUM 02 writes it. Consistent with §0.1 item 1.
- **$\eta_0$ / $z_{\rm init}$.** Not a spec constant: it is initial data in the numerical pipeline,
  set per mode as the redshift a fixed number of e-folds before horizon exit
  (`CosmologyConcepts/wavenumber.py`, `z_exit_suph_eN`). The spec records only the conditions it must
  satisfy: (i) $w$ constant at $z_{\rm init}$ (so $w^*$ is defined, spec 03 §0.2); (ii) every mode
  entering the calculation — $k$, $q$ and $r$ — is super-horizon there, so $T\to1$ and $\phi$ is
  constant. The `MAIN` 14 target is insensitive to $\eta_0$: near $\eta'\to0$ its $Y$-kernel integrand
  behaves as $(\eta')^{1}$ and its $J$-kernel integrand as $(\eta')^{2+2b}$, so the $\eta_0$-dependence
  is $O((k\eta_0)^2)$ and the analytic formula may be read with $\eta_0\to0$; the code's
  `analytic_integral` uses the finite $\eta_{\rm init}=\tau(z_{\rm init})$ in any case.
- **Overloaded symbols** (confirmed as traps, not errors): $\epsilon$ = slow-roll parameter
  $-\dot H/H^2$ and, in jump conditions, an infinitesimal; $\eta$ = conformal time and, in `NUM` 05 p.2
  only, the second slow-roll parameter $d\ln\epsilon/dN$; $\omega_{\rm eff}$ = the Green's-function
  frequency (spec 02 R22) and the transfer-function frequency (spec 01 R27), different functions;
  $Q_s$ written with comoving $q$ (`MAIN` 14) or with $q_{\rm phys}=q/a_0$ (`NUM` 03 p.9), differing
  by $a_0^{-2}$ (absorbed, spec 02 §0 item 1).

All review-queue items for this file are closed.

---

## 1. Source documents

Base directory: `/Users/ds283/Library/CloudStorage/Box-Box/Research projects/SIGWs/4D/Calculations`

| Tag | File | Pages | Date on first page |
|---|---|---|---|
| MAIN 13 | `Green's function formula for P22/13 - 2022:07:21 - tensor Green's function.pdf` | 3 | 21 July 2022 (headed "CALCULATION ⑬, Compute Green's function for $h_{ij}$") |
| NUM 02 | `Numerical implementation/02 - 2024-09-12 - Tensor Green's function in redshift.pdf` | 4 | 12 September 2024 |
| NUM 05 | `Numerical implementation/05 - 2024-09-26 - WKB(J) approximation for Green's functions.pdf` | 8 | 26 Sep 2024 (headed "⑤") |
| NUM 10 | `Numerical implementation/10 - 2024-12-08 - fix calculation for d omega_eff : dz for Gk.pdf` | 1 | 8 Dec 2024 (headed "$d\ln\omega_{\rm eff}/dz$ for $G_k$") |
| NUM 11 | `Numerical implementation/11 - 2025-03-24 - WKB repeat and matching conditions.pdf` | 10 | **23 Mar 2025** (file name says 24 March; see Open questions Q1) |

MAIN 13 is in blue ink; NUM 02, 05, 10 in blue ink with red corrections; NUM 11 in green ink with
red corrections. All are legible.

---

## 2. Conventions in force

### 2.1 Time variable
- **MAIN 13** (all pages): conformal time $\eta$. Response time $\eta$, source time $\eta'$.
- **NUM 02**: starts in conformal time (p.1), converts to redshift $z$ via cosmic time $t$ as an
  intermediate (p.1, $dz = -(1+z)H\,dt = -(1+z)aH\,d\eta$). From p.3 onward everything is in $z$;
  the source time is $z'$ ("$\eta'$ and $z'$ label the same source time", p.2).
- **NUM 05, NUM 10, NUM 11**: redshift $z$ throughout. Neither $\log(1+z)$ nor $\eta$ is used
  as an independent variable. In NUM 05 p.2 the e-fold number $N$ appears only inside the
  definition of a slow-roll parameter ($d\ln\epsilon/dN$).

### 2.2 Meaning of a prime
- **MAIN 13**: prime on a function is $d/d\eta$ ($G_k''$, $a''$, $Z''$; p.1). Prime on Bessel
  functions ($J'_{b+1/2}$, $Y'_{b+1/2}$; p.2–3) is the derivative with respect to the Bessel
  argument $k\eta$. Prime on $\eta$ ($\eta'$) is the source-time label, not a derivative.
- **NUM 02**: $G_k''$ on p.1 is $d^2/d\eta^2$ (copied from MAIN 13); afterwards derivatives are
  written out explicitly as $d/dz$, $d/dt$ or with a dot ($\dot H$, p.2). $z'$ is the source label.
- **NUM 05, 10, 11**: prime is $d/dz$ ($f'$, $\Delta'$, $\epsilon' = d\epsilon/dz$ stated explicitly
  NUM 05 p.2, $\omega'_{\rm eff}$, $G'$, $P'$, $\vartheta'$, $c_s'$). Dummy integration variables
  are $z'$. In NUM 05 the matching point is written $\tilde z$; in NUM 11 it is $z^*$, and a
  starred quantity ($G^*$, $\omega^*_{\rm eff}$, $H^*$, $T^*$, $T'^*$) means "evaluated at $z^*$"
  (NUM 11 p.4, p.9).

### 2.3 Hubble parameters and $\epsilon$
- **MAIN 13 p.1**: $\mathcal{H} = a'/a$ (conformal Hubble rate, script $\mathcal H$; glyph reading
  `medium`, see Open questions Q2). $\mathcal{H} = (1+b)/\eta$ for constant $w$.
- **NUM 02 p.1–3**: $H = \dot a/a$ (cosmic-time Hubble rate), since $dz = -(1+z)H\,dt$. $\epsilon$
  is introduced on p.2–3 via $2a^2H^2 + a^2\dot H = a^2H^2(2-\epsilon)$, i.e. $\dot H = -\epsilon H^2$
  (not written explicitly there; inferred from the algebra).
- **NUM 05 p.2**: $\epsilon = -\dot H/H^2 = (1+z)\,d\ln H/dz$ stated explicitly. Same in NUM 11 p.4,
  p.7 ($\epsilon = (1+z)\,d\ln H/dz$).
- **NUM 05 p.2** also introduces a second slow-roll parameter written $\eta$ (**not** conformal
  time): $\eta = d\ln\epsilon/dN = \epsilon^{-1}\,d\epsilon/(H\,dt)$, so that
  $\epsilon' = d\epsilon/dz = -\epsilon\eta/(1+z)$. See Open questions Q4.
- **NUM 02 p.4** uses $\epsilon$ a second time as an infinitesimal in the jump condition
  $[\,\cdot\,]_{z'-\epsilon}^{z'+\epsilon}$. This is unrelated to the slow-roll $\epsilon$ (Q5).
  **Author note (2026-09-07):** confirmed; see the overloaded-symbols list in §0.2.

### 2.4 Scale-factor normalisation
- **MAIN 13**: $a(\eta)\propto\eta^{1+b}$; no normalisation needed (only $a(\eta')/a(\eta)$ enters).
- **NUM 02 p.3**: $a_0$ kept **explicit**: $k^2/(a^2H^2) = (1+z)^2 k^2/(a_0^2H^2) \equiv (1+z)^2 k_{\rm phys}^2/H^2$,
  and $1/(aH) = (1+z)/(a_0 H)$. So here $k_{\rm phys} \equiv k/a_0$. The final source term carries
  $1/(a_0 H)$; the author comments that this $a_0^{-1}$ "looks unusual" but reflects that the
  scale of the $\chi$ field depends on the normalisation of $a$, and "should disappear when we
  switch the $\chi$ Green's function for the $h$ Green's function" (p.3–4).
  **Author note (2026-09-07):** $a_0$ is being held back, not set to one. In the final assembly
  this $1/a_0$ combines with the comoving momenta in $Q_s$ into physical momenta (§0 items 1, 4).
- **NUM 05, 10, 11 (Green's-function part)**: the mass term is written $k^2/H^2$ with no $a_0$ and
  no subscript "phys". Whether $k$ here means $k_{\rm phys}=k/a_0$ of NUM 02, or $a_0=1$ is
  assumed, is **not stated**; inferred to be $k/a_0$ (Q6).
- **NUM 11 p.7 (transfer-function part)**: $a_0/a = 1+z$ stated explicitly, giving
  $k^2c_s^2/(a_0^2H^2)$ with explicit $a_0$.

### 2.5 Green's function conventions
- **Field**: the Green's function is for the rescaled field $\chi$, with $h^s = \chi^s/a$, $s$ =
  polarisation label (NUM 02 p.1, referring to "2022 Calculation ⑪"). MAIN 13 p.3 calls it
  $G_k[\chi]$ and gives $G_k[h] = \frac{a(\eta')}{a(\eta)}G_k[\chi]$.
- **Source sign, conformal time**: $+\delta(\eta-\eta')$ on the right-hand side (MAIN 13 p.1;
  NUM 02 p.1).
- **Source sign, redshift**: the $\delta$-function is transformed as a density *without* an
  absolute value on the Jacobian: $\delta(\eta-\eta') = \delta(z-z')\,dz/d\eta = -(1+z)aH\,\delta(z-z')$
  (NUM 02 p.3). After dividing through, the source is $-\frac{1}{a_0 H}\delta(z-z')$ (NUM 02 p.3),
  and the jump condition is $[dG_k/dz]_{z'-\epsilon}^{z'+\epsilon} = -1/(a_0H(z'))$ (NUM 02 p.4).
  See Q3.
- **Arguments**: first argument = response time, second = source time: $G_k(\eta,\eta')$ with
  $\eta'$ the source (MAIN 13 p.2, "continuity condition $G_k(\eta')=0$"; NUM 02 p.4
  "$G_k(z',z')=0$"). Retarded/causal: $G_k = 0$ for $\eta<\eta'$ (MAIN 13 p.2–3). In redshift the
  corresponding statement ($G_k=0$ for $z>z'$) is **not written** in NUM 02 (Q7).
- **Boundary conditions**: continuity $G_k(\eta',\eta')=0$ and jump $[G_k']_{\eta=\eta'}=1$
  (MAIN 13 p.2); in $z$: $G_k(z',z')=0$, $[dG_k/dz]_{z'-\epsilon}^{z'+\epsilon} = -1/(a_0H(z'))$ (NUM 02 p.4).
- **Literature**: MAIN 13 p.3: the $h$-field Green's function "differs from Domènech et al.
  formula, but appears to match Adshead et al. formula".

### 2.6 Equation of state
- **MAIN 13 p.1**: $b = \dfrac{1-3w}{1+3w}$ for a period with fixed $w$; $a(\eta)\propto\eta^{1+b}$.
  $c_s^2$ is not used in MAIN 13.
- **NUM 02, 05, 10**: no $w$ or $b$; the background enters only through $H(z)$ and $\epsilon(z)$.
  NUM 05 p.4 specialises to $\Lambda$CDM with $\Omega_{m,0},\Omega_{r,0},\Omega_\Lambda$.
- **NUM 11 p.6–10** (transfer-function part): $c_s^2$ appears, **not defined** in this document;
  $c_s' = dc_s/dz$.

### 2.7 Fourier / power-spectrum conventions
Not stated in any of the five documents. Only a single Fourier label $k$ appears; no $(2\pi)^3$
factors, no $\delta$-function normalisation, no polarisation-tensor normalisation are written.
The polarisation label $s$ is mentioned once (NUM 02 p.1).

### 2.8 Transfer function
Only NUM 11 Steps 6–8 (p.6–10) concern a transfer function, denoted $\phi$ and then $\vartheta$
($\phi = P\vartheta$), with WKB solutions $T_1$, $T_2$ and matching data $T^*$, $T'^*$. Which
potential ($\phi$, $\Phi$, $\zeta$, $\mathcal R$) and its early-time normalisation are **not
stated** in NUM 11; the equation is simply quoted. Cross-reference `01-transfer-function.md`.

### 2.9 Momentum labels
Only $\mathbf k$ (subscript $k$ on $G_k$). No loop momenta appear.

---

## 3. Results

### 3.1 MAIN 13 — tensor Green's function in conformal time (3 pp., 21 July 2022)

**R1** (MAIN 13 p.1). Defining equation of the tensor Green's function (Step 1):
$$
G_k'' + \Big(k^2 - \frac{a''}{a}\Big) G_k = \delta(\eta-\eta').
$$
Green's function for the rescaled tensor field (see R11). Confidence: **high**.

**R2** (MAIN 13 p.1). Constant-$w$ background:
$$
b = \frac{1-3w}{1+3w},\qquad a(\eta)\propto\eta^{1+b},\qquad
\mathcal H = \frac{a'}{a} = \frac{1+b}{\eta},\qquad
\frac{a''}{a} = -\frac{1+b}{\eta^2} + \frac{(1+b)^2}{\eta^2} = \frac{b(1+b)}{\eta^2}.
$$
Confidence: **high** (the $\mathcal H$ glyph itself: medium, Q2).

**R3** (MAIN 13 p.1). Reduced Green's-function equation ("Hence"):
$$
G_k'' + \Big(k^2 - \frac{b(1+b)}{\eta^2}\Big) G_k = \delta(\eta-\eta').
$$
Confidence: **high**.

**R4** (MAIN 13 p.1). Homogeneous solutions of $Z'' + \big(k^2 - b(1+b)/\eta^2\big)Z = 0$ (Step 2):
$$
Z = \sqrt{\eta}\,\alpha\,J_{b+1/2}(k\eta)\ \text{(growing)},\qquad
Z = \beta\sqrt{\eta}\,Y_{b+1/2}(k\eta)\ \text{(decaying)}.
$$
Bessel index $b+\tfrac12$ in both. Confidence: **high** for the index; a stray glyph
(possibly a struck "$\partial$") precedes $Y$ in the decaying solution (see §5).

**R5** (MAIN 13 p.2). Causal (retarded) ansatz and matching conditions (Step 3):
$$
G_k(\eta,\eta') = \begin{cases}\sqrt{k\eta}\,\big(\alpha J_{b+1/2}(k\eta) + \beta Y_{b+1/2}(k\eta)\big) & \eta>\eta'\\ 0 & \eta<\eta'\end{cases}
$$
with jump condition $[G_k']_{\eta=\eta'} = 1$ and continuity $G_k(\eta')=0$, i.e.
$$
\alpha\sqrt{k\eta'}\,J_{b+1/2}(k\eta') + \beta\sqrt{k\eta'}\,Y_{b+1/2}(k\eta') = 0,
$$
$$
\big\{\alpha k J'_{b+1/2}(k\eta') + \beta k Y'_{b+1/2}(k\eta')\big\}\sqrt{k\eta'}
+ \tfrac12\sqrt{k/\eta'}\,\big\{\alpha J_{b+1/2}(k\eta') + \beta Y_{b+1/2}(k\eta')\big\} = 1,
$$
the second brace being annotated (red) "0 by continuity at $\eta=\eta'$". Confidence: **high**.

**R6** (MAIN 13 p.2). Continuity fixes
$$
\beta = -\alpha\,\frac{J_{b+1/2}(k\eta')}{Y_{b+1/2}(k\eta')},
\qquad\text{so}\qquad
\alpha\big\{Y_{b+1/2}J'_{b+1/2} - J_{b+1/2}Y'_{b+1/2}\big\}(k\eta') = \frac{Y_{b+1/2}(k\eta')}{k\,(k\eta')^{1/2}}.
$$
Confidence: **high** for $\beta$; **medium** for the placement of $Y_{b+1/2}(k\eta')$ in the
numerator of the last expression — it is written above and to the right of the fraction bar,
and is dropped on p.3 (Q8).

**R7** (MAIN 13 p.3). Wronskian ("Abel's theorem"):
$$
J_\nu Y_\nu' - Y_\nu J_\nu' = \frac{2}{\pi z}.
$$
Confidence: **high**.

**R8** (MAIN 13 p.3). Normalisation constant, as written:
$$
\alpha\Big(-\frac{2}{\pi}\frac{1}{k\eta'}\Big) = \frac{1}{k\,(k\eta')^{1/2}}
\quad\Rightarrow\quad
\alpha = -\frac{\pi}{2}\,\frac{(k\eta')^{1/2}}{k}.
$$
Confidence: **high** as a reading; note it omits the $Y_{b+1/2}(k\eta')$ factor of R6 (Q8).

**R9** (MAIN 13 p.3). **Retarded Green's function for the $\chi$ field** (final form of Step 3):
$$
G_k(\eta,\eta') = \begin{cases}
\dfrac{\pi}{2}\sqrt{\eta\eta'}\,\Big\{-Y_{b+1/2}(k\eta')\,J_{b+1/2}(k\eta) + J_{b+1/2}(k\eta')\,Y_{b+1/2}(k\eta)\Big\} & \eta>\eta'\\[1ex]
0 & \eta<\eta'.
\end{cases}
$$
Confidence: **high**. The subscript on the first $J$ is written compactly ("$J_{b\frac12}$") but
is read as $b+\tfrac12$ like all the others.

**R10** (MAIN 13 p.3). Green's function for the original tensor field $h$ (Step 4, "Domènech
writes a Green's function for the original tensor field $h$"):
$$
G_k[h] = \frac{a(\eta')}{a(\eta)}\,G_k[\chi],
$$
$$
G_k[h](\eta,\eta') = \begin{cases}
\dfrac{\pi}{2}\,(\eta')^{3/2+b}\,\eta^{-1/2-b}\,\Big\{Y_{b+1/2}(k\eta)\,J_{b+1/2}(k\eta') - Y_{b+1/2}(k\eta')\,J_{b+1/2}(k\eta)\Big\} & \eta>\eta'\\[1ex]
0 & \eta<\eta'.
\end{cases}
$$
Closing remark: "Differs from Domènech et al. formula, but appears to match Adshead et al.
formula." Confidence: **high** (exponents $3/2+b$ and $-1/2-b$ clearly legible).

### 3.2 NUM 02 — tensor Green's function in redshift (4 pp., 12 September 2024)

**R11** (NUM 02 p.1). Restatement of R1 ("see 2022 calculation ⑬") together with the field
identification ("see 2022 Calculation ⑪"):
$$
G_k'' + \Big(k^2 - \frac{a''}{a}\Big)G_k = \delta(\eta-\eta'),\qquad h^s = \frac{1}{a}\chi^s,\quad s = \text{polarisation label},
$$
"where this is the Green's function for the $\chi$ field". Confidence: **high**.

**R12** (NUM 02 p.1). Change of variable (Step 2):
$$
dz = -(1+z)H\,dt = -(1+z)aH\,d\eta,\qquad \frac{d}{d\eta} = -(1+z)aH\frac{d}{dz}.
$$
Confidence: **high**.

**R13** (NUM 02 p.2). Intermediate form of the equation in $z$ (after the red sign correction, see
§5; the ${}^{\prime\prime}$-term identity $a''/a = d(a^2H)/dt$ is used):
$$
(1+z)^2(aH)^2\frac{d^2G_k}{dz^2} + (1+z)(aH)^2\frac{dG_k}{dz}
- a\big(aH^2 + a\dot H\big)(1+z)\frac{dG_k}{dz}
+ \Big\{k^2 - \big(2a^2H^2 + a^2\dot H\big)\Big\}G_k = \delta(\eta-\eta').
$$
Confidence: **medium** — the sign of the third term is the subject of two red annotations
(original "+", red circled "$-$" labelled "sign error" on the first version; a circled "$\oplus$"
with a bar over it on the second version). The sign given here ($-$) is the one consistent with
the "$\epsilon$ overall" annotation on p.3 and with R14. See §5.

**Author sign-off (2026-09-07):** minus sign confirmed (Tier 2.6); see §0.2.

**R14** (NUM 02 p.2–3). Transformation of the source: "the $\delta$-function transforms like a
density", $\int d\eta\, f(\eta)\,\delta(\eta-\eta') = \int dz\, f(z)\,\delta(z-z')$ where $\eta'$ and
$z'$ label the same source time; hence $\frac{d\eta}{dz}\delta(\eta-\eta') = \delta(z-z')$ and
$$
\delta(\eta-\eta') = \delta(z-z')\,\frac{dz}{d\eta} = -(1+z)aH\,\delta(z-z').
$$
Confidence: **high**. Written without an absolute value on $dz/d\eta$ (Q3).

**R15** (NUM 02 p.3). Green's-function equation in redshift, general background ("Hence"; after
the red corrections that replace the friction coefficient $(2-\epsilon)$ by $\epsilon$):
$$
\frac{d^2G_k}{dz^2} + \frac{\epsilon}{1+z}\frac{dG_k}{dz}
+ \Big\{\frac{k^2}{a^2H^2(1+z)^2} - \frac{2-\epsilon}{(1+z)^2}\Big\}G_k
= -\frac{1}{1+z}\frac{1}{aH}\,\delta(z-z').
$$
Confidence: **high** for the final form; the intermediate line above it reads
$(1+z)^2(aH)^2 G_{zz} + (1+z)(aH)^2\{1 \oplus 1 \ominus \epsilon\}G_z + \{k^2 - a^2H^2(2-\epsilon)\}G = -(1+z)aH\,\delta(z-z')$
with red bars/plus signs over the braced coefficient and the note "$\epsilon$ overall".

**R16** (NUM 02 p.3). Rewriting with explicit $a_0$:
$$
\frac{k^2}{a^2H^2} = \frac{k^2}{a_0^2H^2}\frac{a_0^2}{a^2} = (1+z)^2\frac{k^2}{a_0^2}\frac{1}{H^2} = (1+z)^2\frac{k_{\rm phys}^2}{H^2},
\qquad
\frac{1}{aH} = \frac{1}{a_0}\frac{a_0}{a}\frac{1}{H} = \frac{1+z}{a_0}\frac{1}{H}.
$$
Defines $k_{\rm phys} = k/a_0$. Confidence: **high**.

**R17** (NUM 02 p.3). **Green's-function equation in redshift, final form**:
$$
\frac{d^2G_k}{dz^2} + \frac{\epsilon}{1+z}\frac{dG_k}{dz}
+ \Big\{\frac{k_{\rm phys}^2}{H^2} - \frac{2-\epsilon}{(1+z)^2}\Big\}G_k
= -\frac{1}{a_0 H}\,\delta(z-z').
$$
Followed by the remark on the $a_0^{-1}$ quoted in §2.4. Confidence: **high** (the friction
coefficient "$2-\epsilon$" is struck and replaced by "$\epsilon$" in red).

**R18** (NUM 02 p.4). Boundary conditions for the causal Green's function (Step 3):
$$
\text{Continuity:}\quad G_k(z',z') = 0,\qquad
\text{Jump:}\quad \Big[\frac{dG_k}{dz}\Big]_{z'-\epsilon}^{z'+\epsilon} = -\frac{1}{a_0H(z')}.
$$
Confidence: **high**. Here $\epsilon$ is an infinitesimal (Q5). The page ends here; no explicit
statement of which side of $z'$ the Green's function vanishes on (Q7).

**Author note (2026-09-07):** the minus sign in R17–R18 is a convention paired with the orientation of
the $z'$ measure, not a missing absolute value (§0 item 3); Q3 is closed. The code does **not**
implement R18. It implements the unit-jump condition $dG/dz|_{z=z'} = +1$ of spec 03 R25, so
$G_{\rm code} = -a_0H(z')\,G_k^{\rm (R9)}$ (§0 item 2). Support in $z$: $G_{\rm code} = 0$ for $z > z'$ (Q7).

### 3.3 NUM 05 — WKB(J) approximation for Green's functions (8 pp., 26 Sep 2024)

**R19** (NUM 05 p.1). Homogeneous part of the Green's-function equation (Step 1):
$$
\frac{d^2G_k}{dz^2} + \frac{\epsilon}{1+z}\frac{dG_k}{dz} + \Big(\frac{k^2}{H^2} + \frac{\epsilon-2}{(1+z)^2}\Big)G_k = 0.
$$
Same as R17 with $k \to k_{\rm phys}$ implied (Q6). Confidence: **high**.

**R20** (NUM 05 p.1). Friction elimination: set $G_k = f\Delta$ and choose $2f'/f + \epsilon/(1+z) = 0$:
$$
\frac{d\ln f}{dz} = -\frac12\frac{\epsilon}{1+z},\qquad
f = f_0\exp\Big(-\frac12\int_{z_0}^{z}\frac{\epsilon}{1+z'}\,dz'\Big),
$$
so that $\Delta'' + \omega_{\rm eff}^2\Delta = 0$ with
$$
\omega_{\rm eff}^2 = \frac{f''}{f} + \frac{\epsilon}{1+z}\frac{f'}{f} + \frac{k^2}{H^2} + \frac{\epsilon-2}{(1+z)^2}.
$$
Confidence: **high**.

**R21** (NUM 05 p.2). Slow-roll relations used to evaluate $\omega_{\rm eff}^2$:
$$
\frac{f'}{f} = -\frac12\frac{\epsilon}{1+z},\qquad
\frac{f''}{f} - \frac{f'}{f}\frac{f'}{f} = -\frac12\frac{\epsilon'}{1+z} + \frac12\frac{\epsilon}{(1+z)^2},
$$
$$
\epsilon' \equiv \frac{d\epsilon}{dz} = -\frac{1}{1+z}\frac{1}{H}\frac{d\epsilon}{dt} = -\frac{1}{1+z}\,\epsilon\eta
\quad\text{if}\quad \eta = \frac{d\ln\epsilon}{dN} = \frac{1}{\epsilon}\frac{d\epsilon}{H\,dt},
$$
$$
\epsilon = -\frac{\dot H}{H^2} = -\frac{1}{H^2}(-1)(1+z)H\frac{dH}{dz} = (1+z)\frac{d\ln H}{dz},
\qquad
\epsilon' = \frac{d\ln H}{dz} + (1+z)\frac{d^2\ln H}{dz^2} = \frac{\epsilon}{1+z} + (1+z)\frac{d^2\ln H}{dz^2}.
$$
Confidence: **high** for all but the product "$\epsilon\eta$", which is **medium**: it could be
read as a subscripted $\epsilon_\eta$, but the following "if $\eta = d\ln\epsilon/dN$" and the
algebra of R22 support the product of $\epsilon$ with a second slow-roll parameter $\eta$ (Q4).

**R22** (NUM 05 p.2). **Effective frequency for the Green's function**:
$$
\omega_{\rm eff}^2 = -\frac12\frac{\epsilon'}{1+z} + \frac12\frac{\epsilon}{(1+z)^2} + \frac14\frac{\epsilon^2}{(1+z)^2} - \frac12\frac{\epsilon^2}{(1+z)^2} + \frac{k^2}{H^2} + \frac{\epsilon-2}{(1+z)^2}
$$
$$
= \frac{k^2}{H^2} - \frac12\frac{\epsilon'}{1+z} + \frac{\tfrac32\epsilon - \tfrac14\epsilon^2 - 2}{(1+z)^2}
= \frac{k^2}{H^2} + \frac{\tfrac32\epsilon + \tfrac12\epsilon\eta - \tfrac14\epsilon^2 - 2}{(1+z)^2}.
$$
Confidence: **high** (the two forms differ only by substituting $\epsilon' = -\epsilon\eta/(1+z)$).

**R23** (NUM 05 p.3). WKB ansatz (Step 2): assume $\omega_{\rm eff}^2 \gg 1$, $\Delta \sim A(z)\exp(i\Theta(z))$
with $A$ slowly varying and $\Theta$ rapidly varying. Balancing the imaginary terms
$2A'\Theta' + A\Theta'' = 0$ gives $A = A_0(\Theta')^{-1/2}$; balancing the largest real terms
$\Theta'^2 = \omega_{\rm eff}^2$ gives $\Theta' = \omega_{\rm eff}$, hence
$$
A = \frac{A_0}{\omega_{\rm eff}^{1/2}},\qquad
\Theta = \int_{z_0}^{z}dz'\,\omega_{\rm eff} = \int_{z_0}^{z}dz'\,\Big(\frac{k^2}{H^2} + \frac{3\epsilon/2 + \epsilon\eta/2 - \epsilon^2/4 - 2}{(1+z')^2}\Big)^{1/2}.
$$
Confidence: **high** (the last exponent $^{1/2}$ runs off the right margin but is visible).

**R24** (NUM 05 p.4). $\Lambda$CDM background (Step 3), with $3H^2M_P^2 = \rho_{m,0}(1+z)^3 + \rho_{r,0}(1+z)^4 + \rho_\Lambda$:
$$
\frac{d\ln H}{dz} = \frac{3\Omega_{m,0}(1+z)^2 + 4\Omega_{r,0}(1+z)^3}{2\Omega_{m,0}(1+z)^3 + 2\Omega_{r,0}(1+z)^4 + 2\Omega_\Lambda},
$$
$$
\frac{d^2\ln H}{dz^2} = \frac{6\Omega_m(1+z) + 12\Omega_r(1+z)^2}{2\{\Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_\Lambda\}} - 2\Big(\frac{d\ln H}{dz}\Big)^2.
$$
(The subscript "$,0$" is dropped on $\Omega$ after the first line.) Confidence: **high**.

**R25** (NUM 05 p.5). WKB solutions anchored at the matching point $\tilde z$ (Step 4, "we want
to match the numerically computed $G$, $G'$ at $\tilde z$"):
$$
G^{(1)} = \frac{1}{\omega_{\rm eff}^{1/2}(z)}\exp\Big(-\frac12\int_{\tilde z}^{z}dz'\,\frac{\epsilon(z')}{1+z'}\Big)\cos\Big(\int_{\tilde z}^{z}dz'\,\omega_{\rm eff}(z')\Big),
$$
$$
G^{(2)} = \frac{1}{\omega_{\rm eff}^{1/2}(z)}\exp\Big(-\frac12\int_{\tilde z}^{z}dz'\,\frac{\epsilon(z')}{1+z'}\Big)\sin\Big(\int_{\tilde z}^{z}dz'\,\omega_{\rm eff}(z')\Big),
$$
with the phase $\Theta(z) = \int_{\tilde z}^{z}\omega_{\rm eff}$, $\Theta(\tilde z) = 0$. The
required solution is $\alpha G^{(1)} + \beta G^{(2)}$. Confidence: **high** (the matching point
was first written $z^*$ and over-written as $\tilde z$; see §5).

**R26** (NUM 05 p.5–6). **Matching coefficients** from $G(\tilde z)$, $G'(\tilde z)$:
$$
\alpha = \omega_{\rm eff}^{1/2}(\tilde z)\,G(\tilde z),
$$
$$
\beta\,\omega_{\rm eff}^{1/2}(\tilde z) = G'(\tilde z) + \frac12 G(\tilde z)\Big(\frac{\epsilon(\tilde z)}{1+\tilde z} + \frac{\omega'_{\rm eff}(\tilde z)}{\omega_{\rm eff}(\tilde z)}\Big),
\qquad
\beta = \frac{G'(\tilde z)}{\omega_{\rm eff}^{1/2}(\tilde z)} + \frac{G(\tilde z)}{2\omega_{\rm eff}^{1/2}(\tilde z)}\Big(\frac{\epsilon(\tilde z)}{1+\tilde z} + \frac{\omega'_{\rm eff}(\tilde z)}{\omega_{\rm eff}(\tilde z)}\Big).
$$
Confidence: **high**.

**R27** (NUM 05 p.6). Logarithmic derivative of $\omega_{\rm eff}$, **as originally written**
(superseded by R31 of NUM 10; see §5 Corrections). Starting from
$\omega_{\rm eff}^2 = k^2/H^2 - \tfrac{\epsilon'/2}{1+z} + \tfrac{3\epsilon/2 - \epsilon^2/4 - 2}{(1+z)^2}$:
$$
2\frac{d\ln\omega_{\rm eff}}{dz} = \frac{1}{\omega_{\rm eff}^2}\Big[-\frac{\epsilon''/2}{1+z} + \frac{\epsilon'/2}{(1+z)^2} + \frac{3\epsilon'/2 - \epsilon\epsilon'/2}{(1+z)^2} - 2\,\frac{3\epsilon/2 - \epsilon^2/4 - 2}{(1+z)^3}\Big],
$$
$$
\frac{\omega'_{\rm eff}}{\omega_{\rm eff}} = \frac{d\ln\omega_{\rm eff}}{dz} = \frac{1}{2\omega_{\rm eff}^2}\Big[-\frac{\epsilon''/2}{1+z} + \frac{2\epsilon' - \epsilon\epsilon'/2}{(1+z)^2} - \frac{3\epsilon - \epsilon^2/2 - 4}{(1+z)^3}\Big].
$$
Confidence: **high** as a reading (the first line runs slightly off the right margin, but the
consolidated second line is complete). This expression contains no term from the $z$-dependence
of $k^2/H^2$; that is the omission NUM 10 repairs.

**R28** (NUM 05 p.6). Second derivative of $\epsilon$:
$$
\epsilon' = \frac{d\ln H}{dz} + (1+z)\frac{d^2\ln H}{dz^2},\qquad
\epsilon'' = 2\frac{d^2\ln H}{dz^2} + (1+z)\frac{d^3\ln H}{dz^3}.
$$
Confidence: **high**.

**R29** (NUM 05 p.7). Third derivative of $\ln H$ in $\Lambda$CDM ("finally"):
$$
\frac{d^3\ln H}{dz^3} = \frac{6\Omega_m + 24\Omega_r(1+z)}{2\big(\Omega_m(1+z)^3 + \Omega_r(1+z)^4 + \Omega_\Lambda\big)} - 6\,\frac{d\ln H}{dz}\frac{d^2\ln H}{dz^2} - 4\Big(\frac{d\ln H}{dz}\Big)^3.
$$
Confidence: **high**.

**R30** (NUM 05 p.8). Simplified friction factor and **final WKB solutions** ("NOTE — to simplify
the friction term"): since $\epsilon = (1+z)\,d\ln H/dz$,
$$
f = \exp\Big(-\frac12\int_{\tilde z}^{z}dz'\,\frac{\epsilon}{1+z'}\Big) = \exp\Big(-\frac12\int_{\tilde z}^{z}d\ln H\Big) = \Big(\frac{H(\tilde z)}{H(z)}\Big)^{1/2},
$$
$$
G^{(1)} = \frac{1}{\omega_{\rm eff}^{1/2}(z)}\Big(\frac{\tilde H}{H(z)}\Big)^{1/2}\cos\int_{\tilde z}^{z}dz'\,\omega_{\rm eff}(z'),
\qquad
G^{(2)} = \frac{1}{\omega_{\rm eff}^{1/2}(z)}\Big(\frac{\tilde H}{H(z)}\Big)^{1/2}\sin\int_{\tilde z}^{z}dz'\,\omega_{\rm eff}(z'),
$$
with $\tilde H \equiv H(\tilde z)$. Confidence: **high**.

### 3.4 NUM 10 — fix calculation for $d\omega_{\rm eff}/dz$ for $G_k$ (1 p., 8 Dec 2024)

**R31** (NUM 10 p.1). **Corrected derivative of $\omega_{\rm eff}$** (replaces R27). From
$$
\omega_{\rm eff}^2 = \frac{k^2}{H^2} - \frac{\epsilon'/2}{1+z} + \frac{3\epsilon/2 - \epsilon^2/4 - 2}{(1+z)^2},
$$
$$
2\omega_{\rm eff}\frac{d\omega_{\rm eff}}{dz}
= -\frac{2}{H^2}\frac{\epsilon}{1+z}k^2 - \frac{1}{1+z}\frac{\epsilon''}{2} + \frac{1}{(1+z)^2}\frac{\epsilon'}{2}
+ \frac{1}{(1+z)^2}\Big\{\frac{3\epsilon'}{2} - \frac12\epsilon\epsilon'\Big\}
- \frac{2}{(1+z)^3}\Big\{\frac32\epsilon - \frac14\epsilon^2 - 2\Big\}
$$
$$
= \frac{1}{1+z}\Big\{-\frac{\epsilon''}{2} - 2\epsilon\frac{k^2}{H^2}\Big\}
+ \frac{1}{(1+z)^2}\Big\{2\epsilon' - \frac12\epsilon\epsilon'\Big\}
- \frac{2}{(1+z)^3}\Big\{\frac32\epsilon - \frac14\epsilon^2 - 2\Big\}.
$$
Confidence: **high**. Relative to R27 the only change is the new first term
$-2\epsilon k^2/\big(H^2(1+z)\big)$, i.e. the derivative of $k^2/H^2$ via $d\ln H/dz = \epsilon/(1+z)$.
The page does not restate $\omega'_{\rm eff}/\omega_{\rm eff}$; dividing by $2\omega_{\rm eff}^2$ is
left to the reader. See §5.

### 3.5 NUM 11 — WKB repeat and matching conditions (10 pp., 23 Mar 2025)

Headed "Revisit WKB analysis for Green's/transfer functions". Steps 1–5 (p.1–6) repeat and extend
the Green's-function WKB of NUM 05; Steps 6–8 (p.6–10) repeat the analysis for the transfer
function (Group 1 subject matter; transcribed here because the document is assigned to Group 2).

**R32** (NUM 11 p.1–2). Repeat of R19–R22 (Step 1). Homogeneous equation, $G_k = f\Delta$,
$d\ln f/dz = -\tfrac12\epsilon/(1+z)$, $f = f_0\exp\big(-\tfrac12\int_{z_0}^{z}\epsilon(z')/(1+z')\,dz'\big)$,
$\Delta'' + \omega_{\rm eff}^2\Delta = 0$, and
$$
\frac{f''}{f} = \frac{\epsilon^2/4}{(1+z)^2} - \frac{\epsilon'/2}{1+z} + \frac{\epsilon/2}{(1+z)^2},
\qquad
\omega_{\rm eff}^2 = \frac{k^2}{H^2} - \frac{\epsilon'/2}{1+z} + \frac{-\epsilon^2/4 + 3\epsilon/2 - 2}{(1+z)^2},
$$
annotated "↪ matches earlier calculation ⑤". Confidence: **high**.

**R33** (NUM 11 p.2–3). Liouville-Green representation (Step 2): $\Delta \approx A(z)e^{i\Theta(z)}$
assuming $\omega_{\rm eff}^2 \gg 1$; in
$e^{i\Theta}\{A'' - A(\Theta')^2 + \omega_{\rm eff}^2A + i[2A'\Theta' + A\Theta'']\} = 0$ the term
$A''$ is "small", $-A\Theta'^2 + \omega_{\rm eff}^2 A$ "large", and the imaginary bracket must
"vanish separately". Hence
$$
\text{①}\ 2A'\Theta' + A\Theta'' = 0 \Rightarrow \frac{d\ln A}{dz} = -\frac12\frac{d\ln\Theta'}{dz} \Rightarrow A\propto(\Theta')^{-1/2},
\qquad
\text{②}\ -(\Theta')^2 + \omega_{\rm eff}^2 = 0 \Rightarrow \Theta' = \pm\omega_{\rm eff},
\qquad
\Theta = \int_{z_0}^{z}\pm\,\omega_{\rm eff}(z')\,dz'.
$$
Confidence: **high**. Unlike NUM 05 the sign $\pm$ of $\Theta'$ is kept explicit here.

**R34** (NUM 11 p.3). Two independent Liouville-Green modes with the matching point $z^*$ (Step 3,
"match prescribed values of $G_k$ and $dG_k/dz$ at $z=z^*$"); here $\alpha,\beta$ are absorbed
into the mode definitions:
$$
G^{(1)} = \frac{\alpha}{\omega_{\rm eff}^{1/2}}\exp\Big(-\frac12\int_{z^*}^{z}\frac{\epsilon(z')}{1+z'}dz'\Big)\cos\int_{z^*}^{z}\omega_{\rm eff}(z')\,dz',
\qquad
G^{(2)} = \frac{\beta}{\omega_{\rm eff}^{1/2}}\exp\Big(-\frac12\int_{z^*}^{z}\frac{\epsilon(z')}{1+z'}dz'\Big)\sin\int_{z^*}^{z}\omega_{\rm eff}(z')\,dz'.
$$
Confidence: **high**.

**R35** (NUM 11 p.4). Friction factor ("if desired"): with $\epsilon = (1+z)\,d\ln H/dz$,
$\int_{z^*}^{z}\epsilon(z')/(1+z')\,dz' = \ln H(z) - \ln H^*$, so
$$
\exp\Big(-\frac12\int_{z^*}^{z}\frac{\epsilon(z')}{1+z'}dz'\Big) = \Big(\frac{H(z)}{H^*}\Big)^{-1/2} = \Big(\frac{H^*}{H(z)}\Big)^{1/2}.
$$
Confidence: **high**.

**R36** (NUM 11 p.4). **Matching coefficients at $z^*$** (① from $G^*$, ② from $(dG/dz)^*$; the
$\sin(0)$ terms are marked "→ 0" in red):
$$
\alpha = G^*\,\omega_{\rm eff}^{*\,1/2},
\qquad
\beta\,\omega_{\rm eff}^{*\,1/2} = G'^* + \frac{\alpha}{2\omega_{\rm eff}^{*\,1/2}}\Big(\frac{\omega'_{\rm eff}}{\omega_{\rm eff}} + \frac{\epsilon}{1+z}\Big)^*,
\qquad
\beta = \frac{G'^*}{\omega_{\rm eff}^{*\,1/2}} + \frac{G^*}{2\omega_{\rm eff}^{*\,1/2}}\Big(\frac{\omega'_{\rm eff}}{\omega_{\rm eff}} + \frac{\epsilon}{1+z}\Big)^*.
$$
Agrees with R26 (with $\tilde z \to z^*$). Confidence: **high**.

**R37** (NUM 11 p.5). Phase shift to a pure-sine form (Step 4, "this helps match to solutions
with a subhorizon source time"). Writing
$$
B\sin\big(\Theta(z) + \Delta\Theta\big) = \frac{\alpha}{\omega_{\rm eff}^{*\,1/2}}\Big(\frac{H^*}{H}\Big)^{1/2}\cos\Theta(z) + \frac{\beta}{\omega_{\rm eff}^{*\,1/2}}\Big(\frac{H^*}{H}\Big)^{1/2}\sin\Theta(z),
$$
$$
B\cos\Delta\Theta = \frac{\beta}{\omega_{\rm eff}^{*\,1/2}}\Big(\frac{H^*}{H}\Big)^{1/2},\qquad
B\sin\Delta\Theta = \frac{\alpha}{\omega_{\rm eff}^{*\,1/2}}\Big(\frac{H^*}{H}\Big)^{1/2},
$$
$$
\Rightarrow\quad B^2 = \frac{\alpha^2 + \beta^2}{\omega_{\rm eff}^*}\frac{H^*}{H},\qquad \tan\Delta\Theta = \frac{\alpha}{\beta}\ \text{(is time independent)}.
$$
Confidence: **medium** — every $\omega_{\rm eff}$ on this page carries a star (evaluated at
$z^*$), whereas the modes of R34 have $\omega_{\rm eff}^{1/2}(z)$ and the analogous transfer-function
result R46 has no star. See Open questions Q9.

**R38** (NUM 11 p.5–6). Representation for accurate evaluation of a large phase (Step 5):
$$
u = z_i - z,\qquad \Theta(z) = \Theta_i + \omega_{\rm eff}^{i}\,(1+u)\,Q(u),
$$
"then $Q(u)$ will be fairly close to unity when $u\gg1$, if $\omega_{\rm eff}$ does not evolve
much, because $\Theta(z)-\Theta_i$ should grow like $u$". With $du = -dz$:
$$
\frac{d\Theta}{dz} = \frac{d\Theta}{-du} = -\Big\{\omega_{\rm eff}^{i}Q(u) + \omega_{\rm eff}^{i}(1+u)\frac{dQ}{du}\Big\} = \omega_{\rm eff},
\qquad
\frac{dQ}{du} = \frac{-\omega_{\rm eff} - \omega_{\rm eff}^{i}Q}{\omega_{\rm eff}^{i}(1+u)} = -\frac{\omega_{\rm eff}}{\omega_{\rm eff}^{i}}\frac{1}{1+u} - \frac{Q}{1+u}.
$$
$\omega_{\rm eff}^{i} \equiv \omega_{\rm eff}(z_i)$, $\Theta_i \equiv \Theta(z_i)$ (superscript/subscript
$i$, not an imaginary unit). Confidence: **high** as a reading; see Q10 on the sign.

**R39** (NUM 11 p.6). Evolution equation for the transfer function (Step 6, "Now repeat the
analysis for the transfer function"):
$$
\frac{d^2\phi}{dz^2} + \frac{1}{1+z}\big\{\epsilon - 3(1+c_s^2)\big\}\frac{d\phi}{dz}
+ \frac{1}{(1+z)^2}\big\{3(1+c_s^2) - 2\epsilon\big\}\phi + \frac{k^2}{a^2H^2}\frac{c_s^2}{(1+z)^2}\phi = 0.
$$
Quoted, not derived, in this document. Confidence: **high**.

**R40** (NUM 11 p.7). Friction eliminator for the transfer function, $\phi = P\vartheta$ with
$2P'/P + \tfrac{1}{1+z}[\epsilon - 3(1+c_s^2)] = 0$:
$$
\frac{d\ln P}{dz} = -\frac12\frac{1}{1+z}\big[\epsilon - 3(1+c_s^2)\big] = -\frac12\frac{d\ln H}{dz} + \frac32\frac{1+c_s^2}{1+z},
$$
$$
\ln P = -\frac12\ln H + \frac12\ln H^* + \frac32\int_{z^*}^{z}\frac{1+c_s^2(z')}{1+z'}dz',
\qquad
P \propto \Big(\frac{H^*}{H}\Big)^{1/2}\exp\Big\{\frac32\int_{z^*}^{z}\frac{1+c_s^2(z')}{1+z'}dz'\Big\}.
$$
Confidence: **high**.

**R41** (NUM 11 p.7). Second derivative of $P$:
$$
\frac{P''}{P} - \frac{P'}{P}\frac{P'}{P} = -\frac{\epsilon'/2}{1+z} + \frac{\epsilon/2}{(1+z)^2} + \frac{3c_sc_s'}{1+z} - \frac{\tfrac32(1+c_s^2)}{(1+z)^2},
$$
$$
\frac{P''}{P} = \frac{3c_sc_s' - \epsilon'/2}{1+z} + \frac{\epsilon/2 - \tfrac32(1+c_s^2)}{(1+z)^2} + \frac{\tfrac14\big(\epsilon^2 - 6\epsilon[1+c_s^2] + 9[1+c_s^2]^2\big)}{(1+z)^2}.
$$
Confidence: **high**.

**R42** (NUM 11 p.7). Scale-factor normalisation in the transfer-function mass term:
$$
\frac{a_0}{a} = 1+z,\quad (1+z)a = a_0\quad\Rightarrow\quad \frac{1}{(1+z)^2}\frac{k^2c_s^2}{a^2H^2} = \frac{k^2c_s^2}{a_0^2H^2}.
$$
Confidence: **high**.

**R43** (NUM 11 p.8). **Effective frequency for the transfer function**, $\vartheta'' + \omega_{\rm eff}^2\vartheta = 0$ with
$$
\omega_{\rm eff}^2 = \frac{k^2c_s^2}{a_0^2H^2} + \frac{3c_sc_s' - \epsilon'/2}{1+z} + \frac{\tfrac32(1+c_s^2)(1+\epsilon) - \tfrac{\epsilon}{2}\big(3 + \tfrac{\epsilon}{2}\big) - \tfrac94(1+c_s^2)^2}{(1+z)^2}.
$$
The preceding ("i.e.") line writes the last numerator as
$\tfrac32(1+c_s^2) - \tfrac32\epsilon - \tfrac14\epsilon^2 + \tfrac32\epsilon(1+c_s^2) - \tfrac94(1+c_s^2)^2$, which is the
same expression expanded. Confidence: **high** (one struck digit in "$-\tfrac32\epsilon$", see §5).
Note this $\omega_{\rm eff}$ is a different function from the Green's-function $\omega_{\rm eff}$ of R22/R32
despite sharing the symbol.

**R44** (NUM 11 p.8). Liouville-Green solutions for the transfer function (Step 7, "apply
boundary conditions at $z^*$ to match to superhorizon calculation"):
$$
T_1 = \frac{\alpha}{\omega_{\rm eff}^{1/2}}\exp\Big(\int_{z^*}^{z}\frac{-\epsilon/2 + \tfrac32(1+c_s^2)}{1+z'}dz'\Big)\cos\int_{z^*}^{z}\omega_{\rm eff}(z')\,dz',
\qquad
T_2 = \frac{\beta}{\omega_{\rm eff}^{1/2}}\exp\Big(\int_{z^*}^{z}\frac{-\epsilon/2 + \tfrac32(1+c_s^2)}{1+z'}dz'\Big)\sin\int_{z^*}^{z}\omega_{\rm eff}(z')\,dz'.
$$
Confidence: **high** (the exponent of $\omega_{\rm eff}$ in $T_1$ has struck characters before the
final "$1/2$"; see §5).

**R45** (NUM 11 p.9). **Matching coefficients for the transfer function** at $z^*$ (again with
$\sin(0)\to0$ marked in red):
$$
\alpha = T^*\,\omega_{\rm eff}^{*\,1/2},
\qquad
\beta\,\omega_{\rm eff}^{*\,1/2} = T'^* + \frac{\alpha}{2\omega_{\rm eff}^{*\,1/2}}\Big\{\frac{\omega'_{\rm eff}}{\omega_{\rm eff}} + \frac{1}{1+z}\big[\epsilon - 3(1+c_s^2)\big]\Big\}^*,
$$
$$
\beta = \frac{T'^*}{\omega_{\rm eff}^{*\,1/2}} + \frac{T^*}{2\omega_{\rm eff}^{*\,1/2}}\Big\{\frac{\omega'_{\rm eff}}{\omega_{\rm eff}} + \frac{1}{1+z}\big[\epsilon - 3(1+c_s^2)\big]\Big\}^*.
$$
Confidence: **high**.

**R46** (NUM 11 p.9–10). Optional phase shift for the transfer function (Step 8, "to remove the
cos"): writing the solution as $C(z)\{\sin\Theta(z)\cos\Delta\Theta + \cos\Theta(z)\sin\Delta\Theta\}$,
$$
C(z)\cos\Delta\Theta = \frac{\beta}{\omega_{\rm eff}^{1/2}}\Big(\frac{H^*}{H}\Big)^{1/2}\exp\Big(\frac32\int_{z^*}^{z}\frac{1+c_s^2}{1+z'}dz'\Big),
\qquad
C(z)\sin\Delta\Theta = \frac{\alpha}{\omega_{\rm eff}^{1/2}}\Big(\frac{H^*}{H}\Big)^{1/2}\exp\Big(\frac32\int_{z^*}^{z}\frac{1+c_s^2}{1+z'}dz'\Big),
$$
$$
C^2(z) = \frac{\alpha^2 + \beta^2}{\omega_{\rm eff}}\frac{H^*}{H}\exp\Big(3\int_{z^*}^{z}\frac{1+c_s^2}{1+z'}dz'\Big),
\qquad
\tan\Delta\Theta = \frac{\alpha}{\beta}\ \text{(independent of time, as it should be)}.
$$
Confidence: **high**. Here $\omega_{\rm eff}$ carries no star (contrast R37).

---

## 4. Checks

- **NUM 11 rechecks NUM 05** (Green's-function WKB). NUM 11 p.2 states of $\omega_{\rm eff}^2$ (R32):
  "matches earlier calculation ⑤" — agrees with R22. The matching coefficients R36 agree with R26
  term by term (with $\tilde z \to z^*$). The friction factor R35 agrees with R30. No discrepancy
  is noted by the author.
- **NUM 11 vs NUM 05 on the sign of $\Theta'$**: NUM 05 p.3 takes $\Theta' = \omega_{\rm eff}$;
  NUM 11 p.3 writes $\Theta' = \pm\omega_{\rm eff}$ and later (p.6) uses $d\Theta/dz = \omega_{\rm eff}$.
  Not a contradiction, but see Q10.
- **NUM 10 vs NUM 05**: NUM 10 is a correction, not a recheck; see §5.
- **NUM 02 vs MAIN 13**: NUM 02 p.1 quotes R1 verbatim from MAIN 13 without recomputation.
- **NUM 11 transfer-function part vs Group 1**: R39 and R43 should be compared against
  `01-transfer-function.md` (NUM 08/09) by the orchestrator; no check is made within NUM 11 itself.

---

## 5. Corrections and cross-outs

Listed by document. "Later steps use" means the version the same document carries forward.

### MAIN 13
1. p.1, R4: in the decaying solution "$\beta\sqrt\eta\,\partial Y_{b+1/2}(k\eta)$" there is a small
   extra glyph (read as $\partial$ or a struck letter) between $\sqrt\eta$ and $Y$. Transcribed
   without it; later steps use $\beta\sqrt{k\eta}\,Y_{b+1/2}$.
2. p.2, R5: in the jump equation a symbol before "$=1$" is struck ("$= \cancel{3}\ 1$"). The
   second brace is annotated in red "0 by continuity at $\eta=\eta'$"; later steps use the value 1.
3. p.2, R6: a struck symbol precedes the fraction on the right of the last line ("$\cancel{\pm}$").
4. p.3, R10: in the second $J$ of the $G_k[h]$ formula a subscript has been over-written; read as
   $J_{b+1/2}(k\eta)$ (consistent with the rest of the page).

### NUM 02
5. p.1: in the first substituted equation the term "$+\frac1a(1+z)aH\frac{d}{dz}$" is struck and
   replaced by "$-\frac1a\frac{d}{dt/a}\big(\frac{da}{dt/a}\big)$" (i.e. $a''/a$ expressed in cosmic
   time). Intermediate algebra; not transcribed as a result.
6. p.2, first equation (R13): the term "$+\,a\frac{d}{dt}(aH)\cdot(1+z)\frac{dG_k}{dz}$" carries a
   red circled "$-$" with an arrow and the note "sign error". **Original:** $+$. **Correction:** $-$.
7. p.2, second equation: the corresponding term "$\circledast\,a(aH^2 + a\dot H)(1+z)\frac{dG_k}{dz}$"
   has a red circled $\oplus$ with a bar drawn over it. Read as: the sign written in blue was
   corrected in red. Reading of the intended final sign: **medium** on its own, but the p.3 line
   (item 8) fixes it as $-$.
8. p.3, R15 intermediate line: the friction coefficient "$\{1 \oplus 1 \ominus \epsilon\}$" has red
   bars over the two blue signs, red "$-$" over the first and "$+$" over the second, and a brace
   labelled "$\epsilon$ overall". **Original:** $1 + (1 - \epsilon) = 2-\epsilon$. **Correction:**
   $1 - (1-\epsilon) = \epsilon$. Later steps use $\epsilon$.
9. p.3, twice (both displayed equations after "so" and "Hence"): friction coefficient
   "$\frac{2-\epsilon}{1+z}$" struck, "$\epsilon$" written above in red. **Original:** $2-\epsilon$.
   **Correction:** $\epsilon$. Later steps (NUM 05, 11) use $\epsilon/(1+z)$.
10. p.3: in the source "$\delta(\cancel{\eta}\,z-z')$" a stray $\eta$ is struck inside the
    $\delta$-function argument. Later steps use $\delta(z-z')$.
11. p.3: the "$(2-\epsilon)$" in the **mass** term $a^2H^2(2-\epsilon)$ is *not* struck; only the
    friction coefficient is corrected. Later steps use $(\epsilon-2)/(1+z)^2$ in the mass term.

### NUM 05
12. p.3: "$\frac{d\ln A}{d\cancel{\mathbb{E}}z}$" — an over-written character in the denominator;
    read as $dz$.
13. p.5: "Apply boundary conditions at $z = \cancel{z^*}\ \tilde z$" and "at $\cancel{z^*}\ \tilde z$"
    — the matching point is renamed from $z^*$ to $\tilde z$ (twice). Also "$G^{\cancel{*}\,(1)}$"
    — superscript over-written to $(1)$ — and "the numerically computed $\cancel{\ast}\,G, G'$".
    Later steps in NUM 05 use $\tilde z$; NUM 11 reverts to $z^*$.
14. p.6, R27 → **superseded by NUM 10 (R31)**. **Original (NUM 05):** bracket without any
    $k^2/H^2$-derivative term. **Correction (NUM 10):** additional term
    $-\frac{2}{H^2}\frac{\epsilon}{1+z}k^2$, i.e. $\frac{1}{1+z}\{-\frac{\epsilon''}{2} - 2\epsilon\frac{k^2}{H^2}\}$
    in place of $\frac{1}{1+z}\{-\frac{\epsilon''}{2}\}$; all other terms unchanged. Later work should
    use NUM 10. (NUM 10 does not itself refer to NUM 05 by number; the identification is by the
    file title "fix calculation for $d\omega_{\rm eff}/dz$ for $G_k$" and the identical starting
    expression.)

### NUM 10
15. No cross-outs on the page.

### NUM 11
16. p.2: "$= \cancel{\neq}\, e^{i\Theta}\{\dots\}$" — a stray struck symbol before $e^{i\Theta}$.
17. p.3, R33: "$\Theta = \int_{z_0}^{z}\pm\,\omega^{\cancel{2}}_{\rm eff}(z')\,dz'$" — a superscript on
    $\omega_{\rm eff}$ is struck; read as $\omega_{\rm eff}$ (no power).
18. p.4 and p.9: red arrows "→ 0" on every $\sin(0)$ term of the ② matching equations; a struck
    "$=0$" after "$\sin(0)$" in ① on p.4.
19. p.5: "$B\cos\Delta\cancel{z}\Theta$" — stray struck $z$.
20. p.7: page number written "⑦" with a second circled numeral struck beside it.
21. p.8, first displayed equation: "$\frac{\epsilon/2\ \cancel{\,7\,}\ -\tfrac32(1+c_s^2)}{(1+z)^2}$" — a
    struck character between the two terms (read as a slip for the minus sign); and in the "i.e."
    line "$-\tfrac32\cancel{2}\epsilon$" or similar — a struck digit next to $\tfrac32\epsilon$.
    Read as $-\tfrac32\epsilon$, consistent with the final R43.
22. p.8, Step 7 heading: "at $\cancel{\tilde z}\ z^*$" — $\tilde z$ struck, $z^*$ written above.
23. p.8, R44: "$\cancel{\vartheta_T}\ T_1 = \frac{\alpha}{\omega_{\rm eff}^{\cancel{1\,3/2}\,1/2}}$" — the
    solution label and the exponent are both over-written; read as $T_1$ and $1/2$.
24. p.9, R46: "$\frac{\alpha}{\omega_{\rm eff}^{1/2}}\ \cancel{\exp}\ \big(\frac{H^*}{H}\big)^{1/2}\exp(\dots)$"
    — a duplicated "exp" struck.

---

## 6. Open questions

**Q1 — NUM 11 date.** The first page is dated "23 Mar 2025"; the file name carries "2025-03-24".
Recorded as 23 March 2025 (page) in §1.

**Q2 — MAIN 13 p.1, $\mathcal H$ glyph.** "$\mathcal H = a'/a = (1+b)/\eta$": the symbol is a
Roman capital H with a leading curl, read as script $\mathcal H$ (conformal Hubble rate). It is
not used again. Reading: medium.

**Q3 — Sign of the redshift source (NUM 02 p.3–4).** The Jacobian is applied as
$\delta(\eta-\eta') = \delta(z-z')\,dz/d\eta$ with $dz/d\eta = -(1+z)aH < 0$ and **no absolute
value**, giving the negative source $-\frac{1}{a_0H}\delta(z-z')$ in R17 and the negative jump
$-1/(a_0H(z'))$ in R18. This is transcribed exactly as written. Whether the sign is intended
(it propagates directly into the normalisation of the numerically computed Green's function) is
for the author to confirm.
**Closed 2026-09-07:** a convention, not a slip (§0 item 3). It does not propagate into the code.

**Q4 — NUM 05 p.2, "$\epsilon\eta$".** Read as the product of $\epsilon$ with a second slow-roll
parameter $\eta \equiv d\ln\epsilon/dN$, because the text continues "if $\eta = d\ln\epsilon/dN = \epsilon^{-1}d\epsilon/(H\,dt)$"
and because only the product makes $\epsilon' = -\epsilon\eta/(1+z)$ consistent with the final
line of R22. Alternative reading: a single subscripted symbol $\epsilon_\eta$ defined as
$d\ln\epsilon/dN$, in which case the factor of $\epsilon$ in $\epsilon'$ would be missing on the
page. In either reading, this $\eta$ is **not** conformal time.

**Q5 — Double use of $\epsilon$ (NUM 02 p.4).** The jump condition uses $\epsilon$ as an
infinitesimal offset $z'\pm\epsilon$ on the same page-run that uses $\epsilon$ as the slow-roll
parameter (p.3). No ambiguity in meaning, but the spec reader should not conflate them.

**Q6 — $k$ vs $k_{\rm phys}$ in NUM 05, 10, 11.** NUM 02 defines $k_{\rm phys} = k/a_0$ and
writes the mass term as $k_{\rm phys}^2/H^2$. NUM 05, NUM 10 and the Green's-function part of
NUM 11 write $k^2/H^2$ with no $a_0$ and no "phys" subscript. Inferred: $k$ there stands for
$k/a_0$ (or $a_0 = 1$). Not stated on any page. In contrast the transfer-function part of NUM 11
(p.7–8) keeps $a_0$ explicit ($k^2c_s^2/(a_0^2H^2)$).
**Closed 2026-09-07:** $k$ means $k_{\rm phys}=k/a_0$ (§0.2).

**Q7 — Support of the retarded Green's function in redshift.** MAIN 13 states $G_k = 0$ for
$\eta<\eta'$. NUM 02 p.4 gives only continuity and jump at $z=z'$ and stops; it never states
that $G_k(z,z') = 0$ for $z>z'$ (the redshift counterpart). Inferred, not written.

**Q8 — MAIN 13 p.2→p.3, factor $Y_{b+1/2}(k\eta')$ in $\alpha$.** The last line of p.2 (R6) has
$Y_{b+1/2}(k\eta')$ apparently in the numerator on the right-hand side; the first line of p.3
(R8) has $1/(k(k\eta')^{1/2})$ with no $Y$. The final result R9 is what one obtains by keeping the
$Y$ factor (the $Y_{b+1/2}(k\eta')$ from $\alpha$ cancels the $1/Y_{b+1/2}(k\eta')$ in $\beta$), so
R9 does not inherit the omission, but the intermediate line R8 as written is inconsistent with
R6. Flagged for the author; no change made.
**Closed 2026-09-07:** the "$\alpha$" is $\propto$; the $Y$ factor is deliberately suppressed and restored later (§0.2).

**Q9 — NUM 11 p.5, starred $\omega_{\rm eff}$ in the phase-shifted amplitude (R37).** All four
occurrences of $\omega_{\rm eff}$ on p.5 are written $\omega_{\rm eff}^{*\,1/2}$ (or $\omega_{\rm eff}^*$
in $B^2$), i.e. evaluated at the matching point, so $B^2 = (\alpha^2+\beta^2)H^*/(\omega_{\rm eff}^*H)$
carries no $1/\omega_{\rm eff}(z)$. The modes it is built from (R34, p.3) have $\omega_{\rm eff}^{1/2}(z)$,
and the parallel transfer-function result R46 (p.10) has $C^2 \propto 1/\omega_{\rm eff}$ with no
star. Either the stars on p.5 are intended (then the amplitude prefactor differs between the two
constructions) or they are a slip. Reading of the star glyphs: medium. Transcribed as written.

**Q10 — NUM 11 p.5–6, sign in the $Q(u)$ construction (R38).** With $d\Theta/dz = +\omega_{\rm eff}$
as used on p.6 and $u = z_i - z$, the equation $dQ/du = -(\omega_{\rm eff}/\omega_{\rm eff}^i)/(1+u) - Q/(1+u)$
has, for constant $\omega_{\rm eff}$, the fixed point $Q = -1$, while the accompanying text says
$Q(u)$ "will be fairly close to unity when $u\gg1$". The $\pm$ kept in R33 would resolve this with
the lower sign. Transcribed as written; flagged as an apparent inconsistency between the text and
the displayed equation.
**Closed 2026-09-07:** wording only. Under $d\Theta/dz=+\omega_{\rm eff}$ the fixed point is $Q=-1$ and "close to unity" means
$|Q|\to1$; the code implements exactly this and does not depend on the sign of $Q$ (§0.2).

**Q11 — Transfer-function content in NUM 11 (R39–R46).** $\phi$, $c_s^2$ and the early-time
normalisation of the transfer function are not defined in NUM 11; the evolution equation R39 is
quoted from elsewhere (presumably NUM 01/08/09, Group 1). The symbol $\omega_{\rm eff}$ is reused
for a different function from the Green's-function $\omega_{\rm eff}$. Cross-check against
`01-transfer-function.md` is left to the orchestrator.

**Q12 — NUM 10 does not restate $\omega'_{\rm eff}/\omega_{\rm eff}$.** NUM 10 gives
$2\omega_{\rm eff}\,d\omega_{\rm eff}/dz$ only (R31). The corrected replacement for the second line of
R27 (division by $2\omega_{\rm eff}^2$) is implied but not written. No other quantity in NUM 05
(R26, R28–R30) is affected by the correction.
