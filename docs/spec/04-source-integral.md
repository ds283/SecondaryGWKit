# Spec 04 — Source time integral

Transcribed from handwritten notes (D. Seery) as part of the spec-transcription campaign
(`prompts/spec-transcription/README.md`, Group 4). Primary transcription. Nothing here has been
checked against the code; page content was transcribed as written, not corrected.

Sign-off: **Tier 1.1 (Green's-function normalisation), Tier 2.1, 2.2 and the Tier 3 notation items signed off by the author 2026-09-07; see §0. All review-queue items for this file are closed** (Tier 1.2–1.4 do not concern this file).

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

- **2.1 (NUM 06 p.4 last term, p.6 top boxed "so $f=$", p.6 foot).** Every $J_{5/2}$ is $J_{5/2+b}$;
  the bare $\tfrac52$ at these three places is an omission on the page. Primary reading stands
  (`diff-04.md` D1–D3 resolved for the primary).
- **2.2 (NUM 06 p.7 final display, R14; C6, C7).** Both cross-outs confirmed. The struck prime after
  $\eta$ in $(qrc_s^2\eta)^{-1/2-b}$ is correct to strike: this $\eta$ stands outside the $\eta'$
  integration and cannot refer to its variable. In the second integral the kernel is $Y_{b+1/2}(k\eta')$,
  written over a struck $J$.
- **Tier 3, $w$ vs $w_0$ (p.3, Q3).** Same quantity: the background $p_0/\rho_0$ at the source time
  (spec 03 §0.2). **Prime on the transfer function**: $d/dx$ here and in `MAIN` 14; `MAIN` 11's $d/d\eta$
  is the anomaly (spec 05 §0.4).
- **Audit note (2026-09), not author sign-off — Q7 (NUM 06 p.10 vs NUM 07 p.1, Liouville–Green
  convention).** The code's analytic branch (`LiouvilleGreen/bessel_phase.py`) fixes the
  unstated identification: $J_\nu = m\sin\vartheta$, $Y_\nu = -m\cos\vartheta$, i.e. `NUM` 07's R22
  with $\gamma = \Theta - \pi/2$ relative to `NUM` 06 p.10's $\cos\gamma$. The code is self-consistent
  in this convention and reproduces R14 to $10^{-8}$–$10^{-6}$ relative on seven configurations. Cite
  `docs/spec-code-audit/QI-report.md` §1 row R17 and §3 note 6.
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

Base directory: `/Users/ds283/Library/CloudStorage/Box-Box/Research projects/SIGWs/4D/Calculations/Numerical implementation/`

| Tag | File | Pages | Date on first page | Title on first page |
|---|---|---|---|---|
| NUM 06 | `06 - 2024-10-29 - analytic source integral.pdf` | 11 | 29 Oct 2024 | "Analytic quadratic source integral" |
| NUM 07 | `07 - 2024-12-01 - Fabrikant integrals via Levin method.pdf` | 3 | 29 Nov 2024 (file name says 2024-12-01) | "Fabrikant integrals via the Levin method" |

Page structure of NUM 06: Step 1 (p.1–3, set-up and background cosmology), Step 2 (p.3–6,
simplify the source term), Step 3 (p.6–7, collect the time integral), Step 4 (p.7–9, Liouville–Green
form suitable for Levin integration), untitled section "Evaluation of 3-Bessel integrals" (p.10–11).
Pages 10–11 are unnumbered on the page; page numbers used below are the PDF page positions.

Page structure of NUM 07: Step 1 (p.1, Liouville–Green representation), Step 2 (p.1–2, Fabrikant
integrals $\mathcal{J}^{\mu}_{\nu\sigma}$ in Levin form), Step 3 (p.3, closed-form comparison values).

---

## 2. Conventions in force

- **Time variable.** The numerical source integral is written in redshift $z$, with the dummy
  integration variable $z'$ (NUM 06 p.1, p.6). The analytic evaluation is in conformal time
  $\eta$, dummy variable $\eta'$ (NUM 06 p.1, p.6–7, p.10). The text on p.1 says "the analytic
  Green's function in $\tau$" but every formula uses the symbol $\eta$; $\tau$ never appears in a
  formula. Cosmic time $t$ appears only through $dt = a\,d\eta$ (p.2, p.6). NUM 07 uses a generic
  radial variable $x$ (Fabrikant integrals over $x\in[0,\infty)$); no time variable.
- **Meaning of a prime.** On the transfer function, $T'$ means $d/d\eta$ (inferred from the
  combination $T'/\mathcal{H}$ with $\mathcal{H} = a'/a$ on p.3, and from $T_q' = q c_s\,dT/dx$ used
  implicitly on p.4; not stated in words). On $a$, $a' = da/d\eta$ (p.2 "$\mathcal{H} = a'/a$").
  On Bessel functions and on $y, f, \Theta, \vartheta, \beta$ in Step 4 and in NUM 07,
  a prime is $d/dx$ where $x$ is the Bessel argument (p.4, p.7–9; NUM 07 p.1). On $z', \eta'$ the
  prime marks the dummy (source-time) variable, not a derivative.
- **$\mathcal{H}$ vs $H$.** Both used. $H = \dot a/a$ (p.2: $3H^2 M_P^2 = \rho$ with
  $\dot a/a$), $\mathcal{H} = a'/a$ (p.2). Relation used: $\mathcal{H} = a H$ (p.6:
  $dz = -(a_0/a)(\dot a/a)\,a\,d\eta = -(a_0/a)\mathcal{H}\,d\eta = -a_0 H\,d\eta$). No
  $\epsilon$ is used. The author writes $\mathcal{H}$ as a looped script capital that looks like
  "$\partial e$"; it is read as $\mathcal{H}$ throughout because p.2 defines it as $a'/a$.
- **Scale-factor normalisation.** $a_0$ appears explicitly: $1+z = a_0/a$ (p.6), and an overall
  factor $a_0$ survives into the final result (p.7). $a_0 = 1$ is **not** assumed.
  **Author note (2026-09-07):** correct, and neither does the code assume it: $a_0$ is absorbed
  into $k/a_0$ and $a_0\eta$ (§0 item 1). The surviving power on p.7 should be $a_0^2$, see R14 note.
- **Green's function.** "My Green's function" $G_k(z,z')$ is "defined with source $-\delta(z-z')$"
  (p.1). Response argument is $z$ (first slot), source argument is $z'$ (second slot). The
  relation to "the analytic Green's function" is $G_{\rm me} = -H(z')\,G_{\rm them}$ (p.1), with
  $G_{\rm them}(\eta,\eta')$ given explicitly (R3) and vanishing for $\eta < \eta'$ (retarded).
  The source-sign convention and boundary conditions of $G_{\rm them}$ itself are **not stated**
  on the page. Whether $-\delta(z-z')$ refers to a $\delta$ in $z$ with a $dz$ measure is not
  spelt out beyond the phrase quoted.
- **Equation of state.** Fixed $w$ during the epoch (p.2). $b = (1-3w)/(1+3w)$ (p.3, stated).
  Derived on p.3: $w = \tfrac13\,(1-b)/(1+b)$, $3(1+w) = 2(2+b)/(1+b)$, $2/(1+3w) = 1+b$,
  $\mathcal{H} = (1+b)/\eta$, $a \propto \eta^{2/(1+3w)} = \eta^{1+b}$ (p.2–3, p.6). The sound
  speed $c_s$ appears only inside Bessel arguments $k c_s \eta$; the formula
  $c_s^2 = (1-b)/(3(1+b))$ is **not stated** anywhere in either document (had to be inferred as
  external knowledge; not used in the transcription). $\rho_0$, $p_0$ appear as the density and
  pressure in $3H^2 M_P^2 = \rho_0 (a_0/a)^{3(1+w)}$ (p.2) and in the source prefactor
  $2M_P^2/(\rho_0(1+p_0/\rho_0)) = 2M_P^2/(\rho_0(1+w))$ (p.1, p.3). On p.3 the author writes
  $3H^2M_P^2 = \rho_0$ (no scale-factor power), i.e. treats $\rho_0$ there as the density at the
  time in question; see Open questions Q3.
- **Planck mass.** $M_P$ with $3H^2 M_P^2 = \rho$ (p.2), i.e. the reduced Planck mass (inferred
  from the Friedmann equation as written; not stated in words).
- **Fourier / power-spectrum conventions.** None stated in either document. No $(2\pi)^3$
  factors, $\delta$-function normalisations, or polarisation tensors appear. $f(z'\,|\,\mathbf q,
  \mathbf k - \mathbf q)$ is called "the tensor source" (p.1) and is a bilinear in transfer
  functions; its own normalisation is taken from elsewhere (Group 3).
- **Transfer function.** $T_k(\eta) = 2^{3/2+b}\,\Gamma(\tfrac52+b)\,(k c_s\eta)^{-3/2-b}
  J_{3/2+b}(k c_s\eta)$ (p.1, p.4). Which field it transfers ($\phi$, $\Phi$, $\zeta$,
  $\mathcal R$) is **not stated**. Early-time normalisation is not stated; with the prefactor
  written, $T_k \to 1$ as $\eta \to 0$ (inferred from the small-argument form of $J_\nu$; not
  on the page).
- **Momentum labels.** External $\mathbf k$; loop momentum $\mathbf q$; the source is
  $f(z'\,|\,\mathbf q, \mathbf k - \mathbf q)$ (p.1, vectors underlined in the handwriting). From
  p.4 onward the two transfer functions have arguments $q c_s\eta$ and $r c_s\eta$, so
  $r = |\mathbf k - \mathbf q|$ is inferred (never written explicitly). No angular reduction is
  performed in either document. NUM 07 uses $k, q, s$ as the three magnitudes in the Fabrikant
  integrals (p.1, p.3).
- **Liouville–Green (Levin) representation of Bessel functions.** Two different phase
  conventions appear. NUM 07 p.1: $J_\nu(x) = m(x)\sin\Theta(x)$ with
  $m^2 = J_\nu^2 + Y_\nu^2 = \frac{2}{\pi x}\frac{1}{\Theta'(x)}$. NUM 06 p.10: a generic Bessel
  function $Z_i(x_i)$ is replaced by $(2/\pi)^{1/2}\,(x_i\beta_i(x_i))^{-1/2}\cos\gamma_i(x_i)$,
  with $\sin\gamma$ "for the Neumann case" (i.e. $Y$). $\beta_i$ is not defined on the page
  (presumably $\gamma_i'$, by analogy with NUM 07); $\gamma$ and $\Theta$ presumably differ by
  $\pi/2$. Neither identification is written down; see Open questions Q7.

---

## 3. Results

### 3.1 NUM 06 — "Analytic quadratic source integral" (29 Oct 2024)

**R1** — NUM 06 p.1. Source integral computed numerically (Step 1).
$$
\int_{z}^{z_{\rm init}} dz'\; G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{1}{H(z')^2}\,
f(z'\,|\,\mathbf q,\mathbf k-\mathbf q).
$$
$G_k(z,z')$ is "my Green's function defined with source $-\delta(z-z')$"; $f$ is "the tensor
source". Confidence: **high**.

**R2** — NUM 06 p.1. Relation between the author's Green's function and the analytic one.
$$
G_{\rm me} = -\,H(z')\,G_{\rm them}.
$$
$H(z')$ is the Hubble rate at the *source* time. Confidence: **high**.

**Author note (2026-09-07):** $G_{\rm me}$ is **the Green's function the code computes**
(`ComputeTargets/GkNumericIntegration.py`), identical to $\bar G_k$ of spec 03 R25. R2 is written in the
absorbed convention: $G_{\rm them}$ here is to be read in the variable $a_0\eta$ (the code's $\tau$), in which
case no $a_0$ appears. In comoving $\eta$ the same relation is $G_{\rm me} = -a_0H(z')\,G_{\rm them}$. See §0 item 2.

**R3** — NUM 06 p.1. The analytic (conformal-time) Green's function.
$$
G_{\rm them}(\eta,\eta') = \frac{\pi}{2}\sqrt{\eta\eta'}\;
\begin{cases}
-\,Y_{b+\frac12}(k\eta')\,J_{b+\frac12}(k\eta) + J_{b+\frac12}(k\eta')\,Y_{b+\frac12}(k\eta), & \eta>\eta',\\[4pt]
0, & \eta<\eta'.
\end{cases}
$$
Order is $b+\tfrac12$ on all four Bessel functions; the primed argument $k\eta'$ sits on the first
factor of each product. Confidence: **high** (a struck glyph between $Y_{b+1/2}(k\eta')$ and
$J_{b+1/2}(k\eta)$ is recorded in §5, C1).

**R4** — NUM 06 p.1 (repeated p.4 with $x = k c_s \eta$). Transfer function.
$$
T_k(\eta) = 2^{\frac32+b}\,\Gamma\!\left(\tfrac52+b\right)\,(k c_s\eta)^{-\frac32-b}\,
J_{\frac32+b}(k c_s\eta).
$$
Confidence: **high**.

**R5** — NUM 06 p.1. The tensor source as used in this document.
$$
f = T_q T_r + \frac{2M_P^2}{\rho_0\left(1+\dfrac{p_0}{\rho_0}\right)}\;\frac{1}{a^2}\,
\bigl(T_q' + \mathcal{H}\,T_q\bigr)\bigl(T_r' + \mathcal{H}\,T_r\bigr).
$$
$T_q \equiv T_q(\eta)$ and $T_r \equiv T_r(\eta)$ with $q = |\mathbf q|$, $r = |\mathbf k-\mathbf q|$
(inferred). Confidence: **high** for the structure; **medium** for the denominator, read as
$\rho_0(1 + p_0/\rho_0)$ — the stacked fraction is small; p.3 rewrites it as $\rho_0(1+w)$, which
supports this reading.

**R6** — NUM 06 p.3 (Step 2). Simplified source term. Using $3M_P^2\,\mathcal{H}^2/a^2 = \rho_0$
(written on p.3 as $3H^2M_P^2 = \rho_0 \Rightarrow 3M_P^2(a'/a^2)^2 = \rho_0$), so that
$\frac{1}{\rho_0}\frac{\mathcal{H}^2}{a^2} = \frac{1}{3M_P^2}$:
$$
f = T_q T_r + \frac{2M_P^2}{\rho_0(1+w)}\frac{\mathcal{H}^2}{a^2}
\left(\frac{T_q'}{\mathcal{H}} + T_q\right)\left(\frac{T_r'}{\mathcal{H}} + T_r\right)
= T_q T_r + \frac{2M_P^2}{3M_P^2}\,\frac{1}{1+w_0}
\left(\frac{T_q'}{\mathcal{H}} + T_q\right)\left(\frac{T_r'}{\mathcal{H}} + T_r\right).
$$
Confidence: **high** for the structure; the last denominator is written "$1+w_0$" (with a
subscript $0$) whereas the line above has "$1+w$" — see Open questions Q3.

**R7** — NUM 06 p.2–3. Background during an epoch of fixed $w$.
$$
a^{\frac{1+3w}{2}} \propto \eta,\qquad a \propto \eta^{\frac{2}{1+3w}} = \eta^{1+b},\qquad
\mathcal{H} = \frac{a'}{a} = \frac{2}{1+3w}\,\frac{1}{\eta} = \frac{1+b}{\eta}.
$$
The step $a\propto\eta^{1+b}$ is used on p.6 ($a(\eta')/a(\eta) = (\eta'/\eta)^{1+b}$).
Confidence: **high**.

**R8** — NUM 06 p.3. Relations between $w$ and $b$ (from $b = \frac{1-3w}{1+3w}$).
$$
w = \frac13\,\frac{1-b}{1+b},\qquad 3w = \frac{1-b}{1+b},\qquad
3(1+w) = \frac{4+2b}{1+b} = \frac{2(2+b)}{1+b},\qquad \frac{2}{1+3w} = 1+b .
$$
Confidence: **high**.

**R9** — NUM 06 p.4. Derivative of the transfer function, via a Bessel identity.
Identity as written: "$Z_\alpha'(x) - \frac{\alpha}{x} Z_\alpha(x) = -Z_{\alpha+1}(x)$ for any
Bessel function $Z_\alpha$." Then, with $T_k(x) = 2^{3/2+b}\Gamma(\tfrac52+b)\,x^{-3/2-b}J_{3/2+b}(x)$,
$$
\frac{dT}{dx} = 2^{\frac32+b}\,\Gamma\!\left(\tfrac52+b\right) x^{-\frac32-b}
\left\{ J'_{\frac32+b}(x) - \left(\tfrac32+b\right) J_{\frac32+b}(x) \right\}
= -\,2^{\frac32+b}\,\Gamma\!\left(\tfrac52+b\right) x^{-\frac32-b}\, J_{\frac52+b}(x).
$$
Confidence: **high** for the final form (order $\tfrac52+b$ clearly written). In the
intermediate brace the factor $1/x$ multiplying $(\tfrac32+b)$ is **not visible on the page**
although the identity quoted requires it; transcribed as written. See Open questions Q4.

**R10** — NUM 06 p.5. Second Bessel identity used to lower the order.
$$
Z_\alpha(x) - \frac{x}{2\alpha}\,Z_{\alpha+1}(x) = \frac{x}{2\alpha}\,Z_{\alpha-1}(x).
$$
Applied with $\alpha = \tfrac32+b$, so $2\alpha = 3+2b$. Confidence: **high**.

**R11** — NUM 06 p.4–6. Intermediate and final forms of the simplified source term. Intermediate
form after substituting R9 into R6 (p.4, second display; the $(1+b)/(2+b)$ arises from
$\frac23\cdot\frac{1}{1+w} = \frac{1+b}{2+b}$):
$$
f = 2^{3+2b}\,\Gamma\!\left(\tfrac52+b\right)^2 (q c_s\eta)^{-\frac32-b}(r c_s\eta)^{-\frac32-b}
\Bigl\{ \Bigl(1+\tfrac{1+b}{2+b}\Bigr) J_{\frac32+b}(q c_s\eta) J_{\frac32+b}(r c_s\eta)
- \tfrac{q c_s\eta}{2+b} J_{\frac52+b}(q c_s\eta) J_{\frac32+b}(r c_s\eta)
- \tfrac{r c_s\eta}{2+b} J_{\frac52+b}(r c_s\eta) J_{\frac32+b}(q c_s\eta)
+ \tfrac{(q c_s\eta)(r c_s\eta)}{(2+b)(1+b)} J_{\frac52+b}(q c_s\eta) J_{\frac52+b}(r c_s\eta)\Bigr\}.
$$
Factored form (p.5, second display):
$$
f = 2^{3+2b}\,\frac{3+2b}{2+b}\,\Gamma\!\left(\tfrac52+b\right)^2 (q c_s\eta)^{-\frac32-b}(r c_s\eta)^{-\frac32-b}
\Bigl\{ \Bigl(J_{\frac32+b}(q c_s\eta) - \tfrac{q c_s\eta}{3+2b} J_{\frac52+b}(q c_s\eta)\Bigr)
\Bigl(J_{\frac32+b}(r c_s\eta) - \tfrac{r c_s\eta}{3+2b} J_{\frac52+b}(r c_s\eta)\Bigr)
+ (q c_s\eta)(r c_s\eta)\Bigl(\tfrac{1}{(3+2b)(1+b)} - \tfrac{1}{(3+2b)^2}\Bigr)
J_{\frac52+b}(q c_s\eta) J_{\frac52+b}(r c_s\eta)\Bigr\}.
$$
**Final simplified source term (p.6, top, "so $f = $"):**
$$
\boxed{\;
f = \frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2
(q c_s\eta)^{-\frac12-b}\,(r c_s\eta)^{-\frac12-b}
\Bigl\{ J_{\frac12+b}(q c_s\eta)\,J_{\frac12+b}(r c_s\eta)
+ \frac{2+b}{1+b}\,J_{\frac52+b}(q c_s\eta)\,J_{\frac52+b}(r c_s\eta) \Bigr\}.
\;}
$$
Confidence: **high** for the final form (orders $\tfrac12+b$ and $\tfrac52+b$, exponents
$-\tfrac12-b$, prefactor $(3+2b)(2+b)$ and ratio $(2+b)/(1+b)$ are all clearly written).
**Medium** for the two intermediate forms only in the places recorded in §5 (C3, C4, C5), where
subscripts were over-written.

**R12** — NUM 06 p.6 (Step 3). Change of variables from $z$ to $\eta$.
$$
\frac{1+z}{1+z'} = \frac{a(z')}{a(z)} = \frac{a(\eta')}{a(\eta)} = \frac{(\eta')^{1+b}}{\eta^{1+b}},
\qquad 1+z = \frac{a_0}{a},\qquad
dz = -\frac{a_0}{a}\frac{\dot a}{a}\,dt = -\frac{a_0}{a}\frac{\dot a}{a}\,a\,d\eta
= -\frac{a_0}{a}\,\mathcal{H}\,d\eta = -a_0\,H\,d\eta .
$$
Confidence: **high**.

**R13** — NUM 06 p.6–7. The time integral in conformal time, before pulling factors outside.
Substituting R12 and R11 into R1 (p.6, bottom):
$$
-\,a_0\int_{\eta}^{\eta_{\rm init}} H(\eta')\,d\eta'\;G_k(\eta,\eta')\,\frac{(\eta')^{1+b}}{\eta^{1+b}}\,
\frac{1}{H(\eta')^2}\;
\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma\!\left(\tfrac52+b\right)^2
(q c_s\eta')^{-\frac12-b}(r c_s\eta')^{-\frac12-b}
\Bigl\{ J_{\frac12+b}(q c_s\eta)J_{\frac12+b}(r c_s\eta) + \tfrac{2+b}{1+b} J_{\frac52+b}(q c_s\eta)J_{\frac52+b}(r c_s\eta)\Bigr\},
$$
then, using R2 and R3 (the two factors of $H(\eta')$ cancel against $1/H(\eta')^2$; the $(-1)$
comes from reversing the limits) (p.7, top):
$$
= a_0\int_{\eta_{\rm init}}^{\eta} d\eta'\;(-1)\,\frac{\pi}{2}\sqrt{\eta\eta'}\,
\Bigl\{ J_{b+\frac12}(k\eta')\,Y_{b+\frac12}(k\eta) - Y_{b+\frac12}(k\eta')\,J_{b+\frac12}(k\eta) \Bigr\}
\frac{(\eta')^{1+b}}{\eta^{1+b}}\,\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma\!\left(\tfrac52+b\right)^2
(q c_s\eta')^{-\frac12-b}(r c_s\eta')^{-\frac12-b}
\Bigl\{ J_{\frac12+b}(q c_s\eta')J_{\frac12+b}(r c_s\eta') + \tfrac{2+b}{1+b} J_{\frac52+b}(q c_s\eta')J_{\frac52+b}(r c_s\eta')\Bigr\}.
$$
On p.6 the Bessel arguments inside the braces are written with **unprimed** $\eta$ although the
power-law prefactors on the same line carry $\eta'$; on p.7 all of them carry $\eta'$. Transcribed
as written; see Open questions Q5. Confidence: **high** for p.7; the p.6 line is as written.

**R14** — NUM 06 p.7. **Final analytic form of the source time integral** (end of Step 3).
$$
\boxed{\;
\begin{aligned}
\int_{z}^{z_{\rm init}} dz'\; G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{1}{H(z')^2}\, f
\;=\; -\,\frac{a_0\pi}{2}\,\frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2
\bigl(q\,r\,c_s^2\,\eta\bigr)^{-\frac12-b}
\Biggl\{\;& Y_{b+\frac12}(k\eta)\int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,J_{b+\frac12}(k\eta')
\Bigl( J_{\frac12+b}(q c_s\eta')J_{\frac12+b}(r c_s\eta') + \tfrac{2+b}{1+b}\,J_{\frac52+b}(q c_s\eta')J_{\frac52+b}(r c_s\eta') \Bigr) \\[4pt]
-\;& J_{b+\frac12}(k\eta)\int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,Y_{b+\frac12}(k\eta')
\Bigl( J_{\frac12+b}(q c_s\eta')J_{\frac12+b}(r c_s\eta') + \tfrac{2+b}{1+b}\,J_{\frac52+b}(q c_s\eta')J_{\frac52+b}(r c_s\eta') \Bigr)
\Biggr\}.
\end{aligned}
\;}
$$
The left-hand side is not re-written on p.7; it is R1 with $f$ from R11. Key features: overall
sign $-$; prefactor $a_0\pi/2$; the $\eta$-dependence outside the integrals is
$(q r c_s^2\eta)^{-1/2-b}$ (unprimed $\eta$); the weight inside each integral is
$(\eta')^{1/2-b}$; the Green's-function Bessel functions have order $b+\tfrac12$ and the source
Bessel functions have orders $\tfrac12+b$ and $\tfrac52+b$ (the author writes "$b+\tfrac12$" for
the former and "$\tfrac12+b$" for the latter, which are the same order). Confidence: **high** for
every order, exponent and sign. The only marks on this display are the two cross-outs recorded in
§5, C6 and C7 (a struck prime after $\eta$ in $(q r c_s^2\eta)$, and $Y$ written over a struck $J$
in the second integral), both of which resolve to the reading above.

**Author note (2026-09-07):** in comoving variables the prefactor must be $a_0^2$, not $a_0^1$
(covariance test, §0 item 4). The page absorbed one power of $a_0$ in R2 while keeping $a_0$ explicit in
R13. The code's `analytic_integral` (`ComputeTargets/QuadSourceIntegral.py`) is this formula in the
variables $k/a_0$, $a_0\eta$, in which the $a_0$ disappears; it is **not** an $a_0 = 1$ normalisation.
With $a_0^2$, R14 $= -\dfrac{a_0^2}{c^2}\times$ (`MAIN` 14 target, spec 05 R31), $c^2 = (2+b)^2/(3+2b)^2$.

**R15** — NUM 06 p.7–8 (Step 4, "Write in a form suitable for Levin integration"). Liouville
normal form of the Bessel equation. Starting from
$$
y'' + \frac1x y' + \Bigl(1 - \frac{\nu^2}{x^2}\Bigr) y = 0,
$$
write $y = f\tilde y$ and choose $2f'/f = -1/x$, i.e. $f \propto x^{-1/2}$; then (renaming
$\tilde y \to y$) $f''/f + \frac1x f'/f = \frac{1}{4x^2}$ and
$$
y'' + \Bigl(1 + \frac{\tfrac14 - \nu^2}{x^2}\Bigr) y = 0 .
$$
Confidence: **high**.

**R16** — NUM 06 p.9. Liouville–Green phase equations. Seeking $y \sim e^{i\Theta}$, the phase
satisfies the Riccati equation
$$
i\Theta'' - (\Theta')^2 + \omega_{\rm eff}^2 = 0 .
$$
Writing $\Theta = \vartheta + i\beta$, the imaginary and real parts give
$$
\vartheta'' - 2\vartheta'\beta' = 0 \;\Rightarrow\; \beta' = \frac{\vartheta''}{2\vartheta'},\quad
\beta = \tfrac12 \ln\vartheta';\qquad
-\beta'' - \vartheta'^2 + \beta'^2 + \omega_{\rm eff}^2 = 0,
$$
and eliminating $\beta$,
$$
-\frac{\vartheta'''}{2\vartheta'} + \frac34\,\frac{\vartheta''}{\vartheta'}\frac{\vartheta''}{\vartheta'}
- \vartheta'^2 + \omega_{\rm eff}^2 = 0 \qquad\text{("Kummer's diff. eq.")}.
$$
"The leading solution is the WKB one: $-\vartheta'^2 + \omega_{\rm eff}^2 = 0$."
$\omega_{\rm eff}^2$ is not defined on p.9; from R15 it is $\omega_{\rm eff}^2 = 1 + (\tfrac14-\nu^2)/x^2$
(inferred). Confidence: **high** for what is written.

**R17** — NUM 06 p.10 ("Evaluation of 3-Bessel integrals"). Liouville–Green form of the generic
three-Bessel time integral appearing in R14.
$$
\int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,Z_1(x_1)Z_2(x_2)Z_3(x_3)
= \int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,
\Bigl(\frac{2}{\pi}\Bigr)^{3/2}\frac{1}{(x_1x_2x_3)^{1/2}}\,
\frac{1}{\bigl(\beta_1(x_1)\beta_2(x_2)\beta_3(x_3)\bigr)^{1/2}}\,
\cos\gamma_1(x_1)\cos\gamma_2(x_2)\cos\gamma_3(x_3).
$$
The $x_i$ are the three Bessel arguments (unlabelled on the page; in R14 they are $k\eta'$,
$q c_s\eta'$, $r c_s\eta'$). $\beta_i$ and $\gamma_i$ are not defined on the page (see Open
questions Q7). Confidence: **high** for what is written.

**R18** — NUM 06 p.10. Product-to-sum expansion, three cosines (all-$J$ case).
$$
\cos\gamma_1\cos\gamma_2\cos\gamma_3 = \tfrac14\Bigl\{
\cos(\gamma_1+\gamma_2+\gamma_3) + \cos(\gamma_1+\gamma_2-\gamma_3)
+ \cos(\gamma_1-\gamma_2+\gamma_3) + \cos(\gamma_1-\gamma_2-\gamma_3)\Bigr\}.
$$
Confidence: **high**.

**R19** — NUM 06 p.10 ("for the Neumann case"). One sine, two cosines (one $Y$, two $J$).
$$
\sin\gamma_1\cos\gamma_2\cos\gamma_3 = \tfrac14\Bigl\{
\sin(\gamma_1+\gamma_2+\gamma_3) + \sin(\gamma_1+\gamma_2-\gamma_3)
+ \sin(\gamma_1-\gamma_2+\gamma_3) + \sin(\gamma_1-\gamma_2-\gamma_3)\Bigr\}.
$$
Confidence: **high**.

**R20** — NUM 06 p.11. Three sines. Final collected line:
$$
\sin\gamma_1\sin\gamma_2\sin\gamma_3 = \tfrac14\Bigl(
-\sin(\gamma_1+\gamma_2+\gamma_3) + \sin(\gamma_1+\gamma_2-\gamma_3)
+ \sin(\gamma_1-\gamma_2+\gamma_3) - \sin(\gamma_1-\gamma_2-\gamma_3)\Bigr).
$$
Confidence: **medium**. Two signs on this page have been over-written (a "$+$" and a "$-$"
superimposed) rather than cleanly struck: the sign before $\gamma_3$ in the second term of the
final line, and the sign before $\gamma_2$ in the second term of the preceding intermediate line.
The reading above takes both as "$-$"; see §5, C10–C11, and Open questions Q8. All other terms
are cleanly written.

**R21** — NUM 06 p.11. One cosine, two sines. Final collected line:
$$
\cos\gamma_1\sin\gamma_2\sin\gamma_3 = \tfrac14\Bigl(
-\cos(\gamma_1+\gamma_2+\gamma_3) + \cos(\gamma_1+\gamma_2-\gamma_3)
+ \cos(\gamma_1-\gamma_2+\gamma_3) - \cos(\gamma_1-\gamma_2-\gamma_3)\Bigr).
$$
Confidence: **high** for the final line (cleanly written). In the intermediate line a "$\sin$"
was struck and replaced by "$\cos$" (§5, C12).

### 3.2 NUM 07 — "Fabrikant integrals via the Levin method" (29 Nov 2024)

**R22** — NUM 07 p.1 (Step 1). Liouville–Green representation of the Bessel function.
$$
J_\nu(x) = m(x)\sin\Theta(x),\qquad
m^2(x) = J_\nu^2(x) + Y_\nu^2(x) = \frac{2}{\pi x}\,\frac{1}{\Theta'(x)}.
$$
"$m$" is called "the modulus (amplitude) function". Confidence: **high** for the structure;
**medium** for the denominator of the last expression, which is written with a glyph that looks
like "$\pi z$" rather than "$\pi x$" — read as $\pi x$ since $x$ is the only variable in use.
The "$Y_\nu^2$" is written above a struck "$J_\nu^2$" (§5, C13).

**R23** — NUM 07 p.1. Spherical Bessel function.
$$
j_\nu(x) = \sqrt{\frac{\pi}{2x}}\;J_{n+\frac12}(x).
$$
The left-hand side has index $\nu$ and the right-hand side index $n$, as written (evidently the
same index). Confidence: **high** (glyphs); the $\nu$/$n$ mismatch is recorded in Open questions Q9.

**R24** — NUM 07 p.1 (Step 2). Definition of the Fabrikant integrals "from the EFT paper".
$$
\mathcal{J}^{\mu}_{\nu\sigma} = \int_0^\infty x^2\, j_\mu(kx)\, j_\nu(qx)\, j_\sigma(sx)\, dx .
$$
The upper index is attached to the first spherical Bessel function (argument $kx$), the two
lower indices to the second and third (arguments $qx$, $sx$), in that order. "The EFT paper" is
not identified further on the page. Confidence: **high**.

**R25** — NUM 07 p.1. Reduction to cylindrical Bessel functions.
$$
\mathcal{J}^{\mu}_{\nu\sigma}
= \int_0^\infty x^2\sqrt{\frac{\pi}{2xk}}\sqrt{\frac{\pi}{2qx}}\sqrt{\frac{\pi}{2sx}}\;
J_{\mu+\frac12}(kx)\,J_{\nu+\frac12}(qx)\,J_{\sigma+\frac12}(sx)\,dx
= \Bigl(\frac{\pi}{2}\Bigr)^{3/2}\frac{1}{\sqrt{kqs}}\int_0^\infty x^{1/2}\,
J_{\mu+\frac12}(kx)\,J_{\nu+\frac12}(qx)\,J_{\sigma+\frac12}(sx)\,dx .
$$
The "$dx$" of the middle expression runs off the right margin of the page. Two struck glyphs
precede $(\pi/2)^{3/2}$ in the last expression (§5, C14). Confidence: **high**.

**R26** — NUM 07 p.1. **Form written for Levin integration** (amplitude × oscillatory factor).
$$
\mathcal{J}^{\mu}_{\nu\sigma}
= \Bigl(\frac{\pi}{2}\Bigr)^{3/2}\frac{1}{(kqs)^{1/2}}\int_0^\infty x^{1/2}\;
m_{\mu+\frac12}(kx)\,m_{\nu+\frac12}(qx)\,m_{\sigma+\frac12}(sx)\;
\sin\Theta_{\mu+\frac12}(kx)\,\sin\Theta_{\nu+\frac12}(qx)\,\sin\Theta_{\sigma+\frac12}(sx)\;dx .
$$
Amplitude: $x^{1/2}\,m_{\mu+1/2}(kx)\,m_{\nu+1/2}(qx)\,m_{\sigma+1/2}(sx)$ (with $m$ from R22).
Oscillatory factor: the triple product of sines, expanded into four single sines in R28.
Range: $[0,\infty)$ as written; **no range reduction, cut-off or splitting of the integration
domain is written anywhere in the document.** Confidence: **high**.

**R27** — NUM 07 p.2. Product-to-sum identities used.
$$
\cos\alpha\cos\beta = \tfrac12[\cos(\alpha-\beta)+\cos(\alpha+\beta)],\quad
\sin\alpha\sin\beta = \tfrac12[\cos(\alpha-\beta)-\cos(\alpha+\beta)],\quad
\sin\alpha\cos\beta = \tfrac12[\sin(\alpha+\beta)+\sin(\alpha-\beta)].
$$
Confidence: **high**.

**R28** — NUM 07 p.2. Expansion of the triple sine in R26. Writing
$\Theta_\mu \equiv \Theta_{\mu+\frac12}(kx)$, $\Theta_\nu \equiv \Theta_{\nu+\frac12}(qx)$,
$\Theta_\sigma \equiv \Theta_{\sigma+\frac12}(sx)$ (the page writes the arguments out in full each
time):
$$
\sin\Theta_\mu\sin\Theta_\nu\sin\Theta_\sigma
= \tfrac14\Bigl\{ -\sin(\Theta_\mu+\Theta_\nu+\Theta_\sigma) + \sin(\Theta_\mu+\Theta_\nu-\Theta_\sigma)
+ \sin(\Theta_\mu-\Theta_\nu+\Theta_\sigma) - \sin(\Theta_\mu-\Theta_\nu-\Theta_\sigma) \Bigr\}.
$$
Intermediate line (as written): $\tfrac14\{\sin(\Theta_\mu+\Theta_\nu-\Theta_\sigma) +
\sin(\Theta_\mu-\Theta_\nu+\Theta_\sigma)\} - \tfrac14\{\sin(\Theta_\mu+\Theta_\nu+\Theta_\sigma)
+ \sin(\Theta_\mu-\Theta_\nu-\Theta_\sigma)\}$, where the first "$\tfrac14$" was originally
written "$\tfrac12$" and corrected (§5, C15). In the first intermediate line the argument of
$\sin\Theta_{\mu+1/2}$ is written "$(xk)$" rather than "$(kx)$". Confidence: **high** (signs all
cleanly written; the final line agrees with the intermediate one).

**R29** — NUM 07 p.3 (Step 3, "For simple comparisons we have the results").
$$
\mathcal{J}^{0}_{00} = \frac{\pi}{4kqs},\qquad
\mathcal{J}^{1}_{10} = \frac{\pi}{8}\,\frac{k^2+q^2-s^2}{k^2q^2s},\qquad
\mathcal{J}^{2}_{20} = \frac{\pi}{32}\,\frac{3k^4 + 2k^2(q^2-3s^2) + 3(q^2-s^2)^2}{k^3q^3s}.
$$
No conditions (e.g. triangle inequality on $k,q,s$) are stated. Confidence: **high**.

---

## 4. Checks

Neither document is a recheck of another document, so there are no author-recorded agreements or
discrepancies to report. Two items are check-like and are noted for completeness:

- NUM 07 p.3 (R29) gives three closed-form Fabrikant integrals explicitly "for simple
  comparisons", i.e. as test values for the Levin evaluation of R26. No numerical comparison is
  recorded on the page.
- NUM 06 p.11: three of the four terms in the intermediate line of R20 carry small tick marks
  beneath them (the terms $\sin(\gamma_1-\gamma_2+\gamma_3)$, $-\sin(\gamma_1+\gamma_2+\gamma_3)$,
  $+\sin(\gamma_1+\gamma_2-\gamma_3)$); the fourth, whose sign was over-written, has no tick.
  These look like the author's own re-check marks; no discrepancy is written.

---

## 5. Corrections and cross-outs

Listed in page order. "Used later" states which version the subsequent lines on the page follow.

- **C1** — NUM 06 p.1, $G_{\rm them}$ (R3). Between "$-Y_{b+\frac12}(k\eta')$" and
  "$J_{b+\frac12}(k\eta)$" there is a single struck glyph, most likely a "$Y$" begun in error and
  replaced by the following "$J$". Used later: the product $Y_{b+1/2}(k\eta')J_{b+1/2}(k\eta)$
  (confirmed by p.7, where the same bracket is rewritten cleanly).
- **C2** — NUM 06 p.2, background algebra (not transcribed as a result). In the line
  "so $a^{\dots}\,da = \dots$" the exponent on $a$ was first written "$\tfrac12 - \tfrac{3w}{2}$",
  struck, and rewritten above as "$-\tfrac12 + \tfrac{3w}{2}$". Two lines below, a fragment
  "$a^{3/2 - 3w/2}$" (with a struck "$\varepsilon$" or "$e$" before "so") is struck out
  entirely. Used later: the corrected exponent, leading to $a^{1/2+3w/2}/(\tfrac12+\tfrac{3w}{2})
  = \sqrt{\rho_0/3M_P^2}\,a_0^{-1/2+3w/2}(\eta-\eta')$ and hence R7.
- **C3** — NUM 06 p.4, first display of $f$ after substituting the derivatives (R11,
  intermediate). The coefficient of the cross term is written "$+\,2\,\dfrac{1+b}{2(2+b)}$" with a
  struck "$3$" at the left of the denominator, i.e. the original "$\tfrac{2}{3}\cdot\tfrac{3(1+b)}{2(2+b)}$"
  had its factors of 3 cancelled on the page. Used later: $(1+b)/(2+b)$.
- **C4** — NUM 06 p.4, same display, second square bracket: the order of the last Bessel function
  "$J_{\frac32+b}(r c_s\eta)$" has an over-written numerator (a "5" or "3" written over the other).
  Read as $\tfrac32+b$ by symmetry with the first bracket, which is clean. Used later: $\tfrac32+b$
  (the next display has $J_{3/2+b}(q c_s\eta)J_{3/2+b}(r c_s\eta)$ cleanly).
- **C5** — NUM 06 p.5, factored form (R11, second intermediate), first bracket: the order of
  "$J_{\frac52+b}(q c_s\eta)$" has an over-written numerator (looks like "3" corrected to "5").
  Read as $\tfrac52+b$, matching the preceding display and the second bracket. In the "so $f=$"
  display at the foot of p.5 there are three further small struck glyphs: one between
  "$\frac{q c_s\eta}{3+2b}J_{\frac12+b}(q c_s\eta)$" and "$\frac{r c_s\eta}{3+2b}J_{\frac12+b}(r c_s\eta)$"
  (probably a stray "$+$", making the two factors a product), one inside the argument of the
  second $J_{1/2+b}$ before "$r c_s\eta$", and one after "$(q c_s\eta)(r c_s\eta)$" before the
  bracket $\bigl(\frac{3+2b}{1+b}-1\bigr)$. None changes the reading; the p.6 result R11 follows
  from the product reading.
- **C6** — NUM 06 p.7, final display (R14). In the prefactor "$(q r c_s^2\eta)^{-\frac12-b}$"
  there is a small struck mark immediately after $\eta$, consistent with a prime written and then
  struck. Used later: unprimed $\eta$ (this factor stands outside the $\eta'$ integrals, and the
  power counting $\sqrt{\eta\eta'}\,(\eta'/\eta)^{1+b}(\eta')^{-1-2b} = \eta^{-1/2-b}(\eta')^{1/2-b}$
  requires it).
  **Confirmed by the author 2026-09-07 (Tier 2.2).**
- **C7** — NUM 06 p.7, final display (R14), second integral: the Bessel function multiplying
  $(\eta')^{1/2-b}$ is written "$Y_{b+\frac12}(k\eta')$" with the $Y$ placed above a struck
  "$J$". Used later: $Y$ (required by R3: the term with $J_{b+1/2}(k\eta)$ outside carries
  $Y_{b+1/2}(k\eta')$ inside).
  **Confirmed by the author 2026-09-07 (Tier 2.2): $J$ renamed to $Y$.**
- **C8** — NUM 06 p.8: "so (now redefine $\tilde y \to y$)" — a relabelling, not a correction;
  recorded because the symbol $y$ changes meaning mid-page (R15).
- **C9** — NUM 06 p.11, R20 intermediate line: the prefactor "$\tfrac12$" has its "2" struck
  and "4" written beneath. Used later: $\tfrac14$.
- **C10** — NUM 06 p.11, R20 intermediate line, second term "$-\sin(\gamma_1 \,?\, \gamma_2 -
  \gamma_3)$": the sign before $\gamma_2$ is over-written ("$+$" and "$-$" superimposed). Read as
  "$-$", giving $-\sin(\gamma_1-\gamma_2-\gamma_3)$, which is what the final collected line uses.
- **C11** — NUM 06 p.11, R20 final line, second term "$+\sin(\gamma_1+\gamma_2 \,?\, \gamma_3)$":
  the sign before $\gamma_3$ is over-written. Read as "$-$", giving $+\sin(\gamma_1+\gamma_2-\gamma_3)$,
  consistent with the corresponding (clean) term of the intermediate line.
- **C12** — NUM 06 p.11, R21 intermediate line, third term: "$\sin$" is struck through and
  "$\cos$" written above, giving $-\cos(\gamma_1-\gamma_2-\gamma_3)$. Used later: $\cos$ (the final
  collected line has four cosines).
- **C13** — NUM 07 p.1, R22: in "$m^2(x) = J_\nu^2(x) + Y_\nu^2(x)$" the second term was first
  written "$J_\nu^2$", struck, and "$Y$" written above. Used later: $Y_\nu^2$.
- **C14** — NUM 07 p.1, R25 last expression: two glyphs (one looks like a struck "$\tfrac{3}{4}$"
  or "$\tfrac{\pi}{4}$", the other an unreadable struck symbol) precede "$(\frac{\pi}{2})^{3/2}$".
  Used later: $(\pi/2)^{3/2}(kqs)^{-1/2}$ (the next line repeats it cleanly).
- **C15** — NUM 07 p.2, R28 intermediate line: the prefactor "$\tfrac12$" has its "2" struck and
  "4" written beside it. Used later: $\tfrac14$.

---

## 6. Open questions

- **Q1 (NUM 06 p.1).** The meaning of $G_{\rm them}$ — "the analytic Green's function in $\tau$" —
  is not defined beyond the explicit formula R3. Its source sign, normalisation and the identity
  of the reference it is taken from are not stated. Whether $-\delta(z-z')$ for $G_{\rm me}$ is
  meant with respect to the measure $dz$ is also not spelt out. The relation
  $G_{\rm me} = -H(z')G_{\rm them}$ is the only bridge given.
  **Closed 2026-09-07:** $G_{\rm them}$ is `MAIN` 13/14's retarded $\chi$ Green's function (spec 05 R18, R21;
  source $+\delta(\eta-\eta')$). $-\delta(z-z')$ is with respect to the measure $dz$ and gives the unit
  jump $dG/dz|_{z'} = +1$ that the code imposes. See §0 item 2 and the R2 note.
- **Q2 (NUM 06 p.1).** The symbol $\tau$ appears in the text but every formula uses $\eta$;
  assumed to be the same variable (conformal time).
- **Q3 (NUM 06 p.3).** The Friedmann equation is written "$3H^2M_P^2 = \rho_0$" and the source
  prefactor becomes "$\frac{2M_P^2}{3M_P^2}\frac{1}{1+w_0}$" (subscript $0$ on $w$), whereas the
  line above has "$\rho_0(1+w)$" and p.2 has $\rho = \rho_0(a_0/a)^{3(1+w)}$. On p.2 $\rho_0$ is a
  reference density; on p.3 it is used as the density at time $\eta$. The transcription records
  both as written. The subscript on $w_0$ may be a slip; it has no effect on R11, which uses only
  $w$ through $b$.
- **Q4 (NUM 06 p.4).** In the intermediate bracket for $dT/dx$ (R9) the term is written
  "$-(\tfrac32+b)J_{\frac32+b}(x)$" without the factor $1/x$ that the quoted identity
  $Z_\alpha' - \frac{\alpha}{x}Z_\alpha = -Z_{\alpha+1}$ requires. The final line of R9 is
  consistent with the identity, so the $1/x$ was presumably intended. Transcribed as written.
- **Q5 (NUM 06 p.6, bottom display).** The power-law factors carry $\eta'$
  ($(q c_s\eta')^{-1/2-b}$, $(r c_s\eta')^{-1/2-b}$) but the Bessel arguments in the brace are
  written with unprimed $\eta$ ($J_{1/2+b}(q c_s\eta)$ etc.). On p.7 every occurrence is primed.
  The p.6 unprimed arguments appear to be a slip (the source is evaluated at the source time
  $\eta'$); recorded, not corrected, in R13.
- **Q6 (NUM 06 p.7–9).** Step 4 is headed "Write in a form suitable for Levin integration" but
  the section only derives the Liouville normal form and the phase (Riccati/Kummer) equations for
  a single Bessel function; the actual Levin form of the integral R14 is not written down in
  NUM 06. $\omega_{\rm eff}^2$ is used on p.9 without definition (inferred from p.8 to be
  $1 + (\tfrac14-\nu^2)/x^2$).
- **Q7 (NUM 06 p.10 vs NUM 07 p.1).** Two Liouville–Green conventions are used without a
  stated link: $Z_i(x_i) \to (2/\pi)^{1/2}(x_i\beta_i)^{-1/2}\cos\gamma_i$ (NUM 06, $\beta_i$
  undefined on the page) versus $J_\nu = m\sin\Theta$, $m^2 = \frac{2}{\pi x}\frac{1}{\Theta'}$
  (NUM 07). Comparing the two forms suggests $\beta_i = \gamma_i'$ and $\gamma = \Theta - \pi/2$
  for $J$, with $\sin\gamma$ standing for $Y$ ("Neumann case", p.10); none of this is written.
  Also the arguments $x_1, x_2, x_3$ in R17 are not labelled; in R14 they would be $k\eta'$,
  $q c_s\eta'$, $r c_s\eta'$, and the weight $(\eta')^{1/2-b}$ is carried along but not absorbed
  into the amplitude on the page.
- **Q8 (NUM 06 p.11).** The two over-written signs in R20 (C10, C11). Both are read as "$-$";
  the reading is consistent between the intermediate and final lines, but because the marks are
  superimposed rather than struck-and-rewritten, the author should confirm against the page.
  Note also that R20 and R21 (three sines; one cosine and two sines) are not needed for R14,
  whose integrands are $J J J$ ($\cos\cos\cos$, R18) and $Y J J$ ($\sin\cos\cos$, R19); the page
  gives no indication of what R20–R21 are for.
- **Q9 (NUM 07 p.1).** $j_\nu(x) = \sqrt{\pi/2x}\,J_{n+1/2}(x)$ mixes $\nu$ (left) and $n$
  (right). Same index intended.
- **Q10 (NUM 07 p.1).** In $m^2 = \frac{2}{\pi x}\frac{1}{\Theta'(x)}$ the denominator glyph
  reads "$\pi z$"; taken as $\pi x$ (medium confidence in R22).
- **Q11 (NUM 07 p.1).** "The EFT paper" from which the $\mathcal{J}^{\mu}_{\nu\sigma}$ results
  are quoted is not identified on the page.
- **Q12 (NUM 07, whole document).** The document defines the Levin integrand (R26, R28) but
  does not state how the semi-infinite range $[0,\infty)$ is to be handled (no cut-off, no
  splitting at a matching point, no treatment of the small-$x$ region where the
  Liouville–Green form is inaccurate), nor does it give the Levin ODE system, the choice of
  collocation, or the treatment of the $\Theta$ phases (which are only defined implicitly via
  $m^2 = \frac{2}{\pi x}\frac{1}{\Theta'}$). It also does not connect $\mathcal{J}^{\mu}_{\nu\sigma}$
  (integral over $x$ with $x^{1/2}$ weight and three $J$'s) to the finite-range integrals of R14
  (weight $(\eta')^{1/2-b}$, one $J$ or $Y$ of order $b+\tfrac12$ and two $J$'s of order
  $\tfrac12+b$ or $\tfrac52+b$). The Fabrikant integrals on p.3 are labelled "for simple
  comparisons", i.e. as test cases.
- **Q13 (dates).** NUM 07 is dated 29 Nov 2024 on the page but its file name carries 2024-12-01.
