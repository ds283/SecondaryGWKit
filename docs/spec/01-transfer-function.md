# Spec 01 — Transfer function

Transcribed 2026-09-07 from the handwritten notes listed below (spec-transcription campaign,
Group 1). Transcription only: nothing here has been checked against the code, and the physics has
not been "fixed". Page references are of the form "doc 12 p.3" (MAIN 12) or "NUM 01 p.3".

Sign-off: **Tier 1.4 (identity of the seed) signed off by the author 2026-09-07; see the note in §2.8.**
Remaining items (Tier 2.5, Tier 3): *pending author review.*

---

## 1. Source documents

| Short name | File | Pages in PDF | Date on first page | Title on first page |
|---|---|---|---|---|
| doc 12 | `MAIN/12 - 2022:07:21 - compute phi transfer function.pdf` | 7 (README says 9; see Open questions Q1) | 21 July 2022 | "Calculation 12 — Compute transfer function for $\Phi$" |
| NUM 01 | `NUM/01 - 2024-09-10 - Tk equation in redshift.pdf` | 5 (README says 7; see Q1) | 10 September 2024 | "Transfer function in terms of redshift" |
| NUM 04 | `NUM/04 - 2024-09-20 - Initial condition for conformal time.pdf` | 2 | 20 Sep 2024 | "Initial condition for conformal time" |
| NUM 08 | `NUM/08 - 2024-12-07 - WKB(J) phase function for transfer function.pdf` | 3 | 7 Dec 2024 | "Liouville–Green (WKB) phase function representation for the transfer function" |
| NUM 09 | `NUM/09 - 2024-12-08 - WKB(J) phase function for transfer function (check).pdf` | 7 | 8 Dec 2024 | "Check: Liouville–Green phase function for the transfer function" |

`MAIN` = `BASE/Green's function formula for P22`, `NUM` = `BASE/Numerical implementation`,
`BASE = /Users/ds283/Library/CloudStorage/Box-Box/Research projects/SIGWs/4D/Calculations`.

NUM 09 is a recheck of NUM 08 (Steps 1–3) and then continues with new material (Steps 4–5).
The recheck is recorded in §4 (Checks); the new material is recorded as results R26–R31.

---

## 2. Conventions in force

Each item lists what the documents state or use, with page references. "Inferred" marks
conventions that are never written down and had to be deduced from usage.

### 2.1 Time variable

- **doc 12:** conformal time $\eta$ throughout ($dt = a\,d\eta$, doc 12 p.3). The background
  solution is written with an integration constant, $a \propto (\eta-\eta_c)^{2/(1+3w)}$
  (doc 12 p.3), then the Domènech form $a \propto \eta^{1+b}$ is adopted (i.e. $\eta_c = 0$,
  doc 12 p.3–4).
- **NUM 01:** starts in conformal time $\eta$ (p.1, Step 1), converts to redshift $z$ as the
  independent variable (p.1, Step 2 onward). Cosmic time $t$ appears only via dots and
  $dt = a\,d\eta$.
- **NUM 04:** conformal time is written $\tau$ (not $\eta$); redshift $z$ is the integration
  variable. Convention stated (p.1): "our usual formulae assume $\tau \to 0$ as $z \to \infty$".
- **NUM 08, NUM 09:** redshift $z$ throughout.
- $\log(1+z)$ appears only inside $\epsilon = d\ln H / d\ln(1+z)$ (NUM 01 p.3); it is not used
  as an independent variable.

### 2.2 Meaning of a prime

- **doc 12:** prime $= d/d\eta$ (stated implicitly through $\mathcal{H} = a'/a$ with
  $dt = a\,d\eta$, doc 12 p.3). Dot $= d/dt$ ($H = \dot a/a$, doc 12 p.3).
- **NUM 01:** prime $= d/d\eta$ in Step 1 (p.1, "$\mathcal{H} = a'/a = \frac1a \frac{da}{d\eta}$");
  dot $= d/dt$ (p.1). After Step 2 derivatives are written out as $d/dz$.
- **NUM 08, NUM 09:** prime $= d/dz$ — **inferred**, never stated. Evidence: NUM 08 p.2 writes
  $P'/P$ for the quantity obtained from $\frac{2}{P}\frac{dP}{dz}$; $\epsilon'$, $c_s'$, $w'$,
  $w''$, $\epsilon''$ (NUM 08 p.2–3, NUM 09 p.3–7) are all $z$-derivatives by context;
  NUM 09 p.4 writes "$c_s^2 = w(z)$ so $2 c_s c_s' = w'(z)$".
- **NUM 04:** no primes on functions; $z'$ is a dummy integration variable only (p.1–2).

### 2.3 $\mathcal{H}$ vs $H$; $\epsilon$

- $\mathcal{H} = a'/a$ (conformal Hubble rate), $H = \dot a/a$; relations $H = a'/a^2 =
  \mathcal{H}/a$ (doc 12 p.3) and $\mathcal{H} = aH = \dot a$ (NUM 01 p.1).
- $\epsilon \equiv -\dot H/H^2$ "as usual" (NUM 01 p.1; NUM 08 p.1). Equivalent forms used:
  $\mathcal{H}' = a^2H^2(1-\epsilon)$ (NUM 01 p.1); $\epsilon = \frac{1+z}{H}\frac{dH}{dz} =
  \frac{d\ln H}{d\ln(1+z)}$ (NUM 01 p.3; NUM 08 p.2; NUM 09 p.2, p.5). Note $\epsilon > 0$ for
  a decelerating universe with this sign.
- doc 12 does not use $\epsilon$; for constant $w$ it uses $\mathcal{H} = (1+b)/\eta$,
  $\mathcal{H}' = -(1+b)/\eta^2$ (doc 12 p.4).

### 2.4 Scale-factor normalisation

- **doc 12:** no $a_0$. Constant-$w$ background written $a \propto \eta^{1+b}$; the overall
  normalisation is never fixed and drops out of the $\phi$ equation (doc 12 p.4).
- **NUM 01:** $a_0$ appears explicitly through $1+z = a_0/a$ (p.1). In the final equation
  (p.3) the gradient term is written $\frac{c_s^2}{(1+z)^2 a^2 H^2}\nabla^2\phi$ with $a = a(z)$
  (not $a_0$). $a_0 = 1$ is **not** assumed.
- **NUM 04:** $a_0$ explicit: $1+z = a_0/a$, and all conformal-time results are for the product
  $a_0\tau$ (p.1–2). $a_0 = 1$ is not assumed.
- **NUM 08:** the $k^2$ term is written $\frac{k^2 c_s^2}{(1+z)^2 a^2 H^2}$ with $a = a(z)$
  (p.1, p.3) — same as NUM 01.
- **NUM 09:** Steps 1–3 use $a^2H^2$ as NUM 08; in Step 5 (p.5) the author substitutes
  $a = a_0/(1+z)$ so the $k^2$ term becomes $\frac{c_s^2 k^2}{a_0^2 H^2}$ with **no**
  $(1+z)^{-2}$ prefactor, and $a_0$ appears explicitly thereafter (p.5–6).

### 2.5 Green's function

Not used in any Group 1 document. (No source sign, argument or boundary-condition convention to
record.)

### 2.6 Equation of state and derived shorthand

- $\delta p = c_s^2\,\delta\rho = w\,\delta\rho$ (doc 12 p.2; NUM 01 p.4). So in this group
  **$c_s^2$ means $w$** (adiabatic sound speed for constant $w$; NUM 01 p.4 says explicitly
  "from now on, avoid questions about the meaning of $c_s^2$ by using $w$ instead").
  NUM 09 p.4 sets $c_s^2 = w(z)$ for the $z$-dependent mixture.
- $1+b = \frac{2}{1+3w}$ and $b = \frac{1-3w}{1+3w}$ (doc 12 p.3–4; "matches Eq. (2.16) of
  Domènech").
- $1+3w = \frac{2}{1+b}$, $\;w = \frac{1}{3}\frac{1-b}{1+b}$, $\;1+w = \frac{2(2+b)}{3(1+b)}$
  (doc 12 p.4). Since $c_s^2 = w$ this is the README's $c_s^2 = (1-b)/(3(1+b))$.
- $w(z)$ for a matter + radiation + $\Lambda$ background: see R16 (NUM 01 p.4) and R28
  (NUM 09 p.4, where $\Lambda$ and $w_m$ are dropped).
- $\rho \propto a^{-3(1+w)}$ for constant $w$ (doc 12 p.3).
- Planck mass: $3H^2 M_P^2 = \rho$ (doc 12 p.2; NUM 01 p.4; NUM 04 p.1), i.e. $M_P$ is the
  reduced Planck mass.

### 2.7 Fourier and power-spectrum conventions

- Only the replacement $\nabla^2 \to -k^2$ is used: doc 12 p.4 writes $+c_s^2 k^2\phi$ where
  p.2 had $-c_s^2\nabla^2\phi$; NUM 08 p.1 writes $+\frac{k^2 c_s^2}{(1+z)^2 a^2H^2}\phi$ where
  NUM 01 p.3 had $-\frac{c_s^2}{(1+z)^2a^2H^2}\nabla^2\phi$. Fourier normalisation
  ($(2\pi)^3$ factors), $\delta$-function normalisation, power-spectrum definitions and
  polarisation tensors do **not** appear in this group.
- Metric perturbations are written $\phi$ (time–time) and $\Psi$ (space–space), and the
  traceless $ij$ equation gives $\Psi = \phi$ (doc 12 p.1). Signature is **inferred** mostly-plus
  from $G^\eta{}_\eta = -\delta\rho/M_P^2$ and $G^i{}_j = +\delta p\,\delta^i_j/M_P^2$ (doc 12
  p.1). The sign conventions for $\phi,\Psi$ in the line element are not written in doc 12
  (they belong to MAIN 01–08, not transcribed).

### 2.8 Which transfer function, and its early-time normalisation

- The variable evolved is the Newtonian potential $\phi$ (title of doc 12 says "$\Phi$", body
  uses lower-case $\phi$ throughout; NUM 01/08/09 use $\phi$). Not $\zeta$, not $\mathcal{R}$.
  **Author note (2026-09-07, Tier 1.4):** consistent with the rest of the project. The seed of the one-loop
  calculation is $\zeta^*$ (`MAIN` 14 p. 1, `NUM` 03 p. 4), and $\phi^*$ here is the early-time potential
  $\phi^* = \tfrac{3(1+w^*)}{5+3w^*}\zeta^*$, so $T_k = \phi_k/\phi^*_k \to 1$ is the transfer function the code
  integrates and the initial spectrum fed in is $P_\zeta$ (from `PyTransport`/`CppTransport`). See spec 03 §0.4.
- Early-time normalisation (doc 12 p.6–7): the $Y$ solution is projected out and the $J$
  solution is normalised so that $\phi_k(\eta) \to \phi^*_k$ as $k c_s\eta \to 0$, giving R11.
- A symbol "$T_k$" or an explicit definition "transfer function $= \phi_k(\eta)/\phi^*_k$" does
  **not** appear anywhere in the five documents, despite the file name of NUM 01
  ("Tk equation in redshift"). The identification $T_k = \phi_k/\phi^*_k$ is **inferred**
  (see Q9).

### 2.9 Momentum labels

Only a single wavenumber $k$ (magnitude) appears; $\mathbf{q}$, $\mathbf{r}$ and angular
reductions do not occur in this group.

---

## 3. Results

### 3.1 doc 12 — "Compute transfer function for $\Phi$" (21 July 2022)

**R1** (doc 12 p.1, Step 1) — Linearised Einstein tensor for scalar perturbations $\phi$,
$\Psi$ (no anisotropic stress assumed yet):
$$
G^\eta{}_\eta \to \frac{1}{a^2}\bigl(6\mathcal{H}\Psi' + 6\mathcal{H}^2\phi\bigr)
  - \frac{2}{a^2}\nabla^2\Psi ,
$$
$$
G^i{}_j \to \frac{1}{a^2}\delta^i_j\Bigl\{2\mathcal{H}\phi' + 4\mathcal{H}\Psi' + 2\Psi''
  + 2\phi\bigl(2\mathcal{H}'+\mathcal{H}^2\bigr)\Bigr\}
  + \frac{1}{a^2}\delta^i_j\bigl\{\nabla^2\phi - \nabla^2\Psi\bigr\}
  + \frac{1}{a^2}\bigl(\partial^i\partial_j\Psi - \partial^i\partial_j\phi\bigr).
$$
Confidence: **medium** — in $G^\eta{}_\eta$ a term before $6\mathcal{H}\Psi'$ is struck through
and illegible (looks like "$4\mathcal{H}\Psi$"); the transcription above is the surviving text.
See §5 and Q3.

**R2** (doc 12 p.1, Step 2) — Einstein equations $G^\eta{}_\eta = -\delta\rho\,M_P^{-2}$,
$G^i{}_j = M_P^{-2}\,\delta p\,\delta^i_j$. The off-diagonal part gives
$$
\Psi = \phi ,
$$
and the $\eta\eta$ equation becomes
$$
6\mathcal{H}\phi' + 6\mathcal{H}^2\phi - 2\nabla^2\phi = -\frac{a^2\,\delta\rho}{M_P^2},
\qquad\text{or}\qquad
3\mathcal{H}\phi' + 3\mathcal{H}^2\phi - \nabla^2\phi = -\frac{a^2\,\delta\rho}{2M_P^2}.
$$
The page actually shows $+\frac{a^2\delta\rho}{2M_P^2}$ on the right of the second form; a red
annotation dated 26 Apr 2023 says the minus sign was lost and that it does *not* propagate (the
minus sign is correctly restored where the equation is used on p.2). Transcribed here with the
minus sign, as the later steps use it. Confidence: **high** (the annotation is explicit).

**R3** (doc 12 p.2) — Diagonal $ij$ equation after $\Psi = \phi$:
$$
\frac{1}{a^2}\Bigl(6\mathcal{H}\phi' + 2\phi'' + 2\phi(2\mathcal{H}'+\mathcal{H}^2)\Bigr)
 = \frac{\delta p}{M_P^2}
\qquad\Rightarrow\qquad
\phi'' + 3\mathcal{H}\phi' + \phi\,(2\mathcal{H}'+\mathcal{H}^2) = \frac{a^2\,\delta p}{2M_P^2}.
$$
Confidence: **medium** — the first form on p.2 appears to read "$=-\delta p/M_P^2$", but both
the Step 2 (B) equation on p.1 and the "so" line immediately below have a positive sign, and the
positive sign is what makes the sum in R5 vanish. See Q2.

**R4** (doc 12 p.2) — Closure:
$$
\delta p = c_s^2\,\delta\rho = w\,\delta\rho .
$$
Confidence: **high**.

**R5** (doc 12 p.2; "matches Eq. (4.3) of Domènech review") — Adding $c_s^2\times$R2 to R3
(boxed/annotated step, sign confirmed by the 26 Apr 2023 note):
$$
\phi'' + 3\mathcal{H}\bigl(1+c_s^2\bigr)\phi'
 + \Bigl\{2\mathcal{H}' + \mathcal{H}^2\bigl(1+3c_s^2\bigr)\Bigr\}\phi
 - c_s^2\nabla^2\phi = 0 .
$$
This is the $\phi$ evolution equation that NUM 01 and NUM 08/09 start from. Confidence: **high**.

**R6** (doc 12 p.3, Step 3) — Background for constant $w$. From $3H^2M_P^2 = \rho$ and
$\dot\rho + 3H(\rho+p) = 0$:
$$
\rho \propto a^{-3(1+w)}, \qquad 3\mathcal{H}^2 M_P^2 = a^2\rho \propto a^{-(1+3w)},
\qquad a' \propto a^{(1-3w)/2},
$$
$$
\eta - \eta_c \propto a^{(1+3w)/2}, \qquad a \propto (\eta-\eta_c)^{2/(1+3w)} .
$$
Examples written: matter domination $w=0 \Rightarrow a\propto(\eta-\eta_c)^2$; radiation
domination $w=1/3 \Rightarrow a\propto(\eta-\eta_c)$. Domènech writes $a\propto\eta^{1+b}$, so
$$
1 + b = \frac{2}{1+3w}.
$$
Confidence: **high**.

**R7** (doc 12 p.4; "matches Eq. (2.16) of Domènech") — Derived constant-$w$ shorthand:
$$
b = \frac{2}{1+3w} - 1 = \frac{1-3w}{1+3w}, \qquad
\mathcal{H} = \frac{a'}{a} = \frac{1+b}{\eta} = \frac{2}{1+3w}\,\frac{1}{\eta}, \qquad
\mathcal{H}' = -\frac{1+b}{\eta^2} = -\frac{2}{1+3w}\,\frac{1}{\eta^2},
$$
$$
1+3w = \frac{2}{1+b}, \qquad w = \frac{1}{3}\,\frac{1-b}{1+b}, \qquad
1+w = \frac{2(2+b)}{3(1+b)} .
$$
Confidence: **high**.

**R8** (doc 12 p.4–5, Step 4) — Constant-$w$ Newtonian-potential equation in Fourier space.
Substituting R7 into R5 (with $c_s^2 = w$ in the friction/mass terms):
$$
\phi'' + \frac{3(1+b)(1+w)}{\eta}\phi'
 + \Bigl\{-2(1+b) + (1+b)^2(1+3w)\Bigr\}\frac{\phi}{\eta^2} + c_s^2k^2\phi = 0 ,
$$
and after using R7 the $\phi/\eta^2$ coefficient is $-2(1+b) + 2(1+b) = 0$, so
"the mass term for $\phi$ vanishes, yielding"
$$
\phi'' + \frac{2(2+b)}{\eta}\phi' + c_s^2 k^2\phi = 0 .
$$
Confidence: **high**. (The term $-2(1+b)$ replaces a struck $-\tfrac{4}{1+3w}$, which is the
same quantity; see §5.)

**R9** (doc 12 p.5–6) — Removal of the first-derivative term. With $\phi = f\chi$ and
$f'/f = -(2+b)/\eta$:
$$
f = \eta^{-(2+b)}, \qquad \frac{f''}{f} = \frac{2+b}{\eta^2} + \frac{(2+b)^2}{\eta^2},
$$
$$
\chi'' + \Bigl(c_s^2k^2 - \frac{(2+b)(1+b)}{\eta^2}\Bigr)\chi = 0 .
$$
(The page notes "constant is irrelevant (any solution will do)" for the integration constant in
$f$.) Confidence: **high**.

**R10** (doc 12 p.6; "matches Eq. (4.4) of Domènech") — General solution:
$$
\chi = \alpha\sqrt{\eta}\,J_{b+3/2}(kc_s\eta) + \beta\sqrt{\eta}\,Y_{b+3/2}(kc_s\eta),
$$
$$
\phi = (kc_s\eta)^{-3/2-b}\Bigl\{\alpha\,J_{3/2+b}(kc_s\eta) + \beta\,Y_{3/2+b}(kc_s\eta)\Bigr\}.
$$
Bessel order is $3/2 + b$ (written "$b+\tfrac32$" on the first line and "$\tfrac32+b$" on the
second). Note the constants $\alpha,\beta$ are implicitly redefined between the two lines
(a factor $(kc_s)^{-3/2-b}$ is absorbed). Confidence: **high** for the order $3/2+b$ — the
index $5/2+b$ appears only inside the Gamma function in R11, not as a Bessel order.

**R11** (doc 12 p.6–7, Step 5) — Match to the inflationary initial condition. As
$kc_s\eta\to 0$,
$$
(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta) \to \frac{2^{-3/2-b}}{\Gamma(\tfrac52+b)} + O(kc_s\eta),
$$
while $(kc_s\eta)^{-3/2-b}Y_{3/2+b}(kc_s\eta)$ diverges and is projected out
($\beta = 0$). Requiring $\phi\to\phi^*_{\mathbf k}$ gives
$\alpha\,2^{-3/2-b}/\Gamma(\tfrac52+b) = \phi^*_{\mathbf k}$, hence
$$
\boxed{\;
\phi_{\mathbf k}(\eta) = 2^{3/2+b}\,\Gamma\!\bigl(\tfrac52+b\bigr)\,\phi^*_{\mathbf k}\,
 (kc_s\eta)^{-3/2-b}\,J_{3/2+b}(kc_s\eta)\; }
$$
This is the final result of doc 12 (p.7; the rest of p.7 is blank). Confidence: **high** for
p.7. **Medium** for the two limit lines on p.6, where the exponent is written "$-\tfrac32 \mp b$"
with the sign in front of $b$ over-written; the p.7 result and the preceding p.6 line both have
$-\tfrac32 - b$, which is what is transcribed. See §5.

### 3.2 NUM 01 — "Transfer function in terms of redshift" (10 September 2024)

Starts (p.1, Step 1) from R5 verbatim, with $\mathcal{H} = a'/a = \frac1a\frac{da}{d\eta}$,
$dt = a\,d\eta$.

**R12** (NUM 01 p.1, Step 2) — Change of variable to redshift, $1+z = a_0/a$:
$$
dz = -\frac{a_0}{a}\frac{\dot a}{a}\,dt = -(1+z)H\,dt = -(1+z)\,aH\,d\eta,
\qquad
\frac{d}{d\eta} = -(1+z)\,aH\,\frac{d}{dz}.
$$
Confidence: **high**.

**R13** (NUM 01 p.1) — Conformal Hubble rate and its derivative in terms of $H$, $\epsilon$:
$$
\mathcal{H} = \frac{a'}{a} = \dot a = aH, \qquad
\mathcal{H}' = a\frac{d\mathcal{H}}{dt} = a(\dot a H + a\dot H) = a^2H^2\Bigl(1+\frac{\dot H}{H^2}\Bigr)
 = a^2H^2(1-\epsilon), \qquad \epsilon \equiv -\frac{\dot H}{H^2}.
$$
Confidence: **high**.

**R14** (NUM 01 p.3) — The $\phi$ equation in redshift (two equivalent forms; the second is the
one carried into NUM 08/09):
$$
(1+z)^2(aH)^2\frac{d^2\phi}{dz^2} + (1+z)(aH)^2\Bigl\{\epsilon - 3(1+c_s^2)\Bigr\}\frac{d\phi}{dz}
 + (aH)^2\Bigl\{3(1+c_s^2) - 2\epsilon\Bigr\}\phi - c_s^2\nabla^2\phi = 0,
$$
$$
\boxed{\;
\frac{d^2\phi}{dz^2} + \frac{1}{1+z}\Bigl\{\epsilon - 3(1+c_s^2)\Bigr\}\frac{d\phi}{dz}
 + \frac{1}{(1+z)^2}\Bigl\{3(1+c_s^2) - 2\epsilon\Bigr\}\phi
 - \frac{c_s^2}{(1+z)^2\,a^2H^2}\nabla^2\phi = 0 .\;}
$$
Here $a = a(z) = a_0/(1+z)$ and $H = H(z)$. Confidence: **high**. (p.2 and the top of p.3 are
intermediate algebra and are not transcribed; a struck factor on p.3 is noted in §5.)

**R15** (NUM 01 p.3) — $\epsilon$ in redshift:
$$
\epsilon = -\frac{\dot H}{H^2} = -\frac{1}{H^2}\Bigl(-(1+z)H\frac{dH}{dz}\Bigr)
 = \frac{1+z}{H}\frac{dH}{dz} = \frac{d\ln H}{d\ln(1+z)} .
$$
Confidence: **high**.

**R16** (NUM 01 p.4, Step 3) — Background equation of state for matter + radiation + $\Lambda$.
With $3H^2M_P^2 = \rho$, $p = w\rho$, $p = p_m + p_r$, $\rho = \rho_m + \rho_r + \rho_\Lambda$:
$$
w(z) = \frac{p}{\rho} = \frac{w_m\rho_m + w_r\rho_r}{\rho_m+\rho_r+\rho_\Lambda}
 = \frac{w_m\,\rho_{m,0}(1+z)^3 + w_r\,\rho_{r,0}(1+z)^4}
        {\rho_{m,0}(1+z)^3 + \rho_{r,0}(1+z)^4 + \rho_\Lambda}
 = \frac{w_m\,\Omega_{m,0}(1+z)^3 + w_r\,\Omega_{r,0}(1+z)^4}
        {\Omega_{m,0}(1+z)^3 + \Omega_{r,0}(1+z)^4 + \Omega_{\rm cc}} .
$$
The page states that $c_s^2$ "in this calculation has the meaning of $\delta p = c_s^2\delta\rho$,
i.e. $\delta p = w\,\delta\rho$", and that $w$ will be used instead of $c_s^2$ from now on.
Confidence: **medium** — the last denominator term reads "$\Omega_{\rm cc}$" (presumably the
cosmological-constant density parameter, $=\Omega_\Lambda$); the radiation exponent in the last
numerator has a "3" over-written to "4" (see §5). $w_m$, $w_r$ are left symbolic (not set to
$0$, $1/3$).

**R17** (NUM 01 p.5) — Continuity equation for the total density:
$$
\frac{d\rho}{dt} = -3H(1+w_m)\rho_m - 3H(1+w_r)\rho_r - 3H(1+w_\Lambda)\rho_\Lambda
 = -3H\,\frac{(1+w_m)\rho_m + (1+w_r)\rho_r + (1+w_\Lambda)\rho_\Lambda}{\rho_m+\rho_r+\rho_\Lambda}\,\rho
 = -3H\bigl(1+w(z)\bigr)\rho ,
$$
"where $w(z)$ has the same expression obtained from $w = p/\rho$." Confidence: **high** for the
reading; see Q4 for an apparent inconsistency with R16.

### 3.3 NUM 04 — "Initial condition for conformal time" (20 Sep 2024)

**R18** (NUM 04 p.1, Step 1) — Relation between $d\tau$ and $dz$, with $dt = a\,d\tau$ and
$a_0/a = 1+z$:
$$
dz = -(1+z)H\,dt = -(1+z)\,aH\,d\tau = -a_0 H\,d\tau,
\qquad
a_0\bigl(\tau - \tau_{\rm init}\bigr) = -\int_{z_{\rm init}}^{z}\frac{dz'}{H(z')} .
$$
Confidence: **high**.

**R19** (NUM 04 p.1, Step 2) — Convention $\tau\to 0$ as $z\to\infty$:
$$
a_0\,\tau = -\int_{\infty}^{z}\frac{dz'}{H(z')} .
$$
Confidence: **high**.

**R20** (NUM 04 p.1–2) — Radiation domination at and before the initial time $\tau_1
\leftrightarrow z_1$: $3H^2M_P^2 = \rho = \rho_1(1+z)^4/(1+z_1)^4$, so
$$
H = \Bigl(\frac{\rho_1}{3M_P^2}\Bigr)^{1/2}\frac{(1+z)^2}{(1+z_1)^2},
$$
$$
a_0\tau_1 = -\int_\infty^{z_1} dz\,\Bigl(\frac{3M_P^2}{\rho_1}\Bigr)^{1/2}\frac{(1+z_1)^2}{(1+z)^2}
 = -\Bigl(\frac{3M_P^2}{\rho_1}\Bigr)^{1/2}(1+z_1)^2\Bigl[-\frac{1}{1+z}\Bigr]_\infty^{z_1}
 = \Bigl(\frac{3M_P^2}{\rho_1}\Bigr)^{1/2}(1+z_1) .
$$
"Despite appearances this does decay as $z_1\to\infty$, so that $\tau_1\to 0$, because
$\rho_1\propto(1+z_1)^4$. Hence $a_0\tau_1 \propto \frac{1}{1+z_1}\to 0$ as $z_1\to\infty$."
Confidence: **medium** — the exponent on $(\rho_1/3M_P^2)$ on p.1 is over-written (reads
"$1/2$" after correction); on p.2 the middle line has what looks like "$\rho_2$" where the
lines above and below have $\rho_1$ (taken as a slip), and the subscript on $\tau$ in the
first line of p.2 could be read as $1$ or $2$ (taken as $\tau_1$, consistent with the limit
$z_1$). See Q5.

### 3.4 NUM 08 — "Liouville–Green (WKB) phase function representation for the transfer function" (7 Dec 2024)

**R21** (NUM 08 p.1, Step 1) — Starting equation, R14 in Fourier space ($\nabla^2\to-k^2$):
$$
\frac{d^2\phi}{dz^2} + \frac{1}{1+z}\Bigl\{\epsilon - 3(1+c_s^2)\Bigr\}\frac{d\phi}{dz}
 + \frac{1}{(1+z)^2}\Bigl\{3(1+c_s^2) - 2\epsilon\Bigr\}\phi
 + \frac{k^2c_s^2}{(1+z)^2\,a^2H^2}\,\phi = 0 ,
\qquad \epsilon = -\frac{\dot H}{H^2}\ \text{"as usual"}.
$$
Confidence: **high**.

**R22** (NUM 08 p.1, Step 2) — Ansatz $\phi = P\vartheta$. The equation for $\vartheta$ before
choosing $P$:
$$
\frac{d^2\vartheta}{dz^2}
 + \Bigl\{\frac{2}{P}\frac{dP}{dz} + \frac{1}{1+z}\bigl[\epsilon-3(1+c_s^2)\bigr]\Bigr\}\frac{d\vartheta}{dz}
 + \Bigl\{\frac{1}{P}\frac{d^2P}{dz^2} + \frac{1}{P}\frac{dP}{dz}\frac{1}{1+z}\bigl[\epsilon-3(1+c_s^2)\bigr]
 + \frac{1}{(1+z)^2}\Bigl[3(1+c_s^2) - 2\epsilon + \frac{k^2c_s^2}{a^2H^2}\Bigr]\Bigr\}\vartheta = 0 .
$$
Confidence: **high**.

**R23** (NUM 08 p.2) — Friction elimination and the integrating factor $P$. Choosing
$$
\frac{2}{P}\frac{dP}{dz} = -\frac{1}{1+z}\Bigl\{\epsilon - 3(1+c_s^2)\Bigr\}
\quad\Rightarrow\quad
d\ln P = -\frac12\frac{dz}{1+z}\bigl(\epsilon - 3(1+c_s^2)\bigr)
 = -\frac12\,d\ln H + \frac32(1+c_s^2)\frac{dz}{1+z}
$$
(using $\epsilon = (1+z)\,d\ln H/dz$), so
$$
\ln P = \ln H^{-1/2} + \frac32\int(1+c_s^2)\frac{dz'}{1+z'} + \text{const},
\qquad
\boxed{\;P = P_0\Bigl(\frac{H_{\rm init}}{H}\Bigr)^{1/2}
 \exp\Bigl\{\frac32\int_{z_{\rm init}}^{z}(1+c_s^2)\frac{dz'}{1+z'}\Bigr\}.\;}
$$
Confidence: **high**.

**R24** (NUM 08 p.2) — Derivatives of $P$ (prime $= d/dz$):
$$
\frac{P'}{P} = -\frac12\frac{1}{1+z}\Bigl\{\epsilon - 3(1+c_s^2)\Bigr\},
\qquad
\frac{P''}{P} - \frac{P'}{P}\frac{P'}{P}
 = \frac12\frac{1}{(1+z)^2}\Bigl\{\epsilon - 3(1+c_s^2)\Bigr\}
 - \frac12\frac{1}{1+z}\Bigl\{\epsilon' - 6c_sc_s'\Bigr\}.
$$
Confidence: **high** (a struck glyph precedes the $\tfrac12$ in $P'/P$; the surviving sign is
minus, consistent with R23).

**R25** (NUM 08 p.3) — Normal-form equation for $\vartheta$ (final result of NUM 08):
$$
\frac{d^2\vartheta}{dz^2} + \Bigl\{\frac{3c_sc_s' - \epsilon'/2}{1+z}
 + \frac{1}{(1+z)^2}\Bigl[-\frac32\epsilon + \frac32(1+c_s^2) - \frac{\epsilon^2}{4}
   + \frac32\epsilon(1+c_s^2) - \frac94(1+c_s^2)^2 + \frac{k^2c_s^2}{a^2H^2}\Bigr]\Bigr\}\vartheta = 0 .
$$
Intermediate form written on the same page (used by the check in NUM 09):
$$
\frac{d^2\vartheta}{dz^2} + \Bigl\{\frac{3c_sc_s' - \epsilon'/2}{1+z}
 + \frac{1}{(1+z)^2}\Bigl[\frac{\epsilon}{2} - \frac32(1+c_s^2)
   - \frac14\bigl(\epsilon^2 - 6\epsilon(1+c_s^2) + 9(1+c_s^2)^2\bigr)
   + 3(1+c_s^2) - 2\epsilon + \frac{k^2c_s^2}{a^2H^2}\Bigr]\Bigr\}\vartheta = 0 .
$$
Confidence: **high** for the two displayed forms. **Medium** for the first (assembled) line at
the top of p.3: its third term reads $-\frac12\frac{1}{(1+z)^2}[\epsilon - 3(1+c_s^2)]$ with no
visible square, whereas the next line (and NUM 09 p.3) treats it as
$[\epsilon-3(1+c_s^2)]^2$; a digit inside that bracket is also over-written ("3" over "5" or
similar). See Q6.

### 3.5 NUM 09 — new material beyond the check (8 Dec 2024)

Steps 1–3 of NUM 09 re-derive R21–R25; see §4. Steps 2–5 also contain the following results not
present in NUM 08.

**R26** (NUM 09 p.2) — Integrating factor without the arbitrary constant ("constant does not
matter for computation of an integrating factor"), and with the integral written both ways:
$$
P = \Bigl(\frac{H_{\rm init}}{H}\Bigr)^{1/2}\exp\Bigl[\frac32\int_{z_{\rm init}}^{z}(1+c_s^2)\frac{dz'}{1+z'}\Bigr]
  = \Bigl(\frac{H_{\rm init}}{H}\Bigr)^{1/2}\exp\Bigl[-\frac32\int_{z}^{z_{\rm init}}(1+c_s^2)\frac{dz'}{1+z'}\Bigr].
$$
Confidence: **high**.

**R27** (NUM 09 p.3–4, Step 3) — Effective frequency. Definition (p.3):
$$
\omega_{\rm eff}^2 = \frac{P''}{P} + \frac{P'}{P}\frac{1}{1+z}\Bigl\{\epsilon-3(1+c_s^2)\Bigr\}
 + \frac{1}{(1+z)^2}\Bigl\{3(1+c_s^2) - 2\epsilon + \frac{c_s^2k^2}{a^2H^2}\Bigr\},
$$
compact final form (p.4; the $k^2$ term is written in the margin and inserted by an arrow):
$$
\boxed{\;
\omega_{\rm eff}^2 = \frac{1}{1+z}\Bigl(3c_sc_s' - \frac{\epsilon'}{2}\Bigr)
 + \frac{1}{(1+z)^2}\Bigl(\frac32(1+\epsilon)(1+c_s^2) - \frac{\epsilon}{2}\Bigl(3+\frac{\epsilon}{2}\Bigr)
   - \frac94(1+c_s^2)^2 + \frac{c_s^2k^2}{a^2H^2}\Bigr).\;}
$$
Confidence: **high** for the form; **medium** on the placement of the $k^2$ term, which is a
marginal insertion (arrow points into the $(1+z)^{-2}$ bracket, consistent with R21 and R25).

**R28** (NUM 09 p.4, Step 4) — Sound-speed derivative for a matter + radiation background
($\Lambda$ omitted, $w_m$ set to zero here):
$$
c_s^2 = w(z), \qquad 2c_sc_s' = w'(z),
$$
$$
w(z) = \frac{w_r\Omega_r(1+z)^4}{\Omega_m(1+z)^3 + \Omega_r(1+z)^4} = \frac{w_r\Omega_r(1+z)}{\Omega_m + \Omega_r(1+z)},
\qquad
w'(z) = \frac{w_r\Omega_r\Omega_m}{\bigl[\Omega_m + \Omega_r(1+z)\bigr]^2}.
$$
Confidence: **high**. Note $\Omega_m,\Omega_r$ here carry no ",0" subscript (cf. R16).

**R29** (NUM 09 p.5, Step 5) — $\omega_{\rm eff}^2$ in terms of $w$, with $a = a_0/(1+z)$ made
explicit so the $(1+z)^{-2}$ cancels in the $k^2$ term:
$$
\omega_{\rm eff}^2(z) = \frac{1}{1+z}\,\frac12\bigl(3w' - \epsilon'\bigr)
 + \frac{1}{(1+z)^2}\Bigl(\frac32(1+\epsilon)(1+w) - \frac{\epsilon}{2}\Bigl(3+\frac{\epsilon}{2}\Bigr) - \frac94(1+w)^2\Bigr)
 + \frac{w}{H^2}\,\frac{k^2}{a_0^2}.
$$
(In the last term "$c_s^2$" is over-written by "$w$".) Confidence: **high**.

**R30** (NUM 09 p.5–6) — Derivative of the effective frequency. Final form (p.6, using
$\frac{1}{H}\frac{dH}{dz} = \frac{\epsilon}{1+z}$):
$$
2\omega_{\rm eff}\frac{d\omega_{\rm eff}}{dz}
 = \frac{w'}{H^2}\frac{k^2}{a_0^2}
 + \frac{1}{1+z}\Bigl(\frac32 w'' - \frac12\epsilon'' - 2\epsilon\,\frac{w}{H^2}\frac{k^2}{a_0^2}\Bigr)
 + \frac{1}{(1+z)^2}\Bigl(\frac{\epsilon'}{2}\bigl(3w - \epsilon + 1\bigr)
   + \frac32 w'\Bigl(\epsilon - \frac32(1+w)\Bigr)\Bigr)
 - \frac{2}{(1+z)^3}\Bigl(\frac32(1+\epsilon)(1+w) - \frac{\epsilon}{2}\Bigl(3+\frac{\epsilon}{2}\Bigr) - \frac94(1+w)^2\Bigr).
$$
The unsimplified form on p.5 has, inside the $(1+z)^{-2}$ bracket,
$\frac32\epsilon'(1+w) + \frac32(1+\epsilon)w' - \frac{\epsilon'}{2}(3+\epsilon) - \frac{\epsilon\epsilon'}{4} - \frac92(1+w)w'$
(plus the separate $-\frac{1}{(1+z)^2}\frac{3w'-\epsilon'}{2}$), whereas the p.6 intermediate
line has $\frac32\epsilon'(1+w) + \frac32 w'(1+\epsilon) - \frac{\epsilon'}{2}(3+\epsilon) - \frac94 w'(1+w) - \frac32 w' + \frac{\epsilon'}{2}$
(no $\epsilon\epsilon'/4$ term; $\frac94$ instead of $\frac92$). The boxed final line above
follows the p.6 version. Confidence: **high** for the reading of each line; **the two pages
disagree with each other** — see Q7. In "$(3w - \epsilon + 1)$" a glyph after $\epsilon$ is
struck.

**R31** (NUM 09 p.7) — Second derivative of $w(z)$ for the R28 background:
$$
w''(z) = -\frac{2w_r\Omega_r\Omega_m\,\Omega_r}{\bigl[\Omega_m+\Omega_r(1+z)\bigr]^3}
 = -\frac{2w_r\Omega_r^2\Omega_m}{\bigl[\Omega_m+\Omega_r(1+z)\bigr]^3}.
$$
Confidence: **high**. The rest of p.7 is blank.

---

## 4. Checks

NUM 09 (8 Dec 2024) rechecks NUM 08 (7 Dec 2024), Steps 1–3. Comparison by the transcriber
unless stated otherwise.

- **R21 (starting ODE):** NUM 09 p.1 writes the identical equation (with the $k^2$ term as
  $c_s^2k^2/(a^2H^2)$ inside the $(1+z)^{-2}$ bracket). Agrees.
- **R22 ($\vartheta$ equation before choosing $P$):** NUM 09 p.1 bottom, same coefficients.
  Agrees.
- **R23 (friction condition and $P$):** NUM 09 p.2, same condition
  $\frac{2P'}{P} + \frac{1}{1+z}\{\epsilon-3(1+c_s^2)\} = 0$, same $d\ln P$, same $\ln P$. The
  recheck drops the constant $P_0$ and adds the reversed-limit form (R26). Agrees.
- **R24 ($P''/P - (P'/P)^2$):** NUM 09 p.3, identical expression. Agrees.
- **R25 ($\omega_{\rm eff}^2$ / normal form):** NUM 09 p.3–4 reproduces the expanded bracket
  $\frac32(1+c_s^2) - \frac32\epsilon - \frac{\epsilon^2}{4} + \frac32\epsilon(1+c_s^2) - \frac94(1+c_s^2)^2$
  term by term, then regroups it into R27. The author writes "✓ matches 7 Dec 2024
  calculation" (NUM 09 p.4). Transcriber comparison: agrees term by term. In NUM 09 p.3 the
  term that NUM 08 p.3 shows without a visible square is written explicitly as
  $-\frac12\frac{1}{(1+z)^2}\{\epsilon - 3(1+c_s^2)\}^2$, which resolves Q6 in favour of the
  squared reading.

No discrepancy between NUM 08 and NUM 09 was noted by the author. The only internal discrepancy
found is *within* NUM 09 (p.5 vs p.6, Q7), which the author did not flag.

---

## 5. Corrections and cross-outs

**doc 12**
- p.1, $G^\eta{}_\eta$: a term before "$6\mathcal{H}\Psi'$" is struck out (looks like
  "$4\mathcal{H}\Psi$"); the surviving expression is R1. Later steps use the struck-free form.
- p.1, bottom: the rewritten $\eta\eta$ equation has $+\frac{a^2\delta\rho}{2M_P^2}$ (boxed in
  red). Red annotation, 26 Apr 2023: "apparently lost minus sign from previous expression, but
  this does not propagate (where this expression is used below, the RHS is correctly taken to
  have a minus sign)". R2 records the corrected sign.
- p.2: the line "$3\mathcal{H}c_s^2\phi' + 3\mathcal{H}^2c_s^2\phi - c_s^2\nabla^2\phi =
  -\frac{a^2}{2M_P^2}c_s^2\delta\rho$" is boxed in red, with "$\underbrace{\;}_{\delta p}$"
  under $c_s^2\delta\rho$. Red annotation, 26 Apr 2023: "sign here has gone back to being
  correct; probably forgot to correct the second equation above when the sign error was
  spotted." Later steps (R5) use the corrected sign.
- p.4, Step 4: in the mass term the original "$-\frac{4}{1+3w}$" is struck through and replaced
  above by "$-2(1+b)$" (identical quantity since $1+b = 2/(1+3w)$). Red strokes on the last
  line of p.4 cancel $3(1+b)$ against $\frac{2(2+b)}{3(1+b)}$ in the friction term and
  $-2(1+b)$ against $(1+b)^2\frac{2}{1+b}$ in the mass term; result R8.
- p.4, first line: a word/symbol before "$\frac{2}{1+3w}$" in "$b = \ldots$" is struck.
- p.6, Step 5: in "$(kc_s\eta)^{-3/2\mp b}$" (two occurrences) the sign before $b$ is
  over-written. The p.7 final result has $-3/2-b$; R11 uses that.

**NUM 01**
- p.3, first display: a fraction after "$(1+\dot H/H^2)$" is struck through (intermediate step,
  not transcribed).
- p.4: in the $\Omega$ form of $w(z)$, the numerator radiation exponent has "3" over-written to
  "4"; R16 records "4", matching the $\rho$ form on the preceding line.

**NUM 04**
- p.1: in "$-\frac{a_0}{a}\frac{\dot a}{a}dt = \ldots dz$" a "1+" (or similar) written before
  $dz$ is struck; the surviving equation is $dz = -(1+z)H\,dt$ (R18).
- p.1, last line: the exponent on $(\rho_1/3M_P^2)$ is over-written; read as $1/2$
  (R20).
- p.2: "$\rho_2$" in the middle line where $\rho_1$ is expected (not struck; treated as a slip,
  Q5).

**NUM 08**
- p.1: a word before "where $\epsilon = \ldots$" is struck (illegible; probably "or"/"for").
- p.1, third display: a struck symbol between "$-2\epsilon$" and "$+\frac{k^2c_s^2}{a^2H^2}$".
- p.2, "Also $P'/P = \ldots$": a struck glyph before "$\frac12$"; surviving sign is $-$ (R24).
- p.3, first display: a digit in "$[\epsilon - 3(1+c_s^2)]$" of the third term is over-written
  ("3" over "5" or vice versa); read as 3. Also see Q6 (missing square).

**NUM 09**
- p.1: small marks/strikes after "$\vartheta'P$" and "$\vartheta''P$" in the $\phi'$, $\phi''$
  lines (look like a superscript struck out; no effect on content).
- p.2: "$=0$" struck after "$d\ln P = -\frac12\frac{dz}{1+z}\{\epsilon-3(1+c_s^2)\}$".
- p.4: the $k^2$ term "$\frac{c_s^2k^2}{a^2H^2}$" is written in the right margin twice with
  arrows into the $(1+z)^{-2}$ bracket (R27).
- p.5: "$c_s^2$" over-written by "$w$" in the $k^2$ term of R29; a struck glyph after
  "$(1+w)^2$" in the first display; a struck glyph after "$\epsilon$" in "$(3+\epsilon)$" of
  the $2\omega_{\rm eff}\,d\omega_{\rm eff}/dz$ display.
- p.6: a struck glyph after "$\epsilon$" in "$(3w-\epsilon+1)$".

---

## 6. Open questions

**Q1 — Page counts.** The README and the assignment give doc 12 as 9 pages and NUM 01 as 7
pages; the PDF reader reports 7 and 5 pages respectively (a request for pages 8–9 / 6–7 is
rejected as outside the document). A raw count of `/Type /Page` objects in the files gives 9
and 7, so the files may contain unreferenced/duplicate page objects, or the README counts came
from that raw count. The last extracted page of each document carries the final result and is
otherwise blank (doc 12 p.7; NUM 01 p.5 ends with R17), so no content appears to be missing, but
the author should confirm.

**Q2 — Sign of the $ij$ equation, doc 12 p.2 (R3).** The first display on p.2 appears to read
"$\frac{1}{a^2}(6\mathcal{H}\phi' + 2\phi'' + 2\phi(2\mathcal{H}'+\mathcal{H}^2)) = -\delta p/M_P^2$",
whereas the p.1 version has $+M_P^{-2}\delta p\,\delta^i_j$ and the very next line has
$+\frac{a^2\delta p}{2M_P^2}$. The positive sign is what R5 requires. Transcribed as positive;
the apparent minus may be a stray stroke or part of the "=" glyph.

**Q3 — Struck term in $G^\eta{}_\eta$, doc 12 p.1 (R1).** Illegible struck term; the intended
final expression is unambiguous from Step 2, but the original is not recoverable from the scan.

**Q4 — $w(z)$ with and without $\Lambda$ pressure (R16 vs R17).** R16 defines $w(z) = p/\rho$
with $p = p_m + p_r$ (no $p_\Lambda$). R17 then states that the continuity equation gives
$-3H(1+w(z))\rho$ "where $w(z)$ has the same expression". Taken literally, the $\rho_\Lambda$
term in R17 contributes $(1+w_\Lambda)\rho_\Lambda$ to the numerator, which is absent from R16
unless $w_\Lambda = -1$ **and** $p_\Lambda = -\rho_\Lambda$ is included in $p$. Whether R16's $w$
(intended as the perturbation closure $\delta p = w\,\delta\rho$, where $\Lambda$ does not
perturb) and R17's $w$ (background) are meant to be the same function is not resolved on the
page. Not a transcription ambiguity; flagged as an apparent inconsistency.

**Q5 — Subscripts on NUM 04 p.2.** "$\rho_2$" appears once where $\rho_1$ is expected; the
subscript on $\tau$ in the first line of p.2 could be 1 or 2. Both taken as "1" (only one
initial time $\tau_1\leftrightarrow z_1$ is defined on p.1).

**Q6 — Missing square, NUM 08 p.3 first display (R25).** The third term inside the braces is
written $-\frac12\frac{1}{(1+z)^2}[\epsilon - 3(1+c_s^2)]$ with no visible exponent, but the
following line expands it as if squared, and NUM 09 p.3 writes the square explicitly. Almost
certainly a dropped exponent on the page; R25's displayed forms are unaffected.

**Q7 — Internal discrepancy in NUM 09 Step 5 (R30), p.5 vs p.6.** Two coefficients change
between the unsimplified line on p.5 and the regrouped line on p.6:
(i) p.5 has $-\frac92(1+w)w'$ (the $z$-derivative of $-\frac94(1+w)^2$); p.6 has
$-\frac94 w'(1+w)$, and the boxed final line on p.6, $\frac32 w'(\epsilon - \frac32(1+w))$,
follows the $\frac94$ value. (ii) p.5 has $-\frac{\epsilon'}{2}(3+\epsilon) - \frac{\epsilon\epsilon'}{4}$;
p.6 has only $-\frac{\epsilon'}{2}(3+\epsilon)$, and the final $\frac{\epsilon'}{2}(3w-\epsilon+1)$
follows the p.6 version. The author did not mark either change as a correction. Transcribed
both readings; the final form (R30) is the p.6 one. This needs the author's decision before R30
is used as a build target.

**Q8 — $a$ vs $a_0$ in the $k^2$ term.** NUM 01 (R14) and NUM 08 (R21, R25) write
$k^2c_s^2/((1+z)^2a^2H^2)$ with $a = a(z)$; NUM 09 p.5 (R29) rewrites this as
$w k^2/(a_0^2H^2)$ via $a = a_0/(1+z)$. These are consistent, but any reader of NUM 08 alone
should not set $a = a_0$ inside the $(1+z)^{-2}$ form.

**Q9 — Definition of "transfer function".** No document in this group defines $T_k$. The
natural inference from R11 is $T_k(\eta) = \phi_k(\eta)/\phi^*_k$ with $T_k\to 1$ as
$kc_s\eta\to 0$, but this is not written anywhere. Recorded as inferred (§2.8).

**Q10 — Reading of "$\Omega_{\rm cc}$", NUM 01 p.4 (R16).** Presumed to denote the
cosmological-constant density parameter $\Omega_\Lambda$; the glyph is clear but the label is
non-standard.
