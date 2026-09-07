# Cross-spec consistency check (specs 01–05)

Written 2026-09-07 by the cross-spec agent of the spec-transcription campaign
(`prompts/spec-transcription/README.md` §7 step 2). Inputs: the five primary specs
`01-transfer-function.md` … `05-one-loop.md` only. No PDF, no code, no `-B` file was read.
Result labels (`R#`, `Q#`) are those of the spec named; "spec 0N Rk" means result Rk of that spec.

Where this report derives a relation that is **not written in any spec** (e.g. the factor connecting
two Green's functions), it is marked *[derived here]*. Such derivations use only the definitions the
specs record; they are for the author to confirm, not to be treated as transcribed content.

Short names: $c \equiv \dfrac{3(1+w)}{5+3w} = \dfrac{2+b}{3+2b}$ (the super-horizon prefactor
$\phi = c\,\Phi\,S^*$); $c_*$ is the same with $w\to w^*$. `MAIN`/`NUM` document numbers as in the
README.

---

## 1. Conventions matrix

Columns: spec 01 (MAIN 12; NUM 01, 04, 08, 09), spec 02 (MAIN 13; NUM 02, 05, 10, 11), spec 03
(MAIN 10; NUM 03), spec 04 (NUM 06, 07), spec 05 (MAIN 11, 14). "inferred" = the spec says the
convention was deduced, not written.

| README §5 item | Spec 01 | Spec 02 | Spec 03 | Spec 04 | Spec 05 |
|---|---|---|---|---|---|
| **Time variable** | MAIN 12: $\eta$. NUM 01: $\eta\to z$. NUM 04: conformal time written $\tau$, integration in $z$, "$\tau\to0$ as $z\to\infty$". NUM 08/09: $z$. | MAIN 13: $\eta$. NUM 02: $\eta\to z$ via $t$. NUM 05/10/11: $z$. **NUM 05 p.2 reuses $\eta$ for a slow-roll parameter** $d\ln\epsilon/dN$ (Q4). | MAIN 10: $\eta$. NUM 03: $\eta$ (pp.1–3) $\to z$ (p.4 on). | Numerical integral in $z$ (dummy $z'$); analytic evaluation in $\eta$ (dummy $\eta'$). Text says "$\tau$", formulas say $\eta$ (Q2). NUM 07: generic $x$. | $\eta$ throughout. Never $z$. |
| **Prime** | MAIN 12, NUM 01 Step 1: $d/d\eta$. NUM 08/09: $d/dz$ (inferred). | MAIN 13: $d/d\eta$; on Bessel fns $d/d(k\eta)$. NUM 05/10/11: $d/dz$ (stated NUM 05 p.2). $z'$, $\eta'$ = source label. | $d/d\eta$ on pp.1–3 of NUM 03 and in MAIN 10 (inferred); $z$-derivatives always written $d/dz$; $z'$ = source label from p.5. | On $T$ and $a$: $d/d\eta$ (inferred). On Bessel fns, $y,\Theta,\vartheta,\beta$: $d/dx$. | On $h_s,\phi,a$: $d/d\eta$. On $\Phi$: **MAIN 14** $d/dx$, $x=q\eta$ (chain-rule factors $q,r$ explicit); **MAIN 11** behaves as $d/d\eta$ (no $q,r$ factors) — Q1. |
| **$\mathcal H$ vs $H$; $\epsilon$** | $\mathcal H=a'/a$, $H=\dot a/a=\mathcal H/a$. $\epsilon\equiv-\dot H/H^2=(1+z)\,d\ln H/dz$ (NUM 01, 08, 09). MAIN 12: no $\epsilon$; $\mathcal H=(1+b)/\eta$. | Same. $\epsilon=-\dot H/H^2$ stated NUM 05 p.2, NUM 11. NUM 02: $\epsilon$ inferred from $a''/a=(aH)^2(2-\epsilon)$. **$\epsilon$ also used as an infinitesimal** (NUM 02 p.4). | Same. $\epsilon$ inferred from $a''/a=(aH)^2(2-\epsilon)$ (R17). **$\epsilon$ also infinitesimal** (p.6). MAIN 10: no $\epsilon$. | $\mathcal H=aH$ used (R12). No $\epsilon$. | $\mathcal H=a'/a$, $H=\mathcal H/a$; $\mathcal H=\frac{2}{1+3w}\frac1\eta$. No $\epsilon$. |
| **$a_0$** | MAIN 12: absent. NUM 01/04: explicit ($1+z=a_0/a$), $a_0\ne1$. NUM 08: $k^2c_s^2/((1+z)^2a^2H^2)$ with $a=a(z)$. NUM 09 p.5: rewritten $wk^2/(a_0^2H^2)$. | MAIN 13: absent (only $a(\eta')/a(\eta)$). NUM 02: explicit, $k_{\rm phys}\equiv k/a_0$, source $-\frac{1}{a_0H}\delta$. **NUM 05/10/11 (GF part): $k^2/H^2$, no $a_0$, no "phys"** (Q6). NUM 11 TF part: $k^2c_s^2/(a_0^2H^2)$ explicit. | MAIN 10: absent. NUM 03: explicit (pp.4,5,7); $k_{\rm phys},q_{\rm phys}=k/a_0,q/a_0$ inferred; $Q_s/a_0^2$ carried separately, but $Q_\pm$ on p.9 written with $q_{\rm phys}$ (Q7). | Explicit: $1+z=a_0/a$, $dz=-a_0H\,d\eta$; **overall factor $a_0^1$ survives in R14**. But R2 $G_{\rm me}=-H(z')G_{\rm them}$ has no $a_0$ (see conflict B2). | Absent; only $a(\eta')/a(\eta)=(\eta'/\eta)^{1+b}$. |
| **Green's function** | n/a. | For $\chi=ah$. MAIN 13: $G''+(k^2-a''/a)G=+\delta(\eta-\eta')$, retarded ($=0$, $\eta<\eta'$), $G(\eta',\eta')=0$, $[G']=+1$; explicit R9; $G[h]=\frac{a(\eta')}{a(\eta)}G[\chi]$; "differs from Domènech, matches Adshead". NUM 02: $\delta(\eta-\eta')\to-(1+z)aH\,\delta(z-z')$ **without $|\cdot|$**, giving source $-\frac{1}{a_0H}\delta(z-z')$, jump $[dG/dz]=-1/(a_0H(z'))$ (Q3); support in $z$ not stated (Q7). 1st arg response, 2nd source. | NUM 03 R25: $G^\chi_k(z,z')$ with source $+\delta(z-z')$ (operator identical to spec 02 R17 LHS); $\bar G\equiv-G$, $[d\bar G/dz]=-1$, $\bar G=d\bar G/dz=0$ for $z>z'$, $d\bar G/dz\vert_{z\to z'^-}=+1$. 1st arg response. Used with limits $\int_z^{z_{\rm init}}$. No literature relation. | "My" $G_k(z,z')$, source $-\delta(z-z')$ (meaning of measure not spelt out, Q1); $G_{\rm me}=-H(z')\,G_{\rm them}$; $G_{\rm them}$ = MAIN 13 R9 verbatim (R3). 1st arg response. | ${\rm Gr}_k''+(k^2-a''/a){\rm Gr}_k=+\delta(\eta-\eta')$, retarded; explicit R21 = MAIN 13 R9; $h_s=\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}{\rm Gr}_k[\dots]$. Depends on $k=\vert\mathbf k\vert$. |
| **$w$, $b$, $c_s^2$** | $\delta p=c_s^2\delta\rho=w\,\delta\rho$, so **$c_s^2\equiv w$** (stated NUM 01 p.4; $c_s^2=w(z)$ NUM 09). $b=\frac{1-3w}{1+3w}$, $w=\frac13\frac{1-b}{1+b}$ (R7) ⇒ $c_s^2=\frac{1-b}{3(1+b)}$, i.e. **identical to the README formula**. $w(z)$ for m+r+$\Lambda$ (R16; Q4). | $b=\frac{1-3w}{1+3w}$ (MAIN 13). NUM 02/05/10: only $H(z)$, $\epsilon(z)$. NUM 11 TF part: $c_s^2$ **undefined**, $c_s'=dc_s/dz$. | $w_0\equiv p_0/\rho_0$ at the time in the equation (inferred); $w^*$ = "fixed value of $w$ at the initial time" (stated). $b$, $c_s^2$ not used. Whether $w_0$ in $f$ means $w(z')$: not stated (Q5). | Fixed $w$; $b=\frac{1-3w}{1+3w}$ stated; $c_s$ only inside $kc_s\eta$, **$c_s^2$ never defined**. $w$ vs $w_0$ slip (Q3). | Fixed $w$; $b$, $1+w=\frac{2(2+b)}{3(1+b)}$, $5+3w=\frac{2(3+2b)}{1+b}$. **$c_s$ undefined** (Q3). |
| **Fourier / power spectrum / polarisation** | n/a (only $\nabla^2\to-k^2$). | n/a. | $\int\frac{d^3q\,d^3r}{(2\pi)^3}\delta(\mathbf k-\mathbf q-\mathbf r)$; $\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle=(2\pi)^3\delta\,P^*(q)$ (inferred); **$\mathcal P=\frac{q^3}{2\pi^2}P$** (inferred, p.9); $\langle h_sh_{s'}\rangle=(2\pi)^3\delta\,\delta_{ss'}(\dots)$; $Q_\pm=\frac{1}{\sqrt2}q^2_{\rm phys}\sin^2\theta\{\cos2\varphi,\sin2\varphi\}$ ⇒ $e_se_{s'}=\delta_{ss'}$ (inferred). | n/a. | $h_{ij}=\int\frac{d^3k}{(2\pi)^3}e^{i\mathbf k\cdot\mathbf x}\sum_se^s_{ij}h_s$; $e^{ij}_se_{ij\,s'}=\delta_{ss'}$, $e_+=\frac{1}{\sqrt2}(ee-\bar e\bar e)$, $e_\times=\frac{1}{\sqrt2}(e\bar e+\bar ee)$; $\langle S^*S^*\rangle=(2\pi)^3\delta P_*$ (inferred); $\langle h_sh_{s'}\rangle=(2\pi)^3\delta\,\delta_{ss'}P^h_{22}$; **dimensionful $P$ only, no $\mathcal P$**; $Q_\pm=\frac{q^2}{\sqrt2}\sin^2\Theta\{\cos2\phi,\sin2\phi\}$. |
| **Transfer function: field, normalisation** | Field: Newtonian potential $\phi$ (time–time, $e^{2\phi}$ metric inferred). $\phi_{\mathbf k}(\eta)=2^{3/2+b}\Gamma(\tfrac52+b)\phi^*_{\mathbf k}(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta)$, $\phi\to\phi^*$ as $kc_s\eta\to0$ (R11). **$T_k$ never defined**; $T_k=\phi_k/\phi^*_k$ inferred (Q9). | NUM 11 quotes the $\phi$ equation (R39) — field and normalisation not stated (Q11). | $\phi_{\mathbf k}=\frac{3(1+w^*)}{5+3w^*}T_k(z)\,\zeta^*_{\mathbf k}$ (stated, R20); $T_k$ is the TF of $\phi$; $T_k\to1$ inferred. **Seed glyph read as $\zeta$ (medium; "5-like", alternatives $S$, $\mathcal S$).** | $T_k(\eta)=2^{3/2+b}\Gamma(\tfrac52+b)(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta)$; field not stated; $T_k\to1$ inferred. | $\phi_{\mathbf q}=\frac{3(1+w)}{5+3w}\Phi(q\eta)S^*_{\mathbf q}$, $\Phi\to1$ (stated R16); $\Phi(x)=2^{3/2+b}\Gamma(\tfrac52+b)(xc_s)^{-3/2-b}J_{3/2+b}(xc_s)$ (R20). **$S^*$ undefined** (Q2); star written as superscript on $S$, subscript on $P_*$. |
| **Momentum labels, angular reduction** | $k$ only. | $k$ only. | $\mathbf k$; $\mathbf q$; $\mathbf r=\mathbf k-\mathbf q$; $\mathbf t$ (2nd loop). $\theta=\angle(\mathbf q,\mathbf k)$, $\varphi$ azimuth. **$\varphi$-integral done (R33), $\theta$ left implicit, no $d\theta$ written (Q8)**; final form $\int\frac{dq}{q}\sin^5\theta\,\mathcal P^*(q)\frac{\mathcal P^*(r)}{r^3}\{\dots\}^2$ (R35). $f$ args $(\mathbf k,\mathbf k-\mathbf q)$ on pp.5,7 vs $(\mathbf q,\mathbf k-\mathbf q)$ pp.6,9 (Q3). | $\mathbf k$; $\mathbf q$; $r=\vert\mathbf k-\mathbf q\vert$ inferred. No angular reduction. NUM 07: $k,q,s$. | $\mathbf k$; $\mathbf q$; $\mathbf r=\mathbf k-\mathbf q$ (stated); $\mathbf t$, $\mathbf u=\mathbf k'-\mathbf t$. $\Theta$, $\phi$. **No angular reduction**; $Q_s^2=\tfrac12q^4\sin^4\Theta\{\cos^2,\sin^2\}2\phi$ left under $\int d^3q/(2\pi)^3$. |

### 1.1 Conflicts between columns

#### (a) Genuine, deliberate changes of convention (main line 2022 vs numerical branch 2024–25) that downstream code must reconcile

- **A1 — Time variable and orientation.** MAIN 12/13/11/14 and the analytic part of NUM 06 use
  $\eta$; NUM 01–11 use $z$ (NUM 04 calls conformal time $\tau$). $dz=-(1+z)aH\,d\eta=-a_0H\,d\eta$
  (spec 01 R12/R18, spec 02 R12, spec 03 R13, spec 04 R12 — all agree). Because $z$ runs backwards,
  every "$\int_{\eta_0}^{\eta}$" of specs 05/04 becomes "$\int_{z}^{z_{\rm init}}$" of specs 03/04,
  with $\eta_0\leftrightarrow z_{\rm init}$ (neither is defined; spec 05 Q4).
- **A2 — Prime.** $d/d\eta$ on the main line; $d/dz$ in NUM 05/08/09/10/11 (spec 01 inferred, spec 02
  stated). In the numerical branch $z$-derivatives are otherwise written out. Not an inconsistency,
  but a reader of spec 01 R24–R30 or spec 02 R21–R31 must not read a prime as $d/d\eta$.
- **A3 — $a_0$.** The main line never needs $a_0$ (only $a(\eta')/a(\eta)$ appears, spec 05 §2; spec
  02 MAIN 13). The numerical branch keeps $a_0$ explicit and defines $k_{\rm phys}=k/a_0$ (spec 02
  R16, spec 03 §2.2). Consequence for the code: spec 05's $Q_s=e^{lm}_sq_lq_m$ is comoving; spec 03's
  $I_s$ carries $Q_s/a_0^2$ and its p.9 $Q_\pm$ are written directly with $q_{\rm phys}$.
- **A4 — Green's function normalisation in $z$ (the "bar" convention).** Specs 03 and 04 work
  with a Green's function that satisfies the $z$-operator with source $-\delta(z-z')$
  ($\bar G_k$ of spec 03 R25; $G_{\rm me}$ of spec 04 R1) and integrate $\int_z^{z_{\rm init}}dz'$.
  This is a deliberate device ("so that the source integral can be written with limits
  $\int_z^{z_{\rm init}}$", spec 03 §2.2). Relative to the main-line ${\rm Gr}_k$ (source
  $+\delta(\eta-\eta')$, positive just after the source) $\bar G$ is **negative just after the
  source** (spec 03 R25: $d\bar G/dz\vert_{z\to z'^-}=+1$ with $\bar G(z',z')=0$ ⇒ $\bar G\approx z-z'<0$
  for $z<z'$). *[derived here]* Consequently $h_s$ as written in spec 03 R26/R28 is $-1\times$ the
  $h_s$ of spec 05 R18 (with $\zeta^*=S^*$). Harmless in $P_{22}$ (squared), but it is exactly the
  overall minus sign of spec 04 R14 relative to spec 05 R31 (see §2, H4), and it matters for any
  direct comparison of a numerical source integral against MAIN 14.
- **A5 — Placement of the super-horizon prefactor $c^2$.** MAIN 14 absorbs $c^2=\left(\frac{3(1+w)}{5+3w}\right)^2$
  into $f$ (spec 05 R18: $f_{14}=\frac{3(1+w)}{5+3w}\Phi\Phi+\dots$); NUM 03/06 keep $f$ as
  $TT+\frac{2}{3(1+w)}(\dots)(\dots)$ and put $36\left(\frac{1+w^*}{5+3w^*}\right)^2=4c_*^2$ outside
  (spec 03 R22/R26; spec 04 R5). So $f_{14}=c^2f_{03}=c^2f_{06}$ *[derived here, verified on both
  the $w$-form and the Bessel form: spec 05 R28 / spec 04 R11 $=\frac{(2+b)/(3+2b)^3}{1/((3+2b)(2+b))}=c^2$]*.
  In addition NUM 03 distinguishes $w^*$ (initial time, in the prefactor) from $w_0$ (in $f$),
  while MAIN 14 has a single fixed $w$.
- **A6 — $P$ vs $\mathcal P$.** Spec 05 is entirely in dimensionful $P_*$, $P^h_{22}$. Spec 03
  converts $P^*(q)P^*(r)\to\frac{4\pi^4\mathcal P^*(q)\mathcal P^*(r)}{q^3r^3}$ on p.9 (so
  $\mathcal P=q^3P/2\pi^2$, inferred) and its final R35 is in $\mathcal P^*$ but still gives the
  dimensionful $\langle h_sh_{s'}\rangle/(2\pi)^3\delta$. No spec defines a dimensionless tensor
  spectrum or sums over $s$ (spec 05 Q6).
- **A7 — $c_s^2\equiv w$ as the closure, including for a mixture.** Spec 01 states the author's
  deliberate choice $\delta p=w\,\delta\rho$ and $c_s^2=w(z)$ for the matter+radiation background
  (NUM 01 p.4, NUM 09 p.4). For constant $w$ this is the README's $c_s^2=(1-b)/(3(1+b))$ exactly (spec
  01 R7), so **README and spec 01 agree**. Specs 02 (NUM 11), 04 and 05 use $c_s$ without defining it;
  they are consistent with $c_s^2=w$ (the Bessel argument $kc_s\eta$ of spec 05 R20 / spec 04 R4 is
  spec 01 R11 verbatim). The only thing to keep in mind is that in the $z$-dependent case spec 01's
  $c_s^2=w(z)$ is *not* the adiabatic sound speed of the mixture; the author chose it knowingly.
- **A8 — Seed-variable name.** Spec 01: $\phi^*_{\mathbf k}$ (the early-time value of $\phi$).
  Spec 03: $\zeta^*_{\mathbf k}$ with $\phi=c_*T\zeta^*$. Spec 05: $S^*_{\mathbf q}$ (MAIN 14) /
  $S(\mathbf k)$ (MAIN 11) with $\phi=c\,\Phi\,S^*$. Hence $\phi^*=c\,S^*=c_*\zeta^*$ and
  $S^*\equiv\zeta^*$ (as symbols). See B5 for the glyph question.

#### (b) Apparent inconsistencies that may be transcription errors or author slips

- **B1 — Sign of the redshift Green's-function source in NUM 02 (spec 02 R14/R17/R18, Q3).**
  NUM 02 transforms $\delta(\eta-\eta')=\delta(z-z')\,dz/d\eta$ with no absolute value and gets source
  $-\frac{1}{a_0H}\delta(z-z')$, jump $-1/(a_0H(z'))$. *[derived here]* With $\vert dz/d\eta\vert$
  the source is $+\frac{1}{a_0H(z')}\delta(z-z')$: MAIN 13's $G_{\rm them}\approx\eta-\eta'=(z'-z)/(a_0H)$
  just after the source, so $dG/dz\vert_{z'^-}=-1/(a_0H)$, $dG/dz\vert_{z'^+}=0$, jump $+1/(a_0H)$.
  Two other specs are consistent with the *plus* sign and inconsistent with NUM 02's minus:
  spec 04 R2 ($G_{\rm me}=-H(z')G_{\rm them}$ with $G_{\rm me}$ of source $-\delta$) and spec 03 R25
  ($\bar G$ of source $-\delta$ is negative after the source, like $-G_{\rm them}$). So NUM 02's
  sign is an author slip (missing $\vert\cdot\vert$), already flagged by the transcriber as Q3, and it
  does **not** propagate into NUM 03 or NUM 06. But **NUM 02 R18 is the boundary condition of the
  numerically integrated Green's function**; implemented literally it yields $-G_{\rm them}$ (as a
  function of $z$), not $G_{\rm them}$. Highest-priority item.
  **Author resolution 2026-09-07:** the sign is a convention paired with the orientation of $\int dz'$, not a
  slip, and R18 is not the code's boundary condition. The code imposes the unit jump of spec 03 R25.
  See `REVIEW-QUEUE.md` §1.1 and spec 02 §0.
- **B2 — $a_0$ in $G_{\rm me}=-H(z')G_{\rm them}$ (spec 04 R2) and the power of $a_0$ in spec 04 R14.**
  *[derived here]* From spec 03 R25 the unit-jump function satisfies $\bar G_{03}=-a_0H(z')G_{\rm them}$
  (both vanish for $z>z'$; $\bar G\approx z-z'$, $G_{\rm them}\approx(z'-z)/(a_0H)$). Spec 04's
  $G_{\rm me}$ is described the same way ("source $-\delta(z-z')$") but its stated relation lacks the
  $a_0$: $G_{\rm me}=\bar G_{03}/a_0$. Either NUM 06 sets $a_0=1$ in R2 without saying so, or its
  "$-\delta(z-z')$" means $-\delta(z-z')/a_0$. The consequence is the overall $a_0^1$ in spec 04 R14:
  with $G_{\rm me}\to\bar G_{03}$ the prefactor would be $-a_0^2$, and then the $a_0$ cancels exactly
  against the $Q_s/a_0^2$ of spec 03 R28 — which is what spec 03 R29 says must happen ("$h_s$ has the
  right dependence on $a_0$") and what spec 02 §2.4 anticipates (the $a_0^{-1}$ "should disappear
  when we switch the $\chi$ Green's function for the $h$ Green's function"). With $a_0^1$ the
  combination $I_s$ would retain a stray $1/a_0$. Author slip in NUM 06 R2, or $a_0=1$ assumed there.
  **Author resolution 2026-09-07:** the power $a_0^2$ is confirmed. The mechanism is neither a slip nor
  $a_0 = 1$: the project convention absorbs $a_0$ into $k/a_0$ and $a_0\eta$, and R2 is written in that
  convention while R13 is not. See `REVIEW-QUEUE.md` §1.1 and spec 04 §0.
- **B3 — Three differently normalised "numerical" Green's functions.** NUM 02: jump
  $\mp1/(a_0H(z'))$ ⇒ $G_{02}=\mp G_{\rm them}$ (sign per B1). NUM 03: $\bar G_{03}$, jump $-1$
  ⇒ $\bar G_{03}=-a_0H(z')G_{\rm them}$. NUM 06: $G_{\rm me}=-H(z')G_{\rm them}$. They differ by
  factors $a_0H(z')$ and $a_0$ (and a sign). Spec 04 R13 only works (the $H(\eta')$ from $dz'$
  cancels the $1/H^2$) if the code's Green's function carries one power of $H(z')$, i.e. is of the
  NUM 03/06 type, not the NUM 02 type. Which one the code integrates is not decidable from the
  specs; see review queue item 1.
  **Author resolution 2026-09-07:** the code integrates the NUM 03/06 unit-jump function,
  $G_{\rm code} = -a_0H(z')G_{\rm them}$ for any $a_0$. The three rows of the H2 table are one object once
  the absorbed-$a_0$ convention is applied. See `REVIEW-QUEUE.md` §1.1.
- **B4 — Prime on $\Phi$ in MAIN 11 (spec 05 Q1).** MAIN 11 R8 has $\eta\,\Phi'\Phi$, MAIN 14 R18
  has $q\eta\,\Phi'\Phi$. Cross-check: spec 04 R6→R9→R11 starts from $T'=dT/d\eta$, converts with
  $dT/dx$ explicitly, and lands on the same Bessel form as spec 05 R28 (up to $c^2$, A5). So MAIN 14's
  $\Phi'=d\Phi/dx$ is confirmed by an independent 2024 computation; MAIN 11's notation is the slip.
  Resolved in favour of MAIN 14 (the build target).
- **B5 — Seed glyph $\zeta^*$ (spec 03) vs $S^*$ (spec 05).** Spec 03 R20 reads a "5-like character
  with subscript $\mathbf k$ and superscript $*$" as $\zeta$ and names $S$ as an alternative; spec 05
  reads $S^*$ with the star as a superscript. These are very probably the *same* handwritten symbol
  read two ways. Whether it is $\zeta$ or $S$ changes nothing algebraically (A8), but it changes
  whether the notes ever identify the seed with the curvature perturbation. Neither spec records a
  definition, or a sign, relating $S^*/\zeta^*$ to $\zeta$ or $\mathcal R$ (spec 05 Q2).
  **Author resolution 2026-09-07:** same symbol, and it is $\zeta^*$, the primordial curvature perturbation;
  $P_* = P_\zeta$, to be fed from `PyTransport`/`CppTransport`. Sign immaterial. See `REVIEW-QUEUE.md` §1.4, spec 03 §0.4.
- **B6 — Coefficient chain 1024 → 2048 → 1024π → 512π² in NUM 03 pp.7–10 (spec 03 R30–R35, Q2).**
  The author's 2025 annotations propose 1296, 2592, "1292", "646". Cross-check against spec 05 R23
  (§2, H5) shows the outer structures agree **only** with $2\times36^2=2592$; the original
  $1024=32^2$ is inconsistent with MAIN 11/14. "1292" and "646" are then typos for 1296 and 648
  (as spec 03 already suspected). Type (b), resolved in favour of the annotations; author to confirm
  the two typos.
  **Author resolution 2026-09-07:** confirmed. Chain $1296 \to 1296 \to 2592 \to 1296\pi \to 648\pi^2$; "1292" and
  "646" are typos. The 32 of MAIN 11/14 has $c^2$ inside $f$; NUM 03's 36 $= 4\times3^2$ has $c_*^2$ outside. See
  spec 03 §0.2 and `REVIEW-QUEUE.md` §1.2.
- **B7 — Arguments of $f$** written $(\mathbf k,\mathbf k-\mathbf q)$ on NUM 03 pp.5, 7 (spec 03 Q3).
  Spec 05 uses $f(\mathbf q,\mathbf k-\mathbf q,\eta)$ / $f(q,r,\eta)$ throughout. Slip on pp.5, 7;
  resolved.
- **B8 — $w$ vs $w_0$, and at which time.** Spec 04 Q3 ($\rho_0$ as reference density on p.2 but
  as the density at time $\eta$ on p.3; $1+w$ vs $1+w_0$) and spec 03 Q5 ($w_0=w(z')$?) are the same
  question. For fixed $w$ all coincide. The specs' own formulas place the factor
  $2\mathcal H^2M_P^2/(a^2(\rho_0+p_0))$ (spec 05 R4) / $\frac{8M_P^2}{(\rho_0+p_0)a^2}$ (spec 03 R12)
  *inside* the source, i.e. under the $\eta'$ integral, so $\rho_0+p_0$ — hence $w_0$ — is the
  background at the **source** time; but no spec says so in words. Author to confirm.
  **Author resolution 2026-09-07:** confirmed, $w_0 = w(z')$ at the source time; $w^*$ in the prefactor is
  $w(z_{\rm init})$, which must lie in a constant-$w$ epoch. MAIN 11/14's single $w$ is their fixed-$w$ restriction.
  `ComputeTargets/QuadSource.py` evaluates $w$ at the source redshift. See spec 03 §0.2.
- **B9 — $k$ vs $k_{\rm phys}$ in NUM 05/10/11 GF part (spec 02 Q6).** NUM 02 and NUM 03 both use
  $k_{\rm phys}=k/a_0$ explicitly with the identical operator; the inference "$k$ means $k/a_0$" in
  NUM 05/10/11 is consistent with both. Low risk; confirm.
- **B10 — Symbol overloads that are not errors but are traps:** $\epsilon$ = slow-roll parameter and
  infinitesimal (specs 02, 03); $\eta$ = conformal time and (NUM 05 p.2 only) second slow-roll
  parameter; $\omega_{\rm eff}$ = two different functions (Green's-function and transfer-function
  WKB, spec 02 R22 vs R43, spec 01 R27); $Q_s$ with comoving $q$ (spec 03 p.4, spec 05) and with
  $q_{\rm phys}$ (spec 03 p.9); $f$ = source (specs 03–05) and friction-eliminator (spec 02 R20).

---

## 2. Hand-off chain

Pipeline: $T_k$ (spec 01) → $G_k$ (spec 02) → $f$ (spec 03) → time integral (spec 04; spec 05
Step 6) → loop integral (spec 05; spec 03 pp.6–10). One subsection per hand-off.

### H0 — MAIN 10 stress → MAIN 11 source (spec 03 Q11, spec 05 Q7)

Upstream (spec 03 R11, MAIN 10 p.4):
$T^i{}_j\vert_{O(\delta,v)^2}=\frac{4M_P^4}{a^4(\rho_0+p_0)}\partial^i(\phi'+\mathcal H\phi)\partial_j(\phi'+\mathcal H\phi)$.
Downstream (spec 05 R1–R2, MAIN 11 p.1–2): $G^i{}_j\supset\frac{1}{2a^2}\{h''+2\mathcal Hh'-\partial^2h\}$
and, after $G=M_P^{-2}T$ and multiplying by $2a^2$, the projected source
$\frac{8M_P^2}{a^2(\rho_0+p_0)}\partial_a(\phi'+\mathcal H\phi)\partial_b(\phi'+\mathcal H\phi)$.
**Match:** $2a^2\times M_P^{-2}\times\frac{4M_P^4}{a^4(\rho_0+p_0)}=\frac{8M_P^2}{a^2(\rho_0+p_0)}$.
This also fixes the illegible power in spec 05 Q7 ("$a^{2+}$") as $a^4$, consistent with MAIN 10.
The $4\partial\phi\partial\phi$ term is geometric (MAIN 01–08, not transcribed).

### H1 — Transfer function: spec 01 → specs 03, 04, 05 (and spec 02 NUM 11)

Upstream (spec 01 R11): $\phi_{\mathbf k}(\eta)=2^{3/2+b}\Gamma(\tfrac52+b)\,\phi^*_{\mathbf k}\,(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta)$,
$\phi\to\phi^*$ as $kc_s\eta\to0$; $T_k$ not defined.
Consumed as: spec 04 R4 $T_k(\eta)=2^{3/2+b}\Gamma(\tfrac52+b)(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta)$;
spec 05 R20 $\Phi(k\eta)=$ the same; spec 05 R16 $\phi_{\mathbf q}=c\,\Phi(q\eta)S^*_{\mathbf q}$;
spec 03 R20 $\phi_{\mathbf k}=c_*T_k(z)\zeta^*_{\mathbf k}$.
**Match**, with $T_k\equiv\Phi\equiv\phi_k/\phi^*_k$ and $\phi^*=c\,S^*=c_*\zeta^*$. Bessel order
$3/2+b$, exponent $-3/2-b$ and $\Gamma(5/2+b)$ agree in all three. This resolves spec 01 Q9 and spec
04's "field not stated": $T_k$ is the transfer function of the Newtonian potential $\phi$, normalised
to 1 at early times. The sound speed in the argument is spec 01's $c_s^2=w$ (A7), which resolves spec
05 Q3 and spec 04's missing $c_s^2$ definition.

Spec 02 NUM 11 TF part vs spec 01: R39 = spec 01 R21 term by term; R40 = R23/R26; R41 = R24;
R43 = R27/R29 (with $a^2(1+z)^2=a_0^2$, spec 02 R42 = spec 01 Q8). **Match.** Resolves spec 02 Q11
and the last bullet of spec 02 §4. Spec 01 Q7 (NUM 09 p.5 vs p.6 in $d\omega_{\rm eff}/dz$) has no
counterpart in spec 02 and stays open.

Numerical branch: spec 03's $T_k(z)$ and $(1+z)\,dT/dz$ are the objects of spec 01 R14/R21
(NUM 01/08/09); the combination $T-(1+z)\,dT/dz=T+T'/\mathcal H$ (spec 03 R16) is exactly the
$\Phi+\Phi'/\mathcal H$ of spec 05 R4. Match.

### H2 — Green's function: spec 02 → specs 03, 04, 05

Analytic (constant $w$): spec 02 R9 $=$ spec 05 R21 $=$ spec 04 R3:
$\frac\pi2\sqrt{\eta\eta'}\{J_{b+1/2}(k\eta')Y_{b+1/2}(k\eta)-Y_{b+1/2}(k\eta')J_{b+1/2}(k\eta)\}$
for $\eta>\eta'$, source $+\delta(\eta-\eta')$, jump $+1$. **Match** (three independent writings,
2022 ×2 and 2024). Spec 02 R10 $G[h]=\frac{a(\eta')}{a(\eta)}G[\chi]$ is the $\frac{a(\eta')}{a(\eta)}$
of spec 05 R18 and the $\frac{1+z}{1+z'}$ of spec 03 R24 / spec 04 R12. Match.

Numerical (general background), operator: spec 02 R17 LHS
$\frac{d^2}{dz^2}+\frac{\epsilon}{1+z}\frac{d}{dz}+\frac{k_{\rm phys}^2}{H^2}-\frac{2-\epsilon}{(1+z)^2}$
$=$ spec 03 R22/R25 LHS. **Match**, including the red-corrected friction coefficient $\epsilon$
(spec 02 §5 items 8–9): NUM 03 R18 (20 Sep 2024) independently obtains $\epsilon$, confirming the
correction in NUM 02 (12 Sep 2024).

Numerical, source/normalisation: **conflict**, see B1–B3. Summary *[derived here]*:

| Object | Defined in | Source in $z$ | Jump $[dG/dz]_{z'^-}^{z'^+}$ | Relation to $G_{\rm them}(\eta(z),\eta(z'))$ |
|---|---|---|---|---|
| $G_{\rm them}$ itself | spec 02 R9 | $+\frac{1}{a_0H(z')}\delta$ (correct); NUM 02 writes $-\frac{1}{a_0H}\delta$ | $+\frac{1}{a_0H(z')}$ (NUM 02: $-$) | 1 |
| NUM 02 $G$ as specified (R18) | spec 02 | $-\frac{1}{a_0H}\delta$ | $-\frac{1}{a_0H(z')}$ | $-1$ |
| $G^\chi_{03}$ | spec 03 R25 | $+\delta$ | $+1$ | $+a_0H(z')$ |
| $\bar G_{03}=-G^\chi_{03}$ | spec 03 R25 | $-\delta$ | $-1$ | $-a_0H(z')$ |
| $G_{\rm me}$ | spec 04 R1–R2 | "$-\delta$" | (not stated) | $-H(z')$ (as written) |

### H3 — Source $f$: spec 03 (NUM 03, $z$) vs spec 04 (NUM 06, $\eta$) vs spec 05 (MAIN 14) — item (ii)

- Spec 03 R22: $f_{03}=T_qT_r+\frac{2}{3(1+w_0)}\big(T-(1+z)\tfrac{dT}{dz}\big)_q\big(T-(1+z)\tfrac{dT}{dz}\big)_r$.
- Spec 04 R5–R6: $f_{06}=T_qT_r+\frac{2M_P^2}{\rho_0(1+w)}\frac{1}{a^2}(T_q'+\mathcal HT_q)(T_r'+\mathcal HT_r)=T_qT_r+\frac{2}{3(1+w)}\big(T_q+\tfrac{T_q'}{\mathcal H}\big)\big(T_r+\tfrac{T_r'}{\mathcal H}\big)$.
- Spec 05 R18: $f_{14}=\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta)+\frac{6(1+w)}{(5+3w)^2}\big(\Phi+\tfrac{\Phi'_\eta}{\mathcal H}\big)_q\big(\Phi+\tfrac{\Phi'_\eta}{\mathcal H}\big)_r$ (R17 form, $\Phi'_\eta=d/d\eta$).

**Match up to stated convention changes:** $f_{03}=f_{06}$ exactly (spec 03 R16 gives
$T-(1+z)dT/dz=T+T'/\mathcal H$; $w_0\leftrightarrow w$, B8), and $f_{14}=c^2f_{06}$ (A5), since
$\frac{6(1+w)}{(5+3w)^2}=c^2\cdot\frac{2}{3(1+w)}$. Verified independently on the Bessel forms:
spec 04 R11 $f_{06}=\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma^2(\dots)^{-1/2-b}\{J_{1/2+b}J_{1/2+b}+\frac{2+b}{1+b}J_{5/2+b}J_{5/2+b}\}$
and spec 05 R28 $f_{14}=\frac{2+b}{(3+2b)^3}2^{3+2b}\Gamma^2(\dots)^{-1/2-b}\{\text{same}\}$; ratio
$\frac{(2+b)^2}{(3+2b)^2}=c^2$. The two Bessel reductions (2022 and 2024) agree in every order,
exponent and the $\frac{2+b}{1+b}$ coefficient — a genuine independent check of MAIN 14 Steps 5.
The $4c_*^2$ outside spec 03's $f$ equals MAIN 14's $4$ times the $c^2$ inside $f_{14}$ (for
$w^*=w$).

### H4 — Time integral: spec 04 R14 (NUM 06 p.7) vs spec 05 R31 `TARGET` (MAIN 14 p.11) — item (i)

Upstream/downstream as written:

- Spec 05 R31: $\displaystyle\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}{\rm Gr}_k(\eta,\eta')f_{14}(q,r,\eta')
  =\pi\,2^{2+2b}\frac{2+b}{(3+2b)^3}\Gamma(\tfrac52+b)^2\,(c_s^2qr\eta)^{-\frac12-b}\big(Y_{b+\frac12}(k\eta)I_J-J_{b+\frac12}(k\eta)I_Y\big)$.
- Spec 04 R14: $\displaystyle\int_{z}^{z_{\rm init}}dz'\,G_{\rm me}(z,z')\frac{1+z}{1+z'}\frac{f_{06}}{H(z')^2}
  =-\frac{a_0\pi}{2}\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma(\tfrac52+b)^2\,(qrc_s^2\eta)^{-\frac12-b}\big(Y_{b+\frac12}(k\eta)I_J-J_{b+\frac12}(k\eta)I_Y\big)$,
  with the *same* $I_J$, $I_Y$ ($\int_{\eta_{\rm init}}^{\eta}d\eta'(\eta')^{1/2-b}\{J,Y\}_{b+1/2}(k\eta')\,[J_{1/2+b}J_{1/2+b}+\frac{2+b}{1+b}J_{5/2+b}J_{5/2+b}]$).

The integrands and the $\eta$-dependence outside agree glyph for glyph (weight $(\eta')^{1/2-b}$,
prefactor $(c_s^2qr\eta)^{-1/2-b}$, orders $b+\tfrac12$, $\tfrac12+b$, $\tfrac52+b$, sign structure
$YI_J-JI_Y$). In particular NUM 06's $\frac\pi2 2^{3+2b}=\pi2^{2+2b}$ **independently confirms the
over-written exponent $2+2b$ of MAIN 14 pp.10–11** (spec 05 §5).

**Exact relation between the prefactors** *[derived here]*:
$$
\frac{\text{spec 04 R14}}{\text{spec 05 R31}}
=-a_0\cdot\frac{\tfrac\pi2 2^{3+2b}}{\pi2^{2+2b}}\cdot\frac{1/((3+2b)(2+b))}{(2+b)/(3+2b)^3}
=-a_0\,\frac{(3+2b)^2}{(2+b)^2}=-\frac{a_0}{c^2}=-a_0\frac{(5+3w)^2}{9(1+w)^2}.
$$
Equivalently $\text{TARGET}=-\dfrac{c^2}{a_0}\times(\text{spec 04 R14})$. The three factors are
accounted for as follows:

1. $1/c^2$: $f_{14}=c^2f_{06}$ (H3, A5). Reconciled.
2. $-1$: spec 04's $G_{\rm me}$ is of the "bar" type (negative after the source, A4); spec 04 R13
   shows the sign arising from $G_{\rm me}=-H(z')G_{\rm them}$ and the reversal of limits. Reconciled
   *given* R2.
3. $a_0^1$: from $dz'=-a_0H\,d\eta'$ with $G_{\rm me}=-H(z')G_{\rm them}$. **Not fully reconciled** (B2):
   the spec 03 unit-jump function is $\bar G_{03}=-a_0H(z')G_{\rm them}$, which would give $-a_0^2/c^2$,
   and it is $a_0^2$ that cancels the $Q_s/a_0^2$ of spec 03 R28 to make $h_s$ $a_0$-independent.
   The two specs can be reconciled only if NUM 06 R2 is read with $a_0=1$ or as
   $G_{\rm me}=-a_0H(z')G_{\rm them}$. The $a(\eta')/a(\eta)$ factor named in the task is present
   in both (spec 04 R12 $\frac{1+z}{1+z'}=\frac{a(\eta')}{a(\eta)}$) and is not a source of
   discrepancy.

**Verdict:** the two results are the same integral; they match up to the stated convention changes
(sign, $c^2$) plus one $a_0$ power whose value depends on the unresolved normalisation of the
numerical Green's function (B2/B3). They cannot be declared to agree to the last factor until the
author states which of the three $z$-Green's functions in the H2 table the code computes.

### H5 — Outer one-loop structure: spec 03 R28–R35 (NUM 03 pp.7–10, $z$) vs spec 05 R23 (MAIN 14 p.6, $\eta$) — item (iii)

**Author resolution 2026-09-07:** the two are the same integral before and after the measure is split and the $\varphi$ integral done; with $648\pi^2$ they coincide exactly. The build form is spec 03 R35 completed with $\int_0^\infty dq/q\int_0^\pi d\theta$ and $r(\theta)$, stored per polarisation; $\Omega_{\rm GW}$ is a separate layer. See `REVIEW-QUEUE.md` §1.3 and spec 03 §0.3.

As written:

- Spec 05 R23: $P^h_{22}(k)=32\displaystyle\int\frac{d^3q}{(2\pi)^3}Q_s(\mathbf k,\mathbf q)^2P_*(q)P_*(r)\Big(\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}{\rm Gr}_kf_{14}\Big)^2$,
  $Q_s^2=\tfrac12q^4\sin^4\Theta\{\cos^2,\sin^2\}2\phi$, no angular reduction, per polarisation.
- Spec 03 R32: $\langle h_sh_s\rangle=(2\pi)^3\delta\;2048\big(\tfrac{1+w^*}{5+3w^*}\big)^4\displaystyle\int\frac{d^3q}{(2\pi)^3}P^*(q)P^*(r)\,I_s^2$
  [2048 annotated → 2592], $I_s=\int_z^{z_{\rm init}}dz'\bar G_k\frac{1+z}{1+z'}\frac{Q_s}{a_0^2H^2(z')}f_{03}$.
- Spec 03 R35: $(2\pi)^3\delta\,\delta_{ss'}\,512\pi^2\big(\tfrac{1+w^*}{5+3w^*}\big)^4\displaystyle\int\frac{dq}{q}\sin^5\theta\,\mathcal P^*(q)\frac{\mathcal P^*(r)}{r^3}\Big\{\int_z^{z_{\rm init}}dz'\,\bar G_k\frac{1+z}{1+z'}\frac{q^2_{\rm phys}}{H(z')^2}f_{03}\Big\}^2$
  [512 annotated → "646"; $d\theta$ not written; $\bar G$ bar uncertain; $f(z\vert\dots)$ should be $f(z'\vert\dots)$].

**Relation before angular reduction** *[derived here]*: with $\bar G_{03}=-a_0H(z')G_{\rm them}$,
$dz'=-a_0Hd\eta'$, $f_{03}=f_{14}/c^2$ (all for $w^*=w$):
$I_s=\frac{Q_s}{a_0^2}\int_z^{z_{\rm init}}dz'\,\bar G_{03}\frac{1+z}{1+z'}\frac{f_{03}}{H^2}=-\frac{Q_s}{c^2}\int_{\eta_{\rm init}}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}G_{\rm them}f_{14}=-\frac{Q_s}{c^2}\,\text{TARGET}$.
Then spec 03 R32 with the corrected coefficient reads
$2592\,c_*^4/81\cdot\int\frac{d^3q}{(2\pi)^3}P^*P^*\,\frac{Q_s^2}{c^4}\text{TARGET}^2=32\int\frac{d^3q}{(2\pi)^3}Q_s^2P_*P_*\,\text{TARGET}^2$
(using $2592=2\times36^2$ and $36^2(1+w)^4/(5+3w)^4=16c^4$; $2\times16=32$), i.e. **exactly spec 05
R23**. With the original 1024/2048 the two would differ by $(32/36)^2$. This is the basis of B6.

**Angular reduction connecting R23 to R35.** It *is* in spec 03 (R33–R35), and nowhere in spec 05.
The steps on the page are: (1) $Q_\pm^2=\frac12q_{\rm phys}^4\sin^4\theta\{\cos^2,\sin^2\}2\varphi$
and $\int_0^{2\pi}d\varphi\{\cos^2,\sin^2\}2\varphi=\pi$, $\int d\varphi\,Q_+Q_-=0$ (⇒ $\delta_{ss'}$;
this is also spec 05's p.7 remark, Q12); (2) $P^*(q)P^*(r)=4\pi^4\mathcal P^*(q)\mathcal P^*(r)/(q^3r^3)$;
(3) $d^3q=q^2dq\,\sin\theta\,d\theta\,d\varphi$. Net *[derived here]*:
$$
32\int\frac{d^3q}{(2\pi)^3}Q_s^2P_*(q)P_*(r)X^2
=32\cdot\frac{\pi\cdot\tfrac12\cdot4\pi^4}{(2\pi)^3}\int\frac{dq}{q}\int_0^\pi d\theta\,\sin^5\theta\,\mathcal P(q)\frac{\mathcal P(r)}{r^3}\,q^4X^2
=8\pi^2\int\frac{dq}{q}\int_0^\pi d\theta\,\sin^5\theta\,\mathcal P(q)\frac{\mathcal P(r)}{r^3}\,q^4X^2,
$$
and $648\pi^2\big(\frac{1+w}{5+3w}\big)^4=8\pi^2c^4$, which is absorbed by
$\{q^2_{\rm phys}\,\bar G\text{-integral}\}^2=q^4\,\text{TARGET}^2/c^4$. So the corrected spec 03 R35
and spec 05 R23 are the same quantity. What the reduction does **not** contain, in either spec:
the $d\theta$ and its range $[0,\pi]$ (spec 03 Q8; implied by $d^3q$), the relation
$r=\sqrt{k^2+q^2-2kq\cos\theta}$ (spec 03 notes it is "not written out"), any change to
$(q,r)$ or $(u,v)$ variables, and any polarisation sum. The two candidate forms for the one-loop
code are therefore: (A) spec 05 R23 as a 3-d $\int d^3q$ with $Q_s^2$ under the integral, or (B)
spec 03 R35 as a 2-d $(q,\theta)$ integral with coefficient $8\pi^2c_*^4$ (in the spec 03 split) /
$8\pi^2$ (with $c^2$ inside $f$) — related by the three steps above and nothing else.

---

## 3. Open questions that cross specs

### 3.1 Resolved by another spec (subject to the author's confirmation)

| Question | Resolution | By |
|---|---|---|
| Spec 01 Q9 — is $T_k=\phi_k/\phi^*_k$, $T_k\to1$? | Yes: spec 03 R20 and spec 05 R16 both write $\phi=c\,T\,(\text{seed})$ with $T\to1$; spec 04 R4 / spec 05 R20 reproduce spec 01 R11 divided by $\phi^*$. | H1 |
| Spec 01 Q8 — $a$ vs $a_0$ in the $k^2$ term | Consistent with spec 02 R16/R42 and spec 03 §2.2 ($k_{\rm phys}=k/a_0$). | A3 |
| Spec 02 Q3 — sign of the $z$-source in NUM 02 | Slip (missing $\vert dz/d\eta\vert$); the opposite sign is what spec 03 R25 and spec 04 R2 are consistent with. **Consequence for the numerical BC remains** (queue item 1). | B1 |
| Spec 02 Q6 — $k$ vs $k_{\rm phys}$ in NUM 05/10/11 | Consistent with NUM 02 and NUM 03; inference $k\to k/a_0$ stands. | B9 |
| Spec 02 Q7 — support of the retarded GF in $z$ | Spec 03 R25 states $\bar G=d\bar G/dz=0$ for $z>z'$. | H2 |
| Spec 02 Q11 — TF content of NUM 11 | Agrees term by term with spec 01 R21–R27; field is $\phi$, $c_s^2=w$. | H1 |
| Spec 03 Q2 — 1296/2592/"1292"/"646" | 1296 and 2592 confirmed by spec 05 R23; "1292", "646" are typos for 1296, 648. | B6, H5 |
| Spec 03 Q3 — arguments of $f$ | $(\mathbf q,\mathbf k-\mathbf q)$, as spec 05 throughout. | B7 |
| Spec 03 Q7 — meaning of $k_{\rm phys}$ | Defined in spec 02 R16 as $k/a_0$. | A3 |
| Spec 03 Q11 — MAIN 10 R11 vs MAIN 11 R2 factor | $2a^2M_P^{-2}$; consistent. Also fixes spec 05 Q7 ($a^4$). | H0 |
| Spec 03 Q12 — "$Q^s$ is blind to $\mathbf k$" | Spec 05 R23: $Q_{s'}(\mathbf k',\mathbf k'+\mathbf q)=Q_{s'}(\mathbf k',\mathbf q)$ (as $e,\bar e\perp\mathbf k'$) and $Q_{s'}(-\mathbf k,-\mathbf q)=Q_{s'}(\mathbf k,\mathbf q)$. | — |
| Spec 04 Q1 — what is $G_{\rm them}$ | MAIN 13 R9 = MAIN 14 R21: retarded $\chi$-Green's function, source $+\delta(\eta-\eta')$, jump $+1$. The "$-\delta(z-z')$" of $G_{\rm me}$ matches spec 03's $\bar G$ **up to $a_0$** (B2). | H2 |
| Spec 04 Q3 — $w$ vs $w_0$; $\rho_0$ | Same quantity for fixed $w$; see B8 for the time-dependent case. | B8 |
| Spec 04 "field not stated", "$c_s^2$ not stated" | $\phi$; $c_s^2=w$ (spec 01). | H1, A7 |
| Spec 05 Q1 — prime on $\Phi$ in MAIN 11 | MAIN 14 reading confirmed by spec 04 R9–R11. | B4 |
| Spec 05 Q3 — $c_s$ undefined | $c_s^2=w=(1-b)/(3(1+b))$ (spec 01 R4, R7). | A7 |
| Spec 05 Q5 — angular reduction | Done (φ only) in spec 03 R33–R35; θ-integral implicit. | H5 |
| Spec 05 Q12 — $\delta_{ss'}$ from the $\phi$ integral | Spec 03 R33 does the integrals explicitly. | H5 |
| Spec 01 Q1, spec 03 Q1 — page counts | README §1 now carries `pdfinfo` counts; the earlier figures were over-counts. | README |

### 3.2 Cross-spec questions that remain open

1. **Which $z$-Green's function does the code compute** (H2 table)? Not decidable from the specs.
2. **$a_0$ power in spec 04 R14** (B2): $-a_0/c^2$ as written vs $-a_0^2/c^2$ implied by spec 03.
3. **Seed variable**: $\zeta^*$ or $S^*$ glyph (B5); its definition and sign relative to the
   curvature perturbation; relation of $P_*$ to a primordial $\mathcal P_\zeta$ (spec 05 Q2).
4. **Deliverable**: per-polarisation $P^h_{22}$ (spec 05 R23, spec 03 R35 with $\delta_{ss'}$) vs
   a polarisation sum vs dimensionless $\mathcal P_h$ (spec 05 Q6; A6). No spec decides.
5. **Time at which $w$ is evaluated inside $f$** and in the prefactor ($w_0$, $w^*$; B8; spec 03 Q5).
6. **$\eta_0$ / $z_{\rm init}$** (spec 05 Q4): spec 01 R19 (NUM 04) fixes $a_0\tau=-\int_\infty^z dz'/H$
   with $\tau\to0$ as $z\to\infty$, so $\eta_{\rm init}=\eta(z_{\rm init})$ is computable, but no spec
   says whether the analytic TARGET is meant with $\eta_0\to0$ or with a finite $\eta_0$.
7. **Spec 01 Q4** ($w(z)$ with/without $\Lambda$ pressure) and **spec 01 Q7** (NUM 09 p.5 vs p.6):
   no other spec touches them.
8. **Spec 04 Q7/Q12** (Levin phase conventions; connection of $\mathcal J^\mu_{\nu\sigma}$ to $I_{J/Y}$):
   spec 05 R32–R33 contain MAIN 14's own spherical-Bessel ("Fabrikant") rewriting of $I_{J/Y}$ with
   weight $(\eta')^{2-b}$, which is the object NUM 07's $\mathcal J$ (weight $x^{1/2}$ on cylindrical
   $J$'s $=x^2$ on spherical $j$'s) resembles — but README §2 excludes Step 7 from the build, and
   neither spec writes the link. Open, low priority for the build target.
9. **Spec 05 Q11** ($h_{ij}$ normalisation vs Kohri–Terada / Adshead) and spec 02's "matches Adshead"
   for $G[h]$: no spec allows a check.

---

## 4. Review queue (cross-spec), most consequential for the one-loop build first

1. **Normalisation and sign of the numerical Green's function** (B1, B3, H2 table). State which
   object the code integrates: NUM 02 R18 (jump $-1/(a_0H(z'))$, which literally gives
   $-G_{\rm them}$), the unit-jump $\bar G$ of NUM 03 R25 ($=-a_0H(z')G_{\rm them}$), or NUM 06's
   $G_{\rm me}$ ($=-H(z')G_{\rm them}$). Confirm that NUM 02's minus sign is the missing
   $\vert dz/d\eta\vert$. Everything downstream (spec 03 R28, spec 04 R14) presupposes one power of
   $H(z')$ in the Green's function.
2. **Relation of the analytic source integral to the build target** (H4): confirm
   $\text{TARGET}=-\frac{c^2}{a_0}\times(\text{spec 04 R14})$ as written, or $-\frac{c^2}{a_0^2}$ if
   $G_{\rm me}\equiv\bar G_{03}$; decide whether spec 04 R2 should read $G_{\rm me}=-a_0H(z')G_{\rm them}$.
   Note the overall sign: NUM-branch $h_s$ is $-1\times$ MAIN-branch $h_s$ (A4).
3. **One-loop coefficients and form** (B6, H5): confirm 1296 → 2592 → 1296π → 648π² (annotations
   "1292", "646" are typos), and choose form (A) spec 05 R23 or (B) spec 03 R35 with $8\pi^2$
   (equivalently $648\pi^2(\tfrac{1+w^*}{5+3w^*})^4$ with $f_{03}$), $\int_0^\pi d\theta$, and
   $r=\sqrt{k^2+q^2-2kq\cos\theta}$ made explicit.
4. **Seed variable and spectrum input** (B5, A8): is the NUM 03 symbol $\zeta^*$ or $S^*$; what is
   $S^*$ in terms of $\zeta$ (sign included); is $P_*$ the primordial $P_\zeta$; is the deliverable
   $P^h_{22}$ per polarisation, summed, or $\mathcal P_h$ (A6).
5. **Evaluation time of $w$ in the source** (B8): confirm $w_0=w(z')$ inside $f$ and $w^*$ at
   $z_{\rm init}$ in the prefactor; confirm that the code's $c_s^2$ in the transfer-function ODE is
   $w(z)$ as spec 01 states (A7).
6. **Lower limit** (open 6): define $\eta_0$/$z_{\rm init}$ and whether the analytic TARGET assumes
   $\eta_0\to0$.
7. **$k_{\rm phys}$ in NUM 05/10/11** (B9): confirm $k\to k/a_0$ (or $a_0=1$) in the WKB Green's
   function; the transfer-function WKB (spec 01 R29, spec 02 R43) already has $a_0$ explicit.
8. **Spec 01 Q7** (two coefficients in $d\omega_{\rm eff}/dz$ for the transfer function differ
   between NUM 09 p.5 and p.6): not cross-spec, but it is the only unconfirmed formula on the
   transfer-function WKB path and no other document checks it.
9. **Notation traps to record in the audit** (B10): $\epsilon$, $\eta$, $\omega_{\rm eff}$, $Q_s$
   ($q$ vs $q_{\rm phys}$), $f$; and MAIN 11's $\Phi'$ (B4, use MAIN 14).
