# 04 — Source time integral (independent duplicate transcription B)

Transcribed 2026-09-07 from the page images only. This is the `-B` duplicate of Group 4; it
covers `NUM` 06 only (not `NUM` 07). Not signed off.

## 1. Source documents

| Doc | File | Pages | Date on p.1 |
|---|---|---|---|
| `NUM` 06 | `Numerical implementation/06 - 2024-10-29 - analytic source integral.pdf` | 11 | 29 Oct 2024 |

Title on p.1: "Analytic quadratic source integral". Structure: STEP 1 (p.1–3, set-up and fixed-$w$
background), STEP 2 (p.3–6, simplify the source term), STEP 3 (p.6–7, collect the time integral),
STEP 4 (p.7–9, form suitable for Levin integration), and an untitled-step section
"Evaluation of 3-Bessel integrals" (p.10–11).

## 2. Conventions in force

- **Time variable.** The integral "we compute numerically" is written in redshift $z$, with limits
  $z$ (lower) to $z_{\rm init}$ (upper) (NUM 06 p.1, p.6). The analytic work is in conformal time
  $\eta$, introduced through $dt = a\,d\eta$ (NUM 06 p.2). The change of variable is
  $dz = -a_0 H\,d\eta$ (NUM 06 p.6), which flips the limits to $\eta_{\rm init}$ (lower) to $\eta$
  (upper) (NUM 06 p.7). Cosmic time $t$ appears only via $\dot a$ (p.2, p.6).
- **Prime.** On the scale factor, $a' = da/d\eta$ (explicit from $\mathcal{H} = a'/a$ and the
  $\eta$ derivation on NUM 06 p.2). On the transfer functions $T_q', T_r'$ (p.1, p.3) the prime is
  *not* defined on the page; it is inferred to be $d/d\eta$ from the combination $T'+\mathcal{H}T$
  and from the way $T'/\mathcal{H}$ is converted to $\frac{x}{1+b}\,dT/dx$ on p.4 (with
  $x = qc_s\eta$). On p.7–9 (Bessel/Liouville–Green section) the prime is $d/dx$ (inferred; the
  independent variable there is $x$).
- **$\mathcal{H}$ vs $H$.** Both are used. $H = \dot a/a$ in $3H^2M_P^2 = \rho$ (NUM 06 p.2);
  $\mathcal{H} = a'/a$ (NUM 06 p.2, p.3). Relations used: $\mathcal{H} = \frac{2}{1+3w}\frac1\eta$
  (p.2), $\mathcal{H} = \frac{1+b}{\eta}$ (p.3), $\mathcal{H}^2/a^2 = \rho_0/(3M_P^2)$ (p.3), and
  $dz = -\frac{a_0}{a}\mathcal{H}\,d\eta = -a_0 H\,d\eta$ (p.6). No $\epsilon$ is used.
- **Scale-factor normalisation.** $a_0$ appears explicitly: $1+z = a_0/a$ (NUM 06 p.6), and the
  final result carries an overall factor $a_0$ (p.7). $a_0 = 1$ is *not* assumed. On p.2
  $\rho_0$ is the density at $a = a_0$: $3M_P^2(\dot a/a)^2 = \rho_0 (a_0/a)^{3(1+w)}$.
- **Green's function.** $G_k(z,z')$ is "my Green's function defined with source $-\delta(z-z')$"
  (NUM 06 p.1). The first argument is the response time, the second the source time. Relation to
  the analytic ("them") Green's function in $\tau$ [sic; the formula is then written in $\eta$]:
  $G_{\rm me} = -H(z')\,G_{\rm them}$ (p.1), i.e. the conversion factor is evaluated at the
  *source* time. $G_{\rm them}$ is retarded: non-zero for $\eta > \eta'$, zero for $\eta < \eta'$
  (p.1; see R3 for the explicit form).
- **Equation of state.** Fixed $w$ throughout ("During an epoch of fixed w", NUM 06 p.2).
  $b = \frac{1-3w}{1+3w}$ (p.3), with derived identities $w = \frac13\frac{1-b}{1+b}$,
  $3(1+w) = \frac{2(2+b)}{1+b}$, $\frac{2}{1+3w} = 1+b$ (all p.3). The sound speed $c_s$ appears in
  every Bessel argument ($k c_s \eta$, $q c_s \eta$, $r c_s\eta$) but **is not defined in this
  document**; $c_s^2 = (1-b)/(3(1+b))$ is not written anywhere (inferred convention from the
  README, not from the page). On p.3 the source prefactor is written once with $1+w_0$
  (subscript 0) and otherwise with $1+w$; see Open questions.
- **Fourier / power-spectrum conventions.** Not stated anywhere in this document. No
  $(2\pi)^3$ factors, $\delta$-functions or polarisation tensors appear. Not inferable.
- **Transfer function.** Written $T_k(\eta)$, with no statement of whether it is the $\phi$,
  $\Phi$, $\zeta$ or $\mathcal R$ transfer function. Explicit form
  $T_k(\eta) = 2^{3/2+b}\,\Gamma(\tfrac52+b)\,(kc_s\eta)^{-3/2-b} J_{3/2+b}(kc_s\eta)$ (NUM 06 p.1,
  repeated p.4). The normalisation $T_k \to 1$ as $\eta \to 0$ is *inferred* from the small-argument
  limit of this expression (the page does not state it).
- **Momentum labels.** The source is written $f(z'\,|\,\mathbf q, \mathbf k-\mathbf q)$ (NUM 06
  p.1) and thereafter in terms of $T_q$ and $T_r$ with Bessel arguments $qc_s\eta$, $rc_s\eta$. The
  symbol $r$ is never defined on the page; it is inferred to mean $r = |\mathbf k-\mathbf q|$. No
  angular reduction or loop integration appears in this document. $M_P$ is the (reduced) Planck
  mass, from $3H^2M_P^2 = \rho$ (p.2).

## 3. Results

### STEP 1 — set-up (NUM 06 p.1–3)

**R1** (NUM 06 p.1). The source integral computed numerically:
$$
\int_z^{z_{\rm init}} dz'\; G_k(z,z')\,\frac{1+z}{1+z'}\,\frac{1}{H(z')^2}\; f(z'\,|\,\mathbf q,\mathbf k-\mathbf q).
$$
Definition of the quantity to be evaluated analytically; $G_k$ has source $-\delta(z-z')$ and $f$ is
"the tensor source". Confidence: **high**.

**R2** (NUM 06 p.1). Relation between the author's Green's function and the analytic one:
$$
G_{\rm me} = -\,H(z')\,G_{\rm them}.
$$
Confidence: **high** (sign clearly negative; argument of $H$ is primed $z'$).

**R3** (NUM 06 p.1). The analytic Green's function:
$$
G_{\rm them} = \frac{\pi}{2}\sqrt{\eta\eta'}\;
\begin{cases}
-\,Y_{b+\frac12}(k\eta')\,J_{b+\frac12}(k\eta) + J_{b+\frac12}(k\eta')\,Y_{b+\frac12}(k\eta), & \eta > \eta' \\[2pt]
0, & \eta < \eta'.
\end{cases}
$$
Retarded Green's function for the tensor mode equation in a fixed-$w$ background. All four orders
are written $b+\tfrac12$ (index written as "b + ½" with the ½ as a stacked fraction). Confidence:
**high** for the orders and the arguments; **medium** for the operator between the first two
Bessel factors — there is a small struck-through glyph between $Y_{b+1/2}(k\eta')$ and
$J_{b+1/2}(k\eta)$ (see Corrections C1); read as a plain product.

**R4** (NUM 06 p.1; repeated p.4 as $T_k(x)$). Transfer function:
$$
T_k(\eta) = 2^{\frac32+b}\,\Gamma\!\left(\tfrac52+b\right)\,(kc_s\eta)^{-\frac32-b}\,J_{\frac32+b}(kc_s\eta).
$$
Confidence: **high** (exponent $3/2+b$ on the 2, argument $\frac52+b$ in $\Gamma$, power
$-3/2-b$, order $3/2+b$ all clearly legible at zoom).

**R5** (NUM 06 p.1). The tensor source in terms of transfer functions:
$$
f = T_qT_r + \frac{2M_P^2}{\rho_0\left(1+\frac{p_0}{\rho_0}\right)}\;\frac{1}{a^2}\,
\big(T_q' + \mathcal{H}T_q\big)\big(T_r' + \mathcal{H}T_r\big).
$$
Confidence: **medium**. The denominator is written "$\rho_0(1+{}^{p_0}\!/_{\rho_0})$" with the
ratio in very small script; the reading $p_0/\rho_0$ (so that the bracket is $1+w$, as used on p.3)
is the most likely, but the small glyphs could also be read as "$P_0/P_0$". Everything else is
clear.

**R6** (NUM 06 p.2). Fixed-$w$ background in conformal time. From $3M_P^2(\dot a/a)^2 =
\rho_0(a_0/a)^{3(1+w)}$ and $dt = a\,d\eta$:
$$
a^{\frac{1+3w}{2}} \propto \eta, \qquad a \propto \eta^{\frac{2}{1+3w}}, \qquad
\mathcal{H} = \frac{a'}{a} = \frac{2}{1+3w}\,\frac{1}{\eta}.
$$
Confidence: **high** for these three lines. (The intermediate integration line reads
$\frac{a^{1/2+3w/2}}{\frac12+\frac{3w}{2}} = \sqrt{\frac{\rho_0}{3M_P^2}}\,a_0^{-\frac12+\frac{3w}{2}}(\eta-\eta')$
— the "$(\eta-\eta')$" is presumably an integration constant; the author then writes "so assume
$a^{(1+3w)/2}\propto\eta$". Exponents here were corrected, see C2.)

**R7** (NUM 06 p.3). Identities in $b$:
$$
b = \frac{1-3w}{1+3w},\qquad w = \frac13\,\frac{1-b}{1+b},\qquad 3(1+w) = \frac{4+2b}{1+b} = \frac{2(2+b)}{1+b},
\qquad \frac{2}{1+3w} = 1+b,\qquad \mathcal{H} = \frac{1+b}{\eta}.
$$
Confidence: **high**.

### STEP 2 — simplify the source term (NUM 06 p.3–6)

**R8** (NUM 06 p.3). Source with the background eliminated. Starting from
$f = T_qT_r + \frac{2M_P^2}{\rho_0(1+w)}\frac{\mathcal{H}^2}{a^2}\big(\frac{T_q'}{\mathcal H}+T_q\big)\big(\frac{T_r'}{\mathcal H}+T_r\big)$
and $3H^2M_P^2 = \rho_0 \Rightarrow 3M_P^2\,\mathcal{H}^2/a^2 = \rho_0 \Rightarrow
\frac{1}{\rho_0}\frac{\mathcal H^2}{a^2} = \frac{1}{3M_P^2}$:
$$
f = T_qT_r + \frac{2M_P^2}{3M_P^2}\,\frac{1}{1+w_0}\,
\left(\frac{T_q'}{\mathcal H}+T_q\right)\left(\frac{T_r'}{\mathcal H}+T_r\right).
$$
Confidence: **high** for the structure; the subscript on $w_0$ is clearly written on this line
only (see Open questions Q3). Note that on this page $\rho_0$ is used for the density at the time
$\eta$ (in $3H^2M_P^2 = \rho_0$), whereas on p.2 it was the density at $a_0$ (Q2).

**R9** (NUM 06 p.4). Derivative of the transfer function. With the Bessel identity
$$
Z_\alpha'(x) - \frac{\alpha}{x}Z_\alpha(x) = -Z_{\alpha+1}(x)\quad\text{"for any Bessel function }Z_\alpha\text{"},
$$
$$
\frac{dT}{dx} = -\,2^{\frac32+b}\,\Gamma\!\left(\tfrac52+b\right)\,x^{-\frac32-b}\,J_{\frac52+b}(x).
$$
Confidence: **high** for the boxed result (order $\tfrac52+b$ clear). The intermediate line is
written $\frac{dT}{dx} = 2^{3/2+b}\Gamma(\frac52+b)x^{-3/2-b}\{J'_{3/2+b}(x) - (\frac32+b)J_{3/2+b}(x)\}$
with no visible $1/x$ on the second term (Q4).

**R10** (NUM 06 p.4). Source in Bessel form, first version. With
$\frac{2}{3(1+w)} = \frac{1+b}{2+b}$ (written "$+\,2\,\frac{1+b}{2(2+b)}$" after a correction, C3):
$$
f = 2^{3+2b}\,\Gamma\!\left(\tfrac52+b\right)^2 (qc_s\eta)^{-\frac32-b}(rc_s\eta)^{-\frac32-b}
\Big\{ J_{\frac32+b}(qc_s\eta)J_{\frac32+b}(rc_s\eta)
$$
$$
\qquad + \frac{1+b}{2+b}\Big[-\frac{qc_s\eta}{1+b}J_{\frac52+b}(qc_s\eta) + J_{\frac32+b}(qc_s\eta)\Big]
\Big[-\frac{rc_s\eta}{1+b}J_{\frac52+b}(rc_s\eta) + J_{\frac32+b}(rc_s\eta)\Big]\Big\}.
$$
Expanded on the same page as
$$
= 2^{3+2b}\Gamma(\tfrac52+b)^2(qc_s\eta)^{-\frac32-b}(rc_s\eta)^{-\frac32-b}
\Big\{\Big(1+\frac{1+b}{2+b}\Big)J_{\frac32+b}(qc_s\eta)J_{\frac32+b}(rc_s\eta)
- \frac{qc_s\eta}{2+b}J_{\frac52+b}(qc_s\eta)J_{\frac32+b}(rc_s\eta)
- \frac{rc_s\eta}{2+b}J_{\frac52+b}(rc_s\eta)J_{\frac32+b}(qc_s\eta)
+ \frac{(qc_s\eta)(rc_s\eta)}{(2+b)(1+b)}J_{\frac52+b}(qc_s\eta)J_{\frac52}(rc_s\eta)\Big\}.
$$
Confidence: **high** for all orders except the very last factor, which is written
$J_{\frac52}(rc_s\eta)$ with **no** "$+b$" (medium; almost certainly an omission — the same
factor carries $+b$ on p.5 — but transcribed as written). In the first bracketed form the
order of the last $J$ in the second square bracket was over-written ("$J_{3\to\frac32+b}$",
C4).

**R11** (NUM 06 p.5). Intermediate regrouped forms (both on p.5), then the second Bessel
identity. First,
$$
f = 2^{3+2b}\,\frac{3+2b}{2+b}\,\Gamma(\tfrac52+b)^2(qc_s\eta)^{-\frac32-b}(rc_s\eta)^{-\frac32-b}
\Big\{J_{\frac32+b}(qc_s\eta)J_{\frac32+b}(rc_s\eta)
- \frac{qc_s\eta}{3+2b}J_{\frac52+b}(qc_s\eta)J_{\frac32+b}(rc_s\eta)
- \frac{rc_s\eta}{3+2b}J_{\frac52+b}(rc_s\eta)J_{\frac32+b}(qc_s\eta)
+ \frac{(qc_s\eta)(rc_s\eta)}{(3+2b)(1+b)}J_{\frac52+b}(qc_s\eta)J_{\frac52+b}(rc_s\eta)\Big\},
$$
then factorised as
$$
= 2^{3+2b}\,\frac{3+2b}{2+b}\,\Gamma(\tfrac52+b)^2(qc_s\eta)^{-\frac32-b}(rc_s\eta)^{-\frac32-b}
\Big\{\Big(J_{\frac32+b}(qc_s\eta) - \frac{qc_s\eta}{3+2b}J_{\frac52+b}(qc_s\eta)\Big)
\Big(J_{\frac32+b}(rc_s\eta) - \frac{rc_s\eta}{3+2b}J_{\frac52+b}(rc_s\eta)\Big)
$$
$$
\qquad + (qc_s\eta)(rc_s\eta)\Big(\frac{1}{(3+2b)(1+b)} - \frac{1}{(3+2b)^2}\Big)J_{\frac52+b}(qc_s\eta)J_{\frac52+b}(rc_s\eta)\Big\}.
$$
"We also have"
$$
Z_\alpha(x) - \frac{x}{2\alpha}Z_{\alpha+1}(x) = \frac{x}{2\alpha}Z_{\alpha-1}(x).
$$
Confidence: **high**; the order of the $J$ in the first factor of the factorised form was
over-written from "3" to "5" (C5), corrected reading $\frac52+b$.

**R12** (NUM 06 p.5 bottom, then p.6 top). **Final closed form of the source.** p.5:
$$
f = 2^{3+2b}\,\frac{3+2b}{2+b}\,\Gamma(\tfrac52+b)^2(qc_s\eta)^{-\frac32-b}(rc_s\eta)^{-\frac32-b}
\Big\{\frac{qc_s\eta}{3+2b}J_{\frac12+b}(qc_s\eta)\;\frac{rc_s\eta}{3+2b}J_{\frac12+b}(rc_s\eta)
+ \frac{(qc_s\eta)(rc_s\eta)}{(3+2b)^2}\Big(\frac{3+2b}{1+b}-1\Big)J_{\frac52+b}(qc_s\eta)J_{\frac52+b}(rc_s\eta)\Big\},
$$
p.6 (boxed-style final line of STEP 2):
$$
\boxed{\;
f = \frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2 (qc_s\eta)^{-\frac12-b}(rc_s\eta)^{-\frac12-b}
\Big\{ J_{\frac12+b}(qc_s\eta)\,J_{\frac12+b}(rc_s\eta) + \frac{2+b}{1+b}\,J_{\frac52}(qc_s\eta)\,J_{\frac52}(rc_s\eta)\Big\}.\;}
$$
Confidence: **high** for the prefactor $2^{3+2b}/((3+2b)(2+b))$, the powers $-\frac12-b$, and the
orders $\frac12+b$ on the first product. **Medium** for the orders on the second product: on p.6
(both the top line and the copy at the bottom of the page) they are written $J_{\frac52}$ with
**no** "$+b$", whereas on p.5 (immediately above) and on p.7 (all copies) the same factors are
written $J_{\frac52+b}$. Transcribed as written; the intended order is evidently $\frac52+b$
(Q5). The struck glyphs on the p.5 line are recorded in C6.

### STEP 3 — collect the time integral (NUM 06 p.6–7)

**R13** (NUM 06 p.6). Change of variables:
$$
\frac{1+z}{1+z'} = \frac{a(z')}{a(z)} = \frac{a(\eta')}{a(\eta)} = \frac{(\eta')^{1+b}}{\eta^{1+b}},\qquad
1+z = \frac{a_0}{a},\qquad
dz = -\frac{a_0}{a}\frac{\dot a}{a}\,dt = -\frac{a_0}{a}\frac{\dot a}{a}\,a\,d\eta = -\frac{a_0}{a}\mathcal{H}\,d\eta = -a_0H\,d\eta.
$$
Confidence: **high**.

**R14** (NUM 06 p.6 bottom → p.7 top). The integral R1 in conformal time:
$$
\Rightarrow\; -a_0\int_\eta^{\eta_{\rm init}} H(\eta')\,d\eta'\; G_k(\eta,\eta')\,\frac{(\eta')^{1+b}}{\eta^{1+b}}\,\frac{1}{H(\eta')^2}\;
\frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma(\tfrac52+b)^2(qc_s\eta')^{-\frac12-b}(rc_s\eta')^{-\frac12-b}
\Big\{J_{\frac12+b}(qc_s\eta)J_{\frac12+b}(rc_s\eta) + \frac{2+b}{1+b}J_{\frac52+b}(qc_s\eta)J_{\frac52}(rc_s\eta)\Big\}
$$
(as written on p.6: the powers carry $\eta'$ but the Bessel arguments in the brace are written with
**unprimed** $\eta$; the last order is again $\frac52$ without $+b$ — Q5, Q6). Then, substituting
R2 and R3 (p.7):
$$
= a_0\int_{\eta_{\rm init}}^{\eta} d\eta'\,(-1)\,\frac{\pi}{2}\sqrt{\eta\eta'}\,
\Big\{J_{b+\frac12}(k\eta')\,Y_{b+\frac12}(k\eta) - Y_{b+\frac12}(k\eta')\,J_{b+\frac12}(k\eta)\Big\}
\cdot\frac{(\eta')^{1+b}}{\eta^{1+b}}\,\frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma(\tfrac52+b)^2
\cdot (qc_s\eta')^{-\frac12-b}(rc_s\eta')^{-\frac12-b}
$$
$$
\qquad\cdot\Big\{J_{\frac12+b}(qc_s\eta')J_{\frac12+b}(rc_s\eta') + \frac{2+b}{1+b}J_{\frac52+b}(qc_s\eta')J_{\frac52+b}(rc_s\eta')\Big\}.
$$
Confidence: **high** on p.7 (all arguments primed, all orders legible; the $(-1)$ is written
explicitly in parentheses before $\pi/2$; the overall sign $+a_0$ results from
$-a_0 \times (-H) \times$ flipping the limits). Note the order of the two terms inside the
Green's-function brace has been swapped relative to R3 but the expression is the same.

**R15** (NUM 06 p.7). **Final analytic source-integral formula** (end of STEP 3):
$$
\boxed{\;
\begin{aligned}
&= -\,a_0\,\frac{\pi}{2}\,\frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2\,(q\,r\,c_s^2\,\eta)^{-\frac12-b}\\
&\quad\times\Bigg\{\; Y_{b+\frac12}(k\eta)\int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,J_{b+\frac12}(k\eta')
\Big(J_{\frac12+b}(qc_s\eta')J_{\frac12+b}(rc_s\eta') + \frac{2+b}{1+b}J_{\frac52+b}(qc_s\eta')J_{\frac52+b}(rc_s\eta')\Big)\\
&\qquad\;\; -\, J_{b+\frac12}(k\eta)\int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,Y_{b+\frac12}(k\eta')
\Big(J_{\frac12+b}(qc_s\eta')J_{\frac12+b}(rc_s\eta') + \frac{2+b}{1+b}J_{\frac52+b}(qc_s\eta')J_{\frac52+b}(rc_s\eta')\Big)\Bigg\}.
\end{aligned}\;}
$$
Confidence: **high** for the prefactor sign ($-a_0\pi/2$), the weight $(\eta')^{1/2-b}$ (written
"$(\eta')^{\frac12-b}$" in both integrals, stacked fraction), the orders $b+\frac12$ on the
outer $Y$/$J$ and inner $J$/$Y$, and $\frac12+b$, $\frac52+b$ inside the round brackets.
**Medium** on two details: (i) the factor $(q\,r\,c_s^2\,\eta)^{-1/2-b}$ has a small struck-through
mark immediately after $\eta$ (C7; most likely a stray prime, since the $\eta$ here is the
external, unprimed time — consistent with $\sqrt{\eta}\,\eta^{-1-b} = \eta^{-1/2-b}$); (ii) in the
second term the kernel $Y_{b+\frac12}(k\eta')$ was originally written $J_{b+\frac12}(k\eta')$ and
over-written with $Y$ (C8); the corrected reading $Y$ is what is transcribed.

### STEP 4 — form suitable for Levin integration (NUM 06 p.7–9)

**R16** (NUM 06 p.7–8). Normal form of the Bessel equation. From
$y'' + \frac1x y' + \big(1-\frac{\nu^2}{x^2}\big)y = 0$, write $y = f\tilde y$, choose
$2f'/f = -1/x \Rightarrow f \propto x^{-1/2}$; then (renaming $\tilde y\to y$)
$$
y'' + \Big(1 + \frac{\tfrac14-\nu^2}{x^2}\Big)y = 0 .
$$
Confidence: **high**. (So the Bessel functions in R15 are $x^{-1/2}\times$ a solution of this
equation; the identification $\omega_{\rm eff}^2 = 1 + (\tfrac14-\nu^2)/x^2$ used on p.9 is not
written explicitly — inferred.)

**R17** (NUM 06 p.9). Liouville–Green phase. Writing $y \sim e^{i\Theta}$ gives the Riccati
equation $i\Theta'' - (\Theta')^2 + \omega_{\rm eff}^2 = 0$. With $\Theta = \vartheta + i\beta$:
$$
\vartheta'' - 2\vartheta'\beta' = 0,\qquad -\beta'' - \vartheta'^2 + \beta'^2 + \omega_{\rm eff}^2 = 0,
$$
$$
\beta' = \frac{\vartheta''}{2\vartheta'},\qquad \beta = \tfrac12\ln\vartheta',\qquad
\beta'' = \frac{\vartheta'''}{2\vartheta'} - \frac{\vartheta''}{2\vartheta'}\frac{\vartheta''}{\vartheta'},
$$
$$
-\frac{\vartheta'''}{2\vartheta'} + \frac34\,\frac{\vartheta''}{\vartheta'}\frac{\vartheta''}{\vartheta'} - \vartheta'^2 + \omega_{\rm eff}^2 = 0
\quad\text{("Kummer's diff. eq.")},
$$
"The leading solution is the WKB one: $-\vartheta'^2 + \omega_{\rm eff}^2 = 0$."
Confidence: **high**. (The symbol for the real phase is a script theta, transcribed $\vartheta$;
the symbol for the total phase is a circled/script capital, transcribed $\Theta$.)

### Evaluation of 3-Bessel integrals (NUM 06 p.10–11)

**R18** (NUM 06 p.10). Liouville–Green representation of the triple-Bessel integrand:
$$
\int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\,Z_1(x_1)Z_2(x_2)Z_3(x_3)
= \int_{\eta_{\rm init}}^{\eta} d\eta'\,(\eta')^{\frac12-b}\Big(\frac{2}{\pi}\Big)^{3/2}
\frac{1}{(x_1x_2x_3)^{1/2}}\,\frac{1}{\big(\beta_1(x_1)\beta_2(x_2)\beta_3(x_3)\big)^{1/2}}\,
\cos\gamma_1(x_1)\cos\gamma_2(x_2)\cos\gamma_3(x_3).
$$
Confidence: **high** for the reading. The functions $\beta_i(x_i)$ and $\gamma_i(x_i)$ are
**not defined** on the page (Q7); each $Z_i$ is presumably represented as
$\sqrt{2/(\pi x)}\,\beta(x)^{-1/2}\cos\gamma(x)$ (inferred). Note this $\beta$ is not the
$\beta = \frac12\ln\vartheta'$ of p.9.

**R19** (NUM 06 p.10). Product-to-sum, Bessel ($\cos\cos\cos$) case:
$$
\cos\gamma_1\cos\gamma_2\cos\gamma_3 = \tfrac14\Big\{\cos(\gamma_1+\gamma_2+\gamma_3) + \cos(\gamma_1+\gamma_2-\gamma_3)
+ \cos(\gamma_1-\gamma_2+\gamma_3) + \cos(\gamma_1-\gamma_2-\gamma_3)\Big\}.
$$
Confidence: **high** (no corrections on this block).

**R20** (NUM 06 p.10). "For the Neumann case" ($\sin\cos\cos$):
$$
\sin\gamma_1\cos\gamma_2\cos\gamma_3 = \tfrac12\big\{\sin(\gamma_1+\gamma_2)+\sin(\gamma_1-\gamma_2)\big\}\cos\gamma_3
= \tfrac14\Big\{\sin(\gamma_1+\gamma_2+\gamma_3) + \sin(\gamma_1+\gamma_2-\gamma_3)
+ \sin(\gamma_1-\gamma_2+\gamma_3) + \sin(\gamma_1-\gamma_2-\gamma_3)\Big\}.
$$
Confidence: **high** (no corrections on this block).

**R21** (NUM 06 p.11). $\sin\sin\sin$ case. First line:
$\sin\gamma_1\sin\gamma_2\sin\gamma_3 = \tfrac12\big(\cos(\gamma_1-\gamma_2)-\cos(\gamma_1+\gamma_2)\big)\sin\gamma_3$.
Second line (as written, after corrections C9, C10):
$$
= \tfrac14\Big(\sin(\gamma_1-\gamma_2+\gamma_3) - \sin(\gamma_1 \,[\mp]\, \gamma_2-\gamma_3)
- \sin(\gamma_1+\gamma_2+\gamma_3) + \sin(\gamma_1+\gamma_2-\gamma_3)\Big),
$$
final line (as written, C11):
$$
= \tfrac14\Big(-\sin(\gamma_1+\gamma_2+\gamma_3) + \sin(\gamma_1+\gamma_2\,[\mp]\,\gamma_3)
+ \sin(\gamma_1-\gamma_2+\gamma_3) - \sin(\gamma_1-\gamma_2-\gamma_3)\Big).
$$
Confidence: **medium**. The two glyphs marked $[\mp]$ are over-written signs (a "+" with an
extra horizontal bar); it cannot be determined from the ink alone whether "+" was corrected to
"−" or vice versa. For the identity to be correct both must read "−", i.e. second line
$-\sin(\gamma_1-\gamma_2-\gamma_3)$ and final line $+\sin(\gamma_1+\gamma_2-\gamma_3)$; that is the
reading a downstream user should verify against the page. Three of the four terms in the second
line carry small tick marks beneath them (the ones with unambiguous signs); the term with the
over-written sign is un-ticked.

**R22** (NUM 06 p.11). $\cos\sin\sin$ case:
$$
\cos\gamma_1\sin\gamma_2\sin\gamma_3 = \tfrac12\big(\sin(\gamma_1+\gamma_2)-\sin(\gamma_1-\gamma_2)\big)\sin\gamma_3
= \tfrac14\Big(\cos(\gamma_1+\gamma_2-\gamma_3) - \cos(\gamma_1+\gamma_2+\gamma_3)
- \cos(\gamma_1-\gamma_2-\gamma_3) + \cos(\gamma_1-\gamma_2+\gamma_3)\Big)
$$
$$
= \tfrac14\Big(-\cos(\gamma_1+\gamma_2+\gamma_3) + \cos(\gamma_1+\gamma_2-\gamma_3) + \cos(\gamma_1-\gamma_2+\gamma_3) - \cos(\gamma_1-\gamma_2-\gamma_3)\Big).
$$
Confidence: **high**; the third term of the middle line was written "$\sin$", struck through and
replaced by "$\cos$" written above (C12); the final line uses $\cos$.

## 4. Checks

Not applicable: `NUM` 06 does not recheck another document. Internal consistency notes are under
Open questions.

## 5. Corrections and cross-outs

- **C1** (p.1, R3). A small glyph between $Y_{b+1/2}(k\eta')$ and $J_{b+1/2}(k\eta)$ in
  $G_{\rm them}$ is struck through (looks like a stray "+" or "Y" with a slash). Corrected reading:
  plain product. Later steps (p.7) use the plain product.
- **C2** (p.2, R6). Exponent of $a$ in "so $a^{\cdots}da = \ldots$" was written $\frac12-\frac{3w}{2}$,
  struck through, with $-\frac12+\frac{3w}{2}$ written above. On the next line "so $a^{3/2-3w/2}$"
  is struck through entirely and replaced on the following line by
  $a^{\frac12+\frac{3w}{2}}/(\frac12+\frac{3w}{2})$. Later steps use the corrected forms
  ($a^{(1+3w)/2}\propto\eta$).
- **C3** (p.4, R10). Coefficient of the second term written "$+\,2\,\frac{1+b}{\not 3\;2(2+b)}$": a
  "3" in the denominator is struck through and "$2(2+b)$" written beside it, giving
  $\frac{2(1+b)}{2(2+b)} = \frac{1+b}{2+b}$, which is what the next line uses.
- **C4** (p.4, R10). In the second square bracket, the order of the last $J$ ("$J_{\ldots+b}(rc_s\eta)$")
  has a digit over-written; final reading $\frac32+b$, consistent with the expanded line below.
- **C5** (p.5, R11). In the factorised form, the order of the second $J$ in the first factor is
  written with a "3" over-written by "5"; corrected reading $J_{\frac52+b}(qc_s\eta)$, consistent
  with the line above and the identity that follows.
- **C6** (p.5, R12 first form). Three small struck-through marks: (a) between
  $\frac{qc_s\eta}{3+2b}J_{\frac12+b}(qc_s\eta)$ and $\frac{rc_s\eta}{3+2b}J_{\frac12+b}(rc_s\eta)$ (a
  struck "+", making the two factors a product); (b) inside the last argument, a struck dot/glyph
  before $rc_s\eta$; (c) between $\frac{(qc_s\eta)(rc_s\eta)}{(3+2b)^2}$ and the bracket
  $\big(\frac{3+2b}{1+b}-1\big)$ (a struck glyph, read as a product). The p.6 line uses the
  product readings.
- **C7** (p.7, R15). In $(q\,r\,c_s^2\,\eta\,\cdot)^{-\frac12-b}$ a small mark after $\eta$ is struck
  through (probably a prime). Corrected reading: unprimed $\eta$.
- **C8** (p.7, R15). Second term: the kernel inside the integral was written
  $J_{b+\frac12}(k\eta')$; the $J$ is struck and "$Y$" written above. Corrected reading
  $Y_{b+\frac12}(k\eta')$ (as required by R14).
- **C9** (p.11, R21). Prefactor of the second line written $\frac12$, the "2" struck and "4" written
  below: corrected reading $\frac14$.
- **C10** (p.11, R21, second line). Sign inside $\sin(\gamma_1\,\cdot\,\gamma_2-\gamma_3)$ over-written
  ("+" with an extra bar). Original and correction cannot be distinguished; see R21.
- **C11** (p.11, R21, final line). Sign inside $\sin(\gamma_1+\gamma_2\,\cdot\,\gamma_3)$ over-written
  in the same way. See R21.
- **C12** (p.11, R22). Third term of the middle line: "$\sin$" struck through, "$\cos$" written above.
  Final line uses $\cos$.

## 6. Open questions

- **Q1** ($c_s$ undefined). $c_s$ appears in every transfer-function argument but is never defined in
  this document; the relation $c_s^2 = (1-b)/(3(1+b))$ is not written. Must be taken from another
  spec.
- **Q2** ($\rho_0$ used with two meanings). On p.2, $\rho_0$ is the density at $a = a_0$
  ($\rho = \rho_0(a_0/a)^{3(1+w)}$). On p.3 the author writes $3H^2M_P^2 = \rho_0$ and
  $3M_P^2\mathcal{H}^2/a^2 = \rho_0$ at a general time, and uses this to eliminate the
  $2M_P^2/(\rho_0(1+w))\,\mathcal{H}^2/a^2$ prefactor to $\frac{2}{3(1+w)}$. The final results
  (R12, R15) depend only on the combination $\frac{2}{3(1+w)} = \frac{1+b}{2+b}$, which is
  what one gets if $\rho_0$ on p.3 means $\rho(\eta)$. Transcribed as written; author to confirm.
- **Q3** ($w_0$ vs $w$). R8 on p.3 has "$1+w_0$" (subscript 0 visible) where every other
  occurrence is "$1+w$". Likely the same quantity in a fixed-$w$ epoch; not resolvable from the
  page.
- **Q4** (missing $1/x$ in intermediate line, p.4). The line
  $\frac{dT}{dx} = 2^{3/2+b}\Gamma(\frac52+b)x^{-3/2-b}\{J'_{3/2+b}(x) - (\frac32+b)J_{3/2+b}(x)\}$
  shows no $1/x$ on the second term, but the identity applied next has $\frac{\alpha}{x}$ and the
  result R9 is the one that follows *with* the $1/x$. Intermediate algebra only; recorded because
  the primary transcription may render this line differently.
- **Q5** ($J_{5/2}$ vs $J_{5/2+b}$). The second Bessel product in the final source formula is
  written $J_{\frac52}(qc_s\eta)J_{\frac52}(rc_s\eta)$ (no $+b$) at the top of p.6 and again at the
  bottom of p.6 (there as $J_{\frac52+b}(\cdot)J_{\frac52}(\cdot)$), and once on p.4 (last factor).
  Everywhere else — p.5 (twice) and p.7 (four times, in the final result R15) — it is
  $J_{\frac52+b}$. The derivation (R9–R11) produces $\frac52+b$. Transcribed as written in each
  place; the final formula R15 has $+b$ on the page.
- **Q6** (primed vs unprimed $\eta$ in Bessel arguments, p.6 bottom). In the first $\eta$-form of
  the integral (R14, p.6) the powers are written with $\eta'$ but the Bessel arguments inside the
  brace with unprimed $\eta$ ($J_{\frac12+b}(qc_s\eta)$ etc.). On p.7 every argument inside the
  integrand is primed. The p.7 (primed) version is the one that feeds R15.
- **Q7** ($\beta_i$, $\gamma_i$ undefined, p.10). The Liouville–Green amplitude functions
  $\beta_i(x_i)$ and phases $\gamma_i(x_i)$ in R18 are not defined in this document. The natural
  reading (each $Z(x) \approx \sqrt{2/(\pi x)}\,\beta(x)^{-1/2}\cos\gamma(x)$ for $J$, and
  $\sin\gamma$ for $Y$, "the Neumann case") is an inference. Presumably defined in `NUM` 07 or
  `NUM` 05.
- **Q8** (sign glyphs, p.11). See R21 / C10 / C11: two over-written signs in the
  $\sin\sin\sin$ expansion whose direction of correction cannot be read from the ink. The
  trigonometrically correct identity requires both to be "−"; this is stated as a check for the
  author, not as a resolution.
- **Q9** (the p.1 relation is described as being "in $\tau$" but written in $\eta$). Minor: the
  sentence "the analytic Green's function in $\tau$ is" precedes a formula written entirely in
  $\eta,\eta'$. Taken to mean conformal time in both cases.
- **Q10** (R15 sign check, for the reviewer). The route p.6 → p.7 is $-a_0\int_\eta^{\eta_{\rm init}}
  (H\,d\eta')\,G_k\,\frac{1}{H^2}\cdots$ with $G_k = -H\,G_{\rm them}$, giving
  $+a_0\int_\eta^{\eta_{\rm init}} G_{\rm them}\cdots = -a_0\int_{\eta_{\rm init}}^{\eta}G_{\rm them}\cdots$,
  and the page writes this as "$a_0\int_{\eta_{\rm init}}^{\eta} d\eta'(-1)\frac{\pi}{2}\cdots$".
  The overall sign of R15 as written ($-a_0\frac{\pi}{2}\cdots$) is consistent with this, with the
  brace ordered $\{Y(k\eta)\int J(k\eta')\cdots - J(k\eta)\int Y(k\eta')\cdots\}$. Recorded so the
  diff against the primary can locate any sign disagreement quickly.
