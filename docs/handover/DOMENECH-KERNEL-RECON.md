# Domènech reconnaissance: what the general-$b$ kernel actually says

**Prompt:** [`prompts/handover/00-domenech-reconnaissance.md`](../../prompts/handover/00-domenech-reconnaissance.md)
· **Kind:** read-and-report; no code lands · **Written:** 2026-09-19, by Claude Fable 5.1
**Sources:** the LaTeX in [`sources/`](sources/), checksums per [`sources/SOURCES.md`](sources/SOURCES.md),
all eight `shasum -c CHECKSUMS` lines `OK` at the time of writing. Cited below as `1912.tex:NNN`
(`1912.05583-src/inducedSGWBwarxiv_revised_2.tex`) and `review.tex:NNN`
(`2109.01398-src/template_review.tex`). DLMF is cited by equation number from
<https://dlmf.nist.gov/14>, read on the same day.
**Self-checks:** `mpmath 1.3.0`, `scipy 1.15.2`, throwaway scripts in the session scratch
directory, **not** in the repository. Every number below that came from them is marked *[checked]*
and carries the quadrature's own declared error or the precision it was run at. §10 item 8 says
what is worth landing.

---

## 0. What this is, what it is not, and the provenance of every object discussed

This report exists because `prompts/radiation-oracle` found three errata in Kohri & Terada by
numerical disagreement, none by reading. Its job is to make A1's numerical cross-check decisive:
say which convention every symbol is in, where the branch cuts are, what the papers disagree
about, and what a correct implementation must reproduce. **It does not certify equations.** A1 is
still bound by [`README.md`](../../prompts/handover/README.md) §5.1 to check its transcription
against an independent quadrature of (4.11) `eq:Isimpledef`, whatever this document says.

Three kinds of statement appear, and each is labelled at the point it is made:

- **[paper]** — what the LaTeX says, quoted with `file:line`.
- **[derived]** — what I derived from the papers' own definitions by algebra, and where the
  derivation is more than two lines, checked numerically *[checked]*.
- **[inferred]** — a reading or a guess. These are collected in §10 and appear in the body only
  where flagged.

Objects, and whose they are:

| Object | Definition | Whose |
|---|---|---|
| $I_{\rm rev}(x,u,v)$ | $\int_0^x d\tilde x\,G(x,\tilde x)f(\tilde x,u,v)$ with the review's $G$ (4.7) `eq:hgreen` and $f$ (4.9) `eq:fsimple`, in $b,c_s$ | Domènech 2021 (review) |
| $I_{2020}(x,u,v)$ | $2\big(\tfrac{3+3w}{5+3w}\big)^2\int_0^x d\tilde x\,G f$, `1912.tex:191-193` | Domènech 2020 |
| $I_{\rm KT}$ | Kohri & Terada eq. (15), transcribed in `ComputeTargets/tests/kohri_terada.py` | radiation-oracle campaign |
| `total` | the pipeline's source time integral, `QuadSourceIntegral.py:756` | this repository |
| **E** | my `scipy.quad` of $I_{\rm rev}$'s defining integral, built from (4.7) and (4.9) only *[checked]* | this report |
| **target** | (4.10) with the $x\to\infty$ coefficients substituted, §2.4 below | what A1 builds |

The relations between the first four are derived in §7 and §8 and are the load-bearing content:
$I_{2020} = 2c^2 I_{\rm rev} = I_{\rm KT}$ and `total` $= -k_{\rm phys}^{-2} I_{\rm rev}$, with
$c = (2+b)/(3+2b)$.

---

## 1. Headline

**The target object is well defined, and it reduces to the review's (4.12) `eq:Isimple2` only
after two sign corrections to the review.** Both are in the LaTeX, both are invisible in the
review's own use of the result (it only ever squares $I$), and both would put a transcription
from the review off by an $O(1)$ amount that no $O(1/x)$ argument can excuse.

1. **The review's exact kernel (4.10) `eq:Isimple` has the wrong overall sign.** Its Green's
   function (4.7) `eq:hgreen` and source (4.9) `eq:fsimple` give
   $I = \mathcal N\,(c_s^2uvx)^{-b-1/2}\big(Y_{b+1/2}(x)\mathcal I_J - J_{b+1/2}(x)\mathcal I_Y\big)$;
   (4.10) prints $\big(J_{b+1/2}\mathcal I_Y - Y_{b+1/2}\mathcal I_J\big)$
   (`review.tex:650`). The 2020 paper's (3.1) `eq:kernel2` has the correct order,
   $\{Y_\beta\mathcal I^x_J - J_\beta\mathcal I^x_Y\}$ (`1912.tex:213`). *[derived, §4.1;
   checked to $10^{-15}$ on nine $(b,u,v)$ cases, §4.1 table]*
2. **The review's doubly asymptotic (4.12) has the wrong sign on its $\cos$ term** relative to
   its $\sin$ terms. Its $\sin$ terms are those of the correctly signed kernel; its $\cos$ term
   is that of the wrongly signed (4.10). So (4.12) is consistent with *neither* sign of (4.10).
   The 2020 paper's (3.6)–(3.8) are internally consistent and reduce at $w=1/3$ to Kohri &
   Terada's eq. (25) exactly, as the paper claims (`1912.tex:342`). *[derived, §2.4; checked:
   on-cut cases at $b = 0.2, 0.5$ miss the exact quadrature by 1.2–1.9 of the envelope as
   printed and by the expected $O(1/x)$ with the $\cos$ sign flipped, §2.4 table]*

Consequences for [`README.md`](../../prompts/handover/README.md) §2, stated loudly as asked:

- **(h) "The review is therefore the better transcription target" — disagree.** The review's
  variables are the right ones, but its two displayed forms of $I$ are the ones with sign slips.
  A1 should transcribe in $b$ and $c_s$ **with the 2020 paper's ordering**, i.e. the corrected
  form written out in §9, and verify against both the 2020 paper (through $I_{2020} = 2c^2
  I_{\rm rev}$) and Kohri & Terada (through $I_{\rm rev} = \tfrac98 I_{\rm KT}$ at $b=0$).
- **(i2) item 1 — the two prefactors are the same object up to $2c^2$, and the power of
  $(c_s^2uvx)$ does not differ.** $(uvwx)^{-\beta}$ with $w = c_s^2$ and $\beta = b+\tfrac12$
  *is* $(c_s^2uvx)^{-b-1/2}$. The $\Gamma^2[\beta+2]$ against $\Gamma^2[b+\tfrac32]$ is
  compensated exactly by $\alpha^3$ and $(2b+3)$, and what remains is the factor
  $2\big(\tfrac{3+3w}{5+3w}\big)^2 = 2c^2$ that the 2020 paper puts *inside its definition of $I$*
  (`1912.tex:192`) and the review does not (`review.tex:535`). The "opposite order" is the sign
  error in item 1 above. §4.1.
- **(i2) item 2 — the asymmetric factor of 2 is correct in both papers.** It is
  $\Gamma[\nu-\rho+1]$ at $\nu-\rho = 2$ in the Gervois–Navelet off-cut formula
  (`1912.tex:694`, `review.tex:1957`). *[derived, §4.2; checked: with the 2 the target
  approaches the exact quadrature as $1/x$ to $1.7\times10^{-5}$ of the envelope at $x =
  25600$; with 1 it stalls at $8\times10^{-2}$]*
- **(i3) — confirmed**, with the 2020 paper's displayed "$y\to1$" (`1912.tex:291-292`) read as
  $y\to-1^+$, which is what its text says (`1912.tex:280, 284`). Power-divergent
  $\propto(1+y)^{-|b|}$ for $b<0$, logarithmic at $b=0$, finite for $b>0$. Two additions: the
  divergence is in the **coefficients**, so the *target object* diverges at the resonance for
  $b\le0$ while the exact finite-$x$ kernel does not; and for $b>0$ the finite combination is
  written in closed form in §5.2 and evaluable *at* $y=-1$. No fixture case is near the
  resonance (§5.4).
- **(i) — the $O(1/x)$ is not uniform.** The control parameter is the smallest Bessel argument
  in the integrand, $c_s\min(u,v)\,x$, and near the resonance $|c_s(u+v)-1|\,x$. On the
  `q-smooth` shape ($u = 0.01$) the target is still $5\times10^{-2}$ of the envelope off the
  exact kernel at $x = 3200$ (§6.4). A1's decisive normalisation test must use the finite-$x$
  integrals (4.11) by quadrature, and the target object only for large-$x$ regression.

Everything else in §2 holds as stated. In particular (h)'s reparametrisation $\beta = b+\tfrac12$,
$\alpha = b+\tfrac32$, $\tfrac{3(1+w)}{2} = \tfrac{b+2}{b+1}$ is confirmed at every step below.

---

## 2. The two constructions, in $b$ and $c_s$, with the translation shown

### 2.1 Conventions the review fixes

**[paper]** $a\propto\tau^{1+b}$, $b = \tfrac{1-3w}{1+3w}$ (`review.tex:273-275`). The first-order
solution with inflationary initial conditions is
$$\Phi(k\tau) = \Phi_{\mathbf k}\,2^{b+3/2}\Gamma[b+\tfrac52]\,(c_sk\tau)^{-b-3/2}J_{b+3/2}(c_sk\tau)$$
(`review.tex:621-623`), i.e. the transfer function $T_\Phi(x) = 2^{b+3/2}\Gamma[b+\tfrac52](c_sx)^{-b-3/2}
J_{b+3/2}(c_sx)$, which is spec 05's $\Phi(x)$ glyph for glyph
([`spec/05-one-loop.md`](../spec/05-one-loop.md) R20). The source is
$$f = T_\Phi(q\tau)T_\Phi(|\mathbf k-\mathbf q|\tau) + \frac{1+b}{2+b}\Big(T_\Phi+\frac{T'_\Phi}{\mathcal H}\Big)_q\Big(T_\Phi+\frac{T'_\Phi}{\mathcal H}\Big)_{|\mathbf k-\mathbf q|}$$
(`review.tex:497-499`), $x\equiv k\tau$ (`review.tex:636-638`), and the kernel
$I(\tau,k,u,v) \equiv \int_{\tau_i}^\tau d\tilde\tau\,G(\tau,\tilde\tau)f(\tilde\tau,k,u,v)$
(`review.tex:534-536`) with
$$G(\tau,\tilde\tau) = \frac{\pi}{2k}\frac{(k\tilde\tau)^{b+3/2}}{(k\tau)^{b+1/2}}\Big(J_{b+1/2}(k\tilde\tau)Y_{b+1/2}(k\tau) - J_{b+1/2}(k\tau)Y_{b+1/2}(k\tilde\tau)\Big)$$
(`review.tex:632-634`).

**[derived]** Two remarks on these. (a) The $I$ of `review.tex:535` carries $1/k^2$ (its $G$ has
$1/k$ and $d\tilde\tau = d\tilde x/k$); the $I(x,u,v)$ of (4.10) onwards is dimensionless and is
$k^2$ times it, $I_{\rm rev}(x,u,v) = \int_0^x d\tilde x\,G_x(x,\tilde x)f$, with
$G_x \equiv kG$. Only the dimensionless one enters `eq:Phgaussian` (`review.tex:526-528`), so
that is the object throughout. (b) $G$ is the causal Green's function of $h'' + 2\mathcal Hh' + k^2h = S$
(`review.tex:1701-1707`): $G_x(x,\tilde x)\approx x-\tilde x>0$ just after the source, because
$J_\nu(\tilde x)Y_\nu(x) - J_\nu(x)Y_\nu(\tilde x)\approx(x-\tilde x)\,\mathcal W\{J_\nu,Y_\nu\}
= (x-\tilde x)\tfrac{2}{\pi\tilde x}$. This sign is what §4.1 turns on.

**[derived]** With $\nu\equiv b+\tfrac32$, $\mathcal H = (1+b)/\tau$ and
$\tfrac{d}{dz}[z^{-\nu}J_\nu] = -z^{-\nu}J_{\nu+1}$,
$$T_\Phi + \frac{T'_\Phi}{\mathcal H} = \frac{2^\nu\Gamma[\nu+1]}{2b+3}\,z^{1-\nu}\Big[J_{b+1/2}(z) - \tfrac{b+2}{b+1}J_{b+5/2}(z)\Big],\qquad
T_\Phi = \frac{2^\nu\Gamma[\nu+1]}{2b+3}\,z^{1-\nu}\Big[J_{b+1/2}(z) + J_{b+5/2}(z)\Big],$$
$z = c_sx$, using $J_{\nu-1}+J_{\nu+1} = \tfrac{2\nu}{z}J_\nu$. The cross terms
$J_{b+1/2}J_{b+5/2}$ cancel between the two products and
$$f(x,u,v) = \frac{2^{2b+3}\Gamma^2[b+\tfrac52]}{(2b+3)(b+2)}(c_sx)^{-2b-1}(uv)^{-b-1/2}\Big(J_{b+1/2}(c_svx)J_{b+1/2}(c_sux) + \frac{b+2}{b+1}J_{b+5/2}(c_svx)J_{b+5/2}(c_sux)\Big),$$
which is (4.9) `eq:fsimple` (`review.tex:642-644`) exactly. The same algebra on the 2020 paper's
`eq:source` (`1912.tex:199-202`; `SOURCES.md` does not resolve §2 numbers, so §2 equations of the
2020 paper are cited by label and line only), with $\alpha = b+\tfrac32$, $\tfrac{4^\alpha}{6\alpha}
\tfrac{1+3w}{1+w} = \tfrac{2^{2b+3}}{(2b+3)(b+2)}$, $w = c_s^2$ and
$\tfrac{3(1+w)}{2} = \tfrac{b+2}{b+1}$, gives the same function: **$f_{2020} = f_{\rm rev}$**. The
factor $2c^2$ between the papers is not in $f$; it is in the 2020 definition of $I$
(`1912.tex:192`).

### 2.2 The exact kernel (4.10) with (4.11)

**[paper]** (4.10) `eq:Isimple`, `review.tex:649-651`:
$$I(x,u,v) = \pi4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}(c_s^2uvx)^{-b-1/2}\Big(J_{b+1/2}(x)\mathcal I_Y - Y_{b+1/2}(x)\mathcal I_J\Big),$$
(4.11) `eq:Isimpledef`, `review.tex:653-656`:
$$\mathcal I_{J/Y}\equiv\int_0^x d\tilde x\,\tilde x^{1/2-b}\begin{Bmatrix}J_{b+1/2}(\tilde x)\\ Y_{b+1/2}(\tilde x)\end{Bmatrix}\Big(J_{b+1/2}(c_sv\tilde x)J_{b+1/2}(c_su\tilde x) + \frac{b+2}{b+1}J_{b+5/2}(c_sv\tilde x)J_{b+5/2}(c_su\tilde x)\Big).$$

**[derived]** Inserting (4.9) into $\int_0^x G_x f$: the weights combine as
$\tilde x^{b+3/2}\cdot\tilde x^{-2b-1} = \tilde x^{1/2-b}$, matching (4.11); the constant is
$\tfrac\pi2\cdot\tfrac{2^{2b+3}\Gamma^2[b+5/2]}{(2b+3)(b+2)} = \pi4^b\Gamma^2[b+\tfrac32]\tfrac{2b+3}{b+2}
\equiv\mathcal N$ using $\Gamma[b+\tfrac52] = \tfrac{2b+3}{2}\Gamma[b+\tfrac32]$, matching (4.10);
and the Bessel order is
$$\int_0^x G_xf = \mathcal N\,(c_s^2uvx)^{-b-1/2}\Big(Y_{b+1/2}(x)\mathcal I_J - J_{b+1/2}(x)\mathcal I_Y\Big),$$
**the opposite of (4.10) as printed.** The corrected (4.10) is what "the exact kernel" means in
the rest of this document. The numerical confirmation is in §4.1.

### 2.3 The $x\to\infty$ coefficients, translated from (3.3)/(3.4)

**[paper]** (3.3) `eq:IJ` and (3.4) `eq:IY`, `1912.tex:247-255`:
$$\mathcal I^\infty_J = \frac{2^{-\beta}(wZ)^{\beta-1/2}}{\sqrt{\pi uvw}}\Big[\mathsf P^{-\beta+1/2}_{\beta-1/2}(y) + \tfrac{3(1+w)}{2}\mathsf P^{-\beta+1/2}_{\beta+3/2}(y)\Big]\Theta(u+v-w^{-1/2}),$$
$$\mathcal I^\infty_Y = -\frac{2^{-\beta+1}w^{\beta-1/2}}{\pi\sqrt{\pi uvw}}\Big\{Z^{\beta-1/2}\Big[\mathsf Q^{-\beta+1/2}_{\beta-1/2}(y) + \tfrac{3(1+w)}{2}\mathsf Q^{-\beta+1/2}_{\beta+3/2}(y)\Big]\Theta(u+v-w^{-1/2}) - \tilde Z^{\beta-1/2}\Big[\mathcal Q^{-\beta+1/2}_{\beta-1/2}(\tilde y) + 3(1+w)\mathcal Q^{-\beta+1/2}_{\beta+3/2}(\tilde y)\Big]\Theta(w^{-1/2}-u-v)\Big\},$$
with $y = \tfrac{u^2+v^2-w^{-1}}{2uv}$, $Z^2 = 4u^2v^2(1-y^2)$, $\tilde y = -y$,
$\tilde Z^2 = -Z^2$ (`1912.tex:257-260`).

**[derived]** Translation, term by term, with $\beta = b+\tfrac12$, $w = c_s^2$:
$-\beta+\tfrac12 = -b$; $\beta-\tfrac12 = b$; $\beta+\tfrac32 = b+2$; $\tfrac{3(1+w)}{2} =
\tfrac{b+2}{b+1}\equiv\rho$; $3(1+w) = 2\rho$; $Z^{\beta-1/2} = (2uv)^b(1-y^2)^{b/2}$;
$\tilde Z^{\beta-1/2} = (2uv)^b(y^2-1)^{b/2}$; $y = \tfrac{c_s^2(u^2+v^2)-1}{2c_s^2uv}$, which is
(4.13) `eq:y` (`review.tex:671-673`) in both of its forms; $\Theta(u+v-w^{-1/2}) =
\Theta[c_s(u+v)-1]$. The prefactors:
$\tfrac{2^{-\beta}w^{\beta-1/2}(2uv)^{b}}{\sqrt{\pi uvw}} = \tfrac{1}{\sqrt{2\pi}}(c_s^2uv)^{b-1/2}$ and
$\tfrac{2^{-\beta+1}w^{\beta-1/2}(2uv)^b}{\pi\sqrt{\pi uvw}} = \tfrac1\pi\sqrt{\tfrac2\pi}(c_s^2uv)^{b-1/2}$.
So, writing $\Theta_\pm$ for $\Theta[\pm(c_s(u+v)-1)]$,
$$\boxed{\;\mathcal I^\infty_J = \frac{(c_s^2uv)^{b-1/2}}{\sqrt{2\pi}}\,(1-y^2)^{b/2}\,\mathcal P(y)\,\Theta_+,\qquad
\mathcal I^\infty_Y = -\frac{(c_s^2uv)^{b-1/2}}{\pi}\sqrt{\frac2\pi}\Big[(1-y^2)^{b/2}\mathcal Q_F(y)\Theta_+ - (y^2-1)^{b/2}\mathcal Q_O(-y)\Theta_-\Big]\;}$$
$$\mathcal P\equiv\mathsf P^{-b}_b + \rho\,\mathsf P^{-b}_{b+2},\qquad
\mathcal Q_F\equiv\mathsf Q^{-b}_b + \rho\,\mathsf Q^{-b}_{b+2},\qquad
\mathcal Q_O(\tilde y)\equiv\mathcal Q^{-b}_b(\tilde y) + 2\rho\,\mathcal Q^{-b}_{b+2}(\tilde y).$$

**[derived]** The same expressions follow directly from the Gervois–Navelet formulas as the papers
quote them (`1912.tex:663-704`, `review.tex:1924-1971`) with $\rho_{\rm GN} = b+\tfrac12$, $\nu_{\rm GN}
\in\{b+\tfrac12, b+\tfrac52\}$, $c = 1$, $a = c_sv$, $b_{\rm GN} = c_su$
(`review.tex:1973-1976`; the 2020 paper uses $\beta$ for the order and its (3.2) for the
$\int_0^x$ definition, `1912.tex:216-226`): $\cos\varphi = \tfrac{a^2+b^2-c^2}{2ab} = y$, $\sin\varphi = \sqrt{1-y^2}$,
$\cosh\phi = -y$, $\sinh\phi = \sqrt{y^2-1}$, and in the off-cut case $\nu_{\rm GN}-\rho_{\rm GN}
\in\{0,2\}$ makes $\sin[(\nu-\rho)\pi] = 0$ (so $\mathcal I^\infty_J = 0$ there),
$\cos[(\nu-\rho)\pi] = 1$, and $\Gamma[\nu-\rho+1]\in\{\Gamma[1],\Gamma[3]\} = \{1, 2\}$. That
$\Gamma[3]$ is the asymmetric 2 (§4.2). Nothing in the translation is a choice.

### 2.4 The target object, and its reduction to (4.12)

**[derived]** Substituting §2.3 into the corrected (4.10), and using
$(c_s^2uvx)^{-b-1/2}(c_s^2uv)^{b-1/2} = x^{-b-1/2}/(c_s^2uv)$:
$$\boxed{\;I_{\rm target}(x,u,v) = 4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}\,\frac{x^{-b-1/2}}{c_s^2uv}\Big\{\sqrt{\tfrac\pi2}\,Y_{b+1/2}(x)\,A(y)\,\Theta_+ + \sqrt{\tfrac2\pi}\,J_{b+1/2}(x)\big[B(y)\Theta_+ - C(-y)\Theta_-\big]\Big\}\;}$$
$$A(y)\equiv|1-y^2|^{b/2}\mathcal P(y),\qquad B(y)\equiv|1-y^2|^{b/2}\mathcal Q_F(y),\qquad C(\tilde y)\equiv(\tilde y^2-1)^{b/2}\mathcal Q_O(\tilde y).$$
This is the object A1 builds. Closed and regularised forms of $A$, $B$, $C$ are in §3.5 and §5.2.

**[derived]** Expanding the Bessel functions for large argument, `review.tex:1871-1877`
(same as `1912.tex:639-643`): $J_{b+1/2}(x)\approx\sqrt{\tfrac{2}{\pi x}}\cos(x-\tfrac{b\pi}{2}-\tfrac\pi2)
= \sqrt{\tfrac{2}{\pi x}}\sin(x-\tfrac{b\pi}{2})$ and $Y_{b+1/2}(x)\approx\sqrt{\tfrac{2}{\pi x}}
\sin(x-\tfrac{b\pi}{2}-\tfrac\pi2) = -\sqrt{\tfrac{2}{\pi x}}\cos(x-\tfrac{b\pi}{2})$. Then
$$I_{\rm target}\to x^{-b-1}4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}\frac{1}{c_s^2uv}\Big\{-\cos\!\big(x-\tfrac{b\pi}{2}\big)A\,\Theta_+ + \frac2\pi\sin\!\big(x-\tfrac{b\pi}{2}\big)\big[B\,\Theta_+ - C\,\Theta_-\big]\Big\}.$$
Compare (4.12) `eq:Isimple2`, `review.tex:664-669`: same prefactor
$x^{-b-1}4^b\Gamma^2[b+\tfrac32]\tfrac{2b+3}{b+2}\tfrac{|1-y^2|^{b/2}}{c_s^2uv}$, same
$+\tfrac2\pi\sin(\cdot)\mathcal Q_F\Theta_+$ and $-\tfrac2\pi\sin(\cdot)\mathcal Q_O(-y)\Theta_-$,
but **$+\cos(x-\tfrac{b\pi}{2})\mathcal P\,\Theta_+$ where the derivation gives $-\cos$.** Every
constant agrees; one sign does not. Had (4.12) been derived from (4.10) *as printed*, all three
terms would be flipped relative to the correct kernel; instead only the $\cos$ term is. So
(4.12) is inconsistent with (4.10) as printed as well. Because (4.14) `eq:kernelaverage`
(`review.tex:679-684`) squares each term separately, the review's own results are unaffected,
which is presumably how both slips survived.

The 2020 paper's (3.6) `eq:kernel3` (`1912.tex:263-265`) is
$\{\tfrac\pi2\sin(x-\tfrac{\beta\pi}{2}-\tfrac\pi4)I_J + \cos(x-\tfrac{\beta\pi}{2}-\tfrac\pi4)I_Y\}$
with $I_J$, $I_Y$ of (3.7)/(3.8); with $\beta = b+\tfrac12$ the phases are
$-\cos(x-\tfrac{b\pi}{2})$ and $\sin(x-\tfrac{b\pi}{2})$, and expanding (3.1) with (3.3)/(3.4)
reproduces (3.6) including the prefactor $2^\beta\tfrac{3\sqrt2}{w\pi\alpha^3}\tfrac{1+w}{1+3w}
\Gamma^2[\beta+2](uvx)^{-\beta-1/2}$ *[derived, every constant]*. At $\beta = \tfrac12$ it gives
`1912.tex:339-341`, which is Kohri & Terada eq. (25) term for term (`kohri_terada.py:I_RD_asymptotic`).
**The 2020 paper is self-consistent; the review is not.**

*[checked]* The three forms against **E** (quadrature of (4.7)$\circ$(4.9), declared error
$\lesssim10^{-14}$), on-cut cases, distance measured in units of the large-$x$ envelope
$\mathcal N(c_s^2uv)^{-b-1/2}x^{-b-1}\sqrt{2/\pi}\,\lvert(\mathcal I^\infty_J,\mathcal I^\infty_Y)\rvert$:

| $b$ | $(u,v)$ | $y$ | $x$ | $\lvert E-\text{target}\rvert$ | $\lvert E-(4.12)\text{ as printed}\rvert$ | $\lvert E-(4.12),\ \cos\text{ sign flipped}\rvert$ |
|---|---|---|---|---|---|---|
| 0.2 | (1.3, 1.2) | −0.439 | 100 / 1600 / 6400 | 1.6e-2 / 1.5e-3 / 9.9e-6 | 1.19 / 1.44 / 1.70 | 1.7e-2 / 1.4e-3 / 2.3e-5 |
| 0.2 | (1.0, 1.3) | −0.696 | 100 / 1600 / 6400 | 2.5e-2 / 1.8e-3 / 1.9e-5 | 1.30 / 1.63 / 1.93 | 2.4e-2 / 1.7e-3 / 2.5e-5 |
| 0.2 | (2.5, 3.0) | +0.717 | 100 / 1600 / 6400 | 4.9e-4 / 2.1e-4 / 3.4e-5 | 1.33 / 1.63 / 1.93 | 4.3e-4 / 1.7e-4 / 3.0e-5 |
| 0.5 | (2.5, 3.0) | +0.417 | 100 / 1600 / 6400 | 6.2e-4 / 6.2e-5 / 1.8e-6 | 0.50 / 1.96 / 1.93 | 4.4e-3 / 5.9e-5 / 2.3e-5 |

The target and the corrected (4.12) approach **E** like $1/x$; (4.12) as printed does not
approach it at all. In the off-cut branch (no $\cos$ term) the three agree, as they must.

---

## 3. The Legendre functions: conventions, `mpmath` mapping, and how each mapping was verified

### 3.1 What the papers define, and whether it is DLMF

**[paper]** `1912.tex:718-729` and `review.tex:1982-1992` define, for $|x|<1$,
$$\mathsf P^\mu_\nu(x) = \Big(\tfrac{1+x}{1-x}\Big)^{\mu/2}\mathbf F\big(\nu+1,-\nu;1-\mu;\tfrac{1-x}{2}\big),$$
$$\mathsf Q^\mu_\nu(x) = \frac{\pi}{2\sin\mu\pi}\Big\{\cos(\mu\pi)\Big(\tfrac{1+x}{1-x}\Big)^{\mu/2}\mathbf F\big(\nu+1,-\nu;1-\mu;\tfrac{1-x}{2}\big) - \frac{\Gamma(\nu+\mu+1)}{\Gamma(\nu-\mu+1)}\Big(\tfrac{1-x}{1+x}\Big)^{\mu/2}\mathbf F\big(\nu+1,-\nu;1+\mu;\tfrac{1-x}{2}\big)\Big\},$$
$\mathbf F(a,b;c;z) = F(a,b;c;z)/\Gamma(c)$ (`1912.tex:731-734`, `review.tex:2002-2005`). For $|x|>1$
the 2020 paper defines $P^\mu_\nu$ with $\big(\tfrac{x+1}{x-1}\big)^{\mu/2}$ (`1912.tex:738-740`),
$Q^\mu_\nu$ with the $e^{\mu\pi i}$ and the $1/x^2$ hypergeometric (`1912.tex:743-747`), and
$$\mathcal Q^\mu_\nu(x)\equiv e^{-\mu\pi i}\,Q^\mu_\nu(x)/\Gamma[\mu+\nu+1]$$
(`1912.tex:749-751`). The review gives $\mathcal Q$ directly as
$\tfrac{\pi}{2\sin(\mu\pi)\Gamma(\nu+\mu+1)}\big\{(\tfrac{x+1}{x-1})^{\mu/2}\mathbf F(\ldots;1-\mu;\ldots) -
\tfrac{\Gamma(\nu+\mu+1)}{\Gamma(\nu-\mu+1)}(\tfrac{x-1}{x+1})^{\mu/2}\mathbf F(\ldots;1+\mu;\ldots)\big\}$
(`review.tex:1994-2000`) and calls it Olver's function (`review.tex:2001`).

**[derived, checked to $10^{-30}$ at `mp.dps = 30`]** These are DLMF 14.3.1, 14.3.2 (Ferrers),
14.3.6, 14.3.7 and 14.3.10 (Olver's $\boldsymbol{\mathsf Q}^\mu_\nu = e^{-\mu\pi i}Q^\mu_\nu/\Gamma(\nu+\mu+1)$)
**with no change of normalisation**: the $\tfrac{\pi}{2}$, $\cos\mu\pi$ and $\Gamma$ factors are
DLMF's. The review's hypergeometric $\mathcal Q$ and the 2020 paper's $e^{-\mu\pi i}Q/\Gamma$ agree
with each other to $10^{-28}$ at $x = 1.05, 1.8, 150$ for $b\in\{0.2, 0.5, 0.37, -0.5+10^{-9}\}$
and both $\nu$; the review's $\mathcal Q^{-b}_\nu = \mathcal Q^{b}_\nu$ (`review.tex:2052`) is DLMF
14.9.14 and holds to $10^{-28}$. The papers' $\mathsf Q$ **is** DLMF's Ferrers function of the
second kind.

### 3.2 The mapping to `mpmath`

`mpmath.legenp(n, m, z, type)` and `legenq(n, m, z, type)` take **degree first, order second**.
With $\mu = -b$:

| Function | `mpmath` call | Conversion factor | Verified by |
|---|---|---|---|
| $\mathsf P^{-b}_{b}(y)$ | `legenp(b, -b, y, type=2)` | none | closed form (§3.5) to $10^{-31}$; DLMF 14.5.18 |
| $\mathsf P^{-b}_{b+2}(y)$ | `legenp(b+2, -b, y, type=2)` | none | closed form (§3.5) to $10^{-31}$ |
| $\mathsf Q^{-b}_{b}(y)$ | `legenq(b, -b, y, type=2)` | none | paper's hypergeometric to $10^{-30}$; Wronskian DLMF 14.2.4 $\mathcal W\{\mathsf P^\mu_\nu,\mathsf Q^\mu_\nu\} = \tfrac{\Gamma(\nu+\mu+1)}{\Gamma(\nu-\mu+1)(1-x^2)}$ to $10^{-31}$; parity $\mathsf Q(-y) = -\mathsf Q(y)$ to $10^{-31}$; closed forms at $b = 0, \pm\tfrac12$ (§3.4) |
| $\mathsf Q^{-b}_{b+2}(y)$ | `legenq(b+2, -b, y, type=2)` | none | same three |
| $\mathcal Q^{-b}_{b}(\tilde y)$ | `legenq(b, -b, ỹ, type=3)` | **multiply by $e^{+ib\pi}/\Gamma(1)$, i.e. $e^{-\mu\pi i}/\Gamma(\nu+\mu+1)$, and take the real part** | review's and 2020's hypergeometrics to $10^{-30}$; Wronskian DLMF 14.2.8 $\mathcal W\{P^{-\mu}_\nu,\boldsymbol{\mathsf Q}^\mu_\nu\} = -\tfrac{1}{\Gamma(\nu+\mu+1)(x^2-1)}$ to $10^{-31}$; closed forms at $b = 0, \tfrac12$ |
| $\mathcal Q^{-b}_{b+2}(\tilde y)$ | `legenq(b+2, -b, ỹ, type=3)` | **multiply by $e^{+ib\pi}/\Gamma(3) = e^{ib\pi}/2$, real part** | same three |

*[checked]* The raw `type=3` value is **complex** for non-integer $b$: its imaginary part is
$|\sin b\pi|$ of its modulus (0.59 at $b = 0.2$, 1.00 at $b = \pm\tfrac12$). After the phase
$e^{-\mu\pi i}$ the residual imaginary part is $\lesssim10^{-31}$. Do not take `abs()`; the sign
matters and is carried by the real part.

For the DLMF Wronskian 14.2.8 the pairing is $P^{-\mu}_\nu$ (`legenp(nu, -mu, x, type=3)`) with
$\boldsymbol{\mathsf Q}^\mu_\nu$; the $1/\Gamma(\nu+\mu+1)$ on the right is real — at
$\nu+\mu = 2$ it is $\tfrac12$. I first misremembered this as $-1/(x^2-1)$ and got $-\tfrac12$ at
$\nu = b+2$; the DLMF page settled it. A1 should use the equation as DLMF states it.

**Branch cut of `type=3`.** The off-cut branch is evaluated at $\tilde y = -y$ with $y<-1$, i.e.
$\tilde y>1$ on the real axis to the right of the cut $[-1,1]$. Nothing is evaluated on or
across the cut, so the cut's placement is irrelevant **provided A1 passes $-y$, never $y$**: at
real argument $<-1$ the type-3 functions carry the cut's phase and are not the papers'
$\mathcal Q$. Also (§3.4 below) `legenq(..., type=2)` returns `nan` and `legenp(..., type=2)`
returns `-inf` at $y = -1$ exactly *[checked]*; the point itself needs §5.2.

### 3.3 The reduced forms of the two Ferrers-relation typos in the sources

**[paper, checked]** `1912.tex:758-760` prints $\tfrac{2\sin\mu\pi}{\pi}\mathsf Q^\mu_\nu =
\tfrac{\mathsf P^\mu_\nu}{\Gamma[\nu+\mu+1]} - \tfrac{\mathsf P^{-\mu}_\nu}{\Gamma[\nu-\mu+1]}$. At
$b = 0.2$, $\nu = 2.2$, $y = 0.3$ the left side is $0.17600$ and the right side $0.06206$; DLMF
14.9.2, $\tfrac{2\sin\mu\pi}{\pi\Gamma(\nu-\mu+1)}\mathsf Q^{-\mu}_\nu = \tfrac{\mathsf P^\mu_\nu}{\Gamma(\nu+\mu+1)} -
\tfrac{\cos\mu\pi\,\mathsf P^{-\mu}_\nu}{\Gamma(\nu-\mu+1)}$, gives $0.17600$. The paper's line is a
garbled 14.9.2. It is in "useful relations" and is not used in the kernel. Likewise
`review.tex:2022-2026` labels the $\mathsf Q^\mu_\nu\sim\tfrac12\cos(\mu\pi)\Gamma(\mu)(\tfrac{2}{1-x})^{\mu/2}$
asymptotic "$\mu<0$"; DLMF 14.8.4 has $\Re\mu>0$, and the review's own application at
`review.tex:2065` uses it with $\mu = -b>0$ for $b<0$, correctly. Labelling slips only.

### 3.4 Closed forms at $b\in\{0,\pm\tfrac12\}$ and the `mpmath` fragilities

**[paper, checked to $10^{-31}$]** Every special case in `1912.tex:804-822` agrees with the
`mpmath` calls above (with the conversion of §3.2 for $\mathcal Q$): $\mathsf P^0_0, \mathsf P^0_2,
\mathsf Q^0_0, \mathsf Q^0_2, \mathcal Q^0_0, \mathcal Q^0_2$; $\mathsf P^{1/2}_{-1/2}, \mathsf P^{1/2}_{3/2},
\mathsf Q^{1/2}_{-1/2} = 0, \mathsf Q^{1/2}_{3/2}$; $\mathsf P^{-1/2}_{1/2}, \mathsf P^{-1/2}_{5/2},
\mathsf Q^{-1/2}_{1/2}, \mathsf Q^{-1/2}_{5/2}, \mathcal Q^{-1/2}_{1/2}, \mathcal Q^{-1/2}_{5/2}$. These
are DLMF 14.5.11–14.5.14, 14.5.17, 14.5.18 specialised.

*[checked]* Two `mpmath 1.3.0` fragilities A1 will meet:

- At $b = \tfrac12$, `legenq(2.5, -0.5, y, type=2)` raises `hypsum() failed to converge` at
  $y = \pm0.5$ (fine at $-0.49$, $-0.9$), and so does the paper's hypergeometric written out,
  because `hyp2f1(3.5, -2.5, 0.5, 0.75)` fails in `mpmath` (it is $2.0$; `scipy.special.hyp2f1`
  returns it, and a $10^{-15}$ perturbation of $c$ in `mpmath` returns $1.99999999999999$). A
  degenerate connection case ($a-b\in\mathbb Z$, $c$ half-integer). $b = \pm\tfrac12$ and $b = 0$
  are closed-form cases anyway; A1 should special-case them and never route them through
  `legenq`.
- At $b = 0$ (integer order) `legenq(type=2)` works and matches $\tfrac12\ln\tfrac{1+y}{1-y}$ to
  $10^{-31}$, but the $\tfrac{\pi}{2\sin\mu\pi}$ **definition** is $0/0$ there; if A1 codes the
  hypergeometric form directly, $b = 0$ must go through KT's closed form (§7).

### 3.5 $\mu+\nu\in\{0,2\}$: what has a closed form

**[derived, checked to $10^{-31}$ for $b\in\{0.2, 0.5, -0.3, 0.9\}$]** The two $\mathsf P$'s are
elementary. With $\mathbf F(b+1,-b;1+b;z) = (1-z)^b/\Gamma(1+b)$ (Euler's $_2F_1(a,b;a;z)$), and an
Euler transformation that reduces the second to a quadratic,
$$\mathsf P^{-b}_b(y) = \frac{(1-y^2)^{b/2}}{2^b\Gamma(1+b)},\qquad
\mathsf P^{-b}_{b+2}(y) = \frac{(1-y^2)^{b/2}}{2^b\Gamma(1+b)}\Big[1 - \frac{2b+3}{2(b+1)}(1-y^2)\Big],$$
(the first is DLMF 14.5.18), hence
$$\boxed{\;A(y) = |1-y^2|^{b/2}\mathcal P(y) = \frac{(2b+3)}{2^b\Gamma(1+b)(b+1)}\,(1-y^2)^{b}\Big[1 - \frac{b+2}{2(b+1)}(1-y^2)\Big]\;}$$
which is $3y^2$ at $b = 0$ (`1912.tex:329`) and reproduces `review.tex:2058`'s $y\to-1^+$ limit
exactly. **So the $\cos$ coefficient needs no special function at all.** The two $\mathsf Q$'s do not
reduce: each is one genuine $_2F_1$ (§5.2 gives the form to use). $\mathcal Q$ likewise.

---

## 4. The two inter-paper discrepancies, and the test that discriminates

### 4.1 Sign, order and prefactor of the exact kernel

**[paper]** 1912 (3.1) `eq:kernel2`, `1912.tex:212-214`:
$I = 4^\beta\tfrac{3\pi}{2\alpha^3}\tfrac{1+w}{1+3w}\Gamma^2[\beta+2](uvwx)^{-\beta}\{Y_\beta(x)\mathcal I^x_J - J_\beta(x)\mathcal I^x_Y\}$,
defined through `eq:kernel`, `1912.tex:191-193`: $I\equiv2\big(\tfrac{3+3w}{5+3w}\big)^2\int_0^xd\tilde x\,G f$.
Review (4.10), `review.tex:649-651`: $I = \pi4^b\Gamma^2[b+\tfrac32]\tfrac{2b+3}{b+2}(c_s^2uvx)^{-b-1/2}
(J_{b+1/2}\mathcal I_Y - Y_{b+1/2}\mathcal I_J)$, defined through `review.tex:535`: $I\equiv\int G f$.

**[derived]** *Prefactor.* $4^\beta = 2^{2b+1}$; $\alpha^3 = (2b+3)^3/8$; $\tfrac{1+w}{1+3w} = \tfrac{2+b}{3}$;
so the 2020 prefactor is $\pi2^{2b+3}\tfrac{2+b}{(2b+3)^3}\Gamma^2[b+\tfrac52](c_s^2uvx)^{-b-1/2}$, and
$2c^2\mathcal N = 2\tfrac{(2+b)^2}{(3+2b)^2}\pi4^b\Gamma^2[b+\tfrac32]\tfrac{2b+3}{b+2} =
\pi2^{2b+3}\tfrac{2+b}{(2b+3)^3}\Gamma^2[b+\tfrac52]$, using $\Gamma^2[b+\tfrac32] = 4\Gamma^2[b+\tfrac52]/(2b+3)^2$.
**Identical.** The $\Gamma$-argument shift and the $\alpha^3$ are one rewriting; the powers of
$(c_s^2uvx)$ are the same; the only real difference is the definitional $2c^2$, and the two
$I$'s are **not** the same object:
$$\boxed{\;I_{2020}(x,u,v) = 2c^2\,I_{\rm rev}(x,u,v),\qquad c = \frac{2+b}{3+2b} = \frac{3(1+w)}{5+3w}\;}$$
(It is also the 2020 `eq:pgamma` "2" (`1912.tex:177-179`) against the review `eq:Phgaussian` "8"
(`review.tex:526-528`) with
$\mathcal P_\Phi = c^2\mathcal P_{\mathcal R}$: $2\cdot(2c^2)^2 = 8c^4$.) This is the same $c$ as spec
03 §0.1's and KT's $\tfrac{3(1+w)}{5+3w}$.

*Order.* §2.2: the review's own $G$ gives $Y\mathcal I_J - J\mathcal I_Y$. The 2020 paper's
`eq:green2` (`1912.tex:195-197`) is the same $G$ and (3.1) keeps the order. **The review's
(4.10) is $-1\times$ the kernel its (4.7) and (4.9) define. Typo, not convention**, because (4.10)
is presented as derived from them: "Replacing \eqref{eq:fsimple} and \eqref{eq:x} into
Eq.~\eqref{eq:kernelApp} we find that the kernel can be written as" (`review.tex:648`).

*[checked]* **E** against (4.10) with the finite-$x$ $\mathcal I_{J,Y}(x)$ by quadrature, at
$x = 30$ (both quadratures' declared error $\le2\times10^{-14}$):

| $b$ | $(u,v)$ | **E** | (4.10) as printed | (4.10) with $Y\mathcal I_J - J\mathcal I_Y$ |
|---|---|---|---|---|
| 0.2 | (0.909, 1.091) | +1.1724885558e-01 | −1.1724885558e-01 | +1.1724885558e-01 |
| 0.2 | (0.3, 1.6) | +1.2003036441e-01 | −1.2003036441e-01 | +1.2003036441e-01 |
| 0.5 | (0.7, 1.1) | +1.0447050704e-02 | −1.0447050704e-02 | +1.0447050704e-02 |
| −0.3 | (0.3, 1.6) | −7.2141335647e-01 | +7.2141335647e-01 | −7.2141335647e-01 |

(nine cases run; relative agreement of the corrected order with **E** is $\le2\times10^{-15}$ on
all nine.)

**Discriminating tests for A1** (any one is decisive; run all three):

1. **Small $x$.** $G>0$ and $f>0$ near $\tilde x = 0$, so $I_{\rm rev}(x\ll1) = +\tfrac{x^2}{2(2+b)}$
   *[derived: $G_x\to\tfrac{\tilde x}{1+2b}[1-(\tilde x/x)^{1+2b}]$, $\int_0^x = \tfrac{x^2}{2(3+2b)}$,
   $f(0) = \tfrac{3+2b}{2+b}$; checked: E$/(x^2/(2(2+b)))$ = 0.99992, 0.99994, 0.99995 at
   $x = 0.03$ for $b = 0, 0.2, 0.5$]*. (4.10) as printed is negative there.
2. **Against KT at $b = 0$.** $I_{\rm rev} = \tfrac98 I_{\rm KT}$ (§7), sign included; **E** matches
   $\tfrac98\times$`I_RD` to $\le4.5\times10^{-15}$ at six $(u,v,x)$ including the `T-first`
   shape *[checked]*.
3. **Direct.** Quadrature of $\int_0^x G_x f$ against $\pm$(4.10) with $\mathcal I(x)$ by quadrature,
   as in the table.

### 4.2 The asymmetric factor of 2

**[paper]** Off-cut brackets: `1912.tex:254` $[\mathcal Q^{\ldots}_{\beta-1/2} + 3(1+w)\mathcal Q^{\ldots}_{\beta+3/2}]$
against on-cut `1912.tex:248, 253` $\tfrac{3(1+w)}{2}$; identically `1912.tex:274` and
`review.tex:668, 683, 1563` $2\tfrac{b+2}{b+1}$ against $\tfrac{b+2}{b+1}$.

**[derived]** From the appendix formula the papers quote (`1912.tex:684-704`, `review.tex:1947-1971`):
for $c>a+b$ the integral carries $\Gamma[\nu-\rho+1]$ with $\nu-\rho\in\{0,2\}$ for the two
source terms. $\Gamma[1] = 1$, $\Gamma[3] = 2$. The on-cut formula has no such factor. **Correct
in both papers.** The reading-level corroboration: at $b = 0$, $\mathcal Q^0_0(\tilde y) +
4\mathcal Q^0_2(\tilde y) = \tfrac{3\tilde y^2}{2}\ln\tfrac{\tilde y+1}{\tilde y-1} - 3\tilde y$
(`1912.tex:810`), which with $\tilde y = -y$ is $-\tfrac{3y}{2}\big(y\ln\lvert\tfrac{1+y}{1-y}\rvert - 2\big)$,
the same function as the on-cut $\mathsf Q^0_0 + 2\mathsf Q^0_2$ continued through the resonance —
which is what `1912.tex:333-336` displays and what KT eq. (25)'s $\ln\lvert\cdot\rvert$ requires.
With a 2 replaced by 1 it would be $\tfrac14(3\tilde y^2+1)\ln(\cdot) - \tfrac32\tilde y$, which
matches nothing.

*[checked]* Target against **E** in the off-cut branch at $b = 0.2$, distance in envelope units:

| $(u,v)$ | $x$ | with $2\tfrac{b+2}{b+1}$ | with $\tfrac{b+2}{b+1}$ |
|---|---|---|---|
| (0.909, 1.091) | 6400 / 25600 | 1.4e-4 / 1.7e-5 | 2.2e-2 / 7.7e-2 |
| (1.0, 1.0) | 6400 / 25600 | 3.2e-5 / 1.8e-5 | 2.2e-2 / 7.7e-2 |
| (0.6, 0.9) | 6400 / 25600 | 6.9e-5 / 1.1e-4 | 2.7e-3 / 1.0e-2 |

**Discriminating test for A1:** the same, at $x\gtrsim10^4$ where the $1/x$ tail is below the
$O(1)$ change; or quadrature of the single integral $\int_0^X\tilde x^{1/2-b}Y_{b+1/2}(\tilde x)
J_{b+5/2}(c_sv\tilde x)J_{b+5/2}(c_su\tilde x)$ for $c_s(u+v)<1$ against
$\tfrac1\pi\sqrt{\tfrac2\pi}(c_s^2uv)^{b-1/2}(y^2-1)^{b/2}\cdot\Gamma[3]\cdot\mathcal Q^{-b}_{b+2}(-y)$.

---

## 5. The resonance

### 5.1 Divergence structure of the coefficients as $y\to-1^+$

**[paper]** `1912.tex:280-293`: $\mu\equiv-\beta+\tfrac12$, $\nu\in\{\beta-\tfrac12,\beta+\tfrac32\}$,
"$\mu+\nu = 0,2$ and $-1<\mu\le\tfrac12$"; $\mathsf Q^\mu_\nu\propto(1+y)^{-|\mu|/2}$ ($\mu\ne0$),
$\propto\ln(1+y)$ ($\mu = 0$); and, with the $Z\propto\sqrt{1-y^2}$ factor,
$I\propto(1+y)^{-\frac12(\mu+|\mu|)}$ ($\mu\ne0$), $\propto\ln(1+y)$ ($\mu = 0$). The displays at
`1912.tex:291-292` say "$y\to1$"; the text at `1912.tex:280` says the resonant point "corresponds to
$y = -1$" and at `1912.tex:284` "$y\to-1^+$". `review.tex:2056-2068` gives the $y\to-1^+$ limits of
$\mathcal P$ and $\mathcal Q_F$ explicitly, and `review.tex:2069-2077` those of $\mathcal Q_O$ as
"$y\to1^+$" with $(\tfrac{2}{1-y})^{b/2}$, which for $y>1$ must be read $(\tfrac{2}{y-1})^{b/2}$.

**[derived]** With $\mu = -b$: $u+v = c_s^{-1}$ gives $y = -1$ from (4.13), so **$y\to-1^+$ is
meant**; "$y\to1$" at `1912.tex:291-292` is a typo. (The other edge, $y\to+1^-$, is $c_s|u-v|\to1$,
the IR limit of `review.tex:810-820`, reachable only at $c_s = 1$ on the physical triangle.) Since
$\mathsf P^\mu_\nu(-x) = \mathsf P^\mu_\nu(x)$ and $\mathsf Q^\mu_\nu(-x) = -\mathsf Q^\mu_\nu(x)$ for
$\mu+\nu\in\{0,2\}$ (`1912.tex:772-776`, from the general reflection formulas at
`1912.tex:761-770`; *[checked to $10^{-31}$ for both $\nu$ at $b\in\{0.2, 0.5, -0.45\}$]*), the
behaviour at $-1^+$ is the behaviour at $1^-$ with $x\to-y$. Then DLMF 14.8.1, 14.8.4, 14.8.6 give, for the products that enter the
target:

| $b$ | $A = |1-y^2|^{b/2}\mathcal P$ ($\cos$ coefficient) | $B = |1-y^2|^{b/2}\mathcal Q_F$ ($\sin$ coefficient) |
|---|---|---|
| $-\tfrac12\le b<0$ | $\propto(1+y)^{b} = (1+y)^{-|b|}$, diverges | $\propto(1+y)^{-|b|}$, diverges |
| $b = 0$ | $\to3$ (finite, discontinuous across the resonance: $\Theta$ in KT (25)) | $\propto\ln(1+y)$, diverges |
| $0<b<1$ | $\to0$ as $(1+y)^b$ | finite, $\to-\tfrac{2^b\Gamma(b)(3+2b)(1+b+b^2)}{(1+b)\Gamma(2b+3)}$ |

**This confirms README (i3)**, and adds that the divergence is a property of the $x\to\infty$
*coefficients*: the exact finite-$x$ kernel is an integral of bounded functions over a finite
range and is finite at the resonance for every $b$ (KT §2.1/§6.1 is the $b = 0$ instance; **E**
at $b = 0.2$ evaluated within $|c_s(u+v)-1| = 0.002$ of it is unremarkable, §5.4). The target
object therefore **cannot score the resonance for $b\le0$**, and for $b>0$ its accuracy there
degrades (§5.4). The review's `review.tex:791-793` and `800-803` are the squares of the last
column and I reproduced both from the corrected (4.12) *[derived]*.

### 5.2 The regularised combination for $b>0$, finite and evaluable at $y=-1$

**[derived]** The divergence in $\mathsf Q^{-b}_\nu(y) = \tfrac{\pi}{2\sin(-b\pi)}\big[\cos(b\pi)\mathsf P^{-b}_\nu -
\tfrac{\Gamma(\nu-b+1)}{\Gamma(\nu+b+1)}\mathsf P^{b}_\nu\big]$ is entirely in $\mathsf P^b_\nu(y) =
(\tfrac{1+y}{1-y})^{b/2}\mathbf F(\nu+1,-\nu;1-b;\tfrac{1-y}{2})$, whose $\mathbf F$ diverges as
$(\tfrac{1+y}{2})^{-b}$ at $y\to-1$ (DLMF 15.8.4, $c-a-b = -b$). Applying DLMF 15.8.4 to that
$\mathbf F$ and recognising the regular piece as $\mathsf P^{-b}_\nu(-y) = \mathsf P^{-b}_\nu(y)$, the
reflection formulas collapse the coefficients to
$$\boxed{\;(1-y^2)^{b/2}\mathsf Q^{-b}_\nu(y) = \frac\pi2\cot(b\pi)\,(1-y^2)^{b/2}\mathsf P^{-b}_\nu(y) - 2^{b-1}\Gamma(b)\frac{\Gamma(\nu-b+1)}{\Gamma(\nu+b+1)}\;{}_2F_1\Big(-\nu-b,\,1+\nu-b;\,1-b;\,\frac{1+y}{2}\Big)\;}$$
for $\nu\in\{b, b+2\}$, $b\notin\mathbb Z$. The first term is elementary (§3.5) and vanishes at
$y = -1$ for $b>0$; the second is a $_2F_1$ at argument $\tfrac{1+y}{2}\in[0,1)$, analytic on the
whole on-cut range and equal to $1$ at $y = -1$. **Nothing diverges, nothing cancels.** For the
off-cut branch, directly from the review's hypergeometric $\mathcal Q$ with $\mu = -b$:
$$\boxed{\;(\tilde y^2-1)^{b/2}\mathcal Q^{-b}_\nu(\tilde y) = -\frac{\pi}{2\sin(b\pi)\Gamma(\nu-b+1)}\Big[(\tilde y-1)^b\,\mathbf F\big(\nu+1,-\nu;1+b;\tfrac{1-\tilde y}{2}\big) - \frac{\Gamma(\nu-b+1)}{\Gamma(\nu+b+1)}(\tilde y+1)^b\,\mathbf F\big(\nu+1,-\nu;1-b;\tfrac{1-\tilde y}{2}\big)\Big]\;}$$
with both $\mathbf F$ at non-positive argument, smooth for all $\tilde y\ge1$; the first $\mathbf F$ is
elementary ($\big(\tfrac{1+\tilde y}{2}\big)^b/\Gamma(1+b)$ for $\nu = b$). Hence
$$B(y) = R_b(y) + \rho R_{b+2}(y),\qquad C(\tilde y) = S_b(\tilde y) + 2\rho S_{b+2}(\tilde y),$$
with $R_\nu$, $S_\nu$ the two boxed right-hand sides. *[checked to $10^{-30}$ against the naive
products for $b\in\{0.2, 0.5, 0.8, -0.3\}$, $y\in\{-0.999,-0.5,0.4,0.99\}$, $\tilde y\in\{1.001, 1.5, 40\}$]*

At the resonance, for $b>0$:
$$B(-1) = -\frac{2^b\Gamma(b)(3+2b)(1+b+b^2)}{(1+b)\Gamma(2b+3)},\qquad C(1) = +\frac{2^b\Gamma(b)(3+2b)(1+b+b^2)}{(1+b)\Gamma(2b+3)},$$
so the $\sin$ coefficient $B\Theta_+ - C\Theta_-$ is **continuous across the resonance** and the
$\cos$ coefficient $A\Theta_+$ vanishes there: the target is continuous in $(u,v)$ at
$c_s(u+v) = 1$ for $b>0$ (`review.tex:808` says as much). Both values reproduce
`review.tex:2062-2077` *[derived, and checked: at $b = 0.2$ both give $\mp6.2147807897686$]*.

At $b = 0$ the two terms of $R_\nu$ are individually $\infty$ ($\cot b\pi$, $\Gamma(b)$) and
the form is unusable; $b = 0$ goes through KT's $\mathrm{Cin}$-regularised eq. (22)
(`kohri_terada.py:I_RD`), which is the $b\to0$ limit of the same structure. For $b<0$ the
formula holds but the first term genuinely diverges, as it must.

### 5.3 Floating point approaching the resonance

*[checked, $b = 0.2$, double precision via `scipy.special.hyp2f1`, against `mpmath` at 30 digits]*
Three evaluations of $B(y)$ at $1+y = \delta_y$:

| $\delta_y = 1+y$ | naive: Ferrers definition, then $\times(1-y^2)^{b/2}$ | §5.2 form, $1+y$ recomputed from $y$ | §5.2 form, $1+y$ carried |
|---|---|---|---|
| $10^{-2}$ | 3.9e-15 | 7.7e-16 | 0 |
| $10^{-6}$ | 2.2e-11 | 4.2e-13 | 3.1e-16 |
| $10^{-8}$ | 2.1e-9 | 2.8e-11 | 2.9e-16 |
| $10^{-12}$ | 2.2e-5 | 1.9e-8 | 2.9e-16 |
| $0$ | `nan` | 1.4e-16 | 1.4e-16 |

Two lessons. The naive product has no cancellation, but it is a $0\times\infty$ whose factors
are each computed to relative $\epsilon/\delta_y$-ish accuracy, so it loses a digit per decade
below $\delta_y\sim10^{-4}$ and is `nan` at the point. The §5.2 form is exact to rounding **if
$1+y$ is formed from $c_s(u+v)-1$ directly**:
$$1+y = \frac{(c_s(u+v)-1)(c_s(u+v)+1)}{2c_s^2uv},\qquad 1-y = \frac{(1-c_s(u-v))(1+c_s(u-v))}{2c_s^2uv},$$
never as `1 + y` after computing $y$, which costs $\epsilon/\delta_y$. This is the same
discipline `kohri_terada.py` applies to its $c = 1-\tfrac{v+u}{\sqrt3}$. In `mpmath`, `legenq`
type 2 evaluates in $<3$ ms down to $1+y = 10^{-16}$ and returns `nan` at $-1$ *[checked]*.

### 5.4 Non-uniformity of the $O(1/x)$ near the resonance, and the fixtures

*[checked]* Target against **E**, $b = 0.2$, $\delta\equiv c_s(u+v)-1$, envelope units:

| $\delta$ | $x = 400$ | $x = 1600$ | $x = 6400$ |
|---|---|---|---|
| +0.020 | 8.2e-3 | 2.7e-3 | 5.7e-4 |
| −0.020 | 1.6e-2 | 8.0e-3 | 1.1e-3 |
| +0.002 | 9.1e-2 | 3.0e-2 | 6.2e-3 |
| −0.002 | 1.3e-1 | 3.3e-2 | 7.8e-3 |

The error scales roughly as $1/(|\delta|x)$: the finite-upper-limit correction is controlled by
the slowest beat frequency $c_s(u+v)-1$ in the integrand, not by $x$ alone. **The target's
$O(1/x)$ is not uniform in $(u,v)$.** KT §6 says the same of eq. (25). **E** itself is finite and
smooth through $\delta = 0$.

**Fixtures** (`test_quadsource_integral.py:323-327`, `B_VALUES = (0.0, 0.2)`), *[checked]*:

| Shape | $(u,v) = (q/k, r/k)$ | $b$ | $c_s$ | $c_s(u+v)-1$ | $c_s|u-v|$ | $y$ | Branch |
|---|---|---|---|---|---|---|---|
| `together` | (0.909, 1.091) | 0 | 0.5774 | +0.155 | 0.105 | −0.496 | on-cut |
| `together` | (0.909, 1.091) | 0.2 | 0.4714 | −0.057 | 0.086 | −1.252 | off-cut |
| `T-first` | (10, 12) | 0 | 0.5774 | +11.70 | **1.155** | **+1.0042** | **outside both G–N cases** |
| `T-first` | (10, 12) | 0.2 | 0.4714 | +9.37 | 0.943 | +0.9979 | on-cut |
| `q-smooth` | (0.01, 1.1) | 0 | 0.5774 | −0.359 | 0.629 | −81.4 | off-cut |
| `q-smooth` | (0.01, 1.1) | 0.2 | 0.4714 | −0.477 | 0.514 | −149.5 | off-cut |

No fixture is at or near the resonance; the closest is `together` at $b = 0.2$ with
$\delta = -0.057$, where the table above suggests the target is within $\sim10^{-3}$ of the exact
kernel by $x\sim10^3$. The `T-first` shape at $b = 0$ lands at $y>1$, §6.2.

---

## 6. Domain, branches, and the limits A1 can assert against

### 6.1 Validity range in $b$

**[paper]** 2020: $1\ge w>0$, i.e. $1\le\alpha<\tfrac52$, $0\le\beta<\tfrac32$ (`1912.tex:101-104`),
excluding $w = 0$ because $\Phi$ is then not a Bessel function and $w<0$ because the Bessel
functions become modified. Review: $-1<b<\infty$ for the background (`review.tex:276`); the
kernel needs $c_s\ne0$ (`review.tex:624, 1573`); explicit examples at $b = -\tfrac12, 0, \tfrac12, 1, 2$
(`review.tex:1575`).

**[derived]** $-\tfrac12\le b<1$ is the 2020 range; the review's formulas hold for $-1<b$ and
$c_s\ne0$ as far as convergence of (4.11) at $x\to\infty$ goes (integrand $\sim\tilde x^{-1-b}$
oscillatory, conditionally convergent for $b>-1$). What breaks: at $b = -\tfrac12$ nothing
breaks, but $\mathsf Q^{1/2}_{-1/2}\equiv0$ and the resonance coincides with the triangle edge
$u+v = 1$ where the projector kills the integrand (`1912.tex:302`, `review.tex:784-788`), so it is
the only value at which the divergence is harmless; **in this repository $c_s^2 = w =
\tfrac{1-b}{3(1+b)}$** (`wPerturbations`, spec 01), so $b\to1$ sends $c_s\to0$ and every Bessel
argument $c_su\tilde x\to0$: the $x\to\infty$ formulas become meaningless before $b$ reaches 1.
$b = 0$ is the one integer-$\mu$ point inside the range (§3.4). Both fixture values are interior.

### 6.2 Where $y$ lives, and the `T-first` shape

**[derived]** From (4.13): $y\ge-1\iff c_s(u+v)\ge1$ and $y\le1\iff c_s|u-v|\le1$. On the
physical triangle $|u-v|\le1\le u+v$ with $c_s<1$: $c_s|u-v|<1$ always, so **$y\le1$ always**, and
$y<-1$ exactly when $u+v<1/c_s$, which is inside the triangle for every $c_s<1$. So both the
on-cut ($|y|<1$) and off-cut ($y<-1$) branches occur physically (`review.tex:1977` says so), and
$y>1$ does not.

`T-first` has $|u-v| = 2>1$ and is not a closable triangle (KT §8 item 4). At $b = 0$ it has
$c_s|u-v| = 1.155>1$ and $y = 1.0042$: **outside both Gervois–Navelet cases the papers quote**
($|a-b|<c<a+b$ and $c>a+b$; the third, $c<|a-b|$, is not given). The Ferrers functions are
undefined there and a literal transcription will fail or return complex garbage. **The Domènech
form as written is not defined on `T-first` at $b = 0$**, exactly where KT eq. (25) needed its
$\cos$ term dropped. At $b = 0.2$, `T-first` has $c_s|u-v| = 0.943<1$, $y = 0.998$, and is on-cut
and well defined (the $(1-y)^{-b/2}$ growth of $\mathsf Q$ is cancelled by $|1-y^2|^{b/2}$ and is
numerically harmless: $(2/0.002)^{0.1}\approx2$).

*[checked, inferred beyond the papers — §10 item 2]* For $y>1$ at $b = 0.2$, the hypothesis
"$\mathcal I^\infty_J = 0$ and $\mathcal I^\infty_Y = -\big[\mathcal I^\infty_Y$'s off-cut form evaluated at
$+y$ instead of $-y\big]$" matches **E** as $1/x$ ($7.8\times10^{-3}$, $1.1\times10^{-3}$,
$2.4\times10^{-4}$ of envelope at $x = 400, 1600, 6400$ for $(u,v) = (0.5, 3.0)$; similar for
$(10, 12.5)$ and $(0.2, 2.6)$), while the same with $+$ misses by $O(1)$. This is a numerical
inference about a region no physical configuration reaches; A1 needs it only if it wants the
oracle on `T-first` at $b = 0$, and at $b = 0$ KT eq. (22) already covers that shape exactly.

### 6.3 Small $x$

**[derived, checked §4.1]** Exact kernel: $I_{\rm rev}\to\tfrac{x^2}{2(2+b)}$, so
$I_{2020} = I_{\rm KT}\to\tfrac{(2+b)}{(3+2b)^2}x^2$, which is KT's $\tfrac{2x^2}{9}$ at $b = 0$
(their erratum 2, KT §2 item 2). The commented-out `1912.tex:236` would give $\tfrac{2+b}{2b+3}x^2$
for $I_{2020}$, too large by the factor $3+2b$; it is commented out and never printed. **The target object
has no small-$x$ limit**: $Y_{b+1/2}(x)\sim x^{-b-1/2}$ makes it $\sim x^{-2b-1}$ as $x\to0$. Regression
against the small-$x$ limit is a test of the *exact* kernel (or of A1's quadrature harness), not
of the target.

### 6.4 Large $x$

**[derived]** Envelope $\propto x^{-b-1}$, phase $x-\tfrac{b\pi}{2}$, coefficients §2.4. The
target approaches **E** like $1/x$ with the caveats: near the resonance like $1/(|\delta|x)$
(§5.4), and when one Bessel argument is small like $1/(c_s\min(u,v)\,x)$. *[checked]* On the
`q-smooth` $(u,v)$ at $b = 0.2$: 1.0, 0.48, 0.20, 0.055 of envelope at $x = 50, 200, 800, 3200$
($c_sux = 0.24$ to $15$), against $0.23$, $0.05$, $0.009$, $0.0019$ on `together`. At the fixtures'
$x_{\rm resp}\sim1.5\times10^3$ the target is a $\sim5\%$ oracle on `q-smooth` and a $\sim0.2\%$
one on `together`. README (j)'s point stands — at $x\sim10^7$ it is sharp everywhere — but the
finite-$x$ (4.11) by quadrature is the only decisive normalisation instrument at fixture $x$.

**Limits A1 can assert against**, with what each tests:

| Assertion | Tests | Value |
|---|---|---|
| $I_{\rm rev}(x\ll1) = \tfrac{x^2}{2(2+b)}$ | the harness's $G$, $f$ and sign | ratio 0.9999 at $x = 0.03$ |
| $I_{\rm rev}(b=0) = \tfrac98 I_{\rm KT}$, any $x$ | $G$, $f$, normalisation | $\le4.5\times10^{-15}$ *[checked]* |
| $I_{\rm target}(b\to0) = \tfrac98\,$`I_RD_asymptotic` | Legendre limits, both branches, $\cos$ sign | rel. diff. $26b$: $2.6\times10^{-4}$ at $b = 10^{-5}$ *[checked]* |
| $\lvert I_{\rm target}-I_{\rm rev}\rvert/\text{env}\propto1/x$ away from resonance and small arguments | the whole target | table §2.4 |
| $B(-1)$, $C(1)$ at $b = 0.2$ | the regularised forms | $\mp6.2147807897686$ |
| $A(y) = 3y^2$, $B(y) = \tfrac{3y}{2}\big(y\ln\lvert\tfrac{1+y}{1-y}\rvert-2\big)$ at $b\to0$ | $\mathcal P$, $\mathcal Q_F$ | `1912.tex:329, 333-336` |
| Wronskians DLMF 14.2.4 and 14.2.8 | the six calls | §3.2 |

The third row deserves a note: at $b = 0$ the order-$\tfrac12$ Bessel asymptotics are exact,
so **the target object and the corrected (4.12) coincide at $b = 0$**, and both equal
$\tfrac98\times$KT eq. (25). The "target versus doubly-asymptotic" distinction README (i) draws is
real only for $b\ne0$, where it is the $(4\nu^2-1)/(8x)$ Bessel correction: $2\times10^{-3}$ at
$x = 50$ falling to $10^{-6}$ at $3200$ for $b = 0.2$ *[checked, off-cut]*.

---

## 7. The $b\to0$ reduction and the relation to Kohri–Terada

**[derived]** At $b = 0$: $c_s^2 = \tfrac13$, $y = \tfrac{u^2+v^2-3}{2uv} = \tfrac{d}{2uv}$ in KT's
$d$; $\mathcal N = \tfrac{3\pi}{8}$; $\rho = 2$; $A = 3y^2$; $B = \tfrac{3y}{2}(y\ln\tfrac{1+y}{1-y}-2)$;
$C(-y) = -\tfrac{3y}{2}(y\ln\lvert\tfrac{1+y}{1-y}\rvert-2)$, so $B\Theta_+ - C\Theta_- =
\tfrac{3y}{2}(y\ln\lvert\tfrac{1+y}{1-y}\rvert-2)$ on both branches. With $J_{1/2}(x) =
\sqrt{\tfrac{2}{\pi x}}\sin x$, $Y_{1/2}(x) = -\sqrt{\tfrac{2}{\pi x}}\cos x$ exactly,
$$I_{\rm target}(b=0) = -\frac{27\pi d^2}{32u^3v^3x}\Theta(u+v-\sqrt3)\cos x + \frac{27d}{32u^3v^3x}\big(d\ln\lvert\tfrac{3-(u+v)^2}{3-(u-v)^2}\rvert - 4uv\big)\sin x = \frac98\times\text{KT eq. (25)},$$
and `1912.tex:339-341` is KT (25) with the $\tfrac98$ absent, i.e. $I_{2020} = I_{\rm KT}$ as the
paper claims at `1912.tex:342`. **The constant relating the target object at $b = 0$ to KT eq. (25),
and $I_{\rm rev}$ to KT eq. (22), is $\tfrac98 = \tfrac{1}{2c^2}\big|_{b=0}$; it is derived, not
guessed**, from $f_{\rm KT} = 2c^2f_{\rm rev}$ (§8) and identical Green's functions. *[checked:
**E** $= \tfrac98\,$`I_RD` to $\le4.5\times10^{-15}$ at $(u,v,x)\in\{(0.7,1.1,20), (0.7,1.1,200),
(1.2,0.5,100), (0.3,1.6,300), (10,12,141)\}$; $3.9\times10^{-10}$ at $(0.01,1.1,50)$, which is
eq. (22)'s own rounding at $u = 0.01$ (KT §7.2)]*

The difference between $I_{\rm target}(b=0)$ and $\tfrac98 I_{\rm KT}$(eq. 22) is exactly KT's
"eq. (22) vs eq. (25)" column, so README (i)'s numbers ($1.03\times10^{-1}$ at $x = 200$, …) are
the $b = 0$ finite-upper-limit error of the target, unchanged.

---

## 8. Normalisation: does $N(b)$ follow, and with what value

**[derived]** Three steps, each b-explicit.

1. **$f_{\rm code} = f_{\rm rev}$.** `QuadSource.py:64-92` computes
   $\tfrac{5+3w}{3(1+w)}T_qT_r + \tfrac{2}{3(1+w)}\big[-(1+z)(T_qT_r'+T_rT_q') + (1+z)^2T_q'T_r'\big]
   = T_qT_r + \tfrac{2}{3(1+w)}\big(T-(1+z)T'\big)_q\big(T-(1+z)T'\big)_r$ (spec 03 §0.2). With
   $\tfrac{2}{3(1+w)} = \tfrac{1+b}{2+b}$ and $\tfrac{T'_\tau}{\mathcal H} = \tfrac{dT}{dz}\tfrac{dz}{d\eta}\tfrac{1+z}{a_0H}
   = -(1+z)\tfrac{dT}{dz}$, this is `review.tex:497-499` term for term, for the same $T$
   normalised to 1 early. No $b$-dependent factor.
2. **$f_{\rm KT} = 2c^2f_{\rm rev}$ for all $w$.** KT eq. (16) as implemented in
   `kohri_terada.py:f_RD` has $c_1 = \tfrac{6(1+w)}{5+3w}$, $c_2 = \tfrac{6(1+3w)(1+w)}{(5+3w)^2}$,
   $c_3 = \tfrac{3(1+3w)^2(1+w)}{(5+3w)^2}$ on $\Phi\Phi$, $[x\Phi'\Phi + \Phi x\Phi']$, $(x\Phi')(x\Phi')$.
   Expanding $f_{\rm rev}$ with $\tfrac{T'}{\mathcal H} = \tfrac{xT_x}{1+b}$ and $1+b = \tfrac{2}{1+3w}$,
   $2+b = \tfrac{3(1+w)}{1+3w}$, $3+2b = \tfrac{5+3w}{1+3w}$: coefficients $\tfrac{5+3w}{3(1+w)}$,
   $\tfrac{1+3w}{3(1+w)}$, $\tfrac{(1+3w)^2}{6(1+w)}$. All three ratios are
   $\tfrac{18(1+w)^2}{(5+3w)^2} = 2c^2$. This is the 2020 paper's definitional $2c^2$
   (`1912.tex:192`), i.e. $I_{\rm KT} = I_{2020} = 2c^2I_{\rm rev}$ at every $b$.
3. **The Green's function and the sign are $b$-independent.** Spec 04 §0(2) gives
   $G_{\rm code}(z,z') = -a_0H(z')\,\mathrm{Gr}_k$ for the operator with general $\epsilon$, and KT's
   $G_k$ is $\mathrm{Gr}_k$ for any background (KT §3). The KT §3 mapping,
   $\text{total} = \tfrac{a_0^2}{k^2}\tfrac{f_{\rm code}}{f_{\rm KT}}I_{\rm KT}$, therefore holds at every $b$
   with the sign fixed once by spec 04 §0(3)'s orientation convention.

Hence
$$\boxed{\;\text{total} = -\frac{1}{k_{\rm phys}^2}\,I_{\rm rev}(x,u,v)\quad\text{for every }b,\qquad\text{equivalently}\qquad N(b)\equiv\frac{k_{\rm phys}^2\,\text{total}}{I_{\rm KT}} = -\frac{1}{2c^2} = -\frac{(3+2b)^2}{2(2+b)^2}.\;}$$

**$N(b)$ follows from Domènech's definitions of $I$ and $f$, and its value is the predicted one.**
Of the three factors in $-\tfrac98 = -\tfrac12\times\tfrac94$: the **$\tfrac94 = 1/c^2$ carries all
the $b$-dependence** (it is the $\Phi_{\rm prim} = c\,\mathcal R$ fold, `1912.tex:112-114`, that the
2020 paper and KT put into $I$ and the review and the code do not); the **$\tfrac12$ carries
none** (it is the 2020 `eq:pgamma` "2" against KT's "4" in $\mathcal P_h$, `review.tex:396`, the
$h^{\rm us}_{ij} = h^{\rm them}_{ij}/2$ of spec 05); the **sign carries none**. At $b = 0.2$:
$N = -\tfrac{11.56}{9.68} = -1.194214876\ldots$ against $I_{\rm KT}$-convention, and $-1$ exactly
against $I_{\rm rev}$. **This is a derivation, not a measurement**, and KT §0's point stands: what A1
must show is that $N$ is *constant* over the nine $b = 0.2$ cases, using the finite-$x$
(4.11) by quadrature (§6.4). A constant $N$ that is not $-1.1942$ would be a real finding about
step 1 or 3; a drifting $N$ would be about the pipeline. Conventions not to "correct": $a_0$ is
absorbed (the mapping is written in $1/k_{\rm phys}^2$ for that reason); the sign is spec 04
§0(3); $c_s^2$ is `wPerturbations`.

---

## 9. Implementation brief for A1

Everything A1 needs, once. Symbols: $b$ the code's `b`; $c_s^2 = w = \tfrac{1-b}{3(1+b)}$;
$u = q/k$, $v = r/k$; $x = k\tau(z_{\rm resp})$ with $\tau = a_0\eta$; $\rho = \tfrac{b+2}{b+1}$;
$\delta = c_s(u+v)-1$; $\Theta_\pm = \Theta(\pm\delta)$.

**Kinematics** (form the small differences directly, §5.3):
$$1+y = \frac{\delta(2+\delta)}{2c_s^2uv},\qquad 1-y = \frac{(1-c_s(u-v))(1+c_s(u-v))}{2c_s^2uv},\qquad y = \frac{c_s^2(u^2+v^2)-1}{2c_s^2uv},\qquad\tilde y = -y.$$
Guard regions: $c_s|u-v|\ge1$ ($y\ge1$): outside the papers; refuse, or use §6.2's inferred
continuation and say so. $b\in\{0\}$: KT closed form. $b = \pm\tfrac12$: closed forms of §3.4, do
not call `legenq`. $|\delta|$ small with $b\le0$: the target diverges; only the exact finite-$x$
kernel is meaningful.

**The target object** (corrected sign; §2.4):
$$I_{\rm target}(x,u,v) = 4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}\,\frac{x^{-b-1/2}}{c_s^2uv}\Big\{\sqrt{\tfrac\pi2}\,Y_{b+1/2}(x)\,A(y)\,\Theta_+ + \sqrt{\tfrac2\pi}\,J_{b+1/2}(x)\big[B(y)\Theta_+ - C(\tilde y)\Theta_-\big]\Big\}.$$
$J_{b+1/2}(x)$, $Y_{b+1/2}(x)$ at production $x$ come from the repository's own large-argument
machinery (README (l)), not `scipy`.

**Coefficients:**
$$A(y) = \frac{2b+3}{2^b\Gamma(1+b)(b+1)}(1-y^2)^b\Big[1-\frac{b+2}{2(b+1)}(1-y^2)\Big]\quad\text{(elementary)},$$
$$B(y) = R_b(y)+\rho R_{b+2}(y),\qquad R_\nu(y) = \frac\pi2\cot(b\pi)\,(1-y^2)^{b/2}\mathsf P^{-b}_\nu(y) - 2^{b-1}\Gamma(b)\frac{\Gamma(\nu-b+1)}{\Gamma(\nu+b+1)}\,{}_2F_1\Big(-\nu-b,1+\nu-b;1-b;\frac{1+y}{2}\Big),$$
$$C(\tilde y) = S_b(\tilde y)+2\rho S_{b+2}(\tilde y),\qquad S_\nu(\tilde y) = -\frac{\pi}{2\sin(b\pi)\Gamma(\nu-b+1)}\Big[(\tilde y-1)^b\mathbf F\big(\nu+1,-\nu;1+b;\tfrac{1-\tilde y}{2}\big) - \frac{\Gamma(\nu-b+1)}{\Gamma(\nu+b+1)}(\tilde y+1)^b\mathbf F\big(\nu+1,-\nu;1-b;\tfrac{1-\tilde y}{2}\big)\Big],$$
with $(1-y^2)^{b/2}\mathsf P^{-b}_\nu(y)$ from §3.5's elementary forms and $\mathbf F = {}_2F_1/\Gamma(c)$.
**The factor 2 on $S_{b+2}$ is $\Gamma[3]$ and is correct.** Cross-check each of $A$, $B$, $C$
against the six raw calls of §3.2 away from $y = \pm1$: `legenp(nu, -b, y, type=2)`,
`legenq(nu, -b, y, type=2)`, and `Re[exp(i b π) legenq(nu, -b, ỹ, type=3)] / Γ(nu - b + 1)`,
$\nu\in\{b, b+2\}$, `mp.dps ≥ 30`.

**Doubly asymptotic form** (for large-$x$ regression only; not what A1 ships):
$I\approx x^{-b-1}4^b\Gamma^2[b+\tfrac32]\tfrac{2b+3}{b+2}\tfrac{1}{c_s^2uv}\{-\cos(x-\tfrac{b\pi}{2})A\Theta_+ +
\tfrac2\pi\sin(x-\tfrac{b\pi}{2})[B\Theta_+ - C\Theta_-]\}$ — the review's (4.12) with its $\cos$ sign
flipped.

**The exact kernel for the decisive check** (§5.1 of the README): quadrature of
$$I_{\rm rev}(x,u,v) = \frac\pi2x^{-b-1/2}\int_0^x d\tilde x\,\tilde x^{b+3/2}\big[J_{b+1/2}(\tilde x)Y_{b+1/2}(x) - J_{b+1/2}(x)Y_{b+1/2}(\tilde x)\big]f(\tilde x,u,v)$$
with $f$ from §2.1, breaking the range at every period of $1$, $c_s(u\pm v)$, $c_su$, $c_sv$ as
`kohri_terada.py:I_RD_quadrature` does. Then
$\text{total} = -k_{\rm phys}^{-2}\big[I_{\rm rev}(x,u,v) - \text{head}\big]$, the head being the same
integral over $[0, k\tau(z_{\rm source\,max})]$ with the outer $x$ in the kernel (README (k)).

**Assertions** (§6.4 table): small-$x$ $\tfrac{x^2}{2(2+b)}$; $b = 0$ against $\tfrac98\,$`I_RD`
and $\tfrac98\,$`I_RD_asymptotic`; $1/x$ approach of target to $I_{\rm rev}$ away from resonance and
small arguments; $B(-1) = -C(1) = -\tfrac{2^b\Gamma(b)(3+2b)(1+b+b^2)}{(1+b)\Gamma(2b+3)}$ for $b>0$;
sign of $I_{\rm rev}$ positive at small $x$; DLMF 14.2.4 and 14.2.8 Wronskians on the six calls.

**Expected $N$:** $-1$ against $I_{\rm rev}$; $-\tfrac{(3+2b)^2}{2(2+b)^2}$ against the KT/2020
convention. Constancy over $(u,v,x)$ is the statistic.

---

## 10. Open questions this pass could not settle, and what would settle each

1. **The Gervois–Navelet original was not read.** Both papers' transcriptions of it were checked
   only against each other and against **E** numerically (to $O(1/x)$ at $b\in\{0.2, 0.5\}$, both
   branches, and against KT at $b = 0$). That is strong, but a reading of Gervois & Navelet's
   formulas would say whether the papers' appendices have any condition or case the numerics did
   not probe (e.g. $b$ outside $[-\tfrac12,1)$, or $\nu-\rho$ not an integer). *Settled by:* the
   reference itself, or by A1's quadrature at more $b$.
2. **The $y>1$ continuation is inferred numerically, not read.** §6.2's rule (off-cut form at
   $+y$ with the opposite sign, $\mathcal I_J = 0$) matched **E** as $1/x$ on three shapes at
   $b = 0.2$. It is the third G–N case the papers omit, needed only for the unphysical `T-first`
   shape at $b = 0$, where KT eq. (22) is exact anyway. *Settled by:* A1 deciding whether the
   oracle needs `T-first` at $b = 0$ at all; if yes, the G–N paper or a $b = 0$ comparison with
   `I_RD`.
3. **Scoring the resonance for $b\le0$.** The target diverges there; the exact kernel does not.
   The general-$b$ oracle at the resonance is therefore the finite-$x$ quadrature only, which is
   the instrument KT §8 shows failing above $x\sim3\times10^4$. Whether a Levin-type evaluation of
   (4.11) at large $x$ is worth building is a campaign decision, not this pass's.
4. **Non-uniform $O(1/x)$.** §5.4 and §6.4 measured the control parameters ($|\delta|x$,
   $c_s\min(u,v)x$) at a handful of points. A map of the target's error over the physical
   triangle at fixture $x$ would tell A1 where the target is a $10^{-3}$ oracle and where it is a
   $10^{-1}$ one. *Settled by:* a scan, cheap with the harness of item 8.
5. **`mpmath` fragilities** (§3.4): `hyp2f1` degenerate-case failure at $b = \tfrac12$, `nan`/`-inf` at
   $y = -1$, complex `type=3`. All avoidable with §9's forms; none investigated beyond what is
   reported. Whether `scipy.special.hyp2f1` in double precision suffices for $B$ and $C$ on the
   whole domain (it did to $3\times10^{-16}$ near the resonance, §5.3) was not tested at large
   $|y|$ ($\gtrsim100$ on `q-smooth`), where the $\tilde y^{-\nu-1}$ decay (DLMF 14.8.15) and the
   $(\tilde y\pm1)^b$ factors may cancel. *Settled by:* comparing double against `mp.dps = 30` on the
   fixture points.
6. **Source typos, for the record, none load-bearing:** review (4.10) overall sign and (4.12) $\cos$
   sign (§1, load-bearing for $I$, not for $\overline{I^2}$); 2020 `1912.tex:291-292` "$y\to1$"
   for $y\to-1^+$; 2020 `1912.tex:759` garbled DLMF 14.9.2; review `review.tex:2022` "$\mu<0$"
   label; review `review.tex:2073-2074` $(\tfrac{2}{1-y})^{b/2}$ for $y>1$; 2020 commented-out
   `1912.tex:236`. The README (i2) item 1 statement that the powers of $(c_s^2uvx)$ differ is a
   misreading of $w$ for $c_s$; it should be corrected when the README is next touched.
7. **The oscillation average, one paragraph as requested.** Implementing (4.14)
   `eq:kernelaverage` would need only $A$, $B$, $C$ of §9 — it is
   $x^{-2(b+1)}4^{2b}\Gamma^4[b+\tfrac32]\big(\tfrac{2b+3}{b+2}\big)^2\tfrac{1}{2c_s^4u^2v^2}\{A^2\Theta_+ +
   \tfrac{4}{\pi^2}(B\Theta_+ - C\Theta_-)^2\}$ in this document's notation, which I reproduced from the
   corrected (4.12) — plus a definition of "average" the repository does not have (the review's
   is "integrate over half a period and divide by $\pi$", `review.tex:533`), plus a decision about
   what it means at finite $x$ where $J_{b+1/2}(x)$ and $Y_{b+1/2}(x)$ are not pure sinusoids.
   (4.42) `eq:kernelsuperhave2` is a reheating-transition object and does not belong here. The
   review's (4.14) is insensitive to both sign errors of §1, which is why they could survive.
8. **Worth landing.** The two scratch scripts (Legendre conventions and closed forms against
   `mpmath` at 30 digits; **E** by breakpointed `scipy.quad` with the target, (4.10) both orders,
   and (4.12) both signs, plus the KT tie at $b = 0$) are exactly the harness A1's §5.1
   obligation needs. They should be rewritten by A1 under `docs/handover/` per the campaign's
   rules (run from the root with `PYTHONPATH=.`, no Ray, no datastore), not copied: this document
   is an input, and a harness that inherits its mistakes would inherit them silently. The
   numbers in §2.4, §4.1, §4.2, §5.3, §5.4 and §6.4 are what such a harness should reproduce
   first.
9. **What was not derived.** The G–N formulas themselves; the review's superhorizon
   `review.tex:1914-1918` (its $1/v$ asymmetry looks odd and was not needed); the review's IR
   formulas `review.tex:810-820`; anything about $b\ge1$ or $c_s\ne\sqrt w$.
