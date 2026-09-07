# Diff 05 — `05-one-loop.md` (primary) vs `05-one-loop-B.md` (duplicate), MAIN 14 content

Comparison of the two independent transcriptions of
`Green's function formula for P22/14 - 2022:08:30 - recheck P22 power spectrum.pdf` (12 pp.).
Produced 2026-09-07 by the comparison agent from the two spec files only (README §7.1); the PDF
and the code were not consulted, and neither spec file was modified.

Scope: MAIN 14 only. The primary's MAIN 11 results (its R1–R14) are ignored except where its
Checks section bears on MAIN 14. Primary numbering is R15–R33; duplicate numbering is R1–R33.

**Headline.** 35 aligned rows: 33 AGREE, 1 MINOR, 1 DISAGREE, 0 ONLY-IN-PRIMARY, 0 ONLY-IN-DUPLICATE.
The single DISAGREE is the *letter used for the primordial seed variable* (primary reads
$S^*_{\mathbf q}$, duplicate reads $\zeta^*_{\mathbf q}$); it does not change any coefficient,
exponent, Bessel order or sign anywhere. The TARGET (p.11) and the outer $P^h_{22}$ formula (p.6)
**AGREE glyph for glyph**.

---

## 0. Global notational differences (not counted in the verdicts)

These are transcriber typesetting choices applied uniformly throughout each file. Both files
state explicitly that the quantities involved depend only on magnitudes, so none is a
mathematical difference. They are listed once here and ignored in the table.

| Item | Primary writes | Duplicate writes | Remark |
|---|---|---|---|
| Green's-function label | $\mathrm{Gr}_k$ (scalar) except on p.6 where $\mathrm{Gr}_{\mathbf k}\mathrm{Gr}_{-\mathbf k}$ appears | $\mathrm{Gr}_{\mathbf k}$ (bold) throughout | Both state $\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}$ (p.6). |
| Arguments of $f$ | $f(q,r,\eta)$ mostly; $f(\mathbf q,\mathbf k-\mathbf q,\eta')$ on p.5–6 | $f(\mathbf q,\mathbf r,\eta)$ mostly; $f(\mathbf q,\mathbf k-\mathbf q,\eta')$ on p.5–6 | Both state $f$ depends only on $q$, $r$. |
| Arguments of $P_*$ | $P_*(q)\,P_*(\lvert\mathbf k-\mathbf q\rvert)$ | $P_*(\mathbf q)\,P_*(\mathbf k-\mathbf q)$ | Duplicate Q12 says the *page* writes vector arguments; the primary writes magnitudes without comment. Glyph-level but inconsequential (isotropy). |
| Argument of $P^h_{22}$ | $P^h_{22}(k)$ | $P^h_{22}(\mathbf k)$ | As above. |
| Polar angle | $\Theta$ (taken from MAIN 11 p.9) | $\theta$ | Same angle (polar angle of $\mathbf q$ about $\mathbf k$). |
| Seed variable | $S^*_{\mathbf q}$, "$S^*$" | $\zeta^*_{\mathbf q}$, "$\zeta^*$" | **This one is a glyph-reading disagreement, D1 below.** Rows whose formula contains the seed symbol are marked AGREE† (identical apart from that letter). |

---

## 1. Alignment table

Rows are ordered by page. Where one file bundles several displays under one R-number, the
bundle is split so each row is a single formula.

| # | Primary | Duplicate | Page | Description | Verdict |
|---|---|---|---|---|---|
| 1 | R15 | R1 | 1 | Sourced equation for one polarisation, in $w$ and $\mathcal H$ | **MINOR** — primary reads the metric perturbation as lower-case $\phi_{\mathbf q}$, duplicate as $\Phi_{\mathbf q}$ (duplicate Q13 notes the glyph varies). All coefficients identical. |
| 2 | R16 | R2 | 1 | Linear solution $\frac{3(1+w)}{5+3w}\Phi(q\eta)\times$seed; $\Phi\to1$ as $q\eta\to0$ | **DISAGREE** (D1) — seed letter $S^*$ vs $\zeta^*$. Coefficient and limit identical. |
| 3 | R17 (prose), Conventions | R3 | 1 | $\mathbf r = \mathbf k-\mathbf q$; $\mathcal H = \frac{2}{1+3w}\frac1\eta$ | AGREE (primary carries this in prose and in its Conventions table rather than as a numbered result). |
| 4 | R17, 1st display | R4 | 1 | Source in terms of $\Phi$ with chain-rule factors $\frac r{\mathcal H}$, $\frac q{\mathcal H}$, $\frac{qr}{\mathcal H^2}$; coefficient $\frac{6(1+w)}{(5+3w)^2}$ | AGREE† |
| 5 | R17, 2nd display | R5 | 2 | Fixed-$w$ form: $h_s'' + \frac{4}{1+3w}\frac{h_s'}{\eta} + k^2h_s$; coefficient $\frac{3(1+w)(1+3w)}{(5+3w)^2}$; $\frac{1+3w}{2}q\eta\,r\eta\,\Phi'\Phi'$ | AGREE† |
| 6 | R18 (homogeneous eq.) | R6 | 2 | $h_s'' + 2\frac{a'}{a}h_s' + k^2h_s = 0$ | AGREE |
| 7 | R18 ($\chi_s$ eq.) | R7 | 3 | $\chi_s'' + (k^2 - a''/a)\chi_s = 4a\int\ldots f$ with $h_s = \chi_s/a$ | AGREE† |
| 8 | R18 ($f$) | R8 | 3 | $f(q,r,\eta)$ in $w$-form, last term $\frac{1+3w}{2}qr\eta^2\Phi'\Phi'$ | AGREE |
| 9 | R18 (Gr eq.) | R9 | 3 | $\mathrm{Gr}'' + (k^2 - a''/a)\mathrm{Gr} = +\delta(\eta-\eta')$ | AGREE (both note the first $\mathrm{Gr}$ lacks its subscript on the page). |
| 10 | R18 ($h_s$) | R10 | 3 | $h_s = \int_{\eta_0}^{\eta}d\eta'\,4\frac{a(\eta')}{a(\eta)}\mathrm{Gr}(\eta,\eta')\int\ldots f(\ldots,\eta')$ | AGREE† (both record the arrow-inserted $\mathrm{Gr}$ factor). |
| 11 | R19 | R11 | 3 | $Q_s = e_s^{lm}q_lq_m$; $e_+^{lm}$, $e_\times^{lm}$ with $\frac1{\sqrt2}$; $e_s^{lm}e_{s'lm} = \delta_{ss'}$ | AGREE |
| 12 | R19 | R12 | 4 | $Q_+ = \frac{q^2}{\sqrt2}\sin^2\theta\cos2\phi$, $Q_\times = \frac{q^2}{\sqrt2}\sin^2\theta\sin2\phi$ | AGREE ($\Theta$/$\theta$ is global). |
| 13 | R20 | R13 | 4 | $\Phi(k\eta) = 2^{3/2+b}\Gamma(\frac52+b)(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta)$; $b = \frac{1-3w}{1+3w}$ | AGREE |
| 14 | R21 | R14 | 4 | $\mathrm{Gr} = \frac\pi2\sqrt{\eta\eta'}\{J_{b+\frac12}(k\eta')Y_{b+\frac12}(k\eta) - Y_{b+\frac12}(k\eta')J_{b+\frac12}(k\eta)\}$, $\eta>\eta'$; $0$ otherwise | AGREE |
| 15 | R22, 1st display | R15 | 4 | $\langle h_sh_{s'}\rangle$ before contraction: $16\int\!\!\int$, $\mathrm{Gr}_k\mathrm{Gr}_{k'}$, $a$-ratios, $Q_sQ_{s'}$, four-point, $f(q,r,\eta')f(t,u,\eta'')$ | AGREE† (primary: medium on the letter "$u$"; duplicate: high, but Q14 notes $u$ undefined). |
| 16 | R22 (Wick) | R16 | 5 | Two connected pairings + disconnected | AGREE† |
| 17 | R22, p.5 displays | R17 | 5 | $(2\pi)^6P_*P_*\{\delta\delta + \delta\delta\}$; then $\int\frac{d^3q}{(2\pi)^3}(2\pi)^3\delta(\mathbf k+\mathbf k')P_*P_*\{Q_sQ_{s'}(\mathbf k',-\mathbf q)ff + Q_sQ_{s'}(\mathbf k',\mathbf k'+\mathbf q)ff\}$ | AGREE (both record the struck $Q_s$ after the $\delta$). |
| 18 | R23 (identities ①②, definition, double-integral form) | R18 | 6 | $Q_{s'}(\mathbf k',\mathbf k'+\mathbf q) = Q_{s'}(\mathbf k',\mathbf q)$; $f(q,r) = f(r,q)$; $\langle h_sh_{s'}\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')P^h_{22}$; $P^h_{22} = 32\int\!\!\int\ldots\mathrm{Gr}_{\mathbf k}\mathrm{Gr}_{-\mathbf k}\ldots Q_s(\mathbf k,\mathbf q)Q_{s'}(-\mathbf k,-\mathbf q)f(\mathbf q,\mathbf k-\mathbf q,\eta')f(-\mathbf q,\mathbf q-\mathbf k,\eta'')$ | AGREE (duplicate: medium because of a struck glyph in the last $f$; primary: high, but records the same struck glyph in its §5). |
| 19 | R23 ($\delta_{ss'}$ form) | R19 | 6 | $Q_{s'}(-\mathbf k,-\mathbf q) = Q_{s'}(\mathbf k,\mathbf q)$; $f(\mathbf q,\mathbf k-\mathbf q) = f(-\mathbf q,\mathbf q-\mathbf k)$; $\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}$; $32\int\!\!\int\ldots\delta_{ss'}\int Q_s^2 ff$ | AGREE |
| 20 | R23 (boxed) | R20 | 6 | **Outer structure:** $P^h_{22} = 32\int\frac{d^3q}{(2\pi)^3}Q_s^2P_*P_*\big(\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}\mathrm{Gr}f\big)^2$, "for any $s$" | AGREE |
| 21 | R23 (p.7 note) | R21 | 7 | $\int Q_sQ_{s'}\cdots\propto\delta_{ss'}$ from the $\phi$-integral over $2\pi$ if $f$ is $\phi$-independent | AGREE (duplicate: medium because of a stray "e"-like glyph; primary silent on it). |
| 22 | R24 | R22 | 7 | $1+3w = \frac2{1+b}$, $1+w = \frac{2(2+b)}{3(1+b)}$, $5+3w = \frac{2(3+2b)}{1+b}$ | AGREE |
| 23 | R25 | R23 | 7 | $f$ in $b$-form: $\frac{2+b}{3+2b}\Phi\Phi + \frac{2+b}{(3+2b)^2}\{r\eta\Phi\Phi' + q\eta\Phi'\Phi + \frac1{1+b}qr\eta^2\Phi'\Phi'\}$ | AGREE |
| 24 | R26 | R24 | 8 | $\Phi'(x) = -2^{3/2+b}\Gamma(\frac52+b)(xc_s)^{-3/2-b}c_sJ_{5/2+b}(xc_s)$ (with the intermediate $J' - (\frac32+b)\frac1{xc_s}J$ line) | AGREE |
| 25 | R26, R28 (prose) | R25 | 8–9 | Bessel identities $2Z' = Z_{\alpha-1}-Z_{\alpha+1}$, $\frac{2\alpha}xZ = Z_{\alpha-1}+Z_{\alpha+1}$, $Z' - \frac\alpha xZ = -Z_{\alpha+1}$, $Z - \frac x{2\alpha}Z_{\alpha+1} = \frac x{2\alpha}Z_{\alpha-1}$ | AGREE |
| 26 | R27 | R26 | 8 | Four-term Bessel form of $f$: prefactor $\frac{2+b}{3+2b}2^{3+2b}\Gamma^2(q\eta c_s)^{-3/2-b}(r\eta c_s)^{-3/2-b}$; terms $J_{3/2+b}J_{3/2+b} - \frac{r\eta c_s}{3+2b}J_{3/2+b}J_{5/2+b} - \frac{q\eta c_s}{3+2b}J_{5/2+b}J_{3/2+b} + \frac{(q\eta c_s)(r\eta c_s)}{(3+2b)(1+b)}J_{5/2+b}J_{5/2+b}$ | AGREE (duplicate: **medium**, third term's first Bessel order over-written "3"/"5"; primary: high, no mention). |
| 27 | R28 | R27 | 9–10 | Two-term form: $\frac{2+b}{(3+2b)^3}2^{3+2b}\Gamma(\frac52+b)^2(q\eta c_s)^{-\frac12-b}(r\eta c_s)^{-\frac12-b}\{J_{\frac12+b}J_{\frac12+b} + \frac{2+b}{1+b}J_{\frac52+b}J_{\frac52+b}\}$ | AGREE (both note p.9 writes the coefficient as $\frac{3+2b}{1+b}-1$ and has a $k$-for-$r$ slip). |
| 28 | R28 (text) | R28 | 10 | "Matches Domènech Eq. (4.9) up to normalization … using $\Phi^*$ rather than [seed] as we do" | AGREE† (seed letter differs, D1). |
| 29 | R29 | R29, 1st display | 10 | Step 6 first line: $\int\frac{(\eta')^{1+b}}{\eta^{1+b}}\frac\pi2\sqrt{\eta\eta'}(JY-YJ)\times$R27/R28 | AGREE (both note $J_{5/2}$ written without "$+b$"). |
| 30 | R30 | R29, 2nd display | 10 | Step 6 second line: $\pi2^{2+2b}\frac{2+b}{(3+2b)^3}\Gamma^2\eta^{-\frac12-b}\int(\eta')^{\frac32+b}(q\eta'c_s)^{-\frac12-b}(r\eta'c_s)^{-\frac12-b}(JY-YJ)(JJ+\frac{2+b}{1+b}JJ)$ | AGREE (primary: **medium** on the over-written exponent glyph, high on the value; duplicate: high). |
| 31 | **R31 TARGET** | **R30 TARGET** | 11 | Step 6 endpoint: prefactor, $(c_s^2qr\eta)^{-\frac12-b}$, $Y_{b+\frac12}(k\eta)I_J - J_{b+\frac12}(k\eta)I_Y$, $I_{J/Y}$ | **AGREE** (see §2). |
| 32 | R31 (assembly) | R30 (assembly) | — | Transcriber's assembly of p.6 × p.11 (not on any page) | AGREE (primary additionally writes out $Q_s^2 = \frac12q^4\sin^4\Theta\{\cos^2,\sin^2\}2\phi$; duplicate says the same in words, Q16). |
| 33 | R32 | R31 | 11 | Step 7: $I_{J/Y} = (\frac2\pi)^{3/2}(kqrc_s^2)^{1/2}\int(\eta')^{2-b}\{j_b/y_b\}(j_bj_b + \frac{2+b}{1+b}j_{2+b}j_{2+b})$ | AGREE |
| 34 | R33 ($I_{j,y}$) | R32 | 12 | Definition of $I_{j,y}$ with $(\eta')^{2-b}$ | AGREE (both medium on the over-written exponent sign). |
| 35 | R33 (Fabrikant form) | R33 | 12 | $\frac4{\pi^2}\pi2^{2+2b}\ldots(qrk^2c_s^2)^{1/2}\eta^{1/2}(y_bI_j \mp j_bI_y) = \frac{2^{4+2b}}\pi\frac{2+b}{(3+2b)^3}\Gamma(\frac52+b)\,k(c_s^2qr\eta)^{-b}(y_bI_j - j_bI_y)$ | AGREE (both medium; both record $\mp$, the missing square on $\Gamma$, and the over-written letter before $j_b$). |

**Counts.** AGREE 33 (of which 7 are AGREE†, identical apart from the seed letter), MINOR 1
(row 1), DISAGREE 1 (row 2), ONLY-IN-PRIMARY 0, ONLY-IN-DUPLICATE 0.

---

## 2. TARGET check — MAIN 14 p.11, Step 6 endpoint

Primary R31 vs duplicate R30, compared glyph by glyph.

| Element | Primary | Duplicate | Verdict |
|---|---|---|---|
| Left-hand side | $\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}\mathrm{Gr}_k(\eta,\eta')f(q,r,\eta')$ | $\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}\mathrm{Gr}_{\mathbf k}(\eta,\eta')f(\mathbf q,\mathbf r,\eta')$ | Same integral (global notation only). **Presentational difference:** the duplicate states the LHS is *not* re-written on p.11 (the page continues "$= \pi\,2^{2+2b}\cdots$" from p.10); the primary says "the two boxes are on the page as displayed equations". Not a formula difference. |
| Numerical prefactor | $\pi\,2^{2+2b}$ | $\pi\,2^{2+2b}$ | AGREE. Both record the exponent as over-written and resolve to $2+2b$; both note it is forced by $\frac\pi2\cdot2^{3+2b}$. |
| Rational prefactor | $\dfrac{2+b}{(3+2b)^3}$ | $\dfrac{2+b}{(3+2b)^3}$ | AGREE |
| Gamma factor | $\Gamma\!\left(\tfrac52+b\right)^2$ | $\Gamma\!\left(\tfrac52+b\right)^2$ | AGREE (squared in both). |
| Power-law factor | $\big(c_s^2\,q\,r\,\eta\big)^{-\frac12-b}$ | $\left(c_s^2\,q\,r\,\eta\right)^{-1/2-b}$ | AGREE: same base ($c_s^2qr\eta$, i.e. $c_s$ squared, $q$, $r$, and the *outer* time $\eta$), same exponent $-\frac12-b$. |
| Outer Bessel pair | $Y_{b+\frac12}(k\eta)\,I_J - J_{b+\frac12}(k\eta)\,I_Y$ | $Y_{b+1/2}(k\eta)\,I_J - J_{b+1/2}(k\eta)\,I_Y$ | AGREE: order $b+\frac12$, argument $k\eta$ (outer time), $Y$ multiplies $I_J$, $J$ multiplies $I_Y$, sign **minus** between them. |
| $I_{J/Y}$ limits | $\int_{\eta_0}^{\eta}d\eta'$ | $\int_{\eta_0}^{\eta}d\eta'$ | AGREE |
| $I_{J/Y}$ power of $\eta'$ | $(\eta')^{\frac12-b}$ | $(\eta')^{1/2-b}$ | AGREE |
| $I_{J/Y}$ $k$-Bessel | $\{J_{\frac12+b}(k\eta')\ /\ Y_{\frac12+b}(k\eta')\}$, upper $= I_J$, lower $= I_Y$ | $\{J_{1/2+b}(k\eta')\ /\ Y_{1/2+b}(k\eta')\}$, upper $= I_J$, lower $= I_Y$ | AGREE: order $\frac12+b$, argument $k\eta'$ (integration time), no $c_s$. |
| Source bracket, term 1 | $J_{\frac12+b}(q\eta'c_s)\,J_{\frac12+b}(r\eta'c_s)$ | $J_{1/2+b}(q\eta'c_s)\,J_{1/2+b}(r\eta'c_s)$ | AGREE |
| Source bracket, term 2 | $+\frac{2+b}{1+b}\,J_{\frac52+b}(q\eta'c_s)\,J_{\frac52+b}(r\eta'c_s)$ | $+\frac{2+b}{1+b}\,J_{5/2+b}(q\eta'c_s)\,J_{5/2+b}(r\eta'c_s)$ | AGREE: sign plus, coefficient $\frac{2+b}{1+b}$, orders $\frac52+b$, arguments $q\eta'c_s$, $r\eta'c_s$. |
| Confidence | high for every glyph; only inherited uncertainty is the $2^{2+2b}$ over-write | high for every glyph; only over-writing is the $2^{2+2b}$ exponent | AGREE |

**Outer structure (p.6, primary R23 boxed vs duplicate R20):**

| Element | Primary | Duplicate | Verdict |
|---|---|---|---|
| Symmetry factor | $32$ | $32$ | AGREE (both explain as $2\times16$; the primary additionally derives $16 = 4^2$ from the source normalisation). |
| Loop measure | $\int\frac{d^3q}{(2\pi)^3}$ | $\int\frac{d^3q}{(2\pi)^3}$ | AGREE; both state no angular reduction is performed. |
| Polarisation factor | $Q_s(\mathbf k,\mathbf q)^2$ | $Q_s(\mathbf k,\mathbf q)^2$ | AGREE (squared; per polarisation, "for any $s$"). |
| Spectra | $P_*(q)P_*(\lvert\mathbf k-\mathbf q\rvert)$ | $P_*(\mathbf q)P_*(\mathbf k-\mathbf q)$ | AGREE (global notation; see §0). |
| Time integral | $\big(\int_{\eta_0}^{\eta}d\eta'\frac{a(\eta')}{a(\eta)}\mathrm{Gr}_k(\eta,\eta')f(\mathbf q,\mathbf k-\mathbf q,\eta')\big)^2$ | same with $\mathrm{Gr}_{\mathbf k}$ | AGREE (the whole single integral is squared; this is the quantity the TARGET evaluates). |
| Definition of $P^h_{22}$ | $\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')P^h_{22}(k)$ | same, $P^h_{22}(\mathbf k)$ | AGREE |

**TARGET verdict: AGREE.** No difference in any exponent, Bessel order, argument, sign,
integration limit, prefactor, or in the outer $P^h_{22}$ formula. The two transcribers'
assembled build targets (primary after R31, duplicate after R30) are algebraically the same
expression.

---

## 3. Disagreements

### D1 — the seed-variable letter (p.1, and wherever the seed appears; rows 2, 4, 5, 7, 10, 15, 16, 28)

Primary (R16, p.1):
```latex
\phi_{\mathbf q}(\eta) = \frac{3(1+w)}{5+3w}\,\Phi(q\eta)\,S^*_{\mathbf q},\qquad \Phi(q\eta)\to1 \text{ as } q\eta\to0
```
Duplicate (R2, p.1):
```latex
\Phi_{\mathbf q}(\eta) = \frac{3(1+w)}{5+3w}\,\Phi(q\eta)\,\zeta^*_{\mathbf q},\qquad \Phi(q\eta)\to1 \text{ as } q\eta\to0
```

- **Glyph that differs:** the letter of the starred seed variable, read as roman/italic $S$ by
  the primary and as $\zeta$ by the duplicate. The star, the subscript $\mathbf q$, the
  coefficient $\frac{3(1+w)}{5+3w}$, and the normalisation statement are identical.
- **Also affects** the Domènech quotation on p.10: primary "using $\Phi^*$ rather than $S^*$
  as we do"; duplicate "using $\Phi^*$ rather than $\zeta^*$ as we do".
- **Confidence flags:** neither transcriber flagged this glyph medium or low; both read it
  `high`. The primary notes only that "the star is written as a superscript on $S$ and as a
  subscript on $P_*$"; the duplicate (Q10) discusses only what the star means, not the letter.
- **Internal consistency:** each file is internally consistent — the primary uses $S^*$ in every
  occurrence (R16, R17, R18, R22, R28 text, Conventions, Open question 2); the duplicate uses
  $\zeta^*$ in every occurrence (R2, R4, R5, R7, R10, R15–R17, R28, Conventions, Q10). Neither
  file wavers.
- **Context, not adjudication:** the primary had just transcribed MAIN 11, where (per its R7,
  R11) the seed is written $S(\mathbf k)$ with spectrum $P_S$; its Checks table describes MAIN 14
  as "renam[ing] $S\to S^*$". The duplicate had not seen MAIN 11. Either reading could be primed
  (by MAIN 11's $S$, or by the conventional use of $\zeta$). This is a one-glyph page check.
- **Consequence:** none for any coefficient in the build target, but it fixes the *identity* of
  the primordial variable whose spectrum $P_*$ is (README §5, "which transfer function is
  meant"), so it matters for the cross-spec convention check (§7.2) against Groups 1 and 3.

### Near-disagreement (classified MINOR) — the metric-perturbation glyph (p.1, row 1)

Primary R15 writes the field in the starting equation as $\phi_{\mathbf q}$, $\phi'_{\mathbf q}$
(lower case, consistent with MAIN 11); duplicate R1 writes $\Phi_{\mathbf q}$, $\Phi'_{\mathbf q}$
(upper case) and in Q13 itself notes the page alternates between "a capital-looking $\Phi$ and a
lower-case-looking cursive $\phi$". Both agree the field is the scalar metric potential whose
transfer function is $\Phi(x)$, and the field is eliminated at row 2, so there is no
mathematical content in the difference. Listed so the author can confirm which case the page
intends (relevant only to consistency with Groups 1/3 notation).

---

## 4. Coverage gaps

No result on any page of MAIN 14 is present in only one file. Differences in *packaging*:

| Item | Primary | Duplicate | Matters? |
|---|---|---|---|
| $\mathbf r = \mathbf k-\mathbf q$, $\mathcal H = \frac{2}{1+3w}\frac1\eta$ (p.1) | Prose inside R17 + Conventions table | Numbered result R3 | No — content in both. |
| Bessel identity $Z_\alpha - \frac x{2\alpha}Z_{\alpha+1} = \frac x{2\alpha}Z_{\alpha-1}$ (p.9) | Prose inside R28 | Listed in R25 | No. |
| Domènech Eq. (4.9) remark (p.10) | Prose inside R28 | Numbered result R28 | No (but see D1 for the seed letter in the quote). |
| Explicit $Q_s^2 = \frac12q^4\sin^4\Theta\{\cos^2 2\phi\text{ or }\sin^2 2\phi\}$ | Written out in the R31 assembly and Conventions | Stated in words (Q16) | No — a trivial square of row 12, and both say no angular reduction is on the page. |
| Step→page map | Per-result page refs only | Table in §1 | No. |
| Arithmetic self-consistency checks (R4→R5, R23, R27, R29→R30, R30→R31/R33) | Only the $\frac\pi2\cdot2^{3+2b}$ check | Full list in §4 | Helpful but not a transcription gap. |
| MAIN 11 ↔ MAIN 14 comparison table | §4 Checks | Not possible (duplicate did not read MAIN 11) | By design (README §3). |

---

## 5. Conventions (README §5), item by item

| Item | Primary | Duplicate | Verdict |
|---|---|---|---|
| Time variable | Conformal $\eta$ throughout; never $t$, $z$, $\log(1+z)$ | Conformal $\eta$; integration variables $\eta'$, $\eta''$; limits $\eta_0\to\eta$ | Agree |
| Prime on $h_s$, $a$, $\chi_s$ | $d/d\eta$ | $d/d\eta$ (also on $\mathrm{Gr}$ and $\Phi_{\mathbf q}(\eta)$) | Agree |
| **Prime on the transfer function** | $\Phi'(x) = d\Phi/dx$, $x = q\eta$; chain-rule factors $q$, $r$ explicit (p.1, p.8). Inferred, not stated. (Primary adds that MAIN 11 pp.5–6 behave as $d/d\eta$ — outside this diff.) | $d/dx$ w.r.t. own argument; inferred from p.1→p.2 chain-rule factor $q$ and from the factor $c_s$ on p.8. On Bessel functions $Z'_\alpha(x) = d/dx$. | Agree, including that it is inferred |
| $\mathcal H$ | $a'/a$; fixed $w$: $\frac2{1+3w}\frac1\eta$; $H = \mathcal H/a$ (from MAIN 11); $\epsilon$ unused | $a'/a$ (from p.2 homogeneous eq.); fixed $w$: $\frac2{1+3w}\frac1\eta$; $\epsilon$ unused | Agree |
| Scale-factor normalisation | Only $a(\eta')/a(\eta)$ appears; $(\eta'/\eta)^{1+b}$ used without comment; $a_0$ never mentioned | Same; $a\propto\eta^{1+b}$ "used, not stated" | Agree |
| **Green's-function source sign** | $+\delta(\eta-\eta')$ | $+\delta(\eta-\eta')$ | Agree |
| Green's-function arguments / BCs | First arg response $\eta$, second source $\eta'$; retarded ($0$ for $\eta<\eta'$); for $\chi_s = ah_s$; $\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}$; no literature Green's function | Same on every point | Agree |
| $w$, $b$, derived identities | $b = \frac{1-3w}{1+3w}$; $1+3w = \frac2{1+b}$, $1+w = \frac{2(2+b)}{3(1+b)}$, $5+3w = \frac{2(3+2b)}{1+b}$ | Same | Agree |
| $c_s$ | Appears in $\Phi$; **not defined**; no $c_s^2 = w$ or $c_s^2 = \frac{1-b}{3(1+b)}$ written | Same ("never defined in this document") | Agree |
| Fourier convention for $h_{ij}$ | $\int\frac{d^3k}{(2\pi)^3}e^{i\mathbf k\cdot\mathbf x}\sum_se^s_{ij}h_s$ — **from MAIN 11 p.2–3** | Not recorded (only the loop measure $\int\frac{d^3q}{(2\pi)^3}$) | Not a conflict: MAIN 14 does not state it; the primary's source is MAIN 11. |
| Two-point convention | $\langle S_{\mathbf q}S_{\mathbf q'}\rangle = (2\pi)^3\delta(\mathbf q+\mathbf q')P(q)$, inferred from the four-point function; $\langle h_sh_{s'}\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\delta_{ss'}P^h_{22}(k)$ | $\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle = (2\pi)^3\delta(\mathbf q+\mathbf t)P_*(q)$, inferred; $\langle h_sh_{s'}\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')P^h_{22}(\mathbf k)$, $\propto\delta_{ss'}$ | Agree (apart from D1 letter) |
| **Dimensionful $P$ vs dimensionless $\mathcal P$** | Dimensionful $P$; "No dimensionless $\mathcal P = k^3P/2\pi^2$ is used anywhere" | Dimensionful; "no dimensionless $\mathcal P(k) = k^3P/2\pi^2$ appears anywhere" | Agree |
| **Polarisation normalisation** | $e_\pm^{lm}$ with $\frac1{\sqrt2}$; $e_s^{ij}e_{ij\,s'} = \delta_{ss'}$; plus MAIN 11 remarks that the author's $h_{ij}$ is half Kohri–Terada's / Adshead et al.'s | $e_\pm^{lm}$ with $\frac1{\sqrt2}$; $e_s^{lm}e_{s'\,lm} = \delta_{ss'}$ | Agree on MAIN 14; the literature-normalisation remarks are MAIN 11-only. |
| Per-polarisation result | Yes, "for any $s$"; no sum over $s$, no $\Omega_{\rm GW}$ | Yes, per polarisation, no sum | Agree |
| **Symmetry factor** | $32 = 2\times16 = 2\times4^2$ (4 from source normalisation, 2 from the two Wick pairings) | $32 = 2\times16$ (two equal connected pairings) | Agree |
| **Scalar field / transfer function** | $\phi$ = author's own scalar metric perturbation (nonlinear definition differs from Adshead et al. — from MAIN 11); $\phi_{\mathbf q} = \frac{3(1+w)}{5+3w}\Phi(q\eta)S^*_{\mathbf q}$, $\Phi\to1$; explicit $\Phi(x)$ | "Newtonian-gauge-type potential $\Phi$" ("the transfer function for $\Phi$", p.8); $\Phi_{\mathbf q} = \frac{3(1+w)}{5+3w}\Phi(q\eta)\zeta^*_{\mathbf q}$, $\Phi\to1$; same explicit $\Phi(x)$ | **Conflict on the seed letter (D1: $S^*$ vs $\zeta^*$)**; MINOR on the case of the potential's glyph ($\phi$ vs $\Phi$). Both agree the seed is undefined on the page and its relation to $\zeta$/$\mathcal R$ is not stated; both agree the star is a label (primary: placement; duplicate Q10: not conjugation). |
| Momentum labels | $\mathbf k$, $\mathbf q$, $\mathbf r = \mathbf k-\mathbf q$; second copy $\mathbf t$, $\mathbf u = \mathbf k'-\mathbf t$; $\mathbf k' = -\mathbf k$ | Same | Agree |
| Angles | $\Theta$ polar of $\mathbf q$ w.r.t. $\mathbf k$, $\phi$ azimuth in plane $\perp\mathbf k$ (defined in MAIN 11 p.9) | $\theta,\phi$ polar angles of $\mathbf q$ about $\mathbf k$ — "not defined explicitly; inferred" | Agree (duplicate correctly notes MAIN 14 itself does not define them). |
| Angular reduction | Not carried out; only "$\phi$-integral over $2\pi$ gives $\delta_{ss'}$" | Same | Agree |
| $\eta_0$ | Undefined | Undefined | Agree |

**Conflicts:** exactly one — the seed-variable letter (D1). Everything else in §5 agrees,
including the three items singled out for attention: prime on the transfer function ($d/dx$),
Green's-function source sign ($+\delta$), and dimensionful $P$.

---

## 6. Corrections and open questions raised by one transcriber only

### Raised only by the primary (MAIN 14 items)

| Where | Item | Bearing |
|---|---|---|
| p.10, R29 (Open Q9) | Green's-function factor runs off the right margin ("$J_{b+\frac12}(k\eta$" cut); completed from p.4. | Reading forced by p.4 and by the p.10 second line; duplicate reads the same factor without comment. |
| p.12, R33 (§5) | Over-written letter before $j_b(k\eta)$ was originally "$g_b$". | Duplicate says "possibly y or g". Same resolution ($j_b$). Fabrikant part. |
| Open Q6 | Whether the deliverable is $P^h_{22}$, $2P^h_{22}$, or a dimensionless $\mathcal P_h$ is not fixed by the notes. | Duplicate records "per polarisation, no sum" but does not pose the deliverable question. Relevant to the audit, not to the transcription. |
| Open Q12 | The $\delta_{ss'}$ step holds "if $f$ is independent of $\phi$"; the author asserts rather than proves; transcriber remarks the condition is met. | Duplicate R21 quotes the same condition without the remark. |

(Primary Open Q1, Q7, Q11 and several §5 rows concern MAIN 11 only and are out of scope.)

### Raised only by the duplicate

| Where | Item | Bearing |
|---|---|---|
| p.8, R26 / C3 | Third term of the four-term $f$: first Bessel order over-written, "3" and "5" both visible; read as $J_{5/2+b}(q\eta c_s)$ by symmetry with the second term. **Medium.** | Primary reads the same order with `high` confidence and no mention of over-writing. Feeds R27→R28 (two-term form) and hence the TARGET, but the completed-square algebra fixes it; both files land on the same two-term form. |
| p.10, R29 / C4 | $(k\eta'c_s)^{-1/2-b}$ with a small "r" written above the "k" — corrected on the page. | Primary records the *uncorrected* p.9 slip but not this p.10 correction. Same resolved value. |
| p.7, R21 / Q2 | Isolated glyph resembling "e" or "$\in$" between the measure and $Q_s$. **Medium.** | Cosmetic. |
| p.7, Q3 | Stray "+" in the intermediate line above the $b$-form of $f$ ("$r\eta\Phi(q\eta) + \Phi'(r\eta)$"). | Intermediate algebra; final line is a product in both files. |
| p.9, Q5 | Last Bessel function of the p.9 final line written $J_{5/2}(r\eta c_s)$ without "$+b$". | Primary records the analogous omission on p.10 only. Same resolution. |
| Q10 | Star on $\zeta^*$, $P_*$ is a label, not complex conjugation (since $\langle\zeta^*\zeta^*\rangle\propto\delta(\mathbf q+\mathbf t)$). | Primary notes only the star's typographical placement. |
| Q12 | $P_*$ written with vector arguments on p.5–6. | See §0. |
| Q13 | Transfer-function/potential glyph alternates between $\Phi$ and cursive $\phi$ (p.2, p.7, p.9). | See §3 near-disagreement. |
| Q15 | Angles $\theta,\phi$ undefined in MAIN 14. | Primary sources the definition from MAIN 11 p.9. |
| Q18 | No explicit cross-reference to MAIN 11 anywhere in MAIN 14. | Primary's Checks §4 says the same ("the author does not write any explicit agrees/disagrees remarks"). Consistent. |
| §2 / R30 note | The LHS of the TARGET is not re-written on p.11; the page continues from p.10 with "$= \pi\,2^{2+2b}\cdots$". | Presentational; see §2. |

### Raised by both (for completeness; no action beyond the queue)

p.3 arrow-inserted $\mathrm{Gr}$; p.5 struck $Q_s$; p.6 struck glyph in $f(-\mathbf q,\cdot\,\mathbf q-\mathbf k,\eta'')$;
p.9 $k$-for-$r$ slip and coefficient $\frac{3+2b}{1+b}-1$; p.10 $J_{5/2}$ without "$+b$";
p.10–11 over-written exponent $\to 2^{2+2b}$; p.12 "$2\mp b$", "$\mp$", $\Gamma$ without square;
$c_s$, $S^*/\zeta^*$, $\eta_0$ undefined; no angular reduction; "$u$" on p.4.

---

## 7. Review queue for this pair (most consequential first)

1. **p.11, top — TARGET.** Both transcriptions agree on every glyph, so this is a confirmation
   read, not an adjudication. Confirm: $2^{2+2b}$ (over-written; both read $2b$ as the final
   value); $(c_s^2qr\eta)^{-\frac12-b}$; $Y_{b+\frac12}(k\eta)I_J - J_{b+\frac12}(k\eta)I_Y$
   with a **minus**; $I_{J/Y} = \int_{\eta_0}^{\eta}d\eta'(\eta')^{\frac12-b}\{J/Y\}_{\frac12+b}(k\eta')
   \big(J_{\frac12+b}J_{\frac12+b} + \frac{2+b}{1+b}J_{\frac52+b}J_{\frac52+b}\big)$ with arguments
   $q\eta'c_s$, $r\eta'c_s$. Also note whether the LHS integral is re-written on this page
   (duplicate says no).
2. **p.6, bottom — outer $P^h_{22}$ formula.** Both agree: $32\int\frac{d^3q}{(2\pi)^3}Q_s^2P_*P_*(\int\ldots)^2$,
   "for any $s$". Confirm the $32$, the square on $Q_s$, the square on the time integral, and
   whether $P_*$ carries vector or magnitude arguments (duplicate Q12). Also the struck glyph
   in $f(-\mathbf q,\;\cdot\;\mathbf q-\mathbf k,\eta'')$ two lines above (both flag it).
3. **p.1, line defining the linear solution (and p.10 Domènech remark) — D1.** Is the starred
   seed variable an $S$ or a $\zeta$? The only DISAGREE. While there, confirm whether the
   metric perturbation in the first equation is $\phi$ or $\Phi$ (MINOR; duplicate Q13).
4. **p.10, Step 6 lines.** The over-written exponent of 2 (primary flags medium on the glyph);
   the $k\to r$ over-write in $(r\eta'c_s)^{-1/2-b}$ (duplicate C4); $J_{5/2}$ written without
   "$+b$" (both); the right-margin cut of $J_{b+\frac12}(k\eta)$ (primary Q9).
5. **p.8, bottom — four-term Bessel form of $f$, third term.** Over-written Bessel order
   (duplicate medium, primary silent). Both read $J_{5/2+b}(q\eta c_s)J_{3/2+b}(r\eta c_s)$;
   the completed square on p.9 depends on it.
6. **p.9 — completed square.** $k$-for-$r$ slip (both), $J_{5/2}$ without "$+b$" (duplicate
   Q5), coefficient $\frac{3+2b}{1+b}-1$ (both). Both agree on the resulting p.10 two-term form.
7. **p.4 — second argument of the second $f$ ("$u$").** Primary medium; not used again.
8. **p.7 — stray glyphs.** "e"-like mark before $Q_s$ (duplicate Q2); stray "+" in the
   intermediate line (duplicate Q3). Cosmetic.
9. **p.5 — struck $Q_s$ after $\delta(\mathbf k+\mathbf k')$.** Both agree; confirmation only.
10. **p.11–12 — Step 7, Fabrikant form (not to be used).** "$2\mp b$" exponent, "$\mp$" sign,
    $\Gamma(\frac52+b)$ without its square in the last line, over-written letter before $j_b$.
    Both files agree on all readings and all flag medium. Lowest priority.
