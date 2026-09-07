# Diff 04 — `04-source-integral.md` (primary) vs `04-source-integral-B.md` (duplicate)

Scope: `NUM` 06, `06 - 2024-10-29 - analytic source integral.pdf` (11 pages). The primary also
covers `NUM` 07 (its R22–R29 and C13–C15, Q9–Q13); those are out of scope here and were ignored.
Compiled 2026-09-07 from the two spec files only; the PDF was not consulted and no physics
reasoning was used to adjudicate. Result numbers differ between the files, so rows are aligned by
page and content.

Verdict key: AGREE — same formula, same reading. MINOR — typesetting/presentation differs,
mathematically identical. DISAGREE — a different reading of at least one glyph. ONLY-IN-PRIMARY /
ONLY-IN-DUPLICATE — the display is transcribed as a result in one file only.

Summary: **20 AGREE, 1 MINOR, 3 DISAGREE, 0 ONLY-IN-PRIMARY, 3 ONLY-IN-DUPLICATE** (27 rows).
All three DISAGREE rows are the *same* glyph question — whether the Bessel order written as
$\tfrac52$ on pp.4 and 6 carries a "$+b$" — at three page locations. The final p.7 formula is
read identically by both transcribers.

---

## 1. Alignment table

| # | Primary | Duplicate | Page | Description | Verdict |
|---|---|---|---|---|---|
| 1 | R1 | R1 | 1 | Source integral in $z$: $\int_z^{z_{\rm init}} dz'\,G_k(z,z')\frac{1+z}{1+z'}\frac{1}{H(z')^2}f$ | AGREE |
| 2 | R2 | R2 | 1 | $G_{\rm me} = -H(z')\,G_{\rm them}$ | AGREE |
| 3 | R3 | R3 | 1 | $G_{\rm them}(\eta,\eta')$, four Bessel functions of order $b+\tfrac12$, retarded | AGREE (dup flags the struck glyph between the first two factors as *medium*; primary *high*; same reading) |
| 4 | R4 | R4 | 1, 4 | $T_k(\eta) = 2^{3/2+b}\Gamma(\tfrac52+b)(kc_s\eta)^{-3/2-b}J_{3/2+b}(kc_s\eta)$ | AGREE |
| 5 | R5 | R5 | 1 | Tensor source $f = T_qT_r + \frac{2M_P^2}{\rho_0(1+p_0/\rho_0)}\frac{1}{a^2}(T_q'+\mathcal H T_q)(T_r'+\mathcal H T_r)$ | AGREE (both *medium* on the small denominator; dup names an alternative "$P_0/P_0$") |
| 6 | R7 | R6 + part of R7 | 2–3 | Fixed-$w$ background: $a^{(1+3w)/2}\propto\eta$, $a\propto\eta^{2/(1+3w)} = \eta^{1+b}$, $\mathcal H = \frac{2}{1+3w}\frac1\eta = \frac{1+b}{\eta}$ | AGREE (grouping differs; dup additionally quotes the intermediate integration line with "$(\eta-\eta')$") |
| 7 | R8 | R7 | 3 | $w$–$b$ identities: $w=\tfrac13\frac{1-b}{1+b}$, $3(1+w)=\frac{2(2+b)}{1+b}$, $\frac{2}{1+3w}=1+b$ | AGREE (primary also lists $3w = \frac{1-b}{1+b}$; dup also lists the definition of $b$ and $\mathcal H=(1+b)/\eta$) |
| 8 | R6 | R8 | 3 | Source with background eliminated: $f = T_qT_r + \frac{2M_P^2}{3M_P^2}\frac{1}{1+w_0}(\frac{T_q'}{\mathcal H}+T_q)(\frac{T_r'}{\mathcal H}+T_r)$ | AGREE (both record the $w_0$ subscript) |
| 9 | R9 | R9 | 4 | $Z_\alpha' - \frac\alpha x Z_\alpha = -Z_{\alpha+1}$; $\frac{dT}{dx} = -2^{3/2+b}\Gamma(\tfrac52+b)x^{-3/2-b}J_{5/2+b}(x)$; intermediate brace lacks $1/x$ | AGREE |
| 10 | — (described in C3, C4 only) | R10, first display | 4 | $f$ with the derivative substituted, before expansion: $\{J J + \frac{1+b}{2+b}[-\frac{qc_s\eta}{1+b}J_{5/2+b}+J_{3/2+b}][-\frac{rc_s\eta}{1+b}J_{5/2+b}+J_{3/2+b}]\}$ | ONLY-IN-DUPLICATE |
| 11 | R11, first intermediate form | R10, expanded display | 4 | Expanded four-term form of $f$ with $(1+\frac{1+b}{2+b})$, $-\frac{qc_s\eta}{2+b}$, $-\frac{rc_s\eta}{2+b}$, $+\frac{(qc_s\eta)(rc_s\eta)}{(2+b)(1+b)}$ | **DISAGREE** — order of the last Bessel factor: $J_{5/2+b}(rc_s\eta)$ (primary) vs $J_{5/2}(rc_s\eta)$ (dup). See §2, D1. |
| 12 | — | R11, first display | 5 | Regrouped four-term form with $\frac{3+2b}{2+b}$ pulled out and $\frac{1}{3+2b}$ coefficients | ONLY-IN-DUPLICATE |
| 13 | R11, factored form | R11, factorised display | 5 | $\{(J_{3/2+b}-\frac{qc_s\eta}{3+2b}J_{5/2+b})(J_{3/2+b}-\frac{rc_s\eta}{3+2b}J_{5/2+b}) + (qc_s\eta)(rc_s\eta)(\frac{1}{(3+2b)(1+b)}-\frac{1}{(3+2b)^2})J_{5/2+b}J_{5/2+b}\}$ | AGREE (both record the 3→5 over-write, C5) |
| 14 | R10 | R11, "we also have" | 5 | $Z_\alpha - \frac{x}{2\alpha}Z_{\alpha+1} = \frac{x}{2\alpha}Z_{\alpha-1}$ | AGREE |
| 15 | — (described in C5 only) | R12, p.5 display | 5 (foot) | "so $f=$" with $\frac{qc_s\eta}{3+2b}J_{1/2+b}\,\frac{rc_s\eta}{3+2b}J_{1/2+b} + \frac{(qc_s\eta)(rc_s\eta)}{(3+2b)^2}(\frac{3+2b}{1+b}-1)J_{5/2+b}J_{5/2+b}$ | ONLY-IN-DUPLICATE |
| 16 | R11, final boxed form | R12, p.6 boxed | 6 (top) | **Final simplified source:** $f = \frac{2^{3+2b}}{(3+2b)(2+b)}\Gamma(\tfrac52+b)^2(qc_s\eta)^{-1/2-b}(rc_s\eta)^{-1/2-b}\{J_{1/2+b}J_{1/2+b} + \frac{2+b}{1+b}J_{?}J_{?}\}$ | **DISAGREE** — second product $J_{5/2+b}J_{5/2+b}$ (primary, *high*) vs $J_{5/2}J_{5/2}$ (dup, *medium*, "no $+b$"). See §2, D2. |
| 17 | R12 | R13 | 6 | Change of variables: $\frac{1+z}{1+z'} = (\eta'/\eta)^{1+b}$, $1+z = a_0/a$, $dz = -a_0H\,d\eta$ | AGREE |
| 18 | R13, p.6 display | R14, p.6 display | 6 (foot) | Integral in $\eta$ before substituting $G$: $-a_0\int_\eta^{\eta_{\rm init}}H(\eta')d\eta'\,G_k\,\frac{(\eta')^{1+b}}{\eta^{1+b}}\frac{1}{H(\eta')^2}\cdots$; Bessel arguments unprimed | **DISAGREE** — last Bessel factor $J_{5/2+b}(rc_s\eta)$ (primary) vs $J_{5/2}(rc_s\eta)$ (dup). Both agree the brace arguments are unprimed $\eta$. See §2, D3. |
| 19 | R13, p.7 display | R14, p.7 display | 7 (top) | After substituting R2, R3: $a_0\int_{\eta_{\rm init}}^{\eta}d\eta'(-1)\frac\pi2\sqrt{\eta\eta'}\{J_{b+1/2}(k\eta')Y_{b+1/2}(k\eta) - Y_{b+1/2}(k\eta')J_{b+1/2}(k\eta)\}\cdots$, all arguments primed | AGREE |
| 20 | R14 | R15 | 7 | **Final analytic source time integral** (see §2 preamble for the full expression) | AGREE (identical in every sign, order, exponent and prefactor; dup *medium* on the two cross-outs C7/C8 that primary records as C6/C7 with the same resolution) |
| 21 | R15 | R16 | 7–8 | Liouville normal form $y'' + (1 + \frac{1/4-\nu^2}{x^2})y = 0$ via $y = f\tilde y$, $f\propto x^{-1/2}$ | AGREE (primary also writes $f''/f + \frac1x f'/f = \frac{1}{4x^2}$) |
| 22 | R16 | R17 | 9 | Riccati $i\Theta'' - \Theta'^2 + \omega_{\rm eff}^2 = 0$; $\Theta = \vartheta + i\beta$; $\beta = \tfrac12\ln\vartheta'$; Kummer equation; WKB leading solution | AGREE (dup also writes the $\beta''$ line) |
| 23 | R17 | R18 | 10 | Three-Bessel integral in Liouville–Green form with $(2/\pi)^{3/2}(x_1x_2x_3)^{-1/2}(\beta_1\beta_2\beta_3)^{-1/2}\cos\gamma_1\cos\gamma_2\cos\gamma_3$ | AGREE |
| 24 | R18 | R19 | 10 | $\cos\cos\cos$ product-to-sum, all four signs $+$ | AGREE |
| 25 | R19 | R20 | 10 | $\sin\cos\cos$ ("Neumann case"), all four signs $+$ | AGREE (dup also gives the intermediate line) |
| 26 | R20 | R21 | 11 | $\sin\sin\sin$: $\tfrac14(-\sin(1{+}2{+}3) + \sin(1{+}2{-}3) + \sin(1{-}2{+}3) - \sin(1{-}2{-}3))$ | MINOR — primary commits to "$-$" for the two over-written signs (both *medium*); dup transcribes them as "$[\mp]$" and states that the required reading is the primary's. Same resolved formula. |
| 27 | R21 | R22 | 11 | $\cos\sin\sin$: $\tfrac14(-\cos(1{+}2{+}3) + \cos(1{+}2{-}3) + \cos(1{-}2{+}3) - \cos(1{-}2{-}3))$ | AGREE (dup also gives two intermediate lines) |

---

## 2. Disagreements

All three disagreements concern one glyph question: whether the order written on the
"$\tfrac52$" Bessel functions in the *second* product of the simplified source term carries a
"$+b$" at particular locations on pp.4 and 6. The primary reads "$\tfrac52+b$" everywhere; the
duplicate reads a bare "$\tfrac52$" at four specific places and explicitly says so (its Q5),
while reading "$\tfrac52+b$" everywhere else. The two files agree that on p.5 (both displays) and
on p.7 (all four occurrences, including the final formula) the order is $\tfrac52+b$.

For reference, both files read the p.7 final formula (primary R14 / dup R15) identically as
$$
-\,\frac{a_0\pi}{2}\,\frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2
(q\,r\,c_s^2\,\eta)^{-\frac12-b}
\Bigl\{ Y_{b+\frac12}(k\eta)\!\int_{\eta_{\rm init}}^{\eta}\! d\eta'\,(\eta')^{\frac12-b}J_{b+\frac12}(k\eta')\,S(\eta')
- J_{b+\frac12}(k\eta)\!\int_{\eta_{\rm init}}^{\eta}\! d\eta'\,(\eta')^{\frac12-b}Y_{b+\frac12}(k\eta')\,S(\eta') \Bigr\},
$$
with $S(\eta') = J_{\frac12+b}(qc_s\eta')J_{\frac12+b}(rc_s\eta') + \frac{2+b}{1+b}J_{\frac52+b}(qc_s\eta')J_{\frac52+b}(rc_s\eta')$.
No disagreement touches it.

### D1 — p.4, expanded form of $f$ (row 11; primary R11 first form / dup R10 expanded)

Primary (confidence *high* for this display except at C3–C5, none of which is this factor):
$$
+ \tfrac{(q c_s\eta)(r c_s\eta)}{(2+b)(1+b)}\, J_{\frac52+b}(q c_s\eta)\, J_{\frac52+b}(r c_s\eta)
$$
Duplicate (confidence *medium* on this factor; "written $J_{\frac52}(rc_s\eta)$ with **no** '$+b$'
… almost certainly an omission — the same factor carries $+b$ on p.5 — but transcribed as written"):
$$
+ \frac{(qc_s\eta)(rc_s\eta)}{(2+b)(1+b)}\,J_{\frac52+b}(qc_s\eta)\,J_{\frac52}(rc_s\eta)
$$
Differing glyph: the order of the last Bessel factor (argument $rc_s\eta$), $\tfrac52+b$ vs
$\tfrac52$. All other terms of the display agree.

### D2 — p.6 top, boxed final form of the simplified source (row 16; primary R11 final / dup R12 p.6)

Primary (confidence *high*: "orders $\tfrac12+b$ and $\tfrac52+b$ … are all clearly written"):
$$
f = \frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2
(q c_s\eta)^{-\frac12-b}\,(r c_s\eta)^{-\frac12-b}
\Bigl\{ J_{\frac12+b}(q c_s\eta)\,J_{\frac12+b}(r c_s\eta)
+ \frac{2+b}{1+b}\,J_{\frac52+b}(q c_s\eta)\,J_{\frac52+b}(r c_s\eta) \Bigr\}
$$
Duplicate (confidence *medium* for the second product: "on p.6 (both the top line and the copy at
the bottom of the page) they are written $J_{\frac52}$ with **no** '$+b$', whereas on p.5
(immediately above) and on p.7 (all copies) the same factors are written $J_{\frac52+b}$"):
$$
f = \frac{2^{3+2b}}{(3+2b)(2+b)}\,\Gamma\!\left(\tfrac52+b\right)^2 (qc_s\eta)^{-\frac12-b}(rc_s\eta)^{-\frac12-b}
\Big\{ J_{\frac12+b}(qc_s\eta)\,J_{\frac12+b}(rc_s\eta) + \frac{2+b}{1+b}\,J_{\frac52}(qc_s\eta)\,J_{\frac52}(rc_s\eta)\Big\}
$$
Differing glyphs: the orders of *both* Bessel functions in the second product, $\tfrac52+b$ vs
$\tfrac52$. Prefactor, exponents, first product and the ratio $(2+b)/(1+b)$ agree.

This is the one disagreement on a boxed, named result. Internal-consistency note (not a
physics judgement): the duplicate's reading makes the p.6 boxed line differ from the p.5 line it
is derived from and from the p.7 lines derived from it, and the duplicate says so itself; the
primary's reading is uniform across pp.4–7. Either way the p.7 final formula is agreed.

### D3 — p.6 foot, integral in $\eta$ before substituting the Green's function (row 18; primary R13 p.6 / dup R14 p.6)

Primary:
$$
\Bigl\{ J_{\frac12+b}(q c_s\eta)J_{\frac12+b}(r c_s\eta) + \tfrac{2+b}{1+b} J_{\frac52+b}(q c_s\eta)J_{\frac52+b}(r c_s\eta)\Bigr\}
$$
Duplicate ("the last order is again $\frac52$ without $+b$ — Q5"):
$$
\Big\{J_{\frac12+b}(qc_s\eta)J_{\frac12+b}(rc_s\eta) + \frac{2+b}{1+b}J_{\frac52+b}(qc_s\eta)J_{\frac52}(rc_s\eta)\Big\}
$$
Differing glyph: the order of the last Bessel factor (argument $rc_s\eta$), $\tfrac52+b$ vs
$\tfrac52$. Both files agree that the Bessel arguments in this brace are written with *unprimed*
$\eta$ while the power-law factors on the same line carry $\eta'$ (primary Q5, dup Q6), and both
agree the p.7 rewrite is fully primed.

Neither transcriber flagged the primary's reading as anything but *high*; only the duplicate
flagged its own reading (*medium*). The author needs to look at four spots: p.4 last factor of the
expanded display; p.6 top, both factors of the second product; p.6 foot, last factor.

---

## 3. Coverage gaps

No result is present only in the primary. Three displays are transcribed as results only in the
duplicate; the primary knows all three exist (it describes them in its corrections C3, C4, C5) but,
per README §4, treated them as intermediate algebra.

| Row | Only in | Page | What | Does the omission matter? |
|---|---|---|---|---|
| 10 | Duplicate (R10 first display) | 4 | $f$ with $dT/dx$ substituted, before expansion, $\frac{1+b}{2+b}[\cdots][\cdots]$ form | No. Pure intermediate; the next display (row 11, in both files) is what the derivation continues from. It is, however, the display where the coefficient correction C3 and the over-written order C4 live, so it is useful context for those. |
| 12 | Duplicate (R11 first display) | 5 | Four-term form with $\frac{3+2b}{2+b}$ pulled out | No. Intermediate between row 11 and row 13, both of which are in both files. |
| 15 | Duplicate (R12 p.5 display) | 5 (foot) | "so $f=$" line with $\frac{qc_s\eta}{3+2b}J_{1/2+b}$ factors and $(\frac{3+2b}{1+b}-1)$ | Marginal. It is the step immediately before the boxed p.6 result (row 16) and is the last place before p.7 where both transcribers agree the order is $\tfrac52+b$ — so it is relevant evidence for the D2 question. The primary records the struck glyphs on this line (C5) without transcribing the line itself. |

Smaller items present in one file inside an otherwise aligned row (not counted as rows):

- Primary only: $3w = \frac{1-b}{1+b}$ (row 7); $f''/f + \frac1x f'/f = \frac{1}{4x^2}$ (row 21);
  the explicit LHS of the final formula (row 20); correction C8 (the $\tilde y\to y$ relabelling
  on p.8, which the dup mentions inline in R16 but does not list as a correction).
- Duplicate only: the p.2 intermediate integration line
  $\frac{a^{1/2+3w/2}}{1/2+3w/2} = \sqrt{\rho_0/3M_P^2}\,a_0^{-1/2+3w/2}(\eta-\eta')$ quoted as a
  formula (the primary quotes the same line inside C2); the $\beta''$ line on p.9 (row 22); the
  intermediate two-factor lines of the trig identities on pp.10–11 (rows 25–27).

None of these affects any later step.

---

## 4. Conventions

Compared item by item against README §5. **No conflicts.** Every item that one file states, the
other either states identically or does not contradict.

| §5 item | Primary | Duplicate | Status |
|---|---|---|---|
| Time variable | $z$ (numerical, dummy $z'$); $\eta$ (analytic, dummy $\eta'$); "in $\tau$" on p.1 is textual only; $t$ via $dt = a\,d\eta$ | Same; additionally records the limit flip $\int_z^{z_{\rm init}}\to\int_{\eta_{\rm init}}^{\eta}$ | Consistent |
| Prime | $d/d\eta$ on $T$ and $a$ (inferred for $T$ via $T'/\mathcal H$ and $T_q' = qc_s\,dT/dx$); $d/dx$ in Step 4 and p.10; on $z',\eta'$ marks the dummy variable | $d/d\eta$ on $a$ (explicit), on $T$ inferred via $T'/\mathcal H \to \frac{x}{1+b}dT/dx$; $d/dx$ on pp.7–9 (inferred) | Consistent; both call the $T'$ meaning inferred. Primary alone notes the dummy-variable prime and the looped-script glyph for $\mathcal H$. |
| $\mathcal H$ vs $H$ | Both used; $H=\dot a/a$, $\mathcal H = a'/a$; $\mathcal H = aH$ via the $dz$ line; no $\epsilon$ | Same; adds $\mathcal H^2/a^2 = \rho_0/(3M_P^2)$ (p.3) | Consistent |
| Scale factor | $a_0$ explicit; $1+z = a_0/a$; $a_0$ survives to p.7; $a_0=1$ not assumed | Same; adds that on p.2 $\rho_0$ is the density at $a=a_0$ | Consistent |
| Green's function | Source $-\delta(z-z')$; first slot response, second source; $G_{\rm me} = -H(z')G_{\rm them}$; $G_{\rm them}$ retarded; $G_{\rm them}$'s own sign/normalisation *not stated* | Same, minus the explicit remark that $G_{\rm them}$'s conventions are unstated | Consistent; the primary's caveat is additional, not conflicting |
| $w$, $b$, $c_s$ | $b = \frac{1-3w}{1+3w}$ (p.3); derived identities; $c_s^2$ formula *not stated*; $\rho_0,p_0$ usage; $w_0$ slip | Same; $c_s$ "not defined in this document" | Consistent |
| Planck mass | Reduced, inferred from $3H^2M_P^2=\rho$ | Same (stated under Momentum labels) | Consistent |
| Fourier / $P(k)$ | None stated | None stated, "not inferable" | Consistent |
| Transfer function | Explicit $T_k$ formula; field not identified; $T_k\to1$ inferred | Same | Consistent |
| Momentum labels | $\mathbf k$, $\mathbf q$, $\mathbf k-\mathbf q$; $r=\lvert\mathbf k-\mathbf q\rvert$ inferred, never written; no angular reduction | Same | Consistent |
| Liouville–Green representation | Separate convention item contrasting NUM 06 p.10 ($Z_i\to(2/\pi)^{1/2}(x_i\beta_i)^{-1/2}\cos\gamma_i$, $\sin\gamma$ for Neumann) with NUM 07; infers $\beta_i = \gamma_i'$ | Not a convention item; same p.10 form given in R18 with $\beta_i,\gamma_i$ "not defined"; notes that this $\beta$ is *not* the $\beta = \tfrac12\ln\vartheta'$ of p.9 | Consistent. The dup's symbol-clash remark ($\beta$ on p.9 vs p.10) is not in the primary; the primary's $\beta_i=\gamma_i'$ inference relies on NUM 07 (out of scope). |

---

## 5. Corrections and open questions raised by one transcriber only

### Corrections (§5 of each file)

The two lists cover the same page marks with different numbering: primary C1–C7, C9–C12 ↔ dup
C1–C12 (primary C5 = dup C5 + C6; primary C6, C7 = dup C7, C8). Raised by one side only:

- **Primary C8** (p.8): the relabelling "now redefine $\tilde y\to y$" recorded as a correction.
  Dup mentions it inline in R16. Trivial.
- **Dup C1 wording**: the struck glyph in $G_{\rm them}$ on p.1 "looks like a stray '+' or 'Y'";
  primary says "most likely a 'Y' begun in error". Same resolution (plain product), but the dup
  rates its R3 *medium* on this point and the primary *high*.
- **Dup C10/C11** decline to say which way the over-written signs were corrected; **primary
  C10/C11** read both as "$-$". Both files' resolved formulas coincide (row 26).

### Open questions (§6 of each file)

Shared (differently numbered): $\tau$ vs $\eta$ on p.1 (P-Q2 / D-Q9); $\rho_0$ and $w_0$ on p.3
(P-Q3 / D-Q2, D-Q3); missing $1/x$ on p.4 (P-Q4 / D-Q4); unprimed $\eta$ on p.6 (P-Q5 / D-Q6);
$\beta_i,\gamma_i$ undefined on p.10 (P-Q7 / D-Q7); the p.11 signs (P-Q8 / D-Q8).

Raised by the primary only:

- **P-Q1**: $G_{\rm them}$'s source sign, normalisation and literature origin are not stated;
  whether $-\delta(z-z')$ is meant w.r.t. the $dz$ measure is not spelt out.
- **P-Q6**: Step 4 is headed "form suitable for Levin integration" but never writes the Levin form
  of the p.7 integral; $\omega_{\rm eff}^2$ is used on p.9 without definition. (Dup notes the
  $\omega_{\rm eff}^2$ inference inline in R16 but raises no question about the missing Levin form.)
- **P-Q8, second half**: the $\sin\sin\sin$ and $\cos\sin\sin$ identities (pp.11) are not needed
  for the p.7 integrand ($JJJ$ and $YJJ$ only); the page gives no indication what they are for.
- **P-Q7, second half**: the $x_i$ in the p.10 integral are unlabelled, and the weight
  $(\eta')^{1/2-b}$ is not absorbed into the Liouville–Green amplitude on the page.

Raised by the duplicate only:

- **D-Q5**: the $J_{5/2}$ vs $J_{5/2+b}$ observation — the source of all three DISAGREE rows.
- **D-Q1**: $c_s$ never defined in the document (the primary states the same fact under
  Conventions rather than as an open question).
- **D-Q10**: a sign-check of the p.6 → p.7 route, concluding the written $-a_0\pi/2$ prefactor is
  consistent with $G_{\rm me} = -HG_{\rm them}$ and the limit flip. Recorded as a check, not a
  question; the primary's R13 commentary makes the same two points (cancellation of $H$, sign from
  reversing limits) without setting it out.
- **Dup R5**: alternative reading "$P_0/P_0$" for the small stacked fraction in the source
  prefactor (both files read $p_0/\rho_0$).
- **Dup R6**: the "$(\eta-\eta')$" on the p.2 integration line is "presumably an integration
  constant". The primary quotes the same line in C2 without comment.
- **Dup R18**: the $\beta$ of p.10 is a different symbol from the $\beta = \tfrac12\ln\vartheta'$
  of p.9.

---

## 6. Review queue for this pair

Most consequential first. Page numbers are PDF positions (pp.10–11 are unnumbered on the page).

1. **p.6, top display (boxed "so $f=$")** — second Bessel product: is the order $\tfrac52$ or
   $\tfrac52+b$ on each factor? Primary reads $+b$ (*high*), dup reads no $+b$ on both (*medium*).
   Also check the same product on **p.6, foot display** (dup: last factor only lacks $+b$) and on
   **p.4, last term of the expanded display** (dup: last factor lacks $+b$). This settles D1–D3
   together. Both files agree p.5 and p.7 carry $+b$.
2. **p.7, final display** — confirm the two cross-outs both transcribers resolved the same way:
   (a) the struck mark after $\eta$ in $(q\,r\,c_s^2\,\eta)^{-1/2-b}$ is a struck prime, leaving
   unprimed $\eta$; (b) in the second integral the kernel is $Y_{b+1/2}(k\eta')$ written over a
   struck $J$. Dup rated both *medium*; primary *high*. Agreed reading, but it is the final formula.
3. **p.1, $G_{\rm them}$** — the struck glyph between $Y_{b+1/2}(k\eta')$ and $J_{b+1/2}(k\eta)$:
   confirm plain product (dup *medium*, primary *high*). Feeds the sign structure of the p.7 braces,
   though both agree the p.7 rewrite is clean.
4. **p.1, source prefactor denominator** — $\rho_0(1+p_0/\rho_0)$ vs the dup's alternative
   "$P_0/P_0$". Both *medium*. Low impact: p.3 rewrites it as $\rho_0(1+w)$ in both files.
5. **p.3** — the "$w_0$" subscript and the use of $\rho_0$ for the density at time $\eta$. Both
   files record it identically; author to confirm it is a slip with no downstream effect.
6. **p.4, $dT/dx$ intermediate brace** — absent $1/x$ on $(\tfrac32+b)J_{3/2+b}$. Both files agree
   it is absent on the page and that the final line is consistent with the identity; confirm only.
7. **p.11, $\sin\sin\sin$ expansion** — the two over-written signs (before $\gamma_2$ in the second
   line's second term; before $\gamma_3$ in the final line's second term). Both files *medium*;
   both arrive at "$-$". Not used by the p.7 integrand per the primary's Q8.
8. **p.2, integration line** — the "$(\eta-\eta')$" constant of integration (dup remark only).
   Background algebra; no downstream use.
