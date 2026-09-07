# Diff 03 — `03-source-term.md` (primary) vs `03-source-term-B.md` (duplicate), `NUM` 03 only

Comparison agent, 2026-09-07. Scope: the `NUM` 03 content only (primary R12–R35; duplicate R1–R28).
Primary R1–R11 (`MAIN` 10) have no counterpart in the duplicate and are ignored. Neither the PDF nor
the code was consulted; this report compares the two transcriptions against each other only.

Throughout, "P" = primary, "D" = duplicate.

**Global notational differences (not counted per row).** The two transcribers made different
rendering choices that apply uniformly and are mathematically immaterial:

| Item | P | D |
|---|---|---|
| Star on the initial-time equation of state | $w^*$ (superscript) | $w_*$ (subscript) |
| Green's function symbol | $G_k$, $\bar G_k$, $G^\chi_k$ | $\mathrm{Gr}_{\mathbf k}$, $\overline{\mathrm{Gr}}_{\mathbf k}$, $\mathrm{Gr}^\chi_{\mathbf k}$ — D states the author's glyph is literally "Gr" |
| Polarisation-tensor indices | $e_s^{\ell m}$ | $e^{lm}_s$ |
| Azimuthal angle | $\varphi$ (P notes the page writes $\phi$) | $\phi$ (as on the page) |
| Momentum subscripts on $\phi$, $T$ | $\phi_{\mathbf q}$, $T_q$, $T_{k-q}$ | $\phi_{\mathbf q}$, $T_{\mathbf q}$, $T_{\mathbf k-\mathbf q}$ (bold vector labels) |
| Infinitesimal in the jump condition | $\epsilon$ (P notes the clash with the slow-roll $\epsilon$) | $\varepsilon$ |

Both agree on the superscript star on $\zeta^*$ and $P^*$, $\mathcal P^*$. Whether the star on $w$ is a
superscript or subscript on the page is not resolvable from the two files (see §8, low priority).

---

## 1. Alignment table

Rows are keyed to the primary's results; where the duplicate splits one result into two, both D
numbers are given. Verdicts ignore the global notational differences listed above.

| P | D | Page | Description | Verdict |
|---|---|---|---|---|
| R12 | R1 | 1 | Starting sourced tensor equation in $\eta$, quoted from `MAIN` 11 p.3: $-\int\frac{d^3q\,d^3r}{(2\pi)^3}\delta(\mathbf k-\mathbf q-\mathbf r)\,e_s^{\ell m}\{4q_\ell r_m\phi_{\mathbf q}\phi_{\mathbf r} + \frac{8M_P^2}{\rho_0+p_0}\frac{q_\ell r_m}{a^2}(\phi'+\mathcal H\phi)_{\mathbf q}(\phi'+\mathcal H\phi)_{\mathbf r}\}$ | AGREE |
| R13 | R2 | 1 | $3H^2M_P^2=\rho_0$; $d/d\eta = -(1+z)aH\,d/dz$ | AGREE |
| R14 | R3, R4 | 1–2 | $h_s=\chi_s/a$; $\chi_s'' + (k^2-a''/a)\chi_s = a\times$source (D also shows the $\{2\mathcal H^2 - a''/a - 2\mathcal H^2\}$ bracket) | AGREE |
| R15 | R5 | 2 | $e_s^{\ell m}q_\ell r_m \to -e_s^{\ell m}q_\ell q_m$ since $e$ annihilates $\mathbf k$; sign converts $-\int$ to $+4a\int$ | AGREE |
| R16 | — | 2–3 | $\phi'+\mathcal H\phi \to aH[\phi-(1+z)\,d\phi/dz]$ | ONLY-IN-PRIMARY |
| R17 | R6 | 3 | $a''/a = \frac{d}{dt}(a^2H) = 2a^2H^2 + a^2\dot H = (aH)^2(2-\epsilon)$ (D shows two extra intermediate steps) | AGREE |
| R18 | R7 | 3 | Redshift equation before dividing: $(1+z)^2(aH)^2\chi_s'' + (1+z)(aH)^2\epsilon\,\chi_s' + (k^2-(aH)^2(2-\epsilon))\chi_s = 4a\int\frac{d^3q}{(2\pi)^3}e q q\{\phi\phi + \frac{2}{3(1+w_0)}(\dots)(\dots)\}$ | AGREE |
| R19 | R8 | 4 | Divided by $(1+z)^2(aH)^2$; RHS $\frac{4}{a}\frac{1}{H^2}\int\frac{d^3q}{(2\pi)^3}\frac{1}{(1+z)^2}\dots$ | AGREE |
| R20 | R9 | 4 | $\phi_{\mathbf k} = \frac{3(1+w^*)}{5+3w^*}T_k(z)\zeta^*_{\mathbf k}$, "$w^*$ is the fixed value of $w$ at the initial time" | AGREE (P: medium on $\zeta$ glyph; D: high, glyph noted as "5*/S*"-like) |
| R21 | R10 | 4 | $Q_s \equiv e_s^{\ell m}(\mathbf k)q_\ell q_m$; D's R10 also displays the $k_{\rm phys}$ identification, which P gives in R22 prose and §2.2 | AGREE |
| R22 | R11 | 4 | **Red-boxed source term** with transfer functions (see §2a) | AGREE |
| R23 | R12 | 5 | $\chi_s(z)$ Green's-function solution, $\int_{z_{\rm init}}^{z}dz'\,G^\chi_k\frac{1}{a(z')H(z')^2(1+z')^2}\int\dots f(z'\,\vert\,\mathbf k,\mathbf k-\mathbf q)$ | AGREE |
| R24 | R13 | 5 | $h_s=\chi_s/a$ with $a_0$ made explicit, then $\frac{1+z}{1+z'}$ and $Q_s/a_0^2$ | AGREE |
| R25 | R14, R15 | 5–6 | Green's-function defining equation ($+\delta(z-z')$, $k^2/H^2$ without "phys"); $\bar G_k\equiv -G_k$; jump $=-1$; $\bar G=d\bar G/dz=0$ for $z>z'$; $d\bar G/dz\vert_{z=z'}=+1$ | AGREE (D adds a medium flag: the p.6 jump working is written with an *unbarred* symbol; P renders it barred without comment) |
| R26 | R16 | 6 | $h_s(z) = 36(\frac{1+w^*}{5+3w^*})^2\int_z^{z_{\rm init}}dz'\,\bar G_k\frac{1+z}{1+z'}\int\frac{d^3q}{(2\pi)^3}\frac{Q_s}{a_0^2}\zeta^*\zeta^*\frac{f(z'\vert\mathbf q,\mathbf k-\mathbf q)}{H^2(z')}$ | AGREE |
| R27 | R17 | 6 | Dimension check: $[M^3][M^2][M^{-6}][M^{-2}] = [M^{-3}]$ ✓ | AGREE |
| R28 | R18, R19 | 7 | **$I_s$ definition** (see §2b) and $h_s(\mathbf k) = 36(\dots)^2\int\frac{d^3q}{(2\pi)^3}\zeta^*\zeta^* I$ | MINOR (P writes $I_s(z\vert\mathbf k,\mathbf q)$ in the $h_s$ formula; D writes $I(z\vert\mathbf k,\mathbf q)$ without the $s$, and lists this among the as-written variants) |
| R29 | R20 | 7 | $a_0$-dependence remarks; $Q_s/(a_0^2H^2)\propto q_{\rm phys}^2/H^2$ | AGREE |
| R30 | R21, R22 | 7–8 | Wick contraction, three lines, coefficient 1024, two $\delta\delta$ terms, then $(2\pi)^3\delta(\mathbf k+\mathbf k')\,1024\,(\dots)^4\int P^*P^*(I_sI_{s'}(\cdot,-\mathbf q)+I_sI_{s'}(\cdot,\mathbf k'+\mathbf q))$ | AGREE (D's R22 writes the p.8 LHS as $\langle h_sh_s\rangle$; P's chain has a single LHS $\langle h_sh_{s'}\rangle$ on p.7) |
| R31 | R23 | 8 | Symmetry reduction to $2I_s(\mathbf q,\mathbf k-\mathbf q)^2$ with the two quoted justifications (D shows one extra intermediate line with $\mathbf k'+\mathbf q$) | AGREE |
| R32 | R24 | 9 top | $(2\pi)^3\delta\,2048(\dots)^4\int\frac{d^3q}{(2\pi)^3}P^*P^*I_s^2$; exponent 4 over struck 2; annotation "should be 2592" | AGREE (P high, D medium on the exponent) |
| R33 | R25 | 9 | $Q_\pm = \frac{1}{\sqrt2}q_{\rm phys}^2\sin^2\theta\{\cos2\varphi,\sin2\varphi\}$; $\int Q_+Q_-=0$; $\int\cos^2 2\varphi = \int\sin^2 2\varphi = \pi$ | AGREE |
| R34 | R26, R27 | 9 mid, 9 bottom | $1024(\dots)^4\pi\int\frac{d^3q}{(2\pi)^3}P^*P^*\{\dots\sin^2\theta\dots\}^2$ and the $q^2dq\sin\theta\frac{4\pi^4}{(2\pi)^3}\frac{\mathcal P^*(q)}{q^3}\frac{\mathcal P^*(r)}{r^3}\sin^4\theta\{\dots\}^2$ form | AGREE on the results (the reading of a struck factor on this page differs; see §4) |
| R35 | R28 | 10 | **Final formula** $512\pi^2$ (see §2c) | AGREE |

**Counts (24 aligned results):** AGREE 22 · MINOR 1 · DISAGREE 0 · ONLY-IN-PRIMARY 1 · ONLY-IN-DUPLICATE 0.

One further disagreement concerns a *struck-through* (non-result-bearing) item on p.9 and is
recorded in §4 rather than counted above.

---

## 2. Key formulas check

### (a) Red-boxed source term, p.4 (P R22 / D R11)

Glyph-by-glyph comparison:

| Element | P | D | |
|---|---|---|---|
| LHS operator | $\frac{d^2\chi_s}{dz^2} + \frac{\epsilon}{1+z}\frac{d\chi_s}{dz} + \big(\frac{k^2_{\rm phys}}{H^2} + \frac{\epsilon-2}{(1+z)^2}\big)\chi_s$ | same | agree |
| RHS prefactor | $\frac{1}{a}\frac{1}{H^2}\frac{1}{(1+z)^2}$ | same | agree |
| Measure | $\int\frac{d^3q}{(2\pi)^3}$ | same | agree |
| Projection | $Q_s(\mathbf q)$ | $Q_s(\mathbf q)$ | agree |
| Constant | $\frac{36(1+w^*)^2}{(5+3w^*)^2}$ | $\frac{36(1+w_*)^2}{(5+3w_*)^2}$ | agree (star position only) |
| Primordial fields | $\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}$ | same | agree |
| Bracket, term 1 | $T_qT_{k-q}$ | $T_{\mathbf q}T_{\mathbf k-\mathbf q}$ | agree |
| Bracket, coefficient | $\frac{2}{3(1+w_0)}$ | $\frac{2}{3(1+w_0)}$ | agree |
| Bracket, term 2 | $\big(T-(1+z)\frac{dT}{dz}\big)_q\big(T-(1+z)\frac{dT}{dz}\big)_{k-q}$ | $\big(T-(1+z)\frac{dT}{dz}\big)_{\mathbf q}\big(T-(1+z)\frac{dT}{dz}\big)_{\mathbf k-\mathbf q}$ | agree |
| Struck leading "4" | recorded; $36=4\times9$ | recorded; $36 = 4\times 9$ | agree |
| Origin of the 2 | over-written "8" → "2" on pp.2–3 | small "2" above "8" on pp.2–3 | agree |
| Identification with $f$ | P writes out $f(z\vert\mathbf q,\mathbf k-\mathbf q)\equiv\{\dots\}$ as a display and flags it as an inference (never written as an equation on the page) | D says in prose "the boxed braces are what the document subsequently calls $f$" and "R12 is the implicit definition of $f$" | agree |

**Verdict: AGREE.** No glyph difference. Both read the red box around the braces; P classifies it as
a *typed* red box (grouped with the July 2025 annotations, "no text"), D says "boxed in red by the
author" without saying whether it is ink or typed — see §3.

### (b) $I_s$ time integral, p.7 (P R28 / D R18)

| Element | P | D | |
|---|---|---|---|
| LHS | $I_s(z\,\vert\,\mathbf k,\mathbf q)$ | $I_s$ (no argument list; D notes the list varies across pp.7–9) | notation |
| Limits | $\int_z^{z_{\rm init}}dz'$ | same | agree |
| Green's function | $\bar G_k(z,z')$ | $\overline{\mathrm{Gr}}_{\mathbf k}(z,z')$ | agree (glyph rendering) |
| Redshift ratio | $\frac{1+z}{1+z'}$ | same | agree |
| Projection / denominator | $\frac{Q_s(\mathbf k,\mathbf q)}{a_0^2H^2(z')}$ | same | agree; **both flag medium** on the denominator (P: "small mark after $H^2$ could be a subscript 0"; D: "ink blot on/under the $H$ — could be $H^2(z')$ or a struck subscript") |
| Source function | $f(z'\,\vert\,\mathbf k,\mathbf k-\mathbf q)$ | same | agree; both note $\mathbf k$ as first argument on pp.5, 7 vs $\mathbf q$ on pp.6, 9, 10 |

**Verdict: AGREE.** The companion formula $h_s(\mathbf k) = 36(\frac{1+w^*}{5+3w^*})^2\int\frac{d^3q}{(2\pi)^3}\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}I(\dots)$
agrees apart from the $I_s$/$I$ subscript (MINOR, §1).

### (c) Final $\langle hh\rangle$ formula, p.10 (P R35 / D R28)

| Element | P | D | |
|---|---|---|---|
| LHS | $\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle$ | same | agree |
| Delta and polarisation | $(2\pi)^3\delta(\mathbf k+\mathbf k')\,\delta_{ss'}$ | same | agree |
| Numerical prefactor | $512\pi^2$ | $512\pi^2$ | agree |
| EoS power | $\big(\frac{1+w^*}{5+3w^*}\big)^4$ | $\big(\frac{1+w_*}{5+3w_*}\big)^4$ | agree |
| Measure | $\int\frac{dq}{q}$, no $d\theta$ | same, no $d\theta$ | agree; both record the missing $d\theta$ |
| Angular factor | $\sin^5\theta$ | $\sin^5\theta$ | agree |
| Spectra | $\mathcal P^*(q)\,\frac{\mathcal P^*(r)}{r^3}$ | same ($r^{-3}$ under $\mathcal P^*(r)$ only; $q^{-3}$ absorbed into $dq/q$) | agree |
| Time integral | $\big\{\int_z^{z_{\rm init}}dz'\,G_k(z,z')\frac{1+z}{1+z'}\frac{q^2_{\rm phys}}{H(z')^2}f(z\vert\mathbf q,\mathbf k-\mathbf q)\big\}^2$ | same | agree; both read the Green's function *unbarred* on p.10 (barred on p.9), both read $f(z\vert\dots)$ with an unprimed $z$ |
| Red annotation | "512" boxed, "DS 10 July 2025 should be 646" | same | agree |
| Confidence | P: high for coefficients/structure, medium for (i) bar, (ii) $d\theta$, (iii) $z$ vs $z'$ | D: high for glyphs, with the same three items recorded "as written" and raised as Q6–Q8 | same content, different flagging style |

**Verdict: AGREE.** No difference in any glyph. Both transcribers independently read the same three
irregularities (unbarred $G$, unprimed $z$ in $f$, no $d\theta$), which makes it likely they are
genuinely on the page rather than misreadings.

---

## 3. Red annotations ("DS 10 July 2025")

| Page / location | Blue-ink original | Red correction | P | D | Agree? |
|---|---|---|---|---|---|
| p.7, Wick contraction (first and second lines) | 1024 | "Looks like this should be 1296 = 36^2, not 1024" | recorded (R30, §5) | recorded (R21, §5.8) | yes |
| p.8, after $\mathbf t$ integration | 1024 | red box, no text | recorded (§5) | recorded (R22, §5.8) | yes |
| p.9 top, single $I_s^2$ form | 2048 | "should be 2592" | recorded (R32, §5) | recorded (R24, §5.8) | yes |
| p.9 middle, after $\varphi$ integral | 1024 (in $1024(\dots)^4\pi$) | "should be 1292" (sic) | recorded (R34, §5) | recorded (R26, §5.8) | yes |
| p.9 bottom, dimensionless-spectrum form | 1024 (in $1024\pi$) | red box, no text | recorded (§5) | recorded (§5.8) | yes |
| p.10, final formula | 512 (in $512\pi^2$) | "should be 646" (sic) | recorded (R35, §5) | recorded (R28, §5.8) | yes |
| p.4, curly bracket of the source term | — | red box, no text | recorded as a *typed* red box (§5, grouped with the 2025 annotations; undated) | recorded as "boxed in red by the author" (R11) — not listed among the typed annotations in D §5.8 | **classification differs** (see below) |

Both agree exactly on which numbers are blue-ink originals (1024, 1024, 2048, 1024, 1024, 512) and
which are red corrections (1296, —, 2592, "1292", —, "646"), and both transcribe "1292" and "646"
verbatim with a "(sic)". Both state that the blue text was not altered and that no corrected final
formula exists on the page. Both independently observe that "1292" and "646" do not follow from
1296/2592 by the page's own operations and look like typos for 1296 and 648 (P §5 and Open Q2; D
Q2), and both note that 1024 $=32^2$ where $36^2=1296$ would be expected (P Q2; D Q3).

The single difference is the **p.4 red box**: P treats it as a typed (later, undated) annotation; D
does not say whether it is typed or ink, nor whether it is contemporaneous with the Sep-2024 body.
Immaterial to the formulas but worth the author confirming, since it bears on whether the "$f$" box
was part of the original derivation or a 2025 highlight.

D's §1 says "four red typed annotations … on pp.7, 9, 10"; P's §1 says annotations on pp.7, 9, 10.
Both then list the two text-less boxes (p.8, p.9 bottom) in their §5, so the counts are consistent
(four with text, two without).

---

## 4. Disagreements

**No DISAGREE verdict among the 24 aligned results.** One reading disagreement exists on a
struck-through, non-result-bearing item:

### 4.1 p.9 middle — struck factor after $P^*(|\mathbf k-\mathbf q|)$ (P R34 / §5; D R26 / §5.6)

- P: "a struck '$q^4_{\rm phys}/2$' appears outside the braces on the first line"; P then explains
  the coefficient as "the factor $\tfrac12$ from $Q_\pm^2$ has been combined with 2048 to give 1024".
- D: "a factor after $P^*(|\mathbf k-\mathbf q|)$, read as $q^4_{\rm phys}/4$ (possibly $/2$; low
  confidence, it is struck), is crossed out".

Glyph that differs: the denominator of the struck factor, "2" (P) vs "4" (D). D flags **low**
confidence and names $/2$ as the alternative; P does not flag. The item is struck on the page and
neither transcriber uses it in any result; both agree that the live formula has $1024\pi$ and
$q^2_{\rm phys}$ inside the squared braces. Internal note: D's own R26 commentary says "the $1024\pi$
follows from $2048\times\tfrac12\times\pi$", which is the $\tfrac12$ reading rather than the
$\tfrac14$ it transcribes for the struck factor; D does not remark on this.

### 4.2 Confidence-flag differences on agreed readings (not disagreements)

| Location | Reading (both) | P flag | D flag |
|---|---|---|---|
| p.4 $\zeta^*$ glyph | $\zeta$ | medium (alternatives $S$, $\mathcal S$) | high (notation note: "resembles 5*/S*") |
| p.5, first argument of $f$ | $\mathbf k$ | high (raised in Q3) | medium (Q5) |
| p.6, overbar in jump-condition working | P renders barred; D reads unbarred | not flagged | medium (Q6) |
| p.7, $a_0^2H^2(z')$ | $H^2(z')$ | medium | medium |
| p.9 top, exponent 4 over struck 2 | 4 | high | medium |
| p.9 bottom / p.10, no $d\theta$; $f(z\vert\dots)$ unprimed | as written | medium | high for glyphs, raised as Q7/Q8 |
| p.10, unbarred $G_k$ | unbarred | medium | recorded as written, Q6 |

---

## 5. Coverage gaps

| Item | Present in | Assessment |
|---|---|---|
| P R16: $\phi'+\mathcal H\phi \to aH[\phi-(1+z)\,d\phi/dz]$ (pp.2–3) | P only | Intermediate step; D absorbs it into R7 (the bracket $(\phi-(1+z)d\phi/dz)$ appears there). Omission does not matter for the audit, but it is the step that fixes the relative sign inside the $T-(1+z)dT/dz$ bracket, so it is useful to have it isolated as P does. |
| D R4: the intermediate $\{2\mathcal H^2 - a''/a - 2\mathcal H^2\}$ bracket (p.2) | D display; P prose | P describes the same bracket in R14 prose. No gap. |
| D R10: display of $\frac{k^2}{H^2}\frac{1}{a_0^2}\frac{a_0^2}{a^2(1+z)^2}\equiv\frac{k^2_{\rm phys}}{H^2}$ (p.4) | D display; P prose (R22, §2.2) | Same content. No gap. |
| D R23 intermediate line with $I_{s'}(-\mathbf q,\mathbf k'+\mathbf q)$ before $\mathbf k'\to-\mathbf k$ (p.8) | D only | Intermediate; P goes straight to the $\mathbf q-\mathbf k$ form. Does not matter. |
| D §5.2: on p.4 the $\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q}$ and the first $\phi$ in the derivative bracket are struck with "T" written above (the $\phi\to T$ substitution) | D only | A cross-out P did not record. Harmless (both arrive at the same R22/R11), but it is a §5-format item P missed. |
| D R8 note: "$1/(1+z^2)$" sloppy bracketing on p.4 top, read as $(1+z)^2$ | D only | Harmless; both read $(1+z)^2$. |
| P §6.12: remark on the phrase "$Q^s$ is blind to $\mathbf k$" | P only | D quotes the phrase verbatim (R23) without raising it. Interpretive; does not affect formulas. |
| P §6.11: cross-document factor between `MAIN` 10 R11 ($4M_P^4/a^4$) and `NUM` 03 R12 ($8M_P^2/a^2$) | P only (out of D's scope) | For the orchestrator's cross-spec check; not a `NUM` 03 gap. |
| D §1: author's own Step 1–5 structure with page ranges | D only | Useful navigation aid; no formula content. |

Neither file omits a result-bearing formula that the other has. The numbered-result difference
(24 vs 28) is entirely due to D splitting P's R14, R25, R28, R30, R34 into pairs and P having the
extra R16.

---

## 6. Conventions (README §5), item by item

| Item | P §2.2 | D §2 | Verdict |
|---|---|---|---|
| Time variable | $\eta$ on p.1; $z$ from pp.1–4 onward; $t$ transiently pp.2–3; $z'$ = source-time integration variable from p.5 | same; $z'$ from Step 3 | agree |
| Prime | $d/d\eta$ on pp.1–3; $d/dz$ always written out; dot $=d/dt$ | $d/d\eta$ on pp.1–2; same otherwise | agree (page range trivially differs) |
| $\mathcal H$ vs $H$; $\epsilon$ | $\mathcal H=a'/a$ pp.1–2, $H$ cosmic; $\epsilon=-\dot H/H^2$ inferred from $a''/a=(aH)^2(2-\epsilon)$; P notes $\epsilon$ is reused as an infinitesimal on p.6 | same inference, also from the $d\chi_s/dz$ coefficient; D uses $\varepsilon$ for the infinitesimal | agree |
| Scale factor | $a_0$ explicit (pp.4, 5, 7); $a_0\ne1$; $k_{\rm phys}=k/a_0$ inferred from p.4 and p.7 | same; flagged medium (Q4) | agree; **both** say $k_{\rm phys}$ means comoving $k$ divided by $a_0$ (not $k/a(z)$), and both note the p.5 Green's-function operator writes bare $k^2/H^2$ |
| Green's function: source sign | $+\delta(z-z')$ for $G$ | $+\delta$ for $\mathrm{Gr}$ | agree |
| Green's function: arguments | first $=z$ response, second $=z'$ source | same | agree |
| Green's function: $\bar G=-G$ | introduced p.5 so the integral runs $\int_z^{z_{\rm init}}$ | same | agree |
| Green's function: boundary conditions | $\bar G=d\bar G/dz=0$ for $z>z'$; $d\bar G/dz\vert_{z=z'}=+1$ (for the **barred** function) | §2 bullet: "$\mathrm{Gr}_k(z,z')=d\mathrm{Gr}_k/dz=0$ for $z>z'$, giving $d\mathrm{Gr}_k/dz\vert_{z=z'}=+1$", then "the document then redefines $\overline{\mathrm{Gr}}=-\mathrm{Gr}$"; but D's **R15** attributes the same conditions and the $+1$ to $\overline{\mathrm{Gr}}$ | **conflict within D**: D §2 assigns the $+1$ slope to the unbarred function, D R15 (and P) to the barred one. D's Q6 acknowledges the p.6 working is written unbarred. The orchestrator should treat P/D-R15 as the reading and D §2 as a summarising slip, but the page must decide. |
| Literature Green's function | none stated | none stated | agree |
| Author's glyph for the Green's function | rendered $G$ | stated to be literally "Gr" | notation; D's report of the glyph is extra information |
| $w_0$ | $p_0/\rho_0$, background EoS "at the time appearing in the equation"; whether it means $w(z')$ inside the integral is *not stated* (Q5) | background at redshift $z$, "therefore $w(z)$ at the source redshift (inferred)"; Q10 says it "must be evaluated at $z'$" | compatible; D commits to an inference P leaves open |
| $w^*$ / $w_*$ | fixed value at the initial time (stated, p.4) | same | agree (star position is notation) |
| $b$, $c_s^2$ | not used | not used | agree |
| $\langle\zeta\zeta\rangle$ | $(2\pi)^3\delta(\mathbf q+\mathbf t)P^*(q)$, inferred | same, inferred | agree |
| $P$ vs $\mathcal P$ | $\mathcal P=q^3P/(2\pi^2)$ inferred from $4\pi^4$ on p.9 | $P=2\pi^2\mathcal P/q^3$, same inference | agree |
| Tensor two-point function | $(2\pi)^3\delta(\mathbf k+\mathbf k')\delta_{ss'}(\dots)$; no dimensionless tensor spectrum defined | same | agree |
| Polarisation normalisation | $e_s^{\ell m}e^{s'}_{\ell m}=\delta_{ss'}$ inferred from $Q_\pm$; P gives explicit $e_+,e_\times$ with $\mathbf k\parallel\hat z$ | same inference; no explicit tensors | agree |
| $h_s\leftrightarrow h_{ij}$ | not stated (from `MAIN` 11) | not addressed | agree (P more explicit) |
| Transfer function | of the Newtonian potential $\phi$ via $\phi=\frac{3(1+w^*)}{5+3w^*}T\zeta^*$; $T\to1$ inferred | same | agree |
| Momentum labels | $\mathbf k$, $\mathbf q$, $\mathbf r=\mathbf k-\mathbf q$, $\mathbf t$; $r=\vert\mathbf k-\mathbf q\vert$ | same | agree |
| Angles $\theta$, $\varphi$ | P states "$\theta$ is the angle between $\mathbf q$ and $\mathbf k$, $\varphi$ the azimuth about $\mathbf k$" without marking it inferred | D: "not defined on the page; inferred" | minor: D flags as inferred, P does not |
| Angular reduction | $\varphi$ integral done, $\theta$ left implicit; no $d\theta$ | same | agree |

**Conflicts:** one, internal to D (boundary-condition attribution in D §2 vs D R15). No conflict
between P and D on any convention.

---

## 7. Corrections and open questions raised by one transcriber only

Raised by **D** only:
1. p.4, third displayed equation: $\phi$'s struck with "T" written above (the $\phi\to T$ substitution) — D §5.2.
2. p.4 top: "$1/(1+z^2)$" sloppy bracketing, read as $(1+z)^2$ — D R8.
3. p.6: the jump-condition working is written with an unbarred Green's-function symbol although the sentence on p.5 announces the calculation is for the barred one — D R15, Q6.
4. p.9 top: exponent 4 over-written on a 2 flagged *medium* — D R24 (P records the cross-out but rates the reading high).
5. $\theta$, $\phi$ never defined on the page — D Q8.
6. $I_s$ argument-list variants $I(z\vert\mathbf k,\mathbf q)$, $I_s(z\vert\mathbf k,\mathbf q)$, $I_s(\mathbf q,\mathbf k-\mathbf q)$, $I_s(z\vert\mathbf q,\mathbf k-\mathbf q)$ all refer to one object — D Q9.
7. Explicit statement that $w_0$ inside $f$ "must be evaluated at $z'$" — D Q10 (P raises the question without answering it).
8. Whether $k$ in the p.5 Green's-function equation means $k_{\rm phys}$ — D Q4 (P notes the missing "phys" in R25 but does not pose it as a question).
9. p.10 ends with the final formula and the rest of the page is blank (no truncation) — D Q1.
10. PDF metadata: created 20 Sep 2024, modified 11 Jul 2025 — D §1.

Raised by **P** only:
1. p.4 red box classified as typed (later) — P §5.
2. Explicit written-out definition of $f$ as a display formula, flagged as an inference — P R22, Q5.
3. p.6 reuse of $\epsilon$ as an infinitesimal, distinct from the slow-roll $\epsilon$ — P §2.2.
4. Same symbol $Q_s$ / $Q_\pm$ used with comoving $q$ (p.4, $1/a_0^2$ carried separately) and with $q_{\rm phys}$ (p.9, $1/a_0^2$ absorbed) — P Q7 (D's §2 covers the same fact without listing it as open).
5. The phrase "$Q^s$ is blind to $\mathbf k$" — P Q12.
6. Cross-document factor $4M_P^4/a^4$ (`MAIN` 10) vs $8M_P^2/a^2$ (`NUM` 03 R12) — P Q11 (out of D's scope).
7. PDF forensics: 16 `%%EOF` markers explaining the README's 46-page over-count — P §1.

Raised by **both** (no action needed for the diff): page count 10 not 46; the four annotation values
and the "1292"/"646" inconsistency; $1024=32^2$ vs $36^2$; first argument of $f$ ($\mathbf k$ vs
$\mathbf q$) on pp.5, 7; unprimed $z$ in $f$ on pp.9–10; unbarred $G$ on p.10; missing $d\theta$;
$a_0^2H^2(z')$ blot on p.7.

---

## 8. Review queue for this pair (most consequential first)

1. **pp.7–10, numerical prefactor chain.** Blue ink 1024 → 2048 → $1024\pi$ → $512\pi^2$; red
   annotations 1296 → 2592 → "1292" → "646". Both transcribers read both chains identically and both
   note the last two annotated values do not follow from the first two by the page's own steps
   ($\times\tfrac{\pi}{2}$ then $\times\tfrac{\pi}{2}$ would give $1296\pi$ and $648\pi^2$). The
   author should state the intended final coefficient and, if possible, where the original 32 (in
   $32^2=1024$) came from, since it propagates into every result from p.7 onward.
2. **p.4, red-boxed source term.** Confirm the bracket
   $\{T_qT_{k-q} + \frac{2}{3(1+w_0)}(T-(1+z)\frac{dT}{dz})_q(T-(1+z)\frac{dT}{dz})_{k-q}\}$, the
   $36=4\times9$ with the struck leading 4, and the $2$ over-written on the $8$ (pp.2–3). Both
   transcribers agree exactly, so this is a confirmation read rather than a dispute. Also decide
   whether $w_0$ inside $f$ is $w(z')$ (P leaves open; D infers yes), and whether the red box is a
   2025 typed highlight or original.
3. **p.7, $I_s$ denominator $a_0^2H^2(z')$.** Ink blot; both transcribers medium. Confirm it is
   $H(z')$ and not a subscript.
4. **p.10 and p.9 bottom, three as-written irregularities in the final formula:** unbarred $G_k$
   (barred on p.9), $f(z\vert\dots)$ with unprimed $z$ inside the $dz'$ integral, and no $d\theta$.
   Both transcribers read all three identically; confirm intent.
5. **p.9 middle, struck factor** "$q^4_{\rm phys}/2$" (P) vs "$q^4_{\rm phys}/4$" (D, low
   confidence). The only glyph-level reading disagreement in the pair. Struck and unused, but the
   author should confirm the $\tfrac12$ from $Q_\pm^2$ that gives $2048\to1024$.
6. **pp.5–6, Green's-function bookkeeping.** (i) Whether the p.6 jump-condition working is for the
   barred or unbarred function (D flags; D's own §2 and R15 disagree with each other; P and D-R15
   agree the $+1$ slope belongs to $\bar G$). (ii) Whether $k^2/H^2$ in the p.5 defining equation
   means $k^2_{\rm phys}/H^2$ as in the p.4 operator. (iii) First argument of $f$: $\mathbf k$ on
   pp.5, 7 vs $\mathbf q$ on pp.6, 9, 10.
7. **p.9 top, exponent on $(\frac{1+w^*}{5+3w^*})$** over-written 2 → 4 (D medium, P high).
8. **p.4, $\zeta^*$ glyph** (P medium: "5"-like; D high), and the position of the star on $w$
   (P superscript, D subscript). Cosmetic; no mathematical consequence.
9. **p.4 top, "$1/(1+z^2)$" bracketing** (D) — cosmetic.
