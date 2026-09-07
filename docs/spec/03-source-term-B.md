# Spec 03-B: SIGW source term expressed in redshift (independent duplicate transcription)

> **Author note (2026-09-07).** This is the independent duplicate transcription and is kept as written. For the definitive project convention on the normalisation of $a$ (the code absorbs $a_0$ into $k/a_0$ and $a_0\eta$; it does **not** set $a_0 = 1$) and on which Green's function the code computes (the unit-jump $\bar G_k$ of `NUM` 03, $= -a_0H(z')\,{\rm Gr}_k$), see §0 of the primary spec `03-source-term.md`. Tier 1.1 of `REVIEW-QUEUE.md` was signed off on that basis. Tier 1.2 (the coefficient chain, R21–R28 here) was signed off 2026-09-07: the red annotations are right and the corrected chain is $1296 \to 1296 \to 2592 \to 1296\pi \to 648\pi^2$ ("1292" and "646" are typos); see §0.2 of the primary spec. Tier 1.3 was signed off 2026-09-07: R28 here, with $648\pi^2$, $\int_0^\infty dq/q\int_0^\pi d\theta$ and $r=\sqrt{k^2+q^2-2kq\cos\theta}$, is the build form, stored per polarisation; see §0.3 of the primary spec. Tier 1.4 was signed off 2026-09-07: the seed glyph is $\zeta^*$ and $P^* = P_\zeta$; see §0.4 of the primary spec.


Transcription of `NUM` 03 only, produced independently of `docs/spec/03-source-term.md` (README §3, §6 rule 6).

## 1. Source documents

| Doc | File | Pages | Date on first page |
|---|---|---|---|
| `NUM` 03 | `Numerical implementation/03 - 2024-09-20 - SIGW source term expressed in redshift.pdf` | **10** (A4; `pdfinfo` reports 10 pages, PDF created 20 Sep 2024, modified 11 Jul 2025) | 20 Sep 2024 |

Note: the README (§1.2) and the assignment both state 46 pages. The file actually contains 10 pages; see Open questions Q1.

Later annotations: four red typed annotations signed "DS 10 July 2025" on pp. 7, 9, 10, each boxing a numerical prefactor in red and stating what it "should be". These are recorded in §5 and reproduced verbatim at the corresponding results.

Structure of the document (author's own headings): Step 1 (sourced tensor equation, p.1), Step 2 (exchange $\eta$ for $z$, pp.1–4), Step 3 (Green's function solution, pp.5–6), Step 4 (check dimensions, p.6), Step 5 (convert to a power spectrum, pp.7–10).

The document opens by citing its starting equation as "(from ⑪, page 3)", i.e. `MAIN` 11 p.3.

## 2. Conventions in force

- **Time variable.** Step 1 is written in conformal time $\eta$ (NUM 03 p.1). From Step 2 the independent variable is redshift $z$, via $\dfrac{d}{d\eta} = -(1+z)\,aH\,\dfrac{d}{dz}$ (NUM 03 p.1). In Steps 3–5 a second redshift $z'$ appears as the *integration variable* of the Green's-function integral (source time); it is not a derivative.
- **Meaning of a prime.** On pp.1–2 a prime is $d/d\eta$ (on $h_s$, $\chi_s$, $a$, $\phi$). From p.2 onward all redshift derivatives are written out explicitly as $d/dz$; no prime means $d/dz$ anywhere in the document. Overdot is $d/dt$ (cosmic time), used on p.3 only ($\dot a$, $\dot H$).
- **$\mathcal{H}$ vs $H$.** Script $\mathcal{H}$ (conformal Hubble rate, $a'/a$) appears in the $\eta$-form equation on pp.1–2. $H$ is the cosmic Hubble rate from p.1 onward (in $d/d\eta = -(1+z)aH\,d/dz$ and in $3H^2 M_P^2 = \rho_0$). The document uses $a''/a = 2a^2H^2 + a^2\dot H = (aH)^2(2-\epsilon)$ (NUM 03 p.3). **$\epsilon$ is not defined explicitly**; from that line and from the combination $(1+z)(aH)^2\,dχ_s/dz - (1+z)(a^2H^2 + a^2\dot H)\,d\chi_s/dz = (1+z)(aH)^2\epsilon\, d\chi_s/dz$ (p.3) one infers $\epsilon = -\dot H/H^2$ (inferred, not stated).
- **Scale-factor normalisation.** $a_0$ appears explicitly (NUM 03 pp.4–5): the document uses $a_0/a(z) = 1+z$ (p.5, implicitly) and keeps a $1/a_0^2$ that is absorbed into $Q_s/a_0^2 \propto q_{\rm phys}^2$ (pp.5–7). $a_0 = 1$ is **not** assumed. The symbol $k_{\rm phys}$ is introduced on p.4 via $\dfrac{k^2}{H^2}\dfrac{1}{a_0^2}\dfrac{a_0^2}{a^2(1+z)^2} \to \dfrac{k_{\rm phys}^2}{H^2}$, so $k_{\rm phys} = k/a_0$ there (inferred from that line; see Q4). Likewise $q_{\rm phys}$ (pp.7–10) is $q/a_0$ (inferred from $Q_s/(a_0^2 H^2) \propto q_{\rm phys}^2/H^2$, p.7).
- **Green's function.** $\mathrm{Gr}^\chi_k(z,z')$: first argument $z$ is the response (observation) redshift, second $z'$ the source redshift. Defined with a **$+\delta$** source: $\dfrac{d^2\mathrm{Gr}_k}{dz^2} + \dfrac{\epsilon}{1+z}\dfrac{d\mathrm{Gr}_k}{dz} + \Big(\dfrac{k^2}{H^2} + \dfrac{\epsilon-2}{(1+z)^2}\Big)\mathrm{Gr}_k = \delta(z-z')$ (NUM 03 p.5). Causal (retarded in time = advanced in $z$) boundary condition $\mathrm{Gr}_k(z,z') = d\mathrm{Gr}_k/dz = 0$ for $z > z'$, giving $d\mathrm{Gr}_k/dz\big|_{z=z'} = +1$ (p.6). The document then **redefines** $\overline{\mathrm{Gr}}_k \equiv -\mathrm{Gr}_k$ (p.5) and uses $\overline{\mathrm{Gr}}_k$ with the integral written as $\int_z^{z_{\rm init}} dz'$ in Steps 3–5 (pp.6–9). No "literature" Green's function is referenced. The superscript $\chi$ on $\mathrm{Gr}^\chi_k$ (pp.5) marks that it is the Green's function of the rescaled variable $\chi_s = a\,h_s$.
- **Equation of state.** $w_*$ = "the fixed value of $w$ at the initial time" (NUM 03 p.4), entering the initial-condition normalisation $3(1+w_*)/(5+3w_*)$. $w_0$ enters via $\rho_0 + p_0 = \rho_0(1+w_0)$ (p.2–3), where the subscript 0 on $\rho_0, p_0$ denotes the *background* (unperturbed) fluid at redshift $z$, not today (inferred from $3H^2M_P^2 = \rho_0$ on p.1 with $H = H(z)$). $w_0$ is therefore $w(z)$ at the source redshift (inferred). The shorthands $b$ and $c_s^2$ of README §5 do **not** appear.
- **Fourier / power-spectrum conventions.** Loop measure $\int d^3q/(2\pi)^3$ throughout. Wick contraction on p.7 implies $\langle \zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle = (2\pi)^3\delta(\mathbf q + \mathbf t)P^*(q)$ (inferred from the $(2\pi)^6\delta\delta$ structure; not written as a standalone definition). Dimensionful $P^*(q)$ and dimensionless $\mathcal P^*(q)$ related by $P^*(q) = 2\pi^2\,\mathcal P^*(q)/q^3$ (inferred from $P^*(q)P^*(r) \to 4\pi^4\,\mathcal P^*(q)\mathcal P^*(r)/(q^3 r^3)$ on p.9; not written explicitly). Tensor two-point function written as $\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\delta_{ss'}\times\ldots$ (pp.8–10). Polarisation tensor $e^{lm}_s(\mathbf k)$, $s = \pm$, transverse ("$e$ annihilates $\mathbf k$", p.2); $Q_s \equiv e^{lm}_s(\mathbf k)q_l q_m$ with $Q_+ = \tfrac{1}{\sqrt2}q_{\rm phys}^2\sin^2\theta\cos 2\phi$, $Q_- = \tfrac{1}{\sqrt2}q_{\rm phys}^2\sin^2\theta\sin 2\phi$ (p.9), consistent with $e^{lm}_s e_{lm}^{s'} = \delta_{ss'}$ (inferred). $\theta$, $\phi$ are not defined on the page; inferred to be polar and azimuthal angles of $\mathbf q$ relative to $\mathbf k$.
- **Transfer function.** $T_{\mathbf k}(z)$ is the transfer function of the Newtonian potential $\phi$, entering through $\phi_{\mathbf k} = \dfrac{3(1+w_*)}{5+3w_*}T_{\mathbf k}(z)\,\zeta^*_{\mathbf k}$ (NUM 03 p.4); $\zeta^*_{\mathbf k}$ is the primordial curvature perturbation at the initial time (the star denotes the initial time, matching $w_*$). The early-time normalisation of $T$ is **not stated** (presumably $T\to1$, inferred).
- **Momentum labels.** External $\mathbf k$ (and $\mathbf k'$ for the second field), loop momentum $\mathbf q$, and $\mathbf r$ with $\delta(\mathbf k-\mathbf q-\mathbf r)$ integrated out so $\mathbf r = \mathbf k-\mathbf q$ (p.2). In Step 5, $r \equiv |\mathbf k-\mathbf q|$ (pp.9–10) and $\mathbf t$ is the second loop momentum in the four-point contraction (p.7). Angular reduction: $d^3q \to q^2dq\,\sin\theta\,(d\theta)\,d\phi$; the $\phi$ integral is done ($\int_0^{2\pi}\cos^2 2\phi\,d\phi = \pi$), the $\theta$ integral is left (pp.9–10).

## 3. Results

Notation: I write $\mathbf k$ for the author's underlined (vector) $k$; $\zeta^*$ for the author's starred $\zeta$ (glyph resembles "5*"/"S*"); $\mathrm{Gr}$ for the author's "Gr". Subscripts $q$, $k-q$ on $\phi$ and $T$ are vector labels.

### R1 — Starting sourced tensor equation (conformal time)
NUM 03 p.1. Cited as "from ⑪, page 3".
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s = -\int\frac{d^3q\,d^3r}{(2\pi)^3}\,\delta(\mathbf k-\mathbf q-\mathbf r)\;e^{lm}_s(\mathbf k)\Big\{4q_l r_m\,\phi_{\mathbf q}\phi_{\mathbf r} + \frac{8M_P^2}{\rho_0+p_0}\,\frac{q_l r_m}{a^2}\,(\phi'_{\mathbf q}+\mathcal H\phi_{\mathbf q})(\phi'_{\mathbf r}+\mathcal H\phi_{\mathbf r})\Big\}.
$$
Primes are $d/d\eta$. Confidence: high.

### R2 — Background relations used
NUM 03 p.1.
$$3H^2M_P^2 = \rho_0, \qquad \frac{d}{d\eta} = -(1+z)\,aH\,\frac{d}{dz}.$$
Confidence: high.

### R3 — Rescaled variable
NUM 03 p.1 (Step 2(a)). $h_s = \dfrac{1}{a}\chi_s$. Confidence: high.

### R4 — $\chi_s$ equation in conformal time
NUM 03 p.2.
$$\chi_s'' + \Big\{2\mathcal H^2 - \frac{a''}{a} - 2\mathcal H^2\Big\}\chi_s + k^2\chi_s = a\times(\text{source term}),$$
i.e. $\chi_s'' + (k^2 - a''/a)\chi_s = a\times$ (RHS of R1). Confidence: high.

### R5 — Polarisation-tensor contraction after integrating out $\mathbf r$
NUM 03 p.2. With $\mathbf r = \mathbf k-\mathbf q$,
$$e^{lm}_s(\mathbf k)\,q_l r_m \to e^{lm}_s(\mathbf k)q_l k_m - e^{lm}_s(\mathbf k)q_l q_m = -\,e^{lm}_s(\mathbf k)\,q_l q_m,$$
"since $e$ annihilates $\mathbf k$". This sign converts the leading $-\int$ of R1 into $+4a\int$ in R7. Confidence: high.

### R6 — $a''/a$ in terms of $H$ and $\epsilon$
NUM 03 p.3.
$$\frac{a''}{a} = \frac{1}{a}\,a\frac{d}{dt}(a\dot a) = \frac{d}{dt}(a^2H) = 2a\dot aH + a^2\dot H = 2a^2H^2 + a^2\dot H = (aH)^2(2-\epsilon).$$
Confidence: high. ($\epsilon = -\dot H/H^2$ is implied, not written.)

### R7 — Redshift-form $\chi_s$ equation, before dividing through
NUM 03 p.3 ("so we get").
$$
(1+z)^2(aH)^2\frac{d^2\chi_s}{dz^2} + (1+z)(aH)^2\,\epsilon\,\frac{d\chi_s}{dz} + \big(k^2 - (aH)^2(2-\epsilon)\big)\chi_s
= 4a\int\frac{d^3q}{(2\pi)^3}\,e^{lm}_s(\mathbf k)q_lq_m\Big\{\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q} + \frac{2}{3(1+w_0)}\Big(\phi_{\mathbf q}-(1+z)\frac{d\phi_{\mathbf q}}{dz}\Big)\Big(\phi_{\mathbf k-\mathbf q}-(1+z)\frac{d\phi_{\mathbf k-\mathbf q}}{dz}\Big)\Big\}.
$$
The coefficient $\tfrac{2}{3(1+w_0)}$ arises from $\dfrac{8M_P^2}{3H^2M_P^2}\dfrac{1}{a^2}\dfrac{1}{1+w_0}(aH)^2$ with an overall 4 pulled out (a small "2" is written above the "8" on pp.2–3 to record $8 = 4\times2$). Confidence: high.

### R8 — Divided form
NUM 03 p.4 (top).
$$
\frac{d^2\chi_s}{dz^2} + \frac{\epsilon}{1+z}\frac{d\chi_s}{dz} + \Big(\frac{k^2}{a^2H^2}\frac{1}{(1+z)^2} - \frac{2-\epsilon}{(1+z)^2}\Big)\chi_s
= \frac{4}{a}\frac{1}{H^2}\int\frac{d^3q}{(2\pi)^3}\frac{1}{(1+z)^2}\,e^{lm}_s(\mathbf k)q_lq_m\Big\{\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q} + \frac{2}{3(1+w_0)}\Big(\phi-(1+z)\frac{d\phi}{dz}\Big)_{\mathbf q}\Big(\phi-(1+z)\frac{d\phi}{dz}\Big)_{\mathbf k-\mathbf q}\Big\}.
$$
Confidence: high (the $(1+z)^2$ in the $k^2$ term is written as "$1/(1+z^2)$" with sloppy bracketing; read as $(1+z)^2$).

### R9 — Inflationary initial condition / transfer-function normalisation
NUM 03 p.4.
$$\phi_{\mathbf k} = \frac{3(1+w_*)}{5+3w_*}\,T_{\mathbf k}(z)\,\zeta^*_{\mathbf k},\qquad\text{"where $w_*$ is the fixed value of $w$ at the initial time".}$$
Confidence: high.

### R10 — Definitions of $Q_s$ and $k_{\rm phys}$
NUM 03 p.4.
$$Q_s \equiv e^{lm}_s(\mathbf k)\,q_l q_m \quad(\text{brace label under the contraction}),\qquad
\frac{k^2}{H^2}\frac{1}{a_0^2}\frac{a_0^2}{a^2(1+z)^2} \equiv \frac{k_{\rm phys}^2}{H^2}.$$
Confidence: high for $Q_s$; medium for what $k_{\rm phys}$ denotes (the line reads as $k_{\rm phys} = k/a_0$; see Q4). Later written as $Q_s(\mathbf q)$ (pp.4–6) and once as $Q_s(\mathbf k,\mathbf q)$ (p.7).

### R11 — Final redshift-space source equation (boxed source term)
NUM 03 p.4 (bottom; the braces are boxed in red by the author).
$$
\frac{d^2\chi_s}{dz^2} + \frac{\epsilon}{1+z}\frac{d\chi_s}{dz} + \Big(\frac{k_{\rm phys}^2}{H^2} + \frac{\epsilon-2}{(1+z)^2}\Big)\chi_s
= \frac{1}{a}\frac{1}{H^2}\frac{1}{(1+z)^2}\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf q)\,\frac{36(1+w_*)^2}{(5+3w_*)^2}\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}
\;\boxed{\Big\{T_{\mathbf q}T_{\mathbf k-\mathbf q} + \frac{2}{3(1+w_0)}\Big(T-(1+z)\frac{dT}{dz}\Big)_{\mathbf q}\Big(T-(1+z)\frac{dT}{dz}\Big)_{\mathbf k-\mathbf q}\Big\}}.
$$
The boxed braces are what the document subsequently calls $f(z'\,|\,\cdot,\cdot)$ (R12). $36 = 4\times 9$, where $9(1+w_*)^2/(5+3w_*)^2$ comes from squaring R9. Confidence: high. (A leading "4" at the start of this line is struck/over-written, see §5.)

### R12 — Green's function solution for $\chi_s$
NUM 03 p.5 (Step 3).
$$
\chi_s(z) = \frac{36(1+w_*)^2}{(5+3w_*)^2}\int_{z_{\rm init}}^{z}dz'\,\mathrm{Gr}^\chi_{\mathbf k}(z,z')\,\frac{1}{a(z')}\frac{1}{H(z')^2}\frac{1}{(1+z')^2}\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf q)\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,f(z'\,|\,\mathbf k,\mathbf k-\mathbf q).
$$
This is the implicit definition of $f$: $f(z'|\ldots)$ = the boxed braces of R11 evaluated at $z'$. Confidence: high for the formula; medium for the first argument of $f$ (read as $\mathbf k$; on pp.5 and 7 it is written $f(z'|\mathbf k,\mathbf k-\mathbf q)$, on pp.6, 9, 10 as $f(\cdot|\mathbf q,\mathbf k-\mathbf q)$; see Q5).

### R13 — Green's function solution for $h_s$ (with $\mathrm{Gr}$, $+\delta$ convention)
NUM 03 p.5.
$$
h_s(z) = \frac{1}{a}\chi_s(z) = \frac{36(1+w_*)^2}{(5+3w_*)^2}\int_{z_{\rm init}}^{z}dz'\,\mathrm{Gr}^\chi_{\mathbf k}(z,z')\,\frac{1}{H(z')^2}\frac{a_0}{a(z')}\frac{a_0}{a(z)}\frac{1}{(1+z')^2}\int\frac{d^3q}{(2\pi)^3}\frac{Q_s(\mathbf q)}{a_0^2}\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,f(z'|\mathbf k,\mathbf k-\mathbf q)
$$
$$
= \frac{36(1+w_*)^2}{(5+3w_*)^2}\int_{z_{\rm init}}^{z}dz'\,\mathrm{Gr}^\chi_{\mathbf k}(z,z')\,\frac{1}{H(z')^2}\,\frac{1+z}{1+z'}\int\frac{d^3q}{(2\pi)^3}\frac{Q_s(\mathbf q)}{a_0^2}\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,f(z'|\mathbf k,\mathbf k-\mathbf q).
$$
Confidence: high.

### R14 — Defining equation of the Green's function
NUM 03 p.5. "Note, here $\mathrm{Gr}^\chi_{\mathbf k}$ is the usual Green's function satisfying"
$$
\frac{d^2\mathrm{Gr}_{\mathbf k}}{dz^2} + \frac{\epsilon}{1+z}\frac{d\mathrm{Gr}_{\mathbf k}}{dz} + \Big(\frac{k^2}{H^2} + \frac{\epsilon-2}{(1+z)^2}\Big)\mathrm{Gr}_{\mathbf k} = \delta(z-z').
$$
Confidence: high for the glyphs. Note it is written $k^2/H^2$ here, not $k_{\rm phys}^2/H^2$ as in R11 (Q4).

### R15 — Sign-flipped Green's function and causal boundary conditions
NUM 03 pp.5–6. "If we instead define $\overline{\mathrm{Gr}}_{\mathbf k} = -\mathrm{Gr}_{\mathbf k}$, then"
$$
\int_{z'-\varepsilon}^{z'+\varepsilon}\frac{d^2\overline{\mathrm{Gr}}_{\mathbf k}}{dz^2} + (\cdots) = -\int_{z'-\varepsilon}^{z'+\varepsilon}\delta(z-z')\,dz = -1,\qquad
\Big[\frac{d\overline{\mathrm{Gr}}_{\mathbf k}}{dz}\Big]_{z'-\varepsilon}^{z'+\varepsilon} = -1,
$$
"but $\overline{\mathrm{Gr}}_{\mathbf k}(z,z') = d\overline{\mathrm{Gr}}_{\mathbf k}(z,z')/dz = 0$ for $z>z'$ to get the causal Green fn", so
$$\frac{d\overline{\mathrm{Gr}}_{\mathbf k}}{dz}\Big|_{z=z'} = +1.$$
Confidence: high for the formulas. Medium on *which* Green's function the overbar-free symbols on p.6 refer to: on p.6 the jump-condition lines are written with plain "$\mathrm{Gr}_{\mathbf k}$" (no overbar), although the sentence on p.5 says the calculation is for $\overline{\mathrm{Gr}}$; the $-1$ on the right-hand side is consistent with the $\overline{\mathrm{Gr}}$ ($-\delta$) convention, and the $\overline{\mathrm{Gr}}$ is then used in R16. See Q6.

### R16 — $h_s$ in terms of $\overline{\mathrm{Gr}}$ (final Step-3 form)
NUM 03 p.6.
$$
h_s(z) = 36\Big(\frac{1+w_*}{5+3w_*}\Big)^2\int_{z}^{z_{\rm init}}dz'\,\overline{\mathrm{Gr}}_{\mathbf k}(z,z')\,\frac{1+z}{1+z'}\int\frac{d^3q}{(2\pi)^3}\frac{Q_s(\mathbf q)}{a_0^2}\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,\frac{f(z'|\mathbf q,\mathbf k-\mathbf q)}{H^2(z')}.
$$
Note the integration limits have been reversed relative to R13 ($\int_z^{z_{\rm init}}$), absorbing the sign of $\overline{\mathrm{Gr}} = -\mathrm{Gr}$. Confidence: high; medium for the first argument of $f$ (read $\mathbf q$ here).

### R17 — Dimension check (Step 4)
NUM 03 p.6. "In real space $h_s$ and $\zeta$ should be dimensionless. So $h^s_{\mathbf k}$ and $\zeta_{\mathbf k}$ have dimension $[M^{-3}]$." RHS dimension: $\int d^3q\;[M^3]$, $Q_s(\mathbf q)\;[M^2]$, $\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\;[M^{-6}]$, $1/H^2\;[M^{-2}]$, product $[M^{-3}]$ ✓. Confidence: high.

### R18 — Definition of the source redshift integral $I_s$
NUM 03 p.7 (Step 5). "Define the source redshift integral as"
$$
I_s = \int_{z}^{z_{\rm init}}dz'\,\overline{\mathrm{Gr}}_{\mathbf k}(z,z')\,\frac{1+z}{1+z'}\,\frac{Q_s(\mathbf k,\mathbf q)}{a_0^2H^2(z')}\,f(z'|\mathbf k,\mathbf k-\mathbf q).
$$
Confidence: medium. Ambiguous glyphs: the denominator reads $a_0^2 H^2(z')$ with an ink blot on/under the $H$ — could be $H^2(z')$ (most likely, consistent with R16) or a struck subscript. Written elsewhere as $I(z|\mathbf k,\mathbf q)$, $I_s(z|\mathbf k,\mathbf q)$, $I_s(\mathbf q,\mathbf k-\mathbf q)$, $I_s(z|\mathbf q,\mathbf k-\mathbf q)$.

### R19 — $h_s$ in terms of $I_s$
NUM 03 p.7.
$$h_s(\mathbf k) = 36\Big(\frac{1+w_*}{5+3w_*}\Big)^2\int\frac{d^3q}{(2\pi)^3}\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,I(z|\mathbf k,\mathbf q).$$
Confidence: high.

### R20 — Remarks on $a_0$ dependence
NUM 03 p.7 (prose).
$$\frac{Q_s}{a_0^2H^2(z')} \propto \frac{q_{\rm phys}^2}{H^2(z')};$$
"$\int d^3q\,\zeta^*_{\mathbf q}$ is independent of $a_0$"; "$\zeta^*_{\mathbf k-\mathbf q}$ gives $h_s(\mathbf k)$ the right dependence on $a_0$ to be a comoving Fourier transform." Confidence: high.

### R21 — Two-point function: Wick contraction
NUM 03 p.7.
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = 1024\Big(\frac{1+w_*}{5+3w_*}\Big)^4\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\,\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\zeta^*_{\mathbf t}\zeta^*_{\mathbf k'-\mathbf t}\rangle\,I_s(z|\mathbf k,\mathbf q)\,I_{s'}(z|\mathbf k',\mathbf t)
$$
$$
= 1024\Big(\frac{1+w_*}{5+3w_*}\Big)^4\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\Big\{(2\pi)^6\delta(\mathbf q+\mathbf t)\delta(\mathbf k-\mathbf q+\mathbf k'-\mathbf t) + (2\pi)^6\delta(\mathbf q+\mathbf k'-\mathbf t)\delta(\mathbf k-\mathbf q+\mathbf t)\Big\}\,P^*(q)P^*(|\mathbf k-\mathbf q|)\,I_s(z|\mathbf k,\mathbf q)\,I_{s'}(z|\mathbf k',\mathbf t).
$$
**Red annotation (DS 10 July 2025) on the 1024:** "Looks like this should be 1296 = 36^2, not 1024". Confidence: high (the $1024$ is what is written in blue; $36^2 = 1296$ is the annotation).

### R22 — Two-point function after the $\mathbf t$ integration
NUM 03 p.8.
$$
\langle h_s(\mathbf k)h_s(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,1024\Big(\frac{1+w_*}{5+3w_*}\Big)^4\int\frac{d^3q}{(2\pi)^3}\,P^*(q)P^*(|\mathbf k-\mathbf q|)\Big(I_s(z|\mathbf k,\mathbf q)I_{s'}(z|\mathbf k',-\mathbf q) + I_s(z|\mathbf k,\mathbf q)I_{s'}(z|\mathbf k',\mathbf k'+\mathbf q)\Big).
$$
(The 1024 here is boxed in red without a separate annotation.) Confidence: high.

### R23 — Symmetry reduction of the two $I_sI_{s'}$ terms
NUM 03 p.8. "$I_s$ should be regarded as a function of $\mathbf q$ and $\mathbf k-\mathbf q$, not $\mathbf k$ and $\mathbf q$ separately, except for the Green's function":
$I_s(\mathbf q,\mathbf k-\mathbf q)I_{s'}(-\mathbf q,\mathbf k'+\mathbf q) + I_s(\mathbf q,\mathbf k-\mathbf q)I_{s'}(\mathbf k'+\mathbf q,-\mathbf q)$.
"The Green's function only depends on $k = |\mathbf k| = |\mathbf k'|$":
$$\Rightarrow I_s(\mathbf q,\mathbf k-\mathbf q)I_{s'}(-\mathbf q,\mathbf q-\mathbf k) + I_s(\mathbf q,\mathbf k-\mathbf q)I_{s'}(\mathbf q-\mathbf k,-\mathbf q)
= 2\,I_s(\mathbf q,\mathbf k-\mathbf q)I_{s'}(-\mathbf q,\mathbf q-\mathbf k)$$
"because source $f$ is symmetric and $Q^s$ is blind to $\mathbf k$",
$$= 2\,I_s(\mathbf q,\mathbf k-\mathbf q)^2$$
"because source only depends on $|\mathbf q|$ and $|\mathbf k-\mathbf q|$". Confidence: high.

### R24 — Two-point function, single $I_s^2$ form
NUM 03 p.9 (top).
$$
\langle h_s(\mathbf k)h_s(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,2048\Big(\frac{1+w_*}{5+3w_*}\Big)^4\int\frac{d^3q}{(2\pi)^3}\,P^*(q)P^*(|\mathbf k-\mathbf q|)\,I_s(z|\mathbf q,\mathbf k-\mathbf q)^2.
$$
**Red annotation (DS 10 July 2025):** "should be 2592". Confidence: high for 2048 (blue) and the annotation; medium for the exponent on the bracket, which is a "4" over-written on what looks like an earlier "2" (read as 4, consistent with pp.7, 8, 9, 10).

### R25 — Polarisation contractions and azimuthal integrals
NUM 03 p.9.
$$Q_+ = \frac{1}{\sqrt2}\,q_{\rm phys}^2\sin^2\theta\cos 2\phi,\qquad Q_- = \frac{1}{\sqrt2}\,q_{\rm phys}^2\sin^2\theta\sin 2\phi,$$
$$\int_0^{2\pi}d\phi\,Q_+Q_- = 0,\qquad \int_0^{2\pi}d\phi\cos^2 2\phi = \int_0^{2\pi}d\phi\sin^2 2\phi = \pi.$$
Confidence: high.

### R26 — Two-point function after the $\phi$ integral
NUM 03 p.9 (middle).
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,\delta_{ss'}\,1024\Big(\frac{1+w_*}{5+3w_*}\Big)^4\pi\int\frac{d^3q}{(2\pi)^3}\,P^*(q)P^*(|\mathbf k-\mathbf q|)\,\Big\{\int_z^{z_{\rm init}}dz'\,\overline{\mathrm{Gr}}_k(z,z')\,\frac{1+z}{1+z'}\,\sin^2\theta\,\frac{q_{\rm phys}^2}{H(z')^2}\,f(z'|\mathbf q,\mathbf k-\mathbf q)\Big\}^2.
$$
(A factor written after $P^*(|\mathbf k-\mathbf q|)$, of the form $q_{\rm phys}^4/4$, is struck through — its $q^2_{\rm phys}$ has been moved inside the braces; see §5.) **Red annotation (DS 10 July 2025):** "should be 1292". Confidence: high for the blue formula; the $1024\pi$ follows from $2048\times\tfrac12\times\pi$. The annotation value 1292 is transcribed as written — see Q2.

### R27 — Conversion to dimensionless spectra and angular measure
NUM 03 p.9 (bottom).
$$
= (2\pi)^3\delta(\mathbf k+\mathbf k')\,\delta_{ss'}\,1024\pi\Big(\frac{1+w_*}{5+3w_*}\Big)^4\int q^2dq\,\sin\theta\;\frac{4\pi^4}{(2\pi)^3}\,\frac{\mathcal P^*(q)}{q^3}\frac{\mathcal P^*(r)}{r^3}\,\sin^4\theta\,\Big\{\int_z^{z_{\rm init}}dz'\,\overline{\mathrm{Gr}}_k(z,z')\,\frac{1+z}{1+z'}\,\frac{q_{\rm phys}^2}{H(z')^2}\,f(z|\mathbf q,\mathbf k-\mathbf q)\Big\}^2.
$$
Confidence: high for glyphs. Two things are as-written: no $d\theta$ appears in the measure ("$q^2dq\sin\theta$"), and the argument of $f$ is written $f(z|\ldots)$ with an unprimed $z$ (Q7). $r = |\mathbf k-\mathbf q|$.

### R28 — FINAL FORM of the induced tensor two-point function
NUM 03 p.10.
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\delta(\mathbf k+\mathbf k')\,\delta_{ss'}\;512\pi^2\Big(\frac{1+w_*}{5+3w_*}\Big)^4\int\frac{dq}{q}\,\sin^5\theta\;\mathcal P^*(q)\,\frac{\mathcal P^*(r)}{r^3}\;\Big\{\int_z^{z_{\rm init}}dz'\,\mathrm{Gr}_k(z,z')\,\frac{1+z}{1+z'}\,\frac{q_{\rm phys}^2}{H(z')^2}\,f(z|\mathbf q,\mathbf k-\mathbf q)\Big\}^2.
$$
**Red annotation (DS 10 July 2025) on the $512\pi^2$:** "should be 646". Confidence: high for the blue glyphs ($512\pi^2 = 1024\pi\times 4\pi^4/(2\pi)^3$). As written: the Green's function on this page has **no overbar** (cf. $\overline{\mathrm{Gr}}_k$ on p.9), $f$ has an unprimed first argument $z$, and there is no $d\theta$ in the measure (Q6, Q7, Q8). The $r^{-3}$ sits under $\mathcal P^*(r)$ only; the $q^{-3}$ has combined with $q^2dq$ into $dq/q$.

## 4. Checks

Not applicable: `NUM` 03 does not recheck another document. Its own internal check is the dimension count of Step 4 (R17), which the author marks with a tick.

## 5. Corrections and cross-outs

1. **pp.2–3, small "2" written above "8" in $8M_P^2$.** Both the original 8 and the later 2 are on the page; the 2 records the pulling-out of the overall factor 4 ($8 = 4\times2$). Later steps use $\tfrac{2}{3(1+w_0)}$ inside the braces with 4 outside (R7 onward).
2. **p.4, third displayed equation:** $\phi_{\mathbf q}\phi_{\mathbf k-\mathbf q}$ and the first $\phi$ inside the derivative bracket are struck through with "T" written above each — the replacement $\phi\to T$ after inserting R9. Later steps use $T$.
3. **p.4, last line:** a leading "4" before "=" is struck/over-written; the 4 has been absorbed into $36 = 4\times9$. Later steps use 36 (R11–R19).
4. **p.5, top right:** a second circled page number next to "⑤" is scribbled out (a renumbering; no content).
5. **p.7:** a $P^*(\mathbf q)P^*(\mathbf k-\mathbf q)$ written inside the Wick braces is struck through and rewritten below the closing brace as a common factor (R21). No change of content.
6. **p.9, middle:** a factor after $P^*(|\mathbf k-\mathbf q|)$, read as $q^4_{\rm phys}/4$ (possibly $/2$; low confidence, it is struck), is crossed out; the $q^2_{\rm phys}$ appears instead inside the squared braces (R26). Later steps (R27–R28) use the inside-the-braces placement.
7. **p.9, top:** exponent on $\big(\frac{1+w_*}{5+3w_*}\big)$ over-written; read as 4 (R24).
8. **Red typed annotations, "DS 10 July 2025" (later than the Sep-2024 body):**
   - p.7, on "1024": "Looks like this should be 1296 = 36^2, not 1024".
   - p.8, "1024" boxed in red, no text.
   - p.9 top, on "2048": "should be 2592".
   - p.9 middle, on "1024" (in $1024\pi$): "should be 1292".
   - p.9 bottom, "1024" (in $1024\pi$) boxed in red, no text.
   - p.10, on "512" (in $512\pi^2$): "should be 646".
   The blue body text was **not** altered; the corrected prefactors exist only as annotations. Which one "later steps use" is therefore undetermined within this document; the annotations are the author's later view. (Consistency of the annotated numbers is raised in Q2.)

## 6. Open questions

- **Q1 — Page count.** README §1.2 and the assignment give 46 pages for `NUM` 03; the file has 10 pages (verified with `pdfinfo` and by the Read tool refusing pages 11+). Either the README page count is a typo or a longer version of the document exists elsewhere. Nothing in the 10 pages appears truncated: p.10 ends with the final formula and the rest of the page is blank.
- **Q2 — Red-annotation prefactors are not mutually consistent.** Starting from the p.7 annotation $1296 = 36^2$ and following the blue text's own operations: p.8/p.9-top doubling gives $2592$ (matches the p.9 annotation); the $\phi$-integral step multiplies by $\tfrac12\times\pi$ giving $1296\pi$, but the p.9 annotation reads "1292"; the $4\pi^4/(2\pi)^3$ step then gives $648\pi^2$, but the p.10 annotation reads "646". "1292" and "646" look like typos for 1296 and 648, but I have transcribed them as written. The author should confirm the intended final prefactor (blue: $512\pi^2$; implied by annotations: $648\pi^2$).
- **Q3 — Origin of 1024 vs 1296.** The blue text squares the prefactor 36 of R19 and writes 1024 ($=32^2$) rather than $36^2$; the doubling on p.9 gives 2048 (blue). This is the discrepancy the red annotations address; recorded here because it propagates into every result R21–R28.
  **Closed 2026-09-07:** 1024 is $32^2$, the `MAIN` 11/14 coefficient squared; in `NUM` 03 the super-horizon factor $c_*^2$ is outside $f$, so the correct value is $36^2 = 1296$. See §0.2 of `03-source-term.md`.
- **Q4 — $k_{\rm phys}$ and the $k^2/H^2$ in the Green's function equation.** On p.4 the combination $k^2/(a^2H^2(1+z)^2)$ is rewritten as $k^2/(H^2a_0^2)\cdot a_0^2/(a^2(1+z)^2)$ and then as $k_{\rm phys}^2/H^2$, which reads as $k_{\rm phys} \equiv k/a_0$ with $H = H(z)$. The Green's function equation on p.5 (R14) then writes just $k^2/H^2$. Whether $k$ on p.5 means $k_{\rm phys}$ (i.e. $k/a_0$) is not stated; presumably yes, since it is the Green's function of the R11 operator.
- **Q5 — First argument of $f$.** Written $f(z'|\mathbf k,\mathbf k-\mathbf q)$ on pp.5 and 7 (R12, R13, R18) but $f(\cdot|\mathbf q,\mathbf k-\mathbf q)$ on pp.6, 9, 10 (R16, R26–R28). The p.8 prose says the source should be regarded as a function of $\mathbf q$ and $\mathbf k-\mathbf q$. The braces of R11 depend on $\mathbf q$ and $\mathbf k-\mathbf q$ only, so the $\mathbf k$ on pp.5, 7 is most likely a slip, but this is a reading of intent, not of the page.
- **Q6 — Overbar on the Green's function.** $\overline{\mathrm{Gr}}_k = -\mathrm{Gr}_k$ is defined on p.5; the jump-condition working on p.6 is written with plain $\mathrm{Gr}_k$ but with the $-1$ appropriate to $\overline{\mathrm{Gr}}_k$; pp.6–9 use $\overline{\mathrm{Gr}}_k$; the final formula on p.10 (R28) is written with plain $\mathrm{Gr}_k$ (no overbar visible at 260 dpi). Given that the integral in R28 is still $\int_z^{z_{\rm init}}$ and the squared braces are insensitive to the overall sign, this does not change R28's value, but the intended symbol should be confirmed.
- **Q7 — Unprimed $z$ in $f$.** In R27 and R28 (p.9 bottom, p.10) the source function is written $f(z|\mathbf q,\mathbf k-\mathbf q)$ with the outer redshift $z$, whereas in R12–R26 it is $f(z'|\ldots)$ with the integration variable. Almost certainly a slip for $z'$; transcribed as written.
- **Q8 — Missing $d\theta$.** The measure on pp.9–10 is written "$q^2dq\,\sin\theta$" / "$dq/q\,\sin^5\theta$" with no $d\theta$; the $\theta$ integral is nevertheless implied (only $\phi$ has been integrated). Also $\theta$ and $\phi$ are never defined on the page (taken to be the angles of $\mathbf q$ relative to $\mathbf k$).
- **Q9 — $I_s$ notation.** R18 writes $Q_s(\mathbf k,\mathbf q)$ and the denominator $a_0^2H^2(z')$ carries an ink blot on $H$ (medium confidence). The argument list of $I_s$ changes across pp.7–9 ($I(z|\mathbf k,\mathbf q)$, $I_s(z|\mathbf k,\mathbf q)$, $I_s(\mathbf q,\mathbf k-\mathbf q)$, $I_s(z|\mathbf q,\mathbf k-\mathbf q)$); all refer to the same object of R18.
- **Q10 — Meaning of the subscript 0 in $w_0$, $\rho_0$, $p_0$.** Inferred to mean background quantities at redshift $z$ (from $3H^2M_P^2 = \rho_0$ with $H = H(z)$), not present-day values. Not stated on the page. Since $w_0$ appears inside $f$ under the $z'$ integral, it must be evaluated at $z'$; the page writes it without argument.
- **Q11 — Convention definitions not written.** $\epsilon$ (inferred $-\dot H/H^2$), the $\langle\zeta\zeta\rangle$ normalisation, $P^* \leftrightarrow \mathcal P^*$ relation, polarisation normalisation, and the early-time normalisation of $T$ are all inferred from usage (see §2), none is stated.
