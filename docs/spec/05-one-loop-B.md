# Spec 05-B: One-loop (P22) induced tensor power spectrum — independent duplicate transcription of `MAIN` 14

> **Author note (2026-09-07).** This is the independent duplicate transcription and is kept as written. For the definitive project convention on the normalisation of $a$ (the code absorbs $a_0$ into $k/a_0$ and $a_0\eta$; it does **not** set $a_0 = 1$) and on which Green's function the code computes (the unit-jump $\bar G_k$ of `NUM` 03, $= -a_0H(z')\,{\rm Gr}_k$), see §0 of the primary spec `05-one-loop.md`. Tier 1.1 of `REVIEW-QUEUE.md` was signed off on that basis. Tiers 1.2–1.4 were also signed off 2026-09-07; in particular this transcription's reading of the seed as $\zeta^*$ (diff-05 D1) is the correct one, and $P_* = P_\zeta$. See §0 of the primary spec.


Transcription **B** (independent duplicate). Produced without sight of `docs/spec/05-one-loop.md`.
Transcriber has not read `MAIN` 11, nor any code in the repository.

---

## 1. Source documents

| Ref | File | Pages | Date on first page |
|---|---|---|---|
| `MAIN` 14 | `Green's function formula for P22/14 - 2022:08:30 - recheck P22 power spectrum.pdf` | 12 | 30 Aug 2022 (header: "CALCULATION 14", title "Recheck formula for the induced tensor power spectrum") |

The document is organised in seven numbered steps:

| Step | Pages | Content |
|---|---|---|
| 1 | 1–2 | Sourced tensor equation for a single polarisation; substitute linear solution for $\Phi_{\mathbf q}$; specialise $\mathcal H$ to fixed $w$ |
| 2 | 2–3 | Green's function formula for $h_s$ via $\chi_s = a h_s$ |
| 3 | 3–4 | Collect ingredients: polarisation factors $Q_s$, transfer function $\Phi$, Green's function $\mathrm{Gr}_k$ |
| 4 | 4–7 | Build $\langle h_s h_{s'}\rangle$; Wick contraction; define $P^h_{22}(k)$ |
| 5 | 7–10 | Rewrite the source function $f(q,r,\eta)$ in terms of $b$ and in Bessel-function form |
| 6 | 10–11 | **Collect the time integral** — endpoint is the `TARGET` (R30) |
| 7 | 11–12 | Rewrite in "Fabrikant form" (spherical Bessel functions) — **NOT to be used** (README §2) |

---

## 2. Conventions in force

- **Time variable.** Conformal time $\eta$ throughout (p.1: $h_s'' + 2\mathcal H h_s' + k^2 h_s$; p.1: $\mathcal H = \frac{2}{1+3w}\frac1\eta$). Integration variable for the time integrals is $\eta'$ (and $\eta''$ for the second Green's function), lower limit $\eta_0$, upper limit $\eta$ (p.3, p.4). $\eta_0$ is never specified.
- **Meaning of a prime.**
  - On $h_s$, $\chi_s$, $a$, $\mathrm{Gr}$ and on $\Phi_{\mathbf q}(\eta)$: $d/d\eta$ (p.1–3). Stated only implicitly through the equations.
  - On the transfer function $\Phi(x)$ with $x = q\eta$: derivative with respect to its own argument $x$. *Not stated; inferred* from the passage p.1→p.2, where $\Phi'_{\mathbf q}$ becomes $q\,\Phi'(q\eta)$ (factor $q$ from the chain rule), and from p.8 where $\Phi'(x)$ produces a factor $c_s$ from differentiating $J(xc_s)$.
  - On a Bessel function $Z'_\alpha(x)$: $d/dx$ (p.8).
- **$\mathcal H$.** $\mathcal H = a'/a$ (p.2: the homogeneous equation is written $h_s'' + 2\frac{a'}{a}h_s' + k^2 h_s = 0$, matching the $2\mathcal H h_s'$ of p.1). For fixed $w$: $\mathcal H = \frac{2}{1+3w}\frac1\eta$ (p.1). No $\epsilon$ is used.
- **Scale-factor normalisation.** Only the ratio $a(\eta')/a(\eta)$ ever appears (p.3 onward). On p.10 the ratio is evaluated as $(\eta')^{1+b}/\eta^{1+b}$, i.e. $a \propto \eta^{1+b}$ for fixed $w$ (*used, not stated separately*). No $a_0$ appears.
- **Green's function.** Defined for the rescaled variable $\chi_s = a\,h_s$ (p.3), with **$+\delta$ source**: $\mathrm{Gr}''(\eta,\eta') + \left(k^2 - \frac{a''}{a}\right)\mathrm{Gr}_{\mathbf k}(\eta,\eta') = \delta(\eta-\eta')$ (p.3). First argument $\eta$ is the response time, second argument $\eta'$ is the source time (p.3: $h_s = \int_{\eta_0}^{\eta} d\eta'\,4\frac{a(\eta')}{a(\eta)}\mathrm{Gr}_{\mathbf k}(\eta,\eta')\cdots$). Retarded: $\mathrm{Gr} = 0$ for $\eta < \eta'$ (p.4). Fixed-$w$ closed form on p.4 (R14). $\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}$ (p.6). No relation to a "literature" Green's function is stated.
- **Equation of state.** $w$ constant ("an epoch of fixed $w$", p.1). $b = \frac{1-3w}{1+3w}$ (p.4, p.7). Derived relations (p.7): $1+3w = \frac{2}{1+b}$, $1+w = \frac{2(2+b)}{3(1+b)}$, $5+3w = \frac{2(3+2b)}{1+b}$. A sound speed $c_s$ appears in the transfer function (p.4, p.8) but **$c_s$ is never defined in this document**; the relation $c_s^2 = (1-b)/(3(1+b))$ is *not* stated.
- **Fourier / power-spectrum conventions.**
  - Loop measure $\int \frac{d^3q}{(2\pi)^3}$ (p.1 onward).
  - Two-point function of the primordial variable: from p.5, $\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle = (2\pi)^3\,\delta(\mathbf q+\mathbf t)\,P_*(q)$. *Not written as a standalone definition; inferred* from the appearance of $(2\pi)^6 P_*(\mathbf q)P_*(\mathbf k-\mathbf q)\,\delta(\mathbf q+\mathbf t)\,\delta(\ldots)$ after Wick contraction.
  - Tensor spectrum: $\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\,\delta(\mathbf k+\mathbf k')\,P^h_{22}(\mathbf k)$ (p.6), with the result $\propto\delta_{ss'}$ and $P^h_{22}$ "for any $s$" (p.6) — i.e. a **per-polarisation** spectrum, no sum over $s$.
  - Both $P_*$ and $P^h_{22}$ are **dimensionful** $P(k)$; no dimensionless $\mathcal P(k) = k^3P/2\pi^2$ appears anywhere in the document.
  - Polarisation tensors (p.3): $e^{lm}_+(\mathbf k) = \frac{1}{\sqrt2}(e^l e^m - \bar e^l\bar e^m)$, $e^{lm}_\times(\mathbf k) = \frac{1}{\sqrt2}(e^l\bar e^m + \bar e^l e^m)$, normalised so that $e^{lm}_s e_{s'\,lm} = \delta_{ss'}$. Here $e,\bar e$ are the two unit vectors orthogonal to $\mathbf k$ (p.6: "$e,\bar e \perp \mathbf k'$").
  - Polarisation factor $Q_s(\mathbf k,\mathbf q) = e^{lm}_s(\mathbf k)q_l q_m$ (p.3), $Q_+ = \frac{q^2}{\sqrt2}\sin^2\theta\cos2\phi$, $Q_\times = \frac{q^2}{\sqrt2}\sin^2\theta\sin2\phi$ (p.4), with $\theta,\phi$ the polar angles of $\mathbf q$ about $\mathbf k$ (*angles not defined explicitly; inferred*).
- **Transfer function.** Newtonian-gauge-type potential $\Phi$ (the author calls it "the transfer function for $\Phi$", p.8; the same symbol is sometimes written with a lower-case-looking $\phi$ glyph, see Open questions). Normalisation at early times: $\Phi_{\mathbf q}(\eta) = \frac{3(1+w)}{5+3w}\,\Phi(q\eta)\,\zeta^*_{\mathbf q}$ with $\Phi(q\eta)\to1$ as $q\eta\to0$ (p.1). The primordial variable is $\zeta^*$ (p.10: "using $\Phi^*$ rather than $\zeta^*$ as we do"). The star on $\zeta^*$ and $P_*$ is a label for the primordial value, not complex conjugation (*inferred*; see Open questions). Fixed-$w$ closed form $\Phi(x) = 2^{3/2+b}\Gamma(\tfrac52+b)(xc_s)^{-3/2-b}J_{3/2+b}(xc_s)$ (p.4, p.8).
- **Momentum labels.** External $\mathbf k$; loop momentum $\mathbf q$; $\mathbf r = \mathbf k - \mathbf q$ (p.1). For the second copy of $h$: external $\mathbf k'$, loop $\mathbf t$, and $\mathbf u = \mathbf k' - \mathbf t$ (p.4, implicit). Vectors are underlined on the page; transcribed here as bold. After the $\delta(\mathbf k+\mathbf k')$ is extracted the loop integral is left as $\int\frac{d^3q}{(2\pi)^3}$ (p.6): **no reduction to $|\mathbf q|$, $\cos\theta$ variables is performed in this document**. The only angular statement is that the $\phi$-integral over $2\pi$ gives $\delta_{ss'}$ when $f$ is $\phi$-independent (p.7).
- **Symmetry factor.** The two connected Wick pairings contribute equally, giving $2\times16 = 32$ (p.5–6).

---

## 3. Results

Notation: $\mathbf r \equiv \mathbf k - \mathbf q$, $r = |\mathbf r|$, $q = |\mathbf q|$, $k = |\mathbf k|$. Page references are to `MAIN` 14. Underlined vectors on the page are written in bold.

### Step 1 — sourced equation (pp.1–2)

**R1** — MAIN 14 p.1. Sourced tensor equation for a single polarisation (stated as given; not derived here):
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s = 4\int\frac{d^3q}{(2\pi)^3}\,e^{lm}_s(\mathbf k)\,q_l q_m
\left\{\frac{5+3w}{3(1+w)}\,\Phi_{\mathbf q}\Phi_{\mathbf k-\mathbf q}
+ \frac{2}{3(1+w)}\left(\frac{\Phi_{\mathbf q}\Phi'_{\mathbf k-\mathbf q} + \Phi'_{\mathbf q}\Phi_{\mathbf k-\mathbf q}}{\mathcal H}
+ \frac{\Phi'_{\mathbf q}\Phi'_{\mathbf k-\mathbf q}}{\mathcal H^2}\right)\right\}.
$$
Starting point; primes are $d/d\eta$. Confidence: **high**.

**R2** — MAIN 14 p.1. Linear solution for the potential in terms of the primordial variable:
$$
\Phi_{\mathbf q}(\eta) = \frac{3(1+w)}{5+3w}\,\Phi(q\eta)\,\zeta^*_{\mathbf q},\qquad \Phi(q\eta)\to1 \text{ as } q\eta\to0 .
$$
Defines the transfer function $\Phi(x)$ and its early-time normalisation. Confidence: **high**.

**R3** — MAIN 14 p.1. Momentum label and fixed-$w$ Hubble rate:
$$
\mathbf r = \mathbf k - \mathbf q,\qquad \mathcal H = \frac{2}{1+3w}\,\frac1\eta .
$$
Confidence: **high**.

**R4** — MAIN 14 p.1 (bottom). After substituting R2 into R1:
$$
h_s'' + 2\mathcal H h_s' + k^2 h_s = 4\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}
\left\{\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta)
+ \frac{6(1+w)}{(5+3w)^2}\left(\frac{r}{\mathcal H}\Phi(q\eta)\Phi'(r\eta) + \frac{q}{\mathcal H}\Phi'(q\eta)\Phi(r\eta) + \frac{qr}{\mathcal H^2}\Phi'(q\eta)\Phi'(r\eta)\right)\right\}.
$$
Here and below $\Phi'(x) = d\Phi/dx$. Confidence: **high**.

**R5** — MAIN 14 p.2. Same equation with $\mathcal H$ from R3 inserted (final form of Step 1):
$$
h_s'' + \frac{4}{1+3w}\frac{1}{\eta}h_s' + k^2 h_s = 4\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}
\left\{\frac{3(1+w)}{5+3w}\Phi(q\eta)\Phi(r\eta)
+ \frac{3(1+w)(1+3w)}{(5+3w)^2}\left(r\eta\,\Phi(q\eta)\Phi'(r\eta) + q\eta\,\Phi'(q\eta)\Phi(r\eta) + \frac{1+3w}{2}\,q\eta\,r\eta\,\Phi'(q\eta)\Phi'(r\eta)\right)\right\}.
$$
(An intermediate line on p.2 with explicit $\frac{1+3w}{2}$ factors on each term and $\frac{(1+3w)^2}{4}$ on the last is not transcribed.) Confidence: **high**.

### Step 2 — Green's function formula (pp.2–3)

**R6** — MAIN 14 p.2. Homogeneous tensor equation:
$$
h_s'' + 2\frac{a'}{a}h_s' + k^2 h_s = 0 .
$$
Confidence: **high**.

**R7** — MAIN 14 p.3. Rescaling and rescaled equation. With $h_s(\eta) = \chi_s(\eta)/a(\eta)$,
$$
\chi_s'' + \left(k^2 - \frac{a''}{a}\right)\chi_s = 4a(\eta)\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,f(\mathbf q,\mathbf r,\eta).
$$
Confidence: **high**.

**R8** — MAIN 14 p.3. **Definition of the source function** $f$ ($w$-form):
$$
f(\mathbf q,\mathbf r,\eta) = \frac{3(1+w)}{5+3w}\,\Phi(q\eta)\Phi(r\eta)
+ \frac{3(1+w)(1+3w)}{(5+3w)^2}\left(r\eta\,\Phi(q\eta)\Phi'(r\eta) + q\eta\,\Phi'(q\eta)\Phi(r\eta) + \frac{1+3w}{2}\,q\,r\,\eta^2\,\Phi'(q\eta)\Phi'(r\eta)\right).
$$
Note $f$ depends on $\mathbf q,\mathbf r$ only through the magnitudes $q,r$. Confidence: **high**.

**R9** — MAIN 14 p.3. Green's function equation for $\chi_s$ (source $+\delta$):
$$
\mathrm{Gr}''(\eta,\eta') + \left(k^2 - \frac{a''}{a}\right)\mathrm{Gr}_{\mathbf k}(\eta,\eta') = \delta(\eta-\eta').
$$
Confidence: **high**.

**R10** — MAIN 14 p.3. Green's function solution for $h_s$:
$$
h_s = \int_{\eta_0}^{\eta} d\eta'\;4\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\,f(\mathbf q,\mathbf r,\eta').
$$
($\mathrm{Gr}_{\mathbf k}(\eta,\eta')$ was inserted into this line with an arrow — see §5.) Confidence: **high**.

### Step 3 — ingredients (pp.3–4)

**R11** — MAIN 14 p.3. Polarisation factors and tensors:
$$
Q_s(\mathbf k,\mathbf q) = e^{lm}_s(\mathbf k)\,q_l q_m,\qquad
e^{lm}_+(\mathbf k) = \frac{1}{\sqrt2}\left(e^l e^m - \bar e^l\bar e^m\right),\qquad
e^{lm}_\times(\mathbf k) = \frac{1}{\sqrt2}\left(e^l\bar e^m + \bar e^l e^m\right),\qquad
e^{lm}_s\,e_{s'\,lm} = \delta_{ss'} .
$$
Confidence: **high**.

**R12** — MAIN 14 p.4. Explicit angular form of the polarisation factors:
$$
Q_+(\mathbf k,\mathbf q) = \frac{q^2}{\sqrt2}\sin^2\theta\cos2\phi,\qquad
Q_\times(\mathbf k,\mathbf q) = \frac{q^2}{\sqrt2}\sin^2\theta\sin2\phi .
$$
Confidence: **high** (angles $\theta,\phi$ not defined on the page; presumably polar angles of $\mathbf q$ relative to $\mathbf k$ and the $e$-axis).

**R13** — MAIN 14 p.4. Transfer function, fixed $w$:
$$
\Phi(k\eta) = 2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)\,(kc_s\eta)^{-3/2-b}\,J_{3/2+b}(kc_s\eta),\qquad b = \frac{1-3w}{1+3w}.
$$
Confidence: **high**.

**R14** — MAIN 14 p.4. Green's function, fixed $w$:
$$
\mathrm{Gr}_{\mathbf k}(\eta,\eta') = \frac{\pi}{2}\sqrt{\eta\eta'}
\begin{cases}
J_{b+1/2}(k\eta')\,Y_{b+1/2}(k\eta) - Y_{b+1/2}(k\eta')\,J_{b+1/2}(k\eta) & \text{if } \eta > \eta' \\[2pt]
0 & \text{if } \eta < \eta'.
\end{cases}
$$
Confidence: **high**.

### Step 4 — the 22 power spectrum (pp.4–7)

**R15** — MAIN 14 p.4. Two-point function before contraction:
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = 16\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,
\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,\mathrm{Gr}_{\mathbf k'}(\eta,\eta'')\,\frac{a(\eta')}{a(\eta)}\frac{a(\eta'')}{a(\eta)}
\times\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,Q_{s'}(\mathbf k',\mathbf t)\,
\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\zeta^*_{\mathbf t}\zeta^*_{\mathbf k'-\mathbf t}\rangle\,
f(\mathbf q,\mathbf r,\eta')\,f(\mathbf t,\mathbf u,\eta'').
$$
Confidence: **high** ($\mathbf u$ is not defined on the page; by analogy with $\mathbf r$ it is $\mathbf k'-\mathbf t$).

**R16** — MAIN 14 p.5. Wick contraction:
$$
\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf k-\mathbf q}\zeta^*_{\mathbf t}\zeta^*_{\mathbf k'-\mathbf t}\rangle
= \langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle\langle\zeta^*_{\mathbf k-\mathbf q}\zeta^*_{\mathbf k'-\mathbf t}\rangle
+ \langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf k'-\mathbf t}\rangle\langle\zeta^*_{\mathbf k-\mathbf q}\zeta^*_{\mathbf t}\rangle
+ \text{disconnected}.
$$
Confidence: **high**.

**R17** — MAIN 14 p.5. After inserting the two-point functions (implicitly $\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle = (2\pi)^3\delta(\mathbf q+\mathbf t)P_*(q)$):
$$
\langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = 16\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,\frac{a(\eta')a(\eta'')}{a^2(\eta)}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,\mathrm{Gr}_{\mathbf k'}(\eta,\eta'')
\int\frac{d^3q}{(2\pi)^3}\frac{d^3t}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k',\mathbf t)\,(2\pi)^6 P_*(\mathbf q)P_*(\mathbf k-\mathbf q)\,
f(\mathbf q,\mathbf k-\mathbf q,\eta')f(\mathbf t,\mathbf k'-\mathbf t,\eta'')
\left\{\delta(\mathbf q+\mathbf t)\,\delta(\mathbf k-\mathbf q+\mathbf k'-\mathbf t) + \delta(\mathbf q+\mathbf k'-\mathbf t)\,\delta(\mathbf k-\mathbf q+\mathbf t)\right\}
+ \text{disconnected}
$$
$$
= 16\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,\frac{a(\eta')a(\eta'')}{a(\eta)^2}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,\mathrm{Gr}_{\mathbf k'}(\eta,\eta'')
\int\frac{d^3q}{(2\pi)^3}\,(2\pi)^3\delta(\mathbf k+\mathbf k')\,P_*(\mathbf q)P_*(\mathbf k-\mathbf q)
\Big\{Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k',-\mathbf q)\,f(\mathbf q,\mathbf k-\mathbf q,\eta')f(-\mathbf q,\mathbf k'+\mathbf q,\eta'')
+ Q_s(\mathbf k,\mathbf q)Q_{s'}(\mathbf k',\mathbf k'+\mathbf q)\,f(\mathbf q,\mathbf k-\mathbf q,\eta')f(\mathbf k'+\mathbf q,-\mathbf q,\eta'')\Big\}
+ \text{disconnected}.
$$
Confidence: **high** (one struck factor in the second line, see §5).

**R18** — MAIN 14 p.6. Simplifying identities and definition of $P^h_{22}$. "We always want to drop the disconnected contributions." Then
$$
\text{(i)}\;\; Q_{s'}(\mathbf k',\mathbf k'+\mathbf q) = Q_{s'}(\mathbf k',\mathbf q)\;\;\text{since } e,\bar e\perp\mathbf k';\qquad
\text{(ii)}\;\; f(\mathbf q,\mathbf r,\eta)\equiv f(\mathbf r,\mathbf q,\eta),
$$
$$
\Rightarrow\quad \langle h_s(\mathbf k)h_{s'}(\mathbf k')\rangle = (2\pi)^3\,\delta(\mathbf k+\mathbf k')\,P^h_{22}(\mathbf k),
$$
where
$$
P^h_{22}(\mathbf k) = 32\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,\frac{a(\eta')a(\eta'')}{a(\eta)^2}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,\mathrm{Gr}_{-\mathbf k}(\eta,\eta'')
\int\frac{d^3q}{(2\pi)^3}\,P_*(\mathbf q)P_*(\mathbf k-\mathbf q)\,Q_s(\mathbf k,\mathbf q)\,Q_{s'}(-\mathbf k,-\mathbf q)\,
f(\mathbf q,\mathbf k-\mathbf q,\eta')\,f(-\mathbf q,\mathbf q-\mathbf k,\eta'').
$$
Defines $P^h_{22}$ and fixes the $(2\pi)^3\delta$ convention for the tensor spectrum. Confidence: **medium** — a struck or over-written glyph sits between the comma and $\mathbf q-\mathbf k$ in the last argument list of $f$ (read as a deleted false start; see Open questions Q1). All other symbols **high**.

**R19** — MAIN 14 p.6. Further identities and the $\delta_{ss'}$ form:
$$
Q_{s'}(-\mathbf k,-\mathbf q)\equiv Q_{s'}(\mathbf k,\mathbf q),\qquad
f(\mathbf q,\mathbf k-\mathbf q,\eta')\equiv f(-\mathbf q,\mathbf q-\mathbf k,\eta'),\qquad
\mathrm{Gr}_{\mathbf k} = \mathrm{Gr}_{-\mathbf k}
$$
$$
\Rightarrow\quad P^h_{22}(\mathbf k) = 32\int_{\eta_0}^{\eta}d\eta'\int_{\eta_0}^{\eta}d\eta''\,\frac{a(\eta')a(\eta'')}{a(\eta)^2}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,\mathrm{Gr}_{\mathbf k}(\eta,\eta'')\;
\delta_{ss'}\int\frac{d^3q}{(2\pi)^3}\,P_*(\mathbf q)P_*(\mathbf k-\mathbf q)\,Q_s(\mathbf k,\mathbf q)^2\,f(\mathbf q,\mathbf k-\mathbf q,\eta')\,f(\mathbf q,\mathbf k-\mathbf q,\eta'').
$$
The $\delta_{ss'}$ is justified by R21. Confidence: **high**.

**R20** — MAIN 14 p.6 (bottom). **Factorised form of the one-loop spectrum** (final form of Step 4; the structural build target):
$$
P^h_{22}(\mathbf k) = 32\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)^2\,P_*(\mathbf q)\,P_*(\mathbf k-\mathbf q)
\left(\int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,f(\mathbf q,\mathbf k-\mathbf q,\eta')\right)^2
\qquad\text{"for any } s\text{."}
$$
The double time integral factorises into the square of a single time integral; this single integral is what Step 6 evaluates. Confidence: **high**.

**R21** — MAIN 14 p.7. Angular note:
$$
\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)\,Q_{s'}(\mathbf k,\mathbf q)\cdots \;\propto\; \delta_{ss'}
$$
"from integration of the $\phi$ dependence over $2\pi$ **if** $f$ is independent of $\phi$." Confidence: **medium** — an unexplained glyph resembling "e" sits between the measure and $Q_s$ (Open questions Q2); the statement itself is clear.

### Step 5 — source function in terms of $b$ (pp.7–10)

**R22** — MAIN 14 p.7. Relations between $w$ and $b$:
$$
1+3w = \frac{2}{1+b},\qquad b = \frac{1-3w}{1+3w},\qquad 1+w = \frac{2(2+b)}{3(1+b)},\qquad
5+3w = 5 + \frac{2}{1+b} - 1 = 4 + \frac{2}{1+b} = \frac{6+4b}{1+b} = \frac{2(3+2b)}{1+b}.
$$
Confidence: **high**.

**R23** — MAIN 14 p.7 (bottom). Source function in $b$-form:
$$
f(\mathbf q,\mathbf r,\eta) = \frac{2+b}{3+2b}\,\Phi(q\eta)\Phi(r\eta)
+ \frac{2+b}{(3+2b)^2}\left\{r\eta\,\Phi(q\eta)\Phi'(r\eta) + q\eta\,\Phi'(q\eta)\Phi(r\eta) + \frac{1}{1+b}\,q\,r\,\eta^2\,\Phi'(q\eta)\Phi'(r\eta)\right\}.
$$
Confidence: **high** (an intermediate line above it, not transcribed, contains an apparent stray "+", Open questions Q3).

**R24** — MAIN 14 p.8. Transfer function and its derivative:
$$
\Phi(x) = 2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)(xc_s)^{-3/2-b}J_{3/2+b}(xc_s),
$$
$$
\Phi'(x) = 2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)(xc_s)^{-3/2-b}\,c_s\left\{J'_{3/2+b}(xc_s) - \left(\tfrac32+b\right)\frac{1}{xc_s}J_{3/2+b}(xc_s)\right\}
= 2^{3/2+b}\,\Gamma\!\left(\tfrac52+b\right)(xc_s)^{-3/2-b}\,c_s\,(-1)\,J_{5/2+b}(xc_s).
$$
Confidence: **high**.

**R25** — MAIN 14 pp.8–9. Bessel identities used ($Z_\alpha$ any Bessel function):
$$
2Z'_\alpha(x) = Z_{\alpha-1}(x) - Z_{\alpha+1}(x),\qquad
\frac{2\alpha}{x}Z_\alpha(x) = Z_{\alpha-1}(x) + Z_{\alpha+1}(x),
$$
$$
Z'_\alpha(x) - \frac{\alpha}{x}Z_\alpha(x) = -Z_{\alpha+1}(x),\qquad
Z_\alpha(x) - \frac{x}{2\alpha}Z_{\alpha+1}(x) = \frac{x}{2\alpha}Z_{\alpha-1}(x).
$$
Confidence: **high**.

**R26** — MAIN 14 p.8 (bottom). Source function in Bessel form, four terms:
$$
f(\mathbf q,\mathbf r,\eta) = \frac{2+b}{3+2b}\,2^{3+2b}\,\Gamma^2\!\left(\tfrac52+b\right)(q\eta c_s)^{-3/2-b}(r\eta c_s)^{-3/2-b}
\Big\{J_{3/2+b}(q\eta c_s)J_{3/2+b}(r\eta c_s)
- \frac{r\eta c_s}{3+2b}J_{3/2+b}(q\eta c_s)J_{5/2+b}(r\eta c_s)
- \frac{q\eta c_s}{3+2b}J_{5/2+b}(q\eta c_s)J_{3/2+b}(r\eta c_s)
+ \frac{1}{(3+2b)(1+b)}(q\eta c_s)(r\eta c_s)J_{5/2+b}(q\eta c_s)J_{5/2+b}(r\eta c_s)\Big\}.
$$
Confidence: **medium** — the order of the first Bessel function in the third term is over-written (a "3" and a "5" are both visible in the index); read as $J_{5/2+b}(q\eta c_s)$ by symmetry with the second term and consistency with R24. All else **high**.

(p.9 "completes the square": $f = \frac{2+b}{3+2b}2^{3+2b}\Gamma(\tfrac52+b)^2(q\eta c_s)^{-3/2-b}(r\eta c_s)^{-3/2-b}\{(J_{3/2+b}(q\eta c_s) - \frac{q\eta c_s}{3+2b}J_{5/2+b}(q\eta c_s))(J_{3/2+b}(r\eta c_s) - \frac{r\eta c_s}{3+2b}J_{5/2+b}(r\eta c_s)) + [\frac{1}{(3+2b)(1+b)} - \frac{1}{(3+2b)^2}](q\eta c_s)(r\eta c_s)J_{5/2+b}J_{5/2+b}\}$, then applies the last identity of R25. Intermediate algebra; two slips on this page are listed in Open questions Q4, Q5.)

**R27** — MAIN 14 p.9 (bottom) and p.10 (top). **Final Bessel form of the source function** (end of Step 5):
$$
f(\mathbf q,\mathbf r,\eta) = \frac{2+b}{(3+2b)^3}\,2^{3+2b}\,\Gamma\!\left(\tfrac52+b\right)^2\,(q\eta c_s)^{-1/2-b}(r\eta c_s)^{-1/2-b}
\left\{J_{1/2+b}(q\eta c_s)J_{1/2+b}(r\eta c_s) + \frac{2+b}{1+b}\,J_{5/2+b}(q\eta c_s)J_{5/2+b}(r\eta c_s)\right\}.
$$
On p.9 the coefficient of the second term is written $\left(\frac{3+2b}{1+b} - 1\right)$; on p.10 it is written $\frac{2+b}{1+b}$ (equal). Confidence: **high**.

**R28** — MAIN 14 p.10. Literature check (text): "Matches Domènech Eq. (4.9) up to normalization, which appears to come from him using $\Phi^*$ rather than $\zeta^*$ as we do." Confidence: **high**.

### Step 6 — collect the time integral (pp.10–11) — TARGET

The quantity evaluated is the single time integral that appears squared in R20:
$$
\mathcal I \equiv \int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,f(\mathbf q,\mathbf r,\eta').
$$
(The page does not give this integral a name; $\mathcal I$ is the transcriber's label.)

**R29** — MAIN 14 p.10. Time integral with R14, R27 and $a(\eta')/a(\eta) = (\eta'/\eta)^{1+b}$ inserted:
$$
\mathcal I = \int_{\eta_0}^{\eta}d\eta'\,\frac{(\eta')^{1+b}}{\eta^{1+b}}\,\frac{\pi}{2}\sqrt{\eta\eta'}
\Big(J_{b+1/2}(k\eta')Y_{b+1/2}(k\eta) - Y_{b+1/2}(k\eta')J_{b+1/2}(k\eta)\Big)
\frac{2+b}{(3+2b)^3}\,2^{3+2b}\,\Gamma\!\left(\tfrac52+b\right)^2(q\eta' c_s)^{-1/2-b}(r\eta' c_s)^{-1/2-b}
\left\{J_{1/2+b}(q\eta' c_s)J_{1/2+b}(r\eta' c_s) + \frac{2+b}{1+b}J_{5/2+b}(q\eta' c_s)J_{5/2+b}(r\eta' c_s)\right\}
$$
$$
= \pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\,\eta^{-1/2-b}
\int_{\eta_0}^{\eta}d\eta'\,(\eta')^{3/2+b}\,(q\eta' c_s)^{-1/2-b}(r\eta' c_s)^{-1/2-b}
\Big(J_{b+1/2}(k\eta')Y_{b+1/2}(k\eta) - Y_{b+1/2}(k\eta')J_{b+1/2}(k\eta)\Big)
\Big(J_{1/2+b}(q\eta' c_s)J_{1/2+b}(r\eta' c_s) + \frac{2+b}{1+b}J_{5/2+b}(q\eta' c_s)J_{5/2+b}(r\eta' c_s)\Big).
$$
Confidence: **high** for all exponents and indices. Two page-level corrections are folded in and recorded in §5: the factor $(r\eta' c_s)^{-1/2-b}$ is written with $k$ over-written by $r$; the exponent $2+2b$ is written as $2+3b$ with "3b" struck and "2b" written above. In the first line the last two Bessel orders are written $J_{5/2}$ without "$+b$" (Open questions Q6); they are $J_{5/2+b}$ in the second line and on p.11.

**R30 — `TARGET`** — MAIN 14 p.11 (top). **Endpoint of Step 6, before conversion to Fabrikant form.** Verbatim from the page:
$$
\boxed{\;
\int_{\eta_0}^{\eta}d\eta'\,\frac{a(\eta')}{a(\eta)}\,\mathrm{Gr}_{\mathbf k}(\eta,\eta')\,f(\mathbf q,\mathbf r,\eta')
= \pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\,\left(c_s^2\,q\,r\,\eta\right)^{-1/2-b}
\Big(Y_{b+1/2}(k\eta)\,I_J - J_{b+1/2}(k\eta)\,I_Y\Big)
\;}
$$
with
$$
\boxed{\;
I_{J/Y} = \int_{\eta_0}^{\eta}d\eta'\,(\eta')^{1/2-b}
\left\{\begin{matrix}J_{1/2+b}(k\eta')\\ Y_{1/2+b}(k\eta')\end{matrix}\right\}
\left(J_{1/2+b}(q\eta' c_s)\,J_{1/2+b}(r\eta' c_s) + \frac{2+b}{1+b}\,J_{5/2+b}(q\eta' c_s)\,J_{5/2+b}(r\eta' c_s)\right)
\;}
$$
where the upper (lower) entry in the brace defines $I_J$ ($I_Y$). The left-hand side is the transcriber's restatement of the integral being collected (p.10, first line of Step 6); the page writes only "$= \pi\,2^{2+2b}\cdots$" as a continuation of R29.

Reading notes for the TARGET:
- Exponent of 2: written $2^{2+3b}$ with "3b" struck through and "2b" written above; the corrected value $2+2b$ is used (and is what R29's algebra produces: $\tfrac{\pi}{2}\cdot2^{3+2b} = \pi\,2^{2+2b}$).
- Exponent of $(c_s^2 q r\eta)$: $-\tfrac12 - b$. Consistent with R29: $(q\eta' c_s)^{-1/2-b}(r\eta' c_s)^{-1/2-b}\eta^{-1/2-b} = (c_s^2 qr\eta)^{-1/2-b}(\eta')^{-1-2b}$, and $(\eta')^{3/2+b}(\eta')^{-1-2b} = (\eta')^{1/2-b}$, which is the power in $I_{J/Y}$.
- Sign and ordering of the two terms: $Y_{b+1/2}(k\eta)I_J$ **minus** $J_{b+1/2}(k\eta)I_Y$, following the ordering $J(k\eta')Y(k\eta) - Y(k\eta')J(k\eta)$ of R14.
- Bessel orders: $b+\tfrac12$ (Green's function, argument $k\eta$ or $k\eta'$); $\tfrac12+b$ and $\tfrac52+b$ (source, arguments $q\eta' c_s$ and $r\eta' c_s$). The page writes $b+\tfrac12$ for the Green's-function orders and $\tfrac12+b$ for the source orders; they are the same number.
- Relative coefficient of the second source term: $\frac{2+b}{1+b}$.

Confidence: **high** for every glyph in the TARGET; the only over-writing (the $2^{2+2b}$ exponent) is unambiguous once struck text is disregarded and is confirmed by the algebra.

**Assembled build target (transcriber's assembly of R20 + R30; this combined line does NOT appear on any page):**
$$
P^h_{22}(\mathbf k) = 32\int\frac{d^3q}{(2\pi)^3}\,Q_s(\mathbf k,\mathbf q)^2\,P_*(\mathbf q)\,P_*(\mathbf k-\mathbf q)\,
\left[\pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\left(c_s^2 qr\eta\right)^{-1/2-b}
\Big(Y_{b+1/2}(k\eta)I_J - J_{b+1/2}(k\eta)I_Y\Big)\right]^2 .
$$
Note that R30 uses the fixed-$w$ analytic forms of $a(\eta)$, $\mathrm{Gr}_{\mathbf k}$ (R14) and $\Phi$ (R13); the general-cosmology structure is R20 with R8 for $f$.

### Step 7 — Fabrikant form (pp.11–12) — NOT TO BE USED

Transcribed so that the audit can see what is excluded. Implicit convention: $J_{n+1/2}(x) = \sqrt{2x/\pi}\,j_n(x)$, $Y_{n+1/2}(x) = \sqrt{2x/\pi}\,y_n(x)$ (not written on the page; inferred from the prefactors).

**R31** — MAIN 14 p.11. "First we wish to exchange the Bessel functions for spherical Bessel functions. ⇒ time integral $I_{J/Y}$ factor is"
$$
I_{J/Y} = \left(\frac{2}{\pi}\right)^{3/2}\int_{\eta_0}^{\eta}d\eta'\,(\eta')^{1/2-b}\sqrt{k\eta'}
\left\{\begin{matrix}j_b(k\eta')\\ y_b(k\eta')\end{matrix}\right\}
(qrc_s^2)^{1/2}\,\eta'\left(j_b(q\eta' c_s)j_b(r\eta' c_s) + \frac{2+b}{1+b}\,j_{2+b}(q\eta' c_s)j_{2+b}(r\eta' c_s)\right)
$$
$$
= \left(\frac{2}{\pi}\right)^{3/2}(kqrc_s^2)^{1/2}\int_{\eta_0}^{\eta}d\eta'\,(\eta')^{2-b}
\left\{\begin{matrix}j_b(k\eta')\\ y_b(k\eta')\end{matrix}\right\}
\left(j_b(q\eta' c_s)j_b(r\eta' c_s) + \frac{2+b}{1+b}\,j_{2+b}(q\eta' c_s)j_{2+b}(r\eta' c_s)\right).
$$
Confidence: **high**.

**R32** — MAIN 14 p.12. Definition of the spherical-Bessel time integrals:
$$
I_{j,y} = \int_{\eta_0}^{\eta}d\eta'\,(\eta')^{2-b}
\left\{\begin{matrix}j_b(k\eta')\\ y_b(k\eta')\end{matrix}\right\}
\left(j_b(q\eta' c_s)j_b(r\eta' c_s) + \frac{2+b}{1+b}\,j_{2+b}(q\eta' c_s)j_{2+b}(r\eta' c_s)\right).
$$
Confidence: **medium** — the sign in the exponent $(\eta')^{2-b}$ is over-written (looks like "$\mp$"); read as $2-b$ from R31 (Open questions Q9).

**R33** — MAIN 14 p.12. Time integral in Fabrikant form. "Then the time integral can be written"
$$
\frac{4}{\pi^2}\,\pi\,2^{2+2b}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)^2\left(c_s^2qr\eta\right)^{-1/2-b}\cdot\left(qrk^2c_s^2\right)^{1/2}\eta^{1/2}
\Big(y_b(k\eta)I_j \mp j_b(k\eta)I_y\Big)
$$
$$
= \frac{2^{4+2b}}{\pi}\,\frac{2+b}{(3+2b)^3}\,\Gamma\!\left(\tfrac52+b\right)\,k\left(c_s^2qr\eta\right)^{-b}
\Big(y_b(k\eta)I_j - j_b(k\eta)I_y\Big).
$$
Confidence: **medium**. (a) In the first line the sign between the two terms is over-written ("$\mp$" appearance); the second line has a clear "$-$". (b) In the second line $\Gamma(\tfrac52+b)$ appears **without the square** that every preceding line carries — transcribed as written; see Open questions Q7. (c) In the second line the "$j_b$" of the second term is over-written, with a "j" written beneath a struck letter; read as $j_b(k\eta)I_y$. Everything else **high**. Not to be used in any case (README §2).

---

## 4. Checks

`MAIN` 14 is described in the README as a recheck of `MAIN` 11. On the pages themselves:

- **No explicit reference to `MAIN` 11** (or to any earlier calculation number) appears anywhere in the 12 pages, and no discrepancy with an earlier result is noted. The document re-derives $P^h_{22}$ from the sourced equation (R1) rather than comparing line-by-line.
- The only comparison recorded is against the literature (p.10, R28): the Bessel form of the source function $f$ (R27) "matches Domènech Eq. (4.9) up to normalization", the normalisation difference being attributed to Domènech's use of $\Phi^*$ rather than $\zeta^*$ as the primordial variable.
- This transcriber has not read `MAIN` 11 and can make no statement about agreement between the two documents.

Internal consistency checks performed by the transcriber (arithmetic only, not physics):
- R4→R5: $\frac{2}{3(1+w)}\left(\frac{3(1+w)}{5+3w}\right)^2 = \frac{6(1+w)}{(5+3w)^2}$ and $\frac{6(1+w)}{(5+3w)^2}\cdot\frac{1+3w}{2} = \frac{3(1+w)(1+3w)}{(5+3w)^2}$: consistent.
- R23 from R8 with R22: consistent.
- R27 from R26 via the completed square and R25: consistent, including $(3+2b)^2\left[\frac{1}{(3+2b)(1+b)} - \frac{1}{(3+2b)^2}\right] = \frac{2+b}{1+b}$.
- R29→R30 prefactor and $\eta,\eta'$ powers: consistent (see reading notes under R30).
- R30→R31/R33 prefactors $(2/\pi)^{3/2}$, $(2/\pi)^2 = 4/\pi^2$, $k(c_s^2qr\eta)^{-b}$: consistent, except that the square on $\Gamma(\tfrac52+b)$ disappears in the last line of p.12 (Q7).

---

## 5. Corrections and cross-outs

| # | Page | What is written | Reading used by later steps |
|---|---|---|---|
| C1 | p.3 | In the $h_s$ formula (R10) the factor $\mathrm{Gr}_{\mathbf k}(\eta,\eta')$ is written above the line and inserted with a downward arrow between $4\frac{a(\eta')}{a(\eta)}$ and $\int\frac{d^3q}{(2\pi)^3}$. | Inserted factor is part of the formula (transcribed in R10). |
| C2 | p.5 | In the second display of R17, a factor "$Q_s(\mathbf k,\mathbf q)$" written immediately after $(2\pi)^3\delta(\mathbf k+\mathbf k')$ is struck through. | Struck; the $Q_s$ factors appear inside the braces instead. |
| C3 | p.8 | Third term of R26: the order of the first Bessel function is over-written; both "3" and "5" visible in the index numerator. | $J_{5/2+b}(q\eta c_s)J_{3/2+b}(r\eta c_s)$ (symmetric partner of the second term). Medium confidence. |
| C4 | p.10 | R29, first line: $(k\eta' c_s)^{-1/2-b}$ with a small "r" written above the "k". | $(r\eta' c_s)^{-1/2-b}$ (corrected value; consistent with p.9 and p.11). |
| C5 | p.10 and p.11 | Prefactor written $2^{2+3b}$ with "3b" struck through and "2b" written above — occurs twice (R29 second line, R30). | $2^{2+2b}$. Confirmed by algebra ($\tfrac\pi2\cdot2^{3+2b}$). |
| C6 | p.12 | R32: exponent of $\eta'$ written with an over-written sign, appearance "$2\mp b$". | $2-b$ (from R31, second line). Medium confidence. |
| C7 | p.12 | R33 first line: sign between $y_b(k\eta)I_j$ and $j_b(k\eta)I_y$ over-written, appearance "$\mp$". | "$-$" (second line of R33 is unambiguous). |
| C8 | p.12 | R33 second line: second term's "$j_b$" is over-written; a letter (possibly "y" or "g") is struck and "j" written beneath. | $j_b(k\eta)I_y$. |

None of the above affects the TARGET (R30) except C5, which is unambiguous.

---

## 6. Open questions

- **Q1 (p.6, R18).** In $f(-\mathbf q,\;\cdot\;\mathbf q-\mathbf k,\eta'')$ there is a small struck or over-written glyph between the comma and $\mathbf q-\mathbf k$ (could be a struck "$-$", "$k$" or "$\ddagger$"-like mark). Read as a deleted false start; the argument is $\mathbf q-\mathbf k$, as required by $\mathbf k' = -\mathbf k$ and as used on the next line.
- **Q2 (p.7, R21).** An isolated glyph resembling "e" (or "$\in$") sits between $\int\frac{d^3q}{(2\pi)^3}$ and $Q_s(\mathbf k,\mathbf q)$. Not interpretable; possibly an abandoned symbol. Does not affect the statement.
- **Q3 (p.7, intermediate line above R23).** The first term inside the braces reads "$r\eta\,\Phi(q\eta) + \Phi'(r\eta)$" with what looks like a stray "+" between the two factors. The final line (R23) has the product $r\eta\,\Phi(q\eta)\Phi'(r\eta)$, as does R8. Treated as a pen slip in intermediate algebra.
- **Q4 (p.9, intermediate).** After applying the completed square, the first term is written $\frac{(q\eta c_s)(k\eta c_s)}{(3+2b)^2}J_{1/2+b}(q\eta c_s)J_{1/2+b}(r\eta c_s)$ — a "$k$" where the structure requires "$r$". Not corrected on the page; the following line (R27) uses $(r\eta c_s)$. The same slip on p.10 *is* corrected (C4). Treated as a slip.
- **Q5 (p.9 bottom, R27).** The last Bessel function of the final line is written $J_{5/2}(r\eta c_s)$, without "$+b$"; the same term on p.10 (top) reads $J_{5/2+b}(r\eta c_s)$. Treated as an abbreviation/omission.
- **Q6 (p.10, R29 first line).** Both Bessel functions in the second source term are written $J_{5/2}$ without "$+b$"; the second line of R29 and R30 (p.11) have $J_{5/2+b}$. Treated as abbreviation. **Not** in the TARGET.
- **Q7 (p.12, R33 second line).** $\Gamma(\tfrac52+b)$ appears without the square that is present in every preceding line (R27, R29, R30, R33 first line), and the surrounding algebra ($\frac{4}{\pi^2}\pi2^{2+2b} = \frac{2^{4+2b}}{\pi}$, $k(c_s^2qr\eta)^{-b}$) does not remove a $\Gamma$. Apparent omission; transcribed as written. Lies in the excluded Fabrikant part, not the TARGET.
- **Q8 (p.12, R33).** Over-written sign (C7). Resolved as "$-$" by the second line.
- **Q9 (p.12, R32).** Over-written exponent sign (C6). Resolved as $2-b$ by R31.
- **Q10 (throughout).** Meaning of the star in $\zeta^*_{\mathbf q}$ and $P_*$. It is not defined on any page. Since $\langle\zeta^*_{\mathbf q}\zeta^*_{\mathbf t}\rangle\propto\delta(\mathbf q+\mathbf t)$ (p.5), the star cannot be complex conjugation of both factors; it is read as a label ("primordial / initial value"), consistent with p.10 ("$\Phi^*$ rather than $\zeta^*$").
- **Q11 (throughout).** $c_s$ is used (p.4, p.8–12) but never defined in this document; no relation between $c_s$ and $w$ or $b$ is given.
- **Q12 (p.5–6).** $P_*$ is written with a vector argument, $P_*(\mathbf q)$, $P_*(\mathbf k-\mathbf q)$. Whether it is to be read as a function of the magnitude only is not stated (isotropy would imply so).
- **Q13 (p.2, p.7, p.9).** The transfer function is written sometimes with a capital-looking $\Phi$ and sometimes with a lower-case-looking cursive $\phi$ glyph (e.g. p.7 bottom "$\phi(q\eta)\phi(r\eta)$", p.9). Read throughout as the same symbol $\Phi$.
- **Q14 (p.4, R15).** The symbol $\mathbf u$ in $f(\mathbf t,\mathbf u,\eta'')$ is not defined; read as $\mathbf u = \mathbf k'-\mathbf t$ by analogy with $\mathbf r = \mathbf k-\mathbf q$ (confirmed by p.5 where $f(\mathbf t,\mathbf k'-\mathbf t,\eta'')$ is written out).
- **Q15 (p.4, R12).** The angles $\theta,\phi$ in $Q_\pm$ are not defined on the page.
- **Q16 (p.6, R20).** No angular reduction of $\int\frac{d^3q}{(2\pi)^3}$ is performed anywhere in the document; $Q_s(\mathbf k,\mathbf q)^2$ retains its $\sin^4\theta\cos^22\phi$ (or $\sin^22\phi$) dependence, and the $\phi$-integral is only discussed qualitatively (R21). Any conversion to $(q, r)$ or $(q,\cos\theta)$ variables must come from elsewhere.
- **Q17 (throughout).** The lower limit $\eta_0$ of every time integral is never specified.
- **Q18 (§4).** The document is labelled a recheck of `MAIN` 11 in the README, but contains no explicit cross-reference to it.
