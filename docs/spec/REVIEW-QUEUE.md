# Review queue — spec transcription campaign

**Compiled:** 2026-09-07 by the orchestrator from `diff-03.md`, `diff-04.md`, `diff-05.md` and
`cross-spec-check.md`. **Campaign:** `prompts/spec-transcription/README.md`.

This is the list of things the author needs to read against the page images or decide, ordered by
consequence for the one-loop build. Everything not listed here was read identically by two
independent transcribers (where duplicated) or flagged `high` confidence by the single transcriber,
and needs only a light review.

Headline from the diffs: **no disagreement touches any final formula.** The Step 6 endpoint of
`MAIN` 14 (the build target), the `NUM` 06 p.7 analytic integral, the `NUM` 03 p.4 source term and
p.10 one-loop formula were all read glyph-for-glyph identically by both transcribers. The
disagreements that exist are single glyphs in intermediate lines.

| Pair | Aligned | Agree | Minor | Disagree | Final formulas |
|---|---|---|---|---|---|
| 03 (`NUM` 03) | 24 | 22 | 1 | 0 | agree |
| 04 (`NUM` 06) | 27 | 20 | 1 | 3 (one glyph question, 3 places) | agree |
| 05 (`MAIN` 14) | 35 | 33 | 1 | 1 (letter of seed variable) | agree |

---

## Tier 1 — decisions that determine the one-loop build

These are not transcription questions. They are places where the notes, read faithfully, leave a
choice open or contain a slip, and the code cannot be audited or built until the author decides.

### 1.1 Which redshift-space Green's function is "the" Green's function

Three numerical-branch documents define a $z$-space Green's function with three different
normalisations, and the analytic hand-off between `NUM` 06 and `MAIN` 14 does not close by a
factor of $a_0$ (`cross-spec-check.md` §1 B1, B2; §2 hand-off (i)).

| Document | Definition | Normalisation relative to $G_{\rm them}$ (the $\eta$-space retarded Green's function of `MAIN` 13/14) |
|---|---|---|
| `NUM` 02 p.3–4 (spec 02 R17–R18) | source $-\frac{1}{a_0 H}\delta(z-z')$, jump $[G_z] = -1/(a_0H(z'))$ | $-G_{\rm them}$ — **but** the cross-spec agent argues the Jacobian was applied without $\lvert\cdot\rvert$ and the sign should be $+$ |
| `NUM` 03 p.5–6 (spec 03 R25) | $\bar G = -G$, $d\bar G/dz\vert_{z=z'} = +1$ | $-a_0 H(z')\,G_{\rm them}$ |
| `NUM` 06 p.1 (spec 04 R2) | "$G_{\rm me} = -H(z')\,G_{\rm them}$", source $-\delta(z-z')$ | $-H(z')\,G_{\rm them}$ (no $a_0$) |

**Consequence.** With $c^2 \equiv (2+b)^2/(3+2b)^2 = \big(3(1+w)/(5+3w)\big)^2$, the cross-spec
agent finds

$$\text{spec 04 R14 (NUM 06 p.7)} \;=\; -\frac{a_0}{c^2}\times\text{TARGET (MAIN 14 p.11)}.$$

The $-1$ (bar convention) and the $1/c^2$ (placement of the $3(1+w)/(5+3w)$ factor inside $f$ in
`MAIN` 14 vs outside in `NUM` 03/06) are reconciled. The power of $a_0$ is not: if $G_{\rm me}$ is
`NUM` 03's $\bar G$, the relation should carry $a_0^2$, which would then cancel the $Q_s/a_0^2$
in spec 03 R28. **Read:** `NUM` 02 p.3–4 (sign of the Jacobian), `NUM` 06 p.1 (whether the
$G_{\rm me}$ relation was meant to carry $a_0$). **Decide:** which of the three the code should
compute. *Audit note (not for the author to resolve from the notes): the code's `analytic_integral` carries no $a_0$ at all; whether that is a deliberate $a_0=1$ normalisation must be checked in the audit pass.*

### 1.2 The numerical prefactor of the one-loop formula

`NUM` 03 pp.7–10 (spec 03 R30–R35, spec 03-B R21–R28) carry a chain of blue-ink prefactors
$1024 \to 2048 \to 1024\pi \to 512\pi^2$ with typed red annotations dated 10 July 2025 reading
$1296$, $2592$, "$1292$", "$646$". Both transcribers recorded the same originals and the same
annotations, and both noted "$1292$"/"$646$" do not follow the chain (`diff-03.md` §3).

The cross-spec agent finds that `MAIN` 14 p.6 (spec 05 R23, $32\int d^3q/(2\pi)^3\,Q_s^2\ldots$)
agrees with the `NUM` 03 structure **only** with $2\times 36^2 = 2592$, i.e. the red corrections
are right and the original $1024 = 32^2$ was the slip; "$1292$" and "$646$" are then typos for
$1296$ and $648$ (`cross-spec-check.md` §1 B6, §2 hand-off (iii)).

**Read:** `NUM` 03 pp.7, 9, 10. **Confirm:** $1296 \to 2592 \to 1296\pi \to 648\pi^2$, and
correct the two typos on the page or in the spec.

### 1.3 Which form of the one-loop integral the code should implement

Two forms exist and neither document completes the angular reduction
(`cross-spec-check.md` §2 hand-off (iii); both spec 05 transcribers flagged this independently):

- `MAIN` 14 p.6 (spec 05 R23): $P^h_{22}(k) = 32\int\frac{d^3q}{(2\pi)^3}Q_s(\mathbf k,\mathbf q)^2P_*(q)P_*(|\mathbf k-\mathbf q|)\,\big(\int\ldots\big)^2$ — 3D loop measure, no angular reduction.
- `NUM` 03 p.10 (spec 03 R35): $(2\pi)^3\delta\,\delta_{ss'}\,[648\pi^2]\big(\tfrac{1+w^*}{5+3w^*}\big)^4\int\frac{dq}{q}\sin^5\theta\,\mathcal P^*(q)\frac{\mathcal P^*(r)}{r^3}\{\ldots\}^2$ — azimuth done, but **no $d\theta$ written**, $r(\theta)$ not written, limits not written.

**Decide:** which form is the build spec, and supply the missing $\theta$ integral (or the
$(u,v)$ / $(q,r)$ change of variables if that is preferred) and whether the deliverable is
per-polarisation $P^h_{22}$, summed over $s$, or $\Omega_{\rm GW}$. **Read:** `NUM` 03 pp.9–10.

### 1.4 Identity and normalisation of the primordial seed

The seed is written $\phi^*$ (spec 01), a "5-like glyph" read as $\zeta^*$ (spec 03, alternative
$S^*$), $S^*$ (spec 05 primary) and $\zeta^*$ (spec 05 duplicate) — the one DISAGREE in pair 05
(`diff-05.md` §3 D1). No coefficient depends on the letter, but $P_*$ is the spectrum of *this*
variable and the code has to be fed the right one. **Read:** `MAIN` 14 p.1 and `NUM` 03 p.4 (the
glyph). **State:** the definition of the seed relative to $\zeta$ or $\mathcal R$, its sign, and
whether $P_*$ is $P_\zeta$.

---

## Tier 2 — single-glyph checks on intermediate lines

Each is a specific page location where either the two transcribers disagreed or the single
transcriber flagged `medium`. None changes a final formula, but each should be settled so the
spec can be signed off.

| # | Document, page | Question | Source |
|---|---|---|---|
| 2.1 | `NUM` 06 p.6 top (boxed "so $f=$"), p.6 foot, p.4 last term | Do the $J_{5/2}$ factors carry "$+b$"? Primary reads $+b$ (`high`), duplicate reads bare $\tfrac52$ (`medium`) at these three places only; both read $+b$ on p.5 and p.7. Almost certainly an omission on the page. | `diff-04.md` D1–D3 |
| 2.2 | `NUM` 06 p.7 final display | Confirm the two cross-outs: struck prime after $\eta$ in $(qrc_s^2\eta)^{-1/2-b}$; kernel $Y_{b+1/2}(k\eta')$ over-written from $J$. Both read them the same way. | `diff-04.md` queue 2 |
| 2.3 | `MAIN` 14 p.10–11 | Exponent of 2 over-written ($2^{2+3b}\to2^{2+2b}$); both read $2+2b$, and `NUM` 06's $\tfrac\pi2\cdot2^{3+2b}$ confirms it independently. Confirmation only. | `diff-05.md` queue 1 |
| 2.4 | `MAIN` 14 p.8 bottom, p.9 | Over-written Bessel order in the third term of the four-term $f$ (read $J_{5/2+b}J_{3/2+b}$); $k$-for-$r$ slip; $J_{5/2}$ without $+b$. The completed square on p.9 depends on the p.8 reading. | `diff-05.md` queue 4–6 |
| 2.5 | `NUM` 09 p.5 vs p.6 | Two coefficients in $2\omega_{\rm eff}\,d\omega_{\rm eff}/dz$ disagree between consecutive pages ($-\tfrac92(1+w)w'$ vs $-\tfrac94w'(1+w)$; presence of $-\epsilon\epsilon'/4$). The boxed final result follows p.6. Unmarked by the author. **Only unconfirmed formula on the transfer-function WKB path.** | spec 01 Q7 |
| 2.6 | `NUM` 02 p.2 | Sign of the $a(aH^2+a\dot H)$ term under two red annotations. | spec 02 R13 |
| 2.7 | `MAIN` 13 p.3 | Intermediate $\alpha$ drops the $Y_{b+1/2}(k\eta')$ factor present on p.2; final result consistent with keeping it. | spec 02 Q8 |
| 2.8 | `NUM` 11 p.5–6 | $Q(u)$ representation: with $\Theta'=+\omega_{\rm eff}$ the fixed point is $Q=-1$ but the text says "close to unity". | spec 02 Q10 |
| 2.9 | `NUM` 03 p.7 | Ink blot on the denominator $a_0^2H^2(z')$ of $I_s$; both read it the same. | `diff-03.md` queue 3 |
| 2.10 | `NUM` 03 p.9 | Struck, unused factor $q^4_{\rm phys}/2$ vs $/4$. Irrelevant to results. | `diff-03.md` queue 5 |
| 2.11 | `MAIN` 12 p.2 | RHS sign of the $ij$ equation appears negative but context requires positive. | spec 01 R3 |

---

## Tier 3 — notation to fix in the specs, not on the page

Confirm-and-annotate items. The transcribers were told not to fix the physics, so these are
recorded as written; the author should say which reading later steps use so the spec can carry
a one-line note.

- **Prime on the transfer function.** `MAIN` 11 writes $\Phi'$ for $d/d\eta$; `MAIN` 14 uses
  $d/dx$ with explicit chain-rule factors; `NUM` 06 confirms $d/dx$. The `MAIN` 11 notation is
  the slip (`cross-spec-check.md` B4).
- **$f$'s first argument** alternates between $(\mathbf k,\mathbf k-\mathbf q)$ and
  $(\mathbf q,\mathbf k-\mathbf q)$ in `NUM` 03 pp.5–10 (both transcribers). The $q$ form is what
  `MAIN` 14 uses.
- **$z$ vs $z'$ inside $f$** on `NUM` 03 pp.9–10 (written unprimed; should be the source time).
- **$w_0$ vs $w^*$ vs $w$.** `NUM` 03 uses $w^*$ (initial) in the prefactor and $w_0$ inside $f$;
  the cross-spec agent's reading is $w_0 = w(z')$ at the source time. `NUM` 06 p.3 writes
  "$1+w_0$" once with "$1+w$" on the line above. Confirm $w$ inside $f$ is evaluated at the
  source time, and that $c_s^2 = w(z)$ is the closure used everywhere (spec 01 states
  $c_s^2\equiv w$; for constant $w$ this equals the README's $(1-b)/(3(1+b))$).
- **$k$ vs $k_{\rm phys}=k/a_0$** in the $\omega^2_{\rm eff}$ of `NUM` 05/10/11 (bare $k^2/H^2$)
  vs `NUM` 02 ($k^2_{\rm phys}/H^2$).
- **$\eta_0$ / $z_{\rm init}$** never specified; whether the analytic target assumes
  $\eta_0\to0$.
- **Overloaded symbols** to note in the specs' convention sections: $\epsilon$ (slow-roll vs
  infinitesimal), $\eta$ (conformal time vs second slow-roll parameter in `NUM` 05),
  $\omega_{\rm eff}$ (for $G_k$ vs for $\vartheta$), $Q_s$ (with $q$ vs $q_{\rm phys}$, differing
  by $1/a_0^2$).

---

## What was resolved without the author

`cross-spec-check.md` §3.1 closes 20 cross-references the transcribers had left open,
including: the $c_s^2$ definitions agree; the $4M_P^4/a^4$ (`MAIN` 10) vs $8M_P^2/a^2$ (`NUM` 03)
source normalisations are reconciled in `MAIN` 11; $f_{03} = f_{06} = f_{14}/c^2$ exactly; the
`MAIN` 14 recheck confirms `MAIN` 11's final formula (spec 05 Checks). Nine cross-spec questions
remain open and are folded into the tiers above.

## Sign-off

When each tier-1 decision is made and each tier-2 glyph confirmed, add a sign-off line at the
top of the corresponding spec file (README §7 step 3). The audit pass uses signed-off specs only.
