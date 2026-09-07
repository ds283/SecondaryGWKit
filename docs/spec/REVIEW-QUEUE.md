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

### 1.1 Which redshift-space Green's function is "the" Green's function — **RESOLVED 2026-09-07**

**Decision (author).** The code computes the unit-jump causal Green's function for $\chi_s = a h_s$ in
redshift: $G(z',z') = 0$, $dG/dz|_{z=z'} = +1$, zero for $z > z'$, i.e. source $-\delta(z-z')$
(`ComputeTargets/GkNumericIntegration.py`). This is $\bar G_k$ of `NUM` 03 (spec 03 R25) and
$G_{\rm me}$ of `NUM` 06 (spec 04 R1–R2), and it is the **definitive project convention**. Its relation
to the retarded conformal-time function ${\rm Gr}_k$ of `MAIN` 13/14 is
$G_{\rm code} = -a_0H(z')\,{\rm Gr}_k(\eta(z),\eta(z'))$ for any $a_0$. The full statement, with code
references, is §0 of `02-greens-function.md` and `04-source-integral.md`; §0 of `03-source-term.md`
and `05-one-loop.md` carry the compact form.

**Corrections to the original queue entry.**

- *Page reference.* `MAIN` 14 defines ${\rm Gr}_k$ on **p.2–3** (spec 05 R18) and gives its explicit
  retarded form on **p.4** (spec 05 R21). p.11 is where the Step 6 build target lives, not where
  ${\rm Gr}_k$ is defined. `NUM` 02 is about the tensor Green's function throughout (pp.1–4), not only
  pp.3–4.
- *"Three normalisations" — there is one.* `NUM` 02 R17 is ${\rm Gr}_k$ itself re-expressed in $z$: the
  $-\frac{1}{a_0H}$ is the Jacobian, not a new normalisation. Its sign follows from transforming
  $\delta(\eta-\eta')$ without $|\cdot|$, which is a convention paired with the orientation of
  $\int_{z_{\rm init}}^{z}dz'$, not a slip (spec 02 Q3 closed). `NUM` 03 and `NUM` 06 drop the
  Jacobian and impose a unit jump; they define the *same* object, which is what the code computes.
- *$a_0$.* The code does **not** set $a_0 = 1$. It absorbs $a_0$ into $k/a_0$ (physical wavenumber
  today) and $a_0\eta$; every formula is written in these two invariant combinations. `NUM` 06 R2
  is written in that absorbed convention while R13 keeps $a_0$ explicit, which is why R14 shows $a_0^1$
  where the covariance test ($a_0 \to \lambda a_0$, comoving momenta $\to \lambda\times$,
  $\eta \to \eta/\lambda$, LHS invariant) requires $a_0^2$. With $a_0^2$, spec 04 R14 $= -a_0^2/c^2$
  times the `MAIN` 14 target, the $a_0^2$ cancels the $Q_s/a_0^2$ of spec 03 R28, and $h_s$ is
  $a_0$-independent. The cross-spec agent's B2 is confirmed as to the power, with the reading
  "absorbed", not "$a_0 = 1$ assumed".
- *Audit note.* The code's `analytic_integral` carries no $a_0$ because it is written in $k/a_0$,
  $a_0\eta$; in those variables the $a_0^2$ form has no $a_0$. This is the correct normalisation, not
  an omission. Nothing in the code needs to change for this item.

### 1.2 The numerical prefactor of the one-loop formula — **RESOLVED 2026-09-07**

**Decision (author).** The red annotations of 10 July 2025 are right; "1292" and "646" are typos.
Taking the $36\big(\tfrac{1+w^*}{5+3w^*}\big)^2$ of $h_s$ (spec 03 R26–R28) as correct, the chain on
`NUM` 03 pp. 7–10 is

$$1296 \;(\text{p.7, p.8}) \;\to\; 2592 \;(\text{p.9}) \;\to\; 1296\pi \;(\text{p.9, both boxes}) \;\to\; 648\pi^2 \;(\text{p.10}),$$

so the final formula spec 03 R35 carries $648\pi^2\big(\tfrac{1+w^*}{5+3w^*}\big)^4$. The original
1024 is $32^2$: the `MAIN` 11/14 coefficient squared. `MAIN` 11/14 keep the super-horizon factor
$c^2 = \big(\tfrac{3(1+w)}{5+3w}\big)^2$ inside $f$ and so have $2\times4^2 = 32$; `NUM` 03 pulls
$c_*^2$ outside and so has $2\times(4c_*^2)^2 = 2592\big(\tfrac{1+w^*}{5+3w^*}\big)^4$. The two
agree ($32\times81 = 2592$); the cross-spec agent's B6 is confirmed. Full table, derivation and
the code status are in spec 03 §0.2.

**Also settled here (was Tier 3, "$w_0$ vs $w^*$ vs $w$").** `NUM` 03 is right to distinguish
$w_0 = w(z')$ inside $f$ (background at the source time) from $w^* = w(z_{\rm init})$ in the
prefactor $c_*$ (adiabatic super-horizon relation at the initial time). `MAIN` 11/14 use a single
$w$ because they assume a fixed-$w$ epoch (stated on `MAIN` 14 p. 2); that is a restriction, not an
error. Build requirement: $z_{\rm init}$ deep inside an epoch of constant $w^*$. The code already
evaluates $w$ at the source redshift in `ComputeTargets/QuadSource.py`. The $c_s^2 = w$ closure part
of that Tier 3 bullet is **not** yet confirmed.

**Code status.** `OneLoopIntegral.compute()` is a stub: the prefactor, the
$\big(\tfrac{1+w^*}{5+3w^*}\big)^4$ and the $\theta$ integral are not implemented anywhere, so this
item is a build specification rather than an audit finding.

### 1.3 Which form of the one-loop integral the code should implement — **RESOLVED 2026-09-07**

**Decision (author).** There is one form, not two. `MAIN` 14 p. 6 (spec 05 R23) is the loop
integral before the measure is split; splitting $d^3q$, inserting the spin-2 projector factors and
doing the azimuthal integral gives `NUM` 03 p. 10 (spec 03 R35) and nothing else happens in between.
**The build spec is spec 03 R35 with $648\pi^2$ (Tier 1.2) and the measure completed:**
$\int_0^\infty dq/q\int_0^\pi d\theta\,\sin^5\theta$, $r = \sqrt{k^2+q^2-2kq\cos\theta}$, $\theta$
the angle between $\mathbf q$ and $\mathbf k$. Written out in full in spec 03 §0.3. Carrying the
reduction out on spec 05 R23 reproduces it exactly for $w^* = w$, which also confirms 648
independently. A $(u,v)$ or $(q,r)$ change of variables is an implementation choice.

**Deliverable.** The stored quantity is the **per-polarisation** $P^h_{22}(k)$, labelled by $s$.
$\Omega_{\rm GW}$ is built in a **separate, decoupled layer**. Cosmological-collider-type models can
give unequal power in the two polarisations; the present calculation does not handle that, and
keeping the layers apart means a new compute layer for $P^h_{22,s}$ can be shipped later without
touching the layer that builds the observable.

**Method (build question).** How the $q$ and $\theta$ integrals are performed is open; the
Kohri–Terada resonance in the inner time integral has to be handled, possibly by stationary phase.
`OneLoopIntegral.compute()` is a stub.

### 1.4 Identity and normalisation of the primordial seed — **RESOLVED 2026-09-07**

**Decision (author).** The seed is $\zeta^*$, the primordial curvature perturbation, on `MAIN` 14 p. 1
and on `NUM` 03 p. 4 alike; the "5-like glyph" and the "$S^*$" are the same handwritten letter
(diff-05 D1 resolved for the duplicate's $\zeta^*$). The linear relation is
$\phi_{\mathbf k}(z) = \tfrac{3(1+w^*)}{5+3w^*}\,T_k(z)\,\zeta^*_{\mathbf k}$ with $T_k \to 1$ at
$z_{\rm init}$, correct at linear order (§1.2 for the $w^*$ condition). Hence **$P_* = P_\zeta$** and
$\mathcal P^* = \mathcal P_\zeta$, and the $\big(\tfrac{1+w^*}{5+3w^*}\big)^4$ in the build form
(§1.3) is the $\zeta^*\to\phi$ translation. Spec 01's $\phi^*$ is the early-time potential
$\tfrac{3(1+w^*)}{5+3w^*}\zeta^*$, not a different seed. The initial spectrum is specified in $\zeta$
so that `PyTransport`/`CppTransport` output can be fed in directly. The sign convention for $\zeta$
is immaterial because only $P_\zeta$ enters. Recorded in spec 03 §0.4, spec 05 §0.3, spec 01 §2.8.

**All four Tier 1 decisions are now made.**

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
  **Partly resolved 2026-09-07 (see §1.2):** $w_0 = w(z')$ at the source time and $w^* = w(z_{\rm init})$ are
  confirmed, and the code evaluates $w$ at the source redshift. The $c_s^2 = w(z)$ closure is still to be confirmed.
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
