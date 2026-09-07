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

## Tier 2 — single-glyph checks on intermediate lines — **all RESOLVED 2026-09-07**

Each was a specific page location where either the two transcribers disagreed or the single
transcriber flagged `medium`. None changes a final formula. Author's readings, with the spec
location where each is recorded:

| # | Document, page | Resolution | Recorded |
|---|---|---|---|
| 2.1 | `NUM` 06 p.6 top, p.6 foot, p.4 last term | $J_{5/2+b}$ everywhere; the bare $\tfrac52$ is an omission on the page. Primary reading stands. | spec 04 §0.2; `diff-04.md` D1–D3 |
| 2.2 | `NUM` 06 p.7 final display | Both cross-outs confirmed: the prime after $\eta$ is struck (this $\eta$ is outside the $\eta'$ integral and cannot be its variable); the kernel $J$ is renamed $Y$. | spec 04 §0.2, C6–C7 |
| 2.3 | `MAIN` 14 p.10–11 | Exponent changed on the page from $2+3b$ to $2+2b$. Confirmed. | spec 05 §0.4 |
| 2.4 | `MAIN` 14 p.8 bottom, p.9, p.10 | Bessel orders as read; third term is the $q\leftrightarrow r$ exchange of the second; all $J_{5/2}\to J_{5/2+b}$. The "$k$-for-$r$ slip" on p.9 was a **transcription error**: the author re-read the page and it reads $(r\eta c_s)$. Nothing on the page to correct. | spec 05 §0.4 |
| 2.5 | `NUM` 09 p.5 vs p.6 | **Genuine error on p.6, and it propagates.** $\frac{d}{dz}[-\tfrac94(1+w)^2]=-\tfrac92(1+w)w'$; p.5 is right. Corrected final term: $\tfrac32w'\big(\epsilon-3(1+w)\big)$, not $\tfrac32w'\big(\epsilon-\tfrac32(1+w)\big)$. The $\epsilon\epsilon'/4$ difference is not an error (p.6 is the simplified form). **Audit finding:** the p.6 slip is in `ComputeTargets/WKB_Tk.py` (`Tk_d_ln_omegaEff_dz`); see below. | spec 01 R30, Q7 |
| 2.6 | `NUM` 02 p.2 | Minus sign, as transcribed; required for the friction term to collapse to $\epsilon$. | spec 02 §0.2, R13 |
| 2.7 | `MAIN` 13 p.3 | The "$\alpha$" is $\propto$: the $Y$ factor is temporarily suppressed and restored later. | spec 02 §0.2, Q8 |
| 2.8 | `NUM` 11 p.5–6 | Wording only. Intent is $\Theta-\Theta_i\sim\omega_{\rm eff}u$, i.e. $\lvert Q\rvert\to1$. Under the page's and the code's $d\Theta/dz=+\omega_{\rm eff}$ the fixed point is $Q=-1$; the code implements R38 verbatim and is independent of the sign of $Q$. | spec 02 §0.2, Q10 |
| 2.9 | `NUM` 03 p.7 | $a_0^2H^2(z')$ confirmed. | spec 03 §0.5, R28 |
| 2.10 | `NUM` 03 p.9 | Struck and unused; remaining expression correct. | spec 03 §0.5, R34 |
| 2.11 | `MAIN` 12 p.2 | The equation is the diagonal $ij$ (pressure) equation, first display on p.2, whose RHS appears to read $-\delta p/M_P^2$; it is *not* the $\eta\eta$ equation carrying the 26 Apr 2023 annotations (those concern spec 01 R2 and are recorded in its §5). Read as positive; the p.1 form, the next line and R5 all require it. | spec 01 R3, Q2 |

### Audit finding surfaced by 2.5

`ComputeTargets/WKB_Tk.py`, function `Tk_d_ln_omegaEff_dz`, contains
`3.0 / 2.0 * wPrime * (eps - 3.0 / 2.0 * (1.0 + w))`; the inner factor should be `3.0 * (1.0 + w)`.
The function feeds (a) the WKB-validity diagnostic, harmless, and (b) `TkWKBIntegration.store()`,
where it enters `raw_sin_coeff` and so shifts the Liouville–Green amplitude and phase of $T_k$ by a
term proportional to $w'$. `WKB_Gk.py` is unaffected. **Fixed 2026-09-07**, in the same commit as this sign-off;
the corrected derivative was re-verified symbolically against `Tk_omegaEff_sq`.

---

## Tier 3 — notation to fix in the specs, not on the page — **all RESOLVED 2026-09-07**

- **Prime on the transfer function.** `MAIN` 11's $\Phi' = d\Phi/d\eta$ is an anomaly; treat it as such.
  `MAIN` 14 and `NUM` 06 use $d/dx$, and the project converges on $T_k(z)$. (spec 05 §0.4)
- **$f$'s first argument.** $f(\mathbf q,\mathbf k-\mathbf q)$; `MAIN` 14 is correct, `NUM` 03's
  $(\mathbf k,\mathbf k-\mathbf q)$ is a slip. (spec 03 §0.5)
- **$z$ vs $z'$ inside $f$.** $z'$, the source time integrated over. (spec 03 §0.5)
- **$w_0$ vs $w^*$ vs $w$.** $w_0 = p_0/\rho_0$ is the background equation of state ($\Lambda$ included;
  code `wBackground`), evaluated at the source time inside $f$; $w^*$ is $w_0$ at the initial time;
  elsewhere $c_s^2 = w(z)$ of the perturbed fluid is intended ($\Lambda$ unperturbed; code
  `wPerturbations`). (spec 01 head block, spec 03 §0.5)
- **$k$ vs $k_{\rm phys}$.** `NUM` 02's $k_{\rm phys}=k/a_0$ reading is correct for the bare $k^2/H^2$
  of `NUM` 05/10/11. (spec 02 §0.2)
- **$\eta_0$ / $z_{\rm init}$.** Need not be specified in the spec: they are initial data in the
  pipeline (per mode, a fixed number of e-folds before horizon exit). The spec records only the
  conditions on them: constant $w$ at $z_{\rm init}$, and all of $k$, $q$, $r$ super-horizon there.
  The `MAIN` 14 target is insensitive to $\eta_0$ (corrections $O((k\eta_0)^2)$), so it may be read
  with $\eta_0\to0$. (Tier 3 blocks in specs 01, 02, 04, 05)
- **Overloaded symbols.** Confirmed and listed in the Tier 3 blocks of specs 01, 02 and 03.

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

**Status 2026-09-07:** all Tier 1, Tier 2 and Tier 3 items are resolved and recorded in the spec files. The one
audit finding surfaced (2.5, `WKB_Tk.py`) is fixed. **All five specs and their duplicates are signed off.**
