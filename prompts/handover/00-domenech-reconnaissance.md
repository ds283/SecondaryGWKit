# Prompt 00 — Domènech reconnaissance: what the general-$b$ kernel actually says

**Campaign:** [`prompts/handover`](README.md) · **Workstream:** A (instruments), precursor to A1
**Model:** Fable (executed on a separate account; see §8)
**Kind:** **Read-and-report only.** This prompt writes **one document** and **no code**. It does
not touch `ComputeTargets/`, `LiouvilleGreen/`, `Quadrature/`, `main.py`, `config/`, any factory,
any schema, or any test. If you find yourself editing a `.py` file under a package directory, stop.

---

## 1. Why you are being asked

`prompts/handover` needs a general-$b$ oracle for the source time integral, to sit beside the
$b = 0$ Kohri–Terada oracle that `prompts/radiation-oracle` already landed. The intended
construction is **Domènech's exact kernel with the $x\to\infty$ coefficients substituted** — see
[`README.md`](README.md) §2 (i). Implementing it is prompt **A1**'s job.

You are the pass **before** A1, and you exist because of a specific, measured failure mode.
`prompts/radiation-oracle` found **three errata** in Kohri & Terada. Every one was found by
numerical disagreement with a quadrature of the paper's own defining integral — **none** by
reading. One of them, the pairing of the $\mathrm{Ci}$/$\mathrm{Si}$ arguments, is described in
that audit as *"the natural reading of the rendered equation"*.

So your job is **not** to certify the equations. It is to put A1 in a position where its numerical
cross-check is *decisive rather than ambiguous*: to say exactly which convention each symbol is in,
where the branch cuts are, what the papers disagree about, and what a correct implementation must
reproduce in limits that can be checked independently. **A1 will still do the numerical check.**
Your report tells it what to check and what the answer should be.

**Do not resolve a discrepancy by choosing.** Where the two papers differ, report both readings and
say what numerical test discriminates them. Choosing is A1's, on measurement.

---

## 2. Inputs

All in the repository; nothing needs to be fetched.

| Input | Path | Use |
|---|---|---|
| Domènech 2020, LaTeX | `docs/handover/sources/1912.05583-src/inducedSGWBwarxiv_revised_2.tex` | **Authoritative.** The derivation, and Apps. on the Bessel integral and the Legendre functions |
| Domènech 2020, PDF | `docs/handover/sources/1912.05583.pdf` | Cross-reference only |
| Domènech 2021 review, LaTeX | `docs/handover/sources/2109.01398-src/template_review.tex` | **Authoritative.** Same results natively in $b$ and $c_s$ |
| Domènech 2021 review, PDF | `docs/handover/sources/2109.01398.pdf` | Cross-reference only |
| Equation-number map | `docs/handover/sources/SOURCES.md` | Published numbers ↔ `\label`s. **Cite both** |
| Campaign design facts | [`README.md`](README.md) §2 (g)–(l) | What has already been established, and what is only claimed |
| The $b=0$ oracle it must agree with | `docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`, `ComputeTargets/tests/kohri_terada.py` | §5 item 5 below |
| The repository's conventions | `docs/spec/01-transfer-function.md`, `docs/spec/03-source-term.md`, `docs/spec/04-source-integral.md` §0, `docs/spec/05-one-loop.md` | §5 item 6 below |

**The `.tex` is authoritative.** Do not work from the PDF, from a rendered HTML view, or from
memory of these papers. Where you quote, quote the LaTeX and give `file:line`.

`docs/handover/sources/SOURCES.md` resolves published equation numbers against `\label`s. Cite as
`(4.12) eq:Isimple2` throughout — the numbers are derived by counting environments and could be off
by one somewhere; the `\label` cannot.

---

## 3. What to establish

### 3.1 The two constructions, unambiguously

Write out, in full and in the **review's** $b$/$c_s$ variables:

1. The exact kernel (4.10) `eq:Isimple` with (4.11) `eq:Isimpledef`.
2. The $x\to\infty$ coefficients ${\cal I}^\infty_{J,Y}$, translated from (3.3)/(3.4)
   `eq:IJ`/`eq:IY` into $b$ and $c_s$ — **showing the translation**, not asserting it.
3. The doubly asymptotic (4.12) `eq:Isimple2`.
4. **The target object**: (4.10) with (2) substituted. State it explicitly. Then show that
   expanding $J_{b+1/2}(x)$, $Y_{b+1/2}(x)$ for large argument recovers (4.12) **exactly**,
   including every constant. If it does not, that is the single most important finding in your
   report and you should say so at the top.

Item 4 is the load-bearing consistency check available from reading alone, because it ties the
two constructions the papers give to each other through a step you perform.

### 3.2 The Legendre functions — conventions, and the mapping to `mpmath`

This is where a transcription is most likely to fail silently, and `scipy` cannot help
(`scipy.special.lpmv` takes **integer** order; here the order is $-b$, non-integer for $b \ne 0$).

Establish, from App. `App:legendre` of the 2020 paper and `app:legendre` of the review, and from
DLMF:

- What $\mathsf{P}^\mu_\nu$, $\mathsf{Q}^\mu_\nu$ (Ferrers, "on the cut", $|y|<1$) and
  ${\cal Q}^\mu_\nu$ ($|y|>1$) mean **in these papers**, with the papers' own explicit
  hypergeometric expressions.
- Whether the papers' $\mathsf{Q}$ is DLMF's Ferrers function of the second kind or differs by a
  normalisation (the $\pi/2$, $\cos(\mu\pi)$ and $\Gamma$ factors that distinguish the common
  conventions are exactly where this goes wrong).
- **The mapping to `mpmath`**: for each of the six function calls needed
  ($\mathsf{P}^{-b}_{b}$, $\mathsf{P}^{-b}_{b+2}$, $\mathsf{Q}^{-b}_{b}$, $\mathsf{Q}^{-b}_{b+2}$,
  ${\cal Q}^{-b}_{b}$, ${\cal Q}^{-b}_{b+2}$), give the exact `mpmath.legenp`/`legenq` call
  including `type=2` vs `type=3` and any factor needed to convert conventions. State how you
  verified each mapping — a Wronskian, a known special value, a recurrence, or a limit.
- Whether `type=3`'s branch cut placement matters on the domain actually used, given
  $\tilde y = -y$ and $y < -1$ in the off-cut branch.

Note $\mu \equiv -b$ and $\nu \in \{b, b+2\}$, so $\mu+\nu \in \{0, 2\}$ — the 2020 paper remarks on
this and it may admit closed forms. Say whether it does.

### 3.3 The two inter-paper discrepancies

[`README.md`](README.md) §2 (i2) records both. For each: confirm it exists in the LaTeX (quote
`file:line`), determine whether it is a difference of **convention** (the papers define $I$ or $f$
differently) or a **typo** in one of them, and — this is the part A1 needs — **state the numerical
test that discriminates**.

1. **Sign/order and prefactor of the exact kernel.** 1912 (3.1) `eq:kernel2` has
   $\{Y_\beta{\cal I}^x_J - J_\beta{\cal I}^x_Y\}$; review (4.10) `eq:Isimple` has
   $(J_{b+1/2}{\cal I}_{Y} - Y_{b+1/2}{\cal I}_{J})$. Their prefactors differ in the $\Gamma$
   argument ($\Gamma^2[b+5/2]$ against $\Gamma^2[b+3/2]$) and in the power of $(c_s^2uvx)$ (by
   $\tfrac12$). Trace both back to each paper's own definition of $I$ and of the source, and say
   whether the two $I$'s are the same object. Compare the review's (4.9) `eq:fsimple` with the
   2020 paper's source.
2. **The asymmetric factor of 2** in the off-cut branch — $2\frac{b+2}{b+1}$ against
   $\frac{b+2}{b+1}$, present identically in 1912 (3.4)/(3.8) and review (4.12). Determine from
   the Gervois–Navelet result in App. `App:integralbessel` whether it is correct. **It is in both
   papers, so it is not a typo in one of them; say whether it is right in both or wrong in both.**

### 3.4 The resonance

At $c_s(u+v) = 1$, $y = -1$ (review (4.13) `eq:y`). Establish:

- The exact divergence structure of the $x\to\infty$ coefficients as $y \to -1^+$, per $b$ regime.
  [`README.md`](README.md) §2 (i3) claims power-divergent for $b<0$, log at $b=0$, finite for
  $b>0$, from 1912 §3's $I \propto (1+y)^{-\frac12(\mu+|\mu|)}$ / $\ln(1+y)$ with $\mu = -b$.
  **Confirm or correct this**, and note that the 2020 paper's displayed limits say "$y\to1$" where
  its own text says $y\to-1^+$ — establish which is meant.
- Whether the $|1-y^2|^{b/2}$ prefactor cancels the Legendre divergence **analytically** (as
  $\mathrm{Cin}$ does in KT eq. (22)), and if so, write the regularised combination that is finite
  and evaluable *at* $y=-1$ for the $b>0$ case.
- What happens **approaching** the resonance in floating point, and at what distance in
  $c_s(u+v)-1$ a naive evaluation loses accuracy.
- Whether the fixtures' $b = 0.2$ and the triangle condition $|u-v|\le1\le u+v$ put any fixture
  case at or near the resonance.

### 3.5 Domain, validity and limits A1 can assert against

- The papers' stated validity range, in $b$: confirm $-\tfrac12 \le b < 1$ and say what breaks at
  each end.
- Whether $y$ can leave $[-1,1]$ and $|y|>1$ for physical, triangle-closing $(u,v)$, and which
  branch each fixture shape lands in. **Note that the repository's `T-first` fixture shape is
  deliberately *not* triangle-closing** ($|q-r| = 2000 > k = 1000$) — say whether the Domènech form
  is defined there at all, since KT §8 item 4 records eq. (25) failing on exactly that shape.
- **The $b\to0$ limit.** Reduce the target object of §3.1 item 4 analytically at $b=0$ and compare
  with 1912 (3.19)–(3.22), the paper's own radiation special case. Then say how that relates to
  KT eq. (22) — they are the same physical quantity, and `ComputeTargets/tests/kohri_terada.py`
  already implements KT. **Predict the constant relating them**, or say that it cannot be
  predicted from the papers and must be measured. Do not assert a value you have not derived.
- The small-$x$ and large-$x$ behaviour of the target object, for A1's regression tests.

### 3.6 Normalisation against this repository

`docs/OPEN_ISSUES.md` carries `[01-general-w-normalisation-is-predicted-not-measured]`: the
$b = 0$ tie is $\text{total} = -\tfrac98 k_{\rm phys}^{-2} I_{\rm RD}$, and the general-$b$
extrapolation $N(b) = -\frac{(3+2b)^2}{2(2+b)^2}$ is **predicted and unchecked**.

Using `docs/spec/04-source-integral.md` §0 and `docs/spec/03-source-term.md` §0.1, and the mapping
derivation in `docs/radiation-oracle/KOHRI-TERADA-ORACLE.md` §3, say whether $N(b)$ **follows** from
Domènech's definitions of $I$ and $f$, and if so with what value. Be explicit about which of the
three factors in $-\tfrac98 = -\tfrac12\times\tfrac94$ carry $b$-dependence and which do not.

**Read KT §0 before doing this.** The load-bearing statistic is that $N$ is *constant* over
$(u,v,x)$, not that it matches a prediction. If your derivation disagrees with
$-\frac{(3+2b)^2}{2(2+b)^2}$, say so plainly — that is a useful finding, not a failure.

Conventions you must not "correct": $a_0$ is absorbed, never set to 1; the sign of $N$ is
`spec/04-source-integral.md` §0(3)'s orientation convention; $c_s^2$ is `wPerturbations`.

---

## 4. What A1 will need that you should supply

A short, self-contained **implementation brief**: the target object written once, in $b$ and $c_s$,
with every symbol defined; the six `mpmath` calls; the branch conditions; the regularised resonance
form; the guard regions; and the limit values A1 can assert against. Written so that A1 can
implement from your report **without reopening the papers** — while still being required to do its
own numerical cross-check.

---

## 5. Output

**One file: `docs/handover/DOMENECH-KERNEL-RECON.md`.** Nothing else. Suggested structure:

```
§0  What this is, what it is not, and the provenance of every object discussed
§1  Headline: is the target object of §3.1 item 4 well defined, and does it reduce to (4.12)?
§2  The two constructions, in b and c_s, with the translation shown
§3  The Legendre functions: conventions, mpmath mapping, and how each mapping was verified
§4  The two inter-paper discrepancies: what they are, and the test that discriminates
§5  The resonance: divergence structure, the regularised form, and floating-point behaviour
§6  Domain, branches, and the limits A1 can assert against
§7  The b -> 0 reduction and the relation to Kohri-Terada
§8  Normalisation: does N(b) follow, and with what value
§9  Implementation brief for A1
§10 Open questions this pass could not settle, and what would settle each
```

Rules for the document:

- **Quote, with `file:line`.** Every equation you rely on gets its `\label`, its published number,
  and its source line. A paraphrase is not a citation.
- **Distinguish three things at every point**: what the paper *says*; what you *derived*; what you
  *inferred or guessed*. Mark the third explicitly. §10 exists so that guesses have somewhere to
  go that is not the body.
- **A number without its reference's own error beside it is not a measurement.** If you evaluate
  anything numerically to check yourself, say what you evaluated it against.
- **State disagreements with [`README.md`](README.md) §2 loudly.** Those design facts were
  established by grep and consistency-checking, not by reading the derivations. If (h), (i), (i2)
  or (i3) is wrong, the campaign plan changes, and §1 of your report is where that belongs.
- Markdown, `black`-irrelevant (no code ships), wrapped at ~100 columns to match `docs/`.

You may run Python to check yourself (`mpmath`, `scipy` — the venv has both, `mpmath 1.3.0`,
`scipy 1.15.2`). If you do, **put throwaway scripts in a scratch directory, not in the repository**;
if a check is worth keeping, say so in §10 and let A1 land it. `docs/` scripts must run from the
repository root with `PYTHONPATH=.` and must not need Ray or a datastore.

---

## 6. Explicitly out of scope

- Any change to `ComputeTargets/`, `LiouvilleGreen/`, `Quadrature/`, `AdaptiveLevin/`, `main.py`,
  `config/`, any factory, any schema, any test, or `docs/spec/`.
- Implementing the oracle. That is A1.
- Deciding anything. Where the papers disagree, or where a convention is ambiguous, **report both
  and give the discriminating test**.
- The oscillation average. Review (4.14) `eq:kernelaverage` is the general-$b$
  $\overline{I^2(x,u,v)}$ and (4.42) `eq:kernelsuperhave2` is a reheating-transition limit; the
  repository has no oscillation-average capability and is not acquiring one here. **One paragraph
  in §10** on what implementing (4.14) would need is welcome; nothing more.
- Kohri & Terada themselves. That oracle is landed and measured; you read it only for §3.5 and §3.6.

---

## 7. Acceptance

1. `docs/handover/DOMENECH-KERNEL-RECON.md` exists and no other file in the repository has changed.
2. The target object of §3.1 item 4 is written out once, completely, in $b$ and $c_s$, and either
   shown to reduce to (4.12) `eq:Isimple2` or reported as not doing so.
3. Every one of the six Legendre calls has an `mpmath` expression **and a stated verification**.
4. Both §3.3 discrepancies are characterised, each with a discriminating numerical test A1 can run.
5. §2 (i3)'s resonance claim is confirmed or corrected, and a form finite at $y=-1$ is given for the
   $b>0$ case or its impossibility is argued.
6. §8 either derives $N(b)$ or states plainly that the papers do not determine it.
7. §10 is non-empty. A reconnaissance pass that settled everything did not look hard enough.

---

## 8. Execution note

This prompt is executed **outside the campaign's normal flow**, on a separate account, by a model
this repository's usual session cannot reach. Consequently:

- It is numbered **00** and is **not** a `prompts/handover` campaign prompt in the
  [`README.md`](README.md) §5 sense: it lands no code, and the one-commit / log / board rules do
  not bind it.
- Its output is an input to A1, in the way `docs/radiation-oracle/KOHRI-TERADA-ORACLE.md` was an
  input to `prompts/radiation-oracle` prompt 01.
- Whoever commits the report should do so on its own, with a message saying which model produced it
  and against which source checksums (`docs/handover/sources/SOURCES.md`).
- **The report is a document, not an authority.** A1 is still bound by [`README.md`](README.md)
  §5.1: transcription is verified against an independent numerical quadrature of the papers' own
  defining integral (4.11) `eq:Isimpledef`, whatever this report says.
