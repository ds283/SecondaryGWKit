# Prompt 01 — the Domènech general-$b$ oracle

**Campaign:** [`README.md`](README.md) · **Board item:** **A1** ·
**Board:** `IMPLEMENTATION_STATE.md` — **does not exist; this prompt creates it** (§8).
**Closes:** `[01-general-w-normalisation-is-predicted-not-measured]` (radiation-oracle board §3).
**Recommended model:** **Opus**. The code is short. The transcription is unforgiving: the source
paper's own displayed kernel is **mis-signed**, and a faithful transcription of it produces a
function that looks entirely plausible and is wrong by an overall sign.

**Read first, in this order:**

1. [`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) — **§0
   and §1 before anything else**, then **§9**, which is the implementation brief written for you,
   then §2–§6 for the derivations behind it, then §11.
2. [`docs/handover/sources/SOURCES.md`](../../docs/handover/sources/SOURCES.md) — the version pins
   and the **Version hazard** section. The papers are in that directory; **the `.tex` is
   authoritative**, not the PDF, not a rendered HTML view.
3. [`docs/radiation-oracle/KOHRI-TERADA-ORACLE.md`](../../docs/radiation-oracle/KOHRI-TERADA-ORACLE.md)
   **§0 first** — whose object is whose. Then §7.2 and §8.
4. `ComputeTargets/tests/kohri_terada.py` — the $b=0$ oracle already in the tree, and the module
   this one sits beside. Match its shape.
5. Campaign [`README.md`](README.md) §0.4, §2 (g)–(m), §5 and **§5.1**.

**It changes no production code and no stored number.** Nothing under README §0.3's exclusions
appears in the diff.

---

## 1. What to build

A module beside `ComputeTargets/tests/kohri_terada.py` — name and location are yours to justify —
exposing the general-$b$ kernel and nothing more.

- **`I_target(v, u, x, b)`** — recon §9's boxed target object: the **corrected** (4.10)
  `eq:Isimple` with the $x\to\infty$ coefficients of (3.3)/(3.4) `eq:IJ`/`eq:IY` substituted,
  keeping $J_{b+1/2}(x)$ and $Y_{b+1/2}(x)$ **exact**.
- **`I_asymptotic(v, u, x, b)`** — the doubly asymptotic form, for large-$x$ regression only. This
  is the review's (4.12) **with its $\cos$ sign flipped** (recon §2.4).
- **`I_quadrature(v, u, x, b, ...)`** — quadrature of the **finite-$x$** integrals (4.11)
  `eq:Isimpledef` inside the exact kernel, breaking the range as
  `kohri_terada.I_RD_quadrature` does. **This is the decisive instrument**, not `I_target` — see
  §2 item 4.
- The coefficient functions $A$, $B$, $C$ of recon §9, separately callable and separately testable.
- Whatever `head`-style helper the $N$ tie needs (README §2 (k)).

Design constraints:

- **`mpmath` for the Legendre functions.** `scipy.special.lpmv` takes **integer** order; the order
  here is $-b$. Recon §9 gives the six calls and §3.2 gives the verification of each. Use the
  closed forms of recon §3.4 at $b \in \{0, \pm\tfrac12\}$ rather than calling `legenq` there.
- **Do not reach for `scipy.special.jv`/`yv` at production $x$.** README §2 (l) and
  `[01-scipy-jv-yv-high-order-boundary]`. At the $x$ this prompt's own tests use, `scipy` is fine;
  say in the docstring which regime the module is good for and why.
- Pure function of $(v, u, x, b)$. No datastore, no Ray, no global state, no tolerance that changes
  the answer. This is an oracle: it must be **fixed by construction**, like `kohri_terada.I_RD` and
  unlike `analytic_rad`.

---

## 2. The five things that will go wrong

Recon §1, §3.4, §5 and §6 give these in full. They are repeated here because each produces a
*plausible* wrong answer.

1. **The source's own (4.10) is mis-signed.** It prints
   $\big(J_{b+1/2}\mathcal I_Y - Y_{b+1/2}\mathcal I_J\big)$; the correct order is
   $\big(Y_{b+1/2}\mathcal I_J - J_{b+1/2}\mathcal I_Y\big)$. Transcribing the page gives a smooth,
   correctly-scaling function that is $-1$ times the truth. **Three independent derivations agree
   on the corrected order** — recon §2.2, `docs/spec/05-one-loop.md` R31, and a quadrature of the
   review's own $G\cdot f$ — and the version history in recon §11 explains how the slip arose.
2. **The asymmetric factor of 2 is correct.** The off-cut branch carries $2\frac{b+2}{b+1}$ where
   the on-cut branches carry $\frac{b+2}{b+1}$. It is $\Gamma[\nu-\rho+1]$ at $\nu-\rho=2$ and it
   appears identically in both papers. **Do not symmetrise it** (recon §4.2).
3. **The resonance.** At $c_s(u+v)=1$, $y=-1$ exactly, and the $x\to\infty$ coefficients are
   singular — power-divergent for $b<0$, logarithmic at $b=0$, finite for $b>0$ (recon §5.1). The
   finite-$x$ kernel is regular there. For $b>0$ recon §5.2 gives a closed form evaluable **at**
   $y=-1$; implement it and test it there. `mpmath` returns `nan`/`-inf` at $y=-1$ on the raw calls
   (recon §3.4).
4. **The $O(1/x)$ is not uniform, and this changes what you may assert.** The control parameter is
   the smallest Bessel argument $c_s\min(u,v)\,x$, not $x$. On the `q-smooth` shape ($u = 0.01$)
   `I_target` is still **5e-02** of the envelope off the exact kernel at $x = 3200$ (recon §6.4).
   **So the $N$ tie in §3 test 5 must be scored against `I_quadrature`, not against `I_target`.**
   Getting this backwards will look like a pipeline defect and is not one.
5. **`y` can leave $[-1,1]$, and one fixture shape is not a closable triangle.** `T-first` has
   $|q-r| = 2000 > k = 1000$. Recon §6.2 gives an inferred continuation for $y>1$ that was matched
   numerically but **not read from a source** (recon §10 item 2). Either implement it and mark it
   `[inferred]` in the docstring with a test that pins the evidence, or refuse that region with a
   clear error. **Say which you did and why in the log.** Do not implement it silently.

---

## 3. The tests

In `ComputeTargets/tests/`, a new module. Every test offline: no Ray, no datastore.

1. **Legendre mapping.** Each of the six `mpmath` calls of recon §9 against an independent check —
   a Wronskian (DLMF 14.2.4 / 14.2.8), a known special value, or a recurrence. This is the
   foundation; if it is wrong everything above it is wrong in a way no other test sees.
2. **$A$, $B$, $C$ closed forms against the raw calls**, away from $y=\pm1$, at `mp.dps >= 30`, at
   several $b$ in $(-\tfrac12, 1)$.
3. **`I_target` against `I_quadrature`**, away from the resonance and away from small
   $c_s\min(u,v)$: the difference must fall like $1/x$. Recon §2.4's table is the figure to
   reproduce (1.6e-2 → 1.5e-3 → 9.9e-6 at $x = 100/1600/6400$, $b=0.2$, $(u,v)=(1.3,1.2)$).
4. **`I_asymptotic` against `I_target`** at large $x$: the same $1/x$ approach.
5. **The $b=0$ reduction — the load-bearing external tie.** `I_target(b=0)` must equal
   $\tfrac98\times$`kohri_terada.I_RD_asymptotic`, and `I_quadrature(b=0)` must equal
   $\tfrac98\times$`kohri_terada.I_RD`, at the $(u,v,x)$ of recon §7. `kohri_terada` is the one
   object in this test that this campaign did not write.
6. **Small-$x$ limit**: $I \to x^2/(2(2+b))$, which at $b=0$ is $x^2/4 = \tfrac98\cdot\tfrac{2x^2}{9}$.
7. **The resonance is finite and correct at $y=-1$ for $b>0$**, and matches `I_quadrature`
   approaching it. Recon §5.2's $B(-1) = -C(1)$ closed form is an independent check.
8. **$N$ is constant** over the nine $b = 0.2$ fixture cases of
   `ComputeTargets/tests/test_quadsource_integral.py` — **imported, never edited**. Score
   `total` against `I_quadrature` with the head subtracted (README §2 (k)). **The statistic is
   constancy over $(u,v,x)$, not agreement with a predicted number** (KT §0). Recon §8 derives
   $N(b) = -\frac{(3+2b)^2}{2(2+b)^2} = -1.194214876\ldots$ at $b=0.2$; report the spread and the
   value, and say which of the two the test asserts on.

**Deliberate breakage (mandatory).** Before you finish, break your own implementation in each of
these four ways, one at a time, and record in the log **which tests fail and which do not**:

- flip the sign of the $Y\mathcal I_J - J\mathcal I_Y$ combination back to the printed order;
- drop the factor 2 in the off-cut branch;
- use `I_target` in place of `I_quadrature` in test 8;
- swap `type=2` for `type=3` in one Legendre call.

A break that no test catches is a missing test. Fix the test, not the record.

---

## 4. What this prompt does not do

- It does not touch `ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
  `AdaptiveLevin/`, `main.py`, `config/`, any factory, any schema, the six `extract_*.py`, or
  `docs/spec/`.
- It does not edit `ComputeTargets/tests/test_quadsource_integral.py` — **imported, never edited**,
  as `prompts/radiation-oracle` and `prompts/tolerance-convergence` prompt 06 both were.
- It does not edit `ComputeTargets/tests/kohri_terada.py`.
- It does not touch the hand-over. No prompt in workstream A does.
- It does not implement the oscillation average (recon §10 item 7).
- **It does not resolve a disagreement with the recon by choosing.** If the recon is wrong about
  something, that is a finding: record it, open a §3 issue, and — if it invalidates a test — stop
  and ask.

---

## 5. Acceptance

1. The module exists, is a pure function of $(v,u,x,b)$, and moves with no tolerance.
2. All eight test groups pass, and the four deliberate breakages are recorded in the log with the
   tests each one caught.
3. Test 5's $b=0$ tie holds to the precision recon §7 measured ($\le 4.5\times10^{-15}$ except at
   $u = 0.01$, where eq. (22)'s own rounding floors it at $\sim4\times10^{-10}$ — quote both).
4. Test 8 reports the spread of $N$ over the nine $b=0.2$ cases **with the reference's own error
   beside it** (README §5 rule 8).
5. `ComputeTargets` rises by exactly the number of test methods added and does not fall;
   `CosmologyModels` unchanged. Baselines in the orchestrator prompt.
6. `black --check` clean on every file in the diff.
7. `IMPLEMENTATION_STATE.md` exists and is correct (§8), and `docs/OPEN_ISSUES.md` is updated in
   the same commit.

---

## 6. Stop conditions — stop and ask the user

- Test 8's $N$ **drifts** with $(u,v,x)$ rather than being constant. That is a finding about the
  pipeline or about the mapping, not a tolerance to loosen. Report the drift and its shape.
- $N$ is constant but is **not** $-\frac{(3+2b)^2}{2(2+b)^2}$. Recon §8 says a constant-but-wrong
  $N$ is a statement about its step 1 or step 3. Report the measured constant.
- The recon's §9 brief does not construct — a Legendre convention does not check out, or $A$, $B$,
  $C$ disagree with the raw calls.
- Test 5 fails. `kohri_terada` is landed, measured and pinned; a disagreement is yours, not its.
- You cannot decide item 5 of §2 (the $y>1$ continuation) without reading Gervois–Navelet.

---

## 7. The log

`logs/01-domenech-general-b-oracle.md`, template as `qcd-background-audit` README §5.1. Beyond that
template, this prompt's log must carry:

- **the deliberate-breakage record** of §3, as its own section;
- for the $y>1$ continuation, which of the two options of §2 item 5 you took and why;
- the measured $N$ and its spread, with the reference's own error;
- anything in the recon you found to be wrong, in "Observations not acted on" **and** as a §3 issue.

---

## 8. The board

**`IMPLEMENTATION_STATE.md` does not exist.** This prompt creates it, in the shape of
`prompts/radiation-oracle/IMPLEMENTATION_STATE.md`: a §1 status board with a row per prompt
grouping from campaign README §3 (00, A1, A2, B1, B2, C1, D1, D2, E1, E2), a §2 item-level table,
§3 Active and unresolved issues, §4 Resolved issues, and the maintenance-rule blockquote.

Prompt **00** is already landed and belongs on it as a completed row pointing at
[`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md); it has no log, by its
own §8, and the board should say so rather than leaving a gap.

`docs/OPEN_ISSUES.md`: `[01-general-w-normalisation-is-predicted-not-measured]` closes on the
**`radiation-oracle` board's §4**, not this one — it is that campaign's issue. Delete its row from
the index, correct the count and the date, and add any issue this prompt opens. §1.9 of the index
becomes empty; leave the heading with a line saying the campaign's issues are all closed, as other
sections do.
