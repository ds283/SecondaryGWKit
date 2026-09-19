# Hand-over campaign — implementation state

**Last updated:** 2026-09-20 · **Status: STARTED — 2 of 10 groupings landed (00, A1).** Prompt 00
landed the reconnaissance as
[`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md); **prompt
01 (A1) has landed the general-$b$ oracle** as `ComputeTargets/tests/domenech.py` with
`ComputeTargets/tests/test_domenech_oracle.py`, and prompt 01 §5's acceptance is met. A2 and A3 are
written and not run; B1, B2, B3, B4, C1, C2, D1, D2, E1 and E2 are groupings only. Two issues are
open in §3, one is closed in §4 on another board.

**Campaign:** [`README.md`](README.md) ·
**Background:** [`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) ·
[`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.1

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit** —
> the row added, moved or deleted, and the count and date in its header corrected. An issue owned
> by another board is moved to **that** board's §4 and its row deleted from the index. The index is
> an index: one line per issue, pointing here. Where the two disagree, this board is right.
> See `CLAUDE.md`.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 00 | [Domènech reconnaissance](00-domenech-reconnaissance.md) | input to **A1** | Fable 5.1 | ✍️ yes | ✅ 2026-09-19 | *"Read the Domenech papers and write down what the kernel says"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | **none, by its own §8** — it lands no code, and its whole output is [`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md), which is the record. §11 of that document is a later addendum by Claude Opus 5. |
| A1 | [The Domènech general-$b$ oracle](01-domenech-general-b-oracle.md) | **A1** | Opus 5 | ✍️ yes | ✅ 2026-09-20 | *"Land the Domenech general-b oracle with its tests"* | [`logs/01-…`](logs/01-domenech-general-b-oracle.md) |
| A2 | [The realistic-flavour large-$x$ harness](02-realistic-flavour-large-x-harness.md) | **A2** | — | ✍️ yes | ⬜ not run | — | — |
| A3 | [The policy-geometry census](03-policy-geometry-census.md) | **A3** | — | ✍️ yes | ⬜ not run | — | — |
| B1 | Remove the gap | **B1** | — | ⬜ no | ⬜ | — | — |
| B2 | Score the unmasked residue | **B2** | — | ⬜ no | ⬜ | — | — |
| B3 | Give the seam a contract | **B3** | — | ⬜ no | ⬜ | — | — |
| B4 | Make `incomplete` unreachable | **B4** | — | ⬜ no | ⬜ | — | — |
| C1 | The stop point as an identity | **C1** | — | ⬜ no | ⬜ | — | — |
| C2 | The policy vocabulary | **C2** | — | ⬜ no | ⬜ | — | — |
| D1 | The $G_k$ consumer spline | **D1** | — | ⬜ no | ⬜ | — | — |
| D2 | The phase as carried producer→consumer | **D2** | — | ⬜ no | ⬜ | — | — |
| E1 | Re-take the trade-off on the current grid | **E1** | — | ⬜ no | ⬜ | — | — |
| E2 | Choose the depth | **E2** | — | ⬜ no | ⬜ | — | — |

`README.md` §3 groups B into four (B1, B2, B3, B4) where the orchestration prompt's §8 listed
three; §4 item 5 of that README is the argument for splitting the contract (B3) from the invariant
(B4), and both rows are carried here.

**This table is the status, not `README.md` §3.** That README's §3 rows and its "Status" blockquote
carry the annotations they were written with — A1 is still described there as *"written 2026-09-19,
not run"* — and they are the record of what was **asked**. They are not maintained per prompt; this
board is (campaign README §5 rule 4). Where the two disagree, this one is right.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| 00 | **INPUT** | Read both Domènech papers from the committed LaTeX and say what the general-$b$ kernel is, which convention every symbol is in, where the branch cuts are, what the two papers disagree about, and what a correct implementation must reproduce. Lands no code. | 00 | ✅ **Done, 2026-09-19.** [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md). Found **two sign errors in the published review** — (4.10) `eq:Isimple` prints $\big(J_{b+1/2}\mathcal I_Y - Y_{b+1/2}\mathcal I_J\big)$ where its own (4.7) and (4.9) give the opposite order, and (4.12) `eq:Isimple2`'s $\cos$ term is signed against its own $\sin$ terms — resolved both of README §2 (i2)'s apparent inter-paper discrepancies ($I_{2020} = 2c^2I_{\rm rev}$; the asymmetric factor of 2 is $\Gamma[3]$ and is correct), confirmed §2 (i3)'s resonance structure, and **derived** $N(b) = -\frac{(3+2b)^2}{2(2+b)^2}$ rather than leaving it predicted. §11 was added afterwards from the arXiv version history: the review's (4.7) was corrected between v1 and v2 and **the correction was not propagated** to (4.10) or (4.12). Two defects in the document itself are open in §3 below, both found by A1. |
| A1 | **FACILITY** | The general-$b$ oracle in the tree beside `kohri_terada.py`: the **corrected** (4.10) with the $x\to\infty$ coefficients (3.3)/(3.4) substituted and the outer Bessels kept exact; the doubly asymptotic (4.12) with its $\cos$ sign flipped; a quadrature of the finite-$x$ (4.11); $A$, $B$, $C$ separately callable; and a head helper. Tied to the nine $b = 0.2$ fixture cases and bridged to Kohri & Terada at $b = 0$. | 01 | ✅ **Done, 2026-09-20.** `ComputeTargets/tests/domenech.py` (782 lines) with **22 test methods** in `test_domenech_oracle.py`, covering prompt §3's eight groups plus two acceptance tests. **$N(b)$ is now measured:** $-1.194214876033$ on all nine $b = 0.2$ cases, spread **5.285e-13** (4.4e-13 relative), worst deviation from the derived $-\frac{(3+2b)^2}{2(2+b)^2}$ **4.130e-13**, each inside a per-case bound built from the quadrature's declared error (1.7e-14 to 6.9e-12) and the pipeline's own (1.0e-12 to 1.1e-11); against $I_{\rm rev}$, $\lvert N+1\rvert \le$ **3.461e-13**. The $b = 0$ tie to `kohri_terada` holds at **$\le4.0\times10^{-15}$ except at $u = 0.01$, where eq. (22)'s own rounding floors it at 3.9e-10**. Recon §2.4's, §4.1's and §4.2's tables are reproduced to the digit. **The $y>1$ region is refused, not continued** (recon §6.2's rule is inferred, not read — recon §10 item 2); `I_quadrature` is defined there and ties to $\tfrac98$ eq. (22) at 2.18e-15 on `T-first`. All four of prompt §3's deliberate breakages were applied and caught by named tests (log, "The deliberate-breakage record"); breakage 4 exposed a missing anchor in the two Wronskian tests, which was added. `ComputeTargets` 530 → **552** (+22, exactly the methods added), `CosmologyModels` **39**, `LiouvilleGreen` **148** (skipped=1); no existing file in the diff. |
| A2 | **FACILITY** | Extend `docs/radiation-oracle/large_x.py` from the exact flavour to the realistic one, with and without `drop_first_WKB_sample`, at $x_{\rm resp}$ to $10^7$–$10^8$. The single measurement that separates the clamp term from the phase re-spline term. | 02 | ⬜ **Written, not run.** |
| A3 | **MEASUREMENT** | Read-only census over stored `GkSourcePolicyData` rows: how many **source-grid intervals** `crossover_z` has on each side, against the stored `quality` band, plus the type/quality census and the `fail` rows. Needs a datastore. Answers README §7 **D5**. | 03 | ⬜ **Written, not run.** |
| B1 | **REMEDY** | Remove the $T_k$ seam gap by one of README §7 **D1**'s three recorded constructions. Not a tuning decision: holding a phase constant across up to 0.44 rad is wrong under any objective. | — | ⬜ **Grouping only.** |
| B2 | **MEASUREMENT** | Re-run A2 at the same configurations with the gap gone and attribute what is left, by term. | — | ⬜ **Grouping only.** Blocked on B1 and A2 by README §4 item 2's masking argument. |
| B3 | **REMEDY** | Make `quality = "incomplete"` and `type = "fail"` load-bearing; add the missing `_classify_crossover` tests; fix the ungated dump at `main.py:2929`. No prerequisites. | — | ⬜ **Grouping only.** |
| B4 | **REMEDY** | Size the $G$ overlap by construction with L1/L2 so that `_classify_crossover` cannot fail on the grid that ships. Needs A3 (the unit is grid intervals) and README §7 **D7**. | — | ⬜ **Grouping only.** |
| C1 | **REMEDY** | Tighten `find_phase_extremum`'s `root_scalar` to a floor justified by the method; key the two WKB row types on the numeric row rather than on an absolute-`1e-7` match of a $z\sim10^{12}$ float; filter the stored initial values. | — | ⬜ **Grouping only.** |
| C2 | **REMEDY** | One spelling of `numeric_policy` across both metadata objects, and the maximise-numeric branch either reachable or deleted; fix the two warning messages that lose the offending value. Changes a datastore key, which is free. | — | ⬜ **Grouping only.** Owns the three issues README §2 (o) and (p) record, which **prompt 03 opens formally**. |
| D1 | **REMEDY** | The $G_k$ consumer spline at the hand-over: `GkSourcePolicyData.py:638-752` splines the numeric $G$ over source-grid nodes carrying up to 1.24 rad of oscillation per interval, at ×631–×37,700 the solver's error. | — | ⬜ **Grouping only.** Needs A3 first (README §4 item 1). |
| D2 | **SCOPE QUESTION** | The phase as carried producer→consumer, $\delta\theta\simeq h^4x/384$ growing linearly in $x$, plus the anchoring floor and the storage granularity beneath it. The largest scope question in the campaign — README §7 **D4**. | — | ⬜ **Grouping only.** |
| E1 | **MEASUREMENT** | Re-take `[06-source-spline-residual-vs-handover]`'s stale trade-off table on the curvature-criterion grid, at $b = 0$ **and** $b \ne 0$ — which A1 has now made scoreable. | — | ⬜ **Grouping only.** |
| E2 | **DECISION** | Scan $x_T$ against both oracles at large response $x$, at $b = 0$ and $b = 0.2$, with the gap gone and the residue attributed. Move `TkNumericIntegration.py:148-149`'s window, or record on measurement that it stays. | — | ⬜ **Grouping only.** Last, by README §4 item 3. |

---

## 3. Active and unresolved issues

- **[01-recon-off-cut-closed-form-is-ill-conditioned]** *(opened 2026-09-20 by prompt 01)* —
  [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) §9's implementation
  brief prescribes §5.2's second boxed form for the **off-cut** coefficient $C(\tilde y)$, a
  difference of two ${}_2F_1$ at argument $\tfrac{1-\tilde y}{2}$. The form is correct — it
  reproduces `mpmath`'s `legenq(..., type=3)` to $10^{-48}$ term by term at 50 digits — but it is
  **catastrophically ill-conditioned away from $\tilde y = 1$**. At $b = 0.2$, $\tilde y = 150$ its
  two terms are each $2.2426\times10^{5}$ and their difference is $1.02\times10^{-8}$: a 13-order
  cancellation, which amplifies the $1.1\times10^{-17}$ by which a double's `b` differs from the
  number meant into a per-cent-level error in the result. Measured against the raw Olver call, and
  **identical at `mp.dps` 30, 60 and 120** — raising the working precision does not help, because
  the loss is in the inputs: **6.2e-15 at $\tilde y = 1.5$, 9.5e-10 at $\tilde y = 40$, 8.6e-08 at
  $\tilde y = 150$.** The `q-smooth` fixture at $b = 0.2$ sits at $\tilde y = 149.5$, so this is a
  fixture shape and not a corner. Recon §10 item 5 left exactly this untested and this is its
  answer. **Impact:** documentary and prospective. Nothing in the tree is wrong — prompt 01 used the
  2020 paper's own $1/\tilde y^2$ form instead (`1912.tex:743-747`, DLMF 14.3.7), which has no
  cancellation anywhere, is regular at $b = 0$, and agrees with the raw call to rounding at every
  $\tilde y$ tried — but a later agent re-implementing from §9 would inherit the defect, and §5.2's
  box is still the right form **on-cut**, where `domenech._R_nu` uses it. **Next step:** an
  additive subsection or an inline caveat in recon §5.2 and §9 saying which form to use where, by
  whoever next owns `docs/handover/DOMENECH-KERNEL-RECON.md` (CLAUDE.md invariant 6 — the document
  was correct for what it measured, so this is an addition, not a rewrite). Measurement:
  [`logs/01-domenech-general-b-oracle.md`](logs/01-domenech-general-b-oracle.md), "Deviations from
  the prompt" §1. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[01-recon-section-7-N-symbol-drops-a-pi]** *(opened 2026-09-20 by prompt 01)* —
  [`DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md) §2.2 defines
  $\mathcal N \equiv \pi4^b\Gamma^2[b+\tfrac32]\frac{2b+3}{b+2}$, which at $b = 0$ is
  $\pi\cdot\tfrac\pi4\cdot\tfrac32 = \tfrac{3\pi^2}{8}$. §7's $b\to0$ reduction prints
  "$\mathcal N = \tfrac{3\pi}{8}$" — the same constant **without** the $\pi$, i.e. the object
  `domenech.kernel_constant(b)` and not $\mathcal N$. **Impact:** documentary only. §7's displayed
  reduction of $I_{\rm target}(b=0)$ to $\tfrac98\times$ Kohri & Terada's eq. (25) is correct as
  written and was reproduced here to rounding (prompt 01's test 5 holds at $\le4.7\times10^{-15}$
  away from $u = 0.01$), so no arithmetic depends on the symbol. But a reader hand-checking §7
  against §2.2 loses a factor of $\pi$ and will conclude one of the two is wrong. **Next step:**
  one symbol, by whoever next owns that document — naturally together with the issue above, which
  touches the same file. Measurement:
  [`logs/01-domenech-general-b-oracle.md`](logs/01-domenech-general-b-oracle.md), "Observations not
  acted on" item 2. Indexed at `docs/OPEN_ISSUES.md` §1.1.

**Not on this board, deliberately.** The three issues `README.md` §2 (o) and (p) record —
`[03-gksource-policy-accepts-a-value-that-raises]`,
`[03-quadsource-policy-vocabulary-differs]` and `[03-gk-classification-is-diagnosed-and-ignored]` —
are **prompt 03's to open**, by that prompt's own scope, and are indexed in
`docs/OPEN_ISSUES.md` §1.1 as *"none yet"* until it runs. Prompt 01 did not adopt them: they are
not in its files-may-touch list and opening another prompt's issues early would make this board
disagree with the prompt that owns them.

---

## 4. Resolved issues

Closed by prompt 01, 2026-09-20:

- **[01-general-w-normalisation-is-predicted-not-measured]** — *opened 2026-09-18 with the
  Kohri–Terada audit, and **owned by the `radiation-oracle` board**, where it is closed.* That
  campaign's oracle covers $b = 0$ only, so $N = -9/8$ was measured there and the general-$w$
  $N(b) = -(3+2b)^2/(2(2+b)^2)$ was **predicted and unchecked**, with the nine $b = 0.2$ fixture
  cases unscored.

  **Resolution.** Prompt 01 measured it. `total_from_I_rev` scores
  `evaluate_QuadSource_integral`'s `total` against a quadrature of Domènech (4.11)
  `eq:Isimpledef` inside the corrected (4.10), integrated between the code's own limits, on all
  nine $b = 0.2$ cases of `test_quadsource_integral` in the exact flavour at the reference pair
  $(10^{-45}, 10^{-12})$. **$N = -1.194214876033$, constant: min and max agree to 5.285e-13
  (4.425e-13 relative), over $x$ from 5.3 to $1.9\times10^3$, $u$ from 0.01 to 10, and both
  Gervois–Navelet branches.** The worst deviation from the derived
  $-\frac{(3+2b)^2}{2(2+b)^2} = -1.194214876033$ is **4.130e-13**, and against $I_{\rm rev}$ —
  where the prediction is $-1$ exactly — **3.461e-13**. Every per-case deviation is inside the sum
  of the two sides' own declared errors (the quadrature's 1.7e-14 to 6.9e-12, the pipeline's
  1.0e-12 to 1.1e-11). The constancy is the load-bearing statistic
  (`KOHRI-TERADA-ORACLE.md` §0) and it is what the test asserts first; the value is asserted
  separately, and prompt 01's breakage 1 shows the two failing independently. This also removes
  that issue's stated obstacle — its "next step" asked whether a general-$w$ quadrature oracle
  built from the code's own $\Phi$ was worth having "given that it shares machinery with the thing
  it checks". It does **not** share machinery: `domenech.py` is a transcription of two published
  papers, with the outer Bessels from `scipy.special` and the Legendre sector from `mpmath`, and
  it is bridged at $b = 0$ to `kohri_terada.py`, which a different campaign transcribed from a
  third paper. **Closed on the `radiation-oracle` board's §4**, per that campaign's ownership and
  campaign README §5 rule 4; the row is deleted from `docs/OPEN_ISSUES.md` §1.9, which is now
  empty. Measurement:
  [`logs/01-domenech-general-b-oracle.md`](logs/01-domenech-general-b-oracle.md), "Verification
  performed", acceptance item 4.
