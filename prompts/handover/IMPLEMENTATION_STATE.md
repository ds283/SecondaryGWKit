# Hand-over campaign — implementation state

**Last updated:** 2026-09-23 · **Status: STARTED — 3 of 10 groupings landed (00, A1, A2).** Prompt
00 landed the reconnaissance as
[`docs/handover/DOMENECH-KERNEL-RECON.md`](../../docs/handover/DOMENECH-KERNEL-RECON.md); **prompt
01 (A1) has landed the general-$b$ oracle** as `ComputeTargets/tests/domenech.py` with
`ComputeTargets/tests/test_domenech_oracle.py`; **prompt 02 (A2) has landed the realistic-flavour
large-$x$ harness** as [`docs/handover/realistic_large_x.py`](../../docs/handover/realistic_large_x.py)
with [`REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md), and **has separated the
clamp term from the representation term** — the measurement
`[12-handover-clamp-error-in-production]` records as impossible. A3 is written and not run; B1, B2,
B3, B4, C1, C2, D1, D2, E1 and E2 are groupings only. **Seven** issues are open in §3 — four from
prompts 01 and 02, and **three opened 2026-09-23 by the A3 baseline run**, which was stopped by the
user after a read of its cost showed the shipped quadrature tolerance pair unreachable on
production's squeezed triples; one is closed in §4 on another board.

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
| A2 | [The realistic-flavour large-$x$ harness](02-realistic-flavour-large-x-harness.md) | **A2** | Opus 5 | ✍️ yes | ✅ 2026-09-20 | *"Separate the hand-over clamp from the representation floor"* | [`logs/02-…`](logs/02-realistic-flavour-large-x-harness.md) |
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
| A2 | **FACILITY** | Extend `docs/radiation-oracle/large_x.py` from the exact flavour to the realistic one, with and without `drop_first_WKB_sample`, at $x_{\rm resp}$ to $10^7$–$10^8$. The single measurement that separates the clamp term from the phase re-spline term. | 02 | ✅ **Done, 2026-09-20.** `docs/handover/realistic_large_x.py` (908 lines) and [`REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md); **60 cells in 10,772 s**, no Ray, no datastore, checkpointed per cell. **The control reproduces KT §8 Table 8.1 on all sixteen rows and ten columns, digit for digit**, independently re-checked against a separately-taken `large_x.py` run. **The two terms are separated and the $2\times2$ is additive:** clamp term **3.91e-03 – 9.12e-01**, representation term **4.08e-05 – 2.68e-03**, ratio **11.6 – 1034**, interaction $\le$ **2.82e-05** of the clamp term and $\le$ **1.03e-02** of the representation term; every term exceeds its own error bar by $\ge$ **3.5e+03**. **Both of the prompt's $x$-scaling guesses are refuted by measurement:** the clamp term is $x$-independent (slopes −0.07/+0.03/+0.03) because the fixture pins the hand-over at $x_T = 19.1$, giving the campaign README's 0.44 rad held phase at every rung; and the representation term is *also* flat (−0.06/−0.15/−0.12) rather than growing like $h^4x/384$, which over-predicts by $10^5$ in trend — `TkSourceFunctions` no longer re-splines the growing phase, so `GkTk-remedial` prompt 10's fix is confirmed end-to-end in `total`. `drop_first_WKB_sample` proved **inert in the exact flavour** (bit-identical totals, Table 6), so the exact half uses a matched `WKB_region` truncation; the two clamp terms then agree to five significant figures. Gap opened: **1.00 fixture grid step = 1.00 source-grid step = 1.92× production's median, 1.05× its maximum**. Set-up cost is **flat** (0.02–0.10 s, ratio 1.1–2.0), refuting prompt §3 item 1; the **integral** is the binding cost (33–3239×), growing 1.7×/3.6×/**8.6×** per decade by shape, so `q-smooth` realistic stops at $x_{\rm resp} = 10^5$ — prompt §7's second stop condition met by measurement. Suites unchanged. Two issues opened in §3. **Note added 2026-09-22 by `prompts/run-registry` prompt 02: `realistic_large_x.py` predates the run registry by design and is deliberately not adopted into it.** It gates checkpoint reuse on a SHA-256 of its own source, so **editing it by one character discards all sixty cells** in `var/runs/realistic_large_x_cells.jsonl` and costs a three-hour recomputation; that hash is what proved, in four seconds, that the committed script produced the published numbers. Its value as a fixed artefact exceeds its value as a registry client. Anyone proposing to touch it should read `prompts/run-registry/IMPLEMENTATION_STATE.md` `[02-realistic-large-x-is-outside-the-registry]` first. |
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

- **[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]** *(opened 2026-09-23
  by the A3 baseline run, not by a prompt)* — on production's **squeezed** triples the source
  integral's tolerance pair cannot be met, the Levin bisection runs to
  `DEFAULT_LEVIN_MAX_DEPTH = 20`, and the row is stored unconverged at **220× the cost of a row
  that stops short of the cap**. Measured over all **7600** `QuadSourceIntegral` rows in
  `var/datastores/handover-A3-baseline-lambdacdm-shard*.sqlite`, on the tree at `704a12e` (clean):

  | `WKB_Levin_max_depth` | rows | mean `compute_time` |
  |---|---|---|
  | NULL (no Levin call) | 2489 | 0.062 s |
  | 1–19 | 3054 | 0.776 s |
  | **20** | **2057** | **170.6 s** |

  The cap is a cliff, not a scaling: `max_depth` on the expensive rows is **20 exactly, with zero
  variance**. Depth-20 rows carry **97.5 of the store's 98.2 CPU-hours**, and **43.2 %** of them
  (889) are stored with `total_converged = 0`. Of the 195 rows costing more than 60 s, **74.4 %**
  are unconverged. The cost is **entirely Levin**: `WKB_Levin_elapsed` sums to **351,296 s** of the
  **353,489 s** of `compute_time`, against **9 s** of `numeric_quad_compute_time` across all 7600
  rows.

  **The mechanism.** `_adaptive_levin` accepts a region on
  `resolved = abserr < local_atol or relerr < rtol` (`AdaptiveLevin/levin_quadrature.py:1991`, and
  again at `:2179`), and `local_atol` is `atol` distributed across subregions **by length share**
  (`_local_atol`, `:1672`). These rows reach `WKB_Levin_num_regions` of up to **1,530,190**, so a
  region's share of `atol = 1e-32` is around **1e-38** and the first branch is dead. Everything
  must then clear `rtol = 1e-8` against a phase representation that delivers about six digits: the
  median `total_abserr / |total|` on the expensive rows is **2.15e-06**. `levin_quadrature.py:2327`
  describes this state in its own words — *"the cost can be two orders of magnitude higher than
  necessary"* — and the measured ratio is 220.

  **Which constant, and why the change that caused it was reasonable.** `5255ac0` (2026-09-10)
  tightened `DEFAULT_QUADRATURE_ATOL` from `1e-25` to `1e-32`, on `prompts/source-remediation`
  log 12's finding that at `1e-25` ~58 % of production work items "converged" before doing any
  work. That finding is not in dispute and the direction was right. What the change also did,
  unmeasured, is remove the only reachable branch for **this** geometry: the expensive rows have
  median `|total| = 3.31e-24`, so at `1e-25` `atol` was met immediately and at `1e-32` it cannot be
  met at any subdivision. The regime moved from `atol`-bound to `rtol`-bound, and `rtol = 1e-8` is
  unattainable here. **The defect is the pair, not the tightening**: nothing sized `rtol` for a
  phase that cannot deliver eight digits, because until `5255ac0` `atol` was hiding it.

  **Why the sweep that already exists did not see it.** `prompts/tolerance-convergence` prompt 06
  swept exactly this axis and reported `atol` **inert over twenty-eight decades**
  ([`QUADSOURCE-READONLY.md`](../../docs/tolerance-convergence/QUADSOURCE-READONLY.md) §0
  finding 1, §4). **That measurement stands and is not superseded.** It measured the *residual*, on
  the eighteen offline cases of `test_quadsource_integral.py`, and on those cases `atol` is inert.
  It could not see this for two reasons, both structural: its instrument has **no `(k, q, r)`
  triangle** — the squeezed geometry below does not occur in a single constant-$w$ configuration —
  and the statistic that moves is the **cost**, which a residual sweep does not record. Its own
  finding 4 recorded `DEFAULT_LEVIN_MAX_DEPTH = 20` and `limit = 100` as **never chosen**; this
  entry is what that costs. A different question on a different instrument, not a contradiction.

  **The geometry, exactly — three conditions, all necessary.** (i) **`r == k ≫ q`**, the hard leg
  on the response and a soft `q`: every worst triple has this shape, and the mirror `k ≪ q ≈ r`
  costs 0.3 s. (ii) **mid-range `k`**: the blow-up peaks at `k = 9.7e6`/Mpc and is **absent at both
  ends** — `k = 1e5` has zero rows over 60 s, and so does `k = 3e8`. (iii) **low `z_response`**,
  i.e. large `eta_response`: within one triple the cost grows roughly as $\eta_R^3$ and then
  flattens as the cap binds. For `(k, q, r) = (9.7e6, 9.85e5, 9.7e6)`: **15.5 s** at $z = 8343$
  ($\eta_R = 50.8$), **737 s** at $z = 1589$, **6245 s** at $z = 174$, **12,714 s** at $z = 11$.
  Binned by $k\eta_R$ the **median** cost is flat at ~1.3 s across **twelve decades** — Levin is
  doing what Levin is for — and only the mean moves. Any statistic that does not separate the tail
  will miss this.

  **Impact.** Cost, and what a stored `total` is worth. At the settings that shipped, finishing the
  baseline costs **8–10 wall-days** on six CPUs (the issue below), and 74 % of what it would add is
  unconverged. It also reaches **E1** and **B2**, which re-score against this store, and it
  narrows `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` below.

  **Measured 2026-09-23** by
  [`docs/handover/quadsource_atol_sweep.py`](../../docs/handover/quadsource_atol_sweep.py) — nine
  fixed production work items (four severe, three moderate, two controls) over eight tolerance
  pairs, through main.py's own pipeline by five exact substitutions. The zero point is the
  baseline's own rows, read back as lookups: 7237.2 s over the nine, 7 at depth 20, 5 unconverged.
  Each cell is the worst relative difference of `total` against that zero point over the nine:

  | `atol` | `rtol` | worst $\lvert\Delta\rvert/\lvert total\rvert$ | time, 9 cases | speed-up |
  |---|---|---|---|---|
  | 1e-22 | 1e-8 | 1.8e+00 | 3.0 s | — |
  | 1e-25 | 1e-8 | 1.8e+00 | 2.5 s | — |
  | 1e-28 | 1e-8 | 3.2e-01 | 5.6 s | — |
  | 1e-30 | 1e-8 | 1.1e-02 | 100.5 s | — |
  | 1e-32 | 1e-5 | 1.1e-02 | 4.9 s | ×1477 |
  | 1e-32 | 1e-6 | 1.1e-02 | 31.1 s | **×233** |
  | **1e-32** | **1e-7** | **1.9e-06** | **1060.5 s** | **×6.8** |
  | 1e-32 | 1e-8 | — (reference) | 7237.2 s | 1 |

  **`rtol` is the lever and `atol` is not**, which is the opposite of what the cost alone
  suggested. Relaxing `atol` collapses the cost and **destroys the answer**: at 1e-22 and 1e-25
  several cases return with the *wrong sign*, and 1e-28 is still 32 % out. So `5255ac0` was right
  in a stronger sense than log 12 claimed for it — 1e-25 is too loose on *this* geometry, not only
  on the grounds log 12 gave — and `atol` is emphatically **not** inert here, which is a second
  way prompt 06's finding does not transfer. At production `atol`, `rtol = 1e-7` reproduces every
  one of the nine to **1.9e-06** at **1/6.8** of the cost, and `rtol = 1e-6` gives seven digits on
  eight of nine at **1/233**, with one case (the $z = 1589$ moderate) pinned at 1.1e-02 — that
  same case pins the 1e-30 and 1e-5 rows too, so it is a property of the case and not of the
  tolerance.

  **Two qualifications ship with this table, and the first is the binding one.** (i) The
  `(1e-32, 1e-8)` reference is itself stored `total_converged = 0` on **5 of the 9** cases, so
  "agrees to 1.9e-06" means *agrees with a number the code declined to certify*, not with truth.
  This sweep has no independent oracle and `analytic_rad` cannot be one — prompt 06's finding 3
  established that it is computed at the caller's own pair and moves with it. (ii) All nine cases
  sit at $z_{\rm response} \ge 43.7$; the 1024 items at $z \le 6.32$ that
  `[a3-baseline-quadsource-integrals-are-1680-short]` records as never attempted are **unmeasured
  at every tolerance**, and their $\eta_R$ reaches 2.8× the largest sampled here. The speed-ups
  above must not be extrapolated into that block, which is where most of the cost is.

  Incidentally confirming the issue below: at `rtol = 1e-6` seven of the nine rows are still
  flagged unconverged while agreeing to seven digits. At `atol = 1e-32` the `atol` branch of
  `resolved` can never fire, so `total_converged` carries almost no information.

  **The full record, with every table and the structural correction of §6, is
  [`docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md`](../../docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md)**,
  written against `050a7e3` on a clean tree. It is transcribed rather than regenerable: the sweep
  store it was measured on is disposable and is being deleted, so the numbers live in the document
  (`prompts/run-registry` README §0 item 4).

  **Phases 2, 3a and 3b, measured 2026-09-23/24**, extend that table into the two bands phase 1
  could not see. Phase 1's nine cases all sit at $z_{\rm response}$ between 43.7 and 1589, which
  is one band of three.

  | band | cases | `rtol` 1e-6 | `rtol` 1e-7 | cost at 1e-7 | what limits it |
  |---|---|---|---|---|---|
  | seam, $z_{\rm response} > z_{\rm min}$ | 9 | 1.01e-06 | 9.97e-07 | 7.7 s | floor ~1e-06 |
  | mid, $z_{\rm response}\sim10^3$ (c5) | 9 | **1.1e-02** | 1.9e-06 | 1060 s | **quadrature** |
  | low, $z_{\rm response} \le 6.32$ | 4 | 1.08e-05 | 2.84e-06 | 2598 s | floor ~1e-05 |

  (Each figure is the worst relative difference over that band's cases against its own tightest
  rung, which is 1e-8 for the seam and low bands and 1e-8 for the mid band.)

  **`rtol = 1e-7` is the loosest rung converged in all three bands, and 1e-6 is ruled out by the
  mid band alone.** c5 — $(k, q, r) = (3.09\text{e}6, 1\text{e}5, 3.09\text{e}6)$ at
  $z_{\rm response} = 1589$ — is the case that decides it: there `rtol` 1e-5 and 1e-6 agree with
  *each other* to 3.6e-04 and are both **1.1e-02** from the converged value, which 1e-7 reaches in
  245 regions and 1e-8 confirms to 1.0e-06 after 24,015. Two loose rungs agreeing is not
  convergence, and c5 is the counterexample inside this sweep.

  **Phase 3b settles the low band, and the plateau there is real.** Four rungs across three
  decades agree to ~1e-05; on $k = q = r = 3.05\text{e}7$, $z = 0.1$ the 1e-7 and 1e-8 answers are
  **bit-identical** (same value, same 50,043 regions); and the 1e-7 → 1e-8 step is *smaller* than
  the 1e-6 → 1e-8 step, which is the opposite of the c5 signature. So low $z$ is limited by the
  representation and not by the quadrature, and no tolerance reaches beneath that floor — the
  lever there is **D1**/**D2**, not this constant. The region counts say the same thing: 52 → 698
  → 152,634 → 1,238,448 across the four rungs on one case, for a change of 1e-05 in the answer.

  **For science outputs, someone should re-run at `rtol = 1e-9`** (user, 2026-09-24). There is no
  oracle that could substitute: `domenech.py` and `kohri_terada.py` are both constant-$w$ and
  production is LambdaCDM with the Saikawa–Shirai QCD equation of state, so self-convergence is
  the only instrument and one rung tighter is the only way to use it. The qualification to carry
  with that recommendation is the band structure above — at low $z$ a tighter rung will buy cost
  and no accuracy, because the floor is not quadrature; the band where 1e-9 could still bite is
  the mid one, where 1e-7 was still moving.

  **Next step:** regenerate the store at `(atol, rtol) = (1e-32, 1e-7)` (see that issue for the
  decision and the scope). **The remedy for the constant itself is not this campaign's to take.** `DEFAULT_QUADRATURE_ATOL` and `DEFAULT_QUADRATURE_RTOL` belong to
  `prompts/levin-refactor` and `prompts/qsi-phase-groups` (`tolerance-convergence` README §0.4),
  and `DEFAULT_LEVIN_MAX_DEPTH` is owned by nobody. This board measures and hands over. Indexed at
  `docs/OPEN_ISSUES.md` §1.1.

- **[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]** *(opened 2026-09-23 by the A3
  baseline run, not by a prompt)* — `QuadSourceIntegral` records `total_converged` and
  `total_phase_limited`, and **nothing reads either**. A row whose bisection ran to the depth cap
  without meeting a tolerance is written, keyed, and served to every later consumer exactly like a
  row that converged. In the A3 baseline store that is **889 rows** (43.2 % of the 2057 at depth
  20, 11.7 % of all 7600). The flag is there, so the data is recoverable by anyone who thinks to
  select on it; nothing in the pipeline does, and no `QuadSourceIntegral` consumer joins on it.

  **Impact.** Every downstream use of this store, and of any store built at a tolerance pair that
  reaches the cap — so it is the same population as the issue above and disappears with it if that
  pair changes. Today it is latent rather than wrong: the stored `total` is the integral the code
  computed, and `levin_quadrature.py:2328` says the estimate is "usually still good". What is
  missing is any way for a consumer to *decline* it.

  **Next step: subsumed by the regeneration — user decision, 2026-09-23.** The store is to be
  rebuilt from scratch at the chosen tolerance pair, so these rows are not cleaned, they cease to
  exist. What survives the rebuild is the *design* question, and only if the adopted pair still
  reaches the depth cap: recorded so that it is not lost, not so that it is acted on now. Two things are true about it. First, there is **no defined
  route to remove such rows**: the only mechanism in the tree is `main.py --prune-unvalidated`,
  which drops rows whose validation flag is clear, so removing these would mean clearing that flag
  by hand on a selection the schema was not designed to express — a hand edit against a datastore,
  which is the kind of thing this project does not do. Second, it may not need doing: if the sweep
  above moves the tolerance pair, **every one of these rows is regenerated anyway**, because the
  pair is part of the datastore key. The decision therefore waits on the sweep. If the pair does
  *not* move, the right shape is probably a filter at the consumer and a `--require-converged`
  gate, not a deletion. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[a3-baseline-quadsource-integrals-are-1680-short]** *(opened 2026-09-23 by the A3 baseline run,
  not by a prompt)* — the LambdaCDM baseline store holds **7600 of the 9280**
  `QuadSourceIntegral` objects its own geometry defines, and the run that was filling it was
  **stopped by the user on 2026-09-23** rather than allowed to finish. The count is exact and is
  set arithmetic on store serials, not on floats: 145 response redshifts × 64 triangle-closing
  `(k, q, r)` triples = 9280; the store holds 7600; **1680 are missing**. (A first pass over
  round-tripped CSV floats gave 2581 and was wrong — `redshift.z` does not survive a CSV round
  trip, and 901 rows appeared to be off-grid when they were not.)

  **What is missing is the expensive end, entirely.** All 1680 are at `z_response ≤ 174.1`, and
  **1024 of them at `z_response ≤ 6.32` have no row at all** — that block has never been attempted.
  Its $\eta_R$ runs out to **13,728** at $z = 0.1$ (main.py's own "latest tau" line) against a
  largest *measured* $\eta_R$ of **4936** at $z = 8.33$, so the untouched block is a factor 2.8
  beyond anything the cost model above is fitted on. Two independent projections of the remainder
  at the shipped settings: a per-triple power law in $\eta_R$, extrapolated and capped at 4× each
  triple's largest measured cost, gives **1021 CPU-hr ≈ 8.2 wall-days** at the 5.2 effective CPUs
  the run achieved; the flat rate from the run's own last session (48 items in 6 h 55 m at a mean
  of 2710 s) gives **~10.1 wall-days**, and that is a *lower* bound because that session only
  reached $z \ge 8.3$. About **300 of the 1680** — the squeezed triples at
  $k = 3.1\times10^6$, $9.7\times10^6$ and $3.0\times10^7$ — carry roughly **90 %** of it.

  **Impact.** **E1** and **B2**, which need this store as the comparator for the hand-over change,
  and **A3** itself, whose policy rows are complete but whose integral outcomes are not. The store
  is *usable* — 7600 rows over the full $k$ grid and the whole high-$z$ range — but it is **not a
  complete response-redshift grid** and nothing that assumes one may be run against it.

  **Superseded 2026-09-23 by a user decision: the store is to be regenerated from scratch** at
  the tolerance pair the sweep above settles, rather than finished at the shipped one. That closes
  three things at once — the 1680, the 889 unconverged rows of the issue above, and **54 rows the
  first run of the sweep wrote into this store in error** (a copied `ShardedPool` primary still
  names the *source* store's shard files, fixed in `2ebb7b6`; the rows are at six non-production
  pairs and the production population was left at exactly 7600, so nothing was lost). Rebuilding
  costs **~16 minutes** for everything upstream of the integral — measured from this store's own
  timestamps, 21:02:30 to the first `QuadSourceIntegral` row at 21:18:24, covering the background
  model, both $T_k$ sectors, both $G_k$ sectors, `GkSource` and the policies — so there is no case
  for reusing any of it.

  **Next step:** do **not** restart it at the shipped tolerance pair. The sweep named in the first
  issue above decides the pair; if the pair moves, the whole store is regenerated anyway and the
  1680 are moot, and if it does not, the 8–10 days is the honest price and is a decision for
  whoever owns E1. The run itself is recorded at
  `var/runs/handover-03-a3-baseline-resume-20260923T024847` (state `killed`, stage
  "CALCULATE QUADRATIC SOURCE INTEGRALS, 92.31%" — that percentage counts **batches of 750
  dispatched**, not integrals completed, which is what made the remaining work look small). All
  four shards pass `PRAGMA quick_check` after the stop. Indexed at `docs/OPEN_ISSUES.md` §1.1.

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

- **[02-realistic-fixture-Gk-phase-is-the-superseded-construction]** *(opened 2026-09-20 by prompt
  02)* — `ComputeTargets/tests/test_phase_groups.py:299` documents `BesselPhaseGk` as built *"the
  way `GkSourcePolicyData._create_functions` builds the real one: the phase is a `phase_spline`
  over $\log(1+z')$ through (div 2pi, mod 2pi) samples"*. **That has not been true since
  `prompts/GkTk-remedial` prompt 09.** Production's `GkSourcePolicyData._build_phase`
  (`:145-190`, consumed at `:726`) returns a `PrimitivePhase` — the leading $-k\,\Delta\tau$ from
  the background table with only a small residual splined — which is precisely how that campaign
  discharged `[12-phase-spline-error-grows-with-x]`'s $\delta\theta \simeq h^4x/384$ for $G_k$
  (board M14: 4.189e-08 rad against 7.286e-03 for the same samples, a ratio of 1.739e+05).
  `TkSourceFunctions` *is* current — prompt 10 gave it a `PrimitivePhase` too — so the realistic
  flavour's $T$ half is production's and its $G$ half is the construction production abandoned.
  Consequences, in order of how much they matter: the module docstring of
  `test_quadsource_integral.py` claims the realistic flavour "is the accuracy production can
  expect", and that is **stale for $G$**; prompt 02's representation term (4.08e-05 – 2.68e-03,
  `REALISTIC-LARGE-X.md` §4) is therefore an **upper bound** on production's rather than an
  estimate of it, because it carries a term production has already removed; and the Levin cost
  growth of the issue below may be the same staleness seen from the cost side.
  **Impact:** **B2**, which re-runs this harness once the gap is gone and attributes what is left —
  it should fix the fixture before it scores the residue, or its numbers will be pessimistic by an
  unknown factor for $G$. Also any later reader of the test module's docstring. Nothing in
  production is wrong; this is an instrument that has fallen behind the thing it models.
  **Next step:** rebuild `BesselPhaseGk`/`OffsetBesselPhaseGk` on `PrimitivePhase` as
  `_build_phase` does, and correct both docstrings. Not done here: prompt 02 §5 forbids editing
  `test_quadsource_integral.py`, and `test_phase_groups.py` is not in its scope either.
  Measurement: [`logs/02-realistic-flavour-large-x-harness.md`](logs/02-realistic-flavour-large-x-harness.md),
  "Observations not acted on" item 1. Indexed at `docs/OPEN_ISSUES.md` §1.1.

- **[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]** *(opened 2026-09-20 by prompt 02)* —
  the Levin driver's region count is **flat in $x$ on the exact flavour and grows steeply on the
  realistic one**, and the growth may be a property of the fixture rather than of production.
  Measured over the $x_{\rm resp}$ ladder (`REALISTIC-LARGE-X.md` Table 8): exact **7→9**
  (`together`, six rungs), **7→13** (`T-first`), **17→30** (`q-smooth`) — flat, which is what
  Levin quadrature is for. Realistic **1,304→50,463** (`together`, peaking at $10^7$ and falling to
  15,827 at $10^8$), **720→75,570** (`T-first`), **855→31,358** (`q-smooth` over 980 to $10^5$).
  Wall time follows: 33× to 3,239× the exact flavour, growing 1.7× / 3.6× / **8.6×** per decade of
  $x_{\rm resp}$ by shape, which is what forced `q-smooth` realistic to stop at $10^5$.

  **The level is understood; the growth is not.** At the base rung the integration range spans
  5.40 decades of $(1+z)$, i.e. ~540 cells of the 100-per-$\log_{10}z$ grid, against 720–1,304
  regions — the driver bisects to about the spline knot scale, which a piecewise-cubic phase
  forces. But the knot count grows only **540 → 1,041** across the ladder (the range widens by one
  decade per rung, because `z_source_max` tracks $k$ while $z_{\rm response}$ is fixed), a factor of
  1.9, while the region count grows by 12× (`together`), 105× (`T-first`) and 37× (`q-smooth`).
  Regions per knot therefore rise from **1.3–2.4** at the base rung to **15–80** at the top: four to
  six extra levels of bisection the knot count does not account for.

  **Why it may be an artefact.** The fixture's $G$ half is the superseded raw `phase_spline` of the
  issue above, whose representation floor is $\delta\theta \simeq h^4x/384$ — **linear in $x$**.
  `_adaptive_levin` accepts a region on `resolved or phase_limited or depth_max`, and
  `phase_limited` requires `abserr <= phase_err`, so a region is forgiven only once its quadrature
  error has fallen *to* the representation floor. A floor that grows linearly in $x$ forces
  bisection that grows with $x$. If that is the mechanism, **production does not have it** — its
  $G$ phase is a `PrimitivePhase` — and this harness's whole cost curve, including the `q-smooth`
  ceiling, is a statement about the instrument rather than about the pipeline.
  **Impact:** bounded and cost-only. It does **not** touch any $N$ in `REALISTIC-LARGE-X.md`: every
  term in its Table 3 is a difference taken at fixed flavour, so a region count affects both sides
  identically, and the measured additivity (interaction $\le$ 2.82e-05 of the clamp term) would not
  survive if it did. What it touches is how far **B2** and any later re-run of this harness can
  reach, and whether `q-smooth` above $10^5$ is genuinely out of range or merely out of range for a
  stale fixture.
  **Next step:** re-run **one** cell — `q-smooth`, realistic, closed seam, $x_{\rm resp} = 10^5$ —
  with the $G$ phase built as a `PrimitivePhase`, and compare `Levin_regions` (31,358) and
  `integral_time` (514.8 s) against what is recorded here. A collapse confirms the mechanism; no
  change refutes it and points instead at the weakly-oscillatory gate (`Levin_fraction` swings
  0.09–29.1 across the factorial) or at the reference tolerance pair itself. **Deliberately not run
  by prompt 02** — out of its scope, and it needs the fixture change the issue above owns.
  Measurement: [`logs/02-realistic-flavour-large-x-harness.md`](logs/02-realistic-flavour-large-x-harness.md),
  "Observations not acted on" item 2. Indexed at `docs/OPEN_ISSUES.md` §1.1.

  **Narrowed 2026-09-23 by the A3 baseline run.** The strong form of the hypothesis above —
  *"If that is the mechanism, **production does not have it**"* — is **refuted by measurement**.
  Production's $G$ phase **is** a `PrimitivePhase`, and production has a Levin blow-up anyway:
  2057 of 7600 stored `QuadSourceIntegral` rows sit at `max_depth = 20` with up to 1,530,190
  regions, at 220× the mean cost of a row that stops short of the cap
  (`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` above). So a stale
  `phase_spline` floor cannot be the whole explanation of unbounded subdivision in this driver.
  What the new measurement supplies is a **second, sufficient mechanism** that the harness shares:
  `local_atol` is `atol` divided by a region's length share, so at 1e6 regions the `atol` branch
  of `resolved` is dead whatever the phase is, and `rtol` alone must be met. That is present in
  the harness too. **This issue is not closed**: its own question — whether the *fixture's* 12–105×
  growth in $x$ is the fixture's or production's — is still open, and the one-cell experiment
  above is still the way to answer it. What has changed is that a null result there no longer
  means the cost curve is safe, and that the experiment should record `max_depth` and the region
  count against `local_atol` rather than against `phase_err` alone.

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
