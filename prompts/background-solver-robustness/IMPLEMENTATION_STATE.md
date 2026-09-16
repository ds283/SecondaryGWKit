# Implementation state — the background solver robustness campaign

**Campaign:** [`README.md`](README.md) · **Audit:** [`AUDIT.md`](AUDIT.md) ·
**Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Logs:** [`logs/`](logs/) ·
**Orchestrator:** [`orchestrator/`](orchestrator/)
**Planned:** 2026-09-16 at `f023eb8` (`tolerance-convergence`)
**Baseline commit:** `f023eb8` — suites green and re-run at planning time:
`CosmologyModels` **30**, `ComputeTargets` **447**
**Target branch:** `background-solver-robustness`, cut from `f023eb8` (README §4)
**Last updated:** 2026-09-16 · **Status: workstream A in progress — prompt 01 complete.**

> **The impact is zero change to any computed quantity, and that is the point.** `AUDIT.md` §5 and
> README §0.2 are the campaign's framing: the two redshifts `_find_rho_equality` produces are
> printed with `:.4g` and discarded, they are already right to the double-precision floor, and no
> prompt in workstreams A–C moves a stored number or implies a regeneration. What the campaign buys
> is a solve that is correct **by construction** instead of because its caller hands it the answer,
> a reachable convergence guard, a comprehensible failure message, and a provenance entry
> `docs/TOLERANCE-PROVENANCE.md` can state.
>
> **The reconciliation changed one thing about the audit's scope.** `AUDIT.md` §2.1's *"the blast
> radius of this solve is two banner lines"* is true of the **solve** and false of the
> **quantity**: `main.py:549-551` recomputes both equality redshifts from the same closed forms and
> forces them into the production source grid, whose content digest is a `BackgroundModel`
> lookup-key column. The tidy that audit §6 observation 2 gestures at would therefore invalidate
> every stored object of eight types. That is **prompt 03** and README §7 **D2**, and it is the
> single most important thing this campaign writes down.
>
> **Five issues are adopted from `docs/OPEN_ISSUES.md` §1.7**, each of which names as its next step
> "whichever prompt next has these files in scope". This is the first campaign that does.
> (`RECONCILIATION.md` §9.2.)

---

## 1. The board

| # | Prompt | Workstream | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Equality-solve characterisation test](01-equality-solve-characterisation.md) | A | README §2 (a), (d) | Opus | ⚠️ | *"Characterise the equality solve before bracketing it"* (SHA not embedded, per the campaign convention) | [`logs/01-equality-solve-characterisation.md`](logs/01-equality-solve-characterisation.md) |
| 02 | [Bracket the equality solve](02-bracket-the-equality-solve.md) | A | README §2 (b), (c), (e); audit §4.2 | Opus | ⬜ | | |
| 03 | [What the equality redshifts feed](03-equality-redshift-consumers.md) | B | README §2 (f); §7 D2 | Opus | ⬜ | | |
| 04 | [Relocate the crossing probe](04-relocate-the-crossing-probe.md) | B | README §2 (g) | Sonnet | ⬜ | | |
| 05 | [Hoist the range logic](05-hoist-the-range-logic.md) | C | README §2 (h); §7 D4 | Opus | ⬜ | | |
| 06 | [Provenance and close-out](06-provenance-and-close-out.md) | C | README §2 (i); §0.4 | Opus | ⬜ | | |
| 07 | [Report the representation in the inventory](07-inventory-representation.md) | **D — gated** | — | Sonnet | 🔒 | | |
| 08 | [Refresh the agreement threshold](08-refresh-agreement-threshold.md) | **D — gated** | — | Sonnet | 🔒 | | |

Status key: ⬜ not started · 🔄 in flight · ✅ complete · ⚠️ complete with a recorded caveat ·
❌ blocked · 🔒 gated on README §7 D3.

**Ordering.** A (01 → 02) → B (03 → 04) → C (05 → 06). 03 depends on 02's corrected solve; 04 and
05 touch the same file as 02 and are sequenced behind it for that reason alone; 06 depends on
everything. 07 and 08 are independent of the chain and may run at any point after the gate opens,
except that 08 needs 01's log.

---

## 2. Item-level state

One row per thing the campaign claims to establish. A row whose evidence is one model or one pair
is **not** ✅ — both equality pairs on both models is the standard, because the two roots are four
decades apart and nothing about one of them predicts the other.

| Item | Kind | Statement | Prompt | Status |
|---|---|---|---|---|
| (a) | **MACHINERY** | The two equality redshifts, on `QCD_Cosmology` and the pure-radiation stand-in, both pairs, scored against an independent `brentq` at Brent's own $4\varepsilon$ floor — ~~to ≤ 2 ulp~~ **to ≤ 4 ulp**, in a test that needs no Ray and no datastore | 01 | ⚠️ |
| (b) | **FIX** | `_find_rho_equality` is bracketed and Brent, at `xtol=1e-300, rtol=1e-14`, with the comment that chose the values at the point of use and to the standard of `:569-583` | 02 | ⬜ |
| (c) | **FIX** | A failure to bracket raises `_find_rho_equality`'s own `RuntimeError` naming the species pair and the range searched — not a `ValueError` and not a `TemperatureRepresentation` bounds error from two frames down | 02 | ⬜ |
| (d) | **MEASUREMENT** | The monotonicity the bracket rests on is a standing test, not a paragraph in an audit: $\rho_m/\rho_r$ strictly decreasing on $z\in[33,3.4\times10^5]$, $\rho_m/\rho_\Lambda$ strictly increasing on $z\in[0,10]$ | 01 | ✅ |
| (e) | **DISCIPLINE** | Every behaviour-change assertion is shown **failing on `HEAD~1`**, with the output quoted in the log | 02 | ⬜ |
| (f) | **MEASUREMENT** | What the equality redshifts actually feed, written down: three closed-form sites scored against each other and against the corrected solve on three models, the chain from `feature_z` to the `BackgroundModel` lookup key, and `main.py:526`'s stale 4e-13 corrected | 03 | ⬜ |
| (g) | **HYGIENE** | `_temperature_crossing_log1pz` is test machinery in the test tree, with its docstring and its `xtol=1e-15, rtol=1e-15` carried across unchanged | 04 | ⬜ |
| (h) | **MEASUREMENT** | The four loop-invariant `_outward` bounds hoisted in all three classes, **bit-identical** returns and character-identical messages demonstrated, and the `T_photon` cost row closed at ≤ 2.5 µs or escalated to the user | 05 | ⬜ |
| (i) | **PROVENANCE** | An entry for **all three** `root_scalar` sites in `LambdaCDM_GenericEOS.py` — value, method, what it sets, call count, choosing measurement **and its commit**, competing floor, cost, citation — in the shape `docs/TOLERANCE-PROVENANCE.md` will want | 06 | ⬜ |

---

## 3. Active and unresolved issues

Issues opened here must be added to [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) **in the
same commit** (`CLAUDE.md`), with the count corrected.

Opened by the **2026-09-16 planning commit**:

- **[00-equality-solve-is-unbracketed-and-loose]** *(planning; assigned to prompt 02)* —
  `LambdaCDM_GenericEOS._find_rho_equality:1013` runs `root_scalar(match_rho, x0=init_z,
  xtol=1e-6, rtol=1e-4)`: an unbracketed secant at tolerances two orders looser than the file's
  other two solves, with no provenance in any campaign record. **Measured** (`AUDIT.md` §2.2,
  reproduced at `f023eb8`): it returns the right answer to **−4.00e-16** relative in 1 to 3
  evaluations, because the analytic guess its caller supplies **is** the root to rounding wherever
  $g_*$ is flat — which is a property of this equation of state at this redshift, not of the code.
  **Impact:** none today; both results are printed with `:.4g` and discarded. The defect is that
  the solve is correct by accident, that its failure mode escapes its own guard — at a guess
  displaced −30 % it raises `ValueError`, at −50 % a `TemperatureRepresentation` bounds error from
  negative $z$, and **neither is `converged=False`**, so the guard at `:1015` is dead on every path
  that fails — and that `xtol` is an absolute tolerance in $z$ serving two roots four decades apart.
  **Next step:** prompt 02. Both roots are trivially bracketable (`AUDIT.md` §4.1).

- **[00-equality-redshift-closed-form-is-duplicated-three-times]** *(planning; measured by prompt
  03; the decision is README §7 D2 and the user's)* — $1+z_{\rm eq}=\Omega_m/\Omega_r$ and
  $1+z_\Lambda=(\Omega_\Lambda/\Omega_m)^{1/3}$ are computed at three sites in three packages:
  `LambdaCDM_GenericEOS.py:501-506` (as a solver guess, feeding two prints),
  `CosmologyModels/LambdaCDM/LambdaCDM.py:73-74` (two prints), and `main.py:549-551`. **The third
  is load-bearing:** it becomes `feature_z`, is forced into the production source grid by
  `CosmologyConcepts/wavenumber.py:350`, and the grid's content digest is a `BackgroundModel`
  lookup-key column (`Datastore/SQL/ObjectFactories/BackgroundModel.py:182`, `:225`, `:251`). So on
  `QCD_Cosmology` the two equality redshifts are **production sample locations inside a datastore
  identity**, reached by a duplicated closed form rather than by `_find_rho_equality` — which is
  why `AUDIT.md` §2.1's grep found nothing and concluded the blast radius was two banner lines.
  **Impact:** unifying the three, which is the obvious tidy and what `AUDIT.md` §6 observation 2
  gestures at, would move a grid sample if it moved either value by one ulp, and invalidate every
  stored object of the eight types `qcd-background-audit` log 11 §5 prices. `main.py:522-528`
  records the duplication as a deliberate, scoped choice by that campaign's prompt 11.
  **Next step:** prompt 03 measures the three sites against each other and prices the three
  options; README §7 **D2** is the user's decision. **Nothing may unify them on an agent's
  judgement.**

- **[00-main-py-equality-agreement-figure-is-stale]** *(planning; assigned to prompt 03)* —
  `main.py:525-527` claims the closed form agrees with the model's root solve *"to 4e-13 relative
  in z on `QCD_Cosmology` at production parameters"*. Measured at `f023eb8`
  (`RECONCILIATION.md` §6): **−9.34e-16** (matter = radiation) and **−3.66e-16** (matter = $\Lambda$)
  against an independent `brentq` at `xtol=1e-300, rtol=8.9e-16` — three orders tighter.
  **Impact:** none. It is a safe over-estimate used only to argue "far below a grid interval". It is
  an unverified figure in a docstring justifying a production grid choice. **Next step:** prompt 03
  re-takes it against the corrected solve and corrects the sentence.

Also opened by the planning commit, found while reconciling (`RECONCILIATION.md` §9.3) and
**measured by prompt 01**, which has the file open for another reason:

- **[01-agreement-threshold-comment-predates-the-representation]** *(planning; **measured by prompt
  01** 2026-09-16; assigned to prompt 08, gated)* — `CosmologyModels/tests/test_wPerturbations.py:34-41`
  describes the $T(z)$ inversion as a "500-point spline" and quotes ~1.3e-9 and ~4e-7 to justify
  `AGREEMENT_RTOL = 1.0e-8`. Since `qcd-background-audit` prompts 05 and 06 the representation is a
  segmented entropy factor at **3,000 nodes of order 5** and what is tabulated is not $T$.
  **Measured at `3e820eb`** (log 01 §4.3), on exactly what `test_agrees_with_LambdaCDM` compares —
  the `PureRadiationEOS` stand-in against `LambdaCDM.wPerturbations` over that test's own probe set
  $z\in\{0,0.5,1,2,10,10^3\}$: the worst relative disagreement is **8.8818e-16** at
  `max_z = 1e4` (claimed ~1.3e-9) and **6.6613e-16** at `max_z = 1e20` (claimed ~4e-7) — **seven to
  nine orders tighter, and the `max_z` dependence has gone entirely**. On a constant-$g_*$ equation
  of state the tabulated entropy factor is exactly constant, so $T(z) = T_{\rm CMB}(1+z)$ is
  recovered to rounding at any `max_z`; the old figures describe a representation that no longer
  exists. **Impact:** none — every assertion passes and `AGREEMENT_RTOL` is now a ceiling eight
  orders above what the code delivers. Same class as
  `[10-transfer-remedial-tolerance-comments-stale]`. **Next step:** prompt 08 rewrites the comment
  and may retighten the constant, if README §7 D3 opens workstream D. Prompt 01 was forbidden to
  edit that file and did not.

### 3.1 Adopted from other boards

Each is that board's issue; each names "whichever prompt next has these files in scope" as its next
step, and this is that campaign (`RECONCILIATION.md` §9.2). Each carries an `**Assigned
(2026-09-16):**` line on its owning board, and each closes on **that** board's §4 with its row
deleted from `docs/OPEN_ISSUES.md`.

| Issue | Owning board | Assigned to | Why here |
|---|---|---|---|
| `[08-temperature-crossing-solver-is-test-only]` | qcd-background-audit | prompt **04** | Needs `LambdaCDM_GenericEOS.py` and `CosmologyModels/tests/` in scope together. Prompt 04 has both, and the method is one of the three `root_scalar` sites prompt 06 must write provenance for |
| `[07-t-photon-range-logic-recomputes-its-bounds]` | qcd-background-audit | prompt **05** | Out of scope for that campaign's prompt 06, whose prompt did not cover prompt 05's range logic. Numerically null; 0.056 µs × 2 of a ~2.5 µs call |
| `[06-t-photon-call-cost-needs-a-quiet-machine]` | qcd-background-audit | prompt **05** | Its own next step **is** the hoist above. Confirmed miss at 2.596 µs against a 2.5 µs target; predicted landing ~2.49 µs. If it does not clear, README §7 **D4** puts the row to the user |
| `[09-audit-script-section-5-prose-counts-the-wrong-set]` | qcd-background-audit | prompt **05** | Needs `docs/qcd-background-audit/` in scope; prompt 05 re-runs `measure_T_z_representation.py` for its §6 cost row and so has it |
| `[03-qcd-inventory-does-not-report-the-representation]` | qcd-background-audit | prompt **07** (gated) | Orphaned by two prompts of that campaign on scope. Not this campaign's subject either, which is why it is behind README §7 **D3** |

### 3.2 Recorded, **not owned here**, and not scheduled (README §0.5)

- `[11-stop-point-root-tolerance]` — `find_phase_extremum`,
  `LiouvilleGreen/integration_tools.py:92`, confirmed still `root_scalar(xtol=1e-6, rtol=1e-4)` at
  `f023eb8`. It **looks identical to this campaign's subject and is not the same problem**: it is
  bracketed, it is on the production path, and it sets a computed quantity — the numeric→WKB
  hand-over — with a datastore regeneration attached. Owned by the hand-over campaign
  (`docs/OPEN_ISSUES.md` §1.1). `AUDIT.md` §4.3 forbids absorbing it. **No prompt here may touch
  it.**
- `ComputeTargets/QuadSourceIntegral.py:1550`'s stale `DEFAULT_QUADRATURE_ATOL = 1e-25` comment
  (the constant has been `1e-32` since `source-remediation` prompt 12). Already on the
  `tolerance-convergence` board; that campaign's README §0.4 puts the file out of bounds and so
  does this one. Repeated here only so it is not re-discovered as new.
- `_solve_T_z:583` and `_temperature_crossing_log1pz:869`'s **tolerances**. Audited, bracketed, at
  the representable floor, commented. Prompt 04 *moves* the second without changing a character of
  its solve; nothing else in this campaign touches either.

---

## 4. Resolved issues

None yet.

---

## 5. Close-out

Written by prompt 06. Empty until then.

---

## 6. Standing notes

1. **A test that passes both before and after proves nothing** (README §2 (e)). This campaign's
   other stop condition is "nothing moved", which is also what a test that tests nothing reports.
   Where a prompt says "show it fails on `HEAD~1`", the orchestrator **runs that check itself**.
2. **`T_Z_REPRESENTATION_VERSION` is 6 at every commit of this campaign.** A prompt that changes it
   has left its scope.
3. **`ComputeTargets` is 447 at every commit.** Nothing in that package reads `_find_rho_equality`;
   a move there means something else changed.
4. **A figure without the tree it was taken on is not a measurement** (README §5 rule 9).
5. **The two printed banner lines at `LambdaCDM_GenericEOS.py:516-517` are character-identical
   throughout.** They are the user-visible surface of the whole campaign, and the campaign's claim
   is that they do not change.
6. **An agent must never assume `HEAD` is its own.** Planning and orchestration commits land on the
   same branch.
7. **"≤ 2 ulp" in README §6 and prompt 01 §5 is arithmetically incompatible with the audit's own
   figures, and the shipped threshold is 4 ulp** (log 01, deviation D1). `AUDIT.md` §2.2's
   **−4.00e-16** at $z = 3406.67$ *is* −3.0 ulp — one ulp there is $1.335\times10^{-16}$ relative —
   and `RECONCILIATION.md` §6's −9.34e-16 for the closed form is 7 ulp. Measured separations from
   the bracketed reference at `3e820eb`: solve **3 / 2 / 1 / 2 ulp**, $\Lambda$ closed form
   **2 / 2**, matter–radiation closed form **7** (QCD) and **1** (stand-in). The test module ships
   `SOLVE_VS_REFERENCE_ULP = 4`, `LAMBDA_CLOSED_FORM_ULP = 2` and
   `MATTER_RADIATION_CLOSED_FORM_ULP = 8`; the fourth ulp is the **reference's own** bracket
   sensitivity, measured, not slack for the solve. **A later prompt converting a relative figure in
   `AUDIT.md` or `RECONCILIATION.md` into ulp must do the arithmetic rather than copy the "2".**
