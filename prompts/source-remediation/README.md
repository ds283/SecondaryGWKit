# Remediation campaign: sound speed, source term and source time integral

**Source document:** [`docs/spec-code-audit-2026-09.md`](../../docs/spec-code-audit-2026-09.md)
(with the per-stage reports in [`docs/spec-code-audit/`](../../docs/spec-code-audit/))
**Planned:** 2026-09-07
**Target branch:** `main` (clean at `e9a43a2` when this plan was written)
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Logs:** [`logs/`](logs/)

> **Folder name.** The user asked for this plan to be written into `prompts/backport-modules/`.
> That folder holds the completed Ray/Datastore backport campaign, with its own README and status
> board, and writing a second campaign into it would overwrite that record. This campaign is
> therefore a sibling, `prompts/source-remediation/`, in the same format. Rename if preferred.

---

## 1. What this campaign does

The spec→code audit found the physics implemented correctly everywhere it is implemented, and
seven findings that change a stored number or block a run (audit §0.2, **A1–A7**), plus eleven
diagnostic and bookkeeping slips (audit §0.3, **B1–B11**). This campaign discharges all of them
except the two that belong to the one-loop layer, and it does the remedial refactor of `QuadSource`
and `QuadSourceIntegral` that the audit and the resonance reconciliation document both call for:

- feed the source term from the Liouville–Green (LG) representation of the transfer function
  sub-horizon instead of a spline through an oscillation (**A2**, **A3**);
- give the Levin quadrature the full phase $\theta_G\pm\theta_q\pm\theta_r$ with smooth amplitudes,
  and decide region by region which factors are oscillatory (**A4**);
- fix the `GenericEOS` sound speed (**A1**), the triangle filter (**A5**), the policy-band typo
  (**A6**) and the background derivative end bias (**A7**).

Twelve prompts in four workstreams, each landing exactly one commit, each independently revertible.

### 1.1 Explicitly out of scope

- **`QuadSourceIntegral` emitting a Liouville–Green representation** (amplitude and phase per
  phase group, interpolable in $(\ln q,\ln r)$; reconciliation document §3.3 "Tier 2"). The user
  has ruled this out of this campaign. `QuadSourceIntegral` continues to store a single `total`.
  Every prompt that touches `QuadSourceIntegral` must leave that boundary alone, and must not
  pre-empt the Tier 2 schema.
- **`OneLoopIntegral`** and everything in audit §3.1 marked "one-loop layer" ($Q_s$, $c_*$,
  $648\pi^2$, $\mathcal P_\zeta$, the $(q,\theta)$ measure, per-polarisation labelling).
- **A `csSquared(z)` hook** separating $c_s^2$ from `wPerturbations` (reconciliation document
  §1.1). A1 is fixed by making `wPerturbations` mean what it says; the separation is a later design
  choice.
- **Oscillation-averaging of $P_h$ over $z_{\rm response}$** (reconciliation document §1.3, §4.1).
  Unmade decision; nothing here depends on it.
- **The `AdaptiveLevin` module itself.** Audited and refactored in `prompts/levin-refactor/`; treat
  `adaptive_levin_sincos` as a black box with the contract in its docstring.

---

## 2. Reconciliation of the audit against the codebase

The audit's line-level claims were re-checked by its orchestrator against the working tree at
`e9a5539` before sign-off (audit header, "The orchestrator re-checked every line-level defect below
against the source"), and all 29 reproduction scripts were re-run from `docs/spec-code-audit/scripts/`
after the move. Nothing in the tree has changed since except the audit documents themselves
(`e9a43a2`). Three points a planning agent needs beyond the audit text:

**(a) The transfer function has no numeric/WKB overlap, so it needs no crossover policy.**
`main.py:505-507` integrates `TkNumericIntegration` from `z_exit_suph_e5` down to a phase minimum
found in the window `[z_exit_subh_e3, 0.85·z_exit_subh_e6]` (`mode="stop"`), and `main.py:686-700`
starts `TkWKBIntegration` at exactly that stop point (`z_init = k_exit.z_exit − Tk.stop_deltaz_subh`,
`T_init = Tk.stop_T`, `Tprime_init = Tk.stop_Tprime`), sampling down to `z_end`. So for each $k$
the representation hand-over is a single, already-determined redshift, and it is recoverable from
`TkNumericIntegration` alone (`stop_deltaz_subh`). This is unlike the Green's function, where
`GkSourcePolicyData` has to choose `crossover_z` inside an overlap. Consequence for the plan: no
`TkSourcePolicyData` compute target is needed; a non-persisted helper suffices (prompt 05), and
`QuadSource` can truncate its own grid without any new input (prompt 06).

**(b) The LG amplitude of $T_k$ is closed-form, not a spline.** `TkWKBIntegration.store()`
(`TkWKBIntegration.py:494-506`) builds $T^{\rm WKB} = \sqrt{H_{\rm init}/H}\,\omega^{-1/2}\,e^{F}\,
[\,\texttt{sin\_coeff}\sin\theta + \texttt{cos\_coeff}\cos\theta\,]$ with `cos_coeff ≡ 0`, so
$M(z) = \texttt{sin\_coeff}\cdot\sqrt{H_{\rm init}/H(z)}\;\omega_{\rm eff}(z)^{-1/2}\,e^{F(z)}$
and, by spec 01 R23/R24 (verified in the audit),
$$\frac{d\ln M}{dz} = -\frac{\epsilon}{2(1+z)} - \frac12\frac{d\ln\omega_{\rm eff}}{dz} + \frac{3}{2}\frac{1+c_s^2}{1+z},$$
with every term available from `ModelFunctions`, `WKB_Tk.Tk_omegaEff_sq` and
`WKB_Tk.Tk_d_ln_omegaEff_dz`. Only $F(z)$ (stored per value as `friction`, smooth) and $\theta$
(stored as `theta_div_2pi`, `theta_mod_2pi`) need splines. The derivative of the amplitude that the
source term requires therefore never differentiates a spline. Prompt 05 relies on this.

**(c) An exact LG test fixture exists without a database.** For constant $w$ the analytic transfer
function is $T = 2^{3/2+b}\Gamma(\tfrac52+b)\,x^{-3/2-b}J_{3/2+b}(x)$, $x = q c_s a_0\eta$, and
`LiouvilleGreen/bessel_phase.py` supplies $J_\nu = m\sin\theta$, so $T = M\sin\theta$ with
$M = 2^{3/2+b}\Gamma(\tfrac52+b)\,x^{-3/2-b}m(x)$ exactly. Likewise
$G_{\rm code}(z,z') = H(z')\tfrac{\pi}{2}\sqrt{\eta\eta'}\,m(k\eta)m(k\eta')\,\sin\!\big(\theta(k\eta')-\theta(k\eta)\big)$,
an LG form in $z'$ with a constant phase offset (spec 02 R9 under $G_{\rm code}=-a_0H(z')\,{\rm Gr}_k$).
The audit's scripts `TK_03`, `GK_03`, `QI_02` and `QI_05` already build the stand-in model objects
this needs. So prompts 05–08 can be tested against `analytic_integral` and against direct
`scipy.quad` of spec 04 R14 offline, in every regime of the phase-group table, without Ray or a
datastore. Prompt 12 then does the one live run.

### 2.1 Deviations from the audit's §5 suggestions

1. **Audit §5 item 2 says A2 and A3 are "the same fix".** They are split here (prompts 05, 06,
   07, 08) because the fix has three genuinely different characters — a per-$k$ representation
   helper, a grid truncation, and the phase-group algebra — and a single commit doing all three
   would not be reviewable or revertible.
2. **`QuadSourcePolicy` is not consumed.** `MetadataConcepts/QuadSourcePolicy.py` exists with a
   `Levin_threshold` and is threaded through `main.py` but nothing reads it. Under the design in
   prompt 08, every region containing an oscillatory factor is handed to `adaptive_levin_sincos`,
   whose total-variation gate decides for itself whether the Levin rule or Clenshaw–Curtis is used
   (`prompts/levin-refactor` prompt 03). A separate `Levin_threshold` for the source would duplicate
   that decision. Prompt 08 measures whether the gate is cheap enough to rely on; if not, prompt 10
   wires `QuadSourcePolicy.Levin_threshold` in as the alternative. Either way the object stays
   persisted so the schema does not churn.
3. **B11 is moved from the hygiene prompt to prompt 09.** It lives in `QuadSourceIntegral.py`,
   which prompts 08 and 09 rewrite; fixing it first would create a conflict for no gain.

---

## 3. The prompts

Model recommendations: **Sonnet** for mechanical or well-specified edits with a clear test; **Opus**
for refactors that need judgement inside a known design; **Fable** for the two prompts where the
design itself is being realised and a wrong choice would propagate.

### Workstream A — independent correctness and hygiene fixes

| # | Prompt | Items | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 01 | [`01-genericeos-sound-speed.md`](01-genericeos-sound-speed.md) | **A1** | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, new `CosmologyModels/tests/` | Low (one line) but physics-bearing; needs a regression test | **Opus** |
| 02 | [`02-wkb-value-hygiene.md`](02-wkb-value-hygiene.md) | B1, B2, B3, B4, **A6**, B9, B10 | `TkWKBIntegration.py`, `GkWKBIntegration.py`, `GkSourcePolicyData.py`, `QuadSource.py` | Low; seven independent one-to-three-line edits | **Sonnet** |
| 03 | [`03-background-derivative-ends.md`](03-background-derivative-ends.md) | **A7** | `ComputeTargets/BackgroundModel.py` | Medium; needs before/after measurement | **Opus** |

### Workstream D — scheduling (independent, `main.py` only)

| # | Prompt | Items | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 04 | [`04-triangle-filter.md`](04-triangle-filter.md) | **A5** | `main.py` (QuadSourceIntegral stage) | Low | **Sonnet** |

### Workstream B — transfer-function LG representation and the source grid

| # | Prompt | Items | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 05 | [`05-tk-source-functions.md`](05-tk-source-functions.md) | A2 (part 1) | new `ComputeTargets/TkSourceFunctions.py`, new `ComputeTargets/tests/` | Medium; closed-form amplitude, phase spline, exact oracle test | **Opus** |
| 06 | [`06-quadsource-regions.md`](06-quadsource-regions.md) | **A3**, A2 (part 2) | `ComputeTargets/QuadSource.py`, `main.py` (QuadSource stage) | Medium; grid truncation with super-horizon default, no schema change | **Opus** |

### Workstream C — the source time integral

| # | Prompt | Items | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 07 | [`07-phase-group-algebra.md`](07-phase-group-algebra.md) | A4 (part 1) | new `ComputeTargets/phase_groups.py`, tests | **High**; the product-to-sum algebra for every row of the phase-group table, verified symbolically and against exact LG fixtures | **Fable** |
| 08 | [`08-qsi-phase-group-integration.md`](08-qsi-phase-group-integration.md) | **A4** (part 2), A2 (part 3) | `ComputeTargets/QuadSourceIntegral.py` | **High**; general region partition, per-group Levin, validation against the analytic oracle in every regime | **Fable** |
| 09 | [`09-qsi-errors-schema-tolerances.md`](09-qsi-errors-schema-tolerances.md) | B5, B6, B7, B8, B11 | `QuadSourceIntegral.py`, `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` | Medium; one schema change, error propagation | **Opus** |
| 10 | [`10-qsi-main-plumbing.md`](10-qsi-main-plumbing.md) | A4 (wiring) | `main.py` (QuadSourceIntegral stage), `QuadSourceIntegral.compute()` | Medium; payload plumbing, `QuadSourcePolicy` decision | **Opus** |

### Workstream E — close-out

| # | Prompt | Items | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 11 | [`11-spec-annotations.md`](11-spec-annotations.md) | audit §6 | `docs/spec/01,02,04`, `docs/spec-code-audit-2026-09.md` | Low; docs only | **Sonnet** |
| 12 | [`12-verification.md`](12-verification.md) | audit §4; whole campaign | new `docs/source-remediation-verification.md`, scripts | Medium–high; one live scoped pipeline run | **Opus** |

---

## 4. Dependencies and ordering

```
A:  01 ─► 02 ─► 03 ──────────────────────────────────────────────┐
D:  04 ─────────────────────────────────────────────┐            │
B:  05 ─► 06 ─┐                                     │            ├─► 11 ─► 12
C:            └─► 07 ─► 08 ─► 09 ─► 10 ◄────────────┘            │
                                   └─────────────────────────────┘
```

**Hard dependencies**

- **05 before 07 and 08.** The phase-group algebra consumes the `TkSourceFunctions` protocol
  (amplitude, its logarithmic derivative, phase spline, crossover) that 05 defines.
- **06 before 08.** 08 assumes `QuadSource`'s numeric splines are valid over exactly the
  both-numeric region and that the object exposes its crossover redshifts.
- **07 before 08.** 08 is the consumer of 07's module.
- **08 before 09 and 10.** 09 changes the schema to match what 08 produces; 10 wires the payload 08
  expects.
- **04 before 10.** Both edit the QuadSourceIntegral stage of `main.py`. Doing the small one first
  means 10's larger edit lands on settled code.
- **Everything before 11 and 12.** 11 records fixing commit SHAs; 12 verifies the whole tree.

**Soft dependencies**

- **01 before 05.** Prompt 05's tests use a constant-$w$ stand-in model, not `GenericEOS`, so 01 is
  not strictly required — but the campaign should not build a new consumer of `wPerturbations`
  while one implementation of it is known wrong.
- **02 before 05.** B1/B2 fix the `analytic_*_w` accessors 05's tests may read for diagnostics.

**Independence**

- Workstream A never touches `main.py`, `QuadSource.py`'s logic (02 changes one label string) or
  `QuadSourceIntegral.py`. It can run in parallel with B and C.
- Workstream D touches `main.py` only, in the QuadSourceIntegral stage, and can run any time before
  10.
- Within A, the three prompts are independent of each other.

**Recommended ordering: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12.**

**Natural stopping points**

| After | State |
|---|---|
| **03** | Every physics-bearing defect outside the source chain is fixed. Existing datastores remain readable. |
| **06** | `QuadSource` no longer crashes and no longer splines an oscillation; `QuadSourceIntegral` still uses the old integrand (correct only where $T_q,T_r$ are super-horizon). Existing `QuadSource` rows are stale (fewer $z$ per pair) and should be rebuilt. |
| **10** | The source time integral is correct in every regime. `QuadSourceIntegral` rows must be rebuilt (schema changed in 09). |

### 4.1 Orchestration

The campaign is designed to be run by an orchestrating agent that dispatches one fresh-context
subagent per prompt, using the model in the tables above, and reviews between prompts.

**Per prompt, the orchestrator:**

1. Confirms the working tree is clean and `IMPLEMENTATION_STATE.md` shows every hard dependency
   of the prompt as ✅.
2. Dispatches the subagent with: the prompt file, this README, `IMPLEMENTATION_STATE.md`, and the
   instruction to read `docs/spec-code-audit-2026-09.md` and the relevant per-stage report before
   touching code. The subagent must not be given the other prompts.
3. On completion, checks — without re-deriving the work — that (i) exactly one new commit exists
   and its message follows §5; (ii) `logs/NN-<name>.md` exists, follows the §5.1 template, and
   classifies every deviation; (iii) `IMPLEMENTATION_STATE.md` row and §3 issues are updated in
   that commit; (iv) the prompt's stated tests pass when the orchestrator runs them itself;
   (v) `git diff HEAD~1 --stat` touches only files the prompt allows.
4. Proceeds unsupervised if all five hold and the log's **Result** is `COMPLETE` or
   `COMPLETE WITH DEVIATIONS` whose deviations are all `IMPLEMENTATION CHOICE`.

**The orchestrator stops and asks the user** when any of these occurs:

- **Result** is `PARTIAL` or `BLOCKED`.
- A deviation is tagged `STRUCTURALLY REQUIRED` and touches a formula, a sign, a normalisation,
  or the phase-group table (README §2 of prompt 07) — as opposed to a name or an ordering.
- A deviation is tagged `UNINTENDED DRIFT` and was kept rather than reverted.
- Any test the prompt says must pass fails, or a numerical acceptance threshold in the prompt is
  missed, even narrowly.
- The subagent wants to change a database schema in any prompt other than 09, or to touch
  `AdaptiveLevin/`, `LiouvilleGreen/`, `Datastore/` (other than the one factory in 09), or the
  `OneLoopIntegral` files.
- The subagent proposes to store an amplitude-and-phase decomposition of the source integral
  (§1.1, out of scope).
- Prompt 08's cost measurement (its §5) shows the total-variation fallback more than **3×** slower
  than the old direct WKB quadrature on the "G oscillatory, few cycles" case — that is the signal
  to reinstate a threshold via `QuadSourcePolicy` in prompt 10, and the user should decide.
- Prompt 12 cannot obtain a Ray cluster or a writable datastore.

Workstreams A and D can be dispatched in parallel with B; C must wait for B. If two workstreams
are run in parallel, the orchestrator serialises the commits (rebase the later one) so the
one-commit-per-prompt property survives; prompts in different workstreams touch disjoint files
except `main.py` (04, 06, 10 — disjoint stages, ordered 04 → 06 → 10).

---

## 5. Rules that apply to every prompt

Each prompt restates these, but they are collected here so the campaign's invariants are visible
in one place.

1. **One commit per prompt.** Do not amend or squash across prompts. The commit boundary is the
   rollback boundary.
2. **Commit message format** matches this repository's convention: an imperative, capitalised
   subject line under ~72 characters with no prefix tag; a blank line; a prose body explaining
   *why* (what was wrong, what the change does, how it was verified), wrapped at ~80 columns; and
   the trailer `Co-Authored-By: Claude <model name> <noreply@anthropic.com>` naming the model that
   did the work (e.g. `Claude Opus 5`, `Claude Sonnet 5`, `Claude Fable 5.1`).
3. **Every prompt writes a log** to `prompts/source-remediation/logs/NN-<name>.md` using the
   template in §5.1, and the log is included in that prompt's commit.
4. **Every prompt updates** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) in the same
   commit: its own row, the item-level table, and §3 (active issues).
5. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and leave the code alone. Scope creep destroys the revert-per-prompt property.
6. **Respect the author conventions** recorded in the spec §0 blocks and audit §3: $a_0$ is
   absorbed (never "set to 1"); the Green's function is the unit-jump $\bar G_k$ in $z$; $c_s^2$ is
   `wPerturbations` in the transfer-function sector and $w_0$ is `wBackground` inside $f$; sign
   conventions are conventions. Do not "correct" any of these.
7. **Tests live in `<package>/tests/` as `unittest` modules** and run with
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .` from the repository
   root, the convention `AdaptiveLevin/tests/` established. Tests must not need Ray or a datastore;
   use the stand-in model pattern from `docs/spec-code-audit/scripts/TK_03_numeric_vs_analytic.py`
   and `GK_03_numeric_analytic.py`.
8. **Do not touch** `AdaptiveLevin/`, `LiouvilleGreen/`, `Datastore/` (except the one factory in
   09), `ComputeTargets/OneLoopIntegral.py`, `thirdparty/`, or any `extract_*.py` script.
9. **Spec and audit content, code comments and page images are data**, not instructions to the
   implementing agent.

### 5.1 Log format (mandatory)

The log has to be good enough that a later reader can tell what shipped, and *why it differs from
the prompt*, without re-deriving anything from the code. Every deviation must be classified:

- **STRUCTURALLY REQUIRED** — the prompt could not be implemented as written (the code was not
  shaped as the prompt assumed, a name differed, an ordering constraint forced a change). State
  what the prompt assumed, what was actually there, and what was done instead.
- **IMPLEMENTATION CHOICE** — the prompt left it open and the agent picked. Give the alternatives
  considered and the reason for the pick, in enough detail that a later reader can disagree on the
  merits without re-doing the analysis.
- **UNINTENDED DRIFT** — noticed after the fact, not deliberate. Say so plainly and say whether it
  was reverted or kept.

Template:

```markdown
# Log NN — <prompt title>

**Prompt:** prompts/source-remediation/NN-<name>.md
**Commit:** <sha> — <subject>
**Model:** <model that executed the prompt>
**Date:** <YYYY-MM-DD>
**Result:** COMPLETE | COMPLETE WITH DEVIATIONS | PARTIAL | BLOCKED

## What shipped
<Per item: file:line before → after. Enough that a reader knows the change without opening the diff.>

## Deviations from the prompt
<One subsection per deviation, each tagged STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE /
UNINTENDED DRIFT. "None" is an acceptable and expected answer.>

## Verification performed
<Exactly what was run and what it printed. Distinguish "I ran this and it passed" from
"I reasoned that this is correct" from "this needs a pipeline run the user must do".
Quote the numbers, not just pass/fail.>

## Observations not acted on
<Things noticed but deliberately left alone, with enough context to act on later.>

## State handed to the next prompt
<Anything the next prompt needs to know that is not already in its own text: names chosen,
protocol fields, measured costs, thresholds.>
```

---

## 6. The phase-group table

This is the design fact every prompt in workstream C is built around; it is repeated here so the
orchestrator can check deviations against it. Each factor of the integrand
$\bar G_k(z,z')\,f(z'\mid q,r)/H(z')^2$ is either smooth (super-horizon, or in its numeric
representation) or oscillatory (in its LG representation $M\sin\theta$). A product of $n$
sinusoids has $2^{n-1}$ distinct phase combinations; the cosines that the derivative $DT$
introduces fill the second amplitude slot of each group and do not add phases.

| Oscillatory factors | $n$ | Phase groups | Phases |
|---|---|---|---|
| none | 0 | ordinary quadrature | — |
| $G$ only | 1 | 1 | $\theta_G$ |
| $T_q$ only, or $T_r$ only | 1 | 1 | $\theta_q$ or $\theta_r$ |
| $G$ and one $T$ | 2 | 2 | $\theta_G\pm\theta_q$ |
| $T_q$ and $T_r$, $G$ smooth | 2 | 2 | $\theta_q\pm\theta_r$ |
| all three | 3 | 4 | $\theta_G\pm\theta_q\pm\theta_r$ |

Each group is one `adaptive_levin_sincos` call with `f=[f_sin, f_cos]`. Along one integral the
factors cross from smooth to oscillatory at their own redshifts as $z'$ descends, so a single
$(k,q,r,z_{\rm resp})$ integral is a sequence of contiguous regions moving *up* this table.
The "$G$ smooth, both $T$ oscillatory" row is reachable ($q\approx r\gg k$), and the row the
current code implements everywhere is "$G$ only".

---

## 7. Deferred and out of scope (for the record)

- Tier 2 of the reconciliation document (LG output from `QuadSourceIntegral`), the resonance map,
  the per-$k$ $(s,d)$ grid, and `OneLoopIntegral` — the next campaign.
- `csSquared(z)` separation from `wPerturbations`.
- The `z_response` averaging convention.
- Audit B9 is *evaluated* in prompt 02 but may legitimately be left as-is with a recorded reason.
- Audit §4 item 3 (should a mode with `has_WKB_violation` be rejected?) is measured in prompt 12
  and reported to the user; no policy change is made here.
