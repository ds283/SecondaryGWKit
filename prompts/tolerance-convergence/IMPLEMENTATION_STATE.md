# Implementation state — tolerance and convergence campaign

**Campaign:** [`README.md`](README.md) · **Rebase record:** [`RECONCILIATION.md`](RECONCILIATION.md)
· **Logs:** [`logs/`](logs/) · **Orchestrator:** [`orchestrator/`](orchestrator/)
**Planned:** 2026-09-12 at `622b84b` · **Rebased:** 2026-09-16 at `acd5b8e` ·
**Re-anchored:** 2026-09-16 at `bc6dc97` (`RECONCILIATION.md` §7)
**Baseline commit:** `bc6dc97` (`tolerance-convergence`, 25 commits ahead of `main`, clean; suites
re-run for the re-anchor and green — `ComputeTargets` **452**, `CosmologyModels` **39**)
**Superseded baseline:** `acd5b8e`, `ComputeTargets` 447, `CosmologyModels` 30 — the anchor of any
figure in this campaign's documents dated before 2026-09-16 19:32
**Last updated:** 2026-09-17 · **Status: in progress — 2 / 6, plus the insertion 02a landed.**
**Every user decision needed to start is settled** — D1 and D3 are post-audit gates by design, D2
settled 2026-09-12, D4 settled by README §0.4, **D5 settled yes 2026-09-16**.
**Prompts 01, 02 and 02a have all landed; 03–06 may now be written** (§1 below; the 2026-09-17
decision — 02's table says which targets the audits own, and 02a settles the anchor that **T6**,
prompt 03's own row, turns on). **Prompt 02a has landed** (README §3.2a, authorised by §7 **D6**,
2026-09-17): `main.source_grid_spacing_profile` now guards the stencil evaluation where the
Liouville–Green expansion does not exist, so **the version-2 source grid builds at every production
anchor on every production cosmology**. QCD at its own anchor is **2034 samples / `21ffc126`, 53
guarded nodes**; both published grids are **bit-identical** (1996 / `4849552b`, 1778 / `60a3205a`,
zero guarded), so no figure in the record moves. **The two production anchors are now named
separately** — `wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM` and `PRODUCTION_Z_INIT_QCD` — and prompts
03, 04 and 06 must say which one a figure was taken at, exactly as §2 (b) makes them say which grid
generation. **Prompt 01 has landed**: the convergence facility is
`ComputeTargets/tests/convergence_reference.py` and the source-grid generations are named in
`ComputeTargets/tests/wkb_reference.py`. Every later prompt measures through them.
**Prompt 02 has landed**: `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` is the inventory and
`inventory.py` regenerates its tables. **Prompts 03 and 04 take their target lists from log 02's
"State handed to the next prompt", not from README §2 (a).** README §2 (a)'s eight rows are each
confirmed against the tree; what is *not* confirmed is the count — there is a **ninth** keyed
object type, `OneLoopIntegral`, which prompt 02 §8 makes a stop and which is left for the user
(§3 below). README **§0.1's summary sentence** — "of those six only one actually uses it" — is
also wrong: **two** of the six live `(atol, rtol)` sharers use the value, `GkNumericIntegration`
and `wavenumber_exit_time`. §2 (a)'s table says so on both rows; prompt 06 reconciles the README.

> **The campaign is unblocked.** The 2026-09-12 plan was blocked on `prompts/GkTk-remedial` prompt
> 18, because until it landed the reference on `QCDModel` did not converge at four wavenumbers.
> That campaign closed at 20 / 20 and merged (`e8f746d`); `prompts/qcd-background-audit` then ran
> to 16 / 16 and merged (`acd5b8e`). `[17-qcd-reference-not-converged]` is **closed**, and the
> worst QCD reference-convergence drift is now 7.08e-09 ($T_k$) and 3.67e-09 ($G_k$) against the
> 3.4e-08 criterion, zero offenders at all 50 production wavenumbers.
>
> **It is also rebased, and the rebase changed the subject.** The plan counted five compute targets
> sharing two constants across four object types. Measured at `acd5b8e`, **eight** object types are
> keyed on an accuracy parameter; the shared pair keys **six** of them and `rtol` alone keys a
> seventh; and of the six, **only one actually uses the value it is given**. Two targets the plan
> never mentioned — `wavenumber_exit_time`, a live `root_scalar` that fixes where every grid
> begins, and `BackgroundModel`, whose three Gauss orders are the largest instance of the
> integer-order case — are now in scope. `RECONCILIATION.md` scores every claim of the old plan and
> is the document to read before trusting any figure inherited from it.
>
> **Three further things the plan assumed are no longer true.** (i) The production source grid has
> been rebuilt twice and the test tree holds **three disagreeing reproductions** of it, none of
> which is the one `main.py` builds — so every published figure in README §6 was taken on a
> superseded grid. (ii) The cost that D1 turns on was measured in the wrong sector: one decade of
> `rtol` is free across 50 $T_k$ objects per model and is the entire compute decision across
> ~65,000 $G_k$ objects per model, which has never been swept. (iii) The Gauss orders' evidence
> predates two replacements of the background and the removal of the break-point set it was scored
> against; `decision.recommended_scheme` in the fixture still reads `"branch+knots"` and the knots
> do not exist.
>
> **Re-anchored 2026-09-16 onto `bc6dc97`.** A third campaign,
> `prompts/background-solver-robustness`, was planned and run to 9 / 9 **on this branch** after the
> rebase above was written, and merged at `bc6dc97`. It is not a precondition — it is a campaign
> that landed on one of this campaign's own subjects, the `LambdaCDM_GenericEOS.py` root solves.
> `RECONCILIATION.md` **§7** scores it. It moved no tolerance this campaign sets, and
> `config/defaults.py` remains byte-identical to the file the 2026-09-12 plan was written against.
> Three things it changed for the prompts: the three root solves of README §3.2 arrive **already
> settled**, with provenance prompt 02 lifts rather than derives; there are **four**
> production-grid reproductions in the test tree and one is already version 2, so prompt 01
> **hoists rather than builds** (§3 below); and `cosmology_feature_redshifts`, which prompt 01
> lifts, now asks the cosmology for its equality redshifts and raises with no fallback.
>
> **D5 is settled yes** (README §7), so prompt 04 may re-run `residual_convergence.py`, write
> `ComputeTargets/tests/wkb_reference_data.json` and edit `test_background_tau.py` —
> `[01-convergence-block-has-a-separate-generator]` has, for the first time, a prompt allowed to
> close it.

---

## 1. The board

| # | Prompt | Covers | Model | Written? | Status | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | The convergence harness and one production grid | README §2 (b), (h); `[00-three-production-grid-reproductions]` | Opus | ✍️ [`01-…`](01-convergence-harness-and-grid.md) | ✅ | *"Build the convergence harness and name the source-grid generations"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/01-…`](logs/01-convergence-harness-and-grid.md) |
| 02 | The accuracy-parameter inventory | README §2 (a), (c), (g); `RECONCILIATION.md` §2.1 | Opus | ✍️ [`02-…`](02-accuracy-parameter-inventory.md) | ⚠️ | *"Inventory every accuracy parameter in the pipeline"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/02-…`](logs/02-accuracy-parameter-inventory.md) |
| 02a | Make the grid buildable at every production anchor | README §3.2a, §7 **D6**; `[01-v2-density-raises-at-the-qcd-production-anchor]` | Opus | ✍️ [`02a-…`](02a-source-grid-density-guard.md) | ⚠️ | *"Guard the source grid density criterion off-node"* (SHA not embedded, per the convention `prompts/background-solver-robustness` uses) | [`logs/02a-…`](logs/02a-source-grid-density-guard.md) |
| 03 | Audit the adaptive solvers | README §2 (d), (e), (f); review §10.1, §12.5 | Opus | ⏸️ **held** | ⬜ | | |
| 04 | Audit the order-governed targets | README §2 (a); §7 D5 **(settled yes)** | Opus | ⏸️ **held** | ⬜ | | |
| 05 | Decouple | README §2 (a), (g); §7 D1, D3 | Opus | ⏸️ **held** | ⬜ | | |
| 06 | `QuadSourceIntegral`, close-out, the provenance note | README §0.4, §1.2 | Opus | ⏸️ **held** | ⬜ | | |

Status key: ⬜ not started · 🔄 in flight · ✅ complete · ⚠️ complete with a recorded caveat ·
❌ blocked. Written key: ✍️ prompt file exists · ⏸️ deliberately held.

> **Prompts 03–06 are held back by decision, 2026-09-16 — this is not an unfinished plan.**
> README §3 fixes each prompt's charter and §6 fixes its acceptance, so what is held is the
> **method**, not the commitment; the campaign is fully specified and partly written. The reason is
> §4's own dependency graph: *"02 must precede both audits: it is what says which targets 03 and 04
> each own, and the old plan's allocation was wrong."* Writing 03 and 04 now would allocate their
> targets from the same unverified inventory that prompt 02 exists to replace — the error the
> 2026-09-12 plan made, and the reason new 02 has no predecessor. 05's content **is** D1 and D3,
> which do not exist until 03 and 04 report; 06 assembles from the earlier logs.
>
> **The staging, as revised 2026-09-17:** 01, 02 and **02a** now; **03 and 04 written after 02 and
> 02a have both landed**, against 02's table and 02a's hand-off rather than against a guess at
> either; 05 after the user settles D1 and D3; 06 last. 02a joins the precondition because **T6 is
> prompt 03's own row** and 02a is what settles the anchor it turns on — a prompt 03 drafted from 02
> alone would allocate `wavenumber_exit_time` without the measurement that changes its charter. This
> costs nothing in elapsed time — §4.1 already declares a stop after each of 02, 03 and 04, and 02a
> ends in a hand-back for the same reason, so the user is in the loop anyway. Orchestrator prompts
> are staged with them ([`orchestrator/README.md`](orchestrator/README.md)).

> **02a is an insertion, not a renumbering.** It carries a letter so that the charters of README
> §3.3–§3.6, the acceptance rows of §6.2 and the T-numbers of §2 keep the numbers every other
> document in the tree cites. It is the first production change this campaign makes; §7 D6 records
> the widened §0.5 boundary that permits it, and the carve-out is
> `main.source_grid_spacing_profile` alone. Its acceptance is **bit-identity** with the two
> published grid digests, not improvement — a moved digest is a stop under README §4.3.

**The 2026-09-12 board carried five prompts.** The mapping, so that a reader of the old plan is not
lost: old 01 → new 01 (widened by the grid); old 02 → new 03 (widened by `wavenumber_exit_time` and
by $G_k$ becoming the centre); old 03 → new 04 (widened by `BackgroundModel` and by the stale
fixture); old 04 → new 05; old 05 → new 06. **New 02 has no predecessor** — it is the inventory the
old plan assumed and got wrong.

---

## 2. Item-level state

One row per thing the campaign claims to establish. Filled in as prompts land; a row whose evidence
is a single wavenumber or a single model is **not** ✅, which is the specific failure this campaign
was created by. A row whose evidence does not say which source-grid generation it was taken on is
not ✅ either, which is the specific failure the rebase found.

| Item | Kind | Statement | Prompt | Status |
|---|---|---|---|---|
| T1 | **MACHINERY** | One reusable convergence facility, in the test tree, covering every target and calibrated against the constant-$w$ anchors at every use, with "one step tighter" meaning a decade for a tolerance and one order for a Gauss order | 01 | ✅ `convergence_reference.py`: `TolerancePair` / `GaussOrder`, `reference_drift` → `DriftVerdict` (the criterion evaluated, never just reported), `radiation_anchors` over all eleven closed forms, the two numeric sectors folded in from `tk_numeric_atol_sweep.py` |
| T2 | **MACHINERY** | **One** reproduction of the production source grid at `SOURCE_GRID_CONSTRUCTION_VERSION = 2`, with the version-0 and version-1 constructions retained and named rather than silently re-scored | 01 | ⚠️ `wkb_reference.source_grid(SOURCE_GRID_V0/_V1/_V2, …)`, no default generation, bit-identical to all three constructions it replaces. **Caveat:** `production_source_grid` keeps its historic behaviour because 28 call sites in 20 files outside prompt 01's scope import it; it is documented as version 0 rather than made to fail loudly |
| T3 | **MEASUREMENT** | The accuracy-parameter inventory: every parameter, what it keys, whether it reaches a solver, what the real knob is, and the object count of the sector it keys | 02 | ⚠️ `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` + `inventory.py`: **12 keyed tables** derived from `Datastore/SQL/Datastore.py`'s `_factories` by `ast`, **39 parameter rows** in nine columns, **10 hard-coded literals** in production. README §2 (a)'s eight rows are each **confirmed**. **Caveat:** there is a **ninth** keyed object type, `OneLoopIntegral` — prompt 02 §8's stop condition, left unassigned for the user |
| T4 | **MEASUREMENT** | `GkNumericIntegration` characterised over the production response grid on three models in both `atol` and `rtol`, against the consumer-spline floor — the sector with ~65,000 objects per model, never swept, and where the campaign's compute decision actually lives | 03 | ⬜ |
| T5 | **MEASUREMENT** | `TkNumericIntegration` likewise, re-taken on the version-2 grid and under its own `BREAK_POINT_ALL` policy | 03 | ⬜ |
| T6 | **MEASUREMENT** | `wavenumber_exit_time`'s root solve measured at all — nothing in the record says what `xtol = 1e-10`, `rtol = 1e-8` in $\log(1+z)$ buys or costs. **Scored against the exact $z_{\rm exit}$ on `RadiationModel` first** (README §3.1): $1 + z = k/(H_0 e^{N})$, confirmed at the rebase to 2.3e-16 relative or better | 03 | ⬜ |
| T7 | **MEASUREMENT** | $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$ and `RESIDUAL_WKB_REGION_MARGIN` audited at every production $k$ on the corrected background and the 3-point break set, replacing evidence generated 2026-09-10. **Every one of the four orders has a closed-form anchor on `RadiationModel`** (README §3.1), including $\rho_G \equiv 0$, which makes the $N_\rho$ measurement pure quadrature error with no reference to build | 04 | ⬜ |
| T8 | **DECISION** | The decoupled tolerance pairs settled by the user (§7 D1) and shipped with the measurement that chose each, in `config/defaults.py` | 05 | ⬜ |
| T9 | **DECISION** | What replaces the vestigial `atol`/`rtol` key columns on the three order-governed targets (§7 D3) — the user's stated target for the campaign | 05 | ⬜ |
| T10 | **PLUMBING** | Every `object_get` of a retuned target carries its own parameter, with an `ast` guard whose predicate reaches all eight targets and fails on an unclassified site | 05 | ⬜ |
| T11 | **HAND-OFF** | `QuadSourceIntegral` measured read-only and reported to `levin-refactor` / `qsi-phase-groups` | 06 | ⬜ |
| T12 | **PROVENANCE** | `docs/TOLERANCE-PROVENANCE.md` covers **every** accuracy parameter in the pipeline — including the ones this campaign inherits and does not set, and the ones nobody has ever chosen — with value, choosing measurement and its grid generation, competing floor, cost times object count, and citation (README §1.2) | 06 | ⬜ |
| T13 | **MACHINERY** | The version-2 source grid **builds at every production anchor on every production cosmology**, with a node at which the Liouville–Green expansion does not exist marked unusable and *counted* rather than raising, a refusal above a measured fraction of the band, and QCD's own anchor named in the test tree. Acceptance is bit-identity with the two published digests (1996 / `4849552b`, 1778 / `60a3205a`), not accuracy | 02a | ⚠️ QCD at its own anchor builds: **2034 / `21ffc126`, 53 guarded nodes** (Gk 34, Tk 19, 53 of 100 (k, sector) cases, exactly one node each). Both published grids **bit-identical with zero guarded**, so the `except` branch is never entered on either and the guard is provably inert on every figure in the record. Refusal ceiling `SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05`, **64.8x** the worst measured band (7.716e-04). **Caveat:** the constant's only proper home is `CosmologyConcepts/wavenumber.py`, a file the prompt's own file list forbade; the agent stopped rather than editing it, and the **user amended the scope on 2026-09-17** to permit that one append-only definition (log 02a, deviation 1) |

---

## 3. Active and unresolved issues

Issues opened here must be added to [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) **in the
same commit** (`CLAUDE.md`), with the count corrected.

Opened by the **2026-09-16 rebase**:

- **[00-three-production-grid-reproductions]** *(rebase, 2026-09-16; **narrowed at the re-anchor,
  2026-09-16**; assigned to prompt 01)* — the test tree holds three constructions each called "the
  production source grid" and they are three different grids. `ComputeTargets/tests/wkb_reference.py:152` is a bare `np.logspace` reproducing
  `populate_z_sample` and citing `main.py:410-419` — **version 0**, what production built before
  `qcd-background-audit` prompt 11. `ComputeTargets/tests/test_background_segmentation.py:90`
  passes `break_z` and `feature_z` but no `spacing` profile — **version 1**, prompt 11's grid.
  `main.py:911-930` passes the curvature spacing profile — **version 2**,
  `SOURCE_GRID_CONSTRUCTION_VERSION = 2`, 1,996 samples on QCD and 1,778 on LambdaCDM against
  version 0's 1,732. **Impact:** every figure in README §6 and every figure in
  `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` was scored on version 0, because
  `tk_numeric_atol_sweep.py:215` imports the first of the three; a tolerance chosen from those
  figures would be chosen for a grid production does not use. Not a wrong *number* — each was
  correct for the tree it was taken on — but an unmarked one, which is why README §5 rule 6 now
  requires the generation beside the figure. **Next step:** prompt 01 builds one reproduction at
  version 2, lifting `source_grid_spacing_profile` and `cosmology_feature_redshifts` from `main.py`
  with `load_main_py_functions`, and has the other sites import it. Keep versions 0 and 1
  constructible and named: prompt 17's figures and
  `test_background_segmentation`'s assertions are scored on them and must not be silently
  re-based.

  > **Narrowed at the re-anchor, 2026-09-16** (additively; `RECONCILIATION.md` §7.3). It is
  > **four** constructions, not three, and the rebase missed the fourth — which was already there
  > at `acd5b8e`. `ComputeTargets/tests/test_source_grid.py:127` `_production_grid` **is the
  > version-2 construction**, complete, under the suite, and lifting both `main.py` functions the
  > way the next step above prescribes; and `:151` `_production_base_grid` is a v0 base that is
  > deliberate and named, mirroring `main.py:944`'s own base-grid step. **The corrected next step:
  > prompt 01 *hoists* `_production_grid` out of that test module into something the other sites
  > can import, and repoints `wkb_reference.py:152` and `tk_numeric_atol_sweep.py:215` at it.** It
  > is private and it is in a test module; that is now the whole defect. The **impact is
  > unchanged** — every figure in README §6 and in `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` is
  > still scored on version 0, because `tk_numeric_atol_sweep.py:215` still imports the v0 one. One
  > further constraint the re-anchor adds: `cosmology_feature_redshifts` now asks the cosmology for
  > `z_matter_radiation_equality` / `z_matter_lambda_equality` and **raises with no fallback**, so
  > the stand-in prompt 01 lifts it against must answer both (`RECONCILIATION.md` §7.4);
  > `test_source_grid.py:189` already carries one that does.

  > **Narrowed again by prompt 01, 2026-09-16** (additively). The hoist is done and is
  > bit-identical: `ComputeTargets/tests/wkb_reference.py` now carries
  > `source_grid(generation, …)` over three **named** generations — `SOURCE_GRID_V0`,
  > `SOURCE_GRID_V1`, `SOURCE_GRID_V2` — with **no default**, tagged
  > `SOURCE_GRID_V2_REPRODUCES_VERSION = 2` and cross-checked against production's own
  > `SOURCE_GRID_CONSTRUCTION_VERSION` rather than asserting it. `test_source_grid.py:127`,
  > `test_background_segmentation.py:90` and `tk_numeric_atol_sweep.py`'s two geometries are
  > repointed, the last of them naming version 0 explicitly; `test_source_grid.py:152`
  > `_production_base_grid` is left alone, as it should be. Verified: 1,732 / `0960e169` (v0),
  > 1,773 / `81c6e682` (v1 QCD), 1,996 / `4849552b` (v2 QCD), 1,778 / `60a3205a` (v2 LambdaCDM),
  > every one bit-identical to the construction it replaces, and
  > `tk_numeric_atol_sweep.py`'s entire 294-line output unchanged but for the wall clock it prints
  > about itself.
  >
  > **What is left, and it is the whole reason this is a narrowing and not a closure:**
  > `wkb_reference.production_source_grid` still exists with its historic behaviour, because
  > **28 call sites in 20 files** import it — 11 test modules under `ComputeTargets/tests/` and 9
  > scripts under `docs/gktk-remedial/` and `docs/qcd-background-audit/` — and none of those files
  > was in prompt 01's scope. They are all version 0 and all correct; what they do not do is *say
  > so*. The prompt asked that the bare name "fail loudly rather than default", which cannot be
  > done without editing all twenty (log 01, deviation 1). Its docstring now opens "**The
  > version-0 source grid**". **Next step:** a prompt given those twenty files repoints them at
  > `source_grid(SOURCE_GRID_V0, …)` and deletes the alias. It is mechanical and every call is
  > bit-identical; it is scope, not difficulty.

- **[00-gk-numeric-never-swept-and-carries-the-cost]** *(rebase, 2026-09-16; assigned to prompt
  03)* — the campaign's central prior, "the error is set by `rtol`", is one clean measurement and
  one diagonal. `GkTk-remedial` prompt 17 holds `atol = 1e-13` and moves `rtol` alone, so it
  separates the axes — in the sector that is **one object per $k$, 50 per model**
  (`main.py:1180`, `:1215` at the re-anchor). Review §10.1 on $G_k$ moves `(1e-10, 1e-8)` → `(1e-13, 1e-11)`, a diagonal, at
  four source redshifts on two models
  (`docs/gk-wkb-review-fable-2026-09-09.md:469-474`) — in the sector that is one object per
  $(k, z_{\rm source})$, ~65,000 per model. **Impact:** the +23–25 % evaluations one decade of
  `rtol` costs is free where it was measured and is the campaign's entire compute decision where it
  was not; and README §7 D1 attaches the ~65,000 object count to the wrong sector. Worse, the same
  review paragraph says the consumer's cubic spline of the numeric $G$ carries 1e-5 to 1e-4 near
  the hand-over, "the larger error by two orders" — so the honest outcome may be that $G_k$'s
  `rtol` should not move at all. **Next step:** prompt 03 sweeps $G_k$ in both axes on three models
  over the production response grid and measures the consumer-spline floor beside it, before any
  `rtol` is recommended for either sector.

Opened by **prompt 01**, 2026-09-16:

- **[01-density-criterion-imposed-outside-the-wkb-region]** *(orchestrator review of prompt 01,
  2026-09-16; **narrowed 2026-09-17, no longer blocking; measured by prompt 02a, 2026-09-17**; unassigned — candidate for **T7**)* — `main.source_grid_spacing_profile` imposes the
  fourth-derivative equidistribution criterion over the band `residual_node_range` returns, and
  that band reaches **1.5 to 2.1 e-folds outside the horizon**, where the Liouville-Green phase
  spline the criterion exists to protect is never evaluated. The horizon condition is
  `k/aH = omega_0 (1+z) = 1`, with `leading = omega_0^2 = (k/H)^2` and `H` the ordinary Hubble
  rate: **`omega_0 = 1` is not horizon crossing** and misreads the band by five orders of
  magnitude in z. Measured on `QCD_Cosmology` at its own anchor, `Gk` sector:

  | $k$ [1/Mpc] | $z$ at $k = aH$ | band top | $k/aH$ there | e-folds outside |
  |---|---|---|---|---|
  | 1e+05 | 5.08e+10 | 2.67e+11 | 0.196 | 1.63 |
  | 1.39e+05 | 7.04e+10 | 3.21e+11 | 0.228 | 1.48 |
  | 1.92e+05 | 9.77e+10 | 8.25e+11 | 0.128 | 2.06 |
  | 2.26e+05 | 1.15e+11 | 8.44e+11 | 0.147 | 1.92 |

  `RESIDUAL_WKB_REGION_MARGIN`'s own comment states that production anchors sit **three e-folds
  inside** the horizon, so the criterion is imposed roughly **five e-folds beyond the last place
  its consumer exists**. At the raise point of
  `[01-v2-density-raises-at-the-qcd-production-anchor]` the mode is **1.78 e-folds outside** the
  horizon and `|C| / omega_0^2 = 5.9e+04` — the "correction" is 59,000x the leading term, so the
  expansion being differentiated has stopped meaning anything, and the criterion is in effect
  chasing its own breakdown. The cause is a **reuse**, not a coding error: that margin was designed
  as a *permissive* bound so the band never excludes a producer's anchor ("the range covers every
  anchor the producer accepts, with margin"), and the spacing profile reuses it as an upper bound
  on where the spline needs resolving. **Impact:** of the samples version 2 adds over version 1, at
  the LambdaCDM anchor — **QCD 154 of 223 (69%)** lie above horizon crossing for the smallest
  production $k$ and **36 (16%)** above crossing for *every* production $k$; **LambdaCDM 14 of 46
  (30%)** and **0 (0%)**. In that region the consumer is `GkNumericIntegration` /
  `TkNumericIntegration`, whose accuracy is governed by `atol` / `rtol` and the solver's own step
  control, not by a spline-interpolation bound. Because `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` the
  criterion may only ever *refine*, so these samples corrupt no result — they are **unjustified
  rather than wrong**, and 223 in 1,996 is not a cost problem; the sharp consequence is the sibling
  issue's hard failure. **Next step:** decide whether the spacing profile should run over a
  horizon-based band of its own rather than over `residual_node_range`'s anchor-coverage band. That
  is a production change to the source grid, which README §0.5 holds fixed here; item **T7**
  already audits `RESIDUAL_WKB_REGION_MARGIN` at every production $k$, so prompt 04's charter is
  the natural home for it when that prompt is written.

  > **Narrowed 2026-09-17, and no longer blocking.** Prompt **02a** (README §3.2a, §7 D6) makes the
  > grid buildable without touching the band, so this issue stops being a hard failure and becomes
  > what it always was underneath: unjustified refinement with a measurable cost. The mechanism is
  > now stated — the band is established **node-wise** and the criterion is evaluated **off-node**,
  > so the expansion's non-existence is reached at `u ± delta` between two nodes that both pass the
  > margin test. 02a's **guarded-node count is the evidence this issue has been waiting for**: 53
  > nodes at QCD's own anchor, zero at either published anchor, which is a direct measure of how far
  > the band overreaches and where. **Still unassigned as a decision, still a candidate for T7**:
  > whether the spacing profile should run over a horizon-based band of its own is prompt 04's to
  > recommend, and 02a is explicitly forbidden from pre-empting it.

  > **Measured by prompt 02a, 2026-09-17** (additively). The guarded-node census is the direct
  > measure of the overreach this issue asserts, and it is the evidence the issue has been
  > waiting for. At production geometry, **source-grid generation version 2**, fifty wavenumbers,
  > both sectors:
  >
  > | cosmology / anchor | guarded | band-node evaluations | fraction | Gk | Tk | cases | worst single band |
  > |---|---|---|---|---|---|---|---|
  > | QCD at LambdaCDM's, 2.0636395964161516e+16 | 0 | 136492 | 0 | 0 | 0 | 0 / 100 | 0 |
  > | LambdaCDM at its own | 0 | 150932 | 0 | 0 | 0 | 0 / 100 | 0 |
  > | **QCD at its own, 3.30033444460513e+16** | **53** | **136453** | **3.884e-04** | **34** | **19** | **53 / 100** | **7.716e-04** |
  >
  > **The shape matters as much as the total: every affected case guards exactly one node.** So
  > the band reaches *just* past the edge of the region at a single lattice node, rather than
  > running for a stretch through a region of breakdown — which is a different claim from the
  > 69%-of-added-samples figure above, and a narrower one. Both are true and they measure
  > different things: that one counts samples the criterion *added* above horizon crossing, this
  > one counts nodes at which the expansion it differentiates **does not exist at all**.
  >
  > **Still unassigned as a decision, still a candidate for T7.** Prompt 02a was forbidden to
  > touch the band and did not: `residual_node_range` returns exactly what it returned before.
  > `SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05` now caps what the guard may absorb, at 64.8x the
  > worst measured band, so a *materially* worse band refuses rather than being filled silently —
  > but the ceiling is an acceptance bound, not an answer to where the criterion should apply.

Opened by the **orchestrator's review of prompt 02a's charter**, 2026-09-17:

- **[02a-grid-digest-not-reproducible]** *(2026-09-17; assigned — **T6** / prompt 03 for the
  prerequisite, prompt **05** for the fix)* — the source and response grid tags digest the **exact
  bits** of the grid's values (`CosmologyConcepts.redshift.redshift_grid_digest`, `main.py:854`),
  but `z_init` is a root-solve output, so the tag is not reproducible across machines, library
  versions, or any change that forces the anchor to be re-derived. **The binding number is not the
  one the record quotes.** `_solve_horizon_exit` calls `root_scalar(..., xtol=atol, rtol=rtol)` in
  `u = log(1+z)` (`CosmologyConcepts/wavenumber.py:979`) and Brent stops at `xtol + rtol*|u|`; with
  `rtol = DEFAULT_REL_TOLERANCE = 1e-8` at `u ≈ 37.6` that is **3.8e-7 relative**, not the
  `xtol = 1e-10` README §6.2's `wavenumber_exit_time` row and prompt 02's inventory both name. The
  `xtol` term never binds. Measured against a converged re-solve (`xtol=1e-300, rtol=1e-14`, the
  tolerance `_solve_T_z` already uses): the shipped anchor is **3.6e-13** off on LambdaCDM and
  **2.5e-9** off on QCD, and neither is bounded by better than 3.8e-7.
  **Impact.** 3.8e-7 is *coarser* than `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, the tolerance
  at which `Datastore/SQL/ObjectFactories/redshift.py:40` matches an existing redshift row. So the
  two invalidation criteria in the pipeline disagree by six orders of magnitude, and the failure
  mode is silent total cache invalidation rather than corruption: measured across two anchors 3.6e-13
  apart, **all 1778 samples differ bitwise and zero of them exceed 1e-7**, so every redshift row is
  re-matched and reused with its old `store_id` while the grid tag turns over completely and every
  lookup filtered on `SourceZGridSizeTag` misses — in the `Gk` sector, ~65,000 objects per model
  recomputed against rows that were already correct. This is the *inverse* of the collision the
  digest was introduced to prevent (`main.py:837`), so the digest is not wrong; it is quantised
  finer than anything upstream of it is determined to.
  **Not a datastore defect, and not reached by a plain re-run.** `wavenumber_exit_time.build()`
  reads `z_exit_suph_e5` back from a `Float(64)` column when the row exists
  (`Datastore/SQL/ObjectFactories/wavenumber.py:197`, `:292`), so within one datastore lineage the
  anchor is bit-stable and the grid is reproducible — which is the regime every figure in this
  campaign has been taken in, and why nothing has tripped over it. It bites when the anchor is
  *re-derived*: a fresh or dropped store, another machine or libm, or a change of the cosmology row,
  which is what `qcd-background-audit` did twice.
  **Next step, in two parts.** (i) **T6 / prompt 03** measures `wavenumber_exit_time`'s pair and
  recommends the tightening; at `rtol = 1e-14` the anchor is pinned to 3.8e-13, measured. (ii)
  **Prompt 05** defines one design tolerance and applies it to *both* the redshift row match and the
  digest quantisation, so that the tag can never distinguish two grids the datastore cannot. With
  the anchor tightened the wobble budget is 3.8e-13 (anchor), ~1e-14 (break redshifts, already at
  Brent's floor), 3.6e-15 (the `u → z` recovery) and ~1e-15 (libm `pow`/`log10`/`log1p`), so a
  design tolerance around **1e-11** sits two orders above the worst contributor. Carry the coupling
  with it: `SOURCE_GRID_MIN_SEPARATION = 10.0 * DEFAULT_REDSHIFT_RELATIVE_PRECISION`
  (`CosmologyConcepts/wavenumber.py:135`) only *relaxes*, but the four-constraint argument above
  `SOURCE_GRID_BREAK_STANDOFF` (`:79`) loses its first bullet and needs rewriting, not just
  retuning. **Rounding buys a margin, not a proof** — straddle probability ~ `N * wobble / quantum`,
  ~2e-4 per grid at these numbers; the exact alternative is to digest the *determining data*
  (snapped `z_init`, `z_end`, `samples_per_log10z`, construction version, snapped break and feature
  lists, and the integer subdivision vector, whose ties are a measured 6e-3 clear), which prompt 05
  should record as considered even if it ships the quantised-values version.
  **Explicitly not prompt 02a's** (README §4, §7 D6): 02a's acceptance is bit-identity with the
  published digests, and tightening the anchor would move both.

Opened by **prompt 02**, 2026-09-16:

- **[02-oneloopintegral-is-a-ninth-keyed-object-type]** *(prompt 02, 2026-09-16; **unassigned —
  prompt 02 §8's stop condition, the user decides where it goes**)* — README §2 (a) counts eight
  object types keyed on an accuracy parameter. There are **nine**.
  `Datastore/SQL/ObjectFactories/OneLoopIntegral.py:101-116` declares `atol_serial` and
  `rtol_serial` as indexed, non-nullable foreign keys into `tolerance`, exactly as the other eight
  do, and `build()` filters on both at `:153-154`. The table is registered
  (`Datastore/SQL/Datastore.py:120`), sharded on `k` (`config/sharding.py:34`), given a
  client-pool budget (`ClientPool.py:43`) and given a drop action (`Datastore.py:130`).
  **`main.py` never builds one** — its only `OneLoop` strings are two `store_tag` labels at
  `:1021-1022` — and `ComputeTargets/OneLoopIntegral.py`'s `compute()` (`:94-112`) does nothing
  but replace a label, so the object count of the sector is **0**. Two defects inside that stub,
  folded in here rather than opened separately: `:105-107` raises "value haa already been
  computed" when `self._value is **None**` (inverted condition, and the typo), and `store()`'s
  first `raise` at `:116` is followed by an unreachable comment. **Impact:** small today and
  structural tomorrow. Nothing is stored, so nothing is wrong in the datastore; but the campaign's
  count of its own subject is wrong by one for the second time, prompt 05's `ast` guard must
  enumerate **nine** targets rather than eight if this one is in scope, and a target whose schema
  is already keyed on the shared pair will inherit whatever prompt 05 decides **unless someone
  decides otherwise on purpose**. **Next step: the user decides.** It is not obviously prompt 03's
  (no solver to sweep) nor prompt 04's (no order); it may be the cheap moment for prompt 05,
  before any row exists to invalidate, or it may be premature to decouple a target that computes
  nothing. Prompt 02 measured it and stopped there, as §8 requires. Evidence:
  `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.1.

- **[02-wavenumber-exit-time-tolerance-is-an-inequality-key]** *(prompt 02, 2026-09-16; candidate
  for prompts 03 and 05)* — README §2 (g) says "every accuracy parameter is part of its object's
  lookup key, so a new one makes every existing row of that type unreachable". True of eight of the
  nine; **false of `wavenumber_exit_time`**. Its query
  (`Datastore/SQL/ObjectFactories/wavenumber.py:242-254`) joins the `tolerance` table twice and
  filters `stored.log10_tol - requested.log10_tol <= DEFAULT_FLOAT_PRECISION` on each — accept any
  row **at least as tight as** the request — then orders by `log10_tol` **descending** and takes
  `one_or_none()`, i.e. the **loosest** qualifying row. It does the same with
  `stepping >= target_stepping`. **Impact:** tightening misses and recomputes, as an equality key
  would; **loosening hits a tighter stored row and returns its `z_exit`**, and the object's own
  `atol`/`rtol` properties (`CosmologyConcepts/wavenumber.py:729-734`) then report the *stored*
  pair, not the requested one. A tolerance sweep run through the datastore would therefore be
  served the same answer at several of its points and would measure nothing.
  `MultipleResultsFound` is caught at `:260`, so two qualifying rows raise rather than choosing.
  This is deliberate "best available" behaviour and is **not proposed for change here**.
  **Next step:** prompt 03 sweeps this target by calling
  `CosmologyConcepts.wavenumber._solve_horizon_exit` directly and never through `object_get`;
  prompt 05's `ast` guard must not assume equality semantics at this site. Evidence:
  `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.3.

- **[02-shared-atol-doubles-as-a-float-comparison-epsilon]** *(prompt 02, 2026-09-16; candidate for
  prompt 05)* — `DEFAULT_ABS_TOLERANCE` is not only the shared solver tolerance. It is also a bare
  float-comparison epsilon at **seven** sites with no connection to the Green's-function ODE:
  `ComputeTargets/GkSource.py:96`, `:104`, `:275`;
  `Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, `:790`;
  `LiouvilleGreen/WKBtools.py:83`. All seven are `fabs(a - b) < DEFAULT_ABS_TOLERANCE`-shaped
  guards — a redshift-equality check, a residual check, a phase-modulo check. **Impact:** prompt 05
  is chartered to retune or split this constant, and the moment it does, all seven comparison
  thresholds move with it, silently and in modules that are not in that prompt's file list. At
  `1e-10` none of them is near its margin, so nothing is wrong today; the hazard is entirely in the
  change. **Next step:** prompt 05 gives these seven sites a constant of their own — or states, in
  its log, that it has checked each one against the new value. Evidence:
  `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §4, closing paragraph.

- **[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]** *(prompt 02, 2026-09-16;
  unassigned; prompt 05 touches the same six files)* — `extract_TkWKB_data.py:433-445` builds one
  `query_payload` with `"atol": atol` where `atol = DEFAULT_ABS_TOLERANCE` (`:364`) and uses it for
  **both** `TkNumericIntegration` and `TkWKBIntegration`. `main.py` writes every
  `TkNumericIntegration` under `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` — its own comment at
  `main.py:3475-3479` says "every `TkNumericIntegration` `object_get` — the work items and **every
  lookup** — must use it" — and that target's lookup filters `atol_serial ==`. **So this query
  cannot match a production row.** None of the six `extract_*.py` readers imports
  `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` at all. **Impact:** an extraction script that has been unable
  to plot the numeric limb of $T_k$ since `GkTk-remedial` prompt 12 shipped the split constant.
  Not verified by running it — that needs a datastore, which prompt 02 may not stand up — so what
  is established is that the key cannot match, not what the script does next. **Next step:** the
  one-line fix is to import the constant and pass it for the numeric target only; prompt 05 already
  has to revisit all six readers when it splits the constants further, and this is the same edit.
  Evidence: `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §5.1 (the derived
  `atol_serial ==` predicate) and log 02, observation 1.

Assigned to this campaign from other boards (each stays on the board that holds its measurements;
the closure is recorded there):

| Issue | Owning board | Assigned to | Why here |
|---|---|---|---|
| `[12-tk-numeric-atol-largest-k-excursion]` | GkTk-remedial | prompts 03, 05 | Assigned 2026-09-12. Its `atol` half is settled — the user kept `1e-13` — and what remains is the `rtol` retuning, which is D1. It closes when prompt 05 ships a settled `rtol`. **Its cost figures in `docs/OPEN_ISSUES.md` §1.5 were corrected at the rebase** (`RECONCILIATION.md` §2.4): the QCD $T_k$ object is 8,986 right-hand-side evaluations, not ~31.5k |
| `[01-convergence-block-has-a-separate-generator]` | qcd-background-audit | prompt 04 | **Assigned 2026-09-16.** The `convergence` block of `ComputeTargets/tests/wkb_reference_data.json` records $N_\tau = N_{c_s\tau} = N_F = N_\rho = 4$, was generated 2026-09-10, and its `decision.recommended_scheme` is `"branch+knots"` — a knot set `qcd-background-audit` prompt 07 removed. Prompts 08 and 09 of that campaign each declined it on scope. Prompt 04 here is the first prompt anywhere whose charter is the orders themselves, so it cannot avoid re-running the generator; that it must then write a fixture and edit `test_background_tau.py` is README §7 **D5**, **settled yes at the 2026-09-16 re-anchor** — this issue is now closable here |
| `[20-wkb-gauss-orders-not-in-lookup-key]` | GkTk-remedial | prompts 04, 05 | **Assigned 2026-09-16.** `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER`, `RHO_GAUSS_ORDER` and `RESIDUAL_WKB_REGION_MARGIN` are configuration axes in no lookup key, while the `atol`/`rtol` columns that *are* in the key describe nothing. That is README §7 **D3**, and D3 is the user's stated target for the campaign: for a Liouville–Green-type representation the key should carry an order |

Recorded by the rebase, **not owned here** and not scheduled (README §0.5):

- `[11-stop-point-root-tolerance]` (hand-over campaign, `docs/OPEN_ISSUES.md` §1.1) —
  `find_phase_extremum`'s `root_scalar(xtol=1e-6, rtol=1e-4)`,
  `LiouvilleGreen/integration_tools.py:95`. It appears in prompt 02's inventory and in
  `docs/TOLERANCE-PROVENANCE.md`, and it is not retuned here.
- `LambdaCDM_GenericEOS.py:1008` — **settled 2026-09-16 by `prompts/background-solver-robustness`,
  which is "whoever owns that file". It is no longer an unbracketed secant and no longer at
  `:1008`.** It is `_find_rho_equality`'s `root_scalar` at **`:1137`** (`def` at `:1001`), Brent on
  a $\sqrt2$ bracket expanded in $1+z$ about the caller's guess and clamped to the $T(z)$
  representation's own bounds, at **`xtol=1e-300, rtol=8.9e-16`** — Brent's own $4\varepsilon$
  floor, decided by the user on that campaign's README §7 **D1** as amended, because `rtol=1e-14`
  was measured to stop 7 ulp from an independent reference. **Its provenance is
  [`prompts/background-solver-robustness/PROVENANCE.md`](../background-solver-robustness/PROVENANCE.md)
  §3, which is written in the shape `docs/TOLERANCE-PROVENANCE.md` will want: prompt 02 here should
  lift that entry rather than re-derive it.** Two further things prompt 02's inventory needs and
  the old bullet could not have known: the *quantity* is **not** a diagnostic — since that
  campaign's prompt 09 the model answers for its own equality redshifts and they are production
  source-grid sample locations inside a `BackgroundModel` lookup key (PROVENANCE.md §3.1) — and the
  same document's §1 and §2 settle the file's **other two** solves, so all three of `:579`, `:864`
  and `:1008` in README §3.2's list arrive here already established.
- `ComputeTargets/QuadSourceIntegral.py:1550` still says "the pipeline supplies
  `DEFAULT_QUADRATURE_ATOL = 1e-25`"; the constant has been 1e-32 since `source-remediation`
  prompt 12. A stale comment in a file README §0.4 puts out of bounds.

---

## 4. Resolved issues

Closed by **prompt 02a**, 2026-09-17:

- **[01-v2-density-raises-at-the-qcd-production-anchor]** — **CLOSED.** The version-2 source
  grid now builds at every production anchor on every production cosmology. The stencil
  evaluation in `main.source_grid_spacing_profile` is guarded where the Liouville-Green
  expansion does not exist: such a node is marked `usable = False` and filled by the
  log-interpolation the criterion already applies to a declared crossing's neighbourhood, and
  the guarded nodes are **counted** rather than absorbed silently.

  **What closed it, measured.** `_solve_horizon_exit(QCD, k = 3e8/Mpc, -5)` =
  **3.30033444460513e+16**, reproducing the board's figure to 15 digits. The construction
  raised there at every relative `z_init` perturbation from **1e-16 to 1e-8** and first built
  clean at **1e-6**, with a *stable* 53 guarded nodes across the whole trip band — so this was
  never one float's accident, and `main.py` genuinely could not build a QCD source grid. QCD at
  its own anchor now builds at **2034 samples / `21ffc126`, 53 guarded nodes**.

  **The acceptance was bit-identity and it held.** QCD at LambdaCDM's anchor is **1996 /
  `4849552b`** and LambdaCDM at its own is **1778 / `60a3205a`**, each with **zero** guarded
  nodes — so the `except` branch is never entered on either and the guard is provably inert on
  every figure in the record. Neither digest moved.

  **What it did not do**, and whose each piece is: the band is exactly as `residual_node_range`
  returns it (**T7**, prompt 04); the crossing mask is not reordered, because that changes which
  nodes are evaluated and could move a published grid; the anchor solve is not tightened (**T6**,
  prompt 03); the digest is untouched (prompt 05). Evidence:
  [`logs/02a-source-grid-density-guard.md`](logs/02a-source-grid-density-guard.md).

---

## 5. Standing notes

1. **A number without its reference's drift beside it is not a measurement** (README §5 rule 5).
   Every figure quoted against a converged reference carries that reference's drift, and no
   conclusion is drawn from a signal that does not exceed it.

2. **A number without its grid generation beside it is not comparable** (README §5 rule 6, new at
   the rebase). Version 0, 1 or 2 — say which. The two campaigns that closed before this one are
   full of figures from all three, and nothing in the record distinguishes them.

3. **The floors are not targets, and the target rule says how one becomes the other**
   (README §2 (f) and **§6.1**). An agent reporting an accuracy below a declared floor has made an
   error, and it is a campaign-wide stop — not a caveat, not a footnote. **The QCD $H(z)$
   discontinuity floor is no longer one of them**: `qcd-background-audit` prompts 04–06 removed it
   and the equivalent phase error is 0.000e+00 rad. **§6.1 rule 5 is the one qualification**: a
   prompt may *re-measure* a floor and supersede an inherited figure, and for $G_k$'s consumer
   spline it must; the stop is a claim below the floor the prompt has itself just measured.

4. **Cost is a per-object count times an object count** (README §2 (c)). $T_k$ numeric is 50
   objects per model; $G_k$ numeric and both WKB sectors are ~65,000. A percentage without the
   multiplier is not a cost.

5. **Counts, not wall time** (README §2 (i)): this machine's elapsed times overstate by up to 53 %.

6. **Four of the eight keyed targets have no tolerance to converge** (README §2 (a)). An agent
   proposing to tighten a WKB or `BackgroundModel` tolerance has misread the tree; the knob there
   is an integer order.

7. **`QuadSourceIntegral` is read-only here** (README §0.4). Touching it, `QuadSource.py`,
   `phase_groups.py` or `AdaptiveLevin/` is a stop.

8. **More datastore objects is the intended outcome, not a cost** (README §4.2, D2 settled
   2026-09-12 and restated campaign-independently by `qcd-background-audit` `21d80b2`). No prompt
   may argue for keeping a shared constant on the grounds that decoupling multiplies rows.

9. **No parameter without its provenance** (README §1.2, §5 rule 9). One recommended or shipped
   without its five provenance fields in the log is an unfinished prompt.

10. **`DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` is settled** (the user, 2026-09-12, on
    `GkTk-remedial` prompt 17's recommendation). It is not reopened here; its provenance entry is
    written from that campaign's record.

11. **`BREAK_POINT_KIND`, the source grid and `RESIDUAL_WKB_REGION_MARGIN`'s value are held fixed**
    (README §0.5). Prompt 04 measures what the margin is worth, because nobody has; changing any of
    the three is another campaign's decision and touching one is a stop.

12. **The target is the loosest setting that clears the measured floor** (README §6.1), swept
    loose-to-tight, scored at the **maximum** over the whole production grid on all three models,
    with the cost at that setting and one step either side. Where the error is already below the
    floor the target is the word `unchanged` and the row records the dominating factor — that is a
    result. Recommending a setting two decades tighter than the one that first clears is the same
    kind of error as recommending one that misses.

13. **The constant-$w$ anchors cover more than $G_k$ and $T_k$** (README §3.1). `RadiationModel`
    carries exact closed forms for $\tau$, $c_s\tau$, $F$, $\theta_G$, $\rho_G \equiv 0$ and
    $\rho_T$, and $z_{\rm exit}$ is the elementary inversion $1 + z = k/(H_0 e^{N})$ — so **T6 and
    T7 have oracles, not only self-convergence**, and a drift quoted for any of them without the
    oracle error beside it is uncalibrated.

14. **Every measurement in this campaign goes through the facility** (prompt 01, board item T1).
    `ComputeTargets/tests/convergence_reference.py` is the one implementation of the convergence
    test, and it is shaped so that notes 1 and 2 are the easy path: `reference_drift` has no
    default for the smallest difference the caller will report and returns the verdict with the
    numbers, and a geometry cannot be built without naming its source-grid generation. A prompt
    that writes its own drift statistic has stepped around both rules, and the reviewer should ask
    why.

15. **The baseline is `bc6dc97`, not `acd5b8e`** (`RECONCILIATION.md` §7, 2026-09-16).
    `ComputeTargets` **452**, `CosmologyModels` **39**. A prompt quoting a suite count, a file
    line number or a "what the tree does" claim from a document dated before the re-anchor must
    re-resolve it: `main.py` alone gained 42 lines above its citation sites, and §7.6 tabulates
    the ones this campaign's documents used. The five results of README §0.3 are unaffected.

16. **The suite counts moved when prompt 01 landed, and one `ComputeTargets` test fails on this
    machine** (prompt 02, 2026-09-16). `ComputeTargets` is **484**, not the 452 of the `bc6dc97`
    re-anchor: prompt 01 added a test module. `CosmologyModels` is still **39, OK**. Of the 484,
    **483 pass**; `test_tk_wkb_phase.TestCost.test_wall_time_per_object` fails at
    `cold = 0.0605 s` against its `0.06 s` limit, reproducibly, **with no `.py` file changed**.
    It is already attributed to `[07-tk-per-object-cost-is-all-setup]` on the `GkTk-remedial`
    board by `prompts/phase-representation` log 02 observation 5, which saw 0.0638–0.0979 s on
    this machine four runs of four. **A prompt here that sees this failure has not caused it**;
    note 5 (counts, not wall time) is why. No issue is opened for it here.

17. **There are nine keyed object types, not eight, and `wavenumber_exit_time`'s key is an
    inequality** (prompt 02, 2026-09-16; note 6 is unchanged and still correct about the four).
    The ninth is `OneLoopIntegral`, which production never builds. And README §2 (g)'s "a new
    accuracy parameter makes every existing row of that type unreachable" holds for eight of the
    nine: `wavenumber_exit_time` accepts any stored row at least as tight as the request and
    returns the loosest such, so **loosening silently reuses a tighter row**. A sweep of that
    target must bypass the datastore entirely. Both are §3 issues above, with the evidence in
    `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.1 and §2.3.

18. **A number without its *anchor* beside it is not comparable either** (prompt 02a, 2026-09-17;
    the same species as note 2, one level down). The two production cosmologies do **not** share a
    `z_init`: `wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM = 2.0636395964161516e16` and
    `PRODUCTION_Z_INIT_QCD = 3.30033444460513e16`, and `PRODUCTION_Z_INIT` is an alias of the
    first. The constant's comment used to claim one value served both, and that claim is why
    **every version-2 QCD figure previously in the record was taken at LambdaCDM's anchor** —
    including `test_source_grid.py`'s 1996 / `4849552b`, `RECONCILIATION.md` §2.7's "1,996 samples
    on QCD" and `docs/qcd-background-verification.md` §10's density measurements. None of those is
    wrong; each is at the other anchor. **The QCD production grid at QCD's own anchor is 2034
    samples / `21ffc126`**, and it is anchor-sensitive to far below the anchor solve's own 3.8e-7
    convergence (`[02a-grid-digest-not-reproducible]`), so quote the anchor with the digest.

19. **`SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05` bounds what the density guard may absorb**
    (prompt 02a, 2026-09-17). It is in no lookup key and cannot change a grid that builds — it can
    only convert a build into a refusal — so it invalidates nothing. It exists because a guard with
    no ceiling would let an arbitrarily misplaced band be filled by log-interpolation in silence,
    which is worse than the raise it replaced. The worst production band reaches **7.716e-04**;
    the ceiling is **64.8x** that.
