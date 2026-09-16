# Implementation state — tolerance and convergence campaign

**Campaign:** [`README.md`](README.md) · **Rebase record:** [`RECONCILIATION.md`](RECONCILIATION.md)
· **Logs:** [`logs/`](logs/) · **Orchestrator:** [`orchestrator/`](orchestrator/)
**Planned:** 2026-09-12 at `622b84b` · **Rebased:** 2026-09-16 at `acd5b8e` ·
**Re-anchored:** 2026-09-16 at `bc6dc97` (`RECONCILIATION.md` §7)
**Baseline commit:** `bc6dc97` (`tolerance-convergence`, 25 commits ahead of `main`, clean; suites
re-run for the re-anchor and green — `ComputeTargets` **452**, `CosmologyModels` **39**)
**Superseded baseline:** `acd5b8e`, `ComputeTargets` 447, `CosmologyModels` 30 — the anchor of any
figure in this campaign's documents dated before 2026-09-16 19:32
**Last updated:** 2026-09-16 · **Status: planned, rebased and re-anchored; not started.**
**Every user decision needed to start is settled** — D1 and D3 are post-audit gates by design, D2
settled 2026-09-12, D4 settled by README §0.4, **D5 settled yes 2026-09-16**.
**Prompts 01 and 02 are written, with their orchestrator prompts; 03–06 are deliberately held**
until 02's inventory lands (§1 below). The campaign is ready to dispatch prompt 01.

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
| 01 | The convergence harness and one production grid | README §2 (b), (h); `[00-three-production-grid-reproductions]` | Opus | ✍️ [`01-…`](01-convergence-harness-and-grid.md) | ⬜ | | |
| 02 | The accuracy-parameter inventory | README §2 (a), (c), (g); `RECONCILIATION.md` §2.1 | Opus | ✍️ [`02-…`](02-accuracy-parameter-inventory.md) | ⬜ | | |
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
> **The staging:** 01 and 02 now; **03 and 04 written after 02 lands**, against its table rather
> than against a guess at it; 05 after the user settles D1 and D3; 06 last. This costs nothing in
> elapsed time — §4.1 already declares a stop after each of 02, 03 and 04, where the user is in the
> loop anyway. Orchestrator prompts are staged with them
> ([`orchestrator/README.md`](orchestrator/README.md)).

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
| T1 | **MACHINERY** | One reusable convergence facility, in the test tree, covering every target and calibrated against the constant-$w$ anchors at every use, with "one step tighter" meaning a decade for a tolerance and one order for a Gauss order | 01 | ⬜ |
| T2 | **MACHINERY** | **One** reproduction of the production source grid at `SOURCE_GRID_CONSTRUCTION_VERSION = 2`, with the version-0 and version-1 constructions retained and named rather than silently re-scored | 01 | ⬜ |
| T3 | **MEASUREMENT** | The accuracy-parameter inventory: every parameter, what it keys, whether it reaches a solver, what the real knob is, and the object count of the sector it keys | 02 | ⬜ |
| T4 | **MEASUREMENT** | `GkNumericIntegration` characterised over the production response grid on three models in both `atol` and `rtol`, against the consumer-spline floor — the sector with ~65,000 objects per model, never swept, and where the campaign's compute decision actually lives | 03 | ⬜ |
| T5 | **MEASUREMENT** | `TkNumericIntegration` likewise, re-taken on the version-2 grid and under its own `BREAK_POINT_ALL` policy | 03 | ⬜ |
| T6 | **MEASUREMENT** | `wavenumber_exit_time`'s root solve measured at all — nothing in the record says what `xtol = 1e-10`, `rtol = 1e-8` in $\log(1+z)$ buys or costs. **Scored against the exact $z_{\rm exit}$ on `RadiationModel` first** (README §3.1): $1 + z = k/(H_0 e^{N})$, confirmed at the rebase to 2.3e-16 relative or better | 03 | ⬜ |
| T7 | **MEASUREMENT** | $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$ and `RESIDUAL_WKB_REGION_MARGIN` audited at every production $k$ on the corrected background and the 3-point break set, replacing evidence generated 2026-09-10. **Every one of the four orders has a closed-form anchor on `RadiationModel`** (README §3.1), including $\rho_G \equiv 0$, which makes the $N_\rho$ measurement pure quadrature error with no reference to build | 04 | ⬜ |
| T8 | **DECISION** | The decoupled tolerance pairs settled by the user (§7 D1) and shipped with the measurement that chose each, in `config/defaults.py` | 05 | ⬜ |
| T9 | **DECISION** | What replaces the vestigial `atol`/`rtol` key columns on the three order-governed targets (§7 D3) — the user's stated target for the campaign | 05 | ⬜ |
| T10 | **PLUMBING** | Every `object_get` of a retuned target carries its own parameter, with an `ast` guard whose predicate reaches all eight targets and fails on an unclassified site | 05 | ⬜ |
| T11 | **HAND-OFF** | `QuadSourceIntegral` measured read-only and reported to `levin-refactor` / `qsi-phase-groups` | 06 | ⬜ |
| T12 | **PROVENANCE** | `docs/TOLERANCE-PROVENANCE.md` covers **every** accuracy parameter in the pipeline — including the ones this campaign inherits and does not set, and the ones nobody has ever chosen — with value, choosing measurement and its grid generation, competing floor, cost times object count, and citation (README §1.2) | 06 | ⬜ |

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

None yet.

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

14. **The baseline is `bc6dc97`, not `acd5b8e`** (`RECONCILIATION.md` §7, 2026-09-16).
    `ComputeTargets` **452**, `CosmologyModels` **39**. A prompt quoting a suite count, a file
    line number or a "what the tree does" claim from a document dated before the re-anchor must
    re-resolve it: `main.py` alone gained 42 lines above its citation sites, and §7.6 tabulates
    the ones this campaign's documents used. The five results of README §0.3 are unaffected.
