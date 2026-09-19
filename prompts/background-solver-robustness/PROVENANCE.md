# Provenance — the three `root_scalar` solves of the generic-EOS cosmology

**Written:** 2026-09-16, by prompt 06 of
[`prompts/background-solver-robustness/`](README.md), at the campaign's final commit.
**Subject:** every `scipy` root solve that `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`
owned when this campaign opened — two that `prompts/qcd-background-audit` settled, and one that
this campaign settled.

---

## 0. What this document is, and what it is not

`prompts/tolerance-convergence` README §1.2 requires that, when that campaign closes, **no accuracy
parameter in the pipeline is unexplained**, and its prompt 06 creates the project-wide
`docs/TOLERANCE-PROVENANCE.md`. That file does not exist yet and that campaign has not started.
[`AUDIT.md`](AUDIT.md) §7 says the cheap moment to settle these three is *before* that campaign's
prompt 02 takes its inventory, because an entry written afterwards is a follow-up amendment rather
than a settled record.

So this document writes the three entries **in the shape `docs/TOLERANCE-PROVENANCE.md` will
want**, as a campaign artefact. It does **not** create that file: doing so would collide with a
prompt another planned campaign is chartered for, and this campaign does not own it.
`prompts/tolerance-convergence` prompt 02 should **lift these three entries rather than re-derive
them**, and `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3 now points here.

**The code comments at the point of use are the primary record. This document is an index.** Each
entry names the file and line where the reasoning lives; where the two disagree, **the code is
right** and this document is stale. Nothing here was re-measured for its own sake: every figure
carries the campaign, prompt, log and **commit** it was taken at (README §5 rule 9), and the two
entries that `qcd-background-audit` settled are transcriptions of reasoning already in the tree.

**Two of these three are no longer what the audit found.** Solve 2 is no longer in the production
class at all — prompt 04 of this campaign moved it into the test tree, character-identical in its
`root_scalar` call — and solve 3 is no longer an unbracketed secant. The **Site** field of each
entry is the location at this commit; the history is in the entry's body.

---

## 1. Entry — `_solve_T_z`, the $T(z)$ node solve

| Field | Value |
|---|---|
| **Site** | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, `LambdaCDM_GenericEOS._solve_T_z` (`def` at **`:592`**), the `root_scalar` call at **`:636`**. The reasoning is in the code at **`:626-635`**. |
| **Value** | `xtol=1e-300`, `rtol=1e-14` |
| **Method** | Brent, **bracketed** — `bracket_lo`, `bracket_hi` at `:612`, `:609`, from the two analytic bounds $0.95\,T_{\rm CMB}(1+z)/G_{S,\infty}^{1/3}$ and $1.05\,T_{\rm CMB}(1+z)$, with an explicit straddle check at `:621` that raises before the solver is called. |
| **What it sets** | One node of the tabulated $T(z)$ representation (`_build_T_z_spline`). **Computed, and it reaches stored objects** — $T(z)$ is the background every target is built on. It is **not itself in any lookup key**; what is keyed is the *version* of the representation it belongs to, `T_Z_REPRESENTATION_VERSION` (`:428`, currently **6**), which `Datastore/SQL/ObjectFactories/QCD_Cosmology.py:53` carries as an indexed `T_z_representation` column and filters on at `:94`. **A later change to this tolerance therefore has to bump that version**, or two datastore rows that are numerically different become one identity. |
| **Call count** | Once per node, at model construction, never at evaluation time. **Measured at this commit: 3,176 calls** for `QCD_Cosmology(max_z=1e12)` and **3,175** for `max_z=1e20` — `DEFAULT_T_Z_SPLINE_SAMPLES = 3000` (`:72`) plus the segmented representation's per-branch padding. Once per model, per run. |
| **Choosing measurement** | `prompts/qcd-background-audit` **prompt 04** (`71b842a`, 2026-09-14), log `logs/04-tighten-node-solve.md`. On that campaign's 640-point probe set (`CosmologyModels/tests/T_z_reference.probe_set()`), against a defining-equation reference at `rtol=1e-14`: the node solve's max relative error goes **2.496e-05 → 0.0 (bit-identical)**, and the representation's **p90 goes 1.323e-05 → 1.936e-07** with the median 1.890e-07 → 1.071e-07. The argument for why a *loose* tolerance is worse than its size suggests is in the code at `:628-631`: each node converges independently, so a loose tolerance buys **uncorrelated scatter between neighbouring nodes**, and a spline through scattered nodes has a scattered derivative. |
| **Competing floor** | Brent's own convergence floor of $4\varepsilon \approx 8.9\times10^{-16}$; `rtol=1e-14` sits just above it, which is what the comment at `:631-633` means by "the tightest tolerance `root_scalar` can actually resolve". Independently of the tolerance, the representation's accuracy is floored by the interpolation, not by the node solve: after prompt 04 the node solve is bit-identical to the reference while the representation's max is still 7.2615e-04 (log 04). **There is nothing left to buy here by tightening.** |
| **Cost** | `_build_T_z_spline` **13.4 ms** per model construction at the chosen pair (log 04, "State handed to the next prompt"), against ~3,176 node solves; `T_photon` per-call cost **unchanged** by the tolerance, because the solve is not on the evaluation path. In the units this sector counts in — object constructions — the whole cost is paid once per `QCD_Cosmology`. |
| **Citation** | `prompts/qcd-background-audit`, prompt 04, [`logs/04-tighten-node-solve.md`](../qcd-background-audit/logs/04-tighten-node-solve.md); audit §3, T2. `T_Z_REPRESENTATION_VERSION` was **2** at that commit and is **6** now, through prompts 05, 06 and 07 of the same campaign; **the tolerance pair has not changed since `71b842a`.** |

**Not re-opened by this campaign, deliberately.** README §0.5 and `AUDIT.md` §4.3 both forbid it:
the solve is audited, bracketed, at the representable floor and carries its own comment. Prompt 02
of this campaign explicitly declined to carry its `rtol=1e-14` across to solve 3 (see §3's
*Competing floor*), and log 02 records why that is not an argument for changing this one:
`rtol=1e-14` here is per-node in a 3,000-node tabulation where **uncorrelated scatter** is the
thing being bought, not a single value whose last few ulp matter.

**One stale figure, recorded and not fixed.** The comment at `:627` says "~500 nodes, 8.3 us each
-- a few ms total". `DEFAULT_T_Z_SPLINE_SAMPLES` has been **3,000** since `qcd-background-audit`
prompt 06 (`a1d667a`), and the measured count at this commit is **3,176**. Nothing depends on the
figure — it is an aside inside an argument about *scatter*, which the node count does not affect —
and this prompt may not touch a production file. Opened as
`[06-node-solve-comment-quotes-a-superseded-node-count]` on this campaign's board.

---

## 2. Entry — `temperature_crossing_log1pz`, the $T(z)$ crossing probe

| Field | Value |
|---|---|
| **Site** | **`CosmologyModels/tests/T_z_reference.py`**, `temperature_crossing_log1pz(cosmology, T, u_lo, u_hi)` (`def` at **`:229`**), the `root_scalar` call at **`:285`**. The reasoning is in that function's docstring, `:232-274`. **It was `LambdaCDM_GenericEOS._temperature_crossing_log1pz` until prompt 04 of this campaign** moved it out of the production class, character-identical in its `root_scalar` call; `AUDIT.md` §1 and `RECONCILIATION.md` §3 cite it at `:864` and `:869` of the old file and those anchors have no successor in production code. |
| **Value** | `xtol=1e-15`, `rtol=1e-15` — **unchanged by the move**, and unchanged since it was introduced. |
| **Method** | Brent, **bracketed** — the caller supplies `(u_lo, u_hi)` in $u = \log(1+z)$, and the function returns `None` rather than solving when the residual does not straddle (`:282-283`). |
| **What it sets** | **Nothing in production. Zero production calls.** Its one caller is `ComputeTargets/tests/test_numeric_break_points.py::test_hubble_jumps_at_the_declared_crossings_and_not_at_the_kink`, which uses it to find a *neighbourhood* of a crossing. **The trap is the point of the entry:** the residual it root-finds is $\log T_\gamma(z) - \log T$, and since `qcd-background-audit` prompt 06 the representation is segmented at exactly these temperatures, so $T_\gamma$ genuinely **jumps** there and the residual **need not have a root at all**. A bracketing solver applied to it reports `converged=True` and returns a non-root **whose offset depends on its tolerances**. Production locates these crossings with the hand-rolled geometric bisection `_bisect_temperature_crossing_log1pz` (`LambdaCDM_GenericEOS.py:814`), which is correct by construction and is **not** a `scipy` solve; `CosmologyModels/tests/test_T_z_representation.py::test_a_segment_edge_bisected_and_one_root_found_disagree` is the standing demonstration. |
| **Call count** | **0 per model construction and 0 per production run.** Three per invocation of the one test that uses it (`T_LO`, `EOS_T_LO`, `T_120_MEV`). |
| **Choosing measurement** | **The value has no choosing measurement, and this entry says so rather than inventing one.** It was introduced by `prompts/GkTk-remedial` **prompt 03** (`83ef7c5`, 2026-09-11), which extracted the crossing finder out of `integration_break_points` into its own method and recorded the pair as `xtol=rtol=1e-15` without a measurement behind it (log 03, "What shipped"). What *was* measured is the **behaviour** of a bracketing solve on this residual: `qcd-background-audit` prompt 01's case 7, re-taken by **prompt 06** (`a1d667a`, 2026-09-14, log 06): the returned point sits **+1.126e-12** in $u$ above the bisected edge at `root_scalar`'s defaults (317 ulp, and further than the 1e-12 by which each segment's nodes are held inside their own branch), **+3.304e-13** on a local bracket and **+1.421e-14** at `xtol = rtol = 1e-15` — **each reporting `converged=True` with a residual of +8.844e-06.** Prompt **07** (`c2bf596`, 2026-09-14) then took it off the production path entirely. **So the tolerance was never chosen against a criterion; what was established is that tightening it moves the answer without making it a root, and that the answer is not wanted in production.** |
| **Competing floor** | Not a floor but a **category error**: there is no root, so no tolerance converges to one. The offset from the true (bisected) edge is set by the tolerance and by the bracket, and the measurements above span two orders across three reasonable choices. The quantity the test actually needs — a neighbourhood of the crossing — is undisturbed by an offset of ~1e-14 in $u$, which is why the function survives. |
| **Cost** | Three solves in one test method, once per `ComputeTargets` suite run. **Zero in the pipeline.** |
| **Citation** | Value: `prompts/GkTk-remedial` prompt 03, [`logs/03-tau-primitive.md`](../GkTk-remedial/logs/03-tau-primitive.md), `83ef7c5`. Behaviour and the trap: `prompts/qcd-background-audit` prompts 01, 06 and 07, [`logs/06-segment-at-the-jumps.md`](../qcd-background-audit/logs/06-segment-at-the-jumps.md) and [`logs/07-rederive-break-points.md`](../qcd-background-audit/logs/07-rederive-break-points.md). Relocation: `prompts/background-solver-robustness` prompt 04, [`logs/04-relocate-the-crossing-probe.md`](logs/04-relocate-the-crossing-probe.md), which closed `[08-temperature-crossing-solver-is-test-only]`. |

**For `prompts/tolerance-convergence` prompt 02.** This site is **no longer a production accuracy
parameter** and should be inventoried as test machinery, or omitted. Its old anchor
`LambdaCDM_GenericEOS.py:864` is in that campaign's README §3.2 list and is now wrong in two ways:
the line moved, and then the method left the file.

---

## 3. Entry — `_find_rho_equality`, the equality redshifts — **this campaign's own**

| Field | Value |
|---|---|
| **Site** | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, `LambdaCDM_GenericEOS._find_rho_equality` (`def` at **`:1001`**), the `root_scalar` call at **`:1137`**. The reasoning is in the code at **`:1103-1136`**, to the standard of `:626-635`. `AUDIT.md` cites it at `:1008` and `RECONCILIATION.md` at `:1013`; both are correct for the trees they were taken on. |
| **Value** | `xtol=1e-300`, `rtol=8.9e-16` — **Brent's own $4\varepsilon = 8.881784\times10^{-16}$ floor**, below which `scipy` 1.15.2 raises. |
| **Method** | Brent, **bracketed since prompt 02 of this campaign**. Before it: an unbracketed **secant** (`x0=`, no `bracket=`) at `xtol=1e-6, rtol=1e-4`. The bracket is expanded about the caller's guess **multiplicatively in $1+z$** by `BRACKET_EXPANSION_FACTOR = sqrt(2.0)`, capped at `BRACKET_EXPANSION_MAX_STEPS = 140` and **clamped at both ends, and at the guess itself, to the $T(z)$ representation's own tabulated bounds**, until the residual changes sign. The straddle test compares **signs**, not the product `f_lo * f_hi <= 0` that `_solve_T_z:621` uses, because this residual reaches ~$10^{183}$ at the top of the tabulated range at the default `max_z = 1e20` and the product overflows to `+inf` (log 02, D4). |
| **What it sets** | **Diagnostic** — the two values are printed at `:536-539` with `:.4g`. **But read §3.1 before acting on that word: since prompt 09 they are also what the model's two `BaseCosmology` equality properties return, and on a cosmology that declares break points that is a datastore lookup key.** |
| **Call count** | **Twice per model construction** (`__init__`, `:521` and `:524`), once per model, per run. |
| **Choosing measurement** | `prompts/background-solver-robustness` **prompt 02** (2026-09-16, measured at `7fdc49b`, shipped at `921f41c`), log [`logs/02-bracket-the-equality-solve.md`](logs/02-bracket-the-equality-solve.md) D1. The campaign's own recommendation (README §7 **D1**) was `rtol=1e-14`, `_solve_T_z`'s value, and **the measurement rejected it**: `rtol=1e-14` is 75 ulp of slack at $z\sim3.4\times10^3$, Brent stops early inside it, and the solve landed **7 ulp** from prompt 01's independent bracketed reference — **failing a test prompt 01 had already shipped**. The full sweep (three expansion factors × three tolerances, log 02 D1) shows every `rtol=1e-14` row at 7 to 11 ulp and every floor row at 0 to 1. At `rtol=8.9e-16` the four production roots sit **0 / −2 / 0 / −2 ulp** from that reference, and the two that moved moved **onto** it. **The user took the decision on 2026-09-16**, amending README §7 D1 and prompt 02 §4's acceptance row from "bit-identical / ≤ 1 ulp" — unattainable, see the next field — to "≤ 4 ulp against prompt 01's reference". |
| **Competing floor** | Two, and they are the reason the pair is what it is. **(i) The residual's own quantisation.** `match_rho` is a cancellation between two densities of order $10^{112}$; near the root the difference is quantised at ~$7\times10^{100}$, about 0.55 quanta per ulp of $z$ with ±1–2 quanta of evaluation noise. Stepping float by float on `QCD_Cosmology` at matter–radiation equality, the residual is **exactly zero one ulp above the closed form, non-zero either side of it, and changes sign six to seven ulp higher**; on the matter–$\Lambda$ pair it is **exactly zero across five consecutive floats**. **The root is a band a few ulp wide, so bit-identity between two solvers is not a property this problem has**, and where a solver stops inside the band is set by `rtol`. **(ii) Brent's $4\varepsilon$.** `rtol=8.9e-16` *is* that floor; there is nothing tighter to ask for. Separately, `xtol=1e-300` disables the absolute component deliberately, for `_solve_T_z`'s reason at `:626-635` restated for this method's own geometry: it serves **two roots four decades apart** ($z\sim3.4\times10^3$ and $z\sim0.30$), so any finite absolute tolerance in $z$ is meaningless at the first and is the only thing acting at the second. The shipped `xtol=1e-6, rtol=1e-4` did both at once and **neither bounded the error nor predicted it** (`AUDIT.md` §3.1). |
| **Cost** | Measured, not predicted: the four production call sites go from **3, 1, 1, 1** evaluations of `match_rho` to **23, 25, 21, 25** — **+20 to +24 each, twice per model construction**, each one spline evaluation. Model construction is **49.6 ms → 46.7 ms** (mean of 7), i.e. inside the run-to-run scatter and unmeasurable against the ~3,176-node tabulation build. `AUDIT.md` §3.1's "+6 to +9" is the cost of *tightening the secant* and **does not survive bracketing**, which buys the two bracket endpoints and Brent's bisection steps as well; the comment at `:1121-1125` says so. |
| **Citation** | `prompts/background-solver-robustness`, prompt 02, [`logs/02-bracket-the-equality-solve.md`](logs/02-bracket-the-equality-solve.md); characterised by prompt 01, [`logs/01-equality-solve-characterisation.md`](logs/01-equality-solve-characterisation.md); pinned by `CosmologyModels/tests/test_rho_equality.py`. It closed `[00-equality-solve-is-unbracketed-and-loose]`. Source: [`AUDIT.md`](AUDIT.md) §2, §3, §4. |

### 3.1 "Diagnostic" is true of the solve and false of the quantity — read this before reusing the word

**This is the mistake this campaign was created to stop being made twice.**
[`AUDIT.md`](AUDIT.md) §2.1 established, from a correct repository-wide grep, that
`_find_rho_equality`'s two results land in locals, are printed and are discarded, and concluded
that *"the blast radius of this solve is two banner lines"*. Every clause of that was true.
[`RECONCILIATION.md`](RECONCILIATION.md) §5 found that the **quantity** was nevertheless a
production sample location inside a datastore identity, reached by a **duplicated closed form** in
`main.py` rather than by this method — which is exactly why the grep found nothing. Prompt 03 of
this campaign measured the whole chain (`logs/03-equality-redshift-consumers.md` §3) and prompt 09
removed the duplication: `main.py` now asks the cosmology and computes nothing, so **the value this
solve returns is `feature_z[0]` on `QCD_Cosmology`**, is forced into the production source grid,
and reaches the `BackgroundModel` lookup key through the grid's content digest.

**So, at this commit, the accurate statement of the "what it sets" field is:**

- the **printed banner** is a diagnostic;
- the **returned value** is `LambdaCDM_GenericEOS.z_matter_radiation_equality` /
  `z_matter_lambda_equality`, which `main.py`'s `cosmology_feature_redshifts` reads, and which
  therefore **keys stored objects** on any cosmology that declares break points — today
  `QCD_Cosmology` alone.

Prompt 09 landed that on purpose, on the user's README §7 **D2** decision of 2026-09-16, and it
moved the `QCD_Cosmology` production source-grid digest **`a2c32f67` → `4849552b`** on exactly one
sample of 1,996 (index 1540, +7 ulp). **A later reader who tightens or loosens `rtol` here is
moving a lookup key**, not a banner line. That is the opposite of what `AUDIT.md` §2.1 would lead
them to believe, and it is why this entry carries the pointer rather than the single word.

---

## 4. The three at a glance

| | `_solve_T_z` | `temperature_crossing_log1pz` | `_find_rho_equality` |
|---|---|---|---|
| Site at this commit | `LambdaCDM_GenericEOS.py:636` | `tests/T_z_reference.py:285` | `LambdaCDM_GenericEOS.py:1137` |
| Value | `xtol=1e-300, rtol=1e-14` | `xtol=1e-15, rtol=1e-15` | `xtol=1e-300, rtol=8.9e-16` |
| Bracketed | yes | yes | **yes, since prompt 02** |
| Production calls per model | ~3,176 | **0** | **2** |
| In a lookup key | via `T_Z_REPRESENTATION_VERSION` | no | **yes, via the source-grid digest** (§3.1) |
| Chosen by | `qcd-background-audit` 04 (`71b842a`) | **nobody** — introduced by `GkTk-remedial` 03 (`83ef7c5`) | `background-solver-robustness` 02, user decision (README §7 D1, amended) |
| Competing floor | Brent's $4\varepsilon$; the interpolation | there is no root | Brent's $4\varepsilon$; the residual's few-ulp band |

**Every accuracy parameter that `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` owned when
this campaign opened now has an entry.** One of the three — solve 2's — records that the value was
never chosen against a criterion, in those words, as `prompts/tolerance-convergence` README §1.2
requires of a provenance that cannot be established; it is mitigated by that site having no
production caller at all.
