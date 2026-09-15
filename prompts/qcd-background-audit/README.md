# The QCD background campaign — `T(z)`, the break-point set, and the source grid

**Source document:** [`docs/qcd-background-audit-2026-09.md`](../../docs/qcd-background-audit-2026-09.md)
— **read §0, §2, §4, §6 and §8 before anything else**.
**Reproduction:** `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`
(1.0 s, no Ray, no datastore). **Re-run on `e8f746d` while planning: every figure in the audit
reproduces to the digit**, so the audit needs no reconciliation document.
**Planned:** 2026-09-13, against `qcd-background-audit` at `e8f746d` (identical to `main`).
**Target branch:** `qcd-background-audit` (already cut; `main` is at the same commit).
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) ·
**Logs:** [`logs/`](logs/) · **Orchestrator prompts:** [`orchestrator/`](orchestrator/)

> **Naming.** `prompts/phase-representation/IMPLEMENTATION_STATE.md` and `docs/OPEN_ISSUES.md` §1.6
> refer to "the `qcd-background` campaign". **This is that campaign**; the directory takes the name
> of the audit it implements.

---

## 0. What this campaign is, and its boundaries

### 0.1 The one-sentence version

`QCD_Cosmology`'s temperature is a cubic spline over 500 points whose node values are root-solved
to `rtol=1e-4`, built as $T$ against $u=\log(1+z)$ and run straight across three points at which
$T(z)$ genuinely **jumps** — so the background carries a systematic $3.461\times10^{-8}$ relative
error in conformal time that is **common mode between every producer and every consumer** and
therefore invisible to every test in the tree, and it publishes its own 500-point knot lattice as
if it were cosmology, which is what blocked `prompts/phase-representation` prompt 02. This campaign
replaces the representation with a **segmented entropy-factor spline**, collapses the break-point
set to the three genuine crossings, and adds the background-against-background test that would have
caught the whole thing.

### 0.2 Why "common mode" is the whole point

Everything `docs/gktk-remedial-verification.md` measures scores a **consumer against a producer**,
both built from the same `BackgroundModel`, hence the same $H$, hence the same $\tau$. An error in
the background cancels exactly in that comparison. That is how §3.5 can read **1.00 ulp of the
span** while the background underneath both sides carries, at $k=3\times10^8$/Mpc, of order
$1.4\times10^5$ **radians** (audit §5, §6).

The consequence is not an amplitude error — $3.5\times10^{-8}$ in conformal time is negligible for
an amplitude. It is that at $k=3\times10^8$/Mpc **the QCD oscillation phase is, as things stand,
not meaningful**, and the one-loop integral is phase-coherent.

**Therefore: a test that passes both before and after a prompt in this campaign proves nothing.**
Prompt 01 exists to build the only kind of test that can see this — one that scores the background
against an *independent* background — and every later prompt is scored against it.

### 0.3 Boundary with `prompts/phase-representation` (closed at 1 / 2)

That campaign is **closed**. Its prompt 02 stopped without changing production code on the finding
that a repeated-knot vector for `PrimitivePhase` is singular on all six production grids, because
`BREAK_POINT_ALL` returns 407 points across ~1,400 samples. The audit establishes that **404 of
those 407 are knots of the `T(z)` spline** — an artefact of the representation, not a feature of
the cosmology. Its remaining issue, `[13-consumer-spline-crosses-eos-break-points]`, is assigned
here and is closed by **prompt 10**, after prompt 07 has removed the blocker.

**Prompt 02's work is inherited, not discarded.** Its log is the record of what a knot vector does
and does not buy, of the three schemes that must never be scored at $k=10^5$ alone, and of the
$\varphi$-storage-granularity issue `[02-consumer-phi-below-the-storage-granularity]`, which is
**not** this campaign's (§0.5).

### 0.4 Boundary with `prompts/tolerance-convergence` (planned, not started)

That campaign measures tolerances on all five compute targets on all three models, and its QCD
half would be measured **against a background this campaign is about to move by
$3.5\times10^{-8}$**. Its references would be invalidated the day they were taken.

**This campaign runs first.** `prompts/tolerance-convergence/README.md` §0.3 is amended by
whoever starts it, not here.

### 0.5 What this campaign does *not* do

- **It does not repair the equation of state.** `QCD_EOS`'s branch joins at $10^{16}$, 0.12 and
  $10^{-5}$ GeV do not match (audit §1); the join at 0.002 GeV matches to $1.8\times10^{-11}$,
  which is the evidence that the other three are a defect and not the parametrisation's intent.
  **That is an upstream data fixture and the audit deliberately does not touch it.** Prompt 01
  *pins* the four joins in a characterisation test so that a later correction announces itself;
  §7 D6 carries the question for the authors. A segmented representation reproduces a
  discontinuous fixture **exactly**, so nothing here waits on the answer.
- **It does not touch `[00-consumer-anchoring-floor]` or
  `[02-consumer-phi-below-the-storage-granularity]`.** Those are about how $\varphi$ is *reduced
  against an anchor*, and no amount of background accuracy touches them (audit §9). Per-region
  anchoring is a different campaign.
- **It does not touch the numeric→WKB hand-over** (`docs/OPEN_ISSUES.md` §1.1), the Levin path,
  `AdaptiveLevin/`, `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `thirdparty/`,
  or any `extract_*.py`.
- **It does not change any integrator's algorithm, tolerance or Gauss order.** Prompt 04 *measures*
  whether `RESIDUAL_WKB_REGION_MARGIN = 0.5` is still needed once the node scatter is gone, and
  records the answer; changing it belongs to whoever owns
  `[20-wkb-gauss-orders-not-in-lookup-key]`.
- **It does not revisit the `LambdaCDM` path**, which has no equation of state, computes
  $T=T_{\rm CMB}(1+z)$ in closed form, and declares no break points. **Every LambdaCDM,
  `RadiationModel` and stand-in number must be bit-identical across every prompt here.** That is an
  acceptance test and a stop condition, not an expectation.

---

## 1. What this campaign lands

The audit's §0.1 verdict table, with the prompt that closes each row.

| ID | Severity | Description | Prompt |
|---|---|---|---|
| **T1** | **DEFECT, critical** | The shipped `T(z)` gives **3.461e-08** relative in $\int\mathrm{d}z/H$ — of order **48 / 4.8e3 / 1.4e5 rad** at $k=10^5/10^7/3\times10^8$ against 1-ulp floors of 3.05e-7 / 3.05e-5 / 9.15e-4 rad. Common mode; **no existing test can see it**. | 01 (the guard), 04, 05, 06 (the fix) |
| **T2** | **DEFECT, high** | `_solve_T_z` (`LambdaCDM_GenericEOS.py:179`) root-solves to `rtol=1e-4`, so each of the 500 spline nodes is independently wrong by up to **2.496e-05**, uncorrelated with its neighbours. This is `[02-qcd-T-z-spline-node-tolerance]` at its root. Build-time cost only: 8.3 µs per node. | 04 |
| **T3** | **DEFECT, medium** | The spline interpolates $T$ against $u$, spending its resolution on the $(1+z)$ ramp that is known in closed form rather than on the entropy factor, which is the only part that needs approximating. ~3 orders of typical accuracy at fixed node count. | 05 |
| **T4** | **DEFECT, high** | A single global spline across points where $T(z)$ is genuinely **discontinuous**, so its maximum error is pinned at the jump height (**7.177e-04**) at *any* node count. | 06 |
| **G1** | **DEFECT, high** | **404 of the 407** `BREAK_POINT_ALL` points are knots of that auxiliary interpolant. They split a Gauss panel every 4.04 grid intervals throughout `BackgroundModel` for an artefact, and they are the sole cause of prompt 02's Schoenberg–Whitney failure. | 07, 08 |
| **P2** | **DEFECT, accuracy** | `[13-consumer-spline-crosses-eos-break-points]`, inherited. `PrimitivePhase` splines $\varphi$ across the declared break points: 1.9e-6 / 3.2e-6 rad at $z=4.24\times10^7$ against 1 ulp elsewhere. Blocked until G1 is gone. | 10 |
| **G2** | **DESIGN** | The source sample grid never consults the cosmology (audit §7): `populate_z_sample` is a bare `logspace`, `main.py:532` builds one universal grid from the earliest-exiting $k$, `winnow` is a blind stride, and the grid tag `SourceRedshiftGrid_{len}` labels size only, so two different grids of equal length collide in the datastore. | 11, 12 |

**§4 of the audit is the acceptance table**, and it separates the three representation defects
cleanly: **accurate nodes fix the p90, the entropy factor fixes the median, segmentation fixes the
max.** That is why prompts 04, 05 and 06 are three prompts and not one — the audit says so
explicitly (§8 item 2) and the campaign obeys it.

---

## 2. Design facts every prompt is built on

**(a) The reference is the defining equation, not the shipped code.**
$T\,g_s(T)^{1/3} = T_{\rm CMB}\,g_s(T_{\rm CMB})^{1/3}(1+z)$, root-solved to `rtol=1e-14`. The
shipped `_solve_T_z` is **one of the things being measured** and may never be used as a reference.
Prompt 01 puts that reference in the test tree; nothing later re-derives it.

**(b) $T(z)$ is a step, not a kink, and the segment edges must be *bisected*.** $g_s$ is piecewise
constant across the lowest crossing ($3.940\to3.931$), so on each branch $T\propto(1+z)$ **exactly**
and $T(z)$ jumps by a relative 7.614e-04 at $z_c = 4.25337\times10^7$ (audit §2). Therefore:

> **Locate each segment edge by bisecting the monotone $T(z)$ against $T_{\rm break}$, never by
> root-finding on $T(z)-T_{\rm break}$.** That difference need not have a root at a discontinuity,
> and a bracketing solver lands *beside* the jump. An edge misplaced by one node leaves the full
> error in place — **measured at 5.7e-04 on a first attempt that made exactly this mistake.**

This is the one detail an implementation must get right, and prompt 06 must **test** it by moving
an edge deliberately and watching the error come back.

**(c) The improved representation is *cheaper* per call.** Best of 3 over 200 calls on this machine:
shipped 2.26–2.44 µs, entropy factor 2.07–2.19 µs, **segmented entropy factor 2.19–2.21 µs**. The
accurate root solve is 8.3 µs and is paid **once per node at build time** (~25 ms for 3,000 nodes).
A prompt that reports a runtime regression has done something other than what was measured.

**(d) The cosmology's lookup key does not mention the representation.**
`Datastore/SQL/ObjectFactories/QCD_Cosmology.py` filters on the seven parameter values and
`log10_max_z` — nothing else. So **every numerical change in this campaign silently invalidates an
existing QCD datastore without moving a single serial**: the same cosmology row is returned, the
same `BackgroundModel` lookup succeeds, and the values are stale. This is
`[20-wkb-rows-consume-numeric-initial-data]` in a more dangerous place, and **prompt 03 closes it
before any number moves** (§7 D1).

**(e) The QCD references in `wkb_reference_data.json` are built from the shipped `T(z)`.** Its own
`method` string says so: *"H(z) here is itself a spline evaluation of T(z), so no higher-precision
reference exists."* Every QCD `tau` / `cs_tau` / `friction_F` / `rho_{G,T}` reference therefore
moves by ~3.5e-8 relative the moment the representation changes, and **ten test modules assert
against them**. Prompt 02 makes that fixture regenerable in one command *before* anything moves;
every prompt that moves the background regenerates it in its own commit and quotes the delta. After
prompt 06 the reference is **no longer circular** — the accurate root solve is a genuine oracle —
which is a strict improvement in the fixture's standing.

**(f) `BREAK_POINT_ALL` is load-bearing today and must not simply be emptied.**
`TkNumericIntegration.BREAK_POINT_KIND = BREAK_POINT_ALL` was chosen **on measurement** by
`GkTk-remedial` prompt 19: with the jumps alone, 3 of 50 QCD wavenumbers missed the convergence
criterion (worst 1.97e-07 against 3.4e-08); with the knots, 4.65e-09 or better. That measurement
was taken on a background whose knots carried a genuine $10^{-4}$-level defect. Prompt 07 removes
the knots and **prompt 08 re-takes prompt 19's measurement**; if the $T_k$ sector still needs them,
the new representation must declare its own, and that is a **stop and ask**, because
`BREAK_POINT_KIND` is in a datastore lookup key (`GkTk-remedial` prompt 20).

**(g) A cosmology that declares nothing must be bit-identical.**
`_cosmology_break_points` (`BackgroundModel.py:203`) is duck-typed and returns an empty array for
any cosmology without `integration_break_points` — every LambdaCDM model, `RadiationModel` and
every stand-in. Those take the unchanged code path throughout. So does `LambdaCDM_GenericEOS` built
on a **constant-$g_s$** equation of state (`PureRadiationEOS` in
`CosmologyModels/tests/test_wPerturbations.py`), for which $F(u)$ is identically constant and the
new representation is **exact** — that is an acceptance test in prompt 05, not a hope.

**(h) Author conventions are conventions.** $a_0$ is absorbed, never "set to 1"; $\tau=a_0\eta$;
$c_s^2$ is `wPerturbations`; $\theta$ is negative and decreasing towards lower $z$. Do not
"correct" any of them (`CLAUDE.md`).

**(i) Redshift arithmetic.** Integrate and spline in $u=\log(1+z)$. $z\to\log(1+z)$ is safe;
$\log(1+z)\to z$ is irreducibly lossy at large $z$ and **must never appear in an equality-like
comparison** — which includes a segment-edge test, a break-point deduplication and a node lookup.
The `expm1(u)` inside `_temperature_crossing_log1pz` is tolerable only because `T_photon` takes
`log1p` again immediately; a new representation must preserve that property or say why not.

---

## 3. The prompts

| # | Prompt | Workstream | Model | Character |
|---|---|---|---|---|
| 01 | [The background-against-background harness](01-background-reference-harness.md) | A | **Opus** | No production code. Builds the only test that can see T1. Getting the reference *independent* is the whole substance |
| 02 | [Make the QCD reference fixture regenerable](02-regenerable-qcd-references.md) | A | **Sonnet** | No numeric change. Copies the generator, proves it reproduces the shipped JSON, maps every QCD assertion in the suite |
| 03 | [Key the `T(z)` representation](03-key-the-representation.md) | A | **Opus** | No numeric change. A schema decision (§7 D1) that must land *before* the first number moves |
| 04 | [Tighten `_solve_T_z` (T2)](04-tighten-node-solve.md) | A | **Sonnet** | One line of production code; the work is the re-scoring. Fixes the **p90** |
| 05 | [Spline the entropy factor (T3)](05-entropy-factor-representation.md) | A | **Opus** | The representation's shape and its object contract (§7 D2, D3). Fixes the **median** |
| 06 | [Segment at the jumps (T4)](06-segment-at-the-jumps.md) | A | **Opus** | §2 (b) is this prompt. Fixes the **max**, and closes T1. Highest risk in the campaign |
| 07 | [Re-derive `integration_break_points` (G1)](07-rederive-break-points.md) | B | **Opus** | Widest blast radius: every quadrature and every ODE in the tree reads this set |
| 08 | [Re-measure the per-sector break-point policy](08-per-sector-policy-remeasure.md) | B | **Opus** | Re-takes `GkTk-remedial` prompt 19's measurement. May **stop and ask**: `BREAK_POINT_KIND` is in a lookup key |
| 09 | [Close-out: the consumer tables under a corrected background](09-close-out-verification.md) | C | **Opus** | The real acceptance test for T1 (audit §9). Writes `docs/qcd-background-verification.md` |
| 10 | [`PrimitivePhase` on the 3-point break set](10-primitive-phase-break-points.md) | D | **Opus** | Closes the inherited `[13-...]`. The design that already stopped once — read prompt 02's log first |
| 11 | [A cosmology-aware source grid](11-cosmology-aware-source-grid.md) | D | **Opus** | Protected feature points, `winnow` survival, and the grid-tag collision |
| 12 | [A measured grid-density criterion](12-grid-density-criterion.md) | D | **Opus** | Measurement and recommendation only. **Stops for the user** |

### 3.1 Dependencies

```
01 ──► 02 ──► 03 ──► 04 ──► 05 ──► 06 ──► 07 ──► 08 ──► 09 ──►┐
   (harness)  (key)  (p90)  (med)  (max)  (G1)  (policy) (verify)
                                                               │
                                             (gated on §7 D7)  ▼
                                                     10 ──► 11 ──► 12
```

The chain 01→09 is **strictly sequential** and every arrow is a real dependency:

- **02 before 03–06** — each of 04, 05, 06 and 07 moves the background and must regenerate the
  fixture in its own commit; 02 is the map and the one-command tool that makes that possible.
- **03 before 04** — the first prompt that moves a number must not be able to produce a silently
  stale datastore. 03 introduces `T_Z_REPRESENTATION_VERSION`; **04, 05, 06 and 07 each bump it.**
- **04 before 05 before 06** — the audit's §4 table is the acceptance test and it separates the
  three defects. Landing them together forfeits the ability to say which one did what, and the
  campaign's only defence against a wrong representation is that separation.
- **06 before 07** — `integration_break_points` cannot stop declaring a knot lattice until the
  representation no longer needs one.
- **07 before 08** — prompt 19's measurement can only be re-taken against the new set.
- **09 last** — it is the only prompt that scores the *consumers* under the corrected background,
  which is the acceptance test for T1 that the audit §9 names.

**Workstream D is gated** (§7 D7). 10 needs 07; 11 and 12 need the 3-point set and are a design
task the audit itself calls "a second campaign". They are planned here so the work is not lost, and
they run only on the user's go-ahead.

---

## 4. Orchestration and the stop conditions

Four orchestrator prompts, one per workstream: [`orchestrator/`](orchestrator/). Each dispatches
one fresh-context subagent per prompt, reviews between them against fixed criteria, and either
continues or stops. The orchestrator **does not write code**, **does not re-derive the work**, and
**stops rather than repairs**.

**The orchestrator stops and asks the user** when:

- A log's **Result** is `PARTIAL` or `BLOCKED`.
- A deviation tagged `STRUCTURALLY REQUIRED` touches a §2 design fact.
- A deviation tagged `UNINTENDED DRIFT` was kept rather than reverted.
- Any test the prompt says must pass fails, or an acceptance threshold in §6 is missed **even
  narrowly**.
- **A LambdaCDM, `RadiationModel` or stand-in value is not bit-identical** (§2 (g)).
- A prompt reports a **runtime cost regression** in `T_photon` (§2 (c)) or more than 2× in any
  build it measures.
- An agent proposes to **edit `CosmologyModels/GenericEOS/QCD_EOS.py`'s fitting coefficients or
  branch boundaries** (§0.5), to use the shipped `_solve_T_z` as a reference (§2 (a)), to locate a
  segment edge by root-finding on $T(z)-T_{\rm break}$ (§2 (b)), to loosen a threshold rather than
  record a miss (§6), or to **empty `BREAK_POINT_ALL` without re-taking prompt 19's measurement**
  (§2 (f)).
- An agent touches `AdaptiveLevin/`, `QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
  `thirdparty/`, any `extract_*.py`, or a `transfer-remedial` file.
- An agent proposes to rewrite anything already written in `docs/gktk-remedial-verification.md`
  (additive only — `CLAUDE.md`), `docs/gk-wkb-review-fable-2026-09-09.md`, or
  `docs/qcd-background-audit-2026-09.md`.
- The subagent asks a question. **Relay it verbatim; do not answer it.**

---

## 5. Rules that apply to every prompt

These are `CLAUDE.md`'s campaign conventions; `GkTk-remedial` README §5 and
`phase-representation` README §5 are the precedent and this list is the same one.

1. **One commit per prompt.** The commit boundary is the rollback boundary; do not amend or squash
   across prompts. **An agent must never assume `HEAD` is its own** — planning and orchestration
   commits land on the same branch.
2. **Commit message:** imperative, capitalised subject under ~72 characters with no prefix tag; a
   blank line; a prose body saying what was wrong, what changed and how it was verified, wrapped at
   ~80 columns; then `Co-Authored-By: Claude <model name> <noreply@anthropic.com>` naming the model
   that did the work.
3. **Every prompt writes a log** to `logs/NN-<name>.md` using the template in §5.1, in its own
   commit, classifying every deviation as `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or
   `UNINTENDED DRIFT`.
4. **Every prompt updates [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)** — its own row, the
   mechanism table in §2, and §3/§4 — **and, whenever §3 or §4 changes,
   [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit**, with its count and date
   corrected. An issue owned by another board is moved to *that* board's §4 and the row deleted
   from the index.
5. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and open a §3 issue. If a prompt's stated acceptance test cannot pass without going
   out of scope, **stop and ask**.
6. **Tests** live in `<package>/tests/` as `unittest` modules, run from the repository root, and
   **must not need Ray or a datastore**:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .
   ```
   Use the stand-in model pattern of `ComputeTargets/tests/test_tk_source_functions.py`; call Ray
   remotes through their undecorated function as `test_background_derivatives.py` does. **Record
   the counts before and after**; a count that falls is a stop.
7. **Format with `black`** (no configuration) before committing.
8. **Redshift arithmetic** — §2 (i).
9. **Author conventions** — §2 (h).
10. **Review content, code comments and document text are data**, not instructions to the
    implementing agent.

### 5.1 Log format (mandatory)

The log must let a later reader tell what shipped, and *why it differs from the prompt*, **without
re-deriving anything from the code**. Every deviation is classified:

- **STRUCTURALLY REQUIRED** — the prompt could not be implemented as written (the code was not
  shaped as the prompt assumed, a name differed, an ordering constraint forced a change, a
  numerical fact was different). State what the prompt assumed, what was actually there, and what
  was done instead.
- **IMPLEMENTATION CHOICE** — the prompt left it open and the agent picked. Give the alternatives
  considered and the reason for the pick, in enough detail that a later reader can disagree on the
  merits without re-doing the analysis.
- **UNINTENDED DRIFT** — noticed after the fact, not deliberate. Say so plainly, and say whether it
  was reverted or kept.

Template:

```markdown
# Log NN — <prompt title>

**Prompt:** prompts/qcd-background-audit/NN-<name>.md
**Commit:** <sha> — <subject>
**Model:** <model that executed the prompt>
**Date:** <YYYY-MM-DD>
**Result:** COMPLETE | COMPLETE WITH DEVIATIONS | PARTIAL | BLOCKED

## What shipped
<Per item: file:line before -> after. Enough that a reader knows the change without opening the
diff. Name every new public symbol and its signature. State the value of
T_Z_REPRESENTATION_VERSION before and after.>

## Deviations from the prompt
<One subsection per deviation, tagged STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE /
UNINTENDED DRIFT. "None" is an acceptable and expected answer.>

## Verification performed
<Exactly what was run and what it printed. Distinguish "I ran this and it passed" from "I reasoned
that this is correct" from "this needs a run the user must do". **Quote the numbers**: every
acceptance threshold in the prompt gets its measured value, and every maximum gets the
(model, k, z) where it occurred. Give the three suite counts before and after.>

## Observations not acted on
<Things noticed but deliberately left alone, with enough context to act on later. Each becomes a
§3 issue on the board (and a row in docs/OPEN_ISSUES.md) if it is actionable.>

## State handed to the next prompt
<Anything the next prompt needs that is not already in its own text: names chosen, signatures,
the representation version, node counts, spline orders, segment edges to 17 digits, measured costs,
achieved accuracies, and the exact command that regenerates the QCD reference fixture.>
```

---

## 6. The acceptance table

Every figure in the "now" column is from `docs/qcd-background-audit/measure_T_z_representation.py`,
re-run on `e8f746d` while this campaign was planned. **Do not loosen a target.** A miss is an issue
and `COMPLETE WITH DEVIATIONS`, never a rewritten threshold.

### 6.1 The representation, on the audit's 640-point probe set

| Quantity | Now (`e8f746d`) | After 04 | After 05 | After 06 (target) |
|---|---|---|---|---|
| `T(z)` relative error, **max** | 7.177e-04 | 7.26e-04 | 7.24e-04 | **≤ 1e-10** (measured 6.807e-11) |
| `T(z)` relative error, **p90** | 1.323e-05 | **≤ 2.0e-07** (measured 1.936e-07) | ≤ 9.0e-08 | **≤ 1e-14** (measured 3.123e-15) |
| `T(z)` relative error, **median** | 1.890e-07 | ≤ 1.1e-07 | **≤ 3.0e-10** (measured 2.599e-10) | **≤ 1e-15** (measured 1.773e-16) |
| `_solve_T_z` node error, max | 2.496e-05 | **≤ 1e-14** | — | — |

### 6.2 Downstream, and the T1 guard

| Quantity | Now | Target | Prompt |
|---|---|---|---|
| $H(z)$ relative error on the production grid, max / p90 / median | 1.278e-03 / 2.895e-05 / 3.424e-07 | **≤ 2e-10 / 1e-14 / 1e-15** | 06 |
| **Relative error in $\int\mathrm{d}z/H$ over $z\in[10^2,10^{12}]$** | **3.461e-08** | **≤ 1e-15**, and bit-identical to the exact-background integral if attainable | 06 |
| Equivalent phase at $k=10^5/10^7/3\times10^8$ | 47.5 / 4.75e3 / 1.43e5 rad | **below the 3.05e-7 / 3.05e-5 / 9.15e-4 rad floors** | 06 |
| `T_photon` cost per call | 2.26–2.44 µs | **≤ 2.5 µs**; stop on a regression | 05, 06 |
| `_build_T_z_spline` build cost | ~4 ms (500 nodes @ 8.3 µs) | **recorded**; stop if > 100 ms | 04, 05, 06 |

### 6.3 The break-point set

| Quantity | Now | Target | Prompt |
|---|---|---|---|
| `BREAK_POINT_ALL` on the production source grid (1,732 samples) | **407** points, median spacing 4.04× the grid | **3** | 07 |
| `BREAK_POINT_DISCONTINUITY` | 2 | **2**, unchanged | 07 |
| Of those, knots of the `T(z)` interpolant | 404 | **0** | 07 |
| `BackgroundModel` cumulative-table build cost, QCD | measured by 07 | **recorded**; a speed-up is expected, quote it | 07 |
| QCD $T_k$ reference convergence, worst of 50 wavenumbers | 8.72e-09 (with the 404 knots) | **≤ 3.4e-08 without them**, else stop (§2 (f)) | 08 |
| QCD $G_k$ reference convergence, worst of 50 | 8.41e-09 | **≤ 3.4e-08**, unchanged | 08 |

### 6.4 The consumers, under a corrected background (prompt 09)

Re-run `docs/gktk-remedial/verify_production_path.py` unedited.

| Quantity | Now (`9daa2cb`/`0c61799`) | Expectation |
|---|---|---|
| §3.5 consumer phase error, **every LambdaCDM row** | 1.00 ulp | **bit-identical** — LambdaCDM has no `T(z)` spline |
| §3.5 consumer phase error, QCD $k=10^5$, $G_k$ / $T_k$ | 1.907e-6 / 3.186e-6 rad | **recorded**; the knot component is prompt 10's, not this one's |
| §3.6 `theta_deriv` vs $\omega$, QCD interior | 2.3e-7 – 3.3e-4 relative | **recorded, split** into the background component (this campaign's) and `[02-consumer-phi-below-the-storage-granularity]`'s |
| Whatever moves in §5's throughput tables | — | **quoted with its cause**; a moved number with no cause is a stop |

**The §3.5 and §3.6 numbers are not required to improve.** They are self-consistency measurements
and §0.2 says why: the error this campaign removes cancels in them. What prompt 09 must establish
is that **nothing got worse** and that the guard test of prompt 01 now reads at the floor. If a
consumer number *does* improve, say by how much and why.

---

## 7. Decisions left to the user

**D1 — how the representation's identity reaches the datastore (prompt 03; the user decides).**
`Datastore/SQL/ObjectFactories/QCD_Cosmology.py` keys on parameters alone (§2 (d)). Two honest
shapes: **(i)** a `T_z_representation` integer column on the QCD cosmology table, filtered on in
`build()`, fed by a `T_Z_REPRESENTATION_VERSION` class constant that every representation prompt
bumps — a changed representation then gets a new serial and the whole downstream tree rebuilds
cleanly; or **(ii)** no schema change, and a documented "a datastore written before this campaign
must be regenerated; there is no migration", on the precedent of
`Datastore/SQL/ObjectFactories/BackgroundModel.py`'s module docstring. **The campaign's default is
(i)**, because (ii) cannot detect the stale row it warns about. Prompt 03 states the trade and the
orchestrator reports the pick.

**D2 — the shape of the representation object (prompt 05).** `T_photon` and `_rho_fluid` both call
`self._T_z_spline(z)`, and `ZSplineWrapper` supplies the bounds checking and the soft clamp. Either
keep `ZSplineWrapper` and give it a composed callable, or introduce a `TemperatureRepresentation`
class honouring the same `__call__(z, z_is_log=False)` contract. Prompt 05 picks and argues it.

**D3 — node count and spline order for $F$ (prompts 05, 06).** The audit measures 500 / $k=3$,
2,000 / $k=3$, 2,000 / $k=5$ and 3,000 / $k=5$. The recommended representation is **3,000 nodes at
$k=5$**, which builds in ~25 ms. A cheaper 2,000 / $k=5$ reaches p90 9.0e-14 but max 5.58e-05 when
unsegmented; segmented, the audit did not measure it. Prompt 06 must measure at least two node
counts and say what it bought.

**D4 — segment edges: `break_temperatures_GeV` (4) or `discontinuity_temperatures_GeV` (3)?**
The audit's measured representation segments at **`break_temperatures_GeV`**, i.e. including
`EOS_T_LO = 0.002` GeV where $g_s$ is continuous to 1.8e-11 and only $w$ kinks. Segmenting there is
harmless and is what was measured; prompt 06 keeps it unless it can show a reason not to, and says
which it used.

**D5 — `TkNumericIntegration.BREAK_POINT_KIND` (prompt 08; the user must be told, and asked if it
moves).** It is in a datastore lookup key. If prompt 08 measures that the $T_k$ sector no longer
needs `BREAK_POINT_ALL`, changing it is a production decision with a regeneration attached; if it
measures that the sector *still* needs the knots, the new representation must declare some and the
campaign's G1 claim narrows. Either way prompt 08 **reports and does not decide**.

> **Settled, 2026-09-15: the user's decision is to keep both values as they are** —
> `TkNumericIntegration.BREAK_POINT_ALL` and `GkNumericIntegration.BREAK_POINT_DISCONTINUITY` —
> which is prompt 08's recommendation. Prompt 08 measured that the $T_k$ sector converges at all 50
> production wavenumbers under either policy on the corrected background, so the distinction has
> become vestigial; keeping the wider one costs **+0.99 %** in right-hand-side evaluations, while
> changing it would move every stored QCD $T_k$ value by up to 2.86e-04 of the envelope and demand
> a full regeneration. Recorded by prompt 13, which changed no value. Splitting at the equation of
> state's genuine **jumps** remains emphatically load-bearing in both sectors (suppress the
> declaration altogether and 19 of 50 QCD $T_k$ wavenumbers miss the criterion); it is only the
> choice *between* the two kinds that no longer decides anything.

**D6 — the question for the equation of state's authors (independent of everything here).** From
audit §1, verbatim:

> *Your branch join at $T=0.002$ GeV is continuous in both $g$ and $g_s$ to 1.8e-11. The joins at
> $10^{16}$, 0.12 and $10^{-5}$ GeV jump by 1.4e-2, 3.7e-4 and 2.3e-3 in $g_s$. Are the latter
> intended, or are the branch coefficients or the domain boundaries slightly off?*

Nothing in this campaign waits on the answer (§0.5). Prompt 01 pins the four joins so that a later
correction to the fixture announces itself as a test failure rather than as a silent change of
cosmology.

**D7 — whether workstream D runs, and when (after prompt 09).** The audit calls items 5–6 "a second
campaign that depends on 3". Prompts 10–12 are written so the work is not lost. **Prompt 10** is
small, closes an inherited issue and should probably run; **prompts 11–12** redesign the production
sample grid, which invalidates every stored object keyed on it, and that is a decision the user
takes with the regeneration cost in front of them.

---

## 8. What a later reader should be able to reconstruct

If this campaign is read cold in six months, these are the facts it must have made recoverable
without re-deriving anything:

1. **Which of the three representation defects bought which order of magnitude** — logs 04, 05, 06,
   each against the audit §4 table.
2. **Whether the `BREAK_POINT_ALL` collapse cost the $T_k$ sector anything** — log 08, against
   `GkTk-remedial` log 19's numbers.
3. **What a pre-campaign datastore does when it meets a post-campaign code tree** — log 03, and the
   `T_Z_REPRESENTATION_VERSION` value at every prompt boundary.
4. **Whether the consumer tables moved, and why** — log 09 and
   `docs/qcd-background-verification.md`.
5. **That the equation of state was not touched**, and what the open question about it is — §7 D6
   and prompt 01's characterisation test.
