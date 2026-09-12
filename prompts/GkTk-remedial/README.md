# Gk/Tk WKB phase remedial campaign — the shared conformal-time primitive

**Source document:** [`docs/gk-wkb-review-fable-2026-09-09.md`](../../docs/gk-wkb-review-fable-2026-09-09.md)
(§§1–12 the measurements, §13 the planning addendum) — **read §0, §7, §12.7 and §13 first**.
**Reconciliation against the tree:** [`RECONCILIATION.md`](RECONCILIATION.md) — the review checked
against `9ff59d5`; **prefer it where the two disagree**.
**Planned:** 2026-09-10, against `main` at `9ff59d5` (the review inspected `f06f587`, an ancestor;
only `main.py` and `GkSourcePolicyData.py` changed in between, neither in a way that affects a
finding — `RECONCILIATION.md` §0).
**Target branch:** `gktk-remedial` (to be created from `main`; see §4.2 for the merge-order
question with the in-flight `transfer-remedial` branch).
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) ·
**Logs:** [`logs/`](logs/) · **Orchestrator prompts:** [`orchestrator/`](orchestrator/)

---

## 0. What this campaign is, and its boundaries

This is the campaign that [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.3 calls "the
$T_k$ / $G_k$ numerical-precision campaign": the one on whose expectation
`[07-phase-spline-chunking-precision]` was closed WONTFIX. It remediates **how the Liouville–Green
(WKB) phase of the tensor Green's function and of the transfer function is computed, stored and
consumed**, plus a short list of numeric-region items the same review found. It does not touch the
numeric→WKB hand-over, which is a separate campaign (§1.1).

### 0.1 The one-sentence version

The stored WKB phase is wrong by 14 rad at $k=10^5/{\rm Mpc}$ and by 7400 rad at
$k=3\times10^8/{\rm Mpc}$ at $z=0.1$ (review §4), costs up to 64 s per object (2.5 million
right-hand-side evaluations, review §4), and is then re-splined by its consumers with an error that
grows linearly with the accumulated phase (review §5); yet to $2.5\times10^{-7}$ rad on the
production background the phase is **$k$ times a $k$-independent function of redshift**
(review §6), so one Gauss–Legendre table of $\tau=\int dz/H$ per background model, built once in
0.02 s to the double-precision floor (review §7), replaces the two-stage ODE, the resets, the $Q$
variable, the per-object solves, the growing-phase splines and their chunking — for both sectors,
with a second table $\tau_s=\int c_s\,dz/H$ and a third $F=\tfrac32\int(1+c_s^2)\,dz/(1+z)$ for
$T_k$ (review §12.7).

### 0.2 Boundary with `prompts/transfer-remedial` (in flight)

`prompts/transfer-remedial` rebuilds `LiouvilleGreen/bessel_phase.py` — the Bessel *oracle*. It is
**in flight**: prompts 01–05 have landed on branch `claude/workstream-a-orchestrator-4982a3`
(`f71401d`…`f6cbb29`), not yet on `main`; prompts 06–09 remain. Its remaining prompts touch
`LiouvilleGreen/bessel_phase.py`, `LiouvilleGreen/three_bessel_integrals.py`,
`LiouvilleGreen/tests/test_bessel_phase.py`, `test_3bessel_analytic.py`, `test_three_bessel.py`,
`main.py`'s Bessel construction stage (`main.py:499-528` on `9ff59d5`),
`ComputeTargets/QuadSourceIntegral_debug.py`, `plot_besssel_phase.py`,
`docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py`,
`docs/lg-phase-and-handover-followup-2026-09.md`, and — **tolerance constants and comments only** —
`ComputeTargets/tests/test_tk_source_functions.py` and `ComputeTargets/tests/test_phase_groups.py`.

**This campaign must not touch any of those files**, with two exceptions that are stop conditions
rather than free edits (§4.2): prompt 10 must change the *stand-in fixtures* in the last two test
files, and prompt 11 edits two `main.py` hunks far from the Bessel stage.

The division of labour:

| | `prompts/transfer-remedial` | `prompts/GkTk-remedial` (here) |
|---|---|---|
| Owns | `LiouvilleGreen/bessel_*`, `three_bessel_integrals.py`, the Bessel stage of `main.py` | `ComputeTargets/BackgroundModel.py` and its factory, `Quadrature/integrators/WKB_phase_function.py`, `Quadrature/integrators/numeric_with_phase_cut.py`, `Quadrature/supervisors/numeric.py`, `LiouvilleGreen/WKBtools.py`, **`LiouvilleGreen/phase_spline.py`**, `LiouvilleGreen/integration_tools.py`, `ComputeTargets/{Gk,Tk}{WKB,Numeric}Integration.py`, `WKB_Gk.py`, `WKB_Tk.py`, `GkSourcePolicyData.py`, `TkSourceFunctions.py`, and the WKB/numeric stages of `main.py` |
| Meets the other at | `phase_spline` (it *leaves* `bessel_phase` in their prompt 05; on `main` `bessel_phase` still calls it with `chunk_logstep=125`) | prompt 08 keeps `phase_spline`'s signature so `bessel_phase` on `main` keeps working unchanged (§2 (g)) |

### 0.3 Boundary with the hand-over campaign (future)

The six hand-over issues in `docs/OPEN_ISSUES.md` §1.1 — where the numeric→WKB switch sits, the
overlap width, the clamp error, the LG derivative truncation at the hand-over — are **out of scope
here**, as the review's own header says ("everything below starts inside the WKB regime with
specified initial data"). Nothing in this campaign moves `z_exit_subh_e3`/`_e6`, the
$\sqrt{z_{e3}z_{e4}}$ source limit, `GkSourcePolicy`'s crossover selection or the
`find_phase_minimum` window. One item that *is* a hand-over decision is recorded and handed over,
not acted on: the transfer function's LG truncation floor of $\sim1.4\times10^{-4}$ of the envelope
at the production hand-over $x_T\approx15.5$ (review §12.4, §12.7) — issue
`[00-tk-lg-truncation-floor]`.

### 0.4 What this campaign does *not* do

- **Does not change** `Gk_omegaEff_sq`, `Tk_omegaEff_sq`, the LG amplitude
  $\sqrt{H_i/(H\omega)}$, the $(B,\delta)$ initial-data algebra, or the `GkWKBValue`/`TkWKBValue`
  column schemas. The review checked all of these by hand (§1, §12.1) and they are right. Prompt 05
  *refactors* the two frequency modules to expose the non-leading part of $\omega^2$ separately,
  without changing what `*_omegaEff_sq` returns.
- **Does not raise the LG order.** The physical floors (§2 (d)) are recorded so that nobody chases
  a numerical target below them; changing the representation to beat them is a separate decision.
- **Does not implement per-region anchoring** of the consumer phase (review §13.4). The global
  anchor reproduces the $\varepsilon k\tau$ floor — $9\times10^{-4}$ rad at $k=3\times10^8$ —
  which is three to four orders below today's consumer error and at the QCD physical floor. Opened
  as `[00-consumer-anchoring-floor]` for a follow-up.
- **Does not build $G$ from two homogeneous solutions via the Wronskian** (review §10.2 "an
  option, not a recommendation").
- **Does not replace `T=1,\,T'=0` by the super-horizon series** (review §12.5): a spec-level
  initial-data decision. Opened as `[00-tk-superhorizon-ic-series]`.
- **Does not** run the quintic-spline intermediate of review §13.2(c). It would be superseded
  within the same workstream by prompt 03; see `RECONCILIATION.md` §3.

---

## 1. What this campaign does

Thirteen prompts in six workstreams, each landing exactly one commit, each independently
revertible.

**A — measurement and prototypes (01, 02).** No production code changes. Independent references
(mpmath where the background is analytic, converged adaptive quadrature where it is a spline), a
prototype of the table-plus-interval-accessor design measured for accuracy *and throughput* on both
production models, and the one test the review says must precede any implementation: whether a
fixed-order Gauss rule converges for the residual across the QCD model's spline knots (§11, §12.7).

**B — the primitives (03, 04).** `BackgroundModel.compute_background` stops solving
$d\tau/dz=-1/H$ with RK45 and builds a Gauss–Legendre table on its own grid; `functions.tau`
becomes a callable object with a pointwise `tau(z)` and an interval `tau.delta(z_a, z_b)` accessor
over a **double-double node table**; the datastore gains the low-order limb. Then the two siblings
$\tau_s$ and $F$ by the same machinery. After 04 the validation oracles `compute_analytic_G/T` and
the η-limits in `QuadSourceIntegral` are at the floor instead of carrying ~2 rad at $k=10^5$
(review §13.2(b)), and the datastore must be regenerated.

**C — the producers (05, 06, 07).** A residual module for $\rho=\int C/(\omega+\omega_0)\,dz$;
then `WKB_phase_function` is rewritten to $\theta=-[k\,\Delta\tau+\Delta\rho]$ from the tables, the
`GkWKBIntegration` store path drops the no-op sign fix and the cycle rebase and stores $\theta+\delta$
exactly; then `TkWKBIntegration` follows with $\tau_s$, $\rho_T$ and $F$ from tables, and its
friction ODE goes.

**D — the consumers (08, 09, 10).** `phase_spline` loses its chunking (signature preserved); the
two consumers stop splining the growing phase and instead evaluate
$\theta(z_s)=-k\,\Delta\tau(z_s\to z_r)+\varphi(z_s)$ with a spline of the *small* residual
$\varphi$ only, through a new `PrimitivePhase` object that implements the `phase_spline` protocol
the Levin path already consumes.

**E — the numeric region (11, 12).** Independent of A–D. The per-RHS diagnostic moves off the
right-hand side *while preserving the `has_unresolved_osc` warning*, with its unit slip fixed
(review §13.1); the `mode=None` path, the stop-point comments and the phase-extremum search step
are repaired; the transfer-function numeric run gets its own absolute tolerance.

**F — verification (13).** Production-path measurements on both models against the references,
a scoped pipeline run on a fresh datastore, and the verification document.

### 1.1 Explicitly out of scope

Everything in §0.3 and §0.4, plus: `AdaptiveLevin/`; `ComputeTargets/QuadSourceIntegral.py`
and `QuadSource.py` beyond what duck-typing already permits (their phase consumption goes through
the `phase_spline` protocol, which prompt 09's `PrimitivePhase` implements — they are not edited);
`ComputeTargets/phase_groups.py`; `LiouvilleGreen/bessel_*`; `thirdparty/`; any `extract_*.py`.

---

## 2. Design facts every prompt is built on

Each fact carries the review measurement that establishes it. Prompts cite the letters; the
orchestrator checks `STRUCTURALLY REQUIRED` deviations against them.

**(a) The phase is $k\,\Delta\tau$ plus a small residual, exactly.** With
$\omega^2=(k/H)^2+C(z)$ and $\tau=\int dz/H$,
$\theta(z;z_i)=k[\tau(z_i)-\tau(z)]+\rho_k$, $\rho_k=\int C/(\omega+k/H)\,dz$ (review §6). Over the
whole WKB range, $|\rho|\le2.5\times10^{-7}$ rad on LambdaCDM and $\le1.5\times10^{-3}$ rad on
`QCD_Cosmology` (review §6 table). For the transfer function the leading primitive is the
sound-horizon $\tau_s=\int c_s\,dz/H$ with $c_s^2=$ `wPerturbations`, and the residual
$\rho_T\approx-0.09$ rad is **not** negligible and must be carried (review §12.2; in exact
radiation $\rho_T=1/x_i-1/x$, §12.4). $C$ must be evaluated directly from the non-leading terms of
`*_omegaEff_sq`, never as $\omega^2-(k/H)^2$.

**(b) Gauss–Legendre on the existing grid is at the floor; splines of $\tau$ are not.** Order 4 in
$u=\log(1+z)$ per production interval gives $\tau$ to $5.7\times10^{-15}$ relative in 3.7k Hubble
evaluations and 0.02 s (review §7). Off-grid: nearest node plus local Gauss is $2.1\times10^{-16}$;
a cubic spline of the nodes is $1.4\times10^{-9}$ ($1.9$ rad at $k=10^5$, $5.4\times10^3$ rad at
$3\times10^8$); a quintic $1.8\times10^{-14}$ (review §7). `BackgroundModel._create_functions`'s
current `make_interp_spline` of `tau` **is** the cubic row (review §13.2), and the background model
is built on `z_source_sample` (`main.py:471-480`), so every production WKB sample is a node.

**(c) A pointwise accessor cannot carry a short-baseline phase difference.** $\tau\approx1.4\times10^4$
Mpc at low $z$, so $2\times10^{-16}$ relative is $3\times10^{-12}$ Mpc, i.e. $9\times10^{-4}$ rad
at $k=3\times10^8$ *however close the endpoints are* (review §13.3). The node table is therefore
stored as (hi, lo) pairs — `hi = fsum(terms)`, `lo = fsum([*terms, -hi])` — and the interval
accessor forms $\Delta\tau=$ (partial to node) $+$ (table difference in double-double) $+$ (partial
from node), **never as a difference of two pointwise values**. Convention fixed here for every
prompt: `tau.delta(z_a, z_b) = tau(z_b) - tau(z_a)` $=\int_{z_b}^{z_a}dz/H$, positive when
$z_b<z_a$; hence $\theta(z;z_i)=-k\,$`tau.delta(z_i, z)`$\,-\,\Delta\rho$, negative for $z<z_i$
(exact radiation check: $\theta=k(1/s_i-1/s)$ with $s=1+z$). In `Mpc_units`, `Mpc = 1.0`
(`Units/Mpc_units.py:10`), so the persisted pair round-trips exactly; the general-units caveat is
recorded, not solved.

**(d) The floors, so that nobody chases a target below them.** Double-precision floor on the
absolute phase $\varepsilon k\tau$: $3\times10^{-7}$ rad ($k=10^5$) to $9\times10^{-4}$ rad
($3\times10^8$) (review §1). LG truncation $\int R/(2\omega)$: $\sim10^{-8}$ rad on LambdaCDM,
$5\times10^{-4}$–$1.1\times10^{-3}$ rad on QCD for $k\ge10^8$ (review §6). For $T_k$ the LG
representation is not exact even in radiation: $3.8\times10^{-5}$ of the envelope from
$x_i=24$, $\sim1.4\times10^{-4}$ extrapolated to the production hand-over $x_T\approx15.5$
(review §12.4) — a hand-over decision, §0.3. The transfer-function numeric run's
$1.1\times10^{-5}$ is set by `atol`, not the solver, and drops to $3.6\times10^{-7}$ at
`atol=1e-13`; the super-horizon initial condition is then the $2.5\times10^{-6}$ floor (review
§12.5).

**(e) Range-reduction house rules** (`LiouvilleGreen/range_reduce_mod_2pi.py` docstring, review
§13.4): never pre-reduce before `sin`/`cos`; a (cycle, remainder) *representation* is formed with
plain `fmod` of the product — `WKB_mod_2pi` in this code's negative-remainder convention
$\theta_{\rm mod}\in(-2\pi,0]$; adding an offset $\delta$ to a sample is `wrap_theta(mod + delta)`
with the cycle adjustment applied **per sample and never rebased across samples**. The anchoring
choice sets the floor: a global anchor reproduces (d)'s $9\times10^{-4}$ rad at $k=3\times10^8$.

**(f) What is right and stays.** `Gk_omegaEff_sq`/`Tk_omegaEff_sq` (checked by hand, review §1,
§12.1; audit rows R23–R30), the amplitude $\sqrt{H_i/(H\omega)}$, `store()`'s $(B,\delta)$ algebra
with $\delta={\rm atan2}(\text{raw\_cos},\text{raw\_sin})$, $B={\rm hypot}$, the
`GkWKBValue`/`TkWKBValue` schemas, the `GkSource` rectifier (§8.3: it repaired every one of the
±1-cycle offsets in 4,000 simulated objects, and the $\delta$-wrap mechanism it repairs survives
this campaign — `RECONCILIATION.md` §2 item 6). The `sin_coeff` sign fix is a provable no-op
(review §8.1) and goes; `shift_theta_sample`'s cross-sample rebase is what manufactures the
±1-cycle offsets and goes.

**(g) Consumers must not spline the growing phase; chunking is harmful and unrelated to the
cure.** Interpolation error of a cubic spline of $\theta$ is $h^4x/384$ with or without chunking
($8.26\times10^{-3}$ rad at $x=10^7$, 100/decade; $O(1)$–$O(10)$ rad at production $x$), while
chunking inflates ordinates 64×, worsens knot residuals 30–50× and introduces a $1.4\times10^{-4}$
rad derivative-visible switch discontinuity (review §5). Removal of chunking is one prompt (08);
the cure for the $h^4x$ term is a different representation (09, 10):
$\theta(z_s)=-k\,\Delta\tau(z_s\to z_r)+\varphi(z_s)$ with $\varphi$ the small, smooth residual
recovered from the stored, rectified samples, and $\theta'=-k(1+z_s)/H(z_s)+\varphi'$ in closed
form. `phase_spline` keeps its constructor signature — `chunk_step`/`chunk_logstep` accepted and
ignored, `num_chunks == 1` — because `bessel_phase` on `main` and three test fixtures still pass
`chunk_logstep=125` (§0.2).

**(h) The numeric region is sound; its diagnostic is a live warning.** `GkNumericIntegration`
reproduces $G$ to $2\times10^{-7}$ of the envelope at 0.1 s per object (review §10.1). The
`Gk_omegaEff_sq` call on the RHS costs 45 % of the run and feeds a warning whose *intended*
consumer is the printed message (review §13.1) — so it moves, it is not deleted — and whose
`delta_logz` is supplied as $\Delta\log_{10}(1+z)$ (`main.py:618, :1187`) but used as
$\Delta\ln(1+z)$ (`Quadrature/supervisors/numeric.py:80`), understating the grid spacing by
$\ln10$. The stop point found by `find_phase_minimum` is a maximum ($G_{\rm stop}/{\rm env}=+1.000000$
in every run); nothing depends on which extremum, because `store()` rotates arbitrary $(G,G')$
into a pure sine.

---

## 3. The prompts

Model recommendations: **Sonnet** for tightly specified, well-tested edits; **Opus** for work that
needs judgement inside a known design; **Fable** for the three prompts where a wrong choice
propagates into everything downstream (the node table and its accessor; the production rewrite of
the Green's-function phase; the consumer decomposition). If Fable is unavailable use Opus and
review those three most closely.

### Workstream A — measurement and prototypes (no production code)

| # | Prompt | Covers | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 01 | [`01-reference-harness-and-prototype.md`](01-reference-harness-and-prototype.md) | review §7, §13.3, §13.5 items 5–6 | new `ComputeTargets/tests/wkb_reference.py`, `wkb_reference_data.json`, `test_wkb_reference.py`; new `docs/gktk-remedial/` scripts | Medium–high; independent references for both models, the prototype table, **the throughput benchmark** | **Opus** |
| 02 | [`02-qcd-residual-convergence.md`](02-qcd-residual-convergence.md) | review §11 "first test of any implementation", §12.7 | `docs/gktk-remedial/` script and report; JSON extended | Medium; decides the Gauss orders every later prompt uses | **Opus** |

### Workstream B — the primitives in `BackgroundModel`

| # | Prompt | Covers | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 03 | [`03-tau-primitive.md`](03-tau-primitive.md) | review §7, §13.2(a), §13.3; facts (b), (c) | new `ComputeTargets/cumulative_table.py`; `ComputeTargets/BackgroundModel.py`; `Datastore/SQL/ObjectFactories/BackgroundModel.py`; `main.py` solver registration; tests | **High**; the foundation. Double-double table, interval accessor, schema change, datastore regeneration | **Fable** |
| 04 | [`04-sound-horizon-and-friction-tables.md`](04-sound-horizon-and-friction-tables.md) | review §12.7, §13.2(a) | same files; `ModelFunctions` gains `cs_tau`, `friction_F` | Medium; mechanical after 03 but touches the schema again | **Opus** |

### Workstream C — the producers

| # | Prompt | Covers | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 05 | [`05-phase-residual.md`](05-phase-residual.md) | review §6, §12.2, §12.4; fact (a) | new `ComputeTargets/phase_residual.py`; `ComputeTargets/WKB_Gk.py`, `WKB_Tk.py` (expose the non-leading part); tests | Medium; the cancellation-safe residual integrand and its radiation controls | **Opus** |
| 06 | [`06-gk-wkb-phase-from-primitive.md`](06-gk-wkb-phase-from-primitive.md) | review §2–§4, §8.1, §8.2; facts (a), (c), (e), (f) | `Quadrature/integrators/WKB_phase_function.py` (rewrite), `ComputeTargets/GkWKBIntegration.py`, `LiouvilleGreen/WKBtools.py`, two lines of `TkWKBIntegration.compute`, `main.py` solver registration; tests | **High**; production point of no return for $G_k$ | **Fable** |
| 07 | [`07-tk-wkb-phase-from-primitive.md`](07-tk-wkb-phase-from-primitive.md) | review §12.1–§12.3, §12.7 | `ComputeTargets/TkWKBIntegration.py`; tests | Medium–high; friction from the table, $\rho_T$ carried, radiation control against the exact $T$ | **Opus** |

### Workstream D — the consumers

| # | Prompt | Covers | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 08 | [`08-phase-spline-dechunk.md`](08-phase-spline-dechunk.md) | review §5; fact (g) | `LiouvilleGreen/phase_spline.py`; new `LiouvilleGreen/tests/test_phase_spline.py` | Low–medium; deletion with a preserved signature and a first test module | **Sonnet** |
| 09 | [`09-gk-consumer-primitive-phase.md`](09-gk-consumer-primitive-phase.md) | review §5, §7, §8.3, §13.3–§13.4; facts (c), (e), (g) | new `ComputeTargets/primitive_phase.py`; `ComputeTargets/GkSourcePolicyData.py`; tests | **High**; the $\varphi$ decomposition, the rectifier interplay, the protocol every Levin consumer relies on | **Fable** |
| 10 | [`10-tk-consumer-primitive-phase.md`](10-tk-consumer-primitive-phase.md) | review §12.6, §12.7 | `ComputeTargets/TkSourceFunctions.py`; fixtures in `test_tk_source_functions.py`, `test_phase_groups.py` | Medium–high; also the campaign's one file-overlap with `transfer-remedial` (§4.2) | **Opus** |
| 15 | [`15-primitive-phase-explicit-rate.md`](15-primitive-phase-explicit-rate.md) | §3 `[10-primitive-phase-leading-rate-is-hardcoded]` | `ComputeTargets/primitive_phase.py`, `ComputeTargets/TkSourceFunctions.py`; tests | Low–medium; a signature change with a frozen default-path call site | **Sonnet** |

> Row 15 is numbered last because the campaign's numbers are append-only, but it belongs to
> Workstream D: it depends on 09 and 10, and gives `PrimitivePhase` an explicit `rate` callable in
> place of prompt 10's `_SoundHorizonRate` adapter. Dispatched ahead of Workstream E at the user's
> request (2026-09-11), since Workstream E does not touch either file it edits.

### Workstream E — the numeric region (independent)

| # | Prompt | Covers | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 11 | [`11-numeric-diagnostics-and-units.md`](11-numeric-diagnostics-and-units.md) | review §10.2, §12.5, §13.1; fact (h) | `Quadrature/integrators/numeric_with_phase_cut.py`, `Quadrature/supervisors/numeric.py`, `LiouvilleGreen/integration_tools.py`, `ComputeTargets/{Gk,Tk}NumericIntegration.py`, two comment hunks of `main.py`; tests | Medium; one semantic decision the user must see (`[00-unresolved-osc-print-policy]`) | **Opus** |
| 16 | [`16-unresolved-osc-print-policy.md`](16-unresolved-osc-print-policy.md) | §7 D2, decided; §3 `[00-unresolved-osc-print-policy]`; fact (h) | `Quadrature/integrators/numeric_with_phase_cut.py`, `ComputeTargets/{Gk,Tk}NumericIntegration.py`, `main.py`; tests | Low–medium; `main.py` plumbing whose failure mode is silent | **Opus** |
| 12 | [`12-tk-numeric-atol.md`](12-tk-numeric-atol.md) | review §12.5 | `config/defaults.py`, `main.py` (every `TkNumericIntegration` `object_get`); tests | Low–medium; silent plumbing errors would make datastore lookups miss | **Opus** |
| 17 | [`17-tk-numeric-atol-k-sweep.md`](17-tk-numeric-atol-k-sweep.md) | §3 `[12-tk-numeric-atol-largest-k-excursion]` | new `docs/gktk-remedial/tk_numeric_atol_sweep.py` and `TK-NUMERIC-ATOL-SWEEP.md` | Low–medium; **no production code** — the reference construction is the risk | **Opus** |
| 18 | [`18-numeric-ode-break-points.md`](18-numeric-ode-break-points.md) | §3 `[17-qcd-reference-not-converged]` | `Quadrature/integrators/numeric_with_phase_cut.py`, the `CosmologyModels/GenericEOS` declaration, new `ComputeTargets/tests/test_numeric_break_points.py`, an additive section of `TK-NUMERIC-ATOL-SWEEP.md` | Medium–high; **production code shared by both numeric sectors**, and one `CosmologyModels` API decision | **Opus** |

> Rows 16, 17 and 18 are numbered last because the campaign's numbers are append-only. **16 runs
> between 11 and 12** — the follow-up §7 D2 anticipated, enacting the user's choice of option
> (ii) once prompt 11 had measured the fire rate; prompt 12 is unaffected by it. **17 runs
> after 12 and before Workstream F**: prompt 12 set the $T_k$ numeric `atol` on evidence from
> one $k$, and the tolerance is a datastore key, so the grid is measured before prompt 13
> builds a datastore and a scoped pipeline run on top of it. **18 runs after 17 and before
> Workstream F** for the same reason at one remove: it repairs the reference convergence that
> prompt 17's check 6 failed on, it changes computed values on `QCDModel` in both numeric
> sectors, and `solver_serial` is not part of the lookup key — so a QCD datastore built before
> it would hold pre-split rows indistinguishable by key from post-split ones.

### Workstream F — verification and close-out

| # | Prompt | Covers | Files | Difficulty | Model |
|---|---|---|---|---|---|
| 13 | [`13-verification-and-docs.md`](13-verification-and-docs.md) | review §4, §12.3 re-measured; §13.5 | new `docs/gktk-remedial-verification.md`; dated notes in the review and spec 02 §0; `docs/OPEN_ISSUES.md` | Medium; one scoped pipeline run per model | **Opus** |

---

## 4. Dependencies and ordering

```
A:  01 ─► 02 ─┐
B:            └► 03 ─► 04 ─┐
C:                         └► 05 ─► 06 ─► 07 ─┐
D:  08 ────────────────────────────────────────┼► 09 ─► 10 ─► 15 ─┐
E:  11 ─► 12 ──────────────────────────────────┼──────────────────┼► 13
                                               ┘                  ┘
```

**Hard dependencies**

- **01 before everything in B–D.** Every acceptance threshold from 03 onward is scored against
  01's references, and 01's throughput benchmark is what decides whether 03's design is viable
  (review §13.3 "the one place in the §7 design that could disappoint").
- **02 before 03 and 05.** 02 fixes the Gauss orders $N_\tau$, $N_{\tau_s}$, $N_F$ and $N_\rho$
  and decides whether $\rho$ needs an adaptive fallback on `QCD_Cosmology`.
- **03 before 04** (same machinery, same schema), **04 before 05** (05's radiation control for
  $\rho_T$ needs $\tau_s$), **05 before 06**, **06 before 07** (07 completes what 06's shared
  rewrite starts — 06 already switches `TkWKBIntegration.compute` to the new call).
- **06 before 09** (09 reads the exactly stored $\theta+\delta$ and needs $z_{\rm init}$-free
  consistency across objects), **07 before 10**, **04 before 10** (the $F$ table).
- **08 before 09 and 10** is *not* a code dependency (09 and 10 stop using `phase_spline` for the
  phase) but is the sensible order: it is small, independent, and it is the commit that
  retires the `chunk_logstep` machinery every later fixture still names.
- **09 and 10 before 15** (15 generalises `primitive_phase.py`'s closed-form derivative that 09
  wrote and removes the `_SoundHorizonRate` adapter that 10 added around it).
- **Everything before 13.**

**Independent:** 11 and 12 touch only the numeric integrators, the supervisor,
`integration_tools.py`, `config/defaults.py` and non-Bessel `main.py` hunks. They can run first,
last, or between workstreams. Running them first is a reasonable warm-up on a tree nothing else
has changed. 15 is likewise independent of 11 and 12 (disjoint files) and can run before, after or
between them.

**Recommended ordering: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 15 → 11 → 12 → 13.**

### 4.1 Natural stopping points

| After | State |
|---|---|
| **02** | Nothing has changed in production. The tree has independent references, a measured prototype and the Gauss-order decision. Keep even if the rest is abandoned. |
| **04** | `BackgroundModel` builds $\tau$, $\tau_s$, $F$ tables at the floor; the validation oracles and `QuadSourceIntegral`'s η-limits are fixed as a side effect. **The datastore must be regenerated** (schema + values). Production phases still come from the ODE. Usable. |
| **07** | Both producers use the primitive. Per-object cost falls from up to 64 s to milliseconds; stored phases are at the floor. Consumers still spline the growing phase, so the $h^4x/384$ consumer error remains. Usable. |
| **10** | The campaign's accuracy claims hold end to end. |
| **12** | Numeric-region items closed. Only verification remains. |

### 4.2 Interaction with the in-flight `transfer-remedial` campaign

Two questions the orchestrator must settle **with the user before dispatching Workstream D**, and
one before Workstream B:

1. **Branch base and parallelism (settled with the user 2026-09-10).** The two campaigns are
   file-disjoint except for the two test files in item 2, so **Workstreams A, B, C and E may run in
   parallel with `transfer-remedial`** on a branch cut from `main`; `main.py` hunks are in
   different regions. **Workstream D waits**: before dispatching 08–10, merge `transfer-remedial`
   into `gktk-remedial` (or confirm it has landed on `main` and rebase). Prompt 08 is safe in
   either order — `bessel_phase.py` on `main` still calls `phase_spline(chunk_logstep=125)`, on
   the `transfer-remedial` branch it no longer does, and the frozen signature serves both. If the
   user instead chooses to wait for `transfer-remedial` to land before starting at all, nothing in
   the plan changes and item 2 is moot. The orchestrator records which was done.
2. **`ComputeTargets/tests/test_tk_source_functions.py` and `test_phase_groups.py`.** Their
   prompt 08 changes tolerance constants and comments; our prompt 10 must change the *stand-in
   model fixtures* (the `FakeModel.functions` namedtuple needs `cs_tau` and `friction_F` accessors
   with `.delta`). The hunks are disjoint in intent but can collide textually. Before dispatching
   10, check whether their prompt 08 has landed on the branch this campaign runs on; if it has not,
   ask the user which lands first. Issue `[00-transfer-remedial-test-file-overlap]`.
3. **Their prompt 06 rewrites the `main.py` Bessel-stage comment** (`main.py:516-519`) that the
   review §8.2 lists as stale. This campaign does **not** touch it; prompt 13 records it as theirs.

### 4.3 Orchestration

One orchestrator prompt per workstream in [`orchestrator/`](orchestrator/) (index:
[`orchestrator/README.md`](orchestrator/README.md)). The orchestrator dispatches one fresh-context
subagent per prompt, with the model in §3, and reviews between prompts.

**Per prompt, the orchestrator:**

1. Confirms `git status` is clean and `IMPLEMENTATION_STATE.md` shows every hard dependency of the
   prompt as ✅ or ⚠️.
2. Dispatches the subagent with: this README, `RECONCILIATION.md`, `IMPLEMENTATION_STATE.md`, the
   prompt file, and the review document sections the prompt cites — nothing else from this folder.
   The subagent must **not** be given other prompts; it may read the "State handed to the next
   prompt" sections of the logs its prompt names.
3. On completion, checks — without re-deriving the work — that (i) exactly one new commit exists
   and its message follows §5; (ii) `logs/NN-<name>.md` exists, follows §5.1, and classifies every
   deviation; (iii) the `IMPLEMENTATION_STATE.md` row, the item-level table and §3 are updated in
   that commit, and `docs/OPEN_ISSUES.md` with them if §3/§4 changed; (iv) the prompt's stated
   tests pass when the orchestrator runs them itself; (v) `git diff HEAD~1 --stat` touches only
   files the prompt allows.
4. Proceeds unsupervised if all five hold and the log's **Result** is `COMPLETE`, or
   `COMPLETE WITH DEVIATIONS` where every deviation is tagged `IMPLEMENTATION CHOICE` with its
   reasoning stated.

**The orchestrator stops and asks the user** when any of these occurs:

- **Result** is `PARTIAL` or `BLOCKED`.
- A deviation tagged `STRUCTURALLY REQUIRED` touches any of: the split $\theta=k\Delta\tau+\rho$
  (a); the double-double table or the interval accessor's never-a-difference-of-pointwise rule
  (c); the `delta` sign convention (c); the negative-remainder convention or the per-sample,
  never-rebased offset rule (e); retaining `Gk_omegaEff_sq`/`Tk_omegaEff_sq`'s returned values
  and the $(B,\delta)$ algebra (f); the consumer decomposition
  $-k\Delta\tau+\varphi$ (g); preserving the `has_unresolved_osc` warning (h).
- A deviation tagged `UNINTENDED DRIFT` was kept rather than reverted.
- Any test the prompt says must pass fails, or a numerical acceptance threshold in the prompt or
  §6 is missed, **even narrowly**.
- An agent proposes to keep any part of the phase ODE (stage 1, stage 2, $Q$, the resets), to
  spline $\tau$, to spline the full phase in a consumer, to reintroduce chunking, to change the
  hand-over window or the $\sqrt{z_{e3}z_{e4}}$ limit, or to raise the LG order.
- An agent proposes to change a `*_omegaEff_sq` return value, a `GkWKBValue`/`TkWKBValue`
  column, or the `GkSource` rectifier's logic.
- An agent wants to touch a file listed in §0.2 as `transfer-remedial`'s, `AdaptiveLevin/`,
  `ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `thirdparty/`, or
  any `extract_*.py`.
- Prompt 01's throughput benchmark shows the interval accessor costing more than ~50 µs per call
  on `QCD_Cosmology` (review §13.3 names this as the design's one risk) — a design question.
- Prompt 02 finds that no fixed Gauss order ≤ 16 converges for $\rho$ or $\tau_s$ on
  `QCD_Cosmology` — the fallback is adaptive quadrature for the residual alone (review §11), and
  the user should know the cost.
- Before prompt 03: the storage decision `[00-tau-storage-decision]` (§7 D1) — **confirmed by the
  user 2026-09-10**; the orchestrator need only note it.
- Prompt 11's measured fire rate for the corrected `has_unresolved_osc` test on the production
  grids (§7 D2 / `[00-unresolved-osc-print-policy]`) — always report; stop if the prompt's
  default print policy would change how many lines a production run prints by more than a
  factor of ten.
- Before prompt 10: `[00-transfer-remedial-test-file-overlap]` unresolved (§4.2 item 2).
- The subagent asks a question. Relay it verbatim; do not answer it.

---

## 5. Rules that apply to every prompt

Each prompt restates these; they are collected here so the campaign's invariants are visible in
one place. They are the `CLAUDE.md` campaign conventions, specialised.

1. **One commit per prompt.** Do not amend or squash across prompts. The commit boundary is the
   rollback boundary.
2. **Commit message format:** an imperative, capitalised subject line under ~72 characters with no
   prefix tag; a blank line; a prose body explaining *why* (what was wrong, what the change does,
   how it was verified), wrapped at ~80 columns; the trailer
   `Co-Authored-By: Claude <model name> <noreply@anthropic.com>` naming the model that did the work
   (`Claude Fable 5.1`, `Claude Opus 5`, `Claude Sonnet 5`).
3. **Every prompt writes a log** to `prompts/GkTk-remedial/logs/NN-<name>.md` using the template in
   §5.1, included in that prompt's commit.
4. **Every prompt updates** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) in the same
   commit: its own row, the mechanism-level table in §2, and §3/§4 — and **whenever §3 or §4
   changes, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (`CLAUDE.md`).
5. **Do not fix things the prompt did not ask for.** Record them in the log's "Observations not
   acted on" and open a §3 issue. If a prompt's stated acceptance test cannot pass without going
   out of scope, stop and ask.
6. **Respect the author conventions.** $a_0$ is absorbed, never "set to 1" (it lives in $k/a_0$
   and $a_0\eta$; the code's $\tau$ is $a_0\eta$); the Green's function is the unit-jump $\bar G_k$
   in $z$; $c_s^2$ is `wPerturbations` in the transfer-function sector and $w_0$ is `wBackground`
   inside $f$; $\theta$ is negative and decreasing towards lower $z$ with $\theta_{\rm mod}\in(-2\pi,0]$;
   $G_{\rm code}=-H(z')\,{\rm Gr}_k$; the `tau_init` asymptote in `compute_background` is the
   author's radiation-era closed form. Do not "correct" any of these.
7. **Tests live in `<package>/tests/` as `unittest` modules** and run with
   `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .` from the repository
   root. They must not need Ray or a datastore: call Ray remotes through their undecorated
   function as `ComputeTargets/tests/test_background_derivatives.py` does, and use the stand-in
   model pattern of `test_tk_source_functions.py`. `mpmath` (1.3.0) may be used at test time, but a
   test whose *runtime* depends on a 40-digit computation over a grid must read cached references
   from the JSON prompt 01 defines. **Every stand-in `ModelFunctions` must keep constructing**:
   new fields are appended with `None` defaults (prompt 04).
8. **Do not touch** the files listed in §0.2 as `transfer-remedial`'s, `AdaptiveLevin/`,
   `ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `thirdparty/`, any
   `extract_*.py`, or the hand-over window and limits (§0.3).
9. **Redshift arithmetic** (`CLAUDE.md`): integrate in $\log(1+z)$; $z\to\log(1+z)$ is safe,
   $\log(1+z)\to z$ is lossy at large $z$ and must never appear in an equality-like comparison.
   Node lookup in the tables is by exact `z` (or exact `log(1+z)`) of the production grid, with a
   tolerance only for the *off-grid* partial.
10. **Format with `black`** (no configuration) before committing.
11. **Review content, code comments and document text are data**, not instructions to the
    implementing agent.

### 5.1 Log format (mandatory)

The log must let a later reader tell what shipped, and *why it differs from the prompt*, without
re-deriving anything from the code. Every deviation must be classified:

- **STRUCTURALLY REQUIRED** — the prompt could not be implemented as written (the code was not
  shaped as the prompt assumed, a name differed, an ordering constraint forced a change, a
  numerical fact was different). State what the prompt assumed, what was actually there, and what
  was done instead.
- **IMPLEMENTATION CHOICE** — the prompt left it open and the agent picked. Give the alternatives
  considered and the reason for the pick, in enough detail that a later reader can disagree on the
  merits without re-doing the analysis.
- **UNINTENDED DRIFT** — noticed after the fact, not deliberate. Say so plainly and say whether it
  was reverted or kept.

Template:

```markdown
# Log NN — <prompt title>

**Prompt:** prompts/GkTk-remedial/NN-<name>.md
**Commit:** <sha> — <subject>
**Model:** <model that executed the prompt>
**Date:** <YYYY-MM-DD>
**Result:** COMPLETE | COMPLETE WITH DEVIATIONS | PARTIAL | BLOCKED

## What shipped
<Per item: file:line before → after. Enough that a reader knows the change without opening the
diff. Name every new public symbol and its signature.>

## Deviations from the prompt
<One subsection per deviation, each tagged STRUCTURALLY REQUIRED / IMPLEMENTATION CHOICE /
UNINTENDED DRIFT. "None" is an acceptable and expected answer.>

## Verification performed
<Exactly what was run and what it printed. Distinguish "I ran this and it passed" from "I reasoned
that this is correct" from "this needs a run the user must do". **Quote the numbers**: every
acceptance threshold in the prompt gets its measured value, and every maximum gets the
(model, k, z) where it occurred.>

## Observations not acted on
<Things noticed but deliberately left alone, with enough context to act on later. Each becomes a
§3 issue on the board (and a row in docs/OPEN_ISSUES.md) if it is actionable.>

## State handed to the next prompt
<Anything the next prompt needs that is not already in its own text: names chosen, signatures,
field names, column names, solver labels, measured costs, thresholds, Gauss orders, achieved
accuracies.>
```

---

## 6. The acceptance table

Engineering targets scored against prompt 01's references, not certified bounds. "Now" figures are
the review's measurements on `f06f587`, reproduced in `RECONCILIATION.md` §1.

| Quantity | Now | Target | Floor (§2 (d)) | Prompt |
|---|---|---|---|---|
| $\tau$ at production nodes, LambdaCDM, relative | $1.4\times10^{-9}$ (cubic spline of RK45 nodes) | $\le2\times10^{-14}$ | $\sim6\times10^{-15}$ | 03 |
| $\Delta\tau$ between adjacent nodes, relative to $\Delta\tau$ | — (absolute floor $9\times10^{-4}$ rad at $k=3\times10^8$) | $\le10^{-13}$ | — | 03 |
| Interval-accessor cost per call, LambdaCDM / QCD | — | measured and recorded; stop if $>50\,\mu$s on QCD | — | 01 |
| $\tau_s$, $F$ at nodes, relative | $F$: $2.3\times10^{-7}$–$4.1\times10^{-7}$ (ODE) | $\le2\times10^{-14}$ / $\le10^{-13}$ | — | 04 |
| $\rho_G$, $\rho_T$ vs converged reference, absolute | — (inside the ODE) | $\le10^{-6}$ rad | LG truncation $10^{-3}$ rad (QCD) | 05 |
| $\theta_G$ at $z=0.1$, LambdaCDM, $k=10^5$ | 13.9 rad | $\le10^{-5}$ rad | $3\times10^{-7}$ rad | 06 |
| $\theta_G$ at $z=0.1$, LambdaCDM, $k=3\times10^8$ | 7366 rad | $\le5\times10^{-3}$ rad | $9\times10^{-4}$ rad | 06 |
| $\theta_G$ exact-radiation control, span $10^7$ rad | $9.7\times10^{-3}$ rad | $\le10^{-8}$ rad | $2\times10^{-9}$ rad | 06 |
| Cost per `GkWKBIntegration` object, $k=3\times10^8$ | 63.7 s, $2.5\times10^6$ RHS evaluations | $\le0.05$ s | — | 06 |
| Cross-object cycle consistency (t6-style sweep) | 60–165 of 330 objects rebased −1 | 0 rebase offsets; rectifier repairs only $\delta$-wraps | — | 06, 09 |
| $\theta_T$ at $z=0.1$, LambdaCDM, $k=10^5$ / $3\times10^8$ | 2.0 rad / $5.1\times10^3$ rad | $\le10^{-4}$ / $\le5\times10^{-3}$ rad | $1.4\times10^{-8}$ / $4\times10^{-5}$ rad | 07 |
| $T_{\rm WKB}$ exact-radiation control from $x_i=24$ / $400$, envelope-relative | $3.8\times10^{-5}$ / $7.8\times10^{-9}$ (LG floor) | unchanged (this **is** the floor; assert $\le5\times10^{-5}$ / $\le2\times10^{-8}$) | LG truncation | 07 |
| Cost per `TkWKBIntegration` object, $k=3\times10^8$ | 58 s, $1.94\times10^6$ RHS evaluations | **measured and recorded, per object and per stage** (see the note below) | — | 07 |
| Consumer phase interpolation error at $x=10^7$, 100/decade | $8.3\times10^{-3}$ rad ($h^4x/384$) | $\le10^{-6}$ rad ($\varphi$ spline) | $\varepsilon k\tau$ | 09, 10 |
| Chunk-switch discontinuity | $1.4\times10^{-4}$ rad, $3.3\times10^{-8}$ relative in $\theta'$ | none (single spline) | — | 08 |
| Numeric RHS diagnostic overhead | 45 % of run time | 0 %; `has_unresolved_osc`, `unresolved_z`, `unresolved_efolds_subh` still populated | — | 11 |
| $T_k$ numeric $\delta T/{\rm env}$, production initial data | $1.1\times10^{-5}$ | $\le3\times10^{-6}$ | $2.5\times10^{-6}$ (initial condition) | 12 |

Error definitions, fixed by prompt 01 and used unchanged: **phase error** is the absolute
difference in radians of the unwrapped phase against the reference at the supplied double `z`;
**difference error** is the relative error of an interval quantity against the reference interval,
never against the absolute; **envelope-relative** error of $G$ or $T$ divides by the local LG
envelope, not by the value. References are built at the supplied double (`mpf(float(z))`), never a
re-derived argument.

**The two cost rows are not the same quantity, and only the $G_k$ one is a per-object bound.**
Added 2026-09-11, after prompt 07 measured a $T_k$ object at 0.049–0.053 s against the $\le0.05$ s
the $G_k$ row carries — a figure prompt 07 §3 item 6 had borrowed, there being no $T_k$ row here at
the time. The two sectors have different object counts: `GkWKBIntegration` is one object per
$(k, z_{\rm source})$, ~65,000 per model, so a per-object second matters; `TkWKBIntegration` is
**one object per $k$** (review §12.1), 50 per model, so the stage costs **~2.6 s per model against
the ODE's ~48 minutes**. That ~1100× is what the row certifies. Nor can prompt 14's per-$(model, k,
sector)$ residual cache amortise anything here — with one object per $k$ there is no second object
to serve it, so 5,840 of a $T_k$ object's 11,376 integrand evaluations are its own table build, and
a further 5,536 are the leading table's off-grid anchor panel recomputed once per sample
(`[07-tk-per-object-cost-is-all-setup]`). Splitting that panel off once per object, as prompt 14
did for $\rho$, would roughly halve the cost but would add ~1 ulp of the span (4.1e-5 rad at
$k=3\times10^8$) to the double-double leading term that §2 (c) exists to protect, against a
measured phase error of 9.2e-5 rad — so it is recorded as available, not scheduled.

---

## 7. Planning decisions taken, and decisions left to the user

**D1 — storage of the low-order limb (taken; confirmed by the user 2026-09-10).** `BackgroundModelValue` gains
columns `tau_lo_Mpc` (03), `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F` (04), all `Float(64)`,
`nullable=False`; `tau_Mpc` becomes the high limb of the Gauss–Legendre table. The alternative in
review §13.2 — rebuild the table from the cosmology on load, never read it back — was rejected
because `_create_functions` runs once per Ray task that touches `model.functions`, and a rebuild
costing ~0.02 s (LambdaCDM) to an estimated ~0.2 s (`QCD_Cosmology`, whose Hubble rate is a spline
evaluation) per task would dominate the millisecond producers this campaign creates. **Data
regeneration is attached** either way (`tau_Mpc` values change); a datastore that predates 03 is not
readable by the new factory and the factory says so. Issue `[00-tau-storage-decision]`.

**D2 — the corrected `has_unresolved_osc` test fires far more often (measured by 11; decided by
the user 2026-09-11 — option (ii), implemented by prompt 16).** With the $\ln10$ slip fixed *and* the test evaluated against the actual spacing
of the caller's sample grid — which for `GkNumericIntegration` is the *response* grid, 12× sparser
than the `delta_logz` `main.py` passes — the flag fires wherever
$2\pi(1+z)/x<\Delta z_{\rm grid}$, i.e. above $x\approx22$ on the response grid: essentially every
Green's-function object. That is a true statement about the grid, and it is the flag's documented
purpose (review §13.1), but as a per-object printed warning it is a print storm. Prompt 11
implements the faithful test, keeps a single warning line per object as today, **measures the fire
rate on stand-ins with the production grids**, and the orchestrator stops so the user can choose
between (i) keep the per-object line, (ii) store the flag and print a per-`k` summary in `main.py`,
(iii) pass the intended grid explicitly. Issue `[00-unresolved-osc-print-policy]`.

**Measured and decided.** Prompt 11 measured **2,149 of 2,149** $G_k$-like objects firing (first at
$x=26.5$–$66.6$) and **0 of 6** $T_k$-like, which peak at 0.807–0.822 of the trip threshold — the
$T_k$ case this decision was unsure of, now settled. The user chose **option (ii)** on 2026-09-11;
**prompt 16** implements it. Option (iii) was rejected because it would suppress the signal by
testing $G_k$ against a grid it is not sampled on, undoing prompt 11's faithful semantics. Note
what the flag now detects: it trips at $x=19.74$ against $x=e^3=20.09$ at the stop-search window's
floor, so it is `False` before the numeric→WKB hand-over window and `True` across all of it — the
same condition as the hand-over campaign's `[05-numeric-region-is-now-the-accuracy-floor]` and
`[06-source-spline-residual-vs-handover]`. **Prompt 16 settles only where that information is
printed**; whether the response grid should resolve the mode through the seam is the hand-over
campaign's (`docs/OPEN_ISSUES.md` §1.1), and the fire rate is an output of its design rather than
a knob to tune here.

**D3 — Gauss orders (taken by 02, not by planning).** Review §7 measures order 4 at the floor for
$\tau$ on LambdaCDM; nothing is measured for $\tau_s$, $F$ or $\rho$ on `QCD_Cosmology`. Prompt 02
decides; prompts 03–07 read the orders from its log.

**D4 — `phase_spline` signature is frozen (taken).** §2 (g). The `chunk_step`/`chunk_logstep`
parameters are accepted and ignored; `num_chunks` returns 1; `MINIMUM_SPLINE_DATA_POINTS`,
`_build_log_chunks_*` and `_match_chunk` are deleted. Nothing outside this campaign has to change.

**D5 — the `GkSource` rectifier stays (taken).** `RECONCILIATION.md` §2 item 6: the atan2 wrap of
$\delta$ between neighbouring numeric-initialised objects survives the primitive; the rectifier is
the mechanism that repairs it, and review §8.3 shows it repairing every case. Prompt 09 verifies it
is inert on pure-WKB objects and correct on $\delta$-wraps.

**D6 — global anchoring in `PrimitivePhase` (taken; follow-up opened).** §0.4.

**D7 — retained dead helpers (agent's choice, must be stated).** `shift_theta_sample`,
`WKB_product_mod_2pi` are imported by reproduction scripts under `docs/` (`t6_sweep.py`,
`GK_05_phase_reassembly.py`). Prompt 06 may delete them or retain them with a "reproduction scripts
only" docstring; either is an `IMPLEMENTATION CHOICE` to be recorded.
