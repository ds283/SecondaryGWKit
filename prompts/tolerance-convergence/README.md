# Tolerance and convergence campaign — every accuracy knob in the pipeline, calibrated

**Origin:** [`prompts/GkTk-remedial/`](../GkTk-remedial/README.md) prompts 12 and 17, and the
review [`docs/gk-wkb-review-fable-2026-09-09.md`](../../docs/gk-wkb-review-fable-2026-09-09.md)
§10.1 and §12.5.
**Planned:** 2026-09-12 at `622b84b`. **Rebased:** 2026-09-16 at `acd5b8e`, on the tree left by
`GkTk-remedial` (20 / 20) and `qcd-background-audit` (16 / 16). **Re-anchored:** 2026-09-16 at
`bc6dc97`, on the tree left by `background-solver-robustness` (9 / 9), which ran on this branch.
**Not yet started** — prompts 01 and 02 written 2026-09-16, **03–06 deliberately held** until 02's
inventory lands (board §1).
**Branch:** `tolerance-convergence`, cut from `main` at `acd5b8e`, now 25 commits ahead of it.
**What the rebase and the re-anchor changed, and why:** [`RECONCILIATION.md`](RECONCILIATION.md) —
§§1–6 for the rebase, **§7 for the re-anchor** — read it before believing any figure quoted here
against the older documents.
**Status board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) · **Logs:** [`logs/`](logs/)
· **Orchestrator:** [`orchestrator/`](orchestrator/)

---

## 0. What this campaign is, and its boundaries

### 0.1 The one-sentence version

Eight object types in the pipeline are keyed on an accuracy parameter, six of them share one
`(atol, rtol)` pair that was never chosen for any of them, and of those six **only one actually
uses it**: this campaign **establishes what every accuracy knob in the pipeline really is,
calibrates each one on its own terms, and decouples them so that each quantity carries a parameter
whose provenance can be stated.**

The knob is not always a tolerance. Where a quantity is produced by an adaptive solver — the two
numeric sectors, and the root solve that locates horizon exit — it is an `(atol, rtol)` pair and
the campaign's job is to measure it. Where it is produced by a **Liouville–Green-type
representation evaluated on a fixed-order Gauss–Legendre table** — `BackgroundModel`'s three
primitives and the two WKB sectors' phase residual — there is no tolerance at all and the knob is
**an integer order**. Four of the eight carry an `atol`/`rtol` column pair that reaches no solver;
for three of those four the honest key is the order, and putting it there is the campaign's
production change.

It exists because `GkTk-remedial` prompt 12 set one constant on one wavenumber of one model,
prompt 17 then measured the production grid and found the target missed at 3, 13 and 8 of 50
wavenumbers on the three models, and the lever turned out to be `rtol` — a constant prompt 17 was
not allowed to touch. Nobody has checked the rest, and the two campaigns that have run since have
moved the background, the break-point set and the grid underneath every figure that exists.

### 0.2 What "convergence test" means here, and why it is not an oracle comparison

There is no closed-form solution on `LambdaCDMModel` or `QCDModel` — the Hubble rate is a spline —
so for most (target, model, $k$) triples the reference is **a converged run of the same
integrator**, and the test is a self-convergence one: build the reference at a tolerance far
tighter than any candidate, build it again a decade tighter still, and require that the reference
move by at least an order of magnitude less than the smallest difference the measurement intends to
report. If it does not, the measurement is measuring its own reference.

That surrogate is **calibrated, not assumed**. On a constant-$w$ background the closed forms in
[`ComputeTargets/analytic_Gk.py`](../../ComputeTargets/analytic_Gk.py) and
[`analytic_Tk.py`](../../ComputeTargets/analytic_Tk.py) are exact — `compute_analytic_T(k, w, tau)`
is the Bessel form $2^{n}\Gamma(n+1)(kc_s\tau)^{-n}J_{n}(kc_s\tau)$ with $n=\tfrac32+b$, reducing
at $w=\tfrac13$ to $3(\sin x - x\cos x)/x^3$ — so on the radiation control both the drift statistic
and the distance to truth can be computed and compared. `GkTk-remedial` prompt 17 did exactly this
for $T_k$ and found them the same order (4.21e-11 drift against 2.3–4.3e-11 from the oracle). Every
prompt here repeats that calibration for its own target before trusting the drift figure anywhere
else. **Radiation domination is the anchor, and the general constant-$w$ form is what the code
already provides.**

### 0.3 The tree this campaign runs on, and what moved under the old plan

Both campaigns the 2026-09-12 plan waited on have closed and merged; `RECONCILIATION.md` scores
every claim of that plan against the tree at `acd5b8e`. The five results a prompt here must not
re-derive:

- **`[17-qcd-reference-not-converged]` is closed.** The numeric ODE is split at the cosmology's
  declared non-smoothness and *which* kind is the sector's choice, keyed in the datastore
  (`GkTk-remedial` prompts 18, 19, 20). Worst reference-convergence drift on `QCDModel` is
  **7.08e-09** ($T_k$) and **3.67e-09** ($G_k$) against the 3.4e-08 criterion, zero offenders at
  all 50 production wavenumbers (`docs/qcd-background-audit/PER-SECTOR-POLICY.md` §2, §4). **A
  tolerance may now be measured on QCD.**
- **The QCD background is correct.** Its $\int\mathrm{d}z/H$ is bit-identical to an independently
  root-solved exact background where it carried 3.461e-08, and the equivalent phase error is
  0.000e+00 rad at all three probe wavenumbers (`docs/qcd-background-verification.md` §2). **The
  QCD $H(z)$ discontinuity floor prompt 17 warned about is gone**, and no prompt here may quote it.
- **`integration_break_points` declares 3 crossings, not ~404 knots** (`qcd-background-audit`
  prompt 07). `BREAK_POINT_ALL` and `BREAK_POINT_DISCONTINUITY` now differ by one kink, and the
  wider policy costs **+0.99 %** in the $T_k$ sector rather than the +220 % `GkTk-remedial` prompt
  19 measured.
- **The production source grid has been rebuilt twice** (`qcd-background-audit` prompts 11 and 15):
  it is now built around the cosmology's declared crossings and equidistributed on the phase
  residual's curvature, 1,996 samples on QCD and 1,778 on LambdaCDM,
  `SOURCE_GRID_CONSTRUCTION_VERSION = 2`. **Every published figure in §6 was taken on an earlier
  generation of it**, and so does the test tree's reproduction — see (b) below.
- **Nothing in either campaign moved a tolerance.** `config/defaults.py` is byte-identical to the
  file the plan was written against.

**Re-anchored 2026-09-16 at `bc6dc97`** (additively, §5 rule 7). A third campaign has since run —
`prompts/background-solver-robustness`, 9 / 9, planned and merged **on this branch** after the
rebase above was written. It is not a precondition; it is a campaign that happened to land on one
of this campaign's own subjects. `RECONCILIATION.md` **§7** scores it. The four things a prompt
here must know:

- **The baseline moved.** `bc6dc97`, 25 commits ahead of `main`. Suites re-run for the re-anchor:
  `ComputeTargets` **452**, `CosmologyModels` **39**, both OK. The five results above all still
  hold, and `config/defaults.py` is *still* byte-identical — now across three campaigns and two
  rebases.
- **The three `LambdaCDM_GenericEOS.py` root solves are settled, by the campaign that owns that
  file.** Prompt 02 **lifts**
  [`background-solver-robustness/PROVENANCE.md`](../background-solver-robustness/PROVENANCE.md)
  rather than re-deriving them; §3.2 says so and its line numbers are confirmed at `bc6dc97`.
- **The version-2 grid reproduction already exists**, privately, in
  `ComputeTargets/tests/test_source_grid.py`. There are **four** reproductions in the test tree,
  not three, and prompt 01's job is to *hoist* one rather than build it (§3.1, and
  `RECONCILIATION.md` §7.3).
- **`cosmology_feature_redshifts` no longer computes the equality redshifts** — it asks the
  cosmology, and raises with no fallback if the cosmology cannot answer. Prompt 01 lifts that
  function, so its stand-in must answer (`RECONCILIATION.md` §7.4).

### 0.4 Boundary with `AdaptiveLevin` and `QuadSourceIntegral`

`ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py` and `AdaptiveLevin/`
belong to [`prompts/levin-refactor`](../levin-refactor/) and
[`prompts/qsi-phase-groups`](../qsi-phase-groups/), and `docs/OPEN_ISSUES.md` §1.2 and §1.3 park
work against them. `QuadSourceIntegral` is also the one target whose tolerances are **already
decoupled** — `DEFAULT_QUADRATURE_ATOL`/`_RTOL`, distributed per sub-interval by log-width
(`QuadSourceIntegral.py:815-819`) — and whose `atol` was already chosen by measurement against an
analytic oracle (`prompts/source-remediation`, `[12-atol-too-loose-for-the-source-integral]`).

**This campaign does not edit any of those files.** It applies its convergence test to
`QuadSourceIntegral` **read-only**, as prompt 06, and hands the result to whoever owns it. If a
prompt here finds it must change one of those files to proceed, that is a stop.

### 0.5 What this campaign does *not* do

- **Does not change any integrator's algorithm.** No method swaps, no re-derivations. It measures
  what the existing code does and chooses the parameters it is given. Raising a Gauss order is a
  *parameter* change and is in scope; replacing a quadrature rule is not.
- **Does not revisit the initial conditions.** The $T=1,T'=0$ super-horizon condition holds a
  2.52e-6-of-envelope floor (`[00-tk-superhorizon-ic-series]`); it is a *floor to measure against*,
  never a target to beat. Claiming an accuracy below a declared floor is a campaign-wide stop
  (§2 (f), qualified by §6.1 rule 5); this particular floor is not one a prompt here re-measures.
- **Does not move the numeric→WKB hand-over**, which is `docs/OPEN_ISSUES.md` §1.1's — including
  `[11-stop-point-root-tolerance]`, the `root_scalar(xtol=1e-6, rtol=1e-4)` at
  `LiouvilleGreen/integration_tools.py:95`. It is **recorded** in prompt 02's inventory and in the
  provenance note, and it is **not retuned here.**
- **Does not change the source grid**, the break-point policy, or `RESIDUAL_WKB_REGION_MARGIN`'s
  value. Prompt 04 measures what the margin is worth, because
  `[20-wkb-gauss-orders-not-in-lookup-key]` says nobody has; changing it is a separate decision.
- **Does not re-decide the break-point kinds.** `qcd-background-audit` README §7 D5 is open and is
  that campaign's; this one reads `BREAK_POINT_KIND` and holds it fixed.

---

## 1. What this campaign does

Six prompts. The first four change **no production code**, with one exception settled in §7 D5; the
fifth is the only one that touches `main.py` and `config/defaults.py`.

**01 — the harness, and one production grid.** One reusable convergence facility in the test tree,
and — because there are currently three disagreeing reproductions of "the production grid" there —
**one** reproduction of the grid `main.py` actually builds. Every later prompt uses both.

**02 — the inventory.** What every accuracy parameter in the pipeline *is*: which object types it
keys, whether it reaches a solver at all, and what the real knob is where it does not. The old plan
assumed this and got it wrong (`RECONCILIATION.md` §2.1); nothing else here is safe until it is
measured. **Stops for the user.**

**03 — audit the adaptive solvers.** `GkNumericIntegration`, `TkNumericIntegration` and
`wavenumber_exit_time`, over the production grids on all three models, across an `atol`×`rtol`
matrix, anchored on radiation. Produces the evidence for the decoupled pairs and recommends them.
**Stops for the user.**

**04 — audit the order-governed targets.** `BackgroundModel`'s $N_\tau$, $N_{c_s\tau}$, $N_F$ and
the WKB sectors' $N_\rho$ have no tolerance to converge (§2 (a)); the convergence test is applied
to the knob they do have, on a background and a break-point set that have both been replaced since
those orders were chosen. **Stops for the user.**

**05 — decouple.** Per-target accuracy parameters and the `main.py` plumbing to carry them, on the
pattern `GkTk-remedial` prompt 12 established for a single constant, with the same `ast`-based
structural guard, widened. The only prompt that touches production code, and the one that
invalidates the datastore.

**06 — `QuadSourceIntegral`, read-only, and the provenance note.** The harness applied to the one
target this campaign does not own; the hand-off; and **`docs/TOLERANCE-PROVENANCE.md`**, the
standing note (§1.2) that says where every accuracy parameter in the pipeline came from.

### 1.1 Explicitly out of scope

Everything in §0.3–§0.5, plus: the datastore migration that prompt 05 implies (regeneration is a
user decision with a compute cost, §7 D2); any change to `GkSourcePolicy`, the rectifier, or the
`*Value` schemas beyond the key columns prompt 05 moves; and the question of whether
`QuadSourceIntegral`'s per-sub-interval `atol` distribution is the right one, which is
`levin-refactor`'s.

### 1.2 The provenance note — the campaign's durable output

A constant whose justification lives only in a campaign log is a constant the next reader will
change by guesswork. The campaign's lasting deliverable is therefore not the numbers but
**`docs/TOLERANCE-PROVENANCE.md`**: one entry per accuracy parameter in the pipeline, and for each
of them —

- **the value**, and the object types it keys;
- **what measurement chose it** — model, wavenumber or grid, geometry, error measure, and the
  numbers, with the reference's own drift beside them (§5 rule 5) and **the grid generation it was
  taken on** (§2 (b));
- **what it is competing against**: the floor that would dominate if it were tightened further
  (§2 (e)), so a later reader can see immediately whether there is anything left to buy;
- **its cost**, in evaluations, at the setting chosen and one step either side, **times the object
  count of the sector it keys** (§2 (c));
- **the campaign, prompt and log** that established it, and the date.

It must cover the parameters this campaign does *not* set as well as those it does — the ones
inherited from `source-remediation` (`DEFAULT_QUADRATURE_ATOL = 1e-32`,
`DEFAULT_QUADRATURE_RTOL = 1e-8`), from `GkTk-remedial` (`DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`,
settled by the user 2026-09-12), from `qcd-background-audit` (`_solve_T_z`'s `xtol=1e-300`,
`rtol=1e-14`), and the ones nobody has ever chosen (`find_phase_extremum`'s `xtol=1e-6, rtol=1e-4`;
`main.py`'s Bessel `phase_atol=1e-12`, `amplitude_rtol=1e-12`; `DEFAULT_LEVIN_THRESHOLD = 1.0`) —
because the point is that *no* accuracy parameter in the pipeline is unexplained when the campaign
closes. Where the provenance of an existing constant cannot be established from the record, the
note says so in those words rather than inventing one.

The note is a *summary with citations*, not a second copy of the measurements: each entry points at
the campaign document that holds the tables. `config/defaults.py`'s own comments stay the primary
record at the point of use — `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` and `DEFAULT_QUADRATURE_ATOL`
already carry that standard and every constant this campaign touches must match it.

---

## 2. Design facts every prompt is built on

**(a) Four of the eight keyed object types have no tolerance to converge, and three of those have
an integer order instead.** Measured, not assumed — `RECONCILIATION.md` §2.1 has the full table:

| Object type | pair | reaches a solver? | the real knob |
|---|---|---|---|
| `wavenumber_exit_time` | shared | **yes** — `root_scalar(xtol=atol, rtol=rtol)` in $\log(1+z)$ | the pair |
| `GkNumericIntegration` | shared | **yes** — DOP853 | the pair |
| `TkNumericIntegration` | `Tk_numeric_atol` + shared `rtol` | **yes** — DOP853 | the pair |
| `BackgroundModel` | shared | no | `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` (all 4) |
| `GkWKBIntegration` | shared | no | `RHO_GAUSS_ORDER` (4), `RESIDUAL_WKB_REGION_MARGIN` (0.5) |
| `TkWKBIntegration` | shared | no | as above |
| `GkSource` | shared | no | none — it assembles, it does not integrate |
| `QuadSourceIntegral` | `quad_atol`/`quad_rtol` | yes | already decoupled; §0.4 |

`GkTk-remedial` prompt 06 replaced the WKB phase ODE with Gauss–Legendre tables and prompts 03/04
did the same to `BackgroundModel`'s three primitives. Those five column pairs survive **only because
they are part of the datastore lookup key** (`ComputeTargets/GkWKBIntegration.py:334-336`;
`BackgroundModel.py:409`; `prompts/GkTk-remedial/RECONCILIATION.md` §2 item 10: "The primitive has
no tolerances. The columns stay… and the payload `metadata` records the Gauss orders actually
used"). **A prompt that proposes to "tighten the WKB tolerance" has misread the tree.**

**(b) Every published figure was taken on a grid production no longer builds, and so is the test
tree's reproduction.** *Corrected at the 2026-09-16 re-anchor: four reproductions, not three, and
one of them is already version 2 — `RECONCILIATION.md` §7.3.* They disagree:
`ComputeTargets/tests/wkb_reference.py:152` (a bare `np.logspace`, version 0),
`ComputeTargets/tests/test_background_segmentation.py:90` (version 1 — break and feature points, no
spacing profile), `ComputeTargets/tests/test_source_grid.py:127` (**version 2**, the full
construction, but private to that module) and `:151` (a version-0 base, deliberate and named).
Production builds it at `main.py:944-963` — the curvature-equidistributed grid,
`SOURCE_GRID_CONSTRUCTION_VERSION = 2`.
`docs/gktk-remedial/tk_numeric_atol_sweep.py:215` imports the first. **A figure in this campaign
carries the generation it was measured on, or it is not a measurement** — and prompt 01 exists so
that there is one right answer to import.

**(c) Cost is a per-object count times an object count, and the two sectors differ by three orders
of magnitude.** `TkNumericIntegration` is one object per $k$ — **50 per model** (`main.py:1215`,
inside the $k$ loop alone). `GkNumericIntegration` is one per $(k, z_{\rm source})$
(`main.py:1770-1791`), the same shape as `GkWKBIntegration`, whose production count is ~65,000 per
model (`GkTk-remedial` README §6). **The decade of `rtol` prompt 17 recommends is free in the
sector it was measured in and is the whole compute decision in the sector it was not.** No prompt
may quote a per-object percentage without the object count beside it.

**(d) The prior — "the error is set by `rtol`" — rests on one clean measurement and one diagonal.**
`GkTk-remedial` prompt 17 §7 holds `atol = 1e-13` and moves `rtol` alone: one decade takes the
worst wavenumber of every model from 2.5e-4/8.6e-4/2.8e-4 to 1.0e-7/7.4e-8/9.8e-7 for +23–25 %
evaluations, while two decades of `atol` fix nothing. That separates the axes. Review §10.1 on
$G_k$ does **not** — its table moves along the diagonal `(1e-10, 1e-8)` → `(1e-13, 1e-11)`, so
"the error is set by `rtol`" there is an interpretation of a two-variable step. Prompt 03 exists to
confirm or destroy the prior in the sector where it has never been tested cleanly.

**(e) `atol` is not uniform across sectors because the quantities are not.** $|G|\sim10^{10}$ at the
hand-over in `Mpc_units`, so an absolute floor of 1e-10 never binds; $|T|\sim10^{-5}$ deep inside
the horizon, so the same floor acts as a $10^{-5}$ *relative* tolerance. That asymmetry is why
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` exists (`config/defaults.py:7-33`), and it is the argument for
decoupling generalised: **a shared absolute tolerance is a statement about magnitudes, and the eight
targets do not share magnitudes.** The same argument applies to `wavenumber_exit_time`, whose
`atol` is an absolute tolerance in $\log(1+z)$ and whose `rtol` is relative to a root of size ~30 —
a quantity unlike either sector's.

**(f) The floors, which are not targets.** $T_k$ numeric: the $T=1,T'=0$ initial condition holds
2.52e-6 of the envelope, $k$-independent, confirmed against the exact $T$ at all 50 wavenumbers
(`GkTk-remedial` prompt 17 §8). $T_k$ WKB: LG truncation, 3.8e-5 of the envelope from $x_i=24$.
Phase: $\varepsilon k\tau$, 3e-7 rad at $k=10^5$ to 9e-4 rad at $3\times10^8$. $G_k$ numeric: the
consumer's cubic spline of the numeric $G$, **1e-5 to 1e-4 of the value near the hand-over** and
"the larger error by two orders" (review §10.1) — the floor that decides whether tightening $G_k$'s
`rtol` buys anything at all. **On `QCDModel` there is no longer a background floor** (§0.3). **An
agent that reports an accuracy below a floor has made an error, and it is a campaign-wide stop** —
with the one qualification §6.1 rule 5 states. A prompt *may* re-measure a floor and supersede an
inherited figure, and for $G_k$'s consumer spline it must (§3.3); what is a stop is a claim below a
floor the prompt has itself just measured, which is an arithmetic error and never a discovery.

**(g) Decoupling is a datastore change, and the plumbing is the risky half.** Every accuracy
parameter is part of its object's lookup key, so a new one makes every existing row of that type
unreachable. `GkTk-remedial` prompt 12 did this once, for one constant on one target, and still
needed a `STRUCTURALLY REQUIRED` deviation to repair a `RayWorkPool` batch that dispatched over the
wrong list once two objects carried different tolerances. `main.py` names the eight targets 22, 14,
29, 16, 20, 13, 123 and 24 times respectively (`GkSource`'s count is inflated by
`GkSourcePolicy*`; the bare `"GkSource"` literal appears 6 times). Prompt 05 must carry the `ast`-based
site-classification guard prompt 12 built
(`ComputeTargets/tests/test_main_plumbing.tk_numeric_tolerance_sites`) **with its predicate
widened**: it matches only class names ending in `"Integration"` today
(`test_main_plumbing.py:544`) and so sees four of the eight. Both numeric factories have since
grown a second key column, `break_point_kind` (`GkTk-remedial` prompt 20), which prompt 05 must not
disturb.

**(h) The analytic anchors are constant-$w$, not radiation-only.** `compute_analytic_G`,
`compute_analytic_Gprime`, `compute_analytic_T`, `compute_analytic_Tprime` all take $w$. Radiation
($w=\tfrac13$) is the production-relevant case and the one the stand-in `RadiationModel` provides,
but a prompt may use another constant $w$ to separate a $w$-dependent error from a solver one.

**(i) Counts, not wall time.** This machine's elapsed times overstate by up to 53 %
(`GkTk-remedial` `IMPLEMENTATION_STATE.md` §5 note 14). Every cost figure in this campaign is
right-hand-side evaluations, integrand evaluations or Hubble calls.

---

## 3. The prompts

| # | Prompt | Covers | Files | Production code? | Model |
|---|---|---|---|---|---|
| 01 | The convergence harness and one production grid | §2 (b), (h); `[00-three-production-grid-reproductions]` | new `ComputeTargets/tests/convergence_reference.py` + its test; `ComputeTargets/tests/wkb_reference.py` (the grid helpers only); folds in `docs/gktk-remedial/tk_numeric_atol_sweep.py` | **No** (test tree only) | Opus |
| 02 | The accuracy-parameter inventory | §2 (a), (c), (g); `RECONCILIATION.md` §2.1 | new `docs/tolerance-convergence/TOLERANCE-INVENTORY.md` and its script | **No** | Opus |
| 03 | Audit the adaptive solvers | §2 (d), (e), (f); review §10.1, §12.5; `[00-gk-numeric-never-swept-and-carries-the-cost]` | new `docs/tolerance-convergence/solver_sweep.py`, `SOLVER-CONVERGENCE.md` | **No** | Opus |
| 04 | Audit the order-governed targets | §2 (a); `[01-convergence-block-has-a-separate-generator]`; `[20-wkb-gauss-orders-not-in-lookup-key]` | `docs/gktk-remedial/residual_convergence.py`, `ComputeTargets/tests/wkb_reference_data.json`, `ComputeTargets/tests/test_background_tau.py`; new `docs/tolerance-convergence/ORDER-CONVERGENCE.md` | **No, but it writes a fixture and a test — §7 D5, settled yes 2026-09-16** | Opus |
| 05 | Decouple | §2 (a), (e), (g); §7 D1 and D3 once settled | `config/defaults.py`, `main.py`, the `ComputeTargets/*Integration.py` and `BackgroundModel.py` constructors, the matching `Datastore/SQL/ObjectFactories/`, the six `extract_*.py` readers, `ComputeTargets/tests/test_main_plumbing.py` | **Yes — the only one** | Opus |
| 06 | `QuadSourceIntegral`, close-out and the provenance note | §0.4, §1.2, §2 (a) | new `docs/tolerance-convergence/TOLERANCE-CONVERGENCE.md`, new **`docs/TOLERANCE-PROVENANCE.md`**; `docs/OPEN_ISSUES.md` | **No** | Opus |

### 3.1 Prompt 01 — the convergence harness, and one production grid

Two things, and the second is why this prompt is not the one the 2026-09-12 plan described.

**The facility.** `GkTk-remedial` prompt 17 needed it and did not have it: its convergence test is
inline in `sweep_model`, so it could not be reused for the second sector without copying. Build it
properly, in `ComputeTargets/tests/` beside `wkb_reference.py` (so it is importable by tests and by
`docs/` scripts alike, and subject to the suite), with at least:

- a converged reference for a given (target, model, $k$, geometry), at a reference tolerance pair
  the caller supplies;
- the **drift** statistic — the same reference one step tighter, differenced in the caller's error
  measure — with the criterion (drift $\le\frac1{10}$ of the smallest difference to be reported)
  evaluated, not just reported. "One step" is a decade for a tolerance pair and **one order** for a
  Gauss order, so the same facility serves prompt 04;
- the **anchor** comparison against every closed form the tree already provides, wherever the
  model is constant-$w$, so that the drift statistic is calibrated at every use and not only in
  prompt 17. **The anchors are not only `compute_analytic_{G,T}{,prime}`** — see the table below,
  which the 2026-09-12 plan did not have and which is why prompts 03 and 04 were scheduled as
  self-convergence work when three of their five quantities have an oracle;
- the envelope-relative, phase and difference error measures, reusing `wkb_reference`'s definitions
  rather than restating them.

**The anchors, in full.** `RadiationModel` (`ComputeTargets/tests/wkb_reference.py:176`) is an
exact $w = c_s^2 = \tfrac13$, $\epsilon = 2$, $H = H_0(1+z)^2$ control, and **every primitive this
campaign audits has a closed form on it**. The harness exposes all of them as anchors, not just the
two solution oracles, and a prompt that reports a self-convergence drift for a quantity in this
table without the oracle error beside it has not calibrated its measurement (§5 rule 5).

| Quantity | Closed form on `RadiationModel` ($s = 1+z$, $a = kc_s/H_0$) | Where | Audited by |
|---|---|---|---|
| $T_k$, $T_k'$ | $2^{n}\Gamma(n+1)(kc_s\tau)^{-n}J_{n}(kc_s\tau)$, $n = \tfrac32 + b$; $3(\sin x - x\cos x)/x^3$ at $w=\tfrac13$ | `analytic_Tk.py:5`, `:19` | 03 (T5) |
| $G_k$, $G_k'$ | `compute_analytic_G`, `compute_analytic_Gprime`, both constant-$w$ | `analytic_Gk.py:5`, `:27` | 03 (T4) |
| $\tau$ | $\tau(z) = 1/(H_0 s)$, and $\Delta\tau(z_a, z_b) = (z_a - z_b)/(H_0 s_a s_b)$ in factored form — the difference of two $\tau$ values loses a digit per decade of baseline ratio, which is the fact the double-double node table exists to defeat | `wkb_reference.py:220`, `:223` | **04 (T7), $N_\tau$** |
| $c_s\tau$ | $\tau(z)/\sqrt3$, with the same factored delta | `wkb_reference.py:236`, `:239` | **04 (T7), $N_{c_s\tau}$** |
| $F$ | $F(z) - F(z_{\rm ref}) = 2\log\big((1+z)/(1+z_{\rm ref})\big)$, evaluated through `log1p` | `wkb_reference.py:243` | **04 (T7), $N_F$** |
| $\theta_G$ | $k\,(1/s_i - 1/s)$ | `wkb_reference.py:247` | 04 (T7) |
| $\rho_G$ | $\equiv 0$: $C = 0$ identically in exact radiation, so the WKB residual **vanishes** — the sharpest possible test of $N_\rho$, since any non-zero answer is pure quadrature error | `wkb_reference.py:251` | **04 (T7), $N_\rho$** |
| $\rho_T$ | $g(s) - g(s_i)$, $g(s) = -2s/\big(\sqrt{a^2 - 2s^2} + a\big) + \sqrt2\,\arcsin\big(\sqrt2\,s/a\big)$, exact | `wkb_reference.py:255`, `:266` | **04 (T7), $N_\rho$** |
| $z_{\rm exit}$ | $1 + z = k/(H_0 e^{N})$ for $N$ e-folds inside the horizon — **elementary**, because $k(1+z)/H = k/(H_0 s)$ when $H \propto s^2$ | *new*; cf. `wkb_reference.py:88` | **03 (T6)** |

Three consequences the prompts must act on, and none of them is optional:

1. **$N_\rho$ has a two-sided oracle.** $\rho_G \equiv 0$ makes the $G_k$ residual a pure error
   measurement with no reference to build, and $\rho_T$ is exact in closed form. Prompt 04 scores
   $N_\rho$ against both before it scores it against a converged reference on any spline model.
2. **$N_\tau$, $N_{c_s\tau}$ and $N_F$ each have an exact primitive and an exact interval
   quantity.** The interval form is the one that matters — §6's error definitions make
   **difference error** relative to the interval, never the absolute — and `tau_delta` /
   `cs_tau_delta` already supply it to ~1e-16 against ~5e-15 for the naive difference. Prompt 04
   uses the delta accessors, not differences of primitives.
3. **`wavenumber_exit_time` has an oracle, and T6 said it had none.** The board records T6 as
   "never measured"; the reason it was never measured is that nobody noticed the radiation case is
   a one-line inversion. Prompt 03 scores the production `root_scalar` in $\log(1+z)$
   (`CosmologyConcepts/wavenumber.py:979-984`) against $k/(H_0 e^{N}) - 1$ directly, at every
   production $k$ and at each `efolds_subh` the pipeline asks for, **before** it reports anything
   about `xtol = 1e-10, rtol = 1e-8`. Confirmed at the rebase: `horizon_exit_z` agrees with the
   closed form to 2.3e-16 relative or better over $k \in [10^3, 3\times10^8]$ and
   $N \in \{-3, 0, 4\}$.

**The one validity bound.** $\rho_T$'s primitive requires $\omega_T^2 > 0$, i.e.
$1 + z < k/(\sqrt6\,H_0)$ — the mode must be sub-horizon — and `_rho_T_primitive` raises rather
than returning a complex root. A prompt that walks an anchor outside that bound has chosen its
$z_{\rm init}$ wrongly; it is not a finding about the representation.

**The grid.** *Amended at the 2026-09-16 re-anchor (additively, §5 rule 7); the paragraph below
replaces a version that said "three" and "build".* There are **four** reproductions of "the
production source grid" in the tree and they disagree by construction generation (§2 (b)):

| # | Site | Generation |
|---|---|---|
| 1 | `ComputeTargets/tests/wkb_reference.py:152` `production_source_grid` | **v0** — bare `logspace`, citing a `main.py:410-419` that has not been the grid code for two campaigns |
| 2 | `ComputeTargets/tests/test_background_segmentation.py:90` `production_source_grid` | **v1** — `break_z` / `feature_z`, no `spacing` |
| 3 | `ComputeTargets/tests/test_source_grid.py:127` `_production_grid` | **v2** — the full production construction, already correct, already under the suite, and **private** |
| 4 | `ComputeTargets/tests/test_source_grid.py:152` `_production_base_grid` | v0, deliberately and named — it mirrors `main.py:944`'s own base-grid step, which the spacing profile is measured on |

**So the task is to hoist, not to build.** #3 is the version-2 grid and nothing outside its own
module can reach it; that is now the whole defect. Move it into a module the other sites can
import, and repoint #1 and `docs/gktk-remedial/tk_numeric_atol_sweep.py:215` at it. #4 stays where
it is, named, because `main.py:944-963` has the same two-stage structure — a base grid, then a
spacing profile measured on it, then `populate_source_grid`.

`main.source_grid_spacing_profile` and `main.cosmology_feature_redshifts` are the functions that
build it and `main.py` cannot be imported, so lift them with
`ComputeTargets/tests/test_main_plumbing.load_main_py_functions` — the mechanism #2 and #3 both
already use, and #3's `extra_globals` block is the worked example of the twelve module-level
constants that lift needs. **`cosmology_feature_redshifts` changed under the re-anchor**: it asks
the cosmology for `z_matter_radiation_equality` and `z_matter_lambda_equality` and raises
`RuntimeError` if either is missing, with no fallback and on purpose. A stand-in that cannot answer
both will hit that raise; `test_source_grid.py:189` already carries one that can, and prompt 01
reuses it rather than writing a second. Keep the version-0 and version-1 constructions available
and *named* — prompt 17's figures were taken on version 0 and `test_background_segmentation`'s
assertions on version 1, and neither should be silently re-scored.

**Acceptance, in two parts.** (i) The harness reproduces `GkTk-remedial` prompt 17's published
$T_k$ drift figures **exactly, on prompt 17's version-0 grid** — that is the construction check,
and it is why the campaign starts here. (ii) It reports the same statistic on the **version-2
production grid** beside it, and the log records the difference. A harness that can only reproduce
the old number has not been shown to measure the tree.

It must need no Ray and no datastore (call remotes through their undecorated `_function`, use
`wkb_reference.py`'s stand-ins).

### 3.2 Prompt 02 — the accuracy-parameter inventory

The prompt the old plan did not have, and the reason it miscounted its own subject.

Enumerate, **by reading the tree rather than the documents**, every accuracy parameter that reaches
a numerical method or a lookup key: the five `config/defaults.py` constants, the five
`*_GAUSS_ORDER`/margin constants, and every hard-coded `atol=`/`rtol=`/`xtol=` literal in
production code (`LiouvilleGreen/integration_tools.py:95`,
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:579`, `:864`, `:1008`, `main.py:1119`, `:1127`,
`AdaptiveLevin/`). For each: which object types it keys; whether the value reaches a solver, a
lookup key, both or neither; what kind of method consumes it; the object count of the sector; and
where its provenance is recorded, or that it is not.

> **Amended 2026-09-16 (additively, §5 rule 6): the three `LambdaCDM_GenericEOS.py` anchors above
> are stale and all three sites are already settled.** `prompts/background-solver-robustness` ran
> on that file: `:579` is now `:636` (unchanged at `xtol=1e-300, rtol=1e-14`), `:864` **left
> production code entirely** for `CosmologyModels/tests/T_z_reference.py:285`, and `:1008` is now
> `:1137` at `xtol=1e-300, rtol=8.9e-16`, bracketed. Provenance for all three, in this section's
> own shape, is
> [`prompts/background-solver-robustness/PROVENANCE.md`](../background-solver-robustness/PROVENANCE.md) —
> **lift it rather than re-derive it**, and note that it records the crossing probe's tolerance as
> one nobody ever chose, which §1.2 asks be said in those words.

Answer explicitly: **is §2 (a)'s table complete and right?** Which parameters are live, which are
vestigial, and which are vestigial *in the computation* but load-bearing *in the key*? Produce
`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` and the script that regenerates it, so that a
later reader can re-run it rather than re-read `main.py`. No measurement of accuracy here — this
prompt establishes what there is to measure, and prompts 03 and 04 measure it.

### 3.3 Prompt 03 — audit the adaptive solvers

The three targets whose parameter reaches an adaptive method, over a matrix in `atol` and `rtol` —
the point being that prompt 17 held `rtol` fixed and so could only see one axis, and that review
§10.1 moved both at once and so could not separate them.

- **`GkNumericIntegration`**, three models, the production response grids. This is the sector that
  has never been swept and carries ~65,000 objects per model (§2 (c)), so it is the prompt's
  centre of gravity, not its second half. Measure the consumer-spline floor of §2 (f) alongside the
  solver error: if the spline dominates by two orders, the recommendation is "do not tighten", and
  that is a result.
- **`TkNumericIntegration`**, three models, the production source grids, re-taken on the version-2
  grid, under the sector's own `BREAK_POINT_ALL` policy — prompt 17's sweep ran at the module
  default, `BREAK_POINT_DISCONTINUITY`, which is the *other* sector's.
- **`wavenumber_exit_time`**, whose `root_scalar` in $\log(1+z)$ nobody has measured. It fixes
  where every grid begins and every horizon-relative cut sits; a misplaced $z_{\rm exit}$ moves the
  grid, not just a value.

Report per (target, model, $k$, `atol`, `rtol`): maximum, second-largest and median error in the
target's own measure, the location of the maximum, and the evaluation count; and per (target,
model) the distribution over the grid, since prompt 17 established that a single $k$ is not
characteristic. Every figure carries its reference's drift and its grid generation.

Answer explicitly: **is §2 (d)'s prior right in the $G_k$ sector?** What does each target cost at
the setting that first reaches its floor, in evaluations **times objects**? And what would the
decoupled pairs be, with the evidence? Recommend; do not decide (§7 D1).

### 3.4 Prompt 04 — audit the order-governed targets

`BackgroundModel`, `GkWKBIntegration` and `TkWKBIntegration` have no live tolerances (§2 (a)). The
same convergence test applies to the knob they do have: $N_\tau$, $N_{c_s\tau}$, $N_F$, $N_\rho$,
all currently 4, and `RESIDUAL_WKB_REGION_MARGIN = 0.5`, which is in no key, label or tag at all
(`[20-wkb-gauss-orders-not-in-lookup-key]`).

**The evidence for all four orders is stale, and taking it back is this prompt's first job.**
`ComputeTargets/tests/wkb_reference_data.json`'s `convergence` block records them, and it was
generated 2026-09-10, on the `T(z)` representation `qcd-background-audit` prompts 04–06 replaced
and against a break-point set prompt 07 removed — its `decision.recommended_scheme` is
`"branch+knots"`, and the knots do not exist any more. `RECONCILIATION.md` §5 has the detail;
`[01-convergence-block-has-a-separate-generator]` is the issue, and it is **assigned here** because
this is the first prompt anywhere whose charter is the orders themselves.

So: re-run `docs/gktk-remedial/residual_convergence.py` against the corrected background and the
3-point break set — which means updating its scheme sweep, since one of its three schemes is
gone — regenerate the block, and re-measure `QCD_BREAK_POINT_ALIGNMENT_TOL` in
`ComputeTargets/tests/test_background_tau.py`, which stands at 1.5e-04 against an original 1.4e-05
for no reason but the block's age. **Put it back if it will go; if it will not, that is a finding
about the representation and the log says so in those words.**

Then the sweep: order against the analytic anchors and the converged-reference drift, on all three
models, at every production $k$. Answer: are the four orders converged at 4 at every production
$k$, or was `GkTk-remedial` prompt 02 right at the $k$ it measured and lucky at the rest? What does
`RESIDUAL_WKB_REGION_MARGIN` buy, measured rather than assumed? And the API question the campaign
inherits: the `atol`/`rtol` columns on these three targets are part of the lookup key and describe
nothing — **recommend** what replaces them, with the schema-churn cost (§7 D3). No `main.py` and no
`config/defaults.py` either way.

### 3.5 Prompt 05 — decouple

The only `main.py` change in the campaign, and only after the user has settled D1 and D3. Per-target
constants in `config/defaults.py` with the same standard of comment
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` already carries — the measurement that chose the number, in the
file, where the next reader will find it. For the order-governed targets, the key column becomes the
order (§7 D3), which closes `[20-wkb-gauss-orders-not-in-lookup-key]`.

Then the plumbing: one accuracy object per target in the same `ray.get`, **every** `object_get` of
that target switched, and the `ast`-based guard widened past `"…Integration"` so that it enumerates
all eight and fails when an unclassified site appears (§2 (g)). Expect the batch-dispatch hazard
prompt 12 hit; the guard exists because it is silent. The six `extract_*.py` scripts read the same
constants (`extract_Gk_data.py:298`, and five more) and must follow, or they will query for rows
that no longer exist.

### 3.6 Prompt 06 — `QuadSourceIntegral`, close-out, and the provenance note

Apply the harness to `QuadSourceIntegral` **without editing it or anything else in §0.4**: is it
converged at `quad_atol = 1e-32`, `quad_rtol = 1e-8` on the production configuration, measured
against the analytic oracle the `source-remediation` campaign used? Report to `levin-refactor` and
`qsi-phase-groups` through `docs/OPEN_ISSUES.md` §1.2/§1.3 rather than acting. Then the campaign
document: the eight targets, the parameters each ended at, the evidence, and the floors each is now
limited by.

Then **`docs/TOLERANCE-PROVENANCE.md`** to §1.2's specification — the deliverable the user asked for
by name, and the one that outlives the campaign. It is written last because only then is every
number in it measured; but prompts 02, 03, 04 and 05 must each leave their log's "State handed to
the next prompt" carrying the five fields §1.2 lists for every parameter they touched, so that this
prompt assembles rather than re-derives. A prompt that settles a parameter without recording its
provenance has not finished.

---

## 4. Dependencies and ordering

```
01 ──▶ 02 ──▶ 03 ──▶ [user settles D1] ──┐
          └──▶ 04 ──▶ [user settles D3] ──┴──▶ 05 ──▶ 06
```

02 must precede both audits: it is what says which targets 03 and 04 each own, and the old plan's
allocation was wrong (`RECONCILIATION.md` §2.1). 03 and 04 are otherwise independent; run 03 first,
since D1 is the decision with a compute cost attached. 05 must not start until both D1 and D3 are
settled — it is the prompt that invalidates the datastore, and settling a parameter afterwards would
invalidate it twice.

### 4.1 Natural stopping points

After **02**, **03** and **04**, always: each ends in a recommendation the user must accept before
anything is changed. After **05**, because the datastore regeneration is a compute decision
(§7 D2).

### 4.2 Relationship to the campaigns that closed before it

Both preconditions are met and the branch is cut (§0.3). What this campaign inherits is not a
blocker but a **bill**: prompt 05 makes every row of the retuned targets unreachable by its old
key, and the datastore built by `GkTk-remedial` prompt 13 and re-built through
`qcd-background-audit`'s prompts 03, 11, 14 and 15 will be superseded again in the sectors it
retunes.

That cost is accepted deliberately rather than by oversight; the user confirmed the ordering on
2026-09-12 (D2) and `qcd-background-audit` restated the principle campaign-independently
(`21d80b2`: regeneration cost is not a constraint, everything is development, and a superseded
datastore retains archival value). **Growing the object count is the intended outcome here, not a
regression**: the whole point is that each quantity carries its own justified parameter, and
distinct parameters should produce distinct objects. No prompt may argue for a shared constant on
the grounds that decoupling multiplies rows.

### 4.3 Orchestration

As `prompts/GkTk-remedial/README.md` §4.3 and `orchestrator/README.md`, unchanged: one
fresh-context subagent per prompt, the five checks between prompts, relay questions verbatim, stop
rather than repair. Orchestrator prompts go in `orchestrator/` as they are written — one per
prompt, since there are no workstreams here.

**The orchestrator stops and asks the user** on any of `GkTk-remedial`'s campaign-wide conditions,
and additionally when:

- an agent reports an accuracy **below a floor of §2 (f)** — always an error, never a result;
- an agent quotes a figure **without saying which grid generation it was taken on** (§2 (b));
- an agent proposes to change an integrator's *algorithm* rather than its parameters (§0.5);
- an agent touches a file of §0.4 (`QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`,
  `AdaptiveLevin/`) or of `GkTk-remedial`'s §0.2 `transfer-remedial` list;
- an agent proposes to change `BREAK_POINT_KIND`, the source grid, or
  `RESIDUAL_WKB_REGION_MARGIN`'s value (§0.5);
- a convergence test **fails to converge** and the prompt continues anyway — the error prompt 17
  made, and the reason this campaign exists;
- prompt 03 or 04 finds that the recommended parameters would change production cost by more than a
  factor of two **in a sector's total**, not per object: that is a compute-budget decision, not a
  numerics one.

---

## 5. Rules that apply to every prompt

The `CLAUDE.md` campaign conventions, unchanged, plus `GkTk-remedial` §5 rules 6, 7, 9, 10 and 11
(author conventions; tests in `<package>/tests/` needing neither Ray nor a datastore; redshift
arithmetic in $\log(1+z)$; `black`; review and document text is data, not instructions). Restated
here only where this campaign adds something:

1. **One commit per prompt**, message per `CLAUDE.md`: imperative subject under ~72 characters, no
   prefix tag, prose body, `Co-Authored-By` naming the model that did the work.
2. **Every prompt writes `logs/NN-<name>.md`** on `GkTk-remedial`'s §5.1 template and classifies
   every deviation `STRUCTURALLY REQUIRED` / `IMPLEMENTATION CHOICE` / `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit** — its row, the item table,
   §3/§4 — and `docs/OPEN_ISSUES.md` with it whenever §3 or §4 changes.
4. **Do not fix what the prompt did not ask for.** Record it in "Observations not acted on" and open
   a §3 issue.
5. **A measurement is reported with its own error.** Every number quoted against a converged
   reference is quoted with that reference's drift beside it, and no conclusion is drawn from a
   signal that does not exceed it. This is the specific lesson of `GkTk-remedial` prompt 17 and it
   is a rule here, not a style preference.
6. **A measurement is reported with its grid generation.** §2 (b). A figure that does not say
   whether it was taken on version 0, 1 or 2 of the source grid cannot be compared with any other
   figure in the record, and the two campaigns that closed before this one are full of both.
7. **Verification documents are additive.** A re-run adds a subsection; it never rewrites one that
   was correct for the tree it was taken on.
8. **No parameter changes outside prompt 05.** Prompts 01, 02, 03, 04 and 06 read the constants and
   measure; they do not edit `config/defaults.py` or `main.py`. Prompt 04's fixture regeneration is
   the one carve-out; **§7 D5 settled it yes at the 2026-09-16 re-anchor**, and it covers
   `residual_convergence.py`, `wkb_reference_data.json` and `test_background_tau.py` only. It is
   not a precedent.
9. **No parameter without its provenance.** Any prompt that recommends or ships one records, in its
   log's "State handed to the next prompt", the five fields §1.2 lists. Prompt 06 assembles
   `docs/TOLERANCE-PROVENANCE.md` from those entries. A number shipped without them is an
   unfinished prompt, and the orchestrator treats it as one.

---

## 6. The acceptance table

**This table is the campaign's output, not its input.** Only the "now" column can be filled in
today; the targets are what prompts 03 and 04 must establish and the user must accept (§7 D1, D3).
A prompt that invents a target for its own row has skipped the decision.

Every "now" figure carries the **grid generation** it was measured on (§2 (b)). Three of the eight
rows are figures taken on a grid production no longer builds; re-scoring them is part of the work,
not a preliminary to it.

### 6.1 The target rule — how a floor becomes a target

A target is not invented, but neither is it free: **the rule below fixes it, and a prompt applies
the rule rather than choosing a number.** This is what §6's empty Target column is waiting for, and
it exists because the two statements "below the floor is an error" (§2 (f)) and "at the floor is
enough" are not the same, and the campaign previously wrote down only the first. `GkTk-remedial`
§6 made the choice implicitly and inconsistently — $T_k$ numeric at 3e-6 against a 2.5e-6 floor
($1.2\times$), $\theta_G$ at 1e-5 rad against a 3e-7 floor ($33\times$) — and labelled the whole
table "engineering targets… not certified bounds". That precedent is not guidance; this is.

**The rule.** For each row:

1. **Measure the dominating floor first**, on the tree the campaign is actually running on, and
   report it with its own uncertainty. Do not inherit it from a review paragraph — see rule 5
   below.
2. **The target is the loosest setting whose error is at or below that floor**, measured over the
   whole production grid on all three models, not at a representative $k$. "At or below" means the
   row's own error measure (§6's definitions: envelope-relative, phase in radians, difference
   relative to the interval), scored at the **maximum** over the grid, not the median — prompt 17
   established that a single $k$ is not characteristic and that the distribution has a tail three
   orders above its centre.
3. **Loosest, not tightest.** The parameter is swept from loose to tight and the target is the
   *first* setting that clears the floor, with the cost recorded **at that setting and one step
   either side** (§1.2). A setting two decades tighter than the one that first clears buys nothing
   and costs the sector's object count; recommending it is an error of the same kind as
   recommending one that misses.
4. **Where the error is already below the floor, the target is `unchanged`** — written in the cell
   in that word — and the row records **the factor by which the floor dominates**. This is a
   *result*, not a non-result, and it is the expected outcome for `GkNumericIntegration` if review
   §10.1's two-order claim survives prompt 03's re-measurement. No prompt may tighten a parameter
   whose error the floor already swamps, however cheap the tightening looks.
5. **A floor may be re-measured; only a claim against a *freshly measured* floor is a stop.**
   §2 (f) declares an accuracy below a declared floor a campaign-wide stop. That rule is about
   *claims*, not about floors: a prompt is expressly permitted — and for $G_k$'s consumer spline,
   required (§3.3) — to measure a floor itself and supersede the inherited figure, recording both.
   What remains a stop is reporting an accuracy below the floor the prompt has *just measured*,
   because that is an arithmetic error in the measurement and never a discovery. The inherited
   floors of §2 (f) are quoted with their provenance in the Floor column so that a reader can see
   which have been re-taken on this tree and which have not.
6. **Where no floor can be established, there is no target, and the row says so in those words.**
   `wavenumber_exit_time` may be such a row: its consumer is the grid construction rather than a
   value, so what bounds it is a displacement the grid can absorb, not an error in a quantity.
   Prompt 03 states what that bound is, or states that it could not establish one — and in the
   second case the parameter is left where it is and the provenance note records *unestablished*
   rather than inventing a justification (§1.2's closing rule).

**The rule does not decide cost.** A target that clears the floor but moves a sector's total by
more than a factor of two is still a compute-budget decision for the user (§4.3), and the prompt
recommends without deciding. The rule fixes what "good enough" means; it does not fix what the
campaign can afford.

### 6.2 The table

| Target | Parameter today | Accuracy now, and where it was measured | Floor (§2 (f)) | Target | Prompt |
|---|---|---|---|---|---|
| `GkNumericIntegration` | `atol = 1e-10`, `rtol = 1e-8` | 2.3e-7 of envelope — **radiation and LambdaCDM only, at four source redshifts, on an `(atol, rtol)` diagonal** (review §10.1, **v0**). No QCD, no grid sweep, no drift figure at production tolerances. **The sector with ~65,000 objects per model has never been swept** | consumer spline of numeric $G$, 1e-5–1e-4 near the hand-over (review §10.1) | — | 03 |
| `TkNumericIntegration` | `atol = 1e-13`, `rtol = 1e-8` | 3 / 13 / 8 of 50 $k$ above 3e-6 of envelope on Radiation / LambdaCDM / QCD; worst 8.64e-4; median-of-per-$k$-maxima 3.8e-7 / 4.5e-7 / 1.0e-6 (`GkTk-remedial` prompt 17, all 50 $k$, three models, **v0**, and under `BREAK_POINT_DISCONTINUITY` rather than the sector's own `BREAK_POINT_ALL`) | 2.52e-6, initial condition | — | 03 |
| `wavenumber_exit_time` | `xtol = 1e-10`, `rtol = 1e-8` in $\log(1+z)$ | **never measured** | — | — | 03 |
| `BackgroundModel` | `atol`/`rtol` vestigial; $N_\tau = N_{c_s\tau} = N_F = 4$ | $\tau$ 2.611e-16, $\tau_s$ 2.204e-16, $F$ 2.440e-16 relative at three nodes of one scoped LambdaCDM run (`docs/gktk-remedial-verification.md` §4.3). The **orders** were chosen on a background and a break-point set that no longer exist (`RECONCILIATION.md` §5) | double-precision accumulation over the grid | — | 04 |
| `TkWKBIntegration` | none live (§2 (a)); $N_\rho = 4$ | $T_{\rm WKB}$ radiation control 3.8e-5 of envelope from $x_i=24$ (`GkTk-remedial` prompt 07) | LG truncation | — | 04 |
| `GkWKBIntegration` | none live (§2 (a)); $N_\rho = 4$ | $\theta_G$ 13.9 rad at $k=10^5$ and 7366 rad at $3\times10^8$ against target ≤1e-5 / ≤5e-3 rad (`GkTk-remedial` README §6, pre-remediation baseline) | $\varepsilon k\tau$ | — | 04 |
| `GkSource` | `atol`/`rtol` stored, never read | n/a — it assembles | — | — | 02 |
| `QuadSourceIntegral` | `quad_atol = 1e-32`, `quad_rtol = 1e-8`, decoupled already | `atol` chosen against an analytic oracle by `source-remediation` log 12; `rtol` confirmed non-binding (1e-8 → 1e-11 bit-identical on 159 items) | — | read-only | 06 |

Error definitions are `GkTk-remedial` README §6's, unchanged and deliberately: **envelope-relative**
divides by the local Liouville–Green envelope, not by the value; **phase error** is absolute
radians of the unwrapped phase at the supplied double $z$; **difference error** is relative to the
interval quantity, never to the absolute.

---

## 7. Decisions left to the user

**D1 — the decoupled tolerance pairs.** *Partly settled.* The user confirmed on 2026-09-12 that
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` **stays at `1e-13`** — `GkTk-remedial` prompt 17's
recommendation, accepted — so the $T_k$ absolute tolerance is not in question and no prompt here
revisits it. What is open is every **`rtol`**, `GkNumericIntegration`'s `atol`, and
`wavenumber_exit_time`'s pair, none of which has ever been measured on a grid. Prompt 03 recommends
with evidence; prompt 05 may not start until the user accepts.

**The likely shape has changed since 2026-09-12** and the change is the point. From §2 (d), one
decade of `rtol` costs +23–25 % evaluations. In the $T_k$ sector that is 50 objects per model and
free. In the $G_k$ sector it is ~65,000 objects per model — and $G_k$'s solver error may already be
two orders below the consumer spline that reads it (§2 (f)), in which case the right answer is to
tighten nothing and say so with the measurement in hand. **That is the decision prompt 03 exists to
inform, and it is the reverse of the one the 2026-09-12 plan expected.**

**D2 — the datastore. *Settled 2026-09-12: not a problem, and the ordering stands.*** Prompt 05
makes every row of the retuned targets unreachable by its old key, so the store grows a parallel
set of objects. The user has confirmed that this is the **intended outcome, not a cost to be
minimised**, and `qcd-background-audit` restated the principle campaign-independently (§4.2). The
numeric-sector and background rows of the datastore those campaigns built are expected to be
superseded.

**D3 — what replaces the vestigial key columns.** `BackgroundModel`, `GkWKBIntegration` and
`TkWKBIntegration` carry `atol`/`rtol` columns that are part of the lookup key and describe nothing
(§2 (a)), while the integer orders that *do* set their accuracy are in no key, label or tag
(`[20-wkb-gauss-orders-not-in-lookup-key]`). **This is the user's stated target for the campaign:
for a Liouville–Green-type representation the key should carry an order, not a tolerance.** The
options are keep (schema churn avoided, three permanently misleading column pairs, and an order
change that silently serves a stale row), drop (a migration), or **replace with the orders** (a
migration, and the defect closed). Prompt 04 recommends with the schema-churn cost of each; prompt
05 implements. `GkSource` is the fourth vestigial case and is a different question — it integrates
nothing, so there is no order to put there.

**D4 — whether `QuadSourceIntegral` is in or out.** §0.4 puts it out, read-only, because two other
campaigns own those files. If it should instead be retuned here, that has to be agreed with those
campaigns first, and prompt 06 changes character entirely.

**D5 — may prompt 04 write a fixture and edit a test? *Settled 2026-09-16 at the re-anchor:
YES.*** The user accepted it: prompt 04 **may** re-run
`docs/gktk-remedial/residual_convergence.py`, write `ComputeTargets/tests/wkb_reference_data.json`,
and re-measure `QCD_BREAK_POINT_ALIGNMENT_TOL` in `ComputeTargets/tests/test_background_tau.py`.
§5 rule 8's carve-out is live rather than proposed, and
`[01-convergence-block-has-a-separate-generator]` has for the first time a prompt allowed to close
it. **The carve-out is exactly those three files and no others**; anything further is a stop under
§4.3. The argument that was put, and accepted, follows. The
Gauss orders' evidence is stale and the only way to take it back is to re-run
`docs/gktk-remedial/residual_convergence.py`, write
`ComputeTargets/tests/wkb_reference_data.json`, and re-measure `QCD_BREAK_POINT_ALIGNMENT_TOL` in
`ComputeTargets/tests/test_background_tau.py` (`RECONCILIATION.md` §5). That breaks §5 rule 8's
"the audit prompts change nothing", which is why it is put here rather than assumed. The
alternative — audit the orders against evidence known to be two representations out of date — is
not an alternative. `[01-convergence-block-has-a-separate-generator]` has already been declined by
two prompts of another campaign on exactly this scope argument, and it will keep being declined
until some prompt is given the files.
