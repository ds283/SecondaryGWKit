# Implementation state — Gk/Tk WKB phase remedial campaign

**Campaign:** [`README.md`](README.md) · **Source review:** [`docs/gk-wkb-review-fable-2026-09-09.md`](../../docs/gk-wkb-review-fable-2026-09-09.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md)
**Baseline commit:** `9ff59d5` (`main`, clean)
**Last updated:** 2026-09-11 — Prompt 10 landed (the transfer-function *consumer* is on the tables
too: `TkSourceFunctions.phase` is a `PrimitivePhase` with `leading = cs_tau`, `z_anchor = z_init`,
`sign = +1`, and `friction(z)` is `friction_F.delta(crossover_z, z)` read exactly from the
background table with the stored samples kept only as a construction-time cross-check.
`PHASE_SPLINE_CHUNK_LOGSTEP` is gone and no production path builds a `phase_spline` any more. On a
$w=1/3$ fixture reaching $x_T=10^6$ on the production 100/decade grid the phase error falls from
**7.0854e-3 rad** (a cubic spline of the same stored samples) to **3.4482e-10 rad**, a ratio of
**2.055e7**, and the 3.4e-10 is 2.96 ulp of the $10^6$ rad phase — the `div * TWO_PI`
representation floor. Friction matches its closed form to **8.9e-16** absolute. Two deviations went to the
user, and both are now settled. **Prompt 10 §3 item 3's 1e-10 relative on `omega` vs `theta_deriv`
is missed at one abscissa per equation of state** — 1.0492e-10 at $w=1/3$, the not-a-knot end
condition of the residual spline at the top of the WKB region, 5.6e-12 from the fifth sample
inwards, against 4.249e-08 before: the shipped 1e-9 / 1e-11 window pair stands and
`[10-residual-spline-end-condition]` is **left open and assigned to prompt 13**, which re-measures
it on the real background before anyone pays for a quintic. **Five lines of stand-in construction
in `ComputeTargets/tests/test_quadsource_integral.py`, outside the prompt's file list** are
**accepted** — the orchestrator confirmed by restoring the file that §1's friction cross-check
fails there on a genuinely inconsistent fixture — and the prompt's file list is extended in place
with the reasoning (`[10-quadsource-fixture-model-substitution]` resolved, §4).
`[00-transfer-remedial-test-file-overlap]` is **resolved**: `test_phase_groups.py` needed no edit
at all — it is byte-identical to the pre-Workstream-D snapshot — and every line prompt 10 removed
from `test_tk_source_functions.py` blames to `e3348e4`, not to `8ba9159`, so those tolerances are
intact. **Workstream D is complete: the campaign's accuracy claims now hold end to end.**)
Prompt 09 landed (the Green's-function *consumer* stops splining the
growing phase: `ComputeTargets/primitive_phase.py` evaluates
$\theta=-k\,\tau.\mathrm{delta}(z_s,z_r)+\varphi$ from prompt 03's double-double table with a cubic
spline of the small residual $\varphi$ alone, and both `GkSourcePolicyData` call sites — the
`Levin_z` threshold test and `GkSourceFunctions.phase` — now build one. On the review §5 consumer
geometry scaled to $k=10^8$ the error falls from **7.286e-3 rad** to **4.189e-8 rad**, a ratio of
**1.739e5**, and the 4.189e-8 is 2.81 ulp of the 9.09e7 rad span — the $\varepsilon k\tau$
representation floor, not the method's error; $\varphi$ itself is recovered to 2.157e-10 rad. The
`GkSource` rectifier is verified on a faithful copy: 90 of 990 swept objects carry a $+1$-cycle
step at a stop-point transition, the rectifier repairs every one, and after it $\varphi$ is
constant to 3.6e-12 rad; on pure-WKB objects it makes **zero** corrections. **Prompt 09 §4 test
1's 1e-8 rad threshold is below the double-precision floor of its own geometry** — 0.67 ulp — and
was replaced by README §6's 1e-6 rad plus a 6-ulp bound. The orchestrator stopped on that, per
README §4.3; an independent Fable review
([`reviews/09-prompt-09-review-fable.md`](reviews/09-prompt-09-review-fable.md)) accepted the
implementation, confirmed the arithmetic and recommended amending the prompt text, which was done —
`[09-consumer-threshold-below-representation-floor]` is **resolved** (§4).)
Prompt 08 landed (`phase_spline` chunking is gone: one cubic spline
over the whole sample, rebased at the sample's median `theta_div_2pi` rather than selected between
several by a hard switch; `chunk_step`/`chunk_logstep`/`increasing` are accepted no-ops, so
`bessel_phase.py` and the three test fixtures that still pass `chunk_logstep=125` needed no edits;
`num_chunks` always reports 1. The interior interpolation error the review measured, 8.2566e-05 rad at
$k=10^6$ on the `GkSourcePolicyData` geometry (review §5: 8.26e-5), is confirmed unchanged from
before de-chunking —
chunking bought nothing and cost ordinates, knot residuals and a switch discontinuity, all now
gone. Workstream D may proceed to prompt 09.) Workstream C closed; the `transfer-remedial` merge confirmed at `e01c31d`, so Workstream D may start (`[00-transfer-remedial-test-file-overlap]`). Prompt 07 landed (the transfer-function phase and friction now come from the tables too: `friction_RHS` and its state index are gone from the producer and live only in prompt 04's test, `store()`'s no-op sign fix and its cross-sample rebase are gone, and the stored $\theta_T$ is 1.5e-8 rad at $k=10^5$ and 9.2e-5 rad at $3\times10^8$ against prompt 01's references, where the ODE was 2.01 rad and 5.1e3 rad. Review §12.4's LG truncation table reproduced to two figures. **Its cost, 0.049–0.052 s per object, straddles prompt 07 §3 item 6's 0.05 s** — `[07-tk-per-object-cost-is-all-setup]`.) Prompt 14 landed (the residual table is built once per $(model, k, sector)$ on the background grid and memoised in the worker: 142.5 (LambdaCDM) / 163.0 (QCD) residual-integrand evaluations per object over 50 objects of one $k$ against 6,924 / 7,908 — 49× — and 0.0010 s per object at $k=3\times10^8$ against 0.0309 s; $\theta$ bit-identical at every sample of fifteen (model, $k$, sector) cases). Prompt 06 landed (the Green's-function WKB phase is now $-[k\,\Delta\tau+\Delta\rho]$ from the tables: the two-stage phase ODE, the `Q` variable, the resets, the sign fix and the cross-sample rebase are gone; stored phases at the floor; `TkWKBIntegration.compute()` switched, its `store()` awaits 07).

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA, model and log link, update the mechanism-level
> table in §2, and add or clear entries in §3 (Active issues). Do not edit rows other than your own
> except to close an issue you resolved. **Any change to §3 or §4 must also update the project-wide
> index [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (see `CLAUDE.md`).

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — measurement and prototypes

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [Reference harness and prototype](01-reference-harness-and-prototype.md) | review §7, §13.3 | Opus | ⚠️ | *"Add WKB phase references and a measured primitive prototype"* (SHA not embedded, per the campaign convention) | [`logs/01-reference-harness-and-prototype.md`](logs/01-reference-harness-and-prototype.md) |
| 02 | [QCD residual convergence](02-qcd-residual-convergence.md) | review §11, §12.7 | Opus | ⚠️ | *"Measure Gauss-order convergence of the WKB primitives on both models"* (SHA not embedded, per the campaign convention) | [`logs/02-qcd-residual-convergence.md`](logs/02-qcd-residual-convergence.md) |

### Workstream B — the primitives in `BackgroundModel`

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 03 | [τ primitive](03-tau-primitive.md) | review §7, §13.2, §13.3 | **Fable** | ⚠️ | *"Build conformal time as a double-double Gauss-Legendre table"* (SHA not embedded, per the campaign convention) | [`logs/03-tau-primitive.md`](logs/03-tau-primitive.md) |
| 04 | [Sound-horizon and friction tables](04-sound-horizon-and-friction-tables.md) | review §12.7 | Opus | ⚠️ | *"Tabulate the sound horizon and the LG friction integral per model"* (SHA not embedded, per the campaign convention) | [`logs/04-sound-horizon-and-friction-tables.md`](logs/04-sound-horizon-and-friction-tables.md) |

### Workstream C — the producers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 05 | [Phase residual](05-phase-residual.md) | review §6, §12.2, §12.4 | Opus | ⚠️ | *"Add the WKB phase residual as a per-k table"* (SHA not embedded, per the campaign convention) | [`logs/05-phase-residual.md`](logs/05-phase-residual.md) |
| 06 | [Gk WKB phase from the primitive](06-gk-wkb-phase-from-primitive.md) | review §2–§4, §8, §13.4 | **Fable** | ⚠️ | *"Compute the Green function WKB phase from the conformal-time table"* (SHA not embedded, per the campaign convention) | [`logs/06-gk-wkb-phase-from-primitive.md`](logs/06-gk-wkb-phase-from-primitive.md) |
| 14 | [Residual table reuse](14-residual-table-reuse.md) | §3 `[06-residual-table-per-object]` | Opus | ⚠️ | *"Build the WKB phase residual once per wavenumber"* (SHA not embedded, per the campaign convention) | [`logs/14-residual-table-reuse.md`](logs/14-residual-table-reuse.md) |
| 07 | [Tk WKB phase from the primitive](07-tk-wkb-phase-from-primitive.md) | review §12.1–§12.4 | Opus | ⚠️ | *"Compute the transfer-function WKB phase and friction from tables"* (SHA not embedded, per the campaign convention) | [`logs/07-tk-wkb-phase-from-primitive.md`](logs/07-tk-wkb-phase-from-primitive.md) |

> Row 14 is numbered last because the campaign's numbers are append-only, but it **runs
> between 06 and 07**: it removes the per-object residual-table build that 06 introduced,
> and 07 inherits the same producer.

### Workstream D — the consumers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 08 | [`phase_spline` de-chunk](08-phase-spline-dechunk.md) | review §5 | Sonnet | ✅ | *"Drop the chunked phase spline in favour of one rebased spline"* (SHA not embedded, per the campaign convention) | [`logs/08-phase-spline-dechunk.md`](logs/08-phase-spline-dechunk.md) |
| 09 | [Gk consumer on `PrimitivePhase`](09-gk-consumer-primitive-phase.md) | review §5, §7, §8.3, §13.3–§13.4 | **Fable** → Opus (Fable unavailable) | ⚠️ | *"Evaluate the Green function phase from the conformal-time table"* (SHA not embedded, per the campaign convention) | [`logs/09-gk-consumer-primitive-phase.md`](logs/09-gk-consumer-primitive-phase.md) |
| 10 | [Tk consumer on the tables](10-tk-consumer-primitive-phase.md) | review §12.6, §12.7 | Opus | ⚠️ | *"Evaluate the transfer-function phase and friction from tables"* (SHA not embedded, per the campaign convention) | [`logs/10-tk-consumer-primitive-phase.md`](logs/10-tk-consumer-primitive-phase.md) |
| 15 | [`PrimitivePhase` explicit rate](15-primitive-phase-explicit-rate.md) | §3 `[10-primitive-phase-leading-rate-is-hardcoded]` | Sonnet | ⬜ | | |

> Row 15 is numbered last because the campaign's numbers are append-only, but it belongs to
> Workstream D — see README §3.

### Workstream E — the numeric region

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Numeric diagnostics and units](11-numeric-diagnostics-and-units.md) | review §10.2, §12.5, §13.1 | Opus | ⬜ | | |
| 12 | [Tk numeric `atol`](12-tk-numeric-atol.md) | review §12.5 | Opus | ⬜ | | |

### Workstream F — verification

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 13 | [Verification and docs](13-verification-and-docs.md) | review §4, §12.3, §13.5 | Opus | ⬜ | | |

**Progress:** 11 / 15 complete.

---

## 2. Mechanism-level tracking

Traceability from each review finding to the prompt that discharges it. IDs are local to this
campaign; the review section is the authority on each.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| M1 | **DEFECT, accuracy** | Two-stage phase solver: error is a fixed fraction of the *accumulated* phase; 13.9 rad ($k=10^5$) and 7366 rad ($3\times10^8$) at $z=0.1$ on the real background (§2, §4) | 06 | ✅ the ODE is gone; $\theta=-[k\,\Delta\tau+\Delta\rho]$ from the tables. LambdaCDM $k=10^5$: 1.19e-7 rad against prompt 01's references (target 1e-5); $k=3\times10^8$: 9.77e-4 rad = one ulp of $4\times10^{12}$ rad, the representation floor (target 5e-3); QCD $3\times10^8$: 9.77e-4 (log 06) |
| M2 | **DEFECT, accuracy** | The $Q$ variable is not "close to unity" ($-224$…$-11069$); the tolerance protects the wrong quantity; DOP853 dense output amplified by $\omega_i(1+u)$ — 0.33 rad on a linear phase (§3) | 06 | ✅ no $Q$, no dense output, no tolerances: `WKB_phase_function` has no `atol`/`rtol`. Radiation control $10^7$ rad span: 3.7e-9 rad (ODE 9.7e-3); $10^9$ rad: 3.6e-7 (ODE 0.98) |
| M3 | **DEFECT, cost** | Stage 1 cost ∝ span: $2.5\times10^6$ RHS evaluations, 63.7 s per object at $k=3\times10^8$; ~13 CPU-hours per $k$ (§4) | 06, 14 | ✅ 06: **0.031 s and 6,000 integrand evaluations** per object at $k=3\times10^8$ on LambdaCDM over the full response grid (target 0.05 s), 92 % of it the per-object residual-table build. 14 removed that build: **0.0010 s and 468 evaluations** per object (4 residual + 464 leading partials), 142.5 residual evaluations per object amortised over 50 objects of one $k$ against 6,924 (LambdaCDM) and 163.0 against 7,908 (QCD) |
| M4 | **DEFECT, accuracy** | `functions.tau` is a cubic spline of RK45 nodes: $1.4\times10^{-9}$ relative, ~2 rad of *oracle* phase error at $k=10^5$ in `compute_analytic_G/T` and `QuadSourceIntegral`'s η-limits (§7, §13.2) | 03 | ✅ 3.8e-16 relative at the LambdaCDM nodes; the retired accessor measured 3.08 rad off at $k=10^5$ (log 03) |
| M5 | **REQUIREMENT** | Double-double node table and an interval accessor `tau.delta`; a pointwise accessor carries the $\varepsilon\tau$ floor ($9\times10^{-4}$ rad at $3\times10^8$) on short baselines (§13.3) | 03 | ✅ `CumulativeTable` + `TablePrimitive`; one-interval Δτ ≤ 2.5e-16 (LambdaCDM), ≤ 9.4e-15 (QCD) relative |
| M6 | **REQUIREMENT** | Persist the low-order limb (`tau_lo_Mpc`, …); regeneration attached (§13.2; README §7 D1) | 03, 04 | ✅ `tau_lo_Mpc` (03) and `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F` (04); the factory refuses a datastore lacking any of the four by name |
| M7 | **REQUIREMENT** | Sound-horizon table $\tau_s$ and friction table $F$ per model; the friction ODE ($2.3$–$4.1\times10^{-7}$ relative error) goes (§12.2, §12.7) | 04, 07 | ✅ 04 built both tables (LambdaCDM $\tau_s$ 2.5e-16, $F$ 3.3e-16 relative at the checkpoints; QCD 2.1e-14 / 3.3e-16) and measured the ODE it replaces at 2.261e-07 absolute in $F$. 07 switched `TkWKBIntegration` onto them: `friction_RHS` and `FRICTION_INDEX` are gone from the producer (relocated verbatim into prompt 04's own test as `_friction_RHS`, where the 2.261e-07 measurement is unchanged), and `TkWKBValue.friction` is now bit-equal to `friction_F.delta(z_init, z)` — 6.5e-16 / 4.0e-16 / 2.7e-14 relative against prompt 01's references on (LambdaCDM $10^5$, LambdaCDM $3\times10^8$, QCD $3\times10^8$) |
| M8 | **REQUIREMENT** | The residual $\rho$ carried explicitly: $\le1.5\times10^{-3}$ rad for $G_k$ on QCD, $\approx-0.09$ rad for $T_k$; formed without subtraction (§6, §12.2) | 05, 14 | ✅ `ComputeTargets/phase_residual.py`: `build_phase_residual(model, k, z_nodes, sector, order)` → `CumulativeTable`; the correction comes from `*_omegaEff_sq_correction`, never from a subtraction. Worst 3.61e-16 rad against prompt 01's references over all twelve (model, sector, $k$) cases; $\rho_G$ bit-exactly zero in radiation; $\rho_T$ = −0.086 (LambdaCDM) to −0.093 (QCD). 14 added `residual_node_range` (the grid cut at `RESIDUAL_WKB_REGION_MARGIN = 0.5` of the leading term) and `cached_phase_residual`, one table per $(model, k, sector)$: $\rho$ moves by $\le1.4\times10^{-17}$ rad and $\theta$ is bit-identical |
| M9 | **REQUIREMENT** | Gauss orders decided by measurement on `QCD_Cosmology` across its spline knots; adaptive fallback for $\rho$ alone if needed (§11) | 02, 05 | ✅ 02 fixed all four orders at **4** with no adaptive fallback, but only under **break-point subdivision** on QCD; 05 consumes it — `RHO_GAUSS_ORDER = 4`, `RHO_ADAPTIVE_FALLBACK_REQUIRED = False`, and `build_phase_residual` applies the subdivision itself (1.22–1.23× the evaluations of `order × intervals` on QCD, exactly `order × intervals` on LambdaCDM) |
| M10 | **DEFECT, dead logic** | `sin_coeff` sign fix is provably always $+1$ (§8.1) | 06, 07 | ✅ 06 deleted it from `GkWKBIntegration.store()` (`sin_coeff = B`); 206 $(G,G')$ cases incl. $G=0$, $G<0$ confirm the factor was $+1$ and $B>0$ reproduces the initial data to 3e-16. 07 deleted the copy in `TkWKBIntegration.store()`; 206 $(T,T')$ cases incl. $T=0$, $T<0$ give the factor $+1$ every time and $B>0$ reproduces $T_{\rm init}$ to 8.6e-15, and the shipped `store()` returns `sin_coeff == B > 0`, `cos_coeff == 0.0` exactly |
| M11 | **DEFECT, consistency** | `shift_theta_sample` rebases `div_2pi` to the first sample, producing ±1-cycle offsets between objects (§8.1, §8.3) | 06, 07 | ✅ 06 added `WKBtools.apply_phase_offset` (per-sample wrap, no rebase) and switched `GkWKBIntegration.store()` to it; 990-object sweep: 0 rebase offsets, cycle steps only at stop-point transitions (90 = 90); the old helper would have rebased 180 (= the review's 60 of 330). 07 switched `TkWKBIntegration.store()` to it as well — there is one $T_k$ object per $k$ so no cross-object stitching arises (§12.7), but the stored $\theta+\delta$ is now exact. `shift_theta_sample` itself is retained in `WKBtools` for the two `docs/` scripts that still run (D7, log 07 deviation 5) |
| M12 | **DEFECT, hygiene** | Zero-length check compares a redshift to `atol`; 1-element array into `math.fmod` (NumPy deprecation); stale comments at `:262-264`, `:403` (§8.2) | 06 | ✅ exact test `len(z_sample) == 1 and z_sample[0] == z_init`; the `fmod` path and both comments went with the ODE; `Quadrature/supervisors/WKB.py` deleted |
| M13 | **DEFECT, accuracy + trap** | `phase_spline` chunking: no interpolation benefit, ordinates 64× inflated, knot residuals 30–50× worse, $1.4\times10^{-4}$ rad switch discontinuity, no progress guard for `logstep<2` (§5) | 08 | ✅ chunking deleted (`_build_*chunks*`, `_match_chunk`, `MINIMUM_SPLINE_DATA_POINTS` gone); one spline, rebased at the sample's *median* `theta_div_2pi`; `chunk_step`/`chunk_logstep`/`increasing` are accepted no-ops so `bessel_phase.py` and three test fixtures needed no changes. Interior interpolation error at $k=10^6$, 100/decade confirmed unchanged by chunking at 7–10e-5 rad (review 8.26e-5); the old progress-guard defect cannot recur (code path deleted) |
| M14 | **DEFECT, accuracy** | Consumers spline the growing phase: $h^4x/384$, $O(1)$–$O(10)$ rad at production $x$ (§5, §12.6). Same term as `source-remediation`'s `[12-phase-spline-error-grows-with-x]` | 09, 10 | ⚠️ 09 discharged it for $G_k$: on the review §5 consumer geometry at $k=10^8$, 100/decade, the error is **4.189e-8 rad** against the same samples' `phase_spline` at **7.286e-3 rad** — a ratio of **1.739e5**, and the 4.189e-8 is 2.81 ulp of the 9.09e7 rad span, i.e. the representation floor. $\varphi$ alone is recovered to 2.157e-10 rad, matching $h^4\max|\varphi''''|/384$. 10 discharged it for $T_k$: on a $w=1/3$ fixture at $x_T=10^6$, 100/decade, the error is **3.4482e-10 rad** against the same samples' cubic spline at **7.0854e-03 rad** — a ratio of **2.0548e7** — and the 3.4e-10 is 2.96 ulp of the $10^6$ rad phase, i.e. the `div * TWO_PI` floor. `PHASE_SPLINE_CHUNK_LOGSTEP` is deleted and no production path builds a `phase_spline`. ⚠️ because prompt 10 §3 item 3's 1e-10 on $\omega$ vs `theta_deriv` is missed at one abscissa per $w$ (`[10-residual-spline-end-condition]`) |
| M15 | **REQUIREMENT** | `PrimitivePhase`: $\theta=-k\Delta\tau+\varphi$ with the `phase_spline` protocol; closed-form $\theta'$; global anchor with the recorded floor (§7, §13.3, §13.4) | 09 | ✅ `ComputeTargets/primitive_phase.py`: `PrimitivePhase(k, leading, z_anchor, z_samples, phi_samples, *, sign, model_functions, label, spline_order)` with `raw_theta`/`theta_mod_2pi`/`theta_deriv`/`num_chunks`, plus `build_phi_samples`. `sign=-1` for $G_k$ at fixed $z_r$, `sign=+1` for $T_k$ at fixed $z_{\rm init}$ — one object for both sectors (prompt 10 reuses it, unsubclassed). $\theta'=\mathrm{sign}\,k/H+\varphi'$ in closed form, exact to 0.0 relative on the radiation control. Global anchor, floor documented in the module docstring. **Prompt 10 found that closed form is hard-wired to $1/H$**, which is the rate for $\tau$ but not for $\tau_s$ ($c_s/H$); it passes a `_SoundHorizonRate` adapter reporting $H/c_s$ rather than editing `primitive_phase.py` (out of its scope) — `[10-primitive-phase-leading-rate-is-hardcoded]` |
| M16 | **REQUIREMENT** | The `GkSource` rectifier is retained and verified inert on pure-WKB objects, correct on $\delta$-wraps (§8.3; `RECONCILIATION.md` §2 item 6) | 06, 09 | ✅ 06 documented what it must repair; 09 verified it against a faithful copy of `GkSource.py:166-233` (copied, not imported — `assemble_GkSource_values` is a Ray remote over datastore objects), on prompt 06's geometry with the source samples taken from the background grid. **Before rectification $\varphi$ jumps by exactly $+1$ cycle at every stop-point transition and nowhere else** (2 of 22 objects at $k=10^7$, $x_r=10^3$; **90 of 990** over the full sweep, log 06's figure); **the rectifier repairs every one** (90 corrections = 90 transitions) and $\varphi$ is then constant to **3.64e-12 rad**. On pure-WKB objects $\delta=0$ exactly and the rectifier makes **zero** corrections, leaving `theta_div_2pi` untouched. `GkSource.py` was not edited (D5) |
| M17 | **DEFECT, cost** | Per-RHS `*_omegaEff_sq` diagnostic: 45 % of the numeric run; the warning it feeds is live and must be preserved (§10.2, §13.1) | 11 | ⬜ |
| M18 | **DEFECT, units** | `delta_logz` supplied as $\Delta\log_{10}$, used as $\Delta\ln$; and the $G_k$ run is sampled on the response grid, not the source grid the value describes (§10.2, §13.1; `RECONCILIATION.md` §1 item 9) | 11 | ⬜ |
| M19 | **DEFECT, dead path** | `mode.lower()` before the `None` check (§10.2) | 11 | ⬜ |
| M20 | **DEFECT, comments + robustness** | The stop point is a maximum, not a minimum; the "fixed phase to avoid jitter" motivation is obsolete; `find_phase_minimum`'s $10^{-3}z$ step is safe only inside the window (§10.2) | 11 | ⬜ |
| M21 | **DEFECT, misleading** | `0.85·z_e6` truncation requests samples never produced in stop mode (§10.2) — comment only | 11 | ⬜ |
| M22 | **DEFECT, accuracy** | $T_k$ numeric run limited to $1.1\times10^{-5}$ of the envelope by `atol=1e-10` acting as a $10^{-5}$ relative tolerance (§12.5) | 12 | ⬜ |
| M23 | **REQUIREMENT** | Independent references and error definitions; throughput of the interval accessor measured early (§13.3, §13.5) | 01 | ⚠️ |
| M24 | **REQUIREMENT** | Verification on both models against the references; scoped pipeline run; additive docs (§13.5) | 13 | ⬜ |

**Out of scope (do not schedule):** the numeric→WKB hand-over (window, overlap, clamp,
$\sqrt{z_{e3}z_{e4}}$ limit — `docs/OPEN_ISSUES.md` §1.1); the $T_k$ LG truncation floor at the
hand-over (`[00-tk-lg-truncation-floor]`); raising the LG order; per-region anchoring; the
Wronskian two-solution construction; the super-horizon series initial condition; everything in
README §0.2's `transfer-remedial` file list.

---

## 3. Active and unresolved issues

Opened by the planning pass, 2026-09-10, before any prompt runs.

- **[02-qcd-reference-floor]** *(opened by prompt 02, 2026-09-10; inert)* — the QCD $\tau$ and
  $\tau_s$ references in `wkb_reference_data.json` are themselves accurate only to
  **1.88e-14 / 1.89e-14 relative**, measured as the disagreement between prompt 01's
  per-production-interval `quad` and prompt 02's break-aware `quad`
  (`convergence.models.QCDModel.*.json_vs_reference_max_rel`). The order-4 cumulative errors under
  `branch+knots` are 1.89e-14 and 1.90e-14 — i.e. *at* that floor, not above it. **Impact:** a
  floor on what prompts 03 and 13 may assert for QCD $\tau$ at the nodes; it is below README §6's
  $2\times10^{-14}$ target but only just, and asserting tighter would be asserting agreement
  between two references. LambdaCDM is unaffected (mpmath at 40 digits; its floor is
  `[01-lambdacdm-hubble-rounding-floor]`). **Next step:** if 13 needs more headroom, regenerate the
  QCD block of the JSON break-aware; otherwise none.

- **[02-qcd-T-z-spline-node-tolerance]** *(opened by prompt 02, 2026-09-10; not this campaign's)* —
  `LambdaCDM_GenericEOS._solve_T_z` solves $T\,g_S(T)^{1/3}=$ const with
  `root_scalar(..., xtol=1e-6, rtol=1e-4)`, and the 500 node values of `QCD_Cosmology`'s $T(z)$
  spline inherit that. Re-solving eight sampled nodes at `rtol=1e-15` moves the answer by up to
  **2.08e-05 relative** (at $z=10^{13}$; 1.17e-05 at $z=4.2\times10^7$; 0 at $z\le10^5$), which is
  $\sim4\times10^{-5}$ relative in $H$ in the radiation era. This is a property of the cosmology,
  not of any quadrature — every table in this campaign converges to the integral of the function
  the model actually defines — but it bounds what a QCD $\tau$ table *means* physically, and the
  node-to-node scatter is not smooth. Distinct from `source-remediation`'s
  `[01-genericeos-tz-spline-floor]`, which is about the *number* of spline points; this is about
  the accuracy of each point. **Impact:** the physical interpretation of every QCD number in
  prompts 03–07 and 13; no numerical target in this campaign is affected. **Next step:** a
  decision by the author on `_solve_T_z`'s tolerances; not scheduled here.

- **[01-offgrid-accessor-cost-on-qcd]** *(opened by prompt 01, 2026-09-10)* — the interval
  accessor costs **102.2 µs per call on `QCD_Cosmology` with both endpoints off-grid**, above
  README §4.3's 50 µs stop threshold (47.7 µs with one endpoint off-grid, 16 and 8 Hubble
  evaluations respectively at 8.7 µs each). The production case is on-grid at both ends — the
  background model is built on the source grid (`main.py:476`) — where the call costs **4.39 µs
  and zero Hubble evaluations**; only the per-object anchor $z_{\rm init}$ is off-grid.
  **Impact:** prompt 03's design if a consumer ever evaluates off-grid in bulk (prompt 09's
  Levin path is the candidate); the producers are unaffected. **Next step:** the orchestrator
  reports the figures to the user; closes when prompt 09 confirms its evaluation pattern is
  on-grid, or a caching partial is added.
  **Narrowed by prompt 03 (2026-09-11):** on the shipped `functions.tau` (order-4 partials, not the
  prototype's order-8) the figures are **0.33 µs on-grid / 26.1 µs one endpoint off-grid /
  52.0 µs both off-grid** on `QCD_Cosmology` (LambdaCDM 0.44 / 4.6 / 6.6 µs), 20,000 calls, best
  of 3 (log 03). Both-off-grid is still marginally above the 50 µs line; the cost is four
  `QCD_Cosmology.Hubble` evaluations at 7.35 µs each per off-grid endpoint.
  **Narrowed by prompt 06 (2026-09-11):** the producers never evaluate both-off-grid. The
  anchor $z_{\rm init}$ is the only off-grid endpoint and it is paired with an on-grid sample,
  once per sample: 464 partial evaluations (4 × 116 samples) inside a 0.031 s object at
  $k=3\times10^8$ on LambdaCDM; at most ~1,160 samples per `TkWKBIntegration` object on the
  source grid, i.e. $\le$ 30 ms on `QCD_Cosmology` at log 03's 26 µs. The residual table is
  built with $z_{\rm init}$ as a node and needs no partial. Still closes with prompt 09.
  **Narrowed by prompt 09 (2026-09-11); does not close.** The consumer's pattern is *not*
  on-grid: a Levin region evaluates the phase at Chebyshev abscissae, which are off-grid by
  construction. What prompt 09 does establish is that it is always **one** off-grid endpoint,
  never two — `PrimitivePhase`'s anchor is `z_response`, a background-grid node, so it costs
  nothing — so the figure that applies is log 03's **26.1 µs one-endpoint-off-grid on
  `QCD_Cosmology`** (4.6 µs on LambdaCDM), not the 52.0 µs both-off-grid number this issue was
  opened against, and the 50 µs stop threshold is not crossed. Measured on the radiation
  stand-in: `raw_theta` spends **0** integrand evaluations at a node and **exactly 4** (one
  order-4 panel) off-grid; 3.11 µs / 5.85 µs per call, best of 3 over 20 x 286 calls.
  **Next step:** closes if a Levin-side measurement on `QCD_Cosmology` (prompt 13) shows the
  per-call cost acceptable in bulk, or if a cached anchor partial is added.

- **[01-lambdacdm-hubble-rounding-floor]** *(opened by prompt 01, 2026-09-10; inert)* — the
  double-precision evaluation of `LambdaCDM.Hubble` carries 2–9e-15 relative near $z=1$–$10^6$,
  which is the floor on $\Delta\tau$ over one production grid interval whatever the Gauss order
  (4.687e-15 at order 4 and 4.525e-15 at order 20 on the same interval) and whatever the storage
  width. In phase that is $3.8$–$6.1\times10^{-5}$ rad at $k=3\times10^8$ per interval.
  **Impact:** a floor on what prompts 03 and 13 may assert for a single-interval $\Delta\tau$ on
  LambdaCDM; it is *below* README §6's $5\times10^{-3}$ rad target and above the
  $\varepsilon k\tau$ floor, and is a different error from either. **Next step:** none; recorded
  so a later reader does not chase it.

- **[00-unresolved-osc-print-policy]** *(planning, 2026-09-10)* — README §7 D2: with the units fixed
  and the test evaluated on the caller's actual sample grid, `has_unresolved_osc` will fire on
  essentially every `GkNumericIntegration` object (response grid, $x\gtrsim22$). Faithful, but a
  print storm. **Impact:** prompt 11 (implements and measures), production log volume. **Next
  step:** prompt 11 reports the measured fire rates; the user chooses per-object line / per-$k$
  summary / explicit grid; closes when the chosen policy lands (a follow-up prompt if not (i)).

- **[00-consumer-anchoring-floor]** *(planning, 2026-09-10)* — `PrimitivePhase` reduces
  $k\Delta\tau$ against a global anchor ($z_r$ for $G_k$, $z_{\rm init}$ for $T_k$), so its
  `theta_mod_2pi` carries the $\varepsilon k\tau$ floor: $9\times10^{-4}$ rad at $k=3\times10^8$
  (review §13.4). Per-region anchoring would scale the floor with the region's own phase.
  **Impact:** Levin regions at the largest $k$; below the QCD LG floor, three to four orders below
  today's consumer error.
  **Measured by prompt 09 (2026-09-11).** The floor is now the *whole* error: on the review §5
  consumer geometry at $k=10^8$, $z_r=0.1$, $z_s\in[10,10^4]$, `raw_theta` is **4.189e-8 rad**
  from a 50-digit reference, which is **2.81 ulp** of the 9.0899e7 rad span against an
  $\varepsilon k\tau$ floor of 2.019e-8 rad and one ulp of 1.490e-8 rad; at the samples
  themselves it is 3.681e-8 rad. The residual $\varphi$ alone is recovered to **2.157e-10 rad**,
  so everything above the floor has been removed and re-anchoring is the only remaining lever.
  **Next step:** unchanged — a follow-up prompt adding an `anchored(z0)` view to `PrimitivePhase`
  and a Levin-side hook, if the verification in 13 shows the floor matters.

- **[10-residual-spline-end-condition]** *(opened by prompt 10, 2026-09-11)* — prompt 10 §3
  item 3 asks for $\omega(z) =$ `phase.theta_deriv(z)` to $10^{-10}$ relative on its "LG"
  fixture. Measured over the same domain its sibling test uses (`z_WKB[3:-3]`): **1.0492e-10**
  ($w=1/3$) and 7.9502e-11 ($w=0.2$) — a 4.9 % miss at $w=1/3$, against **4.249e-08** and
  2.163e-08 for the representation prompt 10 replaces (a factor 405). The excess is entirely the
  not-a-knot end condition of `PrimitivePhase`'s cubic residual spline at the *top* of the WKB
  region, where $\varphi\sim-1/x$ varies fastest: the error falls ~3× per sample inwards
  (1.049e-10, 3.245e-11, 5.559e-12 at the third, fourth and fifth stored samples) and is
  2.9e-12 well inside; the interior is 1.3e-15 relative, so this is **not** a representation
  floor, and it is not fixture noise (replacing the fixture's `solve_ivp` phase by per-interval
  adaptive `quad` reproduces 1.049e-10 to three figures). The shipped test asserts `< 1e-9` over
  `z_WKB[3:-3]` and `< 1e-11` over `z_WKB[5:-3]`, printing both. **`spline_order=5`, which
  `PrimitivePhase` already accepts, gives 1.794e-12 over `z_WKB[3:-3]` and 9.695e-12 over the
  *whole* WKB region** — the prompt's figure met everywhere with 10× margin — but a quintic needs
  six samples where `TkSourceFunctions.MIN_SPLINE_DATA_POINTS = 5`, and it would make the $T_k$
  consumer's representation differ from prompt 09's $G_k$ consumer for a number nobody asked
  for, so it was not taken. **Impact:** prompt 10 §3 item 3 as written; the same end condition
  applies to prompt 09's $G_k$ consumer and to anything reading `theta_deriv` near the edge of a
  WKB region (`phase_groups`, `AdaptiveLevin`). **Decision (user, 2026-09-11): left open for prompt 13.**
  The shipped 1e-9 / 1e-11 pair stands for now and prompt 10 §3 item 3's text was **not** amended
  — a dated note there records the measurement and points here — because every figure above is
  from a constant-$w$ closed-form stand-in, where $\varphi$ is an analytic residual and the
  samples are not production grid nodes. **Next step:** prompt 13 re-measures this identity on the
  real background for both sectors and then either closes this issue at the cubic or escalates to
  `spline_order=5` for both consumers with `MIN_SPLINE_DATA_POINTS` raised to 6. It should also
  report the error at the second and third samples, not only the window maxima, since the end
  effect is what is in question.

- **[10-primitive-phase-leading-rate-is-hardcoded]** *(opened by prompt 10, 2026-09-11)* —
  `PrimitivePhase.theta_deriv` (`ComputeTargets/primitive_phase.py:287`) computes the leading
  derivative as `sign * k / model_functions.Hubble(z)`, which is
  `d/dz[k tau.delta(z, anchor)] = +k/H` — correct for the Green's function's $\tau$ and wrong by
  $1/c_s$ ($\approx1.73$ in radiation) for the transfer function's $\tau_s$, whose derivative is
  $c_s/H$. `primitive_phase.py` is outside prompt 10's file list, so prompt 10 passes a
  `TkSourceFunctions._SoundHorizonRate` adapter that reports $H/c_s$; `Hubble` is the only thing
  `PrimitivePhase` reads from `model_functions`, so nothing else is affected, but
  `phase._Hubble` on a transfer-function phase now returns $H/c_s$ rather than $H$.
  **Impact:** anyone adding a third leading primitive, or reading `model_functions` off a
  `PrimitivePhase`; a silent factor-1.73 error in `theta_deriv` if a future caller passes the
  model's own `ModelFunctions`. **Next step:** give `PrimitivePhase` an explicit `rate` callable
  (defaulting to `1/Hubble`) in its own commit, and drop the adapter. **Assigned (2026-09-11):
  prompt 15**, dispatched ahead of Workstream E at the user's request.

- **[10-transfer-remedial-tolerance-comments-stale]** *(opened by prompt 10, 2026-09-11)* — five
  tolerance comments in `ComputeTargets/tests/test_tk_source_functions.py` that
  `transfer-remedial` prompt 08 (`8ba9159`) wrote now describe a mechanism prompt 10 deleted, and
  quote numbers three to four orders above what the tests measure: the module docstring's
  "consumer re-spline … is now the binding term in `err_T`"; the `err_M` comment's "backed out
  from `M_exact` and re-splined" (measured 1.272e-13, now 1.655e-15); the `err_T` comment's
  "3.021e-08 … the h^4 cubic fit `TkSourceFunctions` puts through the sampled phase" (now
  2.079e-12); `test_phase_convention`'s "`phase_spline` rebases each chunk"; the
  `[grid refinement]` comment's "this *is* the consumer re-spline floor" (6.090e-06, now
  1.776e-10); and `test_omega_matches_phase_derivative`'s "a `phase_spline` through the exact
  integral of `omega_eff` … 4.249e-08" (now 1.049e-10). The test name
  `test_spline_error_dominates_on_the_production_grid` is also now a misnomer. **None was
  edited**: the orchestrator made `8ba9159`'s tolerance constants and comments a stop condition
  for prompt 10, and all six assertions still pass unchanged (three of them now with four extra
  orders of margin). **Impact:** anyone reading those comments to calibrate a new threshold will
  calibrate against the retired representation. **Next step:** a comments-only commit refreshing
  the six blocks with prompt 10's measured values, which needs only the user's confirmation that
  `8ba9159`'s text may be rewritten now that both campaigns have landed.

- **[10-wrap-theta-loop-at-large-phase]** *(opened by prompt 10, 2026-09-11; inert in
  production)* — `LiouvilleGreen.WKBtools.wrap_theta` (`:69-94`) range-reduces by adding
  `TWO_PI` in a `while` loop, so at $|\theta|\sim10^6$ rad it performs ~1.6e5 additions of a
  quantity $10^6$ times smaller than the accumulator and returns a pair that reconstructs
  $\theta$ only to **1.3862e-06 rad** (and costs 1.6e5 iterations). Production is unaffected:
  its only caller is `apply_phase_offset`, which passes `mod + delta` with `mod` already in
  $(-2\pi,0]$, so the loop runs at most twice. `WKB_mod_2pi` uses `fmod`, is exact, and is what
  prompt 10's $x_T=10^6$ fixture uses. **Impact:** any test fixture that reduces a large
  unwrapped phase with `wrap_theta` — prompt 10's test 3.1 would have been 14× over its own
  1e-7 rad bound on the fixture's arithmetic alone. There is no warning in the docstring.
  **Next step:** a one-line note on `wrap_theta`, or an `fmod` fast path for
  $|\theta| > 2\pi$; three test modules still call it at small $|\theta|$, where it is fine.

- **[00-tk-lg-truncation-floor]** *(planning, 2026-09-10; **assigned to the hand-over campaign**)*
  — the transfer function's LG representation is not exact in radiation: $3.8\times10^{-5}$ of the
  envelope from $x_i=24$, $\sim1.4\times10^{-4}$ at the production hand-over $x_T\approx15.5$,
  scaling as $x_i^{-3}$, with a frozen amplitude offset $\sim x_i^{-4}$ (review §12.4). Below it no
  numerical improvement in this campaign is visible. Remedies — later hand-over ($x_T=50$ gives
  $4\times10^{-6}$), higher-order LG frequency, or the Bessel exact representation in the radiation
  era — are hand-over decisions. **Impact:** the $T_k$ value-level acceptance rows are floored here.
  **Measured by prompt 07 (2026-09-11)**, reproducing review §12.4 to two significant figures with
  the shipped `store()` on the exact radiation background: max $|\delta T|/\text{env}$ over
  $x_i\le x\le10^4$ is **3.8118e-05** ($x_i=24$), **4.0663e-06** (50), **5.0628e-07** (100),
  **7.7760e-09** (400), with frozen amplitude offsets 1.30e-05, 1.14e-06, 1.51e-07 and 2.39e-09 at
  $x=10^4$, and a 24/400 ratio of **4902** against $(400/24)^3=4630$ — the $x_i^{-3}$ law, now
  pinned by `test_tk_wkb_phase.TestRadiationValue`. **Next step:** none here; recorded for the
  hand-over campaign.

- **[00-tk-superhorizon-ic-series]** *(planning, 2026-09-10; inert)* — once prompt 12 lands, the
  $T_k$ numeric floor is the super-horizon initial condition $T=1,T'=0$ at $2.5\times10^{-6}$
  (review §12.5; audit TK-7), removable with the series $T\approx1-x^2/10$ for $w=1/3$ (general $w$
  needs the spec). **Impact:** a floor on what `test_tk_numeric_atol.py` may assert. **Next step:**
  a spec-level decision by the author; not scheduled.

- **[03-backgroundmodelvalue-build-path]** *(opened by prompt 03, 2026-09-11; confirmed)* — the
  `sqla_BackgroundModelValue_factory.build()` path has two latent defects on its
  query-existing-row branch (`RECONCILIATION.md` §2 item 11): the fresh-insert dict uses the key
  `"wkb_serial"` where the column is `model_serial`, and the consistency check reads
  `row_data.Hubble` where the select provides `Hubble_GeV`. Production never takes this path —
  values are inserted through `BackgroundModel.store()` — so neither has fired. Prompt 03 edited
  the neighbouring lines (adding `tau_lo_Mpc`) and did not repair them. **Impact:** anyone who
  calls `pool.object_get("BackgroundModelValue", …)` directly gets an `IntegrityError` (insert) or
  an `AttributeError` (existing row). **Next step:** a two-line fix in its own commit, with a
  test that exercises `build()` against an in-memory SQLite store.

- **[03-qcd-short-baseline-reference-endpoint-rounding]** *(opened by prompt 03, 2026-09-11;
  inert)* — the QCD short-baseline references in `wkb_reference_data.json` (`delta_tau_full`,
  `delta_tau_fraction`) were computed by `quad` in $u=\log(1+z)$ between **rounded double
  endpoints** `log1p(z)`, so each carries up to $\tfrac12{\rm ulp}(u_{\rm hi})+\tfrac12{\rm ulp}(u_{\rm lo})$
  of endpoint error relative to the baseline width $W$: bounds 7.7e-14 / 2.1e-13 (full / 37 %
  fraction at $z=10^6$), 3.9e-14 / 1.1e-13 ($z=10^2$), 9.7e-15 / 2.6e-14 ($z=1$). Measured
  disagreement of the JSON against a `quad` in the exact-width parametrisation
  $1+z=(1+z_{\rm lo})e^t$, $t\in[0,W]$: 4.4e-15 / 3.3e-14, 9.4e-15 / 2.0e-14, 1.9e-15 / 9.6e-15,
  every one inside its bound; the shipped table agrees with the exact-width `quad` to ≤ 8.8e-16
  on all six. The LambdaCDM references are mpmath at exact `mpf(float(z))` endpoints and do not
  carry this. Distinct from `[02-qcd-reference-floor]`, which is the break-unaware/break-aware
  disagreement on the *cumulative* values. **Impact:** a floor on what prompts 03, 04 and 13 may
  assert for QCD short baselines — `test_background_tau.py` asserts README §6's $10^{-13}$, not
  the JSON's recorded self-agreement (~1.6e-15). **Next step:** if 13 needs headroom, regenerate the
  QCD short-baseline records in the exact-width parametrisation; otherwise none.

- **[03-integrationsolver-stepping-minimum-lookup]** *(opened by prompt 03, 2026-09-11; inert)* —
  `sqla_IntegrationSolver_factory` registers `"stepping": "minimum"` and `build()` matches
  `label == label AND stepping >= stepping`, returning the first such row. Every solver registered
  before this campaign has `stepping=0`, so it never mattered; the τ table registers
  `("cumulative-GL", 4)` with the Gauss order as the stepping. If a second order is ever registered
  under the same label, a query for the lower order can be served by the higher order's row (the
  returned `IntegrationSolver` object still reports the requested stepping, but `solver_serial`
  points elsewhere). **Impact:** none while every table uses order 4 (log 02); a trap for whoever
  changes an order. **Next step:** if a second order is registered, either fold the order into the
  label or query with an exact stepping match.

- **[04-background-rhs-evaluations-count]** *(opened by prompt 04, 2026-09-11; inert)* —
  `compute_background` now builds three Gauss–Legendre tables, but `IntegrationData` is a fixed
  namedtuple with a single evaluation counter and prompt 03's `test_background_tau.test_payload_shape`
  asserts `RHS_evaluations == TAU_GAUSS_ORDER * (nodes - 1)` **exactly** on LambdaCDM. Prompt 04's
  §2 item 1 asked for the new integrand evaluations to be added to the returned `IntegrationData`;
  `test_background_tau.py` is not in prompt 04's "files you may touch", so instead
  `RHS_evaluations` still counts the $\tau$ table alone (6,924 on LambdaCDM, 8,552 on QCD) and the
  other two are reported as the payload keys `cs_tau_evaluations` and `friction_F_evaluations`
  (the same numbers again on each model, 20,772 / 25,656 in total — log 02's figures).
  `compute_time` does cover all three tables. **Impact:** the persisted `RHS_evaluations` column
  of `BackgroundModel` understates the build by 3×; anyone reading it as the job's cost is
  misled. **Next step:** if the aggregate is wanted, relax that one assertion in
  `test_background_tau.py` and sum the three counters — a two-line change in its own commit.

- **[06-metadata-column-headroom]** *(opened by prompt 06, 2026-09-11; narrowed by prompt 14)* —
  the `GkWKBIntegration`/`TkWKBIntegration` `metadata` column is
  `sqla.String(DEFAULT_STRING_LENGTH)` = `String(256)` and holds `json.dumps(obj.metadata)`.
  SQLite does not enforce the length; PostgreSQL would truncate or refuse. **Impact:** anyone
  adding a metadata key (prompt 07's friction bookkeeping is the candidate).
  **Narrowed by prompt 14 (2026-09-11):** with prompt 14's `rho_reused` key the longest payload
  is **227 characters** (LambdaCDM, $k=3\times10^8$, `Gk`, the call that builds the table), so
  **29 characters remain**; the four variants measure 226 / 222 (`Gk` built / reused) and
  223 / 219 (`Tk`), and the `initial_data_only` payload is 82. Prompt 06's 206 was the same
  payload without the key, so a key costs ~20 characters. The count is now asserted by
  `test_residual_table_reuse.TestTableIsShared.test_metadata_still_fits_the_column`, which
  fails rather than overflowing. **Next step:** count before adding; or widen the column in a
  schema-touching commit.

- **[14-residual-range-top-margin]** *(opened by prompt 14, 2026-09-11; inert)* — the residual
  table's node range is cut at the top where the Liouville–Green frequency stops keeping
  `RESIDUAL_WKB_REGION_MARGIN = 0.5` of its leading term, not at its turning point. The bare
  $\omega^2>0$ rule prompt 14 asked for does not work on `QCD_Cosmology`: the sign is not
  monotone in $z$ (the Green's-function frequency is positive again at the top of the production
  grid, $z\ge1.9\times10^{16}$), and a panel whose two nodes are both positive can hold a Gauss
  abscissa where $\omega^2<0$ — measured at $z=3.6118639\times10^{15}$, $k=3\times10^8$, where
  the ratio $\omega^2/\omega_0^2$ scatters over $\pm0.1$ between neighbouring nodes
  (`[02-qcd-T-z-spline-node-tolerance]`). **Impact:** the table does not cover an anchor within
  about one e-fold of horizon crossing, where `WKB_phase_function` would now raise a range error
  rather than build a table.
  **Corrected by the orchestrator (2026-09-11).** Log 14 and the commit body justify this as
  "the cut lies above the highest node at which the WKB criterion $|d\ln\omega/dz|/\omega\le1$
  holds … so no anchor a producer can accept lies outside the table". That generalises a
  measurement taken at $k=10^5$ and $3\times10^8$ and **is not true in general**: at
  `QCD_Cosmology`, sector `Gk`, $k=10^7$ the cut is at $z=4.83\times10^{13}$ while the highest
  criterion-satisfying node is $z=3.05\times10^{14}$. The criterion is not monotone in $z$ on
  QCD, so "highest node satisfying it" is not an envelope.
  The invariant that does hold is a **margin between the cut and the production anchor**: every
  producer anchors three e-folds inside the horizon, while the cut sits at ~1.25 e-folds. Over
  all twelve (model, sector, $k$) combinations the ratio cut/anchor is 5.7–5.9× (1.74–1.78
  e-folds) in the `Tk` sector and 103–8.9×10⁶ (4.6–16 e-folds) in `Gk`; the tightest is
  LambdaCDM `Tk` at $k=3\times10^8$, **1.74 e-folds**. At the QCD/`Gk`/$k=10^7$ counterexample
  the production anchor is 186× below the cut, so it is unreachable. The failure mode if it were
  ever reached is a loud `RuntimeError` from `CumulativeTable`, not a wrong number.
  **No test pins either statement** — neither the criterion ordering nor the 1.74-e-fold margin —
  so a change to `RESIDUAL_WKB_REGION_MARGIN`, to the production grid, or to a cosmology would
  not be caught until a producer crashed. **Next step:** prompt 13 measures the cut-to-anchor
  margin across the production $k$ range on both models and both sectors as part of its
  verification, and records it; if it is ever below ~1 e-fold, the margin constant is what to
  revisit.

- **[14-rhs-evaluations-depend-on-build-order]** *(opened by prompt 14, 2026-09-11; inert)* —
  prompt 14 §2 item 3 requires `stage_1_data.RHS_evaluations` to count only the integrand
  evaluations a call actually spent, so the first object of a given $(model, k, sector)$ in a Ray
  worker stores ~7,000 and every later one a few hundred. Which object is first depends on the
  scheduler, so two runs over the same inputs can persist different `RHS_evaluations` for the
  same object. The column is payload data and part of no lookup key, so nothing misses.
  **Impact:** anyone reading the column as "the cost of this object"; prompt 13's timing of the
  scoped pipeline run should sum it over a $k$ rather than sample it. **Next step:** none, unless
  13 wants a per-object cost, in which case the table build belongs in its own counter.

- **[08-docs-scripts-reference-removed-chunking]** *(opened by prompt 08, 2026-09-11; inert)* —
  two `docs/` reproduction scripts read private internals of `phase_spline` that prompt 08 deleted
  along with chunking: `docs/gk-wkb-review-fable-2026-09-09/t5_spline.py:26` reads
  `spl._chunk_list`, `spl._splines` and calls `spl._match_chunk(...)`; `docs/gk-wkb-review-astra-pathfinder-2026-09-08/measure.py:163-164,174,176`
  reads `spl._splines` and calls `spl._match_chunk(...)`. Both scripts measured the chunked tree
  they ran on and the documents they support are correct for it; they were not edited (README §5
  "verification documents are additive"), the same treatment prompt 06 gave the phase-ODE removal
  (`[06-docs-scripts-reference-removed-ode]`, below). **Impact:** anyone re-running either script
  gets an `AttributeError` rather than the chunked-vs-unchunked comparison it printed when the
  review was written. `docs/gk-wkb-review-fable-2026-09-09/t7_jitter.py` and
  `docs/spec-code-audit/scripts/GK_05_phase_reassembly.py` also construct `phase_spline` objects
  but only through the public constructor and `raw_theta`/`theta_mod_2pi`, so they are unaffected.
  **Next step:** none; a dated note in the two review folders' READMEs if someone trips over it.

- **[06-docs-scripts-reference-removed-ode]** *(opened by prompt 06, 2026-09-11; inert)* — the
  reproduction scripts `docs/gk-wkb-review-fable-2026-09-09/{t2_solver,t4b_production_real,
  t7_jitter,t9_warn}.py`, `docs/gktk-remedial/baseline_k1e5.py` and
  `docs/gk-wkb-review-astra-pathfinder-2026-09-08/{measure,alternatives}.py` import
  `integrate_phase_function`, `stage_1_evolution`, `stage_2_evolution` or
  `DEFAULT_OMEGA_WKB_SQ_MAX`, none of which exists after prompt 06. They measured the tree they
  ran on and the documents they support are correct for it; they were not edited (README §5
  "verification documents are additive"). **Impact:** anyone re-running them gets an
  `ImportError`/`AttributeError` rather than the old ODE. **Next step:** none; a dated note in
  the two review folders' READMEs if someone trips over it. `t6_sweep.py` and
  `GK_05_phase_reassembly.py` still run (D7).
  **Widened by prompt 07 (2026-09-11):** two more scripts join them for a different removed
  symbol. `docs/spec-code-audit/scripts/TK_04_WKB_reconstruction.py:27` and
  `docs/gktk-remedial/baseline_k1e5.py:32` do `from ComputeTargets.TkWKBIntegration import
  friction_RHS`, which prompt 07 deleted; the function itself is alive and verbatim as
  `_friction_RHS` at `ComputeTargets/tests/test_background_cs_tau_friction.py:79`, so the one-line
  repair for either script is to import it from there. Prose-only references at
  `ComputeTargets/BackgroundModel.py:180`, `ComputeTargets/tests/wkb_reference.py:25` and
  `ComputeTargets/TkSourceFunctions.py:46` (prompt 10's file) now name a symbol that has moved;
  they break nothing.

- **[07-tk-per-object-cost-is-all-setup]** *(opened by prompt 07, 2026-09-11)* — a
  `TkWKBIntegration` object at $k=3\times10^8$ on LambdaCDM over the 1,384-sample source grid
  costs **0.0494–0.0516 s** across seven timed runs, straddling prompt 07 §3 item 6's
  $\le0.05$ s rather than clearing it (the ODE it replaces: 58 s). Essentially all of it is
  setup, in two halves that are independently removable:
  1. **5,840 of the 11,376 integrand evaluations (0.0321 s) build the per-$k$ residual table**,
     and nothing amortises it. Prompt 14's cache is keyed on $(model, k, sector)$ and pays off
     49× for $G_k$, where ~1,700 objects share a key; there is exactly **one**
     `TkWKBIntegration` object per $k$ (`main.py:682-712`, review §12.1), so the $T_k$ sector
     always pays the build in full. With the table cached the same call is **0.0178–0.0186 s**.
  2. **5,536 evaluations are the *leading* table's off-grid anchor panel, recomputed once per
     sample.** `WKB_phase_function` calls `leading.delta(z_init, z)` directly, so
     `CumulativeTable._locate(z_init)` goes off-grid and re-integrates the identical order-4
     panel 1,384 times. Prompt 14 split exactly this anchor off at `nearest_table_node` for the
     residual table and did not do the same for the leading one. $G_k$ pays it too (464
     evaluations on the 12×-sparser response grid, log 06).
  **Impact:** prompt 07 §3 item 6's threshold, which is not met reliably; prompt 13's cost
  measurements, which should quote both figures and say which is which; and the wall-clock of a
  production $T_k$ stage (50 wavenumbers × 2 models ≈ 5 s, so this is a tidiness issue, not a
  throughput one). **Next step:** apply prompt 14's `rho_anchor_node` split to the leading table
  in `Quadrature/integrators/WKB_phase_function.py` — one `nearest_table_node` call and one
  addition per sample, removing item 2 entirely and making the $T_k$ total ~0.034 s. Item 1 is
  irreducible without changing where the residual table's nodes come from. Neither is in prompt
  07's scope.
  **Widened to the consumer by prompt 09 (2026-09-11).** Item 2 is a property of
  `CumulativeTable.delta`, not of the producer: it re-integrates an off-grid *endpoint*'s panel on
  every call. `PrimitivePhase` escapes it only because its anchor is on-grid — `z_response` is a
  background-grid node — so a consumer whose anchor is off the grid would pay one extra order-4
  panel per evaluation on top of the abscissa's. **Prompt 10's $T_k$ anchor is $z_{\rm init}$, a
  `root_scalar` root** (`RECONCILIATION.md` §2 item 5), so it is exactly that case, and the
  `nearest_table_node` split prompt 14 applied to $\rho$ is the same one-line remedy there.
  **Next step:** unchanged for item 1; for item 2, apply the split in `WKB_phase_function` and —
  if prompt 10 anchors off-grid — inside `PrimitivePhase`.

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation a later prompt has to
> work around, a measured cost that changes a later prompt's decision. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them, and update `docs/OPEN_ISSUES.md`.

---

## 4. Resolved issues

- **[10-quadsource-fixture-model-substitution]** *(opened by prompt 10, 2026-09-11; resolved by the
  user, 2026-09-11)* — prompt 10 edited five lines of stand-in construction in
  `ComputeTargets/tests/test_quadsource_integral.py` (`Case.__init__`'s non-`exact`
  `Tk_builder`), which is **outside its "Files you may touch" list**. That module builds
  `TkSourceFunctions` objects from inputs captured out of `Fixture.exact_functions()` and then
  supplies its own `FakeModel(w)`; now that the friction comes from
  `model.functions.friction_F`, the two disagree by **5.6231e-06** in $F$ (the LG truncation of
  the exact envelope), so five of its 37 tests raised the new cross-check and two more failed on
  message text. The fix substitutes `Fq/Fr.exact_envelope_model()` per wavenumber, exactly as
  the file's own `exact` branch one line above already does; no tolerance, threshold or
  production file is touched. It cannot be done from inside the allowed files, because
  `Case.model` is shared with the Green's-function fixtures and `compute_QuadSource_integral`
  calls `Tk_functions_builder(model, k, …)` with its own model, so only the builder closure can
  substitute per $k$. With the cross-check temporarily disabled the module passes unedited — so
  the check is reporting a genuine fixture defect (its "realistic" transfer functions would
  silently become LG-amplitude rather than exact-envelope), not merely failing on a technicality.
  **Impact:** the prompt's §4 acceptance ("`test_quadsource_integral.py` passes") cannot be met
  without it; the campaign's own §4.3 file stop-list does not name this file. **Resolution (user, 2026-09-11):** the hunk is
  **accepted**. The orchestrator had confirmed the diagnosis independently by restoring the file
  to its pre-prompt content and re-running the module — four `RuntimeError`s from
  `_check_friction_samples` at `TkSourceFunctions.py:437` — so the alternative was to weaken a
  cross-check the prompt's own §1 requires in order to protect a fixture that was quietly
  inconsistent. Prompt 10's "Files you may touch" list was extended in place with a dated note
  giving this reasoning, and `wkb_reference.py` was added to it at the same time, that file having
  always been in scope in substance (§2 names both it and the `ClosedFormPrimitive` helper to put
  there). Both remain **stand-in construction only**.

- **[00-transfer-remedial-test-file-overlap]** *(planning, 2026-09-10; resolved by prompt 10,
  2026-09-11)* — prompt 10 edits the stand-in `ModelFunctions` fixtures in
  `ComputeTargets/tests/test_tk_source_functions.py` and `test_phase_groups.py`;
  `transfer-remedial` prompt 08 edits tolerance constants and comments in the same files, so
  either order risked a textual conflict. **Decision (user, 2026-09-10):** Workstreams A, B, C, E
  run in parallel with `transfer-remedial`; Workstream D waits for the merge (README §4.2 item 1).
  **Merge confirmed by the orchestrator, 2026-09-11 (Workstream C close-out):** `transfer-remedial`
  (all nine prompts) and `qsi-phase-groups` landed in `gktk-remedial` at **`e01c31d`**, whose
  message records that only `docs/OPEN_ISSUES.md` conflicted, resolved as the union; its prompt 08
  is **`8ba9159`** ("Tighten the Bessel tests to the new accuracy"), the last commit to touch
  either shared file.
  **Resolution (prompt 10).** No conflict of any kind arose, and the overlap turned out to be
  smaller than planned: **`test_phase_groups.py` needed no edit at all** — it imports `FakeModel`
  and `Fixture` from `test_tk_source_functions` and never builds a `ModelFunctions` itself, so
  upgrading `FakeModel` was enough, and `git diff` leaves that file byte-identical to `8ba9159`
  (18 tests OK). In `test_tk_source_functions.py`, `git blame` attributes 87 lines to `8ba9159`
  and **prompt 10 modified none of them**; every line it removed is `e3348e4`'s. All six
  assertions `8ba9159` set still pass, three of them with four extra orders of margin
  (`err_M` 1.272e-13 → 1.655e-15, `err_T` 3.021e-08 → 2.079e-12, `[grid refinement]` 6.090e-06 →
  1.776e-10). What did *not* survive is the accuracy of `8ba9159`'s *explanatory* text, which is
  now stale in five places — carried forward as `[10-transfer-remedial-tolerance-comments-stale]`
  (§3) rather than fixed, because editing it was a stop condition for prompt 10. A *third* module
  turned out to construct the consumer as well, which the planning pass did not anticipate:
  `[10-quadsource-fixture-model-substitution]` (§3).

- **[09-consumer-threshold-below-representation-floor]** *(opened by prompt 09, 2026-09-11;
  resolved by the orchestrator, 2026-09-11)* — prompt 09 §4 `test_primitive_phase.py` test 1 asked
  for a consumer phase error $\le10^{-8}$ rad on the geometry it names in the same sentence
  ($k=10^8$, $z_r=0.1$, $z_s\in[10,10^4]$, exact radiation, 100/decade), and called that
  "README §6's consumer row". It is not: README §6's row is $\le10^{-6}$ rad, and $10^{-8}$ rad is
  **below the double-precision floor of the quantity being asserted**. On that geometry $|\theta|$
  reaches 9.0899e7 rad, one ulp of which is 1.490e-8 rad and whose $\varepsilon k\tau$ floor is
  2.019e-8 rad, so the threshold was **0.67 ulp**. The measured error is **4.189e-8 rad = 2.81
  ulp** (3.681e-8 rad at the samples, where the spline contributes nothing); the companion ratio
  assertion passed as written at **1.739e5** against the required $10^5$. Prompt 09 asserted
  README §6's $10^{-6}$ rad plus a floor-aware 6-ulp bound instead and said so in the module
  docstring.

  **Resolution.** An independent review by Claude Fable 5.1
  ([`reviews/09-prompt-09-review-fable.md`](reviews/09-prompt-09-review-fable.md), commissioned by
  the user after the orchestrator stopped) confirmed the arithmetic, decomposed the achieved
  2.81 ulp into 2.39 ulp from `TablePrimitive.delta`'s double return and 0.50 ulp from the
  `k*delta` product, and traced the defect's origin: prompt 08's test 1 is the same $z_s$ band at
  $k=10^6$ (where $10^{-8}$ rad is 86 ulp) and README §6's prompt-06 radiation control is at a
  $10^7$ rad span (5.4 ulp), so both sources make $10^{-8}$ look right; prompt 09 scaled the $k$
  by 100 to reproduce the review's $8.26\times10^{-3}$ rad `phase_spline` figure and did not scale
  the tolerance. Its verdict was **accept, amend the prompt text**. The review also notes that
  "unreachable in double precision" overstates slightly — a correctly-rounded double would be
  0.5 ulp — but reaching that needs `delta` to return a double-double and a compensated product,
  which is prompt 03's file and would be chasing a number below the campaign's own declared
  $\varepsilon k\tau$ floor (§5 note 2). **Prompt 09's §4 test 1 text was therefore corrected in
  place**, with a dated note recording what it said before and why it changed. No code changed:
  the shipped test already asserts the corrected bounds. See
  [`logs/09-gk-consumer-primitive-phase.md`](logs/09-gk-consumer-primitive-phase.md) deviation 1
  and its orchestrator addendum.

- **[06-residual-table-per-object]** *(opened by prompt 06, 2026-09-11; resolved by prompt 14,
  2026-09-11)* — `WKB_phase_function` built the residual table on every call, although it depends
  only on `(model, k, sector)`: 5,536 of the 6,000 integrand evaluations and ~92 % of the 0.031 s
  per object at $k=3\times10^8$ on LambdaCDM, 5–7k evaluations and 82–255 ms on `QCD_Cosmology`,
  times ~1,700 source redshifts per $k$. **Resolution:** `residual_node_range` fixes the nodes
  from $(model, k, sector)$ alone — the background grid, cut at the top where the frequency stops
  keeping half its leading term (`[14-residual-range-top-margin]`) — and `cached_phase_residual`
  memoises one `CumulativeTable` per key in the worker (LRU, 256 entries, 0.395 MB each, ~40 MB
  for a production run's 100 keys). The object's anchor is off that grid and is reached by one
  `CumulativeTable.delta` partial **per object**, split at the nearest node, rather than per
  sample. Measured over 50 objects of one $k$ at $k=3\times10^8$: **142.5 residual-integrand
  evaluations per object on LambdaCDM against 6,924, and 163.0 on QCD against 7,908 — 49× on
  both**; the second and every later object adds nothing to the build (exactly 0 for an on-grid
  anchor, exactly 4 for an off-grid one), and wall time per object falls from 0.0309 s to
  **0.0010 s**. Nothing moved: $\theta$ is **bit-identical** at every sample of fifteen
  (model, $k$, sector) cases and $\rho$ moves by at most $1.4\times10^{-17}$ rad. Every §3.1
  accuracy, cost and sweep threshold of prompt 06 passes at its published number. Prompt 07
  inherits the reuse with no change. See
  [`logs/14-residual-table-reuse.md`](logs/14-residual-table-reuse.md).

- **[02-cosmology-break-point-api]** *(opened by prompt 02, 2026-09-10; resolved by prompt 03,
  2026-09-11)* — the table builders needed a public way to ask a cosmology for its break points.
  **Resolution:** `LambdaCDM_GenericEOS.integration_break_points(z_lo, z_hi) -> np.ndarray` returns
  the ascending $u=\log(1+z)$ values strictly inside the range at which any background quantity
  loses smoothness: the interior knots of the $T(z)$ spline (kept at build as
  `_T_z_spline_knots_log1pz`) plus the crossings of `GenericEOSBase.break_temperatures_GeV` (a new
  property, `()` by default; `QCD_EOS` returns `(T_LO, EOS_T_LO, T_120_MEV, T_HI)`), each solved in
  $u$ by `root_scalar` to `xtol=rtol=1e-15`. `compute_background` and `_create_functions` reach it
  duck-typed through `_cosmology_break_points(cosmology, z_lo, z_hi)`, empty when the cosmology has
  no such method (`LambdaCDM`, the stand-ins). On the production grid `QCD_Cosmology` reports 407
  points (404 knots + 3 crossings), each crossing within 1e-9 of log 02's values. See
  [`logs/03-tau-primitive.md`](logs/03-tau-primitive.md).

- **[00-tau-storage-decision]** *(planning, 2026-09-10; confirmed by the user 2026-09-10; resolved
  by prompt 03, 2026-09-11)* — README §7 D1 implemented: `BackgroundModelValue` gains
  `tau_lo_Mpc` (`Float(64)`, `nullable=False`) after `tau_Mpc`, which is now the high limb; both
  limbs round-trip exactly in `Mpc_units` (asserted for every node of both models). A datastore
  lacking the column is refused by `sqla_BackgroundModelFactory.build()` with a message naming the
  regeneration. The `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F` columns are prompt 04's.

- **[01-qcd-eos-branch-boundaries]** *(opened by prompt 01, 2026-09-10; resolved by prompt 02,
  2026-09-10)* — no fixed-order Gauss–Legendre rule converged on the production intervals of
  `QCD_Cosmology` containing a `QCD_EOS.G(T)` branch boundary: 6.33e-5 relative per interval at
  order 4, falling only as $N^{-2}$, i.e. 4.78 rad at $k=3\times10^8$. **Resolution:** neither of
  the two options the issue named. Prompt 02 measured all three schemes and found (i) a third break
  point, the `EOS_T_LO` clamp in `QCD_EOS.w`, which kinks $c_s^2$ at $z=1.187\times10^{10}$;
  (ii) that **adaptive quadrature is not needed for $\rho$ at all** — $\rho$ is small enough that
  even an unconverged rule delivers it to $10^{-9}$ rad, and what actually fails under `plain` is
  the *leading* $\tau$ term; and (iii) that splitting each production interval at the three
  temperature crossings **and the 404 $T(z)$-spline knots** puts order 4 at the floor everywhere —
  $\tau$ 1.89e-14 relative, 0 of 1,731 intervals above $10^{-12}$ — for 24 % more integrand
  evaluations. Splitting at the temperatures alone is not enough (1.81e-13, 25× the floor); the
  knots are the load-bearing half. Orders: $N_\tau=N_{\tau_s}=N_F=N_\rho=4$.
  See [`logs/02-qcd-residual-convergence.md`](logs/02-qcd-residual-convergence.md) and
  [`docs/gktk-remedial/RESIDUAL-CONVERGENCE.md`](../../docs/gktk-remedial/RESIDUAL-CONVERGENCE.md).

---

## 5. Standing notes for implementers

1. **`RECONCILIATION.md` outranks the review** where they differ; the differences are listed in its
   §2 and each is carried by a named prompt. A *new* conflict must be recorded and, if
   load-bearing, stopped on.
2. **The floors are not targets.** $\varepsilon k\tau$ ($3\times10^{-7}$–$9\times10^{-4}$ rad),
   the LG truncation ($10^{-8}$ LambdaCDM, $10^{-3}$ QCD for $G_k$; $1.4\times10^{-4}$ of the
   envelope for $T_k$ at the hand-over), and the $T_k$ initial condition ($2.5\times10^{-6}$).
   A test asserting below a floor is asserting agreement between two errors.
3. **Never form $C=\omega^2-(k/H)^2$** by subtraction; use the correction functions prompt 05
   exposes — `Gk_omegaEff_sq_correction` / `Tk_omegaEff_sq_correction`, both $k$-independent, both
   in `ComputeTargets/WKB_{Gk,Tk}.py`. Never form $\Delta\tau$ as `tau(b) - tau(a)`; use
   `tau.delta(a, b)`.
4. **`*_omegaEff_sq` return values are stored columns** and must not move by a bit. Delivered by
   prompt 05, which kept the `A + B + C` summation order and asserts exact equality against
   verbatim copies of the pre-refactor bodies
   (`test_omega_eff_split.TestReturnValueUnchanged`). Note that `leading + correction` is **not**
   bit-equal to `*_omegaEff_sq` — floating-point addition is not associative and the two differ
   by one ulp on ~13 % of the production range (log 05, deviation 1) — so do not "simplify"
   `*_omegaEff_sq` to that sum.
5. **`ModelFunctions` stand-ins must keep constructing** with thirteen positional arguments.
   Delivered by prompt 04: `cs_tau` and `friction_F` are appended as fields 14 and 15 with
   namedtuple `defaults=(None, None)`, asserted by
   `test_background_cs_tau_friction.test_model_functions_still_constructs_with_thirteen_positional_arguments`.
   **Delivered by prompt 10 for the transfer-function fixtures:** `test_tk_source_functions.FakeModel`
   now supplies both as `wkb_reference.ClosedFormPrimitive` objects (`cs_tau = sqrt(w) tau`,
   `friction_F = (3/2)(1+w) log(1+z)`), and `test_phase_groups` inherits them by importing
   `FakeModel`. A `ModelFunctions` that leaves either at `None` is now refused **by name** by
   `TkSourceFunctions.__init__`, so a stand-in that reaches the transfer-function consumer must
   supply both. `ClosedFormPrimitive.delta` is `f(b) - f(a)`, which is acceptable only in a
   fixture — see its docstring and README §2 (c).
6. **The `GkSource` rectifier stays** (D5). Its logic is a stop condition.
7. **`phase_spline`'s signature is frozen** (D4); `bessel_phase` on `main` and three fixtures
   depend on it.
8. **Solver labels must be registered in `main.py`** (`:2810-2834` after the `transfer-remedial`
   merge) or `store()` raises `KeyError` in production only. Prompt 03 registers the τ table as
   `BackgroundModel.TAU_SOLVER_LABEL` = `"cumulative-GL-stepping4"` through the class attributes
   `TAU_SOLVER_LABEL_BASE` / `TAU_GAUSS_ORDER`, so `main.py` needs no new import.
9. **Tolerances are datastore lookup keys**; a missed `atol=` site makes lookups miss silently
   (prompt 12).
10. **Author conventions** (README §5 rule 6): $a_0$ absorbed; $\tau=a_0\eta$; unit-jump $\bar G_k$;
    $\theta<0$ decreasing with $\theta_{\rm mod}\in(-2\pi,0]$; `tau_init`'s radiation-era closed
    form; $c_s^2=$ `wPerturbations`. Do not "correct" any of these.
11. **`main.py` cannot be imported**; extract functions with `ast` (`test_main_plumbing.load_main_py_functions`).
12. **Redshift arithmetic** (`CLAUDE.md`): node lookup by exact `z`; a recovered $z$ from
    $\log(1+z)$ is only ever a quadrature endpoint.
14. **Wall-clock durations on the development machine are unreliable; CPU time and best-of-N
    are not.** It sleeps in transport, and `unittest`'s "Ran N tests in T s" is elapsed time: a
    `LiouvilleGreen/tests` run at Workstream C close-out reported 1654.9 s against ~1080 s of CPU
    (`ps -o pid,etime,time`), a 53 % overstatement. This bites **duration** figures — suite times,
    build times, the `compute_time` a payload records. It does **not** bite the per-object **cost**
    acceptance rows of README §6, which are best-of-N minima: a sleep can only inflate a sample,
    never shrink it, so it cannot manufacture a pass, and the $T_k$ figure was reproduced as two
    independent tight bands (0.0494–0.0516 s and 0.0505–0.0528 s). **Prompt 13 should take its
    timings as CPU time or best-of-N, and say which**; a lone elapsed figure from this machine is
    not evidence.

13. **The `transfer-remedial` campaign has landed** (merged at `e01c31d`; README §0.2, §4.2):
    still do not touch its files, and its `main.py` Bessel-stage comment is theirs. The two
    shared test files are no longer a live stop condition — prompt 10 left every `8ba9159` line
    in both of them untouched (`[00-transfer-remedial-test-file-overlap]`, §4) — but their
    tolerance *comments* are now stale and must not be trusted as calibration
    (`[10-transfer-remedial-tolerance-comments-stale]`, §3).
14. **A producer calls `cached_phase_residual`, never `build_phase_residual`** (prompt 14). The
    residual table is one per `(model, k, sector)`, built on `residual_node_range(model, k,
    leading_table.z_nodes, sector)` and memoised in the worker; pass the proxy's `store_id`, and
    nothing derived from the object. The object's anchor is off that grid: split it once at
    `nearest_table_node(rho, z_init)` and add `rho.delta(node, z)` per sample — calling
    `rho.delta(z_init, z)` per sample costs a Gauss panel each time. The *leading* table's anchor
    is **not** split this way, which is half of `[07-tk-per-object-cost-is-all-setup]`.
15. **The retired friction ODE lives in a test** (prompt 07). `friction_RHS` and `FRICTION_INDEX`
    are gone from `ComputeTargets/TkWKBIntegration.py`; the function is verbatim as
    `_friction_RHS` at `ComputeTargets/tests/test_background_cs_tau_friction.py:79`, used only by
    prompt 04's `TestFrictionODEComparison`. Producers read
    `friction_F.delta(z_init, z) = F(z) - F(z_init) < 0` from the background table, with **no sign
    flip**, and multiply the amplitude by `exp()` of it.
