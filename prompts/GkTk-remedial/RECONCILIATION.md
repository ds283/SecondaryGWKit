# Reconciliation — the review checked against the tree at `9ff59d5`

**Campaign:** [`README.md`](README.md) · **Review:** `docs/gk-wkb-review-fable-2026-09-09.md`
**Date:** 2026-09-10 · **Tree:** `main` at `9ff59d5` (clean)

The review inspected `f06f587`. This document records (§0) what changed between that tree and the
one the campaign is planned against, (§1) which review claims were re-verified by inspection of the
current code and reproduce, (§2) what the review omits or states loosely and the prompts therefore
add, and (§3) options the review offers that the plan drops. Where a prompt and the review disagree,
this document is the reason.

---

## 0. Tree drift since the review

`git merge-base --is-ancestor f06f587 9ff59d5` holds. `git diff --stat f06f587 9ff59d5` over every
file the review names touches only:

| File | Change | Effect on findings |
|---|---|---|
| `ComputeTargets/GkSourcePolicyData.py` | +16/−?: `_classify_Levin` returns `Levin_z: None` when no threshold crossing exists (`82896f6`) | None. `_create_functions` (`:587-702`) and its chunked `phase_spline` call (`:671-680`, comment `:666-670`) are as the review describes at "14 lower at `f06f587`" |
| `main.py` | +242/−13 (`f17f2d4` names the k-array sample counts, `5255ac0`, and the merge of `transfer-remedial-plan` docs) | None on the hunks this campaign edits: `delta_logz=1.0/float(source_samples_per_log10z)` at `:618` and `:1187`; `truncate(0.85 * k_exit.z_exit_subh_e6, …)` at `:598` and `:1165`; `solvers` at `:2809-2821`; tolerances at `:2791-2794`; the background model built on `z_source_sample` at `:471-480`. The Bessel-stage comment the review cites as `main.py:424-427` is now at `:516-519` and belongs to `transfer-remedial` prompt 06 |

Every other named line reproduces at the review's line number: `WKB_phase_function.py:262-264`
(the "$Q$ … fairly close to unity" comment), `:359` (the `z = z_init − u` round-trip assertion
against `DEFAULT_ABS_TOLERANCE`), `:403` (the stale "stepped about 1E3 in redshift" comment; the
constant is `DEFAULT_PHASE_RUN_LENGTH = 1e4` rad at `:28`), `:671` (`fabs(z_init − z_sample.min.z)
< atol`); `numeric_with_phase_cut.py:50` (`mode.lower()` before the `None` test);
`GkWKBIntegration.py:403-410` (sign fix), `:414-418` (`shift_theta_sample`), `:425, :432` (`tau`);
`GkNumericIntegration.py:391, :399`; `TkWKBIntegration.py:25-49` (`friction_RHS`), `:452-459`,
`:463-467`, `:476`; `QuadSourceIntegral.py:915-916, :1565-1568`;
`Datastore/SQL/ObjectFactories/BackgroundModel.py:605` (`tau_Mpc`), `:399, :680` (written as
`value.tau / Mpc`), `:288, :695` (read as `row.tau_Mpc * Mpc`); `BackgroundModel.py:144-153`
(`solve_ivp`, RK45, `EXPECTED_SOL_LENGTH = 1`), `:387-402` (`_build_func` with the cubic
`make_interp_spline` default); `Quadrature/supervisors/numeric.py:80`
(`grid_spacing = (1.0 + z) * self._delta_logz`); `LiouvilleGreen/integration_tools.py:10, :26`
(the `1e-3` relative step).

The environment matches the review's: Python 3.12, SciPy 1.15.2, NumPy 2.2.4, mpmath 1.3.0.

---

## 1. Review claims re-verified by inspection

None of the review's *measurements* were re-run for this plan; prompt 01 re-runs the two cheap
ones ($k=10^5$, both sectors) as its baseline. What follows is what the code shows on reading.

1. **The two-stage solver is exactly as described.** Stage 1 (`WKB_phase_function.py:34-235`)
   integrates $\theta'=\omega$ with two terminal events — a reset at $|\theta|=10^4$ rad and a
   switch at $\omega^2=10^6$ — re-basing the state to $\theta_{\rm mod}$ and accumulating
   `current_div_2pi_offset`. Stage 2 (`:238-380`) integrates $Q'=-[\omega/\omega_i+Q]/(1+u)$ from
   $Q(0)=0$ and reconstructs $\theta$ through `WKB_product_mod_2pi(omega_init*(1+u), Q,
   theta_mod_2pi_init)`. Both use `t_eval` (dense output). The recycle branch is tested before the
   terminate branch (`:179, :199`), so two events in one step continue stage 1 (review §8.2).
2. **The NumPy deprecation is real.** `sol.y_events[0]` is a `(1, 1)` array; `values[0]` is a
   1-element array passed to `WKB_mod_2pi` → `math.fmod` (`:184-191`; same at `:201-208`).
3. **The sign fix is a no-op.** `raw_cos = sqrt(omega_i) * G_init`, `deltaTheta = atan2(raw_cos,
   raw_sin)`, so ${\rm sgn}(\sin\delta)={\rm sgn}(G_{\rm init})$ whenever $G_{\rm init}\ne0$ and
   both are `+1` by the `>= 0.0` conventions when it is (`GkWKBIntegration.py:404-410`;
   identical code at `TkWKBIntegration.py:453-459`). Audit TK-8 reached the same conclusion.
4. **`shift_theta_sample` rebases across samples.** `theta_div_2pi_shift_base = theta_div_2pi_shift[0]`
   is subtracted from every sample's shift (`WKBtools.py:105-109`). Within an object this is a
   constant offset; across objects the base differs, which is review §8.3's ±1-cycle mechanism.
5. **`phase_spline` chunking is as described.** `_build_log_chunks_negative` (`:478-499`) places
   boundaries at `round(start*logstep + 1)`; `_chunk_spline` rebases to the chunk's *first* key,
   which for negative phases is the far boundary (`:49-51`); `_match_chunk` (`:501-655`) selects one
   chunk by a distance-to-centre rule (a hard switch); `_build_log_chunks_positive` (`:460-476`) has
   the progress defect: with `chunk_start = 1` and `logstep < 2`, `chunk_end = round(logstep+1) =
   2` and the next `chunk_start = round(0.75*2 − 0.5) = 1`. Production uses `chunk_logstep=125`,
   which progresses.
6. **Consumers of `phase_spline`.** Production: `GkSourcePolicyData._create_functions` (`:671`,
   `chunk_logstep=125`, `increasing=False`), `GkSourcePolicyData._classify_Levin` (`~:170`,
   single chunk), `TkSourceFunctions._build_WKB` (`:272-281`, `PHASE_SPLINE_CHUNK_LOGSTEP = 125`
   at `:127`), and `LiouvilleGreen/bessel_phase.py:262-269` (`chunk_logstep=125`; **removed by
   `transfer-remedial` prompt 05 on its branch, still present on `main`**). Protocol consumers
   (duck-typed, never construct one): `AdaptiveLevin/levin_quadrature.py` through the phase dict,
   `ComputeTargets/phase_groups.py` (`_OscillatoryG.theta_deriv`, `_composed_*`),
   `QuadSourceIntegral._ClampedPhase` (`:157-181`) and its `WKB_phase_spline_chunks` metadata
   (`getattr(Gk_f.phase, "num_chunks", None)`, `:912-914`), `spline_wrappers.GkWKBSplineWrapper`.
   Test fixtures constructing one with `chunk_logstep=125`: `test_phase_groups.py:317-324,
   :1109-1116`; `test_quadsource_integral.py` (through `bessel_phase`). This is why README §2 (g)
   freezes the signature.
7. **`WKBtools` consumers.** `WKB_mod_2pi`, `WKB_product_mod_2pi`: `WKB_phase_function.py` only
   (plus `docs/` scripts). `shift_theta_sample`: the two `store()` methods (plus
   `docs/gk-wkb-review-fable-2026-09-09/t6_sweep.py`, `docs/spec-code-audit/scripts/GK_05_phase_reassembly.py`).
   `wrap_theta`: three test modules (`test_tk_source_functions.py:44`,
   `test_quadsource_integral.py:232`, `test_phase_groups.py:67`) — **must survive**.
8. **`has_unresolved_osc` has no programmatic consumer** (grep: only the two factories persist it
   and the two integrators expose it) — and per review §13.1 the printed warning is the intended
   consumer. Both `GkNumericIntegration.RHS` (`:68-76`) and `TkNumericIntegration.RHS` (`:82-91`)
   call `*_omegaEff_sq` per evaluation and route through the shared
   `NumericIntegrationSupervisor.report_wavelength` (`numeric.py:73-90`).
9. **`main.py` passes $\Delta\log_{10}$** (`1.0 / float(source_samples_per_log10z)`, `:618` and
   `:1187`) for **both** the $T_k$ run (sampled on the source grid) and the $G_k$ run (sampled on
   the *response* grid, `winnow(sparseness=response_sparseness)` at `:422`, 12× sparser). The
   review notes the $\ln10$ slip; the grid mismatch for $G_k$ is additional (§2 item 8 below).
10. **`functions.tau` consumers**: `GkWKBIntegration.py:425, :432`; `TkWKBIntegration.py:476`;
    `GkNumericIntegration.py:391, :399`; `TkNumericIntegration.py:430`;
    `QuadSourceIntegral.py:915, :916, :1565, :1566, :1568`; `main.py:506` (`largest_tau`, the
    Bessel $x_{\max}$ — a pointwise use that stays pointwise). All are pointwise `tau(z)` calls, so
    a callable object with `__call__` keeps every one working.
11. **The background model grid is the source grid** (`main.py:476`, `z_sample=z_source_sample`),
    and the response grid is a subset of it (`:415-425`). So every $G_k$ and $T_k$ WKB sample is a
    table node; only $z_{\rm init}$ (the numeric stop point, a `root_scalar` root) is off-grid.
12. **Stage-2 `IntegrationData` can already be `None`** in the payload when stage 1 exhausts the
    samples (`WKB_phase_function.py:535-536`), and `GkWKBIntegration.__init__` initialises both
    stages with an all-`None` `IntegrationData` (`:60-75`). Prompt 06 returns the all-`None`
    structure for stage 2 rather than `None` and confirms the factory's store path accepts it.
13. **Solver labels are registered in `main.py`** (`:2809-2821`) and looked up by
    `store()` (`self._solver_labels[data["phase_solver_label"]]`, `GkWKBIntegration.py:489`;
    `TkWKBIntegration.py:530-531`; `BackgroundModel.py:535`). A new label that is not registered
    raises `KeyError` in production but not in the offline tests. Prompts 03, 06 and 07 therefore
    each carry a `main.py` hunk at `:2805-2822`. The review does not mention this.
14. **Tolerances are shared.** One `atol`/`rtol` pair (`main.py:2791-2792`) is passed to every
    integration object and is part of every datastore lookup key. Prompt 12's separate $T_k$
    numeric `atol` must be threaded through **every** `pool.object_get("TkNumericIntegration", …)`
    (`main.py:601-621, :745-755, :940-960, :2536-2545` on `9ff59d5`; the agent greps), otherwise
    lookups silently miss and the pipeline recomputes.

---

## 2. What the review omits or leaves loose — corrections the prompts carry

1. **`ModelFunctions` is a namedtuple constructed positionally by stand-ins.**
   `test_tk_source_functions.FakeModel`, `test_phase_groups`, `docs/gk-wkb-review-fable-2026-09-09/realbg.py`
   and the audit scripts all build `ModelFunctions(...)` with exactly the thirteen current fields.
   Adding `cs_tau` and `friction_F` (prompt 04) must use namedtuple `defaults` so those constructors
   keep working; prompt 10 then *upgrades* the two test fixtures it owns to supply real accessors.
2. **The `tau` accessor must remain callable.** Item 1.10. Prompt 03's `tau` is an object with
   `__call__(z) -> float` and `delta(z_a, z_b) -> float`; the `_build_func("tau")` analytic
   shortcut (`hasattr(cosmology, "tau")`) is dropped for `tau` — no cosmology defines it, and an
   analytic pointwise $\tau$ could not supply `delta` at the required accuracy anyway.
3. **`C` must not be formed by subtraction.** Review §6 writes $\omega^2=(k/H)^2+C$; the code's
   `Gk_omegaEff_sq` returns `A + B + C` with `A = (k/H)^2 ≈ 10^{12}`–$10^{24}$ times the rest.
   `omega_sq − (k/H)^2` would lose everything. Prompt 05 splits the two frequency modules into a
   leading part and a correction, keeping the sum bit-identical to today's return value (a test
   asserts it).
4. **$\theta+\delta$ "stored exactly" needs a per-sample wrap, not a reconstruction.** Review §8.1
   says "store $\theta+\delta$ exactly". Forming `div*TWO_PI + mod + delta` and re-reducing is the
   anti-pattern the range-reduction docstring forbids. `wrap_theta(mod + delta)` per sample, with
   the returned cycle shift added to that sample's `div` and **no base subtraction**, is the exact
   operation; prompt 06 adds it to `WKBtools` as `apply_phase_offset`.
5. **The primitive's per-object anchor is off-grid.** For numeric-initialised objects,
   $z_{\rm init}$ is a `root_scalar` root, not a node (item 1.11), so every `GkWKBIntegration`
   built from a hand-over needs exactly one partial Gauss integral (over less than one grid
   interval) at the anchor end, and none at the sample end. This is the case the interval accessor's
   "partial to node" term exists for; prompt 03's tests must exercise an off-grid endpoint.
6. **The `GkSource` rectifier is still needed after the primitive — for a different reason than
   the review states.** Review §7 says the primitive makes the rectifier "unnecessary" because
   phases become "consistent by construction". That is true of the `shift_theta_sample` rebase
   mechanism. It is **not** true of the second mechanism review §8.3 itself measured — "every 2π
   wrap of $\delta$ between neighbours": $\delta={\rm atan2}(\cdot)\in(-\pi,\pi]$ is computed per
   object, and when the numeric stop point $z_{\rm init}(z_s)$ moves to the next extremum between
   neighbouring source redshifts, $-k\Delta\tau(z_{\rm init}\to z_r)$ changes by $\approx2\pi$ and
   $\delta$ wraps by $-2\pi$; the physical phase is smooth but the stored `div` is not. The
   rectifier's monotonicity test catches exactly this (the per-sample increment is 0.5–0.8 rad
   $<\pi$ in the numeric-initialised band, review §8.3). **Decision D5**: the rectifier stays;
   prompt 09 builds $\varphi$ from the *rectified* `theta_div_2pi`, and tests both mechanisms on a
   stand-in.
7. **The consumer decomposition needs no $z_{\rm init}$.** For prompt 09,
   $\varphi(z_s)=\theta_{\rm stored}(z_s)+k\,$`tau.delta`$(z_s, z_r)$ — both terms $O(10^{12})$ at
   $k=3\times10^8$, so $\varphi$ carries $\varepsilon k\tau\approx9\times10^{-4}$ rad of rounding
   noise there (the fact-(d) floor, D6) and $\sim10^{-7}$ rad at $k=10^5$. $\varphi$ is $-\Delta\rho$
   for pure-WKB objects and $O(1)$ rad, smooth, for numeric-initialised ones. It is splined; the
   leading term is evaluated from the table.
8. **The `has_unresolved_osc` grid is the wrong grid for $G_k$** (item 1.9), on top of the $\ln10$
   slip. Prompt 11 evaluates the test against the actual spacing of the sample grid the caller
   supplied — which is what the flag is documented to mean (review §13.1) — and README §7 D2 records
   the consequence.
9. **A `find_phase_minimum` that steps in phase needs $\omega$.** The review recommends stepping
   in phase "using $\omega$" (§10.2). `numeric_with_phase_cut` is generic over the RHS and has no
   $\omega$; prompt 11 adds an `omega_sq` callable parameter (the same function 11 needs to evaluate
   the wavelength on the sample grid) and both integrators pass theirs.
10. **The `GkWKBIntegration`/`TkWKBIntegration` `atol`/`rtol` columns lose their referent.** The
    primitive has no tolerances. The columns stay (schema churn for no gain; they remain part of the
    lookup key) and the payload `metadata` records the Gauss orders actually used. Same call the
    `transfer-remedial` campaign made for `bessel_phase` (its M14).
11. **Two latent defects in the `BackgroundModelValue` factory's `build()` path** are noted, not
    fixed: `:674` inserts with key `"wkb_serial"` where the column is `model_serial`, and `:705`
    compares `row_data.Hubble` which does not exist (`Hubble_GeV` does). Both are on the
    query-existing-row branch that production never takes (values are inserted through `store()`).
    Prompt 03 edits the neighbouring lines and **must not** repair these silently; it records them,
    and the board carries `[03-backgroundmodelvalue-build-path]` if it confirms them.
12. **The persisted double-double pair and units.** `value.tau / Mpc` then `* Mpc` is exact only
    because `Mpc = 1.0` in `Mpc_units`. Prompt 03 asserts the exact round trip in a test and
    documents that a unit system with `Mpc ≠ 1` would need a compensated scaling of the pair.
13. **`increasing=False` in `phase_spline`.** `GkSourcePolicyData` passes it because $\theta$ at
    fixed $z_r$ *decreases* with $z_s$; `TkSourceFunctions` passes `True`. After prompt 08 the flag
    only orders the single chunk's data and is harmless; after 09/10 the two production callers no
    longer construct a `phase_spline` at all. `PrimitivePhase` fixes the sign convention explicitly
    (README §2 (g)) rather than through a flag.

---

## 3. Review options the plan drops, and why

- **Quintic spline of the τ nodes as an intermediate** (review §13.2(c)). Five orders for a
  one-argument change, but it lands in prompt 03's own file and would be replaced in the same
  commit series. Dropped; if Workstream B is abandoned after 02, it is the obvious one-line
  fallback and is recorded here for that case.
- **Rebuild-on-load for the node table** (review §13.2 second option). Rejected for cost per Ray
  task; README §7 D1.
- **Per-region anchoring** (review §13.4). Deferred; `[00-consumer-anchoring-floor]`.
- **Wronskian construction of $G$ from two homogeneous solutions** (review §10.2). Not
  recommended by the review; not scheduled.
- **Super-horizon series initial data for $T_k$** (review §12.5). A spec decision;
  `[00-tk-superhorizon-ic-series]`.
- **Deleting the `mode != "stop"` branch of `numeric_with_phase_cut`** (review §10.2 offers fix or
  delete). Prompt 11 fixes the guard; deleting a code path production does not exercise buys
  nothing and removes an option.
- **Stepping in phase for `find_phase_minimum`** is *kept* (prompt 11) but the search window is
  not widened — widening is the hand-over campaign's.
