# Log 06 — The Green's-function WKB phase from the primitive

**Prompt:** prompts/GkTk-remedial/06-gk-wkb-phase-from-primitive.md
**Commit:** *"Compute the Green function WKB phase from the conformal-time table"* (SHA not
embedded, per the campaign convention)
**Model:** Claude Fable 5.1
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

Two deviations are `STRUCTURALLY REQUIRED` (§Deviations 1 and 2). Neither touches a README §2
design fact: one is a dead-code import the prompt mis-described, the other is the *definition* of
a count in test 5, which the production geometry made undefined as the prompt phrased it. Every
acceptance threshold in the prompt and README §6 is met, most by two to four orders of magnitude.

## What shipped

**The finding, first.** The phases this campaign's datastores hold for the Green's function were
wrong by whole cycles: 13.9 rad at $k=10^5/{\rm Mpc}$ and 7366 rad at $k=3\times10^8/{\rm Mpc}$
at $z=0.1$ on the production background (review §4), so `theta_mod_2pi` — and hence `G_WKB` —
below $z\sim10^3$ was noise for the largest $k$. This commit replaces the mechanism that produced
them. Every existing `GkWKBIntegration`/`TkWKBIntegration` row must be regenerated (the
datastore already has to be, since prompt 03).

### `Quadrature/integrators/WKB_phase_function.py` — rewritten (718 → 342 lines)

Deleted: `stage_1_evolution`, `stage_2_evolution`, `integrate_phase_function`,
`integrate_friction_function`, `THETA_INDEX`, `Q_INDEX`, `FRICTION_INDEX`,
`EXPECTED_PHASE_SOL_LENGTH`, `EXPECTED_FRICTION_SOL_LENGTH`, `DEFAULT_PHASE_RUN_LENGTH`,
`DEFAULT_OMEGA_WKB_SQ_MAX`, the `solve_ivp`/`ThetaSupervisor`/`QSupervisor`/
`NumericIntegrationSupervisor`/`RHS_timer`/`WKB_product_mod_2pi` imports, the `atol`/`rtol`
parameters and the `config.defaults` import. `grep -n "solve_ivp\|Q_INDEX\|DEFAULT_PHASE_RUN_LENGTH"`
on the file is empty (asserted by `TestSourceHygiene`).

New module-level names:

```python
PHASE_SOLVER_LABEL_BASE = "wkb-primitive"
PHASE_SOLVER_STEPPING = RHO_GAUSS_ORDER            # = 4
PHASE_SOLVER_LABEL = "wkb-primitive-stepping4"
SECTOR_LEADING_PRIMITIVE = {"Gk": "tau", "Tk": "cs_tau"}

residual_nodes(grid, z_sample, z_init) -> np.ndarray   # public helper, see below

@ray.remote
def WKB_phase_function(model_proxy, k, z_init: float, z_sample, *,
                       sector: str, omega_sq, d_ln_omega_dz, friction: bool = False,
                       task_label: str = "WKB_phase_function",
                       object_label: str = "(object)") -> dict
```

Implementation, per prompt §2:

1. `model = model_proxy.get()`; `leading = model.functions.tau` (`"Gk"`) or `.cs_tau` (`"Tk"`),
   refused with a `RuntimeError` naming the regeneration if the accessor is `None` or has no
   `delta`. The residual table is `build_phase_residual(model, k, nodes, sector)` with
   `nodes = residual_nodes(leading.table.z_nodes, z_sample, z_init)`: **the background model's
   own grid restricted to $[\min z_{\rm sample}, z_{\rm init}]$, plus every sample redshift,
   plus $z_{\rm init}$ itself as the top node**, strictly descending. So `rho.delta(z_init, z)`
   is on-grid at both ends and never needs a partial; the only off-grid partials are the leading
   table's, one per sample when $z_{\rm init}$ is off-grid. (Prompt §2 item 1 offered "let
   `delta` supply the partial at `z_init`" as the simplest route; this is the alternative it
   asked me to state — see Deviations 4.)
2. Per sample: `theta = -(k * leading.delta(z_init, z) + rho.delta(z_init, z))`;
   `theta_div_2pi, theta_mod_2pi = WKB_mod_2pi(theta)`. Formed once, reduced once, per sample;
   nothing rebased.
3. WKB-criterion diagnostic: the initial-time `RuntimeError`s for `omega_sq_init < 0` and
   criterion $>1$ are kept verbatim; then `|d_ln_omega_dz|/sqrt(omega_sq)` at every sample, the
   first exceeding 1 sets `has_WKB_violation`, `WKB_violation_z`,
   `WKB_violation_efolds_subh = log((1+z) k/H)` and prints one warning line pair in the
   supervisors' format. **These are the supervisors' semantics evaluated on the sample grid**
   rather than at the ODE's steps. A negative `omega_sq` at a sample raises `ValueError` as the
   RHS did.
4. Zero-length case, exact: `len(z_sample) == 1 and z_sample[0] == z_init` → the
   initial-data-only payload (`[0]`, `[0.0]`, all-`None` `IntegrationData` for both stages,
   `metadata["initial_data_only"] = True`; plus `friction_sample=[0.0]`, all-`None`
   `friction_data`, `friction_solver_label` when `friction`).
5. Payload keys: `stage_1_data` (an `IntegrationData` with `compute_time` = wall time of the
   call, `compute_steps` = number of samples, `RHS_evaluations` = residual-build integrand
   evaluations + leading-table partial evaluations, the three per-RHS timings `None`),
   `stage_2_data` (all-`None` `IntegrationData`, **not** `None`), `theta_div_2pi_sample`
   (`List[int]`), `theta_mod_2pi_sample` (`List[float]` in $(-2\pi,0]$),
   `phase_solver_label = "wkb-primitive-stepping4"`, `has_WKB_violation`, `WKB_violation_z`,
   `WKB_violation_efolds_subh`, `metadata`; with `friction=True` also `friction_sample`
   (`[friction_F.delta(z_init, z) for z in samples]`), `friction_data` (all-`None`
   `IntegrationData` — `TkWKBIntegration.store()` reads it at `:398`), `friction_solver_label`
   (the same label). `metadata` is compact because it is persisted as JSON in a
   `String(256)` column: `solver`, `sector`, `N_rho`, `N_lead`, `rho_nodes`, `rho_evals`,
   `lead_partials`, `lead_evals`, `offgrid_init`, `rho_end` (the residual at the last sample),
   and `initial_data_only` when applicable — 206 characters in the production case, 233 with
   `initial_data_only`.
6. The Datastore factories need no change: `sqla_GkWKBIntegration_factory.store()` reads
   `obj.stage_2_data.<field>` guarded by `is not None` on the object, so an all-`None`
   `IntegrationData` writes NULLs into the nullable `stage_2_*` columns
   (`Datastore/SQL/ObjectFactories/GkWKBIntegration.py:146-157`, no `nullable=False`); the Tk
   factory's `friction_*` columns are the same (`TkWKBIntegration.py:158-163`). Verified by
   reading, as prompt §3.5 asked; no factory file is touched.

### `LiouvilleGreen/WKBtools.py`

New:

```python
def apply_phase_offset(div_2pi_sample: Sequence[int], mod_2pi_sample: Sequence[float],
                       delta: float) -> Tuple[List[int], List[float]]
```

`wrap_theta(mod + delta)` per sample, the returned cycle shift added to that sample's `div`; no
base subtraction. `WKB_mod_2pi` and `wrap_theta` unchanged. `shift_theta_sample` and
`WKB_product_mod_2pi` **retained** with "not used by production" comments (D7; Deviations 3):
`shift_theta_sample` is still called by `TkWKBIntegration.store()` until prompt 07.

### `ComputeTargets/GkWKBIntegration.py`

- Imports `apply_phase_offset` (not `shift_theta_sample`) and the three `PHASE_SOLVER_*`
  constants; exposes them as class attributes `GkWKBIntegration.PHASE_SOLVER_LABEL_BASE`,
  `.PHASE_SOLVER_STEPPING`, `.PHASE_SOLVER_LABEL` so `main.py` needs no new import.
- `compute()` (`:320-331` → `:333-348`): `sector="Gk"`, no `atol=`/`rtol=`; a comment records
  that `self._atol`/`self._rtol` stay as lookup keys (`RECONCILIATION.md` §2 item 10).
- `store()`: the sign fix (`:403-410`) is gone — `self._sin_coeff = B`, `self._cos_coeff = 0.0`
  — with a comment stating why $B>0$ already reproduces the initial data; `shift_theta_sample`
  (`:414-418`) → `apply_phase_offset(data["theta_div_2pi_sample"], data["theta_mod_2pi_sample"],
  deltaTheta)`. The long comment block is edited only where it described the sign fix and the
  rebase. The value loop, the $(B,\delta)$ arithmetic, `omega_WKB_sq` (still `Gk_omegaEff_sq`)
  and the `GkWKBValue` schema are untouched. `grep -n "sgn_sin_deltaTheta\|shift_theta_sample"`
  on the file is empty.

### `ComputeTargets/TkWKBIntegration.py` — interim, per prompt §3.4

- `compute()` (`:364-376`): `sector="Tk", friction=True`, no `atol=`/`rtol=`.
- The `FRICTION_INDEX` import is dropped; a module-level `FRICTION_INDEX = 0` replaces it
  (Deviations 1). `friction_RHS` is now dead but importable; `store()` is untouched and still
  runs its sign fix and `shift_theta_sample` on the new payload, which it accepts (the payload
  supplies `friction_data`). Prompt 07 removes all of this.

### `Quadrature/supervisors/WKB.py` — deleted

`ThetaSupervisor` and `QSupervisor` had no consumer left; `Quadrature/supervisors/__init__.py`
is empty and does not re-export them (Deviations 7).

### `main.py` — solver registration only (`:2813-2854`)

One `pool.object_get("IntegrationSolver", label=GkWKBIntegration.PHASE_SOLVER_LABEL_BASE,
stepping=GkWKBIntegration.PHASE_SOLVER_STEPPING)` and the dict entry
`GkWKBIntegration.PHASE_SOLVER_LABEL: wkb_primitive_phase`. `GkWKBIntegration` was already
imported at `main.py:15-18`. `store()`'s `self._solver_labels[data["phase_solver_label"]]` and
`TkWKBIntegration.store()`'s `["friction_solver_label"]` both resolve to it.

### `ComputeTargets/tests/test_gk_wkb_phase.py` — new, 21 tests

Stand-ins: `_Proxy` (`.get()`, `.units`), `_KExit`/`_Wavenumber` (`.k.k`, `.k.k_inv_Mpc`,
`.k.store_id`, `.k.units`); `radiation_model_with_tables`, `lambdacdm_model_with_tables`
(tables from the integrands on the production grid, the cosmology's break points) and
`qcd_model_with_tables` (reconstructed from the `compute_background` payload's limbs, zero
quadrature, as `_build_*_primitive` does); `store_algebra(...)`, a verbatim replica of
`store()`'s $(B,\delta)$ arithmetic; `_sweep(k, x_r, stop)`, the review's t6 geometry. Test
classes follow prompt §4's seven items, plus `TestApplyPhaseOffset` and `TestSourceHygiene`
(the §5 greps and the `main.py` registration as assertions).

## Deviations from the prompt

### 1. `FRICTION_INDEX` is *used* in `TkWKBIntegration.py`, not only imported — STRUCTURALLY REQUIRED

Prompt §3.4 says "drop the `FRICTION_INDEX` import (`:15-18`)". `friction_RHS` reads
`state[FRICTION_INDEX]` at `:38`. Dropping the import alone would have left an importable
function that raises `NameError` when called — dead, but a landmine between this commit and
prompt 07. Editing `friction_RHS`'s body is outside "only the `compute()` call site". What was
done: a module-level `FRICTION_INDEX = 0` with a comment saying prompt 07 deletes it together
with `friction_RHS`. `FRICTION_INDEX` no longer exists in `WKB_phase_function.py`, as §2 item 6
requires.

### 2. Test 5's count is defined against stop-point transitions, not "δ sign wraps" — STRUCTURALLY REQUIRED

Prompt §4 item 5 asks to "count [the one-cycle points] and assert the count equals the number
of δ sign wraps". In the production geometry — the stop point is a **maximum** of $G$,
`G_stop/env = +1.000000` in every run (README §2 (h)) — every object has $\delta=+\pi/2$
exactly (the initial $G'$ is zero at an extremum, so $\delta={\rm atan2}(\sqrt{\omega_i}G_i,0)$),
δ never changes sign, yet the stored cycle count *does* step by one wherever the stop point moves
to the next maximum: 90 one-cycle jumps against 0 sign changes over 990 objects. In the review's
own t6 geometry (alternating maxima and minima, $\delta=\pm\pi/2$) δ flips sign at every
transition but the cycle count steps at every second one: 90 jumps against 180 sign flips. The
literal assertion is false in both geometries. What is true, and is the mechanism
`RECONCILIATION.md` §2 item 6 describes ("when the numeric stop point moves to the next
extremum … the stored `div` is not smooth"), is that the cycle count steps **exactly where the
stop point index changes, and nowhere else**. The test asserts that (`np.array_equal(jumps != 0,
transitions)`) in the production geometry, and in the review's geometry asserts the invariants
(integer cycles to $4\times10^{-12}$ rad; every jump exactly one cycle; jumps ⊆ transitions) and
reports the counts. Prompt 09 should read "δ-wrap point" as "stop-point transition".

### 3. D7: `shift_theta_sample` and `WKB_product_mod_2pi` retained — IMPLEMENTATION CHOICE

Alternatives: delete both (the prompt's first option) or retain with docstrings. Retained,
because `shift_theta_sample` is not yet dead: `TkWKBIntegration.store()` calls it at `:467` and
prompt 07 owns that change; deleting it here would break the transfer-function pipeline between
the two commits. `WKB_product_mod_2pi` *is* dead in production (its only consumer was stage 2)
but is imported by `docs/spec-code-audit/scripts/GK_05_phase_reassembly.py`; retained for
symmetry with the first, each with a comment stating it is unused by the Green's-function
production path and why. Prompt 07 may delete `shift_theta_sample` once `TkWKBIntegration.store()`
no longer calls it; nothing in this campaign's producers may reintroduce either.

### 4. Residual table anchored at `z_init` as its top node — IMPLEMENTATION CHOICE

Prompt §2 item 1: "simplest is to build the residual table on the background model's grid
restricted to $[z_{\min}, z_{\rm init}]$ and let `delta` supply the partial at `z_init`; state
what you did." Alternatives considered: (a) the prompt's — grid nodes $\le z_{\rm init}$, anchor
reached by `rho.delta`'s off-grid partial; (b) the *sample* grid plus `z_init`; (c) the grid
restricted to $[z_{\min}, z_{\rm init}]$ ∪ samples ∪ {`z_init`}. Chose (c). Against (a): the
partial panel is computed either way (as the top panel of the table, or as a partial on every
`delta` call); as a node it is computed once instead of once per sample, and the
"more than one grid interval above the table" refusal of `CumulativeTable` cannot trigger.
Against (b): the response grid is 12× sparser than the grid prompt 02 measured order 4 at the
floor on, and on `QCD_Cosmology` the knots and branch crossings would fall inside 12× wider
panels; building on the background grid costs 4× the evaluations of (b) on LambdaCDM
(5,536 for $k=3\times10^8$, 17–28 ms per log 05) and is the measured configuration. Every
production sample is a background node, so the union adds nothing in production; it protects
stand-ins whose samples are not. Cost: 0.031 s per object at $k=3\times10^8$ on LambdaCDM, against
the 0.05 s target — see Verification 7 and Observations 1 for the per-object rebuild this implies.

### 5. Solver label `"wkb-primitive"`, `stepping = RHO_GAUSS_ORDER` — IMPLEMENTATION CHOICE

Prompt §3.3 offered `"wkb-primitive"` with `stepping=0`, or reusing prompt 03's
`"cumulative-GL"` with the residual order as stepping. Chose a distinct base label with the
residual's order as stepping: reusing `"cumulative-GL"` would make the phase share the background
table's `IntegrationSolver` row and walk straight into
`[03-integrationsolver-stepping-minimum-lookup]`; `stepping=0` would record nothing. The
registered label is `"wkb-primitive-stepping4"`, exposed as `GkWKBIntegration.PHASE_SOLVER_*`
class attributes so `main.py` needs no new import (prompt 03's pattern). The same label serves
`TkWKBIntegration`'s `phase_solver_label` and `friction_solver_label` — the friction integral is
a table lookup with no solver of its own.

### 6. Test 5 evaluates phases with the exact radiation primitive, not `WKB_phase_function` — IMPLEMENTATION CHOICE

Prompt §4 item 5 says "build objects from exact initial data at a phase extremum, apply the
store algebra and `apply_phase_offset`". The 990 objects' phases $\theta(s_i,s)=k(1/s_i-1/s)$
are formed with `RadiationModel`'s factored closed form and reduced by `WKB_mod_2pi`, then the
store algebra and `apply_phase_offset` are applied as in production. Running
`WKB_phase_function` for each object would have built 990 residual tables of ~500 nodes (all
identically zero in radiation) for no information: the producer's accuracy is items 1 and 3, and
item 5 is about cycle bookkeeping. Stated in the test's docstring.

### 7. `Quadrature/supervisors/WKB.py` deleted rather than trimmed — IMPLEMENTATION CHOICE

Prompt: "delete or trim the two supervisors". Both classes served only the two ODE stages;
`Quadrature/supervisors/__init__.py` is empty; no other module imports them (grep). Deleted.

### 8. `stage_1_data`'s contents — IMPLEMENTATION CHOICE

Prompt §2 item 5: "an `IntegrationData` recording the primitive's cost (integrand evaluations
spent on partials and the residual build, wall time)". Mapped as: `compute_time` = wall time of
the whole call; `compute_steps` = number of samples; `RHS_evaluations` = residual-build
evaluations + leading-table partial evaluations; `mean_RHS_time`, `max_RHS_time`,
`min_RHS_time` = `None` (the per-RHS timer has no counterpart and inventing an average would
misdescribe the column). All nullable.

### 9. The `initial_data_only` payload's `theta_div_2pi_sample` is `[0]` (int), was `[0.0]` — IMPLEMENTATION CHOICE

`GkWKBValue.theta_div_2pi` is typed `int` and every other sample's `div` is an `int` from
`WKB_mod_2pi`; the old float was an inconsistency. Harmless either way (the factory column is
`Integer`).

### 10. One printed WKB-violation warning per object is kept — IMPLEMENTATION CHOICE

The supervisors printed a two-line `!! WARNING` once per object when the criterion first
exceeded unity; the new code does the same at the first offending sample. Prompt §2 item 3 asked
for the flags; the print is retained for parity with the previous behaviour and because nothing
else surfaces the flag to a reader of the log.

## Verification performed

All runs from the worktree root with `PYTHONPATH=. ./venv/bin/python`. The numbers below are
what the tests printed on this tree.

### 1. Exact-radiation control at production span (prompt §4 item 1; README §6 row 3)

`RadiationModel(H0=1)`, `tau`/`cs_tau`/`friction_F` tables built from the integrands on the
production source grid (100/decade of $z$) from the 3-e-fold point to $z=0.1$; response grid
`winnow(12)`; anchor the top node.

| $k$ | span [rad] | samples | max phase error [rad] | at $z$ | $\varepsilon\theta$ floor | threshold | ODE (review §2) |
|---|---|---|---|---|---|---|---|
| $10^7$ | $9.0909\times10^6$ | 56 | **$3.7253\times10^{-9}$** | 0.229289 | $2.02\times10^{-9}$ | $10^{-8}$ | $9.7\times10^{-3}$ |
| $10^9$ | $9.0909\times10^8$ | 73 | **$3.5763\times10^{-7}$** | 0.398558 | $2.02\times10^{-7}$ | $10^{-6}$ | 0.98 |

Both at ~2 ulp of the span. `rho_end == 0.0` exactly (the residual table is bit-zero in
radiation, as log 05 established). Every remainder in $(-2\pi,0]$.

### 2. Real background against prompt 01's references (item 2; README §6 rows 1–2)

`theta_ref = -k [tau_minus_top(z) - tau_minus_top(anchor)] - rho_G(z)` from the JSON, at the
checkpoints below each $k$'s 3-e-fold anchor (off-grid); the table's phase reconstructed as
`div*2pi + mod`.

| model | $k$ | checkpoints | anchor $z$ | span [rad] | **max phase error [rad]** | at $z$ | $\varepsilon k\tau$ floor | threshold | ODE (review §4) |
|---|---|---|---|---|---|---|---|---|---|
| LambdaCDM | $10^5$ | 9 | $2.30758\times10^9$ | $1.3728\times10^9$ | **$1.1921\times10^{-7}$** | 10.0123 | $3.05\times10^{-7}$ | $10^{-5}$ | 13.9 |
| LambdaCDM | $3\times10^8$ | 10 | $6.92274\times10^{12}$ | $4.1184\times10^{12}$ | **$9.7656\times10^{-4}$** | 1.00062 | $9.14\times10^{-4}$ | $5\times10^{-3}$ | 7366 |
| QCD | $3\times10^8$ | 11 | $1.0235\times10^{13}$ | $4.1184\times10^{12}$ | **$9.7656\times10^{-4}$** | 1.00062 | $9.14\times10^{-4}$ | $5\times10^{-3}$ | — |

At $k=3\times10^8$ the measured "error" is one ulp of $4\times10^{12}$ rad ($9.77\times10^{-4}=2^{-10}$):
both the reference (a difference of two JSON doubles of size $1.4\times10^4$ Mpc) and the
reconstruction `div*2pi + mod` sit at README §2 (d)'s absolute-phase floor. **This test cannot
see the table's own error at $k=3\times10^8$**; prompt 13 should score interval quantities
(`difference_error`) there. The QCD row is a quadrature check against the same LG integral,
not a physics check (the LG truncation floor there is $\sim10^{-3}$ rad). `rho_end`:
$-2.588\times10^{-7}$ (LambdaCDM $10^5$), $-1.353\times10^{-10}$ ($3\times10^8$),
$-1.180\times10^{-3}$ (QCD $3\times10^8$) — log 05's values.

### 3. Off-grid anchor (item 3)

$k=10^7$ radiation; $z_{\rm init}=490692.0041$, 37 % through $[486525,\,497870]$ in $u$;
`node_index(z_init) is None` asserted. **Max phase error $3.7253\times10^{-9}$ rad at $z=0.1$**
(same as item 1). `lead_partials == 56 == len(samples)`, `lead_evals == 224 == 4 × 56`,
`offgrid_init == True`; the residual table needed no partial. `residual_nodes` unit-tested on a
toy grid (anchor on top, on-grid anchor not duplicated, off-grid sample added).

### 4. The $(B,\delta)$ store algebra (item 4)

LambdaCDM, $k=10^5$, $z_{\rm init}$ the anchor; 200 seeded random $(G,G')$ pairs over 12 decades
of scale plus $(0,\pm1)$, $(-1,0)$, $(-1,0.5)$, $(-2.5,-0.5)$, $(1,0)$ — 206 cases.
$G(z_i)=B\sin\delta/\sqrt{\omega_i}$ reproduced to **$2.948\times10^{-16}$** and
$G'(z_i)=-(G_i/2)(d\ln\omega+\varepsilon/(1+z))+\sqrt{\omega_i}B\cos\delta$ to
**$2.809\times10^{-16}$** of the LG amplitude (threshold $10^{-12}$). $B>0$ in every case, and the
removed factor ${\rm sgn}(\sin\delta)\,{\rm sgn}(G_{\rm init})$ with the `>= 0` conventions was
$+1$ in every case including $G_{\rm init}=0$ and $G_{\rm init}<0$ (review §8.1 confirmed).

### 5. Cross-object cycle consistency (item 5; README §6 row 5)

15 $k$ in ${\rm geomspace}(10^6,10^8)$ × 3 response points $x_r\in\{10^2,10^3,10^4\}$ = 990
objects per geometry, source band $z_s\in[\sqrt{z_{e3}z_{e4}},z_{e3}]$ at 100/decade.

| geometry | objects | stored − exact is an integer number of cycles to | `apply_phase_offset` defect | one-cycle jumps | stop-point transitions | objects `shift_theta_sample` would have rebased |
|---|---|---|---|---|---|---|
| stop at maxima (production, README (h)) | 990 | $3.97\times10^{-12}$ rad | $4.4\times10^{-16}$ rad | **90** | **90** (equal, and at the same neighbours) | 270 |
| alternating extrema (review t6) | 990 | $5.12\times10^{-12}$ rad | $4.4\times10^{-16}$ rad | 90 | 180 | **180 = the review's 60 of 330** |

**0 rebase offsets** by construction: `apply_phase_offset` satisfies
$(d'-d)\,2\pi+m'=m+\delta$ to $4.4\times10^{-16}$ rad on every sample of every object. The
retired `shift_theta_sample` would have rebased 180 of these 990 objects in the review's
geometry, reproducing review §8.3's 60 of 330 exactly. The concrete case prompt 09 needs
(production geometry, $k=10^7$, $x_r=10^3$): 22 objects, 2 cycle jumps, between neighbouring
$z_s = 316700\to324331$ ($+1$) and $z_s=401839\to411522$ ($+1$).

### 6. Payload contract (item 6)

Keys exactly as §What shipped item 5; `stage_2_data` an `IntegrationData` with every field
`None`; `stage_1_data.compute_steps == 56`, `.RHS_evaluations == rho_evals + lead_evals`;
`initial_data_only` returns `[0]`, `[0.0]` and all-`None` stage data, plus `friction_sample=[0.0]`
when `friction=True`; `friction_sample[i] == friction_F.delta(z_init, z_i)` **bit-equal**, and
$7.1\times10^{-15}$ from the closed form $2\log((1+z)/(1+z_i))$, negative for $z<z_i$ (log 04's
sign); the `Tk` sector's phase against the exact radiation $-k\,\Delta\tau_s-\rho_T$:
**$1.863\times10^{-9}$ rad** at $z=0.30235$ over 56 samples, $\rho_T=-0.08634$ rad over the range
(review §12.2's value). Unknown sector → `ValueError`; a model without table accessors →
`RuntimeError`.

### 7. Cost (item 7; README §6 row 4)

LambdaCDM, $k=3\times10^8$, 116 response samples from the off-grid anchor $z=6.923\times10^{12}$
to $0.1$: **best of 3: 0.0311 s** (first run 0.0305 s; in-call `compute_time` 0.031 s), against
the 0.05 s target and the ODE's 63.7 s. **6,000 integrand evaluations** = 5,536 residual (order 4,
1,385 nodes: the background grid from the anchor down, plus the anchor) + 464 leading-table
partials (4 × 116, the anchor being off-grid), against the ODE's $2.5\times10^6$.

### Suites

- `PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_gk_wkb_phase`: 21 tests, OK, 1.3 s.
- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .`: **199 tests, OK, 120 s.**
- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_range_reduce LiouvilleGreen.tests.test_bessel_phase`: 10 tests, OK.
- Greps: `grep -n "solve_ivp\|Q_INDEX\|DEFAULT_PHASE_RUN_LENGTH" Quadrature/integrators/WKB_phase_function.py` empty;
  `grep -n "sgn_sin_deltaTheta\|shift_theta_sample" ComputeTargets/GkWKBIntegration.py` empty
  (both also asserted by `TestSourceHygiene`).
- `black --check` on the six touched Python files: clean. (Tree-wide `black --check .` flags 53
  `docs/` scripts and one `AdaptiveLevin` file that predate this commit and are outside its scope;
  none is touched here.)
- Import smoke: `import ComputeTargets` and the `GkWKBIntegration.PHASE_SOLVER_LABEL` attribute
  resolve.

**Not run (needs Ray and a datastore):** an end-to-end `GkWKBIntegration.compute()`/`store()`
cycle and the Datastore factory `store()` path. The factory path was verified by reading (§What
shipped item 6); prompt 13's scoped pipeline run exercises it live.

## Observations not acted on

1. **The residual table is rebuilt per object although it depends only on `(model, k, sector)`.**
   Every `GkWKBIntegration` object for the same $k$ rebuilds it from scratch: 5.5k integrand
   evaluations and ~20 ms on LambdaCDM, 5–7k and 82–255 ms on `QCD_Cosmology` (log 05), times
   ~1,700 source redshifts per $k$. Against the ODE's 13 CPU-hours per $k$ this is noise, but it
   is the dominant cost of the new producer (5,536 of 6,000 evaluations) and could be cached per
   `(model, k)` — e.g. built once alongside the background model or memoised in the Ray worker.
   Not this prompt's to add. → `[06-residual-table-per-object]`.
2. **The `metadata` column has 23–50 characters of headroom.** `GkWKBIntegration.metadata` is
   persisted as `json.dumps(...)` into a `sqla.String(DEFAULT_STRING_LENGTH)` = `String(256)`
   column. The new metadata is 206 characters (233 with `initial_data_only`). SQLite does not
   enforce the length; a PostgreSQL deployment would truncate or refuse. Anyone adding a key
   should count. → `[06-metadata-column-headroom]`.
3. **`docs/` reproduction scripts that import the removed ODE no longer run**:
   `docs/gk-wkb-review-fable-2026-09-09/{t2_solver,t4b_production_real,t7_jitter,t9_warn}.py`,
   `docs/gktk-remedial/baseline_k1e5.py`,
   `docs/gk-wkb-review-astra-pathfinder-2026-09-08/{measure,alternatives}.py` reference
   `integrate_phase_function`, `stage_1_evolution`, `stage_2_evolution`,
   `DEFAULT_OMEGA_WKB_SQ_MAX`. They measured the tree they ran on and the documents they support
   are correct for it (README §5 rule "verification documents are additive"); they were not
   edited. → `[06-docs-scripts-reference-removed-ode]`.
4. **`TkWKBIntegration` is in an interim state by design** (prompt §3.4): `compute()` uses the
   primitive, `store()` still applies the sign fix and `shift_theta_sample`, `friction_RHS` is
   dead. The interim path is verified in test 6 (θ_T to $1.9\times10^{-9}$ rad on radiation).
   Prompt 07 owns the rest.
5. **The $k=3\times10^8$ reference comparison is floored by representation, not by the table**
   (Verification 2). Prompt 13 should measure `difference_error` of interval phases at that $k$.
6. `[01-offgrid-accessor-cost-on-qcd]` can be narrowed: the producers evaluate the leading table
   with one off-grid endpoint (the anchor) once per sample and never both-off-grid; at most
   ~1,160 samples per Tk object on the source grid, i.e. $\le$ 30 ms on `QCD_Cosmology` at
   log 03's 26 µs. Recorded on the board; closes with prompt 09's confirmation as planned.
7. `GkWKBIntegration.compute()` prints a warning if the WKB criterion exceeds unity at
   `z_init`, and `WKB_phase_function` then raises `RuntimeError` for the same condition —
   pre-existing double handling, left as is.
8. The `_init_efolds_suph` typo (audit GK-8) named by prompt §3.2 is out of scope and was not
   looked for beyond confirming the attribute used is `_init_efolds_subh`.

## State handed to the next prompt

**`WKB_phase_function` signature and payload keys (verbatim):**

```python
from Quadrature.integrators.WKB_phase_function import (
    WKB_phase_function,           # @ray.remote; tests call WKB_phase_function._function
    PHASE_SOLVER_LABEL_BASE,      # "wkb-primitive"
    PHASE_SOLVER_STEPPING,        # RHO_GAUSS_ORDER = 4
    PHASE_SOLVER_LABEL,           # "wkb-primitive-stepping4"
    SECTOR_LEADING_PRIMITIVE,     # {"Gk": "tau", "Tk": "cs_tau"}
    residual_nodes,               # (grid, z_sample, z_init) -> descending np.ndarray
)

@ray.remote
def WKB_phase_function(model_proxy, k, z_init: float, z_sample, *,
                       sector: str, omega_sq, d_ln_omega_dz, friction: bool = False,
                       task_label: str = "WKB_phase_function",
                       object_label: str = "(object)") -> dict
# no atol/rtol. Payload keys, always:
#   "stage_1_data"   IntegrationData(compute_time=wall s, compute_steps=n_samples,
#                                    RHS_evaluations=rho_evals + lead_evals, others None)
#   "stage_2_data"   IntegrationData with every field None (never None itself)
#   "theta_div_2pi_sample"  List[int]      "theta_mod_2pi_sample"  List[float] in (-2pi, 0]
#   "phase_solver_label"    PHASE_SOLVER_LABEL
#   "has_WKB_violation" bool, "WKB_violation_z" Optional[float],
#   "WKB_violation_efolds_subh" Optional[float]
#   "metadata"  {"solver", "sector", "N_rho", "N_lead", "rho_nodes", "rho_evals",
#                "lead_partials", "lead_evals", "offgrid_init", "rho_end"[, "initial_data_only"]}
# with friction=True additionally:
#   "friction_sample"  [friction_F.delta(z_init, z) for z in samples]   (negative for z < z_init)
#   "friction_data"    IntegrationData with every field None
#   "friction_solver_label"  PHASE_SOLVER_LABEL
# theta(z; z_init) = -(k * leading.delta(z_init, z) + rho.delta(z_init, z)); the residual table
# is built by build_phase_residual on the background grid restricted to [min z_sample, z_init]
# plus the samples plus z_init as top node; the leading table's partial at an off-grid z_init is
# evaluated once per sample (order-4 panel, 4 integrand calls on LambdaCDM).
```

**`apply_phase_offset` signature (verbatim):**

```python
from LiouvilleGreen.WKBtools import apply_phase_offset
def apply_phase_offset(div_2pi_sample: Sequence[int], mod_2pi_sample: Sequence[float],
                       delta: float) -> Tuple[List[int], List[float]]
# per sample: shift, new_mod = wrap_theta(mod + delta); new_div = div + shift. No rebase.
# (new_div - div) * 2pi + new_mod == mod + delta to 4.4e-16 rad.
```

**Solver labels registered in `main.py`:** one new `IntegrationSolver`, label `"wkb-primitive"`,
`stepping=4`, dict key `GkWKBIntegration.PHASE_SOLVER_LABEL == "wkb-primitive-stepping4"`; it
serves `GkWKBIntegration.phase_solver_label` and both of `TkWKBIntegration`'s
`phase_solver_label`/`friction_solver_label`. Prompt 07 needs no new label. The class attributes
`GkWKBIntegration.PHASE_SOLVER_LABEL_BASE / PHASE_SOLVER_STEPPING / PHASE_SOLVER_LABEL` exist for
`main.py`; `TkWKBIntegration` has none (read them from `Quadrature.integrators.WKB_phase_function`
or `GkWKBIntegration`).

**D7:** `shift_theta_sample` and `WKB_product_mod_2pi` retained in `LiouvilleGreen/WKBtools.py`
with "not used by the production path" comments. `TkWKBIntegration.store()` still calls
`shift_theta_sample` (`:467`) — prompt 07 replaces it with `apply_phase_offset(data[...],
data[...], deltaTheta)` exactly as `GkWKBIntegration.store()` now does, deletes its sign fix
(`:457-463`), `friction_RHS`, the module-level `FRICTION_INDEX = 0` placeholder at
`TkWKBIntegration.py:21-25`, and the `RHS_timer`/`NumericIntegrationSupervisor` imports that
`friction_RHS` alone needs. After that `shift_theta_sample` may be deleted outright (its
remaining consumers are `docs/` scripts) or kept; either is a recorded choice.

**`TkWKBIntegration.store()` and the payload:** it reads `data["friction_data"]` (`:398`),
`data["friction_sample"]`, `data["friction_solver_label"]` — all supplied. `friction_sample[i]`
is `friction_F.delta(z_init, z_i)`, **negative** for $z<z_{\rm init}$, and
`T_WKB = norm * exp(friction_sample[i]) * ...` needs no sign change (log 04, confirmed to
$7\times10^{-15}$ against the radiation closed form). The `Tk` sector phase is already
$-k\,\Delta\tau_s-\Delta\rho_T$ from the tables: $1.9\times10^{-9}$ rad against the exact
radiation primitive at $k=10^7$ over $x\in[e^3, 10^7]$, $\rho_T=-0.0863$ carried.

**Measured maxima (item 4.2) and cost (item 4.7):** LambdaCDM $k=10^5$: $1.19\times10^{-7}$ rad
(at $z=10.01$); $k=3\times10^8$: $9.77\times10^{-4}$ rad (at $z=1.0006$; one ulp of $4\times10^{12}$
rad, the representation floor — not the table's error); QCD $k=3\times10^8$: $9.77\times10^{-4}$
rad (same floor). Radiation $10^7$/$10^9$ rad spans: $3.7\times10^{-9}$ / $3.6\times10^{-7}$ rad.
Cost at $k=3\times10^8$ on LambdaCDM, 116 response samples: 0.031 s, 6,000 integrand evaluations
(5,536 residual + 464 leading partials); the residual build is 92 % of it and is per object
(`[06-residual-table-per-object]`).

**δ-wrap count for prompt 09 (item 4.5):** in the production geometry (stop at a maximum of $G$,
$\delta=+\pi/2$ for every object) the stored cycle count of $\theta(z_r;z_s)$ steps by exactly
$+1$ wherever the stop point moves to the next maximum and nowhere else: 90 steps among 990
objects (15 $k$ × 3 $x_r$), 0 rebase offsets. Concrete case, $k=10^7$, $x_r=10^3$: 22 objects
in $z_s\in[\sqrt{z_{e3}z_{e4}},z_{e3}]$, 2 steps, between neighbouring $z_s=316700\to324331$ and
$z_s=401839\to411522$ (both $+1$ cycle in the stored `div`). This is the mechanism
`RECONCILIATION.md` §2 item 6 describes and the `GkSource` rectifier repairs (D5); prompt 09
builds $\varphi$ from the *rectified* `theta_div_2pi`. "δ sign wraps" is not the right
description of these points (Deviations 2); "stop-point transitions" is. In the review's
alternating-extremum geometry the same 990 objects give 90 steps at every second of 180
transitions, and `shift_theta_sample` would have rebased 180 of them — the review's 60 of 330.

**Test fixtures prompt 07/09 can reuse** (`ComputeTargets/tests/test_gk_wkb_phase.py`):
`radiation_model_with_tables(z_nodes, H0)`, `lambdacdm_model_with_tables(z_nodes)`,
`qcd_model_with_tables(grid)` (each returns the `wkb_reference` stand-in with `functions.tau`,
`.cs_tau`, `.friction_F` as `TablePrimitive`s), `_Proxy(model, units)`, `_KExit(k, units)`,
`store_algebra(omega_sq_init, d_ln_omega_init, eps_init, z_init, G_init, Gprime_init) ->
(B, deltaTheta, raw_cos, raw_sin)`, `_sweep(k, x_r, stop)`.
