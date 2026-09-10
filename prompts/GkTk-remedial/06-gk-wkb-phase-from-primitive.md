# Prompt 06 — The Green's-function WKB phase from the primitive

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §2–§4 (what the ODE gets wrong and what it costs), §7 "What this replaces",
§8.1 (sign fix, rebase), §8.2 (the four small defects in `WKB_phase_function.py`), §8.3, §13.4
(how the (div, mod) pair is produced)
**Design facts:** README §2 (a), (c), (e), (f); decisions §7 D6, D7.
**Depends on:** 03, 04, 05.
**Recommended model:** **Fable** (Opus if unavailable). **This is the production point of no return
for $G_k$.**
**Files you may touch:** `Quadrature/integrators/WKB_phase_function.py` (rewrite),
`ComputeTargets/GkWKBIntegration.py`, `LiouvilleGreen/WKBtools.py`,
`ComputeTargets/TkWKBIntegration.py` **only the `compute()` call site (§3.4)**, `main.py` **solver
registration hunk only**, `Quadrature/supervisors/WKB.py` (delete or trim the two supervisors), new
`ComputeTargets/tests/test_gk_wkb_phase.py`, plus the log and the status board.
**Do not touch:** `GkSource.py` (the rectifier stays, D5), `GkSourcePolicyData.py`,
`phase_spline.py`, the numeric integrators, the Datastore factories (no schema change is needed —
verify, §3.5).

Read first: README §2 (a), (c), (e), (f), §4.3 stop conditions, §7 D5–D7;
`RECONCILIATION.md` §1 items 1–4, 12, 13 and §2 items 4, 5, 6, 10; review §3, §4, §7, §8.1, §8.2,
§8.3, §13.4; `logs/03-…`, `logs/04-…`, `logs/05-…` "State handed to the next prompt".

---

## 1. What is removed, and the measured reason for each

| Removed | Measured reason (review) |
|---|---|
| Stage 1 ($\theta'=\omega$ with $10^4$-rad resets, `stage_1_evolution`) | 2.5e6 RHS evaluations, 37,301 resets, 63.7 s per object at $k=3\times10^8$ (§4); accurate but cost ∝ span |
| Stage 2 (the $Q$ variable, `stage_2_evolution`) | error $=\omega_i(1+u)\,\delta Q$ with $Q\to-224$…$-11069$; 13.9 rad ($k=10^5$) and 7366 rad ($3\times10^8$) at $z=0.1$ (§3, §4); dense-output error 0.33 rad on a linear phase (§3) |
| `WKB_product_mod_2pi` | only consumer was stage 2 |
| `shift_theta_sample`'s cross-sample rebase | manufactures the ±1-cycle inconsistencies between objects (§8.1, §8.3) |
| The `sin_coeff` sign fix | provably always $+1$ (§8.1; 0 of 4,000 objects) |
| The zero-length check `fabs(z_init − z_min) < atol` (`:671`) | compares a redshift to an ODE tolerance (§8.2); replaced by an exact test |
| `values[0]` 1-element array into `fmod` (`:189`) | NumPy deprecation (§8.2); the code path is gone |
| The comments at `:262-264`, `:403` | false (§8.2) |
| `ThetaSupervisor`, `QSupervisor` | no ODE to supervise |

Retained, unchanged: `Gk_omegaEff_sq`/`Gk_d_ln_omegaEff_dz` (used for the WKB-criterion
diagnostic and, through prompt 05, the residual), the amplitude $\sqrt{H_i/(H\omega)}$, the
$(B,\delta)$ algebra, the `GkWKBValue` schema, the payload key names.

## 2. The new `WKB_phase_function`

Keep the `@ray.remote` name and the return-payload keys so that the two `store()` methods change
minimally. New signature:

```python
@ray.remote
def WKB_phase_function(model_proxy, k, z_init: float, z_sample, *,
                       sector: str,                 # "Gk" or "Tk" -- selects tau/cs_tau and the residual
                       omega_sq, d_ln_omega_dz,     # for the WKB-criterion diagnostic only
                       friction: bool = False,      # Tk: also return F(z) - F(z_init) from the table
                       task_label: str = ..., object_label: str = ...) -> dict
```

`atol`/`rtol` are **no longer parameters**. Implementation:

1. `model = model_proxy.get()`; `leading = model.functions.tau` for `"Gk"`,
   `model.functions.cs_tau` for `"Tk"`; `rho = build_phase_residual(model, k_float, nodes, sector)`
   where `nodes` is the sample list plus `z_init` if off-grid — simplest is to build the residual
   table on the *background model's* grid restricted to $[z_{\min}, z_{\rm init}]$ and let `delta`
   supply the partial at `z_init`; state what you did.
2. For each sample $z$: $\theta=-[k\cdot$`leading.delta(z_init, z)`$+$`rho.delta(z_init, z)`$]$
   (README §2 (c) sign convention; negative for $z<z_{\rm init}$). Then
   `theta_div_2pi, theta_mod_2pi = WKB_mod_2pi(theta)` — the negative-remainder convention
   (README §2 (e)); `WKB_mod_2pi` stays in `WKBtools`.
3. WKB-criterion diagnostic: evaluate `|d_ln_omega_dz|/sqrt(omega_sq)` at `z_init` and at every
   sample; `has_WKB_violation`, `WKB_violation_z`, `WKB_violation_efolds_subh` from the first
   sample exceeding 1 (the same semantics the supervisors had, now on the sample grid — say so).
   Keep the existing initial-time `RuntimeError`s for `omega_sq_init < 0` and criterion $>1$.
4. The **exact** zero-length case: `if len(z_sample) == 1 and z_sample.min.z == z_init` (or the
   `store_id` test the numeric integrator uses) → the initial-data-only payload as today.
5. Payload: `"stage_1_data"` = an `IntegrationData` recording the primitive's cost (integrand
   evaluations spent on partials and the residual build, wall time); `"stage_2_data"` = an
   `IntegrationData` with all-`None` fields (**not** `None`; `RECONCILIATION.md` §1 item 12 — confirm
   the Datastore factory's store path accepts it, and if it does not, that is a `STRUCTURALLY
   REQUIRED` note, not a schema change); `"theta_div_2pi_sample"`, `"theta_mod_2pi_sample"`;
   `"phase_solver_label"` = a new registered label (§3.3); `"has_WKB_violation"`, `"WKB_violation_z"`,
   `"WKB_violation_efolds_subh"`; `"metadata"` with the orders used, the residual at the last
   sample, the number of off-grid partials, and `"initial_data_only"` when applicable; when
   `friction=True`, `"friction_sample"` = `[model.functions.friction_F.delta(z_init, z) for z in samples]`
   and `"friction_solver_label"` = the same new label.
6. Delete `stage_1_evolution`, `stage_2_evolution`, `integrate_phase_function`,
   `integrate_friction_function`, the two constants, `THETA_INDEX`/`Q_INDEX`/`FRICTION_INDEX`
   (check `TkWKBIntegration.py:17` imports `FRICTION_INDEX` — remove that import in §3.4), and the
   `solve_ivp` import. `grep -n "solve_ivp\|Q_INDEX\|DEFAULT_PHASE_RUN_LENGTH" Quadrature/integrators/WKB_phase_function.py`
   must be empty.

## 3. The callers

### 3.1 `LiouvilleGreen/WKBtools.py`

Add

```python
def apply_phase_offset(div_2pi_sample, mod_2pi_sample, delta: float) -> Tuple[List[int], List[float]]:
    """Add delta to every sample's phase: wrap_theta(mod + delta) per sample, the returned cycle
    shift added to that sample's div. No cross-sample rebase (see README §2 (e))."""
```

built on the existing `wrap_theta`. `WKB_mod_2pi` and `wrap_theta` stay (the latter has three test
consumers, `RECONCILIATION.md` §1 item 7). `shift_theta_sample` and `WKB_product_mod_2pi`: delete,
or retain with a docstring stating they are unused by production and kept for the `docs/`
reproduction scripts — **D7, an `IMPLEMENTATION CHOICE` you must record either way**.

### 3.2 `ComputeTargets/GkWKBIntegration.py`

- `compute()`: call `WKB_phase_function.remote(self._model_proxy, self._k_exit, initial_z,
  self._z_sample, sector="Gk", omega_sq=Gk_omegaEff_sq, d_ln_omega_dz=Gk_d_ln_omegaEff_dz, …)`
  — no `atol`/`rtol`. The `self._atol`/`self._rtol` attributes stay (they are lookup keys;
  `RECONCILIATION.md` §2 item 10); add a one-line comment saying so.
- `store()`: delete `:403-410` (the sign fix); `self._sin_coeff = B`, `self._cos_coeff = 0.0`;
  replace the `shift_theta_sample` call (`:414-418`) with `apply_phase_offset(..., deltaTheta)`.
  Everything else in the value loop stays (it already uses `theta_mod_2pi_sample[i]` for `G_WKB`).
  Update the long comment block (`:376-383`) only where it describes the removed sign fix.
- `__init__`'s `_init_efolds_suph` typo (audit GK-8) — **out of scope**, leave it.

### 3.3 `main.py` — solver registration only

Register the new phase-solver label (e.g. `"wkb-primitive"` with `stepping=0`, or reuse
prompt 03's `"cumulative-GL"` label with the residual order as `stepping` — choose, and state it)
at `:2809-2821`, in the pattern of prompt 03. `store()`'s
`self._solver_labels[data["phase_solver_label"]]` must resolve.

### 3.4 `ComputeTargets/TkWKBIntegration.py` — two lines

Only so that the transfer-function pipeline keeps producing *correct* phases between this commit
and prompt 07: in `compute()` (`:364-376`), pass `sector="Tk", friction=True` and drop
`atol`/`rtol`; drop the `FRICTION_INDEX` import (`:15-18`). `store()` is untouched here — it
still runs its sign fix and `shift_theta_sample` on the payload; prompt 07 fixes it. Confirm the
payload keys it reads (`friction_sample`, `friction_data` — check: it reads `data["friction_data"]`
at `:398`; supply that key too, an `IntegrationData` with all-`None` fields, and remove the
placeholder in 07).

### 3.5 Datastore

No schema change: the `GkWKBIntegration` table's `stage_2_*` columns are nullable (`:152-157`) and
the value table is unchanged. **Verify** by reading `Datastore/SQL/ObjectFactories/GkWKBIntegration.py`'s
`store()` for how it serialises `obj.stage_2_data` when its fields are `None`. Same check for the
Tk factory's `friction_*` columns, which prompt 07 will null.

## 4. Tests (`test_gk_wkb_phase.py`)

Call the **undecorated** function with a `ModelProxy` stand-in exposing `.get()` (and whatever
`check_units` needs — read it) and a `wavenumber_exit_time` stand-in exposing `.k.k`,
`.k.k_inv_Mpc`, `.k.store_id` (prompt 01's `key()` helper pattern). Also exercise
`GkWKBIntegration.store()`'s algebra directly where possible, or replicate it.

1. **Exact-radiation control at production span.** `RadiationModel` with the closed-form `tau`
   accessor (build a `CumulativeTable` from $f=1/H$ on the production grid — this is also the
   test that the table on an analytic background is exact), $k=10^7$, from the 3-e-fold point to
   $0.1$ on the 8.3/decade response grid: max phase error $\le10^{-8}$ rad against
   $\theta=k(1/s_i-1/s)$ (review §2: the ODE gives $9.7\times10^{-3}$; the floor is
   $\varepsilon\theta\approx2\times10^{-9}$). Also at $k=10^9$ (span $10^9$ rad; ODE 0.98 rad):
   $\le10^{-6}$ rad.
2. **Real background at production spans**, against prompt 01's references: `LambdaCDMModel`,
   $k=10^5$, from $z_{e3}$ to $0.1$: $\le10^{-5}$ rad at every checkpoint (README §6; review
   §4 row: 13.9 rad); $k=3\times10^8$: $\le5\times10^{-3}$ rad (7366 rad). Quote the measured
   maxima. `QCDModel`, $k=3\times10^8$: report against the reference with the LG floor
   ($\sim10^{-3}$ rad) noted — assert $\le5\times10^{-3}$.
3. **Off-grid anchor.** Start from a $z_{\rm init}$ 37 % of the way through a grid interval (as a
   numeric hand-over does): the same accuracy as item 1.
4. **The $(B,\delta)$ store algebra.** For 200 random $(G_{\rm init}, G'_{\rm init})$ pairs,
   `sin_coeff * sin(theta_mod + ...)` reproduces the LG solution matched to the initial data to
   $10^{-12}$ relative, and `sin_coeff == B > 0` always (the removed sign fix was a no-op —
   assert `sin_coeff > 0` for every case including `G_init = 0` and `G_init < 0`).
5. **Cross-object consistency (the review's t6 sweep, without the rebase).** For the
   numeric-initialised band $z_s\in[\sqrt{z_{e3}z_{e4}}, z_{e3}]$ at 15 $k$ values, build objects
   from exact initial data at a phase extremum, apply the store algebra and `apply_phase_offset`,
   and assert: the reconstructed unwrapped $\theta(z_r; z_s)$ across neighbouring objects differs
   from the exact phase by $\le10^{-9}$ rad **before any rectification** except at the
   $\delta$-wrap points, where it differs by exactly one cycle — count them and assert the count
   equals the number of $\delta$ sign wraps. This documents `RECONCILIATION.md` §2 item 6 for
   prompt 09.
6. **Payload contract.** Keys as §2 item 5; `stage_2_data` is an `IntegrationData` with `None`
   fields; `initial_data_only` case returns one sample; `friction=True` returns `friction_sample`
   equal to `friction_F.delta(z_init, z)`.
7. **Cost.** Wall time per object on `LambdaCDMModel` at $k=3\times10^8$ over the full production
   response grid $\le0.05$ s (review: 63.7 s); integrand evaluations recorded.

## 5. Verification and acceptance

- New tests pass; `discover -s ComputeTargets/tests -t .` passes; `LiouvilleGreen/tests` per-module
  runs `test_range_reduce`, `test_bessel_phase` pass (`WKBtools` changed).
- README §6 rows for $\theta_G$ (three rows), cost, cross-object consistency met.
- The greps in §2 item 6 are empty; `grep -n "sgn_sin_deltaTheta\|shift_theta_sample" ComputeTargets/GkWKBIntegration.py`
  is empty.
- `black --check` clean.

## 6. Log and commit

"State handed to the next prompt", verbatim: the new `WKB_phase_function` signature and payload
keys; `apply_phase_offset`'s signature; the solver label(s) registered; D7's choice; the measured
maxima of item 4.2 and the cost of item 4.7; the $\delta$-wrap count from item 4.5 and the $z_s$
where wraps occur for one $k$ (prompt 09 needs a concrete case).

Commit subject, or something equally specific: `Compute the Green function WKB phase from the conformal-time table`.
