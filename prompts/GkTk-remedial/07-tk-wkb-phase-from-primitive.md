# Prompt 07 — The transfer-function WKB phase and friction from the tables

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §12.1 (what is shared), §12.2 (spans, $\rho_T$, $F$), §12.3 (production
errors, the late-time $Q$ jump), §12.4 (the LG truncation floor for $T$), §12.7
**Design facts:** README §2 (a), (d), (e), (f).
**Depends on:** 06 (the shared function already takes `sector="Tk", friction=True`), 04, 05.
**Recommended model:** Opus
**Files you may touch:** `ComputeTargets/TkWKBIntegration.py`, new
`ComputeTargets/tests/test_tk_wkb_phase.py`, `ComputeTargets/tests/test_background_cs_tau_friction.py`
(**only** to relocate `friction_RHS` — see the decision below), `main.py` solver registration hunk
**only if** a distinct friction label is needed, plus the log and the status board.

> **Decision (user, 2026-09-11), after the executing agent stopped on it.** §2 item 1 below asks
> for `friction_RHS` to be deleted and §4 asks for the `friction_RHS` grep to be empty, but
> prompt 04's `ComputeTargets/tests/test_background_cs_tau_friction.py` imports it **at module
> scope** (`:56`) and calls it as the DOP853 right-hand side in
> `TestFrictionODEComparison.test_friction_ode_is_the_inaccurate_one` (`:784`), the test that
> demonstrates the ODE the `friction_F` table replaces is the inaccurate one. Deleting the
> function is an import-time failure that takes all 17 tests of that module with it. That file is
> prompt 04's, and prompt 07's text predates it.
>
> **Resolution:** move `friction_RHS` **verbatim** into that test module as a module-private
> `_friction_RHS`, so prompt 04's measurement keeps measuring exactly the same ODE, the §4 grep
> empties and the suite stays green. Touch nothing else in that file beyond the relocation, the
> import, and one sentence of docstring saying the ODE now lives only in the test that retires it.
> Record it as a `STRUCTURALLY REQUIRED` deviation naming this decision.

**Do not touch:** `TkSourceFunctions.py` (prompt 10), `TkNumericIntegration.py` (prompts 11, 12),
the Datastore factories (verify nullable columns; no schema change).

Read first: README §2 (a), (d), (f); review §12 in full; `logs/04-…`, `logs/05-…`, `logs/06-…`
"State handed to the next prompt" (in particular the payload keys and the `friction_F.delta`
convention).

---

## 1. Character of this commit

The transfer-function twin of prompt 06, smaller because 06 already rewrote the shared function
and switched `compute()`. What remains is `store()`, the friction ODE, and the tests that establish
the transfer function's own floors.

Differences from $G_k$ that the tests must reflect (review §12.7): one object per $k$ (no
cross-object stitching, no rectifier); the residual $\rho_T\approx-0.09$ rad is **carried** by
prompt 05's table and must appear in the stored phase; the LG representation is **not** exact in
radiation, so the value-level control is against the exact $T=3(\sin x-x\cos x)/x^3$ with the
review's §12.4 floors, not against zero.

## 2. What to build

`ComputeTargets/TkWKBIntegration.py`:

1. Delete `friction_RHS` (`:25-49`) and its imports (`RHS_timer`, `NumericIntegrationSupervisor`
   if now unused). Update the `TkSourceFunctions` module docstring **only in prompt 10** — leave
   that file alone here even though it names `friction_RHS`.
2. `store()`: delete the sign fix (`:452-459`); `self._sin_coeff = B`, `self._cos_coeff = 0.0`;
   replace `shift_theta_sample` (`:463-467`) with `apply_phase_offset`; `friction_sample` from the
   payload is already $F(z)-F(z_{\rm init})$ from the table (prompt 06 §2 item 5) — keep the
   `exp(friction_sample[i])` amplitude factor exactly as at `:501`. `self._friction_data` = the
   all-`None` `IntegrationData` from the payload; `self._friction_solver` = the label from the
   payload. Remove the placeholder note prompt 06 left.
3. `__init__` / properties: nothing changes in the persisted fields. Update the class and method
   comments that describe the ODE (`:421-428` is about the $G_k$ motivation — keep; fix any that
   say "integrate").
4. Datastore: `TkWKBIntegration` table `friction_*` columns (`:158-163`) are nullable; verify the
   factory's `store()` handles `None` fields exactly as prompt 06 verified for `stage_2_*`.

## 3. Tests (`test_tk_wkb_phase.py`)

1. **Exact-radiation phase control.** `RadiationModel`, $k=10^7$, from $x_i=24$ (i.e. $z_i$ such
   that $kc_s\tau(z_i)=24$) down to $x=10^4$: stored $\theta_T$ vs $x_i-x+(1/x_i-1/x)$… — derive
   the exact LG phase from $\omega_T^2=k^2/(3s^4)-2/s^2$ (review §12.4) as
   $\theta_T=\int\omega_T\,dz$ in closed form or by mpmath, and assert $\le10^{-9}$ rad. Separately
   assert the residual part alone: $\theta_T+k\,\Delta\tau_s$ equals $1/x_i-1/x$ to $10^{-12}$.
2. **Exact-radiation value control (the floor, documented).** Reproduce review §12.4's table:
   full `store()` reconstruction $T_{\rm WKB}$ against $T=3(\sin x-x\cos x)/x^3$, envelope-relative,
   from $x_i\in\{24, 50, 100, 400\}$ over $x_i\le x\le10^4$: assert $\le5\times10^{-5}$,
   $\le6\times10^{-6}$, $\le8\times10^{-7}$, $\le2\times10^{-8}$ respectively, and that the
   error **scales as $x_i^{-3}$** (ratio of the $x_i=24$ and $400$ maxima between $3\times10^3$
   and $6\times10^3$). This is the LG truncation floor of README §2 (d); the test's docstring must
   say it is a property of the representation, not of this code.
3. **Real background.** `LambdaCDMModel`, $k=10^5$ and $3\times10^8$, from $z_{e3}$ to $0.1$ on
   the 100/decade source grid, against prompt 01's $\tau_s$ and $\rho_T$ references: phase error
   $\le10^{-4}$ rad and $\le5\times10^{-3}$ rad at every checkpoint (review §12.3: 2.01 rad and
   $5.1\times10^3$ rad). $F(z)-F(z_i)$ vs reference $\le10^{-12}$ relative (review: $2.3$–$4.1\times10^{-7}$).
   `QCDModel`, $k=3\times10^8$: report; assert $\le5\times10^{-3}$.
4. **Store algebra**: `sin_coeff == B > 0` for random $(T_{\rm init}, T'_{\rm init})$ including
   negative $T_{\rm init}$; `cos_coeff == 0.0`.
5. **Payload/attribute contract**: `friction_data`, `stage_2_data` are `IntegrationData` with
   `None` fields; `phase_solver`/`friction_solver` resolve through a stand-in `solver_labels` dict.
6. **Cost**: per object at $k=3\times10^8$ on `LambdaCDMModel` $\le0.05$ s (review: 58 s).

## 4. Verification and acceptance

- New tests pass; `discover -s ComputeTargets/tests -t .` passes — including
  `test_tk_source_functions.py` **unchanged** (it uses stand-in WKB values, not this class).
- README §6 rows for $\theta_T$ and the $T_{\rm WKB}$ radiation control met.
- `grep -n "friction_RHS\|sgn_sin_deltaTheta\|shift_theta_sample\|FRICTION_INDEX" ComputeTargets/TkWKBIntegration.py`
  empty.
- `black --check` clean.

## 5. Log and commit

"State handed to the next prompt", verbatim: what `TkWKBValue.friction` now contains and its
sign convention; the measured item 3 maxima; the item 2 table as measured (prompt 10 and prompt 13
cite it); confirmation of the Datastore `None`-field handling.

Commit subject, or something equally specific: `Compute the transfer-function WKB phase and friction from tables`.
