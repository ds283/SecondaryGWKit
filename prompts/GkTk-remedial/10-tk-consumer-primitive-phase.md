# Prompt 10 — The transfer-function consumer on the tables

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §12.6 (the consumer), §12.7 ("with a tabulated $F$ that spline disappears")
**Design facts:** README §2 (g); §4.2 item 2 (the file overlap with `transfer-remedial`).
**Depends on:** 09 (`PrimitivePhase`), 07 (stored $\theta_T$, `friction` convention), 04 (`friction_F`).
**Recommended model:** Opus
**Files you may touch:** `ComputeTargets/TkSourceFunctions.py`,
`ComputeTargets/tests/test_tk_source_functions.py` (**stand-in fixtures and new assertions only**),
`ComputeTargets/tests/test_phase_groups.py` (**the `FakeModel`/stand-in construction only**),
`ComputeTargets/tests/wkb_reference.py` (the `ClosedFormPrimitive` helper §2 asks for),
`ComputeTargets/tests/test_quadsource_integral.py` (**stand-in model construction only** — added
2026-09-11, see below), plus the log and the status board.

> **File list extended 2026-09-11, after the prompt ran (user decision).**
> `test_quadsource_integral.py` was not originally listed, and §4 nevertheless requires it to pass.
> It builds its own `TkSourceFunctions` from inputs captured out of
> `test_tk_source_functions.Fixture.exact_functions()` but pairs them with a closed-form
> `FakeModel`, so once §1's friction cross-check exists that module **fails**: its stored
> `friction` samples are backed out of the exact Bessel envelope and disagree with the constant-$w$
> Liouville-Green integral by 5.6231e-06 in $F$. The orchestrator confirmed this by restoring the
> file and re-running — four `RuntimeError`s from `_check_friction_samples`. The inconsistency was
> real all along and merely invisible while the consumer splined the stored samples, so the
> cross-check is doing its job. Prompt 10's five-line fix substitutes
> `exact_envelope_model()` per wavenumber in the non-`exact` `Tk_builder`, exactly as that
> fixture's `exact` branch one line above already does. `wkb_reference.py` is listed for the same
> reason: §2 names the file and the helper to put in it, so it was always in scope in substance.
> Still **stand-in construction only** in both: no tolerance constant, no production module.
**Do not touch:** tolerance constants or comments in those two test files that the
`transfer-remedial` campaign's prompt 08 owns (README §0.2, §4.2); `phase_groups.py`;
`QuadSourceIntegral.py`.

**Precondition (orchestrator):** `[00-transfer-remedial-test-file-overlap]` resolved — the user
has said whether `transfer-remedial` prompt 08 lands before or after this prompt on the branch in
use (README §4.2 item 2).

Read first: README §2 (g), §4.2; review §12.6, §12.7; `logs/04-…`, `logs/07-…`, `logs/09-…`
"State handed to the next prompt"; the module docstring of `TkSourceFunctions.py` in full (it is
a specification of the consumer protocol `phase_groups` relies on, and it must be rewritten
truthfully).

---

## 1. What changes

`TkSourceFunctions._build_WKB` (`:239-283`) splines the stored friction and builds a
`phase_spline(chunk_logstep=125, increasing=True)` of the stored phase — review §5 applies verbatim
(§12.6: $3.5\times10^{-3}$ rad at $k=10^5$ rising to ~10 rad at $3\times10^8$ from $h^4x_T/384$).
`omega()` is already closed-form (the right call, review §12.6). After this prompt:

- `phase` is a `PrimitivePhase` with `leading = model.functions.cs_tau`, `z_anchor = crossover_z`
  (the single hand-over redshift), `sign = +1` (θ decreases towards lower $z$; check against the
  stored samples' sign), and $\varphi(z)=\theta_{\rm stored}(z)+k\,$`cs_tau.delta(z_init, z)`
  … (fix the sign so that $\varphi\approx\Delta\rho_T+\delta$, i.e. $O(0.1)$ rad) splined in
  $\log(1+z)$. One object per $k$: no rectifier, `theta_div_2pi` used as stored.
- `friction(z)` returns `model.functions.friction_F.delta(crossover_z, z)` — exact, no spline —
  and `M(z)` uses it. The stored `friction` samples are read only to **cross-check** the table
  (assert agreement to $10^{-12}$ relative at construction; raise a clear error otherwise — this
  detects a datastore produced by the ODE friction against a table-built model).
- `_check_WKB` range discipline unchanged; `WKB_region` unchanged.
- Module docstring rewritten: the amplitude paragraph (F from the table; `friction_RHS` no longer
  exists — cite spec 01 R23 for the integrand), the phase paragraph (`PrimitivePhase`, not
  `phase_spline`; `increasing` no longer a concept), the duck-typed protocol (`Tk_WKB.values`
  still need `.theta_div_2pi`, `.theta_mod_2pi`, `.friction`; `model.functions` now needs
  `cs_tau` and `friction_F` accessors with `.delta`). `PHASE_SPLINE_CHUNK_LOGSTEP` deleted.

## 2. Test fixtures

`test_tk_source_functions.FakeModel` and the corresponding stand-in in `test_phase_groups.py`
build `ModelFunctions(...)` with closed-form constant-$w$ functions. Give them `cs_tau` and
`friction_F` accessor objects with `__call__` and `delta(a, b)` from the closed forms
($\tau_s=\sqrt w\,\tau$; $F=\tfrac32(1+w)\ln((1+z)/(1+z_i))$) — a small `ClosedFormPrimitive(f)`
helper in `wkb_reference.py` (prompt 01's module) whose `delta` is `f(b) - f(a)` is acceptable
**at the fixture's $x\le10^4$ scale**, and its docstring must say why that is acceptable there
and not in production (README §2 (c)). Do this with minimal edits and touch no tolerance constant.

## 3. New assertions in `test_tk_source_functions.py`

1. **The $h^4x_T/384$ term is gone.** Fixture "exact" ($w=1/3$) extended to $x_T$ up to $10^6$
   using the analytic $T=3(\sin x-x\cos x)/x^3$ directly (not `bessel_phase`, whose oracle floor
   on `main` is $x\times10^{-8}$ — `docs/lg-phase-and-handover-followup-2026-09.md` §2.4): the
   phase from `functions.phase.raw_theta` against the exact LG phase $\le10^{-7}$ rad at
   $x_T=10^6$ on the production 100/decade grid, where a `phase_spline` of the same samples has
   $\sim h^4x_T/384\approx0.8$ rad — build both in the test and assert the ratio $>10^5$.
2. `friction(z)` equals the closed form to $10^{-13}$ absolute; `M(z)` matches the exact envelope
   to the LG floor as before (do not tighten the existing LG-floor tolerances — they are
   `transfer-remedial` 08's).
3. `omega(z) == phase.theta_deriv(z)` to $10^{-10}$ relative on the "LG" fixture (the identity the
   module docstring promises; previously "up to spline error").

   > **Measured 2026-09-11 at 1.0492e-10 relative ($w=1/3$) — 4.9 % above this bound, and the
   > bound is deliberately NOT amended.** Unlike prompt 09's test 1 this threshold is reachable:
   > the miss is the not-a-knot end condition of the cubic $\varphi$ spline at the top of the WKB
   > region, 5.5589e-12 from the fifth sample inwards and 1.3e-15 in the interior, against
   > 4.249e-08 before this prompt — a 405× improvement that lands just outside. A quintic
   > $\varphi$ spline measures 9.695e-12 over the *whole* region and would meet $10^{-10}$
   > outright, but needs six samples against `MIN_SPLINE_DATA_POINTS = 5` and would make the $T_k$
   > consumer's spline order differ from prompt 09's $G_k$ one. **User decision, 2026-09-11:** take
   > the shipped two-window assertion ($<10^{-9}$ on `z_WKB[3:-3]`, $<10^{-11}$ on `z_WKB[5:-3]`,
   > both printed) for now and **leave `[10-residual-spline-end-condition]` open for prompt 13**,
   > so the end effect is measured on the real background — where the samples are production grid
   > nodes and $\varphi$ is not the closed-form residual of a constant-$w$ stand-in — before anyone
   > pays for the quintic. Prompt 13 should re-measure this identity and then close or escalate.
4. The construction-time friction cross-check raises on a deliberately inconsistent `friction`
   sample.

## 4. Verification and acceptance

- `test_tk_source_functions.py`, `test_phase_groups.py`, `test_quadsource_integral.py` pass;
  `discover -s ComputeTargets/tests -t .` passes.
- README §6 consumer row met for $T_k$ (test 3.1's ratio).
- `git diff HEAD~1 -- ComputeTargets/tests/test_phase_groups.py` touches only stand-in
  construction (no tolerance constant, no comment about tolerances).
- `grep -n "phase_spline\|PHASE_SPLINE_CHUNK_LOGSTEP\|friction_RHS" ComputeTargets/TkSourceFunctions.py` empty.
- `black --check` clean.

## 5. Log and commit

"State handed to the next prompt": the rewritten protocol paragraph verbatim (prompt 13 cites
it); the measured test 3.1 numbers; which fixture lines changed in the two shared test files (so
the `transfer-remedial` merge can be checked).

Commit subject, or something equally specific: `Evaluate the transfer-function phase and friction from tables`.
