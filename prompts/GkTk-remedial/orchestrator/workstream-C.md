# Orchestrator prompt — Workstream C (prompts 05, 06, 07)

You are orchestrating Workstream C of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

**This is the campaign's substance.** Prompt 05 is a small refactor with one exact-equality test
that must not be weakened. Prompt 06 replaces what production computes for the Green's-function
phase — the point of no return — and deletes ~500 lines of ODE machinery whose every removal has a
measured reason. Prompt 07 completes the transfer function and documents its physical floor.

## What to read

`../README.md` §0–§2 (**(a), (c), (e), (f)**), §4.3, §5, §6, §7 D5–D7; `../RECONCILIATION.md`
§1 items 1–4, 12, 13 and §2 items 3, 4, 5, 6, 10; `../IMPLEMENTATION_STATE.md` (board, §3, §5);
`orchestrator/README.md`; review §2–§4, §6, §8, §12, §13.4; `logs/03-…`, `logs/04-…` "State
handed to the next prompt".

Read each prompt only when about to dispatch it. **Do not read 08–13.**

## Preconditions

`git status` clean; rows 03, 04 ✅/⚠️; 05–07 ⬜. Harness and background tests pass:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_wkb_reference ComputeTargets.tests.test_background_tau ComputeTargets.tests.test_background_cs_tau_friction -v
```

## Dispatching

Standard dispatch text (`workstream-A.md`). Models: **05 → Opus**, **06 → Fable** (Opus if
unavailable; then read every changed function yourself), **07 → Opus**. Run 05 → 06 → 07.

## Reviewing prompt 05 — the residual

Structural checks; allowed files: `phase_residual.py`, `WKB_Gk.py`, `WKB_Tk.py`, two new tests,
log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_omega_eff_split ComputeTargets.tests.test_phase_residual -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
6. **The exact-equality test is `==`**, not `assertAlmostEqual`: read `test_omega_eff_split.py`.
   `omega_WKB_sq` is a stored column; a bit moved is a stop.
7. **No subtraction**: `grep -n "omegaEff_sq(.*) *-\|- *.*_leading" ComputeTargets/phase_residual.py`
   — the integrand must use the `_correction` function, never $\omega^2-\omega_0^2$.
8. Radiation controls asserted exactly ($\rho_G$ `== 0.0`; $\rho_T=1/x_i-1/x$ to $10^{-13}$).
9. Size-sanity bounds present ($\rho_T\in[-0.12,-0.06]$ etc.) — the guard against a sign slip.
10. Numbers: $|\delta\rho|\le10^{-7}$ rad at the checkpoints, both models, three $k$; cost per
    table recorded. If prompt 02 set the fallback flag, the fallback is implemented exactly as
    02's log specified and nothing more.

## Reviewing prompt 06 — the Green's-function producer

Structural checks; allowed files: `WKB_phase_function.py`, `GkWKBIntegration.py`, `WKBtools.py`,
`TkWKBIntegration.py` (**`compute()` call site and imports only** — read the diff), `main.py`
(solver hunk only), `Quadrature/supervisors/WKB.py`, the new test, log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_gk_wkb_phase -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_range_reduce LiouvilleGreen.tests.test_bessel_phase -v
   ```
6. **The removals happened:**
   ```bash
   grep -n "solve_ivp\|Q_INDEX\|DEFAULT_PHASE_RUN_LENGTH\|DEFAULT_OMEGA_WKB_SQ_MAX\|stage_1_evolution\|stage_2_evolution\|integrate_friction_function" Quadrature/integrators/WKB_phase_function.py
   grep -n "sgn_sin_deltaTheta\|shift_theta_sample" ComputeTargets/GkWKBIntegration.py
   ```
   both empty. Each removal's measured reason is in the commit body (prompt 06 §1 tabulates them).
7. **The offset is applied per sample without rebase**: read `WKBtools.apply_phase_offset`. A
   `[0]` base subtraction anywhere in it is a stop (design fact (e)). D7 (delete vs retain
   `shift_theta_sample`) is recorded as an `IMPLEMENTATION CHOICE`.
8. **Sign convention**: the exact-radiation test asserts $\theta=k(1/s_i-1/s)<0$ through the
   production function. Read the line that forms θ: it must be
   $-[k\cdot$`leading.delta(z_init, z)`$+$`rho.delta(z_init, z)`$]$ or an identity the log proves
   equivalent.
9. **`stage_2_data` is an `IntegrationData` with `None` fields**, not `None`, and the log records
   the factory check (`RECONCILIATION.md` §1 item 12). `friction_data` likewise for the interim Tk
   payload.
10. **`TkWKBIntegration` diff is minimal**: `git diff HEAD~1 -- ComputeTargets/TkWKBIntegration.py`
    touches only `compute()`'s call and the `FRICTION_INDEX` import. Anything in `store()` here is
    a stop — it is prompt 07's.
11. **`main.py`**: one hunk at the solver registration.
12. **The cross-object sweep test (4.5) has both halves**: ≤1e-9 rad agreement away from
    $\delta$-wraps, and a wrap count equal to the number of $\delta$ sign wraps. Prompt 09 is built
    on this observation; a test that only asserts smoothness *after* a rectifier is not this test.
13. **Numbers** (README §6): radiation span $10^7$ $\le10^{-8}$ rad; LambdaCDM $k=10^5$
    $\le10^{-5}$ rad; $k=3\times10^8$ $\le5\times10^{-3}$ rad; cost $\le0.05$ s; `sin_coeff > 0`
    in every store-algebra case.
14. The commit message states the finding (phases wrong by cycles) before the cost saving
    (`logs/README.md` requirement 4).

## Reviewing prompt 07 — the transfer-function producer

Structural checks; allowed files: `TkWKBIntegration.py`, the new test, `main.py` solver hunk only
if used, log, board.

4. Tests:
   ```bash
   PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_tk_wkb_phase -v
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
   ```
   `test_tk_source_functions.py` **unchanged** (`git diff HEAD~1 --stat`).
6. `grep -n "friction_RHS\|sgn_sin_deltaTheta\|shift_theta_sample\|FRICTION_INDEX" ComputeTargets/TkWKBIntegration.py` empty.
7. **The radiation value control asserts the floor, not zero**: test 2 exists with the four $x_i$
   thresholds and the $x_i^{-3}$ scaling assertion, and its docstring says the floor is a property
   of the representation. A test that "passes" at $10^{-8}$ from $x_i=24$ is asserting the wrong
   thing and is a stop.
8. The residual part is asserted separately ($\theta_T+k\Delta\tau_s=1/x_i-1/x$).
9. Numbers: $\theta_T$ $\le10^{-4}$ ($k=10^5$) and $\le5\times10^{-3}$ ($3\times10^8$) rad; $F$
   $\le10^{-12}$ relative; cost $\le0.05$ s.

## Continue or stop

Continue on `COMPLETE`-class logs. Stop on the campaign-wide conditions; on 05 check 6 or 7; on
06 checks 6, 7, 8, 10, 12; on 07 check 7; or any missed threshold.

## Completion criterion

Rows 05–07 ✅/⚠️. Report: "Workstream C complete; the tree is at `<SHA>`. Both WKB producers use
the tables; per-object cost is milliseconds; stored phases are at the floor. **Consumers still
spline the growing phase** — the $h^4x/384$ error remains until Workstream D. Ready for prompt
08." Include the README §6 producer rows as measured, the solver labels, and D7's choice.
