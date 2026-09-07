# Prompt 08 — Region partition and phase-group Levin integration in `QuadSourceIntegral` (A4, part 2; A2, part 3)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A2, A4; `QI-report.md` QI-2, QI-4, QI-5, QI-6, QI-7 and §3;
`docs/resonance-scaffolding/sigw-resonance-reconciliation.md` §1.5, §3.1, §4.2
**Depends on:** 05, 06, 07 (hard). Read their logs' "State handed to the next prompt" first.
**Recommended model:** Fable. This is where the region logic that decides *which* physics is
applied *where* gets written, and where the numerical branch is validated against the analytic
oracle across every regime for the first time.
**Files you may touch:** `ComputeTargets/QuadSourceIntegral.py`, new
`ComputeTargets/tests/test_quadsource_integral.py`, plus the log and the status board.
**Do not** change the Datastore factory (prompt 09), `main.py` (prompt 10), or the `compute()`
payload *keys* beyond adding the four new ones named in §4 (prompt 10 supplies them).

Read first: all of `compute_QuadSource_integral` (`QuadSourceIntegral.py:74-330`),
`numeric_quad_integral` / `WKB_quad_integral` / `WKB_Levin_integral` (`:910-1175`),
`QuadSourceIntegral.compute()` (`:1398-1466`); `GkSourcePolicyData` types `numeric`/`WKB`/`mixed`
and the meaning of `crossover_z`, `Levin_z` (`GkSourcePolicyData.py:130-235, 442-560`); QI-report
§2 QI-2 (the measure, which must be preserved exactly) and QI-6 (the two gates you are replacing).

---

## 1. Character of this commit

Replace the fixed three-region scheme (numeric / WKB-quadrature / WKB-Levin, all driven by the
Green's function alone) with a general partition of $[z_{\rm resp}, z_{\rm src,max}]$ at the
hand-over redshifts of **all three** factors, and integrate each sub-region with the method its
regime requires: ordinary quadrature where everything is smooth, and one `adaptive_levin_sincos`
call per phase group (prompt 07) everywhere else. The stored `total` keeps its meaning — the
spec 03 R28 integral with $Q_s/a_0^2$ stripped (audit §3.1) — and the measure
$(1+z_{\rm resp})\int d\log(1+z')\,G f/H^2$ is unchanged (audit QI-2).

**Out of scope, enforced:** no amplitude/phase output from this function (README §1.1); no schema
change (09); no `main.py` change (10). The existing columns `numeric_quad`, `WKB_quad`,
`WKB_Levin` are *re-populated* with the meanings in §5, so the factory does not need to know
anything changed.

## 2. Inputs

`compute_QuadSource_integral` gains four keyword arguments, threaded from `compute()`'s payload:
`Tq_numeric`, `Tq_WKB`, `Tr_numeric`, `Tr_WKB` (the `TkNumericIntegration` and `TkWKBIntegration`
objects for $q$ and $r$). From them build `TkSourceFunctions` for $q$ and $r$ (prompt 05), and
assert their `crossover_z` agree with `source.crossover_z_q/_r` (prompt 06) when those are not
`None`. In `compute()`, add the corresponding compatibility checks (the `store_id` of $q$ and $r$
match the `Tk` objects' `k`) in the same style as the existing ones at `:1408-1445`.

Until prompt 10 lands, `main.py` will not supply these keys. Make `compute()` raise a clear
`RuntimeError` naming the missing key rather than `KeyError`. Record in the board §3 that the
pipeline is non-runnable between 08 and 10 (it already is between 06 and 08 for `QuadSource`
consumers, so this is not a new state).

## 3. The partition

Breakpoints in $z'$, all within $[z_{\rm resp}, z_{\rm src,max}]$ (clip those outside):

| factor | representation above the breakpoint | below | source of the breakpoint |
|---|---|---|---|
| $G$ | numeric (`Gk_f.numeric_Gk`) | LG (`Gk_f.sin_amplitude`, `Gk_f.phase`) | `GkPolicy.crossover_z` for type `mixed`; type `numeric` = no LG region; type `WKB` = no numeric region (§3.1) |
| $T_q$ | numeric (`Tq_f.T`, `dT_dz`), or the $T=1$ default above its grid | LG (`Tq_f.M`, `dlnM_dz`, `phase`, `omega`) | `Tq_f.crossover_z` |
| $T_r$ | same | same | `Tr_f.crossover_z` |

Sort the breakpoints descending; each sub-interval has a regime `(G_osc, q_osc, r_osc)`.
Conditions to assert, with informative errors: every sub-interval lies inside the region where
each factor's chosen representation is defined (`numeric_region`/`WKB_region` of the respective
functions object); `z_resp` is not below any factor's lowest sample.

**Drop `Levin_z` and `LEVIN_MIN_PHASE_DIFF`.** Both gates (`:126-137, :158-186`;
`GkSourcePolicyData._classify_Levin`) look only at $\theta_G$ and are the audit's QI-6. In the new
scheme every sub-interval with at least one oscillatory factor goes to `adaptive_levin_sincos`,
whose total-variation gate (`AdaptiveLevin/levin_quadrature.py:1260`, verified in
`docs/adaptive-levin-verification.md` §4.2) routes weakly oscillatory sub-regions to
Clenshaw–Curtis itself. Keep reading `GkPolicy.Levin_z` out of the object (it is persisted), but do
not use it for control flow; leave `LEVIN_MIN_2PI_CYCLES`/`LEVIN_MIN_PHASE_DIFF` defined with a
comment that they are unused pending prompt 10's decision (§6).

### 3.1 `GkSourcePolicyData` types

- `numeric`: $G$ smooth everywhere. Regimes are `(False, q_osc, r_osc)`; the "$T$ oscillatory,
  $G$ smooth" rows of README §6 are reachable here and were never handled before.
- `WKB`: $G$ oscillatory everywhere in range.
- `mixed`: breakpoint at `crossover_z`.

## 4. Integration per sub-interval

- **All smooth** `(False, False, False)`: exactly the current `numeric_quad_integral`
  (`:910-978`) — `scipy.integrate.quad` of `G·f/H²` in $\log(1+z')$, with $f$ from
  `source.functions.source` (prompt 06's spline; valid by construction on this region). Keep the
  $(1+z_{\rm resp})$ post-multiplication. Return value **and** the `quad` error estimate.
- **Any oscillatory**: `groups = build_phase_groups(regime, Gk=..., Tq=..., Tr=..., ...)` then one
  `adaptive_levin_sincos(x_span, [g.f_sin, g.f_cos], theta={...}, atol, rtol, chebyshev_order=CHEBYSHEV_ORDER,
  notify_label=...)` per group. Sum values; sum `abserr` **linearly** across groups (the analytic
  branch's reasoning at `:686-689` applies verbatim: the groups share a phase construction);
  `converged = all(...)`, `phase_limited = any(...)`. Multiply by $(1+z_{\rm resp})$.
  **Tolerance per group:** the caller's `atol` is for the whole integral. Follow what
  `prompts/levin-refactor` prompt 05 established for distributing `atol` across sub-intervals
  (read `docs/adaptive-levin-audit-2026-09.md` C3 and the driver's docstring): pass each group
  `atol / (n_groups · n_subintervals)` or the scheme the module documents, and say which.
- **`theta_deriv`:** prompt 07 provides it. Decide whether to pass it to the driver. The comment at
  `:1128-1145` records that the spline derivative is *more* accurate than spectral differentiation
  of the raw phase at large $|\theta|$ and within 2 % in cost, "likely a straight win, but not
  validated end-to-end". Validate it end-to-end here (§7 oracle at $|\theta|\sim10^5$–$10^6$ with
  and without) and pass it if it helps or is neutral. Record as IMPLEMENTATION CHOICE with the
  numbers.

## 5. Outputs into the existing columns

Do not change the return dict's keys. Re-populate:

| key | new meaning |
|---|---|
| `numeric_quad` | sum over all-smooth sub-intervals |
| `WKB_quad` | **0.0** — no sub-interval uses direct quadrature of an oscillatory factor any more. Document in a comment; prompt 09 decides whether to drop the column |
| `WKB_Levin` | sum over all Levin-integrated sub-intervals |
| `total` | `numeric_quad + WKB_Levin` |
| `numeric_quad_data` | aggregate `IntegrationData` over the all-smooth sub-intervals (sum steps/evaluations, sum time, min/max of RHS times) |
| `WKB_quad_data` | `None` |
| `WKB_Levin_data` | aggregate `LevinData` over all Levin calls: sum `num_regions`, `evaluations`, `num_simple_regions`, `num_SVD_errors`, `num_order_changes`; min of `chebyshev_min_order`; max of `max_depth`; sum `elapsed` |
| `metadata["WKB_Levin"]` | extend: per-sub-interval list of `{z_max, z_min, regime, groups: [{label, value, abserr, converged, phase_limited, regions, evaluations}]}`, plus the aggregate `abserr`, `converged`, `phase_limited` |
| `metadata["partition"]` | new: the breakpoint list with which factor produced each |

`analytic_rad` and its metadata are untouched.

## 6. Cost measurement (decides prompt 10's `QuadSourcePolicy` question)

On the offline fixture (§7), take a case in the "$G$ oscillatory, both $T$ smooth" regime where
$\theta_G$ changes by only 2–8 cycles across the sub-interval — the case the old
`LEVIN_MIN_2PI_CYCLES = 10` gate sent to `WKB_quad_integral`. Time and count evaluations for
(a) the old `WKB_quad_integral` on that sub-interval and (b) the new single-group Levin call whose
total-variation gate should route it to Clenshaw–Curtis. Report the ratio. **If (b) is more than
3× slower than (a), open a §3 issue** — the orchestrator will stop and the user decides whether
prompt 10 reinstates a cycle-count threshold via `QuadSourcePolicy.Levin_threshold`. Below 3×, the
threshold stays dead and prompt 10 removes it.

## 7. Tests — `ComputeTargets/tests/test_quadsource_integral.py`

Offline, no Ray, no datastore. Build the exact constant-$w$ LG fixtures for $T_q$, $T_r$ (from
prompt 05's tests) and for $G$ (prompt 07's Oracle 2 construction, wrapped in a stand-in exposing
the `GkSourceFunctions` fields and a stand-in `GkPolicy` with `.type`, `.crossover_z`,
`.functions`), a stand-in `QuadSource` exposing prompt 06's `functions`/`crossover_z_*`, and
stand-in `wavenumber`/`redshift` objects as `docs/spec-code-audit/scripts/QI_02_analytic_numeric.py`
does. Then drive `compute_QuadSource_integral` directly (call the underlying function, not the
Ray remote — check how `QI_02` did it).

**Acceptance oracles**, for $b=0$ and $b=0.2$, at three $(k,q,r)$ shapes — $q\approx r\approx k$
(all three cross together), $q\approx r\gg k$ ($T$'s oscillate while $G$ is still numeric), and
$q\ll k\approx r$ (one $T$ smooth to the end):

1. `total` vs `analytic_rad` (the code's own oracle, `analytic_integral`, verified against spec 04
   R14 to $10^{-8}$–$10^{-6}$ by audit QI-1): agree to **$10^{-5}$ relative** or better, with the
   threshold justified from the fixture floors measured in prompts 05 and 07 (state the arithmetic
   in the log). Three `z_resp` values per shape, spanning "all factors oscillatory over most of the
   range" to "hand-overs inside the range".
2. `total` vs a direct `scipy.quad` of the *exact* integrand $G_{\rm exact} f_{\rm exact}/H^2$ on a
   *short* range (so `quad` converges): agree to $10^{-8}$. This isolates the integrator from the
   analytic-branch's own Bessel-phase splines.
3. **Regime coverage.** Assert, from `metadata["partition"]`, that across the test cases every
   row of README §6 with $n\ge1$ occurs at least once, including "$G$ smooth, $T_q$ and $T_r$
   oscillatory".
4. **Seam continuity.** Split one sub-interval artificially at an interior point and confirm the
   two halves sum to the unsplit value to $10^{-10}$; and for a hand-over breakpoint, evaluate the
   integrand just above (smooth representation) and just below (phase-group sum via
   `evaluate_sum`) and check they agree to the numeric-spline accuracy prompt 06 measured.
5. **Old-regime regression.** In a case with both $T$ super-horizon throughout, the new code
   reproduces the pre-commit `total` to $10^{-10}$ (run the pre-commit function from `git show
   HEAD~1:ComputeTargets/QuadSourceIntegral.py` into a temp module, or record its value before
   editing).
6. **Missing-payload error** from `compute()` is a `RuntimeError` naming the key.

Also re-run `docs/spec-code-audit/scripts/QI_03_measure.py` against the new all-smooth path to show
the R28 measure is unchanged ($10^{-15}$ level).

## 8. Verification

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes, under
  ~3 min (Levin on exact fixtures is fast; if a case is slow, shrink the range, not the tolerance).
- Log: a table of `total` vs `analytic_rad` relative differences per case; the §6 cost ratio; the
  `theta_deriv` decision with numbers.

## 9. Log and commit

Log to `logs/08-qsi-phase-group-integration.md`. **State handed to the next prompt:** the four
payload keys, the redefinition of `WKB_quad`, the aggregate rule for `LevinData`, and the §6 ratio.
Board: row 08, items A4 (2/3), A2 (3/3); §3 issue "pipeline non-runnable until 10" and, if
triggered, the §6 cost issue. One commit; the body states the regimes now handled, the oracle
agreement achieved, and that the Green's-function-only Levin gate is gone.
