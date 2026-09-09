# Implementation state — transfer-function remedial campaign (Bessel phase)

**Campaign:** [`README.md`](README.md) · **Design:** [`DRAFT-PLAN.md`](DRAFT-PLAN.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md)
**Baseline commit:** `95cc326` (`transfer-remedial-plan`, clean)
**Last updated:** 2026-09-09 — campaign planned and committed; no prompt executed.
**Planned against:** `c4c4905`; re-pointed to `95cc326` before commit (`RECONCILIATION.md` §0).

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA, model and log link, update the mechanism-level
> table in §2, and add or clear entries in §3 (Active issues). Do not edit rows other than your own
> except to close an issue you resolved.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — measurement, before anything changes

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [Reference harness](01-reference-harness.md) | plan §9 Stage 1 | Opus | ⬜ | | |
| 02 | [SciPy domain boundaries](02-domain-boundary-tests.md) | plan §4.4; recon C1 | Sonnet | ⬜ | | |

### Workstream B — the two-region construction

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 03 | [Closed-form tail](03-closed-form-tail.md) | plan §7.2 | Opus | ⬜ | | |
| 04 | [Near-region sampler](04-near-region-sampler.md) | plan §7.1, §7.3, §4.5 | Opus | ⬜ | | |
| 05 | [Two-region construction](05-two-region-construction.md) | plan §9 Stage 2, §7.4 | Opus | ⬜ | | |

### Workstream C — evaluation, compatibility and consumers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 06 | [Evaluation and compatibility](06-evaluation-and-compatibility.md) | plan §8.1, §9 Stage 3 | Opus | ⬜ | | |
| 07 | [Bessel phase groups](07-bessel-phase-groups.md) | plan §8.2, §9 Stage 4 | Opus | ⬜ | | |

### Workstream D — revalidation and close-out

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 08 | [Fixture revalidation](08-fixture-revalidation.md) | plan §8.3, §9 Stage 5, §10 | Opus | ⬜ | | |
| 09 | [Benchmark and docs](09-benchmark-and-docs.md) | plan §9 Stage 5, §11 | Sonnet | ⬜ | | |

**Progress:** 0 / 9 complete.

---

## 2. Mechanism-level tracking

Traceability from each defect or requirement to the prompt that discharges it. IDs are local to this
campaign; the plan section is the authority on each.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| M1 | **DEFECT, accuracy** | \(Q=\theta/x\) ODE: \(\delta\theta=x\,\delta Q\), so a relative bound on \(Q\) gives no absolute phase bound (§4.1). The phase is a quadrature, not an ODE (§5.1) | 05 | ⬜ |
| M2 | **DEFECT, spurious** | The `phi` root solve returns a non-zero offset at a match point where the phase is already exact; measured \(-4.836537\times10^{-8}\) at \(\nu=5/2\), and it **is** the whole tight-tolerance error (§4.3) | 05, 06 | ⬜ |
| M3 | **DEFECT, accuracy** | Full-phase interpolation errs by \(h^4x/384\) — 6.7e-10 at \(x=10^3\), 6.7e-6 at \(10^7\) (§4.2); chunking has no measurable effect on it (§4.6) | 05 | ⬜ |
| M4 | **DEFECT, silent failure** | `hankel1e` returns exactly `-0j` above 7.13e8 (\(\nu\gtrsim100\)) / 2.247e15 (all \(\nu\)); `isfinite` passes and `log(abs(·))` is `-inf` (§4.4) | 02, 03, 04 | ⬜ |
| M5 | **DEFECT, hard limit** | `jv`/`yv` become O(1)-relatively noisy above \(x\approx2.5\times10^{15}\), the ODE right-hand side stops being \(1+O(\nu^2/x^2)\), and DOP853 at `rtol=5e-14` stalls — construction never returns (recon C1; **not in the plan**) | 02, 03, 05, 09 | ⬜ |
| M6 | **REQUIREMENT** | Closed-form tail from DLMF 10.18.18, with \(a=(1+r')^{-1/2}\) from the Wronskian and a remainder-tested \(x_\star\). **Required, not deferred** (§1, §7.2) | 03, 05 | ⬜ |
| M7 | **REQUIREMENT** | Branch tracking verified, not assumed: 3.685 rad per interval and ~90 wraps at \(\nu=1000.5\) defeat fixed-density `unwrap` (§4.5, recon C2) | 04 | ⬜ |
| M8 | **REQUIREMENT** | Two-sided adaptivity — refine at the turning point, coarsen in the tail (§4.5) | 04 | ⬜ |
| M9 | **REQUIREMENT** | Two-sided \(a_\nu\) plausibility band, measured \([1.0000,\,3.546]\) over the near region (§4.4, recon §3.1) | 04 | ⬜ |
| M10 | **REQUIREMENT** | \(\theta'=e^{-2\ell}\) as a value, plus the mandatory independent check against \(1+r_u/x\) and the reference, since the Wronskian becomes a tautology (§4.7, §7.3) | 04, 05 | ⬜ |
| M11 | **REQUIREMENT** | Split sin/cos evaluation; naive `x+d` loses 4.7e-2 at \(x=10^{15}\) (§6.3, §7.4) | 05 | ⬜ |
| M12 | **REQUIREMENT** | Bounded-angle accessor via `atan2`, and `raw_theta` documented as \(\varepsilon x\)-limited (§7.4) | 05, 06 | ⬜ |
| M13 | **REQUIREMENT** | Declare `theta_abserr`; nothing supplies it today although `AdaptiveLevin` accepts it for exactly this case (§8.1) | 06, 07 | ⬜ |
| M14 | **MIGRATION** | `Q` (pre-offset ODE state) and `phi` have no referent; `atol`/`rtol` describe ODE tolerances that no longer exist (§8.1) | 06 | ⬜ |
| M15 | **MIGRATION** | Ray serialization of the new representation through `BesselPhaseProxy` (§8.1) | 06 | ⬜ |
| M16 | **DEFECT, dead code** | `plot_besssel_phase.py` reads a nonexistent `x_min` key and calls a non-callable `phase`; it cannot run (recon C3) | 06 | ⬜ |
| M17 | **REQUIREMENT** | Phase groups as \(Kt+C+R(t)\); combine leading coefficients before multiplying by \(t\) (§8.2) | 07 | ⬜ |
| M18 | **REQUIREMENT** | Tests that can see the improvement: 50 % and \(10^{-3}\) thresholds cannot (§9 Stage 1, §10) | 01, 08 | ⬜ |
| M19 | **REQUIREMENT** | Separate the Bessel-oracle gain from the consumer re-spline floor and the physical LG truncation (§8.3) | 08 | ⬜ |
| M20 | **REQUIREMENT** | Re-run the capped benchmark tier at \(\kappa=1000\) and correct its note's diagnosis (§9 Stage 5, recon C1) | 09 | ⬜ |
| M21 | **REQUIREMENT** | Update the follow-up document; remove the stale blanket \(x\times10^{-8}\) claim while retaining the historical measurements (§9 Stage 5) | 09 | ⬜ |

**Out of scope (do not schedule):** `ComputeTargets/QuadSourceIntegral.py` (in-flight
`source-remediation` campaign — handed over in prompt 09); `phase_spline`'s chunking and its
`_build_log_chunks_positive` progress guard; general cosmological stored phases; tightening the
high-order target beyond \(10^{-6}\); widening the supported \((\nu,x_{\max})\) domain. See
`README.md` §1.1 and §7.

---

## 3. Active and unresolved issues

Two entries are opened by the planning pass, before any prompt runs, because they are decisions or
risks that the prompts inherit rather than create.

- **[00-plan-vs-tree-corrections]** *(opened by the planning pass, 2026-09-08)* — four claims in
  `DRAFT-PLAN.md` do not survive reconciliation against the tree, and the prompts are built on the
  corrected versions: **(C1)** the ODE build does not take >600 s at \(x_{\max}=10^{13}\) — it takes
  0.09 s, is flat across five decades, and stalls only above \(x\approx2.5\times10^{15}\) because
  Amos `jv`/`yv` noise defeats the adaptive stepper, so the performance motivation is "a cliff was
  removed", not "a cost curve was removed"; **(C2)** the residual is **not** sub-cycle at high order
  — it spans 565.82 rad ≈ 90 cycles at \(\nu=1000.5\), so the correct reason to drop `phase_spline`
  from this module is \(\varepsilon\lvert r\rvert_{\max}\approx1.3\times10^{-13}\), not "never
  exceeds a cycle"; **(C3)** `plot_besssel_phase.py` is already broken, so migrating it is a
  repair-or-delete decision; **(C4)** `test_phase_derivative` already contracts \(\theta'\) to
  \(10^{-6}\) and is the campaign's standing regression gate, which the plan does not mention.
  **Impact:** prompts 01, 04, 05, 06, 09 each carry the relevant correction inline; an agent that
  works from `DRAFT-PLAN.md` alone will write a false justification into a docstring or a commit
  message. **Next step:** nothing — closed by prompt 09, which records the corrections in `docs/`.
  `DRAFT-PLAN.md` is deliberately left unedited as the revision-2 review record.

- **[00-qsi-three-bessel-levin-excluded]** *(opened by the planning pass, 2026-09-08; re-checked
  2026-09-09 against `95cc326`)* — `ComputeTargets/QuadSourceIntegral.py:1175-1442`
  (`_three_bessel_Levin`) makes eight `adaptive_levin_sincos` calls on signed sums of three
  `bessel_phase` `raw_theta` values with **no `theta_deriv`** and no `theta_abserr`, so Levin obtains
  \(\theta'\) by spectral differentiation of the raw phase there — the route
  `three_bessel_integrals._phase_group`'s docstring exists to avoid — and it has the same
  phase-group cancellation problem prompt 07 fixes in the sibling module. **The gap survived that
  file's own rewrite** by `source-remediation` prompts 08–10, and is now anomalous within its own
  file: the new phase-group route passes `theta_deriv` (`:1008`, `LEVIN_USE_THETA_DERIV = True` at
  `:93`) while these eight calls do not. That campaign's board records only B6 (`atol`/`rtol`
  forwarding) here, not the missing derivative. **Impact:** the analytic comparison branch of
  `QuadSourceIntegral` will not inherit the campaign's improvement, and its accuracy after this
  campaign is unknown. **Next step:** prompt 09 hands it to `prompts/source-remediation` as an entry
  in that campaign's own §3. Nothing here closes it.

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation a later prompt has to
> work around, a measured cost that changes a later prompt's decision. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

*(none yet)*

---

## 5. Standing notes for implementers

1. **`RECONCILIATION.md` outranks `DRAFT-PLAN.md`.** The plan is revision 2's design record and is
   not edited by this campaign. Where they conflict, the conflict is already identified in
   `RECONCILIATION.md` §2; a *new* conflict must be recorded and, if load-bearing, stopped on.
2. **Never write "the residual never exceeds a cycle"** or any equivalent into code, docstrings or
   commit messages. It is false above \(\nu\approx630\) (issue `[00-plan-vs-tree-corrections]` C2).
   The correct statement is that \(\varepsilon\lvert r\rvert_{\max}\approx1.3\times10^{-13}\) at the
   largest supported order, below every acceptance target.
3. **Never sell the campaign on construction speed.** The old build is ~0.1 s across the whole
   production range. The result is that a hard cliff at \(x\approx2.5\times10^{15}\) is removed
   (C1).
4. **`test_phase_derivative` (`test_bessel_phase.py:139`) is the standing regression gate.** It
   contracts \(\theta'\) to \(10^{-6}\) relative at \(\nu\in\{2.5,20.5,100.5\}\) and must pass from
   prompt 05 onward. Prompt 08 tightens it; **nobody loosens it**, and loosening it is a stop
   condition.
5. **`theta` is a required key of the Levin phase dict.** `_Basis_SinCos.__init__` raises without it
   (`levin_quadrature.py:948-952`). "`raw_theta` is compatibility-only" means Levin never
   *evaluates* it when `theta_mod_2pi` and `theta_deriv` are both supplied (`:1038`), not that the
   key may be omitted. The derivative carries double duty — basis conditioning **and** subdivision,
   since `phase_span` comes from `theta_prime_Cheb` (`:1090`).
6. **`XSplineWrapper` and `sample_points` are public surface.**
   `LiouvilleGreen/tests/test_three_bessel.py:10` imports `XSplineWrapper` by name and annotates
   with it at `:67`; `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py:82` passes
   `sample_points`. Neither may break before the prompt that owns it (06 and 05 respectively).
7. **The `mpmath` reference must be built at the supplied double.** Use `mpf(float(x))`, never
   `mpf(repr(x))` — a NumPy scalar stringifies as `np.float64(...)` and `mpf` rejects it — and never
   a re-derived argument, because `DRAFT-PLAN.md` §7.5 defines accuracy *at the supplied* \(x\).
8. **Above \(x\approx2\times10^{15}\), `jv`/`yv` are not a reference.** `bessel_reference.scipy_reference`
   raises there by design (prompt 01). Use the cached `mpmath` corners. `np.sin`/`math.sin` are
   unaffected and are correctly rounded to \(10^{16}\).
9. **The residual branch must be resolved against the tail series, not folded into \((-\pi,\pi]\).**
   \(r(x_0)=570.82\) rad at \(\nu=1000.5\), so a naive fold produces a reference that is wrong by an
   exact multiple of \(2\pi\) — the easiest way to poison every downstream comparison. Prompt 01's
   generator asserts the series match; keep that assertion.
10. **Author conventions.** \(a_0\) is absorbed, never "set to 1" — it lives in \(k/a_0\) and
    \(a_0\eta\). Sign and Jacobian conventions are conventions. The Bessel convention is
    \(J_\nu=A_\nu\sin\theta_\nu\), \(Y_\nu=-A_\nu\cos\theta_\nu\) with \(\theta\) increasing in
    \(x\); DLMF differs by exactly \(+\pi/2\), absorbed into \(c_\nu=\pi/4-\pi\nu/2\). Do not
    "correct" any of these.
11. **`main.py` hunk discipline.** This campaign edits only the Bessel construction stage
    (`main.py:516-528`, prompt 06). The `source-remediation` campaign's edits to the QuadSource and
    QuadSourceIntegral stages have landed, so the stage boundaries are settled. Its **prompt 12
    (verification) has not run**: it runs the pipeline, so if it runs after Workstream B has started
    it will be verifying a tree whose Bessel oracle changed under it. Raise this with the user
    before dispatching Workstream B (README §4.2).
13. **The existing `LiouvilleGreen/tests` discovery run is very slow** — it did **not complete
    within 50 minutes** on the planning machine and was abandoned, dominated by
    `test_3bessel_analytic.py` rebuilding `bessel_phase` objects per case. Prompt 01 records the
    baseline, per module if the discovery run will not finish. Do not read a long run as a regression without comparing to it,
    and prefer per-module runs while iterating.

12. **`phase_spline`'s cosmological chunking has now been measured**, by
    `docs/gk-wkb-numerical-review-2026-09.md` §3 (`39ed7fc`) — the item README §1.1 deferred. Its
    verdict is harsher than `DRAFT-PLAN.md` §4.6's, and it found four further defects. One matters
    to this campaign as *extra evidence*, not extra scope: chunk selection is a hard switch between
    two fits, measured at a **1.08e-4 rad phase jump and a 3.51e-8 relative derivative jump**, which
    a Levin consumer sees. Cite it in prompt 05's justification for removing `phase_spline` from
    `bessel_phase`; do not act on `phase_spline` itself.
