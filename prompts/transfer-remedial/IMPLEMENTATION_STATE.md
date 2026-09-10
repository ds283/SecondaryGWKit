# Implementation state — transfer-function remedial campaign (Bessel phase)

**Campaign:** [`README.md`](README.md) · **Design:** [`DRAFT-PLAN.md`](DRAFT-PLAN.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md)
**Baseline commit:** `95cc326` (`transfer-remedial-plan`, clean)
**Last updated:** 2026-09-10 — prompt 07 executed on `f9cc891` (this commit; SHA not self-embedded).
**Planned against:** `c4c4905`; re-pointed to `95cc326` before commit (`RECONCILIATION.md` §0).
**Executing against:** `f9cc891` (prompt 06's commit), 23 commits after `95cc326` (the merge of
`transfer-remedial-plan` into the working branch, `source-remediation` prompt 12's live
verification, the Green-function WKB reviews, three independent fixes, and prompt 01's new
reference harness, plus prompts 01, 02, 03 and 04). None of those touched a previously existing
`LiouvilleGreen/` file — prompts 01–04 only added new ones — so `RECONCILIATION.md` §1, §2 and §3.1
applied verbatim to every prompt up to and including 04, and prompt 01 re-confirmed nine of their
measurements independently (log 01 "Verification performed"). **Prompt 05 was the first commit to
change a previously existing `LiouvilleGreen/` file** and prompt 06 is the first to reach outside
`LiouvilleGreen/` — it migrates `main.py`'s Bessel construction stage and the live
`QuadSourceIntegral_debug` diagnostic. Prompt 05 rewrote `bessel_phase.py`, so
`RECONCILIATION.md` §1's measurements *of the old construction* (`phi`, the ODE cost, the chunk
counts) are now historical rather than current. Its measurements of `hankel1e`, `jv`/`yv`, the tail
series and the split evaluation are properties of SciPy and of the mathematics, and stand unchanged.
Note also that README §4.2's scheduling risk has **cleared**: `source-remediation` prompt 12 has run
(`5b82149`), so Workstream B no longer risks changing the Bessel oracle underneath it. Prompt 07 is
back inside `LiouvilleGreen/` — `three_bessel_integrals.py` and its test — and it is the prompt that
discovered `test_3bessel_analytic.py` has been **failing as a module since prompt 05**, because an
`@unittest.expectedFailure` there now passes; see issue
`[07-abserr-bounds-truth-is-now-an-unexpected-success]`, which is prompt 08's to close.

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA, model and log link, update the mechanism-level
> table in §2, and add or clear entries in §3 (Active issues). Do not edit rows other than your own
> except to close an issue you resolved. **Any change to §3 or §4 must also update the project-wide index
> [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) in the same commit** (see `CLAUDE.md`).

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — measurement, before anything changes

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [Reference harness](01-reference-harness.md) | plan §9 Stage 1 | Opus 5 | ⚠️ | *(this commit; SHA not self-embedded)* | [`01`](logs/01-reference-harness.md) |
| 02 | [SciPy domain boundaries](02-domain-boundary-tests.md) | plan §4.4; recon C1 | Sonnet | ✅ | *(this commit; SHA not self-embedded)* | [`02`](logs/02-domain-boundary-tests.md) |

### Workstream B — the two-region construction

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 03 | [Closed-form tail](03-closed-form-tail.md) | plan §7.2 | Opus 5 | ⚠️ | *(this commit; SHA not self-embedded)* | [`03`](logs/03-closed-form-tail.md) |
| 04 | [Near-region sampler](04-near-region-sampler.md) | plan §7.1, §7.3, §4.5 | Opus 5 | ⚠️ | *(this commit; SHA not self-embedded)* | [`04`](logs/04-near-region-sampler.md) |
| 05 | [Two-region construction](05-two-region-construction.md) | plan §9 Stage 2, §7.4 | Opus 5 | ⚠️ | *(this commit; SHA not self-embedded)* | [`05`](logs/05-two-region-construction.md) |

### Workstream C — evaluation, compatibility and consumers

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 06 | [Evaluation and compatibility](06-evaluation-and-compatibility.md) | plan §8.1, §9 Stage 3 | Opus 5 | ⚠️ | *(this commit; SHA not self-embedded)* | [`06`](logs/06-evaluation-and-compatibility.md) |
| 07 | [Bessel phase groups](07-bessel-phase-groups.md) | plan §8.2, §9 Stage 4 | Opus 5 | ⚠️ | *(this commit; SHA not self-embedded)* | [`07`](logs/07-bessel-phase-groups.md) |

### Workstream D — revalidation and close-out

| # | Prompt | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 08 | [Fixture revalidation](08-fixture-revalidation.md) | plan §8.3, §9 Stage 5, §10 | Opus | ⬜ | | |
| 09 | [Benchmark and docs](09-benchmark-and-docs.md) | plan §9 Stage 5, §11 | Sonnet | ⬜ | | |

**Progress:** 7 / 9 complete.

---

## 2. Mechanism-level tracking

Traceability from each defect or requirement to the prompt that discharges it. IDs are local to this
campaign; the plan section is the authority on each.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| M1 | **DEFECT, accuracy** | \(Q=\theta/x\) ODE: \(\delta\theta=x\,\delta Q\), so a relative bound on \(Q\) gives no absolute phase bound (§4.1). The phase is a quadrature, not an ODE (§5.1) | 05 | ✅ |
| M2 | **DEFECT, spurious** | The `phi` root solve returns a non-zero offset at a match point where the phase is already exact; measured \(-4.836537\times10^{-8}\) at \(\nu=5/2\), and it **is** the whole tight-tolerance error (§4.3) | 05, 06 | ✅ |
| M3 | **DEFECT, accuracy** | Full-phase interpolation errs by \(h^4x/384\) — 6.7e-10 at \(x=10^3\), 6.7e-6 at \(10^7\) (§4.2); chunking has no measurable effect on it (§4.6) | 05 | ✅ |
| M4 | **DEFECT, silent failure** | `hankel1e` returns exactly `-0j` above 7.13e8 (\(\nu\gtrsim100\)) / 2.247e15 (all \(\nu\)); `isfinite` passes and `log(abs(·))` is `-inf` (§4.4) | 02, 03, 04 | ✅ |
| M5 | **DEFECT, hard limit** | `jv`/`yv` become O(1)-relatively noisy above \(x\approx2.5\times10^{15}\), the ODE right-hand side stops being \(1+O(\nu^2/x^2)\), and DOP853 at `rtol=5e-14` stalls — construction never returns (recon C1; **not in the plan**) | 02, 03, 05, 09 | 🟡 |
| M6 | **REQUIREMENT** | Closed-form tail from DLMF 10.18.18, with \(a=(1+r')^{-1/2}\) from the Wronskian and a remainder-tested \(x_\star\). **Required, not deferred** (§1, §7.2) | 03, 05 | ✅ |
| M7 | **REQUIREMENT** | Branch tracking verified, not assumed: 3.685 rad per interval and ~90 wraps at \(\nu=1000.5\) defeat fixed-density `unwrap` (§4.5, recon C2) | 04 | ✅ |
| M8 | **REQUIREMENT** | Two-sided adaptivity — refine at the turning point, coarsen in the tail (§4.5) | 04 | ✅ |
| M9 | **REQUIREMENT** | Two-sided \(a_\nu\) plausibility band, measured \([1.0000,\,3.546]\) over the near region (§4.4, recon §3.1) | 04 | ✅ |
| M10 | **REQUIREMENT** | \(\theta'=e^{-2\ell}\) as a value, plus the mandatory independent check against \(1+r_u/x\) and the reference, since the Wronskian becomes a tautology (§4.7, §7.3) | 04, 05 | ✅ |
| M11 | **REQUIREMENT** | Split sin/cos evaluation; naive `x+d` loses 4.7e-2 at \(x=10^{15}\) (§6.3, §7.4) | 05 | ✅ |
| M12 | **REQUIREMENT** | Bounded-angle accessor via `atan2`, and `raw_theta` documented as \(\varepsilon x\)-limited (§7.4) | 05, 06 | ✅ |
| M13 | **REQUIREMENT** | Declare `theta_abserr`; nothing supplies it today although `AdaptiveLevin` accepts it for exactly this case (§8.1) | 05, 06, 07 | ✅ |
| M14 | **MIGRATION** | `Q` (pre-offset ODE state) and `phi` have no referent; `atol`/`rtol` describe ODE tolerances that no longer exist (§8.1) | 06 | ✅ |
| M15 | **MIGRATION** | Ray serialization of the new representation through `BesselPhaseProxy` (§8.1) | 06 | ✅ |
| M16 | **DEFECT, dead code** | `plot_besssel_phase.py` reads a nonexistent `x_min` key and calls a non-callable `phase`; it cannot run (recon C3) | 06 | ✅ |
| M17 | **REQUIREMENT** | Phase groups as \(Kt+C+R(t)\); combine leading coefficients before multiplying by \(t\) (§8.2) | 07 | ✅ |
| M18 | **REQUIREMENT** | Tests that can see the improvement: 50 % and \(10^{-3}\) thresholds cannot (§9 Stage 1, §10) | 01, 08 | 🟡 |
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

- **[01-scipy-jv-yv-high-order-boundary]** *(opened by prompt 01, 2026-09-10)* — the silent Amos
  failure boundary of `DRAFT-PLAN.md` §4.4 and `RECONCILIATION.md` §1 is **order dependent, and it
  applies to `jv`/`yv`, not only to `hankel1e`**. That measurement covered `hankel1e` only, which
  matters because `jv`/`yv` are what every reference and every existing test in this area is built
  from — `bessel_reference.scipy_reference`, `test_bessel_phase.test_phase_derivative`
  (`:134`), `test_high_order` (`:108-109`) and the plan's own §12.1 script. Measured with
  \(a=\sqrt{\pi x/2}\,\lvert(J,Y)\rvert\), which is \(1+O(\nu^2/x^2)\) and so must be 1 to twelve
  places over \(10^8\le x\le2\times10^{15}\): \(\max\lvert a-1\rvert\) is 4.4e-16 (\(\nu=2.5\)),
  1.0e-14 (20.5), 6.4e-14 (50.5), 1.6e-13 (80.5), **1.8e-13 (85.5)** — and then **1.0 (88.5)**,
  0.998 (89.5), 1.0 (90.5, 100.5, 1000.5). In \(x\), at \(\nu=100.5\), the transition is abrupt:
  \(a\) = 0.9999999872 at \(x=7.108\times10^8\) and 0.0848 at \(7.188\times10^8\), reproducing
  `RECONCILIATION.md` §1's 7.13e8 for `hankel1e` to three figures. The values are finite and
  non-zero, so `isfinite` passes them. **Impact:** prompt 01 made
  `bessel_reference.scipy_reference` refuse above `scipy_reference_max_x(nu)` — 7.13e8 for
  \(\nu>85.5\), 2e15 otherwise — so nothing in this campaign can be scored against a wrong
  reference. What is *not* settled is the order threshold, which is only bracketed between 85.5 and
  88.5 and is set conservatively at the lower end; nor is any of it asserted as a test.
  **Next step:** prompt 02 pins both boundaries for `jv`/`yv` as well as for `hankel1e`, and
  narrows the order threshold if it is cheap to do so. Nothing outside `LiouvilleGreen/tests/` needs
  to change: no consumer in the tree evaluates `jv`/`yv` above 1e7 at high order.

- **[03-draft-plan-tail-coefficient-wrong]** *(opened by prompt 03, 2026-09-10)* — the **third
  coefficient of DLMF 10.18.18** as printed in `DRAFT-PLAN.md` §7.2, in prompt 03 §2 and in its §5
  reproduction script has denominator **15360**; the correct denominator is **5120**, a factor of
  three. Determined numerically, not guessed: with the first two (correct) terms subtracted from a
  120-digit `mpmath` residual on its resolved branch, \((r_{\rm true}-r_2)x^5\) converges to
  0.19999998 (\(\nu=3/2\)), −5.39999928 (5/2) and 864675.13 (20.5) against numerators 1024, −27648
  and 4427136000 — ratio 5120.0 at all three — and Abramowitz & Stegun 9.2.29's grouping gives
  \(32/(5\cdot8^5)=1/5120\) exactly, while its \(4/(3\cdot8^3)=1/384\) reproduces the plan's
  *second* denominator. So the plan folded the wrong power of 8 into the third term only. The
  fourth coefficient, obtained the same way, is **229376** (\(=7\cdot8^7/64\)). **Impact:** an
  agent that takes the series from the plan text ships a "three-term" series that removes only one
  third of the two-term error (measured 1.769e-10 → 1.179e-10 at \(\nu=5/2,x=125\), instead of
  → 2.42e-14) and a crossover sized by an estimator that does not describe the remainder.
  `LiouvilleGreen/bessel_tail.py` ships 5120/229376 and
  `test_bessel_tail.test_coefficients_match_the_published_series` pins them; nothing else in the
  tree is affected — `bessel_reference.tail_residual_series` carries only the first two
  coefficients, 8 and 384, which are correct. **Next step:** prompts 04 and 05 must take
  coefficients from `bessel_tail.tail_series_coefficients`, never from the plan text; prompt 09
  records the correction in `docs/` alongside `[00-plan-vs-tree-corrections]`, which closes this.
  `DRAFT-PLAN.md` is deliberately not edited (README §5 item 9).

- **[04-achieved-estimates-exclude-the-sampling-floor]** *(opened by prompt 04, 2026-09-10)* —
  `NearRegionData`'s three `achieved_*` numbers are measured by resampling `hankel1e` at points
  interior to each panel and comparing against the interpolants there. Both sides of that
  comparison come from the same function, so the estimate is of **interpolation error only**: a
  systematic bias in `hankel1e`'s own phase or modulus cancels and would not be seen. Empirically
  it does not matter yet — against the committed 40-digit corners the estimator *over*-reports the
  measured error by 1.29x to 4.76x at every order from 3/2 to 1000.5, so it is conservative in the
  direction prompt 05 needs — but the margin is not structural, and at \(\nu=1000.5\) it is within
  a factor 5 of `DRAFT-PLAN.md` §4.4's 2.96e-13 `hankel1e` phase floor. That floor was itself
  measured before prompt 04's rotation-constant fix (log 04 Deviation 5), which removed a
  6.6e-14–2.3e-13 contributor to it, and has not been re-measured since. **Impact:** prompt 05
  propagates these numbers into the Levin quadrature's `theta_abserr`, whose whole purpose is that
  the caller sees an honest number; if it does so unmodified it is asserting a bound on
  interpolation, not on the representation.
  **Narrowed (2026-09-10, prompt 05):** the published number is no longer unmodified.
  `bessel_phase.SAMPLED_PHASE_FLOOR = SAMPLED_AMPLITUDE_FLOOR = 3e-13` — `DRAFT-PLAN.md` §4.4's
  measured 2.96e-13 at \(\nu=1000.5\), rounded up and applied at every order — is added to the near
  region's `achieved_*` values before they enter `theta_abserr`, alongside a new
  `EVALUATION_FLOOR = 4\varepsilon` for the angle-addition arithmetic itself (without which
  \(\nu=1/2\), where the representation is exact, declared 0.0 against a measured 1.11e-16).
  `theta_abserr` is now measured to over-report the corner-scored error by 4× to 8000× at every
  order from 1/2 to 1000.5 (log 05), and `test_bessel_two_region` asserts it never under-reports.
  **What remains open** is only the *size* of the constant: it is a pre-prompt-04 measurement
  carried forward, taken before the rotation-constant fix removed a 6.6e-14–2.3e-13 contributor to
  it, and it is not binding — the tail series remainder (2.5e-12) dominates it by a factor 8 at
  every order. **Next step:** re-measure the post-fix `hankel1e` phase floor against `mpmath` and
  reduce the constant, or confirm it. Cheap, and nothing waits on it.

- **[05-3bessel-analytic-not-run-to-completion]** *(opened by prompt 05, 2026-09-10)* — prompt 05's
  §5 acceptance includes `unittest discover -s LiouvilleGreen/tests -t .`, and every module in it
  passed except `test_3bessel_analytic.py`, which was started, made normal progress (tests passing,
  no failures, only the expected `DeprecationWarning`s from its `atol`/`rtol` call sites) and was
  **not seen to completion**. This is the pre-existing multi-hour module of `RECONCILIATION.md`
  §3.4 — it did not finish within 50 minutes on the planning machine and 15 minutes on prompt 01's
  — because it rebuilds `bessel_phase` objects per case over many cases. There is no evidence
  prompt 05 makes it slower: construction is now ~2.5 ms rather than ~60–100 ms, and the two
  sibling modules that exercise the same evaluation path both got faster (`test_bessel_phase`
  1.2 s → 0.18 s, `test_three_bessel` 7.5 s → 4.1 s). **Impact:** its tolerance bands (1e-5/1e-6, and 1e-2/1e-3 near singularities) have not
  been re-scored against the new oracle. `DRAFT-PLAN.md` §9 Stage 4 warns that an eight-order
  improvement can *expose a different limiting error rather than simply pass more easily*, so a
  failure there would be a finding rather than a nuisance. Every other consumer was run and passed:
  `test_bessel_phase` (4), `test_three_bessel` (2), the five other `LiouvilleGreen` modules (69),
  and `ComputeTargets` `test_tk_source_functions` + `test_phase_groups` (30). **Narrowed (2026-09-10, prompt 07):** three of its six tests have now been run individually
  under the new oracle. `test_JJJ` and `test_YJJ` — the two that carry `ABS_TOLERANCE` and
  `REL_TOLERANCE` — **pass**, in 6m41s together, at relative errors 2.3e-13 to 5.5e-10 on their
  random draws; `test_abserr_bounds_truth` is an unexpected success, which is now its own issue
  `[07-abserr-bounds-truth-is-now-an-unexpected-success]`. What is still unrun is the multi-hour
  part: `test_YJJ_log_singularity` and `test_YJJ_log_scaling`, i.e. exactly the 1e-2/1e-3
  singularity bands. **Next step:** run
  `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_3bessel_analytic` on a
  machine that can give it hours, before or as part of prompt 08, which owns those tolerances.

- **[05-quadsource-order-check-docstring-stale]** *(opened by prompt 05, 2026-09-10)* — prompt 05
  added a `"nu"` key to `bessel_phase`'s returned dict, for prompt 07's phase groups.
  `ComputeTargets/QuadSourceIntegral.py:680` (`_check_bessel_order`) has a long docstring stating
  that the dict "carries phase, mod, Q, phi, bessel_j, bessel_y, min_x, max_x and no `nu`", and
  explaining that the order therefore has to be checked *numerically* — by comparing the spline's
  own reconstruction against `jv`, normalised by the envelope. That reasoning is now obsolete.
  **Impact:** none functional. The numeric check still runs and still passes
  (`ComputeTargets.tests.test_quadsource_integral` was not re-run here, but
  `test_tk_source_functions` and `test_phase_groups` were and are OK); only the docstring is wrong,
  and it is wrong in the safe direction. `ComputeTargets/` is out of scope for this campaign
  (README §4.2), so nothing was changed. **Next step:** hand to `prompts/source-remediation`
  alongside `[00-qsi-three-bessel-levin-excluded]` when prompt 09 makes that hand-off; that
  campaign can either correct the docstring or replace the numeric check with a direct comparison
  against `phase_data["nu"]`. **Extended (2026-09-10, prompt 06):** the same docstring is now wrong
  twice over — it lists `Q` among the members the dict "carries", and prompt 06 removed `Q`. Still
  no functional impact (`test_quadsource_integral` passes, 97 `ComputeTargets` tests green), and
  still one docstring for that campaign to correct.

- **[06-levin-theta-docstring-stale]** *(opened by prompt 06, 2026-09-10)* —
  `AdaptiveLevin/levin_quadrature.py:2750` documents the `theta` key as "always used to decide
  whether a subinterval is oscillatory enough for the Levin rule (via the total phase change across
  it)". It is not: `phase_span` comes from `theta_prime_Cheb` (`:1090`) and `need_theta_Cheb` is
  `False` whenever `theta_mod_2pi` and `theta_deriv` are both supplied (`:1038`). `DRAFT-PLAN.md`
  §8.1 predicted the docstring was stale rather than the plan wrong, and prompt 06 now has direct
  evidence: `test_bessel_compatibility.test_the_raw_phase_is_never_evaluated` re-runs a converged
  Levin call with a `theta` that raises `AssertionError` and gets bit-identical `value` and
  `abserr`, on both a near-region and a tail span. **Impact:** a reader of that docstring will
  believe `raw_theta` must be accurate, which at \(x=10^{15}\) it cannot be (2.2e-1 rad); the code
  is right and only the prose is wrong. **Extended (2026-09-10, prompt 07):** a *second* docstring
  in the same file is stale in the same way — `_sample_vectorized` (`levin_quadrature.py:898`) says
  the `three_bessel_integrals.py` / `QuadSourceIntegral.py` phase and modulus callables "do not
  [vectorize], as of this commit: their phase and modulus splines branch on a scalar argument".
  Prompt 05's `BesselPhaseFunction` accessors are array-safe, and prompt 07 checked directly with
  `_detect_vectorized` that all four of a phase group's callables — and the three the old route
  supplied — are detected as vectorising, worth a measured 13.5 ms against 71.8 ms for 200
  samplings of a 13-point grid. **Next step:** correct both docstrings in a future `AdaptiveLevin`
  prompt. `AdaptiveLevin/` is forbidden to this campaign (README §5 item 8), so nothing here can
  close it.

- **[06-measure-bessel-phase-num-chunks]** *(opened by prompt 06, 2026-09-10)* —
  `docs/transfer-remedial/measure_bessel_phase.py` (prompt 01's diagnostic) reads
  `data["phase"].num_chunks` at `:276` and `:307`. That was a `phase_spline` member; prompt 05's
  `BesselPhaseFunction` does not have it, so both of the script's *current-tree* sections now raise
  `AttributeError`. Its historical-module section (`:130`, which loads the old `bessel_phase` out of
  git) is unaffected, as is its `data["phi"]` read at `:283`. **Impact:** the campaign's own
  before/after measurement script cannot be re-run against the new construction without an edit —
  which matters because prompt 09 is the prompt that wants to. Nothing in production or in a test
  reads `num_chunks`. **Next step:** prompt 09 owns `docs/`; either drop the column or replace it
  with `near_region.n_panels`, which is the closest thing the new construction has.

- **[06-three-bessel-plot-calls-a-non-callable-phase]** *(opened by prompt 06, 2026-09-10)* —
  `ComputeTargets/QuadSourceIntegral_debug.three_bessel_plot` calls the phase object directly
  (`phase_A(x1)`, `phase_B(x2)` at `:271-273`, `:315-317` and `:438-462`). Neither `phase_spline`
  nor `BesselPhaseFunction` defines `__call__`, so that function has been dead since before this
  campaign — the same defect as `RECONCILIATION.md` C3's `plot_besssel_phase.py`, in a file the
  consumer inventory lists as a *live* diagnostic. Prompt 06 migrated `Q` in the sibling function
  `bessel_function_plot` (which now runs end to end, 34 files) and was told to leave everything else
  alone, so this was not touched. **Impact:** the three-Bessel integrand comparison plots cannot be
  produced; nobody has noticed, which is itself information about how much the function is used.
  **Next step:** either repair it (`.raw_theta` at each call site, plus a check that the summed
  phases it plots should now be built from `phase.residual` per prompt 07's decomposition rather
  than from raw phases) or delete the function. It is a repair-or-delete judgement like C3's and
  belongs to whoever next needs the plot.

- **[07-generic-K-product-rounding]** *(opened by prompt 07, 2026-09-10)* — prompt 07's
  \(Kt+C+R(t)\) assembly removes the cancellation error of a near-resonant group phase, but the
  group phase is still one double, so it retains one rounding of the **product** \(Kx\). Prompt 07
  §2's error model accounts for \(t\,\delta K\) and not for that, and it is what binds whenever
  \(K\) is not small: measured at orders (1/2, 3/2, 5/2), signs \((+,-,-)\) and \(K=0.1\),
  \(\lvert\delta\Theta\rvert=1.526\times10^{-5}\) rad at \(x=10^{12}\) — exactly one ulp of
  \(Kx=10^{11}\) — for **both** the new and the old route, so a non-resonant group gains nothing
  from the restructure. Near resonance it is irrelevant (\(Kx\) is small) and the new route
  measures 1.4e-13 against the old 2.9e-5. **Impact:** it caps what a three-Bessel integral can
  assert at large `max_x` for a generic triple. `test_3bessel_analytic.py` runs at
  \(x_{\max}=10^{12}\) with a same-sign group of \(K=k+q+s\), so this is a floor on prompt 08's
  re-tightening there, alongside the `MAX_X` truncation of log 07 observation 4. **Next step:**
  either accept it and record it in `docs/` (prompt 09), or split \(K\cdot x\) into a two-product
  \((hi, lo)\) — Dekker/Veltkamp, or `math.fma` on Python 3.13+ — and fold `lo` into the small
  angle of `_PhaseGroup.sin_cos`, which would take the trigonometric path to ~1e-16 at any \(K\).
  Deliberately not done in prompt 07: the prompt prescribes handing the unreduced product to libm,
  and this is a second change to the same expression.

- **[07-abserr-bounds-truth-is-now-an-unexpected-success]** *(opened by prompt 07, 2026-09-10)* —
  `LiouvilleGreen/tests/test_3bessel_analytic.py::test_abserr_bounds_truth` is an
  `@unittest.expectedFailure` asserting that `quad_JJJ`/`quad_YJJ`'s reported `abserr` does *not*
  bound the true error against the analytic oracles. It now **passes**, so `unittest` reports
  `FAILED (unexpected successes=1)` and the whole module fails. Measured on both trees: at
  `bc31493` (prompt 04, before the oracle changed) it is `OK (expected failures=1)` with 5 of 7
  oracles underbound by up to 11.5×; at `f9cc891` (prompt 06) it is an unexpected success with 7 of
  7 bound, because prompt 05 dropped the true error by ~8 orders (3.0e-10 → 3.8e-14 on JJJ110)
  while the reported `abserr` barely moved. **So this is prompt 05's consequence, discovered by
  prompt 07** — nobody saw it earlier because that module takes hours and has never been run to
  completion (`[05-3bessel-analytic-not-run-to-completion]`). Prompt 07 makes the reported `abserr`
  0–1.9 % *larger*, which keeps it passing; it cannot and should not restore the failure.
  **Impact:** `unittest discover -s LiouvilleGreen/tests -t .` cannot pass until this is addressed,
  so prompt 07's acceptance item to that effect is unmet (log 07 Deviation 1). **Next step:** prompt
  08 owns that file. The test's own docstring says closing it needs "`theta_abserr` wired up to
  consume" a declared phase accuracy — that half is now done for this module, so the honest repair
  is to drop the `@unittest.expectedFailure` and keep the assertion, not to weaken it.

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

14. **Take the tail series coefficients from `LiouvilleGreen.bessel_tail.tail_series_coefficients`,
    never from `DRAFT-PLAN.md` §7.2 or prompt 03 §2.** Both print 15360 for the third denominator;
    it is 5120, and the fourth is 229376 (issue `[03-draft-plan-tail-coefficient-wrong]`). Prompt
    03 also settled the crossover: three terms shipped with the fourth as the omitted-term
    estimator, `safety = 0.25` on both budgets, and \(x_\star/\nu\) running from **4.4** (\(\nu=3/2\),
    budget \(10^{-6}\)) to **58.1** (\(\nu=1000.5\), \(10^{-11}\)) — so the sampled near region is
    1.5–4.1 e-folds, not `DRAFT-PLAN.md` §1's "fixed ≈4.6", and \(x_\star\) is **not** a fixed
    multiple of \(\nu\). \(\nu=1/2\) returns \(x_\star=x_0\): the sampler must never run there.
    The full table is in `logs/03-closed-form-tail.md`.

15. **Above \(\nu=20.5\), SciPy `jv`/`yv` are not an adequate reference for series work**, well
    below `scipy_reference_max_x(nu)`. Measured against 50-digit `mpmath` over
    \(5\nu\le x\le100\nu\): \(\lvert\delta r\rvert\) up to 1.2e-12 at \(\nu=100.5\) and 8.4e-12 at
    1000.5, \(\lvert\delta a/a\rvert\) up to 1.7e-11 — above the three-term tail series' own error
    at those orders. Use the `mpmath` tier or the cached corners there; it costs under 10 ms per
    point when \(x\gg\nu\). This is a *precision* floor, distinct from the silent-failure boundary
    of `[01-scipy-jv-yv-high-order-boundary]`.

16. **The near-region residual is on the tail-continuous branch, and `wraps_tracked` is a property
    of the interval rather than of the order.** `bessel_near_region.build_near_region` anchors so
    that \(r\to0\) as \(x\to\infty\) — the branch `bessel_tail`, `RECONCILIATION.md` C2 and
    `bessel_reference_data.json` all use — by taking the *integer cycle only* from
    `tail_residual(nu, x_star)`; the value is always the sampled \(\arg S_\nu\). So the crossover
    comparison at \(x_\star\) is a direct subtraction with no \(2\pi\) bookkeeping, and
    \(r(x_0)=570.820039\) at \(\nu=1000.5\) rather than \(-0.9498\). Consequently `wraps_tracked`
    counts crossings over \([x_{\rm lo},x_\star]\): **89** at \(\nu=1000.5\) with the \(10^{-11}\)
    crossover (\(x_\star=58.09\nu\)) and **83** with the \(10^{-6}\) one (\(11.22\nu\)).
    `RECONCILIATION.md` C2's 90.05 is measured to \(100\nu\); a later test that re-asserts
    \(90\pm1\) must say which build it means (log 04 Deviations 4 and 6).

17. **The \(1+r_u/x\) derivative route is reported, not contracted.** `bessel_near_region` drives
    refinement on the value accuracy of \(r\) and \(\ell\) plus two branch conditions, and
    publishes the alternative route's measured agreement as `achieved_deriv_alt_relerr`. Using it
    as a refinement criterion **diverges**: differentiating an interpolant amplifies the sampling
    floor like \(N^2/h\), so bisection makes the estimate worse — measured, the failing-panel
    count doubled every pass and the build reached 6193 panels at \(\nu=1000.5\) without
    converging (log 04 Deviation 2). The shipped route \(\theta'=e^{-2\ell}\) is contracted by
    `amplitude_rtol`, since \(\delta\theta'/\theta'=-2\,\delta\ell\) exactly. Do not reinstate the
    criterion; the two routes are measured to agree to 1.4e-13 at low order and 6.9e-11 at
    \(\nu=1000.5\), inside every budget.

18. **`bessel_phase` now returns a much larger dict, and the four old members mean what they used
    to.** `phase`, `mod`, `bessel_j`, `bessel_y`, `min_x`, `max_x` keep their names and calling
    conventions; `phi` survives and is **identically 0.0** at every order. **`Q` was removed by
    prompt 06** — `data["Q"]` raises a `KeyError` naming `phase.residual`, `phase.log_amplitude`
    and `phase.raw_theta`, because `Q` meant the *pre-offset ODE state* and there is no ODE, no
    offset and no state (`DRAFT-PLAN.md` §8.1 forbids serving a different quantity under the name).
    The returned mapping is now a `_BesselPhaseData`, a module-level `dict` subclass whose only
    added behaviour is that message; `isinstance(data, dict)` holds and both `pickle` and
    `ray.cloudpickle` round-trip it. Added: `nu`, `x_star`, `theta_abserr`, `amplitude_relerr`,
    `theta_deriv_relerr`, `accuracy` (a 19-key breakdown), `accuracy_met`, `crossover` (prompt
    03's `TailCrossover`) and `near_region` (prompt 04's `NearRegionData`, or **`None`** at
    \(\nu=1/2\), which anything walking `log_x_nodes` must handle). Accessor signatures are in
    log 05's "State handed to the next prompt" and log 06's, verbatim — prompt 06 added
    `phase.log_amplitude(x, x_is_log=False)` (`ell = log a_nu`, the diagnostic partner of
    `residual`) and `phase.theta_abserr_at(x, x_is_log=False)`. Note the keyword names still differ
    between the two objects deliberately: `phase.*` takes `x_is_log` and `mod.*` takes `is_log`,
    matching what each replaced.

19. **`theta_mod_2pi` now returns \((-\pi,\pi]\) from `atan2(sin,cos)` of the split pair**, where
    the old one returned `fmod(theta, 2*pi)` in \((-2\pi,2\pi)\). Both are valid representatives;
    every consumer checked takes only \(\sin\)/\(\cos\) of it (`levin_quadrature.py:1103-1105`,
    `:1120-1121`; `three_bessel_integrals._phase_group`). **Never difference `raw_theta` values to
    build a phase group** — it is \(\varepsilon\theta\)-limited by construction, 2.2e-1 rad at
    \(x=10^{15}\). Prompt 07 should combine the leading coefficients and the `c_nu` constants
    before multiplying by \(t\), and take \(R(t)\) from `phase.residual`, which exists for exactly
    that. `phase.c_nu` and `phase.c_nu_reduced` are both exposed; the reduced one is what
    \(\sin\)/\(\cos\) use, and it is the one to sum when only trigonometric values are needed.

20. **`atol`/`rtol` are accepted, ignored and warned about; nothing translates them.** They named
    tolerances of an ODE solve that no longer exists, so a mapping would be invented. Defaults are
    `None`, so a caller that omits them gets no warning; the new arguments are `phase_atol`
    (absolute radians) and `amplitude_rtol` (relative), both defaulting to `1e-11`, and they win if
    both old and new are supplied. **`main.py` no longer uses the shim**: prompt 06 migrated both
    production builds to `phase_atol=1e-12, amplitude_rtol=1e-12` (declared phase error 5.0e-13 at
    \(\nu=5/2\), 8.9e-16 at \(\nu=1/2\); 1e-13 was measured and buys nothing because the
    sampling floor binds). Five call sites in the tree still pass the old names and now emit
    one `DeprecationWarning` each: `test_three_bessel.py` and `test_3bessel_analytic.py` (prompt 08),
    `docs/adaptive-levin-benchmark/levin_bench/bessel_tier.py` (prompt 09, and it already suppresses
    warnings), plus the two `docs/` scripts, which nobody owns and which still run, and
    `ComputeTargets/tests/test_quadsource_integral.py:482-483`, which
    `RECONCILIATION.md` §3.3 did not list and which prompt 06 added to the inventory (it passes).

21. **Construction cost is now flat in `max_x` and depends only on the order** — 0.0022–0.0034 s
    for \(\nu=5/2\) from \(x_{\max}=10^3\) to \(10^{16}\), 0.043 s at \(\nu=1000.5\). The declared
    ceiling is `MAX_SUPPORTED_X = 1e16`, and \(3\times10^{15}\) and \(8.6\times10^{15}\) — the two
    arguments at which the old construction did not return — build in 2.5 ms. **Still do not claim
    a speed-up ratio** (standing note 3): the result to claim is that a cliff was removed.

22. **`theta_abserr` is a scalar; `theta_abserr_at` is the callable, and the callable is what a
    Levin consumer should pass.** `AdaptiveLevin` accepts either (`levin_quadrature.py:967-982`),
    and it applies the declared value as an endpoint term on *every* region's round-off floor — so a
    domain-wide scalar inflates the reported error of a far-tail region by the ratio of the two
    regions' errors, measured at **262.7×** on a tail span at \(\nu=5/2\) (log 06). The four-key
    dictionary to build is
    `{"theta": phase.raw_theta, "theta_mod_2pi": phase.theta_mod_2pi, "theta_deriv":
    phase.theta_deriv, "theta_abserr": phase.theta_abserr_at}`. Supplying it makes the reported
    `abserr` **larger**, not smaller — 3.54e-13 → 2.54e-11 on a near-region span — and that is the
    point: the phase construction, not the quadrature, is the limit there. A test that expects
    `abserr` to shrink is expecting the wrong thing. For a phase *group*, sum the constituents'
    `theta_abserr_at` at their own arguments; they are independent.

23. **`DECLARED_SERIES_SAFETY = 2.0` is applied to the tail series remainder when it is *declared*
    and nowhere else.** `bessel_tail.tail_first_omitted_term` is an estimator, not a bound: its
    docstring records the truth as 0.98–1.02× it, and measured over 1500 points on
    \([x_\star,3x_\star]\) at five orders the worst measured/estimator ratio was **0.99994** — below
    1 everywhere, with 0.006 % to spare (log 06). Declaring that 1:1 is not a margin, and combining
    the tail terms with `max()` rather than by summation under-reported outright by 3 % at
    \(\nu=20.5,x=2050\). So the declared low-order `theta_abserr` is **5.0e-12**, not log 05's
    2.5e-12, and the worst declared/measured ratio at any cached corner is 2.05. `x_star` is
    unaffected: `bessel_tail.DEFAULT_CROSSOVER_SAFETY = 0.25` still sizes the crossover alone, and
    the two factors must not be conflated. `accuracy["tail_first_omitted_at_x_star"]` reports the
    **raw** estimator; multiply by `accuracy["declared_series_safety"]` for the declared term.

24. **Above \(\nu=20.5\) is not the only place SciPy is too coarse to score a declaration
    against.** Prompt 06's dense `theta_abserr` sweep failed against `scipy_reference` at
    \(\nu=1/2,x=16.2576\) with a "measured" error of **1.07e-14** — where the representation is
    exact (\(r\equiv0\), \(a\equiv1\)) and agrees with the closed form to an ulp, so the whole
    1.07e-14 is Amos's. Standing note 15's precision floor therefore bites two orders of \(\nu\)
    lower than it was measured at, whenever the quantity under test is at the 1e-15 level. Use
    `TIER_MPMATH` or `TIER_EXACT` for anything that small; `mpmath` costs ~2.7 ms per point at
    \(x\le10^7\).

25. **A three-Bessel phase group is now an object, and its four Levin keys must stay
    array-clean.** `three_bessel_integrals._phase_group(...)` returns a `_PhaseGroup` carrying
    \(K\), \(C\), \(C_{\rm reduced}\) (formed once, by `math.fsum`, from the supplied \(k,q,s\) and
    the constituents' `c_nu`/`c_nu_reduced`) plus `residual`, `theta`, `sin_cos`, `theta_mod_2pi`,
    `theta_deriv`, `theta_abserr` and `levin_theta()`. `levin_theta()` supplies **all four** keys,
    so every `adaptive_levin_sincos` call in that module now declares its phase error — the sum,
    linearly, of the three constituents' `theta_abserr_at` at their own arguments. Two constraints
    on anyone editing it: (i) the per-point sums use ordinary addition on purpose, because
    `levin_quadrature._detect_vectorized` only accepts an array-sampling path whose result is
    *bit-identical* to the scalar one, and `math.fsum` cannot be applied to arrays — a scalar-only
    `fsum` silently costs the array sampling of \(\theta'\) (measured 13.5 ms against 71.8 ms for
    200 samplings of a 13-point grid); (ii) the group derivative is
    \(Kx+\sum_i\epsilon_i\,dr_i/d\log x\), the *only* place in the campaign that uses
    `residual_log_deriv` for a value — which is consistent with standing note 17, whose warning is
    about using it as a refinement criterion. Measured, against 60-digit `mpmath` at the exact
    products: \(\lvert\delta\Theta\rvert\) 2.9e-5 → 1.4e-13 at exact resonance and 7.6e-5 →
    1.4e-13 at \(K/\max(k,q,s)\sim10^{-10}\); \(\lvert\delta\,d\Theta/d\log x\rvert\) 3.1e-5 →
    1.0e-12; and **no change at all** for a non-resonant group, where both routes sit on one ulp of
    \(Kx\) (issue `[07-generic-K-product-rounding]`).
