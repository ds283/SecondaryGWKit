# Implementation state — source remediation campaign

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/spec-code-audit-2026-09.md`](../../docs/spec-code-audit-2026-09.md)
**Baseline commit:** `e9a43a2` (`main`, clean)
**Last updated:** 2026-09-09 — prompt 10 complete; the pipeline is constructible again (untested end to end).

> **Maintenance rule.** Every prompt updates this file *in its own commit*, before committing.
> Set your row's status, fill in the commit SHA, model and log link, update the item-level table,
> and add or clear entries in §3 (Active issues). Do not edit rows other than your own except to
> close an issue you resolved.

---

## 1. Status board

Legend: ⬜ not started · 🟡 in flight · ✅ complete · ⚠️ complete with deviations · ⛔ blocked

### Workstream A — independent correctness and hygiene fixes

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 01 | [GenericEOS sound speed](01-genericeos-sound-speed.md) | A1 | Opus | ⚠️ | *"Exclude Lambda from the GenericEOS perturbation sound speed"* (SHA not embedded, see log deviation 4) | [`logs/01-genericeos-sound-speed.md`](logs/01-genericeos-sound-speed.md) |
| 02 | [WKB value hygiene](02-wkb-value-hygiene.md) | B1, B2, B3, B4, A6, B9, B10 | Sonnet | ✅ | *"Fix WKB value, policy and label hygiene slips"* | [`logs/02-wkb-value-hygiene.md`](logs/02-wkb-value-hygiene.md) |
| 03 | [Background derivative ends](03-background-derivative-ends.md) | A7 | Opus | ⚠️ | *"Remove the grid-end bias in the background derivative splines"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/03-background-derivative-ends.md`](logs/03-background-derivative-ends.md) |

### Workstream D — scheduling

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 04 | [Triangle filter](04-triangle-filter.md) | A5 | Sonnet | ✅ | *"Filter QuadSourceIntegral work items to triangle-closing triples"* (serialised onto Workstream C after prompt 09) | [`logs/04-triangle-filter.md`](logs/04-triangle-filter.md) |

### Workstream B — transfer-function LG representation and the source grid

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 05 | [`TkSourceFunctions`](05-tk-source-functions.md) | A2 (1/3) | Opus | ⚠️ | *"Add a two-region LG representation of T_k for source consumers"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/05-tk-source-functions.md`](logs/05-tk-source-functions.md) |
| 06 | [`QuadSource` regions](06-quadsource-regions.md) | A3, A2 (2/3) | Opus | ⚠️ | *"Restrict QuadSource to the region where both T_k are numeric"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/06-quadsource-regions.md`](logs/06-quadsource-regions.md) |

### Workstream C — the source time integral

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 07 | [Phase-group algebra](07-phase-group-algebra.md) | A4 (1/3) | Fable | ⚠️ | *"Add the phase-group decomposition of the source integrand"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/07-phase-group-algebra.md`](logs/07-phase-group-algebra.md) |
| 08 | [`QuadSourceIntegral` phase-group integration](08-qsi-phase-group-integration.md) | A4 (2/3), A2 (3/3) | Fable | ⚠️ | *"Partition the source time integral and Levin-integrate its phase groups"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/08-qsi-phase-group-integration.md`](logs/08-qsi-phase-group-integration.md) |
| 09 | [Errors, schema, tolerances](09-qsi-errors-schema-tolerances.md) | B5, B6, B7, B8, B11 | Opus | ⚠️ | *"Record b, an error bound and honest tolerances on QuadSourceIntegral"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/09-qsi-errors-schema-tolerances.md`](logs/09-qsi-errors-schema-tolerances.md) |
| 10 | [`main.py` plumbing](10-qsi-main-plumbing.md) | A4 (3/3) | Opus | ⚠️ | *"Supply the transfer functions to the source integral stage"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/10-qsi-main-plumbing.md`](logs/10-qsi-main-plumbing.md) |

### Workstream E — close-out

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Spec annotations](11-spec-annotations.md) | audit §6 | Sonnet | ⬜ | | |
| 12 | [Verification](12-verification.md) | audit §4; campaign | Opus | ⬜ | | |

**Progress:** 10 / 12 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| A1 | **DEFECT, physics** | `LambdaCDM_GenericEOS.wPerturbations` divides by the total density incl. $\rho_\Lambda$ | 01 | ✅ |
| A2 | **DEFECT, representation** | `QuadSource` splines the oscillating source; unusable beyond ~95 cycles | 05, 06, 08 | ✅ (3/3: `QuadSourceIntegral` reads the $f$ spline only on the both-numeric region and assembles the oscillatory region from `TkSourceFunctions` via `phase_groups`; nothing splines an oscillation any more) |
| A3 | **DEFECT, regression** | `compute_quad_source` walks the full grid against a both-ends-truncated $T_k$ grid → `IndexError` | 06 | ✅ |
| A4 | **DEFECT, known** | Levin call receives only $\theta_G$; no $T_q,T_r$ input to the Levin decision | 07, 08, 10 | ✅ (3/3: `main.py`'s QuadSourceIntegral stage looks up `TkNumericIntegration`/`TkWKBIntegration` for every $q$ and $r$ once per batch and ships them as `Tq_numeric`/`Tq_WKB`/`Tr_numeric`/`Tr_WKB`, so the Levin decision sees the composed phase. No `QuadSourcePolicy.Levin_threshold` was reinstated — the user accepted prompt 08 §6's cost; see `[10-levin-wholesale-cc-fallback]`. **Not yet exercised end to end** — prompt 12) |
| A5 | **DEFECT, known** | 92 % of scheduled $(k,q,r)$ triples are not triangles | 04 | ✅ |
| A6 | **DEFECT, policy** | `"WKB_minimal"` tests `numeric_clearance` | 02 | ✅ |
| A7 | **DEFECT, accuracy** | `_build_derivative` end bias (ε″ 30 % at the $z=0.1$ end for GenericEOS models) | 03 | ✅ |
| B1 | diagnostic | `TkWKBValue.analytic_*_w` return `_rad` | 02 | ✅ |
| B2 | diagnostic | `GkWKBValue.analytic_*_w` return `_rad` | 02 | ✅ |
| B3 | dead code | pre-flight WKB warnings omit `fabs` | 02 | ✅ |
| B4 | wrong exception | `_init_efolds_suph` typo (Tk and Gk WKB) | 02 | ✅ |
| B5 | tolerance | `Y3` Levin call uses module constants, not passed tolerances | 09 | ✅ (all eight analytic Levin calls take the caller's `atol`/`rtol`; `LEVIN_ABSERR`/`LEVIN_RELERR` deleted — no other user. No numerical change at the shipped tolerances, where the retired `LEVIN_RELERR` *was* `DEFAULT_QUADRATURE_RTOL`) |
| B6 | tolerance | `analytic_integral` ignores its `atol`/`rtol` | 09 | ✅ (both `_three_bessel_integrals` calls forwarded; `analytic_rad` bit-identical on all 18 exact fixtures at `atol` 1e-21 → 1e-25, worst time ratio 1.05, so `atol_serial`/`rtol_serial` now describe `analytic_rad` for free) |
| B7 | provenance | no `b` column on `QuadSourceIntegral` | 09 | ✅ (non-null `b` column, property, carried by the task result; plus a numerical guard that the supplied Bessel phase splines were built at that `b` — `bessel_phase()` records no order, so the check compares its own `bessel_j` against `scipy.jv` against the local envelope) |
| B8 | error bound | `total` has no error bound | 09 | ✅ (`total_abserr` = linear sum of every sub-interval's absolute error, plus `total_converged`/`total_phase_limited`; **a quadrature bound only** — see `[09-abserr-is-a-quadrature-bound]`) |
| B9 | consistency | `Levin_z` θ-spline chunking differs from the evaluated spline | 02 (evaluate) | ✅ (left as-is, commented) |
| B10 | cosmetic | `QuadSource` spline wrapper labelled `"T_k"` | 02 | ✅ |
| B11 | robustness | region-nonempty guards use a ratio in $z$ not $1+z$ | 09 | ✅ (confirmed: the guard is `MIN_SUBINTERVAL_LOG_WIDTH` = 1e-7 in $\log(1+z')$, its value now justified in the comment; merged hand-overs are recorded in `metadata["partition"]["skipped"]`; tested at $z_{\rm resp}=0$ exactly, where the retired ratio guard divided by zero) |
| §4.1 | UNVERIFIED | continuity of $G$ at `crossover_z` | 12 | ⬜ |
| §4.2 | UNVERIFIED | reachability of A6 | 12 | ⬜ |
| §4.3 | UNVERIFIED | whether `has_WKB_violation` modes should be rejected | 12 (measure only) | ⬜ |
| §4.4 | UNVERIFIED | A7 end bias on a real GenericEOS run | 12 | ⬜ |
| §6 | spec edits | close spec 02 Q9, record spec 04 Q7 convention, annotate spec 01 R16/Q4 | 11 | ⬜ |

**Out of scope (do not schedule):** Tier 2 LG output from `QuadSourceIntegral`; `OneLoopIntegral`;
`csSquared(z)`; `z_response` averaging. See `README.md` §1.1.

---

## 3. Active and unresolved issues

- **[01-genericeos-tz-spline-floor]** *(opened by prompt 01, 2026-09-08)* — every quantity a
  `LambdaCDM_GenericEOS`/QCD model returns inherits the interpolation error of the 500-point
  `T(z)` spline built in `_build_T_z_spline`. Measured against the exact closed form for a
  pure-radiation EOS: ~1.3e-9 relative at `max_z = 1e4`, ~6.6e-9 at `1e6`, ~6.4e-7 at the class
  default `max_z = 1e20` (the error scales as `h^4` in the `ln(1+z)` grid spacing, amplified
  fourfold by `rho_r ∝ T^4`). This is why prompt 01's regression test asserts 1e-8 rather than
  the 1e-10 its prompt requested. **Impact:** any later test or verification that compares a
  GenericEOS/QCD model against a closed form — prompt 12 in particular — must not set a
  tolerance below this, nor read a residual of that size as a physics defect. Prompt 05's tests
  use a constant-$w$ stand-in, so they are unaffected. **Next step:** nothing is required for
  this campaign. If tighter agreement is ever wanted, `_build_T_z_spline`'s `samples=500` would
  have to become tunable, or `max_z` reduced (it cannot go below ~3500: the constructor solves
  for matter–radiation equality at $z=3403$ and the spline must cover it).

- **[03-derivative-pad-clamp-on-coarse-grids]** *(opened by prompt 03, 2026-09-08)* — the padded
  fit grid `compute_background` now uses for spline-derived background derivatives clamps its
  low-end padding so that `1+z` never drops below `0.9*(1+z_min)`, because the `GenericEOS`/QCD
  models can only be evaluated down to `z ≈ -0.19` (their `T(z)` spline range). On `main.py`'s grid
  (100 samples per decade) the clamp does not bind and every end-point relative error is within
  2.4x the interior median. On a grid of 50 samples per decade it does bind, the effective padding
  falls to ~2.5 grid spacings, and the end/interior ratios rise to 329 (ε) and 73 (w″) — still 3-7
  orders of magnitude better than before the fix in absolute terms (ε″ 6.0e-08 at the z=0.1 end
  against 3.0e-01 before). **Impact:** anyone who reduces `source_samples_log10z` below ~100, and
  prompt 12 if it verifies A7 on a coarser grid than the production one. **Next step:** nothing is
  required at the shipped resolution. If a coarser grid is ever wanted, either raise
  `DERIVATIVE_FIT_REFINE` (the clamp is on the *padding* only, so refinement is unaffected) or
  extend the GenericEOS `T(z)` spline below `DEFAULT_MIN_TEMPERATURE_Z_REDSHIFT = -0.2` and relax
  `DERIVATIVE_FIT_PAD_FLOOR`.

- **[05-numeric-region-is-now-the-accuracy-floor]** *(opened by prompt 05, 2026-09-08; interpretation partially superseded by `docs/lg-phase-and-handover-followup-2026-09.md` — the residual is mostly a spline end-interval effect at the hand-over, and the LG phase-spline error grows linearly with $x$)* — with
  the new `TkSourceFunctions`, the Liouville-Green branch of $T_k$ reproduces an exact
  oscillation to 6.1e-06 of the local envelope on `main.py`'s 100-per-log10(1+z) grid (all of it
  phase-spline fit error, falling as $\Delta^4$ to 3.0e-08 at 300/decade, and bounded by the
  ~125-cycle chunk range rather than by the total cycle count). The *numeric* branch on the same
  grid gives 7.4e-06 in $T$ and 2.7e-04 in $dT/dz$ between grid points at 3.5 e-folds
  sub-horizon — one to two orders worse, and unimprovable without a denser grid, since the
  numeric branch is by construction a spline through stored samples. **Impact:** prompt 06 (the
  numeric region should be kept as short as LG validity allows, because its lower end is its
  worst point), prompt 08's error budget, and prompt 12's tolerances. Also: comparing the LG
  branch against `scipy.special.jv` rather than against `bessel_phase`'s own $m\sin\vartheta$
  saturates at 2.0e-06 under refinement — that is `bessel_phase`'s phase-function error, out of
  scope here (README §5 item 8). **Next step:** nothing required at the shipped resolution;
  raising `source_samples_per_log10z` is the only lever, and it costs the numeric branch alone.

- **[06-source-spline-residual-vs-handover]** *(opened by prompt 06, 2026-09-08)* — with the
  grid now truncated at the both-numeric hand-over, the spline of $f$ inside that region is
  accurate to **4.5e-04 of the local oscillation envelope** at the realistic hand-over, and
  degrades to $O(1)$ if the hand-over is allowed to fall to the bottom of `main.py`'s search
  window. Measured on the exact analytic radiation source, $q=r=k$, 100 samples per
  log10 z, region from 5 e-folds super-horizon down to the stated hand-over, error
  normalised to the local envelope (the `QS_03_spline_error.py` measure). In radiation with
  $a_0$ absorbed, $x = k c_s a_0\eta = e^N/\sqrt3$ at $N$ e-folds inside the horizon,
  independently of $k$:

  | hand-over | $x$ | cycles of $f$ | max err / envelope |
  |---|---|---|---|
  | `z_exit_subh_e3` (where `find_phase_minimum` normally stops) | 11.6 | 3.7 | **4.493e-04** |
  | `z_exit_subh_e6` (window bottom) | 232.9 | 74.1 | 1.220e+00 |
  | `0.85 · z_exit_subh_e6` (latest possible) | 272.1 | 86.6 | 1.129e+00 |

  $f$ is quadratic in $T$ and so oscillates at *twice* the transfer-function phase, halving
  the nodes per half-cycle; that is why this is one to two orders worse than either
  `TkSourceFunctions` branch (6.1e-06 LG, 7.4e-06 numeric — issue
  `[05-numeric-region-is-now-the-accuracy-floor]`). **Impact:** the all-smooth region, not the
  oscillatory one, is now the dominant term in prompt 08's error budget; do not set a
  tolerance there below ~1e-3, and prompt 12 should not read a residual of this size as a
  physics defect. **Next step:** nothing at the shipped settings. If tighter agreement is ever
  wanted the knobs are the `mode="stop"` search window
  (`TkNumericIntegration.py:130-131`, `z_exit_subh_e3`/`z_exit_subh_e6`) — a hand-over closer
  to horizon crossing shortens the oscillatory part of the smooth region — or
  `source_samples_per_log10z`. Neither was touched here.

- **[07-lg-derivative-truncation-at-handover]** *(opened by prompt 07, 2026-09-08)* — the
  derivative pieces `TkSourceFunctions` supplies in closed form, `omega = sqrt(Tk_omegaEff_sq)`
  and the R23/R24 `dlnM_dz`, are the Liouville–Green frequency and amplitude derivative, and
  differ from the exact $d\theta/dz$ and $d\ln M/dz$ by the LG truncation, $O(x^{-4})$ relative
  (log 05 deviation 5 measured 8.5e-6 at $w=1/3$, 7.0e-5 at $w=0.2$ in $d\ln M/dz$ at the
  hand-over). Because $f$ is dominated by $DT_qDT_r$ sub-horizon, this is the accuracy floor of
  the *oscillatory* integrand $G f/H^2$ near the hand-over even when $M$ and $\theta$ are exact:
  **7.0e-6 ($w=1/3$) and 1.4e-4 ($w=0.2$) of the envelope** at $x\approx20$ on the production
  grid, falling to 3e-6 / 6.5e-6 for $x>100$, and grid-independent (6.3e-6 / 1.1e-4 at
  300/decade). Replacing only `omega` and `dlnM_dz` by exact values drops it to 1.5e-6, which is
  `bessel_phase`'s own floor. With the production-matched LG representation (amplitude and phase
  also LG-truncated) the numeric↔LG seam mismatch of one factor is 4e-5 / 1e-3 at grid nodes.
  **Impact:** prompt 08 — an exact-fixture comparison in the oscillatory regions cannot be
  asserted below ~5e-4, and a region-boundary consistency check at the hand-over should allow
  ~2e-3; prompt 12 — the same numbers are not physics defects. **Next step:** nothing for this
  campaign. The floor falls as $x^{-4}$, so a hand-over deeper inside the horizon (the
  `mode="stop"` search window, `TkNumericIntegration.py:130-131`) or the overlap of
  `docs/lg-phase-and-handover-followup-2026-09.md` §1.4 would reduce it; both are out of scope
  here.

- **[07-phase-spline-chunking-precision]** *(opened by prompt 07, 2026-09-08)* — the rationale
  for composing phases as a signed sum of `theta_mod_2pi` remainders rather than reducing the sum
  of `raw_theta` (reconciliation document §3.1) holds only within `phase_spline`'s first two
  chunks. `chunk_logstep=125` is geometric in the cycle count (`phase_spline.py:460-476`: chunk
  boundaries at 0, 126, 11751, 1.1e6, … cycles), so above ~7e4 rad the rebased spline values are
  as large as the raw phase and both routes round identically: measured 3.8e-8 rad at
  $|\Psi|=2.2\times10^6$ rad on exact quadratic data, against 1.8e-15 rad for the remainder route
  when the constituents' remainders are exact. **Impact:** none practical — 4e-8 rad is far below
  the phase-spline *fit* error the followup document measures (§2), and `sin`/`cos` are unaffected
  at any accuracy this pipeline reaches; but prompt 08 should not cite the remainder sum as a
  precision guarantee, and the module's composition is not where any phase inaccuracy will come
  from. **Next step:** if the guarantee is ever wanted, a linear `chunk_step` (or a log step
  much smaller than 125) in `phase_spline`; `LiouvilleGreen/` is out of scope for this campaign.

- **[10-levin-wholesale-cc-fallback]** *(opened by prompt 10, 2026-09-09)* — the user's decision on
  `[08-levin-fallback-cost-ratio]` (now §4) accepts prompt 08 §6's cost here and asks for the fix
  in the Levin driver instead: *"make the Levin integrator more intelligent about choosing a
  fallback, so that it routes the integrand [to] the Clenshaw–Curtis wholesale at the outset,
  rather than finding that adaptive Levin needs many bisections which end up in Clenshaw–Curtis
  anyway."* The measurement supports this reading — 2.65–3.56× the wall-clock for only 1.00–1.44×
  the integrand evaluations, i.e. per-region overhead across the 20–28 Clenshaw–Curtis regions the
  driver's own total-variation gate produced, not wasted integrand work. The lever is
  `AdaptiveLevin/levin_quadrature.py` (log 08 observation 3: a coarser first bisection, or plain
  adaptive quadrature when the *whole* call's phase span is below a few $2\pi$), which README §1.1
  and §5 item 8 place outside this campaign. **Impact:** a wall-clock factor ≲3.6× on the weakly
  oscillatory sub-intervals only (a few cycles of $\theta_G$ with both $T$ smooth); no accuracy or
  stored-number consequence, so it does not block prompt 12 — but prompt 12's timings will include
  it. **Next step:** a campaign allowed to edit `AdaptiveLevin/`. Nothing in
  `QuadSourceIntegral`/`QuadSourcePolicy` should be changed for it: a threshold there would
  re-decide from $\theta_G$ alone what the driver decides from the composed phase, which is defect
  A4 in weaker form.

- **[10-classify-levin-keyerror]** *(opened by prompt 10, 2026-09-09)* — latent, pre-existing, and
  noticed while adding prompt 10 §4's comment: `GkSourcePolicyData._classify_Levin` (`:135-199`)
  sets `payload["Levin_z"]` only inside its `for z_source` loop, so a Green's function whose
  $|d\theta_G/d\log(1+z)|$ never exceeds `policy.Levin_threshold` anywhere in its WKB range makes
  `apply_GkSource_policy:58` raise `KeyError: 'Levin_z'` instead of storing `None` (the
  early-return path at `:154` does supply `None`). Not fixed: prompt 10 may only comment on that
  file, and no row was available to test reachability. **Impact:** the `--gk-source-policy-queue`
  stage would fail loudly, not silently, if such a mode exists; prompt 12 should recognise the
  exception if it appears. **Next step:** a one-line `payload.setdefault("Levin_z", None)` in a
  prompt that is allowed to edit `GkSourcePolicyData.py`.

- **[08-handover-clamp-error]** *(opened by prompt 08, 2026-09-09)* — in production the WKB grid
  of each $T_k$ starts at the largest source-grid point *below* `z_init` (`main.py:695-697`;
  `TkWKBIntegration` stores no sample at `z_init`), so `TkSourceFunctions.WKB_region[0]` sits up
  to one grid step below `crossover_z` and no accessor of that factor is evaluable in between
  (§5 note 6). `QuadSourceIntegral` partitions at `crossover_z` and, as note 6 prescribes, clamps
  the LG accessors to `WKB_region[0]` across the gap (allowed up to
  `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` mean grid steps; the gap is recorded in
  `metadata["partition"]`). Measured on the production-shaped fixture
  (`Fixture(drop_first_WKB_sample=True)`, $b=0$, $x_{\rm resp}=100$): a one-step gap
  (2.3e-2 in $\log(1+z)$) changes `total` by **5.1e-3** — at $x\approx19$ the held phase is
  wrong by up to 0.44 rad over the gap. The error scales as gap$^2$, so a uniformly distributed
  gap averages ~1/3 of this. **This is now the largest single error term in the chain**, an order
  of magnitude above the spline and LG floors (`[06-source-spline-residual-vs-handover]`,
  `[07-lg-derivative-truncation-at-handover]`). **Impact:** every production
  `QuadSourceIntegral`; prompt 12 must expect residuals of this size against `analytic_rad`
  and should report the recorded `clamp_gaps_log1pz`. **Next step:** the user's choice among
  (a) an LG sample at `z_init` (main.py / TkWKBIntegration — but `z_init` is not a stored
  `redshift`), (b) the overlap of `docs/lg-phase-and-handover-followup-2026-09.md` §1.4, or
  (c) a first-order Taylor extension of the LG phase and amplitude across the gap using the
  closed-form `omega`/`dlnM_dz` inside `QuadSourceIntegral`'s clamp adapter (estimated ~100×
  smaller error; log 08 deviation 2). None is in prompts 09–10's scope.

- **[09-abserr-is-a-quadrature-bound]** *(opened by prompt 09, 2026-09-09)* — the new
  `total_abserr` column is the linear sum of the sub-intervals' quadrature error estimates
  (scipy's, and the Levin driver's), and that is **all** it is: it measures 1.3e-11 to 1.5e-7 of
  `|total|` on the 36 acceptance fixtures, while `|total − analytic_rad|` is 1e0–1.6e4 times
  larger with exact ingredients (there the limit is `analytic_rad`'s own phase/modulus spline
  floor, audit QI-1/QI-11) and up to 4.4e4 times larger with realistic ones (the QuadSource
  spline of $f$, the LG closed forms, and above all the hand-over clamp of
  `[08-handover-clamp-error]`). The bound *is* correct for what it claims: re-running each exact
  case at `rtol = 1e-11` moves `total` by at most 0.65 of the two runs' summed bounds.
  Relatedly, **`total_converged` is `False` on 14 of the 36 fixtures** — all realistic — because
  the Levin driver cannot reach `rtol = 1e-8` against a re-splined phase; none is phase-limited.
  **Impact:** prompt 12 must add the board's representation floors to `total_abserr` before
  judging any residual, and must not read `total_converged = False` as a failure or a physics
  defect; anyone querying the column should know it excludes representation error.
  **Next step:** nothing for this campaign. A meaningful *total* error would need the
  representation error propagated (the hand-over clamp first — `[08-handover-clamp-error]`
  option (c) is the cheap one), which is upstream of this module.

- **[09-WKB_quad-columns-are-vestigial]** *(opened by prompt 09, 2026-09-09)* — `WKB_quad` and
  its six `WKB_quad_*` timing columns have been identically `0.0`/`None` since prompt 08 retired
  direct quadrature of an oscillatory Green's function. Prompt 09 §3 asked for them to be dropped
  "unless you find a reader": there is one, `extract_QuadSourceIntegral_data.py:181,278,293`
  (`obj.WKB_quad`, feeding the `WKB numeric: [z, z]` annotation of `extract_common.py:327-345`
  and a column of the exported table), and that script is out of scope (README §5 item 8). The
  columns and the `WKB_quad`/`WKB_quad_data` properties therefore stay. `tools/` and
  `useful_queries.sql` reference neither. **Impact:** the extract script still runs but its
  `WKB numeric` annotation will never be drawn and its `WKB_quad` column will be all zeros;
  seven columns of every row are dead weight. **Next step:** a later campaign that is allowed to
  edit `extract_*.py` should delete the reader and then the columns, or repurpose them.

> Add an entry here whenever a prompt finishes with something unresolved: a verification step that
> could not be run, an assumption that could not be confirmed, a deviation that a later prompt has
> to work around, a measured cost that changes a later prompt's decision. Format:
>
> - **[NN-shortname]** *(opened by prompt NN, YYYY-MM-DD)* — description. **Impact:** who is
>   affected. **Next step:** what would close it.
>
> Move closed entries to §4 rather than deleting them.

---

## 4. Resolved issues

- **[08-pipeline-non-runnable-until-10]** *(opened by prompt 08, 2026-09-09)* —
  `QuadSourceIntegral.compute()` requires the payload keys `Tq_numeric`, `Tq_WKB`,
  `Tr_numeric`, `Tr_WKB` (the `TkNumericIntegration`/`TkWKBIntegration` objects for $q$ and $r$)
  and raises `RuntimeError` naming any that are missing, while `main.py` supplied only
  `GkPolicy`, `source`, `b`, `Bessel_0pt5`, `Bessel_2pt5`. **Impact:** the
  `--quad-source-integral-queue` stage could not start between prompts 08 and 10. This replaced
  `[06-qsi-blocked-until-08]` below: the stage was already non-runnable, for a different reason,
  since prompt 06.
  **Resolved by prompt 10 (2026-09-09):** the QuadSourceIntegral stage of `main.py` now looks up
  both transfer-function objects for every distinct $q$ and $r$ of the batch (two new
  `RayWorkPool` queues, fully populated — no `_do_not_populate`), caches them by
  `wavenumber_exit_time.store_id` and assembles the nine-key payload in the new module-level
  `build_QuadSourceIntegral_payload`. **The path has not been executed**: no Ray cluster or
  datastore was available, so prompt 12's live run is its first exercise.

- **[08-levin-fallback-cost-ratio]** *(opened by prompt 08, 2026-09-09)* — prompt 08 §6 measured
  the case the retired `LEVIN_MIN_2PI_CYCLES = 10` gate used to send to direct quadrature ("$G$
  oscillatory, both $T$ smooth", $\theta_G$ turning over 3.5–5.4 cycles, realistic fixtures,
  `rtol = 1e-8`): the new single-group `adaptive_levin_sincos` call (total-variation gate →
  Clenshaw–Curtis on every one of its 20–28 regions) is **2.65–3.56× slower in wall-clock** than
  the old `WKB_quad_integral`, above the README §4.1 / prompt 08 §6 trigger of 3× on the three
  $b=0$ rows (3.31, 3.51, 3.56; the $b=0.2$ rows give 3.19, 2.82, 2.65), while doing only
  **1.00–1.44× the integrand evaluations**. Accuracy is the same (both at the fixture floor,
  2e-5–1.6e-4). Full table: log 08 deviation 10.
  **Resolved by the user's decision, executed in prompt 10 (2026-09-09):** accept §6 as it stands.
  No threshold is reinstated — `QuadSourcePolicy` stays persisted and threaded but read by nothing
  (its new class docstring says why), and prompt 08's dead `LEVIN_MIN_2PI_CYCLES`/
  `LEVIN_MIN_PHASE_DIFF` are deleted. The requested remedy — a driver that routes a weakly
  oscillatory integrand to Clenshaw–Curtis wholesale at the outset — is
  `[10-levin-wholesale-cc-fallback]` in §3, out of scope here.

- **[06-qsi-blocked-until-08]** *(opened by prompt 06, 2026-09-08)* — all three regions of
  `compute_QuadSource_integral` read `source_f.source(log_z_source, z_is_log=True)`
  (`QuadSourceIntegral.py:958, 1028, 1102`), and `ZSplineWrapper` raises `RuntimeError` more
  than 1 % (in $\log(1+z)$) below its `min_z` (`spline_wrappers.py:50-53`). Before prompt 06
  that call returned a meaningless spline value below the hand-over; it now raises. **Impact:**
  between prompt 06 and prompt 08 the `--quad-source-integral-queue` stage cannot complete for
  any $(k,q,r,z_{\rm resp})$ whose integration range reaches below
  `QuadSource.numeric_region[1]` — i.e. essentially all of them at production settings. This is
  README §4's stopping-point note for 06 made concrete, and it fails loudly rather than storing
  a wrong number. `QuadSourceIntegral.py:1424` checks only `source.z_sample.max.z`, which the
  truncation does not change, so the ingredient compatibility check still passes.
  **Next step:** prompts 07 and 08, which assemble the region below the hand-over from
  `TkSourceFunctions` instead of from a sampled $f$. Nothing else closes it.
  **Resolved by prompt 08 (2026-09-09):** `compute_QuadSource_integral` now reads the
  `QuadSource` spline only on `source.numeric_region` (clamped, see
  `[08-handover-clamp-error]`) and assembles everything below the hand-over from
  `TkSourceFunctions` through `phase_groups`. The stage is still non-runnable, for the payload
  reason recorded in `[08-pipeline-non-runnable-until-10]`.

---

## 5. Standing notes for implementers

1. **Datastores holding `GenericEOS`/QCD models are stale after prompt 01.** The fix to
   `LambdaCDM_GenericEOS.wPerturbations` changes $c_s^2$ (by a factor 3.21 at $z=0$, 1.28 at
   $z=1$, $<1\%$ for $z\gtrsim5$) and therefore the stored `BackgroundModel` columns
   `wPerturbations`, `d_wPerturbations_dz`, `d2_wPerturbations_dz2`, and everything computed from
   them — `TkNumericIntegration`, `TkWKBIntegration` ($\omega_{\rm eff}^2$, the LG friction) and
   all downstream rows. No migration is possible or attempted: those rows must be rebuilt. Plain
   `LambdaCDM` datastores are unaffected; that class was always correct.
2. **Datastores built before this campaign are stale after prompt 06** (`QuadSource` rows have
   fewer redshifts per pair) **and unreadable after prompt 09** — the `QuadSourceIntegral` table
   gained four columns in prompt 09: `b` (Float, **non-null**), `total_abserr` (Float, nullable),
   `total_converged` and `total_phase_limited` (Boolean, nullable). No migration is attempted;
   the table must be rebuilt, which prompt 08's change of what `total` *means* already required.
   `WKB_quad` and its six `WKB_quad_*` timing columns were kept
   (`[09-WKB_quad-columns-are-vestigial]`). The verification prompt rebuilds from scratch; do not try to migrate. The
   `QuadSource`/`QuadSourceValue` **schema is unchanged** by prompt 06, so old rows are still
   *readable* — which is the hazard: a pre-06 row carries a value per source redshift, so
   `QuadSource.numeric_region` reconstructed from it extends below the both-numeric hand-over
   and the dense-output spline will happily be evaluated in the oscillatory region it cannot
   represent. Prune or re-tag pre-06 `QuadSource` rows rather than reusing them.
3. **Existing `BackgroundModel` rows built with a `GenericEOS`/QCD cosmology are stale after
   prompt 03** in the outermost few redshifts of `d_lnH_dz`, `d2_lnH_dz2`, `d3_lnH_dz3`,
   `d_wPerturbations_dz` and `d2_wPerturbations_dz2` (the change is ≤3e-08 relative and confined to
   the grid ends). Rows built with plain `LambdaCDM` are bit-identical; that class supplies analytic
   derivatives and its branch was not touched. Those rows are being rebuilt for prompt 01 anyway.
4. **`b_value = 0.0` is hardwired in `main.py:419`.** Nothing in this campaign changes that, but
   prompt 09 makes `b` persisted so that a future non-zero run is distinguishable.
5. **The phase sign convention is $d\theta/dz = +\omega_{\rm eff}$ integrated towards smaller
   $z$, so every stored phase is negative and decreasing** (audit §3.2). Composed phases
   $\theta_G\pm\theta_q\pm\theta_r$ inherit this. Nothing depends on the sign, but tests that
   assert monotonicity must assert the right direction.
6. **`TkSourceFunctions` region boundaries are not a single point.** `WKB_region[0]` can sit up
   to one grid step *below* `crossover_z`, because `main.py:695-697` truncates the WKB grid to
   the largest grid point at or below `z_init` and `phase_spline` cannot be extrapolated.
   Partition on `crossover_z`, but clamp quadrature nodes to `numeric_region`/`WKB_region`; every
   accessor raises outside its own range. See `logs/05-tk-source-functions.md` deviation 3.
7. **`QuadSource` is now only the *smooth part* of the source term.** After prompt 06 its
   stored values, its `z_sample` and its dense-output spline all cover exactly
   `numeric_region = (z_source_sample.max.z, max(crossover_z_q, crossover_z_r, Tq_z_min, Tr_z_min))`
   — the region where both $T_q$ and $T_r$ are still numeric, plus the exactly-known
   super-horizon region above it. Read the region from `QuadSource.numeric_region` (or
   `QuadSourceFunctions.numeric_region`; that namedtuple is now `("source", "numeric_region")`),
   which survives a datastore round-trip. The individual hand-over redshifts
   `crossover_z_q`/`crossover_z_r` are **not** persisted and come back `None`, so recompute them
   with `ComputeTargets.QuadSource.numeric_crossover_z(Tk)` = `k_exit.z_exit - Tk.stop_deltaz_subh`.
   `numeric_region[1]` can sit up to one grid step *above* `max(crossover_z_q, crossover_z_r)`;
   trust the region, not the crossovers. See `logs/06-quadsource-regions.md` deviations 1-3.
8. **`ComputeTargets.phase_groups` is the only place the phase-group algebra lives.** Its
   `build_phase_groups(regime, *, Gk, Tq, Tr, model_functions, w_background)` returns 1, 2 or 4
   `PhaseGroup`s whose callables all take $\log(1+z')$, include $1/H^2$ but **not**
   $(1+z_{\rm resp})$, and are scalar-only; a smooth $T$ is the same `TkSourceFunctions` object
   with the regime flag `False`, a smooth $G$ is `Gk_f.numeric_Gk`. `g.levin_theta()` is the
   `theta` dict for `adaptive_levin_sincos`; `evaluate_sum`/`evaluate_envelope` are for
   consistency checks only. Do not re-derive or re-type the coefficients anywhere else — the
   sympy script `ComputeTargets/tests/sympy_phase_groups.py` verifies this module's code path and
   would not see a copy. See `logs/07-phase-group-algebra.md` "State handed to the next prompt".
9. **`evaluate_QuadSource_integral` is the testable entry point of the source time integral**
   (`ComputeTargets/QuadSourceIntegral.py`); the `@ray.remote` `compute_QuadSource_integral`
   only resolves the proxies and calls it. `QuadSourceIntegral.compute()` requires the payload
   keys in `QuadSourceIntegral.REQUIRED_PAYLOAD_KEYS` (the five old ones plus `Tq_numeric`,
   `Tq_WKB`, `Tr_numeric`, `Tr_WKB`). `WKB_quad` is identically `0.0`; the per-part error
   estimates are in `metadata["numeric_quad"]["abserr"]` and `metadata["WKB_Levin"]["abserr"]`,
   the region record in `metadata["partition"]`. `numeric_quad` and `WKB_Levin` cancel by up to
   5× on the test fixtures, so bound `total`'s error by summing the parts' absolute errors, never
   by scaling a relative one. See `logs/08-qsi-phase-group-integration.md` "State handed to the
   next prompt".
10. **Prompt 09 added four fields to the task's return dict and to the object**: `b`,
    `total_abserr` (the sum of the parts' absolute errors — a *quadrature* bound, see
    `[09-abserr-is-a-quadrature-bound]`), `total_converged` and `total_phase_limited`, each with
    a matching column and property. `b` still arrives through `compute()`'s payload and is
    carried out on the result. Every tolerance in the analytic branch is now the caller's, so
    `atol_serial`/`rtol_serial` describe `analytic_rad`; `LEVIN_ABSERR`/`LEVIN_RELERR` are gone.
    `evaluate_QuadSource_integral` now also rejects Bessel phase splines that were not built at
    the payload's `b` (`_check_bessel_order`, tolerance `BESSEL_ORDER_CHECK_TOL = 1e-3` of the
    local envelope), and `metadata["partition"]` carries `skipped` and
    `min_subinterval_log_width`. See `logs/09-qsi-errors-schema-tolerances.md`.
11. **`main.py`'s QuadSourceIntegral stage assembles its payload in one module-level function**,
    `build_QuadSourceIntegral_payload(z_response, k, q, r, Gk_cache, source_cache,
    Tk_numeric_cache, Tk_WKB_cache, b_value, Bessel_0pt5_proxy, Bessel_2pt5_proxy)` (prompt 10),
    next to prompt 04's `closes_triangle`. It supplies all nine
    `REQUIRED_PAYLOAD_KEYS` and raises after the stage's `!! MISSING DATA WARNING` prints if any
    ingredient is unavailable. The stage requires **validated, fully populated**
    `TkNumericIntegration` *and* `TkWKBIntegration` rows (no `_do_not_populate`: `TkSourceFunctions`
    reads `.values`) with the six production tags for every mode appearing as $q$ or $r$ — a
    scoped run must complete both Tk stages for all of them. The payload is ~15–30 % larger than
    before and the four `Tk` objects are re-serialised per work item, which makes the object-store
    TODO at `main.py:2618-2623` a better investment than it was. `main.py` cannot be imported;
    `ComputeTargets/tests/test_main_plumbing.load_main_py_functions` extracts a named top-level
    function from it with `ast` instead. No `QuadSourcePolicy` reaches the integrator, by decision
    (§4 `[08-levin-fallback-cost-ratio]`); `GkSourcePolicyData.Levin_z` and both
    `Levin_threshold` fields are diagnostic only. See `logs/10-qsi-main-plumbing.md`.
