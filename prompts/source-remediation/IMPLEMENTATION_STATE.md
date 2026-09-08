# Implementation state — source remediation campaign

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/spec-code-audit-2026-09.md`](../../docs/spec-code-audit-2026-09.md)
**Baseline commit:** `e9a43a2` (`main`, clean)
**Last updated:** 2026-09-08 — prompt 07 complete.

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
| 04 | [Triangle filter](04-triangle-filter.md) | A5 | Sonnet | ⬜ | | |

### Workstream B — transfer-function LG representation and the source grid

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 05 | [`TkSourceFunctions`](05-tk-source-functions.md) | A2 (1/3) | Opus | ⚠️ | *"Add a two-region LG representation of T_k for source consumers"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/05-tk-source-functions.md`](logs/05-tk-source-functions.md) |
| 06 | [`QuadSource` regions](06-quadsource-regions.md) | A3, A2 (2/3) | Opus | ⚠️ | *"Restrict QuadSource to the region where both T_k are numeric"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/06-quadsource-regions.md`](logs/06-quadsource-regions.md) |

### Workstream C — the source time integral

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 07 | [Phase-group algebra](07-phase-group-algebra.md) | A4 (1/3) | Fable | ⚠️ | *"Add the phase-group decomposition of the source integrand"* (SHA not embedded, per prompt 01 log deviation 4) | [`logs/07-phase-group-algebra.md`](logs/07-phase-group-algebra.md) |
| 08 | [`QuadSourceIntegral` phase-group integration](08-qsi-phase-group-integration.md) | A4 (2/3), A2 (3/3) | Fable | ⬜ | | |
| 09 | [Errors, schema, tolerances](09-qsi-errors-schema-tolerances.md) | B5, B6, B7, B8, B11 | Opus | ⬜ | | |
| 10 | [`main.py` plumbing](10-qsi-main-plumbing.md) | A4 (3/3) | Opus | ⬜ | | |

### Workstream E — close-out

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Spec annotations](11-spec-annotations.md) | audit §6 | Sonnet | ⬜ | | |
| 12 | [Verification](12-verification.md) | audit §4; campaign | Opus | ⬜ | | |

**Progress:** 6 / 12 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| A1 | **DEFECT, physics** | `LambdaCDM_GenericEOS.wPerturbations` divides by the total density incl. $\rho_\Lambda$ | 01 | ✅ |
| A2 | **DEFECT, representation** | `QuadSource` splines the oscillating source; unusable beyond ~95 cycles | 05, 06, 08 | 🟡 (2/3: `TkSourceFunctions` shipped; `QuadSource` now splines $f$ only where both $T_k$ are numeric) |
| A3 | **DEFECT, regression** | `compute_quad_source` walks the full grid against a both-ends-truncated $T_k$ grid → `IndexError` | 06 | ✅ |
| A4 | **DEFECT, known** | Levin call receives only $\theta_G$; no $T_q,T_r$ input to the Levin decision | 07, 08, 10 | 🟡 (1/3: `phase_groups` shipped — the $\theta_G\pm\theta_q\pm\theta_r$ decomposition exists and is verified; nothing consumes it yet) |
| A5 | **DEFECT, known** | 92 % of scheduled $(k,q,r)$ triples are not triangles | 04 | ⬜ |
| A6 | **DEFECT, policy** | `"WKB_minimal"` tests `numeric_clearance` | 02 | ✅ |
| A7 | **DEFECT, accuracy** | `_build_derivative` end bias (ε″ 30 % at the $z=0.1$ end for GenericEOS models) | 03 | ✅ |
| B1 | diagnostic | `TkWKBValue.analytic_*_w` return `_rad` | 02 | ✅ |
| B2 | diagnostic | `GkWKBValue.analytic_*_w` return `_rad` | 02 | ✅ |
| B3 | dead code | pre-flight WKB warnings omit `fabs` | 02 | ✅ |
| B4 | wrong exception | `_init_efolds_suph` typo (Tk and Gk WKB) | 02 | ✅ |
| B5 | tolerance | `Y3` Levin call uses module constants, not passed tolerances | 09 | ⬜ |
| B6 | tolerance | `analytic_integral` ignores its `atol`/`rtol` | 09 | ⬜ |
| B7 | provenance | no `b` column on `QuadSourceIntegral` | 09 | ⬜ |
| B8 | error bound | `total` has no error bound | 09 | ⬜ |
| B9 | consistency | `Levin_z` θ-spline chunking differs from the evaluated spline | 02 (evaluate) | ✅ (left as-is, commented) |
| B10 | cosmetic | `QuadSource` spline wrapper labelled `"T_k"` | 02 | ✅ |
| B11 | robustness | region-nonempty guards use a ratio in $z$ not $1+z$ | 09 | ⬜ |
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

*(none yet)*

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
   fewer redshifts per pair) **and unreadable after prompt 09** (`QuadSourceIntegral` schema
   change). The verification prompt rebuilds from scratch; do not try to migrate. The
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
