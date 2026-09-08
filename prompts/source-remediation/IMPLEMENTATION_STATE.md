# Implementation state — source remediation campaign

**Campaign:** [`README.md`](README.md) · **Source audit:** [`docs/spec-code-audit-2026-09.md`](../../docs/spec-code-audit-2026-09.md)
**Baseline commit:** `e9a43a2` (`main`, clean)
**Last updated:** 2026-09-08 — prompt 05 complete.

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
| 06 | [`QuadSource` regions](06-quadsource-regions.md) | A3, A2 (2/3) | Opus | ⬜ | | |

### Workstream C — the source time integral

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 07 | [Phase-group algebra](07-phase-group-algebra.md) | A4 (1/3) | Fable | ⬜ | | |
| 08 | [`QuadSourceIntegral` phase-group integration](08-qsi-phase-group-integration.md) | A4 (2/3), A2 (3/3) | Fable | ⬜ | | |
| 09 | [Errors, schema, tolerances](09-qsi-errors-schema-tolerances.md) | B5, B6, B7, B8, B11 | Opus | ⬜ | | |
| 10 | [`main.py` plumbing](10-qsi-main-plumbing.md) | A4 (3/3) | Opus | ⬜ | | |

### Workstream E — close-out

| # | Prompt | Items | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|
| 11 | [Spec annotations](11-spec-annotations.md) | audit §6 | Sonnet | ⬜ | | |
| 12 | [Verification](12-verification.md) | audit §4; campaign | Opus | ⬜ | | |

**Progress:** 4 / 12 complete.

---

## 2. Item-level tracking

Traceability from the audit's finding IDs to the prompt that discharges them.

| ID | Severity | Description | Prompt | Status |
|---|---|---|---|---|
| A1 | **DEFECT, physics** | `LambdaCDM_GenericEOS.wPerturbations` divides by the total density incl. $\rho_\Lambda$ | 01 | ✅ |
| A2 | **DEFECT, representation** | `QuadSource` splines the oscillating source; unusable beyond ~95 cycles | 05, 06, 08 | 🟡 (1/3: `TkSourceFunctions` shipped) |
| A3 | **DEFECT, regression** | `compute_quad_source` walks the full grid against a both-ends-truncated $T_k$ grid → `IndexError` | 06 | ⬜ |
| A4 | **DEFECT, known** | Levin call receives only $\theta_G$; no $T_q,T_r$ input to the Levin decision | 07, 08, 10 | ⬜ |
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

- **[05-numeric-region-is-now-the-accuracy-floor]** *(opened by prompt 05, 2026-09-08)* — with
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
   change). The verification prompt rebuilds from scratch; do not try to migrate.
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
