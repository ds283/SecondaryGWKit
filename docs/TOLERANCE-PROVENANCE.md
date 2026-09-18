# The provenance of every accuracy parameter in the pipeline

**Campaign:** [`prompts/tolerance-convergence`](../prompts/tolerance-convergence/README.md) ·
**Prompt:** [06a](../prompts/tolerance-convergence/06a-close-out-and-provenance.md) · **Board item:**
**T12** · **Companion:** [`TOLERANCE-CONVERGENCE.md`](TOLERANCE-CONVERGENCE.md), the campaign's
narrative close-out · **Date:** 2026-09-18

This is the deliverable [`README.md`](../prompts/tolerance-convergence/README.md) §1.2 asks for: one
entry per accuracy parameter that reaches a numerical method or a datastore lookup key anywhere in
the pipeline, so that no constant is left for the next reader to change by guesswork. It covers the
parameters this campaign set, the parameters it measured and left unchanged, and the parameters it
did not touch at all — including the ones nobody has ever chosen. **Where the record contains no
choice, this note says so in those words**, rather than inventing one; a fabricated provenance is
indistinguishable from a real one to a later reader, which is what makes the honest admission the
only acceptable failure mode here.

**This is a summary with citations, not a second copy of the measurements.** Every entry points at
the campaign document or log that holds the tables; `config/defaults.py`'s own comments stay the
primary record at the point of use, to the standard `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` and
`DEFAULT_QUADRATURE_ATOL` already carry.

**Coverage.** The checklist is
[`docs/tolerance-convergence/TOLERANCE-INVENTORY.md`](tolerance-convergence/TOLERANCE-INVENTORY.md)
§5.4's **44 rows**, as prompt 06 left them on 2026-09-18 (`inventory.py --check` passes). The six
constants prompt 05a shipped are *already among* those 44 rows — they are not added again. The
sections below (A–G) are the inventory's own grouping, so a row here and a row there name the same
parameter under the same letter.

Each entry gives the five fields README §1.2 asks for: the **value** and what it keys; **what
measurement chose it**, with the reference's own drift and the source-grid generation where one
applies; **what it competes against** — the floor that would dominate if it were tightened further;
its **cost**, in evaluations at the setting chosen and one step either side, times the object count
of the sector it keys; and the **campaign, prompt, log and date** that established it. Where a field
does not apply — a parameter that reaches no solver and keys no lookup has no "cost" in this sense —
the entry says so rather than forcing an answer.

---

## A. The `config/defaults.py` accuracy constants

### `DEFAULT_ABS_TOLERANCE = 1e-10` (`config/defaults.py:23`)

- **Value, and what it keys:** `1e-10`. **Nothing**, since prompt 05a (2026-09-18). It keyed seven
  object types until prompt 05 removed the pair from the four that never used it and prompt 05a gave
  each of the three that did a constant of its own. What is left is **seven bare float
  comparisons** — `ComputeTargets/GkSource.py:96`, `:104`, `:275`;
  `Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, `:790`;
  `LiouvilleGreen/WKBtools.py:83` — plus the `atol` signature default of
  `numeric_with_phase_cut.integrate_numeric_with_phase_cut`, which every production call overrides.
- **What measurement chose it:** **cannot be established from the record.** No campaign document,
  log or code comment records a measurement behind `1e-10`; it is the value the pipeline was written
  with.
- **What it competes against:** not applicable — it is not a solver tolerance. Comparing two
  redshifts, a residual or a phase modulo is a different quantity wearing a tolerance's name
  (`config/defaults.py:14-22` says so at the point of use).
- **Cost:** 7 comparison call sites; 0 objects keyed.
- **Campaign, prompt, log, date:** unassigned. `[02-shared-atol-doubles-as-a-float-comparison-epsilon]`
  (opened by prompt 02, 2026-09-16; narrowed by prompt 05a, 2026-09-18) is open on
  `prompts/tolerance-convergence`'s own board and is the record of the fact that this constant is
  now only an epsilon and must not be retuned as a solver tolerance by anyone.

### `DEFAULT_REL_TOLERANCE = 1e-08` (`config/defaults.py:24`)

- **Value, and what it keys:** `1e-8`. **Nothing**, since prompt 05a — it keyed eight object types
  before. It survives as the `rtol` signature default of `integrate_numeric_with_phase_cut`, which
  production always overrides, and has no float-comparison use of its own.
- **What measurement chose it:** **cannot be established.** `1e-8` is the value the pipeline was
  written with. Its three former consumers now carry their own settled constants: this document's
  entries for `DEFAULT_HEXIT_REL_TOLERANCE` (changed to `1e-9`), `DEFAULT_GK_NUMERIC_REL_TOLERANCE`
  (kept at `1e-8`) and `DEFAULT_TK_NUMERIC_REL_TOLERANCE` (changed to `3e-11`).
- **What it competes against:** not applicable.
- **Cost:** 0 objects keyed.
- **Campaign, prompt, log, date:** nobody. Recorded here per README §1.2's instruction that the
  point is that *no* accuracy parameter is unexplained when the campaign closes.

### `DEFAULT_HEXIT_REL_TOLERANCE = 1e-09` (`config/defaults.py:53`) — **changed**

- **Value, and what it keys:** `1e-9`, from the shared `1e-8`. Keys `wavenumber_exit_time` **by
  inequality, not equality** (`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`): the
  lookup accepts any stored row at least as tight as the request and returns the loosest such, so
  tightening to `1e-9` misses every existing row and recomputes, while a later loosening would
  silently reuse a tighter one.
- **What measurement chose it:** `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §7.6.
  Three models, all fifty production wavenumbers, three offsets (horizon crossing, the source-grid
  anchor at 5 e-folds superhorizon, and the numeric-region stop at 4 e-folds subhorizon), through
  `_solve_horizon_exit` directly and never through the datastore. Reference `(1e-300, 1e-14)`,
  drift ≤4.97e-14 (LambdaCDM) / ≤7.82e-14 (QCD) in $u$, converged 150/150 on every model; on the
  radiation control the reference sits 3.55e-15 from the exact $1+z = k/(H_0 e^N)$. `1e-9` is the
  loosest `rtol` whose Brent guarantee $x_{\rm tol}+r_{\rm tol}|u|$ at the largest production
  $|u|=38.04$ clears the floor: **3.81e-08 against 1e-07**; the measured displacement at that
  setting is 6.66e-09 (LambdaCDM) / 9.21e-09 (QCD). **The recommendation reads README §6.1's rule
  against Brent's guarantee, not the achieved displacement** — on the achieved reading (production's
  7.86e-08 already clears 1e-7 by 1.3×) the answer would be `unchanged`, and the user accepted `1e-9`
  on the guarantee reading specifically (log 03a, deviation 4).
- **What it competes against:** `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, the tolerance at which
  `Datastore/SQL/ObjectFactories/redshift.py:40` reuses an existing redshift row — the only threshold
  in the tree that says what displacement a grid can absorb.
- **Cost:** 7,009 Hubble evaluations at `1e-9` against 6,963 at production `1e-8` (**+0.7 %**), for
  all 150 (k, offset) solves on all three models; 7,040 at `1e-10` (+1.1 %). 50 objects per model.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 03a, `logs/03a-tk-numeric-and-exit-time.md`,
  `TK-NUMERIC-AND-EXIT-TIME.md` §7.6, 2026-09-17; accepted by the user 2026-09-17 (README §7 D1,
  closed); shipped by prompt 05a, 2026-09-18.

### `DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10` (`config/defaults.py:63`) — **unchanged**

- **Value, and what it keys:** `1e-10`. Keys `wavenumber_exit_time` alongside the `rtol` above, as
  `xtol` of the same `root_scalar`.
- **What measurement chose it:** **cannot be established from the record.** It is
  `DEFAULT_ABS_TOLERANCE`'s inherited value. Prompt 03a measured it binding at **0 of 150**
  (k, offset) pairs on each of the three models, at both the production and the recommended `rtol`.
- **What it competes against:** the same row match as `DEFAULT_HEXIT_REL_TOLERANCE`, but it cannot
  reach it alone: it is **coupled** to `rtol` rather than independent — with `xtol = 1e-10` the pair
  cannot pin the anchor better than **1e-10 relative** however far `rtol` is tightened, `xtol` taking
  over below `rtol ≈ 2.6e-12`. Any design tolerance below `1e-10` requires this constant to move with
  its partner.
- **Cost:** unmeasurable — it never binds in the production range, so no evaluation count depends on
  it.
- **Campaign, prompt, log, date:** measured (not chosen) by `prompts/tolerance-convergence` prompt
  03a, `TK-NUMERIC-AND-EXIT-TIME.md` §7.2, §7.6, 2026-09-17; left at its inherited value and shipped
  by prompt 05a, 2026-09-18, with the coupling recorded in the comment at its point of use.

### `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-08` (`config/defaults.py:86`) — **unchanged**

- **Value, and what it keys:** `1e-8`. Keys `GkNumericIntegration` alone (DOP853,
  `ComputeTargets/GkNumericIntegration.py:380`) — 29,290 objects per model on LambdaCDM, 38,105 on
  QCD, 58,350 on the radiation control, one per $(k, z_{\rm source})$ on the version-2 grid, at each
  cosmology's own anchor.
- **What measurement chose it:** `docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` §5.2, §7, §10.1.
  Three models, all fifty production wavenumbers, **version-2 grid at each cosmology's own anchor**,
  the production response geometry, `BREAK_POINT_DISCONTINUITY`, envelope-relative error, maximum
  over the grid. `rtol = 1e-8` gives **2.60e-07 / 2.48e-07 / 2.62e-07** (Radiation/LambdaCDM/QCD);
  one decade either side gives 2.93e-06/2.69e-06/2.95e-06 (`1e-7`) and 2.51e-08/2.46e-08/2.56e-08
  (`1e-9`). Reference `(1e-18, 1e-12)`, drift ≤2.1e-11 (Radiation, LambdaCDM), ≤3.0e-10 (QCD),
  converged at every wavenumber; the production cell stands ×1.1e+04–×399 above it. Calibrated
  against `compute_analytic_G` on the control: self-convergence and oracle agree at every setting.
  Four decades of `rtol` move the maximum by ×13,300; `atol` cannot bind (see below), which confirms
  README §2 (d)'s prior in this sector.
- **What it competes against:** the consumer's `numeric_Gk` cubic spline
  (`GkSourcePolicyData.py:654-680`), freshly measured here and dominant: **1.6e-04 to 9.4e-03** of
  the envelope near the hand-over against **2.6e-07** for the solver — **×631 to ×37,700**. Under
  README §6.1 rule 4 this makes the target `unchanged`: tightening the solver buys nothing while the
  consumer's own spline is the limiting error by two to four and a half orders.
- **Cost:** median 12,744/12,815/13,258 RHS evaluations per object at `1e-8`; ×objects =
  **0.744/0.375/0.505 ×10⁹** per model. One decade tighter (`1e-9`): **+31.0 %**. One decade looser
  (`1e-7`): **−27.3 %**.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 03, `logs/03-gk-numeric-and-its-floor.md`,
  `GK-NUMERIC-SWEEP.md` §10.1, 2026-09-17; accepted by the user 2026-09-17 (README §7 D1); shipped
  by prompt 05a, 2026-09-18.

### `DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10` (`config/defaults.py:93`) — **unchanged, and unchosen**

- **Value, and what it keys:** `1e-10`. Keys `GkNumericIntegration` alongside the `rtol` above.
- **What measurement chose it:** **nothing did, and the comment says so.** `GK-NUMERIC-SWEEP.md`
  §5.1, §5.3 measure it **inert**: four decades (`1e-8 → 1e-12`) at the production `rtol` move the
  maximum by at most **1.2 %** on any model, and the two `atol` corners of §5.3 agree to three
  significant figures at `rtol = 1e-6`. The reason is magnitude: $|G|$ runs **2.1e+12 to 2.9e+18**
  in `Mpc_units` over the production response grid, so `1e-10` is 1e-22 to 1e-29 of it and no value
  in this range is distinguishable by measurement.
- **What it competes against:** nothing — it never binds. The same consumer spline dominates the
  solver's total error by two to four and a half orders regardless.
- **Cost:** unmeasurable to three figures: 12,728→12,773 (Radiation), 12,788→12,815 (LambdaCDM),
  13,240→13,258 (QCD) median RHS evaluations across the four decades swept.
- **Campaign, prompt, log, date:** measured by `prompts/tolerance-convergence` prompt 03,
  `GK-NUMERIC-SWEEP.md` §10.2, 2026-09-17; left at its inherited value and shipped by prompt 05a,
  2026-09-18, **inert and unchosen in those words**.

### `DEFAULT_TK_NUMERIC_REL_TOLERANCE = 3e-11` (`config/defaults.py:123`) — **changed**

- **Value, and what it keys:** `3e-11`, from the shared `1e-8`. Keys `TkNumericIntegration` alone
  (DOP853, `ComputeTargets/TkNumericIntegration.py:413`) — 50 objects per model, one per $k$.
- **What measurement chose it:** `docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` §4, §4.1.
  Three models, all fifty production wavenumbers, **version-2 grid at each cosmology's own anchor**,
  under the sector's own `BREAK_POINT_ALL`. Reference `(1e-18, 1e-12)`, drift ≤4.26e-11 (Radiation) /
  ≤5.23e-11 (LambdaCDM) / ≤4.65e-09 (QCD), converged 50/50 on every model. Swept loose to tight over
  nine settings (README §6.1 rules 2 and 3 applied mechanically): `3e-11` is the first whose maximum
  over all three models — **3.88e-08** — clears the floor below.
- **What it competes against:** the $T=1,T'=0$ initial-condition truncation, re-measured here at
  **2.39e-06 to 2.64e-06** of the envelope on the version-2 grid under `BREAK_POINT_ALL`, against the
  inherited 2.52e-06 (`GkTk-remedial` prompt 17 §8, **version-0** grid, `BREAK_POINT_DISCONTINUITY`).
  The recommended setting sits **62×** under the floor; the production `1e-8` setting sits 141× *over*
  it at its worst wavenumber (14 of 150 runs above README §6's 3e-6 target).
- **Cost:** right-hand-side evaluations for the whole sector, all three models (50 objects/model):
  `1e-10` **1,727,040** (+35.7 %), **`3e-11` 1,774,977 (+39.4 %)**, `1e-11` 1,810,617 (+42.2 %),
  against 1,272,891 at production `1e-8`.
- **The caveat that ships with the value:** the maximum is **not monotone** in `rtol`
  (`[03a-tk-numeric-excursion-is-sporadic-in-rtol]`, open) — from `1e-9` down to `1e-10` exactly one
  of the 150 runs sits above target at each setting, a **different** run each time. `3e-11` is the
  loosest setting that clears *in this sweep*, not a bound over the sector.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 03a,
  `logs/03a-tk-numeric-and-exit-time.md`, `TK-NUMERIC-AND-EXIT-TIME.md` §4.1, 2026-09-17; accepted
  by the user 2026-09-17 (README §7 D1, closed); shipped by prompt 05a, 2026-09-18.

### `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` (`config/defaults.py:157`) — **unchanged, re-characterised**

- **Value, and what it keys:** `1e-13`. Keys `TkNumericIntegration` alone, alongside the `rtol`
  above.
- **What measurement chose it:** `prompts/GkTk-remedial` prompt 12, confirmed against the production
  grid by prompt 17 and **settled by the user 2026-09-12** (version-0 grid). **Re-characterised,
  value unchanged**, by `prompts/tolerance-convergence` prompt 03a on the version-2 grid: it is
  **not** the accuracy level in this sector — across `1e-12 → 1e-14` at `rtol = 1e-8` it moves the
  median of the per-$k$ maxima by at most 2.1× — it is a **step-selection knob**, moving the maximum
  by up to **205×** by changing which wavenumber draws a bad step sequence. `rtol` sets the level;
  `atol` selects which $k$ excurses.
- **What it competes against:** the same 2.39e-06–2.64e-06 initial-condition floor as `rtol`, but
  `atol` does not converge toward it monotonically — see above.
- **Cost:** not separable from `rtol`'s cost table; `atol` alone does not change the evaluation count
  materially in this sector.
- **Campaign, prompt, log, date:** the user, 2026-09-12, on `GkTk-remedial` prompt 17's
  recommendation; re-characterised by `prompts/tolerance-convergence` prompt 03a,
  `TK-NUMERIC-AND-EXIT-TIME.md` §4, 2026-09-17; **not reopened** (README §7 D1).

### `DEFAULT_QUADRATURE_ATOL = 1e-32` (`config/defaults.py:166`) — **unchanged, confirmed**

- **Value, and what it keys:** `1e-32`. Keys `QuadSourceIntegral`'s absolute half, distributed per
  sub-interval by log-width (`QuadSourceIntegral.py:819`) and per phase group
  (`QuadSourceIntegral.py:1054`), reaching `adaptive_levin_sincos` and `scipy.quad`/DOP853. It also
  reaches `analytic_integral`, so the stored `analytic_rad` column depends on it (standing note 27
  below).
- **What measurement chose it:** `prompts/source-remediation` prompt 12, against the analytic
  oracle, on a **live run**: raised from `1e-25`, where 58 % of work items met their tolerance before
  doing any work. **Re-measured read-only** by `prompts/tolerance-convergence` prompt 06,
  `docs/tolerance-convergence/QUADSOURCE-READONLY.md` §4, §8: **inert at this value**, binding only
  at `1e-16` and looser — 18 fixture cases (2 flavours × 3 shapes × 3 response redshifts), offline,
  through the fixture `ComputeTargets/tests/test_quadsource_integral.py` already builds. **No source
  grid in this sector** (`QuadSourceIntegral` is not built on the source grid; §0.4 excludes it from
  this campaign's own grid).
- **What it competes against:** the representation floor of the *realistic* flavour, freshly
  measured at **2.523e-06 to 4.912e-04 of scale**, dominating the quadrature error at
  `(1e-32, 1e-8)` by **×3.46e+04 to ×2.15e+11** — the campaign's largest dominating factor of the
  three `unchanged` results.
- **Cost:** inert — below `atol = 1e-20` the evaluation count does not move at all. Sector:
  1,275 × 50 × (response $z$) objects per model.
- **Campaign, prompt, log, date:** `prompts/source-remediation` prompt 12 (2026-09-09) for the
  value; `prompts/tolerance-convergence` prompt 06, `logs/06-quadsource-integral-read-only.md`,
  `QUADSOURCE-READONLY.md`, 2026-09-18, for the confirmation. **Owned by** `prompts/levin-refactor` /
  `prompts/qsi-phase-groups` (README §0.4); this campaign measured read-only and handed the finding
  over (`docs/OPEN_ISSUES.md` §1.3).

### `DEFAULT_QUADRATURE_RTOL = 1e-08` (`config/defaults.py:159`) — **unchanged, but the regime has inverted**

- **Value, and what it keys:** `1e-8`. Keys `QuadSourceIntegral`'s relative half, alongside `atol`
  above.
- **What measurement chose it:** **nothing chose it as a level.** `source-remediation` log 12 found
  it *did not bind* — `1e-8 → 1e-11` bit-identical on 159 live items — but that was measured at
  `atol = 1e-25`, where `atol` was the binding half. **That finding does not transfer to this tree**
  (standing note 26): prompt 06 measured the same three decades moving `total` by **×274** at
  `atol = 1e-32`, where `rtol` is now the only lever — about a decade of quadrature error per decade
  of tolerance. The value is nevertheless `unchanged`, because the representation floor is orders
  above both ends.
- **What it competes against:** as `DEFAULT_QUADRATURE_ATOL` — the representation floor, dominating
  by ×3.46e+04 to ×2.15e+11 at production.
- **Cost:** at production, 4,242 `quad` RHS + 222 Levin evaluations over the 18 fixture cases; one
  step either side, `1e-9` is **+3.9 %** and `1e-7` **−5.7 %**; three decades tighter is **+25.0 %**
  and three decades looser **−25.8 %**, on the largest object count in the pipeline
  (1,275 × 50 × (response $z$) per model). README §6.1 rule 3's ladder: the loosest `rtol` clearing
  the floor on every fixture case is `1e-5`, by ×3.32.
- **Campaign, prompt, log, date:** `prompts/source-remediation` prompt 12 (2026-09-09) for the value
  and the (now superseded) non-binding finding; `prompts/tolerance-convergence` prompt 06,
  `QUADSOURCE-READONLY.md` §§5, 8.1, 9, 2026-09-18, for the inversion and the confirmation that the
  value stays `unchanged` regardless. **Owned by** `prompts/levin-refactor` / `prompts/qsi-phase-groups`.

### `DEFAULT_LEVIN_THRESHOLD = 1.0` (`config/defaults.py:45`)

- **Value, and what it keys:** `1.0`. Keys `GkSourcePolicy` and `QuadSourcePolicy` **only as a
  default production never takes** — `main.py:3542`, `:3548`, `:3560`, `:3566` pass `1.5` and `5.0`
  explicitly, so `1.0` never reaches the datastore.
- **What measurement chose it:** **never chosen, and never used.** Nothing reads `Levin_threshold`
  off a policy object; the Levin/direct decision is made per region by `adaptive_levin_sincos`'s own
  total-variation gate.
- **What it competes against:** not applicable.
- **Cost:** 2 policy objects each per run (the `1.5`/`5.0` production values), keying
  `GkSourcePolicyData` through them; this constant itself keys nothing that is ever built.
- **Campaign, prompt, log, date:** nobody. Recorded per README §1.2's closing list.

---

## B. The integer orders and the region margin

### `TAU_GAUSS_ORDER = 4` (`ComputeTargets/BackgroundModel.py:34`) — **unchanged**

- **Value, and what it keys:** `4`. Keys `BackgroundModel` **since prompt 05** (2026-09-18) as
  `tau_gauss_order`, an equality predicate of `build()`. Also reaches the `IntegrationSolver` label.
- **What measurement chose it:** `docs/tolerance-convergence/ORDER-AUDIT.md` §§3.1, 5. Three models,
  **version-2 grid at each cosmology's own anchor**, difference error relative to the interval,
  cumulative from the top of the grid over 25 log-spaced checkpoints and per-interval over all
  1,777–2,305 intervals, on the corrected background and the 3-point break set. On the exact-radiation
  control (closed form `tau_delta`, no drift): order 2 gives 6.53e-11, order 4 gives **3.87e-16**. On
  the two spline models the reference is an order-32 table with drift 4.23e-16 (LambdaCDM) / 2.52e-15
  (QCD), and every order from 4 up is at or below it.
- **What it competes against:** double-precision accumulation over the grid, **2.16e-16** relative —
  the best any order in the ladder reaches. Order 4 sits **1.8×** above it; order 16 does not
  improve on it. Under README §6.1 rule 4 the target is `unchanged`, order 4 dominating order 2's
  floor by **×3.03e5**.
- **Cost:** `order × panels`. 1,777/2,036/2,305 panels per model (Radiation/LambdaCDM/QCD), so
  **7,108/8,144/9,220** evaluations at order 4, **−25 %** at order 3 and **+25 %** at order 5 —
  **one object per (cosmology, grid)** (`main.py:1062`), so the whole sector's cost is one table.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 04,
  `logs/04-order-governed-targets.md`, `ORDER-AUDIT.md`, 2026-09-17; written into
  `ComputeTargets/tests/wkb_reference_data.json`'s `convergence` block by prompt 04b, 2026-09-18;
  keyed by prompt 05, 2026-09-18. **Supersedes** the block's 2026-09-10 evidence, generated on a
  $T(z)$ representation and a break-point set both since replaced.

### `CS_TAU_GAUSS_ORDER = 4` (`ComputeTargets/BackgroundModel.py:43`) — **unchanged**

- **Value, and what it keys:** `4`. Keys `BackgroundModel` since prompt 05 as `cs_tau_gauss_order`.
  Reaches **no** solver label at all today — no trace of it anywhere in the store beyond the key
  itself.
- **What measurement chose it:** as `TAU_GAUSS_ORDER`, same method (`ORDER-AUDIT.md` §§3.1, 5).
  Order 2 gives 6.53e-11 and order 4 gives **3.34e-16** on the control; drift 4.29e-16 (LambdaCDM) /
  2.79e-15 (QCD) on the spline models. Same panel counts, one object per model.
- **What it competes against:** accumulation floor **3.30e-16**, order 4 dominating order 2 by
  **×1.99e5**. `unchanged` (§6.1 rule 4).
- **Cost:** as `TAU_GAUSS_ORDER` — same panel counts, same one object per (cosmology, grid).
- **Campaign, prompt, log, date:** as `TAU_GAUSS_ORDER`.

### `FRICTION_F_GAUSS_ORDER = 4` (`ComputeTargets/BackgroundModel.py:44`) — **unchanged**

- **Value, and what it keys:** `4`. Keys `BackgroundModel` since prompt 05 as
  `friction_F_gauss_order`.
- **What measurement chose it:** as above. The floor here is **not** quadrature: in exact radiation
  the integrand is constant, so every order is exact and the residual is pure accumulation. Order 2
  already clears on the control; order 4 is selected by the two spline models, where order 2 gives
  2.60e-14 (LambdaCDM) / 3.88e-12 (QCD) against **4.06e-16** / **3.64e-16** at order 4.
- **What it competes against:** accumulation floor **8.01e-14** — the accumulation of 2,305 panels
  against the closed form on the radiation control — dominating order 2 by **×48.4**, the smallest
  factor of the four orders. `unchanged` (§6.1 rule 4).
- **Cost:** as `TAU_GAUSS_ORDER` — same panel counts, one object per (cosmology, grid).
- **Campaign, prompt, log, date:** as `TAU_GAUSS_ORDER`.

### `RHO_GAUSS_ORDER = 4` (`ComputeTargets/phase_residual.py:90`) — **unchanged**

- **Value, and what it keys:** `4`. Keys **both** `GkWKBIntegration` and `TkWKBIntegration` since
  prompt 05 as `rho_gauss_order`, an equality predicate of both `build()`s. Also reaches
  `GkWKBIntegration.PHASE_SOLVER_STEPPING` and so a distinct `IntegrationSolver` row.
- **What measurement chose it:** `ORDER-AUDIT.md` §§4, 5. **Fifty production wavenumbers**, three
  models, **both sectors** — 300 cases — on the version-2 grid at each cosmology's own anchor, over
  the band `residual_node_range` returns at the production margin (three e-folds inside the
  horizon), absolute radians. On the control $\rho_G$ is **bit-exactly zero at every order** and
  $\rho_T$ is scored against its closed form: 1.13e-12 at order 2, **6.51e-17** at order 4. Worst
  over all three models: 1.53e-10 at order 2, 6.51e-17 at order 4. **Supersedes** `GkTk-remedial`
  prompts 02 and 06's three-wavenumber evidence, which this measurement finds was right and lucky
  by only ×1.5–×1.7.
- **What it competes against:** the $\rho$ quadrature accumulation floor, **6.51e-17 rad**, which
  order 4 already reaches — order 16 does not improve on it. Order 4 dominates order 2's floor by
  **×2.35e6**, the largest factor of the four orders. `unchanged` (§6.1 rule 4).
- **Cost:** `order × panels`; the band is 828–2,306 nodes, so 827–2,305 panels per table and
  **265,280–461,000** evaluations per 50-table sector at order 4 (exactly ±25 % at orders 3 and 5),
  with **100 tables per model** (50 $k$ × 2 sectors) memoised per worker. Consumers pay a further
  `order` evaluations each for the off-grid anchor partial: **4 evaluations per WKB object**
  (29,290–58,350 `GkWKBIntegration` objects per model, 50 `TkWKBIntegration`).
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 04,
  `logs/04-order-governed-targets.md`, `ORDER-AUDIT.md`, 2026-09-17; written into the `convergence`
  block by prompt 04b, 2026-09-18; keyed by prompt 05, 2026-09-18; **the recorded order made the
  order actually used**, on both the compute and rehydration paths, by prompt 05b, 2026-09-18
  (`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]`, closed).

### `RESIDUAL_WKB_REGION_MARGIN = 0.5` (`ComputeTargets/phase_residual.py:251`) — **unchanged**

- **Value, and what it keys:** `0.5`. **In no key, label or tag whatever** — prompt 05 deliberately
  did not make it a column. Sets the band `residual_node_range` covers, by requiring the
  Liouville–Green frequency to stay this fraction of its leading term; re-used by
  `main.source_grid_spacing_profile` as the band the version-2 source-grid density criterion runs
  over.
- **What measurement chose it:** `ORDER-AUDIT.md` §7.2 — **the first measurement of this constant
  anywhere.** Nine margins from 0.05 to 0.9999, fifty wavenumbers, three models, both sectors,
  applying README §6.1 **rule 6**: no accuracy floor exists, because the residual a producer reads is
  a `delta` between two fixed redshifts whose Gauss panels are the grid's own, so the margin can only
  change the answer by making the anchor unreachable. Bit-identical at **all 30 probes across
  0.05–0.9**; two of thirty depart at 0.9999, and only in the last bits (2e-16 relative), through the
  rounding of a cumulative whose top has moved.
- **What it competes against:** not an accuracy — a **reachability** bound. The band reaches the
  production anchor at every margin up to **0.9** on all three models and both sectors, and stops
  doing so in the `Tk` sector between 0.9 and 0.99; `residual_node_range` never refuses below
  margin 1, where the test degenerates to "is the correction positive?".
- **Cost:** the margin sets the band's size, hence the table's panel count. Going from 0.5 down to
  0.05 adds at most **99 nodes** (the transfer-function band on the radiation control, 828 → 927 of
  2,306) and changes no answer.
- **Campaign, prompt, log, date:** `prompts/tolerance-convergence` prompt 04, `ORDER-AUDIT.md` §7.2,
  2026-09-17; confirmed against by prompt 05 rather than taken from its own prompt, 2026-09-18. **Its
  re-use by the source grid's density criterion is a separate, still-open matter**
  (`[01-density-criterion-imposed-outside-the-wkb-region]`, board item candidate T7, unassigned):
  measured by prompt 04b §8 at 4 %–49 % super-horizon nodes in the `Gk` band, zero in `Tk`, costing
  QCD 114 of 2,034 source-grid samples and nothing on the other two cosmologies.

---

## C. Root solves in production code

### `_solve_horizon_exit` `xtol`/`rtol` (`CosmologyConcepts/wavenumber.py:1017-1022`)

Its own pair since prompt 05a: see `DEFAULT_HEXIT_ABS_TOLERANCE` / `DEFAULT_HEXIT_REL_TOLERANCE` in
section A above for the five fields — this row is the same parameter named at its call site rather
than at its constant.

### `DEFAULT_HEXIT_TOLERANCE = 0.01` (`CosmologyConcepts/wavenumber.py:884`)

- **Value, and what it keys:** `0.01`. Keys nothing.
- **What measurement chose it:** **never chosen**; no comment or log records it. It is not the
  solve's error bound — it is an acceptance guard raising if $|q(z_{\rm root})| > 10^{-2}$ *after*
  `root_scalar` reports convergence, and the bracket search's own step criterion. It is eight orders
  looser than the `xtol` it guards, so it can only catch a gross failure.
- **What it competes against:** not applicable.
- **Cost:** 50 solves per model; negligible.
- **Campaign, prompt, log, date:** nobody. Recorded so a later reader does not mistake it for the
  solve's error bound (`TOLERANCE-INVENTORY.md` §5.4 section C).

### `find_phase_extremum` `xtol=1e-6, rtol=1e-4` (`LiouvilleGreen/integration_tools.py:95-96`)

- **Value, and what it keys:** `xtol = 1e-6`, `rtol = 1e-4`. Keys nothing directly, but fixes the
  numeric stop point $z_{\rm init}$, which **is** a key column
  (`TkNumericIntegration.z_init_serial`, `GkWKBIntegration.z_init`).
- **What measurement chose it:** **never chosen.**
  `[11-stop-point-root-tolerance]` (`docs/OPEN_ISSUES.md` §1.1) records that
  `find_phase_extremum`'s pair places the stop point only to $\sim10^{-4}z$, so
  $|G'|/(|G|\omega)$ there measures 9.8e-6 to 6.5e-5, not the tighter figure a later prompt hoped
  for. `LiouvilleGreen/bessel_phase.py:48` records a case where exactly this pair produced an
  artefact elsewhere.
- **What it competes against:** not measured here — this is the numeric→WKB hand-over, and it is
  the hand-over campaign's subject, not this one's (README §0.5).
- **Cost:** once per `TkNumericIntegration` (50 per model) and once per `GkNumericIntegration` run
  in stop mode.
- **Campaign, prompt, log, date:** the hand-over campaign owns it (`docs/OPEN_ISSUES.md` §1.1);
  **recorded here and not retuned**, per README §0.5.

### `_solve_T_z` `xtol=1e-300, rtol=1e-14` (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:636`)

- **Value, and what it keys:** the pair above. Not itself a lookup key; the *representation version*
  it belongs to is (`QCD_Cosmology.T_z_representation`, currently `6`).
- **What measurement chose it:** `prompts/qcd-background-audit` prompt 04 (`71b842a`),
  `logs/04-tighten-node-solve.md`. **Lifted from
  [`prompts/background-solver-robustness/PROVENANCE.md`](../prompts/background-solver-robustness/PROVENANCE.md)
  §1, not re-derived**, per README §3.2's 2026-09-16 amendment.
- **What it competes against:** not re-measured here; see that campaign's own record.
- **Cost:** ~3,176 calls per `QCD_Cosmology` construction, once per run.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; settled and owned there.

### `_find_rho_equality` `xtol=1e-300, rtol=8.9e-16` (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:1137`)

- **Value, and what it keys:** the pair above. **Reaches a lookup key indirectly**: the two equality
  redshifts it locates are production source-grid sample locations, so they reach the
  `BackgroundModel` lookup key through the grid digest
  (`background-solver-robustness/PROVENANCE.md` §3.1).
- **What measurement chose it:** `prompts/background-solver-robustness` prompt 02, user decision
  2026-09-16. **Lifted from that campaign's `PROVENANCE.md` §3, not re-derived.** `rtol` is Brent's
  own $4\varepsilon$ floor, chosen after `rtol = 1e-14` was measured to stop 7 ulp from an
  independent reference.
- **What it competes against:** not re-measured here.
- **Cost:** 2 per model construction, once per run.
- **Campaign, prompt, log, date:** `prompts/background-solver-robustness`; settled and owned there.

### `temperature_crossing_log1pz` `xtol=1e-15, rtol=1e-15` (`CosmologyModels/tests/T_z_reference.py:285`)

- **Value, and what it keys:** the pair above. Keys nothing.
- **What measurement chose it:** **never chosen.** Introduced by `prompts/GkTk-remedial` prompt 03
  without a measurement, and it is a root solve on a residual that **need not have a root** — the
  $T(z)$ representation is segmented at exactly these temperatures. **Lifted from
  `background-solver-robustness/PROVENANCE.md` §2**, which moved it out of the production class and
  off the production path (`qcd-background-audit` prompt 07).
- **What it competes against:** not applicable — there is no root for a tolerance to converge to.
- **Cost:** 0 in the pipeline; 3 per run of one `ComputeTargets` test.
- **Campaign, prompt, log, date:** nobody — it is test machinery. Listed because README §3.2's
  original anchor `LambdaCDM_GenericEOS.py:864` pointed here before
  `background-solver-robustness` moved it.

---

## D. The Bessel phase representation

### `phase_atol=1e-12`, `amplitude_rtol=1e-12` (`main.py:1119`, `:1127`)

- **Value, and what it keys:** `1e-12` each. Keys nothing; sets the construction budget of the two
  Bessel phase objects ($\nu = 1/2+b$ and $5/2+b$), consumed by every `QuadSourceIntegral`.
- **What measurement chose it:** chosen **by argument, not by sweep**, in the comment at
  `main.py:1105-1114`: one order tighter than the `1e-11` `prompts/transfer-remedial` accepts, with
  the declared errors quoted — 5.0e-13 rad at $\nu=5/2$, 3.0e-13 relative amplitude — and the
  observation that another order buys almost nothing.
- **What it competes against:** the scaled-Hankel sampling floor, which the comment records as
  setting the realised 3.0e-13 rather than the request.
- **Cost:** 2 objects per run, consumed by every `QuadSourceIntegral`.
- **Campaign, prompt, log, date:** `prompts/transfer-remedial` owns the representation; recorded
  here by `prompts/tolerance-convergence` prompt 06.

### `DEFAULT_PHASE_ATOL = 1e-11`, `DEFAULT_AMPLITUDE_RTOL = 1e-11` (`LiouvilleGreen/bessel_phase.py:134`, `:137`)

- **Value, and what it keys:** `1e-11` each. Keys nothing; **`main.py` overrides both at both call
  sites**, so they reach production nowhere.
- **What measurement chose it:** `prompts/transfer-remedial`'s low-order acceptance target, recorded
  at `bessel_phase.py:128-133`.
- **What it competes against:** not applicable in production.
- **Cost:** 0 in the pipeline.
- **Campaign, prompt, log, date:** `prompts/transfer-remedial`.

### `bessel_phase(atol=, rtol=)` — deprecated, `None` and ignored (`LiouvilleGreen/bessel_phase.py:874-875`)

- **Value, and what it keys:** ignored. Keys nothing; a compatibility shim for an ODE representation
  that no longer exists. Supplying either raises a `DeprecationWarning` and maps to nothing.
- **What measurement chose it:** not applicable — it is dead.
- **What it competes against:** not applicable.
- **Cost:** 0.
- **Campaign, prompt, log, date:** nobody. Listed because a keyword sweep for `atol=` finds it and
  would otherwise miss the live budgets above it (`TOLERANCE-INVENTORY.md` §5.4 section D).

---

## E. The Levin driver

### `CHEBYSHEV_ORDER = 24` (`ComputeTargets/QuadSourceIntegral.py:71`)

- **Value, and what it keys:** `24`. Keys **nothing** — `QuadSourceIntegral` stores the *achieved*
  `WKB_Levin_chebyshev_min_order` as a diagnostic column but does not filter on it.
- **What measurement chose it:** `prompts/source-remediation`, a self-consistency sweep recorded at
  `QuadSourceIntegral.py:60-70`, **with the code's own caveat that it "cannot rule out an
  order-independent bias shared by every order tested."** Not re-measured by this campaign; prompt
  06's fixture runs all report `chebyshev_min_order = 24`, i.e. no region needed less.
- **What it competes against:** not measured here.
- **Cost:** the collocation order of every `adaptive_levin_sincos` call in the sector
  (1,275 × 50 × (response $z$) objects per model).
- **Campaign, prompt, log, date:** `prompts/source-remediation`; owned by `prompts/levin-refactor` /
  `prompts/qsi-phase-groups`. Recorded here by `prompts/tolerance-convergence` prompt 06, 2026-09-18.

### `DEFAULT_LEVIN_MAX_DEPTH = 20` (`AdaptiveLevin/levin_quadrature.py:154`)

- **Value, and what it keys:** `20`. Keys nothing; production takes this default, no caller
  overrides it.
- **What measurement chose it:** **cannot be established from the record.** The comment at `:153`
  gives only the geometric reading "1/2^20 is roughly 1E-6" — not a measurement. **Observed, not
  established**, by prompt 06: the deepest bisection any fixture case reached was **8**, so the cap
  is not binding on that instrument, which is a fact about the fixture and not a provenance for the
  constant.
- **What it competes against:** not established.
- **Cost:** maximum bisection depth of every Levin call, together with the caller's `atol`/`rtol`.
- **Campaign, prompt, log, date:** `prompts/levin-refactor` owns it; recorded as **unestablished, in
  those words**, by `prompts/tolerance-convergence` prompt 06, `logs/06-quadsource-integral-read-only.md`,
  2026-09-18.

### `limit=100` (`Quadrature/simple_quadrature.py:92`)

- **Value, and what it keys:** `100`. Keys nothing; the subdivision cap of `scipy.integrate.quad` on
  the `method="quad"` path `QuadSourceIntegral` takes at `:1498` (`_three_bessel_quad`) and `:1721`
  (`numeric_quad_integral`).
- **What measurement chose it:** **cannot be established from the record.** The literal carries no
  comment and no campaign document mentions it. Not exercised to its cap by prompt 06's fixture.
- **What it competes against:** together with `DEFAULT_QUADRATURE_ATOL`/`_RTOL`: a quadrature that
  exhausts the cap returns without meeting the tolerance.
- **Cost:** the non-oscillatory sub-intervals of every `QuadSourceIntegral`.
- **Campaign, prompt, log, date:** owned by `prompts/levin-refactor` / `prompts/qsi-phase-groups`;
  recorded as **unestablished, in those words**, by `prompts/tolerance-convergence` prompt 06,
  2026-09-18.

### `DEFAULT_LEVIN_ABSTOL = 1e-15`, `DEFAULT_LEVIN_RELTOL = 1e-07` (`AdaptiveLevin/levin_quadrature.py:157`, `:160`)

- **Value, and what it keys:** as shown. Keys nothing; **every production call passes its own
  `atol`/`rtol`**, so these defaults reach nothing in the pipeline.
- **What measurement chose it:** **never chosen** — the comments read only "default abs/rel
  tolerance".
- **What it competes against:** not applicable.
- **Cost:** 0 in the pipeline.
- **Campaign, prompt, log, date:** `prompts/levin-refactor`.

### `DEFAULT_LEVIN_CHEBSHEV_ORDER = 16` (`AdaptiveLevin/levin_quadrature.py:150`)

- **Value, and what it keys:** `16`. Keys nothing; `QuadSourceIntegral` passes `24` explicitly, so
  this default reaches nothing in production.
- **What measurement chose it:** `prompts/adaptive-levin-benchmark`'s order sweep, recorded at
  `:139-149`.
- **What it competes against:** not applicable in production.
- **Cost:** 0 in the pipeline.
- **Campaign, prompt, log, date:** `prompts/adaptive-levin-benchmark`; owned by
  `prompts/levin-refactor`.

### `BESSEL_ORDER_CHECK_TOL = 1e-3` (`ComputeTargets/QuadSourceIntegral.py:93`)

- **Value, and what it keys:** `1e-3`. Keys nothing — it is a guard against a wrong Bessel order,
  not an accuracy request; compares a reconstructed $J_\nu$ against the phase object's own envelope.
- **What measurement chose it:** **argued, not swept**, in the comment at `QuadSourceIntegral.py:79-92`:
  nine orders above the reconstruction floor and two below the smallest defect it must catch. **That
  is a provenance and this note does not record it as unestablished** — prompt 06's log is explicit
  that this is the one guard in section E/F whose comment rises to an argument rather than a bare
  assertion.
- **What it competes against:** the reconstruction floor above and the smallest Bessel-order defect
  below, both named in the comment.
- **Cost:** as `CHEBYSHEV_ORDER` — one check per phase object construction.
- **Campaign, prompt, log, date:** `prompts/levin-refactor` / `prompts/qsi-phase-groups`; recorded
  by `prompts/tolerance-convergence` prompt 06, 2026-09-18.

---

## F. Representation orders that are not in any key

### `DEFAULT_T_Z_SPLINE_SAMPLES = 3000`, `DEFAULT_T_Z_SPLINE_ORDER = 5` (`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:72`, `:73`)

- **Value, and what it keys:** `3000` nodes, order `5`. Keys `BackgroundModel` **through
  `T_Z_REPRESENTATION_VERSION` only** — both reach the lookup key via the version and set the
  tabulated $T(z)$ representation's own accuracy directly.
- **What measurement chose it:** `prompts/qcd-background-audit` prompt 06 (`a1d667a`) raised the
  sample count to 3,000.
- **What it competes against:** not re-measured here.
- **Cost:** 1 per `QCD_Cosmology`, once per run.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; owned there.

### `DERIVATIVE_SPLINE_ORDER = 5`, `STORED_SAMPLE_SPLINE_ORDER = 3`, `DERIVATIVE_FIT_PAD_POINTS = 12`, `DERIVATIVE_FIT_REFINE = 3`, `DERIVATIVE_FIT_PAD_FLOOR = 0.9`, `DERIVATIVE_FIT_PAD_FRACTION = 0.05` (`ComputeTargets/BackgroundModel.py:54-69`)

- **Value, and what it keys:** as shown. Keys **nothing** — these govern the interpolation and
  padding of `BackgroundModel`'s own derivative fit and the spline through its stored samples, not
  a lookup.
- **What measurement chose it:** `prompts/qcd-background-audit` prompt 13.
  `[03-derivative-pad-clamp-on-coarse-grids]` is open against the pad clamp on that campaign's board.
- **What it competes against:** not re-measured here.
- **Cost:** 1 `BackgroundModel` per (cosmology, grid); read by every target.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; **not in this campaign's scope**,
  and listed so that prompt 04's answer to "which accuracy parameters are in no key" is complete.

### `BESSEL_ORDER_CHECK_TOL` — see section E above (listed there under its owning group; cross-referenced from inventory §5.4 section F).

---

## G. The source-grid construction constants

**All of these are held fixed by this campaign** (README §0.5): they belong to
`prompts/qcd-background-audit`, and the one narrow exception — prompt 02a's edit to
`main.source_grid_spacing_profile` — changed no constant's value (§7 D6).

### `SOURCE_GRID_CONSTRUCTION_VERSION = 2` (`CosmologyConcepts/wavenumber.py:61`)

- **Value, and what it keys:** `2`. Keys `BackgroundModel.source_grid_construction`, an equality
  predicate — the identity of the construction the constants below define, not an accuracy itself.
- **What measurement chose it:** `prompts/qcd-background-audit` prompt 15.
- **What it competes against:** not applicable — it is an identity tag.
- **Cost:** 1 per (cosmology, grid).
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; held fixed here.

### `SOURCE_GRID_MAX_SPACING_FACTOR = 1.0` (`CosmologyConcepts/wavenumber.py:181`)

- **Value, and what it keys:** `1.0`. Reaches the lookup key through the grid digest, on every
  target that carries one, and caps the density criterion so it may only ever refine, never coarsen.
- **What measurement chose it:** `prompts/qcd-background-audit` prompt 15, on the user's instruction
  quoted verbatim at `wavenumber.py:160-170`; the declined saving is costed in
  `docs/qcd-background-verification.md` §10.5.
- **What it competes against:** not re-measured here.
- **Cost:** the whole pipeline — every target is sampled on this grid.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; held fixed here.

### `SOURCE_GRID_CONSUMER_TARGET_RAD = 1e-06` (`CosmologyConcepts/wavenumber.py:223`)

- **Value, and what it keys:** `1e-6`. Reaches the lookup key through the digest, and is **the
  accuracy target of the source grid itself**: the ceiling on the per-case phase error the density
  criterion equidistributes.
- **What measurement chose it:** `prompts/qcd-background-audit` prompt 10 §5.
- **What it competes against:** not re-measured here. This campaign's own finding
  (`[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`) is that the criterion
  this constant governs is sized for the phase-residual spline and not for $G_k$'s own oscillation,
  and that mismatch is now assigned to the hand-over campaign (`docs/OPEN_ISSUES.md` §1.1).
- **Cost:** the whole pipeline.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; held fixed here.

### The curvature-criterion constants — `SOURCE_GRID_CUBIC_ERROR_CONST`, `SOURCE_GRID_MAX_REFINEMENT`, `SOURCE_GRID_CURVATURE_STEP_U`, `SOURCE_GRID_CURVATURE_FD_STEP_U`, `SOURCE_GRID_CROSSING_MASK_U`, `SOURCE_GRID_SPLINE_EDGE_INTERVALS`, `SOURCE_GRID_SPLINE_EDGE_FACTOR` (`CosmologyConcepts/wavenumber.py:188-244`)

- **Value, and what it keys:** `0.0026041666666666665` / `32` / `0.001` / `0.0001` / `0.006` / `3` /
  `10.0`. Reach the lookup key through the digest and set the version-2 curvature criterion
  $h^4|\varphi''''|/384 \le \varepsilon$ and its five-point stencil.
- **What measurement chose it:** `prompts/qcd-background-audit` prompts 12 and 15; each constant
  carries its own measurement in the comment above it, and `docs/qcd-background-verification.md`
  §10 holds the tables.
- **What it competes against:** not re-measured here. This campaign's prompt 02a found and closed
  the one hazard in the stencil's interaction with the crossing mask
  (`[02a-stencil-reaches-across-a-declared-break-before-the-mask]`, resolved by a guard, not by
  moving these constants — see `TOLERANCE-CONVERGENCE.md` §5 for the close-out of that finding).
- **Cost:** the whole pipeline.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; held fixed here.

### The crossing-neighbourhood constants — `SOURCE_GRID_BREAK_STANDOFF`, `SOURCE_GRID_BREAK_HALF_WIDTH`, `SOURCE_GRID_BREAK_REFINEMENT`, `SOURCE_GRID_MESH_GUARD`, `SOURCE_GRID_MIN_SEPARATION` (`CosmologyConcepts/wavenumber.py:113-145`)

- **Value, and what it keys:** `0.25` / `5` / `2` / `0.25` / `1e-06`. Reach the lookup key through
  the digest, and set the version-1 straddling pair and refined neighbourhood at each declared
  crossing.
- **What measurement chose it:** `prompts/qcd-background-audit` prompts 10 and 11, scored in
  `docs/qcd-background-verification.md` §8 and log 11 §3.
- **What it competes against:** not re-measured here.
- **Cost:** the whole pipeline, concentrated near a crossing.
- **Campaign, prompt, log, date:** `prompts/qcd-background-audit`; held fixed here.

---

## Coverage and the closing rule

**All 44 rows of `TOLERANCE-INVENTORY.md` §5.4** are accounted for above, under the same seven
group letters the inventory uses (11 + 5 + 6 + 5 + 6 + 6 + 5 = 44; `phase_atol`/`amplitude_rtol` at
`main.py:1119`/`:1127` and their two occurrences at `:1197`/`:1205`/`:1200`/`:1201`/`:1208`/`:1209`
count as the one row the inventory gives them). The six constants prompt 05a shipped —
`DEFAULT_HEXIT_ABS_TOLERANCE`, `DEFAULT_HEXIT_REL_TOLERANCE`, `DEFAULT_GK_NUMERIC_ABS_TOLERANCE`,
`DEFAULT_GK_NUMERIC_REL_TOLERANCE`, `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`,
`DEFAULT_TK_NUMERIC_REL_TOLERANCE` — are section A's rows 3–8 above, not an addition to the 44.

**Parameters whose provenance cannot be established from the record, in README §1.2's own words**:

- `DEFAULT_ABS_TOLERANCE = 1e-10` and `DEFAULT_REL_TOLERANCE = 1e-8` (section A) — no longer any
  target's tolerance, but still seven float comparisons' epsilon and a set of signature defaults
  nothing production reaches through.
- `DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10` (section A/C) — inert, unchosen, and coupled to its partner
  below `rtol ≈ 2.6e-12`.
- `DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10` (section A) — inert and unchosen; four decades move the
  maximum by at most 1.2 %.
- `DEFAULT_HEXIT_TOLERANCE = 0.01` (section C) — a bracket-width/acceptance guard, not a tolerance
  despite the name.
- `find_phase_extremum`'s `xtol=1e-6, rtol=1e-4` (section C) — the hand-over campaign's, recorded
  and not retuned.
- `DEFAULT_LEVIN_THRESHOLD = 1.0` (section A) — never used; production always overrides it.
- `DEFAULT_LEVIN_MAX_DEPTH = 20` (section E) — the comment gives a geometric reading, not a
  measurement.
- `limit = 100` (section E) — no comment, no campaign document.

**Parameters argued rather than swept, and correctly *not* filed above as unestablished**:
`BESSEL_ORDER_CHECK_TOL = 1e-3` (section E) and the Bessel phase budget
`phase_atol=1e-12`/`amplitude_rtol=1e-12` (section D) — both carry a comment that reasons from a
named floor to a chosen margin, which is a provenance even though it is not a sweep.

**Two supersessions this note reflects** (§4 of prompt 06a; board standing notes 26 and 27):

- **Standing note 26**, reflected in this document's `DEFAULT_QUADRATURE_RTOL` entry above:
  README §6.2's last row, `config/defaults.py:163`'s comment and
  `ComputeTargets/QuadSourceIntegral.py:1550`'s comment all still state the `source-remediation` log
  12 finding at `atol = 1e-25` ("`rtol` does not bind"), which does not hold at the shipped
  `atol = 1e-32` — at that value `atol` is inert and `rtol` is the only lever. **Neither file may be
  edited by this prompt**; a §3 issue is opened on the campaign board for whoever next touches
  either.
- **Standing note 27**, reflected in this document's `DEFAULT_QUADRATURE_ATOL` entry above:
  `analytic_rad` is computed inside `evaluate_QuadSource_integral` at the caller's own `atol`/`rtol`
  rather than at a fixed reference pair, so a stored `analytic_rad` moves with its row's tolerance
  and any prior reading of it as a stable oracle for a residual comparison is superseded. No file
  states this incorrectly today — it is a fact about the code that nothing in the tree had recorded
  before prompt 06 — so there is nothing to correct and no new issue to open; it is
  `[06-analytic-rad-is-computed-at-the-callers-tolerance]` on `qsi-phase-groups`' own board.
