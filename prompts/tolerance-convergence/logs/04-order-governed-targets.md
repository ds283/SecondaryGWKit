# Log 04 — Audit the order-governed targets

**Prompt:** `prompts/tolerance-convergence/04-order-governed-targets.md`
**Commit:** *(this prompt's own commit)* — "Audit the four Gauss orders and the WKB region margin"
**Model:** Opus 5
**Date:** 2026-09-17
**Result:** **STOPPED, WITH THE MEASUREMENT COMPLETE.** Prompt §11's **first** stop condition fired,
word for word as prompt §2.2 (i) predicted it: the regenerated reference floor is 58× tighter
($\tau$) and 43× tighter ($c_s\tau$) than the 2026-09-10 one, and both threshold tests that read it
— one of them in `test_background_cs_tau_friction.py`, **outside** the D5 carve-out — then fail by
factors of 6.99 and 5.08 with their numerators unmoved to every digit. **The `convergence` block
was therefore not written, `test_background_tau.py` was not edited, and
`QCD_BREAK_POINT_ALIGNMENT_TOL` was not moved.** The tree is green and no production module, test
module or fixture is touched.

**T7 itself is complete and is the answer the campaign asked for.** All four orders are
**`unchanged` at 4**, measured for the first time on the corrected background and the 3-point break
set, over the version-2 grid at each cosmology's own anchor, and — for $N_\rho$ — at **every one of
the fifty production wavenumbers** in both sectors rather than `GkTk-remedial` prompt 02's three.
`RESIDUAL_WKB_REGION_MARGIN = 0.5` is **`unchanged`** under README §6.1 **rule 6**: no accuracy
floor exists for it, and the bound that does exist — the band must reach the producer's anchor — is
cleared at every margin from 0.05 to 0.9, with the residual bit-identical throughout. **D3's
recommendation is `replace with the orders`.** And
`[01-density-criterion-imposed-outside-the-wkb-region]` has, for the first time, **its own
measurement**.

`[01-convergence-block-has-a-separate-generator]` is **not** closed. It is now *blocked on a
decision rather than on scope*, which is a different and better place for it to be, and the log
says exactly what the decision is.

---

## What shipped

**Two files, both under `docs/`, plus this campaign's log, board and `docs/OPEN_ISSUES.md` and the
`qcd-background-audit` board entry.** `git diff --stat HEAD~1 HEAD -- . ':!prompts' ':!docs'` is
**empty**: no production module, no fixture, no test module, nothing under `ComputeTargets/`.

### `docs/tolerance-convergence/ORDER-AUDIT.md` — new

The measurement document. §0 and §1 are prose; **every table in §§2–11 is `order_audit.py`'s
stdout**, from the `<!-- generated -->` comment down. §1 is the stop, with both floors and both
errors.

### `docs/tolerance-convergence/order_audit.py` — new

Regenerates every table in §§2–11. What it measures:

1. **$N_\tau$, $N_{c_s\tau}$, $N_F$** at orders 2, 4, 6, 8, 12, 16 on three models, scored as
   **difference error relative to the interval** over every production interval and over the
   cumulative from the top of the grid. They are $k$-independent and the document says so instead
   of printing fifty identical rows.
2. **$N_\rho$** at the same orders at **all fifty production wavenumbers**, both sectors, three
   models — 300 cases — scored as **phase error in absolute radians** of
   $\rho.\mathrm{delta}(z_{\rm anchor}, z)$ from the production anchor three e-folds inside the
   horizon, through the production builder `phase_residual.build_phase_residual` and over the
   production band `residual_node_range` returns.
3. **Cost**, in integrand evaluations **counted** by a wrapper rather than derived, at the
   recommended order and one order either side, times the object count of the sector.
4. **The margin**, over nine values from 0.05 to 0.9999: nodes retained and discarded per
   `(model, k, sector)` at all fifty wavenumbers, whether the band still reaches the production
   anchor, a **bit-for-bit** comparison of the residual the producer reads, and the margin at
   which `residual_node_range` refuses, bisected.
5. **The band against the WKB region**: how many of the band's nodes are super-horizon in the
   sector's own leading frequency, and what the grid would be if the criterion ran over the
   sub-horizon part alone — measured by executing `main.source_grid_spacing_profile`
   **unmodified** against a band it is handed (deviation 6).
6. **The fixture re-run**, figure by figure against the 2026-09-10 block, which is read **out of
   git** at `74ddb39` rather than transcribed.

Everything goes through `ComputeTargets/tests/convergence_reference.py` (board standing note 14):
`GaussOrder` is the knob, one step is one order, and `reference_drift` will not return a drift
without the verdict attached. **No line of that module is changed.**

### `docs/gktk-remedial/residual_convergence.py` — the D5 carve-out, changed and **run but not
committed to the fixture**

Four changes, and every one is a change to *what is recorded* rather than to what is measured:

1. **`branch+knots` is a control, not a candidate** (prompt §3.2). `CANDIDATE_SCHEMES` is
   `("plain", "branch")` — the schemes `integration_break_points` lets production execute — and
   `CONTROL_SCHEME` is `"branch+knots"`, kept in `schemes` and in the block because two test
   modules index `models.QCDModel["branch+knots"]` by name. A new `decision.knots_control` scores
   what splitting at the knots still buys.
2. **Every case carries its reference's own drift** (§5 rule 5), through
   `convergence_reference.reference_drift`, with `reference_error_bound` the larger of that and
   the independent order-40 Gauss cross-check the script already had.
3. **Every case carries its source-grid generation** (§5 rule 6): a new top-level `grid` block
   naming **version 0**, the anchor, the sample count, and *why* it must stay at version 0 — the
   quantity this block scores is the agreement of a fixed-order rule with the JSON's own
   `checkpoints` and `rho_*` values, and those live on the version-0 grid, as do the tests that
   read the block.
4. **`argparse`**: `--json-out PATH` (the dry run prompt §2.4 requires), `--no-json`, and
   `--legacy-markdown`. **The markdown is off by default**, so
   `docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` is left exactly as `GkTk-remedial` prompt 02
   published it (README §5 rule 7, prompt §11's last stop condition).

Plus one repair the re-run forced, deviation 3.

---

## Deviations from the prompt

### 1. The `convergence` block was not written, and `test_background_tau.py` was not edited — `STRUCTURALLY REQUIRED`

Prompt §11's first stop condition. §2.2 (i) predicted it: *"The corrected background is four orders
more accurate than the one the block was taken on, so the regenerated floor will very likely be
**smaller**, and `test_qcd_checkpoints` in `test_background_cs_tau_friction.py` will then fail
unless the production checkpoints improved by the same factor."* They did not improve at all — they
are the same model measured against the same JSON values, which the re-run does not touch.

Measured, by copying the dry-run block over the fixture, running the three reader modules, and
restoring the fixture (`git diff` clean afterwards):

| | $\tau$ | $c_s\tau$ |
|---|---|---|
| floor, 2026-09-10 | 1.878541e-14 | 1.886653e-14 |
| floor, regenerated | 3.223619e-16 | 4.354138e-16 |
| threshold = 3 × floor, old → new | 5.636e-14 → 9.671e-16 | 5.660e-14 → 1.306e-15 |
| production error, old block → new block | 2.254e-15 → **2.254e-15** | 2.212e-15 → **2.212e-15** |
| factor needed to pass | 6.99 | 5.08 |

`ComputeTargets.tests.test_background_tau`, `test_background_cs_tau_friction` and
`test_phase_residual` together: **41 tests, 2 failures** —
`test_background_tau.test_qcd_nodes_against_adaptive_reference` and
`test_background_cs_tau_friction.test_qcd_checkpoints`. `test_phase_residual` passes: `N_rho` is
still 4 and `rho_adaptive_fallback_required` is still false, so prompt §2.4's other stop did not
fire and prompt §2.2 (iii) is not reached.

Consequences, all of them recorded rather than worked around:

- the block stays at 2026-09-10 and `[01-convergence-block-has-a-separate-generator]` stays open;
- `QCD_BREAK_POINT_ALIGNMENT_TOL` stays at `1.5e-04`, because the test asserts against the block in
  the tree. **Prompt §3.3's question is nevertheless answered**: against the regenerated block the
  worst offset is **1.421085e-14**, so the constant goes back *further* than the 1.4e-05 §3.3 hoped
  for, and the whole of the present excess really is the block's age;
- `docs/gktk-remedial/RESIDUAL-CONVERGENCE.md` is untouched, which it would have been anyway.

The repair is out of scope in all three of its available forms, and the log says so rather than
choosing one: raising `QCD_FLOOR_FACTOR` past 7 is a **two-module** edit; replacing the threshold
with one that does not divide the model's accuracy by the reference's is a better fix and a bigger
decision; and writing the old floor into the new block is what §11 forbids in terms.

### 2. The reference's drift is measured from the **loose** side — `STRUCTURALLY REQUIRED`

Prompt §3.4 asks the regenerated block to carry "the reference's own drift beside the number", and
the campaign's drift is *the movement under one step of tightening*. `scipy.integrate.quad` refuses
an `epsrel` below `50 * eps = 1.11e-14` outright when `epsabs = 0` — it raises `ValueError` — and
the reference sits at `QUAD_EPSREL = 1.5e-14`. **There is no decade below it to step to.** What is
recorded instead is the decade *into* it, `1.5e-13 → 1.5e-14`, with
`reference_drift_step_is_from_the_loose_side: true` in the payload and the reason in
`reference_case_drift`'s docstring. The second leg, the independent order-40 Gauss rule the script
already ran, is folded in as `reference_error_bound`.

### 3. `decide()`'s scheme filter needed a zero-floor guard — `STRUCTURALLY REQUIRED`

With `branch+knots` demoted to a control, the filter "no candidate may be worse than
`FLOOR_FACTOR` × the best floor any measured scheme reaches" became unsatisfiable: on the
regenerated measurements `branch+knots` reaches **exactly 0.0** on `friction`, and nothing can be
three times worse than zero. Every candidate was rejected, including `branch`, which is best on the
other two primitives. The filter now skips a primitive whose best floor is exactly zero and records
it in `decision.zero_best_floor_primitives`. Without the guard the script reports
`no_candidate_scheme_qualified: true` and falls back to `branch` anyway — the same recommendation,
with a false reason attached.

### 4. The drift is measured in the kind the case decides in — `IMPLEMENTATION CHOICE`

Relative for the three primitives, whose decision reads `max_cumulative_rel_error`; **absolute
radians** for the two residuals, whose decision reads `max_cumulative_abs_error` against a target
in radians. The first draft used relative everywhere and reported every $\rho$ case as
`NOT CONVERGED` at a drift of 3.9e-11: a $\rho$ increment is ~1e-13 rad, so an absolutely
negligible movement is enormous against that denominator, and what was being reported was the
conditioning of the denominator and not the convergence of the reference.

### 5. "Clears the floor" means within a factor of 3, not "at or below" — `IMPLEMENTATION CHOICE`

README §6.1 rule 2 says *at or below the floor*. For an integer knob whose floor is
double-precision accumulation, the floor is not independently known — the ladder is the only way to
see it, and it is taken as the best any order reaches. A literal `<= floor` would then select
whichever order happened to hit the minimum, which is one order's rounding. `order_audit.FLOOR_FACTOR
= 3.0` is `residual_convergence.smallest_within_factor`'s value and meaning, kept so that this
re-take and the 2026-09-10 block answer the same question. It is stated in `ORDER-AUDIT.md` §2 and
in `choose_order`'s docstring, and it changes no answer here: at order 2 every quantity misses by
four to five orders and at order 4 every one of them is at the floor.

### 6. The horizon-limited grid is measured by handing the production function a different band — `IMPLEMENTATION CHOICE`

Prompt §5's last paragraph asks "what would the grid look like if the band were the WKB region
alone". `main.source_grid_spacing_profile` is executed **unmodified**: `order_audit` takes its own
lift of that function through `load_main_py_functions` and supplies, in the lift's globals, a
`residual_node_range` that calls the real one and then drops the super-horizon nodes. Nothing in
`main.py` or `ComputeTargets/phase_residual.py` is touched and no grid in the tree moves; the three
production digests are re-verified as a precondition of every run.

### 7. The residual reference is an order-32 table of the same rule, not an adaptive one — `IMPLEMENTATION CHOICE`

README §0.2's self-convergence test with `GaussOrder` as the knob: reference at order 32, drift
against order 33. It is the production builder at a different order, so it inherits production's
break-point splitting, which an adaptive rule laid over the same nodes would not. The
exact-radiation control is what calibrates it, and there the reference is a closed form.

### 8. The margin's bit-identity probe is at five wavenumbers, not fifty — `IMPLEMENTATION CHOICE`

The **node census** of §7.1 is at all fifty, on three models and both sectors, at nine margins. The
**table rebuild** of §7.2 is at five wavenumbers spanning the range: 9 × 50 × 2 × 3 = 2,700 table
builds is 0.1–0.3 s each on QCD and would dominate this script's runtime for a quantity §7.2 shows
to be bit-identical wherever the anchor is reachable at all.

### 9. `order_audit.py` reads the regenerated block from a dry run — `IMPLEMENTATION CHOICE`, forced by deviation 1

With the block not written, §9 has nothing in the tree to compare against. `--rerun-json PATH`
takes the dry run's output instead, and `ORDER-AUDIT.md` §9 prints the two commands that produce
it, so the section regenerates. Without the flag the script says so and compares like with like
rather than inventing a column.

---

## Verification performed

**Both suites, on the tree as committed.**

| | baseline (`74ddb39`) | after |
|---|---|---|
| `ComputeTargets` | 491, OK (185 s) | **491, OK** |
| `CosmologyModels` | 39, OK | **39, OK** |
| `test_convergence_reference` | 32, OK | **32, OK** |

No test is added or removed: this prompt's only two files are under `docs/`.

**`black --check`** clean on `docs/tolerance-convergence/order_audit.py` and
`docs/gktk-remedial/residual_convergence.py`.

**The three grid digests** are asserted as a precondition of every `order_audit.py` run, and held:
`RadiationModel` 2,306 / `3bef2c06`, `LambdaCDMModel` 1,778 / `60a3205a`, `QCDModel` 2,034 /
`21ffc126`, each at its own anchor. **No source-grid digest moved** (prompt §11).

**The stop was measured, not predicted.** The dry-run block was copied over the fixture, the three
reader modules were run, and the fixture was restored from a backup; `git diff` on
`ComputeTargets/tests/wkb_reference_data.json` is empty afterwards and the file's bytes are
unchanged.

**The reference converged where it matters and did not everywhere, and the document says which.**
Of the 45 (model, scheme, integrand) cases in the regenerated block, **4** report
`reference_drift_passed: false` — `LambdaCDMModel plain friction` (3.01e-16 against a threshold of
2.11e-17) and `QCDModel rho_G@1e7` under all three schemes (6.78e-21 against 5.76e-21). Both are
cases where the smallest figure the block reports is below the reference's own movement, which is
what the field exists to say. In `ORDER-AUDIT.md`'s own sweep the same thing is marked per cell
with a dagger: on the two spline models **every order from 4 upwards is at or below the reference's
drift**, so what survives is the step from order 2 and the exact-radiation control.

---

## Observations not acted on

1. **The block's reference floor is used as a test threshold, and that construction is now broken.**
   This is the stop, and it is a defect in its own right rather than a consequence of the re-run:
   `worst_production_error <= 3.0 * json_vs_reference_max_rel` divides the *model's* accuracy by
   the *reference's*, and the two are independent. They were within a factor of 3 while both sat
   at 2e-14; the reference has since improved 58× and the model has not, so they are 7× apart.
   Opened as `[04-convergence-floor-used-as-a-test-threshold]`.

2. **`rho_fixed_order_works_without_subdivision` flips `true` → `false`.** Under `plain`, no fixed
   order up to 16 now reaches the 1e-7 rad target on QCD: the ladder runs 5.44e-06, 4.93e-06,
   2.01e-06, 3.37e-07, 1.47e-06, 4.85e-07. It was `true` in the 2026-09-10 block. Nothing reads the
   field, and the change strengthens rather than weakens the case for `branch` — which reaches
   5.45e-15 at order 4 — but it is a recorded claim that has changed and the document says so.

3. **`LambdaCDMModel`'s `rho_G` reference agrees with an independent order-40 rule only to ~1e-10
   relative.** `crosscheck_gauss40_max_rel` is 8.9e-11 at $k = 10^5$, against 3e-15 for `tau` on the
   same model. $\rho_G$ on LambdaCDM is 1e-10 rad in total, so this is an absolute agreement of
   ~1e-20 rad and nothing turns on it; it is recorded because a reader comparing the cross-check
   column across quantities would otherwise read it as a defect.

4. **`RadiationModel`'s $N_F$ floor is 8.0e-14, three hundred times worse than $N_\tau$'s, and no
   order improves it.** In exact radiation the friction integrand is the constant
   $-\tfrac32(1+\tfrac13)$, so *every* Gauss order is exact and what is left is the accumulation of
   2,305 panels against a closed form evaluated through `log1p`. It is the clearest single
   demonstration in the document that the floor these knobs compete against is
   double-precision accumulation over the grid and not quadrature, which is what README §6.2's row
   claims without measurement.

5. **`convergence_reference.SCIPY_RTOL_FLOOR` fires spuriously for a quadrature too.** Every
   `DriftVerdict` the regenerated block carries notes that its `rtol` is "below SciPy's clamp and
   is silently ignored". That clamp is `solve_ivp`'s; `quad`'s floor is `50 * eps` and it *raises*
   rather than clamping, which is deviation 2. This is the same defect prompt 03a recorded for
   `brentq` in `[03a-scipy-rtol-floor-is-the-wrong-floor-for-a-root-solve]`, now seen on a third
   method, and the issue's "the floor belongs to the method, not to the pair" next step is
   unchanged. Not acted on: `convergence_reference.py` is outside this prompt's file list.

6. **The `Tk` band is never super-horizon, at any wavenumber on any model.**
   `[01-density-criterion-imposed-outside-the-wkb-region]`'s claim is entirely about the `Gk`
   sector. That halves the surface of the issue and is worth carrying into whatever decides it.

---

## State handed to the next prompt

### The five provenance fields (README §1.2), for each of the five parameters

**`TAU_GAUSS_ORDER = 4`** (`ComputeTargets/BackgroundModel.py:35`)

- **Value and what it keys:** 4. It keys **nothing** — it reaches `TAU_SOLVER_LABEL` and so
  produces a distinct `IntegrationSolver` row, but no lookup filters on `solver_serial`
  (`TOLERANCE-INVENTORY.md` §2.4). D3 proposes putting it in the `BackgroundModel` key.
- **What measurement chose it:** `ORDER-AUDIT.md` §§3.1, 5. Three models, version-2 grid at each
  cosmology's own anchor, difference error relative to the interval, cumulative from the top of the
  grid over 25 log-spaced checkpoints and per-interval over all 1,777–2,305 intervals. Order 2
  gives 6.53e-11 and order 4 gives 3.87e-16 on the exact-radiation control, where the closed form
  `tau_delta` is the reference and there is no drift. On the two spline models the reference is an
  order-32 table with drift 4.23e-16 (LambdaCDM) and 2.52e-15 (QCD), and every order from 4 up is
  at or below it.
- **Competing floor:** double-precision accumulation over the grid, **2.16e-16** relative, measured
  as the best any order in the ladder reaches. Order 4 sits 1.8× above it and order 16 does not
  improve on it.
- **Cost:** `order × panels`, counted. 1,777 / 2,036 / 2,305 panels per model, so **7,108 / 8,144 /
  9,220** evaluations at order 4, `-25 %` at order 3 and `+25 %` at order 5, times **one object per
  model** (`main.py:1062`). The whole sector's cost is one table.
- **Established by:** `prompts/tolerance-convergence` prompt 04, log 04,
  `docs/tolerance-convergence/ORDER-AUDIT.md`, 2026-09-17. It **supersedes**
  `wkb_reference_data.json`'s `convergence` block of 2026-09-10, which remains in the tree
  unwritten for the reason deviation 1 gives.

**`CS_TAU_GAUSS_ORDER = 4`** (`BackgroundModel.py:43`) — as above, with floor **3.30e-16**, order 2
at 6.53e-11 and order 4 at 3.34e-16 on the control; drift 4.29e-16 / 2.79e-15. Same panel counts
and the same one object per model. It reaches **no** label at all, so today there is no trace of it
anywhere in the store.

**`FRICTION_F_GAUSS_ORDER = 4`** (`BackgroundModel.py:44`) — floor **8.01e-14**, and the floor is
*not* quadrature: in exact radiation the integrand is constant, so every order is exact and 8.0e-14
is the accumulation of 2,305 panels against the closed form (observation 4). Order 2 already clears
on the control; 4 is selected by the two spline models, where order 2 gives 2.60e-14 (LambdaCDM)
and 3.88e-12 (QCD) against 4.06e-16 and 3.64e-16 at order 4. Same panel counts, one object per
model, no label.

**`RHO_GAUSS_ORDER = 4`** (`ComputeTargets/phase_residual.py:88`)

- **Value and what it keys:** 4. It reaches `GkWKBIntegration.PHASE_SOLVER_STEPPING` and so a
  distinct `IntegrationSolver` row, and no lookup filters on it.
- **What measurement chose it:** `ORDER-AUDIT.md` §§4, 5. **Fifty production wavenumbers**, three
  models, both sectors — 300 cases — on the version-2 grid at each cosmology's own anchor, over the
  band `residual_node_range` returns at the production margin, anchored three e-folds inside the
  horizon, in absolute radians. On the control $\rho_G$ is **bit-exactly zero at every order** and
  $\rho_T$ is scored against its closed form: 1.13e-12 at order 2 and 6.51e-17 at order 4. The
  worst over all three models is 1.53e-10 at order 2 and 6.51e-17 at order 4.
- **Competing floor:** **6.51e-17** rad, double-precision accumulation. Order 4 is at it; order 16
  does not improve on it.
- **Cost:** `order × panels`, counted; the band is 828–2,306 nodes, so **827–2,305 panels** per
  table and **265,280–461,000** evaluations per 50-table sector at order 4, exactly ±25 % at
  orders 3 and 5, with **100 tables per model** (50 $k$ × 2 sectors) memoised per worker. The
  consumers — `GkWKBIntegration` at 29,290 / 38,105 / 58,350 objects per model on the version-2
  grid, and `TkWKBIntegration` at 50 — each pay a further `order` evaluations for the off-grid
  partial that reaches their own anchor, so **order 4 costs 4 evaluations per WKB object** on top
  of the tables.
- **Established by:** as above.

**`RESIDUAL_WKB_REGION_MARGIN = 0.5`** (`phase_residual.py:238`)

- **Value and what it keys:** 0.5. In no key, label or tag whatever.
- **What measurement chose it:** `ORDER-AUDIT.md` §7. Nine margins from 0.05 to 0.9999, fifty
  wavenumbers, three models, both sectors. It is **`unchanged`** under README §6.1 **rule 6**: no
  accuracy floor exists, because the residual a producer reads is a `delta` between two fixed
  redshifts whose Gauss panels are the grid's own, so the margin can only change the answer by
  making the anchor unreachable. Bit-identical at **all 30** probes across 0.05–0.9; two of the 30
  depart at 0.9999 and only in the last bits (2e-16 relative), through the rounding of a cumulative
  whose top has moved.
- **Competing floor / bound:** not an accuracy. The bound is *reachability*: the band must reach
  the production anchor. It does at **every** margin up to **0.9** on all three models and both
  sectors, and stops doing so in the `Tk` sector between 0.9 and 0.99. `residual_node_range` never
  refuses below margin 1, where the test degenerates to "is the correction positive?".
- **Cost:** the margin sets the band's size, and the band's size is the table's panel count, so its
  cost is `order × panels` as above. Going from 0.5 down to 0.05 adds at most **99 nodes** — the
  transfer-function band on the radiation control, 828 → 927 of 2,306 — and changes no answer.
- **Established by:** as above. **It is the first measurement of this constant anywhere.**

### For prompt 05 (D3), and for prompt 06 (the provenance note)

**D3's recommendation is `replace with the orders`**, and the shape is not one column per target:
$N_\tau$, $N_{c_s\tau}$ and $N_F$ all belong to `BackgroundModel`, $N_\rho$ to the two WKB targets.
So `BackgroundModel` trades two tolerance columns for **three** integer ones and the two WKB
targets for **one** each. `ORDER-AUDIT.md` §10 costs it: four tables carry the vestigial pair
(`BackgroundModel` 18 columns, `GkWKBIntegration` 33, `TkWKBIntegration` 39, `GkSource` 13), the
rows a full run writes are 1 / 29,290–58,350 / 50 per model, and the migration is a column addition
plus a backfill in principle and a rebuild in practice, because there is no value to backfill with
that is not a guess about which order wrote the row. **`GkSource` is not this question** — it
integrates nothing, so there is no order to put there — and it is the one case where `drop` is the
whole answer.

**The live defect the option closes is measured, not hypothetical** (`TOLERANCE-INVENTORY.md`
§2.4): raise `TAU_GAUSS_ORDER` today, re-run, and the pipeline serves the order-4 row it already
holds, with a `solver_label` reading `stepping4` while the constant says otherwise.

**Prompt 05 also inherits the stop.** Whatever it does about `[04-convergence-floor-used-as-a-test-threshold]`,
the sequence is fixed: the two threshold tests must be given a bound that is not the reference's
floor *before* or *in the same commit as* the block is regenerated, and
`QCD_BREAK_POINT_ALIGNMENT_TOL` goes to just above **1.421085e-14** in that same commit.

### For whoever takes `[01-density-criterion-imposed-outside-the-wkb-region]`

It now has a measurement, and the measurement is narrower than the issue's framing.
`ORDER-AUDIT.md` §8: in the `Gk` sector 4 %–49 % of the band's nodes are super-horizon, with the
band top reaching **13.0 e-folds** outside on `RadiationModel` and `LambdaCDMModel` and **3.66** on
`QCDModel`; in the `Tk` sector it is **zero everywhere**. Rebuilding the grid with the production
function over a horizon-limited band costs **QCD 114 of 2,034 samples** and the other two
cosmologies **nothing**. So the overreach is real and larger in e-folds than the issue claims, and
its consequence is 5.6 % of one cosmology's grid.
