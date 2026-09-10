# Log 09 — Error bound, `b` column, tolerance plumbing and region guards in `QuadSourceIntegral` (B5, B6, B7, B8, B11)

**Prompt:** prompts/source-remediation/09-qsi-errors-schema-tolerances.md
**Commit:** *(this commit)* — "Record b, an error bound and honest tolerances on QuadSourceIntegral"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Opus 5
**Date:** 2026-09-09
**Result:** COMPLETE WITH DEVIATIONS

Two things a reader should not miss. **(1) `total_abserr` does not bound
`|total − analytic_rad|`** — it is a quadrature-convergence bound and knows nothing about the
error of the ingredients; the ratio the prompt asks for is 1e0–1.6e4 (exact fixtures) and
4.4e1–4.4e4 (realistic), and deviation 3 explains why that is the correct answer rather than a
failure. **(2) `total_converged` is `False` on 14 of the 36 acceptance cases** — every one of them
a realistic fixture — because the Levin driver cannot reach `rtol = 1e-8` against a re-splined
phase. That flag is now a queryable column, which is the point of B8; it is not a new defect.

## What shipped

### `ComputeTargets/QuadSourceIntegral.py` (1949 → 2122 lines)

Line numbers are post-commit unless stated.

- **B5 (audit QI-8).** The `Y3` call of `_three_bessel_Levin` (`:1211-1212` before) passed the
  module constants `LEVIN_ABSERR`/`LEVIN_RELERR` where its seven siblings pass the function's
  `atol`/`rtol`; it now passes `atol`/`rtol`. `LEVIN_ABSERR = 1e-23` and `LEVIN_RELERR = 1e-8`
  (`:61-62` before) had no other use anywhere in the repository (`grep`: only the audit, the
  campaign prompts and two earlier logs mention them) and are **deleted**, replaced by a comment
  recording what they were and why they are gone. `LEVIN_MIN_2PI_CYCLES`/`LEVIN_MIN_PHASE_DIFF`
  are untouched — they are prompt 10's decision.
- **B6 (audit QI-9).** `analytic_integral` (`:1503-1550`) now forwards its own `atol`/`rtol` to
  both `_three_bessel_integrals` calls instead of the hardwired `atol=1e-21, rtol=1e-8`, with a
  comment naming the pipeline values and pointing at the measurement below.
- **B7 (audit QI-10, QI-1).** `evaluate_QuadSource_integral` returns `"b": b`;
  `QuadSourceIntegral.__init__` reads `payload["b"]` (or `None` for a query-only object),
  `store()` reads it from the task's result payload, and a `b` property exposes it. New
  `_check_bessel_order(label, phase_data, nu)` (`:625-665`) is called on both Bessel phase dicts
  at the top of `evaluate_QuadSource_integral` (`:704-706`): it compares the splines'
  reconstruction `bessel_j(x) = m sin(theta)` against `scipy.special.jv(nu, x)` at two abscissae
  just inside the oscillatory region ($x\approx2\nu$ and $5\nu$), normalised by the local
  envelope `m`, with `BESSEL_ORDER_CHECK_TOL = 1e-3`. See deviation 1 for why the check is
  numerical, why it is here rather than in `compute()`, and how the abscissae were chosen.
- **B8 (audit QI-11).** `evaluate_QuadSource_integral` returns `"total_abserr"`
  (`= metadata["numeric_quad"]["abserr"] + metadata["WKB_Levin"]["abserr"]`, both already scaled
  by $(1+z_{\rm resp})$ — checked, not assumed: `numeric_quad_integral:1698-1699` and
  `phase_group_Levin_integral:1008-1009` each scale their `abserr` exactly as they scale their
  `value`), `"total_converged"` (every phase group of every Levin sub-interval converged) and
  `"total_phase_limited"` (any group was phase-limited). All three are persisted and exposed as
  properties, with the per-part and per-sub-interval detail left in `metadata` untouched.
- **B11 (audit QI-4).** The guard was already a width in $\log(1+z')$ after prompt 08
  (`MIN_SUBINTERVAL_LOG_WIDTH`, log 08 deviation 5), so nothing about the *test* changed. What
  this commit adds: the value is now justified in the comment (`:94-107` — 1e-7 is the tolerance at
  which this codebase treats two redshifts as equal, which is what a coincident hand-over is, and
  it is at most 1e-8 of the production range in $\log(1+z')$, i.e. at the quadrature `rtol` and
  below every representation floor on the board); a hand-over that falls inside that width is now
  **recorded** in `metadata["partition"]["skipped"]` (factor, z, the width it would have made, and
  why) instead of being merged silently; and `metadata["partition"]["min_subinterval_log_width"]`
  records the constant. B11 is closed.

### `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`

Four new columns in `register()`, each commented: `b` (`Float(64)`, **non-null**), `total_abserr`
(`Float(64)`, nullable), `total_converged` and `total_phase_limited` (`Boolean`, nullable — the
`sqla.Column("has_WKB_violation", sqla.Boolean)` pattern of `TkWKBIntegration.py:164`). All four
are added to `build()`'s `select`, to `read_batch()`'s `select`, to both payloads handed to the
constructor, and to `store()`'s insert. **`WKB_quad` and its six `WKB_quad_*` timing columns are
kept, not dropped** (deviation 2).

### `ComputeTargets/tests/test_quadsource_integral.py` (1125 → 1576 lines, 15 → 29 tests)

Five new classes, all offline: `TestErrorBound` (2 tests, 73 s), `TestTolerancePlumbing` (2),
`TestBesselOrderGuard` (2), `TestRegionGuards` (5, on new no-`bessel_phase` stand-ins
`_MinimalGk`/`_MinimalTk`/`_MinimalSource`), `TestPersistedSchema` (3). Plus `analytic_integral`,
`MIN_SUBINTERVAL_LOG_WIDTH`, `BESSEL_ORDER_CHECK_TOL` and the module object itself (fetched with
`importlib.import_module`, because `ComputeTargets.QuadSourceIntegral` as a package attribute is
the *class*) added to the imports.

### Not touched

`main.py` (prompt 10 still owes the four `Tk` payload keys), `config/sharding.py` (its
`inventory_config` entry for `QuadSourceIntegral` is `_no_validated_merge` = count and timestamps
only, and references no field name — checked), `extract_*.py`, `LiouvilleGreen/`,
`AdaptiveLevin/`, every other Datastore factory, `ComputeTargets/phase_groups.py`,
`ComputeTargets/TkSourceFunctions.py`, `docs/`.

## Deviations from the prompt

### 1. The `b` guard is numerical and lives in the task body, not in `compute()` — STRUCTURALLY REQUIRED

§2 B7 says to add the guard "in `compute()`", after checking whether the Bessel objects carry
their order and, if they do not, to "record that in the log as an observation and do not modify
`LiouvilleGreen/`".

They do not carry it: `bessel_phase()` returns `{"phase", "mod", "Q", "phi", "bessel_j",
"bessel_y", "min_x", "max_x"}` (`LiouvilleGreen/bessel_phase.py:284-293`) and no `nu`. So the
prompt's literal instruction is satisfied by the observation alone. What shipped instead is a
guard that does not need the order to be recorded: the returned dict already contains a
reconstruction of $J_\nu$ from the splines (`bessel_j = mod * sin(theta_mod_2pi)`) and its
envelope (`mod` $= \sqrt{J^2+Y^2}$), so comparing `bessel_j(x)` with `scipy.special.jv(nu, x)`
against that envelope tests the order directly. Nothing in `LiouvilleGreen/` was modified, and
the check uses only documented outputs of the dict — not, for instance, the internal relation
`min_x = sqrt(nu^2 - 1/4)`, which would also identify $\nu$ but only for $\nu>1/2$ and only by
reaching into `bessel_phase`'s internals.

It is called from `evaluate_QuadSource_integral`, not `compute()`, because in `compute()` the
splines are still `BesselPhaseProxy` objects; resolving them there would `ray.get` the whole
phase dict onto the driver once per work item (63,750 of them at production settings), which is
exactly what the proxy exists to avoid. In the task body they are already local. Cost measured:
**39 µs** per call, so 78 µs for the two splines (independent of $x_{\max}$ up to $10^6$),
against a 0.1–0.9 s integral.

Abscissae and tolerance were chosen by measurement, not by taste. Two points are used so that a
wrong order cannot escape by sitting at a node of the difference at one of them, and they are
placed just inside the oscillatory region, $x_{\rm lo} = \max(1.01x_{\min},2\nu)$ and
$2.5x_{\rm lo}$ (clamped to $x_{\max}$), because both extremes lose the check:

- deep inside the turning point ($x\ll\nu$) $J$ is exponentially small against the envelope, so
  even a wrong order moves it by less than the tolerance — at $x=10^{-4}$, $\nu=0.5$ against
  $0.7$ differs by only 8.6e-5 of the envelope, and the geometric-mean abscissa
  $\sqrt{x_{\min}x_{\max}}$ lands there for the small-$x_{\max}$ fixtures;
- far outside it, the phase spline is at its least accurate, and production builds these splines
  to $x\sim10^9$ ($\sqrt{x_{\min}x_{\max}}$ would then be $\sim3\times10^2$ and the second point
  $\sim2\times10^6$), which risks a *false* rejection that would block a run.

Measured on real `bessel_phase` splines at $\nu\in\{0.5,2.5\}$ and
$x_{\max}\in\{30,10^3,10^6,10^9\}$ — 16 combinations — with the shipped abscissae ($x=1,2.5$ for
$\nu=0.5$; $x=5,12.5$ for $\nu=2.5$):

| | residual / envelope |
|---|---|
| correct order | 7.85e-09 … 4.76e-08, **independent of $x_{\max}$** |
| order claimed 0.2 too low | 4.86e-02 … 2.73e-01 |
| order claimed 0.05 too low | 2.25e-02 … 6.39e-02 |

So `BESSEL_ORDER_CHECK_TOL = 1e-3` sits five orders above the reconstruction's own error and at
least an order below the smallest mismatch it has to catch ($\Delta b = 0.05$), with no false
positive anywhere in the sweep. In the test the $b=0$ splines presented to a $b=0.2$ integral are
rejected at **4.865e-02** of the envelope.

### 2. `WKB_quad` and its six timing columns are kept — IMPLEMENTATION CHOICE (the prompt asked for the grep)

§3 says to drop them "unless you find a reader". There is one:
`extract_QuadSourceIntegral_data.py:181` (`if obj.WKB_quad is not None and fabs(obj.WKB_quad) >
1e-25`, feeding the `WKB numeric: [z, z]` annotation built in `extract_common.py:327-345`) and
`:278,293` (a `WKB_quad` column in the exported table). That script is out of scope
(README §5 item 8), so the columns stay and the property stays. `tools/` and
`useful_queries.sql` contain no reference (grepped). Consequence: from prompt 08 onwards the
script's `WKB numeric` annotation is never drawn and its exported `WKB_quad` column is all
zeros; recorded as board issue `[09-WKB_quad-columns-are-vestigial]`. The alternative — drop the
columns and let the extract script break — was rejected because the campaign's rule is that the
extract scripts are not to be touched, so breaking one would leave no way to repair it inside
this campaign.

### 3. `total_abserr` does not bound `|total − analytic_rad|`, and the test asserts what the number does mean — STRUCTURALLY REQUIRED

§4 asks for a test that `abserr` "bounds `|total − analytic_rad|` on every prompt-08 oracle case
(report the ratio …; it should be ≲1 where the fixture floor is below `abserr`, and the log must
explain any case where it is not)". The fixture floor is above `abserr` in **every** case, so the
ratio exceeds 1 almost everywhere. The arithmetic:

| | worst ratio `|total−analytic| / total_abserr` | worst `total_abserr / |total|` |
|---|---|---|
| exact ingredients (18 cases) | **1.596e+04** | 1.22e-08 |
| realistic fixtures (18 cases) | **4.421e+04** | 1.53e-07 |

`total_abserr` is 1e-11–1.5e-7 of `|total|`, i.e. the two integrators report that they converged
to about the `rtol = 1e-8` they were given. The discrepancy against the oracle is one to four
orders larger because it is made of things neither integrator can see: with realistic fixtures,
the QuadSource spline of $f$ (4.5e-4 of the envelope, board
`[06-source-spline-residual-vs-handover]`), the LG closed forms (7e-6–1.4e-4,
`[07-lg-derivative-truncation-at-handover]`) and the hand-over clamp (5e-3,
`[08-handover-clamp-error]`); and with exact ingredients, `analytic_rad`'s *own* phase/modulus
spline floor, which the audit measured at 1e-8–1e-6 relative (QI-1) and which its reported
`abserr` explicitly excludes (QI-11). A bound on the quadrature cannot bound either.

Shipped: `test_abserr_is_positive_sums_the_parts_and_is_small` reports the ratio for all 36 cases
and asserts what is actually true — `abserr > 0`, `abserr` equals the sum of the two parts to
1e-15 relative, `abserr < 1e-5 |total|` (worst 1.53e-7: if the bound were not a small fraction of
the value, the integrator would be telling us it had failed), and the flags are booleans — plus a
second test, `test_abserr_bounds_the_quadrature_error`, which asserts the meaning of the number:
re-running each of the 18 exact cases at `rtol = 1e-11` moves `total` by **less than** the two
runs' bounds allow, worst ratio **0.653**, in every case. That is the property a quadrature error
bound has, and it is now tested.

**Consequence for prompt 12 and for anyone reading a row:** `total_abserr` is the quadrature
error, not the error. The error of `total` is `total_abserr` plus the representation floors on
the board, which at production settings are dominated by `[08-handover-clamp-error]` (~5e-3
relative), four to five orders larger.

### 4. Naming: the return-dict keys and the constant — IMPLEMENTATION CHOICE

§2 B8 says "add a top-level `abserr` to the return dict". Shipped as `total_abserr`,
`total_converged`, `total_phase_limited` — the column names — because the dict already carries
two other `abserr`s (`metadata["numeric_quad"]["abserr"]`,
`metadata["WKB_Levin"]["abserr"]`) and a bare `abserr` beside `total` would read as ambiguous in
`store()`. Similarly §2 B11 suggests `MIN_REGION_LOG_WIDTH = 1e-8`; prompt 08 had already
introduced the same quantity as `MIN_SUBINTERVAL_LOG_WIDTH = DEFAULT_FLOAT_PRECISION = 1e-7`
(log 08 deviation 5) and the board (§5 item 11) says to close B11 by confirming that rather than
by editing, so the name and value are kept and the *justification* is what was added. 1e-8 would
also have been defensible; 1e-7 is preferred because it is the same tolerance the rest of the
codebase uses to decide that two redshifts are the same redshift, which is the case the guard
exists for, and because at 1e-7 the discarded interval is still only ~1e-8 of the range.

### 5. The bottom-snap branch of the edge list is unreachable, and the test says so instead — UNINTENDED DRIFT (kept, no code change)

§2 B11 asks that sub-intervals narrower than $\delta$ be "recorded … as skipped, not silently
dropped". Writing the test for the *bottom* case showed that the branch cannot be reached from
`build_partition`'s own inputs: `_factor_breakpoint` already classifies a hand-over within
$\delta$ of `z_response` as "never oscillatory here" (`:239-241`), so the edge list never contains
a value that close to the bottom, and the `else` branch that snaps the last edge survives only as
defence against a rounding-level disagreement between the two comparisons. It is left in place
(removing it would be a change nobody asked for) and the test
`test_hand_over_within_the_width_of_the_response_redshift_is_not_a_breakpoint` asserts the
behaviour that *is* reachable: the factor is smooth throughout, its breakpoint is recorded with
`inside_range: False`, and the last sub-interval ends exactly at `z_response`. The interior case
(two hand-overs closer than $\delta$, and $q = r$, where they coincide exactly) is reachable and
is recorded — tested.

### 6. Test stand-ins for the region guards are new and minimal — IMPLEMENTATION CHOICE

§4 asks for a guard test at "$z_{\rm resp}$ near 0". The prompt-08 `Case` machinery derives
`z_response` from $x_r = r c_s a_0\eta$ and needs the LG fixtures to reach it, so it cannot
produce $z_{\rm resp} = 0$; and it costs a `bessel_phase` construction per case. The five region
tests therefore drive `build_partition` directly with `_MinimalGk`/`_MinimalTk`/`_MinimalSource`
(regions, a redshift grid and a store_id; no accessor is ever called), which run in 26 ms and
allow $z_{\rm resp} = 0$ exactly. The alternative, extending `Case`, would have meant a fixture
that reaches $z=0$ — several e-folds more LG samples per case — for no extra coverage.

## Verification performed

All from the repository root with `PYTHONPATH=.` and `./venv/bin/python`.

**`python -m unittest discover -s ComputeTargets/tests -t .`: Ran 74 tests in 210 s — OK**
(14 new; 60 before this commit).
**`python -c "import Datastore.SQL.ObjectFactories.QuadSourceIntegral"`**: clean, and the four
new columns report `b: FLOAT nullable=False`, `total_abserr: FLOAT nullable=True`,
`total_converged: BOOLEAN nullable=True`, `total_phase_limited: BOOLEAN nullable=True`.
`./venv/bin/black` reports all three touched files formatted.

**B6 — what the tolerance pass-through does to the oracle.** `analytic_integral` on the 18
exact-ingredient acceptance cases, `atol = 1e-21` (the retired literal) against
`atol = 1e-25` (`DEFAULT_QUADRATURE_ATOL`, what `main.py:2599-2602` supplies), `rtol = 1e-8`
throughout, best of three timings each:

> **`analytic_rad` is bit-identical on all 18 cases** (relative change 0.000e+00 everywhere) and
> the worst time ratio is **1.05** (median 1.00).

Same comparison against the *pre-commit* `analytic_integral` (loaded from `git show
4afd531:ComputeTargets/QuadSourceIntegral.py`), both driven at the pipeline tolerances: values and
reported `abserr`s identical to every printed digit on all 18 cases, worst time ratio 1.19
(median 1.00, and 0.83 at the other extreme — timing noise on 50–130 ms calls). So the change is
free: the analytic branch is `rtol`-limited at these magnitudes, and `1e-25` never binds where
`1e-21` did not. The prompt's "more than 2× slower with no change in value" clause does not
trigger.

**B5 — the `Y3` asymmetry.** At the shipped tolerances the fix is a no-op to the last digit,
because the retired `LEVIN_RELERR` was itself `1e-8 = DEFAULT_QUADRATURE_RTOL` and every analytic
Levin call is `rtol`-limited there: `Y3_data["abserr"]` is 6.974e-16 (`b=0`, "together",
$x_{\rm resp}=980$) and 3.668e-19 ($x_{\rm resp}=30$) both before and after, one Levin region
each, and `analytic_rad` is unchanged. The asymmetry bites when the caller's `rtol` is not 1e-8:
at `rtol = 1e-10` the whole analytic branch's reported `abserr` falls from 2.647e-19 to 4.578e-21
($x_{\rm resp}=30$) and from 4.754e-20 to 1.235e-21 ($x_{\rm resp}=980$) once every call honours
the caller (that comparison is B5 and B6 together, since the pre-commit code also hardwired
`rtol = 1e-8` for the sub-integrals). Both branches of `_three_bessel_integrals` are exercised on
these cases (`metadata["0pt5"]` carries both a `quad` and a `Levin` block), so the `Y3` line is
live, and the spy test sees all **16** analytic Levin calls — including `analytic Y3` — receive
the caller's `(atol, rtol) = (3.25e-23, 7.5e-9)` and nothing else.

**B7 — the order guard.** `b = 0` splines presented to a `b = 0.2` integral raise
`RuntimeError`: *"the supplied Bessel_0pt5 phase spline does not appear to be a Liouville-Green
representation of the Bessel function of order nu=0.7 … at x=0.041791 the spline gives
J=0.16306283 against scipy's J_nu=0.073372594, a discrepancy of **2.298e-02** of the local
envelope m=3.903 (tolerance 1.0e-03)"*. Correctly built splines pass on all 36 acceptance cases
plus every other test in the module (the guard runs on every `evaluate_QuadSource_integral`
call), i.e. no false positive at $b\in\{0,0.2\}$, three shapes, three response redshifts, both
flavours, and $x_{\max}$ from ~30 to ~1e3.

**B8 — the error bound.** Full table in the test output; the extremes:

| | `total_abserr` | `/|total|` | `|total−analytic|/abserr` | `total_converged` |
|---|---|---|---|---|
| exact, worst ratio (`b=0.2` together $x=980$) | 6.380e-23 | 1.28e-11 | 1.596e+04 | True |
| exact, best ratio (`b=0` q-smooth $x=30$) | 3.717e-19 | 4.96e-10 | 1.924e-02 | True |
| realistic, worst ratio (`b=0` together $x=30$) | 1.920e-18 | 9.63e-09 | 4.421e+04 | False |
| realistic, largest bound (`b=0.2` T-first $x=30$) | 1.970e-16 | 9.37e-09 | 2.695e+02 | True |

`abserr` equals the sum of its two parts exactly on all 36; the quad part dominates wherever a
smooth sub-interval exists and the Levin part dominates the "q-smooth" shape (where $T_q$ never
oscillates and $f$ barely does). Quadrature-error test: `rtol = 1e-8` vs `rtol = 1e-11`,
$|\Delta\,{\tt total}|$ / (sum of the two bounds) = **0.653 worst of 18** (median 6e-6).
**`total_converged = False` on 14 of the 36 cases**, all realistic, none phase-limited: the
driver cannot reach `rtol = 1e-8` against the re-splined phase, which is the representation floor
log 07 measured, not a new defect. Every exact case converges.

**B11 — the region guards.** Two hand-overs 4.000e-08 apart in $\log(1+z)$ (below
`MIN_SUBINTERVAL_LOG_WIDTH = 1.0e-07`) produce 2 sub-intervals, not 3, and one entry in
`metadata["partition"]["skipped"]` naming the factor, the redshift and the width; $q=r$ produces
one skipped entry with `log_width = 0.0`; well-separated hand-overs produce 3 sub-intervals and
an empty `skipped`; and `z_response = 0.0` exactly produces one all-smooth `quad` sub-interval
ending at `z_min = 0.0` without raising — the case the retired ratio guard would have divided by
zero on.

**Schema.** The four columns exist with the intended types and nullability (checked from
`register()` through a bare `sqlalchemy.MetaData`, no datastore); the seven `WKB_quad*` columns
still exist; a payload dict carrying the four new fields round-trips through the constructor to
the `b` / `total_abserr` / `total_converged` / `total_phase_limited` properties, and a
query-only object (`payload=None`) leaves all four `None` and raises the module's usual
`RuntimeError("value has not yet been populated")` when they are read.

**Not verified here, and needing a pipeline run (prompt 12):** that the columns write and read
back through a real `ShardedPool` (no datastore is reachable from a unit test; the insert and the
two selects are the same mechanism every other column on this table uses, and the four names
match between `register()`, `store()`, `build()` and `read_batch()` by inspection); the size of
`total_abserr` on production integrands; whether `total_converged` is `False` as often in
production as on the realistic fixtures.

## Observations not acted on

1. **`bessel_phase()` does not record its own order** (`LiouvilleGreen/bessel_phase.py:284-293`).
   The guard shipped here infers it numerically. If `LiouvilleGreen/` is ever open for edit, a
   `"nu": nu` entry in that dict would let the check become an equality test on one float, and
   would also let `BesselPhaseProxy` label itself. Out of scope (README §5 item 8).
2. **`_check_gap`'s `detail` argument is built eagerly** (`QuadSourceIntegral.py:469-486`), so a
   factor with `crossover_z is None` raises `TypeError: unsupported format string passed to
   NoneType.__format__` from the f-string before the gap is even compared, even when there is no
   gap. `_factor_breakpoint` accepts `None` (it means "smooth throughout"), and
   `TkSourceFunctions.crossover_z` is always a float in production
   (`= TkWKBIntegration.z_init`), so this is unreachable today; it was found while writing the
   `z_response = 0` test, which now passes a float hand-over instead. One `if gap > 0.0` guard or
   a lazily-built message would fix it. Prompt 08's code, not asked for here.
3. **`analytic_rad`'s stored `abserr` still excludes the phase/modulus spline fit error** (~2e-8
   relative, audit QI-11 and the reconciliation document §1.6). B8 was about `total`; nothing
   here changes the oracle's own bound, and the fixture measurements above show that this — not
   the quadrature — is what limits an exact-ingredient comparison.
4. **`docs/spec-code-audit/scripts/QI_02_analytic_numeric.py` passes `rtol=1e-10, atol=1e-25`**,
   which `analytic_integral` used to ignore and now honours, so the script's residuals should
   improve slightly. The script was not modified (README §5 item 5 and item 8); its recorded
   numbers are the pre-fix state.
5. **`metadata` is a `String(DEFAULT_STRING_LENGTH)` column** (`:206-208` of the factory) holding
   `json.dumps` of a dict that prompt 08 made considerably larger (per-sub-interval, per-group
   detail) and this commit adds `skipped` and `min_subinterval_log_width` to. Nothing here
   measured how close a production row comes to the limit. Worth a look in prompt 12: a truncated
   or rejected `metadata` write would lose the partition record, and `store()`'s `except
   TypeError` would not catch a length error.
6. **`total_converged` aggregates only the Levin groups.** `scipy.quad` reports no convergence
   flag through `simple_quadrature` (`Quadrature/simple_quadrature.py:88-106` returns `value` and
   `abserr` only), so an all-smooth integral has `total_converged = True` by construction. Its
   `abserr` is still in the bound. If a quad convergence flag is ever wanted it has to come from
   `simple_quadrature`.

## State handed to the next prompt

Prompts 10, 11 and 12 program against the following.

```python
# the task's return dict gained (prompt 09)
{
    "b": float,                     # persisted, non-null column
    "total_abserr": float,          # quadrature bound on `total`, absolute, (1+z_resp)-scaled
    "total_converged": bool,        # all Levin groups of all sub-intervals converged
    "total_phase_limited": bool,    # any group was phase-limited
    ...                             # everything prompt 08 returned, unchanged
}

# QuadSourceIntegral gained the matching properties
qsi.b, qsi.total_abserr, qsi.total_converged, qsi.total_phase_limited
```

1. **The `QuadSourceIntegral` table has four new columns** — `b` (non-null Float),
   `total_abserr` (nullable Float), `total_converged`, `total_phase_limited` (nullable Boolean).
   **Existing tables are unreadable**; this is the campaign's one schema change and the rebuild
   the board's §5 note 2 already required. `WKB_quad` and the six `WKB_quad_*` timing columns
   were **kept** (deviation 2, board issue `[09-WKB_quad-columns-are-vestigial]`), so prompt 10
   does not need to touch them and `extract_QuadSourceIntegral_data.py` still runs.
2. **`b` is carried by the task's result payload**, not by the object's constructor signature:
   `compute()` still receives it in `payload["b"]` from `main.py` and `store()` picks it up from
   the result. Prompt 10 changes nothing here. `main.py:419` still hardwires `b_value = 0.0`, and
   the Bessel splines it builds at `:429-437` use the same `b_value`, so the new order guard
   passes; a future run at $b\ne0$ that forgets to rebuild the splines now fails loudly instead
   of writing a wrong `analytic_rad`.
3. **`total_abserr` is a quadrature bound only** (deviation 3): 1e-11–1.5e-7 of `|total|` on the
   fixtures, four to five orders below the representation floors on the board. Prompt 12 must add
   the board's floors — above all `[08-handover-clamp-error]`, ~5e-3 — to it before comparing
   `total` with anything, and must not read a residual of that size as a physics defect.
   `|total − analytic_rad| / total_abserr` is 1e0–4.4e4 on the acceptance fixtures **by design**.
4. **`total_converged` is `False` on 14 of the 36 acceptance cases**, every one of them a
   realistic fixture, because `rtol = 1e-8` is below the re-splined phase's own accuracy. Expect
   the same in production and treat the column as a diagnostic, not a rejection criterion; no
   policy decision was made here (that would be audit §4.3's territory, prompt 12).
5. **Every tolerance in the analytic branch is now the caller's** (`atol_serial`/`rtol_serial`
   describe `analytic_rad`, which was B6's point). At the pipeline values this changed no number
   and cost nothing (verification above), so prompt 12 can compare against pre-09 `analytic_rad`
   values directly. `LEVIN_ABSERR`/`LEVIN_RELERR` no longer exist — do not reintroduce them.
   `LEVIN_MIN_2PI_CYCLES`/`LEVIN_MIN_PHASE_DIFF` are still there, still unused, still prompt 10's
   call.
6. **`metadata["partition"]` gained `skipped` and `min_subinterval_log_width`.** A merged
   hand-over is now distinguishable from one that never happened: `skipped` entries carry the
   factor, the redshift, the width they would have made and the reason. Prompt 12 should report
   both `skipped` and `clamp_gaps_log1pz` from production rows.
7. **B11 is closed and B5, B6, B7, B8 are discharged.** The remaining open items in this
   workstream are prompt 10's (`main.py` payload plumbing, and the
   `[08-levin-fallback-cost-ratio]` decision the user still owes) — nothing in this commit
   touches either.
8. **New test machinery** in `ComputeTargets/tests/test_quadsource_integral.py`:
   `_MinimalGk`/`_MinimalTk`/`_MinimalSource` (drive `build_partition` with no `bessel_phase`
   and no integration — useful for any further partition test), `qsi_module` (the module object,
   for spies), and the `TestPersistedSchema._table()` helper (the factory's `register()` columns
   in a bare `sqlalchemy.MetaData`, the pattern for checking any future schema change offline).
