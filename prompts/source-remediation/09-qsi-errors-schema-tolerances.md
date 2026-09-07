# Prompt 09 — Error bound, `b` column, tolerance plumbing and guards in `QuadSourceIntegral` (B5, B6, B7, B8, B11)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.3 B5, B6, B7, B8, B11; `QI-report.md` QI-4 (guards), QI-8, QI-9, QI-10, QI-11
**Depends on:** 08 (hard: the return dict and column meanings it established)
**Recommended model:** Opus
**Files you may touch:** `ComputeTargets/QuadSourceIntegral.py`,
`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` (**the only Datastore edit in this
campaign**), `ComputeTargets/tests/test_quadsource_integral.py` (extend), plus the log and the
status board.

Read first: `logs/08-qsi-phase-group-integration.md`; the factory
`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` in full (schema at `:92-208`, build/store
paths after); how `Datastore/SQL/ObjectFactories/GkSourcePolicyData.py` stores a nullable float
column, as the pattern; `prompts/backport-modules/README.md` §2.3 for the datastore's schema
conventions (registered factory classes with `@staticmethod`s).

---

## 1. Character of this commit

Five bookkeeping fixes that make the stored `QuadSourceIntegral` row honest about what it holds:
an error bound for `total`, the `b` it was computed at, tolerances that mean what the columns say,
and robust region guards. **This is the one schema change in the campaign.** Existing
`QuadSourceIntegral` tables become unreadable; that is already true after prompt 08's semantic
change and is recorded on the board (§5 note 1).

## 2. Items

### B8 — `total` gets an error bound

Prompt 08 returns a `quad` error estimate for every all-smooth sub-interval and a Levin `abserr`
for every oscillatory one. Add a top-level `abserr` to the return dict = linear sum of all
sub-interval errors, scaled by $(1+z_{\rm resp})$ consistently with the values (each contributor
should already be scaled; check). Persist it as a new nullable column `total_abserr`. Also persist
`total_converged` (bool: all Levin groups converged) and `total_phase_limited` (bool: any) — these
already exist per group in `metadata`; the columns make them queryable. Keep the rich per-group
detail in `metadata`.

### B7 — persist `b`

`compute()` receives `payload["b"]` (`:1454`) and it fixes $c_s$, the Bessel orders and the
$\eta'$ weight of `analytic_rad`, but no column records it. Add a non-nullable `b` column, set from
the payload in `store()`, and expose a `b` property. Also add a **guard in `compute()`** for the
unguarded invariant audit QI-1 noted: the `Bessel_0pt5`/`Bessel_2pt5` proxies must have been built
at the same `b` as the payload's — check whether `BesselPhaseProxy`/`bessel_phase` objects carry
their order (`LiouvilleGreen/bessel_phase.py`); if they do, assert `order == 0.5 + b` and
`2.5 + b` to `DEFAULT_FLOAT_PRECISION`; if they do not, record that in the log as an observation
and do not modify `LiouvilleGreen/`.

### B6 — `analytic_integral` must honour its tolerances

`QuadSourceIntegral.py` (post-08 line numbers will differ; the pre-08 locations were `:832-833`,
`:844-845`): the two `_three_bessel_integrals` calls hardwire `atol=1e-21, rtol=1e-8` and ignore
the function's `atol`/`rtol` arguments. Pass the arguments through. **Then check what that does
to the oracle's behaviour**: the pipeline supplies `quad_atol = DEFAULT_QUADRATURE_ATOL = 1e-25`
and `quad_rtol = 1e-8` (`main.py:2588-2591`, `config/defaults.py`). `1e-25` is tighter than the
hardwired `1e-21`; run the prompt-08 oracle test with both and report the change in `analytic_rad`
(should be ≤ its own `abserr`) and in runtime. If the tighter tolerance makes the analytic branch
more than 2× slower with no change in value beyond $10^{-10}$, keep the pass-through (the column
must be honest) but record the cost in the log for the user.

### B5 — the `Y3` call's tolerances

`_three_bessel_Levin`: seven of the eight `adaptive_levin_sincos` calls pass the function's
`atol`/`rtol`; the `Y3` call passes `LEVIN_ABSERR`/`LEVIN_RELERR` (pre-08 `:622-623`). Make it
match the other seven. Then grep for any remaining use of `LEVIN_ABSERR`/`LEVIN_RELERR`; if none,
delete the constants (pre-08 `:48-49`) and say so.

### B11 — region-nonempty guards

The guards `get_z(max_z) / get_z(min_z) > 1.0 + DEFAULT_QUADRATURE_RTOL` (pre-08 `:194, :220,
:246`; prompt 08 will have consolidated them into the partition loop) compare a ratio in $z$, so
the minimum accepted interval width varies by orders of magnitude across the range and the test
divides by zero at $z=0$. Replace with a width test in the integration variable:
$\log(1+z_{\max}) - \log(1+z_{\min}) > \delta$ with $\delta$ a named constant (e.g.
`MIN_REGION_LOG_WIDTH = 1e-8`, or derived from `DEFAULT_REDSHIFT_RELATIVE_PRECISION`); justify the
value in a comment. Sub-intervals narrower than $\delta$ are skipped and must be recorded in
`metadata["partition"]` as skipped, not silently dropped.

## 3. Factory changes

In `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`: add columns `b` (Float, non-null),
`total_abserr` (Float, nullable), `total_converged` (Boolean, nullable), `total_phase_limited`
(Boolean, nullable). Decide about `WKB_quad` and its six `WKB_quad_*` timing columns, which prompt
08 now always writes as 0/`None`: **drop them** (the table must be rebuilt anyway) unless you find
a reader — grep `extract_QuadSourceIntegral_data.py`, `tools/`, `useful_queries.sql` — in which
case keep them and record why. Update `build`/`store`/read paths and the `inventory` merge config
in `config/sharding.py` only if a *field name* it references changed (it references none of
these, but check). Confirm `QuadSourceIntegral`'s `__init__` and the factory's object construction
accept the new fields with `None` defaults for a query-only object.

`extract_QuadSourceIntegral_data.py` is **out of scope** (README §5 item 8); if dropping
`WKB_quad` breaks it, keep the column instead and open a §3 issue listing the script.

## 4. Tests

Extend `test_quadsource_integral.py`:
- `abserr` is returned, positive, and bounds `|total − analytic_rad|` on every prompt-08 oracle
  case (report the ratio `|total−analytic|/abserr`; it should be $\lesssim 1$ where the fixture
  floor is below `abserr`, and the log must explain any case where it is not).
- Tolerance pass-through: a spy/wrapper around `_three_bessel_integrals` sees the caller's values.
- Guard: a sub-interval of width below `MIN_REGION_LOG_WIDTH` is skipped and appears in
  `metadata["partition"]` as skipped; one at $z_{\rm resp}$ near 0 does not raise.
- Factory round-trip **without a datastore**: instantiate the factory's table definition and
  assert the new columns exist with the intended types (the backport campaign's prompt 04 log shows
  a pattern for constructing a `ShardedPool`-free stand-in; a pure SQLAlchemy `MetaData` check is
  enough here).

## 5. Verification

- Test suite passes; quote the error-bound ratios.
- `PYTHONPATH=. ./venv/bin/python -c "import Datastore.SQL.ObjectFactories.QuadSourceIntegral"`.

## 6. Log and commit

Log to `logs/09-qsi-errors-schema-tolerances.md`; the `WKB_quad` decision and the `δ` value are
IMPLEMENTATION CHOICEs. Board: row 09, items B5, B6, B7, B8, B11; §5 note that
`QuadSourceIntegral` tables must be rebuilt. One commit; body lists the schema changes explicitly.
