# Log 20 — Put the numeric break-point policy in the datastore lookup key

**Prompt:** prompts/GkTk-remedial/20-key-the-break-point-policy.md
**Commit:** *(this commit)* — Put the numeric break-point policy in the datastore lookup key
**Model:** Claude Opus 5
**Date:** 2026-09-13
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### 1. One declaration of the policy per sector (§2.1)

- `ComputeTargets/GkNumericIntegration.py:93-107` — the `GkNumericIntegration` class gains the
  constant `BREAK_POINT_KIND = BREAK_POINT_DISCONTINUITY`, in the shape prompt 06 established for
  the WKB phase solver (`GkWKBIntegration.PHASE_SOLVER_LABEL_BASE`,
  `BackgroundModel.TAU_SOLVER_LABEL`). The value is unchanged; only where it is written changed.
- `ComputeTargets/GkNumericIntegration.py:208-216` — new property
  `GkNumericIntegration.break_point_kind -> str`, returning `self.BREAK_POINT_KIND`. It is a
  property of the sector, not of the instance: a row read back from the datastore was selected by
  `build()` on exactly this value, so it cannot carry a different one.
- `ComputeTargets/GkNumericIntegration.py:412` — the `numeric_with_phase_cut.remote(...)` call
  site now passes `break_point_kind=self.BREAK_POINT_KIND` where prompt 19 wrote
  `break_point_kind=BREAK_POINT_DISCONTINUITY`. Prompt 19's measurement comment above it is
  untouched; four lines are added saying why it reads the constant.
- `ComputeTargets/TkNumericIntegration.py:104-118, :222-230, :443` — the same three changes, with
  `BREAK_POINT_KIND = BREAK_POINT_ALL`.

No other code in either module mentions the policy, so there is exactly one place to change it and
three places that read it.

### 2. The key field (§2.2)

Both numeric tables gain a plain string column, `nullable=False`, immediately after `rtol_serial`
and before `solver_serial` — beside the tolerances, because that is what it is:

- `Datastore/SQL/ObjectFactories/GkNumericIntegration.py:118-135` and
  `TkNumericIntegration.py:120-137` —
  `sqla.Column("break_point_kind", sqla.String(DEFAULT_STRING_LENGTH), nullable=False)`.
- `GkNumericIntegration.py:249` / `TkNumericIntegration.py:253` — `build()`'s `.filter(...)` gains
  `table.c.break_point_kind == GkNumericIntegration.BREAK_POINT_KIND` (resp.
  `TkNumericIntegration.BREAK_POINT_KIND`). **Filtered on, not merely selected**, as §2.2 requires.
  Both factories already import their compute target at module scope, so no new import was needed.
- `GkNumericIntegration.py:453-456` / `TkNumericIntegration.py:456-459` — `store()`'s insert dict
  gains `"break_point_kind": obj.break_point_kind`, i.e. the same declaration.

The stored strings are the vocabulary of `CosmologyModels/GenericEOS/GenericEOS.py`
(`BREAK_POINT_ALL = "all"`, `BREAK_POINT_DISCONTINUITY = "discontinuity"`), reached through
`ComputeTargets.BackgroundModel`'s re-export. No temperature, cosmology or equation of state
appears in `Datastore/`.

### 3. Old datastores fail loudly (§2.3)

- `GkNumericIntegration.py:279-297` / `TkNumericIntegration.py:283-301` — `build()` gains an
  `except SQLAlchemyError` arm **after** the existing `except MultipleResultsFound` (which is a
  `SQLAlchemyError` subclass, so the order matters). If `"break_point_kind"` appears in the
  message it raises `RuntimeError` naming the table, the prompt and the regeneration requirement,
  copying `Datastore/SQL/ObjectFactories/BackgroundModel.py:300-309` including its tone. Any other
  `SQLAlchemyError` is re-raised untouched.

  > `GkNumericIntegration.build(): the GkNumericIntegration table has no "break_point_kind"
  > column. This datastore predates the numeric break-point policy becoming part of the lookup key
  > (prompts/GkTk-remedial, prompt 20) and must be regenerated; there is no migration. Its rows
  > were computed under a policy that was never recorded, and on QCD_Cosmology the policies differ
  > by up to 2.8e-4 of the envelope.`

  `Datastore.SQL.Datastore._ensure_tables` creates a table only when it is absent, so an existing
  pre-prompt-20 table keeps its old columns and the query raises `OperationalError` on first use.
  No default is supplied, as §2.3 directs.

### 4. `main.py`: nothing changed (§2.4)

The two numeric `object_get` sites need nothing. The policy is not a per-call configuration: the
compute target declares it, `compute()` passes it to the integrator and the factory reads it from
the same class, so no payload key, no `tolerance` object and no menu entry is involved.
`git diff HEAD~1 --stat` does not list `main.py`.

### 5. Tests

- New `ComputeTargets/tests/test_numeric_break_point_key.py` (11 cases, 0.03 s). Tables are built
  from the factories' own `register()` output in the shape `Datastore._build_schema` builds them;
  the connection is a stand-in that raises instead of executing, so `build()` is run for real with
  no database behind it and its query captured. `store()` is likewise run for real with a
  capturing inserter.
  - `TestOneDeclarationPerSector` — each sector's constant is the vocabulary value its measurement
    chose and the two differ; the property returns the constant; the integrator call site passes
    `self.BREAK_POINT_KIND` (read with `ast`); **re-pointing the one constant moves the query
    criterion, the stored value and the accessor together**.
  - `TestTheKeyField` — the column is registered, `String`, `nullable=False`; `build()` filters on
    it *alongside* `wavenumber_exit_serial`, `model_serial`, `atol_serial`, `rtol_serial`; the two
    sectors compile to different `WHERE` clauses; a row carrying any other policy fails the
    criterion and its own passes; `store()` writes what `build()` asks for.
  - `TestOldSchemaFailsLoudly` — a missing-column `OperationalError` becomes a `RuntimeError`
    naming the table, `prompt 20`, `regenerated` and `no migration`; an unrelated
    `OperationalError` ("database is locked") propagates unchanged.
- `ComputeTargets/tests/test_numeric_break_points.py:1113-1188` —
  `test_each_production_call_site_passes_the_kind_its_sector_decided_on` follows the call site to
  the constant (deviation 1). Its substance is unchanged and slightly stronger: the site still
  names its kind explicitly, and the test now additionally asserts that the class attribute holds
  the expected vocabulary value and that the class-body declaration
  `BREAK_POINT_KIND = <name>` is written from the name imported from
  `ComputeTargets.BackgroundModel`.

## Deviations from the prompt

### 1. `test_each_production_call_site_passes_the_kind_its_sector_decided_on` had to be updated — STRUCTURALLY REQUIRED

**What the prompt assumed.** §6 says the 328 existing cases pass "none removed, none changed in
expectation", pointing at §3 item 1 (no computed value moves) as the reason.

**What was actually there.** Prompt 19's call-site test asserts that the `break_point_kind`
keyword at each production call site is an `ast.Name` whose `id` is `BREAK_POINT_ALL` /
`BREAK_POINT_DISCONTINUITY`. §2.1 of this prompt *requires* that argument to stop being a literal
("The integrator call site then passes *that*, not a literal"), and §4 names this very test as the
natural way to show §3 item 3. Every single-source-of-truth shape — a class constant, a module
constant, anything — makes the argument something other than that `ast.Name`, so the two
requirements cannot both be met as literally written.

**What was done.** The test was generalised rather than weakened: it now asserts the site passes
`self.BREAK_POINT_KIND`, *and* that the class attribute equals the vocabulary value, *and* that
the single class-body `BREAK_POINT_KIND = <name>` declaration is written from the name imported from
`ComputeTargets.BackgroundModel`. Three assertions where there were two, over the same facts plus
one more. No numerical expectation anywhere in the suite moved — §3 item 1 is measured below and
holds bit for bit — and the test's own docstring records why it moved, so a later reader is not
left guessing.

### 2. A plain string column rather than a serial to a shared vocabulary table — IMPLEMENTATION CHOICE

§2.2 left this open and asked for the alternatives.

- **Chosen: a plain `String` column on each numeric table.** The vocabulary has two members, both
  short ASCII constants that already exist in `CosmologyModels/`; the value is written once per
  row and compared for equality once per lookup. Nothing is normalised away, and the stored row is
  self-describing when read by hand or by `sqlite3` — which matters for a field whose whole
  purpose is to let a human tell two datastores apart.
- **Rejected: a serial to a small shared `break_point_kind` table.** More idiomatic for this
  codebase (`tolerance`, `IntegrationSolver`, `store_tag` all work that way) and it would make a
  typo impossible at the database level. But it buys nothing while the vocabulary has two entries:
  it adds a table, a factory, a registration, a join to both numeric `build()` queries and a
  lookup to both `store()` paths, and it puts a *cosmology* concept into `Datastore/` as a
  first-class entity, which prompts 18 and 19 were careful not to do. If the vocabulary ever grows
  ordered or structured members (a per-quantity policy, say), the serial is the right move and the
  column is a one-commit migration away — with a regeneration, which this field carries anyway.
- **Rejected: folding the policy into the solver label.** §1 rules it out, and rightly: the
  `IntegrationSolver` lookup is `label == label AND stepping >= stepping`, an ordered *quality*
  comparison, and a break-point policy is categorical.

**Not indexed**, deliberately. The column has two distinct values over ~65,000 rows per model, so
an index on it would not narrow anything the `wavenumber_exit_serial` index does not narrow
further, while costing storage and insert time. This follows `validated`, the other low-cardinality
filter column on both tables, which is likewise unindexed; every indexed filter column on these
tables is a foreign key.

### 3. The `break_point_kind` accessor reads the class, not stored state — IMPLEMENTATION CHOICE

The alternative was to carry the value read back from the row on the instance (as `_atol` and
`_rtol` are carried), so that a deserialized object reports what it was actually computed under.
That was rejected because `build()` filters on the class constant: a row that does not match is
not returned, so the only value an instance can ever hold *is* the class constant, and storing a
second copy on the instance would create precisely the drift this prompt exists to remove. The
consequence is deliberate and worth stating: an object cannot tell you it came from a datastore
written under a different policy — the `RuntimeError` of §2.3 and the miss of §2.2 are what tell
you that.

## Verification performed

### §3 item 1 — no computed value moves (run, not reasoned)

A production `QCD_Cosmology` object was run in **both sectors** at
$k = 4.972\times10^7/{\rm Mpc}$ — the wavenumber at which `TK-NUMERIC-ATOL-SWEEP.md` §10.5
measures the worst shift between the two policies — on this tree and on a worktree at `HEAD~1`
(`fa87cd8`), with the same driver script in both. Production tolerances and grids
(`atol = 1e-13` for $T_k$ / `1e-10` for $G_k$, `rtol = 1e-8`, `delta_logz = 1/100`, `mode="stop"`,
$T_k$ on the source grid and $G_k$ on the response grid). The script resolves the break-point kind
the way production resolves it — from the class constant where one exists, and otherwise from the
module-level name the old call site used — and records the resolved string in its output, so the
comparison covers the resolution as well as the numbers. Every sample, both `stop_value` and
`stop_deriv`, `stop_deltaz_subh`, `has_unresolved_osc`, `RHS_evaluations` and `compute_steps` are
dumped as hex floats.

```
HEAD~1 (fa87cd8):  Tk: kind=all  samples=494 RHS=32351 steps=32351
                   Gk: kind=discontinuity samples=41 RHS=13258 steps=13258
this tree:         Tk: kind=all  samples=494 RHS=32351 steps=32351
                   Gk: kind=discontinuity samples=41 RHS=13258 steps=13258

cmp base.json new.json  ->  BIT-IDENTICAL (30,434 bytes)
sha256 both: 565e907450f8afb93741e9044a6aa98e61668ebd61cc32e0212cf971f2c6570f
```

This is what one expects of a diff that does not touch `Quadrature/` at all, but it was run rather
than assumed, as §6 requires.

### §3 item 2 — a lookup under the wrong policy misses, under the right policy hits

Compiled with literal binds, with no database behind it:

```
Gk build(): ... AND "GkNumericIntegration".break_point_kind = 'discontinuity'
Tk build(): ... AND "TkNumericIntegration".break_point_kind = 'all'
```

`TestTheKeyField.test_a_row_stored_under_the_other_policy_misses_and_its_own_hits` takes the bound
value out of each sector's `WHERE` clause and applies the equality to candidate stored values: the
sector's own policy matches, the *other sector's* policy (which is what a QCD datastore spanning
prompt 19 holds) does not, and a foreign string does not.

### §3 item 3 — the two uses cannot drift

`TestOneDeclarationPerSector.test_repointing_the_constant_moves_the_query_and_the_stored_value_together`
re-points `BREAK_POINT_KIND` on each compute target to a sentinel and finds the sentinel in all
three places: the `build()` filter's bound value, the dict `store()` inserts, and the accessor.
`test_the_integrator_call_site_reads_the_class_constant` shows the fourth (the integrator argument)
with `ast`.

### §3 item 4 — an old-schema datastore raises

`TestOldSchemaFailsLoudly` drives each production `build()` against a connection raising the
`OperationalError` SQLite produces for a missing column
(`no such column: GkNumericIntegration.break_point_kind`) and asserts the resulting `RuntimeError`
names the table, `prompt 20`, `regenerated` and `no migration`. A connection raising
`"database is locked"` still propagates an `OperationalError`.

### The suite

```
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
before:  Ran 328 tests in 155.426s   OK
after:   Ran 339 tests in 155.860s   OK
```

328 → 339: eleven new cases, none removed, no numerical expectation changed. The one existing
case whose *structural* expectation moved is deviation 1.
`ComputeTargets/tests/test_tk_wkb_phase.py::TestCost::test_wall_time_per_object`, the
machine-load-sensitive wall-clock budget, passed.

`black --check` clean on all six touched files. `git diff HEAD~1 --stat` touches
`ComputeTargets/{Gk,Tk}NumericIntegration.py`,
`Datastore/SQL/ObjectFactories/{Gk,Tk}NumericIntegration.py` and the two test modules — **not**
`Quadrature/`, **not** `config/defaults.py`, **not** `CosmologyModels/`, **not** `main.py`, and no
`IntegrationSolver` or `solver_serial` code.

### The audit the user asked for (§5, report only — nothing changed)

The scope decision was "all five compute targets". Under the design chosen — key the
configuration, leave the solver as provenance — the finding is **mixed**: the other four need no
key change *for the break-point policy*, because they do not run the numeric ODE and have no such
argument; but the general claim that they have "no equivalent free parameter" is **refuted**. Each
of the three has at least one configuration axis that can vary between runs and is in no lookup
key. Issues opened, not acted on.

**`BackgroundModel`** — key: `validated`, `cosmology_type`, `cosmology_serial`, `atol_serial`,
`rtol_serial`, plus the `z_init_serial` filter when supplied and a join per supplied `store_tag`.
`main.py:595-602` supplies `LargestSourceZTag`, `SmallestSourceZTag` and
`SourceSamplesPerLog10ZTag`, so the **node grid the tables are built on is keyed**, through the
tag mechanism. `solver_serial` is stored (`:404`), selected for its label (`:185`) and not
filtered. Its solver is hard-coded in the numeric one's manner but not in its form:
`TAU_SOLVER_LABEL = f"{TAU_SOLVER_LABEL_BASE}-stepping{TAU_GAUSS_ORDER}"`
(`ComputeTargets/BackgroundModel.py:47-48`), no solver argument anywhere, and — unlike
`"solve_ivp+DOP853-stepping0"` — the label *would* move if the Gauss order moved. **Unkeyed axes:**
`TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` (all `= 4`,
`BackgroundModel.py:34-43`) and the quadrature break-point scheme the cumulative tables use
(`BREAK_POINT_ALL`, `integration_break_points`'s default). Changing any of them moves every stored
`tau_Mpc`/`cs_tau_Mpc`/`friction_F` without moving anything in the key.

**`GkWKBIntegration`** — key: `validated`, `wavenumber_exit_serial`, `model_serial`, `atol_serial`,
`rtol_serial`, plus `z_source_serial` and `|z_init − z_init| < DEFAULT_FLOAT_PRECISION` when
supplied, plus tags. `solver_serial` stored and not filtered. `atol`/`rtol` are in the key but
have had no referent since prompt 06 (`RECONCILIATION.md` §2 item 10). Its solver is hard-coded at
`Quadrature/integrators/WKB_phase_function.py:85-87`. **Unkeyed axes:** `RHO_GAUSS_ORDER = 4`
(`ComputeTargets/phase_residual.py:90`), which at least moves `PHASE_SOLVER_LABEL`, and
`RESIDUAL_WKB_REGION_MARGIN = 0.5` (`:238`), which moves **nothing** observable in the row — no
label, no tag, no column. `G_init`/`Gprime_init` are stored `nullable=False` and not filtered.

**`TkWKBIntegration`** — the same key. It has the two solver columns the prompt names,
`phase_solver_serial` and `friction_solver_serial`; since prompt 07 both carry `PHASE_SOLVER_LABEL`,
because phase and friction come from the same primitive. Neither is filtered on. Same unkeyed
axes, plus the background's three Gauss orders at one remove. `T_init`/`Tprime_init` stored and not
filtered.

**The WKB-rows-consume-numeric-values hazard: real in principle, and incidentally covered here.**
The WKB stage takes its initial data from the numeric stop point —
`z_init = k_exit.z_exit − Tk.stop_deltaz_subh`, `T_init = Tk.stop_T`,
`Tprime_init = Tk.stop_Tprime` (`main.py:951-953`; the $G_k$ twin at `:1589`) — and those rows are
keyed independently of the numeric row they came from: there is no foreign key, and `T_init` /
`Tprime_init` (`G_init` / `Gprime_init`) are stored but **not** filtered on. What does protect
them is `z_init`, which *is* filtered, as
`|z_init − stored| < DEFAULT_FLOAT_PRECISION = 1e-7` **absolute** against a $z_{\rm init}$ of order
$10^{11}$–$10^{13}$ — nineteen orders below its own ulp, so in production that comparison is exact
equality and any movement at all in the stop point makes every downstream WKB lookup miss.
Measured on `QCD_Cosmology` at $k = 4.972\times10^7$/Mpc, `BREAK_POINT_ALL` against
`BREAK_POINT_DISCONTINUITY`:

| quantity | `all` | `discontinuity` | shift |
|---|---|---|---|
| $z_{\rm init}$ | 1044583216739.4531 | 1044582758044.2109 | 4.59e+05 (4.4e-07 relative) |
| $T_{\rm init}$ | 0.010439856922658777 | 0.01043947282117206 | 3.84e-07 |
| $T'_{\rm init}$ | −3.418100100402973e-18 | −3.4179628382425886e-18 | 1.37e-22 |

so at this wavenumber the $T_k$ WKB lookup misses, correctly. **But the protection is incidental**:
it comes from the stop point happening to move, not from the WKB key recording anything about the
numeric row. A policy (or tolerance) change that moved the stop *values* while leaving
$z_{\rm init}$ bit-identical would be served a stale WKB row. **For prompt 13 the practical
statement is unchanged and simple: regenerate the whole QCD chain**, which is what prompt 19's
state hand-off already says. Opened as `[20-wkb-rows-consume-numeric-initial-data]`.

## Observations not acted on

1. **The three non-numeric compute targets have unkeyed configuration axes** — the four Gauss
   orders and `RESIDUAL_WKB_REGION_MARGIN` above. `RESIDUAL_WKB_REGION_MARGIN` is the sharp one:
   it is in no label, no tag and no column, so a change to it is invisible everywhere. Prompt 14
   measured its effect as $\le1.4\times10^{-17}$ rad in $\rho$ with $\theta$ bit-identical, so the
   hazard is latent rather than live today. Opened as
   `[20-wkb-gauss-orders-not-in-lookup-key]`. Related to, but distinct from,
   `[03-integrationsolver-stepping-minimum-lookup]`, which is about the `IntegrationSolver` lookup
   itself rather than about who filters on its serial.
2. **WKB rows are keyed independently of the numeric rows they consume**, as measured above.
   Opened as `[20-wkb-rows-consume-numeric-initial-data]`.
3. **`validate_on_startup` does not see the new column**, so an old-schema datastore passes
   startup validation and fails on the first `build()`. That is the same shape as prompts 03/04's
   `BackgroundModel` check and is left alone; moving the detection earlier is a `Datastore/`
   design question, not this prompt's.
4. **`atol`/`rtol` remain in `BackgroundModel`'s and both WKB targets' lookup keys with no
   referent.** `RECONCILIATION.md` §2 item 10 made the call for the WKB pair (keep them, schema
   churn for no gain); the same is now true of `BackgroundModel` since prompt 03. Not a hazard —
   an unnecessary key field over-discriminates, it does not under-discriminate — and recorded only
   so that a later reader does not take their presence as evidence that the build honours them.
5. **`[19-cosmologymodels-docstrings-predate-per-sector-policy]` is still open** and this commit
   does not touch `CosmologyModels/`.

## State handed to the next prompt

**Prompt 13 must build a fresh datastore, and now finds out if it does not.** What has to be
regenerated is unchanged from prompt 19's hand-off — every `TkNumericIntegration` row on
`QCDModel` and everything downstream of it, plus everything prompt 18 named — but the *failure
mode* has changed: a datastore written before this commit has no `break_point_kind` column, so the
first `GkNumericIntegration` or `TkNumericIntegration` lookup raises a `RuntimeError` naming the
prompt and demanding regeneration, instead of silently returning a row computed under an unknown
policy. There is no migration and no default. A datastore built **after** this commit may be kept
across any future change of policy: the policy is now in the key, so a change of policy simply
misses and recomputes, the way a change of `atol` does.

**The shape of the key field and the constant, both public:**

```python
# ComputeTargets/GkNumericIntegration.py
class GkNumericIntegration(DatastoreObject):
    BREAK_POINT_KIND = BREAK_POINT_DISCONTINUITY      # NEW, class constant

    @property
    def break_point_kind(self) -> str: ...            # NEW, returns BREAK_POINT_KIND

# ComputeTargets/TkNumericIntegration.py
class TkNumericIntegration(DatastoreObject):
    BREAK_POINT_KIND = BREAK_POINT_ALL                # NEW, class constant

    @property
    def break_point_kind(self) -> str: ...            # NEW

# Datastore/SQL/ObjectFactories/{Gk,Tk}NumericIntegration.py
sqla.Column("break_point_kind", sqla.String(DEFAULT_STRING_LENGTH), nullable=False)
# ... placed after rtol_serial, filtered on in build(), written from obj.break_point_kind in store()
```

To change a sector's policy, change its one class constant: the integrator argument, the stored
value and the query criterion all follow, and old rows then miss rather than being served.

**Nothing computed moved.** A production QCD object in both sectors is bit-identical to `HEAD~1`
(sha256 `565e907…`, 494 + 41 samples, 32,351 + 13,258 RHS evaluations). `Quadrature/`,
`config/defaults.py`, `CosmologyModels/`, `main.py`, every tolerance and every `solver_serial` are
untouched, and prompt 19's break-point numbers stand unchanged.

**§5 turned up two second instances, both opened and neither acted on.**
`[20-wkb-gauss-orders-not-in-lookup-key]` — `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`,
`FRICTION_F_GAUSS_ORDER`, `RHO_GAUSS_ORDER` and `RESIDUAL_WKB_REGION_MARGIN` are configuration
axes in no lookup key; the last is in no label or tag either, so it is invisible.
`[20-wkb-rows-consume-numeric-initial-data]` — the WKB targets take their initial data from the
numeric stop point and are keyed independently of it; `z_init` is filtered as an absolute 1e-7
comparison against $z\sim10^{12}$, i.e. exactly, which covers the case **measured here**
($z_{\rm init}$ moves by 4.59e5 between the two policies at $k = 4.972\times10^7$/Mpc, so the
lookup misses), but a change that moved `T_init` without moving $z_{\rm init}$ would be served a
stale row. Neither blocks prompt 13, whose remedy for both is the fresh datastore it was already
going to build.

**`[18-numeric-solver-not-in-lookup-key]` is resolved** (§4 of the board), with its recorded
"next step" corrected in place: adding `solver_serial` to the two queries is a **no-op**, because
the solver is hard-coded and every numeric row points at the same `IntegrationSolver` serial. What
was unkeyed was the configuration, and it now is keyed.
