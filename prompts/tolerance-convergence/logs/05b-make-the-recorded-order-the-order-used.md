# Log 05b — Make the recorded order the order that was used

**Prompt:** `prompts/tolerance-convergence/05b-make-the-recorded-order-the-order-used.md`
**Commit:** *(this prompt's own commit)* — "Make an object report the Gauss order it was built at"
**Model:** Opus 5
**Date:** 2026-09-18
**Result:** **DONE.** An object reports the Gauss order it was **built at**, on both paths by which
it can come into existence. On the compute path the order travels as data: `WKB_phase_function`
carries the residual table's own `CumulativeTable.order` out in its payload and `store()` records
it; `compute_background` already echoed its three orders and `BackgroundModel.store()` now records
those. On the rehydration path each factory's `build()` **selects** its order column(s) and hands
them to the constructor, and `BackgroundModel._build_tau_primitive` / `_build_cs_tau_primitive` /
`_build_friction_F_primitive` reassemble their cumulative tables at the **row's** orders rather
than at the module constants. `build()` still filters on the current module constants, unweakened.
**No number changes**: `config/defaults.py` is byte-identical (`git diff` empty) and all four
orders are still **4**. **No computed value moves**: a SHA-256 over every double the production
path produces is identical at `90d0114` and here. `main.py` is untouched, zero lines.
`ComputeTargets` **516, OK**; `CosmologyModels` **39, OK**;
`ComputeTargets/tests/test_numeric_break_point_key.py` passes **unedited**; the three published
source-grid digests are unmoved. **8 production files** in the diff, and no schema change.

`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]` is **closed**, both
the half it names and the rehydration half it does not. Board item **T15**.

---

## 1. The mechanism

Prompt 05 reached each order through **one name resolved at call time**: the compute path read the
module constant, the object's property re-read it, `store()` wrote that property and `build()`
filtered on the same module attribute. That is right about the key and wrong about the object. A
property that re-reads a constant reports what the module *currently says*, not what the object
*is*.

Since this commit the order is **data the object carries**, and the four sites are these:

| path | how the order reaches the object |
|---|---|
| the WKB computation | `cached_phase_residual` builds the table; `WKB_phase_function` reads `rho.order` off that very table into `metadata["N_rho"]` **and** into a payload key `rho_gauss_order`; `{Gk,Tk}WKBIntegration.store()` records `data["rho_gauss_order"]` |
| the background computation | `compute_background` already echoed `tau_order` / `cs_tau_order` / `friction_F_order` from the three `CumulativeTable` constructions; `BackgroundModel.store()` now records those three echoes |
| WKB rehydration | the factory's `build()` **selects** `rho_gauss_order` and passes it as a payload key; `__init__` stores it |
| background rehydration | `build()` **selects** the three order columns and passes them as payload keys; `__init__` stores them, and the three `_build_*_primitive` methods construct their `CumulativeTable` at `self.tau_gauss_order` &c. |

**How the stored order reaches a rehydrated object, given that `_metadata` is not a channel.**
Prompt §3 is right that metadata cannot carry it: `GkWKBIntegration` sets `self._metadata = None`
on the unpopulated path, and on the populated build path the value comes from a **nullable**
`String` column parsed as JSON, so a row whose metadata was never written would hand back `None`.
The order is therefore a payload key of its own, beside `"solver"` and `"values"` — the shape the
`atol`/`rtol` pair had before prompt 05 removed it, which is what §3 points at as the tree's own
precedent. `BackgroundModel` takes the same shape for its three. The column was already in the
schema in both cases; what was missing was that **`build()` selected it and threw it away**
(`BackgroundModel`'s query did not even select the three — see deviation 2).

**A consequence, stated because it is the point of the prompt:** with the record faithful, a table
built at 6 stores 6, and a run configured at 4 simply never sees that row. That is **correct**
behaviour and not a miss. `build()`'s filter is the lookup semantics — *give me a row computed at
the order this run is configured for* — and once the record is a property of the object rather than
a re-reading of the module, the filter and the record no longer have to agree by luck. They agree
because the row says what it is and the query says what it wants.

**What an object with no order does.** A query-shaped or not-yet-computed object has built nothing
and came from no row, so it has no order to report and each accessor raises `RuntimeError` naming
the two paths that give it one. Before this commit it answered with the module constant, which is
the defect in miniature.

## 2. The decision on the `order=` default — kept, as a call-time sentinel

Prompt §2 leaves this open and §4 constrains it. The three `phase_residual` entry points now take
`order: Optional[int] = None` and resolve `RHO_GAUSS_ORDER` **inside the function body**.

Not *required*, which was the "next step" the issue itself proposed, because §4's six scripts
across three campaigns call these entry points without an order and README §5 rule 7 protects
their output, not their right to be rewritten by this prompt: five of them
(`grid_density_criterion.py:214`, `consumer_knot_scheme_scan.py:423`,
`verify_production_path.py:735`, `:947`, `:1061`) would have had to be edited, which §8 makes a
stop. Not *deleted*, because `docs/tolerance-convergence/order_audit.py:864` sweeps the order and
prompt 04 built it to.

And not left as `order: int = RHO_GAUSS_ORDER`, which is the third option and the one that looks
like no change at all. A default argument is evaluated **once, at `def` time**, so that spelling
snapshots the constant when the module is imported: re-point the declaration and the key column
moves while the table goes on being built at the old order — the divergence inverted, and the one
the §6 test below catches at `90d0114`. The sentinel resolves the same single declaration when the
call is made, so the default *is* the order the run is configured at, and whatever the caller does
supply is what the table is built at and what the object records. Against the §2 invariant that is
the honest reading: the parameter is no longer a way to persist an order you did not use, because
the persisted value is read off the table.

`Quadrature/integrators/WKB_phase_function.py` gained the same correction in miniature: it reached
`RHO_GAUSS_ORDER` through a `from ... import`, which is an import-time snapshot with the same
defect, and now reads it through the module. That matters only on the zero-length branch, where no
residual table is built at all and the order reported is the one the call was configured at.

## 3. What shipped

**8 production files, 5 test files, and no schema change.**

- **`ComputeTargets/phase_residual.py`** — the sentinel default on `build_phase_residual`,
  `phase_residual_cache_key` and `cached_phase_residual`, resolved at call time in each, with the
  note above written where the parameter is documented. The cache key resolves it too, so a caller
  who passes nothing and a caller who passes the current constant land on the same key.
- **`Quadrature/integrators/WKB_phase_function.py`** — the module import in place of the name
  import; `metadata["N_rho"]` is now `int(rho.order)`, the table's own; a new payload key
  `rho_gauss_order`, `int(rho.order)` on the main path and the call-time constant on the
  zero-length branch, where nothing is tabulated. No change to the phase algorithm, the anchor,
  the node selection or the leading term.
- **`ComputeTargets/{Gk,Tk}WKBIntegration.py`** — `self._rho_gauss_order`, `None` on the
  unpopulated path, from `payload["rho_gauss_order"]` on the build path, from
  `data["rho_gauss_order"]` in `store()`; the property reports it and refuses when it is absent.
  The now-unused `import ComputeTargets.phase_residual` is gone from both.
- **`ComputeTargets/BackgroundModel.py`** — the same three ways round for the three orders, plus
  the three `_build_*_primitive` methods, which now pass `self.tau_gauss_order`,
  `self.cs_tau_gauss_order` and `self.friction_F_gauss_order` to their `CumulativeTable`. No
  parameter is added to `compute_background`, which §3 says is already correct.
- **`Datastore/SQL/ObjectFactories/{Gk,Tk}WKBIntegration.py`** — `table.c.rho_gauss_order` added
  to `build()`'s **select** list and passed into the constructor payload; the `store()` comment
  corrected, since the value is now the object's own and no longer "never a payload value". A
  `SCHEMA NOTE` records what changed and, in terms, that `build()` still filters on the constant.
- **`Datastore/SQL/ObjectFactories/BackgroundModel.py`** — the same, for three columns that
  `build()` **did not select at all** (deviation 2).

**Tests.** `ComputeTargets/tests/test_gauss_order_key.py` gains **8** assertions in two new classes
and has four repaired; four other modules are repaired for the constructor's new payload keys.
See §4 and §5.

## 4. The two tests this prompt turns on, and their failures at `90d0114`

Both were run against a worktree at `90d0114` with **only this test module** copied in (one line
adapted so the module could import at all there: the stand-in borrows `BackgroundModel._order`,
which does not exist at that commit).

**1. A rehydrated object reports its row, not the module.**
`TestAnObjectReportsTheOrderItWasBuiltAt` drives each production `build()` against a canned row —
no SQLite, no engine — carrying orders no run is configured at (`tau` 6, `cs_tau` 8, `friction_F`
12, `rho` 6). At `90d0114`:

```
FAIL: test_a_rehydrated_background_model_reports_its_row_and_not_the_module
    self.assertEqual(obj.tau_gauss_order, ROW_ORDERS["tau"])
AssertionError: 4 != 6

FAIL: test_a_rehydrated_background_model_rebuilds_its_tables_at_its_row_order
    self.assertEqual(obj._build_tau_primitive().table.order, ROW_ORDERS["tau"])
AssertionError: 4 != 6

FAIL: test_a_rehydrated_wkb_object_reports_its_row_and_not_the_module (GkWKBIntegration)
FAIL: test_a_rehydrated_wkb_object_reports_its_row_and_not_the_module (TkWKBIntegration)
AssertionError: 4 != 6

FAIL: test_an_object_with_no_order_refuses_rather_than_reporting_the_module
AssertionError: RuntimeError not raised
```

`test_build_still_filters_on_the_module_constant`, in the same class, **passes at `90d0114` and
here**, deliberately: it is the invariant this prompt does not weaken.

**2. A table built at a non-default order is recorded at that order.**
`TestTheRecordedOrderIsTheOrderTheTableWasBuiltAt` drives the undecorated `WKB_phase_function` on
the exact-radiation stand-in with the declaration re-pointed to 6, then asserts that the value the
production `store()` writes is the order the table was actually built at (the table is recovered
from the cache and the test asserts the hit, so it is the same object the run used). At
`90d0114`:

```
ERROR: test_the_payload_carries_the_table_s_own_order (order=4)
    self.assertEqual(payload["rho_gauss_order"], table.order)
KeyError: 'rho_gauss_order'

FAIL: test_the_payload_carries_the_table_s_own_order (order=6)
    self.assertEqual(table.order, order)
AssertionError: 4 != 6

ERROR: test_store_writes_the_order_the_table_was_built_at
    obj._rho_gauss_order = int(payload["rho_gauss_order"])
KeyError: 'rho_gauss_order'
```

The `4 != 6` there is the def-time default binding of §2, caught directly: the constant moved, the
key column moved with it, and the table did not.

**Repaired, where they asserted the old mechanism as though it were the invariant:**

- `test_repointing_the_declaration_moves_the_query_and_the_stored_value_together` loses its third
  leg, `_class_accessor`, which called the property on the *class* — possible only while the
  getter read a module constant and nothing else. The query leg and the `store()` leg stand
  unchanged, and the stand-ins now read the constant at construction, which is what a freshly
  computed object does.
- `test_the_phase_integrator_records_the_declared_order` asserted by `ast` that `"N_rho"` is
  written as the bare name `RHO_GAUSS_ORDER`. It is now `rho.order`. The replacement,
  `test_the_phase_integrator_reads_the_order_through_its_module`, asserts the correction the
  module needed instead: the constant is not name-imported and every read of it goes through
  `phase_residual.`.
- `test_no_production_caller_overrides_the_residual_order` keeps its assertion and loses its
  premise: a caller that supplies `order` is no longer a latent defect, so the test is a statement
  about the production path. Beside it,
  `test_the_residual_entry_points_resolve_their_order_at_call_time` asserts the sentinel on all
  three entry points.
- `test_compute_background_builds_each_table_at_its_own_declared_order` still counts three `ast`
  occurrences of each constant; the comment says which three they now are.

## 5. Deviations from the prompt

### 1. Five test modules outside the prompt's named list were repaired — `STRUCTURALLY REQUIRED`

The prompt's file list is `test_gauss_order_key.py` "and any other test module that asserts on the
orders". These construct the objects whose constructor changed, or assert the payload whose shape
changed, and each repair is the addition of the order the fixture was already built at:

- `test_background_tau.py` and `test_background_cs_tau_friction.py` — three `_offline_model`-style
  constructions whose docstrings say "populated the way `store()` populates it". They hold the
  `compute_background` payload, so the repair passes that payload's own `tau_order` /
  `cs_tau_order` / `friction_F_order` through. Nothing 04b put in either module is disturbed and no
  threshold moves.
- `test_run_identity.py` — `write_model` populates the private fields of an uncomputed object to
  stand in for a computed one; it now sets the three orders from the module constants, beside the
  `_solver` and `_data` it already set.
- `test_retire_samples_per_decade_tag.py` — **not edited**; it uses `test_run_identity`'s harness
  and was repaired by that change. It is listed here only because it appeared in the failure set.
- `test_gk_wkb_phase.py` — `TestPayloadContract.REQUIRED_KEYS` is an exact set equality on the
  phase payload's keys, so the new key had to be added to it. That is the test doing its job.

### 2. `BackgroundModel.build()` did not *select* its three order columns at all — `STRUCTURALLY REQUIRED`

Prompt §1 (b) says the three columns "are selected by `build()` (`:340-342`) and never passed to
the constructor". Read against the tree, `:340-342` is the **missing-column detection tuple** in
the `SQLAlchemyError` handler, not the select list: the query selected none of the three. Passing
the row's orders to the constructor therefore required adding them to the select list as well.
Selecting a column that already exists is not a schema change — `register()` is untouched in all
six factories, and `test_gauss_order_key.py`'s schema assertions are unchanged and pass — but it is
more than the prompt's sentence describes, so it is recorded. The two WKB factories did select
theirs.

### 3. The accessors raise rather than returning `None` when there is no order — `IMPLEMENTATION CHOICE`

An object that has computed nothing and came from no row has no order. It could report `None` and
let the factory write a NULL into a `nullable=False` column. It refuses instead, naming the two
paths that give it one, which is the pattern `metadata`, `stage_1_data` and `has_WKB_violation`
already use on these classes for "not populated yet". `BackgroundModel` routes its three through
one `_order` helper so that the message names the accessor that was read.

### 4. `import ComputeTargets.phase_residual` was removed from both WKB compute classes — `IMPLEMENTATION CHOICE`

Once the property stops reading `phase_residual.RHO_GAUSS_ORDER` the import is unused in both
files. Nothing in the tree reaches the constant through those modules (`grep` for
`GkWKBIntegration.phase_residual` and `TkWKBIntegration.phase_residual` finds nothing), and the
factories import the module for themselves.

## 6. Verification performed

| Check | Result |
|---|---|
| `ComputeTargets` suite | **516 tests, OK** (prompt §7 item 7: must not fall below 508 and should rise; 508 at `90d0114`, plus the 8 added to `test_gauss_order_key.py`, which now holds 24) |
| `CosmologyModels` suite | **39 tests, OK** |
| the two §6 tests at `90d0114` | both fail, 6 failures and 2 errors, quoted in §4 |
| `ComputeTargets/tests/test_numeric_break_point_key.py` | passes **unedited**; `git diff` on it is empty |
| **no computed value moves** | a SHA-256 over 2,296 doubles produced by the production path — `compute_background` on LambdaCDM over a 200-node grid (both $\tau$ limbs, both $c_s\tau$ limbs, $F$, $H$, $\rho$ and the three order echoes), and `WKB_phase_function` in **both** sectors on the exact-radiation control at $k=10^7$ over 95 samples (`theta_div_2pi`, `theta_mod_2pi`, the friction samples, and both residual tables' `hi` and `lo` limbs) — is **`3715ef02…dbcd14` at `90d0114` and `3715ef02…dbcd14` here**. Run from the same script in a worktree at `90d0114` and in this tree, bit for bit |
| source-grid digests | `4849552b`, `60a3205a`, `3bef2c06`, `21ffc126` asserted by `test_source_grid.py` and `test_convergence_reference.py` inside the suite |
| `config/defaults.py` | `git diff -- config/defaults.py` **empty**; `TAU_GAUSS_ORDER` 4, `CS_TAU_GAUSS_ORDER` 4, `FRICTION_F_GAUSS_ORDER` 4, `RHO_GAUSS_ORDER` 4 |
| `main.py` | `git diff -- main.py` **empty** — zero lines |
| schema | `git diff` touches no `register()`; `test_gauss_order_key.py`'s three schema tests pass unchanged, `GkSource` included, `break_point_kind` included |
| the six §4 scripts | `python -m py_compile` clean on all of them; all five that pass no order keep working at run time, because the sentinel resolves to the same constant their def-time default did |
| `black --check` | clean on every `.py` in the diff |
| untouched, by `git diff --stat` | `config/defaults.py`, `main.py`, both numeric targets and their factories, `CosmologyConcepts/wavenumber.py`, `QuadSourceIntegral`, `test_numeric_break_point_key.py`, `docs/` — all empty |

## 7. Observations not acted on

1. **`BackgroundModel.TAU_GAUSS_ORDER` and `WKB_phase_function.PHASE_SOLVER_STEPPING` remain
   import-time snapshots** of their declarations, used to build `IntegrationSolver` labels
   (`main.py:3575`, `:3584`). Log 05's observation 3 records them as not-a-defect, and that stands:
   each is assigned from the declaration, so no source edit can make them disagree, and the label
   is not a key. They do not follow a runtime monkeypatch, which is why the tests here patch the
   declaration and drive the code rather than reading those attributes. Repairing them would mean
   editing `main.py`, which §8 forbids.
2. **`phase_residual_cache_key` is public and now takes a sentinel**, so a caller who passes
   `order=None` and one who passes the current constant share a cache entry. That is intended and
   asserted; but the cache is keyed on the order, so a process that computed at 6 and then at 4
   holds two tables for one wavenumber. `RESIDUAL_CACHE_MAX_ENTRIES` bounds it and no production
   run changes the order mid-process, so nothing is opened for it.
3. **The zero-length phase branch records an order for a table it never built.** With one sample at
   the anchor, $\theta = 0$ identically and no residual table exists; the object records the order
   the call was configured at, because the column is not nullable and "the order this object would
   have used" is the only honest answer available. Recorded rather than resolved: an alternative
   would be a nullable column, which is a schema change and D3's, not this prompt's.
4. **No `GkSource` change, still.** It assembles and integrates nothing, so there is no order for
   it to report and nothing here touches it.

## 8. State handed to the next prompt

**Prompt 05a is next. What it inherits from this commit:**

1. **`main.py` is untouched — zero lines.** Log 05's site-by-site table of which `object_get`
   calls still pass a tolerance, and for which target, **stands exactly as prompt 05 left it**.
   Nothing in this commit moved a lookup, a batch dict or a keyword in that file, and the
   conditional pair in `build_missing_GkSource`'s `object_read_batch` — carried for
   `GkNumericValue` and not for `GkWKBValue` — is unchanged.
2. **The four-and-four split is unchanged.** `wavenumber_exit_time`, `GkNumericIntegration`,
   `TkNumericIntegration` and `QuadSourceIntegral` carry a tolerance; `BackgroundModel`,
   `GkWKBIntegration`, `TkWKBIntegration` and `GkSource` carry none. 05a still decouples four
   targets and still widens the `ast` guard past `endswith("Integration")`.
3. **Two constructors gained a required payload key**, which 05a will meet if it touches either:
   `BackgroundModel`'s populated payload must carry `tau_gauss_order`, `cs_tau_gauss_order` and
   `friction_F_gauss_order`, and both WKB classes' must carry `rho_gauss_order`. A fixture that
   builds one of these offline states the order its tables were built at, in the pattern
   `test_background_tau.py` and `test_run_identity.py` now show.
4. **`WKB_phase_function`'s payload has one more key**, `rho_gauss_order`, and
   `TestPayloadContract.REQUIRED_KEYS` in `test_gk_wkb_phase.py` is an exact set equality that
   knows about it.
5. **Nothing is opened.** `[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]`
   closes with both halves, and no issue is re-pointed.

**Provenance (README §1.2).** This prompt ships **no parameter** and sets none; `config/defaults.py`
is byte-identical and all four orders are still 4. The five provenance fields for $N_\tau$,
$N_{c_s\tau}$, $N_F$ and $N_\rho$ are in **log 04's** "State handed to the next prompt" and are not
restated here. What this commit adds to them, for prompt 06 to record beside the value: the order
in a row is now the order that row's tables were **built** at, on both the compute and the
rehydration path, so the provenance note can say that the value it records is the value the object
used — and not merely the value the store is keyed on.
