# Log 05 — Replace the vestigial key columns with the Gauss orders

**Prompt:** `prompts/tolerance-convergence/05-replace-the-vestigial-key-columns.md`
**Commit:** *(this prompt's own commit)* — "Key the order-governed targets on their Gauss orders"
**Model:** Opus 5
**Date:** 2026-09-18
**Result:** **DONE.** The four object types whose `(atol, rtol)` pair reached no solver have lost
it. `BackgroundModel` gains `tau_gauss_order`, `cs_tau_gauss_order` and `friction_F_gauss_order`;
`GkWKBIntegration` and `TkWKBIntegration` each gain `rho_gauss_order`; `GkSource` gains nothing,
because it assembles and integrates nothing. Every one of the four orders is **filtered on, not
merely selected**, and repointing its single declaration moves the compiled `whereclause`, the
value `store()` writes and the compute class's own accessor **together** — the property §3 asks
for, demonstrated in `ComputeTargets/tests/test_gauss_order_key.py` by patching the constant, not
by checking the column exists. **No number changes**: `config/defaults.py` is byte-identical
(`git diff -- config/defaults.py` empty) and `TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`,
`FRICTION_F_GAUSS_ORDER` and `RHO_GAUSS_ORDER` are all still **4**.
`ComputeTargets` **508, OK**; `CosmologyModels` **39, OK**;
`ComputeTargets/tests/test_numeric_break_point_key.py` passes **unedited**; the three published
source-grid digests are unmoved. **15 production files** in the diff.

`[20-wkb-gauss-orders-not-in-lookup-key]` is **closed**, with its `RESIDUAL_WKB_REGION_MARGIN` half
answered on evidence rather than dropped. Board item **T9**.

---

## What shipped

**15 production files and 7 test files.** `config/defaults.py`,
`ComputeTargets/GkNumericIntegration.py`, `TkNumericIntegration.py`,
`CosmologyConcepts/wavenumber.py`, their factories, `QuadSourceIntegral`, `break_point_kind` and
`ComputeTargets/tests/test_numeric_break_point_key.py` are all untouched — the four targets whose
tolerances reach a solver keep them.

### The mechanism, and why a row cannot claim an order it was not computed at

**The order is reached through one name, resolved at call time, on every path.** §3 leaves the
choice open between passing the order in from `main.py` the way `atol` was passed, and having the
compute class and factory read the module constant. This commit takes the second, because the first
can express the divergence and the second cannot:

| path | how it reaches the order |
|---|---|
| the computation | `compute_background` reads the module globals `TAU_GAUSS_ORDER` / `CS_TAU_GAUSS_ORDER` / `FRICTION_F_GAUSS_ORDER` and hands each to its own `CumulativeTable`; `WKB_phase_function` reads `phase_residual.RHO_GAUSS_ORDER` and records it as `metadata["N_rho"]` |
| the compute object | `BackgroundModel.tau_gauss_order` &c. and `{Gk,Tk}WKBIntegration.rho_gauss_order` are **properties** that return those same module globals, evaluated on each call |
| `store()` | writes `obj.tau_gauss_order` &c. — the accessor, never a literal and never a payload value |
| `build()` | filters on `background_model.TAU_GAUSS_ORDER` &c. and `phase_residual.RHO_GAUSS_ORDER` — a module attribute read when the query is formed, never a literal |

So there is **no keyword, no payload key and no default** on the path by which a caller could
supply an order different from the one the tables were built at. `main.py` does not name an order
anywhere; neither does any `extract_*.py`; neither does any payload dict. Editing the declaration
moves all four uses in the same commit, and `test_gauss_order_key.py`'s
`test_repointing_the_declaration_moves_the_query_and_the_stored_value_together` and
`test_no_order_is_written_as_a_literal` are the demonstration — the second exists because a site
that had inlined `4` would pass the first three assertions of the module and fail that one.

**The one remaining case the mechanism does not close by construction**, and it is recorded rather
than fixed: `phase_residual.build_phase_residual` and `cached_phase_residual` take
`order: int = RHO_GAUSS_ORDER`, and a *default argument* is bound once at `def` time. It is safe
while no caller supplies the keyword, and none does —
`test_no_production_caller_overrides_the_residual_order` walks every production module and asserts
it. But the path exists, `ComputeTargets/phase_residual.py` is not in this prompt's file list, and
so it is `[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]` (§3 of the
board) rather than an edit here.

**The deserialised-object question, and why it does not arise.** `store()` writes the *current*
order, so in principle an object read back from a row written at order 4 and then re-stored (the
replication path, `BackgroundModel` being a replicated table) could be written under a different
order. It cannot happen, because `build()` filters on the same constant: an object in hand was
returned by a query that demanded the current order, so the row it came from carries it.

### `Datastore/SQL/ObjectFactories/BackgroundModel.py`

Three integer columns in place of the two foreign keys, `index=True, nullable=False`; the two
`tolerance` aliases and their joins gone from `build()` and from `validate_on_startup`, whose
message now prints `N_tau` / `N_cs_tau` / `N_F` instead of `log10_atol` / `log10_rtol`. The
`SCHEMA NOTE` docstring gains a fourth entry stating what the pair described (nothing), what the
orders describe, why there are three of them and not one, and that there is no migration.

`build()`'s existing missing-column handler gains a branch for the three new names, raising a
`RuntimeError` that names this prompt — the pattern prompt 20 of `GkTk-remedial` set for
`break_point_kind` and the same pattern already present here for `source_grid_digest`. It is
matched against the driver's own message (`e.orig`) and not against `str(e)`, which carries the SQL
statement and therefore names every column the query asks for, missing or not (deviation 3).

### `ComputeTargets/BackgroundModel.py`

`compute_background` loses its `atol` / `rtol` parameters — they were documented in the file as
"accepted for signature compatibility … the table has no tolerances" — and the docstring now says
what the key carries instead. `BackgroundModel.__init__` loses the two arguments and
`self._atol` / `self._rtol`; `compute()` no longer forwards them. The three order properties are
added with the comment that states the invariant at the point of use.

### `Datastore/SQL/ObjectFactories/{Gk,Tk}WKBIntegration.py` and `ComputeTargets/{Gk,Tk}WKBIntegration.py`

One `rho_gauss_order` column each, filtered on in `build()`; the pair gone from `register()`,
`build()`, `store()`, `validate_on_startup`, `inventory()` and from both value-factory lookups
(`_build_impl_model` and `read_batch`), where the two *optional* tolerance filters become one
unconditional order filter — the order is not the caller's to omit. Each factory gains a module
docstring with the same `SCHEMA NOTE`, including the `RESIDUAL_WKB_REGION_MARGIN` finding below,
and a missing-column handler naming the prompt. The two compute classes lose the constructor
arguments and gain `rho_gauss_order`; the comments that used to say "kept because they are part of
the datastore lookup key" now say what is in the key.

### `Datastore/SQL/ObjectFactories/GkSource.py` and `ComputeTargets/GkSource.py`

The pair is dropped and **nothing replaces it**, said in the code where the columns used to be and
in a new module docstring: `compute()` calls `assemble_GkSource_values`, which stitches the numeric
and WKB results together and integrates nothing, so there is no order to record. No missing-column
handler, and deliberately: this factory only *loses* columns, so a store written before this commit
carries two columns nothing queries and remains readable.

### `main.py` — **nine** sites, not four

The pair is removed wherever it fed one of the four targets:

| target | sites |
|---|---|
| `BackgroundModel` | the `object_get` at STEP 1 |
| `TkWKBIntegration` | `build_Tk_WKB_work`'s `query_batch`, its direct `object_get`, and `build_QuadSourceIntegral_batch`'s `Tk_WKB_lookup_batch` |
| `GkWKBIntegration` | `build_Gk_WKB_work`'s `query_batch` and its two direct `object_get` calls |
| `GkSource` | `build_GkSource_work`'s batch, `build_missing_GkSource`'s `object_read_batch` payload, the direct `object_get`, and the four `GkSource` lookup batches inside the `GkSourcePolicy` steps |

Prompt §1's "the four `object_get` sites" is a count of *targets*, not of sites; the tree has nine
distinct payloads across them (deviation 1). Nothing else in `main.py` moves: the `ray.get` block,
the tolerance objects and every `TkNumericIntegration` / `GkNumericIntegration` /
`wavenumber_exit_time` / `QuadSource` / `QuadSourceIntegral` site are 05a's and are untouched.

**One payload serves two classes and had to be split.** `build_missing_GkSource`'s
`object_read_batch` dispatches the *same* dict into `GkNumericValue` and `GkWKBValue`. The first
reaches `GkNumericIntegration`, which keeps its pair and whose value factory filters on it
*optionally* — so dropping the pair there would have silently widened a lookup on a target this
prompt must not touch. The pair is now added for `GkNumericValue` alone.

### The readers — six scripts, and D3's count of three was low

Every one of the six performs a `BackgroundModel` lookup with the pair, and two also look up
`GkSource`:

| script | `BackgroundModel` | `GkSource` | a WKB target | left alone |
|---|---|---|---|---|
| `extract_Gk_data.py` | dropped | — | — | `GkNumericIntegration` batch dict — D3's trap, unchanged |
| `extract_GkWKB_data.py` | dropped | — | `GkWKBIntegration`: payload **split** | `GkNumericIntegration` keeps the pair |
| `extract_TkWKB_data.py` | dropped | — | `TkWKBIntegration`: payload **split** | `TkNumericIntegration` keeps the pair |
| `extract_GkSource_data.py` | dropped | dropped | — | `wavenumber_exit_time` keeps the pair |
| `extract_tensor_source_data.py` | dropped | — | — | `wavenumber_exit_time` keeps the pair |
| `extract_QuadSourceIntegral_data.py` | dropped | dropped | — | `QuadSourceIntegral` keeps `quad_atol`/`quad_rtol` |

Two of them shared one `query_payload` between a numeric target (keeps the pair) and a WKB target
(loses it), exactly as `main.py` did; both are split, with the numeric lookup passing `atol=` and
`rtol=` explicitly and the WKB one taking the common dict. **The README is not corrected here** —
it is the orchestrator's file, as §5 says — and nothing is opened for it.

`docs/tolerance-convergence/inventory.py` and `order_audit.py` read the column names as data and
are **left alone** (§5). Their published output was correct for the tree it was taken on (§5 rule
7); their next run will report differently — `inventory.py` will find nine keyed object types
rather than twelve carrying the pair, and `order_audit.py`'s `has_atol` column will read false for
the three targets it covers. Prompt 06 restates the inventory.

### The tests

**New: `ComputeTargets/tests/test_gauss_order_key.py`**, 16 tests, on
`test_numeric_break_point_key.py`'s pattern and needing neither Ray nor a datastore: the tables are
built from each factory's own `register()`, and the connection raises instead of executing so that
the `build()` query can be compiled and its `WHERE` clause read back. It asserts the schema of §2;
that each order is an equality criterion and reaches the compiled SQL; that the key that was
already there is undisturbed; that no tolerance survives in any of the four criteria or in any
`store()` payload; that moving the declaration moves query, stored value and accessor together;
that a row at any other order fails the equality; that no order is written as a literal; that the
computation reads the same declaration; and that all four orders are still 4.

**Repaired**, all of it forced by the constructor and payload change:

- `test_run_identity.py` — the module docstring's description of the key, the `BackgroundModel`
  construction, both build payloads, and the raw `sqla.insert` at what is now line 727, which
  writes a pre-prompt-14 row directly and therefore has to name every column the current schema
  has. The `tolerance` table and its two fixture objects are gone from `_Schema`: nothing in that
  module reaches them any more.
- `test_background_tau.py`, `test_background_cs_tau_friction.py`, `test_tk_wkb_phase.py` — three
  `atol=None, rtol=None` constructor arguments that no longer exist.
- `wkb_reference.py` — `QCDModel.__init__`'s `atol` / `rtol` parameters, forwarded to
  `compute_background._function`. No caller in the tree passes either.
- `test_main_plumbing.py` — the `ast` guard. See deviation 2.

---

## Deviations from the prompt

### 1. `main.py` has nine sites, not four — `STRUCTURALLY REQUIRED`

Prompt §1's file list says "`main.py` — the four `object_get` sites of those targets". Read against
the tree that is a count of targets: `GkSource` alone is reached from seven payloads, and
`TkWKBIntegration` from three. All nine are the same edit and none of them is anything but a
lookup of one of the four targets; leaving any behind would leave a query filtering on a column
that no longer exists. Nothing else in `main.py` is touched, which is the boundary §8 actually
draws.

### 2. `ComputeTargets/tests/test_main_plumbing.py`'s `ast` guard had to be taught the new truth — `STRUCTURALLY REQUIRED`

The guard classifies every `object_get` whose class name ends in `"Integration"` and fails when a
site's tolerance cannot be read. After this commit six such sites carry no tolerance at all, so the
guard failed — correctly, since it cannot distinguish "no tolerance, by design" from "a spelling I
do not understand". §2 (g) and §3.5a give the guard's widening to prompt **05a**; what is added
here is the minimum that keeps it honest across this commit: a `NO_TOLERANCE_INTEGRATIONS` tuple,
those sites classified as *carrying none* rather than as unreadable, and a new test asserting that
all six carry none and that a tolerance reappearing on either WKB target is a failure. That last is
a regression guard this campaign wants anyway (§3.5a names it in terms). The comment in the file
says 05a widens the whole thing.

**It earned its keep immediately.** The guard is what found
`build_QuadSourceIntegral_batch`'s `Tk_WKB_lookup_batch` — a `TkWKBIntegration` lookup 2,000 lines
from the other two, which a `grep` of the region around each target's own step would have missed.

### 3. The missing-column handlers match `e.orig`, not `str(e)` — `IMPLEMENTATION CHOICE`

The existing prompt-14 handler in the `BackgroundModel` factory matches column names in `str(e)`.
That works for a column the query only *selects*; it does not work for one the query *filters on*,
because `str(e)` of a SQLAlchemy error includes the statement, and the statement names
`tau_gauss_order` whether the column exists or not. Written that way, a store missing
`source_grid_digest` would have been reported as a store missing `tau_gauss_order`. The new
handlers therefore read `getattr(e, "orig", e)`, the driver's own message. The prompt-14 handler is
left as it is — it is correct for what it matches, and changing it is not this prompt's business.

### 4. Three factories gained a missing-column handler the prompt did not ask for — `IMPLEMENTATION CHOICE`

A factory that *gains* a non-nullable column will hit an opaque `OperationalError` against any
store written before this commit. Both numeric factories carry such a handler for
`break_point_kind` and this factory already carried one for the prompt-14 columns, so the pattern
is the tree's own and the message is the one D3's standing instruction dictates: regenerate, there
is no migration, and there is nothing to infer an order from. `GkSource` gains none, because it
only loses columns and an old store stays readable.

### 5. Four test modules outside the prompt's named list were edited — `STRUCTURALLY REQUIRED`

`test_background_tau.py`, `test_background_cs_tau_friction.py`, `test_tk_wkb_phase.py` and
`wkb_reference.py` pass `atol=` / `rtol=` to constructors this commit changes. §5 rule 8's fixture
carve-out is not what is used here — none of these is a fixture regeneration and no threshold,
tolerance or measured constant moves in any of them. The prompt's own file list says "and any other
test module that names the columns", and these name the arguments that carried them. Each edit is
the deletion of two arguments. **`test_background_cs_tau_friction.py` in particular is touched for
one two-line deletion in each of two stand-in constructions**, and nothing 04b put there is
disturbed.

### 6. `validate_on_startup` and `inventory()` now report the order — `IMPLEMENTATION CHOICE`

Both printed `log10_atol` / `log10_rtol` for every unvalidated row. Deleting the joins that fed them
left the message with nothing in that slot, so it prints the order instead. It is a diagnostic
string, not a key, but reporting nothing where the schema's own accuracy parameter now lives would
be a worse choice than reporting it.

---

## Verification performed

| Check | Result |
|---|---|
| `ComputeTargets` suite | **508 tests, OK** (prompt §7 item 8: must not fall below 452, expected ≥ 491; 491 at `HEAD~1`, plus 16 in the new module and 1 in the repaired guard) |
| `CosmologyModels` suite | **39 tests, OK** |
| `ComputeTargets/tests/test_numeric_break_point_key.py` | **passes unedited**; `git diff` on it is empty |
| `test_gauss_order_key.py` | 16 tests, OK |
| `test_run_identity.py` | 30 tests, OK |
| `test_main_plumbing.py` | 22 tests, OK (21 before, plus the new tolerance-free assertion) |
| `config/defaults.py` | `git diff -- config/defaults.py` **empty** |
| the four orders | `TAU_GAUSS_ORDER` 4, `CS_TAU_GAUSS_ORDER` 4, `FRICTION_F_GAUSS_ORDER` 4, `RHO_GAUSS_ORDER` 4 |
| the six `extract_*.py` | all six `python -m py_compile` clean; an `ast` read of every `object_get*` payload confirms no reference to the pair for any of the four targets, and that `extract_Gk_data.py`'s `GkNumericIntegration` dict is unchanged |
| `main.py` | `py_compile` clean; an `ast` walk resolving every `RayWorkPool` batch variable as well as every direct keyword reports **zero** sites passing `atol`/`rtol` into any of the four targets or their value classes |
| source-grid digests | `test_source_grid.py` and `test_convergence_reference.py` assert `4849552b`, `60a3205a`, `3bef2c06` and `21ffc126` and pass inside the suite |
| `black --check` | clean on every `.py` in the diff |
| untouched, by `git diff --stat` | `config/defaults.py`, `ComputeTargets/GkNumericIntegration.py`, `TkNumericIntegration.py`, `CosmologyConcepts/wavenumber.py`, both numeric factories, `QuadSourceIntegral` and its factory, `test_numeric_break_point_key.py` — all empty |

### `RESIDUAL_WKB_REGION_MARGIN` — verified against `ORDER-AUDIT.md`, not taken from §2

Prompt §2 asks that the reading be checked rather than accepted, and it holds, with one
qualification §2's sentence does not carry.

`ORDER-AUDIT.md` §7.2 rebuilds the `delta` a producer actually reads, between two *fixed*
redshifts, at ten margins from 0.05 to 0.9999, on three models in both sectors at five
wavenumbers. Over **0.05 to 0.9** every one of the 30 rows is marked `=`, bit-identical to
production's — the band's node count moves by hundreds and the answer does not move at all,
because the Gauss panels between the two fixed redshifts are the grid's own. That is the claim §2
makes and it is confirmed.

The qualification is at the extremes, above the range in which a production object is producible
at all. At `margin >= 0.99` every `Tk` row reads `anchor clamped`: the band no longer reaches the
production anchor, so the producer does not get a *different* row, it gets no row — a reachability
failure, which is the opposite of an accuracy axis. And at `margin = 0.9999` two `QCD` `Gk` cells
depart by **2e-16**, one ulp, from the rounding of a cumulative whose top node has moved. So the
margin *can* change the last bit of a stored value, at a setting four decades outside anything the
campaign contemplates and after the band has already begun failing to cover the anchors.

**I do not conclude it should be keyed**, and there is no stop here. A key column exists to
separate rows that differ; over the entire range in which both sectors remain producible this one
separates nothing, and at the setting where it moves a bit the object is on the edge of not
existing. It is a bound on *where* the residual may be evaluated, which is what `residual_node_range`'s
own comment says it is. The finding is recorded in both WKB factories' schema notes.

---

## Observations not acted on

1. **`extract_TkWKB_data.py` queries `TkNumericIntegration` under the shared `atol`**, while
   `main.py` writes it under `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, so that lookup cannot match a
   production row. That is `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]`, whose
   hook says "prompt 05 revisits all six readers anyway". This commit did revisit them, and left
   it: the fix is to import the split constant into the readers, which is a decision about
   `TkNumericIntegration`'s tolerance — a target §8 forbids and whose constants are **05a's**. The
   issue stays open and is re-pointed at 05a on the board.

2. **`[02a-grid-digest-not-reproducible]` part (ii) is 05a's, not this prompt's.** The index hook
   and README §4 both assign "applying one design tolerance to the row match and the grid digest
   together" to prompt 05, written before §7 D9 split it. It is a tolerance decision about
   `wavenumber_exit_time`'s `rtol` and `DEFAULT_REDSHIFT_RELATIVE_PRECISION`, and §5 rule 8 as D9
   amended it puts every parameter move in 05a. The board and the index are corrected to say so;
   nothing is measured or changed here.

3. **`BackgroundModel.TAU_GAUSS_ORDER` and `WKB_phase_function.PHASE_SOLVER_STEPPING` are
   class/module *snapshots* of the two declarations**, taken at import time and used only to build
   an `IntegrationSolver` label (`main.py:3575` and `:3584`). They cannot drift from the
   declaration by any source edit — each is assigned from it — but they do not follow a runtime
   monkeypatch, which is why `test_gauss_order_key.py` patches the declaration and not the
   snapshot. Not a defect, and repairing it would mean editing `main.py` outside the four targets'
   lookups.

4. **`GkSource` has no `_do_not_populate`-style archival concern** and needed no handler, but its
   `build()` still *selects* nothing in place of the two `log10_tol` labels it used to. Nothing
   read them; `GkSource` exposes no `log10_atol` property and never did.

5. **The four targets' rows in a pre-existing datastore are unreachable from this commit**, which
   is the intended outcome (§7 D2, and D3's standing instruction). No prompt in this campaign may
   cost that, and this one does not.

---

## State handed to the next prompt

**Prompt 05a is next. What it inherits from this commit:**

1. **Four targets still carry a tolerance, and four carry none.** The four that do:
   `wavenumber_exit_time`, `GkNumericIntegration`, `TkNumericIntegration` and
   `QuadSourceIntegral`. The four that do not: `BackgroundModel`, `GkWKBIntegration`,
   `TkWKBIntegration`, `GkSource`. 05a decouples four targets, not eight, and its `ast` guard is
   built against the enumeration that will stand — which is the whole point of D9's ordering.

2. **The `object_get` sites in `main.py` that still pass `atol=` / `rtol=`**, by target, so that
   05a does not have to re-derive them:

   | target | sites | tolerance passed today |
   |---|---|---|
   | `wavenumber_exit_time` | `build_k_exit_work` (`main.py:964`) | `atol`, `rtol` |
   | `TkNumericIntegration` | `build_Tk_numeric_work`'s batch, its direct `object_get`, `build_Tk_WKB_work`'s numeric lookup batch, `build_QuadSource_work`'s numeric lookup batch, `build_QuadSourceIntegral_batch`'s `Tk_numeric_lookup_batch` — **5 sites** | `Tk_numeric_atol`, `rtol` |
   | `GkNumericIntegration` | `build_Gk_numeric_work`'s batch, its direct `object_get`, `build_Gk_WKB_work`'s numeric lookup batch, and `build_missing_GkSource`'s `GkNumericValue` read batch — **4 sites** | `atol`, `rtol` |
   | `QuadSource` | `build_QuadSource_work`'s query batch — the factory ignores the pair | `atol`, `rtol` |
   | `QuadSourceIntegral` | the missing-instance batch and the direct `object_get` — **2 sites** | `quad_atol`, `quad_rtol` |

   `build_missing_GkSource`'s `object_read_batch` payload now carries the pair **conditionally**,
   for `GkNumericValue` and not for `GkWKBValue`; 05a should keep that conditional and change only
   which tolerance object it names.

3. **`DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE` have fewer consumers than they did.** In
   `main.py` they now reach `wavenumber_exit_time`, `GkNumericIntegration`, `TkNumericIntegration`
   (`rtol` only) and `QuadSource`; `BackgroundModel`, both WKB sectors and `GkSource` no longer see
   them. Outside `main.py` they are unchanged, and
   `[02-shared-atol-doubles-as-a-float-comparison-epsilon]` — `DEFAULT_ABS_TOLERANCE` as a bare
   `fabs(a − b) <` epsilon at seven sites — is untouched and still the hazard that issue says it
   is. Nothing in this commit narrows it.

4. **The `ast` guard is repaired, not widened.** `NO_TOLERANCE_INTEGRATIONS` covers the two WKB
   targets and `EXPECTED_NO_TOLERANCE_SITES = 6`. 05a replaces the `endswith("Integration")`
   predicate with an explicit enumeration of all eight targets — `BackgroundModel` and `GkSource`
   are still invisible to it — and should fold `NO_TOLERANCE_INTEGRATIONS` into the four-and-four
   split §3.5a describes.

5. **Two issues are re-pointed at 05a** rather than left assigned to this prompt:
   `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` and part (ii) of
   `[02a-grid-digest-not-reproducible]` (observations 1 and 2). One is opened:
   `[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]`.

**Provenance (README §1.2).** This prompt ships **no parameter** and sets none;
`config/defaults.py` is byte-identical. The five provenance fields for $N_\tau$, $N_{c_s\tau}$,
$N_F$ and $N_\rho$ are in **log 04's** "State handed to the next prompt" and are **not restated
here**, as prompt §9 requires. What this commit adds to them is one thing: **all four are now in a
lookup key**, on `BackgroundModel` (three columns) and on each WKB target (one), filtered on rather
than merely recorded — so a run at a different order computes a new row instead of being served an
old one, and the provenance note prompt 06 assembles can say that the value it records is the value
the store is keyed on.
