# Prompt 01 — delete the old-store refusals that misfire

**Campaign:** [`README.md`](README.md) · **Board items:** **M1**–**M4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none. First of the campaign. `git status` clean; `python -m RunRegistry list` shows
nothing `running` that this tree's tests would compete with.
**Closes:** `[00-old-store-refusals-misfire-on-any-database-error]`, on this board's §4, with its
index row deleted. **Narrows:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`.
**Opens:** nothing expected.
**Recommended model:** **Sonnet**. Deletions in four factories and one helper module, the tests
that exercised them, and two new tests; no design.

**Read first:**

1. [`README.md`](README.md): §0, §0.1, §1, §5 (rules 7, 9 and 11 especially), §6.1 D2.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): §0, **F0**, **B8**,
   **B9**, **B10** (what must *not* be deleted).
3. [`docs/datastore-generic-audit/str_e_misfire_probe.py`](../../docs/datastore-generic-audit/str_e_misfire_probe.py)
   and its output: why `"<column>" in str(e)` matches every error.
4. `Datastore/SQL/ObjectFactories/BackgroundModel.py`: `sqla_BackgroundModelFactory.build`, from the
   grid-identity setup to the end of the value-table read.
5. `Datastore/SQL/ObjectFactories/TkNumericIntegration.py` and `GkNumericIntegration.py`: each
   `except SQLAlchemyError` naming `break_point_kind`.
6. `Datastore/SQL/ObjectFactories/QCD_Cosmology.py`: the module docstring and `build`'s
   `except SQLAlchemyError`.
7. `extract_common.py`: the module docstring, `RunSelection`, `choose_run_label`,
   `resolve_run_selection`, `describe_background_generation`.
8. The tests named in §2 M3, and `ComputeTargets/tests/test_numeric_break_point_key.py`'s
   `_run_build_against`: how a factory's `build` is driven against an injected error without a
   datastore.

---

## 1. What is wanted

The user has decided that no code accommodates a store written by older code (README §6.1, D2).
Four factories do so by catching a `SQLAlchemyError` and, if a column's name appears in `str(e)`,
replacing it with "this datastore predates prompt N and must be regenerated". `str(e)` carries the
SQL statement, which always names the column, so the test matches any error at all (audit F0,
measured). On `BackgroundModel`'s read path the match does worse than misreport: it re-runs the
query without the grid-identity columns and reports the background as of unknown generation.

Deleting these refusals is the user's remedy for both problems. A store written before a column
existed will then meet SQLAlchemy's own `OperationalError` naming the missing column, which is loud
and true. Nothing a current store does changes, except that a database error now reaches the caller
as itself.

---

## 2. What to change

**M1 — the four factories.** Delete, in each, the `except SQLAlchemyError` handler that exists to
name an older schema, so that the error propagates unchanged. Where the `try` then guards nothing
else, remove the `try`.
- `TkNumericIntegration.py`: in `build` and in `TkNumericValue.build`.
- `GkNumericIntegration.py`: in `build`, `GkNumericValue.build` and `GkNumericValue.read_batch`.
- `QCD_Cosmology.py`: in `build`. Keep the handling of `MultipleResultsFound` described in the
  comment above it, if it is separate; delete the module docstring's paragraph on stores written
  before the representation became part of the key.
- `BackgroundModel.py`, `sqla_BackgroundModelFactory.build`:
  - the **Gauss-order** refusal (it tests `e.orig` and does not misfire, but it is an old-store
    refusal in the same `except`, and the `except` goes as a whole);
  - the **grid-identity** branch: the compute-path refusal, and the read-path re-query
    `_build_query(False)`. `_build_query` loses its `with_grid_identity` parameter and always selects
    the two grid-identity columns;
  - the `hasattr(row, "source_grid_digest")` tests that exist only because a re-queried row lacks
    them (in the "spanning N generations" refusal and where `grid_identity` is set); the "unknown
    generation (this datastore records none)" wording goes with them;
  - the **value-table** refusal (`tau_lo_Mpc`, `cs_tau_Mpc`, `cs_tau_lo_Mpc`, `friction_F`);
  - the module-docstring and inline comments that describe pre-prompt-14, pre-05 or pre-03/04
    stores, and the comment beside `obj._source_grid_identity = None` that describes an old store.
    **Keep** `_source_grid_identity = None` on a miss and the read path's `z_sample=None`
    selection: those are current behaviour (audit B10).

**M2 — `extract_common.py`.**
- `choose_run_label`: a datastore with **no** run labels is no longer read as an unnamed run of
  unknown generation. It is refused, with a message saying the store records no runs. (A current
  store with any rows has at least `Run_default`: `main.py`'s `DEFAULT_RUN_LABEL`.) Delete the
  module-docstring bullet and the `RunSelection` docstring text describing that case;
  `RunSelection.label` is then never `None`, and `resolve_run_selection`'s `if selection.label is
  None` goes.
- `describe_background_generation`: its "predates prompt 14" branch can now be reached only by a
  model that was not found. Make it say so (or raise; *prompt's choice*, recorded in the log), with
  no reference to an older store, and correct its docstring.

**M3 — the tests of what was deleted.** Delete, listing each in the log (README §5 rule 7):
- `ComputeTargets/tests/test_run_identity.py`: `TestAStoreWithNoGridIdentity` as a whole, and
  `TestChoosingARun.test_a_store_with_no_runs_reads_as_unknown`;
- `ComputeTargets/tests/test_cosmology_representation_key.py`: `TestOldSchemaFailsLoudly` as a whole;
- `ComputeTargets/tests/test_numeric_break_point_key.py`: `TestOldSchemaFailsLoudly` as a whole;
- `Datastore/tests/test_value_reads_parent_key.py`: `TestStoreWithoutBreakPointKindIsRefused` as a
  whole.

Then search every suite for any other test that asserts one of the deleted messages ("predates",
"no migration", "unknown generation", "records no run") or builds a table without one of the
columns above. **List each in the log** and either delete it, if it exists only for a deleted path,
or stop and ask (§5), if it tests anything else. `test_background_cs_tau_friction.py`'s
`test_a_value_without_the_new_limbs_is_refused` tests an in-memory guard in
`ComputeTargets/BackgroundModel.py`, not a store: **leave it**.

Add a test that `choose_run_label` refuses a store with no runs.

**M4 — keep the misfire from coming back.** Tests, needing no Ray and no datastore, that an
unrelated database error reaches the caller **unchanged** from each factory and path above, built
with the **statement text the factory really issues** — so that `str(e)` names the columns, exactly
as in the probe. Capture the statement from the factory's own query (by injecting the error from the
connection's `execute`, with SQLAlchemy's compiled statement as the error's statement), not from a
literal. For each:
- the exception that reaches the caller is the one raised, not a `RuntimeError`;
- for `BackgroundModel` on the read path, the connection is executed **once**: no second query.

Cover `TkNumericIntegration.build`, `GkNumericIntegration.build`, `GkNumericValue.read_batch`,
`QCD_Cosmology.build`, and `BackgroundModel.build` on both paths and on the value-table read.

---

## 3. Verification

1. The new tests pass. Every suite, with the counts before and after, and each difference
   accounted for by a test deleted (M3) or added (M2, M4).
2. `grep -n 'in str(e)'` over `Datastore/SQL/ObjectFactories/` finds nothing that names a column.
   Show the output.
3. `black --check` clean on every file changed.
4. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) restore `TkNumericIntegration.build`'s handler, with its `str(e)` test;
   - (ii) restore `BackgroundModel.build`'s read-path re-query alone;
   - (iii) in one M4 test, replace the captured statement with the literal `"SELECT ..."` (the test
     must then fail to catch (i): run (i) and (iii) together and show the M4 test passing, which is
     why the statement is captured — then restore both);
   - (iv) make `choose_run_label` return a selection for a store with no runs.

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. No factory catches a `SQLAlchemyError` to name an older schema in the four files; `BackgroundModel`
   always selects its grid identity and never re-queries.
2. `extract_common` refuses a store with no runs and names no older store.
3. The M3 deletions listed, the M2 and M4 tests, the deliberate-breakage record (i)–(iv).
4. `[00-old-store-refusals-misfire-on-any-database-error]` moved to this board's §4 with a closing
   note naming this commit and the M4 tests; its index row deleted; the count and date corrected.
   `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` narrowed: a dated line naming
   what 01 deleted.
5. `black --check` clean. On this board: the §1 row, items M1–M4, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- A test outside M3's list asserts a deleted behaviour for a reason other than an old store.
- Removing a handler changes what a current store returns (a hit, a miss, or a value) on any path.
- An extraction script relies on `RunSelection.label` being `None`, or on the "unknown generation"
  string, for anything but printing it.

---

## 6. What this prompt does not do

- It does not touch `TkWKBIntegration`, `GkWKBIntegration`, `GkSource` or `QuadSourceIntegral`
  (02), `ShardedPool` (03–05) or the reader and inventory (05).
- It does not touch `ComputeTargets/BackgroundModel.py`'s in-memory guards (audit B10 (c)).
- It changes no table, column or key (README §5 rule 9).

---

## 7. The log and the board

`logs/01-delete-the-old-store-refusals-that-misfire.md`, using README §5.1. In addition: every test
deleted, by module, class and method; the suite counts before and after.

`IMPLEMENTATION_STATE.md`: the §1 row for 01, items M1–M4, §3/§4, §5.
