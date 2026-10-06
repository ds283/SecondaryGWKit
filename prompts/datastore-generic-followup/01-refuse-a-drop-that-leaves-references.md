# Prompt 01 — refuse a drop that leaves references

**Campaign:** [`README.md`](README.md) · **Board items:** **D1**–**D3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** the campaign is written (this commit's parent). `git status` clean. Never in one checkout
while another prompt's change to `ShardedPool.py` is uncommitted.
**Closes:** `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, on
`datastore-generic`'s board. **Narrows:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Opus**. The change adds a refusal to `ShardedPool`'s constructor and a
derivation to `schema.py`, small but behavioural. Its test plan, every table it must name and the
one test that moves were measured with a rough version of the change (README §0.2).

**Read first:**

1. [`README.md`](README.md): §0.2 (item 1, the table of dependents), §1, §4, §5 (rules 3, 4, 7,
   9 and 11) and §6 (the user's decision 1, and the choices of method in §6.2).
2. `prompts/datastore-generic/IMPLEMENTATION_STATE.md`, §3.1: the entry
   `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`, with its measurement and
   **Decided** line.
3. `Datastore/SQL/ShardedPool.py`:
   - the constructor's `drop_tables` parameter and its docstring (`:107`, `:122-125`, `:144-145`);
   - the read-only branch and the undeclared-table refusal's call (`:198-208`);
   - `_refuse_undeclared_drop_tables` (`:698-711`).
4. `Datastore/SQL/Datastore.py`, `_drop_tables` (`:410-436`): what each actor does with the
   tables. It does not change.
5. `Datastore/SQL/schema.py`: `build_schema`, and `drop_order` (`:279-319`), beside which the new
   function goes.
6. `Datastore/store_inventory.py`: its module docstring on imports (`:82-85`), `InventorySpec`
   and its `dependencies()`, and `inventory_specs` (`:912-922`).
7. `config/datastore.py`: `drop_groups` and `tables_to_drop`.
8. The tests:
   - `Datastore/tests/test_reconcile_at_open.py`, `TestPruningAfterRepair.test_an_interrupted_store`
     (`:1084-1135`), the one test that moves;
   - `Datastore/tests/test_one_timestamp_per_write.py:94-97`, which inherits it;
   - `Datastore/tests/test_layer_registry.py`, the undeclared-table refusal tests (`DROP`, `:315`,
     and `:340-410`), as the pattern for opening a stand-in pool with `drop_tables`;
   - `Datastore/tests/test_wkb_numeric_parent_key.py:890-960`, which calls the actor's
     `_drop_tables` directly and must not move.
9. The orchestrator's measurement: [`measure/rough_change.diff`](measure/rough_change.diff) (the
   `ShardedPool.py` and `test_reconcile_at_open.py` hunks), and
   [`measure/scripts/f2_closure.py`](measure/scripts/f2_closure.py) with its output. The rough
   version put the derivation in `ShardedPool`; this prompt puts it in `schema.py` (§2).

Line numbers are at `6de692a`.

---

## 1. What is wanted

`--drop` drops exactly the tables of the groups named. Production runs with foreign-key
enforcement off, so SQLite accepts a drop that leaves other tables' rows naming rows that are
gone. A re-created table leases serials from its largest stored serial, so a recomputed row can
take a serial a stale row still names, which then names an unrelated row.

The user decided on 2026-10-06 that such a drop is **refused** (README §6.1), and that a
reference is **a foreign key or a parent a factory's `inventory_spec` declares, followed
transitively**. Measured at `6de692a` (README §0.2), 8 of the 10 groups are refused alone. Two
of them, `tk-wkb` and `quad-source`, leave no foreign key behind: only `QuadSourceIntegral`'s
declared cross-shard parents name their rows.

- **D1 — the derivation.** `Datastore.SQL.schema.dependent_tables(dropped, factories)` gives the
  tables that must be dropped with `dropped`, from the registry alone.
- **D2 — the refusal.** A read-write `ShardedPool` refuses `drop_tables` whose dependents are not
  all among them, naming the dropped tables and every dependent, before the primary is created or
  opened.
- **D3 — the tests**, and the one test re-fixtured.

**What does not change.**
- What the code writes (README §5 rule 9). A drop that the refusal accepts drops exactly what it
  dropped before, in the same order.
- `main.py`, `config/datastore.py` and the actor's `_drop_tables`.
- The read-only pool, which already refuses any drop table (`ReadOnlyWrite`).
- The undeclared-table refusal, which still comes first.

---

## 2. What to change

**D1 — `dependent_tables(dropped, factories) -> List[str]`, in `Datastore/SQL/schema.py`,
beside `drop_order`.**
- The tables of the registry (`build_schema(sqla.MetaData(), factories).tables`) not in `dropped`
  that **name** a table in `dropped`, or in the result, transitively.
- Table `A` names table `B` when either:
  - `A` declares a foreign key into `B` (`B` is not `A`);
  - `A`'s factory declares an `inventory_spec` whose `dependencies()` hold `B` (`B` is not `A`).
    Use `Datastore.store_inventory.inventory_specs(factories)`. That module imports only the
    standard library, `sqlalchemy` and `Datastore.contract` at module scope, so `schema.py` may
    import it. Check that nothing cycles.
- Return the result in the registry's table order, each table once; empty when nothing must go.
- A name in `dropped` that the registry does not declare as a table raises `ValueError`, naming
  it.
- The docstring states the rule, that it is read from the registry and not from any store's rows
  (README §6.2), and why both kinds of reference count.

**D2 — `ShardedPool._refuse_drop_that_leaves_references()`.**
- Called in the constructor immediately after `self._refuse_undeclared_drop_tables()` (`:208`),
  and so for a read-write pool only, before the primary is created or opened and before any actor
  exists.
- When `self._drop_tables` is empty or `None`, it does nothing.
- Otherwise it calls `dependent_tables(self._drop_tables, self._factories)`. If any dependent is
  not in `self._drop_tables`, it raises `RuntimeError` whose message:
  - begins `Cannot open sharded datastore "<primary>": `, with that prefix exactly once;
  - names every table of `drop_tables`, sorted;
  - names every dependent not among them, in `dependent_tables`' order;
  - says that their rows would name rows that are gone, and that they must be dropped too;
  - ends `Nothing was opened`.
- The message names only tables it was given or derived. It names no `--drop` group (README §6.2)
  and no table by literal (README §5 rule 11).
- Update the constructor's `:param drop_tables:` to say what is refused, and where the rule
  lives.

**D3 — tests.**
- **New module `Datastore/tests/test_drop_refuses_dangling_references.py`, 11 tests.** No Ray,
  nothing under `var/`.
  - `TestDependentTables`, on the real registry (`config.datastore.factories`), through the
    groups of `config.datastore.drop_groups`:
    1. `test_each_group_alone_needs_its_measured_dependents`: for each of the ten groups,
       `dependent_tables` equals README §0.2's table **as a literal list, in registry order**;
    2. `test_each_group_with_its_dependents_needs_nothing_more`: each group's tables plus its
       dependents give `[]`;
    3. `test_every_group_together_needs_nothing`: `tables_to_drop(list(drop_groups))` gives `[]`;
    4. `test_a_declared_parent_without_a_foreign_key_counts`. Assert that `QuadSourceIntegral`
       declares no foreign key into `TkWKBIntegration`. Then assert that `tk-wkb`'s dependents
       hold `QuadSourceIntegral`;
    5. `test_a_foreign_key_without_a_declared_parent_counts`. Assert that `GkSource_parents`'
       factory declares no `inventory_spec`. Then assert that `gk-numeric`'s dependents hold
       `GkSource_parents`;
    6. `test_dependents_are_followed_transitively`. Assert that `QuadSourceIntegral_tags` names
       no `tk-numeric` table directly. Then assert that `tk-numeric`'s dependents hold it;
    7. `test_an_undeclared_name_is_refused`: `ValueError`, naming the name.
  - `TestThePoolRefuses`, through the stand-in pool, as `test_layer_registry` opens one:
    8. `test_a_drop_that_leaves_references_is_refused_before_anything_is_opened`. Build a
       `build_full_store` and record its checksums (`standin_pool.store_checksums`). Open it with
       `tables_to_drop(["gk-source-policy-records"])` and get `RuntimeError`. Assert §2's
       message, with every dropped and dependent table named. The checksums are unchanged;
    9. `test_nothing_is_created_for_an_absent_store`: the same drop on a primary that does not
       exist is refused, and no file is created in its directory;
    10. `test_an_undeclared_name_is_refused_first`: `drop_tables=["GkSourcePolicy", "NotATable"]`
        meets the undeclared-table refusal, not this one;
    11. `test_a_drop_with_its_dependents_is_accepted`. Open a `build_full_store` with
        `gk-source-policy-records` plus every group §0.2 lists for it (`gk-source-policy` and
        `quad-source-integral`). The pool opens.
        - After it closes, every dropped table is present (each actor re-creates its tables after
          dropping them, `Datastore.py:140-141`) and holds no rows on any shard.
        - Assert first that at least one of those tables held rows before, so that the test cannot
          pass on an empty store.
- **Re-fixtured, not deleted** (README §5 rule 7):
  `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`.
  - Its drop becomes `tables_to_drop(["gk-source", "gk-source-policy-records", "gk-source-policy",
    "quad-source-integral"])`.
  - Its docstring says that the drop names `gk-source-policy-records`' dependents because a drop
    without them is refused (this prompt).
  - Nothing else in it changes. `test_one_timestamp_per_write.TestRepairThenPruneAtOpen` inherits
    it, unedited.
- No other test changes.

**Nothing else changes.** Not `main.py`, `config/`, `Datastore.py`, `store_inventory.py`, any
factory, any `docs/` script or any store under `var/`.

---

## 3. Verification

1. **Every suite**, with counts before and after, each difference accounted for by test id:
   - Datastore 721 + 11 = **732** (`[a-f]` 147 + 11 = 158, `[g-o]` 183, `[p-r]` 200, `[s-z]` 191);
   - RunRegistry 280; ComputeTargets 693; AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen
     148 (1 skipped).
2. **`test_layer_is_generic` passes**, and `schema.py`'s new import is allowed by it.
3. **The measurement reproduced.** `dependent_tables` for each group equals
   `measure/out/f2_closure.txt`'s full lists. Show the ten lists.
4. **No accepted drop changes.** For `tables_to_drop(list(drop_groups))` and for each group with
   its dependents, `drop_order` gives the same order as at `6de692a`. `drop_order` does not
   change, so this is a check that nothing reorders its input.
5. `black --check` is clean on every touched Python file.
6. **Deliberate breakage.** Each mutation must make the named tests fail; then restore it.
   - (i) `dependent_tables` ignores declared parents: tests 1 and 4;
   - (ii) `dependent_tables` ignores foreign keys: tests 1 and 5;
   - (iii) `dependent_tables` follows one level only: tests 1 and 6;
   - (iv) the constructor's call of `_refuse_drop_that_leaves_references` removed: tests 8 and 9;
   - (v) the call moved before `_refuse_undeclared_drop_tables()`: test 10;
   - (vi) the re-fixtured test's drop restored to `["gk-source", "gk-source-policy-records"]`:
     that test, and its inherited copy in `test_one_timestamp_per_write`.

   Record each in the log as a diff, exactly as applied, with the failing tests. Check each
   with `git apply --check` and `-R --check`, in `bash`, and leave `git status` clean.

---

## 4. Acceptance

1. **D1–D3** as §2 says, with the 11 tests and the one re-fixture.
2. Every suite has the counts of §3.1.
3. **The records**, all in the same commit, with `black --check` clean:
   - On `datastore-generic`'s board, `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`
     moves from §3.1 to §4 with a dated "Closed" line. It names `dependent_tables`, the refusal,
     this campaign and README §0.2's finding that foreign keys alone undercount.
   - `docs/OPEN_ISSUES.md`: its row leaves §1.19, the count goes from 107 to 106, and the date is
     corrected.
   - This board: the §1 row, D1–D3, §5.
   - README §4's "01 ships" says it shipped, as named, or names what differs.

---

## 5. Stop conditions — stop and ask the user

- `dependent_tables` for any group differs from README §0.2's table.
- A dependent table lies in no drop group. Satisfying the refusal would then need a
  `config/datastore.py` change.
- A test outside §2's plan fails. That is current behaviour moving; never edit it to pass.
- `test_layer_is_generic` fails, or passing it would need a project name in the layer or an
  allow-list entry.
- Any change to what the code writes, or a drop the refusal accepts dropping something different
  (README §5 rule 9).
- The change would open, write or touch any store under `var/`, or start Ray.

---

## 6. What this prompt does not do

- **Files it changes:**
  - `Datastore/SQL/schema.py` and `Datastore/SQL/ShardedPool.py`, as §2 says;
  - new: `Datastore/tests/test_drop_refuses_dangling_references.py`;
  - `Datastore/tests/test_reconcile_at_open.py`, the one re-fixture;
  - the log, this board, the README (§4), `prompts/datastore-generic/IMPLEMENTATION_STATE.md`
    (the issue) and `docs/OPEN_ISSUES.md`.

  Nothing else.
- It does not cascade, and it does not change the groups.
- It does not touch `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`
  (`a3-v2-readiness`).

---

## 7. The log and the board

`logs/01-refuse-a-drop-that-leaves-references.md`, using README §5.1. In addition, it gives:
- `dependent_tables`' ten lists, against README §0.2;
- the refusal's message, as raised for `gk-source-policy-records` alone;
- every test added, by class and method, and the re-fixture with its reason;
- each breakage (i)–(vi), as a diff, with what failed.

`IMPLEMENTATION_STATE.md`: the §1 row for 01, items D1–D3, §5.

`prompts/datastore-generic/IMPLEMENTATION_STATE.md` and `docs/OPEN_ISSUES.md`: the issue closed,
as §4.3 says.
