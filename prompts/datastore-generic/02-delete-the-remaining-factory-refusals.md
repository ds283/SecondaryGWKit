# Prompt 02 — delete the remaining factory refusals

**Campaign:** [`README.md`](README.md) · **Board items:** **N1**–**N3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** after 01 has landed and been reviewed: both edit old-store refusals, and their test
modules overlap. `git status` clean.
**Closes:** nothing. **Narrows:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`.
**Opens:** nothing expected.
**Recommended model:** **Sonnet**. The pattern 01 set, applied to four more factories.

**Read first:**

1. [`README.md`](README.md): §0.1, §1, §5 (rules 7, 9, 11), §6.1 D2.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **B7**, **B10**.
3. `logs/01-delete-the-old-store-refusals-that-misfire.md` and 01's review: how 01 deleted its
   handlers and wrote its pass-through tests (its M4). Follow the same pattern unless the review
   says otherwise.
4. `Datastore/SQL/ObjectFactories/TkWKBIntegration.py`, `GkWKBIntegration.py`, `GkSource.py` and
   `QuadSourceIntegral.py`: the module docstrings and every site audit B7 lists.
5. The tests named in N2.

---

## 1. What is wanted

The same as 01, for the four factories whose old-store refusals test `e.orig` and so do not misfire
(audit B7). They exist only to give a store written before `a171ced`, `90d0114`, `3d7b606`,
`b18bb0b` or `014be95` a clearer message than SQLAlchemy's. Under D2 they go: such a store meets
SQLAlchemy's own `OperationalError` naming the missing column or table.

---

## 2. What to change

**N1 — the four factories.** Delete each handler, helper and comment that exists to refuse an older
schema, so that the error propagates unchanged:
- `TkWKBIntegration.py`: the refusals for no `numeric_serial` and no `rho_gauss_order` in `build`;
  those in `TkWKBValue.build` and `TkWKBValue.read_batch`; the module-docstring text on older stores.
- `GkWKBIntegration.py`: the same, at the sites audit B7 lists.
- `GkSource.py`: `_refuse_a_store_that_predates_the_parent_set` and its two calls; the docstring
  text on older stores; and the stale comment that "a datastore written before this change is
  readable by this factory" (audit B7), which is false at `HEAD`.
- `QuadSourceIntegral.py`: `_refuse_a_store_that_predates_the_policy_key`,
  `_refuse_a_store_that_predates_the_transfer_function_key`, their calls, and the comment that
  explains a column order chosen so that a pre-08 table gets 08's message. The column order itself
  may stay (*prompt's choice*; harmless either way, recorded in the log).

**N2 — the tests of what was deleted.** Delete, listing each in the log:
- `Datastore/tests/test_gksource_parent_set.py`: `TestAnOldStoreIsRefused` as a whole;
  `TestTheWKBValueReadsRefuseAPre07Store`, except `test_a_current_table_is_still_read`, which tests
  current behaviour and is **kept** (moved to a class of its own if its class goes; README §5 rule 7
  — the log says so);
- `Datastore/tests/test_wkb_numeric_parent_key.py`: `TestAnOldStoreIsRefused`;
- `Datastore/tests/test_quadsource_integral_parent_key.py`: `TestAnOldStoreIsRefused`;
- `Datastore/tests/test_quadsource_policy_key.py`: `TestAnOldStoreIsRefused`, except
  `test_another_database_error_is_not_turned_into_it`, whose property (an error passes through
  unchanged) still holds and is **kept**, rewritten as N3 says if its fixture was the old table;
- `Datastore/tests/test_quadsource_parent_key.py`: `TestAnOldStoreIsRead`. It asserts that a store
  from before prompt 09a is still read: a compatibility property with no code behind it, which D2
  withdraws.

`Datastore/tests/test_gksource_parent_set.py`'s `test_a_store_without_the_table_is_named_absent`
tests the **inventory** (audit B5), not a factory: **leave it** for 05. `real_store_fixtures.py`'s
`build_old_store` and its `missing_tables`/`missing_columns` machinery are still used by the reader
and inventory tests: **leave them** for 05.

Then search every suite for any other test asserting a deleted message, and list each; delete it if
it exists only for a deleted path, or stop and ask (§5).

**N3 — pass-through tests.** As 01's M4: for each of the four factories, a test, needing no Ray and
no datastore, that an unrelated database error raised by the factory's real statement reaches the
caller unchanged. One per factory is enough; the WKB value reads share their pattern.

---

## 3. Verification

1. The new tests pass. Every suite, with counts before and after, each difference accounted for.
2. `grep -nE 'predate|older store|written before' Datastore/SQL/ObjectFactories/` shows nothing
   that refers to a store's schema. Show the output.
3. `black --check` clean.
4. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) restore `_refuse_a_store_that_predates_the_parent_set` and its call in `GkSource.build`,
     changed to raise for every `OperationalError`;
   - (ii) in `QuadSourceIntegral`'s lookup, catch `OperationalError` and return a miss;
   - (iii) break the WKB value read that the kept `test_a_current_table_is_still_read` covers (make
     it filter on a column that does not exist), so that the kept test is shown to still guard it.

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. No refusal of an older schema left in the four factories.
2. The N2 deletions and keeps listed; the N3 tests; the deliberate-breakage record (i)–(iii).
3. `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` narrowed: a dated line naming
   what 02 deleted.
4. `black --check` clean. On this board: the §1 row, items N1–N3, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- A deleted handler also handled an error a current store can raise.
- A test outside N2's list asserts a deleted behaviour for a reason other than an old store.

---

## 6. What this prompt does not do

- It does not touch the factories 01 covered, `ShardedPool`, the reader or the inventory.
- It does not delete `real_store_fixtures.py`'s old-store machinery (05).
- It changes no table, column, key or index (README §5 rule 9).

---

## 7. The log and the board

`logs/02-delete-the-remaining-factory-refusals.md`, using README §5.1. In addition: every test
deleted or kept, by module, class and method; the suite counts before and after.

`IMPLEMENTATION_STATE.md`: the §1 row for 02, items N1–N3, §3, §5.
