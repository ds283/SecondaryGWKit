# Prompt 03 — one inventory service: the display and the run labels read the structured inventory

**Campaign:** [`README.md`](README.md) · **Board items:** **F8**, **F9** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[00-inventory-run-prunes-unvalidated-rows-by-default]` (this board), and the
`BackgroundModel` half of `qcd-background-audit`'s
`[03-qcd-inventory-does-not-report-the-representation]`, which closes that issue.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Decision D4** (README §6.2, the user, 2026-09-25): the recommendation was taken. This prompt
retires `ComputeTargets/tests/test_qcd_cosmology_inventory.py` and
re-expresses its claims against the new service (§2, F9 item 6).
**Recommended model:** **Opus**. Most of the change is deletion. The judgement is in two places:
a display that cannot hide a difference the inventory holds, and a retirement that leaves no
caller of the old service behind.

**Read first:**

1. [`README.md`](README.md) in full, especially §1, §4, §6.2 (D4) and §6.3.
2. [`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md): §1, §2 and §8.
3. Prompt 02 and its log, [`logs/02-a-structured-inventory.md`](logs/02-a-structured-inventory.md),
   especially "The key of every class, as shipped" and "State handed to the next prompt".
   `Datastore/store_inventory.py` in full.
4. The consumers:
   - `tools/inventory_report.py` in full;
   - `main.py`: the imports (`:75-86`), the `--prune-unvalidated`, `--drop`, `--inventory` and
     `--inventory-verbose` arguments (`:233-271`), the top-level statements from `parse_args`
     to `ray.init` (`:272-279`), and the pool and the `--inventory` branch (`:3484-3505`);
   - `extract_common.py`: `available_run_labels` (`:105-121`) and `resolve_run_selection`
     (`:182-194`). Every `extract_*.py` calls `resolve_run_selection(pool, args.run_label)` right
     after opening its pool, and `ComputeTargets/tests/test_run_identity.py:981-990` holds each
     script to that exact call.
5. What retires:
   - every factory's `inventory()`: **28** static methods in `Datastore/SQL/ObjectFactories/`,
     two in each file that also defines a `*Value` factory;
   - `Datastore.inventory` (`Datastore/SQL/Datastore.py:700-727`) and `InventoryConfigType`
     (`:165-168`);
   - `ShardedPool.inventory` (`Datastore/SQL/ShardedPool.py:1453-1536`), `_merge_queue`
     (`:1364-1451`), the constructor's `inventory_config` parameter (`:67`), `self._inventory_config`
     (`:240`) and the `InventoryConfigType` import (`:14`);
   - `config/sharding.py:43-94`: `inventory_config` and its three merge-policy dictionaries;
   - the callers that pass `inventory_config`: `main.py:82` and `:3498`, and
     `docs/source-remediation-verification/analyse_greens_and_source.py:46` and `:445` and
     `run_quadsource_integrals.py:58` and `:138`;
   - `ComputeTargets/tests/test_qcd_cosmology_inventory.py`, the only test of an `inventory()`
     (D4).
6. `prompts/qcd-background-audit/IMPLEMENTATION_STATE.md`, the entry for
   `[03-qcd-inventory-does-not-report-the-representation]` in its §3, and its §4 for the form of
   a closure.
7. `CLAUDE.md`: "Repository mechanics", especially that `main.py` cannot be imported, and
   `ComputeTargets/tests/test_main_plumbing.load_main_py_functions`.

---

## 1. What is wanted

Prompt 02 built the structured inventory beside the old one. The tree now has two inventory
services, and the old one is the only one anything uses. This prompt moves both consumers onto the
new one, then retires the old one completely.

- **The display.** `main.py --inventory` today builds the full read-write pool before it reaches
  its branch (audit §2). It needs Ray, it runs `--drop`, it creates missing tables, and by default
  it **deletes the store's unvalidated rows** (`[00-inventory-run-prunes-unvalidated-rows-by-default]`).
  What it prints is store-local serials, counts in place of the headline product, and no tags. It
  becomes a read of the closed store through `read_inventory`: no Ray, no pool, no write. It
  names work items by physical labels, and shows their tag sets and the store's named problems.
- **The run labels.** `extract_common.available_run_labels(pool)` reads
  `pool.inventory("store_tag")`, which serves a replicated class from **one shard chosen at
  random**. It reads the `store_tag` class of the structured inventory instead, which is compared
  across every shard.
- **One service.** Then the old `inventory()` methods, `Datastore.inventory`,
  `ShardedPool.inventory`, `_merge_queue` and `inventory_config` are deleted. No caller remains.

Nothing about what the store holds changes, and no lookup, key or schema changes.

---

## 2. What to change

### F8 — the display, and the run labels

1. **`tools/inventory_report.py` renders a `StoreInventory`.** It is rewritten around
   `format_inventory_report(inventory, db_name, verbose=False) -> str`, where `inventory` is what
   `read_inventory` returns. The three old shapes and their formatters go. For each class:
   - **a header**: the record count, the validated / unvalidated split where the class has the
     flag, the sum of `value_count` and the value table it counts where the class has one, and
     the timestamp range;
   - **its tag sets**, each with its labels and its record count. There are few: one run's
     products share one;
   - **its records, grouped under their tag set**, each rendered by its **physical labels**.
     A parent is rendered by its own resolved key (`StoreInventory.resolve`), never by its
     digest. Without `--inventory-verbose`, at most five records per tag set are shown, with the
     number not shown, as today;
   - **its problems, every one, never truncated.** The report opens with the total number of
     problems in the store, and says "none" when there are none.

   The categories stay close to today's. The `*Value` tables are no longer a category of their
   own; each is reported on its parent's header. The `store_tag` class lists every label, and
   **marks any label that no record carries** (audit §4: tags are created on `object_get`).
2. **Floats are rendered from their stored bits, typed by the schema.** A key's float leaves are
   `float.hex` strings (D1). *Prompt's choice:* the display learns which leaves are floats from
   the schema (`build_schema`'s column types), **never from the shape of a string**. It then
   renders `float.fromhex(value)`:
   - without `--inventory-verbose`, to six significant figures;
   - with it, as `repr`, which round-trips, so two records that differ in the last bit of a
     leaf render differently.

   Either the display consults the schema itself, or `ClassInventory` gains an **additive** field
   naming each class's float leaves, filled by `read_records` from the table it reads. Choose,
   and record which. Nothing about `Record`, `key`, `canonical` or any digest changes.
3. **The record order is physical.** Records sorted by canonical JSON are sorted by parent
   digests, which means nothing to a reader. The display orders each tag set's records by their
   resolved physical values, numerically where they are numbers.
4. **`main.py --inventory` reads the closed store, and nothing else.** Its branch moves to
   immediately after the `args.database is None` check (`:274-276`), **before `ray.init`**
   (`:279`), the `ProfileAgent` and the pool. It calls `read_inventory(args.database)`, prints
   `format_inventory_report(...)`, and exits 0. A reader refusal (a missing store, a journal, a
   missing shard) is printed to stderr, naming what was refused, and exits non-zero.
   - *Prompt's choice:* **`--inventory` never creates a store.** Today, a path that does not
     exist gets a new empty store, which is a write. Now the reader refuses it.
   - *Prompt's choice:* **`--inventory` with `--drop` is refused**, with a message, because a user
     who asks for both expects the drop to happen and it no longer will. Every other write-side
     argument (`--prune-unvalidated`, `--shards`, `--db-timeout`, `--profile-db`) is ignored by
     `--inventory`, and its help text says so.
   - The branch after the pool, `:3501-3505`, is deleted. The `format_inventory_report` import
     stays. Nothing else in `main.py` changes except the `inventory_config` retirement (F9).
5. **`extract_common.available_run_labels(pool)` reads the structured inventory.** Its signature
   stays, because every extract script calls `resolve_run_selection(pool, …)` with its open pool
   and `test_run_identity` holds them to it.
   - It reads `read_inventory(pool.primary)` and returns, as today, the sorted names from the
     `store_tag` records' labels that begin with `RUN_LABEL_TAG_PREFIX`.
   - **`ShardedPool` gains one read-only property, `primary`**, returning `self._primary_file`.
     It is the only change to `ShardedPool` besides F9. A module reaching into `_primary_file`
     from outside would be the alternative, and a worse one.
   - *Prompt's choice:* **the answer does not change.** It is still every `Run_` label in
     `store_tag`, including one that no record carries. Whether a run with no products should
     count is a question about run identity, not about the inventory. Record the difference if
     §4 finds one.
   - *Prompt's choice:* **a named problem in the `store_tag` class is a refusal.** If the
     class's copies diverge across shards, which runs the store holds is ambiguous. The refusal
     names the problem. Problems in other classes do not affect run labels.
   - **The pool is open when this runs**, which the reader does not assume. The extract scripts
     call it after the constructor has returned. The constructor waits on every actor
     (`ShardedPool.py:222-224`), so each actor's opening writes have committed, and nothing
     writes until the script's first `object_get`. The reader's refusal of a journal still
     guards the case where something does. State this in the function's docstring.

### F9 — retire the old service

1. **Delete every factory's `inventory()`.** That is 28 static methods, **and nothing else in
   any factory file**. Imports that only `inventory()` used may go. `inventory_records`, `build`,
   `store`, `read_batch`, `validate`, `validate_on_startup` and `register` do not change. The
   factory diff is deletions only, except where an import line has to be rewritten.
2. **Delete `Datastore.inventory` and `InventoryConfigType`** from `Datastore/SQL/Datastore.py`.
   Nothing else in that file changes.
3. **Delete `ShardedPool.inventory`, `_merge_queue`, the constructor's `inventory_config`
   parameter, `self._inventory_config`, and the `InventoryConfigType` import.** Add `primary`
   (F8 item 5). Nothing else in `ShardedPool.py` changes: not the open, copy, move, routing or
   `read_table` paths.
4. **Delete `inventory_config` and its merge policies from `config/sharding.py`**, with their
   comment block. Nothing else in that file changes.
5. **Remove `inventory_config` from its four callers**: the import and the keyword argument in
   `main.py`, `analyse_greens_and_source.py` and `run_quadsource_integrals.py`. Nothing else in
   the two `docs/` scripts changes (README §1).
6. **D4: retire `ComputeTargets/tests/test_qcd_cosmology_inventory.py`**, and re-express its three
   claims against the new service in a new test module:
   - two `QCD_Cosmology` rows that differ only in `T_z_representation` are two different records,
     and render differently;
   - in the rendered record, `T_z_representation` sits beside `log10_max_z` (its prompt 07 §2
     item 3), or the log says why the new display orders fields another way;
   - two rows with the same representation and different `name` are two records only if they
     differ in the key. `name` is not in the key, so they are one key held twice, and the
     inventory names it a `duplicate`. **This claim changes meaning under the new service**, and
     the log says so. The old test asserted two labels; the new truth is one key and a named
     problem.

   The module's hand-copy of `_build_schema` (audit §3) goes with it. The other three hand-copies
   stay (README §5 rule 7).
7. **Nothing else may refer to the retired names.** After this prompt,
   `git grep -nE '\.inventory\(|def inventory\(|inventory_config|_merge_queue|InventoryConfigType' -- '*.py'`
   prints nothing.

---

## 3. Tests — no Ray, nothing under `var/`

Put them in `Datastore/tests/` (the display, the retirement, the run labels) and
`ComputeTargets/tests/` (the replacement for the D4 module). Use `build_full_store` and its
helpers from `Datastore/tests/real_store_fixtures.py`.

1. **The display says what the inventory holds.** On the full store:
   - every class appears, with its count, its tag sets and, where it has them, the validated
     split and the value-count sum, each equal to what the `StoreInventory` says;
   - with `verbose=True`, **every two distinct records of a class render as distinct lines.**
     This is the test that the display drops no key field: a display that leaves out one field
     makes two records that differ only in it collide;
   - no 64-hex digest appears anywhere in the output;
   - every problem appears, in full, when there are many. Build a store with more than five of
     them.
2. **Floats.** Two stores differing only in the last bit of one `k` render differently in verbose
   mode, and the same at six figures. A string leaf that happens to look like a hex float, such
   as a label, is rendered as the string it is.
3. **The representation, and the grid identity.** A `BackgroundModel` record shows
   `source_grid_digest` and `source_grid_construction`. A `QCD_Cosmology` record shows
   `T_z_representation`. These two claims close `[03-qcd-inventory-does-not-report-the-representation]`
   (F9 item 6 and §5.8).
4. **`main.py --inventory` is read-only and needs no Ray.** Run `main.py --database <full store>
   --inventory` in a child interpreter from the repository root, **through a guard**: the child
   replaces `ray.init` with a function that raises, then runs `main.py` with `runpy.run_path(...,
   run_name="__main__")`. A `ray.init` reached by mistake then fails at once, whatever cluster
   happens to be running. Then:
   - it exits 0, and prints the report;
   - the store's directory is unchanged: prompt 01's `file_state`, hashes, sizes, mtimes and
     listing;
   - **the store's unvalidated rows are all still there.** This is the test of the issue it
     closes;
   - with `--drop <any group>` it is refused and exits non-zero, and nothing is written;
   - on a path that does not exist it exits non-zero, and **no file appears**.
5. **The branch comes first.** Using `ast` on `main.py`, as `load_main_py_functions` reads it, the
   top-level `if args.inventory:` statement comes before the `ray.init` call and before the
   `with ShardedPool(...)` statement, and nothing between `parse_args` and it constructs a pool,
   an actor or an engine.
6. **The run labels.**
   - Against a stand-in pool whose `primary` is a full store with two `Run_` labels,
     `available_run_labels` returns both, sorted, and ignores labels without the prefix.
   - A store whose `store_tag` copies diverge on one shard is refused, naming the problem.
   - `ShardedPool.primary` returns `_primary_file`. Test it on an object made with
     `ShardedPool.__new__`, so that no actor or Ray is involved.
   - `extract_common` no longer calls `pool.inventory`.
7. **The old service is gone.** Every factory in `_factories` lacks `inventory`; `Datastore`'s
   class lacks it; `ShardedPool` lacks `inventory` and `_merge_queue`, and its constructor has no
   `inventory_config` parameter (`inspect.signature`); `config.sharding` has no
   `inventory_config`. A test runs §2 F9 item 7's `git grep` from the repository root, and
   requires that it prints nothing.

**Deliberate breakage.** Each of these must fail the tests written against it. Record each diff
exactly as applied, with the tests that failed.

- (i) the `--inventory` branch is moved back after `ray.init`;
- (ii) the display leaves `source_grid_digest` out of the `BackgroundModel` record;
- (iii) verbose mode renders floats to six significant figures;
- (iv) the display shows at most five problems;
- (v) a parent is rendered by its digest;
- (vi) `available_run_labels` ignores a problem in `store_tag`;
- (vii) `inventory_config` is restored to `config/sharding.py`.

---

## 4. The demonstration — on copies of the sweep store

Never an original (README §3). Work in `var/store-fingerprint-check-03/`, snapshot the originals
before and after, and delete the directory at the end.

1. `python -m RunRegistry list`: nothing `running`. Snapshot the originals read-only, as in prompt
   02 §4.
2. `cp -p` the sweep store's five files into `var/store-fingerprint-check-03/old/` and
   `var/store-fingerprint-check-03/new/`.
3. **The old report, before any code changes**, at the base SHA: run
   `main.py --database <old copy> --inventory --no-prune-unvalidated` through a local Ray, as
   `datastore-portability`'s demonstrations did, and stop that Ray afterwards. Keep the output in
   your scratch space. This is the last time the old service runs, and it is the only cross-check
   of the new display against it.
4. **After the change**, run `main.py --database <new copy> --inventory` through §3 test 4's
   guard, with no Ray running.
   - Record the wall time, and that the copy's directory is byte-identical before and after.
   - Set each class's count, validated split and value-row total against the old report's. They
     must agree. The old report counted value tables directly; the new one sums the parents'
     `value_count`, so any difference is a finding, and is reported.
   - Quote the new report's header, one class in full (`QuadSourceIntegral`, non-verbose), and
     its `BackgroundModel` record.
5. **Verbose.** Run it again with `--inventory-verbose`. Record the output's size and its wall
   time. Confirm that the number of `QuadSourceIntegral` record lines is 7 706.
6. **The run labels.** Call `available_run_labels` on a stand-in pool whose `primary` is the new
   copy, and record the answer. Set it against `store_tag`'s labels from an independent `sqlite3`
   `mode=ro` read of each shard. Report any `Run_` label that no record carries.
7. Re-take the snapshot of the originals. It must be identical.
8. Delete `var/store-fingerprint-check-03/`.

---

## 5. Acceptance

1. §3's tests exist and pass, with no Ray and nothing under `var/`.
2. The deliberate-breakage record, (i)–(vii).
3. §4 is done, with its numbers, and the working directory is deleted.
4. §2 F9 item 7's `git grep` prints nothing.
5. **Scope.** `git diff HEAD~1 HEAD --stat` touches only:
   - `tools/inventory_report.py`, `main.py`, `extract_common.py`, `config/sharding.py`;
   - `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py` and the factories, as F9 says;
   - `Datastore/store_inventory.py`, only if F8 item 2's additive field is the choice;
   - the two `docs/source-remediation-verification/` scripts, for `inventory_config` only;
   - the retired test module (D4), the new test modules, and test data;
   - the log, this board, `docs/OPEN_ISSUES.md`, and `prompts/qcd-background-audit/IMPLEMENTATION_STATE.md`.

   In the factory diff, no line of `inventory_records`, `build`, `store`, `read_batch`,
   `validate`, `validate_on_startup` or `register` changes. `RunRegistry/` is untouched.
6. Every suite matches its baseline, except:
   - `Datastore/tests` rises by exactly the tests you add there;
   - `ComputeTargets/tests` falls by the retired module's three tests and rises by exactly the
     tests that replace them.

   `black --check` is clean.
7. **This board:** `[00-inventory-run-prunes-unvalidated-rows-by-default]` moves to §4; the §1 row
   for 03; items F8–F9; any issue opened; §5's baselines.
8. **`qcd-background-audit`'s board:** `[03-qcd-inventory-does-not-report-the-representation]`
   moves to its §4, with a closure line naming this commit, the test, and that its `QCD_Cosmology`
   half's test was re-expressed under D4.
9. **`docs/OPEN_ISSUES.md`**, in the same commit: both rows deleted, and the count and date
   corrected.

---

## 6. What this prompt does not do

- It does not compute a fingerprint, or touch `RunRegistry/`. That is prompt 04.
- It does not change any key, lookup, schema, `Record`, `canonical` or digest. It does not
  change what `read_inventory` returns, except for F8 item 2's optional additive field.
- It does not change the extract scripts, `resolve_run_selection`, `choose_run_label`, or what
  counts as a run.
- It does not change `ShardedPool` beyond F9 item 3, or `Datastore.py` beyond F9 item 2.
- It does not fix the audit's other §6 defects.

---

## 7. Stop conditions — stop and ask the user

- The old report and the new one disagree on a count in §4 in a way the log cannot explain from
  the definitions.
- A caller of the retired service exists that §2 does not name.
- Retiring anything would change a factory's `build`, `store`, `read_batch`, `validate_on_startup`
  or `register`, or a lookup.
- `available_run_labels` cannot keep its signature, so an extract script would have to change.
- `main.py --inventory` cannot be kept free of `ray.init`, or writes anything.

---

## 8. The log and the board

`logs/03-one-inventory-service.md`, using README §5.1's template. In addition, record:
- F8 item 2's choice, and what the display looks like: the header, one class, one record;
- each *prompt's choice*, and whether it was kept;
- every retired name, with the file it was in;
- the §4 numbers, with the old and new counts side by side.

On `IMPLEMENTATION_STATE.md`: the §1 row for 03; items F8–F9; the closed issue moved to §4; §5's
baselines. On `qcd-background-audit`'s board, the closure. Update `docs/OPEN_ISSUES.md` in the same
commit.
