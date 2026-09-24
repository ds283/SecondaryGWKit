# Prompt 01 — one schema builder, and a read-only reader over a closed store

**Campaign:** [`README.md`](README.md) · **Board items:** **F1**, **F2**, **F3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[00-build-schema-reads-registration-before-its-none-check]`.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. The code is small. The judgement is in two places: proving a
refactor of the schema builder changed nothing, and proving a reader never writes.

**Read first:**

1. [`README.md`](README.md) in full, especially §3, §4 and §5.
2. [`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md): §1, §2, §3 and §7.
3. `Datastore/SQL/Datastore.py`:
   - the factory map (`:84-121`);
   - `__init__` (`:160-250`);
   - `_build_schema` (`:299-402`);
   - `_ensure_tables`, `_ensure_registered_schema` and `inventory` (`:777-804`).
4. `Datastore/SQL/ShardedPool.py`: `_read_closed_store` (`:667-704`), `_journal_paths`,
   `_resolve_shard_rows`, `_shard_file_problems`, `_create_engine` (`:259-311`) and
   `_SQLITE_JOURNAL_SUFFIXES`. **You do not change this file.**
5. `Datastore/shard_paths.py`, `Datastore/tests/shard_store_fixtures.py` and
   `tools/shard_key_audit.py` (its `mode=ro` pattern).
6. The four hand-copies of `_build_schema` in existing tests (audit §3). They are what a shared
   builder replaces for new code. **You do not modify them** (README §5 rule 7).
7. `CLAUDE.md`: "Repository mechanics".

---

## 1. What is wanted

Everything this campaign builds reads a store. Today the only way to read one is a full
`ShardedPool` open, and that open writes. Each `Datastore` actor, on an existing file:
- drops tables for `--drop`;
- creates missing tables;
- prunes unvalidated rows when asked;
- inserts a `version` row when the label is new (audit §2).

It also needs Ray. This prompt gives the campaign two foundations:

- **One schema builder.** Today the `Table` objects are built by an actor method,
  `Datastore._build_schema`, and new code cannot reach it without constructing an actor, whose
  `__init__` writes. The builder moves into a module function, and the actor calls it.
- **A read-only reader** over a closed store's shards. It opens every file `mode=ro` and builds the
  tables with that function. It needs no actor and no Ray, and it cannot write.

Nothing reads through the reader yet. Prompt 02 builds the structured inventory on it.

---

## 2. What to change

### F1 — `Datastore/SQL/schema.py`, `build_schema(metadata, factories)`

1. **Before you change any code, capture the schema as it is.** At `HEAD`, write a description of
   every table `_build_schema` builds into a test data file, `Datastore/tests/data/schema_at_base.json`.
   For each class it records:
   - whether the class has a table;
   - each column's name, type, nullability, primary-key flag, foreign-key targets, index and
     unique flags, in order;
   - the table's constraints and indexes;
   - the schema record's non-callable fields, such as `validate_on_startup`.

   Take the description from the **unchanged** code. The actor class is reachable without Ray as
   `Datastore.__ray_metadata__.modified_class`. Do **not** call its `__init__`, which writes; call
   `_build_schema` on an instance given only the attributes it reads. Record in the log exactly how
   you captured it, and the base SHA. This file is the witness that the refactor changed nothing,
   so it is written once and never regenerated to make a test pass.
2. **Move the logic into `build_schema(metadata, factories)`.** It returns the `Table` objects and
   the per-class schema records: everything `_build_schema` builds **except the inserters**, which
   are bound to the actor's `_insert` and stay in the actor. `Datastore._build_schema` becomes a
   call to it, plus the inserters. `self._tables`, `self._schema` and `self._metadata` hold the same
   things as before.
3. **Fix `[00-build-schema-reads-registration-before-its-none-check]`** while moving the code.
   `registration_data.get(...)` is read before the `is not None` check (`Datastore.py:314` against
   `:320`). No factory returns `None`, so behaviour does not change; the order does.
4. *Prompt's choice:* **the factory map may move into `schema.py`**, provided `Datastore.py`
   imports it from there and there is still one definition. Moving it is not required. The reader
   may import it from where it is.

### F2 — `Datastore/store_reader.py`, `open_read_only(primary)`

A context manager over a **closed** store. It yields an object that gives:
- the primary's path;
- for each shard: its serial, its path, a **read-only** SQLAlchemy engine, the `Table` objects,
  the set of **tables absent** from that shard file, and, per present table, the **columns
  absent** from the file and the **columns in the file that the code does not define**.

It disposes every engine on exit.

- **Shards are found through `ShardedPool._read_closed_store(primary, verb)`**, not a copy of its
  logic. It reads `shards` `mode=ro`, resolves every record with prompt 01's resolver of
  `datastore-portability`, and refuses a missing shard.
- **Every file is opened as `sqlite:///file:{path}?mode=ro&uri=true`**:
  - never read-write, because a read-write open replays a hot journal, which is a write;
  - never `immutable=1`, which would hide a concurrent writer instead of refusing it.
- **Refuse a store with a journal file** (`ShardedPool._journal_paths`) beside the primary or any
  shard, naming the file. Such a store is open or was not closed cleanly. The reader reports this
  and repairs nothing. *Prompt's choice:* refuse rather than warn, because a fingerprint of a store
  in that state describes no instant.
- **The tables are built once, with `build_schema`, into a fresh `MetaData`.** Absent tables come
  from `sqlite_master`, and absent or extra columns from `PRAGMA table_info`. *Prompt's choice:* the
  reader **reports** these and does not refuse. It is prompt 02's inventory that decides how an old
  store's missing table or column is recorded (README §6.3).
- **It never initialises Ray and never constructs an actor.** Importing the module may import
  `ray` transitively, through the factories; that is acceptable. Record in the log what importing
  it loads.
- It has **no write path**: no `create_all`, no `_ensure_tables`, no DDL, no DML.

### F3 — a real store for tests

`Datastore/tests/shard_store_fixtures.py` makes a primary with placeholder shards, which are text
files and cannot be read as databases. Add a **new** fixture module,
`Datastore/tests/real_store_fixtures.py`. It builds a small real store in a temporary directory,
with no Ray:
- a primary written as `ShardedPool` writes one, or through `write_new_store`;
- **two or more shard files**, each with every table from `build_schema`, created on a temporary
  read-write engine that the fixture then disposes;
- **replicated rows copied into every shard, with the same serials**;
- in each of at least two shards, at least one sharded compute target with `*_tags` association
  rows and `*Value` rows.

Prompt 02 extends it, so make the row content easy to add to. Also provide a way to make an
**old** store: one shard missing a table, and one table missing a column.

---

## 3. Tests — in `Datastore/tests/`, no Ray, nothing under `var/`

1. **The schema is unchanged.**
   - `build_schema` reproduces `schema_at_base.json` exactly, for every class.
   - So does the actor's `_build_schema`, called the same way as in F1 step 1. This shows that the
     actor delegates to the function and has not kept its own copy.
2. **The fix.** A factory whose `register()` returns `None` gets a schema record with no table, and
   no `AttributeError`.
3. **The reader reads.** On the F3 store it yields every shard with the right serial and path. Its
   per-table row counts equal an independent `sqlite3` `mode=ro` count of each file.
4. **The reader never writes.**
   - Take the SHA-256, size and `st_mtime_ns` of every file in the store's directory, and the
     directory listing.
   - Open the store with the reader and read every table on every shard.
   - Everything is unchanged, and no new file (such as a `-journal`) has appeared.
   - A write attempted through one of the reader's engines raises.
5. **Old stores.** The missing table and the missing column are reported, as the right names on
   the right shard. The read of every other table still succeeds.
6. **Refusals.** Each of these is refused with the file named, and with nothing written:
   - a journal file beside the primary;
   - a journal file beside a shard;
   - a missing shard, the refusal coming from `_read_closed_store`;
   - a primary with no `shards` table.
7. **No Ray.** In a child interpreter, opening the F3 store with the reader and reading every
   table leaves `ray.is_initialized()` false.

**Deliberate breakage.** Show that each of these mutations makes the tests written against it fail,
then restore it. Record each in the log as a diff, exactly as applied, and name the tests that
failed.

- (i) the reader opens files read-write, without `mode=ro`;
- (ii) `build_schema` omits one of the prepended columns;
- (iii) `_build_schema` keeps a private copy of the loop instead of calling `build_schema`, with one
  column's nullability changed;
- (iv) the absent-column check always returns nothing;
- (v) the journal refusal is removed;
- (vi) the `None` fix is reverted.

---

## 4. The demonstration — on a copy of a real store

**Never point the reader at an original or the backup** (README §3). Read-only snapshots with
`sqlite3` `mode=ro`, `shasum` and `stat` are the only contact with the originals.

1. Run `python -m RunRegistry list` and confirm nothing is `running`. Snapshot, read-only, the
   three stores and their sidecars: each file's size, `st_mtime_ns` and per-table row counts, and
   the primaries' and sidecars' SHA-256.
2. `cp -p` the sweep store's five files into `var/store-fingerprint-check-01/`.
3. Snapshot the copy's directory: each file's SHA-256, size and `st_mtime_ns`, and the listing.
4. Open the copy with `open_read_only`. Record:
   - the shards;
   - every table's row count per shard, set against an independent `sqlite3` `mode=ro` count;
   - every absent table, absent column and extra column on each shard.

   The sweep store was built before some columns existed, so report whatever is found; it is a
   finding, not a failure.
5. Re-take step 3. It must be identical, with no new file.
6. Re-take step 1. It must be identical.
7. **Delete `var/store-fingerprint-check-01/`.** Nothing from this section is committed. Throwaway
   scripts go in your scratch space.

---

## 5. Acceptance

1. §3's tests exist and pass, with no Ray and nothing under `var/`. Every existing test module is
   unmodified.
2. `schema_at_base.json` was captured from the unchanged code, and the log says how and at which
   SHA.
3. The deliberate-breakage record, (i)–(vi), each with its diff and the tests that failed.
4. §4 is done, with its numbers quoted. The working directory is deleted.
5. There is one schema builder. `Datastore._build_schema` calls `build_schema`.
   `git diff HEAD~1 HEAD -- Datastore/SQL/ShardedPool.py Datastore/SQL/ObjectFactories/ main.py RunRegistry/`
   is empty.
6. Every suite matches its baseline, except that `Datastore/tests` rises by exactly the tests you
   add.
7. `black --check` is clean.
8. On the board:
   - `[00-build-schema-reads-registration-before-its-none-check]` has moved to §4;
   - the §1 row for 01, items F1–F3 and §5's baselines are updated.

   In `docs/OPEN_ISSUES.md`, the row is deleted and the count and date are corrected. All of this
   goes in the same commit.

---

## 6. What this prompt does not do

- It does not change any factory, `ShardedPool`, `main.py`, `tools/`, `config/` or `RunRegistry/`.
- It does not change what an actor writes when it opens a store. `_ensure_tables`, the drop
  actions, the pruning and the `version` row stay as they are.
- It does not add or change any inventory. That is prompt 02.
- It does not replace the four hand-copies of `_build_schema` in existing tests.
- It does not fix any other defect in the audit's §6.

---

## 7. Stop conditions — stop and ask the user

- `build_schema` cannot reproduce `schema_at_base.json` exactly without changing what a factory
  registers.
- `_read_closed_store` or `_journal_paths` cannot be used as they are, so the reader would need
  `ShardedPool` to change.
- Any step of §4 changes a byte of the copy, or of an original, or leaves a new file.
- Opening a file `mode=ro` turns out to write anything at all.
- The reader cannot be kept free of `ray.init`.

---

## 8. The log and the board

`logs/01-a-read-only-store-reader.md`, using README §5.1's template. In addition, record:
- how `schema_at_base.json` was captured;
- what importing `Datastore/store_reader.py` loads;
- the §4 numbers, including every absent or extra column found on the sweep copy.

On `IMPLEMENTATION_STATE.md`: the §1 row for 01; items F1–F3; the closed issue moved to §4; §5's
baselines. Also update `docs/OPEN_ISSUES.md` in the same commit.
