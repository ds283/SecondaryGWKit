# Prompt 05b — the integral `read_batch`es: the tagged join, the exit times' serials, `OneLoopIntegral`

**Campaign:** [`README.md`](README.md) · **Board items:** **P1**, **P2**, **P8**, and **P9** for what
they touch · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 05a landed and reviewed; `git status` and `git diff --cached` clean.
**Closes:** `[00-tagged-read-batch-joins-an-unselected-alias]` and
`[00-oneloop-lookup-joins-the-wrong-tag-table]`, both on the store-fingerprint board's §4, with
their index rows deleted.
**Narrows:** `[00-oneloop-factory-cannot-round-trip-a-row]` (this board, §3.2): its `validated`
half is fixed here, and its `source_serial` half stays open, as D3 decided.
**Empties:** `KNOWN_UNFIXED_COLUMNS`.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. Each change is small. The care goes into the tests, which must
tell a right answer from a plausible one: the P2 bug is a wrong serial that looks right.

**Read first:**

1. [`README.md`](README.md): §0.1, §2, §5, and **D3**. D3 bounds what may change in
   `OneLoopIntegral`'s factory; read it before touching that file.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §3.1, §3.2 and §3.12.
3. `docs/datastore-integrity-audit/lookup_keys/probe_01_tagged_read_batch.py`, `probe_01b_fix_shape.py`
   and `probe_06_oneloop_tags_and_redshift_zero.py`, with their `.out` files. Do not edit them.
4. `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` `read_batch`, and all of
   `OneLoopIntegral.py`. Also `ComputeTargets/OneLoopIntegral.py`'s constructor, and
   `Datastore/SQL/Datastore.py` `object_read_batch`, which calls `factory.read_batch`.
5. The two issues' entries on `prompts/store-fingerprint/IMPLEMENTATION_STATE.md` §3, and
   `[00-oneloop-factory-cannot-round-trip-a-row]` on this board.
6. `Datastore/tests/test_factory_select_columns.py`, and the board's reviews of 04, 04b and 05a.
   This prompt moves pins there; §2 lists which.

---

## 1. What is wanted

- **P1 — every tagged `QuadSourceIntegral` `read_batch` fails.** The tag join is written
  `tab.c.parent_serial == query.c.serial`. `query.c` makes SQLAlchemy name an anonymous subquery
  outside the `FROM`, and SQLite raises `no such column: anon_1.serial`. `extract_QuadSourceIntegral_data.py`
  passes the run label as a tag whenever the store records a run, so **extraction from any
  run-labelled store fails, A3 v2 included.** The module-level `OneLoopIntegral.read_batch` has the
  same line.
- **P2 — the same method gives its exit times the wrong serial.** It selects the *wavenumber*
  serials as `q_serial` / `r_serial`, and uses them both for the wavenumbers, which is right, and as
  the exit times' `store_id`, which is wrong.
- **P8 — `OneLoopIntegral`'s factory, bounded by D3.**
  - `build()` joins `QuadSourceIntegral_tags`, while `store()` writes `OneLoopIntegral_tags`.
  - `read_batch` is a module-level function, so `Datastore.object_read_batch`, which calls
    `factory.read_batch`, cannot reach it. It also has no `return`.
  - Its tags default to `None`.
  - `store()` writes a `validated` key to a table with no such column, which SQLAlchemy silently
    drops.

  **Not** in scope, by D3: that every hit raises `KeyError: 'source_serial'`, because the compute
  target reads a payload key the factory never supplies and the table has no column for. That, and
  how `OneLoopIntegral` is keyed, wait until it is built.

---

## 2. What to change

**P1.** In both `read_batch`es, `tab.c.parent_serial == table.c.serial`, as `probe_01b` verified.

**P2.** In `QuadSourceIntegral.read_batch`, select each exit time's own serial:
`q_exit_tab.c.serial.label("q_exit_serial")` and the same for `r`. Use it as `store_id` in
`q_exit_payload` / `r_exit_payload`. The wavenumbers keep `q_serial` / `r_serial`. Check the rest of
`make_object` for any other value taken from the wrong table's serial. Any you find is in scope, and
the log names it.

**P8**, D3's list and nothing more:
- `build()` joins `tables["OneLoopIntegral_tags"]`;
- `read_batch` becomes a `@staticmethod` of `sqla_OneLoopIntegral_factory`, with the P1 join, tags
  defaulting to `[]`, and `return [make_object(row) for row in row_data]` (or equivalent). Remove
  the `# TODO: not yet complete` only if nothing it refers to remains;
- `"validated": False` is removed from `store()`'s insert;
- the comment in `inventory_records` that calls the tag join "not fixed here" is corrected.

Leave `source_serial` alone: in the compute target, in `build()`'s payload and in `make_object`'s.

**The guard's pins.** README §5 rule 7 allows edits to `Datastore/tests/test_factory_select_columns.py`,
named here, for one reason: to follow a factory change that has made the pin fail. For each edit,
the log shows the failing assertion that forced it. The movements this prompt expects:
- `KNOWN_UNFIXED_COLUMNS`: strike `OneLoopIntegral`'s `validated`, leaving it empty;
- `KNOWN_BLIND`: the `OneLoopIntegral.py:<module-level>.read_batch…` entry goes, because the
  function is now a method whose helper is called;
- `EXPECTED_COVERAGE`: one entry joins, for the new method's inlined row binding. It is
  `OneLoopIntegral.py:sqla_OneLoopIntegral_factory.read_batch.row_data` if `make_object` keeps its
  parameter name;
- `KNOWN_UNRESOLVED_TABLES`: three pairs go. They are `QuadSourceIntegral`'s `read_batch` with base
  `query`, and the module-level `OneLoopIntegral.read_batch` with bases `query` and `table`. As a
  method, its `table` resolves to `OneLoopIntegral`, so G2 now checks its 13 `table.c.*`
  references. They were resolved by hand in the 04 review, and all name real columns.

The orchestrator simulated these edits against the guard before this prompt was written, and got
exactly these movements and no others. A movement not on this list is a stop (§6).

**Do not change** `QuadSourceIntegral.build`, any filter of either `read_batch` beyond the join, any
registration or schema, `main.py`, or `extract_QuadSourceIntegral_data.py`.

---

## 3. Tests — a new module `Datastore/tests/test_integral_read_batch.py`

No Ray, nothing under `var/`. The schema is built in memory, as `common.py` does, and the calls go
to the real factory code.
1. **Tagged `QuadSourceIntegral.read_batch`** (probe_01's design). Store three rows, tagged `{A, B}`,
   `{A}` and none. Tags `[]` give all three, `[A]` gives two, `[A, B]` gives one. Each is compared by
   serial, and nothing raises.
2. **Exit-time serials (P2).** Give every exit-time row a serial different from its wavenumber's.
   Each returned integral's `q` and `r` exit times carry their exit-time row's serial, and their
   wavenumbers carry the wavenumber's serial. Without the distinct serials this test cannot fail,
   so say in the log that they are distinct.
3. **`OneLoopIntegral.build` tags.** Two cases:
   - a row tagged in `OneLoopIntegral_tags` is found by its tags;
   - an untagged row whose serial coincides with a `QuadSourceIntegral_tags.parent_serial` is **not**
     matched by those tags, which is probe_06's coincidence case.

   A hit constructs the compute target, and the real constructor raises on `source_serial`. So
   patch the name `OneLoopIntegral` in the factory module with a stand-in that records its
   arguments, scoped by `unittest.mock.patch` (README §5 rule 9).
4. **`OneLoopIntegral.read_batch`.** It is reached as `sqla_OneLoopIntegral_factory.read_batch`,
   the attribute `Datastore.object_read_batch` calls. With the stand-in, it returns a list with the
   right rows untagged, tagged, and with tags omitted.
5. **The deferred defect stays visible.** Constructing the **real** `OneLoopIntegral` from what
   `read_batch` passes raises `KeyError: 'source_serial'`. The test's docstring names
   `[00-oneloop-factory-cannot-round-trip-a-row]`, and says that whoever fixes it must change this
   test.

Tests 1–4 are P9's round trips for these two `read_batch`es.

---

## 4. Verification

1. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) put `query.c.serial` back in `QuadSourceIntegral.read_batch`;
   - (ii) use `row.q_serial` as the `q` exit time's `store_id` again;
   - (iii) point `OneLoopIntegral.build` back at `QuadSourceIntegral_tags`;
   - (iv) delete `read_batch`'s `return`;
   - (v) put `"validated": False` back into `store()`. G2 must fail;
   - (vi) put `None` back as `OneLoopIntegral.read_batch`'s tags default.

   **Every diff in the log is `git diff` output, taken while the mutation is applied and pasted
   unaltered**, then replayed from the committed log: `git apply --check`, `git apply`, the tests,
   `git apply -R`.
2. All six suites, with README §7's command, run serially. Datastore rises by exactly the tests
   added, and the others match board §5.4. `black --check` is clean.

---

## 5. Acceptance

1. P1, P2 and P8 in the tree; §3's module. No Ray, no store, nothing under `var/`.
2. `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are both empty. Every other pin movement is one of
   §2's, shown forced.
3. The deliberate-breakage record, (i)–(vi), replayed from the committed log.
4. The two issues closed on the store-fingerprint board's §4, each with a closing note naming this
   commit by subject, and their index rows deleted. `[00-oneloop-factory-cannot-round-trip-a-row]`
   is narrowed here and in the index. The index's count and date are corrected.
5. The board: the §1 row for 05b, P1, P2 and P8, P9's progress, §3.1, and a new §5.5.
6. One commit, with the log at `logs/05b-repair-the-integral-read-batches.md`.

---

## 6. Stop conditions — stop and ask the user

- A guard pin moves in a way §2 does not list.
- Making `read_batch` a method, or fixing its join, would need a change D3 excludes, such as
  `source_serial`, a key column or the schema.
- An existing test fails.

---

## 7. What this prompt does not do

- It does not key `QuadSourceIntegral` on its policy (08) or its parents (09), and it does not
  touch `OneLoopIntegral`'s keying (out of scope, README §1).
- It does not touch the value factories (05a, 05c).
