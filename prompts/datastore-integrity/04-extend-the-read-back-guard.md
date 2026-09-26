# Prompt 04 — extend the read-back guard to every read path, and to column existence

**Campaign:** [`README.md`](README.md) · **Board items:** **G1**–**G3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none. Independent of 01–03; `git status` must be clean.
**Closes:** `[01-read-batch-is-outside-the-guard]`, on the datastore-readback board's §4, with its
index row deleted. **Opens:** nothing expected. Every defect the extended guard finds is **recorded
in `KNOWN_UNFIXED`, not fixed**: prompt 05 fixes them against this list.
**Recommended model:** **Sonnet**. The analyser exists; this extends its reach. The care is in
making sure the extension cannot pass vacuously.

**Read first:**

1. [`README.md`](README.md): §0, §1, §2 (why the instrument comes before the remedy), §4 and §5.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §3.10–§3.12.
3. `Datastore/tests/test_factory_select_columns.py` in full: its analyser, its docstring's
   "WHAT THIS TEST CANNOT SEE", and `KNOWN_UNFIXED`, `EXPECTED_COVERAGE`, `KNOWN_BLIND` and the four
   tests that pin them.
4. `docs/datastore-integrity-audit/lookup_keys/probe_02_readback_guard_by_hand.py` and
   `probe_03_column_existence.py`, with their `.out` files. They are the method, run by hand; this
   prompt makes them tests.
5. The board entry for `[01-read-batch-is-outside-the-guard]` on
   `prompts/datastore-readback/IMPLEMENTATION_STATE.md`.

---

## 1. What is wanted

The guard checks that every column a `build()` reads off a query result was selected. It never
visits `read_batch`, which is on the live path, nor the `_build_impl_*` helpers that several
`build()` methods delegate their whole lookup to, nor whether a named column exists at all. Run by
hand over those paths, its own analyser finds a real defect (`WKB_criterio`, twice), and a
resolution of every `<table>.c.<name>` against the schema finds three more (audit §3.10).

This prompt makes both checks tests, and records what they find today. It is the instrument that
prompt 05 is scored against, so it lands first and fixes nothing.

---

## 2. What to change

This prompt **modifies `Datastore/tests/test_factory_select_columns.py`**, which README §5 rule 7
allows because this prompt names it: extending it is the task. No other existing test changes.

**G1 — every read path.**
- Visit every function of every factory class, not only those named `build`. That brings in the
  `_build_impl_*` helpers.
- For `read_batch`, link a nested helper's first parameter (`make_obj(row)`, `make_object(row)`) to
  the row variable of the comprehension or loop that calls it, so that its reads are charged to the
  query. `probe_02` does this by inlining; *prompt's choice* whether to inline or to link, provided
  the result is the same for every site `probe_02.out` lists.
- `OneLoopIntegral.read_batch` is a module-level function, not a method, and its helper is never
  called. Visit it too; if it cannot be decided, declare it in `KNOWN_BLIND` with that reason.
- Every new binding the analyser can now decide joins `EXPECTED_COVERAGE`; `probe_02.out` lists 16.
  A different count is not a failure, but the log explains every difference from that list.

**G2 — column existence.** A second test in the same module, or a new module beside it
(*prompt's choice*): every `<table>.c.<name>` in the factory sources, and every key of every dict
passed to an insert, resolves against the tables `build_schema(MetaData(), _factories)` builds.
This one may build the schema; it still needs no Ray, no engine and no store. Resolve a table
variable to its table the way `probe_03` does, and declare in `KNOWN_BLIND` what cannot be
resolved, with the reason.

**G3 — record, do not fix.**
- Every defect G1 and G2 find today goes into `KNOWN_UNFIXED`, keyed as the module already keys it,
  each naming the board item that will fix it (`P1`–`P8`, board §2). Expected, from the probes:
  the two `WKB_criterio` reads, `GkSource.py`'s `z_serial`, `QuadSource.py`'s `model_id`,
  `OneLoopIntegral.py`'s `validated` insert key. Anything else the guard finds is added the same
  way, and listed in the log as **new**.
- `test_known_unfixed_entries_are_still_present` already fails when an entry is fixed, so prompt
  05 is forced to strike each entry it fixes. Leave that test as it is.
- Rewrite the docstring's points 3 and 6, which this prompt removes, and say what the extended guard
  still cannot see: a label on the wrong column (audit §3.11's two `…_is_response`), a selected
  column that is never passed on (`friction`), and meaning (a correctly selected wavenumber serial
  used as an exit-time `store_id`, audit §3.2). Those are prompt 05's, found by its round-trip tests,
  not by this guard.

---

## 3. Verification

1. The extended guard passes on the tree, with `KNOWN_UNFIXED` as G3 records it.
2. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) rename a column read in one `read_batch`'s nested helper (say `TkNumericValue`'s) to one
     not selected;
   - (ii) rename a column read in one `_build_impl_serial`;
   - (iii) rename a `table.c.<name>` in any factory to a column that does not exist;
   - (iv) delete one `KNOWN_UNFIXED` entry without fixing it;
   - (v) make the analyser skip `read_batch` again. `test_analyser_coverage_is_undiminished` must
     fail, so the extension cannot be undone silently.

   Each in the log as a diff, exactly as applied, with the failing tests.
3. The Datastore suite, every existing test unchanged except the module this prompt names.

---

## 4. Acceptance

1. G1 and G2 in the tree; no Ray, no engine, no store, nothing under `var/`.
2. `KNOWN_UNFIXED` holds every defect found, each naming its P-item; the log lists them, marking
   any not in the probes as new.
3. The deliberate-breakage record, (i)–(v).
4. `[01-read-batch-is-outside-the-guard]` moved to the datastore-readback board's §4 with a closing
   note naming this commit; its index row deleted; the count and date corrected.
5. `black --check` clean. On the board: the §1 row, items G1–G3, §3.1, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- The guard finds a defect on a path `main.py` reaches during a build (not only on resume or
  extraction). Prompt 05 is held; a live-path defect may need to move ahead of it, and that is the
  user's call.
- The extension cannot be made without an analyser that runs factory code.
- Any other existing test would have to change.

---

## 6. What this prompt does not do

- It fixes no factory. Every finding goes to `KNOWN_UNFIXED`.
- It adds no round-trip test of `read_batch`; that is prompt 05's P9.
- It does not touch the schema witness or any factory registration.

---

## 7. The log and the board

`logs/04-extend-the-read-back-guard.md`, using the template in README §5.1. In addition: the
`EXPECTED_COVERAGE` additions against `probe_02.out`'s 16, the `KNOWN_UNFIXED` list with P-items,
and the `KNOWN_BLIND` additions with reasons.

`IMPLEMENTATION_STATE.md`: the §1 row for 04, items G1–G3, §3.1 and §5. Leave 05 held; its
release is the orchestrator's, once the user has decided D2 and D3.
