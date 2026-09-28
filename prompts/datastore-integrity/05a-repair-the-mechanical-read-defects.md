# Prompt 05a — repair the mechanical read-path defects: wrong names, a missing argument, tags, raises

**Campaign:** [`README.md`](README.md) · **Board items:** **P6**, **P7**, and **P9** for what they
touch · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 04 and 04b landed and reviewed (board §1); `git status` and `git diff --cached` clean.
First of the three prompts that together make prompt 05 (README §2): **05a → 05b → 05c**, one
commit each, reviewed one at a time.
**Closes:** no issue. These defects were found by the audit (§3.10–§3.11) and recorded by prompt
04's guard. D2 decided that they are fixed. **Empties:** `KNOWN_UNFIXED`, and two of
`KNOWN_UNFIXED_COLUMNS`' three entries. The third, `OneLoopIntegral`'s `validated`, is 05b's.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. Every change is one line or a few. The care goes into proving
each fix by a read that goes through it.

**Read first:**

1. [`README.md`](README.md): §0.1, §2 (the 05 row and why it is split), §5 and D2.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §3.10 and §3.11.
3. `Datastore/tests/test_factory_select_columns.py`: `KNOWN_UNFIXED`, `KNOWN_UNFIXED_COLUMNS` and
   the tests that pin them. This prompt strikes entries from them; it changes nothing else there.
4. `docs/datastore-integrity-audit/lookup_keys/common.py` and `probe_05_value_lookups_and_read_batch.py`
   with its `.out`: the in-memory method, and what each defect does today. Do not edit them.
5. The factories named in §2, at the functions named.
6. The board's reviews of 04 and 04b (§1), for what the guard now pins.

---

## 1. What is wanted

Six kinds of defect make a read path fail on its first real call. Nothing calls most of these paths
today, so nothing has noticed:
- **Two wrong attribute names.** `row_data.WKB_criterio`, missing its final `n`, is read in
  `sqla_GkWKBValue_factory._build_impl_serial` and `sqla_TkWKBValue_factory._build_impl_serial`.
  Every hit raises `AttributeError`.
- **Two wrong column names.** `table.c.z_serial` in `sqla_GkSourceValue_factory._build_impl_serial`,
  where the column is `z_source_serial`; the insert beside it already writes `z_source_serial`.
  And `quadsource_table.c.model_id` in `sqla_QuadSourceValue_factory._build_impl_model`, where the
  column is `model_serial`.
- **A missing argument.** `sqla_TkWKBValue_factory.read_batch` selects `friction` and never passes
  it to `TkWKBValue(...)`, which requires it. So every call raises `TypeError` (probe_05).
- **Tags that default to `None`.** `tags = payload.get("tags", None)` is followed by
  `for tag in tags`, so omitting tags raises `TypeError`. Every other lookup defaults to `[]`.
- **Strings raised.** `raise ("…")` raises `TypeError: exceptions must derive from BaseException`
  instead of its message.

---

## 2. What to change

**P6 — names and the argument.**
- `WKB_criterio` → `WKB_criterion` in both `_build_impl_serial`s (`GkWKBIntegration.py`,
  `TkWKBIntegration.py`).
- `table.c.z_serial` → `table.c.z_source_serial` in `GkSource.py`'s `_build_impl_serial` filter.
- `quadsource_table.c.model_id` → `quadsource_table.c.model_serial` in `QuadSource.py`'s
  `_build_impl_model`.
- `friction=row.friction` in `TkWKBValue.read_batch`'s `make_obj`. Check `TkWKBValue`'s
  constructor for any other required argument the read omits. If there is one, it is in scope;
  say so.

**P7 — tags and raises.**
- `payload.get("tags", None)` → `payload.get("tags", [])` at every site **except
  `OneLoopIntegral.py`'s**, which is 05b's (P8). Measured while writing this prompt, there are nine:
  - `GkNumericIntegration.py`: `GkNumericValue._build_impl_model` and `read_batch`;
  - `GkSource.py`: `GkSourceValue._build_impl_model`;
  - `GkWKBIntegration.py`: `GkWKBValue._build_impl_model` and `read_batch`;
  - `TkNumericIntegration.py`: `TkNumericValue._build_impl_model`;
  - `QuadSource.py`: `QuadSourceValue._build_impl_model`;
  - `TkWKBIntegration.py`: `TkWKBValue._build_impl_model` and `read_batch`.

  The audit says eleven. Recount by `grep`, and explain any difference in the log. Change each type
  annotation from `Optional[List[store_tag]]` to `List[store_tag]` with it.
- `raise ("…")` → `raise RuntimeError("…")` with the same message, at `GkSource.py`
  (`GkSourceValue._build_impl_serial`), `GkWKBIntegration.py` (`GkWKBValue._build_impl_serial`) and
  `TkWKBIntegration.py` (`TkWKBValue._build_impl_serial`). Sweep for any other `raise (` or
  `raise "` under `ObjectFactories/`; any you find is in scope.

**The guard's pins.** README §5 rule 7 allows edits to `Datastore/tests/test_factory_select_columns.py`,
named here, for one reason: to strike an entry this prompt's fix has made stale.
- Strike both `KNOWN_UNFIXED` entries, leaving `KNOWN_UNFIXED = {}`, and the `z_serial` and
  `model_id` entries of `KNOWN_UNFIXED_COLUMNS`. Rewrite the comments above both so they say what
  remains, and who owns it.
- Change **no other** pin. If the fixes move anything else in that module — `EXPECTED_COVERAGE`,
  `KNOWN_BLIND`, `KNOWN_UNRESOLVED_TABLES`, `KNOWN_UNCHECKED_INSERTS`, `KNOWN_BLIND_COLUMNS` —
  stop: it is unexpected, and it means the fix changed more than a name.

**Do not change** any filter, any key, any selected column list except by the renames above, any
registration, `main.py`, or any `extract_*.py` script.

---

## 3. Tests — a new module `Datastore/tests/test_read_path_mechanics.py`

No Ray, nothing under `var/`. Build the schema in memory with
`build_schema(MetaData(), Datastore.SQL.Datastore._factories)`, as `common.py` does. Insert parent
and value rows. Then call the **real** factory functions, through the path each fix is on:
1. **`WKB_criterion`.** For each of `GkWKBValue` and `TkWKBValue`, a `build()` by serial against an
   existing row returns that row's `WKB_criterion` value. Use a value distinct from every other
   column's.
2. **`z_source_serial`.** `GkSourceValue.build()` by `parent_serial` and `z_source` hits the stored
   row, and misses a row at another `z_source`.
3. **`model_serial`.** `QuadSourceValue.build()` by model, `q`, `r` and `z` hits the stored row, and
   misses a row stored under another model.
4. **`friction`.** `TkWKBValue.read_batch` returns every stored row with `friction` equal to the
   stored value, untagged and tagged. This is P9's round trip for this `read_batch`; 05c adds its
   parent-key cases.
5. **Tags omitted.** Each of the nine sites, called with no `tags` key, behaves exactly as with
   `tags=[]`.
6. **Raises.** Each of the three sites, reached with no stored row and no data payload, raises
   `RuntimeError` carrying its message.

Where a `build()` by serial compares the payload with the stored row, as the WKB value factories
do, supply the matching payload. Say in the log which comparisons each test passes through.

---

## 4. Verification

1. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) put `WKB_criterio` back in `TkWKBValue._build_impl_serial`. Both test 1 and the guard
     proper must fail;
   - (ii) put `z_serial` back. Both test 2 and G2 must fail;
   - (iii) remove `friction=row.friction`;
   - (iv) put `None` back as one tags default;
   - (v) put the string raise back at one site;
   - (vi) restore one struck `KNOWN_UNFIXED_COLUMNS` entry without undoing its fix. Its currency
     test must fail.

   **Every diff in the log is `git diff` output, taken while the mutation is applied and pasted
   unaltered.** After committing, extract each from the committed log to a file outside the
   repository and run `git apply --check`, `git apply`, the tests, and `git apply -R`.
2. All six suites, with README §7's command, run serially. Datastore rises by exactly the tests
   added, and the others match board §5.3. `black --check` is clean.

---

## 5. Acceptance

1. P6 and P7 in the tree; §3's module. No Ray, no store, nothing under `var/`.
2. `KNOWN_UNFIXED` empty. `KNOWN_UNFIXED_COLUMNS` holds only `OneLoopIntegral`'s `validated` entry,
   and no other pin has moved.
3. The deliberate-breakage record, (i)–(vi), replayed from the committed log.
4. The board: the §1 row for 05a, P6 and P7 marked done, P9's progress, and a new §5.4.
   `docs/OPEN_ISSUES.md` changes only if you open an issue.
5. One commit, with the log at `logs/05a-repair-the-mechanical-read-defects.md`.

---

## 6. Stop conditions — stop and ask the user

- A fix moves a guard pin other than the entries §2 names.
- An existing test fails. A fixture that relied on a wrong name is not this prompt's to change.
- A fix turns out to be more than the rename or default it looks like: for example, a `z_serial`
  that was meant to be some other column.

---

## 7. What this prompt does not do

- It does not touch `QuadSourceIntegral.py` or `OneLoopIntegral.py` (05b), or any lookup's key,
  `break_point_kind`, `z_init` or the `…_is_response` labels (05c).
- It does not edit the audit's probes, whose `.out` files record the tree they ran on.
