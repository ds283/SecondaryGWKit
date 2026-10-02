# Prompt 06 — count only real references when a store is retired

**Campaign:** [`README.md`](README.md) · **Board items:** **F1**–**F2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none. Independent of 00–05; `git status` must be clean.
**Closes:** `[10-store-retire-reports-every-tombstone-as-a-reference]`, on this board's §4, with
its index row deleted. **Opens:** nothing expected.
**Recommended model:** **Sonnet**. One target set narrowed, with an existing test that must not
move.

**Read first:**

1. [`README.md`](README.md): §0 item 3, §1, §5.
2. This board's §3 entry for `[10-store-retire-reports-every-tombstone-as-a-reference]`.
3. `RunRegistry/stores.py`: `_find_references`, `_names_one_of`, `_strings`, and the part of
   `retire_store` that calls `_find_references` and writes `references` into the tombstone.
4. `prompts/store-retirement/03-retire-a-store.md` §2 (the references report) and its §3 item 1,
   which requires a sidecar naming the store's **directory** to be found: the live A3 `backup`
   shape.
5. `RunRegistry/tests/test_store_retire.py`: `TestRetirement.test_a_retirement`.
6. `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` §1, the two retirements of 2026-10-02,
   and the tombstones they wrote beside
   `var/datastores/datastore-integrity-10-*.manifest.json` (read them; write nothing under `var/`).

---

## 1. What is wanted

`_find_references` takes three kinds of target: the primary, every file in the plan, and the
store's directory, `primary.parent`. The directory is there for a real case: the live A3 store's
sidecar named its backup by the backup's **directory**, and retiring the backup had to report that.
But every tombstone records `references.stores_root`, the directory it searched, and for a store
that sits directly in the stores root that is the same directory. So retiring a store in
`var/datastores/` reports every earlier tombstone there as naming it — two, then three, on
2026-10-02 — and the list grows by one per retirement. Nothing is refused or rewritten; the report
is wrong.

---

## 2. What to change

**F1 — the target set.** The stores root itself never names a store in it: it is the container of
every store, and a field naming it names none in particular. So when `primary.parent` is the stores
root (by `realpath`), it is not a target. A store in its own subdirectory, the backup's shape,
keeps its directory as a target, even when that subdirectory holds another store too, as
`test_a_retirement` requires.

*Prompt's choice*, if it is justified in the log: additionally skip a tombstone's own `references`
block, on the ground that it records what a retirement searched, not what it names. Do not do this
*instead* of F1; a non-tombstone sidecar with a field naming the stores root is the same false
match.

Update `_find_references`' docstring, and the `not_searched` text if what is searched changes, so
that both say what a directory match means.

**F2 — tests**, in a new module or a new class beside `TestRetirement` (*prompt's choice*), in the
existing fixture's temporary stores root:
- three stores directly in the stores root, retired one after another: no report lists an earlier
  tombstone;
- a non-tombstone sidecar in the stores root with a field naming the stores root: not reported;
- a store in a subdirectory, named by another sidecar's field by that subdirectory: reported, as
  now.

`test_a_retirement` stays unchanged and passes.

---

## 3. Verification

1. The RunRegistry suite with every existing test unchanged.
2. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) put `primary.parent` back as an unconditional target;
   - (ii) drop `primary.parent` as a target entirely: `test_a_retirement` must fail;
   - (iii) compare the stores root without `realpath` and give the fixture a symlinked stores root,
     or the equivalent the prompt finds (*prompt's choice*, provided it shows the comparison is not
     textual).

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. F1 in `RunRegistry/stores.py`, its docstring saying what a directory match means.
2. The F2 tests, and the deliberate-breakage record (i)–(iii).
3. `[10-store-retire-reports-every-tombstone-as-a-reference]` moved to this board's §4 with a
   closing note naming this commit; its index row deleted from `docs/OPEN_ISSUES.md` §1.17; the
   count and date corrected.
4. `black --check` clean. On this board: the §1 row, items F1–F2, §3, §4, §5. All in the same
   commit.

---

## 5. Stop conditions — stop and ask the user

- Any existing test would have to change.
- Either existing tombstone under `var/datastores/` records a reference that F1 would no longer
  find and that is a real one.

---

## 6. What this prompt does not do

- It rewrites no tombstone. The ones already written keep the references they recorded; they were
  correct as a record of what that code found.
- It does not change what `store retire` refuses, deletes or writes, only what it reports.

---

## 7. The log and the board

`logs/06-count-only-real-references.md`, using README §5.1. In addition: what `_find_references`
returns, before and after, for the second of the 2026-10-02 retirements, reconstructed in a
temporary stores root rather than read from `var/`.

`IMPLEMENTATION_STATE.md`: the §1 row for 06, items F1–F2, §3, §4 and §5.
