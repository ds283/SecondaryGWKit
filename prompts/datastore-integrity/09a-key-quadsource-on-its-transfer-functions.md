# Prompt 09a — `QuadSource` is keyed on its transfer functions, and the inventory resolves across shards

**Campaign:** [`README.md`](README.md) · **Board items:** **K4**, and **K7**'s cross-shard `Parent` ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Charter:** [`09-key-derived-rows-on-their-parents.md`](09-key-derived-rows-on-their-parents.md),
split on 2026-09-30 by the user's decision into 09a → 09b → 09c
([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md) §0.1). Its §1, §2 and §6 bind this
prompt where they concern `QuadSource` and the inventory's cross-shard parents; this file says
what is 09a's.
**Gate:** 08 landed and reviewed (board §1); `git status` is clean. D1, D4 and D5 are decided
(README §6.2), and so are the user's four decisions of 2026-09-30 (orchestrator note §0.1–§0.4).
**Closes:** nothing. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet**. One existing key is narrowed onto two existing columns, and the
inventory learns one small form. The care is in `main.py`'s two stages and in missing no reader.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §4, §5 (rules 7, 11, 12) and §6.2 **D1** in full, with *What
   D1 is about* and the user's point about looking rows up by physical labels alone; **D4**, **D5**.
2. The charter, prompt 09, in full, and the orchestrator note's §0 as the dispatch brief gives it.
3. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §2.4 and §2.6.
4. [`logs/07-key-wkb-rows-on-their-numeric-row.md`](logs/07-key-wkb-rows-on-their-numeric-row.md)
   and [`logs/08-key-quadsourceintegral-on-its-policy.md`](logs/08-key-quadsourceintegral-on-its-policy.md):
   a parent carried through a factory, a target, `main.py`, the readers and the fingerprint, and
   07's `labels_only` reader mode.
5. `Datastore/SQL/ObjectFactories/QuadSource.py`: `register`, `build`, `store`, `inventory_records`.
   `GkSourcePolicyData.py` is the precedent: it is keyed on its parent `GkSource`.
6. `ComputeTargets/QuadSource.py`: the constructor, `compute()` (which receives `Tq` and `Tr`) and
   `store()`.
7. `Datastore/store_inventory.py`: `Parent`, `ShardContext`, `read_records` and `read_inventory`.
8. `main.py`: `build_tensor_source_work` (the `QuadSource` stage) and, in
   `build_QuadSourceIntegral_batch`, the `QuadSource` lookup and the `TkNumericIntegration` lookup
   after it.
9. The `QuadSource` readers: `extract_tensor_source_data.py`, `extract_QuadSourceIntegral_data.py`
   (its `QuadSource` lookup only) and `docs/source-remediation-verification/run_quadsource_integrals.py`
   (its `QuadSource` lookup only).

---

## 1. What is wanted

A `QuadSource` is computed from exactly two rows, the `TkNumericIntegration`s for `q` and `r`
(`compute()` receives `Tq` and `Tr`, and `store()` records their serials in `Tq_serial` and
`Tr_serial`). Its lookup names neither: `build()` filters on the model, `q`, `r` and the tags. So a
change to the Tk rows under the same tags serves the old source as a match (audit §2.4). After this
prompt the lookup names both, as `GkSourcePolicyData`'s names its `GkSource`.

`Tr` is on `r`'s shard, not the source's (`QuadSource` shards on `q`), so the fingerprint can make it
identity only once the inventory resolves a parent on another shard. That facility is K7's first
half, and it lands here, because `Tr` is the campaign's first cross-shard key parent. K7's second
half, a parent that is a *set* of rows, is 09b's.

---

## 2. What to change

**K4 — the key** (`Datastore/SQL/ObjectFactories/QuadSource.py`).
- `Tq_serial` (on the source's shard, with a foreign key since 06) and `Tr_serial` (on `r`'s shard,
  no foreign key) join `build()`'s filter. **`Tr_serial` becomes indexed**; its comment keeps
  saying why it has no foreign key. Neither gains a column.
- How a lookup names them is *your choice*: two serials, or the two `TkNumericIntegration` objects.
  Say which, and why.
- **The D1 reader mode.** A caller holding only physical labels may still select by them, provided
  the lookup **refuses when more than one stored row matches**, naming the `Tq_serial` and
  `Tr_serial` of each. Follow 07's precedent (`labels_only=True` in the WKB factories' `build()`),
  and name the mode the same way unless you say why not. A lookup that names neither the parents
  nor the mode raises.
- `inventory_records` declares both parents: `Tq` as a same-shard `Parent`, `Tr` as a cross-shard
  one (below). Its comment, which says the two serials are "provenance, not identity", is corrected.

**K4 — the compute target** (`ComputeTargets/QuadSource.py`).
- A `QuadSource` made to be computed carries the two serials it is keyed under, from its
  construction. `compute()` **asserts** that the `Tq` and `Tr` it receives are those rows, and a
  mismatch raises naming both serials. Charter §2's rule: a derived row is keyed on the rows its
  `compute()` receives, and nothing checks that today.
- **`q == r`.** `itertools.combinations_with_replacement` gives the diagonal, so `Tq` and `Tr` are the
  same row. Say, and test, how the lookup, the assertion and the inventory behave then.

**K7 — the cross-shard parent** (`Datastore/store_inventory.py`).
- `Parent` gains a cross-shard form. It resolves the serial against **every** shard's records of
  the parent class, which the driver has already built: `read_inventory` builds each class on every
  shard before the next, in dependency order.
- The resolved identity is the parent's own reference digest, as for a same-shard parent, **never
  its serial**.
- A cross-shard parent that resolves on no shard is an `unresolved-parent` problem, named as a
  same-shard one is today.
- The store is read once, and read-only. Same-shard `Parent` behaves exactly as before.

**`main.py`**, within README §5 rule 12.
- **The `QuadSource` stage** (`build_tensor_source_work`) resolves each `(q, r)`'s `Tq` and `Tr`
  before its existence query, then runs the query keyed on them. After that it loads full Tk rows
  for the missing items only, as it does today. Use `_do_not_populate` where the values are not
  needed; `TkNumericIntegration`'s `build()` supports it.
- **The `QuadSourceIntegral` stage's `QuadSource` lookup** is keyed on its parents: by the user's
  decision (§0.2), a downstream stage finds an upstream derived row by its parents, not by labels.
  The stage already looks the Tk numeric rows up for the same modes. Move that lookup ahead of the
  `QuadSource` one, and key the `QuadSource` lookup on its results. **Nothing else in that stage
  changes here**; its own key is 09c's.
- **Keep the work lists and their literals intact.** Run `substitute_main_py()` afterwards and show
  all five substitutions still match exactly once.
- **`test_main_plumbing.EXPECTED_SITES` may grow, and only by the parent-resolution sites this
  prompt adds** (§0.4). Name each in the log, with its class and the count before and after. Any
  other movement is a stop.

**Readers (D1).** `extract_tensor_source_data.py`, the `QuadSource` lookup in
`extract_QuadSourceIntegral_data.py`, and the one in `run_quadsource_integrals.py`. Each either uses
the reader mode, refusing more than one match, or keys on the Tk rows where the reader already
holds them (*your choice* per reader; say which).

**Old stores (D4).** `QuadSource` gains **no column**: a store written before this prompt already
has both serials, which `store()` has always written. So nothing in this prompt refuses an old
store's `QuadSource` table.
- Show this with a table recreated from `schema_at_datastore-integrity-08.json`'s DDL: a keyed
  lookup finds its row.
- What does change is the fingerprint's identity. The log says that `store retire` of a store
  written before this prompt must be run at a pre-09a SHA.

**D5.**
- A new witness, `Datastore/tests/data/schema_at_datastore-integrity-09a.json`, captured with
  `schema_description.py`'s `__main__`.
- A new golden, `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-09a.json`, in the
  08 golden's pattern.
- `FINGERPRINT_FORMAT` 3 → 4, with a "Format 4" sentence in its comment.
- Each diff goes in the log, and no older file is overwritten.

**Existing tests that must change** are the orchestrator note's §0.5 list for 09a, each only by
following the key. An existing test outside it that fails is a stop.

---

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

1. **Keyed on `Tq`.** Two Tk numeric rows for `q`: a source stored from the first is found by the
   first and missed by the second.
2. **Keyed on `Tr`, across shards.** The same for `r`, with `r` on another shard.
3. **The reader mode.** It refuses on two matches, naming both pairs of serials. With one match it
   returns it without being given any parent. A lookup naming neither the parents nor the mode
   raises.
4. **The assertion.** `compute()` given a `Tq` or `Tr` other than the one the object is keyed under
   raises, naming both serials.
5. **`q == r`**, as §2 says.
6. **The inventory resolves the cross-shard parent.**
   - Two otherwise identical rows differing only in `Tr` are two records, with different
     identities.
   - A `Tr` that exists on no shard is a named `unresolved-parent` problem.
7. **An old store's `QuadSource` table is read** (§2, D4).
8. **`main.py`**, by `ast` or `load_main_py_functions`, never by import:
   - the `QuadSource` stage resolves the Tk rows before its existence query;
   - the `QuadSourceIntegral` stage's `QuadSource` lookup names them.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) drop `Tr_serial` from `build()`'s filter;
- (ii) declare `Tr` a same-shard `Parent` (the charter's mutation (v));
- (iii) let the reader mode take the first of two matches (the charter's (vi));
- (iv) remove `compute()`'s assertion.

Each goes in the log as a `git diff`, exactly as applied, with every failing test, counted.

---

## 4. Acceptance

1. K4 and K7's cross-shard form in the tree, with §3's tests. No Ray, and nothing under `var/`.
2. The parent table: `Tq` and `Tr`, their shards, whether each carries a foreign key, and how each is
   declared to the inventory.
3. The witness diff and the fingerprint re-capture in the log. No older file overwritten.
4. `substitute_main_py()` shown to match all five substitutions exactly once.
5. Every reader changed, and every script that now refuses a store it read before (rule 11), with
   every audit probe's change named.
6. The deliberate-breakage record, (i)–(iv).
7. `black --check` clean.
8. On the board: the §1 row, K4, K7 (partly: its cross-shard half), §3.3 for any issue opened, and
   a new §5 subsection. All in the same commit.

---

## 5. Stop conditions — stop and ask

- The Tk numeric rows are not `QuadSource`'s only direct inputs. Or, in a store written by the
  pipeline's own logic in a test, `Tq_serial`/`Tr_serial` do not name the rows `compute()` received.
- `main.py` cannot resolve the Tk rows before an existence query without changing what work it
  schedules, or without breaking a `substitute_main_py()` substitution.
- The inventory cannot resolve a cross-shard parent in its one read-only pass.
- An existing test outside §0.5's list for 09a would have to change.
- `EXPECTED_SITES` or a read-back guard pin moves other than as §0.4 allows.
- A fingerprint class other than `QuadSource` moves.

---

## 6. What this prompt does not do

- It does not key `GkSource` (09b) or `QuadSourceIntegral` (09c), and adds no set-valued parent to
  the inventory (09b).
- It does not change the numeric or WKB integrations' keys (07), or the policy key (08).
- It adds no foreign key across shards, and makes no reader version-aware (D4).

---

## 7. The log and the board

`logs/09a-key-quadsource-on-its-transfer-functions.md`, in README §5.1's form. In addition:
- the parent table;
- every `main.py` site changed, before and after;
- the witness diff in full, with its hunks counted by kind;
- the fingerprint re-capture, with which classes moved and why;
- the reader list;
- the `substitute_main_py()` output;
- each existing-test edit, by §0.5's list.

`IMPLEMENTATION_STATE.md`: the §1 row for 09a, K4, K7, §5, and §3.3 for anything opened;
`docs/OPEN_ISSUES.md` for anything opened.
