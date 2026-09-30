# Prompt 09b — `GkSource` is keyed on the set of Gk rows it is assembled from

**Campaign:** [`README.md`](README.md) · **Board items:** **K6**, and **K7**'s set-valued parent ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Charter:** [`09-key-derived-rows-on-their-parents.md`](09-key-derived-rows-on-their-parents.md),
split on 2026-09-30 into 09a → 09b → 09c ([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md)
§0.1). Its §1, §2 (K6) and §6 bind this prompt where they concern `GkSource`.
**Gate:** 09a landed and reviewed (board §1); `git status` is clean. D1, D4 and D5 are decided, and
so are the user's four decisions of 2026-09-30 (orchestrator note §0.1–§0.4).
**Closes:** `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` (board §3.3), **assigned here by the
user on 2026-09-30** (§0.3); its index row is deleted. **Opens:** anything out of scope that you find
(§6), **without fixing it**.
**Also, by the user's decisions on the sketch (2026-09-30, orchestrator note §0.7):**
- `tools/inventory_report.py` renders the parent set by its physical labels;
- `GkSourceValue._build_impl_model` refuses more than one matching `GkSource`, naming what
  distinguishes them;
- an empty parent set is refused by `store()`, and named as a problem by the inventory.
**Recommended model:** **Opus**. This is the campaign's one real design. A `GkSource` has a *set* of
parents, and `main.py` decides whether a `GkSource` exists before it reads the rows it would be
assembled from. Six lookups in `main.py` find a `GkSource`, not the one the charter names.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §4, §5 (rules 7, 11, 12) and §6.2 **D1** in full; **D4**,
   **D5**.
2. The charter, prompt 09, in full, **K6** especially. Also the orchestrator note's §0 as the
   dispatch brief gives it, and log 09a's "State handed to the next prompt".
3. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §2.4 and §2.6.
4. `ComputeTargets/GkSource.py`: `assemble_GkSource_values`, which receives a `numeric_data` and a
   `WKB_data` map keyed by `z_source`, and the constructor, `compute()` and `store()`.
5. `Datastore/SQL/ObjectFactories/GkSource.py`: `register`, `build`, `store`, `inventory_records`.
   `GkSourcePolicyData.py`, keyed on its `GkSource`, is the precedent.
6. `GkNumericValue.read_batch` (`GkNumericIntegration.py`) and `GkWKBValue.read_batch`
   (`GkWKBIntegration.py`). Each already selects its parent's serial, to refuse a second parent for
   one `z_source` (prompt 05c), but does not hand it back. Also the two WKB value factories'
   `_build_impl_model`, and `TkWKBValue`'s, for the 07 issue.
7. `main.py`: `build_GkSource_batch` and `build_missing_GkSource`; `build_incomplete_GkSourcePolicy`,
   `build_GkSourcePolicy_batch` and `build_missing_GkSourcePolicy`; the `GkSource` lookup in
   `build_QuadSourceIntegral_batch`.
8. `Datastore/store_inventory.py` as 09a left it, and `Datastore/SQL/Datastore.py`'s `_drop_actions`
   and `_drop_order`.
9. The `GkSource` readers: `extract_GkSource_data.py`, `extract_QuadSourceIntegral_data.py` (its
   `GkSource` lookup), and in `docs/source-remediation-verification/`, `analyse_greens_and_source.py`
   and `run_quadsource_integrals.py` (their `GkSource` lookups).

---

## 1. What is wanted

A `GkSource` for `(k, z_response)` is assembled, for **each** `z_source` of its grid, from the value
at `z_response` of one Gk numeric row, one Gk WKB row, or both. Its lookup names none of them. So a
change to the Gk rows under the same tags serves the old `GkSource` as a match (audit §2.4). After
this prompt the lookup names the set, and the store records it.

Everything downstream of a `GkSource` sees the change: `GkSourcePolicyData` is keyed on its
`GkSource`, and `QuadSourceIntegral` on its `GkSourcePolicyData` (from 09c).

---

## 2. What to change

**Which parents.** The Gk rows whose values `assemble_GkSource_values` receives. They are all on
`k`'s shard, as `GkSource` is, so each can carry a foreign key.
- A value row is identified by its integration row and its `z`, and `GkSource` already keys on
  `z_response`. So naming the **integration** rows is recommended.
- Name the value rows instead only if you say why.

**K6 — the shape.** The charter's recommendation, restated:
- a table **`GkSource_parents`** on the same shard: `(gk_source_serial, z_source_serial,
  numeric_serial nullable, wkb_serial nullable)`, each with its foreign key. It has one row per
  `z_source`, written in the same transaction as the `GkSource` row. It is the record of exactly
  what was assembled;
- a key column **`parents_digest`** on `GkSource`: SHA-256 over the sorted
  `(z_source_serial, numeric_serial, wkb_serial)` tuples, in `build()`'s filter. It lets a lookup
  say "assembled from exactly this set" in one comparison;
- `store()` computes the digest from the rows it writes to `GkSource_parents`, never from anything
  else, and a test shows the two agree.

A different shape is allowed if it keys on the set and records it, and the log says why it is
better. Keying on the Gk rows' *settings* instead of the rows is **not** allowed: D1 is about rows.

**The value reads hand back their parent.** `GkNumericValue.read_batch` and `GkWKBValue.read_batch`
select the parent serial and drop it. Attach it to each object they return, so a caller can build
the set.
- The charter asks for the pre-existence read to be serial-only. Neither read has such a form
  today. A serial-only form, returning `(z_source, parent)` pairs and no values, is *your choice* of
  shape.
- Measure what the pre-existence read costs, in either form, on a stand-in pool, and record it.
  The cost is not an argument against the change (README §5 rule 11), but the next reader should
  know it.

**`[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`** (§0.3). The four WKB value reads,
`GkWKBValue` and `TkWKBValue` in both `_build_impl_model` and `read_batch`, refuse a store that
predates `numeric_serial` with 07's D4 message, whether or not a parent is given. Today a
parentless read succeeds there, and a keyed one raises a raw `OperationalError`.
- The issue's "Next step" names one way: always select the column, and match the driver's message
  as `build()` does.
- The issue closes on the board's §4, with its index row deleted.

**`main.py`**, within README §5 rule 12. **Six** lookups find a `GkSource` by physical labels:
- the GkSource stage's existence query and its construction;
- three in the `GkSourcePolicyData` stage;
- one in the `QuadSourceIntegral` stage.

The orchestrator note's §0.6 lists them. Every one is keyed on the parent set:
- **The GkSource stage** reads the parent set of every `z_response` in its batch **before** its
  existence query, computes the digest, and then runs the query keyed on it. It then assembles the
  missing ones from the values, as today. Say whether the values are read once or twice, and why.
- **The downstream stages** find the `GkSource` by its parents too (the user's decision, §0.2),
  resolving the same set the GkSource stage would. A helper shared by all three stages, returning
  the set and digest for a `(k, z_response)`, is recommended, so that they cannot compute different
  digests.
- `EXPECTED_SITES` may grow **only** by the parent-resolution sites this prompt adds (§0.4). Name
  each, with the count before and after.
- Run `substitute_main_py()`; all five must still match exactly once.

**Readers (D1).** The four named in "Read first" item 9 use a reader mode that selects by physical
labels, as 09a's `QuadSource` mode does. It refuses when more than one stored `GkSource` matches,
naming what distinguishes them: their digests, and the `z_source`s at which their parents differ.
With one match it returns it.

**K7 — the set-valued parent** (`Datastore/store_inventory.py`).
- A `GkSource`'s fingerprint identity includes its parent set, **resolved to the parents' own
  identities** through `GkSource_parents`. That is, for each `z_source`, the reference digests of its
  numeric and WKB rows, or `None`.
- **`parents_digest` is not that identity.** It is computed from serials, and the inventory never
  uses a serial as identity.
- A `GkSource_parents` row naming a row that cannot be resolved is a named problem. So are
  `GkSource_parents` rows with no `GkSource`, like an orphan value row.
- The inventory reads the table as it reads a class's tag and value tables: once, read-only.

**`_drop_actions`.** `GkSource_parents` joins the `gk-source` action.
`TestDropOrder.test_every_declared_foreign_key_between_actions_is_respected` (07) must pass
unmodified. The order *within* an action is `[07-drop-actions-list-a-parent-table-before-its-children]`,
which this prompt does not fix.

**Old stores (D4).** A `GkSource` table without `parents_digest` is refused by `build()`, and by
anything else that filters on it, with 07's message: the store predates the key and is readable at
the SHA its sidecar records.

**D5.**
- A witness `schema_at_datastore-integrity-09b.json` and a golden
  `full_store_fingerprint_datastore-integrity-09b.json`, each beside the older ones.
- `FINGERPRINT_FORMAT` 4 → 5, with its comment sentence.
- Say which classes moved and why. Expect `GkSource`, and `GkSourcePolicyData` through its source.

**Existing tests that must change** are the list the orchestrator measured before this dispatch
and the user granted, each only by following the key (note §0.5). Any other existing test that
fails is a stop.

---

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

1. **Keyed on its set.** Two parent sets differing in one `z_source`'s row give two `GkSource` rows.
   Each is found by its own set.
2. **The record agrees with the key.** `parents_digest` equals the digest recomputed from
   `GkSource_parents`.
3. **The value reads hand back their parent**, in whatever form you ship.
4. **Readers refuse ambiguity**, each naming what distinguishes the matches. With one match they
   return it without being given the set.
5. **The inventory resolves the set.** Two `GkSource`s differing only in one parent are two records
   with different identities. An unresolvable member is a named problem.
6. **An old store is refused clearly**: a `GkSource` table from the 09a witness's DDL.
7. **The 07 issue.** On a WKB table from the 06 witness's DDL, all four WKB value reads refuse with
   the D4 message, keyed or not.
8. **`main.py`**, by `ast` or `load_main_py_functions`, never by import:
   - the GkSource stage resolves the set before its existence query;
   - the other five lookups are keyed on a set.
9. **Foreign keys.** `GkSource_parents`'s keys, checked under `PRAGMA foreign_key_check` on rows of
   the test's own. Each serial must be one that the wrong table cannot satisfy by coincidence (06's
   lesson), and the docstring says why.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) compute `parents_digest` from the requested set, not the rows written (the charter's (iv));
- (ii) drop `parents_digest` from `build()`'s filter;
- (iii) let a reader mode take the first of two matches (the charter's (vi));
- (iv) leave the parent set out of `GkSource`'s inventory identity;
- (v) let a parentless WKB value read succeed on a pre-07 table.

Each goes in the log as a `git diff`, exactly as applied, with every failing test, counted.

---

## 4. Acceptance

1. K6 and K7's set-valued form in the tree, with §3's tests. No Ray, and nothing under `var/`.
2. The parent table: every parent of a `GkSource`, its shard and its foreign key. Also the design
   and why.
3. The witness diff and the fingerprint re-capture in the log. No older file overwritten.
4. `substitute_main_py()` shown to match all five substitutions exactly once.
5. The pre-existence read's cost, measured.
6. Every reader changed, and every script that now refuses a store it read before (rule 11), with
   every audit probe's change named.
7. The deliberate-breakage record, (i)–(v).
8. The 07 issue moved to the board's §4, its index row deleted, and the count and date corrected by
   counting.
9. `black --check` clean. On the board: the §1 row, K6, K7, §3.3 and §5. All in the same commit.

---

## 5. Stop conditions — stop and ask

- The Gk rows are not `GkSource`'s only direct inputs. Or, in a store written by the pipeline's own
  logic in a test, the set recorded is not the set assembled.
- `main.py` cannot resolve the set before an existence query without changing what work it
  schedules, or without breaking a `substitute_main_py()` substitution.
- The inventory cannot resolve the set in its one read-only pass.
- An existing test outside the granted list would have to change.
- `EXPECTED_SITES` or a guard pin moves other than as §0.4 allows.
- A fingerprint class other than `GkSource` and `GkSourcePolicyData` moves.

---

## 6. What this prompt does not do

- It does not key `QuadSourceIntegral` (09c), and does not change `GkSourcePolicyData`'s key.
- It does not change the Gk integrations' keys (07), or fix the drop order within an action.
- It makes no reader version-aware (D4).

---

## 7. The log and the board

`logs/09b-key-gksource-on-its-parent-set.md`, in README §5.1's form. In addition:
- the parent table, and the design and why;
- every `main.py` site changed, before and after;
- the measured cost;
- the witness diff in full, with its hunks counted by kind;
- the fingerprint re-capture;
- the reader list;
- the `substitute_main_py()` output;
- each existing-test edit.

`IMPLEMENTATION_STATE.md`: the §1 row for 09b, K6, K7, §3.3 (the 07 issue moved to §4), and §5;
`docs/OPEN_ISSUES.md`.
