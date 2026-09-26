# Prompt 07 — a WKB row is keyed on the numeric row it starts from

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**–**K2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompts 05 and 06 have landed; `git status` is clean. D1, D4 and D5 are decided (README
§6.2). 05 must precede this prompt because both edit the WKB factories' read paths, and 06 because
it sets up the D5 witness procedure this prompt follows.
**Closes:** `[20-wkb-rows-consume-numeric-initial-data]`, on the GkTk-remedial board's §4, with its
index row deleted. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. The schema edit is small; the judgement is in `main.py`, where
every place that asks "does this WKB row exist?" must now say which numeric row it means, and in
the pure-WKB $G_k$ rows, which have no numeric parent at all.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §4, §5 (rules 7, 11 and 12 especially) and §6.2 **D1**,
   **D4** and **D5**.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §2.3, §2.4, §2.6
   and §3.13.
3. The GkTk-remedial board's entry for `[20-wkb-rows-consume-numeric-initial-data]`, with its
   **Assigned** line.
4. `Datastore/SQL/ObjectFactories/TkWKBIntegration.py` and `GkWKBIntegration.py`: `register`,
   `build`, `store`, `inventory_records`, and the value factories' read paths as prompt 05 left
   them.
5. `ComputeTargets/TkWKBIntegration.py` and `GkWKBIntegration.py`: the constructors, and where
   `z_init`, `T_init`/`G_init` and their derivatives come from.
6. `main.py`: the $T_k$ WKB stage (the existence query, the numeric lookup, the construction from
   the numeric stop point, around `:1416-1552`), the $G_k$ WKB stage (around `:2025-2268`,
   including the **pure-WKB** branch with `G_init = 0`, `Gprime_init = 1`), and the stage lookup
   at `:3344-3348`. Line numbers are from the audit's tree; find each by content.
7. `Datastore/SQL/Datastore.py` `_drop_order` and the drop actions.
8. `extract_TkWKB_data.py` and `extract_GkWKB_data.py`, their WKB lookups.
9. `ComputeTargets/tests/test_numeric_break_point_key.py` (a precedent: a key column added, with a
   refusal for a store that predates it) and `ComputeTargets/tests/test_gauss_order_key.py`.
10. [`logs/06-correct-the-foreign-keys.md`](logs/06-correct-the-foreign-keys.md): the D5 procedure
    as 06 set it up, and the witness constant.

---

## 1. What is wanted

A WKB integration starts from the numeric integration's stop point: `z_init`, the value and the
derivative there. The WKB row records none of that numeric row's identity, and its lookup filters
`z_init` only optionally, by an absolute `1e-7` against $z \sim 10^{9}$–$10^{12}$ — which is exact
equality in practice. `main.py`'s existence queries pass no `z_init` at all. So once any WKB row
exists for `k`, it is served whatever numeric row exists now (audit §2.3). That is exactly the
class D1 decided to close.

After this prompt, every WKB row names the numeric row it was seeded from, and every lookup of a
WKB row says which one it means.

---

## 2. What to change

**K1 — the key.**
- A column `numeric_serial` (README §4) on `TkWKBIntegration` and `GkWKBIntegration`, indexed, with
  a foreign key to `TkNumericIntegration.serial` and `GkNumericIntegration.serial` respectively.
  Numeric and WKB rows both shard on `k`, so the parent is always on the same shard; confirm it
  from `config/sharding.py` and say so in the log.
  - **Tk: NOT NULL.** `main.py` raises when a Tk WKB row has no numeric row to start from.
  - **Gk: nullable, and NULL means pure WKB** — the branch that starts at `z_source` with
    `G_init = 0`, `Gprime_init = 1`. Confirm there is no other way a Gk WKB row can lack a numeric
    parent; if there is, stop (§5).
- The compute targets carry the numeric parent (its `store_id`, or the object, *prompt's choice*)
  from construction to `store()`. A Tk WKB target constructed without one is an error. A Gk WKB
  target constructed without one is a pure-WKB target, and says so.
- `build()` filters `numeric_serial == parent` when a parent is given, and `numeric_serial IS NULL`
  when it is not (Gk only). A Tk lookup without a parent is an error.
- **`z_init` and the initial values leave the key.** Once the parent is in the key they are
  functions of it. They stay stored, as payload. The fragile float match on `z_init` (audit §3.13)
  goes with them.
- **A consistency check at store time.** Before a WKB row is stored, its `z_init` and initial
  values must equal the numeric parent's stop values that `main.py` derived them from, bit for bit.
  *Prompt's choice* where the check lives (compute target or factory), provided it cannot be
  skipped by a caller. A mismatch raises and names both.
- `inventory_records` declares `numeric_serial` a key parent (`Parent("numeric_serial", …)`), since
  it is on the same shard.

**K1 — `main.py`**, within README §5 rule 12:
- every WKB **existence query** passes the numeric parent (Tk) or says pure WKB (Gk), which means
  the numeric row is looked up **before** the existence query, not after. Keep that lookup
  serial-only (`_do_not_populate`) where the values are not needed;
- every WKB **construction** passes the parent;
- the stage lookup at the QSI stage passes the parent it will be consumed with.

After your edits, run `substitute_main_py()` from `docs/handover/quadsource_atol_sweep.py` and show
all five substitutions still match exactly once (rule 12). Never edit the sweep script.

**K1 — readers.** `extract_TkWKB_data.py` and `extract_GkWKB_data.py` look WKB rows up by physical
labels. Under D1 they may keep doing that, provided a lookup that matches **more than one** stored
row refuses, naming the `numeric_serial` values that distinguish them, rather than choosing one
(README §6.2 D1). Say in the log which scripts you changed and how.

**K1 — a store that predates the column.** Follow `TkNumericIntegration`'s `break_point_kind`
precedent: a lookup against a store without `numeric_serial` raises with a message saying the store
predates this key and is readable at the SHA its sidecar records. That is a refusal, not a
version-aware reader (D4); it does not try to read the old row.

**K2 — `_drop_order`.** A WKB row now references a numeric row, so `tk-wkb` must drop before
`tk-numeric`, and `gk-wkb` before `gk-numeric`. Correct `_drop_order` and its comment.

**D5.** Capture a new schema witness, `schema_at_datastore-integrity-07.json`, repoint the constant,
and put the witness diff in the log, in full. The golden fingerprint **will** move: `numeric_serial`
is new key identity. Re-capture it the same way, as a new file beside the old, repoint its test,
and put in the log which records' identities changed and why. Never overwrite either old file.

---

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

At minimum:

1. **Keyed on the parent.** Two Tk numeric rows for the same `k` (different tolerances), a WKB row
   stored from the first: a lookup naming the first finds it; naming the second misses.
2. **Pure WKB is its own key.** For one `(k, z_source)`, a numeric-seeded Gk WKB row and a pure-WKB
   row coexist; each lookup finds its own; neither raises.
3. **A Tk lookup without a parent is an error**, and so is a Tk construction without one.
4. **The consistency check.** A WKB target whose `z_init` or initial value differs from its
   parent's stop value by one ulp refuses to store.
5. **Readers refuse ambiguity.** With two WKB rows for the same physical labels and different
   parents, the extraction path's lookup refuses and names both `numeric_serial` values.
6. **An old store is refused clearly**: a table without `numeric_serial` gives the D4 message.
7. **Drop order.** Dropping `tk-numeric` with WKB rows present, under enforcement, succeeds only in
   the corrected order.
8. **`main.py`.** The existence and construction sites pass the parent: use
   `ComputeTargets/tests/test_main_plumbing.load_main_py_functions` or an `ast` check in the style
   of `test_main_plumbing.py`, never an import of `main.py`.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) drop `numeric_serial` from `build()`'s filter;
- (ii) match a missing Gk parent as "any parent" instead of `IS NULL`;
- (iii) remove the consistency check;
- (iv) put `z_init` back in the key with the absolute `1e-7`, and remove the parent from the
  existence query in `main.py`;
- (v) revert `_drop_order`.

Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. K1–K2 in the tree; the tests of §3 in new modules; no Ray, nothing under `var/`.
2. The witness diff, and the fingerprint re-capture with its explanation, in the log; neither old
   file overwritten.
3. `substitute_main_py()` shown to match all five substitutions exactly once.
4. The list of readers that changed, and of any that now refuse a store they read before (rule 11).
5. The deliberate-breakage record, (i)–(v).
6. `[20-wkb-rows-consume-numeric-initial-data]` moved to the GkTk-remedial board's §4 with a closing
   note naming this commit; its index row deleted; the count and date corrected.
7. `black --check` clean. On the board: the §1 row, items K1–K2, §3.1, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- A Gk WKB row can lack a numeric parent other than on the pure-WKB branch.
- A numeric parent is not guaranteed to share its WKB row's shard.
- `main.py` cannot pass the parent to an existence query without restructuring a stage beyond
  looking the numeric row up first.
- A `substitute_main_py()` substitution stops matching exactly once.
- The consistency check fails on a store written by the unmodified pipeline logic in a test: that
  would mean `main.py` does not seed WKB rows from the stop values the way the audit read it.
- An existing test other than those this prompt names would have to change.

---

## 6. What this prompt does not do

- It does not key `GkSource`, `QuadSource` or `QuadSourceIntegral` on their parents. That is 09.
- It does not change the numeric integrations' own keys, or how the stop point is found
  (`[11-stop-point-root-tolerance]` is the hand-over campaign's).
- It does not read any old store; D4.

---

## 7. The log and the board

`logs/07-key-wkb-rows-on-their-numeric-row.md`, using the template in README §5.1. In addition: the
shard evidence; every `main.py` site changed, before and after; the witness diff; the fingerprint
re-capture with the changed identities; the reader list; the `substitute_main_py()` output.

`IMPLEMENTATION_STATE.md`: the §1 row for 07, items K1–K2, §3.1 and §5.
