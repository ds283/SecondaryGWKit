# Prompt 09c — `QuadSourceIntegral` is keyed on the six rows it is computed from

**Campaign:** [`README.md`](README.md) · **Board items:** **K5**, and **K7** completed ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Charter:** [`09-key-derived-rows-on-their-parents.md`](09-key-derived-rows-on-their-parents.md),
split on 2026-09-30 into 09a → 09b → 09c ([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md)
§0.1). Its §1, §2 (K5) and §6 bind this prompt where they concern `QuadSourceIntegral`.
**Gate:** 09b landed and reviewed (board §1); `git status` is clean. D1, D4, D5 and D6 are decided,
and so are the user's four decisions of 2026-09-30 (orchestrator note §0.1–§0.4).
**Closes:** nothing on another board. With it, D1's class (audit §2.4) is closed on this board.
On this board it closes `[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]`,
**assigned here by the user on 2026-09-30** (orchestrator note §0.7, decision 3):
`QuadSourceValue._build_impl_model` refuses more than one matching `QuadSource`, naming their
`Tq_serial` and `Tr_serial`, and serves one. The issue moves to §4, and its index row is deleted.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. It keys the campaign's most-used target on six parents, four of
them on other shards. It also restructures `main.py`'s largest stage around a work list that the
sweep script filters.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §4, §5 (rules 7, 11, 12) and §6.2 **D1** in full, with the
   user's point about looking a `QuadSourceIntegral` up from `k`, `q`, `r` alone; **D4**, **D5**,
   **D6**.
2. The charter, prompt 09, in full, **K5** especially. Also the orchestrator note's §0 as the
   dispatch brief gives it, and logs 09a and 09b.
3. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §2.4–§2.6.
4. `ComputeTargets/QuadSourceIntegral.py`: the constructor, `compute()`, which receives `source`,
   `GkPolicy` and the four Tk rows and checks their modes, and `store()`.
5. `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` as 08 left it: `register`, `build`, `store`,
   `read_batch`, `inventory_records`, and `_refuse_a_store_that_predates_the_policy_key`.
6. `main.py`: `build_QuadSourceIntegral_batch`, `build_QuadSourceIntegral_payload`, and the
   construction of `qsi_work_items` and `qsi_work_batches`.
7. `extract_QuadSourceIntegral_data.py`, its `object_read_batch("QuadSourceIntegral", …)` especially.
8. `docs/handover/quadsource_atol_sweep.py`'s `substitute_main_py`. One of its five substitutions,
   `BATCH_LINE`, replaces the `qsi_work_batches` line of the stage this prompt restructures.

---

## 1. What is wanted

A `QuadSourceIntegral` is computed from six stored rows: its `QuadSource`, its
`GkSourcePolicyData`, and the `TkNumericIntegration` and `TkWKBIntegration` for each of `q` and `r`.
It records the first two (`source_serial`, `data_serial`) and looks up on neither. It records
nothing of the other four. After this prompt its lookup names all six, and `compute()` refuses
inputs that do not fit together.

09a keyed `QuadSource` on its Tk rows, and 09b keyed `GkSource` on its Gk rows (and through it
`GkSourcePolicyData`). So after this prompt every derived row's key reaches the rows it was
computed from, transitively.

---

## 2. What to change

**The parents, and their shards.** `QuadSourceIntegral` shards on `k` (`config/sharding.py`).
- `source_serial`, the `QuadSource`, is on `q`'s shard. It is an existing column with no foreign
  key.
- `data_serial`, the `GkSourcePolicyData`, is on `k`'s shard. It is an existing column with a
  foreign key.
- **Four new columns**, one each for the `Tq`/`Tr` numeric and WKB rows, on `q`'s and `r`'s shards:
  NOT NULL, indexed, with **no** foreign key, and a comment saying why (like `Tr_serial`'s).
  - Their names are *your choice*; add them to README §4.
  - A mode's shard may happen to be `k`'s. That does not change the declaration.

**K5 — the key.** All six join `build()`'s filter, `store()` writes them, and `read_batch` hands
them back to every object. `inventory_records` declares them:
- `data` as a same-shard `Parent`;
- `source` and the four Tk rows with 09a's cross-shard form.

Both are identity, so `test_store_inventory`'s `NON_IDENTITY` entries for `source_serial` and
`data_serial` move to `IDENTITY` (note §0.5).

**The asserted relations.** `compute()` or `store()` raises, naming both sides, unless:
- the `Tq` numeric row is the source's `Tq_serial`, and the `Tr` numeric row its `Tr_serial`;
- each WKB row's `numeric_serial` is its numeric row.

Today nothing checks the first pair (audit §2.4). The existing checks of modes, policy and
`z_response` stay.

**`main.py`'s `QuadSourceIntegral` stage**, within README §5 rule 12.
- Today it finds the missing integrals by physical labels and only then looks their parents up.
- After this prompt, for **every** work item it first resolves the six parents' serials. Each
  upstream row is found by its own parents (the user's decision, §0.2):
  - the `GkSource` by 09b's helper;
  - the `QuadSource` by its Tk rows (09a).
  
  Use `_do_not_populate` where values are not needed.
- Then it runs the existence query keyed on them, and then loads full parents for the missing items
  only, as today.
- Keep the binning by shard.
- **Keep the construction of `qsi_work_items` and the `qsi_work_batches` line, and its literal,
  intact.** Restructure only what follows.
- Measure the extra lookups' cost on a stand-in pool, and record it. It is not an argument against
  the change (rule 11), but the next reader should know it.
- `EXPECTED_SITES` may grow **only** by the parent-resolution sites this prompt adds (§0.4).

**The reader (D1).** `extract_QuadSourceIntegral_data.py` may keep selecting by physical labels,
with its policy. It must refuse when more than one stored row matches one `(k, q, r, z_response)`,
naming the parents that distinguish them. `read_batch` returning the parent columns is what lets it
name them.

**Old stores (D4).** A table without the four columns is refused by `build()` **and** `read_batch()`,
with 07's message. Use 08's helper pattern: match the driver's message for each new column's name.

**D5.**
- A witness `schema_at_datastore-integrity-09c.json` and a golden
  `full_store_fingerprint_datastore-integrity-09c.json`, beside the older ones.
- `FINGERPRINT_FORMAT` 5 → 6, with its comment sentence.
- Say which classes moved and why. Expect `QuadSourceIntegral` alone.

**Existing tests that must change** are the list the orchestrator measured before this dispatch
and the user granted, each only by following the key (note §0.5). 08's own modules insert
`QuadSourceIntegral` rows and are expected on it. Any other existing test that fails is a stop.

---

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

1. **Keyed on each of its six inputs.** Changing any one misses, each as its own sub-test.
2. **The asserted relations.** An integral whose `Tq` numeric row is not its source's `Tq_serial`
   refuses. So does one whose `Tq` WKB row's `numeric_serial` is not its `Tq` numeric row, and the
   same for `r`.
3. **`read_batch` hands back the six parents**, tagged and untagged.
4. **The reader refuses ambiguity**, naming the distinguishing parents. With one match it returns
   it without being given any parent.
5. **The inventory resolves all six.** Two rows differing only in one cross-shard Tk parent are two
   records with different identities.
6. **An old store is refused clearly** by both readers: a table from the 09b witness's DDL.
7. **`main.py`**, by `ast` or `load_main_py_functions`, never by import:
   - the stage resolves parents before its existence query;
   - the work-list lines are unchanged.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) drop one of the four Tk columns from `build()`'s filter (the charter's (ii));
- (ii) remove the `Tq`-numeric/source assertion (the charter's (iii));
- (iii) let the reader take the first of two matches (the charter's (vi));
- (iv) declare one Tk parent a same-shard `Parent`;
- (v) drop `source_serial` from `read_batch`'s filter.

Each goes in the log as a `git diff`, exactly as applied, with every failing test, counted.

---

## 4. Acceptance

1. K5 in the tree, with §3's tests. No Ray, and nothing under `var/`.
2. The parent table: all six parents, their shards, their foreign keys and their inventory
   declarations.
3. The witness diff and the fingerprint re-capture in the log. No older file overwritten.
4. `substitute_main_py()` shown to match all five substitutions exactly once.
5. The stage's extra lookup cost, measured.
6. Every reader changed, and every script that now refuses a store it read before (rule 11), with
   every audit probe's change named.
7. The deliberate-breakage record, (i)–(v).
8. `black --check` clean. On the board: the §1 row, K5, K7, §3.3 for anything opened, and §5.
   README §4 gains the four column names. All in the same commit.

---

## 5. Stop conditions — stop and ask

- The six rows are not `QuadSourceIntegral`'s only stored inputs. Or a stated relation fails in a
  store written by the pipeline's own logic in a test.
- The stage cannot resolve parents before its existence query without changing what work it
  schedules, or without breaking a `substitute_main_py()` substitution.
- An existing test outside the granted list would have to change.
- `EXPECTED_SITES` or a guard pin moves other than as §0.4 allows.
- A fingerprint class other than `QuadSourceIntegral` moves.

---

## 6. What this prompt does not do

- It does not key `OneLoopIntegral` on anything (6.1.4, D3).
- It does not change `QuadSource`'s (09a), `GkSource`'s (09b), the integrations' (07) or the policy's
  (08) key.
- It adds no foreign key across shards, and makes no reader version-aware (D4).

---

## 7. The log and the board

`logs/09c-key-quadsourceintegral-on-its-inputs.md`, in README §5.1's form. In addition:
- the parent table;
- every `main.py` site changed, before and after;
- the measured cost;
- the witness diff in full, with its hunks counted by kind;
- the fingerprint re-capture;
- the reader list;
- the `substitute_main_py()` output;
- each existing-test edit.

`IMPLEMENTATION_STATE.md`: the §1 row for 09c, K5, K7 (complete), §5, and §3.3 for anything
opened; README §4; `docs/OPEN_ISSUES.md` for anything opened.
