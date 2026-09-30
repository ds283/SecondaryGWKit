# Prompt 09 — every derived row is keyed on the rows it was computed from

> **Split 2026-09-30, by the user's decision, into 09a → 09b → 09c**, dispatched in one
> orchestrator session ([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md) §0.1). This file
> is their charter and is not dispatched itself. Its §1, §2 and §6 bind each part where they
> concern that part's classes; its §3 tests and mutations are distributed across them.
> - [`09a`](09a-key-quadsource-on-its-transfer-functions.md): `QuadSource` (K4), and the
>   inventory's cross-shard `Parent` (K7).
> - [`09b`](09b-key-gksource-on-its-parent-set.md): `GkSource` and its parent set (K6), and the
>   set-valued inventory parent (K7).
> - [`09c`](09c-key-quadsourceintegral-on-its-inputs.md): `QuadSourceIntegral` (K5).

**Campaign:** [`README.md`](README.md) · **Board items:** **K4**–**K7** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 08 has landed; `git status` is clean. D1, D4 and D5 are decided (README §6.2).
**Closes:** nothing on another board. The class was opened by the audit (§2.4), and D1 decided it;
record the result on this board. **Opens:** anything out of scope that you find (§6), **without
fixing it**.
**Recommended model:** **Opus**. This is the campaign's largest change and its one real design: a
`GkSource` has a **set** of parents, not one, and `main.py` today decides what to compute before it
knows which parents it would use. **If the work cannot land as one reviewable commit, stop before
writing code and propose a split** (§5); do not split it on your own.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §4, §5 (rules 7, 11, 12) and §6.2 **D1** in full,
   including *What D1 is about* and the user's point about looking a `QuadSourceIntegral` up from
   `k`, `q`, `r` alone.
2. [`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md) §2.3–§2.6.
3. The logs of prompts 07 and 08: how a parent key was carried through a factory, a target,
   `main.py`, a reader and the fingerprint, twice.
4. The factories: `QuadSource.py`, `QuadSourceIntegral.py`, `GkSource.py`, and
   `GkSourcePolicyData.py`, which is **the precedent**: it is already keyed on its parent
   `GkSource` (`source_serial` in its filter and in `inventory_records`).
5. The compute targets `ComputeTargets/QuadSource.py`, `QuadSourceIntegral.py` and `GkSource.py`:
   where each takes its inputs, and what `compute()` checks about them.
6. `main.py`: the `QuadSource` stage, the `GkSource` stage (`build_missing_GkSource`, which reads
   `GkNumericValue` / `GkWKBValue` by `read_batch` for every `z_source` of one `z_response`), and
   the `QuadSourceIntegral` stage (`build_QuadSourceIntegral_batch`, which finds the missing
   integrals **first** and looks up their sources, policy data and Tk rows **afterwards**, only for
   the missing ones).
7. `Datastore/store_inventory.py`: `Parent`, and how a key parent is resolved — only within one
   shard today.
8. `extract_QuadSourceIntegral_data.py`, `extract_GkSource_data.py` and any `extract_*.py` that looks
   up a `QuadSource`.
9. `docs/handover/quadsource_atol_sweep.py` `substitute_main_py`: the five substitutions, one of
   which filters `main.py`'s `QuadSourceIntegral` work list.

---

## 1. What is wanted

Three derived targets are looked up without saying which upstream rows they were computed from
(audit §2.4):
- `QuadSource` records `Tq_serial` and `Tr_serial` and looks up on model, `q`, `r` and tags only;
- `QuadSourceIntegral` records `source_serial` and `data_serial` and looks up on neither, and
  records nothing of the four Tk rows it consumes (`Tq`/`Tr`, numeric and WKB);
- `GkSource` records none of the Gk numeric and WKB rows it is assembled from.

So a change upstream under the same tags — a Tk or Gk tolerance, a grid, a break-point policy —
serves the old downstream row as a match. D1 decided to close this. After this prompt, a derived
row's lookup names its parents, exactly as `GkSourcePolicyData`'s already does.

**The convenience the user wants kept** (D1). A reader that holds only physical labels — `k`, `q`,
`r`, the tolerances, the tags — may still select by them, **provided it refuses when more than one
stored row matches**, naming the parents that distinguish them. In an archival store built once
there is one match, so such a reader keeps working without looking up the intermediate Tk objects.
`build()`, which decides whether to compute, keys on the parents.

---

## 2. What to change

**Which parents.** Each derived row is keyed on its **direct** inputs — the rows its `compute()`
actually receives — and nothing further up: the parents' own keys carry the rest, transitively.
Where two direct inputs are related (a `QuadSourceIntegral`'s `Tq_numeric` must be its source's
`Tq_serial`; its `Tq_WKB`'s `numeric_serial` must be its `Tq_numeric`), `compute()` or `store()`
**asserts** the relation, and a mismatch raises naming both. Today nothing checks the first
(audit §2.4).

**Same shard or not.** A parent on the derived row's shard is a column with a declared foreign key.
A parent on another shard is an indexed column with **no** foreign key and a comment saying why,
like `Tr_serial`'s after 06. Serials are unique across shards, so either identifies its row.
Establish the shard of every parent from `config/sharding.py` and put the table in the log.

**K4 — `QuadSource`.** `Tq_serial` (same shard, foreign key since 06) and `Tr_serial` (on `r`'s
shard) join `build()`'s filter. `main.py`'s `QuadSource` stage looks the two Tk numeric rows up
before its existence query. Its `q == r` case keys on the same row twice; say how that behaves.

**K5 — `QuadSourceIntegral`.** Keyed on all six direct inputs: `source_serial` (on `q`'s shard;
existing column), `data_serial` (same shard; existing column with a foreign key), and four new
columns for `Tq`/`Tr` numeric and WKB, which are on `q`'s and `r`'s shards. *Prompt's choice:* the
four columns' names; say them in the log and add them to README §4. The relations above are
asserted.

**K6 — `GkSource`, whose parents are a set.** A `GkSource` for `(k, z_response)` is assembled from
one Gk numeric row, one Gk WKB row, or both, for **each** `z_source` of the grid. A single key
column cannot name a set. **Recommended shape:**
- a table `GkSource_parents`, on the same shard: `(gk_source_serial, z_source_serial,
  numeric_serial nullable, wkb_serial nullable)`, with foreign keys, one row per `z_source`,
  written in the same transaction as the `GkSource` row. It is the record of exactly which rows were
  assembled;
- a key column `parents_digest` on `GkSource`: SHA-256 over the sorted `(z_source_serial,
  numeric_serial, wkb_serial)` tuples, in `build()`'s filter. It is how a lookup says "assembled
  from exactly this set" in one comparison;
- `store()` computes the digest from the rows it writes to `GkSource_parents`, never from anything
  else, and a test shows the two agree.

A different shape is allowed if it keys on the set and records it, and you say in the log why it is
better. Keying on the Gk rows' *settings* instead of the rows is **not** allowed: D1 is about rows.

**K6 — `main.py`'s `GkSource` stage.** It must know the parent set before it asks whether the
`GkSource` exists. Read the parent serials first, serial-only, by the same `read_batch` the stage
already uses (prompt 05's P3 and P4 made those reads filter on the parent's full key and refuse
more than one parent per `z_source`), compute the digest,
then run the existence query.

**K5 — `main.py`'s `QuadSourceIntegral` stage.** Today it finds missing integrals by physical labels
and only then looks their parents up. After this prompt it resolves each work item's parent serials
first — serial-only (`_do_not_populate`) where the values are not needed — then runs the existence
query keyed on them, then loads full parents for the missing items only, as today. Keep the
batching by shard. Measure the extra lookups' cost on a stand-in pool and record it; it is not an
argument against the change (README §5 rule 11), but the next reader should know it.

**Readers (D1).** `extract_QuadSourceIntegral_data.py`, `extract_GkSource_data.py` and any
`QuadSource` reader may keep selecting by physical labels, and must refuse on more than one match,
naming the distinguishing parents. `read_batch` gains the parent columns in what it returns, so a
reader can report them.

**K7 — the inventory across shards.** `store_inventory.py` resolves a key parent only within one
shard, so a cross-shard parent cannot be fingerprint identity. Extend `Parent` so that a declared
cross-shard parent resolves against its class's records on **every** shard (the inventory already
reads them all), and say what the resolved identity is: the parent's own physical identity, as for
a same-shard parent, never its serial. Declare every K4–K6 parent in `inventory_records`,
same-shard or cross-shard. A cross-shard parent that resolves nowhere is a named problem, like an
unresolved same-shard parent today.

**A store that predates the columns** is refused with the message 07 established (D4).

**D5.** A new schema witness, `schema_at_datastore-integrity-09.json`, and a re-captured golden
fingerprint, each beside the old with its diff in the log. Neither old file overwritten.

**`main.py` and the sweep script.** Rule 12 applies with extra force: one of `substitute_main_py`'s
five substitutions filters the `QuadSourceIntegral` work list that K5 restructures around. Keep the
work list's construction, and its literal, intact; restructure only what follows it. Show all five
substitutions still match exactly once.

---

## 3. Tests — new modules beside the existing ones, no Ray, nothing under `var/`

1. **`QuadSource` keyed on `Tq`/`Tr`.** Two Tk numeric rows for `q`; a source stored from the first
   is found by the first and missed by the second. The same for `r`, across shards.
2. **`QuadSourceIntegral` keyed on each of its six inputs**: changing any one misses.
3. **The asserted relations.** A `QuadSourceIntegral` whose `Tq_numeric` is not its source's
   `Tq_serial`, or whose `Tq_WKB`'s `numeric_serial` is not its `Tq_numeric`, refuses.
4. **`GkSource` keyed on its set.** Two parent sets differing in one `z_source`'s row give two
   `GkSource` rows; each is found by its own set. `parents_digest` equals the digest recomputed
   from `GkSource_parents`.
5. **Readers refuse ambiguity**, for each reader changed, naming the distinguishing parents; with
   one match they return it without being given any parent.
6. **The inventory resolves cross-shard parents**, and two rows differing only in a cross-shard
   parent are two records with different identities.
7. **An old store is refused clearly.**
8. **`main.py`**: the three stages resolve parents before their existence queries — an `ast` check
   or `load_main_py_functions`, never an import.

**Deliberate breakage.** Each must make a test fail; then restore it:
- (i) drop `Tr_serial` from `QuadSource`'s filter;
- (ii) drop one of the four Tk columns from `QuadSourceIntegral`'s filter;
- (iii) remove the `Tq_numeric`/source assertion;
- (iv) compute `parents_digest` from the requested set instead of the rows written;
- (v) resolve a cross-shard parent within one shard only;
- (vi) let a reader take the first of two matches.

Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. K4–K7 in the tree; the tests of §3; no Ray, nothing under `var/`.
2. The table of every parent, its shard, and whether it carries a foreign key.
3. The witness diff and fingerprint re-capture in the log; neither old file overwritten.
4. `substitute_main_py()` shown to match all five substitutions exactly once.
5. The extra lookup cost in the `QuadSourceIntegral` stage, measured.
6. Every reader changed, and every script that now refuses a store it read before (rule 11).
7. The deliberate-breakage record, (i)–(vi).
8. `black --check` clean. On the board: the §1 row, items K4–K7, §5; README §4 gains the new column
   names. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- **The work will not fit one reviewable commit.** Stop before writing code, and propose a split
  (for example `QuadSource` and `QuadSourceIntegral` first, `GkSource` and its parent set second),
  with each part's acceptance.
- A derived row's direct inputs are not what §2 says, or a stated relation between them does not
  hold in a store written by the pipeline's own logic in a test.
- `main.py` cannot resolve parents before an existence query without changing what work it
  schedules, or without breaking a `substitute_main_py()` substitution.
- The inventory cannot resolve cross-shard parents without reading the store twice, or without a
  change to its read-only contract.
- An existing test other than the fixtures, the inventory `IDENTITY` and the golden files would have
  to change.

---

## 6. What this prompt does not do

- It does not key `OneLoopIntegral` on anything (6.1.4, D3).
- It does not change the numeric or WKB integrations' keys (07), or the policy key (08).
- It does not add foreign keys across shards; it cannot.
- It does not make readers version-aware (D4).

---

## 7. The log and the board

`logs/09-key-derived-rows-on-their-parents.md`, using the template in README §5.1. In addition: the
parent table; the `GkSource` parent-set design and why; every `main.py` site changed, before and
after; the measured lookup cost; the witness diff and fingerprint re-capture; the reader list; the
`substitute_main_py()` output.

`IMPLEMENTATION_STATE.md`: the §1 row for 09, items K4–K7, and §5.
