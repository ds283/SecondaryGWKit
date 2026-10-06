# Prompt 03 — state each closed-store prefix once

**Campaign:** [`README.md`](README.md) · **Board items:** **P1**–**P2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 02 landed and was reviewed. `git status` clean. Never in one checkout while another
prompt's change to `ShardedPool.py` or `store_reader.py` is uncommitted.
**Closes:** `[04-closed-store-refusals-repeat-their-prefix]`, on `datastore-generic`'s board.
**Narrows:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. The change removes a prefix from three `raise`s, drops one
parameter, and adds a `try` in the reader. Its four callers and the one test that moves were
measured with a rough version of the change (README §0.2).

**Read first:**

1. [`README.md`](README.md): §0.2 (item 3), §1, §4, §5 (rules 3, 4, 7 and 9) and §6.1 (decision 3).
2. `prompts/datastore-generic/IMPLEMENTATION_STATE.md`, §3.1: the entry
   `[04-closed-store-refusals-repeat-their-prefix]`, with the doubled messages.
3. `Datastore/SQL/ShardedPool.py`:
   - `_read_closed_store` (`:2621-2661`);
   - its three callers here: `_plan_relocation` (`:2663-`, the call at `:2693`), `_read_back`
     (`:2799-2808`, called inside `_relocate_store`, whose `_failure_message` wraps it) and
     `_plan_deletion` (`:2901-`, the call at `:2944-2946`);
   - the docstrings of `closed_store_files` and `delete_store` (`:2547-2620`), which describe the
     refusals.
4. `Datastore/store_reader.py`: `_VERB` and `_refuse` (`:97-100`), and `open_read_only`'s call of
   `_read_closed_store` (`:206`), the fourth caller, which does not wrap today.
5. `RunRegistry/stores.py:1567-1568`, `_SHARDS_TABLE_UNREADABLE`, which matches the phrase "its
   shards table could not be read" in these messages. The phrase must survive.
6. The tests:
   - `Datastore/tests/test_store_reader.py`,
     `TestRefusals.test_missing_shard_is_refused_by_read_closed_store` (`:304-312`), the one that
     moves;
   - `Datastore/tests/test_absolute_shard_record_refused.py`, which asserts fragments, not the
     prefix, and must not move;
   - `RunRegistry/tests/test_store_copy_move.py:328` and `RunRegistry/tests/test_store_retire.py:790`,
     which must not move.
7. The orchestrator's measurement: the `ShardedPool.py` `_read_closed_store` hunks, the
   `store_reader.py` hunk and the `test_store_reader.py` hunk of
   [`measure/rough_change.diff`](measure/rough_change.diff).

Line numbers are at `6de692a`. 01 edits `ShardedPool.py`'s constructor, so they will have moved:
find each by name.

---

## 1. What is wanted

A refusal of a closed store's operation states its prefix twice:
`Cannot delete sharded datastore "<p>": Cannot delete sharded datastore "<p>": Shard #0 …`.
`_read_closed_store` prefixes its message with `Cannot {verb} sharded datastore "<p>": `, and
`_plan_deletion` and `_plan_relocation` prefix it again.

The user decided on 2026-10-06 that the prefix goes **from `_read_closed_store`** (README §6.1).

Measured at `6de692a` (README §0.2): `_read_closed_store` has **four** callers. Three wrap its
message. The fourth, `store_reader.open_read_only`, relies on the inner prefix, which is why the
reader's message is not doubled. So:

- **P1.** `_read_closed_store` raises the bare reason, and takes no `verb`.
- **P2.** `open_read_only` wraps that reason with its own `_refuse`, as it wraps its other
  refusals. Its message then ends with the reader's ". Nothing was read or repaired", which this
  refusal lacks today.

**What does not change.** Which stores are refused, and why. The reason text after the prefix.
The phrase "its shards table could not be read". What the code writes.

---

## 2. What to change

**P1 — `ShardedPool._read_closed_store(primary, *, missing_ok=False)`.**
- The three `raise`s give only the reason:
  - `its shards table could not be read (<e>)`;
  - the resolver's message, unchanged;
  - the joined file problems, unchanged.
- The `verb` parameter is removed. Its three callers in `ShardedPool.py` stop passing it, and
  each keeps its own wrapping exactly as it is.
- The docstring says that it raises `RuntimeError` naming the problem, and that each caller states
  the operation.

**P2 — `store_reader.open_read_only`.**
- The call becomes `try: … except RuntimeError as e: raise _refuse(primary_path, str(e)) from e`,
  as the reader's other refusals are raised.
- The docstring's list of refusals does not change.

**Tests.**
- **Changed, named here** (README §5 rule 7):
  `test_store_reader.TestRefusals.test_missing_shard_is_refused_by_read_closed_store`.
  - It calls `ShardedPool._read_closed_store(self.store.primary)`, without `"read"`.
  - It asserts that the reader's message contains that call's message, and holds
    `Cannot read sharded datastore` exactly once.
  - Its comment says that the refusal is `_read_closed_store`'s, under the reader's one prefix.
- **New module `Datastore/tests/test_closed_store_refusals.py`, 5 tests.** No Ray, nothing under
  `var/`. Use the absolute-record fixture of `test_absolute_shard_record_refused.py` (a shard
  record `../outside.sqlite`) and a dropped `shards` table.

  In tests 1–4, "once" means that the message begins with the operation's prefix and holds the
  words `sharded datastore` exactly once. No reason text holds them, so a second prefix of any
  verb is caught.
  1. `test_delete_and_closed_store_files_state_their_prefix_once`: `Cannot delete sharded
     datastore` once, and the resolver's reason;
  2. `test_copy_and_move_state_their_prefix_once`: `Cannot copy …` and `Cannot move …` once, with
     the destination named;
  3. `test_the_reader_states_its_prefix_once`: `Cannot read sharded datastore` once, and the
     message ends `Nothing was read or repaired`;
  4. `test_an_unreadable_shards_table_is_named_once_by_each`: with the `shards` table dropped,
     `delete_store`, `copy_store` and `open_read_only` each state their prefix once, and each
     message holds "its shards table could not be read";
  5. `test_read_closed_store_states_no_prefix`: `_read_closed_store`'s own message, for both
     fixtures, does not begin `Cannot`.

---

## 3. Verification

1. **Every suite**, with counts before and after, each difference accounted for by test id:
   - Datastore: 02's count + 5. At 02's expected 734, that is **739** (`[a-f]` 158 + 5 = 163,
     `[g-o]` 185, `[p-r]` 200, `[s-z]` 191);
   - RunRegistry **280**, unchanged: its two tests that match these messages must pass unedited;
   - every other suite unchanged.
2. **The messages, before and after.** For the absolute record and for the dropped `shards`
   table, give each of `delete_store`, `closed_store_files`, `copy_store`, `move_store` and
   `open_read_only`'s message at the parent and at this commit, with temporary paths normalised.
   Each differs only by the removed repetition, and the reader's by the added suffix.
3. `git grep -n '_read_closed_store(' -- '*.py'` shows no call passing a verb.
4. `test_layer_is_generic` passes. `black --check` is clean on every touched Python file.
5. **Deliberate breakage.** Each must make the named tests fail; then restore it.
   - (i) the resolver branch's `raise` prefixed again, with the literal
     `Cannot open sharded datastore "<primary>": `: tests 1, 2, 3 and 5;
   - (ii) the same prefix in the shards-table branch: tests 4 and 5;
   - (iii) the reader's `try` removed, so that its message has no prefix: test 3, and the
     changed `test_store_reader` test;
   - (iv) `_plan_deletion` stops wrapping: test 1.

   Record each in the log as a diff, exactly as applied, with what failed. Check each with
   `git apply --check` and `-R --check`, in `bash`, and leave `git status` clean.

---

## 4. Acceptance

1. **P1–P2** as §2 says, with the five tests and the one named change.
2. Every suite has the counts of §3.1.
3. **The records**, in the same commit:
   - On `datastore-generic`'s board, `[04-closed-store-refusals-repeat-their-prefix]` moves from
     §3.1 to §4 with a dated "Closed" line, naming this campaign and the fourth caller.
   - `docs/OPEN_ISSUES.md`: its row leaves §1.19, the count goes down by one, and the date is
     corrected. With it the last of `datastore-generic`'s issues is closed, and §1.18 says so.
   - This board: the §1 row, P1–P2, §5, and the header **CLOSED — 3 of 3 prompts landed**. The
     README header says so too.
   - `prompts/INDEX.md`: this campaign's line says it is closed.

---

## 5. Stop conditions — stop and ask the user

- A caller of `_read_closed_store` exists beyond the four of §1.
- A test outside §2's plan fails, RunRegistry's included.
- A refusal's reason, after the prefix, changes in any way but the removed repetition.
- The change would open, write or touch any store under `var/`, or start Ray.

---

## 6. What this prompt does not do

- **Files it changes:** `Datastore/SQL/ShardedPool.py` (`_read_closed_store` and its three
  calls); `Datastore/store_reader.py` (the one call); `Datastore/tests/test_store_reader.py` (the
  one test); new `Datastore/tests/test_closed_store_refusals.py`; the log, this board, the README
  header, `prompts/INDEX.md`, `prompts/datastore-generic/IMPLEMENTATION_STATE.md` and
  `docs/OPEN_ISSUES.md`. Nothing else.
- It does not change `_failure_message`, the resolver or any other refusal's text.

---

## 7. The log and the board

`logs/03-state-each-closed-store-prefix-once.md`, using README §5.1. In addition, it gives §3.2's
messages before and after, and each breakage as a diff with what failed.

`IMPLEMENTATION_STATE.md`: the §1 row for 03, items P1–P2, §5, and the header.
