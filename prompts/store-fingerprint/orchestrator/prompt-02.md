# Orchestrator — prompt 02, the structured inventory

Read [`../README.md`](../README.md) first. **You do not write code.** You may *replay* a mutation the
log records, with `git apply`, and then revert it.

**The prompt:** [`02-a-structured-inventory.md`](../02-a-structured-inventory.md)
**Board items:** F4–F7

## 0. What makes this prompt unusual

A structured inventory is easy to write. The review is about whether it says **what the store holds
independently of how it was built**, and five things decide that.

- **No serial in any key.** Test 1 builds the same content with serials permuted and shards
  reassigned, and requires the inventories to be equal. Read that test: it must permute the
  **replicated** serials too, not only the sharded ones, or references to parents by serial would
  still pass.
- **Every identity column is in the key.** Test 2 varies each identity column in turn. Check its
  list against each factory's `build()` lookup yourself, for at least `GkSource`,
  `GkWKBIntegration` and `QuadSourceIntegral`, including the optional filters.
- **Tags.** A tagged parent's digest covers its tags, and `OneLoopIntegral`'s tags come from its own
  association table.
- **Replicated classes are read from every shard**, and a divergence is a named problem, not a
  silent pick.
- **Nothing consumer-facing changes.** The old `inventory()` methods, `ShardedPool.inventory`,
  `inventory_config`, the report and `extract_common` are untouched until prompt 03.

## 1. Before you dispatch

1. **D1 was decided on 2026-09-24** (README §6.2): `float.hex`, which is what the prompt is written
   against. Confirm that the prompt's §2 "Floats" still says so.
2. **Confirm the branch and `HEAD`**, and record the SHA. `HEAD` must be prompt 01's commit, or
   the commit recording its review. Tell the agent the tree is not its own.
3. `python -m RunRegistry list`: nothing `running`.
4. **Baseline every suite.** Re-measure.
5. **Snapshot, read-only,** the three stores and their sidecars, and `var/runs/`, as in prompt 01's
   orchestrator §1.4, under an `orch_` prefix.
6. **Free disk space:** at least 1 GB.
7. `git status` is clean, and `var/` holds nothing new.

## 2. Dispatch

One fresh-context subagent. Give it:
- the prompt, the campaign README and the audit;
- prompt 01 and its log;
- the SHA and the baselines.

Nothing else. Tell it plainly:
- **one commit.** The log goes at `logs/02-a-structured-inventory.md`;
- update the board and `docs/OPEN_ISSUES.md` in the same commit;
- run `black`;
- **never point new code at an original store or the backup.** Work only on a copy under
  `var/store-fingerprint-check-02/`, and delete it afterwards;
- mutations are recorded as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files;
- the prompt's §7 stop conditions mean *stop and ask*. **A duplicate key or a replicated divergence
  in the real copy is a stop**, to be reported and not resolved.

## 3. The review — eleven checks

1. **The key table.** The log has one row per class. Compare it with audit §4, and with each
   factory's `build()` lookup for `GkSource`, `GkWKBIntegration`, `QuadSourceIntegral`,
   `TkNumericIntegration` and `wavenumber_exit_time`. Every filtered column is in the key, and
   nothing store-local is. Every deviation from the audit cites a lookup line.
2. **Physical keys.** Test 1 permutes the replicated serials, the sharded serials and the shard
   assignment, and requires the inventories to be equal.
3. **One `canonical`, and one canonical JSON.** Nothing else formats a float for a key. Check this
   with `git grep -n 'float.hex\|\.hex()\|round(' -- Datastore/`. The digest of a tagged parent
   covers its tags.
4. **Tags.** Tags are read from the association tables, joined within a shard, and sorted.
   `OneLoopIntegral`'s come from `OneLoopIntegral_tags`.
5. **Replicated classes** are read from every shard, and a divergence is a named problem naming the
   shard.
6. **No `*Value` row is held.** Value counts come from `GROUP BY`. Check the code path, and the
   log's peak memory.
7. **Nothing consumer-facing changed.**
   `git diff HEAD~1 HEAD -- tools/ main.py extract_common.py config/ Datastore/SQL/ShardedPool.py RunRegistry/`
   is empty. In `git diff HEAD~1 HEAD -- Datastore/SQL/ObjectFactories/`, each factory only
   **gains** a method, and no existing method changes.
8. **Mutations reproduce.** Replay (i) (a model referenced by serial), (iv) (one shard only) and
   (vii) (a tagged parent's digest without its tags) from the log. Run the named tests, confirm
   that they fail, and revert. `git status` is clean afterwards.
9. **No Ray, no `var/`, in the tests.** Read-only across `read_inventory`: prompt 01's
   never-writes check is exercised on the full inventory.
10. **The demonstration discriminates.**
    - Per-class record counts equal the row counts.
    - The value-count sums equal the value tables' counts.
    - The one deleted `QuadSourceIntegral` row appears as exactly one missing record, named by
      physical labels, with every other class unchanged.

    Re-derive the counts for two classes from your own snapshot of the original sweep store.
11. **The originals are untouched, and the suites and the board are right.**
    - Re-take §1.5, and confirm that `var/store-fingerprint-check-02/` is gone.
    - Every suite matches its baseline, and `Datastore/tests` rises by exactly the tests added.
    - `black --check` is clean.
    - The board and the index are updated.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of the prompt's §7.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes changing a lookup, a key column, a schema, a consumer, or an existing
  `inventory()`.

## 5. After it lands

Report:
- the commit;
- the key table as shipped, and every *prompt's choice* the agent changed;
- the demonstration's numbers: the per-class counts, the tag sets, the deleted record, and the
  size, time and memory;
- any problems found in the real copy;
- the mutations you replayed;
- the state of the originals, before and after;
- the suite counts.

Then stop. Prompts 03 and 04 are written next, against the structure 02 shipped (README §2).
