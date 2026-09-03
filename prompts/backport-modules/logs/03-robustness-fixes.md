# Log 03 — Small robustness fixes (F6, B4, D1, D3, D4, F3)

**Prompt:** prompts/backport-modules/03-robustness-fixes.md
**Commit:** `e81b145` — Tidy up latent faults in ShardedPool, RayWorkPool and ClientPool
(corrected 2026-09-04, by prompt 04's housekeeping pass — the SHA originally recorded here,
`3e8a984`, was itself one amend behind the branch tip, exhibiting the same
`[commit-sha-links-stale]` drift as prompts 01 and 02; see that entry, now resolved, in
`IMPLEMENTATION_STATE.md` §4 for the convention adopted from prompt 04 onward to prevent
recurrence)
**Date:** 2026-09-03
**Result:** COMPLETE

## What shipped

- **F6** — `Datastore/SQL/ShardedPool.py:296-315` (`_write_shard_data`). Both the
  `replicated_table_values` insert and the `sharded_table_values` insert are now guarded with
  `if ... :` before `conn.execute(...)`, each with the upstream comment explaining the SQLAlchemy
  2.x `DEFAULT VALUES` behaviour on an empty list. Before: unconditional
  `conn.execute(sqla.insert(self._sharded_tables_table), sharded_table_values)` (and likewise for
  replicated). After: guarded on truthiness of the value list first.

- **B4** — `RayTools/RayWorkPool.py:283-296` (the task-builder dispatch inside `run()`). Added a new
  `if ref_data is None:` branch ahead of the existing `list`/`tuple`/`set` branch: raises
  `RuntimeError("a task builder returned None, which is not compatible with store_results=True")`
  when `self._store_results` is set, otherwise does nothing (no `store_ref` call, so nothing is
  added to `self._inflight`/`self._data`, and the surrounding `count += 1` still consumes one item
  from `self._todo` — no counter desynchronisation). Before: a `None` return fell through to
  `store_ref`'s final `else`, raising `RuntimeError: could not interpret output from task builder
  (object type="NoneType", ...)`.

- **D1** — `Datastore/SQL/ShardedPool.py:85-88` (constructor directory-guard). `self._db_file` → 
  `self._primary_file` in the f-string. `ShardedPool` never had a `_db_file` attribute (the field is
  `_primary_file`, set at line 70); the message now actually renders instead of raising
  `AttributeError`.

- **D3** — `Datastore/SQL/ClientPool.py:163` (`SerialPoolManager.lease_serial`).
  `_default_serial_batch_size[table]` → `_default_serial_batch_size.get(table, 500)`. 500 matches
  `ClientPool.__init__`'s own `default_batch_size: int = 500` parameter default (`ClientPool.py:54`)
  — the only existing fallback constant in the module — rather than inventing a new one.

- **D4** — `RayTools/RayWorkPool.py:550` (notification bookkeeping in `run()`).
  `self._last_num_available_complete = self._num_store_complete` →
  `self._last_num_available_complete = self._num_available_complete`, matching the pattern of the
  three surrounding assignments (`_last_num_lookup_complete = _num_lookup_complete`, etc.).

- **F3** — Taken (the "recommended" option): both files changed together.
  - `Datastore/SQL/Datastore.py:76` — removed `from MetadataConcepts import version`.
  - `Datastore/SQL/Datastore.py:246` (now `:245`) — `self.object_get(version, **version_payload)` →
    `self.object_get("version", **version_payload)`.
  - `Datastore/SQL/ShardedPool.py:12` — removed `from MetadataConcepts import version`.
  - `Datastore/SQL/ShardedPool.py:154` — `shard0_store.object_get.remote(version,
    label=version_label)` → `shard0_store.object_get.remote("version", label=version_label)`.
  - Verified before deleting: `grep -n "version" Datastore/SQL/ShardedPool.py` and the same in
    `Datastore.py` show every remaining hit is `version_label`, `version_serial`, `self._version`,
    `version_payload`, `version_col`, `use_version`/`uses_version`, `VERSION_ID_LENGTH`, the
    `sqla_version_factory` import, or the `cls_name != "version"` string comparison — nothing else
    referenced the imported class.
  - Confirmed `Datastore.object_get` (`Datastore.py:480-484`) already branches on
    `isinstance(ObjectClass, str)`, so passing `"version"` is supported today, not a speculative
    change.

## Deviations from the prompt

### F6 — guard `replicated_table_values` too (IMPLEMENTATION CHOICE)

The prompt only requires guarding `sharded_table_values` (mirroring upstream `SI`) and leaves
guarding `replicated_table_values` to judgement, noting it is not currently reachable because
`config/sharding.py` always configures 13 replicated tables (the README's "14" is a documentation
slip; `config/sharding.py:3-16` lists 13 — this matches `IMPLEMENTATION_STATE.md` §5 note 7, which
also says 13).

**Chosen: guard it too.** Reasoning: the failure mode (a spurious `DEFAULT VALUES` row silently
inserted into `replicated_tables`) is identical in kind to the one being fixed three lines below,
the fix is one `if` line, and leaving it unguarded — while guarding the near-identical
`sharded_table_values` insert right next to it — would read as an oversight to a future reader
rather than a deliberate choice. There is no cost: since the table list is currently guaranteed
non-empty, the guard is inert today and only activates if `config/sharding.py` is ever edited to
configure zero replicated tables.

### D3 — default value of 500 (IMPLEMENTATION CHOICE)

The prompt asks to find the right default rather than invent one, preferring an existing
module-level constant if one exists. `ClientPool.py` has no module-level constant for this purpose,
but `ClientPool.__init__`'s own `default_batch_size` parameter defaults to `500`
(`ClientPool.py:54`), which is the same class that `SerialPoolManager.lease_serial` is constructing
here. Used `500` as the fallback for that reason — it is the pool's own idea of a reasonable
default, not a new value invented for this fix.

### F3 — taken, not skipped (IMPLEMENTATION CHOICE)

The prompt marks F3 optional with "no functional benefit" as an acceptable reason to skip. Taken it
anyway because: (a) it costs nothing behaviourally (`object_get` already special-cases strings), (b)
it is a strict decoupling improvement (one fewer inter-module import), and (c) doing both files
together, verified by grep as required, removes the risk the audit specifically warned about
(fixing one file and not the other). No reason found to prefer leaving the import in place.

No items were dropped from the commit — all six were small as described and none turned out to be
larger in practice.

## Verification performed

1. **Syntax / compile:** `python3 -m py_compile` on all four touched files — clean.
2. **Formatting:** `black --check` on all four touched files — "All done! 4 files would be left
   unchanged." (black available via `/opt/local/bin/black`).
3. **D1:** `grep -n "_db_file" Datastore/SQL/ShardedPool.py` — no matches (only `_shard_db_files`
   hits remain, which is a distinct, correct attribute).
4. **D4:** `grep -n "_last_num_available_complete\|_num_available_complete"
   RayTools/RayWorkPool.py` — confirms the assignment at line 550 now reads from
   `self._num_available_complete`, matching the declaration/increment/read pattern used for the
   other three counters.
5. **F3:** `grep -n "MetadataConcepts\|version" Datastore/SQL/Datastore.py Datastore/SQL/ShardedPool.py`
   — no `MetadataConcepts` hits in either file; every remaining `version` hit is one of
   `version_label`/`version_serial`/`self._version`/`version_payload`/`version_col`/
   `use_version`/`uses_version`/`VERSION_ID_LENGTH`/the `sqla_version_factory` import/the
   `"version"` string literal.
6. **B4 behavioural check (actually run, not just reasoned about):** wrote a throwaway harness at
   `/private/tmp/.../scratchpad/b4_harness.py` (not committed) that imports `RayWorkPool` directly
   with a dummy `pool` object and a task builder that always returns `None`. Had to work around the
   pre-existing broken `from defaults import DEFAULT_ABS_TOLERANCE` import in
   `LiouvilleGreen/WKBtools.py` (tracked as the open `[02-shard-config-reader]` issue in
   `IMPLEMENTATION_STATE.md` §3) with a `sys.modules["defaults"] = config.defaults` stub inside the
   harness — same workaround pattern prompt 02 used, no source touched. Ran under the project's
   `./venv` (which has `ray` 2.43.0 installed; the outer shell's `python3` does not have `ray`).
   Actual output:
   ```
   Case 1 (store_results=False): completed OK, todo empty = True
   Case 2 (store_results=True): raised RuntimeError as expected: a task builder returned None, which is not compatible with store_results=True
   ```
   This is a genuine execution, not a static read — both branches of the B4 fix (skip-when-False,
   raise-when-True) are confirmed working under `ray.init(local_mode=True)`.
7. **F6, D3:** not independently exercised at runtime — reasoned about from the diff (guard is a
   trivial `if value_list:` before an existing call; `.get(table, 500)` is a standard-library dict
   method with well-defined semantics). Both are single-line, low-risk changes where static
   reasoning is sufficient per the prompt's own risk characterisation ("none carries behavioural
   risk").

## Observations not acted on

- The broken `from defaults import DEFAULT_ABS_TOLERANCE` import in `LiouvilleGreen/WKBtools.py:6`
  (already tracked as the open `[02-shard-config-reader]` issue) was worked around again in this
  prompt's B4 harness, using the same `sys.modules` stub technique as prompt 02. Not fixed here —
  still out of this campaign's tracked-item list, and the existing issue already assigns it to
  "whichever prompt first needs a real import" (prompt 04 at the latest). No new information beyond
  confirming the workaround generalises cleanly.
- `README.md` §3 gives "14 replicated tables" for `SGWK`'s `config/sharding.py`; the file actually
  lists 13, consistent with `IMPLEMENTATION_STATE.md` §5 note 7 ("Note the audit says 14 replicated;
  the config lists 13"). Not a defect and not this prompt's territory to correct — noted here only
  because the F6 replicated-table-guard decision above depended on reading that list.

## State handed to the next prompt

- All six items in this prompt are done; none deferred.
- The `[02-shard-config-reader]` issue in `IMPLEMENTATION_STATE.md` §3 remains open and unchanged —
  prompt 04 is the next prompt in the dependency chain that needs a real (non-stubbed) import of
  this code, per that issue's own "Next step".
- The B4 harness at `/private/tmp/.../scratchpad/b4_harness.py` was not committed (scratchpad,
  per instructions) and will not persist to future sessions; the workaround pattern is recorded here
  and in the `[02-shard-config-reader]` issue text for whoever writes the next one.
- Note for whoever runs anything in this repo that imports `Datastore.SQL.*`: use `./venv/bin/python3`
  — the outer environment's `python3` does not have `ray` installed.
