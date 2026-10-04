# Prompt 04 — delete the legacy shard records

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 03 landed (`d8d4afe`) and reviewed (`4ca36c4`). `git status` clean. Never in one
checkout while another prompt's change to `ShardedPool.py` is uncommitted.
**Closes:** nothing. **Narrows:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`.
**Opens:** one issue (§2, L1), and any other the work finds.
**Recommended model:** **Sonnet**. The deletions are mechanical. The work is in the test plan
(§2, L3), which is fixed here method by method, and in the new refusal tests.

**Read first:**

1. [`README.md`](README.md): §0.1, §1, §3, §4, §5 (rules 4, 7–10), §6.1 **D2**.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **B6**, **B10**,
   **G4**.
3. `Datastore/shard_paths.py`, all of it.
4. `Datastore/SQL/ShardedPool.py`: the import from `Datastore.shard_paths`; `_resolve_shard_rows`;
   `_shard_file_problems`; the docstrings of `copy_store` and `delete_store`; `_read_closed_store`;
   `_plan_deletion`.
5. `tools/shard_key_audit.py`, all of it.
6. `docs/handover/quadsource_atol_sweep.py`: `assert_store_is_self_consistent`.
7. `Datastore/tests/shard_store_fixtures.py`, and the test modules named in §2 L3.

---

## 1. What is wanted

A primary written before `b04671f` (2026-09-24) records each shard by its **absolute path**. Every
primary written since records the shard's **bare file name**. Today an absolute record is read by its
final component, as a sibling of the primary, with a notice when it names another directory
(audit B6). Under D2 that reading goes:

- **L1 — the resolver and the pool.** A `shards` record is a bare file name, or it is refused.
  `resolve_shard_path` raises `ValueError` on an absolute record exactly as it does on any other
  record that is not a bare name. So every reader — the constructor, `copy_store`, `move_store`,
  `closed_store_files`, `delete_store`, `store_reader.open_read_only` and the registry operations
  built on them — reports "Shard #N … has an unusable record". `is_legacy_record` and the
  relocation notice go.
- **L2 — the tools.** `tools/shard_key_audit.py` loses its pre-`a2bd966` branch and its tolerance
  of a primary without `shard_key_config`. It also stops reporting a store as OK when its shard
  record is unusable. `assert_store_is_self_consistent` in the sweep script refuses an absolute
  record with its own `RuntimeError`, not a raw `ValueError`. Its docstring stops describing the
  legacy reading.
- **L3 — the tests.** Those that exist for the legacy reading are deleted. Those that used the
  legacy shape as a convenience are re-fixtured. The safety property the legacy tests guarded is
  pinned anew: a primary whose records name **another store's** files never reads, writes or deletes
  those files. Before 04, that held because the absolute record was read by its name. After 04 it
  holds because the record is refused. `write_legacy_primary` is renamed for the users that remain.

**Keep:**
- `_require_bare_name`;
- the "both resolve to one file" check in `_shard_file_problems`;
- `shard_file_problem`;
- `_plan_deletion`'s refusal of a file outside the primary's directory (defence in depth, pinned by
  `test_delete_store.TestRefusals.test_file_outside_the_primary_directory`).

---

## 2. What to change

**L1 — `Datastore/shard_paths.py` and `ShardedPool`.**
- `resolve_shard_path`:
  - delete the `if PurePath(stored).is_absolute():` branch, so every record goes through
    `_require_bare_name(stored, stored)`;
  - its docstring describes one form;
  - drop the `PurePath` import if nothing else uses it.
- Delete `is_legacy_record`.
- The module docstring:
  - describes the one form of record;
  - says that an absolute record is refused, and why: a copied primary's absolute records name the
    original's files, the failure of 2026-09-23;
  - drops the paragraph on reading a legacy record as a sibling, and the stale sentence that
    `prepare()` re-points a primary (it copies through `RunRegistry.stores.copy_store`).

  Keep the paragraphs on the one definition, the naming rule, failing closed and the
  standard-library-only import.
- `ShardedPool`:
  - remove `is_legacy_record` from the import;
  - in `_resolve_shard_rows`, delete `relocated_legacy` and the notice;
  - its docstring, and those of `copy_store` and `delete_store`, stop describing a legacy absolute
    record. `delete_store`'s may say that a record which is not a bare name is refused before
    anything is deleted.
- **The issue this opens.** A closed-store refusal prints its prefix twice. Today
  `closed_store_files` on a primary with an unusable record raises
  `Cannot delete sharded datastore "<p>": Cannot delete sharded datastore "<p>": Shard #0 …`, and
  `copy_store` likewise. The cause is `_read_closed_store` wrapping a message that its caller wraps
  again. This is current code and not 04's; the user will have it cleaned up later (2026-10-04).
  **Do not fix it.** Open
  `[04-closed-store-refusals-repeat-their-prefix]` in this board's §3, with the measured message,
  and add it to the index.

**L2 — `tools/shard_key_audit.py` and the sweep script.**
- **`shard_key_audit.py`:**
  - Delete the `"wavenumber_serial" in shard_keys_columns` branch. A `shard_keys` table without
    `key_serial` meets the remaining "Unrecognised shard_keys schema" refusal (exit 2).
  - `shard_key_config` becomes a required table, beside `shard_keys` and `shards`. A primary that
    lacks it, or whose `shard_key_config` holds no row, is refused with exit 2: it is not a
    completely written primary.
  - **An unusable shard record is a refusal**, exit 2, naming the record and the resolver's reason.
    It no longer ends in "VERDICT: OK". This is the one behaviour change in the tool beyond the
    deletions. It is in scope because 04 is what sends the stores that were legacy down this
    path. A missing shard *file* keeps today's behaviour: the cross-file check is skipped and
    said to be. *Confirmed by the user, 2026-10-04.*
  - The module docstring stops describing a legacy absolute record.
- **`assert_store_is_self_consistent`:**
  - A record that `resolve_shard_path` refuses is reported among the check's mismatches, so that
    the check raises its own `RuntimeError` naming the serial, the record and the resolver's reason.
    *Confirmed by the user, 2026-10-04.*
  - The docstring:
    - describes the one form;
    - keeps the account of 2026-09-23;
    - says that an absolute record is now refused by the resolver.

  Nothing else in the script changes.

**L3 — the tests.** Every method below is named with its fate. README §5 rule 7 applies to every
other test. Line numbers and the evidence are in the orchestrator's notes.

- **The fixture.**
  - Rename `shard_store_fixtures.write_legacy_primary` to **`write_hand_built_primary`**, and
    update every remaining caller.
  - Its docstring says that it builds no `replication_in_flight` table, so that the constructor
    refuses what it builds (since prompt 03). Its users read it through `read_pool`, the
    closed-store operations, the audit tool or the sweep check, none of which reads that table.
  - The module docstring stops saying that it stands in for rows the old code wrote.
  - `read_pool`'s stdout capture and `pool.read_stdout` exist only for the deleted notice tests:
    remove them if nothing else uses them, after a grep.
- **Delete** (each exists only for the legacy reading; the new tests below pin what is left):

  | Module | Class.method |
  |---|---|
  | `Datastore/tests/test_shard_paths.py` | `TestResolveShardPath.test_legacy_absolute_record_resolves_to_the_sibling_by_name` |
  | | `TestResolveShardPath.test_legacy_record_is_not_used_even_when_it_exists_and_differs` |
  | | `TestResolveShardPath.test_is_legacy_record` |
  | `Datastore/tests/test_shardedpool_shard_paths.py` | `TestLegacyRecords.test_legacy_absolute_records_resolve_to_the_primarys_directory` |
  | | `TestLegacyRecords.test_legacy_records_that_name_their_own_directory_are_silent` |
  | | `TestFailClosed.test_missing_sibling_is_refused_even_when_the_absolute_record_exists` |
  | | `TestNoSideEffect.test_reading_a_relocated_legacy_store_leaves_the_primary_unchanged` |
  | `Datastore/tests/test_copy_move_store.py` | `TestCopyOfALegacySource.test_legacy_source_copied_to_a_new_stem_reads_its_own_files` |
  | | `TestMove.test_legacy_source_in_each_layout` |
  | `Datastore/tests/test_delete_store.py` | `TestLegacyRecordsNamingAnotherStore.test_every_listed_file_is_in_b` |
  | | `TestLegacyRecordsNamingAnotherStore.test_delete_removes_only_b_files_and_a_is_untouched` |
  | | `TestInterruption.test_the_legacy_shape_other_directory_is_untouched_throughout` |
  | `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py` | `SweepPrepareTestCase.test_the_check_accepts_a_legacy_store_naming_its_own_siblings` |
  | | `SweepPrepareTestCase.test_the_check_accepts_the_backups_shape` |
  | `RunRegistry/tests/test_store_retire.py` | `TestLegacyShape.test_a_primary_naming_another_populated_stores_shards` |

  That is twelve in `Datastore` and three in `RunRegistry`. A class left with no test goes with
  them.
- **Re-fixture** (the property survives 04; only the records the test builds change, from absolute
  to bare names, unless stated otherwise):
  - `test_shardedpool_shard_paths`:
    - `TestCopiedStore`'s `_original` builds a bare-name store, which serves both its tests;
    - `TestFailClosed.test_moved_store_with_legacy_records_and_no_shards_is_refused`: a bare-name
      store whose primary is moved alone. It may be renamed to drop "legacy_records"; if so, the
      log says so;
    - `TestFailClosed.test_two_records_resolving_to_one_file_are_refused`: records
      `{0: n0, 1: n0}`.
  - `test_copy_move_store.TestRefusals.test_source_shard_missing_symlinked_irregular_or_shared`:
    the shared case's records become `{0: n0, 1: n0}`.
  - `test_delete_store`:
    - `TestRefusals.test_two_serials_share_a_file`: `{0: n0, 1: n0}`;
    - `TestRefusals.test_file_outside_the_primary_directory`: two bare-name stores, with the
      resolver mocked to resolve into the other store's directory.
  - `test_shard_key_audit_copy.TestAuditOfACopiedStore.setUp`: A's records are bare names.
  - `test_quadsource_atol_sweep_prepare`: `test_the_check_refuses_a_renamed_record`,
    `…_a_missing_serial`, `…_an_extra_serial` and `…_two_swapped_serials` build bare-name
    records. The renamed record becomes a wrong bare name.
- **Drop the legacy case** from a method that runs both shapes, and leave the new-style case
  unchanged:
  - `test_shard_paths.TestResolveShardPath.test_result_is_absolute_and_in_the_primarys_directory`:
    its one absolute value;
  - `test_copy_move_store.TestInterruption`: `test_copy`, `test_copy_rewrite_is_one_transaction`,
    `test_move`, `test_move_rewrite_is_one_transaction`. `"legacy"` comes out of the case list;
  - `test_delete_store.TestInterruption.test_every_point_of_interruption`: `SHAPES` loses `"legacy"`.

  Collapse the helpers that only switched shape (`source(kind)`, `_cases`, `store(shape)`,
  `SHAPES`, the `X` branch of `_move`, the `other` parameter that is now always `None`) only where
  the result is simpler and the surviving cases are unchanged. The log says which.
- **Keep, with the fixture renamed and nothing else changed:**
  - `test_shardedpool_shard_paths.TestFailClosed.test_record_that_is_not_a_bare_name_is_refused_on_read`;
  - `test_copy_move_store.TestRefusals.test_source_shard_record_unusable` and
    `test_source_with_no_shards_or_no_shard_zero`;
  - `test_delete_store.TestRefusals.test_no_shard_recorded` and `test_unusable_record`.
- **Dead helpers and imports** after the deletions go in the same commit, after a grep: `_legacy_store`,
  `legacy_source`, `legacy_pair`, and any import left unused (`IMPLEMENTATION CHOICE`). Module
  docstrings that list a deleted item are corrected.
- **New tests.**
  - **`Datastore/tests/test_absolute_shard_record_refused.py`** (new). Two populated stores, built
    in temporary directories:
    - **A**, with bare names;
    - **B**, whose `shards` records are the absolute paths of A's existing shard files. This is
      the shape of the 2026-09-23 failure.

    For each entry point, B is refused, the error names shard #0 and "unusable record", and
    `tree_state` of **both** directories is unchanged. The entry points:
    - `resolve_shard_path` on one such record (`ValueError`);
    - `read_pool` (the constructor's `_read_shard_data`);
    - `copy_store` and `move_store`;
    - `closed_store_files`, `delete_store` and `delete_store(resume=True)`;
    - `store_reader.open_read_only`.

    A primary whose records are absolute paths naming **its own** siblings, the shape every legacy
    store had, is refused the same way. One test of `read_pool` is enough for it.
  - **The audit tool**, in a new module or in `test_shard_key_audit_copy.py`. Each case exits 2,
    and none prints "VERDICT: OK":
    - a primary with B's records;
    - a primary without `shard_key_config`;
    - one whose `shard_key_config` is empty;
    - a `shard_keys` table without `key_serial`, whose message does not mention `a2bd966`.
  - **The sweep check** (`test_quadsource_atol_sweep_prepare.py`): B's records make
    `assert_store_is_self_consistent` raise `RuntimeError` naming serial 0 and the record, before
    anything is written.
  - **The registry** (`test_store_retire.py`): `retire_store` on a store whose records are the
    absolute paths of another populated store's shards, with a registry sidecar, refuses. Nothing is
    deleted, and both stores are byte-identical afterwards. This replaces `TestLegacyShape`.

---

## 3. Verification

1. Every suite, with counts before and after, each difference accounted for. Datastore and
   RunRegistry move; ComputeTargets (`test_quadsource_policy_main`) and the rest must not.
2. `grep -rnE 'legacy|is_legacy_record' Datastore/shard_paths.py Datastore/SQL/ShardedPool.py
   tools/shard_key_audit.py Datastore/tests/shard_store_fixtures.py` prints nothing. Show it.
   `grep -n 'a2bd966\|wavenumber_serial' tools/shard_key_audit.py` prints nothing.
3. `black --check` clean on every touched Python file.
4. **Deliberate breakage.** Each must make a named test fail; then restore it:
   - (i) in `resolve_shard_path`, restore the absolute branch that reads a record's final
     component as a sibling: the new module fails at every entry point, and so do the sweep and
     registry refusal tests;
   - (ii) **the 2026-09-23 failure**: in `resolve_shard_path`, return `Path(stored)` for an absolute
     record whose file exists. The new module must fail. (Under it `move_store` really moves A's
     files: the damage stays inside the test's temporary directories, which is why every store is
     built there);
   - (iii) in the audit tool, let a primary without `shard_key_config` through: its test fails;
   - (iv) in the audit tool, let an unusable record end in "VERDICT: OK": its test fails;
   - (v) in the sweep check, let the resolver's `ValueError` escape: its test fails or errors.

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. No absolute shard record is read by any reader. `is_legacy_record` and the notice are gone.
   `grep` (§3.2) is clean.
2. The audit tool and the sweep check refuse as §2 L2 says.
3. Every test of §2 L3 has the fate listed, and the log lists each by module, class and method. The
   new tests exist, and the deliberate-breakage record (i)–(v) is in the log.
4. `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` narrowed: a dated line naming
   what 04 deleted. `[04-closed-store-refusals-repeat-their-prefix]` opened. The index updated, its
   count corrected.
5. `black --check` clean. On this board: the §1 row, items L1–L3, §3, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- A test not named in §2 L3 fails after the change. That is current behaviour moving, or a user of
  the legacy form the inventory missed. Never edit it to pass.
- A re-fixtured test cannot pass with bare-name records, or passes only by asserting something
  weaker than before.
- Anything outside the files of §6 reads `shards.filename` without going through
  `resolve_shard_path` and would now behave differently.
- The change would require writing, opening or touching any store under `var/` (README §3).

---

## 6. What this prompt does not do, and what it costs

- **The files it changes:**
  - `Datastore/shard_paths.py`;
  - `Datastore/SQL/ShardedPool.py` (the import, `_resolve_shard_rows` and three docstrings);
  - `tools/shard_key_audit.py`;
  - `docs/handover/quadsource_atol_sweep.py` (`assert_store_is_self_consistent` only);
  - `Datastore/tests/shard_store_fixtures.py`;
  - the test modules of §2 L3, and the new ones;
  - the log, this board and `docs/OPEN_ISSUES.md`.

  Nothing else.
- It does not fix the doubled prefix (the issue it opens).
- It does not touch:
  - the absent-table and absent-column tolerance (B5; 05);
  - the four `numeric_serial` selects (05);
  - the registry's legacy **sidecars** (U3). `RunRegistry/stores.py:81-82` explains the sidecar
    rule as "prompt 01's rule for legacy shard records, applied to JSON". After 04 that rule no
    longer exists for shard records. Record this as an observation; the sentence is U3's to keep
    or change.
- It changes no table, column or key (README §5 rule 9). `_write_shard_data` already writes only
  bare names.
- **It costs no store anything.** The one store with absolute records, the A3 baseline, was
  retired by the user on 2026-10-04 (completed 21:31:59) with `store retire` run at `81c0a9a`. `HEAD` could not retire it, because its fingerprint
  formats differ. Its sidecar is the tombstone (board §3, the fallbacks issue). The log need not
  mention it.

---

## 7. The log and the board

`logs/04-delete-the-legacy-shard-records.md`, using README §5.1. In addition:
- every test deleted, re-fixtured, split, kept with the renamed fixture, and added, each by module,
  class and method;
- the helpers removed;
- the measured double-prefix message.

`IMPLEMENTATION_STATE.md`: the §1 row for 04, items L1–L3, §3 (the narrowing and the new issue),
§5. `docs/OPEN_ISSUES.md`: the fallbacks row, the new row, the count and the date.
