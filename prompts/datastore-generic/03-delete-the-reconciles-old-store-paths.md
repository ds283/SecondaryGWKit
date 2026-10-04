# Prompt 03 — delete the reconcile's old-store paths

**Campaign:** [`README.md`](README.md) · **Board items:** **R1**–**R4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none on 01–02 (different files), but never in one checkout while another prompt's change
to `ShardedPool.py` is uncommitted. `git status` clean. Commit `e53f323` an ancestor of `HEAD`
(merged into `handover-remedial` on 2026-10-04).
**Closes:** nothing. **Narrows:** `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]`.
**Opens:** nothing expected.
**Recommended model:** **Opus**. It reviews a commit written outside the campaign and changes what
the check at open compares, on the path that decides whether a store opens at all.

**Read first:**

1. [`README.md`](README.md): §0.1, §1, §5 (rules 7–9), §6.1 **D3** and **D4**.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **B1**–**B4**, **B9**,
   **B10**.
3. `git show e53f323`, its commit message included. It was written outside the campaign and merged
   into `handover-remedial` on 2026-10-04, at the user's instruction, before this prompt ran.
4. `Datastore/SQL/ShardedPool.py`: the comment block "THE CHECK AT OPEN" above
   `_reconcile_replicated_tables`; `_replicated_table_specs`; `_read_in_flight_record`;
   `_plan_replicated_repair`; `_read_full_rows`; `_recompute_validated`; `_commit_prune_record`; and
   in the constructor, the creation of `replication_in_flight` on an existing store.
5. `Datastore/SQL/Datastore.py`: `_inserters_for` and `_insert` (how a replicated row is stamped).
6. `Datastore/tests/test_reconcile_at_open.py`, `test_replicated_write.py` (`TestCleanWrite`
   especially) and `test_prune_at_open.py`.
7. `prompts/datastore-integrity/logs/01-*.md` and `02-*.md`: the commit-point table, and why the
   two exceptions were written.

---

## 1. What is wanted

The check at open (`datastore-integrity` prompt 02) carries four accommodations of stores written
before `datastore-integrity` prompt 01 (audit B1–B4). Under D2 they go:

- **R1 — `BackgroundModelValue`'s key exception (B2).** Already deleted, outside the campaign, in
  `e53f323` (D3), which is already on the branch. This prompt **reviews that commit**.
- **R2 — `timestamp` is compared (B1, D4).** Every replicated insert since prompt 01 is stamped
  with the pool's one timestamp, so a current store's copies agree; the audit read this, and this
  prompt proves it before relying on it.
- **R3 — a primary without `replication_in_flight` (B3).**
- **R4 — a `replication_in_flight` table with `controller_shard NOT NULL` (B4)**, and B9's `label`
  test in `_recompute_validated`.

A store written before prompt 01 will then be refused at open (`ReplicatedDivergence`, or `no such
table`), which is loud and true.

---

## 2. What to change

**R1 — review `e53f323`.**
- Review it as the orchestrator would review a prompt's commit: read the diff against
  `_plan_replicated_repair`, `_read_full_rows` and `_describe_differences` at this prompt's `HEAD`;
  check that the copy-as-a-unit check now finds a child row's model through its foreign key to
  `BackgroundModel` for **both** child tables (`BackgroundModel_tags` and `BackgroundModelValue`);
  check the argument in its message that the deleted serial-free check cannot fire under serial keys,
  and say in the log whether it holds.
- **Add the tests it lacks:** (a) a store whose `BackgroundModelValue` rows agree on `(model_serial,
  z_serial)` but differ in `serial` is refused at open, named as a serial split; (b) an interrupted
  store of a `BackgroundModel` whose controller holds the model and its values, and another shard
  none of them, is repaired, the values copied under the controller's serials; (c) a shard holding
  the model row but lacking some of its value rows is refused, not repaired. (Some of these may
  already exist in `test_reconcile_at_open.py`; if so, name them in the log instead of adding.)
- Anything the review finds wrong is fixed in this commit and recorded as a deviation; anything
  outside R1's scope is an observation.

**R2 — compare `timestamp`.**
- **First, measure.** Extend `test_replicated_write.py`'s `TestCleanWrite` approach — every shard's
  clock ticks on every call, so only a timestamp the driver chose can agree — to every path that
  writes a replicated row and is not already covered: the repair at open (a copy forward after an
  interrupted get, store and validate), the completion of an interrupted prune, the version row's
  write, and the prune at open. For each, the copies agree on `timestamp` on every shard. If any path
  stamps per shard, **stop and ask** (§5): that is a defect in current code, not an old-store path.
- Then `_replicated_table_specs` compares every column, `timestamp` included; its docstring loses
  the sentence on stores written before prompt 01; the comment block above
  `_reconcile_replicated_tables` is corrected wherever it says `timestamp` is not compared.
- A test: two shards whose copies of one replicated row differ only in `timestamp` are refused at
  open, the difference named.

**R3 — `replication_in_flight` is assumed.** Delete the constructor's creation of the table on an
existing store, and `_read_in_flight_record`'s `has_table` test. Delete
`test_reconcile_at_open.py`'s `test_a_store_written_before_the_record_existed` and
`test_replicated_write.py`'s `test_an_existing_store_gains_an_empty_record_table_when_opened`. If any
other test or fixture builds a primary without the table and then opens it (check
`Datastore/tests/shard_store_fixtures.py`'s `write_legacy_primary`, which creates none), either it
exists for an old store (delete it, and list it) or it uses the shape as a convenience (re-fixture
it to create the table, and list it); 04 deals with `write_legacy_primary` itself.

**R4 — the `NOT NULL` refusal and B9.** Delete `_commit_prune_record`'s `IntegrityError` handler and
the comment on tables created before a prune could be recorded; delete
`test_prune_at_open.py`'s `TestRecordTableBeforePrune.test_a_pruning_open_refuses_before_any_shard_is_touched`.
In `_recompute_validated`, read `label` unconditionally.

---

## 3. Verification

1. Every suite, with counts before and after, each difference accounted for (tests deleted in R3–R4,
   added in R1–R2).
2. `grep -nE 'before prompt 01|written before|predates' Datastore/SQL/ShardedPool.py`: show the
   output, and account for each remaining line (it may describe crash recovery of current code,
   audit B10, which stays).
3. `black --check` clean.
4. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) restore `excluded = {"timestamp"}`;
   - (ii) in `Datastore._insert`, stamp a replicated row with `datetime.now()` instead of the
     driver's timestamp (R2's measurement must catch it on at least the clean write and the repair);
   - (iii) in the unit check, find a child row's model from the first element of its key again;
   - (iv) restore the constructor's creation of `replication_in_flight` on an existing store and
     delete the table from a test primary: the open must now fail, and the test that says so must
     fail under the mutation.

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. `e53f323` reviewed, with the review in the log, and anything it found fixed in this commit.
2. `_replicated_table_specs` compares every column under the primary key; no old-store path for
   `replication_in_flight`; R2's measurement of every replicated write path.
3. The deletions listed; the new tests; the deliberate-breakage record (i)–(iv).
4. `[00-old-store-fallbacks-are-maintained-for-stores-nobody-keeps]` narrowed: a dated line naming
   what 03 deleted, and naming `e53f323` as reviewed.
5. `black --check` clean. On this board: the §1 row, items R1–R4, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- R2's measurement finds a replicated write that stamps per shard.
- The review of `e53f323` finds that it changed what the repair licenses for a current store.
- Deleting R3 makes a current-code recovery path (audit B10) fail.

---

## 6. What this prompt does not do

- It does not touch the absent-table and absent-column tolerance (audit B5; 05), the legacy shard
  records (B6; 04), or any project name in `ShardedPool` (07).
- It changes no table, column or key (README §5 rule 9).

---

## 7. The log and the board

`logs/03-delete-the-reconciles-old-store-paths.md`, using README §5.1. In addition: the review of
`e53f323`; R2's measurement, path by path; every test deleted, by module, class and method.

`IMPLEMENTATION_STATE.md`: the §1 row for 03, items R1–R4, §3, §5.
