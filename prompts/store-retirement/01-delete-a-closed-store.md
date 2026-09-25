# Prompt 01 — delete a closed store's files, and only its files, through the one resolver

**Campaign:** [`README.md`](README.md) · **Board items:** **R1**–**R3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** **D0** (README §6.2). Do not start unless the board records D0 as decided, with the
user's wording for `CLAUDE.md:52`. If it does not, stop and say so.
**Closes:** nothing. **Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. The code is short. The judgement is in making it impossible for
this method to delete a file that is not the store's, including in the one real case where the
store's own records name another store's files (audit §2.7).

**Read first:**

1. [`README.md`](README.md): §0, §1, §4, §5 and §6 in full. §6.1 records the user's decisions. Do
   not reopen them.
2. [`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) §2.7–§2.10.
3. `Datastore/shard_paths.py`, the one resolver, in full.
4. `Datastore/SQL/ShardedPool.py`: the closed-store block, from `# SHARD RECORDS: READ AND CHECK`
   (`:532`) to the end of `_failure_message`. That covers `_resolve_shard_rows`,
   `_shard_file_problems`, `_journal_paths`, `_read_closed_store`, `_plan_relocation`,
   `_relocate_store` and `_failure_message`. Read too the constructor's two branches (`:116-152`),
   which say what a later opener does with what you leave behind.
5. `prompts/datastore-portability/02-copy-and-move-a-store.md` and its log. This prompt adds a third
   operation beside copy and move, in the same style, with the same refusals, and with the same
   kind of interruption table.
6. `Datastore/tests/shard_store_fixtures.py` (`write_new_store`, `write_legacy_primary`,
   `read_pool`, `tree_state`) and `Datastore/tests/test_copy_move_store.py`: the no-Ray pattern you
   extend.
7. `CLAUDE.md`, the repository mechanics section, and line 52.

---

## 1. What is wanted

A registry operation, prompt 03's `store retire`, will remove a store's files and keep its sidecar.
The file half belongs here, beside `copy_store` and `move_store`. Only `ShardedPool`'s one resolver
can say which files are a store's (`datastore-portability` README §6.2–§6.3). This prompt ships the
file half, and nothing else.

`ShardedPool` still knows nothing about sidecars. The method deletes the primary and its shards.
It never deletes, reads, refuses because of or mentions any other file, a `<stem>.manifest.json`
included. It is **not** reachable from `tools/sharded_store.py` (D7). Its only intended caller is
prompt 03.

**The danger this prompt exists to rule out.** The backup's primary records its four shards by
**legacy absolute path**. Those paths are, byte for byte, the **live** A3 store's shard files in
`var/datastores/` (audit §2.7, probed). The resolver reads each record as the sibling of that name,
in the backup's own directory. A deletion that opened a stored record as a path would pass every
test built on a modern store, whose records are bare names. On the backup it would delete the live
A3 store. **Enumerate a store's files through `_read_closed_store` and nothing else**, and prove it
with the test in §3 item 2.

---

## 2. What to change

**R1 — the interface.** Two static methods on `ShardedPool`, beside `copy_store` and `move_store`,
sharing one planning step (*prompt's choice*, README §6.3). That way the list prompt 03 records in
its tombstone is the list that is deleted:

```python
ShardedPool.closed_store_files(primary: PathType) -> List[Path]
ShardedPool.delete_store(primary: PathType, *, resume: bool = False) -> List[Path]
```

`closed_store_files` returns every file of the closed store: its shards in ascending serial, then
the primary, each absolute. It refuses exactly as `delete_store(primary)` would, and never writes.
`delete_store` deletes those files **in that order**, shards first and the primary last, and
returns the paths it deleted, in order. Neither starts Ray, creates an actor or needs an instance.
The primary is opened only `mode=ro`, to read its `shards` table, and **no file is ever opened for
writing**.

*Before anything is deleted*, refuse with a `RuntimeError` that names the file and the reason, and
ends "Nothing was deleted", when:
- the primary does not exist, is a symbolic link or is not a regular file (`shard_file_problem`,
  as `_plan_relocation` checks the source);
- the primary or any shard has a `-journal`, `-wal` or `-shm` beside it. A hot journal means the
  store was not closed cleanly, or is open now. Check the primary's journal before opening it;
- the `shards` table cannot be read, or records no shard;
- any record is unusable, or resolves to a symbolic link, a non-regular file, or a file shared by
  two serials. This is `_read_closed_store`'s check, reused and not copied;
- any shard's file is missing. **Under `resume=True` only, a missing shard is not a refusal**: it
  is what an interrupted deletion leaves (R2);
- **any file to be deleted is not in the primary's own directory.** The resolver guarantees this.
  Assert it anyway, as a separate check, so that a later change to the resolver cannot turn this
  method into one that deletes elsewhere. Name the file and its directory in the message.

*The deletion.* Delete each shard with `os.unlink`, in ascending serial, then the primary. Just
before each unlink, check that the path is still a regular file and not a symbolic link. A file that
changed between the plan and the delete is a refusal at that step, not a silent skip.

*On failure* the method does not clean up and does not retry (README §5 rule 11). It raises,
naming the step that failed and every file of the store still present, in the style of
`_failure_message`. The remedy it names is `delete_store(primary, resume=True)`, run
deliberately.

**R2 — the interruption property.** Establish it, and record it in the log as a table with one row
per point of interruption: *if the process dies after any unlink, the files left behind are a
primary and a subset of its shards, or nothing.*
- The constructor refuses to open that state (`_check_shard_files`, `:150-152`).
- `delete_store(primary)` refuses it, naming the missing shards.
- `delete_store(primary, resume=True)` completes it.

No state may be left in which shards exist that no primary names. That is why the primary goes
last: its `shards` table is the only list of the shard files. Work through at least a new-style
store and a legacy store whose records name another populated directory.

`resume=True` relaxes exactly one refusal, the missing shard, and nothing else. A symbolic link, a
journal, a duplicate, an unreadable table or a file outside the directory is refused under it as
before. With the primary gone there is nothing this method can identify. Then it refuses, saying
so, and does not guess.

**R3 — the charter, in the words D0 records.**
- Amend `CLAUDE.md:52` to the wording the board records for D0, **exactly**.
- Amend the comment at the head of the closed-store block in `ShardedPool.py` (`:613-624`: "They
  never delete a file … Deleting is for a person."). Say that `copy_store` and `move_store` never
  delete a file, and that `delete_store` deletes a closed store's own files, and those only. Say too
  that its one intended caller is the registry's `store retire`, which keeps the sidecar as the
  record.

Change no other comment, docstring or message. In particular, `_failure_message`'s cross-filesystem
advice, "copy it instead, and then delete the source by hand", is left alone, and listed under
"Observations not acted on". Existing tests may assert its text.

---

## 3. Tests — a new module in `Datastore/tests/`, no Ray, nothing under `var/`

Build stores with `shard_store_fixtures.py` in temporary directories. Extend the fixtures if you
need to; do not modify an existing test. At minimum:

1. **A new-style store.** `closed_store_files` lists the shards in ascending serial and then the
   primary. `delete_store` returns the same list and removes exactly those files. The entries
   below sit beside the store, and are **untouched** afterwards, by `tree_state` restricted to
   them. The directory's own mtime changes, as any deletion must change it:
   - a `<stem>.manifest.json`;
   - a `<stem>-notes.txt`;
   - a second complete store in the same directory, under another stem.
2. **The legacy case, the one that matters.** Directory *A* holds a complete, populated store.
   Directory *B* holds a store under the **same file names**, whose primary records its shards by
   absolute paths **into A** (`write_legacy_primary`). `delete_store(B)` removes exactly *B*'s
   five files, and **every file in A is byte-identical, with unchanged mtimes**. Also check
   `closed_store_files(B)`: every path it returns is in *B*.
3. **The refusals.** One test per refusal in R1. Each asserts that the error names the offending
   file and says nothing was deleted. It also asserts that **no file anywhere under the temporary
   root was created, changed or removed**, by `tree_state` before and after. Include a journal
   beside a shard as well as beside the primary, and include a shard record that resolves to a
   symbolic link whose target is a real file elsewhere. The target must survive.
4. **Interruption.** For each unlink of each store shape in R2, make the *n*th `os.unlink` raise by
   monkeypatching it. Then assert, in order:
   - the error names the step and the files still present;
   - the file state is the table's row;
   - the constructor's check refuses the state (`read_pool`, or the `object.__new__` pattern the
     fixtures use);
   - `delete_store(primary)` refuses it;
   - `delete_store(primary, resume=True)` completes it, leaving the tree exactly as a clean delete
     would have left it.

   Cover the legacy shape too: its other directory must be untouched throughout.
5. **`resume` relaxes one thing only.** Under `resume=True`, a journal, a symbolic link and a
   missing primary are each still refused, with the tree unchanged.
6. **No Ray.** After the tests above, `ray.is_initialized()` is false. The module imports
   `ShardedPool`, so `ray` is imported; it must never be initialised.

**Deliberate breakage.** The interface is new, so "fails on the old tree" proves nothing. Show
instead that each of these mutations makes the tests written against it fail, then restore it:
- (i) enumerate the shards by opening each **stored record** as a path (`Path(stored)`), bypassing
  the resolver. Test 2 must fail, whether by deleting *A*'s files or by the directory assertion
  refusing. Say which, and show that with the directory assertion also removed, *A*'s files are the
  ones deleted. That is the failure this prompt exists to rule out, demonstrated on a fixture and
  never on `var/`;
- (ii) delete the primary first;
- (iii) drop the journal check;
- (iv) let `resume=True` tolerate a symbolic link;
- (v) drop the re-check just before each unlink.

Put each mutation in the log as a short diff, **exactly as applied**, so the orchestrator can
replay it with `git apply`. Name the tests that failed. Mutations are never committed.

**No real-store demonstration.** Every real primary under `var/` holds legacy absolute records.
The backup's name the live A3 store's shards (audit §2.7). A demonstration on a hand-copy of a real
store would therefore put a real store's files one defect away from deletion. Test 2 reproduces
that shape exactly. **Never point `closed_store_files` or `delete_store` at anything under `var/`,
or at a copy of anything under it.**

---

## 4. Acceptance

1. §3's tests exist in a new module under `Datastore/tests/`. They need no Ray and open nothing
   under `var/`. Every existing test passes unmodified.
2. The deliberate-breakage record: mutations (i)–(v), each with its diff and the tests that failed.
   Under (i), which of the two outcomes occurred, and the directory-assertion-removed variant.
3. The interruption table in the log, one row per point of interruption, each backed by a test.
4. `closed_store_files` and `delete_store` share one planning step, which reads through
   `_read_closed_store`. No shard record is ever opened as a path.
5. `CLAUDE.md:52` reads exactly as D0 records, and the `ShardedPool` block comment is amended as
   R3 says. No other comment, docstring or message changed.
6. Every existing suite unchanged. `Datastore/tests` rises by exactly the tests you add.
7. `black --check` clean. On the board: the §1 row for 01, items R1–R3, and §5 baselines.
   `docs/OPEN_ISSUES.md` only if you open an issue. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- The board does not record D0 as decided, with its wording.
- Any file the plan would delete is outside the primary's directory, for any store shape you can
  build. That would mean the resolver does not guarantee what audit §2.7 relies on.
- Reading the `shards` table needs any mode but `mode=ro`, or the method needs to open any file for
  writing.
- An existing test would have to change.
- You find yourself wanting to read, delete or refuse because of a sidecar, to consult the
  registry, or to add a `delete` to `tools/sharded_store.py`. The first two are prompt 03's; the
  third is refused by D7.
- Any step would point the new methods, a test, or a script at anything under `var/`.

---

## 6. What this prompt does not do

- It does not read, write, delete or mention any sidecar, and does not consult the registry.
- It does not touch `tools/sharded_store.py`, `copy_store`, `move_store`, the constructor, or any
  error message that exists.
- It does not delete anything under `var/`, or anything a test of its own did not build.
- It does not change `RunRegistry/`. That is prompt 03, which is held.

---

## 7. The log and the board

`logs/01-delete-a-closed-store.md`, using the template in README §5.1. In addition:
- where the shared planning step lives, and why;
- the interruption table;
- the deliberate-breakage record, with diffs, and mutation (i)'s two variants;
- the `CLAUDE.md` and comment changes, quoted before and after.

`IMPLEMENTATION_STATE.md`: the §1 row for 01, items R1–R3, and §5's baselines. Leave the held rows
for 03–05 as they are.
