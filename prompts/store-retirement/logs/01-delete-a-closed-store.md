# Log 01 — Delete a closed store's files, and only its files, through the one resolver

**Prompt:** [`prompts/store-retirement/01-delete-a-closed-store.md`](../01-delete-a-closed-store.md)
**Commit:** *(this commit)* — "Delete a closed sharded store's own files through the resolver"
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. D0 was recorded as decided, with its wording, before work started. No §5
stop condition was met. For every store shape built here, every file the plan names is in the
primary's own directory, and that includes the legacy shape whose records name another populated
store's shards. The `shards` table is read `mode=ro` and nothing is opened for writing. No existing
test changed, and nothing reads, deletes or refuses because of a sidecar. One issue is opened
(§ "Observations not acted on", item 1).

## What shipped

**`Datastore/SQL/ShardedPool.py`** (+207 / −11).

- **R1, the interface.** There are two new static methods beside `copy_store` and `move_store`:
  - `closed_store_files(primary, *, resume=False) -> List[Path]` (`:693`);
  - `delete_store(primary, *, resume=False) -> List[Path]` (`:708`).

  Both call `_plan_deletion(primary, resume)` (`:1048`), which returns a `_DeletionPlan` (`:46`).
  `closed_store_files` returns `plan.files()`: the shards in ascending serial, then the primary,
  each absolute and resolved. `delete_store` walks the same list. Just before each unlink it
  checks the path again with `shard_file_problem`, calls `os.unlink`, and returns what it deleted,
  in order. On any exception it raises a `RuntimeError` built by `_deletion_failure_message`
  (`:1126`). That message names the step, the exception, and every file of the store still
  present, with any journal beside it. It gives the remedy,
  `ShardedPool.delete_store("<primary>", resume=True)`, run deliberately. Nothing is cleaned up or
  retried.

  Neither method starts Ray, creates an actor or needs an instance. The only database connection
  is `_read_closed_store`'s `sqlite3.connect(f"{primary.as_uri()}?mode=ro", uri=True)`. The new
  code contains no `connect`, no `open` and no `mode=rw`.
- **The refusals**, all in `_plan_deletion` and all made before anything is deleted. Each is a
  `RuntimeError` of the form `Cannot delete sharded datastore "<primary>": <reason>. Nothing was
  deleted`. In order:
  1. **The primary.** It must exist, not be a symbolic link and be a regular file
     (`shard_file_problem`, as `_plan_relocation` checks a source). If it does not exist, the
     message adds that its `shards` table is the only record of which files are the store's, so
     "no file can be identified as the store's".
  2. **A `-journal`, `-wal` or `-shm` beside the primary**, checked before the primary is opened.
  3. **`_read_closed_store(primary_file, "delete", missing_ok=resume)`.** It refuses a `shards`
     table that cannot be read, an unusable record, and a shard that is a symbolic link, is not a
     regular file, or is shared by two serials. Without `resume` it also refuses a missing shard.
     The check is reused, not copied.
  4. **A table that records no shard.**
  5. **The directory assertion.** Every resolved shard's parent must equal the resolved primary's
     parent. The message names the file, its directory, and the primary's directory.
  6. **A journal beside any recorded shard**, including a missing one under `resume`.

  Under `resume=True` the plan's `shards` are the recorded shards that still exist (by
  `os.path.lexists`). `recorded` holds all of them, for the failure message.
- **`resume` relaxes one refusal, and only one.** `_read_closed_store` and `_shard_file_problems`
  each gain a keyword-only `missing_ok: bool = False`. In `_shard_file_problems` it skips the
  per-file check for a path at which **no entry of any kind** exists (`not os.path.lexists`). A
  dangling symbolic link is an entry, so it is still refused. The duplicate-file loop is
  untouched, so two serials resolving to one file are refused whether or not that file exists.
  With the default `False`, the constructor's `_check_shard_files`, copy, move and `_read_back`
  behave exactly as before. Deviations, item 1.
- **R3, the block comment** (`:636-650`), quoted in § "The `CLAUDE.md` and comment changes".

**`CLAUDE.md:52`.** The one sentence is replaced with D0's wording, exactly. Nothing else in the
file changed, and the line was not re-wrapped (`git diff` shows one line changed).

**`Datastore/tests/test_delete_store.py`** (new): 29 test methods, most of them running their
cases as subtests. There is no Ray and no constructor call, nothing under `var/` is touched, and the fixtures are
unchanged.

| Class | Methods | Prompt §3 item |
|---|---|---|
| `TestNewStyleStore` | 2 | **1.** The order of the list, as literal names. The list is absolute and writes nothing. `delete_store` returns the listed files and removes exactly those. A `store.manifest.json`, a `store-notes.txt` and a second complete store (`other.sqlite` + 4 shards) in the same directory are untouched by `tree_state` restricted to them. The directory's mtime is set into the past first, then asserted changed. |
| `TestLegacyRecordsNamingAnotherStore` | 2 | **2.** A precondition: B's records are absolute paths into A, and those files exist. `closed_store_files(B)` is B's five files, each with parent B. `delete_store(B)` leaves A's files, with no A file deleted and no B file left, reported together. `tree_state(A)` and A's own directory entry are unchanged, and A still opens against its own marked files. |
| `TestRefusals` | 13 | **3.** One method per refusal: the primary missing, a directory, a symbolic link; a journal beside the primary (×3 suffixes) and beside shard 2 (×3); a table that cannot be read (not a database; no `shards` table); no shard recorded; unusable records (`../outside.sqlite`, `sub/…`, `""`, `..`, each naming a file that exists); a shard record resolving to a symbolic link (target elsewhere, which survives byte-identical); a directory; two serials sharing a file (present, then missing); a missing shard; a file outside the directory. Each case runs for **both methods**, and for **both values of `resume`** except the missing shard. Each asserts the offending path, a message ending "Nothing was deleted", and `tree_state(root)` unchanged. |
| `TestInterruption` | 2 | **4.** Every unlink n = 1…5 of both shapes, with the injected failure **before** the n-th unlink takes effect and **after** it (20 subtests). A second test re-checks the legacy shape's A after the interruption, the refused retry and the resumed completion. |
| `TestResumeRelaxesOneThingOnly` | 5 | **5.** Under `resume=True`, with a shard also missing so that `resume` has something to relax, these are still refused by both methods with the tree unchanged: a journal (beside the primary, beside a present shard, beside the missing shard); a symbolic link, live and then dangling; a missing primary. `closed_store_files(resume=True)` on a store missing shards 1 and 3 is `[#0, #2, primary]`, which is what `delete_store(resume=True)` then deletes. A whole store under `resume` lists and deletes all of it. |
| `TestRecheckBeforeEachUnlink` | 4 | For mutation (v). The real plan runs, then the tree is changed (`mock.patch.object` on `_plan_deletion`). The changes are: a shard replaced by a symbolic link, a shard replaced by a directory, a shard removed, and the primary replaced by a symbolic link. Each is a refusal at that step, "which it was not when the deletion was planned". The link, its target and the directory survive. |
| `TestNoRay` | 1 | **6.** A listing, a refusal and a deletion. Then `"ray" in sys.modules` and `not ray.is_initialized()`. `tearDownModule` also fails if Ray was ever initialised by the module. |

## Where the shared planning step lives, and why

`_plan_deletion` is a private static method at the end of the closed-store block, after
`_failure_message`, which is where `_plan_relocation` sits relative to copy and move. Both public
methods call it and nothing else plans. So the list prompt 03 records in its tombstone,
`closed_store_files(primary, resume=…)`, **is** the list `delete_store` deletes. The steps come
from `plan.shards` and `plan.primary`, and `closed_store_files` returns `plan.files()` of the same
structure.

It reads shards only through `_read_closed_store`, the read-and-check the constructor, copy and
move share. It keeps the stored records it is handed as `_records` and never uses them. It
reuses `_journal_paths` and `shard_file_problem`. It is not folded into `_plan_relocation`,
because the two share only the primary and journal checks. The destination checks and the
shard #0 requirement mean nothing for a deletion.

## The interruption table

A four-shard store, deleted in the order shard #0, #1, #2, #3, then the primary. The claim: *if
the process dies after any unlink, what is left is a primary and a subset of its shards, or
nothing.* No state has shards that no primary names, because the primary, the only list of the
shard files, goes last. An `os.unlink` either removes a directory entry or does not, so there is
no partial state within a step.

Every row holds for **both shapes**:
- **new:** bare records;
- **legacy:** B's absolute records name A, where a complete store of the same file names exists.

In the legacy shape, `tree_state(A)` and A's directory entry are identical after the failure,
after the refused plain retry and after the resumed completion, in every row. Each row is reached
by two injections: the n-th `os.unlink` raising instead of unlinking, and the (n−1)-th unlinking
and then raising. That is 20 subtests of `TestInterruption.test_every_point_of_interruption`.

| Row | Dies after | Left behind | Constructor's check (`read_pool` + `_check_shard_files`) | `delete_store(primary)` | `delete_store(primary, resume=True)` | Reached by |
|---|---|---|---|---|---|---|
| D0 | no unlink | the whole store | **opens** against its own files | would delete it: it is a whole store | deletes all five, which equals `closed_store_files(resume=True)` | n=1, before |
| D1 | shard #0 | primary, #1–#3 | **refuses** (#0 does not exist) | **refuses**, naming #0 and "does not exist"; tree unchanged | deletes #1, #2, #3, primary | n=1 after; n=2 before |
| D2 | #0, #1 | primary, #2–#3 | **refuses** | **refuses**, naming #0 and #1 | deletes #2, #3, primary | n=2 after; n=3 before |
| D3 | #0–#2 | primary, #3 | **refuses** | **refuses**, naming #0–#2 | deletes #3, primary | n=3 after; n=4 before |
| D4 | all four shards | the primary alone | **refuses** (all four missing) | **refuses**, naming all four | deletes the primary | n=4 after; n=5 before |
| D5 | the primary | nothing | nothing to open. A later opener at that path creates a new store, as at any unused name. That is prompt 03's tombstone, reader and `begin` refusal (README §1). | **refuses**: no primary, "no file can be identified"; tree unchanged | **refuses** the same | n=5 after; every completed deletion |

In every row the failure message:
- names `failed at step "delete shard #k"` or `"delete the primary"`;
- lists exactly the files still present, in plan order, asserted equal to the row's
  "left behind";
- carries the injected error;
- names `delete_store("<primary>", resume=True)`.

After the resumed completion, the tree is what a clean delete leaves:
- the store's five entries are gone;
- every other entry under the root is identical, mtime included, except the store's own
  directory's mtime.

## The `CLAUDE.md` and comment changes

**`CLAUDE.md:52`**, before:

> reason for each rule below. It records; it does not schedule, supervise, restart, lock or delete.

After, D0's wording exactly, on the same line, which was not re-wrapped:

> reason for each rule below. It records; it does not schedule, supervise, restart or lock. It deletes nothing but a store's own files, and those only through `store retire`, which a person runs and which leaves the store's sidecar behind as its record. It never deletes a run directory, a sidecar or any other record.

**`Datastore/SQL/ShardedPool.py`**, the closed-store block comment. Before (`:613-624` at `65c23ee`):

```
    # COPY OR MOVE A CLOSED STORE
    #
    # A store is its primary and its shards, and nothing else: these methods copy, move, check and
    # mention no other file. They are static and work on a closed store. An open pool has one
    # Datastore actor per shard holding its file, so they are not methods of an open pool, and they
    # start no Ray, create no actor and need no instance. They cannot tell whether some process
    # has the store open (a rollback-journal store leaves no file while idle); making sure nothing
    # is using it is the caller's job.
    #
    # They never delete a file, never overwrite one, and never write the source. On failure they
    # do not clean up: they raise, naming the step that failed and the store files that exist at
    # each end. Deleting is for a person.
```

After (`:636-650`):

```
    # COPY, MOVE OR DELETE A CLOSED STORE
    #
    # A store is its primary and its shards, and nothing else: these methods copy, move, delete,
    # check and mention no other file. They are static and work on a closed store. An open pool has
    # one Datastore actor per shard holding its file, so they are not methods of an open pool, and
    # they start no Ray, create no actor and need no instance. They cannot tell whether some process
    # has the store open (a rollback-journal store leaves no file while idle); making sure nothing
    # is using it is the caller's job.
    #
    # copy_store and move_store never delete a file, never overwrite one, and never write the
    # source. delete_store deletes a closed store's own files, and those only: the shards its
    # primary's `shards` table names, read through the one resolver, and then the primary. Its one
    # intended caller is the registry's `store retire`, which keeps the store's sidecar behind as
    # its record. On failure none of them cleans up: each raises, naming the step that failed and
    # the store files that exist. Deleting any other file is for a person.
```

No other comment, docstring or message that existed was changed. `_failure_message`, with its
cross-filesystem advice, is untouched (Observations, item 1).

## Deviations from the prompt

1. **`_read_closed_store` and `_shard_file_problems` gain a keyword-only
   `missing_ok: bool = False`.** `STRUCTURALLY REQUIRED`.
   - **Why a change was needed.** R1 says the plan reads through `_read_closed_store`, "reused and
     not copied", and that `resume` relaxes the missing-shard refusal. `_read_closed_store` raises
     on any problem, so one of three things had to give: its result, a copy of its check, or
     parsing its message. The flag is the smallest change that keeps a single check. It also keeps
     the duplicate check over every serial, missing or not, which a check of the present subset
     alone would lose.
   - **What is unchanged.** With the default, the constructor, copy, move and `_read_back` behave
     exactly as before. Their tests pass unmodified.
   - **The docstrings.** R3 says to change no other docstring, so neither function's docstring was
     edited. The flag is explained by new comments at its two points of use.
2. **The block comment's first paragraph and heading changed as well as its second.**
   `IMPLEMENTATION CHOICE`. R3 names the comment at `:613-624`, which is both paragraphs.
   - The heading became "COPY, MOVE OR DELETE A CLOSED STORE".
   - "these methods copy, move, check and mention no other file" gained "delete," and was
     re-wrapped. Otherwise it would describe the block's methods as not including the new one.
   - "Deleting is for a person" became "Deleting any other file is for a person", which is still
     true and is what D0 says.
3. **The missing-primary refusal always adds "no file can be identified as the store's".**
   `IMPLEMENTATION CHOICE`. R2 asks for this "with the primary gone". The reason is the same
   whether or not `resume` is set, so the message is the same under both.
4. **`closed_store_files` refuses with "Cannot delete sharded datastore … Nothing was deleted".**
   `IMPLEMENTATION CHOICE`. It "refuses exactly as `delete_store` … would", and it is the same
   plan, so the message is the same. Prompt 03 can quote it verbatim.
5. **The directory assertion is tested through a patched resolver.** `IMPLEMENTATION CHOICE`. The
   real resolver cannot produce an outside path, and the §5 condition checks that. So
   `test_file_outside_the_primary_directory` patches `Datastore.SQL.ShardedPool.resolve_shard_path`
   with one that follows an absolute record. That is the "later change to the resolver" the
   assertion guards against.
6. **Tests beyond §3's list.** `IMPLEMENTATION CHOICE`.
   - `TestRecheckBeforeEachUnlink`, without which mutation (v) would go undetected.
   - The "after" injection in `TestInterruption`, which reaches row D5 with a raised error and
     reaches D1–D4 a second way.
   - `test_resume_on_a_whole_store_deletes_all_of_it`.
7. **Mutation (i-b) was run with an empty working directory.** `IMPLEMENTATION CHOICE`, for
   safety. Under it a bare record becomes a path relative to the working directory, and the
   repository root holds real `*.sqlite` files. None has a test's stem, but a mutation run must
   not be one name collision away from deleting one.
8. **The scratch probes deleted files in temporary directories of their own**, which are not
   tests. `IMPLEMENTATION CHOICE`. The hard rule's purpose, that nothing under `var/` or a copy of
   it is at risk, is kept.

None is `UNINTENDED DRIFT`.

## Verification performed

- **The gate.** Before any change, the board's Decisions table and README §6.2 both recorded D0 as
  decided, with the wording.
- **`python -m RunRegistry list`**, at the start: 7 runs, 5 finished, 2 unknown, none `running`.
  Nothing under `var/` was otherwise opened.
- **The new module:** `Ran 29 tests … OK`. The count went 177 → 206 in `Datastore`, exactly the 29
  added.
- **All six suites, after the change**, on the staged tree, from the repository root:

  | Suite | Baseline (orchestrator, at `226889f`) | After this prompt |
  |---|---|---|
  | `AdaptiveLevin` | 32 OK | 32 OK |
  | `ComputeTargets` | 552 OK (the wall-clock flake is known) | 552 OK (the flake did not occur) |
  | `CosmologyModels` | 39 OK | 39 OK |
  | `Datastore` | 177 OK | 206 OK (177 + 29) |
  | `LiouvilleGreen` | 148 OK, skipped=1 | 148 OK, skipped=1 |
  | `RunRegistry` | 128 OK | 128 OK |

  Datastore and RunRegistry were read from a first pass that piped each suite through
  `| tail -40`. For the other four, that tail held only the suites' printed banners, which are
  flushed after the verdict, so those four were re-run with the full output captured and grepped.
  Nothing changed in the tree between the two passes.

- **`black --check`** is clean on `Datastore/SQL/ShardedPool.py` and
  `Datastore/tests/test_delete_store.py`.
- **Nothing else changed.** `git diff --cached --stat` names only the files this commit lists.
  The untracked `docs/datastore-integrity-audit/`, `docs/datastore-integrity-audit.md` and
  `prompts/datastore-integrity/` are not this prompt's, and were not read, staged or changed. The
  last two appeared during the session.
- **No other comment, docstring or message changed.** The diff of `ShardedPool.py` touches the
  block comment R3 names, two signatures (`missing_ok`), two new comments, and new code.
  `_failure_message` is untouched.
- **Mentions.** `git grep -n -i "manifest\|registry\|sidecar" -- Datastore/SQL/` now finds one
  line, the block comment's "the registry's `store retire`, which keeps the store's sidecar
  behind as its record". R3 asks for exactly that sentence. No code in `ShardedPool` reads,
  refuses because of or deletes a sidecar. `test_delete_store.py` writes a `store.manifest.json`
  only to prove it untouched.
- **Scratch probes.** `impl01_messages.py` printed the sample messages. `impl01_iii_probe.py` ran
  the journal cases under mutation (iii). Both built and deleted stores only inside a
  `tempfile.TemporaryDirectory()` of their own.

## The deliberate-breakage record

Every mutation was applied to the staged final tree, and the whole `Datastore/tests` suite was run
from the repository root. The exception is (i-b), whose working directory is noted below. The
command was `PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t .`.

The file was restored with `git checkout --` from the index, and `git diff --stat` came back empty
afterwards. Each diff below is exactly as applied, taken with `git diff` against the index blob
`62cdf07`, which is this commit's `ShardedPool.py`. `git apply --check` accepts each one on the
restored tree.

Unmutated, the suite is `Ran 206 tests … OK`. Every failure below is in `test_delete_store.py`;
no other module's test is affected by any mutation.

**(i) enumerate the shards by opening each stored record as a path, bypassing the resolver.**
`Ran 206 … FAILED (failures=48, errors=12)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 62cdf07..3970b89 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1093,6 +1093,7 @@ class ShardedPool:
             )
         except RuntimeError as e:
             raise refuse(str(e)) from e
+        recorded = {serial: Path(stored) for serial, stored in _records.items()}
 
         if len(recorded) == 0:
             raise refuse(f'the primary "{str(primary_file)}" records no shards')
```

**Outcome: the directory assertion refuses.** Test 2 fails with an error. Both
`TestLegacyRecordsNamingAnotherStore` methods raise:

`Cannot delete sharded datastore "…/B/store.sqlite": shard #0 is "…/A/store-shard0000.sqlite", in the directory "…/A", which is not the primary's own directory "…/B". Nothing was deleted`

Nothing in A or B is deleted. A new-style store's bare records become relative paths, whose
parent is `.`, so the same assertion refuses those too.

Failed:
- `TestNewStyleStore`: both methods (errors);
- `TestLegacyRecordsNamingAnotherStore`: both (errors);
- `TestInterruption.test_every_point_of_interruption`: all 20 subtests;
- `TestInterruption.test_the_legacy_shape_other_directory_is_untouched_throughout`: all 5
  (errors);
- `TestRecheckBeforeEachUnlink`: all 4;
- `TestResumeRelaxesOneThingOnly`: `test_a_journal_is_still_refused` (12 subtests: the refusal
  names the directory, not the journal), `test_the_resumed_list_is_what_is_deleted` and
  `test_resume_on_a_whole_store_deletes_all_of_it` (errors);
- `TestRefusals.test_journal_beside_a_shard`: 12 subtests;
- `TestNoRay` (error).

**(i-b) the same, with the directory assertion also removed.** This variant was run with the
working directory set to an empty scratch directory, `PYTHONPATH` and `-s`/`-t` given as absolute
paths into the repository. Under it a bare record is opened relative to the working directory, and
the repository root holds real `*.sqlite` files. The scratch directory was still empty afterwards.
`Ran 206 … FAILED (failures=62, errors=3)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 62cdf07..810ca26 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1093,6 +1093,7 @@ class ShardedPool:
             )
         except RuntimeError as e:
             raise refuse(str(e)) from e
+        recorded = {serial: Path(stored) for serial, stored in _records.items()}
 
         if len(recorded) == 0:
             raise refuse(f'the primary "{str(primary_file)}" records no shards')
@@ -1100,11 +1101,6 @@ class ShardedPool:
         # the resolver puts every shard in the primary's directory. Asserted here as well, so that
         # no later change to the resolver can make this method delete a file anywhere else
         directory = primary_file.parent
-        for serial, path in sorted(recorded.items()):
-            if Path(path).parent != directory:
-                raise refuse(
-                    f'shard #{serial} is "{str(path)}", in the directory "{str(Path(path).parent)}", which is not the primary\'s own directory "{str(directory)}"'
-                )
 
         for serial, path in sorted(recorded.items()):
             refuse_journals(Path(path), f"shard #{serial}")
```

**A's files are the ones deleted.** This is the failure the prompt exists to rule out, shown on a
fixture. Test 2's assertion, with `$TMPDIR/tmp_amp52fw` for the temporary root:

```
- {"A's files deleted": ['$TMPDIR/tmp_amp52fw/A/store-shard0000.sqlite',
-                        '$TMPDIR/tmp_amp52fw/A/store-shard0001.sqlite',
-                        '$TMPDIR/tmp_amp52fw/A/store-shard0002.sqlite',
-                        '$TMPDIR/tmp_amp52fw/A/store-shard0003.sqlite'],
-  "B's files left": ['$TMPDIR/tmp_amp52fw/B/store-shard0000.sqlite',
-                     '$TMPDIR/tmp_amp52fw/B/store-shard0001.sqlite',
-                     '$TMPDIR/tmp_amp52fw/B/store-shard0002.sqlite',
-                     '$TMPDIR/tmp_amp52fw/B/store-shard0003.sqlite']}
```

So all four of A's shards and B's primary were deleted. B's four shards are left, and no primary
names them. `test_every_listed_file_is_in_b` fails the same way: the listing is A's four shards,
then B's primary. For the live stores, that is the A3 store's four shards gone, and the backup's
four left with no primary.

The rest of what failed in (i-b):
- `TestRefusals.test_file_outside_the_primary_directory`, all 4 subtests, and its closing check
  (error). A's files are deleted, and nothing refuses;
- the new-style tests. The relative paths resolve against the empty working directory, so the
  re-check refuses each at step "delete shard #0", and nothing is deleted anywhere;
- the rest as in (i).

**(ii) delete the primary first.** `Ran 206 … FAILED (failures=26, errors=4)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 62cdf07..8826105 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -732,10 +732,10 @@ class ShardedPool:
         deliberately.
         """
         plan = ShardedPool._plan_deletion(primary, resume)
-        steps = [
+        steps = [("delete the primary", plan.primary)] + [
             (f"delete shard #{serial}", plan.shards[serial])
             for serial in sorted(plan.shards)
-        ] + [("delete the primary", plan.primary)]
+        ]
 
         deleted: List[Path] = []
         step = "check the files before deleting them"
```

Failed:
- `TestInterruption.test_every_point_of_interruption`: all 20 subtests. After one unlink the
  primary is gone and four shards remain that nothing names;
- `TestInterruption.test_the_legacy_shape_other_directory_is_untouched_throughout`, unlink = 2–5
  (errors). The resumed completion refuses, because with no primary "no file can be
  identified", so those shards can never be removed through the method;
- `TestNewStyleStore.test_deletes_exactly_its_files_and_nothing_beside_them`,
  `TestLegacyRecordsNamingAnotherStore.test_delete_removes_only_b_files_and_a_is_untouched`,
  `TestResumeRelaxesOneThingOnly.test_the_resumed_list_is_what_is_deleted` and
  `test_resume_on_a_whole_store_deletes_all_of_it`. The returned order is not
  `closed_store_files`';
- `TestRecheckBeforeEachUnlink.test_a_shard_that_became_a_symbolic_link` and
  `test_a_shard_that_vanished`. The primary is already gone when the step refuses.

**(iii) drop the journal check**, beside the primary and beside each shard.
`Ran 206 … FAILED (failures=42)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 62cdf07..9b07dce 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -1082,8 +1082,6 @@ class ShardedPool:
                         f'{what} "{str(path)}" has "{str(journal)}" beside it, so it was not closed cleanly or is open now'
                     )
 
-        # checked before the primary is opened
-        refuse_journals(primary_file, "the primary")
 
         # the shard records, read and checked exactly as the constructor reads them. Under resume a
         # missing shard is not refused, and nothing else is relaxed
@@ -1106,8 +1104,6 @@ class ShardedPool:
                     f'shard #{serial} is "{str(path)}", in the directory "{str(Path(path).parent)}", which is not the primary\'s own directory "{str(directory)}"'
                 )
 
-        for serial, path in sorted(recorded.items()):
-            refuse_journals(Path(path), f"shard #{serial}")
 
         # every shard, or under resume those still present, in ascending serial
         shards = {
```

Failed:
- `TestRefusals.test_journal_beside_the_primary`: 12 subtests;
- `TestRefusals.test_journal_beside_a_shard`: 12;
- `TestResumeRelaxesOneThingOnly.test_a_journal_is_still_refused`: 18.

In the suite, later subtests meet what an earlier, unrefused one deleted. So each case was probed
on a fresh store under the mutation (scratch `impl01_iii_probe.py`, a temporary directory of its
own):

| Journal | Under the mutation |
|---|---|
| `-journal` beside the primary | **refused for the wrong reason**, "its shards table could not be read (attempt to write a readonly database)". The journal is never named. |
| `-wal` beside the primary | **went ahead**. It left `store.sqlite-wal` and a **new** `store.sqlite-shm`. SQLite created the `-shm` while opening the primary `mode=ro`. |
| `-shm` beside the primary | went ahead, left `store.sqlite-shm` |
| `-journal`, `-wal`, `-shm` beside shard #2 | went ahead, left the journal |

So in 5 of 6 cases the explicit check is the only thing that refuses. The `-wal` case also shows
why the primary's journals are checked **before** it is opened: a `mode=ro` open beside a WAL
creates a file.

**(iv) let `resume=True` tolerate a symbolic link.** `Ran 206 … FAILED (failures=6)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 62cdf07..8b81e9d 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -613,7 +613,7 @@ class ShardedPool:
             # missing_ok (delete_store(resume=True) only) passes over a shard of which no entry of
             # any kind exists. A dangling symbolic link is an entry, and is still refused, and so is
             # a pair of serials resolving to one file, missing or not
-            if missing_ok and not os.path.lexists(path):
+            if missing_ok and (not os.path.lexists(path) or os.path.islink(path)):
                 continue
             problem = shard_file_problem(Path(path))
             if problem is not None:
```

Failed:
- `TestRefusals.test_record_resolves_to_a_symbolic_link`, the two `resume=True` subtests;
- `TestResumeRelaxesOneThingOnly.test_a_symbolic_link_is_still_refused`, all 4: the live link and
  the dangling link, each by both methods.

In each, `closed_store_files` no longer refuses (`RuntimeError not raised`). `delete_store` still
raises, but only when the re-check reaches the link, after deleting the shards before it. So
"Nothing was deleted" is false and the tree has changed. The re-check keeps the link's target
safe. The plan no longer does.

**(v) drop the re-check just before each unlink.** `Ran 206 … FAILED (failures=4)`.

```diff
diff --git a/Datastore/SQL/ShardedPool.py b/Datastore/SQL/ShardedPool.py
index 62cdf07..eb42247 100644
--- a/Datastore/SQL/ShardedPool.py
+++ b/Datastore/SQL/ShardedPool.py
@@ -743,11 +743,6 @@ class ShardedPool:
             for step, path in steps:
                 # the plan checked every file; check again just before the unlink, because a file
                 # that changed since is not the file that was planned
-                problem = shard_file_problem(path)
-                if problem is not None:
-                    raise RuntimeError(
-                        f'"{str(path)}" {problem}, which it was not when the deletion was planned'
-                    )
                 os.unlink(path)
                 deleted.append(path)
         except Exception as e:
```

Failed: all 4 of `TestRecheckBeforeEachUnlink`.
- **A shard that became a symbolic link,** and **a primary that became one:** `RuntimeError not
  raised`. `os.unlink` removed the link silently, and the deletion "succeeded" over a file that
  was not the one planned.
- **A shard that became a directory:** a `PermissionError` from `unlink`, not the "not a regular
  file" refusal.
- **A shard that vanished:** a `FileNotFoundError`, not the "does not exist" refusal.

§3's six items alone would not catch this mutation, which is why the class exists (Deviations,
item 6).

## Observations not acted on

1. **`_failure_message`'s cross-filesystem advice now contradicts decision 6.1.1.** Opened as
   **[01-cross-filesystem-move-advice-says-delete-by-hand]** (board §3).
   - A move across filesystems ends with "copy it instead, and then delete the source by hand"
     (`ShardedPool.py`, `_failure_message`).
   - `RunRegistry.stores.move_store` calls `ShardedPool.move_store` (`stores.py:753`), so
     `store move` surfaces that advice for a registered store.
   - After this campaign, a registered store is removed by `store retire` and never by hand
     (README §6.1.1). A person who followed the advice would leave the source's sidecar describing
     nothing, the broken case of audit §2.1.
   - The prompt said to leave the text alone. `test_copy_move_store.py`'s
     `test_move_across_filesystems_fails_before_anything_moves` asserts it verbatim, so rewording
     it needs a decision about that test.
2. **A legacy primary prints the resolver's `!!` notice on stdout** during `closed_store_files`
   and `delete_store`, as in the constructor and copy. That is `_resolve_shard_rows`'s behaviour,
   unchanged. Prompt 03's `store retire` will show it in its output unless it captures it. Not an
   issue.
3. **A plain `delete_store` on a partly deleted store refuses without naming `resume=True`.** It
   names the missing shards and "does not exist", as R2 asks. Only the failure message names the
   remedy. Whether `store retire` says more is prompt 03's to decide. Not an issue.
4. **The `RunRegistry/stores.py` docstring (`:47-49`) and `RunRegistry/__main__.py` (`:13`) still
   say "none deletes anything."** Prompt 03 amends them (README D0). Not touched here.
5. **Mutation (iii) showed that a `mode=ro` open of a primary with a `-wal` beside it creates a
   `-shm`.** The shipped order, journals checked before the open, means this cannot happen here.
   `_plan_relocation` checks in the same order. Not an issue.

## State handed to the next prompt

- **For prompt 03:** `ShardedPool.closed_store_files(primary, *, resume=False)` and
  `ShardedPool.delete_store(primary, *, resume=False)`, as README §4 names them.
  - **What it can rely on:**
    - one plan, so the list recorded before deleting is the list deleted;
    - shards in ascending serial, then the primary, all resolved absolute paths in the primary's
      directory;
    - every refusal a `RuntimeError` ending "Nothing was deleted";
    - every failure a `RuntimeError` beginning `delete of sharded datastore "…" failed at step
      "…"`, listing `Store files still present: [...]` and naming
      `delete_store("…", resume=True)`.
  - **Under `resume=True`,** only a missing shard is tolerated. A missing primary, "no file can be
    identified", is refused under both, so the completion path must check "no listed file
    remains" itself once the primary is gone, as prompt 03 §3.6 already says.
  - **A table that cannot be read** is refused under both, with "its shards table could not be
    read", which is the case README D4's narrowing refuses.
- **R1–R3 are done.** No sidecar, registry or `tools/sharded_store.py` code was touched.
- **Issue opened:** `[01-cross-filesystem-move-advice-says-delete-by-hand]`, unassigned.
