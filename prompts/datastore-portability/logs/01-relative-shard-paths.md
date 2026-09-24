# Log 01 — Shard paths relative to the primary, and fail closed when a shard is missing

**Prompt:** [`prompts/datastore-portability/01-relative-shard-paths.md`](../01-relative-shard-paths.md)
**Commit:** *(this commit)* — "Record shard paths relative to the primary and refuse missing shards"
**Model:** Claude Opus 5.5
**Date:** 2026-09-24
**Result:** COMPLETE, with one deviation in the §4 demonstration that the prompt forced (below,
"Deviations", item 1). No §7 stop condition was met.

## What shipped

**`Datastore/shard_paths.py`** (new, 124 lines, standard library only): the one resolver.

- `resolve_shard_path(primary, stored) -> Path` always returns `primary.parent / <bare name>`.
  A bare name (the current form) is that name beside the primary. An absolute path (the legacy
  form) is **its final component** beside the primary. The absolute path itself is never
  returned, whether or not it exists, and there is no fallback of any kind. Anything else raises
  `ValueError`: empty, `.`, `..`, a record with `/` or `\` in it, a NUL, a non-string, and a
  legacy record whose last component is not a bare name (`/`, `/x/..`, `/x/.`, `/x/`). For
  legacy records the name is taken from the raw string, not from `PurePath`, because `PurePath`
  normalises `/x/A/.` to name `A`. It also requires an absolute `primary`, so the result is
  absolute by construction. That matters because the result goes to Ray actors, whose working
  directory is not the driver's.
- `shard_file_problem(path)` returns `None` for a usable shard. For anything else it says why:
  missing, not a regular file, or **a symbolic link**. It is the one definition of an acceptable
  shard file, and both callers use it.
- `is_legacy_record(stored)`.

**`Datastore/SQL/ShardedPool.py`**: +103 / −11. It covers the constructor's existing-store branch
and the three methods that touch the `shards` table.

- **P1.** `_check_shard_files()` (`:505`) is a method, called at `:128` in the existing-store
  branch, straight after `_read_shard_data()`. It raises `RuntimeError` if any resolved shard is
  missing, is not a regular file, or is a symlink, and also if two serials resolve to the same
  file. The message names the primary, and for each shard gives the serial, the stored record,
  the resolved path and the reason. **`SerialPoolBroker` is now created after the branch**
  (`:146`, moved from `:79`), so nothing on the Ray side exists when the check runs. Deviations,
  item 2, explains why.
- **P2.** `_write_shard_data` (`:287`) records each shard as
  `shard_file.relative_to(primary.parent)`. Every shard is a sibling of the primary, so the record
  is the bare file name. A new store is therefore written in the new form. Before writing,
  the method checks that the record resolves back to the same file. If it does not, it raises.
- **P3.** `_read_shard_data` (`:347`) sends every `shards.filename` through `resolve_shard_path`,
  and a refused record becomes a `RuntimeError` naming the primary and the serial. It keeps the
  stored value in a new `_shard_records` map, so that the P1 message can quote it. **The rows are
  never rewritten.** When a legacy record resolves somewhere other than its literal value, the
  method prints one `!!` line per store naming the old directory and the new one. Nothing is
  printed when the literal value and the resolved path agree. The two real stores are in that
  case, since each names its own shards.
- `_create_engine` is untouched, including its use of the unresolved `_db_name` (prompt §5).

**`tools/shard_key_audit.py`**: +29 / −3. **P4.** The shard it attaches for the cross-file check
comes from `resolve_shard_path` and passes `shard_file_problem`, exactly as `ShardedPool` does.
The tool now prints the path it attaches and the record it read, as
`>> cross-file check against shard #0: <path> (record '<stored>')`. It opens nothing it did
not open before, and everything is still opened `mode=ro`.

**Tests**, in `Datastore/tests/`. 27 new test methods in three modules plus one fixture module. No
Ray, no datastore, and the constructor is never called.

| Module | Methods | Prompt §3 test |
|---|---|---|
| `test_shard_paths.py` | 10 | **1**: the resolver as a pure function, `shard_file_problem`, and that importing the module pulls in no `ray`, `sqlalchemy` or `Datastore.SQL` |
| `test_shardedpool_shard_paths.py` | 14 | **2** round trip and move (plus renaming the primary alone) · **3** legacy records into *A* read from *B* (plus the silent unmoved case) · **4** the copy case, with *A* present and populated (plus the copy with a renamed primary) · **5** fail closed: a missing shard; **no fallback** when the sibling is missing but the absolute record exists; the P0 case after the fix; a symlinked shard; two records collapsing onto one file; a record that is not a bare name · **6** no side effect, legacy and new |
| `test_shard_key_audit_copy.py` | 3 | P4 / §6 item 5: the audit of a copy attaches the copy's shard 0; no fallback to the original when the copy's is missing; the tool imports nothing heavy when run standalone |
| `shard_store_fixtures.py` | — | the hand-built legacy primary (`sqlite3` DDL mirroring `_create_engine`), `object.__new__(ShardedPool)` instances, placeholders |

The two `ShardedPool`-side test modules deliberately do not import `Datastore.shard_paths`. That
way they still load against the unfixed tree for the deliberate-breakage record.

## P0 — what a moved store did on the unfixed tree

The measurement came first, before any change, at `71c4c66`, in session scratch space, through
the real constructor with a local Ray (`num_cpus=2`), `config.sharding`'s tables and 2 shards.

1. Created `…/scratchpad/p0/orig/p0store.sqlite`. Its `shards` table held
   `(0, '/private/tmp/…/p0/orig/p0store-shard0000.sqlite')` and the same for shard 1.
2. `mv p0/orig p0/moved`. Before reopening, `p0/orig/` did not exist.
3. Reopened `p0/moved/p0store.sqlite` in a fresh process. The output was:

   ```
   >> Opened existing sharded datastore "/private/tmp/…/p0/moved/p0store.sqlite" with 2 shards
   P0: pool OPENED; shard files in use:
   P0:   shard 0: /private/tmp/…/p0/orig/p0store-shard0000.sqlite  exists=True
   P0:   shard 1: /private/tmp/…/p0/orig/p0store-shard0001.sqlite  exists=True
   ```

   **No exception, and no warning.** The pool opened. Afterwards **`p0/orig/` had been
   recreated**, holding two new shard databases, `p0/orig/p0store-shard0000.sqlite` and
   `-shard0001.sqlite`. Each had the full 37-table schema and one `version` row
   `(1, 'p0-throwaway')`. The moved store's own shards in `p0/moved/` were beside it, unused.
   The mechanism is the one README §0 read from the code. Each stale path goes to a `Datastore`
   actor, which creates the parent directory and an empty database (`Datastore.py:217–222`).
4. Everything under `p0/` was deleted.

**Verdict: README §0's reading of the code was right, and the `run-registry` issue's sentence "a
primary naming a nonexistent shard raises" was wrong.** A moved store opens without complaint.
It resurrects its old directory with empty shards, and every sharded lookup then misses. No §7
condition applies, because P0 matched one of the two accounts.

**What P1 adds over P0.** P0 found no error at all. P1 turns the silent resurrection into a
`RuntimeError` in `ShardedPool`, raised before any actor exists. I measured this after the
change on a second throwaway, run through the real constructor. Created, it recorded
`[(0, 'p0store-shard0000.sqlite'), (1, 'p0store-shard0001.sqlite')]`, which is §6 item 3. Moved,
it opened against `p0after/moved/…` and nothing was recreated at `orig/`. With shard 1 deleted it
raised this:

```
RuntimeError: Cannot open sharded datastore "/private/tmp/…/p0after/moved/p0store.sqlite": shard #1:
stored record "p0store-shard0001.sqlite" resolves to "/private/tmp/…/p0after/moved/p0store-shard0001.sqlite",
which does not exist
```

It left exactly two files, the primary and shard 0. Shard 1 was **not** recreated. That
throwaway was deleted too.

## Where the resolver lives, and why (P4)

It lives in `Datastore/shard_paths.py`, **inside the `Datastore` package and outside
`Datastore.SQL`**. Importing `Datastore.shard_paths` runs `Datastore/__init__.py`, which imports
only `Datastore/object.py` (`typing`). It does not run `Datastore/SQL/__init__.py`, which imports
the `Datastore` actor and with it `ray` and `sqlalchemy`. A test asserts this in a clean
subprocess (`TestModuleIsStandalone`), and so does the audit test
(`test_tool_imports_no_heavy_dependency`, which prints `HEAVY []`). I rejected the alternative,
`tools/shard_paths.py`, because it would make production datastore code import from `tools/`,
which is the wrong direction for a dependency.

A standalone script run as `python tools/shard_key_audit.py` has `tools/` as `sys.path[0]`, not
the repository root. So the tool puts its own repository root
(`Path(__file__).resolve().parents[1]`) on `sys.path` before the import, if it is not already
there. It therefore still runs from any working directory with no `PYTHONPATH`. The tests run it
from an unrelated temporary directory with `PYTHONPATH` removed from the environment, and the
§4 run invoked it from `/`. It still uses only the standard library plus that one
standard-library-only module, and it still opens everything `mode=ro`. The function is defined
once. §7's sixth condition was not met.

## The deliberate-breakage record (§6 item 2)

To get the unfixed state, I restored `Datastore/SQL/ShardedPool.py` and `tools/shard_key_audit.py`
from `HEAD` and moved `Datastore/shard_paths.py` aside. The tests were not touched. I ran
`python -m unittest -v Datastore.tests.test_shardedpool_shard_paths
Datastore.tests.test_shard_key_audit_copy`, then restored the three production files and checked
them with `cmp` against the saved fixed copies. The broken state was never committed.

| Test | Unfixed | Fixed |
|---|---|---|
| **3** `TestLegacyRecords.test_legacy_absolute_records_resolve_to_the_primarys_directory` | **FAIL**: `{0: …/A/store-shard0000.sqlite, …} != {0: …/B/store-shard0000.sqlite, …}` | ok |
| **4** `TestCopiedStore.test_copied_directory_reads_its_own_shards_not_the_originals` | **FAIL**: same shape, *A*'s paths where *B*'s were expected | ok |
| **4** `TestCopiedStore.test_copied_with_a_renamed_primary_reads_its_own_shards` | **FAIL**: same shape | ok |
| **5** `TestFailClosed.test_missing_shard_of_a_new_store_is_refused_by_name` | **ERROR**: `AttributeError: 'ShardedPool' object has no attribute '_check_shard_files'`. No guard exists on the unfixed tree. | ok |
| **5** `TestFailClosed.test_missing_sibling_is_refused_even_when_the_absolute_record_exists` | **FAIL**: `…/A/store-shard0002.sqlite != …/B/store-shard0002.sqlite`. Unfixed code *uses* the original's shard. | ok |
| **5** `TestFailClosed.test_moved_store_with_legacy_records_and_no_shards_is_refused` | **ERROR**: `AttributeError` as above | ok |
| **5** `TestFailClosed.test_record_that_is_not_a_bare_name_is_refused_on_read` | **FAIL**: `RuntimeError not raised` | ok |
| **5** `…test_symlinked_shard_is_refused`, `…test_two_records_resolving_to_one_file_are_refused` | **ERROR**: `AttributeError` | ok |
| 2 `TestRoundTrip.test_new_store_records_bare_names_and_survives_a_move` | FAIL: stored records are absolute, not bare names | ok |
| P4 `test_audit_of_the_copy_attaches_the_copys_shard` | **FAIL**: the unfixed tool reported `'wavenumber' table (shard #0) row count: 3`, which is **the original's** shard (the copy's has 2) | ok |
| P4 `test_audit_does_not_fall_back_to_the_original_when_the_copy_lacks_shard_0` | **FAIL**: the unfixed tool attached the original and reported `row count: 3` | ok |
| 6 (both), 3 `…name_their_own_directory_are_silent`, P4 `…imports_no_heavy_dependency`, 2 `…renaming_the_primary_alone…` | ok, ok, ok, ok, ERROR (`AttributeError`) | ok |

Unfixed: `Ran 17 tests … FAILED (failures=8, errors=5)`. Fixed: `Ran 17 tests … OK`. The
methods that pass on both trees are guards against regression, not detectors. Test 6 is one of
them: the old code did not rewrite rows either, and the test exists so the new code cannot start
to. Test 5's plain missing-shard case is an **ERROR** rather than a FAIL on the unfixed tree,
because the guard it tests does not exist there. The no-fallback case of test 5 *does* FAIL on a
path assertion, which is the unfixed pool picking the original's file.

The first run of this record made me reorder assertions in two test-4 methods. In those methods,
`_check_shard_files()` came before the path assertion, so on the unfixed tree they ERRORed on the
missing method instead of FAILing on *A*'s paths. I moved the path assertion first, and dropped
the check call from test 6, which then passes on both trees as it should. This changed only
tests, and only the order of assertions, before anything was committed.

## The real-store demonstration (§4)

**Before the change, read-only.** I first recorded the state of all three stores with stdlib
`sqlite3` `mode=ro`: mtimes to the nanosecond, sizes, SHA-256, and per-table row counts of every
file. I then copied `var/datastores/handover-atol-sweep.sqlite` and its four shards with `cp -p`
into `var/portability-check/`, as `pcopy.sqlite` and `pcopy-shard000{0..3}.sqlite`, per §4.1.
The copy was not opened through `ShardedPool`. I read what the unfixed `_read_shard_data`
resolves by calling it on an `object.__new__` instance, with no Ray:

```
resolved shard 0: /Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-atol-sweep-shard0000.sqlite
… shards 1–3 likewise: the ORIGINAL's shards
```

The copy's primary hash `c3cda49d8e35…` was the same before and after. The unfixed audit tool on
the copy reported `'wavenumber' table (shard #0) row count: 8` without naming a path. From the
code (`Path(shard_filename)`) and from the P4 tests above, that was the original's shard.

**After the change, the copy exactly as §4.1 specifies (all five files renamed).** The copy
cannot open, and that is correct. Its legacy records name `handover-atol-sweep-shard000N.sqlite`,
and P3 reads a legacy record by its file name, so it looks for
`var/portability-check/handover-atol-sweep-shard000N.sqlite`. Those files are not there, because
§4.1 renamed them. `main.py --database var/portability-check/pcopy.sqlite --inventory
--no-prune-unvalidated --shards 4 --ray-address local` exited 1 with this output:

```
!! Primary database ".../var/portability-check/pcopy.sqlite" records 4 shard(s) by legacy absolute path in
   ".../var/datastores"; reading them as siblings in ".../var/portability-check" instead (stored records not rewritten)
RuntimeError: Cannot open sharded datastore ".../var/portability-check/pcopy.sqlite": shard #0: stored record
   ".../var/datastores/handover-atol-sweep-shard0000.sqlite" resolves to
   ".../var/portability-check/handover-atol-sweep-shard0000.sqlite", which does not exist; shard #1: …
```

The copy's five files were byte-identical afterwards, and so were the originals. This is P1
failing closed on a real store. On the unfixed tree the same command would have opened the
live sweep shards. It is also direct evidence for `[01-whole-store-rename-is-unsupported]`. See
Deviations, item 1, for why the demonstration then continued in a second shape.

**After the change, the copy in the shape §5 says works.** I renamed the copy's four shards back
to the names its records carry (`mv`, inside `var/portability-check/` only). The primary stayed
`pcopy.sqlite`, in the new directory. So the copy is in a new directory, its primary has a new
name, its records are legacy absolute paths into `var/datastores/`, and the originals are
present and populated at exactly those paths. That is the copied-store failure case. The same
`main.py` command then gave this:

```
!! Primary database ".../var/portability-check/pcopy.sqlite" records 4 shard(s) by legacy absolute path in
   ".../var/datastores"; reading them as siblings in ".../var/portability-check" instead (stored records not rewritten)
>> Opened existing sharded datastore ".../var/portability-check/pcopy.sqlite" with 4 shards
== Datastore inventory: var/portability-check/pcopy.sqlite ==   (exit 0)
```

Inventory against the original's shards (`mode=ro`, taken before anything ran; replicated tables
from shard 0, where all four shards agree; sharded tables summed):

| Table | Kind | Original | Inventory of copy |
|---|---|---:|---:|
| version | replicated | 1 | 1 |
| store_tag | replicated | 10 | 10 |
| redshift | replicated | 1,740 | 1,740 |
| wavenumber | replicated | 8 | 8 |
| wavenumber_exit_time | replicated | 8 | 8 |
| tolerance | replicated | 13 | 13 |
| LambdaCDM | replicated | 1 | 1 |
| QCD_Cosmology | replicated | 1 | 1 |
| IntegrationSolver | replicated | 7 | 7 |
| BackgroundModel | replicated | 1 | 1 |
| BackgroundModelValue | replicated | 1,740 | 1,740 |
| GkSourcePolicy | replicated | 2 | 2 |
| QuadSourcePolicy | replicated | 2 | 2 |
| TkNumericIntegration | sharded | 8 | 8 |
| TkNumericValue | sharded | 3,865 | 3,865 |
| TkWKBIntegration | sharded | 8 | 8 |
| TkWKBValue | sharded | 9,627 | 9,627 |
| QuadSource | sharded | 36 | 36 |
| QuadSourceValue | sharded | 17,178 | 17,178 |
| GkNumericIntegration | sharded | 4,549 | 4,549 |
| GkNumericValue | sharded | 146,445 | 146,445 |
| GkWKBIntegration | sharded | 13,920 | 13,920 |
| GkWKBValue | sharded | 916,937 | 916,937 |
| GkSource | sharded | 1,160 | 1,160 |
| GkSourceValue | sharded | 1,016,160 | 1,016,160 |
| GkSourcePolicyData | sharded | 1,160 | 1,160 |
| QuadSourceIntegral | sharded | 7,706 | 7,706 |
| OneLoopIntegral | sharded | 0 | 0 |

**All 28 match.** Matching counts cannot, on their own, show *which* files the actors read,
because the copy's content is the original's. So I ran a discriminating check. With `sqlite3`, I
deleted one `QuadSourceIntegral` row (serial 329386) **from the copy's shard 0 only**, taking it
from 1,955 rows to 1,954. After the rerun the inventory said **`QuadSourceIntegral: 7,705 rows`**,
which is the copy's count, not the original's 7,706. That settles it: the actors opened the copy.

**Audit tool on the copy** (§6 item 5), run from `/` with no `PYTHONPATH`:

```
>> cross-file check against shard #0: .../var/portability-check/handover-atol-sweep-shard0000.sqlite
   (record '.../var/datastores/handover-atol-sweep-shard0000.sqlite')
>> 'wavenumber' table (shard #0) row count: 8
VERDICT: OK -- no inconsistency found in .../var/portability-check/pcopy.sqlite.
```

**The originals were untouched.** After all the runs I took the state again. `handover-atol-sweep*`,
`handover-A3-baseline-lambdacdm*` and the backup's six files were identical in `st_mtime_ns`,
size, SHA-256 and every per-table row count, file by file (`cmp` of the two JSON records: equal).
For the sweep shards:

| File | mtime (before = after) | rows, all tables (before = after) |
|---|---|---:|
| `handover-atol-sweep-shard0000.sqlite` | 2026-09-24T02:58:24 | 596,542 |
| `handover-atol-sweep-shard0001.sqlite` | 2026-09-23T23:00:42 | 593,534 |
| `handover-atol-sweep-shard0002.sqlite` | 2026-09-24T02:58:24 | 590,516 |
| `handover-atol-sweep-shard0003.sqlite` | 2026-09-24T02:58:24 | 599,069 |
| `handover-atol-sweep.sqlite` (primary) | 2026-09-23T12:41:04 | 41 |

The primaries' SHA-256 were unchanged: sweep `c3cda49d8e35…`, and A3 and backup `fdbe93a54108…`.
The last two are the same, because the backup primary is byte-identical to the live one. After
both `main.py` runs the copy's primary still held its four legacy absolute rows, and its hash
was unchanged at `c3cda49d8e35…`. P3 did not rewrite them.

**The copy was deleted** afterwards (`rm -rf var/portability-check`). `var/` now holds only
`bootstrap-a3-resume.log`, `datastores/` and `runs/`, as before. Nothing from §4 is in the
commit.

## What now happens to the backup of README §0.1

I did not open it, and I reason from the tests. The backup is test 4's shape
(`test_copied_directory_reads_its_own_shards_not_the_originals`). Its primary lives in
`backup-pre-resume-20260921T091011/`, its records are absolute paths into `var/datastores/`, the
live shards exist at exactly those paths, and the backup's own shards sit beside its primary
under the same names. The backup's four shards have those names, and the primary's four
records all resolve (checked read-only against the listing). So opening it now resolves to
**the backup's own four shards**. The pool opens against them and prints one `!!` line naming
`var/datastores` as the old location. The live store is not touched, and the backup's `shards`
rows stay as they are (test 6).

Two consequences. **First, the backup can now be opened in place without touching the live
store.** Second, opening it in place is **no longer a read**. It would do to the backup's own
shards whatever opening does to any store. That means a `version` row if the label is new,
pruning of unvalidated rows unless `--no-prune-unvalidated`, and table creation by
`_ensure_tables`. Anyone who wants the backup kept pristine should still copy it before opening
it. The backup was not repaired, re-copied or edited (§5).

## The atol sweep script (left alone)

`docs/handover/quadsource_atol_sweep.py` is unchanged (`git diff -- docs/handover/` is empty).
Its `prepare()` writes absolute sibling paths with an `UPDATE`. P3 reads those as legacy records
and resolves them to the same siblings, so the pool opens the same files as before. Its
`assert_store_is_self_consistent` compares the table against absolute paths. The sweep store it
checks keeps holding absolute paths, because P3 never rewrites them, so the check keeps passing.
**The re-pointing is now redundant, not wrong.** A copied primary would open against its own
shards without it.

## Deviations from the prompt

1. **§4 was completed in a second shape as well as the one specified.** Class:
   **STRUCTURALLY REQUIRED.** §4.1 asks for a copy with **all five files** renamed to a new stem.
   That is the whole-store rename, and §5 says this prompt does not support it. P3's rule
   ("resolve to `primary.parent / Path(stored).name` … never fall back") cannot find a renamed
   shard from a legacy record, because the record carries the old name. So §4.3's inventory
   cannot succeed on a §4.1 copy under the prompt's own design. Supporting it would mean
   deriving shard names from the primary's stem, which is the design choice §5 reserves for the
   user. I ran the §4.1 shape anyway, and recorded that it fails closed, correctly, with the
   originals untouched. I then used the shape §5 names as working: a new directory, the primary
   renamed, and the shards keeping their recorded names. That shape carries everything §4 is
   there to show. Its records are a genuine legacy store in a new place, the original is present
   and populated at the recorded paths, the inventory matches, and the originals are unchanged.
   No production code was changed to make either shape pass. The orchestrator should check this
   item.
2. **`SerialPoolBroker` creation moved below the existing/new-store branch.** Class:
   **IMPLEMENTATION CHOICE.** P1 says to raise "before any actor is created". The broker is an
   actor, and it was created at the old `:79`, before the branch. Neither branch uses it, since
   `_create_engine`, `_write_shard_data` and `_read_shard_data` are pure SQLAlchemy. Moving it
   below the branch makes the requirement literally true. A `ProfileAgent` actor, if
   `--profile-db` is given, is still created by `main.py` before `ShardedPool` exists. That is
   outside this file and was left alone.
3. **The P1 check refuses more than a missing file.** Class: **IMPLEMENTATION CHOICE.** It also
   refuses a shard that is not a regular file, a shard that is a **symlink**, and **two serials
   resolving to one file**. The symlink case answers §7's first stop condition. The resolver
   keeps every path in the primary's directory, but `Datastore.py:201` resolves symlinks, so a
   symlinked sibling would put the file the actor actually opens elsewhere. Refusing it keeps
   "every shard is a file in the primary's directory" true end to end. The creator never makes
   symlinks, and none exists under `var/datastores/` (checked with `ls -la`). The duplicate case
   exists because reading legacy records by name can merge two distinct absolute paths onto one
   sibling. Two actors on one file must never happen.
4. **The resolver refuses a relative `primary` and a few more record forms than §2 lists.**
   Class: **IMPLEMENTATION CHOICE.** The extra forms are a backslash, a NUL, a non-string, and a
   legacy record whose last component is not a bare name. All of them fail closed, as §2 asks
   for "anything else".
5. **`_write_shard_data` checks its own record by resolving it back.** Class: **IMPLEMENTATION
   CHOICE.** If a record does not resolve back to the shard it was written for, the method
   raises. A no-op today, but it ties the writer to the reader.
6. **The audit tool now prints the path it attaches.** Class: **IMPLEMENTATION CHOICE.** Without
   this, §6 item 5 ("reports its cross-file check against the copy's shard 0") could not be
   observed from its output. Its "does not exist" message is reworded to carry the serial and
   the reason from `shard_file_problem`.
7. **Four tests beyond §3's six**: renaming the primary alone, the symlink, the duplicate, and
   the audit tool's three. Class: **IMPLEMENTATION CHOICE.** §3 says "at minimum".

No **UNINTENDED DRIFT**. The diff touches `Datastore/SQL/ShardedPool.py`, `tools/shard_key_audit.py`,
the new `Datastore/shard_paths.py`, `Datastore/tests/`, and the campaign's own board and log, the
`run-registry` board and `docs/OPEN_ISSUES.md`. It does not touch `Datastore/SQL/Datastore.py` or
`docs/handover/`, and it makes no change to the `shards` table's columns.

## Verification performed

1. **§7 condition 2 (other readers and writers of `shards.filename`).** I grepped the tree for
   `FROM shards`, `UPDATE shards`, `"shards"`, `_shard_file_table`, `_shard_db_files` and
   `.filename` in `*.py`, and for `shards` in notebooks, shell and SQL files. The only readers
   and writers are `ShardedPool`, `tools/shard_key_audit.py` and
   `docs/handover/quadsource_atol_sweep.py`. The `extract_*.py` and `docs/…/analyse_*.py`
   scripts locate shard files by a user-supplied glob and never read the table. Not met.
2. **§7 condition 4 (a caller relying on missing shards being created).** `ShardedPool(` is
   called from `main.py`, six `extract_*.py` scripts and two
   `docs/source-remediation-verification/` scripts. None of them deletes or relies on creating
   a shard of an existing store. Every one opens a store that exists whole or creates one new.
   Not met.
3. **§7 condition 1.** Every accepted record resolves to `primary.parent / name`, where `name`
   contains no separator and is not `.` or `..`. The primary is absolute and resolved. Symlinked
   shards are refused. The only other path computation, `relative_to`, is checked by resolving
   back. `test_result_is_absolute_and_in_the_primarys_directory` and `test_refused_records` pin
   this down. Not met.
4. **Suites** at the commit, each run as `PYTHONPATH=. ./venv/bin/python -m unittest discover -s
   <package>/tests -t .`:

   | Suite | Baseline (`71c4c66`) | Now |
   |---|---|---|
   | AdaptiveLevin | 32 OK | 32 OK |
   | ComputeTargets | 552, 1 failure (the known `test_wall_time_per_object` flake) | 552, 1 failure: the same flake, `0.06065687537193298 not less than or equal to 0.06`; the test re-run alone 3 times passed 3 times |
   | CosmologyModels | 39 OK | 39 OK |
   | Datastore | 10 OK | **37 OK** (+27, exactly the methods added) |
   | LiouvilleGreen | 148 OK (skipped=1) | 148 OK (skipped=1) |
   | RunRegistry | 38 OK | 38 OK |

5. `./venv/bin/python -m black --check .` is clean: 284 files would be left unchanged.
6. `python -m RunRegistry list` showed nothing `running` before any store was touched.

## Observations not acted on

1. **The atol sweep's self-consistency check assumes absolute records.**
   `assert_store_is_self_consistent` (`quadsource_atol_sweep.py:589`) compares `shards.filename`
   with `str(p.resolve())` for each expected shard. It would **reject a store created after this
   change**, whose records are bare names, *unless* that store has first been through
   `prepare()`'s `UPDATE`, which rewrites them to absolute. In the script's own workflow,
   `SWEEP_STORE` is only ever made by `prepare()`, so the check never bites. It would bite
   anyone who reused the function on a freshly created store. The file is the record of a
   measurement and was not edited (§2). Opened as
   `[01-atol-sweep-check-expects-absolute-shard-records]`.
2. **Whole-store rename is unsupported** (§5), and §4.1 exercised it. I opened this as
   `[01-whole-store-rename-is-unsupported]` with both options, and decided neither.
3. **`_create_engine` uses the unresolved `_db_name`** (prompt §5). I judged this inert: within
   one constructor call, `_db_name` and `Path(_db_name).resolve()` name the same file, because
   the working directory does not change in between. The only difference is the spelling of the
   SQLAlchemy URL. No issue opened.
4. **Legacy records are never migrated**, by design (§2 P3 and §5). A relocated legacy store
   prints the `!!` line every time it is opened, until someone rewrites its rows, which this
   prompt forbids. That is noise, not a defect. No issue opened.

## State handed to the next prompt

There is no next prompt, because the campaign has one. `[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]`
is closed on the `run-registry` board's §4. Two issues are open on this campaign's board §3,
`[01-whole-store-rename-is-unsupported]` (the user's design decision) and
`[01-atol-sweep-check-expects-absolute-shard-records]`. Stores created from now on record bare
shard names. The two existing stores and the backup still hold absolute records, which are read
as siblings. Their rows, mtimes, hashes and row counts are unchanged by this prompt.
