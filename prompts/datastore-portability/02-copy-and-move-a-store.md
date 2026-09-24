# Prompt 02 — copy and move a whole store, through one interface on `ShardedPool`

**Campaign:** [`README.md`](README.md) · **Board items:** **P5**–**P8** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[01-whole-store-rename-is-unsupported]` (this board's §3; `docs/OPEN_ISSUES.md` §1.12).
**Does not touch:** `[store-sidecar-manifests-have-no-owner]`, which is prompt 03's and is held.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. The code is short. The judgement is in the order of the file
operations, and in showing that a store interrupted at any point either opens correctly or is
refused, never opened wrongly.

**Read first:**

1. [`README.md`](README.md): §0 in full, §1 including the 2026-09-24 amendment, and **§6 in
   full**. §6 records the user's decisions this prompt implements. Do not reopen them.
2. Prompt 01 and its log, [`logs/01-relative-shard-paths.md`](logs/01-relative-shard-paths.md),
   especially "Deviations" item 1 and "The real-store demonstration". This prompt does what
   prompt 01 §4.1 intended.
3. `Datastore/shard_paths.py`, the one resolver, in full.
4. `Datastore/SQL/ShardedPool.py`: the constructor's new-store branch (the shard naming at
   `:101–111`), `_create_engine`, `_write_shard_data`, `_read_shard_data` and
   `_check_shard_files`.
5. `Datastore/SQL/Datastore.py` `:170–230`: what a `Datastore` actor does with `db_name`. **You do
   not change this file.**
6. `Datastore/tests/shard_store_fixtures.py` and prompt 01's three test modules: the no-Ray pattern
   you extend.
7. `tools/shard_key_audit.py`: the precedent for a standalone script in `tools/`, including how it
   puts the repository root on `sys.path`.
8. `CLAUDE.md`, the repository mechanics section.

---

## 1. What is wanted

Since prompt 01, a store's `shards` table holds bare file names, or legacy absolute paths that are
read by their file name. Moving a store's directory, or renaming its primary alone, already works.
Renaming the shards as well needs one more step: writing the new names into the table. README §6.1
records that this is all it takes, and §6.2 records where it goes.

The user's decisions, which this prompt implements and does not revisit:

- **An interface on `ShardedPool`**, static, operating on a **closed** store. It is not a method of
  an open pool: an open pool has one `Datastore` actor per shard holding its file.
- **Client code decides how to use it.** The first client is a small standalone script in `tools/`.
- **Nothing but the primary and its shards.** `ShardedPool` and the script do not copy, move,
  refuse because of, or mention any other file, including a `<stem>.manifest.json` sidecar
  (README §6.3). That belongs to a registry tool, prompt 03, which is held.
- **The `shards` table stays the authority.** Shard names are **not** derived from the primary's
  stem at read time (prompt 01 §5 option (b) is not taken).

---

## 2. What to change

**P5 — one shard naming rule.** The constructor names shard *i* of a new store
`primary.with_stem(f"{stem}-shard{i:04d}")` inline (`:101–111`). Move that rule into
`Datastore/shard_paths.py` as a small pure function, for example
`shard_file_name(primary: Path, serial: int) -> str`, returning the bare name, with the primary's
suffix. Have the constructor call it. The copy/move interface uses the same function to name the
destination's shards. **Do not write the pattern a second time**: two definitions of what a shard
is called is the same kind of disagreement prompt 01 removed for where a shard lives. New stores
must be named exactly as before; a test pins this. `Datastore/tests/shard_store_fixtures.py`
`write_new_store` (`:67`) repeats the pattern too. Make it call the function, so the fixtures
cannot drift from the creator. Test modules may still spell out expected names as literals: those
are the expectations the function is tested against. `docs/handover/quadsource_atol_sweep.py`
holds two more copies (`:583`, `:711`). Leave them alone, since that script is a measurement
record, and list them under "Observations not acted on".

Likewise, **reading and checking a source store's shard records must reuse prompt 01's code**,
not a second copy of it: the resolver, `shard_file_problem`, and the duplicate-file check in
`_check_shard_files`. `_read_shard_data` also validates the shard-key type and the table lists
against constructor arguments that a copy does not have. So you will probably need to factor out
the part that reads the `shards` table and checks its files, so that both the constructor and the
interface call it. Where that factored code lives is your choice. Justify it in the log. The
constructor's behaviour and its error messages must not change: prompt 01's tests must pass
unmodified.

**P6 — the interface.** Two static methods on `ShardedPool`, sharing one implementation:

```python
ShardedPool.copy_store(src: PathType, dst: PathType) -> Dict[int, Path]
ShardedPool.move_store(src: PathType, dst: PathType) -> Dict[int, Path]
```

`src` and `dst` are **primary** paths. Each method returns the destination's serial → shard path
map, as the resolver reads it back from the finished destination. Neither starts Ray, creates an
actor or needs a `ShardedPool` instance. Opening a source for reading is always `mode=ro`.

*Before anything is written*, refuse with a `RuntimeError` that names the file and the reason when:

- the source primary does not exist or is not a regular file;
- any source shard record is unusable, or resolves to a missing file, a symlink, a non-regular file
  or a file shared by two serials. This is prompt 01's check, reused;
- any **source** primary or shard has a `-journal`, `-wal` or `-shm` file beside it. A hot rollback
  journal means the store was not closed cleanly, and a copy made without it is corrupt;
- `dst` resolves to the same file as `src`;
- `dst` is an existing directory. `dst` names the new primary; it is never a directory to put the
  store in;
- the destination primary, or any destination shard name from P5, **or any of their `-journal`,
  `-wal` or `-shm` names, already exists**. A stale journal at a destination name would be replayed
  into the new file by the next opener. **Nothing is ever overwritten.**

The destination's parent directory is created if it is missing, as the constructor does (`:99`).

*The write.* The destination's shard *serial* is named `shard_file_name(dst, serial)` for every
serial in the source's table. The serials are those in the table, not `range(n)`. Its `shards`
table then holds exactly those bare names, and **no other table, in the primary or in any shard,
is changed**. Rewriting a legacy source's absolute rows to bare names at the destination is
intended. It is an explicit operation the caller asked for, not a side effect of opening, so it
does not conflict with prompt 01's "opening changes nothing".

*The order of operations* is the substance of this prompt. Use this order, or a different one only
if you can show it is at least as strong, and log why:

- **copy:** copy each shard to its destination name; copy the primary to a **temporary name** in
  the destination directory; rewrite that temporary primary's `shards` rows in one transaction;
  `os.replace` it to the destination primary's name. Use `shutil.copy2`. The source is opened only
  `mode=ro`, and never written.
- **move:** rename each shard to its destination name; rename the primary to its destination name;
  rewrite the destination primary's `shards` rows in one transaction. Use `os.rename`. A move across
  filesystems fails at the first rename, before anything has moved. Say in the error that the
  remedy is to copy and then delete the source by hand.

**The property you must establish**, and record in the log as a table with one row per point of
interruption: *if the process dies after any step, every store a later opener could find, at the
source or at the destination, either opens against the right files or is refused by prompt 01's
check or by the constructor's existing "primary is missing, but shard … already exists" guard
(`:106`).* No state may open against a mix of files, against the wrong store's files, or as an
empty new store. Work through at least these cases: same directory with a new stem; new directory
with the same stem; and new directory with a new stem. Do each for a new-style source and for a
legacy source whose absolute records name **another directory that exists and is populated**.

*On failure* the methods do not clean up. They raise, and the message lists the files that exist
at the destination and at the source, and which step failed. Deleting is for a person. **Neither
method ever deletes a file.**

**P7 — the script.** `tools/sharded_store.py`, run as
`./venv/bin/python tools/sharded_store.py {copy,move} SRC DST`. It calls P6 and nothing else. On
success it prints the destination's shard map; on refusal it prints the reason and exits non-zero.
Like `tools/shard_key_audit.py`, it puts its own repository root on `sys.path`, so it runs from any
directory with no `PYTHONPATH`. It imports `ShardedPool`, and so `ray`, but it **never initialises
Ray**; a test checks this. Its module docstring and `--help` say three things:

- it handles the primary and its shards and nothing else. In particular it leaves any
  `<stem>.manifest.json` where it is;
- it cannot tell whether a process has the store open, because these stores use SQLite's rollback
  journal, which leaves no file while idle. Checking that nothing is `running` against the store is
  the caller's job;
- a registry-level tool is the place that does both (README §6.3–§6.4).

It does not consult the registry itself.

---

## 3. Tests — in `Datastore/tests/`, no Ray, no datastore

Build stores with `shard_store_fixtures.py` (extend it if needed) in temporary directories. At
minimum:

1. **Naming rule.** `shard_file_name` reproduces the constructor's old names exactly, for several
   stems, serials and suffixes. A new store written through the fixtures is named by it.
2. **Copy.** New directory, new stem. The destination's `shards` rows are the new bare names.
   `_read_shard_data` + `_check_shard_files` on the destination, through the `object.__new__`
   pattern, return the destination's files, carrying the destination's own contents, not the
   source's. **All of the source's files are byte-identical and have unchanged mtimes afterwards.**
3. **Copy of a legacy source.** Absolute records naming a populated directory *A*, files in *B*,
   copied to *C* under a new stem. *C* reads *C*, rows are bare, and *A* and *B* are unchanged.
   This is the prompt 01 §4.1 case.
4. **Move.** Each of: same directory with a new stem; new directory with the same stem; new
   directory with a new stem. The source names are gone and the destination reads its own files.
5. **Refusals.** One test per refusal in P6. Each asserts the error names the offending file, and
   that **no file was created, changed or removed** anywhere under the temporary root. Compare a
   full listing with hashes and mtimes, before and after.
6. **Interruption.** For each step of each operation, make the step after it fail (monkeypatch
   `shutil.copy2`, `os.rename` or `os.replace` to raise on the *n*th call, or the rewrite to raise),
   then assert what the log's table says. Either the reachable store opens against the right files
   through `_read_shard_data` + `_check_shard_files`, or that check raises, or the destination has
   shards and no primary (the constructor's `:106` guard; assert the file state and cite it). Cover
   at least one legacy source.
7. **Nothing else is touched.** A `<stem>.manifest.json` and an unrelated file beside the source are
   still there, unchanged, after a copy and after a move, and nothing of that name appears at the
   destination.
8. **The script.** Run as a subprocess from a directory other than the repository root, with no
   `PYTHONPATH`. It copies successfully; it refuses with a non-zero exit and no writes; and `ray`
   is imported but `ray.is_initialized()` is false at exit. Check the last one however is
   cleanest, and say how in the log.

**Deliberate breakage.** The interface is new, so "fails on the old tree" proves nothing here.
Instead, show that each of these mutations makes the tests written against it fail, then restore
it:

- (i) skip the `shards` rewrite;
- (ii) write the rows as absolute paths instead of bare names;
- (iii) drop the source-journal check;
- (iv) drop the destination-journal check;
- (v) in copy, write the primary directly to its final name instead of via the temporary name;
- (vi) in the P5 function, change the zero-padding.

Put each mutation in the log as a short diff, **exactly as applied**, so the orchestrator can
replay one with `git apply`. Name the tests that failed. Mutations are never committed.

---

## 4. The real-store demonstration

End to end, through the real script and then the real constructor via `main.py`. **The script is
never pointed at an original** (README §3). All work goes in a new directory under `var/`, for
example `var/portability-check-02/`, and all of it is deleted at the end.

1. `python -m RunRegistry list`: nothing `running`. Snapshot the three stores read-only with
   `sqlite3` `mode=ro`: mtimes, sizes, per-table row counts per shard, and the three primaries'
   SHA-256.
2. **Make the hand-made source.** `cp -p` `var/datastores/handover-atol-sweep.sqlite` and its four
   shards into `var/portability-check-02/src/`, keeping their names. Its primary still names the
   originals' shards by absolute path: a legacy record naming another store that exists. Then, with
   `sqlite3` **on the hand-made copy's shard 0 only**, delete one row from one sharded table, so the
   hand-made source is distinguishable from the original by exactly one row. Record which row, and
   the table's count before and after.
3. **Copy with the script** to `var/portability-check-02/dst/pcopy.sqlite`. Show:
   - the destination's `shards` rows: four bare `pcopy-shard000N.sqlite` names;
   - the hand-made source's five files, byte-identical and with unchanged mtimes;
   - the destination directory holding exactly five files;
   - the originals, unchanged.
4. **Run `tools/shard_key_audit.py` on `pcopy.sqlite`.** It must attach `pcopy-shard0000.sqlite`.
5. **Open `pcopy.sqlite` through `main.py --database <it> --inventory --no-prune-unvalidated`**,
   run as a script. The per-table counts must equal the **original's**, except the edited table,
   which must be one lower. That row is how you know the inventory read the copy's files and not
   the originals'.
6. **Move with the script** to `var/portability-check-02/moved/pmoved.sqlite`: new directory, new
   stem. Show the `shards` rows, and that `dst/` is empty. Open it through `main.py --inventory`
   again: the same counts as step 5.
7. **Show the originals untouched.** Re-take step 1's snapshot; it must be identical.
8. **Delete `var/portability-check-02/`.** Nothing from this section is committed. Throwaway scripts
   go in your scratch space.

---

## 5. Acceptance

1. §3's tests exist in `Datastore/tests/` and need no Ray and no datastore. Prompt 01's 27 tests
   pass unmodified.
2. The deliberate-breakage record, mutations (i)–(vi), each with its diff and the tests that
   failed.
3. The interruption table in the log, one row per point of interruption, each row backed by a test.
4. §4 done. Quote the counts, including the one-row discriminator, and the before/after snapshot.
   The working directory is deleted.
5. The creator and the interface name shards through one function. The shard record reading and
   checking exist once.
6. The script runs standalone from any directory, never initialises Ray, and its `--help` carries
   the three statements in P7.
7. Every existing suite unchanged. `Datastore/tests` rises by exactly the tests you add.
8. `black --check` clean. On the board: `[01-whole-store-rename-is-unsupported]` moved to §4 with
   a closing note naming this prompt and its commit, and the §1 row, the P5–P8 items and §5
   baselines updated. `docs/OPEN_ISSUES.md` updated: its §1.12 row deleted, count and date
   corrected. All in the same commit.

---

## 6. What this prompt does not do

- It does not move, copy, read or mention any sidecar file. It does not consult the registry. It
  does not write any registry code (prompt 03, held).
- It does not derive shard names from the primary's stem at read time.
- It does not rewrite the rows of any store **on open**. Opening still changes nothing.
- It does not change `Datastore.py`, `main.py`, `docs/handover/`, the object factories, or the
  columns of the `shards` table.
- It does not delete files, clean up after a failed operation, or repair any existing store.
- It does not touch `[01-atol-sweep-check-expects-absolute-shard-records]`.

---

## 7. Stop conditions — stop and ask the user

- Anything in a primary other than `shards.filename`, or **anything inside a shard**, records a
  store's path, name or stem. Then a rename would leave a stale reference that this prompt's rewrite
  does not reach. Check the schema of every table in both kinds of file, and say in the log what
  you checked.
- No order of operations satisfies the property in P6. That is, some interruption point leaves a
  state that opens wrongly and cannot be made to be refused within scope.
- Factoring out prompt 01's read-and-check code would change the constructor's behaviour or its
  error messages, or would need prompt 01's tests to change.
- The script cannot run without initialising Ray, or cannot be standalone without copying code.
- Any step would point the script, `ShardedPool` or `main.py` at an original store or the backup,
  or write to either.
- You find yourself wanting to handle, warn about or refuse because of a sidecar, or to consult the
  registry. That is prompt 03, and the user has decided it is not this layer's job (README §6.3).

---

## 8. The log and the board

`logs/02-copy-and-move-a-store.md`, using the template in README §5.1. In addition:

- where the P5 naming function and the factored read-and-check code live, and why;
- the schema check behind the first stop condition;
- the interruption table;
- the deliberate-breakage record, with diffs;
- the §4 demonstration, with its numbers.

`IMPLEMENTATION_STATE.md`: the §1 row for 02, items P5–P8, the rename issue moved from §3 to §4
with a closing note, and §5's baselines. Leave prompt 03's held row and
`[store-sidecar-manifests-have-no-owner]` as they are. `docs/OPEN_ISSUES.md` in the same commit.
