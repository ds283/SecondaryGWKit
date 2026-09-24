# Prompt 01 — shard paths relative to the primary, and fail closed when a shard is missing

**Campaign:** [`README.md`](README.md) · **Board items:** **P0**–**P4** ·
**Board:** `IMPLEMENTATION_STATE.md` — **does not exist yet; this prompt creates it** (§8).
**Closes:** `[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]`, which is on the
[`run-registry`](../run-registry/IMPLEMENTATION_STATE.md) board §3 and indexed at
`docs/OPEN_ISSUES.md` §1.11.
**Opens:** anything out of scope that you find (§5), **without fixing it**.
**Recommended model:** **Opus**. The diff is small. The judgement is in how existing absolute
records are read, and in showing that no path can quietly fall back to someone else's shards.

**Read first:**

1. [`README.md`](README.md) **§0 in full**: the two silent failure modes, and §0.1, the backup that
   cannot be opened in place.
2. `Datastore/SQL/ShardedPool.py` lines `:22–197` (the constructor), `_create_engine` (`:216`),
   `_write_shard_data` (`:270`) and `_read_shard_data` (`:314`). Note which path is resolved
   (`_primary_file`, `:71`) and which is not (`_db_name`, used by `_create_engine` at `:222`).
3. `Datastore/SQL/Datastore.py` `:201–227`: the create-when-missing branch that makes a stale
   shard path silent. **You do not change this file.**
4. `tools/shard_key_audit.py`, especially `:136–146`, the second reader of the `shards` table. Note
   its module docstring: it uses only stdlib `sqlite3` and opens everything `mode=ro`.
5. `docs/handover/quadsource_atol_sweep.py`: `assert_store_is_self_consistent` (`:589`) and
   `prepare()` (`:621`). This is the workaround written after the 2026-09-23 incident, and the
   third reader, and only other writer, of `shards.filename`. **You do not change it** (§2).
6. The `run-registry` board's §3 entry for the issue this prompt closes.
7. `Datastore/tests/test_backgroundmodelvalue_roundtrip.py`, the precedent for a test that builds
   real SQLAlchemy tables in SQLite with no Ray and no datastore.
8. `CLAUDE.md`, the repository mechanics section: tests live in `<package>/tests/`, run from the
   repository root, and **must not need Ray or a datastore**.

---

## 1. The defect, stated precisely

- `_write_shard_data` stores `str(shard_file)` for each shard, where `shard_file` is
  `Path(db_name).resolve().with_stem(f"{stem}-shard{i:04d}")`. The stored value is absolute and
  has symlinks resolved.
- `_read_shard_data` builds `Path(row.filename)` and uses it unchanged.
- `ShardedPool` never issues an `UPDATE` on `shards`. The atol sweep's `prepare()` does, as a
  workaround.
- The constructor passes each path to a `Datastore` actor without checking that the file exists.

Consequences: copy a store and the copy runs against the original's shards. This happened on
2026-09-23. Move or rename one and, according to which account in README §0 is right, it either
raises or opens with empty shards at the old location.

---

## 2. What to change

**P0 — measure first, change nothing.** Settle the disagreement in README §0. On the unfixed tree,
in a directory in your scratch space (never under `var/datastores/`), create a small throwaway store
through the real constructor. Then move it, open it again, and record exactly what happens: an
exception (quote it and say where it is raised), or a pool that opens. If it opens, record which
files it created and at which paths. Then delete everything. This is the "before" for P1. It also
tells the `run-registry` board whether its sentence "a primary naming a nonexistent shard raises"
was right, and that board's §4 entry, when you close the issue, must say which (§8).

**P1 — fail closed on a missing shard.** When opening an **existing** pool (the `else` branch at
`:115`), raise a `RuntimeError` before any actor is created if any resolved shard file does not
exist. The message names the primary, the shard serial, the stored value and the path it resolved
to. Put the check in a method, not inline in `__init__`, so that a test can call it without Ray. If
P0 found that the tree already raises, P1 still ships. An error raised deep inside an actor, after
other actors have opened, is not the same as a guard in `ShardedPool` that runs before any of them
exist. Say in the log what P1 adds over what P0 found.

**P2 — write relative.** `_write_shard_data` stores each shard's path **relative to the primary's
directory**. Every shard is created as a sibling of the primary, so today that is the bare file
name, `foo-shard0000.sqlite`.

**P3 — read both forms through one resolver.** Write a small pure function, for example
`resolve_shard_path(primary: Path, stored: str) -> Path`, and route every read of `shards.filename`
through it. It returns an **absolute** path. That is required, not cosmetic: the result is passed to
Ray actors, whose working directory is not the driver's.

- **Relative (new) records:** `primary.parent / stored`. Refuse any value that is not a bare file
  name: it contains a path separator, is `.` or `..`, or is empty. The creator only ever writes
  siblings, so anything else means something other than this code wrote the row. Fail closed.
- **Absolute (legacy) records:** resolve to `primary.parent / Path(stored).name`. **Never use the
  absolute path itself, and never fall back to it**, not even when the sibling is missing and the
  absolute path exists. That fallback *is* the copied-store failure in README §0, reintroduced. If
  the sibling is missing, P1 raises.
- When a legacy absolute record resolves somewhere other than its literal value, print one line
  per store (not per shard) saying so, in the style of the existing `>>` / `!!` messages.
- **Do not rewrite legacy rows on open.** Opening a store must not modify its `shards` table as a
  side effect: the backup in README §0.1, among others, must stay byte-identical when read.
  Interpreting old rows at read time costs nothing and changes no data.

**P4 — the audit tool uses the same resolver.** `tools/shard_key_audit.py` resolves the shard path
it attaches in exactly the same way. Keep that tool read-only and runnable as a standalone script.
If importing the resolver from its home would pull `Datastore/SQL/__init__.py`, and with it `ray`,
into the tool, then put the resolver in a module both callers can import without that. Put that
module outside the `Datastore.SQL` package, or make it importable on its own, and justify the
choice in the log. **Do not copy the function into two places.** Two definitions of where a shard
lives is the kind of disagreement this prompt exists to remove.

**The atol sweep script is left alone.** `prepare()` writes absolute sibling paths, which P3 reads
as legacy records and resolves to those same siblings. `assert_store_is_self_consistent` compares
against absolute paths, and the sweep store it checks keeps holding absolute paths, because P3
never rewrites them. So the script keeps working unchanged. Its re-pointing becomes redundant, not
wrong. Say so in the log, and state in "Observations not acted on" that its check would reject a
store created after this change. Do not edit it: it is the record of a measurement.

---

## 3. Tests — in `Datastore/tests/`, no Ray, no datastore

`_create_engine`, `_write_shard_data` and `_read_shard_data` touch only a SQLAlchemy engine and a
few attributes, so they can be exercised on an instance made with `object.__new__(ShardedPool)`
and a temporary directory. The constructor, which starts actors, cannot be exercised this way, and
must not be. At minimum:

1. **The resolver, as a pure function:** bare name → sibling; legacy absolute → sibling by name,
   including when the literal absolute path exists and differs; each refused form raises.
2. **Round trip:** write the table for a new store in a temporary directory, read it back, and get
   the same absolute sibling paths. Then **move the directory** (primary and shards together), read
   again, and get the new paths.
3. **Legacy compatibility:** a `shards` table populated with absolute paths pointing into
   directory *A*, with the primary and shard files actually in directory *B*, resolves to *B*.
   Build the table by hand, not by calling `_write_shard_data`. Its whole point is to stand in for
   rows the old code wrote.
4. **The copy case:** with *A* still present and populated, and the store copied to *B*, the paths
   read from *B*'s primary are *B*'s. This is the test for the failure mode that actually bit the
   backup.
5. **Fail closed:** with one shard file deleted, the P1 check raises and names it.
6. **No side effect:** reading a legacy store leaves the primary file's bytes unchanged.

---

## 4. The real-store demonstration

Show the change working end to end, through the real constructor and Ray, **on a copy and never
on an original**.

1. Copy `var/datastores/handover-atol-sweep.sqlite` and its four shards into a **new directory**
   under `var/`, renaming all five files to a **new stem** consistently (`bar.sqlite`,
   `bar-shard0000.sqlite`, …). Leave the originals alone. The copy's primary still holds the old
   absolute paths, so it is a genuine legacy store in a new place.
2. Before the change, **do not open the copy through `ShardedPool`**. On the unfixed tree that
   would write to the original's shards. Show the defect read-only instead: print what
   `_read_shard_data` resolves on the copy, which is the original's paths.
3. After the change, open the copy through `main.py --database <copy> --inventory
   --no-prune-unvalidated` (run it as a script; it cannot be imported). The inventory must match
   per-table row counts taken from the **original's** shards with `sqlite3` in `mode=ro`.
4. Show the originals untouched: mtimes and per-shard row counts of `handover-atol-sweep-shard*`
   the same before and after, and the `shards` table of the copy's primary still holding its
   legacy absolute rows (P3, no rewrite).
5. Delete the copy. Nothing from this step is committed. A throwaway script belongs in your scratch
   directory.

---

## 5. What this prompt does not do

- **It does not support renaming a whole store** (primary and shards together). Once P2 lands,
  moving a store's directory and renaming its primary on its own both work. Renaming the shards too
  breaks the stored names, and supporting that needs either a small rename tool that rewrites the
  `shards` rows or a change to derive shard names from the primary's stem. That is a design choice
  for the user. Open it as a §3 issue with both options; do not pick one.
- It does not change `Datastore.py`, its create-when-missing behaviour, or any object factory.
- It does not rewrite the rows of any existing store, or migrate anything.
- It does not change the schema of the `shards` table.
- It does not repair or re-copy the backup in README §0.1. Once this lands, the backup opens
  against its own shards. Say so in the log, and leave it untouched.
- It does not tidy `_create_engine`'s use of the unresolved `_db_name`. If you judge it matters,
  that is an observation.

---

## 6. Acceptance

1. §3's tests exist in `Datastore/tests/` and need no Ray and no datastore.
2. **Deliberate-breakage record, both directions.** Tests 3 and 4 (legacy and copy) and test 5
   (fail closed) **fail on the unfixed code** and pass on the fixed code. Show both, naming each
   test. A test that has not been seen to fail on the bug it was written for is not known to work.
3. A new store's `shards` table holds bare file names. Show one.
4. §4's demonstration done, with the inventory counts and the before/after mtimes and row counts
   quoted, and the copy deleted afterwards.
5. `tools/shard_key_audit.py`, run against the copy before it is deleted, reports its cross-file
   check against the copy's shard 0, not the original's.
6. P0 done and recorded, with the throwaway store deleted.
7. Every existing suite unchanged. `Datastore/tests` rises by exactly the tests you add.
8. `black --check` clean. Board created, the `run-registry` issue moved to that board's §4, and
   `docs/OPEN_ISSUES.md` updated, all in the same commit.

---

## 7. Stop conditions — stop and ask the user

- Any shard path resolved on any path through the code can end up at a location outside the
  primary's directory.
- Something in the tree other than `ShardedPool`, `tools/shard_key_audit.py` and
  `docs/handover/quadsource_atol_sweep.py` reads `shards.filename`, or writes it.
- P0 shows behaviour that neither account in README §0 describes.
- Making the constructor fail closed would break an existing caller that relies on missing shards
  being created. That would mean some workflow depends on the silent behaviour, and the user should
  hear about it.
- Any step would open an original store or the backup, or write to either.
- The resolver cannot be shared with the audit tool without either making the tool non-standalone
  or copying the function. That is a design question, not one to work around.

---

## 8. The log and the board

`logs/01-relative-shard-paths.md`, using the template in README §5.1. In addition to it:

- the P0 measurement: what a moved store does on the unfixed tree, quoted;
- the deliberate-breakage record of §6 item 2;
- where the resolver lives and why (P4);
- the §4 demonstration, with its numbers;
- a statement of what now happens to the backup in README §0.1 when it is opened. Do not open it
  to find out; reason from the tests.

`IMPLEMENTATION_STATE.md`: create it with a §1 prompt table, a §2 item table (P0–P4), §3 Active and
unresolved issues (at least the whole-store rename issue of §5), §4 Resolved issues, and the
maintenance-rule blockquote. `prompts/datastore-readback/IMPLEMENTATION_STATE.md` is the model.

**Closing the `run-registry` issue.** Move its entry from that board's §3 to its §4, as the owning
board, with a closing note naming this campaign and commit. The note must say what P0 found about
the sentence "a nonexistent shard raises", and add the backup of README §0.1 as a second instance.
Update the `run-registry` board's header counts to match.

`docs/OPEN_ISSUES.md`, in the same commit: delete the issue's row from §1.11; add this board to the
**Boards** line; add a §1 subsection for this campaign with every issue you open; and correct the
count and the date.
