# Log 01 — Fix shard-key persistence (B1 + B5)

**Prompt:** prompts/backport-modules/01-shard-key-persistence.md
**Commit:** `2610abe` — Fix shard-key persistence in ShardedPool
(corrected 2026-09-04, by prompt 04's housekeeping pass — the SHA originally recorded here,
`fbc3a90`, was one amend behind the branch tip; see the `[commit-sha-links-stale]` entry, now
resolved, in `IMPLEMENTATION_STATE.md` §4)
**Date:** 2026-09-03
**Result:** COMPLETE

## What shipped

`Datastore/SQL/ShardedPool.py`:

- `_assign_shard_keys` collection loop (was `ShardedPool.py:772-781`): added a local
  `seen_store_ids` set. The append condition changed from `item.store_id not in
  self._shard_keys` to `item.store_id not in self._shard_keys and item.store_id not in
  seen_store_ids`, with `seen_store_ids.add(item.store_id)` alongside each append. This
  is the B5 fix — a shard-key object appearing twice in one `obj` list previously
  produced two entries in `missing_keys` for the same `store_id`.
- `_assign_shard_keys` write loop (was `ShardedPool.py:799-803`): the INSERT dict key
  changed from `"key_id"` to `"key_serial"`, matching the table's actual primary key
  column (`ShardedPool.py:248` defines `shard_keys.key_serial`). This is the B1 fix —
  previously SQLAlchemy silently dropped the unrecognised `key_id` key, the INSERT
  bound only `shard_id`, and SQLite autoassigned `key_serial` from its rowid counter.
  Added `result = conn.execute(...)`, `assigned_serial =
  result.inserted_primary_key[0]`, and a `print("!! _assign_shard_keys MISMATCH: ...")`
  warning when `assigned_serial != item.store_id` — the ongoing detector the prompt
  asked for. The existing commented-out `# print(f">> assigned shard #...")` diagnostic
  was left exactly as it was (see Deviations, implementation choice, below).

`tools/shard_key_audit.py` (new file): a read-only, standalone auditor.

- Opens the primary database with `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`.
  Verified empirically (see Verification) that a connection opened this way propagates
  URI-mode to subsequent `ATTACH DATABASE 'file:...?mode=ro'` statements, and that a
  write attempt against an attached read-only database raises
  `sqlite3.OperationalError: attempt to write a readonly database`. The script never
  issues a write statement of any kind regardless.
- Schema-generation detection: reads `PRAGMA table_info(shard_keys)`. If `key_serial`
  is absent and `wavenumber_serial` is present, reports the pre-`a2bd966` schema by
  name and exits 2 without attempting further checks. Any other missing-`key_serial`
  shape is reported as "unrecognised schema" and also exits 2.
- For a current-schema store: reports `shard_keys` row count and per-shard
  distribution (single-file, always available); checks `shard_id` referential
  integrity against `shards.serial` and duplicate `key_serial` values (single-file);
  then attempts a cross-file check by reading `shard_key_config.key_type` (e.g.
  `"wavenumber"`), taking the first row of `shards` for a shard file path, and
  `ATTACH`-ing it read-only to compare `shard_keys.key_serial` against that shard's
  copy of the key table's `serial` column. Reports **orphaned** keys
  (`shard_keys` entries with no corresponding key-table row — a real problem, added to
  `problems`) and **unassigned** keys (key-table rows with no `shard_keys` entry —
  informational only, not flagged as a problem, per the prompt).
- Exit codes: `2` for "not a ShardedPool primary DB", "pre-refactor schema", or
  "unrecognised schema" (none of these are audited); `1` for a current-schema store
  with a detected inconsistency; `0` for a clean current-schema store. Prints a
  one-line `VERDICT: ...` on every current-schema path.
- Docstring and inline comments state plainly that there is no automated repair path
  and that a flagged datastore must be rebuilt.

## Deviations from the prompt

### Implementation choice — exit code for "cannot audit" cases

The prompt specifies exit-code behaviour only for a current-schema datastore ("exits
non-zero if any inconsistency is found, zero otherwise"). It does not specify an exit
code for the pre-refactor-schema case beyond "exit[s] without pretending to audit it".
I chose exit code `2`, distinct from the `1` used for a genuine detected inconsistency
in a current-schema store, so a caller can tell "this store cannot be interpreted at
all" apart from "this store was interpreted and found broken." Both are non-zero, so
this does not conflict with anything the prompt states outright.

### Implementation choice — left the commented-out diagnostic untouched

The prompt offered a choice ("Leave the existing commented-out diagnostic alone, or
replace it with upstream's commented-out equivalent — your choice, but say which in
the log"). I left `# print(f">> assigned shard #{new_shard} to key object
#{item.store_id}")` exactly as it was. It already says the same thing upstream's
equivalent says; changing it would be a no-op edit with no benefit, and touching a
disabled line invites an unrelated diff hunk.

### Structurally required — cross-file check needed a working ATTACH pattern, not just a plan

The prompt flags this as a risk up front ("If a clean single-file audit is not
possible, say so in the log..."). It was possible, but only after confirming
empirically (not just by reading SQLAlchemy/sqlite3 docs) that a `sqlite3.connect(...,
uri=True)` connection permits a subsequent URI-style `ATTACH DATABASE 'file:...'`
statement and that the attached database is genuinely write-protected. I built two
throwaway two-file sqlite databases in the scratch directory to check both properties
before writing the real script against that assumption (see Verification). No fallback
to single-file-only checks was needed.

### Unintended drift

None noticed.

## Verification performed

Static checks (all run, all passed):

1. `grep -n "key_id" Datastore/SQL/ShardedPool.py` — no output (confirmed no
   remaining reference).
2. `python3 -c "import ast; ast.parse(...)"` against both `ShardedPool.py` and
   `tools/shard_key_audit.py` — both parse.
3. `black --check Datastore/SQL/ShardedPool.py tools/shard_key_audit.py` — the new
   tool needed one `black` reformatting pass (applied); both files are `black`-clean
   after that.
4. `python3 tools/shard_key_audit.py test-qcd-db.sqlite` — exit code 2, correctly
   identifies the pre-`a2bd966` schema (`wavenumber_serial` PK column) and reports
   that the store must be rebuilt, without attempting to read further. This is the
   negative path the prompt says is exercisable today, and it was exercised.

Additional verification beyond what the prompt required, to build confidence in the
new tool before shipping it (all done against throwaway databases in the session
scratch directory, never against anything in the repository):

5. Confirmed the URI-attach mechanism directly: opened a database with
   `sqlite3.connect("file:...?mode=ro", uri=True)`, attached a second file with
   `ATTACH DATABASE 'file:...?mode=ro' AS other`, successfully read from it, and
   confirmed `INSERT INTO other.t ...` raises `OperationalError: attempt to write a
   readonly database`.
6. Built a synthetic current-schema datastore (a primary file with `shards`,
   `shard_key_config`, `shard_keys` tables plus two shard files each containing a
   `wavenumber` table) that is fully consistent. Ran the audit tool against it: it
   correctly reports the row count, per-shard distribution, the cross-file
   `wavenumber` row count, and `VERDICT: OK`, exit code 0.
7. Corrupted that synthetic datastore (added a `shard_keys` row with no matching
   `wavenumber` row, and a `shard_keys` row pointing at a non-existent `shard_id`).
   First run exposed a real bug in my own script — the `orphaned`/`unassigned` set
   differences were swapped, so a genuinely orphaned key was being reported as
   "unassigned (not necessarily an error)" and the run exited 0. Fixed the two lines
   (`orphaned = shard_key_serials - key_table_serials`, `unassigned =
   key_table_serials - shard_key_serials`) and reran: both injected faults were now
   correctly reported under `!! INCONSISTENT:`, exit code 1.
8. Repaired the synthetic datastore back to consistency and reran: `VERDICT: OK`,
   exit code 0, confirming the fix in step 7 did not introduce a false positive.

Not run — needs a real Ray pipeline (see audit §8 checklist items 1–2, and the "State
handed to the next prompt" section below):

- Creating a fresh `SGWK` sharded datastore end-to-end and confirming every
  `shard_keys.key_serial` equals the corresponding `wavenumber.serial`, with no
  `MISMATCH` lines printed.
- Stopping and resuming a run against that datastore and confirming all
  previously-written records are still reachable.

## Observations not acted on

- `ShardedPool.__init__`'s error path at `self._db_file` (referenced when
  `self._primary_file.is_dir()`) is a pre-existing `AttributeError` — the attribute is
  actually named `self._primary_file`. This is item D1, explicitly scoped to prompt
  03. Left untouched here.
- The commented-out `# print(f">> assigned shard #...")` line remains dead code, as
  before. Not in scope for this prompt.

## State handed to the next prompt

- B1 and B5 are fixed in `_assign_shard_keys`; `_read_shard_data` (prompt 02, B2) is
  untouched, as instructed.
- `tools/shard_key_audit.py` exists and its negative path (pre-refactor schema) is
  exercised against the in-tree `test-qcd-db.sqlite`. Its positive and inconsistency
  paths are exercised only against synthetic databases built for this verification,
  not against a real `SGWK`-produced datastore, because none exists in the tree with
  the current schema.
- **Open item for `IMPLEMENTATION_STATE.md` §3**: audit §8 checklist items 1–2 (fresh
  datastore has no `MISMATCH` lines; stop/resume finds all previously-written records)
  require a real Ray pipeline run and were not exercised. Prompt 10 should pick this
  up when it does the final verification pass.
