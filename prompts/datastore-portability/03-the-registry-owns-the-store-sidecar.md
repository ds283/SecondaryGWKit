# Prompt 03 — the registry owns the store sidecar, and copies and moves stores with it

**Campaign:** [`README.md`](README.md) · **Board items:** **P9**–**P13** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[store-sidecar-manifests-have-no-owner]` (this board's §3; `docs/OPEN_ISSUES.md` §1.12).
**Does not touch:** `[01-atol-sweep-check-expects-absolute-shard-records]`.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Opus**. The code is a few hundred lines of plain Python. The judgement
is in three places: which fields a sidecar operation may change, the order of file operations
across two layers, and what counts as a store being in use.

**Read first:**

1. [`README.md`](README.md): §0, §1 including **both** 2026-09-24 amendments, §3, and **§6 in
   full**. §6.5 holds the user's decisions that this prompt implements. Do not reopen them.
2. Prompt 02, [`02-copy-and-move-a-store.md`](02-copy-and-move-a-store.md), and its log,
   [`logs/02-copy-and-move-a-store.md`](logs/02-copy-and-move-a-store.md). Read especially the
   interruption table, the deviations, and "The real-store demonstration". This prompt sits on
   top of prompt 02's interface and must not weaken its property.
3. `RunRegistry/__init__.py` and `RunRegistry/__main__.py`, in full. Note `write_json_atomic`,
   `_repo_path` / `_resolve`, `git_provenance`, `now_iso`, `begin()`, `Run.results_path`,
   `liveness()` and `list_runs()`.
4. `RunRegistry/tests/test_run_registry.py` (the `RegistryTestCase` pattern, and `a_dead_run`)
   and `RunRegistry/tests/test_pipeline_adoption.py`.
5. `Datastore/SQL/ShardedPool.py`: `copy_store`, `move_store`, `_plan_relocation` and
   `_failure_message`. **You do not change this file.** Also
   `Datastore/tests/shard_store_fixtures.py`, which your tests may import to build stores.
6. The three sidecars, **read-only**:
   - `var/datastores/handover-A3-baseline-lambdacdm.manifest.json`, written by hand;
   - `var/datastores/handover-atol-sweep.manifest.json`, written by
     `docs/handover/quadsource_atol_sweep.py` `prepare()` (`:667`);
   - `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json`,
     whose `datastore` names the live store (README §0.1, §6.5 point 3).
7. `CLAUDE.md`: "Long-running jobs — the run registry" and the repository mechanics section.

---

## 1. What is wanted

A `<stem>.manifest.json` beside a store is where a person learns what the store is for and where
it came from. Today no code owns it. One sidecar was written by hand. One was written by a
measurement script. One names a different store from the one it sits beside. And prompt 02's
bare script leaves a sidecar behind when it moves a store. README §6.5 records the user's
decisions. In short:

- the registry may copy and move stores;
- one `RunRegistry/` module owns the sidecar: its format, its reader and its writer;
- only registry operations write a sidecar, and unknown fields survive every write;
- `datastore` is a bare file name;
- `copied_from` names the immediate parent, beside an append-only `history`;
- a `store_id` is new on a copy and kept on a move, and new run manifests record it;
- copy and move refuse a store that a `running` run names;
- existing sidecars are read as they are, and nothing rewrites them unless asked.

The layering of README §6.2–§6.3 is unchanged and is load-bearing. `ShardedPool` and
`tools/sharded_store.py` still know nothing about sidecars. The registry calls
`ShardedPool.copy_store` / `move_store` for the store's files, and handles the sidecar and the
run records itself. **No file in `Datastore/` or `tools/` changes.**

---

## 2. What to change

Put the new code in **one new module, `RunRegistry/stores.py`**. The two exceptions are small:
the change to `begin()` (P12) and the command line (P9–P11). Everything in §2 marked
*prompt's choice* is a design choice this prompt makes where §6.5 left the method open. Follow
it, or log a stronger alternative as a deviation.

### P9 — the sidecar: its name, its format, its reader and its writer

**The name.** One function, `sidecar_path(primary) -> Path`, returning
`primary.parent / f"{primary.stem}.manifest.json"`. This is what both existing sidecars and
`prepare()` use. No other code spells the pattern.

**The format** (*prompt's choice*: the field names are those the existing sidecars already use,
where one exists). A **registry sidecar** is a JSON object with `"sidecar_format": 1`. Its
fields:

| Field | Required | Meaning | Written by |
|---|---|---|---|
| `sidecar_format` | yes | `1`. Its presence is what tells a registry sidecar from a legacy one | create, adopt |
| `store_id` | yes | 32 lowercase hex characters (`uuid.uuid4().hex`). New on create, adopt and copy; kept on move | create, adopt, copy |
| `datastore` | yes | The primary's **bare file name**, e.g. `pcopy.sqlite`. Never a path (§6.5 point 3) | create, adopt, copy, move |
| `name` | yes | The primary's stem | create, adopt, copy, move |
| `purpose` | yes | One line a stranger can read. A copy is a new store and needs its own | create, adopt (kept if present), copy (given) |
| `created` | yes | When this store identity began, as `now_iso()` | create, copy; adopt keeps a legacy value |
| `copied_from` | no | The immediate parent: `{"store_id": …, "datastore": <parent primary path>}` for a registry copy. A legacy string is kept verbatim | copy |
| `history` | yes | Append-only list, one entry per operation: `{"operation", "from", "to", "when", "git_head", "git_dirty"}`. `from` and `to` are primary paths, and `from` is `null` for create and adopt | every operation appends one |

Every other field is **unknown**, and is preserved **verbatim** by every write: create excepted,
since it has nothing to preserve; adopt, copy and move included. "Verbatim" means
*value-identical after a JSON round trip*. The writer is `write_json_atomic`, which re-indents
and sorts keys, so bytes cannot survive, and the decision does not ask them to. On 2026-09-24
the orchestrator checked, read-only, that all three real sidecars round-trip value-identically
through `json.dumps(…, indent=2, sort_keys=True)`. Re-check it (§4 step 1).

The paths inside `copied_from` and `history` record where something was **at the time**. They
are written as `RunRegistry._repo_path` writes paths: repository-relative under the repository,
absolute otherwise. Nothing ever opens them or resolves a store from them. **Only `datastore`
says where the store is, and it says it by name.** *Prompt's choice:* history entries for create
and adopt. §6.5 point 4 asks for one entry per copy or move. Starting the list with the operation
that assigned the `store_id` means a history always says where its identity came from.

**The reader,** `read_sidecar(primary)`. It never writes and never raises on bad content. It
returns what it found:

- **absent**: there is no file at `sidecar_path(primary)`;
- **unreadable**: the file is there, but is not JSON, or not a JSON object;
- **legacy**: a JSON object with no `sidecar_format`;
- **registry**: `sidecar_format == 1`.

It also returns a list of **problems**, empty when the sidecar describes this primary:

- the primary does not exist or is not a regular file (an orphaned sidecar);
- a registry sidecar lacks a required field, or has a malformed one;
- `datastore` does not name this primary;
- `name` is not this primary's stem.

**A legacy `datastore` that is a path is read by its final component**, never as a path. This is
prompt 01's rule for legacy shard records, applied to JSON. It is why the backup's sidecar, which
records `var/datastores/handover-A3-baseline-lambdacdm.sqlite`, reads as naming the sibling it
sits beside and not the live store. The reader reports that the record is a legacy path; it does
not treat it as a problem. A registry `datastore` containing a separator **is** a problem: the
registry never writes one.

**The writer.** Two private functions over `write_json_atomic`:

- one writes a **new** sidecar. It refuses if the target or its `.tmp` name exists;
- one **updates** an existing sidecar in place. It refuses if the `.tmp` name exists.

**Nothing is ever overwritten** except by the update function, and that only in the two places
this prompt names (adopt, and the move's temporary sidecar).

### P10 — create and adopt

`create_sidecar(primary, purpose)` gives an existing store with no sidecar a registry sidecar. It
refuses, naming the file, when:

- the primary is not an existing regular file (a symlink is refused, as prompt 02 refuses one);
- the sidecar or its `.tmp` name exists;
- `purpose` is empty.

It writes the fields above, with a `create` history entry. *Prompt's choice:* it does not open the
primary. A sidecar is a note beside a store, and whether the store is sound is the constructor's
question, not the note's.

`adopt_sidecar(primary, purpose=None)` upgrades a legacy sidecar in place. It is the only
operation that rewrites an existing legacy file, and it runs only when a person asks (§6.5
point 7). It refuses, naming the file and the reason, when:

- there is no sidecar, or the sidecar is not legacy (use create; or it is already a registry
  sidecar);
- the reader reports any problem;
- the legacy object already has a `store_id` or a `history` field. The registry would be
  guessing at what it means (*prompt's choice*: refuse rather than reinterpret);
- the sidecar has no `purpose` and none is given.

It adds `sidecar_format`, a new `store_id` and an `adopt` history entry. It rewrites `datastore`
to the bare name and sets `name` to the stem, which the reader has already checked. It keeps
`created` and `purpose` if present, keeps a legacy `copied_from` string verbatim, and keeps every
unknown field verbatim.

### P11 — copy and move, with the sidecar, refusing a store in use

```python
copy_store(src, dst, purpose, runs_root=None) -> dict   # the destination's sidecar
move_store(src, dst, runs_root=None) -> dict
```

`src` and `dst` are primary paths, as in prompt 02. `runs_root` defaults to `DEFAULT_ROOT`. Each
calls `ShardedPool.copy_store` / `move_store` for the store's files and nothing else in
`Datastore/`. Import `ShardedPool` **inside** these functions, so that `import RunRegistry` and
`python -m RunRegistry list` never import `ray` or `sqlalchemy` (a test pins this).

**Refusals, all before anything is written,** each a `RuntimeError` naming the file and the reason:

- **the source's sidecar is not a registry sidecar**, or the reader reports a problem with it.
  Tell the caller to run `store create` or `store adopt` first. *Prompt's choice:* copy and move
  never upgrade a legacy sidecar implicitly, because that is exactly the "automatic rewrite" §6.5
  point 7 rules out;
- **a run whose state is `running` names the source or the destination.** "Names" means either:
  - its manifest's `results`, resolved, is the same path as the primary, where both are
    `os.path.realpath`'d, since the file may no longer exist; or
  - its manifest's `results_store_id` (P12) equals the source's `store_id`.

  Refuse on any `running` run, **alive or stale**, and say which, with the run id. *Prompt's
  choice:* liveness is evidence, not proof, and the registry does not decide that a run is dead.
  A stale run is ended by a person, with `Run.finish`, before the store moves. Runs with no
  manifest name nothing, and are not an error;
- **the destination's sidecar name, or its `.tmp` name, exists.** A move also refuses if the move's
  temporary sidecar name (below) exists;
- then **prompt 02's refusals**, raised by `ShardedPool` before it writes. Let them through
  unchanged, adding only that no sidecar was written.

**Copy, in this order:**

1. `ShardedPool.copy_store(src, dst)`;
2. write the destination's sidecar as a **new** file. It holds:
   - the source's fields and every unknown field, verbatim;
   - a new `store_id`;
   - `datastore` and `name` for `dst`;
   - the given `purpose`;
   - `created` now;
   - `copied_from = {"store_id": <source's>, "datastore": <source primary path>}`;
   - `history` = the source's history plus one `copy` entry.

**The source's sidecar is never opened for writing.**

**Move, in this order** (*prompt's choice*, argued below):

1. `ShardedPool.move_store(src, dst)`;
2. `os.rename` the source's sidecar to a **temporary name in the destination directory**,
   `<dst sidecar name>.incomplete-move`;
3. update that temporary file in place: `datastore` and `name` for `dst`, and one `move` entry
   appended to `history`. `store_id`, `purpose`, `created`, `copied_from` and every unknown field
   are kept;
4. `os.rename` it to the destination's sidecar name, after checking that the name is still free.

*Why this order:*

- **Store first.** It matches prompt 02: the thing that marks an operation complete appears
  last. A registry sidecar that validly describes a store therefore exists only once the store
  is in place.
- **The temporary name.** A sidecar never sits under a sidecar name while it is only half
  updated. That matters in a same-stem move to a new directory: an un-updated sidecar there
  would pass every check the reader makes, and yet lack its `move` history entry.

Nothing is deleted at any point. The move renames the one sidecar; it never writes a new one and
removes the old.

**The property to establish,** recorded in the log as a table with one row per point of
interruption, for copy and for move. **Reachable** means found at the source or destination names
by `read_sidecar`, by the constructor or by prompt 02's read-and-check. The property: *if the
process dies after any step, every reachable sidecar either (a) describes, with no problems, the
store beside it, with a complete history, or (b) is reported by the reader as having a problem.
Every reachable store is in one of prompt 02's states.* Neither of these may happen:

- a problem-free sidecar beside the wrong store, or with a missing history entry;
- two problem-free sidecars with the same `store_id` after a move.

Work through at least these layouts: same directory with a new stem; new directory with the same
stem; new directory with a new stem.

*On failure* nothing is cleaned up, as in prompt 02. The error names the step that failed, and
lists the store and sidecar files that exist at each end, including any `.tmp` or
`.incomplete-move` file.

### P12 — run manifests record the store's id

`begin()` gains one manifest field, **`results_store_id`**. When `results` is given and
`read_sidecar` finds a problem-free registry sidecar beside it, the field holds that sidecar's
`store_id`. Otherwise it is `null`: no store, no sidecar, a legacy sidecar, or a sidecar with a
problem. `begin()` reads the sidecar and **never writes one**. Existing manifests are never
rewritten. A manifest without the field reads exactly as before: `list`, `Run.results_path` and
the P11 check all tolerate its absence.

### P13 — the real-store demonstration

§4 below.

Update `begin()`'s docstring, whose last paragraph says every manifest field is one that README §0
would have caught something with. This one is §6.5 point 5: it matches a run to its store after
the store has moved, which a path cannot.

### The command line (part of P9–P11)

Add a `store` subcommand to `RunRegistry/__main__.py`:

```
python -m RunRegistry store show   PRIMARY
python -m RunRegistry store create PRIMARY --purpose TEXT
python -m RunRegistry store adopt  PRIMARY [--purpose TEXT]
python -m RunRegistry store copy   SRC DST --purpose TEXT [--runs-root DIR]
python -m RunRegistry store move   SRC DST [--runs-root DIR]
```

- `show` prints the reader's verdict: its kind, its problems, whether `datastore` is a legacy
  path, and the fields. It also prints every run whose `results` or `results_store_id` names the
  store, with its state and liveness. It is read-only.
- Every command prints the reason and exits non-zero on refusal.
- None initialises Ray (a test checks this, as prompt 02's script test does).
- `list` is unchanged.
- The package docstring's first paragraph ("it schedules nothing, supervises nothing, locks
  nothing and deletes nothing") stays true. Add a sentence recording the charter decision
  (README §6.5): the registry also creates, adopts, copies and moves the stores it manages, and
  still deletes nothing.

---

## 3. Tests — in `RunRegistry/tests/`, no Ray, nothing under `var/`

Build stores with `Datastore/tests/shard_store_fixtures.py` in temporary directories, and runs with
the `RegistryTestCase` pattern, in a temporary runs root. At minimum:

1. **Name and reader.** `sidecar_path` gives `<stem>.manifest.json`. The reader classifies absent,
   unreadable, legacy and registry. It reads a legacy `datastore` path by its name, including one
   naming a populated other directory, which must not be touched. It reports each problem in P9.
   It never writes (compare a `tree_state` listing before and after).
2. **Unknown fields survive.** A legacy sidecar with the **shape of the real A3 sidecar** goes
   through adopt, then copy, then move. The shape means nested `run_history` entries, a
   `restart` object and a `backup` object. Write the fixture inline in the test; do not read
   `var/`. After each step, every unknown field is value-identical to the original.
3. **Create.** The fields, one `create` history entry, `datastore` bare. Plus each refusal.
4. **Adopt.** The sweep-shaped legacy sidecar, whose `datastore` and `copied_from` are paths into
   another directory: `datastore` becomes bare, `copied_from` is kept verbatim, the `store_id` is
   new, and there is one `adopt` entry. Plus each refusal, including one where the legacy
   `datastore` names a different file.
5. **Copy.**
   - The destination sidecar holds a new `store_id`, `copied_from` naming the source's id, and
     the source's history plus one `copy` entry.
   - Its `datastore` and `name` are the destination's, and its `purpose` is the one given.
   - The source's sidecar and store files are byte-identical, with unchanged mtimes.
   - The destination store reads its own shards (prompt 02's `read_pool` pattern).
6. **Move.** In the three layouts:
   - the `store_id` is kept, with one `move` entry;
   - no sidecar, `.tmp` or `.incomplete-move` file remains anywhere but at the destination's
     sidecar name;
   - the store reads its own shards.
7. **Refusals.** One test per refusal in P11. Each asserts that the error names the offending
   file or run, and that **nothing was created, changed or removed** under the temporary root
   (`tree_state`). Among them:
   - a running run that is alive (its pid is the test process), and one that is stale (use
     `a_dead_run`);
   - a match by `results` path, and one by `results_store_id` alone, where the `results` path
     names somewhere else;
   - a finished run naming the store, which does **not** refuse;
   - one of prompt 02's refusals passing through.
8. **Interruption.** For each step of each operation, make the step after it fail, then assert
   the log table's row. Monkeypatch `ShardedPool.copy_store` / `move_store`, `os.rename`,
   `write_json_atomic`, or the update function to raise. Cover the three move layouts.
9. **Run manifests.** `begin(results=…)` records the `store_id` of a registry sidecar. It records
   `null` for none, for legacy, and for a sidecar with a problem. It never creates a sidecar. A
   manifest written without the field still lists, and the P11 check still reads it.
   `test_pipeline_adoption` passes unmodified.
10. **Command line and imports.** Each `store` subcommand, run as `python -m RunRegistry` in a
    subprocess:
    - it succeeds, or refuses with a non-zero exit and no writes;
    - it leaves `ray` not initialised;
    - `python -m RunRegistry list` and `import RunRegistry` load neither `ray` nor `sqlalchemy`
      (check `sys.modules` in a child interpreter).

**Deliberate breakage.** Show that each of these mutations makes the tests written against it
fail, then restore it. Put each in the log as a short diff, **exactly as applied**, so that the
orchestrator can replay it with `git apply`, and name the tests that failed. Mutations are never
committed.

- (i) copy keeps the source's `store_id`;
- (ii) move assigns a new `store_id`;
- (iii) the writer keeps only the known fields;
- (iv) the running-run check considers only runs whose liveness is `alive`;
- (v) the running-run check ignores `results_store_id`;
- (vi) `datastore` is written as a repository path, not a bare name;
- (vii) the reader reads a legacy `datastore` as a path, not by name;
- (viii) move renames the sidecar straight to its final name and updates it there;
- (ix) `begin()` does not record `results_store_id`.

---

## 4. The real-store demonstration

End to end, through `python -m RunRegistry store …` and then the real constructor via `main.py`.
**No new code, and neither `ShardedPool` nor `main.py`, is ever pointed at an original store, at
the backup, or at any of their sidecars.** Read-only snapshots with `sqlite3` `mode=ro`,
`sha256sum` and `json.load` are the only contact. All work goes under `var/portability-check-03/`,
and all of it is deleted at the end. **Nothing is written under `var/runs/` or
`var/datastores/`.** Every `store copy` / `store move` is given
`--runs-root var/portability-check-03/runs`, and every `begin()` in a throwaway script is given
that root.

1. **Before.** Run `python -m RunRegistry list` and confirm nothing is `running`. Snapshot
   read-only:
   - the three stores, as prompt 02 did: mtimes, sizes, per-table row counts per shard, and the
     primaries' SHA-256;
   - **the three sidecars**: SHA-256, mtime, size;
   - the list of run directories under `var/runs/` and their manifests' SHA-256.

   Check with `json` alone that each sidecar round-trips value-identically through
   `json.dumps(…, indent=2, sort_keys=True)`. If one does not, stop (§7).
2. **The hand-made source.**
   - `cp -p` the sweep store's five files **and its sidecar** into
     `var/portability-check-03/src/`, keeping their names. The sidecar is legacy, and its
     `datastore` and `copied_from` are paths into `var/datastores/`: the backup's defect,
     reproduced on a copy.
   - Delete one row from one sharded table of the hand-made copy's shard 0, as prompt 02 did,
     and record which row and the table's count before and after.
   - Also make `var/portability-check-03/a3probe/`: a `cp -p` of the **A3 sidecar** beside an
     empty placeholder file `handover-A3-baseline-lambdacdm.sqlite`. Create and adopt do not
     open the primary (P10), so this exercises the richest real sidecar without copying a second
     store.
3. **`store show`** on the source: legacy, `datastore` a legacy path read by name, no problems.
4. **`store copy`** of the source, before adoption: refused, and the tree under `src/` is
   unchanged.
5. **`store adopt`** the source, then `store adopt` the probe. For each, show the sidecar before
   and after. Show that every unknown field of the probe is value-identical to the real A3
   sidecar's.
6. **In use.** A throwaway script calls `begin(results=<src primary>, root=var/portability-check-03/runs, …)`
   and exits without finishing, which leaves a stale `running` run. Show that its manifest's
   `results_store_id` equals the source's `store_id`. `store copy` is then refused, naming the run
   as stale. Finish the run with `Run.finish("killed")` from a throwaway script, and show that
   `store show` lists it as finished.
7. **`store copy`** to `var/portability-check-03/dst/pcopy.sqlite`, with a purpose. Show:
   - the destination's six files;
   - its sidecar: a new `store_id`, `copied_from` naming the source's id, and a two-entry history
     (`adopt`, `copy`);
   - its `shards` rows: four bare `pcopy-shard000N.sqlite` names;
   - the source's six files, byte-identical and with unchanged mtimes.
8. **Open `pcopy.sqlite` through** `main.py --database <it> --inventory --no-prune-unvalidated
   --shards 4 --ray-address local`, run as a script. Every per-table count equals the
   original's, except the edited table, which is one lower.
9. **`store move`** to `var/portability-check-03/moved/pmoved.sqlite`. Show:
   - the same `store_id` as `pcopy`;
   - a three-entry history;
   - `datastore` = `pmoved.sqlite`;
   - `dst/` holding no file at all, including no `.tmp` or `.incomplete-move`.

   Open it through `main.py --inventory` again, and get the same counts as step 8.
10. **Originals untouched.** Re-take step 1's snapshot, sidecars and `var/runs/` included. It
    must be identical.
11. **Delete `var/portability-check-03/`.** Nothing from this section is committed. Throwaway
    scripts go in your scratch space.

---

## 5. Acceptance

1. §3's tests exist in `RunRegistry/tests/`. They need no Ray, and they open nothing under
   `var/`. `test_run_registry` and `test_pipeline_adoption` pass unmodified.
2. The deliberate-breakage record: mutations (i)–(ix), each with its diff and the tests that
   failed.
3. The interruption table in the log, one row per point of interruption, each row backed by a
   test.
4. §4 is done, with its numbers quoted: the one-row discriminator, the ids, the histories, and
   the before/after snapshot. The working directory is deleted.
5. The sidecar's name, format, reader and writer each exist once, in `RunRegistry/stores.py`.
   `ShardedPool`, `Datastore/` and `tools/` are unchanged:
   `git diff HEAD~1 HEAD -- Datastore/ tools/` is empty.
6. `import RunRegistry` and `python -m RunRegistry list` load neither `ray` nor `sqlalchemy`, and
   no `store` command initialises Ray.
7. Every existing suite is unchanged. `RunRegistry/tests` rises by exactly the tests you add.
8. `black --check` is clean.
9. On the board:
   - `[store-sidecar-manifests-have-no-owner]` has moved to §4, with a closing note naming this
     prompt and its commit;
   - the §1 row for 03, items P9–P13 and §5's baselines are updated.

   In `docs/OPEN_ISSUES.md`, the §1.12 row is deleted and the count and date are corrected. All
   of this goes in the same commit.

---

## 6. What this prompt does not do

- It does not change `Datastore/`, `tools/`, `main.py`, `docs/` or `CLAUDE.md`.
- It does not rewrite any existing sidecar, run manifest or store, except by an explicit
  `store adopt` on a copy in §4. It never adopts, copies or moves an original or the backup. The
  backup's stale field is fixed only if the user asks (§6.5 point 7).
- It does not make any driver, including `main.py`, `scoped_pipeline_run.py` and
  `quadsource_atol_sweep.py` `prepare()`, create or adopt sidecars. `prepare()` still writes one
  by hand. Record that under "Observations not acted on", and open a §3 issue for it, since it
  contradicts §6.5 point 1.
- It does not make the registry delete, clean up, lock, kill, restart, or mark a run finished on
  its own.
- It does not interpret unknown fields, even ones that hold paths (the A3 sidecar's
  `restart.command` and `backup.path`). They are carried verbatim, and after a move they may be
  stale. Say so in the log.
- It does not touch `[01-atol-sweep-check-expects-absolute-shard-records]`.

---

## 7. Stop conditions — stop and ask the user

- Anything would need `ShardedPool`, `Datastore/shard_paths.py`, `tools/sharded_store.py` or
  prompt 02's behaviour to change, or would need either of them to know about a sidecar.
- A real sidecar does not round-trip value-identically, so "verbatim" cannot hold with
  `write_json_atomic`.
- No order of operations satisfies the property in P11.
- P12 would need an existing run manifest to be rewritten, or an existing manifest field to change
  meaning.
- `import RunRegistry` or `python -m RunRegistry list` cannot stay free of `ray` and `sqlalchemy`.
- Any step would point new code, `ShardedPool` or `main.py` at an original, the backup or their
  sidecars, or would write under `var/runs/` or `var/datastores/`.
- You find yourself wanting the registry to delete, clean up, lock, kill or finish a run, or to
  upgrade a legacy sidecar other than by an explicit `adopt`.
- You find that one of §6.5's seven points cannot be met as written. Do not choose between them.

---

## 8. The log and the board

`logs/03-the-registry-owns-the-store-sidecar.md`, using the template in README §5.1. In addition,
record:

- the sidecar format as shipped, and each *prompt's choice* you kept or changed, and why;
- the round-trip check of the three real sidecars;
- the interruption table;
- the deliberate-breakage record, with its diffs;
- the §4 demonstration, with its numbers.

On `IMPLEMENTATION_STATE.md`:

- the §1 row for 03;
- items P9–P13;
- `[store-sidecar-manifests-have-no-owner]` moved from §3 to §4 with a closing note;
- the new `prepare()` issue in §3;
- §5's baselines.

Also `docs/OPEN_ISSUES.md`, in the same commit: delete one row, add one, and correct the count
(net unchanged) and the date.
