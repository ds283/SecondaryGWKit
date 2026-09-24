# Campaign — datastore portability

## 0. Why this campaign exists

A `ShardedPool` datastore is one primary SQLite file plus *N* shard files alongside it
(`foo.sqlite`, `foo-shard0000.sqlite`, …). The primary records where each shard is in its `shards`
table. **It records the absolute path**, taken from `Path(db_name).resolve()` at
`Datastore/SQL/ShardedPool.py:71`. Symlinks are resolved first, so a store opened through a
symlinked directory records the target directory, not the link.

That path is written once, by `_write_shard_data` (`:270`), when the store is created.
`ShardedPool` never updates it. `_read_shard_data` (`:314`) uses it exactly as stored, and
`ShardedPool` never checks that the file is still there. The one other writer is a workaround:
`docs/handover/quadsource_atol_sweep.py` `prepare()` (`:648`) issues an `UPDATE` to re-point a
copied primary at its own shards, because the pool would not do it itself.

- **A store that has been copied** opens against the **original's** shards. It reads them, writes to
  them, and inserts a `version` row into each on open. The only file of its own that it changes is
  the copy's primary, whose `shard_keys` table then disagrees with the original's about which key
  lives on which shard. **This has happened.** On 2026-09-23 the first run of the atol sweep put 54
  rows into the A3 baseline store its own docstring promised to leave alone. That incident is why
  the workaround exists, and it is recorded as
  `[04-sharded-store-paths-are-absolute-and-so-stores-are-not-portable]` on the
  [`run-registry`](../run-registry/IMPLEMENTATION_STATE.md) board §3, owner "datastore code". **This
  campaign is that owner, and closes that issue.**
- **A store that has been moved or renamed**: here the record and a reading of the code disagree.
  The issue says a primary naming a nonexistent shard *raises*. A reading of the code says each path
  goes straight to a `Datastore` actor, which **creates an empty database when its file is missing**
  (`Datastore.py:216`, correct behaviour for a single store). On that reading, the pool opens with
  empty shards at the old path, every sharded lookup misses, and you see a datastore in which
  nothing has been computed rather than an error. Prompt 01 settles which is true by measurement
  before changing anything (its §2 **P0**). Either way the fix is the same, and either way the
  copied case is silent.

**0.1 A second instance, not yet recorded anywhere.** `var/datastores/backup-pre-resume-20260921T091011/`
is the retained backup of the `handover-A3-baseline-lambdacdm` store, and the
`datastore-readback` campaign and that store's manifest both treat it as a safe copy. The backup's
own primary records the **live** store's shard paths
(`/Users/…/var/datastores/handover-A3-baseline-lambdacdm-shard000N.sqlite`, checked 2026-09-24,
read-only). So opening the backup as a datastore would open the live shards and write to them. The
backup is only good for restoring by copying files over the live ones. It cannot be opened in
place, and nothing about it says so.

**0.2 Correctness is the only objective.** Sequence by epistemic dependency, never by urgency or
by what is cheap. What a missing shard does today is measured before anything is changed. The
fail-closed check on missing shards comes before the relative-path change,
because the relative-path change rests on it: without the check, a mistake in path resolution
repeats the failure in §0 silently instead of raising an error.

## 1. Scope

**In scope:** `Datastore/SQL/ShardedPool.py` (the three methods that touch the `shards` table and
the constructor branch that opens an existing store), `tools/shard_key_audit.py` (another reader of
that table), and `Datastore/tests/`. **Not** `docs/handover/quadsource_atol_sweep.py`: it is the
record of a measurement, its re-pointing workaround stays as it is, and prompt 01 §2 explains why it
keeps working unchanged.

**Out of scope:** `Datastore/SQL/Datastore.py`, whose create-when-missing behaviour is correct for a
single store. Also the object factories, `main.py`, any physics, and any change to the columns of
the `shards` table. Renaming a whole store (primary and shards together) is also out of scope; see
prompt 01 §5.

**Amended 2026-09-24, after prompt 01 landed** (decisions in §6). Renaming or copying a whole
store is **now in scope, for prompt 02 only**. That prompt adds a static interface to
`Datastore/SQL/ShardedPool.py` that copies or moves a closed store and rewrites its `shards` rows,
plus a thin command-line script in `tools/` over that interface. Its scope is
`Datastore/SQL/ShardedPool.py`, `Datastore/shard_paths.py`, the new script, and `Datastore/tests/`.
The paragraph above still describes prompt 01, which was correct for the tree it ran on.

**Amended again 2026-09-24, after prompt 02 landed** (decisions in §6.5). Prompt 03 is in scope:
the registry's store operations. Its scope is `RunRegistry/`, meaning a new module for the store
sidecar and the store operations, `RunRegistry/__init__.py`'s `begin()` (which records a
`store_id` beside `results`), `RunRegistry/__main__.py` and `RunRegistry/tests/`. It calls prompt
02's `ShardedPool.copy_store` / `move_store` and does not change them. `Datastore/`, `tools/`,
`main.py`, `docs/` and every existing run manifest stay out of scope.

**Out of scope for `ShardedPool` and the `tools/` script, permanently:** any file other than a
store's primary and its shards. In particular, a `<stem>.manifest.json` sidecar is a registry-layer
artefact. `ShardedPool` and the bare script neither copy it, move it, nor mention it. Whether and
how the registry moves or copies stores, and carries the sidecar with them, is prompt 03, which is
**held** (§2, §6).

## 2. Prompts

| # | Prompt | Covers |
|---|---|---|
| 01 | [`01-relative-shard-paths.md`](01-relative-shard-paths.md) | Measure what a missing shard does today; fail closed on it; record shard paths relative to the primary; read existing absolute records safely; one resolver shared with the audit tool. Closes `run-registry`'s `[04-sharded-store-paths-are-absolute-…]` |
| 02 | [`02-copy-and-move-a-store.md`](02-copy-and-move-a-store.md) | A static `ShardedPool` interface that copies or moves a closed store under a new name and rewrites its `shards` rows. One shard naming rule shared with the creator. Every interrupted state either opens correctly or is refused. A bare `tools/` script over it. Closes `[01-whole-store-rename-is-unsupported]` |
| 03 | *not written; **held*** | The registry's move and copy for stores, calling prompt 02's interface and managing the `<stem>.manifest.json` sidecar. Held on two user decisions: whether the registry's charter extends to acting on stores, and who owns the sidecar and in what format. Tracked as `[store-sidecar-manifests-have-no-owner]`. **Both decided 2026-09-24 (§6.5); released** |

Prompt 03's charter is fixed here; only its method is held. It is written once the user has made
both decisions. It must not be written against a guess at them.

## 3. Datastores

Two stores exist under `var/datastores/` (gitignored), each with 4 shards of about 85 MB:
`handover-A3-baseline-lambdacdm` and `handover-atol-sweep`, plus the backup in §0.1. The sweep
store's primary was re-pointed by `prepare()`, so it names its own shards, as absolute paths. **All three
are read-only to this campaign.** The acceptance test needs a copy, placed in a **different
directory** under `var/` and under a **different stem**, and discarded afterwards. Never open an
original, and never open the backup: on the unfixed tree, opening either writes to the live shards.

Before touching any store, run `python -m RunRegistry list` and confirm nothing is `running`
against it.

**For prompt 02, the new tool is never pointed at an original.** Its demonstration copies the sweep
store by hand (`cp`) into a working directory under `var/`, and runs the tool only on that hand-made
copy and on what the tool produces from it. The hand-made copy's primary still holds the originals'
absolute paths, so it is the hardest legacy case: a copied primary whose records name another store
that exists. About 0.7 GB is needed for the duration; the volume had 13 GB free on 2026-09-24.

## 4. Baselines

At `a0922e7` on `handover-remedial`, the per-package counts are recorded on the boards of the most
recent campaigns. Re-measure before dispatching, and record `Datastore/tests` too, which exists
now:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . 2>&1 | tail -40
```

The suites print model banners on stdout, so `| tail -5` will not show the verdict. **Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
is a known wall-clock flake. Re-run that module on its own before attributing a failure to a
change.

At `b04671f` (prompt 01), the counts are AdaptiveLevin 32, ComputeTargets 552 (the one failure is
the flake), CosmologyModels 39, **Datastore 37**, LiouvilleGreen 148 (1 skipped) and RunRegistry
38. These are recorded on the board's §5. Re-measure before dispatching prompt 02 anyway.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, unchanged, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`.
4. **Do not fix things the prompt did not ask for.** Record them and open a §3 issue.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**

### 5.1 The log template

Subject, commit, result; **What shipped**; **Deviations from the prompt** (each classified);
**Verification performed**; **Observations not acted on**; **State handed to the next prompt**.

## 6. Decisions recorded during the campaign

**6.1 A copy or move may rename files; the only extra step is to rewrite the rows (user,
2026-09-24).** Prompt 01's §4.1 asked for a copy with every file renamed to a new stem. Its §4.4
asked for that copy's primary to keep its legacy absolute rows. P3 reads an absolute record by its
file name, so the two cannot both hold, and the agent substituted a copy with only the primary
renamed (logged `STRUCTURALLY REQUIRED`; prompt 01's log, "Deviations" item 1). The user's reading
is that **nothing prevents the whole-store rename**. With bare-name records, the rename is the file
operations plus one `UPDATE shards SET filename = ?` per serial. `[01-whole-store-rename-is-unsupported]`
overstated the gap. What was missing is an interface, not a capability. The inconsistency was in the
prompt, not in the code. Prompt 02's demonstration does what prompt 01 §4.1 intended.

**6.2 The interface lives in `ShardedPool`; client code decides how to use it (user,
2026-09-24).** Of prompt 01 §5's two options, **(a) is taken**: an explicit operation that renames
the files and rewrites the `shards` rows. Option (b), deriving shard names from the primary's stem
at read time, is **not** taken, so the `shards` table stays the authority on where a shard is. The
operation is a static interface on `ShardedPool` rather than a method of an open pool, because an
open pool has one `Datastore` actor per shard holding its file. The first client is a small
standalone script in `tools/`, which accepts a store and copies or moves it.

**6.3 `ShardedPool` knows nothing about sidecar files (user, 2026-09-24).** A
`<stem>.manifest.json` beside a store is not part of the `ShardedPool` structure. It comes from the
registry layer, and what to do with it is for the tool that manages it. So neither `ShardedPool`
nor the bare `tools/` script copies, moves, refuses because of, or warns about any file other than
the primary and its shards. Moving or copying a store *with* its sidecar would be a separate
registry tool that calls the same interface and manages the sidecar itself.

**6.4 Why prompt 03 is held.** Two facts, found while recording 6.3, stand between the registry
tool and a prompt that could be written now:

- **No code owns the store sidecar.** `RunRegistry` writes `manifest.json` inside each run
  directory under `var/runs/`. The store-level sidecars are something else.
  `handover-A3-baseline-lambdacdm.manifest.json` was written by hand, and
  `handover-atol-sweep.manifest.json` by `docs/handover/quadsource_atol_sweep.py` `prepare()`, whose
  own comment calls it "a human note" that nothing reads. The sweep sidecar's content names the
  store's path (`"datastore"`) and its origin (`"copied_from"`), so a move or copy must decide what
  happens to those fields. The registry has to own the format and a writer before it can manage
  the file.
- **The registry's charter is "it records; it does not act".** The `run-registry` board declined a
  `pull` command on exactly that ground ("transfer is acting, not recording",
  `[04-a-runs-product-is-named-but-never-fingerprinted]`). A local move or copy is not transfer
  between machines, but it is the registry acting on files. Whether to extend the charter is the
  user's decision, and should be made explicitly rather than drifted into.

The registry is also where "is anything using this store?" can be answered. The bare script
cannot tell whether a process has a store open. These stores use SQLite's default rollback
journal, which leaves no file while idle. The registry knows which runs are `running` and which
datastore each names.

**6.5 Both decisions are made; prompt 03 is released (user, 2026-09-24).** The user settled both
decisions of §6.4 on the day they were recorded. They were written down only after prompt 02
landed, so that prompt 02's orchestrator would find `HEAD` where it expected. §6.4 stays as it
is: it was the correct account of what blocked prompt 03 until this entry.

- **The charter.** Moving and copying datastores are tools for managing the registry, and the
  user sees no problem with the registry doing them. The "it records; it does not act" objection
  does not apply to them. Nor does the precedent of the declined `pull`, which was about transfer
  between machines (`run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]`); that
  issue is not reopened by this. What the registry still does not do is unchanged: it does not
  schedule, supervise, restart, lock or delete (`CLAUDE.md`).
- **The owner and the format.** "The registry owns the sidecar" means all seven of the following,
  agreed as proposed:
  1. **One `RunRegistry/` module defines the sidecar**: its required fields, the optional fields it
     knows, a reader and an atomic writer (the package's `write_json_atomic`). Only registry
     operations write a sidecar. **Unknown fields are preserved verbatim.** The hand-written A3
     sidecar carries `run_history`, `restart`, `backup` and `note`, and none of them may be lost
     or reformatted.
  2. **Copy and move update its metadata fields.** There is also a registry create/adopt operation,
     so that no sidecar has to be written by hand.
  3. **`datastore` is stored as the primary's bare file name, not a path.** The backup's sidecar
     already names the live store (§0.1). That is prompt 01's defect again, this time in JSON.
  4. **`copied_from` is kept as the immediate parent**, plus an append-only **`history`** list,
     one entry per copy or move, with the operation, from, to, when and git SHA.
  5. **A stable `store_id` in the sidecar.** A copy gets a new one; a move keeps it. Future run
     manifests record it next to the `results` path. Existing run manifests are immutable and
     are never rewritten. This touches the run-manifest format, which lies outside this
     campaign's original scope (§1, amended again below).
  6. **Registry copy and move refuse a store that a `running` run's `results` names.** This is the
     check the bare script cannot make (§6.4, last paragraph).
  7. **Existing sidecars are read as they are, with no automatic rewrite.** The backup's stale
     `datastore` field is fixed only if the user asks.

These decisions fix what prompt 03 must do. How it does it is the prompt's job, and where the
prompt had to choose, it says so and gives its reason.
