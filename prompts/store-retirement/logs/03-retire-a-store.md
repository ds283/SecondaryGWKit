# Log 03 — Retire a store: delete its files, keep its sidecar as the record

**Prompt:** [`prompts/store-retirement/03-retire-a-store.md`](../03-retire-a-store.md)
**Commit:** *(this commit)* — "Retire a closed store and keep its sidecar as a tombstone"
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. No §7 stop condition was met. Prompt 01's `closed_store_files` and
`delete_store` serve §3 as shipped; nothing in `ShardedPool` changed. `ok` is false for a
tombstone without changing what any caller does with a live store, and no existing test changed.
Nothing under `var/` was opened, fingerprinted or retired. One issue is opened
(§ "Observations not acted on", item 1).

## What shipped

**`RunRegistry/stores.py`.**

- **R6, the format.**
  - `retired` joins `KNOWN_FIELDS`. `_retired_problems` checks its shape, in the style of
    `_fingerprint_problems`. `_registry_problems` also requires that a sidecar carries `retired`
    exactly when its history ends in a `retire` entry. So every writer's `_check_before_writing`
    validates both of the tombstone's writes.
  - `retire` is a history operation (`RETIRE_OPERATION`). `_history_problems` allows it after
    index 0 and only as the last entry. Its `from` must be a path, and its `to` must be `null`.
    Every other entry's `to` must still be a non-empty string.
  - `RETIREMENT_STATES = ("retiring", "retired")` and `DEFAULT_STORES_ROOT`
    (`<repo>/var/datastores`) are new constants.
- **R6, the reader.**
  - `SidecarReading` gains a `retired` property, true for a registry sidecar that carries `retired`,
    and a field `incomplete_retirement`: the text of the one problem an interrupted retirement is,
    or `None`.
  - `ok` is now `kind == "registry" and not problems and not retired`. `store_id` follows it, so it
    is `None` for a tombstone.
  - For a registry sidecar whose `retired` is well-formed, `read_sidecar` skips `_primary_problems`
    and uses `_tombstone_problems` instead:
    - state `retiring` is always one problem, naming the files still present (or "no listed file
      remains") and the remedy, `store retire` again;
    - state `retired` with the primary, or any listed file, present is the problem "retired …, but …
      exists again", which says the sidecar describes the store that was retired;
    - state `retired` with nothing present is no problem.
  - A malformed `retired` is a shape problem, and the primary is then checked as for any sidecar.
  - A legacy sidecar is read exactly as before. A `retired` key in one is uninterpreted, and
    `unknown_fields` still lists it for a legacy reading.
- **R7, the operation.** `retire_store(primary, reason, *, runs_root=None, stores_root=None,
  without_fingerprint=False, dry_run=False) -> dict`, in §3's order:
  1. **§3.1, the sidecar.**
     - A blank or missing reason is refused.
     - An absent, unreadable or legacy sidecar is refused, naming `store create` / `store adopt`.
     - A registry sidecar with problems is refused, except when its problems are exactly
       `[incomplete_retirement]`, which is the completion path.
     - A completed tombstone is refused, naming when, the head, when it completed, and why.
     - The recorded `store_id` is read from the field.
     - A `.tmp` beside the sidecar is refused, before anything else is written or deleted (see
       the interruption table).
  2. **§3.2, in use.** The runs root must be a directory. Then `_running_runs_naming("retire",
     [primary], store_id, runs_root)` refuses any `running` run, alive or stale, by path or by the
     recorded `store_id`. The message names the run, its liveness, how it matched, that a stale run
     is ended by a person with `Run.finish`, and `store fingerprint "<primary>" --write`.
  3. **§3.3, the files and the fingerprint.**
     - The plan is `ShardedPool.closed_store_files(primary)`, and its refusals come through.
     - A refusal containing prompt 01's "its shards table could not be read" gets §3.3's message,
       under the flag or not.
     - Under the flag only, a refused plain plan is re-planned with `resume=True`. That relaxes a
       missing shard and nothing else (log 01), so every other refusal comes through again, and
       `files_present_only` is then true.
     - Without the flag, a sidecar with no `fingerprint` is refused, naming `store fingerprint
       --write`. Otherwise `fingerprint_store(primary, write=False, runs_root=…)` runs, and any
       entry of its `comparison`, `problems` entries included, refuses with every entry's text.
       A fingerprint that cannot be taken is refused too, naming `--without-fingerprint`.
     - With the flag, the fingerprint is attempted all the same. If it succeeds the call refuses.
       If it fails, `str(e)` and `type(e).__name__` are kept for the tombstone.
  4. **§3.4, the references**, by `_find_references`:
     - the runs, from `runs_naming([primary], store_id, runs_root)`, as `{id, state, matched_by}`;
     - every sidecar-named file under `stores_root`, walked recursively with `os.walk` (which does
       not follow directory links), excluding the store's own by realpath. Each is read with
       `json.load` and never opened for writing.
     - A sidecar is listed with the JSON path of each match:
       - `$.copied_from.store_id` when that equals the `store_id`;
       - every string value, found recursively, whose `os.path.realpath(_resolve(text))` is the
         primary, a planned file or the store's directory.

       One that is not JSON is listed as `{"sidecar", "unreadable"}`, and the search goes on.
     - The report carries `runs_root`, `stores_root` and `not_searched`. The last names sidecars
       outside the stores root, `.tmp` and `.incomplete-move` files, run manifests outside the runs
       root, and every reference that is not in a manifest or a sidecar (docs, boards, logs).
     - A stores root that is not a directory is refused.
     - The sidecar suffix is taken from `sidecar_path`, so `test_the_pattern_is_spelled_once` still
       holds.
  5. **§3.5.** The tombstone is built, and checked with `_registry_problems`. With `dry_run` the
     call returns here, having made every refusal the real run makes.
  6. **§3.6, the writes.**
     1. `_update_sidecar` adds `retired` (state `retiring`) and appends the one `retire` entry. Its
        `when` is the entry's `when`.
     2. `ShardedPool.delete_store(primary, resume=resume)`. `resume` is the value the plan was made
        with, so `retired.files` is exactly what is deleted.
     3. The check: no listed file exists (`_present`), and nothing outside the list was deleted.
     4. `_update_sidecar` sets state `retired` and `completed`, and appends nothing.

     A failure at steps 2–4 raises `_retire_failure`. It names the step, the error, the files
     still present, any `.tmp` left behind, and the remedy, `store retire` again. Nothing is
     cleaned up or retried. A failure at step 1 raises too, saying that nothing was deleted and
     what the sidecar now reads as.
  - **The completion path.**
    - It re-makes §3.1 and §3.2, the latter by the recorded `store_id` field.
    - It refuses a reason that differs from the recorded one.
    - While the primary exists, it plans with `resume=True`, refuses any planned file the
      tombstone does not list, and deletes with `resume=True`.
    - If the primary is gone, it refuses if any listed file remains. Such a file can then be
      removed only by a person, since nothing names it through the resolver.
    - Otherwise it goes straight to the check and the completion. It appends no history, and
      rewrites neither `files` nor `references`.
  - **The return value.** `{primary, sidecar, store_id, dry_run, completing, fingerprint, files,
    to_delete, deleted, references, tombstone, fields}`.
- **R8, the guards.** `_tombstone_text(reading)` says the sidecar "is a tombstone", when and why the
  store was retired, the reading's problems, and "A retired name is never reused". It is raised by:
  - `create_sidecar`, before its primary check;
  - `adopt_sidecar`, before "already a registry sidecar";
  - `_prepare`, for copy and move, from a tombstone and to a retired name, before the in-use check;
  - `fingerprint_store`, read-only or `--write`, before anything is read.
- The module docstring, the `_update_sidecar` docstring (which counts its callers) and the
  `fingerprint_store` docstring are updated. `ShardedPool` is imported inside `retire_store`.

**`RunRegistry/__init__.py`, `begin` only.** The `results` sidecar is now read **before**
`os.makedirs`. A tombstone, complete or not, raises `RuntimeError`: `Cannot begin run <id> with
results "<path>": <_tombstone_text>. Give the new store a name of its own. No run directory was
created`. `results_store_id` is taken from the same reading. The docstring gains one sentence.

**`RunRegistry/__main__.py`.**
- `store retire PRIMARY --reason TEXT [--without-fingerprint] [--dry-run] [--runs-root DIR]
  [--stores-root DIR]`. It prints the store and sidecar, then `references:`, `fingerprint check:`,
  `files deleted:` (or `files to be deleted:`) and `tombstone:`, in that order. A retirement, a
  completion and a dry run exit 0. A refusal or a failure prints `!! <message>` on stderr and exits
  1.
- `store show` on a tombstone prints `retirement:` first (state, when, reason, completed,
  fingerprint condition, files, references), then the sidecar as before, then the runs naming the
  store. The runs are found by the `store_id` field, as `_show` already did. It exits 1 when a
  tombstone's reading has a problem, and 0 otherwise.

**`RunRegistry/tests/test_store_retire.py`** (new): 41 test methods, many running their cases as
subtests, in about 40 s. It needs no Ray, and every store and run is in a temporary directory. Every
`retire_store` call goes through a helper that passes `runs_root` and `stores_root`, except
`test_roots_that_are_not_directories`, which passes both itself. Every `store retire` command
passes `--runs-root` and `--stores-root`, and every `store show` passes `--runs-root`.

| Class | Methods | Prompt §5 item |
|---|---|---|
| `TestRetirement` | 1 | **1.** See the item list below the table. |
| `TestLegacyShape` | 1 | **2.** B's `shards` rows name A's four existing shards by absolute path (asserted). After B is retired, `tree_state(A)` (A's four shards, primary and sidecar) is unchanged. A still reads `ok`, and every file deleted is in B. |
| `TestRefusals` | 12 | **3.** A missing and a blank reason. An absent, unreadable, legacy and problem sidecar. A completed tombstone. An alive and a stale running run, and a run matching by `results_store_id` alone. No recorded fingerprint. A row changed after fingerprinting. A mismatch in `problems` alone (an orphan tag row; the precondition asserts the digests agree and only a `problems` entry differs). A journal beside a shard. A `.tmp` beside the sidecar. A runs root and a stores root that are not directories. Each asserts the fragments naming the reason, the message ending "Nothing was written or deleted", and `tree_state` of the whole temporary tree unchanged. The stale-run and no-fingerprint messages are asserted to name their remedies. |
| `TestInterruption` | 8 | **4.** The table below, one method per row. |
| `TestWithoutFingerprint` | 6 | **5.** A corrupt shard: refused without the flag; with it, retired, and `fingerprint` equals `{"condition": "without", "error": str(e), "error_type": …}` of the exception `read_inventory` raises. A store that can be fingerprinted, with and without a recorded fingerprint: refused. A running run, alive and stale, with the flag, on a store the flag would otherwise retire: refused. A journal: refused. A missing shard: refused without the flag; with it, retired, with `files` the four present, `files_present_only` true, and the error naming the missing file. An unreadable `shards` table (`DROP TABLE shards`), with and without the flag: refused with §3.3's message. |
| `TestDryRun` | 3 | **6.** A clean store: `tree_state` unchanged. Then the real run's `deleted` equals the dry `to_delete`, and `references`, `fingerprint` and the tombstone (without `when`, `state`, `completed`) are equal. Four refusals (no fingerprint, a stale run, a journal, a completed tombstone) give the same message dry and real, each with the tree unchanged. A dry run on an incomplete retirement reports what remains and changes nothing. |
| `TestNeverReused` | 3 | **7.** `begin(results=…)` on a completed and an incomplete tombstone raises, naming "tombstone", `when`, the reason and "never reused", and `tree_state` of the runs root is unchanged. Copy from, copy to, move from, move to, fingerprint (read-only and `--write`), adopt and create, each on both tombstones, raise the tombstone message, with the tree unchanged. A store written at the retired name (`write_new_store`) reads as one problem, "exists again", naming the primary; `store show` exits 1 and prints it after `retirement:`; retire and `begin` refuse. |
| `TestHistoryRule` | 3 | **8.** Problems for: an entry after `retire`, two `retire` entries, `retire` at index 0, `retire` with a `to`, `retire` with no `from`, and a `copy` and a `move` given `retire`'s `to` (`null`). No problems for create; create, copy, move; create, move, copy; create, copy, move, retire. Through the reader: a `retire` entry with no `retired`, `retired` with no `retire` entry, and five malformed `retired` values, each a problem. A well-formed completed tombstone whose primary exists reads "exists again". A legacy sidecar with a `retired` key reads as legacy, with no problems, `retired` false, and the key in `unknown_fields`. |
| `TestCommandLine` | 4 | **9.** Exit 1 on a blank reason, and 2 (argparse) with no `--reason`, both writing nothing. A dry run exits 0, writes nothing and prints `files to be deleted:`. A retirement exits 0 and prints the four sections in order, every file, the matched condition and `>> retired:`. `store show` on the tombstone exits 0, starts with `retirement:`, prints its parts in order before `sidecar:` and the runs, and loads neither `ray` nor `sqlalchemy`. A second retirement exits 1. An incomplete retirement: `store show` exits 1 and says "is incomplete"; `store retire` completes it and exits 0. A failure (in-process `main`, with `delete_store` patched to raise) exits 1 with `!! retire of store … failed at step`. A subprocess shows that `import RunRegistry` loads neither `ray` nor `sqlalchemy`. |

What `TestRetirement` sets up and checks (§5 item 1):
- **The stores.** The retired store, four shards and a primary under the stores root, fingerprinted.
  A second full store in the same directory. A full store outside the stores root, whose sidecar
  names the retired store's directory.
- **The references.** A finished run whose `results` names the store. A registry copy, whose
  `copied_from` names the `store_id`. A sidecar whose unknown `backup` field names the store's
  directory, both absolute and relative to the repository root (the live A3 shape). A sidecar that
  is not JSON.
- **The files.** The five files are gone.
- **The whole tree.** Compared by `tree_state`, everything else is identical: the second store,
  the one elsewhere, every other sidecar and the runs root. Only the sidecar and the store's own
  directory entry changed.
- **The tombstone.** A completed tombstone, whose history ends in one `retire` entry. The
  `fingerprint` field, and every other field, are value-identical through `load`, the writer's JSON
  round trip. `files` is the five in order, and the condition is `matched` with the recorded
  digest.
- **The report.** Exactly the copy (with `$.copied_from.store_id`), the backup sidecar (exactly
  `["$.backup.path", "$.backup.relative"]`) and the unreadable one. Not the store's own sidecar, the
  second store's, or the one outside the root. The report names both roots, and what was not
  searched.

## The `retired` shape and `retire`'s `to` rule, as shipped

```
retired = {
  "state":              "retiring" | "retired",
  "when":               now_iso() at the first write; the retire entry's "when",
  "git_head":           git_provenance() at the first write,
  "git_dirty":          bool,
  "reason":             the required text (D3),
  "fingerprint":        {"condition": "matched", "digest": <the recorded overall digest>}
                      | {"condition": "without", "error": str(e), "error_type": type(e).__name__},
  "files":              [_repo_path of each shard, ascending serial, then the primary],
  "files_present_only": bool; true only under "without", when a shard was already missing,
  "references":         {"runs_root", "stores_root",
                         "runs": [{"id", "state", "matched_by"}],
                         "sidecars": [{"sidecar", "fields": [JSON paths]} | {"sidecar", "unreadable"}],
                         "not_searched": text},
  "completed":          null while "retiring"; now_iso() once "retired"
}
```

**The `retire` history entry** is `{"operation": "retire", "from": <the primary's repository
path>, "to": null, "when", "git_head", "git_dirty"}`, appended once, at the first write. `to` is
`null` because a retirement has no destination. `_history_problems` requires `to is None` for
`retire`, and a non-empty string for every other operation, so a `copy` or `move` with a `null`
`to` is still a problem (`TestHistoryRule`, "copy with retire's to"). Both are stated in the
`stores.py` format table.

## The interruption table

The claim: *every state an interrupted retirement can leave reads, through `read_sidecar`, as
exactly one of a live store, untouched; an incomplete retirement, naming what remains; or a
completed tombstone. A second `store retire` then completes it.* The store is four shards and a
primary. The tombstone, with its file list, is written before the first unlink, and the primary
is deleted last (prompt 01), so no state leaves store files that no sidecar lists. `retired` is
written only after the check that no listed file exists, so no sidecar claims a completed
retirement while a listed file remains.

| Row | Interrupted | What is left | `read_sidecar` | A second `store retire` | Test |
|---|---|---|---|---|---|
| R0 | before the first write, at any refusal | the store and its sidecar, byte-identical | the live store, `ok` | retires it once the refusal's cause is gone | every `TestRefusals` method; `TestDryRun` |
| R0a | the first write fails (`_update_sidecar` raises on its first call) | the live store, byte-identical, and **no `.tmp`** | the live store, `ok` | retires it | `TestInterruption.test_I0_the_tombstone_write_fails` |
| R0t | the first write is killed after writing its `.tmp`, before `os.replace` | the live store, and `<stem>.manifest.json.tmp` holding the `retiring` tombstone | the live store, `ok` | **refuses, naming the `.tmp`**, and changes nothing. Once a person removes it, it retires | `…test_I0t_the_tombstone_write_is_killed_and_leaves_its_tmp` |
| R1–R5 | after the tombstone, when the *n*th unlink fails (*n* = 1…5: shard #0 … shard #3, the primary) | the `retiring` tombstone and files *n*…5 | an incomplete retirement naming exactly files *n*…5 | refuses a different reason, then completes with `delete_store(resume=True)` | `…test_I1_to_I5_os_unlink_fails_before_the_nth_deletion` (*n* = 1…5) |
| R5′ | the deletion returns, but a listed file (the primary) remains | the `retiring` tombstone and the primary | an incomplete retirement naming the primary | refuses a different reason, then completes | `…test_I5_the_deletion_reports_success_and_leaves_a_listed_file` |
| R6 | every file is deleted, before the completion write: the last unlink takes effect and then raises, or `_update_sidecar` fails on its second call | the `retiring` tombstone and no store file | an incomplete retirement, "no listed file remains" | refuses a different reason; with no primary it checks that no listed file remains, then completes | `…test_I6_os_unlink_fails_after_deleting_the_primary`, `…test_I6_the_completion_write_fails` |
| R6t | the completion write is killed after writing its `.tmp` | the `retiring` tombstone, no store file, and a `.tmp` holding the `retired` one | an incomplete retirement, "no listed file remains" | refuses a different reason; **refuses, naming the `.tmp`**, and changes nothing. Once a person removes it, it completes | `…test_I6t_the_completion_write_is_killed_and_leaves_its_tmp` |
| R7 | not interrupted | the `retired` tombstone | a completed tombstone: no problems, `ok` false, `store_id` `None` | refuses: already retired, when and why | `TestRetirement.test_a_retirement`, `TestRefusals.test_a_completed_tombstone` |

**Two states no interruption of `retire_store` can leave, each made by hand and each refused.**
- **A listed file reappears after the primary is gone** (R6, then a file is put back). This reads
  as an incomplete retirement naming it. `store retire` refuses: with no primary, nothing names it
  through the resolver, so only a person, outside the registry, can remove it
  (`…test_a_listed_file_that_remains_with_no_primary`).
- **A store is written at a completed tombstone's name.** This reads as "exists again".
  `store retire`, `begin` and every other operation refuse it
  (`TestNeverReused.test_a_store_written_at_the_retired_name`).

In every row from R1 on, the failure message names the step, the injected error, the files still
present and the remedy, `store retire` again. Each completion leaves what an uninterrupted
retirement leaves (`assertCompletedTombstone`):
- a completed tombstone, with one `retire` entry and every other field value-identical;
- the `retired` the first write recorded, `files` and `references` unchanged, with only `state` and
  `completed` new;
- the sidecar alone in the directory.

**What a `.tmp` left by a killed write does.** `write_json_atomic` writes `<sidecar>.tmp`, then
`os.replace`s it onto the sidecar. A process killed between the two leaves the `.tmp`, and the
sidecar holds its previous contents: the live store at R0t, the `retiring` tombstone at R6t. The
`.tmp` is at no sidecar name, so `read_sidecar` never reads it, and no state it leaves is a fourth
kind. It does block every later write: `_update_sidecar` refuses while it exists. So `retire_store`
checks for it itself, before anything is written or deleted, and refuses, naming it and saying
that a person looks at it and removes it. A write that fails by raising, which is what a
monkeypatched `_update_sidecar` does, leaves no `.tmp` (R0a, R6).

## The docstring changes (D0)

**`RunRegistry/stores.py`**, the operations paragraph. Before (`:43-49` at `ecfb024`):

> **The operations** are `create_sidecar`, `adopt_sidecar`, `copy_store`, `move_store` and
> `fingerprint_store`. Copy and move call `ShardedPool.copy_store` / `move_store` for the store's
> files, then carry the sidecar. All three refuse a store that any `running` run names, alive or
> stale; that check is what the bare script cannot make. It is a check and not a lock: a run begun
> after it is not seen. Nothing here deletes a file, overwrites one (except the three in-place
> updates `_update_sidecar` names: adopt, the move's temporary sidecar, and recording a
> fingerprint), cleans up after a failure, or decides that a run is over.

After:

> **The operations** are `create_sidecar`, `adopt_sidecar`, `copy_store`, `move_store`,
> `fingerprint_store` and `retire_store`. Copy and move call `ShardedPool.copy_store` / `move_store`
> for the store's files, then carry the sidecar. Copy, move, fingerprint and retire refuse a store
> that any `running` run names, alive or stale; that check is what the bare script cannot make. It
> is a check and not a lock: a run begun after it is not seen. Every operation but a retirement's
> completion refuses a tombstone.
>
> `retire_store` alone deletes, and only a store's own files: the shards its primary names, read
> through the one resolver, and then the primary, through `ShardedPool.delete_store`. It keeps the
> sidecar, as the store's tombstone, and it deletes no sidecar, run directory or other record.
> Nothing else here deletes a file. Nothing overwrites one except the in-place updates
> `_update_sidecar` names (adopt, the move's temporary sidecar, recording a fingerprint, and a
> retirement's two writes), and nothing cleans up after a failure or decides that a run is over.

**The format table**, before, for the two rows that changed (`history`, `fingerprint`):

```
    history         append-only, one entry per operation, the first being the create or adopt
                    that assigned the store_id: {"operation", "from", "to", "when", "git_head",
                    "git_dirty"}
    fingerprint     ... Copy and move carry it verbatim,
                    `taken` included, because neither changes the content it describes
```

After: `history` adds "After it, `copy` and `move`; and `retire`, which may stand only as the last
entry, and only once. A `retire` entry's `from` is the primary's path and its `to` is null, because
a retirement has no destination; every other entry's `to` is a path". `fingerprint` adds "A
retirement keeps it unchanged: it is what says what the retired store held". A new `retired` row
gives the key-by-key shape above. After the table comes a new paragraph, **A tombstone**, saying
that its `ok` is false, and which tombstones are problems. The import paragraph now names
`retire_store` among the functions that import `ShardedPool`.

**`RunRegistry/__main__.py`**, the module docstring. Before (`:6-13` at `ecfb024`):

> `python -m RunRegistry store {show,create,adopt,copy,move,fingerprint}` manages a datastore and
> its `<stem>.manifest.json` sidecar (`RunRegistry.stores`). `show` is read-only. `create` and
> `adopt` write a sidecar and never open the store. `copy` and `move` move the store's files with
> `ShardedPool` and carry the sidecar, and refuse a store that any `running` run names, alive or
> stale. `fingerprint` reads the closed store read-only, compares its content fingerprint with the
> one the sidecar records, and writes it into the sidecar only with `--write`; it refuses a store a
> `running` run names, and `--listing` writes the full listing to a new file away from the store.
> None of them initialises Ray, and none deletes anything.

After:

> `python -m RunRegistry store {show,create,adopt,copy,move,fingerprint,retire}` manages a
> datastore and its `<stem>.manifest.json` sidecar (`RunRegistry.stores`). `show` is read-only.
> `create` and `adopt` write a sidecar and never open the store. `copy` and `move` move the store's
> files with `ShardedPool` and carry the sidecar, and refuse a store that any `running` run names,
> alive or stale. `fingerprint` reads the closed store read-only, compares its content fingerprint
> with the one the sidecar records, and writes it into the sidecar only with `--write`; it refuses a
> store a `running` run names, and `--listing` writes the full listing to a new file away from the
> store. `retire` deletes a closed store's own files, its shards and then its primary, through
> `ShardedPool.delete_store`, and keeps its sidecar as a tombstone that says when, why and what the
> store held; it refuses a store a `running` run names, alive or stale, and one whose recorded
> fingerprint does not match its content, and `--dry-run` reports what it would do and does nothing.
> None of them initialises Ray. `retire` alone deletes, and only a store's own files; none deletes a
> sidecar, a run directory or any other record, and every other command refuses a tombstone.

## Deviations from the prompt

1. **`store show` exits 1 only for a tombstone with a problem.** `IMPLEMENTATION CHOICE`. §4 says
   "1 when the reading has a problem, as today". But `_show` exits 0 for every reading today,
   problems included. The prompt's "as today" is inaccurate. So the new rule is limited to what
   the prompt adds: 1 for an incomplete or reappeared tombstone, 0 for a completed one. Every other
   sidecar exits 0, exactly as before. Making `show` exit 1 on any problem would change existing
   behaviour, which no prompt asked for.
2. **`--reason` is `required=True` in argparse, so a missing one exits 2, not 1.** `IMPLEMENTATION
   CHOICE`. It follows D3's own precedent, `--purpose … required=True` on create and copy. A
   missing flag is a usage error, which argparse reports with 2 and which writes nothing. A blank
   reason reaches `retire_store` and is refused with exit 1. Both are tested.
3. **`retire_store` refuses a `.tmp` beside the sidecar before anything is deleted.**
   `IMPLEMENTATION CHOICE`. §3 does not list it. Without it, the completion write's
   `_update_sidecar` would refuse only *after* the deletion. The files would be gone, and the
   tombstone would stay `retiring` for a reason the first check could have named.
4. **A stores root that is not a directory is refused.** `IMPLEMENTATION CHOICE`, like the runs
   root's check that copy, move and fingerprint make. D5's report must say what was searched, and
   an absent root would have searched nothing silently.
5. **Extra checks after the plan, on both paths.** `IMPLEMENTATION CHOICE`.
   - The completion path refuses a planned file that the tombstone does not list.
   - Step 3 also fails if `delete_store` returned a file not in the list.

   The list recorded before the deletion is the list deleted, and these check that it stays so.
6. **The references match string values, not object keys.** `IMPLEMENTATION CHOICE`. §3.4 says
   "any string anywhere in it". Keys are field names, and a key that is a path would be a sidecar
   no writer here produces. Empty and blank strings are never matched. `_resolve("")` would be the
   repository root.
7. **The `without` condition records `error_type` beside `error`.** `IMPLEMENTATION CHOICE`.
   `error` is `str(e)`, verbatim as D4 asks. The type is kept beside it, because `str(e)` of a
   `sqlalchemy` error does not say what raised it.
8. **`SidecarReading.unknown_fields` keeps `retired` for a legacy reading.** `IMPLEMENTATION
   CHOICE`. Adding `retired` to `KNOWN_FIELDS` would otherwise have dropped it from a legacy
   sidecar's unknown fields, against §2.3's "read as it is now: an unknown field, uninterpreted".
   Nothing outside the tests reads `unknown_fields`.
9. **A test beyond §5's list, `TestWithoutFingerprint.test_a_running_run_is_still_refused`.**
   `IMPLEMENTATION CHOICE`, added after mutation (viii)'s first run. See the deliberate-breakage
   record, (viii).
10. **`python -m RunRegistry list` was not run at the start.** `IMPLEMENTATION CHOICE`, against
    `CLAUDE.md`'s discovery rule 1. The dispatch said to open nothing under `var/`, and the
    orchestrator runs `list` before dispatch (orchestrator notes §1.2). Nothing long was launched.

None is `UNINTENDED DRIFT`.

## Verification performed

- **Baselines on `ecfb024`**, measured by the orchestrator in this checkout: AdaptiveLevin 32,
  ComputeTargets 552, CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped),
  RunRegistry 128.
- **The new module:** `Ran 41 tests … OK`, run on its own and within the suite, several times.
- **All six suites after the change**, run in the checkout from the repository root with
  `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`:

  | Suite | Before | After |
  |---|---|---|
  | `AdaptiveLevin` | 32 OK | 32 OK |
  | `ComputeTargets` | 552 OK | 552 OK (the flake did not occur) |
  | `CosmologyModels` | 39 OK | 39 OK |
  | `Datastore` | 206 OK | 206 OK |
  | `LiouvilleGreen` | 148 OK, skipped=1 | 148 OK, skipped=1 |
  | `RunRegistry` | 128 OK | 169 OK (128 + the 41 tests prompt 03 added) |

- **`black --check`** is clean on `RunRegistry/` (13 files).
- **Every existing test passes unmodified.** `git diff --cached --stat` shows no existing test
  changed, and `store_fixtures.py` is untouched.
- **Scope.** The commit touches `RunRegistry/stores.py`, `RunRegistry/__init__.py` (every hunk
  inside `begin`), `RunRegistry/__main__.py`, the new test module, this log, the board and
  `docs/OPEN_ISSUES.md`. `Datastore/`, `tools/`, `CLAUDE.md`, the rest of `docs/` and every
  existing test are unchanged. The untracked `docs/datastore-integrity-audit*` and
  `prompts/datastore-integrity/` were not read, staged or changed, and no `orch_*` file was
  touched.
- **Roots.** Every `retire_store` call in the module goes through `RetireTestCase.retire`, which
  passes both roots, except `test_roots_that_are_not_directories`, which passes both explicitly.
  Every `store retire` command passes `--runs-root` and `--stores-root`, and every `store show`
  passes `--runs-root`. Checked by `grep` over the module.
- **No fallback to a naming rule.** `retire_store` names files only through
  `ShardedPool.closed_store_files` and `delete_store`. The diff has no `shard_file_name`, and the
  only `glob`-like search is the sidecar walk. The test module uses `shard_file_name` to build its
  expectations.
- **The non-test uses of `.ok` and `.store_id`**, each checked for a tombstone:
  - `Run._fingerprint_results` passes `write=reading.ok`, which is false. `fingerprint_store`
    refuses the tombstone, so `finish` records a `fingerprint_error` and the run still ends.
    `begin` refuses such a store in the first place.
  - `_prepare` refuses a tombstone, with its own message, before testing `ok`.
  - `fingerprint_store --write` is refused before `ok` is reached.
  - `begin` refuses a tombstone before it would take `store_id`, which would be `None`.
- **`retire_store`'s callers:** `__main__._retire` and the tests only.
- **`import RunRegistry` and `store show` load neither `ray` nor `sqlalchemy`.** Tested by
  subprocess (`TestCommandLine`), and `python -c "import RunRegistry, sys; print('ray' in
  sys.modules, 'sqlalchemy' in sys.modules)"` prints `False False`.
- **Nothing under `var/`** was opened, listed or run against. Scratch files were only
  `impl_03_`-prefixed files in the session scratchpad: two probes, which built stores in their
  own `TemporaryDirectory`, the mutation generator, the diffs, the run outputs and an empty working
  directory.

## The deliberate-breakage record

Each diff below is exactly as applied. Each was made by editing the file and taking `git diff`
against the index, where this commit's three source files were staged, and then restored. Before
each run, `git apply --check` accepted it. The run was `git apply`, then the whole `RunRegistry`
suite, then `git apply -R`. Each was run from an empty scratch working directory, with
`PYTHONPATH` and `-s`/`-t` absolute paths into the checkout:

```bash
PYTHONPATH=$R $R/venv/bin/python -m unittest discover -s $R/RunRegistry/tests -t $R
```

After each reversal, `git diff --stat -- RunRegistry/` was empty and the working directory still
held nothing. Unmutated, that run is `Ran 169 tests … OK`. Every failure below is in
`test_store_retire.py`; no other module's test is affected by any mutation. After the commit, each
diff was extracted from this file and `git apply --check`ed against `HEAD` (§ "State handed to the
next prompt").

**(i) Delete before writing the tombstone.** `FAILED (failures=18, errors=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..16aace3 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1837,6 +1837,14 @@ def retire_store(
     if dry_run:
         return result
 
+    deleted = []
+    if planned:
+        step = "delete the store's files"
+        try:
+            deleted = ShardedPool.delete_store(primary, resume=resume)
+        except Exception as e:
+            raise _retire_failure(primary, step, e, files, reading.path) from e
+
     # 6. the writes. Step 1, the tombstone, is already on disk on the completion path
     if not completing:
         step = "write the tombstone"
@@ -1857,14 +1865,6 @@ def retire_store(
                 + ". Once the failure is understood, run `store retire` again"
             ) from e
 
-    deleted = []
-    if planned:
-        step = "delete the store's files"
-        try:
-            deleted = ShardedPool.delete_store(primary, resume=resume)
-        except Exception as e:
-            raise _retire_failure(primary, step, e, files, reading.path) from e
-
     step = "check that no listed file remains"
     remaining = _present(files, primary)
     unlisted = [str(p) for p in deleted if _repo_path(p) not in files]
```

**Outcome.** An interrupted deletion now leaves the live sidecar, with no tombstone, beside a
store with shards missing. That is "store files with no sidecar listing them": the reader calls it
a live, problem-free store whose shards are gone.
- `TestInterruption`: `test_I1_to_I5_os_unlink_fails_before_the_nth_deletion`, all five subtests,
  and `test_I6_os_unlink_fails_after_deleting_the_primary`. In each, `reading.retired` is false
  where an incomplete retirement is required.
- `test_I0_the_tombstone_write_fails` and `test_I0t_the_tombstone_write_is_killed_and_leaves_its_tmp`:
  the files were deleted before the failed write, so the tree is not unchanged.
- `TestNeverReused.test_every_other_operation_refuses_a_tombstone`, the incomplete case, all eight
  operations. There was no tombstone to refuse, so each gave its ordinary refusal.
- `TestNeverReused.test_begin_refuses_a_retired_store_and_creates_no_run_directory`, the
  incomplete case (error).
- `TestDryRun.test_an_incomplete_retirement` (error).
- `TestCommandLine.test_a_completion_and_show_on_an_incomplete_retirement` and
  `test_a_failure_exits_one`.

**(ii) Skip the fingerprint comparison.** `FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..d3dbc99 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1764,7 +1764,7 @@ def retire_store(
                     f"({type(e).__name__}: {e}). If the store cannot be fingerprinted at all, "
                     f"`--without-fingerprint` retires it and records this error"
                 ) from e
-            differences = list(taken["comparison"])
+            differences = []
             if differences:
                 raise refuse(
                     f"its content does not match the fingerprint its sidecar records, taken "
```

Failed, each with `RuntimeError not raised`, because the store was retired:
- `TestRefusals.test_a_row_changed_after_fingerprinting`;
- `TestRefusals.test_a_mismatch_in_problems_alone`.

**(iii) Compare the fingerprint ignoring `problems`.** `FAILED (failures=1)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..954e506 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1764,7 +1764,7 @@ def retire_store(
                     f"({type(e).__name__}: {e}). If the store cannot be fingerprinted at all, "
                     f"`--without-fingerprint` retires it and records this error"
                 ) from e
-            differences = list(taken["comparison"])
+            differences = [d for d in taken["comparison"] if d["kind"] != "problems"]
             if differences:
                 raise refuse(
                     f"its content does not match the fingerprint its sidecar records, taken "
```

Failed: `TestRefusals.test_a_mismatch_in_problems_alone`. The store, whose digests match and
whose problems do not, was retired.

**(iv) Leave `begin`'s check below `os.makedirs`.** `FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index d96ba49..4584e3d 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -541,6 +541,7 @@ def begin(
     """
     identifier = run_id(campaign, prompt, slug, when)
     path = os.path.join(root or DEFAULT_ROOT, identifier)
+    os.makedirs(path, exist_ok=True)
     # the results store's sidecar is read before anything is created: a retired store is refused
     # (store-retirement README D6), complete or not, and no run directory is left behind
     results_store_id = None
@@ -556,7 +557,6 @@ def begin(
                 f"directory was created"
             )
         results_store_id = reading.store_id
-    os.makedirs(path, exist_ok=True)
     if os.path.exists(os.path.join(path, "manifest.json")):
         raise FileExistsError(
             f"{path} already holds a manifest; a manifest is written once"
```

Failed: `TestNeverReused.test_begin_refuses_a_retired_store_and_creates_no_run_directory`, both
the completed and the incomplete case. `begin` still raises, but `tree_state` of the runs root
gains the empty directory `run-registry-01-reuse-completed-…`.

**(v) Let the reader accept a reappeared primary.** `FAILED (failures=1, errors=1)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..d7bef71 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -448,12 +448,6 @@ def _tombstone_problems(
             f"reason>` again to complete it"
         )
         return [text], text
-    if present:
-        return [
-            f"retired {retired['when']}, but {names} exists again. This sidecar describes the "
-            f"store that was retired, not whatever now sits at its name; a process opened the "
-            f"retired path without the registry, and a retired name is never reused"
-        ], None
     return [], None
 
 
```

Failed:
- `TestNeverReused.test_a_store_written_at_the_retired_name` (error: the reading has no problem to
  unpack);
- `TestHistoryRule.test_the_tombstone_and_its_history_agree`: its last case, a well-formed
  completed tombstone with the primary present, reads with no problems.

**(vi) Let `--without-fingerprint` go on when the fingerprint succeeds.** `FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..5e0c252 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1782,11 +1782,11 @@ def retire_store(
                     "error_type": type(e).__name__,
                 }
             else:
-                raise refuse(
-                    f"it can be fingerprinted, so --without-fingerprint does not apply. Run "
-                    f'`python -m RunRegistry store fingerprint "{primary}" --write`, then `store '
-                    f"retire` without the flag"
-                )
+                condition = {
+                    "condition": "without",
+                    "error": "the fingerprint succeeded",
+                    "error_type": "none",
+                }
 
         # 4. the references
         if not os.path.isdir(stores_root):
```

Failed: `TestWithoutFingerprint.test_a_store_that_can_be_fingerprinted_is_refused`, both
subtests, with and without a recorded fingerprint (`RuntimeError not raised`).

**(vii) Search references by `copied_from` only.** `FAILED (failures=1)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..e199a71 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1534,11 +1534,6 @@ def _find_references(primary, planned, store_id, runs_root, stores_root, own) ->
             parent = payload.get("copied_from") if isinstance(payload, dict) else None
             if isinstance(parent, dict) and parent.get("store_id") == store_id:
                 matched.append("$.copied_from.store_id")
-            matched += [
-                where
-                for where, text in _strings(payload)
-                if _names_one_of(text, targets)
-            ]
             if matched:
                 sidecars.append({"sidecar": _repo_path(path), "fields": matched})
 
```

Failed: `TestRetirement.test_a_retirement`. The report lists the copy, whose `copied_from` names
the `store_id`, and the unreadable sidecar. It does not list the sidecar whose `backup` field names
the store's directory, the live A3 shape.

**(viii) Exclude stale runs from the in-use check.** `FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..06f1a1b 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1682,7 +1682,11 @@ def retire_store(
             f'the runs root "{runs_root}" is not a directory, so whether a running run names '
             f"this store cannot be checked"
         )
-    running = _running_runs_naming("retire", [primary], store_id, runs_root)
+    running = [
+        entry
+        for entry in _running_runs_naming("retire", [primary], store_id, runs_root)
+        if entry["liveness"] != "stale"
+    ]
     if running:
         raise refuse(
             f"{_describe_running(running)}. A stale run is ended by a person, with Run.finish, "
```

Failed:
- `TestRefusals.test_a_stale_running_run_and_its_remedy`. Without the flag the stale run is still
  refused, but by `fingerprint_store`'s own running-run check, surfacing as "a fresh fingerprint
  could not be taken". The message does not name "A stale run is ended by a person, with
  Run.finish".
- `TestWithoutFingerprint.test_a_running_run_is_still_refused`, the stale subtest (`RuntimeError
  not raised`). **The store was retired.** Under the flag, `fingerprint_store`'s refusal of the
  stale run is caught as the error that prevents a fingerprint, and the retirement goes on.

The first run of this mutation, before that test existed, failed only the first test. That run
was on the same source, with 168 tests and `FAILED (failures=1)`. So §5's list alone would have
caught (viii) only by a message. It would have missed the one case in which the mutation deletes
a store a stale run names. The test was added (Deviations, item 9), and every mutation was then
re-run. The results above are all from that second round.

**(ix) Mark the tombstone `retired` before checking that no listed file remains.**
`FAILED (failures=1)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0ab2d7b..2e15d1d 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1865,6 +1865,15 @@ def retire_store(
         except Exception as e:
             raise _retire_failure(primary, step, e, files, reading.path) from e
 
+    step = "mark the tombstone retired"
+    new_fields = _copy.deepcopy(new_fields)
+    new_fields["retired"]["state"] = "retired"
+    new_fields["retired"]["completed"] = now_iso()
+    try:
+        _update_sidecar(reading.path, new_fields, primary)
+    except Exception as e:
+        raise _retire_failure(primary, step, e, files, reading.path) from e
+
     step = "check that no listed file remains"
     remaining = _present(files, primary)
     unlisted = [str(p) for p in deleted if _repo_path(p) not in files]
@@ -1880,15 +1889,6 @@ def retire_store(
             reading.path,
         )
 
-    step = "mark the tombstone retired"
-    new_fields = _copy.deepcopy(new_fields)
-    new_fields["retired"]["state"] = "retired"
-    new_fields["retired"]["completed"] = now_iso()
-    try:
-        _update_sidecar(reading.path, new_fields, primary)
-    except Exception as e:
-        raise _retire_failure(primary, step, e, files, reading.path) from e
-
     result.update(
         deleted=[_repo_path(p) for p in deleted],
         tombstone=new_fields["retired"],
```

Failed: `TestInterruption.test_I5_the_deletion_reports_success_and_leaves_a_listed_file`. The
deletion says it succeeded and left the primary. The tombstone is then marked `retired` before the
check raises, so the reading is "retired …, but … exists again" where an incomplete retirement is
required (`incomplete_retirement` is `None`).

## Observations not acted on

1. **The `RunRegistry/__init__.py` module docstring still says the package "deletes nothing"**
   (`:5-9`, twice). This prompt could change only `begin` in that file. Opened as
   **[03-the-package-docstring-still-says-the-registry-deletes-nothing]** (board §3; index §1.14).
2. **§4's "as today" for `store show`'s exit code is inaccurate.** `_show` exits 0 on every reading
   at `ecfb024`. Handled as Deviations item 1. Not an issue.
3. **Prompt 01's plan refusal repeats its prefix.** It reads `Cannot delete sharded datastore "…":
   Cannot delete sharded datastore "…": its shards table could not be read …`, because
   `_plan_deletion` wraps `_read_closed_store`'s message, which starts the same way. `store retire`
   shows it inside its own "the plan of its files was refused". Cosmetic, and in `ShardedPool`,
   which this prompt does not change. Not an issue.
4. **The directory match lists any sidecar that names the store's directory.** For a store alone
   in its directory, as the backup is, that is D5's intent. The sweep store sits in
   `var/datastores/` itself, beside the live A3 store. So when prompt 05 retires it, any sidecar
   under the stores root with a string resolving to `var/datastores` is reported as referencing
   it. The report only lists; it never acts. Prompt 05's dry run shows what it finds. Not an
   issue.
5. **Under `--without-fingerprint`, `fingerprint_store`'s own refusals are indistinguishable from
   damage.** `retire_store` catches every exception the fingerprint raises as the D4 error. The one
   refusal that must not be caught that way, a running run, is refused first by §3.2. Mutation
   (viii) and `test_a_running_run_is_still_refused` pin that order. The other refusal it could
   raise, a tombstone, cannot reach it, because §3.1 refuses tombstones first. Not an issue.

## State handed to the next prompt

- **For prompt 04 (`store amend`):**
  - `SidecarReading.retired` and `_tombstone_text(reading)` are what the other operations use to
    refuse a tombstone, and amend should use them too.
  - `_history_problems` now accepts `copy`, `move` and a terminal `retire` after index 0. `amend`
    extends the same tuple. It must decide whether an `amend` may follow a `retire`; today nothing
    may.
  - `_registry_problems` requires `retired` if and only if the history ends in `retire`.
- **For prompt 05:**
  - `python -m RunRegistry store retire PRIMARY --reason TEXT --dry-run` shows everything a
    retirement would do and writes nothing. Its refusals are the real run's (`TestDryRun`).
  - The default roots are `var/runs/` and `var/datastores/`, and the report says what was not
    searched.
  - A retirement that fails part-way is completed by the same command, with the same reason.
- **The diffs above** were checked after the commit: each, extracted from this file, passes
  `git apply --check` against `HEAD`.
- **R6–R8 are done.** `ShardedPool`, `tools/sharded_store.py`, `CLAUDE.md` and every run manifest
  are unchanged.
- **Issue opened:** `[03-the-package-docstring-still-says-the-registry-deletes-nothing]`,
  unassigned.
