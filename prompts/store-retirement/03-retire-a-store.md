# Prompt 03 — retire a store: delete its files, keep its sidecar as the record

**Campaign:** [`README.md`](README.md) · **Board items:** **R6**–**R8** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompt 01 has landed. D3–D6 were decided on 2026-09-25 (README §6.2).
**Closes:** nothing. **Opens:** anything out of scope that you find (§7), **without fixing it**.
**Recommended model:** **Opus**. This is the campaign's one irreversible operation. The judgement
is in the order of the writes, and in making every state an interruption can leave say what it is.

**Read first:**

1. [`README.md`](README.md) in full, and **§6 in full**. §6.1 and §6.2 record the user's decisions.
   This prompt implements D3, D4, D5 and D6 **as worded**, and does not reopen them. §4 fixes the
   names you ship.
2. [`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) §1 and §2.
3. Prompt 01's log, `logs/01-delete-a-closed-store.md`: what `ShardedPool.closed_store_files`
   and `ShardedPool.delete_store` actually refuse, what they raise, and the interruption table.
   Build on what shipped. Where it differs from prompt 01's text, the log is right.
4. `RunRegistry/stores.py`, in full. Its docstring states the format and the rules you extend.
5. `RunRegistry/__init__.py` `begin` (`:482-582`), and `RunRegistry/__main__.py`.
6. `RunRegistry/tests/store_fixtures.py`, `test_store_sidecar.py`, `test_store_copy_move.py` and
   `test_store_fingerprint.py`. They show the patterns: stores in temporary directories, runs
   roots of your own, `tree_state`, and the real multi-shard fixture stores that can be
   fingerprinted.
7. `prompts/datastore-portability/03-the-registry-owns-the-store-sidecar.md` §2, for how the
   sidecar's writers keep a half-written file off a sidecar name.

---

## 1. What is wanted

`python -m RunRegistry store retire PRIMARY --reason TEXT` removes a closed store's primary and
shards, and keeps its sidecar, marked retired. The sidecar then says:
- when the store was retired, at what tree, and why;
- what it held, as its recorded fingerprint, which stays in the sidecar;
- which files were deleted;
- what referenced it at the time.

Every reference by path keeps resolving to it, because the name is never reused (D6).

The user's decisions, which this prompt implements and does not revisit:
- **6.1.2** The primary and shards go; the sidecar stays, marked retired.
- **6.1.3** Refuse a store that a `running` run names, alive **or stale**. Refuse unless a recorded
  fingerprint matches the current content.
- **D3** The reason is **required**.
- **D4** `--without-fingerprint` for a store that cannot be fingerprinted. It still needs the
  reason, still refuses a running run, and records the error that prevented the fingerprint,
  verbatim. A hot journal is refused even under it.
- **D5** Records of the past are never rewritten. `store retire` reports every reference it finds,
  before it deletes anything, and says what it did not search.
- **D6** A retired name is never reused. `begin(results=…)` refuses a retired store, the reader
  names a primary that has reappeared, and every other operation refuses a tombstone.

---

## 2. The format — R6

**2.1 The `retired` field.** Add `retired` to `KNOWN_FIELDS`, with a shape check in the style of
`_fingerprint_problems`. It holds, at the least (README §4):
- `state`: `retiring` while the deletion is under way, and `retired` once it has completed;
- `when`, `git_head`, `git_dirty`: `now_iso()` and `git_provenance()` at the first write;
- `reason`: the required text (D3);
- `fingerprint`: how the condition was met. Either `{"condition": "matched", "digest": <overall
  digest>}`, or, under D4, `{"condition": "without", "error": <the text of what prevented it>}`.
  The sidecar's own `fingerprint` field is **kept**, unchanged. It is what says what the store
  held;
- `files`: the files to be deleted, as `_repo_path` writes paths, shards in ascending serial and
  then the primary. It is written **before** anything is deleted;
- `references`: what §3.4 found, and the stores root it searched;
- `completed`: `now_iso()` when the state became `retired`, and `null` until then.

The exact key names and nesting are your choice. State them in the module docstring's format
table, and log the choice.

**2.2 The `retire` history entry.** `retire` becomes a history operation that may stand **only as
the last entry**. An entry of any kind after it, or a second `retire`, is a problem. It is appended
at the **first** write, when the state becomes `retiring`. Completing a retirement never appends
another. Its `from` is the primary's repository path. Its `to` has no destination to name: decide
what it holds (`null` is the natural choice), make `_history_problems` accept exactly that for
`retire` and nothing looser for the others, and log the choice. Copy and move are unchanged; they
refuse a tombstone anyway (§4).

**2.3 The reader.** `SidecarReading.retired` is true when a registry sidecar carries `retired`.
Then:
- **A completed tombstone is not a problem.** It is state `retired`, and neither the primary nor
  any file in `retired.files` exists. `_primary_problems` must not fire for it. `problems` is empty,
  and `ok` is **false**: `ok` means "a problem-free sidecar describing a store that is there", and
  every operation that tests it must refuse a tombstone.
- **An incomplete retirement is a problem.** It is state `retiring`, whatever files remain. The
  text names the files still present and the remedy, `store retire` again.
- **A reappeared store is a problem.** It is state `retired` while the primary or any listed file
  exists: *"retired …, but … exists again"*. That is audit §2.4, a process that opened the retired
  path without the registry. The text says the sidecar describes the store that was retired, not
  whatever now sits at its name.
- `store_id` returns `None` for a tombstone, as for any sidecar that is not `ok`.

Legacy sidecars cannot be retired (§3.1), so a legacy sidecar carrying a `retired` key is read as
it is now: an unknown field, uninterpreted.

---

## 3. The operation — R7

`retire_store(primary, reason, *, runs_root=None, stores_root=None, without_fingerprint=False,
dry_run=False) -> dict`, in this order. Every refusal is a `RuntimeError` that names the store and
the reason, and ends "Nothing was written or deleted".

**3.1 The sidecar.**
- Refuse an empty or blank reason (D3).
- Refuse an absent, unreadable or legacy sidecar, saying to `store create` or `store adopt` it
  first. Retirement never upgrades a sidecar, as copy and move never do.
- Refuse a registry sidecar with problems, **except** the one problem of an incomplete retirement.
  That is the completion path (§3.6).
- Refuse a completed tombstone: it is already retired, and the message says when and why.

**3.2 In use.** Refuse if any run whose state is `running`, alive **or stale**, names the store by
path or `store_id`: `_running_runs_naming("retire", …)`. The message says what copy and move say:
a stale run is ended by a person, with `Run.finish`, before its store is retired. It also names
README D4's first case: once the run is ended, `store fingerprint --write` records the fingerprint
that retirement needs.

**3.3 The files and the fingerprint.**
- Plan with `ShardedPool.closed_store_files(primary)`. Its refusals come through, a hot journal
  among them, with or without `--without-fingerprint` (D4).
- **Without the flag:** refuse if the sidecar records no fingerprint, naming `store fingerprint
  --write` as the remedy. Otherwise take a fresh one read-only (`fingerprint_store(primary,
  write=False, runs_root=…)`), and refuse unless the comparison is empty, **`problems`
  included**. A mismatch is listed in the refusal, entry by entry.
- **With the flag:** attempt the fingerprint all the same.
  - If it **succeeds**, refuse: the store can be fingerprinted, so the flag does not apply. The
    remedy is `store fingerprint --write`, then retirement without the flag. The flag stays
    narrow.
  - If it **fails**, keep its error text verbatim for the tombstone, and go on.
  - A store with a **missing shard** is refused by the plan. Under the flag only, plan and delete
    with `resume=True`, which leaves the missing shard out, and record in the tombstone that the
    list is of the files present.

**The one case D4's wording names that cannot be served: a `shards` table that cannot be read.**
Then nothing can say which files are the store's, and deleting by the naming rule would be a guess.
It is refused **even under the flag**. The message says so, and says that this store's files can be
removed only by a person, outside the registry. Do not add a fallback. The orchestrator has already
reported this narrowing of D4 to the user.

**3.4 The references (D5).**
- **Runs:** `runs_naming([primary], store_id, runs_root)`, each with its id, state and
  `matched_by`.
- **Sidecars:** every `*.manifest.json` under `stores_root` (default `DEFAULT_STORES_ROOT`, i.e.
  `var/datastores/`), recursively, excluding the store's own. A sidecar is listed when:
  - its `copied_from` names this `store_id`; or
  - any string anywhere in it, walked recursively, resolves by `_resolve` to the primary, to a file
    in the plan, or to the store's directory.

  Report each with the JSON path of every matching field. A sidecar that is not JSON is listed as
  unreadable and does not stop the operation. Never write another sidecar: correcting one is prompt
  04's `store amend`, run by a person.
- The report says which roots were searched, and that sidecars elsewhere, and references outside
  run manifests and sidecars (docs, boards, logs), were not.

For the backup today, the report must find the live A3 sidecar's `backup.path`. That is a test
fixture's shape (§5 item 1), not something to run.

**3.5 Dry run.** With `dry_run=True`, stop here. Return and print:
- what §3.1–§3.4 found;
- the tombstone that would be written, `when` aside;
- the files that would be deleted.

Nothing is written or deleted. A dry run makes every refusal a real run would make, and refuses
where the real run would. This is how prompt 05 shows the user the command before the user runs
it.

**3.6 The writes.** In this order:
1. **The tombstone.** Update the sidecar in place through `_update_sidecar`, adding `retired` in
   state `retiring`, with `files` and `references`, and appending the `retire` entry. Nothing has
   been deleted yet.
2. **The deletion.** `ShardedPool.delete_store(primary, resume=…)`. `resume` is `True` only on the
   completion path, or for §3.3's missing shard under the flag.
3. **The check.** None of `retired.files` exists.
4. **Completion.** Update the sidecar again: state `retired`, and `completed`.

**The completion path.** `store retire` on an incomplete retirement:
- refuses a reason different from the recorded one, as adopt refuses a different purpose, rather
  than ignoring it;
- re-makes §3.2's check, by the sidecar's recorded `store_id` as well as by path.
  `SidecarReading.store_id` is `None` for a sidecar that is not `ok`, so read the field. `store
  show` must do the same on a tombstone;
- deletes whatever of `retired.files` is still present, with `delete_store(resume=True)` while the
  primary exists. If the primary is already gone, it checks instead that no listed file remains;
- then completes. It appends no history entry, and rewrites no reference.

**On failure** at any step after step 1, raise. Name the step, the files still present, and the
remedy, `store retire` again. The tombstone stays `retiring`. Nothing is cleaned up or retried
inside the operation (README §5 rule 11).

**The interruption property.** Record it in the log as a table with one row per point of
interruption: *every state an interrupted retirement can leave reads, through `read_sidecar`, as
exactly one of:*
- a live store, untouched;
- an incomplete retirement, naming what remains;
- a completed tombstone.

*A second `store retire` then completes it.* No state may leave store files with no sidecar
listing them, or a sidecar that claims a completed retirement while any listed file remains.

**3.7 The command line.** `store retire PRIMARY --reason TEXT [--without-fingerprint] [--dry-run]
[--runs-root DIR] [--stores-root DIR]` (README §4). It prints:
- the references report;
- the fingerprint check;
- the files, deleted or to be deleted;
- the tombstone.

It exits 0 on a retirement, a completion or a clean dry run, and 1 on a refusal or a failure. A
plain `import RunRegistry`, and `store show`, still load neither `ray` nor `sqlalchemy`. Import
`ShardedPool` and the inventory inside the functions that need them, as copy and fingerprint do.

---

## 4. The guards — R8 (D6)

- **`begin(results=…)` refuses a retired store**, whether complete or incomplete, **before it
  creates the run directory**. Today `os.makedirs` comes first (`:542`), so the check must move
  above it. The message names the retirement's `when` and `reason`, and says a retired name is
  never reused.
- **Every other operation refuses a tombstone with a message that says it is one**, not the
  generic "not a problem-free registry sidecar". That covers copy and move from it, `store
  fingerprint`, and adopt. `create`, and copy and move *to* a retired name, already refuse because
  the sidecar name is taken (audit §2.3). Give them the tombstone message too where the reading
  shows one, and test all of them. Prompt 04's amend refuses one as well, and is not yours.
- **`store show`** on a tombstone prints the retirement first (state, `when`, reason, fingerprint
  condition, files, references), then the sidecar, then the runs naming the store. It exits 0 for
  a completed tombstone, and 1 when the reading has a problem, as today.
- **The docstrings follow D0.** Change the `RunRegistry/stores.py` module docstring ("Nothing here
  deletes a file …", `:47-49`) and the `RunRegistry/__main__.py` docstring ("none deletes
  anything", `:13`). Say that `retire` alone deletes, and only a store's own files, through
  `ShardedPool.delete_store`, keeping the sidecar. Add `retire` to both lists of operations. Update
  the format table.

---

## 5. Tests — a new module in `RunRegistry/tests/`, no Ray, nothing under `var/`

Use real multi-shard fixture stores that can be fingerprinted, in temporary directories, each with
its own runs root and stores root. At minimum:

1. **A retirement.** A fingerprinted store is retired.
   - Its five files are gone.
   - Its sidecar is a completed tombstone: `retired`, empty `problems`, `ok` false, `store_id`
     `None`. The history ends in one `retire` entry, and the `fingerprint` field is value-identical
     to before.
   - Every other field is value-identical, by the JSON round trip the writer uses.
   - The references report finds all three of:
     - a finished run whose `results` names the store;
     - a sidecar whose `copied_from` names its `store_id`;
     - a sidecar with an unknown field naming the store's **directory**, the live A3 `backup`
       shape.

   A second store in the same directory and one elsewhere are untouched, by `tree_state`, and so
   are the other sidecars and the runs root.
2. **The legacy shape.** A registry store whose primary names another populated store's shards by
   absolute path (the backup's shape). Retiring it leaves the other store's files, and sidecar,
   byte-identical.
3. **Refusals.** Each of these leaves the whole temporary tree unchanged:
   - no reason, and a blank reason;
   - an absent, a legacy and a problem sidecar;
   - a completed tombstone;
   - an alive running run, and a stale one;
   - no recorded fingerprint;
   - a mismatched fingerprint, from a row changed after fingerprinting;
   - a mismatch in `problems` alone;
   - a journal beside a shard.

   Each asserts that the message names the reason. The no-fingerprint and stale-run messages name
   their remedies.
4. **Interruption.** Monkeypatch `os.unlink` to fail on the *n*th call, and `_update_sidecar` to
   fail on its second call. For each point:
   - the reading is the table's row;
   - a second `store retire` completes it, leaving exactly the state of an uninterrupted
     retirement;
   - a second call with a different reason is refused.
5. **D4.**
   - A store with one corrupt shard file. Without the flag it is refused; with it, it is retired,
     and the tombstone carries the reader's error text verbatim.
   - A store that can be fingerprinted, with the flag: refused.
   - A journal, with the flag: refused.
   - A missing shard, with the flag: retired, deleting the files present.
   - An unreadable `shards` table, with the flag: refused, with §3.3's message.
6. **Dry run.** On a store that would retire cleanly, and on one that would be refused: the same
   answer as the real run, and `tree_state` unchanged.
7. **D6.**
   - `begin(results=<tombstone>)` raises, and no run directory is created.
   - Copy from, copy to, move from, move to, fingerprint, adopt and create, each on a tombstone,
     raise the tombstone message and change nothing.
   - A new store written at the retired primary's name makes the reading a problem, *"exists
     again"*, and `store show` reports it.
8. **The history rule.** An entry after `retire`, two `retire` entries, and a `retire` at index 0
   are each a problem. Copy and move histories validate as before.
9. **The command line.** Each exit code, the printed sections, and `store show` on a tombstone. A
   subprocess shows that `import RunRegistry` still loads neither `ray` nor `sqlalchemy`.

**Deliberate breakage.** Show that each of these makes the tests written against it fail, then
restore it:
- (i) delete before writing the tombstone;
- (ii) skip the fingerprint comparison;
- (iii) compare the fingerprint ignoring `problems`;
- (iv) leave `begin`'s check below `os.makedirs`;
- (v) let the reader accept a reappeared primary;
- (vi) let `--without-fingerprint` go on when the fingerprint succeeds;
- (vii) search references by `copied_from` only;
- (viii) exclude stale runs from the in-use check;
- (ix) mark the tombstone `retired` before checking that no listed file remains.

Put each in the log as a short diff, **exactly as applied**, for `git apply`. Name the tests that
failed. Mutations are never committed.

**No real store is touched.** Prompt 05 runs this on the real stores, with a dry run first and the
user running the real command.

---

## 6. Acceptance

1. §5's tests exist, need no Ray and open nothing under `var/`. Every existing test passes
   unmodified.
2. The deliberate-breakage record, (i)–(ix), with diffs and the tests that failed.
3. The interruption table, each row backed by a test.
4. The format table in the `stores.py` docstring includes `retired` and `retire`, with the key
   names and `retire`'s `to` rule as shipped.
5. `black --check` clean. On the board: the §1 row for 03, items R6–R8, and §5 baselines.
   `docs/OPEN_ISSUES.md` only if you open an issue. All in the same commit.

---

## 7. Stop conditions — stop and ask the user

- Prompt 01's methods cannot serve §3 as shipped. Then say what is missing: this prompt does not
  change `ShardedPool`.
- Any design would need to rewrite a run manifest, or write any sidecar but the store's own.
- An existing test would need to change. In particular, the reader change for tombstones would
  have to alter how an existing sidecar reads.
- `ok` cannot be false for a tombstone without changing what an existing caller does with a live
  store.
- Any step would open, fingerprint or retire anything under `var/`.

---

## 8. What this prompt does not do

- It does not retire any real store. That is prompt 05.
- It does not write `store amend`. That is prompt 04.
- It does not change `ShardedPool`, `tools/sharded_store.py` or any run manifest.
- It does not correct any reference it finds.

---

## 9. The log and the board

`logs/03-retire-a-store.md`, using the template in README §5.1. In addition:
- the `retired` shape and `retire`'s `to` rule, as shipped;
- the interruption table;
- the deliberate-breakage record, with diffs;
- the docstring changes, quoted before and after.

`IMPLEMENTATION_STATE.md`: the §1 row for 03, items R6–R8, and §5's baselines.
