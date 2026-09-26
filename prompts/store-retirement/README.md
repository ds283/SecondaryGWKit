# Campaign — store retirement

**Written:** 2026-09-25 at `42d4910` on `handover-remedial`, by Claude Opus 5.5, from
[`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) and the user's decisions
recorded in §6.1. **Prompts 01–05 are written.** The user approved D0 and D3–D7 as worded on
2026-09-25 (§6.2), which released 01, and 03 and 04 were written the same day. 05 was written the
same day too, once 01–04 had landed, against what they shipped (see §2). **Prompt 06 written**
2026-09-26 at `0075f72`, after 05's review, on the user's decision to fix the campaign's three
residuals before it closes (§6.4). **Closed** by the user on 2026-09-26, with all six prompts
landed; see the board.

## 0. Why this campaign exists

Two stores in `var/datastores/` hold nothing that is not held elsewhere (audit §3):
- the tolerance-sweep store, `handover-atol-sweep`;
- the pre-resume backup of the A3 baseline, in `backup-pre-resume-20260921T091011/`.

The user wants the registry kept tidy, and wants nothing with future value lost. The first answer
to "can they go?" was to delete the sweep store with `rm`. That answer was wrong about the *how*.
Every other event in a store's life now goes through `RunRegistry.stores` and leaves a `history`
entry: create, adopt, copy, move and fingerprint. A hand `rm` would be the one event the registry
cannot see. Then:
- four run manifests would name a path that answers nothing;
- the store's `store_id`, history and fingerprint would go with its sidecar;
- a missing store would look exactly like one moved or lost, which is the failure
  `datastore-portability` was built to catch.

So a store is removed by a **registry operation that retires it**. The store's own files go; its
sidecar stays behind, marked retired, and says when, why and what the store held.

The audit found that the tree is not yet ready for that operation:
- **The reader.** A sidecar whose primary is gone reads as a broken sidecar (§2.1).
- **The format.** The history format has no word for retirement, so no writer can record one
  (§2.2).
- **The sweep script.** `quadsource_atol_sweep.py --prepare` would overwrite the sweep store's
  tombstone, even without `--force` (§2.6).
- **A danger.** The backup's primary names the **live** store's shards by absolute path. A deletion
  that followed a stored record as a path would destroy the live A3 store (§2.7).
- **The charter.** Nothing in the tree deletes a file, and `CLAUDE.md` says the registry does not
  delete (§2.8).

**0.1 Correctness is the only objective.** Work is sequenced by epistemic dependency.
1. The deletion of a closed store's files comes first. It lives in `ShardedPool`, because only the
   one resolver there can say which files are a store's (§2.7).
2. The registry operation builds on it, with the format and the reader it needs.
3. Anything in the tree that would overwrite a tombstone is fixed before the tombstone it would
   overwrite exists.
4. The one present-tense claim a retirement falsifies is corrected through the registry, never by
   hand.
5. Only then are the two real stores retired.

## 1. Scope

**In scope:**
- `Datastore/SQL/ShardedPool.py`: a static deletion of a closed store's files, beside `copy_store`
  and `move_store`, and that block's comment (prompt 01);
- `docs/handover/quadsource_atol_sweep.py`, for `prepare()`, `assert_store_is_self_consistent`,
  the `--force` flag and the messages that name it, only (prompt 02). This discharges
  `datastore-portability`'s
  `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]` and
  `[01-atol-sweep-check-expects-absolute-shard-records]`, assigned here;
- `RunRegistry/stores.py`, `RunRegistry/__init__.py` (`begin` only) and `RunRegistry/__main__.py`:
  the retirement, the sidecar format and the reader it needs (prompt 03), and the amendment of
  one unknown field (prompt 04);
- `CLAUDE.md:52`, the registry's "does not … delete", **only as the user words it** (D0);
- the tests of all of these;
- the remedial retirement of the two stores, and the correction of the live A3 sidecar's `backup`
  field (prompt 05);
- the `datastore-portability` board, for closing the two issues assigned here;
- **added 2026-09-26 (§6.4):** the `RunRegistry/__init__.py` module docstring's first paragraph,
  and `amend_sidecar`'s identical-value comparison, with a new test module beside
  `test_store_amend.py` (prompt 06).

**Out of scope:**
- `ShardedPool`'s constructor, open and routing paths. It still knows nothing about sidecars
  (`datastore-portability` README §6.3), so a process that opens a retired path without the
  registry still creates a fresh store there (audit §2.4). What the registry does about it is
  prompt 03's reader check and `begin` refusal;
- `tools/sharded_store.py`, which gets no delete (D7);
- every existing run manifest. They are immutable (`datastore-portability` README §6.5 point 5);
- every other part of `quadsource_atol_sweep.py`, and its measurement code and results;
- the live A3 store, which stays;
- `var/runs/`: every run directory, the loose files and `var/runs/a3-pilot/BACKUP_PATH`;
- the three issues the audit opened (§3 of the board). They are recorded, not fixed.

## 2. Prompts

| # | Prompt | Covers | Status |
|---|---|---|---|
| 01 | [`01-delete-a-closed-store.md`](01-delete-a-closed-store.md) | `ShardedPool.closed_store_files` and `ShardedPool.delete_store`. Files named only through `_read_closed_store`. Refuses a hot journal or an unusable shard. Shards go first and the primary last. `resume=True` completes an interrupted deletion. Never follows a stored record as a path: the legacy-absolute case, whose records name another live store's shards, is the central test. | **written**; D0 decided 2026-09-25, so ready |
| 02 | [`02-the-sweep-prepares-through-the-registry.md`](02-the-sweep-prepares-through-the-registry.md) | `quadsource_atol_sweep.py` `prepare()` makes the sweep store with one `RunRegistry.stores.copy_store` call, replacing its hand copy, its `UPDATE` and its hand-written sidecar. `assert_store_is_self_consistent` compares through `resolve_shard_path`, which also unblocks `--build --resume` of the A3 v2 store (audit §2.6). Closes `datastore-portability`'s `[03-…]` and `[01-…]`. After it, `--prepare` at a name whose sidecar exists refuses, which is what a tombstone needs. | **written**; independent of every decision, and may be dispatched first |
| 03 | [`03-retire-a-store.md`](03-retire-a-store.md) | `RunRegistry.stores.retire_store` and `python -m RunRegistry store retire`. A known `retired` field and a terminal `retire` history entry. The reader tells a tombstone from a broken sidecar, and names a primary that has reappeared. `begin(results=…)` refuses a retired store. `store show` renders a tombstone. The references found are reported. | **written** 2026-09-25; after 01 |
| 04 | [`04-amend-an-unknown-field.md`](04-amend-an-unknown-field.md) | `RunRegistry.stores.amend_sidecar` and `store amend`: replace or remove one unknown field of a registry sidecar, with a reason, recording the old value in an `amend` history entry. | **written** 2026-09-25; after 03 |
| 05 | [`05-retire-the-two-stores.md`](05-retire-the-two-stores.md) | Remedial. **The user** retires the sweep store and the backup with `store retire`, and amends the live A3 sidecar's `backup` field. The prompt's agent checks before and after, and records the retirements on the boards. | **written** 2026-09-25, after 01–04 landed |
| 06 | [`06-fix-the-residuals-and-close.md`](06-fix-the-residuals-and-close.md) | The three open issues in this campaign's own code: the package docstring's "deletes nothing", `store amend`'s `==` comparison, and the missing marker-shaped amend test. Then the records that leave the campaign ready to close. The closure itself is the orchestrator's, after its review (§6.4). | **landed** 2026-09-26 (`211d2c4`), written after 05's review; reviewed, and the campaign closed, the same day |

**Why 03–05 were held.** The charters above are fixed, and cannot drift to fit what 01 and 02
find. What waited was their method, which depended on user decisions that did not yet exist
(§6.2). 03's shape depends on D3 (the reason), D4 (a store that cannot be fingerprinted), D5
(references) and D6 (reuse of a retired name). 04 exists only if D5 is taken as recommended.

**Released 2026-09-25.** The user approved D0 and D3–D7 as worded, the day they were recorded. So
03 and 04 were written the same day, against §4's names and §6.2. **05 stays held.** It runs the other four on
real stores, so it is written last, against what they ship.

**Order.** 02 is independent of 01, 03 and 04, and must land before 05 retires the sweep store
(audit §2.6). 01 must land before 03. 03 must land before 04, because both extend the history
vocabulary in the same function, and one extension at a time keeps each revertible.

## 3. Datastores

Three stores live under `var/datastores/`, each about 340 MB over four shards, and each with a
registry sidecar that carries a fingerprint taken on 2026-09-25 at `50a24ac` (audit §1).

**Prompts 01–04 never open anything under `var/`.** Their tests build stores in temporary
directories. **The one exception** is prompt 02. It reads the three real primaries' `shards` tables
`mode=ro`, through its new self-consistency check, to show that the check accepts their legacy
shape. It checks the files unchanged afterwards. It never runs `--prepare`, `copy_store` or
anything else that writes against a real store or name. **Only prompt 05 touches the real
stores**, and there the irreversible step is run by the user (§5 rule 10).

**Before touching any store,** run `python -m RunRegistry list` and confirm that nothing is
`running`.

## 4. The interfaces between prompts

The **names** below are fixed here, so that each prompt can be written against the one before it.
The internals are the implementing prompt's to design.

- **Prompt 01** ships two static methods on `ShardedPool`:
  - **`ShardedPool.closed_store_files(primary, *, resume=False) -> List[Path]`**: every file of
    the closed store, shards by ascending serial and then the primary, resolved through
    `_read_closed_store`. It refuses exactly as `delete_store` with the same `resume` would, and
    under `resume=True` it leaves a missing shard out of the list. *(Amended 2026-09-25, before
    dispatch: `resume` added, so that prompt 03 can record the list for a store with a missing
    shard under D4.)*
  - **`ShardedPool.delete_store(primary, *, resume=False) -> List[Path]`**: deletes the shards and
    then the primary, and returns what it deleted, in order. With `resume=True` it tolerates shards
    that are already gone, which is what an interrupted deletion leaves.
- **Prompt 02** ships no interface. `prepare()` calls `RunRegistry.stores.copy_store`.
- **Prompt 03** ships:
  - **`RunRegistry.stores.retire_store(primary, reason, *, runs_root=None, stores_root=None,
    without_fingerprint=False, dry_run=False) -> dict`**, and **`DEFAULT_STORES_ROOT`**
    (`var/datastores/`), the default place it searches for sidecars that reference the store;
  - the known sidecar field **`retired`**;
  - the history operation **`retire`**;
  - the reading property **`SidecarReading.retired`**;
  - **`python -m RunRegistry store retire PRIMARY --reason TEXT [--without-fingerprint]
    [--dry-run] [--runs-root DIR] [--stores-root DIR]`**. `--dry-run` makes every check and
    reports everything a retirement would do, and writes and deletes nothing. It is how prompt
    05's agent shows the user what the user is about to run. *(Amended 2026-09-25, when 03 was
    written.)*

  `retired` records, at the least:
  - when, the git head and whether the tree was dirty;
  - the reason;
  - how the fingerprint condition was met, with the digest checked or, under D4, the error that
    prevented one;
  - the files to be deleted, as repository paths, written before any is deleted;
  - the references found;
  - whether the deletion has completed.
- **Prompt 04** ships **`RunRegistry.stores.amend_sidecar(primary, field, reason, *, value=…,
  remove=False) -> dict`**, the history operation **`amend`**, and **`python -m RunRegistry store
  amend PRIMARY --field NAME (--json VALUE | --remove) --reason TEXT`**.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, unchanged except as D0 amends them, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified.** A prompt may add tests beside them.
8. **Deliberate breakage.** Each prompt names mutations that its tests must catch. Each is recorded
   in the log as a diff, exactly as applied, so that the orchestrator can replay it with
   `git apply`. Mutations are never committed.
9. **No test needs Ray, and no test opens anything under `var/`.** Stores for tests are built in
   temporary directories, and a test deletes only what it built.
10. **No agent deletes a real store.** Deleting a store under `var/` cannot be undone. In prompt 05
    the prompt's agent prepares and checks, the **user** runs each `store retire`, and the agent
    checks again afterwards. Every other prompt's deletions happen in temporary directories that
    the prompt's own tests created.
11. **Nothing is cleaned up after a failure.** Like copy and move, a failed deletion raises, naming
    the step and every file of the store still present. The remedy is `resume`, run deliberately,
    and never a retry inside the operation.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-09-25), and not reopened here

1. **Retire through the registry, never `rm`.** The registry operation comes before any store is
   removed, the sweep store included.
2. **What stays behind.** The primary and its shards are removed. The sidecar is kept and marked
   retired.
3. **What retirement refuses.** It refuses a store that a `running` run names, alive or stale. It
   also refuses unless the store has a recorded fingerprint that matches its current content, so
   that the tombstone proves what was deleted. The user asked what happens when a crash prevented
   the fingerprint; that is D4.
4. **Other references must end up indicating that the store has been removed.** How, is D5.

### 6.2 Decisions D0 and D3–D7 — approved by the user as worded (2026-09-25)

Each was stated with a recommendation, and the alternatives are kept below. **The user approved
all six as worded on 2026-09-25**, so every recommendation stands as the decision, and D0's
proposed wording is the text prompt 01 writes into `CLAUDE.md:52`, exactly. Prompt 02 depends on
none of them.

- **D0 — the charter. Decided: the proposed wording.** `CLAUDE.md:52` says the registry
  "does not schedule, supervise, restart, lock or delete". `datastore-portability` README §6.5
  re-affirmed that on 2026-09-24 (audit §2.8). Decision 6.1.2 needs the registry to delete a
  store's files, and a `CLAUDE.md` limit binds until the user changes it. **The wording, as approved:**
  "It records; it does not schedule, supervise, restart or lock. It deletes nothing but a store's
  own files, and those only through `store retire`, which a person runs and which leaves the
  store's sidecar behind as its record. It never deletes a run directory, a sidecar or any other
  record." The rationale for the limit, `run-registry` README §0 item 4, was a store that vanished
  with nothing on disk saying where it went. A retirement that keeps the sidecar is the opposite
  of that. `ShardedPool`'s "Deleting is for a person" (`:624`), the `RunRegistry/stores.py`
  docstring (`:47-49`) and the `RunRegistry/__main__.py` docstring (`:13`) are amended by prompts
  01 and 03 to match.
- **D3 — the reason. Decided: required, not optional.** The user asked whether an optional
  comment is worth adding. It is worth more than that. The fingerprint says *what* was deleted.
  Only a person can say *why*, and a tombstone without the why tells a later reader a store went,
  and not whether it was meant to go. The precedent is `purpose`, which `store create` and
  `store copy` already require (`RunRegistry/__main__.py`, `--purpose … required=True`).
  *Alternative:* optional, as asked. The tombstone then says "retired", and nothing more, whenever
  the reason is left out.
- **D4 — a store that cannot meet the fingerprint condition. Decided: an explicit
  `--without-fingerprint` for the one case that needs it.** Audit §2.10 separates three cases.
  - **The writer crashed before `finish`, so no fingerprint was written.** This is not manual. A
    person ends the stale run with `Run.finish("killed")`, runs `store fingerprint --write`, then
    `store retire`. The fingerprint must describe what is deleted, which is the content *now*. No
    new code is needed, and prompt 03 says so in its refusal message.
  - **The retirement itself was interrupted.** `store retire` again completes it, from the file
    list the tombstone recorded before anything was deleted.
  - **The store cannot be fingerprinted at all:** a shard missing or unreadable, or a `shards`
    table that cannot be read. Here, without an override, "manual" means `rm`, the thing this
    campaign replaces. `--without-fingerprint` still requires the reason. It still refuses a
    running run. It records in the tombstone the error that prevented the fingerprint, verbatim.
    A **hot journal** is refused even under it. That is not damage: it may mean the store is in
    use. Its remedy is for a person to open the file once, which rolls the journal back, and then
    fingerprint.

  *Alternative:* no override. An unfingerprintable store is then removed by hand, outside the
  registry, and leaves no tombstone.

  **Narrowed 2026-09-25, when prompt 03 was written, and reported to the user.** Of the three
  damage cases D4 names, a missing shard and an unreadable shard are served by the flag. A
  **`shards` table that cannot be read** is not. That table is the only list of a store's shard
  files, so without it nothing can say which files to delete. Deleting by the naming rule would be
  a guess, and a guess is how audit §2.7's danger happens. Prompt 03 refuses that case even under
  the flag, and says that such a store's files can be removed only by a person, outside the
  registry. The alternative, a fallback to `shard_file_name`, is not taken.
- **D5 — references. Decided: never rewrite a record of the past, correct a claim about the
  present, and report both.** This refines decision 6.1.4. The audit's reference table (§1) has
  two kinds of entry.
  - **Records of the past:** run manifests, `copied_from`, `history`, `status_files`, committed
    logs. They were true when written. Run manifests are immutable by an existing user decision
    (`datastore-portability` §6.5 point 5), and rewriting the past is what the provenance rule
    ("never re-stamped") forbids. They are **not rewritten**. Each still names the retired path,
    and the tombstone sits at that path, so following any of them lands on "retired, when, why,
    and what it held". That is how they come to "indicate that the store has been removed".
  - **Claims about the present:** today exactly one, the live A3 sidecar's
    `backup: {"retained": true, …}`. It is an unknown field, which the registry carries and never
    interprets. So what it should now say is a person's judgement, and the registry must give that
    person a way to write it. That way is prompt 04's `store amend`. It replaces or removes one
    unknown field of a registry sidecar, requires a reason, and records the old value in an
    `amend` history entry, so nothing is lost and a hand edit is never needed.
  - **`store retire` reports every reference it can find, before it deletes anything.** These are
    runs naming the store by path or `store_id`. They are also sidecars under `--stores-root`
    (default `var/datastores/`) whose `copied_from` names its `store_id`, or which hold any string
    that resolves to the primary, a shard or the store's directory. The report goes in its output
    and in the tombstone. It says honestly that sidecars outside the stores root are not searched.

  *Alternative:* `store retire` rewrites every reference it finds. That breaks the immutability of
  run manifests, and it cannot interpret an unknown field such as `backup` anyway.
- **D6 — a retired name is never reused. Decided.** A reference by path must keep resolving to
  the tombstone, never to an unrelated later store. Through the registry this holds already,
  because create, copy and move refuse a taken sidecar name (audit §2.3). Prompt 03 adds three
  things.
  - `begin(results=…)` refuses a retired store, which covers both registered builders (§2.5).
  - The reader calls "retired, but a primary exists again" a problem (§2.4).
  - Every other operation refuses a tombstone: copy, move, fingerprint, adopt and amend.

  An unregistered process can still create a store at the name, because `ShardedPool` knows
  nothing of sidecars (README §1, out of scope). The reader's problem is how that is caught
  afterwards.
- **D7 — the bare script gets no delete. Decided.** `tools/sharded_store.py` copies and moves
  with no sidecar (`datastore-portability` §6.2–6.3). A `delete` there would remove a store and
  leave its sidecar describing nothing, the broken case of audit §2.1. Only `store retire` calls
  `ShardedPool.delete_store`.

### 6.3 Choices the prompts make, where §6.1 leaves the method open

These are marked *prompt's choice* in the prompts, each with its reason. Each may be overridden by
a logged deviation that is at least as strong.
- (01) the shards are deleted before the primary, in ascending serial. The primary's `shards`
  table is then the list of what remains, until the last step;
- (01) `closed_store_files` and `delete_store` share one planning step, so the list `retire`
  records is the list that is deleted;
- (01) a store is refused if any file of it is a symbolic link, as copy and move refuse one;
- (02) `prepare()` keeps its `--force` flag only as a refusal that explains the new rule. A retired
  or existing sweep name is not replaced; a new name is chosen.

### 6.4 The close-out — decided by the user (2026-09-26)

After 05's review, seven issues were open on the board. Three of them are in this campaign's own
code:
- `[03-the-package-docstring-still-says-the-registry-deletes-nothing]`;
- `[04-amend-calls-true-1-and-1-0-identical]`;
- `[04-no-test-amends-a-value-shaped-like-the-marker]`.

The user was offered three ways to close:
- close at bookkeeping only, as `store-fingerprint` did (`42d4910`);
- fix the three in a separate prompt, then close;
- fix the three, then close.

**The user chose to fix the three, then close.** Prompt 06 fixes them and closes them on the
board's §4. The orchestrator records the closure after reviewing 06, in the commit that records
the review. The other four issues are not this campaign's code, and stay open, unassigned, for
their owners:
- `[00-a-sigterm-pipeline-run-is-recorded-as-failed]`;
- `[00-a-launch-log-lives-outside-its-run-directory]`;
- `[00-a-copy-carries-its-sources-present-tense-fields]`;
- `[01-cross-filesystem-move-advice-says-delete-by-hand]`.

## 7. Baselines

At `42d4910`, measured while writing this campaign:

| Suite | Result |
|---|---|
| Datastore | 177 OK |
| RunRegistry | 117 OK |

The other suites were last measured at `50a24ac` by the `store-fingerprint` prompt 05
orchestrator: AdaptiveLevin 32, ComputeTargets 552 (`test_tk_wkb_phase.TestCost.test_wall_time_per_object`
is a known wall-clock flake), CosmologyModels 39, LiouvilleGreen 148 (1 skipped). No code has
changed since.

Re-measure before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . 2>&1 | tail -40
```

The suites print banners, so `| tail -5` will not show the verdict. Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`.
