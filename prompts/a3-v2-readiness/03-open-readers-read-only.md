# Prompt 03 — open the readers read-only

**Campaign:** [`README.md`](README.md) · **Board items:** **O1**–**O4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 01 and 02 landed and reviewed. `git status` clean. One prompt at a time in this checkout.
**Closes:** `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` and
`[10-extraction-scripts-exit-1-on-a-store-without-every-model]`, both on the `datastore-integrity`
board's §4, with their index rows deleted.
**Opens:** whatever it finds that this prompt does not fix.
**Recommended model:** **Opus**. A second mode through `ShardedPool`, `Datastore` and six scripts. The
mode is only worth having if nothing written at open or in a lookup escapes it.

**Read first:**

1. [`README.md`](README.md): §0 items 2 and 5, §1, §4 (the names this prompt ships), §5, §6.1 D3.
2. [`docs/a3-v2-readiness-audit.md`](../../docs/a3-v2-readiness-audit.md): **R1**–**R3**, **M1**,
   **M2**, **V2**'s last paragraph, and the closing section's *03*. **M1's table is this prompt's
   specification of what happens at open.** Do not re-derive it. Where this prompt and M1 differ,
   this prompt says so.
3. The two board entries this prompt closes, on `prompts/datastore-integrity/IMPLEMENTATION_STATE.md`
   §3.3, with their **Widened** lines.
4. Prompt 02's log, and its result in `Datastore/SQL/ShardedPool.py` and `Datastore/SQL/Datastore.py`:
   the pool now writes the version row and the replicated prune, and actors take `set_version`.
   Read `ShardedPool._find_version_row`, `_prune_replicated_tables` and
   `_complete_interrupted_prune`, and `Datastore.__init__`'s `replicated_tables` argument (added
   2026-10-03, with the orchestrator note).
5. `Datastore/store_reader.py` in full: `read_only_url`, `_refuse_journals`, `open_read_only`.
   This prompt **reuses** its refusals rather than writing a second set.
6. The six `extract_*.py` scripts, `extract_common.py`, and `config/model_list.build_model_list`.
7. `docs/a3-v2-readiness/reader_writes_probe.py` and `readonly_open_probe.py`: the instruments, and
   QSI's lookup sequence transcribed.

---

## 1. What is wanted

A reader writes to the store it reads. It commits and clears an in-flight record on every
replicated `object_get`, hit or miss: 42 commits for the QSI reader on a full store. On a miss of
five classes it inserts a row on every shard; a re-inserted tolerance takes a new serial (audit R2).
Given a path where no store exists, it creates one (M1). The user decided (D3) that readers stop
writing by a read-only mode, not by per-site lookups, because only a store opened `mode=ro`
proves that a reader did not write.

Separately, every reader exits 1 at the `QCD_Cosmology` background lookup on a store that holds
only LambdaCDM's background. A3 v2's store is that kind of store (R3).

---

## 2. What to change

**O1 — `ShardedPool(read_only=True)`.** The keyword is README §4's. It is passed to every `Datastore`
actor. Under it, at open, exactly M1's table:
- **No store is created.** A missing primary is refused before anything is made, `mkdir` included.
- **Journals are refused by presence.** A `-journal` (or `-wal`, `-shm`) beside the primary or any
  shard is refused before any engine is made, by `store_reader`'s rule and through its code. An
  empty journal counts: SQLite does not treat it as hot, so the pool must refuse it itself (M2d).
- **Every engine is opened `mode=ro`.** That means the pool's own engine and every actor's,
  through `store_reader.read_only_url`. The check at open's step-1 read-write `sqlite3.connect`
  does not run: it is a write (M2d), and the journals it existed for have already been refused.
- **The check at open compares, and repairs nothing.** It compares on `mode=ro` connections. A
  difference refuses as today. A record on the primary, of any operation including 02's `prune`,
  refuses, naming the record. Nothing is cleared.
  - Since 02, `_reconcile_replicated_tables` sends a `prune` record to
    `_complete_interrupted_prune`, which writes, **before** it compares anything. The read-only
    refusal of a record comes before that branch, so that a `prune` record is refused by name and
    is never met first by the `mode=ro` file's own error (added 2026-10-03).
- **No DDL at all.** No `replication_in_flight.create`, no `_ensure_tables`. A table a shard lacks
  is reported, as `store_reader` reports it, and a lookup on it then fails in SQLite.
- **`drop_actions` or `prune_unvalidated=True` raise `ReadOnlyWrite` at construction**, before
  anything is opened.
- **`_validate_on_startup` runs report-only.**
  - It queries the table of every class that validates at startup. With no `_ensure_tables`, a
    shard that lacks one of those tables would fail the open with `no such table`. Under
    `read_only=True` such a class is skipped and reported, as 02's `_prune_needed` skips it, so
    that a store missing a table still opens (§3 case 5) (added 2026-10-03).
- **The version row is found, never written.**
  - The pool selects it by label on a `mode=ro` connection: `SELECT serial, label FROM version
    WHERE label = ?`, as M2e measured.
  - When some shard lacks it, the comparison has already refused.
  - When every shard lacks it, the pool refuses, naming the label asked for and the labels
    present.
  - It does not call 02's replicated version write. Whether it calls `set_version` with the
    serial it found is the *prompt's choice*, since a read-only actor inserts nothing.
- **No serial is allocated.** The broker, `read_largest_store_ids` and `notify_largest_store_ids`
  may be skipped (M1: economy, not safety). The *prompt's choice*, stated in the log.

**O2 — `ReadOnlyMiss` and `ReadOnlyWrite`.** Both names are README §4's. Both live beside
`ReplicatedDivergence` in `Datastore/replication.py`, and both are importable from the pool's module.
- **A replicated `object_get` on a read-only pool** goes to one shard's actor, chosen as
  `read_table` chooses, with no record and no replication. `_replicated_write` is never entered.
- **A lookup that would insert raises `ReadOnlyMiss(class, payload)`**, before any `INSERT` is
  issued.
  - On a read-only actor, every factory's `build` is handed an inserter that raises it. That
    covers every insert-on-miss in one place (M1), with no change to any factory.
  - `ReadOnlyMiss` names the class and the payload.
  - Classes that return an unpopulated object on a miss (`BackgroundModel`,
    `wavenumber_exit_time`, every sharded `build`) do not insert, and so do not raise. Their
    behaviour is unchanged.
- **Every other write raises `ReadOnlyWrite` before anything is written.**
  - On the pool: `object_store`, `object_validate`, `_assign_shard_keys` when it would insert a
    key, and any `object_get_vectorized` that reaches an inserter.
  - On the actor: `object_store` and `object_validate`.
- **The `mode=ro` file is the backstop.** An `sqlite3.OperationalError` "attempt to write a
  readonly database" from a read-only actor is re-raised as `ReadOnlyWrite`, chained, naming the
  actor. Among the writes it catches are `wavenumber`'s and `redshift`'s flag updates on a hit
  (audit R1), which no reader reaches today.
- **A read-write pool behaves exactly as before.** Nothing on the read-write path changes.

**O3 — the six readers open read-only.** Each script constructs its pool with `read_only=True`, and
drops `prune_unvalidated=False`, which no longer means anything for a reader. Nothing else in the
lookup sequence changes, and no lookup is rewritten to avoid a write. The read-only pool is what
stops a write; a reader that would have written now raises.

**O4 — `--models`, and a model the store does not hold.**
- Each reader takes `--models LABEL [LABEL …]`, with `scoped_pipeline_run.py --models`' meaning.
  - It restricts the models read to these labels.
  - A label that `build_model_list` does not return is an error, naming the labels it does return.
    This is stricter than `scoped_pipeline_run.py`, which raises only when **no** label matches
    and drops an unknown one silently. A reader must not drop a mistyped label, so this rule
    governs (added 2026-10-03).
  - Add the argument once, in `extract_common.py`, beside `add_run_selection_argument`, with the
    filter as a pure function there.
- **The filter applies after `build_model_list` returns.** Do not change `config/model_list.py`.
  - The audit suggested filtering before any cosmology lookup (R3). That is not needed: every store
    `main.py` builds holds both cosmology rows, A3 v2's included (R3, read and measured), because
    `build_model_list` inserts both at the build's first open.
  - On a read-only pool, a store without a cosmology row raises `ReadOnlyMiss` from
    `build_model_list`. That is right: such a store was not built by `main.py`.
- **A model whose `BackgroundModel` is absent is skipped**, whether or not `--models` was given.
  - The reader prints one line naming the model and saying the store holds no background for it,
    and goes on to the next model.
  - It no longer raises `RuntimeError("Could not locate suitable background model…")`.
  - If **no** model is read, the script exits non-zero, saying which models were skipped and why.
- **One shared helper.** The skip is the same in all six scripts, so it is one helper in
  `extract_common.py`, called from each script's model loop. Each script's own body changes only
  where it calls the helper.

---

## 3. Verification

Tests under `Datastore/tests/` and `ComputeTargets/tests/` (where the reader-plumbing tests are), on
stand-in shards, with no Ray (README §5 rule 9), and nothing under `var/`.

1. **Nothing written, on a full store.** QSI's whole lookup sequence (R2's transcription; import it
   if it imports cleanly, otherwise transcribe it in the test) on a read-only stand-in pool. The
   instrument is the SHA-256 of every file, before and after. Every file is byte-identical, and the
   same hits come back as R2 Run 1 recorded.
2. **The instrument counts one first** (A2). The same sequence on a **read-write** pool changes the
   primary's SHA-256 (its in-flight commits). The test shows that before it reports the read-only
   zero.
3. **Each insert-on-miss raises `ReadOnlyMiss`**, naming class and payload, and no file changes:
   `store_tag`, `LambdaCDM`, `QCD_Cosmology`, `tolerance`, `GkSourcePolicy`, each missing in turn
   (R2's Runs 2–9).
4. **Each other write raises `ReadOnlyWrite`**, with no file changed:
   - `object_store`, `object_validate`, a new shard key, `drop_actions`, `prune_unvalidated=True`;
   - and the backstop: a write that reaches SQLite, provoked through an actor method that writes on
     a hit, re-raised as `ReadOnlyWrite`.
5. **At open:**
   - a missing primary is refused and nothing is created, not even its directory;
   - an empty journal and a hot journal beside a shard are each refused, and the hot one is not
     rolled back (its SHA-256 and the shard's unchanged);
   - a primary holding a record is refused, with the record named and not cleared;
   - an absent version label is refused, naming the labels present;
   - a shard lacking a table is reported, and the store opens.
6. **A read-write pool is unchanged.** The existing Datastore suite passes untouched (README §5
   rule 7).
7. **The six scripts**, read with `ast` (they call `ray.init` at module scope and cannot be
   imported; `ComputeTargets/tests/test_main_plumbing.load_main_py_functions` is the pattern):
   - each constructs `ShardedPool` with `read_only=True` and without `prune_unvalidated`;
   - each adds `--models` through `extract_common`;
   - each calls the skip helper in its model loop.
8. **`--models` and the skip**, as pure functions:
   - an unknown label is an error naming the known ones;
   - the filter keeps order;
   - a model with no background is skipped with its line;
   - all skipped exits non-zero with the reason.
9. All six suites, serially, with README §7's command. Every existing count unchanged.

**Deliberate breakage** (README §5 rule 8). Each must make a test fail. Restore it afterwards, and
record each in the log as a diff, exactly as applied:
- (i) a read-only replicated `object_get` goes through `_replicated_write` again;
- (ii) the read-only inserter inserts instead of raising;
- (iii) the actor's engine is opened read-write under `read_only=True`;
- (iv) the empty-journal refusal is removed;
- (v) the check at open's step-1 `sqlite3.connect` runs under `read_only=True`. A checksum cannot
  catch this. Journals are refused by presence before step 1 is reached, and a read-write open of
  a shard with no journal writes nothing. The test that catches it must observe the connection
  itself, for example by recording what `ShardedPool`'s `sqlite3.connect` is called with (added
  2026-10-03);
- (vi) one script keeps `prune_unvalidated=False` and drops `read_only=True`;
- (vii) the skip helper raises again for a model with no background.

---

## 4. Acceptance

1. O1–O4 as §2 states them, with each *prompt's choice* stated and justified in the log.
2. The tests of §3, and the deliberate-breakage record (i)–(vii).
3. Both issues moved to the `datastore-integrity` board's §4. Each gets a closing note naming this
   commit and the tests that show it closed. Both index rows deleted; the count and date
   corrected.
4. `black --check` clean. On this board: the §1 row, O1–O4, §3/§4, §5. All in one commit.

---

## 5. Stop conditions — stop and ask the user

- A lookup any of the six readers makes cannot be served from `mode=ro` files (audit M2b measured
  that every one can).
- A reader turns out to reach a write on a hit (a flag update, a validate) on a full store.
- Making the read-only path work would change the read-write path's behaviour, or an existing test.
- Reusing `store_reader`'s refusals would need its behaviour changed for its existing callers.
- Anything here needs a store under `var/`. The readers are run against a real store by 07, not by
  this prompt.

---

## 6. What this prompt does not do

- It does not change `config/model_list.py`, `main.py`, any driver, or the read-write path.
- It does not rewrite any reader's lookups, nor its value reads, plots or outputs.
- It does not run a reader against a real store. That is 07's L3.
- `--profile-db` writes a profiling database of its own, not the store. It is unchanged.

---

## 7. The log and the board

`logs/03-open-readers-read-only.md`, using README §5.1. In addition:
- M1's table with a column saying where each row is implemented, by function name;
- the full text of each new exception's message, as raised in a test.

`IMPLEMENTATION_STATE.md`: the §1 row for 03, O1–O4, §3/§4, §5.
