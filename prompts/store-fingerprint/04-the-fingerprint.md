# Prompt 04 — the fingerprint: digests of the structured inventory, in the sidecar and the run record

**Campaign:** [`README.md`](README.md) · **Board items:** **F10**, **F11**, **F12** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]`.
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Decisions:** README §6.1 points 1, 3, 7, 8 and 9 fix what a fingerprint is and who computes it.
D2 (README §6.2) makes `docs/handover/quadsource_atol_sweep.py` take one.
**Recommended model:** **Opus**. The code is moderate. The judgement is in three places: a digest
that localises a difference to one class and one tag set, a sidecar write that changes exactly one
field, and a finish that records a fingerprint without ever changing how a run ended.

**Read first:**

1. [`README.md`](README.md) in full, especially §1, §3, §4, §6.1 and §6.3.
2. `run-registry`'s board, [`prompts/run-registry/IMPLEMENTATION_STATE.md`](../run-registry/IMPLEMENTATION_STATE.md):
   the entry for `[04-a-runs-product-is-named-but-never-fingerprinted]` in full, **including its
   amendment of 2026-09-24**. The amendment is the user's specification of the fingerprint.
3. [`docs/store-fingerprint-audit.md`](../../docs/store-fingerprint-audit.md): §7 and §8.
4. Prompt 02's log, [`logs/02-a-structured-inventory.md`](logs/02-a-structured-inventory.md):
   "State handed to the next prompt", and "Observations not acted on" (the reference digest does
   not cover `validated`, and the fingerprint does). `Datastore/store_inventory.py` in full.
5. `RunRegistry/stores.py` in full, `RunRegistry/__init__.py` (`Run`, `begin`, `finish`,
   `write_json_atomic`, `git_provenance`), and `RunRegistry/__main__.py`. Their tests in
   `RunRegistry/tests/`, especially `test_store_command_line.py`'s check that `import RunRegistry`,
   `list` and `store show` load neither `ray` nor `sqlalchemy`.
6. `datastore-portability` README §6.3 and §6.5, on the layering and on when an existing sidecar
   may change.
7. The drivers: `docs/gktk-remedial/scoped_pipeline_run.py` (`register`, `terminal_state`, and the
   `run.finish` calls at `:486-501`), and `docs/handover/quadsource_atol_sweep.py` (`run_build` at
   `:436-578` and `sweep` at `:767-871`). **You change them only as §2 F12 says** (D2).
8. `CLAUDE.md`: "Long-running jobs — the run registry" and "Repository mechanics".

---

## 1. What is wanted

After a store is copied between machines, nothing can say whether the copy holds what the original
held (README §0). Prompt 02 built what can: a structured inventory that names each work item by
physical labels and its tag set, independently of serials and shards. This prompt digests it.

- **A pure function** from a `StoreInventory` to a fingerprint: a format version, and per class
  and per tag set a count and a digest, plus one overall digest. **Digests only**; never a
  listing (§6.1 point 7). A mismatch between two fingerprints names the class and the tag set that
  differ.
- **A known sidecar field**, `fingerprint`, which `RunRegistry.stores` defines, validates, and
  carries through copy and move.
- **`python -m RunRegistry store fingerprint`**: computes it for any closed store, read-only,
  compares it with the recorded one, and writes it into the sidecar only when asked. It refuses a
  store that a `running` run names.
- **At a registered run's finish**, the fingerprint is taken, written into the store's sidecar,
  and copied into the run record. The two drivers that register runs take one (D2).

---

## 2. What to change

### F10 — the fingerprint, a pure function of the inventory

In `RunRegistry/stores.py`, beside the sidecar it lives in. *Prompt's choice:* a module of its own
under `RunRegistry/` is acceptable if `stores.py` would otherwise mix two concerns; record which.

1. **`fingerprint_of(inventory, taken=None) -> dict`**, where `inventory` is what
   `read_inventory` returns. A record's canonical form is `record.canonical_json()`. Anything
   else it needs from `Datastore.store_inventory`, such as `canonical_json` for the overall
   digest, it imports **inside the function body**, because that module imports `sqlalchemy` and
   `import RunRegistry` must not. There is one canonical JSON, and this function does not write a
   second. Its result:
   - **`fingerprint_format`**: `FINGERPRINT_FORMAT = 1`.
   - **`classes`**: for every class in the inventory, in its order:
     - `count`, the number of records;
     - `digest`, over **every** record of the class;
     - `tag_sets`: one entry per distinct tag set, each with its `tags` (the sorted labels),
       `count` and `digest` over the records that carry exactly that set. A class with no
       association table has one entry, with `tags` `[]`.
   - **`digest`**, the overall one, over the format version and each class's `count` and
     `digest`.
   - **`problems`**: per class, the number of named problems of each kind (the word before the
     first colon of each problem string). *Prompt's choice:* **counts only, outside every
     digest.** A problem's text names shards and serials, which are store-local, so it cannot be
     digested without breaking the physical-key property. A store with problems is still
     fingerprinted, and `store fingerprint` prints them.
   - **`taken`**: `{"when", "git_head", "git_dirty", "run_id"}`, supplied by the caller, and
     **outside every digest**.
2. **What a digest covers.** *Prompt's choice:* **the canonical record**,
   `canonical_json(record.as_json())`: key, tags, `validated` **and** `value_count`. The value
   count is in, because a store missing half a parent's value rows does not hold what the
   original held. The flag is in, which closes the gap prompt 02's log noted. **No timestamp is
   in any record** (§6.1 point 7); the test proves it rather than assuming it.
3. **How a digest is formed.** *Prompt's choice:* SHA-256 of the records' canonical JSON, **one per
   line, each followed by `\n`**, in the order `read_inventory` already gives, which is sorted by
   canonical JSON. So a listing written as those lines hashes to the digest it lists. That is what
   lets a person check a listing generated on demand (§6.1 point 7) against a recorded
   fingerprint. The overall digest is SHA-256 of `canonical_json` of
   `{"fingerprint_format", "classes": {name: {"count", "digest"}}}`.
4. **`compare_fingerprints(recorded, current) -> list`**, pure. Two fingerprints of different
   formats are **not compared**: the answer is a single entry saying so, because the remedy is to
   recompute both from the stores (the amendment). Otherwise it returns one entry per difference,
   each naming:
   - the class;
   - the tag set, or that a tag set is present in only one;
   - the two counts.

   Equal overall digests give an empty list. Differences in `problems` are reported as their own
   entries. `taken` is never compared.
5. **The format is pinned.** A golden fingerprint of `build_full_store`'s default store is
   committed as test data. If a later change to the inventory changes it, the test fails, and the
   change must bump `FINGERPRINT_FORMAT` and regenerate the golden **deliberately, in the commit
   that changes the inventory**. This is how a fingerprint recorded under one set of keys is never
   compared with one taken under another.

### F11 — the sidecar field, and `store fingerprint`

1. **`fingerprint` becomes a known field** of `RunRegistry.stores`: added to `KNOWN_FIELDS`,
   optional, and described in the module docstring's format list. `_registry_problems` checks its
   shape: a JSON object with an integer `fingerprint_format`, `classes`, a 64-hex `digest` and a
   `taken` object. A malformed value is a problem, like any other field.
2. **Copy and move carry it verbatim.** Both leave the content unchanged, which is why the
   amendment made it a known field (`copy_store` already copies every field; confirm it and test
   it). *Prompt's choice:* **the copy keeps the source's `taken`**, because it records when that
   content was fingerprinted and by whom, and a copy does not change it. `store fingerprint` on
   the copy then checks the copy against it, which is the two-machine question answered.
3. **`fingerprint_store(primary, *, write=False, runs_root=None, taken_by=None) -> dict`**, the
   registry operation (README §4). In this order:
   1. **refuse if any `running` run, alive or stale, names the store**, through the check
      `copy_store` and `move_store` already make (`runs_naming`), by path or by `store_id`. The
      one exception is the run `taken_by` names, which is its caller and is still `running`
      because it has not finished yet. A fingerprint of a store being written describes no
      instant (audit §7);
   2. `from Datastore.store_inventory import read_inventory`, **inside the function**, so that
      `import RunRegistry` still loads neither `ray` nor `sqlalchemy`;
   3. `read_inventory(primary)`. A reader refusal comes through with its message;
   4. `fingerprint_of(...)`, with `taken` filled from `now_iso()`, `git_provenance()` and
      `taken_by`'s run id;
   5. `read_sidecar(primary)`, and `compare_fingerprints` against its recorded `fingerprint`, if
      there is one;
   6. **only if `write`**: refuse unless the sidecar is a problem-free registry sidecar, naming
      `store create` / `store adopt`; then replace **only the `fingerprint` field**, in place,
      through `_update_sidecar`. Every other field is value-identical after a JSON round trip,
      unknown ones included.

   It returns the fingerprint, the comparison, and whether it wrote. It never writes the store,
   never initialises Ray, and never deletes anything.
4. **A third in-place update.** `_update_sidecar`'s docstring says it is used in exactly two places.
   It becomes three. Update the docstring, and the module docstring's statement of what is
   overwritten, to say so. Recording a fingerprint when a person asks for it, or when a registered
   run finishes (F12), is the explicit request that `datastore-portability` README §6.5 point 7
   requires before an existing sidecar changes. D3 is the same request for prompt 05.
5. **`python -m RunRegistry store fingerprint PRIMARY [--write] [--listing PATH] [--runs-root DIR]`.**
   - Without `--write`, **read-only**. It prints the overall digest, one line per class and tag
     set with the count and a short digest, the problem counts, and the comparison with the
     recorded fingerprint: `none recorded`, `matches` (naming when and by whom the recorded one
     was taken), or each difference. It exits 0 if it matches or none is recorded, and 1 on a
     difference or a refusal.
   - `--write` records it, as F11 item 3 says, and prints what it replaced.
   - *Prompt's choice:* `--listing PATH` writes the full listing, the canonical record lines
     grouped by class and tag set, to a **new** file at `PATH`. It refuses to overwrite, and
     refuses a path inside the store's own directory. This is the listing generated on demand
     (§6.1 point 7). The fingerprint never contains it.
   - `store show` already prints every field, and so prints the recorded fingerprint. It must
     still load neither `ray` nor `sqlalchemy`.

### F12 — the fingerprint at a run's finish

1. **`Run.finish(state, exit_code=None, *, fingerprint=False)`.** With `fingerprint=True`, and a
   manifest that names `results`, it calls
   `fingerprint_store(results, write=<the sidecar is a problem-free registry sidecar>, taken_by=self)`
   **before** writing the terminal state, and puts the fingerprint in `status.json` under
   `fingerprint`. The manifest stays immutable; `status.json` is already the run's one mutable
   file.
   - *Prompt's choice:* **the sidecar is written at finish**, where it is a registry sidecar.
     The fingerprint lives in the sidecar (§6.1 point 1). A run that changed its store and left
     the old fingerprint there would leave a stale value in the one place a reader looks. With no
     sidecar, or a legacy one, only the run record gets it, and `status.json` says why the
     sidecar did not.
   - **A fingerprint never changes how a run ended.** Any refusal or error is recorded in
     `status.json` as `fingerprint_error`, a string, and `finish` goes on to write the state and
     exit code it was given. It never raises for the fingerprint's sake.
   - *Prompt's choice:* **every terminal state `finish(..., fingerprint=True)` reaches takes one**:
     `done`, `failed`, and `killed` as it arrives through `SystemExit`. A failed run's partial
     product is a product, and the A3 resume that failed at 92% is the case. The signal handlers
     are different: they run with the pool still open, so they keep `fingerprint=False`.
   - `fingerprint=False` is the default, so every existing caller is unchanged.
2. **The drivers.** In `docs/gktk-remedial/scoped_pipeline_run.py`, and in both `run_build` and
   `sweep` in `docs/handover/quadsource_atol_sweep.py` (D2), every `run.finish(...)` **after the
   pipeline has returned or raised** gains `fingerprint=True`. By then `main.py`'s
   `with ShardedPool(...)` block has exited, and its `__exit__` has closed every actor's engine,
   so the store is closed. The `terminal` signal handlers are not changed. **Nothing else in
   either script changes**: not the substitutions, the arguments, the measurement code or its
   results.

---

## 3. Tests — in `RunRegistry/tests/`, no Ray, nothing under `var/`

Build stores with `Datastore.tests.real_store_fixtures`, and run directories in a temporary runs
root, as the existing registry tests do.

1. **The fingerprint is physical.** The store built by `relabel_serials` with its rows moved
   between shards, prompt 02's test 1, has a fingerprint equal to the original's, apart from
   `taken`.
2. **A difference is localised.** Each of these changes exactly the digests named, and no other:
   - deleting one `QuadSourceIntegral` row: that class's count and digest, its one tag set's, and
     the overall digest;
   - adding a tag to one record: that record leaves its tag set and joins a new one;
   - deleting one value row: its parent class's digest, through `value_count`;
   - flipping one `validated` flag.

   `compare_fingerprints` names exactly those classes and tag sets.
3. **What is not content does not matter.** Changing every timestamp, every compute-target label
   and every solver serial leaves the fingerprint's digests unchanged.
4. **The format is pinned.** The fingerprint of `build_full_store()` equals the committed golden
   file, apart from `taken`. The golden's `fingerprint_format` equals `FINGERPRINT_FORMAT`.
5. **A listing hashes to its digest.** The lines `--listing` writes for each class and tag set
   hash to that entry's digest.
6. **Formats are not compared.** Two fingerprints of different formats give the one "recompute
   both" entry.
7. **The sidecar field.**
   - A well-formed `fingerprint` is not a problem, and each malformed variant is.
   - `copy_store` and `move_store` carry it verbatim, `taken` included.
   - An existing registry sidecar with no `fingerprint` still reads as problem-free.
8. **`store fingerprint`, read-only.** Without `--write`, the store's `file_state` and the
   sidecar's bytes are unchanged. In a child interpreter, `ray.is_initialized()` stays false.
9. **`store fingerprint --write`** changes only the `fingerprint` field. Every other field,
   unknown ones included, is value-identical. It is refused, writing nothing, on an absent, legacy
   or problem sidecar.
10. **The refusal.** A `running` run naming the store, alive or stale, by path or by `store_id`,
    makes `store fingerprint` refuse, with and without `--write`. The same run, passed as
    `taken_by`, does not refuse itself.
11. **`Run.finish(..., fingerprint=True)`.**
    - A registered run over a fixture store finishes `done` with `fingerprint` in `status.json`,
      and the same fingerprint in the sidecar.
    - A store the reader refuses (for example, with a `-journal` beside a shard) finishes `done`
      with a `fingerprint_error` and no `fingerprint`, and the sidecar is untouched.
    - A legacy sidecar leaves the sidecar untouched and records why.
    - `fingerprint=False` writes neither.
12. **The drivers.** Using `ast` on each driver:
    - every `run.finish(` call outside a `terminal` handler passes `fingerprint=True`;
    - every call inside one does not.
13. **Imports.** `import RunRegistry`, `list` and `store show` still load neither `ray` nor
    `sqlalchemy` (the existing test, unchanged, still passes).

**Deliberate breakage.** Each of these must fail the tests written against it. Record each diff
exactly as applied, with the tests that failed.

- (i) a record's timestamp is added to what is digested;
- (ii) a tag set's digest is taken over every record of its class;
- (iii) `value_count` is left out of what is digested;
- (iv) `store fingerprint` without `--write` writes the sidecar;
- (v) the running-run refusal is removed;
- (vi) `copy_store` drops `fingerprint`;
- (vii) a fingerprint error in `finish` is raised instead of recorded;
- (viii) one key field is removed from one factory's `inventory_records`, with no format bump;
- (ix) records are digested in serial order instead of canonical order.

---

## 4. The demonstration — on copies of the sweep store

Never an original (README §3). Work in `var/store-fingerprint-check-04/`, with a runs root of its
own inside it, never `var/runs/`. Snapshot the originals before and after, and delete the directory
at the end.

1. `python -m RunRegistry list`: nothing `running`. Snapshot the originals read-only, as in prompt
   02 §4, including the three sidecars' bytes.
2. `cp -p` the sweep store's five files **and its sidecar** into `var/store-fingerprint-check-04/a/`.
   The sidecar names its store by bare file name, so it describes the copy beside it. It still
   carries the original's `store_id`. Record that, and use `--runs-root` inside the check
   directory for every command, so that no run under `var/runs/` can be matched to it.
3. `store fingerprint a`: record the overall digest, the per-class table, the problem counts and
   `none recorded`. Record the wall time, the peak resident memory and the fingerprint's size in
   bytes as JSON. Confirm that the copy's `file_state` and its sidecar are unchanged.
4. `store fingerprint a --write`. Show that the sidecar gained `fingerprint` and nothing else
   changed. Run it again without `--write`: `matches`.
5. `store copy a b`, a registry copy of the scratch copy. Show that `b`'s sidecar carries the same
   `fingerprint`, and that `store fingerprint b` says `matches`.
6. **The discriminator.** On `b`'s shard 0, delete `QuadSourceIntegral` serial 329386 with
   `sqlite3`, as prompt 02 did. `store fingerprint b` must name **exactly one difference**:
   `QuadSourceIntegral`, its one tag set, 7 706 against 7 705. Its problem counts show prompt 02's
   `orphan-tag`. Nothing else differs.
7. **The run record.** In the check directory's runs root, `RunRegistry.begin(..., results=a)`,
   then `run.finish("done", exit_code=0, fingerprint=True)`. Show `status.json`'s `fingerprint`
   and the sidecar's, which are equal. Then begin a second run naming `a` and leave it
   `running`. `store fingerprint a` must refuse, naming it. Finish it `done` without a
   fingerprint.
8. Re-take the snapshot of the originals. It must be identical.
9. Delete `var/store-fingerprint-check-04/`.

---

## 5. Acceptance

1. §3's tests exist and pass, with no Ray and nothing under `var/`. Every existing test module is
   unmodified.
2. The deliberate-breakage record, (i)–(ix).
3. §4 is done, with its numbers, and the working directory is deleted.
4. **Scope.** `git diff HEAD~1 HEAD --stat` touches only `RunRegistry/`, the two drivers,
   `RunRegistry/tests/` and its test data, the log, this board, `run-registry`'s board and
   `docs/OPEN_ISSUES.md`. `Datastore/` is untouched. In the drivers, only `run.finish(` lines
   change.
5. Every suite matches its baseline, except that `RunRegistry/tests` rises by exactly the tests
   you add. `black --check` is clean.
6. **This board:** the §1 row for 04; items F10–F12; any issue opened; §5's baselines.
7. **`run-registry`'s board:** `[04-a-runs-product-is-named-but-never-fingerprinted]` moves to its
   §4, with a closure line naming this commit, the format, and the §4 discriminator.
8. **`docs/OPEN_ISSUES.md`**, in the same commit: its row deleted, and the count and date
   corrected.

---

## 6. What this prompt does not do

- It does not write a fingerprint into any of the three real sidecars. That is prompt 05 (D3).
- It does not change the inventory, a key, a lookup or a schema. If a key is wrong, stop (§7).
- It does not change `main.py`, `tools/`, `extract_common.py` or `ShardedPool`.
- It does not add a `pull` or a transfer. Transfer is acting, not recording (the amendment).
- It does not lock, schedule, supervise, restart or delete anything (`CLAUDE.md`).
- It does not change any existing run manifest, or any existing sidecar field.
- In the drivers, it changes nothing but the `finish` calls (D2).

---

## 7. Stop conditions — stop and ask the user

- An existing sidecar already holds an unknown field named `fingerprint`. Making the name known
  would reinterpret a value the registry has promised to carry verbatim.
- Two `store fingerprint` runs on the same unchanged copy differ, or a registry copy of the copy
  does not match it (§4 step 5).
- Localising a difference to a class and tag set would need a change to the inventory, a key or a
  schema.
- The golden fingerprint of `build_full_store` is not deterministic from run to run.
- `fingerprint_store` needs Ray, or writes anything but the sidecar's `fingerprint` field.
- A driver cannot take its fingerprint without changing something besides its `finish` calls.

---

## 8. The log and the board

`logs/04-the-fingerprint.md`, using README §5.1's template. In addition, record:
- the fingerprint's format as shipped, field by field, with one real example from §4;
- each *prompt's choice*, and whether it was kept;
- the §4 numbers: the digests, the per-class table, the discriminator's one difference, the size,
  the time and the memory.

On `IMPLEMENTATION_STATE.md`: the §1 row for 04; items F10–F12; §5's baselines. On `run-registry`'s
board, the closure. Update `docs/OPEN_ISSUES.md` in the same commit.
