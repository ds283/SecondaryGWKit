# Log 04 — The fingerprint: digests of the structured inventory, in the sidecar and the run record

**Prompt:** [`prompts/store-fingerprint/04-the-fingerprint.md`](../04-the-fingerprint.md)
**Commit:** *(this commit)*, "Fingerprint a store's content in its sidecar and run record"
**Base:** `4ed03502afca60787ef0888d9dce5cd2dda9aa12` ("Move the inventory report and run labels onto
one read-only service"), clean. Prompt 03 had landed.
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. F10–F12 shipped. No §7 stop condition arose:
- none of the three real sidecars holds a `fingerprint` field (`grep -c '"fingerprint'` gave 0 on
  each);
- two `store fingerprint` runs on the same unchanged copy printed identical output;
- the registry copy of the copy matched it;
- no key, inventory or schema changed;
- the golden fingerprint is the same from build to build;
- `fingerprint_store` needs no Ray and writes nothing but the sidecar's `fingerprint` field;
- each driver took its fingerprint by changing only its `run.finish(` calls.

On a copy of the sweep store the fingerprint is **6 284 bytes** of compact JSON (9 132 as written
into the sidecar), taken in **8.1 s** at a peak RSS of **233 MB**. Its one-row discriminator names
exactly one content difference, `QuadSourceIntegral`'s one tag set at 7 706 recorded against
7 705 now, and one `orphan-tag` problem entry. All nine deliberate-breakage mutations were caught.
Closed `run-registry`'s `[04-a-runs-product-is-named-but-never-fingerprinted]` (that board's §4).
No new issue was opened.

## What shipped

**F10 — the fingerprint, a pure function of the inventory** (`RunRegistry/stores.py`, in a new
section at the end of the module).

- `FINGERPRINT_FORMAT = 1`, and `TAKEN_KEYS = ("when", "git_head", "git_dirty", "run_id")`.
- `fingerprint_of(inventory, taken=None) -> dict`. It imports `canonical_json` from
  `Datastore.store_inventory` inside its body, so `import RunRegistry` still loads neither `ray`
  nor `sqlalchemy`. A record's line is `record.canonical_json()`, which is
  `canonical_json(record.as_json())`: key, tags, `validated` and `value_count`. There is no second
  canonical JSON. A digest is SHA-256 over the lines, each followed by `\n`, in the order
  `read_inventory` gives them, which is sorted by canonical JSON.
- `compare_fingerprints(recorded, current) -> list`, pure. Its rules:
  - fingerprints of different formats give one `format` entry saying "Recompute both from the
    stores";
  - otherwise there is one entry per difference, each
    `{"kind", "class", "tags", "recorded", "current", "text"}`. `kind` is `tag_set` (a tag set whose
    count or digest differs, or which is present in only one fingerprint; the absent side's count
    is `None`) or `class` (a class present in only one). There are two defensive kinds, which no
    consistent pair can produce: `class` for a class digest that differs while none of its tag sets
    does, and `digest` for overall digests that differ while no class does;
  - `problems` entries give per class and kind the two counts;
  - `taken` is never compared.
- `listing_lines(inventory, fingerprint)`, a generator for the listing made on demand. The first
  line is `# fingerprint_format 1 digest <hex>`. Then each class has
  `# class <name> count <n> digest <hex>`, and under it each tag set has
  `# tags <canonical JSON list> count <n> digest <hex>` followed by that set's record lines.
  - The lines under a tag-set header hash to its digest.
  - A class's record lines, sorted bytewise (`LC_ALL=C sort`), hash to the class digest.
- `check_listing_path(path, primary)` and `write_listing(inventory, fingerprint, path, primary)`.
  The listing is opened with mode `"x"`. The write is refused if the path exists, if its directory
  does not, or if it is inside the store's own directory or below it.

**F11 — the sidecar field and `store fingerprint`.**

- `fingerprint` is in `KNOWN_FIELDS` and optional. It is described in the module docstring's format
  list. `_registry_problems` checks it through `_fingerprint_problems`: a JSON object, with an
  integer (not boolean) `fingerprint_format`, a `classes` object, a 64-lowercase-hex `digest` and a
  `taken` object. A malformed value is a problem, like any other field.
- **Copy and move already carried it.** `copy_store` and `move_store` both `deepcopy` the source's
  fields, and no code change was needed. The test shows it, `taken` included, and mutation (vi)
  shows that the test bites.
- `fingerprint_store(primary, *, write=False, runs_root=None, taken_by=None) -> dict`, in the
  prompt's six steps.
  - Step 1 needs the store's `store_id` for the by-id match. It takes it from a first
    `read_sidecar`: a registry sidecar's `store_id` if that is 32-hex, even when the sidecar has
    another problem, which is what `store show` does.
  - The check is `runs_naming`, through a new helper `_running_runs_naming`, which
    `_refuse_if_in_use` (copy and move) now uses too. Their behaviour and messages are unchanged.
    The helper excludes `taken_by` by the realpath of its run directory.
  - It returns `{"primary", "fingerprint", "recorded", "comparison", "wrote", "sidecar",
    "inventory"}`. `inventory` is returned so that the command line can write a listing of the
    very reading it fingerprinted. `fingerprint_store` itself never writes a listing.
- **The third in-place update.** `_update_sidecar`'s docstring, and the module docstring's statement
  of what is overwritten, now name three uses: adopt, the move's temporary sidecar, and recording a
  fingerprint.
- **`python -m RunRegistry store fingerprint PRIMARY [--write] [--listing PATH] [--runs-root DIR]`**
  (`RunRegistry/__main__.py`, `_fingerprint`). It prints:
  - the store, the overall digest and the format;
  - one line per class (count and a 12-hex short digest), and one per tag set under it;
  - the problem counts;
  - then one of `recorded: none recorded (the sidecar is <kind>)`, `recorded: matches the
    fingerprint taken <when> by run <id> | a person, at <git_head> [(dirty)]`, or each difference
    marked `!!`.

  Without `--write` it exits 0 on a match or when nothing is recorded, and 1 on a difference or a
  refusal. `--write` prints what it replaced. A listing path is checked before the store is read,
  and written after.

**F12 — the fingerprint at a run's finish.**

- `Run.finish(state, exit_code=None, *, fingerprint=False)` (`RunRegistry/__init__.py`). With
  `fingerprint=True` it calls `_fingerprint_results()`, and only then writes the state. That
  helper:
  - records `fingerprint_error` when the manifest names no `results`;
  - otherwise reads the sidecar and calls `fingerprint_store(results,
    write=<the sidecar is a problem-free registry sidecar>, runs_root=<this run's own runs root>,
    taken_by=self)`;
  - returns `fingerprint` and `fingerprint_sidecar`. The second is `"written"`, or `"not written:
    <path> is not a problem-free registry sidecar (<why>)"`.

  Any `Exception` becomes `fingerprint_error`, a string, and the given state and exit code are
  written regardless. The default `fingerprint=False` leaves every existing caller unchanged.
- **The drivers.** Nine `run.finish(` calls gained `, fingerprint=True`: three in
  `docs/gktk-remedial/scoped_pipeline_run.py`, three in `run_build` and three in `sweep` in
  `docs/handover/quadsource_atol_sweep.py`. `black` wrapped two of the sweep's lines. The three
  `terminal` signal handlers are unchanged. `git diff` of the two drivers touches nothing but
  those calls (verification §3).

**Docstrings made false by F12 were corrected.** Four places in `RunRegistry/__init__.py` said the
registry "never opens" a run's results: the module docstring, `results_path`, `begin`, and the
message `record()` raises when there is no ledger. Each now says the registry never *writes* them,
and the first two add that `finish(..., fingerprint=True)` reads them. No test asserts on any of
these texts.

**Tests** — `RunRegistry/tests/test_store_fingerprint.py` (new, **33 test methods** in 11 classes,
about 15 s), and the golden `RunRegistry/tests/data/full_store_fingerprint.json` (8 780 bytes,
overall digest `1fd51d93facee815b3c44584d81103cdc129fb7bd71440875198ae2ea8cf196a`). No existing test
module was modified. None of the new tests needs Ray or opens anything under `var/`.

| §3 | Test class | What it shows |
|---|---|---|
| 1 | `TestPhysical` (2) | The store with relabelled serials and its sharded rows swapped between shards, and the relabelled store with its rows on a third shard, both give a fingerprint equal to the original's, apart from `taken`, and an empty comparison. |
| 2 | `TestLocalised` (5) | Each change moves exactly the entries it names, and the comparison names exactly those tag sets. Only the named class has changed records; the changed entries are that class and the tag set of every changed record; the overall digest changes; `problems` does not. The four changes: deleting `QuadSourceIntegral` 3 and its tag row (the class 3 → 2, `[Run_fixture]` 2 → 1); adding `unused-tag` to it (`[Run_fixture]` 2 → 1, `[Run_fixture, unused-tag]` absent → 1, class count unchanged); deleting `TkNumericValue` 24; flipping `TkNumericIntegration` 3's `validated`. A fifth test deletes the row alone, as §4 does. The comparison is then one `tag_set` entry and one `problems` entry (`orphan-tag` 0 → 1). |
| 3 | `TestNotContent` (2) | One store with every timestamp, every compute-target and `BackgroundModel` label, and every `solver_serial` / `phase_solver_serial` / `friction_solver_serial` changed has the baseline's fingerprint, apart from `taken`. The test asserts that each kind of change really happened. No record's canonical JSON contains `FIXED_TIMESTAMP` in any form. |
| 4 | `TestGolden` (3) | `build_full_store()`'s fingerprint equals the golden, apart from `taken`, on two builds. The golden's format is `FINGERPRINT_FORMAT` and its `taken` is `null`. The shape: class order is the inventory's, the tag-set counts sum to the class count, an untagged class has one `[]` set, `QuadSourceIntegral` has its two tag sets, and a filled-in `taken` passes the sidecar's shape check. `taken` changes no digest and no comparison. |
| 5 | `TestListing` (2) | The listing's tag-set lines hash to each tag set's digest, and each class's lines, sorted, hash to its digest. It refuses an existing file (left as it was), a path in the store's directory or a subdirectory of it, and a missing directory. |
| 6 | `TestFormats` (2) | Formats 1 and 2 give the one "Recompute both" entry. Two equal format-1 fingerprints give none. |
| 7 | `TestSidecarField` (4) | A registry sidecar without the field is problem-free, and one with a well-formed field is too; the field is not unknown. Fourteen malformed variants are each a problem naming `fingerprint`. `copy_store`, then `move_store`, carry it verbatim, `taken` included, and the source is unchanged. |
| 8, 9 | `TestFingerprintCommand` (3) | In child interpreters (no Ray initialised, asserted by the harness):<br>• without `--write`, with a `--listing` outside the store, the store directory's `file_state` (hashes, sizes, mtimes, listing, sidecar included) is unchanged, and the listing's header digest is right;<br>• `--write` changes only `fingerprint`, and an unknown nested `note` field is value-identical;<br>• a second read-only run prints `matches … by a person`;<br>• `store show` prints the digest and loads neither `ray` nor `sqlalchemy`;<br>• a deleted row exits 1, naming the tag set and the `orphan-tag` entry;<br>• `--write` is refused, writing nothing, on an absent, legacy or problem sidecar (function and command line), while read-only still works there. |
| 10 | `TestRunningRefusal` (2) | For alive and stale runs, matched by path only (no sidecar) or by `store_id` only (the manifest's `results` repointed), with and without `write`, the refusal names the run, its liveness and what matched, and writes nothing. The same run as `taken_by` does not refuse. The command refuses with and without `--write`. |
| 11 | `TestFinish` (7) | `done` puts the fingerprint in `status.json` and the same one in the sidecar, with `run_id`, and not in the manifest. `failed` and `killed` take one too. A `-journal` beside a shard gives `done` with a `fingerprint_error` naming it, no `fingerprint`, and the sidecar byte-identical. Another running run naming the store gives `failed` / 2 with an error naming it. A legacy sidecar is byte-identical and `fingerprint_sidecar` says why. No `results` is an error. `fingerprint=False` writes none of the three keys. |
| 12 | `TestDrivers` (1) | By `ast`: scoped has 3 `run.finish(` calls outside a `terminal` handler and 1 inside; the sweep has 6 and 2. Every one outside passes `fingerprint=True`, and none inside does. |
| 13 | *(existing)* | `test_store_command_line.TestImports`, unchanged, passes. |

## The fingerprint's format, as shipped

Field by field:

| Field | Type | Meaning | In a digest? |
|---|---|---|---|
| `fingerprint_format` | int | `FINGERPRINT_FORMAT`, 1 | yes, in the overall digest |
| `classes` | object, class name → entry | every class of `INVENTORY_CLASSES`, built in inventory order. The files are written with `sort_keys`, so on disk the order is alphabetical | — |
| `classes.<name>.count` | int | the number of records | yes, in the overall digest |
| `classes.<name>.digest` | 64-hex | SHA-256 over every record's canonical line + `\n`, in canonical order | yes, in the overall digest |
| `classes.<name>.tag_sets` | list, sorted by tags | one `{"tags", "count", "digest"}` per distinct tag set. The digest is over exactly that set's records, in canonical order. An untagged class has one entry, `[]`; an empty class has none | — (the tag sets partition the class digest's lines) |
| `digest` | 64-hex | SHA-256 of `canonical_json({"fingerprint_format", "classes": {name: {"count", "digest"}}})` | — |
| `problems` | object | per class that has any, `{kind: count}`, `kind` being the word before the first colon | **no** |
| `taken` | object or `null` | `{"when", "git_head", "git_dirty", "run_id"}`; `null` only from a bare `fingerprint_of` | **no** |

**The real example:** the fingerprint `store fingerprint --write` recorded on the §4 copy of the
sweep store (`classes` abridged to three of 21; the full table is in verification §4):

```json
{
  "fingerprint_format": 1,
  "digest": "2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18",
  "classes": {
    "version": {
      "count": 1,
      "digest": "482c17e477216a7821fbd15f13e17852f1f9c0a8993a00e12b092fca8d7f2d9a",
      "tag_sets": [{"count": 1, "digest": "482c17e477216a7821fbd15f13e17852f1f9c0a8993a00e12b092fca8d7f2d9a", "tags": []}]
    },
    "QuadSourceIntegral": {
      "count": 7706,
      "digest": "28c5c5e4d0393b0146c4ef88a88ef3ca5f288e9d26986e10ff113829d1cceb3b",
      "tag_sets": [{"count": 7706,
                    "digest": "28c5c5e4d0393b0146c4ef88a88ef3ca5f288e9d26986e10ff113829d1cceb3b",
                    "tags": ["GkOneLoopDensity", "LargestSourceRedshift_2.0636e+16",
                             "ResponseRedshiftGrid_145_c51d43ac", "ResponseSparsenessZ_12",
                             "Run_default", "SmallestSourceRedshift_0.1",
                             "SourceGridConstruction_2", "SourceRedshiftGrid_1740_8d1d43b6",
                             "TkOneLoopDensity"]}]
    },
    "OneLoopIntegral": {
      "count": 0,
      "digest": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "tag_sets": []
    }
  },
  "problems": {},
  "taken": {
    "git_dirty": true,
    "git_head": "4ed03502afca60787ef0888d9dce5cd2dda9aa12",
    "run_id": null,
    "when": "2026-09-25T03:05:28+01:00"
  }
}
```

`git_dirty` is true because this prompt's changes were in the working tree, uncommitted. `run_id`
is `null` because a person took it. The copy that `Run.finish` wrote in §4 step 7 has the same
digest and `run_id` `store-fingerprint-04-section-4-finish-20260925T030637`.

## Each *prompt's choice*, and whether it was kept

| Choice | Kept? |
|---|---|
| The fingerprint lives in `RunRegistry/stores.py`, or a module of its own if `stores.py` would otherwise mix two concerns | **Kept in `stores.py`.** README §4 fixes `fingerprint_store`, `fingerprint_of`, `compare_fingerprints` and `FINGERPRINT_FORMAT` there. The fingerprint is a sidecar field, and the section is a coherent block beside the writer it uses. |
| Problems are counts only, outside every digest | **Kept.** `problems` holds `{class: {kind: n}}` for classes with any problem, and is `{}` for a clean store. `store fingerprint` prints them. |
| A digest covers the canonical record, `validated` and `value_count` included | **Kept.** The line is `record.canonical_json()`. Tests 2 (value row, validated flag) and 3; mutation (iii). |
| A digest is SHA-256 of lines + `\n` in the inventory's order, so a listing hashes to it | **Kept.** The order is not re-sorted in `fingerprint_of` (the prompt: "in the order `read_inventory` already gives"). Test 5 checks it independently by sorting, which is what catches mutation (ix). |
| The format is pinned by a golden of `build_full_store`'s default store | **Kept.** Test 4; mutation (viii). |
| `--listing PATH` writes the full listing to a new file, never overwriting, never inside the store's directory | **Kept**, and a path in a subdirectory of the store's directory is refused too (deviation 5). |
| A copy keeps the source's `taken` | **Kept**, with no code change: copy and move already carried every field. Test 7; mutation (vi). |
| The sidecar is written at finish where it is a registry sidecar; otherwise only the run record, saying why | **Kept.** `fingerprint_sidecar` is `"written"` or `"not written: …"`. Test 11. |
| Every terminal state `finish(..., fingerprint=True)` reaches takes one; the signal handlers keep `fingerprint=False` | **Kept.** Tests 11 and 12. |

## Deviations from the prompt

1. **`problems` is compared whatever the digests say.** *IMPLEMENTATION CHOICE.* F10 item 4 says
   both "Equal overall digests give an empty list" and "Differences in `problems` are reported as
   their own entries". Problems are outside every digest, so a store can gain an orphan value row
   while its digests stay the same. That is a real change, and an early return would hide it. So
   equal overall digests give **no content entry**, and a problem difference still gives its own.
   Two equal fingerprints still give `[]`.
2. **The discriminator gives two entries, not one.** *Not a deviation*; it is recorded so that the
   §4 count does not surprise. §4 step 6 asks for "exactly one difference" and adds that the
   problem counts show `orphan-tag`. Under F10 item 4 that problem is an entry of its own. So the
   comparison has **one content difference and one `problems` entry**, and nothing else.
3. **`fingerprint_store` reads the sidecar twice.** *STRUCTURALLY REQUIRED.* Step 1 matches runs by
   `store_id`, which is in the sidecar, and the prompt's order reads the sidecar at step 5. The
   first read supplies the id only. The comparison and the write use the step-5 reading, as the
   prompt orders.
4. **What `fingerprint_store` returns includes the inventory.** *IMPLEMENTATION CHOICE.* This lets
   `--listing` list the very reading that was fingerprinted, and keeps `fingerprint_store` from
   writing anything but the sidecar (a §7 condition). The command line writes the listing.
5. **A listing path below the store's directory is refused too**, not only one directly in it.
   *IMPLEMENTATION CHOICE.* "Inside the store's own directory" is read to include its
   subdirectories: an rsync of the directory would carry either one.
6. **`--write` exits 0 once it has written**, even when what it replaced differed. *IMPLEMENTATION
   CHOICE.* The prompt fixes the exit codes of the read-only form. With `--write` the person asked
   for the new value to be recorded, so a difference from the old one is expected, not a failure.
   The differences are still printed.
7. **`finish` writes the state even when interrupted while fingerprinting.** *IMPLEMENTATION
   CHOICE.* Any `Exception` is recorded and the state written, as specified. A `BaseException`
   (such as `KeyboardInterrupt`) writes the state with `fingerprint_error: "interrupted: …"`, and
   is then re-raised. That keeps the "never raises for the fingerprint's sake" contract without
   swallowing an interrupt. In the drivers a SIGINT reaches the `terminal` handler first.
8. **`finish` checks for running runs under its own runs root** (`os.path.dirname(self.path)`), not
   `DEFAULT_ROOT`. *IMPLEMENTATION CHOICE.* For every real run the two are the same directory,
   `var/runs/`. For a run in a temporary root (every test, and §4 step 7) it is the root the run
   is in, and nothing under `var/` is read.
9. **Docstrings made false by F12 were corrected** (What shipped, last paragraph). *STRUCTURALLY
   REQUIRED.* They are in `RunRegistry/`, which is in scope, and leaving "never opens" beside a
   `finish` that reads the store would be a statement this commit makes false.
10. **The §4 snapshot hashes bytes and opens no SQLite connection on the originals.**
    *IMPLEMENTATION CHOICE.* Prompt 02's snapshot also counted every table through `sqlite3`
    `mode=ro`. The briefing forbids pointing any new code at an original. A SHA-256 of the bytes,
    with size, `st_mtime_ns` and the listing, is a stronger "identical" than row counts, and it
    reads through no SQLite code.

No UNINTENDED DRIFT.

## Verification performed

### 1. Suites

**Baselines at `4ed0350`** (from the orchestrator at dispatch): AdaptiveLevin 32, ComputeTargets
552, CosmologyModels 39, Datastore 177, LiouvilleGreen 148 (1 skipped), RunRegistry 84.

**After the change**, all six run concurrently, each with
`PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`:

| Suite | Before (`4ed0350`) | After |
|---|---|---|
| `AdaptiveLevin` | 32 OK | 32 OK |
| `ComputeTargets` | 552 OK | 552 OK (the wall-clock flake passed) |
| `CosmologyModels` | 39 OK | 39 OK |
| `Datastore` | 177 OK | 177 OK |
| `LiouvilleGreen` | 148 OK (skipped=1) | 148 OK (skipped=1) |
| `RunRegistry` | 84 OK | **117 OK** (+33, all in `test_store_fingerprint`) |

The four `RunRegistry/__init__.py` docstring corrections came after the concurrent run.
`RunRegistry/tests` was run again on the final tree: 117 OK.

`black --check` is clean on `RunRegistry/` and the two drivers (13 files).

### 2. Scope

`git diff --cached --stat` touches only:
- `RunRegistry/__init__.py`, `RunRegistry/__main__.py`, `RunRegistry/stores.py`;
- `RunRegistry/tests/test_store_fingerprint.py` and `RunRegistry/tests/data/full_store_fingerprint.json`
  (new);
- the two drivers;
- this log, this board, `run-registry`'s board and `docs/OPEN_ISSUES.md`.

`Datastore/` is untouched, and no existing test module changed. No file named `orch_*` was
touched.

### 3. The drivers change only their `finish` calls

`git diff` of `docs/gktk-remedial/scoped_pipeline_run.py` is three one-line changes, each a
`run.finish(` gaining `, fingerprint=True`. That of `docs/handover/quadsource_atol_sweep.py` is six
calls, two wrapped by `black` onto three lines. No other line of either file changed. The
`terminal` handlers (`scoped_pipeline_run.py:307`, `quadsource_atol_sweep.py:490`, `:800`) are
unchanged, and test 12 holds this.

### 4. The demonstration, on copies of the sweep store

1. **Before.** `python -m RunRegistry list`: 7 runs, **none `running`** (5 finished, 2 `unknown`
   pre-registry). 87 GB free. `grep -c '"fingerprint'` on each of the three real sidecars: 0,
   so the first §7 condition does not arise. **The originals were snapshotted read-only**, 18
   files under `var/datastores/` and the backup: the listing, `lstat` size, `st_mtime_ns` and the
   SHA-256 of the bytes (deviation 10). The prefixes are as prompts 01 and 02 recorded them:
   - primaries A3 `fdbe93a5` (live and backup) and sweep `c3cda49d`;
   - sidecars A3 `f7220408`, sweep `a595b7c9`, backup `ce7b476a`.

   `ls var/runs` was also recorded.
2. `cp -p` of `handover-atol-sweep.sqlite`, its four shards **and its sidecar** into
   `var/store-fingerprint-check-04/a/`, with an empty runs root at
   `var/store-fingerprint-check-04/runs/`, which every command below was given. The copied sidecar
   carries **the original's `store_id`, `04198f22f8704c52a252a72566af94ed`**. It names its store
   by the bare file name, so it describes the copy beside it. The copy's primary records its shards
   by legacy absolute path, and `_read_closed_store` says so on every read (`!! Primary database …
   reading them as siblings …`). It reads them as siblings in `a/`, as prompt 01 of
   `datastore-portability` made it do.
3. **`store fingerprint a`**: exit 0, `recorded: none recorded (the sidecar is registry)`,
   `problems: none`.
   - **Overall digest `2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18`.**
   - **The per-class table.** Every class has exactly one tag set, whose count and digest equal the
     class's, except `OneLoopIntegral`, which is empty and has none. The tag sets are prompt 02's
     **B**, **T**, **G** and **Q**, and `[]` for untagged classes.

     | Class | Count | Digest | Tag sets |
     |---|---:|---|---:|
     | `version` | 1 | `482c17e477216a7821fbd15f13e17852f1f9c0a8993a00e12b092fca8d7f2d9a` | 1 |
     | `store_tag` | 10 | `8e36e81ea00925fbccd5611e1009fc01fe56bb4761c9fba39bd3b05ebd395d37` | 1 |
     | `redshift` | 1 740 | `f5851073507ae2f2989c79e49890d880f386c52e27908988584f8508e7e8c95a` | 1 |
     | `wavenumber` | 8 | `f82c942161647580916d3ccfbcf288f366145a047ccc71892b42fdcbfc084bbe` | 1 |
     | `tolerance` | 13 | `5ef7da79b88f47475ba5d8335bf226366dd02259c2d6d02ec47edc8db593aca8` | 1 |
     | `LambdaCDM` | 1 | `dcbaa9d413d45bf56b63a2a6ac1da865159e363f8b5ea2a479265113b82091b2` | 1 |
     | `QCD_Cosmology` | 1 | `178439a39e42aacff29617bf4fafae2d6b4f2faa27a5a8337781de080355fb4f` | 1 |
     | `IntegrationSolver` | 7 | `317b009bb7febb1b3e0980794a994f6d8bdeed3ca784464e2bc462acef1c18c1` | 1 |
     | `GkSourcePolicy` | 2 | `064c7949b3309290eda4bd27713f1b608e066bc0a73fcc815fb3c91720626ec7` | 1 |
     | `QuadSourcePolicy` | 2 | `064c7949b3309290eda4bd27713f1b608e066bc0a73fcc815fb3c91720626ec7` | 1 |
     | `wavenumber_exit_time` | 8 | `045639da7d40c62a233ac245525e0efb1e0d5740bed35c58310026ec1e6cb3fa` | 1 |
     | `BackgroundModel` | 1 | `e3fb6cb1cf7e179e1f4c94d9b27a1deaccf9d9e1c9cf9613ec2cf8fde51827d7` | 1 (**B**) |
     | `TkNumericIntegration` | 8 | `07061fe0dd0fd8890f3b210fa0ea48bc7845ca7bb6a6787be6f639202363b4f8` | 1 (**T**) |
     | `TkWKBIntegration` | 8 | `35a479b6320ff8e3d780be078657323ac18443cc1725e86a18673766266ecd34` | 1 (**T**) |
     | `GkNumericIntegration` | 4 549 | `c123f591d2b04a2d8358825c64340cfad5a60f8f57bfb9f0a6bb527f820e5bfe` | 1 (**G**) |
     | `GkWKBIntegration` | 13 920 | `7397175220005dab7233546c4a4ece0de5200ab76a6c2818b1a6408a1cd5acc5` | 1 (**G**) |
     | `GkSource` | 1 160 | `2245e52abd42fbbb89010a734fd510413b7c01625395f7c500f6569ecdcfa024` | 1 (**G**) |
     | `GkSourcePolicyData` | 1 160 | `df2424eb174d794a042edd3a69853e4ede786fa01107630384e5a17812716761` | 1 |
     | `QuadSource` | 36 | `2dd5a869a5f2017edda7d5a5e63552a9365c6362eed4b0f210116864b2b75526` | 1 (**T**) |
     | `QuadSourceIntegral` | 7 706 | `28c5c5e4d0393b0146c4ef88a88ef3ca5f288e9d26986e10ff113829d1cceb3b` | 1 (**Q**) |
     | `OneLoopIntegral` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (the empty digest) | 0 |

     The counts are prompt 02's, class for class (30 341 records).
   - **Wall time 8.06 s** for the whole command, imports included (`/usr/bin/time -l`: 4.37 s user,
     1.19 s sys). **Peak resident set 232 849 408 bytes (233 MB).** Prompt 02 measured
     `read_inventory` alone at 6.8 s and 228 MB, so the fingerprint adds little to the read.
   - **Size:** 6 284 bytes as compact JSON with sorted keys, and 9 132 bytes as written into the
     sidecar (`indent=2`). The sidecar went from 806 to 10 563 bytes.
   - The copy's `file_state` (every file's SHA-256, size and `st_mtime_ns`, and the listing,
     sidecar included) was **unchanged**. A **second run printed byte-identical output**, and left
     the copy unchanged again.
4. **`store fingerprint a --write`**: exit 0, `>> wrote the fingerprint into the sidecar; it
   replaced none`. The sidecar gained exactly one key, `fingerprint`, and lost none. Every other
   field is value-identical to the copy's sidecar before the write, legacy `created` format and
   legacy-string `copied_from` included. Of the copy's files, only the sidecar changed. Run again
   without `--write`: `recorded: matches the fingerprint taken 2026-09-25T03:05:28+01:00 by a
   person, at 4ed03502afca60787ef0888d9dce5cd2dda9aa12 (dirty)`, exit 0.
5. **`store copy a b`**, a registry copy (`ShardedPool.copy_store`), exit 0. `b`'s sidecar:
   - carries **`a`'s `fingerprint` verbatim**, `taken` included;
   - has a new `store_id`, `1399c02994a0486fbfb62826afb7aa68`;
   - has `copied_from` `{"store_id": "04198f22…", "datastore":
     "var/store-fingerprint-check-04/a/handover-atol-sweep.sqlite"}`;
   - has history `adopt`, `copy`.

   **`store fingerprint b`**: `recorded: matches the fingerprint taken 2026-09-25T03:05:28+01:00 by
   a person …`, exit 0. Its digest and whole per-class table are identical to `a`'s, which answers
   the two-machine question for this copy.
6. **The discriminator.** On `b`'s shard 0, with `sqlite3`, `DELETE FROM QuadSourceIntegral WHERE
   serial = 329386`. The row's label is
   `handover-atol-sweep-a1e-32-r1e-08-QuadSourceIntegral-k3.05e+07-q3.05e+07-r3.05e+07-zresponse0.1-2026-09-24T00:17:34`,
   and the shard's count went 1 955 → 1 954. No journal was left. **`store fingerprint b`**:
   exit 1. The overall digest is now
   `b46b769e74e4de29d2acf6f0a04e606fcebaf675cc763279e0921a12fde3a70c`, and `QuadSourceIntegral` is
   `7705 16c7338a83b6`. The comparison:
   ```
   recorded: 2 difference(s) from the recorded fingerprint:
     !! QuadSourceIntegral: tag set [GkOneLoopDensity, LargestSourceRedshift_2.0636e+16, ResponseRedshiftGrid_145_c51d43ac, ResponseSparsenessZ_12, Run_default, SmallestSourceRedshift_0.1, SourceGridConstruction_2, SourceRedshiftGrid_1740_8d1d43b6, TkOneLoopDensity] differs: 7706 recorded, 7705 now
     !! QuadSourceIntegral: orphan-tag problems: 0 recorded, 1 now
   ```
   That is **exactly one content difference**: `QuadSourceIntegral`, its one tag set, 7 706
   against 7 705. The **one `problems` entry** is prompt 02's `orphan-tag`, the deleted row's nine
   tag rows (deviation 2). A `diff` of the per-class tables before and after differs in those two
   lines only. Every other class and digest is unchanged.
7. **The run record.** In the check directory's runs root,
   `RunRegistry.begin(campaign="store-fingerprint", prompt="04", slug="section-4-finish", …,
   results=a)`. Its manifest's `results_store_id` is `04198f22…`, the copied sidecar's. Then
   `run.finish("done", exit_code=0, fingerprint=True)`, which took 8.2 s. `status.json`:
   - `state` `done`, `exit_code` 0, `fingerprint_sidecar` `"written"`, no `fingerprint_error`;
   - **`status.json`'s `fingerprint` equals the sidecar's**, whole dicts compared: digest
     `2c2dde68…`, `taken` `{"when": "2026-09-25T03:06:44+01:00", "git_head": "4ed03502…",
     "git_dirty": true, "run_id": "store-fingerprint-04-section-4-finish-20260925T030637"}`.

   A second run, `section-4-running`, was begun naming `a` and left `running`.
   `store fingerprint a --runs-root …/runs` exited 1:
   ```
   !! Cannot fingerprint ".../var/store-fingerprint-check-04/a/handover-atol-sweep.sqlite": run store-fingerprint-04-section-4-running-20260925T030645 is running (alive, pid 88480) and names this store by its results and results_store_id. A fingerprint of a store that is being written describes no instant; a stale run is ended by a person, with Run.finish, first. Nothing was read or written
   ```
   It was then finished `done` without a fingerprint, and its `status.json` has none. The check
   root's `list` showed the two runs, both finished.
8. **The originals were re-snapshotted: identical** (`cmp` of the two JSON snapshots). `ls var/runs`
   was identical, and `RunRegistry list` still showed 7 runs, 5 finished and 2 `unknown`. **Nothing
   was created under `var/runs/`.**
9. **`var/store-fingerprint-check-04/` was deleted.** `var/` holds `bootstrap-a3-resume.log`,
   `datastores` and `runs`, as before. Nothing from §4 is committed. The scripts are in the
   session scratchpad: `impl04_snapshot.py`, `impl04_state.py` and `impl04_mutate.py`.

## The deliberate-breakage record

Each mutation was applied to the working tree with every change of this prompt staged, so each
diff is `git diff` (working tree against the index) and applies with `git apply` to this commit.
`RunRegistry.tests.test_store_fingerprint` was run under each (`impl04_mutate.py`). The file was
then restored with `git checkout -- <file>`, after which `git diff --quiet` was true. No mutation
is committed. Mutations (i), (viii) and (ix) are in `Datastore/`: the fingerprint digests what the
inventory gives, so that is where those faults would live, and only the fingerprint's tests are
the subject here.

### (i) A record's timestamp is added to what is digested

The shard read puts each row's `timestamp` into its record's key, so it reaches the canonical line the fingerprint digests.

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..cbb0b7f 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -533,6 +533,8 @@ def read_records(
     for row in rows:
         mapping = row._mapping
         key: Dict[str, Any] = {leaf: canonical(mapping[leaf]) for leaf in leaves}
+        if has_timestamp:
+            key["timestamp"] = str(mapping["timestamp"])
         missing_parent = None
         for field, parent in parents.items():
             of = parent.of
```
**Failed** (failures 3, 33 run):
- `TestGolden.test_the_golden_fingerprint`;
- `TestNotContent.test_no_timestamp_is_in_any_record`;
- `TestNotContent.test_timestamps_labels_and_solver_serials`.

The store with every timestamp moved no longer matches, a record's line now contains `2026-09-24 12:00:00`, and the golden moves.

### (ii) A tag set's digest is taken over every record of its class

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 961953b..750581e 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -837,7 +837,7 @@ def fingerprint_of(inventory, taken=None) -> dict:
                 {
                     "tags": list(tags),
                     "count": len(group),
-                    "digest": _sha256_lines(group),
+                    "digest": _sha256_lines(lines),
                 }
                 for tags, group in _tag_sets(cls.records)
             ],
```
**Failed** (failures 15, 33 run):
- `TestFingerprintCommand.test_a_difference_exits_one`;
- `TestGolden.test_the_golden_fingerprint`;
- `TestListing.test_the_listing_hashes_to_the_fingerprint`, in 8 subtests;
- `TestLocalised.test_adding_a_tag_to_one_record`;
- `TestLocalised.test_deleting_one_quadsourceintegral_row`;
- `TestLocalised.test_deleting_one_value_row`;
- `TestLocalised.test_deleting_the_row_alone_is_also_a_problem_entry`;
- `TestLocalised.test_flipping_one_validated_flag`.

Every tag set now carries its class's digest. So the localisation is lost: every tag set of a changed class is named, not only the one that changed, and the listing's tag-set lines no longer hash to their digests.

### (iii) `value_count` is left out of what is digested

Every line, of the class digest and of each tag set's, is formed with `value_count` set to `null`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 961953b..631218e 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -800,12 +800,18 @@ def _sha256_lines(lines) -> str:
     return hasher.hexdigest()
 
 
+def _line(record) -> str:
+    from Datastore.store_inventory import canonical_json
+
+    return canonical_json(dict(record.as_json(), value_count=None))
+
+
 def _tag_sets(records) -> list:
     """``[(tags, lines)]``: each distinct tag set, sorted, with the canonical lines of the records
     that carry exactly that set, in the records' own order."""
     groups = {}
     for record in records:
-        groups.setdefault(tuple(record.tags), []).append(record.canonical_json())
+        groups.setdefault(tuple(record.tags), []).append(_line(record))
     return sorted(groups.items())
 
 
@@ -829,7 +835,7 @@ def fingerprint_of(inventory, taken=None) -> dict:
     classes = {}
     problems = {}
     for name, cls in inventory.classes.items():
-        lines = [record.canonical_json() for record in cls.records]
+        lines = [_line(record) for record in cls.records]
         classes[name] = {
             "count": len(lines),
             "digest": _sha256_lines(lines),
```
**Failed** (failures 2, 33 run):
- `TestGolden.test_the_golden_fingerprint`;
- `TestLocalised.test_deleting_one_value_row`.

Deleting a value row no longer changes any digest, and the golden moves.

### (iv) `store fingerprint` without `--write` writes the sidecar

It writes whenever the sidecar would accept it.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 961953b..7cbb810 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1152,7 +1152,7 @@ def fingerprint_store(primary, *, write=False, runs_root=None, taken_by=None) ->
 
     # 6. the write, only when asked
     wrote = False
-    if write:
+    if write or reading.ok:
         if not reading.ok:
             raise RuntimeError(
                 f'Cannot record a fingerprint for "{primary}": its sidecar "{reading.path}" is not '
```
**Failed** (failures 1, 33 run):
- `TestFingerprintCommand.test_read_only_then_write_then_matches`.

The store directory's `file_state` changed under the read-only command: the sidecar was rewritten.

### (v) The running-run refusal is removed

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 961953b..43663e2 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1105,7 +1105,7 @@ def fingerprint_store(primary, *, write=False, runs_root=None, taken_by=None) ->
     running = _running_runs_naming(
         "fingerprint", [primary], store_id, runs_root, exclude=taken_by
     )
-    if running:
+    if False:
         raise RuntimeError(
             f'Cannot fingerprint "{primary}": {_describe_running(running)}. A fingerprint of a '
             f"store that is being written describes no instant; a stale run is ended by a "
```
**Failed** (failures 9, errors 1, 33 run):
- ERROR `TestFinish.test_another_running_run_is_recorded_and_the_run_still_ends`;
- `TestRunningRefusal.test_refused_alive_or_stale_by_path_or_store_id`, in 8 subtests;
- `TestRunningRefusal.test_the_command_refuses`.

Six of the eight refusal cases went ahead (`RuntimeError not raised`). The two by-path cases with `write=True` were refused only by the sidecar check (that store has no sidecar), whose message names no run. The command went ahead with and without `--write`. A finish that should have recorded a refusal found no `fingerprint_error` (`KeyError`).

### (vi) `copy_store` drops `fingerprint`

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 961953b..7ddcf6c 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -718,6 +718,7 @@ def copy_store(src, dst, purpose, runs_root=None) -> dict:
     step = "write the destination's sidecar"
     try:
         fields = _copy.deepcopy(reading.fields)
+        fields.pop("fingerprint", None)
         fields["store_id"] = new_store_id()
         fields["datastore"] = dst.name
         fields["name"] = dst.stem
```
**Failed** (errors 1, 33 run):
- ERROR `TestSidecarField.test_copy_and_move_carry_it_verbatim`.

The copy's sidecar has no `fingerprint` (`KeyError`).

### (vii) A fingerprint error in `finish` is raised instead of recorded

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 4f4075c..bd223dc 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -329,7 +329,7 @@ class Run:
             try:
                 taken = self._fingerprint_results()
             except Exception as e:
-                taken = {"fingerprint_error": f"{type(e).__name__}: {e}"}
+                raise
             except BaseException as e:
                 # an interrupt while fingerprinting: the state is still written, then it goes on
                 self._update(
```
**Failed** (errors 2, 33 run):
- ERROR `TestFinish.test_a_refused_store_is_recorded_and_the_run_still_ends`;
- ERROR `TestFinish.test_another_running_run_is_recorded_and_the_run_still_ends`.

`finish` raised `RuntimeError` (the journal refusal; the other running run) instead of writing `done` / `failed`, so the run was left `running`.

### (viii) One key field is removed from one factory's `inventory_records`, with no format bump

`TkWKBIntegration` loses `rho_gauss_order` from its key.

```diff
diff --git a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
index e4b57e9..26e9d55 100644
--- a/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/TkWKBIntegration.py
@@ -822,7 +822,7 @@ class sqla_TkWKBIntegration_factory(SQLAFactoryBase):
             table,
             tables,
             context,
-            leaves=("rho_gauss_order", "z_init"),
+            leaves=("z_init",),
             parents={
                 "model": Parent("model_serial", "BackgroundModel"),
                 "k": Parent("wavenumber_exit_serial", "wavenumber_exit_time"),
```
**Failed** (failures 1, 33 run):
- `TestGolden.test_the_golden_fingerprint`.

The golden fingerprint differs: `TkWKBIntegration`'s records, and so its digests and the overall digest, change with no format bump.

### (ix) Records are digested in serial order instead of canonical order

The inventory's records are left in (shard, serial) order, the order the rows were read in, instead of being sorted by canonical JSON.

```diff
diff --git a/Datastore/store_inventory.py b/Datastore/store_inventory.py
index 3edb821..a4463f5 100644
--- a/Datastore/store_inventory.py
+++ b/Datastore/store_inventory.py
@@ -653,7 +653,7 @@ def _combine(
         )
 
     stamps = [ts for _, _, _, ts in rows if ts is not None]
-    records = tuple(sorted((r for _, _, r, _ in rows), key=Record.canonical_json))
+    records = tuple(r for _, _, r, _ in rows)
     return ClassInventory(
         name=name,
         replicated=replicated,
```
**Failed** (failures 13, 33 run):
- `TestGolden.test_the_golden_fingerprint`;
- `TestListing.test_the_listing_hashes_to_the_fingerprint`, in 10 subtests;
- `TestPhysical.test_other_serials_and_other_shards`;
- `TestPhysical.test_sharded_rows_on_a_third_shard`.

A store under other serials, or with its rows on other shards, gives another digest, the class digests no longer equal the hash of their sorted lines, and the golden moves.

## Observations not acted on

- **`GkSourcePolicy` and `QuadSourcePolicy` have the same digest** on the sweep store, `064c7949…`.
  Both classes key on (`Levin_threshold`, `numeric_policy`), and on this store each holds the same
  two records. The overall digest keeps them apart, because it keys each class by name. This
  bears on `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`, and does not change it. No new
  issue is opened.
- **Every read of the sweep store copy prints `_read_closed_store`'s legacy-path notice** on stdout,
  because the primary records its shards by legacy absolute path. That is the `datastore-portability`
  behaviour working as designed. `store fingerprint`'s output therefore begins with a `!!` line on
  such a store. It is not an error, and the exit code is unaffected.
- **`store copy` imports `ray`**, through `ShardedPool` (it did before this prompt;
  `test_store_command_line` asserts it). `store fingerprint` imports it too, through the factory
  map `read_inventory` needs. It never initialises Ray, and test 8 asserts that in a child.
  `import RunRegistry`, `list` and `store show` still import neither `ray` nor `sqlalchemy`.
- **`Run.finish(..., fingerprint=True)` takes about as long as a full inventory**, 8 s on the sweep
  store, before the terminal state is written. A run killed with SIGKILL in that window stays
  `running`, and so reads as stale, which is the registry's existing account of a SIGKILL.
- **Prompt 05** can use `python -m RunRegistry store fingerprint <primary> --write` on each real
  store, with the default runs root, once `RunRegistry list` shows nothing `running`.

## State handed to the next prompt

- `from RunRegistry.stores import FINGERPRINT_FORMAT, fingerprint_of, compare_fingerprints,
  fingerprint_store, listing_lines, write_listing`.
  - `fingerprint_store(primary, write=True)` records a fingerprint in a problem-free registry
    sidecar, replacing only that field, after refusing a store any `running` run names.
  - Its result's `comparison` is `None` when nothing was recorded.
- `python -m RunRegistry store fingerprint PRIMARY [--write] [--listing PATH] [--runs-root DIR]`.
  - It exits 0 on a match or when nothing is recorded, 1 on a difference or refusal, and 0 after a
    successful `--write`.
  - `--listing` writes a new file outside the store's directory, whose lines hash to the digests.
- `Run.finish(state, exit_code, fingerprint=True)` puts `fingerprint` (or `fingerprint_error`) and
  `fingerprint_sidecar` in `status.json`. `scoped_pipeline_run.py` and `quadsource_atol_sweep.py`
  (`--build` and the sweep) pass it on every finish after the pipeline.
- **For prompt 05 (D3):** none of the three real sidecars has a fingerprint yet. The sweep store's
  digest, as computed on its copy at `4ed0350` plus this change, is `2c2dde68…`. Prompt 05 should
  find the same digest on the original, since the copy was `cp -p`'d and not changed before step 3.
- **The golden** `RunRegistry/tests/data/full_store_fingerprint.json` pins format 1. A change to
  the inventory's records, or to how a digest is formed, must bump `FINGERPRINT_FORMAT` and
  regenerate the golden in the same commit.
- **Baselines:** `RunRegistry` **117** (84 + 33); the others unchanged.
