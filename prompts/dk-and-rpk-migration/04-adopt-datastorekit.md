# Prompt 04 — adopt `datastorekit`

**Campaign:** [`README.md`](README.md) · **Board items:** **A1**–**A5** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 01–03, after the user reviewed the plan (`4532188`).
**Gate:**
- 03 landed and reviewed;
- `datastorekit` `0.2.2` is installed from the pin at `6d8ee9ae…`, not editable (board, 03's row);
- `git status` is clean;
- nothing is `running` in the registry;
- R0's stores exist (02).

**Closes:** nothing by name. **Opens:** A5's issue, and anything else you find.
**Recommended model:** **Opus**. A wide, mostly mechanical change: about 180 import statements
in some 100 files, with 26 test modules deleted and four staying tests adapted. What it needs is
exactness, and proof that nothing was missed.

**Read first:**

1. [`README.md`](README.md): §0.2, §1, §2.2 (04's charter), §4 (what 04 ships), §4.3 (the
   `CLAUDE.md` section this prompt writes), §5, and §6 (U3, and C2–C6, C10).
2. **DatastoreKit's SGK checklist**, `/Users/ds283/Documents/Code/DatastoreKit/docs/adoption/secondarygwkit.md`:
   items 2, 3, 4, 5 and 7, the `v0.2.2` addendum's "SGK's own stand-in", and its appendix. Its
   `path:line`s are SGK's at `b510bc9`. **Read it; do not edit anything in that repository.**
3. DatastoreKit's `docs/adoption/README.md` §2 (the module map) and §3 (what the wheel does not
   ship).
4. `CLAUDE.md`, the section "The datastore layer is frozen — it is moving to DatastoreKit"
   (`:47-73`), which this prompt replaces.
5. The staying tests this prompt edits beyond imports:
   - `Datastore/tests/test_inventory_retired.py`;
   - `Datastore/tests/test_inventory_declarations.py:445-471`, the one test that moves;
   - `Datastore/tests/test_inventory_report.py`, where it moves to;
   - `Datastore/tests/standin_pool.py:43-48`;
   - `Datastore/tests/test_wkb_numeric_parent_key.py:86-88` and `:909`.
6. [`measure/out/b510bc9-datastore-tests-by-module.txt`](measure/out/b510bc9-datastore-tests-by-module.txt)
   and its script.

Line numbers are at `b510bc9`.

---

## 1. What is wanted

SGK's datastore layer, 17 files, is replaced by the installed package. The package is the same
code at a later release (DatastoreKit `PROVENANCE.md`), so this is a deletion and a rewrite of
module paths. **No name that is imported changes**, and no behaviour SGK relies on changes
(checklist items 5, 6 and 8, with both addenda). The swap is one commit. The tree cannot pass
between its parts: the deletion, the rewrite and the test changes stand or fall together.

- **A1 — the layer goes.**
- **A2 — every layer import outside `prompts/` is rewritten**, by module path only.
- **A3 — `Datastore/tests/` keeps only SGK's tests.**
- **A4 — the prose this commit makes false.**
- **A5 — the six `docs/` files that already fail** are recorded as an issue.

**What does not change.**
- What the code writes. Proved here by fingerprinting R0's store through the package (§3.3), and
  by R1 (05).
- Every factory but its imports; `config/`; `main.py` and the six `extract_*.py` but their
  imports; every pool construction and every call.
- The 98 factory imports from outside the factory package, and every path that reaches a factory
  by file (U3).
- `prompts/`: its 51 layer imports are history (README §1).
- `RayTools/RayWorkPool.py`, but its line 8 (U12).

---

## 2. What to change

### 2.1 A1 — delete the layer

- **Delete** with `git rm`: `Datastore/object.py`, `contract.py`, `replication.py`,
  `shard_paths.py`, `store_reader.py`, `store_inventory.py`; `Datastore/SQL/schema.py`,
  `ShardedPool.py`, `Datastore.py`, `ClientPool.py`, `SerialPoolBroker.py`, `ProfileAgent.py`;
  `Datastore/SQL/ObjectFactories/base.py`; and `tools/sharded_store.py` and
  `tools/shard_key_audit.py`. That is 15 files.
- **Empty** `Datastore/__init__.py` and `Datastore/SQL/__init__.py`, to 0 bytes (U3). They stay
  only so that the factories' folders remain regular packages.
- **Keep** `tools/__init__.py` (empty) and `tools/inventory_report.py`, which `main.py:93`
  imports.

### 2.2 A2 — rewrite the imports

**The rule.** Every statement, string literal and `import_module` call outside `prompts/` that
names one of the 17 layer modules is rewritten by DatastoreKit's adoption README §2 map, with the
**module path only**:

| SGK module | becomes |
|---|---|
| `Datastore` (the root: `DatastoreObject`, `contract`) | `datastorekit` |
| `Datastore.<object, contract, replication, shard_paths, store_reader, store_inventory>` | `datastorekit.<same>` |
| `Datastore.SQL`, `Datastore.SQL.<ShardedPool, Datastore, ClientPool, SerialPoolBroker, ProfileAgent, schema>` | `datastorekit.SQL`, `datastorekit.SQL.<same>` |
| `Datastore.SQL.ObjectFactories.base` | `datastorekit.SQL.factory_base` |
| `tools.sharded_store`, `tools.shard_key_audit` | `datastorekit.tools.<same>` |

- **Names, aliases, nesting and formatting stay** as they are, but where `black` reflows a line.
- **A factory import is not a layer import.** `Datastore.SQL.ObjectFactories.<anything but base>`
  stays.
- **This includes:**
  - the nested imports (41 at `b510bc9`, inside functions and `if`s);
  - `RayTools/RayWorkPool.py:8` (RayPoolKit decision U12): `from datastorekit.SQL.ShardedPool
    import ShardedPool`, and nothing else in that file;
  - every `docs/` statement (C2), including the three `import_module` calls at
    `docs/datastore-integrity-audit/replicated_write_fault_probe.py:37-39`;
  - the string targets in staying tests (§2.3).
- **The module, not the class.** `datastorekit.SQL.__init__` rebinds `datastorekit.SQL.Datastore`
  to the actor class, as SGK's did. Code that needs the module reaches it with
  `importlib.import_module("datastorekit.SQL.Datastore")`, as `test_wkb_numeric_parent_key.py:88`
  already does with SGK's.
- **Private names.** Staying code imports one private name of the layer: `_merge_digests`, at
  `Datastore/tests/test_quadsource_parent_key.py:47`. It exists in the package
  (`datastorekit/store_inventory.py:811` at `v0.2.2`). Check every other name each rewritten
  statement imports in the same way: `getattr` on the package module, in a loop, from the
  script's JSON. A name the package lacks is a stop (§5). The ten `_factories` imports in the six
  failing `docs/` files are the exception, as A5 says.

**How.** Drive the rewrite from DatastoreKit's measuring script, not from a hand search:
1. From DatastoreKit's root, with its venv, run
   `./venv/bin/python docs/extraction/measure_client_imports.py sgk --commit HEAD --json`. Save
   the output under `var/dk-and-rpk-migration-04/`.
2. Write **`prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py`**. It reads that JSON
   and rewrites each listed statement, string and `import_module` call **at its file and line**,
   outside `prompts/`. It refuses a line whose text does not name the module the JSON says, so a
   stale line number cannot rewrite the wrong line. Commit it with this prompt, as the record of
   how the rewrite was made.
3. Run it after A1's and A3's deletions, so that it does not rewrite files that are going, and
   **before any hand edit of a staying file** (§2.3, §2.4), so that every line number in the JSON
   is still the file's.
4. Re-run the measuring script **at a commit of the staged tree** (§3.1). The script reads
   through `git` only, so make a commit object without moving any branch:
   `git add` the work, then `c=$(git commit-tree $(git write-tree) -p HEAD -m "04 scratch")`,
   then measure `--commit $c`. Never commit to or amend the campaign branch before §4.

### 2.3 A3 — the tests

1. **Delete the 26 modules the package carries** (checklist item 7). They are the `DELETE` rows of
   `measure/out/b510bc9-datastore-tests-by-module.txt`, 419 tests:
   - the 8 moved verbatim: `test_shard_paths`, `test_shard_file_name`,
     `test_shardedpool_shard_paths`, `test_copy_move_store`, `test_delete_store`,
     `test_sharded_store_script`, `test_shard_key_audit_copy` and
     `test_shard_key_audit_refusals`;
   - the 18 ported: `test_replicated_write`, `test_reconcile_at_open`, `test_prune_at_open`,
     `test_version_row_at_open`, `test_read_only_pool`, `test_one_timestamp_per_write`,
     `test_absolute_shard_record_refused`, `test_closed_store_refusals`, `test_store_inventory`,
     `test_store_schema`, `test_store_reader`, `test_foreign_key_check`, `test_schema_builder`,
     `test_inventory_declarations`, `test_declared_facts`, `test_layer_registry`,
     `test_drop_refuses_dangling_references` and `test_layer_is_generic`.

   Also delete **`Datastore/tests/schema_description.py`** (C5). Its only importer is
   `test_schema_builder`.
2. **Before deleting `test_inventory_declarations`**, move
   `TestResolve.test_the_report_renders_both_cosmology_types` (`:445-471`) into
   `Datastore/tests/test_inventory_report.py` (C4). It goes in a new class there,
   `TestTheReportRendersBothCosmologyTypes`, whose docstring says where it came from and why it
   stayed in SGK: it asserts on SGK's `tools.inventory_report`, the one test DatastoreKit's
   `compare_ported_tests.py` declares `NOT_PORTED` (U16).
   - Its body is unchanged. Bring the imports and fixtures it uses with it:
     `read_inventory`, `build_full_store`, `format_inventory_report`, `_factories`, `tempfile`,
     `re`, `Path`.
   - Show by `ast` that its body (docstring stripped) dumps identically before and after.
3. **`Datastore/tests/test_inventory_retired.py`** (C3):
   - **keep**:
     - `test_no_factory_has_the_old_method`;
     - `test_config_sharding_has_no_merge_policies`;
     - `test_the_grep_prints_nothing`, its grep now over every `*.py` with no exclusion;
   - **`test_the_datastore_actor_class_has_no_method`** parses the installed package's source:
     the file at `importlib.util.find_spec("datastorekit.SQL.Datastore").origin`, in place of
     `Datastore/SQL/Datastore.py`;
   - **`test_sharded_pool_has_neither_method_nor_parameter`** reads `ShardedPool` from
     `datastorekit.SQL.ShardedPool`, by the A2 rewrite;
   - **delete** `test_the_excluded_file_holds_only_the_new_service_helper`, `HELPER_MODULE`,
     `HELPER_MATCH` and the grep's pathspec exclusion. Their subject, `test_store_inventory.py`,
     is deleted;
   - **the module docstring** says what changed and why: the layer is the package's, so SGK
     checks that the release it pins lacks the old service.

   7 tests become 6.
4. **`Datastore/tests/standin_pool.py`** (C6): only its three `import_module` strings
   (`:43-45`) change, to `"datastorekit.SQL.ShardedPool"`, `"datastorekit.SQL.Datastore"` and
   `"datastorekit.SQL.SerialPoolBroker"`. It does not gain the package stand-in's
   `_ray_is_running` or `ray.kill`.
5. **`Datastore/tests/test_wkb_numeric_parent_key.py:88`**: the string becomes
   `"datastorekit.SQL.Datastore"`. Its comment at `:86-87` names the package.
6. **Everything else in the 18 staying modules, the fixtures, `ComputeTargets/tests` and
   `RunRegistry/tests` changes by A2's rewrite only.** That covers `real_store_fixtures.py`,
   `shard_store_fixtures.py`, `standin_pool.py`, `inventory_report_parsing.py` and `data/`, which
   all stay.

### 2.4 A4 — the prose this commit makes false (C10)

1. **`CLAUDE.md`.** Replace the section "The datastore layer is frozen — it is moving to
   DatastoreKit" (`:47-73`) with a section titled **"The datastore layer is `datastorekit`"**,
   saying, in `CLAUDE.md`'s voice:
   - **What it is.** Since this commit's date the generic layer is the package `datastorekit`
     (DatastoreKit, its local path and GitHub URL), pinned in `requirements.txt` at `v0.2.2`.
     ChamPBH and StochasticInstantons use it too, and SGK adopted it in `dk-and-rpk-migration`.
   - **It is not this repository's code.** A defect in it is recorded on the board of the
     campaign that found it, and in `docs/OPEN_ISSUES.md`. It is reported to DatastoreKit, fixed
     there in a release, and reaches SGK by moving the pin in a commit of its own. It is never
     patched here, and never worked around. A prompt whose acceptance cannot pass without a
     change to the package stops and asks.
   - **Pins.** A pin is a tag. An editable install is never used for a production run. SGK keeps
     its own `ray` and `SQLAlchemy` pins.
   - **Imports name the package.** `from datastorekit import DatastoreObject`;
     `datastorekit.SQL.ShardedPool`; `datastorekit.SQL.factory_base`. The actor's module is
     `importlib.import_module("datastorekit.SQL.Datastore")`. The tools are
     `python -m datastorekit.tools.sharded_store` and `python -m datastorekit.tools.shard_key_audit`.
   - **SGK's side of the layer** changes as usual: the factories under
     `Datastore/SQL/ObjectFactories/`, `config/datastore.py` and `config/sharding.py`,
     `tools/inventory_report.py`, `Datastore/tests/`, and every caller. `Datastore/__init__.py`
     and `Datastore/SQL/__init__.py` are empty, and exist only to make the factories' folders
     packages.
   - **`RayTools/RayWorkPool.py` is frozen** while it moves to RayPoolKit, except its one import,
     which RayPoolKit's decision U12 allows. `dk-and-rpk-migration` prompt 07 replaces it with
     `raypoolkit`.

   Keep it short: the section it replaces is 27 lines, and this one should not be longer. 07
   extends it to `raypoolkit` (README §4.3).
2. **`RunRegistry/stores.py:5`**: "`ShardedPool` and `tools/sharded_store.py` know nothing about
   it" names a deleted file. Write `datastorekit`'s `ShardedPool` and its
   `datastorekit.tools.sharded_store`, and change nothing else in the docstring.
3. **Nothing else.** Dated `docs/` records, `docs/OPEN_ISSUES.md`'s historical prose (`:685`) and
   `prompts/` are left (C10). `CLAUDE.md`'s test command is 09's.

### 2.5 A5 — the six `docs/` files that already fail

Ten statements in six files import `_factories` from `Datastore.SQL.Datastore`, which left that
module at `19f07ee` (checklist item 3). They are:
- `docs/datastore-integrity-audit/fk_sweep_probe.py`;
- `docs/datastore-integrity-audit/lookup_keys/common.py`;
- `docs/datastore-integrity-audit/lookup_keys/probe_03_column_existence.py`;
- `docs/datastore-integrity-audit/tq_fk_probe.py`;
- `docs/tolerance-convergence/inventory.py`;
- `docs/tolerance-convergence/order_audit.py`.

Six more fail through `lookup_keys/common.py`. A2 rewrites their module paths like every other
import, so each still fails, now on `datastorekit.SQL.Datastore`. **Do not repair them** (C2;
README §6.3).
- Show, for each of the six, that it failed before (at `HEAD`, before your change) and fails
  after, with the same `ImportError` on `_factories`. Run each with `PYTHONPATH=.` and no
  arguments, outside `var/`, and stop it if it gets past its imports.
- Open the issue **`[04-six-docs-probes-import-factories-from-the-layer]`** on this board's §3,
  and add its row to `docs/OPEN_ISSUES.md`. It names the files, the commit that broke them
  (`19f07ee`), and the fix: `config.datastore.factories`. Its hook says the decision is the
  user's (README §6.3).

---

## 3. Verification

### 3.1 By DatastoreKit's script, at a commit of the staged tree

Run `measure_client_imports.py sgk --commit <scratch commit>` (§2.2, step 4) from DatastoreKit's
root, with its venv, on the final staged tree. Save the Markdown output as
`measure/out/04-datastorekit-measure_client_imports.md`. The scratch commit will not survive into
history, so the log shows that the landed commit differs from the measured tree only by the
records: `git diff --stat <scratch> HEAD` lists the log, this board, the README, `docs/OPEN_ISSUES.md`
and the saved measurement, and nothing else. It must show:
- **layer files absent:** the 15 of A1. `Datastore/__init__.py` and `Datastore/SQL/__init__.py`
  are present. Show by `git cat-file -s` that both are 0 bytes;
- **outside `prompts/`:** 0 import statements, 0 string literals and 0 `import_module` calls
  naming a layer module;
- **inside `prompts/`:** the 51 statements of `b510bc9`, unchanged, and its strings unchanged;
- **factory imports:** `b510bc9`'s 100, less exactly those in the deleted test modules. Derive the
  expected figure from `b510bc9`'s appendix, and show it.

### 3.2 The suites

All six roots, serially, in the foreground. Datastore runs in its four groups.

| Root | Expected | Why |
|---|---|---|
| `Datastore/tests` | **320** | 739 − 419 deleted, + 1 moved (§2.3.2), − 1 removed (§2.3.3) |
| `RunRegistry/tests` | 03's count | imports only |
| `ComputeTargets/tests` | 693 | imports only |
| `AdaptiveLevin`, `CosmologyModels`, `LiouvilleGreen` | 32, 39, 148 (1 skipped) | untouched |

- Re-derive the Datastore figure with `measure/scripts/count_datastore_tests.py`. Its `DELETE`
  rows must all be absent, and it must load 320 tests in 18 modules.
- Every test that ran before and still exists has the same id and passes, but the six of
  `test_inventory_retired`, which §2.3.3 changes. Compare the sorted lists of test ids before and
  after, from `unittest`'s verbose output, and account for every difference.
- `black --check` is clean on every Python file this prompt touches.

### 3.3 Through the package, on real things

1. `./venv/bin/python -m datastorekit.tools.sharded_store --help` and
   `./venv/bin/python -m datastorekit.tools.shard_key_audit --help` each exit 0.
2. `PYTHONPATH=. ./venv/bin/python -m RunRegistry list` runs, and shows R0's runs.
3. **R0's store, read through the package.**
   `PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint var/datastores/dk-and-rpk-migration-02-reference.sqlite`,
   **without `--write`**, gives format 6, digest `39809dca…`, with all 21 classes equal to the
   sidecar's.
   - This is the first real store read through the package's `read_inventory`.
   - Record every store file's SHA-256 and mtime before and after. They must not change, and no
     journal may appear.
4. **Every entry point imports.** Byte-compile `main.py`, the six `extract_*.py`,
   `extract_common.py` and every rewritten `docs/` file, with `py_compile`.
   - Import, with `PYTHONPATH=.`, every rewritten module outside `docs/` that can be imported
     without side effects. `main.py` cannot be imported (`CLAUDE.md`); its imports are checked
     by `test_main_plumbing`'s `ast` extraction and by 05.
   - Import every `docs/` file the rehearsal and the tests reach:
     - `docs/handover/quadsource_atol_sweep.py`;
     - the two `docs/source-remediation-verification/` scripts;
     - `docs/a3-v2-readiness/rehearsal/store_soundness.py`.

     Load each by path, as `RunRegistry/tests` loads the first. None of them may start Ray on
     import.
5. **The string default.** `config.defaults.DEFAULT_STRING_LENGTH` and
   `datastorekit.defaults.DEFAULT_STRING_LENGTH` are both 256, so the layer's own tables keep
   their schema. Show both.
6. `ray.is_initialized()` is false after every step of §3.3 that imports in-process.

### 3.4 Deliberate breakage (README §5 rule 8)

Each must make the named check fail; then restore it.
- (i) one factory's `SQLAFactoryBase` import is pointed back at
  `Datastore.SQL.ObjectFactories.base`: that factory's tests fail on import, and §3.1's script
  counts 1 layer import;
- (ii) `from Datastore import DatastoreObject` is restored in one `ComputeTargets/` module: its
  importers' tests fail with `ImportError`, since `Datastore/__init__.py` is empty;
- (iii) the moved test is deleted: the Datastore count is 319;
- (iv) `standin_pool.py`'s `"datastorekit.SQL.ShardedPool"` is reverted to
  `"Datastore.SQL.ShardedPool"`: `test_quadsource_policy_main` fails;
- (v) `test_inventory_retired`'s actor check is pointed at a copy of the package's
  `SQL/Datastore.py` with a method `inventory` added: that test fails.

Record each in the log as a diff, exactly as applied, with what failed. Check each with
`git apply --check` and `-R --check`, and leave `git status` clean.

---

## 4. Acceptance

1. **A1–A5** as §2 says.
2. §3.1's measurement, at a commit of the staged tree: 15 absent, two 0-byte `__init__.py`, and 0
   layer imports, strings or `import_module` calls outside `prompts/`.
3. §3.2's counts, each accounted for, and `black --check` clean.
4. **§3.3.3: R0's reference store fingerprints `39809dca…` through the package**, and is
   unchanged.
5. §3.4's five breakages, each caught.
6. **The records**, in the same commit:
   - this board: the §1 row for 04, items A1–A5, A5's §3 entry, and
     `### 5.4 After prompt 04 (<date>)`. It names the rollback: `git revert` alone, since an
     unimported package is inert (README §4.2);
   - `docs/OPEN_ISSUES.md`: A5's row, under a new section for this campaign, with the count and
     date corrected by counting;
   - README §4's "04 ships" says it shipped as named, or names what differs.

---

## 5. Stop conditions — stop and ask

- **A name a staying statement imports is not in the package** (other than A5's `_factories`), or
  a staying test depends on a deleted test module or file other than those §2.3 names.
- A staying test fails for any reason other than an import §2.2 should have rewritten. Never edit
  a test's assertions to pass, other than §2.3's.
- **§3.3.3's fingerprint differs from `39809dca…`**, or the store changes. Record both, and do not
  work round it. A difference here is a package defect (README §5 rule 11).
- The package refuses to open R0's store, or raises anything the layer did not.
- The rewrite would need a change in DatastoreKit, a change to any factory beyond its imports, or
  a change to `config/`.
- §3.1 finds a layer import, string or `import_module` call outside `prompts/` that the rewrite
  cannot reach.
- Anything would start Ray, open a store under `var/` other than R0's reference read-only, or
  write under `var/datastores/`.

---

## 6. What this prompt does not do

- **Files it changes:**
  - the 15 deletions and the two emptied `__init__.py` (A1);
  - every file the script lists with a layer import outside `prompts/` (A2);
  - the 26 test modules and `schema_description.py` deleted, plus `test_inventory_report.py`,
    `test_inventory_retired.py`, `standin_pool.py` and `test_wkb_numeric_parent_key.py` (A3);
  - `CLAUDE.md` and `RunRegistry/stores.py` (A4);
  - new: `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py` and
    `measure/out/04-datastorekit-measure_client_imports.md`;
  - the log, this board, the README's §4 and `docs/OPEN_ISSUES.md`.

  Nothing else.
- It does not touch `RayTools/` beyond `:8`, or any pool site.
- It does not move or rename the factories (U3), and does not repair A5's files.
- It does not change `CLAUDE.md`'s test command, which is 09's.
- It runs no build. R1 is 05.

---

## 7. The log and the board

`logs/04-adopt-datastorekit.md`, using README §5.1. In addition, it gives:
- the rewrite's counts, by group as the checklist tabulates them, before and after;
- §3.1's measurement, against `b510bc9`'s appendix;
- §3.2's table and the test-id comparison;
- §3.3's outputs, the fingerprint in full;
- A5's before-and-after failures;
- each breakage (i)–(v), as a diff, with what failed;
- **for 05:** the commit, the venv's `package_provenance()`, and anything 05's launches should
  expect to differ from R0's.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`. Deletions are staged with `git rm`.
