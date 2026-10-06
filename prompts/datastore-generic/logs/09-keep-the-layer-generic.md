# Log — prompt 09: keep the layer generic

**Subject:** *Keep the layer generic: a guard over every file of the generic layer* · **Model:**
Claude Sonnet 5.5 · **Date:** 2026-10-06 · **Base:** `c96e56d` (code as at `3ff1c80`), clean ·
**Result:** landed; K2 is half done: the fixture half is here, the rehearsal rebuild is the review's.
- **K1:** `Datastore/tests/test_layer_is_generic.py` derives the layer's files and checks, in each,
  the code, the strings and docstrings, the comments and the imports. Prompt 07's check of
  `ShardedPool.py` and prompt 08's two checks of `store_inventory.py` are folded into it and
  deleted. The three sites it flagged on today's layer are fixed.
- **K2:** every suite, and README §5 rule 9 on fixtures against `dbd8a51`: holds, byte for byte.

No stop condition fired. Closes `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]`.
Narrows nothing and opens nothing. The index goes from 108 to 107. The commit is `e980b6e`.

## What shipped

**`Datastore/tests/test_layer_is_generic.py` (new; 8 tests).**
- `layer_files()`: every `.py` under `Datastore/` except `Datastore/tests/` and
  `Datastore/SQL/ObjectFactories/`, but with `ObjectFactories/base.py`; plus `RayTools/*.py`. It
  globs the filesystem, not `git ls-files`. 17 files today:
  `Datastore/__init__.py`, `Datastore/contract.py`, `Datastore/object.py`,
  `Datastore/replication.py`, `Datastore/shard_paths.py`, `Datastore/store_inventory.py`,
  `Datastore/store_reader.py`, `Datastore/SQL/__init__.py`, `Datastore/SQL/ClientPool.py`,
  `Datastore/SQL/Datastore.py`, `Datastore/SQL/ProfileAgent.py`,
  `Datastore/SQL/SerialPoolBroker.py`, `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/schema.py`,
  `Datastore/SQL/ObjectFactories/base.py`, `RayTools/__init__.py`, `RayTools/RayWorkPool.py`.
  They are the audit's 12, `contract.py` (07), `ObjectFactories/base.py` and the three
  `__init__.py`.
- The vocabulary, 234 words: 36 registry tables and registered class names, 184 identifier columns
  (an underscore or a capital), 11 project packages (`AdaptiveLevin`, `ComputeTargets`,
  `CosmologyConcepts`, `CosmologyModels`, `LiouvilleGreen`, `MetadataConcepts`, `Quadrature`,
  `RunRegistry`, `Units`, `thirdparty`, `tools`) and 08's five names (`cosmology`,
  `cosmology_type`, `cosmology_serial`, `COSMOLOGY`, `model_ids`); less `LAYER_WORDS` (the
  `Datastore.contract` names, `serial`, `version`, `timestamp`, `stepping`, `validated`, each with
  its reason in a comment). In strings and comments, also 07's flag phrases and 08's fragments
  `_tags`, `*Value` and `Run_<`.
- The tests:
  - `TestTheLayer`: `test_the_derived_files_hold_the_audits_list`,
    `test_no_test_and_no_client_factory_is_in_it`,
    `test_the_vocabulary_holds_the_registry_and_the_packages`;
  - `TestTheLayerNamesNoProjectWord`: `test_no_code_name`, `test_no_string_or_docstring`,
    `test_no_comment`;
  - `TestTheLayerImportsNoProjectPackage`: `test_every_import_is_allowed`, `test_the_allow_list`.
- The allow-list (`import_allowed`), entry by entry:
  - the standard library, `ray` and `sqlalchemy`: what the layer is built on;
  - `Datastore.*`, but of `Datastore.SQL.ObjectFactories` only `.base`: the layer's own modules,
    and the factory contract the actor imports;
  - `RayTools.*`: the layer's own modules;
  - `config.defaults`: `DEFAULT_STRING_LENGTH`, the width of the layer's own text columns
    (`ShardedPool.py:35`, `ProfileAgent.py:10`); the user's choice of 2026-10-06, over a constant
    in `Datastore/contract.py`;
  - `utilities`: `WallclockTimer` and `format_time` (`Datastore.py:20`, `ProfileAgent.py:11`,
    `RayWorkPool.py:9`), a shared helper that imports only the standard library; the user's choice
    of 2026-10-06.
  Relative imports are resolved before the check.

**The three sites, fixed in the layer.** No behaviour changes.
- `Datastore/store_inventory.py`: `read_records`' local `parent_serial` is `parent`, on its eight
  lines (607, 610, 611, 615, 643, 646, 647, 649) and nowhere else. The function's other uses of the
  name `parent` (its `for field, parent in declared` loop at `:555` and the nested `_reference`)
  rebind it before reading it, so the rename shadows nothing that is read.
- `Datastore/replication.py`, `same_value`'s docstring: "no tolerance belongs here" became "an
  approximate comparison has no place here".
- `Datastore/shard_paths.py`, the module docstring: "``ShardedPool`` and ``tools/shard_key_audit.py``
  both call it" became "``ShardedPool`` and every tool that reads a store's shard records call
  it", reflowed. It names no client file.

## The tests, test by test

### Deleted (README §5 rule 7)

- `Datastore/tests/test_declared_facts.py`, `TestNoProjectNameInThePool`:
  - `test_no_string_or_name_is_a_registry_table`;
  - `test_no_import_names_a_project_package`;
  - `test_no_comment_or_docstring_names_one`.
- `Datastore/tests/test_inventory_declarations.py`, `TestTheLayerKnowsNoProject`:
  - `test_it_imports_no_project_package`;
  - `test_no_project_name_in_its_code_or_text`.

With them, only what they used:
- in `test_declared_facts.py`: `_WORD`, `_LAYER_WORDS`, `_PROJECT_PACKAGES`, `_registry_names`,
  `_FLAG_PHRASES`, `_named` and the comments above them; the module names `REPO_ROOT` and
  `POOL_FILE`; the imports `ast`, `re`, `sys` and `tokenize`. Each name was grepped first: after
  the deletion each had no other use in the module. The module docstring's item 9 is one sentence
  saying that the guard is `test_layer_is_generic`.
- in `test_inventory_declarations.py`: `STORE_INVENTORY` and the imports `ast`, `io` and
  `tokenize`. `re` stays: `TestResolve` uses it. The docstring's item 4 and the section-4 header
  say what remains: the two fresh-interpreter tests, `test_reading_records_loads_no_project_module`
  and `test_read_inventory_loads_only_what_the_registry_loads`, which stay with `PROJECT_PACKAGES`
  and `_run`.

`git grep -nE 'TestNoProjectNameInThePool|_registry_names|test_no_project_name_in_its_code_or_text|test_it_imports_no_project_package' -- ':!prompts'`
prints nothing. (The guard's docstring describes what it replaces without using these names.)
`git grep -n 'parent_serial' -- Datastore/store_inventory.py` prints nothing.

### Added

The eight tests of `test_layer_is_generic.py`, listed above. No other test is edited.

## Deviations from the prompt

- **IMPLEMENTATION CHOICE: `LAYER_WORDS` carries a comment per word,** not one comment for the set
  (the prompt: "a comment gives each reason").
- **IMPLEMENTATION CHOICE: `test_the_allow_list` is in `TestTheLayerImportsNoProjectPackage`,**
  beside `test_every_import_is_allowed`.
- **IMPLEMENTATION CHOICE: the module docstring names the tests it replaces by description** (the
  check of `ShardedPool.py` in `test_declared_facts`, the two of `store_inventory.py` in
  `test_inventory_declarations`), not by their method names, so that the first grep of §3.4 prints
  nothing at all.
- **IMPLEMENTATION CHOICE: the breakage diffs were generated against a scratch git index** (the
  tree's six changed files added to a copy of the index, in the scratchpad), so that the diff of a
  mutation to `test_layer_is_generic.py`, which is not yet committed, shows the mutation alone. The
  real index was not touched.
- **IMPLEMENTATION CHOICE: §3.3 ran from the checkout** as `e980b6e`'s tree, which holds
  `e980b6e`'s code; the records edited afterwards are not code. `dbd8a51` is an export in the
  scratchpad (`git archive`, with `venv` symlinked).
- **IMPLEMENTATION CHOICE: mutation (ix) ran the guard module only.** The prompt requires that it
  fail `test_the_allow_list` and no other test; within the guard module, 7 of 8 pass. Nothing in
  the layer imports another `config` module (the guard passes on the real layer), so no suite can
  depend on `import_allowed`'s answer for one.
- `python -m RunRegistry list` was run, as CLAUDE.md asks: 16 runs, 14 finished and 2 unknown, none
  `running`. It reads `var/runs/`, the registry's own listing; no store under `var/` was opened,
  written or copied.
- No `STRUCTURALLY REQUIRED` and no `UNINTENDED DRIFT` deviation.
- *Added at the orchestrator's review (2026-10-06).* **IMPLEMENTATION CHOICE, not recorded at
  landing: mutation (vii) names `BackgroundModel` in a module constant** (`KIND`), where the
  prompt's §3.6 says the new file's docstring. Both are string constants, which
  `test_no_string_or_docstring` scans alike, so the mutation shows what the prompt asked: a module
  nobody listed is checked.

## Verification performed

**Every suite**, serially in the foreground, verdicts read from files (`README` §7's command):

| Suite | Before (orchestrator, `c96e56d`) | After |
|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK |
| CosmologyModels | 39 OK | 39 OK |
| LiouvilleGreen | 148 OK (1 skipped) | 148 OK (1 skipped) |
| RunRegistry | 280 OK | 280 OK |
| ComputeTargets | 693 OK | 693 OK |
| Datastore `[a-f]` | 150 | 147 OK |
| Datastore `[g-o]` | 177 | 183 OK |
| Datastore `[p-r]` | 200 | 200 OK |
| Datastore `[s-z]` | 191 | 191 OK |
| Datastore total | 718 | **721** = 147 + 183 + 200 + 191 |

The Datastore difference, −5 deleted and +8 added:
- `[a-f]`: `test_declared_facts` loses its three, 150 − 3 = 147;
- `[g-o]`: `test_inventory_declarations` loses its two and `test_layer_is_generic` adds its eight,
  177 − 2 + 8 = 183.

No flake appeared. `black --check` is clean on the six
changed Python files.

**The guard before the three fixes** (`test_layer_is_generic`, 8 tests, `failures=2`): exactly the
ten hits of the prompt's §1.
```
FAIL: test_no_code_name
Datastore/store_inventory.py:607 parent_serial
Datastore/store_inventory.py:610 parent_serial
Datastore/store_inventory.py:611 parent_serial
Datastore/store_inventory.py:615 parent_serial
Datastore/store_inventory.py:643 parent_serial
Datastore/store_inventory.py:646 parent_serial
Datastore/store_inventory.py:647 parent_serial
Datastore/store_inventory.py:649 parent_serial
FAIL: test_no_string_or_docstring
Datastore/replication.py:289 tolerance
Datastore/shard_paths.py:12 tools
Ran 8 tests in 0.311s
FAILED (failures=2)
```
After the fixes: `Ran 8 tests in 0.429s`, `OK`. (On the `dbd8a51` layer the orchestrator measured
64 code, 209 string, 23 comment and 23 import hits; that is the orchestrator's measurement, not
repeated here.)

**Rule 9 at fixture level (§3.3).** `dbd8a51` is an export in the scratchpad; the scripts are the
orchestrator's `m4_*.py` and `checksums.py`, run with the tree as cwd. Nothing of it is committed.
- **(a) Schema and rows.** `build_full_store` and `build_real_store`, built at each tree with its
  own code and fixtures: 6 files, 720 schema entries (`sqlite_master` type, name, table, SQL), 164
  tables and 286 rows; `files 6 6 same names: True`, `identical tables 164 identical rows 286
  differences 0`. A second build at this tree compared the same: 0 differences, so the builds are
  deterministic.
- **(b) A store written at `dbd8a51`, read at both trees.** Fingerprints with `taken` removed,
  `cmp`: identical.
  - `build_full_store`: `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759` (7507
    bytes), 21 classes, 55 records, 0 problems;
  - `build_real_store`: `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f` (4484
    bytes), 21 classes, 18 records, 0 problems.

  The whole inventories, normalised for the polymorphic field, are `cmp`-identical (29954 and
  12301 bytes: the paths here are four characters longer than the orchestrator's). The store's file
  checksums before and after the two reads are identical. At `dbd8a51` `read_inventory` took no
  registry; at `e980b6e` it takes `config.datastore.factories`.
- **(c) A read-write open** of a copy of the `dbd8a51` full store through each tree's stand-in
  pool, with the random seed fixed, under the store's own label (`2025.1.1`) and under a new one
  (`standin`): neither tree raised. The reconciliation (14 tables compared, nothing repaired, both
  shards), the prune (nothing), the version object and the 15 printed lines, normalised for paths
  and times, are `SAME` in both. Under the store's own label no file changed
  (`changed_files []`, in both trees). Under the new label both trees wrote one version row and
  changed the primary and both shards, and the resulting file checksums are identical. `kwargs
  DIFFERS` is expected: 06 added the `factories` argument.

**The rebuild is not this prompt's.** README §3's exception lets the orchestrator's review run
one rehearsal build into a new store under `var/datastores` and compare its fingerprint with
`dbd8a51`'s (`39809dca…`). This prompt ran no build and started no Ray, and opened nothing under
`var/` but the registry's listing. The reason is the one the user was given on 2026-10-06 (README
§6.2): the fixtures write their rows with SQL, so they show that the schema and the read path are
unchanged, and only a build exercises the write path that 01–03 and 07 touched.

## The deliberate-breakage record

Each mutation is a diff exactly as applied, generated from a clean tree at `e980b6e`; `git apply
--check` and `git apply -R --check` passed for each, in `bash`, and `git status` was clean (the
six changed files only, before the commit) after each `git apply -R`. The module under test is
`Datastore.tests.test_layer_is_generic`. The mutations are never committed.

**(i) a registry table (`GkSourceValue`) in a comment of `ClientPool.py`.** Fails: `test_no_comment` (`Datastore/SQL/ClientPool.py:24 GkSourceValue`).

```diff
diff --git a/Datastore/SQL/ClientPool.py b/Datastore/SQL/ClientPool.py
index 6c9e814..a81d131 100644
--- a/Datastore/SQL/ClientPool.py
+++ b/Datastore/SQL/ClientPool.py
@@ -21,6 +21,7 @@ class ClientPool:
         self._batch_size = default_batch_size
 
         self._pool = set()
+        # serials leased for GkSourceValue rows
         self._leased = set()
         self._committed = set()
 
```

**(ii) an identifier column (`model_serial`) as a local name in `schema.drop_order`.** Fails: `test_no_code_name` (`Datastore/SQL/schema.py:290 model_serial`).

```diff
diff --git a/Datastore/SQL/schema.py b/Datastore/SQL/schema.py
index 18b0dd6..d373484 100644
--- a/Datastore/SQL/schema.py
+++ b/Datastore/SQL/schema.py
@@ -287,6 +287,7 @@ def drop_order(tables: Iterable[sqla.Table]) -> List[sqla.Table]:
     tables has no such order, and raises ``ValueError`` naming the tables on or behind it.
     """
     given: List[sqla.Table] = []
+    model_serial = None
     for table in tables:
         if table not in given:
             given.append(table)
```

**(iii) `from CosmologyModels import model_ids` inside `replication.same_value`.** Fails: `test_every_import_is_allowed` (`replication.py:291 CosmologyModels`) and `test_no_code_name` (`replication.py:291 model_ids`).

```diff
diff --git a/Datastore/replication.py b/Datastore/replication.py
index 12fc83c..9b3dba7 100644
--- a/Datastore/replication.py
+++ b/Datastore/replication.py
@@ -288,6 +288,8 @@ def same_value(a, b) -> bool:
     are the same value. Copies of one row are written from one pickled object, so every float is
     bit-identical and an approximate comparison has no place here.
     """
+    from CosmologyModels import model_ids
+
     if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
         return True
     return a == b
```

**(iv) `from config.sharding import replicated_tables` in `store_reader.py`.** Fails: `test_every_import_is_allowed` (`store_reader.py:58 config.sharding`).

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index dd7e4a1..7ffaea9 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -55,6 +55,7 @@ from Datastore.SQL.schema import (
     schema_differences,
 )
 from Datastore.shard_paths import shard_file_problem
+from config.sharding import replicated_tables
 
 PathType = Union[str, os.PathLike]
 
```

**(v) "source and response" in `shard_paths.py`'s docstring.** Fails: `test_no_string_or_docstring` (`shard_paths.py:3 'source and response'`).

```diff
diff --git a/Datastore/shard_paths.py b/Datastore/shard_paths.py
index c5354a8..85912a0 100644
--- a/Datastore/shard_paths.py
+++ b/Datastore/shard_paths.py
@@ -1,5 +1,6 @@
 """
 Where a ShardedPool's shard files live, given what its primary's ``shards`` table says.
+Nothing here reads the source and response columns.
 
 A ShardedPool datastore is one primary SQLite file plus *N* shard files beside it
 (``foo.sqlite``, ``foo-shard0000.sqlite``, ...). The primary records each shard in its ``shards``
```

**(vi) a `*_tags` fragment in a `store_reader.py` comment.** Fails: `test_no_comment` (`store_reader.py:114 '_tags'`).

```diff
diff --git a/Datastore/store_reader.py b/Datastore/store_reader.py
index dd7e4a1..2c41329 100644
--- a/Datastore/store_reader.py
+++ b/Datastore/store_reader.py
@@ -111,6 +111,7 @@ def _refuse_journals(primary: Path, path: Path, what: str) -> None:
 
 def read_only_url(path: Path) -> str:
     """The SQLAlchemy URL of ``path`` opened read-only: ``mode=ro``, never ``immutable=1``."""
+    # the *_tags tables open read-only like any other
     return f"sqlite:///file:{path}?mode=ro&uri=true"
 
 
```

**(vii) a new file, `Datastore/scratch_module.py`, whose module constant `KIND` names `BackgroundModel`; its diff creates the file and `-R` deletes it.** Fails: `test_no_string_or_docstring` (`Datastore/scratch_module.py:3 BackgroundModel`): the derived set checks a module nobody listed.

```diff
diff --git a/Datastore/scratch_module.py b/Datastore/scratch_module.py
new file mode 100644
index 0000000..6569f95
--- /dev/null
+++ b/Datastore/scratch_module.py
@@ -0,0 +1,3 @@
+"""A module added to the layer."""
+
+KIND = "BackgroundModel"
```

**(viii) `layer_files()` narrowed to `Datastore/SQL/`.** Fails: `test_the_derived_files_hold_the_audits_list`.

```diff
diff --git a/Datastore/tests/test_layer_is_generic.py b/Datastore/tests/test_layer_is_generic.py
index ea796cf..53a6119 100644
--- a/Datastore/tests/test_layer_is_generic.py
+++ b/Datastore/tests/test_layer_is_generic.py
@@ -60,15 +60,13 @@ _FACTORY_CONTRACT = "base.py"
 def layer_files() -> list:
     """The layer's modules, relative to the repository root, sorted."""
     found = set()
-    for path in (REPO_ROOT / "Datastore").rglob("*.py"):
+    for path in (REPO_ROOT / "Datastore" / "SQL").rglob("*.py"):
         rel = path.relative_to(REPO_ROOT)
         if rel.parts[:2] == ("Datastore", "tests"):
             continue
         if rel.parts[:3] == _FACTORIES_DIR and rel.name != _FACTORY_CONTRACT:
             continue
         found.add(rel.as_posix())
-    for path in (REPO_ROOT / "RayTools").glob("*.py"):
-        found.add(path.relative_to(REPO_ROOT).as_posix())
     return sorted(found)
 
 
```

**(ix) `import_allowed` accepts every `config.*`.** Fails: `test_the_allow_list`, subtests `refused='config.sharding'` and `refused='config.datastore'`, and no other test of the module (7 of 8 pass; nothing in the layer imports another `config` module).

```diff
diff --git a/Datastore/tests/test_layer_is_generic.py b/Datastore/tests/test_layer_is_generic.py
index ea796cf..e6bfe21 100644
--- a/Datastore/tests/test_layer_is_generic.py
+++ b/Datastore/tests/test_layer_is_generic.py
@@ -189,7 +189,7 @@ def import_allowed(module: str) -> bool:
     root = module.split(".")[0]
     if root in _ALLOWED_ROOTS or module in _ALLOWED_MODULES:
         return True
-    if root == "RayTools":
+    if root in ("RayTools", "config"):
         return True
     if root == "Datastore":
         # the layer's own modules, and of the client's factories only the contract
```

## Observations not acted on

- `tools/inventory_report.py` is a client tool and names project tables (`INVENTORY_CATEGORIES`,
  `VALUE_TABLES`). It is outside the guard, as the prompt says.
- The guard does not check what a module *loads* when run; the two fresh-interpreter tests of
  `test_inventory_declarations` do.
- The guard's English-word policy has no allow-list entry: an English word that collides with a
  registry name (`tolerance`, `parent_serial`) is reworded in the layer. Plain-word columns
  (`source`, `value`, `type`) are not in the vocabulary, since they cannot be told from English;
  only the two declared flags' phrases are checked.
- `read_records`' `for field, parent in declared` loop leaves `parent` bound in the function's scope,
  and the tag and value reads now rebind it. Nothing reads it after the loop, so nothing changes;
  noted because the rename was the prompt's choice of name.

## State handed to the next prompt

There is no next prompt: 09 is the campaign's last.
- **For the review:** README §3's one rehearsal rebuild, compared with `dbd8a51`'s `39809dca…`. The
  campaign closes when it matches (K2's second half). The user then retires the new store.
- Every open issue of this board that is not closed here stays where it is
  (`[04-closed-store-refusals-repeat-their-prefix]`,
  `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`,
  `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`); none is assigned.
- A module added to the layer is checked by `test_layer_is_generic` without being listed. A new
  import outside the allow-list needs the user's decision (README §5 rule 11), not an entry.
