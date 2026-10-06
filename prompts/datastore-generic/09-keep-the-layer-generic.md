# Prompt 09 — keep the layer generic

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**–**K2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** 08 landed (`6646b06`) and reviewed (`3ff1c80`). `git status` clean. Never in one
checkout while another prompt's change to the layer or its tests is uncommitted.
**Closes:** `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]`.
**Narrows:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. The change is small: one new test module, two deleted
tests, three text and local-name edits in the layer, and a fixture-level verification. Its test
plan, every hit the guard finds today and every comparison were measured with a rough version of
the change.

**Read first:**

1. [`README.md`](README.md):
   - §0.1, §1, §3 (and its exception for 09), §4 and §5 (rules 4, 7, 9, 10 and 11);
   - §6.2: the user's four choices of method for 09, taken on 2026-10-06.
2. [`docs/datastore-generic-audit.md`](../../docs/datastore-generic-audit.md): **C** (the
   generic layer's list, C1's comment list, "what is infrastructure") and **C2**.
3. The board's orchestrator review of 08 (§1), and its "Handed on" list.
4. The two name tests this prompt folds into the guard:
   - `Datastore/tests/test_declared_facts.py`: the module docstring's item 9 (`:24`), `REPO_ROOT`
     and `POOL_FILE` (`:60-61`), the helpers `_WORD` (`:679`), `_LAYER_WORDS` (`:684`),
     `_PROJECT_PACKAGES` (`:698`), `_registry_names` (`:705`), `_FLAG_PHRASES` (`:724`) and
     `_named` (`:729`), and `TestNoProjectNameInThePool` (`:733-806`);
   - `Datastore/tests/test_inventory_declarations.py`: the module docstring's item 4 (`:13`),
     `STORE_INVENTORY` (`:48`), section 4 (`:409-`): `PROJECT_PACKAGES`, `_run` and
     `TestTheLayerKnowsNoProject`'s four methods.
5. The three sites the guard flags today:
   - `Datastore/store_inventory.py:607-649`, the local `parent_serial`;
   - `Datastore/replication.py:289`, `same_value`'s docstring;
   - `Datastore/shard_paths.py:11-13`, the module docstring.
6. The imports the user allowed: `config.defaults` (`ShardedPool.py:35`, `ProfileAgent.py:10`)
   and `utilities` (`Datastore.py:20`, `ProfileAgent.py:11`, `RayTools/RayWorkPool.py:9`).

Line numbers are at `3ff1c80`.

---

## 1. What is wanted

After 08 every row of audit C1 is gone. The layer names none of the project's tables, and
imports nothing of the project but two generic modules. Nothing yet stops that recurring: the
re-coupling the audit found came from prompts that described behaviour table by table (README §0).
Two tests check single files, `ShardedPool.py` (07) and `store_inventory.py` (08). No test covers
the rest of the layer, or a module added to it.

- **K1 — a guard.** One test module checks every file of the generic layer, derived rather than
  listed. It checks the code, the strings, docstrings and comments, and the imports. The two
  per-file tests are folded into it and deleted. What the layer may still import is a stated
  allow-list, each entry with its reason.
- **K2 — the closing verification.** Every suite, and README §5 rule 9's closing check: a store
  written at `dbd8a51`, the campaign's first tree, opens, reads and fingerprints the same at this
  commit, and nothing the code writes has changed. This prompt does the fixture half. The
  orchestrator's review does the other half, one rehearsal rebuild (README §3's exception, §6.2).

**The user's choices, 2026-10-06 (README §6.2):**
- `config.defaults` is **allow-listed**. `ShardedPool` and `ProfileAgent` keep importing
  `DEFAULT_STRING_LENGTH` from it.
- `utilities` is **allow-listed**. `Datastore.py`, `ProfileAgent` and `RayWorkPool` keep importing
  `WallclockTimer` and `format_time` from it.
- The two per-file name tests are **folded into the guard and deleted**: one definition of what
  the layer may not name.
- Rule 9's closing check is **the fixture checks here, plus one rebuild at the review**.

**Measured on 2026-10-06 with a rough version of this change** (orchestrator, at `3ff1c80`; the
diff, the scripts and the outputs are in `orchestrator/measure-09/`):
- **The layer's files.** The derivation gives 17. These are the audit's 12, `Datastore/contract.py`
  (07), and four more: `ObjectFactories/base.py` and the three `__init__.py` files. All four are
  clean.
- **The vocabulary**: 234 words. That is 36 registry tables, 184 identifier columns, 11 project
  packages and 08's five cosmology names, plus 07's flag phrases and 08's fragments.
- **The guard on today's layer: 10 hits, at three sites, and none is project knowledge.**

  | Site | Kind | What it is |
  |---|---|---|
  | `store_inventory.py:607, 610, 611, 615, 643, 646, 647, 649` | code | the local `parent_serial`, which shares its name with a column of `GkSource_parents` |
  | `replication.py:289` | text | "no tolerance belongs here": `tolerance` is a registry table |
  | `shard_paths.py:12` | text | `tools/shard_key_audit.py`: `tools` is a project package |

  The audit's other comment sites are gone. `ClientPool.py:37-38` moved to `config/` with 06,
  `shard_paths.py:14` went with 04, and `replication.py:200, 204` now say "version row" and
  "version label", the layer's own words. On the `dbd8a51` layer the same guard finds 64 code,
  209 string, 23 comment and 23 import hits.
- **The suites with the rough change:**
  - Datastore 720 = 147 + 182 + 200 + 191: 718 − 5 deleted + 7 new;
  - RunRegistry 280, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
    (1 skipped), all unchanged.

  The guard module runs in 0.3 s.
- **Eight mutations**, (i)–(viii) of §3 below. Each failed the guard.
- **Rule 9 at fixture level, `dbd8a51` against the rough tree:**
  - **(a) Schema and rows.** `build_full_store` and `build_real_store`, built at each tree with its
    own code and fixtures, have identical `sqlite_master` (type, name, table, SQL) and identical
    rows. That is 6 files, 720 schema entries, 164 tables and 286 rows, with 0 differences. A
    second build at one tree also gives 0, so the builds are deterministic.
  - **(b) A store written at `dbd8a51`, read at both trees.** The fingerprints, with `taken`
    removed, are byte-identical:
    - `build_full_store`: `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`;
    - `build_real_store`: `d932853819d07ce40fbb9cce544dabe4bbe968e4a7c908bb33e80eecbca3264f`.

    So are the whole inventories, normalised for the polymorphic field. The checksums did not
    change.
  - **(c) A read-write open** of a copy of the `dbd8a51` full store, through each tree's stand-in
    pool:
    - neither tree raised;
    - the reconciliation (14 tables, nothing repaired), the prune (nothing), the version object
      and the 15 printed lines (normalised for paths and times) are identical;
    - under the store's own label no file changed. Under a new label both trees wrote one version
      row, and the resulting files are byte-identical.

    The pool's constructor arguments differ, as 06 made them.

**What it does not change.**
- No behaviour of the layer. The edits are one local renamed and two docstring sentences rewritten.
- Nothing a store holds (README §5 rule 9).
- No test but the five deleted ones, and the two module docstrings and imports they leave behind.

---

## 2. What to change

**K1 — `Datastore/tests/test_layer_is_generic.py` (new).** The rough version is in
`orchestrator/measure-09/out/test_layer_is_generic.py.txt`. Build on it; this section governs.

- **The layer's files are derived**, by a function `layer_files()`:
  - every `.py` under `Datastore/`, except `Datastore/tests/` and `Datastore/SQL/ObjectFactories/`;
  - but `ObjectFactories/base.py` is included: it is the factory contract the actor imports;
  - plus `RayTools/*.py`.

  It globs the filesystem, not `git ls-files`, so that a new module is checked before it is
  committed.
- **The forbidden vocabulary** is the union of the two deleted tests':
  - the registry's tables (`build_schema(sqla.MetaData(), config.datastore.factories).tables`) and
    registered class names;
  - its identifier columns (an underscore or a capital);
  - both less `LAYER_WORDS`, which is 07's `_LAYER_WORDS` unchanged: the `Datastore.contract`
    names, `serial`, `version`, `timestamp`, `stepping` and `validated`. A comment gives each
    reason;
  - the project's packages: the top-level directories with an `__init__.py`, less `Datastore` and
    `RayTools` (both the layer) and `config` (the import check handles it, and the word is
    English);
  - 08's names `cosmology`, `cosmology_type`, `cosmology_serial`, `COSMOLOGY` and `model_ids`.

  In strings and comments the guard also checks 07's flag phrases ("source and response",
  "response flag", …) and 08's fragments `_tags`, `*Value` and `Run_<`.
- **The checks**, one test method each. Every failure lists `file:line word`, for all the files at
  once.
  - **`test_no_code_name`**: by `ast`, every name, attribute, definition, argument, keyword and
    imported name (with its alias), whole and split at underscores, as 07's test did.
  - **`test_no_string_or_docstring`**: every string constant, docstrings and f-string text
    included.
  - **`test_no_comment`**: every comment, by `tokenize`.
  - **`test_every_import_is_allowed`**: relative imports resolved. The allow-list is:
    - the standard library, `ray` and `sqlalchemy`;
    - `Datastore.*`, but of `Datastore.SQL.ObjectFactories` only `.base`;
    - `RayTools.*`;
    - `config.defaults` and `utilities`, each with a comment naming the user's choice of
      2026-10-06 and why.
- **The guard's own pins.** A guard whose input came out empty would pass every scan.
  - `test_the_derived_files_hold_the_audits_list`: the derived set contains audit §C's list and
    `Datastore/contract.py`, as a literal.
  - `test_no_test_and_no_client_factory_is_in_it`: `base.py` is in, and no other factory and no
    test file is.
  - `test_the_vocabulary_holds_the_registry_and_the_packages`: a few known tables, columns and
    packages are in it, and no `LAYER_WORDS` is.
  - **`test_the_allow_list`** (new; the rough version had none). `import_allowed` accepts
    `config.defaults`, `utilities`, `Datastore.SQL.ObjectFactories.base`, `RayTools.RayWorkPool`
    and `sqlalchemy.exc`. It refuses `config.sharding`, `config.datastore`,
    `Datastore.SQL.ObjectFactories.GkSource`, `CosmologyModels.model_ids` and `MetadataConcepts`.
    This pins the allow-list without a mutation.
- The module docstring states what the layer is, how its files are derived, what the guard
  forbids and allows and why, and that it replaces 07's and 08's per-file tests. It says nothing
  under `var/` is read, and that no Ray starts.

So the module has **8 tests**.

**K1 — the two tests folded in, deleted** (README §5 rule 7: each listed in the log by module,
class and method):
- `Datastore/tests/test_declared_facts.py`:
  - `TestNoProjectNameInThePool`, all three methods: `test_no_string_or_name_is_a_registry_table`,
    `test_no_import_names_a_project_package` and `test_no_comment_or_docstring_names_one`;
  - the helpers only they use: `_WORD`, `_LAYER_WORDS`, `_PROJECT_PACKAGES`, `_registry_names`,
    `_FLAG_PHRASES`, `_named`, and the comments above them;
  - the imports and module names then unused. Grep each one first; the measurement found `ast`,
    `re`, `sys`, `tokenize`, `REPO_ROOT` and `POOL_FILE`;
  - the docstring's item 9, rewritten as one sentence saying that the guard is
    `test_layer_is_generic`.
- `Datastore/tests/test_inventory_declarations.py`:
  - `TestTheLayerKnowsNoProject.test_it_imports_no_project_package` and
    `test_no_project_name_in_its_code_or_text`;
  - the imports and module names then unused (measured: `ast`, `io`, `tokenize`,
    `STORE_INVENTORY`);
  - the docstring's item 4 and the section-4 header, edited to say what remains: the two
    fresh-interpreter load tests, `test_reading_records_loads_no_project_module` and
    `test_read_inventory_loads_only_what_the_registry_loads`. These stay, with `PROJECT_PACKAGES`
    and `_run`, which they use. They check what a module **loads** when run, which the guard does
    not.
- A `git grep` for each deleted name, outside `prompts/`, prints nothing (measured).

**K1 — the three sites, fixed in the layer** (the rough version's fixes are in
`orchestrator/measure-09/out/phaseC_layer_fixes.diff`):
- **`store_inventory.py`:** the local `parent_serial` becomes `parent`, on its eight lines and
  nowhere else. Nothing else in the function changes.
- **`replication.py:289`:** "no tolerance belongs here" is rewritten so that the sentence keeps
  its meaning. For example: "…so every float is bit-identical, and an approximate comparison has
  no place here."
- **`shard_paths.py:11-13`:** the docstring names no client tool. The rough version wrote
  "the shard-key audit script (``shard_key_audit.py``)", which the vocabulary does not catch but
  which still names a client file. Write instead that `ShardedPool` and every tool that reads a
  store's shard records call it, then reflow the paragraph.
- **No allow-list entry for an English collision.** If the guard flags anything else, stop (§5).

**K2 — every suite**, and **rule 9 at fixture level**, as §3.3. The scripts are in
`orchestrator/measure-09/scripts/` (`m4_build.py`, `m4_sqlite_dump.py`, `m4_compare_dumps.py`,
`m4_inventory.py`, `m4c_open_rw.py`, `m4c_compare.py`, `checksums.py`); the agent may use its own.
Nothing of K2 is committed but the log's record of it.

**The rebuild is not this prompt's.** README §3's exception lets the orchestrator's review run
one rehearsal build into a new store under `var/datastores`, and compare its fingerprint with
`dbd8a51`'s (`39809dca…`). The agent starts no Ray, runs no build, and opens nothing under `var/`.

**Nothing else changes.** Not `ShardedPool`, the actor, `schema.py`, `store_reader`,
`ClientPool`, `ProfileAgent`, `RayWorkPool`, `config/`, `utilities.py`, any factory, `main.py` or
any `docs/` script.

---

## 3. Verification

1. **Every suite**, with counts before and after, each difference accounted for by test id:
   - Datastore: 718 − 5 + 8 = **721** (`[a-f]` 147, `[g-o]` 183, `[p-r]` 200, `[s-z]` 191);
   - RunRegistry 280; ComputeTargets 693;
   - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
2. **The guard, before and after.**
   - Before the three fixes, it reports exactly the 10 hits of §1, in `test_no_code_name` and
     `test_no_string_or_docstring`. Show the output.
   - After them it passes.
   - `layer_files()` gives the 17 files of §1; list them.
3. **Rule 9 at fixture level** (§1's (a)–(c)), at an export of `dbd8a51` and at this commit,
   from scratch scripts that are not committed:
   - (a) `sqlite_master` and every table's rows, for both fixture stores built at each tree,
     identical. Give the counts;
   - (b) a store built with `dbd8a51`'s code, read at both trees: the fingerprint (with `taken`
     removed) and the whole inventory byte-identical. Give both digests. The checksums are
     unchanged;
   - (c) a read-write open of a copy of that full store through each tree's stand-in pool:
     the reconciliation, the prune, the version object and the printed lines (normalised)
     identical, nothing raised, and under the store's own label no file changed.

   At `dbd8a51`, `read_inventory` takes no registry; at this commit it takes
   `config.datastore.factories`.
4. **Greps.**
   - `git grep -nE 'TestNoProjectNameInThePool|_registry_names|test_no_project_name_in_its_code_or_text|test_it_imports_no_project_package' -- ':!prompts'`
     prints nothing, but the guard's own docstring if it names what it replaces. Show the
     output.
   - `git grep -n 'parent_serial' -- Datastore/store_inventory.py` prints nothing.
5. `black --check` is clean on every touched Python file.
6. **Deliberate breakage.** Each mutation must make a named test fail; then restore it. The
   measurement's diffs are in `orchestrator/measure-09/out/m3/`; re-make them against this tree.
   - (i) a registry table (`GkSourceValue`) added to a comment in `ClientPool.py`:
     `test_no_comment`;
   - (ii) an identifier column (`model_serial`) as a local name in `schema.drop_order`:
     `test_no_code_name`;
   - (iii) `from CosmologyModels import model_ids` inside a function in `replication.py`:
     `test_every_import_is_allowed`, and `test_no_code_name`;
   - (iv) `from config.sharding import replicated_tables` in `store_reader.py`:
     `test_every_import_is_allowed`;
   - (v) "source and response" in `shard_paths.py`'s docstring: `test_no_string_or_docstring`;
   - (vi) a `*_tags` fragment in a `store_reader.py` comment: `test_no_comment`;
   - (vii) a new file, `Datastore/scratch_module.py`, whose docstring names `BackgroundModel`:
     `test_no_string_or_docstring`. This shows that the derived set checks a module nobody
     listed. Remove the file afterwards;
   - (viii) `layer_files()` narrowed to `Datastore/SQL/`: `test_the_derived_files_hold_the_audits_list`;
   - (ix) `import_allowed` accepts every `config.*`: `test_the_allow_list`, and no other test
     (nothing in the layer imports another `config` module).

   Record each in the log as a diff, exactly as applied, with the failing tests. Check each
   with `git apply --check` and `-R --check`, in `bash`, and leave `git status` clean.

---

## 4. Acceptance

1. **K1.**
   - `Datastore/tests/test_layer_is_generic.py` exists, with the eight tests of §2.
   - The five tests of §2 are deleted, with their now-unused helpers and imports.
   - The three sites are fixed in the layer, and the guard passes.
   - No allow-list entry exists but the four groups of §2.
2. **K2.**
   - Every suite has the counts of §3.1.
   - §3.3 (a)–(c) hold, byte for byte.
   - The log says that the rebuild is the review's, and why.
3. **The records**, all in the same commit, with `black --check` clean:
   - `[00-the-generic-datastore-layer-hard-codes-project-schema-facts]` is **closed**. It moves
     from §3 to §4 with a dated "Closed" line: every C1 row is gone (06–08), and the guard
     (`test_layer_is_generic`) keeps it so. Its row leaves the index, and the index goes from
     108 to 107.
   - On this board: the §1 row, items K1 (✅) and K2. K2 is "✅ fixture half; the rebuild at the
     review".
   - README §4's "09 ships" names the guard module, `layer_files()`, `import_allowed` and the
     allow-list.

---

## 5. Stop conditions — stop and ask the user

- The guard, before the fixes, finds anything but §1's ten hits. A new hit may be genuine
  project knowledge. Do not allow-list it, and do not rewrite it without asking.
- A test outside §2's plan fails after the change. That is current behaviour moving; never edit
  it to pass.
- Any comparison of §3.3 differs between the two trees.
- A fix would change what any code does, beyond renaming the one local. README §5 rule 9 is
  part of this.
- The change would open, write or touch any store under `var/`, or start Ray.
- Passing the guard would need a project name added to the layer, or an allow-list entry beyond
  §2's (README §5 rule 11).

---

## 6. What this prompt does not do

- **Files it changes:**
  - new: `Datastore/tests/test_layer_is_generic.py`;
  - `Datastore/tests/test_declared_facts.py` and `Datastore/tests/test_inventory_declarations.py`,
    as §2 says;
  - `Datastore/store_inventory.py` (the local), `Datastore/replication.py` and
    `Datastore/shard_paths.py` (docstrings), as §2 says;
  - the log, this board, the README (§4) and `docs/OPEN_ISSUES.md`.

  Nothing else.
- It runs no build and opens nothing under `var/`; the rebuild is the review's.
- It does not move `DEFAULT_STRING_LENGTH` or `utilities`' helpers (the user's choice).
- It does not touch `tools/inventory_report.py`, which is a client tool and names project tables.
- It changes no behaviour of any code.

---

## 7. The log and the board

`logs/09-keep-the-layer-generic.md`, using README §5.1. In addition, it gives:
- every test deleted and added, by module, class and method;
- `layer_files()`'s 17 files, and the vocabulary's size by kind;
- the guard's output before the fixes, and each fix;
- the allow-list, entry by entry, with its reason;
- §3.3's comparisons, with their counts and digests;
- what 09 leaves for the review: the rebuild, and the campaign's closure.

`IMPLEMENTATION_STATE.md`:
- the §1 row for 09, and items K1–K2;
- the closed generic-layer issue, moved to §4;
- §5.

`docs/OPEN_ISSUES.md`: the row removed, the count (107) and the date.
