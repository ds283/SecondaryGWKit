# Log — prompt 04, adopt `datastorekit`

**Prompt:** [`../04-adopt-datastorekit.md`](../04-adopt-datastorekit.md) ·
**Board items:** A1–A5 · **Date:** 2026-10-11 · **Model:** Opus 5.5 ·
**Commit:** this commit · **Result:** landed, after one stop. A1–A5 as the prompt says. SGK's
layer is the installed package: 42 files deleted, 2 emptied, 141 lines in 88 files rewritten by
module path, 6 files hand-edited. DatastoreKit's script finds 0 layer imports, strings or
`import_module` calls outside `prompts/`. **R0's reference store fingerprints `39809dca…` through
the package**, unchanged. Datastore 320, the other five roots at their counts, all OK. Two issues
opened (106 → 108).

Dispatched at `67dd316` (code of `9ec603a`), tree clean, registry 21 runs (19 finished, 2
`unknown`), none `running`, no Ray process. Governed by the prompt and the orchestrator's note for
04 (its §0), which corrects five of the prompt's statements. Saves are in
`var/dk-and-rpk-migration-04/` (gitignored): `before/` and `after/` (verbose suite outputs, A5,
the count by module, the alone re-runs), `measure-67dd316.{json,md}` (the JSON the rewrite read),
`measure-scratch.{json,md}` (§3.1), `rewrite-listing.txt` and `rewrite-dry-run.txt`,
`moved-test-{before,after}.{src,ast}`, `test-id-comparison.txt`, `stale-prose.txt`, `s33/` and
`breakage/`. No scratch file was written under `Datastore/`, `RayTools/` or `RunRegistry/`.

**The stop.** The first ComputeTargets run after the swap failed one staying test for a reason other
than a missed import, so I stopped (prompt §5). The test was
`test_quadsource_integral.TestOldRegimeRegression.test_G_only_case_reproduces_the_pre_commit_total`.
Its helper `load_pre_commit_module()` execs `git show 39ed7fc:ComputeTargets/QuadSourceIntegral.py`,
whose `:15` is `from Datastore import DatastoreObject`. The rewrite cannot reach history, and
`Datastore/__init__.py` is now empty. The user chose option (a) on 2026-10-11, relayed by the
orchestrator: map that one statement in the helper. That is deviation 7, below.

## What shipped

- **A1.** `git rm` of the 15 layer files: `Datastore/{object,contract,replication,shard_paths,
  store_reader,store_inventory}.py`, `Datastore/SQL/{schema,ShardedPool,Datastore,ClientPool,
  SerialPoolBroker,ProfileAgent}.py`, `Datastore/SQL/ObjectFactories/base.py`,
  `tools/{sharded_store,shard_key_audit}.py`.
  - `Datastore/__init__.py` (was `from .object import DatastoreObject`) and
    `Datastore/SQL/__init__.py` (was `from .Datastore import Datastore`) are emptied to 0 bytes.
  - `tools/__init__.py` (0 bytes) and `tools/inventory_report.py` stay.
- **A2.** [`../scripts/rewrite_layer_imports.py`](../scripts/rewrite_layer_imports.py), new, run
  once on the JSON of DatastoreKit's `measure_client_imports.py sgk --commit 67dd316 --json`, after
  A1's and A3's deletions and before any hand edit of a staying file.
  - It rewrote 134 `from` statements (35 nested) and 7 `import_module` calls, at **141 lines in 88
    files**, by module path only. It rewrote 0 string literals: the 10 outside `prompts/` were all
    in deleted files.
  - Line-number safety: the script refuses a line that differs from the line at the JSON's commit
    (read with `git show`), and a line without exactly one whole-name match. It refuses a statement
    that is not a `from` import of the one module the JSON names. Any refusal writes nothing.
  - It checks every name a rewritten statement imports with `getattr` on the package module.
- **A3.**
  - `git rm` of the 26 carried test modules (419 tests) and `Datastore/tests/schema_description.py`.
  - The unported test moved into `Datastore/tests/test_inventory_report.py`, as
    `TestTheReportRendersBothCosmologyTypes.test_the_report_renders_both_cosmology_types`, with
    `import re`.
  - `test_inventory_retired.py` changed as §2.3.3 says.
  - `standin_pool.py:43-45` and `test_wkb_numeric_parent_key.py:88` changed by the script alone;
    the comment at `:86-87` was changed by hand.
- **A4.**
  - `CLAUDE.md:47-73` (27 lines) is replaced by "The datastore layer is `datastorekit`", 26 lines,
    covering §2.4.1's six points.
  - `RunRegistry/stores.py:5` names `datastorekit`'s `ShardedPool` and its
    `datastorekit.tools.sharded_store`.
- **A5.** The six `docs/` files are rewritten like any other and fail as before.
  `[04-six-docs-probes-import-factories-from-the-layer]` is opened.
- **The user's decision on the stop:** `ComputeTargets/tests/test_quadsource_integral.py`, in
  `load_pre_commit_module()` only:

  ```python
      # the package replaced SGK's layer in dk-and-rpk-migration 04, and Datastore/__init__.py is
      # now empty (U3), so the old file's one layer import is pointed at the package
      old_import = "from Datastore import DatastoreObject\n"
      if text.count(old_import) != 1:
          raise AssertionError(
              f"expected one {old_import.strip()!r} in the pre-commit file, "
              f"found {text.count(old_import)}"
          )
      text = text.replace(old_import, "from datastorekit import DatastoreObject\n")
  ```
  `PRE_COMMIT_SHA` is unchanged, and no other line of the file changed.
- **New:** `prompts/dk-and-rpk-migration/measure/out/04-datastorekit-measure_client_imports.md`
  (§3.1).
- **Records:** this log; the board (§1 row, A1–A5, two §3 entries, `### 5.4`); README §4's "04
  ships" line; `docs/OPEN_ISSUES.md` (two rows under §1.20, 106 → 108).
- **Hand-edited files (6):** `test_inventory_report.py`, `test_inventory_retired.py`,
  `test_wkb_numeric_parent_key.py` (comment), `test_quadsource_integral.py` (the decision),
  `CLAUDE.md`, `RunRegistry/stores.py`. The first three and `stores.py` also hold
  script-rewritten lines; `CLAUDE.md` and `test_quadsource_integral.py` do not.

**File counts in the commit:** 42 deleted; 2 emptied; 88 rewritten by the script, of which 4 were
also hand-edited; 2 more hand-edited (`CLAUDE.md`, `test_quadsource_integral.py`); 3 added (the
script, the measurement, this log); plus the board, README and `docs/OPEN_ISSUES.md`.

## Deviations from the prompt

The orchestrator's note for 04 (its §0, "Corrections to the prompt") governs where it differs from
the prompt. Each of its five corrections is listed first.

1. **The counts** (prompt §0 and §2.5; README §0.2 and §1). At `67dd316` there are 234 statements
   in 139 files, 18 of them in `docs/` (14 files), not "about 180" or "17 in 13". The extra one is
   `docs/dk-and-rpk-migration/rehearsal/store_soundness.py:30`. After the deletions the rewrite was
   134 statements, 7 calls and 0 strings in 88 files, exactly as the note measured.
   **IMPLEMENTATION CHOICE** (orchestrator's note, correction 1).
2. **A5 is six statements, not ten** (prompt §2.5 and §2.2's last bullet). Four of the ten
   `_factories` imports are in `prompts/datastore-generic/orchestrator/measure-05/scripts/`, which
   A2 leaves alone. `order_audit.py` does not fail at import: it was loaded by path and its
   `d3_tables()` called, before and after. **STRUCTURALLY REQUIRED** (orchestrator's note,
   correction 2): run with no arguments, `order_audit.py` computes for minutes before reaching the
   import.
3. **`shard_key_audit` has no `--help`** (prompt §3.3.1). `python -m
   datastorekit.tools.shard_key_audit --help` prints `!! No such file: …/--help` and exits 2. SGK's
   own `python -m tools.shard_key_audit --help`, run before the deletion, did the same. In its place
   the package's tool ran on R0's reference: `VERDICT: OK`, exit 0, inside §3.3.3's hashes.
   **STRUCTURALLY REQUIRED** (orchestrator's note, correction 3): §3.3.1 cannot pass as written for
   either copy.
4. **The index already had this campaign's section** (prompt §4.6). Both rows go under §1.20, which
   02 opened; no new section. **IMPLEMENTATION CHOICE** (orchestrator's note, correction 4).
5. **`store_soundness.py` changes** (README §4, "02 ships": "05 and 08 run them unchanged"). A2
   rewrote its `:30`, and nothing else in it. README §4's "04 ships" line says so, and §3.3.4 loads
   it by path. **STRUCTURALLY REQUIRED** (orchestrator's note, correction 5): it would fail
   otherwise.
6. **A second issue, `[04-comments-name-the-layers-old-module-paths]`.** It records the 23 comment
   and docstring lines the rewrite leaves stale, and changes none of them, as the orchestrator's
   note directs. Its next step is 09's acceptance 2. **IMPLEMENTATION CHOICE** (orchestrator's
   note, "Prose the rewrite leaves stale").
7. **`ComputeTargets/tests/test_quadsource_integral.py` changes beyond any import** (the stop,
   above). Its helper maps the one layer import of the `39ed7fc` file it execs, and checks that the
   statement occurs exactly once. No assertion and no other test changed. The prompt's file list
   (§6) did not name the file. **STRUCTURALLY REQUIRED**, by the user's decision of 2026-10-11 to
   take option (a), relayed by the orchestrator: the test could not pass otherwise, since
   `Datastore/__init__.py` is empty (U3). No issue was opened for it, as decided.
8. **The moved test went in before the deletions, appended to the end of
   `test_inventory_report.py`, and `import re` was added after the rewrite.** This "moves first"
   (the dispatch's order) while every line number the JSON gives for that file stays the file's
   (prompt §2.2 step 3). **IMPLEMENTATION CHOICE.**
9. **`RunRegistry/stores.py:5`'s substitution re-flowed lines 5–8 of its paragraph.** The new line
   would have run past 100 columns. No other word of the docstring changed.
   **IMPLEMENTATION CHOICE.** The `test_wkb_numeric_parent_key.py:86-87` comment was re-wrapped
   for the same reason.
10. **Breakage (i)'s scratch commit is made from a temporary index read from the final staged
    tree's scratch commit (`addc09f`), not from `HEAD`.** `HEAD` was `67dd316`, before the swap, so
    reading it would have measured the old layer. The checkout's index was never touched by a
    mutation. **IMPLEMENTATION CHOICE.**
11. **The before and after test-id lists record outcomes from the `FAIL:`/`ERROR:` headers** and the
    `... skipped` marks, because tests' own printing interleaves with `-v`'s lines. Every id is
    found anywhere on a line. The id counts equal each run's `Ran` total. **IMPLEMENTATION CHOICE.**
12. **A breakage (vi)** was added at the orchestrator's direction, for the change of deviation 7.
    **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`. `black` changed one thing, a blank line left by my edit of
`test_inventory_retired.py`. No rewritten line reflowed.

## Verification performed

### Order of work

1. Before-saves at `67dd316`: six roots `unittest -v` (Datastore in four groups), A5, both SGK
   tools' `--help`.
2. Moved test appended, `ast`-checked.
3. A1 and A3 `git rm`, both `__init__.py` emptied.
4. DatastoreKit's script at `67dd316` (JSON, Markdown).
5. Rewrite: a dry run, then for real.
6. Hand edits (`import re`, `test_inventory_retired`, the `:86-87` comment, `CLAUDE.md`,
   `stores.py:5`).
7. `black`.
8. Stage, scratch commit, §3.1.
9. Suites.
10. The stop.
11. The decision applied.
12. Re-staged, new scratch commit (`addc09f`, superseding `dde5be3`), §3.1 again.
13. §3.3, A5 after, breakages (i)–(vi), records, commit.

### The rewrite's counts, by group

From the JSON at `67dd316`. The `b510bc9` appendix's groups are the same, but `docs/` 17 → 18.

| Group | Statements at `67dd316` | In deleted files | Rewritten (nested) | After (§3.1) |
|---|---|---|---|---|
| `(top level)` | 16 | 0 | 16 (0) | 0 |
| `ComputeTargets/` | 11 | 0 | 11 (0) | 0 |
| `CosmologyConcepts/` | 2 | 0 | 2 (0) | 0 |
| `CosmologyModels/` | 1 | 0 | 1 (0) | 0 |
| `Datastore/SQL/ObjectFactories/` | 43 | 0 | 43 (23) | 0 |
| `Datastore/tests/` | 68 | 49 | 19 (1) | 0 |
| `MetadataConcepts/` | 5 | 0 | 5 (0) | 0 |
| `Quadrature/` | 1 | 0 | 1 (0) | 0 |
| `RayTools/` | 1 | 0 | 1 (0) | 0 |
| `RunRegistry/` | 14 | 0 | 14 (6) | 0 |
| `config/` | 1 | 0 | 1 (0) | 0 |
| `docs/` | 18 | 0 | 18 (4) | 0 |
| `tools/` | 2 | 0 | 2 (1) | 0 |
| `prompts/` | 51 | 0 | 0 (left) | 51 |
| **total** | **234** | **49** | **134 (35)** | **51** |

- `import_module` calls: 11 at `67dd316`. 7 rewritten (`standin_pool.py:43-45`,
  `test_wkb_numeric_parent_key.py:88`, `replicated_write_fault_probe.py:37-39`), 3 in deleted
  files, 1 in `prompts/` left. After: 1.
- String literals: 13 at `67dd316`. 10 in deleted files, 3 in `prompts/` left. After: 3, plus the
  rewrite script's own 17.

The script's summary, from `var/dk-and-rpk-migration-04/rewrite-listing.txt`, which holds the full
`file:line old -> new` listing:

```
JSON: var/dk-and-rpk-migration-04/measure-67dd316.json (commit 67dd316)
  import_module: 7
  import_module, file deleted (skipped): 3
  import_module, prompts/ (left): 1
  literal, file deleted (skipped): 10
  literal, prompts/ (left): 3
  statement: 99
  statement, file deleted (skipped): 49
  statement, nested: 35
  statement, prompts/ (left): 51
lines changed: 141, in 88 files
imported names checked: 162; missing: 6
```

- The six misses are the A5 `_factories` (`fk_sweep_probe.py:8`, `lookup_keys/common.py:18`,
  `probe_03_column_existence.py:25`, `tq_fk_probe.py:8`, `inventory.py:196`,
  `order_audit.py:1253`). The other 156 names resolve, `_merge_digests`
  (`test_quadsource_parent_key.py:47`) among them.
- **The rewrite is exact.** A check against `git show HEAD` was run on the 84 files the script alone
  touched. It found every changed line to be one of the JSON's sites, and equal to the old line with
  the map applied, with no other line changed and no site missed. In the four script-plus-hand
  files, each site's mapped line is present. Result: 141 sites, 0 problems.

### The moved test

```
Datastore/tests/test_inventory_declarations.py:445-471  ast-dump chars 3079
Datastore/tests/test_inventory_report.py:424-450  ast-dump chars 3079
41b38b185041f1fb60245c6ce9a202657b55891db85ed08376c19b5aa793ee70  moved-test-after.ast
41b38b185041f1fb60245c6ce9a202657b55891db85ed08376c19b5aa793ee70  moved-test-before.ast
```

The method's source (docstring included) is byte-identical too. It brings no layer or factory
import. `test_inventory_report.py` already imported `read_inventory`, `build_full_store`,
`format_inventory_report`, `_factories`, `tempfile` and `Path`, and gains `import re`.

### §3.1 — DatastoreKit's script at a commit of the staged tree

`c=$(git commit-tree $(git write-tree) -p HEAD -m "04 scratch")` gave
`addc09feac9026e50533550e51cfd488cafed72a`. An earlier one, `dde5be3`, was taken before the
decision and is superseded. The script was run from DatastoreKit's root with its venv, and its
Markdown is saved as
[`../measure/out/04-datastorekit-measure_client_imports.md`](../measure/out/04-datastorekit-measure_client_imports.md).

| What | Expected (note) | At `addc09f` |
|---|---|---|
| layer files absent | 15 | **15** (the 15 of A1) |
| `Datastore/__init__.py`, `Datastore/SQL/__init__.py` | present, 0 bytes | present; `git cat-file -s` gives `0` and `0` |
| statements outside `prompts/` | 0 | **0** |
| string literals outside `prompts/` | 0 | **0** |
| `import_module` calls outside `prompts/` | 0 | **0** |
| `prompts/` statements | 51 in 31 files, `b510bc9`'s | **51 in 31 files**, the list equal to `67dd316`'s entry for entry |
| `prompts/` strings | 3 + the script's own | **20** = `m5_imports.py:14-16` (unchanged) + `rewrite_layer_imports.py:50-65` (16) and `:70` (1) |
| `prompts/` `import_module` calls | 1 + the script's own | **1**, `m2_open_behaviour.py:30`; the script has none |
| factory imports | 99 (97) | **99 (97)**: `Datastore/tests/` 46 → 45 |

- **The factory-import figure, derived.** `b510bc9`'s appendix gives 100 (98 outside the factory
  package). The deleted modules hold one factory import:
  `test_inventory_declarations.py:474`, `from Datastore.SQL.ObjectFactories.cosmology_types
  import COSMOLOGY_TYPES`, found by `git grep` at `HEAD` over the 27 deleted test files.
  (`test_declared_facts.py:52` imports `…ObjectFactories.base`, a layer import, not a factory
  import.) So 100 − 1 = 99, and 98 − 1 = 97.
- Files read: 425 (451 at `67dd316`, less the 26 modules).
- **The landed commit differs from the measured tree by the records only:**

  `git diff --cached --stat addc09f` with the records staged (the same as `git diff --stat addc09f
  HEAD` at this commit) lists exactly the five records:
  ```
   docs/OPEN_ISSUES.md                                |   7 +-
   .../dk-and-rpk-migration/IMPLEMENTATION_STATE.md   | 110 +++-
   prompts/dk-and-rpk-migration/README.md             |   7 +
   .../logs/04-adopt-datastorekit.md                  | 629 +++++++++++++++++++++
   .../out/04-datastorekit-measure_client_imports.md  | 148 +++++
   5 files changed, 892 insertions(+), 9 deletions(-)
  ```
  (The log's line count is the one before this block was inserted.)

### §3.2 — the suites

README §7's command with `-v`, each in the foreground, serially, output to a file and the verdict
grepped from it. Datastore ran in its four groups. Before: `67dd316`, load 9–78. After: the final
staged tree, load 5–25.

| Root | Before (`67dd316`) | After | Expected | Why |
|---|---|---|---|---|
| `Datastore/tests` a–f | 163 OK | **54** OK | 54 | |
| `Datastore/tests` g–o | 185 OK | **103** OK | 103 | `test_inventory_report` 18 + 1, `test_inventory_retired` 7 − 1 |
| `Datastore/tests` p–r | 200 OK | **101** OK | 101 | |
| `Datastore/tests` s–z | 191 OK | **62** OK | 62 | |
| `Datastore/tests` | 739 | **320** | 320 | 739 − 419 + 1 − 1 |
| `RunRegistry/tests` | 286 OK | **286** OK | 286 | imports only |
| `ComputeTargets/tests` | 693 OK | **693** OK | 693 | imports, and deviation 7 |
| `AdaptiveLevin/tests` | 32 OK | **32** OK | 32 | untouched |
| `CosmologyModels/tests` | 39 OK | **39** OK | 39 | untouched |
| `LiouvilleGreen/tests` | 148 OK (1 skipped) | **148** OK (1 skipped) | 148 | untouched |

- **`count_datastore_tests.py` after:** "320 tests loaded in 18 modules". All 26 `DELETE` rows
  are absent (its "not found" lists all 26). `test_inventory_report` has 19 and
  `test_inventory_retired` 6; the other 16 staying modules are at `b510bc9`'s counts.
- **ComputeTargets, at the stop.** The first after-run (`after/ComputeTargets-at-stop.txt`) gave
  693 tests, FAILED (failures=1, errors=1):
  - the error was `TestOldRegimeRegression`'s `ImportError`, which is the stop;
  - the failure was README §7's wall-clock flake,
    `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, at 0.0602 s against 0.06 s, load
    about 25.

  Re-run alone at load 20–23 (`after/rerun-alone.txt`), the flake failed (0.0604 s), passed,
  failed (0.0609 s) and passed. After the decision, `TestOldRegimeRegression` passed alone
  (`after/oldregime-alone.txt`), and the root's re-run gave **693 OK**, the flake included.
- **The test ids** (`test-id-comparison.txt`):
  - 1937 ids before and 1518 after, equal to each run's `Ran` totals.
  - **Gone, 420:** the 419 tests of the 26 deleted modules (`test_inventory_declarations`' 26
    included), and
    `test_inventory_retired.TestNoReferenceRemains.test_the_excluded_file_holds_only_the_new_service_helper`
    (§2.3.3).
  - **New, 1:**
    `Datastore.tests.test_inventory_report.TestTheReportRendersBothCosmologyTypes.test_the_report_renders_both_cosmology_types`,
    ok.
  - **Same id, outcome differs: none.** Every other id that ran before ran after, under the same
    id, ok. The only non-ok is LiouvilleGreen's
    `test_3bessel_analytic.Test3BesselAnalytic.test_YJJ_log_scaling`, skipped before and after.
  - The other five of `test_inventory_retired` keep their ids and pass.
- **`black --check`** is clean on all 92 Python files this commit touches (the 88, the two
  `__init__.py`, `test_quadsource_integral.py`, the rewrite script).

### §3.3 — through the package, on real things

1. **Tools** (`s33/`).
   - `./venv/bin/python -m datastorekit.tools.sharded_store --help`: exit 0,
     `usage: python -m datastorekit.tools.sharded_store [-h] {copy,move} src dst`.
   - `./venv/bin/python -m datastorekit.tools.shard_key_audit --help`: exit 2,
     `!! No such file: /Users/ds283/Documents/Code/SecondaryGWKit/--help`.
   - SGK's own copies, before the deletion (`before/`): `python -m tools.shard_key_audit --help`
     gave the same line and exit 2; `tools.sharded_store --help` exited 0.
2. **`PYTHONPATH=. ./venv/bin/python -m RunRegistry list`**: exit 0, "21 run(s) … 19 finished, 2
   unknown". It shows R0's four `dk-and-rpk-migration-02-*` runs.
3. **R0's reference store, read through the package**, read-only, inside one before-and-after
   record of all six files.
   - Before (`s33/store-before.txt`): SHA-256, `st_mtime_ns`, size.
     ```
     9b7a0acc8bcca22171ada0a9ba08b892aa6eb470e24d8fbe35c587efbdc682a8 1791725417111208533 36864 var/datastores/dk-and-rpk-migration-02-reference.sqlite
     3e469052e8810e1efec8dee53c3a89ea4a2a343d3346cd2a64681170fd478bd5 1791725880295300805 23732224 var/datastores/dk-and-rpk-migration-02-reference-shard0000.sqlite
     210f494aedfd154fde34754d0dffb8030c42a1ae701fb3b10c9f94434293b8f8 1791725880293528131 23719936 var/datastores/dk-and-rpk-migration-02-reference-shard0001.sqlite
     37b12c1592c2f03e22a862797c30dc6f2b01c4e8c3af4f83bd672d79fed21ec9 1791725417030008225 1077248 var/datastores/dk-and-rpk-migration-02-reference-shard0002.sqlite
     db35d513a3d6278f09729dcdb7c0ba3ca645562dfaf1df4de5b9f4070099f1ce 1791725880274060223 23363584 var/datastores/dk-and-rpk-migration-02-reference-shard0003.sqlite
     755da82a40a26c4b530f4ad6451939cc6863f9370ba42ef006f275c621f595f2 1791725935837698991 10466 var/datastores/dk-and-rpk-migration-02-reference.manifest.json
     journal/wal/shm files under var/datastores: []
     ```
   - `PYTHONPATH=. ./venv/bin/python -m RunRegistry store fingerprint var/datastores/dk-and-rpk-migration-02-reference.sqlite`
     (no `--write`): exit 0.
     ```
     digest:   39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2
     format:   6
     problems: none
     recorded: matches the fingerprint taken 2026-10-11T14:38:55+01:00 by a person, at a5392fd1c4ae1c88aa929525ed3f1e6a55c51c82
     ```
     All 21 classes equal the sidecar's, by name, count and the 12-hex digest the CLI prints, and the
     whole digest equals the sidecar's. The CLI's output and the
     sidecar JSON were compared with no third read of the store (`s33/fingerprint-vs-sidecar.txt`):
     ```
     digest 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2 sidecar 39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2 True format 6 6
     sidecar classes type dict 21 cli classes 21
     classes equal (count and digest prefix): 21 of 21 names equal: True
     ```
     The full class table is in `s33/fingerprint.txt`. **This is the first real store read through
     the package's `read_inventory`.**
   - `./venv/bin/python -m datastorekit.tools.shard_key_audit var/datastores/dk-and-rpk-migration-02-reference.sqlite`:
     exit 0, `VERDICT: OK -- no inconsistency found in …/dk-and-rpk-migration-02-reference.sqlite.`
   - After (`s33/store-after.txt`): `diff` against before is empty. Every SHA-256, mtime and size
     is unchanged, and no `-journal`, `-wal` or `-shm` file is under `var/datastores/`.
4. **Entry points** (`s33/entry-points.txt`).
   - `py_compile` of `main.py`, `extract_common.py`, the six `extract_*.py` and the 14 rewritten
     `docs/` files succeeded, 22 files. The `.pyc` files were written to the scratchpad.
   - With `PYTHONPATH=.` in one process, every rewritten module outside `docs/` was imported: 68,
     `extract_common` included, all but `main.py` and the six `extract_*.py`.
   - Loaded by path: `docs/handover/quadsource_atol_sweep.py`, both
     `docs/source-remediation-verification/` scripts, `docs/a3-v2-readiness/rehearsal/store_soundness.py`
     and `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` (correction 5).
   - **The stale layer cannot be imported.** In a fresh process, `import <m>` of each of the 15
     deleted modules gives `ModuleNotFoundError` (15 of 15). `from Datastore import DatastoreObject`
     gives `ImportError: cannot import name 'DatastoreObject' from 'Datastore'`.
5. **The string default:** `config.defaults.DEFAULT_STRING_LENGTH 256 datastorekit.defaults.DEFAULT_STRING_LENGTH 256`.
6. **Ray:** `ray.is_initialized()` was `False` after every in-process import and load above, and
   at the end. It was also `False` after the rewrite script's name check and after the suites. No
   Ray process was started.

### A5 — the six `docs/` files, before and after

Each was run with `PYTHONPATH=.` from the repository root, under `timeout 120`.
`order_audit.py` was loaded by path, and its `d3_tables()` called, through a scratchpad driver
(correction 2). None got past its imports.

| File | Before (`67dd316`) | After |
|---|---|---|
| `docs/datastore-integrity-audit/fk_sweep_probe.py` | exit 1, `ImportError: cannot import name '_factories' from 'Datastore.SQL.Datastore' (…/Datastore/SQL/Datastore.py)` | exit 1, `ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (…/site-packages/datastorekit/SQL/Datastore.py)` |
| `docs/datastore-integrity-audit/lookup_keys/common.py` | the same | the same |
| `docs/datastore-integrity-audit/lookup_keys/probe_03_column_existence.py` | the same | the same |
| `docs/datastore-integrity-audit/tq_fk_probe.py` | the same | the same |
| `docs/tolerance-convergence/inventory.py` | the same | the same |
| `docs/tolerance-convergence/order_audit.py` (`d3_tables()`) | exit 1, the same | exit 1, the same |

## The deliberate-breakage record

Each diff was made in the working tree against the staged final tree, saved with `git diff`, and
reverted with `git apply -R`. Each was checked with `git apply -R --check` while applied and
`git apply --check` after the revert. Saves are in `var/dk-and-rpk-migration-04/breakage/`. The
index was never touched, and `git status` showed nothing unstaged after each. Replay in `bash` from
a clean tree at this commit.

### (i) a factory's `SQLAFactoryBase` import pointed back at `Datastore.SQL.ObjectFactories.base`

```diff
diff --git a/Datastore/SQL/ObjectFactories/redshift.py b/Datastore/SQL/ObjectFactories/redshift.py
index 5430f83..d1dc94f 100644
--- a/Datastore/SQL/ObjectFactories/redshift.py
+++ b/Datastore/SQL/ObjectFactories/redshift.py
@@ -5,7 +5,7 @@ from sqlalchemy import and_
 
 from ComputeTargets import ModelProxy, BackgroundModel
 from CosmologyConcepts import redshift
-from datastorekit.SQL.factory_base import SQLAFactoryBase
+from Datastore.SQL.ObjectFactories.base import SQLAFactoryBase
 from config.defaults import DEFAULT_REDSHIFT_RELATIVE_PRECISION
 
 
```

- **Datastore a–f**, `-p 'test_[a-f]*.py'`: `Ran 9 tests`, `FAILED (errors=3)`. The three modules
  that import `config.datastore`, and so every factory (`test_database_errors_reach_the_caller`,
  `test_exact_identity_lookups`, `test_factory_select_columns`), fail to import with
  `ModuleNotFoundError: No module named 'Datastore.SQL.ObjectFactories.base'`.
- **§3.1's script counts 1 layer import.** The commit `dd4c3bcf71bd1ae306d099e53f57a31fc0a82e4e` was
  made from a temporary index (`GIT_INDEX_FILE=<scratch>/idx git read-tree addc09f`,
  `git update-index --add` the mutated file, `write-tree`, `commit-tree -p addc09f`). It gives
  `layer imports outside prompts: 1 [('Datastore/SQL/ObjectFactories/redshift.py', 8,
  'Datastore.SQL.ObjectFactories.base')]`, 52 in all. No ref names it.

### (ii) `from Datastore import DatastoreObject` restored in one `ComputeTargets/` module

```diff
diff --git a/ComputeTargets/BackgroundModel.py b/ComputeTargets/BackgroundModel.py
index 62c16b6..1091b1b 100644
--- a/ComputeTargets/BackgroundModel.py
+++ b/ComputeTargets/BackgroundModel.py
@@ -21,7 +21,7 @@ from CosmologyModels.GenericEOS.GenericEOS import (
     BREAK_POINT_ALL,
     BREAK_POINT_DISCONTINUITY,
 )
-from datastorekit import DatastoreObject
+from Datastore import DatastoreObject
 from MetadataConcepts import store_tag
 from Quadrature.integration_metadata import IntegrationSolver, IntegrationData
 from Quadrature.supervisors.base import RHS_timer, IntegrationSupervisor
```

- `python -m unittest ComputeTargets.tests.test_background_derivatives
  ComputeTargets.tests.test_background_tau Datastore.tests.test_backgroundmodelvalue_roundtrip`:
  `FAILED (errors=3)`. Each module fails to import with `ImportError: cannot import name
  'DatastoreObject' from 'Datastore' (…/Datastore/__init__.py)`.

### (iii) the moved test deleted

```diff
diff --git a/Datastore/tests/test_inventory_report.py b/Datastore/tests/test_inventory_report.py
index 5bef664..279bb63 100644
--- a/Datastore/tests/test_inventory_report.py
+++ b/Datastore/tests/test_inventory_report.py
@@ -414,42 +414,5 @@ class TestTheRepresentationAndTheGridIdentity(_Stores):
             self.assertEqual(dict(fields_of(line))["T_z_representation"], "3")
 
 
-class TestTheReportRendersBothCosmologyTypes(unittest.TestCase):
-    """Moved here, unchanged but for its class, from
-    ``test_inventory_declarations.TestResolve`` by ``dk-and-rpk-migration`` prompt 04, when that
-    module was deleted because ``datastorekit`` carries it. This one test stayed in SGK: it asserts
-    on SGK's ``tools.inventory_report``, so it is the one test of that module that DatastoreKit's
-    ``compare_ported_tests.py`` declares ``NOT_PORTED`` (its decision U16).
-    """
-
-    def test_the_report_renders_both_cosmology_types(self):
-        """The report resolves a polymorphic parent through ``ClassInventory.parent_class``, as
-        ``resolve()`` does: every cosmology field of the full store's report is the key of the class
-        its ``cosmology_type`` names. A ``QCD_Cosmology``'s key carries ``T_z_representation``, a
-        ``LambdaCDM``'s does not. Added at the review of prompt 08, whose breakage (iv) the report
-        tests did not catch.
-        """
-        from CosmologyModels.model_ids import LAMBDACDM_IDENTIFIER, QCD_EOS_IDENTIFIER
-
-        with tempfile.TemporaryDirectory() as tmp:
-            inventory = read_inventory(build_full_store(Path(tmp)).primary, _factories)
-        report = format_inventory_report(inventory, "the-store.sqlite")
-        self.assertNotIn("<unresolved reference>", report)
-        # a type and the cosmology it names sit at one level of a rendered key
-        rendered = re.findall(
-            r"cosmology_type=(\d+), [^{}]*?cosmology=\{([^{}]*)\}", report
-        )
-        self.assertEqual(
-            {int(kind) for kind, _ in rendered},
-            {LAMBDACDM_IDENTIFIER, QCD_EOS_IDENTIFIER},
-        )
-        for kind, key in rendered:
-            with self.subTest(cosmology_type=kind, key=key):
-                self.assertIn("omega_m=", key)
-                self.assertEqual(
-                    "T_z_representation=" in key, int(kind) == QCD_EOS_IDENTIFIER
-                )
-
-
 if __name__ == "__main__":
     unittest.main()
```

- `count_datastore_tests.py`: `Datastore/tests: 319 tests loaded in 18 modules`, with
  `test_inventory_report: 18`.
- Datastore g–o: `Ran 102 tests`, OK.

### (iv) `standin_pool.py`'s `"datastorekit.SQL.ShardedPool"` reverted

```diff
diff --git a/Datastore/tests/standin_pool.py b/Datastore/tests/standin_pool.py
index faa84c4..aea83ba 100644
--- a/Datastore/tests/standin_pool.py
+++ b/Datastore/tests/standin_pool.py
@@ -40,7 +40,7 @@ from unittest import mock
 
 import ray
 
-sp_mod = importlib.import_module("datastorekit.SQL.ShardedPool")
+sp_mod = importlib.import_module("Datastore.SQL.ShardedPool")
 ds_mod = importlib.import_module("datastorekit.SQL.Datastore")
 broker_mod = importlib.import_module("datastorekit.SQL.SerialPoolBroker")
 
```

- `python -m unittest ComputeTargets.tests.test_quadsource_policy_main`: `FAILED (errors=1)`. The
  module fails to import with `ModuleNotFoundError: No module named 'Datastore.SQL.ShardedPool'`.

### (v) the actor check sees a method `inventory` added to the package's `Datastore` class

```diff
diff --git a/Datastore/tests/test_inventory_retired.py b/Datastore/tests/test_inventory_retired.py
index 5ef66ab..662521c 100644
--- a/Datastore/tests/test_inventory_retired.py
+++ b/Datastore/tests/test_inventory_retired.py
@@ -70,7 +70,10 @@ class TestTheOldServiceIsGone(unittest.TestCase):
         # Datastore is a Ray actor class, so its methods are read from its source, not its object:
         # the installed package's module, found without importing it
         source = importlib.util.find_spec("datastorekit.SQL.Datastore").origin
-        tree = ast.parse(Path(source).read_text())
+        head, sep, tail = Path(source).read_text().partition("\nclass Datastore:\n")
+        tree = ast.parse(
+            head + sep + "    def " + _INVENTORY + "(self):\n        pass\n" + tail
+        )
         (actor,) = [
             n
             for n in tree.body
```

- `python -m unittest -v Datastore.tests.test_inventory_retired`: `Ran 6 tests`,
  `FAILED (failures=1)`.
  - `test_the_datastore_actor_class_has_no_method` fails: `AssertionError: 'inventory'
    unexpectedly found in {…}`.
  - Nothing else failed. `test_the_grep_prints_nothing` and
    `test_this_module_does_not_match_itself` pass, since the name is built from `_INVENTORY`.
- The mutation names no file outside the repository: it edits the parsed text, not the installed
  file.

### (vi) the pre-commit substitution reverted (deviation 7)

```diff
diff --git a/ComputeTargets/tests/test_quadsource_integral.py b/ComputeTargets/tests/test_quadsource_integral.py
index 5ca38cc..6c720dc 100644
--- a/ComputeTargets/tests/test_quadsource_integral.py
+++ b/ComputeTargets/tests/test_quadsource_integral.py
@@ -1014,7 +1014,6 @@ def load_pre_commit_module():
             f"expected one {old_import.strip()!r} in the pre-commit file, "
             f"found {text.count(old_import)}"
         )
-    text = text.replace(old_import, "from datastorekit import DatastoreObject\n")
     with tempfile.NamedTemporaryFile("w", suffix="_old_qsi.py", delete=False) as fh:
         fh.write(text)
         path = fh.name
```

- `python -m unittest ComputeTargets.tests.test_quadsource_integral.TestOldRegimeRegression`:
  `FAILED (errors=1)`. `test_G_only_case_reproduces_the_pre_commit_total` raises
  `ImportError: cannot import name 'DatastoreObject' from 'Datastore' (…/Datastore/__init__.py)`.

## Observations not acted on

- **Stale prose**, opened as `[04-comments-name-the-layers-old-module-paths]`. A regex over every
  file in the index outside `prompts/` and `docs/` finds 23 lines (`stale-prose.txt`), the same 23
  the orchestrator's note listed. The regex matches any `Datastore.<object|contract|replication|
  shard_paths|store_reader|store_inventory|SQL.<schema|ShardedPool|Datastore|ClientPool|
  SerialPoolBroker|ProfileAgent|ObjectFactories.base>>`, any of those files' paths,
  `tools/sharded_store` and `tools/shard_key_audit` (not after `datastorekit.`), and
  ``` ``Datastore/SQL/`` ```. It skips `from`/`import` statement lines and `import_module(` calls.
  None was changed (prompt §2.4.3).
- **The orchestrator's note names `config/datastore.py` as `config/`'s one rewritten import.** It
  is `config/model_list.py:4` (`ShardedPool`). `config/datastore.py` has no layer import; it imports
  the factories only. The rule held either way: only `config/model_list.py:4` changed under
  `config/`.
- **`CLAUDE.md`'s old section said "the `RayTools/` files are not frozen either".** The new section
  says `RayTools/RayWorkPool.py` is frozen while it moves to RayPoolKit (RayPoolKit U5, U12), as
  §2.4.1 directs.
- **DatastoreKit's checklist item 3 counts ten A5 statements**, four of them in `prompts/`. The
  board's A5 entry says so.
- **README §0.2 and §1 still say "17 `docs/` statements in 13 files"**: `b510bc9`'s count. 02's
  rehearsal script added the 18th. Left as the record of that measurement.
- **`test_inventory_report.py`'s module docstring** still describes only store-fingerprint prompt
  03's tests. The moved class's own docstring says where it came from. Left, since the prompt asks
  only for the class docstring.

## State handed to the next prompt

- **The commit:** this commit. Its code is the tree measured at scratch commit `addc09f`.
- **The venv**, unchanged by 04. `RunRegistry.package_provenance()`:
  ```json
  {"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit", "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false}, "raypoolkit": null}
  ```
- **What 05's launches should expect to differ from R0's:**
  - Every manifest's `packages.datastorekit` is the record above, not `null`.
  - The code is the package's `v0.2.2`. R0's layer was SGK's `b510bc9` copy. Between them come
    DatastoreKit's `v0.2.1` fixes (a vectorized get sends copies; a reopen whose primary records an
    unknown sharded table raises `RuntimeError`, not `KeyError`; a refused open closes its actors
    and engine) and `v0.2.2`'s actor-name release on close. None changes what a build writes
    (checklist addenda), so the fingerprint should be `39809dca…` as at R0. The package's
    `read_inventory` already reproduces it on R0's store (§3.3.3).
  - On close, a pool now kills its named actors (`v0.2.2`). R0's pools left them to the collector.
    Log lines at shutdown may differ, but nothing L1–L4 compares should.
  - S1's `store_soundness.py` now imports `datastorekit.store_inventory` (`:30`). That is its only
    change.
  - `[02-the-policy-statistics-block-orders-its-lines-by-completion]` still holds: compare L4's
    block as counts, or as sorted lines where only the order differs.
- **Registry:** 21 runs, none `running`. 04 launched nothing. Nothing under `var/datastores/` was
  written, and R0's reference is byte-identical with the same mtimes.
- **`docs/OPEN_ISSUES.md`:** 108.
