# Log 03 — One inventory service: the display and the run labels read the structured inventory

**Prompt:** [`prompts/store-fingerprint/03-one-inventory-service.md`](../03-one-inventory-service.md)
**Commit:** *(this commit)*, "Move the inventory report and run labels onto one read-only service"
**Base:** `a0958062c1e2af2c72507e07b6d794a4e180a9a2` ("Record the user's decision D4 for the store
fingerprint"), clean. Prompt 04 had not landed.
**Model:** Claude Opus 5.5
**Date:** 2026-09-25
**Result:** COMPLETE. F8 and F9 shipped. The one §7 stop was F9 item 7's `git grep`, which could not
print nothing without editing prompt 02's test module (below). It was taken to the user, who chose
option (b) on 2026-09-25, relayed by the orchestrator. On a copy of the sweep store, the new
report agrees with the old one, run at the base SHA before any change, on **every count, validated
split and value-row total**. It left the copy byte-identical. All seven deliberate-breakage
mutations were caught. Closed:
- `[00-inventory-run-prunes-unvalidated-rows-by-default]` (this board);
- `qcd-background-audit`'s `[03-qcd-inventory-does-not-report-the-representation]` (that board).

No new issue was opened.

## What shipped

**F8 — the display and the run labels.**

- **`tools/inventory_report.py`** is rewritten around `format_inventory_report(inventory,
  db_name, verbose=False) -> str`, which renders what `read_inventory` returns. The three old shapes
  and their formatters are gone. For each class, in the old categories less "Value tables", it
  gives:
  - **a header**: whether the class is replicated or sharded; the record count; where the class
    has a `validated` column, the validated / unvalidated split (and a `validated NULL` count when
    there is one); where the class has a value table, the sum of `value_count` and the table's name
    (`VALUE_TABLES`); and the timestamp range;
  - **a "common to every record" line**, holding each key field whose value is the same for every
    record of the class (deviation 3);
  - **its tag sets**, each with its record count and labels;
  - **its records, grouped under their tag set**, each by its physical labels, with `[unvalidated]`
    or `[validated NULL]` and `| values: N` where they apply. Records are ordered by their resolved
    physical values, numerically where they are numbers. Without `verbose`, at most five are shown
    per tag set, followed by `... and N more`;
  - **every problem**, never truncated.

  The report opens with the total number of problems ("none" when there are none), then the shard
  serials and the total record count. `store_tag` lists every label whatever `verbose` is, and
  marks one that no record carries `(carried by no record)`. A class the categories do not name
  would be printed under "Other classes", so nothing the inventory holds can go unprinted.
- **`main.py`**:
  - The `if args.inventory:` branch moved to directly after the `args.database is None` check,
    before `ray.init`, the `ProfileAgent` and the `ShardedPool`.
  - `--inventory` with any `--drop` is refused on stderr, with exit 2. Otherwise it calls
    `read_inventory(args.database)`. A reader refusal (a missing store, a journal, a missing shard)
    goes to stderr as `!! --inventory: <the refusal>`, with exit 1. Otherwise it prints the report
    and exits 0.
  - The old branch inside the pool is deleted. `read_inventory` is imported beside
    `format_inventory_report`.
  - The `--inventory` and `--inventory-verbose` help texts say what they now do.
- **`extract_common.available_run_labels(pool)`** keeps its signature:
  - it reads `read_inventory(pool.primary)`, and returns the sorted names of the `store_tag`
    records' labels that begin with `RUN_LABEL_TAG_PREFIX`;
  - a problem in the `store_tag` class raises `RuntimeError`, naming every such problem;
  - problems in other classes are ignored;
  - its docstring states why reading a store whose pool is open is sound.
- **`ShardedPool.primary`**: a read-only property returning `self._primary_file`.

**F9 — the old service retired.** Every retired name, with its file:

| Retired | File |
|---|---|
| `inventory()` (static), 28 methods | `Datastore/SQL/ObjectFactories/`: `BackgroundModel.py` (×2: `sqla_BackgroundModelFactory`, `sqla_BackgroundModelValue_factory`), `GkNumericIntegration.py` (×2), `GkSource.py` (×2), `GkWKBIntegration.py` (×2), `QuadSource.py` (×2), `TkNumericIntegration.py` (×2), `TkWKBIntegration.py` (×2), `wavenumber.py` (×2: `wavenumber`, `wavenumber_exit_time`), `GkSourcePolicy.py`, `GkSourcePolicyData.py`, `LambdaCDM.py`, `OneLoopIntegral.py`, `QCD_Cosmology.py`, `QuadSourceIntegral.py`, `QuadSourcePolicy.py`, `integration_metadata.py`, `redshift.py`, `store_tag.py`, `tolerance.py`, `version.py` |
| `Datastore.inventory` | `Datastore/SQL/Datastore.py` |
| `InventoryConfigType` (and its comment) | `Datastore/SQL/Datastore.py` |
| `ShardedPool.inventory`, `ShardedPool._merge_queue` | `Datastore/SQL/ShardedPool.py` |
| the constructor's `inventory_config` parameter, `self._inventory_config`, the `InventoryConfigType` import | `Datastore/SQL/ShardedPool.py` |
| `inventory_config`, `_compute_target_merge`, `_no_validated_merge`, `_value_table_merge`, and their comment block | `config/sharding.py` |
| the `inventory_config` import and keyword argument | `main.py`, `docs/source-remediation-verification/analyse_greens_and_source.py`, `docs/source-remediation-verification/run_quadsource_integrals.py` |
| the three old shapes and their formatters (`_format_bucketed`, `_format_value_list`, `_format_count`, `_format_entry`, `_format_value_lines`, `_sort_key`) | `tools/inventory_report.py` |
| `ComputeTargets/tests/test_qcd_cosmology_inventory.py` (3 tests, and its hand-copy of `_build_schema`) | deleted (D4) |

- **The factory diff is 679 deletions and 0 insertions.** It deletes each method and the blank line
  after it. No import changed, because no import was used only by `inventory()` (checked with an
  AST scan of every touched file before and after). No line of `inventory_records`, `build`,
  `store`, `read_batch`, `validate`, `validate_on_startup` or `register` changed.
- **`Datastore.py`** loses exactly `inventory` and `InventoryConfigType`.
- **`ShardedPool.py`** loses exactly what F9 item 3 names, and gains `primary`. Its
  `from datetime import datetime` is now unused, because only `_merge_queue` used it. It was left in
  place, because F9 item 3 says nothing else in that file changes (observation below).

**Tests** (no Ray, nothing under `var/`; stores from `build_full_store` in temporary directories):

| Module | Tests | §3 | What it shows |
|---|---:|---|---|
| `Datastore/tests/test_inventory_report.py` | 18 | 1–3 | The categories cover the 21 classes once. The title and "problems: none". Every class's count, replicated/sharded, validated split and value-row sum equal the `StoreInventory`'s, plain and verbose. `VALUE_TABLES` is exactly the classes with `value_count`, and each sum equals an independent `sqlite3` count. The tag sets and counts. **Verbose: every distinct record is a distinct (tag set, line), and every key field of every class is shown**, on its record lines or its common line. No 64-hex digest, plain or verbose. `store_tag` lists every label and marks `unused-tag` only. Numeric order (`-9, -7, -5`). Five shown of nine, the smallest five in order. **Every problem, in full, with more than five in one class** (seven shards, and 13 problems). Floats: the last bit of one `k` changes the verbose report and not the plain one; `1/3` is `0.333333` plain and `repr` verbose; `float.hex(3.0)` as a `store_tag` label and `float.hex(8.0)` as a solver label print as themselves, and they are the only `0x` in the report. `BackgroundModel` records show `source_grid_digest` and `source_grid_construction`, for two models and for a lone one. `QCD_Cosmology` shows `T_z_representation`. |
| `Datastore/tests/test_inventory_consumers.py` | 11 | 4–6 | `main.py --inventory` in a child interpreter, run with `runpy` through a guard whose `ray.init` raises. It exits 0 and prints the report. The store's `file_state` is unchanged, and **every unvalidated row is still there** (read with `sqlite3` `mode=ro`). Verbose likewise. `--drop tk-numeric` is refused with nothing written. A missing path exits non-zero, names the path, and leaves the directory empty. `ast`: `if args.inventory:` comes after `parse_args`, and before `ray.init` and `with ShardedPool(...)`. Nothing between calls a pool, an actor, an engine, `.remote` or `.options`. The branch calls `read_inventory`, `format_inventory_report` and `sys.exit`, and nothing named `ray` or `ShardedPool`. Run labels: `["fixture", "second"]` from a stand-in pool, with the uncarried `Run_second` kept. A divergent `store_tag` on shard 1 is refused, naming `replicated-divergence: store_tag` and shard #1. An absent `QuadSourceIntegral_tags` does not matter. `extract_common` has no attribute access named `inventory`. `ShardedPool.primary` on an object made with `__new__` returns `_primary_file`, and cannot be assigned. |
| `Datastore/tests/test_inventory_retired.py` | 7 | 7 | No factory in `_factories` has `inventory`. The `Datastore` actor class, read by `ast` because it is a Ray actor class, has no `inventory` method and no `InventoryConfigType`. `ShardedPool` has neither method, its constructor has no `inventory_config` (`inspect.signature`), and `primary` is a property. `config.sharding` has no `inventory_config`. **The narrowed `git grep` prints nothing** and exits 1. **The excluded file's matches are only `_Stores.inventory`**, which calls `read_inventory`. The module does not match its own pattern. |
| `ComputeTargets/tests/test_qcd_cosmology_inventory_record.py` | 3 | D4 | The retired module's three claims, re-expressed (below). |

`Datastore/tests/inventory_report_parsing.py` is a helper module, not a test module. It reads a
report back into class sections, tag sets, record lines, the common line and problems, and splits
a line into its top-level `field=value` pairs.

**D4: the three claims, re-expressed** (`test_qcd_cosmology_inventory_record.py`):
1. Two `QCD_Cosmology` rows that differ only in `T_z_representation` (3 and 4) are two records.
   Every other key field is equal, and neither has a problem. They render as two different lines,
   plain and verbose, and the shared parameters are on the common line.
2. **`T_z_representation` sits beside `log10_max_z`**: directly **after** it, not before it as prompt
   07 of `background-solver-robustness` placed it. The display follows the order of the class's
   key, which its factory's `inventory_records` declares (`…, 'log10_max_z', 'T_z_representation'`,
   prompt 02). Reordering it for display alone would make one class an exception to the rule that
   fields appear in key order. Adjacency, which is what the old placement was for, holds.
3. **This claim changes meaning.** The old test asserted that two rows with the same
   representation and different `name` gave two labels. Under the structured inventory, `name` is
   not in the key. So they are **one key held twice**: two records, both kept, with one
   `duplicate: QCD_Cosmology: 1 key-and-tag set(s) are held by more than one record …` problem,
   which the report prints in full. The two record lines are identical, and neither shows the
   name. What the old test protected, that the representation is visible, is kept by claims 1 and 2.

## F8 item 2's choice, and what the display looks like

**The display consults the schema itself.** `_Renderer` builds `build_schema(sqla.MetaData(),
_factories)` once per report. It treats a key leaf as a float iff its column's type is an
`sqla.Float`, and never from the shape of a string. `ClassInventory` gained no field, and
`Datastore/store_inventory.py` is **unchanged**. `Record`, `key`, `canonical` and every digest are
untouched. Why: the schema is already the one definition of the column types, and an additive field
would have been a second copy of it, filled by `read_records`.

The display does not call `StoreInventory.resolve`. It resolves a parent reference through its own
index, which maps each class's reference digest to its records, built once per class (deviation 4).

On the copy of the sweep store, without `--inventory-verbose`. The report is preceded by the
reader's own legacy-path line (observation below). The header:

```
== Datastore inventory: var/store-fingerprint-check-03/new/handover-atol-sweep.sqlite ==
   problems: none
   read-only; 4 shards (#0, #1, #2, #3); 30,341 records in 21 classes
```

`QuadSourceIntegral`, in full:

```
      @@ QuadSourceIntegral (sharded): 7,706 records | 2026-09-20 21:18 – 2026-09-24 02:58
         common to every record: model={cosmology_type=0, tau_gauss_order=4, cs_tau_gauss_order=4, friction_F_gauss_order=4, source_grid_construction=2, source_grid_digest=8d1d43b6, cosmology={omega_m=0.3111, omega_cc=0.6889, h=0.6766, f_baryon=0.15817, T_CMB_Kelvin=2.7255, Neff=3.046}, z_init=0.1, tags=[LargestSourceRedshift_2.0636e+16, Run_default, SmallestSourceRedshift_0.1, SourceGridConstruction_2]}, policy={Levin_threshold=1.5}, z_source_max=1.63911e+16
         tag set 1 of 1 (7,706 records): GkOneLoopDensity, LargestSourceRedshift_2.0636e+16, ResponseRedshiftGrid_145_c51d43ac, ResponseSparsenessZ_12, Run_default, SmallestSourceRedshift_0.1, SourceGridConstruction_2, SourceRedshiftGrid_1740_8d1d43b6, TkOneLoopDensity
            - k=100000, q=100000, r=100000, z_response=174.127, atol=-32, rtol=-8
            - k=100000, q=100000, r=100000, z_response=229.562, atol=-32, rtol=-8
            - k=100000, q=100000, r=100000, z_response=302.644, atol=-32, rtol=-8
            - k=100000, q=100000, r=100000, z_response=398.992, atol=-32, rtol=-8
            - k=100000, q=100000, r=100000, z_response=526.012, atol=-32, rtol=-8
            ... and 7,701 more
```

How to read it:
- `k`, `q` and `r` are `wavenumber_exit_time` parents. Each is printed as its `k`, because every
  other field of that class (cosmology, tolerances, stepping) is the same for all eight exit times.
  Those fields are on `wavenumber_exit_time`'s own common line.
- `policy` is printed as `{Levin_threshold=1.5}`, because the two `GkSourcePolicy` rows differ only
  there.
- `atol` and `rtol` are `log10_tol`.

The `BackgroundModel` class, with its one record:

```
      @@ BackgroundModel (replicated): 1 record: 1 validated, 0 unvalidated; 1,740 BackgroundModelValue rows | 2026-09-20 21:03 – 2026-09-20 21:03
         tag set 1 of 1 (1 record): LargestSourceRedshift_2.0636e+16, Run_default, SmallestSourceRedshift_0.1, SourceGridConstruction_2
            - cosmology_type=0, tau_gauss_order=4, cs_tau_gauss_order=4, friction_F_gauss_order=4, source_grid_construction=2, source_grid_digest=8d1d43b6, cosmology={omega_m=0.3111, omega_cc=0.6889, h=0.6766, f_baryon=0.15817, T_CMB_Kelvin=2.7255, Neff=3.046}, z_init=0.1  | values: 1,740
```

## Each *prompt's choice*, and whether it was kept

| Choice | Kept? |
|---|---|
| (F8 item 2) The display types a float leaf by the schema, never by the shape of a string, and renders it to six figures, or as `repr` when verbose | **Kept.** The display consults `build_schema` itself. Tests: last bit of `k`; `1/3`; hex-looking labels. Mutation (iii). |
| (F8 item 4) `--inventory` never creates a store | **Kept.** The reader refuses a missing primary. The test shows a non-zero exit and an empty directory. |
| (F8 item 4) `--inventory` with `--drop` is refused | **Kept**, with exit 2 and a message naming the groups. |
| (F8 item 5) `available_run_labels`'s answer does not change: every `Run_` label in `store_tag`, including one no record carries | **Kept.** Test: an uncarried `Run_second` is returned. On the sweep copy the answer is `['default']`, which is also what `store_tag` holds on each of the four shards, read independently. |
| (F8 item 5) A named problem in `store_tag` is a refusal; problems elsewhere do not matter | **Kept.** Tests for both. Mutation (vi). |

The README §6.3 choices that belong to prompt 02 (keys, tags, validated, replicated comparison) are
unchanged. This prompt reads them and changes none.

## Deviations from the prompt

1. **F9 item 7's `git grep` is narrowed by one file.** *STRUCTURALLY REQUIRED*, decided by the user
   on 2026-09-25 (option (b)), relayed by the orchestrator.
   - **Why it is needed.** Once everything §2 names was retired, the prompt's command still printed
     **18 lines**, all in `Datastore/tests/test_store_inventory.py`, prompt 02's test module. Every
     one is that module's helper `_Stores.inventory(cls, **kwargs)` (`:303`), which returns
     `read_inventory(cls.store(**kwargs).primary)` (the new service), or one of its `cls.inventory()`
     / `self.inventory(...)` calls. None is a caller of the old service.
   - **Why it went to the user.** Making the grep print nothing would have meant editing an existing
     test module, which README §5 rule 7 forbids and §5.5 leaves out of scope. So the work stopped
     before any code changed, and the question went to the user. The only work done by then was §4
     steps 1–3.
   - **The decision.** Leave the module untouched. Exclude it from the grep with a git pathspec.
     Add an assertion that every match of the pattern in that file is the helper's definition or
     one of its `cls.inventory(` / `self.inventory(` calls, so that a real caller of the old service
     cannot hide there.
   - **The command the review can run**, from the repository root. It prints nothing and exits 1:
     ```bash
     git grep -nE '\.inventory\(|def inventory\(|inventory_config|_merge_queue|InventoryConfigType' -- '*.py' ':!Datastore/tests/test_store_inventory.py'
     ```
     The file it excludes is checked by
     `Datastore.tests.test_inventory_retired.TestNoReferenceRemains.test_the_excluded_file_holds_only_the_new_service_helper`:
     - each match on each line of `git grep -nE <pattern> -- Datastore/tests/test_store_inventory.py`
       must lie inside a match of `\b(cls|self)\.inventory\(` or `\bdef inventory\(cls\b`;
     - the only function named `inventory` in that module is `_Stores.inventory`, and it calls
       `read_inventory`;
     - there is no module-level function of that name.
2. **The retired names in `test_inventory_retired.py` are built at run time** (`"inven" + "tory"`,
   and so on), and its docstring avoids them. *IMPLEMENTATION CHOICE*, approved by the user with
   deviation 1. The module holds the pattern it greps for, and its checks name
   `inventory_config` and `_merge_queue`, so written plainly it would match itself.
   `test_this_module_does_not_match_itself` reads the file directly, which works before it is
   tracked, and requires no match. The other new modules contain none of the retired strings.
3. **What the display elides.** *IMPLEMENTATION CHOICE.* F8 item 1 says a parent is rendered "by
   its own resolved key". Rendered literally, every `QuadSourceIntegral` line would repeat the
   background model, its cosmology and tags, and three exit times, each with its cosmology and
   tolerances: about 1 kB per line on the sweep store. So:
   - on a class's record lines, a key field whose value is the same for every record of the class
     is left off, and printed once on that class's **"common to every record"** line;
   - a parent is rendered by those fields of its resolved key (and its tags) that **vary across its
     own class**, because the others are the same for every parent it could be. If none vary, the
     class holds one distinct key, and it is rendered by its whole key and tags;
   - a lone field is printed bare where the parent's key has only that field, or where the field is
     itself a reference (`z_response=0.1`, `k=100000`). Otherwise it keeps its name
     (`policy={Levin_threshold=1.5}`).

   **Why this hides nothing.** Every choice is made per class, so every record of a class is
   rendered through the same fields.
   - Two records whose keys differ, differ in a field that varies, so the field is shown.
   - If that field is a parent, the two parents' digests differ, so their keys or tags differ, in a
     field that varies across their class, and that field is shown, by the same argument one level
     down.
   - Records that differ only in `validated` or `value_count` differ in the line's markers.
   - Records that differ only in tags are under different tag-set headers.

   Tests: every distinct record is a distinct (tag set, line) in verbose, and every key field of
   every class appears on a record line or a common line. Nothing is elided that is not printed on
   the class's own common line. The rule is stated in the module docstring.
4. **The display resolves parents through its own index, not `StoreInventory.resolve`.**
   *IMPLEMENTATION CHOICE.* `resolve` calls `find`, which recomputes the reference digest of every
   record of the parent class on every call. Across the verbose sweep report that would be tens of
   millions of SHA-256s (13 920 `GkWKBIntegration` records against 1 740 redshifts, for one). And
   for a parent held twice (a `duplicate`), `find` returns two records, so `resolve` returns the
   digest, which the display must never print. The index maps each class's reference digest to its
   records, built once per class with `StoreInventory.digest_of`, the same digest rule. A parent
   whose digest is in no record (possible only beside a `replicated-divergence` or an unresolved
   reference) is printed `<unresolved {class} reference>`, never by digest.
5. **The distinct-lines test compares (tag set, line) pairs.** *IMPLEMENTATION CHOICE.* Tags are
   printed on the tag-set line, not on every record line, so two records that differ only in tags
   can share a line text under different tag-set headers. §3 test 1 asks for "distinct lines"; the
   pair is the line as it appears in the report. The field-coverage test beside it holds every key
   field on a record line or a common line.
6. **`--inventory`'s help names the write-side arguments it ignores**: `--prune-unvalidated`,
   `--shards`, `--db-timeout`, `--profile-db`, and also `--ray-address`. *IMPLEMENTATION CHOICE.* The
   prompt says "its help text says so". It is read here as `--inventory`'s help, which keeps the
   `main.py` diff to the lines the prompt names. `--ray-address` is ignored too, and is listed. The
   `--inventory-verbose` help changed to describe what verbose now does (every record, and floats
   as `repr`).
7. **Exit codes:** 2 for the `--drop` refusal, and 1 for a reader refusal. *IMPLEMENTATION CHOICE.*
   The prompt asks only for "non-zero".
8. **`main.py` gains one import line**, `from Datastore.store_inventory import read_inventory`,
   beside `format_inventory_report`. *STRUCTURALLY REQUIRED.* The branch calls it. It is imported at
   module scope, because `store_inventory` imports only the standard library and `sqlalchemy` at
   module scope. `extract_common` imports it likewise.
9. **`VALUE_TABLES` in the display** names the value table each class counts, for the header.
   *IMPLEMENTATION CHOICE.* `ClassInventory` does not carry the table's name, and adding a field
   would change `store_inventory.py`, which the prompt allows only for F8 item 2's float field. The
   map could drift from the factories' `values=` declarations, so a test holds it to them: it names
   exactly the classes whose records carry `value_count`, and each sum equals the table's own row
   count, read with `sqlite3`.
10. **A helper module, `Datastore/tests/inventory_report_parsing.py`**, is shared by the `Datastore`
    and `ComputeTargets` tests. *IMPLEMENTATION CHOICE.* It is test support, within §5.5's "new test
    modules, and test data". It has no `test_` prefix, so discovery does not collect it.

No UNINTENDED DRIFT. One test was strengthened during the breakage pass, before the commit, and
the record below says so: mutation (iv) was run after it.

## Verification performed

- **Baselines at `a095806`** (from the orchestrator at dispatch): AdaptiveLevin 32, ComputeTargets
  552, CosmologyModels 39, Datastore 141, LiouvilleGreen 148 (1 skipped), RunRegistry 84.
- **After the change.** All six suites were run concurrently, each with
  `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t .`:
  - AdaptiveLevin **32 OK**;
  - ComputeTargets **552 OK (the wall-clock flake passed)**, which is 552 − 3 retired + 3 replacements;
  - CosmologyModels **39 OK**;
  - Datastore **177 OK**, which is 141 + 36 (18 + 11 + 7);
  - LiouvilleGreen **148 OK** (skipped=1);
  - RunRegistry **84 OK**.
- **Scope.** `git diff --cached --stat` touches only §5.5's list:
  - `tools/inventory_report.py`, `main.py`, `extract_common.py` and `config/sharding.py`;
  - `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, and 20 factory files
    (deletions only);
  - the two `docs/source-remediation-verification/` scripts, 2 lines each;
  - the retired D4 module, four new test modules, and one test helper;
  - this log, this board, `prompts/qcd-background-audit/IMPLEMENTATION_STATE.md` and
    `docs/OPEN_ISSUES.md`.

  `Datastore/store_inventory.py`, `RunRegistry/` and every existing test module are untouched.
- **F9 item 7**, narrowed as in deviation 1, prints nothing (exit 1).
- **`black --check`** is clean on every Python file added or changed.
- **Pre-checks.**
  - `python -m RunRegistry list` shows 7 runs, none `running`, before and after.
  - No Ray was running at the start. The one Ray this prompt started (§4 step 3,
    `--ray-address local`) ended with its process. `ray stop` found no Ray processes, and `ray
    status` failed, both after step 3 and at the end.

### §4 — on copies of the sweep store

1. **Before.** `python -m RunRegistry list` showed 7 runs, none `running` (5 finished, 2 `unknown`).
   No Ray was running (`ray status` failed), and 72 GB were free.

   The originals were snapshotted read-only: every file under `var/datastores/`, the backup's
   included, 18 files. For each file the snapshot recorded the listing, the `lstat` size and
   `st_mtime_ns`, and the SHA-256 of the bytes. For each `.sqlite` it also recorded every table's
   `COUNT(*)`, read through stdlib `sqlite3` `mode=ro`.
2. **The copies.** `cp -p` of `handover-atol-sweep.sqlite` and its four shards into
   `var/store-fingerprint-check-03/old/` and `.../new/`, each then snapshotted the same way.
3. **The old report, at the base SHA, before any code changed:**
   `main.py --database var/store-fingerprint-check-03/old/handover-atol-sweep.sqlite --inventory
   --no-prune-unvalidated --shards 4 --ray-address local` (`PYTHONPATH=.`).
   - It exited 0 in **11.6 s**, with its local Ray. `ray stop` then found no Ray processes, and
     `ray status` failed.
   - The output (124 lines) is kept in the session scratchpad as `impl03_old_report.txt`.
   - The old copy's five files had the same bytes, mtimes and table counts afterwards. This store
     holds no unvalidated row and lacks no table, and the `version` row was already there, so the
     old read-write open wrote nothing here.
4. **The new report, after the change:** `main.py --database var/store-fingerprint-check-03/new/handover-atol-sweep.sqlite
   --inventory`, through §3 test 4's guard (`ray.init` raises), with no Ray running.
   - It exited 0 in **8.3 s** wall, imports included.
   - **The copy's directory was byte-identical before and after**: listing, sizes, mtimes, SHA-256s
     and table counts all compared equal to its snapshot from step 2.
   - An earlier run of the same command, before the flattening rule of deviation 3 was settled,
     took 6.2 s and also left the copy identical.

   **Old against new**:
   - the old column is the old report's lines;
   - the new column is the new report's headers;
   - "values" is the old report's value-table count against the new report's sum of the parents'
     `value_count`.

   | Class | Old report | New report | Agree? |
   |---|---|---|---|
   | `version` | 1 value | 1 record | yes |
   | `store_tag` | 10 values | 10 records | yes |
   | `LambdaCDM` | 1 value | 1 record | yes |
   | `QCD_Cosmology` | 1 value | 1 record | yes |
   | `BackgroundModel` | 1 validated, 0 unvalidated | 1 record: 1 validated, 0 unvalidated | yes |
   | `BackgroundModelValue` | 1,740 rows | 1,740 `BackgroundModelValue` rows (on `BackgroundModel`) | yes |
   | `wavenumber` | 8 values | 8 records | yes |
   | `redshift` | 1,740 values | 1,740 records | yes |
   | `wavenumber_exit_time` | 8 rows | 8 records | yes |
   | `tolerance` | 13 values | 13 records | yes |
   | `IntegrationSolver` | 7 values | 7 records | yes |
   | `GkSourcePolicy` | 2 values | 2 records | yes |
   | `QuadSourcePolicy` | 2 values | 2 records | yes |
   | `TkNumericIntegration` | 8 validated, 0 unvalidated | 8: 8 validated, 0 unvalidated | yes |
   | `TkNumericValue` | 3,865 rows | 3,865 (on `TkNumericIntegration`) | yes |
   | `TkWKBIntegration` | 8 validated, 0 unvalidated | 8: 8 validated, 0 unvalidated | yes |
   | `TkWKBValue` | 9,627 rows | 9,627 | yes |
   | `QuadSource` | 36 validated, 0 unvalidated | 36: 36 validated, 0 unvalidated | yes |
   | `QuadSourceValue` | 17,178 rows | 17,178 | yes |
   | `GkNumericIntegration` | 4,549 validated, 0 unvalidated | 4,549: 4,549 validated, 0 unvalidated | yes |
   | `GkNumericValue` | 146,445 rows | 146,445 | yes |
   | `GkWKBIntegration` | 13,920 validated, 0 unvalidated | 13,920: 13,920 validated, 0 unvalidated | yes |
   | `GkWKBValue` | 916,937 rows | 916,937 | yes |
   | `GkSource` | 1,160 validated, 0 unvalidated | 1,160: 1,160 validated, 0 unvalidated | yes |
   | `GkSourceValue` | 1,016,160 rows | 1,016,160 | yes |
   | `GkSourcePolicyData` | 1,160 rows | 1,160 records | yes |
   | `QuadSourceIntegral` | 7,706 rows | 7,706 records | yes |
   | `OneLoopIntegral` | 0 rows (empty) | 0 records (empty) | yes |

   **Every count agrees. There is no difference to explain.** The timestamp ranges agree too, class
   for class. For example, `tolerance` runs 2026-09-20 21:03 – 2026-09-23 12:41 in both, and
   `QuadSourceIntegral` 2026-09-20 21:18 – 2026-09-24 02:58.

   Beyond the counts, the new report adds:
   - the tag set of every tagged class: one per class here, as prompt 02 found;
   - every record by its physical labels, where the old report listed store-local serials
     (`wavenumber_exit=1, model=1, atol=16, rtol=12`) or nothing at all (`QuadSourceIntegral:
     7,706 rows`);
   - the problems of the store: none.
5. **Verbose:** the same command with `--inventory-verbose` exited 0 in **6.4 s** wall. The output
   is **3,031,076 bytes in 30,395 lines**.
   - It has **7,706 `QuadSourceIntegral` record lines**, all distinct, and no `... and N more`
     line.
   - Every class has exactly as many record lines as records.
   - No 64-hex digest appears.
   - The copy was byte-identical afterwards.
6. **The run labels.** `available_run_labels(SimpleNamespace(primary=<new copy's primary>))`
   returned **`['default']`** in 3.2 s. An independent `sqlite3` `mode=ro` read of `store_tag` on each
   of the four shards gives the same 10 labels on every shard. The only `Run_` label among them is
   `Run_default`. **No `Run_` label is carried by no record**: `Run_default` is in the tag set of
   every tagged class. So the prompt's choice to count uncarried labels changes nothing on this
   store. The copy was byte-identical afterwards.
7. **The originals were re-snapshotted, and are identical** (`cmp` of the two JSON snapshots).
8. **`var/store-fingerprint-check-03/` was deleted.** `var/` holds `.DS_Store` (dated 2026-09-20,
   before this work), `bootstrap-a3-resume.log`, `datastores` and `runs`. `RunRegistry list` is
   unchanged: 7 runs, none `running`. Nothing from §4 is committed. The scripts and outputs are in
   the session scratchpad:
   - `impl03_snapshot.py` and `impl03_guard.py`;
   - `impl03_old_report.txt`, `impl03_new_report.txt` and `impl03_new_verbose.txt`;
   - the snapshots.

## The deliberate-breakage record

Each mutation was applied to the working tree with every change of this prompt staged, so each
diff below is `git diff` (working tree against the index) and applies with `git apply` to this
commit. The named test modules were run under each, and the file was then restored with
`git checkout -- <file>`, after which `git diff --quiet` was true. No mutation is committed.

### (i) The `--inventory` branch is moved back after `ray.init`

```diff
diff --git a/main.py b/main.py
index a7b1278..206778c 100644
--- a/main.py
+++ b/main.py
@@ -283,6 +283,9 @@ if args.database is None:
     parser.print_help()
     sys.exit()
 
+# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
+ray.init(address=args.ray_address)
+
 # --inventory reads the closed store read-only (Datastore.store_inventory.read_inventory), before
 # anything below can start Ray, build a ProfileAgent or open a ShardedPool -- whose open would run
 # --drop, create missing tables and, by default, prune unvalidated rows. It never creates a store:
@@ -307,9 +310,6 @@ if args.inventory:
     )
     sys.exit(0)
 
-# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
-ray.init(address=args.ray_address)
-
 VERSION_LABEL = "2025.1.1"
 
 specified_drop_actions = [x.lower() for x in args.drop]
```
Run: `Datastore.tests.test_inventory_consumers`. **Failed** (5 failures, 11 run):
- `TestTheBranchComesFirst.test_the_inventory_branch_precedes_ray_and_the_pool`, where the branch
  index is after `ray.init`'s;
- all four `TestMainInventoryIsReadOnly` tests (`test_it_prints_the_report_and_writes_nothing`,
  `test_verbose`, `test_drop_is_refused` and
  `test_a_path_that_does_not_exist_is_refused_and_not_created`). The guard's `ray.init` raised
  `GUARD: ray.init was reached` in each child before anything else happened.

### (ii) The display leaves `source_grid_digest` out of the `BackgroundModel` record

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 984f57f..fbea0a8 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -181,7 +181,15 @@ class _Renderer:
 
     def fields(self, name: str) -> Tuple[str, ...]:
         records = self.cls(name).records
-        return tuple(records[0].key) if len(records) > 0 else ()
+        return (
+            tuple(
+                f
+                for f in records[0].key
+                if not (name == "BackgroundModel" and f == "source_grid_digest")
+            )
+            if len(records) > 0
+            else ()
+        )
 
     def varying(self, name: str) -> Tuple[Tuple[str, ...], bool]:
         """The key fields whose value is not the same for every record of ``name``, in key order,
```
Run: `Datastore.tests.test_inventory_report` and
`ComputeTargets.tests.test_qcd_cosmology_inventory_record`. **Failed** (1 failure, 3 errors, 21
run):
- `TestTheDisplaySaysWhatTheInventoryHolds.test_verbose_shows_every_key_field_of_every_class`
  (`BackgroundModel`);
- `TestTheRepresentationAndTheGridIdentity.test_background_model_records_show_the_grid_identity`,
  plain and verbose. These are errors, not failures: sorting the missing digests raises on `None`;
- `TestTheRepresentationAndTheGridIdentity.test_a_lone_background_model_shows_every_field`, an
  error (`KeyError: 'source_grid_digest'`).

### (iii) Verbose mode renders floats to six significant figures

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 984f57f..00a56e1 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -220,7 +220,7 @@ class _Renderer:
     def leaf_text(self, name: str, field: str, value: Any) -> str:
         if field in self.float_fields(name) and value is not None:
             x = self._float(value)
-            return repr(x) if self.verbose else f"{x:.6g}"
+            return f"{x:.6g}"
         if isinstance(value, str):
             if value == "" or value != value.strip() or _QUOTE_IF & set(value):
                 return repr(value)
```
Same run. **Failed** (2 failures):
- `TestFloats.test_the_last_bit_of_k_shows_in_verbose_mode_only`, where the verbose reports of the
  two stores are equal;
- `TestFloats.test_floats_are_numbers_at_six_figures_and_repr_in_verbose`.

### (iv) The display shows at most five problems

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 984f57f..b331614 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -398,7 +398,7 @@ def _class_lines(
 
     if len(cls.problems) > 0:
         lines.append(f"{_DETAIL_INDENT}{_plural(len(cls.problems), 'problem')}:")
-        for problem in cls.problems:
+        for problem in cls.problems[:5]:
             lines.append(f"{_RECORD_INDENT}!! {problem}")
 
     return lines
```
Same run. **Failed** (2 failures):
`TestTheDisplaySaysWhatTheInventoryHolds.test_every_problem_appears_in_full_when_there_are_many`,
plain and verbose.

**The test was strengthened before this run.** Its first version had seven `absent-table` problems,
one in each of seven classes. So a per-class cap of five, which is where this mutation puts it,
would have passed it. It now builds a seven-shard store: `QuadSourceIntegral_tags` is absent from
shards #1–#6, giving 6 problems in `QuadSourceIntegral` alone, and shard #1 also lacks six other
association tables, giving 13 in all. It asserts that one class has more than five. The mutation
was run against the strengthened test, and the diff above applies to this commit.

### (v) A parent is rendered by its digest

```diff
diff --git a/tools/inventory_report.py b/tools/inventory_report.py
index 984f57f..983a139 100644
--- a/tools/inventory_report.py
+++ b/tools/inventory_report.py
@@ -295,7 +295,7 @@ class _Renderer:
     def value_text(self, name: str, record: Record, field: str) -> str:
         value = record.key[field]
         if field in self.cls(name).parents:
-            return self.ref_text(self.parent_class(name, record, field), value)
+            return value
         return self.leaf_text(name, field, value)
 
     def value_sort(self, name: str, record: Record, field: str) -> tuple:
```
Same run. **Failed** (2 failures):
- `TestTheDisplaySaysWhatTheInventoryHolds.test_no_digest_appears`;
- `TestFloats.test_the_last_bit_of_k_shows_in_verbose_mode_only`, where the plain reports now differ
  through the exit times' digests.

### (vi) `available_run_labels` ignores a problem in `store_tag`

```diff
diff --git a/extract_common.py b/extract_common.py
index d03b67f..da31cdc 100644
--- a/extract_common.py
+++ b/extract_common.py
@@ -128,7 +128,7 @@ def available_run_labels(pool) -> List[str]:
     inventory = read_inventory(pool.primary)
     store_tag = inventory["store_tag"]
 
-    if len(store_tag.problems) > 0:
+    if False:
         raise RuntimeError(
             "extract: cannot tell which runs this datastore holds, because its store_tag class "
             "has problems: " + "; ".join(store_tag.problems)
```
Run: `Datastore.tests.test_inventory_consumers`. **Failed** (1 failure, 11 run):
`TestRunLabels.test_a_problem_in_store_tag_is_a_refusal`, where no `RuntimeError` was raised.

### (vii) `inventory_config` is restored to `config/sharding.py`

Applied by reverse-applying this commit's own hunk for that file
(`git diff --cached -- config/sharding.py | git apply -R`).

```diff
diff --git a/config/sharding.py b/config/sharding.py
index b619bcd..34cf76e 100644
--- a/config/sharding.py
+++ b/config/sharding.py
@@ -40,6 +40,60 @@ read_table_config = {
 }
 
 
+# Merge policies for pool.inventory() calls on sharded tables.
+# Each field in the factory's inventory() return value needs a merge policy:
+#   lists/sets → "extend"
+#   datetimes  → "earliest" or "latest"
+#   numbers    → "sum" (also "min"/"max")
+#
+# Only sharded classes go in inventory_config. Replicated classes are served
+# from a single shard and are never merged, so an entry for one here would be
+# harmless but misleading.
+
+_compute_target_merge = {
+    "validated": {
+        "labels": "extend",
+        "earliest_timestamp": "earliest",
+        "latest_timestamp": "latest",
+    },
+    "unvalidated": {
+        "labels": "extend",
+        "earliest_timestamp": "earliest",
+        "latest_timestamp": "latest",
+    },
+}
+
+_no_validated_merge = {
+    "count": "sum",
+    "earliest_timestamp": "earliest",
+    "latest_timestamp": "latest",
+}
+
+_value_table_merge = {"count": "sum"}
+
+inventory_config = {
+    # Group A: compute targets with a validated/unvalidated split
+    "TkNumericIntegration": _compute_target_merge,
+    "TkWKBIntegration": _compute_target_merge,
+    "QuadSource": _compute_target_merge,
+    "GkNumericIntegration": _compute_target_merge,
+    "GkWKBIntegration": _compute_target_merge,
+    "GkSource": _compute_target_merge,
+    # Group B: compute targets with no validated column, and potentially
+    # numerous, so no label list is reported
+    "GkSourcePolicyData": _no_validated_merge,
+    "QuadSourceIntegral": _no_validated_merge,
+    "OneLoopIntegral": _no_validated_merge,
+    # Group C: high-volume value tables, "timestamp": False -- count only
+    "TkNumericValue": _value_table_merge,
+    "TkWKBValue": _value_table_merge,
+    "QuadSourceValue": _value_table_merge,
+    "GkNumericValue": _value_table_merge,
+    "GkWKBValue": _value_table_merge,
+    "GkSourceValue": _value_table_merge,
+}
+
+
 shard_key_type = wavenumber
 
 
```
Run: `Datastore.tests.test_inventory_retired`. **Failed** (2 failures, 7 run):
- `TestTheOldServiceIsGone.test_config_sharding_has_no_merge_policies`;
- `TestNoReferenceRemains.test_the_grep_prints_nothing`, where the narrowed grep prints
  `config/sharding.py:…` lines.

## Observations not acted on

None of these is opened as an issue. None is a defect with an impact on a result.

- **The reader's legacy-path notice goes to stdout, at the head of the report.** On the sweep copy
  the first line of `main.py --inventory`'s output is `!! Primary database "…" records 4 shard(s) by
  legacy absolute path in "…/var/datastores"; reading them as siblings …`.
  `ShardedPool._read_closed_store` prints it, and prompt 01's reader reuses that method. The old
  report printed the same line, and `available_run_labels` prints it too. It is accurate and
  harmless, but a consumer that parses the report's stdout must skip it. `ShardedPool`'s read paths
  are out of this campaign's scope (README §1).
- **`Datastore/SQL/ShardedPool.py` still imports `datetime`, which nothing there uses now.** Only
  `_merge_queue` used it. F9 item 3 says nothing else in that file changes, so the import stays.
  Whichever campaign next edits that file's imports can drop it.
- **`available_run_labels` now reads the whole structured inventory** to get one class. That took
  3.2 s on the sweep copy (`read_inventory` does every class, because each class's records depend on
  its parents'), and it grows with the store. Every extract script pays it once, at start-up. The
  prompt prescribes `read_inventory(pool.primary)`. A `store_tag`-only read would need
  `store_inventory.py` to change, which this prompt may not do, and correctness is the objective.
- **Every `QuadSourceIntegral` and `GkSourcePolicyData` record in the sweep store uses the
  `GkSourcePolicy` with `Levin_threshold=1.5`.** The `5.0` row is referenced by no record there.
  The new report shows this on the two common lines, which the old report could not. It is a fact
  about that store, recorded for prompt 05's reading of the others.
- **The stale sentence in `docs/OPEN_ISSUES.md`'s preamble** ("Of the 90 above, 87 …"), which logs
  01 and 02 recorded, is still there. It is prose, not a row. This commit corrected the count in the
  header (102 → 100) and the two paragraphs that name this campaign's closures.

## State handed to the next prompt

- **One inventory service:** `Datastore.store_inventory.read_inventory`. There is no `inventory()`
  anywhere: not on a factory, not on `Datastore`, not on `ShardedPool`. `config.sharding` has no
  merge policies. `ShardedPool(...)` takes no `inventory_config`, and a caller passing it now gets a
  `TypeError`.
- `tools/inventory_report.format_inventory_report(inventory, db_name, verbose=False)`.
  `Datastore/tests/inventory_report_parsing.py` reads its output back into sections, for tests.
- `main.py --inventory` is read-only and needs no Ray. It exits 0, or 1 on a reader refusal, or 2
  on `--drop`.
- `ShardedPool.primary` is the resolved primary path, read-only.
  `extract_common.available_run_labels(pool)` reads `read_inventory(pool.primary)["store_tag"]`,
  and refuses a problem there.
- **For prompt 04:**
  - `Datastore/store_inventory.py` is exactly as prompt 02 left it, so its fingerprint digests the
    same records;
  - `RunRegistry/` is untouched;
  - the narrowed F9 grep, and `test_inventory_retired`, will fail if anything reintroduces a
    retired name.
- **Baselines:** `Datastore` **177**; `ComputeTargets` **552** (3 retired, 3 added); the others
  unchanged.
