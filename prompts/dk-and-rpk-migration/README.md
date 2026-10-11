# Campaign — dk-and-rpk-migration

**Written:** 2026-10-11 on branch `handover-remedial` at `b510bc9`, by Claude Opus 5.5. It is
planned from the two planning briefs in this folder, from DatastoreKit's and RayPoolKit's SGK
adoption checklists, and from the user's decisions U1–U4 of the same day (§6.1).
- **Briefs:** [`sgk-datastorekit-planning-brief.md`](sgk-datastorekit-planning-brief.md) and
  [`sgk-raypoolkit-planning-brief.md`](sgk-raypoolkit-planning-brief.md).
- **Checklists:** DatastoreKit's `docs/adoption/secondarygwkit.md`, read with both addenda, and
  RayPoolKit's `docs/adoption/secondarygwkit.md`.

§0.2 re-measures both checklists at `b510bc9`. §2 fixes each prompt's charter and acceptance.
**All nine prompts are written**: 01–04 on 2026-10-11, after the user reviewed the plan at
`4532188`, and 05–09 the same day, after 04's review (§2.1).

## 0. Why this campaign exists

The datastore layer (`Datastore/`, `ShardedPool`) and the Ray work queue (`RayTools/RayWorkPool.py`)
have each been extracted into a package of their own:
- **DatastoreKit**, `datastorekit` **`v0.2.2`** (tag `c32d027` on `6d8ee9a`);
- **RayPoolKit**, `raypoolkit` **`v0.2.0`** (tag `4e7eb15` on `eaef0ce`).

SGK, ChamPBH and StochasticInstantons will all depend on these packages. SGK's copies have been
frozen since `b510bc9` (`CLAUDE.md`, "The datastore layer is frozen"; RayPoolKit decision U5). The
freeze ends when SGK adopts the packages. This campaign does that, and it is the last step before
the A3 v2 numerical campaign: A3 v2's store is to be built through the packages, not through the
frozen copies.

Each package names its own acceptance, its gate **G2**:
- **DatastoreKit's G2:** SGK has adopted `v0.2.2`, its remaining suites pass, and a rehearsal
  rebuild through the package reproduces the reference fingerprint. That fingerprint is format 6,
  digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, with all 21 classes
  equal (`docs/a3-v2-readiness-verification.md:25-27`).
- **RayPoolKit's G2:** `RayTools/` is gone, nothing imports it, the pin is in place, SGK's six test
  roots pass, and every one of the 54 pool sites constructs.

### 0.1 Order, and why there are three rehearsals

The user chose **DatastoreKit first** (U1). Each swap is followed by its own rehearsal (U2):

```
01 provenance ─ 02 R0 (start tree) ─ 03 DK pin ─ 04 DK swap ─ 05 R1 ─ 06 RPK pin ─ 07 RPK swap ─ 08 R2 ─ 09 close
```

- **R0 is needed, and is not a formality.** The last rebuild that reproduced the fingerprint was
  `datastore-generic`'s 09 rehearsal, at `53d4e18`. Since then three commits have changed the
  layer:
  - `fee1fbf`, the drop refusal;
  - `4c34f13`, `resolve()`'s `None`;
  - `086c81a`, the closed-store prefix.

  None of them was rehearsed. DatastoreKit's import commit `6f7f291` comes *after* all three, so
  the package is a copy of a tree no rehearsal has run. If R1 missed the fingerprint without R0,
  nothing could say whether the package or `datastore-generic-followup` caused it. R0 at the start
  tree removes that ambiguity.
- **R1 isolates DatastoreKit.** Between R0 and R1 the only change to what runs is the package
  swap: 03 and 04, plus 01's manifest field, which runs no layer code.
- **R2 isolates RayPoolKit.** R2 is compared with R1. It also measures the one behavioural change
  RayPoolKit's checklist names, at `main.py:3247` (§6.2 C8), at no extra cost.
- **Why DatastoreKit first.** `test_layer_is_generic.py` is deleted once, by 04, and never edited
  only to be deleted. The one line of `RayTools/` that DatastoreKit's swap must touch,
  `RayTools/RayWorkPool.py:8`, is the line RayPoolKit's decision U12 allows. The larger swap is
  measured against the frozen pool, which is the configuration every earlier rehearsal ran.

### 0.2 Reconciliation against `b510bc9`

Measured on 2026-10-11. The outputs are in [`measure/out/`](measure/out/).

**The checklists hold at `b510bc9`, byte for byte.**
- DatastoreKit's `docs/extraction/measure_client_imports.py sgk --commit b510bc9`, run from its
  root with its venv, equals its checklist's appendix byte for byte
  ([`b510bc9-datastorekit-measure_client_imports.md`](measure/out/b510bc9-datastorekit-measure_client_imports.md)).
  It counts 233 layer imports in 138 files (41 nested), 13 string literals and 11 `import_module`
  calls, and 100 factory imports (98 outside the factory package).
- RayPoolKit's `docs/adoption/measure_client_pools.py sgk --commit b510bc9` equals its appendix
  byte for byte
  ([`b510bc9-raypoolkit-measure_client_pools.md`](measure/out/b510bc9-raypoolkit-measure_client_pools.md)).
  It counts 7 imports, 54 `RayWorkPool(...)` calls (every one "constructs"), 8 string literals and
  0 `import_module` calls.

So every figure the checklists give is today's. A prompt re-measures at its own `HEAD` before it
acts, and its numbers govern.

**Both freezes held.**
- `git diff --stat 6f7f291 b510bc9` over the 17 layer files is empty.
- `b510bc9:RayTools/RayWorkPool.py` is blob `b4666fa89abb…`.

**Both tags are on their remotes.**
- `git ls-remote` gives `refs/tags/v0.2.2` → `c32d027` (DatastoreKit);
- it gives `refs/tags/v0.2.0` → `4e7eb15` (RayPoolKit).

**The venv.**
- Python 3.12.15, `ray==2.43.0`, `SQLAlchemy==2.0.39`: the low end both packages test.
- `pip show datastorekit raypoolkit` finds neither, and `pip check` reports nothing broken.
- `pip freeze` is saved as [`b510bc9-pip-freeze.txt`](measure/out/b510bc9-pip-freeze.txt). 03 and
  06 compare against it.
- Each package declares only `ray>=2.43` (and `datastorekit` `sqlalchemy>=2.0.39,<2.1`), which the
  venv already satisfies. So an install from either pin should add one distribution and move none.

**The six test roots at `b510bc9`**, run in the foreground, serially, at load 7–9:

| Root | Tests | Result |
|---|---|---|
| `AdaptiveLevin/tests` | 32 | OK |
| `CosmologyModels/tests` | 39 | OK |
| `LiouvilleGreen/tests` | 148 | OK (1 skipped) |
| `RunRegistry/tests` | 280 | OK |
| `ComputeTargets/tests` | 693 | 1 failure: `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, the known wall-clock flake. It passed when re-run alone ([`b510bc9-flake-rerun.txt`](measure/out/b510bc9-flake-rerun.txt)) |
| `Datastore/tests` | 739 = 163 + 185 + 200 + 191 | OK, in the four groups of §7 |

**Datastore/tests by module**
([`count_datastore_tests.py`](measure/scripts/count_datastore_tests.py) →
[`b510bc9-datastore-tests-by-module.txt`](measure/out/b510bc9-datastore-tests-by-module.txt)):
- 739 tests load in 44 modules;
- **419** are in the 26 modules the package now carries (checklist item 7), all 26 present;
- **320** are in the 18 that stay.

So after 04 the root holds 320 tests:
- plus 1, the unported `test_the_report_renders_both_cosmology_types`, which moves;
- less what 04 removes from `test_inventory_retired` (§6.2 C3: 1 test).

The fall from 739 to 320 is by design; the moved 419 now run in DatastoreKit's own suite.

**The rehearsal.**
- **Cost.** The `--rehearsal` build (3 k over $10^5$–$10^6$/Mpc, `zend` 250, atol $10^{-32}$,
  rtol $10^{-7}$, 6 CPUs) took 4 min at `53d4e18` and 13 min at `dbd8a51`, by the manifests'
  `created` and last heartbeat. So prompt 07's protocol, of four launches plus the readers, costs
  well under an hour per rehearsal.
- **The `main.py:3247` measurement comes free.** The rehearsal's `stdout.log` prints
  `** GkSourcePolicyData SUMMARY STATISTICS` (`main.py:3265`). In the 09 rehearsal's log it is at
  `:83`, with 291 instances, 109 / 26 / 156 numeric / mixed / WKB. That is the measurement
  RayPoolKit's checklist item 5 advises, so comparing R1's block with R2's gives it.

**What the swaps touch, by grep at `b510bc9`.**
- No staying file uses a plain `import Datastore…` whose attribute uses would also need rewriting.
  The four such statements are in staying tests, and import either `Datastore.tests.standin_pool`
  or a factory module, both of which stay.
- The two tool paths are named, outside `prompts/` and dated `docs/`, only at:
  - `CLAUDE.md:62`;
  - `RunRegistry/stores.py:5`, a docstring;
  - the two tools' own usage lines, which are deleted with the tools.

**No run is live.** `python -m RunRegistry list` shows 17 runs: 15 finished and 2 `unknown` (the
pre-registry `a3-pilot` pair), none `running`.

## 1. Scope

**In scope:**
- `RunRegistry/__init__.py`'s manifest: the installed packages' provenance (01);
- `requirements.txt`: the two pins (03, 06);
- the 17 layer files: 15 deleted, and the two `__init__.py` emptied (04, U3);
- every layer import outside `prompts/`, rewritten by the module map (04). This takes in:
  - the factories, `config/`, `main.py`, the six `extract_*.py` and `extract_common.py`;
  - `ComputeTargets/`, `CosmologyConcepts/`, `CosmologyModels/`, `MetadataConcepts/` and
    `Quadrature/`;
  - `RunRegistry/` and `tools/inventory_report.py`;
  - `RayTools/RayWorkPool.py:8` (U12);
  - the staying tests and fixtures;
  - the 17 statements in 13 `docs/` files (§6.2 C2);
- `Datastore/tests/`: the 26 modules the package carries, deleted, with `schema_description.py`;
  one test moved; `test_inventory_retired.py` changed (04);
- `RayTools/`, deleted, and its 7 imports rewritten (07);
- the three rehearsals, R0, R1 and R2, and their verification document (02, 05, 08);
- prose that a prompt's own commit makes false (§6.2 C10);
- the hand-over of the four inherited issues (09).

**Out of scope:**
- **What the code writes.** No table, column, key or stored value changes, and the fingerprint
  must not move (§5 rule 9).
- **DatastoreKit and RayPoolKit themselves.** No prompt edits either repository (§5 rule 12).
  A defect found in a package stops the prompt (§5 rule 11).
- **The factories' location.** They stay in `Datastore/SQL/ObjectFactories/` (U3), and their 98
  imports from outside the package are untouched.
- **The 44 lookup-only pool sites' handler keywords.** `store_handler=None, persist_handler=None`
  stay (§6.2 C7).
- **`prompts/`.** Its 51 layer imports and 156 lines naming `RayTools` are history, and are not
  rewritten.
- **Dated `docs/` prose** (§6.2 C10).
- **The six `docs/` files that already fail at `b510bc9`.** They import `_factories` from
  `Datastore.SQL.Datastore`, which left that module at `19f07ee`. 04 rewrites their module paths
  like every other import, so they fail as before, but on the package's module (§6.2 C2).
- **Every store under `var/`** but the six this campaign builds (§3).
- **The A3 v2 build itself.**

## 2. Prompts

| # | Prompt | Covers | Model | Status |
|---|---|---|---|---|
| 01 | [`01-record-the-installed-packages-in-each-manifest.md`](01-record-the-installed-packages-in-each-manifest.md) | U4. A run's manifest records the installed `datastorekit` and `raypoolkit` (version, VCS URL and commit, editable or not), or `null` for one not installed | Sonnet | written 2026-10-11 |
| 02 | [`02-rehearse-the-start-tree.md`](02-rehearse-the-start-tree.md) | **R0.** §4.1's protocol at the start tree. Every fingerprint is `39809dca`. It closes the `53d4e18` → `b510bc9` gap (§0.1) | Opus | written 2026-10-11 |
| 03 | [`03-pin-datastorekit-v0.2.2.md`](03-pin-datastorekit-v0.2.2.md) | The `datastorekit` pin in a commit of its own; the venv installed from it, adding one distribution, at the tag's commit | Sonnet | written 2026-10-11 |
| 04 | [`04-adopt-datastorekit.md`](04-adopt-datastorekit.md) | **The DatastoreKit swap.** The 17 files go (two emptied, U3), every layer import outside `prompts/` is rewritten, the carried tests are deleted, and the staying ones are adapted. The freeze section of `CLAUDE.md` is rewritten | Opus | written 2026-10-11 |
| 05 | [`05-rehearse-through-datastorekit.md`](05-rehearse-through-datastorekit.md) | **R1.** §4.1's protocol through the package. It equals R0 and `39809dca`. **DatastoreKit's G2 holds** | Opus | written 2026-10-11 |
| 06 | [`06-pin-raypoolkit-v0.2.0.md`](06-pin-raypoolkit-v0.2.0.md) | The `raypoolkit` pin in a commit of its own, installed as 03 | Sonnet | written 2026-10-11 |
| 07 | [`07-adopt-raypoolkit.md`](07-adopt-raypoolkit.md) | **The RayPoolKit swap.** `RayTools/` goes and 7 imports are rewritten; every one of the 54 sites constructs | Sonnet | written 2026-10-11 |
| 08 | [`08-rehearse-through-both-packages.md`](08-rehearse-through-both-packages.md) | **R2.** §4.1's protocol through both packages. It equals R1 and `39809dca`; the `GkSourcePolicyData` statistics equal R1's. **RayPoolKit's G2 holds** | Opus | written 2026-10-11 |
| 09 | [`09-close-the-migration.md`](09-close-the-migration.md) | The four inherited issues handed to DatastoreKit; `CLAUDE.md`'s test command names all six roots; the final measurements; the G2 records drafted for the two package repositories | Sonnet | written 2026-10-11 |

### 2.1 When each prompt file is written

Every charter below is fixed by a checklist or by §4.1, so no prompt waits on another's *scope*.
The files are still written in batches, each just before its dispatch, so that each carries the
previous review's findings:
- **01–04 are written first**, once the user has reviewed this plan.
- **05–08** follow 04's review: 04's log fixes the expected counts and names that they cite.
- **09** follows 08's review.

**As written** (2026-10-11): 05–08 after 04's review, as planned. **09 was written with them, at
the user's request**, not after 08's review. Its task is fixed by its charter, so nothing in it
waits on 05–08's results; what they produce (counts, findings, issues opened) reaches 09 through
its orchestrator note, which is written after 08's review and governs where the two differ.

### 2.2 The charters

**01 — Record the installed packages in each run's manifest** (Sonnet).
- **What it changes.**
  - `RunRegistry/__init__.py` gains `RECORDED_DISTRIBUTIONS = ("datastorekit", "raypoolkit")`
    and `package_provenance() -> dict`.
  - For each name, `package_provenance()` gives `None` if the distribution is not installed.
    Otherwise it gives `{"version", "url", "commit", "requested_revision", "editable"}`, read
    through `importlib.metadata` and the distribution's PEP 610 `direct_url.json`, with `None` for
    any field the install does not record.
  - `begin()` writes it to the manifest as `packages`, beside `git_provenance()`
    (`RunRegistry/__init__.py:586`).
- **Acceptance.**
  1. Tests with stand-in distributions, and no real install, give the right answer for each of
     four cases: not installed; a VCS install at a tag (commit and requested revision); an
     editable install (`editable: true`); and a `direct_url.json` that is absent or unreadable.
  2. A manifest written by a test's `begin()` holds `packages`, with both entries `null` in
     today's venv.
  3. RunRegistry's 280 tests pass, plus the new ones, and the other five roots are unchanged.
- **Out.**
  - The store sidecar's `history` and `taken` records. Their key sets are checked exactly
    (`RunRegistry/stores.py:162`, `:207`), and changing them is not asked for.
  - `list`'s output.
  - Any refusal. The registry records; it does not refuse, even for an editable install.

**02 — Rehearse the start tree, R0** (Opus).
- **What it does.** §4.1's protocol, slug prefix `dk-and-rpk-migration-02`, at 01's tree. The
  layer is `b510bc9`'s.
- **Acceptance.**
  1. §4.1's L1–L3 hold, and both stores' fingerprints equal `39809dca…` with all 21 classes.
  2. Every manifest's `packages` is `{"datastorekit": null, "raypoolkit": null}`.
  3. The reference run's `GkSourcePolicyData SUMMARY STATISTICS` block and its stage counts are
     recorded as the campaign's baseline (L4).
- **Its own stop condition.** If R0 misses the fingerprint, the regression is in SGK's own tree,
  between `53d4e18` and `b510bc9`, and it is not this campaign's to fix. The prompt stops and
  reports, and nothing after it is dispatched.
- **It creates** `docs/dk-and-rpk-migration-verification.md`, adds its §1, and ships the rehearsal
  scripts (§4).

**03 — Pin `datastorekit` `v0.2.2`** (Sonnet).
- **What it changes.**
  - `requirements.txt` gains
    `datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2`, in sorted position, and
    nothing else changes.
  - The venv is installed from it: `./venv/bin/pip install -r requirements.txt`, which needs the
    network.
- **Acceptance.**
  1. `pip freeze` differs from `b510bc9-pip-freeze.txt` by exactly one added line, `datastorekit`
     at `6d8ee9ae2c05…`.
  2. `pip check` is clean.
  3. `package_provenance()["datastorekit"]` gives version `0.2.2`, commit `6d8ee9ae…`, requested
     revision `v0.2.2`, and `editable: false`.
  4. The installed `datastorekit/` equals `git -C <DatastoreKit> archive v0.2.2 datastorekit`,
     less `datastorekit/tests`, file by file.
  5. All six roots give §0.2's counts. Nothing imports the package yet.
- **Rollback** is the revert plus `./venv/bin/pip uninstall -y datastorekit`, recorded in the log.

**04 — Adopt `datastorekit`** (Opus). The swap is one commit, because the tree cannot pass
between its parts: the deletion, the rewrite and the test changes stand or fall together.
- **What it changes.**
  1. It deletes the 15 layer files, and replaces `Datastore/__init__.py` and
     `Datastore/SQL/__init__.py` with empty files (U3). `tools/__init__.py` and
     `tools/inventory_report.py` stay.
  2. It rewrites every layer import outside `prompts/` by the adoption README §2 module map,
     **module path only**. That includes:
     - the nested imports;
     - `RayTools/RayWorkPool.py:8` (U12);
     - the string targets in staying code: `Datastore/tests/standin_pool.py:43-48` and
       `Datastore/tests/test_wkb_numeric_parent_key.py:88`;
     - the 17 `docs/` statements and the three `import_module` calls at
       `docs/datastore-integrity-audit/replicated_write_fault_probe.py:37-39` (§6.2 C2).

     The module is reached as `importlib.import_module("datastorekit.SQL.Datastore")` wherever
     the code needs the module and not the actor class.
  3. In `Datastore/tests/` it:
     - deletes the 26 modules and `schema_description.py` (§6.2 C5);
     - moves `test_inventory_declarations.TestResolve.test_the_report_renders_both_cosmology_types`
       into `test_inventory_report.py`, changing nothing but its imports (C4);
     - changes `test_inventory_retired.py` as §6.2 C3 says;
     - leaves `standin_pool.py` but for its import strings (C6).
  4. It rewrites prose that this commit makes false (C10):
     - `CLAUDE.md`'s section "The datastore layer is frozen" becomes the rule after adoption
       (§4.3);
     - `CLAUDE.md:62`'s tool paths;
     - `RunRegistry/stores.py:5`'s docstring.
- **Acceptance.**
  1. **By DatastoreKit's script, at the commit.** 15 layer files are absent, and the two
     `__init__.py` are present and 0 bytes. Outside `prompts/` there are 0 import statements,
     0 string literals and 0 `import_module` calls naming a layer module. `prompts/` still holds
     its 51 statements. The factory-import count changes only by the deleted tests' imports.
  2. **The suites.**
     - Datastore gives exactly the expected count: 320 + 1 − C3's removals, so 320 if C3 removes
       one. The log derives it from the by-module count.
     - The other five roots give their counts as at 03.
     - `black --check` is clean on every Python file touched.
  3. **Commands through the package.**
     - `python -m datastorekit.tools.sharded_store --help` and
       `python -m datastorekit.tools.shard_key_audit --help` exit 0.
     - `PYTHONPATH=. ./venv/bin/python -m RunRegistry list` runs.
     - `python -m RunRegistry store fingerprint` of R0's reference store, **without** `--write`,
       read-only, gives `39809dca…` with all 21 classes. This first reads a real store through the
       package's `read_inventory` before any build is run through it.
  4. **`main.py` and the six `extract_*.py`** byte-compile, and every module they import
     resolves. `test_main_plumbing`'s `ast` extraction is the means for `main.py`; it is never
     executed.
  5. **The deliberate breakages of §5 rule 8 are caught.** The prompt names at least these:
     - a factory's `SQLAFactoryBase` import pointed back at `Datastore.SQL.ObjectFactories.base`;
     - a `from Datastore import DatastoreObject` restored in a target module;
     - the moved test deleted;
     - `standin_pool.py`'s `"datastorekit.SQL.ShardedPool"` string reverted.

**05 — Rehearse through `datastorekit`, R1** (Opus).
- **What it does.** §4.1's protocol, slug prefix `dk-and-rpk-migration-05`, at 04's tree.
- **Acceptance.**
  1. §4.1's L1–L3 hold, both fingerprints equal `39809dca…`, and so equal R0's.
  2. Every manifest's `packages.datastorekit` names commit `6d8ee9ae…` and `editable: false`, and
     `packages.raypoolkit` is `null`.
  3. **L4 against R0.** The `GkSourcePolicyData` block, the stage counts and what the resume
     recomputed are each R0's.
  4. The board records **DatastoreKit's G2 as holding** (§6.2 C12).

**06 — Pin `raypoolkit` `v0.2.0`** (Sonnet). As 03, for
`raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0`.
- **Acceptance.**
  1. `pip freeze` differs from 03's by exactly one line, `raypoolkit` at `eaef0ce60e2f…`.
  2. `pip check` is clean.
  3. `package_provenance()["raypoolkit"]` gives `0.2.0`, commit `eaef0ce6…` and
     `editable: false`.
  4. The installed `raypoolkit/` equals `git archive v0.2.0 raypoolkit`, less any test package.
  5. The six roots give 04's counts.

**07 — Adopt `raypoolkit`** (Sonnet).
- **What it changes.**
  1. It deletes `RayTools/RayWorkPool.py` and `RayTools/__init__.py`. `utilities.format_time`
     stays SGK's.
  2. It rewrites the 7 imports to `from raypoolkit import RayWorkPool`, keeping the bare name
     that five `ComputeTargets/tests` modules' `ast` matches need.
  3. It changes no pool site (C7).
  4. Prose made false by this commit (C10):
     - `RunRegistry/tests/test_pipeline_adoption.py:32-33`'s comment, whose line numbers are the
       deleted copy's;
     - `CLAUDE.md`'s `RayTools/` sentence.
- **Acceptance.**
  1. `git ls-tree HEAD RayTools` is empty.
  2. **By RayPoolKit's script, at the commit.** 0 imports and 0 `import_module` calls name
     `RayTools`. The string literals left are `docs/tolerance-convergence/inventory.py:382` and
     the `prompts/` record (C10). The script finds 54 calls, every one "constructs".
  3. The six roots give 04's counts, and `black --check` is clean.
  4. The deliberate breakages of §5 rule 8 are caught. At least: one call spelled
     `raypoolkit.RayWorkPool(...)`, which the `ast`-matching tests must catch; and the import of
     one `extract_*.py` reverted.

**08 — Rehearse through both packages, R2** (Opus).
- **What it does.** §4.1's protocol, slug prefix `dk-and-rpk-migration-08`, at 07's tree.
- **Acceptance.**
  1. §4.1's L1–L3 hold, and both fingerprints equal `39809dca…`.
  2. Both `packages` entries name their tag's commit, with `editable: false`.
  3. **L4 against R1.** The `GkSourcePolicyData SUMMARY STATISTICS` block is R1's, line for line.
     This measures the change at `main.py:3247`, RayPoolKit item 5 (C8).
  4. The registry's stage and progress fields are filled as in R1. The printed lines are an
     interface, and `StageTracker` must parse the package's.
  5. The board records **RayPoolKit's G2 as holding** (C12).

**09 — Close the migration** (Sonnet).
- **What it changes.**
  1. It hands over the four inherited issues (C11), on their boards and in the index.
  2. `CLAUDE.md`'s test command (in "Repository mechanics") names all six roots, and the
     Datastore root's four groups.
  3. It takes the final measurements at `HEAD` by both packages' scripts and saves them, with the
     final counts of the six roots.
  4. It drafts, in the verification document, the G2 records for DatastoreKit's and RayPoolKit's
     extraction boards. Neither is committed there (C12).
  5. It updates `prompts/INDEX.md`, and closes the board.
- **Acceptance.**
  1. The index's count and date agree with its rows.
  2. Nothing outside `prompts/` and dated `docs/` names a deleted file as a current path.
  3. The six roots pass.

## 3. Datastores

**The campaign builds six throwaway stores and touches no other.** Each rehearsal builds two:
- `var/datastores/dk-and-rpk-migration-NN-reference.sqlite`;
- `var/datastores/dk-and-rpk-migration-NN-resumed.sqlite`;

for NN = 02, 05 and 08. Each gets a sidecar (`store create`) and a written fingerprint, as in
`a3-v2-readiness` 07.

**No prompt retires a store.** The stores stay for each review, and R0's and R1's are compared
with later ones. The user retires them after 09's review, or not.

**Before touching anything,** a prompt runs `python -m RunRegistry list` and reports anything
`running` or stale. Before each launch it also checks that no Ray process of this repository is
up, waiting by pid for a killed build's agents
(`[07-a-killed-builds-ray-agents-outlive-the-driver]`).

## 4. The interfaces between prompts

- **01 ships** `RunRegistry.RECORDED_DISTRIBUTIONS`, `RunRegistry.package_provenance()` and the
  manifest key `packages`, shaped as in §2.2. 03, 05, 06 and 08 read it.
  **Shipped as named** (prompt 01's log): no difference.
- **02 ships the rehearsal scripts** under `docs/dk-and-rpk-migration/rehearsal/` (§4.1, L3):
  `run_readers.sh <primary> <output dir>`, `stage_counts.py <run dir>...` and
  `policy_statistics.py <run dir>`.
  05 and 08 run them unchanged. A change to one is a deviation, logged and reviewed, since L4
  compares their outputs.
  **Shipped as named** (prompt 02's log), `run_readers.sh` taking an optional trailing
  `nomodels <script>`, plus three more that 05 and 08 run too: `prune_check.py <closed_store_state
  output> <primary>`, `compare_with_recorded.py <primary>... --against <run dir>...`, and
  `store_soundness.py <primary>...`, which replaces 07's (it fails since `19f07ee`).
- **02, 05 and 08 each add a section** to `docs/dk-and-rpk-migration-verification.md`, which 02
  creates. The sections are §1 R0, §2 R1 and §3 R2, each with §4.1's tables. 03, 04, 06 and 07
  record their measurements in their logs, and 09 adds §4, the close.
- **03 and 06 ship the pins** exactly as §2.2 gives them, each in `requirements.txt` and in no
  other file.
  **03 shipped as named** (prompt 03's log): the one line at `requirements.txt:17`, installed
  after a dry run that added only `datastorekit` 0.2.2, at `6d8ee9ae…`, not editable.
  **06 shipped as named** (prompt 06's log): the one line at `requirements.txt:76`, installed after a
  dry run that added only `raypoolkit` 0.2.0, at `eaef0ce6…`, not editable.
- **04 ships** `Datastore/__init__.py` and `Datastore/SQL/__init__.py` at 0 bytes, no
  `Datastore.<layer module>` anywhere outside `prompts/`, and
  `prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py`, the record of how the rewrite
  was made. 07 may reuse its pattern for its seven imports.
  **Shipped as named** (prompt 04's log), with two differences:
  - 02's `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` changes at `:30`, its one layer
    import, by the rewrite. That is the only change to S1's scripts, so "05 and 08 run them
    unchanged" (02's line above) now reads "as 04 left them".
  - `ComputeTargets/tests/test_quadsource_integral.py`'s `load_pre_commit_module()` maps the one
    layer import of the historical file it execs (`39ed7fc`), by the user's decision of 2026-10-11.
    No assertion changes.

### 4.1 The rehearsal protocol

This is `a3-v2-readiness` prompt 07's protocol
(`docs/a3-v2-readiness-verification.md` §1.1, §1.3 and §3), with only the names changed. NN is
the prompt's number.

**The launches.** Each is run from the repository root, detached, with this command verbatim but
for the store, the slug, the purpose and `--resume`:

```bash
nohup sh -c 'PYTHONPATH=. ./venv/bin/python -u docs/handover/quadsource_atol_sweep.py \
    --build --rehearsal [--resume] \
    --database var/datastores/dk-and-rpk-migration-NN-<store>.sqlite \
    --job-name dk-and-rpk-migration-NN --atol 1e-32 --rtol 1e-7 --cpus 6 \
    --register <slug> --purpose "<one line>" --campaign dk-and-rpk-migration --prompt NN; \
  echo "** driver exit status $?"' \
  > var/runs/dk-and-rpk-migration-NN-<slug>.launch.log 2>&1 &
disown
```

The four launches, in order:
1. **`reference`**, on the `reference` store, straight through to `done`.
2. **`resumed-a`**, on the `resumed` store. It is killed by `SIGTERM` in
   `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`, by `interrupt_when.py` as at 07.
3. **`resumed-b`**, with `--resume`. It is killed in `CALCULATE QUADRATIC SOURCE INTEGRALS`.
4. **`resumed-c`**, with `--resume`, run to `done`.

**After each launch** the agent makes **one** check:
- the manifest exists;
- `status.json` says `running`, with a pid that answers `kill -0`;
- `RunRegistry list` shows the run.

Then it ends its turn (`CLAUDE.md`, run registry rules 3–4). The orchestrator waits once per
launch, on the run's liveness, and resumes the agent when the run is no longer alive. Neither
polls.

**What must hold.**
- **L1 — every launch ends as recorded.** `reference` and `resumed-c` end `done`/0;
  `resumed-a` and `resumed-b` end `killed`/143, each with its fingerprint. Each driver's exit
  status agrees.
- **L2 — the resume is the reference.**
  - `compare_fingerprints` of the two stores is empty, and each is format 6,
    `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes equal.
  - The resume recomputed nothing it had stored, and the prune at open removed only unvalidated
    rows.
  - Both stores pass the read-only integrity checks of 07 §3: `closed_store_state.py`,
    `prune_serials.py`, `replicated_serial_census.py` and `store_soundness.py`.
- **L3 — every reader, read-only.** The six `extract_*.py` each run once on the closed `resumed`
  store, with `--models LambdaCDM`, writing under `var/dk-and-rpk-migration-NN/extract/`. Each
  exits 0, none raises `ReadOnlyMiss` or `ReadOnlyWrite`, and every store file is byte-identical
  before and after.
  - `docs/a3-v2-readiness/rehearsal/run_readers.sh` and `stage_counts_07.py` hardcode 07's store
    and run ids.
  - So 02 writes copies, parameterised by store, output directory and run ids, under
    `docs/dk-and-rpk-migration/rehearsal/`, and 05 and 08 reuse them unchanged.
  - 07's originals are records, and are left as they are.
- **L4 — equal to the previous rehearsal.** Against the campaign's previous rehearsal (none for
  R0), these are the same:
  - the reference run's `GkSourcePolicyData SUMMARY STATISTICS` block;
  - each stage's work-item counts;
  - the rows the resume pruned and recomputed.

  If L1–L3 pass and L4 differs, that is a finding to explain, not an automatic failure. The
  fingerprint is the acceptance.

**If a fingerprint misses**, the prompt stops. It records both fingerprints and
`compare_fingerprints`, and does not change code. R0's miss belongs to SGK (§2.2, 02). R1's or
R2's is reported to the package's repository as a defect (§5 rule 11).

### 4.2 Rollback

Each prompt is one commit, and is reverted by `git revert`.
- **The venv is shared**, so reverting 03 or 06 also needs
  `./venv/bin/pip uninstall -y <package>`.
- **Reverting 04 or 07 alone needs nothing in the venv.** An installed, unimported package is
  inert.
- **The campaign runs in the main checkout**, not a worktree. A worktree here has no `venv` of
  its own and is given a symlink to the main checkout's, so a pin installed from one lands in the
  other.

### 4.3 What `CLAUDE.md` says after 04 and 07

The section "The datastore layer is frozen — it is moving to DatastoreKit" is replaced by a short
section saying:
- the layer is `datastorekit`, pinned in `requirements.txt`, and the work queue is `raypoolkit`
  (from 07);
- a defect in either is reported to its repository, fixed there in a release, and reaches SGK by
  moving the pin in a commit of its own. It is never patched here, and never worked around;
- an editable install is never used for a production run;
- the factories under `Datastore/SQL/ObjectFactories/`, `config/` and every caller are SGK's.

04 writes it for DatastoreKit, keeping the `RayTools/` sentence true under RayPoolKit's U5 and
U12. 07 completes it.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit,** and `docs/OPEN_ISSUES.md`
   whenever a §3 or §4 entry moves, in the same commit.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue on this
   board.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. 04's
   import rewrite of staying tests is named, and so are C3 and C4.
8. **Deliberate breakage.** Each prompt names mutations its tests must catch. The log records each
   as a diff, exactly as applied, so that the orchestrator can replay it with `git apply`.
   Mutations are never committed.
9. **Nothing the code writes changes.** The fingerprint is the measurement, and a prompt that
   cannot keep it stops.
10. **No test needs Ray, and no test opens anything under `var/`.** A rehearsal's reads of its own
    stores, and 04's read-only fingerprint of R0's store, are verification steps, not tests.
11. **A defect in a package is not patched or worked around in SGK.** The prompt stops and
    records it. The user reports it to that package's repository, where it is fixed in a new
    release, and a new pin prompt moves SGK to it.
12. **No change is made in DatastoreKit or RayPoolKit.** Their repositories are read, and their
    measuring scripts run from their roots with their own venvs. Nothing there is edited or
    committed.
13. **Pins.** Each pin is a tag, in a commit of its own, never an editable install, and SGK keeps
    its own `ray` and `SQLAlchemy` pins.
14. **Long-running jobs** follow `CLAUDE.md`'s run-registry rules: discovery first, detached
    launch, one check, end the turn.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**, with every count against §0.2's;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-10-11), and not reopened here

| | Decided | Rejected |
|---|---|---|
| **U1** — the order | **DatastoreKit first**, then RayPoolKit, each followed by its own rehearsal | RayPoolKit first: it edits `test_layer_is_generic.py` only for 04 to delete it. Both in one swap: one rehearsal after both cannot say which package caused a miss |
| **U2** — the rehearsal | **§4.1's full protocol** (prompt 07's) at R0, R1 and R2: reference, killed-and-resumed build, readers | A straight-through build only, the checklist's minimum. It leaves the resume, prune and reconcile paths and the read-only opens, all package code now, unexercised through the package |
| **U3** — the factories | **They stay** in `Datastore/SQL/ObjectFactories/`, and `Datastore/__init__.py` and `Datastore/SQL/__init__.py` become **empty files**. Nothing about their registration changes: `config/datastore.py` passes the classes to `ShardedPool(factories=…)` as today | Namespace packages, which are implicit and merge silently with any other `Datastore` on `sys.path`. A move, which renames 98 imports and 5 paths and changes no behaviour |
| **U4** — provenance | **Yes, first** (01): each run's manifest records the installed packages, so R0–R2 each say what they ran | Leaving a run's package versions implied by `requirements.txt` at its `HEAD` |

The user asked, at U3, why the factories' location was a question at all. Once the layer is gone,
only the classes' registration is needed. The answer: only Python's packaging is at issue. The two
deleted files are the package markers of the folders that hold the factories, so something must
make `Datastore` and `Datastore.SQL` importable. Whether a package named `Datastore` that holds no
datastore should later be renamed is a naming question, left for another day.

### 6.2 Choices of method made in writing this plan

These are the orchestrator's, and are recorded so that the user can reverse any of them.
- **C1 — the branch.** `handover-remedial`, where A3 v2 runs and where both checklists were
  measured. The campaign runs in the main checkout (§4.2).
- **C2 — the rewrite rule.** Every layer import outside `prompts/` is rewritten, by module path
  only.
  - This takes in all 17 `docs/` statements. That covers the three a test or the rehearsal
    reaches:
    - `docs/handover/quadsource_atol_sweep.py:693`;
    - `docs/source-remediation-verification/analyse_greens_and_source.py:35`;
    - `docs/source-remediation-verification/run_quadsource_integrals.py:41`.

    It also covers the four live probes, among them `store_soundness.py`, which §4.1 runs.
  - The six files already failing on `_factories` are rewritten the same way and still fail. A
    name that left the layer before this campaign is not restored here. 04 opens an issue for
    them.
  - So DatastoreKit's acceptance reads "0 imports outside `prompts/`", with no exceptions to
    track.
- **C3 — `test_inventory_retired.py`.**
  - **It keeps** its SGK assertions: no factory and no `config.sharding` has the old service, and
    the `git grep` over SGK's `*.py` finds no retired name.
  - **Its two layer assertions read the installed package.** The actor's source comes from
    `importlib.util.find_spec("datastorekit.SQL.Datastore").origin`, and the pool from
    `datastorekit.SQL.ShardedPool`. So SGK still checks that the release it pins lacks the old
    service.
  - **It loses** `test_the_excluded_file_holds_only_the_new_service_helper` and the grep's
    exclusion of `test_store_inventory.py`, whose subject is deleted.
- **C4 — the unported test** moves into `Datastore/tests/test_inventory_report.py`, the module of
  SGK's `tools.inventory_report`, which it tests.
- **C5 — `schema_description.py`** is deleted. Its only importer, `test_schema_builder`, is
  deleted, and nothing in `docs/` imports it.
- **C6 — `standin_pool.py`** changes its import strings only. It does not stand in
  `_ray_is_running` or `ray.kill` as the package's stand-in does. Under `v0.2.2` its pools close
  killing nothing (DatastoreKit `v0.2.2` addendum, "SGK's own stand-in"), and the tests pass
  either way. The only test using it drives `main.py`'s policy queue, not the pool's close.
- **C7 — the 44 lookup-only sites** keep `store_handler=None, persist_handler=None`, so 07's diff
  is the imports alone (RayPoolKit's advice).
- **C8 — `main.py:3247`** is measured by L4 at 08, from the rehearsals' own `stdout.log`, at no
  extra run.
- **C9 — the test roots.** All six, at every prompt, against stated counts.
- **C10 — stale prose.**
  - A statement the prompt's own commit makes false is corrected in that commit:
    - `CLAUDE.md`'s freeze section and `:62` (04);
    - the `RayTools/` sentence (04, then 07);
    - `RunRegistry/stores.py:5` (04);
    - `test_pipeline_adoption.py:32-33` (07);
    - the test command (09).
  - Dated `docs/` audits and verification records, and `prompts/`, are left as the record of the
    tree they describe, as both checklists advise. `docs/tolerance-convergence/inventory.py:382`
    keeps `"RayTools"` in `PRODUCTION_TREES`, since `production_files()` skips a tree that does
    not exist.
- **C11 — the four inherited issues** are handed over at 09.
  - **Three are wholly layer code**, and close on their boards' §4 as transferred to
    DatastoreKit, with a pointer to its `docs/OPEN_ISSUES.md` §1.2:
    - `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]` and
      `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]` (`a3-v2-readiness`);
    - `[00-the-inventory-cannot-see-a-serial-split]` (`datastore-integrity`).
  - **`[01-cross-filesystem-move-advice-says-delete-by-hand]`** (`store-retirement`) has an SGK
    half: `store move` passes the advice on for a registered store. It is narrowed to that half,
    or transferred whole if 09 finds no SGK code in it.
- **C12 — the gates.** This board records each G2 when it holds (05, 08). Recording G2 on
  DatastoreKit's and RayPoolKit's extraction boards is a records commit in each of those
  repositories. 09 drafts the text, and the user makes the commits there, or has a session in
  that repository make them.
- **C13 — provenance's reach.** The run manifest only (01, "Out").
- **C14 — a package defect** stops the prompt that meets it (rule 11). The campaign waits for a
  new release, and resumes with a new pin prompt.

### 6.3 Left for the user at the close

- Retiring the six rehearsal stores (§3).
- The two G2 records in DatastoreKit and RayPoolKit (C12).
- Whether to repair the six `docs/` files that fail on `_factories` (C2).
- Whether to rename the `Datastore` package later (U3).

## 7. Baselines

Re-measure every suite at `HEAD`, clean, before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

Run the Datastore suite in four groups of modules (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`,
`[s-z]`), each in the foreground.

**Last measured at `b510bc9`** (§0.2):
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped);
- RunRegistry 280, ComputeTargets 693;
- Datastore 739 (163 + 185 + 200 + 191).

**Under load, four tests are known to fail on wall clock or chance:**
- `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which did at `b510bc9` and passed alone;
- `test_background_tau.test_build_cost`;
- `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
- LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

Re-run a failure alone before calling it real.
