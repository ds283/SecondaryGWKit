# dk-and-rpk-migration campaign — implementation state

**Last updated:** 2026-10-11 · **Status: LIVE — 9 of 9 prompts written, 01–08 landed. R0, R1 and R2 hold; SGK's layer is `datastorekit` 0.2.2 and its work queue `raypoolkit` 0.2.0, both pinned and installed, and both packages' G2 hold; the close (09) is next.**
The plan is [`README.md`](README.md), written on 2026-10-11 at `b510bc9` with the user's decisions
U1–U4, and committed at `4532188`. The user reviewed it the same day, and 01–04 were written
against it. 05–09 were written the same day, after 04's review (README §2.1).

The campaign adopts `datastorekit` `v0.2.2` and `raypoolkit` `v0.2.0`, in that order. A
fingerprint rehearsal comes before the first swap (R0) and after each swap (R1, R2). Each
package's gate G2 is recorded here when it holds.

**Campaign:** [`README.md`](README.md) ·
**Briefs:** [`sgk-datastorekit-planning-brief.md`](sgk-datastorekit-planning-brief.md),
[`sgk-raypoolkit-planning-brief.md`](sgk-raypoolkit-planning-brief.md) ·
**Code:** `RunRegistry/__init__.py`, `requirements.txt`, `Datastore/`, `RayTools/`, and every layer
import outside `prompts/` ·
**Verification:** `docs/dk-and-rpk-migration-verification.md` (created by 02) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md), a section opened with this board's
first issue

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-11 (README §6.1):**

| Decision | Decided |
|---|---|
| **U1** — the order | DatastoreKit first, then RayPoolKit, each followed by its own rehearsal |
| **U2** — the rehearsal | prompt 07's full protocol (README §4.1) at R0, R1 and R2 |
| **U3** — the factories | stay in `Datastore/SQL/ObjectFactories/`; `Datastore/__init__.py` and `Datastore/SQL/__init__.py` become empty files |
| **U4** — provenance | each run's manifest records the installed packages, from prompt 01 |

The orchestrator's choices of method, C1–C14, are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Record the installed packages in each manifest](01-record-the-installed-packages-in-each-manifest.md) | **U4** | Sonnet | ✍️ 2026-10-11 | ✅ 2026-10-11 | `f54f1c5` | [log](logs/01-record-the-installed-packages-in-each-manifest.md) |
| 02 | [Rehearse the start tree (R0)](02-rehearse-the-start-tree.md) | §0.1's gap `53d4e18` → `b510bc9` | Opus | ✍️ 2026-10-11 | ✅ 2026-10-11 | `7650102` | [log](logs/02-rehearse-the-start-tree.md) |
| 03 | [Pin `datastorekit` `v0.2.2`](03-pin-datastorekit-v0.2.2.md) | the DatastoreKit pin | Sonnet | ✍️ 2026-10-11 | ✅ 2026-10-11 | `9ec603a` | [log](logs/03-pin-datastorekit-v0.2.2.md) |
| 04 | [Adopt `datastorekit`](04-adopt-datastorekit.md) | the DatastoreKit swap; **U3** | Opus | ✍️ 2026-10-11 | ✅ 2026-10-11 | `0c9e2cf` | [log](logs/04-adopt-datastorekit.md) |
| 05 | [Rehearse through `datastorekit` (R1)](05-rehearse-through-datastorekit.md) | **DatastoreKit G2** | Opus | ✍️ 2026-10-11 | ✅ 2026-10-11 | `f3f0879` | [log](logs/05-rehearse-through-datastorekit.md) |
| 06 | [Pin `raypoolkit` `v0.2.0`](06-pin-raypoolkit-v0.2.0.md) | the RayPoolKit pin | Sonnet | ✍️ 2026-10-11 | ✅ 2026-10-11 | `b6cc119` | [log](logs/06-pin-raypoolkit-v0.2.0.md) |
| 07 | [Adopt `raypoolkit`](07-adopt-raypoolkit.md) | the RayPoolKit swap | Sonnet | ✍️ 2026-10-11 | ✅ 2026-10-11 | `643c3b2` | [log](logs/07-adopt-raypoolkit.md) |
| 08 | [Rehearse through both packages (R2)](08-rehearse-through-both-packages.md) | **RayPoolKit G2**; `main.py:3247` | Opus | ✍️ 2026-10-11 | ✅ 2026-10-11 | `f539015` | [log](logs/08-rehearse-through-both-packages.md) |
| 09 | [Close the migration](09-close-the-migration.md) | the inherited issues; the close | Sonnet | ✍️ 2026-10-11 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-11, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-11,
used for 02); [`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-11, used for 03);
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-11, used for 04);
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) (2026-10-11, used for 05);
[`orchestrator/prompt-06.md`](orchestrator/prompt-06.md) (2026-10-11, used for 06);
[`orchestrator/prompt-07.md`](orchestrator/prompt-07.md) (2026-10-11, used for 07);
[`orchestrator/prompt-08.md`](orchestrator/prompt-08.md) (2026-10-11, used for 08).

**Written in two batches (README §2.1).**
- **01–04** after the user's review of the plan.
- **05–09** after 04's review, whose log fixes the counts and names they cite. **09 was written
  with them at the user's request**, not after 08's review as planned: its orchestrator note,
  written after that review, carries 05–08's results to it and governs where they differ.

No prompt waits on another's *scope*. Each charter is fixed in README §2.2.

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **P1** | 01 | `RunRegistry.package_provenance()` and `RECORDED_DISTRIBUTIONS` | ✅ |
| **P2** | 01 | the manifest's `packages`, written by `begin()` | ✅ |
| **P3** | 01 | `TestPackageProvenance` and the manifest test | ✅ |
| **S1** | 02 | `docs/dk-and-rpk-migration/rehearsal/`: `run_readers.sh`, `stage_counts.py`, `policy_statistics.py` (with `prune_check.py`, `compare_with_recorded.py` and `store_soundness.py` beside them) | ✅ |
| **L1**–**L4** (R0) | 02 | README §4.1 at the start tree; L4 recorded as the baseline | ✅ |
| **K1** | 03 | the `datastorekit` pin in `requirements.txt` | ✅ |
| **K2** | 03 | the venv installed from it | ✅ |
| **K3** | 03 | the proof: one distribution added, at `6d8ee9ae…`, the tag's files, not editable | ✅ |
| **A1** | 04 | the 15 layer files deleted, two `__init__.py` emptied | ✅ |
| **A2** | 04 | every layer import outside `prompts/` rewritten, by `scripts/rewrite_layer_imports.py` | ✅ |
| **A3** | 04 | `Datastore/tests/`: 26 modules and `schema_description.py` deleted, one test moved, `test_inventory_retired` and the two import strings changed | ✅ |
| **A4** | 04 | `CLAUDE.md`'s layer section, `RunRegistry/stores.py:5` | ✅ |
| **A5** | 04 | `[04-six-docs-probes-import-factories-from-the-layer]` opened | ✅ |

| **L1**–**L4** (R1) | 05 | README §4.1 through `datastorekit`; L4 against R0 | ✅ |
| **G-DK** | 05 | DatastoreKit's G2 recorded as holding (§2) | ✅ |
| **Q1** | 06 | the `raypoolkit` pin in `requirements.txt` | ✅ |
| **Q2** | 06 | the venv installed from it, after a dry run | ✅ |
| **Q3** | 06 | the proof: one distribution added, at `eaef0ce…`, the tag's files, not editable | ✅ |
| **B1** | 07 | `RayTools/` deleted | ✅ |
| **B2** | 07 | the seven `RayTools` imports rewritten to `from raypoolkit import RayWorkPool` | ✅ |
| **B3** | 07 | `CLAUDE.md`'s package section completed; `test_pipeline_adoption.py`'s comment | ✅ |
| **L1**–**L4** (R2) | 08 | README §4.1 through both packages; L4 against R1, with `main.py:3247` (C8) | ✅ |
| **G-RPK** | 08 | RayPoolKit's G2 recorded as holding (§2) | ✅ |
| **Z1** | 09 | the four inherited issues handed to DatastoreKit (C11) | ⬜ |
| **Z2** | 09 | the prose of `[04-comments-name-the-layers-old-module-paths]` rewritten; the issue closed | ⬜ |
| **Z3** | 09 | `CLAUDE.md`'s test command names all six roots and Datastore's four groups | ⬜ |
| **Z4** | 09 | the final measurements at `HEAD` | ⬜ |
| **Z5** | 09 | verification §4, with the two G2 records drafted | ⬜ |
| **Z6** | 09 | the board, README, `prompts/INDEX.md` and the index closed | ⬜ |
| **Z7** | 09 | only if the user chooses it: the six `docs/` probes repaired (README §6.3) | ⬜ |

The measurement the plan was written against is [`measure/`](measure/): both packages' scripts at
`b510bc9`, each equal to its checklist's appendix byte for byte; the six roots' counts; the
Datastore root by module; and the venv's `pip freeze` (README §0.2).

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ✅ **holds** (2026-10-11): both stores `39809dca…`, all 21 classes, equal to each other and to 07's and 09's recorded fingerprints (verification §1.3.4) |
| DatastoreKit G2 | the 17 files are gone (two emptied, U3), the pin is in place and installed, the suites pass, and R1 reproduces `39809dca…` | 04, 05 | ✅ **holds** (2026-10-11): the deletions at `0c9e2cf` (04); the pin at `9ec603a` (03), installed and recorded in every R1 manifest; the six roots pass at `6861a9f` (05); R1's runs `dk-and-rpk-migration-05-reference-20261011T192802` and `dk-and-rpk-migration-05-resumed-c-20261011T195044`, both `done`, both `39809dca…`, equal to R0's, 07's and 09's (verification §2.10); recorded by 05 in `f3f0879` |
| RayPoolKit G2 | `RayTools/` is gone, 0 imports, the pin is installed, the six roots pass, all 54 sites construct, and R2 reproduces `39809dca…` | 07, 08 | ✅ **holds** (2026-10-11): `RayTools/` gone at `643c3b2` (07); 0 imports, 0 `import_module` calls and 54 sites `constructs` by RayPoolKit's script at `5631f0e` (08, `measure/out/08-raypoolkit-measure_client_pools.md`); the pin at `b6cc119` (06), installed and recorded in every R2 manifest; the six roots pass at `5631f0e` (08); R2's runs `dk-and-rpk-migration-08-reference-20261011T224347` and `dk-and-rpk-migration-08-resumed-c-20261011T230717`, both `done`, both `39809dca…`, equal to R1's, R0's, 07's and 09's (verification §3.10); recorded by 08 in `f539015` |

Recording each gate on its package's own extraction board is a commit in that repository, not
here (README §6.2 C12).

## 3. Active and unresolved issues

- **[02-the-policy-statistics-block-orders-its-lines-by-completion]** *(opened 2026-10-11 by prompt
  02; measured; bears on 05's and 08's L4)*
  - **The defect.** `main.py`'s `** GkSourcePolicyData SUMMARY STATISTICS` block
    (`main.py:3265-3278`) prints nested plain dictionaries in insertion order. They are filled by
    `postprocess_GkSourcePolicy` (`main.py:2871-2893`), the `APPLY GKSOURCE POLICIES` queue's
    `post_handler`, which `RayWorkPool` calls in Ray's completion order. So the order of the quality
    lines under a type, and in principle of the types, depends on which object completed first.
  - **Measured.** R0's reference prints `minimal` before `complete` under `>> mixed`; `datastore-generic`
    09's and `a3-v2-readiness` 07's print `complete` first; R0's own `resumed-b`, on the same tree,
    prints `complete` first and `resumed-c` `minimal` first. Every line's content and every count is
    the same, and the blocks sorted are equal (verification §1.5.2).
  - **Impact.** README §4.1's L4 and 08's acceptance 3 compare the block "line for line", and 08's
    measures `main.py:3247` (C8) by it. A comparison of the raw blocks can then fail on order alone.
    The store, the fingerprint and the counts are unaffected.
  - **Next step.** Not assigned. For 05 and 08, the comparison is of the counts: compare the blocks
    line for line and, where only the order differs, as sorted lines, saying so. A fix in `main.py`
    (sorted keys) would change the printed order once, and is the user's to decide.
- **[02-07s-store-soundness-calls-read-inventory-without-its-registry]** *(opened 2026-10-11 by
  prompt 02; measured; verification tool only)*
  - **The defect.** `docs/a3-v2-readiness/rehearsal/store_soundness.py` calls
    `read_inventory(primary)`. Since `19f07ee` ("Give the datastore layer its registries instead of
    importing them") `read_inventory` takes the registry too, so the script raises `TypeError:
    read_inventory() missing 1 required positional argument: 'factories'` after printing the first
    store's pragmas. README §4.1's L2 names it, and 04 §3.3 imports it (an import that succeeds).
  - **Measured.** On R0's two stores at `a5392fd` (verification §1.3.5). It was not changed: it is
    07's record. R0 ran `docs/dk-and-rpk-migration/rehearsal/store_soundness.py`, a copy that passes
    `config.datastore.factories` as `RunRegistry.stores.fingerprint_store` does, and reads the shards
    from the primary's `shards` table.
  - **Impact.** None on a build or a store. 05 and 08 run the copy; 04's import rewrite reaches it as
    it reaches 07's.
  - **Next step.** Not assigned. 07's script stays a record of `dbd8a51`; this campaign's copy is the
    one to run.

- **[04-six-docs-probes-import-factories-from-the-layer]** *(opened 2026-10-11 by prompt 04;
  measured; `docs/` probes only; README §6.3)*
  - **The defect.** Six `docs/` files import `_factories` from the layer's `SQL/Datastore.py`, a
    name that left it at `19f07ee` ("Give the datastore layer its registries instead of importing
    them"): `docs/datastore-integrity-audit/fk_sweep_probe.py:8`, `lookup_keys/common.py:18`,
    `lookup_keys/probe_03_column_existence.py:25`, `tq_fk_probe.py:8`;
    `docs/tolerance-convergence/inventory.py:196` and `order_audit.py:1253`. Six more probes fail
    through `lookup_keys/common.py` (`probe_01`, `01b`, `04`, `04b`, `05`, `06`). DatastoreKit's
    checklist item 3 counts ten such statements; the other four are in `prompts/`, which is not
    rewritten.
  - **Measured.** Before 04 each raised `ImportError: cannot import name '_factories' from
    'Datastore.SQL.Datastore'`; after it, the same from `'datastorekit.SQL.Datastore'` (the
    package's module). Five on a run with no arguments; `order_audit.py` by loading it by path and
    calling `d3_tables()`, whose nested import its `main` reaches only at stage 6 (log 04, A5).
  - **Impact.** None on a build, a store or a test: no test or rehearsal reaches them. They are
    audit probes that cannot run.
  - **Next step.** The fix is `from config.datastore import factories as _factories`. Whether to
    repair them is the user's decision at the close (README §6.3); 04 rewrote their module paths
    only (C2).
- **[04-comments-name-the-layers-old-module-paths]** *(opened 2026-10-11 by prompt 04; measured;
  prose only)*
  - **The defect.** 23 comment and docstring lines outside `prompts/` and `docs/` name a layer
    module or file by its pre-04 path. 04's rewrite reaches statements, string literals and
    `import_module` calls only, and its prompt's §2.4.3 changes no other prose. They are:
    - ``Datastore.SQL.Datastore._build_schema`` / ``._factories`` in docstrings:
      `ComputeTargets/tests/test_cosmology_representation_key.py:32`, `test_gauss_order_key.py:60`,
      `test_numeric_break_point_key.py:31`, `test_run_identity.py:41`;
      `Datastore/tests/test_database_errors_reach_the_caller.py:37`,
      `test_exact_identity_lookups.py:29`, `test_integral_read_batch.py:26`,
      `test_quadsource_policy_key.py:28`, `test_read_path_mechanics.py:25`,
      `test_value_reads_parent_key.py:27`, `test_wkb_numeric_parent_key.py:35` (`._factories` has
      been wrong since `19f07ee`);
    - `Datastore.store_inventory…`, `Datastore.shard_paths…`, `Datastore.SQL.schema…`:
      `RunRegistry/stores.py:107`, `:114`, `:1155`;
      `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py:7`; `config/datastore.py:18`;
      `extract_common.py:157`; `main.py:283`; `tools/inventory_report.py:5`;
    - file paths: `config/datastore.py:4-5`; `RunRegistry/stores.py:209`;
      `Datastore/tests/real_store_fixtures.py:403`.
  - **Measured.** By a regex over every file in the index outside `prompts/` and `docs/`, skipping
    import statements and `import_module` calls (log 04, "Observations not acted on"); the same 23
    lines the orchestrator's note listed at `41d9bc0`.
  - **Impact.** None on code. A reader is pointed at a file that no longer exists in SGK.
  - **Next step.** 09's acceptance 2 ("nothing outside `prompts/` and dated `docs/` names a deleted
    file as a current path").

## 4. Resolved issues

None yet.

## 5. After each prompt

Each prompt adds `### 5.N After prompt NN (<date>)`: what landed, the counts against README §7,
and the state handed to the next prompt. The orchestrator's review of each follows it.

### 5.1 After prompt 01 (2026-10-11)

**What landed (`f54f1c5`).** `RunRegistry.RECORDED_DISTRIBUTIONS`,
`RunRegistry.package_provenance()` and the manifest key `packages`, written by `begin()` after
`git_provenance()`; `TestPackageProvenance` (five tests) and
`TestManifest.test_manifest_records_the_installed_packages`. `begin()`'s docstring no longer says
that only one manifest field lies outside README §0: it names `packages` as the second
(`STRUCTURALLY REQUIRED`). In this venv `package_provenance()` is
`{"datastorekit": null, "raypoolkit": null}`, shown by a hand-made manifest in a temporary root
outside the repository. Nothing under `var/` was touched.

**Counts against README §7.** RunRegistry 286 (280 + 6), AdaptiveLevin 32, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), ComputeTargets 693, Datastore 739 (163 + 185 + 200 + 191). All OK
except one ComputeTargets test, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, a
load-sensitive one that also passed and failed when re-run alone, and did so on an export of the
tree before this prompt too. Breakages (i)-(v) each fail the named tests (log).

**Handed to 02 and on.** Every manifest written from now on says what was installed; R0's says
neither package. No §3 entry was opened, and `docs/OPEN_ISSUES.md` is unchanged at 104.

**Orchestrator review of prompt 01 (2026-10-11).** Dispatched from `6edb2f9` to one Sonnet
subagent. Reviewed against its commit `f54f1c5`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–3 are met. No stop condition fired.
- **Scope (check 1).** Five files: `RunRegistry/__init__.py`,
  `RunRegistry/tests/test_run_registry.py`, the log, this board, and one line under README §4's
  "01 ships". Nothing else moved, and
  `docs/OPEN_ISSUES.md` is untouched at 104.
- **The function, by reading (check 2).** `RunRegistry` uses `import importlib.metadata`. Only
  `PackageNotFoundError` gives `None`. `url`, `vcs_info.commit_id`, `vcs_info.requested_revision`
  and `dir_info.editable` are each read from their PEP 610 place. A `direct_url.json` that is
  absent, does not parse, is not an object, or has a non-object `vcs_info` or `dir_info` gives the
  defaults. `editable` is `dir_info.get("editable") is True`, a stricter reading than "true" that
  the log records as an IMPLEMENTATION CHOICE. The docstring says what the prompt asks.
- **The odd cases (check 3).** No call raised. Tried: a list `vcs_info` with a string `dir_info`;
  `editable: false`; a `vcs_info` without `requested_revision`; `editable: "yes"` (gives `False`);
  and `read_text` raising `OSError`, `PermissionError` or `UnicodeDecodeError`.
- **The tests (check 4).** The stand-in replaces `RunRegistry.importlib.metadata.distribution`
  (`mock.patch.object`), which is what the function calls. Test 2's commit is 40 hex, and every
  field is asserted as a whole dict. Test 4 has seven sub-cases, each asserting all four defaults.
  Test 6 asserts the key set and equality with `package_provenance()`, returned and on disk. It
  asserts nothing about what this venv holds, so it passes unchanged after 03 and 06.
- **The probe without a mock (check 5).** A scratch `datastorekit-0.2.2.dist-info/` was put first
  on `sys.path`, in a fresh `-I` interpreter. With a PEP 610 VCS record at `v0.2.2`, commit
  `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`, the result gave all five fields, with
  `editable: False` and `raypoolkit: None`. An editable record gave `editable: True`, and with the
  file removed it gave the defaults. Each time, a `begin()` into a temporary root wrote the same
  `packages`. `var/runs/` gained nothing.
- **`ast` against the pre-images (check 6).** In `RunRegistry/__init__.py`, every function and
  class but `begin` dumps identically with docstrings stripped. The module gains the import, the
  tuple and `package_provenance`. `begin`'s body differs by the one assignment, between
  `manifest.update(git_provenance())` and `write_json_atomic`. The docstring change is the
  "but one" → "but two" correction and a paragraph naming `packages`, as the note asked. In the
  tests, every existing method dumps identically.
- **The pins (check 7).** Replayed in `bash` from the log's diffs. Each applied cleanly, failed
  exactly the agent's reported tests, and reverted to a clean tree:
  - (i) fails test 3 alone;
  - (ii) fails six of test 4's seven sub-cases (the seventh has an object at the top level);
  - (iii) fails test 6;
  - (iv) fails tests 1, 2 and 5;
  - (v) fails test 6, on the key set (`'raypoolkit'` missing).
- **Suites (check 8), at `f54f1c5`, load 8–19.**
  - Five roots pass: AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
    RunRegistry 286, Datastore 739 (163 + 185 + 200 + 191).
  - ComputeTargets ran 693 with the README §7 wall-clock flake,
    `test_tk_wkb_phase.TestCost.test_wall_time_per_object`. Alone at load 16–19 it passed twice
    and failed once, at 0.056 s against 0.05 s. The agent saw the same on an export of the tree
    before the prompt, and the test touches nothing 01 changed.
  - `black --check` is clean on both files.
- **Residue fixed here:** "this commit" → `f54f1c5` on this board and in the log's header, and the
  orchestrator-notes list above. **Handed on:** nothing. 02's manifests will record
  `{"datastorekit": null, "raypoolkit": null}` (its acceptance 2). From 03 on, the record is read
  from a real VCS install for the first time. 03's acceptance 3 checks it against the tag.

### 5.2 After prompt 02 (2026-10-11)

**What landed (`7650102`).** R0: README §4.1's protocol at `a5392fd`, whose code is `f54f1c5`'s and
whose layer is `b510bc9`'s. Four registered launches of `quadsource_atol_sweep.py --build
--rehearsal`, the readers and the checks, recorded in
[`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md) §1.
S1's scripts, and three more, are in `docs/dk-and-rpk-migration/rehearsal/`, with the small outputs
under names carrying `02`. No production code and no test changed.

**R0 holds.**
- `reference` `done`/0 (8 m 0 s); `resumed-a` `killed`/143 in the `GkSource` stage; `resumed-b`
  `killed`/143 in the `QuadSourceIntegral` stage; `resumed-c` `done`/0. Each driver's exit status
  agrees, each has its fingerprint, and every manifest has `git_dirty: false`, `git_head`
  `a5392fd`, and `packages` `{"datastorekit": null, "raypoolkit": null}`.
- **Both stores are `39809dca…`**, format 6, all 21 classes equal. `compare_fingerprints` is empty
  between them, against each sidecar, and against the fingerprints recorded by 07's reference and
  09's rehearsal. So `fee1fbf`, `4c34f13` and `086c81a` change nothing a build writes.
- The resume pruned the 43 `GkSource` rows `resumed-a` left unvalidated (all 43 serials absent),
  recomputed 215 = 291 − 119 + 43, and `resumed-c` computed 532 = 864 − 332 `QuadSourceIntegral`
  rows. Nothing stored was recomputed; nothing is duplicated.
- The census, `integrity_check`, `foreign_key_check`, the version row and the inventory are clean on
  both stores. The guard refused, naming the five files, and changed none.
- The six readers and `extract_Gk_data.py` without `--models` exit 0, refuse nothing, and leave
  every store file byte-identical. `extract_tensor_source_data.py` exits 0 (07a's fix holds).
- No Ray process of this repository outlived either killed driver (07's finding did not recur, at
  loads 7–17).

**L4, the baseline for 05 and 08** (verification §1.5): the statistics block (291; 109 numeric, 26
mixed of which 25 complete and 1 minimal, 156 WKB); the stage-count table; the prune and recompute
table; the stage times. R0's block equals 09's but for the order of two lines, which follows Ray's
completion order. Opened `[02-the-policy-statistics-block-orders-its-lines-by-completion]`.

**Counts against README §7**, unchanged: AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1
skipped), RunRegistry 286, ComputeTargets 693, Datastore 739 (163 + 185 + 200 + 191), all OK.

**Opened:** the statistics-order issue, and
`[02-07s-store-soundness-calls-read-inventory-without-its-registry]`. `docs/OPEN_ISSUES.md` gains
§1.20 and goes from 104 to 106.

**Handed to 03 and on.** R0's stores and run directories, for 04's read-only fingerprint and for
R1's comparison. 05 and 08 run S1's scripts unchanged, with `prune_check.py`,
`compare_with_recorded.py` (`--against` R0's `reference` run directory) and this campaign's
`store_soundness.py`, not 07's. The interruption points used here are recorded with the reason for
each; R1 should choose its own from its own reference, as R0 did.

**Orchestrator review of prompt 02 (2026-10-11).** Dispatched from `a5392fd` to one Opus subagent,
resumed with `SendMessage` after each of the four launches, with one wait per launch on the run's
liveness. Reviewed against its commit `7650102`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–6 are met. No stop condition fired. **R0 holds.**
- **Scope (check 1).** 27 files: the verification record, `docs/dk-and-rpk-migration/rehearsal/`
  (six scripts, sixteen outputs), the log, this board, `docs/OPEN_ISSUES.md` and one statement
  under README §4's "02 ships". Only the board, README and the index are modified. Nothing else
  moved.
- **The runs (check 2).** Read from each run directory. `reference` and `resumed-c` are `done`/0;
  `resumed-a` and `resumed-b` are `killed`/143, each with its fingerprint (`eaf590af…`,
  `fc02616e…`) and no `fingerprint_error`. Each launch log ends with the matching driver exit
  status. Each killed `stderr.log` holds the handler's two lines and nothing else: no traceback, no
  `SIGTERM received` banner. `resumed-b`'s second line says `SystemExit(1)`, as at 07. Each
  interrupter log has one `sent SIGTERM`: 14:41:42 after a 10 s hold on the `GkSource` banner, and
  14:44:53 after a 70 s hold on the QSI banner. The handler's lines are 14:41:42 and 14:44:54. No
  process the agent started is alive, and no Ray process of this checkout.
- **The manifests (check 3).** All four: `git_head` `a5392fd`, `git_dirty: false`, `packages`
  `{"datastorekit": null, "raypoolkit": null}`, one `script_sha256` (`d745783e…`), `--build
  --rehearsal` on all, `--resume` on `resumed-b` and `-c` only, the rehearsal `scope`, and the
  right `results`.
- **Nothing stored was recomputed (check 4).** S1's `stage_counts.py`, run on the four runs,
  equals the committed table. Every stage finished before an interruption computes 0 on the next
  launch. `GkSource` 119 + 215 − 43 = 291, and QSI 332 + 532 = 864, with no excess for rows in
  flight.
- **The prune (check 5).** `prune_check.py` exits 0. Independently, the 43 serials listed after
  `resumed-a` (11 on shard 0, 10 on shard 1 and 22 on shard 3, all `GkSource`) were each looked up
  on the shard read `mode=ro`, and none is present.
- **The fingerprints (check 6).** `compare_stores.py` exits 0 ("verdict: equal", each store equal
  to its sidecar). `fingerprint_store(write=False)` gives both stores `39809dca…` with 21 classes,
  and `compare_fingerprints` against 07's reference and 09's rehearsal `status.json` is `[]` for
  each.
- **Integrity (check 7).** This campaign's `store_soundness.py` exits 0 on both stores. Every file
  is `integrity_check` `ok` with no foreign-key rows. There is one `(1, '2025.1.1')` version row
  per shard, and no inventory problem or `duplicate`. The census gives "no divergence".
  `replication_in_flight` is empty on both primaries, and no journal file is present. 07's original
  raises the `TypeError` the issue records. The copy differs from it only by passing `factories`
  and reading the shards from the `shards` table.
- **The readers (check 8).** All seven logs exit 0, with no `ReadOnlyMiss`, `ReadOnlyWrite` or
  traceback, and the store unchanged around each. Run here through `run_readers.sh … nomodels
  extract_TkWKB_data` into the scratchpad, the reader exits 0 with the QCD skip line, and all ten
  store files are byte-identical to the hashes taken at the review's start. `extract_Gk_data`
  wrote 324 files with `--models` and 423 without. That is not a defect: the reader draws its
  source redshifts at random, and both runs are LambdaCDM only.
- **S1 and L4 (check 9).** The scripts carry their usage, name no prompt and take every path as
  an argument. `run_readers.sh` reads its shards from the `shards` table, and refused a misnamed
  script with exit 1, the store unchanged. `stage_counts.py` takes labels from the manifest, and
  labels 07's four runs. `policy_statistics.py` gives 14 lines for R0's reference and for 09, and
  exits 1 on `resumed-a`, which has no block. R0's reference differs from 09 only by the swap of
  `minimal` and `complete` under `mixed`. R0's own `resumed-b` equals 09's exactly, and sorted,
  the blocks are equal. `main.py:2871-2893` inserts each quality on first sight, in completion
  order, as the issue says.
- **The guard (check 10).** It refused with `RuntimeError`, naming the five files. All twelve
  files, the sidecars included, matched hashes taken before it, and no run was registered.
- **The records (check 11).** The board, README's statement, the index (106 by counting, §1.20
  new) and the verification record agree with the run directories. The run table's times and pids
  equal the manifests and `status.json`.
- **Suites (check 12), at `7650102`, load 4–17.** All six pass: AdaptiveLevin 32, CosmologyModels
  39, LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 739
  (163 + 185 + 200 + 191). The wall-clock test did not fail. `black --check` is clean on the five
  new Python files.
- **Two statements of the note were wrong, and the agent caught both.**
  - §0 said 07's `store_soundness.py` runs unchanged. It has failed since `19f07ee`, and the second
    issue records it.
  - At `resumed-b`'s launch the orchestrator said a QSI progress line could come after about 60 s.
    That holds for `RayWorkPool`'s defaults only. `main.py` gives the QSI queue a 300 s minimum and
    a batch of 2000 (`main.py:119`, `:4155-4156`), as the agent's log records. The pattern change
    it prompted is harmless either way.

  The note stays as written, since it is the record of what the agent was given.
- **Residue fixed here:** "this commit" → `7650102` on this board and in the log's header, and
  `orchestrator/prompt-02.md` added to the list of notes.
- **Handed on.**
  - **04.** Its read-only fingerprint of R0's reference store (acceptance 3) has the store, at
    `39809dca…`.
    - The new `docs/dk-and-rpk-migration/rehearsal/store_soundness.py` imports
      `Datastore.store_inventory`. That makes one more `docs/` layer import than README §0.2's 17.
    - 04 measures at its own `HEAD`, so its rewrite reaches the new import. Its note should give
      the new count.
  - **05 and 08,** when written:
    - L4's block comparison must allow for `[02-the-policy-statistics-block-orders-its-lines-by-completion]`,
      comparing counts, or sorted lines where only the order differs. 08's acceptance 3 says
      "line for line", and needs that wording.
    - Interruption points are chosen from their own reference, as here.

### 5.3 After prompt 03 (2026-10-11)

**What landed (`9ec603a`).** K1: `requirements.txt` gains
`datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2` at `:17`, between `cycler` and
`distlib`, and no other line changes. K2: the venv was installed from it, after a `--dry-run` that
said `Would install datastorekit-0.2.2` and nothing else. K3: the proof, in
[the log](logs/03-pin-datastorekit-v0.2.2.md). No code and no test changed.

**What the venv holds.**
- `pip freeze` differs from `b510bc9-pip-freeze.txt` by exactly
  `datastorekit @ git+https://github.com/ds283/DatastoreKit@6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`;
  `pip check` is clean.
- `package_provenance()["datastorekit"]` is version `0.2.2`, URL `https://github.com/ds283/DatastoreKit`,
  commit `6d8ee9ae…`, requested revision `v0.2.2`, `editable: false`; `["raypoolkit"]` is `None`.
- The installed `datastorekit/` holds the tag's 20 `.py` files outside `tests/`, byte for byte, no
  `tests/`, nothing else `.py`. The distribution owns 48 entries, all under `datastorekit/` or its
  `dist-info`.
- `import datastorekit, datastorekit.SQL.ShardedPool` succeeds with `ray.is_initialized()` false. Local
  and remote tags agree: object `c32d0278…`, commit `6d8ee9ae…`.

**Counts against the dispatch baselines**, unchanged: AdaptiveLevin 32, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 739
(163 + 185 + 200 + 191). All OK except `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (README
§7's), which failed once in the suite (0.0612 s against 0.06 s) and passed alone three times.

**Rollback (README §4.2).** `git revert` of `9ec603a` plus
`./venv/bin/pip uninstall -y datastorekit`. The venv is not in git, so the revert alone leaves the
package installed (inert while nothing imports it).

**Opened:** nothing. `docs/OPEN_ISSUES.md` is unchanged at 106.

**Handed to 04.** The package is installed and importable, and nothing in SGK imports it yet.

**Orchestrator review of prompt 03 (2026-10-11).** Dispatched from `1321e0e` to one Sonnet
subagent, after the six roots were re-measured at `09cc40a` (same counts, all OK, load 4–11).
Reviewed against its commit `9ec603a`, with nothing landed after it. **Every check of the note's §3
passed**, and acceptance 1–5 are met. No stop condition fired.
- **Scope (check 1).** Four files: `requirements.txt`, the log, this board, and README §4's one
  line ("03 shipped as named"). Nothing outside `requirements.txt` and this folder moved.
  `requirements.txt` gains the one line at `:17`, between `cycler==0.12.1` and `distlib==0.3.9`,
  spelled as K1.
- **The venv (check 2).** `pip freeze` against `b510bc9-pip-freeze.txt` is exactly
  `16a17 > datastorekit @ git+https://github.com/ds283/DatastoreKit@6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`,
  and `pip check` is clean. With the `datastorekit` line removed from each, `pip freeze` equals
  `requirements.txt`. `pip show` gives `Location` in this venv's `site-packages`, `Requires: ray,
  sqlalchemy`, and no editable location.
- **The record (check 3).** The raw `direct_url.json` is `url` `https://github.com/ds283/DatastoreKit`
  and `vcs_info` `{vcs: git, commit_id: 6d8ee9ae…, requested_revision: v0.2.2}`, with no
  `dir_info`. `package_provenance()` in a fresh process gives the five fields of K3.3, and
  `raypoolkit` `None`. The dry-run report's `install` list has one entry, `datastorekit` 0.2.2, at
  `6d8ee9ae…`.
- **The files (check 4).** The log's script, taken from the log verbatim (equal to the agent's
  file), on a fresh `git archive v0.2.2 datastorekit` gives 20 byte-identical, none missing,
  differing or extra, and no `tests/`. A sha256 listing of the 20 installed `.py` files equals the
  archive's. `importlib.metadata.files("datastorekit")` has 48 entries, under `datastorekit` and
  `datastorekit-0.2.2.dist-info` only.
  - **Non-vacuity.** The same script against `git archive v0.2.1 datastorekit` exits 1 with
    `differ: ['SQL/ShardedPool.py']`, and nothing else differs. That is the one non-test file
    `v0.2.1` → `v0.2.2` changes. So the comparison would have caught the wrong tag.
- **The import (check 5).** In `python -I`, `import datastorekit, datastorekit.SQL.ShardedPool`
  resolves to the venv's `site-packages`, and `ray.is_initialized()` is false. No Ray process was
  up before or after.
- **The tags (check 6).** Local `rev-parse` gives `c32d0278…` and `^{commit}` `6d8ee9ae…`.
  `ls-remote` gives the same pair. DatastoreKit's checkout is clean at `460b012`, untouched.
- **Inert (check 7).** `git grep` for an `import`/`from datastorekit` statement outside `prompts/`
  finds nothing.
- **The registry and `var/` (check 8).** `RunRegistry list` shows the same 21 runs, none
  `running`. `var/dk-and-rpk-migration-03/` holds the seven saves, and its `freeze-before.txt`
  equals `b510bc9-pip-freeze.txt`.
- **Suites (check 9), at `9ec603a`, load 4–7.** All six pass: AdaptiveLevin 32, CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 739
  (163 + 185 + 200 + 191). The wall-clock test the agent saw fail once
  (`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, 0.0612 s against 0.06 s) passed here.
- **The records (check 10).** The log shows every check with its command and output, the dry run,
  pip's output and the suites against the baselines. Its three deviations are each an
  `IMPLEMENTATION CHOICE` the note directed. The board's row, K1–K3 and §5.3, with the rollback,
  agree with the venv. The index is untouched at 106.
- **The agent's observation on pip's log** holds, and is not a defect. pip's "Resolved … to
  commit `c32d0278…`" prints the annotated tag object's id. The checkout of a tag object lands on
  its commit, and `direct_url.json` and `pip freeze` record `6d8ee9ae…`.
- **Residue fixed here:** "this commit" → `9ec603a` on this board and in the log's header, and
  `orchestrator/prompt-03.md` added to the list of notes.
- **Handed to 04.**
  - Every manifest written from now on records `datastorekit` at `6d8ee9ae…`, `editable: false`.
  - 04's note must give the `docs/` layer-import count measured at its own `HEAD`. 02's review
    found one more than README §0.2's 17: `docs/dk-and-rpk-migration/rehearsal/store_soundness.py`.

### 5.4 After prompt 04 (2026-10-11)

**What landed (`0c9e2cf`).** The DatastoreKit swap, in one commit, from `67dd316` (code of
`9ec603a`):
- **A1.** The 15 layer files deleted; `Datastore/__init__.py` and `Datastore/SQL/__init__.py` at
  0 bytes (U3); `tools/__init__.py` and `tools/inventory_report.py` kept.
- **A2.** 134 `from` statements (35 nested) and 7 `import_module` calls, at 141 lines in 88 files,
  rewritten by module path only, by `scripts/rewrite_layer_imports.py` from DatastoreKit's
  measuring script's JSON at `67dd316`. `RayTools/RayWorkPool.py:8` and
  `docs/dk-and-rpk-migration/rehearsal/store_soundness.py:30` are among them. Every name each
  statement imports exists in the package, but the six `_factories` of A5.
- **A3.** The 26 carried test modules (419 tests) and `schema_description.py` deleted; the unported
  test moved to `test_inventory_report.TestTheReportRendersBothCosmologyTypes`, identical by `ast`;
  `test_inventory_retired` reads the package (7 tests → 6).
- **A4.** `CLAUDE.md`'s section is now "The datastore layer is `datastorekit`" (26 lines);
  `RunRegistry/stores.py:5` names the package.
- **A5.** The six failing `docs/` probes fail as before, now on `datastorekit.SQL.Datastore`.
- **By the user's decision of 2026-10-11** (after 04 stopped on it):
  `ComputeTargets/tests/test_quadsource_integral.py`'s `load_pre_commit_module()` maps the one
  layer import of the historical `39ed7fc` file it execs, which `Datastore/__init__.py`'s emptying
  broke. No assertion changes.

**Measured** ([log](logs/04-adopt-datastorekit.md)):
- DatastoreKit's script at a commit of the staged tree
  ([`measure/out/04-datastorekit-measure_client_imports.md`](measure/out/04-datastorekit-measure_client_imports.md)):
  15 absent; 0 layer imports, strings and `import_module` calls outside `prompts/`; `prompts/`'s 51
  statements unchanged; factory imports 99 (97), `b510bc9`'s 100 less `test_inventory_declarations.py:474`.
- **R0's reference store fingerprints `39809dca…` through the package**, format 6, all 21 classes
  equal to its sidecar's; `datastorekit.tools.shard_key_audit` gives `VERDICT: OK`; the six store
  files kept their SHA-256 and mtimes, and no journal appeared.
- Breakages (i)–(vi) each fail what they should.

**Counts against README §7:** Datastore **320** (54 + 103 + 101 + 62, 18 modules); RunRegistry 286,
ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped). All OK.
1937 test ids before, 1518 after: 420 gone (the 26 modules and one `test_inventory_retired` test),
1 added (the moved test), no outcome changed. The wall-clock flake
`test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed in one root run and passed in the
final one and twice of four alone.

**Rollback (README §4.2).** `git revert` of `0c9e2cf` alone. The installed, unimported package is
then inert; nothing in the venv changes.

**Opened:** `[04-six-docs-probes-import-factories-from-the-layer]` and
`[04-comments-name-the-layers-old-module-paths]`. `docs/OPEN_ISSUES.md` goes from 106 to 108, both
under §1.20.

**Handed to 05.** Every `ShardedPool`, actor, `read_inventory` and `open_read_only` SGK runs is the
package's. Its manifests should record `packages.datastorekit` at `6d8ee9ae…`, `editable: false`,
and `raypoolkit` `null`. S1's scripts are as 04 left them (`store_soundness.py:30` imports
`datastorekit.store_inventory`).

**Orchestrator review of prompt 04 (2026-10-11).** Dispatched from `67dd316` to one Opus subagent,
after the six roots were re-measured at `41d9bc0` (same counts, all OK, load 4–9). The agent
stopped once, on a staying test the note had not foreseen (below), and was resumed with the user's
decision. Reviewed against its commit `0c9e2cf`, with nothing landed after it. **Every check of the
note's §3 passed**, and acceptance 1–6 are met. **R0's reference store fingerprints `39809dca…`
through the package.**
- **The stop.** `ComputeTargets/tests/test_quadsource_integral.py`'s `load_pre_commit_module()`
  execs `QuadSourceIntegral.py` as it was at `39ed7fc`, by `git show`. Its `:15`, `from Datastore
  import DatastoreObject`, fails once `Datastore/__init__.py` is empty. The rewrite cannot reach it
  (it is in history) and DatastoreKit's script cannot see it; the root's count did not move, since
  the test errors rather than disappears. It is the only staying test that loads historical code.
  The user chose to map that one line before exec, with an assertion that it occurs exactly once;
  no assertion changed, and no issue was opened.
- **Scope (check 1).** 42 `D` (the 15, the 26 and `schema_description.py`, each as expected), 95 `M`
  (the 88 rewritten files, the two `__init__.py`, `CLAUDE.md`, `test_quadsource_integral.py`, this
  board, README and the index) and 3 `A` (the script, the measurement, the log). `RayTools/`'s
  diff is `:8` alone. `tools/` holds `__init__.py` and `inventory_report.py`.
- **The rewrite is exact (check 2).** The module map, applied here independently to `67dd316`'s
  text at the 141 sites of a fresh JSON, reproduces all 84 script-only files byte for byte. In the
  four hand-edited files the residue is the hand edit alone.
- **By the script at the commit (check 3).** 15 absent; outside `prompts/` 0 statements, 0 strings,
  0 `import_module` calls; `prompts/` 51 statements in 31 files, 20 strings (the 3 of
  `m5_imports.py` and 17 in the rewrite script's map) and 1 call; factory imports 99 (97). The
  saved measurement equals this run but for its header, and `git diff --stat addc09f 0c9e2cf`
  lists only the five records.
- **The layer is gone (check 4).** Both `__init__.py` are 0 bytes. In a fresh `python -I` each of
  the 15 modules raises `ModuleNotFoundError`; all 38 factories subclass
  `datastorekit.SQL.factory_base.SQLAFactoryBase`; Ray stays down.
- **The rewrite script (check 5).** It plans every site before writing and writes nothing on a
  refusal. Re-run on the committed tree with its own JSON, with and without `--dry-run`, it refuses
  at the first site ("the line is not the line at 67dd316") and leaves `git status` clean.
- **The moved test (check 6).** Its method is identical by `ast` (docstring stripped) and as
  source; the class docstring gives its origin and U16.
- **`test_inventory_retired` (check 7).** By `ast`: four tests identical; the actor test differs in
  the file it parses, the grep test in its pathspec; `HELPER_MODULE`, `HELPER_MATCH` and the
  excluded-file test are gone; one import added (`importlib.util`) and one rewritten.
- **Suites (check 8), at `0c9e2cf`, load 7–18.** All six pass: Datastore 320 (54 + 103 + 101 + 62),
  RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148
  (1 skipped). Against the agent's before-lists, the ids gone are the 419 of the deleted modules and
  `test_the_excluded_file_holds_only_the_new_service_helper`, and the one new id is the moved test.
  `black --check` is clean on the 92 touched `.py` files.
- **R0's store (check 9).** `store fingerprint` without `--write`, and `fingerprint_store(write=False)`
  in-process: format 6, `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21
  classes, comparison with the sidecar `[]`, `wrote: false`, `read_inventory` from
  `datastorekit.store_inventory`. The six files' SHA-256 and mtimes equal those taken before
  dispatch, after the agent's reads and this one, and no journal appeared.
- **The breakages (check 10).** Replayed in `bash` from the log's diffs, each applying cleanly at
  `0c9e2cf` and reverting to a clean tree:
  - (i) three Datastore a–f modules fail on `ModuleNotFoundError: …ObjectFactories.base`, and the
    script, at a commit from a temporary index, counts the one layer import at `redshift.py:8`;
  - (ii) every importer of `BackgroundModel` fails with the `DatastoreObject` `ImportError`: the
    log's three ComputeTargets modules, and all 18 Datastore modules here;
  - (iii) Datastore g–o runs 102, so the root 319;
  - (iv) `test_quadsource_policy_main` fails on `No module named 'Datastore.SQL.ShardedPool'`;
  - (v) only `test_the_datastore_actor_class_has_no_method` fails;
  - (vi) `TestOldRegimeRegression` errors with the `ImportError`.
- **A5 (check 11).** All six raise the `_factories` `ImportError`, now from
  `datastorekit.SQL.Datastore`: five run bare, `order_audit.py` through `d3_tables()`.
- **`CLAUDE.md` (check 12).** One hunk, 26 lines for 27, the six points of §2.4.1; the test command
  is untouched. `RunRegistry/stores.py`'s docstring diff is `:5-8`, re-flowed, words unchanged
  but for the names.
- **Entry points (check 13).** `py_compile` passes on `main.py`, the seven `extract_*.py`
  (`extract_common.py` included) and the 14 rewritten `docs/` files; the five `docs/` scripts load
  by path with Ray down.
- **The records (check 14)** agree with the tree: the board's row, items, two §3 entries and §5.4;
  README §4's line, which names both differences; the index at 108 by counting, both rows under
  §1.20.
- **One statement of the note was wrong, and the agent caught it.** §0 said `config/`'s one
  rewritten import was in `config/datastore.py`; it is `config/model_list.py:4`. The note stays as
  written, as the record of what the agent was given.
- **Residue fixed here:** "this commit" → `0c9e2cf` on this board and in the log, and
  `orchestrator/prompt-04.md` added to the list of notes.
- **Handed to 05–08, whose prompts are written next (README §2.1).**
  - R1 runs at `0c9e2cf` or a later commit touching no code. Every manifest's `packages` should
    be `datastorekit` at `6d8ee9ae…`, `editable: false`, and `raypoolkit` `null`.
  - S1's scripts are as 04 left them: `store_soundness.py:30` now imports
    `datastorekit.store_inventory`.
  - Under `v0.2.2` the pools release their actor names on close (DatastoreKit's `v0.2.2`
    addendum); R1 is the first build to run that code.

### 5.5 After prompt 05 (2026-10-11)

**What landed (`f3f0879`).** R1: README §4.1's protocol at `6861a9f`, whose code is `0c9e2cf`'s.
The layer is the installed `datastorekit` 0.2.2, and the work queue is the frozen `RayTools/`.
- Four registered launches of `quadsource_atol_sweep.py --build --rehearsal`, then the readers and
  the checks. They are recorded in
  [`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md)
  §2, with the small outputs under names carrying `05`.
- No code, no test and no script changed.

**R1 holds.**
- **Every launch ended as recorded.**
  - `reference` `done`/0 (12 m 13 s, at loads of 14–78).
  - `resumed-a` `killed`/143 in the `GkSource` stage.
  - `resumed-b` `killed`/143 in the `QuadSourceIntegral` stage.
  - `resumed-c` `done`/0.
  - Each driver's exit status agrees, and each launch has its fingerprint.
- **Every manifest** has `git_dirty: false`, `git_head` `6861a9f`, `script_sha256` `b7935877…`,
  and `packages` with `datastorekit` 0.2.2 at `6d8ee9ae…`, `editable: false`, and `raypoolkit`
  `null`.
- **Both stores are `39809dca…`**, format 6, all 21 classes equal. `compare_fingerprints` is empty:
  - between the two stores;
  - against each sidecar;
  - against the fingerprints recorded by R0's reference, 07's reference and 09's rehearsal.
- **The resume.**
  - It pruned the 45 `GkSource` rows `resumed-a` left unvalidated (all 45 serials absent), and
    recomputed 235 = 291 − 101 + 45.
  - `resumed-c` computed 530 = 864 − 334 `QuadSourceIntegral` rows.
  - Nothing stored was recomputed, and nothing is duplicated.
- **Integrity.** The census, `integrity_check`, `foreign_key_check`, the version row and the
  package's inventory are clean on both stores. The census and soundness outputs equal R0's once
  the prompt number is normalised. The guard refused, naming the five files, and changed none.
- **The readers.** The six, and `extract_Gk_data.py` without `--models`, exit 0, refuse nothing,
  and leave every store file byte-identical.
- No Ray process of this repository outlived either killed driver.
- R0's twelve files are byte-identical, with the same mtimes, before and after.

**L4 equals R0's** (verification §2.5):
- **The statistics counts are equal:** 291; numeric 109; mixed 26, of which 25 complete and 1
  minimal; WKB 156. The block equals R0's sorted. Line for line, it differs by the order of the two
  `mixed` lines, and equals 09's (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
- The reference's stage counts are equal, stage by stage.
- The resumed chain's four identities hold.
- The reference's `** ` sequence is equal.

**The logs** differ from R0's only by the SciPy `IntegrationWarning` moving from `resumed-c` to
`resumed-b`, and by the integrity warning's entry count (45, not 43). The interruption points
explain both. `resumed-a`'s handler printing `SystemExit(1)` is 07's recorded `ray.wait` case. No
log holds a traceback, a new warning, or a line from the package's close.

**DatastoreKit's G2 holds** (§2; verification §2.10):
- **Part 1, the deletions:** `0c9e2cf` (04).
- **Part 2, the pin:** `9ec603a` (03), recorded in every R1 manifest.
- **Part 3, the six roots:** pass at `6861a9f` (05).
- **Part 4, the fingerprint:** reproduced by R1's two runs that finished `done`,
  `dk-and-rpk-migration-05-reference-20261011T192802` and
  `dk-and-rpk-migration-05-resumed-c-20261011T195044`.

**Counts against README §7**, unchanged, all OK: AdaptiveLevin 32, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 320
(54 + 103 + 101 + 62). The wall-clock test, which failed at dispatch, passed.

**Opened:** nothing. `docs/OPEN_ISSUES.md` is unchanged at 108.

**Handed to 06 and on.**
- R1's stores and run directories stay, for 08's comparison: `compare_with_recorded.py --against
  var/runs/dk-and-rpk-migration-05-reference-20261011T192802`, and verification §2.5's tables.
- R1's reference stage times are verification §2.5.5's. Its QSI stage, at 289.7 s, came within
  10 s of the queue's 300 s minimum notify interval, so 08 must read its own reference for a
  progress line.
- Recording G2 on DatastoreKit's board is for 09 to draft (C12).

**Orchestrator review of prompt 05 (2026-10-11).** Dispatched from `6861a9f` to one Opus subagent,
after the six roots were re-measured there (all at their counts; the wall-clock test failed in the
root run and five times alone at load ~15). The agent was resumed with `SendMessage` after each of
the four launches, with one wait per launch on the run's liveness; every wait ended `finished`,
heartbeat never stale. Reviewed against its commit `f3f0879`, with nothing landed after it. **Every
check of the note's §3 passed**, and acceptance 1–7 are met. No stop condition fired. **R1 holds,
and DatastoreKit's G2 holds.**
- **Scope (check 1).** 20 files: verification §2 appended at `:732` with no line of the header or §1
  touched, 17 saved outputs named `*_05*`, the log, and this board. Only the verification record and
  this board are modified; README, `docs/OPEN_ISSUES.md`, every script and all code are untouched.
- **The runs (check 2).** Read from each run directory. `reference` and `resumed-c` are `done`/0;
  `resumed-a` and `resumed-b` are `killed`/143, with fingerprints `b6b89d84…` and `09170e40…` and no
  `fingerprint_error`. Each launch log ends with the matching driver exit status. The killed
  `stderr.log`s hold the handler's two lines (both `SystemExit(1)`, 07's `ray.wait` case) and, in
  `resumed-b`, the SciPy warning; no traceback. The interrupters signalled at 19:46:10 after a 10 s
  hold on the `GkSource` banner, and at 19:49:45 after a 95 s hold on the QSI banner; each handler
  stamp is the same second. No interrupter, Ray process or build of this checkout is alive.
- **The manifests (check 3).** All four: `git_head` `6861a9f`, `git_dirty: false`, `packages` the
  prompt's §2 record exactly (compared as JSON), `script_sha256` `b7935877…`, `--build --rehearsal`
  on all and `--resume` on `resumed-b` and `-c` only, the rehearsal `scope`, the right `results`,
  and a `--purpose` naming R1, the package's commit and `39809dca`.
- **Nothing stored was recomputed (check 4).** `stage_counts.py` on the four runs equals the
  committed table, and its `reference` column equals R0's stage by stage. `GkSource` 101 + 235 − 45
  = 291, QSI 334 + 530 = 864, and every stage finished before an interruption computes 0 next.
- **The prune (check 5).** `prune_check.py` on `after-resumed-a.txt`: 45 serials, 45 absent, none
  reused, none unvalidated.
- **The fingerprints (check 6).** `compare_stores.py`: verdict equal, each store equal to its
  sidecar. `compare_with_recorded.py` on both primaries against R0's reference, 07's reference and
  09's rehearsal: 0 differences in each of the six comparisons, format 6, `39809dca…`, 21 classes.
- **Integrity (check 7).** `store_soundness.py` exits 0, `=> clean`: every file `integrity_check`
  `ok`, 0 foreign-key rows, one `(1, '2025.1.1')` version row per shard, no inventory problem or
  `duplicate`. The census gives "no divergence" for both. `replication_in_flight` is empty on both
  primaries, and no journal file is present.
- **The readers (check 8).** All seven exit 0, with the store byte-identical around each and no
  `ReadOnlyMiss`, `ReadOnlyWrite` or traceback; the run without `--models` prints the QCD skip
  line. Run here through `run_readers.sh … nomodels extract_TkWKB_data` into the scratchpad, the
  reader exits 0 with the skip line and the store unchanged.
- **L4 (check 9).** `policy_statistics.py` on R1's reference: 14 lines, every count R0's; against
  R0's block, only `complete` and `minimal` under `mixed` swap, sorted equal, and it equals 09's
  line for line. The 26 `** ` lines of R1's reference equal R0's in order, numbers stripped.
- **The log differences (check 10).** Each `stderr.log` and launch log compared here with R0's same
  slug, as sets of line kinds with times, pids, paths and numbers stripped. `reference` and
  `resumed-a` are identical; the only difference elsewhere is the four lines of the one SciPy
  `IntegrationWarning`, in `resumed-b` rather than `resumed-c`, because R1's `resumed-b` ran 25 s
  more of the QSI stage (334 rows stored against R0's 332). Nothing from the package's close
  (`_kill_actors`) appears in any log, and neither killed build left an orphan.
- **G-DK and R0 (check 11).** In a fresh `python -I`, the layer's modules raise
  `ModuleNotFoundError`, and `datastorekit.__file__` is in the venv's `site-packages`. The §2 row
  names each of G2's four parts with its commit or run. R0's twelve files equal the table taken
  before dispatch, by SHA-256, mtime and size, and all 24 store files of R0 and R1 were unchanged by
  this review.
- **The records (check 12)** agree with the run directories: the §1 row, L1–L4 (R1) and G-DK ✅,
  the §2 G2 row, §5.5; the index is 108 by counting, with nothing opened. The log classifies both of
  the note's corrections, and its deviations, observations and hand-off hold.
- **Suites (check 13), at `f3f0879`, load 4–13.** AdaptiveLevin 32, CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped), RunRegistry 286 and Datastore 320 (54 + 103 + 101 + 62) pass.
  ComputeTargets 693 failed once, on `test_tk_wkb_phase.TestCost.test_wall_time_per_object`; alone
  it passed three times at load 5, with best-of-3 times of 0.0493–0.0499 s against its 0.05 s
  threshold. At dispatch it failed five times alone at load ~15 (0.065–0.085 s), and the agent's
  root run passed. It is the known wall-clock flake, with almost no margin on this machine. No
  Python file was added, so `black` has nothing to check.
- **The note's two corrections held.** `prune_serials.py` was not needed, and every manifest
  stamped `b7935877…`. The note's statement that the package prints nothing new held as well.
- **Residue fixed here:** "this commit" → `f3f0879` on this board (the §1 row, the §2 G2 row and
  §5.5) and in the log's header, and `orchestrator/prompt-05.md` added to the list of notes.
- **Handed on.**
  - **06** pins `raypoolkit` and changes the venv. R1's stores and runs stay for 08.
  - **08** compares with R1 through `compare_with_recorded.py --against
    var/runs/dk-and-rpk-migration-05-reference-20261011T192802` and verification §2.5. R1's QSI
    stage took 289.7 s, within 10 s of the 300 s minimum notify interval, so R2's reference may
    print a QSI progress line. 08 chooses its points from its own reference.
  - **09** drafts DatastoreKit's G2 record (C12) from verification §2.10.

### 5.6 After prompt 06 (2026-10-11)

**What landed (`b6cc119`).** Q1: `requirements.txt` gains
`raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0` at `:76`, between `ray==2.43.0` and
`referencing==0.36.2`, and no other line changes (99 -> 100 lines). Q2: the venv was installed from it,
after a `--dry-run` whose report's `install` list held `raypoolkit` 0.2.0 alone. Q3: the proof, in
[the log](logs/06-pin-raypoolkit-v0.2.0.md). No code and no test changed.

**What the venv holds.**
- `pip freeze` differs from 03's after-freeze by exactly
  `raypoolkit @ git+https://github.com/ds283/RayPoolKit@eaef0ce60e2fc354bb32f0ad576010e6a83c1976`;
  `pip check` is clean. With the `datastorekit` and `raypoolkit` lines removed, `requirements.txt`
  equals the freeze.
- `package_provenance()["raypoolkit"]` is version `0.2.0`, URL `https://github.com/ds283/RayPoolKit`,
  commit `eaef0ce6…`, requested revision `v0.2.0`, `editable: false`. `["datastorekit"]` is unchanged
  (`6d8ee9ae…`).
- The installed `raypoolkit/` holds the tag's 4 `.py` files outside `tests/`, byte for byte, no
  `tests/`, nothing else `.py`. Against `v0.1.0` the same script fails, naming all four files.
- `import raypoolkit; from raypoolkit import RayWorkPool` succeeds with `ray.is_initialized()` false.
  Local and remote tags agree: object `4e7eb153…`, commit `eaef0ce6…`. Nothing outside `prompts/`
  imports it.

**Counts against the dispatch baselines**, unchanged: AdaptiveLevin 32, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 320
(54 + 103 + 101 + 62). All OK except `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (README
§7's), which failed once in the suite (0.0611 s against 0.06 s, load ~13) and passed alone three times.

**Rollback (README §4.2).** `git revert` of `b6cc119` plus `./venv/bin/pip uninstall -y raypoolkit`.
The venv is not in git, so the revert alone leaves the package installed (inert while nothing imports it).

**Opened:** nothing. `docs/OPEN_ISSUES.md` is unchanged at 108.

**Handed to 07.** The package is installed and importable, and nothing in SGK imports it yet.

**Orchestrator review of prompt 06 (2026-10-11).** Dispatched from `0972599` to one Sonnet
subagent, after the six roots were re-measured at `2d3429c` (every count 04's; the wall-clock test
failed in the root run and three times alone at load ~20). Reviewed against its commit `b6cc119`,
with nothing landed after it. **Every check of the note's §3 passed**, and acceptance 1–5 are met.
No stop condition fired.
- **Scope (check 1).** Four files: `requirements.txt`, the log, this board, and README §4's one
  line ("06 shipped as named"). `requirements.txt` gains the one line at `:76`, between
  `ray==2.43.0` and `referencing==0.36.2`, spelled as Q1 (99 → 100 lines).
- **The venv (check 2).** `pip freeze` against 03's after-freeze is exactly
  `75a76 > raypoolkit @ git+https://github.com/ds283/RayPoolKit@eaef0ce60e2fc354bb32f0ad576010e6a83c1976`,
  and `pip check` is clean. With both packages' lines removed from each, `pip freeze` equals
  `requirements.txt`. `pip show` gives `Location` in this venv's `site-packages`, `Requires: ray`,
  and no editable location.
- **`datastorekit` untouched (check 3).** The snapshot of its 28 installed files and `dist-info`,
  taken before dispatch, is equal after, by SHA-256 and mtime. The `site-packages` listing gains
  exactly `raypoolkit` and `raypoolkit-0.2.0.dist-info`. pip's saved outputs show `datastorekit`
  cloned at `v0.2.2` and skipped in both the dry run and the install, as the note predicted from
  pip's resolver; the dry-run report's `install` list holds `raypoolkit` alone, and the install
  printed no uninstall and installed `raypoolkit-0.2.0` alone.
- **The record (check 4).** The raw `direct_url.json` is `url` `https://github.com/ds283/RayPoolKit`
  and `vcs_info` `{vcs: git, commit_id: eaef0ce6…, requested_revision: v0.2.0}`, with no
  `dir_info`. `package_provenance()` in a fresh process gives the five fields for each package,
  `datastorekit`'s unchanged.
- **The files (check 5).** 03's script, taken from 03's log (byte-equal to the agent's copy), on a
  fresh `git archive v0.2.0 raypoolkit` gives 4 byte-identical, none missing, differing or extra,
  and no `tests/`. A sha256 listing of the 4 installed `.py` files equals the archive's.
  `importlib.metadata.files("raypoolkit")` has 16 entries, under `raypoolkit` and
  `raypoolkit-0.2.0.dist-info` only; `top_level.txt` is `raypoolkit`.
  - **Non-vacuity.** The same script against `git archive v0.1.0 raypoolkit` exits 1 with
    `differ: ['RayWorkPool.py', '__init__.py', '_timing.py', 'contract.py']`, as the note
    predicted and the log records. So the comparison would have caught the wrong tag.
- **The import (check 6).** In `python -I`, `import raypoolkit; from raypoolkit import RayWorkPool`
  resolves to the venv's `site-packages`, `RayWorkPool` is the class, and `ray.is_initialized()`
  is false. No Ray process was up before or after.
- **The tags (check 7).** Local `rev-parse` gives `4e7eb153…` and `^{commit}` `eaef0ce6…`;
  `ls-remote` gives the same pair. RayPoolKit's checkout is clean on `main` at `c9b0e11`, and
  DatastoreKit's is clean; both untouched.
- **Inert (check 8).** `git grep` for an `import`/`from raypoolkit` statement outside `prompts/`
  finds nothing.
- **The registry and `var/` (check 9).** `RunRegistry list` shows the same 25 runs, none
  `running`. `var/dk-and-rpk-migration-06/` holds the seven saves, and its `freeze-before.txt`
  equals 03's after-freeze.
- **Suites (check 10), at `b6cc119`, load 8–18.** All six pass: AdaptiveLevin 32, CosmologyModels
  39, LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693 (the wall-clock test
  passed in the root run this time), Datastore 320 (54 + 103 + 101 + 62).
- **The records (check 11).** The log shows every §2.3 check with its command and output, the dry
  run and its report, pip's output, the comparison's arguments and both runs, and the suites
  against the baselines. It gives the script by reference to 03's log (`:139-172`) rather than
  pasting it; the reference is exact, so this is not residue. Its one deviation, README §4's line,
  is an `IMPLEMENTATION CHOICE` the note directed. The board's row, Q1–Q3 and §5.6, with the
  rollback, agree with the venv. The index is untouched at 108.
- **Residue fixed here:** "this commit" → `b6cc119` on this board (the §1 row and §5.6) and in the
  log's header; README §4's new line rewrapped to the file's width; and `orchestrator/prompt-06.md`
  added to the list of notes.
- **Handed to 07.**
  - Every manifest written from now on records `raypoolkit` at `eaef0ce6…` beside `datastorekit`
    at `6d8ee9ae…`, both `editable: false`.
  - `from raypoolkit import RayWorkPool` gives the class, which is the form 07's rewrite uses.
  - 07's note must re-measure RayPoolKit's `measure_client_pools.py` at its own `HEAD` (7 imports,
    54 calls at `b510bc9`).

### 5.7 After prompt 07 (2026-10-11)

**What landed (`643c3b2`).** B1: `RayTools/RayWorkPool.py` and `RayTools/__init__.py` deleted by
`git rm` (the directory's ignored bytecode removed too, so `RayTools` cannot be imported as a
namespace package). B2: seven lines, `main.py:63` and the six `extract_*.py`, one line each, now
`from raypoolkit import RayWorkPool`, and nothing else on a line or in a file changed. B3:
`CLAUDE.md`'s section is "The datastore layer is `datastorekit`; the work queue is `raypoolkit`"
(27 lines, was 26), and `RunRegistry/tests/test_pipeline_adoption.py:32-33`, the comment's first
sentence, gives the package's places (`:206`, `:588`, `:606`) and says the block holds five lines.
Nothing else in the test file changed. No pool site, handler keyword or other test changed.

**Measured** ([the log](logs/07-adopt-raypoolkit.md); [the script's output](measure/out/07-raypoolkit-measure_client_pools.md)).
- RayPoolKit's script at a scratch commit of the staged tree: `RayTools/` absent, **0 imports**, **0
  `import_module` calls**, the 2 string literals (`docs/tolerance-convergence/inventory.py:382`, a
  `prompts/` script), **54 calls, every one `constructs`**, the handler table identical to `HEAD`'s.
- Six roots at their counts: AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
  RunRegistry 286, ComputeTargets 693 (the wall-clock test passed), Datastore 320 (54 + 103 + 101 +
  62). The five `ast`-matching `ComputeTargets/tests` modules and `test_pipeline_adoption` have equal
  test ids and outcomes before and after. `black --check` clean.
- In a fresh `python -I`: `RayWorkPool.__module__` is `raypoolkit.RayWorkPool`, `import RayTools`
  raises `ModuleNotFoundError`, `ray.is_initialized()` is false. Ray was never started.
- Four breakages, each caught: (i) `raypoolkit.RayWorkPool(...)` at `main.py:3686` fails six tests in
  two modules; (ii) and (iii) one reverted import is counted by the script, and with `main.py:63`
  reverted `test_main_plumbing` still passes, so the script, not the suite, is B2's proof; (iv) the
  `A3_OUTPUT` banner changed fails one test and errors two.

**The gate.** RayPoolKit's G2 (§2) waits for R2 (08); 07 supplies `RayTools/` gone, 0 imports, the
pin installed, the six roots passing and all 54 sites constructing.

**Rollback (README §4.2).** `git revert` of `643c3b2` alone. An unimported package is inert.

**Opened:** nothing. `docs/OPEN_ISSUES.md` is unchanged at 108.

**Handed to 08.** The package is the only pool; `main.py:3247`'s post handler now gets the persisted
object (C8), measured by 08 from the policy statistics block. The log's last section lists what R2's
launches should expect to differ from R1's.

**Orchestrator review of prompt 07 (2026-10-11).** Dispatched from `789b72d` to one Sonnet
subagent, after the six roots were re-measured at `cd03ec4` (every count 04's; the wall-clock test
failed once in the root run, 0.0602 s, and passed alone three times). Reviewed against its commit
`643c3b2`, with nothing landed after it. **Every check of the note's §3 passed**, and acceptance
1–6 are met. No stop condition fired.
- **Scope (check 1).** `D` for the two `RayTools/` files; `A` for the log and the saved
  measurement; `M` for the seven importing files, `CLAUDE.md`, `test_pipeline_adoption.py`, this
  board and README. Nothing else; `docs/` and `docs/OPEN_ISSUES.md` untouched.
- **The seven hunks (check 2).** Each `git diff -U0` is one line, `-from RayTools.RayWorkPool import
  RayWorkPool` / `+from raypoolkit import RayWorkPool`, at the prompt's line. Independently, `sed` of
  each `HEAD~1` blob hashes to the committed blob, all seven.
- **By the script at the commit (check 3).** RayPoolKit's `measure_client_pools.py sgk --commit
  643c3b2` (RayPoolKit clean at `c9b0e11`): `RayTools/` absent, **0 imports**, **0 `import_module`
  calls**, the 2 strings, **54 calls, every one `constructs`**. Against the note's own scratch
  measurement of the bare swap (`2bdfdaa`) it differs only by the tracked-file count and the record
  lines naming `RayWorkPool` (`CLAUDE.md:51`, `:65`, the test comment, `prompts/`); every call and
  handler table is identical. The saved `measure/out/07-…md` differs from it only by commit, file
  count and record lines, and `git diff --stat 5624f4c 643c3b2` lists only the four records.
  `git ls-tree 643c3b2 RayTools` is empty.
- **The queue is the package's (check 4).** In a fresh `python -I`: `import RayTools` and
  `import RayTools.RayWorkPool` raise `ModuleNotFoundError`; `RayWorkPool` is
  `raypoolkit.RayWorkPool.RayWorkPool` from the venv's `site-packages`; `ray.is_initialized()`
  false. `main.py` and the six `extract_*.py` byte-compile. The agent's removal of the ignored
  `RayTools/__pycache__/` (its deviation 4) was needed for the first of these: with it left, the
  directory imports as a namespace package. It is gone.
- **`CLAUDE.md` (check 5).** Retitled as the prompt says; 27 lines (`:47-73`, limit 29); names
  RayPoolKit with its path, URL and `v0.2.0`; "Neither is this repository's code", each package
  reported to its own repository; the import and the bare name, with the `ast.Name` reason; the
  `RayTools/` paragraph gone. The diff touches nothing outside the section.
- **`test_pipeline_adoption.py` (check 6).** The agent's diff is the first sentence, `:32-33`, naming
  `:206`, `:588` and `:606` of the installed file (each checked) and five lines with no completion
  line. `ast.dump` equals `HEAD~1`'s.
- **Suites (check 7), at `643c3b2`, load 11–17.** AdaptiveLevin 32, CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped), RunRegistry 286, Datastore 320 (54 + 103 + 101 + 62), all OK.
  ComputeTargets 693 with one failure, `test_wall_time_per_object` (0.0607 s, load ~15), which alone
  passed twice and failed once (0.066 s): README §7's wall-clock test, touching nothing 07 changed.
  `black --check` clean on the eight touched `.py` files.
- **Breakages (check 8).** All four replayed from the log's diffs in `bash` (`git apply --check`,
  apply, check, `-R`), the tree clean after each. (i) at `main.py:3686`: the same six failures the
  note predicted, in `test_quadsource_policy_main` and `test_main_plumbing`; the other three modules
  pass. (ii) `extract_Gk_data.py:32` and (iii) `main.py:63`, each on a temporary-index scratch commit
  of `643c3b2`: the script counts 1 import, at that line, and still 54 `constructs`; with (iii)
  applied, `test_main_plumbing` passes (27). (iv) 1 failure and 2 errors in `test_pipeline_adoption`,
  as predicted.
- **Item 5 (check 9).** The log's row-by-row table gives "nothing" for every row but `main.py:3247`,
  each with its reason, and traces `:3247` through the installed `datastorekit` as the note's
  correction 2 directs. It also finds the other post-handler sites at `:1427`, `:2122`, `:3235` (the
  checklist's `:1420`, `:2115`, `:3229`) and `:2985`, a lookup-only pool with no persist.
- **The machine (check 10).** `RunRegistry list`: the same 25 runs, none `running`; no Ray process.
  RayPoolKit unchanged. **DatastoreKit's checkout is not as at 06's review**: its `HEAD` is
  `389e13b` and `pyproject.toml` is modified (`version` `0.2.2` → `0.2.3`, saved 22:11). That is
  DatastoreKit's own `factory-defaults` campaign, run by another session there, whose commits
  continue through 22:02; nothing of 07's touched it. SGK pins the tag, and
  `package_provenance()["datastorekit"]` is unchanged (`6d8ee9ae…`, `v0.2.2`, not editable).
- **The records (check 11).** The log has the note's three corrections classified, both
  measurements, the seven hunks, item 5's table, the suites beside the baselines, the four breakages
  as diffs, and what 08 should expect. Its six deviations are classified; none is drift. Board row,
  B1–B3 and §5.7 with the rollback, and README §4's "07 ships" line, agree with the tree. Index 108.
- **The note's own slip.** Its correction 2 put `_type` and `_quality` at
  `ComputeTargets/GkSourcePolicyData.py:573-574`; they are at `:574-575`, as the checklist says. The
  agent measured and logged it.
- **The comment's provenance (check 12).** `test_pipeline_adoption.py:33-37` said `A3_OUTPUT` was
  "transcribed from" `var/runs/a3-pilot/run.out`. Neither of its progress lines is in that file or in
  `a3-pilot-resume/run.out`, and their `+0100` stamps are an offset the pool never prints. Judged
  residue, not an issue: one comment in a file 07 touched, with no assertion behind it. Fixed here
  to "modelled on … not copied from it", with the reason; `ast` unchanged and the module passes.
- **Residue fixed here:** "this commit" → `643c3b2` on this board (the §1 row and §5.7) and in the
  log (header, breakage preamble, state, rollback); the comment above; and
  `orchestrator/prompt-07.md` added to the list of notes.
- **Handed to 08.**
  - Every pool is `raypoolkit` 0.2.0's, and every manifest records both packages at their tags'
    commits, neither editable.
  - L4's `GkSourcePolicyData` block is the measurement of `main.py:3247` (C8); compare it with R1's
    as sorted lines where only the order differs
    (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
  - The printed lines are the copy's strings at new places in the file; the registry's stage and
    progress fields are the run-level test.
  - DatastoreKit is moving towards `0.2.3` in its own campaign. 08 runs at the `v0.2.2` pin; a move
    of the pin is a pin prompt of its own, never part of 08.

### 5.8 After prompt 08 (2026-10-11)

**What landed (`f539015`).** R2: README §4.1's protocol at `5631f0e`, whose code is `643c3b2`'s
but for one test comment (`af26914`, `ast` unchanged). The layer is the installed `datastorekit`
0.2.2, and the work queue the installed `raypoolkit` 0.2.0.
- Four registered launches of `quadsource_atol_sweep.py --build --rehearsal`, then the readers and
  the checks. They are recorded in
  [`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md)
  §3, with the small outputs under names carrying `08`.
- RayPoolKit's `measure_client_pools.py sgk --commit 5631f0e`, saved as
  [`measure/out/08-raypoolkit-measure_client_pools.md`](measure/out/08-raypoolkit-measure_client_pools.md).
- No code, no test and no script changed.

**R2 holds.**
- **Every launch ended as recorded.**
  - `reference` `done`/0 (11 m 25 s).
  - `resumed-a` `killed`/143 in the `GkSource` stage.
  - `resumed-b` `killed`/143 in the `QuadSourceIntegral` stage.
  - `resumed-c` `done`/0.
  - Each driver's exit status agrees, and each launch has its fingerprint.
- **Every manifest** has `git_dirty: false`, `git_head` `5631f0e` and `script_sha256` `b7935877…`.
  Its `packages` is the prompt's §2 record: `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit`
  0.2.0 at `eaef0ce6…`, neither editable.
- **Both stores are `39809dca…`**, format 6, all 21 classes equal. `compare_fingerprints` is empty:
  - between the two stores;
  - against each sidecar;
  - against the fingerprints recorded by R1's reference, R0's reference, 07's reference and 09's
    rehearsal.
- **The resume.**
  - `resumed-a` was killed 10–15 s into the `GkSource` stage, as at R1, but had stored no
    `GkSource` row. So `resumed-b` pruned nothing and computed 291 − 0 + 0 = 291.
  - **R2 does not exercise the `GkSource` prune at open**; L2's prune clause holds vacuously. R1
    exercised it with the same `datastorekit` 0.2.2 (its 45 rows, §5.5).
  - `resumed-c` computed 583 = 864 − 281 `QuadSourceIntegral` rows.
  - Nothing stored was recomputed, and nothing is duplicated.
- **Integrity.** The census, `integrity_check`, `foreign_key_check`, the version row and the
  package's inventory are clean on both stores. The census and soundness outputs equal R1's once
  the prompt number is normalised. The guard refused, naming the five files, and changed none.
- **The readers**, now on `raypoolkit`'s queues. The six, and `extract_Gk_data.py` without
  `--models`, exit 0, refuse nothing, and leave every store file byte-identical. Their lookups are
  R1's.
- No Ray process of this repository outlived either killed driver.
- R0's and R1's 24 files are byte-identical, with the same mtimes, before and after.

**L4 equals R1's** (verification §3.5):
- **The statistics counts are equal:** 291; numeric 109; mixed 26, of which 25 complete and 1
  minimal; WKB 156. This holds in `reference` and in `resumed-b`. Each block equals R1's sorted.
  Line for line, each differs by the order of the two `mixed` lines, and the reference's equals
  R0's (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
- The reference's stage counts are equal, stage by stage.
- The resumed chain's identities hold, the prune's vacuously.
- The reference's `** ` sequence is equal.
- **The printed lines are R1's.** The completion, `Queue summary` and `(no compute items …)` lines
  are equal in number and form in all four runs. Every `status.json` `stage` is R1's, and
  `progress` is `null` in all four, as at R1. No build printed a progress line, at R1 or R2; the
  GkWKB reader's two have the copy's form.

**`main.py:3247` (C8) is measured** (verification §3.11).
- The post handler now gets the persisted object: the local object pickled to the shard actor and
  back, with `store_id` set, traced through the installed `datastorekit`.
- In `reference` and `resumed-b` it ran 291 times on it, and every count is R1's.
- `resumed-c`'s block comes through the lookup exit and does not exercise it.
- C8 changes nothing the block records.

**The logs** differ from R1's only by two things, both explained by the interruption points:
- the SciPy `IntegrationWarning` moving from `resumed-b` to `resumed-c`;
- `resumed-b`'s integrity warning, absent because nothing was unvalidated.

No build log holds a traceback, a new warning, a `ValueError` or a line from the new pool. The GkWKB
reader's log holds two matplotlib warnings from a randomly drawn plot (the log, observation 3).

**RayPoolKit's G2 holds** (§2; verification §3.10):
- **Part 1, `RayTools/` gone:** `643c3b2` (07).
- **Part 2, 0 imports, 0 `import_module` calls, 54 sites `constructs`:** RayPoolKit's script at
  `5631f0e` (08).
- **Part 3, the pin:** `b6cc119` (06), recorded in every R2 manifest.
- **Part 4, the six roots:** pass at `5631f0e` (08).
- **Part 5, the fingerprint:** reproduced by R2's two runs that finished `done`,
  `dk-and-rpk-migration-08-reference-20261011T224347` and
  `dk-and-rpk-migration-08-resumed-c-20261011T230717`.

**Counts against README §7**, unchanged, all OK: AdaptiveLevin 32, CosmologyModels 39,
LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693, Datastore 320
(54 + 103 + 101 + 62). The wall-clock test, which failed at dispatch, passed.

**Opened:** nothing. `docs/OPEN_ISSUES.md` is unchanged at 108.

**Handed to 09.**
- All six of the campaign's stores and their run directories stay, for the user to retire after
  09's review (README §3).
- RayPoolKit's G2 record, for 09 to draft for RayPoolKit's own board (C12), is verification §3.10;
  DatastoreKit's is §2.10.
- The registry holds 29 runs, none `running`. The venv is unchanged.

**Orchestrator review of prompt 08 (2026-10-11).** Dispatched from `5631f0e` to one Opus subagent,
after the six roots were re-measured at `af26914` (every count 07's; the wall-clock test failed in
the root run and six times alone, 0.106–0.120 s at load 12–16, with the machine shared with other
sessions). The agent was resumed with `SendMessage` after each of the four launches, with one wait
per launch on the run's liveness; every wait ended `finished`, heartbeat never stale. Reviewed
against its commit `f539015`, with nothing landed after it. **Every check of the note's §3 passed**,
and acceptance 1–7 are met. No stop condition fired. **R2 holds, and RayPoolKit's G2 holds.**
- **Scope (check 1).** 23 files: verification §3 appended after `:1620` (1063 lines added, none
  removed, so the header, §1 and §2 are untouched), 19 saved outputs named `*_08*`, the saved
  measurement, the log, and this board. README, `docs/OPEN_ISSUES.md`, every script and all code
  are untouched.
- **The runs (check 2).** Read from each run directory. `reference` and `resumed-c` are `done`/0;
  `resumed-a` and `resumed-b` are `killed`/143, with fingerprints `ff6c4257…` and `aa860ed8…`, 21
  classes, problems `{}`, no `fingerprint_error`. Each launch log ends with the matching driver exit
  status. The killed `stderr.log`s hold the handler's two lines (both `SystemExit(1)`) and nothing
  else; no traceback or `ValueError` anywhere. The interrupters signalled at 23:01:02, 10 s after
  first reading the `GkSource` banner, and at 23:05:44, 100 s after first reading the QSI banner
  (handler stamped 23:05:45). No interrupter, build or Ray process of this checkout is alive; the
  Ray processes up at the review were another repository's.
- **The manifests (check 3).** All four: `git_head` `5631f0e`, `git_dirty: false`, `packages` the
  prompt's §2 record exactly (compared as JSON), `script_sha256` `b7935877…`, `--build --rehearsal`
  on all and `--resume` on `resumed-b` and `-c` only, the rehearsal `scope`, the right `results`, and
  a `--purpose` naming R2, both packages' commits and `39809dca`.
- **Nothing stored was recomputed (check 4).** `stage_counts.py` on the four runs equals the
  committed table, and its `reference` column equals R1's stage by stage. `GkSource` 0 + 291 − 0 =
  291, QSI 281 + 583 = 864, and every stage finished before an interruption computes 0 next.
- **The prune (check 5), and the one thing R2 does not cover.** `resumed-a` was signalled 10–15 s
  into the `GkSource` stage, as at R0 and R1, but had stored **no** `GkSource` row (0 on all four
  shards; R1 101 with 45 unvalidated, R0 119 with 43). So `resumed-b`'s prune at open had nothing to
  remove, `prune_check.py` reports "nothing to check", and L2's prune clause holds only vacuously at
  R2. The review judged this an observation, not an issue: it is not a missed interruption (the
  run was killed in the right stage), the copy-to-package diff shows the queue schedules as before,
  the prune runs in `datastorekit`'s shard actors at open, not in the queue, and R1 exercised it
  with the same `datastorekit` 0.2.2. Why nothing had landed 10–15 s in is not in the record; the
  log labels its load explanation as an inference. What R2 leaves unexercised is the queue
  interrupted between store and validate; `resumed-b`'s kill exercised it interrupted with stores
  in flight, and `resumed-c` then computed exactly the 583 missing rows.
- **The fingerprints (check 6).** `compare_stores.py`: verdict equal, each store equal to its
  sidecar. `compare_with_recorded.py` on both primaries against R1's reference, R0's reference,
  07's reference and 09's rehearsal: 0 differences in each of the eight comparisons, format 6,
  `39809dca…`, 21 classes.
- **Integrity (check 7).** `store_soundness.py` exits 0, `=> clean`: every file `integrity_check`
  `ok`, 0 foreign-key rows, one `(1, '2025.1.1')` version row per shard, no inventory problem or
  `duplicate`. The census gives "no divergence" for both. No journal file.
- **The readers (check 8).** All seven exit 0 with the store byte-identical around each and no
  `ReadOnlyMiss`, `ReadOnlyWrite` or traceback. Run here through `run_readers.sh … nomodels
  extract_TkWKB_data` into the scratchpad, the reader exits 0, prints the skip line, writes 27
  files, and leaves the store unchanged. The two matplotlib `UserWarning`s in the GkWKB reader's log
  come from its `random.sample` of source redshifts (`extract_GkWKB_data.py:444`); each plot gets
  its refs from `object_get` directly, so the queue cannot change what it plots.
- **L4 (check 9).** The 26 `** ` lines of each run equal R1's same slug, in order, numbers
  stripped (23 for `resumed-a`). The chain identities are in the log, logged `STRUCTURALLY
  REQUIRED`.
- **C8 (check 10).** `policy_statistics.py` on R2's `reference`, `resumed-b` and `resumed-c`: 14
  lines each, every count R1's (291; 109 / 26 (25 complete, 1 minimal) / 156), each equal to its
  committed output. `reference` and `resumed-b` differ from R1's only by the order of the two
  `mixed` lines, and sorted are equal; `reference` equals R0's line for line; `resumed-c` equals
  R1's line for line. So the persisted object changes nothing the block records. The trace through
  the installed `datastorekit` is verification §3.11, and `resumed-c` is named as not exercising it.
- **The printed lines (check 11).** For each slug, the counts of `ALL WORK ITEMS COMPLETE`,
  `Queue summary:` and `(no compute items` lines equal R1's, and every line kind of `stdout.log`
  is R1's once numbers are stripped. No build printed a progress line at R1 or R2 (R2's reference
  QSI took 299.2 s, 0.8 s inside the 300 s minimum); the GkWKB reader's two are the copy's form.
  Every final and at-the-signal `status.json` `stage` is R1's.
- **The log differences (check 12).** As line kinds, numbers stripped, against R1's same slug:
  `reference` and `resumed-a` identical in all three files; `resumed-b` lacks the SciPy
  `IntegrationWarning` (4 kinds) and the integrity warning's entries (12 kinds in `stdout.log`, 16
  in the launch log), and `resumed-c` gains the warning. The first moved because R2's `resumed-b`
  stored 281 QSI rows at its point against R1's 334; the second is absent because nothing was
  unvalidated. Nothing else differs.
- **G-RPK, R0 and R1 (check 13).** `RayTools` is absent and cannot be imported. RayPoolKit's script
  at `f539015` (RayPoolKit clean at `c9b0e11`): 0 imports, 0 `import_module` calls, 2 strings, 54
  calls all `constructs`; it differs from the saved `measure/out/08-…md` only by commit, file count
  and record lines. The §2 row names each of G2's five parts with its commit or run. R0's and R1's
  24 files equal the note's table, and all 36 store files of R0, R1 and R2 were unchanged by this
  review's reads.
- **Records, suites and `black` (check 14).** The §1 row, L1–L4 (R2) and G-RPK ✅, the §2 G2 row,
  §5.8, the log's 13 classified deviations (the note's five corrections among them) and its four
  observations agree with the run directories; the index is 108 by counting, nothing opened. At
  `f539015`, load 13–44: AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
  RunRegistry 286 and Datastore 320 (54 + 103 + 101 + 62) pass; ComputeTargets 693 with the one
  wall-clock failure (0.082 s), which failed three times more alone (0.079–0.083 s, load 11–30)
  against the agent's pass at its own root run. It is README §7's known flake; nothing it reaches
  has changed since `643c3b2`. No Python file was added.
- **The note held.** Its five corrections were each needed and each followed; its facts (script
  hash, manifest record, 24 hashes, C8's reading, the printed lines unchanged) all held.
- **Residue fixed here:** "this commit" → `f539015` on this board (the §1 row, the §2 G2 row and
  §5.8) and in the log (header and hand-off), and `orchestrator/prompt-08.md` added to the list of
  notes.
- **Handed to 09.**
  - Both packages' G2 hold; 09 drafts the two records for DatastoreKit's and RayPoolKit's boards
    from verification §2.10 and §3.10 (C12).
  - The six stores and twelve run directories stay for the user to retire after 09's review.
  - The wall-clock test failed in every run here, alone included, at loads of 11–30, and passed
    in the agent's root run; 09's final measurement should re-run it alone at low load before
    reporting it.
