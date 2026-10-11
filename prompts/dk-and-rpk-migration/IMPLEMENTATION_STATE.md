# dk-and-rpk-migration campaign — implementation state

**Last updated:** 2026-10-11 · **Status: LIVE — 4 of 9 prompts written (01–04), 01–04 landed. R0 holds; SGK's layer is now `datastorekit` 0.2.2, pinned and installed; R1 (05) is next.**
The plan is [`README.md`](README.md), written on 2026-10-11 at `b510bc9` with the user's decisions
U1–U4, and committed at `4532188`. The user reviewed it the same day, and 01–04 were written
against it. 05–09 are held (README §2.1).

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
| 05 | Rehearse through `datastorekit` (R1) | **DatastoreKit G2** | Opus | ⬜ | ⬜ | — | — |
| 06 | Pin `raypoolkit` `v0.2.0` | the RayPoolKit pin | Sonnet | ⬜ | ⬜ | — | — |
| 07 | Adopt `raypoolkit` | the RayPoolKit swap | Sonnet | ⬜ | ⬜ | — | — |
| 08 | Rehearse through both packages (R2) | **RayPoolKit G2**; `main.py:3247` | Opus | ⬜ | ⬜ | — | — |
| 09 | Close the migration | the inherited issues; the close | Sonnet | ⬜ | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-11, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-11,
used for 02); [`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-11, used for 03);
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-11, used for 04).

**Held, by design (README §2.1).**
- **01–04** are written after the user's review of the plan.
- **05–08** are written after 04's review, whose log fixes the counts and names they cite.
- **09** is written after 08's review.

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

05–09's items are added when they are written.

The measurement the plan was written against is [`measure/`](measure/): both packages' scripts at
`b510bc9`, each equal to its checklist's appendix byte for byte; the six roots' counts; the
Datastore root by module; and the venv's `pip freeze` (README §0.2).

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ✅ **holds** (2026-10-11): both stores `39809dca…`, all 21 classes, equal to each other and to 07's and 09's recorded fingerprints (verification §1.3.4) |
| DatastoreKit G2 | the 17 files are gone (two emptied, U3), the pin is in place and installed, the suites pass, and R1 reproduces `39809dca…` | 04, 05 | ⬜ |
| RayPoolKit G2 | `RayTools/` is gone, 0 imports, the pin is installed, the six roots pass, all 54 sites construct, and R2 reproduces `39809dca…` | 07, 08 | ⬜ |

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
