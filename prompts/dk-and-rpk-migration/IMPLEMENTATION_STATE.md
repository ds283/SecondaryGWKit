# dk-and-rpk-migration campaign — implementation state

**Last updated:** 2026-10-11 · **Status: LIVE — 4 of 9 prompts written (01–04), 01 landed.**
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
| 02 | [Rehearse the start tree (R0)](02-rehearse-the-start-tree.md) | §0.1's gap `53d4e18` → `b510bc9` | Opus | ✍️ 2026-10-11 | ⬜ | — | — |
| 03 | [Pin `datastorekit` `v0.2.2`](03-pin-datastorekit-v0.2.2.md) | the DatastoreKit pin | Sonnet | ✍️ 2026-10-11 | ⬜ | — | — |
| 04 | [Adopt `datastorekit`](04-adopt-datastorekit.md) | the DatastoreKit swap; **U3** | Opus | ✍️ 2026-10-11 | ⬜ | — | — |
| 05 | Rehearse through `datastorekit` (R1) | **DatastoreKit G2** | Opus | ⬜ | ⬜ | — | — |
| 06 | Pin `raypoolkit` `v0.2.0` | the RayPoolKit pin | Sonnet | ⬜ | ⬜ | — | — |
| 07 | Adopt `raypoolkit` | the RayPoolKit swap | Sonnet | ⬜ | ⬜ | — | — |
| 08 | Rehearse through both packages (R2) | **RayPoolKit G2**; `main.py:3247` | Opus | ⬜ | ⬜ | — | — |
| 09 | Close the migration | the inherited issues; the close | Sonnet | ⬜ | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-11, used for 01).

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
| **S1** | 02 | `docs/dk-and-rpk-migration/rehearsal/`: `run_readers.sh`, `stage_counts.py`, `policy_statistics.py` | ⬜ |
| **L1**–**L4** (R0) | 02 | README §4.1 at the start tree; L4 recorded as the baseline | ⬜ |
| **K1** | 03 | the `datastorekit` pin in `requirements.txt` | ⬜ |
| **K2** | 03 | the venv installed from it | ⬜ |
| **K3** | 03 | the proof: one distribution added, at `6d8ee9ae…`, the tag's files, not editable | ⬜ |
| **A1** | 04 | the 15 layer files deleted, two `__init__.py` emptied | ⬜ |
| **A2** | 04 | every layer import outside `prompts/` rewritten, by `scripts/rewrite_layer_imports.py` | ⬜ |
| **A3** | 04 | `Datastore/tests/`: 26 modules and `schema_description.py` deleted, one test moved, `test_inventory_retired` and the two import strings changed | ⬜ |
| **A4** | 04 | `CLAUDE.md`'s layer section, `RunRegistry/stores.py:5` | ⬜ |
| **A5** | 04 | `[04-six-docs-probes-import-factories-from-the-layer]` opened | ⬜ |

05–09's items are added when they are written.

The measurement the plan was written against is [`measure/`](measure/): both packages' scripts at
`b510bc9`, each equal to its checklist's appendix byte for byte; the six roots' counts; the
Datastore root by module; and the venv's `pip freeze` (README §0.2).

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ⬜ |
| DatastoreKit G2 | the 17 files are gone (two emptied, U3), the pin is in place and installed, the suites pass, and R1 reproduces `39809dca…` | 04, 05 | ⬜ |
| RayPoolKit G2 | `RayTools/` is gone, 0 imports, the pin is installed, the six roots pass, all 54 sites construct, and R2 reproduces `39809dca…` | 07, 08 | ⬜ |

Recording each gate on its package's own extraction board is a commit in that repository, not
here (README §6.2 C12).

## 3. Active and unresolved issues

None yet.

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
