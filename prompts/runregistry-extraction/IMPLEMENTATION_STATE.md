# runregistry-extraction campaign — implementation state

**Last updated:** 2026-10-12 · **Status: IN PROGRESS — 2 of 9 prompts written (01, 02), 1 landed (01).
03–09 held on two releases in other repositories (E1, E2). D1–D5 and Q1–Q8 ruled by the user.**
The plan is [`README.md`](README.md), written on 2026-10-12 at `c8a7f14` from
[`docs/runregistry-extraction-audit.md`](../../docs/runregistry-extraction-audit.md).

The campaign moves SGK's `RunRegistry/` into RunRegistryKit and adopts it, with the store
fingerprint's encoding and algorithm moved to DatastoreKit under a version of their own, before the
A3 v2 compute run. A fingerprint rehearsal comes at the start tree (R0), after the DatastoreKit pin
(R1) and after the RunRegistryKit swap (R2).

**Campaign:** [`README.md`](README.md) ·
**Briefs:** [`datastorekit-fingerprint-brief.md`](datastorekit-fingerprint-brief.md),
[`runregistrykit-extraction-brief.md`](runregistrykit-extraction-brief.md) ·
**Code:** `RunRegistry/` (frozen by 01, deleted by 06), `requirements.txt`, `config/fingerprint.py`
(new), the client config file (new), `RunDrivers/tests/` (new), every `RunRegistry` import outside
`prompts/`, `CLAUDE.md` ·
**Verification:** `docs/runregistry-extraction-verification.md` (created by 02) ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.21

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**: the
> row is added, moved or deleted, and the count and date in its header are corrected. See
> `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-12 (README §6.1):**

| Decision | Decided |
|---|---|
| **D1** — where it goes | a repository of its own, RunRegistryKit |
| **D2** — when | before the A3 v2 compute run |
| **D3** — the fingerprint | encoding and digest to DatastoreKit, versioned there; content specification and format number grouped in one client module; RunRegistryKit takes a client fingerprinter |
| **D4** — no fingerprinter | not supported; `[stores]` is a dependency split, not a degraded mode |
| **D5** — client spec coverage | ChamPBH 09 and SI 10 prove it by a test (ChamPBH `64ef9aa`; SI's plan, uncommitted) |
| **Q1–Q8** | the audit's §8, all as recommended: the names; a config file at the root; a worktree's run recorded in the main checkout; one distribution with a `[stores]` extra; the spec-digest guard (07); a test root for the driver tests; no manifest format field during the move; two value tables per spec left to SI |

The orchestrator's choices of method, C1–C11, are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Freeze the run registry](01-freeze-the-run-registry.md) | the freeze; the frozen tree; the baselines | Sonnet | ✍️ 2026-10-12 | ✅ | this commit | [log](logs/01-freeze-the-run-registry.md) |
| 02 | [Rehearse the start tree (R0)](02-rehearse-the-start-tree.md) | **R0**; the `f539015` → `511ef5f` gap | Opus | ✍️ 2026-10-12 | ⬜ | — | — |
| 03 | Pin `datastorekit` at E1's release | the DatastoreKit pin | Sonnet | ⏸ E1 | ⬜ | — | — |
| 04 | Rehearse through the new DatastoreKit (R1) | **R1**; the fingerprint both ways | Opus | ⏸ E1, 03 | ⬜ | — | — |
| 05 | Pin `runregistrykit` at E2's release | the RunRegistryKit pin | Sonnet | ⏸ E2 | ⬜ | — | — |
| 06 | Adopt `runregistrykit` | the swap; closes both `00-` issues | Opus | ⏸ E2, 04, 05 | ⬜ | — | — |
| 07 | Guard the fingerprint format | **Q5** | Sonnet | ⏸ 06 | ⬜ | — | — |
| 08 | Rehearse through both packages (R2) | **R2**; **RunRegistryKit G2** | Opus | ⏸ 07 | ⬜ | — | — |
| 09 | Close the campaign | the records; the gates drafted; the hand-over to A3 v2 | Sonnet | ⏸ 08 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–02 now; 03–04 when E1's release is tagged; 05–08 when E2's
release is tagged and its SGK checklist exists; 09 after 08's review. The holds are for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, under
`orchestrator/`.

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **F1** | 01 | `CLAUDE.md`'s freeze section | ✅ |
| **F2** | 01 | the frozen tree, file by file (§2.1 below) | ✅ |
| **F3** | 01 | the baselines: `list`, `store show` on the 14 sidecars, the seven roots | ✅ |
| **L1**–**L4** (R0) | 02 | README §4.1 at the start tree; L4 against `dk-and-rpk-migration`'s R2 | ⬜ |
| **K1** | 03 | the `datastorekit` pin moved to E1's release; the venv installed from it | ⬜ |
| **L1**–**L4** (R1) | 04 | README §4.1 through E1's release; L4 against R0 | ⬜ |
| **W1** | 04 | the fingerprint both ways, equal on every store (C3) | ⬜ |
| **K2** | 05 | the `runregistrykit[stores]` pin; the venv installed from it | ⬜ |
| **A1** | 06 | the client config file and `config/fingerprint.py` | ⬜ |
| **A2** | 06 | every `RunRegistry` import outside `prompts/` rewritten | ⬜ |
| **A3** | 06 | the 101 SGK tests moved (C5); `RunRegistry/` deleted | ⬜ |
| **A4** | 06 | `CLAUDE.md`: the registry section, the package section, the roots | ⬜ |
| **A5** | 06 | the worktree check (Q3) | ⬜ |
| **V1** | 07 | the spec-digest guard and its breakages | ⬜ |
| **L1**–**L4** (R2) | 08 | README §4.1 through both packages; L4 against R1 | ⬜ |
| **G-RRK** | 08 | RunRegistryKit's G2 recorded as holding (§2) | ⬜ |
| **Z1** | 09 | the final measurements; the gate records drafted; the close | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ⬜ |
| **E1** | DatastoreKit has tagged a release carrying the fingerprint algorithm and a version for the encoding and algorithm, with an SGK-facing note ([brief](datastorekit-fingerprint-brief.md)) | — | ⬜ requested 2026-10-12 |
| R1 | E1's release reproduces `39809dca…`, and DatastoreKit's fingerprint equals SGK's on every store | 04 | ⬜ |
| **E2** | RunRegistryKit has tagged a release imported from 01's frozen tree (its G1), with the client config, a `[stores]` extra over E1's release, and SGK's adoption checklist ([brief](runregistrykit-extraction-brief.md)) | — | ⬜ requested 2026-10-12 |
| RunRegistryKit G2 | `RunRegistry/` is gone and nothing imports it, the pin is installed, the roots pass, `list` gives 01's runs, and R2 reproduces `39809dca…` | 06, 08 | ⬜ |

Recording a gate on RunRegistryKit's or DatastoreKit's own board is a commit in that repository
(README §6.2 C11).

### 2.1 The frozen tree

01 fills this table: every tracked file under `RunRegistry/` at 01's commit, with its blob hash. It
is what E2's G1 compares RunRegistryKit's import with.

Taken at prompt 01's commit ("this commit"), from `git ls-tree -r HEAD RunRegistry/`: 25 rows,
equal to `git ls-tree -r --name-only HEAD RunRegistry/ | wc -l`, and identical to `c8a7f14`'s
(`git diff --stat c8a7f14 HEAD -- RunRegistry/` is empty).

| File | Blob |
|---|---|
| `RunRegistry/__init__.py` | `0228a1a066793332191affe90c4b309a153aa60c` |
| `RunRegistry/__main__.py` | `88cf6ddd2b7603fc718fd6b3d7ed9722261a874f` |
| `RunRegistry/stores.py` | `a0f18ba49a25fe795af79baa8f82ce0e09cf3b25` |
| `RunRegistry/tests/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `RunRegistry/tests/data/full_store_fingerprint.json` | `58b35c3677f40e5421c7390ac282f1e6e808cd68` |
| `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-07.json` | `2fea28b072cd9f4cdb15062ab355c9c08a45d57e` |
| `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-08.json` | `511ff2d3873437480cbb6af028ba8fbdc5699b87` |
| `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-09a.json` | `bd6a855c9d109a8da654ab790eadfc681ad5cea8` |
| `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-09b.json` | `01d7ef02e88c53a4160e5660711f0746f3ae4a90` |
| `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-09c.json` | `6cf0a39496c0e96bd3b2640324482e7d51686908` |
| `RunRegistry/tests/store_fixtures.py` | `391f31fc2155c1f2d8f2350d306272ee6e0ae54e` |
| `RunRegistry/tests/test_build_rehearsal_scope.py` | `8b3c8519d3956b97cce84ea8224c621f37d88cd0` |
| `RunRegistry/tests/test_killed_handler.py` | `213f2a8d7d017f845b85ea209f8e87886539988b` |
| `RunRegistry/tests/test_pipeline_adoption.py` | `f3205799afa5626fe440e409fce37c1cb9fa04c7` |
| `RunRegistry/tests/test_prune_on_resume.py` | `23533e68f7682a172ecf7246b84bbd749b1e5ff8` |
| `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py` | `527a9878d588309fd6f82f214bc655be627cea5d` |
| `RunRegistry/tests/test_run_registry.py` | `01c3ea37d8491c63300667f5a181f3bad91000ab` |
| `RunRegistry/tests/test_store_amend.py` | `8b7db110f76ae74bfb4716fe12d7192bac38265e` |
| `RunRegistry/tests/test_store_amend_residuals.py` | `0c6e6a2ca5d59b74dcb960111a5cd05ed2d991f4` |
| `RunRegistry/tests/test_store_command_line.py` | `6f111e29e9b695340a88bf19eb18a50e89a9ea05` |
| `RunRegistry/tests/test_store_copy_move.py` | `4be78d983aa83d5d7d39fc440519347cc87386bd` |
| `RunRegistry/tests/test_store_fingerprint.py` | `5039b4792210c1227230ec09b48da641b6f62e9f` |
| `RunRegistry/tests/test_store_retire.py` | `3cf05cc02d6b00849a4eff4972ccbedd0a4fa51d` |
| `RunRegistry/tests/test_store_retire_references.py` | `9a3264a7f5977311161f724651b4e1d27cfa0866` |
| `RunRegistry/tests/test_store_sidecar.py` | `533d1118447527cfef65a833525516fcadbbc2e7` |

## 3. Active and unresolved issues

- **[00-a-run-launched-from-a-worktree-records-into-the-worktree]** *(opened 2026-10-12 by the
  plan, from audit §2; **Assigned (2026-10-12):** to 06, which closes it with RunRegistryKit's root
  rule, Q3)*
  - **The defect.** `RunRegistry.REPO_ROOT` is the parent of the package's own directory
    (`RunRegistry/__init__.py:57`), and `DEFAULT_ROOT` is `var/runs` under it (`:58`). A run
    launched from a git worktree of this repository imports the worktree's copy, and so records
    into the worktree's `var/runs/`.
  - **Impact.** The run's record is deleted with the worktree, which breaks `CLAUDE.md`'s
    durability rule (rule 5) without any error. No worktree exists today (`git worktree list`,
    2026-10-12), so no record has been lost; campaigns here do run in worktrees.
  - **Next step.** 06: the root becomes the parent of `git rev-parse --git-common-dir` (Q3), and
    acceptance 4 shows a worktree's run recorded in the main checkout.

- **[00-the-fingerprint-encoding-has-no-version-of-its-own]** *(opened 2026-10-12 by the plan,
  from audit §3.2; a DatastoreKit defect, reported by
  [`datastorekit-fingerprint-brief.md`](datastorekit-fingerprint-brief.md); **Assigned
  (2026-10-12):** SGK's half to 06)*
  - **The defect.** A fingerprint's one version, `FINGERPRINT_FORMAT` (`RunRegistry/stores.py:206`),
    is SGK's and enters the digest. The encoding it digests, `datastorekit.store_inventory`
    (`canonical`, `canonical_json`, `reference_digest`, `Record`), has no version. Three of the six
    format bumps extended that encoding together with a key change (`a171ced`, `dbf2cf2`,
    `3d7b606`).
  - **Impact.** A DatastoreKit release that changed the encoding would change every digest while
    the format stayed `6`. `compare_fingerprints` would then report a content difference in every
    class, not a format mismatch. Today only the rehearsal acceptance (`39809dca…`) would catch it.
  - **Next step.** DatastoreKit versions the encoding and algorithm (E1). 06 records that version
    in every fingerprint SGK takes, and closes SGK's half; R2 (08) confirms it.

## 4. Resolved issues

None yet.

## 5. Dispatch record

### 5.0 Planning (2026-10-12)

- The audit, `docs/runregistry-extraction-audit.md`, was written and committed at `a97a59c`, with
  the user's decisions D1–D5. The user then accepted its §8 (Q1–Q8) as recommended, recorded at
  `c8a7f14`.
- D5's amendments were made in the two client repositories the same day: ChamPBH `64ef9aa` (prompt
  09's acceptance item 5, its README §6.5); SI's still-uncommitted plan (prompt 10's acceptance item
  5, U8).
- This plan, its board, the two briefs and prompts 01–02 were written at `c8a7f14`. The plan opened
  the two `00-` issues above, each assigned to 06.

### 5.1 After prompt 01 (2026-10-12)

Prompt 01 landed at "this commit", with `HEAD` `d94d66b` as its parent. Its log is
[`logs/01-freeze-the-run-registry.md`](logs/01-freeze-the-run-registry.md).
- `CLAUDE.md` carries the freeze section (F1); §2.1 above is the frozen tree, 25 files (F2).
- The baselines (F3) are under [`measure/out/`](measure/out/): `01-list.txt` (29 runs, 27 finished,
  2 `unknown`, none running), `01-store-show.txt` (14 sidecars, each `kind: registry`,
  `problems: none`, exit 0) and `01-roots.txt` (seven roots, every count as README §0.2's, no
  failure).
- No code or test was added; nothing under `RunRegistry/` or `var/` changed. No issue opened or
  closed; the index stays at 105.
