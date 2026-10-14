# runregistry-extraction campaign — implementation state

**Last updated:** 2026-10-14 · **Status: IN PROGRESS — 5 of 10 prompts written (01, 02, 02a, 03,
04), 3 landed (01, 02, 02a); R0 holds. E1 holds: DatastoreKit `v0.4.0` (§5.5), so 03 and 04 are
written. 05–09 held on RunRegistryKit's release (E2). D1–D5, Q1–Q8 and D6 ruled by the user.**
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
| **D6** — fingerprints SGK holds without `encoding_format` (2026-10-14) | the `v0.4.0` addendum's option (c): R0's and R1's four stores are retired after 04's review, while SGK's own code still compares them (§5.5) |
| **Q1–Q8** | the audit's §8, all as recommended: the names; a config file at the root; a worktree's run recorded in the main checkout; one distribution with a `[stores]` extra; the spec-digest guard (07); a test root for the driver tests; no manifest format field during the move; two value tables per spec left to SI |

The orchestrator's choices of method, C1–C11, are in README §6.2.

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Freeze the run registry](01-freeze-the-run-registry.md) | the freeze; the frozen tree; the baselines | Sonnet | ✍️ 2026-10-12 | ✅ | `a45aefc` | [log](logs/01-freeze-the-run-registry.md) |
| 02 | [Rehearse the start tree (R0)](02-rehearse-the-start-tree.md) | **R0**; the `f539015` → `511ef5f` gap | Opus | ✍️ 2026-10-12 | ✅ | `74eff08` | [log](logs/02-rehearse-the-start-tree.md) |
| 02a | [The rehearsal-lock helper](02a-the-rehearsal-lock-helper.md) | `tools/rehearsal_lock.py`; the interop and real-lock proofs; `CLAUDE.md`'s lock section | Sonnet | ✍️ 2026-10-12 | ✅ | `c863055` | [log](logs/02a-the-rehearsal-lock-helper.md) |
| 03 | [Pin `datastorekit` at `v0.4.0`](03-pin-datastorekit-at-v0.4.0.md) | the DatastoreKit pin; the three addenda | Sonnet | ✍️ 2026-10-14 | ⬜ | — | — |
| 04 | [Rehearse through the new DatastoreKit (R1)](04-rehearse-through-the-new-datastorekit.md) | **R1**; the fingerprint both ways | Opus | ✍️ 2026-10-14; ⏸ 03 | ⬜ | — | — |
| 05 | Pin `runregistrykit` at E2's release | the RunRegistryKit pin | Sonnet | ⏸ E2 | ⬜ | — | — |
| 06 | Adopt `runregistrykit` | the swap; closes both `00-` issues | Opus | ⏸ E2, 04, 05 | ⬜ | — | — |
| 07 | Guard the fingerprint format | **Q5** | Sonnet | ⏸ 06 | ⬜ | — | — |
| 08 | Rehearse through both packages (R2) | **R2**; **RunRegistryKit G2** | Opus | ⏸ 07 | ⬜ | — | — |
| 09 | Close the campaign | the records; the gates drafted; the hand-over to A3 v2 | Sonnet | ⏸ 08 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–02 now; 02a after R0, when the shared lock was adopted; 03–04 when E1's release was tagged (written 2026-10-14); 05–08 when E2's
release is tagged and its SGK checklist exists; 09 after 08's review. The holds are for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-12, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-12,
used for 02); [`orchestrator/prompt-02a.md`](orchestrator/prompt-02a.md) (2026-10-12, used for
02a).

### 1.1 Items

| Item | Prompt | What | State |
|---|---|---|---|
| **F1** | 01 | `CLAUDE.md`'s freeze section | ✅ |
| **F2** | 01 | the frozen tree, file by file (§2.1 below) | ✅ |
| **F3** | 01 | the baselines: `list`, `store show` on the 14 sidecars, the seven roots | ✅ |
| **L1**–**L4** (R0) | 02 | README §4.1 at the start tree; L4 against `dk-and-rpk-migration`'s R2 | ✅ |
| **H1** | 02a | `tools/rehearsal_lock.py`: five commands, the owner format, no way to break a lock | ✅ |
| **H2** | 02a | its tests, and the six mutations they catch | ✅ |
| **H3** | 02a | the interoperability proof against SI's and ChamPBH's scripts | ✅ |
| **H4** | 02a | the real-lock proof, with an in-process probe | ✅ |
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
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ✅ holds (`74eff08`): `runregistry-extraction-02-reference-20261012T091921` and `-resumed-c-20261012T101145`, both `done`, both `39809dca…`, equal to `dk-and-rpk-migration` R2's recorded fingerprints |
| **E1** | DatastoreKit has tagged a release carrying the fingerprint algorithm and a version for the encoding and algorithm, with an SGK-facing note ([brief](datastorekit-fingerprint-brief.md)) | — | ✅ holds (2026-10-14, §5.5): `v0.4.0`, tag object `8e7c1d77…` on `c8570e84…`, CI run 38538726292 green on `c8570e8`; the note is DatastoreKit's `docs/adoption/secondarygwkit.md`, "Addendum: v0.4.0" |
| R1 | E1's release reproduces `39809dca…`, and DatastoreKit's fingerprint equals SGK's on every store | 04 | ⬜ |
| **E2** | RunRegistryKit has tagged a release imported from 01's frozen tree (its G1), with the client config, a `[stores]` extra over E1's release, and SGK's adoption checklist ([brief](runregistrykit-extraction-brief.md)) | — | ⬜ requested 2026-10-12 |
| RunRegistryKit G2 | `RunRegistry/` is gone and nothing imports it, the pin is installed, the roots pass, `list` gives 01's runs, and R2 reproduces `39809dca…` | 06, 08 | ⬜ |

Recording a gate on RunRegistryKit's or DatastoreKit's own board is a commit in that repository
(README §6.2 C11).

### 2.1 The frozen tree

01 fills this table: every tracked file under `RunRegistry/` at 01's commit, with its blob hash. It
is what E2's G1 compares RunRegistryKit's import with.

Taken at prompt 01's commit, `a45aefc` (`a45aefc702de643c330372dd6d2d983f38bb8eed`), from
`git ls-tree -r HEAD RunRegistry/`: 25 rows, equal to
`git ls-tree -r --name-only HEAD RunRegistry/ | wc -l`, and identical to `c8a7f14`'s
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
  - **Narrowed (2026-10-14, at E1):** DatastoreKit's half is closed by `v0.4.0`.
    `datastorekit.store_fingerprint` holds the algorithm, with `ENCODING_FORMAT = 1` recorded beside
    the digest as `encoding_format` (outside every digest), the client's format a required argument,
    and contract §9.7's rule for when the package bumps its version. SGK's half, recording it, is
    still 06's.

- **[00-recorded-fingerprints-lack-the-encoding-format]** *(opened 2026-10-14 at E1, from
  DatastoreKit's `v0.4.0` addendum to its SGK checklist, item 3; **Assigned (2026-10-14):** by the
  user's ruling D6, to the orchestrator's retirement of R0's and R1's stores after 04's review, and
  checked at 06's gate)*
  - **The defect.** The package's `compare_fingerprints` reads `fingerprint_format`, then
    `encoding_format`; if either differs it gives one entry for each and compares nothing else.
    Every fingerprint SGK's own code takes (`RunRegistry/stores.py:1182-1253`) lacks
    `encoding_format`. So once SGK compares through the package (06), each fingerprint SGK already
    holds compares as one `encoding` entry, whatever the stores hold: a sidecar's `fingerprint`, a
    run's `status.json`, and the six golden files.
  - **Impact.**
    - On a **live** store with such a sidecar, `store fingerprint` would report one difference and
      exit 1, and `store retire` would refuse the store, since it needs an empty comparison. There
      are two such stores today, R0's (`runregistry-extraction-02-{reference,resumed}`; the other 14
      sidecars are tombstones, never fingerprinted again). R1 (04) adds two more.
    - **Run directories** keep their recorded fingerprints as records. A later comparison with one,
      as 08's acceptance 1 makes with R0's and R1's, compares the digests and classes field by field
      with `encoding_format` set aside, not through the package's comparison (README §2.2, 08).
    - The golden files are 06's to change, with `test_the_shape` and `test_the_golden_fingerprint`
      (the addendum's item 2).
  - **Next step.** The user ruled option (c) of the addendum (D6, 2026-10-14). After 04's review,
    the orchestrator retires R0's and R1's four stores with `store retire`, each after a dry run, in
    a commit of its own, as `718e650` did for `dk-and-rpk-migration`'s six. Option (b), SGK's
    comparison kept beside the package's for old fingerprints, is a version-aware reader, which the
    user's standing rule excludes; option (a), `--write` over the recorded fingerprint, would replace
    R0's record in its sidecar.
  - **Closes** at 06's gate, when no sidecar under `var/datastores/` is a live store whose recorded
    fingerprint lacks `encoding_format`.

- **[00-main-and-the-readers-default-to-ray-address-auto]** *(opened 2026-10-12 by the plan's
  rehearsal-lock amendment, README §4.1, from SGK's review of SI's rehearsal-lock brief;
  **Assigned (2026-10-12):** to `handover`, by the user)*
  - **The defect.** `main.py`'s `--ray-address` defaults to `auto` (`DEFAULT_RAY_ADDRESS`,
    `main.py:99`; `ray.init` at `:308`), and so does each of the six `extract_*.py`.
    - In SGK's Ray, 2.43.0, `auto` with `RAY_ADDRESS` unset first reads
      `/tmp/ray/ray_current_cluster`. If that gives nothing, it takes any `gcs_server` address it
      finds by scanning running processes, without a warning
      (`ray/_private/services.py:389-430`).
    - SI and ChamPBH now run isolated `ray start` clusters, which write that file into their own
      temp dirs, not `/tmp/ray`.
  - **Impact.**
    - A `main.py` build or a reader left at the default, run while any other Ray instance is up,
      attaches to that instance.
    - **Against another project's cluster,** today it fails at `ray.init`. Ray refuses to connect
      across versions, and SGK runs Ray 2.43.0 on Python 3.12, against SI's 2.55.1 on 3.13 and
      ChamPBH's 2.53.0 on 3.13. That protection is incidental: it is read from the code, not
      observed, and it disappears if the versions converge.
    - **Against SGK's own in-process instance,** for example a rehearsal or build in flight, the
      versions match. A reader at `auto` would then attach and submit its work into that
      instance.
    - Nothing has gone wrong yet. The rehearsal and the A3 v2 driver,
      `quadsource_atol_sweep.py --build`, force `address="local"`, and `run_readers.sh` passes
      `local`; README §4.1 now requires both.
  - **Next step.** Change the default to `local`, or make the flag required, as SI did (SI's
    `d6d550b`). Either is SGK code, and neither belongs in this campaign:
    - **Required** breaks `quadsource_atol_sweep._main_py_argv`, which passes no
      `--ray-address`. That list is pinned element by element by
      `RunRegistry/tests/test_prune_on_resume.py`, which is frozen until 06.
    - **Either change** would land an SGK change in R1's window, which isolates the DatastoreKit pin
      (README §0.1).

    Until it is fixed, every run states its address.
  - **Assigned (2026-10-12):** to `handover`, by the user. That campaign owns the A3 v2 launch
    (`a3-v2-readiness` README §1), the next long build that runs `main.py`. The fix lands there,
    before A3 v2 or with it.

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

Prompt 01 landed at `a45aefc`, with `HEAD` `d94d66b` as its parent. Its log is
[`logs/01-freeze-the-run-registry.md`](logs/01-freeze-the-run-registry.md).
- `CLAUDE.md` carries the freeze section (F1); §2.1 above is the frozen tree, 25 files (F2).
- The baselines (F3) are under [`measure/out/`](measure/out/): `01-list.txt` (29 runs, 27 finished,
  2 `unknown`, none running), `01-store-show.txt` (14 sidecars, each `kind: registry`,
  `problems: none`, exit 0) and `01-roots.txt` (seven roots, every count as README §0.2's, no
  failure).
- No code or test was added; nothing under `RunRegistry/` or `var/` changed. No issue opened or
  closed; the index stays at 105.

**Orchestrator review of prompt 01** (2026-10-12, against `a45aefc`, by
[`orchestrator/prompt-01.md`](orchestrator/prompt-01.md) §3). All seven checks pass; nothing opened.
1. **Scope.** Six files, all on the prompt's §6 list. `RunRegistry/`, `README.md`,
   `prompts/INDEX.md`, `orchestrator/` and `docs/` are untouched; `docs/OPEN_ISSUES.md` stays at 105.
2. **The freeze section** sits between the registry section and "Repository mechanics", titled as
   the prompt asks, and makes its four points. It limits the freeze to `RunRegistry/` and names no
   caller. `CLAUDE.md`'s diff is one inserted hunk.
3. **The frozen tree**, parsed from the committed board by script: 25 rows, path order, every hash
   40 hex, and equal row for row to `git ls-tree -r` at `a45aefc` and at `c8a7f14`, and to the
   orchestrator's own reading at `e3a7077`.
4. **`01-list.txt`**: taken at `d94d66b`; 29 runs, 27 finished, 2 `unknown`. Run ids and states equal
   a fresh `list` at review time, row for row.
5. **`01-store-show.txt`**: 14 blocks, the backup's among them, each headed by its command. Each is
   byte-identical to a fresh `store show` at review time, and every exit status is 0.
   **`01-roots.txt`**: seven roots at §0.2's counts, LiouvilleGreen with 1 skipped. No flake fired,
   though the load reached 29 from another session.
6. **Nothing moved.** `find var -newer <marker>`, with the marker set just before dispatch, is
   empty. `git diff --stat c8a7f14 a45aefc -- RunRegistry/` is empty.
7. **The records.** The board header, the §1 row, F1–F3, §2.1 and §5.1 are as the note asks. The log
   has every §5.1 heading. Of its five deviations, four are the note's own requests. The fifth,
   "`HEAD` is `d94d66b`", is not a deviation, since the prompt never put `HEAD` at `c8a7f14`, but
   the label does no harm. The commit message is in `CLAUDE.md`'s form.

**Residue fixed in the orchestrator's follow-up:** "this commit" → `a45aefc` on the board and in the
log; the SHA added beside "prompt 01 of `runregistry-extraction`" in `CLAUDE.md`'s freeze section;
the list of orchestrator notes; `prompts/INDEX.md`'s row (1 of 9 landed).

**Handed on.** To 02: nothing beyond the prompt. To RunRegistryKit's campaign, through the user:
**the import commit is `a45aefc`**, or any later commit at which `RunRegistry/` equals §2.1.

### 5.2 After prompt 02 (2026-10-12)

Prompt 02 landed at `74eff08`, with `HEAD` `fe16d8e` as its parent. Its log is
[`logs/02-rehearse-the-start-tree.md`](logs/02-rehearse-the-start-tree.md); the record is
[`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md)
§1.
- **R0 holds** (L1–L4, gate R0). Four registered builds at `fe16d8e`, whose code is `a45aefc`'s:
  `reference` and `resumed-c` `done`/0, `resumed-a` and `resumed-b` `killed`/143. Both stores are
  format 6, `39809dca…`, all 21 classes, equal to each other, to their sidecars, and to the
  fingerprints `dk-and-rpk-migration`'s R2 recorded in its `reference` and `resumed-c` run
  directories.
- **The factory move (`511ef5f`) is invisible to a build.** No build or reader raised on an import
  or an unpickling. All 38 registered factories come from `ObjectFactories`, and the fingerprint did
  not move (verification §1.10).
- **The manifests say what ran.** Every `packages` is R2's record, compared as JSON. Every
  `git_dirty` is `false`, every `git_head` `fe16d8e7…`, every `script_sha256` `b7935877…`.
- **L4 equals R2's.** The statistics counts are equal (291; numeric 109; mixed 26, 25 complete and 1
  minimal; WKB 156), in the reference and both resumed runs that print the block. The reference's
  stage counts and its `** ` sequence are equal, and the printed lines and registry fields are R2's.
  The block's line order is R1's, not R2's: `[02-the-policy-statistics-block-orders-its-lines-by-completion]`
  again.
- **The prune at open is exercised**, which R2 held only vacuously. The orchestrator's 40 % hold left
  164 `GkSource` rows, 63 unvalidated, after `resumed-a`. `resumed-b` pruned exactly those 63, and
  computed 190 = 291 − 164 + 63. `resumed-c` computed 378 = 864 − 486 QSI rows.
- **The load.** The machine was heavily loaded by processes not of this repository: five-minute
  load up to 339. The reference took 29 m 19 s, against R2's 11 m 25 s. Its QSI stage, 1230.1 s,
  printed three pool progress lines, and `resumed-a`'s GkWKB stage one. These are the first from a
  registered build under `raypoolkit`. `resumed-a`'s interrupter read one back from `status.json`.
- **Nothing else under `var/` moved.** The 14 sidecars keep their SHA-256, mtime and size
  (`023d139f…` over them).
- **The seven roots** are at 01's counts. ComputeTargets has the one known wall-clock flake
  (`test_wall_time_per_object`), which passed twice in three runs alone.
- No code or test was added. Nothing was opened or closed; the index stays at 105. The two stores
  stay, with their sidecars (README §3).

**Orchestrator review of prompt 02** (2026-10-12, against `74eff08`, by
[`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) §3). Dispatched from `fe16d8e` to one Opus
subagent, after the seven roots were measured at `cf978c6`. The agent was resumed with
`SendMessage` after each of the four launches, with one wait per launch on the run's liveness;
every wait ended `ENDED finished`, heartbeat never stale. Nothing landed after the agent's commit.
**All thirteen checks pass**, acceptance 1–7 are met, and no stop condition fired. **R0 holds.**
1. **Scope.** 25 files, all added but this board: the verification document, 20 `*_02*` outputs
   and the log. README, `prompts/INDEX.md`, `orchestrator/`, `docs/OPEN_ISSUES.md`, `RunRegistry/`
   and every script are untouched. The verification header names §1–§4 and holds §1 alone.
2. **The runs**, read from each run directory. `reference` and `resumed-c` are `done`/0;
   `resumed-a` and `resumed-b` are `killed`/143, with fingerprints `9bb753bd…` and `a0d053af…`, 21
   classes, problems `{}`, no `fingerprint_error`. Each launch log ends with the matching driver
   exit status. The killed `stderr.log`s hold the handler's two lines and, in `resumed-b`, the
   SciPy `IntegrationWarning`; no traceback anywhere. Each interrupter sent one `SIGTERM`:
   10:02:51, 31 s after first reading the `GkSource` banner, and 10:10:17, 250 s after first
   reading the QSI banner. No process of this checkout is alive.
3. **The manifests.** All four: `git_head` `fe16d8e`, `git_dirty` false, `packages` equal to R2's
   as JSON, `script_sha256` `b7935877…`, `--build --rehearsal` on all and `--resume` on `resumed-b`
   and `-c` only, the rehearsal `scope`, the right `results`, and a purpose naming R0, `fe16d8e`
   and `39809dca`.
4. **Nothing stored was recomputed.** `stage_counts.py` on the four runs equals the committed
   table byte for byte, and its `reference` column equals R2's stage by stage. `GkSource` 164 +
   190 = 291 + 63 pruned; QSI 486 + 378 = 864; every stage finished before an interruption computes
   0 next.
5. **The prune.** `prune_check.py` on the saved state after `resumed-a`: 63 serials checked, 63
   absent, 0 reused, 0 unvalidated. `resumed-b`'s integrity warnings name the same 63: shard 1's
   block lists 20 rows, and Ray folds the other two shards' 23 + 20 into `[repeated 43x across
   cluster]` (`stdout.log:13`, `:142-146`). The verification record says the warnings are the
   prune's but does not count them; this review does. R2 held this clause only vacuously; R0
   exercises it, through correction 4's 40 % hold.
6. **The fingerprints.** `compare_stores.py`: equal, each store equal to its sidecar.
   `compare_with_recorded.py` on both primaries against R2's `reference` and `resumed-c` run
   directories: 0 differences in each of the four comparisons. Format 6, `39809dca…`, 21 classes.
7. **Integrity.** `store_soundness.py` exits 0, `=> clean`, on both: `integrity_check` `ok`, 0
   foreign-key rows, one `(1, '2025.1.1')` version row per shard, no inventory problem or
   `duplicate`. The census gives "no divergence" for both. No journal file.
8. **The readers.** All seven logs: exit 0, store identical around each, and no `Traceback`,
   `ReadOnlyMiss`, `ReadOnlyWrite`, `ModuleNotFoundError`, pickling error or `Datastore.SQL`.
   Run here through `run_readers.sh … nomodels extract_TkWKB_data` into the scratchpad, the reader
   exits 0, prints the skip line, writes 27 files, and leaves all five store files byte-identical.
9. **L4.** `policy_statistics.py` on `reference`, `resumed-b` and `resumed-c`: 14 lines each, every
   count R2's (291; 109 / 26 (25 complete, 1 minimal) / 156), each equal to its committed output and
   to R2's sorted; `resumed-c` equals R2's line for line, the other two differ by the order of the
   two `mixed` lines (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`). The `** `
   lines of each run equal R2's same slug in order, numbers and prefix stripped (26, 23, 26, 26).
10. **The printed lines.** Completion, `Queue summary` and `(no compute items` counts equal R2's in
    every run. The new kinds are the progress lines: three in `reference`'s QSI stage (1230.1 s),
    one in `resumed-a`'s GkWKB stage. They have the copy's form
    (`a3-v2-readiness-07-reference…/stdout.log:100-101`), the later ones with the instantaneous
    rate an earlier registered run's lines also carry. Every final `status.json` `stage` is R2's,
    and `progress` is `null` at the end of all four.
11. **The log differences**, as line kinds, numbers, pids, paths and prefix stripped: `reference`
    and `resumed-a` add only the progress kinds (and `resumed-a` one `Queue summary` in `/min`);
    `resumed-b` adds the integrity warning's kinds and a completion time in `Nm Ns`, and the
    `IntegrationWarning`, which `resumed-c` loses; each `stderr.log` otherwise equals R2's. Each is
    explained in verification §1.2.5 by the point or the load. No import or unpickling error.
12. **`var/`.** The 14 sidecars hash to `023d139f…` as before dispatch; the only new files under
    `var/datastores/` are R0's ten store files and two sidecars; nothing else under `var/` is newer
    than the pre-dispatch marker but R0's runs and working directory. The guard refused, exit 1,
    naming the five files.
13. **Records and suites.** The board and the log are as the note asks; the log has every §5.1
    heading and 16 classified deviations, the note's eight corrections among them. At `74eff08`,
    load 17–55: AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 271, tools 49, LiouvilleGreen
    148 (1 skipped) and RunRegistry 286 pass; ComputeTargets 693 with the one wall-clock failure
    (0.057 s against 0.05 s), which failed three times more alone at load 38–45. It is README §8's
    known flake; no code has changed since `a45aefc`. No Python file was added.
- **The note held.** All eight corrections were needed and followed. Correction 4 was the one that
  changed the outcome: it turned R2's vacuous prune clause into a measured one.
- **Residue fixed here:** "this commit" → `74eff08` on this board (§1 row, §2 R0 row, §5.2) and in
  the log (header and hand-off); this note added to the list of notes; `prompts/INDEX.md`'s row (2
  of 9 landed).
- **Handed on.** 03 waits on E1. R0's two stores and four run directories stay (README §3); 04
  compares with R0 through `runregistry-extraction-02-reference-20261012T091921` and
  `-resumed-c-20261012T101145`. The wall-clock test failed in every run of this review; 09's final
  measurement should re-run it alone at low load.

### 5.3 The rehearsal lock (2026-10-12, after 02)

At the user's request, the plan's §4.1 adopts the rehearsal lock that StochasticInstantons (SI)
proposed. SI's documents are its `.prompts/dk-and-rpk-migration/rehearsal-lock-brief.md` at
`d6d550b` and the handover note at `5fb4a1d`. The lock applies to every rehearsal from R1 (04) on.
It is not a prompt, and it changes no code.
- **Why.** `ray stop` kills every Ray process on the machine. Two `ray start` clusters on Ray's
  default ports collide. Load moves results that have a wall-clock budget. So the user ruled
  (SI's U9) that each project runs and stops only its own Ray, and only one rehearsal runs at a
  time.
- **SGK's part is the lock alone.** SGK's builds already start an in-process instance, and SGK
  runs no `ray stop`. SI's `ray_cluster.sh` is not adopted: it manages a `ray start` cluster, and
  its lock functions are SI's. It does refuse, and leave untouched, a lock whose owner is SGK.
- **What this plan adds to the brief.** One lock covers all four launches and the readers, and
  `pid` is set again at each launch. SI's §5 assumed a single driver.
- **Opened:** `[00-main-and-the-readers-default-to-ray-address-auto]` (§3), the gap SGK's review
  found in the brief. SI confirmed it and amended its brief at `d6d550b`. The index goes to 106.
  The user then assigned it to `handover`, which owns the A3 v2 launch.
- **Prompt 02a** was then written, at the user's request. It replaces §4.1's hand-run lock steps
  with `tools/rehearsal_lock.py`, and adds a check for this repository's Ray processes. R0's
  machine-wide `pgrep` would now also see other projects' clusters. 02a lands before 04.
- When the lock was adopted, it was free and the registry showed nothing `running`.

### 5.4 After prompt 02a (2026-10-12)

[`02a`](02a-the-rehearsal-lock-helper.md) landed at `c863055`; its log is
[`logs/02a-the-rehearsal-lock-helper.md`](logs/02a-the-rehearsal-lock-helper.md). It opened and closed
no issue, so `docs/OPEN_ISSUES.md` is unchanged (106).
- **H1.** `tools/rehearsal_lock.py`: `acquire`, `set-pid`, `release`, `status`, `ray-processes`;
  standard library only; no way to break, wait for or retry a lock.
- **H2.** `tools/tests/test_rehearsal_lock.py`, 42 tests; `tools` is 91 (49 + 42). The six mutations
  of acceptance 6 are each caught (log, "The deliberate-breakage record"). The other six roots are at
  02's counts; `ComputeTargets` shows only `test_wall_time_per_object`, which also failed alone in 3
  of 4 runs at load 26–27.
- **H3.** [`measure/out/02a-interop.txt`](measure/out/02a-interop.txt): both directions for SI
  (`8e04a7c`) and ChamPBH (`288b385`), every lock byte-identical wherever a refusal left it. Their
  `release_lock` was read, not run (orchestrator correction 1, `STRUCTURALLY REQUIRED`); each other
  project's lock was left in its temporary `HOME`.
- **H4.** [`measure/out/02a-real.txt`](measure/out/02a-real.txt): the real lock held 14:42:29 to
  14:42:56, an in-process probe (pid 2778); `release` refused while it lived; all ten of its pids
  were gone 3–5 s after `kill -TERM`; the lock is free and `ray-processes` exits 0.
- **`CLAUDE.md`** has "The shared rehearsal lock", one inserted hunk. README §4.1 was not edited.

**Orchestrator review of prompt 02a** (2026-10-12, against `c863055`, by
[`orchestrator/prompt-02a.md`](orchestrator/prompt-02a.md) §3). Dispatched from `96a7f6e` to one
Sonnet subagent, after the seven roots were measured at `c89fc9e`. Before writing the note the
orchestrator's dry run of SI's `release_lock` was denied by Claude Code's removal check; the user
ruled that those calls are dropped (the note's correction 1). Nothing landed after the agent's
commit. **All eleven checks pass**, and no stop condition fired.
1. **Scope.** Five files added (the module, its tests, the two proofs, the log) and two modified
   (`CLAUDE.md`, this board). README, `prompts/INDEX.md`, `orchestrator/`, `docs/`,
   `RunRegistry/`, `main.py` and every `extract_*.py` are untouched.
2. **The module, by reading.** Standard library only; the five commands and their exit statuses
   are §2.1's; `argparse` offers nothing to break, wait or retry. `os.mkdir` without `exist_ok`,
   `owner.tmp` then `os.replace`, seven keys in the brief's order, the default lock directory
   resolved at call time. `set-pid` and `release` check `project` and `ray_temp_dir`; `release`
   removes `owner`, `owner.tmp` and the directory by name, no `rmtree`. `ray_processes()` matches
   the venv's Ray path anywhere in the line, in both spellings, adds descendants, and never lists
   itself, its ancestors or its own children.
3. **The commands, run here** in a temporary `HOME`: `status` 0, `acquire` 0, `acquire` 3 (owner
   printed, byte-identical), `set-pid` of a live process 0 (only `pid` changed), `release` 1 (pid
   alive), then after the kill `release` 0 and 0 again ("no lock"); `set-pid 1x`, a bare
   `acquire` and an unknown command exit 2; `ray-processes` 0.
4. **The tests, by reading.** §2.2's eight points are present. Every refusal test leaves one
   reason standing, and the agent added the two owners that make that true for the `project`
   check (another project at `in-process`; SGK's name with a cluster's temp dir). The `set-pid`
   test starts from `2026-01-01T00:00:00+00:00`. `setUp` patches `HOME` and `_ps_output` for every
   lock test. No Ray, nothing under `var/`.
5. **The mutations**, re-applied from the log's diffs to a copy of `tools/` in the scratchpad:
   each fails exactly the tests the log names (4, 2, 2, 1, 1 and 5). The copy's own baseline also
   failed three tests. They assert that `gcs_server` appears in the output, which the 150-column
   cut hides once the checkout's path is longer than the repository's by more than 36 characters,
   as the scratchpad's is, and as a worktree's under `.claude/worktrees/` would be. Fixed in the
   follow-up (below).
6. **The interop proof.** Both directions for both projects, every exit status as the note
   amends §2.4, every refused owner byte-identical by hash, `release_lock` quoted (SI 130–147,
   ChamPBH 131–148) and never run. The extracted copies hash to `b934a75c…` and `ad4b9021…`, and
   the two locks left in the agent's scratchpad still hash to `913da846…` and `61746274…`. Direction
   2 for SI, re-run here in a fresh temporary `HOME`: `acquire` 3, `set-pid` 1, `release` 1 (both
   sides printed), `status` 0, the owner byte-identical.
7. **The real proof.** The eight steps, each exit status as §2.5 says. The lock was held 14:42:29
   to 14:42:56. The probe's pid, 2778, came the same from `$!` and from its output. `ray-processes`
   listed nine processes: `gcs_server`, `raylet`, one `ray::IDLE`, and six Python processes that
   are the probe's children. The agents' argv[0] is MacPorts' framework `Python`, and the venv's
   Ray path sits past the 150-column cut, so the record cannot name them. Every pid was gone 3–5 s
   after `kill -TERM`. Now the lock is absent and `ray-processes` exits 0.
8. **`CLAUDE.md`.** One inserted hunk after the freeze section, titled as §2.3 asks, making its four
   points; it claims nothing the helper does not do.
9. **Nothing outside moved.** `find var -newer <marker>` is empty. `~/ScienceData` holds the same
   single entry, `ChamPBH-rehearsals`. Its mtime moved to 14:42, from the real proof's `mkdir` and
   `rmdir`, as it must.
10. **The records.** Board, log and commit message are in the campaign's form. The log
    classifies the note's seven corrections and five choices of its own, records the six mutations
    as diffs, and its counts agree with the outputs.
11. **The suites and `black`.** At `c863055`, load 18–39: AdaptiveLevin 32, CosmologyModels
    39, ObjectFactories 271, **tools 91** (49 + 42), LiouvilleGreen 148 (1 skipped) and RunRegistry
    286 pass; ComputeTargets 693 with the one wall-clock failure (0.106 s against 0.06 s), which then
    failed three times alone at load 19–25 (0.081–0.095 s). No code it runs has changed since
    `74eff08`; it is README §8's known flake, now failing alone at this machine's load as well.
    `~/ScienceData` was listed identically before and after the `tools` root. `black --check` is
    clean on both new files.
- **Residue fixed here:** "this commit" → `c863055` on this board (§1 row, §5.4) and in the log's
  header; this note added to the list of notes; `prompts/INDEX.md`'s row (3 of 10 landed). The
  three tests of check 5 now assert on the pid column (`    100 1 `), which no path length moves,
  and in the scratchpad copy the baseline then passes and each mutation fails the same number of
  tests as the log records.
- **Handed on.** 04 is the first rehearsal to take the lock with the helper. A refusal there lists
  agents by pid and a cut command line; their names are found with `ps -p <pid> -o command=`. The
  wall-clock flake failed alone in three of four runs at load 26–27 during the agent's session;
  09's final measurement should re-run it alone at low load, as §5.2 already asks.

### 5.5 E1 holds (2026-10-14)

DatastoreKit released `v0.4.0`, and the user brought its report to this session. The orchestrator
checked it against DatastoreKit's repository and this tree, at `a5a6f6d`, read-only.
- **The tag.** `v0.4.0` is an annotated tag, object `8e7c1d77b3792b57b98983e84b99fc68ab88cb89`, on
  commit `c8570e849334d09085bbd84876095fc9e22bd005`, the same locally and on `origin`.
  `pyproject.toml` there gives `version = "0.4.0"`, and differs from `v0.2.2`'s by that line alone.
- **CI.** Run 38538726292, on `c8570e8` itself, passed both jobs, `suite (high)` and `suite
  (low)`. DatastoreKit's later `b01737f` touches its records only.
- **What moved under `datastorekit/`.** Between `v0.2.2` and `v0.4.0`: `SQL/factory_base.py`
  (`v0.2.3`), `object.py` (`v0.3.0`) and the new `store_fingerprint.py`. Between `v0.3.0` and
  `v0.4.0`, and between the brief's `d88615e` and `v0.4.0`, the new module alone.
  `store_inventory.py`, which SGK's own fingerprint code reads, is unchanged since `v0.2.2`.
- **The dry run.** `pip install --dry-run` with line 17 moved to `v0.4.0` gave `Would install
  datastorekit-0.4.0` and nothing else.
- **The note** is DatastoreKit's `docs/adoption/secondarygwkit.md`, "Addendum: v0.4.0 (fingerprint
  prompt 02, 2026-10-14)", in the tag. At the pin it asks nothing of SGK; at the call (06) it names
  seven items. Its two findings on SGK's code were checked here: `fingerprint_store` passes `taken`
  as the second positional argument (`RunRegistry/stores.py:1499-1507`), and so does
  `test_taken_is_outside_every_digest` (`RunRegistry/tests/test_store_fingerprint.py:378`).
- **Where the release differs from the brief**, as DatastoreKit reports it, and what each means here:
  - `fingerprint_of(inventory, fingerprint_format, taken=None)`, the format required. 06 passes it;
    the ten calls and the two positional `taken` are 06's (addendum item 1).
  - The version is `encoding_format`, not the audit's placeholder `inventory_format`. No record
    here is corrected: the audit and the brief named it as a suggestion.
  - The listing's first line gains ` encoding_format <e>` at its end. 04's script checks it.
  - Two SGK tests change when SGK calls the package (`test_the_shape`, the golden comparison): 06
    names them, by README §5 rule 7.
- **Opened:** `[00-recorded-fingerprints-lack-the-encoding-format]` (§3), and the user ruled it
  (D6, option (c)). **Narrowed:** `[00-the-fingerprint-encoding-has-no-version-of-its-own]`,
  DatastoreKit's half closed. The index goes to 107.
- **Written:** prompts [03](03-pin-datastorekit-at-v0.4.0.md) and
  [04](04-rehearse-through-the-new-datastorekit.md). The plan is amended where the release and D6
  made it inexact: §2.2's 03 (02a's counts), 04 (`encoding_format` aside, on all four live stores)
  and 08 (the recorded fingerprints compared field by field), §3 and §6.3 (R0's and R1's stores
  retired after 04's review), and §6.1 (D6).
- The registry showed nothing `running`, and exactly two live stores, R0's.

