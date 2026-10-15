# runregistry-extraction campaign — implementation state

**Last updated:** 2026-10-15 · **Status: IN PROGRESS — 9 of 10 prompts written (01–08 and 02a), 7
landed (01, 02, 02a, 03, 04, 05, 06); R0 and R1 hold. E1 holds: DatastoreKit `v0.4.0` (§5.5), 03 pinned
it (§5.6), and 04 rehearsed through it and took the fingerprint both ways (§5.7). E2 holds:
RunRegistryKit `v0.1.0` with its SGK checklist (§5.10), so 05–08 are written; 05 pinned it (§5.11),
and 06 adopted it: `RunRegistry/` is gone (§5.12). 09 follows 08's review. D1–D5, Q1–Q8 and D6
ruled by the user.**
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
| 03 | [Pin `datastorekit` at `v0.4.0`](03-pin-datastorekit-at-v0.4.0.md) | the DatastoreKit pin; the three addenda | Sonnet | ✍️ 2026-10-14 | ✅ | `5b81f2e` | [log](logs/03-pin-datastorekit-at-v0.4.0.md) |
| 04 | [Rehearse through the new DatastoreKit (R1)](04-rehearse-through-the-new-datastorekit.md) | **R1**; the fingerprint both ways | Opus | ✍️ 2026-10-14 | ✅ | `ec61569` | [log](logs/04-rehearse-through-the-new-datastorekit.md) |
| 05 | [Pin `runregistrykit` at `v0.1.0`](05-pin-runregistrykit-at-v0.1.0.md) | the RunRegistryKit pin | Sonnet | ✍️ 2026-10-15 | ✅ | `3dcab1b` | [log](logs/05-pin-runregistrykit-at-v0.1.0.md) |
| 06 | [Adopt `runregistrykit`](06-adopt-runregistrykit.md) | the swap; closes the three `00-` issues assigned to it | Opus | ✍️ 2026-10-15 | ✅ | `740bafc` | [log](logs/06-adopt-runregistrykit.md) |
| 07 | [Guard the fingerprint format](07-guard-the-fingerprint-format.md) | **Q5** | Sonnet | ✍️ 2026-10-15; ⏸ 06 | ⬜ | — | — |
| 08 | [Rehearse through both packages (R2)](08-rehearse-through-both-packages.md) | **R2**; **RunRegistryKit G2** | Opus | ✍️ 2026-10-15; ⏸ 07 | ⬜ | — | — |
| 09 | Close the campaign | the records; the gates drafted; the registry issues drafted for RunRegistryKit; the hand-over to A3 v2 | Sonnet | ⏸ 08 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not yet · ✅ landed.

**Written in batches (README §2.1).** 01–02 now; 02a after R0, when the shared lock was adopted; 03–04 when E1's release was tagged (written 2026-10-14); 05–08 when E2's
release was tagged and its SGK checklist existed (written 2026-10-15); 09 after 08's review. The holds are for **method**,
not scope: every charter is fixed in README §2.2.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
(2026-10-12, used for 01); [`orchestrator/prompt-02.md`](orchestrator/prompt-02.md) (2026-10-12,
used for 02); [`orchestrator/prompt-02a.md`](orchestrator/prompt-02a.md) (2026-10-12, used for
02a); [`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) (2026-10-14, used for 03);
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) (2026-10-14, used for 04);
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) (2026-10-15, used for 05);
[`orchestrator/prompt-06.md`](orchestrator/prompt-06.md) (2026-10-15, used for 06).

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
| **K1** | 03 | the `datastorekit` pin moved to E1's release; the venv installed from it | ✅ |
| **L1**–**L4** (R1) | 04 | README §4.1 through E1's release; L4 against R0 | ✅ |
| **W1** | 04 | the fingerprint both ways, equal on every store (C3) | ✅ |
| **K2** | 05 | the `runregistrykit[stores]` pin; the venv installed from it | ✅ |
| **A1** | 06 | the client config file and `config/fingerprint.py` | ✅ |
| **A2** | 06 | every `RunRegistry` import outside `prompts/` rewritten | ✅ |
| **A3** | 06 | the 101 SGK tests moved (C5); `RunRegistry/` deleted | ✅ |
| **A4** | 06 | `CLAUDE.md`: the registry section, the package section, the roots | ✅ |
| **A5** | 06 | the worktree check (Q3) | ✅ |
| **V1** | 07 | the spec-digest guard and its breakages | ⬜ |
| **L1**–**L4** (R2) | 08 | README §4.1 through both packages; L4 against R1 | ⬜ |
| **G-RRK** | 08 | RunRegistryKit's G2 recorded as holding (§2) | ⬜ |
| **Z1** | 09 | the final measurements; the gate records drafted; the close | ⬜ |

## 2. The gates

| Gate | Holds when | Prompt | State |
|---|---|---|---|
| R0 | the start tree reproduces `39809dca…` under README §4.1 | 02 | ✅ holds (`74eff08`): `runregistry-extraction-02-reference-20261012T091921` and `-resumed-c-20261012T101145`, both `done`, both `39809dca…`, equal to `dk-and-rpk-migration` R2's recorded fingerprints |
| **E1** | DatastoreKit has tagged a release carrying the fingerprint algorithm and a version for the encoding and algorithm, with an SGK-facing note ([brief](datastorekit-fingerprint-brief.md)) | — | ✅ holds (2026-10-14, §5.5): `v0.4.0`, tag object `8e7c1d77…` on `c8570e84…`, CI run 38538726292 green on `c8570e8`; the note is DatastoreKit's `docs/adoption/secondarygwkit.md`, "Addendum: v0.4.0" |
| R1 | E1's release reproduces `39809dca…`, and DatastoreKit's fingerprint equals SGK's on every store | 04 | ✅ holds (`ec61569`): `runregistry-extraction-04-reference-20261014T154651` and `-resumed-c-20261014T161739`, both `done`, both `39809dca…`, equal to R0's recorded fingerprints; `fingerprint_both_ways.py` exits 0 on R1's two stores and R0's two, every field equal but `encoding_format` |
| **E2** | RunRegistryKit has tagged a release imported from 01's frozen tree (its G1), with the client config, a `[stores]` extra over E1's release, and SGK's adoption checklist ([brief](runregistrykit-extraction-brief.md)) | — | ✅ holds (2026-10-15, §5.10): `v0.1.0`, tag object `fd5ce480…` on `9f8ebe43…`, CI run 38754093289 green on `9f8ebe4` in all three jobs; the checklist is RunRegistryKit's `docs/adoption/secondarygwkit.md` at `aaab408` |
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

- **[00-open-issues-about-the-registry-code-stay-on-sgk-boards]** *(opened 2026-10-15 at E2, by
  the orchestrator, in writing 05–08; **Assigned (2026-10-15):** to 09, which drafts the hand-over,
  README §2.2)*
  - **The defect.** Four open issues on SGK's boards describe a defect or a decided change in code
    under `RunRegistry/`, which 06 deletes and which is RunRegistryKit's from `v0.1.0`. Neither
    the brief nor RunRegistryKit's campaign carried them: its `docs/OPEN_ISSUES.md` holds two
    issues, both about its own test fixtures, and its SGK checklist lists two of the four only as
    `docs/OPEN_ISSUES.md` lines that name `RunRegistry` (its appendix, item 7). By their hooks:
    - `[04-runs-do-not-say-which-machine-produced-them]` (`run-registry`): a manifest field and a
      `list` refusal, decided 2026-09-24 and not built;
    - `[00-a-copy-carries-its-sources-present-tense-fields]` (`store-retirement`): `copy_store`
      carries unknown fields verbatim;
    - `[01-cross-filesystem-move-advice-says-delete-by-hand]` (`store-retirement`), SGK's half:
      `store move` passes DatastoreKit's advice on (`_store_files`, `move_store`);
    - `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`
      (`a3-v2-readiness`): `finish` and the heartbeat share `status.json.tmp`.
  - **Impact.** After 06 each points at code that is no longer in this repository. By CLAUDE.md a
    package's defect is fixed there, in a release, so none of them can be fixed here, and
    RunRegistryKit does not know of them. The decided `machine` field is the one a later campaign
    would most likely look for. Nothing is wrong in a record today.
  - **Next step.** 09 checks each of the four against `runregistrykit` at `v0.1.0`: whether the
    defect is still there and where, by file and line. It drafts each for RunRegistryKit's
    `docs/OPEN_ISSUES.md`, and the user hands them over (README §6.2 C11). Each SGK board's entry
    then says where its issue went. 06 does not touch them.

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
  - **Closed (2026-10-15, prompt 06, `740bafc`).** `runregistrykit` finds the record root from the
    config's directory, and for a linked worktree takes the main checkout, the parent of `git rev-parse
    --git-common-dir` (Q3). `docs/runregistry-extraction/worktree_run_check.py`, run from a scratch
    worktree at the probe `ec781383730ff477c63430f0ea8fc10e106e0efc` with the main checkout's venv, began
    `runregistry-extraction-06-worktree-check-20261015T224836` in the main checkout's `var/runs/`, its
    manifest's `git_head` the probe and `git_dirty` false while the main checkout was at `85f273d` and
    dirty; the run ended `done`/0, and `runregistrykit list` shows it after the worktree was removed and
    pruned ([`measure/out/06-worktree.txt`](measure/out/06-worktree.txt)). The check's breakage left a
    second run there too, `…-worktree-check-breakage-20261015T225237`, `failed`/1.

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
  - **Closed (2026-10-15, prompt 06, `740bafc`), SGK's half.** Every fingerprint SGK takes now goes
    through `config/fingerprint.py`'s `DatastoreKitFingerprinter`, which records DatastoreKit's
    `encoding_format` beside the digest. `TestConfigFingerprint.test_the_fingerprinter_takes_the_golden_fingerprint`
    shows it (`encoding_format` 1, the fingerprint equal to `fingerprint_of(read_inventory(primary,
    factories), 6)` and to the new golden file, digest `6576b85d…`); the scratch store's sidecar, written
    by `runregistrykit store fingerprint --write`, carries it and compares empty
    ([`measure/out/06-both-ways.txt`](measure/out/06-both-ways.txt)). `FINGERPRINT_FORMAT` stays 6, and
    the format history in `config/fingerprint.py` says that an encoding change does not bump it. R2 (08)
    confirms it on real stores.

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
  - **Narrowed (2026-10-14, after 04's review):** the four stores are retired (§5.8). All 18
    sidecars under `var/datastores/` are tombstones, so there is no live store for the package's
    comparison to refuse. What remains is 06's: the recorded fingerprints in run directories and
    the six golden files, and the check at its gate.
  - **Closed (2026-10-15, prompt 06, `740bafc`).** The gate's check holds: all 18 sidecars under
    `var/datastores/` are tombstones, each read by `runregistrykit store show` exit 0 with a `retired`
    block and `problems: none`, and no `.sqlite` file remains there
    ([`measure/out/06-after.txt`](measure/out/06-after.txt)), so no live store's recorded fingerprint can
    be refused. The golden data: the six golden files moved unchanged to `ObjectFactories/tests/data/`,
    and a seventh, `full_store_fingerprint_runregistry-extraction-06.json`, is `09c`'s with
    `encoding_format` added (C13), pinned by `TestConfigFingerprint`. The run directories' recorded
    fingerprints stay as records; `fingerprint_both_ways.py`, rewritten, compares them field by field with
    `encoding_format` set aside, and on a `09c` record it measures the package's comparison as exactly one
    `encoding` entry (06-both-ways.txt). 08 makes that comparison with R0's and R1's.

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
  as the second positional argument (`RunRegistry/stores.py:1500-1508`), and so does
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

### 5.6 After prompt 03 (2026-10-14)

[`03`](03-pin-datastorekit-at-v0.4.0.md) landed at `5b81f2e`; its log is
[`logs/03-pin-datastorekit-at-v0.4.0.md`](logs/03-pin-datastorekit-at-v0.4.0.md). It opened and
closed no issue, so `docs/OPEN_ISSUES.md` is unchanged (107).
- **K1.** `requirements.txt:17` is `datastorekit @ ...@v0.4.0`, one line. The dry run said `Would
  install datastorekit-0.4.0` and nothing else; the install replaced 0.2.2 and nothing else.
  `package_provenance()` gives `datastorekit` 0.4.0 at `c8570e849334d09085bbd84876095fc9e22bd005`,
  `requested_revision` `v0.4.0`, not editable; `raypoolkit` unchanged (`eaef0ce6…`, `v0.2.0`);
  `pip freeze` differs by line 17 alone; `pip check` clean. `ENCODING_FORMAT` reads `1`.
  [`measure/out/03-pip.txt`](measure/out/03-pip.txt).
- **SGK's own fingerprint through the new pin.** R0's two stores, read-only: format 6,
  `39809dca…`, 21 classes, `recorded: matches`, exit 0; all twelve store files byte-identical
  (SHA-256, size, mtime) around the reads.
  [`measure/out/03-fingerprint.txt`](measure/out/03-fingerprint.txt).
- **The roots** at 02a's counts: AdaptiveLevin 32, ComputeTargets 693 (the known wall-clock
  failure alone, 0.147 s against 0.06 s, and 0.090 s and 0.118 s alone), CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped), ObjectFactories 271, RunRegistry 286, tools 91.
  [`measure/out/03-roots.txt`](measure/out/03-roots.txt).
- **The three addenda** tabled in [`measure/out/03-addenda.txt`](measure/out/03-addenda.txt):
  `v0.2.3` and `v0.3.0` ask nothing of SGK; `v0.4.0` asks nothing at the pin and names seven items
  at the call. **06's:** the ten `fingerprint_of` calls (two with `taken` second positionally),
  `test_the_shape` and the golden comparison, the retirement of R0's and R1's stores first (D6),
  the listing and `listing_lines` points, and format 6 to `config/fingerprint.py`. **04's:** the
  fingerprint both ways. Every claim the `v0.4.0` table makes about SGK's tree holds at `HEAD`.
- **Deviation worth knowing:** the prompt's `git grep -n 'store_fingerprint'` is not empty
  (`full_store_fingerprint`, two lines, unchanged by the install); the module-naming pattern is
  empty (log, deviation 3).
- **Venv.** Not recorded by any commit: `git revert` of 03 needs `pip install -r requirements.txt`
  afterwards (README §4.2).

**Orchestrator review of prompt 03** (2026-10-14, against `5b81f2e`, by
[`orchestrator/prompt-03.md`](orchestrator/prompt-03.md) §3). Dispatched from `3b6ef6a` to one
Sonnet subagent, after the seven roots were measured at `fd65055`. Nothing landed after the agent's
commit. **All six checks pass**, and no stop condition fired.
1. **Scope.** Seven files: `requirements.txt`, the four `measure/out/03-*` files, the log and this
   board. `requirements.txt`'s diff is line 17 alone. Nothing under `RunRegistry/`, `docs/`,
   `README.md`, `prompts/INDEX.md` or `orchestrator/`; the index untouched, at 107.
2. **The venv, measured here.** `package_provenance()` gives `datastorekit` 0.4.0 at `c8570e84…`,
   `requested_revision` `v0.4.0`, not editable, and `raypoolkit` exactly as before; `pip check`
   clean; `pip install --dry-run -r requirements.txt` now installs nothing. `03-pip.txt` has the
   dry run before the install.
3. **The fingerprint.** Both `store fingerprint` reads re-run here: format 6, `39809dca…`,
   `recorded: matches`. The twelve files of R0's two stores hash, by size and mtime too, exactly as
   the agent's listing, and identically before and after the re-run.
4. **The roots,** re-run here at `5b81f2e`, load 18–37: AdaptiveLevin 32, CosmologyModels 39,
   ObjectFactories 271, tools 91, LiouvilleGreen 148 (1 skipped) and RunRegistry 286 pass;
   ComputeTargets 693 with the one wall-clock failure, README §8's known flake, as at `fd65055`.
5. **Nothing else moved.** `find var -newer <marker>`, the marker set just before dispatch, is
   empty. `git status --short` is clean.
6. **The records.** The log has every §5.1 heading and classifies its three deviations; the
   addenda table gives every item a disposition and checks the `v0.4.0` table's SGK claims at
   `HEAD`. The commit message is in `CLAUDE.md`'s form. Deviation 3 is the prompt's fault, not the
   agent's: its pattern matches `full_store_fingerprint`. 04's script names the module in full.
- **The agent's one correction is right.** In `RunRegistry/stores.py`, `fingerprint =
  fingerprint_of(` is line 1500 and the call closes at 1508, as the addendum cites. §5.5 above and
  the orchestrator's note cited `:1499-1507`. §5.5 is corrected here; the note stays as it was
  used.
- **Residue fixed here:** "this commit" → `5b81f2e` on this board (§1 row, §5.6) and in the log;
  §5.6 and the header rewrapped to the board's width; this note added to the list of notes;
  `prompts/INDEX.md`'s row (4 of 10 landed).
- **Handed on.** 04, R1, is next, at `5b81f2e`'s tree: the first rehearsal under the lock, run on
  `datastorekit` 0.4.0. R0's two stores stay live for it, and after its review the orchestrator
  retires R0's and R1's four under D6.

### 5.7 After prompt 04 (2026-10-14)

[`04`](04-rehearse-through-the-new-datastorekit.md) landed at `ec61569`, with `HEAD` `89ff8b7` as
its parent; its log is
[`logs/04-rehearse-through-the-new-datastorekit.md`](logs/04-rehearse-through-the-new-datastorekit.md),
and the record is
[`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md)
§2. It opened and closed no issue, so `docs/OPEN_ISSUES.md` is unchanged (107).
- **R1 holds** (L1–L4, gate R1). Four registered builds at `89ff8b7`, whose code is `5b81f2e`'s,
  through `datastorekit` 0.4.0: `reference` and `resumed-c` `done`/0, `resumed-a` and `resumed-b`
  `killed`/143. Both stores are format 6, `39809dca…`, all 21 classes, equal to each other, to their
  sidecars, and to the fingerprints R0 recorded in `runregistry-extraction-02-reference-20261012T091921`
  and `-resumed-c-20261012T101145`. So `v0.2.3`, `v0.3.0` and `v0.4.0` move nothing a build writes or
  reads. No build log names `datastorekit`, and none holds a `DeprecationWarning` or `_timestamp`.
- **W1 holds.** `docs/runregistry-extraction/fingerprint_both_ways.py` exits 0 on R1's two stores
  and R0's two: the package's `fingerprint_of(inventory, 6)` equals SGK's in every field but
  `encoding_format`, keys and classes in the same order; the listings are 7542 lines each, equal but
  the first, which gains ` encoding_format 1`; the package's `compare_fingerprints(sgk, pkg)` is one
  `encoding` entry (`[00-recorded-fingerprints-lack-the-encoding-format]`, measured); each sidecar
  compares empty. Every store file was unchanged by it. Its five breakages each exit 1, naming what
  failed (log).
- **The manifests say what ran.** Every `packages` is `datastorekit` 0.4.0 at `c8570e84…`
  (`v0.4.0`) and `raypoolkit` 0.2.0 at `eaef0ce6…`, neither editable, compared as JSON. Every
  `git_dirty` is `false`, every `git_head` `89ff8b72…`, every `script_sha256` `b7935877…`.
- **L4 equals R0's.** The statistics counts are equal (291; numeric 109; mixed 26, 25 complete and 1
  minimal; WKB 156) in the reference and both resumed runs that print the block; the reference's
  block has R2's line order, not R0's (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`
  again). The reference's stage counts, every run's `** ` sequence, the printed lines and the
  registry fields are R0's.
- **The prune at open is exercised again.** The 40 % hold (41 s of a 102.9 s stage) left 207
  `GkSource` rows, 48 unvalidated, after `resumed-a`. `resumed-b` pruned exactly those 48 and
  computed 132 = 291 − 207 + 48. `resumed-c` computed 440 = 864 − 424 QSI rows.
- **The lock.** The first rehearsal under it: `acquire` before `reference`, `set-pid` at each launch,
  held through the readers and the guard, `release` at 16:33:42; every command exit 0, all in
  `lock_04.txt`. `status` reports it free, and `ray-processes` lists nothing.
- **The load** was lighter than at R0 (one-minute 11–100 at the checks, 161 during the readers). The
  reference took 13 m 57 s. Its QSI stage, 494.9 s, and `resumed-c`'s, 5 m 43.8 s, printed one
  progress line each, in R0's form.
- **Nothing else under `var/` moved.** The 26 files present before keep their SHA-256, mtime and size
  (`3e0e6bad…` over them, `023d139f…` over the 14 tombstones); the only new files are R1's ten store
  files and two sidecars.
- **The seven roots** are at 03's counts. ComputeTargets has two of README §8's wall-clock tests:
  `test_build_cost`, which passed three times alone, and `test_wall_time_per_object`, which failed
  three times alone at load 11–13.
- **For the orchestrator's waits:** a killed run's `status.json` says `killed` 5–6 s before its
  fingerprint and final heartbeat are written, by the launcher's and the registry's design. A wait
  should end on the pid's death (log, observation 1). No issue opened.
- **Handed on.** By D6 the orchestrator retires R0's and R1's four stores after this prompt's review,
  while SGK's own code still compares them. 05–09 wait on E2. 08 runs `fingerprint_both_ways.py`
  again, on the tree 06 leaves.

**Orchestrator review of prompt 04** (2026-10-14, against `ec61569`, by
[`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) §3). Dispatched from `89ff8b7` to one Opus
subagent, after the seven roots were measured and every rehearsal script, a scratch prototype of W1
and both interrupter dummy tests were run at `e150335`. The agent was resumed with `SendMessage`
after each of the four launches, with one wait per launch; every wait ended `ENDED finished`,
heartbeat never stale. Nothing landed after the agent's commit. **All fifteen checks pass**,
acceptance 1–7 are met, and no stop condition fired. **R1 holds.**
1. **Scope.** 26 files: 24 added (the log, the script, 22 `*_04*` outputs) and two modified (the
   verification document and this board). README, `prompts/INDEX.md`, `orchestrator/`,
   `docs/OPEN_ISSUES.md`, `RunRegistry/` and every other script are untouched. The verification
   diff is one hunk appending §2; §1 and the header are byte-identical.
2. **The runs**, read from each run directory. `reference` and `resumed-c` are `done`/0, both
   `39809dca…`; `resumed-a` and `resumed-b` are `killed`/143, with fingerprints `a410cce0…` and
   `8146dab8…`, 21 classes, problems `{}`. Each launch log ends with the matching driver exit status;
   each killed `stderr.log` holds the handler's two lines (`SystemExit(143)`, then `SystemExit(1)`,
   as at R0) and no traceback. Each interrupter sent one `SIGTERM`: 16:11:09, 41 s after it read the
   `GkSource` banner, and 16:16:22, 165 s after the QSI banner, the holds being 40 % of 102.9 s and a
   third of 494.9 s from R1's own reference.
3. **The manifests.** All four: `git_head` `89ff8b72…`, `git_dirty` false, `packages` equal as JSON
   to the note's record, `script_sha256` `b7935877…`, `--build --rehearsal` on all and `--resume` on
   `resumed-b` and `-c` only, the rehearsal `scope`, the right `results`, and a purpose naming R1,
   `89ff8b7`, `datastorekit` 0.4.0 at `c8570e8` and `39809dca`.
4. **Nothing stored was recomputed.** `stage_counts.py` on the four runs equals the committed table
   byte for byte, and its `reference` column equals R0's stage by stage. `GkSource` 207 − 48 + 132 =
   291; QSI 424 + 440 = 864; every stage finished before an interruption computes 0 next.
5. **The prune.** `prune_check.py` on the saved state after `resumed-a`: 48 checked, 48 absent, 0
   reused, 0 unvalidated. `resumed-b`'s integrity warnings name the same 48: shard 3's block lists
   15 rows, and Ray folds shards 0 and 1's 14 + 19 into `[repeated 33x across cluster]`
   (`stdout.log:13-130`, `:134-137`).
6. **The fingerprints.** `compare_stores.py`: equal, each store equal to its sidecar.
   `compare_with_recorded.py` on both primaries against R0's `reference` and `resumed-c` run
   directories: 0 differences in each of the four comparisons. Format 6, `39809dca…`, 21 classes.
7. **Integrity.** `store_soundness.py` exits 0, `=> clean`, on both: `integrity_check` `ok` on all
   ten files, 0 foreign-key rows, one `(1, '2025.1.1')` version row per shard, no inventory problem or
   `duplicate`. The census gives "no divergence" for both.
8. **The readers.** All seven logs: exit 0, every store file identical around each, the `nomodels`
   run printing the skip line. No `Traceback`, `ReadOnlyMiss`, `ReadOnlyWrite`,
   `ModuleNotFoundError`, unpickling error, `_timestamp` or `datastorekit` in any of them; the only
   `DeprecationWarning` is the `MatplotlibDeprecationWarning` in `extract_GkSource_data`, which R0's
   log holds five times. No reader was run here, since that would have meant taking the lock again.
9. **L4.** `policy_statistics.py` on `reference`, `resumed-b` and `resumed-c`: 14 lines each, every
   count R0's, each equal to its committed output; `resumed-b` and `resumed-c` equal R0's line for
   line, `reference` equal sorted (`minimal` first, R2's order). The `** ` lines of each run equal
   R0's same slug in order, numbers and prefix stripped (26, 23, 26, 26).
10. **The printed lines.** Completion and `Queue summary` counts (10, 7, 9, 10) and `(no compute
    items` counts (1, 1, 7, 9) equal R0's in every run. One progress line each in `reference`'s and
    `resumed-c`'s QSI stages. The SciPy `IntegrationWarning` is in `reference` and `resumed-c`, not
    `resumed-b`, which stored 424 QSI rows before its point against R0's 486. No build log holds a
    `Traceback`, `ModuleNotFoundError`, `DeprecationWarning`, `_timestamp`, or `datastorekit` outside
    a path.
11. **W1, by reading.** The script does the prompt's §2 items 1–5 in order. It reads the inventory
    as `fingerprint_store` does, passes `FINGERPRINT_FORMAT` from SGK's module, and removes only
    `encoding_format` before the comparison. Order is checked at the top level and over the classes
    (correction 5), and every differing path is printed with both values. It writes nothing, imports
    nothing that starts Ray, and is `black`-clean.
12. **W1, run here.** On the four stores: exit 0, 20 of 20 checks on each, output equal to
    `fingerprint_both_ways_04.txt`; all 24 store files identical by SHA-256, size and `mtime_ns`
    around it. The log's five diffs, re-applied to scratchpad copies: each exits 1, with 18, 18, 15,
    18 and 19 of 20 holding, and names what the prompt says it must.
13. **The lock.** `lock_04.txt`: `acquire` 15:46:42, exit 0, nine seconds before `reference`'s
    manifest (15:46:51); `set-pid` 3243, 11561, 17571 and 20688, each exit 0 within a second of its
    launch's check; `release` exit 0 at 16:33:42, after the last reader's log ended (16:32:17) and
    the guard. Now `status` is free and `ray-processes` lists nothing; `~/ScienceData/` holds
    `ChamPBH-rehearsals/` alone, as before.
14. **`var/`.** The 26 files present before dispatch hash to `3e0e6bad…` as before; the only new
    files under `var/datastores/` are R1's ten store files and two sidecars. Nothing under `var/`
    is newer than the pre-dispatch marker but R1's runs, stores and working directory, and the
    directories holding them. The guard refused, exit 1.
15. **Records and suites.** The board and the log are as the note asks; the log has every §5.1
    heading and 18 classified deviations, the note's eight corrections among them, no `UNINTENDED
    DRIFT`; verification §2 has §2.1–§2.9 in §1's shape, plus §2.1.3 for the lock and §2.10 for W1.
    At `ec61569`, load 5–8: all seven roots pass at 03's counts, ComputeTargets 693 with **no
    failure**. The two wall-clock tests that failed in the agent's run, at higher load, passed here.
- **The note held.** All eight corrections were followed. The race in the first wait was the
  note's: its wait script ended on `state == "finished"`, which a killed run's `status.json` reaches
  5–6 s before `finish_after_exec` adds the fingerprint. The later three waits ended on the pid's
  death. The agent traced the ordering to the launcher's handler and `Run.finish`, by design; the
  wait in 08's note should end on the pid's death.
- **Residue fixed here:** "this commit" → `ec61569` on this board (§1 row, §2 R1 row, §5.7) and in
  the log (header and hand-off); this note added to the list of notes; `prompts/INDEX.md`'s row (5
  of 10 landed).
- **Handed on.** R1 holds, so the three DatastoreKit releases move nothing a build does, and the
  package's `fingerprint_of` is SGK's on real stores. The four stores are retired next, under D6, in
  a commit of their own. **For 08's prompt**, written after E2: `fingerprint_both_ways.py` imports
  `RunRegistry`, which 06 deletes, and R0's and R1's stores will be gone. So "08 runs it again"
  (README §4) cannot mean this script as it stands. 08 compares through recorded fingerprints field
  by field (README §2.2, 08, acceptance 1), and its prompt must say what, if anything, of W1 it
  repeats.

### 5.8 R0's and R1's stores retired (2026-10-14, D6)

**The four rehearsal stores retired, by the user's ruling D6 and instruction.** Each was retired with
`python -m RunRegistry store retire` at `e7d18bd`, after a dry run of each, while SGK's own code
still compares their recorded fingerprints (`[00-recorded-fingerprints-lack-the-encoding-format]`):
- `var/datastores/runregistry-extraction-02-reference.sqlite` and `-02-resumed.sqlite` (R0);
- `var/datastores/runregistry-extraction-04-reference.sqlite` and `-04-resumed.sqlite` (R1).

**The dry runs** exited 0. In each, a fresh read-only fingerprint matched the recorded
`39809dca…`, and the only references were the store's own building runs: one for each `reference`
and three for each `resumed`, and no sidecar.

**The retirements** exited 0, between 16:59:25 and 16:59:39. Five files went per store (the primary
and four shards), 281 MB in all. Each sidecar stays as its tombstone, with `completed` set. The
eight run directories stay as the record. The outputs are under
`var/runregistry-extraction-04/retire/` (`dry-run-*.log`, `retire-*.log`).

**Afterwards** all 18 sidecars under `var/datastores/` are tombstones, and no `.sqlite` file
remains there. The issue above is narrowed, and `docs/OPEN_ISSUES.md`'s row with it (the count
stays 107).

### 5.9 The plan amended: what 08 repeats of W1 (2026-10-14)

At the user's request, after 04's review (§5.7, "Handed on"), README §4 now says what 08 repeats of
W1. The script compares SGK's `fingerprint_of` with the package's. 06 deletes the first, and R0's
and R1's stores are retired (§5.8), so the script cannot run at 08 as 04 left it.
- **What 08 repeats.** 08 repeats W1's claim across stores. The package's fingerprint of each R2
  store equals R0's and R1's recorded fingerprints in every field but `encoding_format`. That is
  08's acceptance 1, which is unchanged.
- **06 rewrites the script into that comparison**, under C6:
  `fingerprint_both_ways.py <primary>... --against <run dir>...`. The checks of W1 that still have
  two sides are kept, and 06 names the script's breakages.
- **`compare_with_recorded.py` is run at 08 on R2's own stores only.** Through the package, every
  fingerprint SGK's code recorded compares as one `encoding` entry.

The charters in README §2.2 are not changed. 06's acceptance 1 already requires the script's
`RunRegistry` imports to go, and 08's acceptance 1 already makes the comparison field by field.

### 5.10 E2 holds (2026-10-15)

RunRegistryKit closed its `extraction` campaign and released `v0.1.0`, and the user asked this
session to inspect it. The orchestrator checked it against RunRegistryKit's repository and this
tree at `14bcac0`, read-only, and the user then asked for E2 to be recorded and 05–08 written.
- **The tag.** `v0.1.0` is an annotated tag, object `fd5ce4806a3cea09a27d92d5d811600c62dac376`, on
  commit `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`, the same locally and on `origin`. GitHub has no
  Release object for it (`gh release view v0.1.0`: not found). Neither DatastoreKit nor RayPoolKit
  has one either, and the pin needs only the tag. `pyproject.toml` there gives `version = "0.1.0"`,
  `requires-python = ">=3.12"`, no core dependency, and `[stores]` = `datastorekit>=0.4.0`.
- **CI.** Run 38754093289, on `9f8ebe4` itself, passed all three jobs: `suite (low)`, `suite
  (high)` and `core (no store layer)`. Every later push to `main` is green too, to
  `caad864`.
- **After the tag.** Nothing under `runregistrykit/` and nothing in `pyproject.toml` changed between
  `9f8ebe4` and `caad864`. The later commits are records, the verification, and the adoption
  checklists.
- **The checklist** is `docs/adoption/secondarygwkit.md`, written at `aaab408` (extraction prompt
  07), a commit after the tag and unchanged since. It describes the package at the tag, and it
  measured this repository at `14bcac0`, which is `HEAD` here: its lines and counts are current.
  Checked here: the 101 staying tests (28, 26, 15, 18, 10, `TestGolden` 3, `TestDrivers` 1); 21
  specs over the 38-name registry; and the 13 files outside `RunRegistry/` that name it.
- **G1**, as RunRegistryKit's board records it: the source side holds (`git ls-tree -r a45aefc --
  RunRegistry/` equals §2.1, 25 rows, unchanged to `14bcac0`). The package side holds for the files
  its prompts 01 and 02 imported. The three test modules its 04 ported were rebuilt on a store of
  its own and are its tests now. That does not touch SGK.
- **The dry run.** `pip install --dry-run -r requirements.txt` with the `runregistrykit[stores]`
  line added gave `Would install runregistrykit-0.1.0` and nothing else. `datastorekit` 0.4.0
  satisfies the extra.
- **Its open issues**, two, are about its own test fixtures' use of `ShardedPool` internals and a
  message-text match on DatastoreKit's. Neither asks anything of SGK.
- **What the checklist leaves to SGK**, decided here as choices of method (README §6.2 C12–C18), so
  that the user can reverse any: the drivers' runs root, the golden file, the test helpers, the
  probe commit, the rewrite boundary, `config/fingerprint.py`'s form, and its tests. The two that
  matter most:
  - `test_store_fingerprint.py:378`'s `fingerprint_of(self.baseline_inventory, taken)` binds at
    `v0.1.0` and puts `taken` on the format. 06 names it.
  - Two tests patched `DEFAULT_ROOT`, and neither seam was tested. 06 adds an assertion and a test
    (C12).
- **Since 01's baseline**, R0 and R1 added 8 runs and 4 sidecars. The registry shows 37 runs, none
  `running` or stale, and 18 sidecars, all tombstones. 06's acceptance 3 is amended to compare
  before and after the swap, with 01's 29 inside it; 08's acceptance 4 counts 06's two worktree
  runs (README §2.2).
- **Opened:** `[00-open-issues-about-the-registry-code-stay-on-sgk-boards]` (§3), assigned to 09.
  The index goes to 108.
- **Written:** prompts [05](05-pin-runregistrykit-at-v0.1.0.md), [06](06-adopt-runregistrykit.md),
  [07](07-guard-the-fingerprint-format.md) and [08](08-rehearse-through-both-packages.md). The plan
  is amended where E2 made it inexact: the README's status line, §2's table, §2.1, §2.2 (05's tag,
  06's acceptance 2–4, 08's acceptance 4, 09's hand-over), §4 (what 06 ships), and §6.2 (C12–C18).
- **RunRegistryKit's records**: its board records E2 as holding, from its side. G2 is SGK's to show
  (08), and its record there is drafted by 09.

### 5.11 After prompt 05 (2026-10-15)

[`05`](05-pin-runregistrykit-at-v0.1.0.md) landed at `3dcab1b`; its log is
[`logs/05-pin-runregistrykit-at-v0.1.0.md`](logs/05-pin-runregistrykit-at-v0.1.0.md). It opened and
closed no issue, so `docs/OPEN_ISSUES.md` is unchanged (108).
- **K2.** `requirements.txt:81` is `runregistrykit[stores] @ ...@v0.1.0`, one added line. The dry run
  said `Would install runregistrykit-0.1.0` and nothing else (it printed the tag object `fd5ce480…`,
  as pip does); the install added that and nothing else. `direct_url.json`: commit
  `9f8ebe431503fb8c79ee1407cc460501f1ca2d92`, `requested_revision` `v0.1.0`, no `dir_info`, so not
  editable. `pip freeze` differs by one added line; `pip check` clean; `package_provenance()`
  byte-identical before and after (`datastorekit` 0.4.0 at `c8570e84…`, `raypoolkit` 0.2.0 at
  `eaef0ce6…`; it does not name `runregistrykit`, which enters `packages` at 06).
  [`measure/out/05-pip.txt`](measure/out/05-pip.txt).
- **Nothing imports it, and importing it is light.** `import runregistrykit,
  runregistrykit.fingerprint` loads none of `ray`, `sqlalchemy` or `datastorekit` (`[]`). No code
  names the package; the only non-pin hit outside `prompts/` and `docs/` is `CLAUDE.md:114`'s prose.
- **The roots** at 03's counts, no failure: AdaptiveLevin 32, ComputeTargets 693, CosmologyModels 39,
  LiouvilleGreen 148 (1 skipped), ObjectFactories 271, RunRegistry 286, tools 91.
  [`measure/out/05-roots.txt`](measure/out/05-roots.txt).
- **The checklist's item 1** holds against the venv: Python 3.12.15, `ray` 2.43.0, `SQLAlchemy`
  2.0.39, `datastorekit` 0.4.0, `raypoolkit` 0.2.0.
- **Deviation worth knowing:** the prompt's `git grep` outside `prompts/` and `docs/` is not empty
  (`CLAUDE.md:114`); excluding `CLAUDE.md` it is empty before and gives only the pin after (log,
  deviation 2).
- **For the orchestrator:** README §5 rule 9 says `packages` names `runregistrykit` "(05)"; it enters
  at 06, as `RECORDED_DISTRIBUTIONS` is frozen (log, "Observations not acted on").
- **Venv.** Not recorded by any commit: rolling 05 back needs `pip uninstall -y runregistrykit`
  after the revert (README §4.2).

**Orchestrator review of prompt 05** (2026-10-15, against `3dcab1b`, by
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md) §3). Dispatched from `e381906` to one
Sonnet subagent, after the seven roots were measured at `17512cb`. Nothing landed after the agent's
commit. **All six checks pass**, and no stop condition fired.
1. **Scope.** Five files: `requirements.txt`, `measure/out/05-pip.txt` and `05-roots.txt`, the log
   and this board. `requirements.txt`'s diff is the one added line, between `rpds-py` and `rsa`.
   Nothing under `RunRegistry/`, `docs/`, `README.md`, `prompts/INDEX.md` or `orchestrator/`; the
   index untouched, at 108.
2. **The venv, measured here.** `direct_url.json`: `commit_id` `9f8ebe43…`, `requested_revision`
   `v0.1.0`, no `dir_info`. The installed files are the five modules, with no
   `runregistrykit/tests/`. `pip freeze` against the agent's 2.1 freeze differs by the one line;
   `datastorekit` and `raypoolkit` unchanged; `pip check` clean; `pip install --dry-run -r
   requirements.txt` now installs nothing. `05-pip.txt` has the dry run before the install.
3. **The imports.** The `sys.modules` check prints `[]`; `package_provenance()` equals the agent's
   2.1 reading. `git grep -n runregistrykit -- . ':!prompts' ':!docs'` gives the pin and
   `CLAUDE.md:114`, prose in the freeze section; no code names the package.
4. **The roots,** re-run here at `3dcab1b`, load 9–22: all seven pass at 03's counts with no
   failure, as at `17512cb` and in the agent's run.
5. **Nothing else moved.** `find var -newer <marker>`, the marker set just before dispatch, is
   empty. `git status --short` is clean. RunRegistryKit's checkout is unchanged, at `caad864`,
   clean.
6. **The records.** The log has every §5.1 heading and classifies its three deviations, no
   `UNINTENDED DRIFT`; checklist item 1's five versions hold; README §5 rule 9 is under
   "Observations not acted on", as the note asked. The commit message is in `CLAUDE.md`'s form.
- **Deviation 2 is the prompt's and the note's, not the agent's.** Both said the `git grep` outside
  `prompts/` and `docs/` would be empty but for the pin; `CLAUDE.md:114` was always going to match.
  06's checks of "nothing names `RunRegistry`" should scope to code, or name `CLAUDE.md`'s prose.
- **One explanation corrected.** The log says `pip show` lists no `Requires:` partly because
  `datastorekit` was already installed. The installed `METADATA` carries `Requires-Dist:
  datastorekit>=0.4.0; extra == "stores"`; `pip show` lists unconditional requirements only, so the
  line would be empty either way. The install is right; the log stays as written.
- **Residue fixed here:** "this commit" → `3dcab1b` on this board (§1 row, §5.11) and in the log
  (commit line and rollback); the header rewrapped to the board's width; this note added to the list
  of notes; README §5 rule 9's "(05)" → "(06)"; `prompts/INDEX.md`'s row (6 of 10 landed).
- **Handed on.** 06, the swap, is next, at `3dcab1b`'s tree, with `runregistrykit` 0.1.0 installed
  and nothing yet importing it. Its note is written after this review.

### 5.12 After prompt 06 (2026-10-15)

[`06`](06-adopt-runregistrykit.md) landed at `740bafc`, with `HEAD` `85f273d` as its parent; its
log is [`logs/06-adopt-runregistrykit.md`](logs/06-adopt-runregistrykit.md). It closed three issues
(§4) and opened none, so `docs/OPEN_ISSUES.md` goes from 108 to 105.
- **A1.** `runregistrykit.toml` at the root, the checklist's item 3; `config/fingerprint.py` with the
  format history moved verbatim, a dated paragraph, `FINGERPRINT_FORMAT = 6` and `fingerprinter`, a
  `DatastoreKitFingerprinter` over `config.datastore.factories`.
- **A2.** Every `RunRegistry` import, literal, comment and command outside `prompts/` and dated `docs/`
  prose rewritten, in twelve files; `fingerprint_both_ways.py` rewritten whole as README §4's
  comparison across stores. The two seams, `scoped_pipeline_run.register(..., runs_root=None)` and
  `quadsource_atol_sweep.prepare(..., runs_root=None)`, each tested (C12). `git grep -nP
  'RunRegistry(?!Kit)'` outside `prompts/` and `docs/*.md|*.txt` prints `CLAUDE.md`'s one sentence and
  the format history's one line (the orchestrator's ruling 1); `import RunRegistry` fails.
- **A3.** The 101 tests moved, counted first (28, 26, 15, 18, 10, 3, 1): the five driver modules to
  `RunDrivers/tests/` by `git mv`, with `registry_fixtures.py` (C14) and `test_finish_calls.py`;
  `TestGolden` to `ObjectFactories/tests/test_store_fingerprint_golden.py` with `TestConfigFingerprint`
  (C18); the six golden files to `ObjectFactories/tests/data/`, and a seventh beside them, `09c`'s with
  `encoding_format` (C13), digest `6576b85d…` unmoved. `RunRegistry/` deleted, from disk too.
- **A4.** `CLAUDE.md`: the package section names RunRegistryKit at `v0.1.0` with `[stores]` (and
  `datastorekit` at `v0.4.0`, ruling 8); the registry section points at `runregistrykit` and its README;
  the freeze section is gone; the roots list `RunDrivers/tests`.
- **A5.** The worktree check: from a scratch worktree at the probe `ec781383…`, the run
  `runregistry-extraction-06-worktree-check-20261015T224836` is in the main checkout's `var/runs/` with
  the probe's `git_head` and `git_dirty` false, `done`, and listed after the worktree is gone. Breakage
  8 left `…-worktree-check-breakage-20261015T225237`, `failed`. These two are the only things 06 wrote
  under `var/`; 08's acceptance 4 counts them.
- **The registry reads as before** (§7.1): `list` gives the same 37 runs, and `store show` the same 18
  blocks byte for byte, once the tool's name and the age column are set aside.
- **The roots** at the staged tree, no failure: AdaptiveLevin 32, ComputeTargets 693, CosmologyModels
  39, LiouvilleGreen 148 (1 skipped), **ObjectFactories 277**, **RunDrivers 99**, tools 91.
- **The checklist's measure** at the probe: 0 imports, 0 literals, 0 `-m RunRegistry`.
- **The probe** is `ec781383730ff477c63430f0ea8fc10e106e0efc`, tree
  `f2e20c2ad7ff0165d69dbc3051066b7a9d42a30b`; `740bafc` differs from it by the records alone.
- **Deviations worth knowing** (log): the golden file is written with sorted keys, as `09c` was; the
  both-ways script compares key order only where a record keeps it, since every record is written
  with `sort_keys`; and breakage 4 fails `TestConfigFingerprint` test 1 at its config assertion, before
  the loader's refusal, which was shown directly instead (log, deviation 9, `UNINTENDED DRIFT`).
- **RunRegistryKit G2** (§2): 06's half holds (nothing imports `RunRegistry`, the pin is installed, the
  roots pass, `list` gives 01's runs); R2 (08) is the other half.

**Orchestrator review of prompt 06** (2026-10-15, against `740bafc`, by
[`orchestrator/prompt-06.md`](orchestrator/prompt-06.md) §3). Dispatched from `85f273d` to one Opus
subagent, after the seven roots were measured at `29a3f2d`. Nothing landed after the agent's commit.
**All twelve checks pass**, and no stop condition fired.
1. **Scope.** `git show --name-status 740bafc` is the note's list: 12 `R` (the five driver modules,
   the six golden files at 100%, and the two empty `__init__.py`, which git pairs: log, deviation
   10), 13 `D` under `RunRegistry/`, the added client files, tests and scripts, and the thirteen
   modified files. Not `README.md`, `prompts/INDEX.md`, `orchestrator/`, `requirements.txt` or any
   `.md`/`.txt` under `docs/` but `OPEN_ISSUES.md`. The probe `ec781383…` is a commit whose parent
   is `85f273d`; `git diff --stat ec781383 740bafc` is ten record files.
2. **Nothing names `RunRegistry`.** The grep at `740bafc` prints `CLAUDE.md:82` and
   `config/fingerprint.py:31` (ruling 1) and nothing else; `RunRegistry/` is not on disk and
   `import RunRegistry` is `ModuleNotFoundError`. RunRegistryKit's measure, re-run here at
   `740bafc`: 0 import statements, 0 literals, 0 `-m RunRegistry`, 421 files.
3. **The moves, by diff.** Every hunk of the five modules is one §3.2 names, plus the removal of the
   `mock` imports the patches left unused. `test_prune_on_resume.py` is a 100% rename. `TestDrivers`
   and `_finish_calls` are byte-identical to `a45aefc`'s; `_FullStores`, `without_taken` and
   `TestGolden` differ from `a45aefc`'s only by the format passed, the `encoding_format` assertions,
   `_fingerprint_problems`' module and `maxDiff`. `registry_fixtures.py` copies the two helpers, and
   names both sources.
4. **The counts.** `RunDrivers` 99, `ObjectFactories` 277, by run (check 11).
5. **The client files.** The config is item 3's five keys under one comment line. The history in
   `config/fingerprint.py` is `a45aefc:RunRegistry/stores.py:194-205`, all twelve lines verbatim,
   then one dated paragraph. Importing it after `config.datastore` adds `config.fingerprint`,
   `runregistrykit`, `.client`, `.fingerprint` and `tomllib`: no DatastoreKit module, no Ray.
6. **The seams.** Each driver's diff is the keyword, its forwarding, a docstring sentence (log,
   deviation 7), §2.1's rewrites and §2.3's literals; `main()`'s call is unchanged. Breakages 5 and
   6, re-applied here, each fail their test.
7. **The golden.** `diff 09c 06` is the one added line `"encoding_format": 1,`; test 3 pins the rest
   to `09c` and both digests to `6576b85d…`.
8. **The registry.** `06-before.txt` and `06-after.txt`, with command lines, the header and the
   AGE column removed, are identical: 37 runs, 18 `store show` blocks, 19 exit statuses 0, and 69
   "by results" lines in each, so the block the note singled out reads as before. 01's 29 are all
   present with their states. `runregistrykit list` now: 39, the two new ones `done` and `failed`.
9. **The worktree run.** Both manifests read here: `git_head` the probe; `git_dirty` `false` for the
   check, `true` for the breakage; `script` and `cwd` in the scratch worktrees; `packages` naming
   `runregistrykit` 0.1.0 at `9f8ebe43…` first, then `datastorekit` and `raypoolkit` at their pins,
   none editable. The check's `script_sha256` `5aff8eda…` is the committed script's. `find var
   -newer <marker>` lists those two run directories and nothing else; `git worktree list` shows this
   checkout alone.
10. **The breakages and the script.** `fingerprint_both_ways.py` on a fresh scratch set built as §5
    says (fixture store, `store create`, `store fingerprint --write`, a run directory holding `09c`):
    exit 0, 8 of 8 checks. Breakages 1–6 and 7a re-applied to the working tree and reverted, 7b to a
    scratch record: each fails exactly as the log says (1: three tests; 2: test 2 errors with
    `StoreSchemaMismatch`, test 1 fails; 3: three tests; 4: test 1; 5, 6: the new assertion and the
    new test; 7a, 7b: exit 1 naming `encoding_format` and the tag set's `count`). Breakage 8 by its
    record (check 9). The tree and `var/runs/` were unchanged after.
11. **The roots,** re-run here at `740bafc`, serially, load 6–18: AdaptiveLevin 32, CosmologyModels
    39, ObjectFactories 277, tools 91, LiouvilleGreen 148 (1 skipped), RunDrivers 99, ComputeTargets
    693, all OK. `RunDrivers` takes 3 s, where `RunRegistry/tests` took 73 s: the 185 that left were
    the package's.
12. **The records.** `CLAUDE.md`: the package section names the three pins truly (ruling 8), the
    registry section points at `runregistrykit` and its README with the six rules unchanged, the
    freeze section is gone, the roots list `RunDrivers/tests`; nothing else moved. The log has every
    §5.1 heading, eleven classified deviations and a twelfth saying there is no other and the rulings logged. The board's A1–A5, §4 and
    §5.12; `docs/OPEN_ISSUES.md` at 105 rows, counted, with §1.21's prose true.
- **Breakage 4 is the prompt's inconsistency, not drift.** §3.3 orders test 1's assertions with the
  config's `stores` table first, and §8 expected the loader's refusal; the mutation can only fail at
  the first. The test catches a config naming the class, and the loader's refusal of a class is the
  package's behaviour, tested there. The agent's `UNINTENDED DRIFT` stays in the log as written; no
  change is made.
- **The both-ways script's order check (deviation 8) is right.** Every record is written with
  `sort_keys`, so a recorded fingerprint keeps neither its keys' order nor its classes'; only the
  tag-set lists keep theirs. 08 runs the script as it stands.
- **Seen in passing, not opened:** `python -m runregistrykit list | head` ends with a
  `BrokenPipeError` traceback once `head` closes the pipe. Cosmetic, and RunRegistryKit's; it is
  for 09 to carry over with the four issues, if the user wants it carried.
- **Residue fixed here:** "this commit" → `740bafc` on this board (§1 row, §4's three closures,
  §5.12) and in the log; this note listed; `prompts/INDEX.md`'s row (7 of 10 landed, 2 rows). Of the
  log's observations: `compare_with_recorded.py` now reads a `null` `taken` as empty instead of
  raising (observation 3); the sweep driver's `--force` message no longer says `store retire` does
  not exist yet (observation 4); and `CLAUDE.md`'s formatting line says that the tree outside
  `prompts/` is clean under `black --check`, the 49 measure scripts under `prompts/datastore-generic/`
  being records (observation 6).
- **Handed on.** 07, the guard, is next, at `740bafc`'s tree. Its note is written after this review.
