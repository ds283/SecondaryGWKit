# Store retirement campaign — implementation state

**Last updated:** 2026-09-25 · **Status: 4 of 5 prompts landed (01, 02, 03, 04). 05 is written and
ready.** The user approved D0 and D3–D7 as worded on 2026-09-25 (README §6.2), which released 03
and 04. 05 was written the same day, after 01–04 had landed. 01 and 02 are independent of each
other; 03 follows 01, and 04 follows 03. D4 was narrowed when 03 was written: an unreadable
`shards` table is refused even under `--without-fingerprint` (README §6.2).

The campaign was opened on 2026-09-25, when the user decided that a store is removed by a registry
operation that retires it, and never by `rm`. The primary and its shards go, and the sidecar stays
behind, marked retired (README §6.1). The two stores it exists to retire are the tolerance-sweep
store and the pre-resume backup of the A3 baseline. Neither holds anything that is not held
elsewhere ([`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) §3). The audit
found five things in the way:
- a sidecar with no primary reads as broken (§2.1);
- the history format cannot record a retirement (§2.2);
- the sweep's `--prepare` would overwrite the sweep store's tombstone (§2.6);
- the backup's primary names the live store's shards, so a deletion that followed a stored record
  would destroy the live A3 store (§2.7);
- `CLAUDE.md` says the registry does not delete (§2.8).

The audit opened three issues (§3). Two issues from `datastore-portability` were assigned here. One
of them, it found, also blocks `--build --resume` of the A3 v2 store.

**Campaign:** [`README.md`](README.md) ·
**Code:** `Datastore/SQL/ShardedPool.py`, `docs/handover/quadsource_atol_sweep.py`, `RunRegistry/stores.py`, `RunRegistry/__init__.py`, `RunRegistry/__main__.py`, `CLAUDE.md` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.14

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. An
> issue owned by another board is moved to **that** board's §4, and its row is deleted from the
> index. The index is an index: one line per issue, pointing here. Where the two disagree, this
> board is right. See `CLAUDE.md`.

### Decisions — all approved by the user as worded, 2026-09-25

| Decision | Decided (README §6.2) | Blocked, until decided |
|---|---|---|
| **D0** — amend `CLAUDE.md:52`'s "does not … delete" | "It records; it does not schedule, supervise, restart or lock. It deletes nothing but a store's own files, and those only through `store retire`, which a person runs and which leaves the store's sidecar behind as its record. It never deletes a run directory, a sidecar or any other record." Prompt 01 writes this into `CLAUDE.md:52`, exactly | **01**, and so 03–05 |
| **D3** — the retirement's reason | required, as `purpose` is on create and copy | 03 |
| **D4** — a store that cannot be fingerprinted | `--without-fingerprint`, which still needs the reason and records the error. A hot journal is refused even under it | 03 |
| **D5** — references to a retired store | never rewrite records of the past, which resolve to the tombstone; correct present-tense unknown fields with a new `store amend`; `retire` reports every reference it finds | 03, 04 |
| **D6** — reuse of a retired name | never. `begin` refuses it; a reappeared primary is a reader problem; every other operation refuses a tombstone | 03 |
| **D7** — a delete in `tools/sharded_store.py` | none | 01 (as a stop condition) |

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Delete a closed store](01-delete-a-closed-store.md) | **R1**–**R3** | Opus | ✍️ yes, 2026-09-25 | ✅ 2026-09-25 | *"Delete a closed sharded store's own files through the resolver"* | [`logs/01-…`](logs/01-delete-a-closed-store.md) |
| 02 | [The sweep prepares through the registry](02-the-sweep-prepares-through-the-registry.md) | **R4**–**R5** | Sonnet | ✍️ yes, 2026-09-25 | ✅ 2026-09-25 | *"Make quadsource_atol_sweep prepare and check through the registry"* | [`logs/02-…`](logs/02-the-sweep-prepares-through-the-registry.md) |
| 03 | [Retire a store](03-retire-a-store.md) | **R6**–**R8** | Opus | ✍️ yes, 2026-09-25 | ✅ 2026-09-25 | *"Retire a closed store and keep its sidecar as a tombstone"* | [`logs/03-…`](logs/03-retire-a-store.md) |
| 04 | [Amend an unknown field](04-amend-an-unknown-field.md) | **R9** | Sonnet | ✍️ yes, 2026-09-25 | ✅ 2026-09-25 | *"Add amend_sidecar and store amend, for one unknown field"* | [`logs/04-…`](logs/04-amend-an-unknown-field.md) |
| 05 | [Retire the two stores](05-retire-the-two-stores.md) | **R10**–**R12** | Opus | ✍️ yes, 2026-09-25 | ⬜ after 04 | — | — |

**Orchestrator review of prompt 04 (2026-09-25).** All ten checks in `orchestrator/prompt-04.md`
§3 passed on `66617c9`, from one dispatch. There are two findings, both opened in §3, and a
correction to log 04's account of two mutations.
- **Scope.** Only `RunRegistry/stores.py`, `RunRegistry/__main__.py`, the new `test_store_amend.py`,
  the log and this board changed. `Datastore/`, `tools/`, `CLAUDE.md`, `RunRegistry/__init__.py`,
  `store_fixtures.py` and every existing test are untouched, and so is the index. The `stores.py`
  hunks are in the module docstring, the constants, `_history_problems`, the new
  `_amend_slot_problems`, `_update_sidecar`'s docstring and the new `amend_sidecar` at the end of
  the file. None falls inside `retire_store`, `copy_store`, `move_store`, `_prepare`,
  `fingerprint_store` or the reader.
- **Only unknown fields.** The refusal tests `field in KNOWN_FIELDS`, before any write, and names
  the owner from `_FIELD_OWNERS`. A test iterates over `KNOWN_FIELDS`, `retired` included, and a
  field missing from `_FIELD_OWNERS` would raise `KeyError` there. Test 1 checks `history[:-1]`
  and every other field value-identical. Nothing retakes the fingerprint.
- **Nothing lost.** The markers are a wrapper, `{"present": true, "value": …}` or
  `{"present": false}`, one level above the value, so no field value can be mistaken for one.
  `before` is a deep copy. Test 7 reads two amendments back as a sequence. **Finding:** no
  committed test amends a field whose value has the marker's own shape. R9, log 04 and the
  implementer's report all said one did. A probe in the scratchpad amended
  `{"present": false}` to `{"present": true, "value": 1}` and read both back unambiguously. So the
  design holds and the gap is only the test:
  `[04-no-test-amends-a-value-shaped-like-the-marker]`.
- **The refusals.** Every refusal in the prompt's §2.1 is covered, including complete and
  incomplete tombstones with the tombstone message, and alive and stale runs. The running-run
  check is `_running_runs_naming("amend", [primary], fields["store_id"], runs_root)`. Amend
  imports no `ShardedPool`, and `assertOnlySidecarChanged` checks that `tree_state` changes only in
  the sidecar. **Finding, beyond the checks:** the "identical value" refusal compares with
  Python `==`, where `True == 1 == 1.0` at any depth. A probe confirmed that `1` cannot be amended
  to `true` or `1.0`, nor `{"k": 1}` to `{"k": true}`:
  `[04-amend-calls-true-1-and-1-0-identical]`.
- **The history rule.** `amend` may stand only after index 0 and before `retire`. Its four keys
  are required on it and are a problem on a `copy`. Its `from` and `to` must both be null. The
  rules for `copy`, `move` and `retire`, and `_registry_problems`, are unchanged, and
  `test_store_retire.TestHistoryRule` passes unmodified.
- **Copy and move** carry the amended field and its entry (`TestCopyAndMove`). `copy_store`
  appends its own entry to the carried history (`stores.py:1091`), so the `copy` entry comes after
  the `amend` by construction. No test asserts that order.
- **Docstrings.** `_update_sidecar` counts five uses, which is true: adopt, the move's temporary
  sidecar, the fingerprint, retire's two writes, and amend. The `stores.py` operations list, the
  in-place-update list and the format table's `history` row carry `amend`, its keys, its markers
  and its null `from`/`to`. `__main__.py` adds `amend`. "Every operation but a retirement's
  completion refuses a tombstone" is still true.
- **Tests.** The new module passed twice, 21 tests. Every `amend_sidecar`, `copy_store`,
  `move_store` and `begin` call passes `runs_root`, and every `store` command passes
  `--runs-root`. `import RunRegistry` loads neither `ray` nor `sqlalchemy`.
- **Mutations.** (i)–(v) applied with plain `git apply` and were run from an empty working
  directory. They reproduced the logged counts exactly:

  | Mutation | Result | What fails |
  |---|---|---|
  | (i) | failures=12 | The refusal itself for `purpose`, `created` and `copied_from`, which are amended. For the other seven fields, `retired` with `--remove`, and the command line, the validator still refuses, and the test fails on the owner-naming message. |
  | (ii) | failures=1, errors=7 | The validator's `lacks ['before']` refusal, raised on every successful path. |
  | (iii) | failures=3 | The refusal itself. |
  | (iv) | failures=1 | The hand-built blank-reason entry. |
  | (v) | failures=2 | The message only: `ok` still refuses, and the test asserts "is a tombstone". |

  Log 04 explains (i) as "12 of the 20 subtests". `KNOWN_FIELDS` has ten members, and every one
  fails. It explains (ii) as a `KeyError` in `amend_sidecar`'s return statement. The return
  statement reads the local `before`, and the errors are the validator's refusal. The counts are
  right and the explanations are not. This board is right where they disagree. The tree and the
  working directory were clean afterwards.
- **`var/` and the repository root.** The snapshot, 71 entries, was identical before dispatch and
  after the replay and the probe. That includes the live A3 sidecar's SHA-256 and the
  `physics-test-n20-*` store's mtimes.
- **Suites.** AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 206,
  LiouvilleGreen 148 (1 skipped), RunRegistry 190 (+21). `black --check` is clean.
- **For prompt 05.** Its amendment of `backup` changes `retained` from `true` to `false`, bool to
  bool, so the `==` finding does not reach it. With 04 landed, 01–04 are done, and README §2
  releases 05 to be written. Prompt 03's review lists the points for 05's orchestrator.

**Orchestrator review of prompt 03 (2026-09-25).** All ten checks in `orchestrator/prompt-03.md`
§3 passed on `854e2ae`, from one dispatch.
- **Scope.** Only `RunRegistry/stores.py`, `__init__.py` (inside `begin`), `__main__.py`, the new
  `test_store_retire.py`, the log, this board and the index changed. `Datastore/`, `tools/`,
  `CLAUDE.md` and every existing test are untouched. The index changed for the one issue opened,
  with its count and date right.
- **The order of the writes.** One `_update_sidecar` writes `retired` in state `retiring`, with
  `files`, `references` and the one `retire` entry. `delete_store` follows, with `resume` true only
  on the completion path or for a missing shard under the flag. `files` comes from the same plan,
  with the same `resume`. Then the check that no listed file remains, then the second write, which
  appends no history. Nothing between them retries or cleans up. `_update_sidecar`'s docstring
  still counts its callers correctly.
- **The fingerprint.** `compare_fingerprints` emits `problems` entries whatever the digests say,
  and any entry refuses. The diff adds no naming-rule fallback. An unreadable `shards` table is
  recognised by prompt 01's refusal text, and a test pins it.
- **The reader and the history.** A test gives a copy entry and a move entry `retire`'s null `to`,
  and finds a problem in each. Each of the four non-test uses of `.ok` refuses a tombstone, or
  returns `None` for one: `__init__.py:355` and `stores.py:236`, `:915`, `:1434`.
- **Guards.** `begin`'s check is above `os.makedirs`. Copy and move from and to, fingerprint,
  adopt and create each refuse with the tombstone message. `retire_store`'s only caller outside
  the tests is `__main__.py`. Deviation 1, that `store show` exited 0 on every reading, is correct:
  `_show` at `ecfb024` returns 0 unconditionally.
- **Mutations.** (i)–(ix) applied with plain `git apply`, from an empty working directory, and
  reproduced the log exactly:

  | Mutation | Result |
  |---|---|
  | (i) | failures=18, errors=2 |
  | (ii) | failures=2 |
  | (iii) | failures=1 |
  | (iv) | failures=2 |
  | (v) | failures=1, errors=1 |
  | (vi) | failures=2 |
  | (vii) | failures=1 |
  | (viii) | failures=2 |
  | (ix) | failures=1 |

  The tree and the working directory were clean afterwards.
- **Tests.** The new module passed twice. `import RunRegistry` loads neither `ray` nor
  `sqlalchemy`. The interruption table names a test for every row, and says what a `.tmp` left by
  a killed write does.
- **`var/` and the repository root.** The snapshot, 71 entries, was identical before dispatch,
  after the replay and after the suites. It includes the `physics-test-n20-*` store's hashes and
  mtimes.
- **Suites.** AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 206,
  LiouvilleGreen 148 (1 skipped), RunRegistry 169 (+41). `black --check` is clean.
- **For prompt 05's orchestrator.**
  - **The sweep store's references report will be broad.** The sweep store sits directly in
    `var/datastores/`, so any sidecar with a string resolving to that directory will be listed
    (log 03, observation 4). Read its dry-run report with that in mind.
  - **`--without-fingerprint` treats every exception from the fingerprint as D4's error.** That
    includes one unrelated to damage, such as a transient `OSError`. The running-run and tombstone
    refusals come first, so neither reaches it (log 03, observation 5). The error is recorded
    verbatim, and neither real store should need the flag. If a dry run asks for it, stop and ask.

**Orchestrator review of prompt 01 (2026-09-25).** All eight checks in `orchestrator/prompt-01.md`
§3 passed on `ff65f9f`. The first dispatch was cut off by a usage limit after measuring baselines
only. It changed nothing. Its baselines, taken in an exported copy of the tree, showed four spurious
failures from git-dependent tests, and were discarded. The second dispatch landed the prompt.
- **Scope.** `RunRegistry/`, `tools/` and `docs/handover/` are untouched.
- **The charter.** `CLAUDE.md:52` equals D0's wording character for character. The orchestrator
  then re-wrapped that one sentence to the file's line width, in the review commit, with the words
  unchanged.
- **The deletion.** Both methods go through `_plan_deletion`, which reads only through
  `_read_closed_store`, and nothing opens a stored record as a path. The directory assertion is
  separate, and every `os.unlink` follows a regular-file, not-a-link check. `missing_ok` passes
  over only a shard of which no entry exists, so a dangling link is still refused.
- **Mutations.** (i), (i-b) and (ii) applied with plain `git apply`, and were run from an empty
  working directory, as the agent had done for (i-b). They gave exactly the logged results:
  `failures=48, errors=12`, `failures=62, errors=3` and `failures=26, errors=4`. The working
  directory was empty afterwards, and the repository root's real `physics-test-n20-*` store kept
  its 2026-09-10 mtimes.
- **Tests.** The new module passed twice.
- **`var/`.** The snapshot, 51 entries, was identical before and after.
- **Suites.** All six re-run: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore 206 (+29), LiouvilleGreen 148 (1 skipped), RunRegistry 128.
- **An observation, outside this campaign.** The repository root holds a pre-registry store,
  `physics-test-n20-lambdacdm-zend0p1*.sqlite`: a primary, four shards and a `-profile` file, about
  85 MB, from 2026-09-10. It is gitignored, has no sidecar, and is named on the
  `qcd-background-audit` board. The first cleanup answer looked only under `var/` and missed it.
  It is raised with the user, and no issue is opened.

**Orchestrator review of prompt 02 (2026-09-25).** All eight checks in `orchestrator/prompt-02.md`
§3 passed on `226889f`.
- **Scope.** The script's diff is confined to `prepare()`, `assert_store_is_self_consistent`,
  `--force` and their text. `prepare()` has one `copy_store` call, with the purpose verbatim.
- **Tests.** The new module passed twice.
- **Mutations.** (i) reproduced its two recorded failures, and (ii) its `failures=4, errors=4`.
  One defect in the log: its diffs do not apply as printed. They are indented as a Markdown block,
  and (i)'s hunk header counts 16 old lines where the hunk has 21. So each needed `git apply
  --recount --ignore-whitespace`, and (ii) also `-C1`. The code they describe is what was tested.
- **`var/`.** The orchestrator's own snapshot of every file under `var/datastores/` and
  `var/runs/`, 51 entries, was identical before and after.
- **Suites.** All six re-run: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore 177, LiouvilleGreen 148 (1 skipped), RunRegistry 128 (+11). `black --check` is clean.
- **Index.** Two stale sentences saying the issues were open, in `docs/OPEN_ISSUES.md` §1.12 and
  `datastore-portability`'s header, were corrected in the review commit.
- **One latent point, not an issue.** The check compares `resolve_shard_path(primary, …)`, which
  does not resolve symbolic links, against `shard_paths(primary)[i].resolve()`, which does. Given an
  absolute primary path through a symbolic link, it would refuse a sound store. That is fail-closed,
  never silent, and every caller passes a resolved path (`REPO_ROOT` is resolved, and `run_build`
  resolves `--database`), so nothing is affected today.

**03 and 04 were held on decisions, and were released on 2026-09-25** when the user approved D0
and D3–D7 as worded, and were written the same day against README §4's names, which gained
`resume` on `closed_store_files` and `--dry-run` on `store retire` before anything was dispatched. **05 is held, not
unplanned.** Its charter is fixed in README §2, and it is written last, against what 01–04 ship.
In 05 **the user** runs each `store retire`: no agent deletes a real store (README §5 rule 10).
**05 was written on 2026-09-25**, after 04's review (`00e0ec6`), against what 01–04 shipped. It
runs in three phases: the agent prepares and dry-runs, the user retires the two stores, the agent
checks and drafts the amendment, the user amends, and the agent records. Its orchestrator notes are
[`orchestrator/prompt-05.md`](orchestrator/prompt-05.md).

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| R1 | **FACILITY** | `ShardedPool.closed_store_files` and `ShardedPool.delete_store`. They share one planning step through `_read_closed_store`, never follow a stored record as a path, and refuse a hot journal, an unusable shard or a file outside the primary's directory. Shards go first and the primary last. | 01 | ✅ `_plan_deletion` is the one plan. Legacy records into a populated other directory delete only their siblings (log 01, test 2; mutation (i-b) shows the other directory's shards deleted without the resolver and the directory assertion). |
| R2 | **GUARD** | The interruption property: an interrupted deletion leaves a primary and some of its shards. The constructor refuses that state, `delete_store` refuses it, and `resume=True` completes it. `resume` relaxes the missing-shard refusal and nothing else. | 01 | ✅ rows D0–D5 of log 01's table, for a new-style and a legacy store, each by two injections |
| R3 | **CHARTER** | `CLAUDE.md:52` in D0's wording, and `ShardedPool`'s closed-store comment to match. | 01 | ✅ both quoted before and after in log 01 |
| R4 | **REMEDY** | `quadsource_atol_sweep.py` `prepare()` through one `RunRegistry.stores.copy_store` call. `--force` refuses with the new rule. After it, `--prepare` refuses at a name whose sidecar exists. Closes `datastore-portability`'s `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]`. | 02 | ✅ |
| R5 | **REMEDY** | `assert_store_is_self_consistent` compares serial by serial through `resolve_shard_path`, which unblocks `--build --resume` of every store built since `datastore-portability` prompt 01. Closes `[01-atol-sweep-check-expects-absolute-shard-records]`. | 02 | ✅ |
| R6 | **FORMAT** | A known `retired` field and a terminal `retire` history operation. The reader tells a tombstone from a broken sidecar (`SidecarReading.retired`), and calls a primary that has reappeared at a retired name a problem. | 03 | ✅ `retired` and `retire` are in the `stores.py` format table as shipped; `retire`'s `to` is null, for `retire` only. A completed tombstone reads with no problems and `ok` false; `retiring` and a reappeared primary are problems (log 03, tests 1, 7, 8; mutation (v)) |
| R7 | **FACILITY** | `retire_store` and `python -m RunRegistry store retire`. It refuses a `running` run, alive or stale, and a missing or mismatched fingerprint, except under D4. It writes the tombstone, with its file list, before deleting anything, then calls `delete_store`, then marks the tombstone complete. A second call completes an interrupted one. It reports the references it finds (D5). | 03 | ✅ the interruption table in log 03, one test per row, with a `.tmp` left by a killed write in two of them; mutations (i), (ii), (iii), (vi), (vii), (viii), (ix) |
| R8 | **GUARD** | `begin(results=…)` refuses a retired store. Copy, move, fingerprint, adopt and amend refuse a tombstone (D6). `store show` renders one. The `RunRegistry/stores.py` and `__main__.py` docstrings follow D0. | 03 | ✅ for begin, copy and move from and to, fingerprint, adopt and create; amend's own tombstone refusal is confirmed by prompt 04 (log 04, `TestRefusals.test_a_tombstone_complete_or_not`; mutation (v)). Both docstrings quoted before and after in log 03; mutation (iv). The package docstring in `RunRegistry/__init__.py` still says "deletes nothing": `[03-the-package-docstring-still-says-the-registry-deletes-nothing]` |
| R9 | **FACILITY** | `amend_sidecar` and `store amend`: replace or remove one unknown field of a registry sidecar, with a required reason, recording the old value in an `amend` history entry (D5). | 04 | ✅ every refusal in the prompt's §2.1 (log 04, `TestRefusals`); the `before`/`after` markers are a tagged wrapper, never a sentinel (`TestAmend`, `TestHistoryRule.test_the_amend_markers_shape`; no committed test amends a value shaped like the marker, see the orchestrator's review of prompt 04 and `[04-no-test-amends-a-value-shaped-like-the-marker]`); copy and move carry an amended field and its entry unchanged (`TestCopyAndMove`); two amendments of one field read back as a sequence (`TestTwoAmendments`); mutations (i)–(v) |
| R10 | **REMEDY** | Remedial: the sweep store retired by the user with `store retire`, checked before and after by the prompt's agent. `QUADSOURCE-TOLERANCE-SWEEP.md:15-17` then becomes true, and is not edited. | 05 | ⏸️ |
| R11 | **REMEDY** | Remedial: the backup retired the same way. The live A3 sidecar's `backup` field is then corrected by `store amend`. | 05 | ⏸️ |
| R12 | **RECORD** | The retirements recorded on the `run-registry` board, with `var/runs/a3-pilot/BACKUP_PATH` explained there rather than edited, and on the `handover` board. | 05 | ⏸️ |

---

## 3. Active and unresolved issues

Three were opened on 2026-09-25 by the audit
([`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md)), which is not a prompt,
one by prompt 01, one by prompt 03 and two by the orchestrator's review of prompt 04 the same
day. None is assigned to a prompt of this campaign. Each is recorded here for its owner. Two further
issues, owned by `datastore-portability`, are assigned to prompt 02. They stay on that board, and
are indexed at `docs/OPEN_ISSUES.md` §1.14.

- **[00-a-sigterm-pipeline-run-is-recorded-as-failed]** *(opened 2026-09-25 by the audit)*
  - **The defect.** The registered run `handover-03-a3-baseline-resume-20260923T024847` is
    recorded `failed`, exit code 1. The `handover` board records it `killed`
    (`prompts/handover/IMPLEMENTATION_STATE.md:309-313`). The launcher's capture,
    `var/bootstrap-a3-resume.log`, holds a SIGTERM trace at 10:52:13, two seconds before the last
    heartbeat. The run was at `704a12e`, after `ee46e5c` added `terminal_state`, which maps
    `SystemExit(15)` to `killed` (`docs/gktk-remedial/scoped_pipeline_run.py:248-264`). Exit code 1
    is the `except BaseException` branch (`:495-496`). The signal arrived inside
    `ray::core::CoreWorker::Wait`, where Ray's handler has replaced the script's. What reached the
    `exec` frame is not established.
  - **Impact.** The registry's record of how a run ended is wrong for the one deliberately stopped
    pipeline run it holds. A reader would conclude that the run crashed. `killed` and `failed`
    call for different next steps, which is why `terminal_state` exists. One instance, from a run
    stopped while blocked in `ray.wait`.
  - **Next step.** For whoever owns `scoped_pipeline_run.py`'s registration (`run-registry`'s
    area). Reproduce with a registered scoped run on a toy grid, sent SIGTERM while the parent is
    inside `ray.get` / `ray.wait`, and record the exception that reaches the `exec` frame. Not this
    campaign's.
  - **Measurement:** audit §2.11. Indexed at `docs/OPEN_ISSUES.md` §1.14.

- **[00-a-launch-log-lives-outside-its-run-directory]** *(opened 2026-09-25 by the audit)*
  - **The defect.** `var/bootstrap-a3-resume.log` (787 185 bytes, 2026-09-23 10:52) sits at the top
    of `var/`, outside every run directory, attributed by nothing on disk. It is a strict superset
    of the registered run's `stdout.log` and `stderr.log`: every line of theirs, and 67 more. Those
    are the scoped $k$ sample printed before `begin`, and the SIGTERM trace above. It is the only
    record on disk that the run was signalled. `CLAUDE.md`'s run-registry rule 3 puts stdout and
    stderr into the run directory. The launcher's own capture did not follow it.
  - **Impact.** Low, and bounded. Nothing reads it. But a tidy of `var/` that did not know what it
    was would delete the only evidence behind the issue above, and `RunRegistry list` does not
    see it.
  - **Next step.** A person, or a prompt granted `var/runs/`, moves it into
    `var/runs/handover-03-a3-baseline-resume-20260923T024847/` under a name that says what it is
    (for example `launcher.log`), and records the move on the `run-registry` board. A launcher
    that captures a registered child's output should write that capture inside the run directory.
    Not this campaign's: README §1 leaves `var/runs/` out of scope.
  - **Measurement:** audit §2.11. Indexed at `docs/OPEN_ISSUES.md` §1.14.

- **[00-a-copy-carries-its-sources-present-tense-fields]** *(opened 2026-09-25 by the audit)*
  - **The defect.** `copy_store` replaces seven known fields and carries every unknown field
    verbatim (`RunRegistry/stores.py:719-733`), by design (`datastore-portability` README §6.5
    point 1). Some unknown fields make claims about the store they sit in. A copy of the live A3
    store carries all three of these (`store-fingerprint` log 05 Phase C):
    - `backup`, whose `"retained": true` describes a backup *of the source*;
    - `restart.command`, whose `--database` is the **source's** path;
    - `run_history`, the source's runs.
  - **Impact.** Nothing reads these fields, so no code misbehaves. A person who followed a copy's
    `restart` would resume the pipeline **into the source store**. That is the 2026-09-23 accident
    (`2ebb7b6`), which wrote 54 rows into the live A3 store, by another route. After this
    campaign's prompt 02, every `--prepare` makes such a copy.
  - **Next step.** A decision about `copy_store`'s design: carry, drop, or flag such fields. That is
    for `datastore-portability`'s area, not this campaign's. If D5 is taken, prompt 04's
    `store amend` is the remedy for a given copy.
  - **Measurement:** audit §2.12. Indexed at `docs/OPEN_ISSUES.md` §1.14.

- **[01-cross-filesystem-move-advice-says-delete-by-hand]** *(opened 2026-09-25 by prompt 01)*
  - **The defect.** When `ShardedPool.move_store` fails with `EXDEV`, `_failure_message` advises
    "copy it instead, and then delete the source by hand". `RunRegistry.stores.move_store` calls
    `ShardedPool.move_store` (`RunRegistry/stores.py:753`), so `store move` gives that advice about
    a registered store.
  - **Impact.** Low, and only on a failed cross-filesystem move. The advice contradicts decision
    6.1.1: a registered store is removed by `store retire`, never by hand. A person who followed
    it would leave the source's sidecar describing nothing, the broken case of audit §2.1, with no
    tombstone. Nothing acts on the advice automatically.
  - **Next step.** Reword the advice so that a registered store's source is retired, not deleted
    by hand. `Datastore/tests/test_copy_move_store.py`'s
    `test_move_across_filesystems_fails_before_anything_moves` asserts the current text verbatim,
    so the rewording needs a decision about that test, which this campaign's rule 7 forbids
    modifying. Prompt 01 was told to leave the text alone. Unassigned.
  - **Measurement:** log 01, "Observations not acted on", item 1. Indexed at
    `docs/OPEN_ISSUES.md` §1.14.

- **[03-the-package-docstring-still-says-the-registry-deletes-nothing]** *(opened 2026-09-25 by
  prompt 03)*
  - **The defect.** The `RunRegistry/__init__.py` module docstring (`:5-9`) says the package
    "schedules nothing, supervises nothing, locks nothing and deletes nothing", and that "it still
    deletes nothing". Since prompt 03, `store retire` deletes a store's own files, through
    `RunRegistry.stores.retire_store`. That is what D0 and `CLAUDE.md:52-54` now say.
  - **Impact.** Documentation only. It is the first thing a reader of the package sees, and it
    contradicts the charter and the two docstrings prompt 03 changed. Nothing reads it.
  - **Next step.** Replace the two phrases with D0's wording in substance: the package deletes
    nothing but a store's own files, and those only through `store retire`, which keeps the
    sidecar. Prompt 03 could change only `begin` in that file, so it was left. A one-paragraph
    edit, for prompt 04 or 05 if its orchestrator admits it, or any later prompt with
    `RunRegistry/__init__.py` in scope. Unassigned.
  - **Measurement:** log 03, "Observations not acted on", item 1. Indexed at
    `docs/OPEN_ISSUES.md` §1.14.

- **[04-amend-calls-true-1-and-1-0-identical]** *(opened 2026-09-25 by the orchestrator's review
  of prompt 04)*
  - **The defect.** `amend_sidecar` refuses a value "identical, after a JSON round trip" to the
    current one by testing `fields[field] == normalised` (`RunRegistry/stores.py:2110`). In
    Python `True == 1 == 1.0`, and the same holds at any depth of nesting. So `true`, `1` and
    `1.0` count as identical although their JSON texts differ. A probe on `66617c9` found that
    a field holding `1` cannot be amended to `true` or to `1.0`, and `{"k": 1}` cannot be amended
    to `{"k": true}`.
  - **Impact.** Low. A real change is refused and nothing is written, so nothing is lost or
    corrupted. A person can get round it with `--remove` and then a second amendment, at the cost
    of two history entries. Prompt 05's amendment of `backup` changes `retained` from `true` to
    `false` and is not affected.
  - **Next step.** Compare type-strictly, for example by the canonical JSON text of each side
    (`json.dumps(…, sort_keys=True)`). Add a test that amends `1` to `true` and `1` to `1.0`.
    That is a change to `amend_sidecar` and `test_store_amend.py`. Unassigned.
  - **Measurement:** the orchestrator's review of prompt 04, §1 above. Indexed at
    `docs/OPEN_ISSUES.md` §1.14.

- **[04-no-test-amends-a-value-shaped-like-the-marker]** *(opened 2026-09-25 by the orchestrator's
  review of prompt 04)*
  - **The defect.** `orchestrator/prompt-04.md` check 3 asks for a test that amends or removes a
    field whose value has the marker's own shape, and reads it back unambiguously. No test in
    `RunRegistry/tests/test_store_amend.py` does. R9's first wording, log 04's "as shipped"
    section and the implementer's report each said one did.
  - **Impact.** Verification only. The wrapper design is correct by construction. A probe in the
    review amended `{"present": false}` to `{"present": true, "value": 1}` and read back
    `before = {"present": true, "value": {"present": false}}`, which is unambiguous. Without a
    test, a later change that flattened the markers would pass.
  - **Next step.** Add that test to `test_store_amend.py`. It needs no change to `stores.py`, and
    it can go with the fix for the issue above. Unassigned.
  - **Measurement:** the orchestrator's review of prompt 04, §1 above. Indexed at
    `docs/OPEN_ISSUES.md` §1.14.

---

## 4. Resolved issues

None yet.

---

## 5. Baselines

At `42d4910`, measured when the campaign was written: Datastore 177 OK, RunRegistry 117 OK. The
other suites were last measured at `50a24ac` by the `store-fingerprint` prompt 05 orchestrator:
AdaptiveLevin 32, ComputeTargets 552 (the known wall-clock flake aside), CosmologyModels 39,
LiouvilleGreen 148 (1 skipped). No code has changed between the two.

| Suite | At `42d4910` (campaign written) | After prompt 02 | After prompt 01 | After prompt 03 | After prompt 04 |
|---|---|---|---|---|---|
| `AdaptiveLevin` | 32 OK (at `50a24ac`) | not re-run (prompt 02 touches neither its code nor its imports) | 32 OK | 32 OK | 32 OK |
| `ComputeTargets` | 552 OK (at `50a24ac`; the flake is known) | 552 OK | 552 OK (the flake did not occur) | 552 OK (the flake did not occur) | 552 OK (the flake did not occur) |
| `CosmologyModels` | 39 OK (at `50a24ac`) | not re-run (prompt 02 touches neither its code nor its imports) | 39 OK | 39 OK | 39 OK |
| `Datastore` | 177 OK | 177 OK | 206 OK (177 + the 29 tests prompt 01 added) | 206 OK | 206 OK |
| `LiouvilleGreen` | 148 OK, skipped=1 (at `50a24ac`) | not re-run (prompt 02 touches neither its code nor its imports) | 148 OK, skipped=1 | 148 OK, skipped=1 | 148 OK, skipped=1 |
| `RunRegistry` | 117 OK | 128 OK (117 + the 11 tests prompt 02 added) | 128 OK | 169 OK (128 + the 41 tests prompt 03 added) | 190 OK (169 + the 21 tests prompt 04 added) |

Prompt 01 landed after prompt 02, so its column is the later one. Its baselines were the
orchestrator's at `226889f`: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore
177, LiouvilleGreen 148 (1 skipped), RunRegistry 128. Prompt 03's baselines were
measured on `ecfb024` before dispatch: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
Datastore 206, LiouvilleGreen 148 (1 skipped), RunRegistry 128. Prompt 04's baselines were measured
on `89529f4` (this dispatch's HEAD) before dispatch: AdaptiveLevin 32, ComputeTargets 552,
CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped), RunRegistry 169 — all matching
the state prompt 03 left.

Prompt 02's own verification is in
[`logs/02-the-sweep-prepares-through-the-registry.md`](logs/02-the-sweep-prepares-through-the-registry.md),
prompt 01's in [`logs/01-delete-a-closed-store.md`](logs/01-delete-a-closed-store.md), prompt
03's in [`logs/03-retire-a-store.md`](logs/03-retire-a-store.md), and prompt 04's in
[`logs/04-amend-an-unknown-field.md`](logs/04-amend-an-unknown-field.md).
