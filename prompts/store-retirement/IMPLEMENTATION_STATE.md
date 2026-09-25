# Store retirement campaign — implementation state

**Last updated:** 2026-09-25 · **Status: 2 of 5 prompts landed (01, 02). 03 and 04 are written;
03 is ready.** The user approved D0 and D3–D7 as worded on 2026-09-25 (README §6.2), which released
03 and 04. 05 is held until 01–04 land. 01 and 02 are independent of each other; 03 follows 01, and
04 follows 03. D4 was narrowed when 03 was written: an unreadable `shards` table is refused even
under `--without-fingerprint` (README §6.2).

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
| 03 | [Retire a store](03-retire-a-store.md) | **R6**–**R8** | Opus | ✍️ yes, 2026-09-25 | ⬜ after 01 | — | — |
| 04 | [Amend an unknown field](04-amend-an-unknown-field.md) | **R9** | Sonnet | ✍️ yes, 2026-09-25 | ⬜ after 03 | — | — |
| 05 | Retire the two stores | **R10**–**R12** | Opus | ⏸️ held until 01–04 land | — | — | — |

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

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| R1 | **FACILITY** | `ShardedPool.closed_store_files` and `ShardedPool.delete_store`. They share one planning step through `_read_closed_store`, never follow a stored record as a path, and refuse a hot journal, an unusable shard or a file outside the primary's directory. Shards go first and the primary last. | 01 | ✅ `_plan_deletion` is the one plan. Legacy records into a populated other directory delete only their siblings (log 01, test 2; mutation (i-b) shows the other directory's shards deleted without the resolver and the directory assertion). |
| R2 | **GUARD** | The interruption property: an interrupted deletion leaves a primary and some of its shards. The constructor refuses that state, `delete_store` refuses it, and `resume=True` completes it. `resume` relaxes the missing-shard refusal and nothing else. | 01 | ✅ rows D0–D5 of log 01's table, for a new-style and a legacy store, each by two injections |
| R3 | **CHARTER** | `CLAUDE.md:52` in D0's wording, and `ShardedPool`'s closed-store comment to match. | 01 | ✅ both quoted before and after in log 01 |
| R4 | **REMEDY** | `quadsource_atol_sweep.py` `prepare()` through one `RunRegistry.stores.copy_store` call. `--force` refuses with the new rule. After it, `--prepare` refuses at a name whose sidecar exists. Closes `datastore-portability`'s `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]`. | 02 | ✅ |
| R5 | **REMEDY** | `assert_store_is_self_consistent` compares serial by serial through `resolve_shard_path`, which unblocks `--build --resume` of every store built since `datastore-portability` prompt 01. Closes `[01-atol-sweep-check-expects-absolute-shard-records]`. | 02 | ✅ |
| R6 | **FORMAT** | A known `retired` field and a terminal `retire` history operation. The reader tells a tombstone from a broken sidecar (`SidecarReading.retired`), and calls a primary that has reappeared at a retired name a problem. | 03 | ⬜ |
| R7 | **FACILITY** | `retire_store` and `python -m RunRegistry store retire`. It refuses a `running` run, alive or stale, and a missing or mismatched fingerprint, except under D4. It writes the tombstone, with its file list, before deleting anything, then calls `delete_store`, then marks the tombstone complete. A second call completes an interrupted one. It reports the references it finds (D5). | 03 | ⬜ |
| R8 | **GUARD** | `begin(results=…)` refuses a retired store. Copy, move, fingerprint, adopt and amend refuse a tombstone (D6). `store show` renders one. The `RunRegistry/stores.py` and `__main__.py` docstrings follow D0. | 03 | ⬜ |
| R9 | **FACILITY** | `amend_sidecar` and `store amend`: replace or remove one unknown field of a registry sidecar, with a required reason, recording the old value in an `amend` history entry (D5). | 04 | ⬜ |
| R10 | **REMEDY** | Remedial: the sweep store retired by the user with `store retire`, checked before and after by the prompt's agent. `QUADSOURCE-TOLERANCE-SWEEP.md:15-17` then becomes true, and is not edited. | 05 | ⏸️ |
| R11 | **REMEDY** | Remedial: the backup retired the same way. The live A3 sidecar's `backup` field is then corrected by `store amend`. | 05 | ⏸️ |
| R12 | **RECORD** | The retirements recorded on the `run-registry` board, with `var/runs/a3-pilot/BACKUP_PATH` explained there rather than edited, and on the `handover` board. | 05 | ⏸️ |

---

## 3. Active and unresolved issues

Three were opened on 2026-09-25 by the audit
([`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md)), which is not a prompt,
and one by prompt 01 the same day. None is assigned to a prompt of this campaign. Each is recorded
here for its owner. Two further
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

---

## 4. Resolved issues

None yet.

---

## 5. Baselines

At `42d4910`, measured when the campaign was written: Datastore 177 OK, RunRegistry 117 OK. The
other suites were last measured at `50a24ac` by the `store-fingerprint` prompt 05 orchestrator:
AdaptiveLevin 32, ComputeTargets 552 (the known wall-clock flake aside), CosmologyModels 39,
LiouvilleGreen 148 (1 skipped). No code has changed between the two.

| Suite | At `42d4910` (campaign written) | After prompt 02 | After prompt 01 |
|---|---|---|---|
| `AdaptiveLevin` | 32 OK (at `50a24ac`) | not re-run (prompt 02 touches neither its code nor its imports) | 32 OK |
| `ComputeTargets` | 552 OK (at `50a24ac`; the flake is known) | 552 OK | 552 OK (the flake did not occur) |
| `CosmologyModels` | 39 OK (at `50a24ac`) | not re-run (prompt 02 touches neither its code nor its imports) | 39 OK |
| `Datastore` | 177 OK | 177 OK | 206 OK (177 + the 29 tests prompt 01 added) |
| `LiouvilleGreen` | 148 OK, skipped=1 (at `50a24ac`) | not re-run (prompt 02 touches neither its code nor its imports) | 148 OK, skipped=1 |
| `RunRegistry` | 117 OK | 128 OK (117 + the 11 tests prompt 02 added) | 128 OK |

Prompt 01 landed after prompt 02, so its column is the later one. Its baselines were the
orchestrator's at `226889f`: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore
177, LiouvilleGreen 148 (1 skipped), RunRegistry 128.

Prompt 02's own verification is in
[`logs/02-the-sweep-prepares-through-the-registry.md`](logs/02-the-sweep-prepares-through-the-registry.md),
and prompt 01's in [`logs/01-delete-a-closed-store.md`](logs/01-delete-a-closed-store.md).
