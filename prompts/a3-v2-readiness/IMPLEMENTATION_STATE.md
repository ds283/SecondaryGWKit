# A3 v2 readiness campaign — implementation state

**Last updated:** 2026-10-02 · **Status: IN PROGRESS — 1 of 8 prompts landed.** Audit 00 has landed
([`docs/a3-v2-readiness-audit.md`](../../docs/a3-v2-readiness-audit.md)); 01, 05 and 06 are written;
02, 03 and 04 stay held until the user has read the audit and the orchestrator writes them; 07 is
held on 01–06 (README §2).

The campaign was opened on 2026-10-02, when the user asked what remained to fix before the A3 v2
LambdaCDM store is built, and decided that all seven defects of README §0 are fixed first, by one
campaign (D4). Four are on the `datastore-integrity` board and close there; one was assigned there
to `handover` and is reassigned here; one had no board and is taken over by this one; one is
opened here.

**Campaign:** [`README.md`](README.md) ·
**Code:** `config/defaults.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ShardedPool.py`,
`extract_*.py`, `docs/gktk-remedial/scoped_pipeline_run.py`, `docs/handover/quadsource_atol_sweep.py`,
`RunRegistry/stores.py` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.17

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. An
> issue owned by another board is moved to **that** board's §4, and its row is deleted from the
> index. The index is an index: one line per issue, pointing here. Where the two disagree, this
> board is right. See `CLAUDE.md`.

### Decisions

**Made by the user, 2026-10-02 (README §6.1):**

| Decision | Decided |
|---|---|
| **D1** — A3 v2's pair | `(atol, rtol) = (1e-32, 1e-7)`, as recommended 2026-09-24; the science check is one rung tighter than the build's pair, so `1e-8` |
| **D2** — how the extractor gets the pair | `DEFAULT_QUADRATURE_RTOL` is set to `1e-7`; no flag |
| **D3** — how readers stop writing | a read-only pool, every file `mode=ro`, not per-site lookups |
| **D4** — the campaign | one campaign, `a3-v2-readiness`, owns all seven |

**Put to the user by audit 00 (2026-10-02), not yet taken** (audit, closing section, each with its
recommendation):

| Decision | Recommended |
|---|---|
| **U1** — does 02 also put the prune at open of a replicated row through a record (`[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]`)? | yes, in 02: the pool prunes replicated classes under a `prune` record the check at open repairs by re-running it |
| **U2** — what is done about `[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]`? | record only; no code change |

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 00 | [Audit the readiness defects](00-audit-the-readiness-defects.md) | **R1**–**R3**, **S1**–**S4**, **V1**–**V2**, **M1**–**M2** | Opus | ✍️ yes, 2026-10-02 | ✅ 2026-10-02 | *"Audit what the A3 v2 readiness prompts rest on"* | [`logs/00-…`](logs/00-audit-the-readiness-defects.md) |
| 01 | [Set the production quadrature `rtol`](01-set-the-production-quadrature-rtol.md) | **T1**–**T2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 02 | Write the version row through the recorded write | **W1**–**W2** | Opus | ⏸ held on 00 (V) | ⬜ | — | — |
| 03 | Open readers read-only | **O1**–**O4** | Opus | ⏸ held on 00 (R, M); after 01, 02 | ⬜ | — | — |
| 04 | Record a `SIGTERM` as killed | **K1**–**K3** | Opus | ⏸ held on 00 (S) | ⬜ | — | — |
| 05 | [Prune on resume](05-prune-on-resume.md) | **P1**–**P2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 06 | [Count only real references](06-count-only-real-references.md) | **F1**–**F2** | Sonnet | ✍️ yes, 2026-10-02 | ⬜ | — | — |
| 07 | Rehearse the A3 v2 launch | **L1**–**L3** | Opus | ⏸ held on 01–06 | ⬜ | — | — |

**Legend.** ✍️ written · ⏸ held, with what it waits on · ⬜ not landed · ✅ landed.

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-00.md`](orchestrator/prompt-00.md)
(2026-10-02, ready).

---

## 2. Items

| Item | Prompt | What | Status |
|---|---|---|---|
| **R1** | 00 | Every `pool.` call the six readers and `build_model_list` make: class, replicated or not, whether it inserts on a miss | ✅ audit R1: seven replicated classes `object_get`, five insert on a miss; eight sharded lookups, none writes |
| **R2** | 00 | The writes one reader's lookup sequence makes, counted on a stand-in, on a full store and on one missing a row | ✅ audit R2: QSI's whole sequence, 21 replicated gets = 42 primary commits and no row on a full store; a missing replicated row is inserted on every shard |
| **R3** | 00 | Whether A3 v2 holds a `QCD_Cosmology` row; each reader's path on a store without a model's `BackgroundModel` | ✅ audit R3: yes (read); each exits 1 at that model's `BackgroundModel` lookup |
| **S1** | 00 | A handler installed after `ray.init`: does it run during `ray.wait`, and what reaches the frame | ✅ audit S1: it runs, replaces absl's, arms no alarm; `SystemExit(143)` arrives as `SystemExit(1)` |
| **S2** | 00 | Ray's abort about 3 s after a `SIGTERM`: what it is, and whether `alarm(0)` stops it | ✅ audit S2: absl's 3 s `SIGALRM` → `SIGABRT`, 2.78–3.06 s; `alarm(0)` stops it |
| **S3** | 00 | What leaves a `gcs_server` behind | ✅ audit S3: any exit without `ray.shutdown()` (7 of 12); none through it (0 of 10) |
| **S4** | 00 | How long `finish(fingerprint=True)` takes on a real-sized store | ✅ audit S4: 2.48 s on the A3 baseline (333 MiB, partial), read-only, unchanged |
| **V1** | 00 | The version row's write sequence at open, and the crash window | ✅ audit V1: measured; two sibling windows opened |
| **V2** | 00 | The forms 02 can take, and how a read-only open finds the row | ✅ audit V2: form A recommended; found by label `mode=ro`, refuse when absent |
| **M1** | 00 | Every write at open, and what `read_only=True` does with each | ✅ audit M1: table, no row left to decide |
| **M2** | 00 | Whether every reader lookup can be served `mode=ro` | ✅ audit M2: yes, if replicated lookups skip `_replicated_write` |
| **T1** | 01 | `DEFAULT_QUADRATURE_RTOL = 1e-7`, its comment, every stale statement of its value | ⬜ |
| **T2** | 01 | The provenance subsection, the `handover` board line, a value pin | ⬜ |
| **W1** | 02 | The version row written once through `_replicated_write` | ⏸ |
| **W2** | 02 | The check at open treats a missing version row like any other replicated row | ⏸ |
| **O1** | 03 | `ShardedPool(read_only=True)`, every file `mode=ro`, nothing written at open | ⏸ |
| **O2** | 03 | `ReadOnlyMiss` on a lookup that would have inserted; `ReadOnlyWrite` on any write | ⏸ |
| **O3** | 03 | The six readers open read-only | ⏸ |
| **O4** | 03 | `--models` on each reader; a model the store lacks is skipped with a message | ⏸ |
| **K1** | 04 | One handler, installed after `ray.init` by both drivers, writing `killed` at once | ⏸ |
| **K2** | 04 | The fingerprint best-effort after the state; Ray shut down; no `gcs_server` left | ⏸ |
| **K3** | 04 | `terminal_state`'s docstring made true | ⏸ |
| **P1** | 05 | `--no-prune-unvalidated` out of both invocations and both usage lines | ⬜ |
| **P2** | 05 | A test that keeps it out | ⬜ |
| **F1** | 06 | The stores root is not a target | ⬜ |
| **F2** | 06 | Tests: sequential retirements, a sidecar naming the root, the subdirectory shape | ⬜ |
| **L1** | 07 | A registered `--build` at `(1e-32, 1e-7)` into a throwaway store, `SIGTERM`ed and resumed | ⏸ |
| **L2** | 07 | The recorded states, and the fingerprint against an uninterrupted reference | ⏸ |
| **L3** | 07 | Every reader, read-only, against the resumed store | ⏸ |

---

## 3. Active and unresolved issues

### 3.1 Assigned to this campaign (2026-10-02)

Each stays on its owning board, which carries an **Assigned** line, and closes there.

| Issue | Owning board | Prompt |
|---|---|---|
| `[02-the-version-row-is-written-outside-the-in-flight-record]` | `datastore-integrity` §3.3 | 02 |
| `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` | `datastore-integrity` §3.3 | 03 |
| `[10-extraction-scripts-exit-1-on-a-store-without-every-model]` | `datastore-integrity` §3.3 | 03 |
| `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]` | `datastore-integrity` §3.3 | 04 |
| `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` | `datastore-integrity` §3.3; reassigned from `handover` | 05 |

### 3.2 Held on this board

- **[10-store-retire-reports-every-tombstone-as-a-reference]** *(seen 2026-10-02, when the user
  retired `datastore-integrity` prompt 10's two stores; indexed the same day in `424469e` with no
  board, and taken over by this board when it was created)*
  - **The defect.** `_find_references` (`RunRegistry/stores.py`) takes the store's directory,
    `primary.parent`, as a target beside the primary and the planned files. Every tombstone
    records `references.stores_root`, the directory its retirement searched. For a store directly
    in `var/datastores/` the two are the same directory, so each `store retire` reports every
    earlier tombstone there as a sidecar naming the store.
  - **Measured.** Retiring `datastore-integrity-10-reference` listed two tombstones, and
    `datastore-integrity-10-resumed` three: `handover-atol-sweep`, the backup, and
    `datastore-integrity-10-reference`, each by `$.retired.references.stores_root` and by nothing
    else (read 2026-10-02 from the resumed store's tombstone).
  - **Impact.** Report only: nothing is refused or rewritten. But the list grows by one with every
    retirement, and a real reference in it is harder to see.
  - **Next step.** Prompt 06: the stores root is not a target; a store alone in a subdirectory,
    the backup's shape, keeps its directory as one.
- **[00-the-prune-at-open-deletes-a-replicated-row-outside-the-in-flight-record]** *(opened
  2026-10-02 by audit 00, V1; does not block A3 v2)*
  - **The defect.** With `prune_unvalidated=True` (`main.py`'s default, and A3 v2's resume once
    prompt 05 lands), each `Datastore` actor deletes an unvalidated `BackgroundModel` — a
    replicated class — with its values and tags from its own shard, in its own constructor
    (`_validate_on_startup` → `sqla_BackgroundModelFactory.validate_on_startup`), outside the
    `replication_in_flight` record. The same defect as
    `[02-the-version-row-is-written-outside-the-in-flight-record]`, for the other replicated write
    a constructor makes.
  - **Measured** (`docs/a3-v2-readiness/version_window_probe.py`, case 4, stand-in shards): every
    actor after the first killed during a pruning open left shard 2 with no `BackgroundModel` and
    shards 0 and 1 with `[(1, 0)]`; the next open was refused by the check at open with five
    differences (`BackgroundModel`, three `BackgroundModelValue`, one `BackgroundModel_tags`).
  - **Impact.** A store that needs a person. Reached only when a pruning open of a store holding an
    unvalidated `BackgroundModel` is interrupted between actors; for A3 v2, only if a launch is
    interrupted between the background's store and its validate and the resume is then interrupted
    in its first seconds.
  - **Next step.** User decision U1 (audit, closing section): recommended, in prompt 02, the pool
    prunes replicated classes itself, before any actor exists, under a `prune` record that the
    check at open repairs by running the idempotent prune again.
- **[00-a-new-stores-first-open-can-leave-a-primary-without-its-shards]** *(opened 2026-10-02 by
  audit 00, V1; does not block A3 v2)*
  - **The defect.** `ShardedPool.__init__` writes a new store's primary and its `shards` table
    first, and each shard file is created by its actor's constructor afterwards. Interrupted in
    between, the store's next open is refused by `_check_shard_files` for a shard that does not
    exist.
  - **Measured** (`version_window_probe.py`, cases 1 and 2): `shard #0 … which does not exist;
    shard #1 … which does not exist`, and `shard #1 … which does not exist`.
  - **Impact.** None to data: the store is empty. But a person must delete it, since `--build`
    refuses an existing path and `--build --resume` checks the store first.
  - **Next step.** User decision U2 (audit, closing section): recommended, record only; 07's
    rehearsal can confirm the refusal makes the remedy clear.
- **[00-quadsource-extraction-reads-at-the-default-rtol]** *(opened 2026-10-02 by this campaign's
  plan)*
  - **The defect.** `extract_QuadSourceIntegral_data.py` looks up the tolerance rows its
    `QuadSourceIntegral` reads are keyed on with `pool.object_get("tolerance",
    tol=DEFAULT_QUADRATURE_RTOL)`, and `DEFAULT_QUADRATURE_RTOL` is `1e-8`. A3 v2 is built at
    `1e-7` (D1) by `quadsource_atol_sweep.py --build`, which substitutes `tol=SWEEP_QUAD_RTOL` for
    `tol=DEFAULT_QUADRATURE_RTOL` in `main.py`'s source; `config.defaults` never learns the pair.
  - **Read, not run.** The rows A3 v2 holds are keyed on the `1e-7` tolerance row, so the
    extractor matches none of them; and its `object_get` on the replicated `tolerance` class
    inserts a `1e-8` row if the store holds none (which `[08-…]` makes a write in any case).
    `datastore-integrity` prompt 10 could not see this: it built at `main.py`'s defaults.
  - **Impact.** Nothing can be extracted from A3 v2's `QuadSourceIntegral` rows, which are the
    point of the store.
  - **Next step.** Prompt 01, under D2: the default becomes `1e-7`.

---

## 4. Resolved issues

None yet.

---

## 5. Baselines

Inherited, README §7: AdaptiveLevin 32, ComputeTargets 665, CosmologyModels 39, Datastore 551,
LiouvilleGreen 148 (1 skipped), RunRegistry 199, at `f4f1896`. Each prompt adds a subsection here
with the counts after its commit.

### 5.1 After prompt 00 (2026-10-02)

Measured 13:12–13:19 at the tree of this commit (the audit adds no test), serially, each suite's
output written to a file and its verdict read from it (log 00, deviation 9):

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 665 OK |
| CosmologyModels | 39 OK |
| Datastore | 551 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

Unchanged from README §7. Load average 10–27 during the run; no flake fired.
