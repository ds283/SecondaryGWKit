# Datastore integrity campaign — implementation state

**Last updated:** 2026-10-03 · **Status: COMPLETE — 12 of 12 prompts landed. Closed by the user on
2026-10-02, after the orchestrator's review of 10.** (01, 02, 03, 04, 04b, 05 as 05a–05c,
06, 07, 08, 09 as 09a–09c, 09d and 10; all reviewed, 10 on 2026-10-02.) The user decided D2–D6 on 2026-09-26 (README §6.2), and D1 (yes) later
the same day. 04b landed the same day it was written, from the 04 review's findings. **05 landed
2026-09-28** as 05a → 05b → 05c, in one orchestrator session
([`orchestrator/prompt-05.md`](orchestrator/prompt-05.md)). **02 landed 2026-09-28**, and closes
`[00-replicated-writes-can-diverge-across-shards]`; reviewed the same day. **03 landed 2026-09-29**,
closing `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`,
`[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` and
`[03-integrationsolver-stepping-minimum-lookup]` on their owning boards' §4, and narrowing
`[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]` on the
tolerance-convergence board. Reviewed the same day: the code passed, and L5's account of what
the two verification scripts do after a miss was wrong (both raise; they do not finish) and is
corrected on that board and in the index. The review opened one issue, closed the same day (§4). **06 landed 2026-09-29**, closing
`[00-quadsource-tq-serial-has-the-wrong-foreign-key]` on the store-fingerprint board's §4 and
setting up D5's witness procedure; reviewed the same day, with two small remnants fixed in a follow-up commit. **07 landed 2026-09-29**, closing
`[20-wkb-rows-consume-numeric-initial-data]` on the GkTk-remedial board's §4 and opening two
issues (§3.3); reviewed the same day, with three small remnants fixed in a follow-up commit.
**08 landed 2026-09-29**, closing `[00-quadsourcepolicy-rows-are-referenced-by-nothing]` on the
store-fingerprint board's §4 and opening one issue (§3.3); reviewed 2026-09-30, with three small
remnants fixed in a follow-up commit.
**09 was split by the user on 2026-09-30 into 09a → 09b → 09c**, for one orchestrator session
([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md)). **09a landed 2026-09-30** (`QuadSource` keyed on `Tq` and `Tr`; the inventory's cross-shard `Parent`; one issue opened, §3.3); reviewed the same day, with one small remnant fixed in a follow-up commit. **09b landed 2026-09-30** (`GkSource` keyed on its set of Gk rows, recorded in `GkSource_parents` and digested in `parents_digest`; the inventory's set-valued parent; closing `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` on this board's §4 and opening two issues, §3.3); reviewed the same day, with one small remnant fixed in a follow-up commit. **09c landed 2026-09-30** (`QuadSourceIntegral` keyed on the six stored rows it is computed from -- its `QuadSource`, its `GkSourcePolicyData`, and the Tk numeric and WKB rows for q and r, in four new columns -- with the relations between them asserted in `compute()`; its `main.py` stage resolves every work item's parents before its existence query; closing `[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]` on this board's §4 and opening none); reviewed 2026-10-01, with three small remnants fixed in a follow-up commit. **09 is complete**: all three parts have landed and been reviewed. **09d landed 2026-10-01** (the inventory report refers to a derived row, and to a `GkSource`'s parent set, by a label and prints each once, in its own section; closing `[09b-inventory-report-renders-every-member-of-a-parent-set]` on this board's §4 and opening one issue, §3.3); reviewed the same day, with three small remnants fixed in a follow-up commit. **10 landed 2026-10-02** (a scoped build
into a throwaway store, interrupted twice and resumed, then read back by everything that reads a
store; closing `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]` on this
board's §4 and opening four issues, §3.3, one of them assigned to `handover`). After the user's
pruned resume the resumed store's fingerprint equals the reference's; the record is
[`docs/datastore-integrity-verification.md`](../../docs/datastore-integrity-verification.md).
`store-retirement` closed on 2026-09-26 at 6 / 6, which discharges both cross-campaign constraints
(README §2).

The campaign was opened on 2026-09-25, when the user asked that every known datastore-integrity
defect be fixed before the A3 v2 store is built, the first store meant to have archival value. The
audit ([`docs/datastore-integrity-audit.md`](../../docs/datastore-integrity-audit.md)) re-verified
the eleven open issues in that class, all still present, and found each class wider than recorded:
- the replicated write can diverge, and a resume turns a missing row into a serial split the
  fingerprint cannot see; the three real stores are clean serial by serial (§4);
- the exit-time and `IntegrationSolver` lookups serve the wrong row, and identity lookups hide
  duplicates (§3.5–§3.9);
- every tagged `QuadSourceIntegral` `read_batch` fails, so extraction from A3 v2 would fail
  (§3.1); value reads omit parts of their parent's key (§3.3–§3.4);
- one foreign key names the wrong table and another is missing (§2.1–§2.2);
- derived rows are not keyed on what they were computed from (§2.3–§2.5).

**Campaign:** [`README.md`](README.md) ·
**Code:** `Datastore/SQL/ShardedPool.py`, `Datastore/SQL/Datastore.py`, `Datastore/SQL/ObjectFactories/`, `Datastore/store_inventory.py`, `Datastore/tests/`, the WKB / `QuadSource` / `QuadSourceIntegral` / `GkSource` compute targets, `main.py`, `extract_*.py` ·
**Index:** [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.15

> **Maintenance rule.** Whenever an entry is added to, narrowed in, or closed out of §3 or §4
> below, [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) is updated **in the same commit**:
> the row is added, moved or deleted, and the count and date in its header are corrected. An
> issue owned by another board is moved to **that** board's §4, and its row is deleted from the
> index. The index is an index: one line per issue, pointing here. Where the two disagree, this
> board is right. See `CLAUDE.md`.

### Decisions

**Made by the user, 2026-09-25 (README §6.1):** the replicated write is made atomic;
`QuadSourcePolicy` is kept and added to `QuadSourceIntegral`'s key; the exit-time tolerance lookup
becomes an exact match; the adjacent issues are out of scope.

| Decision | Decided (README §6.2) | Blocked, until decided |
|---|---|---|
| **D1** — key derived rows on their parents | **yes** (2026-09-26), once explained; a reader holding only physical labels may still select by them, and must refuse on more than one match | — |
| **D2** — fix latent read-path defects no caller reaches | **yes** (2026-09-26) | — |
| **D3** — how much of `OneLoopIntegral`'s factory to repair | **the mechanical defects only**; the rest when `OneLoopIntegral` is built (2026-09-26) | — |
| **D4** — existing stores | **readers are not made version-aware** (2026-09-26); the ordering against `store-retirement` was discharged when it closed | — |
| **D5** — deliberately changing the schema witness and golden fingerprint | **as recommended** (2026-09-26): a new witness per schema-changing prompt, never overwritten, the diff in the log | — |
| **D6** — which `QuadSourcePolicy` each `QuadSourceIntegral` gets | **`QuadSource_policy_1pt5` only, for now** (2026-09-26) | — |

---

## 1. Prompts

| # | Prompt | Covers | Model | Written? | Landed? | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Checked replicated write](01-checked-replicated-write.md) | **W1**–**W4** | Opus | ✍️ yes, 2026-09-25 | ✅ 2026-09-26 | *"Make the replicated write checked, idempotent and recorded"* | [`logs/01-…`](logs/01-checked-replicated-write.md) |
| 02 | [Repair or refuse at open](02-repair-or-refuse-at-open.md) | **W5**–**W7** | Opus | ✍️ yes, 2026-09-25 | ✅ 2026-09-28 | *"Repair or refuse divergent replicated tables when a store opens"* | [`logs/02-…`](logs/02-repair-or-refuse-at-open.md) |
| 03 | [Exact identity lookups](03-exact-identity-lookups.md) | **L1**–**L5** | Sonnet | ✍️ yes, 2026-09-25 | ✅ 2026-09-29 | *"Match the exit-time and identity lookups exactly, and raise on a duplicate"* | [`logs/03-…`](logs/03-exact-identity-lookups.md) |
| 04 | [Extend the read-back guard](04-extend-the-read-back-guard.md) | **G1**–**G3** | Sonnet | ✍️ yes, 2026-09-25 | ✅ 2026-09-26 | *"Extend the read-back guard to read_batch and column existence"* | [`logs/04-…`](logs/04-extend-the-read-back-guard.md) |
| 04b | [Tighten the read-back guard](04b-tighten-the-read-back-guard.md) | **G4**–**G6** | Sonnet | ✍️ yes, 2026-09-28 | ✅ 2026-09-28 | *"Tighten the read-back guard to every function and every declared hole"* | [`logs/04b-…`](logs/04b-tighten-the-read-back-guard.md) |
| 05a | [Repair the mechanical read defects](05a-repair-the-mechanical-read-defects.md) | **P6**, **P7** | Sonnet | ✍️ yes, 2026-09-28 | ✅ 2026-09-28 | *"Repair the mechanical read-path defects: names, an argument, tags, raises"* | [`logs/05a-…`](logs/05a-repair-the-mechanical-read-defects.md) |
| 05b | [Repair the integral `read_batch`es](05b-repair-the-integral-read-batches.md) | **P1**, **P2**, **P8** | Sonnet | ✍️ yes, 2026-09-28 | ✅ 2026-09-28 | *"Repair the integral read_batch join, exit-time serials and OneLoopIntegral"* | [`logs/05b-…`](logs/05b-repair-the-integral-read-batches.md) |
| 05c | [Key the value reads on their parent](05c-key-the-value-reads-on-their-parent.md) | **P3**–**P5**, **P9** | Opus | ✍️ yes, 2026-09-28 | ✅ 2026-09-28 | *"Key the value reads on their parent and refuse a second parent"* | [`logs/05c-…`](logs/05c-key-the-value-reads-on-their-parent.md) |
| 06 | [Correct the foreign keys](06-correct-the-foreign-keys.md) | **S1**–**S5** | Sonnet | ✍️ yes, 2026-09-26 | ✅ 2026-09-29 | *"Correct the foreign keys the schema declares, and check them"* | [`logs/06-…`](logs/06-correct-the-foreign-keys.md) |
| 07 | [Key WKB rows on their numeric row](07-key-wkb-rows-on-their-numeric-row.md) | **K1**–**K2** | Opus | ✍️ yes, 2026-09-26 | ✅ 2026-09-29 | *"Key each WKB row on the numeric row it starts from"* | [`logs/07-…`](logs/07-key-wkb-rows-on-their-numeric-row.md) |
| 08 | [Key `QuadSourceIntegral` on its policy](08-key-quadsourceintegral-on-its-policy.md) | **K3** | Sonnet | ✍️ yes, 2026-09-26 | ✅ 2026-09-29 | *"Key QuadSourceIntegral on its QuadSourcePolicy"* | [`logs/08-…`](logs/08-key-quadsourceintegral-on-its-policy.md) |
| 09 | [Key derived rows on their parents](09-key-derived-rows-on-their-parents.md) | **K4**–**K7** | — | ✍️ yes, 2026-09-26; **split 2026-09-30** into 09a–09c, the charter | ✅ 2026-09-30, as 09a–09c, all three landed | — | — |
| 09a | [Key `QuadSource` on its transfer functions](09a-key-quadsource-on-its-transfer-functions.md) | **K4**, **K7** (cross-shard) | Sonnet | ✍️ yes, 2026-09-30 | ✅ 2026-09-30 | *"Key QuadSource on the transfer functions it is computed from"* | [`logs/09a-…`](logs/09a-key-quadsource-on-its-transfer-functions.md) |
| 09b | [Key `GkSource` on its parent set](09b-key-gksource-on-its-parent-set.md) | **K6**, **K7** (set-valued) | Opus | ✍️ yes, 2026-09-30 | ✅ 2026-09-30 | *"Key GkSource on the set of Gk rows it is assembled from"* | [`logs/09b-…`](logs/09b-key-gksource-on-its-parent-set.md) |
| 09c | [Key `QuadSourceIntegral` on its inputs](09c-key-quadsourceintegral-on-its-inputs.md) | **K5**, **K7** (complete) | Opus | ✍️ yes, 2026-09-30 | ✅ 2026-09-30 | *"Key QuadSourceIntegral on the six rows it is computed from"* | [`logs/09c-…`](logs/09c-key-quadsourceintegral-on-its-inputs.md) |
| 09d | [Refer to derived parents by label in the inventory report](09d-refer-to-derived-parents-by-label-in-the-inventory-report.md) | — (closes `[09b-inventory-report-renders-every-member-of-a-parent-set]`) | Sonnet | ✍️ yes, 2026-10-01; orchestrator note [`orchestrator/prompt-09d.md`](orchestrator/prompt-09d.md) | ✅ 2026-10-01 | *"Refer to derived parents by label in the inventory report"* | [`logs/09d-…`](logs/09d-refer-to-derived-parents-by-label-in-the-inventory-report.md) |
| 10 | [Build a throwaway store end to end](10-build-a-throwaway-store-end-to-end.md) | **V1**–**V3**; closes `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]` | Opus | ✍️ yes, 2026-10-01; orchestrator note [`orchestrator/prompt-10.md`](orchestrator/prompt-10.md); D10-1 to D10-5 decided 2026-10-01, as recommended | ✅ 2026-10-02 | *"Build a throwaway store end to end, interrupt it and read it back"* | [`logs/10-…`](logs/10-build-a-throwaway-store-end-to-end.md) |

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
and [`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) exist (2026-09-26). 01 and 04 are
dispatched one at a time, never concurrently in this checkout, in either order. Since then:
[`prompt-04b.md`](orchestrator/prompt-04b.md) and [`prompt-05.md`](orchestrator/prompt-05.md)
(2026-09-28, both used), [`prompt-02.md`](orchestrator/prompt-02.md) (2026-09-28, used), and
[`prompt-03.md`](orchestrator/prompt-03.md) (2026-09-29, used), and
[`prompt-06.md`](orchestrator/prompt-06.md) (2026-09-29, used), and
[`prompt-07.md`](orchestrator/prompt-07.md) (2026-09-29, used), and
[`prompt-08.md`](orchestrator/prompt-08.md) (2026-09-29, used), and
[`prompt-09.md`](orchestrator/prompt-09.md) (2026-09-30, used for 09a–09c), and
[`prompt-09d.md`](orchestrator/prompt-09d.md) (2026-10-01, used), and
[`prompt-10.md`](orchestrator/prompt-10.md) (2026-10-01, ready).

**Decided for 10 (2026-10-01, the user, after its prompt and note were written).** The prompt's §1.1
states five decisions with recommendations, and the user took all five.
- **D10-1, the scope.** LambdaCDM only; three wavenumbers log-spaced over $10^5$–$10^6$/Mpc;
  `--zend 250`, `--source-samples-log10z 100`, the default response sparseness; `--shards 4`;
  `main.py`'s default tolerances, policies and tags; `--cpus 6`.
  - The bound rests on the orchestrator's read-only measurement of the A3 baseline (note §0). Its
    rows with $k$, $q$ and $r$ all at or below $1.1\times10^6$ cost 6,315 CPU-seconds in all, and
    none lies below $z_{\rm response} = 132$.
- **D10-2, two builds compared.** A reference build straight through, and the same scope
  interrupted and resumed. Their fingerprints must be equal.
- **D10-3, two interruptions,** by `SIGTERM`: one in the `GkSource` stage and one in the
  `QuadSourceIntegral` stage, each leaving its stage partly stored.
- **D10-4, fault copies at open.** Four tampered plain copies of the closed resumed store, outside
  `var/`, test prompt 02's three refusals and its repair on real shards.
- **D10-5, the stores afterwards.** Both stay in `var/datastores/` with their sidecars for the
  review. Retiring them is the user's, afterwards.

**Decided during 10 (2026-10-01 and 2026-10-02, the user, on the agent's stops).** Log 10 records
each, with its evidence.
- **The interrupted launches' states.** A `SIGTERM` in `ray.wait` reaches the driver as
  `SystemExit(1)`, so `resumed-a` is recorded `failed`/1; its evidence of a deliberate interruption
  (the interrupter's log, Ray's `SIGTERM received` banner, no traceback, the part-stored stage)
  stands in place of §2.1's `killed`. `resumed-b` was aborted 3 s after its `SIGTERM` and wrote no
  state; the agent ended it `killed`, by `Run.finish`, as a person's act on the user's
  authorisation, and killed its orphaned `gcs_server` with `SIGKILL`.
- **The `GkSource` duplicate is `--no-prune-unvalidated`'s.** The user: "The bug here is the
  `--no-prune-unvaiidated`. Pruning is there precisely to deal with this situation." So a fifth
  launch, `resumed-d`, resumed with `main.py`'s default pruning, and §2–§3 ran on the store it
  left. The issue is opened against A3 v2's launch, owned by `handover` (§3.3).
- **The readers' exit 1.** Every `extract_*.py` script reads LambdaCDM in full and exits 1 on
  `QCD_Cosmology`, which the store does not hold. That result, with each script's unmoved
  fingerprint, stands in place of "each must exit 0", and the scripts are opened as a defect (§3.3).

**Decided for 07 (2026-09-29, the user, while its note was written).** Prompt 07 was written before
05c and 06 landed, and as written it would hit its own §5 stop or silently change a reader in four
places. The user took the recommendation each time; the note's §0.1–§0.4 hold the detail.
- **Existing tests that must follow the key are granted,** each limited to following it:
  - the fixture's default WKB rows;
  - the WKB entries of `test_store_inventory.py`'s `IDENTITY` and its missing-key-column test;
  - the two raw-insert helpers that write Tk WKB rows;
  - 05c's `z_init` tests, which its log handed to 07;
  - the parentless construction in `test_tk_wkb_phase.py`;
  - and, under D5, the two witness constants.
- **`FINGERPRINT_FORMAT` goes from 1 to 2.** `RunRegistry/stores.py`'s own rule requires it for a
  change to what the inventory records. Existing stores' fingerprints then compare as another
  format, not as differing content.
- **A Gk lookup names its parent or an explicit pure-WKB marker, never neither.** Readers use an
  explicit labels-only mode that refuses on more than one match (D1). Otherwise
  `extract_GkWKB_data.py`, which passes no parent, would silently see pure-WKB rows only.
- **Moving the Gk seeded-or-pure choice ahead of the existence query is within scope.** The choice
  is at `main.py:2182`, and it goes pure below `z_source_limit` even when a numeric row exists.
  Moving it unchanged is part of "looking the numeric row up first", not §5's restructuring.

**Measured for 07 (2026-09-29, while its note was written).** A throwaway sketch of the schema half,
in a worktree outside the tree and since removed, added the two columns, `store()`'s write and
`inventory_records`' new parent. It measured four things.
- **Adding the column moves Datastore to `failures=56, errors=18`.** Giving the fixture's rows
  parents and `Parent` a nullable form brings it to `failures=11, errors=16`, which is exactly the
  granted list plus the witness. RunRegistry fails only the golden. ComputeTargets 552 and
  CosmologyModels 39 were unchanged.
- **The store inventory's `Parent` cannot resolve a NULL.** Every pure-WKB Gk row became an
  `unresolved-parent` problem and dropped out of the records. So a nullable form is structurally
  required, and README §1 puts `store_inventory.py` in 07's scope.
- **Only `TkWKBIntegration` and `GkWKBIntegration` move in the fingerprint.** Their counts and
  tag-set counts are unchanged, and no problems appear.
- **The witness diff is 12 hunks and 86 lines**, six per class, because this is a new, indexed
  column.

Two further facts shape the tests:
- **The fixture has the same coincidence 06 found.** Tk numeric 1 and Tk WKB 1 share shard 0, so a
  wrong key would pass there.
- **`_drop_order` is already right for Gk.** Only the Tk pair is reversed.

**Measured for 06 (2026-09-29, while its note was written).** A throwaway sketch of S1–S3 and
the two flags, in a worktree outside the tree and since removed, did three things:
- It failed only `test_schema_builder`'s two witness tests, for exactly the four classes it
  touched. It left the golden fingerprint, ComputeTargets 552 and CosmologyModels 39 unchanged.
- Its witness diff was 9 hunks and 30 lines, because each foreign key appears three times (the
  column, the constraint, the DDL).
- It showed that the fixture cannot give S4's "before S1" failure as it stands. Each fixture
  `QuadSource`'s `Tq_serial` equals a `QuadSource` serial on its own shard, so the wrong key passes
  `PRAGMA foreign_key_check` by coincidence, and `relabel_serials` follows whatever key is declared.
  The test needs fixed rows of its own. Mutation (ii) fires on the fixture as it is.

**Measured for 03 (2026-09-29, while its note was written).** A throwaway sketch of L1–L4, in a
worktree outside the tree and since removed, left every existing test passing unmodified:
Datastore 325, ComputeTargets 552 and CosmologyModels 39. That includes the read-back guard, whose
coverage sets pin the shape of the `build()`s 03 edits. So 03 can be met without touching a
test, and a guard failure under 03 is a reason to change the shape, not the guard. Production
requests the exit time at the pair and stepping (0) it stores, and each `IntegrationSolver` at
the stepping it inserts, so L1 and L2 move no production lookup.

**Decided for 02 (2026-09-28, the user, while its note was written).**
`test_replicated_write.py`'s `TestWriteOverASetRecord.test_a_reopened_pool_refuses` pins 01's
behaviour: a pool reopened over a set record refuses every replicated write. That is the state 02
repairs at open, so the test fails by design. The user chose to rewrite it to 02's rule on the
same fixture: the reopen repairs, clears the record, and the write succeeds. The note names it
under README §5 rule 7, as 02's only permitted edit to an existing test.

**Orchestrator review of prompt 01 (2026-09-26).** All nine checks in `orchestrator/prompt-01.md`
§3 passed on `c08a9d0`, dispatched from `65cd4b7` with the tree and the index clean.
- **Scope.** Eleven files: the four named source files, the new `Datastore/replication.py`
  (`ReplicationMismatch`, `ReplicationInFlight`, `differing_columns`), the helper module
  `Datastore/tests/standin_pool.py`, the test module `test_replicated_write.py`, the log, the two
  boards and the index. No existing test is modified. Every factory hunk is inside `store()`, and
  the controller's value payload is unchanged, refactored into a local function. No
  `ShardedPool.py` hunk touches the closed-store block.
- **One helper.** `ShardedPool._replicated_write` alone reads, writes and clears
  `replication_in_flight`. Outside it the table appears only in its declaration
  (`_create_engine`), its creation for a new store (`_write_shard_data`), and
  `create(checkfirst=True)` in the constructor after `_check_shard_files`. No other `.remote()`
  call reaches a replicated class: `object_get_vectorized` refuses one.
- **The commit-point table.** Every row checks against the code. The record is committed on
  `self._engine`, in its own transaction, before `submit_controller` on every path. It is written
  for **every** replicated call, including a get that inserts nothing, because only the
  controller's answer says whether it inserted. Any exception after that commit skips the delete,
  so a controller that raised before committing also leaves the record set, over identical
  shards. That is fail-closed, never silent. No row reads "shards differ, no record".
- **Checked answers and replica stores.** Scalar and vector get, store (serial and value serials)
  and validate are checked before the delete. A record found set raises `ReplicationInFlight`
  before anything is written. The agent flagged one choice, which is accepted: in the
  `BackgroundModel` replica, "key held under another serial" counts only **validated** rows with
  exactly the same tags. That is what `build()` serves (it filters `validated == True`), and it
  lets a resume store beside the unvalidated row an interrupted store left on every shard.
- **One timestamp.** `insert_timestamp` is a keyword-only call argument, bound per call by
  `_inserters_for`. With none, `_insert` calls `datetime.now()` as before, and no sharded path
  passes one.
- **Tests.** The new module passed twice, 26 tests. `ray.is_initialized()` is false after it, and
  every store is in a `TemporaryDirectory`.
- **Mutations.** All five of the log's diffs applied with plain `git apply`, with no flags, and gave
  exactly the logged results: (i) `failures=2`, (ii) `failures=6`, (iii) `failures=16` across 11
  tests, (iv) `failures=2` (the full-row comparison), and (v) `failures=3, errors=6`. The tree was
  clean after each revert.
- **`var/`.** The snapshot (8 store files, 32 run files, 6 root `physics-test-n20-*` files, 10
  directories) was byte-identical before and after, including the live A3 primary's SHA-256.
- **Suites.** All six re-run: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore 232 (+26), LiouvilleGreen 148 (1 skipped), RunRegistry 199. The ComputeTargets
  wall-clock flake fired once, with the six suites running in parallel, and a serial re-run gave
  552 OK. `black --check` is clean.
- **Boards.** The store-fingerprint entry is narrowed, not closed, and says what 01 made loud. The
  index row changed with it, and the count stays at 104.

**Carried to prompt 02's orchestrator notes.**
- The stand-in pool is `Datastore/tests/standin_pool.py` (`StandinCluster`), and the exceptions are
  in `Datastore/replication.py`. 02 imports both and copies neither.
- 02's reopen table is built on log 01's commit-point table. It must treat a record set over
  **identical** shards as the common case: a get that inserted nothing, or a controller that
  raised before committing. Verify agreement, then clear it. `store_id` is NULL for a vectorized
  get, and for the windows before `P2` in scalar get and store. `started` equals the timestamp
  stamped on every row the write inserted.
- A store written before 01 has no `replication_in_flight` table until the constructor creates it.
  02's check runs after `_check_shard_files`, and must read a missing table as "no record".
- **Until 02 lands, nothing should run a real build on this tree.** A record left set refuses every
  replicated write, and only deleting the row by hand clears it (log 01, observation 2). Prompt 10
  is the first real run in any case.
- Opening an existing store with this tree adds an empty table to its primary. The sidecar records
  no file hash, and the inventory reads only shards, so no fingerprint moves (log 01,
  observation 3).

**Orchestrator review of prompt 04 (2026-09-26).** Run against `6a97d8a`, dispatched from
`304ad95` with the tree and the index clean. Checks 1, 5, 6 and 8 passed. Checks 2, 3, 4 and 7
passed with the findings below. Nothing on a build path, so §5's stop did not fire.
- **Scope.** Five files: the one named test module, the log, the two boards and the index.
  `Datastore/SQL/` and `docs/datastore-integrity-audit/` are untouched, and no other test file is
  modified.
- **Nothing weakened.** Every removed line in the module is docstring point 3 or 6, the old
  `typing` import, `analyse_factories`' docstring, the `build()`-only loop, or `KNOWN_UNFIXED = {}`,
  and each is replaced by a stronger form. All 20 `EXPECTED_COVERAGE` entries at `HEAD~1` are still
  there. The one new `KNOWN_BLIND` entry has a reason. The four tests keep their names and
  assertions.
- **`KNOWN_UNFIXED`.** Exactly the five the probes predicted, none new: `WKB_criterio` ×2 (P6) in
  `KNOWN_UNFIXED`, and `z_serial`, `model_id` (P6) and the `validated` insert key (P8) in a new
  `KNOWN_UNFIXED_COLUMNS`. Check 6: those last three are `probe_03.out`'s three.
- **Live paths**, checked independently of the log. `_build_impl_*` is reached only through its
  own Value factory's `build()`. No `object_get` in `main.py`, `ComputeTargets/` or `extract_*.py`
  names a Value class, and no factory calls a Value factory's `build()`. `main.py` reads those rows
  only through `object_read_batch` (`main.py:2429`). `OneLoopIntegral` is never built. So all five
  defects are dead code: not reached during a build, on a resume, or from extraction.
- **Mutations.** (i)–(iv) applied with plain `git apply` and gave their logged results: (i)
  `failures=2`, (ii) `failures=1`, (iii) `failures=1` (G2 only), (iv) `failures=1`. **(v) did not
  apply** (finding 4). With only its hunk header corrected it gave the logged `failures=4`,
  `test_analyser_coverage_is_undiminished` among them. The tree was clean after each revert.
- **`var/`.** Snapshot of 46 files and 10 directories, byte-identical before and after.
- **Suites.** Run serially: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore 235 (+3), LiouvilleGreen 148 (1 skipped), RunRegistry 199. `black --check` is clean.
- **Boards.** The issue is on the datastore-readback board's §4 with a closing note naming the
  commit by subject. Its index row is gone, and the count went from 104 to 103.

**Findings.**
1. **Reach is not "every function" (check 3).** The analyser visits `build`, `read_batch` and a
   hard-coded pair, `_ROW_HELPERS = {"_build_impl_serial", "_build_impl_model"}`. G1 asked for
   every function, and item G1 above says so. Measured: `validate_on_startup` holds a decidable row
   binding in seven factories (`BackgroundModel`, `GkNumericIntegration`, `GkSource`,
   `GkWKBIntegration`, `QuadSource`, `TkNumericIntegration`, `TkWKBIntegration`). None is visited,
   and **all seven are clean today**. A future helper under another name would be skipped with no
   test failing. The log does not record this as a deviation.
2. **The coverage pin does not see the helpers separately (check 3).** A binding's identity has no
   function name in it, so each class's `_build_impl_serial` and `_build_impl_model` collapse into
   one key (16 → 10; the log explains the arithmetic but not this consequence). Measured by removing
   one name from `_ROW_HELPERS`:
   - removing `_build_impl_model` **fails no test at all**;
   - removing `_build_impl_serial` fails only because both `WKB_criterio` entries live there. Once
     prompt 05 strikes them, that also fails nothing.

   Mutation (v), which removes everything together, is caught. Removing the helpers one at a time
   is not.
3. **G2 skips an unresolved table without declaring it (check 3).** G2 asked that "what cannot be
   resolved" be declared in `KNOWN_BLIND` with its reason. `KNOWN_BLIND_COLUMNS` declares only
   dynamic *names*. The 30 references whose *table* is not resolved (29 `subquery.c.*`, 1
   `query.c.serial`, of 1,224) are skipped. Docstring point 6 mentions them, but no test pins them,
   so a new unresolvable base would be skipped silently. G2 also walks only class methods, never
   `OneLoopIntegral`'s module-level `read_batch` (21 references). I resolved those 21 by hand, and
   all name real columns, so **nothing is missed today**.
4. **Mutation (v)'s recorded diff is corrupt (check 7).** Its header says `@@ -462,16 +462,12 @@`,
   but the hunk has 14 old lines and 10 new. `git apply` refuses it (`corrupt patch at line 24`).
   The log says all five passed `git apply --check`. With `-462,14 +462,10` it applies and behaves
   as logged. The log is not edited; this entry is the correction.
5. **`KNOWN_UNFIXED` is two structures, not one (unlogged).** README §4 fixes the interface as one
   `KNOWN_UNFIXED` holding every defect. Column-existence defects are keyed by
   (module, class, function, table, name), not the module's `module:class → [attrs]`, and they went
   into a separate `KNOWN_UNFIXED_COLUMNS` with its own currency test. That is a reasonable shape,
   but prompt 05 must be written to empty **both**.

None of these hides a defect in today's tree. Findings 1–3 limit how well the instrument protects
prompt 05's work against a later refactor. **Whether they are fixed before 05, carried into 05, or
accepted is the user's call**, and prompt 05 is released to be written once that is decided.

**Decided (2026-09-28):** the user chose to fix findings 1–3 before 05, in a short follow-up prompt,
[`04b-tighten-the-read-back-guard.md`](04b-tighten-the-read-back-guard.md) (items G4–G6). Its target
figures come from [`orchestrator/review_04_measurements.py`](orchestrator/review_04_measurements.py)
and its `.out`. Findings 4 and 5 stand as recorded here: 04b takes its mutation diffs from `git diff`, and
05 is written to empty both unfixed lists. 05 is released once 04b has landed and been reviewed.

**Orchestrator review of prompt 04b (2026-09-28).** Run against `c6432dd`, dispatched from
`9330c49` with the tree and the index clean. All seven checks in `orchestrator/prompt-04b.md` §3
passed, with no findings.
- **Scope.** Three files: the named test module, the log and this board. `Datastore/SQL/`,
  `docs/` and `orchestrator/` are untouched, and no other test is modified.
- **Nothing weakened.** Every removed line is a coverage or blind key rewritten to its
  function-qualified form, `_ROW_HELPERS` and its gate, docstring points 3 and 6 (point 3 extended,
  not cut), or G2's function body, which is replaced by a superset. Computed against `6a97d8a`: all 30
  of 04's `EXPECTED_COVERAGE` entries are projections of the new 43, and `KNOWN_UNFIXED`,
  `KNOWN_UNFIXED_COLUMNS` and `KNOWN_BLIND_COLUMNS` are identical as objects.
- **Reach (finding 1).** `iter_functions` yields every class method and module-level function, and
  no name list is left. The module's own findings give 43 decidable and 2 blind, equal to its pins
  and to `review_04_measurements.out` §D.
- **The pin (finding 2).** Skipping each of `build`, `_build_impl_serial`, `_build_impl_model`,
  `read_batch` and `validate_on_startup` alone fails the coverage pin, losing 20, 6, 6, 4 and 7
  entries. This is independent of `KNOWN_UNFIXED`, so it survives prompt 05 striking the
  `WKB_criterio` entries.
- **G2 (finding 3).** `KNOWN_UNRESOLVED_TABLES` holds 12 pairs (44 references) and
  `KNOWN_UNCHECKED_INSERTS` 13 sites; each equals what the code finds. No resolution rule was used,
  so there was nothing to test for soundness. `OneLoopIntegral`'s module-level `read_batch` is
  walked. The one `continue` left on a dict-literal insert whose target does not resolve catches
  nothing today (measured: 0), and every `inserter` call has two arguments.
- **Mutations.** All six, extracted from the committed log, applied with plain `git apply` and gave
  the logged `failures=1` on the logged test: (i)/(ii) the coverage pin, (iii) the guard proper,
  (iv) column existence, (v)/(vi) the unresolved-table pin. The tree was clean after each.
- **`var/`, suites, board.** Snapshot byte-identical (46 files, 10 directories). AdaptiveLevin 32,
  ComputeTargets 552, CosmologyModels 39, Datastore 237 (+2), LiouvilleGreen 148 (1 skipped),
  RunRegistry 199, run serially; `black --check` clean. The §1 row, G4–G6 and §5.3 are present.

**Carried to prompt 05's writing.**
- 05 empties **both** unfixed lists: `KNOWN_UNFIXED` (2, P6) and `KNOWN_UNFIXED_COLUMNS` (3; P6 ×2,
  P8). Each currency test forces the fix to strike its own entry.
- P8 makes `OneLoopIntegral.read_batch` a method with a return value. That moves its entries: the
  `<module-level>` `KNOWN_BLIND` entry and `KNOWN_UNRESOLVED_TABLES` pair go, and a class-method
  binding joins `EXPECTED_COVERAGE`. 05 must make those pin edits deliberately, each with a reason in
  its log. Afterwards, mutation (vi)'s case (skip module-level functions) has nothing left to catch.
  That is correct, not a hole.
- Any row read 05 adds or moves shows up as a change to a function-qualified pin. 05's log states
  each one.

**Prompt 05 is released to be written** (2026-09-28): its charter is README §2's, and its method
follows D2 and D3.

**Prompt 05 written (2026-09-28), as 05a–05c.** P1–P9 span eight factory files, two live read
paths and five round trips, and several round trips cannot pass until the mechanical defects are
fixed. So the charter is split three ways: mechanical first (05a: P6, P7), then the integral
`read_batch`es (05b: P1, P2, P8), then the value reads' parent key (05c: P3–P5, closing P9). They
run in one session under [`orchestrator/prompt-05.md`](orchestrator/prompt-05.md), each reviewed
before the next is dispatched. Two findings came from writing it:
- **`TkWKBValue`'s reads omit `z_init` too**, and refuse nothing when two parents exist for one
  `k`. It is the same class as P4, which the audit named only for `GkWKBValue`. P4 is widened, and
  05c covers it.
- **05a's and 05b's pin movements were simulated** against the guard on a scratch copy of the
  factories. 05a moves nothing but the struck entries. 05b adds one coverage entry, removes one
  blind entry and three unresolved-table pairs, and empties `KNOWN_UNFIXED_COLUMNS`. Each prompt
  lists exactly these, and any other movement is a stop.

`GkSourceValue` and `QuadSourceValue` were checked while writing: their reads match their parents'
keys, so neither needs a key change.

**Orchestrator review of prompt 05a (2026-09-28).** Run against `7ce8c66`, dispatched from
`fd6205e` with the tree and the index clean, nothing running, and the §5.3 baselines re-measured
on `HEAD`. Every check in `orchestrator/prompt-05.md` §3 that applies to 05a passed. There are
findings on the log's record, and one is carried into 05c's brief. None needs the user.
- **Scope.** Ten files: the six named factories, the new `test_read_path_mechanics.py`, the guard
  module, the log and this board. `main.py`, `extract_*.py`, `docs/` and `orchestrator/` are
  untouched. The guard module is the only modified test. All 17 factory hunks are §2 changes: two
  `WKB_criterion`s, `z_source_serial`, `model_serial`, `friction=row.friction`, nine tags defaults
  with their annotations, and three `RuntimeError`s. Each renamed column is declared in its table,
  and `friction` was already selected. No `raise (`, `raise "` or `get("tags", None)` is left except
  `OneLoopIntegral.py:284`, which is 05b's.
- **The guard.** Only the struck entries moved: `KNOWN_UNFIXED` → `{}`, and `KNOWN_UNFIXED_COLUMNS`
  lost `z_serial` and `model_id`, keeping `OneLoopIntegral`'s `validated`. This is §0's prediction.
  Recomputed from the new tree with the guard's own functions, every pin equals what is pinned:
  coverage 43, blind 2, no shortfalls, one column problem, 12 unresolved-table pairs and 13
  unchecked inserts.
- **The fixes.** Each has a test that goes through it on the real factory code: `build()` by serial
  returning a distinctive `WKB_criterion`, a hit and a miss by `z_source` and by model, `friction`
  tagged and untagged, omitted tags against `tags=[]` at all nine sites (each site's loop runs
  before any early return, so a `None` would raise), and the three raises.
- **The tags recount.** Explained by the review, not the log. The audit's tree (`417c647`) has ten
  `get("tags", None)` sites, the same ten as `fd6205e`. The audit's "eleven" (§3.11) is not marked
  **[checked]**, so it is a miscount. The log's "cannot be reconciled" is superseded by this entry.
- **Mutations.** All six, extracted from the committed log, applied with plain `git apply` and
  reverted clean. Each failed the tests the log names.
- **`var/`, suites, `black`.** The snapshot was byte-identical (46 files, 10 directories).
  AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 257 (+20, the new module's
  20 methods), LiouvilleGreen 148 (1 skipped), RunRegistry 199, run serially. `black --check` is
  clean.

**Findings.**
1. **Three of 05a's tests will meet 05c's required tolerances** (carried into 05c's brief).
   `TestTagsOmittedDefaultToEmpty`'s `test_gk_numeric_value_build_impl_model`,
   `test_gk_numeric_value_read_batch` and `test_tk_numeric_value_build_impl_model` omit `atol` and
   `rtol`. 05c §2 makes those keys required, so the three will raise `KeyError`. Their fixtures'
   break-point kinds are the compute targets' constants, and each has one parent, so no other 05c
   change affects them. The orchestrator names these three tests, for this one reason, under README
   §5 rule 7. 05c may add `atol`/`rtol` stand-ins matching the stored `atol_serial`/`rtol_serial`
   to exactly these three payloads, with the failing assertion shown in its log. Nothing else in the
   module changes.
2. **The mutation record under-reports** (log not edited). (ii) also errors
   `test_gk_source_value_build_impl_serial`, because `z_serial` raises before the `RuntimeError`.
   (iii) also errors `TestTagsOmittedDefaultToEmpty.test_tk_wkb_value_read_batch`. (vi) also fails
   the guard proper, `test_column_references_resolve_against_schema`. Every logged failure occurs.
   The extra ones are what the mutations must cause.
3. **The replay paragraph was written before the commit.** It describes the post-commit replay in
   the past tense. The review's own replay confirms what it says.

**Prompt 05b landed 2026-09-28**, from `7ce8c66` (the 05a commit reviewed at `76e813b`). P1, P2 and
P8 shipped: the tagged join in both `QuadSourceIntegral.read_batch` and
`sqla_OneLoopIntegral_factory.read_batch` on `table.c.serial`; the exit times' own serials in
`QuadSourceIntegral.read_batch`; `OneLoopIntegral`'s factory repaired to D3's bound (its own tag
table, `read_batch` a method with a return, tags default `[]`, `"validated"` dropped from `store()`)
with `source_serial` left untouched. `KNOWN_UNFIXED_COLUMNS` is empty. Every guard pin movement
matches §2's list exactly, each forced by the failing assertion the guard gave before the pin was
edited (log §"What shipped"). Ten new tests in `Datastore/tests/test_integral_read_batch.py`; no
existing test changed. Log: [`logs/05b-…`](logs/05b-repair-the-integral-read-batches.md). This is
the implementer's own record; the orchestrator's review, when it runs, is recorded separately
below, in the form the 01/04/04b/05a reviews above are.

**Orchestrator review of prompt 05b (2026-09-28).** Run against `a293423`, dispatched from
`76e813b`, the review of 05a, not `7ce8c66` as the paragraph above says. At dispatch the tree and
the index were clean, nothing was running, and the §5.4 baselines had been re-measured on `HEAD`.
Every check in `orchestrator/prompt-05.md` §3 that applies to 05b passed. There are findings on the
record, and none needs the user.
- **Scope.** Eight files: the two factories, the new `test_integral_read_batch.py`, the guard
  module, the log, both boards and the index. `main.py`, `extract_*.py`, `ComputeTargets/`,
  `orchestrator/` and every other `docs/` file are untouched. The guard module is the only modified
  test, contrary to the "no existing test changed" above.
  - `QuadSourceIntegral.py`: four hunks, all in `read_batch`. They are the join (P1), and the two
    exit serials selected and used as `store_id` (P2).
  - `OneLoopIntegral.py`: `build()` joins `OneLoopIntegral_tags`, `"validated"` leaves `store()`, and
    the `inventory_records` comment is corrected. `read_batch` moved into the class. Diffed
    de-indented against the old function, it differs in five places. Three are §2's: tags default
    `[]`, the join, and the `return`. The fourth is the `# TODO: not yet complete` removal, which
    the log justifies. The fifth is unlogged (finding 1).
  - No code line mentioning `source_serial` is added or removed; the word appears only in tests,
    docs and one comment.
- **The guard.** Exactly §0's prediction, and nothing else moved:
  - `EXPECTED_COVERAGE` 43 → 44, gaining `OneLoopIntegral.py:sqla_OneLoopIntegral_factory.read_batch.row_data`;
  - `KNOWN_BLIND` 2 → 1;
  - `KNOWN_UNRESOLVED_TABLES` 12 → 9, the three named pairs;
  - `KNOWN_UNFIXED_COLUMNS` → `set()`.

  The log shows each movement forced by the failing assertion. Recomputed with the guard's own
  functions, every pin equals what is pinned. **Both unfixed lists are empty.**
- **The fixes.** P2's fixture gives the exit-time rows serials 11–13 and their wavenumbers 1–3, and
  the test asserts 12/13 against 2/3. The `OneLoopIntegral` stand-in is always patched with
  `mock.patch.object` as a context manager. The coincidence case puts only a
  `QuadSourceIntegral_tags` row at the `OneLoopIntegral` serial, and asserts a miss.
  `TestOneLoopIntegralSourceSerialStaysBroken` constructs the real target and pins
  `KeyError('source_serial')`. Its docstring names `[00-oneloop-factory-cannot-round-trip-a-row]`
  and says whoever closes it must change the test.
- **Mutations.** All six, extracted from the committed log, applied with plain `git apply`. Each
  gave exactly the logged counts: (i) `failures=1, errors=2`, (ii) `failures=1`, (iii)
  `failures=2`, (iv) `errors=3`, (v) `failures=1`, (vi) `errors=1`. Each reverted clean. The 05a
  review's finding 2 is answered: each record lists every failing test.
- **`var/`, suites, `black`, boards.** The snapshot was byte-identical. AdaptiveLevin 32,
  ComputeTargets 552, CosmologyModels 39, Datastore 267 (+10, the module's 10 methods),
  LiouvilleGreen 148 (1 skipped), RunRegistry 199, run serially. `black --check` is clean. Both issues
  are on the store-fingerprint board's §4, each with a closing note naming the commit by subject.
  Their index rows are gone, the narrowed row is rewritten, and the count went from 103 to 101
  (rows counted: 103 → 101), dated 2026-09-28.

**Findings.**
1. **An unlisted edit to `OneLoopIntegral.read_batch`.** The unused `tol_table =
   tables["tolerance"].alias("tol")` was dropped in the move. It changes no behaviour, since the
   alias was never joined, but it is outside D3's list and the log does not record it. It should
   have been an `UNINTENDED DRIFT` entry. It is accepted here, and not reverted.
2. **The log's hand-off miscounts one pin.** It gives `KNOWN_BLIND_COLUMNS` as 3 entries. It has 4,
   unchanged since 04b. 05c's brief carries the correct starting pins: coverage 44, blind 1,
   unresolved 9, unchecked inserts 13, blind columns 4, and both unfixed lists empty.

Nothing new carries into 05c beyond the 05a review's finding 1.

**Prompt 05c landed 2026-09-28**, from `199be29` (the review of 05b). This is the implementer's own
record; the orchestrator's review, when it runs, is recorded separately.
- **What shipped.** P3, P4 and P5, and P9's close.
  - The numeric value reads filter the parent on `break_point_kind`, and require `atol`/`rtol`.
  - `TkNumericValue` reads one key, `z_init`.
  - The Gk and Tk WKB value reads take an optional `z_init`, matched with the parents' own
    expression.
  - The three value `read_batch`es refuse with `MultipleResultsFound`, naming the parents, when
    more than one parent contributes rows for one `z_source` (Gk), or for one `(model, k)` (Tk
    WKB).
  - Both `…_is_response` labels read `response`.
  - `READ_BATCH_ROUND_TRIPS` pins every `read_batch` to its round-trip module.
- **Guard and tests.** No guard pin moved. `main.py` is unchanged.
- **The §6 stop.** One stop fired and was decided. `test_read_path_mechanics.py`'s
  `TestTkWKBValueFriction.test_read_batch_untagged_returns_every_row_with_its_own_friction`
  asserted a two-parent mixture. The user decided the fixture was wrong, not the refusal, and it
  now reads a one-parent store. With the three payloads the 05a review named, those are the only
  existing-test edits.
- **Issue opened.** `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]` (§3.3).
- **Log:** [`logs/05c-…`](logs/05c-key-the-value-reads-on-their-parent.md).

**Carried to 07 and 09** (log 05c, observations 1–2).
- **For 09's K6: the refusal is decided on the rows returned.** When `z` is given, as `main.py`
  always gives it, a second parent matching the key but with **no value row at that `z`** is not
  seen. The read returns the other parent's value, unrefused. K6 plans to read the `GkSource`
  parent set through these `read_batch`es. If it needs every parent matching the key, it must
  decide that on the parent query, not on the value rows.
- **`TkNumericValue.build`'s dispatch** (and `GkNumericValue.build`'s) decides the model path on
  `model`, `k` and `z_init` (`z_source`) alone. A payload that has those but lacks a tolerance
  passes the dispatch and fails with `KeyError` inside `_build_impl_model`, not with the
  dispatch's `RuntimeError`. No caller outside the tests reaches either model path. Separately,
  `test_read_path_mechanics.py`'s `test_tk_numeric_value_build_impl_model` keeps a stale comment
  and an unused `z_source` key, both outside 05c's permitted edits.
- **For 07.** The four `z_init` filters in the WKB value reads are what 07's `numeric_serial`
  replaces. The refusal keys on `wkb_serial`, and stays valid.

**Orchestrator review of prompt 05c (2026-09-28).** Run against `1d0f08f`, dispatched from
`199be29` (the review of 05b) with the tree and the index clean, nothing running, and the §5.5
baselines re-measured on `HEAD`. The dispatch stopped once, at 05c §6's "an existing test fails".
The user decided the stop in chat (see the landing paragraph above) and the agent resumed.
Every check in `orchestrator/prompt-05.md` §3 passed.
- **Scope.** Ten files: the four named factories, the new `test_value_reads_parent_key.py`,
  `test_read_path_mechanics.py`, the log, both boards and the index. `main.py`, `extract_*.py`,
  `ComputeTargets/`, the guard module, `orchestrator/` and every other `docs/` file are
  untouched. In `test_read_path_mechanics.py` there are exactly four edits: the three payloads the
  05a review named gain `atol`/`rtol`, and the friction test reads its own one-parent store, as the
  user decided. Every factory hunk is a §2 change:
  - `break_point_kind` and the required tolerances in the three numeric reads, with a refusal like
    the parents' for a store without the column;
  - `z_init` as `TkNumericValue`'s one key, with its two messages corrected;
  - the optional `z_init` in the four WKB reads;
  - the parent serial selected into each value `read_batch`'s row query, and the refusal;
  - P5's two labels.
- **The refusal.** Read in the code:
  - `GkNumericValue.read_batch` and `GkWKBValue.read_batch` group the fetched rows by
    `z_source_serial` and raise `MultipleResultsFound` when a group has more than one distinct
    parent serial (`integration_serial` / `wkb_serial`);
  - `TkWKBValue.read_batch` raises when the rows hold more than one `wkb_serial`.

  One parent returns what it returned before, and no parent returns `[]`. There is one query, and
  the refusal is decided on its rows. I ran test 3, `main.py`'s payload shape over one parent per
  `z_source` at three `z_source`s, and both live reads passed.
- **The keys match the parents.** `break_point_kind` is compared with
  `GkNumericIntegration.BREAK_POINT_KIND` / `TkNumericIntegration.BREAK_POINT_KIND`. These are the
  compute targets' constants, the same expression as the parents' `build()` (`GkNumericIntegration.py:249`,
  `TkNumericIntegration.py:253`). `z_init` is matched with the parents' own
  `abs(z_init − z) < DEFAULT_FLOAT_PRECISION` (`GkWKBIntegration.py:280`, `TkWKBIntegration.py:287`).
- **Live paths.** Checked independently by `git grep` over every `.py` outside the factories and
  tests. The only caller of the six reads is `main.py`'s `build_missing_GkSource`
  (`:2391–2429`), which reads `GkNumericValue` and `GkWKBValue` through `object_read_batch`.
  `main.py:2412` already passes the tolerance pair for `GkNumericValue`, and passes no `z_init`.
  Nothing else reads the value classes. `docs/gktk-remedial/analyse_scoped_run.py` and
  `tools/inventory_report.py` only name their tables, and `probe_05` is a measurement record.
  Nothing outside the tests reads `redshift.is_response`, so P5 changes no live behaviour. No store
  under `var/` was opened. Nothing in the code suggests a store built under one policy and one set
  of run tags holds two parents per `(k, z_source)`. Where one does, the read now refuses at
  exactly the point where `main.py` would have silently overwritten one parent's value with the
  other's.
- **The `read_batch` pin.** `READ_BATCH_ROUND_TRIPS` has five entries. They equal both the test's
  own walk and an independent AST walk of `ObjectFactories/`: `GkNumericValue`, `GkWKBValue`,
  `TkWKBValue`, `QuadSourceIntegral`, `OneLoopIntegral`. **P9 is closed.**
- **The guard.** Nothing moved, as predicted. Recomputed with the guard's own functions, every
  pin equals what is pinned: coverage 44, blind 1, 9 unresolved-table pairs, 13 unchecked inserts,
  no shortfalls and no column problems. The one shape chosen for the guard's sake, a `for` loop in
  place of a set comprehension, is logged, and it opened the issue below.
- **Mutations.** All six, extracted from the committed log, applied with plain `git apply` and gave
  exactly the logged counts: (i) `failures=1, errors=1`, (ii) `failures=4`, (iii) `errors=2`, (iv)
  `failures=2`, (v) `failures=1`, (vi) `failures=1`. Each reverted clean.
- **`var/`, suites, `black`, boards.** The snapshot was byte-identical (46 files, 10 directories),
  as it was before every dispatch this session. AdaptiveLevin 32, ComputeTargets 552,
  CosmologyModels 39, Datastore 287 (+20, the new module's 20 methods), LiouvilleGreen 148
  (1 skipped), RunRegistry 199, run serially. `black --check` is clean. The issue is on the
  store-fingerprint board's §4, with a closing note naming the commit by subject. The index went
  from 101 to 101 (one row deleted, `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`
  added; rows counted), dated 2026-09-28.

**Findings.**
1. **The 05a review missed the two-parent friction fixture.** Its finding 1 said the tolerance
   payloads were the only collision with 05c. `TestTkWKBValueFriction` also stored two parents for
   one `(model, k)` and read them with `tags=[]`. The agent stopped correctly, and the user decided
   it. The lesson for later reviews: when a prompt adds a refusal, check every existing fixture
   against the refusal's condition, not only against the fields the prompt changes.
2. **The scratchpad was shared.** The 05c agent wrote its own `replay.py` over the orchestrator's
   script of the same name in the session scratchpad. Its assertion fired before anything was
   written, and the tree was untouched. Later orchestrator notes should give each dispatched agent
   its own scratch subdirectory.
3. **The row-based refusal**, carried to 09 above, is as 05c §2 specifies. It is not a defect of
   05c. It is a limit that 09's K6 must not rely past.

None needs the user.

**Prompt 05 has landed (2026-09-28), as 05a–05c, all reviewed.** Its charter is met:
- P1–P9 are done, and P9 is closed;
- `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are both empty;
- its three issues are closed: `[00-tagged-read-batch-joins-an-unselected-alias]`,
  `[00-oneloop-lookup-joins-the-wrong-tag-table]` and
  `[00-numeric-value-parent-lookup-omits-break-point-kind]`.

**Next dispatches: 02 → 03 → 06**, each with its own orchestrator note, written just before it is
dispatched. 07's gate is 05 **and** 06, and 06 follows 03, which follows 02. So 07 is not next,
although 05 has landed.

**Carried to 07's orchestrator notes** (in addition to the three bullets above):
- The WKB value reads now take an optional `z_init`, filtered with the parents' absolute
  `DEFAULT_FLOAT_PRECISION` expression, and refuse a second parent: per `z_source` for
  `GkWKBValue`, per `(model, k)` for `TkWKBValue`. When 07 removes `z_init` from the key and adds
  `numeric_serial`, it replaces those four filters. It does not replace the refusal, which is
  decided on `wkb_serial`.
- `test_value_reads_parent_key.py`'s `TestZInitSelects` and the two `…two_z_inits` refusal tests
  exercise `z_init` as a key. 07 must name them and say how they follow its key (README §5 rule 7).
- `main.py`'s `build_missing_GkSource` passes no `z_init`, so today it relies on the refusal alone.
  07's own change to `main.py`'s existence queries must keep test 3's live payload working.

**Prompt 02 landed 2026-09-28**, from `65ee1d3` (its orchestrator note). This is the implementer's
own record; the orchestrator's review, when it runs, is recorded separately.
- **What shipped.** W5–W7.
  - The check. `ShardedPool._reconcile_replicated_tables` runs in the constructor after
    `_check_shard_files`, before the broker or any actor, with no Ray. It rolls back hot journals
    and compares every replicated table and `BackgroundModel_tags` across the shards by full row:
    keyed by serial, `BackgroundModelValue` by `(model_serial, z_serial)` without its serial, the
    tag table by its composite key, `timestamp` never compared. It also checks the primary's
    `shard_keys`.
  - The repair. With a record set, it copies forward verbatim from the record's controller, and
    only rows of the record's class that the recorded write inserted. A `BackgroundModel` goes as
    a unit, a flag is only turned on, and `validated` is recomputed by the factory's rule. It then
    compares again inside the repair's transactions, and clears the record last.
  - The refusal. Anything else raises the new `ReplicatedDivergence`
    (`Datastore/replication.py`), and nothing is written.
  - A clean store opens writing nothing, checked by checksum. The added open time is about 0.1 s
    at 4 shards and 0.25 s at 20, at production size.
- **Tests.** The new `Datastore/tests/test_reconcile_at_open.py` has 38 tests, and the log's
  reopen table gives each row of log 01's commit-point table one outcome. The one test the
  orchestrator note names, `TestWriteOverASetRecord.test_a_reopened_pool_refuses`, is rewritten to
  `…_repairs_from_the_controller_and_the_write_succeeds` on the same fixture. No other existing
  test changed.
- **Mutations.** (i)–(v) each fail tests.
- **Issue closed.** `[00-replicated-writes-can-diverge-across-shards]`, on the store-fingerprint
  board's §4.
- **Issue opened.** `[02-the-version-row-is-written-outside-the-in-flight-record]` (§3.3).
- **Log:** [`logs/02-…`](logs/02-repair-or-refuse-at-open.md).

**Orchestrator review of prompt 02 (2026-09-28).** Run against `f4b2c91`, dispatched from
`65ee1d3` with the tree and the index clean, nothing `running`, and no other worktree. All eleven
checks in `orchestrator/prompt-02.md` §3 passed. The repair and refusal logic was read against the
code, not the log.
- **Scope.** Nine files: the four permitted code files, the new test module, the log, the two
  boards and the index. The forbidden paths are empty, and `docs/` changes only in
  `docs/OPEN_ISSUES.md`. `--diff-filter=M` over tests shows only `test_replicated_write.py` and
  `standin_pool.py`, which gains two functions and changes none.
  - `ShardedPool.py` has six hunks: two import lines (deviation 9), three in the constructor, and one
    block of new methods placed straight after `_check_shard_files` ends, before the shared
    `SHARD RECORDS` static helpers.
  - No hunk is in `_replicated_write`, the three `_…_impl_replicated_table` methods,
    `_assign_shard_keys` or the closed-store block.
- **Before any actor, no Ray.** The call is on the existing-store branch only, after
  `_check_shard_files`, before the in-flight table's `create(checkfirst=True)` and the broker. A
  new store skips it and keeps `reconciliation = None`. README §4's three names exist as fixed.
- **The comparison is W5's.**
  - **Classes.** They come from the constructor's `replicated_tables`, `config/sharding.py`'s list,
    checked here: all 13 names have a table, so none is skipped silently.
  - **Tag tables.** The foreign-key rule picks up `BackgroundModel_tags` and nothing else. The other
    eight tag tables have sharded parents.
  - **Keys.** Rows are keyed by serial, with only `timestamp` excluded. `BackgroundModelValue` is
    keyed by `(model_serial, z_serial)` without `serial`, and the tag table by its primary key.
    Those are the census's exclusions exactly (`replicated_serial_census.py`).
  - `shard_keys` is checked against every shard's `wavenumber` rows. No sharded table is read.
- **Repair only under the licence.**
  - **Code.** `_plan_replicated_repair` returns early with no record. It copies only from
    `controller_shard`, refuses any difference outside the record's unit, and refuses a row the
    controller lacks.
  - **What is copied.** Rows go over verbatim as full mappings read from the controller. A
    `BackgroundModel` goes only as a whole unit.
  - **Flags.** A flag is turned on only False → True, and only after a get.
  - **`validated`.** It is recomputed by the factory's own `validate`, driver-side, inside a
    savepoint. Anything that does not turn it on is rolled back. The factory writes `False` when
    it counts short, so the savepoint is what keeps a `False` off disk.
  - **The order.** The second comparison reads inside the open transactions. The shards commit, then
    the record is deleted on the primary, with a rowcount check.
  - **Grep.** Over the added lines, the only `DELETE` is the record's, and the only `UPDATE` sets
    `True`.
  - **Deviation 1's extra conditions are accepted.** They are the `started` timestamp, the
    `store_id` serial, no rows under a validate, the whole unit, and free value serials. Each only
    refuses more.
  - **Deviation 10 is accepted.** The removed "equal but for serial" check was redundant: a split
    always holds a row the controller lacks.
- **Refusal otherwise.** Every refusal test goes through `refused_reopen`. It asserts the checksum
  of every file in the store's directory, and that no actor was constructed. The split tests assert
  both serials, the class, "Nothing was repaired" and "needs a person, or regeneration".
- **The reopen table.** It has 19 rows, one per row of log 01's "After" table. Each disk state has
  one outcome and a test:
  - "same, record set" is cleared;
  - "controller holds more" is repaired from the controller;
  - a mismatch's split is refused;
  - every kill-and-reopen case ends with a second reopen that repairs nothing and changes no
    checksum.
- **The named test.** One hunk in `test_replicated_write.py`: the method is renamed and rewritten,
  and its fixture and siblings are untouched. It asserts the copy from the controller, the cleared
  record and the write that succeeds.
- **A clean store; the open time.** Test 7 checksums the primary, every shard and any journal. The
  check costs 94 ms at 4 shards and 252 ms at 20, at production size. That is small beside starting
  Ray actors, so it is not raised.
- **Callers, and extraction.** `grep` finds `main.py`, the six `extract_*.py`, the two
  `docs/source-remediation-verification/` scripts and the audit's fault probe (stand-in shards).
  **Every extraction now runs the check at open.**
  - A clean store opens as before.
  - A divergent one is repaired or refused before any actor exists.
  - No store under `var/` was opened to see which.
- **Mutations.** (i)–(v) were replayed from `git show HEAD:<log>` with plain `git apply`. Each gave
  exactly its logged counts and the same distinct failing tests. The tree was clean after each
  revert.
  - (i) `failures=4, errors=1`
  - (ii) `failures=1, errors=16`
  - (iii) `failures=16, errors=26`
  - (iv) `failures=1`
  - (v) `failures=1`

  (v) is caught by one test only, the refusal after a repair. That is the only state in which
  clearing the record early is observable.
- **`var/`, suites, boards.**
  - The snapshot (46 files, 10 directories) is byte-identical.
  - Suites, serially: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 325
    (+38), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
  - `black --check` is clean, and `ray.is_initialized()` is false after the new module.
  - The issue is on the store-fingerprint board's §4, with a note naming both commits.
  - The index has 101 rows, counted: one deleted, and one added for the version-row issue.
- **Raised with the user, not blocking.**
  - **The version row.** `[02-the-version-row-is-written-outside-the-in-flight-record]` is a
    state that healed before 02 and now refuses. It is reachable only from a crash while actors
    start on a new store, or under a new `VERSION_LABEL`.
  - **Real stores.** Log 02's observation 3: the check refuses a duplicate `BackgroundModelValue`
    key and a `shard_keys` row naming a missing wavenumber. The census could not see either, and
    §4.4 records no row counts. Nothing known writes either state, and after 07–09 (D4) the live
    A3 store is unreadable by the tree anyway. So this matters only if that store is opened before
    07.
- **Discharged.** Log 01's observation 2, "until 02 lands, a set record refuses every replicated
  write", is discharged: a reopen now repairs or clears the record.

**Next: 03**, with its own orchestrator note carrying this review's findings. 03 changes lookups on
the classes 02 reconciles. Prompt 10 is still the first real run; nothing in 02 has run under Ray.

**Orchestrator review of prompt 03 (2026-09-29).** Run against `f2f026f`, dispatched from
`ae4d8f9` with the tree and the index clean, nothing `running`, and no other worktree. The
baselines were re-measured on `ae4d8f9` first and matched §5.7. **Ten of the eleven checks in
`orchestrator/prompt-03.md` §3 passed. Check 8 did not.** The code is right. What is wrong is
L5's account of what the two verification scripts do after a miss, and that account was copied
into the `[05a-…]` narrowing and its index hook. Both are corrected in this commit, additively.
- **Scope.** Seventeen files: the ten permitted factory files, one new test module, the log, the
  four boards and the index. The forbidden paths are empty, and `docs/` changes only in
  `docs/OPEN_ISSUES.md`. `--diff-filter=M` over `*/tests/*` is empty.
  - Every factory hunk was read. Each is inside its class's `build()`, and none is in a
    `register()`, a `store()` or an `if "serial" in payload` branch.
  - Apart from the change itself, the hunks add only comments.
- **L1.** The filter is `atol_serial`, `rtol_serial` and `stepping` by `==`, with no `log10_tol`
  comparison left. The `order_by` is gone. `conn.execute(query).one_or_none()` runs once, after
  both `add_columns` loops.
  - The joins are kept, and feed the returned `log10_tol` values. The returned `atol`/`rtol`
    carry the stored row's serials, which now equal the requested ones.
  - The orchestrator's sketch did not survive the session, so the shapes were not compared. The
    shape is the one this note's §0 describes, and the guard is unmoved.
- **L2.** `stepping == stepping` with `.scalar_one_or_none()`; the `stepping < 0` clamp is
  unchanged.
- **L3.** Exactly the eight sites change, with no filter.
  - `grep` leaves the 21 other `.scalar()` calls: seven `validate()` triples, each a primary-key
    `z_samples` read plus two `count()`s. The log's list matches them line for line.
  - The `QCD_Cosmology` handler was read, and its test asserts `MultipleResultsFound` itself.
- **L4.** `z_filter` is `table.c.z == 0.0` when `z == 0.0`, and otherwise the unchanged relative
  expression, passed to one `.filter(...)`.
- **Tests.** The 18 tests of `test_exact_identity_lookups.py` cover all five of the prompt's cases:
  - the three misses, the two-row case, and stepping 0 against 1;
  - the solver's 4/8/6;
  - one duplicate for each of the eight classes, and three `z = 0` builds giving one row;
  - test 2 counts `SELECT`s with a `before_cursor_execute` listener and asserts one.

  `tearDownModule` refuses if `ray.is_initialized()` is true, and every store is `sqlite://`.
- **The guard is unmoved.** `test_factory_select_columns.py` has the same blob hash at both
  commits (`0d4fdea`), and passes.
- **L5, production.** Checked here against the code, not the log.
  - `main.py:993` and the six `extract_*.py` all request `DEFAULT_HEXIT_*` = (1e-10, 1e-9)
    (`config/defaults.py:53,63`) with no `stepping`, which is the pair and the 0 every computed
    exit time stores.
  - The seven `main.py:3621-3640` solver lookups each request the stepping they insert.
  - **So nothing in production moves.**
- **L5, the two verification scripts: check 8 fails.** Both do miss after L1. The log's account
  of what happens next is wrong. It says both scripts record every case as a failure and
  finish, and that `analyse_greens_and_source.py` "does not compute, store or crash".
  - **The fact the log missed.** `DatastoreObject.store_id` **raises**
    `RuntimeError("Attempt to read datastore id before it has been set")` on an unpopulated
    object (`Datastore/object.py:13-17`); it does not return `None`. The log assumed that
    `== k_exit.store_id` becomes `IS NULL`.
  - **`analyse_greens_and_source.py`.** Shard routing works, because
    `shard_key_wavenumber_store_id` reads `k_exit.k.store_id`. But `sqla_GkSource_factory.build`
    evaluates `k_exit.store_id` while it builds its filter (`GkSource.py:211`), so the first
    sampled case's `GkSource` lookup raises inside the shard's actor. It reaches the driver
    through `ray.get` at `:273`, and nothing between there and `main()` catches it. The script
    stops with a traceback at its first sampled case, before `if not source.available:`.
  - **`run_quadsource_integrals.py`.** `distinct_k = {k.store_id: k …}` at `:265` raises on the
    driver as soon as `triples` is non-empty. No dict collapses, `payload_ok` is never reached,
    and no failure is counted.
  - **Measured in memory**, with no Ray and nothing under `var/`, using the new module's fixture:
    - a stored (1e-10, 1e-9) row;
    - a real exit-time `build()` at (1e-10, 1e-8), which returns `available=False`;
    - the `:265` comprehension then raises that `RuntimeError`;
    - so does a real `sqla_GkSource_factory.build(k=k_exit, …)`.

    The script was `orch_03_miss_raises.py`, in the session scratchpad.
  - **So the verdict and the charter stand, and the mechanism is corrected.** Both scripts now
    fail loudly on first use, where they used to quietly report the stored pair. Neither computes
    or stores anything. The `[05a-…]` entry on the tolerance-convergence board carries a dated
    correction beneath 03's narrowing, and its index hook is rewritten to match. Log 03 is left
    as written, as the record of what the agent concluded.
- **Mutations.** (i)–(v) were replayed from `git show HEAD:<log>` with plain `git apply`, first on
  the new module and then on the whole Datastore suite. Each gave exactly its logged result and
  the same distinct failing tests, and the tree was clean after each revert.
  - (i) `failures=1`, one sub-test, `(atol=-10.0, rtol=-8.0)`
  - (ii) `failures=1`
  - (iii) `failures=1, errors=1`
  - (iv) `failures=1`
  - (v) `failures=1`

  The agent reports the same results from its own post-commit replay. The log's pointer to a
  "Verification performed (the replay)" section dangles, because that section does not exist.
  That is cosmetic, and the replay itself was done.
- **Boards.** The three issues are on their owning boards' §4, each with a closing note naming
  the commit by subject. `[05a-…]` is narrowed on its board and in the index. The campaign board
  has its §1 row, L1–L5, §3.1 and §5.8, and the `"stepping": "minimum"` flags are handed to 06
  (log observation 1).
- **`var/`, suites, `black`.**
  - The snapshot (46 files, 10 directories) is byte-identical.
  - Suites, serially: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 343
    (+18), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
  - `black --check` is clean on the eleven Python files.
- **Issue opened by this review.** `[03-exit-time-build-comment-describes-the-removed-order-by]`
  (§3.3). The comment at `wavenumber.py:195` still says the query is ordered "so that we get the
  best computed value we hold", which is exactly the behaviour L1 removed. The index goes to
  **99** rows, counted.

**Next: 06**, with its own orchestrator note carrying this review's findings. It corrects the two
`"stepping": "minimum"` flags to say what 03 made true, and is the first schema change, under D5.
Prompt 10 is still the first real run, and nothing in 03 has run under Ray.

**Orchestrator review of prompt 06 (2026-09-29).** Run against `15c724b`, dispatched from
`8ed150f` with the tree and the index clean, nothing `running`, and no other worktree. The
baselines were re-measured on `8ed150f` first and matched §5.8. **All nine checks in
`orchestrator/prompt-06.md` §3 passed.** The review found two small remnants, fixed in
`f070a9d`, and three inaccuracies in the record, noted here.
- **Scope.** Thirteen files: the four factories, one new witness, one new test module, the three
  permitted test-file edits, the log, the two boards and the index. The forbidden paths are empty
  apart from `docs/OPEN_ISSUES.md`. `--diff-filter=M` over `*/tests/*` gives exactly
  `real_store_fixtures.py`, `schema_description.py` and `test_schema_builder.py`, and each hunk
  was read: the map; the constant and the docstring; a docstring. Every factory hunk is inside
  `register()`, on the four named columns or flags.
- **S1, S2, S3.** `Tq_serial` targets `TkNumericIntegration.serial`. `Tr_serial` has no key, and
  its comment names `TkNumericIntegration` and gives the cross-shard reason.
  `wavenumber_exit_serial` targets `wavenumber_exit_time.serial`, with `index` and `nullable`
  unchanged. The map has `"Tr_serial": "TkNumericIntegration"` and no `GkSourcePolicyData` entry,
  and `QuadSourceIntegral.source_serial` is unchanged.
- **S4.** `test_foreign_key_check.py`, 8 tests, runs `foreign_key_check` on every shard, `mode=ro`.
  - Its own rows add `QuadSource` 10 on shard 0, whose `Tq_serial` 3 is a Tk serial there and not
    a `QuadSource` serial. A test asserts that property of the data.
  - The docstring names `Tr_serial`, `QuadSourceIntegral.source_serial` and `shard_keys`.
  - With the parent's `QuadSource.py` checked out over it, 4 of the 8 fail, reporting
    `('QuadSource', 10, 'QuadSource', 0)` on shard 0. After the restore the tree was clean and
    all 8 passed.
  - The stand-in-shard store reports one `GkSource` row by design, from its placeholder
    `source_serial` (log deviation 3). The test pins that row and allows no other, which is sound.
- **S5, D5.**
  - `schema_at_base.json` has blob `4cc19dd` at both commits, and the new witness is a new file.
  - A witness re-captured from `HEAD` into the scratchpad is byte-identical to the committed one.
  - The orchestrator's own `diff` of the two witnesses is **8 hunks and 30 lines**, byte-identical
    to the log's paste. Every line is S1, S2 or one of the two flags.
  - **The note's "9 hunks" was wrong.** Its "one more hunk" for `GkSourcePolicyData`'s constraint
    list is the second of that class's three, counted twice. The agent's account is right.
- **The fingerprint is unmoved.** `full_store_fingerprint.json` has no diff, and
  `test_store_fingerprint` gave 33 OK on the final tree and under each mutation.
- **Mutations.** (i)–(iv) were extracted from `git show HEAD:<log>` and replayed with plain
  `git apply`, each over the whole Datastore suite and `test_store_fingerprint`. Each gave exactly
  its logged failing tests and sub-tests, and the tree was clean after each revert.
  - (i) `failures=8`. It fails S4's `test_every_shard_passes_foreign_key_check` on the added row,
    while `test_the_default_stores_pass_too` passes, which is the coincidence §0 measured.
  - (ii) `failures=9`, including `test_the_default_stores_pass_too`, on the fixture's own
    cross-shard pair.
  - (iii) `failures=10`, the four classes plus the whole file, in each witness test.
  - (iv) `failures=4`: `IntegrationSolver` plus the whole file, twice.

  The agent reports the same results from its own post-commit replay.
- **Boards.** The issue is on the store-fingerprint board's §4, with a closing note naming the
  commit by subject. Counted, the index has **96** rows, down from 97, with the date. This board
  has its §1 row, S1–S5, §3.1 and §5.9.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `f070a9d`, serially: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
    Datastore 351 (+8), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
  - `black --check` is clean on the eight Python files.
  - `ray.is_initialized()` is false after the new module.
- **Remnants, fixed in `f070a9d`** (*"Tidy two remnants of datastore-integrity prompt 06"*), each
  too small to be an issue:
  - The store-fingerprint board's header still said **"Five"** of the audit's issues had closed on
    its §4, while listing six. The agent's report said it had corrected this, and it had not.
  - `test_the_default_stores_pass_too` unpacked `full_rows()` into three unused names.
- **Inaccuracies in log 06, left as written.** The log is the record of what the agent did.
  - The "before S1" `-v` output (§"Which stores S4 checks") drops the status of three tests. Each
    has a docstring, so unittest prints the status on a second line, which the paste lost.
    `test_the_added_row_is_not_satisfied_by_coincidence` passed there; the other two failed, as
    the log's count of 4 says.
  - "Verification performed" says "the four boards"; there are two boards and the index.
- **Decided here: `TestSchemaIsUnchanged` keeps its name** (log observation 1). It still reads
  truly, as "the schema is unchanged from the current witness". Log 06's mutation record cites
  it, and so will every later witness change, so renaming it would make those records stop
  matching.

**Next: 07**, with its own orchestrator note carrying this review's findings. It is the first
prompt to add a column and change a lookup key. It adds its witness beside 06's, by the procedure
06 set, and each new foreign key needs S4-style rows that the wrong table cannot satisfy by
coincidence. It is also the first whose foreign key constrains a drop, so `_drop_order` is its
concern. Prompt 10 remains the first real run.

**Orchestrator review of prompt 07 (2026-09-29).** Run against `a171ced`, dispatched from
`794728d` with nothing `running` and no other worktree. **All eleven checks in
`orchestrator/prompt-07.md` §3 passed.** The review found three small remnants, fixed in
`d1829bb`, and one script the log's rule-11 list omits.
- **Dispatch.**
  - `HEAD` was `794728d`, not the note's `a2a8b65`. The two commits between were another
    session's, touching docs and one docstring and moving no code. That session's work, which
    §1.2 required committed, was committed.
  - So the index held **97** rows, not the note's 96, and the agent was told so.
  - The baselines were re-measured on `794728d` and matched §5.9. `substitute_main_py()`'s five
    literals each occurred once.
  - While the agent worked, another session wrote `docs/derivation/` in this checkout, committed
    as `ccb3078`. The agent staged by path only, and `a171ced` touches none of its files.
- **The agent stopped once, correctly.** Two existing tests outside §0.1 had to change:
  - `test_gauss_order_key.py` calls both WKB `build()`s with no parent, and `store()` with no
    connection. Measured on a minimal patch, it gave 16 errors.
  - `test_factory_select_columns.py`'s guard pins grow by the check's new row reads.

  The note's "ComputeTargets unchanged" was measured on a sketch that did not touch `build()`.
  **The user granted both on 2026-09-29, limited to following the key (grants A and B)**, and
  accepted the three `main.py` behaviour changes the agent flagged (log deviations 1–3).
- **Scope.**
  - 29 files, all on the brief's list. The forbidden paths of §3.1 have an empty diff.
  - `--diff-filter=M` over `*/tests/*` gives §0.1's seven files plus the two grants. Every hunk
    was read: each follows the key and asserts nothing new or less.
  - `test_store_inventory`'s missing-column test now recreates shard 1's table from the 06
    witness's DDL, since SQLite cannot `DROP COLUMN` a foreign-key column (deviation 4). Its
    assertions change only the column name. That is sound.
  - `Datastore.py` is `_drop_order` and its comment, `store_inventory.py` is `nullable` and its
    three-line use, and `stores.py` is the one constant.
- **K1, schema and lookups.**
  - The key:
    - `numeric_serial` is indexed with its `ForeignKey`, NOT NULL on Tk and nullable on Gk;
    - `z_init` is in no `build()` filter;
    - `PURE_WKB` compiles to `IS NULL`;
    - a lookup naming neither parent nor mode raises, for Tk and Gk alike, and so does a Tk
      construction with no parent;
    - `labels_only` refuses a second match and names the parents;
    - the four value reads take an optional parent in place of `z_init`.
  - **The consistency check** is in each factory's `store()`, before the insert. It reads the
    parent joined to its exit time and compares `z_exit - stop_deltaz_subh` and the stop values
    by `struct.pack` bit pattern, and it checks `k`, model and, for Gk, `z_source`. A pure-WKB
    row is checked against `G = 0`, `G' = 1` and the `z_source` row's `z`.
  - With the parent's two WKB factories checked out over `TestTheConsistencyCheck`, 11 tests gave
    `failures=10, errors=4`. All nine one-ulp sub-tests failed. The factories were restored.
- **`main.py`.**
  - Every hunk is a site of §0.6's table.
  - `Gk_WKB_numeric_parent` holds `z_source_limit = sqrt(k_exit.z_exit_subh_e3 *
    k_exit.z_exit_subh_e4)` and `z_source.z >= z_source_limit and Gk.available` character for
    character, and the long comment became its docstring.
  - The Gk numeric lookup's `_do_not_populate` is sound: the seeded branch reads only
    `stop_G`, `stop_Gprime`, `stop_deltaz_subh` and `store_id`, all row columns.
  - `test_main_plumbing.py` is unmodified and passes.
  - The five `substitute_main_py()` literals each occur once, now at `:3582`, `:3689`, `:3690`,
    `:3791` and `:3803`.
- **Readers.**
  - Both extraction scripts pass `labels_only=True`, and `run_quadsource_integrals.py` keys on
    the numeric row it looked up.
  - A grep of every `"GkWKBIntegration"`/`"TkWKBIntegration"` lookup outside the tests finds none
    that passes neither a parent nor the mode.
  - **The rule-11 list omits one script.**
    `docs/datastore-integrity-audit/lookup_keys/probe_05_value_lookups_and_read_batch.py` runs
    cleanly at `a171ced~1` and now raises `IntegrityError: NOT NULL constraint failed:
    TkWKBIntegration.numeric_serial`. It inserts a raw Tk WKB row without the column.
  - All fourteen audit probes were run at both commits:
    - four already fail at the parent, since the fixes they shape have landed, and 07 does not
      change them;
    - `replicated_divergence_probe.py` gains one correct problem line, a WKB row whose numeric
      parent is unresolved;
    - the rest differ only by the new column, moved line numbers and a larger count of column
      references checked.
  - The probe is audit evidence, and the note barred the agent from probes, so it is left as
    written. See the user's note below.
- **K2.** `tk-wkb` drops before `tk-numeric`, and `gk-wkb` before `gk-numeric`.
  `TestDropOrder` runs under `PRAGMA foreign_keys = ON`, on a store it first asserts clean, and
  derives the order from every declared foreign key between actions.
- **D5 and the fingerprint.**
  - `schema_at_base.json`, `schema_at_datastore-integrity-06.json` and
    `full_store_fingerprint.json` have no diff.
  - A witness re-captured from `HEAD` is byte-identical to the new one.
  - The orchestrator's own diff of the 06 and 07 witnesses is **12 hunks and 86 lines**,
    byte-identical to the log's paste, and every hunk is one of §0.6's six kinds for one of the
    two WKB classes.
  - An independently re-captured fingerprint of `build_full_store()` equals the new golden:
    format 2, `problems` empty, 21 classes. Only `TkWKBIntegration` and `GkWKBIntegration` moved,
    and their counts (3, 2) and tag-set counts ([2, 1], [1, 1]) are unchanged.
- **Mutations.**
  - (i)–(v) were extracted from `git show HEAD:<log>` and applied with plain `git apply` in a
    detached worktree of `HEAD`, each over Datastore, ComputeTargets and RunRegistry.
  - Each gave its logged counts: Datastore 12, 8, 14, 2 and `2 + 1 error`; ComputeTargets 2 under
    (iv).
  - Every failing test and sub-test is one the log names under that mutation, and the tree was
    clean after each revert.
  - Two differences were environmental:
    - ComputeTargets under (v) failed `test_wall_time_per_object` at 0.0624 s against 0.06 s, the
      known flake, under the review's parallel load. That module alone under (v) gave 17 OK.
    - RunRegistry reported `skipped=1` in the worktree, the A3-manifest test, which needs a
      gitignored file under `var/`.
- **Boards.**
  - The issue is on the GkTk-remedial board's §4, with a closing note naming the commit by
    subject.
  - Counted, the index has **98** rows: 97, less the one closed, plus the two opened on §3.3.
  - This board has its §1 row, K1–K2, §3.1, §3.3 and §5.10.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `d1829bb`, serially: AdaptiveLevin 32, ComputeTargets 566 (+14), CosmologyModels 39,
    Datastore 397 (+46), LiouvilleGreen 148 (1 skipped), RunRegistry 199, all OK.
  - `black --check .` is clean.
  - `ray.is_initialized()` is false after each of the three new modules, which run 41, 5 and 14
    tests.
- **Remnants, fixed in `d1829bb`** (*"Tidy three remnants of datastore-integrity prompt 07"*), each
  too small to be an issue. The agent recorded all three and fixed none, as rule 4 asks.
  - `test_schema_builder.py`'s docstring named the 06 witness as current.
  - The comment above `FINGERPRINT_FORMAT` said a bump *regenerates*
    `full_store_fingerprint.json`, which D5 forbids.
  - This board's header said "Next: 07".
- **Log 07 is accurate**, apart from the omitted probe. Its counts, its witness paste and its
  mutation record were checked against the diff and the replays.

**For the user.** `probe_05` is the one script 07 breaks that its log does not name. Four other
audit probes already stopped running when the fixes they measure landed, and none of those was
recorded as an issue. So the convention appears to be that a probe is evidence for the tree its
audit read. The probe would run again with `numeric_serial=1` in its `tk_wkb_parent` insert, but
what it prints about `z_init` would then describe a filter that no longer exists.
**Decided by the user (2026-09-29): `probe_05` is left as it is.** Audit probes are evidence for
the tree their audit read. A later prompt's log names any probe it stops, in its rule-11 list,
and does not edit it.

**Next: 08**, with its own orchestrator note carrying this review's findings
([`orchestrator/prompt-08.md`](orchestrator/prompt-08.md)). It adds
`quad_policy_serial` to `QuadSourceIntegral`'s key, a witness and a golden beside 07's, and very
likely a second format bump. `TestDropOrder` will hold its new foreign key to `_drop_order`.

**Orchestrator review of prompt 08 (2026-09-30).** Run against `b18bb0b`, dispatched from
`479736d` with nothing `running` and no other worktree. **All twelve checks in
`orchestrator/prompt-08.md` §3 passed.** The agent did not stop, and needed no grant beyond the
note's. The review found three small remnants, fixed in `a9b6ff6`.
- **Dispatch.**
  - `HEAD` was `479736d`, the commit adding the note, as the note expected.
  - The index held **98** rows. `substitute_main_py()`'s five literals each occurred once, at
    `:3582`, `:3689`, `:3690`, `:3791` and `:3803`.
  - The baselines matched §5.10. In the full ComputeTargets run `test_wall_time_per_object`
    failed at 0.059 s against 0.05 s, under a load average of about 15. Alone it passed two of
    three times: the known flake.
  - No other session's change appeared in this checkout during the dispatch.
- **Scope.**
  - 24 files, all on the brief's list. The forbidden paths of §3.1 have an empty diff.
  - `--diff-filter=M` over `*/tests/*` gives exactly §0.3's six files.
    `test_factory_select_columns.py` is untouched: no pin moved, since `read_table` is a
    comprehension (§0.1 allowed either shape).
  - Every existing-test hunk was read. Each adds only the column, the `quad_policy` argument or
    payload entry, the second fixture policy, the `IDENTITY` entry, or a D5 repointing with its
    docstring sentence. None asserts anything new or less.
  - The other hunks are as bounded: `QuadSourcePolicy`'s factory, `read_table` alone;
    `sharding.py`, one entry; `defaults.py`, one constant; `MetadataConcepts/QuadSourcePolicy.py`,
    docstring alone; `stores.py`, the constant and its comment.
- **K3.**
  - `quad_policy_serial` is NOT NULL, indexed, with `ForeignKey("QuadSourcePolicy.serial")`.
    `policy_serial` is unchanged.
  - `build()` and `read_batch()` both filter on it and hand the policy to every object. `store()`
    writes it. The constructor requires it, and refuses `None`.
  - `inventory_records` declares `Parent("quad_policy_serial", "QuadSourcePolicy")`, and its
    comment no longer says "not a QuadSourcePolicy".
  - The D4 refusal is one helper, called from both readers' `except SQLAlchemyError`, matching
    the driver's message. `TestAnOldStoreIsRefused` recreates the table from the 07 witness's DDL
    and asserts it lacks the column. Mutations (i) and (ii) each break its `build` or
    `read_batch` test, since then no query names the missing column.
- **`main.py`.**
  - Both QSI sites pass `QuadSource_policy_1pt5`, and the 1.5 creation is
    `object_get("QuadSourcePolicy", **DEFAULT_QUADSOURCE_POLICY_1PT5)`. `_5pt0` is untouched.
  - `test_main_plumbing.py` is unmodified and passes (27 tests, with the new module's 19).
  - The five literals each occur once, now at `:3585`, `:3692`, `:3693`, `:3791` and `:3803`.
- **The extraction script.**
  - `find_QuadSource_policy` reads through `read_table`, whose filter is `build()`'s own. It
    refuses on none, naming the parameters, and on several, naming the serials.
  - The tree's only `object_get("QuadSourcePolicy", …)` calls are `main.py`'s two.
  - The `GkSourcePolicy` lookups are unchanged, and
    `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` is open.
  - A grep for every `QuadSourceIntegral` lookup and construction outside the tests finds none
    without the policy. `main.py:3163` is the vectorised query whose payloads `:3131` builds.
- **The refusal is safe.** `TestTheExtractionScript` runs the script's own function, compiled out
  with `ast`, on a real `ShardedPool` over stand-in shards. In every case it asserts that the only
  calls are `read_table` on `QuadSourcePolicy`, that no shard's rows change, and that
  `replication_in_flight` is empty. A control test shows that `object_get` is a different call.
- **D5 and the fingerprint.**
  - Both new files are new. The two older witnesses, `schema_at_base.json` and both older goldens
    have the same blob at both commits.
  - A witness re-captured from `HEAD` is byte-identical to the new one.
  - The orchestrator's own diff of the 07 and 08 witnesses is **6 hunks, 43 content lines**. It
    has one hunk of each of §0.6's six kinds, all in `QuadSourceIntegral`, and a JSON
    comparison finds no other class or top-level key differing.
  - A re-captured fingerprint of `build_full_store()` equals the new golden: format 3,
    `problems` empty, 21 classes. Only `QuadSourceIntegral` and `QuadSourcePolicy` differ from
    the 07 golden.
- **The foreign-key test** puts `quad_policy_serial` 3 on a row, and serial 3 is no
  `GkSourcePolicy` serial.
  - With `b18bb0b~1`'s factory checked out over it, 1 of its 4 tests fails.
  - With the key pointed at `GkSourcePolicy.serial`, 2 fail, among them
    `test_every_shard_passes_foreign_key_check`. That is the coincidence 06 found.
  - Both were restored.
- **Mutations.**
  - (i)–(iv) were extracted from `git show HEAD:<log>` and applied with plain `git apply` in a
    detached worktree of `HEAD`. Each was run over Datastore, ComputeTargets and RunRegistry.
  - Each gave exactly its logged failures and sub-tests: Datastore `2 + 1 error`, 4, 0 and 5;
    ComputeTargets 2 under (iii); RunRegistry 1 under (iv).
  - The tree was clean after each revert. RunRegistry's `skipped=1` in the worktree is the
    A3-manifest test, as in 07's review.
- **Readers and probes.**
  - All fourteen audit probes were run at `479736d` and at `b18bb0b`, from `git archive` exports.
  - Only `probe_01_tagged_read_batch.py` newly fails (`NOT NULL constraint failed:
    QuadSourceIntegral.quad_policy_serial`), and log 08's rule-11 list names it.
  - `fk_sweep_probe`, `probe_02` and `probe_03` change only as the log says. The five that
    already failed are unchanged, apart from paths.
- **Boards.**
  - The issue is on the store-fingerprint board's §4, with a closing note naming the commit by
    subject and decisions 6.1.2 and D6.
  - Counted, the index has **98** rows: one deleted, one opened. The date is 2026-09-29, the
    commit's own.
  - This board has its header, §1 row, K3, §3.1, the §3.3 entry and §5.11.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `b18bb0b`, serially: AdaptiveLevin 32, ComputeTargets 585 (+19), CosmologyModels
    39, Datastore 422 (+25), LiouvilleGreen 148 (1 skipped), RunRegistry 199. All OK.
  - `black --check` is clean on the 18 Python files.
  - Each of the three new modules asserts `not ray.is_initialized()` in its `tearDownModule`, and
    each passes.
- **Remnants, fixed in `a9b6ff6`** (*"Tidy three remnants of datastore-integrity prompt 08"*). Each
  is too small to be an issue.
  - `QuadSourcePolicy`'s docstring still gave "so that the schema … do[es] not churn" as a reason
    to keep the class. The note had said this sentence becomes false, and the agent kept it.
  - The store-fingerprint board's closing paragraph said "Seven issues remain open in §3". Log 08
    recorded this and did not fix it (observation 5).
  - `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`'s site list lacked the new
    `read_table`. Log 08 recorded this (observation 4). It is now added on this board and in the
    index hook.
- **Log 08 is accurate** against `git show`. Two of its line citations in the site table are
  one line off: the helper starts at `:80`, and the inventory parent is at `:880`. These are
  left as written.

**Next: 09**, with its own orchestrator note carrying this review's findings
([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md)), **split by the user on 2026-09-30 into
09a → 09b → 09c** (the note's §0.1–§0.4 record that decision and three others). It keys
`QuadSource` on `Tq_serial` and `Tr_serial`, `QuadSourceIntegral` on its source, policy data and
Tk rows, and `GkSource` on its Gk rows (K4–K7). It resolves cross-shard parents in the inventory.
It is the natural owner of `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`, and its record
change is a third format bump, to 4. Prompt 10 remains the first real run.

**Orchestrator review of prompt 09a (2026-09-30).** Run against `dbf2cf2`, dispatched from
`2252b8c` (the note's own commit, which touches no code) with nothing `running` and no other
worktree. **All twelve checks in `orchestrator/prompt-09.md` §3 passed.** The agent did not stop,
and needed no grant beyond the note's §0.4 and §0.5. The review found one small remnant, fixed in
`f161b2f`, and one question for the user, on the issue 09a opened (below).
- **Dispatch.**
  - `HEAD` was `2252b8c`. The index held **98** rows. `substitute_main_py()`'s five literals each
    occurred once, at `:3585`, `:3692`, `:3693`, `:3791` and `:3803`.
  - The baselines matched §5.11. In the full ComputeTargets run `test_wall_time_per_object`
    failed. Alone it passed one of three times (0.061–0.065 s against 0.06 s), under a load
    average near 27: the known flake.
  - The fourteen audit probes at `2252b8c` were identical, apart from paths and timestamps, to
    the 08 review's outputs at `b18bb0b`.
  - No other session's change appeared in this checkout during the dispatch.
- **Scope.**
  - 20 files, all on the brief's list. The diff over every older witness and golden,
    `schema.py`, `ShardedPool.py`, `docs/handover/`, `docs/datastore-integrity-audit/` and
    `orchestrator/` is empty.
  - `--diff-filter=M` over `*/tests/*` gives five files: §0.5's four for 09a, and
    `test_main_plumbing.py` under §0.4. Every hunk was read.
    - `IDENTITY["QuadSource"]` gains `Tq` and `Tr` (both varied to Tk numeric 3), and
      `NON_IDENTITY` loses `Tq_serial`.
    - `TestTags` expects `QuadSource` too, with a sentence saying why.
    - `test_quadsource.py`'s `payload=None` construction takes the two serials in place of the
      manual assignment after it. The `:435` construction is untouched, since its payload
      already carries both.
    - The two D5 repointings each come with their docstring or comment sentence.
    - `EXPECTED_SITES["TkNumericIntegration"]` goes 5 → 6, with a comment.

    None of these edits asserts anything new or less. The read-back guard's pins did not move.
- **K4.**
  - `build()` filters on both serials, named by serial. A lookup naming neither the parents nor
    `labels_only` raises, and so does one naming both. `Tr_serial` is indexed, with no foreign
    key, and its comment says why.
  - `store()` writes both, and refuses an object without them.
  - `labels_only` reads the candidates' pairs from the lookup's own query and refuses more than
    one, naming each pair.
  - The target takes the serials at construction, and no longer overwrites them in `compute()`.
    `compute()` asserts both before anything is computed, naming both sides.
  - With `dbf2cf2~1`'s target over the new tests, 17 of `test_quadsource_parent_main`'s 25 fail.
    With `dbf2cf2~1`'s factory, 22 of `test_quadsource_parent_key`'s 37 fail. Both were restored.
- **K7 (cross-shard).**
  - `Parent(cross_shard=True)` resolves against `ShardContext.all_digests`. `read_inventory`
    fills it only after a class is built on every shard, so a parent class not yet built is
    still the dependency-order `RuntimeError`.
  - A serial that no shard holds is `unresolved-parent`. So is one that two shards hold with
    different digests, rather than one of them being chosen.
  - Same-shard resolution is unchanged, and the store is still read once.
- **`main.py`.**
  - All six hunks are §0.6's sites. The `QuadSource` stage resolves every distinct `k` of its
    batch serial-only (`_do_not_populate`) before the existence query, with the same tags and
    tolerance pair as its later full lookup. The construction passes the full rows' serials.
  - The `QuadSourceIntegral` stage's `QuadSource` lookup now follows its Tk numeric lookup. The
    Tk lookup's `missing_Tk` is drawn from the same missing items as `missing_source_b`, so every
    serial the keyed lookup indexes exists.
  - `qsi_work_batches` is untouched. The five literals each occur once, now at `:3670`, `:3777`,
    `:3778`, `:3876` and `:3888`.
  - All 29 of the log's site-table citations match `git show`.
- **Readers.**
  - The three named readers pass `labels_only=True`.
  - A grep of every `"QuadSource"` lookup outside the tests finds none that passes neither the
    parents nor the mode.
  - `QuadSourceValue._build_impl_model` is a labels-only `QuadSource` lookup inside another
    factory. The agent opened it as an issue rather than fixing it, which is correct under rule 4.
- **D4.**
  - The witness diff leaves `QuadSource`'s table DDL unchanged and adds only the index. So a
    pre-09a table differs only in lacking the index, and a keyed lookup reads it.
  - `TestAnOldStoreIsRead` builds such a table from the 08 witness's own DDL, and asserts the
    index is absent.
- **D5 and the fingerprint.**
  - Both new files are new. Every older witness and golden has the same blob at both commits.
  - A witness re-captured from `HEAD` is byte-identical to the new one.
  - The orchestrator's own diff of the 08 and 09a witnesses is **3 hunks, 14 lines**, all in
    `Tr_serial`'s index, exactly as §0.6 measured on the sketch.
  - A re-captured fingerprint of `build_full_store()` equals the new golden: format 4, `problems`
    empty, 21 classes. Against the 08 golden only `QuadSource` moved (2 → 2 records).
- **Mutations.**
  - (i)–(iv) were extracted from `git show HEAD:<log>` and applied with plain `git apply` in a
    `git archive` export of `HEAD`, not a worktree, so that the checkout stayed alone. Each was
    run over Datastore, ComputeTargets and RunRegistry.
  - Each gave exactly its logged failing entries, sub-tests included. Excluding the flake, those
    are: Datastore 2, 39 (15 of them sub-tests), 3 and 0; ComputeTargets 0, 0, 0 and 6;
    RunRegistry 0, 4, 0 and 0.
  - Four tests that need a git repository fail identically in every run of the export, and are
    not the mutations'. They are `test_inventory_retired`'s two and one each in
    `test_run_registry` and `test_store_sidecar`.
  - Mutation (ii) is check 8, the inventory form broken. The export matched `HEAD` after the
    reverts.
- **Probes.**
  - All fourteen were run at `2252b8c` and `dbf2cf2`, from exports. None newly fails, and the
    same six fail at both.
  - Three change output: `probe_02` (line numbers), `probe_03` (1254 → 1256) and
    `replicated_divergence_probe` (two `unresolved-parent: QuadSource` lines, from its own
    fixture). Log 09a's rule-11 list names all three.
- **Boards.**
  - The §1 row, K4, K7 (partial), the §3.3 entry and §5.12 are present.
  - Counted, the index has **99** rows, with one opened. The date is 2026-09-30.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `dbf2cf2`, serially, in the checkout: AdaptiveLevin 32, ComputeTargets **610**
    (+25), CosmologyModels 39, Datastore **459** (+37), LiouvilleGreen 148 (1 skipped),
    RunRegistry 199. All OK; the flake did not fire.
  - `black --check` is clean on the 15 Python files.
  - Both new modules assert `not ray.is_initialized()` in `tearDownModule`, and pass.
- **Remnant, fixed in `f161b2f`** (*"Correct who decided the QuadSourceIntegral stage's keyed
  lookup"*). The `QuadSourceIntegral` stage's new comment attributed §0.2 to "the orchestrator's
  decision". It was the user's.
- **Not changed.** The `Tk_numeric_unavailable` message still says only that the mode's
  `TkWKBIntegration` cannot be looked up (log 09a, observation 2). The statement is true, and the
  comment above it now says the `QuadSource` lookup depends on the same rows.
- **For the user.** `[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]` has no
  owner. Its "next step" names "the prompt that next edits `QuadSourceValue`'s reads", and no
  planned prompt does. It is D1's class: a labels-only lookup of a keyed row, which refuses but
  names no parent. It is put to the user with 09b's grant.
- **Log 09a is accurate** against `git show`. Deviation 7's heading, "Two tests of
  `test_quadsource.py`, not one", reads the wrong way round: the prompt named two constructions,
  and one needed the change. Its body is right, and it is left as written.

**Next: 09b**, dispatched from this review's commit once the user has answered §1 step 7's sketch
and grant, and the question above.

**The user answered on 2026-09-30** (`e080895`, the note's §0.5 "Granted for 09b" and §0.7):
- the measured grant, with `test_gauss_order_key.py` on its condition;
- `tools/inventory_report.py` for the set's rendering;
- `GkSourceValue`'s model read to 09b, and `QuadSourceValue`'s, with the 09a issue, to 09c;
- an empty parent set refused.

**Orchestrator review of prompt 09b (2026-09-30).** Run against `3d7b606`, dispatched from
`e080895` with nothing `running` in this repository's registry and no other worktree. **All twelve
checks in `orchestrator/prompt-09.md` §3 passed.** The agent did not stop. It stopped once on an API
usage limit, and was resumed with its staged work intact. The review found one small remnant,
fixed in `8014f1d`.
- **Dispatch.**
  - `HEAD` was `e080895`, which touches no code. The index held **99** rows. The five literals
    each occurred once, at `:3670`, `:3777`, `:3778`, `:3876` and `:3888`.
  - The baselines matched §5.12, all OK, the flake included.
- **Scope.**
  - 28 files, all on the brief's list. The diff over every older witness and golden, `schema.py`,
    `ShardedPool.py`, `config/`, `test_inventory_report.py`, `docs/handover/`,
    `docs/datastore-integrity-audit/` and `orchestrator/` is empty.
  - `--diff-filter=M` over `*/tests/*` gives six files: §0.5's five and
    `test_factory_select_columns.py` under §0.4. Every hunk was read.
    - The fixture gains `parents_digest` and six `GkSource_parents` rows naming its real Gk rows.
    - `IDENTITY["GkSource"]` gains `parents`, varied through row 604.
    - `test_gauss_order_key.py` gains a stand-in set and a `GkSource_parents` inserter, and no
      assertion.
    - The two D5 repointings.
    - `EXPECTED_COVERAGE` gains 09b's own three row-read sites, each named. It goes 47 → 50, and no
      other pin moved.
  - Two edits are outside the brief's function lists, and both are sound:
    - `validate_on_startup`'s prune deletes a pruned source's members, which is structurally
      required;
    - `ClientPool` gets a batch size of 7500.
  - `config/sharding.py` is untouched, and this is right. Tables written by their parent's actor,
    such as `GkSource_tags`, are not listed there either. The broker leases serials for any table
    on demand. At open, every registered table with a serial reports its maximum. The check at open
    treats an unlisted table as replicated only if every foreign key names a replicated table, and
    `GkSource_parents`' do not.
- **K6.**
  - `parents_digest` is in `build()`'s filter. `store()` forms it from the rows it writes, which
    are the set `assemble_GkSource_values` records it was handed. `compute()` asserts that this
    equals the set the object is keyed on, so a stored digest always equals the looked-up one.
  - `store()` refuses an empty set. The value reads hand back `_parent_serial`, and have a
    serial-only `parents_only` form that keeps 05c's refusal of a second parent per `z_source`.
  - With `3d7b606~1`'s target or factory over the new modules, each module fails at import, since
    the names it tests do not exist there.
- **`main.py`.**
  - All 17 hunks are as the log lists them, and 35 of its site-table citations match `git show`.
  - `resolve_GkSource_parents` reads with `build_missing_GkSource`'s own payload (tags, the
    numeric pair) over the same `z_source_sample.truncate(z_response, keep="higher")`, so the two
    sets cannot differ.
  - All six lookups are keyed on its result. The GkSource stage resolves before its existence
    query.
  - `EXPECTED_SITES` did not move, since the helper reads through `object_read_batch`. The five
    literals each occur once, now at `:3792`, `:3899`, `:3900`, `:3998` and `:4010`.
- **Readers.** The four named pass `labels_only=True`. `GkSourceValue._build_impl_model` refuses two
  matches, naming their digests and where their sets differ.
- **K7 (set-valued).**
  - `ParentSet` reads each member table once per shard. It resolves members to reference digests,
    never serials or `parents_digest`.
  - It names `empty-parent-set`, `orphan-member`, an unresolvable member (`unresolved-parent`) and
    an absent member table.
- **D4.** `TestAnOldStoreIsRefused` (3 tests) and `TestTheWKBValueReadsRefuseAPre07Store` (4 tests
  and a control) pass: all four WKB value reads refuse a pre-07 table, keyed or not. The 07 issue's
  defect is gone.
- **D5 and the fingerprint.**
  - Both new files are new, and every older witness and golden is unchanged.
  - A witness re-captured from `HEAD` is byte-identical to the new one. The orchestrator's own diff
    against 09a is **9 hunks, 246 changed lines, 256 lines**, at the sketch's positions, all in
    `GkSource` and `GkSource_parents`.
  - A re-captured fingerprint equals the new golden: format 5, `problems` empty. Against the 09a
    golden only `GkSource` and `GkSourcePolicyData` moved (3 → 3 records each), and 19 did not.
- **Mutations.**
  - (i)–(v) were extracted from `git show HEAD:<log>` and applied in a `git archive` export of
    `HEAD`. This time the export was made into its own git repository, so no git-dependent test
    failed there. Each mutation was run over Datastore, ComputeTargets and RunRegistry.
  - Each gave exactly its logged failing entries. Datastore gave 1, 4, 1, 10 (2 of them sub-tests)
    and 2 (both sub-tests); RunRegistry gave 1, under (iv).
  - The flake fired once, under (ii). The export's tracked diff was empty after the reverts.
- **Probes.** All fourteen were run at `7728518`'s code and at `3d7b606`. None newly fails, and the
  exit codes are identical. `fk_sweep_probe`, `probe_02`, `probe_03` (1256 → 1268) and
  `replicated_divergence_probe` change, as log 09b's rule-11 list says, and so does the fault
  probe's timestamp line.
- **Boards.**
  - The §1 row, K6, K7 (still partial), the 07 issue on §4 and the two §3.3 entries are present.
  - Counted, the index has **100** rows: one deleted, two opened.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `3d7b606`, serially, in the checkout: AdaptiveLevin 32, CosmologyModels 39,
    Datastore **505** (+46), LiouvilleGreen 148 (1 skipped) and RunRegistry 199 were OK.
    ComputeTargets ran **635** (+25) with 2 failures, under a load average of 116–150 from a job in
    another of the user's workspaces, outside this repository.
    - `test_background_tau.test_build_cost` (0.66 s against 0.5 s) also fails alone on the
      pre-09b code at that load.
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways` saw a time-triggered status
      print, and passed alone at `HEAD` in all three runs.
    - 09b touches neither module nor what they test. The agent's run on the staged tree, at lower
      load, was 635 OK.
  - `black --check` is clean on the 23 Python files.
  - Both new modules assert `not ray.is_initialized()` in their `tearDownModule`.
- **Remnant, fixed in `8014f1d`** (*"Name GkSource in GkSource.build()'s multiple-results
  message"*). `GkSource.build()` printed a line naming `GkWKBIntegration` before re-raising.
  09b's reader mode reaches that branch, and log 09b recorded it as residue.
- **For the user, later.** `[09b-inventory-report-renders-every-member-of-a-parent-set]` asks how
  the report should summarise a set. It changes what the user decided on the sketch, so it is
  theirs to answer, and it does not block 09c.
- **Log 09b is accurate** against `git show`.

**Next: 09c**, from this review's commit, after §1 step 7's sketch and the user's grant.

**The user answered on 2026-09-30** (`9a18063`, the note's §0.5 "Granted for 09c" and §0.8):
- the measured grant, with four edits outside the candidate list;
- `data` kept a same-shard parent;
- the one non-conforming fixture row kept, with a comment;
- `read_batch` to select `quad_policy_serial` first;
- the two constructor-bearing modules granted on their condition.

**Orchestrator review of prompt 09c (2026-10-01).** Run against `014be95`, dispatched from
`9a18063` with nothing `running` in this repository's registry and no other worktree. **All twelve
checks in `orchestrator/prompt-09.md` §3 passed.** The agent did not stop.
- It stopped twice on API usage limits. It was resumed once with its staged work intact, and it
  committed before the second stop. That stop came during its second mutation replay, so it sent
  no final report.
- The review was therefore made from the commit, the log and the agent's replay outputs alone.
- It found three small remnants, fixed in `b5fd484`.

The checks:
- **Dispatch.**
  - `HEAD` was `9a18063`, which touches no code. The index held **100** rows. The five literals
    each occurred once, at `:3792`, `:3899`, `:3900`, `:3998` and `:4010`.
  - The baselines matched §5.13. The one failure was the flake, at a load average near 35 from a
    job outside this repository.
- **Scope.**
  - 24 files, all on the brief's list. The diff over every older witness and golden, `schema.py`,
    `ShardedPool.py`, `Datastore.py`, `config/`, `docs/handover/`,
    `docs/datastore-integrity-audit/` and `orchestrator/` is empty.
  - `--diff-filter=M` over `*/tests/*` gives ten files, all in §0.5's 09c grant or under §0.4.
    Every hunk was read:
    - the fixture gains the four columns, `_UNDECLARED_REFERENCES` gains them, and the one
      non-conforming row has its comment;
    - `IDENTITY` gains the six parents, each varied to a real row, and `NON_IDENTITY` loses two;
    - the two `TestTags` sets gain the class, each with its sentence;
    - the duplicate test copies the parent chain;
    - `check()`'s `cascade` asserts more, not less. The cascaded class's only problems are
      `unresolved-parent`; its records are a strict subset, short by exactly the named count; and
      with no cascade, `changed` must be empty;
    - the raw inserts, 08's payload and stored row, and the two conditional constructor modules
      pass only the serials;
    - the two D5 repointings.
  - `EXPECTED_SITES` and the pins:
    - `TkNumericIntegration`, `TkWKBIntegration`, `QuadSource`, `GkSource` and
      `GkSourcePolicyData` each rise by one, from the one new helper,
      `resolve_QuadSourceIntegral_parents`, and no count fell;
    - `EXPECTED_COVERAGE` goes 50 → 51, by 09c's own `QuadSourceValue` candidate site.
- **K5.**
  - The four columns are `Tq_numeric_serial`, `Tq_WKB_serial`, `Tr_numeric_serial` and
    `Tr_WKB_serial`: NOT NULL, indexed, no foreign key.
  - All six parents are in `build()`'s filter, required and integer. `store()` writes them and
    refuses an object without them. `read_batch` filters on any it is given and hands all six
    back.
  - `inventory_records` declares `data` same-shard, and the other five cross-shard.
  - The target requires the six at construction. `compute()` asserts that the six rows it is
    handed are they, that each numeric row is its source's `Tq_serial`/`Tr_serial`, and that each
    WKB row's `numeric_serial` is its numeric row, all before computing.
  - With `014be95~1`'s target over `test_quadsource_integral_parent_main`, 23 of 30 error. With its
    factory over `test_quadsource_integral_parent_key`, the module's 27 tests give 12 failures and
    40 errors, counted with their sub-tests.
- **`main.py`.**
  - All 12 hunks are as the log lists them. 38 of its site-table citations match `git show`.
  - The helper resolves every work item's six parents serial-only, each found by its own parents
    as the stage's full lookups find them, before the existence query.
  - The query runs over the resolvable items only. Its results zip against that list, and every
    item not found, or not resolvable, is missing.
  - The construction passes the six serials the query used.
  - `qsi_work_items` and `qsi_work_batches` are untouched. The five literals each occur once, now at
    `:4148`, `:4255`, `:4256`, `:4354` and `:4366`.
- **The reader.**
  - `extract_QuadSourceIntegral_data.py` refuses more than one row at a `z_response`, naming the
    parents that differ, inside `plot_QuadSourceIntegral`.
  - That task's ref is tracked by `RayWorkPool`, which `ray.get`s each completed ref, so the
    refusal stops the script.
  - `QuadSourceValue._build_impl_model` refuses two matching `QuadSource`s, naming their
    `Tq_serial` and `Tr_serial`.
- **D4.** A pre-09c table is refused by `build()` and `read_batch()`. A pre-08 one is still refused
  with 08's message, since `quad_policy_serial` is selected first.
- **D5 and the fingerprint.**
  - Both new files are new, and every older witness and golden is unchanged.
  - A witness re-captured from `HEAD` is byte-identical to the new one. The orchestrator's own diff
    against 09b is **5 hunks, 102 changed lines, 108 lines**, at the sketch's positions, all in
    `QuadSourceIntegral`.
  - A re-captured fingerprint equals the new golden: format 6, `problems` empty. Against the 09b
    golden only `QuadSourceIntegral` moved, and 20 did not.
- **Mutations.**
  - All six were extracted from `git show HEAD:<log>`: (i), (ii), (iii), (iv) for `Tr`, (iv) for
    `Tq`, and (v). They were applied in a `git archive` export of `HEAD` made into its own git
    repository, and each was run over Datastore, ComputeTargets and RunRegistry.
  - Each gave exactly its logged failing entries, sub-test parameters included: Datastore 2, 0, 0,
    62, 3 and 1; ComputeTargets 0, 2, 3, 0, 0 and 0; RunRegistry 0, 0, 0, 9, 0 and 0.
  - **Mutation (iv) is caught for a `Tq` parent**, which §0.8 found the fixture alone would not
    catch.
  - The flake fired in four runs. The export's tracked diff was empty after the reverts.
  - The agent's own second replay, found in its scratch directory, agrees.
- **Probes.**
  - All fourteen were run at `3d7b606`'s code and at `014be95`. None newly fails, and the exit
    codes are identical.
  - Four change, as log 09c's rule-11 list says, besides the fault probe's timestamp line:
    - `fk_sweep_probe` (the four columns);
    - `probe_02` (the new `candidate` row, and moved lines);
    - `probe_03` (1268 → 1287);
    - `replicated_divergence_probe` (one cascade line).
- **Boards.**
  - The §1 rows for 09c and 09, K5, K7 (complete), the 09a issue on §4, §5.14 and README §4's four
    names are present.
  - Counted, the index has **99** rows: one deleted, none opened.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `014be95`, serially, in the checkout, at a load average near 50 from the same outside
    job:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199;
    - Datastore **532** (+27);
    - ComputeTargets **665** (+30), with only the flake failing.
  - `black --check` is clean on the 18 Python files.
  - Both new modules assert `not ray.is_initialized()` in their `tearDownModule`.
- **The measured cost.** About 1.1 s per batch of 750 for the six parents (208 actor calls). Add
  about 0.9 s for the Gk sets, now resolved for every `(k, z_response)` of a batch. That is of order
  23 minutes of serial-only lookups over a production pass, recorded under rule 11 and not argued
  against.
- **Remnants, fixed in `b5fd484`** (*"Tidy three remnants of datastore-integrity prompt 09c"*):
  - `[09b-inventory-report-renders-every-member-of-a-parent-set]`'s board entry and index hook
    understated it. Every integral's report line now carries its source's whole parent set (log
    09c, observation 1), and the entry gains a **Widened** line saying so.
  - A comment in 09b's `test_gksource_parent_set_main` counted six `GkSource` sites where there are
    now seven (observation 4).
  - Two pre-existing "for for" typos in `build_QuadSourceIntegral_payload`'s warnings (observation
    5).
- **Not changed.** `QuadSourceIntegral.source_serial` and `.data_serial` still raise on an
  unpopulated object, although both are now known from construction (observation 2). The
  `parents` property does not raise. This is behaviour, not prose, so it is left as it is.
- **Log 09c is accurate** against `git show`.

**09 is complete.** K4–K7 are done, and D1's class (audit §2.4) is closed on this board: every
derived row's key reaches the rows it was computed from, transitively, and the inventory resolves
them across shards and as sets. **Next: 10**, the first real run, with its own orchestrator note
carrying the 09a, 09b and 09c reviews. It is the first time any of these keys, the cross-shard and
set-valued inventory parents, and the stages' re-resolution meet a pipeline.

**Orchestrator review of prompt 09d (2026-10-01).** Run against `9540314`, dispatched from
`0dda6c8` with nothing `running` in this repository's registry and no other worktree. **All ten
checks in `orchestrator/prompt-09d.md` §3 passed.** The agent did not stop, and sent its report.
The review found three small remnants, fixed in `96fb98d`.

The checks:
- **Dispatch.**
  - `HEAD` was `0dda6c8`, which touches no code past `a761914`. The index held **99** rows.
  - The baselines matched §5.14. The one failure was the flake, at a load average of 80–160 from a
    job outside this repository; it failed alone too (0.098 s against 0.06 s).
  - The fixture report at `0dda6c8`: 31,466 characters verbose, 31,198 not; record lines of
    `QuadSourceIntegral` 4,316–4,462, `QuadSource` 1,109–1,200, `GkSourcePolicyData` 746–762,
    `TkWKBIntegration` 565, `GkSource` 533–550.
- **Scope.**
  - Five files, all on the brief's list. `--diff-filter=M` over `*/tests/*` is empty. The diff
    over `Datastore/store_inventory.py`, the factories, every witness and golden, `main.py` and
    `orchestrator/` is empty; `main.py` is byte-identical to `0dda6c8`.
- **(P1) Distinct stays distinct.**
  - The new tests cover every pair the prompt's §3 test 1 names, each line compared with its own
    label taken off.
  - The orchestrator's own pairs, built on the fixture's inventory rather than the agent's
    stand-in: a second `GkSource` identical to a fixture one but for the Gk numeric row at one
    `z_source` (same member count and `z_source` values) prints `parents=S2` beside `parents=S1`,
    with two distinct set lines that differ only in `GkNumericIntegration#2` / `#3`. A
    `GkSourcePolicyData` over it, a `QuadSourceIntegral` differing only in that `data`, and one
    differing only in `Tq_numeric`, each print as distinct lines once the own label is removed.
- **(P2) Every label is defined.** With the orchestrator's own regex, on the fixture verbose and
  not, and with `MAX_SHOWN` patched to 1 and 2: every `<Class>#<n>` and `S<n>` printed has exactly
  one defining line; no label appears on any other kind of line; every `> ` row is one that some
  printed line names; `sections()` sees at most `MAX_SHOWN` records per tag set, and `more` is
  right in every section.
- **(P3) Lines are bounded.**
  - Re-rendered at `9540314`: 14,776 characters verbose; `QuadSourceIntegral` 564–565,
    `QuadSource` 321–336, `GkSourcePolicyData` 206–210, `GkSource` 169–170, `TkWKBIntegration`
    173. These match the log. The `QuadSourceIntegral` line's ~165 characters over its pre-09
    ~400 are the six key fields 09c added, now ~25 characters each.
  - On the agent's stand-in at 250 and at 1,500 members, no record line of any class changes
    length, verbose or not. A set's line grows by about 58 characters a member. The 1,500-member
    report renders in under a second.
- **Nothing else changed.** The fixture reports at `0dda6c8` and `9540314`, section by section:
  the twelve replicated classes and `OneLoopIntegral` are byte-identical. In the eight that
  change, header, common line, tag sets and record order are identical; the only fields that
  differ are the sharded parents, each now a label (`TkWKBIntegration.numeric`, `QuadSource.Tq`
  and `.Tr`, `GkSource.parents`, `GkSourcePolicyData.source`, and `QuadSourceIntegral`'s
  `source`, `data` and four Tk rows); every other field, every replicated parent included, is
  byte-identical; and the tail beyond the fields is identical once the own label is removed.
  `TkNumericIntegration`, `GkNumericIntegration` and `GkWKBIntegration` change only by their own
  label.
- **The property's own tests.** `test_inventory_report` and `test_inventory_consumers` pass
  unmodified; `test_no_digest_appears` searches the whole report, so it covers the new lines. `HEX64`
  finds nothing in any report rendered here.
- **Labels are stable.** Under `relabel_serials` (offset 1,000), verbose, not, and with `MAX_SHOWN`
  1, every label, sets included, names the same physical row, and the whole reports are
  byte-identical.
- **Mutations.** All five were extracted from `git show HEAD:<log>` and applied with plain `git
  apply` in a `git archive` export of `9540314` made into its own git repository. Over the three
  Datastore modules that import the report, each gave exactly its logged failing entries,
  sub-test parameters included: 9, 7, 6, 8 and 8. `test_qcd_cosmology_inventory_record` passed
  under each, as logged. The export's tracked diff was empty after every revert. The agent ran
  these four modules rather than whole suites (its deviation 9); `git grep` confirms they and
  `main.py` are the only importers of `tools/inventory_report.py`.
- **Probes.** None is run: `git grep -l inventory_report -- docs/` finds three Markdown documents
  and no script.
- **Boards.** The §1 row, the 09b issue on §4 with its closing note, the new §3.3 entry and §5.15
  are present. Counted, the index has **99** rows: one deleted, one opened.
- **`var/`, suites, `black`, Ray.**
  - The snapshot (46 files, 10 directories) is identical.
  - Suites on `9540314`, serially, in the checkout, at a load average of 80–160 from the same
    outside job:
    - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199;
    - Datastore **551** (+19, the new module's);
    - ComputeTargets 665, with only the flake failing (0.080 s).
  - `black --check` is clean on both Python files.
  - `ray.is_initialized()` is false after the new module.
- **Remnants, fixed in `96fb98d`** (*"Tidy three remnants of datastore-integrity prompt 09d"*):
  - The header of a class's not-shown listing said the rows were "referred to by the records
    above". A row is defined in its own class's section, which usually comes *before* the records
    that name it (a `TkNumericIntegration` row named by a `QuadSourceIntegral`), so the header was
    false. It now reads `N row(s) not shown above, and referred to elsewhere in this report:`.
    Log 09d quotes the old text as it shipped, and is left as the record of that commit.
  - This board's header still counted "10 of 11 prompts landed". With 09d it is 11 of 12.
  - README §2's 09d row still read "**written** 2026-10-01; after 09c", and its header paragraph
    did not record 09d's landing (the agent's report flagged the row; it was outside its files).
- **Left for the user.** `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]`
  is real: since every labelled class's line ends in its own unique label, the existing
  distinct-lines test cannot fail for those seven classes. 09d's own tests pin the property with
  the label removed, and mutation (i) shows them failing, so nothing is unchecked today. Fixing
  the old test modifies an existing test, which README §5 rule 7 reserves to a prompt that names
  it, so it is not treated as residue.
  - **Decided (2026-10-01):** the user folded the fix into prompt 10 (§3.3).
- **Log 09d is accurate** against `git show`.

**09d is complete.** The report is bounded: no record line grows with a parent set or a chain of
derived parents, and every distinct record is still a distinct line. **Next: 10**, whose store's
`main.py --inventory` is the first print of this report on rows written by the keyed stages, and
may be used as its evidence.

**Orchestrator review of prompt 10 (2026-10-02).** Run against `b1774d8`, dispatched from
`d144d04` with nothing `running` in this repository's registry and no other worktree. **All eleven
checks in `orchestrator/prompt-10.md` §3 passed**, on the store `resumed-d` left. The agent stopped
four times, each time correctly, and the user decided each stop; "Decided during 10" above lists
them. Two small remnants are fixed in `53fa6c5`.

The checks:
- **Dispatch.**
  - `HEAD` was `d144d04`, which touches no code past `813cdcb`. The index held **99** rows.
  - The baselines matched §5.15, at a load average of 16–20.
  - `df -h var/`: 44 GB free, 96 % used, and 36 GB by 23:08 with nothing of this campaign running.
  - Both store names were free: `store show` gave kind `absent`, with no runs naming either.
- **Scope.**
  - 30 files: the one test, the record and its directory, the log, this board and the index.
  - `--diff-filter=M` over `*/tests/*` is `test_inventory_report.py` alone. Its three hunks are
    one import (`OWN_LABEL`, from 09d's test module) and the named test's body and comment.
  - The diff over `ComputeTargets/`, `main.py`, `tools/`, `RunRegistry/`, `config/`, both
    drivers, `docs/handover/`, every `extract_*.py`, `docs/datastore-integrity-audit/` and
    `orchestrator/` is empty. `main.py` is byte-identical to `d144d04`.
- **The runs.** Every manifest's `git_head` is `d144d04`, with `git_dirty` true. The log and the
  record show `git status --short` as `?? docs/datastore-integrity-verification/` at every launch.

  | Run | State | Exit | Results |
  |---|---|---|---|
  | `reference` | `done` | 0 | `…-reference.sqlite` |
  | `resumed-a` | `failed` | 1 | `…-resumed.sqlite` |
  | `resumed-b` | `killed` | `null` | the same, `store_id` `a71da55b…` |
  | `resumed-c` | `done` | 0 | the same |
  | `resumed-d` | `done` | 0 | the same |

  - `resumed-a`'s `failed` and `resumed-b`'s `killed` are the user's decisions, the latter written
    by a person through `Run.finish`. `resumed-d` alone omits `--no-prune-unvalidated`.
  - Each interrupter log holds exactly one `sent SIGTERM`, naming pid and stage.
  - After the review, none of the eight pids the agent started answers `kill -0` (the five
    builds, both interrupters, and the orphaned `gcs_server` 6426). No Ray process is alive, and
    `ray status` finds no instance.
- **(R1) Nothing stored was recomputed.** The orchestrator parsed every `Queue summary` line of
  the five `stdout.log`s itself, as lookup / compute / store.
  - **Before the interruptions.** Every stage that finished before an interruption shows 0 / 0 / 0
    on every later launch. `resumed-b`'s policy stage did 291 / 291 / 291, but `resumed-a` never
    reached that stage.
  - **`GkSource`.** `resumed-a` printed no summary for this stage; it stored 161 rows by its
    fingerprint. `resumed-b` did 175. 161 + 175 = 336, and less the 45 pruned by `resumed-d`, 291,
    the reference's.
  - **`QuadSourceIntegral`.** `resumed-b`'s last progress line read 485 computed and 451 stored.
    485 rows were on the store after it, so the 34 stores in flight landed. `resumed-c` did 379 /
    379 / 379, and 485 + 379 = 864, the reference's. `resumed-d` did 0 / 0 / 0 everywhere.
- **(R1) The fingerprints.** The orchestrator ran `fingerprint_store`, read-only, on both stores.
  - Both are at format 6, digest `10b8725f…1a1f`, with problems `{}`. Each equals its sidecar
    (comparison `[]`, nothing written).
  - `compare_fingerprints(reference, resumed)` is `[]`. All 21 classes agree in count and digest.
  - Both stores' files have the same SHA-256 before and after the review's reads.
- **(R2) Integrity.**
  - On all ten files, opened `mode=ro`: `integrity_check` is `ok` and `foreign_key_check` returns
    no rows.
  - `replicated_serial_census.py` on both primaries: every table `OK`, `=> no divergence`.
  - Its 14 tables are `config/sharding.py`'s 13 replicated tables plus `BackgroundModel_tags`.
    Every other table on a shard that neither list names is a sharded object's `_tags` or
    `GkSource_parents`. So it omits nothing.
  - `read_inventory` gives no store-level and no class problem on either store.
  - Both primaries' `replication_in_flight` are empty, and no `*-journal` exists.
- **(R3) The readers.**
  - Each of the six logs under `var/datastore-integrity-10/extract/` has exactly one
    run-selection line, `reading run "default", the only run in this datastore`. Each ends in the
    accepted `RuntimeError` on `QCD_Cosmology`.
  - After each, `run_extractions.out` records the fingerprint matching the sidecar, an empty
    `replication_in_flight` and no Ray process left. The orchestrator's fingerprint above is the
    one after the last reader.
  - The `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` write
    was not observable. The `GkSourcePolicy` rows it would get exist, so there was nothing to
    insert; the fingerprint and the in-flight record did not move. Byte-level file hashes around
    that one script were not taken.
- **(R3) The inventory report.** The orchestrator rendered all four with its own command.

  | Store | Mode | Exit | Wall | Bytes | Lines |
  |---|---|---|---|---|---|
  | `reference` | plain | 0 | 6 s | 881,293 | 1,782 |
  | `reference` | verbose | 0 | 6 s | 13,282,654 | 7,843 |
  | `resumed` | plain | 0 | 6 s | 881,291 | 1,782 |
  | `resumed` | verbose | 0 | 7 s | 13,282,652 | 7,843 |

  The two-byte difference is the store's name on the first line.
  - **Labels.** The orchestrator's own regex treats a `- ` or `> ` line ending `  #<n>` (before any
    `  | …`) as defining `<Class>#<n>`, and a `parent set S<n>:` line as defining `S<n>`. It
    counts every other `<Class>#<n>` and `S<n>` as a reference.
    - Plain: 1,674 labels defined, 1,663 named. Verbose: 5,730 defined, 5,727 named.
    - In all four, none is defined twice, none is named without a definition, and no 64-hex
      digest appears. The counts match the record's (1,669 + 5 and 5,439 + 291 defined).
  - **The two stores' reports** are identical line for line, plain and verbose, once the first
    line and every header's timestamp range are masked.
  - **The longest record line by labelled class, verbose:** `GkWKBIntegration` 136, `QuadSource`
    127, `GkSource` 100, `GkNumericIntegration` 82, `TkWKBIntegration` 84, `TkNumericIntegration`
    78, `GkSourcePolicyData` 60. This is the first measurement of 09d's report on real rows.
    Plain, a `> ` not-shown row carries its tag list, up to 334 characters (`GkWKBIntegration`).
    Most of each report's bytes are its parent-set lines, which are not record lines.
- **(R2) The fault copies** (`fault_cases.txt`).
  - Cases 1–3 refuse with `ReplicatedDivergence`, verbatim, each with "Nothing was repaired". Case
    1 names `redshift`, shard [1], serial 1182. Case 2 names `BackgroundModelValue`, shard 1, key
    `(model_serial=1, z_serial=1)`. Case 3 names `shard_keys`, `key_serial` 1003.
  - Case 4 prints `copied 1 row(s) of "redshift" from shard 0 to shard 2: 1182`. It opens, and its
    census afterwards is `no divergence`.
  - No `.sqlite` remains anywhere in the session scratchpad. The `var/` snapshot finds no new path
    off the list.
- **(R4) The test fix.** The orchestrator extracted the mutation from `git show HEAD:<log>` and
  applied it with plain `git apply` in a `git archive` export of `b1774d8`, made into its own git
  repository.
  - The amended test gives `FAILED (failures=4)`: `TkNumericIntegration`, `TkWKBIntegration` and
    `GkSource` at `2 != 3`, `GkSourcePolicyData` at `1 != 3`. These are exactly the logged
    classes.
  - With `d144d04`'s test file in place, it gives `OK`.
  - The export's tracked diff is empty after `git apply -R`.
  - The three classes it cannot reach on this fixture are
    `[10-distinct-lines-test-cannot-fail-for-three-labelled-classes-on-the-fixture]`.
- **Records.**
  - The §1 row, "Decided during 10", V1–V3, the 09d issue on §4 with its closing note, four §3.3
    entries (one with its `**Assigned (2026-10-02):**` line to `handover`) and §5.16 are present.
    The 09d issue is gone from §3.3.
  - Counted, the index has **102** rows: one deleted, four opened, the `handover` one under §1.1.
    §1.15 has its sentence.
  - The record's §2.6 checklist names all twelve issues open on this board.
  - The record's run table matches the run directories in pid, state and exit code.
- **`var/`, suites, `black`.**
  - All 46 files of the pre-dispatch snapshot are byte-identical. Of 39 new files and 5 new
    directories, none is off the list.
  - Suites on `b1774d8`, serially, in the checkout, at a load average of 15–18: AdaptiveLevin 32,
    ComputeTargets 665, CosmologyModels 39, Datastore 551, LiouvilleGreen 148 (1 skipped) and
    RunRegistry 199, all OK. Every count is unchanged.
  - `black --check` is clean on all nine Python files added or changed.
- **Probes.** None is run. The commit's diff over `Datastore/`, `ComputeTargets/` and `tools/`
  is the one test file.

- **Remnants, fixed in `53fa6c5`** (*"Tidy two remnants of datastore-integrity prompt 10"*):
  - README §2's row for 10 read "**written** 2026-10-01, its decisions taken; after 09d", and its
    header paragraph did not record 10's landing. That row was outside the agent's files; its
    report flagged it.
  - This board's header said "all but 10 reviewed".
- **What the run found that no stand-in could.**
  - **A3 v2's launch resumes without pruning.** It was the user's diagnosis: pruning at startup
    is the designed recovery for a row stored and not yet validated. Under
    `--no-prune-unvalidated`, `resumed-b` stored 45 `GkSource` rows a second time. It is
    `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`, assigned to `handover`. With
    pruning (`resumed-d`), the resume equals the reference.
  - **A `SIGTERM` to the driver is never recorded as `killed`.** It is `failed`/1 if cleanup takes
    under ~3 s, and otherwise absl aborts the driver and nothing is written. It also leaves an
    orphaned `gcs_server`. This is
    `[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]`.
  - **The six readers exit 1 on a LambdaCDM-only store, A3 v2's.** This is
    `[10-extraction-scripts-exit-1-on-a-store-without-every-model]`.
- **Log 10 is accurate** against the run directories and `git show`.

**10 is complete.** A real build, interrupted twice and resumed, holds exactly what an
uninterrupted build holds once pruning runs. Both stores are clean replica by replica, refuse or
repair at open as prompt 02 designed, and read back without moving. **The user decides** whether
the campaign closes, and whether the two stores are retired (D10-5; `store retire` is theirs to
run).
  - **Decided (2026-10-02):** the user closed the campaign, and decided to retire both stores.
  - **Retired (2026-10-02), by the user**, with `store retire` at `f4f1896`: `datastore-integrity-10-reference` at 01:19:49 and `datastore-integrity-10-resumed` at 01:19:58.
    Each fresh fingerprint matched its recorded one, digest `10b8725f…1a1f`. Each store's primary
    and four shards were deleted, and its sidecar is left as a tombstone in `var/datastores/`. The
    run directories under `var/runs/datastore-integrity-10-*` and the extraction outputs under
    `var/datastore-integrity-10/` remain, and every number is in
    `docs/datastore-integrity-verification.md`.

**05–10 are held, not unplanned.** Their charters are fixed in README §2 and cannot drift to fit
what 01–04 find; what waits is their method.

---

## 2. Items

| Item | Kind | Description | Prompt | Status |
|---|---|---|---|---|
| W1 | **FACILITY** | `replication_in_flight` in the primary: one row, committed before the controller's call, deleted after the last replica returns. One helper through which the get, store and validate paths all go. | 01 | ✅ `ShardedPool._replicated_write`; the record is committed before every replicated call, and left set by any failure after it |
| W2 | **GUARD** | Every replica's answer checked against the controller's serial, scalar and vector forms; `ReplicationMismatch` names the shard, class and both serials. | 01 | ✅ `Datastore/replication.py`; get (both forms), store and validate checked before the record is cleared |
| W3 | **REMEDY** | Replica stores of `wavenumber_exit_time` and `BackgroundModel` insert-or-verify by serial; `BackgroundModel` replicas pass each value's serial, so copies are row-identical. | 01 | ✅ |
| W4 | **REMEDY** | One timestamp per replicated write, chosen by the driver and passed to every shard through `Datastore._insert`. | 01 | ✅ `insert_timestamp`, a per-call argument; absent on every sharded write |
| W5 | **FACILITY** | `_reconcile_replicated_tables()` in the constructor, before any actor, no Ray: hot journals rolled back, every replicated table compared across shards by full row (timestamp exempt for old rows). | 02 | ✅ after `_check_shard_files`, before the broker; 13 replicated tables and `BackgroundModel_tags` read through `build_schema`'s tables on plain SQLAlchemy connections, keyed by serial (`BackgroundModelValue` by `(model_serial, z_serial)` without `serial`; the tag table by its key), `timestamp` never compared, plus the primary's `shard_keys`; a clean store writes nothing (checksums); +94 ms at 4 shards, +252 ms at 20 (production size, medians) |
| W6 | **REMEDY** | With an in-flight record set, the missing rows of its class copied forward from its controller, `BackgroundModel` as a unit with `validated` recomputed; the record then cleared. | 02 | ✅ from the record's `controller_shard` only, rows of its class only, verbatim, each carrying the record's `started`; `source`/`response` only turned on; `validated` recomputed by `sqla_BackgroundModelFactory.validate` in a savepoint, never written `False`; a second full comparison inside the repair's transactions, then the shards committed and the record cleared, last; one printed line per class and shard |
| W7 | **GUARD** | Any other difference refuses the open with `ReplicatedDivergence`, including a serial split and a row the controller lacks. Closes `[00-replicated-writes-can-diverge-across-shards]`. | 02 | ✅ `Datastore/replication.py`; each difference named by class, shard and serial or key, a split with both serials; before any actor, nothing written, the record left as it was; sharded tables never read. Issue closed on the store-fingerprint board's §4 |
| L1 | **REMEDY** | `wavenumber_exit_time.build` by serial equality on atol, rtol and `==` on stepping; the dead `ORDER BY` removed; the query out of the loop. Closes `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` and `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`. | 03 | ✅ joins to the tolerance table kept, feeding the returned `log10_tol` values; one `SELECT` per `build()`, pinned by `TestExitTimeOneStatement` |
| L2 | **REMEDY** | `IntegrationSolver.build` by `stepping ==` and `one_or_none()`. Closes `[03-integrationsolver-stepping-minimum-lookup]`. | 03 | ✅ `.scalar_one_or_none()` |
| L3 | **GUARD** | `.scalar()` → `one_or_none()` on the eight identity lookups (audit §3.8), so a duplicate raises. | 03 | ✅ `.scalar_one_or_none()` on all eight; the other 21 `.scalar()` calls in `ObjectFactories/` judged not identity lookups, listed in the log |
| L4 | **REMEDY** | `redshift.build` matches `z = 0` (audit §3.9). | 03 | ✅ an explicit `z == 0` branch inside the one `.filter(...)` call |
| L5 | **RECORD** | `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]` narrowed: after L1 the two scripts miss rather than silently reuse. | 03 | ✅ narrowed on the tolerance-convergence board and in the index; every production caller (`main.py`, the six `extract_*.py` scripts) confirmed unchanged |
| G1 | **GUARD** | The read-back guard visits every function of every factory, `read_batch` and `_build_impl_*` included, linking a nested helper's parameter to its row. | 04 | ✅ `analyse_read_batch` / `inline_nested_row_helper`; `EXPECTED_COVERAGE` grows by 10 (probe_02.out's 16 collapse under the module's existing "module:class.variable" key — see log) |
| G2 | **GUARD** | A column-existence test: every `<table>.c.<name>` and every insert key resolves against the built schema. | 04 | ✅ `TestColumnExistence` / `column_existence_problems`, same module |
| G3 | **RECORD** | `KNOWN_UNFIXED`: what G1–G2 find today, for prompt 05 to empty. Closes `[01-read-batch-is-outside-the-guard]`. | 04 | ✅ `KNOWN_UNFIXED` +2 (both `WKB_criterio`, P6), `KNOWN_UNFIXED_COLUMNS` (new) +3 (`z_serial`/`model_id`, P6; `validated` insert key, P8) — the five the probes predicted, nothing new |
| G4 | **GUARD** | The analyser visits every function of every factory class and every module-level function, not a list of names; `validate_on_startup`'s seven bindings join coverage. | 04b | ✅ `iter_functions`; `_ROW_HELPERS` gone; `EXPECTED_COVERAGE` +13 (30→43: the 7 `validate_on_startup` bindings plus 6 that regain a sibling once the function rejoins the key) |
| G5 | **GUARD** | `EXPECTED_COVERAGE` and `KNOWN_BLIND` keyed by function, so each helper is pinned on its own; its projection keeps all 30 of 04's entries. | 04b | ✅ `Finding.coverage_key` ("module:class.function.variable"); `Finding.key` unchanged, so `KNOWN_UNFIXED`/`KNOWN_UNFIXED_COLUMNS` keep their keying and contents exactly; projection verified by computation in the log |
| G6 | **GUARD** | G2 walks module-level functions, reports an unknown `tables["X"]` as a problem, and pins every unresolved table base and unchecked insert with a reason. | 04b | ✅ `KNOWN_UNRESOLVED_TABLES` (12 pairs), `KNOWN_UNCHECKED_INSERTS` (13 sites), both declared with no resolution rule applied (log's deviation 1); `tables["X"]`/`inserters["X"]` naming no schema table now convicted (none in this tree today) |
| P1 | **REMEDY** | The tagged join in `QuadSourceIntegral` / `OneLoopIntegral` `read_batch` on `table.c.serial`. Closes `[00-tagged-read-batch-joins-an-unselected-alias]`. | 05b | ✅ both joins now `tab.c.parent_serial == table.c.serial`; `probe_01b`'s verified shape |
| P2 | **REMEDY** | `QuadSourceIntegral.read_batch` gives its exit times their own serials (audit §3.2). | 05b | ✅ `q_exit_serial`/`r_exit_serial` selected and used as `q_exit_payload`/`r_exit_payload`'s `store_id`; rest of `make_object` checked, no other wrong-table serial found |
| P3 | **REMEDY** | `break_point_kind` in the Tk/Gk numeric value reads and `GkNumericValue.read_batch`, with atol and rtol required and more than one parent per `z_source` refused; one key name for `z_init`. Closes `[00-numeric-value-parent-lookup-omits-break-point-kind]`. | 05c | ✅ `break_point_kind == <Target>.BREAK_POINT_KIND` in all three, from the compute target's constant; `payload["atol"]`/`["rtol"]`; a store without the column refused with `RuntimeError`; `GkNumericValue.read_batch` refuses a second parent per `z_source`; `TkNumericValue` reads `z_init` only. Issue closed on the store-fingerprint board's §4 |
| P4 | **REMEDY** | `z_init` in the `GkWKBValue` reads; more than one parent per `z_source` refused (audit §3.4). *(Widened 2026-09-28 when 05c was written: `TkWKBValue`'s reads have the same omission, refused per `(model, k)`.)* | 05c | ✅ optional `z_init` in all four reads, filtered with the parents' absolute `DEFAULT_FLOAT_PRECISION` expression; `GkWKBValue.read_batch` refuses per `z_source`, `TkWKBValue.read_batch` per `(model, k)` |
| P5 | **REMEDY** | The two `…_is_response` flags read `response` (audit §3.11). | 05c | ✅ `GkNumericIntegration.build`'s `z_source_is_response` and `GkNumericValue.read_batch`'s `z_response_is_response` |
| P6 | **REMEDY** | `WKB_criterio`, `z_serial`, `model_id`, `friction` (audit §3.10–§3.11), under D2. | 05a | ✅ `WKB_criterion` in both WKB value factories' `_build_impl_serial`; `z_source_serial` in `GkSourceValue._build_impl_serial`; `model_serial` in `QuadSourceValue._build_impl_model`; `friction=row.friction` added to `TkWKBValue.read_batch`'s `make_obj`. `KNOWN_UNFIXED` and the two P6 `KNOWN_UNFIXED_COLUMNS` entries struck |
| P7 | **REMEDY** | Tags default to `[]` at every value helper and `read_batch`; the three `raise ("…")`. | 05a | ✅ nine sites (recount below), `Optional[List[store_tag]]` → `List[store_tag]` with each; the three `raise ("…")` → `raise RuntimeError("…")`, message unchanged; sweep for any other `raise (`/`raise "` under `ObjectFactories/` found none beyond the three named |
| P8 | **REMEDY** | `OneLoopIntegral`: its own tag table, `read_batch` a method with a return value, as D3 bounds. Closes `[00-oneloop-lookup-joins-the-wrong-tag-table]`. | 05b | ✅ `build()` joins `OneLoopIntegral_tags`; `read_batch` is now `sqla_OneLoopIntegral_factory`'s `@staticmethod`, tags default `[]`, returns a list; `"validated": False` dropped from `store()`; `source_serial` left alone (D3) |
| P9 | **GUARD** | Round-trip tests of every `read_batch`, tagged and untagged; `KNOWN_UNFIXED` empty. | 05a–05c | ✅ 05a's and 05b's shares: `test_read_path_mechanics.py` (05a) and the new `test_integral_read_batch.py` (05b, 10 tests: tagged `QuadSourceIntegral.read_batch`, P2's exit serials, `OneLoopIntegral.build`'s tags including probe_06's coincidence case, `OneLoopIntegral.read_batch` tagged/untagged/omitted, and the deferred `source_serial` defect kept visible). `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are both empty. ✅ **Closed by 05c** (2026-09-28): the new `test_value_reads_parent_key.py` round-trips `GkNumericValue`, `GkWKBValue` and `TkWKBValue.read_batch` (the live payload shape, the refusal, `z_init` selection), and its `READ_BATCH_ROUND_TRIPS` pins every `read_batch` under `ObjectFactories/`, found by walking the sources as the guard does, to the module that round-trips it; a new `read_batch` fails until its test exists |
| S1 | **REMEDY** | `QuadSource.Tq_serial` → `TkNumericIntegration.serial`; the `Tr_serial` comment. Closes `[00-quadsource-tq-serial-has-the-wrong-foreign-key]`. | 06 | ✅ `Tq` shown to share `QuadSource`'s shard (quoted in the log); `Tr_serial` keeps no key, its comment now names `TkNumericIntegration` and the cross-shard reason. Issue closed on the store-fingerprint board's §4 |
| S2 | **REMEDY** | `GkSourcePolicyData.wavenumber_exit_serial` gains its foreign key. | 06 | ✅ `ForeignKey("wavenumber_exit_time.serial")`; `index` and `nullable` unchanged |
| S3 | **REMEDY** | `real_store_fixtures.py`'s undeclared-reference map corrected. | 06 | ✅ `Tr_serial` → `TkNumericIntegration`; the `GkSourcePolicyData` entry removed (now declared); `QuadSourceIntegral.source_serial` left undeclared. The inventory and fingerprint relabel tests do not move (`Tq_serial`/`Tr_serial` are provenance) |
| S4 | **GUARD** | A test that runs `PRAGMA foreign_key_check` over every shard of a store written by the real factories and finds no violation. | 06 | ✅ `test_foreign_key_check.py`, 8 tests: the full store on the test's own rows (a `QuadSource` whose `Tq_serial` is a Tk serial on its shard and no `QuadSource` serial there, because the fixture's own rows satisfy the wrong key by coincidence), a stand-in-shard store, and two dangling controls. Cannot see `Tr_serial`, `QuadSourceIntegral.source_serial` or `shard_keys` |
| S5 | **RECORD** | The first deliberate schema-witness change, under D5, including the exit-time and `IntegrationSolver` `stepping_mode` flags made exact. | 06 | ✅ `schema_at_datastore-integrity-06.json` beside `schema_at_base.json`; `WITNESS` repointed; the diff (8 hunks, 30 lines: S1, S2 and the two flags, nothing else) is in the log; the golden fingerprint did not move |
| K1 | **REMEDY** | `numeric_serial` on both WKB tables, in the lookup; `z_init` and the initial values leave the key; `main.py` passes the parent. Closes `[20-wkb-rows-consume-numeric-initial-data]`. | 07 | ✅ `numeric_serial`, indexed, `ForeignKey` to its numeric table (both shard on `k`): NOT NULL on Tk, nullable on Gk with NULL = pure WKB (the only parentless branch). `build()` keys on a serial or `GkWKBIntegration.PURE_WKB` (`IS NULL`) and raises on neither; an explicit `labels_only=True` reader mode refuses two matches, naming their parents; the four value reads take an optional parent in place of 05c's `z_init`. `z_init` and the initial values are payload; `store()` refuses them unless they are the parent's stop values bit for bit (and the parent is for the same `k`, model and `z_source`). A pre-07 table is refused with the D4 message. `main.py`: the numeric lookup ahead of each existence query, every WKB lookup and construction passes the parent, the Gk choice lifted unchanged into `Gk_WKB_numeric_parent`. Inventory: `numeric` a key parent, `z_init` out, `Parent(nullable=True)`; `FINGERPRINT_FORMAT` 2. Issue closed on the GkTk-remedial board's §4 |
| K2 | **REMEDY** | `_drop_order` respects the new key. | 07 | ✅ `tk-wkb` before `tk-numeric` (`gk-wkb` already preceded `gk-numeric`); the comment corrected; a test derives the order from every declared foreign key between drop actions. The order *within* an action is `[07-drop-actions-list-a-parent-table-before-its-children]` (§3.3) |
| K3 | **REMEDY** | `quad_policy_serial` on `QuadSourceIntegral`, in the key, through target, factory, `main.py` and the extraction script. Closes `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`. | 08 | ✅ `quad_policy_serial`, NOT NULL, indexed, `ForeignKey("QuadSourcePolicy.serial")`, after `policy_serial`; in `build()`'s filter, written by `store()`, in `read_batch`'s filter and a key parent in `inventory_records`. The target takes `quad_policy` beside `policy`, required. `main.py`'s existence query and compute call pass `QuadSource_policy_1pt5`, which is created from `config.defaults.DEFAULT_QUADSOURCE_POLICY_1PT5`; the extraction script finds it through a new `QuadSourcePolicy.read_table` and refuses on none or several. A pre-08 table is refused (D4) in `build()` and `read_batch()`. Witness 08 (6 hunks, 43 lines, one class); fingerprint format 3, `QuadSourceIntegral` and `QuadSourcePolicy` moved |
| K4 | **REMEDY** | `QuadSource` keyed on `Tq_serial` and `Tr_serial`. | 09a | ✅ both serials in `build()`'s filter, by serial; `Tr_serial` indexed (still no foreign key); a `labels_only` reader mode refusing two matches, naming both pairs; the target carries the serials from construction and `compute()` asserts the rows it is handed, raising naming both; `inventory_records` declares `Tq` (same shard) and `Tr` (cross-shard). `main.py`'s `QuadSource` stage and the `QuadSourceIntegral` stage's `QuadSource` lookup keyed on the Tk rows; three readers use the reader mode. No column added, so no old store refused (D4). Witness 09a (3 hunks, 10 lines, `Tr_serial`'s index); fingerprint format 4, only `QuadSource` moved |
| K5 | **REMEDY** | `QuadSourceIntegral` keyed on `source_serial`, `data_serial` and its Tk rows. | 09c | ✅ four new columns, `Tq_numeric_serial`, `Tq_WKB_serial`, `Tr_numeric_serial`, `Tr_WKB_serial` (NOT NULL, indexed, no foreign key: the Tk rows are on q's and r's shards); all six parents in `build()`'s filter (by serial, all required), written by `store()` (which refuses an object missing one), handed back on every object by `read_batch()`, which filters on any it is given; `compute()` asserts that the six rows it is handed are the six it is keyed on, that the Tq and Tr numeric rows are its source's `Tq_serial` and `Tr_serial`, and that each WKB row starts from its numeric row, raising naming both; `main.py`'s stage resolves every work item's six parents serial-only, in `resolve_QuadSourceIntegral_parents`, before its existence query; the extraction script refuses two rows at one `(k, q, r, z_response)`, naming their distinguishing parents; a pre-09c table refused with 07's form of message, a pre-08 one still with 08's |
| K6 | **REMEDY** | `GkSource` keyed on its Gk numeric and WKB rows, which are a set: recommended, a `GkSource_parents` table and a `parents_digest` key column. | 09b | ✅ the recommended shape. `GkSource_parents` (own serial; `gk_source_serial`, `z_source_serial`, nullable `numeric_serial` and `wkb_serial`, each an indexed foreign key, all on `k`'s shard), one row per `z_source`, written by `store()` in the `GkSource`'s transaction from the set `assemble_GkSource_values` reports it assembled; `parents_digest` (`String(64)`, NOT NULL, indexed) computed from those rows, in `build()`'s filter; an empty set refused. The Gk value reads attach their parent (`_parent_serial`) and have a serial-only form (`parents_only`); they and `TkWKBValue`'s now refuse a pre-07 store. `compute()` asserts the values it is handed are read from the set it is keyed on. `main.py`: one helper, `resolve_GkSource_parents`, resolves the set serial-only; all six `GkSource` lookups are keyed on it, the GkSource stage's before its existence query. A `labels_only` reader mode (four readers, and `GkSourceValue._build_impl_model`) refuses two matches, naming their digests and where their sets differ. A pre-09b table is refused (D4). Witness 09b (9 hunks, 246 lines, `GkSource` and `GkSource_parents`); fingerprint format 5, `GkSource` and `GkSourcePolicyData` moved |
| K7 | **FACILITY** | The store inventory resolves key parents across shards, so cross-shard parents are fingerprint identity. | 09a (cross-shard `Parent`), 09b (set-valued), 09c (complete) | ✅ 09a: `Parent(cross_shard=True)` resolves against every shard's records of its class (`ShardContext.all_digests`), to the parent's reference digest and never its serial; a serial two shards hold differently, or none holds, is `unresolved-parent`; one read-only pass. 09b: `ParentSet`, a key field whose value is the digest of the sorted members of a member table, each member resolved to its parents' reference digests (never `parents_digest`, never a serial); `GkSource` declares its set through `GkSource_parents`, read once per shard; an unresolvable member is `unresolved-parent`, a member row with no owner `orphan-member`, an owner with none `empty-parent-set`, an absent member table `absent-table`; the members are kept in `ClassInventory.members` and `tools/inventory_report.py` renders them by physical labels. **09c: complete.** `QuadSourceIntegral` declares its six parents: `data` same-shard, and `source` and the four Tk rows with 09a's cross-shard form, each resolving to its row's own identity; a `Tq` parent varied to another shard's row is resolved and is identity (a test beside the fixture, since the fixture's own `Tq` rows are all on their integral's shard). |
| V1 | **VERIFY** | A scoped build through the registry into a new throwaway store, interrupted and resumed. | 10 | ✅ `reference` straight through; `resumed-a` killed in the `GkSource` stage, `resumed-b` in the `QuadSourceIntegral` stage, `resumed-c` and the user's pruned `resumed-d` to `done`. No stage finished before an interruption computed anything afterwards; `compare_fingerprints(reference, resumed)` empty at format 6. Under `--no-prune-unvalidated` the resume duplicated 45 `GkSource` rows (§3.3), which `resumed-d`'s pruning removed. Neither interruption recorded `killed` (§3.3) |
| V2 | **VERIFY** | Every target extracted from it; its fingerprint; the census of audit §4.4. | 10 | ✅ census `no divergence`, `integrity_check` `ok` and `foreign_key_check` empty on all ten files, no inventory problem, on both stores; `--inventory` and `--inventory-verbose` clean on both, every label defined once, no digest, the reports identical with timestamps masked; the four fault copies refused or repaired as prompt 02 designed. Every `extract_*.py` reads LambdaCDM in full and exits 1 on `QCD_Cosmology` (accepted by the user; §3.3); no reader moved the fingerprint |
| V3 | **RECORD** | The campaign closed; what A3 v2's launch must now pass. | 10 | ✅ the checklist in `docs/datastore-integrity-verification.md` §2.6, each item pointing to where 10 exercised it, with what 10 could not exercise and every open issue's meaning for A3 v2 |

---

## 3. Active and unresolved issues

### 3.1 Assigned to this campaign (2026-09-26)

Each stays on its owning board, which carries an **Assigned** line, and closes on that board's §4.

| Issue | Owning board | Prompt |
|---|---|---|
| `[00-replicated-writes-can-diverge-across-shards]` | store-fingerprint | 01 (**narrowed** 2026-09-26), 02 (**closed** 2026-09-28, on that board's §4) |
| `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` | tolerance-convergence | 03 (**closed** 2026-09-29, on that board's §4) |
| `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` | store-fingerprint | 03 (**closed** 2026-09-29, on that board's §4) |
| `[03-integrationsolver-stepping-minimum-lookup]` | GkTk-remedial | 03 (**closed** 2026-09-29, on that board's §4) |
| `[01-read-batch-is-outside-the-guard]` | datastore-readback | 04 (**closed** 2026-09-26, on that board's §4) |
| `[00-tagged-read-batch-joins-an-unselected-alias]` | store-fingerprint | 05b (**closed** 2026-09-28, on that board's §4) |
| `[00-numeric-value-parent-lookup-omits-break-point-kind]` | store-fingerprint | 05c (**closed** 2026-09-28, on that board's §4) |
| `[00-oneloop-lookup-joins-the-wrong-tag-table]` | store-fingerprint | 05b (**closed** 2026-09-28, on that board's §4) |
| `[00-quadsource-tq-serial-has-the-wrong-foreign-key]` | store-fingerprint | 06 (**closed** 2026-09-29, on that board's §4) |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | 07 (**closed** 2026-09-29, on that board's §4) |
| `[00-quadsourcepolicy-rows-are-referenced-by-nothing]` | store-fingerprint | 08 (**closed** 2026-09-29, on that board's §4) |

### 3.2 Opened by the audit (2026-09-25), recorded for their owners

None is assigned to a prompt of this campaign.

- **[00-the-inventory-cannot-see-a-serial-split]** *(opened 2026-09-25 by the audit)*
  - **The defect.** The structured inventory compares replicated classes across shards by key,
    tags, `validated` and value count. It does not compare serials, the `source`/`response` flags,
    timestamps or value contents (audit §4.3, `replicated_divergence_probe.py`). A store holding a
    serial split, or shards that disagree on a flag, fingerprints with no problem.
  - **Impact.** The fingerprint is not evidence that a store's replicated tables agree. After
    prompt 02, the check at open is that evidence for any store the tree can open. The fingerprint
    reads stores the tree may not open.
  - **Next step.** Decide whether the fingerprint should carry replicated agreement, or say plainly
    that it does not. Not this campaign's: README §1.
- **[00-oneloop-factory-cannot-round-trip-a-row]** *(opened 2026-09-25 by the audit; **narrowed**
  2026-09-28 by prompt 05b)*
  - **The defect.** `sqla_OneLoopIntegral_factory.build` puts no `source_serial` in the payload
    and the table has no such column, while `ComputeTargets/OneLoopIntegral.py:43` reads
    `payload["source_serial"]`, so every hit raises `KeyError`. **The `store()` half — a
    `validated` key written to a table with no such column — is fixed** (prompt 05b, D3, "Repair
    the integral read_batch join, exit-time serials and OneLoopIntegral"); only the
    `source_serial` half of the original audit §3.12 finding remains, and
    `Datastore/tests/test_integral_read_batch.py`'s
    `TestOneLoopIntegralSourceSerialStaysBroken` keeps it visible (constructing the real
    `OneLoopIntegral` from what `read_batch()` passes still raises `KeyError: 'source_serial'`).
  - **Impact.** None while nothing builds the target. The first one-loop build cannot read back a
    row it stored.
  - **Next step.** Part of how `OneLoopIntegral` is keyed, which the user put downstream beside
    `[02-oneloopintegral-is-a-ninth-keyed-object-type]` (README §6.1.4, D3).
- **[00-the-5pt0-policy-pair-is-created-and-never-used]** *(opened 2026-09-25 by the audit)*
  - **The defect.** `main.py` creates `GkSource_policy_5pt0` and `QuadSource_policy_5pt0` and
    passes them into `run_pipeline`, which uses neither (audit §2.5).
  - **Impact.** Two dead rows in every store, which a reader may take for configuration that
    applied.
  - **Next step.** An author's decision: remove them, or say what they are for. Not a key defect,
    so not this campaign's.

### 3.3 Opened by this campaign's prompts

- **[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]** *(opened 2026-10-02 by prompt 10)*
  - **The defect.** A3 v2's launch, `docs/handover/quadsource_atol_sweep.py --build` (`run_build`),
    passes `--no-prune-unvalidated` to `main.py`, and so does `--build --resume`, which runs the same
    `sys.argv`. Prompt 10's §1.3 copied the flag from there. Pruning at startup is the designed
    recovery for a row stored but not yet validated when a build is interrupted (the user,
    2026-10-01: "The bug here is the `--no-prune-unvaiidated`. Pruning is there precisely to deal
    with this situation."). Under the flag such a row stays; the class's existence query serves
    only validated rows, so the resume computes it again and stores a second row beside it.
  - **Measured** (log 10; `docs/datastore-integrity-verification.md` §2.1.4). A `SIGTERM` in the
    `GkSource` stage left 161 rows, 45 of them with all their values and `validated = 0`. The
    resume under the flag computed 175 (130 missing + those 45) and stored them, so the store
    held 336 `GkSource` rows, and the inventory reported `duplicate: GkSource: 45 key-and-tag
    set(s) are held by more than one record (90 records, all kept)`. Each pair: the same key,
    `parents_digest` and `z_samples`, one unvalidated and one validated. The fingerprint differed
    from the uninterrupted reference's in `GkSource` alone. One more launch with `main.py`'s default
    pruning removed the 45 (and their values, tags and parent-set members), computed nothing, and
    left a store whose fingerprint equals the reference's.
  - **Which classes can do it.** From the code, not measured: all seven classes with
    `validate_on_startup`, which are the ones with a `validated` flag that a store can leave unset —
    `BackgroundModel`, `TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`,
    `GkWKBIntegration`, `QuadSource` and `GkSource`. Each one's `build()` filters on
    `validated == True`, and `BackgroundModel`'s replica store says it in a comment: "a store
    interrupted before its validation leaves one on every shard, which a later store legitimately
    writes beside". `QuadSourceIntegral`, `GkSourcePolicyData` and `OneLoopIntegral` have no
    validation step. Only `GkSource` was caught between store and validate here.
  - **Impact.** **It blocks A3 v2's resume as planned.** An interrupted A3 v2 build resumed with
    `--build --resume` gains a duplicate for every row stored but not validated at the
    interruption: an inventory problem, a fingerprint that differs from an uninterrupted build's,
    and, for a reader that refuses more than one match, a failure.
  - **Next step.** Drop `--no-prune-unvalidated` from `run_build`'s `sys.argv`, at least for
    `--resume`, or say why it is needed and prune unvalidated rows another way. That is the sweep
    script's, which this campaign does not edit (README §5 rule 12).
  - **Assigned (2026-10-02):** to the **`handover`** campaign, which owns A3 v2's launch and
    `quadsource_atol_sweep.py`, by the user's decision that the flag is the defect. Its board is not
    edited here; the index row is under §1.1.
  - **Assigned (2026-10-02, again):** reassigned from `handover` to `prompts/a3-v2-readiness` prompt
    **05**, by the user's decision that one campaign owns every A3 v2 readiness defect (that
    campaign's D4). Index row moved from `docs/OPEN_ISSUES.md` §1.1 to §1.17.
- **[10-distinct-lines-test-cannot-fail-for-three-labelled-classes-on-the-fixture]** *(opened
  2026-10-02 by prompt 10)*
  - **The defect.** Prompt 10 amended `test_verbose_renders_every_distinct_record_as_a_distinct_line`
    to compare record lines with their own label taken off. It compares lines within a tag set, and
    `build_full_store` holds one record per tag set of `QuadSource`, `GkNumericIntegration` and
    `GkWKBIntegration`, so for those three it cannot fail whatever the line shows.
  - **Measured** (log 10, the deliberate-breakage record). Under a mutation that prints a labelled
    class's record with neither fields nor value count, the amended test fails for
    `TkNumericIntegration`, `TkWKBIntegration`, `GkSource` and `GkSourcePolicyData` (4 sub-tests),
    and not for the three.
  - **Impact.** None today: 09d's `TestDistinctStaysDistinct` builds pairs of each that differ in
    one field, and `test_verbose_shows_every_key_field_of_every_class` still sees a dropped field.
  - **Next step.** Give the fixture a second record of each in one tag set, or compare across tag
    sets; either edits fixtures or tests that prompt 10 may not.
- **[09b-gksource-members-are-not-checked-against-their-rows]** *(opened 2026-09-30 by prompt
  09b)*
  - **The defect.** `sqla_GkSource_factory.store()` writes the `GkSource_parents` rows it is given
    without reading them: nothing checks that a member's numeric and WKB rows are for the
    source's `k` and model, that each is at the member's own `z_source`, or that a member's WKB
    row starts from its numeric row (its `numeric_serial` is the member's numeric row, or NULL).
    Measured on the in-memory store of `test_gksource_parent_set.py`: a source stored with
    members `((21, 42, None), (22, 41, 52))` -- Gk numeric 42 is at `z_source` 22, 41 at 21, and
    Gk WKB 52 at 23 -- is written as given (log 09b, observations).
  - **Impact.** None in the pipeline: `compute()` asserts that the values it is handed were read
    from exactly the set the object is keyed on, and that set is formed from reads filtered on
    `k`, model, tags and `z_response`, at the `z_source`s the rows themselves report. A declared
    foreign key checks only that a row exists, and SQLite does not enforce it. It is the relation
    the charter's §2 asks to be asserted between related direct inputs, which 07's
    `_check_numeric_parent` asserts for a WKB row and 09c will for `QuadSourceIntegral`.
  - **Next step.** Decide whether `store()` reads each member's rows, as 07's check reads a WKB
    row's parent, and refuses a mismatch; it would cost one query per `GkSource` over its members.
- **[07-drop-actions-list-a-parent-table-before-its-children]** *(opened 2026-09-29 by prompt 07)*
  - **The defect.** Each entry of `_drop_actions` (`Datastore/SQL/Datastore.py`) lists the
    integration table first and its tag and value tables after it, e.g. `"tk-wkb":
    ["TkWKBIntegration", "TkWKB_tags", "TkWKBValue"]`, and `_drop_actions` drops them in that
    order. `TkWKBValue.wkb_serial` and `TkWKB_tags.wkb_serial` declare foreign keys to
    `TkWKBIntegration.serial`, so under `PRAGMA foreign_keys = ON` dropping the parent while value
    rows exist fails (SQLite's `DROP TABLE` performs an implicit `DELETE`). Measured on prompt 07's
    test store, clean under `PRAGMA foreign_key_check`, holding one Tk WKB row and its two
    `TkWKBValue` rows: the real `_drop_actions(["tk-wkb"])`, the action alone, raised
    `IntegrityError: FOREIGN KEY constraint failed` (log 07, observations).
  - **Impact.** Inert: production never turns enforcement on, so every declared key is
    documentation. It makes `_drop_order`'s comment true only between actions. Prompt 07's
    `TestDropOrder` uses a store with no value rows for this reason.
  - **Next step.** Order each action's list children-first (values and tags, then the parent), and
    extend `TestDropOrder`'s schema-derived check to tables within one action. `_drop_actions` was
    outside prompt 07's licence (`_drop_order` and its comment only).
- **[05c-read-back-guard-cannot-see-reads-in-a-comprehension]** *(opened 2026-09-28 by prompt 05c)*
  - **The defect.** `Datastore/tests/test_factory_select_columns.py`'s `analyse_build` walks
    nodes in pre-order. For `[f(row.x) for row in rows]` it reaches the element `row.x` before the
    `comprehension` node that binds `row`. So the read is charged to an earlier binding of that
    name, or, with none, is dropped: it is neither checked nor declared.
  - **Where, on this tree** (an `ast` walk for comprehensions whose element reads an attribute of
    their own target, log 05c observation 3):
    - `redshift.py:113` and `wavenumber.py:112` (`read_table`);
    - `QuadSourcePolicy.py:102` (`read_table`, added by prompt 08; its four reads are the four
      columns its query selects, checked by hand; log 08, observation 4). *(Added 2026-09-30, by
      the review of 08.)*
    - `BackgroundModel.py:401` (`build`, already inside `KNOWN_BLIND`'s function);
    - the seven `invalid_serials = [nv.serial for nv in not_validated]` in `validate_on_startup`.
  - **Impact.** Nothing is missed today. The three `read_table`s select every column they read,
    checked by hand, and the seven `validate_on_startup` reads take `serial`, which their queries
    select. But the guard's docstring does not declare the hole, and no pin would catch a future defect in
    this shape. 05c wrote `TkWKBValue.read_batch`'s refusal as a `for` loop for this reason (log
    05c, deviation 6).
  - **Next step.** Teach the analyser to bind a comprehension's targets before walking its
    element, and pin what that newly decides. Or declare the shape in the docstring and in a
    pinned list. A change to the guard, which is not 05c's to make.

---

## 4. Resolved issues

- **[10-a-sigterm-to-the-scoped-driver-is-recorded-as-failed-or-not-at-all]** *(opened 2026-10-02
  by prompt 10; **closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 04)*
  - **The defect.** `docs/gktk-remedial/scoped_pipeline_run.py`'s `terminal_state` docstring says
    that once Ray is up a `SIGTERM` reaches the driver as `SystemExit(15)`, recorded `killed`. That
    is false while the driver waits in `ray.wait`, which is where `main.py` spends a stage. There
    the handler's `SystemExit(15)` arrives as `SystemExit(1)`, so the run is recorded `failed`/1
    (probe: `docs/datastore-integrity-verification/sigterm_in_ray_wait_probe.py` catches
    `builtins.SystemExit: code=1`; that Ray's Cython signal check does the conversion is inferred
    from `_raylet.so`'s strings, the code is measured). And Ray's C++ handler, which prints `*** SIGTERM
    received ***` and a stack, aborts the process about 3 s later (measured; absl's default alarm is
    the inferred mechanism): a driver whose cleanup takes
    longer dies with status 134 (SIGABRT) before `run.finish` runs, so **no terminal state is
    written** and the run reads `stale` (probe: `sigterm_alarm_probe.py`, alive at 2.5 s, gone before
    3.0 s). The aborted driver's `gcs_server` is left running with parent 1, hung after its own
    `SIGTERM`, until it is killed with `SIGKILL`.
  - **Measured** (log 10). `resumed-a`: `SIGTERM` at 21:05:25, cleanup done in 2.2 s, recorded
    `failed`, 1. `resumed-b`: `SIGTERM` at 23:15:29.145 with 379 computes in flight, the driver
    failed at 23:15:32.164 in the GCS log, nothing recorded; its `gcs_server` (pid 6426) survived.
  - **Impact.** A deliberate interruption is never recorded as one. For A3 v2, whose `--build` uses
    the same `terminal_state` and the same registration, an interrupted launch is indistinguishable
    in the registry from a real failure, or reads as a crashed run, and may leave a `gcs_server`
    behind. Nothing in the store is affected: both interrupted stores closed with no journal and no
    in-flight record.
  - **Next step.** Record the signal in the driver's own handler before Ray replaces it (or
    re-install one after `ray.init`), so that `terminal_state` can tell a received `SIGTERM` from a
    `SystemExit(1)`, and write the terminal state before Ray's abort; then shut Ray down. The driver
    belongs to `GkTk-remedial`'s verification, and prompt 10 changes no driver.
  - **Assigned (2026-10-02):** to `prompts/a3-v2-readiness` prompt **04**, which covers
    `quadsource_atol_sweep.py --build` as well as this driver, since both use `terminal_state`. 04 is
    held on that campaign's audit 00, which measures whether a handler installed after `ray.init`
    survives `ray.wait` and whether Ray's abort can be cancelled. Index row moved to
    `docs/OPEN_ISSUES.md` §1.17.
  - **Widened (2026-10-02):** by `a3-v2-readiness` audit 00 (`docs/a3-v2-readiness-audit.md` §S),
    with Ray 2.43.0 and committed probes (`docs/a3-v2-readiness/sigterm_probe.py`). Both inferred
    mechanisms are now measured as behaviour. Any `SystemExit` a Python `SIGTERM` handler raises
    inside `ray.wait` or `ray.get` reaches the frame as `SystemExit(1)`; a handler raising
    `SystemExit(143)` delivers code 1. That Ray's Cython signal check does the conversion is still
    inferred from strings. The abort is a 3 s `SIGALRM` armed by absl's `AbslFailureSignalHandler`
    in `_raylet.so`: after the signal `signal.alarm(0)` returns 3, and the `SIGALRM` handler is
    `ImmediateAbortSignalHandler`. Death by `SIGABRT` came 2.78–3.06 s after the signal, in 6 runs
    of 6, and `signal.alarm(0)` cancels it. A Python handler installed **after** `ray.init`
    replaces absl's: it runs at once, and no banner and no alarm follow. **Wider than recorded:**
    the `gcs_server` is left behind by any exit that skips `ray.shutdown()` (7 of 12 runs: the
    abort, and `os._exit`), not by the abort as such; 0 of 10 exits through `ray.shutdown()` left
    one. The fingerprint `finish(fingerprint=True)` takes was 2.48 s on the A3 baseline.
  - **Prompt written (2026-10-02):** `prompts/a3-v2-readiness/04-record-a-sigterm-as-killed.md`, a
    handler installed after `ray.init`, against that campaign's audit S1–S4.
  - **Closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 04, this commit.
    - **The handler, after Ray.** Both drivers (`scoped_pipeline_run.main`,
      `quadsource_atol_sweep.run_build`) start Ray with `address="local"`. Straight after it
      returns they call `scoped_pipeline_run.install_killed_handler(run, latch)`, registered or
      not. Its handler, for `SIGTERM` and `SIGINT`:
      - remembers the signal in a `SignalLatch`;
      - writes `killed`, `128 + signum`, with no fingerprint;
      - prints one line of its own;
      - raises `SystemExit(128 + signum)`.

      It replaces absl's handler, so no banner is printed and no alarm is armed.
    - **After the `exec`.** Each driver's one `except BaseException` calls `finish_after_exec`. When
      the latch holds a signal, whatever exception reached the driver, it:
      - calls `signal.alarm(0)`;
      - writes `killed`, `128 + signum`, with the fingerprint;
      - calls `ray.shutdown()`;
      - exits `128 + signum`.

      Otherwise it records the outcome exactly as before, then `ray.shutdown()`, and the driver
      re-raises.
    - **`terminal_state`**'s docstring is corrected against audit S1–S3; its mapping is
      unchanged.
    - **Shown without Ray** in `RunRegistry/tests/test_killed_handler.py` (26 tests):
      - `TestTheHandler.test_its_first_call_writes_killed_143_remembers_prints_and_raises`;
      - `TestSignalled.test_it_is_killed_143_with_the_fingerprint_whatever_reaches_the_driver`
        (`SystemExit(1)`, `KeyboardInterrupt`, `RuntimeError` and four more: never `failed`);
      - `TestNotSignalled.test_every_outcome_is_recorded_as_it_was`;
      - `TestTheDrivers`.
    - **Shown on real Ray** in `docs/a3-v2-readiness/sigterm_handler_probe.txt`: three
      invocations, each of the plain, slow-fingerprint and second-`SIGTERM` variants run three
      times, 27 runs in all. Every run recorded `killed`/143 with the fingerprint present, exited
      143 with no abort and no Ray banner, and left no `gcs_server`.
    - **What it leaves.** `killed` was first written 0.001–0.007 s after the signal in 23 runs, and
      1.003–1.006 s in 4, because Ray checks for signals once a second inside a wait. That is
      `a3-v2-readiness`'s `[04-a-sigterm-in-ray-wait-is-handled-up-to-a-second-late]`, beside its
      three other issues from prompt 04.
    - Log: [`../a3-v2-readiness/logs/04-record-a-sigterm-as-killed.md`](../a3-v2-readiness/logs/04-record-a-sigterm-as-killed.md).
- **[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]** *(opened
  2026-09-29 by prompt 08; **closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 03)*
  - **The defect.** `extract_QuadSourceIntegral_data.py` looks its two `GkSourcePolicy` rows up
    with `pool.object_get("GkSourcePolicy", ...)` (`run_pipeline`, the `GkSource_policy_1pt5,
    GkSource_policy_5pt0` pair). `object_get` on a replicated class **inserts a row when none
    matches**. And every such call goes through `ShardedPool._replicated_write`, which commits a
    `replication_in_flight` record on the primary *before* the controlling shard's call, so **even
    a hit writes, and clears, a record on the primary**. `extract_GkSource_data.py` makes the same
    pair of calls; no other script was audited for it.
  - **Impact.** A reader that writes to the store it reads. Against a store that lacks the policy
    it silently adds a row; a crash inside the call leaves the record set, which prompt 02's check
    at open then acts on (repair, or refuse). Prompt 08 needed its own new lookup to refuse
    without this, and gave it `read_table` for that reason (log 08, §0.1).
  - **Next step.** Look the two `GkSourcePolicy` rows up with a `read_table` on that class (a
    `read_table_config` entry and a factory method, as `QuadSourcePolicy` now has), refusing on a
    miss, in this script and in any other reader that does the same. Prompt 08 was bounded by the
    user's grant to the `QuadSourcePolicy` lookup and did not change the `GkSourcePolicy` one.
  - **Assigned (2026-10-02):** to `prompts/a3-v2-readiness` prompt **03**, which opens every reader
    on a read-only pool (that campaign's D3) rather than repairing lookups one by one. Read the same
    day, the defect is wider than this entry: every `object_get` on a replicated class takes
    `_replicated_write`, and the readers also look up `tolerance`, the wavenumbers, the exit times
    and both cosmologies that way. That campaign's audit 00 measures how wide, and records it here.
    Index row moved to `docs/OPEN_ISSUES.md` §1.17.
  - **Widened (2026-10-02):** by `a3-v2-readiness` audit 00 (`docs/a3-v2-readiness-audit.md` R1–R2).
    Every reader `object_get`s seven replicated classes — `store_tag`, `LambdaCDM`, `QCD_Cosmology`,
    `tolerance`, `BackgroundModel`, `wavenumber_exit_time`, `GkSourcePolicy` — and every such call
    commits and clears an in-flight record on the primary, hit or miss. Measured on stand-in shards
    (`docs/a3-v2-readiness/reader_writes_probe.py`) for `extract_QuadSourceIntegral_data.py`'s whole
    sequence on a store holding every row it asks for: 21 replicated gets, 42 commits on the
    primary, no row inserted and no commit on any shard. With one row missing, `store_tag`'s lookup
    is skipped, while `LambdaCDM`, `QCD_Cosmology`, `tolerance` and `GkSourcePolicy` are inserted on
    every shard. No reader writes a sharded row. The wavenumbers, redshifts and `QuadSourcePolicy`
    are read with `read_table`, which writes nothing.
  - **Prompt written (2026-10-02):** `prompts/a3-v2-readiness/03-open-readers-read-only.md`, a
    read-only pool to the specification of that campaign's audit M1.
  - **Closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 03, `20977cb`, by that campaign's
    D3: a read-only pool, not per-site lookups. `ShardedPool(read_only=True)` opens every file
    `mode=ro` (the pool's engine and every actor's, through `store_reader.read_only_url`) and
    writes nothing at open: no store is created, journals are refused by presence through
    `store_reader.open_read_only`, a `replication_in_flight` record of any operation is refused by
    name, nothing is created, dropped, pruned or repaired, and the version row is found, never
    written. A replicated `object_get` goes to one shard's actor with no record and no
    replication, so `_replicated_write` is never entered. A lookup that would insert raises
    `ReadOnlyMiss`, from the inserter every read-only actor hands every factory's `build`; any
    other write raises `ReadOnlyWrite`, and a write that still reaches SQLite is re-raised as
    `ReadOnlyWrite`. All six readers open the pool read-only. Shown on stand-in shards in
    `Datastore/tests/test_read_only_pool.py`:
    `TestNothingWrittenOnAFullStore.test_the_instrument_counts_a_read_write_pool_then_read_only_writes_nothing`
    (QSI's whole sequence: on a read-write pool the primary changes and `_replicated_write` runs
    21 times; on a read-only pool every file's SHA-256 is unchanged, `_replicated_write` never
    runs, no engine issues a write, and every step returns what the read-write run did);
    `TestEachMissRaisesReadOnlyMiss` (`store_tag`, `LambdaCDM`, `QCD_Cosmology`, `tolerance`,
    `GkSourcePolicy`, each with no `INSERT` issued and no file changed);
    `TestOtherWritesRaiseReadOnlyWrite`; `TestAtOpen`. The readers' wiring is pinned in
    `ComputeTargets/tests/test_reader_read_only.py`
    (`TestTheSixReaders.test_each_opens_its_pool_read_only`). Log:
    [`../a3-v2-readiness/logs/03-open-readers-read-only.md`](../a3-v2-readiness/logs/03-open-readers-read-only.md).
- **[10-extraction-scripts-exit-1-on-a-store-without-every-model]** *(opened 2026-10-02 by prompt 10;
  **closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 03)*
  - **The defect.** The six `extract_*.py` scripts loop over `config.model_list.build_model_list`,
    LambdaCDM and `QCD_Cosmology`, with no `--models` option, and raise `RuntimeError: Could not
    locate suitable background model instance in the datastore` for a model whose background the
    store does not hold.
  - **Measured** (log 10). On the LambdaCDM-only resumed store each read LambdaCDM in full, wrote
    its `LambdaCDM/` outputs, and exited 1 on `QCD_Cosmology`, in 15–67 s; none moved the
    fingerprint.
  - **Impact.** A3 v2's store is LambdaCDM-only, so every extraction from it exits 1 after a
    complete LambdaCDM read. The outputs are written, but a caller that checks the exit status sees
    a failure, and a failure in LambdaCDM's read looks the same from outside. The user accepted the
    exit 1 for prompt 10 (2026-10-02).
  - **Next step.** Give the scripts a model filter, as `scoped_pipeline_run.py --models` does for
    `main.py`, or skip a model the store does not hold, saying so. The scripts are not this
    prompt's to change.
  - **Assigned (2026-10-02):** to `prompts/a3-v2-readiness` prompt **03**, beside `[08-…]`: both are
    the readers' setup, and `build_model_list` is where each starts. Index row moved to
    `docs/OPEN_ISSUES.md` §1.17.
  - **Prompt written (2026-10-02):** `prompts/a3-v2-readiness/03-open-readers-read-only.md`, O4:
    `--models`, and a model with no `BackgroundModel` skipped with a message.
  - **Closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 03, `20977cb`. Each of the six
    readers takes `--models LABEL [LABEL …]` (`extract_common.add_models_argument`), applied to
    `build_model_list`'s result by `extract_common.select_models`, which refuses any label
    `build_model_list` does not return, naming those it does. A model whose `BackgroundModel` the
    store does not hold is skipped with one line (`extract_common.require_background` at the old
    `RuntimeError`'s site, `extract_common.read_models` in place of the model loop), whether or
    not `--models` was given, and a reader that reads no model exits 1 saying which were skipped
    and why. `config/model_list.py` is unchanged: every store `main.py` builds holds both
    cosmology rows. Pinned in `ComputeTargets/tests/test_reader_read_only.py`:
    `TestTheSixReaders` (each script, read with `ast`), `TestSelectModels`, and `TestTheSkip`
    (`test_a_model_with_no_background_is_skipped_with_its_line`,
    `test_all_skipped_exits_non_zero_with_the_reason`). Log:
    [`../a3-v2-readiness/logs/03-open-readers-read-only.md`](../a3-v2-readiness/logs/03-open-readers-read-only.md).
- **[02-the-version-row-is-written-outside-the-in-flight-record]** *(opened 2026-09-28 by prompt 02;
  **closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 02)*
  - **The defect.** The `version` table is replicated, but its row is not written through
    `ShardedPool._replicated_write`. `ShardedPool.__init__` builds the first `Datastore` actor,
    whose constructor inserts the version label's row, and then each later actor, whose
    constructor inserts it under that serial (`version_serial`). Each insert is in its own actor's
    constructor, and nothing records the sequence.
  - **Measured** (log 02, observation 1, on stand-in shards). A store opened under label `L1`,
    then under `L2`, with every actor after the first killed, holds `L2` on one shard only. The
    next open under `L2` is refused by prompt 02's check: `class "version", shard [0, 1], serial 2:
    held by shard(s) [2], missing on shard(s) [0, 1]`. At `65ee1d3` it would have healed, as the
    later actors insert the row under its serial.
  - **Impact.** Only an open under a label the store has never seen writes these rows.
    `VERSION_LABEL` is a constant (`"2025.1.1"`) in `main.py` and the six extraction scripts. So
    the window is a new store's first open, or the first open after the label changes. A crash
    inside it now leaves a store that refuses to open and needs a person.
  - **Next step.** Decide whether the version rows go through the recorded write, or whether the
    check at open treats a `version` row missing under a label being opened differently. Either
    changes `Datastore.py`'s constructor or the check's licence, which prompt 02 could not.
  - **Assigned (2026-10-02):** to `prompts/a3-v2-readiness` prompt **02**, by the user's decision
    that every defect a build or a reader of A3 v2 would meet is fixed before it is built (that
    campaign's README §6.1, D4). 02 is held on that campaign's audit 00, which measures the window
    and whether a read-only open can find the row. Index row moved to `docs/OPEN_ISSUES.md` §1.17.
  - **Prompt written (2026-10-02):** `prompts/a3-v2-readiness/02-write-the-version-row-through-the-recorded-write.md`,
    form A of that campaign's audit V2. Under the user's U1 it also moves the replicated prune at
    open into the pool, under a record.
  - **Closed 2026-10-03** by `prompts/a3-v2-readiness` prompt 02, `b647c0b` (*"Write the
    version row and the replicated prune through a record"*), form A of that campaign's audit V2.
    `Datastore.__init__` no longer looks up or inserts the version row. `ShardedPool.__init__`,
    once every actor exists, reads the label's row `mode=ro` from every shard. Only when no shard
    holds it does it write the row once, through `_get_impl_replicated_table` and so
    `_replicated_write`, under a record. It then calls `set_version` on every actor, and until then
    `_insert` refuses any versioned insert. An interrupted version write leaves a `get`/`version`
    record, which the check at open's generic `get` repair completes unchanged. Healed, on stand-in
    shards, in `Datastore/tests/test_version_row_at_open.py`: `TestNewLabelInterrupted` (after the
    controller commits, before it, and on one replica: the case measured above) and
    `TestNewStoreInterrupted`. The audit probe's case 3 now reopens with `version serial 2` on every
    shard. A clean reopen still writes nothing (`test_reconcile_at_open.TestCleanStoreUntouched`,
    unchanged). Log: [`../a3-v2-readiness/logs/02-write-the-version-row-through-the-recorded-write.md`](../a3-v2-readiness/logs/02-write-the-version-row-through-the-recorded-write.md).
- **[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]** *(opened 2026-10-01 by
  prompt 09d)*
  - **The defect.** Since 09d a record line of a class that another class names ends, beyond its
    fields, in its own label `  #<n>`, which is different for every record. So
    `test_inventory_report.test_verbose_renders_every_distinct_record_as_a_distinct_line`, which
    compares whole lines (`len(set(section.records)) == len(set(cls.records))`), can no longer fail
    for those seven classes whatever fields the line shows. `test_verbose_shows_every_key_field_of_every_class`
    still checks that every key field is printed, so a dropped field is still caught, but the test
    that states the property is vacuous for them.
  - **Impact.** None today: 09d's `test_inventory_report_labels.py` pins the property with the label
    taken off (`TestDistinctStaysDistinct`), and mutation (i) of log 09d shows it failing. A later
    change to the report that relies on the old test alone would not be checked.
  - **Next step.** Compare lines with the own label stripped (`fields_of` already stops at the first
    double space), in whichever prompt next edits `test_inventory_report.py`. Prompt 09d could not:
    README §5 rule 7 keeps existing tests as they are.
  - **Assigned (2026-10-01):** to prompt **10**, by the user's decision on the review of 09d. README
    §2's row for 10 names the test and the reason, as rule 7 requires, so that prompt may amend it.
  - **Closed 2026-10-02**, by prompt 10, in the commit *"Build a throwaway store end to end,
    interrupt it and read it back"*. The test now compares `(tag set, line)` pairs with the record's
    own label removed by 09d's `OWN_LABEL` (one import, nothing else in the test changed). A mutation
    printing a labelled class's record without its fields or value count fails it for four classes
    and passes the parent commit's test (log 10). On this fixture it cannot fail for three classes,
    which is `[10-distinct-lines-test-cannot-fail-for-three-labelled-classes-on-the-fixture]` (§3.3).
- **[09b-inventory-report-renders-every-member-of-a-parent-set]** *(opened 2026-09-30 by prompt
  09b)*
  - **The defect.** By the user's decision on the sketch, `tools/inventory_report.py` renders a
    `GkSource`'s parent set by its members' physical labels, and it renders **every** member, on
    the record's line. Measured on `build_full_store`: about 200 characters per member, verbose or
    not. A production `GkSource` has a member for each `z_source` above its `z_response`, up to
    ~1,200 at 100 samples per decade, so one record line is up to ~240 kB; every
    `GkSourcePolicyData` line repeats its source's set, since the set is a field that varies.
  - **Impact.** `main.py --inventory` on a production store prints lines no one can read, and
    with `verbose` a report of order 1 GB (5,000 sources, and as many policy rows). Nothing is
    lost or wrong; `test_inventory_report.py` passes, and no digest is printed.
  - **Next step.** Decide how the report summarises a set (its size, the `z_source` range, the
    distinct numeric and WKB parents' labels), which changes what the user decided and so is the
    user's.
  - **Widened (2026-10-01, the review of 09c):** since prompt 09c a `QuadSourceIntegral`'s key
    names its `GkSourcePolicyData`, and the report renders a parent reference recursively, so
    every integral's line carries its source's whole parent set as well (log 09c, observation
    1). Measured on `build_full_store`: about 4,300 characters an integral line, from about 450
    before 09c. In production each of ~500,000 integrals would repeat its `GkSource`'s up to
    ~1,200 members, so the report is not usable on such a store until the set is summarised.
  - **Assigned (2026-10-01):** to prompt **09d**, a small prompt of its own, by the user's
    decision after the review of 09c. A summary in place of the members would let two distinct
    records print as one line, which the report's tests forbid
    (`test_verbose_renders_every_distinct_record_as_a_distinct_line`). So 09d prints each
    distinct set once, under a label, and every record refers to it by the label. The user widened
    09d the same day to every reference to a derived (sharded) row: since 09c those references
    make up ~3,500 of a `QuadSourceIntegral` line's ~4,300 characters on the fixture, beyond the
    set, so each is rendered as a label too, with the row printed once in its own section.
  - **Closed 2026-10-01**, by prompt 09d, in the commit *"Refer to derived parents by label in the
    inventory report"*. `tools/inventory_report.py` prints a reference to a sharded row as
    `<Class>#<n>` (its ordinal in its class's verbose listing) and a parent set as `S<n>`, each
    defined once in its own class's section; without `verbose`, a row or set that a shown record
    names and the report did not show is listed too. On `build_full_store`, a verbose
    `QuadSourceIntegral` line goes from 4,316-4,462 to 564-565 characters and a `GkSource` line from
    533-550 to 169-170; on a stand-in with a 1,703-member set, the record lines of `GkSource`,
    `GkSourcePolicyData` and `QuadSourceIntegral` are 27, 21 and 25 characters at any set size (each
    was about 878,000) and a set's line grows by about 56 characters a member (log 09d).
    The report is still one string. Its tests (`test_inventory_report_labels.py`) keep the property
    this entry named: two distinct records never print as one line, here including pairs that differ
    only in one member of a set.
- **[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]** *(opened 2026-09-30 by
  prompt 09a)*
  - **The defect.** `QuadSourceValue._build_impl_model` finds its parent `QuadSource` by model, `q`,
    `r`, `validated` and tags, and joins the value table to **every** match. Before 09a a second
    `QuadSource` under the same labels could not be stored; now two can, one per `(Tq_serial,
    Tr_serial)`. Measured (stand-in store, two validated sources under the same labels each with
    a value at one `z`): the read raises `MultipleResultsFound`, naming no parent.
  - **Impact.** It refuses rather than serving a wrong value. No caller in the tree
    (`main.py`, the extraction scripts) uses it; only `probe_05` does. It is D1's labels-only
    reader shape without the refusal naming the parents.
  - **Next step.** Key it on `Tq_serial` and `Tr_serial`, or give it the reader mode, in the
    prompt that next edits `QuadSourceValue`'s reads. Outside 09a's list of files.
  - **Assigned (2026-09-30):** to prompt **09c**, by the user's decision on the review of 09a
    ([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md) §0.7, decision 3). 09c closes D1's
    class on this board, so it gives this read D1's reader treatment: it refuses more than one
    matching `QuadSource`, naming their `Tq_serial` and `Tr_serial`, and serves one. 09b does the
    same for `GkSourceValue._build_impl_model`, which the sketch of 09b found.
  - **Closed 2026-09-30** by prompt 09c, in the commit *"Key QuadSourceIntegral on the six rows it
    is computed from"*. `QuadSourceValue._build_impl_model` now selects its candidate
    `QuadSource`s' `serial`, `Tq_serial` and `Tr_serial`, and refuses more than one with
    `MultipleResultsFound` naming each serial and pair (`serial=…: Tq_serial=1, Tr_serial=2; …`),
    whether or not each holds a value at `z`; one match is served as before, and none is a miss
    (`test_quadsource_integral_parent_key.TestTheQuadSourceValueModelRead`, 4 tests). Its candidate
    row is the one new `EXPECTED_COVERAGE` entry of the read-back guard (50 → 51).
- **[07-wkb-value-reads-do-not-refuse-a-pre-07-store]** *(opened 2026-09-29 by prompt 07)*
  - **The defect.** Prompt 07's D4 refusal is in the two WKB factories' `build()`. The four WKB
    value reads (`GkWKBValue`/`TkWKBValue` `_build_impl_model` and `read_batch`) name
    `numeric_serial` only when a parent is given. Measured on a `GkWKBIntegration` table recreated
    from the 06 witness's DDL, holding one pure-WKB row: a parentless `GkWKBValue.read_batch`
    returned its 2 values; a keyed one raised a raw `OperationalError` (`no such column:
    GkWKBIntegration.numeric_serial`), not the D4 message (log 07, observations).
  - **Impact.** `main.py`'s GkSource stage reads `GkWKBValue` with no parent, so on a pre-07 store
    a run of that stage alone would read WKB values the tree otherwise refuses, rather than being
    told the store predates the key. D4 says such a store is read at its own SHA; nothing here
    reads one today (no store under `var/` is opened by this tree's pipeline).
  - **Next step.** Decide whether the value reads should refuse a store without the column too
    (e.g. by always selecting it, and matching the driver's message as `build()` does), or whether
    D4's refusal is `build()`'s alone. Prompt 09 keys `GkSource` on its Gk rows and is a natural
    owner.
  - **Assigned (2026-09-30):** to prompt **09b**, by the user's decision
    ([`orchestrator/prompt-09.md`](orchestrator/prompt-09.md) §0.3). 09b already edits
    `GkNumericValue.read_batch` and `GkWKBValue.read_batch` so that they hand back their parent,
    and it closes all four reads here: `GkWKBValue`'s and `TkWKBValue`'s `_build_impl_model` and
    `read_batch`.
  - **Closed 2026-09-30** by prompt 09b, in the commit *"Key GkSource on the set of Gk rows it is
    assembled from"*. All four reads, `GkWKBValue`'s and `TkWKBValue`'s `_build_impl_model` and
    `read_batch`, now always select `numeric_serial` in their parent query, keyed or not, and
    match the driver's message as `build()` does: on a table from the 06 witness's DDL each
    raises 07's D4 `RuntimeError` (no `numeric_serial` column; readable at the SHA its sidecar
    records), parentless, keyed, pure-WKB-keyed and, for `GkWKBValue.read_batch`, in its new
    serial-only form (`test_gksource_parent_set.TestTheWKBValueReadsRefuseAPre07Store`, 11
    sub-tests; a control reads the current tables). Mutation (v) of log 09b, which drops the
    column from the parentless `GkWKBValue.read_batch`, is caught there. The GkSource stage's
    parentless read of a pre-07 store is therefore refused, not served.
- **[03-exit-time-build-comment-describes-the-removed-order-by]** *(opened 2026-09-29 by the
  orchestrator's review of prompt 03)*
  - **The defect.** In `sqla_wavenumber_exit_time_factory.build` (`wavenumber.py:195`), the
    comment above the tolerance aliases reads "order by descending values of abs and relative
    tolerances, so that we get the best computed value we hold". Prompt 03 removed that
    `order_by`, and with it the "best value we hold" behaviour: the lookup is now an exact match.
    The comment was left in place.
  - **Impact.** None on behaviour. But a reader is told that the lookup prefers a tighter row,
    which is precisely the defect L1 fixed, and the comment sits in the function that says so.
  - **Next step.** Delete or rewrite that one line, in whichever prompt next has that `build()`
    in scope. It was not 03's to fix after its commit (one commit per prompt, no amend).
  - **Closed 2026-09-29**, at the user's request, in the commit *"Correct the exit-time build
    comment the ORDER BY's removal left false"*. The line now says the tolerance table is joined
    twice only to return each tolerance's `log10_tol`, and that the match is exact on the two
    serials. It is a comment only: no statement changed, and the read-back guard and `black` are
    unaffected.

---

## 5. Baselines

README §7. Re-measured at every dispatch.

### 5.1 After prompt 01 (2026-09-26)

Measured on the tree of prompt 01's commit, with README §7's command. Before it, at `65cd4b7`: the
same, with Datastore at 206.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK (the known wall-clock flake did not fire) |
| CosmologyModels | 39 OK |
| Datastore | **232 OK** (206 + the 26 of `test_replicated_write.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint unchanged |

### 5.2 After prompt 04 (2026-09-26)

Measured on the tree of prompt 04's commit, with README §7's command, serially (ComputeTargets run
alone, to avoid the known wall-clock flake under parallel load).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **235 OK** (232 + the 3 of the new `TestColumnExistence`; G1 added no new test methods) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check Datastore/tests/test_factory_select_columns.py` is clean.

### 5.3 After prompt 04b (2026-09-28)

Measured on the tree of prompt 04b's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **237 OK** (235 + the 2 of the new `test_unresolved_table_sites_are_the_declared_ones` / `test_unchecked_insert_sites_are_the_declared_ones`; G4/G5 added no new test methods) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check Datastore/tests/test_factory_select_columns.py` is clean.

### 5.4 After prompt 05a (2026-09-28)

Measured on the tree of prompt 05a's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **257 OK** (237 + the 20 of the new `test_read_path_mechanics.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check` is clean on every file this prompt touched.

### 5.5 After prompt 05b (2026-09-28)

Measured on the tree of prompt 05b's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **267 OK** (257 + the 10 of the new `test_integral_read_batch.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check` is clean on every file this prompt touched.

### 5.6 After prompt 05c (2026-09-28)

Measured on the tree of prompt 05c's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **287 OK** (267 + the 20 of the new `test_value_reads_parent_key.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check` is clean on every Python file this prompt touched. No guard pin moved.

### 5.7 After prompt 02 (2026-09-28)

Measured on the tree of prompt 02's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **325 OK** (287 + the 38 of the new `test_reconcile_at_open.py`; the one rewritten test in `test_replicated_write.py` is not an addition) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check` is clean on every Python file this prompt touched.

### 5.8 After prompt 03 (2026-09-29)

Measured on the tree of prompt 03's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load; it did not fire).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **343 OK** (325 + the 18 of the new `test_exact_identity_lookups.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

`black --check` is clean on every Python file this prompt touched.

### 5.9 After prompt 06 (2026-09-29)

Measured on the tree of prompt 06's commit, with README §7's command, serially (ComputeTargets
run alone, to avoid the known wall-clock flake under parallel load; it did not fire).

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 552 OK |
| CosmologyModels | 39 OK |
| Datastore | **351 OK** (343 + the 8 of the new `test_foreign_key_check.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint unchanged |

`black --check` is clean on every Python file this prompt touched.

### 5.10 After prompt 07 (2026-09-29)

Measured on the final tree of prompt 07's commit, with README §7's command, serially.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | **566 OK** (552 + the 14 of the new `test_wkb_numeric_parent_main.py`; the known wall-clock flake did not fire) |
| CosmologyModels | 39 OK |
| Datastore | **397 OK** (351 + the 41 of the new `test_wkb_numeric_parent_key.py` and the 5 of the new `test_wkb_parent_foreign_key.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint re-captured as `full_store_fingerprint_datastore-integrity-07.json`, format 2 |

`black --check` is clean on every Python file this prompt touched.

### 5.11 After prompt 08 (2026-09-29)

Measured on the final tree of prompt 08's commit, with README §7's command, serially.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | **585 OK** (566 + the 19 of the new `test_quadsource_policy_main.py`; the known wall-clock flake did not fire in the final run) |
| CosmologyModels | 39 OK |
| Datastore | **422 OK** (397 + the 21 of the new `test_quadsource_policy_key.py` and the 4 of the new `test_quadsource_policy_foreign_key.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint re-captured as `full_store_fingerprint_datastore-integrity-08.json`, format 3 (`QuadSourceIntegral` and `QuadSourcePolicy` moved; the other 19 classes did not) |

The read-back guard's pins did not move: `EXPECTED_COVERAGE` 47, `KNOWN_UNRESOLVED_TABLES` 11,
`KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4, `KNOWN_UNFIXED` and
`KNOWN_UNFIXED_COLUMNS` empty. The witness is `schema_at_datastore-integrity-08.json`.

`black --check` is clean on every Python file this prompt touched.

### 5.12 After prompt 09a (2026-09-30)

Measured on the final staged tree of prompt 09a's commit, with README §7's command, serially.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | **610** (585 + the 25 of the new `test_quadsource_parent_main.py`); OK apart from the known wall-clock flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, which fails at `HEAD` too under this load (average 12-29); see log |
| CosmologyModels | 39 OK |
| Datastore | **459 OK** (422 + the 37 of the new `test_quadsource_parent_key.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint re-captured as `full_store_fingerprint_datastore-integrity-09a.json`, format 4 (only `QuadSource` moved) |

The read-back guard's pins did not move: `EXPECTED_COVERAGE` 47, `KNOWN_UNRESOLVED_TABLES` 11,
`KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4, `KNOWN_UNFIXED` and
`KNOWN_UNFIXED_COLUMNS` empty. `test_main_plumbing.EXPECTED_SITES`: `TkNumericIntegration` 5 → 6,
the `QuadSource` stage's serial-only parent lookup; every other count unchanged. The witness is
`schema_at_datastore-integrity-09a.json`.

`black --check` is clean on every Python file this prompt touched.

### 5.13 After prompt 09b (2026-09-30)

Measured on the final staged tree of prompt 09b's commit, with README §7's command, serially.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | **635 OK** (610 + the 25 of the new `test_gksource_parent_set_main.py`; the known wall-clock flake did not fire in the final run) |
| CosmologyModels | 39 OK |
| Datastore | **505 OK** (459 + the 46 of the new `test_gksource_parent_set.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint re-captured as `full_store_fingerprint_datastore-integrity-09b.json`, format 5 (`GkSource` and `GkSourcePolicyData` moved; the other 19 classes did not) |

The read-back guard's pins: `EXPECTED_COVERAGE` **47 → 50**, the three row-read sites 09b adds
(`sqla_GkSource_factory.build.candidate`, `sqla_GkSourceValue_factory._build_impl_model.candidate`,
`<module-level>._read_parent_sets.member`, all in `GkSource.py`); `KNOWN_UNRESOLVED_TABLES` 11,
`KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4, `KNOWN_UNFIXED` and
`KNOWN_UNFIXED_COLUMNS` empty, unchanged. `test_main_plumbing.EXPECTED_SITES` unchanged
(`GkSource` 6; `TkNumericIntegration` 6). The witness is `schema_at_datastore-integrity-09b.json`.

`black --check` is clean on every Python file this prompt touched.

### 5.14 After prompt 09c (2026-09-30)

Measured on the final staged tree of prompt 09c's commit, with README §7's command, serially.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | **665 OK** (635 + the 30 of the new `test_quadsource_integral_parent_main.py`; the known wall-clock flake did not fire in the final run) |
| CosmologyModels | 39 OK |
| Datastore | **532 OK** (505 + the 27 of the new `test_quadsource_integral_parent_key.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK; the golden fingerprint re-captured as `full_store_fingerprint_datastore-integrity-09c.json`, format 6 (only `QuadSourceIntegral` moved; the other 20 classes did not) |

The read-back guard's pins: `EXPECTED_COVERAGE` **50 → 51**, the one row-read site 09c adds
(`QuadSource.py:sqla_QuadSourceValue_factory._build_impl_model.candidate`); `KNOWN_UNRESOLVED_TABLES`
11, `KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13, `KNOWN_BLIND_COLUMNS` 4, `KNOWN_UNFIXED` and
`KNOWN_UNFIXED_COLUMNS` empty, unchanged. `test_main_plumbing.EXPECTED_SITES`: five counts +1, each
by one lookup of the new `resolve_QuadSourceIntegral_parents` (`TkNumericIntegration` 6 → 7,
`TkWKBIntegration` 3 → 4, `QuadSource` 3 → 4, `GkSource` 6 → 7, `GkSourcePolicyData` 4 → 5); every
other count unchanged. The witness is `schema_at_datastore-integrity-09c.json`.

`black --check` is clean on every Python file this prompt touched.

### 5.15 After prompt 09d (2026-10-01)

Measured on the staged tree of prompt 09d's commit, with README §7's command, serially. The
Datastore suite ran twice: the first run failed `test_inventory_retired`'s grep on a helper method
of the new module named `inventory`, which was renamed (log 09d, "Suites after this commit").

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 665 run, 1 failure: the known wall-clock flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (0.0804 s against 0.06 s under a load average of 50–120 from other work; the same test fails alone, 0.0805 s, in an export of `HEAD`); the count is unchanged |
| CosmologyModels | 39 OK |
| Datastore | **551 OK** (532 + the 19 of the new `test_inventory_report_labels.py`) |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

No witness, golden or read-back-guard pin moved: the prompt changes one display module, and
`schema_at_datastore-integrity-09c.json` and `full_store_fingerprint_datastore-integrity-09c.json`
are the current ones. `test_main_plumbing.EXPECTED_SITES` is unchanged. `black --check` is clean on
every Python file this prompt touched. `docs/OPEN_ISSUES.md` stays at 99 rows: one deleted, one
opened.

### 5.16 After prompt 10 (2026-10-02)

Measured on the staged tree of prompt 10's commit, with README §7's command, serially, at a load
average of 10–18.

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 665 OK (the known wall-clock flake did not fire) |
| CosmologyModels | 39 OK |
| Datastore | 551 OK, unchanged in count: `test_verbose_renders_every_distinct_record_as_a_distinct_line` amended, nothing added |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 199 OK |

No production code, witness, golden or read-back-guard pin moved; `test_main_plumbing.EXPECTED_SITES`
is unchanged, and `substitute_main_py()` matches all five literals once. `black --check` is clean
on every Python file this prompt adds or changes. `docs/OPEN_ISSUES.md` goes from 99 to 102 rows:
one deleted, four opened.
