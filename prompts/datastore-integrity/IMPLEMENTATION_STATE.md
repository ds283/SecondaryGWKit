# Datastore integrity campaign — implementation state

**Last updated:** 2026-09-28 · **Status: 3 of 10 prompts landed (01, 04, 04b), all reviewed.** The
user decided D2–D6 on 2026-09-26 (README §6.2), and D1 (yes) later the same day. 04b landed the
same day it was written, from the 04 review's findings. **05 is written** (2026-09-28) as three prompts,
05a → 05b → 05c, for one orchestrator session ([`orchestrator/prompt-05.md`](orchestrator/prompt-05.md)); 05a is ready.
02 and 03 are ready too, in a separate chain. 10 waits on 01–09. `store-retirement`
closed on 2026-09-26 at 6 / 6, which discharges both cross-campaign constraints (README §2).

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
| 02 | [Repair or refuse at open](02-repair-or-refuse-at-open.md) | **W5**–**W7** | Opus | ✍️ yes, 2026-09-25 | ⬜ after 01 | — | — |
| 03 | [Exact identity lookups](03-exact-identity-lookups.md) | **L1**–**L5** | Sonnet | ✍️ yes, 2026-09-25 | ⬜ after 02 | — | — |
| 04 | [Extend the read-back guard](04-extend-the-read-back-guard.md) | **G1**–**G3** | Sonnet | ✍️ yes, 2026-09-25 | ✅ 2026-09-26 | *"Extend the read-back guard to read_batch and column existence"* | [`logs/04-…`](logs/04-extend-the-read-back-guard.md) |
| 04b | [Tighten the read-back guard](04b-tighten-the-read-back-guard.md) | **G4**–**G6** | Sonnet | ✍️ yes, 2026-09-28 | ✅ 2026-09-28 | *"Tighten the read-back guard to every function and every declared hole"* | [`logs/04b-…`](logs/04b-tighten-the-read-back-guard.md) |
| 05a | [Repair the mechanical read defects](05a-repair-the-mechanical-read-defects.md) | **P6**, **P7** | Sonnet | ✍️ yes, 2026-09-28 | ⬜ ready | — | — |
| 05b | [Repair the integral `read_batch`es](05b-repair-the-integral-read-batches.md) | **P1**, **P2**, **P8** | Sonnet | ✍️ yes, 2026-09-28 | ⬜ after 05a | — | — |
| 05c | [Key the value reads on their parent](05c-key-the-value-reads-on-their-parent.md) | **P3**–**P5**, **P9** | Opus | ✍️ yes, 2026-09-28 | ⬜ after 05b | — | — |
| 06 | [Correct the foreign keys](06-correct-the-foreign-keys.md) | **S1**–**S5** | Sonnet | ✍️ yes, 2026-09-26 | ⬜ after 03 | — | — |
| 07 | [Key WKB rows on their numeric row](07-key-wkb-rows-on-their-numeric-row.md) | **K1**–**K2** | Opus | ✍️ yes, 2026-09-26 | ⬜ after 05 and 06 | — | — |
| 08 | [Key `QuadSourceIntegral` on its policy](08-key-quadsourceintegral-on-its-policy.md) | **K3** | Sonnet | ✍️ yes, 2026-09-26 | ⬜ after 07 | — | — |
| 09 | [Key derived rows on their parents](09-key-derived-rows-on-their-parents.md) | **K4**–**K7** | Opus | ✍️ yes, 2026-09-26 | ⬜ after 08 | — | — |
| 10 | Build a throwaway store end to end | **V1**–**V3** | Opus | ⏸️ held until 01–09 land | — | — | — |

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
and [`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) exist (2026-09-26). 01 and 04 are
dispatched one at a time, never concurrently in this checkout, in either order.

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
| W5 | **FACILITY** | `_reconcile_replicated_tables()` in the constructor, before any actor, no Ray: hot journals rolled back, every replicated table compared across shards by full row (timestamp exempt for old rows). | 02 | ⬜ |
| W6 | **REMEDY** | With an in-flight record set, the missing rows of its class copied forward from its controller, `BackgroundModel` as a unit with `validated` recomputed; the record then cleared. | 02 | ⬜ |
| W7 | **GUARD** | Any other difference refuses the open with `ReplicatedDivergence`, including a serial split and a row the controller lacks. Closes `[00-replicated-writes-can-diverge-across-shards]`. | 02 | ⬜ |
| L1 | **REMEDY** | `wavenumber_exit_time.build` by serial equality on atol, rtol and `==` on stepping; the dead `ORDER BY` removed; the query out of the loop. Closes `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` and `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]`. | 03 | ⬜ |
| L2 | **REMEDY** | `IntegrationSolver.build` by `stepping ==` and `one_or_none()`. Closes `[03-integrationsolver-stepping-minimum-lookup]`. | 03 | ⬜ |
| L3 | **GUARD** | `.scalar()` → `one_or_none()` on the eight identity lookups (audit §3.8), so a duplicate raises. | 03 | ⬜ |
| L4 | **REMEDY** | `redshift.build` matches `z = 0` (audit §3.9). | 03 | ⬜ |
| L5 | **RECORD** | `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]` narrowed: after L1 the two scripts miss rather than silently reuse. | 03 | ⬜ |
| G1 | **GUARD** | The read-back guard visits every function of every factory, `read_batch` and `_build_impl_*` included, linking a nested helper's parameter to its row. | 04 | ✅ `analyse_read_batch` / `inline_nested_row_helper`; `EXPECTED_COVERAGE` grows by 10 (probe_02.out's 16 collapse under the module's existing "module:class.variable" key — see log) |
| G2 | **GUARD** | A column-existence test: every `<table>.c.<name>` and every insert key resolves against the built schema. | 04 | ✅ `TestColumnExistence` / `column_existence_problems`, same module |
| G3 | **RECORD** | `KNOWN_UNFIXED`: what G1–G2 find today, for prompt 05 to empty. Closes `[01-read-batch-is-outside-the-guard]`. | 04 | ✅ `KNOWN_UNFIXED` +2 (both `WKB_criterio`, P6), `KNOWN_UNFIXED_COLUMNS` (new) +3 (`z_serial`/`model_id`, P6; `validated` insert key, P8) — the five the probes predicted, nothing new |
| G4 | **GUARD** | The analyser visits every function of every factory class and every module-level function, not a list of names; `validate_on_startup`'s seven bindings join coverage. | 04b | ✅ `iter_functions`; `_ROW_HELPERS` gone; `EXPECTED_COVERAGE` +13 (30→43: the 7 `validate_on_startup` bindings plus 6 that regain a sibling once the function rejoins the key) |
| G5 | **GUARD** | `EXPECTED_COVERAGE` and `KNOWN_BLIND` keyed by function, so each helper is pinned on its own; its projection keeps all 30 of 04's entries. | 04b | ✅ `Finding.coverage_key` ("module:class.function.variable"); `Finding.key` unchanged, so `KNOWN_UNFIXED`/`KNOWN_UNFIXED_COLUMNS` keep their keying and contents exactly; projection verified by computation in the log |
| G6 | **GUARD** | G2 walks module-level functions, reports an unknown `tables["X"]` as a problem, and pins every unresolved table base and unchecked insert with a reason. | 04b | ✅ `KNOWN_UNRESOLVED_TABLES` (12 pairs), `KNOWN_UNCHECKED_INSERTS` (13 sites), both declared with no resolution rule applied (log's deviation 1); `tables["X"]`/`inserters["X"]` naming no schema table now convicted (none in this tree today) |
| P1 | **REMEDY** | The tagged join in `QuadSourceIntegral` / `OneLoopIntegral` `read_batch` on `table.c.serial`. Closes `[00-tagged-read-batch-joins-an-unselected-alias]`. | 05b | ⬜ |
| P2 | **REMEDY** | `QuadSourceIntegral.read_batch` gives its exit times their own serials (audit §3.2). | 05b | ⬜ |
| P3 | **REMEDY** | `break_point_kind` in the Tk/Gk numeric value reads and `GkNumericValue.read_batch`, with atol and rtol required and more than one parent per `z_source` refused; one key name for `z_init`. Closes `[00-numeric-value-parent-lookup-omits-break-point-kind]`. | 05c | ⬜ |
| P4 | **REMEDY** | `z_init` in the `GkWKBValue` reads; more than one parent per `z_source` refused (audit §3.4). *(Widened 2026-09-28 when 05c was written: `TkWKBValue`'s reads have the same omission, refused per `(model, k)`.)* | 05c | ⬜ |
| P5 | **REMEDY** | The two `…_is_response` flags read `response` (audit §3.11). | 05c | ⬜ |
| P6 | **REMEDY** | `WKB_criterio`, `z_serial`, `model_id`, `friction` (audit §3.10–§3.11), under D2. | 05a | ⬜ |
| P7 | **REMEDY** | Tags default to `[]` at every value helper and `read_batch`; the three `raise ("…")`. | 05a | ⬜ |
| P8 | **REMEDY** | `OneLoopIntegral`: its own tag table, `read_batch` a method with a return value, as D3 bounds. Closes `[00-oneloop-lookup-joins-the-wrong-tag-table]`. | 05b | ⬜ |
| P9 | **GUARD** | Round-trip tests of every `read_batch`, tagged and untagged; `KNOWN_UNFIXED` empty. | 05a–05c | ⬜ |
| S1 | **REMEDY** | `QuadSource.Tq_serial` → `TkNumericIntegration.serial`; the `Tr_serial` comment. Closes `[00-quadsource-tq-serial-has-the-wrong-foreign-key]`. | 06 | ⏸️ |
| S2 | **REMEDY** | `GkSourcePolicyData.wavenumber_exit_serial` gains its foreign key. | 06 | ⏸️ |
| S3 | **REMEDY** | `real_store_fixtures.py`'s undeclared-reference map corrected. | 06 | ⏸️ |
| S4 | **GUARD** | A test that runs `PRAGMA foreign_key_check` over every shard of a store written by the real factories and finds no violation. | 06 | ⏸️ |
| S5 | **RECORD** | The first deliberate schema-witness change, under D5, including the exit-time and `IntegrationSolver` `stepping_mode` flags made exact. | 06 | ⏸️ |
| K1 | **REMEDY** | `numeric_serial` on both WKB tables, in the lookup; `z_init` and the initial values leave the key; `main.py` passes the parent. Closes `[20-wkb-rows-consume-numeric-initial-data]`. | 07 | ⏸️ |
| K2 | **REMEDY** | `_drop_order` respects the new key. | 07 | ⏸️ |
| K3 | **REMEDY** | `quad_policy_serial` on `QuadSourceIntegral`, in the key, through target, factory, `main.py` and the extraction script. Closes `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`. | 08 | ⏸️ |
| K4 | **REMEDY** | `QuadSource` keyed on `Tq_serial` and `Tr_serial`. | 09 | ⏸️ |
| K5 | **REMEDY** | `QuadSourceIntegral` keyed on `source_serial`, `data_serial` and its Tk rows. | 09 | ⏸️ |
| K6 | **REMEDY** | `GkSource` keyed on its Gk numeric and WKB rows, which are a set: recommended, a `GkSource_parents` table and a `parents_digest` key column. | 09 | ⏸️ |
| K7 | **FACILITY** | The store inventory resolves key parents across shards, so cross-shard parents are fingerprint identity. | 09 | ⏸️ |
| V1 | **VERIFY** | A scoped build through the registry into a new throwaway store, interrupted and resumed. | 10 | ⏸️ |
| V2 | **VERIFY** | Every target extracted from it; its fingerprint; the census of audit §4.4. | 10 | ⏸️ |
| V3 | **RECORD** | The campaign closed; what A3 v2's launch must now pass. | 10 | ⏸️ |

---

## 3. Active and unresolved issues

### 3.1 Assigned to this campaign (2026-09-26)

Each stays on its owning board, which carries an **Assigned** line, and closes on that board's §4.

| Issue | Owning board | Prompt |
|---|---|---|
| `[00-replicated-writes-can-diverge-across-shards]` | store-fingerprint | 01 (**narrowed** 2026-09-26), closed by 02 |
| `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` | tolerance-convergence | 03 |
| `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` | store-fingerprint | 03 |
| `[03-integrationsolver-stepping-minimum-lookup]` | GkTk-remedial | 03 |
| `[01-read-batch-is-outside-the-guard]` | datastore-readback | 04 (**closed** 2026-09-26, on that board's §4) |
| `[00-tagged-read-batch-joins-an-unselected-alias]` | store-fingerprint | 05 |
| `[00-numeric-value-parent-lookup-omits-break-point-kind]` | store-fingerprint | 05 |
| `[00-oneloop-lookup-joins-the-wrong-tag-table]` | store-fingerprint | 05 |
| `[00-quadsource-tq-serial-has-the-wrong-foreign-key]` | store-fingerprint | 06 |
| `[20-wkb-rows-consume-numeric-initial-data]` | GkTk-remedial | 07 |
| `[00-quadsourcepolicy-rows-are-referenced-by-nothing]` | store-fingerprint | 08 |

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
- **[00-oneloop-factory-cannot-round-trip-a-row]** *(opened 2026-09-25 by the audit)*
  - **The defect.** `sqla_OneLoopIntegral_factory.build` puts no `source_serial` in the payload
    and the table has no such column, while `ComputeTargets/OneLoopIntegral.py:43` reads
    `payload["source_serial"]`, so every hit raises `KeyError`. `store()` writes a `validated` key
    to a table with no such column (audit §3.12, `probe_06`).
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

---

## 4. Resolved issues

None yet.

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
