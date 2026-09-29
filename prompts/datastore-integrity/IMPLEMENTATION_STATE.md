# Datastore integrity campaign — implementation state

**Last updated:** 2026-09-29 · **Status: 6 of 10 prompts landed (01, 02, 03, 04, 04b, and 05 as
05a–05c), all reviewed.** The user decided D2–D6 on 2026-09-26 (README §6.2), and D1 (yes) later
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
corrected on that board and in the index. The review opened one issue, closed the same day (§4). **Next: 06**, after 03; 07 waits on 05 and 06. 10 waits on 01–09.
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
| 06 | [Correct the foreign keys](06-correct-the-foreign-keys.md) | **S1**–**S5** | Sonnet | ✍️ yes, 2026-09-26 | ⬜ after 03 | — | — |
| 07 | [Key WKB rows on their numeric row](07-key-wkb-rows-on-their-numeric-row.md) | **K1**–**K2** | Opus | ✍️ yes, 2026-09-26 | ⬜ after 05 and 06 | — | — |
| 08 | [Key `QuadSourceIntegral` on its policy](08-key-quadsourceintegral-on-its-policy.md) | **K3** | Sonnet | ✍️ yes, 2026-09-26 | ⬜ after 07 | — | — |
| 09 | [Key derived rows on their parents](09-key-derived-rows-on-their-parents.md) | **K4**–**K7** | Opus | ✍️ yes, 2026-09-26 | ⬜ after 08 | — | — |
| 10 | Build a throwaway store end to end | **V1**–**V3** | Opus | ⏸️ held until 01–09 land | — | — | — |

**Orchestrator notes** are written one prompt at a time, just before its dispatch, so that each
carries the previous review's findings: [`orchestrator/prompt-01.md`](orchestrator/prompt-01.md)
and [`orchestrator/prompt-04.md`](orchestrator/prompt-04.md) exist (2026-09-26). 01 and 04 are
dispatched one at a time, never concurrently in this checkout, in either order. Since then:
[`prompt-04b.md`](orchestrator/prompt-04b.md) and [`prompt-05.md`](orchestrator/prompt-05.md)
(2026-09-28, both used), [`prompt-02.md`](orchestrator/prompt-02.md) (2026-09-28, used), and
[`prompt-03.md`](orchestrator/prompt-03.md) (2026-09-29, used).

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
| `[00-replicated-writes-can-diverge-across-shards]` | store-fingerprint | 01 (**narrowed** 2026-09-26), 02 (**closed** 2026-09-28, on that board's §4) |
| `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]` | tolerance-convergence | 03 (**closed** 2026-09-29, on that board's §4) |
| `[02-exit-time-lookup-runs-inside-the-subhorizon-loop]` | store-fingerprint | 03 (**closed** 2026-09-29, on that board's §4) |
| `[03-integrationsolver-stepping-minimum-lookup]` | GkTk-remedial | 03 (**closed** 2026-09-29, on that board's §4) |
| `[01-read-batch-is-outside-the-guard]` | datastore-readback | 04 (**closed** 2026-09-26, on that board's §4) |
| `[00-tagged-read-batch-joins-an-unselected-alias]` | store-fingerprint | 05b (**closed** 2026-09-28, on that board's §4) |
| `[00-numeric-value-parent-lookup-omits-break-point-kind]` | store-fingerprint | 05c (**closed** 2026-09-28, on that board's §4) |
| `[00-oneloop-lookup-joins-the-wrong-tag-table]` | store-fingerprint | 05b (**closed** 2026-09-28, on that board's §4) |
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

- **[05c-read-back-guard-cannot-see-reads-in-a-comprehension]** *(opened 2026-09-28 by prompt 05c)*
  - **The defect.** `Datastore/tests/test_factory_select_columns.py`'s `analyse_build` walks
    nodes in pre-order. For `[f(row.x) for row in rows]` it reaches the element `row.x` before the
    `comprehension` node that binds `row`. So the read is charged to an earlier binding of that
    name, or, with none, is dropped: it is neither checked nor declared.
  - **Where, on this tree** (an `ast` walk for comprehensions whose element reads an attribute of
    their own target, log 05c observation 3):
    - `redshift.py:113` and `wavenumber.py:112` (`read_table`);
    - `BackgroundModel.py:401` (`build`, already inside `KNOWN_BLIND`'s function);
    - the seven `invalid_serials = [nv.serial for nv in not_validated]` in `validate_on_startup`.
  - **Impact.** Nothing is missed today. Both `read_table`s select every column they read, checked
    by hand, and the seven `validate_on_startup` reads take `serial`, which their queries select.
    But the guard's docstring does not declare the hole, and no pin would catch a future defect in
    this shape. 05c wrote `TkWKBValue.read_batch`'s refusal as a `for` loop for this reason (log
    05c, deviation 6).
  - **Next step.** Teach the analyser to bind a comprehension's targets before walking its
    element, and pin what that newly decides. Or declare the shape in the docstring and in a
    pinned list. A change to the guard, which is not 05c's to make.
- **[02-the-version-row-is-written-outside-the-in-flight-record]** *(opened 2026-09-28 by prompt 02)*
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

---

## 4. Resolved issues

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
