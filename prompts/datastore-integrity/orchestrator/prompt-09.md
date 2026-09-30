# Orchestrator — prompt 09, derived rows keyed on their parents (09a → 09b → 09c)

Read [`../README.md`](../README.md) first: §0.1, §2, §4, §5, and §6.2 **D1** (in full), **D4**, **D5**
and **D6**. **You do not write code.** You may *replay* a mutation a log records, with `git apply`,
and then revert it. You may *sketch* in a throwaway worktree to measure, and then remove it (§1).

**The charter** is [`09-key-derived-rows-on-their-parents.md`](../09-key-derived-rows-on-their-parents.md).
It is unchanged, and **split by the user on 2026-09-30** (§0.1) into three prompts, dispatched in
this order within one session:

| # | Prompt | Items | Model | Closes |
|---|---|---|---|---|
| 09a | [`09a-key-quadsource-on-its-transfer-functions.md`](../09a-key-quadsource-on-its-transfer-functions.md) | K4; K7's cross-shard `Parent` | Sonnet | — |
| 09b | [`09b-key-gksource-on-its-parent-set.md`](../09b-key-gksource-on-its-parent-set.md) | K6; K7's set-valued parent | **Opus** | `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` (§0.3) |
| 09c | [`09c-key-quadsourceintegral-on-its-inputs.md`](../09c-key-quadsourceintegral-on-its-inputs.md) | K5; K7 complete | **Opus** | — (D1's class, on this board) |

**Gate:** 08 landed and reviewed (board §1; `b18bb0b`, remnants `a9b6ff6`, review `9e0f263`).
**Never dispatched concurrently with anything else in this checkout.**

## 0. What makes this session unusual

- **Three prompts, one charter.** Each part is its own commit, log, witness, golden and format bump
  (4, 5, 6).
  - **Review each fully, and record the review, before dispatching the next.** Each review's
    findings go into the next dispatch's brief.
  - A finding that needs the user's decision ends the session there. Report it, and do not
    dispatch the next part.
- **The order is epistemic.** It follows README §0.1: a row's key is made to reach its parents
  before anything downstream is keyed on that row.
  - `QuadSource` goes first. It is `QuadSourceIntegral`'s parent, and its `Tr` is the campaign's
    first cross-shard key parent, so the inventory's cross-shard form lands with it.
  - `GkSource` goes second. It is the parent of `GkSourcePolicyData` and so of
    `QuadSourceIntegral`, and it needs the set-valued inventory parent.
  - `QuadSourceIntegral` goes last, keyed on parents that are by then themselves keyed.
- **The charter was written on 2026-09-26**, before 05c, 07 and 08 landed. What moved under it is
  §0.6, measured at `9e0f263`.
- **09a was sketched** in a throwaway worktree of `9e0f263`, since removed. The sketch had:
  - `Tr_serial` indexed, and both serials in `build()`'s filter;
  - a cross-shard `Parent`, with `Tq` and `Tr` declared;
  - `test_store_inventory`'s three following edits.

  It did not touch the target, `main.py`, the readers or any other test, and the agent is not given
  it. **09b and 09c are sketched before their own dispatch** (§1), on the tree the previous part
  leaves, because their measurements depend on it.

**The user decided four things on 2026-09-30, while this note was written.** They are §0.1–§0.4,
and they bind every part as the charter does.

### 0.1 The split (the user: three, by dependency)

The charter's own first stop is "the work will not fit one reviewable commit … propose a split".
Measured, it would not fit:
- **07** was 29 files and **08** was 24. Each keyed one class on one parent.
- The charter keys three classes, on 2, 6 and a set of parents.
- It adds a table and a set-valued inventory parent.
- It restructures three `main.py` stages, touching at least **nine** lookup sites (§0.6). The
  charter names three.

**Decided:** three prompts, in the order above, one orchestrator session like 05's, with a review
between each. Each part's prompt file says what is its own; the charter's §1, §2 and §6 bind it,
and its §3 tests and (i)–(vi) mutations are distributed across the three parts.

### 0.2 Downstream stages find a row by its parents (the user: keyed on parents)

**Where the question arises.** Three `main.py` stages fetch an upstream *derived* row by physical
labels:
- the `GkSourcePolicyData` stage fetches `GkSource`, three times;
- the `QuadSourceIntegral` stage fetches `GkSource` and `QuadSource`.

After 09 each of those rows is keyed on its parents. So a downstream lookup either re-resolves the
parents, or selects by labels and refuses more than one match (07's `labels_only`).

**Decided: keyed on parents.** A resume after an upstream change under the same tags then works. A
labels-only lookup would stop there instead. The cost is extra serial-only lookups, including a Gk
parent-set read wherever a `GkSource` is fetched. 09b therefore recommends one helper shared by the
three stages, so that none can compute a different digest.

**Readers are unchanged by this.** The extraction scripts and the
`docs/source-remediation-verification/` scripts still select by labels and refuse more than one
match, as D1 allows.

### 0.3 `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]` is 09b's (the user: assign it)

**Decided:** assigned to 09b, which already edits `GkNumericValue.read_batch` and
`GkWKBValue.read_batch` to hand back their parents.
- The issue covers four reads: `GkWKBValue`'s and `TkWKBValue`'s `_build_impl_model` and
  `read_batch`. 09b closes all four.
- This note's commit adds the issue's **Assigned** line on the board and marks its index hook
  "**→ prompt 09b.**".

### 0.4 `EXPECTED_SITES` may grow, and only by each part's own sites (the user: granted)

Resolving parents ahead of an existence query will likely add lookup call sites. That moves
`ComputeTargets/tests/test_main_plumbing.EXPECTED_SITES` (at `9e0f263`: `TkNumericIntegration` 5,
`GkSource` 6, `QuadSource` 3, `GkNumericIntegration` 3, `GkWKBIntegration` 3, `TkWKBIntegration` 3,
`GkSourcePolicyData` 4, `QuadSourceIntegral` 2).

**Decided:**
- A part may raise a count only by the parent-resolution sites it adds. Its log names each site,
  with the class and the count before and after.
- A count that falls, or moves for any other reason, is a stop.
- This is the one edit to `test_main_plumbing.py` any part may make.

**The read-back guard's pins** (`test_factory_select_columns.py`: `EXPECTED_COVERAGE`,
`KNOWN_UNRESOLVED_TABLES` and the rest) are under the same rule: growth only by the part's own new
sites, each named. *(Granted by the user on 2026-09-30, with §0.5's 09a list.)*

### 0.5 The existing tests that must follow the key

**09a — measured on the sketch** (the schema half and the inventory, with no target or `main.py`
change):

| Suite | Result on the sketch | Failing |
|---|---|---|
| Datastore | `failures=7` | `test_schema_builder.TestSchemaIsUnchanged`: 2 tests, each with a `cls='QuadSource'` sub-test (D5). `test_store_inventory`: `TestIdentityColumnsMatter.test_the_key_fields_are_the_lookup_columns` (`cls='QuadSource'`); `TestNonIdentityDoesNotMatter.test_non_identity_columns` (`table='QuadSource', column='Tq_serial'`); `TestTags.test_a_tag_on_the_background_model_reaches_its_children` |
| RunRegistry | `failures=1` | `test_store_fingerprint.TestGolden.test_the_golden_fingerprint` (D5) |
| ComputeTargets | 585 OK | — |

With the three `test_store_inventory` edits below, that module ran 49 OK.

**Granted for 09a** (the user, 2026-09-30). Each is limited to following the key: `Tq` or `Tr`
joins, and nothing else asserted changes.
- **`Datastore/tests/test_store_inventory.py`:**
  - `IDENTITY["QuadSource"]` gains `"Tq": ("QuadSource", 1, "Tq_serial", 3)` and
    `"Tr": ("QuadSource", 1, "Tr_serial", 3)`. The fixture already has the rows: Tk numeric 3 is
    real, on shard 0. It is the same shard as `QuadSource` 1's `Tq`, and another shard from its
    `Tr` (2, on shard 1), so the variation tests identity, not resolution. *Agent's choice* of
    other real serials, with the reason.
  - `NON_IDENTITY` loses `("QuadSource", 1, "Tq_serial", 9)`, which is now identity.
  - `test_a_tag_on_the_background_model_reaches_its_children` expects `QuadSource` too. Model 2's
    Tk numeric 3 is `QuadSource` 2's `Tr`. The comment "model 2 is the parent of
    TkNumericIntegration 3 only" stays true; a sentence may say why the change now reaches the
    source.
- **`ComputeTargets/tests/test_quadsource.py`:** the two `QuadSource(...)` constructions (`:372`,
  `:435`) gain the serials, *if* the constructor comes to require them.
- **The two witnesses, under D5:** `test_schema_builder.py`'s `WITNESS` and its docstring's
  "current one" sentence; `test_store_fingerprint.py`'s `GOLDEN`.
- **`test_main_plumbing.EXPECTED_SITES`**, as §0.4.

**09b and 09c.** Measured before each dispatch, by §1 step 7. The grant has the same shape: each
existing test that must change, the reason, and "follows the key, nothing else". It goes to the user
before the dispatch, and the brief carries it as this section's list for that part. The candidates
at `9e0f263`:
- **09b:**
  - `real_store_fixtures.py`'s three `GkSource` rows gain a digest and `GkSource_parents` rows;
  - `test_store_inventory`'s `IDENTITY["GkSource"]` and any `TestTags` set that a Gk change now
    reaches;
  - `shard_store_fixtures.py`, `test_store_reader.py`, `test_foreign_key_check.py`'s map and
    `test_gauss_order_key.py`, if they build `GkSource` tables or rows.
- **09c:**
  - the fixture's three `QuadSourceIntegral` rows gain the four columns;
  - `IDENTITY["QuadSourceIntegral"]` gains the six parents, and `NON_IDENTITY` loses
    `source_serial` and `data_serial`;
  - `test_integral_read_batch.py`'s raw inserts and payloads;
  - 08's `test_quadsource_policy_key.py` and `test_quadsource_policy_foreign_key.py`, which insert
    `QuadSourceIntegral` rows;
  - `test_quadsource_integral.py`, if the constructor changes;
  - `_UNDECLARED_REFERENCES` in `real_store_fixtures.py`.

**Granted for 09b** (the user, 2026-09-30, on the sketch of §0.7). Each is limited to following the
key: the parent set joins, and nothing else asserted changes.
- **`Datastore/tests/real_store_fixtures.py`:** the three `GkSource` rows gain `parents_digest`, and
  the fixture gains `GkSource_parents` rows naming real Gk numeric and WKB rows of each source's own
  shard and exit time, at those rows' own `z_source`. *Agent's choice* of which rows, with the
  reason.
- **`Datastore/tests/test_store_inventory.py`:** `IDENTITY["GkSource"]` gains the parent-set field,
  varied through a `GkSource_parents` row to another real Gk row on the same shard, so that the
  variation tests identity, not resolution.
- **`Datastore/tests/test_gauss_order_key.py`**, *only if* the new `GkSource` `store()` makes it
  follow the key. It calls the real `store()` with a stand-in and names its inserters. The sketch
  left `store()` alone, so this was not measured.
- **The two witnesses, under D5:** `test_schema_builder.py`'s `WITNESS` and its docstring's "current
  one" sentence; `test_store_fingerprint.py`'s `GOLDEN`.
- **`test_main_plumbing.EXPECTED_SITES`** and the read-back guard's pins, as §0.4.

Not needed on the sketch: `TestTags` (no set reaches a Gk change), `NON_IDENTITY`,
`shard_store_fixtures.py`, `test_store_reader.py`, `test_foreign_key_check.py`'s map.
`test_inventory_report.py` failed, and is **not** granted. The user granted `tools/inventory_report.py`
instead (§0.7).

### 0.6 What moved under the charter, and what 09a measured

**Shards** (`config/sharding.py`). This is the parent table the charter asks each log to carry.

| Row | Its shard | Parent | Parent's shard | Foreign key |
|---|---|---|---|---|
| `QuadSource` | `q` | `Tq_serial` (Tk numeric at `q`) | `q` | yes, since 06 |
| | | `Tr_serial` (Tk numeric at `r`) | `r` | no; **not even indexed** at `9e0f263` |
| `GkSource` | `k` | the Gk numeric/WKB rows, per `z_source` | `k` | possible for each |
| `QuadSourceIntegral` | **`k`** | `source_serial` (`QuadSource`) | `q` | no |
| | | `data_serial` (`GkSourcePolicyData`) | `k` | yes |
| | | `Tq` numeric, `Tq` WKB | `q` | no (new) |
| | | `Tr` numeric, `Tr` WKB | `r` | no (new) |

**Direct inputs, checked against each `compute()`.**
- `QuadSource.compute` receives `Tq` and `Tr`, both `TkNumericIntegration`; no WKB row.
- `QuadSourceIntegral.compute` receives the six rows above. It also receives `b`, `Bessel_0pt5`
  and `Bessel_2pt5`, which are not stored rows.
- `GkSource.compute` hands `assemble_GkSource_values` two maps keyed by `z_source`, of
  `GkNumericValue` and `GkWKBValue`.

The charter's first stop does not trigger for any of the three.

**What moved since the charter was written.**
- **`QuadSource` gains no column in 09a.** `Tq_serial` and `Tr_serial` exist, and `store()` has
  always written them. So 09a refuses no old store; only its fingerprint identity moves. The
  charter's "a store that predates the columns is refused" applies to 09b and 09c only.
- **The Gk value reads** (05c, 07) filter on the parent's full key, and refuse a second parent per
  `z_source`. To do so they already select the parent serial, `integration_serial` and
  `wkb_serial`, **but drop it**: the objects they return do not carry it. Neither read has a
  serial-only form.
- **Six `main.py` lookups find a `GkSource`**, at `9e0f263`, all by labels:
  - `build_GkSource_batch` (`:2428`, the existence query);
  - `build_missing_GkSource` (`:2585`, the construction);
  - `build_incomplete_GkSourcePolicy` (`:2761`);
  - `build_GkSourcePolicy_batch` (`:2859`);
  - `build_missing_GkSourcePolicy` (`:2984`);
  - `build_QuadSourceIntegral_batch` (`:3252`).

  The charter names only the GkSource stage. By §0.2, 09b keys all six.
- **`QuadSource` lookups**, all by labels:
  - `main.py:1703` and `:1817` (the stage);
  - `:3336` (the `QuadSourceIntegral` stage, which looks the Tk numeric rows up *after* it, at
    `:3397`, for the same modes).
- **`QuadSourceIntegral`** has its existence query at `:3163` (payloads built at `:3131`) and its
  construction at `:3518`. `qsi_work_batches` is at `:3585`, which is the sweep's `BATCH_LINE`.
- **`parents_digest` cannot be fingerprint identity.** As the charter specifies it, it is SHA-256
  over serials. It is sound as a lookup key, since serials identify rows. But the inventory never
  uses a serial as identity. So K7 has a second, set-valued half, resolved through
  `GkSource_parents`, and that half is 09b's.
- **The inventory can do K7 in one pass.** `read_inventory` builds each class on every shard
  before the next, in dependency order (`COMPUTE_TARGETS`: Tk, Gk, `GkSource`,
  `GkSourcePolicyData`, `QuadSource`, `QuadSourceIntegral`). So a cross-shard parent resolves
  against digests already built. The charter's fourth stop does not trigger.
- **Readers**, each by labels:
  - `extract_tensor_source_data.py:372` (`QuadSource`);
  - `extract_GkSource_data.py:879` (`GkSource`);
  - `extract_QuadSourceIntegral_data.py`: `:1154` (`GkSource`), `:1172` (`QuadSource`) and its
    `read_batch` (`:1138`);
  - `docs/source-remediation-verification/analyse_greens_and_source.py:282` (`GkSource`, with no
    tags);
  - `run_quadsource_integrals.py`: `:291` (`QuadSource`) and `:359` (`GkSource`).

  `tools/inventory_report.py` and `order_audit.py` read table names only.
- **`_drop_actions`.** `gk-source` already drops before `gk-wkb` and `gk-numeric`, and
  `GkSource_parents` joins it (09b). The within-action order is
  `[07-drop-actions-list-a-parent-table-before-its-children]`, which 09b does not fix.
- **The fixture** (`build_full_store`) already has cross-shard parents for 09a:
  - `QuadSource` 1 on shard 0 has `Tq` 1 (shard 0) and `Tr` 2 (shard 1);
  - `QuadSource` 2 on shard 1 has `Tq` 2 (shard 1) and `Tr` 3 (shard 0).

  No fixture row is needed for 09a.

**09a, measured on the sketch.**
- **The witness diff** against `schema_at_datastore-integrity-08.json` is **3 hunks, 14 lines**, all
  in `QuadSource`'s `Tr_serial`: `"index": null` → `true`, one index DDL line, and one index
  description. A line anywhere else is a stop.
- **The fingerprint** of `build_full_store()`, compared with the 08 golden: **only `QuadSource`
  moves** (2 → 2 records, tag sets [1, 1] → [1, 1]). The other 20 classes are unchanged, and
  `problems` is empty.
- **With `Tr` declared same-shard** (mutation (ii)), `problems` is
  `{'QuadSource': {'unresolved-parent': 2}}` and `QuadSource` has no records. Both fixture rows'
  `Tr` is cross-shard, so the mutation is caught.
- **The fourteen audit probes**, run on the sketch against `b18bb0b`'s outputs. None newly fails.
  Three change:
  - `probe_02`'s line numbers;
  - `probe_03`'s count, 1254 → 1256;
  - `replicated_divergence_probe` gains two correct lines. Its own fixture's `QuadSource` rows
    name a `Tq` and a `Tr` it does not hold, so they are now `unresolved-parent`.

  The five that already fail are unchanged, and `probe_01` still fails, as after 08.
- **`EXPECTED_SITES`** did not move on the sketch, which did not touch `main.py`.

**Carried from the review of 08** (board, after the 07 review):
- **Another session may write in this checkout.** Stage by explicit path only, never `git add -A`,
  `.` or `commit -a`. Show `git diff --cached --name-only` before committing. Stop and ask if
  another session modifies a tracked file the agent needs.
- **Count, then paste.** The witness diff by kind, and the pins by name. Paste `-v` output
  unfiltered.
- **A report states only what the diff shows**, and every claim is checked against `git show`. Log
  08's site table had two citations one line off; line numbers come from `git show`, not from a
  `sed` view of the working tree.
- **Mutations.** A mutation record lists every failing test, counted, and says which are sub-tests.
  The replay is done twice: before the commit from the final log file, and after it from
  `git show HEAD:<log>`, reported with no amend.
- **The flake**, `test_tk_wkb_phase.test_wall_time_per_object`, fires under load: it fired in the
  08 baseline at a load average near 15. Re-run it alone before calling it real.
- **Untracked test files.** `test_inventory_retired`'s `git grep` sees only tracked files, so a new
  module's name clash shows only once it is staged (log 08, deviation 8). Run the suites on the
  staged tree.
- **Adjacent prose.** A docstring or comment beside a granted line that the change makes false is
  granted too, and must be fixed. Log 08 kept `QuadSourcePolicy`'s "the schema … do[es] not churn",
  and the review had to fix it.
- **`TestSchemaIsUnchanged` keeps its name** (the 06 review).
- **Residue.** The agent follows README §5 rule 4, recording and not fixing. Small residue found in
  review is the orchestrator's, in a follow-up commit (the user, 2026-09-29).
- **Scratch files.** The agent works in its own scratch subdirectory, not the session
  scratchpad's top level.
- **Audit probes are left as they are** (the user, 2026-09-29). Each log names every probe it stops
  in its rule-11 list.

### 0.7 Before 09b: the 09a review, the sketch, and the user's decisions

*(Added 2026-09-30, after the review of 09a, `7728518`.)*

**Carried from the review of 09a** (board §1, after the 08 review). All twelve checks passed.
- **A comment attributed a user's decision to the orchestrator.** The 09a review fixed it in
  `f161b2f`. §0.1–§0.4 and this section's decisions are **the user's**, and code comments say so.
- **Log 09a's site table was exact against `git show`, all 29 citations.** Keep to that.
- **The mutation replay runs in a `git archive` export**, so the checkout stays alone. There, four
  tests that need a git repository fail every run: `test_inventory_retired`'s two, and one each in
  `test_run_registry` and `test_store_sidecar`. They are not a mutation's. An agent that replays in
  a worktree of its own does not see them.
- The 09a golden and witness are the ones to diff from: `schema_at_datastore-integrity-09a.json`,
  `full_store_fingerprint_datastore-integrity-09a.json`, `FINGERPRINT_FORMAT` 4 → 5.

**09b, sketched** in an export of `7728518`, since removed. The sketch had:
- `GkSource_parents`, with its own `serial` and four foreign keys (to `GkSource`, `redshift`,
  `GkNumericIntegration` and `GkWKBIntegration`), each indexed;
- `parents_digest` (`String(64)`, NOT NULL, indexed) on `GkSource`, in `build()`'s filter, with
  07's D4 refusal;
- a set-valued inventory form, resolving each member to its parents' reference digests on the
  source's shard, whose key field is a digest of the sorted resolved members;
- the table in the `gk-source` drop action;
- the fixture rows above.

It did not touch `store()`, `main.py`, the value reads or the readers, and the agent is not given it.
- **Existing tests.** Datastore 459, ComputeTargets 610 and RunRegistry 199 at base. On the
  sketch, before any test edit, four failed:
  - the two `TestSchemaIsUnchanged` tests (D5);
  - `test_store_inventory`'s `test_the_key_fields_are_the_lookup_columns` (`cls='GkSource'`);
  - RunRegistry's `test_the_golden_fingerprint`.

  With the fixture rows in and no report edit, a fifth failed:
  `test_inventory_report.TestTheDisplaySaysWhatTheInventoryHolds.test_no_digest_appears`. The report
  prints a key field it does not know as a leaf, so the set's digest reached the display. With
  §0.5's edits and the report edit, all three suites passed.
- **The witness diff** against 09a: **9 hunks, 246 changed lines, 256 lines of output**. All are in
  `GkSource` (its column in the record, the table and the DDL; its index in `index_ddl` and
  `indexes`) and `GkSource_parents`: its class entry, and its name in `class_order`,
  `metadata_tables` and `table_order`. The implementer's may differ in detail, since the sketch is
  not theirs; a hunk outside those two classes is a stop.
- **The fingerprint** against the 09a golden: `GkSource` (3 records) and `GkSourcePolicyData` (3,
  through its source) moved, and the other 19 did not. `problems` was empty. With the set left out
  of `GkSource`'s identity (mutation (iv)), it went back to the 09a fingerprint exactly.
- **Pins and `EXPECTED_SITES`** did not move (the sketch did not touch `main.py`).
- **Probes** against `7728518`: none newly fails. Four change:
  - `fk_sweep_probe` gains `GkSource_parents`' block;
  - `probe_02`'s line numbers;
  - `probe_03`'s count, 1256 → 1258;
  - `replicated_divergence_probe` gains two correct cascade lines: `GkSource` 3's member is
    unresolved on shard 1, so `GkSourcePolicyData` 3 is too.
- **Found by the sketch:**
  - A **seventh labels-only `GkSource` lookup**: `GkSourceValue._build_impl_model`, outside
    `main.py`, with no caller.
  - **An empty parent set** would get the digest of `[]` and no named problem.
  - An **09a store's inventory**, lacking `GkSource_parents`, gives `absent-table` for `GkSource`, and
    then `unresolved-parent` for `GkSourcePolicyData`. That is D4's expected consequence, and not a
    decision.
  - `relabel_serials` leaves the fixture's `parents_digest` stale. That is harmless, since the
    fingerprint never uses it.

**The user decided four more things on 2026-09-30**, on the sketch. They bind 09b (and the third
binds 09c) as §0.1–§0.4 do.
1. **09b's existing-test grant** is §0.5's "Granted for 09b", `test_gauss_order_key.py` included
   on its condition.
2. **`tools/inventory_report.py` joins 09b's files**, limited to rendering the parent set by its
   physical labels (each member's `z_source`, numeric and WKB rows, as the report already renders a
   parent reference). `test_inventory_report.py` is **not** modified, and still forbids a digest in
   the display. *Not taken:* putting the resolved member list in the key itself, which breaks two
   more existing tests.
3. **The two labels-only value model reads.**
   - **09b** gives `GkSourceValue._build_impl_model` D1's reader treatment. It refuses more than
     one matching `GkSource`, naming what distinguishes them, as 09b's readers do, and serves one
     match.
   - **09c** does the same for `QuadSourceValue._build_impl_model`, and closes
     `[09a-quadsourcevalue-model-read-joins-every-source-under-its-labels]`, assigned to it on the
     board.
4. **An empty parent set is refused.** `store()` refuses to write a `GkSource` with no
   `GkSource_parents` rows, and the inventory names such a row as a problem.

**Also in 09b's files**, if the new `store()` needs them for the new table: one entry each in
`config/sharding.py` and `ClientPool`'s serial batch-size map. The sketch did not exercise a pool
write.

## 1. Before each dispatch

1. Record the branch and `HEAD`. For 09a it is this note's commit, or a later one that touches no
   code; say which. For 09b and 09c it is the previous part's review commit.
2. `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows this
   checkout alone. If another session has uncommitted work here, do not dispatch; ask the user.
3. `python -m RunRegistry list`: nothing `running`.
4. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is missing:
   SHA-256, size and mtime of every file, and the directory list. Never put it in the tree. The 08
   session's snapshot held 46 files and 10 directories.
5. Baselines: the board's latest §5 subsection. Before 09a that is §5.11: AdaptiveLevin 32,
   ComputeTargets 585, CosmologyModels 39, Datastore 422, LiouvilleGreen 148 (1 skipped),
   RunRegistry 199. Re-measure all six on `HEAD`, serially.
6. Run the `substitute_main_py()` count on `HEAD`: each of the five literals occurs once. At
   `9e0f263` they are at `:3585`, `:3692`, `:3693`, `:3791` and `:3803`.
7. **For 09b and 09c: sketch, measure, and ask.**
   - In a throwaway worktree of `HEAD`, sketch the part's schema half: the columns and table, the
     factory's filter, the inventory declarations, and the fixture rows NOT NULL requires.
   - Run the Datastore, ComputeTargets and RunRegistry suites, and record every failing existing
     test.
   - Capture the witness diff and the fingerprint's moved classes. The fingerprint script is
     `orch_09_fingerprint.py` in the session scratchpad; write it again if it is missing.
   - Run the fourteen audit probes against the previous part's outputs.
   - Put the list of existing tests that must follow the key to the user, as §0.5's grant for that
     part, together with anything else that needs a decision. Then remove the worktree.
   - Do not dispatch until the user has answered.
8. Count the index rows (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`), and tell the agent the number.
   It is 98 at `9e0f263`, and this note's commit does not change it.

## 2. Dispatch

One fresh-context subagent per part, on the model the table names. Give it:
- the part's prompt, and the charter;
- the README, with 6.2's D1, D4, D5 and D6;
- the audit, `docs/datastore-integrity-audit.md` §2.4–§2.6;
- logs 07 and 08 (the D5 procedure, twice), and every earlier part's log;
- `HEAD`, the baselines, the index count and the `substitute_main_py()` lines;
- **§0 of this note, up to "## 1. Before each dispatch" only**, with §0.5's list for that part and
  any review findings since. Give it as a copy in the session scratchpad; the agent does not read
  `orchestrator/`.

Tell it plainly:
- **one commit**, with its log at the part's `logs/` path, in README §5.1's form, with the part's
  §7 additions;
- in the same commit:
  - this board: its header status line, its §1 row, its items, §3.3 for anything opened, and a new
    §5 subsection (`5.12 After prompt 09a`, then `5.13`, `5.14`);
  - for 09b, the 07 issue moved to this board's §4 with a closing note naming the commit by
    subject;
  - for 09c, README §4 with the four column names;
  - `docs/OPEN_ISSUES.md`, with its count and date corrected by counting;
- **the files it may change** are those its prompt's §2 names, and for each only what §2 says:
  - the factories and compute targets of its classes;
  - `Datastore/store_inventory.py`, for K7's form;
  - `main.py`'s stages as §0.6 lists them;
  - the readers its prompt names;
  - `Datastore/SQL/Datastore.py`'s `_drop_actions`, for 09b only;
  - `RunRegistry/stores.py`'s `FINGERPRINT_FORMAT` and its comment;
  - the existing tests of §0.5, as it bounds them;
  - one new witness and one new golden;
  - new test modules;
  - the log, the boards and the index;
- **nothing else**:
  - no existing witness or golden;
  - not `schema.py` or `ShardedPool.py`;
  - not another part's classes;
  - not `docs/handover/quadsource_atol_sweep.py`, any audit probe, or anything in `orchestrator/`;
- **no Ray, and nothing opens anything under `var/`**, or a copy of anything under it. Every store is
  in a temporary directory;
- every mutation diff is pasted from `git diff`, and replayed twice as §0.6's carried rules say;
- run `black`, and the suites on the staged tree;
- the part's §5 stop conditions mean *stop and ask*. So do:
  - an existing test outside §0.5's list for that part failing;
  - a fingerprint class moving other than those its prompt expects;
  - a witness line outside the part's measured kinds;
  - `EXPECTED_SITES` or a guard pin moving other than as §0.4 allows.

## 3. The review of each part

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's files for that part.
   - `git diff HEAD~1 HEAD --` over every older witness and golden, `schema.py`, `ShardedPool.py`,
     `docs/handover/`, `docs/datastore-integrity-audit/` and `orchestrator/` is empty.
   - `--diff-filter=M` over `*/tests/*` shows exactly §0.5's files for the part. Read every hunk
     against §0.5: each follows the key, and asserts nothing new or less.
   - Adjacent prose the change made false is fixed.
2. **The key.** Every parent the part names is in `build()`'s filter, written by `store()`, and, where
   there is a `read_batch`, filtered on and handed back. Each is declared to the inventory in the
   right form: same-shard, cross-shard or set.
3. **The target.** `compute()` or `store()` asserts every relation the part names, and a mismatch
   raises naming both sides. With the parent commit's target checked out over the new tests, they
   fail; then restore.
4. **`main.py`.**
   - Every hunk is a site of §0.6.
   - Each existence query now runs after its parents are resolved.
   - The work-list lines are untouched.
   - `EXPECTED_SITES` moves only as §0.4 allows, each site named.
   - Run the five `substitute_main_py()` counts yourself.
5. **Readers.** Each reader the part names refuses more than one match, naming what distinguishes
   them, and returns a single match without parents. Grep the tree for any lookup of the part's
   class that passes neither parents nor the reader mode.
6. **D4.** Re-create an old table from the previous witness's DDL, and check that each reader the
   part names refuses it with the D4 message, or, for 09a, reads it.
7. **D5 and the fingerprint.**
   - Both new files are new, and every older witness and golden is byte-identical at both commits.
   - Re-capture the witness from `HEAD`: byte-identical.
   - Diff it against the previous witness yourself, and match the log's paste. For 09a expect
     §0.6's 3 hunks and 14 lines.
   - Re-capture the fingerprint of `build_full_store()` from `HEAD`: it equals the new golden, and
     only the expected classes differ from the previous golden.
8. **The inventory form.** Break it (for 09a, declare `Tr` same-shard) and confirm the problem is
   named and the test fails; restore.
9. **Replay the mutations** from the committed log, with plain `git apply`, in a worktree of `HEAD`.
   Each gives its logged failures. Revert each, leaving the tree clean.
10. **Probes.** Run all fourteen at `HEAD~1` and `HEAD`. Every change is named in the log's rule-11
    list.
11. **Boards.** The part's §1 row, items and §5 subsection. For 09b, the 07 issue on §4. Count the
    index rows, and check the date.
12. **`var/`, suites, `black`, Ray.**
    - The snapshot is identical.
    - Run all six suites serially. Each rises by exactly the tests added.
    - `black --check` is clean.
    - `ray.is_initialized()` is false after each new module.

## 4. After each review

Record it on the board, after the previous review. Fix any small residue in a follow-up commit of
its own (the user, 2026-09-29). Then carry the findings into the next part's brief, and run §1 for
it.

**After 09c's review**, K4–K7 are done and D1's class is closed on this board. **Prompt 10 is next.**
It is the first real run: the first time any of these keys, the cross-shard and set-valued
inventory parents, and the downstream stages' re-resolution meet a pipeline. Its own orchestrator
note carries all three reviews.
