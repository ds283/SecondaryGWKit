# Campaign — datastore-generic-followup

**Written:** 2026-10-06 on branch `handover-remedial` at `6de692a`, by Claude Opus 5.5, from the
three issues `datastore-generic` left open and the user's decision on each (§6.1). Its audit is
`datastore-generic`'s board, which holds each issue's original measurement. §0.2 re-measures each
issue at `6de692a`. **All three prompts are written**: their tasks are fixed by §6.1, and a rough
version of each change was measured against the tree (§0.2, [`measure/`](measure/)).
**The campaign closed on 2026-10-07, 3 of 3 prompts landed** (01 `fee1fbf`, 02 `4c34f13`, 03
`086c81a`); [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md) holds the record.

## 0. Why this campaign exists

`datastore-generic` closed on 2026-10-06 (`f58c785`) with three issues open on its board, each a
decision for the user. The user took all three the same day (§6.1):

1. **`[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`.** A `--drop` of one group
   leaves other tables' rows naming rows it dropped. A re-created table then leases serials those
   stale rows still name. **Decided: refuse the drop.**
2. **`[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`.** `StoreInventory.resolve`
   shows a nullable parent that the row does not have as a parent it failed to find.
   **Decided: render it as `None`**, as a parent-set member already is.
3. **`[04-closed-store-refusals-repeat-their-prefix]`.** The closed-store operations' refusals
   say "Cannot delete sharded datastore …" twice. **Decided: remove the prefix in
   `_read_closed_store`**, so that each caller states it once.

### 0.1 Order

No prompt depends on another's result: each changes different code, and each is measured against
the same tree. The order is the user's numbering of the decisions. 01 and 03 both edit
`ShardedPool.py`, so they never run concurrently in one checkout.

### 0.2 Reconciliation against `6de692a`

Measured on 2026-10-06 with the scripts in [`measure/scripts/`](measure/scripts/). The rough
change is [`measure/rough_change.diff`](measure/rough_change.diff), made in a scratch export of
`6de692a`; the verdicts are in [`measure/out/`](measure/out/).

**1. The drop (prompt 01).**
- **The issue undercounted.** Its measurement used `PRAGMA foreign_key_check`, which sees only
  foreign keys. The factories also declare parents with no foreign key, in their `inventory_spec`
  (prompt 08 of `datastore-generic`):
  - `QuadSourceIntegral` names `TkNumericIntegration`, `TkWKBIntegration` and `QuadSource` rows
    across shards;
  - `GkSource` names `GkNumericIntegration` and `GkWKBIntegration` rows through its parent set;
  - `wavenumber_exit_time` and `BackgroundModel` name a cosmology through the polymorphic
    reference.

  So `--drop tk-wkb` and `--drop quad-source` alone also leave `QuadSourceIntegral` rows naming
  rows that are gone. This is not visible to any foreign-key check (`measure/out/f1_groups.txt`).
- **Under the user's rule (§6.1)**, foreign keys and declared parents, followed transitively,
  8 of the 10 groups are refused alone. Each must be dropped with these tables
  (`measure/out/f2_closure.txt`):

  | Group | Must also drop |
  |---|---|
  | `tk-numeric` | `TkWKBIntegration`, `TkWKB_tags`, `TkWKBValue`, `QuadSource`, `QuadSource_tags`, `QuadSourceValue`, `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `tk-wkb` | `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `quad-source` | `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `gk-numeric` | `GkWKBIntegration`, `GkWKB_tags`, `GkWKBValue`, `GkSourcePolicyData`, `GkSource`, `GkSource_tags`, `GkSourceValue`, `GkSource_parents`, `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `gk-wkb` | `GkSourcePolicyData`, `GkSource`, `GkSource_tags`, `GkSourceValue`, `GkSource_parents`, `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `gk-source` | `GkSourcePolicyData`, `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `gk-source-policy-records` | `GkSourcePolicyData`, `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `gk-source-policy` | `QuadSourceIntegral`, `QuadSourceIntegral_tags` |
  | `quad-source-integral`, `1loop-integral` | nothing |

  Every table in the right-hand column belongs to some group, so a refused drop can always be
  made by naming more groups. All ten groups together are accepted.
- **Each half of the rule matters.** Foreign keys alone miss the `QuadSourceIntegral` and
  `GkSource` edges. Declared parents alone miss every value table, tag table and
  `GkSource_parents`. Without transitivity the tables behind the first level are missed. Each of
  the three changes the list for some group (`f2_closure.txt`).
- **What moves.** With the rough refusal in `ShardedPool`'s constructor, one test body fails:
  `test_reconcile_at_open.TestPruningAfterRepair.test_an_interrupted_store`, which drops
  `gk-source` and `gk-source-policy-records`. It runs twice, because
  `test_one_timestamp_per_write.TestRepairThenPruneAtOpen` inherits it.
  - Its point is that a dropped replicated table goes on every shard alike.
  - Re-fixtured to also drop `gk-source-policy` and `quad-source-integral`, it passes, with its
    point intact.
  - Nothing else moves. The tests that call the actor's `_drop_tables` directly
    (`test_wkb_numeric_parent_key`) and the undeclared-table refusal test (`test_layer_registry`,
    `DROP = ["GkSource", "NotATable"]`) are untouched, because the rough refusal runs in the pool,
    after the undeclared check.
- **No caller passes a refused drop.** `main.py` is the only caller that passes `drop_tables`;
  the extract scripts pass none. No document outside the verification records tells anyone to
  run a single refused group.

**2. `resolve()` (prompt 02).**
- A record's key holds `None` for a parent field only when the parent is nullable and the row has
  none. `read_records` drops a row whose parent cannot be resolved, so `None` has no other
  meaning there. `_resolve_reference` already renders a `None` member as `None`.
- With the rough change, `resolve()` of every record of `build_full_store` and `build_real_store`
  differs from `6de692a`'s in exactly 11 entries. Each is a `numeric` field that changes from
  `{"class": "GkNumericIntegration", "digest": null, "unresolved": 0}` to `null`:
  - 2 are direct `GkWKBIntegration` records;
  - 9 are reached through records that reference them (3 each from `GkSource`,
    `GkSourcePolicyData` and `QuadSourceIntegral`).

  The issue's hook said all 11 came through `GkSource`'s parent set; this corrects it.
- No existing test pins the unresolved form for a `None` value. `TestResolve`'s
  `{"class": None, "digest": da, "unresolved": 0}` is a digest whose type the map does not hold,
  and is unaffected. The fingerprint, the inventory report and the inventory do not call
  `resolve()`.

**3. The prefix (prompt 03).**
- `_read_closed_store` has **four** callers, not the three the issue names:
  - `_plan_relocation` and `_plan_deletion` wrap its message with their own prefix;
  - `_read_back` is wrapped by `_relocate_store`'s `_failure_message`;
  - `store_reader.open_read_only` passes `_VERB` and does not wrap. That is why its message is not
    doubled today.

  So removing the prefix inside means the reader must wrap the message, as the others do. Its
  message then ends with the reader's own ". Nothing was read or repaired", which it does not
  today. `_read_closed_store`'s `verb` parameter has no use left, and goes.
- With the rough change, `delete_store`, `copy_store` and `open_read_only` each give the prefix
  once.
- **What moves:** one test, `test_store_reader.TestRefusals.test_missing_shard_is_refused_by_read_closed_store`.
  It calls `_read_closed_store(primary, "read")` and asserts that the reader's message equals that
  one. It changes to assert that the message contains `_read_closed_store`'s text under one
  prefix.
- `RunRegistry/stores.py` matches the phrase "its shards table could not be read" in these
  messages. The phrase survives, and RunRegistry runs 280, OK.

**The suites with all three rough changes**, at load 3–6:
- Datastore 721 = 147 + 183 + 200 + 191, OK;
- RunRegistry 280, OK (1 skipped: `test_pipeline_adoption`'s manifest is under the gitignored
  `var/`, absent from an export);
- ComputeTargets 693 (1 skipped: the pre-commit `QuadSourceIntegral.py` regression needs git
  history; and `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, the known wall-clock
  flake, which passed alone).

## 1. Scope

**In scope:**
- `Datastore/SQL/ShardedPool.py`: the constructor's refusals (01); `_read_closed_store` and its
  three callers there (03);
- `Datastore/SQL/schema.py`: the function that derives a drop's dependents (01, §4);
- `Datastore/store_inventory.py`: `StoreInventory.resolve` (02);
- `Datastore/store_reader.py`: `open_read_only`'s call of `_read_closed_store` (03);
- the tests of all of these, under §5 rule 7, and the new ones the prompts add.

**Out of scope:**
- **what the code writes** (§5 rule 9). No table, column, key or stored value changes;
- `config/datastore.py`'s groups, which stay as they are: the refusal names tables, and the
  groups already hold every table it can require (§0.2);
- `main.py`'s `--drop` handling. It passes the tables to the pool as today, and meets the pool's
  refusal;
- the actor's `_drop_tables`, which only the pool calls with real drop tables;
- `a3-v2-readiness`'s `[00-a-drop-action-drops-a-replicated-table-outside-the-in-flight-record]`.
  A drop that names every dependent still drops replicated tables in each actor, outside any
  record. 01 does not change that;
- every store under `var/` (§3).

## 2. Prompts

| # | Prompt | Covers | Model | Status |
|---|---|---|---|---|
| 01 | [`01-refuse-a-drop-that-leaves-references.md`](01-refuse-a-drop-that-leaves-references.md) | §6.1 decision 1. A read-write pool refuses `drop_tables` that leave any other table naming a dropped one, by foreign key or by a declared parent, transitively. The refusal names the dropped tables and every table that must go with them, before the primary is opened. Closes `[06-dropping-one-group-leaves-rows-that-name-what-it-dropped]`. | Opus | **landed** 2026-10-07 (`fee1fbf`) |
| 02 | [`02-resolve-an-absent-parent-as-none.md`](02-resolve-an-absent-parent-as-none.md) | §6.1 decision 2. `resolve()` gives `None` for a key parent whose value is `None`, as `_resolve_reference` does for a member. Closes `[08-resolve-shows-an-absent-nullable-parent-as-unresolved]`. | Sonnet | **landed** 2026-10-07 (`4c34f13`) |
| 03 | [`03-state-each-closed-store-prefix-once.md`](03-state-each-closed-store-prefix-once.md) | §6.1 decision 3. `_read_closed_store` raises the bare reason, and each of its four callers states its prefix once; its `verb` parameter goes. Closes `[04-closed-store-refusals-repeat-their-prefix]`. | Sonnet | **landed** 2026-10-07 (`086c81a`) |

## 3. Datastores

**No prompt opens, writes, copies or retires a store under `var/`.** Tests build stores in
temporary directories, with the stand-in pool (`Datastore/tests/standin_pool.py`) or the
real-store fixtures. No rebuild is needed: no prompt touches the write path (§5 rule 9).

**Before touching anything,** run `python -m RunRegistry list` and report anything `running`.

## 4. The interfaces between prompts

The prompts are independent; these names are fixed so that the board and the index can name them.
- **01 ships** (shipped 2026-10-07, as named) `Datastore.SQL.schema.dependent_tables(dropped,
  factories) -> List[str]`:
  - the tables, other than `dropped`, that name a row of a dropped table, or of one of
    themselves, through a foreign key or a parent their factory's `inventory_spec` declares;
  - transitively, in registry order.

  It also ships `ShardedPool._refuse_drop_that_leaves_references()`, called in the constructor
  immediately after `_refuse_undeclared_drop_tables()`.
- **02 ships** no new name. `StoreInventory.resolve`'s docstring says what a `None` parent gives.
- **03 ships** `ShardedPool._read_closed_store(primary, *, missing_ok=False)`, without `verb`.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`
   (§1.19), plus `datastore-generic`'s board, which owns the issue it closes. The issue closes on
   that board's §4.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue on this
   board.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. A test
   that used a now-refused input only as a way to reach something else is **re-fixtured**, not
   deleted, and the log says which.
8. **Deliberate breakage.** Each prompt names mutations its tests must catch. The log records each
   as a diff, exactly as applied, so the orchestrator can replay it with `git apply`. Mutations are
   never committed.
9. **Nothing the code writes changes.** No table, column, key, constraint or stored value. A prompt
   that finds it cannot do its task without such a change stops and asks.
10. **No test needs Ray, and no test opens anything under `var/`.**
11. **The generic layer gains no project knowledge.** `test_layer_is_generic` (datastore-generic
    09) must pass after every prompt. A refusal names tables it was given or derived, never a
    table by literal.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-10-06), and not reopened here

- **Decision 1 — refuse a drop that would leave references** (not cascade, not document).
  - **What counts as a reference:** a foreign key, **and** a parent a factory's `inventory_spec`
    declares, followed transitively. The user chose this on 2026-10-06, after §0.2's finding that
    foreign keys alone miss the cross-shard and parent-set references.
  - **Rejected:** cascade, which deletes more than was named; documenting the hazard; foreign keys
    only, which leaves `--drop tk-wkb` and `--drop quad-source` dropping rows that
    `QuadSourceIntegral` still names.
- **Decision 2 — render an absent nullable parent as `None`** in `resolve()`, as a parent-set
  member is. **Rejected:** keeping the unresolved form and documenting it.
- **Decision 3 — remove the prefix in `_read_closed_store`**, not in its callers. The callers know
  the operation and the destination, and each already states them.

### 6.2 Choices of method made in writing the prompts

These are the orchestrator's, and are recorded so that the user can reverse any of them.
- **01's refusal is structural**, from the registry's declarations, not from the rows a store
  holds. A dependent table that happens to be empty must still be named. Dropping an empty table
  costs nothing, and a structural refusal gives the same answer for every store.
- **01 refuses in the pool's constructor, before the primary is opened**, beside the
  undeclared-table refusal. It does not repeat the check in the actor. The actor's own
  undeclared-name check exists for an actor built directly, and nothing builds one with real drop
  tables but the pool.
- **01's message names tables, not `--drop` groups.** The groups are the client's vocabulary, which
  the layer must not know (§5 rule 11). Every table it can name lies in some group (§0.2), and
  `config/datastore.py` lists each group's tables.
- **01's derivation lives in `schema.py`**, beside `drop_order`, which orders the same drop.

## 7. Baselines

Re-measure every suite at `HEAD`, clean, before every dispatch, with:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

Run the Datastore suite in four groups of modules (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`,
`[s-z]`), each in the foreground. Last measured at `53d4e18` (datastore-generic 09's review, no
code since): AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry
280, ComputeTargets 693, Datastore 721 (147 + 183 + 200 + 191).

Under load, four tests are known to fail on wall clock or chance:
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, `test_background_tau.test_build_cost`,
`test_numeric_phase_cut.test_the_print_is_gated_both_ways` and LiouvilleGreen's
`test_3bessel_analytic.test_JJJ`. Re-run a failure alone before calling it real.
