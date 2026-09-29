# Orchestrator — prompt 08, `QuadSourceIntegral` keyed on its `QuadSourcePolicy`

Read [`../README.md`](../README.md) first: §0.1, §2, §4, §5, §6.1.2 and §6.2 **D4**, **D5** and
**D6** in full. **You do not write code.** You may *replay* a mutation the log records, with
`git apply`, and then revert it.

**The prompt:** [`08-key-quadsourceintegral-on-its-policy.md`](../08-key-quadsourceintegral-on-its-policy.md)
**Board item:** K3 · **Closes:** `[00-quadsourcepolicy-rows-are-referenced-by-nothing]`
(store-fingerprint) · **Gate:** 07 landed and reviewed (board §1; `a171ced`, review `b9d2b1c`).
Never dispatched concurrently with 09 in this checkout.

## 0. What makes this prompt unusual

The prompt was written on 2026-09-26, before 07 landed. Its task is unchanged. What moved under it
is below, measured at `b9d2b1c`. The orchestrator sketched the schema half in two throwaway
worktrees, since removed:
- the column;
- `build()`, `store()` and `read_batch()` on it;
- the inventory parent;
- the compute target's new argument;
- the fixture's three `QuadSourceIntegral` rows given `quad_policy_serial` 1.

The second worktree also added a second `QuadSourcePolicy` fixture row and the `IDENTITY` entry.
The first also sketched a `read_table` and the shared constant (§0.2). The sketch is not the
agent's to copy, and it was not given to it. It did not touch the refusal of an old store, the
extraction script or any test.

**No computation reads a `QuadSourcePolicy`'s fields.** This is prompt §5's first stop, and it
does not trigger:
- `grep` for `.Levin_threshold` and `.numeric_policy` outside `docs/` finds reads only on
  `GkSourcePolicy` (`ComputeTargets/GkSourcePolicyData.py:250`, `:398`) and in the two policy
  factories' own `build()` filters.
- `main.py` passes `QuadSource_policy_1pt5` and `_5pt0` into `run_pipeline` (`:1021-1022`) and
  uses neither inside it.

**The user decided three things on 2026-09-29, while this note was written.** They are §0.1–§0.3.
Each resolves a place where the prompt, as written, would hit its own §5 stop or would have to do
something unsafe. They bind the agent as the prompt does.

### 0.1 The extraction script refuses through `read_table`, not `object_get` (the user: granted)

Prompt §2 says the script must refuse, naming the parameters, when a store lacks the policy,
rather than insert one. **Refusing from inside `object_get` is unsafe.**
- Every `object_get` of a replicated class runs through `ShardedPool._replicated_write`. That
  function commits a `replication_in_flight` record on the primary *before* the controlling
  shard's call. Its own comment says: "any exception after step 1 leaves the record set".
- So a `QuadSourcePolicy.build()` that raised on a miss would leave the record behind. Prompt 02's
  check at open would then refuse the store the reader was reading.
- So would any other way of refusing from inside the call.

**The grant.**
- A `read_table` method on `Datastore/SQL/ObjectFactories/QuadSourcePolicy.py`.
- One entry, `"QuadSourcePolicy": {…}`, in `config/sharding.py`'s `read_table_config`.
- `ShardedPool.read_table` reads one shard through the actor's `read_table` and writes nothing. The
  `redshift` and `wavenumber` factories' `read_table`s are the precedent.
- Its match must be `build()`'s own: `numeric_policy` equal, and `Levin_threshold` within
  `DEFAULT_FLOAT_PRECISION`. Two stores then cannot disagree about which row is "the" policy.
- The script refuses on **no row**, naming the parameters, and on **more than one**, naming their
  serials.
- **Measured** with a `read_table` written as a list comprehension: `test_factory_select_columns`
  and `test_inventory_retired`, 16 tests, OK, and no pin moved.
  - That is the guard's known blind spot, `[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`.
  - A loop form would add decidable entries to `EXPECTED_COVERAGE`. Either is acceptable; §0.4
    grants the pin growth. Do not choose a shape to stay out of the guard's view.

The script's existing `GkSourcePolicy` lookups (`extract_QuadSourceIntegral_data.py:1062-1075`)
stay `object_get`s. Prompt §2 says to record their insert-on-miss and open a §3 issue. The issue
should also say that even a hit writes and clears an in-flight record on the primary. A reader
that calls `object_get` on a replicated class writes to the store it reads.

### 0.2 The parameters live in one shared constant (the user: shared config constant)

Prompt §5's second stop is "cannot find the policy's parameters without hard-coding them".
- Today the three values exist only as literals in `main.py:3759-3772`:
  `label='policy="maximize-WKB"-Levin-threshold="1.5"'`, `Levin_threshold=1.5` and
  `numeric_policy="maximize-WKB"`.
- The script already copies `GkSourcePolicy`'s literals.

**The grant.**
- One named constant in `config/defaults.py` holding the 1.5 policy's `label`, `Levin_threshold`
  and `numeric_policy` (*agent's choice* of name and shape).
- `main.py`'s `QuadSource_policy_1pt5` creation uses it.
- The script's `read_table` lookup uses it.
- `QuadSource_policy_5pt0` is left as literals, created and unused, as
  `[00-the-5pt0-policy-pair-is-created-and-never-used]` records.
- The `GkSourcePolicy` literals are not touched, in either file.

**Measured** in the sketch, with `pool.object_get("QuadSourcePolicy", **CONSTANT)` and the constant
imported through `main.py`'s existing `from config.defaults import (…)`:
- `test_main_plumbing`: 27 OK, and `EXPECTED_SITES`' `"QuadSourcePolicy": 2` holds, since the
  finder counts calls by their first argument;
- `substitute_main_py()`: all five literals once each;
- the two `scoped_pipeline_run.py` drivers substitute only the k-grid literals, which do not move.

### 0.3 The existing tests that must follow the key (the user: granted)

Prompt §2 names `test_quadsource_integral.py`, the store-inventory `IDENTITY` and the fixture's
`QuadSourceIntegral` rows. Measured, with the schema half, the target's argument and the fixture
rows only:

| Suite | Result | Failing |
|---|---|---|
| ComputeTargets | `failures=1, errors=2` | `test_quadsource_integral`: `TestComputePayload.test_missing_transfer_function_keys_raise_RuntimeError_naming_them`, `TestPersistedSchema.test_object_round_trips_the_new_fields`; and the wall-clock flake, which is not 08's |
| Datastore | `failures=5, errors=4` | `test_integral_read_batch` (4 errors, all `NOT NULL constraint failed: QuadSourceIntegral.quad_policy_serial`); `test_schema_builder`, 2 tests and 2 sub-tests (D5); `test_store_inventory.TestIdentityColumnsMatter.test_the_key_fields_are_the_lookup_columns` (`cls='QuadSourceIntegral'`) |
| RunRegistry | `failures=1` | `test_store_fingerprint.TestGolden.test_the_golden_fingerprint` (D5) |
| AdaptiveLevin, CosmologyModels, LiouvilleGreen | unchanged | — |

Each grant is limited to **following the new key**: the column, or the `QuadSourcePolicy` argument
or payload entry, joins. Nothing else a test asserts may change. Every hunk will be read against
this list.
- **`ComputeTargets/tests/test_quadsource_integral.py`**, named by the prompt. There are two edits,
  not the audit's three: `:1127`, and the shared `common` dict (`:2218-2228`), which both `:2229`
  and `:2236` use.
- **`Datastore/tests/real_store_fixtures.py`.**
  - The three `QuadSourceIntegral` rows (`fixture-qsi-1`, `-2` on shard 0; `-3` on shard 1) take
    `quad_policy_serial` 1.
  - **One new `QuadSourcePolicy` row, serial 2**, beside `qpolicy-a`. It is granted here as
    `STRUCTURALLY REQUIRED`, and §0.5 explains why. The second sketch gave it `label` `qpolicy-b`,
    `Levin_threshold` 0.5 and `numeric_policy` `maximize-WKB`, mirroring `GkSourcePolicy`'s
    `policy-b` (*agent's choice* of values).
  - The fixture's rows are frozen at import, as 07 found.
- **`Datastore/tests/test_store_inventory.py`**, named by the prompt. `IDENTITY["QuadSourceIntegral"]`
  (`:194-203`) gains the new parent, varied to serial 2, beside `"policy"`. With §0.5's row, the
  second sketch's Datastore run no longer failed in `test_store_inventory`.
- **`Datastore/tests/test_integral_read_batch.py` — not named by the prompt; granted by the user.**
  - Its two raw `QuadSourceIntegral` inserts (`:211`, `:292`) gain `quad_policy_serial`.
  - Its two `read_batch` payloads (`:235`, `:307`) gain the policy, as a `_Ref`.
- **The two witnesses, under D5:**
  - `test_schema_builder.py`'s `WITNESS`, and the one docstring sentence naming the current
    witness, which 07's review had to fix after the fact (`d1829bb`);
  - `test_store_fingerprint.py`'s `GOLDEN`, repointed at a new file beside the 07 golden.

### 0.4 The read-back guard's pins may grow, and only by 08's sites

This is the same grant as 07's grant B, given in advance. `EXPECTED_COVERAGE` and
`KNOWN_UNRESOLVED_TABLES` in `test_factory_select_columns.py` may gain exactly the new sites 08
introduces, if any: the `read_table`, and any row read added to the refusal paths. Each is named in
the log with the before and after count. A pin that loses an entry, or moves for any other reason,
is a stop. In the sketch neither moved.

### 0.5 The fixture needs a second `QuadSourcePolicy` — structurally required

- **The reason.** `TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` varies
  one column of one row and requires the key field's values to change. 07 varied `numeric_serial`
  to another real parent on the same shard: Tk numeric 3, on shard 0 beside Tk WKB 2.
- **The fixture has one `QuadSourcePolicy`, serial 1.** Varying to a serial that does not exist
  makes the row an `unresolved-parent` problem. It drops out of the records, and the test passes
  for the wrong reason: it would be testing parent resolution, not identity.
- **Measured** with the second row and the `IDENTITY` entry: Datastore `failures=4, errors=4`,
  exactly §0.3's list less `test_store_inventory`; RunRegistry the golden alone.
- **Its cost is in the fingerprint** (§0.6): `QuadSourcePolicy` moves too.

### 0.6 Everything else that moved

- **The fingerprint format goes from 2 to 3.**
  - The rule is `RunRegistry/stores.py`'s comment above `FINGERPRINT_FORMAT` (as `d1829bb` left
    it): a change to what the inventory records bumps the format and captures a new golden
    "beside the old ones and never over them".
  - 08 changes `QuadSourceIntegral`'s records. Log 07's hand-off says the same.
  - The edit is `FINGERPRINT_FORMAT = 2` → `3`, plus the comment's "Format 2: …" sentence gaining
    a line for format 3. That is granted.
  - The new golden sits beside `full_store_fingerprint_datastore-integrity-07.json`. Neither
    older golden is touched.
  - `store retire` of a store written before 08 must be run at a pre-08 SHA, and the log says so.
- **Measured fingerprint** (at format 2, second sketch; `problems` empty):

  | Class | Count | Tag-set counts | Why |
  |---|---|---|---|
  | `QuadSourceIntegral` | 3 → 3 | [2, 1] → [2, 1] | its key gains `quad_policy` |
  | `QuadSourcePolicy` | 1 → 2 | [1] → [2] | §0.5's fixture row |

  The other 19 classes are unchanged. If the agent's fingerprint moves any other class, that is a
  stop.
- **The witness diff is 6 hunks and 43 lines** in the sketch: 07's six kinds, for one class.
  - The six are the schema record's column list, the column's description, its
    `ForeignKeyConstraint`, the `CREATE TABLE` DDL, the index DDL and the index description.
  - A line outside those six, or in any other class, is the stop. In particular a `read_table`
    or a `read_table_config` entry changes no witness line.
- **The foreign-key check and the coincidence.**
  - The fixture's `QuadSourcePolicy` 1 is also a `GkSourcePolicy` serial (it has 1 and 2), so a key
    pointing at the wrong policy table passes `foreign_key_check` by coincidence. That is how 06
    found `Tq_serial`.
  - §0.5's second row is serial 2, which is a `GkSourcePolicy` serial too.
  - A check of the new key needs rows of the test's own: a `QuadSourceIntegral` whose
    `quad_policy_serial` is a `QuadSourcePolicy` serial and no `GkSourcePolicy` serial, say 3. Its
    docstring says so.
  - It goes in a new module. `test_foreign_key_check.py` (06's) and `test_wkb_parent_foreign_key.py`
    (07's) are not modified.
- **The refusal of an old store needs a new branch in both `build()` and `read_batch()`.**
  - `QuadSourceIntegral`'s factory has no `SQLAlchemyError` handler today. Follow 07's form: match
    the driver's message, `str(getattr(e, "orig", e))`, for `quad_policy_serial`.
  - Unlike 07's WKB value reads, `read_batch` here always filters on the new column, so without its
    own branch a pre-08 store raises a raw `OperationalError`.
  - That is the gap 07 opened as `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`. 08 must not
    open its twin.
  - The message says the store predates the key and is readable at the SHA its sidecar records
    (D4). It does not say "must be regenerated".
- **`_drop_order` does not move.**
  - `QuadSourcePolicy` is in no `_drop_actions` entry, and `quad-source-integral` is already first
    after `1loop-integral`.
  - 07's `TestDropOrder.test_every_declared_foreign_key_between_actions_is_respected` checks the new
    key and must pass unmodified.
- **Every `QuadSourceIntegral` site outside the tests**, at `b9d2b1c`:

  | Site | What it does | After 08 |
  |---|---|---|
  | `main.py:3127-3150` | QSI existence query, vectorized (`"policy": GkSource_policy_1pt5` at `:3132`) | also passes `QuadSource_policy_1pt5` |
  | `main.py:3514-3540` | QSI compute call (`policy=GkSource_policy_1pt5` at `:3518`) | also passes `QuadSource_policy_1pt5` |
  | `main.py:3759-3772` | the two `QuadSourcePolicy` creations | the 1.5 one takes §0.2's constant |
  | `extract_QuadSourceIntegral_data.py:1087-1100` | `object_read_batch("QuadSourceIntegral", …)` | passes the policy it read through §0.1's `read_table` |
  | `docs/handover/quadsource_atol_sweep.py` | `exec`s `main.py` | keyed through `main.py`; its sweep store under `var/` is pre-08 and is now refused (D4). Not edited |

  - `docs/source-remediation-verification/run_quadsource_integrals.py` calls
    `compute_QuadSource_integral` directly, not the target, and does not move.
  - `tools/inventory_report.py`, `docs/gktk-remedial/analyse_scoped_run.py` and the
    `docs/tolerance-convergence/` scripts read raw SQL or table names. None of them moves.
  - `docs/tolerance-convergence/inventory.py:750` cites `MetadataConcepts/QuadSourcePolicy.py:25-36`
    by line. The docstring edit moves those lines. That is an observation, not an edit.
- **`test_main_plumbing.EXPECTED_SITES` must not move** (`"QuadSourceIntegral": 2`,
  `"QuadSourcePolicy": 2`). Adding a payload entry keeps the count. A count that must move is a
  stop.
- **`substitute_main_py()`'s five sites** are `main.py:3582`, `:3689`, `:3690`, `:3791` and `:3803`
  at `b9d2b1c`. Each occurs once, and none is near an 08 site but `:3582`, which 08 does not touch.
- **Audit probes (rule 11).** All fourteen were run on the sketch against `HEAD`:
  - **`docs/datastore-integrity-audit/lookup_keys/probe_01_tagged_read_batch.py` stops running**
    (exit 1: its raw `QuadSourceIntegral` insert lacks the column).
  - `fk_sweep_probe.py`, `probe_02_…` and `probe_03_…` change their output only by the new column,
    line numbers and counts. The other probes are unchanged, including the five that already fail
    at `HEAD`.
  - **The user decided on 2026-09-29 that audit probes are left as they are**: evidence for the
    tree their audit read. So the log **names** `probe_01` in its rule-11 list and does not edit it.
  - 07's log missed its own probe (`probe_05`); the review had to find it.
- **The compute target.** `ComputeTargets/QuadSourceIntegral.py:1740-1756` takes `policy:
  GkSourcePolicy`, checked against the `GkSourcePolicyData` at `:2033`. The new argument sits beside
  it (prompt: *prompt's choice* of name, with the reason). Nothing reads its fields, so nothing
  checks it at compute time.
- **`MetadataConcepts/QuadSourcePolicy.py`.**
  - Its docstring opens "**Nothing reads it.** It is created in `main.py` and threaded through
    `run_pipeline`, but …". It also says the object "stays persisted and threaded so that the schema
    and the run signatures do not churn".
  - Both become false in part. The prompt's amendment covers them: `QuadSourceIntegral` is keyed on
    it, and no computation reads its fields yet.
  - The two bullets explaining why neither field is read stay as they are.
- **The index.** 98 rows at `b9d2b1c`. 08 closes one and opens at least one (§0.1's issue). Rows
  are counted, not computed.

**Carried from the 07 review.**
- **Another session may write in this checkout.** 07's run saw `docs/derivation/` appear
  mid-dispatch. So:
  - stage by explicit path only, never `git add -A`, `.` or `commit -a`;
  - show `git diff --cached --name-only` before committing;
  - stop and ask if another session modifies a tracked file the agent needs.
- **Count, then paste.** The witness diff by kind, and the pins by name. Paste `-v` output
  unfiltered.
- **A report states only what the diff shows.** Every claim is checked against `git show`.
- **Mutations.**
  - A mutation record lists every failing test, with counts, and says where a failing test has
    sub-tests.
  - The replay is done twice: before the commit from the final log file, and after it from
    `git show HEAD:<log>`, reported with no amend.
  - The known flake, `test_tk_wkb_phase.test_wall_time_per_object`, fires under load. It fired once
    in each of the review's and the sketch's runs. It is re-run alone before it is called real.
- **Stale prose next to a granted line.** 07's grants covered one line each in two files whose
  comments then went stale. §0.3 and §0.6 grant the adjacent sentence this time.
- **`TestSchemaIsUnchanged` keeps its name** (the 06 review).
- **Residue.** The agent still follows README §5 rule 4, recording and not fixing. Small residue
  found in review is the orchestrator's, in a follow-up commit (the user, 2026-09-29).
- **Scratch files.** The agent works in its own scratch subdirectory, not the session
  scratchpad's top level.

## 1. Before you dispatch

1. Record the branch and `HEAD`. It should be the commit that adds this note, or a later one that
   touches no code. Say which.
2. `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows this
   checkout alone. If another session has uncommitted work here, do not dispatch; ask the user.
3. `python -m RunRegistry list`: nothing `running`.
4. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing (SHA-256, size and mtime of every file; the directory list), and never put it in the
   tree. The 07 session's snapshot held 46 files and 10 directories.
5. Baselines, board §5.10: AdaptiveLevin 32, ComputeTargets 566, CosmologyModels 39, Datastore 397,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially.
6. Run the `substitute_main_py()` count on `HEAD` (each of the five literals occurs once), so the
   agent's after-count has a before.
7. Count the index rows (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`), and tell the agent the number.

## 2. Dispatch

One fresh-context subagent (**Sonnet**, as the prompt recommends). Give it:
- the prompt;
- the README, with 6.1.2, D4, D5 and D6;
- the audit, `docs/datastore-integrity-audit.md` §2.5 and §2.6;
- the issue's entry on the store-fingerprint board (`IMPLEMENTATION_STATE.md` §3, with its
  **Assigned** line);
- log 07's "State handed to the next prompt" (**For 08**), and log 07's D5 sections as the worked
  example;
- `HEAD`, the baselines and the index count;
- §0 of this note, **lines up to "## 1. Before you dispatch" only**, including the user's three
  decisions.

Tell it plainly:
- **one commit**, with its log at `logs/08-key-quadsourceintegral-on-its-policy.md`, in README
  §5.1's form, with prompt 08 §7's additions:
  - §0's evidence that no computation reads the policy's fields, quoted;
  - every site changed, in each file, before and after (§0.6's table);
  - the witness diff **in full**, pasted from `diff`, with its hunks counted by the six kinds;
  - the fingerprint re-capture: the bump to 3, which classes moved and why (§0.6's table), and that
    no other did;
  - where the extraction script takes the policy from (§0.2), and how it refuses (§0.1);
  - the reader list, and every script that now refuses a store it read before (rule 11), with
    `probe_01` and the sweep's store named;
  - the `substitute_main_py()` output;
  - each existing-test edit, by §0.3's list, with the one-line reason;
- in the same commit:
  - the board: its §1 row, K3, §3.1, §3.3 for any issue opened, and a new §5 subsection
    `5.11 After prompt 08`;
  - the store-fingerprint board: the issue moves to its §4, with a closing note naming this commit
    by subject and the user's decision (6.1.2, D6);
  - `docs/OPEN_ISSUES.md`: its row deleted, any opened row added, and the count and date corrected
    by counting;
  - the campaign board's header status line (07's review had to fix it after the fact);
- **the files it may change:**
  - `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`: `register`, `build`, `store`,
    `read_batch`, `inventory_records` and their comments;
  - `ComputeTargets/QuadSourceIntegral.py`: the constructor and what carries the policy;
  - `MetadataConcepts/QuadSourcePolicy.py`: the docstring only;
  - `Datastore/SQL/ObjectFactories/QuadSourcePolicy.py`: a new `read_table` only (§0.1);
  - `config/sharding.py`: the one `read_table_config` entry (§0.1);
  - `config/defaults.py`: the one constant (§0.2);
  - `main.py`: the three sites of §0.6's table, and the import of the constant;
  - `extract_QuadSourceIntegral_data.py`: its `QuadSourcePolicy` lookup and its `read_batch` call;
  - `RunRegistry/stores.py`: `FINGERPRINT_FORMAT` and its comment (§0.6);
  - the existing tests of §0.3, each as §0.3 bounds it, and the guard's pins as §0.4 bounds them;
  - one new witness, `Datastore/tests/data/schema_at_datastore-integrity-08.json`, captured with
    `schema_description.py`'s `__main__`;
  - one new golden fingerprint beside `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-07.json`,
    in its pattern;
  - new test modules under `Datastore/tests/`, `ComputeTargets/tests/` or `RunRegistry/tests/`;
  - the log, the two boards and the index.

  **Nothing else:**
  - not any existing witness or golden, which are never overwritten;
  - not `schema.py`, `ShardedPool.py` or `Datastore.py`;
  - not the `GkSourcePolicy` class, factory or lookups, and not the 5.0 policies;
  - not `QuadSource`'s, `OneLoopIntegral`'s or `GkSource`'s keys, and not `QuadSourceIntegral`'s
    `source_serial`, `data_serial` or Tk parents (09);
  - not `test_foreign_key_check.py`, `test_wkb_parent_foreign_key.py`, `test_main_plumbing.py`, or
    any other existing test;
  - not `docs/handover/quadsource_atol_sweep.py`, any audit probe, or anything in `orchestrator/`;
- **no Ray, and nothing opens anything under `var/`**, or a copy of anything under it. Every store is
  in a temporary directory;
- the foreign-key check of the new key uses §0.6's rows and says so in its docstring;
- every mutation diff is pasted from `git diff`, and replayed twice as §0 says;
- run `black`;
- prompt 08 §5's stop conditions mean *stop and ask*. So do:
  - an existing test outside §0.3's list failing;
  - a fingerprint class moving other than `QuadSourceIntegral` and `QuadSourcePolicy`;
  - a witness diff line outside the six kinds for `QuadSourceIntegral`;
  - `EXPECTED_SITES` having to move;
  - a guard pin moving other than as §0.4 allows.

## 3. The review

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's files.
   - `git diff HEAD~1 HEAD -- Datastore/SQL/schema.py Datastore/SQL/ShardedPool.py Datastore/SQL/Datastore.py Datastore/tests/data/schema_at_base.json Datastore/tests/data/schema_at_datastore-integrity-06.json Datastore/tests/data/schema_at_datastore-integrity-07.json RunRegistry/tests/data/full_store_fingerprint.json RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-07.json Datastore/tests/test_foreign_key_check.py Datastore/tests/test_wkb_parent_foreign_key.py ComputeTargets/tests/test_main_plumbing.py docs/handover/ docs/datastore-integrity-audit/ prompts/datastore-integrity/orchestrator/`
     is empty.
   - `--diff-filter=M` over `*/tests/*` shows exactly §0.3's files, plus `test_factory_select_columns.py`
     if §0.4 applied. Read every hunk against §0.3: each follows the key, and asserts nothing new or
     less.
   - `QuadSourcePolicy.py`'s factory hunk is `read_table` alone. `config/sharding.py`'s is one
     entry, `config/defaults.py`'s is one constant, and `MetadataConcepts/QuadSourcePolicy.py`'s is
     docstring alone. `stores.py`'s is the constant and its comment.
2. **K3, the schema.** `quad_policy_serial` is NOT NULL, indexed, with
   `ForeignKey("QuadSourcePolicy.serial")`. `policy_serial` is unchanged.
3. **K3, the lookups.**
   - `build()` and `read_batch()` both filter on it, and `read_batch` passes the policy back to each
     object.
   - `store()` writes it, and the constructor requires it.
   - `inventory_records` declares `Parent("quad_policy_serial", "QuadSourcePolicy")`, and its
     comment no longer says "not a QuadSourcePolicy".
   - A pre-08 table is refused with the D4 message by **both** `build()` and `read_batch()`. Check
     this with a table recreated from the 07 witness's DDL, as 07's tests did.
4. **`main.py`.**
   - Both QSI sites pass `QuadSource_policy_1pt5`, and the 1.5 creation takes the constant.
   - `_5pt0` is untouched.
   - `EXPECTED_SITES` passes unmodified.
   - Run the five `substitute_main_py()` counts yourself.
5. **The extraction script.**
   - It reads the policy through `read_table`, with `build()`'s criterion, and refuses on none and
     on more than one.
   - It never calls `object_get("QuadSourcePolicy", …)`.
   - Its `GkSourcePolicy` lookups are unchanged, with their §3 issue open.
   - Grep the tree for any `QuadSourceIntegral` lookup or construction that passes no
     `QuadSourcePolicy`.
6. **The refusal is safe.** Show, on a stand-in or temporary store, that the script's refusal leaves
   no `replication_in_flight` row on the primary. That was the reason for §0.1.
7. **D5 and the fingerprint.**
   - Both new files are new, and every older witness and golden is byte-identical at both commits.
   - Re-capture the schema witness from `HEAD` into the scratchpad: byte-identical.
   - Diff the 07 and 08 witnesses yourself. Expect 6 hunks and about 43 lines, and match them to
     the log's paste.
   - Re-capture the fingerprint of `build_full_store()` from `HEAD`: it equals the new golden, with
     `fingerprint_format` 3, and only `QuadSourceIntegral` and `QuadSourcePolicy` differ from the
     07 golden.
8. **The foreign-key test** uses a `QuadSourcePolicy` serial that is no `GkSourcePolicy` serial.
   With the parent's factory checked out over it, it fails; restore.
9. **Replay the mutations** (i)–(iv) from the committed log with plain `git apply`, in a worktree
   of `HEAD`. Each gives its logged failures. Revert each, leaving the tree clean.
10. **Readers and probes.** Run all fourteen audit probes at `HEAD~1` and `HEAD`. Only `probe_01`
    should newly fail, and the log's rule-11 list must name it.
11. **Boards.**
    - The issue is on the store-fingerprint board's §4, with a closing note naming this commit.
    - Count the index rows, and check the date.
    - The campaign board has its header, §1 row, K3, §3.1, any §3.3 entry and §5.11.
12. **`var/`, suites.**
    - The snapshot is identical.
    - Run all six suites serially. Each rises by exactly the tests added, and nothing else moves.
    - `black --check` is clean.
    - `ray.is_initialized()` is false after each new module.

## 4. After the review

Record it on the board, after the review of 07. Any small residue the review finds is fixed in a
follow-up commit of its own (the user, 2026-09-29).

Then:
- **09 is next** (README §2), with its own orchestrator note carrying this review's findings.
  - It keys `QuadSource` on `Tq_serial` and `Tr_serial`, `QuadSourceIntegral` on its source,
    policy data and Tk rows, and `GkSource` on its Gk rows (K4–K7). It resolves cross-shard parents
    in the inventory.
  - It is the natural owner of `[07-wkb-value-reads-do-not-refuse-a-pre-07-store]`. After 09, the
    GkSource stage's refusal of a second Gk WKB row (07's log, §0.3 of the 07 note) has its
    keyed replacement.
  - Its record change is another format bump.
- Prompt 10 remains the first real run.
