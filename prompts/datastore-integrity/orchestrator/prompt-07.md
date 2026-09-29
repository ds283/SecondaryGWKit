# Orchestrator — prompt 07, the WKB rows keyed on their numeric row

Read [`../README.md`](../README.md) first: §0.1, §2, §4, §5 and §6.2 **D1**, **D4** and **D5** in
full. **You do not write code.** You may *replay* a mutation the log records, with `git apply`,
and then revert it.

**The prompt:** [`07-key-wkb-rows-on-their-numeric-row.md`](../07-key-wkb-rows-on-their-numeric-row.md)
**Board items:** K1–K2 · **Closes:** `[20-wkb-rows-consume-numeric-initial-data]` (GkTk-remedial)
· **Gate:** 05 (as 05a–05c) and 06 landed and reviewed (board §1, `b9713ea`). Never dispatched
concurrently with 08–09 in this checkout.

## 0. What makes this prompt unusual

The prompt was written on 2026-09-26, before 05c and 06 landed. Its task is unchanged. What moved
under it is below, measured at `b9713ea`. The orchestrator sketched the schema half in a throwaway
worktree, since removed:
- the two columns;
- `store()` writing `numeric_serial`;
- `inventory_records` with the new parent and without `z_init`;
- the fixture's rows given parents;
- a nullable `Parent`.

The sketch is not the agent's to copy, and was not given to it. It did not touch `build()`,
`main.py`, the readers or the consistency check.

**The user decided four things on 2026-09-29, while this note was written.** They are §0.1–§0.4.
Each resolves a place where the prompt, as written, would hit its own §5 stop or would silently
change a reader. They bind the agent as the prompt does.

### 0.1 The existing tests that must follow the key (the user: granted)

The prompt's §5 stops on "an existing test other than those this prompt names". It names none,
because it predates the tests below. **Measured**: with the column alone, and nothing else,
Datastore gave `failures=56, errors=18`, and RunRegistry `failures=4`. With the fixture given
parents and `Parent` made nullable (§0.5), it dropped to `failures=11, errors=16` and
`failures=1`. What remained was exactly the list below, plus the two D5 witnesses.
ComputeTargets 552 and CosmologyModels 39 were unchanged.

Each grant is limited to **following the new key**: `z_init` leaves it, the numeric parent joins
it. Nothing else a test asserts may change. Every hunk will be read against this list.
- **`Datastore/tests/real_store_fixtures.py`, the default WKB rows.** Each needs a
  `numeric_serial`, and they are frozen into `FULL_SHARDED_ROWS` at import (`:998`).
  - Tk WKB 1 and 2 (shard 0, exit 1) take Tk numeric 1. Tk WKB 3 (shard 1, exit 2) takes Tk
    numeric 2.
  - The two Gk WKB rows are at `z_source` 2, and the only Gk numeric rows are at `z_source` 1. So
    they are **pure WKB**: `numeric_serial` is NULL. That is what the data says, and it is the
    case §0.5 exists for.
  - Do not "correct" the rows' other values. They were never consistent with their parents
    (`T_init` 1.0 against `stop_T` 0.5; `stop_deltaz_subh` NULL). The fixture writes raw rows,
    and a check in `store()` is not reached by it.
- **`Datastore/tests/test_store_inventory.py`.**
  - The two WKB entries of `IDENTITY` (`:153-178`): `z_init` out, the numeric parent in. Put
    `z_init` into `NON_IDENTITY` beside `T_init` (`:230`).
  - `test_a_missing_key_column_makes_the_class_incomplete` (`:875`) uses `z_init` as the WKB key
    column that goes missing. It becomes `numeric_serial`.
- **Raw-insert helpers that write Tk WKB rows** without the new NOT NULL column. Each gains a
  `numeric_serial`, naming a numeric row the helper also writes where the test has one:
  - `test_read_path_mechanics.py` `_tk_wkb_parent` (`:174`). Its callers are
    `TestTagsOmittedDefaultToEmpty` (9 errors), `TestTkWKBValueFriction` (2) and
    `TestWKBCriterionSpelling` (1).
  - `test_value_reads_parent_key.py`'s `_tk_wkb` (`:264`).
- **05c's `z_init` tests**, which 05c's own log hands to 07 ("will need to follow 07's key"):
  - `test_value_reads_parent_key.py` `TestZInitSelects` (`:666`);
  - `TestSecondParentIsRefused.test_gk_wkb_value_two_z_inits` and `test_tk_wkb_value_two_z_inits`
    (`:617`, `:641`).

  Each becomes the same test with two numeric parents in place of two `z_init`s.
- **`ComputeTargets/tests/test_tk_wkb_phase.py`**, the construction at `:256`. It builds a
  `TkWKBIntegration` with no parent. The prompt makes that a construction-time error, and the user
  kept that. The helper gains a parent. Nothing about the phase or the wall time changes.
- **The two witnesses, under D5:**
  - `test_schema_builder.py`'s `WITNESS`;
  - `test_store_fingerprint.py`'s `GOLDEN` (`:60`), repointed at a new file beside
    `full_store_fingerprint.json`.

### 0.2 The fingerprint format is bumped (the user: bump to 2)

- **The rule.** `RunRegistry/stores.py:194-196`: "a change to what the inventory records, or to how
  a digest is formed, bumps this and regenerates the golden fingerprint … in the same commit". The
  store-fingerprint log 04 (`:709`) says the same.
- **Why it applies.** 07 changes both WKB classes' records.
- **The edit.** `FINGERPRINT_FORMAT = 1` becomes `2`, a one-line change to `RunRegistry/stores.py`,
  granted here.
- **The golden.** A new file beside the old one: never overwrite `full_store_fingerprint.json`.
  `GOLDEN` is repointed at it. The test asserts `golden["fingerprint_format"] ==
  FINGERPRINT_FORMAT`, so the old golden cannot pass after this, and that is expected.
- **Its effect.** A store's recorded format-1 fingerprint is now reported by `compare_fingerprints`
  as "different formats are not compared" (`stores.py:1258-1270`), not as a content difference.
  That is D4's statement, made by the fingerprint.
  - So `store retire` of any existing store must be run at a pre-07 SHA. The log says so.
- **Measured, at format 1 in the sketch.** Only `TkWKBIntegration` and `GkWKBIntegration` moved.
  - Their counts (3, 2) and tag-set counts ([2, 1], [1, 1]) did not move, and nothing appeared
    in `problems`.
  - No downstream class moved, because nothing keys on a WKB row's digest.
  - If the agent's fingerprint moves any other class, that is a stop.

### 0.3 A Gk lookup says which parent it means (the user: an explicit marker)

The prompt says a Gk `build()` with no parent means pure WKB (`IS NULL`). But
`extract_GkWKB_data.py:469` passes no parent, since it holds only an unresolved `GkN_ref`. It would
silently start seeing pure-WKB rows only. So:
- **Keyed.** `build()` takes a numeric parent, or an explicit pure-WKB marker (*agent's choice* of
  name), and filters `numeric_serial == parent` or `IS NULL`. **A Gk `build()` given neither
  raises**, as a Tk `build()` without a parent does.
- **Readers.** A labels-only mode, requested explicitly and never by omission, is what D1 allows
  readers: the physical labels only, **refusing when more than one stored row matches** and naming
  their `numeric_serial` values (a NULL named as pure WKB). It is for readers only.
  - `main.py` never uses it; every `main.py` site is keyed.
  - For Tk it is likewise explicit. "A Tk lookup without a parent is an error" holds for any
    payload that asks for neither.
- **Mutation (ii)** becomes "the pure-WKB marker matches any parent instead of `IS NULL`".
- **The value reads.** 05c's hand-off names four filters: the optional `z_init` in
  `GkWKBValue._build_impl_model` and `read_batch`, and in `TkWKBValue._build_impl_model` and
  `read_batch`. Each is replaced by an optional numeric parent (Gk: parent, or pure-WKB marker).
  With none given, a value read spans every parent, and 05c's refusal of a second parent
  (`wkb_serial`) guards it. That refusal stays.
  - The GkSource stage (`main.py:2396-2448`) reads every `z_source` with no parent. A second Gk
    WKB row for one `(k, z_source)` is now possible: a resume under a changed numeric row
    computes one where the old existence query served the stale one. That stage will then refuse
    loudly. **That is correct until 09** keys `GkSource` on its Gk rows. The log says so.

### 0.4 The Gk seeded-or-pure choice moves ahead of the existence query (the user: within scope)

`main.py:2182`: `if z_source.z >= z_source_limit and Gk.available:` seeds from the numeric row.
Otherwise the row is pure WKB (`:2225-2268`: `G_init=0.0`, `Gprime_init=1.0`, no `z_init`).
- So the pure-WKB branch runs in two cases:
  - no numeric row exists;
  - **one exists, but `z_source` is below `z_source_limit`** (`:2152`: the geometric mean of the
    3- and 4-e-fold points).
- A Gk WKB row lacks a parent only on this branch, so prompt §5's first stop does not trigger. The
  log must quote `:2152` and `:2182` as the evidence.
- **The consequence.** To pass the right parent, or the marker, to the existence query, the
  numeric lookup and this choice both move ahead of the query. The user ruled that part of
  "looking the numeric row up first", not a restructuring. The choice moves **unchanged**.
- The agent may lift it into a module-level function, so that
  `test_main_plumbing.load_main_py_functions` can test it (*agent's choice*).
- The two `build_*_WKB_work`s are closures inside `run_pipeline` (`:1407`, `:2016`), which that
  loader cannot reach.
- Anything beyond moving the choice unchanged is still §5's stop.

### 0.5 The inventory cannot resolve a NULL parent — structurally required

- **The defect.** `Parent` has no NULL form. In `read_records` (`Datastore/store_inventory.py`,
  the loop after `types = cosmology_classes()`), `context.digests[of].get(None)` is `None`, so the
  row is appended to `unresolved` and becomes an `unresolved-parent` problem, dropped from the
  records.
- **Measured.** With `Parent("numeric_serial", "GkNumericIntegration")` and the fixture's two
  pure-WKB rows:
  - every Gk WKB record vanished;
  - `test_store_inventory` failed across nine classes of test;
  - `test_inventory_report` failed 9 times.
- **The grant.** `Parent` gains a nullable form (*agent's choice* of spelling; the sketch used
  `nullable: bool = False`), in which a NULL column gives the key field `None`. It is granted here
  as `STRUCTURALLY REQUIRED`: the prompt's "`Parent("numeric_serial", …)`" cannot be met on a
  nullable column without it, and README §1 puts `store_inventory.py` in scope for 07–09.
- **Its limit.** The nullable form and nothing else in `store_inventory.py`. A non-nullable
  `Parent` whose column is NULL must still be an `unresolved-parent` problem, and a test says so.

### 0.6 Everything else that moved

- **`_drop_order`** (`Datastore/SQL/Datastore.py:149-160`) is half right already. `gk-wkb` (`:155`)
  drops before `gk-numeric` (`:156`). Only the Tk pair is wrong: `tk-numeric` (`:158`) before
  `tk-wkb` (`:159`).
  - 06's `QuadSource.Tq_serial` key is already respected, since `quad-source` (`:157`) drops
    before `tk-numeric`.
  - The comment at `:148` claims the order respects the foreign keys. A test that derives the
    order from the schema's own foreign keys, over every drop action, would hold 08 and 09 to it
    too (*agent's choice*, beside the prompt's test 7).
- **The old-store refusal.** Follow the `rho_gauss_order` form in these same factories
  (`TkWKBIntegration.py:311-327`). It matches the **driver's** message,
  `str(getattr(e, "orig", e))`.
  - Not `TkNumericIntegration`'s `break_point_kind` form (`:290`), which matches `str(e)`. That
    string includes the SQL statement, which names the column either way.
  - Today's handler re-raises anything without `rho_gauss_order` in it, so a store lacking only
    `numeric_serial` would escape as a raw `OperationalError`. It needs its own branch.
  - The message says the store predates the key and is readable at the SHA its sidecar records
    (D4). It does not say "must be regenerated".
- **The consistency check.**
  - `z_init` is **derived**, not stored on the parent: `k_exit.z_exit - Tk.stop_deltaz_subh`
    (`main.py:1535`; Gk `:2186`), with `T_init = Tk.stop_T` and `Tprime_init = Tk.stop_Tprime`.
  - The parent's `stop_*` are row columns, returned even under `_do_not_populate`
    (`TkNumericIntegration.py:223-225`, `:406-408`; Gk `:218-220`, `:404-406`). So a
    serial-only lookup still carries them.
  - A check in `store()` can read the parent row on its own shard and recompute the same
    expression, bit for bit. A caller cannot skip it there.
  - It should also check that the parent is for the same `k`, and for Gk the same `z_source`. For
    a pure-WKB row, `G_init == 0`, `Gprime_init == 1`, and `z_init` is `z_source.z`
    (`ComputeTargets/GkWKBIntegration.py:196-200`).
- **Every WKB lookup and construction outside the tests**, at `b9713ea`:

  | Site | What it does | After 07 |
  |---|---|---|
  | `main.py:1416-1452` | Tk existence query, no `z_init` | keyed; the numeric lookup (`:1466-1513`) moves ahead of it, over the whole batch |
  | `main.py:1539-1560` | Tk construction from the stop point | passes the parent |
  | `main.py:2025-2069` | Gk existence query, vectorized | keyed, parent or marker, by §0.4's choice |
  | `main.py:2200-2226`, `:2248-2268` | Gk seeded and pure constructions | parent; marker |
  | `main.py:3342-3377` | QSI stage Tk WKB lookup | keyed on the numeric row from `:3340`, which `build_QuadSourceIntegral_payload` (`:378-381`) consumes beside it |
  | `extract_TkWKB_data.py:459` | labels, beside an unresolved `TkNumeric_ref` | resolve the numeric first and key on it, or the reader mode (*agent's choice*) |
  | `extract_GkWKB_data.py:469` | labels and `z_source` | the reader mode, since a row may be pure WKB |
  | `docs/source-remediation-verification/run_quadsource_integrals.py:318-328` | Tk WKB beside the numeric it just looked up | keyed on that numeric; **granted**, because the user does not want a verification driver that crashes (2026-09-29), and it would, under "a Tk lookup without a parent is an error" |

  - `analyse_greens_and_source.py`, `docs/gktk-remedial/analyse_scoped_run.py`,
    `tools/inventory_report.py` and `order_audit.py` read raw SQL or table names. None of them
    moves.
- **`test_main_plumbing.EXPECTED_SITES` must not move.** It pins TkNumeric 5, GkNumeric 3, TkWKB 3
  and GkWKB 3 sites. Moving a lookup keeps the count; adding one does not. A count that must move
  is a stop.
- **`substitute_main_py()`'s five sites** are `main.py:3496`, `:3603`, `:3604`, `:3705` and
  `:3717`, all outside the WKB stages, and each occurs once at `b9713ea`.
- **Shard evidence, for the log.** `config/sharding.py:20-29` shards `TkNumericIntegration`,
  `TkWKBIntegration`, `GkNumericIntegration` and `GkWKBIntegration` all on `"k"`. The log quotes
  the lines.
- **The foreign-key check (06's S4) and the coincidence.**
  - On the fixture's own shard 0, `numeric_serial` 1 is both a Tk numeric serial and a Tk WKB
    serial. So a key pointing at the wrong table passes `foreign_key_check` by coincidence, as
    `Tq_serial` did.
  - A check of the new keys needs rows of the test's own: a `numeric_serial` that is a numeric
    serial on its shard and no WKB serial there.
  - Gk needs a seeded row, which the fixture has none of: a Gk numeric at the WKB row's
    `z_source`, added in the test's own rows.
  - Put this in a new module. `test_foreign_key_check.py` is 06's and is not modified.
- **The witness diff is 12 hunks and 86 lines** in the sketch, six hunks per class, because this is
  a new, indexed column and not only a key. The six are the schema record's column list, the
  column's description, its `ForeignKeyConstraint`, the `CREATE TABLE` DDL, the index DDL, and the
  index description. The column's position in `register()` moves the line numbers, not the counts.
  A line outside those six, for those two classes, is the stop.
- **Old stores.** The three stores under `var/` have no `numeric_serial`. After 07 the tree
  refuses to look up their WKB rows (D4), and their recorded fingerprints are another format
  (§0.2). Nothing under `var/` is opened to show it.
- **The index.** 96 rows at `b9713ea`. 07 closes one, plus any it opens. Rows are counted, not
  computed.
  - Another session has an uncommitted edit to `docs/OPEN_ISSUES.md` (97 rows, "eighteen
    campaigns") and two untracked directories, `docs/one-loop-planning/` and
    `prompts/perturbation-sound-speed/`.
  - **07 is not dispatched until that work is committed**, since 07 edits the same index.

**Carried from the 06 review.**
- **Count, then paste.**
  - The 06 note's "9 hunks" was its own miscount. The agent counts the witness diff by category
    and pastes it.
  - Paste `-v` output unfiltered. A test with a docstring prints its status on a second line, and
    06's paste lost three.
- **A report states only what the diff shows.** 06's agent reported correcting a count that its
  commit had not corrected. Every claim in the final report is checked against `git show`.
- **A log names only what it contains**, and gets its counts of files and boards right.
- **Mutations.**
  - A mutation record lists every failing test, with counts, and says where a failing test has
    sub-tests.
  - The replay is done twice: before the commit from the final log file, and after it from
    `git show HEAD:<log>`, reported with no amend.
  - A diff whose trailing context does not survive markdown is recorded with `-U1`, and the log
    says so.
- **`TestSchemaIsUnchanged` keeps its name** (the 06 review).
- **Residue.** The agent still follows README §5 rule 4, recording and not fixing. Small residue
  found in review is the orchestrator's, in a follow-up commit (the user, 2026-09-29).
- **Scratch files.** The agent works in its own scratch subdirectory, not the session
  scratchpad's top level.

## 1. Before you dispatch

1. Record the branch and `HEAD`. It must be the commit that adds this note.
2. `git status --short` and `git diff --cached` are both empty, and `git worktree list` shows this
   checkout alone. **In particular the other session's `docs/OPEN_ISSUES.md` edit and its two
   untracked directories are committed or gone.** If they are not, do not dispatch; ask the user.
3. `python -m RunRegistry list`: nothing `running`.
4. Snapshot `var/datastores/`, `var/runs/` and the root `physics-test-n20-lambdacdm-zend0p1*.sqlite`
   read-only, with `orch_04_snapshot.py` in the session scratchpad. Write it again if it is
   missing (SHA-256, size and mtime of every file; the directory list), and never put it in the
   tree. The 06 session's snapshot held 46 files and 10 directories.
5. Baselines, board §5.9: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 351,
   LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`, serially.
6. Run the `substitute_main_py()` count on `HEAD` (each of the five literals occurs once), so the
   agent's after-count has a before.

## 2. Dispatch

One fresh-context subagent (**Opus**, as the prompt recommends). Give it:
- the prompt;
- the README, with D1, D4 and D5;
- the audit, `docs/datastore-integrity-audit.md` §2.3, §2.4, §2.6 and §3.13;
- the issue's entry on the GkTk-remedial board (`IMPLEMENTATION_STATE.md` §3, with its
  **Assigned** line);
- log 05c's "State handed to the next prompt" (**For 07**) and log 06's "State handed to the next
  prompt";
- `HEAD` and the baselines;
- §0 of this note, including the user's four decisions.

Tell it plainly:
- **one commit**, with its log at `logs/07-key-wkb-rows-on-their-numeric-row.md`, in README §5.1's
  form, with prompt 07 §7's additions:
  - the shard evidence, quoted;
  - §0.4's evidence that the pure-WKB branch is the only parentless one, quoted;
  - every `main.py` site changed, before and after;
  - the witness diff **in full**, pasted from `diff`, with its hunks counted by §0.6's six kinds;
  - the fingerprint re-capture: the format bump, which classes moved and why, and that no other
    did;
  - the reader list, by §0.6's table, and every script that now refuses a store it read before
    (rule 11);
  - the `substitute_main_py()` output;
  - each existing-test edit, by §0.1's list, with the one-line reason;
- in the same commit:
  - the board: its §1 row, K1–K2, §3.1, and a new §5 subsection `5.10 After prompt 07`;
  - the GkTk-remedial board: the issue moves to its §4, with a closing note naming this commit by
    subject;
  - `docs/OPEN_ISSUES.md`: its row deleted, and the count and date corrected;
- **the files it may change:**
  - the two WKB factories, `Datastore/SQL/ObjectFactories/TkWKBIntegration.py` and
    `GkWKBIntegration.py`: `register`, `build`, `store`, `inventory_records`, the value factories'
    `_build_impl_model` and `read_batch` (§0.3), and the `validate` paths only if the new column
    requires it;
  - the two compute targets, `ComputeTargets/TkWKBIntegration.py` and `GkWKBIntegration.py`: the
    constructors and what carries the parent;
  - `main.py`: the sites of §0.6's table, and at most one new module-level helper (§0.4);
  - `extract_TkWKB_data.py`, `extract_GkWKB_data.py` and
    `docs/source-remediation-verification/run_quadsource_integrals.py`: their WKB lookups;
  - `Datastore/SQL/Datastore.py`: `_drop_order` and its comment, nothing else;
  - `Datastore/store_inventory.py`: `Parent`'s nullable form and its use in `read_records`
    (§0.5);
  - `RunRegistry/stores.py`: `FINGERPRINT_FORMAT` (§0.2);
  - the existing tests of §0.1, each as §0.1 bounds it;
  - one new witness, `Datastore/tests/data/schema_at_datastore-integrity-07.json`, captured with
    `schema_description.py`'s `__main__`;
  - one new golden fingerprint beside `RunRegistry/tests/data/full_store_fingerprint.json`
    (*agent's choice* of name, in D5's pattern);
  - new test modules under `Datastore/tests/`, `ComputeTargets/tests/` or `RunRegistry/tests/`;
  - the log, the two boards and the index.

  **Nothing else:**
  - not `schema_at_base.json`, `schema_at_datastore-integrity-06.json` or
    `full_store_fingerprint.json`, which are never overwritten;
  - not `schema.py` or `ShardedPool.py`;
  - not the numeric integrations' factories or targets, and not how the stop point is found;
  - not `GkSource`, `QuadSource` or `QuadSourceIntegral`'s keys (09);
  - not `test_foreign_key_check.py`, and no other existing test;
  - not `docs/handover/quadsource_atol_sweep.py`, any audit probe, or anything in `orchestrator/`;
- **no Ray, and nothing opens anything under `var/`**, or a copy of anything under it. Every store is
  in a temporary directory;
- the foreign-key check of the new keys needs the rows §0.6 describes, and says so in its
  docstring;
- every mutation diff is pasted from `git diff`, and replayed twice as §0 says;
- run `black`;
- prompt 07 §5's stop conditions, as §0.4 reads the third, mean *stop and ask*. So do:
  - an existing test outside §0.1's list failing;
  - a fingerprint class moving other than the two WKB classes;
  - a witness diff line outside §0.6's six kinds;
  - `EXPECTED_SITES` having to move.

## 3. The review

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only §2's files.
   - `git diff HEAD~1 HEAD -- Datastore/SQL/schema.py Datastore/SQL/ShardedPool.py Datastore/tests/data/schema_at_base.json Datastore/tests/data/schema_at_datastore-integrity-06.json RunRegistry/tests/data/full_store_fingerprint.json Datastore/tests/test_foreign_key_check.py docs/handover/ prompts/datastore-integrity/orchestrator/`
     is empty.
   - `--diff-filter=M` over `*/tests/*` shows exactly §0.1's files. Read every hunk against §0.1:
     each follows the key, and asserts nothing new or less.
   - `Datastore.py`'s hunk is `_drop_order` and its comment. `store_inventory.py`'s is §0.5's
     nullable form. `stores.py`'s is the one constant.
2. **K1, the schema.** `numeric_serial` is indexed, with its foreign key. It is NOT NULL on Tk and
   nullable on Gk. `z_init` and the initial values remain stored and are in no filter of `build()`.
3. **K1, the lookups.**
   - Keyed on a parent, a Gk marker matches `IS NULL`, and a reader mode refuses on more than one
     row.
   - A Tk lookup asking for neither raises, a Gk one likewise, and a Tk construction without a
     parent raises.
   - The four value-read filters are §0.3's.
4. **The consistency check.** It sits in `store()` or the target, where no caller can skip it. It
   recomputes `z_exit - stop_deltaz_subh` and compares the stop values bit for bit, with the same
   `k` (and `z_source`). Check out the parent's WKB factories over the new tests, and confirm the
   one-ulp test fails. Then restore.
5. **`main.py`.**
   - Read every hunk against §0.6's table.
   - The Gk choice at `z_source_limit` is unchanged, character for character where it was not
     lifted, and equivalent where it was.
   - `EXPECTED_SITES` passes unmodified.
   - Run the five `substitute_main_py()` counts yourself.
6. **Readers.**
   - `extract_*WKB_data.py` and `run_quadsource_integrals.py` each state their mode.
   - Neither silently narrows to pure WKB: grep for any Gk WKB lookup that passes neither a parent
     nor a mode.
   - The log's rule-11 list matches.
7. **K2.** `tk-wkb` is before `tk-numeric`, `gk-wkb` still before `gk-numeric`. The drop test runs
   under enforcement.
8. **D5 and the fingerprint.**
   - Both new files are new, and every old witness and golden is byte-identical at both commits.
   - Re-capture the schema witness from `HEAD` into the scratchpad: byte-identical.
   - Diff the 06 and 07 witnesses yourself. Expect 12 hunks and about 86 lines, by §0.6's six
     kinds, and match it to the log's paste.
   - Re-capture the fingerprint of `build_full_store()` from `HEAD`: it equals the new golden, with
     `fingerprint_format` 2, and only the two WKB classes differ from the old golden.
9. **Replay the mutations** (i)–(v) from the committed log with plain `git apply`. Each gives its
   logged failures. Revert each, leaving the tree clean.
10. **Boards.** The issue is on the GkTk-remedial board's §4, with a closing note naming this
    commit. Count the index rows, and check the date. The campaign board has its §1 row, K1–K2,
    §3.1 and §5.10.
11. **`var/`, suites.**
    - The snapshot is identical.
    - Run all six suites serially. Each rises by exactly the tests added, and nothing else moves.
    - `black --check` is clean.
    - `ray.is_initialized()` is false after each new module.

## 4. After the review

Record it on the board, after the review of 06. Any small residue the review finds is fixed in a
follow-up commit of its own (the user, 2026-09-29).

Then:
- **08 is next** (README §2: 06 → 07 → 08 → 09), with its own orchestrator note carrying this
  review's findings.
  - It adds `quad_policy_serial` to `QuadSourceIntegral`'s key, and a witness and a golden beside
    07's, by the same procedure.
  - Its record change is another format bump, unless 07's review finds a reason otherwise.
- Prompt 10 remains the first real run. After 07 it is also the first time a WKB row's parent, and
  its consistency check, meet a real pipeline.
