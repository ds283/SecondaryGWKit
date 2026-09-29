# Log 06 — Correct the foreign keys

**Prompt:** [`prompts/datastore-integrity/06-correct-the-foreign-keys.md`](../06-correct-the-foreign-keys.md)
**Commit:** *(this commit)* — "Correct the foreign keys the schema declares, and check them"
**Model:** Claude Sonnet 5.5
**Date:** 2026-09-29
**Result:** COMPLETE. S1–S5 shipped. No §5 stop condition was met:
- `Tq` is on `QuadSource`'s shard (quoted below);
- nothing but the schema description reads `stepping_mode` (quoted below);
- the golden fingerprint did not move (the run is quoted below);
- S4 found no violation on any store other than the ones S1 and S2 correct, and the one reference
  its own stand-in data leaves dangling (deviation 3), which is a placeholder and not a wrong key;
- the only existing tests that changed are the two this prompt names (`test_schema_builder.py`'s
  `WITNESS`, and its docstring), and `real_store_fixtures.py`'s map.

`Datastore` rises from 343 to **351** (the 8 tests of the new `test_foreign_key_check.py`); every
other suite is unchanged. The index `docs/OPEN_ISSUES.md` goes from 97 to **96** rows (one
deleted, none opened).

## What shipped

**S1 — `QuadSource`** (`Datastore/SQL/ObjectFactories/QuadSource.py`, `register()`).
- `Tq_serial` now declares `sqla.ForeignKey("TkNumericIntegration.serial")`, in place of
  `"QuadSource.serial"`, with its comment corrected to say what `Tq` is.
- `Tr_serial` still has no foreign key. Its commented-out line and the comment beside it named
  `QuadSource.serial`; they now name `TkNumericIntegration.serial`, and give the cross-shard reason
  (`Tr` is the Tk numeric row at `k = r`, on `r`'s shard, and `QuadSource` shards on `q`).

**S2 — `GkSourcePolicyData`** (`GkSourcePolicyData.py`, `register()`).
`wavenumber_exit_serial` declares `sqla.ForeignKey("wavenumber_exit_time.serial")`; `index=True`
and `nullable=False` are unchanged. The exit time is replicated, so it is on every shard.

**S3 — the fixtures' map** (`Datastore/tests/real_store_fixtures.py`, `_UNDECLARED_REFERENCES`).
`"QuadSource": {"Tr_serial": "TkNumericIntegration"}` (was `"QuadSource"`), and the
`"GkSourcePolicyData": {"wavenumber_exit_serial": ...}` entry is removed because S2 makes it a
declared key, which `references()` now finds from the schema. `QuadSourceIntegral.source_serial`
is unchanged: it stays undeclared (`QuadSourceIntegral` shards on `k`, `QuadSource` on `q`).
Checked with `references()`: `QuadSource.Tq_serial` and `.Tr_serial` map to
`TkNumericIntegration`, and `GkSourcePolicyData.wavenumber_exit_serial` to `wavenumber_exit_time`.

**S4 — the check** (new: `Datastore/tests/test_foreign_key_check.py`, 8 tests; see "Which stores
S4 checks" below).

**S5 — the witness, under D5.**
- `"stepping": "minimum"` becomes `"stepping": "exact"` in `wavenumber.py`
  (`sqla_wavenumber_exit_time_factory.register`) and `integration_metadata.py`
  (`sqla_IntegrationSolver_factory.register`) — the two flags log 03 handed on. Only the string
  changed.
- A new witness, `Datastore/tests/data/schema_at_datastore-integrity-06.json`, captured with
  `PYTHONPATH=. ./venv/bin/python Datastore/tests/schema_description.py
  Datastore/tests/data/schema_at_datastore-integrity-06.json` (the `__main__` that captured the
  first one: `describe_actor_schema()` through `dumps`). `schema_at_base.json` is untouched
  (`git status` never listed it).
- `Datastore/tests/test_schema_builder.py`: `WITNESS` now names the new file, and the module
  docstring states D5's rule (`schema_at_base.json` witnessed a refactor that changed nothing; each
  deliberate schema change adds a witness beside it and repoints the constant; no witness is ever
  regenerated to make a test pass).
- `Datastore/tests/schema_description.py`: its module docstring only, so that it agrees (it said the
  tests compare against "that file"). No code changed.

## The evidence that `Tq` shares `QuadSource`'s shard (prompt §5, first stop)

Every line below was re-read at `8ed150f` for this log.

`main.py` builds the Tk cache from the datastore, keyed by the wavenumber exit time's serial, then
takes `Tq` and `Tr` from it by the exit times `q` and `r`:

```
1705:            task_builder=lambda x: pool.object_get("TkNumericIntegration", **x),
...
1719:        Tk_cache = {
1720:            k.store_id: obj for k, obj in zip(missing_Tk, Tk_lookup_queue.results)
1721:        }
...
1732:            Tq: TkNumericIntegration = Tk_cache[q.store_id]
1733:            Tr: TkNumericIntegration = Tk_cache[r.store_id]
```

`ComputeTargets/QuadSource.py` records their store ids, and those are what `Tq_serial` and
`Tr_serial` hold:

```
451:        Tq: TkNumericIntegration = payload["Tq"]
452:        Tr: TkNumericIntegration = payload["Tr"]
...
473:        self._Tq_serial = Tq.store_id
474:        self._Tr_serial = Tr.store_id
```

`config/sharding.py` shards `TkNumericIntegration` on `k` and `QuadSource` on `q`, and both go
through the one function that turns a wavenumber (or an exit time) into a shard key:

```
20:    "TkNumericIntegration": "k",
24:    "QuadSource": "q",
...
48: def shard_key_wavenumber_store_id(obj):
49:     if isinstance(obj, wavenumber):
50:         return obj.store_id
52:     if isinstance(obj, wavenumber_exit_time):
53:         return obj.k.store_id
```

So `Tq` — the Tk numeric row at the exit time `q` — is keyed on `q.k`, the same wavenumber
`QuadSource` is keyed on: always the same shard. `Tr` is keyed on `r.k`, which is `q.k`'s shard
only by chance. The fixture agrees: shard 0 holds `QuadSource` 1 (q's wavenumber 1) and
`TkNumericIntegration` 1 and 3 (wavenumber 1); its `Tr_serial` 2 is a Tk row on shard 1.

## The evidence that nothing else reads `stepping_mode` (prompt §5, second stop)

`grep -rn stepping_mode . --include='*.py'` (excluding `venv/`, and `prompts/`), and the same
without `--include`, at `8ed150f`, gives exactly these:

```
Datastore/SQL/schema.py:37:    ``use_timestamp``, ``use_stepping`` and ``stepping_mode``), those columns, the factory's own
Datastore/SQL/schema.py:120:            _stepping_mode = None if not isinstance(use_stepping, str) else use_stepping
Datastore/SQL/schema.py:121:            schema["stepping_mode"] = _stepping_mode
Datastore/tests/data/schema_at_base.json:4861:    "stepping_mode": "minimum",
Datastore/tests/data/schema_at_base.json:10975:    "stepping_mode": "minimum",
```

`schema.py:105-121` sets it:

```
105:        use_stepping = registration_data.get("stepping", False)
106:        if isinstance(use_stepping, str):
107:            if use_stepping not in ["minimum", "exact"]:
...
113:        _use_stepping = isinstance(use_stepping, str) or use_stepping is True
114:        schema["use_stepping"] = _use_stepping
...
120:            _stepping_mode = None if not isinstance(use_stepping, str) else use_stepping
121:            schema["stepping_mode"] = _stepping_mode
```

and both strings are accepted. The only reader is `schema_description.py`, which dumps the record's
keys without naming this one (it appears in the witness because `describe_schema` writes the whole
record). `Datastore/SQL/Datastore.py:610` reads `use_stepping`, not the mode:

```
610:        uses_stepping = schema.get("use_stepping", False)
```

The flag is a label; changing it changes the schema description and nothing else.

## The witness diff, in full (prompt §7)

Pasted from `diff Datastore/tests/data/schema_at_base.json
Datastore/tests/data/schema_at_datastore-integrity-06.json` (the default, "normal" diff format,
which lists each change with its line numbers in both files; the two files are pretty-printed with
one JSON value per line, so the long `"ddl"` strings are single lines).

**8 hunks, 30 changed lines** (`grep -c '^[0-9]'` = 8, `grep -c '^[<>]'` = 30). The orchestrator's
note expected 9 hunks; the 30 lines are the same and every one is accounted for, so the difference
is the note's arithmetic (its list of hunks by class adds up to 8: three for `GkSourcePolicyData`,
three for `QuadSource`, two for `stepping_mode`). Each foreign key appears three times, as the note
said: the column's `foreign_keys`, the table's `ForeignKeyConstraint` list, and the `CREATE TABLE`
DDL.

- `2926c2926,2932`, `3037a3044,3053`, `3041c3057` — **S2**, `GkSourcePolicyData`: the column's
  foreign key, the new constraint, the DDL.
- `4861c4877` — the `IntegrationSolver` `stepping_mode` (base line 4861 is inside the record
  named `"IntegrationSolver"`).
- `6066c6082`, `6139c6155`, `6190c6206` — **S1**, `QuadSource.Tq_serial`: the column's foreign key
  target, the constraint's target, the DDL.
- `10975c10991` — the `wavenumber_exit_time` `stepping_mode`.

No line is outside those four classes (S1, S2, the two flags). `Tr_serial` does not appear in it.

```
2926c2926,2932
<       "foreign_keys": [],
---
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "wavenumber_exit_time.serial"
>        }
>       ],
3037a3044,3053
>       ]
>      },
>      {
>       "columns": [
>        "wavenumber_exit_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "wavenumber_exit_time.serial"
3041c3057
<     "ddl": "CREATE TABLE \"GkSourcePolicyData\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tsource_serial INTEGER NOT NULL, \n\tpolicy_serial INTEGER NOT NULL, \n\twavenumber_exit_serial INTEGER NOT NULL, \n\ttype SMALLINT NOT NULL, \n\tquality SMALLINT NOT NULL, \n\tcrossover_z FLOAT, \n\t\"Levin_z\" FLOAT, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(source_serial) REFERENCES \"GkSource\" (serial), \n\tFOREIGN KEY(policy_serial) REFERENCES \"GkSourcePolicy\" (serial)\n)",
---
>     "ddl": "CREATE TABLE \"GkSourcePolicyData\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tsource_serial INTEGER NOT NULL, \n\tpolicy_serial INTEGER NOT NULL, \n\twavenumber_exit_serial INTEGER NOT NULL, \n\ttype SMALLINT NOT NULL, \n\tquality SMALLINT NOT NULL, \n\tcrossover_z FLOAT, \n\t\"Levin_z\" FLOAT, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(source_serial) REFERENCES \"GkSource\" (serial), \n\tFOREIGN KEY(policy_serial) REFERENCES \"GkSourcePolicy\" (serial), \n\tFOREIGN KEY(wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial)\n)",
4861c4877
<     "stepping_mode": "minimum",
---
>     "stepping_mode": "exact",
6066c6082
<         "target": "QuadSource.serial"
---
>         "target": "TkNumericIntegration.serial"
6139c6155
<        "QuadSource.serial"
---
>        "TkNumericIntegration.serial"
6190c6206
<     "ddl": "CREATE TABLE \"QuadSource\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\tmodel_serial INTEGER NOT NULL, \n\tq_wavenumber_exit_serial INTEGER NOT NULL, \n\tr_wavenumber_exit_serial INTEGER NOT NULL, \n\t\"Tq_serial\" INTEGER NOT NULL, \n\t\"Tr_serial\" INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tcompute_time FLOAT, \n\tvalidated BOOLEAN NOT NULL, \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(q_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(r_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(\"Tq_serial\") REFERENCES \"QuadSource\" (serial)\n)",
---
>     "ddl": "CREATE TABLE \"QuadSource\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\tmodel_serial INTEGER NOT NULL, \n\tq_wavenumber_exit_serial INTEGER NOT NULL, \n\tr_wavenumber_exit_serial INTEGER NOT NULL, \n\t\"Tq_serial\" INTEGER NOT NULL, \n\t\"Tr_serial\" INTEGER NOT NULL, \n\tz_samples INTEGER NOT NULL, \n\tcompute_time FLOAT, \n\tvalidated BOOLEAN NOT NULL, \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(q_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(r_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(\"Tq_serial\") REFERENCES \"TkNumericIntegration\" (serial)\n)",
10975c10991
<     "stepping_mode": "minimum",
---
>     "stepping_mode": "exact",
```

## Which stores S4 checks, the rows it relies on, and what it cannot see

`Datastore/tests/test_foreign_key_check.py` runs `PRAGMA foreign_key_check` over **every shard
file** of each store it builds, opened `mode=ro`, and expects no row. No Ray, nothing under
`var/`; every store is in a `tempfile.TemporaryDirectory`. `tearDownModule` asserts
`ray.is_initialized()` is false, as `test_exact_identity_lookups.py` does.

Stores:
- **`TestFullStore`** — `real_store_fixtures.build_full_store` on the test's **own** `sharded`
  rows: `full_rows()` plus one `QuadSource` on shard 0. Also `build_full_store` and
  `build_real_store` on their default rows (`test_the_default_stores_pass_too`), and the own rows
  relabelled by `relabel_serials` (`test_a_relabelled_store_passes`).
- **`TestStandinStore`** — a three-shard store written through `ShardedPool` on the stand-in
  shards of `standin_pool.py`: a replicated exit time, a `GkSourcePolicy`, and a
  `GkSourcePolicyData` (a sharded class) that names the exit time. The rows are written by the
  real `store()`; the test imports the helpers (`make_exit_time`, `make_policy_data`,
  `get_wavenumber`, `get_tolerance`) rather than copying them.
- **`TestTheCheckSeesAViolation`** — two dangling references, so that a check that could not fail
  would be seen not to fail: a `Tq_serial` naming a Tk row on the other shard, and a
  `wavenumber_exit_serial` naming no exit time.

**The rows chosen for "fails before S1", and why.** The fixture's own `QuadSource` rows cannot
show the wrong key, as the orchestrator's note measured and this prompt re-confirmed
(`test_the_default_stores_pass_too` passes with `Tq_serial` pointing back at `QuadSource`, under
mutation (i) below):

| Shard | `QuadSource` serials | `TkNumericIntegration` serials | `QuadSource` row: `serial`, `Tq_serial`, `Tr_serial` |
|---|---|---|---|
| 0 | 1 | 1, 3 | 1, **1**, 2 |
| 1 | 2 | 2 | 2, **2**, 3 |

Each `Tq_serial` is both a `QuadSource` serial and a Tk serial on its own shard, so the wrong key
passes `foreign_key_check` by coincidence, and `relabel_serials` renumbers each reference by the
table its declared key names, so it follows whichever key is declared. The test adds, to shard 0
only, one extra `QuadSource` with `serial` 10, `q_wavenumber_exit_serial` 1,
`r_wavenumber_exit_serial` 2, `Tq_serial` **3** and `Tr_serial` 2. Shard 0's `QuadSource`
serials are now {1, 10} and its `TkNumericIntegration` serials {1, 3}, so `Tq_serial` 3 is a Tk row
there and **not** a `QuadSource` row, and the wrong key must report it.
`test_the_added_row_is_not_satisfied_by_coincidence` asserts that property of the data. The
fixture module's default rows are not changed. The store written through the stand-in shards has
no `QuadSource` (a real one needs a whole Tk pipeline), so it does not exercise `Tq_serial` and has
no coincidence to show; it is there for S2.

Results on the tree before S1 (S4's test written first, S1–S3 and the flags not yet applied;
`before_s1` run, `-v`):

```
test_a_relabelled_store_passes (Datastore.tests.test_foreign_key_check.TestFullStore.test_a_relabelled_store_passes)
test_every_shard_passes_foreign_key_check (Datastore.tests.test_foreign_key_check.TestFullStore.test_every_shard_passes_foreign_key_check) ... FAIL
test_the_added_row_is_not_satisfied_by_coincidence (Datastore.tests.test_foreign_key_check.TestFullStore.test_the_added_row_is_not_satisfied_by_coincidence)
test_the_added_rows_are_in_the_written_shard (Datastore.tests.test_foreign_key_check.TestFullStore.test_the_added_rows_are_in_the_written_shard) ... ok
test_the_default_stores_pass_too (Datastore.tests.test_foreign_key_check.TestFullStore.test_the_default_stores_pass_too) ... ok
test_a_store_written_by_the_real_write_path_passes (Datastore.tests.test_foreign_key_check.TestStandinStore.test_a_store_written_by_the_real_write_path_passes) ... ok
test_a_dangling_exit_time_in_the_policy_data_is_reported (Datastore.tests.test_foreign_key_check.TestTheCheckSeesAViolation.test_a_dangling_exit_time_in_the_policy_data_is_reported) ... FAIL
test_a_dangling_tq_is_reported (Datastore.tests.test_foreign_key_check.TestTheCheckSeesAViolation.test_a_dangling_tq_is_reported)
Ran 8 tests in 1.037s
FAILED (failures=4)
```

Four of the eight fail: the added row's violation (`('QuadSource', 10, 'QuadSource', 0)`) is
reported on shard 0 — in the full store, in the relabelled store, and by the dangling-`Tq` and
dangling-exit-time tests, which then see that extra row. The stand-in store and the default stores
pass, as expected. After S1–S3 all eight pass.

**What it cannot see.** `foreign_key_check` sees only declared foreign keys, and a key can be
declared only to a table in the same file. So it does not check the three references that may name
a row on another shard, which the test's docstring names:
- `QuadSource.Tr_serial` (a `TkNumericIntegration` on `r`'s shard);
- `QuadSourceIntegral.source_serial` (a `QuadSource`, sharded on `q`, from a table sharded on `k`);
- `shard_keys` (the primary's table, naming a shard for each wavenumber serial).

And it cannot tell a wrong key from a right one where the wrong table happens to hold the serial
(the reason for the added row). It was not run against anything under `var/`.

## Deviations from the prompt

1. **An added `QuadSource` row in the test's own data.** `IMPLEMENTATION CHOICE`, forced by the
   fixture's coincidence (the orchestrator's note, and the table above). The prompt (§3.1) asks that
   S4 fail before S1 and offers only "the `Tq_serial` row is a violation"; on the fixture's rows it
   is not one. The fixture module is not changed. The row is passed as the test's own `sharded`
   argument.
2. **The witness diff is 8 hunks, not 9.** Not a deviation from the prompt (its rule, "exactly S1,
   S2 and the two flags", holds for every line); recorded because the note's count was off by one.
3. **The stand-in store reports one violation, by design, and the test asserts exactly it.**
   `IMPLEMENTATION CHOICE`. `make_policy_data` builds a `GkSourcePolicyData` over a placeholder
   `source_serial` (1) that no `GkSource` row backs, and `GkSourcePolicyData.source_serial` is a
   declared key to `GkSource`; writing a real `GkSource` needs the whole Gk pipeline. So on the
   shard that holds the policy data `foreign_key_check` reports `('GkSourcePolicyData', 1,
   'GkSource', n)`. This is the test's own data, not a wrong key: the test asserts that this is
   the **only** row on that shard, that the other shards report none, and that none names
   `wavenumber_exit_time`. It does not compare the fourth field (the foreign key's index in the
   table), because S2's new key moves it (it read 1 before S2's edit was applied, 2 after).
4. **`schema_description.py`'s module docstring edited.** `IMPLEMENTATION CHOICE`: not in the
   prompt's own list, granted by the orchestrator so that the two docstrings agree. No code line of
   that module changed.
5. **`GkSourcePolicyData`'s map entry removed from `_UNDECLARED_REFERENCES`.** Not a deviation — S3
   says to remove an entry S2 turns into a declared key — recorded so that the diff is read right.

No `UNINTENDED DRIFT` was found. `git diff --cached -U0` before the commit shows only the hunks
"What shipped" names.

## Verification performed

- **The starting tree.** `git status` clean at `8ed150f`. The orchestrator's baselines there were
  AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 343, LiouvilleGreen 148 (1
  skipped), RunRegistry 199; this session re-ran `Datastore` and `RunRegistry` before the constant
  was repointed (below) and all six on the final tree.
- **Before the constant was repointed** (S1–S3 and the flags applied, `WITNESS` still at
  `schema_at_base.json`): `Datastore` `Ran 351 tests`, `FAILED (failures=10)`, and the ten are
  `test_schema_builder.TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` and
  `..test_actor_build_schema_reproduces_the_witness`, each with sub-tests for `GkSourcePolicyData`,
  `IntegrationSolver`, `QuadSource` and `wavenumber_exit_time` plus the whole-file assertion
  (2 × 5). Nothing else failed. `RunRegistry` `Ran 199 tests ... OK` on the same tree.
- **The golden fingerprint did not move.** `PYTHONPATH=. ./venv/bin/python -m unittest -v
  RunRegistry.tests.test_store_fingerprint`, on the final tree:

  ```
  test_taken_is_outside_every_digest (RunRegistry.tests.test_store_fingerprint.TestGolden.test_taken_is_outside_every_digest) ... ok
  test_the_golden_fingerprint (RunRegistry.tests.test_store_fingerprint.TestGolden.test_the_golden_fingerprint) ... ok
  test_the_shape (RunRegistry.tests.test_store_fingerprint.TestGolden.test_the_shape) ... ok
  ...
  Ran 33 tests in 13.625s

  OK
  ```

  `RunRegistry/tests/data/full_store_fingerprint.json` is untouched.
- **The relabel tests did not move.** After S1 and S3, `relabel_serials` renumbers `Tq_serial` and
  `Tr_serial` as `TkNumericIntegration` serials, where before it renumbered them as `QuadSource`
  serials. On the default full rows at `offset=100`: shard 0's `QuadSource` 1502 has
  `Tq_serial` 1903, `Tr_serial` 1902, shard 1's 1501 has 1902 and 1901, against Tk serials
  {1903, 1901} and {1902}. `QuadSource.inventory_records` treats both as provenance, not identity
  (`Datastore/SQL/ObjectFactories/QuadSource.py:472`: "Tq_serial and Tr_serial are provenance, not
  identity"), so the relabelled store fingerprints the same. Confirmed by running
  `Datastore.tests.test_store_inventory` (`test_different_serial_assignments`,
  `test_replicated_serials_permuted_and_rows_moved`) and
  `RunRegistry.tests.test_store_fingerprint` (`test_other_serials_and_other_shards`,
  `test_sharded_rows_on_a_third_shard`): all pass unmodified.
- **The six suites on the final tree**, README §7's command, serially:

  | Suite | Result |
  |---|---|
  | AdaptiveLevin | 32 OK |
  | ComputeTargets | 552 OK (the known wall-clock flake did not fire; run alone) |
  | CosmologyModels | 39 OK |
  | Datastore | **351 OK** (343 + the 8 of the new `test_foreign_key_check.py`) |
  | LiouvilleGreen | 148 OK (1 skipped) |
  | RunRegistry | 199 OK; the golden fingerprint unchanged |

- **Formatting.** `./venv/bin/python -m black --check` on the eight Python files this prompt
  touched (the four factories, `real_store_fixtures.py`, `schema_description.py`,
  `test_schema_builder.py`, `test_foreign_key_check.py`): "8 files would be left unchanged".
- **Scope.** `git diff --cached --stat` before the commit lists only the files in the prompt's
  and the orchestrator's list, plus the four boards and this log. No `build()`, `store()`,
  `read_batch` or `inventory_records` is touched; `schema_at_base.json`, the golden fingerprint,
  `schema.py`, `ShardedPool.py`, `Datastore.py`, `store_inventory.py`, `_drop_order`, `main.py` and
  every `extract_*.py` are untouched. No Ray was started and nothing under `var/` was opened.

## The deliberate-breakage record

Each mutation was applied to the tree of S1–S5 (staged, so `git diff` shows the mutation alone),
the full `Datastore` suite and `RunRegistry.tests.test_store_fingerprint` were run, and the
mutation was reverted. The diffs are pasted from `git diff`, exactly as applied, except that (iii)
is `git diff -U1`: its default form ends in two blank context lines, whose trailing space does not
survive a markdown file, and the change is the same line. `Datastore` has 351 tests. `failures=`
counts each failing sub-test as one failure and the whole-file assertion at the end of a witness
test as one more; the lists below say which is which.

All four mutations were replayed twice. **Before the commit**, the four diff blocks below were
extracted from this file and applied with plain `git apply`, the `Datastore` suite and
`test_store_fingerprint` run, and each reverted with `git apply -R`: every failing set and count is
identical to the one listed under its mutation (8, 9, 10 and 4 failures; `test_store_fingerprint`
33 OK each time). **After the commit** they are replayed from
`git show HEAD:prompts/datastore-integrity/logs/06-correct-the-foreign-keys.md`, and that result is
reported to the orchestrator, not written here (this file cannot record a run that follows its
own commit).

### Mutation (i) `Tq_serial` pointed back at `QuadSource.serial`

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSource.py b/Datastore/SQL/ObjectFactories/QuadSource.py
index 79f3fa0..a9fc4ca 100644
--- a/Datastore/SQL/ObjectFactories/QuadSource.py
+++ b/Datastore/SQL/ObjectFactories/QuadSource.py
@@ -113,7 +113,7 @@ class sqla_QuadSource_factory(SQLAFactoryBase):
                     "Tq_serial",
                     sqla.Integer,
                     sqla.ForeignKey(
-                        "TkNumericIntegration.serial"
+                        "QuadSource.serial"
                     ),  # Tq is the Tk numeric row at k = q, which is held on this shard, because we shard by q.
                     index=True,
                     nullable=False,
```

Failing, `Datastore` suite (`Ran 351 tests in 120.216s`, `FAILED (failures=8)`; 8 failing entries, counting each sub-test as an entry):

- `test_foreign_key_check.TestFullStore.test_a_relabelled_store_passes` — the test itself (no sub-tests)
- `test_foreign_key_check.TestFullStore.test_every_shard_passes_foreign_key_check` — the test itself (no sub-tests)
- `test_foreign_key_check.TestTheCheckSeesAViolation.test_a_dangling_exit_time_in_the_policy_data_is_reported` — the test itself (no sub-tests)
- `test_foreign_key_check.TestTheCheckSeesAViolation.test_a_dangling_tq_is_reported` — the test itself (no sub-tests)
- `test_schema_builder.TestSchemaIsUnchanged.test_actor_build_schema_reproduces_the_witness` — sub-tests `cls='QuadSource'` and the whole-file assertion
- `test_schema_builder.TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` — sub-tests `cls='QuadSource'` and the whole-file assertion

`RunRegistry.tests.test_store_fingerprint`: `Ran 33 tests in 15.662s`, `OK` (the golden fingerprint does not move under a schema mutation, as expected).

### Mutation (ii) `Tr_serial` given a foreign key to `TkNumericIntegration.serial`

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSource.py b/Datastore/SQL/ObjectFactories/QuadSource.py
index 79f3fa0..aa796c5 100644
--- a/Datastore/SQL/ObjectFactories/QuadSource.py
+++ b/Datastore/SQL/ObjectFactories/QuadSource.py
@@ -121,7 +121,7 @@ class sqla_QuadSource_factory(SQLAFactoryBase):
                 sqla.Column(
                     "Tr_serial",
                     sqla.Integer,
-                    # sqla.ForeignKey("TkNumericIntegration.serial"),   # Don't impose foreign key. Tr is the Tk numeric row at k = r, which is held on r's shard, not necessarily this one. We shard by q.
+                    sqla.ForeignKey("TkNumericIntegration.serial"),   # Don't impose foreign key. Tr is the Tk numeric row at k = r, which is held on r's shard, not necessarily this one. We shard by q.
                     # index=True,
                     nullable=False,
                 ),
```

Failing, `Datastore` suite (`Ran 351 tests in 125.476s`, `FAILED (failures=9)`; 9 failing entries, counting each sub-test as an entry):

- `test_foreign_key_check.TestFullStore.test_a_relabelled_store_passes` — the test itself (no sub-tests)
- `test_foreign_key_check.TestFullStore.test_every_shard_passes_foreign_key_check` — the test itself (no sub-tests)
- `test_foreign_key_check.TestFullStore.test_the_default_stores_pass_too` — the test itself (no sub-tests)
- `test_foreign_key_check.TestTheCheckSeesAViolation.test_a_dangling_exit_time_in_the_policy_data_is_reported` — the test itself (no sub-tests)
- `test_foreign_key_check.TestTheCheckSeesAViolation.test_a_dangling_tq_is_reported` — the test itself (no sub-tests)
- `test_schema_builder.TestSchemaIsUnchanged.test_actor_build_schema_reproduces_the_witness` — sub-tests `cls='QuadSource'` and the whole-file assertion
- `test_schema_builder.TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` — sub-tests `cls='QuadSource'` and the whole-file assertion

`RunRegistry.tests.test_store_fingerprint`: `Ran 33 tests in 15.806s`, `OK` (the golden fingerprint does not move under a schema mutation, as expected).

### Mutation (iii) the witness constant repointed at `schema_at_base.json`

```diff
diff --git a/Datastore/tests/test_schema_builder.py b/Datastore/tests/test_schema_builder.py
index 5308521..4ebd96c 100644
--- a/Datastore/tests/test_schema_builder.py
+++ b/Datastore/tests/test_schema_builder.py
@@ -40,3 +40,3 @@ from Datastore.tests.schema_description import (
 WITNESS = (
-    Path(__file__).resolve().parent / "data" / "schema_at_datastore-integrity-06.json"
+    Path(__file__).resolve().parent / "data" / "schema_at_base.json"
 )
```

Failing, `Datastore` suite (`Ran 351 tests in 109.999s`, `FAILED (failures=10)`; 10 failing entries, counting each sub-test as an entry):

- `test_schema_builder.TestSchemaIsUnchanged.test_actor_build_schema_reproduces_the_witness` — sub-tests `cls='GkSourcePolicyData'`, `cls='IntegrationSolver'`, `cls='QuadSource'`, `cls='wavenumber_exit_time'` and the whole-file assertion
- `test_schema_builder.TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` — sub-tests `cls='GkSourcePolicyData'`, `cls='IntegrationSolver'`, `cls='QuadSource'`, `cls='wavenumber_exit_time'` and the whole-file assertion

`RunRegistry.tests.test_store_fingerprint`: `Ran 33 tests in 15.693s`, `OK` (the golden fingerprint does not move under a schema mutation, as expected).

### Mutation (iv) one byte of the new witness edited (`IntegrationSolver`'s `stepping_mode`, `exact` to `exacu`)

```diff
diff --git a/Datastore/tests/data/schema_at_datastore-integrity-06.json b/Datastore/tests/data/schema_at_datastore-integrity-06.json
index c88ca6c..1b4ff4e 100644
--- a/Datastore/tests/data/schema_at_datastore-integrity-06.json
+++ b/Datastore/tests/data/schema_at_datastore-integrity-06.json
@@ -4874,7 +4874,7 @@
      "column": "stepping",
      "of_table": "IntegrationSolver"
     },
-    "stepping_mode": "exact",
+    "stepping_mode": "exacu",
     "table": {
      "table": "IntegrationSolver"
     },
```

Failing, `Datastore` suite (`Ran 351 tests in 121.012s`, `FAILED (failures=4)`; 4 failing entries, counting each sub-test as an entry):

- `test_schema_builder.TestSchemaIsUnchanged.test_actor_build_schema_reproduces_the_witness` — sub-tests `cls='IntegrationSolver'` and the whole-file assertion
- `test_schema_builder.TestSchemaIsUnchanged.test_build_schema_reproduces_the_witness` — sub-tests `cls='IntegrationSolver'` and the whole-file assertion

`RunRegistry.tests.test_store_fingerprint`: `Ran 33 tests in 14.826s`, `OK` (the golden fingerprint does not move under a schema mutation, as expected).

Read across the four: (i) fails S4 because of the added row, while the fixture's own rows pass it
(`test_the_default_stores_pass_too` is not in (i)'s list), so the added row is what catches the
wrong key; (ii) fails S4 on the fixture's own rows (`test_the_default_stores_pass_too`), which
already hold the cross-shard pair (`QuadSource` 1: `Tr_serial` 2 is on shard 1) so no row had to be
added for that; (iii) and (iv) fail only the witness tests, (iv) only the sub-test of the class
whose byte was changed.

## Observations not acted on

1. **`TestSchemaIsUnchanged` is now a misnomer.** `test_schema_builder.py`'s test class says
   "unchanged", which was true of the base witness. The prompt grants only `WITNESS` and the
   docstring, so it was left. A rename would be a reviewer's call. No issue opened.
2. **`foreign_key_check`'s fourth field moves when a key is added.** It is the foreign key's index in
   the table, so a test that pins it is fragile. Not pinned by this module. Nothing to act on.
3. **The stand-in store has no `QuadSource`.** A stand-in write of a real `QuadSource` needs Tk
   rows, and the `Tq_serial` key is therefore checked only on the fixture's own rows. Prompt 10's
   end-to-end store is the first place a real `QuadSource` row goes through `store()` and
   `foreign_key_check`.
4. **`QuadSourceIntegral.source_serial` stays undeclared.** It is the third of the three unchecked
   cross-shard references, and remains in `_UNDECLARED_REFERENCES` for `relabel_serials`. Prompt 09
   owns it (K5/K7).
5. **Nothing here has run under Ray or on a real store.**

## State handed to the next prompt

- **The D5 procedure is established.** A schema-changing prompt (07, 08, 09) captures a new witness
  `Datastore/tests/data/schema_at_<campaign>-<NN>.json` with `schema_description.py`'s `__main__`,
  repoints `WITNESS` in `test_schema_builder.py`, leaves every earlier witness in place, and puts
  the diff between the new witness and this one in its log. The diff format used here is
  `diff old new` (normal); each foreign key appears three times.
- **For 07.** `QuadSource`'s and the WKB tables' foreign keys are now checked by
  `test_foreign_key_check.py`: a new key `TkWKBIntegration.numeric_serial` is on the same shard
  as its numeric row, so it is testable the same way, on rows of the test's own that the wrong
  table does not satisfy by coincidence. `_drop_order` is still untouched, per the prompt.
- **Names.** No new public name. `references()` in `real_store_fixtures.py` now finds
  `GkSourcePolicyData.wavenumber_exit_serial` and `QuadSource.Tq_serial` from the schema.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **351**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
