# Log 08 — Key QuadSourceIntegral on its QuadSourcePolicy

**Prompt:** [`prompts/datastore-integrity/08-key-quadsourceintegral-on-its-policy.md`](../08-key-quadsourceintegral-on-its-policy.md)
**Commit:** *(this commit)* — "Key QuadSourceIntegral on its QuadSourcePolicy"
**Model:** Claude Sonnet 5.5
**Date:** 2026-09-29
**Result:** COMPLETE. K3 shipped, under the user's decisions 6.1.2 and D6, D4 and D5, and the three
decisions of the orchestrator note's §0.1–§0.3 (made 2026-09-29):
- **§0.1.** The extraction script finds its `QuadSourcePolicy` through a new `read_table`, never
  `object_get`, and refuses on none or several.
- **§0.2.** The policy's three parameters live in one constant in `config/defaults.py`.
- **§0.3.** The existing tests listed there follow the key, and nothing else.

No stop condition triggered. No computation reads a `QuadSourcePolicy`'s fields (evidence below).

Suites after this commit, serially, on the final tree:
- ComputeTargets **566 → 585** (the 19 of the new `test_quadsource_policy_main.py`);
- Datastore **397 → 422** (the 21 of `test_quadsource_policy_key.py` and the 4 of
  `test_quadsource_policy_foreign_key.py`);
- AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped) and RunRegistry 199,
  unchanged.

The index `docs/OPEN_ISSUES.md` stays at **98** rows, counted (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`):
one deleted and one opened.

## What shipped

**K3 — the key** (`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`).
- **`quad_policy_serial`**, NOT NULL, indexed, `ForeignKey("QuadSourcePolicy.serial")`, after
  `policy_serial` in `register()`.
- **`build()`** takes `payload["quad_policy"]` and filters on it; it hands it to the constructor on a
  miss and on a hit.
- **`store()`** writes `quad_policy_serial`.
- **`read_batch()`** takes `payload["quad_policy"]` (required), filters on it, and hands it to every
  object it builds.
- **`inventory_records`** gains `"quad_policy": Parent("quad_policy_serial", "QuadSourcePolicy")`,
  and its comment now names the QuadSourcePolicy as part of the identity.
- **A store that predates the column (D4).** Both `build()` and `read_batch()` wrap the query in
  `except SQLAlchemyError` and call one module-level helper,
  `_refuse_a_store_that_predates_the_policy_key`. It matches `str(getattr(e, "orig", e))` for
  `quad_policy_serial`, as 07 did, and raises `RuntimeError` from the original: *"QuadSourceIntegral.
  build(): the QuadSourceIntegral table has no "quad_policy_serial" column. This datastore predates
  QuadSourceIntegral rows being keyed on the QuadSourcePolicy they were made under
  (prompts/datastore-integrity, prompt 08). This tree does not read it; it remains readable at the
  git SHA its sidecar records."* It does not say "regenerated". Any other database error is
  re-raised unchanged. In `build()` the `MultipleResultsFound` handler stays first.

**K3 — the compute target** (`ComputeTargets/QuadSourceIntegral.py`). The constructor takes
`quad_policy: QuadSourcePolicy`, required and positional after `policy`; `None` raises
`ValueError`. It is stored as `_quad_policy` and exposed as a `quad_policy` property. The name
`quad_policy` sits beside the existing `policy` (the `GkSourcePolicy`, checked against the
`GkSourcePolicyData` at compute time) without ambiguity: `policy` alone would have named either.
Nothing reads its fields, so nothing checks it at compute time.

**`MetadataConcepts/QuadSourcePolicy.py`.** The docstring's opening paragraph only: it no longer says
"Nothing reads it", and says instead that no computation reads its *fields* and that
`QuadSourceIntegral` is keyed on it. The two bullets, the closing paragraph and the class are
unchanged.

**§0.1 — `read_table`** (`Datastore/SQL/ObjectFactories/QuadSourcePolicy.py`), with the one
`read_table_config` entry `"QuadSourcePolicy": {"tables_arg": False}` in `config/sharding.py`.
`read_table(conn, table, Levin_threshold, numeric_policy)` matches as `build()` does
(`numeric_policy` equal; `Levin_threshold` within `DEFAULT_FLOAT_PRECISION`), inserts nothing, and
returns every match in serial order.

**§0.2 — the constant** (`config/defaults.py`): `DEFAULT_QUADSOURCE_POLICY_1PT5`, a dict of the
`label`, `Levin_threshold` and `numeric_policy` `main.py` used to spell out. `main.py` creates
`QuadSource_policy_1pt5` with `pool.object_get("QuadSourcePolicy", **DEFAULT_QUADSOURCE_POLICY_1PT5)`.
`QuadSource_policy_5pt0` is left as literals, created and unused. The `GkSourcePolicy` literals are
untouched in both files.

**`main.py`**: the constant's import, the existence query's payload and the compute call each gain
`quad_policy` = `QuadSource_policy_1pt5`, and the creation uses the constant (table below).

**`extract_QuadSourceIntegral_data.py`**: a module-level `find_QuadSource_policy(pool)` (see "Where
the extraction script takes the policy from"), its call in `run_pipeline`, and `quad_policy=` in the
`object_read_batch` call.

**D5 and the fingerprint.**
- `Datastore/tests/data/schema_at_datastore-integrity-08.json`, captured with `PYTHONPATH=. ./venv/bin/python
  Datastore/tests/schema_description.py Datastore/tests/data/schema_at_datastore-integrity-08.json`;
  `test_schema_builder.WITNESS` repointed at it, and its docstring's "current one" sentence
  updated.
- `RunRegistry/tests/data/full_store_fingerprint_datastore-integrity-08.json`;
  `test_store_fingerprint.GOLDEN` repointed at it. `FINGERPRINT_FORMAT` 2 → 3, and its comment gains
  the format-3 sentence.
- `schema_at_base.json`, `schema_at_datastore-integrity-06.json`, `…-07.json`,
  `full_store_fingerprint.json` and `full_store_fingerprint_datastore-integrity-07.json` are untouched.

**Tests** (new):
- `Datastore/tests/test_quadsource_policy_key.py`, 21 tests: §3 tests 1, 2 and 4; the D4 refusal in
  both readers; `read_table`.
- `Datastore/tests/test_quadsource_policy_foreign_key.py`, 4 tests: the new key under `PRAGMA
  foreign_key_check`, on rows of the test's own (the docstring says why).
- `ComputeTargets/tests/test_quadsource_policy_main.py`, 19 tests: §3 tests 3 and 5; the constant;
  the extraction script's lookup and its refusals, on a real `ShardedPool` on stand-in shards.
- Each module's `tearDownModule` asserts `ray.is_initialized()` is false.

## The evidence that no computation reads a `QuadSourcePolicy`'s fields (prompt §5, first stop)

Re-run at `479736d`, the tree before this commit, excluding `docs/`, `venv/` and `tests/`:

```
$ grep -rnE "\.Levin_threshold|\.numeric_policy" --include="*.py" . | grep -v "^./venv\|^./docs/\|/tests/"   # at 479736d
Datastore/SQL/ObjectFactories/QuadSourcePolicy.py:38:                sqla.func.abs(table.c.Levin_threshold - Levin_threshold)
Datastore/SQL/ObjectFactories/QuadSourcePolicy.py:40:                table.c.numeric_policy == numeric_policy,
Datastore/SQL/ObjectFactories/GkSourcePolicy.py:38:                sqla.func.abs(table.c.Levin_threshold - Levin_threshold)
Datastore/SQL/ObjectFactories/GkSourcePolicy.py:40:                table.c.numeric_policy == numeric_policy,
MetadataConcepts/QuadSourcePolicy.py:19:        GkSourcePolicy.numeric_policy does for the Green's function. The transfer function has
ComputeTargets/GkSourcePolicyData.py:57:    # GkSourcePolicy.Levin_threshold, which sets it, is likewise diagnostic from here on.
ComputeTargets/GkSourcePolicyData.py:211:    # |d theta_G / d log(1+z)| never exceeds policy.Levin_threshold anywhere in its WKB range is
ComputeTargets/GkSourcePolicyData.py:250:                > policy.Levin_threshold
ComputeTargets/GkSourcePolicyData.py:398:            policy_name = policy.numeric_policy
ComputeTargets/QuadSourceIntegral.py:42:# than reviving them through MetadataConcepts/QuadSourcePolicy.Levin_threshold. Prompt 08 section 6

$ grep -rn "Levin_threshold\|numeric_policy" --include="*.py" . | grep -v "^./venv\|^./docs/\|/tests/" | grep -v "Datastore/SQL/ObjectFactories/\(Quad\|Gk\)SourcePolicy.py" | cut -c1-140
extract_QuadSourceIntegral_data.py:1067:                Levin_threshold=1.5,
extract_QuadSourceIntegral_data.py:1068:                numeric_policy="maximize-WKB",
extract_QuadSourceIntegral_data.py:1073:                Levin_threshold=5.0,
extract_QuadSourceIntegral_data.py:1074:                numeric_policy="maximize-WKB",
main.py:3746:                Levin_threshold=1.5,
main.py:3747:                numeric_policy="maximize-WKB",
main.py:3752:                Levin_threshold=5.0,
main.py:3753:                numeric_policy="maximize-WKB",
main.py:3764:                Levin_threshold=1.5,
main.py:3765:                numeric_policy="maximize-WKB",
main.py:3770:                Levin_threshold=5.0,
main.py:3771:                numeric_policy="maximize-WKB",
extract_GkSource_data.py:861:                Levin_threshold=1.5,
extract_GkSource_data.py:862:                numeric_policy="maximize-WKB",
extract_GkSource_data.py:867:                Levin_threshold=5.0,
extract_GkSource_data.py:868:                numeric_policy="maximize-WKB",
docs/source-remediation-verification/run_quadsource_integrals.py:234:                Levin_threshold=1.5,
docs/source-remediation-verification/run_quadsource_integrals.py:235:                numeric_policy="maximize-WKB",
docs/source-remediation-verification/analyse_greens_and_source.py:265:            Levin_threshold=1.5,
docs/source-remediation-verification/analyse_greens_and_source.py:266:            numeric_policy="maximize-WKB",
docs/tolerance-convergence/inventory.py:84:        "Levin_threshold",
docs/tolerance-convergence/inventory.py:745:            reaches="**neither.** Nothing reads `Levin_threshold` off a policy object, and "
MetadataConcepts/QuadSourcePolicy.py:18:      * `numeric_policy` would choose a numeric/WKB crossover inside an overlap region, as
MetadataConcepts/QuadSourcePolicy.py:19:        GkSourcePolicy.numeric_policy does for the Green's function. The transfer function has
MetadataConcepts/QuadSourcePolicy.py:25:      * `Levin_threshold` would be a minimum oscillation rate below which a sub-interval of the
MetadataConcepts/QuadSourcePolicy.py:39:    `Levin_threshold` from a compute target without revisiting the reasoning above.
MetadataConcepts/QuadSourcePolicy.py:45:        Levin_threshold: int = DEFAULT_LEVIN_THRESHOLD,
MetadataConcepts/QuadSourcePolicy.py:46:        numeric_policy: str = DEFAULT_QUADSOURCE_NUMERIC_POLICY,
MetadataConcepts/QuadSourcePolicy.py:55:        self._Levin_threshold = Levin_threshold
MetadataConcepts/QuadSourcePolicy.py:57:        if numeric_policy not in _allowed_numeric_policies:
MetadataConcepts/QuadSourcePolicy.py:58:            numeric_policy = DEFAULT_QUADSOURCE_NUMERIC_POLICY
MetadataConcepts/QuadSourcePolicy.py:60:                f'!! Unknown Quadsource numeric policy "{numeric_policy}". Defaulting to policy "{DEFAULT_QUAD
MetadataConcepts/QuadSourcePolicy.py:62:        self._numeric_policy = numeric_policy
MetadataConcepts/QuadSourcePolicy.py:65:    def Levin_threshold(self) -> float:
MetadataConcepts/QuadSourcePolicy.py:66:        return self._Levin_threshold
MetadataConcepts/QuadSourcePolicy.py:69:    def numeric_policy(self) -> str:
MetadataConcepts/QuadSourcePolicy.py:70:        return self._numeric_policy
MetadataConcepts/GkSourcePolicy.py:13:        Levin_threshold: int = DEFAULT_LEVIN_THRESHOLD,
MetadataConcepts/GkSourcePolicy.py:14:        numeric_policy: str = DEFAULT_GKSOURCE_NUMERIC_POLICY,
MetadataConcepts/GkSourcePolicy.py:23:        self._Levin_threshold = Levin_threshold
MetadataConcepts/GkSourcePolicy.py:25:        if numeric_policy not in _allowed_numeric_policies:
MetadataConcepts/GkSourcePolicy.py:26:            numeric_policy = DEFAULT_GKSOURCE_NUMERIC_POLICY
MetadataConcepts/GkSourcePolicy.py:28:                f'!! Unknown GkSource numeric policy "{numeric_policy}". Defaulting to policy "{DEFAULT_GKSOURCE
MetadataConcepts/GkSourcePolicy.py:30:        self._numeric_policy = numeric_policy
MetadataConcepts/GkSourcePolicy.py:33:    def Levin_threshold(self) -> float:
MetadataConcepts/GkSourcePolicy.py:34:        return self._Levin_threshold
MetadataConcepts/GkSourcePolicy.py:37:    def numeric_policy(self) -> str:
MetadataConcepts/GkSourcePolicy.py:38:        return self._numeric_policy
ComputeTargets/GkSourcePolicyData.py:57:    # GkSourcePolicy.Levin_threshold, which sets it, is likewise diagnostic from here on.
ComputeTargets/GkSourcePolicyData.py:211:    # |d theta_G / d log(1+z)| never exceeds policy.Levin_threshold anywhere in its WKB range is
ComputeTargets/GkSourcePolicyData.py:250:                > policy.Levin_threshold
ComputeTargets/GkSourcePolicyData.py:398:            policy_name = policy.numeric_policy
ComputeTargets/QuadSourceIntegral.py:32:# Levin_threshold) and the net phase change theta_G(z_response) - theta_G(Levin_z) >
ComputeTargets/QuadSourceIntegral.py:42:# than reviving them through MetadataConcepts/QuadSourcePolicy.Levin_threshold. Prompt 08 section 6
```

Reading the output:
- `.Levin_threshold` and `.numeric_policy` are read off a policy object only at
  `ComputeTargets/GkSourcePolicyData.py:250` and `:398`, on a `GkSourcePolicy`; the other hits are
  the two policy factories' own `build()` filters (table columns, not the object), and comments.
- `QuadSourcePolicy`'s own `Levin_threshold` and `numeric_policy` properties are read by nothing
  that the second grep lists: the remaining hits are the classes' constructors and the literals
  `main.py` and the scripts create the policies from.
- `main.py` passes `QuadSource_policy_1pt5` and `_5pt0` into `run_pipeline` (`:1021-1022`) and, at
  `479736d`, uses neither inside it.

So keying on the policy changes the record and no number, and D6 stands.

## Every site changed

### `main.py`

"Before" is `479736d`, "after" is this commit.

| Site | Before | After |
|---|---|---|
| The import of the constant | — | `:72` (`DEFAULT_QUADSOURCE_POLICY_1PT5` in the `config.defaults` import) |
| QSI existence query, vectorized: `"quad_policy": QuadSource_policy_1pt5` beside `"policy": GkSource_policy_1pt5` | `:3132` | `:3133-3134` |
| QSI compute call: `quad_policy=QuadSource_policy_1pt5` beside `policy=GkSource_policy_1pt5` | `:3518` | `:3520-3521` |
| The two `QuadSourcePolicy` creations; the 1.5 one takes the constant, in place of six lines | `:3759-3772` | `:3762-3775` (`:3764-3766` is the 1.5 creation and its two comment lines) |

`git diff -U0 main.py` has four hunks (`@@ -71,0 +72 @@`, `@@ -3132,0 +3134 @@`,
`@@ -3518,0 +3521 @@`, `@@ -3761,6 +3764,3 @@`), each inside the regions above.
`test_main_plumbing.EXPECTED_SITES` did not move: `test_main_plumbing` passes unmodified.

### The other files

| File | Site | Before | After |
|---|---|---|---|
| `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` | `register` | `policy_serial` at `:136` | `quad_policy_serial` column and its comment at `:165-174`, after it |
| same | `build` | `:216-` (`policy` at `:218`, filter at `:269`) | `quad_policy` at `:254`, filter at `:306`, handler at `:336-338`, constructor calls at `:346`, `:402` |
| same | `store` | `policy_serial` written at `:399` | `quad_policy_serial` written at `:442` |
| same | `read_batch` | `:503-` (`policy` at `:505`, filter at `:627`) | `quad_policy` at `:549`, handler at `:731-735`, constructor call at `:836` |
| same | `inventory_records` | `:815-` | `quad_policy` parent at `:879` |
| same | new helper | — | `_refuse_a_store_that_predates_the_policy_key`, `:78-100` |
| `ComputeTargets/QuadSourceIntegral.py` | constructor | `policy` only, `:1744-1756` | `quad_policy` parameter, checked and stored beside `_policy`; a property |
| `MetadataConcepts/QuadSourcePolicy.py` | docstring | "Nothing reads it." | "No computation reads its fields", and says it is keyed on |
| `Datastore/SQL/ObjectFactories/QuadSourcePolicy.py` | new `read_table` | — | between `build` and `inventory_records` |
| `config/sharding.py` | `read_table_config` | `wavenumber`, `redshift` | `+ "QuadSourcePolicy"` |
| `config/defaults.py` | constant | — | `DEFAULT_QUADSOURCE_POLICY_1PT5`, after `DEFAULT_QUADSOURCE_NUMERIC_POLICY` |
| `extract_QuadSourceIntegral_data.py` | import | — | `:50` |
| same | the policy lookup | `GkSourcePolicy` pair only, `:1062-1075` | `find_QuadSource_policy` defined at `:942-983`; called at `:1127` |
| same | the `object_read_batch` | `:1087-1100`, `policy=` at `:1091` | `quad_policy=` at `:1142` |
| `RunRegistry/stores.py` | `FINGERPRINT_FORMAT` | 2 | 3, and its comment |

## The witness diff, in full (D5)

Pasted from `diff Datastore/tests/data/schema_at_datastore-integrity-07.json
Datastore/tests/data/schema_at_datastore-integrity-08.json`, default "normal" format, unedited. It has **6 hunks and 43 changed
lines** (`grep -c '^[0-9]'` = 6, `grep -c '^[<>]'` = 43). Counted by the six kinds, each hunk located
by the JSON key that encloses it (checked by scanning up from each hunk to its enclosing keys):

| Kind | `QuadSourceIntegral` |
|---|---|
| 1. schema record's column list (`record.columns`) | `6331a6332,6335` |
| 2. the column's description (`table.columns`) | `6677a6682,6700` |
| 3. its `ForeignKeyConstraint` (`table.constraints`) | `7172a7196,7205` |
| 4. the `CREATE TABLE` DDL (`table.ddl`) | `7229c7262` |
| 5. the index DDL (`table.index_ddl`) | `7236a7270` |
| 6. the index description (`table.indexes`) | `7284a7319,7325` |

**1 class × 6 kinds = 6 hunks, and no hunk outside them or outside `QuadSourceIntegral`.** The
`read_table` and the `read_table_config` entry change no witness line.

```
6331a6332,6335
>       "column": "quad_policy_serial",
>       "of_table": "QuadSourceIntegral"
>      },
>      {
6677a6682,6700
>       "foreign_keys": [
>        {
>         "ondelete": null,
>         "onupdate": null,
>         "target": "QuadSourcePolicy.serial"
>        }
>       ],
>       "index": true,
>       "name": "quad_policy_serial",
>       "nullable": false,
>       "primary_key": false,
>       "server_default": null,
>       "sqlite_type": "INTEGER",
>       "type": "Integer()",
>       "unique": null
>      },
>      {
>       "autoincrement": "'auto'",
>       "default": null,
7172a7196,7205
>        "quad_policy_serial"
>       ],
>       "kind": "ForeignKeyConstraint",
>       "name": null,
>       "targets": [
>        "QuadSourcePolicy.serial"
>       ]
>      },
>      {
>       "columns": [
7229c7262
<     "ddl": "CREATE TABLE \"QuadSourceIntegral\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\tmodel_serial INTEGER NOT NULL, \n\tk_wavenumber_exit_serial INTEGER NOT NULL, \n\tq_wavenumber_exit_serial INTEGER NOT NULL, \n\tr_wavenumber_exit_serial INTEGER NOT NULL, \n\tatol_serial INTEGER NOT NULL, \n\trtol_serial INTEGER NOT NULL, \n\tpolicy_serial INTEGER NOT NULL, \n\tsource_serial INTEGER NOT NULL, \n\tdata_serial INTEGER NOT NULL, \n\tz_response_serial INTEGER NOT NULL, \n\tz_source_max_serial INTEGER NOT NULL, \n\tb FLOAT NOT NULL, \n\ttotal FLOAT NOT NULL, \n\ttotal_abserr FLOAT, \n\ttotal_converged BOOLEAN, \n\ttotal_phase_limited BOOLEAN, \n\tnumeric_quad FLOAT NOT NULL, \n\t\"WKB_Levin\" FLOAT NOT NULL, \n\tanalytic_rad FLOAT, \n\teta_source_max FLOAT, \n\teta_response FLOAT, \n\tnumeric_quad_compute_time FLOAT, \n\tnumeric_quad_compute_steps INTEGER, \n\t\"numeric_quad_RHS_evaluations\" INTEGER, \n\t\"numeric_quad_mean_RHS_time\" FLOAT, \n\t\"numeric_quad_max_RHS_time\" FLOAT, \n\t\"numeric_quad_min_RHS_time\" FLOAT, \n\t\"WKB_Levin_num_regions\" INTEGER, \n\t\"WKB_Levin_evaluations\" INTEGER, \n\t\"WKB_Levin_simple_regions\" INTEGER, \n\t\"WKB_Levin_SVD_errors\" INTEGER, \n\t\"WKB_Levin_order_changes\" INTEGER, \n\t\"WKB_Levin_chebyshev_min_order\" INTEGER, \n\t\"WKB_Levin_max_depth\" INTEGER, \n\t\"WKB_Levin_elapsed\" FLOAT, \n\t\"WKB_phase_spline_chunks\" INTEGER, \n\tcompute_time FLOAT, \n\tanalytic_compute_time FLOAT, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(k_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(q_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(r_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(atol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(rtol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(policy_serial) REFERENCES \"GkSourcePolicy\" (serial), \n\tFOREIGN KEY(data_serial) REFERENCES \"GkSourcePolicyData\" (serial), \n\tFOREIGN KEY(z_response_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_source_max_serial) REFERENCES redshift (serial)\n)",
---
>     "ddl": "CREATE TABLE \"QuadSourceIntegral\" (\n\tserial INTEGER NOT NULL, \n\tversion INTEGER, \n\ttimestamp DATETIME, \n\tlabel VARCHAR(256), \n\tmodel_serial INTEGER NOT NULL, \n\tk_wavenumber_exit_serial INTEGER NOT NULL, \n\tq_wavenumber_exit_serial INTEGER NOT NULL, \n\tr_wavenumber_exit_serial INTEGER NOT NULL, \n\tatol_serial INTEGER NOT NULL, \n\trtol_serial INTEGER NOT NULL, \n\tpolicy_serial INTEGER NOT NULL, \n\tquad_policy_serial INTEGER NOT NULL, \n\tsource_serial INTEGER NOT NULL, \n\tdata_serial INTEGER NOT NULL, \n\tz_response_serial INTEGER NOT NULL, \n\tz_source_max_serial INTEGER NOT NULL, \n\tb FLOAT NOT NULL, \n\ttotal FLOAT NOT NULL, \n\ttotal_abserr FLOAT, \n\ttotal_converged BOOLEAN, \n\ttotal_phase_limited BOOLEAN, \n\tnumeric_quad FLOAT NOT NULL, \n\t\"WKB_Levin\" FLOAT NOT NULL, \n\tanalytic_rad FLOAT, \n\teta_source_max FLOAT, \n\teta_response FLOAT, \n\tnumeric_quad_compute_time FLOAT, \n\tnumeric_quad_compute_steps INTEGER, \n\t\"numeric_quad_RHS_evaluations\" INTEGER, \n\t\"numeric_quad_mean_RHS_time\" FLOAT, \n\t\"numeric_quad_max_RHS_time\" FLOAT, \n\t\"numeric_quad_min_RHS_time\" FLOAT, \n\t\"WKB_Levin_num_regions\" INTEGER, \n\t\"WKB_Levin_evaluations\" INTEGER, \n\t\"WKB_Levin_simple_regions\" INTEGER, \n\t\"WKB_Levin_SVD_errors\" INTEGER, \n\t\"WKB_Levin_order_changes\" INTEGER, \n\t\"WKB_Levin_chebyshev_min_order\" INTEGER, \n\t\"WKB_Levin_max_depth\" INTEGER, \n\t\"WKB_Levin_elapsed\" FLOAT, \n\t\"WKB_phase_spline_chunks\" INTEGER, \n\tcompute_time FLOAT, \n\tanalytic_compute_time FLOAT, \n\tmetadata VARCHAR(256), \n\tPRIMARY KEY (serial), \n\tFOREIGN KEY(version) REFERENCES version (serial), \n\tFOREIGN KEY(model_serial) REFERENCES \"BackgroundModel\" (serial), \n\tFOREIGN KEY(k_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(q_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(r_wavenumber_exit_serial) REFERENCES wavenumber_exit_time (serial), \n\tFOREIGN KEY(atol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(rtol_serial) REFERENCES tolerance (serial), \n\tFOREIGN KEY(policy_serial) REFERENCES \"GkSourcePolicy\" (serial), \n\tFOREIGN KEY(quad_policy_serial) REFERENCES \"QuadSourcePolicy\" (serial), \n\tFOREIGN KEY(data_serial) REFERENCES \"GkSourcePolicyData\" (serial), \n\tFOREIGN KEY(z_response_serial) REFERENCES redshift (serial), \n\tFOREIGN KEY(z_source_max_serial) REFERENCES redshift (serial)\n)",
7236a7270
>      "CREATE INDEX \"ix_QuadSourceIntegral_quad_policy_serial\" ON \"QuadSourceIntegral\" (quad_policy_serial)",
7284a7319,7325
>       "unique": false
>      },
>      {
>       "columns": [
>        "quad_policy_serial"
>       ],
>       "name": "ix_QuadSourceIntegral_quad_policy_serial",
```

## The fingerprint re-capture (D5, §0.2)

`FINGERPRINT_FORMAT` goes from 2 to 3. The new golden was captured the way
`test_store_fingerprint._FullStores` builds its baseline: `build_full_store` into a temporary
directory, `read_inventory(...)`, `fingerprint_of(...)` with `taken` None, serialised as
`json.dumps(indent=2, sort_keys=True) + "\n"`. **The script was first run at `479736d` (in a detached
worktree, removed) and reproduced `full_store_fingerprint_datastore-integrity-07.json` byte for
byte.** The script is in this prompt's scratch directory and is not part of the tree.

- **Moved**, checked by comparing the two goldens class by class:

  | Class | Count | Tag-set counts | Digest | Why |
  |---|---|---|---|---|
  | `QuadSourceIntegral` | 3 → 3 | [2, 1] → [2, 1] | `f413fb6bb4a5f28b…` → `817af21962788bc4…` | its key gains the `quad_policy` parent; both tag sets moved |
  | `QuadSourcePolicy` | 1 → 2 | [1] → [2] | `1cfd124beaf6998f…` → `15634c5ae8ceb3cf…` | the fixture's second row (structurally required, below) |

- **Not moved: the other 19 classes**, identical in count, digest and tag sets: `BackgroundModel`,
  `GkNumericIntegration`, `GkSource`, `GkSourcePolicy`, `GkSourcePolicyData`, `GkWKBIntegration`,
  `IntegrationSolver`, `LambdaCDM`, `OneLoopIntegral`, `QCD_Cosmology`, `QuadSource`,
  `TkNumericIntegration`, `TkWKBIntegration`, `redshift`, `store_tag`, `tolerance`, `version`,
  `wavenumber` and `wavenumber_exit_time`. Nothing keys on a `QuadSourceIntegral`'s digest.
- `problems` is `{}` before and after; `taken` is null in both. The overall digest:
  `cd04f246e59f1233…` → `46685ed2201f7bcd…`.
- `test_store_fingerprint` passes unmodified apart from `GOLDEN`.
- **`store retire` of a store written before 08 must be run at a pre-08 SHA.** Its recorded
  fingerprint is format 2 (or 1); at this tree `compare_fingerprints` says the formats differ and
  does not compare, and an inventory recomputed here finds `QuadSourceIntegral` `incomplete` for
  want of `quad_policy_serial`. `store fingerprint` and `store amend` at this tree record format 3.

## Where the extraction script takes the policy from (§0.2), and how it refuses (§0.1)

- **From `config.defaults.DEFAULT_QUADSOURCE_POLICY_1PT5`**, the constant `main.py` creates the row
  from. Nothing is hard-coded in the script, and no serial is assumed: the serial is found.
- **By `ray.get(pool.read_table("QuadSourcePolicy", Levin_threshold=…, numeric_policy=…))`**, in
  `find_QuadSource_policy(pool)`. `read_table` reads one shard and writes nothing. Refusing from
  inside `object_get` would leave a `replication_in_flight` record on the primary
  (`ShardedPool._replicated_write` commits it before the controller's call, and a refusal leaves
  it), which prompt 02's check at open would then refuse the store the script was reading.
- **On no row:** `RuntimeError`, *"this datastore holds no QuadSourcePolicy with
  Levin_threshold=1.5 and numeric_policy="maximize-WKB", the policy main.py keys every
  QuadSourceIntegral on, so there are no QuadSourceIntegral rows for this script to read. It does
  not create one."*
- **On more than one:** `RuntimeError`, *"this datastore holds N QuadSourcePolicy rows with
  Levin_threshold=1.5 and numeric_policy="maximize-WKB" (serials 1, 2), and it is not possible to
  say which one the QuadSourceIntegral rows are keyed on."*
- **The refusal writes nothing**, tested on a real `ShardedPool` on stand-in shards
  (`TestTheExtractionScript`): the only calls made are `read_table` on `QuadSourcePolicy`, the
  `QuadSourcePolicy` rows on every shard are unchanged, and `replication_in_flight` is empty.
  A control test shows that creating the policy through `object_get` makes an `object_get` call,
  which is what those assertions tell apart.
- The script's `GkSourcePolicy` lookups stay `object_get`s (see "Observations not acted on").

## Readers (README §5 rule 11), and what now refuses a store it read before

The `QuadSourceIntegral` readers, and every script that names the class:

| Reader | Before | After |
|---|---|---|
| `main.py`, the QSI stage (existence query, compute call) | keyed on `policy` | keyed on `policy` and `quad_policy` |
| `extract_QuadSourceIntegral_data.py` | `read_batch` on `policy` | finds the policy by `read_table`, passes `quad_policy` |
| `docs/handover/quadsource_atol_sweep.py` | `exec`s `main.py`, five substitutions | keyed through `main.py`; not edited; its sweep store under `var/` is pre-08 and **is now refused under D4 (not opened here)** |
| `docs/gktk-remedial/scoped_pipeline_run.py` | `exec`s `main.py`, k-grid substitutions | keyed through `main.py`; builds its own store; not edited |
| `docs/source-remediation-verification/run_quadsource_integrals.py` | calls `compute_QuadSource_integral` directly | unchanged (it does not construct the target) |
| `tools/inventory_report.py`, `docs/gktk-remedial/analyse_scoped_run.py`, `docs/tolerance-convergence/*` | raw SQL or table names | unchanged, not touched |
| `docs/datastore-integrity-audit/lookup_keys/probe_01_tagged_read_batch.py` | runs (exit 0) | **stops running** (exit 1): its raw `QuadSourceIntegral` insert lacks the new NOT NULL column. Not edited, by the user's decision of 2026-09-29 (audit probes are evidence for the tree their audit read) |

**What now refuses a store it read before:**
- **`main.py`'s QSI stage and `extract_QuadSourceIntegral_data.py`, on every store written before
  this commit** (the three under `var/` included, not opened here). Both get the D4 message from
  `build()` or `read_batch()`: the store predates the key and is readable at the SHA its sidecar
  records. The extraction script first finds the policy (the table exists in an old store) and then
  meets the message.
- **`docs/handover/quadsource_atol_sweep.py` against its existing sweep store** — through `main.py`.
- **`store retire` and `store fingerprint` comparisons**, as above.
- **`probe_01_tagged_read_batch.py`**, as above.

### The fourteen audit probes, before and after

Run at `479736d` (a detached worktree of `HEAD`, since removed) and on this tree, each in a
subprocess with `PYTHONPATH` at its root. No probe opens anything under `var/`: each builds its
schema in memory or a store in a temporary directory. `replicated_serial_census.py` takes store
paths as arguments; it was run with none, so it opened no store. Outputs compared after
normalising the repository path.

| Probe | Exit before → after | Output |
|---|---|---|
| `fk_sweep_probe.py` | 0 → 0 | **changed**: one new line, `quad_policy_serial FK=['QuadSourcePolicy.serial'] nullable=False` |
| `hot_journal_probe.py` | 0 → 0 | identical |
| `lookup_keys/probe_01_tagged_read_batch.py` | **0 → 1** | **newly fails**: `IntegrityError: NOT NULL constraint failed: QuadSourceIntegral.quad_policy_serial` |
| `lookup_keys/probe_01b_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `lookup_keys/probe_02_readback_guard_by_hand.py` | 0 → 0 | **changed**: `read_batch` moved from `@503` to `@546`, and its row binding from line 688 to 737; still "64 reads -> clean" |
| `lookup_keys/probe_03_column_existence.py` | 0 → 0 | **changed**: "checked 1245" → "checked 1254 resolvable column references" |
| `lookup_keys/probe_04_exit_time_and_solver.py` | 0 → 0 | identical |
| `lookup_keys/probe_04b_exact_match_fix_shape.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `lookup_keys/probe_05_value_lookups_and_read_batch.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `lookup_keys/probe_06_oneloop_tags_and_redshift_zero.py` | 1 → 1 | identical (already failing at `HEAD`) |
| `replicated_divergence_probe.py` | 0 → 0 | identical |
| `replicated_serial_census.py` | 0 → 0 | identical (no store argument) |
| `replicated_write_fault_probe.py` | 1 → 1 | identical but for the temporary directory name and a timestamp in one line (already failing at `HEAD`) |
| `tq_fk_probe.py` | 0 → 0 | identical |

## The `substitute_main_py()` output

`docs/handover/quadsource_atol_sweep.py` loaded by path with `importlib.util.spec_from_file_location`
and `substitute_main_py()` called on this tree (it only reads `main.py`). The five `** quadsource_atol_sweep:`
lines are its own, and it raises unless each literal occurs exactly once; the last five lines count
each literal in `main.py` and give its line:

```
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_ATOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of 'tol=DEFAULT_QUADRATURE_RTOL' in main.py
** quadsource_atol_sweep: replaced 1 occurrence of '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))' in main.py
substituted source: 162046 characters
count=1 line=3791 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)'
count=1 line=3803 'np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)'
count=1 line=3692 'tol=DEFAULT_QUADRATURE_ATOL'
count=1 line=3693 'tol=DEFAULT_QUADRATURE_RTOL'
count=1 line=3585 'qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))'
```

All five still match exactly once. The three that moved (`:3585`, `:3692`, `:3693`) each moved by the
three lines this commit adds above them (from `:3582`, `:3689`, `:3690`); the k-grid literals stay
at `:3791` and `:3803`. The sweep script is not edited.

## Deviations from the prompt

1. **The constructor's argument, the payload key and the property are named `quad_policy`.**
   `IMPLEMENTATION CHOICE` (the prompt's choice, with the reason given under "What shipped").
2. **The constructor also refuses `None`, with `ValueError`.** `IMPLEMENTATION CHOICE`. A required
   argument only fails on omission; `None` would otherwise fail later, in `store()`, on
   `.store_id`. Tested.
3. **One module-level helper, `_refuse_a_store_that_predates_the_policy_key`, serves both
   refusals.** `IMPLEMENTATION CHOICE`. 07 wrote the branch twice, in the two `build()`s; here
   the message and the match are one place, and the function name is a parameter of the message.
4. **`QuadSourcePolicy.read_table` is written as a list comprehension over the query's rows**,
   like `wavenumber.read_table`. `IMPLEMENTATION CHOICE`, the shape the note's §0.1 measured. It
   is inside the read-back guard's known blind spot
   (`[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`), so **no guard pin moved** (below).
   The shape was chosen because it matches the precedent, not to stay out of the guard's view; a
   loop would have added decidable entries to `EXPECTED_COVERAGE`, which §0.4 grants.
5. **The extraction script's lookup is a module-level function, `find_QuadSource_policy(pool)`,
   defined before `run_pipeline`.** `IMPLEMENTATION CHOICE`. The script is a program (it parses
   `sys.argv` and opens a datastore at module scope) and cannot be imported; a module-level
   function can be compiled out of it with `ast` for a test, as `load_main_py_functions` does for
   `main.py`. It refuses with `RuntimeError`.
6. **The fixture gains a second `QuadSourcePolicy` (serial 2, `qpolicy-b`, `Levin_threshold` 0.5,
   `maximize-WKB`).** `STRUCTURALLY REQUIRED`, granted by §0.3 and explained in §0.5:
   `TestIdentityColumnsMatter` varies a key field to another *real* parent, and with one policy
   the variation would be an unresolved parent, which drops out of the records and passes for the
   wrong reason. Its cost is in the fingerprint (`QuadSourcePolicy` 1 → 2, above).
7. **The new tests import their store helpers from `test_integral_read_batch.py`** (`Store`,
   `_exit_time_row`, `_exit_time_obj`, `_ModelProxy`, `_Ref`) rather than copying them.
   `IMPLEMENTATION CHOICE`. That module is not modified.
8. **A defect in this commit's own first draft, found by the mutation runs and fixed before the
   commit.** The first draft of `test_quadsource_policy_key.py` had a helper method named
   `inventory(`. `Datastore/tests/test_inventory_retired.py`'s `git grep` for the retired
   `.inventory(` service name only sees tracked files, so it passed while the new files were
   untracked and failed once they were staged; the first mutation runs showed it failing in all
   four. The method is renamed `read_the_store`, the six suites were re-run on the staged tree, and
   the four mutations re-run from scratch. Every figure in this log is from those re-runs.
   `UNINTENDED DRIFT` in the sense of the log template, caught and corrected within the commit; it
   left nothing in the tree.

No `UNINTENDED DRIFT` remains in the tree. `git diff --cached -U0` shows only the hunks this log
lists.

## Existing-test edits, by §0.3's list

| File | Edit | Reason |
|---|---|---|
| `ComputeTargets/tests/test_quadsource_integral.py` | `quad_policy=Policy(),` added after `policy=Policy(),` in `TestComputePayload.test_missing_transfer_function_keys_raise_RuntimeError_naming_them` (the audit's `:1127`) and in the shared `common` dict of `TestPersistedSchema.test_object_round_trips_the_new_fields` (the audit's `:2229` and `:2236` both use it). **Two edits, not three** | the constructor requires the policy; nothing asserted changed |
| `Datastore/tests/real_store_fixtures.py` | the three `QuadSourceIntegral` rows (`fixture-qsi-1`, `-2`, `-3`) gain `quad_policy_serial` 1; one new `QuadSourcePolicy` row, serial 2 | NOT NULL; Deviation 6 |
| `Datastore/tests/test_store_inventory.py` | `IDENTITY["QuadSourceIntegral"]` gains `"quad_policy": ("QuadSourceIntegral", 2, "quad_policy_serial", 2)` beside `"policy"` | the key's fields now include it |
| `Datastore/tests/test_integral_read_batch.py` | the two raw `QuadSourceIntegral` inserts gain `quad_policy_serial=1`; the two `read_batch` payloads gain `quad_policy=_Ref(1)` | NOT NULL column; the payload key `read_batch` now requires (§0.3, granted) |
| `Datastore/tests/test_schema_builder.py` | `WITNESS` → `schema_at_datastore-integrity-08.json`; the docstring sentence naming the current witness | D5 |
| `RunRegistry/tests/test_store_fingerprint.py` | `GOLDEN` → `full_store_fingerprint_datastore-integrity-08.json` | D5 |

`Datastore/tests/test_factory_select_columns.py` was **not modified**: **no guard pin moved.**
Before and after this commit (`python -c` over the module): `EXPECTED_COVERAGE` 47 → 47,
`KNOWN_UNRESOLVED_TABLES` 11 → 11, `KNOWN_BLIND` 1, `KNOWN_UNCHECKED_INSERTS` 13,
`KNOWN_BLIND_COLUMNS` 4, `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` empty. `read_table` sits in the
comprehension blind spot, and the refusal path reads no row.

## Verification performed

- **The starting tree.** `git status` clean at `479736d`. `RunRegistry list` was not run (no long job
  launched; nothing under `var/` opened).
- **The six suites**, README §7's command, serially, on the final staged tree:

  | Suite | Result |
  |---|---|
  | AdaptiveLevin | 32 OK |
  | ComputeTargets | **585 OK** (566 + 19; the known wall-clock flake did not fire in this run. It had failed once in the earlier run of the same tree and passed 3 of 3 alone) |
  | CosmologyModels | 39 OK |
  | Datastore | **422 OK** (397 + 21 + 4) |
  | LiouvilleGreen | 148 OK (1 skipped) |
  | RunRegistry | 199 OK |

- **The new modules, `-v`, unfiltered:**

  ```
test_another_database_error_is_not_turned_into_it (Datastore.tests.test_quadsource_policy_key.TestAnOldStoreIsRefused.test_another_database_error_is_not_turned_into_it)
The branch matches the driver's message for the column, and nothing else. ... ok
test_build (Datastore.tests.test_quadsource_policy_key.TestAnOldStoreIsRefused.test_build) ... ok
test_read_batch_untagged_and_tagged (Datastore.tests.test_quadsource_policy_key.TestAnOldStoreIsRefused.test_read_batch_untagged_and_tagged) ... ok
test_a_hit_and_a_miss_carry_the_policy_they_were_asked_under (Datastore.tests.test_quadsource_policy_key.TestKeyedOnThePolicy.test_a_hit_and_a_miss_carry_the_policy_they_were_asked_under) ... ok
test_build_requires_the_policy (Datastore.tests.test_quadsource_policy_key.TestKeyedOnThePolicy.test_build_requires_the_policy) ... ok
test_found_under_the_first_and_missed_under_the_second (Datastore.tests.test_quadsource_policy_key.TestKeyedOnThePolicy.test_found_under_the_first_and_missed_under_the_second) ... ok
test_the_same_computation_under_both_policies_coexists (Datastore.tests.test_quadsource_policy_key.TestKeyedOnThePolicy.test_the_same_computation_under_both_policies_coexists)
Otherwise identical rows are two rows, and each lookup finds its own: no ... ok
test_the_stored_row_records_its_policy (Datastore.tests.test_quadsource_policy_key.TestKeyedOnThePolicy.test_the_stored_row_records_its_policy) ... ok
test_a_policy_with_no_rows_finds_none (Datastore.tests.test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_a_policy_with_no_rows_finds_none) ... ok
test_read_batch_requires_the_policy (Datastore.tests.test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_read_batch_requires_the_policy) ... ok
test_tagged (Datastore.tests.test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_tagged) ... ok
test_the_objects_carry_the_policy_they_were_read_under (Datastore.tests.test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_the_objects_carry_the_policy_they_were_read_under) ... ok
test_untagged (Datastore.tests.test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_untagged) ... ok
test_every_match_is_returned_in_serial_order (Datastore.tests.test_quadsource_policy_key.TestReadTable.test_every_match_is_returned_in_serial_order)
Two rows build() would call the same policy. A reader must be able to refuse them. ... ok
test_it_finds_the_policy_by_its_parameters (Datastore.tests.test_quadsource_policy_key.TestReadTable.test_it_finds_the_policy_by_its_parameters) ... ok
test_no_match_is_an_empty_list_and_inserts_nothing (Datastore.tests.test_quadsource_policy_key.TestReadTable.test_no_match_is_an_empty_list_and_inserts_nothing) ... ok
test_the_threshold_is_matched_as_build_matches_it (Datastore.tests.test_quadsource_policy_key.TestReadTable.test_the_threshold_is_matched_as_build_matches_it)
Within ``DEFAULT_FLOAT_PRECISION``, not exactly; and not beyond it. ... ok
test_the_fingerprint_moves_with_the_policy (Datastore.tests.test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_fingerprint_moves_with_the_policy) ... ok
test_the_key_carries_the_policy (Datastore.tests.test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_key_carries_the_policy) ... ok
test_the_same_policy_would_be_a_duplicate (Datastore.tests.test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_same_policy_would_be_a_duplicate)
The control: the same two rows under one policy share an identity, which the ... ok
test_two_otherwise_identical_rows_under_two_policies_are_two_records (Datastore.tests.test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_two_otherwise_identical_rows_under_two_policies_are_two_records) ... ok
test_a_dangling_policy_is_reported (Datastore.tests.test_quadsource_policy_foreign_key.TestTheNewKey.test_a_dangling_policy_is_reported)
Policy 4 exists on no shard: the check reads the declared key. ... ok
test_every_shard_passes_foreign_key_check (Datastore.tests.test_quadsource_policy_foreign_key.TestTheNewKey.test_every_shard_passes_foreign_key_check) ... ok
test_the_added_row_is_not_satisfied_by_coincidence (Datastore.tests.test_quadsource_policy_foreign_key.TestTheNewKey.test_the_added_row_is_not_satisfied_by_coincidence) ... ok
test_the_default_stores_pass_too (Datastore.tests.test_quadsource_policy_foreign_key.TestTheNewKey.test_the_default_stores_pass_too) ... ok
test_each_site_passes_QuadSource_policy_1pt5 (ComputeTargets.tests.test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_each_site_passes_QuadSource_policy_1pt5) ... ok
test_nothing_in_run_pipeline_reaches_for_the_5pt0_policy (ComputeTargets.tests.test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_nothing_in_run_pipeline_reaches_for_the_5pt0_policy)
It is created, and passed into run_pipeline, and unused (D6). ... ok
test_the_5pt0_policy_is_still_created_from_literals (ComputeTargets.tests.test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_the_5pt0_policy_is_still_created_from_literals) ... ok
test_the_GkSourcePolicy_is_still_passed_beside_it (ComputeTargets.tests.test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_the_GkSourcePolicy_is_still_passed_beside_it)
The key gained a column; it did not swap one for the other. ... ok
test_the_two_sites_are_found (ComputeTargets.tests.test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_the_two_sites_are_found)
A finder that found nothing would pass everything below. ``test_main_plumbing``'s ... ok
test_a_populated_object_needs_it_too (ComputeTargets.tests.test_quadsource_policy_main.TestTheConstructorRequiresIt.test_a_populated_object_needs_it_too) ... ok
test_it_is_carried (ComputeTargets.tests.test_quadsource_policy_main.TestTheConstructorRequiresIt.test_it_is_carried) ... ok
test_none_is_an_error (ComputeTargets.tests.test_quadsource_policy_main.TestTheConstructorRequiresIt.test_none_is_an_error) ... ok
test_omitting_it_is_an_error (ComputeTargets.tests.test_quadsource_policy_main.TestTheConstructorRequiresIt.test_omitting_it_is_an_error) ... ok
test_a_different_policy_alone_is_no_row (ComputeTargets.tests.test_quadsource_policy_main.TestTheExtractionScript.test_a_different_policy_alone_is_no_row) ... ok
test_it_finds_the_policy_main_py_created (ComputeTargets.tests.test_quadsource_policy_main.TestTheExtractionScript.test_it_finds_the_policy_main_py_created) ... ok
test_it_is_not_confused_by_the_5pt0_policy (ComputeTargets.tests.test_quadsource_policy_main.TestTheExtractionScript.test_it_is_not_confused_by_the_5pt0_policy) ... ok
test_more_than_one_row_is_refused_naming_their_serials (ComputeTargets.tests.test_quadsource_policy_main.TestTheExtractionScript.test_more_than_one_row_is_refused_naming_their_serials) ... ok
test_no_row_is_refused_naming_the_parameters_and_writes_nothing (ComputeTargets.tests.test_quadsource_policy_main.TestTheExtractionScript.test_no_row_is_refused_naming_the_parameters_and_writes_nothing) ... ok
test_the_control_object_get_is_a_different_call (ComputeTargets.tests.test_quadsource_policy_main.TestTheExtractionScript.test_the_control_object_get_is_a_different_call)
The reason for read_table (§0.1): the pool's replicated write commits an in-flight ... ok
test_main_py_creates_the_1pt5_policy_from_it (ComputeTargets.tests.test_quadsource_policy_main.TestTheParametersAreOneConstant.test_main_py_creates_the_1pt5_policy_from_it) ... ok
test_main_py_imports_it (ComputeTargets.tests.test_quadsource_policy_main.TestTheParametersAreOneConstant.test_main_py_imports_it) ... ok
test_the_extraction_script_imports_it_and_hard_codes_no_quad_policy (ComputeTargets.tests.test_quadsource_policy_main.TestTheParametersAreOneConstant.test_the_extraction_script_imports_it_and_hard_codes_no_quad_policy) ... ok
test_the_values (ComputeTargets.tests.test_quadsource_policy_main.TestTheParametersAreOneConstant.test_the_values)
What ``main.py`` spelled out before prompt 08. They name the row in every existing ... ok

----------------------------------------------------------------------
Ran 44 tests in 3.360s

OK
  ```

- **Formatting.** `./venv/bin/python -m black --check` on the 18 Python files this commit touches:
  "18 files would be left unchanged".
- **A wrong foreign key would be caught.** With `ForeignKey("GkSourcePolicy.serial")` in place of
  `ForeignKey("QuadSourcePolicy.serial")`, `test_quadsource_policy_foreign_key` failed 2 of its 4
  tests (`test_a_dangling_policy_is_reported`, `test_every_shard_passes_foreign_key_check`); reverted.
  (Not one of the prompt's four mutations, so not in the record below.)
- **Scope.** `git diff --cached --name-only` is in the final report and was checked against the
  brief's list. Staging was by explicit path only.
- **No Ray, nothing under `var/`.** No test module starts Ray, and each new one asserts it is not
  initialised at teardown. The stand-in shards are in temporary directories. No store under `var/`
  was opened.

## The deliberate-breakage record

Each mutation was applied with `git apply` to the staged tree of this commit, so that `git diff`
showed the mutation alone. Then the `Datastore`, `ComputeTargets` and `RunRegistry` suites were run
serially and the mutation reverted with `git apply -R`, checking afterwards that `git diff -- '*.py'`
was empty. AdaptiveLevin, CosmologyModels and LiouvilleGreen were not run under the mutations: none
imports a mutated module.

The diffs are pasted from `git diff`, exactly as applied. Each failing entry is one `FAIL:` or
`ERROR:` line of the unittest output; a failing sub-test is its own entry, with its parameters.

**Replays.** Before the commit, each of the four diff blocks below was extracted from this file and
applied with plain `git apply`, the same three suites were run, and it was reverted; the result is
in the final report, not here, since this file cannot record a run that follows itself. After the
commit they are replayed again from `git show HEAD:prompts/datastore-integrity/logs/08-key-quadsourceintegral-on-its-policy.md`;
that result goes to the orchestrator, and is not written here.

### Mutation (i) — `quad_policy_serial` dropped from `build()`'s filter

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 053f620..93d7494 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -303,7 +303,6 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
         ).filter(
             table.c.model_serial == model_proxy.store_id,
             table.c.policy_serial == policy.store_id,
-            table.c.quad_policy_serial == quad_policy.store_id,
             table.c.k_wavenumber_exit_serial == k.store_id,
             table.c.q_wavenumber_exit_serial == q.store_id,
             table.c.r_wavenumber_exit_serial == r.store_id,
```

`Datastore`: `Ran 422 tests in 127.716s`, `FAILED (failures=2, errors=1)`; 3 failing entries, 0 of them sub-tests:

- `ERROR` `test_quadsource_policy_key.TestKeyedOnThePolicy.test_the_same_computation_under_both_policies_coexists` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestAnOldStoreIsRefused.test_build` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestKeyedOnThePolicy.test_found_under_the_first_and_missed_under_the_second` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 585 tests in 145.265s`, `OK`.

`RunRegistry`: `Ran 199 tests in 62.644s`, `OK`.

### Mutation (ii) — `quad_policy_serial` dropped from `read_batch`'s filter

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 053f620..de86e8f 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -669,7 +669,6 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
             .filter(
                 table.c.model_serial == model_proxy.store_id,
                 table.c.policy_serial == policy.store_id,
-                table.c.quad_policy_serial == quad_policy.store_id,
                 table.c.k_wavenumber_exit_serial == k.store_id,
                 table.c.atol_serial == atol.store_id,
                 table.c.rtol_serial == rtol.store_id,
```

`Datastore`: `Ran 422 tests in 123.271s`, `FAILED (failures=4)`; 4 failing entries, 0 of them sub-tests:

- `FAIL` `test_quadsource_policy_key.TestAnOldStoreIsRefused.test_read_batch_untagged_and_tagged` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_a_policy_with_no_rows_finds_none` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_tagged` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestReadBatchFiltersOnIt.test_untagged` — the test itself (no sub-tests)

`ComputeTargets`: `Ran 585 tests in 144.056s`, `OK`.

`RunRegistry`: `Ran 199 tests in 67.755s`, `OK`.

### Mutation (iii) — `QuadSource_policy_5pt0` passed at the compute call in `main.py`

```diff
diff --git a/main.py b/main.py
index b0ee7cf..43a6649 100644
--- a/main.py
+++ b/main.py
@@ -3518,7 +3518,7 @@ def run_pipeline(
                         "QuadSourceIntegral",
                         model=model_proxy,
                         policy=GkSource_policy_1pt5,
-                        quad_policy=QuadSource_policy_1pt5,
+                        quad_policy=QuadSource_policy_5pt0,
                         k=k,
                         q=q,
                         r=r,
```

`Datastore`: `Ran 422 tests in 129.919s`, `OK`.

`ComputeTargets`: `Ran 585 tests in 145.333s`, `FAILED (failures=2)`; 2 failing entries, 1 of them sub-tests:

- `FAIL` `test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_each_site_passes_QuadSource_policy_1pt5` — sub-test `site='construction'`
- `FAIL` `test_quadsource_policy_main.TestMainPyKeysOnQuadSourcePolicy1pt5.test_nothing_in_run_pipeline_reaches_for_the_5pt0_policy` — the test itself (no sub-tests)

`RunRegistry`: `Ran 199 tests in 63.320s`, `OK`.

### Mutation (iv) — `quad_policy` omitted from `inventory_records`

```diff
diff --git a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
index 053f620..303e22d 100644
--- a/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
+++ b/Datastore/SQL/ObjectFactories/QuadSourceIntegral.py
@@ -877,7 +877,6 @@ class sqla_QuadSourceIntegral_factory(SQLAFactoryBase):
             parents={
                 "model": Parent("model_serial", "BackgroundModel"),
                 "policy": Parent("policy_serial", "GkSourcePolicy"),
-                "quad_policy": Parent("quad_policy_serial", "QuadSourcePolicy"),
                 "k": Parent("k_wavenumber_exit_serial", "wavenumber_exit_time"),
                 "q": Parent("q_wavenumber_exit_serial", "wavenumber_exit_time"),
                 "r": Parent("r_wavenumber_exit_serial", "wavenumber_exit_time"),
```

`Datastore`: `Ran 422 tests in 124.504s`, `FAILED (failures=5)`; 5 failing entries, 2 of them sub-tests:

- `FAIL` `test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_fingerprint_moves_with_the_policy` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_the_key_carries_the_policy` — the test itself (no sub-tests)
- `FAIL` `test_quadsource_policy_key.TestTheInventoryIdentityIncludesIt.test_two_otherwise_identical_rows_under_two_policies_are_two_records` — the test itself (no sub-tests)
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_each_identity_column_changes_the_records` — sub-test `cls='QuadSourceIntegral', field='quad_policy'`
- `FAIL` `test_store_inventory.TestIdentityColumnsMatter.test_the_key_fields_are_the_lookup_columns` — sub-test `cls='QuadSourceIntegral'`

`ComputeTargets`: `Ran 585 tests in 143.149s`, `OK`.

`RunRegistry`: `Ran 199 tests in 64.064s`, `FAILED (failures=1)`; 1 failing entries, 0 of them sub-tests:

- `FAIL` `test_store_fingerprint.TestGolden.test_the_golden_fingerprint` — the test itself (no sub-tests)

Mutation (iii) is caught only by the `ast` tests of `test_quadsource_policy_main.py`, as designed:
`main.py` is never run, so no store-level test can see it. Mutation (iv) is caught by three tests
of the new module, by the two of `test_store_inventory` whose `IDENTITY` gained the entry, and by
the golden fingerprint.

## Observations not acted on

1. **`extract_QuadSourceIntegral_data.py`'s `GkSourcePolicy` lookups are `object_get`s** → opened
   `[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]` (board §3.3).
   `object_get` on a replicated class inserts on a miss, and even a hit commits and clears a
   `replication_in_flight` record on the primary. `extract_GkSource_data.py` makes the same pair of
   calls; no other script was audited for it. Not changed: the prompt and §0.1 bound the fix to the
   `QuadSourcePolicy` lookup.
2. **`docs/tolerance-convergence/inventory.py:750` cites `MetadataConcepts/QuadSourcePolicy.py:25-36`
   by line.** The docstring edit moved those lines (the note's §0.6). Not edited.
3. **`probe_01_tagged_read_batch.py` stops running.** Recorded above; by the user's decision the
   audit's probes are left as they are.
4. **`[05c-read-back-guard-cannot-see-reads-in-a-comprehension]`'s list of affected sites is one
   longer.** `QuadSourcePolicy.read_table` has the shape (a comprehension whose element reads its own
   target's attributes). Its reads are of the four columns the query selects, checked by hand. That
   issue's "Where, on this tree" list was not edited.
5. **`prompts/store-fingerprint/IMPLEMENTATION_STATE.md` line 134 still says "Seven issues remain
   open in §3"**, a sentence written when that board closed on 2026-09-25 and stale since before
   this prompt. The brief bounded the edit there to the §3 lead-in's count, which is corrected.
6. **`test_inventory_retired`'s `git grep` sees only tracked files**, so a test file that names a retired
   service passes until it is staged (Deviation 8). Not changed.
7. **Nothing here has run under Ray or on a real store.** Prompt 10's end-to-end store is the first
   place `main.py`'s keyed QSI stage and the extraction script's `read_table` meet production data.

## State handed to the next prompt

- **Names.** The column `quad_policy_serial` (README §4); the payload key and constructor argument
  `quad_policy`; `config.defaults.DEFAULT_QUADSOURCE_POLICY_1PT5`;
  `sqla_QuadSourcePolicy_factory.read_table` and its `read_table_config` entry;
  `extract_QuadSourceIntegral_data.find_QuadSource_policy`.
- **`QuadSourceIntegral`'s key is now** model, `GkSourcePolicy`, `QuadSourcePolicy`, `k`, `q`, `r`,
  `z_response`, `z_source_max`, `atol`, `rtol`, with its tags. **For 09:** `source_serial`,
  `data_serial` and the Tk rows still to be added; 09's re-capture bumps `FINGERPRINT_FORMAT` to 4
  by `stores.py`'s own rule, with its own golden beside the 08 one. The witness to diff from is
  `schema_at_datastore-integrity-08.json`, and the drop-order test's schema-derived check will hold
  09's new foreign keys to `_drop_order` too.
- **The refusal pattern**: match `str(getattr(e, "orig", e))` for the new column's name, in
  `build()` *and* `read_batch()`, as `_refuse_a_store_that_predates_the_policy_key` does.
- **The guard's pins:** `EXPECTED_COVERAGE` 47; `KNOWN_UNRESOLVED_TABLES` 11; `KNOWN_BLIND` 1,
  `KNOWN_UNCHECKED_INSERTS` 13 and `KNOWN_BLIND_COLUMNS` 4; `KNOWN_UNFIXED` and
  `KNOWN_UNFIXED_COLUMNS` empty. None moved.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets **585**, CosmologyModels 39,
  Datastore **422**, LiouvilleGreen 148 (1 skipped), RunRegistry 199.
