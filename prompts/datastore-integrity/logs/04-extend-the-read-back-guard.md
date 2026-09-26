# Log 04 — Extend the read-back guard to read_batch, _build_impl_*, and column existence

**Prompt:** [`prompts/datastore-integrity/04-extend-the-read-back-guard.md`](../04-extend-the-read-back-guard.md)
**Commit:** *(this commit)* — "Extend the read-back guard to read_batch and column existence"
**Model:** Claude Sonnet 5
**Date:** 2026-09-26
**Result:** COMPLETE. G1–G3 shipped. No §5 stop condition was met: every defect the extended guard
found is on a path `main.py` reaches only from a hand-written probe or a script, never during a
build or a resume (see "Live-path classification" below); the extension needed no analyser that
runs factory code (`ast` only, as before); and no existing test other than
`Datastore/tests/test_factory_select_columns.py` changed. `Datastore` rises from 232 to 235 (three
new `TestColumnExistence` tests; G1 added no new test methods, only extended the four existing
ones' reach). `[01-read-batch-is-outside-the-guard]` closes on the `datastore-readback` board's §4.

## What shipped

**G1 — every read path.**
- `analyse_factories()` now visits, per class, `build()` **and** every `_build_impl_serial` /
  `_build_impl_model` helper (`_ROW_HELPERS`), calling `analyse_build` on each directly — they
  already have the same straight-line "look up, then unpack" shape `build()` does, so nothing
  extra was needed to make them visible.
- For `read_batch`, a new `inline_nested_row_helper(fn)` copies the function, finds its nested
  `make_obj`/`make_object` helper, finds the row variable the comprehension that calls the helper
  iterates over (`for row in row_data` from `[make_obj(row) for row in row_data]`), deletes the
  nested `FunctionDef` and splices its body in as a `for <param> in <row_var>:` loop just before the
  function's `return`. The result is walked by the unmodified `analyse_build`, which now sees the
  reads that used to be invisible (the helper's parameter is not a query result the analyser can
  resolve on its own). This is the same method `probe_02_readback_guard_by_hand.py` uses "by hand";
  the prompt's instruction was to make it a test, and it is now `analyse_read_batch`, called for
  every class method and module-level function named `read_batch`.
- `OneLoopIntegral.read_batch` is a module-level function, not a method, so `analyse_factories` also
  walks `tree.body` of every module for a top-level `read_batch`. Its `make_object` is defined but
  never called and the function has no `return` — inlining finds no comprehension to link to, so
  `inline_nested_row_helper` returns `(fn, None)` and `analyse_build` sees no bindable row at all.
  A hole that produces zero findings is invisible to `test_analyser_coverage_is_undiminished` and
  `test_blind_spots_are_the_declared_ones` alike — exactly the silently-blind failure mode the guard
  exists to refuse — so `analyse_read_batch` manufactures one undecidable `Finding` for the whole
  function when inlining links nothing, which puts `"OneLoopIntegral.py:<module-level>.read_batch"`
  into `KNOWN_BLIND` with the reason, rather than nowhere.

**G2 — column existence** (`TestColumnExistence`, same module).
- `column_existence_problems()` builds the schema once (`build_schema(MetaData(), _factories)`,
  no Ray, no engine, no store) and walks every factory function's AST for `<var>.c.<name>` /
  `<var>.c["<name>"]`, resolving `<var>` to a schema table the way `probe_03_column_existence.py`
  does: the function's own `table`, a `tables["X"]`, a local bound to either, or an `.alias(...)` of
  either. A resolved table with a name absent from its schema columns is a problem; a resolved table
  with a **dynamic** name (an f-string, or a plain loop variable) is declared in
  `KNOWN_BLIND_COLUMNS` instead, never silently dropped. It also walks every
  `inserter(conn, {...})` / `value_inserter(conn, {...})` call and checks the dict's literal keys
  the same way.
- Two tests mirror G1's pattern: `test_column_references_resolve_against_schema` (the guard proper)
  and `test_known_unfixed_columns_are_still_present` (a struck-but-not-fixed entry fails, mirroring
  `test_known_unfixed_entries_are_still_present`); `test_dynamic_column_sites_are_the_declared_ones`
  mirrors `test_blind_spots_are_the_declared_ones` for the dynamic-name holes.

**G3 — record, do not fix.**
- `KNOWN_UNFIXED` gained two entries (both `WKB_criterio`, GkWKBValue and TkWKBValue), each naming
  P6.
- `KNOWN_UNFIXED_COLUMNS` (new) holds the three column-existence defects: `GkSource.py`'s
  `z_serial` and `QuadSource.py`'s `model_id` (P6), and `OneLoopIntegral.py`'s `validated` insert
  key (P8).
- `KNOWN_BLIND` gained the `OneLoopIntegral.py:<module-level>.read_batch` entry above.
  `KNOWN_BLIND_COLUMNS` (new) holds four sites: the two f-string loops already known from the audit
  (`wavenumber.py`'s and `QuadSourceIntegral.py`'s `z_exit_..._e{z_offset}`), plus two more this
  extension found that the audit's probe did not enumerate — `table.c[name] for name in
  key_columns` in `wavenumber_exit_time.store` and `BackgroundModel.store`'s replica
  insert-or-verify check (prompt 01, W3): `name` is a loop variable, not a literal, even though
  `key_columns` a few lines above it is a tuple of string literals. Resolving it would mean
  tracking a loop variable back to the tuple it ranges over, which this resolver does not do; it is
  a **new** entry, not one of the audit's five, and is declared rather than silently skipped.
- The module docstring's points 3 and 6 are rewritten as the prompt specified; no other point
  changed.

## Deviations from the prompt

1. **IMPLEMENTATION CHOICE — `EXPECTED_COVERAGE` gained 10 entries, not `probe_02.out`'s 16.**
   `Finding.key` is `f"{module}:{cls}"` (no function name in it — a convention already in the
   module, e.g. a variable rebound mid-`build()` is one key, not two) and `EXPECTED_COVERAGE`'s
   identity is `f"{finding.key}.{finding.row_var}"`. A class with both a `_build_impl_serial` and a
   `_build_impl_model` binds the same `row_data` name in both, so the two collapse to one key
   (`GkNumericValue`, `GkSourceValue`, `GkWKBValue`, `QuadSourceValue`, `TkNumericValue`,
   `TkWKBValue` all do); a class with only `read_batch` contributes one more key, `row`
   (`QuadSourceIntegral`); a class with both `_build_impl_*` and `read_batch` contributes two, not
   three (`GkNumericValue`, `GkWKBValue`, `TkWKBValue`). Counting: `GkNumericValue` 2 + `GkSourceValue`
   1 + `GkWKBValue` 2 + `QuadSourceValue` 1 + `QuadSourceIntegral` 1 + `TkNumericValue` 1 +
   `TkWKBValue` 2 = **10**. Verified directly: adding the 16 probe-shaped keys made
   `test_analyser_coverage_is_undiminished` fail, naming exactly these 10 as "in the first set
   [actually produced] but not the second [expected]" before the fix; adding the correct 10 made
   all four `TestFactorySelectColumns` tests pass. This is a difference in what the *existing* test
   module's identity granularity is, not a defect in the probe or the extension: widening the key to
   include the function name was rejected because it would also have had to be applied
   retroactively to every existing `build()` entry (none of which currently carries a function
   name), which is a larger, unrequested change to a convention prompt 04 did not ask to revisit.
2. **IMPLEMENTATION CHOICE — G2's `KNOWN_BLIND_COLUMNS` names two sites beyond the audit's two.**
   Section 3.2 of the prompt says "declare in `KNOWN_BLIND` what cannot be resolved, with the
   reason"; `probe_03_column_existence.py` silently skips a dynamic subscript rather than reporting
   it, so its `.out` file does not enumerate them. Running the resolver over every factory function
   (not only the two the audit's prose names) found two more resolvable-table/dynamic-name sites —
   `wavenumber.py:sqla_wavenumber_exit_time_factory.store` and
   `BackgroundModel.py:sqla_BackgroundModelFactory.store`, both `table.c[name] for name in
   key_columns` from prompt 01's replica insert-or-verify check. Declared, not silently dropped, per
   the prompt's instruction; noted here as **new**, beyond the audit's citations.
3. **IMPLEMENTATION CHOICE — `inline_nested_row_helper` and `analyse_read_batch` are written fresh
   in the test module**, not imported from `probe_02_readback_guard_by_hand.py`. The prompt forbids
   editing the probes but says nothing against importing from them; importing was rejected because
   it would make a permanent test module depend on a one-off audit script under `docs/`, which nothing
   else in `Datastore/tests/` does, and because the probe's version does one thing this prompt does
   not need (the `.label(...)` mislabelled-flag scan, left for prompt 05 per the rewritten docstring
   point 3). The logic itself — copy, find the nested helper, find the row variable via the calling
   `ListComp`, splice as a `for` loop — is the same method, by design ("they are the method, run by
   hand; this prompt makes them tests").

No `STRUCTURALLY REQUIRED` or `UNINTENDED DRIFT` deviations.

## EXPECTED_COVERAGE against probe_02.out's 16

`probe_02.out` lists 16 bindings (one per function: `_build_impl_serial`, `_build_impl_model` and
`read_batch` counted separately even when they share a class). `EXPECTED_COVERAGE` gained **10**,
because the binding identity this test module already used before this prompt —
`"module:class.row_var"`, with no function in it — merges a class's `_build_impl_serial` and
`_build_impl_model` (both bind `row_data`) into one key. See deviation 1 above for the arithmetic
and the verification that produced it.

## KNOWN_UNFIXED / KNOWN_UNFIXED_COLUMNS, with P-items

| Entry | P-item | Found by |
|---|---|---|
| `GkWKBIntegration.py:sqla_GkWKBValue_factory` → `WKB_criterio` | P6 | G1 (predicted) |
| `TkWKBIntegration.py:sqla_TkWKBValue_factory` → `WKB_criterio` | P6 | G1 (predicted) |
| `GkSource.py`/`sqla_GkSourceValue_factory`/`_build_impl_serial` → `z_serial` not a column of `GkSourceValue` | P6 | G2 (predicted) |
| `QuadSource.py`/`sqla_QuadSourceValue_factory`/`_build_impl_model` → `model_id` not a column of `QuadSource` | P6 | G2 (predicted) |
| `OneLoopIntegral.py`/`sqla_OneLoopIntegral_factory`/`store` → `validated` not a column of `OneLoopIntegral` [insert key] | P8 | G2 (predicted) |

All five are exactly the five the prompt predicted from the probes; nothing new was found by G1 or
G2's "guard proper" tests. (Two more sites *were* found, but by the dynamic-column-name check, not
the existence check — see `KNOWN_BLIND_COLUMNS` above and deviation 2.)

## KNOWN_BLIND / KNOWN_BLIND_COLUMNS additions, with reasons

- `KNOWN_BLIND["OneLoopIntegral.py:<module-level>.read_batch"]` — `make_object(row_data)` is
  defined but never called, and the function has no `return` statement, so no row variable can be
  linked to a query. (Predicted: the prompt names this exact case in §2 G1.)
- `KNOWN_BLIND_COLUMNS["wavenumber.py:sqla_wavenumber_exit_time_factory.build"]` — the two
  `z_exit_..._e{z_offset}` f-string loops (audit §3.10, predicted).
- `KNOWN_BLIND_COLUMNS["QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.read_batch"]` — the
  four `z_exit_..._e{z_offset}` f-string loops there (predicted).
- `KNOWN_BLIND_COLUMNS["wavenumber.py:sqla_wavenumber_exit_time_factory.store"]` — **new**,
  `table.c[name] for name in key_columns` (deviation 2).
- `KNOWN_BLIND_COLUMNS["BackgroundModel.py:sqla_BackgroundModelFactory.store"]` — **new**, same
  pattern (deviation 2).

## Live-path classification (§5's stop condition)

Checked by grepping `main.py`, every `ComputeTargets/*.py` and every `extract_*.py` for a call to
`build()`/`object_get` on each affected class, since the prompt's stop condition is conditioned on
whether `main.py` reaches the defect **during a build**, not only on resume or extraction:

- **`GkWKBValue`, `TkWKBValue`, `GkSourceValue`, `QuadSourceValue`** (`_build_impl_serial` /
  `_build_impl_model`, the `WKB_criterio`, `z_serial` and `model_id` defects). `main.py` reads these
  four classes' rows only through `pool.object_read_batch(...)` (`main.py:2429`-area, and the
  `ComputeTargets` parent targets' `.values` properties, `extract_GkWKB_data.py`,
  `extract_TkWKB_data.py`, `extract_GkSource_data.py`), never through `object_get`/`build()`. No
  call anywhere in `main.py`, `ComputeTargets/` or `extract_*.py` passes `wkb_serial`, a `model`+`k`+
  `z_source` triple to these classes' `build()`, or a `model`+`q`+`r` pair to `QuadSourceValue`'s.
  The only callers of `_build_impl_serial`/`_build_impl_model` in the whole tree are
  `docs/datastore-integrity-audit/lookup_keys/probe_05_value_lookups_and_read_batch.py` and this
  guard's own analyser (which never executes them). **Classification: reached by neither a build
  nor a resume nor an extraction script — dead code today**, consistent with the audit's own framing
  (probe_02 "runs the guard's own analyser ... by hand", not from a live call site).
- **`OneLoopIntegral.store`** (the `validated` insert key). Audit §3.12, reconfirmed by grep: no
  site in `main.py` builds `OneLoopIntegral` at all (`ComputeTargets/tests/test_main_plumbing.py`
  pins `"OneLoopIntegral" not in OBJECT_TYPE_TOLERANCES` and not in the build sites). **Classification:
  reached by nothing** — the target is never built, matching D3 and the existing
  `[00-oneloop-factory-cannot-round-trip-a-row]` issue.

No entry is on a path `main.py` reaches during a build. No stop condition applies.

## Verification performed

- All four original `TestFactorySelectColumns` tests still pass, unchanged in name and assertion,
  against the extended analyser and the grown `KNOWN_UNFIXED`/`EXPECTED_COVERAGE`/`KNOWN_BLIND`.
- The three new `TestColumnExistence` tests pass.
- `probe_02_readback_guard_by_hand.py` and `probe_03_column_existence.py` re-run unmodified against
  this tree (not edited) and produce the same defects this commit records (the three-defect
  `probe_03` count differs only in "checked 1186" vs the document's "1176", from unrelated source
  growth since the audit was written on 2026-09-25; the same three problems are named).
- `black --check Datastore/tests/test_factory_select_columns.py` is clean.
- `git status --short` and `git diff --cached` were clean before starting and are clean after
  committing (checked).

## The deliberate-breakage record

Each diff below is exactly as applied (via `git apply` from the repository root against the tree
this commit produces), with the resulting test outcome, then reverted before the next. All five
were also checked with `git apply --check` against the final committed tree after the fact.

**(i) rename a column read in one `read_batch`'s nested helper to one not selected**
(`GkWKBValue.read_batch`'s `make_obj`, since `TkNumericValue` — the prompt's example — has no
`read_batch`; `GkWKBValue.read_batch` is used instead, noted as an implementation choice):

```diff
--- a/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkWKBIntegration.py
@@ -1224,7 +1224,7 @@ class sqla_GkWKBValue_factory(SQLAFactoryBase):
                 theta_div_2pi=row.theta_div_2pi,
                 omega_WKB_sq=row.omega_WKB_sq,
                 WKB_criterion=row.WKB_criterion,
-                G_WKB=row.G_WKB,
+                G_WKB=row.G_WKB_typo,
                 analytic_G_rad=row.analytic_G_rad,
                 analytic_Gprime_rad=row.analytic_Gprime_rad,
                 analytic_G_w=row.analytic_G_w,
```

Result: `FAILED (failures=2)` — `test_build_reads_only_columns_its_select_requests` (the merged
`GkWKBIntegration.py:sqla_GkWKBValue_factory` key now also carries `G_WKB_typo`) and
`test_known_unfixed_entries_are_still_present` (the recorded `['WKB_criterio']` no longer equals
the actual `['G_WKB_typo', 'WKB_criterio']`). Reverted; tree matched the pre-mutation state.

**(ii) rename a column read in one `_build_impl_serial`** (`GkNumericValue`'s):

```diff
--- a/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
+++ b/Datastore/SQL/ObjectFactories/GkNumericIntegration.py
@@ -801,7 +801,7 @@ class sqla_GkNumericValue_factory(SQLAFactoryBase):
             attribute_set = {"_new_insert": True}
         else:
             store_id = row_data.serial
-            analytic_G_rad = row_data.analytic_G_rad
+            analytic_G_rad = row_data.analytic_G_rad_typo
             analytic_Gprime_rad = row_data.analytic_Gprime_rad
             analytic_G_w = row_data.analytic_G_w
             analytic_Gprime_w = row_data.analytic_Gprime_w
```

Result: `FAILED (failures=1)` — `test_build_reads_only_columns_its_select_requests` only (this
class had no prior `KNOWN_UNFIXED` entry to collide with). Reverted.

**(iii) rename a `table.c.<name>` in any factory to a column that does not exist**
(`wavenumber.py`'s exit-time `build()`, a filter column not read back by the object, to isolate G2
from G1):

```diff
--- a/Datastore/SQL/ObjectFactories/wavenumber.py
+++ b/Datastore/SQL/ObjectFactories/wavenumber.py
@@ -216,7 +216,7 @@ class sqla_wavenumber_exit_time_factory(SQLAFactoryBase):
                 .filter(
                     sqla.and_(
                         table.c.wavenumber_serial == k.store_id,
-                        table.c.cosmology_type == cosmology.type_id,
+                        table.c.cosmology_type_typo == cosmology.type_id,
                         table.c.cosmology_serial == cosmology.store_id,
                         atol_table.c.log10_tol - target_atol.log10_tol
                         <= DEFAULT_FLOAT_PRECISION,
```

Result: `FAILED (failures=1)` — `test_column_references_resolve_against_schema` names
`('wavenumber.py', 'sqla_wavenumber_exit_time_factory', 'build', 'wavenumber_exit_time',
'cosmology_type_typo')`. Reverted.

**(iv) delete one `KNOWN_UNFIXED` entry without fixing it** (the `GkWKBValue` one):

```diff
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -494,9 +494,6 @@
 # column are both `WKB_criterion`. Real -- every hit raises AttributeError -- and carried
 # deliberately rather than fixed here; prompt 05's P6 empties it.
 KNOWN_UNFIXED = {
-    "GkWKBIntegration.py:sqla_GkWKBValue_factory": [
-        "WKB_criterio"
-    ],  # P6; GkWKBIntegration.py:947
     "TkWKBIntegration.py:sqla_TkWKBValue_factory": [
         "WKB_criterio"
     ],  # P6; TkWKBIntegration.py:1001
```

Result: `FAILED (failures=1)` — `test_build_reads_only_columns_its_select_requests` (the actual
shortfall dict still has the `GkWKBValue` key; `KNOWN_UNFIXED` no longer does). Reverted.

**(v) make the analyser skip `read_batch` again** (and the `_build_impl_*` helpers, restoring the
original `build()`-only loop):

```diff
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -462,16 +462,12 @@
     for path in sorted(FACTORY_DIR.glob("*.py")):
         tree = ast.parse(path.read_text(), filename=str(path))
         for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
-            for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
-                if fn.name == "build" or fn.name in _ROW_HELPERS:
-                    findings.extend(analyse_build(fn, path.name, cls.name))
-                elif fn.name == "read_batch":
-                    findings.extend(analyse_read_batch(fn, path.name, cls.name))
-        # OneLoopIntegral.py's read_batch is a module-level function, not a method, and build()
-        # delegates nothing to it -- it is simply never wired up (audit §3.12).
-        for fn in tree.body:
-            if isinstance(fn, ast.FunctionDef) and fn.name == "read_batch":
-                findings.extend(analyse_read_batch(fn, path.name, "<module-level>"))
+            for fn in [
+                n
+                for n in cls.body
+                if isinstance(n, ast.FunctionDef) and n.name == "build"
+            ]:
+                findings.extend(analyse_build(fn, path.name, cls.name))
     return findings
```

Result: `FAILED (failures=4)` — `test_analyser_coverage_is_undiminished` (the 10 added bindings
vanish from `covered`), `test_blind_spots_are_the_declared_ones` (the manufactured
`OneLoopIntegral.py:<module-level>.read_batch` blind entry vanishes too), and both
`KNOWN_UNFIXED`/`test_build_reads_only_columns_its_select_requests` /
`test_known_unfixed_entries_are_still_present` (the two `WKB_criterio` entries no longer have a
matching shortfall, since their only source — `_build_impl_serial` — is no longer visited). As
required, `test_analyser_coverage_is_undiminished` is among the failures. Reverted.

All five diffs were re-verified with `git apply --check` against the tree at this commit's hash
(from a copy in a scratch directory, not the repository), and each applies and reverts cleanly.

## Observations not acted on

1. **Two `KNOWN_BLIND_COLUMNS` sites beyond the audit's citations** — see deviation 2. Not a defect;
   recorded rather than fixed, per this prompt's own rule (record, don't fix) and CLAUDE.md's rule 4.
2. **`_build_impl_serial` / `_build_impl_model` for the four Value factories, and the whole of
   `OneLoopIntegral`'s factory, are dead code today** — no caller in `main.py`, `ComputeTargets/` or
   `extract_*.py` reaches them (see "Live-path classification"). Not opened as a new issue: this is
   restating what the existing `[00-oneloop-factory-cannot-round-trip-a-row]` issue already says for
   `OneLoopIntegral`, and the audit already frames the Value-factory helpers as reached only "by
   hand" (probe_02); nothing here narrows or contradicts either.
3. **`probe_03`'s "checked" count moved from 1176 to 1186`** between the audit (2026-09-25) and this
   commit (2026-09-26), from ordinary source growth elsewhere in the tree (e.g. prompt 01's
   `wavenumber.py`/`BackgroundModel.py` replica checks, which added `table.c[name]` references of
   their own — see deviation 2). The three defects it finds are unchanged.

## State handed to the next prompt

- **For prompt 05.** `KNOWN_UNFIXED` (2 entries) and `KNOWN_UNFIXED_COLUMNS` (3 entries) in
  `Datastore/tests/test_factory_select_columns.py` are the list to empty, each naming its P-item
  (P6 ×4, P8 ×1). `test_known_unfixed_entries_are_still_present` /
  `test_known_unfixed_columns_are_still_present` force every fix to strike its own entry.
- **Names.** `analyse_read_batch`, `inline_nested_row_helper`, `_ROW_HELPERS` (G1);
  `column_existence_problems`, `_table_of`, `_column_ref`, `ColumnProblem`, `KNOWN_BLIND_COLUMNS`,
  `KNOWN_UNFIXED_COLUMNS`, `TestColumnExistence` (G2).
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **235** (232 + 3), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
