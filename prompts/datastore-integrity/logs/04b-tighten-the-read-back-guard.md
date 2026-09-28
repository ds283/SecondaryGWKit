# Log 04b — Tighten the read-back guard: every function, per-function coverage, nothing skipped unsaid

**Prompt:** [`prompts/datastore-integrity/04b-tighten-the-read-back-guard.md`](../04b-tighten-the-read-back-guard.md)
**Commit:** *(this commit)* — "Tighten the read-back guard to every function and every declared hole"
**Model:** Claude Sonnet 5
**Date:** 2026-09-28
**Result:** COMPLETE. G4–G6 shipped. No §5 stop condition was met: the tightened guard finds no
new defect (`KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS` are byte-identical to prompt 04's, in
keying and contents); every one of 04's 30 `EXPECTED_COVERAGE` entries is the projection of a new
entry; and the change needed no analyser that runs factory code — `ast` only, as before, and no
other test file changed. `Datastore` rises from 235 to 237 (two new `TestColumnExistence` tests;
G4/G5 added no new test methods, only widened what the four existing `TestFactorySelectColumns`
tests see).

## What shipped

**G4 — every function, not a hard-coded list of names.**
- `iter_functions(tree)` replaces the hand-written class/method walk plus the separate
  module-level `read_batch` loop: it yields `(class name, FunctionDef)` for every method of every
  class, and `("<module-level>", FunctionDef)` for every function at module scope — one generator,
  reused by both `analyse_factories` (G1/G4) and `column_existence_problems` (G2/G6).
- `analyse_factories()` now dispatches every function `iter_functions` yields: `read_batch` through
  `analyse_read_batch` (as before — its nested row helper needs inlining first), and every other
  function — `build`, `_build_impl_serial`, `_build_impl_model`, `validate_on_startup`, and
  anything not yet imagined — through `inline_nested_row_helper` followed by `analyse_build`
  directly. `_ROW_HELPERS` is gone: it gated nothing else in the module and had one remaining use,
  the hard-coded pair the review's finding 1 named.
- `inline_nested_row_helper` and `analyse_build` needed **no change** for this: they already
  handle a function with no nested row helper and a function with no row reads by producing no
  findings, which is exactly the "not a hole" case the prompt asks not to manufacture a finding
  for. Driving every function through the existing logic was verified directly, with
  `prompts/datastore-integrity/orchestrator/review_04_measurements.py` §A, which does the same
  dispatch against the **unmodified** prompt-04 module and reports the same seven newly-visited
  `validate_on_startup` bindings, all decidable and clean, that this commit's `analyse_factories`
  now reports.

**G5 — the function joins the coverage identity.**
- `Finding` gained an `fn` attribute (the function name), set from `analyse_build`'s own `fn`
  parameter in `bind_row`, and from `analyse_read_batch`'s manufactured blind Finding.
- `Finding.key` (`"module:class"`) is **unchanged** — `KNOWN_UNFIXED` and `KNOWN_UNFIXED_COLUMNS`
  still use it, unmodified, exactly as the prompt requires.
- A new `Finding.coverage_key` (`"module:class.function.variable"`) is what `EXPECTED_COVERAGE`
  and `KNOWN_BLIND` are now keyed by, and what
  `test_analyser_coverage_is_undiminished`/`test_blind_spots_are_the_declared_ones` compare
  against. `EXPECTED_COVERAGE` grew from 30 to 43: 04's 10 `_build_impl_*`/`read_batch` keys become
  16 again (the function distinguishes `_build_impl_serial` from `_build_impl_model`, matching
  `probe_02.out`), plus the 7 `validate_on_startup` bindings G4 exposes. `KNOWN_BLIND` stays at 2
  entries, both re-keyed with their function
  (`BackgroundModel.py:sqla_BackgroundModelFactory.build.row_data` and
  `OneLoopIntegral.py:<module-level>.read_batch.read_batch`).
- **The projection.** Verified by computation (below), not by inspection: dropping the function
  component of every one of the 43 `EXPECTED_COVERAGE` entries reproduces a superset of 04's 30
  entries, with nothing missing.

**G6 — G2 declares everything it does not check.**
- `column_existence_problems()` now walks `iter_functions(tree)` (module-level functions
  included) instead of only class methods, for both the `.c.` walk and the insert-payload walk.
- A `tables["X"]` / `inserters["X"]` naming no schema table is now a **problem**, added to the same
  `problems` set a missing column is, wherever the literal subscript appears — not only where it
  is later dereferenced with `.c.`. There are none in this tree today (mutation (iv) below shows
  the code path convicts one when it exists).
- Every `<base>.c.<name>` whose `base` does not resolve to a table (`t is None`) is recorded as
  `(function site, unparsed base)` and pinned against a new declared dict, `KNOWN_UNRESOLVED_TABLES`,
  with a reason per entry, instead of being silently `continue`d as prompt 04 left it.
- Every `inserter(conn, x)` / `value_inserter(conn, x)` call whose `x` is not an `ast.Dict` literal
  is recorded as `(module, class, function, inserter-name)` and pinned against a new declared
  dict, `KNOWN_UNCHECKED_INSERTS`, with a reason per entry, instead of being silently skipped.
- Module docstring point 6 rewritten to say what G2 now checks (module-level functions, an unknown
  `tables["X"]`/`inserters["X"]`) and what it declares (`KNOWN_UNRESOLVED_TABLES`,
  `KNOWN_UNCHECKED_INSERTS`, alongside the existing `KNOWN_BLIND_COLUMNS`).
- **No resolution rule was used.** Every one of the 12 unresolved-table pairs and 13 unchecked
  insert sites is declared, none resolved. See "Resolution rules considered" below for why.

## Deviations from the prompt

No `STRUCTURALLY REQUIRED` or `UNINTENDED DRIFT` deviations.

1. **IMPLEMENTATION CHOICE — no resolution rule was applied in G6; every unresolved-table pair and
   unchecked insert is declared.** §2's G6 explicitly makes this a choice ("*Your choice:* you may
   resolve a case instead of declaring it, if the rule is sound and stated"). Two candidate rules
   were identified:
   - *Ten of the 13 insert sites* follow one shape: a local first bound to a dict literal
     (`insert_data = {...}`), then conditionally extended by exactly one line,
     `insert_data["serial"] = payload["serial"]` — a rule that tracks that one augmentation could
     check the literal's keys.
   - *The `table` base in `OneLoopIntegral.py`'s module-level `read_batch`* (13 of the 44 unresolved
     `.c.` references) is the function's own parameter, and `OneLoopIntegral.py` registers exactly
     one factory — a rule resolving a bare `table` parameter in a module-level function of a
     single-factory module to that module's table could check those 13.

   Both were rejected in favour of declaring: the prompt's own bar for using one is that it "can
   find a defect" (verified by a rename-and-fail mutation) and is "sound" — but neither rule adds
   anything a person reading `KNOWN_UNRESOLVED_TABLES`/`KNOWN_UNCHECKED_INSERTS` does not already
   have from the declared reason, and each adds resolver complexity (tracking a second assignment
   statement's effect on an already-bound name; tracking module-registration cardinality) this
   prompt does not need to carry to satisfy G6, which only requires that nothing be silently
   skipped. Declaring all 25 sites is uniformly simpler, equally sound (nothing is silently
   skipped), and equally verified by mutation (v)/(vi) below, which show the declared dictionaries
   catch a regression back to prompt 04's silent-skip behaviour.

## Verification performed

**The projection, by computation.** Every one of prompt 04's 30 `EXPECTED_COVERAGE` entries,
dropping the function component of every one of this commit's 43, is present:

```
>>> def project(k):
...     module_cls, rest = k.split(":", 1)
...     cls, fn, var = rest.split(".")
...     return f"{module_cls}:{cls}.{var}"
>>> projection = {project(k) for k in EXPECTED_COVERAGE}   # the 43 of this commit
>>> old_30 - projection                                     # 04's 30 entries
set()
```

No entry is missing. `projection - old_30` has 7 members, all `validate_on_startup` bindings (the
new coverage G4 exposes), consistent with 30 + 7 (`validate_on_startup`) + 6 (the six
`_build_impl_*`/`read_batch` keys that regain a sibling once the function is back in the identity)
= 43.

**The figures, against `orchestrator/review_04_measurements.py` (run unmodified, as instructed;
neither it nor its `.out` was edited):**

| Figure | Expected (`.out`) | Measured against this commit | Match |
|---|---|---|---|
| Decidable, function-qualified coverage entries | 43 | `len(EXPECTED_COVERAGE)` = 43 | yes |
| Blind, function-qualified entries | 2 | `len(KNOWN_BLIND)` = 2 | yes |
| Unresolved (function, base) pairs in G2 | 12 (44 references) | `len(KNOWN_UNRESOLVED_TABLES)` = 12, summing to 44 references | yes |
| Non-literal insert calls | 13 | `len(KNOWN_UNCHECKED_INSERTS)` = 13 | yes |

`KNOWN_UNRESOLVED_TABLES`'s 12 pairs: nine keyed on `subquery` (a local bound to
`<query>.subquery()`, which `_table_of` does not follow), two on `query` (the select statement
itself, read via `.c.` while a tag alias is joined in — `OneLoopIntegral.py`'s module-level
`read_batch` and `QuadSourceIntegral.py`'s `read_batch`), and one on `table` (the bare parameter of
`OneLoopIntegral.py`'s module-level `read_batch`, which has no `cls_to_table` entry because it is
not a class method) — matching `review_04_measurements.py` §C exactly (9 `subquery` + 2 `query` +
1 `table` = 12; 5+1+1+4+8+1+13+1+1+1+4+4 = 44 references).

`KNOWN_UNCHECKED_INSERTS`'s 13 sites, measured with a one-off AST walk over every function
`iter_functions` yields, looking for `<x>inserter(conn, y)` where `y` is not an `ast.Dict`:
3 whose payload is a bare parameter or a value built by another function
(`BackgroundModel.py.store`'s `inserter(conn, payload)` and
`value_inserter(conn, value_data)` where `value_data = value_payload(...)`, and
`wavenumber.py.store`'s `inserter(conn, payload)`), and 10 whose payload is a dict literal locally
bound and then conditionally extended by one `x["serial"] = payload["serial"]` line
(`GkSourcePolicy.py`, `LambdaCDM.py`, `QCD_Cosmology.py`, `QuadSourcePolicy.py`,
`integration_metadata.py`, `redshift.py`, `store_tag.py`, `tolerance.py`, `version.py`,
`wavenumber.py`'s `sqla_wavenumber_factory.build`) — see deviation 1 for why none of the ten is
resolved instead.

**`KNOWN_UNFIXED` / `KNOWN_UNFIXED_COLUMNS` unchanged**, checked by direct comparison against 04's
committed values (`6a97d8a`): identical keying and identical contents, both dicts and both tests
(`test_known_unfixed_entries_are_still_present`, `test_known_unfixed_columns_are_still_present`)
still pass. No new defect was found.

**All nine tests pass** (the four `TestFactorySelectColumns` tests, widened in reach but unchanged
in name and assertion, plus the five `TestColumnExistence` tests — three from prompt 04, two new
here). `black --check Datastore/tests/test_factory_select_columns.py` is clean.

`git status --short` and `git diff --cached` were clean before starting and are clean after
committing (checked).

## The deliberate-breakage record

Each diff below is exactly as applied — `git diff` (unstaged, against the index this commit's
change was staged into) taken **while the mutation was applied**, pasted unaltered — with the
resulting test outcome, then reverted (`git checkout --`) before the next. All six were re-checked
with `git apply --check` from files extracted from this committed log (see below), and each
applies, runs, and reverts (`git apply -R`) exactly as recorded here.

**(i) make the analyser skip functions named `_build_impl_model` and nothing else** (finding 2's
case):

```diff
diff --git a/Datastore/tests/test_factory_select_columns.py b/Datastore/tests/test_factory_select_columns.py
index cd45683..cac860f 100644
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -514,6 +514,8 @@ def analyse_factories() -> List[Finding]:
     for path in sorted(FACTORY_DIR.glob("*.py")):
         tree = ast.parse(path.read_text(), filename=str(path))
         for cls, fn in iter_functions(tree):
+            if fn.name == "_build_impl_model":
+                continue
             if fn.name == "read_batch":
                 findings.extend(analyse_read_batch(fn, path.name, cls))
             else:
```

Result: `FAILED (failures=1)` — `test_analyser_coverage_is_undiminished` only, naming the six
`_build_impl_model.row_data` keys as missing from `covered`. Reverted; tree matched the
pre-mutation state.

**(ii) make it skip `validate_on_startup`:**

```diff
diff --git a/Datastore/tests/test_factory_select_columns.py b/Datastore/tests/test_factory_select_columns.py
index cd45683..a7943d1 100644
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -514,6 +514,8 @@ def analyse_factories() -> List[Finding]:
     for path in sorted(FACTORY_DIR.glob("*.py")):
         tree = ast.parse(path.read_text(), filename=str(path))
         for cls, fn in iter_functions(tree):
+            if fn.name == "validate_on_startup":
+                continue
             if fn.name == "read_batch":
                 findings.extend(analyse_read_batch(fn, path.name, cls))
             else:
```

Result: `FAILED (failures=1)` — `test_analyser_coverage_is_undiminished` only, naming the seven
`validate_on_startup` keys as missing from `covered`. Reverted.

**(iii) rename a column read in one `validate_on_startup` to one its query does not select**
(`sqla_BackgroundModelFactory.validate_on_startup`, an f-string read of `model.friction_F_gauss_order`):

```diff
diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
index 07f82de..4d76e49 100644
--- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
+++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
@@ -910,7 +910,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
             msgs.append(
                 f'       -- "{model.label}" (store_id={model.serial}, '
                 f"N_tau={model.tau_gauss_order}, N_cs_tau={model.cs_tau_gauss_order}, "
-                f"N_F={model.friction_F_gauss_order})"
+                f"N_F={model.friction_F_gauss_order_typo})"
             )
             rows = conn.execute(
                 sqla.select(sqla.func.count(value_table.c.serial)).filter(
```

Result: `FAILED (failures=1)` — `test_build_reads_only_columns_its_select_requests` (the guard
proper), naming a new shortfall
`{'BackgroundModel.py:sqla_BackgroundModelFactory': ['friction_F_gauss_order_typo']}` not present
in `KNOWN_UNFIXED`. Reverted.

**(iv) change one factory's `tables["X"]` to a table name the schema does not have**
(`sqla_BackgroundModelFactory.build`'s `tables["IntegrationSolver"]`):

```diff
diff --git a/Datastore/SQL/ObjectFactories/BackgroundModel.py b/Datastore/SQL/ObjectFactories/BackgroundModel.py
index 07f82de..8d3d429 100644
--- a/Datastore/SQL/ObjectFactories/BackgroundModel.py
+++ b/Datastore/SQL/ObjectFactories/BackgroundModel.py
@@ -244,7 +244,7 @@ class sqla_BackgroundModelFactory(SQLAFactoryBase):
         z_sample: redshift_array = payload["z_sample"]
         z_init: redshift_array = payload.get("z_init", None)
 
-        solver_table = tables["IntegrationSolver"]
+        solver_table = tables["IntegrationSolver_typo"]
         tag_table = tables["BackgroundModel_tags"]
         redshift_table = tables["redshift"]
 
```

Result: `FAILED (failures=1)` — `test_column_references_resolve_against_schema` (G2 only), naming
`('BackgroundModel.py', 'sqla_BackgroundModelFactory', 'build', 'tables[X]', 'IntegrationSolver_typo')`.
Reverted.

**(v) make G2 skip an unresolved base silently again** (04's behaviour):

```diff
diff --git a/Datastore/tests/test_factory_select_columns.py b/Datastore/tests/test_factory_select_columns.py
index cd45683..c55ad98 100644
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -1093,8 +1093,6 @@ def column_existence_problems() -> Tuple[
                 base, name, dynamic = ref
                 t = _table_of(base, env)
                 if t is None:
-                    key = (site, ast.unparse(base))
-                    unresolved_tables[key] = unresolved_tables.get(key, 0) + 1
                     continue
                 if t not in columns:
                     continue
```

Result: `FAILED (failures=1)` — `test_unresolved_table_sites_are_the_declared_ones`, naming all 12
declared pairs as missing from the (now empty) measured set. Reverted.

**(vi) make G2 stop walking module-level functions:**

```diff
diff --git a/Datastore/tests/test_factory_select_columns.py b/Datastore/tests/test_factory_select_columns.py
index cd45683..6bb8241 100644
--- a/Datastore/tests/test_factory_select_columns.py
+++ b/Datastore/tests/test_factory_select_columns.py
@@ -1048,6 +1048,8 @@ def column_existence_problems() -> Tuple[
     for path in sorted(FACTORY_DIR.glob("*.py")):
         tree = ast.parse(path.read_text(), filename=str(path))
         for cls, fn in iter_functions(tree):
+            if cls == "<module-level>":
+                continue
             own = cls_to_table.get(cls)
             env: Dict[str, str] = {}
             if own is not None:
```

Result: `FAILED (failures=1)` — `test_unresolved_table_sites_are_the_declared_ones`, naming the two
`OneLoopIntegral.py:<module-level>.read_batch` pairs (`query`, `table`) as missing (they can only
be found by walking a module-level function). Reverted.

Each mutation was applied alone, against the tree with G4–G6 already staged (`git add`), so `git
diff` (unstaged) shows exactly that one mutation; `git diff --cached` (the staged G4–G6 change)
was unaffected throughout, and `git status --short` returned to `M  Datastore/tests/test_factory_select_columns.py`
after every `git checkout --`.

**Replay from the committed log.** After committing, each of the six diffs above was extracted
verbatim from this file (the committed log) into its own file under a scratch directory outside
the repository, and for each: `git apply --check`, `git apply`, the module run, and `git apply -R`.
All six applied cleanly on the first try (no hand-corrected hunk header was needed, unlike prompt
04's mutation (v) — see its review finding 4), reproduced the failure recorded above exactly, and
reverted the tree to clean (`git status --short` empty) afterwards.

## Observations not acted on

1. **The ten dict-literal-plus-one-augmentation insert sites, and the single-factory-module
   `table` parameter, are resolvable** — see deviation 1. Not fixed or resolved here, per this
   prompt's own scope (§6: "it fixes no factory") and its explicit choice not to require
   resolution.
2. **Review findings 4 and 5 stand, as the board already records.** This prompt's own mutation
   diffs are taken as `git diff` output while each mutation was applied (never hand-written), which
   is what finding 4 asked future logs to do; and `KNOWN_UNFIXED` / `KNOWN_UNFIXED_COLUMNS` remain
   two separate structures (finding 5), unchanged by this prompt, which fixes no factory and empties
   neither — that is prompt 05's.

## State handed to the next prompt

- **For prompt 05.** `KNOWN_UNFIXED` (2 entries) and `KNOWN_UNFIXED_COLUMNS` (3 entries) are
  unchanged in keying and contents from prompt 04 — still the list to empty. The coverage pins
  (`EXPECTED_COVERAGE`, `KNOWN_BLIND`) are now function-qualified, so 05's fixes (which touch
  `_build_impl_serial`, not `_build_impl_model`, for the `WKB_criterio` entries; and
  `_build_impl_serial`/`_build_impl_model` for the two column-existence entries) will not
  accidentally satisfy the wrong helper's pin — each helper now has its own key.
- **Names.** `iter_functions` (G4, shared by G1's dispatch and G2's walk); `Finding.fn`,
  `Finding.coverage_key` (G5); `KNOWN_UNRESOLVED_TABLES`, `UnresolvedTableSite`,
  `KNOWN_UNCHECKED_INSERTS`, `InsertSite` (G6). `_ROW_HELPERS` is gone.
- **Baselines** after this commit: AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39,
  Datastore **237** (235 + 2), LiouvilleGreen 148 (1 skipped), RunRegistry 199.
