# Log 06 — Fix the three residuals in this campaign's own code, and leave it ready to close

**Prompt:** [`prompts/store-retirement/06-fix-the-residuals-and-close.md`](../06-fix-the-residuals-and-close.md)
**Commit:** *(this commit)* — "Fix the three residuals of the store retirement campaign"
**Model:** Claude Sonnet 5
**Date:** 2026-09-26
**Result:** COMPLETE. No §5 stop condition was met. `RunRegistry/tests/test_store_amend.py` is
not touched (`git diff` of it is empty). Nothing under `var/` was opened, read-only included:
every store in the new module is built in a temporary directory, every `amend_sidecar` call
passes `runs_root`, and every command passes `--runs-root`.

## What shipped

**R13, the charter (`RunRegistry/__init__.py`).** The module docstring's first paragraph, from
"It is a convention" to the end of the paragraph, now reads:

Before:

```
It is a convention with a little code behind it, not a framework: it schedules nothing, supervises
nothing, locks nothing and deletes nothing. It also creates, adopts, copies and moves the stores it
manages, with their `<stem>.manifest.json` sidecars (`RunRegistry.stores`), because the user
decided that managing stores is part of managing the registry (`prompts/datastore-portability`
README §6.5); it still deletes nothing.
```

After:

```
It is a convention with a little code behind it, not a framework: it schedules nothing, supervises
nothing and locks nothing. It also creates, adopts, copies and moves the stores it manages, with
their `<stem>.manifest.json` sidecars (`RunRegistry.stores`), because the user decided that
managing stores is part of managing the registry (`prompts/datastore-portability` README §6.5). It
deletes nothing but a store's own files, and those only through `store retire`, which a person
runs and which leaves the store's sidecar behind as its record (`prompts/store-retirement` README
§6.2, D0). It never deletes a run directory, a sidecar or any other record.
```

This is the prompt's §2.1 text verbatim, re-wrapped to the file's own line width (the surrounding
paragraphs run 90–103 characters; the new paragraph's lines run 89–99). Nothing else in the
docstring changed; `list_runs`'s "Nothing here writes or deletes anything" stays, since it is true
of that function.

```
$ grep -n "deletes nothing" RunRegistry/__init__.py
$ echo $?
1
```

(no output; exit 1, meaning no match)

**R14, the guard (`RunRegistry/stores.py`, inside `amend_sidecar`).** A new private helper,
`_canonical_json(value)`, placed directly after `_amend_slot` and before `amend_sidecar`:

```python
def _canonical_json(value) -> str:
    """``value``'s canonical JSON text: `json.dumps(value, sort_keys=True)`, exactly as
    `write_json_atomic` writes a sidecar. Two values compare equal here iff they would write the
    same bytes for that field: `True`, `1` and `1.0` differ, because their texts do, and an object
    with its keys in another order does not, because its text is the same either way."""
    return json.dumps(value, sort_keys=True)
```

The comparison itself, before and after:

```diff
-        if field in fields and fields[field] == normalised:
+        if field in fields and _canonical_json(fields[field]) == _canonical_json(
+            normalised
+        ):
```

`sort_keys=True` is what `write_json_atomic` already uses to write a sidecar, so two values with
the same canonical text write the same bytes for that field — refusing exactly those is "nothing
would change", which is what the refusal already says. Nothing else about the refusal changed: it
is still ninth, after the JSON-serialisability check; its message still contains the fragment
`identical, after a JSON round trip`, unmodified; and what is written does not change —
`new_value` is still `normalised`, and every other line of `amend_sidecar` is untouched. The
docstring's item 9 was extended to say how the comparison is made (canonical JSON text, so
`true`, `1` and `1.0` are three different values and only an object's key order is not).

**The `NaN` behaviour change (§2.2).** `write_json_atomic` (`json.dump(..., sort_keys=True)`, the
stdlib default `allow_nan=True`) emits a bare `NaN` for a field holding `float("nan")`, and
`json.loads` reads it back as `float("nan")`. Under the old `==` comparison, `float("nan") ==
float("nan")` is `False` in Python, so amending a `NaN` field to `NaN` was *accepted* — written,
even though it changed nothing. Under `_canonical_json`, both sides' text is the literal `NaN`,
so the two texts are equal and the amendment is now *refused*. This was verified directly:

```python
>>> import json
>>> from RunRegistry.stores import _canonical_json
>>> _canonical_json(float("nan")) == _canonical_json(float("nan"))
True
>>> float("nan") == float("nan")
False
```

No test is required for it (prompt §2.2), and none is added.

**R15, the test (`RunRegistry/tests/test_store_amend_residuals.py`, new).** A module beside
`test_store_amend.py`, importing `AmendTestCase` and `REASON` from it rather than copying them.
9 tests, in about 6 s, run alongside the existing 21:

| Class | Prompt §3 item | Test(s) |
|---|---|---|
| `TestTypeStrictComparison` | **1.** Accepted, with the type kept: `1`↔`true`, `1`↔`1.0`, `true`↔`1`, `{"k":1}`↔`{"k":true}`, `[1]`↔`[true]`. | `test_1_to_true`, `test_1_to_1_0`, `test_true_to_1`, `test_dict_value_1_to_true`, `test_list_value_1_to_true` |
| `TestStillRefusedWithTheTreeUnchanged` | **2.** Still refused, tree unchanged: `1`→`1`; `{"a":1,"b":2}`→`{"b":2,"a":1}`. | `test_1_to_1`, `test_an_object_with_its_keys_reordered` |
| `TestCommandLineTypeChange` | **3.** The command line: `--json true` on a field holding `1` exits 0 and prints `>> amended:`; the same command run again exits 1, naming the identical-value refusal. | `test_true_over_1_then_refused_the_second_time` |
| `TestMarkerShapedValue` | **4–7.** One field, `m`, added, replaced, removed, each value shaped like the marker itself, then read back. | `test_add_replace_remove_and_read_back` (subTests `add`, `replace`, `remove`, `read back`) |

For each of items 1's five cases: `assertOnlySidecarChanged` holds; the entry's `before`/`after`
markers are asserted equal to the old/new value, **and** `assertIs(type(entry[...]["value"]), …)`
against `bool`, `float` or `int` as appropriate — `assertEqual` alone cannot tell `1` from `True`.
The sidecar's raw text on disk is also checked by regex (`stores.sidecar_path(primary).read_text()`)
for `true`/`1.0` where the new value now sits, and, for the `true`→`1` case, that the text has
neither `true` nor `1.0` there. Item 2's two cases use `refused_amend`, asserting the fragment
`identical, after a JSON round trip` and that the whole temporary tree is otherwise unchanged.
Item 3 amends via the command line (`store amend … --field n --json true --reason …
--runs-root …`), asserts exit 0 and `>> amended:` in stdout, then runs the identical command
again and asserts exit 1 with the refusal fragment in stderr.

Items 4–7 (`TestMarkerShapedValue.test_add_replace_remove_and_read_back`) run as four `subTest`
blocks in one method, add / replace / remove / read back, matching the prompt's own numbering:
- **add:** `m` absent → `{"present": False}`; `before` is `None` (`m` was absent), and the sidecar's
  `history[-1]["before"]` is exactly `{"present": False}`, `["after"]` exactly
  `{"present": True, "value": {"present": False}}`;
- **replace:** → `{"present": True, "value": 1}`; `before` is exactly `{"present": True, "value":
  {"present": False}}` and `after` exactly `{"present": True, "value": {"present": True, "value":
  1}}`;
- **remove:** `before` is exactly `{"present": True, "value": {"present": True, "value": 1}}` and
  `after` exactly `{"present": False}`;
- **read back:** the sidecar reads `ok`, with `stores._history_problems` of its history empty.
  Beyond the prompt's own wording, each `amend` entry's `before` is also asserted equal to the
  *previous* entry's `after` (a consistency check the per-step assertions above do not by
  themselves make in one place) — this is the check that independently catches deliberate
  breakage (iii) at the "read back" step, not only at the "replace"/"remove" steps. Then a small
  walk unwraps each of the four markers (`before` of the first entry, `after` of each of the
  three) one level, reconstructing the sequence absent → `{"present": false}` → `{"present":
  true, "value": 1}` → absent, and nothing else. `store show` on the store exits 0, with
  `problems: none` in its output.

Every marker above is asserted by `assertEqual` of the **whole** marker (`entry["before"]`,
`entry["after"]`, never just `entry["before"]["value"]`), so a marker which lost its wrapper, or
gained a second, fails.

## Deviations from the prompt

1. **A consistency check ("each entry's `before` equals the previous entry's `after`") was added
   to the "read back" step, beyond the prompt's own wording of item 7.** `IMPLEMENTATION CHOICE`.
   While drafting the deliberate-breakage record for mutation (iii) it became clear that the
   prompt's own reconstruction walk — which samples only the *first* entry's `before` and every
   entry's `after` — never reads the `before` of the second or third `amend` entry, and those are
   exactly the two markers mutation (iii) corrupts (the mutated `_amend_slot` only ever supplies
   the `before` side; `after` is always built directly in `amend_sidecar` and is untouched by it).
   Without this addition, the "read back" subTest alone would have passed under mutation (iii),
   even though the "replace" and "remove" subTests, above it, already fail on their own direct
   marker assertions (§ below: `failures=4` becomes `failures=5` with this addition, all in the
   same test method, as three separate subTest failures). The addition strengthens exactly the
   check the prompt's item 7 describes ("a small walk... reconstructs the sequence... and nothing
   else") without changing what it asserts about the *value* sequence.

None is `UNINTENDED DRIFT`; none is `STRUCTURALLY REQUIRED` beyond what the prompt already asked
for.

## Verification performed

- **Baselines, measured in this checkout before touching anything, at `5e9b16f`** (`HEAD`, as the
  dispatch states): AdaptiveLevin 32, ComputeTargets 552, CosmologyModels 39, Datastore 206,
  LiouvilleGreen 148 (1 skipped), RunRegistry 190. All matching the stated pre-dispatch baselines.
- **The new module**, run alone and beside `test_store_amend.py`: `Ran 30 tests ... OK` (21 + 9),
  in about 6 s, run twice with identical results.
- **All six suites after the change**, run from the repository root with `PYTHONPATH=.
  ./venv/bin/python -m unittest discover -s <package>/tests -t .`:

  | Suite | Before (pre-dispatch) | After |
  |---|---|---|
  | `AdaptiveLevin` | 32 OK | 32 OK |
  | `ComputeTargets` | 552 OK | 552 OK (the wall-clock flake did not occur) |
  | `CosmologyModels` | 39 OK | 39 OK |
  | `Datastore` | 206 OK | 206 OK |
  | `LiouvilleGreen` | 148 OK, skipped=1 | 148 OK, skipped=1 |
  | `RunRegistry` | 190 OK | 199 OK (190 + the 9 tests this prompt adds), run twice |

- **`black --check`** is clean on every file touched: `RunRegistry/stores.py`,
  `RunRegistry/__init__.py`, `RunRegistry/tests/test_store_amend_residuals.py`.
- **`test_store_amend.py` is untouched.** `git diff --stat -- RunRegistry/tests/test_store_amend.py`
  is empty, both before and after every mutation was reverted, and
  `test_a_value_identical_after_a_json_round_trip` (the one existing test that exercises the
  identical-value refusal) still passes unmodified: a tuple `(1, 2, {"a": None})`, which JSON
  round-trips to the same list either way, is still refused under the canonical-text comparison
  exactly as it was under `==`.
- **Scope.** `git diff --cached --stat` (this commit) touches exactly `RunRegistry/stores.py`
  (inside `_amend_slot`'s neighbourhood and `amend_sidecar` only — no hunk falls inside
  `retire_store`, `copy_store`, `move_store`, `_prepare`, `fingerprint_store` or the reader),
  `RunRegistry/__init__.py` (the module docstring's first paragraph only), the new test module,
  this log, the board and the index. `Datastore/`, `tools/`, `CLAUDE.md`, `RunRegistry/__main__.py`
  and every existing test file are untouched. The untracked `docs/datastore-integrity-audit*` and
  `prompts/datastore-integrity/` were not staged, read for content, or changed, and no `orch_*`
  file was touched.
- **Nothing under `var/` was opened.** Every store in the new module is built in a
  `tempfile.TemporaryDirectory` (`StoreTestCase.setUp`, via `AmendTestCase`). Every `amend_sidecar`
  call passes `runs_root`, and every `store amend`/`store show` command passes `--runs-root`.
- **Import cost.** `test_store_amend_residuals.py` imports only `RunRegistry.stores` and
  `RunRegistry.tests.test_store_amend`; it makes no new subprocess check of its own, relying on
  `test_store_amend.py`'s existing `test_import_loads_neither_ray_nor_sqlalchemy`, which still
  passes unmodified.

## The deliberate-breakage record

Each diff below is exactly as applied: `RunRegistry/stores.py` (with the fix already staged) was
edited, `git diff -- RunRegistry/stores.py` was taken against the index, the affected test modules
were run from the repository root (not an empty working directory — `PYTHONPATH=.
./venv/bin/python -m unittest RunRegistry.tests.test_store_amend_residuals
RunRegistry.tests.test_store_amend -v`, cwd the repository root), and the file was restored with
`git checkout -- RunRegistry/stores.py` before the next mutation. Unmutated, that run is `Ran 30
tests ... OK`. After the commit, each diff below was extracted from this file and checked with
`git apply --check` against `HEAD`.

**(i) Restore the `==` comparison.** `FAILED (failures=1, errors=5)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 18d6981..57aed2d 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2118,9 +2118,7 @@ def amend_sidecar(
             raise refuse(
                 f"the given value is not JSON-serialisable ({type(e).__name__}: {e})"
             )
-        if field in fields and _canonical_json(fields[field]) == _canonical_json(
-            normalised
-        ):
+        if field in fields and fields[field] == normalised:
             raise refuse(
                 f"the given value is identical, after a JSON round trip, to {field!r}'s current "
                 f"value; nothing would change"
```

**Outcome.** Every type-changing amendment in test 1 is refused as "identical" under `==` (`True
== 1 == 1.0`), so it raises where the test expects success:
`TestTypeStrictComparison.test_1_to_true`, `.test_1_to_1_0`, `.test_true_to_1`,
`.test_dict_value_1_to_true` and `.test_list_value_1_to_true` all `ERROR`
(`RuntimeError` propagating out of `self.amend`, uncaught). Test 3
(`TestCommandLineTypeChange.test_true_over_1_then_refused_the_second_time`) `FAIL`s on its first
assertion, `self.assertEqual(report["code"], 0)` — the command that should succeed now exits 1.
Test 2 (`TestStillRefusedWithTheTreeUnchanged`) is unaffected: both its cases are refused under
either comparison. `test_store_amend.py` is entirely unaffected (`Ran 21 tests ... OK`
separately): its own `test_a_value_identical_after_a_json_round_trip` amends values whose JSON
texts agree regardless of `sort_keys`, so it does not distinguish the two comparisons.

**(ii) Compare `json.dumps(…)` without `sort_keys`.** `FAILED (failures=1)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 18d6981..0b17a5a 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2015,7 +2015,7 @@ def _canonical_json(value) -> str:
     `write_json_atomic` writes a sidecar. Two values compare equal here iff they would write the
     same bytes for that field: `True`, `1` and `1.0` differ, because their texts do, and an object
     with its keys in another order does not, because its text is the same either way."""
-    return json.dumps(value, sort_keys=True)
+    return json.dumps(value)
 
 
 def amend_sidecar(
```

**Outcome.** Only `TestStillRefusedWithTheTreeUnchanged.test_an_object_with_its_keys_reordered`
fails (`RuntimeError not raised`): `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` now serialise to
different texts (insertion order, not sorted), so the refusal does not fire and the amendment
succeeds where the test expects a refusal. Every other test, in both modules, is unaffected: no
other case in this campaign amends a multi-key object into a reordering of itself, so no other
comparison's outcome changes when key order stops being normalised.

**(iii) Make `_amend_slot` return the bare value, not the wrapper, when the field is present.**
`FAILED (failures=5, errors=9)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 18d6981..793be31 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2006,7 +2006,7 @@ def _amend_slot(fields: dict, name: str) -> dict:
     Deep-copied, so later mutation of ``fields`` cannot reach back into a written history entry.
     """
     if name in fields:
-        return {"present": True, "value": _copy.deepcopy(fields[name])}
+        return _copy.deepcopy(fields[name])
     return {"present": False}
 
 
```

**Outcome.** `_amend_slot` builds only the **`before`** side of a marker (`after` is built
directly in `amend_sidecar`, untouched by this mutation), and only when the field is *already
present*. Two different ways this shows up:

- **Where the bare value is not itself marker-shaped** (a plain string, number or a dict without a
  boolean `present` key), `entry["before"]` fails `_amend_slot_problems` and `_update_sidecar`'s
  own `_check_before_writing` refuses the write with a `RuntimeError` *before* `amend_sidecar`
  returns — a validator refusal, not one of `amend_sidecar`'s own `refuse(...)` calls, and
  **uncaught** wherever the caller does not expect one:
  - `TestTypeStrictComparison`'s five tests each seed the field once (absent → present, the
    unaffected branch) and then amend it a second time (present → a new value, the affected
    branch): all five `ERROR` with this `RuntimeError`.
  - `test_store_amend.py`'s `TestAmend.test_replace_an_unknown_field`,
    `TestRemove.test_remove_an_unknown_field`,
    `TestLiveA3Shape.test_amend_backup_as_prompt_05_will` and
    `TestTwoAmendments.test_two_amendments_of_one_field_read_back_as_a_sequence` each amend or
    remove a field that already holds a non-marker-shaped value (a dict, a string, or a second
    amendment of a string): all four `ERROR` the same way.
  - `TestCommandLineTypeChange.test_true_over_1_then_refused_the_second_time` hits the same
    `RuntimeError`, but inside a child process, where `_amend` catches `RuntimeError` and prints a
    refusal, exit 1: the test's first assertion, that the command exits 0, `FAIL`s instead.
  - `test_store_amend.py`'s `TestCommandLine.test_json_remove_bad_json_and_exit_codes` hits the
    same thing on its `--remove` step (removing `note`, which by then holds `{"a": 1}`, not
    marker-shaped): `FAIL`s on the same assertion.
- **Where the bare value happens to itself be marker-shaped** — exactly `TestMarkerShapedValue`'s
  own design, since `m`'s value at every step *is* `{"present": ...}` — the write does **not**
  raise: the flattened `before` is still a syntactically well-formed marker, just the wrong one.
  `test_add_replace_remove_and_read_back` `FAIL`s at three separate `subTest`s: `[replace]` and
  `[remove]` on their own direct `assertEqual` of the whole `before` marker, and `[read back]` on
  the added before-equals-previous-after consistency check (Deviation 1, above) — without that
  addition, `[read back]`'s own assertions (which sample only the *first* entry's `before` and
  every entry's `after`, none of which this mutation touches) would not have failed on their own.
  `[add]` is unaffected: `m` is absent at that point, the unaffected branch.

Total: `failures=5` (`TestCommandLineTypeChange` ×1, `test_json_remove_bad_json_and_exit_codes`
×1, and the three `subTest`s of `test_add_replace_remove_and_read_back`), `errors=9`
(`TestTypeStrictComparison` ×5, `TestAmend`/`TestRemove`/`TestLiveA3Shape`/`TestTwoAmendments` ×1
each). The tree and the working directory were clean afterwards; the file was restored and all 30
new-module-plus-existing tests, and the full `RunRegistry` suite (199 tests), pass again.

## Observations not acted on

1. **`RunRegistry/__init__.py`'s docstring format table and other prose were not re-checked for
   other stale "deletes nothing"-adjacent phrasing beyond the one paragraph R13 names.** Not
   acted on: the prompt's §2.1 names exactly one paragraph, "from 'It is a convention' to the end
   of the paragraph", and §2.3 says nothing else in the file changes. A single further read of the
   file (this dispatch) found no other such phrase; not an issue, since nothing was found.
2. **A general note on `==` elsewhere in `stores.py`.** The prompt's §2.2 asks that any other place
   comparing JSON values with `==` where the difference could matter be recorded, not changed. A
   read of every `==` in `stores.py` outside `amend_sidecar` found none that compares a field's
   *value* against another JSON-derived value in a way where `True`/`1`/`1.0` could be conflated —
   the other equality comparisons in the file are on strings (paths, operation names, `store_id`),
   booleans compared to booleans, or structural checks (`set(value)` against a fixed key set),
   none of which mixes the numeric/boolean types this issue was about. Nothing is opened.

No new issue is opened by this prompt.

## State handed to the next prompt

There is no prompt 07: this is the last prompt of the campaign (README §2, §6.4). The board's
header is left exactly as the prompt's §4.4 states — 6 of 6 prompts landed, ready for the
orchestrator's review and, after it, the user's closure — and is not written as closed here.
- **R13–R16 are done.** The three issues this campaign opened in its own code are moved to §4,
  each with a `Closed (2026-09-26) by prompt 06` paragraph; `docs/OPEN_ISSUES.md` §1.14's three
  rows are deleted, and the header count corrected 104 → 101.
- **The diffs above** were checked after the commit: each, extracted from this file, passes
  `git apply --check` against `HEAD`.
- **Issue opened:** none.
