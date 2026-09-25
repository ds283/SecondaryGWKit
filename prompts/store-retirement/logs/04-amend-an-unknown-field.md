# Log 04 — Amend one unknown field of a sidecar, recording what it said before

**Prompt:** [`prompts/store-retirement/04-amend-an-unknown-field.md`](../04-amend-an-unknown-field.md)
**Commit:** *(this commit)* — "Add amend_sidecar and store amend, for one unknown field"
**Model:** Claude Sonnet 5
**Date:** 2026-09-25
**Result:** COMPLETE. No §5 stop condition was met. Amending an unknown field never touches a
known field (`KNOWN_FIELDS`, `retired` included). No existing test changed. Nothing under `var/`
was opened, and `amend_sidecar` and `store amend` were never called against anything under `var/`
or a copy of it.

## What shipped

**`RunRegistry/stores.py`.**

- **R9, the format.**
  - `AMEND_OPERATION = "amend"` and `AMEND_KEYS = ("field", "reason", "before", "after")` are new
    constants. `_history_problems` allows `amend` after index 0, alongside `copy`, `move` and
    `retire`; the existing "a retirement may stand only as the last entry" rule already forbids
    anything, `amend` included, from following a `retire` entry, so no separate check was needed
    for that.
  - An `amend` entry's `from` and `to` are both required to be `null` (an amend has neither a
    source nor a destination); every other operation's `from`/`to` rule is unchanged.
  - An `amend` entry must carry `AMEND_KEYS` beside `HISTORY_KEYS`: `field` and `reason` each a
    non-empty string, and `before`/`after` each a well-formed marker (`_amend_slot_problems`).
    Every other operation's entry is a problem if it carries any of `AMEND_KEYS`.
  - `_FIELD_OWNERS` maps each of `KNOWN_FIELDS` to the operation(s) that own it, named in
    `amend_sidecar`'s known-field refusal.
- **R9, the markers.** `before` and `after` are `{"present": true, "value": <the field's value>}`
  or `{"present": false}` — a tagged wrapper one level above the value, never a sentinel string.
  Whatever a field's value is, even one shaped exactly like the marker itself (§ "The markers, as
  shipped" test), it sits inside `value`, one level below the wrapper's own `present` key, so it
  can never be mistaken for "absent" or "removed". `_amend_slot(fields, name)` builds one side of
  the pair from the sidecar's current fields; `amend_sidecar` builds the other from what it is
  about to write.
- **R9, the operation.** `amend_sidecar(primary, field, reason, *, value=..., remove=False,
  runs_root=None) -> dict`, in this order (each refusal a `RuntimeError` naming the sidecar and
  the reason, ending "Nothing was written"):
  1. a blank or missing `reason`;
  2. both `value` and `remove`, or neither;
  3. a tombstone, complete or not (`SidecarReading.retired`, `_tombstone_text`) — checked before
     the generic "not a problem-free registry sidecar" refusal, so the message says "tombstone";
  4. an absent, unreadable or legacy sidecar, or a registry sidecar with problems (`store create`
     / `store adopt` named, as `_prepare` names them for copy and move);
  5. a known field, `field in KNOWN_FIELDS` — the message names the owning operation(s)
     (`_FIELD_OWNERS`);
  6. a store that a `running` run names, alive or stale, by path or by the sidecar's *recorded*
     `store_id` field (`_running_runs_naming("amend", [primary], store_id, runs_root)`), which also
     refuses a `runs_root` that is not a directory;
  7. `remove` of a field that is absent;
  8. a `value` that is not JSON-serialisable;
  9. a `value` that is identical, after a JSON round trip, to the field's current value.

  The write is one call to `_update_sidecar` (its fifth use), replacing or deleting only `field`
  and appending one `amend` entry. Every other field, known or unknown, is value-identical
  afterwards (`TestAmend`, `TestRemove`). Returns `{"primary", "sidecar", "field", "before",
  "after", "entry", "fields"}`: `before`/`after` are the plain values (`None` when absent or
  removed), `entry` is the appended history entry, and `fields` is the sidecar's new fields.
  Amend never imports `ShardedPool` or `Datastore.store_inventory`, and never opens the store's
  files: only the sidecar is read or written.
- **The docstrings.**
  - The module docstring's operations paragraph now lists `amend_sidecar`, adds it to the
    running-run-refusal sentence, adds "and an amendment" to the in-place-updates list, and gains a
    new paragraph describing what `amend_sidecar` does and does not touch.
  - The format table's `history` row describes `amend`'s place, its null `from`/`to`, and its four
    extra keys and their marker shape (§ "as shipped" below).
  - `_update_sidecar`'s own docstring: "exactly four places" → "exactly five places", with
    "`amend_sidecar`'s one write" added to the list. This corrects the count the prompt itself
    said was already wrong (prompt header note, amended 2026-09-25 after prompt 03 landed): prompt
    04 was written saying amend would be the *fourth* use, before 03 added retirement's two writes,
    which made it five before 04 existed to add a further one. No other line in log 03 or the
    board needed correction; only this one docstring's own count.

**`RunRegistry/__main__.py`.**
- The module docstring adds `amend` to the command list and a sentence: "`amend` replaces or
  removes one **unknown** field of the sidecar, with a required `--reason`, and records the
  field's old value in an `amend` history entry; it is the one way to change an unknown field, and
  it never touches a known one. It refuses a store a `running` run names, alive or stale."
- `store amend PRIMARY --field NAME (--json VALUE | --remove) --reason TEXT [--runs-root DIR]`.
  `--field` and `--reason` are `required=True` (argparse), so a missing one exits 2, as `--purpose`
  and `--reason` already do elsewhere. `--json` and `--remove` are both optional at the argparse
  level; `amend_sidecar`'s own "both or neither" refusal is what actually enforces exactly one,
  which keeps that rule inside the operation, as the prompt's §2.1 states it, and gives it exit 1
  like every other refusal, not the exit-2 usage error a `required` mutually exclusive group would
  give. A `--json` string that does not parse is caught in `_amend` itself (`json.JSONDecodeError`)
  and printed as a `!! Cannot amend "…": --json '…' does not parse (...). Nothing was written`
  refusal, exit 1 — not an argparse usage error, since it is a malformed value, not a missing flag.
  `_amend` prints `store:`, `sidecar:`, `field:`, `before:`, `after:` and `history entry:` (the
  full JSON entry), then `>> amended: <sidecar>` and exits 0; a caught `RuntimeError` prints
  `!! <message>` to stderr and exits 1.

**`RunRegistry/tests/test_store_amend.py`** (new): 21 test methods, several running subtests, in
about 5 s. No Ray, nothing under `var/`; every store is the lightweight placeholder fixture
(`Datastore.tests.shard_store_fixtures.write_new_store`, through `StoreTestCase.a_registry_store`),
because `amend_sidecar` never opens a store's files. Every `amend_sidecar` call goes through
`AmendTestCase.amend`, which passes `runs_root`, except `test_a_runs_root_that_is_not_a_directory`,
which passes it directly; every `store amend` command passes `--runs-root`.

| Class | Prompt §3 item |
|---|---|
| `TestAmend` | **1.** Replace an unknown field: the new value in place, the last history entry's `field`/`reason`/`before`/`after`, every other field value-identical, and only the sidecar changed (`tree_state`, via `assertOnlySidecarChanged`). |
| `TestRemove` | **2.** Remove an unknown field: gone from the fields, the entry records `before` and the removal marker. |
| `TestLiveA3Shape` | **3.** Adopt the `a3_shaped` legacy sidecar, amend `backup` from `"retained": true` to a value recording the backup's retirement, then `store show`: the new value is the current `backup.retained` (`False`), and the old one (`True`) sits in the last history entry's `before`. |
| `TestRefusals` | **4.** Every refusal in §2.1: no/blank reason; both or neither of value/remove; every member of `KNOWN_FIELDS` (iterated), `retired` included, by value and by `--remove`; remove of an absent field; a value identical after a JSON round trip; a non-serialisable value; an absent/unreadable/legacy/problem sidecar; a complete and an incomplete tombstone; an alive and a stale running run; a run matching by `results_store_id` alone; a runs root that is not a directory. Each leaves the whole temporary tree unchanged (`refused_amend`) and ends "Nothing was written". |
| `TestHistoryRule` | **5.** An `amend` entry missing each of `AMEND_KEYS` in turn, with a blank `reason` or `field`, at index 0, after a `retire`, carrying a non-null `from` or `to`; a `copy` entry carrying `AMEND_KEYS`; well-formed `create, amend`, `create, amend, amend` and `create, amend, retire` sequences have no problems. A separate test malforms the `before`/`after` marker itself: not an object, present with no `value`, absent but carrying a `value`, and `present` not a boolean. |
| `TestCopyAndMove` | **6.** Copy and move after an amendment carry the amended field and the one `amend` entry, unchanged; the source is untouched by a copy. |
| `TestTwoAmendments` | **7.** Two amendments of one field: two entries, the second's `before` equal to the first's `after`, so the full sequence of values reads back from the history. |
| `TestCommandLine` | **8.** A missing `--field` and a missing `--reason` each exit 2 and write nothing; bad `--json` exits 1; neither/both of `--json`/`--remove` exits 1; a successful `--json` amendment and a `--remove` each exit 0 and print `field:`, `before:`, `after:` and the history entry; a known field exits 1. A subprocess check that `import RunRegistry` loads neither `ray` nor `sqlalchemy`. |

## The `amend` entry's shape, its markers, and the `from`/`to` rule, as shipped

```
{"operation": "amend", "from": null, "to": null, "when", "git_head", "git_dirty",
 "field": <the field's name, a non-empty string>,
 "reason": <the required text, a non-empty string>,
 "before": {"present": true, "value": <the field's old value>} | {"present": false},
 "after":  {"present": true, "value": <the field's new value>} | {"present": false}}
```

`from` and `to` are both `null` because an amend has neither a source nor a destination, unlike
`copy`/`move` (both paths) or `retire` (`from` the primary, `to` null). `_history_problems` flags
an `amend` entry that carries a non-null `from` or `to`, and flags any *other* operation's entry
that carries `field`, `reason`, `before` or `after` — the extra keys belong to `amend` alone
(prompt §3 test 5, "the extra keys on a copy entry are a problem").

The `before`/`after` markers are a tagged wrapper (`{"present": bool}`, with `"value"` present iff
`"present"` is `true`), never a sentinel string or a special JSON value. This is what the prompt's
"nothing is lost" requirement needs: a field's value can legitimately be any JSON value, including
one shaped exactly like `{"present": false}` or `{"present": true, "value": "…"}`, and a sentinel
at the same level as the value could be confused with it. The wrapper sits one level above the
value, so the value itself, whatever its shape, is read back unambiguously
(`TestAmend.test_replace_an_unknown_field` amends a nested value; the tombstone test's mutations
exercise the marker's own shape-validation directly). `_amend_slot_problems` is the validator for
one marker; `_registry_problems` → `_history_problems` runs it on every write, on both `before` and
`after`.

## Deviations from the prompt

1. **`amend_sidecar` takes a `runs_root` keyword parameter.** `STRUCTURALLY REQUIRED`. README §4
   lists the signature as `amend_sidecar(primary, field, reason, *, value=…, remove=False)`, with
   no `runs_root`. But §2.1's last refusal is "a store that a `running` run names, alive or
   stale", which needs a runs root to check against — `retire_store`, `copy_store`, `move_store`
   and `fingerprint_store` all take one for the same reason. Without it, the running-run refusal
   could only ever use the default `var/runs/`, which would silently read the *real* run registry
   from a test in a temporary directory, exactly what the hard rules forbid ("every test passes
   `runs_root` explicitly"). The hard rules governing this dispatch already assume the parameter
   exists ("Every `amend_sidecar`... call in your tests passes `runs_root`"), so this is read as
   README's interface line being abbreviated, not as a instruction to omit the parameter. README.md
   is outside the set of files this prompt may touch, so the signature there is not corrected; this
   deviation is the record of the difference.
2. **The CLI's `--json`/`--remove` are both optional at the argparse level; `amend_sidecar` enforces
   "exactly one".** `IMPLEMENTATION CHOICE`. §2.1 lists "both value and remove, or neither" as one
   of the operation's own `RuntimeError` refusals, alongside the sidecar and field checks, not as a
   usage error. Making the CLI flags an argparse `required` mutually exclusive group would move that
   rule out of the operation and change its exit code from 1 to 2, which nothing in the prompt asks
   for; `--field`/`--reason` are the ones given `required=True`, matching the existing
   `--purpose`/`--reason` precedent for a genuinely *missing* flag.
3. **A bad `--json` string is refused inside `_amend`, not the operation.** `IMPLEMENTATION CHOICE`.
   §2.1 draws the line explicitly: "a `value` that is not JSON-serialisable. On the command line,
   `--json` that does not parse" — two different failure modes at two different layers. `_amend`
   catches `json.JSONDecodeError` itself and prints a refusal in the same `!! ... Nothing was
   written` shape, exit 1, before ever calling `amend_sidecar`.
4. **`_FIELD_OWNERS` is a fixed mapping, not a single "amend, fingerprint, retire" sentence.**
   `IMPLEMENTATION CHOICE`. §2.1 says the message "names the operation that owns it: create, adopt,
   copy, move, fingerprint or retire" without specifying whether that is one shared list or a
   per-field one. A per-field mapping (e.g. `copied_from` → `copy` alone, `retired` → `retire`
   alone) is more informative than repeating the full list for every field, and is still built from
   exactly that vocabulary.
5. **The known-field check runs before the running-run check.** `IMPLEMENTATION CHOICE`. §2.1's
   bullet order lists the known-field refusal (3rd) ahead of the running-run refusal (9th, after
   the remove/value checks), which independently is the order best matched by placing the check
   right after the sidecar is confirmed problem-free, mirroring where `retire_store` places its own
   sidecar-validity checks before its running-run check.

None is `UNINTENDED DRIFT`.

## Verification performed

- **Baselines on `89529f4`** (this dispatch's HEAD), measured in this checkout: AdaptiveLevin 32,
  ComputeTargets 552, CosmologyModels 39, Datastore 206, LiouvilleGreen 148 (1 skipped), RunRegistry
  169. All OK, matching the dispatch's stated baselines.
- **The new module:** `Ran 21 tests … OK`, run on its own and within the `RunRegistry` suite,
  several times, in under 6 s.
- **All six suites after the change**, run from the repository root with `PYTHONPATH=.
  ./venv/bin/python -m unittest discover -s <package>/tests -t .`:

  | Suite | Before | After |
  |---|---|---|
  | `AdaptiveLevin` | 32 OK | 32 OK |
  | `ComputeTargets` | 552 OK | 552 OK |
  | `CosmologyModels` | 39 OK | 39 OK |
  | `Datastore` | 206 OK | 206 OK |
  | `LiouvilleGreen` | 148 OK, skipped=1 | 148 OK, skipped=1 |
  | `RunRegistry` | 169 OK | 190 OK (169 + the 21 tests this prompt adds) |

- **`black --check`** is clean on the three changed files.
- **Every existing test passes unmodified.** `git diff --cached --stat` (before this commit) shows
  the diff confined to `RunRegistry/stores.py`, `RunRegistry/__main__.py`, the new test module, this
  log, the board and (not needed here) `docs/OPEN_ISSUES.md`; no existing test file appears in it,
  and `store_fixtures.py` is untouched.
- **`import RunRegistry` loads neither `ray` nor `sqlalchemy`.** `python -c "import RunRegistry,
  sys; print('ray' in sys.modules, 'sqlalchemy' in sys.modules)"` → `False False`. Also checked by
  subprocess inside the new module (`TestCommandLine.test_import_loads_neither_ray_nor_sqlalchemy`)
  and by every `command()` call in `AmendTestCase`, which asserts `not report["ray_initialised"]`.
- **Nothing under `var/` was opened.** Every store in the new module is built in a
  `tempfile.TemporaryDirectory` (`StoreTestCase.setUp`). No `amend_sidecar` or `store amend` call,
  anywhere in this dispatch, named a path under `var/` or a copy of one; the only stores touched are
  the ones the tests built.
- **The tombstone tests never call `retire_store`.** `AmendTestCase.a_tombstone` builds a tombstone
  sidecar by hand, in the style `TestHistoryRule` in `test_store_retire.py` already uses for its
  own hand-built tombstone case, so no real deletion, `ShardedPool` call or fingerprint is ever
  needed to test that amend refuses one.
- **Roots.** Every `amend_sidecar` call in the new module goes through `AmendTestCase.amend`, which
  passes `runs_root`, except `test_a_runs_root_that_is_not_a_directory`, which passes it directly to
  demonstrate the refusal. Every `store amend` command passes `--runs-root`. Checked by reading the
  module; no call omits it.
- **Scope.** `git diff --cached --stat` (this commit) touches exactly `RunRegistry/stores.py`,
  `RunRegistry/__main__.py`, `RunRegistry/tests/test_store_amend.py`, this log,
  `IMPLEMENTATION_STATE.md`. `Datastore/`, `tools/`, `CLAUDE.md`, `RunRegistry/__init__.py`, README
  and every existing test file are untouched. The untracked `docs/datastore-integrity-audit*` and
  `prompts/datastore-integrity/` were not staged, read for content, or changed, and no `orch_*` file
  was touched.

## The deliberate-breakage record

Each diff below is exactly as applied: `RunRegistry/stores.py` was edited, `git diff` was taken
against the index (this commit's three source files were staged first), the affected test module
was run, and the file was restored with `git checkout --`. Each mutation was checked with `git
apply --check` before being applied this way, and the tree was confirmed clean
(`git diff --stat -- RunRegistry/` empty) after each reversal. The run was always
`PYTHONPATH=. ./venv/bin/python -m unittest RunRegistry.tests.test_store_amend`. Unmutated, that
run is `Ran 21 tests … OK`. After the commit, each diff below was extracted from this file and
`git apply --check`ed against `HEAD` (§ "State handed to the next prompt").

**(i) Allow a known field.** `FAILED (failures=12)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 54f6cf6..fe04cc4 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2079,13 +2079,6 @@ def amend_sidecar(
         )
     fields = reading.fields
 
-    if field in KNOWN_FIELDS:
-        owner = _FIELD_OWNERS.get(field, "another operation")
-        raise refuse(
-            f"{field!r} is a known field, owned by {owner}; amend replaces or removes only an "
-            f"unknown field, one this registry carries and never interprets"
-        )
-
     # the recorded field, read directly: reading.ok is already established, but the pattern
     # matches retire_store's, which must read it this way for a tombstone
     store_id = fields["store_id"]
```

**Outcome.** A known field, `retired` included, can now be amended directly, bypassing the
operation that owns it. `TestRefusals.test_every_known_field_is_refused_naming_its_owner` fails
for every member of `KNOWN_FIELDS` that amend can actually touch without producing an unrelated
error (12 of the 20 subtests: the others, e.g. `sidecar_format` or `history`, still fail some other
way — a malformed sidecar, or a validator problem — once written with a nonsense value, which is
itself evidence that nothing here is silently fine).

**(ii) Omit `before` from the entry.** `FAILED (failures=1, errors=7)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 54f6cf6..721ead5 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2119,7 +2119,7 @@ def amend_sidecar(
 
     provenance = git_provenance()
     entry = _history_entry(AMEND_OPERATION, None, None, provenance)
-    entry.update(field=field, reason=reason, before=before, after=after)
+    entry.update(field=field, reason=reason, after=after)
 
     new_fields = _copy.deepcopy(fields)
     if remove:
```

**Outcome.** Most failures are `KeyError: 'before'`, not a clean refusal: `amend_sidecar`'s own
return statement reads `before["value"] if before["present"] else None`, using the local variable
`before` it already built — that variable still exists, so the write of an entry that *lacks*
`before` succeeds structurally at the Python level, and only the return-value construction and
every caller reading `entry["before"]` breaks. This is prompt 03's validator layer *not* catching
it (the `_check_before_writing` gate is about the *shape* of `AMEND_KEYS`' values when present;
here the write itself does not go through `_check_before_writing` with a `before`-shaped problem —
`missing_amend` would in fact catch a truly-missing key, but this mutation is caught first, before
the write, by the Python-level `KeyError` in the return statement). `errors=7`:
`test_replace_an_unknown_field`, `test_copy_carries_an_amended_field_and_its_entry`,
`test_move_carries_an_amended_field_and_its_entry`, `test_amend_backup_as_prompt_05_will`,
`test_a_value_identical_after_a_json_round_trip`, `test_remove_an_unknown_field`,
`test_two_amendments_of_one_field_read_back_as_a_sequence`. `failures=1`:
`test_json_remove_bad_json_and_exit_codes` (the CLI's successful-amendment assertions on `before:`
no longer hold, since a `KeyError` inside `_amend` is not a caught `RuntimeError` and propagates as
a subprocess traceback instead of a clean refusal).

**(iii) Skip the running-run check.** `FAILED (failures=3)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 54f6cf6..c415e09 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2089,12 +2089,6 @@ def amend_sidecar(
     # the recorded field, read directly: reading.ok is already established, but the pattern
     # matches retire_store's, which must read it this way for a tombstone
     store_id = fields["store_id"]
-    running = _running_runs_naming("amend", [primary], store_id, runs_root)
-    if running:
-        raise refuse(
-            f"{_describe_running(running)}. A stale run is ended by a person, with Run.finish, "
-            f"before its sidecar is amended; the registry does not decide that a run is dead"
-        )
 
     if remove:
         if field not in fields:
```

**Outcome.** A store a `running` run names, alive or stale, or by `results_store_id` alone, is
now amended without refusal: `TestRefusals.test_an_alive_and_a_stale_running_run` and
`test_a_running_run_by_store_id_alone` fail (`RuntimeError not raised`). A third test also fails,
`test_a_runs_root_that_is_not_a_directory`: the runs-root validity check lives inside
`_running_runs_naming` itself, not as a separate check in `amend_sidecar` (matching
`fingerprint_store`'s own pattern), so removing the running-run check silently removes that
refusal too.

**(iv) Let the validator accept an amend with no reason.** `FAILED (failures=1)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 54f6cf6..7860172 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -366,10 +366,6 @@ def _history_problems(history) -> List[str]:
                     problems.append(
                         f"{where} (amend) has a malformed field, {entry.get('field')!r}"
                     )
-                if not _nonempty_string(entry.get("reason")):
-                    problems.append(
-                        f"{where} (amend) has a malformed reason, {entry.get('reason')!r}"
-                    )
                 for key in ("before", "after"):
                     problems.extend(
                         _amend_slot_problems(entry.get(key), f"{where} (amend)'s {key}")
```

**Outcome.** `amend_sidecar` itself already refuses a blank or missing `reason` before it ever
builds an entry, so this mutation is invisible to every `amend_sidecar`-level test — it can only be
seen by feeding `_history_problems` a hand-built `amend` entry directly, which is what
`TestHistoryRule.test_amend_stands_after_index_0_and_before_any_retire`'s "amend with a blank
reason" subtest does. Failed: that one subtest (`FAILED (failures=1)`), confirming the review
note's caution that a layered refusal needs a test aimed at the layer, not just at the operation.

**(v) Allow an amendment of a tombstone.** `FAILED (failures=2)`.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 54f6cf6..7c5b36e 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -2069,8 +2069,6 @@ def amend_sidecar(
         raise refuse("neither a value nor --remove was given; exactly one is required")
 
     reading = read_sidecar(primary)
-    if reading.retired:
-        raise refuse(_tombstone_text(reading))
     if not reading.ok:
         raise refuse(
             f'its sidecar "{reading.path}" is not a problem-free registry sidecar '
```

**Outcome.** A tombstone is still refused — `SidecarReading.ok` is false for any `retired` sidecar,
tombstone-specific check or not, so `not reading.ok` still catches it — but the message changes
from "is a tombstone … never reused" to the generic "is not a problem-free registry sidecar"
wording. `TestRefusals.test_a_tombstone_complete_or_not` fails both subtests (`FAILED
(failures=2)`), because it asserts the tombstone-specific fragments, not merely that *some*
`RuntimeError` was raised. This is exactly the layered-refusal case the dispatch's carried-forward
notes named explicitly: without a message-level assertion, this mutation would pass every test
silently, since the store is still refused either way.

## Observations not acted on

1. **README §4's listed `amend_sidecar` signature omits `runs_root`.** Recorded as Deviation 1.
   `README.md` is not in the set of files this prompt may touch, so its interface line is not
   corrected here. Not a new issue: the deviation record above is the place this is tracked, and a
   later prompt with README in scope can fold it in when convenient.
2. **`[03-the-package-docstring-still-says-the-registry-deletes-nothing]` stays open.** Unchanged
   by this prompt: `RunRegistry/__init__.py` was not in scope here either (prompt 04's file list is
   `RunRegistry/stores.py`, `RunRegistry/__main__.py`, tests, the log and the board). See the board
   §3 and index §1.14.

No new issue is opened by this prompt.

## State handed to the next prompt

- **For prompt 05:** `python -m RunRegistry store amend PRIMARY --field NAME (--json VALUE |
  --remove) --reason TEXT --runs-root DIR` is ready to correct the live A3 sidecar's `backup`
  field once the backup itself is retired. It refuses a tombstone, a known field, and a store a
  `running` run names; it needs no `--stores-root` (amend never searches for references — that is
  `retire_store`'s job under D5, already done in prompt 03).
- **R6–R9 are done.** `ShardedPool`, `tools/sharded_store.py`, `CLAUDE.md`, every run manifest and
  `RunRegistry/__init__.py` are unchanged by this prompt.
- **The diffs above** were checked after the commit: each, extracted from this file, passes
  `git apply --check` against `HEAD`.
- **Issue opened:** none.
