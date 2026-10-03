# Log — prompt 06: count only real references when a store is retired

**Subject:** *Count only real references when a store is retired* · **Model:** Claude Sonnet 5.5 ·
**Date:** 2026-10-03 · **Base:** `5e2d97a`, clean · **Result:** landed. `_find_references` no
longer takes the stores root as a target, so retiring a store directly in `var/datastores/` no
longer lists every earlier tombstone as naming it. A store in its own subdirectory, the live A3
backup's shape, keeps its directory as a target. Neither stop condition of the prompt's §5 fired.
Nothing was written under `var/`, no `store retire` was run against the real stores root, no
tombstone was rewritten, no Ray was started. `[10-store-retire-reports-every-tombstone-as-a-reference]`
is closed on this board's §4, with its index row deleted (103 to 102). Nothing was opened. The
commit is this commit, filled in by the orchestrator's review on the board's §1 row and in the
closing note.

## What shipped

- **`RunRegistry/stores.py`, `_find_references`** (the only function changed). The target set was
  `{realpath(p) for p in [primary, *planned, primary.parent]}`. It is now
  `{realpath(p) for p in [primary, *planned]}`, and `os.path.realpath(primary.parent)` is added
  only `if` it differs from `os.path.realpath(stores_root)`. The root that is compared is the root
  `os.walk(stores_root)` walks, resolved the same way (cwd-relative for a relative root). The
  docstring now says what a match by a sidecar naming the store's directory means: the sidecar
  names the directory the store sits in, which is how the live A3 sidecar named its backup, and
  which may be shared with other stores, so the match is weaker than one by the primary; and why
  the stores root is excluded. `not_searched` is unchanged: F1 changes what is a target, not what
  is walked. Nothing else in the file changed (`git diff --cached --stat`: 22 insertions, 2
  deletions, all in this function).
- **`RunRegistry/tests/test_store_retire_references.py`** (new, 10 tests), below.
- **This log**, the board (§1 row, F1–F2, §3.2 to §4, a new §5.12), and `docs/OPEN_ISSUES.md` (the
  row deleted; count and date).
- Not touched: `RunRegistry/__main__.py` (it prints `sidecars naming it, under …: N` from the
  report and consumes nothing else), every other module, every existing test (including
  `test_store_retire.py`, no line of which changed), `var/`, and
  `prompts/a3-v2-readiness/orchestrator/`.

### The tests (`RunRegistry/tests/test_store_retire_references.py`, 10)

Built on `test_store_retire.RetireTestCase` (its temporary directory, `runs/` and `datastores/`,
and its real multi-shard fixture stores), reached by importing the module rather than the class so
that the base class is not collected a second time. A store directly in the stores root is
`a_full_store("", stem=…)`. Every test that expects no match first reads, from the tombstone
itself, that its `retired.references.stores_root` resolves to the stores root, so that it cannot
pass because the field that used to match is absent.

- **`TestTheStoresRootIsNotATarget`** (5), E1.
  - `test_three_stores_in_the_root_retired_in_turn`: no report lists an earlier tombstone, and the
    tombstone's own `references.sidecars` is `[]`.
  - `test_through_a_stores_root_that_is_a_symlink`: the same, with the stores root passed as a
    symlink to the real one (one store by the real root, two through the link).
  - `test_through_a_primary_that_is_reached_by_a_symlink`: the primary passed through the link, the
    root real.
  - `test_a_sidecar_that_is_not_a_tombstone_naming_the_root`: a registry sidecar in the root with
    a field naming the root by absolute path and by a path relative to the repository: not
    reported. It is still a store afterwards.
  - `test_the_same_sidecar_is_reported_for_a_field_naming_the_primary`: that sidecar, with a third
    field naming the primary, is reported by that field alone, `["$.notes.primary"]`.
- **`TestADirectoryIsStillATarget`** (5), E2.
  - `test_a_store_in_a_subdirectory_named_by_the_subdirectory`: a sidecar names `S/` by absolute
    and relative path and also names the root: reported by the first two fields only; a sidecar
    that names only the root is not reported.
  - `test_a_subdirectory_shared_with_another_store_is_still_a_match`: as `test_a_retirement`'s
    shape, a second store in `S/`.
  - `test_a_tombstone_in_the_root_does_not_name_a_subdirectory`: a store retired directly in the
    root, then one in `S/`: the second's report is empty.
  - `test_a_store_outside_the_stores_root_keeps_its_directory`.
  - `test_a_store_in_the_root_is_still_found_by_what_names_it_itself`: a copy's
    `$.copied_from.store_id` and `$.copied_from.datastore` are found with the root excluded.

## Deviations from the prompt

1. **A new module, not a class appended to `test_store_retire.py`.** The prompt leaves it to the
   prompt; the new module leaves that file's every line as it was. **IMPLEMENTATION CHOICE.**
2. **A tombstone's own `references` subtree is not skipped.** The prompt allows it, on the ground
   that it records what a retirement searched. Not taken: F1 alone removes every false match found
   (the five in the tombstones under `var/datastores/`, and the three reconstructed below), and the
   skip would be a second rule. What it would hide is every string inside `$.retired.references`
   of every tombstone: `runs_root`, `stores_root`, `runs[].id`, `runs[].matched_by` and
   `sidecars[].sidecar`. Each records a search root or a sidecar that was found, none of them
   names a store by its primary, a planned file or its directory, so nothing real is hidden
   today, but a `--stores-root` that is another store's own directory would be. **IMPLEMENTATION
   CHOICE.**
3. **Tests beyond F2's three bullets**: the two symlink tests (they carry breakage (iii)), the
   store outside the root, the copy's `copied_from`, and the tombstone-then-subdirectory case.
   **IMPLEMENTATION CHOICE.**
4. **`not_searched` is unchanged**, as the prompt allows. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT` was found. No `STRUCTURALLY REQUIRED` deviation arose.

## Verification performed

### The tombstones already under `var/datastores/`, read, never written (stop condition 2)

Read with a script that opens each `*.manifest.json` under `var/datastores/` for reading and prints
each tombstone's `retired.references.sidecars`:

| Tombstone | Matches it lists |
|---|---|
| `datastore-integrity-10-reference` (2026-10-02) | `handover-atol-sweep` and the backup's sidecar, each by `$.retired.references.stores_root` alone |
| `datastore-integrity-10-resumed` (2026-10-02) | `datastore-integrity-10-reference`, `handover-atol-sweep` and the backup's sidecar, each by `$.retired.references.stores_root` alone |
| `handover-atol-sweep` (2026-09-25) | none |
| `backup-pre-resume-…/handover-A3-baseline-lambdacdm` (2026-09-25) | the live A3 sidecar, by `$.backup.path` |

F1 removes the five `$.retired.references.stores_root` matches. None names a store: each field is
a recorded search root. The one real match, the backup's, is by a directory that is not the stores
root (`backup-pre-resume-20260921T091011/`), so F1 keeps it. Stop condition 2 does not fire. No
tombstone is rewritten.

### Stop condition 1 (an existing test would have to change)

Not fired: with F1 applied, every existing RunRegistry test passes unchanged, and no existing
test was edited (`git diff --cached` shows `RunRegistry/stores.py` and the new test module as the
only Python files).

### The second 2026-10-02 retirement, before and after (the prompt's §7)

Rebuilt in the fixture's temporary stores root (never under `var/`), by real retirements in
sequence, in the order of the real ones: `handover-atol-sweep` (directly in the root); the live A3
sidecar (directly in the root, its `backup.path` naming the backup's directory) and then the backup
in `backup-pre-resume-20260921T091011/`; `datastore-integrity-10-reference`; and then
`datastore-integrity-10-resumed`, whose report is the one asked for. "Before" is
`_find_references` exactly as the base commit `5e2d97a` has it (read with `git show` and `ast`, and
put in place of `stores._find_references`); "after" is this commit's. The script is in the scratch
directory, not the repository, and is given in full below so that it can be re-run (from the
repository root, in `bash`; it needs `5e2d97a` in the history for "before").

Before (3 sidecars), as the real `datastore-integrity-10-resumed` tombstone read:

```
second of the pair's report:
    datastore-integrity-10-reference.manifest.json ['$.retired.references.stores_root']
    handover-atol-sweep.manifest.json ['$.retired.references.stores_root']
    backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json ['$.retired.references.stores_root']
   count: 3
```

After (none):

```
second of the pair's report:
   count: 0
```

The same script prints the other two reports: at both trees the backup's retirement lists
`handover-A3-baseline-lambdacdm.manifest.json` (the live sidecar) by `$.backup.path` alone, the
real match kept; and the first of the pair lists two sidecars before (the sweep's and the backup's
tombstone) and none after, as the real one listed two.

```python
"""Rebuild, in a temporary stores root, the second 2026-10-02 retirement and print what
`_find_references` returns for it. Run from the repository root, in bash:

    PYTHONPATH=. ./venv/bin/python <this file>             # after: the working tree's function
    BEFORE=1 PYTHONPATH=. ./venv/bin/python <this file>    # before: the function at 5e2d97a, the base
                                                           # (its RunRegistry/stores.py is c372bee's)
Nothing is written under var/.
"""

import ast
import json
import os
import subprocess
import sys
import unittest

from RunRegistry import stores
from RunRegistry.tests.store_fixtures import write_sidecar_json
from RunRegistry.tests import test_store_retire


def head_find_references():
    """`_find_references` exactly as the base commit 5e2d97a has it."""
    source = subprocess.run(
        ["git", "show", "5e2d97a:RunRegistry/stores.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (node,) = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_find_references"
    ]
    scope = dict(vars(stores))
    exec(compile(ast.Module([node], []), "5e2d97a:stores.py", "exec"), scope)
    return scope["_find_references"]


class Reconstruction(test_store_retire.RetireTestCase):
    def test_the_second_retirement(self):
        if os.environ.get("BEFORE"):
            stores._find_references = head_find_references()
        rel = lambda p: os.path.relpath(p, self.stores_root)
        backup_dir = "backup-pre-resume-20260921T091011"

        # 1. the sweep store, directly in the root, retired (2026-09-25)
        self.retire(self.a_full_store("", stem="handover-atol-sweep"))
        # 2. the live A3 sidecar, directly in the root, naming its backup's directory; then the
        #    backup, alone in that subdirectory, retired (2026-09-25)
        live = self.a_registry_store(
            "datastores", stem="handover-A3-baseline-lambdacdm"
        )
        backup = self.a_full_store(backup_dir, stem="handover-A3-baseline-lambdacdm")
        fields = self.sidecar(live)
        fields["backup"] = {"path": str(backup.parent), "retained": True}
        write_sidecar_json(live, fields)
        result = self.retire(backup)
        print(
            "backup tombstone's report:",
            json.dumps(
                [
                    (rel(e["sidecar"]), e["fields"])
                    for e in result["references"]["sidecars"]
                ]
            ),
        )
        # 3. the first of the 2026-10-02 pair, directly in the root
        reference = self.a_full_store("", stem="datastore-integrity-10-reference")
        result = self.retire(reference)
        print(
            "first of the pair's report:",
            json.dumps(
                [
                    (rel(e["sidecar"]), e["fields"])
                    for e in result["references"]["sidecars"]
                ]
            ),
        )
        # 4. the second of the pair
        resumed = self.a_full_store("", stem="datastore-integrity-10-resumed")
        result = self.retire(resumed)
        entries = result["references"]["sidecars"]
        print("second of the pair's report:")
        for e in entries:
            print("   ", rel(e["sidecar"]), e["fields"])
        print("   count:", len(entries))
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(argv=["x", "Reconstruction"], verbosity=0)
```

### The other checks

- **§3 item 1.** The RunRegistry suite is 253 OK: the 243 at `HEAD` and the 10 new tests, with no
  existing test changed (the suites table below).
- **E4.** F1 changes the target set only. A `git diff` of `_find_references` shows the `return`,
  the sidecar walk and the unreadable-sidecar branch untouched, so refusals, deletions and the
  tombstone's other fields are as they were; `test_store_retire`, including its interruption
  tables, passes unchanged.
- **A breakage must fail for its own reason.** Each failure below was read: every one is an
  `AssertionError` on the report (a sidecar that should not be listed, or one that should be), and
  none is an error from the fixture.

### Suites

Measured 21:29 onward at the tree of this commit's code (the two Python files staged, the Markdown
edits unstaged), serially, with README §7's command, each suite's output written to a file and its
verdict grepped from it:

| Suite | Result |
|---|---|
| AdaptiveLevin | 32 OK |
| ComputeTargets | 686 run, 1 failure: the known wall-clock flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (below); the other 685 pass |
| CosmologyModels | 39 OK |
| Datastore | 598 OK |
| LiouvilleGreen | 148 OK (1 skipped) |
| RunRegistry | 253 OK |

RunRegistry is 243 + the 10 tests of `test_store_retire_references.py`; every other count is as at
`HEAD`. The 1-minute load average on 10 CPUs was 11.5 at the start and 10–21 during the run, beside
another project's test run.

**The one failure.** `test_wall_time_per_object` asserts a wall-clock bound
(`0.06 s`). It failed in the ComputeTargets run, and **alone** in each of five re-runs: four at
1-minute loads of 13–17 (one measured, 0.0653 s) and one at 6.9 (0.0907 s), each time on `... not
less than or equal to 0.06`. The brief records it failing at `HEAD`, with no code changed, at a load of ~17, and passing
at `dd7f272`. Nothing under `ComputeTargets/` changed, and no file in its import chain imports
`RunRegistry` (`grep -rl RunRegistry` over `ComputeTargets`, `Datastore`, `config` and
`CosmologyModels` finds only `Datastore/tests/test_quadsource_policy_key.py`). I call it the
known flake, not a regression, but did not see it pass on this machine in this session.

### Formatting and staging

`./venv/bin/python -m black --check` on the two Python files: each "would be left unchanged".
Files were staged by explicit path, and `git diff --cached --name-only` was shown before the
commit (see the report).

## The deliberate-breakage record

Each diff is exactly as applied, taken with `git diff -- RunRegistry/stores.py` against the staged
file, and reverted with `git checkout -- RunRegistry/stores.py` afterwards. Each was checked with
`git apply --check` in `bash` against the tree before the commit, with the Markdown edits present
but unstaged. None needs a test change that is not in the committed tests. The tests are in
`RunRegistry/tests/test_store_retire_references.py` unless named otherwise; `unittest` counts a
failing test once.

**(i) the directory back as an unconditional target** (pre-F1 behaviour). 5 failures, from the whole
RunRegistry suite (253 tests); no other test failed.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0687bb0..c80d887 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1623,8 +1623,7 @@ def _find_references(primary, planned, store_id, runs_root, stores_root, own) ->
 
     targets = {os.path.realpath(p) for p in [primary, *planned]}
     directory = os.path.realpath(primary.parent)
-    if directory != os.path.realpath(stores_root):
-        targets.add(directory)
+    targets.add(directory)
     own = os.path.realpath(own)
     # the sidecar suffix, taken from `sidecar_path`, where alone the pattern is spelled
     suffix = sidecar_path(Path("x.sqlite")).name[len("x") :]
```

- `TestTheStoresRootIsNotATarget.test_three_stores_in_the_root_retired_in_turn`
- `TestTheStoresRootIsNotATarget.test_through_a_stores_root_that_is_a_symlink`
- `TestTheStoresRootIsNotATarget.test_through_a_primary_that_is_reached_by_a_symlink`
- `TestTheStoresRootIsNotATarget.test_a_sidecar_that_is_not_a_tombstone_naming_the_root`
- `TestTheStoresRootIsNotATarget.test_the_same_sidecar_is_reported_for_a_field_naming_the_primary`
  (it lists `$.notes.relative` and `$.notes.root` beside `$.notes.primary`)

Each fails on the report: a tombstone listed "by `$.retired.references.stores_root`", or the
non-tombstone sidecar's root fields listed.

**(ii) the directory dropped as a target entirely.** 4 failures, from `test_store_retire` and this
module together (51 tests).

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0687bb0..29c86bc 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1623,8 +1623,7 @@ def _find_references(primary, planned, store_id, runs_root, stores_root, own) ->
 
     targets = {os.path.realpath(p) for p in [primary, *planned]}
     directory = os.path.realpath(primary.parent)
-    if directory != os.path.realpath(stores_root):
-        targets.add(directory)
+    del directory
     own = os.path.realpath(own)
     # the sidecar suffix, taken from `sidecar_path`, where alone the pattern is spelled
     suffix = sidecar_path(Path("x.sqlite")).name[len("x") :]
```

- `test_store_retire.TestRetirement.test_a_retirement`, unchanged: the expected sidecar set lacks
  `datastores/D/store.manifest.json`, the backup's, which names `S` by `$.backup.path`.
- `TestADirectoryIsStillATarget.test_a_store_in_a_subdirectory_named_by_the_subdirectory`
- `TestADirectoryIsStillATarget.test_a_subdirectory_shared_with_another_store_is_still_a_match`
- `TestADirectoryIsStillATarget.test_a_store_outside_the_stores_root_keeps_its_directory`

This is E2: dropping the directory outright loses the backup's match, and the existing test sees it.

**(iii) the stores root compared by text, not by `realpath`.** 2 failures, from the same 51 tests;
every test whose root is not a symlink still passes, which is the point.

```diff
diff --git a/RunRegistry/stores.py b/RunRegistry/stores.py
index 0687bb0..52bb4ca 100644
--- a/RunRegistry/stores.py
+++ b/RunRegistry/stores.py
@@ -1623,7 +1623,7 @@ def _find_references(primary, planned, store_id, runs_root, stores_root, own) ->
 
     targets = {os.path.realpath(p) for p in [primary, *planned]}
     directory = os.path.realpath(primary.parent)
-    if directory != os.path.realpath(stores_root):
+    if str(primary.parent) != str(stores_root):
         targets.add(directory)
     own = os.path.realpath(own)
     # the sidecar suffix, taken from `sidecar_path`, where alone the pattern is spelled
```

- `TestTheStoresRootIsNotATarget.test_through_a_stores_root_that_is_a_symlink`
- `TestTheStoresRootIsNotATarget.test_through_a_primary_that_is_reached_by_a_symlink`

In the plain tests the two spellings are the same string, so a textual comparison passes them. The
fixture's `top` is already `.resolve()`d, so only the symlink makes the two paths differ as text
while naming one directory. Each failure lists the earlier tombstone by
`$.retired.references.stores_root`.

## Observations not acted on

1. **README §2's row for 06 describes a narrower rule than the one shipped, and is stale.** It
   says `_find_references` "stops matching a store's directory when that directory is shared with
   other stores" and that the backup shape is "a store alone in its own directory". The shipped
   rule excludes the stores root only, and a directory shared with other stores, as in
   `test_a_retirement`, still matches (that test requires it). The row's status ("**written**;
   independent") is also stale now that 06 has landed. The README is outside this prompt's file
   list.
2. **A directory match is weak by design.** A sidecar that names `S/` is reported for every store
   in `S/`, which is why the docstring calls it that. The report does not say, per field, that a
   match was by directory and not by the primary or a planned file; `fields` are JSON paths only.
   Not asked for here.
3. **Only the stores root is excluded.** A directory above it, or any other directory that happens
   to hold every store (a stores root that is itself a subdirectory of another), is not
   special-cased. No such case exists.
4. **The first run of breakage (i) used the whole RunRegistry suite** (about 2.3 minutes) rather
   than the two modules the other breakages used, because it was started in the same command that
   restored the file. The three results are of the same tests where they overlap.
5. **`docs/OPEN_ISSUES.md`'s §1.17 intro prose** gives no count of open issues, so only the
   header count changed; the header date was already 2026-10-03.

## State handed to the next prompt

- **07** (`rehearse the A3 v2 launch`) is held on 01–06, and 06 touches nothing it reads. After
  this commit `store retire` of a store directly in `var/datastores/` reports the references to
  that store and not a growing list of tombstones.
- **Baselines** (board §5.12): see the suites table above.
- **Index:** 102 open, from 103. Nothing opened.
- **The orchestrator's residue:** this commit's SHA, in the board's §1 row for 06 and in the
  closing note of `[10-store-retire-reports-every-tombstone-as-a-reference]` on §4; and README §2's
  row for 06 (observation 1).
