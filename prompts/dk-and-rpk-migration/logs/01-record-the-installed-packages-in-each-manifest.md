# Log — prompt 01, record the installed packages in each manifest

**Prompt:** [`../01-record-the-installed-packages-in-each-manifest.md`](../01-record-the-installed-packages-in-each-manifest.md) ·
**Board items:** P1–P3 · **Date:** 2026-10-11 · **Model:** Sonnet 5.5 ·
**Commit:** this commit · **Result:** landed. P1–P3 as the prompt says; every suite at its
README §7 count plus the six new tests; no issue opened or closed.

Started from `6edb2f9` (code identical to `b510bc9`), tree clean, `RunRegistry list` showing 17
runs and none `running`. Neither `datastorekit` nor `raypoolkit` is installed
(`importlib.metadata.distribution` raises `PackageNotFoundError` for both).

## What shipped

**`RunRegistry/__init__.py`**
- `import importlib.metadata` (standard library). The function calls
  `importlib.metadata.distribution`, so a test patching that attribute replaces the name the module
  actually calls.
- `RECORDED_DISTRIBUTIONS = ("datastorekit", "raypoolkit")`, with a comment saying why (U4).
- `package_provenance(names=RECORDED_DISTRIBUTIONS) -> dict`, beside `git_provenance`. Maps each
  name to `None` when `PackageNotFoundError` is raised, and otherwise to a dict with exactly
  `version`, `url`, `commit`, `requested_revision`, `editable`, read from PEP 610's
  `direct_url.json` (`url`; `vcs_info.commit_id`; `vcs_info.requested_revision`;
  `dir_info.editable is True`). Only `PackageNotFoundError` gives `None` for the whole entry. A
  `direct_url.json` that is absent, empty, not JSON, not a JSON object, or whose `vcs_info` or
  `dir_info` is not an object, gives `None, None, None, False` and never raises. It imports no
  package it describes.
- `begin()`: `manifest["packages"] = package_provenance()` after `manifest.update(git_provenance())`,
  before `manifest.json` is written, so it is written once with the rest.
- `begin()`'s docstring: see the first deviation.

**`RunRegistry/tests/test_run_registry.py`**: `import importlib.metadata`, a helper class
`StandInDistribution`, the new class `TestPackageProvenance` and one method in `TestManifest`.
No existing test changes.

**The signature and its output in this venv.**
```
package_provenance(names=RECORDED_DISTRIBUTIONS) -> dict
>>> RunRegistry.package_provenance()
{'datastorekit': None, 'raypoolkit': None}
```
The hand check (prompt §3.2): `RunRegistry.begin(campaign="dk", prompt="01", slug="hand", ...,
root=<temporary directory under the session scratchpad>)` from `./venv/bin/python`. The manifest's
`packages` is `{"datastorekit": null, "raypoolkit": null}`, both in the returned object and as
read back from `manifest.json`. `var/runs/` gained nothing (it holds the same 17 runs, and
`RunRegistry list` still says 15 finished, 2 unknown).

Not asked for, and not committed: a probe against the real `importlib.metadata`, with no mock. A
`datastorekit-0.2.2.dist-info` directory holding `METADATA` and a `direct_url.json` was put on
`PYTHONPATH` in the scratchpad, outside the repository. `package_provenance()` gave `{'datastorekit':
{'version': '0.2.2', 'url': 'git+https://x/y.git', 'commit': 'abc', 'requested_revision': 'v0.2.2',
'editable': False}, 'raypoolkit': None}`.

**Tests added**, all in `RunRegistry/tests/test_run_registry.py`:
- `TestPackageProvenance`:
  - `test_a_distribution_not_installed_is_none`
  - `test_a_vcs_install_records_its_commit_and_tag`
  - `test_an_editable_install_is_recorded_as_editable`
  - `test_an_absent_or_unreadable_direct_url_gives_nones` (seven sub-cases: absent, not JSON, a
    list, a string, `null`, empty text, and `vcs_info`/`dir_info` that are not objects)
  - `test_the_recorded_names_are_the_two_packages`
- `TestManifest`:
  - `test_manifest_records_the_installed_packages`: `packages` has exactly the keys of
    `RECORDED_DISTRIBUTIONS` and equals `package_provenance()`, in the returned manifest and on
    disk. It asserts nothing about what the venv holds, so it passes unchanged after prompts 03
    and 06.

`TestPackageProvenance` patches `RunRegistry.importlib.metadata.distribution` with
`mock.patch.object`, which is the object the function calls, so the patch is not vacuous.

## Deviations from the prompt

1. **`begin()`'s docstring corrected.** It said "Every manifest field but one is one README §0
   would have caught something with; there are no others. The one is `results_store_id`". After
   this change that is false. It now says "but two", `results_store_id` becomes "the first", and a
   paragraph says `packages` is the second: what was installed of each adopted package when the run
   began, there so a run says which release it ran (U4) instead of that being implied by
   `requirements.txt`. Nothing else in the docstring changed. **STRUCTURALLY REQUIRED** (README
   §6.2 C10: prose a commit makes false is corrected in that commit; and the prompt asks for
   `packages` to be mentioned where the docstring lists the manifest).
2. **`test_an_absent_or_unreadable_direct_url_gives_nones` has seven sub-cases, not two.** The
   prompt names two (`None`, and text that is not a JSON object); the rest cover the brief's
   requirement that odd metadata never raises. **IMPLEMENTATION CHOICE.**
3. **`test_manifest_records_the_installed_packages` also reads `manifest.json` back** and checks
   `packages` on disk. The prompt asks only for the returned manifest. **IMPLEMENTATION CHOICE.**
4. **`editable` is `dir_info.get("editable") is True`**, so a non-boolean truthy value (for
   example the string `"yes"`) gives `False`. The prompt says "`True` when `dir_info.editable` is
   true". PEP 610 makes it a boolean; the strict test cannot raise on odd metadata. **IMPLEMENTATION
   CHOICE.**
5. **Mutations were diffed against the staged tree.** The two files were `git add`-ed before the
   mutations, so each diff below is a diff against this commit's content and replays from a clean
   tree at this commit. **IMPLEMENTATION CHOICE.**

No `UNINTENDED DRIFT`.

## Verification performed

Counts at `6edb2f9` (before) against after. Load averages are the 1-minute figure when each ran.

| Root | Before | After | Load | Verdict |
|---|---|---|---|---|
| RunRegistry | 280 | 286 | 22.5 | OK |
| AdaptiveLevin | 32 | 32 | 17.7 | OK |
| CosmologyModels | 39 | 39 | 17.7 | OK |
| LiouvilleGreen | 148 (1 skipped) | 148 (1 skipped) | 16.6 | OK |
| ComputeTargets | 693 | 693 | 13.3 | FAILED (1), see below |
| Datastore `[a-f]` | 163 | 163 | 12.4 | OK |
| Datastore `[g-o]` | 185 | 185 | 11.8 | OK |
| Datastore `[p-r]` | 200 | 200 | 13.8 | OK |
| Datastore `[s-z]` | 191 | 191 | 21.5 | OK |

Datastore total 739. `black --check` is clean on both touched files.

**ComputeTargets** failed one test,
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`, one of README §7's four load-sensitive
tests. Re-run alone it failed twice and passed twice at load 12 to 17, with a best-of-3 of 0.049 to
0.073 s against the 0.05 s threshold. The same test was also run on a `git archive HEAD` export of
the tree (the tree before this prompt's changes, in the scratchpad): it failed once and passed
once, with the same spread. It is the test the README says has about 1x margin, it touches nothing
this prompt changed, and it behaves the same on the tree before and after. Not a failure of this
prompt. The other 692 tests passed.

## The deliberate-breakage record

Each diff is exactly as applied, against this commit. Each was checked with `git apply -R --check`
and `git apply --check` (both passed), and the working tree restored afterwards. All were run
in `bash`.

### (i) `editable` is always `False`

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 0228a1a..69bc18a 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -164,7 +164,7 @@ def package_provenance(names=RECORDED_DISTRIBUTIONS) -> dict:
             "url": origin.get("url"),
             "commit": vcs_info.get("commit_id"),
             "requested_revision": vcs_info.get("requested_revision"),
-            "editable": dir_info.get("editable") is True,
+            "editable": False,
         }
     return packages
 
```
Failed: `TestPackageProvenance.test_an_editable_install_is_recorded_as_editable`. Only that test
(of five run).

### (ii) a `direct_url.json` that is not a JSON object raises

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 0228a1a..7153ca8 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -152,7 +152,7 @@ def package_provenance(names=RECORDED_DISTRIBUTIONS) -> dict:
         except (TypeError, ValueError, OSError):
             origin = None
         if not isinstance(origin, dict):
-            origin = {}
+            raise TypeError("direct_url.json is not a JSON object")
         vcs_info = origin.get("vcs_info")
         if not isinstance(vcs_info, dict):
             vcs_info = {}
```
Failed: `TestPackageProvenance.test_an_absent_or_unreadable_direct_url_gives_nones`, six of its
seven sub-cases erroring (`absent`, `not json`, `a list`, `a string`, `null`, `empty`). The seventh
(`vcs_info`/`dir_info` not objects) has an object at the top, so this mutation does not reach it.

### (iii) `begin()` no longer writes `packages`

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 0228a1a..6db04ee 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -647,7 +647,6 @@ def begin(
         "heartbeat_means": heartbeat_means,
     }
     manifest.update(git_provenance())
-    manifest["packages"] = package_provenance()
     write_json_atomic(os.path.join(path, "manifest.json"), manifest)
     run = Run(path)
     run._update(
```
Failed: `TestManifest.test_manifest_records_the_installed_packages`.

### (iv) `RECORDED_DISTRIBUTIONS` loses `"raypoolkit"`

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 0228a1a..96b977a 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -116,7 +116,7 @@ def git_provenance(cwd=REPO_ROOT) -> dict:
 # adopts (README §0, U4). From prompt 04 on a run executes code that is not in this repository's
 # tree, so the manifest records which release of it was installed, and R0, R1 and R2 each say what
 # they ran.
-RECORDED_DISTRIBUTIONS = ("datastorekit", "raypoolkit")
+RECORDED_DISTRIBUTIONS = ("datastorekit",)
 
 
 def package_provenance(names=RECORDED_DISTRIBUTIONS) -> dict:
```
Failed: `TestPackageProvenance.test_the_recorded_names_are_the_two_packages` (the intended pin), and
also `test_a_distribution_not_installed_is_none` (it expects both names) and
`test_a_vcs_install_records_its_commit_and_tag` (an error: it reads `result["raypoolkit"]`).
`TestManifest` did not fail: its test compares the manifest with `package_provenance()` and the
key set with `RECORDED_DISTRIBUTIONS`, both of which move together.

### (v) `begin()` writes `package_provenance(names=("datastorekit",))`

```diff
diff --git a/RunRegistry/__init__.py b/RunRegistry/__init__.py
index 0228a1a..81091ca 100644
--- a/RunRegistry/__init__.py
+++ b/RunRegistry/__init__.py
@@ -647,7 +647,7 @@ def begin(
         "heartbeat_means": heartbeat_means,
     }
     manifest.update(git_provenance())
-    manifest["packages"] = package_provenance()
+    manifest["packages"] = package_provenance(names=("datastorekit",))
     write_json_atomic(os.path.join(path, "manifest.json"), manifest)
     run = Run(path)
     run._update(
```
Failed: `TestManifest.test_manifest_records_the_installed_packages`, on the key set.

## Observations not acted on

- `RunRegistry/__init__.py`'s module docstring lists no manifest key, so it was not changed.
- `README.md` §0.2's account of the venv (neither package installed) held when this ran.

## State handed to the next prompt

- `RunRegistry.RECORDED_DISTRIBUTIONS`, `RunRegistry.package_provenance()` and the manifest key
  `packages` exist, shaped as README §2.2 says. Every manifest written from this commit on carries
  `packages`; in today's venv, `{"datastorekit": null, "raypoolkit": null}`. Manifests written
  before it lack the key and are read as before.
- The sidecar, `list` and `status.json` are untouched.
- Counts: RunRegistry 286; the other five roots as README §7.
- The index `docs/OPEN_ISSUES.md` stays at 104.
