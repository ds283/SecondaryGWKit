# Log — prompt 03, pin `datastorekit` `v0.2.2`

**Prompt:** [`../03-pin-datastorekit-v0.2.2.md`](../03-pin-datastorekit-v0.2.2.md) ·
**Board items:** K1–K3 · **Date:** 2026-10-11 · **Model:** Sonnet 5.5 ·
**Commit:** `9ec603a` · **Result:** landed. K1–K3 as the prompt says. The venv gained one
distribution, `datastorekit` 0.2.2 at `6d8ee9ae…`, not editable, its files the tag's. Every suite at
its dispatch count, except the load-sensitive ComputeTargets test, which passed alone three times. No
issue opened or closed.

Dispatched at `1321e0e` (code of `f54f1c5`), tree clean, registry 21 runs none `running`, no Ray
started by this prompt. Python 3.12.15, pip 26.2.1. Saves are in `var/dk-and-rpk-migration-03/`
(gitignored): `freeze-before.txt`, `check-before.txt`, `show-before.txt`, `dry-run-output.txt`,
`dry-run-report.json`, `install-output.txt`, `freeze-after.txt`. The tag's archive was extracted into
the session scratchpad, outside both repositories.

## What shipped

- **K1.** `requirements.txt`, one added line at `:17`, between `cycler==0.12.1` and
  `distlib==0.3.9`. No other line changed (98 → 99 lines).
- **K2.** The venv installed from it, by the dry run and then the prompt's command (below).
- **K3.** Every check of §2.3 holds (below).
- This log, the board (§1 row, K1–K3, §2 unchanged, `### 5.3`), and one line under README §4's "03
  and 06 ship the pins".

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -14,6 +14,7 @@ click==8.1.8
 colorful==0.5.6
 contourpy==1.3.1
 cycler==0.12.1
+datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2
 distlib==0.3.9
 filelock==3.18.0
 fonttools==4.56.0
```

## Deviations from the prompt

1. **A `pip install --dry-run` came before K2**, as the orchestrator's note directs, so that
   "stop before accepting any such change" has a point at which to stop. **IMPLEMENTATION CHOICE.**
2. **README §4's "03 and 06 ship the pins" gains one line** saying 03 shipped as named. The prompt
   §5 does not list README; the orchestrator's note does. **IMPLEMENTATION CHOICE.**
3. **K3.4 also lists `importlib.metadata.files("datastorekit")` and `top_level.txt`**, and the raw
   `direct_url.json` is shown beside `package_provenance()`'s reading, as the note directs.
   **IMPLEMENTATION CHOICE.**

No network or cache failure occurred, so neither the sandbox override nor `--no-cache-dir` was used.
No `UNINTENDED DRIFT`.

## Verification performed

### Before (prompt §2.1)

```
$ ./venv/bin/pip freeze > var/dk-and-rpk-migration-03/freeze-before.txt
$ diff var/dk-and-rpk-migration-03/freeze-before.txt prompts/dk-and-rpk-migration/measure/out/b510bc9-pip-freeze.txt
(no output: equal)
$ ./venv/bin/pip check
No broken requirements found.
$ ./venv/bin/pip show datastorekit        # exit 1
WARNING: Package(s) not found: datastorekit
```

### The dry run (before K2)

```
$ ./venv/bin/pip install --dry-run --report var/dk-and-rpk-migration-03/dry-run-report.json -r requirements.txt
... (97 "Requirement already satisfied" lines)
Collecting datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2 (from -r requirements.txt (line 17))
  Running command git clone --filter=blob:none --quiet https://github.com/ds283/DatastoreKit ...
  Running command git checkout -q c32d0278d71c5ed543399546908bde8498905e59
  Resolved https://github.com/ds283/DatastoreKit to commit c32d0278d71c5ed543399546908bde8498905e59
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): finished with status 'done'
Would install datastorekit-0.2.2
```

The report's `install` list has one entry, `datastorekit` 0.2.2, with `download_info` `url`
`https://github.com/ds283/DatastoreKit`, `commit_id` `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`,
`requested_revision` `v0.2.2`. Nothing else would change.

### K2, the install

```
$ ./venv/bin/pip install -r requirements.txt        # exit 0
... (97 "Requirement already satisfied" lines)
Collecting datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2 (from -r requirements.txt (line 17))
  Resolved https://github.com/ds283/DatastoreKit to commit c32d0278d71c5ed543399546908bde8498905e59
  Installing build dependencies / Getting requirements to build wheel / Preparing metadata: done
Building wheels for collected packages: datastorekit
  Created wheel for datastorekit: filename=datastorekit-0.2.2-py3-none-any.whl size=98811 sha256=65267beabc3e20fa9cd25bc4601028dc1a462bb6b6404487de58e985aabe5bc8
Successfully built datastorekit
Installing collected packages: datastorekit
Successfully installed datastorekit-0.2.2
```

No flag but `-r`. No uninstall, upgrade or other distribution touched.

### K3.1, `pip freeze`

```
$ ./venv/bin/pip freeze > freeze-after.txt; diff freeze-before.txt freeze-after.txt
16a17
> datastorekit @ git+https://github.com/ds283/DatastoreKit@6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6
```

### K3.2, `pip check`

`No broken requirements found.` `pip show datastorekit` gives Version 0.2.2, `Requires: ray,
sqlalchemy`, `Required-by:` nothing, Location the venv's `site-packages`.

### K3.3, the record

```
$ ./venv/bin/python -c "import RunRegistry,json;print(json.dumps(RunRegistry.package_provenance(),indent=1))"
{"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit",
  "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2",
  "editable": false}, "raypoolkit": null}
```
(reformatted; the call printed it indented.) The source, `direct_url.json`:

```
{"url": "https://github.com/ds283/DatastoreKit", "vcs_info": {"commit_id": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "vcs": "git"}}
```

No `dir_info`, so `editable` is false.

### K3.4, the installed files are the tag's

Extracted: `git -C /Users/ds283/Documents/Code/DatastoreKit archive v0.2.2 datastorekit | tar -x -C
<scratch>/tag`. The installed directory is `list(importlib.util.find_spec("datastorekit").submodule_search_locations)[0]`
(`…/venv/lib/python3.12/site-packages/datastorekit`). The script, verbatim, taking the two
directories as arguments so it can be rerun against another tag (`python -I compare_tree.py
<installed> <extracted>`):

```python
"""Usage: python -I compare_tree.py <installed datastorekit dir> <tag's extracted datastorekit dir>

Checks K3.4: every .py outside tests/ in the tag is installed byte for byte, tests/ is not
installed, and no other .py is installed. __pycache__ ignored. Exit 1 on any failure."""
import sys
from pathlib import Path

inst, tag = Path(sys.argv[1]), Path(sys.argv[2])


def pys(root):
    return {
        p.relative_to(root).as_posix(): p
        for p in root.rglob("*.py")
        if "__pycache__" not in p.parts
    }


I, T = pys(inst), pys(tag)
T_main = {k: v for k, v in T.items() if not k.startswith("tests/")}
T_tests = {k for k in T if k.startswith("tests/")}
ok = True
print(f"tag .py files: {len(T)} ({len(T_main)} outside tests/, {len(T_tests)} under tests/)")
print(f"installed .py files: {len(I)}")
missing = sorted(set(T_main) - set(I))
extra = sorted(set(I) - set(T_main))
differ = sorted(k for k in T_main if k in I and I[k].read_bytes() != T_main[k].read_bytes())
identical = [k for k in T_main if k in I and k not in differ]
print(f"byte-identical: {len(identical)}; missing: {missing}; differ: {differ}")
print(f"extra installed .py (incl. any tests/): {extra}")
print("tests/ installed:", any(k.startswith("tests/") for k in I), (inst / "tests").exists())
ok = not (missing or extra or differ)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
```

Output:

```
tag .py files: 67 (20 outside tests/, 47 under tests/)
installed .py files: 20
byte-identical: 20; missing: []; differ: []
extra installed .py (incl. any tests/): []
tests/ installed: False False
RESULT: PASS
```

The tag's only non-`.py` files are three `.json` under `tests/data/`; the install holds no non-`.py`
file in the package. The distribution's ownership, from `importlib.metadata.distribution("datastorekit").files`:
48 entries, every one under `datastorekit/` or `datastorekit-0.2.2.dist-info/` (outside: `[]`);
top-level names `['datastorekit', 'datastorekit-0.2.2.dist-info']`; `top_level.txt` is `datastorekit`.
The dist-info holds INSTALLER, METADATA, RECORD, REQUESTED, WHEEL, `direct_url.json`,
`licenses/LICENSE`, `top_level.txt`.

### K3.5, the import

```
$ ./venv/bin/python -c "import datastorekit, datastorekit.SQL.ShardedPool, ray; print(datastorekit.__file__, ray.is_initialized())"
…/venv/lib/python3.12/site-packages/datastorekit/__init__.py False
```
(`import ray` was added only to ask the question; `ray.init` was never called.)

### K3.6, the tags agree

```
$ git -C /Users/ds283/Documents/Code/DatastoreKit rev-parse v0.2.2 'v0.2.2^{commit}'
c32d0278d71c5ed543399546908bde8498905e59
6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6
$ git ls-remote --tags https://github.com/ds283/DatastoreKit 'v0.2.2*'
c32d0278d71c5ed543399546908bde8498905e59	refs/tags/v0.2.2
6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6	refs/tags/v0.2.2^{}
```

### Inert: nothing in SGK imports it

`git grep -il datastorekit -- . ':!prompts' ':!docs'` gives `CLAUDE.md`, `RunRegistry/__init__.py`,
`RunRegistry/tests/test_run_registry.py`, `requirements.txt`: the freeze section, the
`RECORDED_DISTRIBUTIONS` tuple, strings in the tests, and the pin. No import.

### The suites

README §7's command, in the foreground, serially, at `1321e0e` plus the pin; outputs in the
scratchpad. Load is the 1-minute figure when each started.

| Root | Dispatch baseline | Now | Load | Verdict |
|---|---|---|---|---|
| AdaptiveLevin | 32 | 32 | 5.9 | OK |
| CosmologyModels | 39 | 39 | 5.7 | OK |
| LiouvilleGreen | 148 (1 skipped) | 148 (1 skipped) | 5.7 | OK |
| RunRegistry | 286 | 286 | 5.4 | OK |
| ComputeTargets | 693 | 693 | 5.4 | FAILED (1), see below |
| Datastore `[a-f]` | 163 | 163 | 8.1 | OK |
| Datastore `[g-o]` | 185 | 185 | 8.1 | OK |
| Datastore `[p-r]` | 200 | 200 | 46.9 | OK |
| Datastore `[s-z]` | 191 | 191 | 17.1 | OK |

Datastore total 739.

**ComputeTargets** failed one test, `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
(`0.0612 not less than or equal to 0.06`), README §7's first load-sensitive test. Re-run alone three
times at load 8 it passed each time (0.42 s). It touches nothing 03 changed. Not a failure of this
prompt; the other 692 passed.

## The deliberate-breakage record

None: 03 adds no code and no test (prompt §3.4). The checks of §2.3 are the proof. K3.4 is not
vacuous by construction: its three sets (missing, differing, extra) are each printed, and the tag it
reads has 20 files outside `tests/`, so an install that dropped, altered or added a `.py` would show.

## Observations not acted on

- **pip's "Resolved … to commit c32d0278…" line prints the tag object's id**, not the commit's. The
  installed record (`direct_url.json`, `pip freeze`) correctly holds the commit `6d8ee9ae…`. Only
  pip's log wording; nothing to open.
- **DatastoreKit's adoption README §1 still shows `@v0.2.0`**, as the orchestrator's note says; its
  `v0.2.2` addendum supersedes it. That repository's record; nothing opened.
- **Importing the package wrote `__pycache__` under `site-packages/datastorekit/`.** Ignored by
  K3.4, as the prompt says.

## State handed to the next prompt

- **The venv** holds `datastorekit` 0.2.2 at `6d8ee9ae…` and nothing else new. Every manifest written
  from now on records it through `package_provenance()`. `["raypoolkit"]` is `None`.
- **04** may import `datastorekit`. The package is inert until it does. Rollback of this prompt is
  `git revert` plus `./venv/bin/pip uninstall -y datastorekit` (README §4.2).
- Nothing running; no Ray started; nothing under `var/runs/` added. `var/dk-and-rpk-migration-03/` holds
  the saves listed above.
