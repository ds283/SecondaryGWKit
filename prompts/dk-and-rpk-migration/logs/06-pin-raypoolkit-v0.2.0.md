# Log — prompt 06, pin `raypoolkit` `v0.2.0`

**Prompt:** [`../06-pin-raypoolkit-v0.2.0.md`](../06-pin-raypoolkit-v0.2.0.md) ·
**Board items:** Q1–Q3 · **Date:** 2026-10-11 · **Model:** Sonnet 5.5 ·
**Commit:** this commit · **Result:** landed. Q1–Q3 as the prompt says. The venv gained one
distribution, `raypoolkit` 0.2.0 at `eaef0ce6…`, not editable, its files the tag's. Every suite at its
dispatch count, except the load-sensitive ComputeTargets test, which passed alone three times. No issue
opened or closed.

Dispatched at `0972599` (code of `0c9e2cf`), tree clean, registry 25 runs none `running`, no Ray started
by this prompt. Python 3.12.15, pip 26.2.1. Saves are in `var/dk-and-rpk-migration-06/` (gitignored):
`freeze-before.txt`, `check-before.txt`, `show-before.txt`, `dry-run-output.txt`, `dry-run-report.json`,
`install-output.txt`, `freeze-after.txt`. The two tags' archives were extracted into the session
scratchpad (`agent-06/tag020`, `agent-06/tag010`), outside both repositories.

## What shipped

- **Q1.** `requirements.txt`, one added line at `:76`, between `ray==2.43.0` and `referencing==0.36.2`.
  No other line changed (99 -> 100 lines).
- **Q2.** The venv installed from it, by the dry run and then the prompt's command (below).
- **Q3.** Every check of §2.3 holds (below).
- This log, the board (§1 row, Q1–Q3, §5.6), and one line under README §4's "03 and 06 ship the pins".

```diff
--- a/requirements.txt
+++ b/requirements.txt
@@ -73,6 +73,7 @@ python-dateutil==2.9.0.post0
 pytz==2025.1
 PyYAML==6.0.2
 ray==2.43.0
+raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0
 referencing==0.36.2
 requests==2.32.3
 rich==13.9.4
```

## Deviations from the prompt

1. **README §4's "03 and 06 ship the pins" gains one line** saying 06 shipped as named. The prompt §5
   does not list README; the orchestrator's brief does. **IMPLEMENTATION CHOICE.**

The dry run, `importlib.metadata.files()`, `top_level.txt` and the raw `direct_url.json` are in the
prompt itself this time, so are not deviations. No network or cache failure occurred, so neither the
sandbox override nor `--no-cache-dir` was used. 03's comparison script was run verbatim (lines 139–172 of
03's log, docstring unchanged). No `UNINTENDED DRIFT`.

## Verification performed

### Before (prompt §2.1)

```
$ ./venv/bin/pip freeze > var/dk-and-rpk-migration-06/freeze-before.txt
$ diff freeze-before.txt var/dk-and-rpk-migration-03/freeze-after.txt
(no output: equal)
$ ./venv/bin/pip check
No broken requirements found.
$ ./venv/bin/pip show raypoolkit        # exit 1
WARNING: Package(s) not found: raypoolkit
```

### The dry run (before Q2)

```
$ ./venv/bin/pip install --dry-run --report var/dk-and-rpk-migration-06/dry-run-report.json -r requirements.txt
... "Requirement already satisfied" lines
Collecting datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2 (from -r requirements.txt (line 17))
  Cloning https://github.com/ds283/DatastoreKit (to revision v0.2.2) to <pip temp dir>
  Resolved https://github.com/ds283/DatastoreKit to commit c32d0278d71c5ed543399546908bde8498905e59
  Installing build dependencies / Getting requirements to build wheel / Preparing metadata: done
Collecting raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0 (from -r requirements.txt (line 76))
  Cloning https://github.com/ds283/RayPoolKit (to revision v0.2.0) to <pip temp dir>
  Resolved https://github.com/ds283/RayPoolKit to commit 4e7eb153ca9837c6aa6ec51d702b5fa6ccf08fef
  Installing build dependencies / Getting requirements to build wheel / Preparing metadata: done
Would install raypoolkit-0.2.0
```

Exit 0. The report's `install` list has one entry, `raypoolkit` 0.2.0, with `download_info` `url`
`https://github.com/ds283/RayPoolKit`, `commit_id` `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`,
`requested_revision` `v0.2.0`. Nothing else would change.

### Q2, the install

```
$ ./venv/bin/pip install -r requirements.txt        # exit 0
... "Requirement already satisfied" lines
Collecting datastorekit @ git+... (line 17)
  Resolved https://github.com/ds283/DatastoreKit to commit c32d0278d71c5ed543399546908bde8498905e59
Collecting raypoolkit @ git+... (line 76)
  Resolved https://github.com/ds283/RayPoolKit to commit 4e7eb153ca9837c6aa6ec51d702b5fa6ccf08fef
Building wheels for collected packages: raypoolkit
  Created wheel for raypoolkit: filename=raypoolkit-0.2.0-py3-none-any.whl size=13565 sha256=0459aaf48a3a7e157e73d59086dcbd0deb8da67ef466263a7e0b7c8b411cb001
Successfully built raypoolkit
Installing collected packages: raypoolkit
Successfully installed raypoolkit-0.2.0
```

No flag but `-r`. No `Attempting uninstall` or `Found existing installation`.

### Q3.1, `pip freeze`

```
$ ./venv/bin/pip freeze > freeze-after.txt; diff freeze-before.txt freeze-after.txt
75a76
> raypoolkit @ git+https://github.com/ds283/RayPoolKit@eaef0ce60e2fc354bb32f0ad576010e6a83c1976
```

With the `datastorekit` and `raypoolkit` lines removed from both, `requirements.txt` equals
`freeze-after.txt` (`diff` empty).

### Q3.2, `pip check`

`No broken requirements found.` `pip show raypoolkit`: Version 0.2.0, `Requires: ray`, `Required-by:`
nothing, Location the venv's `site-packages`.

### Q3.3, the record

```
$ ./venv/bin/python -c "import RunRegistry,json;print(json.dumps(RunRegistry.package_provenance(),indent=1))"
{"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit",
  "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false},
 "raypoolkit": {"version": "0.2.0", "url": "https://github.com/ds283/RayPoolKit",
  "commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "requested_revision": "v0.2.0", "editable": false}}
```
(reformatted; the call printed it indented.) `datastorekit` is unchanged from 03's record. The source,
`raypoolkit-0.2.0.dist-info/direct_url.json`:

```
{"url": "https://github.com/ds283/RayPoolKit", "vcs_info": {"commit_id": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "requested_revision": "v0.2.0", "vcs": "git"}}
```

No `dir_info`, so `editable` is false.

### Q3.4, the installed files are the tag's

Extracted: `git -C /Users/ds283/Documents/Code/RayPoolKit archive v0.2.0 raypoolkit | tar -x -C
<scratch>/tag020` (and `v0.1.0` into `<scratch>/tag010`). Script: 03's, verbatim from lines 139–172 of
its log, saved as `compare_tree.py` in the scratchpad. Arguments: `python -I compare_tree.py
/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/raypoolkit
<scratch>/tag020/raypoolkit`.

```
tag .py files: 14 (4 outside tests/, 10 under tests/)
installed .py files: 4
byte-identical: 4; missing: []; differ: []
extra installed .py (incl. any tests/): []
tests/ installed: False False
RESULT: PASS                                  (exit 0)
```

**Non-vacuity**, the same script against `git archive v0.1.0 raypoolkit`:

```
tag .py files: 14 (4 outside tests/, 10 under tests/)
installed .py files: 4
byte-identical: 0; missing: []; differ: ['RayWorkPool.py', '__init__.py', '_timing.py', 'contract.py']
extra installed .py (incl. any tests/): []
tests/ installed: False False
RESULT: FAIL                                  (exit 1)
```

All four files are reported. `importlib.metadata.distribution("raypoolkit").files`: 16 entries, every one
under `raypoolkit/` or `raypoolkit-0.2.0.dist-info/` (outside: `[]`); top-level names
`['raypoolkit', 'raypoolkit-0.2.0.dist-info']`; `top_level.txt` is `raypoolkit`. The dist-info holds
INSTALLER, METADATA, RECORD, REQUESTED, WHEEL, `direct_url.json`, `licenses/LICENSE`, `top_level.txt`.

### Q3.5, the import

```
$ ./venv/bin/python -I -c "import raypoolkit; from raypoolkit import RayWorkPool; import ray; print(raypoolkit.__file__, ray.is_initialized(), RayWorkPool)"
.../venv/lib/python3.12/site-packages/raypoolkit/__init__.py False <class 'raypoolkit.RayWorkPool.RayWorkPool'>
```
(`import ray` was added only to ask the question; `ray.init` was never called.)

### Q3.6, the tags agree

```
$ git -C /Users/ds283/Documents/Code/RayPoolKit rev-parse v0.2.0 'v0.2.0^{commit}'
4e7eb153ca9837c6aa6ec51d702b5fa6ccf08fef
eaef0ce60e2fc354bb32f0ad576010e6a83c1976
$ git ls-remote --tags https://github.com/ds283/RayPoolKit 'v0.2.0*'
4e7eb153ca9837c6aa6ec51d702b5fa6ccf08fef	refs/tags/v0.2.0
eaef0ce60e2fc354bb32f0ad576010e6a83c1976	refs/tags/v0.2.0^{}
```

### Q3.7, nothing imports it

`git grep -n -E '^\s*(import|from)\s+raypoolkit' -- . ':!prompts'` finds nothing (exit 1).

### The suites

README §7's command, foreground, serially, after the install, at `0972599` plus the pin. Load is the
1-minute figure when each started (Datastore groups, 11–13).

| Root | Dispatch baseline | Now | Verdict |
|---|---|---|---|
| AdaptiveLevin | 32 | 32 | OK |
| CosmologyModels | 39 | 39 | OK |
| LiouvilleGreen | 148 (1 skipped) | 148 (1 skipped) | OK |
| RunRegistry | 286 | 286 | OK |
| ComputeTargets | 693 (1 known failure) | 693 | FAILED (1), see below |
| Datastore `[a-f]` | 54 | 54 | OK |
| Datastore `[g-o]` | 103 | 103 | OK |
| Datastore `[p-r]` | 101 | 101 | OK |
| Datastore `[s-z]` | 62 | 62 | OK |

Datastore total 320.

**ComputeTargets** failed one test, `test_tk_wkb_phase.TestCost.test_wall_time_per_object` (`0.0611 not
less than or equal to 0.06`, load ~13), README §7's load-sensitive test, which also failed at dispatch.
Re-run alone three times at load ~12 it passed each time (0.35 s). It touches nothing 06 changed; the
other 692 passed.

## The deliberate-breakage record

None: 06 adds no code and no test (prompt §3.4). The non-vacuity run against `v0.1.0` is the evidence
that the file check can fail.

## Observations not acted on

- **pip cloned DatastoreKit at `v0.2.2` again** in both the dry run and the install, and then skipped it,
  as the orchestrator's brief §0 expected. `package_provenance()["datastorekit"]` is unchanged. (The
  review compares its installed files and mtimes; this log did not.)
- **pip's "Resolved … to commit 4e7eb153…" line prints the tag object's id**, not the commit's; the
  installed record holds `eaef0ce6…`. Only pip's wording.
- **`from raypoolkit import RayWorkPool` gives the class**, not the module (`__init__.py` rebinds it), as
  the brief says. Not a defect.
- **Importing the package may write `__pycache__`** under `site-packages/raypoolkit/`; K3.4's script
  ignores it.

## State handed to the next prompt

- **The venv** holds `raypoolkit` 0.2.0 at `eaef0ce6…` and nothing else new. `package_provenance()` now
  records both packages. Rollback of this prompt is `git revert` plus `./venv/bin/pip uninstall -y
  raypoolkit` (README §4.2).
- **07** may import `raypoolkit`. Nothing in SGK does yet.
- Nothing running; no Ray started; nothing under `var/runs/` added. `var/dk-and-rpk-migration-06/` holds
  the saves listed above.
