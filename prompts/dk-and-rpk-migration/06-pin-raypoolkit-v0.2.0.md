# Prompt 06 — pin `raypoolkit` `v0.2.0`

**Campaign:** [`README.md`](README.md) · **Board items:** **Q1**–**Q3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 05 and 07–09, after the review of 04 (`0b6565e`).
**Gate:**
- 05 landed and reviewed, with **DatastoreKit's G2 holding** (board §2);
- `git status` is clean;
- nothing is `running` in the registry, and no Ray process of this repository is up.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One line of `requirements.txt` and an install, measured
closely: 03's prompt with the names changed.
**Needs the network**: the install fetches the tag from GitHub and the build backend from PyPI.
The orchestrator dispatches this prompt with that permission, for the dry run, the install and
`git ls-remote` only.

**Read first:**

1. [`README.md`](README.md): §0.2 ("The venv"), §2.2 (06's charter), §4.2 (rollback) and §5
   (rules 11–13).
2. **[`03-pin-datastorekit-v0.2.2.md`](03-pin-datastorekit-v0.2.2.md) and
   [its log](logs/03-pin-datastorekit-v0.2.2.md).** This prompt is 03 for the other package. 03's
   log holds the dry run, the file comparison script (K3.4) and the outputs yours should match in
   shape.
3. RayPoolKit's `README.md` ("Installing", "Supported versions") and `docs/adoption/README.md` §1,
   and its SGK checklist, `docs/adoption/secondarygwkit.md`, item 1
   (`/Users/ds283/Documents/Code/RayPoolKit`). **Read them; do not edit anything in that
   repository** (README §5 rule 12).
4. RayPoolKit's `pyproject.toml` at `v0.2.0`: `git -C /Users/ds283/Documents/Code/RayPoolKit show
   v0.2.0:pyproject.toml`.

---

## 1. What is wanted

As 03: SGK depends on a tagged release, pinned in `requirements.txt` in a commit of its own, and
the venv is installed from it. Nothing in SGK imports the package until 07.

- **Q1 — the pin.** `requirements.txt` gains exactly this line:
  ```text
  raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0
  ```
  in sorted position, case-insensitive as `pip freeze` orders it: after `ray==2.43.0` (`:75` at
  `0b6565e`) and before `referencing==0.36.2`, so at `:76`. No other line changes.
- **Q2 — the install**, after a dry run:
  1. `./venv/bin/pip install --dry-run --report var/dk-and-rpk-migration-06/dry-run-report.json -r requirements.txt`.
     It must say `Would install raypoolkit-0.2.0` and nothing else, and the report's `install`
     list must hold that one entry. If it would install, upgrade or remove anything else, stop.
  2. `./venv/bin/pip install -r requirements.txt`.

  The rules of 03's K2 hold: never `-e`, a local path, `--upgrade`, `--force-reinstall` or
  `--no-deps`. If pip fails only because it cannot write its cache, add `--no-cache-dir` to both
  commands, and log it.
- **Q3 — the proof that the venv holds the tag.**

---

## 2. What to do

1. **Before.** Save to `var/dk-and-rpk-migration-06/`, which is gitignored:
   - `./venv/bin/pip freeze`. It must equal `var/dk-and-rpk-migration-03/freeze-after.txt`, 03's
     after-install freeze. If it does not, stop: the venv has moved since 03;
   - `./venv/bin/pip check`;
   - `./venv/bin/pip show raypoolkit`, which must find nothing.
2. **Q1**, then **Q2**. Keep pip's full output in the same directory.
3. **Q3, after:**
   1. **`pip freeze`** differs from the saved one by exactly one added line:
      `raypoolkit @ git+https://github.com/ds283/RayPoolKit@eaef0ce60e2fc354bb32f0ad576010e6a83c1976`.
      Show the diff. With the `datastorekit` and `raypoolkit` lines removed from each,
      `pip freeze` equals `requirements.txt`.
   2. **`pip check`** reports no broken requirement.
   3. **`RunRegistry.package_provenance()`**, in a fresh process:
      - `["raypoolkit"]`: version `0.2.0`, URL `https://github.com/ds283/RayPoolKit`, commit
        `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`, requested revision `v0.2.0`,
        `editable: false`;
      - `["datastorekit"]`: unchanged from 03's record (commit `6d8ee9ae…`).

      Show the raw `direct_url.json` beside it.
   4. **The installed files are the tag's.** Run 03's K3.4 comparison script, from 03's log
      verbatim, on the installed `raypoolkit/` against
      `git -C /Users/ds283/Documents/Code/RayPoolKit archive v0.2.0 raypoolkit`, extracted into a
      new directory outside both repositories. At `v0.2.0` the package holds 4 `.py` files outside
      `raypoolkit/tests/`: `RayWorkPool.py`, `__init__.py`, `_timing.py`, `contract.py`. All 4
      are installed byte for byte, nothing under `tests/` is, and nothing else `.py` is.
      - **Non-vacuity.** Run the same comparison against `git archive v0.1.0 raypoolkit`. It must
        report a difference; name the files it reports.
      - `importlib.metadata.files("raypoolkit")` lies wholly under `raypoolkit/` and
        `raypoolkit-0.2.0.dist-info/`. Give `top_level.txt` if there is one.
   5. **The import.** `./venv/bin/python -I -c "import raypoolkit; from raypoolkit import RayWorkPool"`
      succeeds, `raypoolkit.__file__` lies in the venv's `site-packages`, and
      `ray.is_initialized()` is false afterwards. Start no Ray.
   6. **The tags agree.** `git -C /Users/ds283/Documents/Code/RayPoolKit rev-parse v0.2.0` gives the
      tag object `4e7eb153ca9837c6aa6ec51d702b5fa6ccf08fef`, and `v0.2.0^{commit}` gives
      `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`; `git ls-remote --tags
      https://github.com/ds283/RayPoolKit 'v0.2.0*'` gives the same pair.
   7. **Nothing imports it yet.** `git grep` for an `import`/`from raypoolkit` statement outside
      `prompts/` finds nothing.
4. **The suites.** All six roots at 04's counts: Datastore 320 (54 + 103 + 101 + 62), RunRegistry
   286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped).
   Nothing imports the package yet, so nothing may move.

---

## 3. Acceptance

1. **Q1:** `git diff` of `requirements.txt` is the one added line, at `:76`.
2. **Q2–Q3:** the dry run and every check of §2.3 hold, each shown in the log.
3. The six roots give 04's counts.
4. **No deliberate breakage** (README §5 rule 8): 06 adds no code and no test. §2.3.4's
   non-vacuity run is the evidence that the file check can fail.
5. **The records**, in the same commit:
   - this board: the §1 row for 06, items Q1–Q3, and `### 5.6 After prompt 06 (<date>)`, naming
     the rollback: `git revert` plus `./venv/bin/pip uninstall -y raypoolkit` (README §4.2);
   - README §4's "03 and 06 ship the pins": one line saying 06 shipped as named, or what differs;
   - a §3 entry for anything opened, with `docs/OPEN_ISSUES.md` in the same commit.

---

## 4. Stop conditions — stop and ask

- `pip freeze` before the install differs from 03's after-freeze.
- The dry run, or the install, would change, upgrade or add any distribution but `raypoolkit`, or
  `pip check` reports a broken requirement. **Stop before the install.** If pip has already moved
  something, record exactly what, restore it from the saved freeze, and stop.
- The installed commit is not `eaef0ce…`, the installed files differ from the tag's, or the
  install is editable.
- `package_provenance()["datastorekit"]` has moved.
- The install cannot reach GitHub. Do not substitute a local path, a wheel built from the
  checkout, or an editable install.
- Any suite's count or result moves.

---

## 5. What this prompt does not do

- **Files it changes:** `requirements.txt`, the log, this board and README §4's one line. Nothing
  else.
- It imports the package from no SGK file, and deletes nothing. `RayTools/` stays until 07.
- It changes nothing in RayPoolKit or DatastoreKit.

---

## 6. The log

`logs/06-pin-raypoolkit-v0.2.0.md`, using README §5.1. It holds:
- the dry run's output and report summary;
- every check of §2.3, with its command and output, the comparison script's arguments, and the
  non-vacuity run;
- pip's install output, abridged to what it collected, built and installed;
- the six suites beside the dispatch baselines.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
