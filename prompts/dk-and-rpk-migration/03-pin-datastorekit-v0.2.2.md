# Prompt 03 — pin `datastorekit` `v0.2.2`

**Campaign:** [`README.md`](README.md) · **Board items:** **K1**–**K3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 01, 02 and 04, after the user reviewed the plan (`4532188`).
**Gate:**
- 02 landed and reviewed, with **R0's fingerprints equal to `39809dca…`** (board §2);
- `git status` is clean;
- nothing is `running` in the registry.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. One line of `requirements.txt` and an install, measured
closely.
**Needs the network**: the install fetches the tag from GitHub. The orchestrator dispatches this
prompt with that permission.

**Read first:**

1. [`README.md`](README.md): §0.2 ("The venv"), §2.2 (03's charter), §4.2 (rollback) and §5
   (rules 11–13).
2. DatastoreKit's `README.md`, "Installing" and "Supported versions"
   (`/Users/ds283/Documents/Code/DatastoreKit/README.md`), and its `docs/adoption/README.md` §1
   and §3. **Read them; do not edit anything in that repository** (README §5 rule 12).
3. DatastoreKit's `pyproject.toml`: the declared dependencies, and what `packages.find` excludes.
4. 01's log, for `RunRegistry.package_provenance()`.
5. [`measure/out/b510bc9-pip-freeze.txt`](measure/out/b510bc9-pip-freeze.txt).

---

## 1. What is wanted

SGK depends on a tagged release, pinned in `requirements.txt`, in a commit of its own (DatastoreKit
adoption README §1; README §5 rule 13). This prompt adds the pin and installs the venv from it.
Nothing in SGK imports the package until 04.

- **K1 — the pin.** `requirements.txt` gains exactly this line:
  ```text
  datastorekit @ git+https://github.com/ds283/DatastoreKit@v0.2.2
  ```
  It goes in sorted position. The file is `pip freeze`'s order, case-insensitive, so it goes
  between `cycler==0.12.1` and `distlib==0.3.9`. No other line changes: `ray==2.43.0` and
  `SQLAlchemy==2.0.39` stay as they are.
- **K2 — the install.** `./venv/bin/pip install -r requirements.txt`, from the repository root.
  - Never `pip install -e`, and never a local path to the DatastoreKit checkout. The install comes
    from the tag on GitHub, so that it is reproducible from the pin.
  - Never `--upgrade`, `--force-reinstall` or `--no-deps`. The declared dependencies (`ray>=2.43`,
    `sqlalchemy>=2.0.39,<2.1`) are already satisfied, so pip should add one distribution and move
    none. That is K3's first check.
- **K3 — the proof that the venv holds the tag.**

---

## 2. What to do

1. **Before.** Save to `var/dk-and-rpk-migration-03/`, which is gitignored:
   - `./venv/bin/pip freeze`. It must equal `measure/out/b510bc9-pip-freeze.txt`. If it does not,
     stop: the venv has moved since the plan;
   - `./venv/bin/pip check`;
   - `./venv/bin/pip show datastorekit`, which must find nothing.
2. **K1**, then **K2**. Keep pip's full output in the same directory.
3. **K3, after:**
   1. **`pip freeze`** differs from the saved one by exactly one added line:
      `datastorekit @ git+https://github.com/ds283/DatastoreKit@6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`.
      Pip records the commit the tag resolved to. Show the diff.
   2. **`pip check`** reports no broken requirement.
   3. **`RunRegistry.package_provenance()["datastorekit"]`** gives:
      - version `0.2.2`;
      - URL `https://github.com/ds283/DatastoreKit`;
      - commit `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`;
      - requested revision `v0.2.2`;
      - `editable: false`.

      `["raypoolkit"]` is `None`.
   4. **The installed files are the tag's.** Compare the installed `datastorekit/` package (from
      `importlib.util.find_spec("datastorekit").submodule_search_locations`) with
      `git -C /Users/ds283/Documents/Code/DatastoreKit archive v0.2.2 datastorekit`, extracted
      into a new directory outside both repositories.
      - Every `.py` file the archive holds outside `datastorekit/tests/` is installed, byte for
        byte.
      - `datastorekit/tests/` is not installed (adoption README §3).
      - Nothing else `.py` is installed. Ignore `__pycache__`.

      Show the file counts and the result.
   5. **The import.** `./venv/bin/python -c "import datastorekit, datastorekit.SQL.ShardedPool"`
      succeeds with `ray.is_initialized()` false afterwards. Import nothing else, and start no Ray.
   6. **The tags agree.** `git -C <DatastoreKit> rev-parse v0.2.2^{commit}` is `6d8ee9ae…`, and
      `git ls-remote --tags https://github.com/ds283/DatastoreKit v0.2.2` gives tag object
      `c32d0278…`, as README §0.2.
4. **The suites.** All six roots at README §7's counts (and 01's for RunRegistry), Datastore in
   its four groups. Nothing imports the package yet, so nothing may move.

---

## 3. Acceptance

1. **K1:** `git diff` of `requirements.txt` is the one added line, in sorted position.
2. **K2–K3:** every check of §2.3 holds, and each is shown in the log.
3. The six roots give their dispatch counts.
4. **No deliberate breakage** (README §5 rule 8): 03 adds no code and no test. §2.3's checks are
   its proof. Each is a check that a wrong install would fail.
5. **The records**, in the same commit:
   - this board: the §1 row for 03, items K1–K3, and `### 5.3 After prompt 03 (<date>)`. It names
     the rollback: `git revert` plus `./venv/bin/pip uninstall -y datastorekit` (README §4.2);
   - a §3 entry for anything opened, with `docs/OPEN_ISSUES.md` in the same commit.

---

## 4. Stop conditions — stop and ask

- `pip freeze` before the install differs from `b510bc9-pip-freeze.txt`.
- The install would change, upgrade or add any distribution but `datastorekit`, or `pip check`
  reports a broken requirement. **Stop before accepting any such change.** If pip has already
  moved something, record exactly what, restore it from the saved `pip freeze`, and stop.
- The installed commit is not `6d8ee9ae…`, the installed files differ from the tag's, or the
  install is editable.
- The install cannot reach GitHub. Do not substitute a local path, a wheel built from the
  checkout, or an editable install.
- Any suite's count or result moves.

---

## 5. What this prompt does not do

- **Files it changes:** `requirements.txt`, the log and this board. Nothing else.
- It imports the package from no SGK file, and deletes nothing.
- It changes nothing in DatastoreKit.
- It does not pin `raypoolkit`, which is 06.

---

## 6. The log

`logs/03-pin-datastorekit-v0.2.2.md`, using README §5.1. It holds:
- every check of §2.3, with its command and output;
- pip's install output, abridged to what it collected, built and installed;
- the six suites beside the dispatch baselines.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
