# Orchestrator — prompt 06, pin `raypoolkit` `v0.2.0`

Read [`../README.md`](../README.md) first: §0.2 ("The venv"), §2.2 (06's charter), §4, §4.2, §5
(rules 11–13) and §7. Then read [`prompt-03.md`](prompt-03.md), the note for the other pin, whose
conventions, dispatch and review this note carries over except where §0 below says otherwise, and
this board's *Orchestrator review of prompt 03* and *of prompt 05*.

**You do not write code.** You may:
- read the venv's installed files and metadata, and compare them with the tag's;
- run the suites;
- run in-process checks from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

You install nothing and uninstall nothing. The venv is changed once, by the agent's
`pip install`.

**The prompt:** [`06-pin-raypoolkit-v0.2.0.md`](../06-pin-raypoolkit-v0.2.0.md)
**Board items:** Q1–Q3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 05 landed (`f3f0879`) and was reviewed (`2d3429c`), with DatastoreKit's G2 holding (board §2).
  The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`, and in particular `requirements.txt`,
    `RunRegistry/`, `Datastore/`, `RayTools/`, `config/`, `main.py` and `docs/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- Nothing is `running` in the registry, and no Ray process of this repository is up.
- **One prompt at a time in this checkout.** The venv is shared (README §4.2): nothing else may
  run from it while pip changes it.
- **The network.** The install fetches both packages' tags from GitHub (§0) and the build backend
  from PyPI. The user's dispatch of this prompt authorises exactly that: `pip install`, its
  `--dry-run`, and `git ls-remote`. No other download.

## 0. What makes this prompt unusual

**It is 03 for the other package, and 03 went cleanly.** The prompt already carries what 03's note
added to 03's prompt: the dry run before the install, `importlib.metadata.files()` and
`top_level.txt`, the raw `direct_url.json`, and README §4's one line. So none of those is a
deviation this time. The review has 03's two halves:
- **the venv holds the tag and nothing else moved**: one distribution added, at commit
  `eaef0ce…`, not editable, its files the tag's files, every other distribution at its version;
- **SGK does not see it yet**: nothing imports `raypoolkit`, and the six roots give their counts.
  `RayTools/` still runs every pool until 07.

**What is new: the venv now holds a VCS-installed package that `requirements.txt` names, and pip
will look at it again.** `pip install -r requirements.txt` reads the `datastorekit @ git+…@v0.2.2`
line as well as the new one. pip 26.2.1 has no shortcut for an installed direct-URL distribution
(`resolution/resolvelib/factory.py` builds an `AlreadyInstalledCandidate` only for named
requirements). So both the dry run and the install will **clone DatastoreKit at `v0.2.2` and
prepare its metadata**, print `Collecting datastorekit @ git+…`, and then skip it: the resolver
(`resolver.py:139-167`) reinstalls only on `--force-reinstall`, a different version, an editable
side, or a `file://` link, and none applies. That is expected, and is not a stop. What would be a
stop:
- `Would install` naming anything but `raypoolkit-0.2.0`, or the report's `install` list holding
  any entry but `raypoolkit` (the report is built from `requirements_to_install`, so a skipped
  `datastorekit` is absent from it);
- the install printing `Attempting uninstall`, `Found existing installation`, or
  `Successfully installed` with anything but `raypoolkit-0.2.0`;
- `package_provenance()["datastorekit"]` moving.

The agent logs the clone of DatastoreKit as an observation, citing this note. The review checks
independently that `datastorekit`'s installed files and `dist-info` are byte-identical, with the
same mtimes, before and after (§3 check 3).

**pip's log will name the tag object, not the commit**, as at 03: `Resolved
https://github.com/ds283/RayPoolKit to commit 4e7eb153…`. That is the annotated tag's id; the
checkout lands on its commit, and `direct_url.json` and `pip freeze` record `eaef0ce…`. Not a
defect.

**`from raypoolkit import RayWorkPool` gives the class, not the module.** The package's
`__init__.py:16` rebinds the name. So `raypoolkit.RayWorkPool` is the class after the import; the
module is `sys.modules["raypoolkit.RayWorkPool"]`. Not a defect, and it is what 07 relies on.

**No correction to the prompt.** The orchestrator checked every fact it states against the tree
at `2d3429c` and against RayPoolKit; each holds as written.

**What moves on purpose:**
- `requirements.txt` gains one line, at `:76`. No other line changes (99 → 100 lines).
- The venv gains `raypoolkit` 0.2.0. Nothing else in it moves; `datastorekit` is untouched.
- The log, this board (§1 row, Q1–Q3, `### 5.6 After prompt 06`), and README §4's one line.
- No issue opens or closes. The index stays at 108.

**The facts, checked by the orchestrator on 2026-10-11 at `2d3429c`.** The code outside `prompts/`
and `docs/` is `0c9e2cf`'s (`git diff --stat 0c9e2cf 2d3429c -- . ':!prompts' ':!docs'` is empty),
and `docs/` has gained only 05's records. Pass them on.
- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `RAY_ADDRESS` unset; `pgrep -lf
  'gcs_server|raylet|ray::'` finds nothing, and no process command line holds
  `SecondaryGWKit/venv`. Load 9–24 during the checks. 73 GB free on `var/`'s volume.
- **The registry.** `python -m RunRegistry list` shows 25 runs: 23 finished and 2 `unknown` (the
  pre-registry `a3-pilot` pair), none `running`.
- **`requirements.txt`** has 99 lines and ends with a newline. `ray==2.43.0` is at `:75` and
  `referencing==0.36.2` at `:76`, so the pin becomes the new `:76`. With the `datastorekit` line
  removed, it equals the venv's `pip freeze` with that line removed.
- **The venv.** Python 3.12.15, pip 26.2.1.
  - `pip freeze` equals `var/dk-and-rpk-migration-03/freeze-after.txt` (03's after-freeze).
  - `pip show raypoolkit` finds nothing; `importlib.util.find_spec("raypoolkit")` is `None`;
    `pip check` reports no broken requirement.
  - `site-packages` holds no `.egg-link` and no `.pth` but `distutils-precedence.pth`, and
    `PYTHONPATH` is empty. So nothing can make `import raypoolkit` find the RayPoolKit checkout.
  - `package_provenance()` gives, keys sorted:
    `{"datastorekit": {"commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "editable": false,
    "requested_revision": "v0.2.2", "url": "https://github.com/ds283/DatastoreKit", "version":
    "0.2.2"}, "raypoolkit": null}`. `datastorekit`'s raw `direct_url.json` agrees.
- **The tag**, local and remote, agree:
  - `git -C /Users/ds283/Documents/Code/RayPoolKit rev-parse v0.2.0` gives the annotated tag
    object `4e7eb153ca9837c6aa6ec51d702b5fa6ccf08fef`, and `v0.2.0^{commit}` gives
    `eaef0ce60e2fc354bb32f0ad576010e6a83c1976`;
  - `git ls-remote --tags https://github.com/ds283/RayPoolKit 'v0.*'` gives `4e7eb153…` for
    `refs/tags/v0.2.0` and `eaef0ce6…` for `^{}` (and `0a9cdb2b…`/`4e90a8aa…` for `v0.1.0`).

  RayPoolKit's checkout is clean on `main`, ten commits past the tag, none touching `raypoolkit/`
  or `pyproject.toml`. That is irrelevant: the pin names the tag.
- **`pyproject.toml` at the tag:**
  - the build requires `setuptools>=64`, fetched into an isolated build environment, not into
    the venv;
  - `requires-python = ">=3.12"`, and the one dependency `ray>=2.43`, which `ray==2.43.0`
    satisfies (the README's tested low end: Python 3.12.15, Ray 2.43.0);
  - `include-package-data = false`; `packages.find` includes `raypoolkit*` and excludes
    `raypoolkit.tests` and `raypoolkit.tests.*`.
- **The tag's package** (`git archive v0.2.0 raypoolkit`, extracted in the scratchpad):
  - 14 `.py` files: 10 under `raypoolkit/tests/` and **4** outside it: `RayWorkPool.py`,
    `__init__.py`, `_timing.py`, `contract.py`;
  - no file that is not `.py`; two directories, `raypoolkit` and `raypoolkit/tests`.

  So the install should hold exactly those 4 `.py` files, and nothing under `tests/`.
- **Non-vacuity, in advance.** `v0.1.0` has the same four non-test files, and **all four differ**
  from `v0.2.0`'s byte for byte (`git diff --name-status v0.1.0 v0.2.0` gives `M` for each). So
  03's script against `git archive v0.1.0 raypoolkit` must exit 1 with
  `differ: ['RayWorkPool.py', '__init__.py', '_timing.py', 'contract.py']`, none missing and none
  extra.
- **03's comparison script is package-neutral.** It takes the two directories as arguments and
  names no package in its code; only its docstring says `datastorekit`. Run it verbatim from 03's
  log (`logs/03-pin-datastorekit-v0.2.2.md`, K3.4), on `…/site-packages/raypoolkit` and the
  extracted `…/raypoolkit`.
- **What the package imports.** `__init__.py` imports `RayWorkPool` and `DatastorePool` from its
  own modules; `RayWorkPool.py` imports `time`, `datetime`, `typing`, `ray`, and its own
  `contract` and `_timing`; `contract.py` imports `typing` and `ray.ObjectRef`. Nothing calls
  `ray.init()` at import. So the import check needs nothing the venv lacks and starts no Ray.
- **RayPoolKit's documents agree with the prompt.** Its `README.md` "Installing" (`:79`), its
  adoption README §1 (`:35`) and its SGK checklist item 1 (`:37`) each give
  `raypoolkit @ git+https://github.com/ds283/RayPoolKit@v0.2.0`. Unlike DatastoreKit's at 03,
  there is no addendum to reconcile.
- **Nothing in SGK imports the package.** `git grep -n -E '^\s*(import|from)\s+raypoolkit' -- .
  ':!prompts'` is empty. `raypoolkit` is named outside `prompts/` and `docs/` only in
  `CLAUDE.md:71-72`, `RunRegistry.RECORDED_DISTRIBUTIONS` (`RunRegistry/__init__.py:119`), and
  strings in `RunRegistry/tests/test_run_registry.py`.
- **The RunRegistry tests are silent on the venv.** `TestPackageProvenance` stands in
  `importlib.metadata.distribution`, so its `"raypoolkit": None` assertions (`:326`, `:351`) are
  about stand-ins. `test_manifest_records_the_installed_packages` (`:276`) compares the manifest
  with `package_provenance()` and asserts no value; its comment says 06 must leave it passing.
  So RunRegistry stays at 286.
- **Expected counts after 06:** the dispatch baselines (§1 step 5), unchanged.
- **The index:** 108, unchanged.

**What the review exists to establish.**
- **(E1) The pin.** `requirements.txt` differs by the one line, in sorted position at `:76`.
- **(E2) One distribution added.** `pip freeze` differs from 03's after-freeze by exactly
  `raypoolkit @ git+https://github.com/ds283/RayPoolKit@eaef0ce60e2fc354bb32f0ad576010e6a83c1976`,
  and `pip check` is clean.
- **(E3) The record.** `direct_url.json` and `package_provenance()["raypoolkit"]` give version
  `0.2.0`, URL `https://github.com/ds283/RayPoolKit`, commit `eaef0ce…`, requested revision
  `v0.2.0`, `editable: false`.
- **(E4) The files are the tag's.** The 4 `.py` files byte for byte, no `tests/`, nothing else
  `.py`, and the distribution owns nothing outside `raypoolkit/` and its `dist-info`. The
  comparison reports all four against `v0.1.0`.
- **(E5) `datastorekit` did not move.** Its record, its installed files and its `dist-info` are as
  before, though pip cloned it again.
- **(E6) It is inert.** It imports without starting Ray, and nothing in SGK imports it. The six
  roots give their counts.
- **(E7) The records.** The log shows every check with its command and output. The board has the
  §1 row, Q1–Q3 and §5.6, which names the rollback. README §4 has the one line.

**Conventions.** Those of [`prompt-03.md`](prompt-03.md) §0 bind the agent, with the package's
names changed:
- **The venv.** The agent runs exactly two `pip install` commands: the prompt's dry run, and its
  install. It never uses `-e`, a local path, `--upgrade`, `--force-reinstall` or `--no-deps`. If
  pip fails only because it cannot write its cache, it may add `--no-cache-dir` to both commands,
  and logs it. If the network is refused only by the tool sandbox, it may rerun exactly those two
  commands and `git ls-remote` with the sandbox override, and logs it. Any other flag is a stop.
  It runs no `pip uninstall` unless the prompt's §4 restore applies, and then it stops.
- **Scratch.** The saves the prompt names go under `var/dk-and-rpk-migration-06/` (gitignored):
  `freeze-before.txt`, `check-before.txt`, `show-before.txt`, `dry-run-output.txt`,
  `dry-run-report.json`, `install-output.txt`, `freeze-after.txt`, 03's names. The tags' archives
  are extracted into the agent's own subdirectory of the session scratchpad, outside both
  repositories, never under `/tmp`.
- **The comparison.** 03's script, verbatim, with the paths it took as arguments given in the log.
  If the agent changes even its docstring, the changed text goes in the log and is logged as a
  deviation.
- **Ray.** The import check starts no Ray, and the agent starts none. Any cluster of another
  project is left alone.
- **RayPoolKit and DatastoreKit.** The agent reads RayPoolKit and runs `git rev-parse`, `git show`,
  `git archive` and `git ls-remote` against it. It changes, fetches into, checks out and commits
  nothing in either repository (README §5 rule 12).
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §7's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each in the foreground, serially, after the install. Datastore in the four
  groups. Under load, the four tests of README §7 are known to fail: re-run a failure alone before
  calling it real. `test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed in the root run
  and three times alone at dispatch (§1 step 5), as at 05's dispatch; it measures no code 06
  touches.
- **No breakage diffs.** 06 adds no code and no test (prompt §3.4). The non-vacuity run against
  `v0.1.0` is the evidence that the file check can fail.

## 1. Before you dispatch

Done on 2026-10-11 at `2d3429c`, before this note was committed. Repeat steps 1–4 at the
dispatch `HEAD` if anything lands in between.

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 2d3429c..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** `pgrep -lf 'gcs_server|raylet|ray::'`, and `ps -axo pid,ppid,command` for lines
   holding `SecondaryGWKit/venv`. Nothing of this repository.
4. **The venv.** `pip show raypoolkit` finds nothing, `pip check` is clean, `pip freeze` equals
   03's after-freeze, and `package_provenance()` is §0's record.
   - **Snapshot `datastorekit`** for E5: for every file under `site-packages/datastorekit/` and
     `datastorekit-0.2.2.dist-info/`, ignoring `__pycache__`, its SHA-256 and mtime; and the
     listing of `site-packages`. Taken at `2d3429c`: 28 files, 206 entries, in the session
     scratchpad's `o6/dk-snapshot-before.txt` and `o6/sp-listing-before.txt`.
5. **Baselines**, at `2d3429c`, serially, in the foreground:

   | Root | Tests | Result | Load |
   |---|---|---|---|
   | AdaptiveLevin | 32 | OK | 18 |
   | CosmologyModels | 39 | OK | 18 |
   | LiouvilleGreen | 148 | OK (1 skipped) | 18 |
   | RunRegistry | 286 | OK | 18 |
   | ComputeTargets | 693 | 1 failure: `test_wall_time_per_object` (0.077 s against 0.06 s); alone, failed three times (0.061–0.073 s) | 9–22 |
   | Datastore | 320 = 54 + 103 + 101 + 62 | OK | 22–24 |

   Every count is 04's. The one failure is README §7's wall-clock test under a load of ~20, as at
   05's dispatch (load ~15, five failures alone).
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 108.
7. **The network.** `git ls-remote` against RayPoolKit answers, and `https://pypi.org/simple/`
   gives 200.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- 03's log, which holds the script and the outputs its own should match in shape;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/06-pin-raypoolkit-v0.2.0.md`, in README §5.1's
  form, with the prompt's §6 additions.
- **The files it may add or change:** `requirements.txt`, the log, this board, and README §4's one
  line. In particular, not `docs/OPEN_ISSUES.md` (unless it opens an issue), `orchestrator/`,
  `measure/`, any `.py` file, or anything in RayPoolKit or DatastoreKit.
- **The dry run comes before the install**, and its result is a stop condition like the install's.
  pip cloning DatastoreKit and skipping it is expected (§0); pip installing, upgrading or
  uninstalling it is a stop.
- **The network is authorised** for the dry run, the install and `git ls-remote`, and for nothing
  else.
- **Stop and ask** on any of the prompt's §4 conditions. A suite that moves is a stop, not
  something to fix; the known wall-clock test failing under load, and passing or failing alone
  as at dispatch, is not a moved count.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — eleven checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only `requirements.txt` and
   `prompts/dk-and-rpk-migration/`.
   - In `prompts/dk-and-rpk-migration/`, only the log, `IMPLEMENTATION_STATE.md` and `README.md`
     move, and in `README.md` only §4's one line.
   - `git diff HEAD~1 HEAD -- requirements.txt` is one added line, at `:76`, between `ray` and
     `referencing`, spelled exactly as the prompt's Q1.
2. **E2, independently.** `./venv/bin/pip freeze` against
   `var/dk-and-rpk-migration-03/freeze-after.txt` is exactly the one added line, and `pip check`
   is clean. `pip freeze` with the `datastorekit` and `raypoolkit` lines removed equals
   `requirements.txt` with both removed. `pip show raypoolkit` gives `Location` in this venv's
   `site-packages`, `Requires: ray`, and no `Editable project location`.
3. **E5, `datastorekit` untouched.** Retake §1 step 4's snapshot and diff it with the saved one:
   every SHA-256 and mtime equal. The `site-packages` listing gains exactly `raypoolkit` and
   `raypoolkit-0.2.0.dist-info`. `package_provenance()["datastorekit"]` is §0's record. The log's
   dry-run and install outputs show `datastorekit` collected and not installed, and the report's
   `install` list holds `raypoolkit` alone.
4. **E3, from the source.** Read the installed `raypoolkit-0.2.0.dist-info/direct_url.json` raw.
   Its `url`, `vcs_info.vcs`, `commit_id` and `requested_revision` are as E3 says, and it has no
   `dir_info`. `RunRegistry.package_provenance()` in a fresh process gives the five fields.
5. **E4, the files, and that the comparison can fail.**
   - Rerun 03's script, from 03's log (and check that the agent's copy in its log equals it),
     against a fresh extraction of `git archive v0.2.0 raypoolkit`. It reports 4 byte-identical,
     none missing, none differing, none extra, no `tests/`.
   - Check it independently: sha256 every `.py` under the installed `raypoolkit/`, ignoring
     `__pycache__`, against the archive's 4.
   - **Non-vacuity.** Rerun it against `git archive v0.1.0 raypoolkit`. It must exit 1 reporting
     all four files as differing (§0), and the agent's log must name the same four.
   - `importlib.metadata.files("raypoolkit")` lies wholly under `raypoolkit/` and
     `raypoolkit-0.2.0.dist-info/`; `top_level.txt`, if present, is `raypoolkit` alone.
6. **E6, the import.** In a fresh `./venv/bin/python -I` from the repository root:
   `import raypoolkit; from raypoolkit import RayWorkPool`. `raypoolkit.__file__` lies in the
   venv's `site-packages`, `RayWorkPool` is a class, and `ray.is_initialized()` is false. Then
   `pgrep` shows no Ray process.
7. **The tags.** Local `rev-parse` and remote `ls-remote` as §0, and as the log shows.
   RayPoolKit's checkout is clean on `main` at `c9b0e11`, as before.
8. **Nothing imports it.** `git grep -n -E '^\s*(import|from)\s+raypoolkit' -- . ':!prompts'` is
   empty.
9. **The registry and `var/`.** `RunRegistry list` shows the same 25 runs, none `running`.
   `var/dk-and-rpk-migration-06/` holds the before-saves, the dry-run output and report, pip's
   output and the after-freeze; its `freeze-before.txt` equals 03's after-freeze.
10. **The suites.** Run all six roots, serially, at `HEAD`. Every count is §1 step 5's. There is
    no Python change, so `black` has nothing to check.
11. **E7, the records.**
    - The log has every §2.3 check with its command and output, pip's output abridged to what it
      collected, built and installed, the dry run and its report, the comparison's arguments and
      both runs, the six suites beside the baselines, and the deviations classified.
    - The board has the §1 row (✅, "this commit", the log), Q1–Q3 ✅, and
      `### 5.6 After prompt 06` naming the rollback: `git revert` plus
      `./venv/bin/pip uninstall -y raypoolkit`.
    - README §4 has the one line under "03 and 06 ship the pins".
    - No §3 entry, and `docs/OPEN_ISSUES.md` untouched at 108, unless the agent opened one, with
      the index row, count and date in the same commit.

## 4. After the review

- **Record it on this board** as a §5.6 *Orchestrator review of prompt 06* paragraph: what each
  check found, the non-vacuity result, `datastorekit`'s snapshot, and anything handed on to 07.
  Fix small residue in a follow-up commit of its own. That includes:
  - "this commit" → the SHA, on the board and in the log's header;
  - `orchestrator/prompt-06.md` added to the board's list of notes, marked "used for 06".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - the pin and where it sits;
  - what the venv gained, and that nothing else in it moved, `datastorekit` included;
  - the provenance record as `package_provenance()` now reads it;
  - the file comparison, and its non-vacuity;
  - the counts of the six roots.
- **Then 07**, the swap. From now on every manifest records `raypoolkit` at `eaef0ce…` beside
  `datastorekit` at `6d8ee9ae…`. 07's note is written after this review, and must re-measure
  RayPoolKit's `measure_client_pools.py` at its own `HEAD` (7 imports, 54 calls at `b510bc9`).
