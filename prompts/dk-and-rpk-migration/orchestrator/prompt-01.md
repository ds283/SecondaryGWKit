# Orchestrator — prompt 01, record the installed packages in each manifest

Read [`../README.md`](../README.md) first: §0.1, §0.2, §2.2 (01's charter), §4, §5, §6.1 (U4),
§6.2 C13 and §7. This is the campaign's first note, so it states the conventions in full. They are
`datastore-generic-followup`'s
([`../../datastore-generic-followup/orchestrator/prompt-01.md`](../../datastore-generic-followup/orchestrator/prompt-01.md)
§0's last list), carried over and adjusted for a change that touches no layer file.

**You do not write code.** You may:
- run the suites and the tests the log names;
- replay the log's deliberate-breakage diffs with `git apply`, and revert them;
- run an in-process check or a probe from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`01-record-the-installed-packages-in-each-manifest.md`](../01-record-the-installed-packages-in-each-manifest.md)
**Board items:** P1–P3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- The campaign is written (`382d776`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`, and in particular `RunRegistry/`, `Datastore/`,
    `RayTools/`, `config/`, `main.py`, `requirements.txt` and `docs/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- One prompt at a time in this checkout. The campaign runs in the main checkout (README §4.2), and
  02 reads the manifests that 01 writes.

## 0. What makes this prompt unusual

**It is small, additive and runs no layer code**, and it is the first thing R0, R1 and R2 rely on.
Each rehearsal's manifest is to say what was installed: R0 neither package, R1 `datastorekit`, R2
both. So the review has two halves:
- **the record is right**: each of the five fields is read from the right place in PEP 610's
  `direct_url.json`, a missing or odd file never raises, and the test stand-in replaces what the
  function actually calls;
- **nothing else moves**: every existing manifest key keeps its value, `status.json`, `list` and
  the store sidecar are untouched, and nothing is refused.

**The tests stand in `importlib.metadata`, so a mock aimed at the wrong name passes vacuously.** If
the module does `from importlib.metadata import distribution`, a patch of
`importlib.metadata.distribution` replaces nothing the function calls, and the "not installed"
test passes only because nothing is installed. The breakages catch part of this. The review's
probe (§3 check 5) catches the rest, by giving the real `importlib.metadata` a stand-in
distribution on `sys.path`, with no mock at all.

**One sentence of `begin()`'s docstring becomes false.** `RunRegistry/__init__.py:519-522` says:
"Every manifest field but one is one README §0 would have caught something with; there are no
others. The one is `results_store_id`…". After 01 there are two. The prompt asks the agent to
mention `packages` where `begin()`'s docstring lists what the manifest records; this is that place,
and README §6.2 C10 asks that prose a commit makes false be corrected in that commit. The agent
corrects it, saying why `packages` exists (U4), and logs it as `STRUCTURALLY REQUIRED`. The module
docstring (`:1-45`) describes the layout and lists no manifest key, so it need not change.

**What moves on purpose:**
- `RunRegistry/__init__.py` gains `RECORDED_DISTRIBUTIONS`, `package_provenance()`, one standard-
  library import if needed (`importlib.metadata`), one line in `begin()` after `:586`, and the
  docstring sentence above.
- `RunRegistry/tests/test_run_registry.py` gains `TestPackageProvenance` (at least five tests) and
  one method in `TestManifest`. No existing test changes.
- The log, the board and README §4's "01 ships".
- No issue opens or closes. The index stays at 104.

**The facts, checked by the orchestrator on 2026-10-11 at `382d776`** (the code is `b510bc9`'s:
`git diff --stat b510bc9 382d776 -- . ':!prompts'` is empty). Pass them on.
- **The prompt's line numbers hold.** Checked: `RunRegistry/__init__.py` `git_provenance` at `:97`
  (body to `:111`), `begin` at `:484`, the manifest dict at `:568-585` (the prompt's `:566-586`
  brackets it), `manifest.update(git_provenance())` at `:586`; `test_run_registry.py`
  `RegistryTestCase` at `:112`, `TestManifest` at `:221`,
  `test_manifest_carries_the_provenance_triple` at `:267`, `TestCheckpoint` at `:276`; `stores.py` `HISTORY_KEYS` at `:162` and `TAKEN_KEYS` at `:207`.
- **The imports.** `RunRegistry/__init__.py` imports only `hashlib`, `json`, `os`, `re`,
  `subprocess`, `sys`, `time` and `datetime` at module scope (`:47-54`); `importlib` is not among
  them. `test_run_registry.py` already imports `json`, `tempfile` and `from unittest import mock`
  (`:8-18`).
- **The venv.** Python 3.12.15. `pip show datastorekit raypoolkit` finds neither, and
  `importlib.metadata.distribution("datastorekit")` raises `PackageNotFoundError`. **No
  distribution in this venv has a `direct_url.json`** (`ray`'s `read_text` gives `None`), so there
  is no real VCS or editable install to test against: the tests' stand-ins and the review's
  `sys.path` probe are the only evidence. `importlib.metadata.Distribution` has no `origin`
  attribute in 3.12, so `read_text("direct_url.json")` plus `json` is the way, as the prompt says.
- **Who reads a manifest's keys.** No tracked `.py` file compares a manifest's key set:
  `git grep` for `set(manifest`, `manifest.keys`, `sorted(manifest` and `keys() ==` finds nothing.
  Readers take named fields: `stores.py:921` (`results_store_id`),
  `test_pipeline_adoption.py:351` (five fields by name), `test_store_sidecar.py:567`. So a new key
  breaks no reader.
- **The sidecar is untouched.** `HISTORY_KEYS` (`:162`) and `TAKEN_KEYS` (`:207`) are checked
  exactly, and the prompt does not touch them (README §6.2 C13).
- **The registry.** `python -m RunRegistry list` shows 17 runs, 15 finished and 2 `unknown` (the
  pre-registry `a3-pilot` pair), none `running`.
- **Not touched, and staying:** every other file under `RunRegistry/`, every other test, `main.py`,
  `requirements.txt`, every `docs/` script, and every store and run directory under `var/`.
- **Expected counts after 01:**
  - RunRegistry 280 + the new tests (at least 6, so at least 286);
  - AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ComputeTargets 693,
    Datastore 739 (163 + 185 + 200 + 191): unchanged.
- **The index:** 104, unchanged.

**What the review exists to establish.**
- **(E1) The function.** `package_provenance()` gives `None` for a distribution not installed, and
  otherwise exactly the five keys, each read from its PEP 610 place. An absent or non-object
  `direct_url.json` gives `None, None, None, False` and never raises. It imports nothing it
  describes.
- **(E2) It works against the real `importlib.metadata`**, not only against the mock: §3 check 5.
- **(E3) The manifest.** `begin()` writes `packages` once, after `git_provenance()`, before
  `status.json`. Every other key and value is as before.
- **(E4) Nothing else in `RunRegistry/` changes.** Every function of `__init__.py` but `begin`
  dumps identically by `ast`, docstrings stripped; `begin` differs by the one assignment.
- **(E5) The pins.** (i)–(v) each fail the named tests.
- **(E6) The records.** The board has the §1 row, P1–P3 and §5.1; README §4's "01 ships" says it
  shipped. No §3 entry; the index is untouched.

**Conventions.** All of these bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the diff and the tests show.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.**
  - Use README §7's command, write the output to a file in the agent's scratch directory, and grep
    the verdict from it.
  - Run each suite **in the foreground**, serially, within the turn.
  - Run the Datastore suite in module groups (`-p 'test_[a-f]*.py'`, `[g-o]`, `[p-r]`, `[s-z]`),
    and add the counts. A Bash call is capped at ten minutes.
  - Under load, four tests are known to fail on wall clock or chance (README §7):
    - `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
    - `test_background_tau.test_build_cost`;
    - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`;
    - LiouvilleGreen's `test_3bessel_analytic.test_JJJ`.

    Re-run a failure alone before calling it real. If it persists alone, settle it on an export of
    the tree before the prompt.
- **Breakage diffs** are recorded so that `git apply` replays them from a clean tree at the agent's
  commit. Check each with `git apply --check` and `-R --check`, and replay in `bash`, not zsh.
- **Scratch.** The agent uses its own subdirectory of the session scratchpad, never `/tmp` and
  never `var/`. Python's `tempfile` in tests is fine. §3.2's hand check uses a temporary `root`
  outside the repository.
  - No scratch `.py` file goes under `Datastore/`, `RayTools/` or `RunRegistry/`, even for a
    moment.
- **The venv.** The agent installs nothing, uninstalls nothing and runs no `pip install`. 03 is the
  first prompt that changes the venv.
- **Ray.** The agent starts no Ray. ChamPBH's cluster, if running, is left alone.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. Review against the agent's commit, and check
  `git log` before measuring at `HEAD`.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 382d776..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **The venv.** `./venv/bin/pip show datastorekit raypoolkit` must find neither (the prompt's §5
   stop condition).
5. **Baselines.** Re-measure all six roots at `HEAD`, serially, in the foreground, with the load
   average (README §7). The last measurement was at `b510bc9` on 2026-10-11 (§0.2): AdaptiveLevin
   32, CosmologyModels 39, LiouvilleGreen 148 (1 skipped), ComputeTargets 693 (the wall-clock
   flake, passed alone), RunRegistry 280, Datastore 739 (163 + 185 + 200 + 191). Give the agent
   the numbers.
6. **The pre-images**, for check 6. Record `git rev-parse HEAD:<path>` for
   `RunRegistry/__init__.py` and `RunRegistry/tests/test_run_registry.py`. At `382d776` they are
   `a6a3e6a7…` and `b1b84f10…`.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is
  `logs/01-record-the-installed-packages-in-each-manifest.md`, in README §5.1's form, with the
  prompt's §7 additions. `logs/` does not exist yet; it creates it.
- **The files it may add or change are the prompt's §6 list, and nothing else:**
  `RunRegistry/__init__.py`, `RunRegistry/tests/test_run_registry.py`, the log, this board and
  README §4. In particular, it must not touch `RunRegistry/stores.py`, `RunRegistry/__main__.py`,
  any other test, `docs/OPEN_ISSUES.md` (unless it opens an issue), `orchestrator/`, `measure/` or
  `var/`.
- **The `begin()` docstring sentence** at `:519-522` is corrected, as §0 says.
- **It runs no build, installs nothing and starts no Ray.**
- **Stop and ask** on any of the prompt's §5 conditions. An existing test that fails is current
  behaviour moving, never a reason to edit it.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — eight checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the prompt's §6 list.
   - The diff over everything outside `RunRegistry/__init__.py`,
     `RunRegistry/tests/test_run_registry.py` and `prompts/dk-and-rpk-migration/` is empty.
   - Within `prompts/dk-and-rpk-migration/`, only the log, `IMPLEMENTATION_STATE.md` and
     `README.md` move, and in `README.md` only §4's "01 ships".
2. **E1, by reading.** Read `RECORDED_DISTRIBUTIONS`, `package_provenance` and the `begin()` change
   in full.
   - The tuple is exactly `("datastorekit", "raypoolkit")`, with a comment citing README §0 and U4.
   - Only `PackageNotFoundError` gives `None`. Any failure to read or parse `direct_url.json`,
     including a JSON value that is not an object, gives the four defaults, and no other exception
     escapes.
   - `url` is the top-level `url`; `commit` and `requested_revision` are `vcs_info`'s `commit_id`
     and `requested_revision`; `editable` is `bool(dir_info.editable)`, `False` when `dir_info` is
     absent.
   - The docstring says what each field is, that it is read at `begin()`, that a VCS install's
     `commit` is what the tag resolved to at install, and that an editable install is recorded,
     not refused.
   - The function imports nothing it describes, and uses the standard library only.
3. **E1, the odd cases.** In the scratchpad, with the agent's stand-in pattern, call the function
   on: a `direct_url.json` object whose `vcs_info` or `dir_info` is not an object (a list, a
   string); one with `dir_info: {"editable": false}`; one with `vcs_info` lacking
   `requested_revision`; and a `read_text` that raises `OSError`. Each must return without
   raising. Any that raises is a finding: residue if one guard fixes it, otherwise a §3 issue.
4. **The tests, by reading.** Read `TestPackageProvenance` and the new `TestManifest` method in
   full.
   - The patch replaces the name the module actually calls.
   - Test 2's commit is 40 hex and its URL a `git+https` URL at `v0.2.2`, and every field is
     asserted, not only some.
   - Test 4 has both sub-cases, each asserting all four defaults.
   - Test 6 asserts the key set of `packages` and its equality with `package_provenance()`, and
     **does not assert `None`** for either package, so it passes unchanged after 03 and 06.
   - No test passes on an empty input, needs Ray, or opens anything under `var/`.
5. **E2, the probe without a mock.** In the scratchpad, make a directory holding
   `datastorekit-0.2.2.dist-info/` with a minimal `METADATA` (`Metadata-Version`, `Name`,
   `Version`) and a PEP 610 `direct_url.json` for
   `git+https://github.com/ds283/DatastoreKit` at `v0.2.2`, commit
   `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6` (`git -C ../DatastoreKit rev-parse v0.2.2^{commit}`).
   Put the directory first on `sys.path` in a fresh `./venv/bin/python -I` process with the
   repository on its path, and call `package_provenance()`.
   - `datastorekit` gives every field, with `editable: False`; `raypoolkit` gives `None`.
   - Repeat with an editable `direct_url.json` (`{"url": "file:///…", "dir_info": {"editable":
     true}}`) and with the file removed.
   - Then `RunRegistry.begin(...)` into a temporary `root` in the same process, and read the
     manifest's `packages`: it equals the call's result.
6. **E3 and E4, by `ast` against the pre-images, docstrings stripped.**
   - `RunRegistry/__init__.py`: every function and class statement but `begin` dumps identically;
     the module differs by the import, the tuple and the new function; `begin` differs only by the
     `manifest["packages"] = package_provenance()` line, after `manifest.update(git_provenance())`
     and before `write_json_atomic`.
   - `begin()`'s docstring: the sentence at `:519-522` is corrected, and nothing else in it moves.
   - `test_run_registry.py`: every existing class and method dumps identically.
7. **E5, the pins.** In `bash`, replay each breakage diff (i)–(v) with `git apply`, run the named
   tests, see them fail, and see nothing else fail. Revert each with `git apply -R`, and check that
   `git status` is clean. Breakage (v) must fail test 6 on the key set, not by accident.
8. **E6 and the suites.**
   - The board has the §1 row (✅, "this commit", the log), P1–P3 ✅ and `### 5.1 After prompt 01`;
     README §4's "01 ships" says it shipped as named, or names what differs.
   - No §3 entry, and `docs/OPEN_ISSUES.md` is untouched at 104 — unless the agent opened one, in
     which case the index row, count and date move in the same commit.
   - Run all six roots, serially, at `HEAD`. RunRegistry is 280 + the new tests; every other count
     is §1 step 5's. `black --check` is clean on both touched Python files.
   - The log's hand check (prompt §3.2) shows both `null`, and was made in a temporary `root`
     outside the repository: `var/runs/` gains no directory.

## 4. After the review

- **Record it on this board** as a §5.1 *Orchestrator review of prompt 01* paragraph: what each
  check found, the probe's outcome, and anything handed on to 02 or 03. Fix small residue in a
  follow-up commit of its own. That includes:
  - "this commit" → the SHA;
  - a list of the orchestrator notes on the board, with this one marked "used for 01".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - the function, its fields, and its output in this venv;
  - the tests added, by name;
  - the probe's outcome against the real `importlib.metadata`;
  - that the existing manifest keys, `status.json`, `list` and the sidecar are unchanged;
  - the counts of the six roots.
- **Then 02**, R0. It reads 01's `packages` in every manifest it writes (its acceptance 2). Its
  note is written after this review.
