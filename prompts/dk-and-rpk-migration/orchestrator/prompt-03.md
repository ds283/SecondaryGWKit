# Orchestrator — prompt 03, pin `datastorekit` `v0.2.2`

Read [`../README.md`](../README.md) first: §0.2 ("The venv"), §2.2 (03's charter), §4, §4.2, §5
(rules 11–13) and §7. Then read [`prompt-01.md`](prompt-01.md) §0's conventions, which carry over
except where §0 below says otherwise, and this board's *Orchestrator review of prompt 02*.

**You do not write code.** You may:
- read the venv's installed files and metadata, and compare them with the tag's;
- run the suites;
- run in-process checks from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

You install nothing and uninstall nothing. The venv is changed once, by the agent's
`pip install`.

**The prompt:** [`03-pin-datastorekit-v0.2.2.md`](../03-pin-datastorekit-v0.2.2.md)
**Board items:** K1–K3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 02 landed (`7650102`) and was reviewed (`09cc40a`), with R0's fingerprints equal to
  `39809dca…` (board §2). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code.
  - "Code" here means anything outside `prompts/`, and in particular `requirements.txt`,
    `RunRegistry/`, `Datastore/`, `RayTools/`, `config/`, `main.py` and `docs/`.
  - If a later commit touches code, re-check every fact below that it could move, and hand the
    agent the corrections as an addendum.
- Nothing is `running` in the registry.
- **One prompt at a time in this checkout.** The venv is shared (README §4.2): nothing else may
  run from it while pip changes it.
- **The network.** The install fetches the tag from GitHub and the build backend from PyPI. The
  user's dispatch of this prompt authorises exactly that: `pip install`, its `--dry-run`, and
  `git ls-remote`. No other download.

## 0. What makes this prompt unusual

**It changes the venv, which no commit records and `git revert` does not undo.** Every earlier
prompt left the venv as README §0.2 measured it. This one adds a distribution to it. So the review
has two halves:
- **the venv holds the tag and nothing else moved**: one distribution added, at commit
  `6d8ee9ae…`, not editable, its files the tag's files, every other distribution at its version;
- **SGK does not see it yet**: nothing imports `datastorekit`, and the six roots give their
  counts. An installed, unimported package is inert (README §4.2), and that is what 04 relies on.

**The prompt says "stop before accepting any such change", but a plain `pip install` has no
point at which to stop.** So the agent runs a dry run first, after K1 and before K2:

```bash
./venv/bin/pip install --dry-run --report var/dk-and-rpk-migration-03/dry-run-report.json -r requirements.txt
```

It must say `Would install datastorekit-0.2.2` and nothing else, and the report's `install` list
must hold that one entry. The dry run clones the tag and builds its metadata, but installs
nothing. Only then is K2 run, with the prompt's command. The agent logs the dry run as an
`IMPLEMENTATION CHOICE`.

**Two more checks that the prompt does not spell out:**
- **What the distribution owns, not only what is in `datastorekit/`.** The prompt's K3.4 compares
  the installed `datastorekit/` with the tag's. A wheel could also carry a second top-level
  package (a stray `tests/` or `docs/`) that that comparison never sees. So the agent lists
  `importlib.metadata.files("datastorekit")` and shows that every entry lies under `datastorekit/`
  or `datastorekit-0.2.2.dist-info/`, and gives `top_level.txt` if there is one.
- **The raw `direct_url.json`**, shown in the log beside `package_provenance()`'s reading of it,
  so that the record is checked against its source.

**README §4's "03 ships" line.** The prompt's §5 does not list README. But 01 and 02 each recorded
their interface there, and 04 reads this one. So the agent adds one line under "03 and 06 ship the
pins", saying 03 shipped as named (or what differs). It logs this as an `IMPLEMENTATION CHOICE`,
citing this note. It changes nothing else in README.

**What moves on purpose:**
- `requirements.txt` gains one line, at `:17`. No other line changes.
- The venv gains `datastorekit` 0.2.2. Nothing else in it moves.
- The log, this board (§1 row, K1–K3, `### 5.3 After prompt 03`), and README §4's one line.
- No issue opens or closes. The index stays at 106.

**The facts, checked by the orchestrator on 2026-10-11 at `09cc40a`.** The code outside `prompts/`
and `docs/` is `f54f1c5`'s (`git diff --stat f54f1c5 09cc40a -- . ':!prompts' ':!docs'` is empty).
Pass them on.
- **`requirements.txt`** has 98 lines and ends with a newline. It is byte-identical to
  [`../measure/out/b510bc9-pip-freeze.txt`](../measure/out/b510bc9-pip-freeze.txt), which is in
  turn equal to the venv's `pip freeze` today. `cycler==0.12.1` is at `:16` and `distlib==0.3.9` at
  `:17`, so the pin becomes the new `:17`. So `pip install -r` asks for nothing the venv lacks
  except the pin.
- **The venv.** Python 3.12.15, pip 26.2.1.
  - `pip show datastorekit raypoolkit` finds neither, and `pip check` reports no broken
    requirement.
  - `site-packages` holds no `.egg-link` and no `.pth` but `distutils-precedence.pth`, and
    `PYTHONPATH` is empty. So nothing can make `import datastorekit` find the DatastoreKit checkout
    instead of the install.
- **The tag**, local and remote, agree:
  - `git -C /Users/ds283/Documents/Code/DatastoreKit rev-parse v0.2.2` gives the annotated tag
    object `c32d0278d71c5ed543399546908bde8498905e59`, and `v0.2.2^{commit}` gives
    `6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`;
  - `git ls-remote --tags https://github.com/ds283/DatastoreKit 'v0.2.2*'` gives `c32d0278…` for
    `refs/tags/v0.2.2` and `6d8ee9ae…` for `^{}`.

  DatastoreKit's `main` is at `460b012`, two records commits past the tag. That is irrelevant: the
  pin names the tag.
- **`pyproject.toml` at the tag** (equal to `main`'s):
  - the build requires `setuptools>=64`, which pip fetches into an isolated build environment, not
    into the venv;
  - it depends on `ray>=2.43` and `sqlalchemy>=2.0.39,<2.1`, which `ray==2.43.0` and
    `SQLAlchemy==2.0.39` satisfy;
  - `packages.find` includes `datastorekit*` and excludes `datastorekit.tests` and
    `datastorekit.tests.*`.
- **The tag's package** (`git archive v0.2.2 datastorekit`, extracted in the scratchpad):
  - 67 `.py` files: 47 under `datastorekit/tests/` and **20** outside it;
  - no file outside `datastorekit/tests/` that is not `.py`;
  - three package directories, `datastorekit`, `datastorekit/SQL` and `datastorekit/tools`, each
    with its `__init__.py`.

  So the install should hold exactly those 20 `.py` files, and nothing under `tests/`.
- **What pip records.** For a `git+https` requirement, PEP 610's `url` is the URL without the
  `git+` prefix: `https://github.com/ds283/DatastoreKit`. `vcs_info` holds `vcs: git`, `commit_id`
  and `requested_revision: v0.2.2`. That is what the prompt's K3.3 expects. `pip freeze` then
  prints `datastorekit @ git+https://github.com/ds283/DatastoreKit@6d8ee9ae…`, the commit and not
  the tag.
- **DatastoreKit's adoption README §1** still shows the pin as `…@v0.2.0` (`:45`). Its header
  (`:13-14`) says the addendum at the end supersedes the statements it names, and the `v0.2.2`
  addendum (`:277-288`) gives `…@v0.2.2`. This is that repository's record, not a defect, and
  nothing is opened for it.
- **Nothing in SGK imports the package.** `git grep -i datastorekit` outside `prompts/` and `docs/`
  finds `CLAUDE.md`'s freeze section, `RunRegistry.RECORDED_DISTRIBUTIONS`, and strings in
  `RunRegistry/tests/test_run_registry.py`. No tracked script reads `requirements.txt`.
- **The RunRegistry tests are silent on the venv.** `TestPackageProvenance` stands in
  `importlib.metadata.distribution`. `test_manifest_records_the_installed_packages` compares the
  manifest with `package_provenance()` and asserts no value. So RunRegistry stays at 286 after the
  install, as 01's review intended.
- **The registry.** `python -m RunRegistry list` shows 21 runs: 19 finished and 2 `unknown` (the
  pre-registry `a3-pilot` pair), none `running`. No Ray process is up.
- **Expected counts after 03:** the dispatch baselines (§1 step 5), unchanged.
- **The index:** 106, unchanged.

**What the review exists to establish.**
- **(E1) The pin.** `requirements.txt` differs by the one line, in sorted position.
- **(E2) One distribution added.** `pip freeze` differs from `b510bc9-pip-freeze.txt` by exactly
  `datastorekit @ git+https://github.com/ds283/DatastoreKit@6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6`,
  and `pip check` is clean.
- **(E3) The record.** `direct_url.json` and `package_provenance()["datastorekit"]` give version
  `0.2.2`, URL `https://github.com/ds283/DatastoreKit`, commit `6d8ee9ae…`, requested revision
  `v0.2.2`, `editable: false`. `["raypoolkit"]` is `None`.
- **(E4) The files are the tag's.** The 20 `.py` files byte for byte, no `tests/`, nothing else
  `.py`, and the distribution owns nothing outside `datastorekit/` and its `dist-info`.
- **(E5) It is inert.** It imports without starting Ray, and nothing in SGK imports it. The six
  roots give their counts.
- **(E6) The records.** The log shows every check with its command and output. The board has the
  §1 row, K1–K3 and §5.3, which names the rollback. README §4 has the one line.

**Conventions.** Those of [`prompt-01.md`](prompt-01.md) §0 bind the agent, with these changes:
- **The venv.** The agent runs exactly two `pip install` commands: the dry run above, and the
  prompt's K2. It never uses `-e`, a local path, `--upgrade`, `--force-reinstall` or `--no-deps`.
  If pip fails only because it cannot write its cache, it may add `--no-cache-dir` to both
  commands, and logs it. Any other flag is a stop. It runs no `pip uninstall` unless the prompt's
  §4 restore applies, and then it stops.
- **Scratch.** The saves the prompt names go under `var/dk-and-rpk-migration-03/` (gitignored).
  The tag's archive is extracted into the agent's own subdirectory of the session scratchpad,
  outside both repositories, never under `/tmp`.
- **The comparison.** Whatever code the agent uses for K3.4 goes into the log verbatim, with the
  paths it took as arguments, so that the orchestrator can rerun it against another tag.
- **Ray.** The import check starts no Ray, and the agent starts none. ChamPBH's cluster, if
  running, is left alone.
- **DatastoreKit.** The agent reads it and runs `git rev-parse`, `git archive` and
  `git ls-remote` against it. It changes, fetches into, checks out and commits nothing there
  (README §5 rule 12).
- **No breakage diffs.** 03 adds no code and no test (prompt §3.4), so there is nothing to replay.
  The non-vacuity of K3.4's comparison is checked in the review instead (§3 check 4).

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`.
   - `git status --short` and `git diff --cached` must be empty.
   - `git worktree list` must show this checkout alone.
   - `git log --oneline 09cc40a..HEAD` must touch no code; otherwise see the gate.
2. **The registry.** `./venv/bin/python -m RunRegistry list` must show nothing `running`.
3. **Ray.** Record `pgrep -lf 'gcs_server|raylet|ray::'`. No listed process may come from this
   repository's `venv`.
4. **The venv.** `./venv/bin/pip show datastorekit raypoolkit` finds neither, `pip check` is
   clean, and `pip freeze` equals `b510bc9-pip-freeze.txt`.
5. **Baselines.** Re-measure all six roots at `HEAD`, serially, in the foreground, with the load
   average (README §7). The last measurement was at `7650102` (02's review): AdaptiveLevin 32,
   CosmologyModels 39, LiouvilleGreen 148 (1 skipped), RunRegistry 286, ComputeTargets 693,
   Datastore 739 (163 + 185 + 200 + 191). Give the agent the numbers.
6. **The network.** `git ls-remote` against the tag answers, and `https://pypi.org/simple/` is
   reachable.

## 2. Dispatch

Dispatch one fresh-context subagent, on **Sonnet**. Give it:
- the prompt;
- the campaign README and this board;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end.** Its log is `logs/03-pin-datastorekit-v0.2.2.md`, in README §5.1's
  form, with the prompt's §6 additions.
- **The files it may add or change:** `requirements.txt`, the log, this board, and README §4's one
  line. In particular, not `docs/OPEN_ISSUES.md` (unless it opens an issue), `orchestrator/`,
  `measure/`, any `.py` file, or anything in DatastoreKit.
- **The dry run comes before the install**, and its result is a stop condition like the install's.
- **The network is authorised** for the dry run, the install and `git ls-remote`, and for nothing
  else.
- **Stop and ask** on any of the prompt's §4 conditions. A suite that moves is a stop, not
  something to fix.
- **Run the Datastore suite in four groups**, in the foreground.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only `requirements.txt` and
   `prompts/dk-and-rpk-migration/`.
   - In `prompts/dk-and-rpk-migration/`, only the log, `IMPLEMENTATION_STATE.md` and `README.md`
     move, and in `README.md` only §4's one line.
   - `git diff HEAD~1 HEAD -- requirements.txt` is one added line, at `:17`, between `cycler` and
     `distlib`, spelled exactly as the prompt's K1.
2. **E2, independently.** `./venv/bin/pip freeze` against `b510bc9-pip-freeze.txt` is exactly the
   one added line, and `pip check` is clean. Also, `pip freeze` with that line removed equals
   `requirements.txt` with the pin removed. `pip show datastorekit` gives `Location` in this venv's
   `site-packages`, `Requires` ray and sqlalchemy, and no `Editable project location`.
3. **E3, from the source.** Read the installed `direct_url.json` raw. Its `url`,
   `vcs_info.vcs`, `commit_id` and `requested_revision` are as §0 says, and it has no `dir_info`
   with `editable: true`. `RunRegistry.package_provenance()` in a fresh process gives the five
   fields of the prompt's K3.3, and `raypoolkit` `None`.
4. **E4, the files, and that the comparison can fail.**
   - Rerun the agent's K3.4 comparison, from the log, against a fresh extraction of
     `git archive v0.2.2 datastorekit`. It reports 20 equal, none missing, none extra, no `tests/`.
   - Check it independently: sha256 every `.py` under the installed `datastorekit/`, ignoring
     `__pycache__`, against the archive's 20.
   - **Non-vacuity.** Rerun the agent's comparison against `git archive v0.2.1 datastorekit`. It
     must report exactly one difference, `datastorekit/SQL/ShardedPool.py`. That is the only
     non-test file `v0.2.1` → `v0.2.2` changes (`git diff --stat`, +64 lines). A comparison that
     reports nothing there is vacuous, and the agent's K3.4 is unproven.
   - `importlib.metadata.files("datastorekit")` lies wholly under `datastorekit/` and
     `datastorekit-0.2.2.dist-info/`.
5. **E5, the import.** In a fresh `./venv/bin/python -I` from the repository root:
   `import datastorekit, datastorekit.SQL.ShardedPool`. `datastorekit.__file__` lies in the venv's
   `site-packages`, and `ray.is_initialized()` is false. Then `pgrep` shows no new Ray process.
6. **The tags.** Local `rev-parse` and remote `ls-remote` as §0, and as the log shows.
7. **Nothing imports it.** `git grep -n -E '^\s*(import|from)\s+datastorekit'` outside `prompts/`
   is empty.
8. **The registry and `var/`.** `RunRegistry list` shows the same 21 runs, none `running`, so the
   agent registered nothing. `var/dk-and-rpk-migration-03/` holds the before-saves, the dry-run
   report and pip's output. Check that the before-freeze there equals `b510bc9-pip-freeze.txt`.
9. **The suites.** Run all six roots, serially, at `HEAD`. Every count is §1 step 5's. There is no
   Python change, so `black` has nothing to check.
10. **E6, the records.**
    - The log has every K3 check with its command and output, pip's output abridged to what it
      collected, built and installed, the dry run, the six suites beside the baselines, and the
      deviations classified.
    - The board has the §1 row (✅, "this commit", the log), K1–K3 ✅, and `### 5.3 After prompt 03`
      naming the rollback: `git revert` plus `./venv/bin/pip uninstall -y datastorekit`.
    - README §4 has the one line.
    - No §3 entry, and `docs/OPEN_ISSUES.md` untouched at 106, unless the agent opened one, with
      the index row, count and date in the same commit.

## 4. After the review

- **Record it on this board** as a §5.3 *Orchestrator review of prompt 03* paragraph: what each
  check found, the non-vacuity result, and anything handed on to 04. Fix small residue in a
  follow-up commit of its own. That includes:
  - "this commit" → the SHA, on the board and in the log's header;
  - `orchestrator/prompt-03.md` added to the board's list of notes, marked "used for 03".
- Anything the review finds that is not residue is opened on this board's §3 and in the index.
- **Report to the user:**
  - the pin and where it sits;
  - what the venv gained, and that nothing else in it moved;
  - the provenance record as `package_provenance()` now reads it;
  - the file comparison, and its non-vacuity;
  - the counts of the six roots.
- **Then 04**, the swap. From now on every manifest records `datastorekit` at `6d8ee9ae…`. 04's
  note is written after this review, and must give the `docs/` layer-import count that 02's review
  handed on (18, not README §0.2's 17).
