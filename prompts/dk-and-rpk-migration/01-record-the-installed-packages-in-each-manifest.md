# Prompt 01 — record the installed packages in each run's manifest

**Campaign:** [`README.md`](README.md) · **Board items:** **P1**–**P3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 02–04, after the user reviewed the plan (`4532188`).
**Gate:**
- the plan is committed;
- `git status` is clean;
- `python -m RunRegistry list` shows nothing `running`.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. A small, additive change to `RunRegistry/__init__.py`, with
tests that need no real install.

**Read first:**

1. [`README.md`](README.md): §0.1, §2.2 (01's charter), §4 (what 01 ships, and who reads it), §5
   and §6.1 (U4).
2. `CLAUDE.md`, "Long-running jobs — the run registry", rule 6 (provenance).
3. `RunRegistry/__init__.py`:
   - the module docstring, for the manifest's place in the layout (`:1-50`);
   - `git_provenance` (`:97-111`), beside which the new function goes;
   - `begin` (`:484-600`), and the manifest it writes (`:566-586`).
4. `RunRegistry/tests/test_run_registry.py`: `RegistryTestCase` (`:112`) and `TestManifest`
   (`:221-274`), whose `test_manifest_carries_the_provenance_triple` (`:267`) is the pattern for
   the new manifest test.
5. PEP 610, "Recording the Direct URL Origin of installed distributions": the `direct_url.json`
   fields `url`, `vcs_info.{vcs, commit_id, requested_revision}` and `dir_info.editable`.
   `importlib.metadata.distribution(name).read_text("direct_url.json")` reads it.

Line numbers are at `b510bc9`.

---

## 1. What is wanted

From prompt 04 on, a pipeline run runs code that is not in SGK's tree: `datastorekit` from 04,
`raypoolkit` from 07. Today a run's manifest records SGK's `git_head` and `git_dirty`, and
nothing of an installed package. So which release a run used would be implied only by
`requirements.txt` at its `HEAD`, and only if the venv was installed from it. The user decided
(U4) that each manifest records what was installed. This lands first, so that R0, R1 and R2 each
say what they ran: R0 neither package, R1 `datastorekit`, R2 both.

- **P1 — `package_provenance()`.** A function in `RunRegistry/__init__.py` that describes the
  installed distributions the campaign adopts.
- **P2 — the manifest's `packages`.** `begin()` writes it, beside `git_provenance()`.
- **P3 — the tests.**

**What does not change.**
- Every existing manifest key and its value.
- `list`'s output, and the `status.json` format.
- The store sidecar: its `history` and `taken` records keep their exact key sets
  (`RunRegistry/stores.py:162`, `:207`; README §6.2 C13).
- Nothing is refused. The registry records; it does not refuse, even for an editable install.

---

## 2. What to change

**P1 — in `RunRegistry/__init__.py`, beside `git_provenance`:**
- `RECORDED_DISTRIBUTIONS = ("datastorekit", "raypoolkit")`, a module-level tuple, with a comment
  saying why these two (README §0, U4).
- `package_provenance(names=RECORDED_DISTRIBUTIONS) -> dict`, mapping each name to:
  - `None` when `importlib.metadata.distribution(name)` raises `PackageNotFoundError`;
  - otherwise a dict with exactly these keys:
    - `"version"`: the distribution's version;
    - `"url"`: `direct_url.json`'s `url`, or `None`;
    - `"commit"`: its `vcs_info.commit_id`, or `None`;
    - `"requested_revision"`: its `vcs_info.requested_revision`, or `None`;
    - `"editable"`: `True` when `dir_info.editable` is true, and `False` otherwise.
- A `direct_url.json` that is absent, or does not parse as a JSON object, gives `None` for the
  URL, commit and requested revision fields and `False` for `editable`. It never raises.
  `begin()` must not fail because a package's metadata is odd.
- Its docstring says:
  - what each field is, and that it is read from the installed distribution at `begin()`;
  - that a VCS install's `commit` is the commit the tag resolved to when pip installed it, which
    is what makes the record reproducible;
  - that an editable install is recorded, not refused (`CLAUDE.md`: the registry records).
- Read the metadata with the standard library only (`importlib.metadata`, `json`). Import no
  package being described.

**P2 — in `begin()`:** after `manifest.update(git_provenance())` (`:586`), set
`manifest["packages"] = package_provenance()`. It is written once, with the rest of the manifest,
before the work starts. Mention `packages` where the module docstring or `begin()`'s docstring
lists what the manifest records, if either does.

**P3 — tests**, in `RunRegistry/tests/test_run_registry.py`, as a new class
`TestPackageProvenance`. No Ray, nothing under `var/`, no real install. Stand in
`importlib.metadata.distribution` with `unittest.mock`, through the name the module uses, with a
stand-in whose `version` and `read_text` you control. At least:
1. `test_a_distribution_not_installed_is_none`.
2. `test_a_vcs_install_records_its_commit_and_tag`: a `direct_url.json` of a `git+https` install
   at `v0.2.2`, with a 40-hex commit id, gives every field.
3. `test_an_editable_install_is_recorded_as_editable`: `dir_info: {"editable": true}`.
4. `test_an_absent_or_unreadable_direct_url_gives_nones`, with two sub-cases: `read_text` returns
   `None`, and it returns text that is not a JSON object.
5. `test_the_recorded_names_are_the_two_packages`: `RECORDED_DISTRIBUTIONS` is exactly
   `("datastorekit", "raypoolkit")`.
6. In `TestManifest`, `test_manifest_records_the_installed_packages`. A manifest written by
   `self.begin(...)`:
   - holds `packages`, with exactly the keys of `RECORDED_DISTRIBUTIONS`;
   - and equals `package_provenance()` taken in the same process.

   **It must not assert what this venv holds.** 03 and 06 install the packages, and the test must
   pass unchanged before and after. §3.2 shows what this venv holds, by hand.

---

## 3. Verification

1. **Every suite**, with counts before and after:
   - RunRegistry 280 + your new tests, OK;
   - the other five roots at README §7's counts.

   Run the Datastore root in its four groups. The README §7 load-sensitive tests are re-run alone
   before being called failures.
2. **The real thing, once, by hand.** In a temporary `root`, call `RunRegistry.begin(...)` from
   `./venv/bin/python` and show the manifest's `packages`: both `null`. Delete nothing under `var/`;
   use a temporary directory outside the repository.
3. `black --check` is clean on both touched Python files.
4. **Deliberate breakage** (README §5 rule 8). Each must make the named tests fail; then restore
   it.
   - (i) `package_provenance` drops `editable` (always `False`): test 3.
   - (ii) a `direct_url.json` that is not a JSON object raises: test 4.
   - (iii) `begin()` no longer writes `packages`: test 6.
   - (iv) `RECORDED_DISTRIBUTIONS` loses `"raypoolkit"`: test 5.
   - (v) `begin()` writes `package_provenance(names=("datastorekit",))`: test 6.

   Record each in the log as a diff, exactly as applied, with what failed. Check each with
   `git apply --check` and `-R --check`, and leave `git status` clean.

---

## 4. Acceptance

1. **P1–P3** as §2 says.
2. Every suite has the counts of §3.1, and §3.2's manifest shows both `null`.
3. **The records**, in the same commit:
   - this board: the §1 row for 01, items P1–P3, and `### 5.1 After prompt 01 (<date>)`;
   - README §4's "01 ships" says it shipped as named, or names what differs.
   - No §3 entry is expected. If you open one, `docs/OPEN_ISSUES.md` follows in the same commit.

---

## 5. Stop conditions — stop and ask

- An existing test fails, other than README §7's load-sensitive ones re-run alone. Never edit an
  existing test to pass.
- Either distribution is already installed in the venv. README §0.2 says neither is, and 03
  depends on that.
- Recording `packages` would need a change to the sidecar, `list` or `status.json`.
- Anything would open or touch a store or run directory under `var/`, or start Ray.

---

## 6. What this prompt does not do

- **Files it changes:**
  - `RunRegistry/__init__.py` and `RunRegistry/tests/test_run_registry.py`;
  - the log, this board and the README's §4.

  Nothing else.
- It installs nothing, and changes no pin.
- It does not record packages in the store sidecar, or in the fingerprint's `taken` record.

---

## 7. The log and the board

`logs/01-record-the-installed-packages-in-each-manifest.md`, using README §5.1. In addition, it
gives:
- the function's signature and its output in this venv;
- every test added, by class and method;
- each breakage (i)–(v), as a diff, with what failed.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
