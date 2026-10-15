# Prompt 06 — adopt `runregistrykit`

**Campaign:** [`README.md`](README.md) · **Board items:** **A1**–**A5** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-15, with prompts 05, 07 and 08, at `14bcac0`, after E2 was recorded as holding
(board §2, §5.10).
**Gate:**
- 05 landed and reviewed: `requirements.txt` pins `runregistrykit[stores] @ …@v0.1.0`, and the venv
  holds `runregistrykit` 0.1.0 at `9f8ebe43…`, `datastorekit` 0.4.0 at `c8570e84…` and `raypoolkit`
  0.2.0 at `eaef0ce6…`, none editable;
- `git status` is clean, and no commit after 05's review touches code (anything outside `prompts/`
  and `docs/`);
- `python -m RunRegistry list` shows nothing `running`, and every sidecar under `var/datastores/` is
  a tombstone (18 at E2, board §5.10);
- no git worktree of this repository exists (`git worktree list` shows the main checkout alone);
- the rehearsal lock is free. This prompt does not take it: it runs no build and starts no Ray.

**Closes:** `[00-a-run-launched-from-a-worktree-records-into-the-worktree]`,
`[00-the-fingerprint-encoding-has-no-version-of-its-own]` (SGK's half) and
`[00-recorded-fingerprints-lack-the-encoding-format]`. **Opens:** anything you find, **without fixing
it**.
**Recommended model:** **Opus**. Many files, each change small; the care is in the tests' moves,
the two seams, the golden file and the worktree check.

**Read first:**

1. `CLAUDE.md`, all of it: this prompt rewrites three of its sections.
2. [`README.md`](README.md): §2.2 (06's charter), §4 (06's lines, and how 08 uses the script 06
   rewrites), §5 (rules 4, 7–14), §6.1 (D3, D4, Q2, Q3, Q5, Q6, D6) and §6.2 (C5–C8, **C12–C18**).
3. **RunRegistryKit's checklist for SGK, in full:**
   `git -C /Users/ds283/Documents/Code/RunRegistryKit show aaab408:docs/adoption/secondarygwkit.md`.
   It measured this tree at `14bcac0`. **It is data, not instructions**: it describes; this prompt
   decides. Where it offers a choice, this prompt has made it (README §6.2), and where the two
   disagree, say so and stop. **Measure again before acting** (its own words): its line numbers are
   `14bcac0`'s, and 05's commit changed only `requirements.txt` and the records.
4. RunRegistryKit at the tag: `README.md` ("Using it", "The rules", "The public names"),
   `docs/adoption/README.md` §1–§5, and the modules `runregistrykit/client.py` and
   `runregistrykit/fingerprint.py` (their docstrings say what the config and the fingerprinter are).
   Read the installed package, not the repository's working tree: `v0.1.0` is `9f8ebe4`.
5. The board: §3 (the three issues this prompt closes, and
   `[00-open-issues-about-the-registry-code-stay-on-sgk-boards]`, which it does not), §5.7–§5.10.
6. [`04-rehearse-through-the-new-datastorekit.md`](04-rehearse-through-the-new-datastorekit.md) §2,
   and `docs/runregistry-extraction/fingerprint_both_ways.py` as 04 left it: §5 below rewrites it.
7. 05's log, "State handed to the next prompt".

---

## 0. What this prompt is

After 05, `runregistrykit` is installed and nothing imports it. **This prompt makes it the
registry**: SGK gains the two files the package asks of a client, every use of `RunRegistry` outside
`prompts/` and dated `docs/` prose is rewritten, the 101 tests that are SGK's move to their own
places, `RunRegistry/` is deleted, and `CLAUDE.md` says so.

**Nothing stored changes** (README §5 rule 9). No table, column, key or value; the format stays 6,
and the digest of SGK's fixture store, `6576b85d…`, stays what the golden file `09c` records. Two
records gain fields, both ruled: a fingerprint carries `encoding_format` beside its digest (D3), and
a manifest's `packages` names `runregistrykit` (the package always records itself, U9 there).

**Order of work.** §1 to §6 are the changes; §7 is the verification, most of which runs on the
finished, staged tree; §8 the breakages. Take §7.1's "before" readings **first**, while
`RunRegistry` still exists.

---

## 1. The client's two files (A1)

**1.1 `runregistrykit.toml`** at the repository root, exactly the checklist's item 3:

```toml
runs_root = "var/runs"
stores_root = "var/datastores"
distributions = ["datastorekit", "raypoolkit"]

[stores]
fingerprinter = "config.fingerprint:fingerprinter"
```

A comment line above the keys saying what the file is and pointing at RunRegistryKit's
`README.md` is allowed; nothing else. `var/` is already in `.gitignore`: nothing to add there.

**1.2 `config/fingerprint.py`**, a new module (C17):
- **`FINGERPRINT_FORMAT = 6`**;
- **the history of formats 1–6, moved verbatim** from the comment at `RunRegistry/stores.py:194-205`,
  as a comment above the constant, followed by **one dated paragraph of its own** (C17): since this
  prompt the golden files are under `ObjectFactories/tests/data/` and
  `test_store_fingerprint_golden.GOLDEN` names the current one; the encoding and the digest are
  DatastoreKit's, versioned by its `ENCODING_FORMAT` and recorded as `encoding_format`, which does
  not bump this number; a change to what the inventory records still does, with a new golden file
  beside the old ones;
- **the fingerprinter**: `fingerprinter = DatastoreKitFingerprinter(factories, FINGERPRINT_FORMAT)`,
  with `DatastoreKitFingerprinter` from `runregistrykit.fingerprint` and `factories` from
  `config.datastore`. An instance, as the loader requires.
- A module docstring saying what the module holds and why it is here (D3: what goes into a
  fingerprint is the client's), in the register of `config/datastore.py`'s.
- Importing it must load nothing more than `config.datastore` already does: `DatastoreKitFingerprinter`
  imports DatastoreKit only inside its methods.

07 adds the Q5 pin to this module, beside the format. Leave room; add nothing for it.

---

## 2. The rewrites in live code (A2)

**The rule** is C6 with C16's boundary: every `RunRegistry` import, string literal, comment and
command in a tracked file outside `prompts/` is rewritten, **except** `.md` and `.txt` files under
`docs/`, which are dated records and recorded output, and `docs/OPEN_ISSUES.md`'s rows, which are
the boards' records. `CLAUDE.md` may keep one sentence naming `RunRegistry/` as the registry's
former home (§6). The test is `git grep -nP 'RunRegistry(?!Kit)'` with those exclusions (§7.6).

**2.1 The imports and names**, the checklist's item 4 table, measured again (13 statements in 10
files at `14bcac0`):
- `import RunRegistry` → `import runregistrykit`; `import RunRegistry.stores [as x]` →
  `import runregistrykit.stores [as x]`; every `RunRegistry.<name>` use with it.
- **Moved to DatastoreKit:** `fingerprint_of`, `compare_fingerprints`, `listing_lines`, from
  `datastorekit.store_fingerprint`. **`fingerprint_of` now requires the format**:
  `fingerprint_of(inventory, FINGERPRINT_FORMAT[, taken])`, with `FINGERPRINT_FORMAT` from
  `config.fingerprint`. A call that passed `taken` second positionally passes the format first.
- **Deleted, now SGK's:** `FINGERPRINT_FORMAT` → `config.fingerprint.FINGERPRINT_FORMAT`.
- **Renamed:** `RunRegistry._slug` → `runregistrykit.slugify` (`stage_counts.py:37`, `:38`).
- **Kept:** `begin`, `read_json`, `now_iso`, `pid_alive`, `liveness`, `TERMINAL_STATES`,
  `DEFAULT_STALE_AFTER`, `list_runs`, and `stores.fingerprint_store`, `copy_store`,
  `create_sidecar`, `read_sidecar`, `sidecar_path`: the same parameters.
- `docs/a3-v2-readiness/sigterm_handler_probe.py:210` patches `registry_stores.fingerprint_store`
  by assignment. It still takes effect at `v0.1.0` (checklist item 4); the package promises no patch
  target. Rewrite the import and leave the patch; record it in the log's observations.

**2.2 The two seams** (C12). Two tests redirected a driver's runs root by patching a name `v0.1.0`
deletes (`DEFAULT_ROOT`). Each driver gains one keyword, with a default that changes nothing:
- `docs/gktk-remedial/scoped_pipeline_run.py`: `register(args, main_args, k_sample, database,
  runs_root=None)`, passing `root=runs_root` to `runregistrykit.begin`. `main()`'s call is
  unchanged. This is A3 v2's driver: the default path, `root=None`, is the config's `runs_root`,
  which is `var/runs`, exactly as before.
- `docs/handover/quadsource_atol_sweep.py`: `prepare(force=False, runs_root=None)`, passing
  `runs_root=runs_root` to `runregistrykit.stores.copy_store`. Its callers are unchanged.

Each signature change is `STRUCTURALLY REQUIRED` in the log. Nothing else in either driver changes
but §2.1's rewrites and §2.3's literals.

**2.3 String literals, comments and docstrings** naming `RunRegistry` in the files §2.1 touches,
and in the others the checklist's appendix item 5 lists (18 literals at `14bcac0`), and in
`docs/a3-v2-readiness/interrupter_build_dummy_test.sh:6`:
- **The printed commands**, three, `PYTHONPATH=. ./venv/bin/python -m RunRegistry list`
  (`scoped_pipeline_run.py:580`, `quadsource_atol_sweep.py:595`, `:906`) →
  `./venv/bin/python -m runregistrykit list`.
- **The refusal** at `scoped_pipeline_run.py:526-532`: `import runregistrykit` cannot fail for want
  of `PYTHONPATH=.` once the package is installed. Reword its message to say the package is missing
  from the venv and to install `requirements.txt`; keep the `try`.
- **`--register`'s help** (`:611`) and the module and function docstrings name `runregistrykit`
  where they named the module. A docstring that names a file under `RunRegistry/tests/` names its
  new place (`quadsource_atol_sweep.py:419` → `RunDrivers/tests/test_prune_on_resume.py`).
  `RunRegistry store retire` in prose becomes `runregistrykit store retire`.
- History stays history: a phrase such as "`store-retirement` prompt 03" is not changed.

---

## 3. The 101 tests that stay SGK's (A3)

They are under `RunRegistry/tests/` at `14bcac0`: five modules (97 tests), and `TestGolden` (3) and
`TestDrivers` (1) of `test_store_fingerprint.py` (C5). Count them again by discovery before moving
anything; stop if the counts are not 28, 26, 15, 18, 10, 3 and 1.

**Move with `git mv`** where a file moves whole, so that history follows it.

**3.1 `RunDrivers/tests/`**, a new root (C5), with `RunDrivers/__init__.py` and
`RunDrivers/tests/__init__.py` (empty, as the other roots'):
- `test_build_rehearsal_scope.py` (28), `test_killed_handler.py` (26), `test_pipeline_adoption.py`
  (15), `test_prune_on_resume.py` (18), `test_quadsource_atol_sweep_prepare.py` (10), moved whole;
- `test_finish_calls.py`, new, holding `TestDrivers` (1) with `_finish_calls` and whatever else of
  `test_store_fingerprint.py`'s module level it uses, and its own `REPO_ROOT` computed from
  `__file__` (`parents[2]`, as `test_pipeline_adoption.py:29` does), replacing
  `Path(RunRegistry.REPO_ROOT)`;
- `registry_fixtures.py`, new (C14): **SGK's own copy of what the staying tests use** of
  `RunRegistry/tests/store_fixtures.StoreTestCase`, taken from the frozen tree. Copy only what is
  used: by measurement at `14bcac0` that is `setUp` (`self.top`, resolved, and `self.root`, a runs
  root inside it) and `state()`, with `tree_state` from `StoreFixtures.shard_store_fixtures`, and,
  for §3.2's new test, `RegistryTestCase.begin` (a run in `self.root`), calling
  `runregistrykit.begin`. Its docstring says where it came from.

**3.2 The changes each moved test needs**, every one named here, so that README §5 rule 7 is met.
Each is from the checklist's item 7, measured again:
- **Every module:** `RunRegistry` → `runregistrykit`, and `from RunRegistry.tests.<m> import …` →
  `from RunDrivers.tests.<m> import …` (`test_build_rehearsal_scope.py:39`,
  `test_killed_handler.py:27`).
- **`test_build_rehearsal_scope.py:317`**: the asserted spelling `"RunRegistry.begin"` →
  `"runregistrykit.begin"`, with §2.1's rewrite of `quadsource_atol_sweep.py:556`.
- **`test_pipeline_adoption.py`**:
  - `:146`, `:158`, the module name in `[sys.executable, "-m", "RunRegistry", "list", …]` →
    `"runregistrykit"` (run with `cwd=REPO_ROOT`, it finds SGK's config; `--root` names the runs
    root);
  - `:278`, `root = RunRegistry.DEFAULT_ROOT` →
    `runregistrykit.find_client_root(client_root=REPO_ROOT).runs_root` (the same directory, read
    the same way: a listing, before and after);
  - `:330`, `mock.patch.object(RunRegistry, "DEFAULT_ROOT", self.root)` around
    `self.driver.register(…)` → no patch; `self.driver.register(…, runs_root=self.root)` (§2.2);
  - `test_the_datastore_is_registered_as_results_and_not_as_a_ledger` gains **one assertion**: the
    run directory's parent is `self.root`. The patch guaranteed it without saying so; the keyword
    does not, and without the assertion a seam that forgot to forward the root would write a real
    run under `var/runs/` and pass (§8, breakage 5).
- **`test_quadsource_atol_sweep_prepare.py`**:
  - `:27`, `StoreTestCase` from `RunDrivers.tests.registry_fixtures` (§3.1);
  - `:73-75`, the patch of `stores.DEFAULT_ROOT` → removed; every `self.sweep.prepare(…)` call
    passes `runs_root=self.root` (§2.2). Measure the calls (five at `14bcac0`, `:118`, `:151`,
    `:168`, `:182`, `:187`);
  - `:134`, `RunRegistry._repo_path(baseline)`, private, whose signature changed. The expected
    `copied_from.datastore` is what the package records for a primary outside the record root: read
    `runregistrykit._repo_path` at `v0.1.0` to see it, and state the expectation without calling the
    private function (the checklist's advice: `os.path.abspath(baseline)`, if that is what it gives
    for a resolved temporary path). Record the choice;
  - the docstrings naming `RunRegistry.stores.DEFAULT_ROOT` say what the test now does;
  - **one new test** (C12), `test_a_running_run_in_the_given_runs_root_blocks_the_copy`: a run
    begun in `self.root`, still `running`, with `results` the baseline, makes `prepare(runs_root=
    self.root)` refuse, and nothing is written. No existing test reaches the runs root at all: the
    patch only kept `copy_store`'s read of it away from `var/runs/`. Without this test the seam
    is untested (§8, breakage 6). Its run is begun in the temporary root, with `script` or
    `client_root` such that it finds SGK's config, and never touches `var/`.
- **`test_killed_handler.py`**: `:333`, `:432` patch `Run._fingerprint_results`, a private method
  that is still there at `v0.1.0`. Leave them; record them in the observations with §2.1's probe.
- **`TestDrivers`**: nothing but its `REPO_ROOT`.

**3.3 `ObjectFactories/tests/test_store_fingerprint_golden.py`**, new (C5, C13, C18), holding:
- `_FullStores`, `without_taken` and the module-level names they use, moved from
  `test_store_fingerprint.py`, with `fingerprint_of` from `datastorekit.store_fingerprint` and
  `FINGERPRINT_FORMAT` from `config.fingerprint`: `:147` and `:162` become
  `fingerprint_of(…, FINGERPRINT_FORMAT)`;
- **`TestGolden`** (3), with these changes, each named:
  - `GOLDEN` names the new golden file (§3.4);
  - `test_the_golden_fingerprint` also asserts `golden["encoding_format"] ==
    datastorekit.store_fingerprint.ENCODING_FORMAT`;
  - `test_the_shape`: the key set gains `"encoding_format"`; `:374`'s
    `stores._fingerprint_problems(…)` becomes `runregistrykit.stores._fingerprint_problems(…)`, a
    private name that is present at `v0.1.0` with the same parameter (checklist item 7). Record it
    in the observations;
  - `test_taken_is_outside_every_digest`: **`:378`, `fingerprint_of(self.baseline_inventory,
    taken)` → `fingerprint_of(self.baseline_inventory, FINGERPRINT_FORMAT, taken)`**. As written it
    binds and puts `taken` on the format, which DatastoreKit refuses; the checklist flags it by
    reading. `compare_fingerprints` from DatastoreKit.
- **`TestConfigFingerprint`** (3), new (C18), the tests of `config/fingerprint.py` and of the
  config's naming of it:
  1. `test_the_config_names_the_fingerprinter`: `find_client_root(client_root=REPO_ROOT)`'s
     `stores` is `{"fingerprinter": "config.fingerprint:fingerprinter"}`; `load_fingerprinter` of it
     is `config.fingerprint.fingerprinter`, a `DatastoreKitFingerprinter`, whose
     `fingerprint_format` is `FINGERPRINT_FORMAT`, which is `6`, and whose `factories` is
     `config.datastore.factories`;
  2. `test_the_fingerprinter_takes_the_golden_fingerprint`: `fingerprinter.take(primary, None)` on
     a `build_full_store()` gives a fingerprint whose `encoding_format` is DatastoreKit's
     `ENCODING_FORMAT`, which equals `fingerprint_of(read_inventory(primary, factories),
     FINGERPRINT_FORMAT)` exactly, and which equals the golden file with `taken` aside. This is
     README §2.2 06's acceptance 5;
  3. `test_the_new_golden_is_the_old_one_with_its_encoding`: the new golden file with
     `encoding_format` removed equals `full_store_fingerprint_datastore-integrity-09c.json` exactly,
     digest `6576b85d72e8acb547bca59f9fd0214c3e96ef45a3cbc41298a4dff85192e759`.
  These read `runregistrykit.toml` at the root and import `config.fingerprint`; they open nothing
  under `var/` (README §5 rule 10).

**3.4 The golden files** (C13). `git mv` the six `RunRegistry/tests/data/full_store_fingerprint*.json`
to `ObjectFactories/tests/data/`, unchanged. Then **capture one new golden file beside them**,
`full_store_fingerprint_runregistry-extraction-06.json`, through `config.fingerprint.fingerprinter`
on `build_full_store()`, `taken` `None`, written in `09c`'s layout (compare the two files' bytes but
for the added key: same indentation, key order as `fingerprint_of` returns it, trailing newline).
The convention the format history states holds: a new golden beside the old ones, never over them.
The format is not bumped: the digest has not moved.

**3.5 What is left of `RunRegistry/tests/`** is the package's 185 tests, now RunRegistryKit's
(its suite, 276 at `v0.1.0`). They are deleted with the directory (§4).

---

## 4. `RunRegistry/` deleted

`git rm -r RunRegistry/`, after §3's moves, so that nothing under it remains tracked or on disk
(remove its `__pycache__` too). The freeze ends with this prompt; the frozen-tree table (board §2.1)
stays as the record G1 was checked against.

---

## 5. `fingerprint_both_ways.py` rewritten (README §4)

04's script compares SGK's `fingerprint_of` with the package's on one inventory. SGK's is deleted
here, so the script becomes the comparison 08 makes **across stores**, as README §4 specifies:

```text
docs/runregistry-extraction/fingerprint_both_ways.py <primary>... --against <run dir>...
```

- Each store's fingerprint is taken through `config.fingerprint.fingerprinter.take(primary, None)`,
  read-only, with no Ray.
- Each `--against` is a run directory whose `status.json` carries a `fingerprint`, or that
  `status.json` (as `compare_with_recorded.py` reads it).
- **The checks, for each store and each recorded fingerprint**, each printed with its result:
  - the store's `encoding_format` is `datastorekit.store_fingerprint.ENCODING_FORMAT`;
  - the recorded fingerprint lacks `encoding_format`;
  - **field by field, `encoding_format` and `taken` set aside**: the same keys in the same order,
    `fingerprint_format`, the overall `digest`, `problems`, and for every class in the same order
    its `count`, `digest` and every tag set's `tags`, `count` and `digest`. Every differing path is
    printed with both values;
  - the overall digest is `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
  - the package's `compare_fingerprints(recorded, current)` is exactly one entry, of kind
    `encoding` (`[00-recorded-fingerprints-lack-the-encoding-format]`, measured, not a failure);
  - **against the sidecar**, which the package wrote: `compare_fingerprints(sidecar's, current)` is
    empty.
- It exits 0 if and only if every check holds for every pair, 1 otherwise, 2 on a usage error. It
  writes nothing. It imports `runregistrykit` and `config.fingerprint`, never `RunRegistry`.
- A `--expect-digest` flag, defaulting to `39809dca…`, so that the script can be exercised on a
  fixture store whose digest is `6576b85d…` (§8). The default is what 08 uses.
- Its docstring says what it compares, why the comparison is field by field, and that 04's form of
  it is at `ec61569`.

**How to exercise it here**, with no live store: in your scratch directory, build a full store with
`StoreFixtures.real_store_fixtures.build_full_store`; give it a sidecar and a written fingerprint
with `./venv/bin/python -m runregistrykit store create` and `store fingerprint --write` (the
sidecar lands beside the scratch primary; nothing under `var/` is touched); and make a scratch "run
directory" whose `status.json` holds `{"fingerprint": <the 09c golden>}`, the fixture's digest as
SGK's code recorded it. Then run the script with `--expect-digest 6576b85d…`. It must exit 0. Save
the output as `measure/out/06-both-ways.txt`. §8's breakages run against the same scratch set.

---

## 6. `CLAUDE.md` (A4)

Three sections change, and one goes. Keep each short, in the file's register.
- **"Long-running jobs — the run registry"**: the six rules stay, binding, as they are; the opening
  says that `runregistrykit` (RunRegistryKit, pinned at `v0.1.0`, configured by
  `runregistrykit.toml` at the root and the fingerprinter in `config/fingerprint.py`) records that a
  job exists; rule 1's command is `./venv/bin/python -m runregistrykit list`; a pointer to
  RunRegistryKit's `README.md`, "The rules", which tells each rule with its failure. The reference
  to `prompts/run-registry/README.md` §0 stays.
- **"`RunRegistry/` is frozen"**: removed.
- **"The datastore layer is `datastorekit`; the work queue is `raypoolkit`"**: names the third
  package, RunRegistryKit (`/Users/ds283/Documents/Code/RunRegistryKit`,
  https://github.com/ds283/RunRegistryKit), pinned at `v0.1.0` with the extra `[stores]`; "Neither
  is this repository's code" becomes true of all three; its imports
  (`import runregistrykit`, `runregistrykit.stores`); its CLI (`python -m runregistrykit`). One
  sentence may say that until this prompt the registry was `RunRegistry/`, and that campaign
  records from before then use that name and are left as written, as the section already says of
  the factories' old paths.
- **"Repository mechanics"**: the test roots list `RunDrivers/tests` in place of
  `RunRegistry/tests`; still seven.

The rehearsal-lock section is unchanged.

---

## 7. What to verify

**7.1 The registry, before and after** (acceptance 3). **Before any change**, with `RunRegistry`:
`python -m RunRegistry list` and `store show` on every sidecar under `var/datastores/`
(`find var/datastores -name '*.manifest.json' | sort`, 18 at E2), saved as
`measure/out/06-before.txt`, in 01's form. **After**, at the staged tree, the same through
`./venv/bin/python -m runregistrykit list` and `… store show`, saved as `06-after.txt`.
- `list`: the same runs (37 at E2: 01's 29 and R0's and R1's 8), with the same ids, states and
  purposes, compared as data, not bytes (the age column moves). Each of 01's 29 as
  `measure/out/01-list.txt` has it.
- `store show`: every sidecar read, exit 0, kind and fields as before. A line that differs only in
  naming the tool, or in how a path is printed, is listed and explained; any other difference stops
  the prompt.

**7.2 The probe commit** (C15). When the work is finished and staged (and `black`-clean), make a
commit object that no branch names: `tree=$(git write-tree)` and `probe=$(git commit-tree "$tree" -p
HEAD -m "runregistry-extraction 06 probe; not a branch")`. Record both hashes. §7.3 and §7.4 run at
it. After the real commit, `git diff --stat "$probe" HEAD` must name only files under
`prompts/runregistry-extraction/` and `docs/OPEN_ISSUES.md` (the records written after the probe).

**7.3 The checklist's measure** (its item 9, 1). From RunRegistryKit's root, with its venv:
`./venv/bin/python docs/adoption/measure_client_registry.py sgk --commit <probe>`, saved whole as
`measure/out/06-measure.txt`. It must report **0** imports of a `RunRegistry` module in live code
and none in the staying tests. It reads this repository through git only (README §5 rule 12).

**7.4 The worktree check** (acceptance 4; Q3; C15). A new script,
`docs/runregistry-extraction/worktree_run_check.py`, committed, which:
- calls `runregistrykit.begin(campaign="runregistry-extraction", prompt="06", slug=<--slug,
  default "worktree-check">, purpose=…, script=__file__)`, the purpose saying it is this check, at
  the probe, and that its record is meant to outlive the worktree;
- computes the main checkout as the parent of `git rev-parse --git-common-dir` (from the script's
  directory), and checks that the run directory lies directly under that checkout's `var/runs/`,
  and that the manifest's `git_head` is the worktree's `HEAD`, not the main checkout's;
- ends the run with `finish("done", exit_code=0)`, prints its id and path, and exits 0 if and only
  if every check holds.

Then:
1. `git worktree add --detach <your scratch directory>/sgk-06-worktree <probe>`;
2. from inside the worktree, `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python
   docs/runregistry-extraction/worktree_run_check.py`: the main checkout's venv, by absolute path
   (a worktree has no `venv/`; the script needs nothing but `runregistrykit`);
3. `git worktree remove <path>`, then `git worktree prune`;
4. from the main checkout, `./venv/bin/python -m runregistrykit list` shows the run, `done`, after
   its worktree is gone.

Save every command and its output as `measure/out/06-worktree.txt`. The manifest records `git_head`
the probe and `git_dirty` `false` while the main checkout is dirty: that difference is the point.
**The run is a record: never delete it.** With §8's breakage 8 it is one of exactly two runs this
prompt leaves under `var/runs/`, which 08's acceptance 4 counts.

**7.5 The roots.** All seven, serially, by README §8's command, at the staged tree. Expected:
AdaptiveLevin 32, ComputeTargets 693, CosmologyModels 39, LiouvilleGreen 148 (1 skipped),
**ObjectFactories 277** (271, `TestGolden`'s 3 and `TestConfigFingerprint`'s 3), **RunDrivers 99**
(97, `TestDrivers`' 1 and §3.2's new sweep test), tools 91. `RunRegistry/tests` is gone. Save as `measure/out/06-roots.txt`.

**7.6 Nothing names `RunRegistry`** (acceptance 1):

```bash
git grep -nP 'RunRegistry(?!Kit)' -- ':!prompts' ':!docs/*.md' ':!docs/*.txt'
```

(a pathspec `*` matches across directories, so the two exclusions cover every `.md` and `.txt`
under `docs/`). At `14bcac0`, outside `RunRegistry/`, it names 13 files: `CLAUDE.md`, the policy-key
test and eleven scripts under `docs/`. After this prompt it prints only `CLAUDE.md`'s one sentence,
if §6 kept it, and nothing else; and `import RunRegistry` from the root fails with
`ModuleNotFoundError`.

**7.7 The issues** (acceptance 6, 7). Close the three of the header on the board's §4, each with its
evidence: the worktree run (§7.4); a fingerprint carrying `encoding_format` (§3.3's test 2, §5's
run); and, for the recorded fingerprints, that every sidecar under `var/datastores/` is a tombstone
(§7.1), the golden data handled by §3.4, and the run directories' records left as records, which 08
compares field by field.

**7.8 The commands for README §4.1**, from now on, stated in the log for 08: `./venv/bin/python -m
runregistrykit list`; `… -m runregistrykit store show | create | fingerprint [--write]`; how
`package_provenance` is now called (`runregistrykit.package_provenance(names)` takes the names);
and each of `docs/dk-and-rpk-migration/rehearsal/`'s scripts as rewritten. Run each once, read-only,
on a tombstone or the scratch store, so that 08 does not meet a broken command first.

---

## 8. Deliberate breakage (README §5 rule 8)

Each applied to the working tree or a scratch copy, run, shown failing, and reverted. Never
committed. The log records each as a diff, exactly as applied, with the failing test or the exit
status and the line that names the failure.
1. `config/fingerprint.py`: `FINGERPRINT_FORMAT = 5` → `TestConfigFingerprint` test 1 and
   `TestGolden.test_the_golden_fingerprint` fail.
2. `config/fingerprint.py`: the fingerprinter made over a copy of `factories` with one class
   removed → test 2 fails.
3. The new golden file with one class's `digest` altered → test 3 and
   `test_the_golden_fingerprint` fail, naming the class.
4. `runregistrykit.toml`: `fingerprinter = "config.fingerprint:DatastoreKitFingerprinter"` (the
   class) → test 1 fails with the loader's refusal.
5. `scoped_pipeline_run.register`: `root=os.path.join(runs_root, "elsewhere") if runs_root else
   None` in place of `root=runs_root` → the registration test's new assertion fails. The mutation
   keeps the run inside the temporary directory; **do not** apply the plainer mutation (the keyword
   dropped), which would write a real run under `var/runs/`.
6. `quadsource_atol_sweep.prepare`: `runs_root=runs_root` dropped → the new sweep test fails
   (`copy_store` then reads the real `var/runs/`, read-only, finds no running run naming the
   baseline, and copies).
7. `fingerprint_both_ways.py`: `encoding_format` not set aside → exit 1, naming it; one tag set's
   `count` in the recorded fingerprint incremented → exit 1, naming the class and the tags.
8. `worktree_run_check.py`: the expected root computed from `git rev-parse --show-toplevel` (the
   worktree's) → exit 1. Run the mutated copy with `--slug worktree-check-breakage`, from a second
   scratch worktree at the probe, and remove that worktree. Its run is a record too, and it ends
   `done` or `failed` as the mutated script leaves it; say which. No other breakage writes a run
   outside a temporary directory.

If a breakage does not fail as stated, the test or the script is not testing what it claims: stop.

---

## 9. Acceptance

1. **No `RunRegistry` remains** outside `prompts/` and dated `docs/` prose (§7.6), and the
   checklist's measure reports 0 imports at the probe (§7.3).
2. **The roots pass** at §7.5's counts; a failure among README §8's four known wall-clock tests is
   re-run alone and recorded.
3. **`runregistrykit list` and `store show`** read every run and sidecar as `RunRegistry` did
   (§7.1), and 01's 29 runs as 01's baseline has them.
4. **A run begun from a linked worktree is recorded in the main checkout's `var/runs/`**, with the
   worktree's `git_head`, and is listed after the worktree is gone (§7.4).
5. **A fingerprint taken through the new path carries `encoding_format`**, and equals
   `fingerprint_of`'s on the fixture store (§3.3, test 2).
6. **The three issues are closed** on the board (§7.7), with `docs/OPEN_ISSUES.md` in the same
   commit.
7. **The breakages of §8** each fail as stated.
8. `RunRegistry/` is gone; `git diff --stat "$probe" HEAD` is the records alone (§7.2).

## 10. Stop conditions — stop and ask

- A count in §3 or §7.5 differs from this prompt's, or a moved test fails other than by README §8's
  known tests.
- The checklist and the tree disagree on something this prompt relies on (a line, a name, a
  signature), and the difference changes what to do.
- A test cannot be moved without a change this prompt does not name (README §5 rule 7).
- `store show` or `list` reads a record differently in substance (§7.1).
- The worktree run lands anywhere but the main checkout's `var/runs/`.
- A package does something its README or the checklist says it does not. That is RunRegistryKit's
  or DatastoreKit's defect (README §5 rule 11): record it, change nothing to work around it, stop.
- Anything would write under `var/` other than §7.4's run and §8's breakage-8 run.

## 11. What this prompt does not do

- It adds no guard (07), runs no build, starts no Ray, and does not take the rehearsal lock.
- It changes nothing in what is stored; no format bump.
- It does not rewrite `.md` or `.txt` files under `docs/`, `docs/OPEN_ISSUES.md`'s rows, or
  anything under `prompts/` but this campaign's records.
- It does not hand SGK's open registry issues to RunRegistryKit
  (`[00-open-issues-about-the-registry-code-stay-on-sgk-boards]`, 09's).
- It changes nothing in RunRegistryKit, DatastoreKit or RayPoolKit (README §5 rule 12).

## 12. The log and the board

**The files you may change**: `runregistrykit.toml`, `config/fingerprint.py`; the twelve files
other than `CLAUDE.md` that §7.6's grep names at `14bcac0` (`ObjectFactories/tests/test_quadsource_policy_key.py`
only at `:411` and its three calls, §2.1; the eleven `docs/` scripts as §2 says);
`RunDrivers/`; `ObjectFactories/tests/test_store_fingerprint_golden.py` and
`ObjectFactories/tests/data/`; `RunRegistry/` (deleted); `CLAUDE.md`; `docs/runregistry-extraction/`
(`fingerprint_both_ways.py`, `worktree_run_check.py`); under `measure/out/`, the `06-*` files; the
log; this campaign's board; and `docs/OPEN_ISSUES.md`. Every `.py` `black`-clean.

**The log**, at `logs/06-adopt-runregistrykit.md`, in README §5.1's form, with: every name
rewritten, by file; every test change of §3.2 and §3.3, each classified; §7.8's commands; §8's
breakages as diffs; and in the observations, the private names the staying tests and probes still
reach (§2.1, §3.2, §3.3).

**The board:** the header, the §1 row for 06, items A1–A5, the three issues moved to §4 with their
evidence, and `### 5.12 After prompt 06 (<date>)`; `docs/OPEN_ISSUES.md` §1.21's three rows deleted,
its prose, count and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only (with `git rm` and `git mv` for
the moves), after showing `git diff --cached --name-only` and `--stat`.
