# Orchestrator — prompt 06, adopt `runregistrykit`

Read [`../README.md`](../README.md) first: §2.2 (06's charter), §4 (06's lines and how 08 uses the
rewritten script), §4.2, §5 (rules 4, 7–14) and §6.2 (C5–C8, C12–C18). Then the board's §3 (the
three issues 06 closes), §5.10 (E2) and §5.11 (05 and its review). The conventions are
[`prompt-01.md`](prompt-01.md)'s §0 last list, carried over. [`prompt-02a.md`](prompt-02a.md) is
the model for a prompt that adds code and breakages; what differs is said here.

**You do not write code.** You may:
- run the suites, `python -m RunRegistry list` and `store show` (before the agent's commit), and
  `python -m runregistrykit list` and `store show` (after it);
- read RunRegistryKit's repository with `git`, and run its `measure_client_registry.py` from its
  root with its venv, never changing it;
- apply the log's breakage diffs to a scratchpad copy, or to the working tree and revert them, and
  run a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`06-adopt-runregistrykit.md`](../06-adopt-runregistrykit.md)
**Board items:** A1–A5 · **Closes:** `[00-a-run-launched-from-a-worktree-records-into-the-worktree]`,
`[00-the-fingerprint-encoding-has-no-version-of-its-own]` (SGK's half),
`[00-recorded-fingerprints-lack-the-encoding-format]` · **Narrows:** nothing · **Opens:** anything
found · **Model:** Opus, as the prompt recommends.

**Gate:**
- 05 landed (`3dcab1b`) and was reviewed (`29a3f2d`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (anything outside
  `prompts/` and `docs/`).
- Nothing is `running` in the registry; the rehearsal lock is free; `ray-processes` lists nothing;
  `git worktree list` shows this checkout alone.
- The venv holds `runregistrykit` 0.1.0 at `9f8ebe43…`, `datastorekit` 0.4.0 at `c8570e84…`,
  `raypoolkit` 0.2.0 at `eaef0ce6…`, none editable.
- One prompt at a time in this checkout (README §4.2).

## 0. What makes this prompt unusual

**It is the swap, and it touches more files than any prompt of the campaign.** Two new client
files, thirteen files rewritten, a new test root with six modules and a fixtures module, a new
golden module and seven golden files, `RunRegistry/` deleted, `CLAUDE.md`'s three sections, two new
`docs/` scripts. Each change is small. The care is in four places, and so is the review's:
- **the moved tests** change only where the prompt names a change (README §5 rule 7);
- **the two seams** forward the runs root, and a test of each catches it not doing so;
- **the golden file** is `09c`'s with `encoding_format` added, and the digest stays `6576b85d…`;
- **the worktree check** records into the main checkout, from a commit no branch names.

**It writes two records under `var/runs/` that are meant to stay.** §7.4's worktree run and §8's
breakage-8 run are the only things this prompt may write under `var/`. They are records: never
deleted, and 08's acceptance 4 counts them.

**It is the last prompt that can read the registry with `RunRegistry`.** §7.1's "before" reading
must be taken before anything changes; once `RunRegistry/` is deleted it cannot be retaken except
from a checkout of an older commit.

**Rulings on the prompt** (the orchestrator's, made in writing this note; the user may reverse any).
Pass them on; the agent logs each as the classification given.
1. **The format history's `RunRegistry` survives §7.6.** §1.2 moves the comment at
   `RunRegistry/stores.py:194-205` *verbatim* (C17), and its line `:197` names
   `` `RunRegistry/tests/data/full_store_fingerprint*.json` ``. §7.6 says the grep prints nothing
   but `CLAUDE.md`'s one sentence. The two conflict, and C17 is explicit: the history is a record,
   and the dated paragraph that follows it exists to say where the golden files are now. So the
   history moves verbatim, and §7.6's expected output is `CLAUDE.md`'s one sentence (if kept) **and
   exactly one line of `config/fingerprint.py`**, the history's. Log it as `STRUCTURALLY REQUIRED`,
   citing this ruling. Any other line is a miss.
2. **`take()` returns a pair.** `DatastoreKitFingerprinter.take(primary, taken)` returns
   `(fingerprint, inventory)` (`runregistrykit/fingerprint.py:121-129`). §3.3's test 2 and §5's
   script take `[0]`. Not a deviation.
3. **`registry_fixtures.py` copies from two frozen files.** `StoreTestCase` is in
   `RunRegistry/tests/store_fixtures.py:114` and overrides `setUp`; `RegistryTestCase.begin` is in
   `RunRegistry/tests/test_run_registry.py:113-125`. Its docstring names both, at `a45aefc`.
4. **Breakages that fail more than the prompt names are not a stop; fewer is.** By reading the tests
   as specified: breakage 1 also fails `TestConfigFingerprint` test 2 (the golden comparison);
   breakage 2 also fails test 1 (its `factories` assertion); breakage 3 also fails test 2. The log
   lists every failing test of each breakage, not only the named ones. A named test that does *not*
   fail is §8's stop.
5. **The worktree check never leaves a run `running`.** It finishes its run on every path: `done`/0
   when every check holds, and `failed`/1 otherwise, including when a check raises (`try`/`finally`).
   `IMPLEMENTATION CHOICE`. So breakage 8's run ends `failed`.
6. **Breakage 8's mutation lives in the second worktree** and is reverted there (`git -C <wt>
   checkout -- <file>`) before `git worktree remove`, without `--force`. Its run records
   `git_dirty: true`, as it should.
7. **`RunRegistry/` is gone from disk, not only from git.** A directory left holding
   `__pycache__/` imports as a namespace package, and `import RunRegistry` then succeeds. §7.6's
   `ModuleNotFoundError` is the test.
8. **`CLAUDE.md`'s DatastoreKit pin is stale.** Its package section says `datastorekit` is pinned
   at `v0.2.2` (`CLAUDE.md:51`); 03 moved it to `v0.4.0` and did not correct it, and no review
   noticed. §6 rewrites that section to state the pins, so it states all three truly: `datastorekit`
   `v0.4.0`. `IMPLEMENTATION CHOICE`, citing this ruling.
9. **Order of work.** §7.1 "before" first. Then §1–§6, staged and `black`-clean. Then §7.1
   "after" and §7.5 at the staged tree, **before** the probe's runs exist, so that `06-after.txt`
   lists 37 runs. Then the probe, §7.3, §7.4 and §8. Every breakage is reverted: after §8,
   `git diff` (working tree against the index) is empty and `git write-tree` equals the probe's
   tree. Then the records, and the commit.

**What moves on purpose:**
- **Added:** `runregistrykit.toml`; `config/fingerprint.py`; `RunDrivers/__init__.py`,
  `RunDrivers/tests/__init__.py`, `registry_fixtures.py`, `test_finish_calls.py`;
  `ObjectFactories/tests/test_store_fingerprint_golden.py`;
  `ObjectFactories/tests/data/full_store_fingerprint_runregistry-extraction-06.json`;
  `docs/runregistry-extraction/worktree_run_check.py`; `measure/out/06-*.txt`; the log.
- **Renamed** (`git mv`, so `R` in `--name-status`): the five driver test modules to
  `RunDrivers/tests/`, and the six golden files to `ObjectFactories/tests/data/`, unchanged.
- **Deleted:** every other file under `RunRegistry/`: 25 tracked at `a45aefc`, less the 11 moved,
  so 14 `D` lines (if git's rename detection pairs `test_store_fingerprint.py` with one of its
  successors, read that `R` as the split it is).
- **Modified:** `CLAUDE.md`; the twelve other files the §7.6 grep names at `HEAD`
  (`ObjectFactories/tests/test_quadsource_policy_key.py` only at `:411` and its calls; eleven
  `docs/` scripts, `fingerprint_both_ways.py` among them, rewritten whole by §5); the board;
  `docs/OPEN_ISSUES.md`.
- **Issues:** three closed, none expected opened; the index 108 → **105**, and §1.21's prose
  corrected.

**The facts, checked by the orchestrator on 2026-10-15 at `29a3f2d`** (`git diff --stat 14bcac0
29a3f2d -- . ':!prompts'` is `requirements.txt` and `docs/OPEN_ISSUES.md` alone, so the
checklist's lines are current). Pass them on.
- **The tree.** Clean; `git worktree list` shows this checkout alone. The registry: 37 runs, 35
  finished and 2 `unknown`, none `running` or stale; 18 sidecars under `var/datastores/`, all
  tombstones, and no `.sqlite`. Lock free; no Ray process of this repository.
- **The counts.** By discovery: 28, 26, 15, 18, 10, `TestGolden` 3 and `TestDrivers` 1, as §3
  requires.
- **The grep.** `git grep -lP 'RunRegistry(?!Kit)' -- ':!prompts' ':!docs/*.md' ':!docs/*.txt'
  ':!RunRegistry'` names the 13 files of §7.6, with 61 matching lines: `CLAUDE.md` 7,
  `test_quadsource_policy_key.py` 1, `fingerprint_time_probe.py` 4,
  `interrupter_build_dummy_test.sh` 1, `sigterm_handler_probe.py` 4, `compare_stores.py` 1,
  `interrupt_when.py` 12, `compare_with_recorded.py` 2, `stage_counts.py` 4, `store_soundness.py` 1,
  `scoped_pipeline_run.py` 7, `quadsource_atol_sweep.py` 13, `fingerprint_both_ways.py` 4.
- **Every line the prompt cites** was read at `29a3f2d` and says what the prompt says:
  `stores.py:194-205`; `scoped_pipeline_run.py:526-532`, `:580`, `:611`;
  `quadsource_atol_sweep.py:419`, `:556`, `:595`, `:906`; `test_pipeline_adoption.py:29`, `:146`,
  `:158`, `:278`, `:330`; `test_quadsource_atol_sweep_prepare.py:27`, `:73-75`, `:134` and the five
  `prepare(` calls at `:118`, `:151`, `:168`, `:182`, `:187`; `test_killed_handler.py:27`, `:333`,
  `:432`; `test_build_rehearsal_scope.py:39`, `:317`; `test_store_fingerprint.py:147`, `:162`,
  `:374`, `:378`; `stage_counts.py:37-38`; `test_quadsource_policy_key.py:411`;
  `sigterm_handler_probe.py:210`; `interrupter_build_dummy_test.sh:6`.
- **The package, installed** (`venv/lib/python3.12/site-packages/runregistrykit/`):
  - `begin(..., script=None, ..., root=None, when=None, *, client_root=None)` walks up from
    `script`'s directory, else the working directory, to `runregistrykit.toml`; `root` overrides
    the runs root; `git_head` and `git_dirty` are the start directory's. `packages` is
    `runregistrykit` first, then the config's `distributions` (`_recorded_names`).
  - `find_client_root(start=None, client_root=None)`; for a linked worktree the record root is the
    main checkout (`client._record_root`, from `--git-common-dir`). `ClientRoot.stores` is the
    `[stores]` table as read.
  - `load_fingerprinter(client)` refuses a class with "is the class …, not an instance of it"
    (breakage 4). For that refusal to be what breakage 4 meets, `config/fingerprint.py` must import
    the name `DatastoreKitFingerprinter` into its namespace (`from runregistrykit.fingerprint
    import DatastoreKitFingerprinter`); otherwise the loader says "has no attribute".
  - `stores.copy_store(src, dst, purpose, runs_root=None, *, client_root=None)` records
    `copied_from.datastore` as `_repo_path(src, client.root)`, which is `os.path.abspath(src)` for a
    path outside the root; `_prepare` takes `os.path.abspath(src)` first. For the sweep test's
    resolved temporary baseline that is `str(baseline)`.
  - `_repo_path(path, root)` (now two parameters), `stores._fingerprint_problems(value)` and
    `Run._fingerprint_results(self)` exist at `v0.1.0`: the private names the prompt names.
  - The CLI: `python -m runregistrykit [--client-root DIR] list [--root DIR]`, and `store show |
    create --purpose | fingerprint [--write] [--listing] | copy | move | retire | amend`. **Every
    `store` command loads the fingerprinter**, `store show` on a tombstone included: before the
    config exists it exits 1 naming the missing `[stores]`.
  - `datastorekit.store_fingerprint`: `ENCODING_FORMAT = 1`; `fingerprint_of(inventory,
    fingerprint_format, taken=None)`; `compare_fingerprints` returns one `encoding` entry, and
    nothing else, for a recorded fingerprint without `encoding_format` and equal format.
- **A dry run of §7.1, read-only**, through a scratch config outside the repository naming SGK's
  `var/runs` and `var/datastores` and a scratch fingerprinter of `config.datastore.factories` at
  format 6: `list` gives the 37 runs; `store show` exits 0 on all 18 sidecars, with output equal to
  `RunRegistry`'s **except** in "runs naming this store", where every run matched "by results"
  was lost and only `results_store_id` matches remained. That is the scratch config's doing: its
  record root was the scratch directory, so the manifests' relative `results` paths resolved there.
  With `runregistrykit.toml` at SGK's root they resolve to SGK. **So "runs naming this store" is the
  block to compare in §7.1**: a lost "by results" match there is a difference in substance (§10),
  not formatting. `find var -newer <marker>` was empty after.
- **RunRegistryKit's checkout** is at `caad864`, clean, with a venv; its
  `docs/adoption/measure_client_registry.py` takes `sgk --commit <commit>`.
- **The roots**, at `29a3f2d`, serially, load 12–20, all pass at 05's counts:

  | Root | Ran | Result | Wall |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | OK | 0 s |
  | `CosmologyModels/tests` | 39 | OK | 3 s |
  | `ObjectFactories/tests` | 271 | OK | 15 s |
  | `tools/tests` | 91 | OK | 12 s |
  | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 25 s |
  | `RunRegistry/tests` | 286 | OK | 73 s |
  | `ComputeTargets/tests` | 693 | OK | 146 s |

- **A pre-existing read of `var/runs/` in a test** stays as it is:
  `test_pipeline_adoption.test_a_refused_datastore_leaves_no_run_behind` lists the real runs root
  before and after a refused driver run (`:278-299`). The prompt keeps it, reading the root through
  `find_client_root`; it is a listing, and nothing is written.
- **Not touched, and staying:** `requirements.txt`, the venv, `README.md`, `prompts/INDEX.md`,
  `orchestrator/`, every `.md` and `.txt` under `docs/`, every store and sidecar, every run
  directory but the two this prompt adds; RunRegistryKit, DatastoreKit and RayPoolKit.

**What the review exists to establish.**
- **(S1) Nothing names `RunRegistry`** but the two lines ruling 1 allows; the measure reports 0
  imports at the probe; `import RunRegistry` fails.
- **(S2) The moved tests are the old tests**, changed only where §3.2 names, with every test
  accounted for: 101 moved, 3 + 1 new, and the counts 277 and 99.
- **(S3) The client files.** The config is the checklist's item 3; `config/fingerprint.py` is the
  history verbatim, the dated paragraph, the format and an instance; importing it loads nothing
  `config.datastore` does not.
- **(S4) The seams** forward the runs root, are tested, and change nothing at their defaults.
- **(S5) Nothing stored changes.** `09c`'s digest `6576b85d…` stands; the new golden is `09c` plus
  `encoding_format`; the format is 6.
- **(S6) The registry reads the same** (§7.1), and the worktree run is in the main checkout with the
  worktree's provenance (§7.4).
- **(S7) The breakages** fail as stated, and the script and the check exit as stated.
- **(S8) The records**: the log, the board, three issues closed, the index at 105, one commit whose
  difference from the probe is the records.

**Conventions,** in addition to `prompt-01.md`'s:
- **Staging.** By explicit path, with `git mv` and `git rm`; show `git diff --cached --name-status`
  and `--stat` before the probe and before the commit.
- **`var/`.** Read freely; written only by §7.4's run and §8's breakage-8 run. Scratch stores,
  sidecars and the fake run directory live in the agent's own scratch directory, never `/tmp`. No
  `store retire`, `copy`, `move` or `amend` on anything under `var/`.
- **Worktrees.** Under the agent's scratch directory, at the probe, removed and pruned after;
  nothing installed in them; the main checkout's venv by absolute path.
- **The venv.** No `pip install` or `pip uninstall`.
- **Ray and the lock.** None, and not taken. A rehearsal script that would start Ray is not run in
  §7.8: say so.
- **The SHA.** "this commit" where 06's hash belongs; the probe's hash and tree are written out.

## 1. Before you dispatch

Done at `29a3f2d` on 2026-10-15; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; clean; one worktree.
2. **The registry.** Nothing `running` or stale; 37 runs; 18 tombstones.
3. **The lock and Ray.** `tools.rehearsal_lock status`: free; `ray-processes`: none.
4. **The package and the checklist's lines.** As §0.
5. **Baselines.** The seven roots, as §0's table.
6. **The `var/` marker**, for check 9: immediately before dispatch, `touch` a marker in the
   orchestrator's scratch directory, and save `ls var/runs` beside it.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it the prompt, the README, the board, `HEAD`, the
baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a copy in the
session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §12 list.
- **The order of ruling 9**, with §7.1's "before" first.
- **The rulings of §0**, each logged as classified there.
- **Two runs under `var/runs/`, and nothing else written under `var/`.**
- **The wall-clock flakes** are expected under load, and are re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §10 conditions. A stop after `RunRegistry/` is deleted
  but before the commit leaves the working tree as it is, uncommitted, and says so; nothing is
  reverted without the user.

## 3. The review — twelve checks

1. **Scope.** `git show --name-status <06>`: the files of §0's "What moves on purpose", `R` for
   the eleven moves with similarity reported, 14 `D` under `RunRegistry/`, and nothing else. Not
   `README.md`, `prompts/INDEX.md`, `orchestrator/`, `requirements.txt`, any `.md` or `.txt` under
   `docs/`. `git diff --stat <probe> <06>`: only under `prompts/runregistry-extraction/` and
   `docs/OPEN_ISSUES.md`. `git cat-file -t <probe>` is `commit`, its parent the dispatch `HEAD`.
2. **S1, the grep.** §7.6's command at `<06>`: `CLAUDE.md`'s sentence and the history's line in
   `config/fingerprint.py`, nothing else. `ls RunRegistry` fails; `import RunRegistry` from the root
   is `ModuleNotFoundError`. `measure/out/06-measure.txt` is whole, against the probe, and says 0;
   re-run it here at `<06>` from RunRegistryKit's root: the same.
3. **S2, the moves, by diff.** For each of the five whole modules, `git diff -M <06>~1 <06> --` the
   pair: every hunk is one §3.2 names (the imports and `RunRegistry` → `runregistrykit`; `:317`;
   `:146`, `:158`, `:278`, `:330` and the new assertion; `:27`, `:73-75`, the five calls, `:134`,
   the docstrings, the new test). For `test_store_fingerprint.py`, check by script that
   `TestGolden`'s and `TestDrivers`' bodies, `_FullStores`, `without_taken` and `_finish_calls`
   equal the frozen tree's but for the named lines. The six golden files are renamed at 100%.
4. **S2, the counts.** By discovery at `<06>`: `RunDrivers` 99 (28, 26, 15, 18, 11, 1);
   `ObjectFactories` 277; the new test names as the prompt gives them.
5. **S3, the client files, by reading.** `runregistrykit.toml` is item 3's five lines and at most a
   comment. `config/fingerprint.py`: diff its history comment against `a45aefc:RunRegistry/stores.py`
   `:194-205`, which must be identical; one dated paragraph saying what C17 says; `FINGERPRINT_FORMAT
   = 6`; `fingerprinter` an instance. `python -X importtime -c 'import config.fingerprint'` adds no
   `datastorekit` module that `import config.datastore` does not, and neither imports `ray`.
6. **S4, the seams.** `git diff <06>~1 <06> --` the two drivers: the keyword, its forwarding, §2.1's
   rewrites and §2.3's literals, nothing else; `main()`'s call unchanged. Apply breakages 5 and 6
   here to a scratchpad copy, or the working tree and revert: each fails its test.
7. **S5, the golden.** The new golden file, loaded, minus `encoding_format`, equals `09c` exactly;
   its digest `6576b85d…`; its bytes equal `09c`'s but for the added line (`diff`). The roots pass.
8. **S6, the registry.** `06-before.txt` and `06-after.txt` compared here as data: the 37 runs, ids
   and states equal; 01's 29 as `01-list.txt`; every `store show` exit 0, and "runs naming this
   store" equal block for block. A fresh `runregistrykit list` now: 39, the two new ones `done` and
   `failed` (ruling 5), nothing `running`.
9. **S6, the worktree run.** Read both new runs' `manifest.json` and `status.json` yourself: under
   the main checkout's `var/runs/`; `git_head` the probe, `git_dirty` `false` for the check and
   `true` for the breakage; `script` and `cwd` the worktree's; `packages` naming `runregistrykit`
   0.1.0 at `9f8ebe43…`, `datastorekit` and `raypoolkit` at their pins, none editable.
   `git worktree list` shows this checkout alone. `find var -newer <marker>` lists those two run
   directories and nothing else; `ls var/runs` against the saved listing differs by those two.
10. **S7, the breakages and the script.** Apply each of the log's eight diffs, as written, to a
    scratchpad copy where possible and the working tree otherwise (reverting after): each fails
    the tests or exits as the log says. Run `fingerprint_both_ways.py` here on a fresh scratch set
    built as §5 says: exit 0. Read it against §5's checks list.
11. **The roots,** re-run here at `<06>`, serially: §7.5's counts, `RunRegistry/tests` gone,
    `RunDrivers/tests` present; a known flake re-run alone.
12. **S8, the records.** `CLAUDE.md`'s three sections and the removal, read in full, with ruling 8's
    pin; nothing else in `CLAUDE.md` moved. The log's §5.1 headings, every deviation classified, the
    rulings logged, §7.8's commands, the eight breakages as diffs, the private names in the
    observations. The board: header, §1 row, A1–A5, the three issues in §4 with their evidence,
    §5.12. `docs/OPEN_ISSUES.md`: three rows gone, 105, the date, §1.21's prose true. The commit
    message in `CLAUDE.md`'s form, ending `Co-Authored-By: Claude Opus …`.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 06* paragraph in §5.12, with
  residue fixed in a follow-up commit: "this commit" → the SHA; this note listed as "used for 06";
  `prompts/INDEX.md`'s row (7 of 10 landed); README's "06 ships" if anything shipped differs.
- **Report to the user:** the swap in one paragraph; the grep and the measure; the counts; the
  registry before and after; the worktree run, with its id; the three issues closed; anything
  opened.
- **Then 07**, the guard. Its note is written after this review.
