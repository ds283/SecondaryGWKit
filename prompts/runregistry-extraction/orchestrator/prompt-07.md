# Orchestrator — prompt 07, guard the fingerprint format

Read [`../README.md`](../README.md) first: §2.2 (07's charter), §5 (rules 4, 7–10) and §6.1 (Q5).
Then the board's §5.12 (06 and its review). The conventions are [`prompt-01.md`](prompt-01.md)'s §0
last list, carried over. [`prompt-06.md`](prompt-06.md) is this note's model for rulings on a
prompt's breakages; what differs is said here.

**You do not write code.** You may:
- run the suites and `python -m runregistrykit list`;
- read RunRegistryKit's repository with `git`, never changing it;
- apply the log's breakage diffs to the working tree and revert them, or run a breakage's scratch
  runner, and run a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`07-guard-the-fingerprint-format.md`](../07-guard-the-fingerprint-format.md)
**Board items:** V1 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing expected ·
**Model:** Sonnet, as the prompt recommends.

**Gate:**
- 06 landed (`740bafc`) and was reviewed (`7af17bd`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code. The one commit
  after the review, `ba15458`, touches `docs/OPEN_ISSUES.md` and `prompts/` only.
- Nothing is `running` in the registry; the rehearsal lock is free; `ray-processes` lists nothing;
  `git worktree list` shows this checkout alone.
- One prompt at a time in this checkout (README §4.2).

## 0. What makes this prompt unusual

**It is the smallest code prompt of the campaign, and almost all of its acceptance is its
breakages.** One test module, one pinned table and its comment. The test is easy to write so that it
passes; what the review must establish is that it fails for the right reason, naming the right
class, and that the key conversion it depends on is itself pinned.

**Two of the prompt's predictions are wrong**, measured by the orchestrator on SGK's registry at
`ba15458` with the recipe's sketch run in-process (no file changed). Neither is a stop: the guard
does what Q5 asks, and the prompt misdescribed what that looks like. Rulings 1 and 2 restate them
in the form the agent records.

**Rulings on the prompt** (the orchestrator's, made in writing this note; the user may reverse any).
Pass them on; the agent logs each as the classification given.
1. **Breakage 4 names `BackgroundModel` and `wavenumber_exit_time`, not `wavenumber`.** The
   `int`-keyed `Parent.types` is in those two specs (`ObjectFactories/cosmology_types.py`'s
   docstring says so; the sketch finds the keys `0` and `1` under `wavenumber_exit_time.parents` and
   `BackgroundModel.parents`, and in no other class). §1's sentence "the `int`-keyed `Parent.types`
   of `BackgroundModel` and `wavenumber`" and §3.4's expectation are the prompt's slip; §1's list of
   21 classes is right. With `COSMOLOGY_TYPES`' key `0` written as `"0"`, exactly those two digests
   move. `STRUCTURALLY REQUIRED`, citing this ruling.
2. **Breakage 5's second run: test 1 fails, it does not pass.** Eleven of the 21 specs have
   string-keyed mappings (`parents`, `parent_sets`, `members`): `wavenumber_exit_time`,
   `BackgroundModel`, `TkNumericIntegration`, `TkWKBIntegration`, `GkNumericIntegration`,
   `GkWKBIntegration`, `GkSourcePolicyData`, `GkSource`, `QuadSource`, `QuadSourceIntegral`,
   `OneLoopIntegral`. `str(k)` writes `k` where `json.dumps(k)` writes `"k"`, so with `plain`
   weakened all eleven digests move against a pin taken with `json.dumps`, and test 1 fails in both
   runs. The prompt's claim, that a guard built on `str(k)` cannot see a key's type change, is
   shown instead by: **with `str(k)`, `spec_digests(factories)` with breakage 4 applied equals it
   without**, and test 1's message names the same eleven classes in both runs. The agent records
   the two runs, the failing tests and the message's class lines of each, and the equality, from a
   scratch script that prints both lists. (Test 1 failing here is a property worth stating: while
   the pin stands, weakening `plain` is caught by test 1 as well as test 2.) `STRUCTURALLY
   REQUIRED`, citing this ruling.
3. **Breakage 3 patches the guard module's own name.** The test imports `factories` from
   `config.datastore` (§2.2), which binds it in the test module at import; patching
   `config.datastore.factories` does not reach it. `factories` is a `dict` of 38 names to factory
   classes, and `inventory_specs` needs only an `inventory_spec` attribute on the value. So the least
   intrusive way is a scratch runner, in the agent's scratch directory, that imports the test
   module, patches its `factories` with `mock.patch.object` to `{**factories, "<StandIn>":
   <stand-in>}`, and runs `TestFormatGuard`: the working tree is not touched. The log records the
   runner whole, in place of a diff. `IMPLEMENTATION CHOICE`.
4. **Breakages that fail more than the prompt names are not a stop; fewer is.** Under each
   breakage, run the guard module **and** the `ObjectFactories` root, and list every failing or
   erroring test, not only the named ones. A leaf added to `QuadSourceIntegral`'s spec may well
   make `TestGolden` and `TestConfigFingerprint` fail or error too; the format bumped to 7 fails at
   least `TestGolden.test_the_golden_fingerprint` and `TestConfigFingerprint` test 1. A named test
   that does *not* fail, or a failure that does not name the class the prompt (as ruled) names, is
   §5's stop.
5. **The docstring of `config/fingerprint.py`** lists the module's names (`FINGERPRINT_FORMAT`,
   `fingerprinter`). The agent may add one bullet for `SPEC_DIGESTS` there, as part of "the pin and
   its comment" (acceptance 4). Nothing else in the module moves: not the history, not the dated
   paragraph, not the two assignments. `IMPLEMENTATION CHOICE`, if taken.
6. **Breakage 2 fails with a message, not a `KeyError`.** The recipe's one-line form raises
   `KeyError`; §2.2 asks test 1 to check membership first with a message saying the format was
   bumped without a pin. The log quotes that message.

**What moves on purpose:**
- **Added:** `ObjectFactories/tests/test_fingerprint_format_guard.py`; `measure/out/07-roots.txt`;
  `logs/07-guard-the-fingerprint-format.md`.
- **Modified:** `config/fingerprint.py` (the pin, its comment, and at most ruling 5's bullet); the
  board (header, §1 row 07, V1, `### 5.13 After prompt 07`).
- **Not modified:** every factory and spec, `ObjectFactories/cosmology_types.py`, every test 06
  wrote, every golden file, `README.md`, `prompts/INDEX.md`, `orchestrator/`, `CLAUDE.md`,
  `requirements.txt`, `docs/`. `docs/OPEN_ISSUES.md` only with a new §3 entry.
- **Issues:** none expected; the index stays at **106** (`ba15458` opened
  `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]` after 06's review).

**The facts, checked by the orchestrator on 2026-10-15 at `ba15458`.** Pass them on.
- **The tree.** Clean; one worktree. `git diff --stat 740bafc ba15458 -- . ':!prompts'` is
  `CLAUDE.md`, `docs/OPEN_ISSUES.md` and two `docs/` scripts, all from the review's commit or
  `ba15458`; nothing under `config/`, `ObjectFactories/` or `StoreFixtures/`.
- **The registry.** 39 runs, 37 finished and 2 `unknown`, none `running` or stale. Lock free; no Ray
  process of this repository.
- **`config/fingerprint.py`** as 06 left it: a module docstring, the import of
  `DatastoreKitFingerprinter` and `factories`, the twelve-line history, the dated paragraph,
  `FINGERPRINT_FORMAT = 6`, `fingerprinter = DatastoreKitFingerprinter(factories,
  FINGERPRINT_FORMAT)`. It is `black`-clean.
- **The recipe** is RunRegistryKit's `docs/adoption/README.md` §6 at `aaab408`, which is also what
  `caad864` (the checkout's `HEAD`, clean) holds. Its sketch runs as written on SGK's registry.
- **The registry, through the sketch.** `config.datastore.factories` is a `dict` of 38;
  `inventory_specs` gives **21**, in the order of the prompt's §1. Leaf types `str`, `bool` and
  `None` only; every mapping key a `str`, but the polymorphic `types`' `int` keys `0` and `1` in
  `wavenumber_exit_time` and `BackgroundModel`. No set, enum or float: nothing the sketch cannot
  make plain. Two pairs equal, as §1 says.
- **The pin, as the orchestrator's run gives it** (first twelve hex digits, for the agent's
  cross-check only; the values are generated by the test's own function, never typed):

  | Class | Digest | Class | Digest |
  |---|---|---|---|
  | `version` | `02929e9353a2` | `TkWKBIntegration` | `54fd4b367c34` |
  | `store_tag` | `02929e9353a2` | `GkNumericIntegration` | `24b1b18adab8` |
  | `redshift` | `3e69e5a91230` | `GkWKBIntegration` | `39f965c25bb8` |
  | `wavenumber` | `a0ec49ae295f` | `GkSourcePolicy` | `419062a9ba9d` |
  | `wavenumber_exit_time` | `731ec560422f` | `GkSourcePolicyData` | `8bf4b980bf27` |
  | `tolerance` | `38ff5e8bf9db` | `GkSource` | `1bee7981b7f4` |
  | `LambdaCDM` | `af013d6dd11c` | `QuadSourcePolicy` | `419062a9ba9d` |
  | `QCD_Cosmology` | `f9b29777a4c6` | `QuadSource` | `8bc871a10849` |
  | `IntegrationSolver` | `6b7d69e652c0` | `QuadSourceIntegral` | `324451ac2570` |
  | `BackgroundModel` | `9baee59f0fc4` | `OneLoopIntegral` | `d94934bae9fc` |
  | `TkNumericIntegration` | `71826ae42648` | | |

  A generated value that disagrees with its prefix is a stop.
- **Breakages 4 and 5, simulated in-process** (the dict mutated in memory, no file changed): with
  `json.dumps`, breakage 4 moves `wavenumber_exit_time` and `BackgroundModel` and nothing else; with
  `str(k)`, the list differs from the pin at the eleven classes of ruling 2, and is the same with
  breakage 4 as without. `plain({0: "a"}) != plain({"0": "a"})` with `json.dumps`, and equal with
  `str`.
- **`QuadSourceIntegral.inventory_spec()`** (`ObjectFactories/QuadSourceIntegral.py:965`) declares
  sixteen parents and `tags`, and no `leaves`, `values` or `validated`. Breakage 1 adds a field
  there (for instance `leaves=("<a column>",)`); the log says which.
- **The roots** at `ba15458`, serially (§1 step 5's table).

**What the review exists to establish.**
- **(G1) The pin** is generated, is 21 `[class, digest]` pairs in registry order, equals the
  orchestrator's own run, and sits beside the format with the rule stated.
- **(G2) The test** is the recipe, opens nothing and needs no Ray, and fails with a message that
  names what changed and says what to do.
- **(G3) The breakages** fail as stated, as ruled; the key conversion is pinned.
- **(G4) Nothing stored changes**: no spec, factory, registry entry, golden file, or format number.
- **(G5) The records.**

**Conventions,** in addition to `prompt-01.md`'s:
- **Staging.** By explicit path only; show `git diff --cached --name-only` before the commit.
- **Breakages.** Applied to the working tree and reverted with `git checkout -- <file>`, or run from
  a scratch runner (ruling 3). After the last, `git status --short` shows only the prompt's files.
  Never committed.
- **`var/`.** Read by `list` only; nothing written. No `begin()`, no `store` command.
- **The venv.** No `pip install` or `pip uninstall`.
- **Ray and the lock.** None, and not taken.
- **The SHA.** "this commit" where 07's hash belongs; the orchestrator substitutes it.

## 1. Before you dispatch

Done at `ba15458` on 2026-10-15; the results are §0's facts.
1. **The tree.** Branch `handover-remedial`; clean; one worktree.
2. **The registry.** Nothing `running` or stale; 39 runs.
3. **The lock and Ray.** `tools.rehearsal_lock status`: free; `ray-processes`: none.
4. **The sketch.** Run on the registry from the session scratchpad; the pin saved there whole, for
   check 2.
5. **Baselines.** The seven roots, serially, at `ba15458`:

   | Root | Ran | Result | Wall |
   |---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 1 s |
   | `CosmologyModels/tests` | 39 | OK | 2 s |
   | `ObjectFactories/tests` | 277 | OK | 14 s |
   | `tools/tests` | 91 | OK | 11 s |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 25 s |
   | `RunDrivers/tests` | 99 | OK | 3 s |
   | `ComputeTargets/tests` | 693 | OK | 139 s |

   Load fell from 10 to 4 over the run; no known flake fired.

6. **The `var/` marker**, for check 8: immediately before dispatch, `touch` a marker in the
   orchestrator's scratch directory.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README, the board, `HEAD`, the
baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a copy in the
session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the end**, staged by explicit path from the prompt's §7 list.
- **The rulings of §0**, each logged as classified there; rulings 1 and 2 replace the prompt's
  expectations for breakages 4 and 5.
- **Five breakages, each reverted**, every failing test of each recorded (ruling 4).
- **Nothing written under `var/`.**
- **The wall-clock flakes** are expected under load, and are re-run alone and recorded, not a stop.
- **Stop and ask** on any of the prompt's §5 conditions as ruled here.

## 3. The review — nine checks

1. **Scope.** `git show --name-status <07>`: `A` for the test module, `07-roots.txt` and the log;
   `M` for `config/fingerprint.py` and the board; `docs/OPEN_ISSUES.md` only with a new §3 entry.
   Nothing else: not a factory, `cosmology_types.py`, a golden file, a test 06 wrote, `README.md`,
   `prompts/INDEX.md`, `orchestrator/`, `CLAUDE.md`.
2. **G1, the pin.** `git diff <07>~1 <07> -- config/fingerprint.py` is added lines only (the pin,
   its comment, at most ruling 5's bullet). `SPEC_DIGESTS` has the one key `6`, and its list equals,
   pair for pair and in order, the orchestrator's saved run. The comment states the rule (bump, new
   pin beside the old, never over it, new golden), how the digest is computed, and names the test.
3. **G2, the test, by reading.** `plain` and `spec_digests` are the recipe's, with `json.dumps(k)`;
   test 1 checks membership with a message before comparing, and its difference message names
   changed, added and removed classes separately and says what to do; test 2 is the two
   inequalities. Imports: `config.datastore`, `config.fingerprint`, `datastorekit.store_inventory`
   and the standard library; no `ray`, no store, nothing under `var/`. `black --check` on both files.
4. **G3, breakages 1, 2 and 4,** re-applied here to the working tree from the log's diffs as
   written, and reverted: each fails the tests the log lists, and test 1's message names
   `QuadSourceIntegral`; "no pin for format 7"; `BackgroundModel` and `wavenumber_exit_time` (ruling
   1).
5. **G3, breakage 3,** by running the log's runner from the scratchpad: test 1 fails naming the
   added class as added. The runner touches no tracked file.
6. **G3, breakage 5,** both runs re-applied here: test 2 fails in both; test 1's class lines equal
   between the runs (the eleven of ruling 2); and the `str(k)` lists with and without breakage 4
   equal. `git status --short` clean after.
7. **The roots,** re-run here at `<07>`, serially: §1's counts with `ObjectFactories` **279**; a
   known flake re-run alone. `07-roots.txt` agrees.
8. **G4, nothing moved.** `git diff <07>~1 <07> --stat -- ObjectFactories ':!ObjectFactories/tests'
   StoreFixtures config/datastore.py config/sharding.py` is empty; `FINGERPRINT_FORMAT` is 6.
   `find var -newer <marker>` is empty; `runregistrykit list` shows 39, nothing `running`.
9. **G5, the records.** The log's §5.1 headings; every deviation classified, the rulings logged;
   the five breakages as diffs (3 as its runner), each with its failing tests and message line. The
   board: header, §1 row, V1, §5.13. The commit message in `CLAUDE.md`'s form, ending
   `Co-Authored-By: Claude Sonnet …`.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 07* paragraph in §5.13, with
  residue fixed in a follow-up commit: "this commit" → the SHA; this note listed as "used for 07";
  `prompts/INDEX.md`'s row (8 of 10 landed).
- **Report to the user:** the guard and its pin; the breakages, with rulings 1 and 2 said plainly;
  the roots; anything opened.
- **Then 08**, R2. Its note is written after this review; it takes the rehearsal lock.
