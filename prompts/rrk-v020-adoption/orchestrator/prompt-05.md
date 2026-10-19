# Orchestrator — prompt 05, print the drivers' refusals cleanly

Read [`../README.md`](../README.md) first: §2.2 (05's charter), §5 (rules 2, 4, 7, 8, 10, 11), §6.1
(**U5**) and §6.2 (**C3**, **C5**). Then the board's §1, §1.1 (P1–P3) and §5.5. The conventions are
[`prompt-01.md`](prompt-01.md) §0's, carried over; [`prompt-02.md`](prompt-02.md) is the model for a
prompt that changes code and adds a test module. What differs is said here.

**You do not write code.** You may:
- run the suites, `python -m runregistrykit list`, `tools.rehearsal_lock status` and
  `ray-processes`;
- read RunRegistryKit with `git`, never changing, fetching or committing there;
- apply a breakage in the scratchpad's copy to confirm the agent's record, never in the tree;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`05-print-refusals-cleanly.md`](../05-print-refusals-cleanly.md)
**Board items:** P1–P3 · **Closes:** nothing · **Narrows:** nothing · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 04 landed (`34bf4f7`) and was reviewed (`454c7d8`); R3's stores are retired (`b87a4df`); the
  prompt is written (`13d993c`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches nothing outside `prompts/`.
- Nothing is `running` or stale in the registry; the lock is free; no Ray process of this
  repository is up. 05 starts no Ray, but its subprocess tests import `ray`, and the suites must
  not share the machine with a build.

## 0. What makes this prompt unusual

**It reopens a closed campaign for a code change that runs before any work starts.** The change is
small and its risk is the reverse of R3's: not that a build path moves, but that a failure is
turned into a `Refusal` and loses the traceback that is its evidence, or that a refusal stops being
a `RuntimeError` and a caller that catches one stops catching it. The test module exists to pin
both, and the review checks the pin, not only the diff.

**The prompt's citations hold at `13d993c`.** Every line number in its §1 and §2.2–§2.5 was checked
against the two drivers and against RunRegistryKit's `begin` at `v0.2.0`: `run_build` `:512`,
`import ray` `:531`, its refusals `:537` and `:547`, `begin` `:556`; the model filter `:627`;
`assert_store_is_self_consistent` `:669`, its raise `:736`; `prepare` `:739`, `:755`, `:779`;
`sweep` `:862`, `begin` `:869`, the child exit `:948`; the case selector `:361`;
`substitute_main_py` `:399`; `run_child`'s model filter `:486`; `main` `:970`; `__main__` `:1054`.
In `scoped_pipeline_run.py`: `register` `:517`, `:531`, `begin` `:544`; `main` `:589`, `:657`,
`:662`; the model filter `:713`; the literal count `:729`; `__main__` `:767`. At the tag, `begin`
is `__init__.py:987`, its refusals `:1102` and `:1119`, its `FileExistsError` `:1131`.

**What moves on purpose:**
- the two drivers: the class, the sites of prompt §2.2–§2.3 (with ruling 1's handler), the
  `__main__` block, and nothing else;
- `RunDrivers/tests/test_clean_refusals.py`, new;
- `docs/rrk-v020-adoption-verification.md`: one appended §4;
- `measure/out/05-roots.txt`, new; the log `logs/05-print-refusals-cleanly.md`, new; the board;
- **the launcher's hash.** Both drivers' SHA-256 change. At `13d993c` they are
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`
  (`docs/handover/quadsource_atol_sweep.py`, R3's) and
  `244ad70e7e0901c612bb01eb4f0fec10ee9bbd8e3bb3a17904024e111b6646df`
  (`docs/gktk-remedial/scoped_pipeline_run.py`). Verification §4 gives both, before and after.

**What changes in a log that nobody asked about, and is expected.** `sweep` runs each tolerance
pair as `quadsource_atol_sweep.py --child`, whose `run_child` calls
`assert_store_is_self_consistent` first. After 05 a child that meets that refusal prints one `!!`
line and exits 1, as before, where it printed a traceback. The parent's "child … exited 1"
`RuntimeError` is unchanged and keeps its traceback. The agent records this under "What shipped";
it is not a deviation.

**The two `Refusal` classes are distinct.** `quadsource_atol_sweep.py` loads
`scoped_pipeline_run.py` by path as `_scoped_pipeline_run`, so the launcher's `Refusal` and the
loaded module's are different classes. Nothing in the launcher calls `scoped.register` or any
function of the scoped driver that raises, so its `__main__` never needs to catch the other's. The
agent confirms this from the code and says so in the log; it is a stop if it is not so.

**Rulings on the prompt** (the orchestrator's, made in writing this note; the user may reverse any).
Pass them on; the agent logs each as the classification given.

1. **`ConfigError` is a `RuntimeError`, and keeps its traceback.** At `v0.2.0`,
   `runregistrykit.client.ConfigError` subclasses `RuntimeError` (`client.py:84`), and is exported
   as `runregistrykit.ConfigError`. `begin` raises it before anything is created, from
   `find_client_root` and through `read_sidecar` (`stores.py:691`, whose docstring says a
   `ConfigError` "comes through"). A bare `except RuntimeError` round `begin`, as prompt §2.3 is
   written, would therefore turn a `ConfigError` into a `Refusal`, which §2.3's own next sentence
   forbids. This is prompt §4's fourth stop condition, **settled here, not a stop**: each of the
   three `begin` calls is wrapped as

   ```python
   try:
       run = runregistrykit.begin(...)
   except runregistrykit.ConfigError:
       raise
   except RuntimeError as e:
       raise Refusal(str(e)) from e
   ```

   with the `ConfigError` clause first. Test 5 gains a second case: with `runregistrykit.begin`
   patched to raise `runregistrykit.ConfigError("…")`, `run_build` raises `ConfigError`, not
   `Refusal`. Breakage 7 removes that clause from `run_build` and must fail it. Test 2 pins the
   handler order at each of the three calls (ruling 3). `STRUCTURALLY REQUIRED`.

   The orchestrator also read `begin`'s other paths at the tag: `_running_runs_naming`'s own
   `RuntimeError` (`stores.py:1014`, the runs root is not a directory) is unreachable from `begin`,
   which calls it only `if os.path.isdir(runs_root)`; the `FileExistsError` and the `OSError`s of
   `os.makedirs` are not `RuntimeError`s. **The agent repeats this reading** from
   `git -C /Users/ds283/Documents/Code/RunRegistryKit show v0.2.0:runregistrykit/…` and records
   it. Any other `RuntimeError` it finds reachable from `begin` is a stop.

2. **Test 3 runs from the repository root and checks only stderr's shape, not its whole text.** The
   stderr of the launcher's subprocess may carry Ray's or NumPy's import-time warnings before the
   `!!` line. "stderr beginning `!! `" is read as **the first line of stderr that is not such a
   warning begins `!! `**, only if a warning is in fact seen. If none is seen, the test asserts the
   strict form, stderr's first line begins `!! `, and the log says which was needed.
   `IMPLEMENTATION CHOICE` if relaxed, none if not.

3. **Test 2 counts by the innermost enclosing `def`**, named by its dotted path
   (`run_build._filtered_build`, `run_child._filtered_build`, `make_case_selector.select`), so
   that the nested model filters are not confused with their enclosing functions. The expected
   counts at the end of 05:
   - `quadsource_atol_sweep.py`, `raise Refusal(`: `run_build` 3 (two pre-checks, `begin`),
     `assert_store_is_self_consistent` 1, `prepare` 2, `sweep` 1 — **7**;
     `raise RuntimeError(`: `make_case_selector.select` 1, `substitute_main_py` 1,
     `run_child._filtered_build` 1, `run_build._filtered_build` 1, `sweep` 1 — **5**;
   - `scoped_pipeline_run.py`, `raise Refusal(`: `main` 2, `register` 2 (the missing package,
     `begin`) — **4**; `raise RuntimeError(`: `main._filtered_build_model_list` 1, `main` 1 (the
     literal count) — **2**;
   - at each of the three `begin` calls, the enclosing `try` has exactly two handlers, in order
     `runregistrykit.ConfigError` with a bare `raise`, and `RuntimeError` raising `Refusal` with a
     `from` (ruling 1).

   Any other count the agent finds in the code before editing is a stop, not a re-count.
   `IMPLEMENTATION CHOICE`.

4. **Test 4 at `register` reuses `test_hard_killed_predecessor` by subclassing its base class**,
   `HardKilledPredecessorTestCase`, which holds `setUp`, `a_child_run` and `register` and no test
   method. It imports that class alone — never `TestTheRefusal` or another class with tests, which
   discovery would then run twice. Its `register()` already catches `RuntimeError` and restores
   what was taken over; the new test asserts the outcome is `self.driver.Refusal`, its message
   `begin`'s (`No run directory was created` at the end), its `__cause__` a `RuntimeError` that is
   not a `Refusal`, and that nothing was taken over. The child run is **alive** (`a_child_run`, not
   `a_hard_killed_run`), as the prompt says, which reaches `begin`'s `:1119` refusal through a
   liveness the existing test does not use. `IMPLEMENTATION CHOICE`.

5. **Test 4 at `run_build` and `sweep` loads the launcher fresh**, by importing
   `test_quadsource_atol_sweep_prepare`'s `_load_script` (or `test_build_rehearsal_scope`'s loader,
   whichever is module-level), and builds `args` either as an `argparse.Namespace` holding every
   field `main()`'s parser defines, or by calling `main()` with `sys.argv` patched and `run_build`
   / `sweep` stubbed to capture the parsed `args`, as `test_build_rehearsal_scope` does. **No parser
   function is added to the driver**: that is a seam (prompt §4's last stop). `run_build`'s
   `import ray` runs in the test process; it starts nothing, and the test asserts
   `ray.is_initialized()` is false afterwards. `IMPLEMENTATION CHOICE`.

6. **Breakage 2 is expected to fail more than test 1.** With `Refusal(Exception)`,
   `test_quadsource_atol_sweep_prepare`'s `assertRaises(RuntimeError)` tests fail, and so does
   `test_hard_killed_predecessor.TestTheRefusal.test_register_meets_the_refusal_and_takes_nothing_over`
   (its `register()` catches only `RuntimeError`, so the `Refusal` escapes as an error). The log
   names every test that fails, by id, with the run's last lines. That is what the breakage proves:
   that the subclassing matters. Not a deviation.

7. **The roots.** `RunDrivers` is 103 at `13d993c`, plus the new module's tests. The other six at
   §1's counts. A failure among README §8's four wall-clock tests is re-run alone, and both runs
   are recorded with the load average (04's ruling 9). **The machine's load was high at dispatch
   (§1)**; the agent records the load beside every root.

**Conventions** (prompt-01 §0's, carried over; restated because they bind):
- **Staging and reporting.** Stage by explicit path from prompt §6's list only; show
  `git diff --cached --name-only` before committing. Count, then paste. The log states only what
  the diff and the commands show.
- **The SHA.** "this commit" wherever 05's hash belongs; the orchestrator substitutes it.
- **Generated, not typed.** `05-roots.txt` is commands' output: each root's command, the `HEAD`,
  the load and the grep, as `04-roots.txt`.
- **Suites** run in the foreground, serially, one root per Bash call, full output to the agent's
  scratch directory.
- **Scratch.** The agent's own subdirectory of the session scratchpad; never `/tmp`, never `var/`.
- **`var/` is not touched.** The subprocess tests use temporary directories; `register`'s test a
  temporary runs root. No `list` but the read-only one, no `begin` outside a test's temporary
  root, no `end`, no store command. The lock is not taken.
- **Ray.** None started. `ray.init` is never called, in a test or by hand.
- **Breakages** are applied in the tree, run, recorded as the exact diff, and reverted with
  `git checkout -- <file>` or by hand; `git diff` after each revert shows only 05's own change.
- **Stop conditions are real.** A stop goes to the user through the orchestrator, and the agent
  resumes with `SendMessage`.
- **Other sessions** may land commits on this branch. The agent checks `git log` before editing and
  before committing.

## 1. Before you dispatch

Done at `13d993c` on 2026-10-19, before this note was committed.
1. **The tree.** Branch `handover-remedial`; `git status --short` empty; `git worktree list` shows
   this checkout alone.
2. **The registry.** 51 runs, none `running` or stale.
3. **The lock and Ray.** `tools.rehearsal_lock status`: free; `ray-processes`: none.
4. **RunRegistryKit.** Its `HEAD` is now `33a2cd2` ("Record that SGK has adopted v0.2.0: the
   interrupted-runs board's G1 holds"), the user's G1 commit, on `f6d4b80`; the tag `v0.2.0` is
   unchanged. 05 reads the tag only.
5. **Baselines.** The seven roots, serially, at `13d993c`, each in the foreground of one shell.

   | Root | Ran | Result | Wall | Load (1 m) |
   |---|---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 18 s | 427 |
   | `CosmologyModels/tests` | 39 | OK | 87 s | 373 |
   | `ObjectFactories/tests` | 279 | OK | 727 s | 395 |
   | `tools/tests` | 91 | OK | 112 s | 244 |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 91 s | 132 |
   | `RunDrivers/tests` | 103 | OK | 24 s | 121 |
   | `ComputeTargets/tests` | 700 | FAILED (failures=1) | 585 s | 74 |

   ComputeTargets' one failure is README §8's first known flake,
   `test_tk_wkb_phase.TestCost.test_wall_time_per_object`. Re-run alone twice at load 66–69 it
   failed both times, the second `0.0921 not less than or equal to 0.06`: a wall-clock assertion with
   about 1× margin, on a machine under load. Every count is 02's. Ruling 7 applies to it.

   The load is I/O, not CPU: during the baselines a backup (ChronoSync), Box's sync, `diskutil`
   and Munki's software update were running, and no process of this repository. The CPU was mostly
   idle. ObjectFactories took 727 s against 04's 26 s.
6. **The index:** 100.
7. **The `var/` marker**: immediately before dispatch, `touch` a marker in the orchestrator's
   scratch directory.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README, the board, `CLAUDE.md`,
`HEAD`, the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a
copy in the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after the breakages and the suites, staged by explicit path.
- **The order.** Read and count first (ruling 3's counts, ruling 1's reading of `begin`), before
  any edit; then the two drivers; then the test module; then the breakages; then the suites; then
  verification §4, the log and the board.
- **The rulings of §0**, each logged with the classification given, citing this note.
- **Every existing test passes unmodified**; one that does not is a stop.
- **Stop and ask** on any of the prompt's §4 conditions, and on §0's two confirmations (the
  `Refusal` classes; `begin`'s `RuntimeError`s).

## 3. The review — eight checks

1. **Scope.** `git show --name-status <05>`: `A` for the test module, the log and `05-roots.txt`;
   `M` for the two drivers, the verification document and the board; nothing else (the index only
   if §3 gained an entry). Verification: one hunk appending §4, no removed line.
2. **The diffs.** In each driver, `git diff 13d993c <05>` shows the class, the `raise` sites of
   ruling 3 and their handlers, and the `__main__` block, and nothing else. Every message is
   byte-identical: extract every string constant of every `raise` from both trees with `ast`, and
   compare.
3. **The counts** of ruling 3, re-counted here with a script of the orchestrator's own, not the
   agent's test.
4. **The clean print, by hand.** The two commands of prompt §2.6 test 3, run here: exit 1, one `!!`
   line, no `Traceback`, no `bootstrapping`. And `quadsource_atol_sweep.py --prepare --force`, a
   refusal reached without a store, which must print the same way. The `ConfigError` path is not
   provoked by hand (it would mean breaking the real `runregistrykit.toml`); it rests on test 5.
5. **The test module, read whole.** No `var/` path, no `ray.init`, no runs root but a temporary
   one; children reaped; the base class imported alone (ruling 4); no seam in either driver.
6. **The breakages, re-applied here.** At least 3, 5 and 7, in the scratchpad's copy of the tree,
   each failing as the log says.
7. **Nothing else moved.** `find var -newer <marker>` is empty; RunRegistryKit's checkout is
   unchanged at `33a2cd2`; `git status --short` clean after the agent's commit.
8. **The suites.** All seven roots, serially, at the commit, every count as §1 plus the new module.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 05* paragraph in §5.6, in 04's
  form, and fix small residue in a follow-up commit of its own: "this commit" → the SHA on the
  board, in the log's header, in README §2's row and in verification §4; this note listed as "used
  for 05"; the board's header back to **closed**, 5 of 5, if the agent left it reopened;
  `prompts/INDEX.md`'s row.
- Anything the review finds that is not residue is opened on the board's §3 and in the index.
- **Report to the user:** the refusal sites and their count; the output form quoted from check 4;
  ruling 1 and why; the launcher's new hash, which A3 v2's manifests will carry; and that the
  campaign is closed again.
