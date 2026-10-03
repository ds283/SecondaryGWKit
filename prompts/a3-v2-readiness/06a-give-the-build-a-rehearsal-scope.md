# Prompt 06a — give `--build` a rehearsal scope

**Campaign:** [`README.md`](README.md) · **Board items:** **B1**–**B3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-03, after 06 landed and was reviewed, under the user's decision **D5**
(README §6.1). **Gate:** 06 landed and reviewed; `git status` is clean.
**Closes:** nothing. **Opens:** nothing expected.
**Recommended model:** **Sonnet**. One flag that selects one fixed scope, existing test modules
that must keep passing, one existing test that may gain one keyword, and a one-constant change to
a verification tool.

**Read first:**

1. [`README.md`](README.md): §0, §1, §2's rows for 06a and 07, §4 (the 06a entry), §5 and §6.1 (D5).
2. `docs/handover/quadsource_atol_sweep.py`: `K_MIN`/`K_MAX`/`K_COUNT`, `k_sample`,
   `substitute_main_py`, `_main_py_argv`, `build_main_argv`, `child_main_argv`, `run_build` and
   `main`.
3. `RunRegistry/tests/test_prune_on_resume.py`, in full. It pins `build_main_argv`'s list element by
   element, the exact form of `run_build`'s `sys.argv` assignment, and that neither driver writes an
   option string of its own.
4. `RunRegistry/tests/test_killed_handler.py` (`TestTheDrivers`),
   `RunRegistry/tests/test_store_fingerprint.py` (`TestDrivers`),
   `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py`, and
   `docs/a3-v2-readiness/sigterm_handler_probe.py`, which each read `run_build` or load the module.
5. `docs/datastore-integrity-verification.md` §0 and §2.1.2. They describe the scope this flag
   selects: the same grid, `--zend` and sampling, built by `scoped_pipeline_run.py` on 2026-10-01.
6. `docs/datastore-integrity-verification/interrupt_when.py` and `interrupter_dummy_test.sh`.

---

## 1. What is wanted

Prompt 07 rehearses A3 v2's launch: a registered `--build`, interrupted by `SIGTERM` and resumed
with `--build --resume`. It must go through **`run_build`**, because that is the code 04 and 05
changed, and A3 v2 will run it. But `run_build` builds the production work list only. The grid is
`k_sample()` (8 log-spaced k over $10^5$–$3\times10^8$/Mpc), and `--zend 0.1` is a literal in
`_main_py_argv`. That is A3 v2 itself, and it takes days.

So `--build` gains one flag, **`--rehearsal`**. It selects one fixed, smaller scope, and changes
nothing else. Without the flag, every list, string and value is what it is today.

07 also interrupts the build, with `datastore-integrity`'s `interrupt_when.py`. That tool cannot
interrupt a `--build` today (B3). Both changes land here, so that 07 launches with no tracked
file modified, and every manifest of its runs names a commit holding everything that ran (D5).

**The rehearsal scope** is `datastore-integrity` prompt 10's, D10-1:
- $k$: `np.logspace(5, 6, 3)`, which is `[1e5, 316227.7660168379, 1e6]`, identical as floats to
  `np.logspace(np.log10(1e5), np.log10(1e6), 3)`. It replaces `SWEEP_K_SAMPLE` for both of
  `main.py`'s grids, as `k_sample()` does now;
- `--zend 250` in `main.py`'s argv, in place of `0.1`;
- everything else as without the flag: `--shards 4`, `--source-samples-log10z 100`, the LambdaCDM
  filter, the four substitutions, the tolerance pair from `--atol`/`--rtol`, registration, the
  handler and `finish_after_exec`.

That scope built a store in 15 minutes on 2026-10-01, at `rtol = 1e-8`, under a load of 22. Its
`QuadSourceIntegral` stage stays above every triangle's measured minimum $z_{\rm response}$, so its
cost is bounded by measurement (D10-1).

---

## 2. What to change

**B1 — the flag and the scope**, in `docs/handover/quadsource_atol_sweep.py`:
- **`--rehearsal`**, `action="store_true"`. Its help says that it is a rehearsal scope, that it is
  not A3 v2, and that it is valid only with `--build`. `main()` refuses it with `parser.error`
  unless `--build` is given: with `--child`, with `--prepare`, with `--report`, and with no mode
  (the sweep). The refusal comes **before** any mode runs: `--prepare` copies a store, and the
  sweep launches a child build per tolerance pair. A test of a refusal must therefore be unable to
  reach either; stub every mode on the loaded copy of the module, as `parsed_arguments` stubs
  `run_build`.
- **One module-level function decides the scope from the parsed arguments.** It returns:
  - the k sample;
  - the `--zend` string;
  - the text for the manifest's `scope`.

  It imports nothing and needs no Ray, so a test can call it. Its name, and the rehearsal
  constants' names, are the *prompt's choice*.
- **`run_build`** takes the k sample for `K_GRID_NAME` in its `exec` namespace from that function,
  and the manifest's `scope=` from it. Its `sys.argv` line stays exactly
  `sys.argv = build_main_argv(args, database)`, because `test_prune_on_resume` pins its `ast` form.
  It prints one line naming the rehearsal scope when the flag is given (*prompt's choice* of
  wording).
- **`build_main_argv`** takes `--zend` from that function. `_main_py_argv` may take `zend` as a
  parameter. `child_main_argv` must still produce `0.1`.
- **The manifest's `scope` text.**
  - Without the flag it is **character for character** what `run_build` writes today.
  - With the flag it says that this is a rehearsal build. It names the actual grid (3 log-spaced k
    over `1e5`–`1e6` /Mpc) and `zend 250`. It does **not** say `9280`, which is the production work
    list's size and not this one's.
- **`run_build`'s docstring** says what `--rehearsal` does.

**Unchanged, and the review compares them with `ast`:** `run_child`, `sweep`, `substitute_main_py`,
`k_sample`, `make_case_selector`, `prepare`, `report`, `assert_store_is_self_consistent`,
`shard_paths`, every constant that exists today, and the module docstring.

**B2 — tests**, in a new module `RunRegistry/tests/test_build_rehearsal_scope.py`. No Ray. Parse
with the script's own parser, as `test_prune_on_resume.parsed_arguments` does; import that helper
rather than copy it.
- **Without the flag:**
  - the k sample equals `k_sample()`;
  - the argv list is the production list (a literal, as in `test_prune_on_resume`);
  - the scope text equals today's string, written as a literal in the test.
- **With the flag:**
  - the k sample equals the literal `[1e5, 316227.7660168379, 1e6]`;
  - the argv list is the literal list with `"250"` after `--zend`. With `--resume` it is the same
    list;
  - the scope text names the grid and `zend 250` and does not contain `9280`.
- **The refusals:** `--rehearsal` with `--child`, `--prepare`, `--report` and no mode each exits 2.
- **`run_build` takes its grid from the scope**, read with `ast`:
  - the value its namespace gives `K_GRID_NAME` comes from the scope function's result;
  - `run_build` holds no call of `k_sample()` of its own.
- **`run_child` is still production**, read with `ast`: it still gives `K_GRID_NAME` the value
  `k_sample()`.

**The one existing test that may change** (README §5 rule 7):
`test_prune_on_resume.TestTheListsAreWhatTheyWere.test_the_job_name_is_the_namespaces` builds
`argparse.Namespace(job_name="some-other-label", resume=False)` by hand and passes it to
`build_main_argv`. If `build_main_argv` reads `args.rehearsal`, that `Namespace` may gain
`rehearsal=False`, and **nothing else in the file changes**. Not needing the change is also
acceptable. Either way, say which in the log.

**B3 — the interrupter names the run's own driver**, in
`docs/datastore-integrity-verification/interrupt_when.py`. Before it signals, it checks that the pid
is still the run's: `ps -p <pid> -o command=` must contain the constant
`DRIVER = "scoped_pipeline_run.py"` and `--register <slug>`. A `quadsource_atol_sweep.py` build can
never pass that, so the tool would never fire on one.
- **Change:** the driver's name is the basename of the run's own manifest `argv[0]`, the script the
  registry recorded as launched (`RunRegistry.begin` records `sys.argv` before `run_build` replaces
  it). Read it beside the slug. A manifest with no `argv[0]` makes the interrupter log that and exit
  without signalling, as a manifest with no slug does today.
- Update its docstring and the refusal's message. Nothing else in the file changes.
- `interrupter_dummy_test.sh` must pass **unchanged**. Its manifests write `argv[0]` as
  `docs/gktk-remedial/scoped_pipeline_run.py`, and its children run as
  `python docs/gktk-remedial/scoped_pipeline_run.py --register …`.
- **One more dummy case**, as a script of yours, `docs/a3-v2-readiness/interrupter_build_dummy_test.sh`,
  with its output committed beside it. It uses a manifest whose `argv[0]` is
  `docs/handover/quadsource_atol_sweep.py`, and a `sleep` child named for it with `exec -a`, as the
  existing test does. The interrupter must signal that child. Under the same manifest it must
  refuse a child named for `scoped_pipeline_run.py`.

---

## 3. Verification

1. **Every existing test passes**, `test_prune_on_resume`, `test_killed_handler`,
   `test_store_fingerprint` and `test_quadsource_atol_sweep_prepare` among them. Only the one
   `Namespace` above may change.
2. **`substitute_main_py(filter_cases=False)`** still replaces each of its four literals exactly
   once. Paste its four lines.
3. **The two scopes, from the parser.** For each of `--build …`, `--build --rehearsal …` and
   `--build --rehearsal --resume …` (with `--database`, `--atol 1e-32`, `--rtol 1e-7`), print the k
   sample, `build_main_argv`'s list and the scope text. Paste all three. Launch nothing: 07 runs
   it.
4. **The interrupter.** Run `interrupter_dummy_test.sh`, unchanged, into a new directory outside
   `var/` (the session scratchpad), then B3's new dummy case. Paste both outputs.
5. **Deliberate breakage.** Each must make a test fail. Restore it afterwards.
   - (i) The rehearsal's `zend` used without the flag. `test_prune_on_resume` must fail.
   - (ii) `run_build` gives `K_GRID_NAME` `k_sample()` again, with the argv still the scope's.
   - (iii) The rehearsal's scope text is the production string.
   - (iv) `main()` accepts `--rehearsal` with `--child`.
   - (v) `--rehearsal`'s default made `True`.
   - (vi) `interrupt_when.py` compares the command line with the constant `"scoped_pipeline_run.py"`
     again. B3's new dummy case must fail to signal, and the existing dummy test must still pass.

   Each goes in the log as a diff, taken with `git diff --` the one code file it changes, so that
   `git apply` replays it at this commit. Name the tests, or the dummy case, that fail.

---

## 4. Acceptance

1. B1 in `quadsource_atol_sweep.py`; without the flag nothing changes, by test and by `ast`.
2. B2's module, and the deliberate-breakage record (i)–(vi).
3. B3 in `interrupt_when.py`; the existing dummy test passes unchanged, and the new case signals and
   refuses as B3 says.
4. Verification 2, 3 and 4 pasted in the log.
5. `black --check` clean. On this board: the §1 row for 06a, items B1–B3, and §5. All in the same
   commit.

---

## 5. Stop conditions — stop and ask

- An existing test other than the one named in §2 would have to change.
- The production argv list, k sample or scope text would have to change.
- `substitute_main_py` no longer matches a literal exactly once.
- `interrupter_dummy_test.sh` cannot pass unchanged after B3.

---

## 6. What this prompt does not do

- It launches nothing, starts no Ray and writes nothing under `var/`. 07 runs the scope.
- It does not change `interrupter_dummy_test.sh`, or any other file of
  `docs/datastore-integrity-verification/` but `interrupt_when.py`.
- It does not touch `run_child` or `sweep`. Those carry
  `[04-the-sweeps-run-child-shares-the-sigterm-and-cluster-address-defects]`, which stays open.
- It makes `--resume` check nothing new. A store built with one scope and resumed with the other
  would mix them, and nothing records a store's scope except its runs' manifests. Record that in
  the log's observations; do not fix it.
- It adds no other knob: no `--k-min`, no `--zend`. One fixed scope means the reference and the
  resumed build of 07 cannot differ in it.

---

## 7. The log and the board

`logs/06a-give-the-build-a-rehearsal-scope.md`, in README §5.1's form. In addition, it records:
- verification 2, 3 and 4 verbatim;
- whether the named `Namespace` changed.

`IMPLEMENTATION_STATE.md`:
- the §1 row for 06a;
- items B1–B3;
- `### 5.14 After prompt 06a (<date>)`.

**The files you may change, and nothing else:**
- `docs/handover/quadsource_atol_sweep.py`, as §2 says;
- `RunRegistry/tests/test_build_rehearsal_scope.py`, new;
- the one `Namespace` in `RunRegistry/tests/test_prune_on_resume.py`, if needed;
- `docs/datastore-integrity-verification/interrupt_when.py`, as B3 says;
- `docs/a3-v2-readiness/interrupter_build_dummy_test.sh` and its output, new;
- the log and this board.

`docs/OPEN_ISSUES.md` changes only if you open an issue.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
