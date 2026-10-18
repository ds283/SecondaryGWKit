# Prompt 02 — adopt the hard-kill route

**Campaign:** [`README.md`](README.md) · **Board items:** **A1**, **A2**, **A3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-18, with prompts 01 and 03, at `79434bf`.
**Gate:**
- 01 landed and reviewed. The venv holds `runregistrykit` 0.2.0 at `0bae3a3c…`, not editable.
- `git status` is clean, and no commit after 01's review touches code.
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale.
- The roots pass at 01's counts.

**Closes:** nothing. **Opens:** only what the work finds.
**Recommended model:** **Sonnet**. Two paragraphs of `CLAUDE.md`, one option of a verification
script and its dummy-test case, and one new test module.

**Read first:**

1. `CLAUDE.md`, "Long-running jobs — the run registry", as 01 left it.
2. [`README.md`](README.md): §0 (what changes for SGK), §2.2 (02's charter), §5 (rules 7, 8, 10,
   15), and §6.2 **C3** and **C5** (why neither driver changes, and why the test's seam is
   `register`).
3. **RunRegistryKit at the tag**, read with `git -C /Users/ds283/Documents/Code/RunRegistryKit show
   v0.2.0:<path>`:
   - `README.md`, "The rules" and "Ending a run whose end was never recorded";
   - `docs/adoption/v0.2.0/secondarygwkit.md` §3–§4 and `docs/adoption/v0.2.0/README.md` §4 (the
     route after a hard kill, step by step, with the refusal text quoted);
   - `runregistrykit/tests/test_end.py`, `a_hard_killed_run` (`:123-154`) and its `CHILD`. Your
     child process is built the same way.
4. `docs/gktk-remedial/scoped_pipeline_run.py`, `register` (`:517-580`).
   `RunDrivers/tests/test_pipeline_adoption.py`, its `register` helper (`:311-345`), which saves and
   restores both output streams and the two signal handlers that `register` takes over.
   `RunDrivers/tests/registry_fixtures.py` (`REPO_ROOT`).
5. `docs/datastore-integrity-verification/interrupt_when.py` whole, and
   `docs/a3-v2-readiness/interrupter_build_dummy_test.sh` whole.

---

## 1. What is wanted

Under `v0.2.0` a build killed without its handler is resumed by **`end`, then the resume**: the
resume's `begin` is refused until a person has ended the dead run. SGK's code needs no change for
this; the adoption note's §3–§8 found nothing in it that breaks, and 01 measured the roots passing.
What SGK needs is three things, so that A3 v2 can rely on the route:
- **the rule**, written where every session reads it;
- **a test** that SGK's own driver meets the refusal, so that a later change to `register` cannot
  lose it unnoticed;
- **the means to rehearse it**: the interrupter able to send `SIGKILL`, for 03.

## 2. What to do

**2.1 `CLAUDE.md`, "Long-running jobs — the run registry".**
- **Rule 2, liveness.** Keep its first sentence's test (pid answers `kill -0`, heartbeat inside the
  window) and its warning against pattern matches. Add, in the package's terms (its README, "The
  rules"):
  - the test is made **on the host and boot the run recorded**;
  - a `running` run recorded on another host is `foreign` and is not tested here;
  - a `running` run from an earlier boot of this host is `stale` whatever its pid answers;
  - a run with no host recorded, which is every run written before `v0.2.0`, is judged as before.
- **A new rule, after rule 4 ("Do not babysit")**, numbered so that the rules stay in order: **the
  route after a hard kill.** A run killed by `SIGKILL`, a power loss, a reboot or the OOM killer
  never records its end, and its status says `running` for ever. Then:
  - `list` shows it `stale`, `!!`;
  - a resume's `begin` on its store is refused, and nothing is created;
  - a person checks that it is over and runs `./venv/bin/python -m runregistrykit end RUN_ID
    --reason TEXT` **on the host it ran on**. `end` refuses a pid that answers, with no override.
    `--unverified` is for a run on another host, or one whose status does not say, once checked
    there;
  - the run is then `lost`, and the resume is begun.

  **`Run.finish` is never called by hand on a run that is not the caller's**: it checks nothing.
  One sentence names the case `end` refuses that is not a live run: a killed process that its
  parent has not yet reaped still answers `kill -0` (RunRegistryKit's
  `[04-a-killed-process-not-yet-reaped-answers-kill-0]`). A detached launch through `nohup sh -c`
  is reaped by its `sh`.
- **The pointer** to RunRegistryKit's README gains its section "Ending a run whose end was never
  recorded".
- Nothing else in `CLAUDE.md` changes. Keep the section's voice: short rules, each with its reason.

**2.2 `interrupt_when.py --signal`.**
- **The option.** `--signal {TERM,KILL}`, default `TERM`. `watch` takes the signal and sends it
  where it now sends `signal.SIGTERM` (`:203`). The log line says which signal it sent ("sent
  SIGKILL to pid …"). Every check before the signal is unchanged: the stage, the percentage, the
  hold, `kill -0`, and the command line naming the run's driver and slug.
- **The docstring.** Its first line and usage name the option. One paragraph says that a `KILL`
  runs no handler, so the run's status stays `running` and its end must be recorded with `end`
  (`CLAUDE.md`, 2.1).
- **The dummy test.** `docs/a3-v2-readiness/interrupter_build_dummy_test.sh` gains a **case D**,
  after its last case, by case A's pattern:
  - a `--build` manifest, a child named for the sweep script, and the interrupter run with
    `--signal KILL`;
  - it must signal, its log must say `SIGKILL`, and the child must die of signal 9 (`wait`'s
    status 137);
  - cases A–C are not changed, and their expected outputs stay as they are.

  `docs/datastore-integrity-verification/interrupter_dummy_test.sh` is not changed. It covers a
  `scoped_pipeline_run.py` run, and its signal is the default.

**2.3 The driver test**, a new module, `RunDrivers/tests/test_hard_killed_predecessor.py`.

**The child.** It is built as RunRegistryKit's `a_hard_killed_run` builds one, with this
repository's own `runregistrykit`, not the package's test helpers:
- a `subprocess.Popen` of `sys.executable -c <CHILD>`, with `cwd` the repository root
  (`registry_fixtures.REPO_ROOT`);
- the child calls `runregistrykit.begin(campaign=…, prompt=…, slug="hard-killed",
  purpose=…, argv=[…], results=<database>, root=<the test's runs root>, client_root=<REPO_ROOT>)`.
  `<database>` is a path in the test's temporary directory that is never created, as
  `test_pipeline_adoption`'s registration test uses. `begin` matches a store by path when there is
  no sidecar to give a `results_store_id`. The orchestrator probed this on 2026-10-18 with
  RunRegistryKit's own venv at the tag's code: a second `begin` on a never-created path, with an
  alive first run, was refused, the message ending "names this store by its results";
- it beats once, prints its run's path, and sleeps;
- the test reads the path, sends `SIGKILL`, and reaps the child (`wait()` returns `-9`). Then it
  asserts that the run's `status.json` is byte-identical to what it was before the kill, with
  `state` `running` and a `host`;
- cleanup kills and reaps the child whatever happens. No child outlives the test.

**The tests**, each with a temporary runs root (`StoreTestCase` or `RegistryTestCase` from
`registry_fixtures`):
1. **The refusal.** `scoped_pipeline_run.register(args, main_args, k_sample, database,
   runs_root=<root>)`, called through a helper that saves and restores what it takes over, as
   `test_pipeline_adoption`'s does, raises `RuntimeError`. Then assert:
   - its message names the dead run's id, `stale`, and `python -m runregistrykit end`, and ends
     `No run directory was created`;
   - the runs root's directory listing is unchanged;
   - `sys.stdout`, `sys.stderr` and the two handlers are those from before the call. `begin` raises
     before `register` takes anything over, and the test pins that.
2. **`end`, then the second registration.** `./venv/bin/python -m runregistrykit end <id> --reason
   "<text>" --root <root>`, run as a subprocess with `cwd` the repository root, exits 0. Then
   assert:
   - the dead run's `status.json` has `state` `lost` and `exit_code` `null`;
   - its `ended` has `verified` `true`, `not_verified_because` `null`, and evidence `host` `this`
     and `pid_answers` `false`;
   - every other key the status held before is kept;
   - a second `register` on the same database is then accepted, and its run is made in the root.
3. **`end` is refused on a live run.** Begin a child as above, do **not** kill it, and run `end`
   on it. It exits non-zero, and the run's `status.json` is unchanged. Then kill and reap the
   child. This pins the guarantee that `CLAUDE.md`'s new rule leans on: `end` cannot end a live
   build.

**The constraints:**
- no Ray is imported or initialised;
- nothing under `var/` is read or written;
- each child is reaped before its test ends;
- the module runs in well under a second per test, apart from the child's start;
- it loads `scoped_pipeline_run` through `test_pipeline_adoption`'s `load_driver()`, imported
  from there, not copied.

## 3. Acceptance

1. **The roots pass**: `RunDrivers` at 01's count plus this module's tests, and every other root at
   01's counts. A failure among README §8's four known wall-clock tests is re-run alone and
   recorded.
2. **Both interrupter dummy tests pass.** The build test's cases A–C give the same outcome as in
   `runregistry-extraction` 08's `interrupter_build_dummy_test_08.txt`, and case D signals with
   `SIGKILL`, the child dying of signal 9. Run each into `measure/out/02-dummy-*.txt`. The script
   prints its outcomes and asserts nothing, so a case passes when its printed exit statuses and
   log lines match the recorded output.
3. **The deliberate breakages are caught** (README §5 rule 8). Each is applied, run, recorded as a
   diff in the log, and reverted:
   1. `register` passes `results=None` to `begin`: test 1 fails;
   2. `register` wraps `begin` in `try: … except RuntimeError: return None`: test 1 fails;
   3. `register` takes over `sys.stdout` before calling `begin`: test 1's restoration assertion
      fails;
   4. the interrupter sends `signal.SIGTERM` whatever `--signal` says: case D fails;
   5. the interrupter's `--signal` default becomes `KILL`: case A's child exits 137, not the
      recorded 143, and its log names `SIGKILL`.
4. **`CLAUDE.md`** says what §2.1 says, and nothing that RunRegistryKit's README at the tag
   contradicts. Quote each changed paragraph in the log.
5. `black --check` is clean on every changed `.py` file.

## 4. Stop conditions — stop and ask

- **`begin` does not refuse the second registration** with the dead run in the root, or `end`
  refuses the dead run, or `end` ends a live one. Each is RunRegistryKit's behaviour at the tag,
  against its own README. Record what happened, write no workaround, and stop (README §5 rule 11).
- A root fails other than by README §8's known tests, or a count other than `RunDrivers`' moves.
- The test cannot be written without changing `scoped_pipeline_run.py` or
  `quadsource_atol_sweep.py` (README §6.2 C3, C5).
- A case of either dummy test other than case D changes outcome.

## 5. What this prompt does not do

- It changes neither driver, `main.py` nor any reader (README §6.2 C3).
- It modifies no existing test (README §5 rule 7). It adds one module and a case to one dummy test.
- It runs no build, starts no Ray, and does not take the rehearsal lock.
- It writes nothing under `var/`, and runs `end` only on runs in a test's own temporary runs root.

## 6. The log and the board

**The files you may change, and nothing else:**
- `CLAUDE.md` (§2.1);
- `docs/datastore-integrity-verification/interrupt_when.py` and
  `docs/a3-v2-readiness/interrupter_build_dummy_test.sh` (§2.2);
- `RunDrivers/tests/test_hard_killed_predecessor.py` (new);
- under `measure/out/`, `02-roots.txt` and `02-dummy-*.txt`;
- the log, this campaign's board, and `docs/OPEN_ISSUES.md` only if §3 gains an entry.

**The log**, at `logs/02-adopt-the-hard-kill-route.md`, in README §5.1's form.

**The board:** the header, the §1 row for 02, items A1–A3, and
`### 5.2 After prompt 02 (<date>)`.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
